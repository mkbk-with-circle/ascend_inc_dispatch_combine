"""Allocation-free hot-path owner for device route metadata and buffers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.distributed as dist

from inc_moe_runtime import PlanKey, select_route_lanes, use_parallel_route


_DISPATCH_ROW_BYTES = 32
_ASSIGNMENT_BYTES = 32
_WAVE_BYTES = 64
_STATUS_BYTES = 64
_CACHE_LINE_WORDS = 16
_MAX_ROUTE_LANES = 64


@dataclass(frozen=True, slots=True)
class RouteRuntimeConfig:
    tokens_per_wave: int
    slot_count: int = 3
    activation_waves: int = 2
    # None selects the live device vector-core count during prepare.  Zero is
    # an explicit scalar fallback used for qualification and old hardware.
    route_lanes: int | None = None

    def validate(self) -> None:
        if self.tokens_per_wave <= 0:
            raise ValueError("tokens_per_wave must be positive")
        if self.slot_count < 3:
            raise ValueError("slot_count must be at least three")
        if self.activation_waves <= 0:
            raise ValueError("activation_waves must be positive")
        if self.route_lanes is not None and not 0 <= self.route_lanes <= \
                _MAX_ROUTE_LANES:
            raise ValueError("route_lanes must be in [0, 64]")


def _parallel_scratch_words(
    wave_capacity: int, expert_count: int, lane_count: int
) -> int:
    expert_stride = (
        (expert_count + _CACHE_LINE_WORDS - 1) // _CACHE_LINE_WORDS
    ) * _CACHE_LINE_WORDS
    return (
        wave_capacity * lane_count * expert_stride
        + wave_capacity * lane_count * _CACHE_LINE_WORDS
        + wave_capacity * expert_stride
        + lane_count * _CACHE_LINE_WORDS
    )


class PreparedTorchRoute:
    """One prepared capacity bucket; ``pack`` performs no tensor allocation.

    The one EP all-gather exchanges ``[wave_capacity, experts+1]`` int32.
    The extra column carries active-token count, avoiding a second collective
    or a device-to-host ``item()`` for uneven rank lengths.
    """

    def __init__(
        self,
        key: PlanKey,
        config: RouteRuntimeConfig,
        expert_owner: Sequence[int],
        expert_local_index: Sequence[int],
        ep_group: Any,
        device: torch.device | str,
    ) -> None:
        key.family.validate()
        config.validate()
        family = key.family
        if len(expert_owner) != family.expert_count or len(
            expert_local_index
        ) != family.expert_count:
            raise ValueError("expert placement length disagrees with family")
        if dist.get_world_size(ep_group) != family.worker_count:
            raise ValueError("EP group size disagrees with worker_count")
        if dist.get_rank(ep_group) != family.worker_rank:
            raise ValueError("EP group rank disagrees with worker_rank")
        if family.dtype.lower() not in ("bf16", "bfloat16", "torch.bfloat16"):
            raise ValueError("the current route/executor ABI requires BF16")
        if key.token_capacity <= 0:
            raise ValueError("token capacity must be positive")
        if any(owner < 0 or owner >= family.worker_count for owner in expert_owner):
            raise ValueError("expert owner is outside worker group")
        pairs = list(zip(expert_owner, expert_local_index, strict=True))
        if len(set(pairs)) != len(pairs):
            raise ValueError("expert placement contains duplicate owner/local pairs")
        for owner in range(family.worker_count):
            local = sorted(index for rank, index in pairs if rank == owner)
            if local != list(range(len(local))):
                raise ValueError("local expert indices must be contiguous per owner")

        self.key = key
        self.config = config
        self.ep_group = ep_group
        self.device = torch.device(device)
        if self.device.type != "npu":
            raise ValueError("prepared route requires an NPU device")
        if self.device.index is None:
            self.device = torch.device(f"npu:{torch.npu.current_device()}")
        if config.route_lanes is None:
            device_index = self.device.index
            if device_index is None:
                device_index = torch.npu.current_device()
            properties = torch.npu.get_device_properties(device_index)
            live_lanes = int(properties.vector_core_num)
        else:
            live_lanes = config.route_lanes
        if not 0 <= live_lanes <= _MAX_ROUTE_LANES:
            raise RuntimeError(
                f"device vector-core count {live_lanes} exceeds route ABI"
            )
        self.route_lanes = live_lanes
        self.wave_capacity = (
            key.token_capacity + config.tokens_per_wave - 1
        ) // config.tokens_per_wave
        self.local_expert_count = sum(
            owner == family.worker_rank for owner in expert_owner
        )
        if self.local_expert_count == 0:
            raise ValueError("every compute worker must own at least one expert")
        count_width = family.expert_count + 1
        self.local_counts = torch.empty(
            (self.wave_capacity, count_width),
            dtype=torch.int32,
            device=self.device,
        )
        self.global_counts = torch.empty(
            (family.worker_count, self.wave_capacity, count_width),
            dtype=torch.int32,
            device=self.device,
        )
        self.active_token_counts = torch.empty(
            family.worker_count, dtype=torch.int32, device=self.device
        )
        max_rows = key.token_capacity * min(family.topk, family.worker_count)
        max_assignments = key.token_capacity * family.topk
        self.dispatch_rows = torch.empty(
            max_rows * _DISPATCH_ROW_BYTES,
            dtype=torch.uint8,
            device=self.device,
        )
        self.assignments = torch.empty(
            max_assignments * _ASSIGNMENT_BYTES,
            dtype=torch.uint8,
            device=self.device,
        )
        self.group_lists = torch.empty(
            (self.wave_capacity, self.local_expert_count),
            dtype=torch.int64,
            device=self.device,
        )
        self.waves = torch.empty(
            self.wave_capacity * _WAVE_BYTES,
            dtype=torch.uint8,
            device=self.device,
        )
        scratch_words = 2 * family.expert_count if self.route_lanes == 0 else \
            _parallel_scratch_words(
                self.wave_capacity, family.expert_count, self.route_lanes
            )
        self.scratch = torch.empty(
            scratch_words,
            dtype=torch.int32,
            device=self.device,
        )
        self.status = torch.empty(
            _STATUS_BYTES, dtype=torch.uint8, device=self.device
        )
        self.expert_owner = torch.tensor(
            expert_owner, dtype=torch.int32, device=self.device
        )
        self.expert_local_index = torch.tensor(
            expert_local_index, dtype=torch.int32, device=self.device
        )
        # Only used when framework router tensors are not already in the native
        # ABI. These are capacity buffers, not per-forward allocations.
        self.id_staging = torch.empty(
            (key.token_capacity, family.topk),
            dtype=torch.int64,
            device=self.device,
        )
        self.weight_staging = torch.empty(
            (key.token_capacity, family.topk),
            dtype=torch.float32,
            device=self.device,
        )

    def _normalize(
        self, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        family = self.key.family
        if topk_ids.ndim != 2 or tuple(topk_weights.shape) != tuple(
            topk_ids.shape
        ):
            raise ValueError("topk ids/weights must share [tokens, topk]")
        tokens, topk = topk_ids.shape
        if tokens > self.key.token_capacity:
            raise RuntimeError(
                f"active tokens {tokens} exceed prepared capacity "
                f"{self.key.token_capacity}"
            )
        if topk != family.topk:
            raise ValueError("topk disagrees with prepared plan")
        if topk_ids.device != self.device or topk_weights.device != self.device:
            raise ValueError("router tensors are not on the prepared NPU")
        if topk_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("topk_ids must be int32 or int64")
        if topk_ids.is_contiguous():
            ids = topk_ids
        else:
            ids = self.id_staging[:tokens]
            ids.copy_(topk_ids)
        if topk_weights.dtype is torch.float32 and topk_weights.is_contiguous():
            weights = topk_weights
        else:
            weights = self.weight_staging[:tokens]
            weights.copy_(topk_weights)
        return ids, weights

    def pack(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        executor_handle: int,
        use_inc_route_relay: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids, weights = self._normalize(topk_ids, topk_weights)
        torch.ops.inc_fusion_native.route_count_out(
            ids, self.local_counts, self.status, self.config.tokens_per_wave
        )
        # Output first dimension is world_size times the input first dimension.
        # Both views are persistent and the metadata payload is the same for all
        # four attribution modes.
        # INC modes use the already-live star service for this metadata: every
        # worker uploads one disjoint histogram block to INC, then INC fans the
        # committed complete table back out.  The HCCL path remains available
        # as an explicit diagnostic fallback and for worker-direct SHMEM modes,
        # which intentionally have no INC sidecar.
        if use_inc_route_relay:
            torch.ops.inc_fusion_native.worker_route_exchange_out(
                executor_handle, self.local_counts, self.global_counts
            )
        else:
            dist.all_gather_into_tensor(
                self.global_counts.view(
                    self.key.family.worker_count * self.wave_capacity, -1
                ),
                self.local_counts,
                group=self.ep_group,
            )
        # ABI v3 introduced this contiguous source-length vector and ABI v6
        # keeps it in the single remote request record. This is an
        # allocation-free device copy from the metadata collective's extra
        # column; no host item() or second collective is introduced.
        self.active_token_counts.copy_(self.global_counts[:, 0, -1])
        route_lanes = select_route_lanes(
            ids.shape[0], self.key.family.expert_count, self.route_lanes
        )
        use_parallel = use_parallel_route(
            ids.shape[0], ids.shape[1], route_lanes
        )
        pack_op = torch.ops.inc_fusion_native.route_pack_parallel_out \
            if use_parallel else torch.ops.inc_fusion_native.route_pack_out
        pack_args = (
            ids,
            weights,
            self.global_counts,
            self.expert_owner,
            self.expert_local_index,
            self.dispatch_rows,
            self.assignments,
            self.group_lists,
            self.waves,
            self.scratch,
            self.status,
            self.key.family.worker_rank,
            self.key.family.hidden,
            2,
            self.config.tokens_per_wave,
            self.config.slot_count,
            self.config.activation_waves,
        )
        if use_parallel:
            pack_op(*pack_args, route_lanes)
        else:
            pack_op(*pack_args)
        return ids, weights


__all__ = ["PreparedTorchRoute", "RouteRuntimeConfig"]
