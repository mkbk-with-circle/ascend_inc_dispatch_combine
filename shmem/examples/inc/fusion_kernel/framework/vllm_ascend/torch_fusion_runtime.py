"""Explicit setup/teardown owner for the native single-INC worker executor.

The symmetric allocation and FFTS address come from the process-wide Fusion
SHMEM bootstrap.  This module never initializes SHMEM implicitly and never
creates an executor from the model forward path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch

from inc_moe_runtime import FusionPlanInfo, PlanKey, WorkerExecutorConfig
from torch_route_runtime import PreparedTorchRoute


def _native_op(name: str) -> Any:
    namespace = getattr(torch.ops, "inc_fusion_native", None)
    op = getattr(namespace, name, None) if namespace is not None else None
    if op is None:
        raise RuntimeError(
            f"inc_fusion_native::{name} is missing; rebuild/load the Torch "
            "bridge with FUSION_API_LIBRARY before engine setup"
        )
    return op


def query_plan_info(
    key: PlanKey,
    config: WorkerExecutorConfig,
    expert_owner: Sequence[int],
    expert_local_index: Sequence[int],
) -> FusionPlanInfo:
    """Compute exact allocation requirements before SHMEM initialization."""
    config.validate(key)
    family = key.family
    if len(expert_owner) != family.expert_count or len(
        expert_local_index
    ) != family.expert_count:
        raise ValueError("expert placement length disagrees with family")
    owner = torch.tensor(expert_owner, dtype=torch.int32, device="cpu")
    local = torch.tensor(expert_local_index, dtype=torch.int32, device="cpu")
    values = _native_op("plan_info")(
        owner,
        local,
        config.live_aiv,
        config.live_aic,
        family.worker_count,
        family.worker_rank,
        config.inc_pe,
        family.hidden,
        family.intermediate,
        family.expert_count,
        family.topk,
        key.token_capacity,
        config.tokens_per_wave,
        config.slot_count,
        config.service_ring_size,
        config.activation_waves,
        config.spin_cap,
    )
    return FusionPlanInfo.from_native(list(values))


def query_all_worker_plan_info(
    key: PlanKey,
    config: WorkerExecutorConfig,
    expert_owner: Sequence[int],
    expert_local_index: Sequence[int],
) -> tuple[FusionPlanInfo, ...]:
    """Query every rank and prove one common persistent-service layout."""
    infos = []
    for rank in range(key.family.worker_count):
        rank_key = PlanKey(
            replace(key.family, worker_rank=rank), key.token_capacity
        )
        infos.append(query_plan_info(
            rank_key, config, expert_owner, expert_local_index
        ))
    first = infos[0]
    for info in infos[1:]:
        first.require_same_service_layout(info)
    return tuple(infos)


class PreparedTorchExecutor:
    """One fixed-capacity worker executor bound to one remote INC service.

    Construction is an engine setup operation and may allocate/synchronize.
    ``output_for`` only creates a view over persistent output HBM.  ``close``
    is explicit because invoking Torch operators from a Python finalizer during
    interpreter shutdown is unsafe.
    """

    def __init__(
        self,
        route: PreparedTorchRoute,
        config: WorkerExecutorConfig,
        worker_pes: torch.Tensor,
        symmetric_base: int,
        symmetric_bytes: int,
        ffts_addr: int,
        backend_mode: int,
    ) -> None:
        config.validate(route.key)
        family = route.key.family
        if worker_pes.device != route.device:
            raise ValueError("worker_pes is not on the prepared NPU")
        if worker_pes.dtype is not torch.int32 or not worker_pes.is_contiguous():
            raise TypeError("worker_pes must be a contiguous int32 NPU tensor")
        if worker_pes.ndim != 1 or worker_pes.numel() != family.worker_count:
            raise ValueError("worker_pes must contain one PE per worker")
        if symmetric_base <= 0 or symmetric_bytes <= 0:
            raise ValueError("symmetric allocation address/size must be positive")
        if ffts_addr < 0:
            raise ValueError("ffts_addr must be non-negative")
        if backend_mode not in (1, 2, 3, 4):
            raise ValueError("backend_mode must be one of the four 2x2 modes")

        self.route = route
        self.config = config
        self.backend_mode = backend_mode
        self.worker_pes = worker_pes
        self.returns_global_output = backend_mode in (2, 4) and family.topk >= 2
        output_capacity = route.key.token_capacity * (
            family.worker_count if self.returns_global_output else 1
        )
        self.output = torch.empty(
            (output_capacity, family.hidden),
            dtype=torch.bfloat16,
            device=route.device,
        )
        self._closed = False
        self._handle = int(_native_op("worker_prepare")(
            route.expert_owner,
            route.expert_local_index,
            worker_pes,
            route.active_token_counts,
            int(symmetric_base),
            int(symmetric_bytes),
            int(ffts_addr),
            config.live_aiv,
            config.live_aic,
            family.worker_count,
            family.worker_rank,
            config.inc_pe,
            family.hidden,
            family.intermediate,
            family.expert_count,
            family.topk,
            route.key.token_capacity,
            config.tokens_per_wave,
            config.slot_count,
            config.service_ring_size,
            config.activation_waves,
            config.spin_cap,
            config.executor_ring_size,
            backend_mode,
        ))
        if self._handle <= 0:
            raise RuntimeError("native worker_prepare returned an invalid handle")

    @classmethod
    def from_session(
        cls,
        route: PreparedTorchRoute,
        config: WorkerExecutorConfig,
        worker_pes: torch.Tensor,
        session: "FusionShmemSession",
        backend_mode: int,
    ) -> "PreparedTorchExecutor":
        if route.device != session.device:
            raise ValueError("route and SHMEM session use different NPUs")
        return cls(
            route,
            config,
            worker_pes,
            session.symmetric_base,
            session.symmetric_bytes,
            session.ffts_addr,
            backend_mode,
        )

    @property
    def handle(self) -> int:
        if self._closed:
            raise RuntimeError("single-INC worker executor is already closed")
        return self._handle

    @property
    def uses_inc_route_relay(self) -> bool:
        return self.backend_mode in (2, 4) and self.config.inc_route_relay

    def accepts(self, active_tokens: int) -> bool:
        return not self._closed and self.config.accepts(
            self.route.key, active_tokens
        )

    def output_for(
        self, x: torch.Tensor, global_output_rows: int | None = None
    ) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("single-INC worker executor is already closed")
        family = self.route.key.family
        if x.device != self.route.device or x.dtype is not torch.bfloat16:
            raise TypeError("input must be BF16 on the prepared NPU")
        if x.ndim != 2 or x.shape[1] != family.hidden:
            raise ValueError("input must have shape [tokens, hidden]")
        if not self.config.accepts(self.route.key, x.shape[0]):
            raise RuntimeError(
                f"active tokens {x.shape[0]} exceed persistent service "
                f"capacity {self.route.key.token_capacity}; rebuild setup "
                "with a larger symmetric-memory/HBM budget"
            )
        if self.returns_global_output:
            if global_output_rows is None:
                # Uniform TP slices are the common fast path.  The explicit
                # row count lets the caller support torch.tensor_split's
                # uneven last slices without reading active_token_counts back
                # from the device.
                rows = x.shape[0] * family.worker_count
            else:
                rows = global_output_rows
            if rows < 0 or rows > self.output.shape[0]:
                raise RuntimeError(
                    f"global output rows {rows} exceed persistent capacity "
                    f"{self.output.shape[0]}"
                )
        else:
            if global_output_rows is not None:
                raise ValueError(
                    "global_output_rows is only valid for INC global fanout"
                )
            rows = x.shape[0]
        return self.output[:rows]

    def close(self) -> None:
        if self._closed:
            return
        # The anchor supplies the NPU dispatch key/device context.  Native
        # destroy waits only for this executor's in-flight event ring.
        _native_op("worker_destroy")(self.route.expert_owner, self._handle)
        self._closed = True

    def __enter__(self) -> "PreparedTorchExecutor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()


class FusionShmemSession:
    """Explicit owner of one ACLSHMEM PE and its symmetric fusion tensor.

    UID distribution belongs to the launcher (for example a multiprocessing
    pipe or rendezvous store), not to vLLM's worker ProcessGroup.  Barriers are
    ACLSHMEM-world barriers and therefore include the INC sidecar.
    """

    def __init__(
        self,
        uid: Any,
        pe: int,
        world_size: int,
        heap_bytes: int,
        symmetric_bytes: int,
        device: torch.device | str,
    ) -> None:
        if not 0 <= pe < world_size:
            raise ValueError("PE is outside the Fusion SHMEM world")
        if symmetric_bytes <= 0 or heap_bytes < symmetric_bytes:
            raise ValueError("SHMEM heap must cover the symmetric allocation")
        self.device = torch.device(device)
        if self.device.type != "npu":
            raise ValueError("Fusion SHMEM session requires an NPU device")
        # Import lazily so framework config/unit tests remain host-only.
        import shmem

        self._shmem = shmem
        # The ACLSHMEM UID config-store defaults to TLS, while the local
        # single-node launcher intentionally has no certificate bundle.  The
        # official UID examples disable config-store TLS on every PE before
        # init; omitting this makes rank 0 fail immediately and leaves peers
        # waiting in bootstrap.
        if shmem.set_conf_store_tls(False, "") != 0:
            raise RuntimeError("failed to disable ACLSHMEM config-store TLS")
        shmem.core.init(
            uid=uid,
            rank=pe,
            nranks=world_size,
            initializer_method="uid",
            mem_size=heap_bytes,
        )
        device_index = self.device.index
        if device_index is None:
            device_index = torch.npu.current_device()
            self.device = torch.device(f"npu:{device_index}")
        if torch.npu.current_device() != device_index:
            raise RuntimeError(
                "select the session NPU with torch.npu.set_device before "
                "initializing Fusion SHMEM"
            )
        self.symmetric = shmem.aclshmem_create_tensor(
            (symmetric_bytes,), torch.uint8, device_id=device_index
        )
        self.symmetric.zero_()
        self.ffts_addr = int(shmem.get_ffts_config())
        self.pe = pe
        self.world_size = world_size
        self._closed = False

    @property
    def symmetric_base(self) -> int:
        if self._closed:
            raise RuntimeError("Fusion SHMEM session is closed")
        return int(self.symmetric.data_ptr())

    @property
    def symmetric_bytes(self) -> int:
        return int(self.symmetric.numel())

    def barrier(self) -> None:
        if self._closed:
            raise RuntimeError("Fusion SHMEM session is closed")
        # Make setup writes visible before the host collective returns.
        torch.npu.current_stream(self.device).synchronize()
        self._shmem.aclshmem_barrier_all()

    def close(self) -> None:
        if self._closed:
            return
        self._shmem.aclshmem_free_tensor(self.symmetric)
        self._shmem.core.finalize()
        self._closed = True


class PreparedIncService:
    """INC-side plan/service owner for a model-free launcher sidecar."""

    def __init__(
        self,
        key: PlanKey,
        config: WorkerExecutorConfig,
        expert_owner: Sequence[int],
        expert_local_index: Sequence[int],
        worker_pes: Sequence[int],
        session: FusionShmemSession,
    ) -> None:
        config.validate(key)
        family = key.family
        if session.pe != config.inc_pe:
            raise ValueError("INC session PE disagrees with executor config")
        if session.world_size <= max([config.inc_pe, *worker_pes]):
            raise ValueError("PE mapping exceeds Fusion SHMEM world")
        if len(expert_owner) != family.expert_count or len(
            expert_local_index
        ) != family.expert_count:
            raise ValueError("expert placement length disagrees with family")
        if len(worker_pes) != family.worker_count:
            raise ValueError("worker_pes length disagrees with worker_count")
        self.session = session
        self.config = config
        self.key = key
        self.expert_owner = torch.tensor(
            expert_owner, dtype=torch.int32, device=session.device
        )
        self.expert_local_index = torch.tensor(
            expert_local_index, dtype=torch.int32, device=session.device
        )
        self.worker_pes = torch.tensor(
            worker_pes, dtype=torch.int32, device=session.device
        )
        self._handle = int(_native_op("service_prepare")(
            self.expert_owner,
            self.expert_local_index,
            self.worker_pes,
            session.symmetric_base,
            session.symmetric_bytes,
            session.ffts_addr,
            config.live_aiv,
            config.live_aic,
            family.worker_count,
            config.inc_pe,
            family.hidden,
            family.intermediate,
            family.expert_count,
            family.topk,
            key.token_capacity,
            config.tokens_per_wave,
            config.slot_count,
            config.service_ring_size,
            config.activation_waves,
            config.spin_cap,
            int(config.inc_route_relay),
        ))
        if self._handle <= 0:
            raise RuntimeError("native service_prepare returned invalid handle")
        self._started = False
        self._closed = False

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("single-INC service is closed")
        if self._started:
            return
        _native_op("service_start")(self.expert_owner, self._handle)
        self._started = True

    def close(self) -> None:
        if self._closed:
            return
        _native_op("service_destroy")(self.expert_owner, self._handle)
        self._closed = True



__all__ = [
    "FusionShmemSession",
    "PreparedIncService",
    "PreparedTorchExecutor",
    "query_plan_info",
    "query_all_worker_plan_info",
]
