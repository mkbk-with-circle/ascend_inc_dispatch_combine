"""Framework-neutral configuration used by the vLLM-Ascend adapter.

This module intentionally imports neither torch nor vLLM.  Model workers can
construct and test the backend/capacity policy before loading the device
extension, and unit tests can run on a host-only development environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, TypeVar


class BackendMode(str, Enum):
    NATIVE_VLLM = "native_vllm"
    SERIAL_SHMEM = "serial_shmem"
    SERIAL_INC = "serial_inc"
    FUSED_SHMEM = "fused_shmem"
    FUSED_INC = "fused_inc"

    @property
    def is_factorial(self) -> bool:
        return self is not BackendMode.NATIVE_VLLM

    @property
    def uses_inc(self) -> bool:
        return self in (BackendMode.SERIAL_INC, BackendMode.FUSED_INC)

    @property
    def is_fused(self) -> bool:
        return self in (BackendMode.FUSED_SHMEM, BackendMode.FUSED_INC)


class WeightLayout(str, Enum):
    FUSION_ND = "fusion_nd"          # w13[E,2I,H], w2[E,H,I]
    TRANSPOSED_ND = "transposed_nd"  # w13[E,H,2I], w2[E,I,H]


@dataclass(frozen=True, slots=True)
class CapacityPolicy:
    """Map an arbitrary non-negative token count to a reusable plan capacity.

    There is no protocol-level maximum.  ``maximum`` is an optional deployment
    HBM budget and a request exceeding it fails explicitly instead of being
    truncated.  Power-of-two buckets avoid one plan per dynamic batch size.
    """

    minimum: int = 1
    maximum: int | None = None

    def capacity_for(self, active_tokens: int) -> int:
        if active_tokens < 0:
            raise ValueError("active_tokens must be non-negative")
        if self.minimum <= 0:
            raise ValueError("minimum capacity must be positive")
        required = max(active_tokens, self.minimum)
        capacity = 1 << (required - 1).bit_length()
        if self.maximum is not None and capacity > self.maximum:
            raise MemoryError(
                f"token capacity {capacity} exceeds configured HBM budget "
                f"{self.maximum}; active_tokens={active_tokens}"
            )
        return capacity


@dataclass(frozen=True, slots=True)
class PlanFamily:
    worker_count: int
    worker_rank: int
    hidden: int
    intermediate: int
    expert_count: int
    topk: int
    dtype: str
    expert_placement_digest: int
    hardware_profile_digest: int

    def validate(self) -> None:
        if self.worker_count <= 0:
            raise ValueError("worker_count must be positive")
        if not 0 <= self.worker_rank < self.worker_count:
            raise ValueError("worker_rank is outside the worker group")
        if min(self.hidden, self.intermediate, self.expert_count, self.topk) <= 0:
            raise ValueError("MoE dimensions must be positive")
        if self.topk > self.expert_count:
            raise ValueError("topk exceeds expert_count")
        if not self.dtype:
            raise ValueError("dtype is required")
        if self.expert_placement_digest == 0 or self.hardware_profile_digest == 0:
            raise ValueError("placement and hardware digests must be non-zero")


@dataclass(frozen=True, slots=True)
class PlanKey:
    family: PlanFamily
    token_capacity: int


@dataclass(frozen=True, slots=True)
class FusionPlanInfo:
    symmetric_bytes: int
    worker_workspace_bytes: int
    inc_workspace_bytes: int
    wave_count: int
    local_expert_count: int
    fusion_abi_version: int
    remote_service_bytes: int

    @classmethod
    def from_native(cls, values: list[int]) -> "FusionPlanInfo":
        if len(values) != 7 or any(value < 0 for value in values):
            raise RuntimeError(f"invalid native plan info: {values}")
        info = cls(*values)
        if info.symmetric_bytes <= 0 or info.wave_count <= 0 or \
                info.local_expert_count <= 0:
            raise RuntimeError(f"incomplete native plan info: {values}")
        return info

    def require_same_service_layout(self, other: "FusionPlanInfo") -> None:
        left = (
            self.symmetric_bytes,
            self.wave_count,
            self.fusion_abi_version,
            self.remote_service_bytes,
        )
        right = (
            other.symmetric_bytes,
            other.wave_count,
            other.fusion_abi_version,
            other.remote_service_bytes,
        )
        if left != right:
            raise RuntimeError(
                "worker plans disagree on persistent service layout: "
                f"{left} != {right}"
            )


@dataclass(frozen=True, slots=True)
class SymmetricHeapPolicy:
    """Deployment policy, deliberately separate from the device protocol."""

    minimum_bytes: int = 512 * 1024 * 1024
    reserve_bytes: int = 64 * 1024 * 1024
    alignment_bytes: int = 2 * 1024 * 1024

    def heap_bytes(self, info: FusionPlanInfo) -> int:
        if min(self.minimum_bytes, self.reserve_bytes) < 0 or \
                self.alignment_bytes <= 0:
            raise ValueError("invalid symmetric heap policy")
        required = max(
            self.minimum_bytes, info.symmetric_bytes + self.reserve_bytes
        )
        return (
            (required + self.alignment_bytes - 1) // self.alignment_bytes
        ) * self.alignment_bytes


@dataclass(frozen=True, slots=True)
class FusionPeMapping:
    worker_pes: tuple[int, ...]
    inc_pe: int
    worker_devices: tuple[int, ...]
    inc_device: int

    def validate(self, worker_count: int) -> None:
        if len(self.worker_pes) != worker_count or len(
            self.worker_devices
        ) != worker_count:
            raise ValueError("worker PE/device mapping length mismatch")
        all_pes = (*self.worker_pes, self.inc_pe)
        if sorted(all_pes) != list(range(worker_count + 1)):
            raise ValueError("Fusion PE mapping must be a permutation of W+1")
        all_devices = (*self.worker_devices, self.inc_device)
        if min(all_devices) < 0 or len(set(all_devices)) != worker_count + 1:
            raise ValueError("Fusion NPU mapping must contain distinct devices")


@dataclass(frozen=True, slots=True)
class WorkerExecutorConfig:
    """Setup-only hardware/protocol configuration for one persistent service.

    A remote INC service has one fixed symmetric-memory layout.  Consequently
    one engine worker owns one executor at the engine's maximum token capacity;
    arbitrary smaller (including empty) local batches reuse that executor via
    ``active_token_counts``.  A larger request must be rejected before forward
    and requires rebuilding the engine/service with a larger HBM budget.
    """

    live_aiv: int
    live_aic: int
    inc_pe: int
    tokens_per_wave: int
    slot_count: int = 3
    service_ring_size: int = 4
    activation_waves: int = 2
    spin_cap: int = 0
    # A long-lived engine can enqueue several forwards before the NPU reaches
    # the oldest worker kernel.  The maximum protocol ring is inexpensive
    # (64 argument records/events) and avoids a setup-specific small default.
    executor_ring_size: int = 64
    # Relay the tiny route histogram through the already-persistent INC star
    # instead of launching one HCCL all-gather per MoE layer.
    inc_route_relay: bool = False

    def validate(self, key: PlanKey) -> None:
        key.family.validate()
        if key.token_capacity <= 0:
            raise ValueError("token_capacity must be positive")
        if not 1 <= self.live_aiv <= 64:
            raise ValueError("live_aiv must be in [1, 64]")
        if self.live_aic <= 0:
            raise ValueError("live_aic must be positive")
        if self.inc_pe < 0:
            raise ValueError("inc_pe must be non-negative")
        if self.tokens_per_wave <= 0:
            raise ValueError("tokens_per_wave must be positive")
        if self.slot_count < 3:
            raise ValueError("slot_count must be at least three")
        if not 2 <= self.service_ring_size <= 64:
            raise ValueError("service_ring_size must be in [2, 64]")
        if self.activation_waves <= 0:
            raise ValueError("activation_waves must be positive")
        if self.spin_cap < 0:
            raise ValueError("spin_cap must be non-negative")
        if not 2 <= self.executor_ring_size <= 64:
            raise ValueError("executor_ring_size must be in [2, 64]")

    def accepts(self, key: PlanKey, active_tokens: int) -> bool:
        self.validate(key)
        if active_tokens < 0:
            raise ValueError("active_tokens must be non-negative")
        return active_tokens <= key.token_capacity


T = TypeVar("T")


# Scalar packing scans top-k once to form destination rows and again for every
# non-empty destination group, so avoidable regroup work is proportional to
# T*K*(K-1); top-k=1 has none.  Parallel packing pays two extra launches.
# Requiring two 64-record token batches overall and one 64-record work batch
# per live lane follows the implementation's UB buffer granularity and avoids
# a model-shape lookup table.  Keep this host-only so policy tests need no torch.
_PARALLEL_MIN_TOKENS = 2 * 64
_PARALLEL_WORK_PER_LANE = 64
_PARALLEL_MIN_TOKENS_PER_LANE = 16
_PARALLEL_HISTOGRAM_WORD_BUDGET = 2048


def select_route_lanes(tokens: int, expert_count: int, live_lanes: int) -> int:
    """Bound route lanes by useful token work and histogram footprint.

    The parallel protocol allocates one cache-line-aligned expert histogram
    per lane.  Launching every physical AIV for a small token wave or a large
    expert family increases prefix/scratch traffic without exposing more
    independent records.  These two implementation invariants provide a
    hardware-portable bound and avoid a model/shape tuning table.
    """
    if tokens <= 0 or expert_count <= 0 or live_lanes <= 0:
        return 0
    expert_stride = ((expert_count + 15) // 16) * 16
    token_lanes = max(1, tokens // _PARALLEL_MIN_TOKENS_PER_LANE)
    histogram_lanes = max(
        1, _PARALLEL_HISTOGRAM_WORD_BUDGET // expert_stride
    )
    return min(live_lanes, token_lanes, histogram_lanes)


def use_parallel_route(tokens: int, topk: int, lanes: int) -> bool:
    """Select analyze/prefix/emit from protocol work, never a shape table."""
    return (
        lanes > 1
        and tokens >= _PARALLEL_MIN_TOKENS
        and tokens * topk * (topk - 1) >=
        lanes * _PARALLEL_WORK_PER_LANE
    )


class PreparedPlanCache(Generic[T]):
    """Small explicit cache; plan creation never occurs in a timed hot call.

    ``prepare`` is a setup/warmup operation.  ``lookup`` only returns an
    existing plan and therefore cannot hide allocation/JIT inside a benchmark.
    """

    def __init__(
        self,
        factory: Callable[[PlanKey], T],
        policy: CapacityPolicy | None = None,
    ) -> None:
        self._factory = factory
        self._policy = policy or CapacityPolicy()
        self._plans: dict[PlanKey, T] = {}

    def key_for(self, family: PlanFamily, active_tokens: int) -> PlanKey:
        family.validate()
        return PlanKey(family, self._policy.capacity_for(active_tokens))

    def prepare(self, family: PlanFamily, active_tokens: int) -> T:
        key = self.key_for(family, active_tokens)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._factory(key)
            self._plans[key] = plan
        return plan

    def lookup(self, family: PlanFamily, active_tokens: int) -> T:
        key = self.key_for(family, active_tokens)
        try:
            return self._plans[key]
        except KeyError as exc:
            raise RuntimeError(
                f"plan {key} was not prepared; allocation/JIT is forbidden "
                "inside the measured forward path"
            ) from exc

    def __len__(self) -> int:
        return len(self._plans)


__all__ = [
    "BackendMode",
    "CapacityPolicy",
    "FusionPlanInfo",
    "FusionPeMapping",
    "PlanFamily",
    "PlanKey",
    "PreparedPlanCache",
    "SymmetricHeapPolicy",
    "WorkerExecutorConfig",
    "WeightLayout",
    "use_parallel_route",
]
