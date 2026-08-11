"""vLLM worker lifecycle adapter for the prepared single-INC runtime.

This module is imported by vLLM through ``worker_extension_cls``.  It keeps
all data-plane state inside each engine worker and exposes only setup/teardown
control methods through ``collective_rpc``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from inc_moe_runtime import (
    BackendMode,
    PlanFamily,
    PlanKey,
    WeightLayout,
    WorkerExecutorConfig,
)
from single_inc_comm_method import SingleIncMoECommMethod
from torch_fusion_runtime import (
    FusionShmemSession,
    PreparedTorchExecutor,
    query_plan_info,
)
from torch_registration import register_inc_fusion_op
from torch_route_runtime import PreparedTorchRoute, RouteRuntimeConfig
from torch_weight_runtime import PreparedWeightCache


_BACKEND_MODE_ID = {
    BackendMode.SERIAL_SHMEM: 1,
    BackendMode.SERIAL_INC: 2,
    BackendMode.FUSED_SHMEM: 3,
    BackendMode.FUSED_INC: 4,
}


def _install_comm_method(runtime: "_WorkerRuntime") -> None:
    from vllm_ascend.ascend_forward_context import MoECommType
    import vllm_ascend.ops.fused_moe.moe_comm_method as comm_module

    if not comm_module._MoECommMethods:
        raise RuntimeError("vLLM did not register a MoE communication method")
    method = SingleIncMoECommMethod(
        next(iter(comm_module._MoECommMethods.values())).moe_config,
        runtime.mode,
        prepared_weight_resolver=runtime.resolve_weights,
        prepared_executor_resolver=runtime.resolve_executor,
    )
    for comm_type in MoECommType:
        comm_module._MoECommMethods[comm_type] = method


@dataclass(slots=True)
class _WorkerRuntime:
    session: FusionShmemSession
    executor: PreparedTorchExecutor
    mode: BackendMode
    family: PlanFamily
    weight_caches: dict[tuple[int, int], PreparedWeightCache] = field(
        default_factory=dict
    )

    def resolve_executor(self, active_tokens: int) -> PreparedTorchExecutor:
        if not self.executor.accepts(active_tokens):
            raise RuntimeError(
                f"vLLM batch has {active_tokens} local tokens but the "
                f"single-INC engine capacity is {self.executor.route.key.token_capacity}"
            )
        return self.executor

    def resolve_weights(
        self, w13: torch.Tensor, w2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(w13.data_ptr()), int(w2.data_ptr()))
        cache = self.weight_caches.get(key)
        if cache is None:
            local = self.executor.route.local_expert_count
            expected = (
                (local, 2 * self.family.intermediate, self.family.hidden),
                (local, self.family.hidden, self.family.intermediate),
            )
            transposed = (
                (local, self.family.hidden, 2 * self.family.intermediate),
                (local, self.family.intermediate, self.family.hidden),
            )
            shapes = (tuple(w13.shape), tuple(w2.shape))
            if shapes == expected:
                layout = WeightLayout.FUSION_ND
            elif shapes == transposed:
                layout = WeightLayout.TRANSPOSED_ND
            else:
                raise RuntimeError(
                    "unsupported vLLM MoE weight layout: "
                    f"w13={shapes[0]}, w2={shapes[1]}, expected "
                    f"{expected} or {transposed}"
                )
            cache = PreparedWeightCache(
                w13,
                w2,
                layout,
                local,
                self.family.hidden,
                self.family.intermediate,
            )
            self.weight_caches[key] = cache
        return cache.resolve(w13, w2)


class SingleIncWorkerExtension:
    """Methods dynamically mixed into vLLM-Ascend's NPUWorker."""

    def init_single_inc_runtime(
        self,
        uid: bytes,
        mode: str,
        worker_count: int,
        hidden: int,
        intermediate: int,
        expert_count: int,
        topk: int,
        token_capacity: int,
        tokens_per_wave: int,
        activation_waves: int,
        live_aiv: int,
        live_aic: int,
        bridge_library: str,
        heap_bytes: int,
    ) -> dict[str, int | str]:
        if hasattr(self, "_single_inc_runtime"):
            raise RuntimeError("single-INC runtime is already initialized")
        selected_mode = BackendMode(mode)
        if not selected_mode.uses_inc:
            raise ValueError("this lifecycle adapter only initializes INC modes")

        from vllm.distributed import get_ep_group
        ep = get_ep_group()
        if ep.world_size != worker_count:
            raise RuntimeError(
                f"vLLM EP size {ep.world_size} != requested workers {worker_count}"
            )
        rank = int(ep.rank_in_group)
        device = torch.device(f"npu:{torch.npu.current_device()}")
        torch.ops.load_library(bridge_library)
        register_inc_fusion_op()

        local_experts = expert_count // worker_count
        if local_experts * worker_count != expert_count:
            raise ValueError("first vLLM qualification requires even expert placement")
        owner = tuple(
            min(expert // local_experts, worker_count - 1)
            for expert in range(expert_count)
        )
        local = tuple(expert % local_experts for expert in range(expert_count))
        family = PlanFamily(
            worker_count=worker_count,
            worker_rank=rank,
            hidden=hidden,
            intermediate=intermediate,
            expert_count=expert_count,
            topk=topk,
            dtype="bf16",
            expert_placement_digest=0x51474550,
            hardware_profile_digest=0x910B2C,
        )
        key = PlanKey(family=family, token_capacity=token_capacity)
        config = WorkerExecutorConfig(
            live_aiv=live_aiv,
            live_aic=live_aic,
            inc_pe=worker_count,
            tokens_per_wave=tokens_per_wave,
            activation_waves=activation_waves,
        )
        info = query_plan_info(key, config, owner, local)
        if heap_bytes < info.symmetric_bytes:
            raise MemoryError(
                f"SHMEM heap {heap_bytes} is below required {info.symmetric_bytes}"
            )
        session = FusionShmemSession(
            uid=uid,
            pe=rank,
            world_size=worker_count + 1,
            heap_bytes=heap_bytes,
            symmetric_bytes=info.symmetric_bytes,
            device=device,
        )
        route = PreparedTorchRoute(
            key,
            RouteRuntimeConfig(
                tokens_per_wave=tokens_per_wave,
                activation_waves=activation_waves,
            ),
            owner,
            local,
            ep.device_group,
            device,
        )
        worker_pes = torch.arange(
            worker_count, dtype=torch.int32, device=device
        )
        executor = PreparedTorchExecutor.from_session(
            route, config, worker_pes, session, _BACKEND_MODE_ID[selected_mode]
        )
        runtime = _WorkerRuntime(session, executor, selected_mode, family)
        session.barrier()

        _install_comm_method(runtime)
        self._single_inc_runtime = runtime
        return {
            "rank": rank,
            "device": int(torch.npu.current_device()),
            "symmetric_bytes": info.symmetric_bytes,
            "worker_workspace_bytes": info.worker_workspace_bytes,
            "abi": info.fusion_abi_version,
            "mode": selected_mode.value,
        }

    def close_single_inc_runtime(self) -> dict[str, int]:
        runtime: _WorkerRuntime | None = getattr(
            self, "_single_inc_runtime", None
        )
        if runtime is None:
            return {"closed": 0}
        torch.npu.synchronize()
        runtime.executor.close()
        runtime.session.barrier()
        runtime.session.close()
        del self._single_inc_runtime
        return {"closed": 1, "rank": runtime.family.worker_rank}

    def get_single_inc_runtime_info(self) -> dict[str, int | str]:
        runtime: _WorkerRuntime | None = getattr(
            self, "_single_inc_runtime", None
        )
        if runtime is None:
            raise RuntimeError("single-INC runtime is not initialized")
        return {
            "rank": runtime.family.worker_rank,
            "device": int(torch.npu.current_device()),
            "capacity": runtime.executor.route.key.token_capacity,
            "mode": runtime.mode.value,
        }


# Imported only inside vLLM worker processes.  The subclass is selected with
# ``worker_cls`` so SHMEM can be initialized before the 30B model consumes HBM.
from vllm_ascend.worker.worker import NPUWorker  # noqa: E402


class SingleIncNPUWorker(NPUWorker, SingleIncWorkerExtension):
    """NPUWorker with a pre-weight-load ACLSHMEM lifecycle."""

    def _single_inc_parameters(self) -> dict[str, Any]:
        raw = os.environ.get("INC_VLLM_WORKER_CONFIG")
        if not raw:
            raise RuntimeError("INC_VLLM_WORKER_CONFIG is required")
        return json.loads(raw)

    def init_device(self) -> None:
        """Join ACLSHMEM before vLLM creates its HCCL process groups."""
        params = self._single_inc_parameters()
        mode = BackendMode(params["mode"])
        workers = int(params["worker_count"])
        hidden = int(params["hidden"])
        intermediate = int(params["intermediate"])
        experts = int(params["expert_count"])
        topk = int(params["topk"])
        capacity = int(params["token_capacity"])
        wave = int(params["tokens_per_wave"])
        activation_waves = int(params.get("activation_waves", 1))
        bridge = str(params["bridge_library"])
        if experts % workers:
            raise RuntimeError("experts must divide the vLLM EP world")
        if self.vllm_config.parallel_config.world_size != workers:
            raise RuntimeError("vLLM world size disagrees with INC config")
        rank = int(self.rank % workers)
        worker_devices = tuple(
            int(value) for value in params.get(
                "worker_devices", tuple(range(workers))
            )
        )
        if len(worker_devices) != workers or len(set(worker_devices)) != workers:
            raise RuntimeError("invalid single-INC worker device mapping")
        physical_device = worker_devices[rank]
        # NPUWorker._init_device() derives both its model device and HCCL
        # device from ``local_rank``.  Keep rank/PE numbering logical, but
        # replace that device selector before calling the base initializer;
        # merely calling set_device() here is insufficient because the base
        # class would switch back to physical 0..W-1.
        self._single_inc_logical_local_rank = int(self.local_rank)
        self.local_rank = physical_device
        device = torch.device(f"npu:{physical_device}")
        torch.npu.set_device(device)
        import shmem

        if shmem.set_conf_store_tls(False, "") != 0:
            raise RuntimeError("failed to disable ACLSHMEM config-store TLS")
        uid_file = Path(str(params["uid_path"]))
        if rank == 0:
            uid = shmem.core.get_unique_id()
            temporary = uid_file.with_name(uid_file.name + ".tmp")
            temporary.write_text(bytes(uid).hex())
            os.replace(temporary, uid_file)
        else:
            deadline = time.monotonic() + 300.0
            while not uid_file.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"worker rank 0 did not publish UID: {uid_file}"
                    )
                time.sleep(0.01)
            uid = bytes.fromhex(uid_file.read_text().strip())
        torch.ops.load_library(bridge)
        register_inc_fusion_op()
        local_experts = experts // workers
        owner = tuple(
            min(expert // local_experts, workers - 1)
            for expert in range(experts)
        )
        local = tuple(expert % local_experts for expert in range(experts))
        family = PlanFamily(
            worker_count=workers,
            worker_rank=rank,
            hidden=hidden,
            intermediate=intermediate,
            expert_count=experts,
            topk=topk,
            dtype="bf16",
            expert_placement_digest=0x51474550,
            hardware_profile_digest=0x910B2C,
        )
        key = PlanKey(family, capacity)
        config = WorkerExecutorConfig(
            live_aiv=int(params["live_aiv"]),
            live_aic=int(params["live_aic"]),
            inc_pe=workers,
            tokens_per_wave=wave,
            activation_waves=activation_waves,
            spin_cap=int(params.get("spin_cap", 0)),
        )
        info = query_plan_info(key, config, owner, local)
        ready_dir = params.get("worker_ready_dir")
        if mode.uses_inc:
            if not ready_dir:
                raise RuntimeError("INC mode requires worker_ready_dir")
            marker = Path(str(ready_dir)) / f"worker_{rank}.ready"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch(exist_ok=False)
        session = FusionShmemSession(
            uid=uid,
            pe=rank,
            world_size=workers + 1 if mode.uses_inc else workers,
            heap_bytes=int(params["heap_bytes"]),
            symmetric_bytes=info.symmetric_bytes,
            device=device,
        )
        self._single_inc_preload = (
            session,
            mode,
            family,
            key,
            config,
            owner,
            local,
            (
                None
                if int(params.get("route_lanes", -1)) < 0
                else int(params["route_lanes"])
            ),
        )
        # ACLSHMEM is now fully initialized. vLLM may safely create HCCL/EP
        # groups and its model runner without reinitializing the device.
        #
        # NPU's expandable allocator reserves a large segment when the first
        # live SHMEM tensor is created.  MemorySnapshot reports that reusable
        # reservation as unavailable even though subsequent model tensors are
        # allocated from it.  vLLM's startup guard therefore cannot compare
        # the post-SHMEM free counter with a utilization target based on total
        # HBM.  Use a conservative post-SHMEM value for that guard only, then
        # restore the requested process budget while charging the persistent
        # heap and worker workspace explicitly.
        target_utilization = float(self.cache_config.gpu_memory_utilization)
        free_bytes, total_bytes = torch.npu.mem_get_info()
        guard_utilization = min(
            target_utilization,
            max(0.01, float(free_bytes) / float(total_bytes) - 0.01),
        )
        self.cache_config.gpu_memory_utilization = guard_utilization
        try:
            super().init_device()
        finally:
            self.cache_config.gpu_memory_utilization = target_utilization
        persistent_bytes = int(params["heap_bytes"]) + int(
            info.worker_workspace_bytes
        )
        requested = int(total_bytes * target_utilization) - persistent_bytes
        if requested <= 0:
            raise MemoryError(
                "single-INC persistent runtime exceeds the requested vLLM "
                "HBM utilization budget"
            )
        self.requested_memory = requested

    def load_model(self) -> None:
        preload = getattr(self, "_single_inc_preload", None)
        if preload is None:
            raise RuntimeError("single-INC session was not prepared in init_device")
        session, mode, family, key, config, owner, local, route_lanes = preload
        workers = family.worker_count

        from vllm.distributed import get_ep_group

        ep = get_ep_group()
        rank = int(ep.rank_in_group)
        if ep.world_size != workers or rank != family.worker_rank:
            raise RuntimeError("vLLM EP topology disagrees with INC config")
        device = session.device
        route = PreparedTorchRoute(
            key,
            RouteRuntimeConfig(
                tokens_per_wave=config.tokens_per_wave,
                activation_waves=config.activation_waves,
                route_lanes=route_lanes,
            ),
            owner,
            local,
            ep.device_group,
            device,
        )
        worker_pes = torch.arange(workers, dtype=torch.int32, device=device)
        executor = PreparedTorchExecutor.from_session(
            route, config, worker_pes, session, _BACKEND_MODE_ID[mode]
        )
        session.barrier()

        # Model construction populates vLLM's communication registry and loads
        # source weights.  Install our method only after that construction has
        # finished, but before engine profile/warmup forwards begin.
        super().load_model()
        runtime = _WorkerRuntime(session, executor, mode, family)
        _install_comm_method(runtime)
        self._single_inc_runtime = runtime
        del self._single_inc_preload


__all__ = ["SingleIncNPUWorker", "SingleIncWorkerExtension"]
