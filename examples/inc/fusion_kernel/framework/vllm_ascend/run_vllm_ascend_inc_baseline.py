#!/usr/bin/env python3
"""Run a real vLLM-Ascend engine with the persistent single-INC backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from inc_moe_runtime import (
    BackendMode,
    FusionPeMapping,
    PlanFamily,
    PlanKey,
    SymmetricHeapPolicy,
    WorkerExecutorConfig,
)
from inc_sidecar import IncSidecarController, IncSidecarSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--mode",
        choices=("serial_shmem", "serial_inc", "fused_shmem", "fused_inc"),
        required=True,
    )
    parser.add_argument("--workers", type=int, choices=(2, 4), required=True)
    parser.add_argument(
        "--worker-devices",
        default=None,
        help=(
            "comma-separated physical NPU ids; defaults to 0..workers-1. "
            "This does not change logical PE/rank numbering."
        ),
    )
    parser.add_argument(
        "--inc-device",
        type=int,
        default=None,
        help="physical NPU id for the INC sidecar; defaults to workers",
    )
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure", type=int, default=3)
    parser.add_argument(
        "--enable-graph",
        action="store_true",
        help=(
            "enable vLLM torch.compile/NPU graph mode; the default remains "
            "eager until the replay-safe generation protocol is qualified"
        ),
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help=(
            "enable vLLM compilation but disable NPU graph capture/replay; "
            "this keeps host-generated INC tickets replay-safe"
        ),
    )
    parser.add_argument(
        "--piecewise-inc-graph",
        action="store_true",
        help=(
            "capture graph-compatible model segments while keeping "
            "vllm::moe_forward (and its live INC ticket) outside replay"
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--bridge-library",
        default="/workspace/inc-runtime/build/torch-bridge-cann851/libinc_fusion_torch_bridge.so",
    )
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=768)
    parser.add_argument("--expert-count", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--live-aiv", type=int, default=40)
    parser.add_argument("--live-aic", type=int, default=20)
    parser.add_argument(
        "--tokens-per-wave",
        type=int,
        default=0,
        help="0 selects min(128, max-num-batched-tokens)",
    )
    parser.add_argument("--activation-waves", type=int, default=1)
    parser.add_argument(
        "--spin-cap",
        type=int,
        default=0,
        help="protocol wait limit; zero keeps production waits unbounded",
    )
    parser.add_argument(
        "--service-ring-size",
        type=int,
        default=4,
        help="persistent INC descriptor ring depth",
    )
    parser.add_argument(
        "--executor-ring-size",
        type=int,
        default=64,
        help="forwards the host may enqueue before the oldest worker kernel runs",
    )
    parser.add_argument(
        "--route-lanes",
        type=int,
        default=-1,
        help="-1 selects the device policy; 0 forces the scalar route packer",
    )
    parser.add_argument(
        "--torch-profiler-dir",
        type=Path,
        default=None,
        help="optional diagnostic trace directory; profiles the first measured request",
    )
    parser.add_argument(
        "--scenarios-json",
        default='[{"name":"prefill_128_b1","input_len":128,"output_len":1,"batch_size":1}]',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_modes = sum((
        bool(args.enable_graph),
        bool(args.compile_only),
        bool(args.piecewise_inc_graph),
    ))
    if graph_modes > 1:
        raise ValueError(
            "enable-graph, compile-only and piecewise-inc-graph are "
            "mutually exclusive"
        )
    selected_mode = BackendMode(args.mode)
    worker_devices = (
        tuple(range(args.workers))
        if args.worker_devices is None
        else tuple(int(value) for value in args.worker_devices.split(","))
    )
    inc_device = args.workers if args.inc_device is None else args.inc_device
    if len(worker_devices) != args.workers:
        raise ValueError("worker-devices must contain exactly workers ids")
    if any(device < 0 for device in worker_devices) or inc_device < 0:
        raise ValueError("physical NPU ids must be non-negative")
    if len(set((*worker_devices, inc_device))) != args.workers + 1:
        raise ValueError("worker and INC physical NPU ids must be unique")
    if args.expert_count % args.workers:
        raise ValueError("expert_count must divide workers")
    scenarios = json.loads(args.scenarios_json)
    # vLLM's All2All prepare slices the scheduler's global token batch across
    # TP/EP ranks before calling fused_experts.  The persistent executor is
    # rank-local, so sizing every worker for the unsliced global scheduler cap
    # creates W times too many protocol waves and W times too much workspace.
    # tensor_split gives the largest rank ceil(global/W) rows.
    worker_token_capacity = (
        args.max_num_batched_tokens + args.workers - 1
    ) // args.workers
    tokens_per_wave = args.tokens_per_wave or min(
        128, worker_token_capacity
    )
    if not 1 <= tokens_per_wave <= worker_token_capacity:
        raise ValueError(
            "tokens_per_wave must be in [1, per-worker token capacity]"
        )
    if args.activation_waves <= 0:
        raise ValueError("activation_waves must be positive")
    if args.spin_cap < 0:
        raise ValueError("spin_cap must be non-negative")
    if not -1 <= args.route_lanes <= 64:
        raise ValueError("route_lanes must be -1 or in [0, 64]")
    owner = tuple(
        expert // (args.expert_count // args.workers)
        for expert in range(args.expert_count)
    )
    local = tuple(
        expert % (args.expert_count // args.workers)
        for expert in range(args.expert_count)
    )
    family = PlanFamily(
        worker_count=args.workers,
        worker_rank=0,
        hidden=args.hidden,
        intermediate=args.intermediate,
        expert_count=args.expert_count,
        topk=args.topk,
        dtype="bf16",
        expert_placement_digest=0x51474550,
        hardware_profile_digest=0x910B2C,
    )
    key = PlanKey(family, worker_token_capacity)
    config = WorkerExecutorConfig(
        live_aiv=args.live_aiv,
        live_aic=args.live_aic,
        inc_pe=args.workers,
        tokens_per_wave=tokens_per_wave,
        activation_waves=args.activation_waves,
        spin_cap=args.spin_cap,
        service_ring_size=args.service_ring_size,
        executor_ring_size=args.executor_ring_size,
    )
    mapping = FusionPeMapping(
        worker_pes=tuple(range(args.workers)),
        inc_pe=args.workers,
        worker_devices=worker_devices,
        inc_device=inc_device,
    )

    import os
    from vllm import LLM, SamplingParams

    worker_ready_dir = Path(
        "/workspace/inc-runtime/control"
    ) / f"vllm-inc-{os.getpid()}-{time.time_ns()}"
    worker_ready_dir.mkdir(parents=True, exist_ok=False)
    uid_path = worker_ready_dir / "aclshmem.uid"
    # query_plan_info in the sidecar computes the exact requirement; the heap
    # policy keeps the allocation identical on all W+1 PEs.
    sidecar = None
    if selected_mode.uses_inc:
        spec = IncSidecarSpec(
            key=key,
            config=config,
            mapping=mapping,
            expert_owner=owner,
            expert_local_index=local,
            bridge_library=args.bridge_library,
            worker_ready_dir=str(worker_ready_dir),
            uid_path=str(uid_path),
        )
        sidecar = IncSidecarController(spec, None)
        prepared = sidecar.wait_state("PREPARED", 120.0)
        heap_bytes = int(prepared["heap_bytes"])
    else:
        # Worker-direct SHMEM still uses the same symmetric wave arenas as
        # the INC modes.  A fixed minimum heap only happens to cover small
        # token capacities and rejects larger, otherwise valid plans.  Query
        # the exact plan before vLLM spawns its workers, then apply the same
        # deployment reserve/alignment policy used by the INC sidecar.
        import torch
        torch.ops.load_library(args.bridge_library)
        from torch_fusion_runtime import query_all_worker_plan_info

        direct_info = query_all_worker_plan_info(
            key, config, owner, local
        )[0]
        heap_bytes = SymmetricHeapPolicy().heap_bytes(direct_info)
    os.environ["INC_VLLM_WORKER_CONFIG"] = json.dumps(
        {
            "uid_path": str(uid_path),
            "mode": args.mode,
            "worker_count": args.workers,
            "worker_devices": list(worker_devices),
            "inc_device": inc_device,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "expert_count": args.expert_count,
            "topk": args.topk,
            "token_capacity": worker_token_capacity,
            "tokens_per_wave": tokens_per_wave,
            "activation_waves": args.activation_waves,
            "spin_cap": args.spin_cap,
            "route_lanes": args.route_lanes,
            "live_aiv": args.live_aiv,
            "live_aic": args.live_aic,
            "bridge_library": args.bridge_library,
            "heap_bytes": heap_bytes,
            "worker_ready_dir": str(worker_ready_dir),
        }
    )
    try:
        profiler_config = None
        if args.torch_profiler_dir is not None:
            args.torch_profiler_dir.mkdir(parents=True, exist_ok=True)
            profiler_config = {
                "profiler": "torch",
                "torch_profiler_dir": str(args.torch_profiler_dir),
            }
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.workers,
            enable_expert_parallel=True,
            enforce_eager=not (
                args.enable_graph
                or args.compile_only
                or args.piecewise_inc_graph
            ),
            compilation_config=(
                {"cudagraph_mode": "NONE"}
                if args.compile_only
                else {
                    "mode": "VLLM_COMPILE",
                    "cudagraph_mode": "PIECEWISE",
                    "splitting_ops": [
                        "vllm::unified_attention",
                        "vllm::unified_attention_with_output",
                        "vllm::unified_mla_attention",
                        "vllm::unified_mla_attention_with_output",
                        "vllm::mamba_mixer2",
                        "vllm::mamba_mixer",
                        "vllm::short_conv",
                        "vllm::linear_attention",
                        "vllm::plamo2_mamba_mixer",
                        "vllm::gdn_attention_core",
                        "vllm::olmo_hybrid_gdn_full_forward",
                        "vllm::kda_attention",
                        "vllm::sparse_attn_indexer",
                        "vllm::rocm_aiter_sparse_attn_indexer",
                        "vllm::unified_kv_cache_update",
                        "vllm::unified_mla_kv_cache_update",
                        "vllm::moe_forward",
                    ],
                }
                if args.piecewise_inc_graph
                else None
            ),
            enable_prefix_caching=False,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=False,
            worker_cls="vllm_engine_runtime.SingleIncNPUWorker",
            profiler_config=profiler_config,
        )
    except BaseException:
        if sidecar is not None:
            sidecar.abort()
        raise
    if sidecar is not None:
        sidecar.wait_state("READY", 120.0)
    setup = llm.collective_rpc(
        "get_single_inc_runtime_info", timeout=120.0
    )
    print("INC_VLLM_SETUP " + json.dumps(setup, sort_keys=True))

    rng = np.random.default_rng(20260809)
    results: list[dict[str, object]] = []
    try:
        for scenario in scenarios:
            prompts = [
                {"prompt_token_ids": row.tolist()}
                for row in rng.integers(
                    100,
                    10000,
                    size=(int(scenario["batch_size"]), int(scenario["input_len"])),
                )
            ]
            sampling = SamplingParams(
                temperature=0.0,
                ignore_eos=True,
                max_tokens=int(scenario["output_len"]),
                detokenize=False,
            )
            for _ in range(args.warmup):
                llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
            samples = []
            measured_token_ids: list[list[list[int]]] = []
            for measure_index in range(args.measure):
                profile_this_request = (
                    args.torch_profiler_dir is not None and measure_index == 0
                )
                if profile_this_request:
                    llm.start_profile("inc_fusion")
                begin = time.perf_counter()
                request_outputs = llm.generate(
                    prompts, sampling_params=sampling, use_tqdm=False
                )
                samples.append(time.perf_counter() - begin)
                measured_token_ids.append([
                    [int(token) for token in request.outputs[0].token_ids]
                    for request in request_outputs
                ])
                if profile_this_request:
                    llm.stop_profile()
            mean_s = statistics.mean(samples)
            result = {
                **scenario,
                "latencies_s": samples,
                "mean_s": mean_s,
                "median_s": statistics.median(samples),
                "cv_percent": statistics.pstdev(samples) / mean_s * 100.0
                if len(samples) > 1
                else 0.0,
                "output_tokens_per_second": int(scenario["batch_size"])
                * int(scenario["output_len"])
                / mean_s,
                # Deterministic greedy output signature.  It is collected
                # outside the timed device execution and permits exact
                # native-vs-INC correctness comparison for every benchmark
                # scenario, rather than treating "no runtime error" as
                # numerical correctness.
                "measured_token_ids": measured_token_ids,
                "output_stable": all(
                    value == measured_token_ids[0]
                    for value in measured_token_ids[1:]
                ),
            }
            results.append(result)
            print("INC_VLLM_RESULT " + json.dumps(result, sort_keys=True))
    finally:
        if sidecar is not None:
            sidecar.request_stop()
            sidecar.wait_state("STOPPING", 120.0)
        closed = llm.collective_rpc("close_single_inc_runtime", timeout=300.0)
        if sidecar is not None:
            sidecar.wait_state("STOPPED", 120.0)
            sidecar.join(30.0)
        print("INC_VLLM_CLOSED " + json.dumps(closed, sort_keys=True))

    payload = {
        "mode": args.mode,
        "workers": args.workers,
        "worker_devices": list(worker_devices),
        "inc_device": inc_device,
        "model": args.model,
        "warmup": args.warmup,
        "measure": args.measure,
        "tokens_per_wave": tokens_per_wave,
        "activation_waves": args.activation_waves,
        "global_token_capacity": args.max_num_batched_tokens,
        "worker_token_capacity": worker_token_capacity,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
