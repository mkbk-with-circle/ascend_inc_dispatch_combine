#!/usr/bin/env python3
"""Measure the unmodified vLLM-Ascend execution path with one model load.

This is deliberately independent of the INC bridge.  It is a system-level
reference, not an isolated one-layer MoE comparison: prompts pass through the
normal vLLM scheduler, model runner, attention and native Ascend MoE path.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Optional scheduler cap; set it to the custom path's value for a fair comparison.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable the native graph/compile path (diagnostic only).",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--torch-profiler-dir",
        type=Path,
        default=None,
        help="optional diagnostic trace directory; profiles the first measured request",
    )
    parser.add_argument(
        "--scenarios-json",
        default='[{"name":"prefill_128_b1","input_len":128,"output_len":1,"batch_size":1},'
        '{"name":"prefill_512_b1","input_len":512,"output_len":1,"batch_size":1},'
        '{"name":"decode_32_b8","input_len":32,"output_len":32,"batch_size":8}]',
    )
    return parser.parse_args()


def synchronize() -> None:
    import torch

    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def main() -> None:
    args = parse_args()
    scenarios = json.loads(args.scenarios_json)
    if args.tensor_parallel_size < 1 or args.warmup < 0 or args.measure < 1:
        raise ValueError("invalid benchmark count or parallel size")

    # Imported lazily so --help remains cheap and does not initialize an NPU.
    from vllm import LLM, SamplingParams

    profiler_config = None
    if args.torch_profiler_dir is not None:
        args.torch_profiler_dir.mkdir(parents=True, exist_ok=True)
        profiler_config = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.torch_profiler_dir),
        }
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=True,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=False,
        profiler_config=profiler_config,
    )
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(**llm_kwargs)
    rng = np.random.default_rng(20260809)
    results: list[dict[str, object]] = []
    for scenario in scenarios:
        input_len = int(scenario["input_len"])
        output_len = int(scenario["output_len"])
        batch_size = int(scenario["batch_size"])
        if input_len + output_len > args.max_model_len:
            raise ValueError(f"scenario exceeds max_model_len: {scenario}")
        prompts = [
            {"prompt_token_ids": row.tolist()}
            for row in rng.integers(100, 10000, size=(batch_size, input_len))
        ]
        sampling = SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=output_len,
            detokenize=False,
        )

        for _ in range(args.warmup):
            llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
        latencies = []
        measured_token_ids: list[list[list[int]]] = []
        for measure_index in range(args.measure):
            profile_this_request = (
                args.torch_profiler_dir is not None and measure_index == 0
            )
            if profile_this_request:
                llm.start_profile("native_vllm")
            synchronize()
            begin = time.perf_counter()
            request_outputs = llm.generate(
                prompts, sampling_params=sampling, use_tqdm=False
            )
            synchronize()
            latencies.append(time.perf_counter() - begin)
            measured_token_ids.append([
                [int(token) for token in request.outputs[0].token_ids]
                for request in request_outputs
            ])
            if profile_this_request:
                llm.stop_profile()

        mean_s = statistics.mean(latencies)
        output_tokens = batch_size * output_len
        results.append(
            {
                **scenario,
                "latencies_s": latencies,
                "mean_s": mean_s,
                "median_s": statistics.median(latencies),
                "cv_percent": (
                    statistics.pstdev(latencies) / mean_s * 100.0
                    if len(latencies) > 1
                    else 0.0
                ),
                "output_tokens_per_second": output_tokens / mean_s,
                "measured_token_ids": measured_token_ids,
                "output_stable": all(
                    value == measured_token_ids[0]
                    for value in measured_token_ids[1:]
                ),
                "interpretation": (
                    "batch TTFT proxy (full native prefill plus first token)"
                    if output_len == 1
                    else "full native prefill+decode batch completion"
                ),
            }
        )
        print("VLLM_NATIVE_RESULT " + json.dumps(results[-1], sort_keys=True))

    payload = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "expert_parallel": True,
        "fused_mc2": False,
        "enforce_eager": args.enforce_eager,
        "warmup": args.warmup,
        "measure": args.measure,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
