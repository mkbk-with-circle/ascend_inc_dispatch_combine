"""Synthetic device-only route count+pack latency sweep (no metadata HCCL)."""

from __future__ import annotations

import csv
import os
import statistics
import struct
import sys
import time

import torch
import torch_npu  # noqa: F401


def parallel_scratch_words(waves: int, experts: int, lanes: int) -> int:
    expert_stride = ((experts + 15) // 16) * 16
    return (
        waves * lanes * expert_stride
        + waves * lanes * 16
        + waves * expert_stride
        + lanes * 16
    )


def run_case(workers: int, tokens: int, topk: int, experts: int = 64) -> list:
    device = torch.device("npu:0")
    tokens_per_wave = min(256, max(1, tokens))
    waves_count = (tokens + tokens_per_wave - 1) // tokens_per_wave
    ids = (
        torch.arange(tokens * topk, dtype=torch.int64, device=device)
        .view(tokens, topk)
        .remainder_(experts)
    )
    weights = torch.full(
        (tokens, topk), 1.0 / topk, dtype=torch.float32, device=device
    )
    owner = torch.arange(experts, dtype=torch.int32, device=device).remainder_(
        workers
    )
    local = torch.arange(experts, dtype=torch.int32, device=device).floor_divide_(
        workers
    )
    local_experts = (experts + workers - 1) // workers
    counts = torch.empty(
        (waves_count, experts + 1), dtype=torch.int32, device=device
    )
    global_counts = torch.empty(
        (workers, waves_count, experts + 1),
        dtype=torch.int32,
        device=device,
    )
    status = torch.empty(64, dtype=torch.uint8, device=device)
    rows = torch.empty(
        tokens * min(topk, workers) * 32, dtype=torch.uint8, device=device
    )
    assignments = torch.empty(tokens * topk * 32, dtype=torch.uint8, device=device)
    groups = torch.empty(
        (waves_count, local_experts), dtype=torch.int64, device=device
    )
    waves = torch.empty(waves_count * 64, dtype=torch.uint8, device=device)
    use_parallel = os.environ.get("INC_FUSION_ROUTE_PARALLEL") == "1"
    route_lanes = int(os.environ.get(
        "INC_FUSION_ROUTE_LANES",
        torch.npu.get_device_properties(0).vector_core_num,
    )) if use_parallel else 0
    scratch_words = parallel_scratch_words(
        waves_count, experts, route_lanes
    ) if use_parallel else 2 * experts
    scratch = torch.empty(scratch_words, dtype=torch.int32, device=device)

    torch.ops.inc_fusion_native.route_count_out(
        ids, counts, status, tokens_per_wave
    )
    global_counts.copy_(counts.unsqueeze(0).expand_as(global_counts))
    torch.npu.synchronize()
    debug = os.environ.get("INC_FUSION_ROUTE_DEBUG") == "1"
    if debug:
        count_words = struct.unpack_from("<8I", bytes(status.cpu().tolist()))
        print(
            f"count W{workers} T{tokens} K{topk}: status={count_words} "
            f"active={global_counts[:, 0, -1].cpu().tolist()} "
            f"ptrs=ids:{ids.data_ptr():x},weights:{weights.data_ptr():x},"
            f"counts:{counts.data_ptr():x},global:{global_counts.data_ptr():x},"
            f"owner:{owner.data_ptr():x},local:{local.data_ptr():x},"
            f"rows:{rows.data_ptr():x},assign:{assignments.data_ptr():x},"
            f"groups:{groups.data_ptr():x},waves:{waves.data_ptr():x},"
            f"scratch:{scratch.data_ptr():x},status:{status.data_ptr():x}",
            file=sys.stderr,
        )

    def enqueue() -> None:
        torch.ops.inc_fusion_native.route_count_out(
            ids, counts, status, tokens_per_wave
        )
        pack_op = torch.ops.inc_fusion_native.route_pack_parallel_out \
            if use_parallel else torch.ops.inc_fusion_native.route_pack_out
        args = (
            ids,
            weights,
            global_counts,
            owner,
            local,
            rows,
            assignments,
            groups,
            waves,
            scratch,
            status,
            0,
            256,
            2,
            tokens_per_wave,
            3,
            2,
        )
        if use_parallel:
            pack_op(*args, route_lanes)
        else:
            pack_op(*args)

    for _ in range(5):
        enqueue()
    torch.npu.synchronize()
    samples_us = []
    for _ in range(20):
        begin = time.perf_counter()
        enqueue()
        torch.npu.synchronize()
        samples_us.append((time.perf_counter() - begin) * 1e6)
    mean = statistics.mean(samples_us)
    cv = statistics.pstdev(samples_us) / mean * 100.0
    status_words = struct.unpack_from("<8I", bytes(status.cpu().tolist()))
    if debug and status_words[0] != 0:
        print(
            f"pack failure W{workers} T{tokens} K{topk}: "
            f"status={status_words} owner={owner.cpu().tolist()} "
            f"local={local.cpu().tolist()} "
            f"active={global_counts[:, 0, -1].cpu().tolist()}",
            file=sys.stderr,
        )
    return [
        "parallel" if use_parallel else "scalar",
        route_lanes,
        workers,
        tokens,
        topk,
        experts,
        tokens_per_wave,
        len(samples_us),
        f"{mean:.3f}",
        f"{statistics.median(samples_us):.3f}",
        f"{min(samples_us):.3f}",
        f"{max(samples_us):.3f}",
        f"{cv:.3f}",
        status_words[0],
        status_words[1],
        status_words[2],
        status_words[3],
        status_words[4],
        status_words[5],
    ]


def main() -> None:
    torch.ops.load_library(os.environ["INC_FUSION_TORCH_LIBRARY"])
    torch.npu.set_device(0)
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "route_impl",
            "route_lanes",
            "workers",
            "tokens",
            "topk",
            "experts",
            "tokens_per_wave",
            "measure",
            "mean_us",
            "median_us",
            "min_us",
            "max_us",
            "cv_percent",
            "status_error",
            "error_token",
            "error_ordinal",
            "error_expert",
            "dispatch_rows",
            "assignments",
        ]
    )
    for workers in (2, 4):
        for topk in (2, min(8, 64)):
            for tokens in (32, 128, 512, 2048, 8192):
                writer.writerow(run_case(workers, tokens, topk))
                sys.stdout.flush()


if __name__ == "__main__":
    main()
