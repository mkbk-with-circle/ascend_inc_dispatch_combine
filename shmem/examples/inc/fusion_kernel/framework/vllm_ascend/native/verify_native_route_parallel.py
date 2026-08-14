"""Byte-exact and error-path qualification for parallel route packing."""

from __future__ import annotations

import os
import struct

import torch
import torch_npu  # noqa: F401

from bench_native_route_ops import parallel_scratch_words


def status_words(status: torch.Tensor) -> tuple[int, ...]:
    return struct.unpack_from("<8I", bytes(status.cpu().tolist()))


def buffers(workers: int, tokens: int, topk: int, experts: int, id_dtype):
    device = torch.device("npu:0")
    tokens_per_wave = min(256, tokens)
    wave_count = (tokens + tokens_per_wave - 1) // tokens_per_wave
    ids = torch.arange(tokens * topk, dtype=id_dtype, device=device).view(
        tokens, topk
    ).remainder_(experts)
    weights = torch.arange(
        tokens * topk, dtype=torch.float32, device=device
    ).view(tokens, topk).remainder_(17).add_(1).div_(17)
    owner = torch.arange(experts, dtype=torch.int32, device=device).remainder_(
        workers
    )
    local = torch.arange(experts, dtype=torch.int32, device=device).floor_divide_(
        workers
    )
    counts = torch.empty(
        (wave_count, experts + 1), dtype=torch.int32, device=device
    )
    global_counts = torch.empty(
        (workers, wave_count, experts + 1), dtype=torch.int32, device=device
    )
    rows = torch.empty(
        tokens * min(topk, workers) * 32, dtype=torch.uint8, device=device
    )
    assignments = torch.empty(
        tokens * topk * 32, dtype=torch.uint8, device=device
    )
    local_experts = experts // workers
    groups = torch.empty(
        (wave_count, local_experts), dtype=torch.int64, device=device
    )
    waves = torch.empty(wave_count * 64, dtype=torch.uint8, device=device)
    status = torch.empty(64, dtype=torch.uint8, device=device)
    torch.ops.inc_fusion_native.route_count_out(
        ids, counts, status, tokens_per_wave
    )
    global_counts.copy_(counts.unsqueeze(0).expand_as(global_counts))
    return (
        ids, weights, owner, local, global_counts, rows, assignments, groups,
        waves, status, tokens_per_wave,
    )


def pack_args(values, scratch):
    (
        ids, weights, owner, local, global_counts, rows, assignments, groups,
        waves, status, tokens_per_wave,
    ) = values
    return (
        ids, weights, global_counts, owner, local, rows, assignments, groups,
        waves, scratch, status, 0, 256, 2, tokens_per_wave, 3, 2,
    )


def exact_case(workers: int, tokens: int, topk: int, id_dtype) -> None:
    experts = 64
    values = buffers(workers, tokens, topk, experts, id_dtype)
    rows, assignments, groups, waves, status = (
        values[5], values[6], values[7], values[8], values[9]
    )
    scalar_scratch = torch.empty(2 * experts, dtype=torch.int32, device="npu:0")
    torch.ops.inc_fusion_native.route_pack_out(
        *pack_args(values, scalar_scratch)
    )
    torch.npu.synchronize()
    scalar_status = status_words(status)
    assert scalar_status[0] == 0, scalar_status
    row_bytes = scalar_status[4] * 32
    assignment_bytes = scalar_status[5] * 32
    expected = (
        rows[:row_bytes].clone(), assignments[:assignment_bytes].clone(),
        groups.clone(), waves.clone(),
    )

    lanes = int(torch.npu.get_device_properties(0).vector_core_num)
    parallel_scratch = torch.empty(
        parallel_scratch_words(groups.shape[0], experts, lanes),
        dtype=torch.int32, device="npu:0",
    )
    torch.ops.inc_fusion_native.route_pack_parallel_out(
        *pack_args(values, parallel_scratch), lanes
    )
    torch.npu.synchronize()
    parallel_status = status_words(status)
    assert parallel_status == scalar_status, (scalar_status, parallel_status)
    actual = (
        rows[:row_bytes], assignments[:assignment_bytes], groups, waves,
    )
    assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(actual, expected)), (
        workers, tokens, topk, id_dtype
    )


def error_case(kind: str) -> None:
    workers, tokens, topk, experts = 4, 37, 8, 64
    values = list(buffers(workers, tokens, topk, experts, torch.int64))
    if kind == "expert":
        values[0][3, 1] = experts
        expected = (2, 3, 1, experts)
    elif kind == "weight":
        values[1][3, 1] = float("nan")
        expected = (5, 3, 1, 25)
    else:
        raise AssertionError(kind)
    lanes = int(torch.npu.get_device_properties(0).vector_core_num)
    scratch = torch.empty(
        parallel_scratch_words(values[7].shape[0], experts, lanes),
        dtype=torch.int32, device="npu:0",
    )
    torch.ops.inc_fusion_native.route_pack_parallel_out(
        *pack_args(values, scratch), lanes
    )
    torch.npu.synchronize()
    observed = status_words(values[9])
    assert observed[:4] == expected, (kind, observed, expected)


def main() -> None:
    torch.ops.load_library(os.environ["INC_FUSION_TORCH_LIBRARY"])
    torch.npu.set_device(0)
    for workers in (2, 4):
        for tokens in (37, 513, 8192):
            for topk in (2, 8):
                for dtype in (torch.int32, torch.int64):
                    exact_case(workers, tokens, topk, dtype)
    error_case("expert")
    error_case("weight")
    print("parallel route: 24 large/tail byte-exact cases and 2 errors PASS")


if __name__ == "__main__":
    main()
