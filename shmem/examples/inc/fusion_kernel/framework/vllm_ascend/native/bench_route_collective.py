#!/usr/bin/env python3
"""Measure tiny route-metadata collectives on the active EP plane."""

from __future__ import annotations

import json
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def measure(call, warmup: int = 20, samples: int = 100) -> dict[str, float]:
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    values = []
    for _ in range(samples):
        begin = time.perf_counter()
        call()
        torch.npu.synchronize()
        values.append((time.perf_counter() - begin) * 1e6)
    mean = statistics.mean(values)
    return {
        "mean_us": mean,
        "median_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
        "cv_percent": statistics.pstdev(values) / mean * 100.0,
    }


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    workers = dist.get_world_size()
    local = torch.full((1, 129), rank + 1, dtype=torch.int32, device="npu")
    gathered = torch.empty((workers, 1, 129), dtype=torch.int32, device="npu")
    reduced = local.clone()

    def gather() -> None:
        dist.all_gather_into_tensor(gathered.view(workers, 129), local)

    def all_reduce() -> None:
        reduced.copy_(local)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)

    result = {
        "rank": rank,
        "workers": workers,
        "bytes_per_rank": local.numel() * local.element_size(),
        "all_gather": measure(gather),
        "all_reduce": measure(all_reduce),
    }
    if rank == 0:
        print(json.dumps(result, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
