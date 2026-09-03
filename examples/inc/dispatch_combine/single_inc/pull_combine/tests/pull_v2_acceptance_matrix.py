#!/usr/bin/env python3
"""Generate the immutable Pull-Dispatch/Combine V2 acceptance manifest.

This module is deliberately plan-only: it never imports accelerator libraries,
starts a process, or touches an NPU.  The device runner should consume this
manifest after its JSONL sample ABI is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MIB = 1 << 20
GIB = 1 << 30
WORKERS = (2, 4)
DTYPES = ("bf16", "fp16", "fp32")
SIZE_STEPS = (
    0,
    4 << 10,
    64 << 10,
    1 << 20,
    16 << 20,
    64 << 20,
    128 << 20,
    256 << 20,
    512 << 20,
    1 << 30,
    2 << 30,
)
WORKLOADS = (
    "sym_k1_rr",
    "sym_dense",
    "same_gpu_multi_expert",
    "rank_token_skew",
    "empty_sources",
    "hotspot_all1",
    "hotspot_90_10",
    "ragged_topk",
    "repeated_token_id",
)

DISPATCH_GATE = {2: 51.52, 4: 103.04}


def derived_seed(campaign_seed: int, case_id: str) -> int:
    material = f"pull-v2:{campaign_seed}:{case_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def case(
    campaign_seed: int,
    suite: str,
    operator: str,
    workers: int,
    workload: str,
    *,
    target_bytes: int | str,
    dtype: str,
    per_worker_hidden_payload_bytes: int | str | None = None,
    row_bytes: int | None = None,
    token_count: int | None = None,
    ready_order: str = "forward",
    fault: str = "none",
) -> dict[str, Any]:
    shape = f"B{target_bytes}" if isinstance(target_bytes, int) else target_bytes
    details = [suite, operator, f"W{workers}", workload, dtype, shape]
    if row_bytes is not None:
        details.append(f"row{row_bytes}")
    if token_count is not None:
        details.append(f"tokens{token_count}")
    if ready_order != "forward":
        details.append(ready_order)
    if fault != "none":
        details.append(fault)
    case_id = "-".join(str(value) for value in details)
    hard_gate = (
        suite == "performance"
        and operator == "dispatch"
        and workload == "sym_dense"
        and per_worker_hidden_payload_bytes == 128 * MIB
    )
    hard_gate_gbps = DISPATCH_GATE[workers] if hard_gate else None
    return {
        "case_id": case_id,
        "suite": suite,
        "operator": operator,
        "workers": workers,
        "workload": workload,
        "target_logical_bytes": target_bytes,
        "per_worker_hidden_payload_bytes": per_worker_hidden_payload_bytes,
        "dtype": dtype,
        "row_bytes": row_bytes,
        "token_count": token_count,
        "ready_order": ready_order,
        "fault": fault,
        "seed": derived_seed(campaign_seed, case_id),
        "warmup": 3 if suite in ("performance", "workload", "overlap") else 0,
        "measure": 10 if suite in ("performance", "workload", "overlap") else 1,
        "hard_gate": hard_gate,
        "hard_gate_GBps": hard_gate_gbps,
        "require_cv_le": 0.05 if suite in ("performance", "workload", "overlap") else None,
        "report_relative_to": (
            "matched_sym_dense"
            if suite == "workload" and workload != "sym_dense" else None
        ),
    }


def boundary_cases(seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Values are selected around cache-line, transport tile and reducer tile
    # boundaries.  Invalid dtype alignment is intentionally omitted here and
    # belongs to the negative host-validation suite.
    row_bytes_by_dtype = {
        "bf16": (2, 62, 64, 66, 8190, 8192, 8194),
        "fp16": (2, 62, 64, 66, 8190, 8192, 8194),
        "fp32": (4, 60, 64, 68, 8188, 8192, 8196),
    }
    token_counts = (0, 1, 2, 3, 31, 32, 33, 255, 256, 257)
    for workers in WORKERS:
        for operator in ("dispatch", "combine"):
            for dtype in DTYPES:
                for row_bytes in row_bytes_by_dtype[dtype]:
                    for token_count in token_counts:
                        rows.append(case(
                            seed, "boundary", operator, workers,
                            "sym_dense" if token_count else "empty_sources",
                            target_bytes=row_bytes * token_count * workers,
                            dtype=dtype, row_bytes=row_bytes,
                            token_count=token_count,
                        ))
            if operator == "combine":
                for hidden in (1535, 1536, 1537):
                    rows.append(case(
                        seed, "boundary", operator, workers, "sym_dense",
                        target_bytes=hidden * 4 * workers,
                        dtype="fp32", row_bytes=hidden * 4,
                        token_count=1,
                    ))
    return rows


def workload_cases(seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workers in WORKERS:
        for operator in ("dispatch", "combine"):
            for workload in WORKLOADS:
                for target in (1 * MIB, 64 * MIB, 256 * MIB):
                    rows.append(case(
                        seed, "workload", operator, workers, workload,
                        target_bytes=target,
                        dtype="bf16" if operator == "dispatch" else "fp32",
                    ))
    return rows


def performance_cases(seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workers in WORKERS:
        for payload in SIZE_STEPS:
            rows.append(case(
                seed, "performance", "dispatch", workers, "sym_dense",
                # In symmetric dense routing every source byte is pulled once
                # and sent once to each of W unique destinations.
                target_bytes=payload * workers * (workers + 1),
                dtype="bf16",
                per_worker_hidden_payload_bytes=payload,
            ))
        rows.append(case(
            seed, "performance", "dispatch", workers, "sym_dense",
            target_bytes="max_hbm_safe",
            dtype="bf16",
            per_worker_hidden_payload_bytes="max_hbm_safe",
        ))
        for target in SIZE_STEPS:
            rows.append(case(
                seed, "performance", "combine", workers, "sym_dense",
                target_bytes=target, dtype="fp32",
            ))
        rows.append(case(
            seed, "performance", "combine", workers, "sym_dense",
            target_bytes="max_hbm_safe", dtype="fp32",
        ))
    return rows


def ordering_cases(seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    orders = ("reverse", "rotate1", "rotate2", *(f"random{i}" for i in range(8)))
    for workers in WORKERS:
        for operator in ("dispatch", "combine"):
            for order in orders:
                rows.append(case(
                    seed, "ordering", operator, workers, "ragged_topk",
                    target_bytes=16 * MIB,
                    dtype="bf16" if operator == "dispatch" else "fp32",
                    ready_order=order,
                ))
    return rows


def fault_cases(seed: int) -> list[dict[str, Any]]:
    dispatch_faults = (
        "ready_identity", "ready_publication", "ready_duplicate",
        "ready_missing", "header_identity", "header_offset",
        "header_overflow", "metadata_digest", "csr_gap",
        "destination_oob", "expert_oob", "ordinal_duplicate",
        "weight_nonfinite", "capacity_minus_one", "stale_replay",
        "early_slot_reuse",
    )
    combine_faults = (
        "cookie_mismatch", "ready_identity", "ready_publication",
        "ready_duplicate", "ready_missing", "region_oob",
        "payload_size", "contributor_missing", "contributor_extra",
        "destination_row_oob", "stale_replay", "early_slot_reuse",
    )
    rows: list[dict[str, Any]] = []
    for workers in WORKERS:
        for operator, faults in (
            ("dispatch", dispatch_faults), ("combine", combine_faults)
        ):
            for fault in faults:
                rows.append(case(
                    seed, "fault_recovery", operator, workers,
                    "ragged_topk", target_bytes=1 * MIB,
                    dtype="bf16" if operator == "dispatch" else "fp32",
                    fault=fault,
                ))
    return rows


def overlap_cases(seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    schedules = (
        "simultaneous", "dispatch_lead_10us", "dispatch_lead_100us",
        "dispatch_lead_25pct", "dispatch_lead_75pct",
        "combine_lead_10us", "combine_lead_100us",
        "combine_lead_25pct", "combine_lead_75pct", "rank_jitter",
        "four_wave_steady_state",
    )
    for workers in WORKERS:
        for target in (64 * MIB, 256 * MIB, 1 * GIB):
            for schedule in schedules:
                rows.append(case(
                    seed, "overlap", "dispatch+combine", workers,
                    schedule, target_bytes=target, dtype="bf16+fp32",
                ))
    return rows


def generate(seed: int) -> dict[str, Any]:
    cases = (
        boundary_cases(seed)
        + workload_cases(seed)
        + performance_cases(seed)
        + ordering_cases(seed)
        + fault_cases(seed)
        + overlap_cases(seed)
    )
    ids = [row["case_id"] for row in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate case_id in Pull V2 acceptance matrix")
    return {
        "schema": "single-inc-pull-v2-acceptance-plan.v1",
        "campaign_seed": seed,
        "dispatch_gate_GBps": DISPATCH_GATE,
        "dispatch_gate_scope": (
            "complete dispatch; sym_dense; exactly 128 MiB source hidden "
            "payload per worker"
        ),
        "combine_regression_ratio_min": 0.98,
        "performance_gate_min_samples": 10,
        "performance_cv_max": 0.05,
        "case_count": len(cases),
        "cases": cases,
        "deferred_suites": {
            "random": "1000 mixed-shape waves per W",
            "soak": "at least 10000 waves and at least one hour",
            "logical_train": [4 * GIB, 8 * GIB],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    manifest = generate(args.seed)
    rendered = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=None if args.compact else 2,
        separators=(",", ":") if args.compact else None,
    ) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite existing plan: {args.output}")
        args.output.write_text(rendered, encoding="utf-8")
        print(f"PLAN_WRITTEN cases={manifest['case_count']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
