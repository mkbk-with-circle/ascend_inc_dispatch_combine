#!/usr/bin/env python3
"""Run paired W4/128-MiB fixed expert-top-k4/top-k8 cases."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "pull_v2_overlap_helpers", HERE / "pull_v2_overlap_qualification.py")
HELPERS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = HELPERS
SPEC.loader.exec_module(HELPERS)


def wait_iteration(gate: Path, role: str, iteration: int, workers: int,
                   procs: list[subprocess.Popen[str]], timeout: float) -> None:
    expected = [gate / f"{role}_iter{iteration}_pe{pe}.ready"
                for pe in range(workers + 1)]
    deadline = time.monotonic() + timeout
    while not all(path.is_file() for path in expected):
        if any(proc.poll() is not None for proc in procs):
            raise RuntimeError(
                f"{role} exited before iteration {iteration} gate")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{role} iteration {iteration} gate timeout")
        time.sleep(0.01)
    stem = gate / f"{role}_iter{iteration}"
    start_ns = time.monotonic_ns() + 200_000_000
    Path(f"{stem}.start_ns").write_text(f"{start_ns}\n", encoding="utf-8")
    Path(f"{stem}.go").write_text("go\n", encoding="utf-8")


def summary(path: Path, expected_test: str) -> dict:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("test") == expected_test:
            rows.append(row)
    if len(rows) != 1 or rows[0].get("correct") is not True:
        raise RuntimeError(f"missing or failed summary in {path}: {rows}")
    return rows[0]


def selected_devices_idle(first_npu: int, count: int) -> bool:
    for device in range(first_npu, first_npu + count):
        proc = subprocess.run(
            ["npu-smi", "info", "-t", "proc-mem", "-i", str(device),
             "-c", "0"], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        if proc.returncode != 0 or "No process in device." not in proc.stdout:
            return False
    return True


def wait_selected_devices_idle(first_npu: int, count: int,
                               timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not selected_devices_idle(first_npu, count):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"NPU {first_npu}--{first_npu + count - 1} not idle")
        time.sleep(0.1)


def run_group(args: argparse.Namespace, role: str, workload: str,
              combine_rows: int, output: Path) -> dict:
    case = output / f"{role}_{workload}"
    gate = case / "gate"
    endpoint = f"tcp://127.0.0.1:{HELPERS.free_port()}"
    if role == "dispatch":
        commands = [[
            "4", str(pe), endpoint, str(args.first_npu),
            str(args.payload_bytes), workload, str(args.hidden),
            str(args.expert_count), str(args.channels), "3", "10",
            str(args.seed), "0",
        ] for pe in range(5)]
        binary = args.build_dir / "bin/inc_dc_pull_dispatch_v2_device_e2e"
        test_name = "pull_dispatch_v2_device_e2e_summary"
    else:
        commands = [[
            "4", str(pe), endpoint, str(args.first_npu), str(args.hidden),
            str(combine_rows), workload, "3", "10",
        ] for pe in range(5)]
        binary = args.build_dir / "bin/inc_dc_pull_combine_v2_npu_e2e"
        test_name = "pull_combine_v2_npu_e2e_summary"

    procs: list[subprocess.Popen[str]] = []
    logs = []
    try:
        procs, logs = HELPERS.start_group(
            binary, commands, gate, role, case / "logs", args.build_dir,
            args.timeout)
        for iteration in range(13):
            wait_iteration(gate, role, iteration, 4, procs, args.timeout)
        deadline = time.monotonic() + args.timeout
        for proc in procs:
            proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        codes = [proc.returncode for proc in procs]
        if any(code != 0 for code in codes):
            raise RuntimeError(f"{role}/{workload} rank exits: {codes}")
        for pe in range(5):
            text = (case / "logs" / f"pe{pe}.log").read_text(
                errors="replace")
            if role == "dispatch" and "[PASS]" not in text:
                raise RuntimeError(f"{role}/{workload} PE{pe} lacks PASS")
            if role == "combine" and text.count("kernel_status=0") != 13:
                raise RuntimeError(
                    f"{role}/{workload} PE{pe} lacks 13 successful waves")
        row = summary(case / "logs/pe4.log", test_name)
        minimum = float(row.get("protocol_min_gb_s",
                                row.get("min_logical_gb_s", 0.0)))
        mean = float(row.get("protocol_mean_gb_s",
                             row.get("mean_logical_gb_s", 0.0)))
        cv = float(row.get("protocol_cv_percent",
                           row.get("cv_pct", 100.0)))
        expert_topk = 8 if workload == "sym_k8_gpu4" else 4
        gpu_topk = 2 if workload == "sym_k4_gpu2" else 4
        return {
            "operator": role,
            "workload": workload,
            "expert_topk": expert_topk,
            "gpu_topk": gpu_topk,
            "partial_topk": None if role == "dispatch" else
                gpu_topk,
            "payload_bytes_per_worker": args.payload_bytes,
            "warmup": 3,
            "measure": 10,
            "min_gb_s": minimum,
            "mean_gb_s": mean,
            "cv_pct": cv,
            "reference_gate_gb_s": 103.04,
            "reference_gate_pass": minimum >= 103.04 and cv <= 5.0,
            "correct": True,
        }
    finally:
        HELPERS.terminate(procs)
        for log in logs:
            log.close()
        wait_selected_devices_idle(args.first_npu, 5, args.timeout)
        time.sleep(0.25)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-npu", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--expert-count", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--payload-bytes", type=int, default=128 << 20)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--variants", default="sym_k4_gpu4,sym_k4_gpu2,sym_k8_gpu4")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    wait_selected_devices_idle(args.first_npu, 5, args.timeout)
    row_bytes = args.hidden * 4
    if args.payload_bytes % row_bytes:
        raise ValueError("payload must be divisible by FP32 row bytes")
    rows_per_b = args.payload_bytes // row_bytes
    catalog = {
        "sym_k4_gpu4": rows_per_b,
        "sym_k4_gpu2": rows_per_b * 2,
        "sym_k8_gpu4": rows_per_b,
    }
    selected = args.variants.split(",")
    if not selected or any(name not in catalog for name in selected) or \
            len(set(selected)) != len(selected):
        raise ValueError(f"invalid --variants: {args.variants}")
    variants = [(name, catalog[name]) for name in selected]
    results = []
    for workload, combine_rows in variants:
        results.append(run_group(args, "dispatch", workload, combine_rows,
                                 args.output))
        results.append(run_group(args, "combine", workload, combine_rows,
                                 args.output))
    lines = [
        "# Pull V2 W4 fixed expert-top-k4/top-k8 formal run",
        "",
        "W4+1INC on NPU0--4; H=8192; 128 MiB/worker; "
        "3 warmup + 10 measure.", "",
        "| Operator | Route | Expert K | GPU K | Partial K | Min GB/s | "
        "Mean GB/s | CV | vs 103.04 reference |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        partial = "-" if row["partial_topk"] is None else row["partial_topk"]
        lines.append(
            f'| {row["operator"]} | {row["workload"]} | '
            f'{row["expert_topk"]} | '
            f'{row["gpu_topk"]} | {partial} | {row["min_gb_s"]:.3f} | '
            f'{row["mean_gb_s"]:.3f} | {row["cv_pct"]:.3f}% | '
            f'{"PASS" if row["reference_gate_pass"] else "FAIL"} |')
    lines += ["", "103.04 GB/s is the existing W4 reference gate; the "
              "repository previously defined its hard-gate scope only for "
              "sym_k2_balanced. This report preserves that distinction."]
    (args.output / "README.md").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
