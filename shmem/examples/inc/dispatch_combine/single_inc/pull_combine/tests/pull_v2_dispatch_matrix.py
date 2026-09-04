#!/usr/bin/env python3
"""Run an isolated Pull-Dispatch V2 robustness matrix on one HCCS plane."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


MIB = 1 << 20
DEFAULT_SIZES = (0, 4 << 10, 64 << 10, 1 * MIB, 16 * MIB,
                 64 * MIB, 128 * MIB, 256 * MIB)
DEFAULT_ROUTES = ("sym_k1_rr", "sym_k2_balanced", "sym_dense",
                  "ragged", "hotspot")
DEFAULT_TOKEN_SKEWS = (0, 25, 50, 75, 100)
DEFAULT_READY_SKEWS = (0, 100, 1000)


def csv_values(text: str, cast: Any) -> tuple[Any, ...]:
    return tuple(cast(value.strip()) for value in text.split(",")
                 if value.strip())


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def device_count() -> int:
    result = subprocess.run(["npu-smi", "info", "-l"], text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    match = re.search(r"Total Count\s*:\s*(\d+)", result.stdout)
    if result.returncode != 0 or match is None:
        raise RuntimeError("cannot determine live NPU count")
    return int(match.group(1))


def all_idle(count: int) -> bool:
    result = subprocess.run(["npu-smi", "info"], text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    return result.returncode == 0 and result.stdout.count(
        "No running processes found in NPU") == count


def wait_idle(count: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not all_idle(count):
        if time.monotonic() >= deadline:
            raise TimeoutError("NPU teardown did not converge to all-idle")
        time.sleep(0.2)


def terminate(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and any(
            process.poll() is None for process in processes):
        time.sleep(0.05)
    for process in processes:
        if process.poll() is None:
            process.kill()
    for process in processes:
        process.wait(timeout=5.0)


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def run_case(args: argparse.Namespace, ordinal: int, workers: int,
             size: int, route: str, token_skew: int,
             ready_skew: int) -> dict[str, Any]:
    name = (f"{ordinal:04d}_w{workers}_b{size}_{route}_"
            f"ts{token_skew}_rs{ready_skew}")
    output = args.output_dir / name
    output.mkdir()
    if not all_idle(args.device_count):
        raise RuntimeError("NPU_NOT_IDLE before " + name)
    endpoint = f"tcp://127.0.0.1:{free_port()}"
    channels = args.channels_w2 if workers == 2 else args.channels_w4
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        f"{args.build_dir / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}")
    env["SHMEM_UID_SESSION_ID"] = (
        f"pdv2-matrix-{os.getpid()}-{ordinal}-{time.monotonic_ns()}")
    processes: list[subprocess.Popen[str]] = []
    logs: list[Any] = []
    started = time.monotonic()
    timed_out = False
    try:
        for pe in range(workers + 1):
            log = (output / f"pe{pe}.log").open("w", encoding="utf-8")
            logs.append(log)
            command = [str(args.binary), str(workers), str(pe), endpoint,
                       str(args.first_npu), str(size), route,
                       str(args.hidden), str(args.expert_count), str(channels),
                       str(args.warmup), str(args.measure), str(args.seed), "0",
                       str(token_skew), str(ready_skew)]
            processes.append(subprocess.Popen(
                command, env=env, text=True, stdout=log,
                stderr=subprocess.STDOUT))
        deadline = time.monotonic() + args.timeout
        for process in processes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                break
    finally:
        terminate(processes)
        for log in logs:
            log.close()
        wait_idle(args.device_count, args.timeout)
    codes = [process.returncode for process in processes]
    inc_rows = read_json_rows(output / f"pe{workers}.log")
    summaries = [row for row in inc_rows if row.get("test") ==
                 "pull_dispatch_v2_device_e2e_summary"]
    samples = [row for row in inc_rows if row.get("test") ==
               "pull_dispatch_v2_device_e2e"]
    correct = (not timed_out and all(code == 0 for code in codes) and
               len(summaries) == 1 and
               all(row.get("correct") is True for row in samples))
    result = {
        "name": name, "workers": workers, "payload_bytes": size,
        "route": route, "token_skew_percent": token_skew,
        "ready_skew_us": ready_skew, "seed": args.seed,
        "channels": channels, "timed_out": timed_out,
        "return_codes": codes, "correct": correct,
        "wall_seconds": time.monotonic() - started,
        "operator_summary": summaries[0] if len(summaries) == 1 else None,
    }
    (output / "case.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path,
                        default=Path("/tmp/shmem-pull-dispatch-v2-build"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-npu", type=int, default=0)
    parser.add_argument("--plane-size", type=int, default=8)
    parser.add_argument("--workers", default="2,4")
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--token-skews",
                        default=",".join(map(str, DEFAULT_TOKEN_SKEWS)))
    parser.add_argument("--ready-skews-us",
                        default=",".join(map(str, DEFAULT_READY_SKEWS)))
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--expert-count", type=int, default=64)
    parser.add_argument("--channels-w2", type=int, default=5)
    parser.add_argument("--channels-w4", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--measure", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()
    args.binary = args.build_dir / "bin/inc_dc_pull_dispatch_v2_device_e2e"
    args.device_count = device_count()
    workers = csv_values(args.workers, int)
    sizes = csv_values(args.sizes, int)
    routes = csv_values(args.routes, str)
    token_skews = csv_values(args.token_skews, int)
    ready_skews = csv_values(args.ready_skews_us, int)
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    if not args.binary.is_file():
        raise SystemExit(f"missing binary: {args.binary}")
    if any(value not in (2, 4) for value in workers):
        raise SystemExit("workers must contain only 2 and/or 4")
    if any(route not in DEFAULT_ROUTES for route in routes):
        raise SystemExit("unsupported route")
    if any(value not in DEFAULT_TOKEN_SKEWS for value in token_skews):
        raise SystemExit("unsupported token skew")
    if args.first_npu < 0 or any(
            args.first_npu + value >= args.device_count or
            args.first_npu // args.plane_size !=
            (args.first_npu + value) // args.plane_size for value in workers):
        raise SystemExit("placement is not within one configured HCCS plane")
    if not all_idle(args.device_count):
        raise SystemExit("NPU_NOT_IDLE: refusing to interfere")
    cases = list(itertools.product(workers, sizes, routes,
                                   token_skews, ready_skews))
    if args.max_cases > 0:
        cases = cases[:args.max_cases]
    args.output_dir.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    for ordinal, case in enumerate(cases):
        result = run_case(args, ordinal, *case)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    summary = {
        "schema": "pull-dispatch-v2-robustness-matrix.v1",
        "case_count": len(results),
        "full_cartesian_case_count": (len(workers) * len(sizes) *
                                       len(routes) * len(token_skews) *
                                       len(ready_skews)),
        "all_correct": all(row["correct"] for row in results),
        "fixed_seed": args.seed,
        "matrix": {"workers": workers, "sizes": sizes, "routes": routes,
                   "token_skews": token_skews,
                   "ready_skews_us": ready_skews},
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["all_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
