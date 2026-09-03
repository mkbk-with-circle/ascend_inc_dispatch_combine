#!/usr/bin/env python3
"""Qualify Pull-Dispatch V2 + Pull-Combine V2 on one physical INC.

Both operators keep independent SHMEM sessions and strong identities, but map
onto the same worker/INC NPUs. Their already-qualified E2E harnesses stop at a
host-only launch gate after allocation and oracle construction. This runner
then releases both half-of-live-AIV kernels at controlled clock offsets and
uses the INC device-cycle timelines to prove actual overlap.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SYSTEM_CYCLE_US = 0.02  # 50 MHz on the qualified 910B2C host.
MIB = 1 << 20


@dataclass(frozen=True)
class Schedule:
    name: str
    dispatch_offset_us: int
    combine_offset_us: int
    rank_jitter_us: int = 0


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def detected_device_count() -> int:
    count = subprocess.run(
        ["npu-smi", "info", "-l"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, check=False,
    )
    match = re.search(r"Total Count\s*:\s*(\d+)", count.stdout)
    if count.returncode != 0 or match is None:
        raise RuntimeError("cannot detect NPU count")
    return int(match.group(1))


def all_npus_idle(device_count: int) -> bool:
    info = subprocess.run(
        ["npu-smi", "info"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    return bool(
        device_count > 0 and info.returncode == 0 and
        info.stdout.count("No running processes found in NPU") ==
        device_count
    )


def wait_all_npus_idle(device_count: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not all_npus_idle(device_count):
        if time.monotonic() >= deadline:
            raise TimeoutError("NPU teardown did not reach all-idle state")
        time.sleep(0.1)


def placement_is_one_plane(first_npu: int, workers: int,
                           device_count: int, plane_size: int) -> bool:
    last_npu = first_npu + workers
    return bool(
        first_npu >= 0 and workers >= 1 and device_count > 0 and
        plane_size > 0 and last_npu < device_count and
        first_npu // plane_size == last_npu // plane_size
    )


def timeline_metrics(dispatch: dict[str, Any], combine: dict[str, Any],
                     solo_dispatch_cycles: int,
                     solo_combine_cycles: int) -> dict[str, float | bool]:
    ds = int(dispatch["cycle_kernel_start"])
    de = int(dispatch["cycle_kernel_done"])
    cs = int(combine["cycle_kernel_start"])
    ce = int(combine["cycle_kernel_done"])
    if not (0 < ds <= de and 0 < cs <= ce):
        raise ValueError("invalid device timeline")
    if solo_dispatch_cycles <= 0 or solo_combine_cycles <= 0:
        raise ValueError("invalid solo baseline duration")
    overlap_cycles = max(0, min(de, ce) - max(ds, cs))
    concurrent_dispatch_cycles = de - ds
    concurrent_combine_cycles = ce - cs
    solo_serial_cycles = solo_dispatch_cycles + solo_combine_cycles
    concurrent_cycles = max(de, ce) - min(ds, cs)
    return {
        "solo_dispatch_duration_us": solo_dispatch_cycles * SYSTEM_CYCLE_US,
        "solo_combine_duration_us": solo_combine_cycles * SYSTEM_CYCLE_US,
        "concurrent_dispatch_duration_us":
            concurrent_dispatch_cycles * SYSTEM_CYCLE_US,
        "concurrent_combine_duration_us":
            concurrent_combine_cycles * SYSTEM_CYCLE_US,
        "dispatch_minus_combine_start_us": (ds - cs) * SYSTEM_CYCLE_US,
        "overlap_us": overlap_cycles * SYSTEM_CYCLE_US,
        "overlapped": overlap_cycles > 0,
        "overlap_ratio_of_shorter": overlap_cycles /
        max(1, min(concurrent_dispatch_cycles, concurrent_combine_cycles)),
        "solo_serial_baseline_us": solo_serial_cycles * SYSTEM_CYCLE_US,
        "concurrent_makespan_us": concurrent_cycles * SYSTEM_CYCLE_US,
        "theoretical_speedup": solo_serial_cycles /
        max(solo_dispatch_cycles, solo_combine_cycles),
        "real_speedup": solo_serial_cycles / max(1, concurrent_cycles),
        "real_time_saved_pct": 100.0 *
        (solo_serial_cycles - concurrent_cycles) /
        max(1, solo_serial_cycles),
        "overlap_geometry_speedup":
            (concurrent_dispatch_cycles + concurrent_combine_cycles) /
            max(1, concurrent_cycles),
        "dispatch_contention_inflation":
            concurrent_dispatch_cycles / solo_dispatch_cycles,
        "combine_contention_inflation":
            concurrent_combine_cycles / solo_combine_cycles,
    }


def read_iteration(path: Path, test_name: str) -> dict[str, Any]:
    found: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("test") == test_name and row.get("iteration") == 0:
            found.append(row)
    if len(found) != 1:
        raise RuntimeError(f"expected one {test_name} row in {path}")
    row = found[0]
    if row.get("status") != 0 or row.get("correct") is not True:
        raise RuntimeError(f"operator correctness failed: {row}")
    return row


def wait_ready(gate: Path, role: str, workers: int,
               procs: list[subprocess.Popen[str]], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    expected = [gate / f"{role}_iter0_pe{pe}.ready"
                for pe in range(workers + 1)]
    while not all(path.is_file() for path in expected):
        failed = [proc.returncode for proc in procs
                  if proc.poll() is not None]
        if failed:
            raise RuntimeError(f"{role} exited before gate: {failed}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{role} gate timeout")
        time.sleep(0.01)


def terminate(procs: list[subprocess.Popen[str]]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and any(
            proc.poll() is None for proc in procs):
        time.sleep(0.05)
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
    for proc in procs:
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"failed to reap overlap child pid={proc.pid}") from error


def cleanup_case(args: argparse.Namespace,
                 procs: list[subprocess.Popen[str]], logs: list[Any]) -> None:
    terminate(procs)
    for log in logs:
        log.close()
    wait_all_npus_idle(args.live_device_count, args.timeout)
    if args.inter_case_cooldown_ms > 0:
        time.sleep(args.inter_case_cooldown_ms / 1000.0)


def start_group(binary: Path, commands: list[list[str]], gate: Path,
                role: str, output: Path, build_dir: Path,
                timeout_s: float) -> tuple[list[subprocess.Popen[str]], list[Any]]:
    output.mkdir(parents=True)
    gate.mkdir(parents=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        f"{build_dir / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    )
    env["INC_DC_PULL_V2_START_GATE_DIR"] = str(gate)
    env["INC_DC_PULL_V2_START_GATE_TIMEOUT_MS"] = str(int(timeout_s * 1000))
    env["SHMEM_UID_SESSION_ID"] = (
        f"pull-v2-overlap-{role}-{os.getpid()}-{time.monotonic_ns()}"
    )
    procs: list[subprocess.Popen[str]] = []
    logs: list[Any] = []
    for pe, tail in enumerate(commands):
        log = (output / f"pe{pe}.log").open("w", encoding="utf-8")
        logs.append(log)
        procs.append(subprocess.Popen(
            [str(binary), *tail], env=env, text=True,
            stdout=log, stderr=subprocess.STDOUT,
        ))
    return procs, logs


def operator_commands(args: argparse.Namespace, role: str, endpoint: str,
                      ordinal: int) -> list[list[str]]:
    if role == "dispatch":
        tail = lambda pe: [
            str(args.workers), str(pe), endpoint, str(args.first_npu),
            str(args.payload_bytes), "sym_k2_balanced", str(args.hidden),
            str(args.expert_count), str(args.channels), "0", "1",
            str(args.seed + ordinal), "0",
        ]
    elif role == "combine":
        row_bytes = args.hidden * 4
        if args.payload_bytes % row_bytes != 0:
            raise ValueError(
                "payload bytes must be divisible by FP32 row bytes")
        rows_per_b = args.payload_bytes // row_bytes
        if rows_per_b * args.workers % 2 != 0:
            raise ValueError("top-k=2 total A-token count is not integral")
        combine_rows = rows_per_b * args.workers // 2
        tail = lambda pe: [
            str(args.workers), str(pe), endpoint, str(args.first_npu),
            str(args.hidden), str(combine_rows), "sym_k2_balanced", "0", "1",
        ]
    else:
        raise ValueError(f"unknown operator role: {role}")
    return [tail(pe) for pe in range(args.workers + 1)]


def write_release(gate: Path, role: str, starts_ns: list[int]) -> None:
    stem = gate / f"{role}_iter0"
    if len(set(starts_ns)) == 1:
        Path(f"{stem}.start_ns").write_text(
            f"{starts_ns[0]}\n", encoding="utf-8")
    else:
        for pe, value in enumerate(starts_ns):
            Path(f"{stem}_pe{pe}.start_ns").write_text(
                f"{value}\n", encoding="utf-8")
    Path(f"{stem}.go").write_text("go\n", encoding="utf-8")


def run_solo_baseline(args: argparse.Namespace, role: str,
                      ordinal: int) -> dict[str, Any]:
    case = args.output_dir / f"baseline_{role}_solo"
    if case.exists():
        raise FileExistsError(f"refusing to overwrite {case}")
    case.mkdir(parents=True)
    gate = case / "gate"
    endpoint = f"tcp://127.0.0.1:{free_port()}"
    commands = operator_commands(args, role, endpoint, ordinal)
    binary = args.dispatch_bin if role == "dispatch" else args.combine_bin
    test_name = ("pull_dispatch_v2_device_e2e" if role == "dispatch" else
                 "pull_combine_v2_npu_e2e")
    procs: list[subprocess.Popen[str]] = []
    logs: list[Any] = []
    try:
        procs, logs = start_group(
            binary, commands, gate, role, case / role, args.build_dir,
            args.timeout)
        wait_ready(gate, role, args.workers, procs, args.timeout)
        start_ns = time.monotonic_ns() + 1_000_000_000
        write_release(gate, role, [start_ns] * (args.workers + 1))
        deadline = time.monotonic() + args.timeout
        for proc in procs:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{role} solo timeout")
            proc.wait(timeout=remaining)
        codes = [proc.returncode for proc in procs]
        if any(code != 0 for code in codes):
            raise RuntimeError(f"{role} solo nonzero ranks: {codes}")
        sample = read_iteration(
            case / role / f"pe{args.workers}.log", test_name)
        duration_cycles = (
            int(sample["cycle_kernel_done"]) -
            int(sample["cycle_kernel_start"])
        )
        if duration_cycles <= 0:
            raise RuntimeError(f"{role} solo has invalid timeline")
        result = {
            "schema": "single-inc-pull-v2-solo-baseline.v2",
            "role": role,
            "duration_cycles": duration_cycles,
            "duration_us": duration_cycles * SYSTEM_CYCLE_US,
            "sample": sample,
            "correct": True,
        }
        (case / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        return result
    finally:
        cleanup_case(args, procs, logs)


def run_case(args: argparse.Namespace, schedule: Schedule,
             ordinal: int, solo_dispatch_cycles: int,
             solo_combine_cycles: int) -> dict[str, Any]:
    case = args.output_dir / f"overlap_{ordinal:02d}_{schedule.name}"
    if case.exists():
        raise FileExistsError(f"refusing to overwrite {case}")
    case.mkdir(parents=True)
    dgate, cgate = case / "dispatch_gate", case / "combine_gate"
    endpoint_d = f"tcp://127.0.0.1:{free_port()}"
    endpoint_c = f"tcp://127.0.0.1:{free_port()}"
    dispatch_commands = operator_commands(
        args, "dispatch", endpoint_d, ordinal)
    combine_commands = operator_commands(
        args, "combine", endpoint_c, ordinal)

    procs: list[subprocess.Popen[str]] = []
    logs: list[Any] = []
    try:
        # Serialize session initialization on the shared physical devices.
        dp, dl = start_group(args.dispatch_bin, dispatch_commands, dgate,
                             "dispatch", case / "dispatch", args.build_dir,
                             args.timeout)
        procs += dp
        logs += dl
        wait_ready(dgate, "dispatch", args.workers, dp, args.timeout)
        cp, cl = start_group(args.combine_bin, combine_commands, cgate,
                             "combine", case / "combine", args.build_dir,
                             args.timeout)
        procs += cp
        logs += cl
        wait_ready(cgate, "combine", args.workers, cp, args.timeout)

        rng = random.Random(args.seed + ordinal)
        base_ns = time.monotonic_ns() + 2_000_000_000
        dstarts = []
        cstarts = []
        for _pe in range(args.workers + 1):
            dj = rng.randint(0, schedule.rank_jitter_us)
            cj = rng.randint(0, schedule.rank_jitter_us)
            dstarts.append(base_ns +
                           (schedule.dispatch_offset_us + dj) * 1000)
            cstarts.append(base_ns +
                           (schedule.combine_offset_us + cj) * 1000)
        write_release(dgate, "dispatch", dstarts)
        write_release(cgate, "combine", cstarts)

        deadline = time.monotonic() + args.timeout
        for proc in procs:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("operator timeout")
            proc.wait(timeout=remaining)
        codes = [proc.returncode for proc in procs]
        if any(code != 0 for code in codes):
            raise RuntimeError(f"nonzero ranks: {codes}")

        dispatch = read_iteration(
            case / "dispatch" / f"pe{args.workers}.log",
            "pull_dispatch_v2_device_e2e")
        combine = read_iteration(
            case / "combine" / f"pe{args.workers}.log",
            "pull_combine_v2_npu_e2e")
        metrics = timeline_metrics(
            dispatch, combine, solo_dispatch_cycles, solo_combine_cycles)
        row = {
            "schema": "single-inc-pull-v2-overlap-case.v2",
            "schedule": schedule.name,
            "workers": args.workers,
            "first_npu": args.first_npu,
            "inc_npu": args.first_npu + args.workers,
            "aiv_partition_policy": "floor(live_vector_cores/2)",
            "placement_guard": {
                "scope": "qualification-only; not a production constraint",
                "device_count": args.device_count,
                "hccs_plane_size": args.plane_size,
            },
            "inter_case_cooldown_ms": args.inter_case_cooldown_ms,
            "dispatch": dispatch,
            "combine": combine,
            "timeline": metrics,
            "correct": True,
        }
        (case / "result.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        if not metrics["overlapped"]:
            raise RuntimeError(f"device timelines did not overlap: {metrics}")
        return row
    finally:
        cleanup_case(args, procs, logs)


def schedules(args: argparse.Namespace) -> list[Schedule]:
    rows = [
        Schedule("simultaneous", 0, 0),
        Schedule(f"dispatch_lead_{args.lead_us}us", 0, args.lead_us),
        Schedule(f"combine_lead_{args.lead_us}us", args.lead_us, 0),
    ]
    for index in range(args.random_cases):
        rng = random.Random(args.seed + 1000 + index)
        delta = rng.randint(-args.lead_us, args.lead_us)
        rows.append(Schedule(
            f"random_skew_{index}_{delta:+d}us",
            max(delta, 0), max(-delta, 0), args.rank_jitter_us,
        ))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path,
                        default=Path("/tmp/shmem-pull-dispatch-v2-build"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(2, 4), required=True)
    parser.add_argument("--first-npu", type=int, required=True)
    parser.add_argument(
        "--plane-size", type=int, required=True,
        help="qualification placement guard only; not a protocol limit")
    parser.add_argument(
        "--device-count", type=int, default=0,
        help="0 detects the live count with npu-smi")
    parser.add_argument("--payload-bytes", type=int, default=128 * MIB)
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--expert-count", type=int, default=64)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--lead-us", type=int, default=500)
    parser.add_argument("--rank-jitter-us", type=int, default=100)
    parser.add_argument("--random-cases", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--inter-case-cooldown-ms", type=int, default=2000,
        help="cooldown after every fully reaped, all-idle case")
    args = parser.parse_args()
    args.dispatch_bin = args.build_dir / "bin/inc_dc_pull_dispatch_v2_device_e2e"
    args.combine_bin = args.build_dir / "bin/inc_dc_pull_combine_v2_npu_e2e"
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    if not args.dispatch_bin.is_file() or not args.combine_bin.is_file():
        raise SystemExit("Pull V2 qualification binaries are missing")
    live_device_count = detected_device_count()
    args.live_device_count = live_device_count
    if args.device_count == 0:
        args.device_count = live_device_count
    if args.device_count > live_device_count:
        raise SystemExit("configured device count exceeds live npu-smi count")
    if not placement_is_one_plane(
            args.first_npu, args.workers, args.device_count,
            args.plane_size):
        raise SystemExit(
            "qualification placement crosses the configured HCCS plane")
    # The harness queries the live vector-core count and enforces its own
    # floor(live/2) allocation. Do not duplicate a SKU-specific AIV limit.
    if args.channels <= 0:
        raise SystemExit("Dispatch channels must be positive")
    if args.inter_case_cooldown_ms < 0:
        raise SystemExit("inter-case cooldown must be non-negative")
    if not all_npus_idle(live_device_count):
        raise SystemExit("NPU_NOT_IDLE: refusing to interfere with a process")
    args.output_dir.mkdir(parents=True)
    solo_dispatch = run_solo_baseline(args, "dispatch", 1000000)
    solo_combine = run_solo_baseline(args, "combine", 2000000)
    results = [run_case(
        args, schedule, index, solo_dispatch["duration_cycles"],
        solo_combine["duration_cycles"])
               for index, schedule in enumerate(schedules(args))]
    summary = {
        "schema": "single-inc-pull-v2-overlap-summary.v2",
        "workers": args.workers,
        "aiv_partition_policy": "floor(live_vector_cores/2)",
        "placement_guard": {
            "scope": "qualification-only; not a production constraint",
            "device_count": args.device_count,
            "hccs_plane_size": args.plane_size,
        },
        "inter_case_cooldown_ms": args.inter_case_cooldown_ms,
        "solo_baselines": {
            "dispatch": solo_dispatch,
            "combine": solo_combine,
        },
        "case_count": len(results),
        "all_correct": solo_dispatch["correct"] and
            solo_combine["correct"] and
            all(row["correct"] for row in results),
        "all_overlapped": all(row["timeline"]["overlapped"]
                              for row in results),
        "cases": [{
            "schedule": row["schedule"],
            **row["timeline"],
        } for row in results],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["all_correct"] and summary["all_overlapped"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
