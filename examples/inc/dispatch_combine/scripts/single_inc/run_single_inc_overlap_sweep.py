#!/usr/bin/env python3
"""Run auditable single-INC Dispatch/Combine overlap cases.

The two operators use independent SHMEM sessions but share one physical NPU
placement.  Separate external gates let this runner qualify simultaneous and
staggered submission without changing either production data path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def monotonic_ns() -> int:
    return time.monotonic_ns()


def npu_idle() -> bool:
    result = subprocess.run(
        ["npu-smi", "info"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    count = subprocess.run(
        ["npu-smi", "info", "-l"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, check=False,
    )
    match = re.search(r"Total Count\s*:\s*(\d+)", count.stdout)
    return bool(
        result.returncode == 0 and match and
        result.stdout.count("No running processes found in NPU") == int(match.group(1))
    )


def wait_idle(timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not npu_idle():
        if time.monotonic() >= deadline:
            raise RuntimeError("NPU_NOT_IDLE")
        time.sleep(0.5)


def write_gate(gate: Path, start_ns: int) -> None:
    (gate / "start_ns").write_text(f"{start_ns}\n", encoding="utf-8")
    (gate / "go").write_text("go\n", encoding="utf-8")


def wait_ready(gate: Path, role: str, npes: int, procs: list[subprocess.Popen[str]],
               timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        if all((gate / f"{role}_{pe}.ready").is_file() for pe in range(npes)):
            return
        failed = [proc.returncode for proc in procs if proc.poll() is not None]
        if failed:
            raise RuntimeError(f"{role.upper()}_EXIT_BEFORE_GATE rc={failed}")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{role.upper()}_GATE_TIMEOUT")
        time.sleep(0.01)


def terminate(procs: list[subprocess.Popen[str]]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and any(proc.poll() is None for proc in procs):
        time.sleep(0.05)
    for proc in procs:
        if proc.poll() is None:
            proc.kill()


def parse_case(case_dir: Path, workers: int, run_dispatch: bool,
               run_combine: bool) -> dict[str, Any]:
    expected = workers + 1
    result: dict[str, Any] = {}
    if run_dispatch:
        text = "\n".join(
            path.read_text(errors="replace")
            for path in sorted((case_dir / "dispatch").glob("pe*.log"))
        )
        passed = len(re.findall(r"STREAM_DISPATCH_RESULT .* pass=1", text))
        values = [float(x) for x in re.findall(
            r"STREAM_DISPATCH_TIMING .*?rank_us=([0-9.]+)", text)]
        byte_values = [int(x) for x in re.findall(
            r"STREAM_DISPATCH_TIMING .*?physical_dispatch_bytes=(\d+)", text)]
        result["dispatch"] = {
            "passed_ranks": passed,
            "expected_ranks": expected,
            "correct": passed == expected,
            "rank_us_max": max(values) if values else None,
            "physical_bytes": max(byte_values) if byte_values else None,
        }
    if run_combine:
        text = "\n".join(
            path.read_text(errors="replace")
            for path in sorted((case_dir / "combine").glob("pe*.log"))
        )
        passed = len(re.findall(r"DYNCSR_RESULT SUCCESS", text))
        values = [float(x) for x in re.findall(
            r"DYNCSR_TIMING .*?rank_us=([0-9.]+)", text)]
        byte_values = [int(x) for x in re.findall(
            r"DYNCSR_TIMING .*?physical_ingress_bytes=(\d+)", text)]
        result["combine"] = {
            "passed_ranks": passed,
            "expected_ranks": expected,
            "correct": passed == expected,
            "rank_us_max": max(values) if values else None,
            "physical_bytes": max(byte_values) if byte_values else None,
        }
    return result


def launch_case(args: argparse.Namespace, workers: int, tag: str,
                dispatch_offset_us: int | None,
                combine_offset_us: int | None) -> dict[str, Any]:
    run_dispatch = dispatch_offset_us is not None
    run_combine = combine_offset_us is not None
    npes = workers + 1
    case_dir = args.output_dir / f"W{workers}" / tag
    ddir, cdir = case_dir / "dispatch", case_dir / "combine"
    dgate, cgate = case_dir / "dispatch_gate", case_dir / "combine_gate"
    for directory in (ddir, cdir, dgate, cgate):
        directory.mkdir(parents=True, exist_ok=True)

    worker_phys = args.worker_phys[workers]
    pe_map = ",".join(
        [f"{rank}:{phy}" for rank, phy in enumerate(worker_phys)] +
        [f"{workers}:{args.inc_phy}"]
    )
    common = os.environ.copy()
    common["INC_PE_TO_NPU_MAP"] = pe_map
    common["LD_LIBRARY_PATH"] = f"{args.build_dir / 'lib'}:{common.get('LD_LIBRARY_PATH', '')}"
    # Large correctness shapes can spend well over 30 seconds constructing
    # and publishing the peer operator's host/device oracle before both
    # independent sessions reach the measurement gate.  Couple the external
    # gate timeout to the case timeout so an already-ready operator does not
    # fail merely because its peer has a larger setup phase.  This remains
    # outside the measured operator interval.
    common["INC_DC_EXTERNAL_START_GATE_TIMEOUT_MS"] = str(
        max(30_000, int(args.timeout * 1000))
    )

    # Keep Dispatch and Combine at the same caller-selected physical byte
    # count.  K never exceeds workers.
    topk = workers
    hidden_bytes = 8192
    dispatch_denominator = workers * topk * hidden_bytes
    if args.physical_bytes % dispatch_denominator != 0:
        raise ValueError("physical bytes are not divisible by Dispatch shape")
    dispatch_tokens = args.physical_bytes // dispatch_denominator
    combine_hidden = 4096
    combine_denominator = topk * combine_hidden * 2
    if args.physical_bytes % combine_denominator != 0:
        raise ValueError("physical bytes are not divisible by Combine shape")
    combine_results = args.physical_bytes // combine_denominator
    procs: list[subprocess.Popen[str]] = []
    logs: list[Any] = []
    dispatch_ready = False
    started = time.time()
    try:
        if run_dispatch:
            env = common.copy()
            env.update({
                "SHMEM_UID_SESSION_ID": f"nb-overlap-d-{os.getpid()}-{tag}",
                "INC_DC_EXTERNAL_START_GATE_DIR": str(dgate),
                "INC_STREAM_ADAPTIVE_TILE": "1",
                "INC_STREAM_TX_PINGPONG": "1",
                "INC_STREAM_DIRECT_DCCI": "0",
            })
            endpoint = f"tcp://127.0.0.1:{port()}"
            for pe in range(npes):
                log = (ddir / f"pe{pe}.log").open("w", encoding="utf-8")
                logs.append(log)
                cmd = [str(args.dispatch_bin), str(npes), str(pe), endpoint,
                       "16", "0", str(dispatch_tokens), str(hidden_bytes),
                       str(topk), "262144", "1048576", "0", "1", "0", "0"]
                procs.append(subprocess.Popen(cmd, env=env, text=True,
                                              stdout=log, stderr=subprocess.STDOUT))
        d_count = len(procs)
        if run_dispatch and run_combine:
            # ACL/HYBM process initialization is not reliably re-entrant when
            # two independent sessions create contexts on the same physical
            # NPU at exactly the same time.  Production servers initialize
            # their long-lived sessions once as well.  Serialize setup only;
            # the external device deadlines below still launch both operators
            # concurrently (or at the requested arbitrary offset).
            wait_ready(dgate, "dispatch", npes, procs[:d_count], args.timeout)
            dispatch_ready = True
        if run_combine:
            env = common.copy()
            env.update({
                "SHMEM_UID_SESSION_ID": f"nb-overlap-c-{os.getpid()}-{tag}",
                "INC_DC_EXTERNAL_START_GATE_DIR": str(cgate),
                "INC_DYNCSR_DEVICE_PRODUCER": "1",
                "INC_DYNCSR_OVERLAP": "1",
                "INC_DYNCSR_BATCHED_READY": "1",
                "INC_DYNCSR_READY_SCOPE": "auto",
                "INC_DYNCSR_DEVICE_COMPLETION": "0",
                "INC_DYNCSR_PRODUCER_LANES": "0",
                "INC_DYNCSR_PRODUCER_WINDOW": "32",
                "INC_DYNCSR_COALESCED_GROUP_PUT": "1",
                "INC_DYNCSR_REMOTE_TX": "1",
                "INC_DYNCSR_TX_WINDOW": "0",
                "INC_DYNCSR_WIDE_VECTOR_TILE": "1",
                "INC_DYNCSR_PERSISTENT_LOCAL_TRIGGER": "1",
                "INC_DYNCSR_SERVICE_EPOCHS": "1",
                "INC_DYNCSR_SERVICE_WARMUP_EPOCHS": "0",
            })
            endpoint = f"tcp://127.0.0.1:{port()}"
            for pe in range(npes):
                log = (cdir / f"pe{pe}.log").open("w", encoding="utf-8")
                logs.append(log)
                cmd = [str(args.combine_bin), str(npes), str(pe), endpoint,
                       "16", "0", str(workers), "1", str(topk),
                       str(combine_results), str(combine_hidden), "0"]
                procs.append(subprocess.Popen(cmd, env=env, text=True,
                                              stdout=log, stderr=subprocess.STDOUT))

        if run_dispatch and not dispatch_ready:
            wait_ready(dgate, "dispatch", npes, procs[:d_count], args.timeout)
        if run_combine:
            wait_ready(cgate, "combine", npes, procs[d_count:], args.timeout)
        base_ns = monotonic_ns() + 2_000_000_000
        if run_dispatch:
            write_gate(dgate, base_ns + int(dispatch_offset_us) * 1000)
        if run_combine:
            write_gate(cgate, base_ns + int(combine_offset_us) * 1000)

        deadline = time.monotonic() + args.timeout
        for proc in procs:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(proc.args, args.timeout)
            proc.wait(timeout=remaining)
        return_codes = [proc.returncode for proc in procs]
        if any(code != 0 for code in return_codes):
            raise RuntimeError(f"NONZERO_RANKS={return_codes}")
        parsed = parse_case(case_dir, workers, run_dispatch, run_combine)
        correct = all(value["correct"] for value in parsed.values())
        if not correct:
            raise RuntimeError(f"CORRECTNESS_FAIL={parsed}")
        parsed.update({
            "tag": tag,
            "workers": workers,
            "topk": topk,
            "dispatch_offset_us": dispatch_offset_us,
            "combine_offset_us": combine_offset_us,
            "pe_map": pe_map,
            "correct": True,
            "wall_s": time.time() - started,
            "log_dir": str(case_dir),
        })
        if run_dispatch and run_combine:
            du = float(parsed["dispatch"]["rank_us_max"])
            cu = float(parsed["combine"]["rank_us_max"])
            ds, cs = float(dispatch_offset_us), float(combine_offset_us)
            parsed["concurrent_makespan_us"] = max(ds + du, cs + cu) - min(ds, cs)
            parsed["overlap_us"] = max(0.0, min(ds + du, cs + cu) - max(ds, cs))
            parsed["true_overlap"] = parsed["overlap_us"] > 0.0
        (case_dir / "case.json").write_text(
            json.dumps(parsed, indent=2) + "\n", encoding="utf-8")
        return parsed
    except Exception as exc:
        terminate(procs)
        failure = {"tag": tag, "workers": workers, "correct": False,
                   "error": str(exc), "log_dir": str(case_dir)}
        (case_dir / "case.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        for log in logs:
            log.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", default="2,4")
    parser.add_argument("--physical-bytes", type=int, default=128 << 20)
    parser.add_argument("--solo-repeats", type=int, default=2)
    parser.add_argument("--sim-repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--inc-phy", type=int, default=int(os.environ.get("INC_SINGLE_INC_PHY", "0")))
    parser.add_argument(
        "--duplex-roofline", action="append", default=[],
        metavar="W:INGRESS_GBPS:EGRESS_GBPS",
        help="measured per-direction duplex roofline; repeat once per scale",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.build_dir = args.build_dir.resolve()
    if args.physical_bytes <= 0 or args.solo_repeats <= 0 or args.sim_repeats <= 0:
        parser.error("physical bytes and repeat counts must be positive")
    args.dispatch_bin = args.build_dir / "bin/inc_dc_single_inc_stream"
    args.combine_bin = args.build_dir / "bin/inc_dc_sv2_dyn_csr_combine"
    workers = [int(value) for value in args.workers.split(",")]
    duplex_roofline_specs = args.duplex_roofline
    args.duplex_roofline = {}
    for raw in duplex_roofline_specs:
        try:
            worker_raw, ingress_raw, egress_raw = raw.split(":")
            worker = int(worker_raw)
            ingress = float(ingress_raw)
            egress = float(egress_raw)
        except ValueError:
            parser.error(f"invalid --duplex-roofline: {raw}")
        if worker not in workers or ingress <= 0.0 or egress <= 0.0:
            parser.error(f"invalid --duplex-roofline: {raw}")
        args.duplex_roofline[worker] = (ingress, egress)
    args.worker_phys = {}
    for worker in workers:
        raw = os.environ.get(f"INC_SINGLE_INC_WORKER_PHYS_W{worker}", "")
        phys = [int(value) for value in raw.split()]
        if len(phys) != worker:
            parser.error(f"INC_SINGLE_INC_WORKER_PHYS_W{worker} is required")
        args.worker_phys[worker] = phys
    if not args.dispatch_bin.is_file() or not args.combine_bin.is_file():
        parser.error("missing freshly built operator binaries")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stop_procs: list[subprocess.Popen[str]] = []
    def interrupted(_signum: int, _frame: Any) -> None:
        terminate(stop_procs)
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    wait_idle()
    all_cases: list[dict[str, Any]] = []
    for worker in workers:
        for repeat in range(args.solo_repeats):
            all_cases.append(launch_case(args, worker, f"dispatch_solo_r{repeat}", 0, None))
            wait_idle()
            all_cases.append(launch_case(args, worker, f"combine_solo_r{repeat}", None, 0))
            wait_idle()
        for repeat in range(args.sim_repeats):
            all_cases.append(launch_case(args, worker, f"simultaneous_r{repeat}", 0, 0))
            wait_idle()
        all_cases.append(launch_case(args, worker, "dispatch_first_500us", 0, 500))
        wait_idle()
        all_cases.append(launch_case(args, worker, "combine_first_500us", 500, 0))
        wait_idle()

    summaries: list[dict[str, Any]] = []
    for worker in workers:
        cases = [case for case in all_cases if case["workers"] == worker]
        dsolo = statistics.median(float(case["dispatch"]["rank_us_max"])
                                  for case in cases if case["tag"].startswith("dispatch_solo"))
        csolo = statistics.median(float(case["combine"]["rank_us_max"])
                                  for case in cases if case["tag"].startswith("combine_solo"))
        concurrent = [case for case in cases if "concurrent_makespan_us" in case]
        for case in concurrent:
            dispatch_offset = float(case["dispatch_offset_us"])
            combine_offset = float(case["combine_offset_us"])
            theoretical_makespan = (
                max(dispatch_offset + dsolo, combine_offset + csolo) -
                min(dispatch_offset, combine_offset)
            )
            case["sequential_solo_us"] = dsolo + csolo
            # This is only the resource-free scheduling ceiling.  It must not
            # be reported as a topology/bandwidth theory; that requires a
            # measured per-direction duplex roofline for the active profile.
            case["resource_free_makespan_us"] = theoretical_makespan
            case["resource_free_ratio"] = theoretical_makespan / (dsolo + csolo)
            case["resource_free_gain_ceiling"] = 1.0 - case["resource_free_ratio"]
            case["resource_free_speedup_ceiling"] = (dsolo + csolo) / theoretical_makespan
            case["ratio"] = case["concurrent_makespan_us"] / (dsolo + csolo)
            case["speedup"] = (dsolo + csolo) / case["concurrent_makespan_us"]
            case["actual_gain"] = 1.0 - case["ratio"]
            case["resource_free_gain_realization"] = (
                case["actual_gain"] / case["resource_free_gain_ceiling"]
                if case["resource_free_gain_ceiling"] > 0.0 else None
            )
        simultaneous = [
            case for case in concurrent if case["tag"].startswith("simultaneous")
        ]
        simultaneous_ratio = statistics.median(case["ratio"] for case in simultaneous)
        simultaneous_theoretical_ratio = max(dsolo, csolo) / (dsolo + csolo)
        summary = {
            "workers": worker,
            "dispatch_solo_median_us": dsolo,
            "combine_solo_median_us": csolo,
            "concurrent_cases": len(concurrent),
            "correct_cases": sum(bool(case.get("correct")) for case in cases),
            "total_cases": len(cases),
            "true_overlap_cases": sum(bool(case.get("true_overlap")) for case in concurrent),
            "simultaneous_resource_free_ratio": simultaneous_theoretical_ratio,
            "simultaneous_resource_free_gain_ceiling": 1.0 - simultaneous_theoretical_ratio,
            "simultaneous_resource_free_speedup_ceiling": 1.0 / simultaneous_theoretical_ratio,
            "simultaneous_ratio_median": simultaneous_ratio,
            "simultaneous_actual_gain_median": 1.0 - simultaneous_ratio,
            "simultaneous_speedup_median": statistics.median(
                case["speedup"] for case in simultaneous),
            "simultaneous_resource_free_gain_realization": (
                (1.0 - simultaneous_ratio) /
                (1.0 - simultaneous_theoretical_ratio)
            ),
        }
        if worker in args.duplex_roofline:
            ingress_gbps, egress_gbps = args.duplex_roofline[worker]
            dispatch_bytes = max(
                int(case["dispatch"]["physical_bytes"])
                for case in cases if case["tag"].startswith("dispatch_solo")
            )
            combine_bytes = max(
                int(case["combine"]["physical_bytes"])
                for case in cases if case["tag"].startswith("combine_solo")
            )
            # This qualification shape uses balanced R=K=W routing.
            down_bytes = dispatch_bytes + combine_bytes / worker
            up_bytes = dispatch_bytes / worker + combine_bytes
            down_us = down_bytes / (egress_gbps * 1e9) * 1e6
            up_us = up_bytes / (ingress_gbps * 1e9) * 1e6
            topology_us = max(dsolo, csolo, down_us, up_us)
            topology_ratio = topology_us / (dsolo + csolo)
            topology_gain = 1.0 - topology_ratio
            for case in concurrent:
                # Staggered cases must satisfy both the total-duplex-traffic
                # lower bound and their resource-free release-time bound.
                case_topology_us = max(
                    topology_us, float(case["resource_free_makespan_us"])
                )
                case_topology_ratio = case_topology_us / (dsolo + csolo)
                case_topology_gain = 1.0 - case_topology_ratio
                case.update({
                    "topology_theoretical_makespan_us": case_topology_us,
                    "topology_theoretical_ratio": case_topology_ratio,
                    "topology_theoretical_gain": case_topology_gain,
                    "topology_theoretical_speedup": 1.0 / case_topology_ratio,
                    "topology_gain_realization": (
                        case["actual_gain"] / case_topology_gain
                        if case_topology_gain > 0.0 else None
                    ),
                })
            summary.update({
                "topology_duplex_ingress_gbps": ingress_gbps,
                "topology_duplex_egress_gbps": egress_gbps,
                "topology_down_bytes": int(down_bytes),
                "topology_up_bytes": int(up_bytes),
                "topology_theoretical_makespan_us": topology_us,
                "topology_theoretical_ratio": topology_ratio,
                "topology_theoretical_gain": topology_gain,
                "topology_theoretical_speedup": 1.0 / topology_ratio,
                "topology_gain_realization": (
                    (1.0 - simultaneous_ratio) / topology_gain
                    if topology_gain > 0.0 else None
                ),
            })
        summaries.append(summary)
    report = {
        "schema": "single_inc_overlap_nb.v1",
        "environment": "nb-borrow/910B2C/CANN-9.1.0-beta.3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "build_dir": str(args.build_dir),
        "physical_bytes_per_operator": args.physical_bytes,
        "cases": all_cases,
        "summaries": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
