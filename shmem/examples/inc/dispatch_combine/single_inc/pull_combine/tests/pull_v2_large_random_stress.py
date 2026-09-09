#!/usr/bin/env python3
"""128-MiB randomized routing stress for unified Pull V2 Dispatch/Combine."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time


MIB = 1 << 20


def endpoint():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{sock.getsockname()[1]}"


def assert_plane_idle(first_npu: int, workers: int) -> None:
    first = first_npu // 8 * 8
    for device in range(first, first + 8):
        result = subprocess.run(
            ["npu-smi", "info", "-t", "proc-mem", "-i", str(device), "-c", "0"],
            capture_output=True, text=True, check=True)
        if "No process in device." not in result.stdout:
            raise RuntimeError(f"NPU {device} is not idle")


def start_group(binary: Path, commands: list[list[str]], output: Path,
                build: Path, label: str) -> tuple[list[subprocess.Popen[str]], list[object]]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{build / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    env["SHMEM_UID_SESSION_ID"] = f"large-random-{label}-{os.getpid()}-{time.monotonic_ns()}"
    procs, logs = [], []
    for pe, command in enumerate(commands):
        log = (output / f"pe{pe}.log").open("w", encoding="utf-8")
        logs.append(log)
        procs.append(subprocess.Popen([str(binary), *command], env=env,
                                      stdout=log, stderr=subprocess.STDOUT,
                                      text=True))
    return procs, logs


def finish(procs: list[subprocess.Popen[str]], logs: list[object], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    try:
        for proc in procs:
            proc.wait(timeout=max(0.1, deadline - time.monotonic()))
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15)
        for log in logs:
            log.close()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(errors="replace").splitlines()
            if line.startswith("{")]


def dispatch_case(args, ordinal: int, workers: int, seed: int,
                  token_skew: int, ready_skew: int) -> dict:
    name = f"d_w{workers}_s{seed}_ts{token_skew}_rs{ready_skew}"
    out = args.output / name
    out.mkdir()
    assert_plane_idle(args.dispatch_first_npu, workers)
    ep = endpoint()
    channels = 5 if workers == 2 else 3
    command = [[str(workers), str(pe), ep, str(args.dispatch_first_npu),
                str(128 * MIB), "ragged", "8192", "64", str(channels),
                "3", "10", str(seed), "0", str(token_skew), str(ready_skew)]
               for pe in range(workers + 1)]
    procs, logs = start_group(args.build_dir / "bin/inc_dc_pull_dispatch_v2_device_e2e",
                              command, out, args.build_dir, name)
    finish(procs, logs, args.timeout)
    if any(proc.returncode for proc in procs):
        raise RuntimeError(f"{name}: exits {[p.returncode for p in procs]}")
    sample = rows(out / f"pe{workers}.log")
    values = [r["downlink_gb_s"] for r in sample
              if r.get("test") == "pull_dispatch_v2_device_e2e" and r.get("phase") == "measure"]
    if len(values) != 10 or not all(r.get("correct") for r in sample
                                    if r.get("test") == "pull_dispatch_v2_device_e2e"):
        raise RuntimeError(f"{name}: missing or failed samples")
    summary = next(r for r in sample if r.get("test") ==
                   "pull_dispatch_v2_device_e2e_summary")
    return {"case": name, "operator": "dispatch", "workers": workers,
            "seed": seed, "token_skew": token_skew, "ready_skew_us": ready_skew,
            "minimum_gb_s": min(values), "mean_gb_s": sum(values) / len(values),
            "cv_percent": summary["downlink_cv_percent"],
            "correct": True}


def combine_case(args, ordinal: int, workers: int, seed: int) -> dict:
    name = f"c_w{workers}_s{seed}"
    out = args.output / name
    out.mkdir()
    assert_plane_idle(args.combine_first_npu, workers)
    ep = endpoint()
    rows_per_batch = (128 * MIB // (8192 * 4)) * workers
    command = [[str(workers), str(pe), ep, str(args.combine_first_npu), "8192",
                str(rows_per_batch), "mixed_k", "3", "10", "0", str(seed)]
               for pe in range(workers + 1)]
    procs, logs = start_group(args.build_dir / "bin/inc_dc_pull_combine_v2_npu_e2e",
                              command, out, args.build_dir, name)
    finish(procs, logs, args.timeout)
    if any(proc.returncode for proc in procs):
        raise RuntimeError(f"{name}: exits {[p.returncode for p in procs]}")
    sample = rows(out / f"pe{workers}.log")
    values = [r["uplink_gb_s"] for r in sample
              if r.get("test") == "pull_combine_v2_npu_e2e" and not r.get("warmup")]
    if len(values) != 10 or not all(r.get("correct") for r in sample
                                    if r.get("test") == "pull_combine_v2_npu_e2e"):
        raise RuntimeError(f"{name}: missing or failed samples")
    summary = next(r for r in sample if r.get("test") ==
                   "pull_combine_v2_npu_e2e_summary")
    return {"case": name, "operator": "combine", "workers": workers,
            "seed": seed, "minimum_gb_s": min(values),
            "mean_gb_s": sum(values) / len(values),
            "cv_percent": summary["uplink_cv_pct"], "correct": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dispatch-first-npu", type=int, default=0)
    parser.add_argument("--combine-first-npu", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.dispatch_first_npu // 8 == args.combine_first_npu // 8:
        raise ValueError("Dispatch and Combine stress planes must differ")
    args.output.mkdir(parents=True)
    results = []
    dispatch_seeds = (17, 20260910, 987654321)
    for workers in (2, 4):
        for seed in dispatch_seeds:
            for token_skew, ready_skew in ((0, 0), (50, 500), (100, 1000)):
                results.append(dispatch_case(args, len(results), workers, seed,
                                             token_skew, ready_skew))
    for workers in (2, 4):
        for seed in (17, 20260910, 987654321, 0x5EEDBEEF):
            results.append(combine_case(args, len(results), workers, seed))
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"cases": len(results), "all_correct": all(r["correct"] for r in results),
                      "dispatch_min": min(r["minimum_gb_s"] for r in results if r["operator"] == "dispatch"),
                      "combine_min": min(r["minimum_gb_s"] for r in results if r["operator"] == "combine")}, indent=2))


if __name__ == "__main__":
    main()
