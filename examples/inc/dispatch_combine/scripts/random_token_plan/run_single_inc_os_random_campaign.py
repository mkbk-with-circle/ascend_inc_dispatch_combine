#!/usr/bin/env python3
"""Run distinct getrandom-backed token plans through production single-INC paths."""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
SINGLE_INC_SCRIPTS = Path(__file__).resolve().parents[1] / "single_inc"
DEFAULT_BUILD = ROOT / "build-cann91-fixed"
DEFAULT_CANN = Path("/opt/Ascend-9.1/cann-9.1.0-beta.1")
COMBOS = [
    (2, 1), (2, 2), (2, 4), (2, 8), (2, 16),
    (4, 1), (4, 2), (4, 4), (4, 8), (4, 16), (4, 32),
    (8, 1), (8, 2), (8, 4), (8, 8), (8, 16), (8, 32), (8, 64),
]
MIB = 1024 * 1024
PERF_GATE_BYTES = 128 * MIB
DISPATCH_ROOFLINE_GBPS = 123.339


def getrandom_below(limit: int) -> int:
    """Uniform bounded draw backed directly by Linux getrandom(2)."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    bits = (limit - 1).bit_length()
    byte_count = max(1, (bits + 7) // 8)
    mask = (1 << bits) - 1
    while True:
        value = int.from_bytes(os.getrandom(byte_count), "little") & mask
        if value < limit:
            return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_dispatch_fanout(workers: int, topk: int) -> float:
    # Experts are sampled without replacement from 8 experts/rank.
    total = workers * 8
    absent = (math.comb(total - 8, topk) / math.comb(total, topk)
              if topk <= total - 8 else 0.0)
    return workers * (1.0 - absent)


def random_shape(operator: str, size: str, workers: int, topk: int):
    # Draw the row width and target independently.  Large starts at 192 MiB so
    # random Dispatch fanout skew cannot silently turn a performance case into
    # a sub-128-MiB correctness-only case.  The generated plan is checked below
    # and rejected if it nevertheless lands in the wrong class.
    target_mib = ((4 + getrandom_below(61)) if size == "small"
                  else (192 + getrandom_below(193)))
    target = target_mib * MIB
    if operator == "dispatch":
        # 4--64 KiB, 64-byte aligned, with every case independently drawn.
        row_bytes = (64 + getrandom_below(961)) * 64
        denominator = workers * expected_dispatch_fanout(workers, topk) * row_bytes
        nominal = max(1, int(round(target / denominator)))
        # A second entropy draw makes total tokens independently variable even
        # when target/row happens to quantize to the same nominal count.
        jitter = 90 + getrandom_below(21)
        count = max(1, nominal * jitter // 100)
        return count, row_bytes, target
    # FP16 Combine hidden is in elements.  Draw a 16--256 KiB row on the
    # backend's 64-byte alignment.  Weight remains fixed at 1.0 in the plan
    # generator, keeping this a route/shape campaign rather than a weight test.
    row_bytes = (256 + getrandom_below(3841)) * 64
    nominal = max(1, int(round(target / (topk * row_bytes))))
    jitter = 90 + getrandom_below(21)
    count = max(1, nominal * jitter // 100)
    return count, row_bytes // 2, target


def valid_shape(operator: str, size: str, count: int, width: int,
                workers: int, topk: int) -> bool:
    if count <= 0 or width <= 0 or workers not in (2, 4, 8) or topk <= 0:
        return False
    row_bytes = width if operator == "dispatch" else width * 2
    if row_bytes % 64 != 0:
        return False
    if operator == "dispatch" and row_bytes > 256 * 1024:
        return False
    if operator == "combine" and row_bytes > 256 * 1024:
        return False
    return size in ("small", "large")


def dispatch_gate(topk: int, workers: int) -> float:
    fanout = min(topk, workers)
    return 0.93 * DISPATCH_ROOFLINE_GBPS / (1.0 + 0.127 / fanout)


def expected_aiv(operator: str, workers: int):
    if operator == "dispatch":
        return (48 // 3 + workers - 1) // workers, 48 // 3
    topology_share = (48 + workers - 1) // workers + 48 // 12
    return min(24, max(12, topology_share)), 48 - 48 // 3


def load_dispatch_plan_stats(path: Path):
    destinations = {}
    assignments = 0
    with path.open() as handle:
        header = handle.readline().split()
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            source, row, _, destination = map(int, fields[:4])
            destinations.setdefault((source, row), set()).add(destination)
            assignments += 1
    physical_rows = sum(len(value) for value in destinations.values())
    return {
        "header": header[0],
        "logical_assignments": assignments,
        "physical_rows_from_plan": physical_rows,
        "fanout_mean": physical_rows / len(destinations),
        "fanout_min": min(map(len, destinations.values())),
        "fanout_max": max(map(len, destinations.values())),
    }


def texts(log_dir: Path):
    return [(path, path.read_text(errors="replace"))
            for path in sorted(log_dir.glob("pe*.log"))]


def parse_dispatch(log_dir: Path, workers: int):
    logs = texts(log_dir)
    sample_max = {}
    physical = set()
    logical = set()
    input_bytes = set()
    worker_lanes = set()
    inc_lanes = set()
    assignment_sum = 0
    result_pass = 0
    for _, text in logs:
        if "STREAM_DISPATCH_RESULT" in text and "pass=1" in text:
            result_pass += 1
        result = re.search(r"STREAM_DISPATCH_RESULT[^\n]*verified_assignments=(\d+)", text)
        if result:
            assignment_sum += int(result.group(1))
        config = re.search(r"STREAM_DISPATCH_CONFIG[^\n]*", text)
        if config:
            line = config.group(0)
            for pattern, output in [
                (r"physical_output_bytes=(\d+)", physical),
                (r"logical_output_bytes=(\d+)", logical),
                (r"input_bytes=(\d+)", input_bytes),
            ]:
                match = re.search(pattern, line)
                if match:
                    output.add(int(match.group(1)))
            pe = re.search(r"pe=(\d+)", line)
            inc = re.search(r"inc_lanes=(\d+)", line)
            upload = re.search(r"upload_lanes=(\d+)", line)
            if pe and int(pe.group(1)) < workers and upload:
                worker_lanes.add(int(upload.group(1)))
            if pe and int(pe.group(1)) == workers and inc:
                inc_lanes.add(int(inc.group(1)))
        for line in text.splitlines():
            if "STREAM_DISPATCH_TIMING" not in line:
                continue
            sample = re.search(r"sample=(\d+)", line)
            timing = re.search(r"rank_us=([0-9.]+)", line)
            if sample and timing:
                index = int(sample.group(1))
                sample_max[index] = max(sample_max.get(index, 0.0),
                                        float(timing.group(1)))
    physical_one = next(iter(physical)) if len(physical) == 1 else 0
    total_us = sum(sample_max.values())
    total_bytes = physical_one * len(sample_max)
    return {
        "success_ranks": result_pass,
        "expected_ranks": workers + 1,
        "sample_count": len(sample_max),
        "slow_rank_us_sum": total_us,
        "physical_bytes_per_sample": physical_one,
        "logical_bytes_per_sample": next(iter(logical)) if len(logical) == 1 else 0,
        "input_bytes_per_sample": next(iter(input_bytes)) if len(input_bytes) == 1 else 0,
        "verified_assignments_workers": assignment_sum,
        "bandwidth_GBps": total_bytes / total_us / 1000.0 if total_us else 0.0,
        "worker_aiv_lanes": sorted(worker_lanes),
        "inc_aiv_lanes": sorted(inc_lanes),
    }


def parse_combine(log_dir: Path, workers: int):
    logs = texts(log_dir)
    times = []
    physical = set()
    result_pass = 0
    worker_lanes = set()
    inc_owners = set()
    for _, text in logs:
        if "DYNCSR_RESULT SUCCESS" in text:
            result_pass += 1
        timing = re.search(r"DYNCSR_TIMING[^\n]*", text)
        if timing:
            line = timing.group(0)
            match = re.search(r"rank_us=([0-9.]+)", line)
            if match:
                times.append(float(match.group(1)))
            match = re.search(r"physical_ingress_bytes=(\d+)", line)
            if match:
                physical.add(int(match.group(1)))
        evidence = re.search(r"DYNCSR_EVIDENCE[^\n]*", text)
        if evidence:
            line = evidence.group(0)
            pe = re.search(r"pe=(\d+)", line)
            worker = re.search(r"producer_lanes=(\d+)", line)
            owner = re.search(r"owner_count=(\d+)", line)
            if pe and int(pe.group(1)) < workers and worker:
                worker_lanes.add(int(worker.group(1)))
        register = re.search(r"DYNCSR_REGISTER[^\n]*", text)
        if register:
            line = register.group(0)
            pe = re.search(r"pe=(\d+)", line)
            owner = re.search(r"owner_count=(\d+)", line)
            if pe and int(pe.group(1)) == workers and owner:
                inc_owners.add(int(owner.group(1)))
    slow_us = max(times) if times else 0.0
    physical_total = next(iter(physical)) if len(physical) == 1 else 0
    return {
        "success_ranks": result_pass,
        "expected_ranks": workers + 1,
        "slow_rank_us": slow_us,
        "physical_ingress_bytes": physical_total,
        "bandwidth_GBps": physical_total / slow_us / 1000.0 if slow_us else 0.0,
        "worker_aiv_lanes": sorted(worker_lanes),
        "inc_owner_count": sorted(inc_owners),
    }


def run_logged(command, env, output: Path):
    started = time.monotonic()
    with output.open("w") as handle:
        process = subprocess.run(command, env=env, stdout=handle,
                                 stderr=subprocess.STDOUT, check=False)
    return process.returncode, time.monotonic() - started


def write_checkpoint(path: Path, document):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases-per-group", type=int, default=100)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--cann", type=Path, default=DEFAULT_CANN)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-new-hard-failure", action="store_true")
    args = parser.parse_args()
    if args.cases_per_group < 1:
        parser.error("--cases-per-group must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "campaign.json"
    rows = []
    if args.resume and checkpoint.exists():
        rows = json.loads(checkpoint.read_text())["cases"]
    completed = {row["case_id"] for row in rows}
    generator = args.build / "bin/inc_dc_os_random_plan"
    if not generator.is_file():
        raise SystemExit(f"missing generator: {generator}")
    if not (args.cann / "set_env.sh").is_file():
        raise SystemExit(f"missing qualified CANN set_env.sh: {args.cann}")
    base_env = os.environ.copy()
    base_env.update({
        "ASCEND_HOME_PATH": str(args.cann),
        "INC_BUILD_DIR": str(args.build),
        "LD_LIBRARY_PATH": f"{args.build / 'lib'}:{base_env.get('LD_LIBRARY_PATH', '')}",
    })
    groups = [(op, size) for op in ("dispatch", "combine")
              for size in ("small", "large")]
    total = len(groups) * args.cases_per_group
    campaign_start = time.monotonic()
    generation_rejections = 0
    stop_reason = None
    for operator, size in groups:
        for index in range(args.cases_per_group):
            case_id = f"{operator}-{size}-{index:03d}"
            if case_id in completed:
                continue
            workers, topk = COMBOS[index % len(COMBOS)]
            case_dir = args.output / operator / size / f"case-{index:03d}-W{workers}-K{topk}"
            case_dir.mkdir(parents=True, exist_ok=True)
            plan = case_dir / ("plan.txt" if operator == "dispatch" else "plan.bin")
            rejected = []
            plan_stats = None
            for attempt in range(32):
                count, width, target = random_shape(operator, size, workers, topk)
                if not valid_shape(operator, size, count, width, workers, topk):
                    rejected.append({"attempt": attempt, "reason": "invalid_shape",
                                     "count": count, "width": width})
                    continue
                generated = subprocess.run(
                    [str(generator), operator, str(workers), str(topk),
                     str(count), str(plan)], text=True, capture_output=True,
                    check=False)
                (case_dir / f"generator-attempt-{attempt:02d}.log").write_text(
                    generated.stdout + generated.stderr)
                if generated.returncode != 0:
                    rejected.append({"attempt": attempt, "reason": "generator_rc",
                                     "rc": generated.returncode})
                    continue
                if operator == "dispatch":
                    plan_stats = load_dispatch_plan_stats(plan)
                    physical_shape_bytes = (
                        plan_stats["physical_rows_from_plan"] * width)
                else:
                    physical_shape_bytes = count * topk * width * 2
                correct_size_class = ((physical_shape_bytes < PERF_GATE_BYTES)
                                      if size == "small" else
                                      (physical_shape_bytes >= PERF_GATE_BYTES))
                if not correct_size_class:
                    rejected.append({"attempt": attempt,
                                     "reason": "physical_size_class",
                                     "physical_shape_bytes": physical_shape_bytes})
                    continue
                break
            else:
                raise RuntimeError(f"shape generation exhausted: {case_id}")
            generation_rejections += len(rejected)
            row = {
                "case_id": case_id,
                "operator": operator,
                "size": size,
                "index": index,
                "workers": workers,
                "topk": topk,
                "count": count,
                "hidden_bytes" if operator == "dispatch" else "hidden": width,
                "random_target_bytes": target,
                "physical_shape_bytes": physical_shape_bytes,
                "plan_sha256": sha256(plan),
                "entropy_source": "Linux getrandom(2), fresh draw per bounded choice",
                "shape_entropy": "Linux getrandom(2), no fixed seed",
                "generation_rejections_before_accept": rejected,
                "case_dir": str(case_dir),
            }
            env = base_env.copy()
            if operator == "dispatch":
                env.update({
                    "INC_STREAM_TOKEN_PLAN_FILE": str(plan),
                    "INC_STREAM_WARMUP": "3",
                    "INC_STREAM_MEASURE": "10",
                })
                command = [
                    str(SINGLE_INC_SCRIPTS / "run_single_inc_stream_dispatch_case.sh"),
                    str(workers), str(count), str(width), str(topk),
                    str(case_dir / "device"), "180", "os_random",
                ]
                row["plan_stats"] = plan_stats
            else:
                env.update({
                    "INC_DYNCSR_LOGICAL_PLAN_FILE": str(plan),
                    "INC_DYNCSR_SERVICE_EPOCHS": "10",
                    "INC_DYNCSR_SERVICE_WARMUP_EPOCHS": "3",
                    "INC_DYNCSR_RANK_DEDUP": "0",
                })
                command = [
                    str(SINGLE_INC_SCRIPTS / "run_single_inc_dyn_case.sh"),
                    str(workers), str(topk), str(count), str(width), "0",
                    str(case_dir / "device"), "180",
                ]
            rc, wall = run_logged(command, env, case_dir / "launcher.log")
            row["process_rc"] = rc
            row["wall_seconds"] = wall
            if operator == "dispatch":
                row["measurement"] = parse_dispatch(case_dir / "device", workers)
            else:
                row["measurement"] = parse_combine(case_dir / "device", workers)
            row["correctness_pass"] = (
                rc == 0 and
                row["measurement"]["success_ranks"] == workers + 1 and
                row["measurement"]["bandwidth_GBps"] > 0.0)
            if operator == "dispatch" and row["correctness_pass"]:
                row["correctness_pass"] = (
                    row["measurement"]["verified_assignments_workers"] ==
                    row["plan_stats"]["logical_assignments"] and
                    row["measurement"]["physical_bytes_per_sample"] ==
                    physical_shape_bytes)
            worker_expected, inc_expected = expected_aiv(operator, workers)
            worker_actual = row["measurement"]["worker_aiv_lanes"]
            inc_key = ("inc_aiv_lanes" if operator == "dispatch"
                       else "inc_owner_count")
            inc_actual = row["measurement"][inc_key]
            row["aiv_expected"] = {"worker": worker_expected,
                                   "inc": inc_expected}
            row["aiv_map_pass"] = (worker_actual == [worker_expected] and
                                   inc_actual == [inc_expected])
            gated = physical_shape_bytes >= PERF_GATE_BYTES
            gate = (dispatch_gate(topk, workers) if operator == "dispatch"
                    else 120.0)
            row["performance_gated"] = gated
            row["performance_gate_GBps"] = gate if gated else None
            row["performance_pass"] = (not gated or
                                       row["measurement"]["bandwidth_GBps"] >= gate)
            row["known_k1_perf_gap"] = bool(
                gated and operator == "combine" and topk == 1 and
                not row["performance_pass"] and row["correctness_pass"])
            row["new_hard_failure"] = bool(
                not row["correctness_pass"] or not row["aiv_map_pass"] or
                (gated and not row["performance_pass"] and
                 not row["known_k1_perf_gap"]))
            row["pass"] = bool(not row["new_hard_failure"])
            rows.append(row)
            document = {
                "schema": "single-inc-os-random-campaign-v2-random-shape",
                "complete": len(rows) == total,
                "expected_cases": total,
                "completed_cases": len(rows),
                "cases_per_group": args.cases_per_group,
                "groups": [f"{op}-{sz}" for op, sz in groups],
                "performance_gate_bytes": PERF_GATE_BYTES,
                "dispatch_roofline_GBps": DISPATCH_ROOFLINE_GBPS,
                "generation_rejections": generation_rejections,
                "stop_reason": stop_reason,
                "route_randomness": (
                    "unique experts per token sampled without replacement; "
                    "every bounded draw calls Linux getrandom(2); no seed/PRNG"),
                "cases": rows,
            }
            write_checkpoint(checkpoint, document)
            elapsed = time.monotonic() - campaign_start
            print(
                f"RANDOM_CAMPAIGN progress={len(rows)}/{total} case={case_id} "
                f"W={workers} K={topk} correct={int(row['correctness_pass'])} "
                f"gate={int(row['performance_pass'])} aiv={int(row['aiv_map_pass'])} "
                f"GBps={row['measurement']['bandwidth_GBps']:.3f} "
                f"case_s={wall:.1f} elapsed_s={elapsed:.1f}", flush=True)
            if args.stop_on_new_hard_failure and row["new_hard_failure"]:
                stop_reason = f"new_hard_failure:{case_id}"
                document["stop_reason"] = stop_reason
                write_checkpoint(checkpoint, document)
                break
        if stop_reason:
            break
    failures = sum(row["new_hard_failure"] for row in rows)
    print(f"RANDOM_CAMPAIGN_DONE cases={len(rows)} new_hard_failures={failures} "
          f"known_k1_perf_gaps={sum(r['known_k1_perf_gap'] for r in rows)} "
          f"report={checkpoint}")
    return 0 if failures == 0 and len(rows) == total else 1


if __name__ == "__main__":
    sys.exit(main())
