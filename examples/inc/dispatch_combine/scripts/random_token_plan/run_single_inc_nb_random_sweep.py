#!/usr/bin/env python3
"""Reproducible W2/W4/W8 single-INC random-route sweep for 910b2c-nb.

The manifest is exhaustive through 8 GiB.  Execution is resumable and each
case owns a deterministic seed, so a failure can be replayed exactly without
retaining multi-gigabyte campaign artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

from run_single_inc_os_random_campaign import parse_combine, parse_dispatch


SCRIPT_DIR = Path(__file__).resolve().parent
SINGLE_INC_SCRIPTS = SCRIPT_DIR.parent / "single_inc"
ROOT = SCRIPT_DIR.parents[4]
DEFAULT_PROFILE = ROOT / "docs/inc/configs/910b2c-nb.env"
SIZES = [
    4 << 10, 16 << 10, 64 << 10, 256 << 10,
    1 << 20, 4 << 20, 16 << 20, 64 << 20,
    128 << 20, 256 << 20, 512 << 20,
    1 << 30, 2 << 30, 4 << 30, 8 << 30,
]
TOPKS = {
    2: [1, 2, 4, 16],
    4: [1, 2, 4, 6, 8, 16, 32],
    8: [1, 2, 4, 6, 8, 16, 32, 64],
}


def case_seed(campaign_seed: int, case_id: str) -> int:
    material = f"{campaign_seed}:{case_id}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def expected_fanout(workers: int, topk: int) -> float:
    experts = workers * 8
    absent = (math.comb(experts - 8, topk) / math.comb(experts, topk)
              if topk <= experts - 8 else 0.0)
    return workers * (1.0 - absent)


def shape(operator: str, target: int, workers: int, topk: int) -> tuple[int, int]:
    if operator == "dispatch":
        hidden_bytes = 8192
        count = max(1, round(target / (workers * expected_fanout(workers, topk) * hidden_bytes)))
        return int(count), hidden_bytes
    row_bytes = 16384 if topk == 1 else 65536
    count = max(1, round(target / (topk * row_bytes)))
    return int(count), row_bytes // 2


def matrix(seed: int) -> list[dict]:
    rows = []
    for operator in ("dispatch", "combine"):
        for workers in (2, 4, 8):
            for topk in TOPKS[workers]:
                for target in SIZES:
                    case_id = f"{operator}-W{workers}-K{topk}-B{target}"
                    count, width = shape(operator, target, workers, topk)
                    rows.append({
                        "case_id": case_id,
                        "operator": operator,
                        "workers": workers,
                        "topk": topk,
                        "target_bytes": target,
                        "count": count,
                        "hidden_bytes" if operator == "dispatch" else "hidden": width,
                        "seed": case_seed(seed, case_id),
                    })
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profiled_command(profile: Path, command: list[str]) -> list[str]:
    return ["bash", "-c", 'source "$1"; shift; exec "$@"',
            "single-inc-profile", str(profile), *command]


def classify(row: dict, push_roofline: float) -> None:
    measurement = row.get("measurement", {})
    bw = float(measurement.get("bandwidth_GBps", 0.0))
    correct = row.get("process_rc") == 0 and measurement.get("success_ranks") == row["workers"] + 1
    row["correct"] = bool(correct)
    if row["target_bytes"] < 128 << 20:
        row["gate_GBps"] = None
        row["gate_status"] = "PASS_CORRECTNESS" if correct else "FAIL"
    elif row["operator"] == "dispatch" and push_roofline > 0.0:
        rank_fanout = min(row["topk"], row["workers"])
        gate = 0.93 * push_roofline / (1.0 + 0.127 / rank_fanout)
        row["gate_GBps"] = gate
        row["gate_status"] = "PASS" if correct and bw >= gate else "FAIL"
    elif row["operator"] == "combine":
        row["gate_GBps"] = 120.0
        row["gate_status"] = "PASS" if correct and bw >= 120.0 else "FAIL"
    else:
        row["gate_GBps"] = None
        row["gate_status"] = "UNVERIFIED_NO_NB_ROOFLINE" if correct else "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--campaign-seed", type=int, default=20260806)
    parser.add_argument("--push-roofline-gbps", type=float, default=0.0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-plans", action="store_true")
    parser.add_argument("--workers", type=int, choices=(2, 4, 8))
    parser.add_argument("--operator", choices=("dispatch", "combine"))
    parser.add_argument("--min-target-bytes", type=int, default=0)
    parser.add_argument("--max-target-bytes", type=int, default=8 << 30)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--min-free-gib", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    planned = [r for r in matrix(args.campaign_seed)
               if (args.workers is None or r["workers"] == args.workers)
               and (args.operator is None or r["operator"] == args.operator)
               and args.min_target_bytes <= r["target_bytes"] <= args.max_target_bytes]
    manifest = {
        "schema": "single-inc-910b2c-nb-random-sweep-v1",
        "campaign_seed": args.campaign_seed,
        "route_generator": "splitmix64 + partial Fisher-Yates, unique experts/token",
        "sizes_bytes": SIZES,
        "topk_by_workers": TOPKS,
        "case_count": len(planned),
        "cases": planned,
    }
    (args.output / "plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.plan_only:
        print(f"PLAN_ONLY cases={len(planned)} output={args.output / 'plan.json'}")
        return 0
    free = shutil.disk_usage(args.output).free
    if free < args.min_free_gib << 30:
        raise SystemExit(f"refusing sweep: only {free / (1 << 30):.1f} GiB free")
    generator = args.build / "bin/inc_dc_os_random_plan"
    if not generator.is_file():
        raise SystemExit(f"missing generator: {generator}")
    checkpoint = args.output / "results.json"
    completed_rows = []
    if args.resume and checkpoint.exists():
        completed_rows = json.loads(checkpoint.read_text()).get("cases", [])
    completed = {r["case_id"] for r in completed_rows}
    for row in planned:
        if row["case_id"] in completed:
            continue
        case_dir = args.output / row["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        plan = case_dir / ("plan.txt" if row["operator"] == "dispatch" else "plan.bin")
        env = os.environ.copy()
        env["INC_DC_RANDOM_SEED"] = str(row["seed"])
        generated = subprocess.run(
            [str(generator), row["operator"], str(row["workers"]), str(row["topk"]),
             str(row["count"]), str(plan)], env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (case_dir / "generator.log").write_text(generated.stdout)
        if generated.returncode != 0:
            row.update(process_rc=generated.returncode, error="plan_generation")
        else:
            row["plan_sha256"] = sha256(plan)
            device = case_dir / "device"
            run_env = os.environ.copy()
            if row["operator"] == "dispatch":
                run_env.update(INC_STREAM_TOKEN_PLAN_FILE=str(plan), INC_STREAM_WARMUP="3", INC_STREAM_MEASURE="10")
                command = [str(SINGLE_INC_SCRIPTS / "run_single_inc_stream_dispatch_case.sh"),
                           str(row["workers"]), str(row["count"]), str(row["hidden_bytes"]),
                           str(row["topk"]), str(device), str(args.timeout), "nb_random"]
            else:
                run_env.update(INC_DYNCSR_LOGICAL_PLAN_FILE=str(plan), INC_DYNCSR_SERVICE_EPOCHS="10",
                               INC_DYNCSR_SERVICE_WARMUP_EPOCHS="3", INC_DYNCSR_RANK_DEDUP="0")
                command = [str(SINGLE_INC_SCRIPTS / "run_single_inc_dyn_case.sh"), str(row["workers"]),
                           str(row["topk"]), str(row["count"]), str(row["hidden"]), "0",
                           str(device), str(args.timeout)]
            started = time.monotonic()
            with (case_dir / "launcher.log").open("w") as log:
                proc = subprocess.run(profiled_command(args.profile, command), env=run_env,
                                      stdout=log, stderr=subprocess.STDOUT, check=False)
            row["process_rc"] = proc.returncode
            row["wall_seconds"] = time.monotonic() - started
            row["measurement"] = (parse_dispatch(device, row["workers"])
                                  if row["operator"] == "dispatch" else parse_combine(device, row["workers"]))
            classify(row, args.push_roofline_gbps)
        if not args.keep_plans and plan.exists():
            plan.unlink()
        completed_rows.append(row)
        checkpoint.write_text(json.dumps({"schema": manifest["schema"], "complete": False,
                                          "cases": completed_rows}, indent=2) + "\n")
    checkpoint.write_text(json.dumps({"schema": manifest["schema"], "complete": True,
                                      "cases": completed_rows}, indent=2) + "\n")
    failures = [r for r in completed_rows if r.get("gate_status", "FAIL").startswith("FAIL")]
    print(f"SWEEP_DONE cases={len(completed_rows)} failures={len(failures)} output={checkpoint}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
