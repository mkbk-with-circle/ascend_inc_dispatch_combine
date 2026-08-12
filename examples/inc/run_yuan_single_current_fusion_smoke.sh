#!/usr/bin/env bash
# Correctness-only smoke for the yuan single-INC + current fusion delivery.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
OUT=${1:-/tmp/yuan-single-current-fusion-smoke-$(date +%Y%m%d-%H%M%S)}
BUILD=${INC_HYBRID_BUILD_DIR:-/tmp/shmem-build-yuan-single-current-fusion}
JOBS=${INC_HYBRID_BUILD_JOBS:-16}
CANN_HOME=${ASCEND_HOME_PATH:-}

if [[ -e "$OUT" && -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing non-empty result directory: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"

finish() {
  local rc=$?
  if [[ "$rc" -eq 0 ]]; then
    printf 'PASS\n' >"$OUT/STATUS.txt"
  else
    printf 'FAIL rc=%s\n' "$rc" >"$OUT/STATUS.txt"
  fi
}
trap finish EXIT
printf 'RUNNING\n' >"$OUT/STATUS.txt"

if [[ -z "$CANN_HOME" || ! -f "$CANN_HOME/set_env.sh" ]]; then
  for candidate in \
    /usr/local/Ascend/cann-9.1.0-beta.3 \
    /usr/local/Ascend/ascend-toolkit; do
    if [[ -f "$candidate/set_env.sh" ]]; then
      CANN_HOME=$candidate
      break
    fi
  done
fi
[[ -f "$CANN_HOME/set_env.sh" ]] || {
  echo "cannot locate CANN set_env.sh" >&2
  exit 2
}
# shellcheck source=/dev/null
source "$CANN_HOME/set_env.sh"
export ASCEND_HOME_PATH="$CANN_HOME"

"$ROOT/examples/inc/fusion_kernel/tools/prepare_catlass_dependency.sh" |
  tee "$OUT/catlass_prepare.log"

cmake -S "$ROOT" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DUSE_EXAMPLES=ON -DSOC_TYPE=Ascend910B |
  tee "$OUT/cmake_configure.log"
cmake --build "$BUILD" -j"$JOBS" --target \
  inc_dc_single_inc_stream inc_dc_sv2_dyn_csr_combine inc_fusion_e2e |
  tee "$OUT/cmake_build.log"

export INC_BUILD_DIR="$BUILD"
export INC_SINGLE_INC_PHY=${INC_SINGLE_INC_PHY:-0}
export INC_SINGLE_INC_WORKER_PHYS_W2=${INC_SINGLE_INC_WORKER_PHYS_W2:-"1 2"}
export INC_SINGLE_INC_EXPECTED_RELATIONS_W2=${INC_SINGLE_INC_EXPECTED_RELATIONS_W2:-"HCCS HCCS"}

DC_SCRIPTS=$ROOT/examples/inc/dispatch_combine/scripts/single_inc
INC_STREAM_WARMUP=1 INC_STREAM_MEASURE=1 \
  "$DC_SCRIPTS/run_single_inc_stream_dispatch_case.sh" \
  2 8 256 1 "$OUT/dispatch_w2_k1" 120 smoke
INC_STREAM_WARMUP=1 INC_STREAM_MEASURE=1 \
  "$DC_SCRIPTS/run_single_inc_stream_dispatch_case.sh" \
  2 8 256 2 "$OUT/dispatch_w2_k2" 120 smoke

INC_DYNCSR_SERVICE_WARMUP_EPOCHS=1 INC_DYNCSR_SERVICE_EPOCHS=2 \
  "$DC_SCRIPTS/run_single_inc_dyn_case.sh" \
  2 1 32 128 0 "$OUT/combine_w2_k1" 120
INC_DYNCSR_SERVICE_WARMUP_EPOCHS=1 INC_DYNCSR_SERVICE_EPOCHS=2 \
  "$DC_SCRIPTS/run_single_inc_dyn_case.sh" \
  2 2 32 128 0 "$OUT/combine_w2_k2" 120

INC_FUSION_BUILD_DIR="$BUILD" \
INC_FUSION_SWEEP_WARMUP=1 \
INC_FUSION_SWEEP_MEASURE=1 \
INC_FUSION_SWEEP_TIMEOUT_SEC=180 \
INC_FUSION_W2_MAP="0:1,1:2,2:0" \
INC_FUSION_W2_PHYSICAL_MAP="0:1,1:2,2:0" \
INC_FUSION_SWEEP_CASES="2 17 192 320 1 2" \
  "$ROOT/examples/inc/fusion_kernel/ascend/tests/run_inc_fusion_nb_sweep.sh" \
  "$OUT/fusion"

python3 - "$ROOT" "$OUT" "$CANN_HOME" <<'PY'
import csv
import hashlib
import json
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
cann_home = sys.argv[3]

def marker(path, text):
    value = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    return text in value

cases = {
    "dispatch_w2_k1": marker(out / "dispatch_w2_k1" / "pe0.log", "STREAM_DISPATCH_RESULT") and all(
        marker(out / "dispatch_w2_k1" / f"pe{pe}.log", "pass=1") for pe in range(3)
    ),
    "dispatch_w2_k2": all(
        marker(out / "dispatch_w2_k2" / f"pe{pe}.log", "pass=1") for pe in range(3)
    ),
    "combine_w2_k1": all(
        marker(out / "combine_w2_k1" / f"pe{pe}.log", "DYNCSR_RESULT SUCCESS") for pe in range(3)
    ),
    "combine_w2_k2": all(
        marker(out / "combine_w2_k2" / f"pe{pe}.log", "DYNCSR_RESULT SUCCESS") for pe in range(3)
    ),
}
with (out / "fusion" / "results.csv").open(newline="", encoding="utf-8") as handle:
    fusion_rows = list(csv.DictReader(handle))
cases["fusion_w2"] = len(fusion_rows) == 1 and fusion_rows[0]["status"] == "PASS"
if not all(cases.values()):
    raise SystemExit(f"smoke marker verification failed: {cases}")

patch = root / "examples/inc/fusion_kernel/third_party_patches/catlass_grouped_matmul_zero_m.patch"
summary = {
    "schema": "yuan_single_current_fusion_smoke_v1",
    "source_commit": subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip(),
    "single_inc_source_commit": "e06ce80d875a3509c9660d20b7a2ec0f6ad68c2f",
    "fusion_source_commit": "e06ce80d875a3509c9660d20b7a2ec0f6ad68c2f",
    "catlass_commit": "7d4c8401ae2b2aeb8a5786671e4fe7f53ca96c18",
    "catlass_patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
    "cann_home": cann_home,
    "performance_tested": False,
    "cases": cases,
    "pass": True,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY

echo "HYBRID_SMOKE_PASS out=$OUT performance_tested=false"
