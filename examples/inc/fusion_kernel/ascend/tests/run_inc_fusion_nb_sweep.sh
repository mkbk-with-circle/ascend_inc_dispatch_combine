#!/usr/bin/env bash
# Reproducible W2/W4 single-plane fusion sweep for nb-borrow.  This runner is
# deliberately a machine profile: the kernel/API itself contains no physical
# NPU IDs.  Override INC_FUSION_W2_MAP/INC_FUSION_W4_MAP when the qualified
# same-plane placement changes.
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../../../.." && pwd)
BUILD=${INC_FUSION_BUILD_DIR:-/tmp/shmem-build-fusion}
BIN="$BUILD/bin/inc_fusion_e2e"
LOG_ROOT=${1:-/tmp/inc-fusion-nb-sweep-$(date +%Y%m%d-%H%M%S)}
MEASURE=${INC_FUSION_SWEEP_MEASURE:-5}
WARMUP=${INC_FUSION_SWEEP_WARMUP:-2}
TIMEOUT_SEC=${INC_FUSION_SWEEP_TIMEOUT_SEC:-300}
W2_MAP=${INC_FUSION_W2_MAP:-0:1,1:2,2:0}
W4_MAP=${INC_FUSION_W4_MAP:-0:1,1:2,2:3,3:4,4:0}
W2_PHYSICAL_MAP=${INC_FUSION_W2_PHYSICAL_MAP:-$W2_MAP}
W4_PHYSICAL_MAP=${INC_FUSION_W4_PHYSICAL_MAP:-$W4_MAP}

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 2; }
[[ "$MEASURE" =~ ^[1-9][0-9]*$ && "$WARMUP" =~ ^[0-9]+$ ]] || exit 2
mkdir -p "$LOG_ROOT"
SUMMARY="$LOG_ROOT/results.csv"
printf '%s\n' 'environment,workers,tokens,hidden,intermediate,topk,activation_waves,measure,mean_us,median_us,cv_percent,dc_theoretical_max,dc_window_speedup,overlap_realized,status,log_dir' >"$SUMMARY"

if [[ -f /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
  echo "no CANN environment found" >&2
  exit 2
fi
export LD_LIBRARY_PATH="$BUILD/lib:${LD_LIBRARY_PATH:-}"

exec 9>/tmp/inc_fusion_nb_npu.lock
flock 9

map_npu() {
  local map=$1 rank=$2
  tr ',' '\n' <<<"$map" | awk -F: -v r="$rank" '$1==r {print $2; exit}'
}

plane_npus() {
  local workers=$1 map=$2 topo row inc_phy col relation
  inc_phy=$(map_npu "$map" "$workers")
  [[ "$inc_phy" =~ ^[0-9]+$ ]] || return 1
  topo=$(npu-smi info -t topo -i 0 2>/dev/null) || return 1
  # The topology header itself starts with "NPU0" on this npu-smi build.
  # Skip it explicitly so an INC placed on physical NPU0 does not make the
  # header look like a topology row (and report peers as "NPU1", "NPU2", ...).
  row=$(awk -v name="NPU${inc_phy}" 'NR > 1 && $1==name {print; exit}' <<<"$topo")
  [[ -n "$row" ]] || return 1
  for ((col=2; col<=17; ++col)); do
    relation=$(awk -v c="$col" '{print $c}' <<<"$row")
    [[ "$relation" == X || "$relation" == HCCS ||
       "$relation" == HCCS_SW ]] && echo $((col - 2))
  done
  # Do not leak the status of the final (usually cross-plane) relation test.
  # The caller consumes the emitted NPU list and treats a nonzero status as a
  # topology-query failure.
  return 0
}

wait_target_plane_idle() {
  local workers=$1 map=$2
  local deadline=$((SECONDS + 300))
  while true; do
    local phy busy=0 plane
    plane=$(plane_npus "$workers" "$map") || return 1
    [[ -n "$plane" ]] || return 1
    while read -r phy; do
      npu-smi info -t proc-mem -i "$phy" 2>/dev/null |
        grep -q 'No process in device' || { busy=1; break; }
    done <<<"$plane"
    (( busy == 0 )) && return 0
    (( SECONDS >= deadline )) && return 1
    sleep 5
  done
}

verify_same_plane() {
  local workers=$1 map=$2 topo row rank phy inc_phy relation seen=,
  topo=$(npu-smi info -t topo -i 0 2>/dev/null) || return 1
  inc_phy=$(map_npu "$map" "$workers")
  [[ "$inc_phy" =~ ^[0-9]+$ ]] || return 1
  row=$(awk -v name="NPU${inc_phy}" 'NR > 1 && $1==name {print; exit}' <<<"$topo")
  [[ -n "$row" ]] || return 1
  for ((rank=0; rank<=workers; ++rank)); do
    phy=$(map_npu "$map" "$rank")
    [[ "$phy" =~ ^[0-9]+$ ]] || return 1
    [[ "$seen" != *",${phy},"* ]] || {
      echo "refused: duplicate physical NPU ${phy} in mapping" >&2
      return 1
    }
    seen+="${phy},"
    (( rank == workers )) && continue
    relation=$(awk -v col=$((phy + 2)) '{print $col}' <<<"$row")
    [[ "$relation" == HCCS || "$relation" == HCCS_SW ]] || {
      echo "refused: INC NPU${inc_phy} -> worker NPU${phy} is ${relation:-unknown}" >&2
      return 1
    }
  done
}

append_result() {
  local workers=$1 tokens=$2 hidden=$3 intermediate=$4 topk=$5 aw=$6 dir=$7 rc=$8
  python3 - "$SUMMARY" "$workers" "$tokens" "$hidden" "$intermediate" "$topk" "$aw" "$dir" "$rc" <<'PY'
import glob, math, os, re, statistics, sys
summary, workers, tokens, hidden, intermediate, topk, aw, directory, rc = sys.argv[1:]
rows = []
passes = 0
trace = ("", "", "")
for path in sorted(glob.glob(os.path.join(directory, "pe*.log"))):
    text = open(path, encoding="utf-8", errors="replace").read()
    pe_match = re.search(r"pe(\d+)\.log$", path)
    pe = int(pe_match.group(1)) if pe_match else -1
    if re.search(r"FUSION_E2E_RESULT .* pass=1", text):
        passes += 1
    match = re.search(r"FUSION_SAMPLES pe=\d+ us=([^\n]+)", text)
    if match and 0 <= pe < int(workers):
        rows.append([float(value) for value in match.group(1).split(",")])
    match = re.search(r"dc_window_speedup=([0-9.eE+-]+) dc_theoretical_max=([0-9.eE+-]+)", text)
    overlap = re.search(r"overlap_realized=([0-9.eE+-]+)", text)
    if match and overlap:
        trace = (match.group(2), match.group(1), overlap.group(1))
npes = int(workers) + 1
status = "PASS" if int(rc) == 0 and passes == npes and len(rows) == int(workers) else "FAIL"
if rows and len({len(row) for row in rows}) == 1:
    makespan = [max(values) for values in zip(*rows)]
    mean = statistics.mean(makespan)
    median = statistics.median(makespan)
    cv = statistics.pstdev(makespan) / mean * 100.0 if len(makespan) > 1 else 0.0
    metrics = (f"{mean:.3f}", f"{median:.3f}", f"{cv:.3f}")
else:
    metrics = ("", "", "")
with open(summary, "a", encoding="utf-8") as out:
    out.write(",".join([
        "910B2C-nb", workers, tokens, hidden, intermediate, topk, aw,
        str(len(rows[0]) if rows else 0), *metrics, *trace, status, directory
    ]) + "\n")
print(f"SWEEP_RESULT W={workers} T={tokens} H={hidden} I={intermediate} K={topk} status={status} mean_us={metrics[0]}")
PY
}

run_case() {
  local workers=$1 tokens=$2 hidden=$3 intermediate=$4 topk=$5 aw=$6
  local npes=$((workers + 1)) map physical_map port dir pe pid rc=0
  if (( workers == 2 )); then
    map=$W2_MAP; physical_map=$W2_PHYSICAL_MAP
  else
    map=$W4_MAP; physical_map=$W4_PHYSICAL_MAP
  fi
  verify_same_plane "$workers" "$physical_map" || return 20
  wait_target_plane_idle "$workers" "$physical_map" || {
    echo "target HCCS plane idle wait timed out" >&2; return 21;
  }
  port=$((24000 + ($$ + workers * 101 + tokens + hidden + topk) % 15000))
  while ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"; do port=$((port + 1)); done
  dir="$LOG_ROOT/W${workers}_T${tokens}_H${hidden}_I${intermediate}_K${topk}_A${aw}"
  mkdir -p "$dir"
  export INC_PE_TO_NPU_MAP=$map
  export SHMEM_UID_SESSION_ID="fusion-sweep-$$-${port}"
  export FUSION_ITERATIONS=$((WARMUP + MEASURE))
  export FUSION_WARMUP=$WARMUP
  local pids=()
  for ((pe=0; pe<npes; ++pe)); do
    timeout --kill-after=5 "$TIMEOUT_SEC" "$BIN" \
      "$npes" "$pe" "tcp://127.0.0.1:${port}" 16 0 \
      "$tokens" "$hidden" "$intermediate" "$topk" "$aw" \
      >"$dir/pe${pe}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
  append_result "$workers" "$tokens" "$hidden" "$intermediate" "$topk" "$aw" "$dir" "$rc"
  wait_target_plane_idle "$workers" "$physical_map" || return 21
  return "$rc"
}

# Runtime shape matrix.  It includes non-aligned tails, balanced medium FFNs,
# dense top-k and multi-packet rows without encoding per-shape kernel tuning.
CASES=${INC_FUSION_SWEEP_CASES:-'
2 17 192 320 1 2
2 32 256 512 2 2
2 64 1024 1024 4 2
2 128 2048 1024 2 2
2 256 16384 128 2 2
4 17 192 320 1 2
4 32 256 512 2 2
4 64 1024 1024 4 2
4 128 2048 1024 8 2
4 256 16384 128 2 2
'}

overall=0
while read -r workers tokens hidden intermediate topk aw; do
  [[ -z "${workers:-}" ]] && continue
  run_case "$workers" "$tokens" "$hidden" "$intermediate" "$topk" "$aw" || overall=1
done <<<"$CASES"

printf 'FUSION_SWEEP_DONE status=%s summary=%s\n' "$overall" "$SUMMARY"
exit "$overall"
