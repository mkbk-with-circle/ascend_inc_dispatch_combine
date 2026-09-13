#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../../../.." && pwd)
# shellcheck source=inc_single_inc_common.sh
source "$SCRIPT_DIR/inc_single_inc_common.sh"

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 WORKERS TOKENS_PER_WORKER HIDDEN_BYTES TOPK LOG_DIR [TIMEOUT_SEC]" >&2
  exit 2
fi
W=$1 TOKENS=$2 HIDDEN_BYTES=$3 TOPK=$4 LOG_DIR=$5 TIMEOUT_SEC=${6:-180}
case "$W" in 2|4) ;; *) echo "V2 qualification supports W2 or W4" >&2; exit 2 ;; esac
for value in "$TOKENS" "$HIDDEN_BYTES" "$TOPK" "$TIMEOUT_SEC"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || exit 2
done
(( TOPK <= W )) || { echo "TOPK must be <= WORKERS" >&2; exit 2; }

inc_single_source_cann
BUILD=${INC_BUILD_DIR:-$ROOT/build}
BIN="$BUILD/bin/inc_dc_single_inc_inline"
[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 2; }
export LD_LIBRARY_PATH="$BUILD/lib:${LD_LIBRARY_PATH:-}"
export INC_PE_TO_NPU_MAP="$(inc_single_pe_map "$W")"

# Two private-MTE credits define one portable 32-KiB transport window.  This
# derives the microbatch from hidden width rather than a W/K lookup table.
WINDOW_BYTES=$((2 * 16 * 1024))
BATCH_TOKENS=${INC_INLINE_V2_BATCH_TOKENS:-$((WINDOW_BYTES / HIDDEN_BYTES))}
(( BATCH_TOKENS >= 1 )) || BATCH_TOKENS=1
(( BATCH_TOKENS <= TOKENS )) || BATCH_TOKENS=$TOKENS
WARMUP=${INC_INLINE_V2_WARMUP:-2}
MEASURE=${INC_INLINE_V2_MEASURE:-5}
INC_LANES=${INC_INLINE_V2_INC_LANES:-0}
WORKER_LANES=${INC_INLINE_V2_WORKER_LANES:-0}
G_NPUS=${INC_INLINE_V2_G_NPUS:-16}
FIRST_NPU=${INC_INLINE_V2_FIRST_NPU:-0}

mkdir -p "$LOG_DIR"
exec 9>/tmp/inc_single_inc_npu.lock
flock 9
inc_single_verify_live_topology "$W" | tee "$LOG_DIR/topology.log"
inc_single_wait_for_npu_idle "inline_v2_dispatch_W${W}"

port=$((24000 + ($$ % 16000)))
while ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"; do
  port=$((port + 1))
done
NPES=$((W + 1))
export SHMEM_UID_SESSION_ID="single-inc-inline-v2-${W}-$$-$(date +%s)"
pids=()
rc=0
for ((pe=0; pe<NPES; ++pe)); do
  timeout --kill-after=5 "$TIMEOUT_SEC" stdbuf -oL -eL "$BIN" \
    "$NPES" "$pe" "tcp://127.0.0.1:${port}" "$G_NPUS" "$FIRST_NPU" \
    "$TOKENS" "$HIDDEN_BYTES" "$TOPK" dispatch "$BATCH_TOKENS" \
    "$WARMUP" "$MEASURE" "$INC_LANES" "$WORKER_LANES" \
    >"$LOG_DIR/pe${pe}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
inc_single_wait_for_npu_idle "inline_v2_dispatch_W${W}_post"

passed=0
for ((pe=0; pe<NPES; ++pe)); do
  if grep -q 'INLINE_V2_DISPATCH_RESULT .* pass=1' "$LOG_DIR/pe${pe}.log"; then
    passed=$((passed + 1))
  fi
done
echo "SINGLE_INC_INLINE_V2_DISPATCH workers=$W topk=$TOPK batch_tokens=$BATCH_TOKENS passed_ranks=$passed expected_ranks=$NPES process_rc=$rc log_dir=$LOG_DIR"
[[ "$rc" -eq 0 && "$passed" -eq "$NPES" ]]
