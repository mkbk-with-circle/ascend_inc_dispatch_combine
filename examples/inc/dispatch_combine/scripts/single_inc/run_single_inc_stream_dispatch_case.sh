#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../../../.." && pwd)
# shellcheck source=inc_single_inc_common.sh
source "$SCRIPT_DIR/inc_single_inc_common.sh"

if [[ $# -lt 5 || $# -gt 7 ]]; then
  echo "usage: $0 WORKERS TOKENS_PER_WORKER HIDDEN_BYTES TOPK LOG_DIR [TIMEOUT_SEC] [TAG]" >&2
  exit 2
fi
W=$1 TOKENS=$2 HIDDEN_BYTES=$3 TOPK=$4 LOG_DIR=$5 TIMEOUT_SEC=${6:-90} TAG=${7:-case}
case "$W" in 2|4|8) ;; *) echo "workers must be 2, 4, or 8" >&2; exit 2 ;; esac
for value in "$TOKENS" "$HIDDEN_BYTES" "$TOPK" "$TIMEOUT_SEC"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || exit 2
done

inc_single_source_cann
BUILD=${INC_BUILD_DIR:-$ROOT/build}
BIN="$BUILD/bin/inc_dc_single_inc_stream"
[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 2; }
export LD_LIBRARY_PATH="$BUILD/lib:${LD_LIBRARY_PATH:-}"
export INC_PE_TO_NPU_MAP="$(inc_single_pe_map "$W")"

# 256 KiB amortizes upload/ready control while retaining enough tiles to fill
# the W8 direct pipeline.  A row wider than that remains one legal whole row.
# This is byte/shape based rather than a W2/W4/W8 tuning table.
DEFAULT_TILE_BYTES=262144
if (( HIDDEN_BYTES > DEFAULT_TILE_BYTES )); then
  DEFAULT_TILE_BYTES=$HIDDEN_BYTES
fi
if [[ -n "${INC_STREAM_TILE_BYTES+x}" ]]; then
  TILE_BYTES=$INC_STREAM_TILE_BYTES
  export INC_STREAM_ADAPTIVE_TILE=${INC_STREAM_ADAPTIVE_TILE:-0}
else
  TILE_BYTES=$DEFAULT_TILE_BYTES
  export INC_STREAM_ADAPTIVE_TILE=${INC_STREAM_ADAPTIVE_TILE:-1}
fi
PACKET_BYTES=${INC_STREAM_PACKET_BYTES:-1048576}
WARMUP=${INC_STREAM_WARMUP:-2}
MEASURE=${INC_STREAM_MEASURE:-5}
INC_LANES=${INC_STREAM_INC_LANES:-0}
# Zero asks the binary to distribute one runtime-sized transport cohort across
# workers (8/4/2 lanes per worker for W2/W4/W8 on the current 48-AIV
# platform).  Explicit positive overrides remain available for qualification.
UPLOAD_LANES=${INC_STREAM_UPLOAD_LANES:-0}
export INC_STREAM_TX_PINGPONG=${INC_STREAM_TX_PINGPONG:-1}
export INC_STREAM_DIRECT_DCCI=${INC_STREAM_DIRECT_DCCI:-0}

mkdir -p "$LOG_DIR"
exec 9>/tmp/inc_single_inc_npu.lock
flock 9
inc_single_verify_live_topology "$W" | tee "$LOG_DIR/topology.log"
inc_single_wait_for_npu_idle "stream_dispatch_W${W}_${TAG}"

port=$((24000 + ($$ % 16000)))
while ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"; do
  port=$((port + 1))
done
NPES=$((W + 1))
export SHMEM_UID_SESSION_ID="single-inc-stream-dispatch-${W}-$$-$(date +%s)"
pids=()
rc=0
for ((pe=0; pe<NPES; ++pe)); do
  timeout --kill-after=5 "$TIMEOUT_SEC" "$BIN" \
    "$NPES" "$pe" "tcp://127.0.0.1:${port}" 16 0 \
    "$TOKENS" "$HIDDEN_BYTES" "$TOPK" "$TILE_BYTES" "$PACKET_BYTES" \
    "$WARMUP" "$MEASURE" "$INC_LANES" "$UPLOAD_LANES" \
    >"$LOG_DIR/pe${pe}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
inc_single_wait_for_npu_idle "stream_dispatch_W${W}_${TAG}_post"

passed=0
for ((pe=0; pe<NPES; ++pe)); do
  if grep -q 'STREAM_DISPATCH_RESULT .* pass=1' "$LOG_DIR/pe${pe}.log"; then
    passed=$((passed + 1))
  fi
done
echo "SINGLE_INC_STREAM_DISPATCH workers=$W passed_ranks=$passed expected_ranks=$NPES process_rc=$rc log_dir=$LOG_DIR"
[[ "$rc" -eq 0 && "$passed" -eq "$NPES" ]]
