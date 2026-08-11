#!/usr/bin/env bash
# Safe one-case launcher for the dynamic-CSR single-INC combine backend.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../../../.." && pwd)
# shellcheck source=inc_single_inc_common.sh
source "$SCRIPT_DIR/inc_single_inc_common.sh"

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 WORKERS TOPK RESULTS HIDDEN MODE LOG_DIR [TIMEOUT_SEC]" >&2
  exit 2
fi
W=$1 K=$2 R=$3 H=$4 MODE=$5 LOG_DIR=$6 TIMEOUT_SEC=${7:-30}
case "$W" in 2|4|8) ;; *) echo "workers must be 2, 4, or 8" >&2; exit 2 ;; esac
for value in "$K" "$R" "$H" "$TIMEOUT_SEC"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "topk/results/hidden/timeout must be positive integers" >&2
    exit 2
  }
done
[[ "$MODE" =~ ^[0-9]+$ ]] || { echo "mode must be an integer" >&2; exit 2; }

inc_single_source_cann
BUILD=${INC_BUILD_DIR:-$ROOT/build}
BIN="$BUILD/bin/inc_dc_sv2_dyn_csr_combine"
[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 2; }
export LD_LIBRARY_PATH="$BUILD/lib:${LD_LIBRARY_PATH:-}"
export INC_PE_TO_NPU_MAP="$(inc_single_pe_map "$W")"
export INC_DYNCSR_DEVICE_PRODUCER=${INC_DYNCSR_DEVICE_PRODUCER:-1}
export INC_DYNCSR_OVERLAP=${INC_DYNCSR_OVERLAP:-1}
export INC_DYNCSR_BATCHED_READY=${INC_DYNCSR_BATCHED_READY:-1}
export INC_DYNCSR_READY_SCOPE=${INC_DYNCSR_READY_SCOPE:-auto}
export INC_DYNCSR_DEVICE_COMPLETION=${INC_DYNCSR_DEVICE_COMPLETION:-0}
# Zero/absent selects the hardware/topology-fixed producer cohort. Explicit
# positive values are qualification-only and still obey the worker half cap.
export INC_DYNCSR_PRODUCER_LANES=${INC_DYNCSR_PRODUCER_LANES:-0}
export INC_DYNCSR_PRODUCER_WINDOW=${INC_DYNCSR_PRODUCER_WINDOW:-32}
export INC_DYNCSR_COALESCED_GROUP_PUT=${INC_DYNCSR_COALESCED_GROUP_PUT:-1}
export INC_DYNCSR_REMOTE_TX=${INC_DYNCSR_REMOTE_TX:-1}
# Zero leaves the credit policy to the binary, which derives it from the live
# row size and the backend's 16-KiB private-packet boundary.  A positive value
# remains an explicit qualification/debug override.
export INC_DYNCSR_TX_WINDOW=${INC_DYNCSR_TX_WINDOW:-0}
export INC_DYNCSR_WIDE_VECTOR_TILE=${INC_DYNCSR_WIDE_VECTOR_TILE:-1}
# Formal measurements use a prequeued device-side trigger.  Every rank first
# completes allocation/session setup, then maps one common CLOCK_MONOTONIC
# deadline into its local device cycle counter.  Startup/JIT/host scheduling
# skew is therefore outside rank_us, while real protocol waits remain inside.
export INC_DYNCSR_PERSISTENT_LOCAL_TRIGGER=${INC_DYNCSR_PERSISTENT_LOCAL_TRIGGER:-1}

mkdir -p "$LOG_DIR"
exec 9>/tmp/inc_single_inc_npu.lock
flock 9
inc_single_verify_live_topology "$W" | tee "$LOG_DIR/topology.log"
inc_single_wait_for_npu_idle "dyncsr_W${W}_K${K}"

port=$((18000 + ($$ % 20000)))
while ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"; do
  port=$((port + 1))
done
NPES=$((W + 1))
export SHMEM_UID_SESSION_ID="single-inc-dyn-${W}-${K}-$$-$(date +%s)"
GATE_DIR="$LOG_DIR/start_gate_$$"
mkdir -p "$GATE_DIR"
export INC_DC_EXTERNAL_START_GATE_DIR="$GATE_DIR"
gate_timeout_sec=${INC_DYNCSR_GATE_TIMEOUT_SEC:-$((TIMEOUT_SEC / 2))}
(( gate_timeout_sec < 30 )) && gate_timeout_sec=30
export INC_DC_EXTERNAL_START_GATE_TIMEOUT_MS=$((gate_timeout_sec * 1000))
: >"$LOG_DIR/pids.txt"
pids=()
rc=0
cleanup_owned_ranks() {
  local pid
  for pid in "${pids[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
}
trap cleanup_owned_ranks EXIT INT TERM
for ((pe=0; pe<NPES; ++pe)); do
  timeout --kill-after=5 "$TIMEOUT_SEC" "$BIN" \
    "$NPES" "$pe" "tcp://127.0.0.1:${port}" 16 0 \
    "$W" "$K" "$R" "$H" "$MODE" \
    >"$LOG_DIR/pe${pe}.log" 2>&1 &
  pids+=("$!")
  echo "$!" >>"$LOG_DIR/pids.txt"
done

gate_deadline=$((SECONDS + gate_timeout_sec))
while true; do
  gate_ready=0
  for ((pe=0; pe<NPES; ++pe)); do
    [[ -f "$GATE_DIR/combine_${pe}.ready" ]] && gate_ready=$((gate_ready + 1))
  done
  [[ "$gate_ready" -eq "$NPES" ]] && break
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[FAIL] rank exited before aligned device start gate" >&2
      rc=1
      break 2
    fi
  done
  if (( SECONDS >= gate_deadline )); then
    echo "[FAIL] aligned device start gate timed out" >&2
    rc=1
    break
  fi
  sleep 0.01
done
if [[ "$rc" -eq 0 ]]; then
  # Large/skewed plans can spend several hundred milliseconds calibrating all
  # ranks after the host gate.  Keep a full second of lead so transient host
  # scheduling does not turn a valid operator case into DEADLINE_TOO_CLOSE;
  # this lead time is outside the measured device interval.
  start_ns=$(python3 -c 'import time; print(time.monotonic_ns() + 1_000_000_000)')
  printf '%s\n' "$start_ns" >"$GATE_DIR/start_ns"
  printf 'go\n' >"$GATE_DIR/go"
else
  cleanup_owned_ranks
fi
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
trap - EXIT INT TERM
inc_single_wait_for_npu_idle "dyncsr_W${W}_K${K}_post"

success=0
for ((pe=0; pe<NPES; ++pe)); do
  if grep -q 'DYNCSR_RESULT SUCCESS' "$LOG_DIR/pe${pe}.log"; then
    success=$((success + 1))
  fi
done
echo "SINGLE_INC_DYN_CASE workers=$W topk=$K results=$R hidden=$H mode=$MODE success_ranks=$success expected_ranks=$NPES process_rc=$rc log_dir=$LOG_DIR"
[[ "$rc" -eq 0 && "$success" -eq "$NPES" ]]
