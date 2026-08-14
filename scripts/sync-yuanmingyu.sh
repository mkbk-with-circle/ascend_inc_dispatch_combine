#!/usr/bin/env bash
# Sync ascend-样机 with remote yuanmingyu-ymy (local is source of truth).
set -euo pipefail

REMOTE_HOST="${INC_SYNC_HOST:-yuanmingyu-ymy}"
REMOTE_DIR="${INC_SYNC_DIR:-/home/ymy/work/ascend-样机}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOCAL_ROOT=$(dirname "$SCRIPT_DIR")

RSYNC_EXCLUDES=(
  --exclude '.git/objects'
  --exclude 'shmem/build/'
  --exclude '**/build/'
  --exclude '*.o'
  --exclude '*.tmp.json'
  --exclude 'shmem/docs/inc/report/*_logs/'
  --exclude 'shmem/docs/inc/report/c12c_logs/'
  --exclude 'shmem/docs/inc/report/c12c_bw_tune_logs/'
  --exclude 'shmem/docs/inc/report/c12c_tuned_logs/'
  --exclude 'shmem/docs/inc/report/d03_dc_ll_logs/'
  --exclude 'shmem/docs/inc/hardware_profiles/910b-yuanmingyu/gates/'
)

usage() {
  echo "usage: $0 push|pull|push-profile|bandwidth-test"
  echo "  push         rsync project to remote"
  echo "  pull         rsync 910b-yuanmingyu profile data from remote"
  echo "  push-profile push only hardware_profiles/910b-yuanmingyu (after gates)"
  echo "  bandwidth-test quick 100MB scp up/down timing"
}

do_rsync() {
  if command -v rsync >/dev/null 2>&1 && ssh -o BatchMode=yes "$REMOTE_HOST" 'command -v rsync' >/dev/null 2>&1; then
    rsync -avz --delete "${RSYNC_EXCLUDES[@]}" "$@"
  else
    echo "WARN: rsync unavailable, using scp for small trees only" >&2
    return 1
  fi
}

cmd_push() {
  do_rsync "$LOCAL_ROOT/" "${REMOTE_HOST}:${REMOTE_DIR}/" || {
    ssh "$REMOTE_HOST" "mkdir -p ${REMOTE_DIR}"
    scp -r "$LOCAL_ROOT/shmem" "$LOCAL_ROOT/hardware_config.md" "$LOCAL_ROOT/road_map.md" \
      "${REMOTE_HOST}:${REMOTE_DIR}/" 2>/dev/null || scp -r "$LOCAL_ROOT/shmem" "${REMOTE_HOST}:${REMOTE_DIR}/"
  }
  echo "OK push -> ${REMOTE_HOST}:${REMOTE_DIR}"
}

cmd_pull() {
  PROFILE_LOCAL="$LOCAL_ROOT/shmem/docs/inc/hardware_profiles/910b-yuanmingyu"
  mkdir -p "$PROFILE_LOCAL"
  do_rsync "${REMOTE_HOST}:${REMOTE_DIR}/shmem/docs/inc/hardware_profiles/910b-yuanmingyu/" "$PROFILE_LOCAL/" \
    || scp -r "${REMOTE_HOST}:${REMOTE_DIR}/shmem/docs/inc/hardware_profiles/910b-yuanmingyu/" "$PROFILE_LOCAL/"
  echo "OK pull profile -> $PROFILE_LOCAL"
}

cmd_push_profile() {
  PROFILE_LOCAL="$LOCAL_ROOT/shmem/docs/inc/hardware_profiles/910b-yuanmingyu"
  do_rsync "$PROFILE_LOCAL/" "${REMOTE_HOST}:${REMOTE_DIR}/shmem/docs/inc/hardware_profiles/910b-yuanmingyu/" \
    || scp -r "$PROFILE_LOCAL/"* "${REMOTE_HOST}:${REMOTE_DIR}/shmem/docs/inc/hardware_profiles/910b-yuanmingyu/"
  echo "OK push-profile"
}

cmd_bandwidth_test() {
  local tmp="/tmp/inc-sync-bench-100m.bin"
  dd if=/dev/urandom of="$tmp" bs=1m count=100 2>/dev/null
  echo "=== UPLOAD ==="
  /usr/bin/time -p scp -q "$tmp" "${REMOTE_HOST}:/tmp/inc-sync-bench.bin"
  echo "=== DOWNLOAD ==="
  /usr/bin/time -p scp -q "${REMOTE_HOST}:/tmp/inc-sync-bench.bin" "/tmp/inc-sync-bench-dl.bin"
  rm -f "$tmp" "/tmp/inc-sync-bench-dl.bin"
}

case "${1:-}" in
  push) cmd_push ;;
  pull) cmd_pull ;;
  push-profile) cmd_push_profile ;;
  bandwidth-test) cmd_bandwidth_test ;;
  *) usage; exit 2 ;;
esac
