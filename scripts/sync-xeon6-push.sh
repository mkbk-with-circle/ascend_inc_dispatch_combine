#!/usr/bin/env bash
# 将本地 ascend-样机 推送到远程样机（local -> remote）
# 远程：xeon6:/home/u2200013153/ascend-样机
#
# 用法：
#   ./sync-xeon6-push.sh
#   ASCEND_SYNC_HOST=xeon6 ASCEND_SYNC_REMOTE_DIR=/home/u2200013153 ./sync-xeon6-push.sh
set -euo pipefail

REMOTE_HOST="${ASCEND_SYNC_HOST:-xeon6}"
REMOTE_DIR="${ASCEND_SYNC_REMOTE_DIR:-/home/u2200013153}"
REMOTE_PATH="${REMOTE_DIR}/ascend-样机"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -f "${SCRIPT_DIR}/../.rsync-filter" ]]; then
  LOCAL_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
else
  LOCAL_DIR="${SCRIPT_DIR}/ascend-样机"
fi

RSYNC_FILTER="${LOCAL_DIR}/.rsync-filter"

if [[ ! -d "${LOCAL_DIR}" ]]; then
  echo "ERROR: 本地目录不存在: ${LOCAL_DIR}" >&2
  exit 1
fi

RSYNC_EXCLUDES=(
  --exclude '.git/objects/'
  --exclude 'shmem/build/'
  --exclude '**/build/'
  --exclude '**/CMakeFiles/'
  --exclude '*.o'
  --exclude '*.log'
  --exclude '*rank_*.log'
  --exclude 'driver.log'
  --exclude 'launcher'
  --exclude 'shmem/docs/inc/report/d03_dc_ll_logs/'
  --exclude 'shmem/docs/inc/report/*_logs/'
  --exclude 'shmem/docs/inc/report/c12c_logs/'
  --exclude 'shmem/docs/inc/report/c12c_bw_tune_logs/'
  --exclude 'shmem/docs/inc/report/c12c_tuned_logs/'
  --exclude 'shmem/docs/inc/report/p5/native_baseline_logs/'
  --exclude 'shmem/docs/inc/report/multi_inc_bw_matrix_*/'
  --exclude 'shmem/docs/inc/report/**/*.bin'
  --exclude '**/native/'
  --exclude '**/golden/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
)

if [[ -f "${RSYNC_FILTER}" ]]; then
  RSYNC_EXCLUDES=(--filter="merge ${RSYNC_FILTER}" "${RSYNC_EXCLUDES[@]}")
else
  echo "WARN: 未找到 ${RSYNC_FILTER}，仅使用脚本内 --exclude 列表" >&2
fi

ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_PATH}'"

echo "==> push ${LOCAL_DIR}/ -> ${REMOTE_HOST}:${REMOTE_PATH}/"
rsync -avP \
  "${RSYNC_EXCLUDES[@]}" \
  "${LOCAL_DIR}/" \
  "${REMOTE_HOST}:${REMOTE_PATH}/"

echo "OK upload complete"
