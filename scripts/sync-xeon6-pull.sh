#!/usr/bin/env bash
# 从远程样机拉取 ascend-样机 到本地（remote -> local）
# 远程：xeon6:/home/u2200013153/ascend-样机
#
# 用法：
#   ./sync-xeon6-pull.sh
#   ASCEND_SYNC_HOST=xeon6 ASCEND_SYNC_REMOTE_DIR=/home/u2200013153 ./sync-xeon6-pull.sh
#
# 脚本位置（二选一）：
#   A) 与 ascend-样机 同级：  sync-tools/sync-xeon6-pull.sh  → LOCAL=../ascend-样机
#   B) 在 ascend-样机 内：    ascend-样机/scripts/sync-xeon6-pull.sh → LOCAL=..
set -euo pipefail

REMOTE_HOST="${ASCEND_SYNC_HOST:-xeon6}"
REMOTE_DIR="${ASCEND_SYNC_REMOTE_DIR:-/home/u2200013153}"
REMOTE_PATH="${REMOTE_DIR}/ascend-样机"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# 若脚本在 ascend-样机/scripts/ 下，LOCAL 为仓库根；否则为同级目录 ascend-样机/
if [[ -f "${SCRIPT_DIR}/../.rsync-filter" ]]; then
  LOCAL_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
else
  LOCAL_DIR="${SCRIPT_DIR}/ascend-样机"
fi

RSYNC_FILTER="${LOCAL_DIR}/.rsync-filter"

# 与 .rsync-filter 叠加的显式排除（filter 文件缺失时仍安全）
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

mkdir -p "${LOCAL_DIR}"

echo "==> pull ${REMOTE_HOST}:${REMOTE_PATH}/ -> ${LOCAL_DIR}/"
rsync -avP \
  "${RSYNC_EXCLUDES[@]}" \
  "${REMOTE_HOST}:${REMOTE_PATH}/" \
  "${LOCAL_DIR}/"

echo "OK download complete"
