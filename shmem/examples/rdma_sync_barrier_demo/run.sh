#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

DEMO_TYPE=${1:-sync_all}

usage() {
    echo "Usage: bash run.sh [demo_type]"
    echo "  demo_type:"
    echo "    sync_all                - Device侧 aclshmemx_roce_sync_all demo (default)"
    echo "    sync_all_buf            - Device侧 aclshmemx_roce_sync_all(buf, sync_id) demo"
    echo "    barrier_all             - Device侧 aclshmemx_roce_barrier_all demo"
    echo "    barrier_all_buf         - Device侧 aclshmemx_roce_barrier_all(buf, sync_id) demo"
    echo "    sync_team               - Device侧 aclshmemx_roce_team_sync(team) demo"
    echo "    sync_team_buf           - Device侧 aclshmemx_roce_team_sync(team, buf, sync_id) demo"
    echo "    barrier_team            - Device侧 aclshmemx_roce_barrier(team) demo"
    echo "    barrier_team_buf        - Device侧 aclshmemx_roce_barrier(team, buf, sync_id) demo"
}

case "$DEMO_TYPE" in
    sync_all|sync_all_buf|barrier_all|barrier_all_buf|sync_team|sync_team_buf|barrier_team|barrier_team_buf)
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "[ERROR] unknown demo_type: $DEMO_TYPE"
        usage
        exit 1
        ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd ${script_dir}/../../ && pwd)"
export PROJECT_ROOT=${project_root}
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH

export SHMEM_UID_SESSION_ID=127.0.0.1:8899
cd ${PROJECT_ROOT}
pids=()
./build/bin/rdma_sync_barrier_demo 2 0 tcp://127.0.0.1:8899 2 0 0 ${DEMO_TYPE} & # pe 0
pid=$!
pids+=("$pid")

./build/bin/rdma_sync_barrier_demo 2 1 tcp://127.0.0.1:8899 2 0 0 ${DEMO_TYPE} & # pe 1
pid=$!
pids+=("$pid")

ret=0
for pid in ${pids[@]}; do
    wait $pid
    r=$?
    if [[ $r -ne 0 ]]; then
        ret=$r
    fi
    echo "wait $pid finished"
done
exit $ret
