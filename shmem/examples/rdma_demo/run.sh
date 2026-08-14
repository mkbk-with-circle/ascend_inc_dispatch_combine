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

usage() {
    echo "Usage: $0 -pes <number_of_pes>"
    echo "  number_of_pes must be a positive integer"
    exit 1
}

num_pes=2
while [[ $# -gt 0 ]]; do
    case "$1" in
        -pes)
            if [[ -z "${2:-}" ]]; then
                usage
            fi
            if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                usage
            fi
            num_pes="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd ${script_dir}/../../ && pwd)"
export PROJECT_ROOT=${project_root}
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH

export SHMEM_UID_SESSION_ID=127.0.0.1:8899
cd ${PROJECT_ROOT}
pids=()
for ((i=0; i<num_pes; i++)); do
    ./build/bin/rdma_demo ${num_pes} ${i} tcp://127.0.0.1:8899 ${num_pes} 0 0 &
    pid=$!
    pids+=("$pid")
done

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
