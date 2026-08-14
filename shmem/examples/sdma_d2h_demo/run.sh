#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

ACLSHMEM_MAX_PES=16384

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_non_negative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

detect_npu_count() {
    local count=""
    if command -v npu-smi >/dev/null 2>&1; then
        count=$(npu-smi info -l 2>/dev/null | awk '/^[[:space:]]*[0-9]+[[:space:]]/ { n++ } END { print n + 0 }')
    fi
    if [[ -z "$count" || "$count" -eq 0 ]]; then
        count=$(find /dev -maxdepth 1 -type c -name 'davinci[0-9]*' 2>/dev/null | wc -l)
    fi
    echo "$count"
}

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")

EXAMPLE=sdma_d2h_demo

cd "$SCRIPT_DIR"

PE_SIZE="2"
IPPORT="tcp://127.0.0.1:8766"
TEST_OP="all"
TEST_TYPE="int"
ELEM_COUNT="1048576"
HEAP_MB="16"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -ipport)
            if [ -n "$2" ]; then
                IPPORT="$2"
                shift 2
            else
                echo "Error: -ipport requires a value."
                exit 1
            fi
            ;;
        -pes)
            if [ -n "$2" ]; then
                PE_SIZE="$2"
                shift 2
            else
                echo "Error: -pes requires a value."
                exit 1
            fi
            ;;
        -op)
            if [ -n "$2" ]; then
                TEST_OP="$2"
                shift 2
            else
                echo "Error: -op requires a value."
                exit 1
            fi
            ;;
        -type)
            if [ -n "$2" ]; then
                TEST_TYPE="$2"
                shift 2
            else
                echo "Error: -type requires a value."
                exit 1
            fi
            ;;
        -elems)
            if [ -n "$2" ]; then
                ELEM_COUNT="$2"
                shift 2
            else
                echo "Error: -elems requires a value."
                exit 1
            fi
            ;;
        -heap_mb)
            if [ -n "$2" ]; then
                HEAP_MB="$2"
                shift 2
            else
                echo "Error: -heap_mb requires a value."
                exit 1
            fi
            ;;
        -example)
            if [ -n "$2" ]; then
                EXAMPLE="$2"
                shift 2
            else
                echo "Error: -example requires a value."
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknown option $1."
            exit 1
            ;;
    esac
done

if ! is_positive_integer "$PE_SIZE"; then
    echo "Error: -pes must be a positive decimal integer."
    exit 1
fi
if [[ "$PE_SIZE" -gt "$ACLSHMEM_MAX_PES" ]]; then
    echo "Error: -pes must be no greater than ACLSHMEM_MAX_PES=${ACLSHMEM_MAX_PES}."
    exit 1
fi

NPU_COUNT=$(detect_npu_count)
if ! is_positive_integer "$NPU_COUNT"; then
    echo "Error: failed to detect available NPU count."
    exit 1
fi
if [[ "$PE_SIZE" -gt "$NPU_COUNT" ]]; then
    echo "Error: -pes ${PE_SIZE} requires ${PE_SIZE} NPU(s), but only ${NPU_COUNT} available."
    exit 1
fi

rm -rf ./output
export SHMEM_UID_SESSION_ID=127.0.0.1:8899
export LD_LIBRARY_PATH="${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}"

pids=()
started_count=0
for (( idx = 0; idx < PE_SIZE; idx = idx + 1 )); do
    # 多PE样例必须同时拉起多个进程，否则SHMEM初始化和barrier会等待缺失PE。
    "${PROJECT_ROOT}/build/bin/${EXAMPLE}" "${PE_SIZE}" "${idx}" "${IPPORT}" "${TEST_OP}" "${TEST_TYPE}" \
        "${ELEM_COUNT}" "${HEAP_MB}" &
    pid=$!
    pids+=("$pid")
    started_count=$((started_count + 1))
    echo "$pid background process recorded"
done

if [[ "$started_count" -eq 0 ]]; then
    echo "Error: no background process started."
    exit 1
fi

ret=0
for pid in "${pids[@]}"; do
    wait "$pid"
    cur_ret=$?
    echo "wait process $pid done"
    if [[ $cur_ret -ne 0 ]]; then
        ret=$cur_ret
    fi
done

cd "$CURRENT_DIR"
exit $ret
