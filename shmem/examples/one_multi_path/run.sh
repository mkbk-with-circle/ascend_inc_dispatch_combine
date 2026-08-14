#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")

EXAMPLE=one_multi_path
PE_SIZE="2"
IPPORT="tcp://127.0.0.1:8899"
G_NPUS="2"
FIRST_PE="0"
FIRST_NPU="0"
DATA_SIZE_KB="2048"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -pes)
            PE_SIZE="$2"
            shift 2
            ;;
        -ipport)
            IPPORT="$2"
            shift 2
            ;;
        -gnpus)
            G_NPUS="$2"
            shift 2
            ;;
        -fpe)
            FIRST_PE="$2"
            shift 2
            ;;
        -fnpu|-dev)
            FIRST_NPU="$2"
            shift 2
            ;;
        -size)
            DATA_SIZE_KB="$2"
            shift 2
            ;;
        *)
            echo "Error: unknown or incomplete option $1."
            exit 1
            ;;
    esac
done

export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH

# FIRST_NPU's upper bound depends on the logical devices visible to CANN; aclrtSetDevice validates it at runtime.
if ! [[ "$PE_SIZE" =~ ^[0-9]+$ && "$G_NPUS" =~ ^[0-9]+$ && "$FIRST_PE" =~ ^[0-9]+$ && \
        "$FIRST_NPU" =~ ^[0-9]+$ && "$DATA_SIZE_KB" =~ ^[0-9]+$ ]]; then
    echo "Error: PE, NPU and size arguments must be non-negative integers."
    exit 1
fi

if (( PE_SIZE < 2 || G_NPUS <= 0 || FIRST_PE + G_NPUS > PE_SIZE || DATA_SIZE_KB <= 0 )); then
    echo "Error: one_multi_path requires at least 2 PEs and a valid local PE mapping."
    exit 1
fi

echo "============================================"
echo " One Path + Multi Path Single-Core Split Copy"
echo " PE_SIZE:    ${PE_SIZE}"
echo " IPPORT:     ${IPPORT}"
echo " G_NPUS:     ${G_NPUS}"
echo " FIRST_PE:   ${FIRST_PE}"
echo " FIRST_NPU:  ${FIRST_NPU}"
echo " LINK_TYPES: one_path=2, multi_path=3"
echo " DATA_SIZE:  ${DATA_SIZE_KB} KB"
echo "============================================"

pids=()
for ((idx = 0; idx < G_NPUS; idx++)); do
    pe_id=$((FIRST_PE + idx))
    "${PROJECT_ROOT}/build/bin/${EXAMPLE}" "$PE_SIZE" "$pe_id" "$IPPORT" "$G_NPUS" "$FIRST_PE" "$FIRST_NPU" \
        "$DATA_SIZE_KB" &
    pid=$!
    pids+=("$pid")
    echo "PE ${pe_id} background process recorded: ${pid}"
done

ret=0
for pid in "${pids[@]}"; do
    wait "$pid"
    cur_ret=$?
    echo "wait process ${pid} done, ret=${cur_ret}"
    if [[ ${cur_ret} -ne 0 ]]; then
        ret=${cur_ret}
    fi
done

cd "$CURRENT_DIR" || exit
exit $ret
