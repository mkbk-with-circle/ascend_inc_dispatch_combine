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

# Default test type: 0 for all-gather, 1 for put signal
test_type=0
# Global number of PEs across all machines (single-node default: 8)
n_pes=8
# Number of NPUs used on THIS machine
g_npus=8
# Bootstrap endpoint (IP:PORT). MUST be node0's IP, reachable by every node.
ipport=127.0.0.1:8899
# First global PE number used on THIS machine
f_pe=0
# Number of PEs (processes) to launch on THIS machine (defaults to n_pes)
local_pes=
# First NPU card number used on THIS machine
f_npu=0

# Positional arguments:
#   run.sh <test_type> <n_pes> <g_npus> <ipport> <f_pe> <local_pes> <f_npu>
test_type=${1:-${test_type}}
n_pes=${2:-${n_pes}}
g_npus=${3:-${g_npus}}
ipport=${4:-${ipport}}
f_pe=${5:-${f_pe}}
local_pes=${6:-${local_pes}}
f_npu=${7:-${f_npu}}

# Default number of PEs on THIS machine to the global PE count
local_pes=${local_pes:-${n_pes}}

# To run memory checks, launch the whole script with mssanitizer, for example:
#   mssanitizer -- bash examples/udma_demo/run.sh 1
# Both kernels initialize their complete WQE staging scratch once (see
# init_udma_wqe_scratch in udma_demo_kernel.cpp). This experiment verifies
# whether caller-side initialization is sufficient for structure-field WQE
# construction and the full-block DataCopyPad observed by initcheck.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd ${script_dir}/../../ && pwd)"
export PROJECT_ROOT=${project_root}
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH

export SHMEM_UID_SESSION_ID=${ipport}
cd ${PROJECT_ROOT}

pids=()
cleanup() {
    echo -e "\n[Terminating] Caught Ctrl+C, killing background processes..."
    if [ ${#pids[@]} -ne 0 ]; then
        kill "${pids[@]}" 2>/dev/null
    fi
    exit 1
}
trap cleanup SIGINT SIGTERM

for i in $(seq 0 $((local_pes - 1))); do
    pe_id=$((f_pe + i))
    ./build/bin/udma_demo ${n_pes} ${pe_id} tcp://${ipport} ${g_npus} ${f_pe} ${f_npu} $test_type & # pe ${pe_id}
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
