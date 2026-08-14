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

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
BIN="${PROJECT_ROOT}/build/bin/udma_qp_demo"

setup_shmem_dynamic_endpoints() {
    if [[ -z "${IPPORT:-}" ]]; then
        export IPPORT="tcp://127.0.0.1:$((27010 + RANDOM % 900))"
    fi
    if [[ -z "${SHMEM_UID_SESSION_ID:-}" ]]; then
        export SHMEM_UID_SESSION_ID="127.0.0.1:$((8899 + RANDOM % 900))"
    fi
}

warn_shmem_stale_processes() {
    if pgrep -f "torch_test_.*\.py" >/dev/null 2>&1; then
        echo "[WARN] 检测到仍在运行的 torch_test 进程，可能占用 SHMEM 端口。" >&2
    fi
}

setup_shmem_runtime_env() {
    if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
        export CANN_SET_ENV="${CANN_SET_ENV:-/home/developer/Ascend/cann-9.0.0/set_env.sh}"
        if [[ ! -f "${CANN_SET_ENV}" ]]; then
            echo "[ERROR] CANN set_env.sh not found: ${CANN_SET_ENV}" >&2
            return 1
        fi
        source "${CANN_SET_ENV}"
    fi
    if [[ ! -f "${PROJECT_ROOT}/install/set_env.sh" ]]; then
        echo "[ERROR] ${PROJECT_ROOT}/install/set_env.sh not found; build examples first" >&2
        return 1
    fi
    source "${PROJECT_ROOT}/install/set_env.sh"
    export LD_LIBRARY_PATH="${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
    setup_shmem_dynamic_endpoints
    warn_shmem_stale_processes
}

PES=2
QP_COUNT=2
OP=put
ELEMS=1048576
HEAP_MB=1024
FIRST_NPU=0
IPPORT_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -pes) PES="$2"; shift 2 ;;
        -qp_count) QP_COUNT="$2"; shift 2 ;;
        -op) OP="$2"; shift 2 ;;
        -elems) ELEMS="$2"; shift 2 ;;
        -heap_mb) HEAP_MB="$2"; shift 2 ;;
        -first_npu) FIRST_NPU="$2"; shift 2 ;;
        -ipport) IPPORT_ARG="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ -n "${IPPORT_ARG}" ]]; then
    export IPPORT="${IPPORT_ARG}"
fi
setup_shmem_runtime_env

if [[ ! -x "${BIN}" ]]; then
    echo "Binary not found: ${BIN}. Build with: bash scripts/build.sh -examples -soc_type Ascend950" >&2
    exit 1
fi

pids=()
for ((pe = 0; pe < PES; ++pe)); do
    "${BIN}" -pe "${pe}" -pes "${PES}" -qp_count "${QP_COUNT}" -op "${OP}" -elems "${ELEMS}" \
        -heap_mb "${HEAP_MB}" -first_npu "${FIRST_NPU}" -ipport "${IPPORT}" &
    pids+=("$!")
done

ret=0
for pid in "${pids[@]}"; do
    wait "${pid}" || ret=1
done
exit "${ret}"
