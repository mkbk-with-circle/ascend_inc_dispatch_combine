#!/bin/bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should have received a copy of the License along with this program.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# --- 默认值 ---
PE_SIZE=2
IPPORT="tcp://127.0.0.1:8760"
GNPU_NUM=2
FIRST_NPU=0
FIRST_PE=0
BLOCK_MIN=32
BLOCK_MAX=32
BLOCK_LIST=""
LOOP_COUNT=1000
MIN_EXPONENT=3
MAX_EXPONENT=20
UB_SIZE=16
# 操作类型 / 数据类型：留空表示不传给二进制（按编译期 OP_TYPE/DATA_SIZE 运行）。
# 若显式指定，二进制会校验其是否与编译期常量一致，不一致则报错退出。
TEST_TYPE=""
DATA_TYPE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -pes)
            PE_SIZE="$2"
            if [[ "$GNPU_NUM" =~ ^[0-9]+$ && "$PE_SIZE" =~ ^[0-9]+$ && "$GNPU_NUM" -gt "$PE_SIZE" ]]; then
                GNPU_NUM="$PE_SIZE"
                echo "Because GNPU_NUM is greater than PE_SIZE, GNPU_NUM is assigned the value of PE_SIZE=${PE_SIZE}."
            fi
            shift 2
            ;;
        -ipport)            IPPORT="$2"; shift 2 ;;
        -gnpus)
            GNPU_NUM="$2"
            if [[ "$GNPU_NUM" =~ ^[0-9]+$ && "$PE_SIZE" =~ ^[0-9]+$ && "$GNPU_NUM" -gt "$PE_SIZE" ]]; then
                GNPU_NUM="$PE_SIZE"
                echo "Because GNPU_NUM is greater than PE_SIZE, GNPU_NUM is assigned the value of PE_SIZE=${PE_SIZE}."
            fi
            shift 2
            ;;
        -fnpu)              FIRST_NPU="$2"; shift 2 ;;
        -fpe)               FIRST_PE="$2"; shift 2 ;;
        -t|--test-type)     TEST_TYPE="$2"; shift 2 ;;
        -d|--datatype)      DATA_TYPE="$2"; shift 2 ;;
        -b|--block-size)    BLOCK_MIN="$2"; BLOCK_MAX="$2"; shift 2 ;;
        --block-range)      BLOCK_MIN="$2"; BLOCK_MAX="$3"; shift 3 ;;
        --block-list)       BLOCK_LIST="$2"; shift 2 ;;
        --loop-count)       LOOP_COUNT="$2"; shift 2 ;;
        -e|--exponent)      MIN_EXPONENT="$2"; MAX_EXPONENT="$2"; shift 2 ;;
        --exponent-range)   MIN_EXPONENT="$2"; MAX_EXPONENT="$3"; shift 3 ;;
        --ub-size)          UB_SIZE="$2"; shift 2 ;;
        --help)
            cat <<'EOF'
Usage: run.sh [options]

Launches a fixed 2-card (Active PE0 / Passive PE1) gm2gm RMA perf test.

Options:
  -pes <int>                Number of PEs, must be 2.                    (default: 2)
  -ipport <ip:port>         Bootstrap communication address.            (default: tcp://127.0.0.1:8760)
  -gnpus <int>              Number of NPUs used on this node.           (default: 2)
  -fnpu <int>               First NPU id; device = pe_id % gnpus + fnpu.(default: 0)
  -fpe <int>                First PE id. Kept for CLI compatibility; unused.(default: 0)
  -t|--test-type <put|get|none>
                            Operation type. Optional; if given it must match the
                            compile-time OP_TYPE or the binary exits with an error.
  -d|--datatype <float|int8|int16|int32|int64|uint8|uint16|uint32|uint64|char>
                            Data type. Optional; converted to DATA_SIZE (bits) and
                            must match the compile-time DATA_SIZE or the binary errors.
  -b|--block-size <int>     Cores (blocks) used per PE.                 (default: 32)
  --block-range <min> <max> Cores (blocks) sweep range, one CSV row per count. (default: 32 32)
  --block-list <b1,b2,...>  Explicit core counts, comma-separated (e.g. 2,4,8).
                            Overrides --block-size/--block-range.
  --loop-count <int>        Sampled iterations, averaged for results.   (default: 1000)
  -e|--exponent <int>       Single transfer-size exponent; size = 2^e bytes.
  --exponent-range <min> <max>  Transfer-size exponent range; size = 2^exp bytes. (default: 3 20)
  --ub-size <int>           Unified Buffer size per core in KB.         (default: 16)
                            Only affects the SIMD path; ignored in SIMT (default) mode.
  --help                    Show this message.

The test sweeps 2^min .. 2^max bytes, one CSV row per size.
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if ! [[ "$MIN_EXPONENT" =~ ^[0-9]+$ && "$MAX_EXPONENT" =~ ^[0-9]+$ ]]; then
    echo "Error: exponent values must be non-negative integers."
    exit 1
fi
if [[ "$MIN_EXPONENT" -lt 3 || "$MAX_EXPONENT" -gt 20 || "$MIN_EXPONENT" -gt "$MAX_EXPONENT" ]]; then
    echo "Error: exponent range must satisfy 3 <= min <= max <= 20."
    exit 1
fi
if ! [[ "$GNPU_NUM" =~ ^[0-9]+$ ]]; then
    echo "Error: -gnpus must be an integer."
    exit 1
fi
if [[ "$GNPU_NUM" -ne 2 ]]; then
    echo "Error: -gnpus must be 2 for simt_rma_perftest fixed 2-card Active/Passive model."
    exit 1
fi

# --- 环境与路径设置 ---
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname $(dirname $(dirname "$SCRIPT_DIR")))

EXAMPLE=simt_rma_perftest
EXECUTABLE="${PROJECT_ROOT}/build/bin/${EXAMPLE}"

export SHMEM_UID_SESSION_ID=127.0.0.1:8899
export ACLSHMEM_UID_SESSION_ID=127.0.0.1:8899
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH

cd "${SCRIPT_DIR}"

pids=()
cleanup() {
    echo -e "\n[Terminating] Caught Ctrl+C, killing background processes..."
    if [ ${#pids[@]} -ne 0 ]; then
        kill "${pids[@]}" 2>/dev/null
    fi
    exit 1
}
trap cleanup SIGINT SIGTERM
echo "Starting $GNPU_NUM processes..."

# 仅在用户显式指定时转发 --test-type/--datatype，否则交由二进制按编译期常量运行。
OPTIONAL_ARGS=()
if [[ -n "${TEST_TYPE}" ]]; then
    OPTIONAL_ARGS+=(--test-type "${TEST_TYPE}")
fi
if [[ -n "${DATA_TYPE}" ]]; then
    OPTIONAL_ARGS+=(--datatype "${DATA_TYPE}")
fi
# --block-list（若指定）优先于 --block-size/--block-range，由二进制据此决定核数集合。
if [[ -n "${BLOCK_LIST}" ]]; then
    OPTIONAL_ARGS+=(--block-list "${BLOCK_LIST}")
fi

for (( idx = 0; idx < ${GNPU_NUM}; idx = idx + 1 )); do
    "${EXECUTABLE}" \
        --pes "${PE_SIZE}" \
        --pe-id "${idx}" \
        --ipport "${IPPORT}" \
        --gnpus "${GNPU_NUM}" \
        --fpe "${FIRST_PE}" \
        --fnpu "${FIRST_NPU}" \
        --block-size-min "${BLOCK_MIN}" \
        --block-size-max "${BLOCK_MAX}" \
        --loop-count "${LOOP_COUNT}" \
        --bytes-in-exp-min "${MIN_EXPONENT}" \
        --bytes-in-exp-max "${MAX_EXPONENT}" \
        --ub-size "${UB_SIZE}" \
        "${OPTIONAL_ARGS[@]}" &

    pid=$!
    pids+=("$pid")
    echo "[Rank $idx] Started with PID $pid"
done

ret=0
for pid in "${pids[@]}"; do
    wait "$pid"
    cur_ret=$?
    if [[ $cur_ret -ne 0 ]]; then
        ret=$cur_ret
    fi
done

echo "All processes done. Exit code: $ret"
cd "${CURRENT_DIR}"
exit $ret
