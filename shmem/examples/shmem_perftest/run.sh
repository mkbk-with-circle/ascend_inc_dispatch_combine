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

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))

# 默认测试类型为put
TEST_TYPE="put"
TEST_TYPE_PROVIDED="0"
# 默认数据类型为float
DATA_TYPE="float"
DATA_TYPE_PROVIDED="0"
# 默认核数范围
MIN_BLOCK_SIZE="32"
MAX_BLOCK_SIZE="32"
BLOCK_LIST=""
# 默认幂数范围
MIN_EXPONENT="3"
MAX_EXPONENT="20"
# 默认循环次数
LOOP_COUNT="1000"
# 默认UB size(KB)
UB_SIZE="16"
# 默认SHMEM内存类型: hbm/dram，仅shmem模式使用
MEMORY_TYPE="hbm"
# 批量提交粒度（仅 BW 路径，当前仅支持 udma_perftest）：0=全异步(默认)，1=同步
BATCH="0"
# 默认运行模式: ascendc/mte/udma/simt/all (all表示全跑)
MODE="all"
# 分析模式: none/plot/md (none表示都不生成, plot表示只生成图, md表示同时生成图和md), 默认none
ANALYSE_MODE="none"
# 默认RANK配置
PE_SIZE="2"
IPPORT="tcp://127.0.0.1:8760"
GNPU_NUM="2"
FIRST_NPU="0"
FIRST_PE="0"

function usage() {
    echo "Usage: bash $0 [options]"
    echo "Options:"
    echo "  -h|--help                 Show this help message"
    echo "  -t|--test-type <type>      设置测试类型 (put|get|ub2gm_local|ub2gm_remote|gm2ub_local|gm2ub_remote|all)"
    echo "  -d|--datatype <type>      设置数据类型 (float|int8|int16|int32|int64|uint8|uint16|uint32|uint64|char|all)"
    echo "  -b|--block-size <size>          设置核数"
    echo "  --block-range <min> <max>       设置连续核数范围"
    echo "  --block-list <b1,b2,...>        设置离散核数列表（ascendc/mte/simt_rma_perftest），如 2,4,6,8"
    echo "  -e|--exponent <exponent>        设置数据量的幂数"
    echo "  --exponent-range <min> <max>    设置数据量的幂数范围"
    echo "  --loop-count <count>            设置循环次数"
    echo "  --ub-size <size>                设置UB size(KB), 默认16"
    echo "  --memory-type <hbm|dram>        设置SHMEM内存类型, 默认hbm"
    echo "  --batch <N>                     BW 路径下每 N 次 *_nbi 后 quiet (0=loop_count 全异步, 1=同步, 默认 0；当前仅支持 udma_perftest)"
    echo "  -pes <size>                     设置PE大小"
    echo "  -ipport <ip:port>               设置IP端口"
    echo "  -gnpus <num>                   设置NPU数量"
    echo "  -fnpu <id>                     设置首个NPU ID"
    echo "  -fpe <id>                      设置首个PE ID"
    echo "  -m|--mode <ascendc|mte|udma|simt|all>    设置运行模式 (ascendc=只跑ascendc, mte=只跑mte, udma=只跑udma, simt=只跑simt, all=全跑), 默认all"
    echo "  -a|--analyse <none|plot|md>    设置分析模式 (none=不生成, plot=只生成图, md=同时生成图和md), 默认none"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 1
            ;;
        -t|--test-type)
            if [ -n "$2" ]; then
                TEST_TYPE="$2"
                TEST_TYPE_PROVIDED="1"
                shift 2
            else
                echo "Error: -t|--test-type requires a value."
                exit 1
            fi
            ;;
        -d|--datatype)
            if [ -n "$2" ]; then
                DATA_TYPE="$2"
                DATA_TYPE_PROVIDED="1"
                shift 2
            else
                echo "Error: -d|--datatype requires a value."
                exit 1
            fi
            ;;
        -b|--block-size)
            if [ -n "$2" ]; then
                MIN_BLOCK_SIZE="$2"
                MAX_BLOCK_SIZE="$2"
                BLOCK_LIST=""
                shift 2
            else
                echo "Error: -b|--block-size requires a value."
                exit 1
            fi
            ;;
        --block-range)
            if [ -n "$2" ] && [ -n "$3" ]; then
                MIN_BLOCK_SIZE="$2"
                MAX_BLOCK_SIZE="$3"
                BLOCK_LIST=""
                shift 3
            else
                echo "Error: --block-range requires two values."
                exit 1
            fi
            ;;
        --block-list)
            if [ -n "$2" ]; then
                BLOCK_LIST="$2"
                shift 2
            else
                echo "Error: --block-list requires a value."
                exit 1
            fi
            ;;
        -e|--exponent)
            if [ -n "$2" ]; then
                MIN_EXPONENT="$2"
                MAX_EXPONENT="$2"
                shift 2
            else
                echo "Error: -e|--exponent requires a value."
                exit 1
            fi
            ;;
        --exponent-range)
            if [ -n "$2" ] && [ -n "$3" ]; then
                MIN_EXPONENT="$2"
                MAX_EXPONENT="$3"
                shift 3
            else
                echo "Error: --exponent-range requires two values."
                exit 1
            fi
            ;;
        --loop-count)
            if [ -n "$2" ]; then
                LOOP_COUNT="$2"
                shift 2
            else
                echo "Error: --loop-count requires a value."
                exit 1
            fi
            ;;
        --ub-size)
            if [ -n "$2" ]; then
                UB_SIZE="$2"
                shift 2
            else
                echo "Error: --ub-size requires a value."
                exit 1
            fi
            ;;
        --memory-type)
            if [ -n "$2" ]; then
                MEMORY_TYPE="$2"
                shift 2
            else
                echo "Error: --memory-type requires a value."
                exit 1
            fi
            ;;
        --batch)
            if [ -n "$2" ]; then
                BATCH="$2"
                shift 2
            else
                echo "Error: --batch requires a value."
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
        -ipport)
            if [ -n "$2" ]; then
                IPPORT="$2"
                shift 2
            else
                echo "Error: -ipport requires a value."
                exit 1
            fi
            ;;
        -gnpus)
            if [ -n "$2" ]; then
                GNPU_NUM="$2"
                shift 2
            else
                echo "Error: -gnpus requires a value."
                exit 1
            fi
            ;;
        -fnpu)
            if [ -n "$2" ]; then
                FIRST_NPU="$2"
                shift 2
            else
                echo "Error: -fnpu requires a value."
                exit 1
            fi
            ;;
        -fpe)
            if [ -n "$2" ]; then
                FIRST_PE="$2"
                shift 2
            else
                echo "Error: -fpe requires a value."
                exit 1
            fi
            ;;
        -m|--mode)
            if [ -n "$2" ]; then
                MODE="$2"
                shift 2
            else
                echo "Error: -m|--mode requires a value."
                exit 1
            fi
            ;;
        -a|--analyse)
            if [ -n "$2" ]; then
                ANALYSE_MODE="$2"
                shift 2
            else
                echo "Error: -a|--analyse requires a value."
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknown option $1"
            usage
            exit 1
            ;;
    esac
done

VALID_MEMORY_TYPES="hbm dram"
if [[ ! " $VALID_MEMORY_TYPES " =~ " $MEMORY_TYPE " ]]; then
    echo "错误: SHMEM内存类型必须是 'hbm' 或 'dram'"
    exit 1
fi

if [ -n "$BLOCK_LIST" ] && [[ "$MODE" == "udma" || "$MODE" == "all" ]]; then
    echo "WARN: --block-list takes effect for ascendc/mte/simt_rma_perftest; udma_perftest always uses block_size=1."
fi

echo "=============================================="
echo "example/shmem_perftest"
echo "=============================================="
echo "测试类型: $TEST_TYPE"
echo "数据类型: $DATA_TYPE"
if [ -n "$BLOCK_LIST" ]; then
    echo "核数列表: $BLOCK_LIST"
else
    echo "核数范围: $MIN_BLOCK_SIZE-$MAX_BLOCK_SIZE"
fi
echo "幂数范围: $MIN_EXPONENT-$MAX_EXPONENT"
echo "循环次数: $LOOP_COUNT"
echo "UB size(KB): $UB_SIZE"
echo "SHMEM内存类型: $MEMORY_TYPE"
echo "运行模式: $MODE"
echo "PE_SIZE: $PE_SIZE, GNPU_NUM: $GNPU_NUM"
echo "FIRST_NPU: $FIRST_NPU, FIRST_PE: $FIRST_PE"
echo "=============================================="

DEVICE1=$FIRST_NPU
DEVICE2=$((FIRST_NPU + 1))

# 清理之前的output
echo -e "\n========== Cleaning previous outputs =========="
if [[ "$MODE" == "ascendc" || "$MODE" == "all" ]]; then
    rm -rf "${SCRIPT_DIR}/ascendc_perftest/output"
    echo "Cleaned: ${SCRIPT_DIR}/ascendc_perftest/output"
fi
if [[ "$MODE" == "mte" || "$MODE" == "all" ]]; then
    rm -rf "${SCRIPT_DIR}/mte_perftest/output"
    echo "Cleaned: ${SCRIPT_DIR}/mte_perftest/output"
fi
if [[ "$MODE" == "udma" || "$MODE" == "all" ]]; then
    rm -rf "${SCRIPT_DIR}/udma_perftest/output"
    echo "Cleaned: ${SCRIPT_DIR}/udma_perftest/output"
fi
if [[ "$MODE" == "simt" || "$MODE" == "all" ]]; then
    rm -rf "${SCRIPT_DIR}/simt_rma_perftest/output"
    echo "Cleaned: ${SCRIPT_DIR}/simt_rma_perftest/output"
fi

# 清理外层output目录
rm -rf "${SCRIPT_DIR}/output"
echo "Cleaned: ${SCRIPT_DIR}/output"

if [[ "$MODE" == "ascendc" || "$MODE" == "all" ]]; then
    ASCENDC_VALID_TT="put get ub2gm_local ub2gm_remote gm2ub_local gm2ub_remote all"
    if [[ ! " $ASCENDC_VALID_TT " =~ " $TEST_TYPE " ]]; then
        echo "WARN: ascendc_perftest does not support -t $TEST_TYPE (supports: $ASCENDC_VALID_TT). Skipping ascendc_perftest."
    else
        echo -e "\n========== Running ascendc_perftest =========="
        if [ -n "$BLOCK_LIST" ]; then
            echo "Command: bash ${SCRIPT_DIR}/ascendc_perftest/run.sh -t \"$TEST_TYPE\" -d \"$DATA_TYPE\" --block-list \"$BLOCK_LIST\" --exponent-range \"$MIN_EXPONENT\" \"$MAX_EXPONENT\" --loop-count \"$LOOP_COUNT\" --device1 \"$DEVICE1\" --device2 \"$DEVICE2\" --ub-size \"$UB_SIZE\""
            bash "${SCRIPT_DIR}/ascendc_perftest/run.sh" -t "$TEST_TYPE" -d "$DATA_TYPE" --block-list "$BLOCK_LIST" --exponent-range "$MIN_EXPONENT" "$MAX_EXPONENT" --loop-count "$LOOP_COUNT" --device1 "$DEVICE1" --device2 "$DEVICE2" --ub-size "$UB_SIZE"
        else
            echo "Command: bash ${SCRIPT_DIR}/ascendc_perftest/run.sh -t \"$TEST_TYPE\" -d \"$DATA_TYPE\" --block-range \"$MIN_BLOCK_SIZE\" \"$MAX_BLOCK_SIZE\" --exponent-range \"$MIN_EXPONENT\" \"$MAX_EXPONENT\" --loop-count \"$LOOP_COUNT\" --device1 \"$DEVICE1\" --device2 \"$DEVICE2\" --ub-size \"$UB_SIZE\""
            bash "${SCRIPT_DIR}/ascendc_perftest/run.sh" -t "$TEST_TYPE" -d "$DATA_TYPE" --block-range "$MIN_BLOCK_SIZE" "$MAX_BLOCK_SIZE" --exponent-range "$MIN_EXPONENT" "$MAX_EXPONENT" --loop-count "$LOOP_COUNT" --device1 "$DEVICE1" --device2 "$DEVICE2" --ub-size "$UB_SIZE"
        fi
    fi
fi

if [[ "$MODE" == "mte" || "$MODE" == "all" ]]; then
    MTE_VALID_TT="put bi_put get bi_get all"
    if [[ ! " $MTE_VALID_TT " =~ " $TEST_TYPE " ]]; then
        echo "WARN: mte_perftest does not support -t $TEST_TYPE (supports: $MTE_VALID_TT). Skipping mte_perftest."
    else
        echo -e "\n========== Running mte_perftest =========="
        if [ -n "$BLOCK_LIST" ]; then
            echo "Command: bash ${SCRIPT_DIR}/mte_perftest/run.sh -t \"$TEST_TYPE\" -d \"$DATA_TYPE\" --block-list \"$BLOCK_LIST\" --exponent-range \"$MIN_EXPONENT\" \"$MAX_EXPONENT\" --loop-count \"$LOOP_COUNT\" -pes \"$PE_SIZE\" -ipport \"$IPPORT\" -gnpus \"$GNPU_NUM\" -fnpu \"$FIRST_NPU\" -fpe \"$FIRST_PE\" --ub-size \"$UB_SIZE\" --memory-type \"$MEMORY_TYPE\""
            bash "${SCRIPT_DIR}/mte_perftest/run.sh" -t "$TEST_TYPE" -d "$DATA_TYPE" --block-list "$BLOCK_LIST" --exponent-range "$MIN_EXPONENT" "$MAX_EXPONENT" --loop-count "$LOOP_COUNT" -pes "$PE_SIZE" -ipport "$IPPORT" -gnpus "$GNPU_NUM" -fnpu "$FIRST_NPU" -fpe "$FIRST_PE" --ub-size "$UB_SIZE" --memory-type "$MEMORY_TYPE"
        else
            echo "Command: bash ${SCRIPT_DIR}/mte_perftest/run.sh -t \"$TEST_TYPE\" -d \"$DATA_TYPE\" --block-range \"$MIN_BLOCK_SIZE\" \"$MAX_BLOCK_SIZE\" --exponent-range \"$MIN_EXPONENT\" \"$MAX_EXPONENT\" --loop-count \"$LOOP_COUNT\" -pes \"$PE_SIZE\" -ipport \"$IPPORT\" -gnpus \"$GNPU_NUM\" -fnpu \"$FIRST_NPU\" -fpe \"$FIRST_PE\" --ub-size \"$UB_SIZE\" --memory-type \"$MEMORY_TYPE\""
            bash "${SCRIPT_DIR}/mte_perftest/run.sh" -t "$TEST_TYPE" -d "$DATA_TYPE" --block-range "$MIN_BLOCK_SIZE" "$MAX_BLOCK_SIZE" --exponent-range "$MIN_EXPONENT" "$MAX_EXPONENT" --loop-count "$LOOP_COUNT" -pes "$PE_SIZE" -ipport "$IPPORT" -gnpus "$GNPU_NUM" -fnpu "$FIRST_NPU" -fpe "$FIRST_PE" --ub-size "$UB_SIZE" --memory-type "$MEMORY_TYPE"
        fi
    fi
fi

if [[ "$MODE" == "udma" || "$MODE" == "all" ]]; then
    UDMA_VALID_TT="put bi_put get bi_get put_signal all"
    if [[ ! " $UDMA_VALID_TT " =~ " $TEST_TYPE " ]]; then
        echo "WARN: udma_perftest does not support -t $TEST_TYPE (supports: $UDMA_VALID_TT). Skipping udma_perftest."
    else
        if [[ "$MEMORY_TYPE" != "hbm" ]]; then
            echo "WARN: udma_perftest only operates on HBM symmetric memory; --memory-type=$MEMORY_TYPE is not forwarded to it."
        fi
        echo -e "\n========== Running udma_perftest =========="
        echo "Command: bash ${SCRIPT_DIR}/udma_perftest/run.sh -t \"$TEST_TYPE\" -d \"$DATA_TYPE\" --exponent-range \"$MIN_EXPONENT\" \"$MAX_EXPONENT\" --loop-count \"$LOOP_COUNT\" -pes \"$PE_SIZE\" -ipport \"$IPPORT\" -gnpus \"$GNPU_NUM\" -fnpu \"$FIRST_NPU\" -fpe \"$FIRST_PE\" --ub-size \"$UB_SIZE\" --batch \"$BATCH\""
        bash "${SCRIPT_DIR}/udma_perftest/run.sh" -t "$TEST_TYPE" -d "$DATA_TYPE" --exponent-range "$MIN_EXPONENT" "$MAX_EXPONENT" --loop-count "$LOOP_COUNT" -pes "$PE_SIZE" -ipport "$IPPORT" -gnpus "$GNPU_NUM" -fnpu "$FIRST_NPU" -fpe "$FIRST_PE" --ub-size "$UB_SIZE" --batch "$BATCH"
    fi
fi

if [[ "$MODE" == "simt" || "$MODE" == "all" ]]; then
    SIMT_VALID_TT="put get none"
    SIMT_EXEC="${PROJECT_ROOT}/build/bin/simt_rma_perftest"
    if [[ ! -x "$SIMT_EXEC" ]]; then
        if [[ "$MODE" == "simt" ]]; then
            echo "ERROR: simt_rma_perftest binary not found or not executable (${SIMT_EXEC}). Build with SIMT support to run it."
            exit 1
        fi
        echo "WARN: simt_rma_perftest binary not found or not executable (${SIMT_EXEC}). Build with SIMT support to run it. Skipping simt_rma_perftest."
    elif [[ "$TEST_TYPE_PROVIDED" == "1" && "$TEST_TYPE" == "all" ]]; then
        echo "WARN: simt_rma_perftest does not support -t all because OP_TYPE is fixed at compile time. Skipping simt_rma_perftest."
    elif [[ "$TEST_TYPE_PROVIDED" == "1" && ! " $SIMT_VALID_TT " =~ " $TEST_TYPE " ]]; then
        echo "WARN: simt_rma_perftest does not support -t $TEST_TYPE (supports: $SIMT_VALID_TT). Skipping simt_rma_perftest."
    elif [[ "$DATA_TYPE_PROVIDED" == "1" && "$DATA_TYPE" == "all" ]]; then
        echo "WARN: simt_rma_perftest does not support -d all because DATA_SIZE is fixed at compile time. Skipping simt_rma_perftest."
    else
        SIMT_ARGS=()
        if [[ "$TEST_TYPE_PROVIDED" == "1" ]]; then
            SIMT_ARGS+=("-t" "$TEST_TYPE")
        fi
        if [[ "$DATA_TYPE_PROVIDED" == "1" ]]; then
            SIMT_ARGS+=("-d" "$DATA_TYPE")
        fi
        # 核数集合：--block-list（若指定）优先，否则用连续区间。
        if [ -n "$BLOCK_LIST" ]; then
            SIMT_ARGS+=("--block-list" "$BLOCK_LIST")
        else
            SIMT_ARGS+=("--block-range" "$MIN_BLOCK_SIZE" "$MAX_BLOCK_SIZE")
        fi
        echo -e "\n========== Running simt_rma_perftest =========="
        echo "Command: bash ${SCRIPT_DIR}/simt_rma_perftest/run.sh ${SIMT_ARGS[*]} --exponent-range \"$MIN_EXPONENT\" \"$MAX_EXPONENT\" --loop-count \"$LOOP_COUNT\" -pes \"$PE_SIZE\" -ipport \"$IPPORT\" -gnpus \"$GNPU_NUM\" -fnpu \"$FIRST_NPU\" -fpe \"$FIRST_PE\" --ub-size \"$UB_SIZE\""
        bash "${SCRIPT_DIR}/simt_rma_perftest/run.sh" "${SIMT_ARGS[@]}" --exponent-range "$MIN_EXPONENT" "$MAX_EXPONENT" --loop-count "$LOOP_COUNT" -pes "$PE_SIZE" -ipport "$IPPORT" -gnpus "$GNPU_NUM" -fnpu "$FIRST_NPU" -fpe "$FIRST_PE" --ub-size "$UB_SIZE"
    fi
fi

# 拷贝output到外层output目录
echo -e "\n========== Copying outputs =========="
mkdir -p "${SCRIPT_DIR}/output"

if [[ "$MODE" == "ascendc" || "$MODE" == "all" ]]; then
    if [ -d "${SCRIPT_DIR}/ascendc_perftest/output" ]; then
        mkdir -p "${SCRIPT_DIR}/output/ascendc_perftest"
        cp -r "${SCRIPT_DIR}/ascendc_perftest/output"/* "${SCRIPT_DIR}/output/ascendc_perftest/"
        echo "Copied: ascendc_perftest -> ${SCRIPT_DIR}/output/ascendc_perftest"
    fi
fi

if [[ "$MODE" == "mte" || "$MODE" == "all" ]]; then
    if [ -d "${SCRIPT_DIR}/mte_perftest/output" ]; then
        mkdir -p "${SCRIPT_DIR}/output/mte_perftest"
        cp -r "${SCRIPT_DIR}/mte_perftest/output"/* "${SCRIPT_DIR}/output/mte_perftest/"
        echo "Copied: mte_perftest -> ${SCRIPT_DIR}/output/mte_perftest"
    fi
fi

if [[ "$MODE" == "udma" || "$MODE" == "all" ]]; then
    if [ -d "${SCRIPT_DIR}/udma_perftest/output" ]; then
        mkdir -p "${SCRIPT_DIR}/output/udma_perftest"
        cp -r "${SCRIPT_DIR}/udma_perftest/output"/* "${SCRIPT_DIR}/output/udma_perftest/"
        echo "Copied: udma_perftest -> ${SCRIPT_DIR}/output/udma_perftest"
    fi
fi

if [[ "$MODE" == "simt" || "$MODE" == "all" ]]; then
    if [ -d "${SCRIPT_DIR}/simt_rma_perftest/output" ]; then
        mkdir -p "${SCRIPT_DIR}/output/simt_rma_perftest"
        cp -r "${SCRIPT_DIR}/simt_rma_perftest/output"/* "${SCRIPT_DIR}/output/simt_rma_perftest/"
        echo "Copied: simt_rma_perftest -> ${SCRIPT_DIR}/output/simt_rma_perftest"
    fi
fi

echo -e "\n=============================================="
echo "All tests completed!"
echo "Output directory: ${SCRIPT_DIR}/output"
echo "=============================================="

# 运行 perf_data_process.py 分别处理 output 下的子目录
PERF_SCRIPT="${SCRIPT_DIR}/../utils/perf_data_process.py"
if [ -f "${PERF_SCRIPT}" ]; then
    if [ "$ANALYSE_MODE" = "plot" ] || [ "$ANALYSE_MODE" = "md" ]; then
        echo -e "\n========== Generating performance charts =========="
        cmd_args=("-d" "${SCRIPT_DIR}/output")
        if [ "$ANALYSE_MODE" = "md" ]; then
            # md模式，图和md都生成
            true
        else
            # plot模式，只生成图，不生成md
            cmd_args+=("--no-markdown")
        fi
        python3 "${PERF_SCRIPT}" "${cmd_args[@]}"
    fi
fi
