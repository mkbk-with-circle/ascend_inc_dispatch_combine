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
readonly CURRENT_DIR=$(pwd)
readonly SCRIPT_DIR=$(dirname $(readlink -f "$0"))
readonly PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
readonly EXAMPLES_DIR=${PROJECT_ROOT}/examples/
readonly BUILD_DIR=${PROJECT_ROOT}/build/

exec_name=$1

function run_allgather()
{
    echo "begin run allgather"
    if [[ ! -f ${BUILD_DIR}/bin/allgather ]]; then
        echo "allgather build output doesn't exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/allgather/
    cd ${cur_dir}
    bash run.sh -pes 2 -type int
    return $?
}

function run_kv_shuffle()
{
    echo "begin run kv_shuffle"
    if [[ ! -f ${BUILD_DIR}/bin/kv_shuffle ]]; then
        echo "kv_shuffle build output doesn't exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/kv_shuffle/
    cd ${cur_dir}
    bash scripts/run.sh 2
    return $?
}

function run_rdma_demo()
{
    echo "begin run rdma_demo"
    if [[ ! -f ${BUILD_DIR}/bin/rdma_demo ]]; then
        echo "rdma_demo build output doesn't exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/rdma_demo/
    cd ${cur_dir}
    bash run.sh
    return $?
}

function run_unuse_handlewait()
{
    echo "begin run unuse_handlewait"
    if [[ ! -f ${BUILD_DIR}/bin/unuse_handlewait ]]; then
        echo "unuse_handlewait build output doesn't exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/rdma_handlewait_test/unuse_handlewait/
    cd ${cur_dir}
    bash run.sh
    return $?
}

function run_use_handlewait()
{
    echo "begin run use_handlewait"
    if [[ ! -f ${BUILD_DIR}/bin/use_handlewait ]]; then
        echo "use_handlewait build output doesn't exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/rdma_handlewait_test/use_handlewait/
    cd ${cur_dir}
    bash run.sh
    return $?
}

function run_rdma_perftest_demo()
{
    echo "begin run rdma_perftest_demo"
    if [[ ! -f ${BUILD_DIR}/bin/rdma_perftest_demo ]]; then
        echo "rdma_perftest_demo build output doesn't exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/rdma_perftest_demo/
    cd ${cur_dir}
    bash run.sh
    return $?
}

function run_python_extesion()
{
    echo "begin run python extension"
    cur_dir=${EXAMPLES_DIR}/python_extension/
    cd $cur_dir
    bash run.sh
    return $?
}

function run_all()
{
    run_allgather || return 1
    run_kv_shuffle || return 7
    # run_rdma_demo || return 11
    # run_unuse_handlewait || return 12
    # run_use_handlewait || return 13
    # run_rdma_perftest_demo || return 14
    #run_python_extesion || return 15
    return 0
}

function main()
{
    case $exec_name in
        "")
            run_all
            ;;
        allgather)
            run_allgather || return 1
            ;;
        kv_shuffle)
            run_kv_shuffle || return 7
            ;;
        rdma_demo)  # not ready
            run_rdma_demo || return 11
            ;;
        unuse_handlewait)
            run_unuse_handlewait || return 12
            ;;
        use_handlewait)     # not ready
            run_use_handlewait || return 13
            ;;
        rdma_perftest_demo)  # not ready
            run_rdma_perftest_demo || return 14
            ;;
        python_extension)
            run_python_extesion || return 15
            ;;
        *)
            echo "unknown example name: ${exec_name}"
            ;;
    esac
    return 0
}

main
ret=$?
if [[ $ret -ne 0 ]]; then
    echo "run example failed return ${ret}"
else
    echo "run example finished"
fi
exit $ret
