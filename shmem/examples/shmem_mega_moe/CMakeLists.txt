# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

aclshmem_add_fusion_example(shmem_mega_moe main.cpp)

if(USE_EXAMPLES)
    if(NOT EXISTS "${CATLASS_ROOT}/include/catlass/gemm/block/block_mmad_mx_tla.hpp")
        message(FATAL_ERROR
            "ShmemMegaMoe requires an Ascend950-capable Catlass checkout. "
            "Set -DCATLASS_ROOT=<catlass-source-root>.")
    endif()

    set(SHMEM_MEGA_MOE_ASCENDC_ROOT "${ASCEND_HOME_PATH}/asc/include")
    if(NOT EXISTS "${SHMEM_MEGA_MOE_ASCENDC_ROOT}/basic_api/kernel_basic_intf.h")
        set(SHMEM_MEGA_MOE_ASCENDC_ROOT
            "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include")
    endif()
    if(NOT EXISTS "${SHMEM_MEGA_MOE_ASCENDC_ROOT}/basic_api/kernel_basic_intf.h")
        message(FATAL_ERROR "ShmemMegaMoe cannot find the CANN AscendC basic API headers.")
    endif()

    get_filename_component(SHMEM_MEGA_MOE_ASCENDC_IMPL_ROOT "${SHMEM_MEGA_MOE_ASCENDC_ROOT}" DIRECTORY)
    target_compile_options(shmem_mega_moe PRIVATE
        --cce-auto-sync=off
        -Wno-c++11-narrowing
        "SHELL:-Xhost-start -D__ASC_NPU_HOST__=1 -Xhost-end")
    target_compile_definitions(shmem_mega_moe PRIVATE SHMEM_MEGA_MOE_EXAMPLE_ARCH35=1)
    target_include_directories(shmem_mega_moe BEFORE PRIVATE
        ${PROJECT_SOURCE_DIR}/examples/shmem_mega_moe/include
        ${PROJECT_SOURCE_DIR}/examples/shmem_mega_moe/kernel
        ${PROJECT_SOURCE_DIR}/examples/shmem_mega_moe/kernel/arch35
        ${SHMEM_MEGA_MOE_ASCENDC_ROOT}
        ${SHMEM_MEGA_MOE_ASCENDC_IMPL_ROOT}
        ${SHMEM_MEGA_MOE_ASCENDC_IMPL_ROOT}/impl/c_api
        ${SHMEM_MEGA_MOE_ASCENDC_ROOT}/c_api)
endif()
