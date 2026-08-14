# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_executable(one_multi_path main.cpp one_multi_path_kernel.cpp)
if(DEFINED ENABLE_ASCENDC_DUMP AND ENABLE_ASCENDC_DUMP)
    target_compile_definitions(one_multi_path PRIVATE ENABLE_ASCENDC_DUMP ASCENDC_DUMP=1 ASCENDC_DEBUG)
endif()
target_compile_options(one_multi_path PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} ${CCE_AICORE_ARCH} ${SANITIZER_FLAGS})
target_include_directories(one_multi_path PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/examples/one_multi_path
    ${PROJECT_SOURCE_DIR}/examples/utils
    $ENV{ASCEND_HOME_PATH}/tools/mssanitizer/include
)
target_link_options(one_multi_path PRIVATE --cce-fatobj-link ${CCE_AICORE_ARCH} ${SANITIZER_FLAGS})
if(DEFINED ENABLE_ASCENDC_DUMP AND ENABLE_ASCENDC_DUMP)
    target_link_libraries(one_multi_path PRIVATE ascend_dump)
endif()
target_link_libraries(one_multi_path PRIVATE shmem)
