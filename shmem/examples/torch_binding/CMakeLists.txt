# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if(NOT DEFINED Python3_EXECUTABLE)
    message(WARNING "Python3_EXECUTABLE is not defined, using default python3")
else()
    message("Using python: ${Python3_EXECUTABLE}")
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch;print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE _torch_error
    RESULT_VARIABLE _torch_query_result
)
if(NOT _torch_query_result EQUAL 0)
    message(FATAL_ERROR "Failed to locate torch via ${Python3_EXECUTABLE}. "
                        "Make sure torch is installed for this interpreter."
                        "\nError: ${_torch_error}")
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import os,torch_npu;print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE _torch_npu_error
    RESULT_VARIABLE _torch_npu_query_result
)
if(NOT _torch_npu_query_result EQUAL 0)
    message(FATAL_ERROR "Failed to locate torch_npu via ${Python3_EXECUTABLE}. "
                        "Make sure torch_npu is installed for this interpreter."
                        "\nError: ${_torch_npu_error}")
endif()

find_package(Torch REQUIRED)
message(STATUS "Found Torch: ${Torch_DIR}")
message(STATUS "Torch Libraries: ${TORCH_LIBRARIES}")
message(STATUS "Found torch_npu: ${TORCH_NPU_PATH}")

if(NOT DEFINED ACLSHMEM_ALL_KERNEL_TARGETS OR ACLSHMEM_ALL_KERNEL_TARGETS STREQUAL "")
    message(FATAL_ERROR "No kernel targets passed from parent! Please check collective example build.")
else()
    foreach(TARGET IN LISTS ACLSHMEM_ALL_KERNEL_TARGETS)
        message(STATUS "  - ${TARGET}")
    endforeach()
endif()

add_library(aclshmem_torch SHARED
    src/torch_bindings.cpp
)

set_target_properties(aclshmem_torch PROPERTIES
    PREFIX ""
)

if(NOT EXISTS "${TORCH_NPU_PATH}/lib")
    message(WARNING "torch_npu lib directory not found: ${TORCH_NPU_PATH}/lib")
endif()
if(NOT EXISTS "${TORCH_NPU_PATH}/include")
    message(WARNING "torch_npu include directory not found: ${TORCH_NPU_PATH}/include")
endif()

target_link_directories(aclshmem_torch PRIVATE
    ${CMAKE_BINARY_DIR}
    ${TORCH_NPU_PATH}/lib
)

target_link_libraries(aclshmem_torch PRIVATE
    ${ACLSHMEM_ALL_KERNEL_TARGETS}
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
    torch_npu shmem tiling_api
)

include_directories(
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_PATH}/include
    ${ACLSHMEM_SHARED_LIB_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../examples
    ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/utils
)

install(TARGETS aclshmem_torch DESTINATION lib COMPONENT aclshmem_torch)