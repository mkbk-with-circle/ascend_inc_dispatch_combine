# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Compile host source files here with coverage feature ON
file(GLOB_RECURSE ACLSHMEM_HOST_FILES ${PROJECT_SOURCE_DIR}/src/host/*.cpp)
list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "python_wrapper")
list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/bootstrap")
if(NOT SOC_TYPE STREQUAL "Ascend950")
    list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/transport/device_udma")
endif()
if(NOT SOC_TYPE STREQUAL "Ascend950")
    list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/transport/topo")
else()
    list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/transport/topo/rootinfo/aclshmemi_hal")
endif()
if(NOT (ACLSHMEM_RDMA_SUPPORT AND SOC_TYPE STREQUAL "Ascend950"))
    list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/transport/device_rdma/device_rdma_transport_manager_v2")
endif()
if(USE_MSSANITIZER)
    list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/utils/mstx/mstx_mem_register_empty.cpp")
    message(STATUS "USE_MSSANITIZER=ON")
else()
    list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/utils/mstx/mstx_mem_register.cpp")
    message(STATUS "USE_MSSANITIZER=OFF")
endif()

file(GLOB_RECURSE TEST_HOST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp")
if(NOT SOC_TYPE STREQUAL "Ascend950")
    list(FILTER TEST_HOST_FILES EXCLUDE REGEX "topo/")
    list(FILTER TEST_HOST_FILES EXCLUDE REGEX "transport/device_udma/")
endif()
file(GLOB_RECURSE CONFIG_STORE_FILES ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/*.cpp)
file(GLOB_RECURSE UNDER_API_FILES ${PROJECT_SOURCE_DIR}/src/host/utils/under_api/*.cpp)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

include(CheckCXXSourceCompiles)
set(_ACLSHMEM_SAVED_CMAKE_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES}")
set(CMAKE_REQUIRED_INCLUDES
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/hcomm
)
unset(ACLSHMEM_UT_HCOMM_HAS_CHANNEL_NAME CACHE)
check_cxx_source_compiles("#include <hcomm/hcomm_res_defs.h>
int main()
{
    HcommChannelDesc desc{};
    desc.channelName = nullptr;
    return 0;
}"
    ACLSHMEM_UT_HCOMM_HAS_CHANNEL_NAME
)
set(CMAKE_REQUIRED_INCLUDES "${_ACLSHMEM_SAVED_CMAKE_REQUIRED_INCLUDES}")
unset(_ACLSHMEM_SAVED_CMAKE_REQUIRED_INCLUDES)

if(SOC_TYPE STREQUAL "Ascend950")
    file(GLOB_RECURSE TOPO_SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/host/transport/topo/*.cpp)
    list(FILTER TOPO_SOURCE_FILES EXCLUDE REGEX "root_info_generate")
    list(FILTER TOPO_SOURCE_FILES EXCLUDE REGEX "aclshmemi_hal")
    file(GLOB MOCK_TOPO_FILES "${CMAKE_CURRENT_SOURCE_DIR}/topo/mock_hal.cpp")
else()
    set(TOPO_SOURCE_FILES "")
    set(MOCK_TOPO_FILES "")
endif()

add_library(aclshmem_unittest_host OBJECT ${ACLSHMEM_HOST_FILES} ${TEST_HOST_FILES} ${CONFIG_STORE_FILES} ${TOPO_SOURCE_FILES} ${UNDER_API_FILES} ${MOCK_TOPO_FILES})

if(ACLSHMEM_SIMT_SUPPORT)
    target_compile_definitions(aclshmem_unittest_host PRIVATE USE_SIMT)
endif()
if(ACLSHMEM_UT_HCOMM_HAS_CHANNEL_NAME)
    target_compile_definitions(aclshmem_unittest_host PRIVATE ACLSHMEM_UT_HCOMM_HAS_CHANNEL_NAME=1)
    message(STATUS "HCOMM channelName detected: UDMA multi-QP data-path tests enabled")
else()
    message(STATUS "HCOMM channelName not detected: UDMA multi-QP data-path tests will pass without execution")
endif()

target_compile_options(aclshmem_unittest_host PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_compile_options(aclshmem_unittest_host PRIVATE -g -fprofile-arcs -ftest-coverage)
target_include_directories(aclshmem_unittest_host PRIVATE
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/src/host
    ${PROJECT_SOURCE_DIR}/src/device
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/include
    ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
    ${PROJECT_SOURCE_DIR}/tests/unittest/include
    ${PROJECT_SOURCE_DIR}/tests/unittest/include/unittest/utils/
    ${PROJECT_SOURCE_DIR}/tests/unittest/
    ${PROJECT_SOURCE_DIR}/tests/unittest/team/team/
    ${PROJECT_SOURCE_DIR}/tests/unittest/team/team_allgather/
    ${PROJECT_SOURCE_DIR}/src/host/entity
    ${PROJECT_SOURCE_DIR}/src/host/mem/heap
    ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store
    ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/include
    ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc
    ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc/common
    ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc/security
    ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc/under_api
    ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc/under_api/openssl
    ${PROJECT_SOURCE_DIR}/src/host/utils
    ${PROJECT_SOURCE_DIR}/src/host/utils/under_api
    ${PROJECT_SOURCE_DIR}/src/host/transport
    ${PROJECT_SOURCE_DIR}/src/host/transport/device_rdma
    ${PROJECT_SOURCE_DIR}/src/host/transport/device_sdma
    ${PROJECT_SOURCE_DIR}/src/host/transport/device_udma
    ${PROJECT_SOURCE_DIR}/src/host/transport/topo/rootinfo
    ${PROJECT_SOURCE_DIR}/src/host/mem/heap/driver
    ${PROJECT_SOURCE_DIR}/src/host/mem/heap/driver/userspace
)
if(SOC_TYPE STREQUAL "Ascend950")
    target_include_directories(aclshmem_unittest_host PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/json/single_include)
endif()

if(ACLSHMEM_SIMT_SUPPORT)
    target_include_directories(aclshmem_unittest_host PRIVATE
        ${PROJECT_SOURCE_DIR}/src/device_simt/
    )
endif()
