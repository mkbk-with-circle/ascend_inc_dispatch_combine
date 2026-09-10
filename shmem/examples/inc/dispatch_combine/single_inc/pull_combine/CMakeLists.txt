# Current single-INC Pull V2 implementation. Historical V1/Fusion sources are
# available from `archive/pre-minimal-v1-fusion-20260904`, not this tree.

set(INC_DC_PULL_V2_DIR ${CMAKE_CURRENT_LIST_DIR})
set(INC_DC_PULL_V2_TEST_DIR
    ${CMAKE_CURRENT_LIST_DIR}/../../tests/single_inc)

add_library(inc_dc_source_partition_layout STATIC inc_dc_source_partition_layout.cpp)
inc_host_target(inc_dc_source_partition_layout)
target_include_directories(inc_dc_source_partition_layout PUBLIC ${INC_DC_PULL_V2_DIR})
add_executable(inc_dc_source_partition_layout_tests tests/test_inc_dc_source_partition_layout.cpp)
inc_host_target(inc_dc_source_partition_layout_tests)
target_compile_options(inc_dc_source_partition_layout_tests PRIVATE -UNDEBUG -Wall -Wextra -Werror)
target_link_libraries(inc_dc_source_partition_layout_tests PRIVATE inc_dc_source_partition_layout)

add_library(inc_dc_partitioned_combine SHARED inc_dc_partitioned_combine.cpp)
inc_device_target(inc_dc_partitioned_combine)
target_include_directories(inc_dc_partitioned_combine PRIVATE ${INC_DC_PULL_V2_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../../common/platform)

add_library(inc_dc_pull_dispatch_v2_protocol STATIC
    inc_dc_pull_dispatch_v2.cpp)
inc_host_target(inc_dc_pull_dispatch_v2_protocol)
target_include_directories(inc_dc_pull_dispatch_v2_protocol PUBLIC
    ${INC_DC_PULL_V2_DIR})

add_library(inc_dc_pull_combine_v2_protocol STATIC
    inc_dc_pull_combine_v2.cpp)
inc_host_target(inc_dc_pull_combine_v2_protocol)
target_include_directories(inc_dc_pull_combine_v2_protocol PUBLIC
    ${INC_DC_PULL_V2_DIR})
target_link_libraries(inc_dc_pull_combine_v2_protocol PUBLIC
    inc_dc_pull_dispatch_v2_protocol)

add_library(inc_dc_pull_dispatch_v2_device_kernel SHARED
    inc_dc_pull_dispatch_v2_device_kernel.cpp)
inc_device_target(inc_dc_pull_dispatch_v2_device_kernel)
target_include_directories(inc_dc_pull_dispatch_v2_device_kernel PRIVATE
    ${INC_DC_PULL_V2_DIR})

add_library(inc_dc_pull_combine_v2_device_kernel SHARED
    inc_dc_pull_combine_v2_device_kernel.cpp)
inc_device_target(inc_dc_pull_combine_v2_device_kernel)
target_include_directories(inc_dc_pull_combine_v2_device_kernel PRIVATE
    ${INC_DC_PULL_V2_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../../common/platform)

add_library(inc_dc_pull_v2_api STATIC inc_dc_pull_v2_api.cpp)
inc_host_target(inc_dc_pull_v2_api)
target_include_directories(inc_dc_pull_v2_api PUBLIC ${INC_DC_PULL_V2_DIR})
target_compile_options(inc_dc_pull_v2_api PRIVATE -Wall -Wextra -Werror)

add_executable(inc_dc_pull_dispatch_v2_tests
    ${INC_DC_PULL_V2_TEST_DIR}/test_inc_dc_pull_dispatch_v2.cpp)
inc_host_target(inc_dc_pull_dispatch_v2_tests)
target_compile_options(inc_dc_pull_dispatch_v2_tests PRIVATE -UNDEBUG)
target_link_libraries(inc_dc_pull_dispatch_v2_tests PRIVATE
    inc_dc_pull_dispatch_v2_protocol)

add_executable(inc_dc_pull_combine_v2_tests
    ${INC_DC_PULL_V2_TEST_DIR}/test_inc_dc_pull_combine_v2.cpp)
inc_host_target(inc_dc_pull_combine_v2_tests)
target_compile_options(inc_dc_pull_combine_v2_tests PRIVATE
    -UNDEBUG -Wall -Wextra -Werror)
target_link_libraries(inc_dc_pull_combine_v2_tests PRIVATE
    inc_dc_pull_combine_v2_protocol)

add_executable(inc_dc_pull_v2_api_tests
    ${INC_DC_PULL_V2_TEST_DIR}/test_inc_dc_pull_v2_api.cpp)
inc_host_target(inc_dc_pull_v2_api_tests)
target_compile_options(inc_dc_pull_v2_api_tests PRIVATE
    -UNDEBUG -Wall -Wextra -Werror)
target_link_libraries(inc_dc_pull_v2_api_tests PRIVATE inc_dc_pull_v2_api)

add_executable(inc_dc_pull_combine_v2_contract_tests
    inc_dc_pull_combine_v2_device_e2e.cpp)
inc_host_target(inc_dc_pull_combine_v2_contract_tests)
target_compile_options(inc_dc_pull_combine_v2_contract_tests PRIVATE
    -UNDEBUG -Wall -Wextra -Werror)
target_link_libraries(inc_dc_pull_combine_v2_contract_tests PRIVATE
    inc_dc_pull_combine_v2_protocol)

add_executable(inc_dc_pull_dispatch_v2_device_e2e
    inc_dc_pull_dispatch_v2_device_e2e.cpp)
inc_host_target(inc_dc_pull_dispatch_v2_device_e2e)
target_include_directories(inc_dc_pull_dispatch_v2_device_e2e PRIVATE
    ${INC_DC_PULL_V2_DIR})
target_compile_options(inc_dc_pull_dispatch_v2_device_e2e PRIVATE
    -Wall -Wextra -Werror -Wno-missing-field-initializers)
target_link_libraries(inc_dc_pull_dispatch_v2_device_e2e PRIVATE
    shmem inc_dc_pull_dispatch_v2_protocol
    inc_dc_pull_dispatch_v2_device_kernel)
add_dependencies(inc_dc_pull_dispatch_v2_device_e2e
    aclshmem_bootstrap_config_store)

add_executable(inc_dc_pull_combine_v2_npu_e2e
    inc_dc_pull_combine_v2_npu_e2e.cpp)
inc_host_target(inc_dc_pull_combine_v2_npu_e2e)
target_include_directories(inc_dc_pull_combine_v2_npu_e2e PRIVATE
    ${INC_DC_PULL_V2_DIR})
target_compile_options(inc_dc_pull_combine_v2_npu_e2e PRIVATE
    -Wall -Wextra -Werror -Wno-missing-field-initializers)
target_link_libraries(inc_dc_pull_combine_v2_npu_e2e PRIVATE
    shmem inc_dc_pull_dispatch_v2_protocol
    inc_dc_pull_combine_v2_protocol inc_dc_pull_combine_v2_device_kernel)
add_dependencies(inc_dc_pull_combine_v2_npu_e2e
    aclshmem_bootstrap_config_store)

add_executable(inc_dc_source_partition_device_e2e inc_dc_source_partition_device_e2e.cpp)
inc_host_target(inc_dc_source_partition_device_e2e)
target_include_directories(inc_dc_source_partition_device_e2e PRIVATE ${INC_DC_PULL_V2_DIR})
target_compile_options(inc_dc_source_partition_device_e2e PRIVATE
    -Wall -Wextra -Werror -Wno-missing-field-initializers)
target_link_libraries(inc_dc_source_partition_device_e2e PRIVATE shmem
    inc_dc_source_partition_layout inc_dc_pull_dispatch_v2_protocol inc_dc_pull_combine_v2_protocol
    inc_dc_pull_dispatch_v2_device_kernel inc_dc_partitioned_combine)
add_dependencies(inc_dc_source_partition_device_e2e aclshmem_bootstrap_config_store)

add_executable(inc_dc_pull_v2_same_session_probe
    inc_dc_pull_v2_same_session_probe.cpp)
inc_host_target(inc_dc_pull_v2_same_session_probe)
target_include_directories(inc_dc_pull_v2_same_session_probe PRIVATE
    ${INC_DC_PULL_V2_DIR})
target_compile_options(inc_dc_pull_v2_same_session_probe PRIVATE
    -Wall -Wextra -Werror -Wno-missing-field-initializers)
target_link_libraries(inc_dc_pull_v2_same_session_probe PRIVATE
    shmem inc_dc_pull_dispatch_v2_protocol inc_dc_pull_combine_v2_protocol
    inc_dc_pull_dispatch_v2_device_kernel
    inc_dc_pull_combine_v2_device_kernel)
add_dependencies(inc_dc_pull_v2_same_session_probe
    aclshmem_bootstrap_config_store)

add_executable(inc_dc_pull_v2_api_example inc_dc_pull_v2_api_example.cpp)
inc_host_target(inc_dc_pull_v2_api_example)
target_compile_options(inc_dc_pull_v2_api_example PRIVATE
    -Wall -Wextra -Werror)
target_link_libraries(inc_dc_pull_v2_api_example PRIVATE inc_dc_pull_v2_api)
