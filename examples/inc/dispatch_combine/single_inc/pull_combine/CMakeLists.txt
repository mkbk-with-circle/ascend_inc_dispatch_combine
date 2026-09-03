# Experimental single-INC push-Dispatch / pull-Combine protocol v1.
# This subdirectory is deliberately independent of the qualified V1 backend
# and public SingleInc API until its device gates pass.

set(INC_DC_PULL_COMBINE_DIR ${CMAKE_CURRENT_LIST_DIR})
set(INC_DC_PULL_COMBINE_TEST_DIR
    ${CMAKE_CURRENT_LIST_DIR}/../../tests/single_inc)

# Pull-Dispatch V2 is isolated from the stable Push-Dispatch checkpoint until
# its host, device and performance gates all pass.
add_library(inc_dc_pull_dispatch_v2_protocol STATIC
    inc_dc_pull_dispatch_v2.cpp)
inc_host_target(inc_dc_pull_dispatch_v2_protocol)
target_include_directories(inc_dc_pull_dispatch_v2_protocol PUBLIC
    ${INC_DC_PULL_COMBINE_DIR})

add_executable(inc_dc_pull_dispatch_v2_tests
    ${INC_DC_PULL_COMBINE_TEST_DIR}/test_inc_dc_pull_dispatch_v2.cpp)
inc_host_target(inc_dc_pull_dispatch_v2_tests)
target_compile_options(inc_dc_pull_dispatch_v2_tests PRIVATE -UNDEBUG)
target_link_libraries(inc_dc_pull_dispatch_v2_tests PRIVATE
    inc_dc_pull_dispatch_v2_protocol)

add_library(inc_dc_pull_dispatch_v2_get_kernel SHARED
    inc_dc_pull_dispatch_v2_get_kernel.cpp)
inc_device_target(inc_dc_pull_dispatch_v2_get_kernel)
target_include_directories(inc_dc_pull_dispatch_v2_get_kernel PRIVATE
    ${INC_DC_PULL_COMBINE_DIR})

add_executable(inc_dc_pull_dispatch_v2_get_e2e
    inc_dc_pull_dispatch_v2_get_e2e.cpp)
inc_host_target(inc_dc_pull_dispatch_v2_get_e2e)
target_include_directories(inc_dc_pull_dispatch_v2_get_e2e PRIVATE
    ${INC_DC_PULL_COMBINE_DIR})
target_link_libraries(inc_dc_pull_dispatch_v2_get_e2e PRIVATE
    shmem inc_dc_pull_dispatch_v2_protocol
    inc_dc_pull_dispatch_v2_get_kernel)
add_dependencies(inc_dc_pull_dispatch_v2_get_e2e
    aclshmem_bootstrap_config_store)

add_library(inc_dc_pull_combine_protocol STATIC
    inc_dc_dynamic_journal.cpp
    inc_dc_endpoint_dispatch_packet.cpp
    inc_dc_pull_combine_dispatch.cpp
    inc_dc_pull_combine_plan.cpp
    inc_dc_pull_combine_state.cpp)
inc_host_target(inc_dc_pull_combine_protocol)
target_include_directories(inc_dc_pull_combine_protocol PUBLIC
    ${INC_DC_PULL_COMBINE_DIR})

foreach(test_name IN ITEMS plan state fuzz dispatch)
    add_executable(inc_dc_pull_combine_${test_name}_tests
        ${INC_DC_PULL_COMBINE_TEST_DIR}/test_inc_dc_pull_combine_${test_name}.cpp)
    inc_host_target(inc_dc_pull_combine_${test_name}_tests)
    target_compile_options(inc_dc_pull_combine_${test_name}_tests PRIVATE
        -UNDEBUG)
    target_link_libraries(inc_dc_pull_combine_${test_name}_tests PRIVATE
        inc_dc_pull_combine_protocol)
endforeach()

add_executable(inc_dc_endpoint_dispatch_packet_tests
    ${INC_DC_PULL_COMBINE_TEST_DIR}/test_inc_dc_endpoint_dispatch_packet.cpp)
inc_host_target(inc_dc_endpoint_dispatch_packet_tests)
target_compile_options(inc_dc_endpoint_dispatch_packet_tests PRIVATE
    -UNDEBUG)
target_link_libraries(inc_dc_endpoint_dispatch_packet_tests PRIVATE
    inc_dc_pull_combine_protocol)

add_executable(inc_dc_dynamic_journal_tests
    ${INC_DC_PULL_COMBINE_TEST_DIR}/test_inc_dc_dynamic_journal.cpp)
inc_host_target(inc_dc_dynamic_journal_tests)
target_compile_options(inc_dc_dynamic_journal_tests PRIVATE -UNDEBUG)
target_link_libraries(inc_dc_dynamic_journal_tests PRIVATE
    inc_dc_pull_combine_protocol)

# Correctness-first transport probe.  It intentionally calls the high-level
# SHMEM RMA API so 910B selects MTE/SDMA while a future 950 build may select
# UDMA without changing the protocol implementation.
add_library(inc_dc_pull_combine_transport_probe_kernel SHARED
    inc_dc_pull_combine_transport_probe_kernel.cpp)
inc_device_target(inc_dc_pull_combine_transport_probe_kernel)

add_executable(inc_dc_pull_combine_transport_probe
    inc_dc_pull_combine_transport_probe.cpp)
inc_host_target(inc_dc_pull_combine_transport_probe)
target_include_directories(inc_dc_pull_combine_transport_probe PRIVATE
    ${INC_DC_PULL_COMBINE_DIR})
target_link_libraries(inc_dc_pull_combine_transport_probe PRIVATE
    shmem inc_dc_pull_combine_transport_probe_kernel)
add_dependencies(inc_dc_pull_combine_transport_probe
    aclshmem_bootstrap_config_store)

# First device correctness closure for ready -> pull -> FP32 reduce ->
# selective push -> ACK.  It is a qualification binary, not the public API.
add_library(inc_dc_pull_combine_device_e2e_kernel SHARED
    inc_dc_pull_combine_device_e2e_kernel.cpp)
inc_device_target(inc_dc_pull_combine_device_e2e_kernel)
target_include_directories(inc_dc_pull_combine_device_e2e_kernel PRIVATE
    ${INC_DC_PULL_COMBINE_DIR})

add_executable(inc_dc_pull_combine_device_e2e
    inc_dc_pull_combine_device_e2e.cpp)
inc_host_target(inc_dc_pull_combine_device_e2e)
target_include_directories(inc_dc_pull_combine_device_e2e PRIVATE
    ${INC_DC_PULL_COMBINE_DIR})
target_link_libraries(inc_dc_pull_combine_device_e2e PRIVATE
    shmem inc_dc_pull_combine_device_e2e_kernel)
add_dependencies(inc_dc_pull_combine_device_e2e
    aclshmem_bootstrap_config_store)

# Endpoint-owned Dispatch qualification: worker PUT packet -> INC online
# parse/count transpose -> destination-deduplicated fan-out PUT.
add_library(inc_dc_endpoint_dispatch_device_e2e_kernel SHARED
    inc_dc_endpoint_dispatch_device_e2e_kernel.cpp)
inc_device_target(inc_dc_endpoint_dispatch_device_e2e_kernel)
target_include_directories(inc_dc_endpoint_dispatch_device_e2e_kernel PRIVATE
    ${INC_DC_PULL_COMBINE_DIR})

add_library(inc_dc_device_journal_index_kernel SHARED
    inc_dc_device_journal_index_kernel.cpp)
inc_device_target(inc_dc_device_journal_index_kernel)
target_include_directories(inc_dc_device_journal_index_kernel PRIVATE
    ${INC_DC_PULL_COMBINE_DIR})

add_library(inc_dc_sparse_combine_device_e2e_kernel SHARED
    inc_dc_sparse_combine_device_e2e_kernel.cpp)
inc_device_target(inc_dc_sparse_combine_device_e2e_kernel)
target_include_directories(inc_dc_sparse_combine_device_e2e_kernel PRIVATE
    ${INC_DC_PULL_COMBINE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../../common/platform)

# Thin asynchronous integration API.  It owns no buffers and adds no device
# synchronization; production runtimes keep their SHMEM session/workspace.
add_library(inc_dc_endpoint_device_api STATIC
    inc_dc_endpoint_device_api.cpp)
inc_host_target(inc_dc_endpoint_device_api)
target_include_directories(inc_dc_endpoint_device_api PUBLIC
    ${INC_DC_PULL_COMBINE_DIR})
target_link_libraries(inc_dc_endpoint_device_api PUBLIC
    inc_dc_endpoint_dispatch_device_e2e_kernel
    inc_dc_device_journal_index_kernel
    inc_dc_sparse_combine_device_e2e_kernel)

add_executable(inc_dc_endpoint_device_api_tests
    ${INC_DC_PULL_COMBINE_TEST_DIR}/test_inc_dc_endpoint_device_api.cpp)
inc_host_target(inc_dc_endpoint_device_api_tests)
target_compile_options(inc_dc_endpoint_device_api_tests PRIVATE -UNDEBUG)
target_link_libraries(inc_dc_endpoint_device_api_tests PRIVATE
    inc_dc_endpoint_device_api)

add_executable(inc_dc_endpoint_dispatch_device_e2e
    inc_dc_endpoint_dispatch_device_e2e.cpp)
inc_host_target(inc_dc_endpoint_dispatch_device_e2e)
target_include_directories(inc_dc_endpoint_dispatch_device_e2e PRIVATE
    ${INC_DC_PULL_COMBINE_DIR})
target_link_libraries(inc_dc_endpoint_dispatch_device_e2e PRIVATE
    shmem inc_dc_pull_combine_protocol inc_dc_endpoint_device_api)
add_dependencies(inc_dc_endpoint_dispatch_device_e2e
    aclshmem_bootstrap_config_store)
