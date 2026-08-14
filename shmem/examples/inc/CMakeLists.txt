# Maintained INC delivery targets only.
# Retired AG/RS, dc_ab, dc_ll probes and phase gates are intentionally absent.
if(NOT USE_EXAMPLES)
    return()
endif()

set(INC_DC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dispatch_combine)
set(INC_DC_COMMON_DIR ${INC_DC_DIR}/common)
set(INC_DC_COMMON_API_DIR ${INC_DC_COMMON_DIR}/api)
set(INC_DC_COMMON_PROTOCOL_DIR ${INC_DC_COMMON_DIR}/protocol)
set(INC_DC_COMMON_PLATFORM_DIR ${INC_DC_COMMON_DIR}/platform)
set(INC_DC_SINGLE_DIR ${INC_DC_DIR}/single_inc)
set(INC_DC_SINGLE_DISPATCH_DIR ${INC_DC_SINGLE_DIR}/dispatch)
set(INC_DC_SINGLE_COMBINE_DIR ${INC_DC_SINGLE_DIR}/combine)
set(INC_DC_SINGLE_PLANNING_DIR ${INC_DC_SINGLE_DIR}/planning)
set(INC_DC_SINGLE_RUNTIME_DIR ${INC_DC_SINGLE_DIR}/runtime)
set(INC_FUSION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fusion_kernel/ascend)
set(INC_DC_PUBLIC_INCLUDES
    ${INC_DC_COMMON_API_DIR} ${INC_DC_COMMON_PROTOCOL_DIR}
    ${INC_DC_COMMON_PLATFORM_DIR} ${INC_DC_SINGLE_DISPATCH_DIR}
    ${INC_DC_SINGLE_COMBINE_DIR} ${INC_DC_SINGLE_PLANNING_DIR}
    ${INC_DC_SINGLE_RUNTIME_DIR} ${INC_DC_SINGLE_DIR}/inline_route
    ${INC_FUSION_DIR})
set(INC_HOST_INCLUDES
    ${PROJECT_SOURCE_DIR}/include ${INC_DC_PUBLIC_INCLUDES}
    ${PROJECT_SOURCE_DIR}/examples/utils ${PROJECT_SOURCE_DIR}/src/host)
set(INC_DEVICE_INCLUDES
    ${PROJECT_SOURCE_DIR}/include ${INC_DC_PUBLIC_INCLUDES}
    ${PROJECT_SOURCE_DIR}/examples/utils ${PROJECT_SOURCE_DIR}/src/device
    ${PROJECT_SOURCE_DIR}/src)

function(inc_host_target target)
    target_compile_options(${target} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(${target} PRIVATE ${INC_HOST_INCLUDES})
endfunction()

function(inc_device_target target)
    target_compile_options(${target} PRIVATE
        ${CMAKE_CCE_COMPILE_OPTIONS} ${CCE_AICORE_ARCH_VEC} ${SANITIZER_FLAGS})
    target_include_directories(${target} PRIVATE ${INC_DEVICE_INCLUDES})
    target_link_options(${target} PRIVATE
        --cce-fatobj-link ${CCE_AICORE_ARCH_VEC} ${SANITIZER_FLAGS})
endfunction()

# Shared framework-facing API.
add_library(inc_dc_framework_api SHARED
    dispatch_combine/common/api/inc_dc_framework_c_api.cpp
    dispatch_combine/common/api/inc_dc_chunk_planner.cpp
    dispatch_combine/common/api/inc_dc_dispatch_route_plan.cpp)
inc_host_target(inc_dc_framework_api)
target_include_directories(inc_dc_framework_api PUBLIC ${INC_DC_PUBLIC_INCLUDES})

add_library(inc_dc_easy_api SHARED
    dispatch_combine/common/api/inc_dc_easy_api.cpp
    dispatch_combine/single_inc/planning/inc_dc_single_inc_stream_plan_compiler.cpp)
inc_host_target(inc_dc_easy_api)
target_link_libraries(inc_dc_easy_api PUBLIC inc_dc_framework_api)
target_include_directories(inc_dc_easy_api PUBLIC ${INC_DC_PUBLIC_INCLUDES})

add_library(inc_dc_inference_api SHARED
    dispatch_combine/common/api/inc_dc_inference_api.cpp)
inc_host_target(inc_dc_inference_api)
target_link_libraries(inc_dc_inference_api PUBLIC inc_dc_easy_api)
target_include_directories(inc_dc_inference_api PUBLIC ${INC_DC_PUBLIC_INCLUDES})

# Recommended product-facing facade.  It only removes lifecycle boilerplate;
# all qualified V1 kernels and backend calls remain unchanged.
add_library(inc_dc_single_inc_api SHARED
    dispatch_combine/common/api/inc_dc_single_inc_api.cpp)
inc_host_target(inc_dc_single_inc_api)
target_link_libraries(inc_dc_single_inc_api PUBLIC inc_dc_inference_api)
target_include_directories(inc_dc_single_inc_api PUBLIC ${INC_DC_PUBLIC_INCLUDES})

# Transport-neutral token-carried route ABI.  This host-only contract gate
# has no effect on the qualified V1 device path while the V2 INC parser is
# brought up behind an explicit API.
add_executable(inc_dc_inline_route_protocol_tests
    dispatch_combine/tests/common/test_inc_dc_inline_route_protocol.cpp)
inc_host_target(inc_dc_inline_route_protocol_tests)
target_compile_options(inc_dc_inline_route_protocol_tests PRIVATE -UNDEBUG)

add_library(inc_dc_inline_route_parser STATIC
    dispatch_combine/common/protocol/inc_dc_inline_route_parser.cpp)
inc_host_target(inc_dc_inline_route_parser)
target_include_directories(inc_dc_inline_route_parser PUBLIC
    ${INC_DC_COMMON_PROTOCOL_DIR})

add_executable(inc_dc_inline_route_parser_tests
    dispatch_combine/tests/common/test_inc_dc_inline_route_parser.cpp)
inc_host_target(inc_dc_inline_route_parser_tests)
target_compile_options(inc_dc_inline_route_parser_tests PRIVATE -UNDEBUG)
target_link_libraries(inc_dc_inline_route_parser_tests PRIVATE
    inc_dc_inline_route_parser)

add_library(inc_dc_inline_route_journal STATIC
    dispatch_combine/common/protocol/inc_dc_inline_route_journal.cpp)
inc_host_target(inc_dc_inline_route_journal)
target_include_directories(inc_dc_inline_route_journal PUBLIC
    ${INC_DC_COMMON_PROTOCOL_DIR})

add_executable(inc_dc_inline_route_journal_tests
    dispatch_combine/tests/common/test_inc_dc_inline_route_journal.cpp)
inc_host_target(inc_dc_inline_route_journal_tests)
target_compile_options(inc_dc_inline_route_journal_tests PRIVATE -UNDEBUG)
target_link_libraries(inc_dc_inline_route_journal_tests PRIVATE
    inc_dc_inline_route_journal)

add_library(inc_dc_single_inc_inline_plan STATIC
    dispatch_combine/single_inc/inline_route/inc_dc_single_inc_inline_plan.cpp)
inc_host_target(inc_dc_single_inc_inline_plan)
target_include_directories(inc_dc_single_inc_inline_plan PUBLIC
    ${INC_DC_SINGLE_DIR}/inline_route)

add_executable(inc_dc_single_inc_inline_plan_tests
    dispatch_combine/tests/single_inc/test_inc_dc_single_inc_inline_plan.cpp)
inc_host_target(inc_dc_single_inc_inline_plan_tests)
target_compile_options(inc_dc_single_inc_inline_plan_tests PRIVATE -UNDEBUG)
target_link_libraries(inc_dc_single_inc_inline_plan_tests PRIVATE
    inc_dc_single_inc_inline_plan)

add_executable(inc_dc_single_inc_inline_protocol_tests
    dispatch_combine/tests/single_inc/test_inc_dc_single_inc_inline_protocol.cpp)
inc_host_target(inc_dc_single_inc_inline_protocol_tests)
target_compile_options(inc_dc_single_inc_inline_protocol_tests PRIVATE
    -UNDEBUG)
target_link_libraries(inc_dc_single_inc_inline_protocol_tests PRIVATE
    inc_dc_inline_route_parser inc_dc_inline_route_journal)

add_library(inc_dc_single_inc_inline_kernel SHARED
    dispatch_combine/single_inc/inline_route/inc_dc_single_inc_inline_kernel.cpp)
inc_device_target(inc_dc_single_inc_inline_kernel)

add_executable(inc_dc_single_inc_inline
    dispatch_combine/single_inc/inline_route/inc_dc_single_inc_inline_main.cpp)
inc_host_target(inc_dc_single_inc_inline)
target_link_libraries(inc_dc_single_inc_inline PRIVATE
    shmem inc_dc_single_inc_inline_kernel)
add_dependencies(inc_dc_single_inc_inline aclshmem_bootstrap_config_store)

# Single-kernel MoE fusion.  Its planner is deliberately independent of the
# standalone Dispatch/Combine host backends: both execution modes share wire
# semantics, while the fused worker consumes the protocol from device code.
add_library(inc_fusion_plan STATIC
    fusion_kernel/ascend/inc_fusion_plan.cpp
    fusion_kernel/ascend/inc_fusion_route.cpp)
inc_host_target(inc_fusion_plan)
target_include_directories(inc_fusion_plan PUBLIC ${INC_FUSION_DIR})

add_executable(inc_fusion_plan_tests
    fusion_kernel/ascend/tests/test_inc_fusion_plan.cpp)
inc_host_target(inc_fusion_plan_tests)
target_compile_options(inc_fusion_plan_tests PRIVATE -UNDEBUG)
target_link_libraries(inc_fusion_plan_tests PRIVATE inc_fusion_plan)

# Host-only benchmark contract.  It guards the 2x2 attribution experiment and
# contains no code on the inference/device hot path.
add_library(inc_fusion_benchmark STATIC
    fusion_kernel/ascend/inc_fusion_benchmark.cpp)
inc_host_target(inc_fusion_benchmark)
target_include_directories(inc_fusion_benchmark PUBLIC ${INC_FUSION_DIR})

add_executable(inc_fusion_benchmark_tests
    fusion_kernel/ascend/tests/test_inc_fusion_benchmark.cpp)
inc_host_target(inc_fusion_benchmark_tests)
target_compile_options(inc_fusion_benchmark_tests PRIVATE -UNDEBUG)
target_link_libraries(inc_fusion_benchmark_tests PRIVATE inc_fusion_benchmark)

# Device-only dense top-k route packing.  The small global count matrix is
# gathered by the framework; payload transport remains backend-specific.
add_library(inc_fusion_route_pack_kernel SHARED
    fusion_kernel/ascend/inc_fusion_route_pack_kernel.cpp)
inc_device_target(inc_fusion_route_pack_kernel)
target_include_directories(inc_fusion_route_pack_kernel PRIVATE
    ${INC_FUSION_DIR})

add_executable(inc_fusion_route_pack_tests
    fusion_kernel/ascend/tests/test_inc_fusion_route_pack.cpp)
inc_host_target(inc_fusion_route_pack_tests)
target_compile_options(inc_fusion_route_pack_tests PRIVATE -UNDEBUG)
target_link_libraries(inc_fusion_route_pack_tests PRIVATE
    inc_fusion_plan inc_fusion_route_pack_kernel ascendcl)

add_library(inc_fusion_protocol_kernel SHARED
    fusion_kernel/ascend/inc_fusion_protocol_kernel.cpp)
inc_device_target(inc_fusion_protocol_kernel)
target_include_directories(inc_fusion_protocol_kernel PRIVATE ${INC_FUSION_DIR})

add_library(inc_fusion_compute_probe_kernel SHARED
    fusion_kernel/ascend/inc_fusion_compute_probe_kernel.cpp)
if(NOT IS_NEW_VERSION)
    target_compile_definitions(inc_fusion_compute_probe_kernel PRIVATE
        INC_FUSION_LEGACY_OP_SYSTEM_RUN_CFG=1)
endif()
target_compile_options(inc_fusion_compute_probe_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS} ${CCE_AICORE_ARCH} ${SANITIZER_FLAGS})
target_include_directories(inc_fusion_compute_probe_kernel PRIVATE
    ${INC_DEVICE_INCLUDES} ${INC_FUSION_DIR}
    ${PROJECT_SOURCE_DIR}/3rdparty/catlass/include)
target_link_options(inc_fusion_compute_probe_kernel PRIVATE
    --cce-fatobj-link ${CCE_AICORE_ARCH} ${SANITIZER_FLAGS})

add_executable(inc_fusion_compute_tests
    fusion_kernel/ascend/tests/test_inc_fusion_compute.cpp)
inc_host_target(inc_fusion_compute_tests)
target_link_libraries(inc_fusion_compute_tests PRIVATE
    inc_fusion_compute_probe_kernel ascendcl)

add_library(inc_fusion_kernel SHARED
    fusion_kernel/ascend/inc_fusion_kernel.cpp)
if(NOT IS_NEW_VERSION)
    target_compile_definitions(inc_fusion_kernel PRIVATE
        INC_FUSION_LEGACY_OP_SYSTEM_RUN_CFG=1)
endif()
target_compile_options(inc_fusion_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS} ${CCE_AICORE_ARCH} ${SANITIZER_FLAGS})
target_include_directories(inc_fusion_kernel PRIVATE
    ${INC_DEVICE_INCLUDES} ${INC_FUSION_DIR}
    ${PROJECT_SOURCE_DIR}/3rdparty/catlass/include)
target_link_options(inc_fusion_kernel PRIVATE
    --cce-fatobj-link ${CCE_AICORE_ARCH} ${SANITIZER_FLAGS})

add_executable(inc_fusion_e2e
    fusion_kernel/ascend/tests/inc_fusion_e2e_main.cpp)
inc_host_target(inc_fusion_e2e)
target_link_libraries(inc_fusion_e2e PRIVATE
    shmem inc_fusion_plan inc_fusion_kernel inc_fusion_api ascendcl)
add_dependencies(inc_fusion_e2e aclshmem_bootstrap_config_store)

add_library(inc_fusion_api SHARED
    fusion_kernel/ascend/inc_fusion_api.cpp)
inc_host_target(inc_fusion_api)
target_include_directories(inc_fusion_api PUBLIC ${INC_FUSION_DIR})
target_link_libraries(inc_fusion_api PUBLIC
    inc_fusion_plan inc_fusion_kernel ascendcl)

add_executable(inc_fusion_api_tests
    fusion_kernel/ascend/tests/test_inc_fusion_api.cpp)
inc_host_target(inc_fusion_api_tests)
target_compile_options(inc_fusion_api_tests PRIVATE -UNDEBUG)
target_link_libraries(inc_fusion_api_tests PRIVATE inc_fusion_api)

# Small public-C-API examples for integrators.  The directory owns its targets
# so examples stay readable without growing this delivery manifest further.
add_subdirectory(fusion_kernel/examples)

# Single-INC Dispatch.
add_library(inc_dc_single_inc_stream_kernel SHARED
    dispatch_combine/single_inc/dispatch/inc_dc_single_inc_stream_kernel.cpp)
inc_device_target(inc_dc_single_inc_stream_kernel)

add_executable(inc_dc_single_inc_stream
    dispatch_combine/single_inc/dispatch/inc_dc_single_inc_stream_main.cpp)
inc_host_target(inc_dc_single_inc_stream)
target_link_libraries(inc_dc_single_inc_stream PRIVATE
    shmem inc_dc_single_inc_stream_kernel)
add_dependencies(inc_dc_single_inc_stream aclshmem_bootstrap_config_store)

add_library(inc_dc_native_dispatch_backend SHARED
    dispatch_combine/single_inc/runtime/inc_dc_native_dispatch_backend.cpp)
inc_host_target(inc_dc_native_dispatch_backend)
target_link_libraries(inc_dc_native_dispatch_backend PUBLIC
    shmem inc_dc_easy_api inc_dc_single_inc_stream_kernel)
target_include_directories(inc_dc_native_dispatch_backend PUBLIC
    ${INC_DC_PUBLIC_INCLUDES})

# Single-INC Combine and plan compilation.
add_library(inc_dc_sv2_dyn_csr_combine_kernel SHARED
    dispatch_combine/single_inc/combine/inc_dc_combine_kernel.cpp)
inc_device_target(inc_dc_sv2_dyn_csr_combine_kernel)

add_library(inc_dc_single_inc_combine_plan STATIC
    dispatch_combine/single_inc/planning/inc_dc_single_inc_combine_plan_compiler.cpp
    dispatch_combine/single_inc/runtime/inc_dc_native_expert_layout_adapter.cpp
    dispatch_combine/single_inc/runtime/inc_dc_native_combine_workspace.cpp
    dispatch_combine/single_inc/combine/inc_dc_combine_logical_plan.cpp
    dispatch_combine/single_inc/combine/inc_dc_combine_topology.cpp
    dispatch_combine/single_inc/combine/inc_dc_combine_plan_compiler.cpp)
inc_host_target(inc_dc_single_inc_combine_plan)
target_link_libraries(inc_dc_single_inc_combine_plan PUBLIC inc_dc_easy_api)
target_include_directories(inc_dc_single_inc_combine_plan PUBLIC
    ${INC_DC_PUBLIC_INCLUDES})

set(INC_SINGLE_COMBINE_HOST_SOURCES
    dispatch_combine/single_inc/combine/inc_dc_combine_launcher.cpp
    dispatch_combine/single_inc/combine/inc_dc_combine_logical_plan.cpp
    dispatch_combine/single_inc/combine/inc_dc_combine_plan_wire.cpp
    dispatch_combine/single_inc/combine/inc_dc_combine_topology.cpp
    dispatch_combine/single_inc/combine/inc_dc_combine_plan_compiler.cpp)
add_executable(inc_dc_sv2_dyn_csr_combine ${INC_SINGLE_COMBINE_HOST_SOURCES})
inc_host_target(inc_dc_sv2_dyn_csr_combine)
target_link_libraries(inc_dc_sv2_dyn_csr_combine PRIVATE
    shmem inc_dc_sv2_dyn_csr_combine_kernel)
add_dependencies(inc_dc_sv2_dyn_csr_combine aclshmem_bootstrap_config_store)

add_executable(inc_dc_sv2_dyn_csr_combine_overlap ${INC_SINGLE_COMBINE_HOST_SOURCES})
inc_host_target(inc_dc_sv2_dyn_csr_combine_overlap)
target_link_libraries(inc_dc_sv2_dyn_csr_combine_overlap PRIVATE
    shmem inc_dc_sv2_dyn_csr_combine_kernel)
add_dependencies(inc_dc_sv2_dyn_csr_combine_overlap aclshmem_bootstrap_config_store)

add_library(inc_dc_native_combine_backend SHARED
    dispatch_combine/single_inc/runtime/inc_dc_native_combine_backend.cpp)
inc_host_target(inc_dc_native_combine_backend)
target_link_libraries(inc_dc_native_combine_backend PUBLIC
    shmem inc_dc_easy_api inc_dc_single_inc_combine_plan
    inc_dc_sv2_dyn_csr_combine_kernel)
target_include_directories(inc_dc_native_combine_backend PUBLIC
    ${INC_DC_PUBLIC_INCLUDES})

add_library(inc_dc_native_composite_backend STATIC
    dispatch_combine/single_inc/runtime/inc_dc_native_composite_backend.cpp)
inc_host_target(inc_dc_native_composite_backend)
target_link_libraries(inc_dc_native_composite_backend PUBLIC inc_dc_framework_api)
target_include_directories(inc_dc_native_composite_backend PUBLIC
    ${INC_DC_PUBLIC_INCLUDES})

add_library(inc_dc_native_inc_service STATIC
    dispatch_combine/single_inc/runtime/inc_dc_native_inc_service.cpp)
inc_host_target(inc_dc_native_inc_service)
target_link_libraries(inc_dc_native_inc_service PUBLIC inc_dc_single_inc_api)
target_include_directories(inc_dc_native_inc_service PUBLIC
    ${INC_DC_PUBLIC_INCLUDES})

add_executable(inc_dc_native_full_example
    dispatch_combine/single_inc/runtime/inc_dc_native_full_example_main.cpp)
inc_host_target(inc_dc_native_full_example)
target_link_libraries(inc_dc_native_full_example PRIVATE
    inc_dc_native_dispatch_backend inc_dc_native_combine_backend
    inc_dc_native_composite_backend inc_dc_native_inc_service)
add_dependencies(inc_dc_native_full_example aclshmem_bootstrap_config_store)

# Random token-plan generator used by the retained single-INC sweeps.
add_executable(inc_dc_os_random_plan
    dispatch_combine/single_inc/tools/inc_dc_os_random_plan.cpp
    dispatch_combine/single_inc/combine/inc_dc_combine_logical_plan.cpp
    dispatch_combine/single_inc/combine/inc_dc_combine_plan_wire.cpp)
inc_host_target(inc_dc_os_random_plan)

# Small product-facing host gates.
foreach(pair IN ITEMS
    "inc_dc_framework_c_api_tests;test_inc_dc_framework_c_api.cpp;inc_dc_framework_api;common"
    "inc_dc_easy_api_tests;test_inc_dc_easy_api.cpp;inc_dc_easy_api;common"
    "inc_dc_inference_api_tests;test_inc_dc_inference_api.cpp;inc_dc_single_inc_api;common"
    "inc_dc_chunk_planner_tests;test_inc_dc_chunk_planner.cpp;inc_dc_framework_api;common"
    "inc_dc_dispatch_route_plan_tests;test_inc_dc_dispatch_route_plan.cpp;inc_dc_framework_api;common"
    "inc_dc_single_inc_stream_plan_compiler_tests;test_inc_dc_single_inc_stream_plan_compiler.cpp;inc_dc_easy_api;single_inc"
    "inc_dc_single_inc_combine_plan_tests;test_inc_dc_single_inc_combine_plan.cpp;inc_dc_single_inc_combine_plan;single_inc"
    "inc_dc_native_composite_backend_tests;test_inc_dc_native_composite_backend.cpp;inc_dc_native_composite_backend;single_inc"
    "inc_dc_native_expert_layout_adapter_tests;test_inc_dc_native_expert_layout_adapter.cpp;inc_dc_single_inc_combine_plan;single_inc")
    list(GET pair 0 target)
    list(GET pair 1 source)
    list(GET pair 2 library)
    list(GET pair 3 test_product)
    add_executable(${target} dispatch_combine/tests/${test_product}/${source})
    inc_host_target(${target})
    target_compile_options(${target} PRIVATE -UNDEBUG)
    target_link_libraries(${target} PRIVATE ${library})
endforeach()

# The single maintained application example.  Its CPU mock backend keeps the
# short public API executable on development hosts without touching any NPU.
add_executable(inc_dc_single_inc_api_example
    dispatch_combine/common/examples/single_inc_api/inc_dc_single_inc_api_example.cpp)
inc_host_target(inc_dc_single_inc_api_example)
target_link_libraries(inc_dc_single_inc_api_example PRIVATE inc_dc_single_inc_api)

add_custom_target(inc_dc_framework_c_header_tests
    COMMAND ${CMAKE_C_COMPILER} -std=c11 -fsyntax-only
        -I${INC_DC_COMMON_API_DIR}
        ${INC_DC_DIR}/tests/common/test_inc_dc_framework_c_header.c VERBATIM)
add_custom_target(inc_dc_easy_c_header_tests
    COMMAND ${CMAKE_C_COMPILER} -std=c11 -fsyntax-only
        -I${INC_DC_COMMON_API_DIR}
        ${INC_DC_DIR}/tests/common/test_inc_dc_easy_c_header.c VERBATIM)
add_custom_target(inc_dc_inference_c_header_tests
    COMMAND ${CMAKE_C_COMPILER} -std=c11 -fsyntax-only
        -I${INC_DC_COMMON_API_DIR}
        ${INC_DC_DIR}/tests/common/test_inc_dc_inference_c_header.c VERBATIM)
add_custom_target(inc_dc_single_inc_c_header_tests
    COMMAND ${CMAKE_C_COMPILER} -std=c11 -fsyntax-only
        -I${INC_DC_COMMON_API_DIR}
        ${INC_DC_DIR}/tests/common/test_inc_dc_single_inc_c_header.c VERBATIM)
