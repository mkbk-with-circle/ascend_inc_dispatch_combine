# Minimal current INC delivery: Single-INC Pull V2 only.
if(NOT USE_EXAMPLES)
    return()
endif()

set(INC_DC_PULL_V2_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch_combine/single_inc/pull_combine)
set(INC_DC_PLATFORM_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch_combine/common/platform)
set(INC_HOST_INCLUDES
    ${PROJECT_SOURCE_DIR}/include
    ${INC_DC_PULL_V2_DIR}
    ${INC_DC_PLATFORM_DIR}
    ${PROJECT_SOURCE_DIR}/examples/utils
    ${PROJECT_SOURCE_DIR}/src/host)
set(INC_DEVICE_INCLUDES
    ${PROJECT_SOURCE_DIR}/include
    ${INC_DC_PULL_V2_DIR}
    ${INC_DC_PLATFORM_DIR}
    ${PROJECT_SOURCE_DIR}/examples/utils
    ${PROJECT_SOURCE_DIR}/src/device
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

add_subdirectory(dispatch_combine/single_inc/pull_combine)
