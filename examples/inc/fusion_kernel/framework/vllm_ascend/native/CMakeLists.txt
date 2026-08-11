cmake_minimum_required(VERSION 3.16)
project(inc_fusion_torch_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(TORCH_NPU_PATH "" CACHE PATH "directory containing the torch_npu package")
if(NOT TORCH_NPU_PATH)
    # Locate the package without importing it.  Some vLLM Ascend images emit
    # driver/profiler logs during import; capturing those lines as a CMake path
    # makes configuration fail and needlessly initializes the NPU runtime.
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
                "import importlib.util,pathlib; s=importlib.util.find_spec('torch_npu'); assert s and s.origin; print(pathlib.Path(s.origin).parent)"
        OUTPUT_VARIABLE TORCH_NPU_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
endif()
if(NOT EXISTS "${TORCH_NPU_PATH}/include")
    message(FATAL_ERROR
        "TORCH_NPU_PATH must name a torch_npu package with an include directory")
endif()

set(SHMEM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../../.." CACHE PATH
    "aclshmem source root")
set(ROUTE_PACK_LIBRARY "" CACHE FILEPATH
    "libinc_fusion_route_pack_kernel.so built for this container CANN")
if(NOT EXISTS "${ROUTE_PACK_LIBRARY}")
    message(FATAL_ERROR "ROUTE_PACK_LIBRARY must name the built route-pack library")
endif()
set(FUSION_API_LIBRARY "" CACHE FILEPATH
    "optional libinc_fusion_api.so enabling worker lifecycle/moe_out")

find_library(TORCH_NPU_LIBRARY torch_npu
    PATHS "${TORCH_NPU_PATH}/lib" NO_DEFAULT_PATH REQUIRED)

add_library(inc_fusion_torch_bridge SHARED inc_fusion_torch_bridge.cpp)
target_include_directories(inc_fusion_torch_bridge PRIVATE
    ${TORCH_INCLUDE_DIRS}
    "${TORCH_NPU_PATH}/include"
    "${SHMEM_ROOT}/examples/inc/fusion_kernel/ascend")
target_compile_options(inc_fusion_torch_bridge PRIVATE ${TORCH_CXX_FLAGS})
target_link_libraries(inc_fusion_torch_bridge PRIVATE
    ${TORCH_LIBRARIES} ${TORCH_NPU_LIBRARY} "${ROUTE_PACK_LIBRARY}")
if(FUSION_API_LIBRARY)
    if(NOT EXISTS "${FUSION_API_LIBRARY}")
        message(FATAL_ERROR
            "FUSION_API_LIBRARY was set but does not name an existing file")
    endif()
    target_sources(inc_fusion_torch_bridge PRIVATE
        inc_fusion_torch_runtime.cpp)
    set(ASCEND_INCLUDE_DIR "" CACHE PATH
        "directory containing acl/acl.h for the target CANN")
    if(NOT ASCEND_INCLUDE_DIR)
        find_path(ASCEND_INCLUDE_DIR acl/acl.h
            HINTS
                "$ENV{ASCEND_HOME_PATH}/include"
                "$ENV{ASCEND_HOME_PATH}/x86_64-linux/include"
                "/usr/local/Ascend/ascend-toolkit/latest/include"
                "/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/include"
                "/usr/local/Ascend/cann-9.1.0-beta.3/x86_64-linux/include")
    endif()
    if(NOT ASCEND_INCLUDE_DIR)
        message(FATAL_ERROR
            "ASCEND_INCLUDE_DIR must contain acl/acl.h for runtime bridge")
    endif()
    target_include_directories(inc_fusion_torch_bridge PRIVATE
        "${ASCEND_INCLUDE_DIR}")
    target_link_libraries(inc_fusion_torch_bridge PRIVATE
        "${FUSION_API_LIBRARY}")
    get_filename_component(FUSION_API_LIBRARY_DIR
        "${FUSION_API_LIBRARY}" DIRECTORY)
    set(INC_FUSION_RUNTIME_RPATH ";${FUSION_API_LIBRARY_DIR}")
    message(STATUS "single-INC worker runtime bridge: enabled")
else()
    set(INC_FUSION_RUNTIME_RPATH "")
    message(STATUS
        "single-INC worker runtime bridge: disabled (route bridge only)")
endif()
set_target_properties(inc_fusion_torch_bridge PROPERTIES
    BUILD_RPATH
        "${TORCH_NPU_PATH}/lib;$<TARGET_FILE_DIR:inc_fusion_torch_bridge>${INC_FUSION_RUNTIME_RPATH}"
    INSTALL_RPATH
        "${TORCH_NPU_PATH}/lib${INC_FUSION_RUNTIME_RPATH}")
