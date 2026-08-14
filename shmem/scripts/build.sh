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
if [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
fi

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}

ascend_dir=$(dirname "$_ASCEND_INSTALL_PATH")
env_script_path_old="${ascend_dir}/set_env.sh"
env_script_path_new="${ascend_dir}/ascend-toolkit/set_env.sh"

if [ -n "$_ASCEND_INSTALL_PATH" ] && [ -f "$env_script_path_old" ] && [ -x "$env_script_path_old" ] && \
   [ -f "$env_script_path_new" ] && [ -x "$env_script_path_new" ]; then
    echo "[WARNING] Both old and new set_env.sh files are detected!"
    echo "          Old path: $env_script_path_old"
    echo "          New path: $env_script_path_new"
    echo "          The new path file will be used by priority!"
fi

if [ -n "$_ASCEND_INSTALL_PATH" ] && [ -f "$env_script_path_new" ] && [ -x "$env_script_path_new" ]; then
    source "$env_script_path_new"
elif [ -n "$_ASCEND_INSTALL_PATH" ] && [ -f "$env_script_path_old" ] && [ -x "$env_script_path_old" ]; then
    source "$env_script_path_old"
else
    if [ -z "$_ASCEND_INSTALL_PATH" ]; then
        echo "[WARNING] Environment variable _ASCEND_INSTALL_PATH is not set, cannot find set_env.sh script" >&2
    else
        echo "[WARNING] Valid set_env.sh script not found!" >&2
        echo "       Check path 1: $env_script_path_old (does not exist or is not executable)" >&2
        echo "       Check path 2: $env_script_path_new (does not exist or is not executable)" >&2
    fi
fi

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
# Package/run version: env VERSION overrides repo-root VERSION file
_DEFAULT_VERSION="$(tr -d '[:space:]' < "${PROJECT_ROOT}/VERSION" 2>/dev/null || true)"
if [ -z "${_DEFAULT_VERSION}" ]; then
    echo "[ERROR] Missing or empty ${PROJECT_ROOT}/VERSION" >&2
    exit 1
fi
export VERSION="${VERSION:-${_DEFAULT_VERSION}}"
unset _DEFAULT_VERSION
OUTPUT_DIR=$PROJECT_ROOT/install

rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
THIRD_PARTY_DIR=$PROJECT_ROOT/3rdparty
mkdir -p $THIRD_PARTY_DIR
RELEASE_DIR=$PROJECT_ROOT/ci/release
UNDER_DIR=$PROJECT_ROOT/src/

BUILD_TYPE=RELEASE
PYEXPAND_TYPE=OFF
PACKAGE=OFF
USE_CXX11_ABI=ON
USE_MSSANITIZER=OFF
ENABLE_EXAMPLES=OFF
PYEXPAND_EXAMPLE=OFF
BUILD_ALL=OFF

COMPILE_OPTIONS=""

COVERAGE_TYPE=""
GEN_DOC=OFF
SOC_TYPE=""
RDMA_BACKEND=""

cann_default_path="/usr/local/Ascend/ascend-toolkit"

cd ${PROJECT_ROOT}

function print_usage()
{
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -uttests                   Build with unit tests"
    echo "  -cann                      Enable CANN build"
    echo "  -debug                     Build in debug mode"
    echo "  -examples                  Build with examples"
    echo "  -enable_rdma               Enable RDMA support. For Ascend950, must be used with -rdma_backend"
    echo "  -enable_simt               Enable SIMT support"
    echo "  -enable_relay              Enable UDMA relay (detour) support. Requires -soc_type Ascend950 (UDMA)"
    echo "  -python_extension          Build Python extension"
    echo "  -python_example            Build Python example"
    echo "  -gendoc                    Generate documentation"
    echo "  -onlygendoc                Only generate documentation"
    echo "  -enable_ascendc_dump       Enable AscendC dump"
    echo "  -package                   Build package"
    echo "  -full                      Full build (all components)"
    echo "  -use_cxx11_abi1            Use CXX11 ABI=1"
    echo "  -use_cxx11_abi0            Use CXX11 ABI=0"
    echo "  -mssanitizer               Enable memory sanitizer"
    echo "  -soc_type <type>           Specify SOC type (e.g., Ascend950)"
    echo "  -rdma_backend <backend>    Specify RDMA backend (XSCALE or HNS_1825, only for Ascend950)"
    echo "                             Requires -enable_rdma and -soc_type Ascend950"
}

function fn_build()
{
    mkdir -p build && cd build
    # Relay depends on UDMA (Ascend950 only). When UDMA is off, drop the relay flag so this build
    # does not hit the top-level CMake FATAL_ERROR (relates to the -full path building examples /
    # unittests on non-950 SOCs while COMPILE_OPTIONS still carries -DACLSHMEM_RELAY_SUPPORT=ON).
    local build_compile_options="${COMPILE_OPTIONS}"
    if [ "$SOC_TYPE" != "Ascend950" ] && \
       [ -n "$(echo "$build_compile_options" | grep -o '\-DACLSHMEM_RELAY_SUPPORT=ON')" ]; then
        echo "[WARN] -enable_relay ignored for SOC_TYPE='${SOC_TYPE}': relay requires UDMA (Ascend950 only)."
        build_compile_options=$(echo "${build_compile_options}" | sed 's/-DACLSHMEM_RELAY_SUPPORT=ON//g')
    fi

    cmake $build_compile_options -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DUSE_CXX11_ABI=$USE_CXX11_ABI -DUSE_MSSANITIZER=$USE_MSSANITIZER -DSOC_TYPE=${SOC_TYPE} -DPYEXPAND_EXAMPLE=$PYEXPAND_EXAMPLE ..
    make install -j$(nproc)
    cd -
}

function _get_cann_build_version() {
    # 从 ASCEND_HOME_PATH 下的 version.info 中提取 CANN 版本号
    local cann_ver="UNKNOWN"
    if [ -n "${ASCEND_HOME_PATH:-}" ]; then
        for candidate in \
            "$ASCEND_HOME_PATH/opp/version.info" \
            "$ASCEND_HOME_PATH/version.info"; do
            if [ -f "$candidate" ]; then
                cann_ver=$(grep -m1 -iE "Version|version" "$candidate" | awk -F'=' '{print $NF}' | tr -d ' "' || echo "UNKNOWN")
                if [ "$cann_ver" != "UNKNOWN" ] && [ -n "$cann_ver" ]; then
                    break
                fi
            fi
        done
    fi
    echo "$cann_ver"
}

function _write_version_info() {
    local output_path="$1"

    local arch
    arch=$(uname -m)

    local branch
    branch=$(git symbolic-ref -q --short HEAD 2>/dev/null || echo "unknown")

    local commit_id
    commit_id=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

    local build_timestamp
    build_timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    local build_type_display="${BUILD_TYPE:-RELEASE}"
    local soc_display="${SOC_TYPE:-Ascend910B}"
    local rdma_display="${RDMA_BACKEND:-NONE}"
    # Wheel with dual RDMA backends: auto-detect from filesystem if both 950 variants exist
    if [ -d "${PROJECT_ROOT}/install/shmem/backends/950_hns1825" ] && \
       [ -d "${PROJECT_ROOT}/install/shmem/backends/950_xscale" ]; then
        rdma_display="XSCALE, HNS_1825"
    fi
    local cann_version=$(_get_cann_build_version)

    local has_root_info="否"
    if [ -f "${PROJECT_ROOT}/install/shmem/bin/root_info_generate" ]; then
        has_root_info="是"
    elif [ -f "${PROJECT_ROOT}/build/bin/root_info_generate" ]; then
        has_root_info="是"
    fi

    local udma_support="否"
    if grep -q 'SOC_TYPE:.*=Ascend950' "${PROJECT_ROOT}/build/CMakeCache.txt" 2>/dev/null; then
        udma_support="是"
    fi

    {
        echo "SHMEM Version : ${VERSION}"
        echo "Platform : ${arch}"
        echo "branch : ${branch}"
        echo "commit id : ${commit_id}"
        echo "Build Timestamp : ${build_timestamp}"
        echo "Build Type : ${build_type_display}"
        echo "SOC Type : ${soc_display}"
        echo "CANN Version : ${cann_version}"
        echo "RDMA Backend : ${rdma_display}"
        echo "Root Info Generate Tool : ${has_root_info}"
        echo "UDMA Support : ${udma_support}"
    } > "$output_path"
}

function fn_whl_build()
{
    echo "Python extension enabled. Building multi-SOC wheel (910 + 950)..."
    export SOC_TYPE=${SOC_TYPE}
    # setup.py reads COMPILE_OPTIONS from the environment and forwards it to CMake. Without
    # exporting it here, wheel builds silently drop every -D flag accumulated in COMPILE_OPTIONS
    # (e.g. -DACLSHMEM_RELAY_SUPPORT/-DACLSHMEM_RDMA_SUPPORT/-DACLSHMEM_SIMT_SUPPORT), producing a
    # wheel whose C++ build differs from the native fn_build output.
    export COMPILE_OPTIONS=${COMPILE_OPTIONS}
    cd "${PROJECT_ROOT}/src/python"
    rm -rf shmem.egg-info ${PROJECT_ROOT}/dist
    cd "${PROJECT_ROOT}"

    py_ver=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    arch=$(uname -m)
    if [ "$arch" = "x86_64" ]; then
        plat="linux_x86_64"
    elif [ "$arch" = "aarch64" ]; then
        plat="linux_aarch64"
    else
        plat="linux_${arch}"
    fi
    echo "Wheel tag: ${py_ver}-${py_ver}-${plat} (auto-detected)"

    # Clean install directory
    rm -rf "${PROJECT_ROOT}/install"

    # ============================================================
    # Build SOC 910 (default / Ascend910)
    # ============================================================
    # Relay is a UDMA-only (Ascend950) feature. The 910 backend is built with UDMA support OFF,
    # and the top-level CMake FATAL_ERRORs when relay is requested for a non-950 backend.
    # Strip the relay flag from the 910 backend options; the 950 backend below keeps it.
    COMPILE_OPTIONS_910=${COMPILE_OPTIONS}
    if [ -n "$(echo "$COMPILE_OPTIONS_910" | grep -o '\-DACLSHMEM_RELAY_SUPPORT=ON')" ]; then
        echo "[WARN] -enable_relay applied to the 950 backend only; the 910 wheel backend has UDMA disabled."
        COMPILE_OPTIONS_910=$(echo "${COMPILE_OPTIONS_910}" | sed 's/-DACLSHMEM_RELAY_SUPPORT=ON//g')
    fi

    # Determine step total early for correct [1/N] label
    local _step_total=4
    if [ -z "$(echo "$COMPILE_OPTIONS" | grep -o '\-DACLSHMEM_RDMA_SUPPORT=ON')" ]; then
        _step_total=3
    fi
    echo "===== [1/${_step_total}] Building backend: 910 ====="
    # Ascend910B backend covers Ascend910 A2/A3 series.
    # 910B 不依赖特定 RDMA Backend，剔除 -DACLSHMEM_RDMA_BACKEND 但保留 RDMA_SUPPORT
    local compile_opts_910=$COMPILE_OPTIONS
    compile_opts_910=$(echo "$compile_opts_910" | sed 's/-DACLSHMEM_RDMA_BACKEND=\S*//g')
    [ -d build ] && rm -rf build
    mkdir -p build && cd build
    cmake $compile_opts_910 \
        -DCMAKE_INSTALL_PREFIX=../install \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DUSE_CXX11_ABI=${USE_CXX11_ABI} \
        -DUSE_MSSANITIZER=${USE_MSSANITIZER} \
        -DSOC_TYPE="Ascend910B" \
        -DPYEXPAND_EXAMPLE=${PYEXPAND_EXAMPLE} \
        -DBUILD_PYTHON=ON \
        .. || { echo "[ERROR] cmake failed for backend 910"; exit 1; }
    make install -j$(nproc) || { echo "[ERROR] make install failed for backend 910"; exit 1; }
    # Copy shared libraries from install/shmem/lib/ to wheel backend directory
    mkdir -p ../install/shmem/backends/910
    cp ../install/shmem/lib/*.so ../install/shmem/backends/910/ || { echo "[ERROR] Failed to copy 910 backend libraries"; exit 1; }
    cd -
    echo "===== Backend 910 built ====="

    # nlohmann_json is required for Ascend950 UDMA（两轮 950 构建共用）
    fn_build_nlohmann_json || { echo "[ERROR] Failed to build nlohmann_json dependency for backend 950"; exit 1; }

    # 判断是否需要构建双 RDMA 后端（仅当 RDMA 启用时）
    local rdma_enabled=false
    if [ -n "$(echo "$COMPILE_OPTIONS" | grep -o '\-DACLSHMEM_RDMA_SUPPORT=ON')" ]; then
        rdma_enabled=true
    fi

    # ============================================================
    # Build SOC 950 - XSCALE backend（默认，backends/950/ + backends/950_xscale/）
    # ============================================================
    local step_950=2
    local step_total=4
    if [ "$rdma_enabled" != "true" ]; then
        step_total=3
    fi
    echo "===== [${step_950}/${step_total}] Building backend: 950 (XSCALE) ====="
    [ -d build ] && rm -rf build

    # _pyshmem.so is built only once in the 910 pass (BUILD_PYTHON=ON) and
    # reused for 950.  It is a thin SOC-agnostic pybind11 wrapper around the
    # uniform aclshmem_* C API surface.  At runtime, __init__.py preloads the
    # SOC-specific libshmem.so from backends/<soc>/, so the 910-compiled
    # _pyshmem.so resolves 950 symbols correctly on Ascend950 hardware.
    local compile_opts_950_xscale=${COMPILE_OPTIONS}
    compile_opts_950_xscale=$(echo "$compile_opts_950_xscale" | sed 's/-DACLSHMEM_RDMA_BACKEND=\S*//g')
    if [ "$rdma_enabled" = "true" ]; then
        compile_opts_950_xscale="${compile_opts_950_xscale} -DACLSHMEM_RDMA_BACKEND=XSCALE"
    fi
    mkdir -p build && cd build
    cmake ${compile_opts_950_xscale} \
        -DCMAKE_INSTALL_PREFIX=../install \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DUSE_CXX11_ABI=${USE_CXX11_ABI} \
        -DUSE_MSSANITIZER=${USE_MSSANITIZER} \
        -DSOC_TYPE="Ascend950" \
        -DPYEXPAND_EXAMPLE=${PYEXPAND_EXAMPLE} \
        -DBUILD_PYTHON=OFF \
        .. || { echo "[ERROR] cmake failed for backend 950 (XSCALE)"; exit 1; }
    make install -j$(nproc) || { echo "[ERROR] make install failed for backend 950 (XSCALE)"; exit 1; }
    # Copy shared libraries from install/shmem/lib/ to wheel backend directory
    mkdir -p ../install/shmem/backends/950
    cp ../install/shmem/lib/*.so ../install/shmem/backends/950/ || { echo "[ERROR] Failed to copy 950 (XSCALE) backend libraries (default)"; exit 1; }
    if [ "$rdma_enabled" = "true" ]; then
        mkdir -p ../install/shmem/backends/950_xscale
        cp ../install/shmem/lib/*.so ../install/shmem/backends/950_xscale/ || { echo "[ERROR] Failed to copy 950 (XSCALE) backend libraries (symmetry)"; exit 1; }
    fi
    cd -
    echo "===== Backend 950 (XSCALE) built ====="

    # ============================================================
    # Build SOC 950 - HNS_1825 backend（backends/950_hns1825/）
    # ============================================================
    if [ "$rdma_enabled" = "true" ]; then
        echo "===== [3/${step_total}] Building backend: 950 (HNS_1825) ====="
        [ -d build ] && rm -rf build

        local compile_opts_950_hns=${COMPILE_OPTIONS}
        compile_opts_950_hns=$(echo "$compile_opts_950_hns" | sed 's/-DACLSHMEM_RDMA_BACKEND=\S*//g')
        compile_opts_950_hns="${compile_opts_950_hns} -DACLSHMEM_RDMA_BACKEND=HNS_1825"
        mkdir -p build && cd build
        cmake ${compile_opts_950_hns} \
            -DCMAKE_INSTALL_PREFIX=../install \
            -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
            -DUSE_CXX11_ABI=${USE_CXX11_ABI} \
            -DUSE_MSSANITIZER=${USE_MSSANITIZER} \
            -DSOC_TYPE="Ascend950" \
            -DPYEXPAND_EXAMPLE=${PYEXPAND_EXAMPLE} \
            -DBUILD_PYTHON=OFF \
            .. || { echo "[ERROR] cmake failed for backend 950 (HNS_1825)"; exit 1; }
        make install -j$(nproc) || { echo "[ERROR] make install failed for backend 950 (HNS_1825)"; exit 1; }
        mkdir -p ../install/shmem/backends/950_hns1825
        cp ../install/shmem/lib/*.so ../install/shmem/backends/950_hns1825/ || { echo "[ERROR] Failed to copy 950 (HNS_1825) backend libraries"; exit 1; }
        cd -
        echo "===== Backend 950 (HNS_1825) built ====="
    fi

    # Write unified version info after all backends are built
    _write_version_info "${PROJECT_ROOT}/src/python/shmem/version.info"

    # ============================================================
    # Package wheel (skip cmake in setup.py via _SHMEM_PREBUILT)
    # ============================================================
    local pkg_step_label="[${step_total}/${step_total}]"
    echo "===== ${pkg_step_label} Packaging wheel ====="

    # Assert backends exist before packaging
    if [ ! -d "${PROJECT_ROOT}/install/shmem/backends/910" ]; then
        echo "[ERROR] Backend 910 directory not found: ${PROJECT_ROOT}/install/shmem/backends/910"
        exit 1
    fi
    if [ ! -d "${PROJECT_ROOT}/install/shmem/backends/950" ]; then
        echo "[ERROR] Backend 950 directory not found: ${PROJECT_ROOT}/install/shmem/backends/950"
        exit 1
    fi
    if [ "$rdma_enabled" = "true" ]; then
        if [ ! -d "${PROJECT_ROOT}/install/shmem/backends/950_xscale" ]; then
            echo "[ERROR] Backend 950_xscale directory not found: ${PROJECT_ROOT}/install/shmem/backends/950_xscale"
            exit 1
        fi
        if [ ! -d "${PROJECT_ROOT}/install/shmem/backends/950_hns1825" ]; then
            echo "[ERROR] Backend 950_hns1825 directory not found: ${PROJECT_ROOT}/install/shmem/backends/950_hns1825"
            exit 1
        fi
    fi

    export _SHMEM_PREBUILT=1
    trap 'unset _SHMEM_PREBUILT' EXIT
    python3 setup.py bdist_wheel --plat-name "$plat" --python-tag "$py_ver"
    trap - EXIT
    unset _SHMEM_PREBUILT
    echo "===== Wheel built ====="
}

function make_package()
{
    rm -rf "${PROJECT_ROOT}/package"
    if [ $( uname -a | grep -c -i "x86_64" ) -ne 0 ]; then
        ARCH="x86_64"
    elif [ $( uname -a | grep -c -i "aarch64" ) -ne 0 ]; then
        ARCH="aarch64"
    else
        exit 1
    fi

    mkdir -p "${PROJECT_ROOT}"/package/$ARCH/
    if [ "$PYEXPAND_TYPE" = "ON" ]; then
         cp "${PROJECT_ROOT}"/dist/*.whl "${PROJECT_ROOT}"/package/$ARCH/
         whl_name=`basename ${PROJECT_ROOT}/src/python/dist/*.whl`
         echo "${whl_name} is copy to ${PROJECT_ROOT}/package"
    fi
    cp -r "${PROJECT_ROOT}"/install/$ARCH "${PROJECT_ROOT}"/package
    echo "SHMEM_${VERSION}_linux-${ARCH}.run is copy to ${PROJECT_ROOT}/package"
}

function fn_make_run_package()
{
    if [ $( uname -a | grep -c -i "x86_64" ) -ne 0 ]; then
        echo "it is system of x86_64"
        ARCH="x86_64"
    elif [ $( uname -a | grep -c -i "aarch64" ) -ne 0 ]; then
        echo "it is system of aarch64"
        ARCH="aarch64"
    else
        echo "it is not system of x86_64 or aarch64"
        exit 1
    fi

    mkdir -p $OUTPUT_DIR
    _write_version_info "$OUTPUT_DIR/version.info"

    mkdir -p $OUTPUT_DIR/scripts
    mkdir -p $RELEASE_DIR/$ARCH
    cp $PROJECT_ROOT/scripts/install.sh $OUTPUT_DIR
    cp $PROJECT_ROOT/scripts/set_env.sh $OUTPUT_DIR
    cp $PROJECT_ROOT/scripts/uninstall.sh $OUTPUT_DIR/scripts
    cp $PROJECT_ROOT/scripts/preinstall_check.sh $OUTPUT_DIR/scripts

    sed -i "s/SHMEMPKGARCH/${ARCH}/" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/scripts/uninstall.sh

    chmod +x $OUTPUT_DIR/*.sh
    chmod +x $OUTPUT_DIR/scripts/*.sh

    makeself_dir=${ASCEND_HOME_PATH}/toolkit/tools/op_project_templates/ascendc/customize/cmake/util/makeself/
    ${makeself_dir}/makeself.sh --header ${makeself_dir}/makeself-header.sh \
        --help-header $PROJECT_ROOT/scripts/help.info --gzip --complevel 4 --nomd5 --sha256 --chown \
        ${OUTPUT_DIR} $RELEASE_DIR/$ARCH/SHMEM_${VERSION}_linux-${ARCH}.run "SHMEM-api" ./install.sh
    [ -d "$OUTPUT_DIR/$ARCH" ] && rm -rf "$OUTPUT_DIR/$ARCH"
    cp -r $RELEASE_DIR/$ARCH $OUTPUT_DIR
    echo "SHMEM_${VERSION}_linux-${ARCH}.run is successfully generated in $OUTPUT_DIR/$ARCH"
}

function fn_build_googletest()
{
    if [ -d "$THIRD_PARTY_DIR/googletest/lib" ]; then
        return 0
    fi
    cd $THIRD_PARTY_DIR
    [[ ! -d "googletest" ]] && git clone --branch v1.14.x --depth 1 https://gitcode.com/GitHub_Trending/go/googletest.git
    cd googletest

    mkdir -p build && cd build
    if [ "$USE_CXX11_ABI" == "ON" ]
    then
        sed -i '21 a add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)' ../CMakeLists.txt
    else
        sed -i '21 a add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)' ../CMakeLists.txt
    fi

    cmake .. -DCMAKE_INSTALL_PREFIX=$THIRD_PARTY_DIR/googletest -DCMAKE_SKIP_RPATH=TRUE -DCMAKE_CXX_FLAGS="-fPIC"
    cmake --build . --parallel $(nproc)
    cmake --install . > /dev/null
    [[ -d "$THIRD_PARTY_DIR/googletest/lib64" ]] && cp -rf $THIRD_PARTY_DIR/googletest/lib64 $THIRD_PARTY_DIR/googletest/lib
    echo "Googletest is successfully installed to $THIRD_PARTY_DIR/googletest"
    cd ${PROJECT_ROOT}
}

function fn_build_nlohmann_json()
{
    if [ -f "$THIRD_PARTY_DIR/json/single_include/nlohmann/json.hpp" ]; then
        return 0
    fi

    cd $THIRD_PARTY_DIR
    rm -rf json
    git clone --branch v3.11.3 --depth 1 https://gitcode.com/GitHub_Trending/js/json.git json
    cd ${PROJECT_ROOT}
}

function fn_build_doxygen()
{
    if [ -d "$THIRD_PARTY_DIR/doxygen" ]; then
        return 0
    fi
    cd $THIRD_PARTY_DIR
    wget --no-check-certificate https://gitcode.com/gh_mirrors/do/doxygen/releases/download/Release_1_9_6/doxygen-1.9.6.src.tar.gz
    tar -xzvf doxygen-1.9.6.src.tar.gz
    cd doxygen-1.9.6
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$THIRD_PARTY_DIR/doxygen
    cmake --build . --parallel $(nproc)
    cmake --install . > /dev/null
    rm -rf $THIRD_PARTY_DIR/doxygen-1.9.6
    cd ${PROJECT_ROOT}
}

function fn_build_sphinx()
{
    [[ "$COVERAGE_TYPE" != "" ]] && return 0
    pip install sphinx
    pip install sphinx_rtd_theme
    pip install myst_parser
    pip install breathe
    pip install linkify-it-py
}

function fn_gen_doc()
{
    cd $PROJECT_ROOT
    branch=$(git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || echo $branch)
    local doxyfile=$PROJECT_ROOT/docs/Doxyfile
    local doxygen_output_dir=$PROJECT_ROOT/docs/$branch
    [[ -f "$doxyfile" ]] && rm -rf $doxyfile
    [[ -d "$doxygen_output_dir" ]] && rm -rf $doxygen_output_dir
    mkdir -p $doxygen_output_dir
    $THIRD_PARTY_DIR/doxygen/bin/doxygen -g $doxyfile
    sed -i "s#PROJECT_NAME           =.*#PROJECT_NAME           = \"Shmem\"#g" $doxyfile
    sed -i "s#PROJECT_NUMBER         =.*#PROJECT_NUMBER         = $branch#g" $doxyfile
    sed -i "s#OUTPUT_DIRECTORY       =.*#OUTPUT_DIRECTORY       = $doxygen_output_dir#g" $doxyfile
    sed -i "s#OUTPUT_LANGUAGE        =.*#OUTPUT_LANGUAGE        = English#g" $doxyfile
    sed -i "s#INPUT                  =.*#INPUT                  = $PROJECT_ROOT/include/host $PROJECT_ROOT/include/device $PROJECT_ROOT/include/host_device#g" $doxyfile
    sed -i "s#RECURSIVE              =.*#RECURSIVE              = YES#g" $doxyfile
    sed -i "s#USE_MDFILE_AS_MAINPAGE =.*#USE_MDFILE_AS_MAINPAGE = $PROJECT_ROOT/README.md#g" $doxyfile
    sed -i "s#HTML_EXTRA_STYLESHEET  =.*#HTML_EXTRA_STYLESHEET  = $PROJECT_ROOT/docs/doxygen/custom.css#g" $doxyfile
    sed -i "s#GENERATE_LATEX         =.*#GENERATE_LATEX         = NO#g" $doxyfile
    sed -i "s#HAVE_DOT               =.*#HAVE_DOT               = NO#g" $doxyfile
    sed -i "s#WARNINGS_AS_ERROR      =.*#WARNINGS_AS_ERROR      = NO#g" $doxyfile
    sed -i "s#EXTRACT_ALL            =.*#EXTRACT_ALL            = YES#g" $doxyfile
    sed -i "s#USE_MATHJAX            =.*#USE_MATHJAX            = YES#g" $doxyfile
    sed -i "s#WARN_NO_PARAMDOC       =.*#WARN_NO_PARAMDOC       = YES#g" $doxyfile
    sed -i "s#GENERATE_TREEVIEW      =.*#GENERATE_TREEVIEW      = YES#g" $doxyfile
    sed -i "s#WARN_AS_ERROR          =.*#WARN_AS_ERROR          = YES#g" $doxyfile
    sed -i "s#GENERATE_XML           =.*#GENERATE_XML           = YES#g" $doxyfile
    sed -i "s#EXPAND_ONLY_PREDEF     =.*#EXPAND_ONLY_PREDEF     = NO#g" $doxyfile
    sed -i "s#SKIP_FUNCTION_MACROS   =.*#SKIP_FUNCTION_MACROS   = NO#g" $doxyfile
    sed -i "s#ALLOW_DUPLICATE_MEMBERS =.*#ALLOW_DUPLICATE_MEMBERS = YES#g" $doxyfile
    sed -i "s#EXCLUDE_SYMBOLS        =.*#EXCLUDE_SYMBOLS        = shmem* addrGm#g" "$doxyfile"
    $THIRD_PARTY_DIR/doxygen/bin/doxygen $doxyfile
    [[ "$COVERAGE_TYPE" != "" ]] && return 0
    local sphinx_out_dir=$PROJECT_ROOT/docs/$branch/guide
    [[ -d "$sphinx_out_dir" ]] && rm -rf $sphinx_out_dir
    mkdir -p $sphinx_out_dir
    sphinx-build -M html $PROJECT_ROOT/docs $sphinx_out_dir
}

set -euo pipefail
while [[ $# -gt 0 ]]; do
    case "$1" in
        -uttests)
            fn_build_googletest
            BUILD_TYPE=Debug
            cd $THIRD_PARTY_DIR; [[ ! -d "catlass" ]] && git clone https://gitcode.com/cann/catlass.git; cd $PROJECT_ROOT
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_UNIT_TEST=ON"
            shift
            ;;
        -cann)
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DENABLE_CANN_BUILD=ON"
            shift
            ;;
        -debug)
            BUILD_TYPE=Debug
            COMPILE_OPTIONS="${COMPILE_OPTIONS}"
            shift
            ;;
        -examples)
            cd $THIRD_PARTY_DIR; [[ ! -d "catlass" ]] && git clone https://gitcode.com/cann/catlass.git; cd $PROJECT_ROOT
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_EXAMPLES=ON -DPython3_EXECUTABLE=$(which python3)"
            ENABLE_EXAMPLES=ON
            shift
            ;;
        -enable_rdma)
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DACLSHMEM_RDMA_SUPPORT=ON"
            shift
            ;;
        -enable_simt)
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DACLSHMEM_SIMT_SUPPORT=ON"
            shift
            ;;
        -enable_relay)
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DACLSHMEM_RELAY_SUPPORT=ON"
            shift
            ;;
        -python_extension)
            PYEXPAND_TYPE=ON
            shift
            ;;
        -python_example)
            cd $THIRD_PARTY_DIR; [[ ! -d "catlass" ]] && git clone https://gitcode.com/cann/catlass.git; cd $PROJECT_ROOT
            PYEXPAND_EXAMPLE=ON
            shift
            ;;
        -gendoc)
            fn_build_doxygen
            fn_build_sphinx
            GEN_DOC=ON
            shift
            ;;
        -onlygendoc)
            fn_build_doxygen
            fn_build_sphinx
            fn_gen_doc
            exit 0
            shift
            ;;
        -enable_ascendc_dump)
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DENABLE_ASCENDC_DUMP=ON"
            shift
            ;;
        -package)
            PACKAGE=ON
            PYEXPAND_TYPE=ON
            shift
            ;;
        -full)
            BUILD_ALL=ON
            fn_build_googletest
            cd $THIRD_PARTY_DIR; [[ ! -d "catlass" ]] && git clone https://gitcode.com/cann/catlass.git; cd $PROJECT_ROOT
            shift
            ;;
        -use_cxx11_abi1)
            USE_CXX11_ABI=ON
            shift
            ;;
        -use_cxx11_abi0)
            USE_CXX11_ABI=OFF
            shift
            ;;
        -mssanitizer)
            USE_MSSANITIZER=ON
            shift
            ;;
        -soc_type)
            SOC_TYPE="$2"
            shift 2
            ;;
        -rdma_backend)
            if [ "$RDMA_BACKEND" != "" ]; then
                echo "Error: -rdma_backend can only be specified once."
                print_usage
                exit 1
            fi
            shift
            RDMA_BACKEND="$1"
            if [ "$RDMA_BACKEND" != "XSCALE" ] && [ "$RDMA_BACKEND" != "HNS_1825" ]; then
                echo "Error: Invalid RDMA_BACKEND value '$RDMA_BACKEND'. Must be 'XSCALE' or 'HNS_1825'."
                print_usage
                exit 1
            fi
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DACLSHMEM_RDMA_BACKEND=${RDMA_BACKEND}"
            shift
            ;;
        *)
            echo "Error: Unknown option $1."
            print_usage
            exit 1
            ;;
    esac
done

# Validate relay parameter dependencies: relay is built on top of UDMA, which is only enabled for
# Ascend950. Fail early with a clear message instead of
# relying solely on the CMake FATAL_ERROR.
# Exception: the wheel/package build (fn_whl_build) always builds both the 910B and 950 backends
# internally and only enables relay on the 950 (UDMA) backend, so -enable_relay is allowed there
# regardless of SOC_TYPE.
if [ -n "$(echo "$COMPILE_OPTIONS" | grep -o '\-DACLSHMEM_RELAY_SUPPORT=ON')" ]; then
    if [ "$PYEXPAND_TYPE" != "ON" ] && [ "$BUILD_ALL" != "ON" ] && [ "$SOC_TYPE" != "Ascend950" ]; then
        echo "Error: -enable_relay requires UDMA support, which is only enabled for Ascend950."
        echo "       Please add -soc_type Ascend950 when using -enable_relay,"
        echo "       or use -python_extension / -full to build the multi-SOC wheel (relay is"
        echo "       applied only to the 950 backend)."
        print_usage
        exit 1
    fi
fi

# Validate RDMA parameter dependencies
if [ -n "$RDMA_BACKEND" ]; then
    if [ "$SOC_TYPE" != "Ascend950" ] && [ "$PACKAGE" != "ON" ]; then
        echo "Error: -rdma_backend can only be specified when SOC_TYPE is Ascend950 (or with -package for multi-SOC)."
        echo "       Please use -soc_type Ascend950 or -package when specifying -rdma_backend."
        print_usage
        exit 1
    fi
    if [ -z "$(echo "$COMPILE_OPTIONS" | grep -o '\-DACLSHMEM_RDMA_SUPPORT=ON')" ]; then
        echo "Error: -rdma_backend requires -enable_rdma to be specified."
        echo "       Please add -enable_rdma flag when using -rdma_backend."
        print_usage
        exit 1
    fi
fi

if [ "$SOC_TYPE" = "Ascend950" ] || [ "$PACKAGE" = "ON" ]; then
    if [ -n "$(echo "$COMPILE_OPTIONS" | grep -o '\-DACLSHMEM_RDMA_SUPPORT=ON')" ]; then
        if [ -z "$RDMA_BACKEND" ]; then
            # wheel 构建（-python_extension / -package / -full）会自动构建 XSCALE + HNS_1825 双后端
            if [ "$PYEXPAND_TYPE" = "ON" ] || [ "$BUILD_ALL" = "ON" ]; then
                echo "[INFO] -enable_rdma specified without -rdma_backend. Wheel will build both XSCALE and HNS_1825 backends."
            elif [ "$SOC_TYPE" = "Ascend950" ]; then
                echo "Error: -rdma_backend must be specified when SOC_TYPE is Ascend950 and RDMA is enabled."
                print_usage
                exit 1
            else
                echo "WARNING: -enable_rdma specified without -rdma_backend. RDMA for 950 may not link correctly."
                echo "         Consider adding -rdma_backend XSCALE or -rdma_backend HNS_1825."
            fi
        fi
    fi
    fn_build_nlohmann_json
fi

# 清空 build
[ -d build ] && rm -rf build

if [ "$BUILD_ALL" = "ON" ]; then
    OLD_COMPILE_OPTIONS=${COMPILE_OPTIONS}
    # build whl
    fn_whl_build

    # build examples
    COMPILE_OPTIONS="${OLD_COMPILE_OPTIONS} -DUSE_EXAMPLES=ON"
    fn_build

    # build uttests
    BUILD_TYPE=Debug
    COMPILE_OPTIONS="${OLD_COMPILE_OPTIONS} -DUSE_UNIT_TEST=ON"
    fn_build

    # build package
    PYEXPAND_TYPE=ON
    fn_make_run_package
    make_package
else
    if [ "$PYEXPAND_TYPE" = "ON" ]; then
        fn_whl_build
    fi

    # fn_build 使用全局 SOC_TYPE（默认 910），剔除 950 专用的 RDMA_BACKEND
    # 除非用户显式指定了 SOC_TYPE=Ascend950
    _compile_opts_default=$COMPILE_OPTIONS
    if [ "$SOC_TYPE" != "Ascend950" ]; then
        _compile_opts_default=$(echo "$_compile_opts_default" | sed 's/-DACLSHMEM_RDMA_BACKEND=[^ ]*//g')
    fi
    COMPILE_OPTIONS="$_compile_opts_default"
    rm -rf build   # 确保干净构建，清除可能残留的 CMake 缓存
    fn_build
    fn_make_run_package
    if [ "$PACKAGE" == "ON" ]; then
        make_package
    fi

    if [ ${GEN_DOC} == "ON" ]; then
        fn_gen_doc
    fi
fi

cd ${CURRENT_DIR}
