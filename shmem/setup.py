#!/usr/bin/env python
# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""python api for shmem."""

import os
import shutil
import subprocess
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.dist import Distribution
from setuptools.command.build_py import build_py

# 消除whl压缩包的时间戳差异
os.environ['SOURCE_DATE_EPOCH'] = '315532800'  # 1980-01-01, min valid ZIP timestamp


def _resolve_package_version() -> str:
    """Prefer env VERSION; otherwise read repo-root VERSION file (single source of truth)."""
    env_ver = os.getenv("VERSION", "").strip()
    if env_ver:
        return env_ver
    version_path = Path(__file__).resolve().parent / "VERSION"
    if version_path.is_file():
        for line in version_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    raise RuntimeError(
        "SHMEM package version is unset: export VERSION=x.y.z or add a root VERSION file"
    )


current_version = _resolve_package_version()


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    @property
    def is_pure(self):
        return False

# ========================
# Custom build_py: Integrate the C++ build process
# ========================
class BuildCppLibs(build_py):
    def run(self):
        self._build_cpp()

        self._copy_libraries_to_package()

        super().run()

    def _build_cpp(self):
        # Skip cmake build when backends are pre-built (multi-SOC mode)
        if os.getenv("_SHMEM_PREBUILT", "") == "1":
            print("_SHMEM_PREBUILT=1, skipping cmake build (using pre-built backends)")
            return

        build_dir = Path("build")
        install_dir = Path("install")

        build_dir.mkdir(exist_ok=True)

        pyexpand_type = os.getenv("PYEXPAND_TYPE", "ON")
        build_type = os.getenv("BUILD_TYPE", "Release")
        use_cxx11_abi = os.getenv("USE_CXX11_ABI", "ON")
        compile_options_str = os.getenv("COMPILE_OPTIONS", "")
        use_mssanitizer = os.getenv("USE_MSSANITIZER", "OFF")
        soc_type = os.getenv("SOC_TYPE", "")
        enable_udma_support = os.getenv("ACLSHMEM_UDMA_SUPPORT", "OFF")

        cmake_cmd = [
            "cmake",
            f"-DBUILD_PYTHON={pyexpand_type}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DUSE_CXX11_ABI={use_cxx11_abi}",
            "-DCMAKE_INSTALL_PREFIX=../install",
            f"-DUSE_MSSANITIZER={use_mssanitizer}",
            f"-DSOC_TYPE={soc_type}",
            f"-DACLSHMEM_UDMA_SUPPORT={enable_udma_support}",
            "-DCMAKE_SKIP_RPATH=TRUE",
        ]

        if compile_options_str.strip():
            cmake_cmd.extend(compile_options_str.split())

        cmake_cmd.append("..")

        try:
            subprocess.check_call(cmake_cmd, cwd=build_dir)
            subprocess.check_call(["make", "install", "-j17"], cwd=build_dir)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"C++ build failed: {e}")

        # CMake installs shared libraries to install/shmem/lib/.
        # Copy them into the backend-specific directory expected by the
        # package layout (install/shmem/backends/<name>/), matching the
        # backend name convention from CMake.
        if soc_type in ("", "Ascend910B"):
            backend_name = "910"
        elif soc_type == "Ascend950":
            backend_name = "950"
        else:
            backend_name = "910"

        backend_dir = Path("install/shmem/backends") / backend_name
        backend_dir.mkdir(parents=True, exist_ok=True)

        lib_dir = Path("install/shmem/lib")
        if not lib_dir.exists():
            raise RuntimeError("C++ build succeeded but 'install/shmem/lib' directory not found!")
        for so_file in lib_dir.glob("*.so"):
            shutil.copy(so_file, backend_dir)
        print(f"Copied {backend_name} backend libraries from install/shmem/lib/ to {backend_dir}")

        if not backend_dir.exists():
            raise RuntimeError(
                f"C++ build succeeded but backend directory not found: {backend_dir}"
            )

    def _copy_libraries_to_package(self):
        install_backends = Path("install/shmem") / "backends"
        package_src_dir = Path("src/python") / "shmem"

        scripts_dir = package_src_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        shell_script_src = Path("scripts") / "preinstall_check.sh"
        if shell_script_src.exists():
            shutil.copy(shell_script_src, scripts_dir / "preinstall_check.sh")
            print(f"Copied {shell_script_src} -> {scripts_dir / 'preinstall_check.sh'}")

        if not install_backends.exists():
            print("Warning: install/shmem/backends not found, skipping so copy")
            return

        package_src_dir.mkdir(parents=True, exist_ok=True)

        dst_backends = package_src_dir / "backends"
        if dst_backends.exists():
            shutil.rmtree(dst_backends)
        shutil.copytree(install_backends, dst_backends)
        print(f"Copied backends from {install_backends} -> {dst_backends}")

        for backend_dir in sorted(install_backends.iterdir()):
            if backend_dir.is_dir():
                so_list = list(backend_dir.glob("*.so"))
                print(f"  Backend {backend_dir.name}: {len(so_list)} .so files")

        # Copy root_info_generate binary (Ascend950 UDMA tool)
        install_bin = Path("install/shmem") / "bin"
        if install_bin.exists() and install_bin.is_dir():
            dst_bin = package_src_dir / "bin"
            if dst_bin.exists():
                shutil.rmtree(dst_bin)
            shutil.copytree(install_bin, dst_bin)
            print(f"Copied {install_bin} -> {dst_bin}")

        src_include = Path("include")
        dst_include = package_src_dir / "include"
        if dst_include.exists():
            shutil.rmtree(dst_include)
        shutil.copytree(src_include, dst_include)
        print(f"Copied {src_include} -> {dst_include}")

        dst_src = package_src_dir / "src"
        if dst_src.exists():
            shutil.rmtree(dst_src)
        dst_src.mkdir(parents=True, exist_ok=True)
        # Only src/device is required (public headers cross-reference shmemi_device_cc.h
        # and its transitive closure). src/device_simt and src/host_device are not
        # referenced by any public header and are excluded from the wheel.
        src_device = Path("src") / "device"
        if src_device.exists() and src_device.is_dir():
            shutil.copytree(src_device, dst_src / "device")
        print(f"Copied src/device -> {dst_src}")

        # version.info 由 build.sh 写入 install/ 目录
        version_file = Path("install") / "version.info"
        if version_file.exists():
            shutil.copy(version_file, package_src_dir / "version.info")


setup(
    name="cann-shmem",
    version=current_version,
    author="",
    author_email="",
    description="CANN SHMEM - shared memory communication library for Ascend NPU",
    packages=find_packages(where="src/python", exclude=("tests*",)),
    package_dir={"": "src/python"},
    license="Apache License Version 2.0",
    install_requires=["torch-npu"],
    python_requires=">=3.7",
    package_data={

        "shmem": [
            "*.so",
            "version.info",
            "include/**/*.h",
            "src/**/*.h",
            "src/**/*.hpp",
            "backends/**/*.so",
            "bin/root_info_generate",
            "scripts/*.sh",
        ]
    },
    entry_points={
        "console_scripts": [
            "shmem-config=shmem.shmem_config:main",
        ],
    },
    distclass=BinaryDistribution,
    cmdclass={"build_py": BuildCppLibs},
)
