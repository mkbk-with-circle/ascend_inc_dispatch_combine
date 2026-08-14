# Compilation and Build
## SHMEM Build
### Downloading the SHMEM Source Code

```shell
git clone https://gitcode.com/cann/shmem.git
```

You can choose your desired branch.

### Build

Enter the root directory of SHMEM and build the project.

```shell
cd shmem
bash scripts/build.sh
```

For more information about the commands, see see the *README* and the `scripts/build.sh` file in the SHMEM's `master` directory.

### SHMEM Build Description
The basic SHMEM build command is `bash build.sh`. The default build mode generates version information and creates an installation package. By default, the RDMA capability, test cases, tests, and Python APIs are not built. The following parameters can be added to implement different functions:
- `-use_cxx11_abi1`: enables `C++11 ABI`. This is the default setting.
- `-use_cxx11_abi0`: disables `C++11 ABI`.
- `-cann`: CANN open APIs can be used for build if the CANN version is 8.5 or later.
- `-uttests`: builds all UT cases in the `tests/unittest` directory.
- `-examples`: displays all cases in the `examples` directory.
- `-python_example`: provides the PyTorch access capability for some cases in the `examples` directory.
- `-enable_rdma`: builds and enables RDMA capabilities. By default, the RDMA backend type is configured for Ascend 910B/C. For Ascend 950, you need to use `-rdma_backend` to specify the backend type.
- `-rdma_backend`: specifies the RDMA backend type (not supported by Ascend 910B/C). The value can be `XSCALE` (Yunsilicon NICs are used). This option must be used together with `-enable_rdma`. Otherwise, an error is returned. The parameter sequence is not limited.
- `-enable_ascendc_dump`: enables the `AscendC_Dump` mode for debugging the operator kernel code.
- `-package`:
    * builds a .whl package extended by Python.
    * generates `SHMEM\_{version}\_linux-{arch}.run` in the `{project_root}/package/{arch}/` directory.
    * generates the Python .whl package `shmem-xxx.whl` in the `{project_root}/package/{arch}/` directory.
- `-python_extension`: generates the Python .whl package `shmem-xxx.whl` in the `{project_root}/dist/` directory.
- `-gendoc`: generates documents.
- `-onlygendoc`: generates documents without building the source code.
- `-debug`: sets the build type to `Debug`.
- `-mssanitizer`: enables the msSanitizer memory detection tool for samples. The script can be executed only after msSanitizer is used to start the task. The AllGather sample running script provides the `tool` option to use the tool. For details about how to use the tool, see [Exception Detection Tool (msSanitizer, MindStudio Sanitizer)](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0039.html). Note: If this option is enabled, ensure that the msSanitizer tool is used to start the operator. If other methods are used to start the operator, there may be unknown errors.
- `-soc_type`: If the SoC is Ascend 950, add the `-soc_type Ascend950` parameter. You can run the `npu-smi info` command to view the parameter. For other SoCs, you do not need to add this parameter.
- `-enable_simt`: enables the SIMT programming mode.
- `-full`: compiles the package, Python extension, UT tests, and examples. The SHMEM automatic compilation script automatically downloads the dependency libraries, compiles the project and UT test cases, and packages the libraries.

### RDMA Parameters

To use the RDMA function, enable the `-enable_rdma` parameter during build. Then, you need to configure the backend based on the server type and NIC type.

- Ascend 910B/C platform
  - Only the `-enable_rdma` parameter is required. No additional parameter is required. The backend uses the default configuration.
- Ascend 950 platform
  - Use the `-enable_rdma` parameter.
  - Explicitly specify `-soc_type Ascend950`.
  - Specify the `-rdma_backend` parameter. Otherwise, the build script will report an error.

**Parameter dependency rules**:
- The `-enable_rdma` parameter must be specified when the `-rdma_backend` parameter is used.
- The `-rdma_backend` parameter is valid only when `-soc_type Ascend950` is used. Otherwise, the build script will report an error.
- If `-rdma_backend` is specified but `-enable_rdma` is not specified, the build will fail, and an error message will be displayed.
- The sequence of the three parameters (`-rdma_backend`, `-enable_rdma`, `-soc_type xxx`) is not limited, but all dependent parameters must be specified.

**Valid command examples**:

```shell
# Enabling RDMA on the Ascend 950 platform and specifying XSCALE as the backend (parameter sequence example 1)
bash scripts/build.sh -soc_type Ascend950 -enable_rdma -rdma_backend XSCALE

# Enabling RDMA on the Ascend 950 platform and specifying XSCALE as the backend (parameter sequence example 2)
bash scripts/build.sh -enable_rdma -soc_type Ascend950 -rdma_backend XSCALE

# Enabling RDMA on the Ascend 950 platform and specifying XSCALE as the backend (parameter sequence example 3)
bash scripts/build.sh -rdma_backend XSCALE -enable_rdma -soc_type Ascend950

# Enabling RDMA on Ascend 910B/C (backend not specified)
bash scripts/build.sh -enable_rdma
```

**Invalid command examples**:

```shell
# Error: -rdma_backend is specified but -enable_rdma is not enabled.
bash scripts/build.sh -soc_type Ascend950 -rdma_backend XSCALE
# Error message: "Error: -rdma_backend requires -enable_rdma to be specified."

# Error: -rdma_backend is used on a non-Ascend 950 platform.
bash scripts/build.sh -enable_rdma -rdma_backend XSCALE
# Error message: "Error: -rdma_backend can only be specified when SOC_TYPE is Ascend950."
```

**RDMA backend types**:

The `-rdma_backend` parameter supports the following backend types:
- `XSCALE`: Yunsilicon NIC (supported only by Ascend 950)

**⚠️ Important**:

1. **Automatically-generated compilation definitions**:
   - CMake automatically generates the compilation definitions below based on the value of `-rdma_backend`. You do not need to manually set them.
     - When `-rdma_backend` is set to `XSCALE`, `-DACLSHMEMI_RDMA_K_BACKEND_XSCALE=1` is automatically added.
     - When no backend is specified, `-DACLSHMEMI_RDMA_K_BACKEND_IN_DIE=1` is automatically added.
   - Manually defining these macros may cause compilation conflicts or function exceptions.

2. **Internal macros that cannot be manually defined**:
   - Do not manually add the compilation definition named `ACLSHMEMI_K_RDMA_BACKEND`.
   - This is an internal macro used by the device code and is set by the system in `src/device/gm2gm/engine/shmem_device_rdma.hpp` based on the automatically-generated definition.
   - Manually defining this macro may cause compilation errors or runtime function exceptions.

## Key SHMEM Files
### `scripts` Directory
   - `install.sh`: installation script
   - `uninstall.sh`: uninstallation script
   - `build.sh`: build script
   - `release.sh`: automatic build and packaging script
   - `set_env.sh`: SHMEM environment variable settings
### run.sh Script Usage
UT case execution script
```sh
bash scripts/run.sh
```
Multiple parameters are provided to support custom case execution.
```sh
-ranks          # Total number of ranks
-frank          # First rank on the server
-ipport         # IP address and port
-fnpu           # First NPU started on each server
-gnpus          # The number of devices used by a single server
-test_filter    # gtest_filter

# Example
bash scripts/run.sh -ranks 4 -fnpu 2 -gnpus 4 -test_filter ScalarP # The test will be performed on devices 2 to 6.

```
### install.sh
Installation/Uninstallation script of the generated .run file
Installation directory
```
${INSTALL_PATH}
    |--shmem
        |--latest
        |--${version}
            |--shmem
                |--include (header file)
                |--lib      (SO library)
            |--scripts      (uninstallation script)
```
Note: `INSTALL_PATH` indicates the user-defined installation directory.
### uninstall.sh
Uninstallation script, which can be used to uninstall the SHMEM library installed in the corresponding path or uninstall the SHMEM library in the default path using the `--uninstall` option of the .run file

### release.sh
Script for generating a package, which is used after compilation. After the compilation products are packaged, other files in the `install` directory are deleted. You are advised to use the `build.sh` script to complete the packaging.


### build.sh
File name: `scripts/build.sh`
SHMEM compilation file, which does not need to be modified

### set_env.sh
​File name: `scripts/set_env.sh`
After SHMEM is installed, the process-level environment variable setting script `set_env.sh` is provided to automatically set environment variables. The environment variables automatically become invalid after the user process ends.
