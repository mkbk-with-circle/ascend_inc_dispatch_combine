# **1. .so File Deployment and Usage Guide**

## 1.1 Overview
This guide is intended for development and O&M personnel in the Linux environment. It describes .so file deployment, dependencies, and troubleshooting and verification methods.

### 1.1.1 Deployment Principles
- Copy the .so files to the user's project directory (for example, `${CUSTOM_PROJECT}/lib/`).
- The .so files are loaded from the user's own project directory at runtime and do not depend on the system's public directory.
- In the current implementation, the bootstrap plugins (`aclshmem_bootstrap_config_store.so` and `aclshmem_bootstrap_mpi.so`) are preferentially loaded from the absolute path in the same directory as `aclshmem.so`. During deployment, the plugins must be placed in the same directory as `libshmem.so`.
- Other external runtime .so files (such as those related to CANN/OpenSSL) must meet their own installation and visibility requirements.

## 1.2 List of .so Files That Must Be Copied
| File Name| Function| Core Dependency|
|---|---|---|
| `libshmem.so` | Main SHMEM function library, which provides external core APIs| Yes|
| `aclshmem_bootstrap_config_store.so` | Default/Unique ID boot initialization capability| Yes|
| `aclshmem_bootstrap_mpi.so` | MPI boot initialization capability (only in the MPI scenario)| No|
| `libshmem_utils.so` (if the build product exists)| Library of common tools and basic capability| Yes|

### 1.2.1 External .so Files That Are Dynamically Loaded at Runtime (Visible in the Target Environment)
| File Name| Source| Purpose|
|---|---|---|
| `libascend_hal.so` | CANN/Driver runtime| HAL capability loading|
| `libascendcl.so` | CANN Toolkit | ACL API loading|
| `libruntime.so` | CANN Toolkit | Runtime API loading|
| `libopapi.so` | CANN Toolkit | OP API loading|
| `libra.so` | CANN communication component| HCCP-related capabilities|
| `libtsdclient.so` | CANN communication component| TSD capability|
| `libssl.so` | OpenSSL | TLS capability|
| `libcrypto.so` | OpenSSL | Encryption capability|

Note: `libssl.so` and `libcrypto.so` are dynamically loaded from the absolute paths by the configuration storage module. The paths are obtained from `EP_OPENSSL_PATH`.

## 1.3 .so File Dependencies (Linux)
### 1.3.1 Dependency Level Description
- Strong dependency: If a strongly dependent file is missing, the program fails to be started or core capabilities are unavailable.
- Weak dependency: If a weakly dependent file is missing, only the corresponding extended capabilities are unavailable.

### 1.3.2 Mermaid Dependency Diagram (Example)
```mermaid
graph TD
    A[${CUSTOM_PROJECT}/lib/libshmem.so] --> B[${CUSTOM_PROJECT}/lib/libshmem_utils.so]
    A --> C[${CUSTOM_PROJECT}/lib/aclshmem_bootstrap_config_store.so]
    A --> D[${CUSTOM_PROJECT}/lib/aclshmem_bootstrap_mpi.so]
    A --> E[libascend_hal.so]
    A --> F[libascendcl.so]
    A --> G[libruntime.so]
    A --> H[libopapi.so]
    A --> I[libra.so]
    A --> J[libtsdclient.so]
    C --> K[libssl.so]
    C --> L[libcrypto.so]
    A --> M[/lib64/libc.so.6]
    A --> N[/lib64/libpthread.so.0]
    A --> O[/lib64/libdl.so.2]
    D --> P[/usr/lib64/libmpi.so]

    classDef strong fill:#FDE2E2,stroke:#C53030,stroke-width:1px;
    classDef weak fill:#E6FFFA,stroke:#2C7A7B,stroke-width:1px;

    class A,C,E,F,G,H,I,J,K,L,M,N,O strong;
    class B,D,P weak;
```

## 1.4 Deployment Procedure (Linux)
### 1.4.1 Confirming that the Build Output Is Generated in the `install/` Directory
The output directory of `build.sh` must be `install/`. Do not use the historical default output directory.

Linux commands (CentOS/Ubuntu):
```bash
# Run the command in the root directory.
bash build.sh install

# Product check
ls -l install/lib/*.so
```

### 1.4.2 Copying the .so File to the User's Project Directory
Linux commands (CentOS/Ubuntu):
```bash
# Replace the variables on demand.
CUSTOM_PROJECT=/path/to/user_project
mkdir -p ${CUSTOM_PROJECT}/lib
cp -f install/lib/*.so ${CUSTOM_PROJECT}/lib/
chmod 755 ${CUSTOM_PROJECT}/lib/*.so
```

Note: According to the current bootstrap loading logic, the preceding deployment mode can meet the requirements for loading the bootstrap plugin.

### 1.4.3 Recommended Fixed Loading Visibility Policies (User Side)
To avoid loading drift caused by the coexistence of multiple versions of libraries, you are advised to use the following fixed visibility policies:
- Deploy `libshmem.so`, `aclshmem_bootstrap_config_store.so`, `aclshmem_bootstrap_mpi.so` and `libshmem_utils.so` in `${CUSTOM_PROJECT}/lib/`.
- Service programs are preferentially started from `${CUSTOM_PROJECT}/bin/` to ensure that the mapping between the main program and the target library directory is fixed.
- Do not retain .so files that of old versions and have the same names in the system public directory to avoid hitting the historical residual library.
- During troubleshooting, use `LD_DEBUG=libs,files` to confirm the actual loading path and ensure that `aclshmem_bootstrap_config_store.so` is from the same directory as `libshmem.so`.

Supplementary information (two-phase loading logic):
- Phase A (dynamic linking phase): The system's dynamic linker parses the `NEEDED` dependency between the main program and `libshmem.so`. The actual loading sequence is determined by the linking relationship and is not used as an interface semantic constraint.
- Phase B (initialization phase): After `aclshmemx_init_attr` is called, `libshmem.so` selects and loads `aclshmem_bootstrap_config_store.so` or `aclshmem_bootstrap_mpi.so` in bootstrap mode.
- Python package scenario: `src/python/shmem/__init__.py` has a preloading behavior in the import phase. This behavior is only used for dependency fallback during Python encapsulation and cannot be equated with the plugin loading sequence semantics in the `libshmem.so` initialization phase.

## 1.5 Build Product Path Configuration (build.sh Path Adjustment)
### 1.5.1 Adjustment Objective
Change the .so file output directory in `build.sh` to `install/` to replace the old default path (for example, `output/`).

### 1.5.2 Modification Example (Directly Copyable)
Example for Linux:
```bash
# build.sh snippet (example)
# Old configuration: DESTDIR=./output
# New configuration: DESTDIR=./install

DESTDIR=./install
INSTALL_LIB_DIR=${DESTDIR}/lib
mkdir -p "${INSTALL_LIB_DIR}"

# Example: Installing the build product
cp -f ${BUILD_DIR}/lib/*.so "${INSTALL_LIB_DIR}/"
```

### 1.5.3 Command Execution After Adjustment
Linux commands (CentOS/Ubuntu):
```bash
bash build.sh install
ls -l install/lib/
```

## 1.6 Dependency Check Method (Linux)
### 1.6.1 Using `ldd` to Check for Missing Dependencies
Linux commands:
```bash
ldd ${CUSTOM_PROJECT}/lib/libshmem.so
ldd ${CUSTOM_PROJECT}/lib/aclshmem_bootstrap_config_store.so
```

Result interpretation:
- If `not found` is displayed, the dependency is missing or the path is incorrect.
- If the dependencies are parsed to a specific absolute path, the dependency relationship is normal.

### 1.6.2 Using `objdump` to Check NEEDED Entries
Linux commands:
```bash
objdump -p ${CUSTOM_PROJECT}/lib/libshmem.so | grep NEEDED
objdump -p ${CUSTOM_PROJECT}/lib/aclshmem_bootstrap_config_store.so | grep NEEDED
```

Result interpretation:
- `NEEDED` displays the dynamic link library that .so files actually depend on.
- If the key dependencies are inconsistent with those in the deployment list, backtrack the build configuration or repackage the dependencies.

### 1.6.3 Common Dependency Issues and Solutions
- Missing dependencies: Add the missing .so files to `${CUSTOM_PROJECT}/lib/` and verify `ldd` again.
- Version mismatch: Replace the .so files with those of the same version as the build environment and perform the test again.
- 3Incomplete copy: Run `cp -f install/lib/*.so ${CUSTOM_PROJECT}/lib/` again.

## 1.7 Verification Method (Linux)
### 1.7.1 File and Permission Verification
```bash
ls -l ${CUSTOM_PROJECT}/lib/*.so
```
Expected result: The target .so files are complete, and the permission is `-rwxr-xr-x` (755) or higher, allowing for read and execute operations.

### 1.7.2 Dependency Integrity Verification
```bash
ldd ${CUSTOM_PROJECT}/lib/libshmem.so | grep -i "not found" && echo " Dependency is abnormal." || echo "Dependency is normal"
```
Expected result: "Dependency is normal" is displayed.

### 1.7.3 Loading Verification on the User Side (Example)
```bash
# The service program is used as an example. Replace it with the actual executable file.
${CUSTOM_PROJECT}/bin/app --version
```
Expected result: The program is started properly, and no .so file loading failure log is generated.

### 1.7.4 Loading Path Consistency Verification (Recommended)
```bash
LD_DEBUG=libs,files ${CUSTOM_PROJECT}/bin/app --version 2>&1 | grep -E "libshmem.so|aclshmem_bootstrap_config_store.so"
```
Expected result: `aclshmem_bootstrap_config_store.so` and `libshmem.so` are from the same deployment directory. The plugins are correctly loaded in bootstrap mode during initialization.

## 1.8 FAQs
### 1.8.1 Failed to Adjust the build.sh Path
Symptom: The product is still output to the old directory.

Solution:
- Check whether there are old variables (such as `DESTDIR=./output`) in `build.sh`.
- Confirm the path overwriting sequence to prevent `DESTDIR` from being rewritten again in subsequent logic.
- Delete the old directory and perform the build again.
```bash
rm -rf build output install
bash build.sh install
```

### 1.8.2 Insufficient Permission on the Project Path
Symptom: The error message "Permission denied" is displayed during copy or running.

Solution:
```bash
chmod -R u+rwX ${CUSTOM_PROJECT}/lib
chmod 755 ${CUSTOM_PROJECT}/lib/*.so
```
If necessary, the O&M personnel should adjust the directory owner and owner group in a unified manner.

### 1.8.3 .so File Compatibility Between Different Linux Distributions
Symptom: After the .so files are directly reused between CentOS and Ubuntu, the loading fails.

Solution:
- You are advised to rebuild and redeploy `install/lib/*.so` on the target distribution.
- Ensure that the glibc version is the same as that in the build baseline.
- Use `ldd` and `objdump -p` to check dependencies one by one.
