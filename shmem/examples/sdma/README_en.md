# SDMA Usage Description
## Environment Requirements and Preparations
The SDMA feature is newly supported in CANN 9.0.0 or later (trial version). You need to download and install the following CANN and OPS software packages first:
- Toolkit package ([CANN master OBP image website](https://mirror-centralrepo.devcloud.cn-north-4.huaweicloud.com/artifactory/cann-run-mirror/software/master/))
- ops-legacy package (Download a required version based on the hardware platform: [A2 x86_64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260520_newest/cann-910b-ops-legacy_9.1.0_linux-x86_64.run)/[A2 aarch64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260520_newest/cann-910b-ops-legacy_9.1.0_linux-aarch64.run)/[A3 x86_64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260520_newest/cann-A3-ops-legacy_9.1.0_linux-x86_64.run)/[A3 aarch64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260520_newest/cann-A3-ops-legacy_9.1.0_linux-aarch64.run))

## Instructions for Using an Example
1. Build and install the software package in the `shmem/` directory:
```bash
bash scripts/build.sh -package
./install/*/SHMEM_1.0.0_linux-*.run --install
```

2. Build the examples in the `shmem/` directory:
```bash
bash scripts/build.sh -examples
```

3. Run the demo in the `shmem/examples/sdma` directory:
```bash
bash run.sh -pes ${PES} -type ${TYPES}
````
  - **Parameter description**:
      - PES: the number of devices (NPUs) used for running the demo, limited to a single server
      - TYPES: type of the data to be transferred. Currently, the following data types are supported: int, uint8, int64, and fp32.

## SDMA API Usage Description

### aclshmemx_sdma_put_nbi
For example, consider an API with pointer-type parameters:
```c++
ACLSHMEM_DEVICE void aclshmemx_sdma_put_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                            uint32_t elem_size, int pe, uint32_t sync_id)
```
API function: This API transfers elements (with the number specified by `elem_size`) from the address specified by `src` on the PE specified by `pe` to the address specified by `dst`.
| Parameter      | Description                                                                |
|--------------|----------------------------------------------------------------------|
| dst          | Symmetric address of the target device's destination address on the local device                                  |
| src          | Source address on the local device                                                      |
| buf          | Buffer address                                                          |
| ub_size      | Buffer size                                                          |
| elem_size    | Number of elements                                                            |
| pe           | Destination PE                                                              |
| sync_id      | Synchronization ID                                                              |

### aclshmemx_sdma_get_nbi
For example, consider an API with pointer-type parameters:
```c++
ACLSHMEM_DEVICE void aclshmemx_sdma_get_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                            uint32_t elem_size, int pe, uint32_t sync_id)
```
API function: This API transfers elements (with the number specified by `elem_size`) from the address specified by `dst` on the PE specified by `pe` to the address specified by `src`.
| Parameter      | Description                                                                |
|--------------|----------------------------------------------------------------------|
| dst          | Symmetric address of the target device's destination address on the local device                                  |
| src          | Source address on the local device                                                      |
| buf          | Buffer address                                                          |
| ub_size      | Buffer size                                                          |
| elem_size    | Number of elements                                                            |
| pe           | Destination PE                                                              |
| sync_id      | Synchronization ID                                                              |

## Precautions
Both `aclshmemx_sdma_put_nbi` and `aclshmemx_sdma_get_nbi` are non-blocking APIs. They return control to the caller right away and do not wait for the data transfer to finish. To ensure that the data transfer has finished, you can adopt one of the following methods:
1. For each kernel that calls `aclshmemx_sdma_put/get_nbi`, call the `aclshmemx_sdma_quiet` API within the operator after the SDMA task ends, and wait until all SDMA operations are complete.
Scenario: Subsequent operations within the operator depend on the completion of the SDMA task. For example, subsequent operations need to use the data transferred by the SDMA.
2. For each kernel that calls `aclshmemx_sdma_put/get_nbi`, call the `aclshmemx_sdma_notify_record` API within the operator after the SDMA task ends, then call the `aclrtWaitAndResetNotify` API on the host, and wait until the synchronization with the specified ID has completed. For details, see [NotifyWait Mechanism Usage Description](../notifywait/README_en.md).
Scenario: Kernels on other streams must wait for the SDMA task to finish before continuing execution.
