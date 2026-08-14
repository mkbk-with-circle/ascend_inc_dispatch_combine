# NotifyWait Mechanism Usage Guide
## Environment Requirements and Preparations
The SDMA feature is newly supported in CANN 9.0.0 or later (trial version). You need to download and install the following CANN and OPS software packages first:
- Toolkit package ([CANN master OBP image website](https://mirror-centralrepo.devcloud.cn-north-4.huaweicloud.com/artifactory/cann-run-mirror/software/master/))
- ops-legacy package (Download a required version based on the hardware platform: [A2 x86_64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260520_newest/cann-910b-ops-legacy_9.1.0_linux-x86_64.run)/[A2 aarch64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260520_newest/cann-910b-ops-legacy_9.1.0_linux-aarch64.run)/[A3 x86_64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260520_newest/cann-A3-ops-legacy_9.1.0_linux-x86_64.run)/[A3 aarch64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260520_newest/cann-A3-ops-legacy_9.1.0_linux-aarch64.run))
## Example Execution Description
1. Build and install the software package in the `shmem/` directory:
```bash
bash scripts/build.sh -package
./install/*/SHMEM_1.0.0_linux-*.run --install
```

2. Build the examples in the `shmem/` directory:
```bash
bash scripts/build.sh -examples
```

3. Run the demo in the `shmem/examples/notifywait` directory:
```bash
bash run.sh -pes ${PES} -type ${TYPES}
````
- **Parameter description:**
    - PES: the number of devices (NPUs) used for running the demo, limited to a single server
    - TYPES: type of the data to be transferred. Currently, the following data types are supported: int, uint8, int64, and fp32.
## NotifyWait Usage Description

### Example
![notifywait](../../docs/images/notifywait_en.png)
```c++
//Step 1:
Kernel 1 on stream 1: Call the SDMA API to transfer data and call aclshmemx_sdma_notify_record.
//Step 2:
Host: aclrtWaitAndResetNotify(notify_id, stream2, 0)
//Step 3:
Kernel 2 on stream 2: Use the data transferred via SDMA.

```

### Usage Description
In `aclshmemx_sdma_notify_record`, a record-type SQE is issued to STARS. The host then waits for the notify record to complete before continuing with subsequent kernels. Compared to `aclshmemx_sdma_quiet`, which relies on AIV flag polling, this mechanism allows timely release of AIV resources.
