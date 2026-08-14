# Cache Maintenance Operation (CMO) Function Demonstration and Read Performance Test Example

## Function Description

This example demonstrates how to use the Cache Maintenance Operation (CMO) API of SHMEM to optimize the global memory (GM) access performance. The CMO API provides L2 cache management operations. It allows data to be prefetched from the GM to the L2 cache in advance, reducing data access latency and improving overall computing performance.

### L2 Cache Background

The Ascend AI Processor uses a multi-level cache architecture. The L2 cache is a level-2 cache located between the AI Core and the global memory (HBM) and has the following characteristics:

- **Capacity**: large-capacity high-speed cache (classic value for A2/A3: 192 MB)
- **Access speed**: The cache hit bandwidth is about 2 to 4 times the cache miss bandwidth.
- **Cache management**: Data can be loaded to the cache in advance to mask memory access latency.

By properly using CMO prefetch operations, the next batch of data can be prepared in advance while computation is ongoing, improving overall performance.

### Test Scenarios

This example compares the GM read performance under the following three cache prefetch policies:

1. **NO_PREFETCH**: Data is directly copied from the GM without any cache optimization.
2. **HOST_PREFETCH**: The host-side API `aclrtCmoAsync` is used to prefetch the entire block to be copied.
3. **DEVICE_BLOCK_PREFETCH**: CMO prefetch is performed on the memory locations to be copied for each block within the kernel.

The performance of the CMO API `aclshmemx_cmo_nbi` on the device side is tested, and the operation latency under different prefetch sizes is compared.

### Core APIs

#### CMO API (SHMEM Extension API)

```c
template <typename T>
void aclshmemx_cmo_nbi(__gm__ T *src, uint32_t elem_size, ACLSHMEMCMOTYPE cmo_type,
                     __ubuf__ T *buf, uint32_t ub_size, uint32_t sync_id);
```

- **Function**: Asynchronously triggers CMO operations on the device side and submits operation tasks to the STARS queue.
- **Parameter description**:
  - `src`: global memory address
  - `elem_size`: the number of elements
  - `cmo_type`: CMO operation type (Currently, only CMO_TYPE_PREFETCH is supported.)
  - `buf`: address of the temporary Unified Buffer
  - `ub_size`: Unified Buffer size (at least 64 bytes, 64-byte aligned)
  - `sync_id`: synchronization ID
- **Characteristics**: Based on the SDMA engine, core-level fine-grained control is supported.

##### CMO Operation Types

**Note**: Currently, SHMEM supports only the `CMO_TYPE_PREFETCH` operation.

- **CMO_TYPE_PREFETCH**: prefetch operation, which loads data from the global memory to the L2 cache in advance.
- **CMO_TYPE_WRITEBACK**: writeback operation, which writes the modified data in the L2 cache back to the global memory and retains a copy in the cache.
- **CMO_TYPE_INVALID**: invalidation operation, which discards the data blocks in the L2 cache.
- **CMO_TYPE_FLUSH**: flush operation, which forcibly writes the data in the L2 cache back to the global memory and removes the data from the cache.

#### SDMA Quiet API (SHMEM Extension API)

```c
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_quiet(AscendC::LocalTensor<T> &buf, uint32_t sync_id);
```

- **Function**: Waits for the completion of operation tasks in the STARS queue for synchronization.
- **Parameter description**:
  - `buf`: address of the temporary Unified Buffer
  - `ub_size`: Unified Buffer size
  - `sync_id`: synchronization ID
- **Characteristics**: An SDMA flag task is delivered, and the flag is polled until the operations in the STARS queue are complete.

## Environment Requirements

### Hardware Requirements
- Ascend AI Processor (Atlas 200I A2/A3, Atlas 300T A2/A3, Ascend950, etc.)
- Architecture compatibility: AArch64 and x86

### Software Dependencies
Refer to [CANN Version Description](../../docs/quickstart.md#43-cann) and [Compilation and Build Guide](../../docs/compilation_build_guide.md) to configure a CANN version that supports CMO.

| Platform | CANN Version Required for CMO | Toolkit Package | Ops Package |
| --- | --- | --- | --- |
| A2/A3 | CANN 9.0.0-beta.2 or later | Toolkit package 9.0.0-beta.2 or later: [Community Resources](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0-beta.2) | Ops package 9.0.0-beta.2 or later: [Community Resources](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0-beta.2) |
| Ascend950 | CANN 9.1.0 or later | Toolkit package 9.1.0: [x86_64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260610120325172/Ascend-cann-toolkit_9.1.0_linux-x86_64.run) / [aarch64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260610120325172/Ascend-cann-toolkit_9.1.0_linux-aarch64.run) | Ops package 9.1.0: [950 x86_64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260610120325172/Ascend-cann-950-ops_9.1.0_linux-x86_64.run) / [950 aarch64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260610120325172/Ascend-cann-950-ops_9.1.0_linux-aarch64.run) |

Install the toolkit and Ops packages in the same directory:

```bash
# Customize the CANN installation directory as required.
export INSTALL_PATH=/home/user/ascend
chmod +x Ascend-cann-toolkit_{cann_version}_linux-$(uname -m).run
chmod +x Ascend-cann-{soc_name}-ops_{cann_version}_linux-$(uname -m).run
./Ascend-cann-toolkit_{cann_version}_linux-$(uname -m).run --install --install-path=${INSTALL_PATH}
./Ascend-cann-{soc_name}-ops_{cann_version}_linux-$(uname -m).run --install --install-path=${INSTALL_PATH}
source ${INSTALL_PATH}/ascend-toolkit/set_env.sh
```

### Function Dependencies

**Important**: In this example, the CMO API `aclshmemx_cmo_nbi` on the device side depends on the SDMA function. You need to configure `attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_SDMA` by referring to example/sdma or example/cmo to start the SDMA engine.

### Platform Support

Ascend950 supports only CMO.

Read/write data transfers through the SDMA put/get interfaces are not currently supported. Therefore, SDMA put/get interfaces such as `aclshmemx_sdma_put_nbi` and `aclshmemx_sdma_get_nbi` are not applicable to Ascend950 and later platforms.

## Build Procedure

### 1. Build and install the SHMEM software package.

```bash
cd shmem/
bash scripts/build.sh -package
./install/*/SHMEM_1.0.0_linux-*.run --install
source install/set_env.sh
```

### 2. Build a sample program.

```bash
cd shmem/
bash scripts/build.sh -examples
```

After the build is successful, the executable file is stored in `build/bin/cmo`.

## Running Method

```bash
cd shmem/examples/cmo
bash run.sh -pes ${PEs} -type ${TYPE}
```

### Parameters

- PEs: the number of devices (NPUs) used for running the program, limited to a single server
- TYPE: type of the data to be transferred. Currently, the following data types are supported: int, uint8, int64, fp16, and fp32.

### Example: Using Two NPUs to Test int Data
```bash
bash run.sh -pes 2 -type int
```

## Output Results

### Console Output

When the program is running, the completion information of each PE is displayed:
```
PE 0 Finished!
PE 1 Finished!
[SUCCESS] demo run success in pe 0
[SUCCESS] demo run success in pe 1
```

### CSV File Output

The program generates the following CSV files in the `output/` directory:

#### 1. `{PE_ID}_band.csv` - Bandwidth Performance Test Results

The file contains the following columns:
- `loop_times`: number of loops (100 by default)
- `copy_size_per_loop`: size of data copied in each loop (less than the L2 cache size to verify the effect of full-block prefetch)
- `blocks`: number of blocks used
- `copypad_size`: data size of a single DataCopy operation
- `no_prefetch_time/us`: average copy time without prefetching (in microseconds)
- `no_prefetch_band/Gbps`: average copy bandwidth without prefetching (in GB/s)
- `host_prefetch_time/us`: average copy time after host-side full-block prefetching (in microseconds)
- `host_prefetch_band/Gbps`: average copy bandwidth after host-side prefetching (in GB/s)
- `device_block_prefetch_time/us`: average copy time after device block prefetching (in microseconds)
- `device_block_prefetch_band/Gbps`: average copy bandwidth after device block prefetching (in GB/s)

#### 2. `{PE_ID}_cmo.csv` - CMO Operation Latency Test Results

The file contains the following columns:
- `loop_times`: number of loops (100 by default)
- `blocks`: number of blocks used
- `cmo_size`: data size of the CMO operation
- `cmo_send_time_p05/us`: 5th percentile of CMO send time (in microseconds)
- `cmo_send_time_p50/us`: 50th percentile of CMO send time (in microseconds)
- `cmo_send_time_p95/us`: 95th percentile of CMO send time (in microseconds)
- `cmo_flag_time_p05/us`: 5th percentile of CMO synchronization wait time (in microseconds)
- `cmo_flag_time_p50/us`: 50th percentile of CMO synchronization wait time (in microseconds)
- `cmo_flag_time_p95/us`: 95th percentile of CMO synchronization wait time (in microseconds)

### Performance Metrics

- **Bandwidth**: used to measure the data transmission rate, in GB/s
- **Latency**: used to measure operation completion time, in microseconds
- **Percentile**: used to collect statistics on the distribution. p50 indicates the median.

## References

- [CANN Application Development API Documentation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900beta1/appdevg/acldevg/acldevg_0001.html)
- [Memory Management aclrtCmoAsync](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/appdevgapi/aclcppdevg_03_0123.html)
