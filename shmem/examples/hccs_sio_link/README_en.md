# HCCS SIO Link

The HCCS/SIO link test tool is designed to verify the correctness of SIO and HCCS links between NPUs.

> **Note**: This feature is not available in the current CANN release. It will be supported in CANN 9.1.0.

## Link Description

In the A3 chip, each NPU contains two dies. The dies are interconnected via the original SIO link. Building upon this existing SIO link, the solution introduces an additional HCCS link, enabling dual-path parallel transmission via both SIO and HCCS. This architecture significantly accelerates inter-die data transfer.

- **SIO**: the original interconnect link between dies. After SHMEM initialization, the default virtual address (VA) access to the peer die is routed through the SIO link.

- **HCCS**: a new interconnect link between dies. By calling [`aclrtMemMapSelectedLink`](https://gitcode.com/cann/runtime/blob/master/docs/zh/api_ref/11-04_virtual_memory_management.md#aclrtmemmapselectedlink), a new VA can be mapped to the same physical address (PA) as the original SIO VA. `ACL_RT_MEM_LINK_IDX_1` is used to select an HCCS link. In this case, access through the new VA is routed via the HCCS link.

- **SIO + HCCS dual-link parallel transmission**: Data can be transmission simultaneously over both SIO and HCCS links, fully utilizing dual-link bandwidth to enhance transmission performance.

## Core Functions

### `setup_hccs_mapping`

An HCCS link mapping is set up to create a VA channel for the current PE for accessing the heap memory of the peer PE through the HCCS link. This function involves the following steps:

1. **Obtaining a local heap base address**: Call `aclshmemx_get_heap_base()` to obtain the symmetric heap base address of the current PE.
2. **Translating to a peer address**: Call `aclshmem_ptr()` to convert the local heap base address into a peer-accessible VA (`peer_heap_base`). This address is routed through the SIO link.
3. **Reserving a VA space**: Call `aclrtReserveMemAddress()` to reserve an unmapped VA range (`hccs_ptr`) in the VA space of the current PE.
4. **Mapping to an HCCS link**: Call [`aclrtMemMapSelectedLink`](https://gitcode.com/cann/runtime/blob/master/docs/zh/api_ref/11-04_virtual_memory_management.md#aclrtmemmapselectedlink) to map the reserved VA to the PA corresponding to `peer_heap_base` and specify `ACL_RT_MEM_LINK_IDX_1` to select an HCCS link. At this point, accesses via `hccs_ptr` are routed through the HCCS link, while accesses via the original SIO VA continue to use the SIO link. Both VAs point to the same PA but traverse different physical links.

> **Key principle**: The SIO and HCCS links share the same PA, but the traffic is distributed using different VAs and link indexes, thus supporting dual-link parallel transmission.

#### Parameter Description

| Parameter| Type| Description|
|------|------|------|
| `peer` | `int` | ID of the peer PE, that is, the target PE for which the HCCS mapping is to be established.|
| `local_mem_size` | `uint64_t` | Identical to the symmetric heap size (in bytes) specified during `aclshmemx_init_attr` initialization. The actual mapped range size is `local_mem_size + ACLSHMEM_EXTRA_SIZE`.|
| `hccs_ptr` | `void **` | Output parameter. Upon success, the function returns the base virtual address of the HCCS link mapping. If it fails, the value depends on the step at which the error occurred. If `aclrtReserveMemAddress` succeeds but subsequent steps fail, `*hccs_ptr` may contain a reserved but unmapped address. In this case, you need to call `teardown_hccs_mapping` or `HccsMappingGuard` to release the addresses to prevent leaks.|

#### Return Values

- `true`: The HCCS mapping is successfully established, and `*hccs_ptr` is a valid HCCS base virtual address.
- `false`: The mapping fails at one of the steps. In this case, the function prints an error message and returns.

### `teardown_hccs_mapping`

This function is paired with `setup_hccs_mapping` and is used to release HCCS mapping resources:

1. Call `aclrtUnmapMem()` to remove the mapping between the VA and PA.
2. Call `aclrtReleaseMemAddress()` to release the reserved VA space.

> **Note**: In the sample code, the `HccsMappingGuard` structure automatically calls `teardown_hccs_mapping` during destruction using the RAII mechanism to ensure that mapping resources are not leaked.

## Build

To build this example, enable the `-examples` build option. During the build process, CMake automatically detects whether the current CANN version supports the `aclrtMemMapSelectedLink` function. If it does, CMake automatically builds this example.

```bash
bash scripts/build.sh -examples
```

Build output: `build/bin/hccs_sio_link`

## Prerequisites

- The SHMEM project has been built as described above.
- The environment variable `ASCEND_HOME_PATH` has been properly configured.

> **Multi-instance description**: The function `aclshmemx_get_heap_base` returns the heap base address of the currently active instance. In multi-instance scenarios, you must first switch to the target instance using `aclshmemx_instance_ctx_set_impl`, and then call `aclshmemx_get_heap_base`.

## Usage

This tool is started using the `run.sh` script. The script starts a background process for each PE, and communication between PEs is established through SHMEM initialization.

### Run Command

```bash
bash run.sh [option]
```

### Typical Cases

```bash
# Default configuration: two PEs, full-link test for SIO + HCCS, 4 KB of data, and int type
bash run.sh

# If four PEs are specified, test the HCCS link only.
bash run.sh -pes 4 -mode hccs

# If eight PEs, with 8 MB of data and the fp32 type, are specified, test the SIO link only.
bash run.sh -pes 8 -size 8 -type fp32 -mode sio

# SIO + HCCS hybrid test (3/5 of data via SIO and 2/5 via HCCS)
bash run.sh -mode mixed
```

## Parameter Description

| Parameter| Default Value| Description|
|------|--------|------|
| `-ipport` | `tcp://127.0.0.1:8766` | Communication initialization address|
| `-pes` | `2` | Total number of PEs (same as the number of NPUs) involved in the test|
| `-fpe` | `0` | ID of the first PE|
| `-fnpu` | `0` | ID of the first NPU|
| `-type` | `int` | Test data type: `int` / `int64` / `fp32`|
| `-mode` | `all` | Test mode (see the table below)|
| `-size` | `4` | Data size (KB) of each PE|

### Test Modes

| Mode| Description|
|------|------|
| `sio` | SIO link correctness test|
| `hccs` | HCCS link correctness test|
| `all` | SIO + HCCS full-link correctness test|
| `mixed` | SIO + HCCS hybrid link correctness test (3/5 data via SIO and 2/5 via HCCS)|

## Output Example

Correctness test:

```
PE 0: [SIO] path verification PASSED for PE 1
PE 1: [SIO] path verification PASSED for PE 0
PE 0: [HCCS] path verification PASSED for PE 1
PE 1: [HCCS] path verification PASSED for PE 0
```

Hybrid test:

```
PE 0: [MIXED-SIO] path verification PASSED for PE 1
PE 0: [MIXED-HCCS] path verification PASSED for PE 1
PE 1: [MIXED-SIO] path verification PASSED for PE 0
PE 1: [MIXED-HCCS] path verification PASSED for PE 0
```
