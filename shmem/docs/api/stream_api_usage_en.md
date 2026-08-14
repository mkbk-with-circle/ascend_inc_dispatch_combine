# SHMEM C++ Stream API Usage

## I. Overview

This document describes how to use four stream-based SHMEM C++ APIs. These APIs allow data transmission and signal operations on a specified ACL stream.

**Cross-server support**:

| API                                        | Cross-Server Support| Description                                                                                                |
| ------------------------------------------ | ---- | -------------------------------------------------------------------------------------------------- |
| aclshmemx\_putmem\_on\_stream              | Supported  | MTE is preferred when HCCS is connected. Otherwise, RDMA is used when the RDMA links are available.                                                                    |
| aclshmemx\_getmem\_on\_stream              | Supported  | MTE is preferred when HCCS is connected. Otherwise, RDMA is used when the RDMA links are available.                                                                    |
| aclshmemx\_signal\_op\_on\_stream          | Partially supported| `ACLSHMEM_SIGNAL_SET`: MTE is preferred when HCCS is connected. Otherwise, RDMA is used when the RDMA links are available. `ACLSHMEM_SIGNAL_ADD` does not support cross-server RDMA, but supports MTE cross-server communication when the HCCS is connected.|
| aclshmemx\_signal\_wait\_until\_on\_stream | Not supported | Semantic: Wait for the address on the card to reach a value.                                                                                   |

Policy for selecting a data transmission path in the cross-server scenario:

- **MTE preferred**: When HCCS is connected, the MTE path is preferentially used.
- **RDMA**: When HCCS is disconnected but RDMA is available, the RDMA path is automatically used for cross-server communication.

For automatic selection of the MTE or RDMA path, perform initialization in the following manner:

```cpp
aclshmemx_init_attr_t attributes;
// Configure the MTE and RDMA paths.
attributes.option_attr.data_op_engine_type = static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_ROCE);
int status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
```

For details about the initialization, see the `void test_cross_init()` function of `tests\unittest\host\main_test.cpp`.

***

## 2. API Description

### 1. aclshmemx\_putmem\_on\_stream

**Function:**

Copies data from the symmetric memory of the local card to the address of a specified PE through a specified stream. This is a synchronous API.

**Cross-server support**: Supported. MTE is preferred when HCCS is connected. Otherwise, RDMA is used when the RDMA links are available.

**Prototype**:

```cpp
ACLSHMEM_HOST_API void aclshmemx_putmem_on_stream(void* dst, void* src, size_t elem_size, int32_t pe, aclrtStream stream);
```

**Parameter description**:

| Parameter        | Type         | Description                  |
| ---------- | ----------- | -------------------- |
| dst        | void\*      | Pointer to the symmetric address of the target PE's destination address on the local card |
| src        | void\*      | Pointer to the local source data            |
| elem\_size | size\_t     | Size of the data to be transferred, in bytes        |
| pe         | int32\_t    | ID of the target PE             |
| stream     | aclrtStream | Stream to be used (If the value is `NULL`, the default stream is used.)|

**Sample**:
`tests\unittest\host\mem\shmem_host_put_stream_test.cpp` provides a complete example of how to use the `aclshmemx_putmem_on_stream` API to transmit data through a specified stream. The following shows only the core process.

```cpp
// Pseudocode example: process of using aclshmemx_putmem_on_stream

// 1. Perform initialization (MTE and RDMA need to be configured in cross-server scenarios).
aclshmemx_init_attr_t attributes;
attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_ROCE;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

// 2. Create a stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// 3. When RDMA APIs are called across servers, aclshmem_malloc must be used to allocate symmetric memory. aclrtMalloc cannot be used.
size_t data_size = 1024; // 1KB
int32_t target_pe = 1; // Target PE ID
void* src_ptr = aclshmem_malloc(data_size);
void* dst_ptr = aclshmem_malloc(data_size);
// 4. Initialize and assign a value to src_ptr.
// 5. Perform the put operation.
aclshmemx_putmem_on_stream(dst_ptr, src_ptr, data_size, target_pe, stream);

// 6. When RDMA APIs are called across servers, aclshmemx_handle_wait must be called to wait until the transmission is complete.
aclshmem_handle_t handle;
handle.team_id = ACLSHMEM_TEAM_WORLD;
aclshmemx_handle_wait(handle, stream);
aclrtSynchronizeStream(stream);

// 7. Destroy allocations.
aclshmem_free(src_ptr);
aclshmem_free(dst_ptr);
aclshmem_finalize();
aclrtDestroyStream(stream);
aclrtResetDevice(device_id);
aclFinalize();
```

***

### 2. aclshmemx\_getmem\_on\_stream

**Function**:

Copies continuous data on the specified PE in the symmetric memory to the address of the local PE through the specified stream. This is a synchronous API.

**Cross-server support**: Supported. MTE is preferred when HCCS is connected. Otherwise, RDMA is used when the RDMA links are available.

**Prototype**:

```cpp
ACLSHMEM_HOST_API void aclshmemx_getmem_on_stream(void* dst, void* src, size_t elem_size, int32_t pe, aclrtStream stream);
```

**Parameter description**:

| Parameter        | Type         | Description                     |
| ---------- | ----------- | ----------------------- |
| dst        | void\*      | Pointer (destination address) on the local PE         |
| src        | void\*      | Symmetrical address of the remote PE's source data address on the local card    |
| elem\_size | size\_t     | Size of the data to be transferred, in bytes           |
| pe         | int32\_t    | ID of the remote PE                |
| stream     | aclrtStream | ACL stream to be used (If the value is `NULL`, the default stream is used.)|

**Sample**:
`tests\unittest\host\mem\shmem_host_get_stream_test.cpp` provides a complete example of how to use the `aclshmemx_getmem_on_stream` API to transmit data through a specified stream. The following shows only the core process.

```cpp
// Pseudocode example: process of using aclshmemx_getmem_on_stream

// 1. Perform initialization (MTE and RDMA need to be configured in cross-server scenarios).
aclshmemx_init_attr_t attributes;
attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_ROCE;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

// 2. Create a stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// 3. When RDMA APIs are called across servers, aclshmem_malloc must be used to allocate symmetric memory. aclrtMalloc cannot be used.
size_t data_size = 1024; // 1KB
int32_t source_pe = 0; // Source PE ID
void* src_ptr = aclshmem_malloc(data_size);
void* dst_ptr = aclshmem_malloc(data_size);

// 4. Perform the get operation.
aclshmemx_getmem_on_stream(dst_ptr, src_ptr, data_size, source_pe, stream);

// 5. When RDMA APIs are called across servers, aclshmemx_handle_wait must be called to wait until the transmission is complete.
aclshmem_handle_t handle;
handle.team_id = ACLSHMEM_TEAM_WORLD;
aclshmemx_handle_wait(handle, stream);

aclrtSynchronizeStream(stream);

// 6. Destroy allocations.
aclshmem_free(src_ptr);
aclshmem_free(dst_ptr);
aclshmem_finalize();
aclrtDestroyStream(stream);
aclrtResetDevice(device_id);
aclFinalize();
```

***

### 3. aclshmemx\_signal\_op\_on\_stream

**Function**:

Performs operations on signal variables on a specified PE. The operations are performed in a given stream. This is a synchronous API. It is often used together with `aclshmemx_signal_wait_until_on_stream`.

**Cross-server support**: Partially supported

- `ACLSHMEM_SIGNAL_SET`: MTE cross-server communication and RDMA are supported when HCCS is connected.
- `ACLSHMEM_SIGNAL_ADD`: Cross-server RDMA is not supported. MTE cross-server communication is supported when HCCS is connected.

**Prototype**:

```cpp
ACLSHMEM_HOST_API void aclshmemx_signal_op_on_stream(int32_t *sig_addr, int32_t signal, int sig_op, int pe, aclrtStream stream);
```

**Parameter description**:

| Parameter       | Type         | Description                                                                         |
| --------- | ----------- | --------------------------------------------------------------------------- |
| sig\_addr | int32\_t\*  | Symmetric address of the signal variable on the local card                                                              |
| signal    | int32\_t    | Value used for atomic operations                                                                   |
| sig\_op   | int         | Operation to be performed on the remote signal variable. Supported operation: `ACLSHMEM_SIGNAL_SET`: sets the signal value. `ACLSHMEM_SIGNAL_ADD`: accumulates the signal value.|
| pe        | int         | ID of the remote PE                                                                    |
| stream    | aclrtStream | ACL stream to be used (If the value is `NULL`, the default stream is used.)                                                    |

**Sample**:
`tests\unittest\host\sync\signal\signal_host_test.cpp` provides a complete example of how to use the `aclshmemx_signal_op_on_stream` API to perform signal operations on a specified stream. The following shows only the core process.

```cpp
// Pseudocode example: process of using aclshmemx_signal_op_on_stream

// 1. Perform initialization (MTE and RDMA need to be configured in cross-server scenarios).
aclshmemx_init_attr_t attributes;
attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_ROCE;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

// 2. Create a stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// 3. When RDMA APIs are called across servers, aclshmem_malloc must be used to allocate symmetric memory. aclrtMalloc cannot be used.
int32_t target_pe = 1; // Target PE ID
int32_t* signal_var = (int32_t*)aclshmem_malloc(sizeof(int32_t));

// 4. Set the signal value (cross-server RDMA is supported).
aclshmemx_signal_op_on_stream(signal_var, 1, ACLSHMEM_SIGNAL_SET, target_pe, stream);

// When RDMA APIs are called across servers, aclshmemx_handle_wait must be called to wait until the transmission is complete.
aclshmem_handle_t handle;
handle.team_id = ACLSHMEM_TEAM_WORLD;
aclshmemx_handle_wait(handle, stream);
// 5. Accumulate the signal value. (Cross-server RDMA is not supported. MTE cross-server communication is supported only when HCCS is connected.)
aclshmemx_signal_op_on_stream(signal_var, 10, ACLSHMEM_SIGNAL_ADD, target_pe, stream);

// 6. Wait until the operation is complete.
aclrtSynchronizeStream(stream);

// 7. Destroy allocations.
aclshmem_free(signal_var);
aclshmem_finalize();
aclrtDestroyStream(stream);
aclrtResetDevice(device_id);
aclFinalize();
```

***

### 4. aclshmemx\_signal\_wait\_until\_on\_stream

**Function**:

This is a synchronous API. Calling this API will be blocked until the value of `sig\_addr` on the PE meets the waiting condition specified by the comparison operator `cmp` and the comparison value `cmp\_val`.

**Cross-server support**: Not supported.

**Prototype**:

```cpp
ACLSHMEM_HOST_API void aclshmemx_signal_wait_until_on_stream(int32_t *sig_addr, int cmp, int32_t cmp_val, aclrtStream stream);
```

**Parameter description**:

| Parameter       | Type         | Description                                                                                                                                                |
| --------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| sig\_addr | int32\_t\*  | Local address of the source signal variable                                                                                                                                        |
| cmp       | int         | Comparison operator. Supported comparison operator: `ACLSHMEM_CMP_EQ` (equal to), `ACLSHMEM_CMP_NE` (not equal to), `ACLSHMEM_CMP_GT` (greater than), `ACLSHMEM_CMP_GE` (greater than or equal to), `ACLSHMEM_CMP_LT` (less than), or `ACLSHMEM_CMP_LE` (less than or equal to)|
| cmp\_val  | int32\_t    | Value used for comparison                                                                                                                                           |
| stream    | aclrtStream | ACL stream to be used (If the value is `NULL`, the default stream is used.)                                                                                                                           |

**Sample**:
`tests\unittest\host\sync\signal\signal_host_test.cpp` provides a complete example of how to use the `aclshmemx_signal_wait_until_on_stream` API to wait for a signal condition on a specified stream. The following shows only the core process.

```cpp
// Pseudocode example: process of using aclshmemx_signal_wait_until_on_stream

// 1. Perform initialization.
aclshmemx_init_attr_t attributes;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

// 2. Create a stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// 3. When RDMA APIs are called across servers, aclshmem_malloc must be used to allocate symmetric memory. aclrtMalloc cannot be used.
int32_t* signal_var = (int32_t*)aclshmem_malloc(sizeof(int32_t));

// 4. Set the signal value.
aclshmemx_signal_op_on_stream(signal_var, 2, ACLSHMEM_SIGNAL_SET, target_pe, stream);

// 5. Wait until the local signal is equal to the specified value (block until the condition is met). This is usually the case in multi-device scenarios.
aclshmemx_signal_wait_until_on_stream(signal_var, ACLSHMEM_CMP_EQ, 2, stream);

// 6. Wait until the operation is complete.
aclrtSynchronizeStream(stream);

// 7. Destroy allocations.
aclshmem_free(signal_var);
aclshmem_finalize();
aclrtDestroyStream(stream);
aclrtResetDevice(device_id);
aclFinalize();
```

***

### Typical Application Scenario: Ring Signal Synchronization

The following example shows the ring signal synchronization between multiple PEs. Each PE waits for the signal set by the previous PE and then sets the signal for the next PE. For a complete example, see the `test_signal_eq_all_pes` function in `tests\unittest\host\sync\signal\signal_host_test.cpp`.

```cpp
// Pseudocode example: ring signal synchronization
// Assume that there are *N* PEs that form a ring: PE0 -> PE1 -> PE2 -> ... -> PE(N-1) -> PE0.

// 1. Perform initialization.
aclshmemx_init_attr_t attributes;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

// 2. Create a stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// 3. Allocate signal variables (symmetric memory).
int32_t* sig_addr = (int32_t*)aclshmem_malloc(sizeof(int32_t));
aclrtMemset(sig_addr, sizeof(int32_t), 0, sizeof(int32_t)); // Initialize to 0.

// 4. Compute the ring relationship.
int32_t expected_signal = pe_id + 1;        // Expected signal value (from the previous PE)
int next_pe = (pe_id + 1) % n_pes;         // ID of the next PE
int32_t my_signal = next_pe + 1;            // Signal value to be sent to the next PE

// 5. Set the signal value of the next PE.
aclshmemx_signal_op_on_stream(sig_addr, my_signal, ACLSHMEM_SIGNAL_SET, next_pe, stream);

// Call aclshmemx_handle_wait to wait until the signal setting is complete.
aclshmem_handle_t handle;
handle.team_id = ACLSHMEM_TEAM_WORLD;
aclshmemx_handle_wait(handle, stream);

aclrtSynchronizeStream(stream);

// 6. Wait until the current PE receives the expected signal value (block until the condition is met).
aclshmemx_signal_wait_until_on_stream(sig_addr, ACLSHMEM_CMP_EQ, expected_signal, stream);
aclrtSynchronizeStream(stream);

// 7. Verify the signal value.
int32_t host_value;
aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
// host_value should be equal to expected_signal.

// 8. Destroy allocations.
aclshmem_free(sig_addr);
aclshmem_finalize();
aclrtDestroyStream(stream);
aclrtResetDevice(device_id);
aclFinalize();
```

***
## 3. Precautions

1. **Memory allocation**: When RDMA APIs are called across servers, `aclshmem_malloc` must be used to allocate symmetric memory.
2. **Cross-server communication**:
   - `aclshmemx_putmem_on_stream` and `aclshmemx_getmem_on_stream` support cross-server communication. When HCCS is connected, MTE is preferentially used. Otherwise, RDMA is used when the RDMA links are available.
   - `aclshmemx_signal_op_on_stream`: `ACLSHMEM_SIGNAL_SET` supports cross-server RDMA. `ACLSHMEM_SIGNAL_ADD` does not support cross-server RDMA. MTE cross-server communication is supported when HCCS is connected.
   - `aclshmemx_signal_wait_until_on_stream` does not support cross-server RDMA but supports MTE cross-server communication when HCCS is connected.
   - In cross-server scenarios, ensure that the network configuration is correct and related environment variables are correctly configured.
   - **RDMA path synchronization**: When RDMA APIs are used, `aclshmemx_handle_wait` needs to be called to ensure that data has been received.
