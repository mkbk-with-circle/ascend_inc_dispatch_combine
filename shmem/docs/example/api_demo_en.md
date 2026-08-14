# SHMEM API Examples
SHMEM provides two types of APIs: host-side APIs and device-side APIs. Host-side APIs are identified by the ACLSHMEM_HOST_API macro, and device-side APIs are identified by the ACLSHMEM_DEVICE macro.

This section provides samples of using common APIs.

## Init APIs
SHMEM initialization API example

### Initialization Statuses
```c++
enum {
    ACLSHMEM_STATUS_NOT_INITIALIZED = 0,    // Not initialized
    ACLSHMEM_STATUS_SHM_CREATED,           // Shared memory heap created
    ACLSHMEM_STATUS_IS_INITIALIZED,         // Initialized
    ACLSHMEM_STATUS_INVALID = INT_MAX,
};
```

### Attributes Required for Initialization
```c++
// Initialization attributes
typedef struct {
    int version;                            // Version
    int my_rank;                             // Current rank
    int n_ranks;                             // Total number of ranks
    char ip_port[ACLSHMEM_MAX_IP_PORT_LEN];      // IP address and port
    uint64_t local_mem_size;                  // Size of the locally allocated memory
    aclshmem_init_optional_attr_t option_attr; // Optional parameters
} aclshmemx_init_attr_t;

// Optional attributes
typedef struct {
    data_op_engine_type_t data_op_engine_type; // Data engine
    // timeout
    uint32_t shm_init_timeout;
    uint32_t shm_create_timeout;
    uint32_t control_operation_timeout;
    // apply available port in advance
    int32_t sockFd;
} aclshmem_init_optional_attr_t;
```

### Initialization Example
```c++
#include <iostream>
#include <unistd.h>
#include <acl/acl.h>
#include "shmem.h"
aclInit(nullptr);
int status;
int device_id = 0;
int rank_id = 0;
int n_ranks = 8;
uint64_t local_mem_size = 1024UL * 1024UL * 1024;
const char* test_global_ipport = "tcp://127.0.0.1:8666";
status = aclrtSetDevice(device_id);

aclshmemx_init_attr_t* attributes = new aclshmemx_init_attr_t{rank_id, n_ranks, test_global_ipport, local_mem_size, {0, ACLSHMEM_DATA_OP_MTE, 120, 120, 120}}; // Custom attributes
aclshmemx_init_attr(attributes);
delete attributes;
attributes = nullptr;

status = aclshmemx_init_status();
if (status == ACLSHMEM_STATUS_IS_INITIALIZED) {
    std::cout << "Init success!" << std::endl;
}
//################Your task#################

//#########################################
status = aclshmem_finalize();
aclrtResetDevice(device_id);
aclFinalize();

```

### Custom Log Printing

Custom log printing is optional. If the custom log printing function is not registered, logs are printed on the screen by default. The following is an API example:
```c
void cpp_logger_example(int level, const char* msg)
{
    // do print here
}

// set self-defined log
int ret = aclshmemx_set_extern_logger(cpp_logger_example);

// set log level. 0-debug, 1-info, 2-warn, 3-error
ret = aclshmemx_set_log_level(level);
```

### Registering the Private Key Password Decryption Function

This function is used for TCP communication on the service plane between multiple devices of SHMEM. To ensure communication security, the decryption feature is enabled by default. When using this function, you need to pass the initialized TLS information. For details about the TLS information format, see the example in *SECURITY*. The private key is in ciphertext, and the private key password is stored in ciphertext. The password needs to be decrypted using the registered decryption callback function. The caller implements the specific decryption logic.

```c
int my_key_password_decrypt_handler(const char *cipherText, size_t cipherTextLen, char *plainText, size_t &plainTextLen)
{
    // cipherText: input encrypted key password
    // plainText: output decrypted key password
    // plainTextLen: output decrypted key password length
    // do decrypt here
}

const char *pk = "xxx";
uint32_t pk_len = strlen(pk);

const char *password = "xxxx";
uint32_t pw_len = strlen(password);
int ret = aclshmemx_set_config_store_tls_key(pk, pk_len, password, pw_len, my_key_password_decrypt_handler);
```

To disable the encryption feature, call the API below. After the encryption feature is disabled, you do not need to call the `aclshmem_set_config_store_tls_key` API.
```c
int ret = aclshmemx_set_conf_store_tls(false, nullptr, 0);
```

### Team APIs
SHMEM team management API examples

### Host-side API example

```c++
// ################### Call the initialization API. ###########################
//...
// ###################### Sub-team division #############################
aclshmem_team_t team_odd;
int start = 1;
int stride = 2;
int team_size = 4;
aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);

// ##################### Host-side values ###############################
if (rank_id & 1) {

    // aclshmem_team_n_pes(team_odd): Returns the number of PEs in the team.
    int team_n_pes = aclshmem_team_n_pes(team_odd); // team_n_pes == team_size
    // aclshmem_team_my_pe(team_odd): Returns the number of the calling PE in the specified team.
    int team_my_pe = aclshmem_team_my_pe(team_odd); // team_my_pe == rank_id / stride
    // aclshmem_n_pes(): Returns the number of PEs running in the program.
    int n_pes = aclshmem_n_pes(); // n_pes == n_ranks
    // aclshmem_my_pe(): Returns the PE number of the local PE
    int my_pe = aclshmem_my_pe(); // my_pe == rank_id
}

// #################### Destroy allocations. ################################
aclshmem_team_destroy(team_odd);
// ################### Call the deinitialization API. ###########################
//...
```

### Device-side API example
```c++
class kernel_state_test {
public:
    __aicore__ inline kernel_state_test() {}
    __aicore__ inline void Init(GM_ADDR gva, aclshmem_team_t team_id)
    {
        gva_gm = (__gm__ int *)gva;
        team_idx= team_id;

        rank = aclshmem_my_pe();         // Obtain the current rank.
        rank_size = aclshmem_n_pes(); // Obtain the total number of ranks.
    }
    __aicore__ inline void Process()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        // ##################### Device-side values ###############################
        // aclshmem_int32_p is an API provided by RMA. It stores the result of the function's second input parameter on the device.

        // aclshmem_n_pes(): Returns the number of PEs running in the program.
        aclshmem_int32_p(gva_gm, aclshmem_n_pes(), rank);
        // aclshmem_my_pe(): Returns the PE number of the local PE
        aclshmem_int32_p(gva_gm + 1, aclshmem_my_pe(), rank);
        // aclshmem_team_my_pe(team_idx): Returns the number of the calling PE in the specified team.
        aclshmem_int32_p(gva_gm + 2, aclshmem_team_my_pe(team_idx), rank);
        // aclshmem_team_n_pes(team_idx): Returns the number of PEs in the team.
        aclshmem_int32_p(gva_gm + 3, aclshmem_team_n_pes(team_idx), rank);
        // aclshmem_team_translate_pe(team_idx, 1, ACLSHMEM_TEAM_WORLD): Translate a given PE number in one team into the corresponding PE number in another team.
        aclshmem_int32_p(gva_gm + 4, aclshmem_team_translate_pe(team_idx, 1, ACLSHMEM_TEAM_WORLD), rank);
    }
private:
    __gm__ int *gva_gm;
    aclshmem_team_t team_idx;

    int64_t rank;
    int64_t rank_size;
};

extern "C" __global__ __aicore__ void device_state_test(GM_ADDR gva, int team_id)
{
    kernel_state_test op;
    op.Init(gva, (aclshmem_team_t)team_id);
    op.Process();
}

void get_device_state(uint32_t block_dim, void* stream, uint8_t* gva, aclshmem_team_t team_id)
{
    device_state_test<<<block_dim, nullptr, stream>>>(gva, (int)team_id);
}
```
## Memory APIs
SHMEM memory management API example

```c++
// ###################Call the initialization API.###########################
//...
// ################## Call the memory management API. ###########################
// Allocate 1024 bytes and return the pointer to the allocated memory.
void *ptr = aclshmem_malloc(1024);
// Free the allocated memory corresponding to the pointer.
aclshmem_free(ptr);
// ################### Call the deinitialization API. ###########################
//...
```

## RMA APIs
SHMEM remote memory access API example

```c++
class kernel_p {
public:
    __aicore__ inline kernel_p() {}
    __aicore__ inline void Init(GM_ADDR gva, float val)
    {
        gva_gm = (__gm__ float *)gva;
        value = val;

        rank = aclshmem_my_pe();         // Obtain the current rank.
        rank_size = aclshmem_n_pes(); // Obtain the total number of ranks.
    }
    __aicore__ inline void Process()
    {
        // Put the value of `value` to the corresponding position of the shared memory gva_gm in (rank + 1) % rank_size.
        aclshmem_float_p(gva_gm, value, (rank + 1) % rank_size);
    }
private:
    __gm__ float *gva_gm;
    float value;

    int64_t rank;
    int64_t rank_size;
};

extern "C" __global__ __aicore__ void p_num_test(GM_ADDR gva, float val)
{
    kernel_p op;
    op.Init(gva, val);
    op.Process();
}

void put_one_num_do(uint32_t block_dim, void* stream, uint8_t* gva, float val)
{
    p_num_test<<<block_dim, nullptr, stream>>>(gva, val);
}
```

## Sync APIs
SHMEM synchronization management API example

```c++
// Task 1
// ...
// Blocks until all tasks are complete.
aclshmem_barrier_all();
// Task 2
// ...
```
