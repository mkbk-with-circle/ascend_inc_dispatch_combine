# Initialization
The sample project is stored in the `examples/init` directory.

SHMEM provides three flags for initialization:
    - `ACLSHMEMX_INIT_WITH_DEFAULT`: default initialization process, which can be completed without dependency on a third-party library. However, the specified IP address and port must be idle.
    - `ACLSHMEMX_INIT_WITH_MPI`: initialization process that depends on the multi-process management capability of MPI.
    - `ACLSHMEMX_INIT_WITH_UNIQUEID`: initialization process that depends on the broadcast capability (such as MPI_Bcast/torch.distributed.broadcast) of a third-party library.

In this sample, all initialization process calling methods for SHMEM are implemented. You can compile and run the initialization processes corresponding to different flags by using script parameters. By default, the loopback address 127.0.0.1 and port 8666 are used. In this sample, only a single server is used. Ensure that the port is idle and can be bound.

## ACLSHMEMX_INIT_WITH_DEFAULT

### Sample Running
Call `scripts/build.sh` to control the backend in the root directory.
```bash
bash scripts/build.sh
```

```bash
cd examples/init
# Build and run the default initialization process of two PEs.
bash run.sh -mode default -pesize 2
```
### Code Implementation
Before calling the SHMEM initialization API, you need to create attributes. A method for creating attributes is provided in this sample. You can also set attributes based on the `aclshmemx_init_attr_t` definition as needed.
```cpp
// Function for setting attributes
int32_t test_set_attr(int32_t my_pe, int32_t n_pes, uint64_t local_mem_size, const char *ip_port,
                       aclshmemx_uniqueid_t *default_flag_uid, aclshmemx_init_attr_t *attributes)
{
    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), (size_t)(ACLSHMEM_MAX_IP_PORT_LEN - 1));

        std::copy_n(ip_port, ip_len, attributes->ip_port);
        if (attributes->ip_port[0] == '\0') {
            return ACLSHMEM_INVALID_VALUE;
        }
    }

    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = my_pe;
    attributes->n_pes = n_pes;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT,
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void *>(default_flag_uid);
    aclshmemx_uniqueid_t *uid_args = (aclshmemx_uniqueid_t *)(attributes->comm_args);

    return ACLSHMEM_SUCCESS;
}
```

```cpp
// Default initialization process
int run_main(int argc, char* argv[]) {
    int pe = atoi(argv[1]);
    int pe_size = atoi(argv[2]);
    std::string ipport = argv[3];
    aclshmemx_uniqueid_t default_flag_uid = ACLSHMEM_UNIQUEID_INITIALIZER;
    aclshmemx_init_attr_t attributes;

    int status = ACLSHMEM_SUCCESS;
    // Initialize ACL.
    aclInit(nullptr);
    aclrtSetDevice(pe);

    uint64_t local_mem_size = 1024 * 1024 * 1024;
    // Create attributes.
    test_set_attr(pe, pe_size, local_mem_size, ipport.c_str(), &default_flag_uid, &attributes);
    // Initialization
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    /*
    Your task
    */

    // Deinitialization
    status = shmem_finalize();
    aclrtResetDevice(pe);
    aclFinalize();

    return 0;
}
```
## ACLSHMEMX_INIT_WITH_MPI

### Sample Running
Before compilation, you need to install and configure MPI environment variables. Otherwise, the MPI capability .so file cannot be compiled.
```bash
# In the commands, MPICH is installed in the default path. Replace the path with the actual path.
export PATH=/usr/local/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH
```

Call `scripts/build.sh` in the root directory.
```bash
bash scripts/build.sh
```

```bash
cd examples/init
# Build and run the MPI initialization process of two PEs.
bash run.sh -mode mpi -pesize 2
```
### Code Implementation
```cpp
// MPI initialization process
int run_main() {
    // MPI initialization
    MPI_Init(nullptr, nullptr);
    int pe;
    int pe_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &pe);
    MPI_Comm_size(MPI_COMM_WORLD, &pe_size);
    int status = ACLSHMEM_SUCCESS;
    // Initialize ACL.
    aclInit(nullptr);
    aclrtSetDevice(pe);
    uint64_t local_mem_size = 1024 * 1024 * 1024;
    // Create attributes. If the MPI is used, `ipport` is not required.
    aclshmemx_init_attr_t attributes = {
        pe, pe_size, "", local_mem_size, {0, ACLSHMEM_DATA_OP_MTE, 120, 120, 120}};
    // Initialization
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_MPI, &attributes);

    /*
    Your task
    */

    // Deinitialization
    status = shmem_finalize();
    aclrtResetDevice(pe);
    aclFinalize();
    MPI_Finalize();
    return 0;
}
```
## ACLSHMEMX_INIT_WITH_UNIQUEID
### Sample Running
Before compilation, you need to install and configure MPI environment variables. Otherwise, the MPI capability .so file cannot be compiled.
```bash
# In the commands, MPICH is installed in the default path. Replace the path with the actual path.
export PATH=/usr/local/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH
```

Call <idp:inline displayname="code" id="code3899112363513">scripts/build.sh</idp:inline> to control the backend in the root directory.
```bash
bash scripts/build.sh
```

```bash
cd examples/init
# Build and run the MPI initialization process of two PEs.
bash run.sh -mode mpi -pesize 2
```
### Code Implementation
```cpp
// UID initialization process
int run_main() {
    // MPI initialization
    MPI_Init(nullptr, nullptr);
    int pe;
    int pe_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &pe);
    MPI_Comm_size(MPI_COMM_WORLD, &pe_size);
    int status = ACLSHMEM_SUCCESS;
    // Initialize ACL.
    aclInit(nullptr);
    aclrtSetDevice(pe);

    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t uid = ACLSHMEM_UNIQUEID_INITIALIZER;

    int64_t local_mem_size = 1024 * 1024 * 1024;
    // Obtain the UID of PE 0.
    if (pe == 0) {
        status = aclshmemx_get_uniqueid(&uid);
    }
    // Broadcast the UID of PE 0 to the other PE.
    MPI_Bcast(&uid, sizeof(aclshmemx_uniqueid_t), MPI_UINT8_T, 0, MPI_COMM_WORLD);
    // Call the `set` API for the UID to create attributes.
    status = aclshmemx_set_attr_uniqueid_args(pe, pe_size,
                                                local_mem_size,
                                                &uid, &attributes);
    // Initialization
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attributes);

    /*
    Your task
    */

    // Deinitialization
    status = shmem_finalize();
    aclrtResetDevice(pe);
    aclFinalize();
    MPI_Finalize();
    return 0;

}
```
# AllGather
The sample project is stored in the `examples/allgather` directory.

In this sample, an AllGather communication operator with lower latency is implemented when the communication volume is small (less than 2 MB per PE). Each PE first pushes the data stored at the local input address to its symmetric memory. After confirming that the task on the remote PE is complete, each PE pulls the data from the symmetric memory of the remote PE to complete the AllGather operation. This sample demonstrates the usage of multiple SHMEM APIs, such as `aclshmemx_mte_put_nbi`, `aclshmemx_signal_op`, and `aclshmemx_mte_get_nbi`, for P2P communication and task synchronization.

## Kernel Function Implementation
```c++

#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"
using namespace AscendC;

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;

template<typename T>
ACLSHMEM_DEVICE void all_gather_small_data(uint64_t fftsAddr, __gm__ T* input, __gm__ T* output, __gm__ T* gva, int elements, int magic)
{
    const int64_t aivNum = GetBlockNum() * 2;
    const int64_t aivIndex = GetBlockIdx();

    const int64_t data_offset = aivNum * SYNC_FLAG_INTERVAL;
    const int64_t flag_offset = aivIndex * SYNC_FLAG_INTERVAL;

    int64_t my_rank = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();

    __gm__ T *input_gm = (__gm__ T *)input;
    __gm__ T *output_gm = (__gm__ T *)output;

    __gm__ T *gva_data_gm = (__gm__ T*)((__gm__ int32_t*)gva + data_offset);
    __gm__ int32_t *gva_sync_gm = (__gm__ int32_t *)gva;

    __ubuf__ T* tmp_buff = (__ubuf__ T*)(64);

    // data move parameters
    const uint32_t ub_size = UB_DMA_MAX_SIZE;
    uint32_t input_offset, output_offset, gva_offset, num_per_core;

    // [AllGather Step 1] local input gm -> symmetric mem.
    num_per_core = elements / aivNum;
    input_offset = aivIndex * num_per_core;
    gva_offset = aivIndex * num_per_core;
    if (aivIndex == aivNum - 1) {
        num_per_core = elements - num_per_core * aivIndex;
    }
    aclshmemx_mte_put_nbi(gva_data_gm + gva_offset, input_gm + input_offset, tmp_buff, ub_size, num_per_core, my_rank, EVENT_ID0);

    const int64_t core_per_rank = aivNum / pe_size;
    const int64_t core_rank_idx = aivIndex % core_per_rank;
    const int64_t x = aivIndex / core_per_rank;

    // Sync Ensure Corresponding Tasks Done.
    aclshmem_quiet();

    aclshmemx_signal_op(gva_sync_gm + flag_offset, magic, ACLSHMEM_SIGNAL_SET, my_rank);
    aclshmem_signal_wait_until((__gm__ int32_t *)aclshmem_ptr(gva_sync_gm, x) + flag_offset, ACLSHMEM_CMP_EQ, magic);

    // [AllGather Step 2] symmetric mem -> local output.
    num_per_core = elements / core_per_rank;
    output_offset = x * elements + core_rank_idx * num_per_core;
    gva_offset = core_rank_idx * num_per_core;
    if (core_rank_idx == core_per_rank - 1) {
        num_per_core = elements - num_per_core * core_rank_idx;
    }
    aclshmemx_mte_get_nbi(output_gm + output_offset, gva_data_gm + gva_offset, tmp_buff, ub_size, num_per_core, x, EVENT_ID0);
}
```
