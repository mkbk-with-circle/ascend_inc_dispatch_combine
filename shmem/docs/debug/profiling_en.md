# Using the Profiler to Collect Profile Data in a Sample Project

This section describes how to use the profiler based on GetSystemCycle to analyze the time consumed by data movement operations. By collecting the number of system clock cycles and converting it into actual time, the tool accurately quantifies the MTE movement performance in different blocks (compute cores) and frames (trace point IDs).

Core functions of the profiler:
Collects statistics on the number of operation executions (count) and total time-consuming cycles (cycles) by block (core) or frame (trace point ID).
Automatically converts the number of cycles to microseconds (μs) and outputs the value accurate to three decimal places.
Displays only the blocks and frames with valid data and filters out empty data to improve readability.

For details about GetSystemCycle, see [GetSystemCycle](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0282.html).

## Inserting Debugging Code
1. Insert `SHMEMI_PROF_START` and `SHMEMI_PROF_END` before and after the position where tracing is required in the kernel code. For details, see `examples/allgather/allgather_kernel.cpp`.
    ```diff
    --- a/examples/allgather/allgather_kernel.cpp
    +++ b/examples/allgather/allgather_kernel.cpp
    @@ -12,6 +12,9 @@
    #include "acl/acl.h"
    #include "shmem.h"

    +// shmem prof
    +#include "utils/prof/shmemi_prof.h"
    +
    #undef inline
    #include "opdev/fp16_t.h"
    #include "opdev/bfloat16.h"
    @@ -72,6 +75,7 @@ ACLSHMEM_DEVICE void all_gather_origin(__gm__ T *input, __gm__ T *output, __gm__
            int64_t times = 0;
            int64_t flag = 0;
            while (copy_total_size >= copy_ub_size) {
    +            SHMEMI_PROF_START(0);
                aclshmemx_mte_put_nbi(gva_data_gm + aivIndex * len_per_core + times * copy_ub_num,
                                    input_gm + aivIndex * len_per_core + times * copy_ub_num, tmp_buff, copy_ub_size,
                                    copy_ub_num, my_rank, EVENT_ID0);
    @@ -80,7 +84,7 @@ ACLSHMEM_DEVICE void all_gather_origin(__gm__ T *input, __gm__ T *output, __gm__
                times += 1;
                flag = times + magic;
                aclshmemx_signal_op(gva_sync_gm + flag_offset, flag, ACLSHMEM_SIGNAL_SET, my_rank);
    -
    +            SHMEMI_PROF_END(0);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    ```

2. Add the function of displaying profile data to the application on the host. For details, see `examples/allgather/main.cpp`.

    ```diff
    --- a/examples/allgather/main.cpp
    +++ b/examples/allgather/main.cpp
    @@ -120,6 +120,8 @@ int test_aclshmem_all_gather(int rank_id, int n_ranks)
            }
            status = aclrtSynchronizeStream(stream);

    +        aclshmemx_show_prof(nullptr, true);
    +
            // Result Check
            T *output_host;
            size_t output_size = n_ranks * trans_size * sizeof(T);
    ```
   Note: Only the collected PE and block are printed. If there is no data, the printing is skipped.

## Build and Run

1. Build the operator sample.

   ```sh
   bash scripts/build.sh -examples
   ```
2. Run the demo in the `examples/allgather` directory:

   ```sh
   export SHMEM_CYCLE_PROF_PE=0  # Set the PE to be collected. Currently, only a specified PE can be collected.
   bash run.sh -ranks 2
   ```
3. Observe the demo output. The following information is displayed:
    ```sh
    ============================================================
    BlockID   FrameID   Cycles         Count          AvgTime(us)
    ------------------------------------------------------------
    0         0         7506966        34050          4.409
    1         0         7485800        34050          4.397
    2         0         8290500        34050          4.870
    3         0         8279083        34050          4.863
    4         0         8255644        34050          4.849
    5         0         8275272        34050          4.861
    6         0         8429026        34050          4.951
    7         0         8404425        34050          4.937
    ============================================================
    ```
    Field description

    | keyword      | description   |
    |--------------|---------------|
    | BlockID      | Index of the device core  |
    | FrameID      | Trace ID        |
    | Cycles       | The number of system cycles      |
    | Count        | Total execution times       |
    | AvgTime(us)  | Average execution time     |
