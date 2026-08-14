# Using Ascend C Operator Debugging APIs in a Sample Project

Ascend C operator debugging APIs are the debugging capabilities of Ascend C. They can be used to print internal kernel information and view tensor content (Dump).

For details about kernel debugging APIs, see [DumpTensor](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0192.html) and [printf](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0193.html).

## Inserting Debugging Code
1. Modify the kernel function entry and its related calling code, and add the build-time code for enabling the debugging function (`#if defined(ENABLE_ASCENDC_DUMP)`). For details, see `examples/allgather_matmul/main.cpp`.
2. Add the debugging API call at the desired level. For example:

   ```diff
   // examples/allgather_matmul/main.cpp
   #if defined(ENABLE_ASCENDC_DUMP)
   __global__ __aicore__
   void ShmemAllGatherMatmul(
      uint64_t fftsAddr,
      GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmSymmetric,
      uint32_t m, uint32_t n, uint32_t k, GM_ADDR dump)
   {
      AscendC::InitDump(false, dump, ALL_DUMPSIZE);
   #else
   __global__ __aicore__
   void ShmemAllGatherMatmul(
      uint64_t fftsAddr,
      GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmSymmetric,
      uint32_t m, uint32_t n, uint32_t k)
   {
   #endif
      // Set FFTS address
      AscendC::SetSyncBaseAddr(fftsAddr);

   +  AscendC::printf("fftsAddr is %d\n", fftsAddr);
   +  AscendC::GlobalTensor<ElementB> gmT;
   +  gmT.SetGlobalBuffer((__gm__ ElementB*)gmB, k * n);
   +  AscendC::DumpTensor(gmT, n, 16);
      ...
   }
   ```
   Note: The macros and APIs such as `ALL_DUMPSIZE` and `aclCheck` are defined in the `examples\utils\debug.h` file. The default value of `ALL_DUMPSIZE` is 75 MB. You can modify it as required.

## Build and Run

1. Enable the tool's build switch `-enable_ascendc_dump` to enable the Ascend C operator debugging APIs for building the operator sample.

   ```sh
   bash scripts/build.sh -enable_ascendc_dump -examples
   ```
2. Run the demo in the `examples/allgather_matmul` directory:

   ```sh
   bash scripts/run.sh 6,7
   ```
