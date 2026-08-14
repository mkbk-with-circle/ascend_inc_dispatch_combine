# 在样例工程使用Ascend C算子调测API

AscendC算子调测API是AscendC提供的调试能力，可进行kernel内部的打印、Tensor内容的查看(Dump)。

关于kernel调测api的详细介绍，可参考[DumpTensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0192.html)和[printf](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0193.html).

## 插入调试代码

1. 修改使用该功能的核函数入口和相关调用代码，增加开启调测功能（`#if defined(ENABLE_ASCENDC_DUMP)`）的编译时代码，具体可参考`examples/sdma/main.cpp`。
2. 在想进行调试的层级，增加调测API调用，如下所示：

   ```diff
   // examples/sdma/main.cpp
   template <typename T>
   __global__ __aicore__ void allgather_sdma_tensor(GM_ADDR gva, int elem_size, GM_ADDR dump, bool is_put)
   {
       AscendC::TPipe pipe;
   +#if defined(ENABLE_ASCENDC_DUMP)
   +    AscendC::InitDump(false, dump, ALL_DUMPSIZE);
   +#endif
       if ASCEND_IS_AIV {
   +        AscendC::printf("my_pe is %d\n", aclshmem_my_pe());
   +        AscendC::GlobalTensor<T> gmT;
   +        gmT.SetGlobalBuffer((__gm__ T*)gva, elem_size);
   +        AscendC::DumpTensor(gmT, elem_size, 16);
           ...
       }
   }
   ```

   注意：`ALL_DUMPSIZE`及`aclCheck`等宏和接口定义位于文件`examples/utils/debug.h`中，其中`ALL_DUMPSIZE`默认为75MB，用户可根据需要进行自定义修改。

## 编译运行

1. 打开工具的编译开关`-enable_ascendc_dump`， 使能AscendC算子调测API编译算子样例。

   - A2/A3 平台:

   ```sh
   bash scripts/build.sh -enable_ascendc_dump -examples
   ```

   - Ascend950 平台:

   ```sh
   bash scripts/build.sh -soc_type Ascend950 -enable_ascendc_dump -examples
   ```

2. 在examples/sdma目录执行demo:

   ```sh
   bash run.sh -pes 2 -type int
   ```
