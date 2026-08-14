# SDMA使用说明

> **暂不支持 Ascend950**：当前暂不支持在 Ascend950 平台配套编译运行。

## 环境要求和准备

SDMA put/get接口需要CANN 9.0.0-beta.2及以上版本支持，可用于read/write数据搬运。请参考[CANN版本说明](../../docs/quickstart.md#431-cann-版本说明)下载并安装对应版本的toolkit包；使能SDMA时，还需要安装与toolkit版本和设备类型匹配的ops包。

## 支持设备

SDMA put/get接口当前支持在Atlas 200I A2/A3、Atlas 300T A2/A3等A2/A3平台使用read/write数据搬运能力。Ascend950及以上平台暂不支持通过SDMA put/get接口使用read/write数据搬运能力。

## example使用方式

1. 在`shmem/`目录编译软件包并安装：

    ```bash
    bash scripts/build.sh -package
    ./install/*/SHMEM_1.0.0_linux-*.run --install
    ```

2. 在`shmem/`目录下编译examples：

    ```bash
    bash scripts/build.sh -examples
    ```

3. 在`shmem/examples/sdma`目录执行demo:

    ```bash
    bash run.sh -pes ${PES} -type ${TYPES}
    ```

    **参数说明**：
      - PES：指定用于运行的设备（NPU）数量，限定单台机器内。
      - TYPES：指定传输数据类型，当前支持：int，uint8，int64，fp32。

## SDMA接口使用说明

### aclshmemx_sdma_put_nbi

以指针类型参数接口为例：

```c++
ACLSHMEM_DEVICE void aclshmemx_sdma_put_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                            uint32_t elem_size, int pe, uint32_t sync_id)
```

接口功能：把PE pe上的src地址中的数据传输到dst地址，传输elem_size个元素。

| 参数名       | 含义                                                                 |
|--------------|----------------------------------------------------------------------|
| dst          | 目标卡上目的地址在本卡上的对称地址                                   |
| src          | 本卡上的源地址                                                       |
| buf          | 缓冲区地址                                                           |
| ub_size      | 缓冲区大小                                                           |
| elem_size    | 元素个数                                                             |
| pe           | 目标PE                                                               |
| sync_id      | 同步ID                                                               |

### aclshmemx_sdma_get_nbi

以指针类型参数接口为例：

```c++
ACLSHMEM_DEVICE void aclshmemx_sdma_get_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                            uint32_t elem_size, int pe, uint32_t sync_id)
```

接口功能：把PE pe上的dst地址中的数据传输到src地址，传输elem_size个元素。

| 参数名       | 含义                                                                 |
|--------------|----------------------------------------------------------------------|
| dst          | 目标卡上目的地址在本卡上的对称地址                                   |
| src          | 本卡上的源地址                                                       |
| buf          | 缓冲区地址                                                           |
| ub_size      | 缓冲区大小                                                           |
| elem_size    | 元素个数                                                             |
| pe           | 目标PE                                                               |
| sync_id      | 同步ID                                                               |

## 注意事项

`aclshmemx_sdma_put_nbi`和`aclshmemx_sdma_get_nbi`都是非阻塞接口，调用后立即返回，不等待数据传输完成。用户使用时，可通过以下两种方式确保数据传输完成：

1. 所有调用`aclshmemx_sdma_put/get_nbi`的核，在sdma任务结束后，算子内调用`aclshmemx_sdma_quiet`接口，等待所有SDMA操作完成。
适用场景：算子内后续操作依赖sdma任务完成，例如后续算子需要使用sdma传输好的数据。
2. 所有调用`aclshmemx_sdma_put/get_nbi`的核，在sdma任务结束后，算子内调用`aclshmemx_sdma_notify_record`接口，然后在host侧调用`aclrtWaitAndResetNotify`接口，等待指定的同步ID完成（详细用法可查看[NotifyWait机制使用说明](../notifywait/README.md)）。

适用场景：其它stream上的kernel需要等待sdma任务完成后才能继续执行。
