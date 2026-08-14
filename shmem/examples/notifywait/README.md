# NotifyWait机制使用说明

> **暂不支持 Ascend950**：当前暂不支持在 Ascend950 平台配套编译运行。

## 环境要求和准备

SDMA功能需要CANN 9.0.0-beta.2及以上版本支持。请参考[CANN版本说明](../../docs/quickstart.md#431-cann-版本说明)下载并安装对应版本的toolkit包；使能SDMA时，还需要安装与toolkit版本和设备类型匹配的ops包。

## example执行说明

1. 在`shmem/`目录编译软件包并安装：

    ```bash
    bash scripts/build.sh -package
    ./install/*/SHMEM_1.0.0_linux-*.run --install
    ```

2. 在`shmem/`目录下编译examples：

    ```bash
    bash scripts/build.sh -examples
    ```

3. 在`shmem/examples/notifywait`目录执行demo:

    ```bash
    bash run.sh -pes ${PES} -type ${TYPES}
    ```

    **参数说明**：
    - PES：指定用于运行的设备（NPU）数量，限定单台机器内。
    - TYPES：指定传输数据类型，当前支持：int，uint8，int64，fp32。

## NotifyWait用法说明

### 用法示例

![notifywait](../../docs/images/notifywait.png)

```c++
// 步骤1：
stream1上的kernel1：调用sdma接口搬运数据+aclshmemx_sdma_notify_record
// 步骤2：
host:aclrtWaitAndResetNotify(notify_id, stream2, 0)
// 步骤3：
stream2上的kernel2：使用sdma搬运好的数据
```

### 用法说明

aclshmemx_sdma_notify_record中会下发record类型的sqe到stars，后续在host侧等待notify记录完成，再继续执行后续kernel。相比aclshmemx_sdma_quiet使用AIV轮询flag的方式，可及时释放AIV资源。
