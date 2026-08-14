## 概述

本样例基于 SHMEM 工程，介绍 device kernel 侧通过 SDMA 批量数据传输接口在普通 Device 内存和 Host 侧 SHMEM 对称内存之间搬运数据的使用。

样例覆盖 `put`、`get` 和 `all` 三种功能验证模式：

- `put`：验证 `aclshmemx_sdma_put_nbi`，数据方向为本 PE 普通 Device 内存到本 PE 和目标 PE 的 `HOST_SIDE` SHMEM 内存。
- `get`：验证 `aclshmemx_sdma_get_nbi`，数据方向为本 PE 和目标 PE 的 `HOST_SIDE` SHMEM 内存到本 PE 普通 Device 内存。
- `all`：依次执行 `put` 和 `get`。

## 支持的产品型号

- Atlas A3 训练系列产品/Atlas A3 推理系列产品

## 样例实现

本样例呈现的是 SHMEM SDMA 批量 put/get 接口在普通 Device 内存和 Host 侧 SHMEM 对称内存之间搬运数据的使用流程，以下简称 SDMA HOST_SIDE put/get 接口。

### 测试用例实现

（1）初始化 [ACL](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/appdevgapi/aclcppdevg_03_1945.html) 和 [SHMEM](../../README.md)。初始化 SHMEM 时将数据通路配置为 `ACLSHMEM_DATA_OP_SDMA`，使 device kernel 可以调用 SDMA 接口。

（2）根据运行参数分配测试数据内存。每个 PE 都按 `[PE0 segment][PE1 segment]...[PEn segment]` 的方式准备结果区，每段保存一个 PE 的数据。

（3）执行 `put` 模式时，每个 PE 准备一段普通 Device 源数据 `device_src`，值为 `my_pe + 10`，并准备全 0 的 `HOST_SIDE` SHMEM 目标区 `host_dst`。kernel 将本 PE 的 `device_src` 写入本 PE 和其他 PE 的 `host_dst` 中本 PE 对应的 segment，其中 `pe == my_pe` 验证 D2H，`pe != my_pe` 验证 D2rH。

（4）执行 `get` 模式时，每个 PE 准备 `HOST_SIDE` SHMEM 源区 `host_src`，只初始化本 PE 对应的 segment，值为 `my_pe + 10`，并准备全 0 的普通 Device 目标区 `device_dst`。kernel 从本 PE 和各个远端 PE 的 `host_src` 中拉取对应 segment，写入本 PE 的 `device_dst`，其中 `pe == my_pe` 验证 H2D，`pe != my_pe` 验证 rH2D。

（5）kernel 执行前后通过 `aclshmem_barrier_all` 做同步，保证各 PE 的数据初始化、SDMA 访问和结果校验顺序正确。

（6）kernel 执行完成后，将结果拷回 Host 侧校验。每个 PE 的结果区都应包含所有 PE 的 segment，例如 PE0 段为 10、PE1 段为 11、PE2 段为 12。

（7）清理释放 SHMEM 和 ACL 相关资源。

### Kernel 实现

（1）kernel 侧获取本 PE 编号 `my_pe` 和总 PE 数量 `n_pes`。

（2）根据元素个数和数据类型计算每个 PE 的 segment 字节数，并按 block 划分当前 kernel 实例负责搬运的数据范围。

（3）`put` kernel 中，调用 `aclshmemx_sdma_put_nbi` 将本 PE 普通 Device 内存中的数据写入本 PE 或目标 PE 的 `HOST_SIDE` SHMEM 对称地址。接口内部根据 PE 编号完成 Host 侧地址转换，`pe == my_pe` 时验证本地 D2H。

（4）`get` kernel 中，调用 `aclshmemx_sdma_get_nbi` 从本 PE 或目标 PE 的 `HOST_SIDE` SHMEM 对称地址读取数据，写入本 PE 普通 Device 内存。接口内部根据源 PE 编号完成 Host 侧地址转换，`pe == my_pe` 时验证本地 H2D。

（5）`aclshmemx_sdma_put_nbi` 和 `aclshmemx_sdma_get_nbi` 为非阻塞接口，kernel 中调用 `aclshmemx_sdma_quiet` 等待当前 block 提交的 SDMA 任务完成。

## 编译执行

环境配置请参考[快速上手](../../docs/quickstart.md)。完成环境配置后，执行如下命令可进行功能验证。

```bash
# 执行编译
bash scripts/build.sh -examples -cann
cd examples/sdma_d2h_demo
# 运行默认用例，默认执行 all 模式
bash run.sh
```

也可以指定运行模式、数据类型和元素个数：

```bash
bash run.sh -pes 2 -op all -type int -elems 1024
bash run.sh -pes 2 -op put -type uint8 -elems 1048576
bash run.sh -pes 2 -op get -type int64 -elems 262144
bash run.sh -pes 2 -op all -type fp32 -elems 262144
bash run.sh -pes 2 -op put -type uint8 -elems 1048576 -heap_mb 16
```

用例执行完成，打屏信息出现“[SUCCESS] put op pass in pe <my_pe>”，说明当前 PE 的 `put` 结果校验成功；打屏信息出现“[SUCCESS] get op pass in pe <my_pe>”，说明当前 PE 的 `get` 结果校验成功；打屏信息出现“[SUCCESS] sdma_d2h_demo run success in pe <my_pe>”，说明当前 PE 样例执行成功且资源释放正常。

样例还会打印每个 PE 负责 segment 的前 8 个值，例如：

```text
[RESULT] pe 0 put segment 0 expected 10 first 8 values: 10 10 10 10 10 10 10 10
[RESULT] pe 0 put segment 1 expected 11 first 8 values: 11 11 11 11 11 11 11 11
```

如果每个 segment 的前 8 个值均与 `expected` 一致，可以直观看到各 PE 数据已经写入或拉取到正确位置。

### 运行参数

`run.sh` 支持如下参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-pes` | `2` | PE 总数。 |
| `-op` | `all` | 运行模式，支持 `put`、`get`、`all`。 |
| `-type` | `int` | 数据类型，支持 `int`、`uint8`、`int64`、`fp32`。 |
| `-elems` | `1048576` | 每个 PE 的元素个数。 |
| `-heap_mb` | `16` | 每个 PE 的 SHMEM heap 大小，单位 MB。 |

## 约束限制

### PE 数量和启动进程要求

`-pes` 表示 SHMEM 初始化时声明的 PE 总数。当前样例要求每个 PE 对应一个进程和一张 NPU，脚本会按 `-pes` 自动启动同等数量的进程，并按 PE 编号绑定同编号 NPU。`-pes` 不能超过当前环境可用 NPU 数量。

例如需要运行 4 个 PE 时，直接指定 `-pes 4`：

```bash
bash run.sh -pes 4 -op put -type uint8 -elems 1048576
```
