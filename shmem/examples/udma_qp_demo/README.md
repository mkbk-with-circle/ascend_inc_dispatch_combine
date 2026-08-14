## 概述

本样例基于 SHMEM 工程，介绍 device kernel 在直连 UDMA 场景下，使用多个 QP 并发完成同一对端 PE 的 Put/Get/PutSignal 数据传输。

样例覆盖三种功能验证模式：

- `put`：当前 PE 使用多个 QP，将本地普通 Device 内存中的数据写入下一个 PE 的 SHMEM 对称内存。
- `get`：当前 PE 使用多个 QP，从下一个 PE 的 SHMEM 对称内存读取数据到本地普通 Device 内存。
- `put_signal`：每个 QP 使用一个 `WRITE_WITH_NOTIFY` WQE 写入自己的数据分片和独立 signal word。

## 版本和平台支持说明

- UDMA 多 QP 样例仅支持 Ascend950 平台（`__NPU_ARCH__ == 3510`），非 Ascend950 平台不支持运行该样例。
- CANN 需提供 Ascend950 UDMA/HCOMM 资源接口，并安装与 Ascend950 匹配的 toolkit 和 ops 包，建议使用 CANN 9.1.0 及以上版本。低版本 CANN 如果缺少 `HcommEndpointCreate`、`HcommMemReg`、`HcommChannelCreate` 等 HCOMM UDMA API，构建阶段会关闭 `ACLSHMEM_UDMA_SUPPORT`，该样例无法运行。
- 多 QP 还要求当前 HCOMM 的 `HcommChannelDesc` 支持 `channelName`。
- 编译前需完成 CANN 环境配置，并使用 `-soc_type Ascend950` 编译。

若 CMake 输出以下信息，说明 Host UDMA Transport 已关闭，不能创建 UDMA Channel 或运行该样例：

```text
Required Hcomm UDMA APIs not found, disabling ACLSHMEM_UDMA_SUPPORT
```

## 样例实现

本样例呈现多个 QP 在同一对端 PE 之间并发执行 UDMA Put/Get/PutSignal 的基本使用流程。

### 测试用例实现

（1）初始化 ACL 和 SHMEM，将数据通路配置为 `ACLSHMEM_DATA_OP_UDMA`，并在 SHMEM 初始化前通过进程级接口 `aclshmemx_set_qp_num` 设置每个对端 PE 使用的 QP 数量。

（2）每个 PE 对应一个进程和一张 NPU，并与环形拓扑中的下一个 PE 通信。Put 和 PutSignal 模式向下一个 PE 写数据，Get 模式从下一个 PE 读数据。

（3）每个 PE 分配一段 SHMEM 对称内存和一段普通 Device 内存，并准备包含 PE 编号和元素位置的测试数据。PutSignal 模式额外为每个 QP 分配一个独占的对称 signal word。

（4）Host 按照 QP 数量启动相同数量的 AIV。各 AIV 分别使用一个 QP，处理互不重叠的数据分片。

（5）kernel 执行完成后，将结果拷回 Host 逐字节校验。Put 和 PutSignal 结果应来自上一个 PE，Get 结果应来自下一个 PE；PutSignal 还校验每个 QP 独立编码的 signal 值。

（6）校验完成后释放 SHMEM、设备和 ACL 相关资源。

### Kernel 实现

（1）每个 AIV 使用自身的任务块编号作为 QP 编号，并根据 AIV 总数均分数据。

（2）Put 模式调用 `aclshmemx_udma_qp_put_nbi`，Get 模式调用 `aclshmemx_udma_qp_get_nbi`，PutSignal 模式调用 `aclshmemx_udma_qp_put_signal_nbi`。每个 AIV 只提交自己负责的数据分片。

（3）提交后调用 `aclshmemx_udma_qp_quiet`，等待当前对端 PE、当前 QP 上的操作完成。PutSignal 的数据和 signal 位于同一个 WQE；不同 QP 之间不提供隐式顺序。

## 编译执行

环境配置请参考[快速上手](../../docs/quickstart.md)。完成环境配置后，在仓库根目录执行：

```bash
# 执行编译
bash scripts/build.sh -examples -soc_type Ascend950

# 运行默认用例，默认使用 2 个 PE、每个对端 PE 使用 2 个 QP，并执行 put 模式
bash examples/udma_qp_demo/run.sh
```

也可以指定运行模式、QP 数和元素个数：

```bash
# 使用 4 个 QP 验证 put
bash examples/udma_qp_demo/run.sh -pes 2 -qp_count 4 -op put -elems 1048576

# 使用 8 个 QP 验证 get
bash examples/udma_qp_demo/run.sh -pes 2 -qp_count 8 -op get -elems 1048576

# 使用 4 个 QP 验证 put_signal
bash examples/udma_qp_demo/run.sh -pes 2 -qp_count 4 -op put_signal -elems 1048576

# 从逻辑 NPU 4 开始绑定 4 个 PE
bash examples/udma_qp_demo/run.sh -pes 4 -qp_count 4 -op put -first_npu 4
```

用例执行成功时，每个 PE 输出：

```text
[PASS] op=put pe=0 elements=1048576
[PASS] op=put pe=1 elements=1048576
```

PutSignal 还会输出 signal 校验结果：

```text
[PASS] op=put_signal pe=0 elements=1048576
[PASS] op=put_signal pe=0 signals=4
[PASS] op=put_signal pe=1 elements=1048576
[PASS] op=put_signal pe=1 signals=4
```

数据校验失败时，程序打印首个错误位置、实际值和期望值。`run.sh` 会等待所有 PE 进程，并在任意进程失败时返回非 0。

### 运行参数

`run.sh` 支持如下参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-pes` | `2` | PE 总数，必须至少为 2；脚本启动相同数量的进程。 |
| `-qp_count` | `2` | 每个对端 PE 创建的 QP 数，范围为 `[1, 32]`，所有 PE 必须一致。 |
| `-op` | `put` | 运行模式，支持 `put`、`get`、`put_signal`。 |
| `-elems` | `1048576` | 每个 PE 搬运的 `uint8_t` 元素数，必须大于 0 且能被 `qp_count` 整除。 |
| `-heap_mb` | `1024` | 每个 PE 的 SHMEM 对称堆大小，单位为 MiB；PutSignal 模式必须同时容纳数据缓冲区和每 QP 一个 `uint64_t` signal word。 |
| `-first_npu` | `0` | PE0 对应的逻辑 NPU 编号；PE `i` 使用 `first_npu + i`。 |
| `-ipport` | 动态本机端口 | 引导初始化地址；未指定时由脚本随机选择本机端口。 |
