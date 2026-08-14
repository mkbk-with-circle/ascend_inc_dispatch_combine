# rdma_perftest

## 示例概述

`rdma_perftest` 是用于**测试 shmem RDMA 低阶接口性能**的参数化测试示例，平行于同目录下的 `mte_perftest`（针对 MTE 引擎）和 `udma_perftest`（针对 UDMA 引擎）。该示例通过 [SHMEMI_PROF_START/END](../../../src/device/utils/prof/shmemi_prof.h) 宏采集性能数据，覆盖 `aclshmemx_roce_put_nbi` / `aclshmemx_roce_get_nbi` 两个 RDMA 低阶接口在不同数据量下的传输带宽。**该脚本测试结果仅做参考，性能以实际场景为准**。

## 测试目的

针对以下 RDMA 数据传输操作的性能：

1. **单向 Put** (`put`)：仅 PE0 调用 RDMA put 接口，将数据传输到对端 PE。
2. **双向 Put** (`bi_put`)：两个 PE 同时调用 put，互相传输数据。
3. **单向 Get** (`get`)：仅 PE0 调用 RDMA get 接口，从对端 PE 拉取数据。
4. **双向 Get** (`bi_get`)：两个 PE 同时调用 get，互相拉取数据。

## 与 `mte_perftest`、`udma_perftest` 的差异

| 维度 | `mte_perftest` (MTE) | `udma_perftest` (UDMA) | `rdma_perftest` (RDMA) |
|------|----------------------|----------------------|----------------------|
| 引擎 | 默认 MTE | 显式 `ACLSHMEM_DATA_OP_UDMA` | RDMA 引擎 |
| 多核并发 | 同 peer 多核 (默认 32 核切分数据) | **强制单核** (`block_dim=1`)：UDMA 不允许同 peer 多核并发 | **强制单核** (`block_dim=1`)：RDMA 不允许同 peer 多核并发 |
| `-b/--block-size`、`--block-range` | 控制核数 | 入参兼容，但**强制 1**，输入其他值会打印 WARN 后忽略 | 入参兼容，但**强制 1**，输入其他值会打印 WARN 后忽略 |
| UB 缓冲 | MTE 必需，影响传输 | UDMA 必须，用于低阶接口 UB 入参 | RDMA 必须，大小至少为 192B，默认为 192B，自动 64B 对齐 |
| 测试模式 | put / bi_put / get / bi_get | put / bi_put / get / bi_get / **put_signal** | put / bi_put / get / bi_get |
| SOC 限制 | 通用 | **仅 Ascend950**：非 950 上 device kernel 内置 abort | **Ascend950（需指定 `XSCALE` 或 `HNS_1825` 后端）或 A2/A3* |
| CSV 文件名 | `<test>_<dtype>_<pe>.csv` | `udma_<test>_<dtype>_<pe>.csv` | `rdma_<test>_<dtype>_<pe>.csv` |

## 环境要求

同[rdma_demo](../../rdma_demo/README.md)中的环境要求。

## 编译说明

RDMA 功能需要在编译时启用 `-enable_rdma` 参数，并根据 SOC 类型配置后端。RDMA 编译参数（A2/A3，以及 Ascend950 的 `XSCALE` / `HNS_1825` 后端）详见 [编译与构建 - RDMA 参数使用说明](../../../docs/compilation_build_guide.md#rdma参数使用说明)。

## 使用方法

### 基本用法

> 注：Ascend950 平台需设置 `IBV_EXTEND_DRIVERS` 环境变量，参见[环境变量说明](../../rdma_demo/README.md#ibv_extend_drivers-环境变量)。

```bash
cd examples/shmem_perftest/rdma_perftest/
bash run.sh [选项]
```

### 命令行参数

| 参数 | 缩写 | 描述 | 默认值 |
|------|------|------|--------|
| `--test-type <type>` | `-t <type>` | 测试类型 (put / bi_put / get / bi_get / all) | `put` |
| `--datatype <type>` | `-d <type>` | 数据类型 (float / int8 / int16 / int32 / int64 / uint8 / uint16 / uint32 / uint64 / char / all) | `float` |
| `--block-size <size>` | `-b <size>` | RDMA 目前强制为 1，其他值打 WARN 后忽略 | 1 |
| `--block-range <min> <max>` | - | 同上 | 1 1 |
| `--exponent <exponent>` | `-e <exponent>` | 数据量幂数 (2^exponent 字节) | - |
| `--exponent-range <min> <max>` | - | 数据量幂数范围 | 3 17 |
| `--loop-count <count>` | - | 循环次数 | 1000 |
| `--ub-size <size>` | - | UB size (B)；自动 64B 对齐；至少 192B；XSCALE 预热至少需要 `64 + 128 * 100 = 12864B`，聚合组还需满足 `64 + 128 * batch` 字节 | 192 |
| `--batch <count>` | - | 带宽路径中单 QP 在两次 quiet 之间连续提交的 NBI 个数；XSCALE 要求 `1 <= batch < 1024`，且脚本要求不大于 `loop-count`；`0` 或越界值在 XSCALE 下自动改为 100 | 0 |
| `--metric <bw\|lat>` | - | 性能指标: `bw`=带宽, `lat`=接口延迟 | `bw` |
| `--sync-id <id>` | - | 显式传给 Put、Get、Quiet 的同步 ID | 0 |
| `-q/--qp <num>` | - | QP 的个数，当前版本仅支持单 QP | 1 |
| `-pes <size>` | - | PE 数量 (目前强制为 2) | 2 |
| `-ipport <ip:port>` | - | 通信地址 | tcp://127.0.0.1:8768 |
| `-gnpus <num>` | - | NPU 数量 | 2 |
| `-fnpu <id>` | - | 首个 NPU ID | 0 |
| `-fpe <id>` | - | 首个 PE ID | 0 |
| `-a/--analyse <mode>` | - | 分析模式 (none / plot / md) | none |

### HBM 与对称内存约束

本示例仅测试 HBM (DEVICE_SIDE) 内存路径，**不支持 D2H / `HOST_SIDE` (DRAM)**。

`main.cpp` 通过 `aclshmem_malloc` 分配输入、输出缓冲区。该示例直接调用 RDMA 引擎专用接口，因此 Put/Get
的本地和远端操作数都必须指向对称内存，且各自的完整传输范围不得越过其所在的内存分配。

默认 1 GB 本地内存；当数据量较大时，程序会自动上调 `local_mem_size`（最多 40 GB）。

### Batch 参数说明

`--batch` 表示带宽 (`--metric bw`) 路径中，单个 QP 在两次 `aclshmemx_roce_quiet` 之间连续提交的 NBI 操作数。perftest 会按 `batch` 对 Put/Get NBI 分组提交，每组结束后调用一次 quiet。

`--metric lat` 路径不按 `batch` 划分主测试窗口，因此 `batch` 主要影响带宽测试。`batch=1` 是合法值，但不会走 XSCALE 聚合组 WQE，而是逐次提交并 quiet。

云脉 XSCALE 网卡在单次传输消息小于 `64KB` 时会开启批量组 WQE 路径，通过 defer/submit action 将同组 NBI 操作聚合提交；消息大小达到或超过 `64KB` 时不走该 XSCALE 聚合路径，但带宽路径仍会按 `batch` 控制 quiet 分组。源码中每组聚合 WQE 需要 `64 + 128 * batch` 字节 UB，且提交时要求 `batch < SQ depth`；当前 SQ depth 为 1024，因此聚合路径的最大合法值是 1023。

`run.sh` 会通过 `IBV_EXTEND_DRIVERS` 中的 `xscale`/`libxscale` 线索，或 `ibv_devinfo` 输出中的 `xscale` 关键字，自动识别当前环境是否为 XSCALE。识别到 XSCALE 后：

1. UB 至少会调整到 warmup 聚合所需的 `64 + 128 * 100 = 12864B`。
2. `--batch 0` 表示自动选择，脚本将其设置为 100。
3. `--batch > loop-count` 或 `--batch >= 1024` 会被视为非法值，脚本打印中文警告并设置为 100。
4. 非数字或负数的 `--batch` 仍属于参数格式错误，会直接退出。

默认 `--loop-count 1000 --ub-size 192 --batch 0` 会自动调整为 `UB size=12864B`、`batch=100`。如果 `loop-count` 小于 100，可执行程序会按实际 `loop-count` 处理最后的有效批次；这不影响脚本的固定回退值。

### 使用示例

```bash
# 单向 PUT，float，幂数 8-20 带宽测试 (默认 metric=bw)
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000

# 双向 GET，int32
./run.sh -t bi_get -d int32 --exponent-range 8 20 --loop-count 1000

# 单向 PUT, 所有 NBI 执行过后确认一次
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 --batch 0

# 单向 PUT, 每 128 个 NBI 确认一次
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 --batch 128

# 单向 PUT 延迟测试
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 --metric lat

# 双向 PUT
./run.sh -t bi_put -d float --exponent-range 8 20 --loop-count 1000
```

## CSV 输出

CSV 格式如图：

```bash
DataSize/B, Npus, Blocks, UBsize/KB, Bandwidth/GB/s, Bandwidth/GiB/s, CoreMaxTime/us
```

`Blocks` 列恒为 1。文件名前缀为 `rdma_<metric>_`：`output/rdma_<metric>_<test_type>_<dtype>_<pe>.csv`。

## 输出示例

```bash
测试类型: bi_get
数据类型: float
幂数范围: 11-20
循环次数: 1000
警告: XSCALE 聚合预热至少需要 UB size 12864B；已将 UB size 从 192 调整为 12864
警告: XSCALE 下 --batch 0 使用自动值；已将 batch size 设置为 100
UB size(B): 12864
Batch size: 100
Sync ID: 0
QP num: 1
Metric: lat
PE_SIZE: 2, GNPU_NUM: 2
FIRST_NPU: 0
[INFO] rdma_perftest start, pe=0, t=bi_get, d=float, exp=11-20, loop=1000, ub=12864B, metric=lat, batch=100, sync_id=0, qp=1
[INFO] rdma_perftest start, pe=1, t=bi_get, d=float, exp=11-20, loop=1000, ub=12864B, metric=lat, batch=100, sync_id=0, qp=1
```

## 已知约束

1. RDMA 头文件注明：concurrent RMA/AMO operations to the same PE are not supported。本 perftest 通过 `block_dim=1` 规避，多核场景留作后续扩展。
2. RDMA 功能需要在编译时启用 `-enable_rdma` 参数，否则编译期会报错；Ascend950 平台还需额外指定 `-soc_type Ascend950` 及 `-rdma_backend XSCALE`（或 `HNS_1825`）参数。
3. **不支持 D2H / `HOST_SIDE` (DRAM)**: RDMA 引擎当前未对 Host 侧 DRAM 提供 RMA 路径，仅测 HBM。
4. 原子操作不在本 perftest 范围。
