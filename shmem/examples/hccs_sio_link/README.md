# HCCS SIO Link

HCCS/SIO 链路测试工具，用于验证 NPU 之间 SIO / HCCS 链路的正确性。

> **依赖**：
>
> - CANN 版本 >= 9.1.0
> - Ascend HDK 版本 >= 26.0.0
> - LingQu Computing Network 版本 >= 1.5.3。下载地址：[LingQu Computing Network](https://support.huawei.com/enterprise/zh/ascend-computing/lingqu-computing-network-pid-258003841/software)

## 链路说明

对于 A3 芯片，一个 NPU 包含两个 Die，两 Die 之间通过 SIO 链路互连。本方案在 SIO 链路基础上新增 HCCS 链路，实现 SIO/HCCS 双路并行传输，从而提高 Die 间数据传输速度。

- **SIO**：Die 间原有互连链路。shmem 初始化后默认获得的 VA 访问对端 Die 时即走 SIO 链路。

- **HCCS**：新增的 Die 间互连链路。通过 [`aclrtMemMapSelectedLink`](https://gitcode.com/cann/runtime/blob/master/docs/zh/api_ref/11-04_virtual_memory_management.md#aclrtmemmapselectedlink) 将新 VA 映射到与原 SIO VA 相同的 PA，并使用 `ACL_RT_MEM_LINK_IDX_1` 选中 HCCS 链路，此时通过新 VA 访问即走 HCCS 链路。

- **SIO + HCCS 双路并行**：同时使用 SIO 和 HCCS 两条链路传输数据，充分利用双链路带宽提升传输性能。

## 核心函数说明

### `setup_hccs_mapping`

建立 HCCS 链路映射，为当前 PE 创建一条通过 HCCS 链路访问对端 PE 堆内存的虚拟地址通道。该函数执行以下四个步骤：

1. **获取本地堆基址**：调用 `aclshmemx_get_heap_base()` 获取当前 PE 的对称堆基址。
2. **转换为对端地址**：调用 `aclshmem_ptr()` 将本地堆基址转换为对端 PE 的可访问虚拟地址（`peer_heap_base`），该地址走 SIO 链路。
3. **预留虚拟地址空间**：调用 `aclrtReserveMemAddress()` 在当前 PE 的虚拟地址空间中预留一段未映射的 VA 区域（`hccs_ptr`）。
4. **映射到 HCCS 链路**：调用 [`aclrtMemMapSelectedLink`](https://gitcode.com/cann/runtime/blob/master/docs/zh/api_ref/11-04_virtual_memory_management.md#aclrtmemmapselectedlink) 将预留的 VA 映射到与 `peer_heap_base` 所对应的物理地址（PA），并指定 `ACL_RT_MEM_LINK_IDX_1` 选中 HCCS 链路。此时通过 `hccs_ptr` 访问即走 HCCS 链路，而通过原 SIO 路径的 VA 访问仍走 SIO 链路——两条 VA 指向同一 PA，但走不同物理链路。

> **关键原理**：SIO 和 HCCS 两条链路共享同一物理地址（PA），但通过不同的虚拟地址（VA）和链路索引实现分流，从而支持双路并行传输。

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `peer` | `int` | 对端 PE 编号，即需要建立 HCCS 映射的目标 PE |
| `local_mem_size` | `uint64_t` | 与 `aclshmemx_init_attr` 初始化时设定的对称堆大小相同（字节），实际映射区域大小为 `local_mem_size + ACLSHMEM_EXTRA_SIZE` |
| `hccs_ptr` | `void **` | 输出参数，成功时返回 HCCS 链路映射的虚拟地址基址；失败时值取决于失败发生的步骤，若 `aclrtReserveMemAddress` 已成功但后续步骤失败，`*hccs_ptr` 可能包含已预留但未映射的地址，需通过 `teardown_hccs_mapping` 或 `HccsMappingGuard` 释放以防泄漏 |

#### 返回值

- `true`：HCCS 映射建立成功，`*hccs_ptr` 为有效的 HCCS 虚拟地址基址。
- `false`：映射过程中任一步骤失败，函数会打印错误信息并返回。

### `teardown_hccs_mapping`

与 `setup_hccs_mapping` 配对的清理函数，用于释放 HCCS 映射资源：

1. 调用 `aclrtUnmapMem()` 解除 VA 到 PA 的映射关系。
2. 调用 `aclrtReleaseMemAddress()` 释放预留的虚拟地址空间。

> **注意**：在示例代码中，`HccsMappingGuard` 结构体利用 RAII 机制在析构时自动调用 `teardown_hccs_mapping`，确保映射资源不会泄漏。

## 编译

本示例需要通过 `-examples` 编译选项启用，CMake 会自动检测当前 CANN 版本是否支持 `aclrtMemMapSelectedLink` 函数，若支持则自动编译本示例：

> **仅支持 Atlas A3**：本示例依赖 Die 间 SIO/HCCS 链路，仅在 A3（Atlas A3 训练系列/Atlas A3 推理系列）上支持，不支持 A2（Atlas A2 训练系列/Atlas A2 推理系列）与 Ascend950。

```bash
bash scripts/build.sh -examples
```

编译产物：`build/bin/hccs_sio_link`

## 前置条件

- 已按上述方式编译 shmem 项目
- 已设置 `ASCEND_HOME_PATH` 环境变量

> **多实例说明**：`aclshmemx_get_heap_base` 返回当前活跃实例的堆基址。在多实例场景下，需先通过 `aclshmemx_instance_ctx_set_impl` 切换到目标实例，再调用 `aclshmemx_get_heap_base`。

## 用法

本工具通过 `run.sh` 脚本启动，脚本会为每个 PE 启动一个后台进程，各 PE 之间通过 shmem 初始化建立通信。

### 运行命令

```bash
bash run.sh [选项]
```

### 典型用例

```bash
# 默认配置：2 个 PE，SIO+HCCS 全链路测试，4KB 数据，int 类型
bash run.sh

# 指定 4 个 PE，仅测 HCCS 链路
bash run.sh -pes 4 -mode hccs

# 指定 8 个 PE，8MB 数据，fp32 类型，仅测 SIO 链路
bash run.sh -pes 8 -size 8 -type fp32 -mode sio

# SIO + HCCS 混合测试（3/5 数据走 SIO，2/5 数据走 HCCS）
bash run.sh -mode mixed

# 混合 Get 性能测试（PE 0 采集 cycle 数据）
export SHMEM_CYCLE_PROF_PE=0
bash run.sh -mode mixed_get_perf

# 混合 Put 性能测试（PE 1 采集 cycle 数据）
export SHMEM_CYCLE_PROF_PE=1
bash run.sh -mode mixed_put_perf
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-ipport` | `tcp://127.0.0.1:8766` | 通信初始化地址 |
| `-pes` | `2` | 参与测试的 PE 总数（与 NPU 数量一致） |
| `-fpe` | `0` | 首个 PE 编号 |
| `-fnpu` | `0` | 首个 NPU 编号 |
| `-type` | `int` | 测试数据类型：`int` / `int64` / `fp32` |
| `-mode` | `all` | 测试模式（见下表） |
| `-size` | `4` | 每个 PE 的数据大小（KB） |

### 测试模式

| 模式 | 说明 |
|------|------|
| `sio` | SIO 链路正确性测试 |
| `hccs` | HCCS 链路正确性测试 |
| `all` | SIO + HCCS 全链路正确性测试 |
| `mixed` | SIO + HCCS 混合正确性测试（3/5 数据走 SIO，2/5 数据走 HCCS） |
| `mixed_get_perf` | SIO + HCCS 混合 Get 性能测试（3/5 数据走 SIO，2/5 数据走 HCCS） |
| `mixed_put_perf` | SIO + HCCS 混合 Put 性能测试（3/5 数据走 SIO，2/5 数据走 HCCS） |

## 性能测试

`mixed_get_perf` 和 `mixed_put_perf` 模式用于测量 SIO + HCCS 双路并行传输的性能（cycle 数），分别测试 Get（远端读）和 Put（远端写）操作。

### 工作原理

- 数据按 3:5 比例分配到 SIO 和 HCCS 链路（3/5 数据走 SIO，2/5 数据走 HCCS）
- Kernel 内部按 block 维度划分：部分 block 负责 SIO 链路传输，其余 block 负责 HCCS 链路传输
- 使用 `aclshmemx_mte_get_nbi` / `aclshmemx_mte_put_nbi` 进行非阻塞 DMA 传输
- 通过 cycle 计数器采集每次传输的耗时，并利用 shmem profiling 机制（`aclshmemx_get_prof`）输出统计结果

### 环境变量

| 环境变量 | 说明 |
|----------|------|
| `SHMEM_CYCLE_PROF_PE` | 指定进行性能采集的 PE 编号，默认为 `0`。仅该 PE 会执行 cycle 采集逻辑，其余 PE 仅参与 barrier 同步 |

### 运行示例

```bash
# 混合 Get 性能测试，2 个 PE，4KB 数据，int 类型，PE 0 采集性能数据
export SHMEM_CYCLE_PROF_PE=0
bash run.sh -mode mixed_get_perf

# 混合 Put 性能测试，2 个 PE，8KB 数据，fp32 类型，PE 1 采集性能数据
export SHMEM_CYCLE_PROF_PE=1
bash run.sh -pes 2 -size 8 -type fp32 -mode mixed_put_perf
```

### 内部参数

性能测试内部参数定义在 `utils/hccs_sio_link_config.h` 中：

| 参数 | 常量名 | 值 | 说明 |
|------|--------|----|------|
| kernel block 数 | `HCCS_SIO_BLOCK_DIM` | `32` | kernel 启动的 block 总数 |
| UB 缓冲区大小 | `HCCS_SIO_UB_SIZE_KB` | `16` | 每个 block 的 Unified Buffer 大小（KB） |
| SIO 比例分子 | `HCCS_SIO_RATIO_NUM` | `3` | SIO 数据量 = total × NUM / DEN |
| SIO 比例分母 | `HCCS_SIO_RATIO_DEN` | `5` | HCCS 数据量 = total - SIO 数据量 |
| 数据大小下限 | `HCCS_SIO_PERF_MIN_LOG2_BYTES` | `4` | 最小传输数据量的 log2(bytes)，4 = 16B |
| 数据大小上限 | `HCCS_SIO_PERF_MAX_LOG2_BYTES` | `20` | 最大传输数据量的 log2(bytes)，20 = 1MB |
| 数据大小步进 | `HCCS_SIO_PERF_STEP_LOG2` | `1` | log2 步进，1 = 每次翻倍 |
| 预热轮次 | `HCCS_SIO_PERF_WARMUP` | `100` | 预热迭代次数（不计入统计） |
| 测试轮次 | `HCCS_SIO_PERF_LOOP_COUNT` | `1000` | 正式测量迭代次数 |
| 单向/双向模式 | `HCCS_SIO_PERF_IS_UNILATERAL` | `true` | `true`=单向（仅 prof_pe 执行），`false`=双向（所有 PE 执行） |

> **注意**：性能采集受 `ACLSHMEM_CYCLE_PROF_MAX_BLOCK`（最大记录核数）和 `ACLSHMEM_CYCLE_PROF_FRAME_CNT`（最大记录帧数）限制，超出范围的 block 或帧不会被记录。

## 输出示例

正确性测试：

```text
PE 0: [SIO] path verification PASSED for PE 1
PE 1: [SIO] path verification PASSED for PE 0
PE 0: [HCCS] path verification PASSED for PE 1
PE 1: [HCCS] path verification PASSED for PE 0
```

混合测试：

```text
PE 0: [MIXED-SIO] path verification PASSED for PE 1
PE 0: [MIXED-HCCS] path verification PASSED for PE 1
PE 1: [MIXED-SIO] path verification PASSED for PE 0
PE 1: [MIXED-HCCS] path verification PASSED for PE 0
```
