# SHMEM MegaMoE 示例（Ascend950）

## 样例介绍

本样例展示了一个面向 Ascend950（Arch35）的 SHMEM 原生 MegaMoE 前向流水线。算子将
Token 路由、跨 Rank 数据派发、两次专家矩阵乘、SwiGLU 激活和多专家结果合并放在同一个
Device 侧流水线中，避免在各阶段之间回到 Host 或重复组织通信。

实现中各组件的职责如下：

- ACLSHMEM 提供多 Rank 对称内存、远端地址访问和全局同步能力；
- AscendC AIV 核完成路由掩码生成、Token 量化、SwiGLU 后处理和最终结果合并；
- Catlass AIC 核完成专家网络的两次 MXFP8 GEMM；
- AIC/AIV 之间通过标志位和双缓冲协作，使矩阵计算、后处理与通信可以流水执行。

本目录只包含 Ascend950/Arch35 实现，不包含 Arch22/A3 兼容内核。当前支持以下两种
Token 和权重量化模式：

| 运行模式 | FP8 数据格式 | Scale 格式 |
| --- | --- | --- |
| `arch35_e4m3` | E4M3 | E8M0 |
| `arch35_e5m2` | E5M2 | E8M0 |

## 计算定义

设当前 Rank 上有 `token_count` 个输入 Token，每个 Token 选择 `experts_per_token` 个专家；
系统共有 `rank_count` 个 Rank，每个 Rank 部署 `local_expert_count` 个专家，因此全局专家数为：

```text
global_expert_count = rank_count * local_expert_count
```

全局专家编号与所在 Rank、Rank 内局部专家编号之间的关系为：

```text
expert_rank  = global_expert_id / local_expert_count
local_expert = global_expert_id % local_expert_count
```

对于专家 `e` 收到的输入 `X_e`，专家前向计算为：

```text
U_e = X_e * W1_e
[Gate_e, Up_e] = Split(U_e, 2)
A_e = SiLU(Gate_e) .* Up_e
Z_e = A_e * W2_e
```

第一次投影输出宽度为 `ffn_dim`，等分为 Gate 和 Up 两部分。SwiGLU 将两部分融合为
`ffn_dim / 2` 宽度的激活，再送入第二次投影。原始 Token `t` 的最终输出是它命中的
Top-K 专家结果按路由权重加权求和：

```text
Y_t = Sum(routing_weights[t, k] * Z_expert_ids[t, k], k=0..experts_per_token-1)
```

主要数据及形状如下。每个专家实际参与计算的行数由路由结果决定，且不能超过
`max_received_tokens`。

| 数据 | 类型 | 逻辑形状 | 说明 |
| --- | --- | --- | --- |
| `x` | BF16 | `[token_count, model_dim]` | 当前 Rank 的输入 Token |
| `expert_ids` | INT32 | `[token_count, experts_per_token]` | 每条路由对应的全局专家编号 |
| `routing_weights` | FP32 | `[token_count, experts_per_token]` | Top-K 专家合并权重 |
| `weight1` | FP8 | `[local_expert_count, model_dim, ffn_dim]` | 第一次专家投影权重 |
| `weight1_scale` | E8M0 | 分块 Scale | 第一次投影的 MXFP Scale |
| `weight2` | FP8 | `[local_expert_count, ffn_dim / 2, model_dim]` | 第二次专家投影权重 |
| `weight2_scale` | E8M0 | 分块 Scale | 第二次投影的 MXFP Scale |
| `y` | BF16 | `[token_count, model_dim]` | 加权合并后的本 Rank 输出 |
| `expert_token_counts` | INT32 | `[local_expert_count]` | 本 Rank 各专家实际处理的 Token 数 |

## 整体数据流

算子的主流程位于 `kernel/arch35/shmem_mega_moe_kernel.h` 的 `Process()` 中，数据路径可概括为：

```text
本 Rank BF16 Token
        |
        v
生成目标 Rank 路由掩码 -----> 远端 Rank 的 SHMEM 对称内存
        |
        v
BF16 -> FP8(E4M3/E5M2) + E8M0 Scale
        |
        v
目标 Rank 统计并接收 Token，生成 routing metadata
        |
        v
Catlass GEMM 1: model_dim -> ffn_dim
        |
        v
AIV SwiGLU + 再量化: ffn_dim -> ffn_dim / 2
        |
        v
Catlass GEMM 2: ffn_dim / 2 -> model_dim
        |
        v
根据 routing metadata 写回源 Rank 的 combinedTokens
        |
        v
源 Rank 按 routing_weights 累加 Top-K 结果 -> BF16 输出
```

### 1. 生成跨 Rank 路由掩码

`BuildSendMasks` 遍历本 Rank 的 `expert_ids`，将每个 Token 的每条路由转换为目标 Rank 和
目标局部专家，并生成按目标 Rank 划分的 Token 位图。位图和待发送 Token 数直接写入目标
Rank 的 `inboundMask` 对称内存，使接收方无需由 Host 参与即可知道需要拉取哪些 Token。

### 2. 量化本地 Token

`QuantizeLocalTokens` 将 Token 按 AIV 核划分。每个 AIV 核把 BF16 输入转换为运行模式指定的
E4M3 或 E5M2 数据，同时生成 E8M0 Scale，并写入本 Rank 的 `quantizedTokens` 对称内存。
后续接收 Rank 可以通过 ACLSHMEM 远端地址直接读取这些量化数据。

### 3. 接收 Token 并建立路由元数据

每个局部专家首先执行 `CountReceivedTokens`，统计所有源 Rank 发往该专家的 Token 数；随后
`BuildRoutingMetadataAndDispatch` 按 `max_received_tokens` 上限搬运 Token，并为每条有效路由
记录源 Rank、源 Token 索引、Top-K 槽位和路由权重等信息。这些元数据既决定专家 GEMM 的
输入行，也用于第二次投影后把结果准确写回原始 Token。

### 4. 第一次专家投影与 SwiGLU

`RunFirstProjection` 使用 Catlass 执行第一次 MXFP8 GEMM。AIC 核将结果分块写入 UB，AIV 核
消费这些分块，完成以下操作：

1. 将 `ffn_dim` 宽度的输出切分为 Gate 和 Up；
2. 计算 `SiLU(Gate) * Up`；
3. 将 BF16 激活重新量化为 FP8，并生成第二次 GEMM 所需的 E8M0 Scale；
4. 通过双缓冲标志通知 AIC 对应分块已经就绪。

因此，第一次 GEMM、SwiGLU 和再量化无需把完整中间张量落回全局内存后再启动新的算子。

### 5. 第二次专家投影

`RunSecondProjection` 使用 Catlass 执行第二次 MXFP8 GEMM，将 `ffn_dim / 2` 宽度的激活映射
回 `model_dim`。当前样例在 Host Tiling 中固定使用 `COMBINE_NO_QUANT`：第二次投影输出以
BF16 形式交给 Combine 流程，不对 Combine 结果再做额外量化。

### 6. 远端写回专家结果

第二次投影结束后，AIV 根据路由元数据把每一行专家结果映射回源 Rank、源 Token 和 Top-K
槽位，并写入源 Rank 的 `combinedTokens` 对称内存。写回过程使用乒乓标志同步数据生产和消费，
防止不同轮次复用缓冲区时覆盖尚未被读取的数据。

### 7. 合并 Top-K 结果

所有专家结果返回后，源 Rank 执行 `CombineRoutedTokens`。它读取每个 Token 的全部路由结果，
乘以对应的 `routing_weights` 后累加，并将最终结果转换为 BF16 写入 `y`。

## AIC/AIV 分工与同步

| 处理单元 | 主要工作 |
| --- | --- |
| AIC | 两次 Catlass MXFP8 GEMM、矩阵分块调度 |
| AIV | 路由掩码生成、Token 量化、路由数据搬运、SwiGLU、再量化和最终 Combine |

每个局部专家对应一组计算资源。运行时会查询设备上的 AIC/AIV 数量，并将结果写入 Tiling；
也可以通过环境变量覆盖核数，便于受控测试。AIC 和 AIV 使用以下机制协调：

- 第一次投影采用双缓冲，AIC 生产 GEMM 分块，AIV 消费分块并执行 SwiGLU；
- 输入就绪、激活就绪和 Combine 就绪状态通过独立标志区传递；
- 路由派发按固定行数分波次处理，避免整批 Token 串行等待；
- 多 Rank 阶段边界使用 ACLSHMEM 同步，确保远端写入对接收方可见。

运行前会检查 `local_expert_count` 不大于可用 AIC 数量。如果资源不足，程序会直接报错，
避免多个局部专家错误地竞争同一组 AIC 流水资源。

## 内存布局

### SHMEM 对称内存

每个 Rank 创建相同大小和布局的 ACLSHMEM 对称内存，核心区域包括：

| 区域 | 用途 |
| --- | --- |
| 控制与同步区 | 保存跨 Rank 屏障、标志位等控制信息 |
| `inboundMask` | 接收其他 Rank 写入的 Token 路由位图和计数 |
| `quantizedTokens` | 保存本 Rank 已量化的 Token，供目标 Rank 远端读取 |
| `combinedTokens` | 接收远端专家返回的、尚未按权重累加的 Token 结果 |

主要缓冲区按 512 字节对齐，以满足搬运和矩阵计算要求。样例默认申请 1 GiB 对称内存；
大 Shape 运行前应结合 Rank 数、Token 数和 `max_received_tokens` 检查容量。

### Kernel Workspace

算子 Workspace 由 `memory_layout.h` 统一计算偏移，主要包含：

| 区域 | 用途 |
| --- | --- |
| `receivedTokenData` | 当前 Rank 接收到的专家 Token 数据 |
| `receivedTokenScales` | 接收 Token 对应的 E8M0 Scale |
| `activatedTokenData` | SwiGLU 后、供第二次 GEMM 使用的量化激活 |
| `activatedTokenScales` | 量化激活对应的 E8M0 Scale |
| `receivedTokenCounts` | 各局部专家接收的 Token 数 |
| `routingMetadata` | 结果写回所需的源 Rank、Token 和路由槽位信息 |
| `inputReadyFlags` | 路由输入分波次就绪标志 |
| `activationReadyFlags` | SwiGLU 激活分块就绪标志 |

`memory_layout.h` 还为量化 Combine 模式预留了 `matmulOutput` 和 `rowGroupReadyFlags` 布局；
当前命令行样例固定走 BF16 非量化 Combine 路径，因此不会使用这两个可选区域。

## 目录结构

```text
examples/shmem_mega_moe/
|-- CMakeLists.txt                         # Ascend950 样例编译入口
|-- README.md                              # 本文档
|-- include/
|   |-- shmem_mega_moe_host.h              # Host 资源、参数解析和校验
|   |-- shmem_mega_moe_main.h              # Host 入口声明
|   `-- shmem_mega_moe_main_arch35.h       # Arch35 初始化、Tiling、运行与校验
|-- kernel/
|   |-- arch35/
|   |   |-- shmem_mega_moe_kernel.h        # 算子主流水线
|   |   |-- shmem_mega_moe_tiling.h        # Tiling 数据结构
|   |   |-- shmem_mega_moe_memory_layout.h # 对称内存和 Workspace 布局
|   |   |-- shmem_mega_moe_catlass_pipeline.h
|   |   |                                      # 两次 Catlass GEMM 流水
|   |   |-- shmem_mega_moe_swiglu_epilogue.h # SwiGLU 与激活量化
|   |   `-- shmem_mega_moe_combine_pipeline.h # 结果写回与 Combine 流水
|   |-- common/                             # 公共数学函数
|   `-- dispatch/                           # 路由派发和量化函数
|-- main.cpp                                # 可执行程序入口
`-- scripts/run_shmem_mega_moe_arch35.sh    # 多 Rank 拉起脚本
```

## 编译

### 环境依赖

- Ascend950 环境及与之匹配的 CANN Toolkit；
- 支持 Ascend950 MXFP8 GEMM 的 Catlass 源码；
- CMake 和 C++ 编译工具链。

从仓库根目录执行：

```bash
cmake -S . -B build_arch35 \
  -DSOC_TYPE=Ascend950 \
  -DUSE_EXAMPLES=ON \
  -DCATLASS_ROOT=/path/to/catlass

cmake --build build_arch35 --target shmem_mega_moe -j
```

如果 CMake 无法自动找到 CANN AscendC 头文件，请显式设置：

```bash
export ASCEND_HOME_PATH=/path/to/ascend-toolkit/latest
```

CMake 会检查 SHMEM MegaMoE 的 Host、Tiling、Kernel 头文件及 Catlass 依赖，只有依赖完整时
才注册 `shmem_mega_moe` 目标。

## 运行

### 命令行接口

每个 Rank 启动一个进程，命令格式为：

```bash
shmem_mega_moe <rank_size> <rank> <ip:port> <npu_num> <first_npu> \
  <mode> <token_count> <model_dim> <ffn_dim> \
  <experts_per_token> <local_expert_count> \
  [max_received_tokens] [warmup] [loop]
```

| 参数 | 含义 |
| --- | --- |
| `rank_size` | 参与通信的进程/Rank 总数 |
| `rank` | 当前进程的 Rank 编号，范围为 `[0, rank_size)` |
| `ip:port` | ACLSHMEM 初始化地址，例如 `tcp://127.0.0.1:8766` |
| `npu_num` | 从 `first_npu` 开始参与映射的 NPU 数量 |
| `first_npu` | 第一个逻辑 NPU 编号 |
| `mode` | `arch35_e4m3` 或 `arch35_e5m2` |
| `token_count` | 每个 Rank 的本地 Token 数 |
| `model_dim` | 输入、输出隐藏维度 |
| `ffn_dim` | 第一次投影输出维度，必须为偶数 |
| `experts_per_token` | 每个 Token 选择的专家数，即 Top-K |
| `local_expert_count` | 每个 Rank 部署的专家数 |
| `max_received_tokens` | 每个局部专家的最大接收容量；传 `0` 时自动计算 |
| `warmup` | 不计入统计的预热轮数，默认 `0` |
| `loop` | 计时运行轮数，默认 `1` |

当前进程使用的设备按下式映射：

```text
device_id = rank % npu_num + first_npu
```

### 多 Rank 启动脚本

推荐使用脚本一次拉起所有 Rank：

```bash
cd examples/shmem_mega_moe

BIN=../../build_arch35/bin/shmem_mega_moe \
RANK_SIZE=2 NPU_NUM=2 FIRST_NPU=0 \
MODE=arch35_e5m2 \
TOKEN_COUNT=256 MODEL_DIM=4096 FFN_DIM=1024 \
EXPERTS_PER_TOKEN=6 LOCAL_EXPERT_COUNT=4 \
WARMUP=5 LOOP=20 \
bash scripts/run_shmem_mega_moe_arch35.sh
```

脚本支持的环境变量及默认值如下：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `BIN` | `./shmem_mega_moe` | 可执行文件路径 |
| `RANK_SIZE` | `2` | Rank 数量 |
| `IP_PORT` | `tcp://127.0.0.1:8766` | ACLSHMEM 初始化地址 |
| `NPU_NUM` | `8` | 用于 Rank 映射的 NPU 数量 |
| `FIRST_NPU` | `0` | 起始 NPU 编号 |
| `MODE` | `arch35_e4m3` | FP8 模式 |
| `TOKEN_COUNT` | `4` | 每 Rank Token 数 |
| `MODEL_DIM` | `4096` | 隐藏维度 |
| `FFN_DIM` | `1024` | 第一次投影输出维度 |
| `EXPERTS_PER_TOKEN` | `2` | 每 Token 路由专家数 |
| `LOCAL_EXPERT_COUNT` | `1` | 每 Rank 专家数 |
| `MAX_RECEIVED_TOKENS` | `0` | 每个局部专家接收上限，`0` 表示自动计算 |
| `WARMUP` | `0` | 预热轮数 |
| `LOOP` | `1` | 计时轮数 |
| `AIC_NUM` | 未设置 | 在硬件上限内覆盖 AIC 数量 |
| `AIV_NUM` | 未设置 | 在硬件上限内覆盖 AIV 数量 |

脚本会记录所有 Rank 的进程号并等待它们结束。任一 Rank 失败时，脚本终止仍在运行的其他
Rank，并返回非零状态，避免通信错误被某个成功进程掩盖。

设置 `AIC_NUM` 或 `AIV_NUM` 时，覆盖值不能超过运行时检测到的硬件核数，并且必须保持
`AIC_NUM:AIV_NUM=1:2` 的 MIX Kernel 比例；同时覆盖两项可以避免单项覆盖后比例不匹配。

## 参数约束

运行前 Host 会进行 Shape、资源和容量校验，主要约束如下：

- `mode` 必须为 `arch35_e4m3` 或 `arch35_e5m2`；
- `ffn_dim` 必须能被 2 整除，以便执行 Gate/Up 切分；
- `experts_per_token <= rank_size * local_expert_count`；
- `local_expert_count` 不能超过运行时可用 AIC 数量；
- 输入、权重、Scale、输出和 Workspace 的计算长度不能溢出；
- 所有 Rank 必须使用相同的 Shape、运行模式和 ACLSHMEM 初始化地址；
- `max_received_tokens=0` 时，容量按下式自动设置：

```text
token_count * rank_size * min(experts_per_token, local_expert_count)
```

显式设置 `max_received_tokens` 时，当前数值验证程序要求它不小于上述容量；否则 Host 参数
校验直接失败，不会启动多 Rank Kernel。

## 正确性校验

为了让样例结果可复现，Host 使用确定性数据构造输入：

- 输入 `x` 的 BF16 元素初始化为 `1`；
- 专家编号按 Token、Top-K 槽位和 Rank 确定性生成；
- 所有路由权重初始化为 `1 / experts_per_token`；
- FP8 权重初始化为能精确表示 `1` 的 E4M3/E5M2 编码；
- E8M0 Scale 初始化为表示 `1` 的编码；
- 当前样例中所有输入 Token 均处于激活状态。

每轮运行结束后执行两类校验：

1. **专家 Token 计数校验**：检查每个局部专家实际接收的 Token 数与 Host 路由结果一致；
2. **逐元素数值校验**：CPU 按两次投影、SwiGLU 和路由权重计算解析 Golden，逐个比较 BF16
   输出元素。

在当前全 1 输入和权重下，单条专家路径的理论结果为：

```text
first_projection = model_dim
silu_value       = first_projection / (1 + exp(-first_projection))
expert_output    = silu_value * first_projection * (ffn_dim / 2)
```

E4M3 和 E5M2 使用不同的相对误差阈值：E4M3 为 5%，E5M2 为 15%，绝对误差阈值均为 1。
校验通过时日志会包含：

```text
verify=pass(expert_token,numeric_golden)
```

该校验不依赖固定的 Rank 数，也覆盖非 Tile 对齐的 Token 数。目标环境验收时建议至少覆盖
2、4、8 Rank，两种 FP8 模式以及多组 Token 数。

## 性能指标说明

样例同时输出 Kernel Event 和端到端两种耗时：

| 指标 | 统计范围 |
| --- | --- |
| `kernel_event_avg_ms` | ACL Event 包围的 Device 侧算子执行时间 |
| `e2e_avg_ms` | 从算子启动到 Stream 同步及跨 Rank Host Barrier 完成的总时间 |

`warmup` 轮不计入统计，`loop` 控制正式计时轮数。比较算子本身性能时应使用
`kernel_event_avg_ms`；评估完整调用开销时使用 `e2e_avg_ms`。多 Rank 场景下应同时确认所有
Rank 的校验结果，并以相同 Shape、核数和运行模式进行对比。

## 常见问题

### CMake 没有生成 `shmem_mega_moe` 目标

确认 `SOC_TYPE=Ascend950`、`USE_EXAMPLES=ON`，并检查 `CATLASS_ROOT` 是否指向包含所需
Catlass 头文件的源码目录，同时确保本样例依赖的 Host、Tiling 和 Kernel 头文件完整。

### 找不到 AscendC 头文件

设置正确的 `ASCEND_HOME_PATH`，并确认 CANN Toolkit 版本支持 Ascend950 和样例使用的
AscendC 接口。

### 多 Rank 运行阻塞

确认所有 Rank 均已启动，`rank_size` 和 `IP_PORT` 完全一致，且 Rank 编号不重复。建议使用
随样例提供的启动脚本，任一进程失败时同时检查其他 Rank 的日志，通常最先报错的 Rank 才是
根因。

### 对称内存或 Workspace 不足

减小 Token/维度/Rank 数，或检查 `max_received_tokens` 是否设置过大。若显式设置的接收容量
小于数值验证所需下限，Host 会直接报错。大 Shape 运行前应先根据内存布局估算每个 Rank 的
实际占用。
