# nb-borrow 融合算子与 vLLM-Ascend 原生基线（2026-08-09）

## 结论

- 单 INC 融合算子的 W2/W4 六个正式 case 全部通过完整协议检查和 golden 正确性检查。
- 相对同一二进制、同一 shape、同一持久化服务中的严格 token-wave 串行 INC 路径，融合路径真实端到端加速为 **1.1101x–1.7398x**，节省时间 **9.91%–42.52%**。
- vLLM-Ascend **完全原生默认路径**已经在 W2/W4 跑通；没有加载 INC bridge，保留原生 scheduler、attention、Ascend MoE/EP、编译和 ACL Graph。
- 当前 vLLM-Ascend `v0.19.1rc1` 源码中没有名为 MegaMoE 的实现。其最接近的内置融合通信 MoE 是 Fused-MC2，但官方随镜像文档明确限定为 **Atlas A3 + W8A8**；本机是 Atlas A2/910B2C，模型为 BF16，所以该项记为 **N/A（不支持）**，不强行运行。

## 基线定义

| 基线 | 状态 | 定义 |
|---|---:|---|
| serial INC | 已测 | 同一持久 kernel/service 内，worker 必须等当前 token wave 的 combine 结果回传后才能发下一 wave；本地 GMM2 完成后才暴露 combine。它去除了 D/C 与跨 wave 的流水交叠，但保留相同 launch/setup，因而是偏保守的融合收益基线。 |
| fused INC | 已测 | 单 INC，Dispatch/Combine 各占一半 AIV cohort；四个 token wave 可任意时序并发流水。 |
| vLLM-Ascend native | 已测 | `Qwen3-30B-A3B` BF16，`enable_expert_parallel=True`，vLLM 默认编译与 ACL Graph，`VLLM_ASCEND_ENABLE_FUSED_MC2=0`，不导入 INC bridge。 |
| serial SHMEM | 未实现 | 当前融合 ABI 明确拒绝该模式，不能把 INC 或诊断路径冒充 SHMEM baseline。 |
| fused SHMEM | 未实现 | 当前融合 ABI 明确拒绝该模式。 |
| MegaMoE / Fused-MC2 | N/A | 镜像无 MegaMoE；Fused-MC2 不支持本机 A2 + BF16 组合。 |

## 算子级 A/B

环境为 CANN 8.5.1、torch-npu 2.9.0；INC 固定物理 NPU0，worker 为 NPU1–2 或 NPU1–4，均在同一 HCCS 平面。每个 case 预热 3 次、计量 10 次，时间为最慢 worker makespan。

| W | T/H/I/K | 严格串行 | 融合 | 真实加速 | 真实省时 | D+C 窗口理论上限 | 融合 CV |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | 32/256/512/2 | 1216.458 us | 699.176 us | **1.7398x** | 42.52% | 1.5126x | 1.732% |
| 2 | 128/2048/768/2 | 2988.762 us | 1735.260 us | **1.7224x** | 41.94% | 1.7930x | 1.512% |
| 2 | 512/2048/768/4 | 11491.150 us | 9248.380 us | **1.2425x** | 19.52% | 1.9382x | 1.242% |
| 4 | 32/256/512/2 | 1344.496 us | 781.152 us | **1.7212x** | 41.90% | 1.5643x | 3.159% |
| 4 | 128/2048/768/2 | 3711.650 us | 2338.832 us | **1.5870x** | 36.99% | 1.8551x | 1.182% |
| 4 | 512/2048/768/8 | 39021.550 us | 35152.800 us | **1.1101x** | 9.91% | 1.9644x | 1.496% |

这里的“D+C 窗口理论上限”严格表示 `(Td + Tc) / max(Td, Tc)`，只描述 INC 上 Dispatch 与 Combine 两段通信的理想交叠，不是整个 `D→GMM1→GMM2→C` pipeline 的理论上限。真实端到端加速还包括通信与计算、不同 token wave 之间的流水，所以小 case 可以高于该 D+C 子窗口比值；大 token/top-k case 中 FFN 占比增加，端到端收益反而下降。rank 数量本身不会单调决定加速比。

## vLLM-Ascend 完全原生路径

模型是本地 `Qwen3-30B-A3B`：hidden 2048、MoE intermediate 768、128 experts、top-k 8。物理卡为同平面的 NPU1–2 / NPU1–4。每档预热 3 次、计量 10 次。`output_len=1` 是离线 batch 的 TTFT proxy（完整 prefill + 第一个 token）；decode 行是完整 prefill+decode 的 batch 完成时间。

| W | 场景 | 平均延迟 | 中位延迟 | CV | output tok/s |
|---:|---|---:|---:|---:|---:|
| 2 | prefill 128, B1, out1 | 42.821 ms | 42.404 ms | 3.008% | 23.353 |
| 2 | prefill 512, B1, out1 | 70.868 ms | 70.666 ms | 0.980% | 14.111 |
| 2 | input32, decode32, B8 | 793.960 ms | 805.409 ms | 3.841% | **322.434** |
| 4 | prefill 128, B1, out1 | 35.369 ms | 35.064 ms | 3.989% | 28.273 |
| 4 | prefill 512, B1, out1 | 72.151 ms | 71.460 ms | 2.427% | 13.860 |
| 4 | input32, decode32, B8 | 662.396 ms | 661.868 ms | 1.458% | **386.476** |

W4 对小 prefill128 和 B8 decode 有收益，但 prefill512 B1 与 W2 接近；这是原生全模型路径中额外 EP/TP 通信与单 batch 并行度不足共同造成的合理结果，不能用单层融合算子的微秒值直接相除。

## 可复现性与原始数据

- 运行前均要求 `npu-smi info` 显示 16/16 NPU 无进程。
- 宿主上已有其他人的长期容器，未停止、未修改。为避免默认 HCCL `60000–60031` 端口冲突，本任务仅在自身进程环境中使用 W2 `HCCL_IF_BASE_PORT=62000`、W4 `62100`，未修改 sysctl。
- Docker 镜像：`quay.io/ascend/vllm-ascend:v0.19.1rc1`；源码只读挂载，模型只读挂载，构建/日志单独落在 `.borrow/inc-vllm-0191`。
- 正式原始日志：`/export/home/yinjinrun.montyyin/.borrow/inc-vllm-0191/logs/baseline_formal_20260809/`。
- 首轮 eager、3-measure、HCCL 失败尝试均保留在不同子目录，没有覆盖正式结果。
