# 单 INC Fusion Kernel 唯一保留路径（2026-08-11）

## 选择结果

当前保留的自定义单 INC 最优路径只有一条：

```text
qualified source snapshot: 780334b
logical delivery commits: 227879d + 6bfb81b + 57e4475 + 9343813
mode: fused_inc
protocol ABI: 13
data path: worker -> INC Dispatch -> FFN -> INC Combine -> worker
schedule: token-wave，Dispatch/Combine 独立 AIV cohort 并发
```

选择规则是先通过算子契约内的全量 golden、协议检查和稳定性 gate，再比较性能。未通过
这些 gate 的后续未提交性能实验已经从工作树移除，不进入本报告，也不作为可选版本保留。

这里的“正确性”严格限定为 fusion 算子契约：路由、fanout、两次 GMM、SwiGLU、加权归约与
回传结果对 CPU golden 的 BF16 容差检查。它不等同于真实 vLLM 整网与 native 路径逐 token
完全等价；后者仍是独立的集成 gate，不能用本报告冒充已通过。

## 重新资格化结果

环境为 `nb-borrow`、Ascend 910B2C 单 HCCS 平面、CANN 8.5.1 构建。每个 case warmup 3 次、
measure 10 次；时延取同一 iteration 所有参与 PE 的 makespan。

| workers | tokens | hidden | intermediate | top-k | mean | CV | D/C 窗口加速 | 交叠实现度 | golden |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W2 | 32 | 256 | 512 | 2 | 342.798 us | 1.804% | 1.5962x | 97.82% | PASS，全量 |
| W4 | 32 | 256 | 512 | 2 | 368.540 us | 1.423% | 1.6058x | 97.19% | PASS，全量 |
| W2 | 8 | 16384 | 128 | 2 | 760.702 us | 1.197% | 1.4149x | 98.30% | PASS，全量、多 packet |
| W4 | 8 | 16384 | 128 | 2 | 799.092 us | 0.798% | 1.4741x | 98.64% | PASS，全量、多 packet |

四个 case 的所有 worker 与 INC 均返回 `pass=1`。中等 shape 最大相对误差为 0.480%，
32 KiB/row 多 packet shape 最大相对误差为 0.460%，均通过既有 BF16 golden gate。
复测前后 `npu-smi info` 均显示 16 张 NPU 无其他计算进程或残留进程。

机读结果见 [results.csv](results.csv)。原始日志位于隔离运行目录：

```text
/export/home/yinjinrun.montyyin/.borrow/inc-vllm-0191/logs/abi13_requalify_20260811/
```

其 `results.csv` SHA-256 为
`1973295d3cf9c6e3faef4c300d516f51dd5091205e135b7f2945717c64a5b209`；全部 PE 日志的
有序 SHA-256 清单再哈希为
`817962dadf32860e08f0379164a5a7385e24a2cbd5096d0051038c81838b963a`。

## 验证范围

本次从原始源码快照 `780334b` 重新构建，并通过：

- plan、benchmark contract、prepared API 单测；
- grouped GEMM 设备 golden，最大绝对误差 `0.0053215`；
- 融合 FFN 设备 golden，输出最大绝对误差 `0.00189412`；
- scalar/parallel route-pack 对 CPU compiler 的 int32/int64 与 inactive-wave 检查；
- 上表 W2/W4 全量 E2E golden、并发 D/C 与多 packet 回归。

当前分支没有改变上述资格化源码，只把交付内容按开发逻辑拆分为单 INC、fusion 核心、
vLLM 接入与构建接线；不需要额外选择运行时协议参数，也不包含本轮已经剔除的
未提交优化分支。
