# nb 源分区随机路由与流水优化（进行中）

本轮不改变源分区、READY/Notice、INC GET、Journal、fan-out、归约和 ACK/Completion 协议。以下是阶段记录，不表示无回退验收完成。没有修改带宽分子或扩大普通 AIV 缓冲预算。

原源分区构建的 19 个 case 已全部完成且通过，见 [随机压力测试基准汇总](source_partition_random_baseline.csv)：18 个方向 case 各 3 次预热+10 次测量，另一个非对称真实 D→C case 10 次测量，共 244 个含预热的完整检查轮次。不要将这些结果归给三缓冲候选。

候选正在平面 A 以 `random-combine` suite 回放相同种子和卡组，输出 `/tmp/inc-partition-combine-ab-planeA-20260910/summary.csv`；平面 B 的 `combine-regression` 继续验证规则/随机路由与边界输入。两组输出未齐全前，不宣布所有 case 均提升。

## 测试隔离与口径

- 平面 A（0–7）运行原源分区构建 `/tmp/inc-k4-build-20260909` 的 19-case 压测；平面 B（8–15）运行独立候选构建 `/tmp/inc-partition-opt-20260910`。各 case 启动前检查整平面无进程，不修改正在运行的动态库。
- W2 使用平面内首两张 worker 和第三张 INC；W4 使用首四张 worker 和第五张 INC。
- 每源输入 8192×8192 BF16 = 128 MiB，64 experts；随机 expert 路由在同 GPU 上去重。Combine 的 FP32 partial 输入量依据实际唯一贡献 GPU 数，不固定为每 worker 128 MiB。
- Dispatch = 实际 fan-out hidden 字节 / 完整 Dispatch 时间；Combine = 实际 partial ingress 字节 / 完整 Combine 时间。Combine 使用真实 D Journal，但 D 和端侧 partial 准备在 Combine 计时之外。
- expert top-k 相同不代表唯一目标 GPU 数相同；不能把固定 GPU fan-out 的数值视作随机 expert 路由的同工作量对照。

## 已进行的优化

1. 导出已有 D/C 设备阶段时间戳，读取与日志输出在计时区间之外。`joined_payload` 是本 origin AIV 组完成时间，不伪称精确最后一条 GET 指令时间。
2. 同 token 的下一分块 partial GET 提前至当前结果 PUT 前。仅在已验证且 READY 的贡献源、当前 token、当前 AIV 任务范围内预取；异常退出排空未消费的 GET event。
3. Combine 从两输入+两输出的 4×6 KiB 改为两输入+一输出的 3×8 KiB，仍为 24 KiB。复用输出前等待 PUT 完成；任意 H 保留精确尾部处理。

仅提前 GET 的 W2 随机短测从 34.1668 到 34.2073 GB/s，变化不足以证明有效提升。三缓冲候选 W2 短测为 34.6129 GB/s（同平面对照约 +1.31%）；W4/K4 随机短测为 67.9687 GB/s。短测均 1 warmup + 3 measure，不能替代正式回归，也不能将两个平面的数据作严格 A/B。

已通过的候选功能短测：W4 `[31,0,7,19]` / H33 / K4，W2 64 tokens/source / H8192 / K2，均真实 D→C，3 轮结果检查通过。更长规则/随机路由、边界 H、空源与延迟 READY 正在回归；结果以各 case 的 PASS 与 CSV 为准。

## 可追溯产物

运行所在机器的原始记录：

```text
/tmp/inc-source-partition-random-stress-20260910/summary.csv
/tmp/inc-partition-profile-20260910/                 # 同平面原始 Combine
/tmp/inc-partition-prefetch-20260910/                # 仅提前 GET
/tmp/inc-partition-threebuf-20260910/                # 三缓冲短测
/tmp/inc-partition-dispatch-profile-20260910/         # D 阶段时间戳
/tmp/inc-partition-combine-regression-20260910/       # 正式回归增量 CSV
```

候选 Combine 动态库 SHA256：`819f414361ff91995248655b83d71f39947340589d9f084daf2dfb371511c38c`。

可复跑脚本：`shmem/examples/inc/dispatch_combine/single_inc/pull_combine/tests/pull_v2_source_partitions.py`，新增 `random-stress` 和 `combine-regression` suite。每次使用新输出目录；失败即停并保留全部 rank 日志。
