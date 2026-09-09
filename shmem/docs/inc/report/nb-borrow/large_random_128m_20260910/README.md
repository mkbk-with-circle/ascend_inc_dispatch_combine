# 128 MiB 随机 token 路由压力测试（nb，2026-09-10）

> 尺度勘误：本批D的128 MiB是输入上限。ragged在skew=0时也将W4源量设为
> 128/96/64/32 MiB，非每个worker等量128 MiB。C W4的rows设置使上行总量
> 约1 GiB（seed17为1027.3125 MiB），平均每worker约256 MiB，亦非128 MiB。
> 本页保留压力测试证据，但不能直接与等量128 MiB的正式带宽表比较。

代码：本地 `193a3c2`。所有 Dispatch 路由共用 `RelaySourceRuns`，所有 Combine
贡献数共用 `ReduceContributorTaskRange`。每组为 3 warmup + 10 measure，所有
rank 的数值、协议状态、completion/ACK 和 guard 均检查。

Dispatch 使用 `ragged` 路由：每 token 的 destination/expert assignment 由 seed
生成，可形成不同数量和不同位置的唯一目标 GPU。每个 worker 输入为128 MiB BF16
hidden；带宽为 fan-out 下行字节 / 完整 Dispatch 时间。三个 seed 为 `17`、
`20260910`、`987654321`；每个 seed 运行 `(token skew, READY skew)`：
`(0%, 0µs)`、`(50%, 500µs)`、`(100%, 1000µs)`。

Combine 使用 `mixed_k`：每 token 对每个目标 GPU 的选择由 route_seed 伪随机决定，
因此 K 可在0到worker数之间变化；partial 还使用正负、不同数量级的FP32数据并由
FP64 oracle 校验。每个 worker 输入为128 MiB FP32 partial；带宽为归约上行字节 /
完整 Combine 时间。seed 为 `17`、`20260910`、`987654321`、`0x5EEDBEEF`。

| 算子 | 规模 | case数 | min范围 GB/s | mean范围 GB/s | CV范围 |
|---|---:|---:|---:|---:|---:|
| Dispatch ragged | W2 | 9 | 9.150–12.350 | 9.348–12.437 | 0.365–1.442% |
| Dispatch ragged | W4 | 9 | 9.442–16.073 | 9.594–16.210 | 0.490–1.260% |
| Combine mixed-K | W2 | 4 | 34.266–34.600 | 34.405–34.793 | 0.247–0.362% |
| Combine mixed-K | W4 | 4 | 69.058–69.214 | 69.456–69.551 | 0.255–0.382% |

共26个独立case，260个测量 wave和78个warmup wave，全部正确。最慢的 Dispatch
为W2、seed=20260910、100% token skew及1000µs READY skew：min 9.150、
mean 9.348 GB/s、CV 1.442%。这是刻意构造的最坏时序，不适用对称 raw gate。

普通对称128 MiB结果、绝对gate和统一前后A/B在
`dispatch_unified_20260909` 与 `combine_unified_candidate` 中单独记录。随机压力
不能替代对称性能验收；反过来，对称带宽也不能说明随机路由稳定性。

复现：

```bash
python3 examples/inc/dispatch_combine/single_inc/pull_combine/tests/pull_v2_large_random_stress.py \
  --build-dir /tmp/inc-k4-build-20260909 \
  --output /tmp/inc-large-random-128m-new
```

Dispatch 使用NPU0–4，Combine 使用NPU8–12，两个测试平面各自要求整张HCCS平面
空闲。当前 runner 串行运行每个case；输出目录保存逐rank日志与`summary.json`，
不提交原始日志。
