# nb-borrow 单 INC 一 pull 多纯 GET 峰值（2026-09-03）

本目录只记录 Pull-Dispatch V2 的**纯 GET 物理腿**标定，不是完整
Dispatch 算子结果。测试使用 nb-borrow 的 HCCS 平面 B（NPU 8--15），
消息大小为每个 worker 128 MiB；所有 case 均逐字节校验通过。

## 拓扑与口径

- W2：NPU 8、9 为 worker，NPU 10 为 INC。
- W4：NPU 8--11 为 worker，NPU 12 为 INC。
- 每个 case：3 次 warmup、10 次迭代。
- `aggregate_total_GBps` 的分子包含 warmup 与正式迭代传输的全部字节，
  时间为同一次 kernel launch 的完整主机观测区间。
- `aggregate_conservative_GBps` 采用相同时间，但分子只计算 10 次正式
  迭代，因此会有意把 warmup 时间计入开销。
- nominal raw：W2 为 2 x 28 = 56 GB/s，W4 为 4 x 28 = 112 GB/s。

## lanes 初扫

| workers | lanes/worker | aggregate total GB/s |
|---:|---:|---:|
| 2 | 1 | 22.729 |
| 2 | 2 | 41.839 |
| 2 | 3 | 41.612 |
| 2 | 4 | **41.842** |
| 2 | 6 | 41.337 |
| 4 | 1 | 44.175 |
| 4 | 2 | **83.682** |
| 4 | 3 | 83.246 |
| 4 | 4 | 83.682 |
| 4 | 6 | 83.040 |

超过约 2 lanes/worker 后已经基本饱和；W2 的 2 与 4 lanes 差异小于
0.01 GB/s，不能据此在生产协议中写死 W2=4。

## 最优配置三次独立复测

| workers | lanes/worker | total min / mean GB/s | CV | 相对 nominal raw | conservative min / mean GB/s |
|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 41.8328 / 41.8381 | 0.00907% | 74.71% | 32.1791 / 32.1832 |
| 4 | 2 | 83.6648 / 83.6712 | 0.00598% | 74.71% | 64.3575 / 64.3625 |

若纯 GET 物理腿另设本机实测 roof 的 92% 诊断线，则：

- W2：41.8381 x 92% = **38.491 GB/s**；
- W4：83.6712 x 92% = **76.978 GB/s**。

正式完整算子 gate 只用于每 worker 128 MiB、worker 输入量与目的端负载均
对称的 workload，固定为 nominal raw 56/112 的 92%，即 W2
**51.52 GB/s**、W4 **103.04 GB/s**。当前 SHMEM GET 单向实现无法达到
这一物理腿口径，但不能因此下调完整 Dispatch gate：完整算子的分子同时
计算实际 GET ingress 与 PUT egress 字节，且两腿能够双向流水重叠。不能
把完整算子通过该 gate 解释为单独 GET 达到 raw 的 92%；完整协议还需另测
GET+PUT relay duplex roof，并报告链路腿和完整 makespan 两种口径。

## 复现

二进制：

```text
/tmp/shmem-pull-dispatch-v2-build/bin/inc_dc_pull_combine_transport_probe
```

参数：

```text
<pes> <pe> <ipport> <gpus> <first_npu> <bytes>
<inc_pe> <lanes_per_worker> <warmup> <iterations>
```

本次固定 `first_npu=8`、`bytes=134217728`、`warmup=3`、
`iterations=10`；W2 使用 `pes=3, inc_pe=2`，W4 使用
`pes=5, inc_pe=4`。每次启动前确认 16 张 NPU 均无其他进程。

原始日志保存在：

- `raw/sweep/`：W2/W4 x lanes 1、2、3、4、6；
- `raw/repeat/`：最优配置各三次独立复测。
