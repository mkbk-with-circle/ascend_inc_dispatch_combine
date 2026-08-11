# Single-INC overlap 理论/实测校准（CANN 9.1，2026-08-04）

## 结论

- 并发资源是硬件/拓扑固定策略：48 AIV 上 INC Dispatch/Combine=`16/32`，worker W2/W4/W8 D/C=`8/24`,`4/16`,`2/12`；不随 K、bytes、result/token 数或 route plan 变化。
- 跨 session 共同 deadline 改为设备侧预驻留 + target cycle 释放。修复前曾出现 `overlap_us=0` 或仅 0.2 ms；修复后正式 owner31 矩阵 30/30 真实交叠，owner32 定向 6/6 真实交叠。
- 恢复 owner32 是为了消除 owner31 对 balanced Combine 约 5% 的回退；预驻留已解决先前 48/48 满分区的启动竞态。普通单算子 `start_target_cycle=0`，不走该资格化路径。

## CANN 9.1 push roofline

W8，每个 worker 128 MiB，总计每方向 1 GiB，push-only，8 KiB chunk：

| 项 | GB/s |
|:---|---:|
| 单向 down（INC→workers） | 123.339 |
| 单向 up（workers→INC） | 146.217 |
| 等量 duplex down | 112.124 |
| 等量 duplex up | 142.748 |

1 MiB chunk 的交叉检查为 down/up 单向 122.424/146.666 GB/s；不用其偶发高于单向的 duplex-up 样本抬高理论上限。roofline probe 同时修复了 CANN 9.1 要求 symmetric heap 按 2 MiB 页对齐的可移植性问题。

## 理论公式

令 `D/U` 为 Dispatch 下行/Combine 上行的计量分子，`R` 为 Dispatch 物理目标 rank 数，`K` 为 Combine top-k：

```
down_bytes = D + U/K
up_bytes   = D/R + U
αdown = single_down/duplex_down - 1 = 0.10002
αup   = single_up/duplex_up - 1     = 0.02430
Tdown = (down_bytes + αdown·up_bytes) / single_down
Tup   = (up_bytes + αup·down_bytes) / single_up
Ttheory = max(Tdispatch_solo, Tcombine_solo, Tdown, Tup)
gain_theory = 1 - Ttheory/(Tdispatch_solo + Tcombine_solo)
```

`Ttheory` 是不可突破的乐观下界；实测 makespan 若比它低超过 2% 必须 FAIL，因为这意味着理论或计量口径有误。脚本已将原先的「两个方向共用一个 GB/s」修正为非对称 duplex 包络并增加该 gate。

## 256 MiB/K8 对照

| W / route | 理论 makespan us | 实测 us | 高于理论 | 理论收益 | 实测收益 |
|:---|---:|---:|---:|---:|---:|
| W2 balanced | 2804.00 | 3098.59 | 10.51% | 33.48% | 26.50% |
| W2 all-to-one | 3721.93 | 4382.96 | 17.76% | 27.52% | 14.65% |
| W4 balanced | 2720.57 | 2761.36 | **1.50%** | 34.42% | 33.44% |
| W4 all-to-one | 4087.77 | 5245.72 | 28.33% | 36.15% | 18.06% |
| W8 balanced | 2693.36 | 2729.10 | **1.33%** | 33.89% | 33.02% |
| W8 all-to-one | 4879.91 | 5883.77 | 20.57% | 32.18% | 18.22% |

balanced W4/W8 已基本贴合理论。all-to-one 的额外 gap 来自单个热点 worker 上双向 push 和 Combine protocol 同时受压；均匀 W8 duplex roofline 是严格的乐观下界，不包含热点拓扑惩罚。Combine-first stagger 0.5–2.2 ms 的 5 个定向实验没有降低 makespan，因此不引入 workload-dependent 调度特判。

W8 balanced 补充 5 次为 2661.76–2798.34 us，CV=1.63%；与正式首轮合并 10 次均值 2723.61 us、CV=3.88%，关闭 H11 稳定性边缘项。

## 证据

- 正式 owner31 过渡矩阵的关键数据已并入本文；原始过渡 JSON 已清理。
- `/tmp/roofline-cann91-8k-1785782463-{ingress,egress,duplex}`
- `/tmp/overlap-soak-w8-bal-*`
- `/tmp/overlap-owner32-prelaunch-*`

## 跟进压测

2026-08-04 已增加 bytes 非对称、尺度、K1/K64、热点与 submit-order
矩阵，理论仍使用本文非对称 duplex 包络。矩阵和预计方向下界见
`single_inc_overlap_followup_plan_cann91_20260804.md`；实测在新增 200 个系统熵
plan 后自动执行，避免 NPU 争用污染结果。
