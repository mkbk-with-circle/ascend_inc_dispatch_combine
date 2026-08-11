# Single-INC Dispatch/Combine 最终压测（CANN 9.1，2026-08-04）

## 结论

- 代表 sweep：W2/W4/W8 × K1/K8 × Dispatch/Combine，256 MiB，warmup=3、measure=10，**12/12 正确**。
- 正则矩阵性能 gate：**11/12 PASS**。唯一失败是 Combine W8/K1=114.643 GB/s，既低于临时 115 gate，也低于最终 120 目标；它与已知 K1 GAP 一致，不是连续下发导致的回退。
- 连续下发：Dispatch W8/K8 同一 stream 20 epoch 的 device 区间严格串行，Combine 同一 service stream 的 6 个正则 case 各 20 epoch 均全部正确。与对应 10-epoch 单测相比，带宽差仅 **-0.11%…+0.68%**。
- 资源不变：INC D/C=`16/32`；worker W2/W4/W8 D/C=`8/24`,`4/16`,`2/12`，所有 K1/K8 与连续 epoch 都使用同一硬件/拓扑固定分配。

## 正则代表 sweep

| Operator | W2/K1 | W2/K8 | W4/K1 | W4/K8 | W8/K1 | W8/K8 |
|:---|---:|---:|---:|---:|---:|---:|
| Dispatch GB/s | 120.240 | 127.582 | 119.768 | 133.687 | 119.847 | 133.345 |
| Dispatch CV | 0.40% | 0.68% | 0.62% | 1.16% | 1.09% | 0.61% |
| Combine GB/s | 115.069 | 133.796 | 115.676 | 129.455 | **114.643** | 126.538 |

Combine 的该次 sweep 用显式正则矩阵：K1 row=16 KiB，general row=8 KiB。这是资格化测试 shape，不会改变 operator AIV 分配。原 sweep harness 的 W8/K8 24 KiB 非正则 row 只有 103.72 GB/s；该失效中间报告已在项目清理时删除，不作当前证据。

## 连续下发对照

Dispatch W8/K8，20 epoch：

- mean=2015.435 us，133.190 GB/s，CV=0.72%，min/max=1994.24/2051.86 us。
- 对应 10 epoch sweep=133.345 GB/s，差 -0.12%。
- 20 个 `[start_cycle,end_cycle]` 均满足 `start[i] >= end[i-1]`，最小间隔 25187 cycles；无重叠，严格串行。

Combine 各 20 epoch：

| Case | 10-epoch sweep GB/s | 20-epoch queue GB/s | 差异 |
|:---|---:|---:|---:|
| W2/K1 | 115.069 | 115.337 | +0.23% |
| W2/K8 | 133.796 | 134.706 | +0.68% |
| W4/K1 | 115.676 | 115.545 | -0.11% |
| W4/K8 | 129.455 | 129.490 | +0.03% |
| W8/K1 | 114.643 | 115.197 | +0.48% |
| W8/K8 | 126.538 | 126.603 | +0.05% |

Combine 的 20 epoch 是同一 ACL stream 上的 generation 队列；stream 语义保证串行，设备 service span 覆盖首个 generation start 到最后一个 generation end。上表证明队列填充/排空未稀释单算子带宽。

## 证据

- 正则 sweep：`single_inc_final_regular_sweep_cann91_20260804.json`
- Dispatch 20 epoch：`/tmp/dispatch-seq20-regular-w8-k8-20260804`
- Combine 20 epoch：`/tmp/combine-seq20-1785783358-*`

## 未关闭项

Combine K1 最终 `>=120 GB/s` 仍为 GAP。本轮没有放宽该目标，也没有因为压测正确/稳定就把 K1 标记为达标。
