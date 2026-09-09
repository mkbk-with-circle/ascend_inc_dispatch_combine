# Single-INC Pull V2 — Live Status

本文是当前独立 Dispatch/Combine 的状态真源。当前所有性能和稳定性结论仅来自
**nb-borrow（910B2C）**；本实现尚未在其他集群上验证。

## 当前协议

```text
Dispatch: A READY → INC GET token+route → parse/reorder → INC PUT B
Compute : B expert compute → same-GPU local weighted reduce
Combine : B Notice → INC GET FP32 partials → reduce → INC PUT original A
```

关键约束：

1. Payload 必须经过唯一 INC，不允许 worker 直达 worker。
2. 每个 source token hidden 到 INC 只传一份；INC 按 unique destination fan-out。
3. Dispatch Journal 保存 owner/contributor 关系，Combine 使用同一 generation 的
   sealed Journal，不重新猜测路由。
4. 每个 Worker/Wave 只发布一次 READY 或 Notice；publication-last，所有容量、
   epoch、cookie、digest 和 ring-slot 检查 fail-closed。
5. INC 普通 AIV 动态二分给 Dispatch/Combine，不写死某一 SKU 的 AIV 总数。
6. 正确性失败、timeout、hang、guard 损坏或偶发失败均不得计为性能 PASS。

## 当前资格范围

| 项 | 值 |
|---|---|
| 环境 | nb-borrow，16× Ascend 910B2C |
| CANN | 9.1.0-beta.3 |
| 拓扑 | 0–7、8–15 两个 HCCS 平面 |
| 正式 placement | 同一平面内 W2/W4 + 1 INC |
| W8 | 未测；本机无法构造同平面 8W+1INC |
| 正式 workload | 128 MiB/worker，top-k2 balanced |

## Gate 与正式结果

当前带宽口径为：Dispatch 的 fan-out 下行 hidden 字节 / 完整算子时间；Combine
的归约上行 partial 字节 / 完整算子时间。旧 GET+PUT 相加口径及报告已从当前树删除。

本机 W2/W4 nominal raw 为 56/112 GB/s，92% raw gate 为 51.52/103.04 GB/s；
put-only 既有实测 W2 约 42.7、W4 约 83.3--85.4 GB/s，仅作为调优参照，
不能当作严格上界或用于降低 gate。完整结果与效率见
`nb-borrow/pull_v2_directional_20260909/README.md`。

| 算子 | 规模/路由 | Min GB/s | Mean GB/s | CV |
|---|---|---:|---:|---:|
| Dispatch | W2 top-k2 | 37.618 | 37.849 | 0.509% |
| Combine | W2 top-k2 | 39.289 | 39.429 | 0.197% |
| Dispatch | W4 top-k2 | 69.117 | 69.705 | 0.533% |
| Combine | W4 top-k2 | 77.468 | 77.730 | 0.190% |
| Dispatch | W4 expert-k4/GPU4 | 76.632 | 76.970 | 0.215% |
| Combine | W4 expert-k4/GPU4 | 78.405 | 78.600 | 0.154% |

## 稳定性与扩展性

- W2/W4 Dispatch 与 Combine 各 100 个连续 device waves，全正确。
- 正式 256 MiB/worker 扩展 case 正确且保持带宽等级。
- 62/62 压力矩阵 PASS；覆盖空输入、奇数尾块、不同 top-k、重复目的、hotspot、
  ragged、token/READY skew 和两个 ring slot 复用。
- Host reference 覆盖 5000 个随机 packet、500 个 W2–W8 随机 layout/control plan、
  三种 dtype、可变 top-k、重复 token ID 与非法输入拒绝。

## 当前优化版本安全验证

新增设备安全回归共 14 组、236 wave，通过固定 K2/K4 的坏索引、链环、
token/owner 越界拒绝、连续 ring 复用、少 AIV 和尾块检查。当前优化版本的
Dispatch/Combine 交叠性能尚需重新测定。

## 未关闭事项

- Ragged/non-uniform 路由使用安全重整路径，性能明显低于 uniform fast path。
- Framework/vLLM 热路径仍需实现并资格化 Pull V2 `BackendOps` device adapter。
- 任何新集群都必须重新探测拓扑和 roofline；当前没有跨集群性能结论。

## 权威入口

- `examples/inc/dispatch_combine/single_inc/pull_combine/README.md`
- `examples/inc/dispatch_combine/single_inc/pull_combine/FLOW.md`
- `docs/inc/report/nb-borrow/pull_v2_directional_20260909/README.md`

原始 JSONL/日志不进入 Git，只保留汇总、测试口径与复现脚本。
