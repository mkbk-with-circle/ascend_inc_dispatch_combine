# nb-borrow Pull-Dispatch V2 完整算子（2026-09-03）

本目录与旧 Push-Dispatch、V1 以及其他机器结果隔离。`raw/pre_parallel/`
保存并行 parser 优化前的正确性检查点和匹配 workload 的 relay roof；后续正式
92% gate 结果写入新的子目录，不覆盖这里。

## Gate

固定带宽 gate 只适用于 BF16、每 worker 128 MiB hidden、worker 输入与目标
负载均对称的完整 Dispatch：

| 规模 | nominal raw | 92% gate |
|---:|---:|---:|
| W2 | 56 GB/s | min >= 51.52 GB/s |
| W4 | 112 GB/s | min >= 103.04 GB/s |

完整 logical bytes 为 `hidden GET 一次 + 每个 unique destination PUT 一次`；
READY、metadata GET/校验、路由解析、动态布局、journal、目标 metadata、completion
和 source ACK 的时间全部计入 makespan。

## 并行 parser 前的完整算子检查点

| case | logical bytes | makespan / bandwidth | 正确性 |
|---|---:|---:|---|
| W2，1 MiB/worker，单 wave | 6 MiB | 754.756 us / 8.34 GB/s | 全量 PASS |
| W2，1 MiB/worker，双 slot 4 wave | 6 MiB/wave | measure mean 623.042 us / 约 10.10 GB/s | 4/4 PASS，CV 0.671% |
| W4，1 MiB/worker，双 slot 3 wave | 20 MiB/wave | measure mean 2103.577 us / 约 9.97 GB/s | 3/3 PASS，CV 0.813% |
| W2，128 MiB/worker，对称 dense | 768 MiB | 68.984 ms / **11.674 GB/s** | 全量 PASS，未过性能 gate |

128 MiB W2 timeline 显示：READY+metadata/FNV 约 14 ms、block0 串行
route/layout 约 38 ms、hidden GET->PUT relay 约 13.6 ms。当前缺口属于控制面
串行化，不属于链路不稳定；因此后续采用 24-AIV cohort 两遍解析，而不是遍历
经验参数。

## 匹配 fanout 的纯 relay roof

每 worker 128 MiB，8 KiB/5 channels 的第一轮对照：

| case | tile | min / mean GB/s | 结论 |
|---|---:|---:|---|
| W2 fanout2 | 8 KiB | 60.376 / 60.537 | 可过 W2 gate |
| W4 fanout4 | 8 KiB | 65.420 / 67.295 | 多目的 PUT 持有 UB 过久 |

按 fanout 缩短 tile 后：

| case | tile | channels/source | min / mean GB/s |
|---|---:|---:|---:|
| W2 fanout2 | 4 KiB | 5 | 59.945 / 60.234 |
| W2 fanout2 | **6 KiB** | 5 | **61.069 / 61.110** |
| W4 fanout4 | 2 KiB | 5 | 102.723 / 102.807 |
| W4 fanout4 | **3 KiB** | 5 | **103.788 / 103.818** |
| W4 fanout4 | 4 KiB | 4 | 103.571 / 103.623 |
| W4 fanout4 | 4 KiB | 5 | 103.675 / 103.732 |
| W4 fanout4 | 4 KiB | 6 | 101.584 / 101.841 |
| W4 fanout4 | 6 KiB | 5 | 79.251 / 81.675 |

据此采用与规模无关的初始协议规则：

```text
relay_tile_bytes = min(8 KiB, align64(12 KiB / max_unique_fanout))
```

它在 fanout2 选择 6 KiB、fanout4 选择 3 KiB。正式 gate 仍必须由完整算子
10 个 measure 的最小值判定；纯 relay 只说明数据面屋顶，不能代替完整结果。
