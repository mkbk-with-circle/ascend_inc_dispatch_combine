# Single-INC Pull V2 — 910b2c-nb 环境状态

本页记录nb-borrow硬件事实及已有链路定标。当前源分区协议的资格范围以
[分区报告](../../report/nb-borrow/source_partitions_20260910/README.md)为准；下方旧压力数据仅作历史对照。

## 环境

| 项 | 值 |
|---|---|
| NPU | 16× Ascend 910B2C，65536 MiB HBM/卡 |
| CANN | `/usr/local/Ascend/cann-9.1.0-beta.3` |
| 驱动 | `25.0.rc1.1` |
| 普通 AIV | 48/卡，运行时查询 |
| 拓扑 | HCCS 平面 A=0–7，平面 B=8–15 |
| 正式 placement | 同平面 W2/W4 + 1 INC |
| W8 | N/A：8W+1INC 无法放入一个 8-card 平面 |

### 物理互联与 Rank 映射

```text
平面 A：NPU0--7   HCCS full mesh，每条 peer link 224 Gbit/s ≈ 28 GB/s raw
平面 B：NPU8--15  HCCS full mesh，每条 peer link 224 Gbit/s ≈ 28 GB/s raw
跨平面：PIX / PHB / SYS，不属于当前资格路径
```

| 规模 | Worker Rank/Phy | INC Rank/Phy | World size | 同平面链路 |
|---|---|---|---:|---|
| W2 | `0/0`, `1/1` | `2/2` | 3 | 2×HCCS |
| W4 | `0/0`…`3/3` | `4/4` | 5 | 4×HCCS |

协议只使用 Worker↔INC 边；Worker 之间的 full-mesh HCCS 不允许作为 bypass。平面 A
剩余卡和整个平面 B 在正式 case 中空闲。W8 world size=9，无法容纳于单平面，故
没有可与 W2/W4 等价的本机 W8 baseline。

## 链路 Roofline

每条worker↔INC HCCS peer标称224 Gbit/s（约28 GB/s raw）。
当前讲解已撤去旧raw百分比gate，使用以下已有put-only链路峰值定标作为实测参照。
表中Mean是峰值测试均值，Min是最低样本，并非新测单样本最大值。

| 规模 | 方向 | Min | Mean | CV |
|---|---|---:|---:|---:|
| W2 | all→INC | 42.714 | 42.734 | 0.0221% |
| W2 | INC→all | 42.804 | 42.815 | 0.0136% |
| W4 | all→INC | 85.445 | 85.478 | 0.0244% |
| W4 | INC→all | 83.258 | 84.038 | 1.2484% |

这些单向值是本机链路实测参照，本轮未重跑定标。它们不构成严格物理理论上限。
Dispatch使用下行hidden字节、Combine使用上行partial字节，均除以完整算子时间。

## 旧紧凑布局结果（历史）

128 MiB/worker、top-k2 balanced、3 warmup + 10 measure：

当前性能统一使用单方向有效带宽：Dispatch 统计 fan-out 下行，Combine 统计归约
上行，并除以完整算子时间。最新 W2/W4、K2/K4/K8 结果见下方唯一权威报告。

## 旧紧凑布局稳健性（不作为新分区资格）

- W2/W4 Dispatch 与 Combine 各 100 个连续 device waves，全正确；
- 256 MiB/worker 扩展 case 正确；
- 62/62 size/skew/hotspot/ragged/READY-skew 压力 case PASS；
- 覆盖 0 token、1 byte、4 KiB–256 MiB、top-k1/top-k2/top-k=all、重复目的、
  token skew 0%–100% 和两个 ring slot 复用。

## 报告入口

- 当前源分区：`docs/inc/report/nb-borrow/source_partitions_20260910/README.md`
- 历史紧凑布局：`docs/inc/report/nb-borrow/pull_v2_directional_20260909/README.md`

原始 JSONL、PE 日志和 build 产物保存在实验机本地，不进入 Git。
