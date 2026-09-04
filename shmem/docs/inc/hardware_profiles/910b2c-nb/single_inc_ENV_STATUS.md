# Single-INC Pull V2 — 910b2c-nb 环境状态

本页只记录当前 Pull V2 在 nb-borrow 上的环境事实和资格结果。

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

## 链路 Roofline

每条 worker↔INC HCCS peer raw 约 28 GB/s；W2/W4 raw 聚合口径分别为
56/112 GB/s。

| 规模 | 方向 | Min | Mean | CV |
|---|---|---:|---:|---:|
| W2 | all→INC | 42.714 | 42.734 | 0.0221% |
| W2 | INC→all | 42.804 | 42.815 | 0.0136% |
| W4 | all→INC | 85.445 | 85.478 | 0.0244% |
| W4 | INC→all | 83.258 | 84.038 | 1.2484% |

这些单向 roofline 只用于解释物理链路；Pull V2 正式 gate 使用完整 GET+PUT 流水的
logical bytes 口径，固定为 raw 聚合的 92%。

## Pull V2 正式结果

128 MiB/worker、top-k2 balanced、3 warmup + 10 measure：

| 算子 | 规模 | Gate | Min | Mean | CV | 结果 |
|---|---:|---:|---:|---:|---:|---|
| Dispatch | W2 | 51.52 | 56.505 | 56.906 | 0.237% | PASS |
| Dispatch | W4 | 103.04 | 104.907 | 105.627 | 0.476% | PASS |
| Combine | W2 | 51.52 | 56.641 | 56.750 | 0.105% | PASS |
| Combine | W4 | 103.04 | 107.254 | 107.663 | 0.194% | PASS |

## 稳健性

- W2/W4 Dispatch 与 Combine 各 100 个连续 device waves，全正确；
- 256 MiB/worker 扩展 case 正确；
- 62/62 size/skew/hotspot/ragged/READY-skew 压力 case PASS；
- 覆盖 0 token、1 byte、4 KiB–256 MiB、top-k1/top-k2/top-k=all、重复目的、
  token skew 0%–100% 和两个 ring slot 复用。

## 权威报告

- `docs/inc/report/nb-borrow/pull_v2_qualified_20260904/README.md`
- `docs/inc/report/nb-borrow/pull_v2_overlap_stress_20260905/README.md`

原始 JSONL、PE 日志和 build 产物保存在实验机本地，不进入 Git。
