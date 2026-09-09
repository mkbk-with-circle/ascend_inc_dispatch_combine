# Single-INC Pull V2 Sweep 状态

当前实现只在 **nb-borrow / 910B2C / CANN 9.1.0-beta.3** 上完成资格测试。
仓库不保留其他集群的旧 sweep 数值，也不得把旧实现的结果用于当前 Pull V2 判定。

## 环境与范围

| 项 | 当前状态 |
|---|---|
| Hardware profile | `910b2c-nb` |
| 拓扑 | 两个独立 8-card HCCS 平面 |
| 正式规模 | 同一平面内 W2+1INC、W4+1INC |
| W8 | N/A：8 worker + 1 INC 无法放入同一 8-card 平面 |
| 正式消息 | 128 MiB/worker，top-k2 balanced |
| 稳定性 | 3 warmup + 10 measure，CV ≤ 5% |

## 正式 Gate 与结果

本机 W2/W4 目标为 51.52/103.04 GB/s，当前尚未达到。
Dispatch 计 fan-out 下行字节，Combine 计归约上行字节，均除以完整算子时间。
最新数值统一见下方单方向报告。

## 交叠与稳健性

- 当前 Combine 优化版本的交叠性能需要重新测定。
- 62 个 size/skew/hotspot/ragged/READY-skew case：62/62 PASS。
- Dispatch/Combine W2/W4 各连续 100 device waves：全部正确，无 timeout、guard
  损坏或状态泄漏。
- 覆盖 0 token、1 byte、4 KiB–256 MiB、top-k1/top-k2/top-k=all、重复目的、
  ragged、token skew 0%–100% 和 READY skew 0–1000 us。

## 当前开放项

1. Ragged/non-uniform 安全重整路径仍需性能优化与新口径复测。
2. 独立 API 的真实框架热路径仍需部署环境对应的 `BackendOps` device adapter。
3. W8 与其他硬件/拓扑均未验证；移植后必须重新测 roofline、正确性和 gate。

## 证据

- [当前单方向带宽与稳健性报告](../../../../docs/inc/report/nb-borrow/pull_v2_directional_20260909/README.md)
- [协议、API 与构建](pull_combine/README.md)
- [Dispatch/Combine 流程图](pull_combine/FLOW.md)

逐 PE JSONL 和原始日志仅保存在实验机本地，不进入协作仓库。
