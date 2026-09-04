# Pull V2 抽象流程

本文只描述协议阶段和生命周期，不对应某个具体函数。

## Dispatch

```text
┌──────────────────────── A 端 Worker ────────────────────────┐
│ 1. 生成 Hidden、Token ID、Top-k GPU/Expert/Weight           │
│ 2. 写入一个完整且不可变的 Source Slot                       │
│ 3. 每个 Wave 只发布一次 READY                               │
└────────────────────────────┬─────────────────────────────────┘
                             │ READY：只表示整个 Slot 已就绪
                             ▼
┌──────────────────────── INC Dispatch 半区 ──────────────────┐
│ 4. 主动 GET Header，验证 Session / Wave / Shape / Capacity  │
│ 5. 并行 GET Token Metadata 与 Route Metadata                │
│ 6. 在线校验 CSR、GPU、Expert、Ordinal、Weight、Digest       │
│ 7. 计算每个目标的 Row / Assignment / Expert 前缀            │
│ 8. 建立 Journal：Owner Token ↔ Contributor B Rows           │
│                                                              │
│ 9. 每个 Token Hidden 只从源 Worker GET 一次                 │
│    ├─ Uniform：Worker → INC UB → 多个目标直接 PUT            │
│    └─ Non-uniform：INC 按目标重整 → 每目标一次 Bulk PUT       │
│                                                              │
│10. PUT DestinationRow / ExpertAssignment / ExpertCount      │
│11. 等待所有远端数据可见，Journal 进入 DISPATCH_SEALED       │
└───────────────┬───────────────────────────────┬──────────────┘
                │ Destination Completion        │ Source ACK
                ▼                               ▼
┌──────────────── B 端 Worker ─────────────┐  ┌── A 端 Worker ─┐
│ 可以消费 Expert 输入，开始 Expert Compute │  │ 可以复用源 Slot │
└───────────────────────────────────────────┘  └────────────────┘
```

Dispatch 的核心语义：

```text
一份源 Token Hidden → INC → 每个命中的 Unique Destination 一份 Hidden
```

同一 GPU 上多个 Expert 只增加 Assignment，不增加网络 Hidden 副本。

## Combine

```text
┌──────────────────────── B 端 Worker ─────────────────────────┐
│ 1. 执行 Expert FFN                                           │
│ 2. 同一 Token 在本 GPU 上的多个 Expert 先做 Local Reduce      │
│ 3. FP32 Partial Rows 与 128B READY 写入注册 Ring Slot         │
│ 4. READY 留在 B 端，只向 INC 发布一次 64B Notice              │
└────────────────────────────┬──────────────────────────────────┘
                             │ Notice
                             ▼
┌──────────────────────── INC Combine 半区 ────────────────────┐
│ 5. 验证 Notice 的 Session / Generation / Wave / Source       │
│ 6. 主动 GET 128B READY，校验 Cookie / Row Count / Offset     │
│ 7. 读取 Dispatch Seal 时生成的确定性 Pull Index              │
│ 8. 已 READY 的 Source 可以先处理，不等待 Rank 顺序            │
│                                                               │
│ 9. 按 Token GET 各 B 的 Partial Rows                          │
│    ├─ Top-k2：两个 GET → 一次 FP32 Add                        │
│    └─ Generic：按 Journal 固定顺序逐 Contributor Reduction    │
│                                                               │
│10. 上一 Task 的 Owner PUT 与下一 Task 的 Partial GET 交叠     │
│11. 最终结果只 PUT 给原始 Owner Rank / Owner Row               │
│12. 全部远端可见后，Journal 进入 COMPLETE 或 ABORTED           │
└───────────────┬───────────────────────────────┬───────────────┘
                │ Combine Source ACK            │ Owner Completion
                ▼                               ▼
┌──────────────── B 端 Worker ─────────────┐  ┌── A 端 Owner ───┐
│ 可以复用 Partial Slot                     │  │ 可以消费最终输出 │
└───────────────────────────────────────────┘  └─────────────────┘
```

Combine 的核心语义：

```text
多个 B Local Partial → INC FP32 Reduction → 原始 A Owner 的一份结果
```

## 不同 Wave 的交叠

```text
时间 ─────────────────────────────────────────────────────────►

INC Dispatch 半区： [ Dispatch Wave N+1 ─────────────────── ]
INC Combine 半区：       [ Combine Wave N ─────────── ]
B Expert Compute：                                         [ N+1 ]
```

`Combine(N)` 可以和 `Dispatch(N+1)` 任意错峰；同一 Wave 的 Combine 仍必须等待
对应 Dispatch Journal Sealed。
