# Inference API example / 推理 API 示例

## 中文

### 这个目录是干什么的？

演示 **Inference API** 的推荐用法：创建 session → 为固定 `(tokens, topk)` bucket
预分配 Dispatch/Combine plan → 在 scheduler 热路径异步提交 → 显式管理
request、route handle、plan、session 生命周期。

### 为什么要单独示例（而不是复用 Easy 示例）？

- Easy 示例偏「每次拼 op」；推理热路径需要 **prepare once, submit many**。
- 同 plan 上 Dispatch 与 Combine 可同时 in-flight——这是 Inference 层的关键语义，必须有示例钉死。
- 不进入运行库；可作为 SDK 示例随源码提供。

### 文件一览

| 文件 | 用途 | 为什么要有 |
|---|---|---|
| `inference_loop.c` | 唯一示例：session/plan 预分配 + 热路径异步提交 + 生命周期销毁 | 钉死推理 scheduler 接入契约；C11 编译门禁 |

---

## English

### What is this directory?

The recommended **Inference API** usage: create a session, prepare Dispatch and
Combine plans for a fixed `(tokens, topk)` bucket, submit asynchronously from a
scheduler hot path, and manage request/route/plan/session lifetimes explicitly.

### Why a separate sample from Easy?

- Easy samples build ops ad hoc; inference needs **prepare once, submit many**.
- One plan may have Dispatch and Combine in flight together—that semantic must
  be demonstrated.
- Not runtime library code; suitable as an SDK source sample.

### Files

| File | Purpose | Why it exists |
|---|---|---|
| `inference_loop.c` | Sole sample: prepare session/plans, async submit, destroy | Pins the scheduler contract; C11 compile gate |
