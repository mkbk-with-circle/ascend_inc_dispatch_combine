# Single-INC planning / 单 INC 计划编译

## 中文

### 这个目录是干什么的？

把框架侧的 **dense token route** 编译成单 INC 设备能吃的 **有界 workspace / assignment / tile**，
并生成与 Dispatch **严格反向** 的 Combine contribution layout。

### 为什么必须独立存在（不能塞进热路径）？

- 任意大输入要分页；编译器负责 deterministic chunk/generation，热路径只消费结果。
- Dispatch 与 Combine 必须共享同一顺序语义；两边各写一套临时逻辑必然漂移。
- 挪进 runtime 临时拼装会重新引入分配、不可复现路由和难测边界条件。

### 文件一览

| 文件 | 用途 | 为什么要有 |
|---|---|---|
| `inc_dc_single_inc_stream_plan_compiler.h` / `.cpp` | dense route → Dispatch 物理行、assignment、tile/workspace、硬件派生 lane | Dispatch backend / service 的计划真相源 |
| `inc_dc_single_inc_combine_plan_compiler.h` / `.cpp` | 从同一 Dispatch 顺序生成严格反向 Combine contribution layout | 保证 Combine 与 Dispatch 一一对应，避免「正反向不一致」类 bug |

---

## English

### What is this directory?

Compilers that turn framework **dense token routes** into single-INC **bounded
workspaces / assignments / tiles**, and produce a Combine contribution layout
that is the **exact reverse** of Dispatch ordering.

### Why not fold this into the hot path?

- Large inputs must be paged; compilers own deterministic chunks/generations.
- Dispatch and Combine must share one ordering; duplicated ad-hoc logic drifts.
- Hot-path assembly reintroduces allocation, non-reproducible routes, and hard-to-test edges.

### Files

| File | Purpose | Why it exists |
|---|---|---|
| `inc_dc_single_inc_stream_plan_compiler.*` | Dense route → Dispatch rows, assignments, tiles, lanes | Dispatch plan source of truth |
| `inc_dc_single_inc_combine_plan_compiler.*` | Exact reverse Combine layout from Dispatch order | Forward/reverse fidelity |
