# Single-INC tests / 单 INC 测试

## 中文

### 这个目录是干什么的？

针对 `../../single_inc/planning` 与 `../../single_inc/runtime` 的 **host 侧** 回归：
计划编译器、Combine 反向映射、composite backend、expert layout adapter。

### 为什么这些测不放进 `tests/common/`？

- 它们依赖单 INC 特有类型与 provider，不属于跨拓扑公共 ABI。
- 与真机 `scripts/single_inc/` 互补：这里验证「计划/路由/布局」逻辑，脚本验证端到端设备行为。

### 文件一览

| 文件 | 用途 | 为什么要有 |
|---|---|---|
| `test_inc_dc_single_inc_stream_plan_compiler.cpp` | Dispatch tile/workspace 与硬件派生 lane 规划 | 防止 lane/workspace 推算回归 |
| `test_inc_dc_single_inc_combine_plan.cpp` | Dispatch→Combine 反向映射与 assignment 保真 | 钉死正反向一致性（最易漂的一类 bug） |
| `test_inc_dc_native_composite_backend.cpp` | 双 backend 路由、ticket、错误传播 | Easy/Inference 注入的 composite 语义 |
| `test_inc_dc_native_expert_layout_adapter.cpp` | expert-major padding、正反 permutation、零 token expert | 推理布局桥接边界（含空专家） |

---

## English

### What is this directory?

**Host-side** regressions for `../../single_inc/planning` and
`../../single_inc/runtime`: plan compilers, Combine reverse mapping, composite
backend, and expert layout adapter.

### Why not under `tests/common/`?

- These depend on single-INC-specific types/providers, not the cross-topology ABI.
- They complement `scripts/single_inc/`: logic here, end-to-end device behavior there.

### Files

| File | Purpose | Why it exists |
|---|---|---|
| `test_inc_dc_single_inc_stream_plan_compiler.cpp` | Dispatch tiling/workspace + hardware-derived lanes | Catch lane/workspace regressions |
| `test_inc_dc_single_inc_combine_plan.cpp` | Dispatch→Combine reverse mapping fidelity | Pin forward/reverse consistency |
| `test_inc_dc_native_composite_backend.cpp` | Dual-provider routing, tickets, error propagation | Composite semantics for Easy/Inference |
| `test_inc_dc_native_expert_layout_adapter.cpp` | Expert-major padding, permutations, zero-token experts | Inference layout edge cases |
