# Shared layer / 共享层

## 中文

### 这个目录是干什么的？

`common/` 只保存 **跨产品共享** 或 **对框架公开** 的代码：不放单 INC / 多 INC 的具体 kernel。  
公开符号统一使用 `inc_dc_` 前缀，以保持 ABI 兼容。

### 为什么要单独成层？

- 框架（Megatron / vLLM / 融合算子）应只依赖稳定 C ABI 与协议头，而不是某个拓扑的 `.cpp`。
- 协议与平台策略被单 INC、多 INC、测试、脚本共同引用；集中放置可避免复制漂移。
- 示例放在这里做 C11 编译门禁，防止公开头文件「C++ 化」后破坏 C 接入。

### 子目录

| 子目录 | 用途 | 为什么要有 |
|---|---|---|
| `api/` | 稳定 C ABI、Easy/Inference API、共享 host planner | 对外接入面；库化时的最小/推荐边界 |
| `protocol/` | handle、route、packet、layout 等协议类型 | host/device 与跨模块契约，不能散落在实现里 |
| `platform/` | 硬件能力、物理映射、AIV 策略、AIC 原语 | 可移植性与资源分配策略的唯一真相源 |
| `examples/` | 仅编译验证的 C 接入示例 | 防止公开头文件漂移；不进入运行库 |

---

## English

### What is this directory?

`common/` holds only **cross-product** or **framework-facing** code. Topology-specific
kernels live under `single_inc/` / `multi_inc/`. Public symbols keep the `inc_dc_`
prefix for ABI compatibility.

### Why this layer exists

- Frameworks should depend on a stable C ABI and protocol headers, not on a
  particular topology implementation.
- Protocol and platform policy are shared by backends, tests, and scripts; one
  home prevents copy-drift.
- C examples act as C11 compile gates so public headers stay C-callable.

### Subdirectories

| Subdirectory | Role | Why it exists |
|---|---|---|
| `api/` | Stable C ABI, Easy/Inference APIs, shared host planners | Public integration surface |
| `protocol/` | Handles, routes, packets, layouts | Host/device and cross-module contracts |
| `platform/` | Capabilities, physical map, AIV policy, AIC primitives | Portability and resource policy source of truth |
| `examples/` | Compile-checked C integration samples | Header drift guards; not runtime library code |
