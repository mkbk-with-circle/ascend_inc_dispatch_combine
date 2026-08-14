# Shared layer / 共享层

## 中文

### 这个目录是干什么的？

`common/` 保存最短 SingleInc API、跨模块协议和平台原语，不放具体 kernel。
当前唯一公开入口是 `api/inc_dc_single_inc.hpp`。

### 为什么要单独成层？

- 框架只依赖最短 SingleInc API 与必要协议头，而不是内部适配层。
- 协议与平台策略被单 INC、测试、脚本共同引用；集中放置可避免复制漂移。
- 示例只演示一条 `create → dispatch → compute → combine → destroy` 主路径。

### 子目录

| 子目录 | 用途 | 为什么要有 |
|---|---|---|
| `api/` | 唯一公开的 SingleInc C++ 入口及内部 host 支撑 | 普通调用只看 `inc_dc_single_inc.hpp` |
| `protocol/` | handle、route、packet、layout 等协议类型 | host/device 与跨模块契约，不能散落在实现里 |
| `platform/` | 硬件能力、物理映射、AIV 策略、AIC 原语 | 可移植性与资源分配策略的唯一真相源 |
| `examples/` | 最短调用的完整正确性示例 | 不进入运行库 |

---

## English

### What is this directory?

`common/` holds the minimal SingleInc API plus shared protocol and platform
primitives. Product kernels live under `single_inc/`. The only public entry
point is `api/inc_dc_single_inc.hpp`.

### Why this layer exists

- Frameworks use the minimal SingleInc API and required protocol headers, not
  internal adapter layers.
- Protocol and platform policy are shared by backends, tests, and scripts; one
  home prevents copy-drift.
- The example shows one `create -> dispatch -> compute -> combine -> destroy`
  path.

### Subdirectories

| Subdirectory | Role | Why it exists |
|---|---|---|
| `api/` | Single public C++ facade plus internal host support | Read only `inc_dc_single_inc.hpp` for normal use |
| `protocol/` | Handles, routes, packets, layouts | Host/device and cross-module contracts |
| `platform/` | Capabilities, physical map, AIV policy, AIC primitives | Portability and resource policy source of truth |
| `examples/` | Complete minimal-call correctness example | Not runtime library code |
