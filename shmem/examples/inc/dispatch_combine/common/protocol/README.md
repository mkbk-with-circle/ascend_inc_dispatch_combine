# Common protocol / 公共协议

## 中文

### 这个目录是干什么的？

定义 **host/device 共享** 或 **跨模块契约** 的类型与常量：版本号、组映射、handle、route、packet、对称内存布局、防溢出算术。  
本目录几乎全是头文件；它们是 ABI 的一部分，不是「随便内联进某个 kernel」的实现细节。

### 为什么要单独拆文件？

- 错误合并会放大重编译范围，并模糊「谁拥有 ABI」。
- device kernel、host planner、Framework API、测试必须对同一布局达成一致；集中定义避免静默错位。
- `checked_arith` 单独存在，是为了强制尺寸计算 fail-closed，而不是散落 `size_t` 乘法。

### 文件一览

| 文件 | 用途 | 为什么要有 |
|---|---|---|
| `inc_dc_types.h` | ABI/packet/layout/handle 版本；组规模/topk/expert 上限；Dispatch/Combine/overlap 的 AIV block 区间；公共状态码 | 全树共享的容量与状态「宪法」 |
| `inc_dc_group.h` | logical rank ↔ SHMEM PE ↔ expert 映射 | 路由必须知道「专家在哪」；与物理实现解耦 |
| `inc_dc_handle.h` | token 身份、Dispatch handle、Combine inverse entry | Combine 散写后能按身份找回归并位置 |
| `inc_dc_route.h` | tensor 描述、route 规格、assignment 查询 | 描述「token×topk→expert」而不绑定某 backend |
| `inc_dc_packet.h` | host/device 共享 packet header 与 assignment metadata | 设备上可见的包格式契约 |
| `inc_dc_inline_route_protocol.h` | V2 token 自带路由、INC 在线解析的 transport-neutral wire ABI | 未来真实 INC 与当前 SHMEM 样机共享语义 |
| `inc_dc_layout.h` | Dispatch 对称内存布局描述 | 各 rank 可见 buffer 的摆放规则 |
| `inc_dc_checked_arith.h` | 防溢出的 size/offset 算术 | 避免 workspace/offset 静默 wrap 导致内存破坏 |

---

## English

### What is this directory?

Host/device and cross-module **protocol contracts**: versions, group maps,
handles, routes, packets, symmetric layouts, and overflow-safe arithmetic.
Almost all headers—these are ABI, not implementation details to inline into a
single kernel.

### Why separate files?

- Merging them only to reduce file count widens rebuild scope and blurs ABI
  ownership.
- Kernels, planners, Framework API, and tests must agree on one layout.
- Checked arithmetic is isolated so size math fails closed instead of wrapping.

### Files

| File | Purpose | Why it exists |
|---|---|---|
| `inc_dc_types.h` | ABI versions, capacity ceilings, AIV block ranges, status codes | Shared “constitution” of limits and status |
| `inc_dc_group.h` | Logical-rank ↔ SHMEM-PE ↔ expert mapping | Routing without baking in a backend |
| `inc_dc_handle.h` | Token identity, Dispatch handle, Combine inverse entries | Recover combine destinations after scatter |
| `inc_dc_route.h` | Tensor descriptors, routes, assignment queries | Describe token×topk→expert independently |
| `inc_dc_packet.h` | Shared packet headers and assignment metadata | Device-visible packet contract |
| `inc_dc_inline_route_protocol.h` | Transport-neutral V2 token-carried route ABI parsed by the INC | Shared semantics for real INC and the SHMEM emulator |
| `inc_dc_layout.h` | Dispatch symmetric-memory layout | How cross-rank buffers are arranged |
| `inc_dc_checked_arith.h` | Overflow-safe size/offset math | Prevent silent wrap in workspace math |
