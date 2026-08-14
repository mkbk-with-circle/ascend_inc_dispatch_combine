# Common tests / 公共层测试

## 中文

### 这个目录是干什么的？

针对 `../../common/api` 的内部 host 回归与头文件编译门禁；这里不是用户调用示例。

### 为什么仍测试 Framework / Easy / Inference？

- `.cpp`：测状态机、并发、错误码、生命周期等行为。
- `.c`：证明对应公开头可被纯 C11 编译——ABI「C 可调用」硬门禁。
- `chunk_planner` / `dispatch_route_plan` **只有** `.cpp` 行为测试（无单独 `.c` 头门禁文件）。

### 文件一览

| 文件 | 用途 | 为什么要有 |
|---|---|---|
| `test_inc_dc_framework_c_api.cpp` | request 状态机、workspace、错误、并发语义 | Framework ABI 行为门禁 |
| `test_inc_dc_easy_api.cpp` | Easy communicator、token plan、route handle、生命周期 | Easy 层行为门禁 |
| `test_inc_dc_inference_api.cpp` | prepared plan、D/C 并发、销毁保护 | Inference 热路径语义门禁 |
| `test_inc_dc_chunk_planner.cpp` | 超大输入分页、边界与溢出 | 可扩展性原语正确性 |
| `test_inc_dc_dispatch_route_plan.cpp` | W/top-k route 编译与容量边界 | Dispatch route compiler 正确性 |
| `test_inc_dc_framework_c_header.c` | framework header 纯 C11 编译 | 防止头文件 C++ 化 |
| `test_inc_dc_easy_c_header.c` | Easy header 纯 C11 编译 | 同上 |
| `test_inc_dc_inference_c_header.c` | inference header 纯 C11 编译 | 同上 |

---

## English

### What is this directory?

Internal host regressions for `../../common/api`; this directory is not a usage guide.

### Why Framework / Easy / Inference are still tested

- `.cpp`: behavior—state machines, concurrency, errors, lifetimes.
- `.c`: proves those public headers still compile as C11.
- Planners (`chunk_planner`, `dispatch_route_plan`) have `.cpp` tests only.

### Files

| File | Purpose | Why it exists |
|---|---|---|
| `test_inc_dc_framework_c_api.cpp` | Request state machine, workspaces, errors, concurrency | Framework behavior gate |
| `test_inc_dc_easy_api.cpp` | Easy communicator, token plans, route handles, lifetime | Easy behavior gate |
| `test_inc_dc_inference_api.cpp` | Prepared plans, D/C overlap, destroy guards | Inference semantics gate |
| `test_inc_dc_chunk_planner.cpp` | Large-input paging, boundaries, overflow | Scalability primitive correctness |
| `test_inc_dc_dispatch_route_plan.cpp` | Worker/top-k route compile + capacity | Route compiler correctness |
| `test_inc_dc_framework_c_header.c` | Pure-C11 framework header compile | Stop C++-only header drift |
| `test_inc_dc_easy_c_header.c` | Pure-C11 Easy header compile | Same |
| `test_inc_dc_inference_c_header.c` | Pure-C11 inference header compile | Same |
