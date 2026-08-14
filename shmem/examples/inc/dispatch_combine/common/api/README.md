# Common API / 公共 API

## 中文

### 这个目录是干什么的？

对外稳定的 **C 接入层** 与 **纯 host planner**。新接入只需要看
`inc_dc_single_inc_api.h`；其余层是实现与高级扩展边界。

### 为什么要有这一层？

- 故意不暴露 C++ 对象、SHMEM rank、INC owner、aclrt 头：框架可用纯 C / ctypes / FFI 对接。
- Planner 把「任意大逻辑输入」变成「有界设备 epoch」，避免热路径分配与 workspace 随 batch 线性膨胀。
- SingleInc facade 给普通调用者最小表面；Easy / Inference 保留给需要自定义 scheduler 的高级接入。

### 文件一览

| 文件 | 用途 | 为什么要有 | 库化判断 |
|---|---|---|---|
| `inc_dc_single_inc_api.h` / `.cpp` | **推荐入口**：一个 operator，直接 Dispatch/Combine | 隐藏 session、plan、workspace、request 样板 | 当前产品入口 |
| `inc_dc_framework_c_api.h` / `.cpp` | 底层稳定 C ABI：context、plan、异步 request、workspace、backend vtable、Dispatch/Combine enqueue | 公共 ABI 核心；融合算子与框架对接入口 | **否，不可删** |
| `inc_dc_easy_api.h` / `.cpp` | communicator + token plan 的简化 facade；异步 D/C | 减少拼 framework 描述符；Inference 依赖 | 可选上层，当前交付保留 |
| `inc_dc_inference_api.h` / `.cpp` | session/plan 预分配，热路径尽量无分配；同 plan 可 D∥C | 推理 scheduler 的稳定接入边界 | 适配层，当前交付保留 |
| `inc_dc_chunk_planner.h` / `.cpp` | 将任意大 `logical_rows` 分页为有界 chunk | 可扩展性原语；workspace 不随总行数线性涨 | **否，不可删** |
| `inc_dc_dispatch_route_plan.h` / `.cpp` | 编译每个 Dispatch chunk 的 route / cell 容量 | Dispatch backend 依赖；可 query 再 compile | **否，不可删** |

依赖关系：`SingleInc → Inference → Easy → Framework → backend vtable`；planner 被 Framework / backend 在 host 侧调用。
单 INC 真机 provider 注入时，vtable 通常是 **`native_composite_backend`**（内部再分发到 dispatch/combine），而不是 Framework 直接各挂一个裸 kernel。

首次接入使用 `inc_dc_single_inc_api.h`，并从
[`../examples/inference_api/inc_dc_inference_api_example.cpp`](../examples/inference_api/inc_dc_inference_api_example.cpp)
复制完整生命周期。

---

## English

### What is this directory?

The public **C integration surface** and **host-only planners**. New
integrations only need `inc_dc_single_inc_api.h`; lower layers remain advanced
extension points.

### Why this layer exists

- No C++ objects, SHMEM ranks, INC owners, or `aclrt` headers in the ABI—so
  frameworks can bind from pure C / FFI.
- Planners turn arbitrarily large logical inputs into bounded device epochs,
  avoiding hot-path allocation and linear workspace growth.
- Easy/Inference give Megatron/vLLM-style schedulers a smaller, preparable surface.

### Files

| File | Purpose | Why it exists | Removable? |
|---|---|---|---|
| `inc_dc_single_inc_api.*` | **Recommended entry**: one operator, direct Dispatch/Combine calls | Hides session/plan/workspace/request boilerplate | Product entry |
| `inc_dc_framework_c_api.h` / `.cpp` | Low-level C ABI: context, plan, async requests, workspaces, backend vtable | Public ABI core | **No** |
| `inc_dc_easy_api.h` / `.cpp` | Communicator + token-plan facade | Less descriptor boilerplate; used by Inference | Optional layer; retain now |
| `inc_dc_inference_api.h` / `.cpp` | Prepared session/plan; allocation-free hot path; D∥C on one plan | Inference scheduler boundary | Adapter; retain now |
| `inc_dc_chunk_planner.h` / `.cpp` | Page large `logical_rows` into bounded chunks | Scalability primitive | **No** |
| `inc_dc_dispatch_route_plan.h` / `.cpp` | Compile per-chunk Dispatch routes/capacities | Backend dependency | **No** |

Dependency: `SingleInc → Inference → Easy → Framework → backend vtable`. For single-INC,
the vtable is usually a **`native_composite_backend`** that fans out to the
dispatch/combine providers. Planners are host-side helpers.
