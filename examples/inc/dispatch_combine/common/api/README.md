# Common API / 公共 API

## 中文

### 这个目录是干什么的？

对外稳定的 **C 接入层** 与 **纯 host planner**。  
通信库最小核心 = Framework C API + 两个 planner；Easy / Inference 是更薄的上层边界。

### 为什么要有这一层？

- 故意不暴露 C++ 对象、SHMEM rank、INC owner、aclrt 头：框架可用纯 C / ctypes / FFI 对接。
- Planner 把「任意大逻辑输入」变成「有界设备 epoch」，避免热路径分配与 workspace 随 batch 线性膨胀。
- Easy / Inference 给 Megatron/vLLM 风格调度器提供更少样板、可预分配的热路径。

### 文件一览

| 文件 | 用途 | 为什么要有 | 库化判断 |
|---|---|---|---|
| `inc_dc_framework_c_api.h` / `.cpp` | 底层稳定 C ABI：context、plan、异步 request、workspace、backend vtable、Dispatch/Combine enqueue | 公共 ABI 核心；融合算子与框架对接入口 | **否，不可删** |
| `inc_dc_easy_api.h` / `.cpp` | communicator + token plan 的简化 facade；异步 D/C | 减少拼 framework 描述符；Inference 依赖 | 可选上层，当前交付保留 |
| `inc_dc_inference_api.h` / `.cpp` | session/plan 预分配，热路径尽量无分配；同 plan 可 D∥C | 推理 scheduler 的稳定接入边界 | 适配层，当前交付保留 |
| `inc_dc_chunk_planner.h` / `.cpp` | 将任意大 `logical_rows` 分页为有界 chunk | 可扩展性原语；workspace 不随总行数线性涨 | **否，不可删** |
| `inc_dc_dispatch_route_plan.h` / `.cpp` | 编译每个 Dispatch chunk 的 route / cell 容量 | Dispatch backend 依赖；可 query 再 compile | **否，不可删** |

依赖关系：`Inference → Easy → Framework → backend vtable`；planner 被 Framework / backend 在 host 侧调用。  
单 INC 真机 provider 注入时，vtable 通常是 **`native_composite_backend`**（内部再分发到 dispatch/combine），而不是 Framework 直接各挂一个裸 kernel。

首次接入建议使用 `inc_dc_inference_api.h`，并从
[`../examples/inference_api/inc_dc_inference_api_example.cpp`](../examples/inference_api/inc_dc_inference_api_example.cpp)
复制完整生命周期。

---

## English

### What is this directory?

The public **C integration surface** and **host-only planners**.  
Minimum library core = Framework C API + both planners; Easy/Inference are thinner facades.

### Why this layer exists

- No C++ objects, SHMEM ranks, INC owners, or `aclrt` headers in the ABI—so
  frameworks can bind from pure C / FFI.
- Planners turn arbitrarily large logical inputs into bounded device epochs,
  avoiding hot-path allocation and linear workspace growth.
- Easy/Inference give Megatron/vLLM-style schedulers a smaller, preparable surface.

### Files

| File | Purpose | Why it exists | Removable? |
|---|---|---|---|
| `inc_dc_framework_c_api.h` / `.cpp` | Low-level C ABI: context, plan, async requests, workspaces, backend vtable | Public ABI core | **No** |
| `inc_dc_easy_api.h` / `.cpp` | Communicator + token-plan facade | Less descriptor boilerplate; used by Inference | Optional layer; retain now |
| `inc_dc_inference_api.h` / `.cpp` | Prepared session/plan; allocation-free hot path; D∥C on one plan | Inference scheduler boundary | Adapter; retain now |
| `inc_dc_chunk_planner.h` / `.cpp` | Page large `logical_rows` into bounded chunks | Scalability primitive | **No** |
| `inc_dc_dispatch_route_plan.h` / `.cpp` | Compile per-chunk Dispatch routes/capacities | Backend dependency | **No** |

Dependency: `Inference → Easy → Framework → backend vtable`. For single-INC,
the vtable is usually a **`native_composite_backend`** that fans out to the
dispatch/combine providers. Planners are host-side helpers.
