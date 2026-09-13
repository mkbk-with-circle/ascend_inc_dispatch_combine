# Native runtime / 原生运行时

## 中文

### 这个目录是干什么的？

把最短 SingleInc API **接到真实 ACLSHMEM kernel**。
这里的 `native` 表示「原生/真实 backend」，**不是** naive 实现。

### 为什么要有 runtime 层？

- Host 入口不能直接 `#include` 设备 kernel；内部仍需 session、workspace、mailbox 和 generation rendezvous。
- Dispatch 与 Combine 各有一个 native backend，内部合成后由最短 API 使用。
- 常驻 INC service 把 INC 侧生命周期从「每次 launch 起停」里抽出来，降低调度抖动。
- expert layout adapter 把 rank-major 专家配置对齐到 grouped-GEMM 的 padded 布局，服务推理集成。

新接入不要在业务热路径手工调用 service mailbox。worker 初始化时构造
`NativeSingleIncWorkerControl{dispatch_session, service_client}`，再调用
`BindNativeSingleIncWorkerControl(&control, &single_inc_config)`；模型代码之后只走
`SingleInc::dispatch/combine`。

`inc_dc_native_full_example_main.cpp` 仍显式展开内部 service 步骤，因为它还要
在 PREPARE 与 enqueue 之间插入故障注入门禁；它是资格化 harness，不是推荐的业务模板。

### 文件一览

| 文件 | 用途 | 为什么要有 | 库化 |
|---|---|---|---|
| `inc_dc_native_dispatch_backend.h` / `.cpp` | Dispatch session、registered-buffer 快路径、backend vtable | Framework Dispatch op 的真实 provider | 必需 |
| `inc_dc_native_combine_backend.h` / `.cpp` | Combine session、producer/INC enqueue、backend vtable | Framework Combine op 的真实 provider | 必需 |
| `inc_dc_native_composite_backend.h` / `.cpp` | 把 D/C 两个 backend 合成一个内部 communicator | SingleInc 只持有一个 backend | 必需 |
| `inc_dc_native_inc_service.h` / `.cpp` | 常驻 INC proxy、worker mailbox、generation rendezvous | INC 侧常驻服务；跨 op 复用 | 必需 |
| `inc_dc_native_combine_workspace.h` / `.cpp` | 构建有界 Combine workspace、拷贝描述与 metadata | Combine enqueue 前的可复用缓冲 | 必需 |
| `inc_dc_native_expert_layout_adapter.h` / `.cpp` | rank-major expert ↔ grouped-GEMM padded layout | 推理侧专家张量布局桥接 | 推理必需 |
| `inc_dc_native_full_example_main.cpp` | 真 backend 的 Dispatch→expert→Combine 整链示例/门禁 | 集成冒烟；不进入库 | SDK/测试可执行文件 |

---

## English

### What is this directory?

Adapters that bind the minimal SingleInc API to **real ACLSHMEM kernels**.
Here `native` means a real provider—not a naive stub.

### Why a runtime layer?

- The host API must not include device kernels directly; it needs sessions,
  workspaces, mailboxes, and generation rendezvous.
- Separate Dispatch/Combine backends are joined internally for the minimal API.
- The resident INC service removes per-call INC process churn.
- The expert layout adapter bridges rank-major expert configs to padded
  grouped-GEMM layouts for inference.

### Files

| File | Purpose | Why it exists | Library |
|---|---|---|---|
| `inc_dc_native_dispatch_backend.*` | Dispatch session, registered-buffer fast path, vtable | Real Dispatch provider | Required |
| `inc_dc_native_combine_backend.*` | Combine session, producer/INC enqueue, vtable | Real Combine provider | Required |
| `inc_dc_native_composite_backend.*` | One internal communicator over both backends | Single backend for SingleInc | Required |
| `inc_dc_native_inc_service.*` | Resident INC proxy, mailbox, generation rendezvous | Persistent INC service | Required |
| `inc_dc_native_combine_workspace.*` | Bounded Combine workspace + metadata | Reusable pre-enqueue buffers | Required |
| `inc_dc_native_expert_layout_adapter.*` | Rank-major ↔ padded grouped-GEMM layout | Inference tensor bridge | Required for inference |
| `inc_dc_native_full_example_main.cpp` | Real-backend D→expert→C integration gate | Smoke/integration executable | SDK/test only |
