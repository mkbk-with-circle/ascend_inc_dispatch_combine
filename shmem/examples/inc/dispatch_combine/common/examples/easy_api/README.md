# Easy API examples / Easy API 示例

## 中文

### 这个目录是干什么的？

演示如何用 **Easy API** 完成单 INC Dispatch/Combine 接入：从创建 communicator，
到构造 token plan，再到异步 enqueue/wait/release，最后给出整链 `full_example`。

### 为什么要拆成多个 `.c`？

- 每个文件对应一个可单独讲解/拷贝的生命周期阶段，避免「一个巨型 main」难读。
- CMake 可对每个文件做编译门禁；改某个 API 时失败点更清晰。
- 调用方在 request release 前必须保活 tensor / route / workspace——示例把这一点写进分步流程。

### 文件一览

| 文件 | 用途 | 为什么要有 |
|---|---|---|
| `single_inc_example.h` | 共享声明：allocator/copy callback、plan holder、公共辅助 | 避免五个示例 `.c` 复制粘贴同一套回调 |
| `init_example.c` | 创建或借用 communicator/context | 展示部署层如何注入 backend vtable |
| `token_plan_example.c` | 构造、校验、上传、释放 dense token plan | plan 是 Easy 路径的路由载体，必须单独正确 |
| `dispatch_example.c` | workspace query → Dispatch enqueue → wait → release | 展示异步 Dispatch 与 workspace 所有权 |
| `combine_example.c` | Combine enqueue → wait → release | 与 Dispatch 对称的异步 Combine |
| `full_example.c` | init→plan→Dispatch→expert 占位→Combine→destroy | 端到端抄作业入口 |

这些文件可不安装到运行时，但建议随 SDK 源码保留。

---

## English

### What is this directory?

Stepwise samples for the **Easy API**: create a communicator, build a token
plan, async enqueue/wait/release for Dispatch and Combine, plus a full chain.

### Why multiple `.c` files?

- Each file is one copyable lifecycle stage instead of one unreadable mega-main.
- CMake can fail at the exact stage whose API drifted.
- Callers must keep tensors/routes/workspaces alive until request release; the
  stepwise flow makes that ownership explicit.

### Files

| File | Purpose | Why it exists |
|---|---|---|
| `single_inc_example.h` | Shared decls: alloc/copy callbacks, plan holder | Avoid duplicating helpers across the five `.c` samples |
| `init_example.c` | Create/borrow communicator/context | Show backend vtable injection |
| `token_plan_example.c` | Build/validate/upload/release dense token plan | Plan is the Easy routing carrier |
| `dispatch_example.c` | Workspace query + Dispatch enqueue/wait/release | Async Dispatch ownership |
| `combine_example.c` | Combine enqueue/wait/release | Symmetric Combine path |
| `full_example.c` | Full init→plan→D→expert stub→C→destroy | End-to-end copy-paste entry |

Runtime packages need not install these; keep them in an SDK source package.
