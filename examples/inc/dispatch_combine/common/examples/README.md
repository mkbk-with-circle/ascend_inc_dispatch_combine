# C examples / C 接入示例

## 中文

### 这个目录是干什么的？

给框架集成者看的 **纯 C 接入示例**。  
示例 **不链接进运行库**；CMake 用 C11 编译它们，作为公开头文件的兼容性门禁。

### 为什么要有？

- 公开 API 声明「支持 C」，就必须有纯 `.c` 文件持续编译，否则头文件会在不知不觉中依赖 C++。
- 把「怎么用」拆成 Easy 分步示例与 Inference 热路径示例，对应两类真实接入方式。

### 子目录

| 子目录 | 用途 | 为什么要有 |
|---|---|---|
| `easy_api/` | 分步骤：init → token plan → Dispatch → Combine → full | 教学与 SDK 源码包；覆盖 Easy 全生命周期 |
| `inference_api/` | 预分配 session/plan 后的异步热路径 | 对应推理 scheduler 的真实用法 |

---

## English

### What is this directory?

Pure-C integration samples for framework authors.  
They are **not** linked into the runtime library; CMake compile-checks them as
C11 API compatibility gates.

### Why it exists

- Claiming “C ABI” without a continuously compiled `.c` sample lets headers
  quietly become C++-only.
- Easy stepwise samples and Inference hot-path samples match the two real
  integration styles.

### Subdirectories

| Subdirectory | Role | Why it exists |
|---|---|---|
| `easy_api/` | Stepwise init → plan → Dispatch → Combine → full | Teaching / SDK source package |
| `inference_api/` | Prepared session/plan async hot path | Matches inference schedulers |
