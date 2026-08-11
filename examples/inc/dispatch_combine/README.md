# INC Dispatch / Combine

## 中文

### 这个目录是干什么的？

这里是当前维护的 **INC Dispatch/Combine（简称 INC DC）** 实现树。  
语义上对应 MoE 的 **Dispatch / Combine 两段通信**：

1. **Dispatch**：按路由把 token 从 worker 发送到专家所在位置（经 INC 中转）。
2. **Combine**：把各专家贡献收集并加权归约，再写回源 token 布局。

历史 AG/RS、阶段 gate、未进入产品闭包的实验实现已经移除。  
目录根部故意不放源码，避免「靠文件名前缀猜所有权」；所有权以物理路径为准。

**新人快速了解单 INC Dispatch/Combine 主逻辑（推荐首读）：**  
[`single_inc/QUICKSTART.md`](single_inc/QUICKSTART.md)  
**当前 sweep 进度 / 环境 / baseline：**  
[`single_inc/SWEEP_STATUS.md`](single_inc/SWEEP_STATUS.md)

### 为什么要单独成树？

- 把 **稳定公开 ABI**（`common/`）与 **具体拓扑实现**（`single_inc/`、`multi_inc/`）分开，框架只依赖前者。
- 把 **资格化脚本/测试** 与运行库分开，二进制安装包可以裁剪，但源码仓保留门禁。
- 单 INC（星型 W+1）与多 INC（destination-major pipeline）共享协议，不共享 runtime，避免错误耦合。

### 子目录

| 目录 | 用途 | 为什么要有 | 交付属性 |
|---|---|---|---|
| `common/` | 稳定 C API、共享协议、硬件策略、C 接入示例 | 框架与多实现共用的契约层；改 ABI 必须可控 | 生产必需 |
| `single_inc/` | 单 INC Dispatch、Combine、常驻服务、计划编译 | 当前正式星型产品数据面 | 生产必需 |
| `multi_inc/` | 多 INC destination-major Dispatch pipeline | 扩展拓扑；与单 INC 并行维护 | 生产必需 |
| `scripts/` | 真机 case launcher 与随机 token-plan sweep | 强制拓扑/空闲保护，避免手跑绕过门禁 | 资格化工具 |
| `tests/` | Host 侧 API / planner / backend 回归 | 交付门禁；不依赖真机也能挡 ABI 漂移 | 交付门禁 |

### 构建入口

正式构建入口在 `examples/inc/CMakeLists.txt`（由上层 `shmem` 构建系统拉入）：

```bash
# 在 shmem 的父目录执行；若已在 shmem/ 内则用 cmake -S .
cmake -S shmem -B build -DCMAKE_BUILD_TYPE=Release -DUSE_EXAMPLES=ON
cmake --build build -j --target \
  inc_dc_single_inc_stream inc_dc_sv2_dyn_csr_combine inc_dc_dn_pipeline
```

真机资格化请走 `scripts/single_inc/`（裸跑 bin 会绕过拓扑/空闲门禁，不作交付证据）。  
注意：exe/target 名可能仍带 `sv2`，**源文件以 CMake 列表中的稳定名为准**（见 `single_inc/combine/README.md`）。

---

## English

### What is this directory?

This tree is the maintained **INC Dispatch/Combine (INC DC)** implementation.

1. **Dispatch**: route tokens from workers toward expert locations (via INC).
2. **Combine**: gather/reduce expert contributions back into the source-token layout.

Historical AG/RS experiments, phase gates, and code outside the product
dependency closure have been removed. No sources live at this directory root;
physical paths define ownership.

**Newcomer quickstart for single-INC Dispatch/Combine (start here):**  
[`single_inc/QUICKSTART.md`](single_inc/QUICKSTART.md)  
**Sweep progress / environments / baseline gates:**  
[`single_inc/SWEEP_STATUS.md`](single_inc/SWEEP_STATUS.md)

### Why this layout?

- Keep a **stable public ABI** (`common/`) separate from topology-specific
  implementations (`single_inc/`, `multi_inc/`).
- Keep **qualification scripts/tests** out of the runtime library so packages
  can be trimmed without losing source gates.
- Share protocol/platform policy across single- and multi-INC without sharing
  their runtimes.

### Subdirectories

| Directory | Role | Why it exists | Delivery |
|---|---|---|---|
| `common/` | Stable C APIs, protocol, platform policy, C examples | Framework-facing contract shared by backends | Required runtime |
| `single_inc/` | Single-INC Dispatch/Combine/service/planning | Qualified star-topology product path | Required runtime |
| `multi_inc/` | Multi-INC destination-major Dispatch pipeline | Extended topology, maintained in parallel | Required runtime |
| `scripts/` | Device launchers and random token-plan sweeps | Enforce topology/idle guards | Qualification tooling |
| `tests/` | Host regressions for APIs, planners, backends | Delivery gates without requiring devices | Delivery gates |

Build entry is `examples/inc/CMakeLists.txt` (pulled in by the `shmem` build).
Qualify via `scripts/single_inc/`; raw binaries bypass topology/idle gates.
CMake **target/exe** names may still say `sv2`; compiled **sources** use stable
`inc_dc_combine_*` names—see `single_inc/combine/README.md`.
