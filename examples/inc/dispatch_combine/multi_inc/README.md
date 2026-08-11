# Multi INC / 多 INC

## 中文

### 这个目录是干什么的？

当前维护的 **多 INC destination-major persistent Dispatch pipeline**：  
两跳 `worker → INC → worker`，按目的地组织，而不是单 INC 星型拓扑下的 fan-out。

### 为什么要单独成树（不复用 single_inc/runtime）？

- 拓扑与同步模型不同：多 INC 需要队列/ingress transport 与 workspace pool。
- 与单 INC **共享** `../common/` 的协议与平台策略，**不共享** 单 INC runtime，避免错误耦合。
- 历史多 INC campaign 脚本已移除；集群资格化应由工作负载 launcher 驱动。

### 子目录

| 子目录 | 用途 | 为什么要有 |
|---|---|---|
| `pipeline/` | 完整两跳 pipeline、device 原语、workspace pool、launcher | 多 INC Dispatch 产品数据面 |
| `transport/` | 共享队列 ABI、device 队列原语、ingress transport kernel | pipeline 依赖的传输契约；可单独演进 |

---

## English

### What is this directory?

The maintained **multi-INC destination-major persistent Dispatch pipeline**:
two-hop `worker → INC → worker`, organized by destination rather than a single
star INC.

### Why a separate tree from single_inc/runtime?

- Different topology and sync model: queues, ingress transport, workspace pools.
- Shares `../common/` protocol/platform policy but **not** the single-INC runtime.
- Historical campaign scripts are gone; cluster qualification should be driven
  by workload launchers.

### Subdirectories

| Subdirectory | Role | Why it exists |
|---|---|---|
| `pipeline/` | Full two-hop pipeline, device primitives, workspace pool, launcher | Multi-INC Dispatch data path |
| `transport/` | Queue ABI, device queue primitives, ingress kernel | Transport contract the pipeline depends on |
