# Single INC / 单 INC

> **快速了解主逻辑（新人首读）→ [`QUICKSTART.md`](QUICKSTART.md)**  
> **当前 sweep 进度 / 环境 / baseline 口径 → [`SWEEP_STATUS.md`](SWEEP_STATUS.md)**  
> **最短 API 完整示例 → [`../common/examples/single_inc_api/`](../common/examples/single_inc_api/README.md)**
> Quick mental model → [`QUICKSTART.md`](QUICKSTART.md) · Progress board → [`SWEEP_STATUS.md`](SWEEP_STATUS.md)

## 中文

### 这个目录是干什么的？

**单 INC 星型拓扑** 产品实现：`W` 个 worker + `1` 个 INC。

- Dispatch：worker 上传 → INC fan-out → 目标 worker 收齐。
- Combine：各 worker 贡献 → INC 归约 → 结果 fan-back。

### 为什么使用单 INC？

- 星型拓扑是当前资格化最完整、框架对接最清晰的路径（常驻 INC service + native backend）。

### 子目录

| 子目录 | 用途 | 为什么要有 |
|---|---|---|
| `dispatch/` | 正式 Dispatch 数据面 + standalone launcher | 独立资格化 Dispatch，不绑 Combine |
| `combine/` | 正式 Combine 数据面、拓扑、计划、sidecar | Combine 比 Dispatch 更重（归约/谱系/活性） |
| `planning/` | 框架 route → 有界设备 workspace 的编译器 | 保证 D/C 正反向语义一致，热路径零分配 |
| `runtime/` | Framework backend vtable + 常驻 INC service | 公共 API 获得真实 ACLSHMEM 能力的 provider |
| `tools/` | 随机 token plan 生成器 | sweep/鲁棒性；不进运行库 |

真机资格化 / 正式门禁 **要求** 使用 `../scripts/single_inc/` launcher（拓扑、空闲锁、多 rank 结果）。  
裸跑 bin 会绕过这些检查，结果不作交付证据。进度见 [`SWEEP_STATUS.md`](SWEEP_STATUS.md)。

---

## English

### What is this directory?

The **single-INC star topology** product: `W` workers + `1` INC.

- Dispatch: worker upload → INC fan-out → destination workers.
- Combine: contributions → INC reduce → result fan-back.

### Why single-INC

- It is the most fully qualified path for framework integration (resident INC
  service + native backends).

### Subdirectories

| Subdirectory | Role | Why it exists |
|---|---|---|
| `dispatch/` | Qualified Dispatch data path + launcher | Qualify Dispatch independently of Combine |
| `combine/` | Combine data path, topology, plans, sidecars | Heavier path (reduce/lineage/liveness) |
| `planning/` | Framework route → bounded device workspace | Forward/reverse fidelity; no hot-path alloc |
| `runtime/` | Framework backend vtable + resident INC service | Real ACLSHMEM provider behind public APIs |
| `tools/` | Random token-plan generator | Sweep/robustness; not runtime |

Qualification requires `../scripts/single_inc/` launchers; raw binaries bypass
topology/idle gates and are not delivery evidence. Progress: [`SWEEP_STATUS.md`](SWEEP_STATUS.md).
