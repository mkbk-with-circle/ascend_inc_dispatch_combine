# Host regression tests / 宿主回归测试

## 中文

### 这个目录是干什么的？

**不依赖真机 NPU** 的 host 侧回归：公共 C ABI、Easy/Inference、planner、单 INC
plan/backend/layout。测试 **不进入运行库**，但是交付门禁，不建议从源码仓删除。

### 为什么要有 host 测试（而不只靠真机 sweep）？

- ABI/状态机/溢出边界可以在秒级反馈，不必占集群。
- 纯 C11 header 门禁防止公开头文件漂成 C++-only。
- 与 `scripts/` 真机门禁互补：这里挡逻辑，那边挡拓扑与性能。

### 子目录

| 子目录 | 用途 | 为什么要有 |
|---|---|---|
| `common/` | SingleInc facade + Framework/Easy/Inference + 共享 planner | 公共层交付门禁 |
| `single_inc/` | plan compiler、expert layout、composite backend | 单 INC provider 交付门禁 |

---

## English

### What is this directory?

**Host-side regressions** that do not require live NPUs: public C ABI,
Easy/Inference, planners, and single-INC plan/backend/layout logic. Tests are
not runtime code, but they are delivery gates and should stay in source.

### Why host tests besides device sweeps?

- ABI/state-machine/overflow bugs get second-scale feedback without a cluster.
- Pure-C11 header gates stop public headers from becoming C++-only.
- Complements `scripts/` device gates: logic here, topology/perf there.

### Subdirectories

| Subdirectory | Role | Why it exists |
|---|---|---|
| `common/` | SingleInc facade + Framework/Easy/Inference APIs + shared planners | Common-layer delivery gate |
| `single_inc/` | Plan compilers, expert layout, composite backend | Single-INC provider delivery gate |
