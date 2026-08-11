# Multi-INC transport / 多 INC 传输层

## 中文

### 这个目录是干什么的？

多 INC pipeline 的 **传输契约**：ingress transport 结构、byte-aware queue-v2、
device 发布原语、独立 ingress kernel。

### 为什么要和 `pipeline/` 分开？

- 队列/ingress 可独立演进与单测，不必每次改 pipeline 调度。
- 四个文件构成同一契约，不能删其中一个却假装 pipeline 仍完整。
- 历史名 `dn` 保留是为了与已落盘报告/符号对齐；重命名必须 ABI 同步。

### 文件一览

| 文件 | 用途 | 为什么要有 |
|---|---|---|
| `inc_dc_dn_transport_abi.h` | ingress transport 的 host/device 共享结构与容量 | transport 顶层 ABI |
| `inc_dc_dn_queue_v2_abi.h` | byte-aware queue-v2 ring、counter、payload ABI | 精确到字节的队列布局 |
| `inc_dc_dn_queue_v2_device.h` | DCCI、tail/head publish、P6 transport device 原语 | device 侧队列操作实现 |
| `inc_dc_dn_ingress_transport_kernel.cpp` | 独立 ingress transport kernel 与跨 AIV 汇总 | 可单独资格化的 ingress 路径 |

---

## English

### What is this directory?

The multi-INC pipeline **transport contract**: ingress structures, byte-aware
queue-v2, device publish primitives, and a standalone ingress kernel.

### Why separate from `pipeline/`?

- Queues/ingress can evolve and be tested without touching pipeline scheduling.
- All four files form one contract; deleting any one breaks the pipeline.
- The historical `dn` name stays for report/symbol continuity; rename only with
  synchronized ABI migration.

### Files

| File | Purpose | Why it exists |
|---|---|---|
| `inc_dc_dn_transport_abi.h` | Shared ingress-transport structures/capacities | Top-level transport ABI |
| `inc_dc_dn_queue_v2_abi.h` | Byte-aware queue-v2 rings/counters/payloads | Precise queue layout |
| `inc_dc_dn_queue_v2_device.h` | DCCI, tail/head publish, P6 device primitives | Device-side queue ops |
| `inc_dc_dn_ingress_transport_kernel.cpp` | Ingress kernel + cross-AIV aggregation | Standalone-qualifiable ingress path |
