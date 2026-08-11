# Multi-INC pipeline / 多 INC Pipeline

## 中文

### 这个目录是干什么的？

destination-major **两跳 Dispatch pipeline** 的完整实现：ABI、device 原语、
persistent kernel、host workspace pool、standalone launcher。

### 为什么拆成多个头文件 + kernel + main？

- ABI 必须 host/device 共享且版本化；device 原语按「通用 publish」与「完整 Dispatch」分层，避免一个巨型头。
- `source_major` 原语服务 ingress/staging；`full_dispatch` 覆盖 count/prefix/route gather。
- workspace pool 按 shape bucket 复用，避免每次 case 重新分配导致资格化抖动。
- `main` 只做资格化，可与库化数据面分离。

### 文件一览

| 文件 | 用途 | 为什么要有 |
|---|---|---|
| `inc_dc_dn_pipeline_abi.h` | destination-major 两跳 pipeline 的 host/device workspace ABI | 跨 host/device 的布局契约 |
| `inc_dc_dn_pipeline_device.h` | publish、reclaim、channel 辅助原语 | INC/worker 侧通用流水原语 |
| `inc_dc_dn_pipeline_full_dispatch_device.h` | count/prefix/route gather 的完整 Dispatch 原语 | 完整 Dispatch 语义，不塞进通用头 |
| `inc_dc_dn_pipeline_source_major_device.h` | source-major ingress 与 staging SPSC 原语 | 上行/暂存路径与 destination-major 主路径配合 |
| `inc_dc_dn_pipeline_workspace_pool.h` | host 侧 grow-only、按 shape bucket 复用的 workspace pool | 降分配抖动；资格化可复现 |
| `inc_dc_dn_pipeline_kernel.cpp` | worker upload、INC ingress/egress 的 persistent kernel | 多 INC 数据面本体 |
| `inc_dc_dn_pipeline_main.cpp` | standalone 多 INC 资格化 launcher | 真机入口；库化时可只作工具 |

除 `main.cpp` 可只作为工具安装外，其余均为当前多 INC 数据面依赖。  
历史前缀 `dn` 日后可同步改名为 `multi_inc`，但必须与 kernel ABI 一起迁移。

---

## English

### What is this directory?

The full **destination-major two-hop Dispatch pipeline**: ABI, device primitives,
persistent kernels, host workspace pool, and a standalone launcher.

### Why this file split?

- ABI must be shared and versioned; device primitives are layered (generic
  publish vs full Dispatch) to avoid one mega-header.
- Source-major primitives cover ingress/staging alongside the destination-major
  main path.
- A grow-only workspace pool keyed by shape buckets removes allocation jitter.
- `main` stays qualification-only for packaging.

### Files

| File | Purpose | Why it exists |
|---|---|---|
| `inc_dc_dn_pipeline_abi.h` | Host/device workspace ABI for the two-hop pipeline | Cross-side layout contract |
| `inc_dc_dn_pipeline_device.h` | Publish/reclaim/channel helpers | Shared pipeline primitives |
| `inc_dc_dn_pipeline_full_dispatch_device.h` | Count/prefix/route-gather Dispatch primitives | Full Dispatch semantics |
| `inc_dc_dn_pipeline_source_major_device.h` | Source-major ingress + staging SPSC | Uplink/staging companion path |
| `inc_dc_dn_pipeline_workspace_pool.h` | Grow-only host pool by shape bucket | Stable qualification allocations |
| `inc_dc_dn_pipeline_kernel.cpp` | Persistent worker upload + INC ingress/egress | Data path |
| `inc_dc_dn_pipeline_main.cpp` | Standalone multi-INC launcher | Qualification entry |

Only `main.cpp` is tool-only. The historical `dn` prefix may later become
`multi_inc` via a synchronized ABI-neutral rename.
