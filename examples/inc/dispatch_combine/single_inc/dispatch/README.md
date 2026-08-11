# Single-INC Dispatch / 单 INC Dispatch

## 中文

### 这个目录是干什么的？

单 INC **Dispatch 数据面**：host/device 共享 ABI、设备 kernel、standalone 资格化 launcher。  
跨算子复用的 MTE gather 原语在 `../../common/platform/inc_dc_gather_mte_aicore.h`，不在本目录重复实现。

资格化路径是 **`worker → INC → worker` push-only**（H1）。kernel 里若仍见
`kStreamFlagWorkerDirect`（bypass INC），资格化 launcher 固定关闭；交付勿启用。  
主逻辑见 [`../QUICKSTART.md`](../QUICKSTART.md) §2。

### 为什么要独立成目录？

- Dispatch 可单独扫带宽/正确性，不依赖 Combine 归约逻辑。
- ABI / kernel / main 三分开：库化时 `main` 可只做资格化可执行文件，ABI+kernel 留在运行时。

### 文件一览

| 文件 | 用途 | 为什么要有 |
|---|---|---|
| `inc_dc_single_inc_stream_abi.h` | generation、lane、workspace、telemetry 的 host/device 共享 ABI | 防止 host 填表与 device 解读错位；版本化契约 |
| `inc_dc_single_inc_stream_kernel.cpp` | worker upload、INC fan-out、completion 的设备 kernel | 真正的数据面；运行时必需 |
| `inc_dc_single_inc_stream_main.cpp` | ACLSHMEM 初始化、workspace 分配、启动、正确性与带宽输出 | standalone 资格化入口；库化时可移出运行库 |

三个文件都在当前正式 Dispatch 构建依赖中。

---

## English

### What is this directory?

The single-INC **Dispatch data path**: shared host/device ABI, device kernels, and
a standalone qualification launcher. Shared MTE gather lives in
`../../common/platform/inc_dc_gather_mte_aicore.h`.

Qualification is **push-only via INC** (H1). Do not enable `WorkerDirect`
bypass even if the kernel still contains the branch. See
[`../QUICKSTART.md`](../QUICKSTART.md) §2.

### Why a dedicated directory?

- Dispatch can be bandwidth/correctness-qualified without Combine reduction.
- Split ABI / kernel / main so packaging can keep ABI+kernel and ship `main`
  only as a qualification executable.

### Files

| File | Purpose | Why it exists |
|---|---|---|
| `inc_dc_single_inc_stream_abi.h` | Shared generation/lane/workspace/telemetry ABI | Host/device contract |
| `inc_dc_single_inc_stream_kernel.cpp` | Worker upload, INC fan-out, completion kernels | Runtime data path |
| `inc_dc_single_inc_stream_main.cpp` | ACLSHMEM setup, launch, correctness, bandwidth | Qualification entry |

All three are current build dependencies.
