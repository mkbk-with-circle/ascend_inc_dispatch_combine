# Single-INC launchers / 单 INC 启动脚本

## 中文

### 这个目录是干什么的？

单 INC 真机 case 的 **唯一推荐入口**：公共函数库 + Dispatch case + Combine dyn-CSR case + operator sweep。

### 为什么要有这些脚本？

- 统一路径、日志、锁与空闲检查，避免各人私有 shell 分叉。
- sweep 把 worker/top-k/数据量矩阵变成可汇总的 gate，而不是零散手工命令。

### 文件一览

| 文件 | 用途 | 为什么要有 |
|---|---|---|
| `inc_single_inc_common.sh` | 路径、锁、NPU 空闲检查、拓扑/profile、日志公共函数 | 所有 launcher 的共享地基；改门禁只改一处 |
| `run_single_inc_stream_dispatch_case.sh` | 启动一个 Dispatch W2/W4/W8 case 并检查所有 rank | 正式 Dispatch 单点门禁 |
| `run_single_inc_dyn_case.sh` | 启动一个 Combine dynamic-CSR case 并检查结果 | 正式 Combine dyn-CSR 单点门禁 |
| `run_single_inc_operator_sweep.sh` | 组合 worker/top-k/数据量矩阵并汇总 gate | 批量回归与发布前矩阵 |
| `run_single_inc_overlap_sweep.py` | 两个独立 SHMEM session 的 solo/同时/错峰交叠验证 | 证明真实设备区间交叠并保留逐 rank 证据 |

资格化 / 正式门禁应走这些 launcher；裸跑 `$BUILD/bin/inc_dc_*` 会绕过空闲锁与拓扑保护，结果不作交付证据。

**二进制名提醒：** Combine exe 叫 `inc_dc_sv2_dyn_csr_combine`，但源是  
`single_inc/combine/inc_dc_combine_launcher.cpp`（见 combine/README）。

**当前矩阵跑到哪了、环境与 baseline/gate 怎么定义：**  
见 [`../../single_inc/SWEEP_STATUS.md`](../../single_inc/SWEEP_STATUS.md)（权威详表仍在 `docs/inc/report/`）。

---

## English

### What is this directory?

The **recommended entry points** for single-INC device cases: shared helpers,
Dispatch case, Combine dyn-CSR case, and operator sweep.

### Why these scripts exist

- Centralize paths, locks, idle checks, and logging so private shells do not fork.
- Turn worker/top-k/size matrices into summarizable gates.

### Files

| File | Purpose | Why it exists |
|---|---|---|
| `inc_single_inc_common.sh` | Shared paths, locks, idle-NPU checks, topology/profile, logging | One place to change launch policy |
| `run_single_inc_stream_dispatch_case.sh` | One Dispatch W2/W4/W8 case + all-rank checks | Formal Dispatch gate |
| `run_single_inc_dyn_case.sh` | One dynamic-CSR Combine case + result checks | Formal Combine dyn-CSR gate |
| `run_single_inc_operator_sweep.sh` | Worker/top-k/size matrix + gate summary | Batch regression before release |
| `run_single_inc_overlap_sweep.py` | Solo/simultaneous/staggered overlap across independent SHMEM sessions | Prove device-interval overlap with per-rank evidence |

Use these launchers for qualification; raw `$BUILD/bin/inc_dc_*` bypasses
idle/topology gates. Combine exe name is still `inc_dc_sv2_dyn_csr_combine`
while the source is `inc_dc_combine_launcher.cpp`.

**Where are we on the sweep matrix (env + baseline/gates)?**  
See [`../../single_inc/SWEEP_STATUS.md`](../../single_inc/SWEEP_STATUS.md).
