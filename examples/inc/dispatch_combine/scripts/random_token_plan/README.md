# Random token-plan sweeps / 随机 Token Plan Sweep

## 中文

### 这个目录是干什么的？

在正式单 INC launcher 之上做 **大规模 / 随机 route** 覆盖。  
脚本本身 **不复制** 通信实现，只生成计划并调用 `../single_inc/`。

### 为什么要两套（seeded 矩阵 vs OS 随机）？

- seeded 可恢复矩阵：CI/断点续跑、对拍到固定上限（含大 workspace）。
- OS 真随机 campaign：抓「没人想到的」route/shape 组合，补确定性矩阵的盲区。

### 文件一览

| 文件 | 用途 | 为什么要有 |
|---|---|---|
| `run_single_inc_nb_random_sweep.py` | seeded、可恢复、覆盖到约 8 GiB 的确定性矩阵；支持 `--plan-only` | 可复现的大矩阵回归；可只生成计划 |
| `run_single_inc_os_random_campaign.py` | 每 case 用 `getrandom(2)` 生成独立 route/shape 的 campaign | 长尾鲁棒性；不进运行库安装包也可 |

---

## English

### What is this directory?

Large-scale / random-route coverage **on top of** the formal single-INC
launchers. Scripts do **not** duplicate the data path; they generate plans and
call `../single_inc/`.

### Why both seeded and OS-random?

- Seeded resumable matrices: CI/resume and fixed-ceiling coverage.
- OS-random campaigns: catch unanticipated route/shape combinations.

### Files

| File | Purpose | Why it exists |
|---|---|---|
| `run_single_inc_nb_random_sweep.py` | Seeded, resumable matrix (~8 GiB) with `--plan-only` | Reproducible large-matrix regression |
| `run_single_inc_os_random_campaign.py` | Per-case `getrandom(2)` route/shape campaign | Long-tail robustness |

Runtime-only packages may exclude these.
