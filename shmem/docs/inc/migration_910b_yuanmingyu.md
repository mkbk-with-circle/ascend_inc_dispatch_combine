# ascend-样机 → 910B (yuanmingyu) 迁移文档

> 实施日期：2026-07-08  
> 活跃 profile：`910b-yuanmingyu`  
> 历史归档：`legacy-910c`（只读，不得作 910B pass_line 分母）

---

## 1. 环境对照

| 项 | legacy-910c（旧样机归档） | 910b-yuanmingyu（新环境） |
|----|---------------------------|---------------------------|
| 主机 | `hostname-mspqn.foreman.pxe` | `probing-real-worker-0`（K8s worker 容器） |
| NPU | 8×910B，`IT22HMDA_4_S`，16 PE | 8×910B，`IT22HMDA_1_S`，16 PE |
| PCI | `0x19E5:0xD803` | 同左 |
| HBM | 64 GiB / die | 64 GiB / die |
| CANN | 9.0.0 | **8.5.0**（`/usr/local/Ascend/cann-8.5.0`） |
| 驱动 | 25.5.2 | 25.3.rc1.2 |
| 部署 | 裸机/PXE | 共享 Pod（overlay 14T） |
| 数据目录 | `hardware_profiles/legacy-910c/` | `hardware_profiles/910b-yuanmingyu/` |

命名说明：`legacy-910c` 为项目约定文件夹名；旧样机硬件实测亦为 **910B**，非 SuperPod 910C 模组。

---

## 2. 账号与 SSH

| 用途 | Host | 用户 | 说明 |
|------|------|------|------|
| 日常开发 | `yuanmingyu-ymy` | `ymy` | 隔离工作区，推荐 |
| 排障 | `yuanmingyu-root` / `yuanmingyu` | `root` | 装包、建用户 |

连接经 PJLab HTTPS 网关 CONNECT（`lab-gw-client/ssh-connect-proxy.py`），不依赖 Clash TUN 做 SSH。

```bash
ssh yuanmingyu-ymy
cd ~/work/ascend-样机/shmem
source scripts/inc_env_910b.sh
```

密钥：`~/.ssh/weibozhen.p`（管理员发放）

---

## 3. 隔离与防污染 checklist

- [x] 工作目录仅 `/home/ymy/work/ascend-样机`（不在 `/root` build）
- [x] Gate 结果写入 `hardware_profiles/910b-yuanmingyu/`，不覆盖 `legacy-910c`
- [x] `ASCEND_HOME_PATH` 显式指定，未修改系统 CANN 8.5 安装
- [x] CANN 9.0 **仅用户目录** `/home/ymy/Ascend/cann-9.0.0`（`install_cann90_user.sh`）
- [x] **默认仍用系统 8.5**；仅 `INC_USE_CANN_90=1` 或 `inc_env_910b_cann90.sh` 时切 9.0
- [x] **不修改** `~/.bashrc`、不 symlink `/usr/local/Ascend/cann`
- [ ] Pod 重启后需 `sync-yuanmingyu.sh push` 恢复代码
- [ ] 跑 gate 前 `npu-smi info` 确认无他人占满 16 PE

---

## 4. 同步工作流

本地为真源（git + rsync）。

```bash
# 本地 ascend-样机 目录
bash scripts/sync-yuanmingyu.sh push      # 代码 → 远程
bash scripts/sync-yuanmingyu.sh pull    # gate 结果 ← 远程
bash scripts/sync-yuanmingyu.sh bandwidth-test  # 100MB 上下行测速
```

远程路径：`/home/ymy/work/ascend-样机`

---

## 5. CANN 策略（已执行阶梯 B）

| 阶梯 | 状态 |
|------|------|
| A 用户目录 CANN 9.0 | `bash scripts/install_cann90_user.sh` → `/home/ymy/Ascend/` |
| **B 系统 CANN 8.5.0** | **默认**（`source scripts/inc_env_910b.sh`） |
| C 显式 9.0 会话 | `source scripts/inc_env_910b_cann90.sh` 或 `INC_USE_CANN_90=1` |

环境入口：

```bash
source docs/inc/configs/910b-yuanmingyu.env   # INC_HW_PROFILE, ASCEND_HOME_PATH, INC_P5_BASELINE
source scripts/inc_env_910b.sh                # set_env + LD_LIBRARY_PATH
bash scripts/build.sh -examples
```

远程额外安装（root 一次性）：`cmake`、`rsync`

---

## 6. Gate 无损迁移状态（2026-07-09 闭合）

汇总见 [`hardware_profiles/910b-yuanmingyu/gates/migration_status.json`](hardware_profiles/910b-yuanmingyu/gates/migration_status.json) 与 [`MIGRATION_COMPLETE.json`](hardware_profiles/910b-yuanmingyu/gates/MIGRATION_COMPLETE.json)。

| 指标 | 值 |
|------|-----|
| Legacy formal 目标 | **43** gate（排除 5 条 legacy 未过项） |
| 910B CANN 8.5 通过 | **40** |
| CANN 9.0 补跑通过 | **4**（c05d, c08 roofline, p1, c07 聚合） |
| 迁移闭合 | **是**（`migration_complete: true`，43/43） |

### 6.1 P5 baseline（910B 本地分母）

| 项 | 值 |
|----|-----|
| 路径 | `hardware_profiles/910b-yuanmingyu/p5/p5_native_shmem_combine_baseline.json` |
| 方法 | **inc_dc_roofline fallback**（native SHMEM 因 `halMemExportToShareableHandleV2` 缺失无法跑通） |
| `combine_g8_h4096_k1` comm | **0.858 GB/s**（roofline reduce stage） |
| `stage_goal` (÷8) | **0.107 GB/s** |
| `pass_line` (×0.9) | **0.096 GB/s** |

入口脚本：`scripts/run_p5_baseline_910b.sh`（自动 fallback → `run_p5_baseline_inc_dc_910b.sh`）

### 6.2 Gate 批次执行（CANN 8.5）

```bash
source scripts/inc_env_910b.sh
bash scripts/build_gate_targets_910b.sh
bash scripts/run_p5_baseline_910b.sh
bash scripts/run_gate_910b.sh batch B1   # 正确性
bash scripts/run_gate_910b.sh batch B3   # dispatch infra
bash scripts/run_gate_910b.sh batch B5   # P0–P5 子 gate
bash scripts/run_gate_910b.sh batch B2   # combine 带宽
bash scripts/run_gate_910b.sh batch B4   # d03 v4 带宽 ladder
# 或一键：bash scripts/run_all_gates_910b.sh
```

### 6.3 CANN 9.0 补跑（已完成）

| Gate | 结果 |
|------|------|
| `c05d_precision_align` | PASS（CANN 9.0） |
| `c08_combine_stage_roofline` | PASS（CANN 9.0） |
| `p1_dc_device_stage_roofline` | PASS（CANN 9.0） |
| `c07_combine_freeze` | PASS（聚合子 gate；`c08f_poll_prefill` 为可选排除项） |

```bash
export INC_USE_CANN_90=1 INC_GATE_TIMEOUT=600
source scripts/inc_env_910b_cann90.sh
bash scripts/run_gate_retry_cann90.sh
```

### 6.4 已通过代表项（910B）

Combine 带宽链：**c08a–c08e, c10–c15b, c12a–c12c**；Dispatch infra：**d02, d03 contract/visibility, d03_bw_v4**；**d03_dc_ll_bandwidth_gate_v4**（v4 ladder）；P5 子集：**p2–p5, p5c1/c2, p5d0, d06, p4**。

---

## 7. 可移植性冒烟（2026-07-08 早期）

汇总见 [`hardware_profiles/910b-yuanmingyu/gates/portability_smoke_summary.json`](hardware_profiles/910b-yuanmingyu/gates/portability_smoke_summary.json)

| 步骤 | 结果 | 备注 |
|------|------|------|
| S00 env check | PASS | `snapshots/s00_env_check.json` |
| SHMEM smoke | PASS | notifywait_8pe, perftest_put, allgather_16pe |
| inc_dc_d03_tests | PASS | host |
| inc_dc_combine_tests | PASS | host |
| inc_dc_reference_tests | PASS | host |
| C00 combine semantic | PASS | host contract |
| **C10 combine LL persistent** | **PASS** | device_correctness + pipeline |
| D02 dispatch barrier | PASS | bringup_only |
| **D03 dispatch pipeline** | **FAIL** | legacy 亦 FAIL，非回归目标 |

---

## 8. 故障排查

| 现象 | 处理 |
|------|------|
| `/root/.bashrc: setenv.bash: No such file or directory` | 容器内 driver 路径不完整；用 `inc_env_910b.sh`（`set +u` 再 source CANN） |
| `cmake: command not found` | root 执行 `dnf install -y cmake` |
| `rsync: command not found` | root 执行 `dnf install -y rsync` |
| SSH `Connection closed` 不经 ProxyCommand | 必须用 `ProxyCommand` 走网关 CONNECT |
| D03 pipeline fail + NPU 被占 | `scripts/inc_npu_idle.sh` + `npu-smi`；device/bw 仅空闲窗口跑 |
| Native P5 baseline OOM / nullptr | driver 缺 `halMemExportToShareableHandleV2`；用 inc_dc roofline fallback |
| Gate v5 脚本默认 CANN 9.0 | B4 已改 `run_d03_dc_ll_final_ladder_v4.sh` |
| Pod 重启丢 build | 本地 push 后远程重编 |

---

## 9. 工具脚本

| 脚本 | 说明 |
|------|------|
| `scripts/run_gate_910b.sh` | Gate wrapper + batch B1–B5 |
| `scripts/compare_gate_manifest.py` | legacy vs 910B manifest |
| `scripts/build_gate_targets_910b.sh` | 仅编 gate 相关 target |
| `scripts/run_p5_baseline_910b.sh` | P5 baseline（含 inc_dc fallback） |
| `scripts/run_all_gates_910b.sh` | 一键全批 gate |
| `scripts/run_gate_retry_cann90.sh` | CANN 9.0 失败项补跑 |

---

## 10. 相关文件

| 文件 | 说明 |
|------|------|
| `docs/inc/configs/910b-yuanmingyu.env` | profile 环境变量 |
| `scripts/inc_env_910b.sh` | 远程 build/gate 入口 |
| `scripts/run_gate_retry_cann90.sh` | CANN 9.0 失败项补跑 |
| `scripts/sync-yuanmingyu.sh` | 本地 ↔ 远程同步 |
| `docs/inc/hardware_profiles/` | 多机器 gate/baseline 根目录 |
| `docs/inc/report/ACTIVE_HW_PROFILE.md` | 当前活跃 profile 指针 |
