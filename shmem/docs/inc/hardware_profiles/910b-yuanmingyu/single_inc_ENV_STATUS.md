# 单 INC — 环境状态：910b-yuanmingyu

> **本环境私有。** 硬约束 / gate 公式 / 全局峰值榜 / 开放队列 →  
> [`../../report/single_inc_LIVE_STATUS.md`](../../report/single_inc_LIVE_STATUS.md)  
> 新机请克隆 [`../../report/env/ENV_STATUS_TEMPLATE.md`](../../report/env/ENV_STATUS_TEMPLATE.md)，**不要**改写别的环境的本文件。

| 字段 | 值 |
|:---|:---|
| profile 名 | `910b-yuanmingyu` |
| 上次更新 | 2026-08-06 |
| 更新者 | Codex（回写未过 performance gate 清单） |
| 对应共享文档 | `docs/inc/report/single_inc_LIVE_STATUS.md` |

---

## 1. 本机身份与软件（含 CANN 多版本）

| 项 | 值 |
|:---|:---|
| 主机 / SSH | `tmp-worker-0` / `yuanmingyu` · `yuanmingyu-direct`（历史亦称 probing-real-worker-0） |
| 代码根 | `/root/work/ascend-样机/shmem`（亦有 `/home/ymy/work/ascend-样机` 隔离区） |
| NPU / 型号 | 8× Ascend910B（`npu-smi` 见 16 逻辑 die / Phy 0–15） |
| **CANN（本环境可选，互不覆盖系统）** | 系统 **8.5.0**：`/usr/local/Ascend/cann-8.5.0`。已资格化复测 **9.1.0-beta.1**：`/opt/Ascend-9.1/cann-9.1.0-beta.1`。**9.0.0 在当前驱动上所有 rank 均 aclshmem 初始化失败，不可比。** |
| 离线包目录 | `cann-packages/{8.5.0,9.0.0,9.1.0-beta.1}/`（aarch64 toolkit+910b-ops；9.1 另含 ops-legacy） |
| 当前 ACTIVE 资格化 CANN | **9.1.0-beta.1**；新 shell 必须显式 source `/opt/Ascend-9.1/cann-9.1.0-beta.1/set_env.sh`，不得因系统默认 8.5 而误以为在跑 9.1 |
| 驱动 | npu-smi **25.3.rc1.2** |
| env 配置 | `docs/inc/configs/910b-yuanmingyu.env`（内含 8.5 / 9.0 选择逻辑） |

> **与共享文档的关系**：CANN/驱动版本只写在本 ENV，不写进 shared LIVE_STATUS。换机或换 CANN 版本时只更新本节，并在 §4 复测；不要把 8.5 上的带宽数字当成 9.0 已验证。

---

## 2. 本机拓扑与 AIV

| 项 | 值 |
|:---|:---|
| Live AIV 总数 | **48**（旧机 dense / overlap 时期；改机后须重查） |
| INC rank / Phy | 默认 INC=rank W → **Phy0**（须 live topo 复核） |
| Worker Phy 集合 | 典型走 `HCCS_SW` worker 集合（见 profile / 运行配置） |
| 禁用链路 | 曾禁 **Phy1 / SIO** |
| 当前 D/C AIV 分配（实测所用） | live query + 硬件/拓扑固定策略，**不随 K/bytes/results/route 变化**。48 AIV 上 Dispatch INC=16、Combine INC owner=32；worker W2/W4/W8 的 D/C 分别为 `8/24`、`4/16`、`2/12`，全部 `<=24`。这些数是 live AIV/W 公式在本机的结果，不是跨硬件常量。 |
| Overlap INC partition | 本机默认不相交 partition 为 **dispatch 16 + combine 32 = 48/48**；外部共同 deadline 时两算子均在设备侧预驻留并按 target cycle 释放，避免跨进程 host 提交抖动伪造串行。 |
| NPU 空闲 / 独占 | 跑 gate 前用 `npu-smi` 确认 |

---

## 3. 本机性能锚点

| 项 | 值 |
|:---|:---|
| 单向 PushRoofline | CANN 9.1 实测 down/up=**123.339/146.217 GB/s**（8 KiB）；CANN 8.5 旧 down 参考=140.12 |
| 锚点测量/报告 | `single_inc_overlap_theory_cann91_20260804.md`；`single_inc_final_stress_cann91_20260804.md` |
| 套用 shared `gate_dispatch` 时锚点是否有效 | **YES（CANN 9.1 已重测）**；duplex down/up=112.124/142.748 GB/s，理论参见 `single_inc_overlap_theory_cann91_20260804.md` |

`gate_dispatch(R) = 0.93 * 123.339 / (1 + 0.127/R)`（公式在 shared；数字使用
当前 ACTIVE CANN 9.1 的实测 down roofline，140.12 仅作 CANN 8.5 历史对照）。

---

## 4. 本机复测红绿灯

图例：`PASS` / `FAIL` / `GAP` / `UNVERIFIED`。  
对照：shared §5 峰值榜；gate：shared §2 + 本机 §3 锚点。

### 4.1 历史已达成（CANN 8.5，consolidated 2026-08-03）— 迁移对照

| 套件 | 状态 | 摘要 |
|:---|:---|:---|
| Dispatch dense | PASS* | 300/300；旧叙述偏 ≥256MiB gate |
| Combine K>1 | PASS* | 240/240；≥256MiB >120；**须按 ≥128MiB 重判 128M 列** |
| Combine K1 | **GAP** | 正确；峰值 118.32 &lt; 120 |
| Overlap 256M/K8 | PASS | bal ~1.35–1.41× |

### 4.2 CANN 9.1.0-beta.1 最小复测与自适应回归

> **口径说明**：本节用于 CANN 版本等效性和代表性回归判断。当轮 `warmup=2, measure=5` 样本不满足 shared LIVE_STATUS H11 的正式闭环口径，不得单独据此宣称 hard gate 已关闭。

| Case | 正确性 | 带宽或 ratio | vs Gate | vs 共享峰值榜 | 报告路径 |
|:---|:---|---:|:---|:---|:---|
| D W8/K8 @128MiB | UNVERIFIED | | | | |
| D W8/K8 @256MiB | PASS（9/9） | 133.345 GB/s（10 epoch） | PASS | 与旧水位等效 | `single_inc_final_regular_sweep_cann91_20260804.json` |
| D W8/K1 @128MiB | UNVERIFIED | | | | |
| D W8/K1 @256MiB | PASS（9/9） | 119.847 GB/s（10 epoch） | PASS | 与旧 R=1 水位等效 | `single_inc_final_regular_sweep_cann91_20260804.json` |
| D W2/K8 @256MiB | PASS（3/3） | 127.582 GB/s（10 epoch） | PASS | 同量级 | `single_inc_final_regular_sweep_cann91_20260804.json` |
| C W8/K8 @128MiB | UNVERIFIED | | | | |
| C W8/K8 @256MiB | PASS（9/9） | 126.538 GB/s（10 epoch 正则矩阵） | PASS（≥120） | 连续 20 epoch=126.603 | `single_inc_final_regular_sweep_cann91_20260804.json` |
| C W8/K1 @128MiB | UNVERIFIED | | | | |
| C W8/K1 @256MiB | PASS 正确性（9/9） | 114.643 GB/s（10 epoch） | **FAIL（<120）** | 仍 GAP | `single_inc_final_regular_sweep_cann91_20260804.json` |
| Fixed-AIV overlap W2/4/8 × bal/all1 @256MiB/K8 | **PASS（30/30）** | speedup=1.172–1.502×；W8 balanced 10 次 CV=3.88% | 资源/正确性/真实交叠 PASS | balanced W4/W8 距理论 1.50%/1.33% | `single_inc_overlap_theory_cann91_20260804.md` |
| Final regular sweep @256MiB | **12/12 正确，11/12 perf PASS** | D=119.768–133.687；C K8=126.538–133.796；C K1=114.643–115.676 GB/s | 唯一 W8/K1 FAIL，K1 120 仍 GAP | 连续 20 epoch 较 10 epoch 单测 -0.11%…+0.68% | `single_inc_final_regular_sweep_cann91_20260804.json` |
| K>W/K>8 Dispatch @128MiB | **PASS** | K=9/16/64；115.205–133.708 GB/s；100 sample CV<=0.74% | PASS | W8/K64 vs K8 -0.53%，无回退 | `single_inc_topk_rank_stress_cann91_20260804.md` |
| K>W Combine @256MiB | **PASS** | W8 K8/K16/K64=126.863/127.053/127.767 GB/s | PASS | K64 vs K8 +0.71% | `single_inc_topk_rank_stress_cann91_20260804.md` |
| Combine @128MiB 边界/路由压力 | 正确性 PASS / 性能 GAP | balanced K8/K16/K64=116.177/116.413/116.161；ragged=109.959；all1=60.108 GB/s | **FAIL（formal 120）** | all1 约为单 worker roofline 96%；ragged 是真实开销缺口 | `single_inc_topk_rank_stress_cann91_20260804.md` |
| K64 overlap @256MiB | **PASS（正确性/资源/真交叠）** | balanced/all1 speedup=1.688×/1.267× | 距理论下界 15.28%/16.32% | INC=16+32，worker=2/12，不随 K/路由变 | `single_inc_topk_rank_stress_cann91_20260804.md` |
| Random Combine K1 ingress 修复 | **PASS** | 原失败 plan 12/12；final-build fresh 8/8；K1 transport 中位对比 W2/W4/W8=-2.06%/+0.01%/+0.24% | correctness blocker 关闭；120 GB/s gate 仍 GAP | K8 Combine=132.72/127.98/120.35 GB/s；Dispatch W4/K8=130.45 GB/s | `single_inc_os_random_plan_campaign_cann91_20260804.md` |
| W2/K1 private-MTE + pair-ready | **PASS / PERF GAP** | 256MiB=116.884 GB/s，较同构建回退路径 114.645 GB/s 提升 1.95%；128MiB 分阶段 A/B 延迟 -2.56%/-0.57% | 历史 W2 失败 plan 4/4 + 新系统熵 4/4；仍未过 120 | K8 Combine=134.82/128.37/123.31 GB/s；Dispatch W4/K8=130.57 GB/s；worker AIV=24/16/12 | `single_inc_k1_private_mte_optimization_cann91_20260804.md` |

本机结论（一句话）：CANN 9.1 下 canonical Dispatch 和 Combine K>1 保持旧水位，W8 overlap 明显改善；2026-08-05 native 非 2 次幂 sweep 的 Combine K>1 为 135.04–144.69 GB/s，Combine K1 为 112.81–117.40 GB/s，并暴露 native Dispatch K1 68.30–70.21 GB/s 缺口。AIV map 未变。

### 4.2.1 本机未过 performance gate 明细（2026-08-06 回写）

> **全是性能 FAIL，不是正确性 FAIL。** 共享摘要见 LIVE_STATUS §5.1。  
> 本地最后成功 sync≈2026-08-06 16:45；**未见新容器 Phase-1 复测新报告**。

**Native 非 2 次幂 24-case（11/24 perf PASS）** — `single_inc_nonpow2_sweep_cann91_20260805.md`

| 类别 | 未过数 | Case / 带宽 | 分类 |
|:---|---:|:---|:---|
| Native Dispatch K1 | 6 | W2/W4/W8 ×128/256MiB：**68.30–70.21 GB/s** | 结构性大缺口（几乎不随消息上升；远低于 R=1≈102） |
| Native Dispatch W2/K2 @128M | 1 | **82.93 GB/s**（同 shape @256M=124.48 PASS） | 小尺度/固定成本悬崖 |
| Combine K1 | 6 | **112.81–117.40 GB/s**（&lt;120） | 老 GAP，差约 2–6% |

已过对照：Combine K≥2 六组 **135–145**；Dispatch K≥4/8 及 256M W2/K2 **125–134**。

**Canonical 正则 @256MiB** — `single_inc_final_regular_sweep` / K1 优化报告

| Case | GB/s | 相对 120 |
|:---|---:|:---|
| C W2/K1 | 115.07 → 优化后 **116.88** | GAP ≈2.6% |
| C W4/K1 | 115.68 | GAP |
| C W8/K1 | **114.64** | 正则矩阵显式 FAIL |

### 4.3 当前数据面语义（已拍板）

- Dispatch 对同一 `(token, destination rank)` 只跨 HCCS 发一份 hidden；同 rank 上的多个 expert assignment 全部保留在 metadata，由目标 worker 在 grouped GEMM 前本地展开。这是当前保持不变的生产方向。
- Combine 默认不在 worker 合并 expert-instance：各 expert 输出独立 push 到 INC，由 INC 做完整加权归约。`INC_DYNCSR_RANK_DEDUP=1` 可启用 rank-local pre-reduce，但目前仅显式 opt-in/诊断，不是默认路径。

---

## 5. 本机迁移 Checklist

- [x] 代码根 `/root/work/ascend-样机`（已 sync 过主树）  
- [x] 确认资格化 session 使用 CANN 9.1.0-beta.1  
- [x] `npu-smi` 空闲  
- [x] live topo → §2  
- [x] live AIV=48 → §2  
- [ ] PushRoofline 仍有效或重测 → §3  
- [x] 跑 §4.2 256MiB 最小代表集  
- [x] 刷新全局 overlap 峰值 / 队列 → **回写 shared LIVE_STATUS**  
- [ ] 补 128MiB 全矩阵与长时 soak  

---

## 6. 本机 Changelog

| 日期 | 谁 | 变更 |
|:---|:---|:---|
| 2026-08-03 | handoff | 从模板创建；记录 CANN 8.5 默认 + 9.0 可选；填入历史锚点 140.12 与 4.1 对照 |
| 2026-08-04 | Codex | 回写 CANN 9.1.0-beta.1 最小复测、自适应 AIV 代表回归、K1 未过 120 及 W8 overlap ratio 64.62% |
| 2026-08-04 | Codex | 记录 Dispatch 按 destination-rank 物理去重/本地 expert 展开和 Combine 默认 INC 完整归约语义 |
| 2026-08-04 | Codex | 补 K>W/K>8、最大 64× rank 去重、100 epoch 和 Combine 热点/ragged 压测；确认 256MiB 无 top-k 回退，记录 W8 128MiB/ragged 缺口 |
| 2026-08-04 | Codex | 修复随机 Combine K1 worker→INC ingress 损坏；原 12 失败 plan 和 final-build fresh plan 全过，记录 W2 约 2.06% 剩余成本以及 W4/W8/K>1/Dispatch 带宽回归 |
| 2026-08-04 | Codex | 优化 W2/K1 修复路径：独立 16-KiB row 改用双 private-MTE credit，两行共享已 quiet 的 ready generation；256MiB 提升到 116.884 GB/s，历史/新系统熵复测全过，K>1/Dispatch/AIV 无回退 |
| 2026-08-04 | Codex | 统一 D/C hardware policy；修正本节 CANN 9.1 dispatch gate anchor 为 123.339（历史 final-sweep JSON 内嵌的 140.12 保留为当时记录，不再作为 ACTIVE 判定锚点）。W2 tiny smoke：Dispatch 3/3 rank、Combine 3/3 rank PASS，实际资源 D INC/worker=16/8、C=32/24。 |
| 2026-08-05 | Codex | native D/C 非 2 次幂 token 24-case sweep：最终 24/24 全 rank 正确，性能 11/24 过 hard gate；首试有 1 个 ACLSHMEM init FAIL，精确重跑 PASS 但不抹掉首试失败。详见 `docs/inc/report/single_inc_nonpow2_sweep_cann91_20260805.md`。 |
| 2026-08-06 | Codex | 新增 §4.2.1：逐条列出未过 performance gate（native Dispatch K1×6、D W2/K2@128M、Combine K1×6 + canonical K1 代表）；注明 08-06 重连后尚无新 Phase-1 复测。共享侧对应 LIVE_STATUS §5.1 / W19。 |
