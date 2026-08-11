# Single-INC Sweep 进度看板

> **给刚进仓库的人看「现在测到哪了」**。  
> 本文是 **examples 树内的可读摘要**；硬约束、完整峰值榜与开放队列的权威真源仍是：  
> [`../../../../docs/inc/report/single_inc_LIVE_STATUS.md`](../../../../docs/inc/report/single_inc_LIVE_STATUS.md)  
> 换机 / 换 CANN 后以 ACTIVE profile 的 `ENV_STATUS` 为准，不要只抄本文数字。

| 字段 | 值 |
|---|---|
| 摘要上次整理 | 2026-08-08（对照仓库内已归档证据，非本日新跑） |
| ACTIVE 硬件 profile | **`910b-yuanmingyu`**（见 [`docs/inc/report/ACTIVE_HW_PROFILE.md`](../../../../docs/inc/report/ACTIVE_HW_PROFILE.md)） |
| 候选迁移环境 | **`910b2c-nb`**（npu-borrow；W8 本环境不可全 HCCS 等效，只验 W2/W4） |
| 正式 sweep 入口 | `../scripts/single_inc/run_single_inc_operator_sweep.sh` |
| 主逻辑导读 | [`QUICKSTART.md`](QUICKSTART.md)（含 H1 / dyn-CSR / CMake 稳定名说明） |

---

## 中文

### 1. 环境是什么（两套，勿混读）

#### 1.1 ACTIVE：`910b-yuanmingyu`（主资格化机）

| 项 | 值 |
|---|---|
| 主机 / SSH | `tmp-worker-0` / `yuanmingyu`（亦称 probing-real-worker-0） |
| 代码根（远端典型） | `/root/work/ascend-样机/shmem` |
| NPU | 8× Ascend 910B（`npu-smi` 可见 16 逻辑 die / Phy 0–15） |
| **资格化 CANN** | **`/opt/Ascend-9.1/cann-9.1.0-beta.1`**（须显式 `source …/set_env.sh`；系统默认 8.5 **不是** 当前资格化环境） |
| 驱动 | npu-smi **25.3.rc1.2** |
| Live AIV | **48 / 卡**（策略按 live 查询；本机实测常用 INC D/C=`16/32`，worker W2/W4/W8 D/C=`8/24`、`4/16`、`2/12`） |
| 拓扑角色 | INC 默认逻辑 PE=`W` → 典型 **Phy0**；workers 走 HCCS 集合（禁跨平面乱拼） |
| 本机状态全文 | [`docs/inc/hardware_profiles/910b-yuanmingyu/single_inc_ENV_STATUS.md`](../../../../docs/inc/hardware_profiles/910b-yuanmingyu/single_inc_ENV_STATUS.md) |

#### 1.2 候选：`910b2c-nb`（npu-borrow / Codex remote）

| 项 | 值 |
|---|---|
| 主机 / SSH | `A03-R40-I48-253-0046364.JD.LOCAL` / project `npu-borrow`（`nb`） |
| 代码根 | `/export/home/yinjinrun.montyyin/.cursur/projects/default/shmem` |
| NPU | live **16× Ascend 910B2C**，65536 MiB HBM/卡 |
| 本次实测 CANN | `/usr/local/Ascend/cann-9.1.0-beta.3` |
| 驱动 | npu-smi **25.0.rc1.1** |
| Live AIV | **48 / 卡** |
| 拓扑限制 | 两个 8-card HCCS 岛（0–7、8–15）；**W8+1INC 无法全在同一 HCCS 平面 → 本环境不测 W8**；正式只验 **W2/W4**（INC Phy0，workers Phy1..） |
| 本机状态全文 | [`docs/inc/hardware_profiles/910b2c-nb/single_inc_ENV_STATUS.md`](../../../../docs/inc/hardware_profiles/910b2c-nb/single_inc_ENV_STATUS.md) |

---

### 2. Baseline / Gate 是怎么来的（读进度前必读）

这里的「baseline」**不是**某次随便跑出来的对照二进制，而是 **三层可复现口径**：

#### 2.1 计量口径（H3，全环境同一套）

定义见 shared LIVE_STATUS / architecture 文档：

```text
dispatch_bytes = Σ unique(token, destination_rank) × hidden_bytes   # INC→workers 下行
combine_bytes  = contribution_count × hidden × element_bytes      # workers→INC 上行
makespan_us    = max(rank_device_protocol_us)                       # 最慢 rank 协议区间
useful_GB/s    = operator_bytes / makespan_us / 1000
```

- 计时窗 **不含** 启动、session 建立、host setup、校验。  
- mismatch → 不得报性能达标。  
- 正式样本：warmup≥3、measure≥10，带宽 **CV≤5%**（H11）；偶发 FAIL 禁止「重跑到过」抹掉。  
- **≥128 MiB** 才强制过性能 gate；更小消息只强制正确性 + 稳定/鲁棒。

#### 2.2 PushRoofline 锚点（按环境实测，不是写死 192）

对每个环境，用同拓扑、同方向、push 语义、大消息、充分 AIV 的 **put-only roofline** 测单向聚合上限，记为 `PushRoofline_env`。

| 环境 | Dispatch 用的 down 锚点 | 怎么来的 |
|---|---|---|
| `910b-yuanmingyu` / CANN 9.1 | **123.339 GB/s** | CANN 9.1 实测 down roofline（见 ENV_STATUS §3；overlap theory 文档） |
| 同上，CANN 8.5 历史 | 140.12 GB/s | **仅迁移对照**，不是当前 ACTIVE gate 锚点 |
| `910b2c-nb` | 本机 put-only：W2 多打一 min≈**42.71**、W4 多打一 min≈**85.45** GB/s 等 | 每条 HCCS peer raw≈28 GB/s；W2/W4 同时用 2/4 条 → raw 聚合≈56/112；实测约 raw 的 76%。见 ENV_STATUS §3 |

整机「单方向 192 GB/s」不是单 INC W2/W4 的拓扑屋顶，**不能**拿来当这里的 gate。

#### 2.3 性能 Gate 公式（共享；数字用本机锚点）

**Dispatch**（shared）：

```text
gate_dispatch(R) = 0.93 × PushRoofline_env / (1 + 0.127/R)
  其中 R = min(K, W)
```

在 ACTIVE CANN 9.1、`PushRoofline=123.339` 时，R=1/2/4/8 的门槛约为  
**101.8 / 107.9 / 111.2 / 112.9 GB/s**。

**Combine**（shared，直至用户改口）：

```text
K1 与 K>1：≥128 MiB 时 hard gate ≥ 120 GB/s
```

**nb 可移植性 sweep 用的是另一套「缩放门」**（不改 kernel、不进协议）：  
把旧 yuanmingyu 锚点按本机链路聚合比例缩放，例如  
Dispatch ≈ `123.339 × (56/192)`（W2）或 `×(112/192)`（W4）；  
Combine K≥2 用 35/70，K1 duplex 用 33.864/65.944。  
详见  
[`portability_regression_20260807/README.md`](../../../../docs/inc/hardware_profiles/910b2c-nb/portability_regression_20260807/README.md)。

---

### 3. 当前进度一览（截至归档证据）

#### 3.1 ACTIVE 机（`910b-yuanmingyu` / CANN 9.1）— 产品主线

| 维度 | 状态 | 代表证据 | 一句话 |
|---|---|---|---|
| 正确性 | **强** | 多轮 全 rank PASS；随机 plan Combine K1 correctness blocker **已关** | 功能路径可用 |
| Dispatch dense / 正则 @256MiB | **PASS** | final regular：D≈119.8–133.7 GB/s，12/12 正确 | 过 ACTIVE dispatch gate |
| Combine **K>1** @≥128MiB | **PASS** | native 非 2 次幂 K>1 ≈ **135–145** GB/s；正则 K8 ≈126–134 | 过 120 hard gate |
| Combine **K1** | **GAP** | 正则 @256MiB ≈114.6–116.9；native ≈112.8–117.4 | 正确但 **<120** |
| Native Dispatch **K1** | **结构性 GAP** | 非 2 次幂 128/256MiB 全在 **68.3–70.2** | 几乎不随消息变大 → 实现/固定开销缺口 |
| Native Dispatch 128MiB W2/K2 | **小尺度 FAIL** | 82.93 vs 同 shape@256MiB 124.48 PASS | 小消息悬崖 |
| Overlap | **PASS（资源/正确/真交叠）** | W8 balanced 距理论约 1.33% | 继续向理论下界压 |
| 连续下发 | **PASS** | 同 stream 20 epoch vs 10 epoch 差 −0.11%…+0.68% | 无明显排队稀释 |

**最新详表（请点进去看完整矩阵）：**

| 报告 | 内容 |
|---|---|
| [`single_inc_final_stress_cann91_20260804.md`](../../../../docs/inc/report/single_inc_final_stress_cann91_20260804.md) | 正则 W2/W4/W8×K1/K8 @256MiB + 连续下发 |
| [`single_inc_final_regular_sweep_cann91_20260804.json`](../../../../docs/inc/report/single_inc_final_regular_sweep_cann91_20260804.json) | 可机读正则 sweep |
| [`single_inc_nonpow2_sweep_cann91_20260805.md`](../../../../docs/inc/report/single_inc_nonpow2_sweep_cann91_20260805.md) | native 非 2 次幂 24 组 @~128/256MiB |
| [`single_inc_LIVE_STATUS.md` §5.1](../../../../docs/inc/report/single_inc_LIVE_STATUS.md) | **未过 performance gate 清单**（全是性能 FAIL，正确性均 PASS） |

**正则 @256MiB 速查（canonical，CANN 9.1）：**

| Op | W2/K1 | W2/K8 | W4/K1 | W4/K8 | W8/K1 | W8/K8 |
|---|---:|---:|---:|---:|---:|---:|
| Dispatch GB/s | 120.240 | 127.582 | 119.768 | 133.687 | 119.847 | 133.345 |
| Combine GB/s | 115.069 | 133.796 | 115.676 | 129.455 | **114.643** | 126.538 |

（Combine 该轮唯一相对最终 120 目标的显式缺口在 W8/K1；K1 全系列仍 GAP。）

#### 3.2 候选机（`910b2c-nb`）— 可移植性 / 迁移

| 套件 | 结果 | 证据目录 |
|---|---|---|
| put-only roofline W2/W4 | PASS（H11 稳定性 OK） | ENV_STATUS §3 |
| Legacy W4/K8 D/C 复跑 | 正确 PASS；相对本机 roofline **PERF GAP**（D mean 56.97 / C 73.52） | `operator_replay_w4_k8_20260806/` |
| Portability refactor W2/W4 40-case | **40/40 正确**；gated **29/30 PASS**；唯一 FAIL=Combine W4/K1@128MiB 62.676&lt;65.944（边界；独立 3 跑 mean 66.205） | [`portability_regression_20260807/`](../../../../docs/inc/hardware_profiles/910b2c-nb/portability_regression_20260807/) |
| W8 | **N/A**（拓扑做不到全 HCCS W8+1） | ENV_STATUS |
| Random-route campaign | UNVERIFIED | 队列中 |

---

### 4. 开放项（别人接手时优先看）

1. **Combine K1 → ≥120 GB/s**（ACTIVE，≥128MiB）— 产品 P1a。  
2. **Native Dispatch K1 68–70 GB/s 结构性缺口** — 与 canonical ~120 对比鲜明。  
3. Native Dispatch **128MiB W2/K2** 小尺度悬崖。  
4. nb：把 shared gate 锚点与本机 roofline 正式对齐前，**不要**用 yuanmingyu 的 120/123 直接判 nb PASS。  
5. nb：reproducible random-route campaign 未完成。

---

### 5. 想自己复现

```bash
# ACTIVE 机：确认 source 的是 9.1，不是系统 8.5
source /opt/Ascend-9.1/cann-9.1.0-beta.1/set_env.sh

cmake -S shmem -B build -DCMAKE_BUILD_TYPE=Release -DUSE_EXAMPLES=ON
cmake --build build -j --target inc_dc_single_inc_stream inc_dc_sv2_dyn_csr_combine

cd shmem/examples/inc/dispatch_combine/scripts/single_inc
./run_single_inc_operator_sweep.sh   # 正式矩阵；持 NPU 锁 + 空闲检查
```

随机覆盖：`../random_token_plan/`。  
新人逻辑导读：[`QUICKSTART.md`](QUICKSTART.md)。

---

## English

### What this file is

A **progress board inside the examples tree**. Authoritative gates/queues live in
`docs/inc/report/single_inc_LIVE_STATUS.md`. Do not treat numbers here as a
substitute after a machine/CANN change—re-read the ACTIVE `ENV_STATUS`.

### Environments

- **ACTIVE `910b-yuanmingyu`**: 8×910B, qualification CANN
  `/opt/Ascend-9.1/cann-9.1.0-beta.1`, driver 25.3.rc1.2, 48 AIV/card.
- **Candidate `910b2c-nb`**: 16×910B2C on npu-borrow, CANN
  `cann-9.1.0-beta.3`, driver 25.0.rc1.1; **W2/W4 only** (W8+1 cannot sit on one
  HCCS island).

### How baseline / gates are defined

1. **Metric (H3)**: dispatch bytes = deduped INC→worker payload; combine bytes =
   contributions×hidden; time = max rank protocol interval; GB/s from that ratio.
   Warmup≥3, measure≥10, CV≤5%; ≥128 MiB required for perf gates.
2. **Roofline anchor**: per-env measured push-only aggregate
   (`PushRoofline_env`). ACTIVE CANN 9.1 down anchor = **123.339 GB/s**.
3. **Dispatch gate**: `0.93 * PushRoofline_env / (1 + 0.127/min(K,W))`.  
   **Combine gate**: ≥120 GB/s for both K1 and K>1 (≥128 MiB).  
   nb portability sweep uses **scaled** gates (56/192, 112/192, …)—see that
   campaign README; kernels never read gate constants.

### Current headline

| Area | Status |
|---|---|
| Correctness | Strong on ACTIVE; random-plan Combine K1 correctness closed |
| Dispatch (canonical / dense @256MiB) | PASS |
| Combine K>1 | PASS (~135–145 native non-pow2; ~126–134 regular K8) |
| Combine K1 | **GAP** (~113–117, target 120) |
| Native Dispatch K1 | **Structural GAP** (~68–70) |
| Overlap | PASS; keep closing toward theoretical bound |
| nb portability W2/W4 | 40/40 correct, 29/30 gated PASS; W8 N/A |

Open P0/P1: Combine K1≥120; native Dispatch K1 path; nb random campaign.
