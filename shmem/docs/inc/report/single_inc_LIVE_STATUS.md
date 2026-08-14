# 单 INC Dispatch / Combine — SHARED LIVE STATUS

> **跨环境共享真源（稳定路径，勿改名）。**  
> 放：**硬约束、gate 公式、产品进度、已知峰值、开放队列**。  
> **不放**主机名 / CANN / Phy / 本机复测表 → 那些在各环境的 `single_inc_ENV_STATUS.md`。

| 字段 | 值 |
|:---|:---|
| 文档角色 | 全环境共享：约束 + gate + 进度 |
| 上次更新 | 2026-08-14 |
| 更新者 | Codex（公开 API 收口；历史性能数字与 gate 原样保留） |
| 活跃环境指针 | `../report/ACTIVE_HW_PROFILE.md` → `hardware_profiles/<profile>/single_inc_ENV_STATUS.md` |
| 环境模板 | `env/ENV_STATUS_TEMPLATE.md` |
| 详表快照 | `single_inc_nonpow2_sweep_cann91_20260805.md` + `single_inc_native_api_closure_cann91_20260804.md` + `single_inc_final_stress_cann91_20260804.md` + `single_inc_final_regular_sweep_cann91_20260804.json` + `single_inc_topk_rank_stress_cann91_20260804.md` + `single_inc_os_random_plan_campaign_cann91_20260804.md` + `single_inc_k1_private_mte_optimization_cann91_20260804.md` |

```text
共享本文  ←→  每个环境一份 ENV_STATUS（只改自己的）
   H* / Gate / 队列 / 峰值榜          主机·topo·AIV·本机复测·本机 roofline
```

---

## 0. 两层文档怎么更新

| 变更类型 | 改哪里 |
|:---|:---|
| 硬约束 / gate 公式 / 产品目标 / 开放队列优先级 | **本文（shared）** |
| 某环境测出新峰值、关缺口、进度结论 | **本文**峰值榜 + 队列状态；并写清「在哪个 env」 |
| 主机、CANN、驱动、AIV 数、INC Phy、本机 PushRoofline、本机复测红绿灯 | **仅该 env 的 `ENV_STATUS`** |
| 换机 | 1）新建/填写新 profile 的 `ENV_STATUS` 2）改 `ACTIVE_HW_PROFILE.md` 3）本文队列可加「env X 复测」项，**不要**把旧机数字抄进新 ENV |

Agent 开场：**先读本文 §1–§3 + §5，再读 ACTIVE 指向的 ENV_STATUS。**

---

## 1. 硬性限制（全环境同一套）

未经用户明确改口，**FAIL = 交付无效**：

| ID | 约束 |
|:---|:---|
| H1 | **路径唯一**：`worker → INC → worker` push-only relay。禁止 pull、worker-direct、bypass。 |
| H2 | **拓扑语义**：W workers `[0,W)` + 1 INC（rank `W`）。验收 W∈{2,4,8}；ABI 不写死仅这些 W。 |
| H3 | **计量**：dispatch 分子=INC→workers 下行总字节；combine 分子=workers→INC 上行总字节。分母=最慢 rank 完整 device 协议区间。不含启动/warmup/host setup/校验。 |
| H4 | **正确性优先**：mismatch 不得报性能达标。 |
| H5 | **无 shape 特判传输入口**。 |
| H6 | **拓扑可移植**：INC/worker Phy 由各环境 live topo 决定；禁止把某一环境的 Phy 列表写进共享协议当真理。 |
| H7 | **AIV 不锁死跨环境常量**：禁止「dispatch=16/combine=32」作为全环境硬编码终态。各环境查询 live AIV 再分配；策略可复现，且不牺牲已达标带宽/交叠。 |
| H8 | **废止数字**：已清理的历史 completion 报告及旧 overlap 1.68×/1.83× 不得引用为现状。 |
| H9 | **鲁棒性（功能）**：同一路径覆盖 balanced / all-to-one / 任意 token plan；多 epoch 无 Missing/丢 tile/静默错数。 |
| H10 | **鲁棒性（并发）**：D∥C 独立 session；AIV cohort 不相交；无 hang / 隐式全局死锁。 |
| H11 | **稳定性（可重复）**：warmup≥3、measure≥10（overlap concurrent≥5）。带宽 **CV≤5%**（能逐 repeat 时强制）。偶发 FAIL 计不合格，禁止「重跑到过」抹掉。 |
| H12 | **稳定性（完成）**：有限时间完成或明确错误；hang/超时=FAIL。 |
| H13 | **性能门槛范围**：**≥128 MiB** 必须过 §2；`<128 MiB` 只强制正确性 + H9–H12。 |
| H14 | **Dispatch 逻辑全量、物理去重**：每个 token 的所有 expert assignment 都必须保留；对同一 destination rank 的多个 expert，跨 HCCS 只发一份 hidden，由目标 worker 根据 expert/ordinal/weight metadata 本地展开。不得丢 assignment，也不默认改回跨卡每 expert 重复发送。 |
| H15 | **Combine 归约语义**：默认每个 expert-instance 独立 push 到 INC，由 INC 做完整加权归约。rank-local pre-reduce 只是显式 opt-in/诊断路径，不得被 shape 启发式默认选中。 |
| H16 | **Worker AIV 硬上限**：Dispatch 和 Combine 在每个 worker 上都必须使用 `<= floor(live_AIV/2)`，包括所有显式 override/调试开关；48 AIV 上 **24 合法**。默认应选满足带宽的尽可能小 cohort，为模型计算保留 AIV。 |
| H17 | **INC AIV 硬件自适应**：INC 侧不受「一半」上限，但只能由 live capability/硬件拓扑确定；不得把本机 16/32 写成跨硬件常量。D∥C 时 cohort 必须不相交，且按实际 block 数合计不超过当前 INC 可用 AIV。 |
| H18 | **无回退**：提升 Combine K1 或修改 AIV/分块/传输策略时，Dispatch、Combine K>1、其他已资格化 case 与 overlap 不得出现超过测量噪声的系统性带宽/墙钟回退，且任何已有 hard gate 不得由 PASS 变 FAIL。 |
| H19 | **CANN/硬件切换先等效复测**：上传或切换 CANN、驱动或机器后，先用固定代表集确认正确性和旧带宽等级基本不变；显著异常先排查 CANN/topology/link/AIV/干扰/计量，禁止先改算法掩盖环境问题。 |
| H20 | **可移植/可扩展**：W2/W4/W8 是验收规模，不是 ABI 或 kernel 的只合法规模。禁止用 W/K/Phy/CANN 查表切换数据面；调优常量必须属于显式硬件 profile/artifact capability，能力未知或不足时 fail closed，不得静默套用当前 910B 假设。 |
| H21 | **开关不得绕过约束**：环境变量和调试 override 可用于资格化/A-B，但不得绕过 H1、H4、H10、H14–H20；无效或被忽略的配置项必须拒绝或明确报告，不得伪装成已生效。 |
| H22 | **禁止 workload-adaptive AIV**：硬件环境/拓扑确定后，INC 与 worker 的 AIV map 必须确定且可复现，在 K、message bytes、token/result 数、balanced/all-to-one/任意 route plan 之间不得变化。W 作为硬件拓扑规模可参与 worker cohort 计算。 |

---

## 2. 性能 Gate（公式共享；锚点按环境）

### 2.1 适用范围

- **≥128 MiB** → 必须过带宽 gate。  
- **`<128 MiB`** → 只正确性 + 鲁棒/稳定。  
- 报告字段：`R=min(K,W)`、`C(R)`（若用）、实测、`实测/C(R)`。

### 2.2 Dispatch

```
gate_dispatch(R) = 0.93 * PushRoofline_env / (1 + 0.127/R)
```

- `PushRoofline_env`：**写在该环境的 ENV_STATUS**，不写死在本文。  
- 参考：yuanmingyu 曾测 ≈140.12 GB/s（见该 ENV）。新环境未测前判 PASS/FAIL 必须标注「锚点未更新」。

### 2.3 Combine（≥128 MiB）

| 系列 | Gate（全环境同一数字门槛，直至用户改口） |
|:---|:---|
| K>1 | **≥ 120 GB/s** |
| K1 | **≥ 120 GB/s** |

可选升级为按双向带宽建模的 shape gate：见 `single_inc_overlap_theory_cann91_20260804.md`；批准前以 120 为准。

### 2.4 Overlap 口径（共享）

`ratio = concurrent_makespan / (Td_solo + Tc_solo)`（越低越好）；理论下界 `max(Td,Tc)/(Td+Tc)`。  
产品目标：守住 D/C gate + H9–H12，把 ratio **推近理论下界**。

### 2.5 回归红线（共享）

1. 所有 ≥128 MiB dispatch：相对**该环境当时** `gate_dispatch` 零 performance fail。  
2. Combine K>1 ≥128 MiB：不得低于 **120 GB/s**，也不得相对同环境同口径资格基线出现超出噪声的系统性下降。  
3. Combine K1 关闭缺口：W2/W4/W8 的正式样本均正确且 **≥120 GB/s**；关闭后的日常回归无需重扫全矩阵，可选一个正则、有代表性的矩阵，但必须同时保留一个 K>1 与一个 Dispatch 无回退样本。  
4. Overlap：正确、无 hang；ratio 不得相对同环境已资格基线明显恶化，并持续向理论下界收敛。  
5. 鲁棒抽样无新增 FAIL；任何偶发 mismatch/hang/timeout 都不得用重跑的 PASS 覆盖。

### 2.6 正式资格与定向调优的区别

- **正式关 gate**：必须守 H3/H4/H11/H12，在同一 CANN、topology、物理字节口径下给出逐 sample 或可复算统计；不得使用 warmup=2/measure=5 的代表复测直接宣告 H11 已满足。
- **K1 尚未达标时**：允许定向 A/B 和小集合快速迭代，但每次候选变更至少要跑 Combine K1 代表、Combine K>1 代表、Dispatch 代表和正确性。
- **K1 达标后**：不再要求 sweep 所有 shape；按 §2.5 第 3 项用代表正则矩阵和无回退样本维护即可。
- **换硬件/CANN**：代表集可用于先判定「基本等效」；只有在该环境重测 roofline/正式样本后，才能把旧锚点改成新环境的 hard-gate 依据。

---

## 3. 产品目标与推进顺序（共享）

| 目标 | Done |
|:---|:---|
| G-Dispatch | 正确 + H9–H12 + §2.2（≥128MiB） |
| G-Combine | 正确 + H9–H12 + §2.3（含 K1） |
| G-Overlap | §2.4；逼近理论下界 |
| G-Portable | 每个活跃环境有完整 ENV_STATUS，且最小复测可对照峰值榜 |

```text
P0  当前 ACTIVE 环境：填 ENV_STATUS + 最小复测
P1a Combine K1 → ≥120（≥128MiB）     ⎫ 可部分并行
P1b 自适应 AIV（H7）                  ⎭ 变更后多环境或至少 ACTIVE 回归
P2  抬 overlap → 理论下界
```

---

## 4. 跨环境进度总览

图例：`PASS` / `FAIL` / `GAP` / `UNVERIFIED` / `N/A`。

| 环境 profile | ENV 文档 | D dense | C K>1 | C K1 | Overlap | 备注 |
|:---|:---|:---|:---|:---|:---|
| 910b-yuanmingyu / CANN 9.1.0-beta.1 | [ENV_STATUS](../hardware_profiles/910b-yuanmingyu/single_inc_ENV_STATUS.md) | **CANONICAL PASS / NATIVE GAP**（非 2 次幂 native K1=68.30–70.21） | **PASS**（非 2 次幂 K>1 135.04–144.69） | **GAP**（112.81–117.40，<120） | **PASS**（W8 balanced 距理论 1.33%） | native 128/256MiB 非 2 次幂 24/24 最终正确；首试 1 个 init FAIL 仍记录 |

256MiB 代表集证明环境与旧带宽等级基本等效；**共享最终进度仍以现行 ≥128MiB + 各 ENV 完整 gate 为准**。

---

## 5. 已知峰值榜（共享；注明来源环境）

> 只记录**已验证**最好结果，便于迁移对照。新环境超过则更新本表并写明 env。

| 指标 | 最佳值 | Shape / 条件 | 来源环境 | 日期 |
|:---|:---|:---|:---|:---|
| Dispatch 峰值 | **137.09 GB/s** | W8/K8 @8G | 910b-yuanmingyu | 2026-08-03 |
| Dispatch R=1 大消息 | ~120.9 GB/s | W8/K1 @≥256M | 910b-yuanmingyu | 2026-08-03 |
| Combine K>1 峰值 | **~134.8 GB/s** | W2/K8 @8G 一带 | 910b-yuanmingyu | 2026-08-03 |
| Combine K1 峰值 | **118.32 GB/s**（**GAP vs 120**） | W2 @2G | 910b-yuanmingyu | 2026-08-03 |
| Overlap speedup（bal） | **1.413×**（ratio 70.8%） | W2 / 256M / K8 | 910b-yuanmingyu | 2026-08-03 |
| Overlap speedup（bal）W8 | **1.5475×**（ratio 64.62%） | W8 / 256M / K8，CANN 9.1 | 910b-yuanmingyu | 2026-08-03 |
| PushRoofline（单向） | **140.12 GB/s** | 参考锚点 | 910b-yuanmingyu | 见 ENV |

最新详表：`single_inc_nonpow2_sweep_cann91_20260805.md`（native 非 2 次幂） +
`single_inc_final_stress_cann91_20260804.md` / `single_inc_final_regular_sweep_cann91_20260804.json`（canonical 正则基线）。

### 5.1 当前未过 performance gate 清单（共享摘要）

> **全是性能 FAIL，不是正确性 FAIL。** 下列 case 最终正确性均 PASS。  
> 来源环境：`910b-yuanmingyu` / CANN 9.1.0-beta.1。  
> 截至 2026-08-06：本地最后成功 sync≈16:45；**未见新容器上的 Phase-1 复测新报告**。

**Gate 口径**

| 算子 | Gate |
|:---|:---|
| Dispatch | 相对 ACTIVE down roofline **123.339 GB/s**：R=1/2/4/8 ≈ **101.8 / 107.9 / 111.2 / 112.9 GB/s** |
| Combine | 硬门槛 **≥120 GB/s**（K1 与 K>1 相同） |

**A. Native Dispatch K1（结构性大缺口）** — `single_inc_nonpow2_sweep_cann91_20260805.md`，**6/6 FAIL**

| Case | 约消息量 | tokens/worker | mean GB/s | 判定 |
|:---|---:|---:|---:|:---|
| D W2/K1 | 128M | 4099 | 68.46 | FAIL |
| D W4/K1 | 128M | 2053 | 68.33 | FAIL |
| D W8/K1 | 128M | 1031 | 68.30 | FAIL |
| D W2/K1 | 256M | 8201 | 69.56 | FAIL |
| D W4/K1 | 256M | 4099 | 70.21 | FAIL |
| D W8/K1 | 256M | 2053 | 70.09 | FAIL |

特征：带宽几乎不随消息变大上升（68–70），远低于 R=1 门槛 ~102 → **native Dispatch K1 路径固定开销/实现缺口**。对比：同轮 canonical 口径 Dispatch K1 @256MiB 约 **119.8–120.2**（过线）。

**B. Native Dispatch 128MiB W2/K2（小尺度悬崖）** — 同上，**1/1 FAIL**

| Case | mean GB/s | 对照 | 判定 |
|:---|---:|:---|:---|
| D W2/K2 @128M（tokens=2053） | **82.93** | 同 shape @256M = **124.48 PASS** | FAIL |

**C. Combine K1（老 GAP，差约 2–6%）**

Native 非 2 次幂（**6/6 FAIL**）：

| Case | mean GB/s |
|:---|---:|
| C W2/K1 @128M | 113.65 |
| C W4/K1 @128M | 115.23 |
| C W8/K1 @128M | 117.40 |
| C W2/K1 @256M | 117.34 |
| C W4/K1 @256M | 112.81 |
| C W8/K1 @256M | 115.30 |

Canonical 正则 @256MiB（相对最终目标 120 均未达标；JSON 临时 gate=115 时仅 W8 记 FAIL）：

| Case | GB/s | 备注 |
|:---|---:|:---|
| C W2/K1 | 115.07 → 优化后 **116.88** | private-MTE+pair-ready；仍 GAP ≈2.6% |
| C W4/K1 | 115.68 | 未进 W2 新分支 |
| C W8/K1 | **114.64** | 正则 sweep 唯一显式 perf FAIL |

**D. 相关但未并入上表的开放缺口（见队列 W8）**

- Combine W8 @128MiB balanced K8/K16/K64 ≈116.2；ragged≈110.0；all1≈60.1（相对 formal 120 仍 FAIL）

**已过对照（避免误读）**

- Combine **K≥2**：native 6/6 PASS，约 **135–145 GB/s**
- Dispatch **K≥4/8** 及 256M W2/K2：约 **125–134 GB/s**
- 随机 plan / Combine K1 **correctness** blocker：**已关闭**

---

## 6. 开放工作队列（共享进度）

| ID | 优先级 | 项 | 状态 | 完成定义 |
|:---|:---|:---|:---|:---|
| W0 | P0 | ACTIVE 环境：CANN 9.1 最小复测 | **DONE（256MiB 代表集）** | 正确性全过，带宽与旧水位等效；128MiB 全矩阵并入 W4b |
| W1 | P1 | Combine K1 ≥120（≥128MiB，W2/4/8） | **GAP（W2 已优化）** | 见 §5.1-C。W2 private-MTE + pair-ready：256MiB=116.884 GB/s（+1.95% vs 回退路径）；native 非 2 次幂 K1 六组 112.8–117.4 全未过 120。W4/W8 不进入 W2 新分支；K>1/Dispatch 代表无回退 |
| W2 | P1 | Live AIV 查询 + D/C 硬件固定分配 | **DONE（policy + native workspace 已消费） / PARTIAL（外部 resource broker）** | `inc_dc_resource_policy.h` 是唯一公式入口，只依赖 live AIV/W；48 AIV 导出 INC D/C=`16/32`，worker W2/W4/W8 D/C=`8/24`,`4/16`,`2/12`。Dispatch/Combine native workspace 均已使用该 fixed map，tiny call 不缩 cohort，越界 override fail closed；仍需与上层计算资源 broker 协同预留 |
| W3 | P2 | Overlap 逼近理论下界 | **BALANCED DONE / FOLLOW-UP QUEUED** | CANN 9.1 非对称 duplex 理论已校准；W4/W8 balanced 实测仅高理论 1.50%/1.33%，all-to-one 仍高 17.76–28.33%。已排队 12-row bytes/top-k/尺度/热点/顺序矩阵，见 `single_inc_overlap_followup_plan_cann91_20260804.md` |
| W4 | P2 | 各环境自有 PushRoofline 写入 ENV，再套 §2.2 | TODO | 公式在 shared，数字在 ENV |
| W4b | P1 | 按 ≥128MiB 重判 dense performance | **NATIVE NONPOW2 DONE / PERF GAPS OPEN** | 见 §5.1。24/24 正确、11/24 perf PASS；未过=Dispatch K1×6 + D W2/K2@128M + Combine K1×6。首试 1 个 ACLSHMEM init FAIL 已原样保留 |
| W19 | P0 | Native Dispatch K1 ≥ R=1 gate（≥128MiB） | **OPEN（~69 GB/s）** | 见 §5.1-A。native 路径 W2/W4/W8 ×128/256MiB 六组仅 68.3–70.2，远低于 ~102；与 canonical Dispatch K1（~120）形成对照。关闭前不得用 K>1 代表掩盖 |
| W5 | 低 | Megatron / vLLM 框架接入 | **NATIVE LAYOUT ADAPTER DONE / FRAMEWORK BINDING + GROUPED GEMM OPEN** | expert layout ABI 覆盖 permutation/inverse、logical/padded offsets、tokens-per-expert 与 alignment；route handle 绑 generation。`BuildNativeExpertLayout` 使用模型配置 expert 全集，保留 zero-token expert，交付 physical Dispatch→padded expert-major 及 Combine inverse map。W2/W4/W8 真机已经 alignment=8 device padded buffer 数值 PASS。仍缺 Megatron/vLLM/PyTorch 薄绑定和真实 grouped GEMM |
| W6 | P1 | 可移植性收口 | **PARTIAL（fixed map + planner + native launch DONE）** | C ABI `inc_dc_resolve_fixed_aiv_map`、joint planner 与 Dispatch/Combine native workspace 共用唯一 hardware/topology-only policy；request knob 只能与固定值一致，artifact 不兼容时 fail closed。CANN 9.1 下 W2/W4/W8 真机均通过。仍需多硬件 profile 重定资格和上层 resource broker 协同 |
| W7 | P0 | K>W/K>8 + rank 去重压测 | **DONE（正确性/扩展）** | Dispatch K=9/16/64 与 Combine K=16/64 全部正确；W8/K64 Dispatch balanced/all1 100 sample CV=0.74%/0.71%；256MiB Combine K64=127.767；8190/2-byte row 与 8 个 host gate 全过，无 top-k 回退 |
| W8 | P1 | Combine 128MiB/ragged/热点压力缺口 | **OPEN** | W8 balanced 128MiB K8/K16/K64=116.177/116.413/116.161；ragged=109.959。all1=60.108 约为单 worker 屋顶 96%，需补 active-source roofline 报告但未经批准不改 120 formal gate |
| W9 | P2 | K64 交叠距理论下界 | **OPEN** | W8/256MiB balanced/all1 全 rank 正确且 AIV 合规；speedup=1.688×/1.267×，makespan 高于理论下界 15.28%/16.32% |
| W10 | P1 | 单 INC 最短 C++ API + example | **PUBLIC SURFACE CLOSED / NATIVE SERVICE DONE** | 当前唯一业务入口为 `common/api/inc_dc_single_inc.hpp`：`create -> dispatch -> expert compute -> combine -> destroy`。`SingleIncBatch` 自动持有/释放 generation 与 route；async request、query/cancel、plan/stats 不公开。唯一示例为 `common/examples/single_inc_api/inc_dc_single_inc_api_example.cpp`。Framework/Easy/Inference 仅保留为已验证 V1 runtime 的内部依赖。native provider、registered view 与同机 persistent service 的历史带宽证据不变 |
| W11 | P0 | 系统熵随机 token-plan 带宽压测 | **DONE（K1 correctness blocker closed）** | 原始 400 case 保留为 388 PASS/12 FAIL 基线；最终修复将原 12 个失败 plan 复测为 12/12。本轮 private-MTE/pair-ready 又复测 W2 历史 4/4 + 新系统熵 4/4，恢复了部分 W2 修复成本；资源 map 不变且 worker `<=1/2` |
| W12 | P0 | 资源/语义单一真源与静态 contract gate | **DONE** | 修正核心架构、pipeline、框架契约和 CANN 9.1 dispatch gate 锚点；新增静态 gate，禁止旧 24/24、workload-adaptive worker AIV、本地预归约默认计量和静默越界 override 回流。policy 单测覆盖 live AIV=16/24/32/48/64、W=1..16；CANN 9.1 W2 tiny D/C 真机均全 rank PASS，资源为 D=`16/8`、C=`32/24` |
| W13 | P0 | API/runtime 生命周期与数值 fail-closed | **DONE（host 4096 + native D/C 100 + full 256 GEN）** | 历史内部 runtime 生命周期 4096 代；native Dispatch/Combine 单算子各 100 代；单 communicator D→identity→C 同一 sessions/workspace/route 连续 256 代，3/3 rank 与数值 PASS。公开 `SingleIncBatch` 将 route 生命周期收进 RAII；device telemetry 故障恢复证据不变 |
| W14 | P1 | 可观测性与运行时证据 | **INTERNAL COUNTERS + D/C ACTIVE-LANE TELEMETRY DONE / PUBLIC SNAPSHOT DEFERRED** | context stats 与 active-lane telemetry 继续作为内部门禁，不属于当前最短公开 API。Dispatch/Combine 仍校验 lane generation/fail code，故障可 fail-closed 并恢复；若真实框架接入需要，再设计精简公共 snapshot，而不是恢复整套 request/query API |
| W15 | P0 | 产品 host 发布门禁 | **DONE** | 当前公开门禁为 `inc_dc_single_inc_api_example`、`inc_dc_single_inc_c_header_tests` 与 `inc_dc_inference_api_tests`；native full target 继续验证真实 backend 闭包。2026-08-14 重构后均构建通过，最短示例数值 PASS、workspace alloc/free=2/2。Framework/Easy/Inference 的额外测试只是内部回归，不是使用示例 |
| W16 | P1 | 单 communicator native D→expert→C 整链 | **IDENTITY PIPELINE DONE / GROUPED GEMM BINDING OPEN** | `NativeCompositeBackend` 在内部 communicator 中转发 D/C；reverse compiler 导出 `contributor_dispatch_rows`，精确 route 由公开 `SingleIncBatch` 跨 D/C 持有。W2/W4/W8 历史真机均全 rank PASS。中间是 expert-major padded device buffer + identity expert，只验证布局/生命周期，不报 GEMM 性能 |
| W17 | P1 | worker 隐藏 INC 的 persistent proxy | **SAME-HOST DONE / CROSS-HOST ADAPTER OPEN** | `NativeIncService` 长驻 INC，收齐 `DISPATCH_PREPARE`→`DISPATCH_READY`→`COMBINE_READY` 后启动真实 kernel；消息校验 version/op/rank/generation，I/O 有界。W2 256 代、W4/W8 和故障恢复全过，无逐代 W+1 barrier。当前 transport 是同机 loopback，跨主机需 bootstrap/RDMA adapter |
| W18 | P1 | expert-major/padded native adapter | **DONE（LAYOUT CORE + DEVICE CONSUMPTION）** | 输入配置 expert 全集而非 workload 活跃集；导出 offsets/tokens/permutation/inverse，zero-token expert 保留，非 2 的幂 alignment 拒绝。native full W2/W4 实际经过 alignment=8 padded device buffer；identity 只是 GEMM 可替换槽，不报计算性能 |

---

## 7. 文档地图

| 文档 | 层 | 角色 |
|:---|:---|:---|
| **本文** | **共享** | 约束、gate 公式、峰值榜、队列 |
| `hardware_profiles/<p>/single_inc_ENV_STATUS.md` | **环境** | 本机事实 + 本机复测 |
| `env/ENV_STATUS_TEMPLATE.md` | 模板 | 新机克隆 |
| `ACTIVE_HW_PROFILE.md` | 指针 | 当前 ACTIVE 环境名 |
| `single_inc_native_api_closure_cann91_20260804.md` | **native API 快照** | D/C provider、复用、真机带宽与开放边界 |
| `single_inc_final_stress_cann91_20260804.md` | **压测快照** | CANN 9.1 sweep + 连续下发 |
| `single_inc_nonpow2_sweep_cann91_20260805.md` | **native 非 2 次幂 sweep** | 24-case 正确/性能明细；§5.1 主证据 |

公开入口：`examples/inc/dispatch_combine/common/api/inc_dc_single_inc.hpp`。唯一完整示例：
`examples/inc/dispatch_combine/common/examples/single_inc_api/inc_dc_single_inc_api_example.cpp`。
设备实现位于 `single_inc/dispatch/`、`single_inc/combine/`、`single_inc/planning/`
和 `single_inc/runtime/`；Framework/Easy/Inference 只作为内部实现与回归依赖。

---

## 8. Changelog（共享，只追加）

| 日期 | 谁 | 变更 |
|:---|:---|:---|
| 2026-08-14 | Codex | 公开 API 收口为 `inc_dc_single_inc.hpp` 的五步同步生命周期；删除旧 Easy/Inference 示例及 SingleInc async/request/query/cancel/plan/stats 导出，保留底层兼容实现作为稳定 V1 内部依赖；唯一 CPU mock 示例与 host/native 构建门禁通过，历史性能数据不改写。 |
| 2026-08-03 | handoff | 补齐 `910b-yuanmingyu/single_inc_ENV_STATUS.md`（CANN 8.5 默认 / 9.0 可选写在 ENV）；ACTIVE 指针改为两层 |
| 2026-08-03 | handoff | **结构变更**：拆成 shared 本文 + per-env `ENV_STATUS`；峰值榜/队列留共享，主机与本机复测下沉环境文档 |
| 2026-08-03 | handoff | 约束 H9–H13；≥128MiB gate；双目标 K1+overlap；自适应 AIV |
| 2026-08-04 | Codex | 回写 CANN 9.1 P0 复测、自适应 AIV 代表结果与 W8 overlap 64.62%；K1 仍 GAP |
| 2026-08-04 | Codex | 增加 H14/H15：Dispatch 逻辑全量、跨 rank 物理去重并本地展开；Combine 默认 INC 完整归约，本地预归约仅 opt-in |
| 2026-08-04 | Codex | 记录可移植性/框架审计：当前 C ABI 为抽象层，真实 Megatron/vLLM backend 和当前 kernel artifact 对齐尚未完成 |
| 2026-08-04 | Codex | 补齐 H16–H21 和 §2.5–§2.6：worker AIV `<=1/2`（24 合法）、INC 自适应/并发隔离、其他 case 无回退、CANN 等效复测、可移植 fail-closed 与 K1 达标后代表矩阵策略 |
| 2026-08-04 | Codex | 按用户澄清增加 H22：AIV 只对硬件/拓扑自适应，确定环境后禁止 workload-adaptive；CANN 9.1 固定资源 6 组 overlap 矩阵 30/30 正确且真实交叠，最新汇总见 `single_inc_overlap_theory_cann91_20260804.md` |
| 2026-08-04 | Codex | 完成 CANN 9.1 单向/双向 roofline 与非对称理论校准；设备 target-cycle 预驻留消除跨 session 伪串行；balanced W4/W8 距理论 1.50%/1.33%，owner32 恢复单算子性能，详见 `single_inc_overlap_theory_cann91_20260804.md` |
| 2026-08-04 | Codex | 完成最终正则 sweep 与连续下发压测：12/12 正确、11/12 性能 PASS；Dispatch/Combine 同 stream 20 epoch 串行且带宽较 10 epoch 单测仅 -0.11%…+0.68%；K1 120 仍 GAP，详见 `single_inc_final_stress_cann91_20260804.md` |
| 2026-08-04 | Codex | 完成 K>W/K>8、rank 去重、热点/ragged 与 100 epoch 压测；Dispatch 最大验证 K64 且 64× 去重无丢 assignment，Combine 256MiB 无 top-k 回退；新增 W8 128MiB/ragged 开放缺口，详见 `single_inc_topk_rank_stress_cann91_20260804.md` |
| 2026-08-04 | Codex | 新增 `inc_dc_easy_api` 简洁 C facade、communicator/workspace/async request 生命周期、纯 C 头文件门禁和 `easy_api_example/` README + Dispatch/Combine 示例；全部编译/单测通过。真实高带宽 native backend provider 仍列为开放项，未使用 mock 假充生产完成 |
| 2026-08-04 | Codex | 完成 400 个 `getrandom(2)` 系统熵 route plan 压测：Dispatch 200/200、Combine K>1 166/166 正确，但 Combine K1 出现 12 个 route-dependent mismatch；随机 route 大消息也暴露明显 gate 长尾。AIV 对所有 plan/K/bytes 保持硬件固定，详见 `single_inc_os_random_plan_campaign_cann91_20260804.md` |
| 2026-08-04 | Codex | 完善 Easy API 测试接入：公开 dense token-plan wire ABI 和 host builder，显式演示 expert/destination/weight 构造、device 上传、同 rank 多 expert 物理去重、Dispatch/Combine 调用与参数含义；新增 `token_plan_example.c`/`full_example.c`，全部 C11 编译和 API 单测 PASS |
| 2026-08-04 | Codex | 关闭随机 Combine K1 correctness blocker：定位到 worker→INC 相邻 16-KiB row 长包 ingress 损坏，改为单 row chunk，W2 使用拓扑限定 8-credit 非相邻 train。原失败 plan 12/12 + final-build fresh 8/8 正确；K1 带宽对比 W2 -2.06%、W4 +0.01%、W8 +0.24%，K>1/Dispatch 代表回归通过 |
| 2026-08-04 | Codex | W2/K1 默认改为两个私有 UB/event credit 的独立 16-KiB put，并用两行 generation 释放；256MiB `114.645 -> 116.884 GB/s`，历史/新系统熵各 4/4，K8 W2/W4/W8=134.82/128.37/123.31，Dispatch W4/K8=130.57，AIV 与连续 epoch 语义不变。K1 120 仍 GAP，详见 `single_inc_k1_private_mte_optimization_cann91_20260804.md` |
| 2026-08-04 | Codex | 启动新增 200-case 系统熵 token-plan tmux campaign；扩展 overlap runner 以支持独立 D/U bytes 和可选 W/route，排队 12-row 跟进矩阵。补充已有实测的收益实现率：balanced W4/W8=97.15%/97.43%，all-to-one W4/W8=49.96%/56.62% |
| 2026-08-04 | Codex | 产品化 P0-1：新增共享 `inc_dc_resource_policy.h`，Dispatch/Combine host 不再各自复制公式；移除 Dispatch tiny-row cohort 缩减，INC/worker override 统一 fail closed，非法 rank-dedup 配置拒绝。修正 24/24、18/12/5、默认本地预归约计量等陈旧文档，并将 ACTIVE CANN 9.1 dispatch anchor 改为 123.339。静态 contract gate、policy 单测和 W2 tiny 真机 D/C 均 PASS。 |
| 2026-08-05 | Codex | 完成 native D/C 128/256MiB 非 2 次幂 token sweep：24/24 最终正确、protocol 1360/1360、CV 0.24%–1.37%；K>1 Combine 全过，Combine K1 仍 GAP，新暴露 native Dispatch K1 与 128MiB W2/K2 缺口。保留一次首试 ACLSHMEM init FAIL 及精确重跑 PASS；AIV map 不变。 |
| 2026-08-06 | Codex | 将未过 performance gate 收成共享 §5.1 清单（Dispatch K1×6、D W2/K2@128M、Combine K1 native×6 + canonical 代表；对照已过项）；队列新增 W19（native Dispatch K1），并回写 W1/W4b 指向 §5.1。注明 08-06 容器重连后尚无新 Phase-1 复测落盘。 |
| 2026-08-04 | Codex | 产品化 P0-2：补齐 Easy/Framework API 生命周期门禁与 FP32 权重 ABI；workspace 随活跃 request pin，borrowed context 不被 Easy communicator 误销毁，同一 communicator/workspace 4096 代复用 PASS，NaN/Inf route weight fail closed。修复 `inc_dc_combine_vector_reduce_tests` 的无关全量源链接缺口；Easy、Framework、service semantics、FP32 vector reduce 回归全部 PASS。真机长时 soak 等 native provider 接入后执行。 |
| 2026-08-04 | Codex | 产品化 P0-3：将共享 hardware/topology-only AIV policy 暴露为稳定 C ABI，runtime 显式 worker/INC joint planner 改为固定 map；拒绝与固定 cohort 不一致的 legacy request override，不兼容 artifact 直接 fail closed，且不再对生产 joint session 启用 time-multiplex 隐式回退。更新 910B 资格 artifact 边界并通过 C/C++ ABI、discovery、runtime、client 全回归。 |
| 2026-08-04 | Codex | 产品化 P0-4：新增 framework expert-layout extension，覆盖 grouped-GEMM 所需 permutation/inverse、logical/padded expert offsets、tokens-per-expert 与 alignment；非 2 的幂 alignment、重复扩展和未知扩展均 fail closed。Framework/Easy 单测及 C examples 编译门禁 PASS；未将 ABI 完成误报为 native Megatron/vLLM 接入完成。 |
| 2026-08-04 | Codex | 产品化 P0-5：新增 Framework/Easy context stats，无 device sync 返回 D/C enqueue、终态、timeout/enqueue-failure、当前/峰值 inflight 及 plan/workspace lease。4096 代生命周期回归对 `2049 D + 2049 C、4097 completed、1 cancelled、1 timeout、0 live` 做精确 gate；README 补充调用契约。 |
| 2026-08-04 | Codex | 产品化 P0-6：Framework/Easy 新增 generation-checked opaque route handle；从活跃 dispatch request 捕获精确 route，dispatch request 释放后 handle 仍可供 combine 复用，combine inflight 期间 handle release=`BUSY`，handle 存活期间 plan release=`BUSY`，释放后 stale handle 被拒绝。Framework/Easy/C examples 回归 PASS。 |
| 2026-08-04 | Codex | 产品化 P0-7：新增统一 host release gate，把资源硬约束、API/ABI、生命周期、数值语义、portability/discovery/runtime/client 收成一个可重复入口；静态 contract 同时要求 fixed-map、expert-layout、opaque-handle 和 stats 标记存在。当前完整 host gate PASS。 |
| 2026-08-04 | Codex | 产品化 P0-8：新增 backend enqueue 故障注入，覆盖显式 `BACKEND_ERROR` 与 backend 错误返回 `OK+空 ticket` 两条路径；两者均必须回滚 framework request slot、plan inflight 和 workspace/route pin，并证明紧接的下一代 enqueue-wait-release 可恢复。stats 精确记录 2 次 enqueue failure。 |
| 2026-08-04 | Codex | 产品化 P0-9（Dispatch native）：抽取 canonical token-plan→stream task/workspace compiler 和真实 Framework backend，worker/INC 直接启动已资格化的高带宽 kernel；修复多代中“快 rank 越代清理”与 INC 早于 worker staging 起跑的竞态，并将全活跃 AIV 协议错误纳入 request 终态。W2 100 代与 256MiB 代表矩阵正确；API 端到端含本地 D2D 为 99.30–101.95 GB/s，standalone kernel event 参考约 122–126 GB/s，底层带宽代码未修改。 |
| 2026-08-04 | Codex | 产品化 P0-10（Combine native + registered view）：从 Dispatch route/assignment 唯一真源生成 reverse logical plan 和 mode-6 DYN-CSR workspace，Framework/Easy 直接启动 worker producer/INC full-reduce/remote-result kernel；修复 expert-row 逐行 D2D 带宽陷阱，改为稳定 result/ordinal 布局和整 rank 连续 staging。D/C 均提供 registered symmetric view；W2 各 100 代、W4/W8 扩展正确，Dispatch 256MiB protocol=124.288–128.047 GB/s，Combine 178,978,816 B ingress=141.09–143.79 GB/s，与 canonical 同口径无系统性回退。详见 `single_inc_native_api_closure_cann91_20260804.md`。 |
| 2026-08-04 | Codex | 产品化 P0-11（单 communicator 整链）：新增 composite backend 和显式 physical Dispatch row→Combine contributor row 映射，同一 Easy communicator/stream 复用 route handle 串联 D→device identity expert expansion→C。W2 3 代与 W4 1 代全 rank 数值 PASS。修复 Release 下 compiler test 的带副作用 `assert` 被删除问题；identity 阶段未冒充 grouped GEMM 实现。 |
| 2026-08-04 | Codex | 产品化 P0-12（fault/recovery + soak）：新增正常路径零额外命令的一次性 device telemetry 注入；W2 第 1 代 Dispatch lane error、第 2 代 Combine producer error 均 fail-closed，第 3 代全 rank 恢复。同一整链 sessions/composite communicator/workspace/route 另连续 256 代 3/3 rank 正确。真实 kernel hang/link 中断仍需 persistent abort mailbox。 |
| 2026-08-04 | Codex | 产品化 P0-13（persistent INC proxy）：新增同机持久 service/client mailbox，worker 整链不再调 INC enqueue 或逐代 W+1 barrier。首轮实测定位 worker readiness 被 INC 后 clear 抹除的竞态，因此固化 prepare→dispatch-ready→combine-ready 三阶段 generation 握手。W2/W4、256 代和两次 telemetry fault 恢复全过；跨主机 transport adapter 仍开放。 |
| 2026-08-04 | Codex | 产品化 P0-14（native expert layout adapter）：从 canonical assignment/reverse contribution 生成 expert-major logical/padded offsets、tokens-per-expert、physical Dispatch permutation 和 Combine inverse permutation；配置零 token expert 不丢失，alignment 非 2 的幂 fail-closed。整链真实经过 alignment=8 device padded buffer，W2/W4 全 rank 数值 PASS；identity expert 未冒充 grouped GEMM。 |
| 2026-08-04 | Codex | 产品化 P0-15（最终 release 回归）：host 产品门禁纳入 composite/layout 后全 PASS；service+layout 整链 W8 9/9 rank PASS。单算子 registered-view 大消息复测 Dispatch 256MiB=123.399–126.917 GB/s（高于 123.339 anchor），Combine slowest-rank=141.251–143.215 GB/s，正确性全过，未观察到系统性回退。 |

---

## 9. Agent 开场白（可复制）

```text
1) 读 docs/inc/report/single_inc_LIVE_STATUS.md（共享：H*、gate、峰值、队列）
2) 读 ACTIVE_HW_PROFILE.md 指向的
   docs/inc/hardware_profiles/<profile>/single_inc_ENV_STATUS.md（本机）
3) 只走 worker-INC-worker；Dispatch 逻辑全量/按目标 rank 物理去重；Combine 默认 INC 完整归约
4) worker D/C 均 `<=floor(live_AIV/2)`（48 上 24 合法）；AIV 只对硬件/拓扑自适应，确定后不随 workload 变；D∥C cohort 不相交
5) ≥128MiB 过带宽 gate；K1 目标 120；其他 case 不回退；正式资格守 H11
6) 本机事实只更新 ENV_STATUS；进度/峰值/约束更新 SHARED
7) 新机：克隆 env/ENV_STATUS_TEMPLATE.md → 新 profile，改 ACTIVE；先等效复测再优化
```
