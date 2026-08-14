# 融合算子开发规格（token-wave × 单 INC）

本文件是该融合算子的**唯一开发准则**。开工前整篇读完；实现与本文冲突时，按
**§5 硬约束 > §8 验收口径 > §1–§2 设计规定 > §3–§4 原理** 的顺序裁决。
§6 给出的配置是**已定的默认值**，改动需在本文件里改并写明理由，不得只改代码。

依据：DeepSeek-V4 技术报告 §3.1；DeepGEMM `sm90_fp8_mega_moe.cuh` / `scheduler/mega_moe.cuh`（快照见 `megamoe/nvidia/`）；本仓 `example/`（v1–v4 融合基线）；单 INC 通信库 `../dispatch_combine/single_inc/`。

配套必读：

- 单 INC 通信主逻辑：[`../dispatch_combine/single_inc/QUICKSTART.md`](../dispatch_combine/single_inc/QUICKSTART.md)
- 单 INC 当前进度与基线：[`../dispatch_combine/single_inc/SWEEP_STATUS.md`](../dispatch_combine/single_inc/SWEEP_STATUS.md)
- 共享硬约束 H1–H22：`../../../docs/inc/report/single_inc_LIVE_STATUS.md`
- 融合基线现状与已知坑：[`example/README.md`](example/README.md)

> **2026-08-09 实施决议（优先于下文旧 P0/P1 路径描述）**：用户确认采用方案 B，
> worker 的 D/GMM1/SwiGLU/GMM2/C 必须在同一 MIX kernel 内；INC 使用同一次
> service kernel 中互不重叠的 24/24 AIV cohort。实现落点已改为 `ascend/` 并进入
> `examples/inc/CMakeLists.txt`。`example/` 与独立单 INC 库继续作为语义、性能基线，
> 不再是本实现的代码落点。为了让 commit/credit 与 D∥C 真正在同一 kernel 中推进，
> fusion 侧拥有专用固定深度 transport state machine；但其语义和 gate 仍必须对齐
> `dispatch_combine/single_inc`，不得另造带宽口径。
>
> **2026-08-09 vLLM / baseline 决议（覆盖下文旧 v1/v3/v4 正式基线描述）**：
> 正式算子归因采用同一 route-pack、同一 expert placement、同一权重布局和同一
> GMM1/SwiGLU/GMM2 实现上的 2×2：`serial_shmem / serial_inc /
> fused_shmem / fused_inc`。原生 vLLM 是第五个端到端外部基线，不进入 2×2 理论归因。
> `example/` v1–v4 与 `INC_FUSION_EXEC_SERIALIZE_INC_DC` 继续用于历史复现和压力诊断，
> 但前者计算/路由/拓扑不完全相同，后者含 cohort 等待与流水气泡，二者都不得命名为
> 正式“串行 SHMEM/INC”并用于计算融合收益。可执行检查见
> `ascend/inc_fusion_benchmark.h`。

---

## 1. 目标与范围

### 1.1 目标

保留 MegaMoE 的阶段语义（Dispatch → GMM1 → activation → GMM2 → Combine），把它的
**expert-wave 主调度换成 token-wave（microbatch）主调度**，并把通信从
peer SHMEM / 模拟 switch rank 换成**真实单 INC**。稳态：

```text
INC-Dispatch(mb+1)  ∥  FFN(mb)  ∥  INC-Combine(mb-1)
```

收益分两类，必须分别度量（§4、§8.2）：

- **A 算通交叠**：用 FFN 藏掉 Dispatch / Combine 的暴露时间；
- **B 单 INC 上的 D∥C**：同一时间窗内 `INC-D(mb+1)` 与 `INC-C(mb-1)` 并发推进。

### 1.2 范围

**在范围内**：token-wave 流水；单 INC 接入；activation wave 化；D∥C 并发与其对照开关；tile / expert→block 映射等计算侧优化；单节点 bf16。

**不在范围内**（列为后续项，不得在本轮引入）：fp8 dispatch 量化；per-tile 融合到不落 GM 的 gate_up→SwiGLU→down；多节点；把 expert-wave 作为主调度；任何 host 编排的 collective 作为交付形态。

### 1.3 目标形态

外层（token-wave / microbatch）——**主调度**：

```text
输入 tokens → 切成 MB 个 microbatch（MB ≥ 2，见 §5.2）
稳态三路交叠：INC-Dispatch(mb+1) ∥ FFN(mb) ∥ INC-Combine(mb-1)
触发条件：FFN(mb) 的启动条件是「本 microbatch 路由到本 rank 的行到齐」，
          不是「某个 expert 集合到齐」。
```

内层（单个 microbatch 内的 FFN）：

```text
for each local expert e:            # 行在全部 cube 核上切分
    gate_up = GMM1(x_e, w13_e)      # AIC
    act     = SwiGLU(gate_up)       # AIV，切成 W_act 个 wave（§2.3）
    y_e     = GMM2(act, w2_e)       # AIC
scatter-add(y * weight) → 该 mb 的 combine 输入
```

### 1.4 与现有基线的增量

`example/` v3/v4 已经具备的部分**不要重写**：

| 已存在 | 位置 |
|--------|------|
| microbatch 三级流水（dispatch[mb] / swiglu[mb-1] / combine[mb-2]），叠在 2 个 ping-pong slot 上 | `example/custom_moe_fused.cpp`（v3）、`custom_moe_fused_inc.cpp`（v4） |
| 单核 AIC/AIV 分工：AIV 通信 + SwiGLU，AIC 两次 GEMM（一次一个本地 expert，行切到全部 cube 核） | 同上 |
| 通信经中继节点的形态（v4：rank `W` 管 dispatch、rank `W+1` 管 combine） | `switch_dispatch_mb` / `switch_combine_mb` |
| 路由元数据、启动、显存预算、`MOE_CHECK` 正确性比对 | `example/custom_moe.py` |

本轮真正要做的增量：

1. 把 v4 的**两个模拟 switch rank 收敛成一个真实 INC rank**（W+1 拓扑，D 与 C 共用），通信改走单 INC 通信库；同时把 INC 从 vLLM 的 world 里摘出去，**不加载模型**（§6.5）；
2. 同窗**显式打开 `INC-D ∥ INC-C`**，并实现可切换的 `D/C 互斥` 开关用于归因；
3. **activation wave 化**（§2.3）；
4. 计算侧 tile 优化（§7 的 P3 清单）。

---

## 2. 设计规定

### 2.1 从 MegaMoE 继承什么

**继承**：阶段切分与「数据到齐即开算」的触发式流水；通信与 GEMM 在同一 kernel 内交叠而非算子级串行；细粒度到达计数（MegaMoE 的 `l1_arrival_count`）——在本项目里换成「本 microbatch 行到齐」。

**不继承**：

| MegaMoE 机制 | 原因 |
|---|---|
| `kNumExpertsPerWave` / `get_wave_expert_end_idx()` 的 expert-wave 主调度 | 与 INC 的自然会合粒度（token / microbatch 切片）错位，会同时削弱网内归约优势和 D∥C 的天然错峰（§3.2） |
| NVLink **pull**（读远端 activation） | 单 INC 数据面是 `worker → INC → worker` **push-only**（H1）；不存在 worker 直读远端的合法路径 |
| warp specialization / TMA / TMEM epilogue 融合 SwiGLU | Ascend 无对应物；AIC↔AIV 靠 GM + set/wait 交接，代价模型不同 |
| CUDA Graph 类静态化 | MoE 动态路由下救不了动态 launch；以固定 plan + 动态 invocation 为准 |

代码基线是 `example/` v3/v4，**不是** MegaMoE；MegaMoE 只作重叠质量对照（「能藏掉多少通信」）。

### 2.2 microbatch 流水

- 切分键是 **token 行区间**，粒度由 `MOE_MICROBATCH_SIZE` 给定，必须整除 `BATCH_SIZE`；
- 三级流水叠在 ping-pong slot 上，slot 数 `NSLOT` 与流水级数配套（§5.2 第 2 条）；
- 只同步**真依赖边**：`DISP_DONE(mb) → FFN(mb)`、`FFN_READY(mb) → COMB(mb)`。禁止每 microbatch 的世界级 barrier；
- 每个 mb 的传输**批量 put → 一次 finish**，不要每个细粒度 signal 都硬 fence（会把重叠预算吃光）。

### 2.3 activation 的 wave 化

在一个 microbatch 内把 SwiGLU 切成 `W_act` 个 wave：

- **切分键**：按**行区间**切，且与 expert 边界对齐（不跨 expert 切一个 wave，避免多一层索引）；
- **目的**：让 AIV 的 `act(w0)` 与 AIC 的 `GMM1(w1)` 交叠、`act(w1)` 与 `GMM2(w0)` 交叠，形成 AIC/AIV 两级软流水，而不是「整块 gate_up 算完再整块 act」；
- **交接**：每个 wave 一次 AIC→AIV / AIV→AIC 的 set/wait。**禁止每 tile 一次交接**——当前构建上 AIC→AIV 方向已知会挂（见 `example/README.md`）；
- `W_act` 是可调常量（合法值 1/2/4，默认 2），**不写死**；`W_act = 1` 退化为现状，用作对照。

### 2.4 FFN 内层

- 保持「一次一个本地 expert、该 expert 的行切到全部 cube 核」，直到 P3 的 expert→block 映射改造被实测证明更优为止；
- 本地 reduce（`y * weight` 的 scatter-add）当前每个本地 expert 之间插 `shmemi_barrier_core()`，每 mb 有 E 次 barrier——P3 改成按**目的行**分区后可去掉（§7）。

### 2.5 单 INC 接入方式

- 方案 B 下复用单 INC 的协议语义、路由去重和归约口径，但 transport state machine
  内嵌在同一个 fusion kernel；独立单 INC 库仍负责 roofline/gate，不复制它的 host
  launcher 或按 shape 调参表；
- 接入层用 **Inference API**（prepared、热路径不分配），plan 在 host 侧由 chunk planner 预编译；参考 `../dispatch_combine/common/api/` 与 [`QUICKSTART.md`](../dispatch_combine/single_inc/QUICKSTART.md)；
- Dispatch 语义按 H14：每个 token 的所有 expert assignment 都保留，对同一目的 rank 的多 expert 跨 HCCS 只发一份 hidden，由目标 worker 依 metadata 本地展开；
- Combine 语义按 H15：默认每个 expert-instance 独立 push 到 INC，由 INC 做完整加权归约；rank-local pre-reduce 只能是显式 opt-in 的诊断路径。

---

## 3. 原理背景

### 3.1 V4 与 expert-wave

EP 的瓶颈是 **Dispatch / Combine 通信** 与 **L1 / L2 GEMM** 的串行暴露。V4 的观察是：单层 MoE 内通信总时间往往小于计算总时间 → 把通信藏进计算流水后，系统可容忍更低互连带宽。

```text
Dispatch (comm) → Linear1 GEMM → SwiGLU(+quant) → Linear2 GEMM → Combine (comm)
```

Comet 类方案做两段重叠（Dispatch∥L1、L2∥Combine）；V4 MegaMoE 更细——把专家切成 **expert wave**，wave 内流水：某一 wave 的 experts 通信完成后即可开始该 wave 的计算，稳态下「当前 wave 计算 ∥ 下一 wave Pull ∥ 已完成 expert 的 Combine」三路并行。

代码锚点：

- `megamoe/nvidia/deep_gemm/scheduler/mega_moe.cuh`：`kNumExpertsPerWave`、`get_wave_expert_end_idx()`，先扫完 wave 内 Linear1 再 Linear2；
- `megamoe/nvidia/deep_gemm/impls/sm90_fp8_mega_moe.cuh`：PHASE2 路由交换 → PHASE3 Pull（按 expert 槽）→ GMM1/SwiGLU → GMM2 → Combine；Pull 用 NVLink pull，并对 `l1_arrival_count` 做细粒度到达，使 GMM1 与 Pull 交叠；
- `megamoe/nvidia/deep_gemm/docs/sm90_fp8_mega_moe_warp_report.md`：warp 角色（dispatch / TMA / math）。

报告的 Observations 还给了两条结论：Dispatch 用 pull 以避免细粒度 push 的通知延迟；能否完全隐藏通信取决于 **C/B（算力/带宽比）**，而不是一味加带宽。

### 3.2 为什么单 INC 要配 token-wave

Expert-wave 的「可开始计算」条件是**某个 expert 集合上的 token 都到了**，这适合端侧 pull 与以 expert 为工作单元的 Cube 调度。单 INC 擅长的是另一类会合：

- 动态 count / layout / contributor set 在网络侧建立；
- payload 可与 metadata 交叠进网；
- Combine 侧可对**同一 token 的多 expert partial** 做归约（contributor set），不必先等齐一批 expert。

所以继续用 expert-wave，INC 只能当「更快的 pipe」，网内归约/会合优势用不满，且调度粒度（expert）与 INC 的自然粒度（token / microbatch 切片）错位。换成 token-wave 后，两者的粒度对齐：

```text
batch → 切成若干 microbatch (token-wave)
  for each wave:
      INC-Dispatch(wave)          # 网内边传边建 layout
      local FFN on received rows  # 仍按 expert 聚合成 GEMM，触发条件是「本 wave token 到齐」
      INC-Combine(wave)           # 网内按 token contributor 归约
```

### 3.3 对照

| 维度 | Expert-wave（MegaMoE） | Token-wave / microbatch（本项目） |
|------|------------------------|--------------------------------------|
| 流水切分键 | 本地 expert 子集 | token 子集（mbs） |
| 「可算」条件 | wave 内 experts 的 recv 到齐 | 本 microbatch 路由到本 rank 的行到齐 |
| Dispatch 形态 | 按 expert 槽 Pull / 填充 pool | 按 dest/token 切片，经 INC push relay |
| Combine 形态 | 按 token top-k 片 scatter + 本地累加 | 走 INC reduction（contributor set） |
| 算通交叠（A） | 本 wave 算 ∥ 下 wave Pull ∥ 完成 Combine | `FFN(i) ∥ D(i+1) ∥ C(i-1)` |
| D∥C 交叠（B） | 同核内 warp/阶段分工，非两套 INC | 两套单 INC 会合按 mb 错峰并发 |
| 长尾小 batch | 缩小 `block_m` / wave 宽 | 缩小 mbs |
| 与单 INC | 间接 | 直接对齐会合语义 |

---

## 4. 双收益

细粒度、device-initiated、one-sided 才进得了真重叠；bulk collective / 每 wave 全局 barrier 会把重叠写没。这是 V4 MegaMoE 与 FlashMoE / FlashDMoE、DySHARP、跨节点 megakernel、Ascend HyperParallel-MoE 的共识。

### 4.1 收益 A：计算 ∥ 通信

```text
mb i   :  FFN
mb i+1 :  INC-Dispatch          ⎫ 与 mb i 的 FFN 交叠
mb i-1 :  INC-Combine           ⎭
```

| 要点 | 含义 | 若不做会怎样 |
|------|------|----------------|
| **重叠预算** | 单 mb 的 FFN 墙钟 ≥ 该 mb 相关的网络 RTT/会合时间 | timeline 好看、墙钟不涨 |
| **只同步真依赖边** | 见 §2.2；禁止每 mb 世界级 barrier | 退回 naive 串行 |
| **资源配额** | AIV / 带宽在「打 INC」与「SwiGLU/epilogue」之间显式分配 | 通信一忙 Cube 空等，或反过来 |

### 4.2 收益 B：单 INC-Dispatch ∥ 单 INC-Combine

Dispatch 与 Combine 在方向与负载上不对称；网内 multicast / reduction 省掉流量后，上下行不对称更明显。若把 D 与 C 做成隔离的 bulk 阶段（整层 D 完再整层 C），即使各自更快，端到端仍会卡在较忙的那一侧带宽。token-paced 的 microbatch 流水把互补方向叠进同一时间窗，流量削减才落成墙钟收益。

单 INC 侧的条件：

- D 与 C 是两套会合语义（layout / contributor set 不同），但共享交换机与端侧出口；
- 只要处于**不同 microbatch**（`INC-D(mb+1)` ∥ `INC-C(mb-1)`），协议上可并行推进；
- 网内 reduce 把 Combine 的重活卸到 INC 后，端侧 AIV 更轻，更有余量与下一波 Dispatch 共享时间窗。

```text
时间 →
mb i-1 : ………… FFN ……… INC-Combine ════════════════╗
mb i   : INC-Dispatch … FFN ═════════════════════════╬═ 同窗：C(i-1) ∥ D(i+1) ∥ FFN(i)
mb i+1 :            INC-Dispatch ════════════════════╝
```

### 4.3 两类收益的相互约束

```text
                    ┌─ 过粗同步 / 全局 barrier ──→ A、B 同时被抹平
细粒度 mb 流水 ────┤
                    └─ mb 过碎 → 同步税↑、Cube 效率↓；过粗 → A/B 重叠窗口变窄

                    ┌─ INC D/C 共用出口无配额 ──→ B 抢带宽伤害 A（FFN 等数据）
资源 / 功耗 ───────┤
                    └─ 算+网同时打满 → power throttle → A/B 表观重叠≠墙钟

                    ┌─ layout 不对齐 → 多一次 gather，吃掉 A 的预算
协议 / 缓冲 ───────┤
                    └─ generation / ping-pong WAR → hang 或脏读；B 的并发 D/C 最易踩
```

若 B 实测不显著，按 **同步粒度 → 资源配额 → 协议/缓冲** 的顺序逐项排查，排查完仍不显著则如实记录为结论，不得靠换口径把 A 的收益算进 B。

### 4.4 实现注意点

- **Device-initiated + one-sided**：不要退回「host HCCL 做完再 launch FFN」。
- **Buffer 复用 ≠ barrier 够了**：等待「写完」不等于「读完才可覆写」；MegaMoE combine 双缓冲曾出现 WAR，D∥C 并发时 ping-pong 更敏感。
- **Fence / quiet 过密会隐式串行化**：同 mb 批量 put → 一次 finish。
- **上下行不对称**：只优化 D 或只优化 C 而不做 token-paced 双通道，B 落不到墙钟。
- **Ascend AIC/AIV**：Cube↔Vector 靠 GM + set/wait，通信多在 AIV；flag 不成对、`blockDim` 非法、workspace 未清零都会 hang；INC 完成通知落在哪条核路径要画清楚。
- **动态路由 + 静态图**：以固定 plan + 动态 invocation / 少 launch 为准。

---

## 5. 硬约束

### 5.1 继承单 INC 通信库的红线

完整定义见 `../../../docs/inc/report/single_inc_LIVE_STATUS.md`；下表是对本项目的含义。

| 约束 | 含义 |
|------|------|
| **H1 路径唯一** | 数据面只能是 `worker → INC → worker` push-only。禁止 pull、worker-direct、bypass INC。 |
| **H2 拓扑** | W workers `[0,W)` + **1 个** INC（rank `W`），D 与 C 共用这一个 INC。v4 的「两个 switch rank」只是模拟实现，不是交付形态。 |
| **H4 正确性优先** | 与 torch 参考不一致时，任何性能数字都不算数。 |
| **H5 无 shape 特判** | 传输入口不得按 shape 分支切数据面。 |
| **H9 / H10 鲁棒性** | 覆盖 balanced / all-to-one / 任意 token plan；D∥C 用独立 session、AIV cohort 不相交、无 hang。 |
| **H11 / H12 稳定性** | warmup ≥ 3、measure ≥ 10（并发 overlap ≥ 5），CV ≤ 5%；hang / 超时 = FAIL，禁止「重跑到过」。 |
| **H13 门槛范围** | `< 128 MiB` 只强制正确性与 H9–H12，不设带宽 gate。 |
| **H14 Dispatch 语义** | 逻辑全量、物理去重（见 §2.5）。 |
| **H15 Combine 语义** | 默认 INC 侧完整加权归约（见 §2.5）。 |
| **H16 / H17 / H22 AIV** | worker 侧 D 与 C 各自 ≤ `floor(live_AIV/2)`，并要为 FFN 留核；AIV map 必须确定、可复现，**不得随 K / 字节数 / route plan 自适应**；D∥C 时 cohort 不相交。 |
| **H18 无回退** | 打开 D∥C 或改 tile 策略，不得让已资格化的 Dispatch/Combine case 或 overlap 出现超过噪声的回退。 |
| **H21 开关不绕约束** | `D/C 互斥` 之类的 A-B 开关允许存在，但不得绕过 H1/H4/H10/H14–H20；无效配置必须显式报错而非静默忽略。 |

### 5.2 融合算子自身的不变量

1. **MB ≥ 2**。`example/custom_moe.py` 已断言；`MB=1` 是代码内真实故障（`507015`，5/5 复现），流水也无物可叠。`MOE_ALLOW_MB1=1` 只用于重开该问题。
2. **slot 数与流水级数配套**。现状是 3 级流水叠在 **2 个 ping-pong slot** 上（`slot = mb & 1`），这是能支撑该流水的最少配置。加深流水（例如把 dispatch 拆成 GET 与 gather）必须同时把 slot 提到 3、并把 `mb & 1` 换成 `mb % NSLOT`，否则直接踩 WAR。
3. **produce / consume / reuse 三者配对**：等「写完」不等于「可覆写」。D∥C 并发下这是最高频的 hang / 脏读来源。
4. **归约权威只在一处**（INC 或 owner），不得两边都加。
5. **不得回退 host 编排**：host collective 只能作为对照基线（v1/v2），不是交付形态。
6. **不得引入每 microbatch 的世界级 barrier**。
7. **热路径不分配、不查表切数据面**：plan 在 host 侧预编译。
8. **`__gm__` 指针不得作为参数穿过 `MOE_NOINLINE` 边界**——基址会丢，故障点随访问方式漂移（MTE out-of-range / UB bus error / CCU address check）。应在函数内部从 args 结构体重新推导。
9. **计时前移除 `TEMP-PROBE` 写回**（`grep -n TEMP-PROBE`），它们是热路径上的无条件写。
10. **读 probe 必须用 `-DMOE_DEBUG_PROBE=ON` 构建**，否则默认编译选项会抑制 scalar store 的 cache flush，看到的是任意陈旧子集。

---

## 6. 配置基线

下列取值为**默认已定**。要改，先改本文件并写明理由。

### 6.1 拓扑与资源

| 项 | 取值 |
|---|---|
| 拓扑 | W workers `[0,W)` + 1 个 INC（rank `W`），**D 与 C 共用同一个 INC rank** |
| **开发与验收环境** | **`910b2c-nb`**（npu-borrow）。16× Ascend 910B2C，65536 MiB HBM/卡，CANN `/usr/local/Ascend/cann-9.1.0-beta.3`，驱动 25.0.rc1.1，live AIV 48/卡。代码根 `/export/home/yinjinrun.montyyin/.cursur/projects/default/shmem` |
| 拓扑限制（本环境） | 两个 8 卡 HCCS 岛（Phy 0–7、8–15）。**`W + 1` 张卡必须落在同一个岛内**，因此本环境合法的 W 上限是 **7**；不测 W8+1 |
| 验收规模 | 主力 W = 4，另测 W = 2；**W 必须是运行期参数，不得只对某几个值合法**（H20，见下条「任意卡数」） |
| **任意卡数** | 数据面、ABI、host 侧换算都要支持单机任意 `W`（受上面 HCCS 岛约束）。禁止 `world = W + 2` 之类的写死换算，禁止按 W 查表切策略（H22）。`E` 不被 `W` 整除时（如 W=3/6/7 对 E=64），expert→rank 映射必须支持每 rank expert 数不等，且 FFN 与 combine 的分块不得假设 `E_loc` 相同 |
| 备用环境 | **`910b-yuanmingyu`**（ACTIVE 资格化机，8×910B，CANN 9.1.0-beta.1，48 AIV/卡）。一旦可用即在其上补测，尤其是 **W8**；两机数字不得混报，各自按本机 roofline 判定 |
| 节点 | 单节点；多节点为后续项 |
| INC rank 的模型加载 | **不加载**。INC 只做转发与加权归约，用不到任何权重，也不需要 KV cache。见 §6.6 |
| worker AIV 预算 | D、C 各 ≤ `floor(live_AIV/2)`（H16），且必须为 FFN 的 Vector 计算显式留核。具体数字在 P1 环境确定后**写进报告并冻结**，不得随负载变化（H22） |
| INC AIV | 由 live capability 决定，不写死跨硬件常量；D∥C 时两个 cohort 不相交（H17） |

### 6.2 形状与数据类型

| 项 | 取值 |
|---|---|
| dtype | bf16（非 bf16 构建在编译期被拒）；fp8 dispatch 为后续项 |
| `MOE_MICROBATCH_SIZE` | 默认 512，扫描 {256, 512, 1024}；必须整除 `BATCH_SIZE` |
| MB | 至少覆盖 {2, 4, 8}；MB ≥ 2 是硬约束 |
| 模型 | OLMoE（`allenai/OLMoE-1B-7B-0924`），E = 64、top_k = 8。`E_loc` 由 `W` 运行期决定（W=4 → 16；W 不整除 E 时各 rank 不等，见 §6.1「任意卡数」） |
| backend 语义 | **ALLTOALL 类去重语义**为主（与 H14 一致）；AGRS 仅作对照 |
| `W_act` | 默认 2，合法 1/2/4 |
| `NSLOT` | P0–P2 保持 2；加深流水时同步提到 3（§5.2 第 2 条）。INC 不加载模型后显存不再是约束，见 §6.5 |

### 6.3 接入层

Inference API（prepared、热路径不分配）+ host 侧 chunk planner 预编译 plan。见 §2.5。

### 6.4 buffer 生命周期

现状里「每次 launch 前把路由元数据拷到 host」这一步顺带同步了 torch stream，**掩盖了「某 rank 结束 kernel 后，下一层的 dispatch 覆盖 peer 仍在读的 `send_buf`/`recv_buf`」这个窗口**。新实现不得依赖这个偶然保护：必须显式给出跨层的 buffer 生命周期方案（每核独立 flag slot 的退出握手是候选形态，参考 `shmem_allgather.hpp`；注意 kernel 末尾的全局退出 barrier 已被验证会死锁）。

### 6.5 INC rank 不加载模型

**规定：INC rank 不加载权重、不预留 KV cache。** 它只做转发与加权归约，用不到任何模型参数；`example/` v4 的 switch rank 加载整模是把中继逻辑塞进普通 vLLM rank 的副作用，不是需求。

**做法**：把 INC 从 vLLM 的 world 里摘出去——vLLM 的 process group 只含 W 个计算 rank，INC 是一个独立进程，只参加融合算子自己的 SHMEM 对称世界（`SHMEM_IP` / `SHMEM_PORT` rendezvous）。这样既不需要说服 vLLM 构造一个「不建模型的 worker」，也避免了「INC 在 torch PG 里但不参与 vLLM 发起的集合通信」导致的挂死。

**INC 侧真正需要的显存**：D 的 per-source landing / staging、C 的 per-origin 累加器与归约暂存、路由与 layout 元数据、flag/signal 区，按 `W × NSLOT × mbs × H × sizeof(bf16)` 量级估算（沿用 `custom_moe.py` 的 `shmem_heap_bytes` 口径）。

**这带来的正收益要在设计里用掉**，不要只当省显存：整张卡的 HBM 都可以给 D/C 缓存，因此

- `NSLOT` 可以从 2 提到 3，为加深流水（§7 P3 第 4 项）解除显存约束；
- `MOE_MICROBATCH_SIZE` 的上限放宽，可以在「mb 过碎 → 同步税↑」与「mb 过粗 → 重叠窗口窄」之间取更优点（§4.3）；
- INC 侧可以为 D 与 C 各留独立、互不复用的 buffer，直接消掉 D∥C 并发时最容易踩的 ping-pong WAR（§5.2 第 3 条）。

**摘出 vLLM world 后要一并处理的**：INC 需要每层的路由 / layout 元数据（数据量很小），这条元数据通路要显式设计，不能再依赖「vLLM rank 顺带就有」；`custom_moe.py` 里 `world = W + 2`、`n_compute = world - 2` 一类的 rank 数换算，以及显存预算函数，都要跟着改。

### 6.6 已知限制与风险（写进每份报告）

0. **融合算子工作在通信库的小消息区间**。单 INC 的性能 gate 只在 **≥128 MiB** 生效（H13），更小的消息只保正确性。而本算子每个 microbatch 的传输量是 MiB 级（`mbs × H × 2B`，`mbs=512 / H=2048` 时单源约 2 MiB），且已知存在小尺度悬崖（native Dispatch 在 128 MiB / W2 / K2 实测 82.93 GB/s，同 shape @256 MiB 为 124.48）。**A 与 B 的收益都取决于这一段的效率与 per-call 固定开销**，因此 P0 必须先在本算子真实消息尺寸上测 D/C 的有效带宽与每次调用的固定开销，并把结果作为重叠预算（§4.1）的输入。注：OLMoE `top_k=8` 对应 K8，避开了 Combine K1 与 native Dispatch K1 的已知 gap（见 [`SWEEP_STATUS.md`](../dispatch_combine/single_inc/SWEEP_STATUS.md) §4）。

   **判定锚点用本机的，不要用 ACTIVE 机的 120/123 GB/s。** `910b2c-nb` 每条 HCCS peer raw ≈ 28 GB/s，put-only 聚合实测约 raw 的 76%：W2 ≈ 42.71、W4 ≈ 85.45 GB/s。带宽相对算力更稀缺意味着**重叠预算更紧**（同一 mb 的通信时间更长、更难被 FFN 藏住），所以 `MOE_MICROBATCH_SIZE` 的取值要在本机重新扫，ACTIVE 机上的结论不能预设。
1. **INC 独占一张卡是本方法的成本，不做设备数归一**：INC 是一种优化手段，那张卡的开销已经包含在被测系统里。端到端对照就用**墙钟直比**——`W` 张卡的 native 基线对 `W` 个计算 rank + 1 个 INC 的融合形态，**不按设备数折算，也不把 INC 那张卡算成额外收益**。报告里如实写明拓扑是 W+1 张卡即可，不需要构造 W+1 全参与计算的基线。不加载模型使「与其他作业共卡」在显存上成为可能，但共卡下的干扰与计量口径本轮不做。
2. **单节点限制**：多节点下重叠预算会变紧（跨节点 RTT 变长），本轮结论不外推。
3. **power throttle**：算 + 网同时打满时表观重叠可能不转化为墙钟收益。

---

## 7. 阶段与出口条件

先正确，再拿 A，再显式打开 B，最后做计算侧优化。

| 阶段 | 内容 | 出口条件（全部满足才算过） |
|------|------|--------------------------|
| **P0 统一 2×2 基线 + 小消息量化** | 先冻结 route/placement/weight-layout/compute/timing digest，再由统一执行器实现 `serial_shmem / serial_inc / fused_shmem / fused_inc`；旧 v1/v3/v4 只作历史诊断。并在本算子真实消息尺寸（MiB 级）上单测单 INC D/C 的有效带宽与 per-call 固定开销（§6.6 第 0 条） | 四个 case 两两通过 `inc_fusion_benchmark_validate_factorial_pair`；正确性通过；各自 warmup ≥ 3 / measure ≥ 10、CV ≤ 5%；给出有限 wave 理论上限和「单 mb 的 FFN 墙钟 vs 该 mb 的 D/C 时间」。预算为负时先调 token-wave 容量，不直接宣称融合收益 |
| **P1 接真单 INC** | 用单 INC Dispatch/Combine 替换 v4 的两个 switch rank，收敛到 W+1 拓扑；INC 摘出 vLLM world、不加载模型（§6.5）；外层三级流水保持不变 | 正确性与 P0 一致（§8.1 阈值）；D / C 各自可单独跑通任意 route plan（H9）；无 hang（H12）；**INC 进程既不加载权重也不预留 KV cache，其 HBM 占用与 §6.5 的估算相符**；AIV 分配表已冻结并写入报告；**W=2 与 W=4 均跑通，且用一个非 2 的幂的 W（如 3 或 5，此时 `E_loc` 各 rank 不等）做冒烟，证明 W 确实是运行期参数** |
| **P2 建立 A，再打开 B** | 先确认 `FFN(i) ∥ D(i+1)` 与 `FFN(i) ∥ C(i-1)`；再在同窗允许 `INC-D ∥ INC-C`，并实现 `D/C 互斥` 开关做对照 | A、B 分别给出 §8.2 的两组对照数字；D∥C 下 AIV cohort 不相交（H10/H17）；无 H18 回退 |
| **P3 计算侧优化** | 按下方清单逐项做，每项单独 A-B | 每一项都不得让 A/B 收益或已资格化 case 回退；未达预期的项如实记录 |

P3 清单（按 `example/README.md` 的预期收益排序，不必全做）：

1. **activation wave 化**（§2.3）；
2. **本地 reduce 按目的行分区**——现在每个本地 expert 之间插 `shmemi_barrier_core()`，每 mb 有 E 次 barrier，是 combine 批量化后的主导同步开销。反转索引（由 `recv_group_idx` 经 argsort + counts 在 Python 端构建）后，每个核拥有互不重叠的目的行集合，expert 循环之间无需 barrier，每行只写一次；
3. **每个 cube block 一个 expert**，而不是把同一 expert 切到所有 block——当前估算每次 GEMM 只有 M ≈ 64 行，且每个 block 重载同一份约 8 MiB 的 B；需要在 `local_expert_offsets` 上做负载均衡；
4. **加深流水**（3 级 → 4 级，需 `NSLOT = 3`）；
5. **grouped matmul**（Catlass `08_grouped_matmul` / `02_grouped_matmul_slice_m` 的 block 层）；
6. **backend 按 shape 选择**——只有当 token 到不了每个 peer 时 ALLTOALL 才优于 AGRS；应按 `(top_k, W, E)` 决定而不是靠默认。

---

## 8. 验收与报告

### 8.1 正确性

- `MOE_CHECK=n` 与 torch 参考逐 forward 比对，报 `rel`、`max|y-ref|`、偏差 > 5% 的行数。**要求：偏差 > 5% 的行数 = 0**，`rel` 与 P0 基线同量级。
- 区分故障类型：`call=0` 即大误差 → 索引 / 布局错；`call=0` 很小而后逐次增大 → 累加 / 复用错。
- 覆盖多 epoch 与多 route plan（balanced / all-to-one / 随机 token plan）（H9）。
- **数值一致性**：INC 侧归约与端侧 scatter-add 的累加顺序不同，bf16 下存在可解释的位级差异。判据是「与同一计算实现的串行基线同量级」，不要求逐位一致；但差异**不得随 MB 或 W 单调增大**——那是丢/重复贡献，不是舍入。

### 8.2 性能：统一 2×2 与外部基线

| 调度 / 通信 | SHMEM | 单 INC |
|---|---|---|
| 严格串行 `D→FFN→C` | `serial_shmem` | `serial_inc` |
| token-wave 融合 | `fused_shmem` | `fused_inc` |

四格必须共享 route、expert placement、权重布局、计算实现、token-wave 容量和计时边界；
每轮 route-pack 计时，setup/JIT/分配/权重转换不计时，样本取所有 worker makespan。
原生 vLLM 单独列为第五个端到端外部基线，不能参与下面的内部归因。

- INC 机制收益：`serial_inc` 对 `serial_shmem`；
- SHMEM 上的融合收益：`fused_shmem` 对 `serial_shmem`；
- INC 上的融合收益：`fused_inc` 对 `serial_inc`；
- 综合收益：`fused_inc` 对 `serial_shmem`，并报告是否存在非加性交互。

单独 D/C 窗口的理论上限为
`(Td + Tc) / max(Td, Tc) <= 2`，与 W 不存在直接单调关系。有限 N 个 token-wave 的
理想流水时间为 `Td + Tf + Tc + (N-1)*max(Td,Tf,Tc)`；因此 MB=1 的理论融合收益就是
1×，不能套用稳态 2×。上述值是 shape/route 对应的乐观上界，不是实测收益。

`INC_FUSION_EXEC_SERIALIZE_INC_DC` 只用于 cohort/协议压力诊断：它会引入额外 barrier
与气泡，不等价于严格 `serial_inc`，其墙钟比值不得解释成 B 或融合收益。

### 8.3 度量口径

- **主指标是墙钟**：每层 MoE 端到端时间 + 分解（Dispatch 暴露 / FFN / Combine 暴露 / 同步等待）。按 §6.6 第 1 条**不做设备数归一**，直接与 `W` 卡 native 基线比墙钟。
- 通信侧若引用带宽，沿用 H3 口径：dispatch 分子 = INC→workers 下行总字节，combine 分子 = workers→INC 上行总字节，分母 = 最慢 rank 的完整 device 协议区间，不含 warmup / host setup / 校验。
- 重复性：warmup ≥ 3、measure ≥ 10（并发 overlap ≥ 5），CV ≤ 5%（H11）。
- 小消息：`< 128 MiB` 只强制正确性与 H9–H12（H13）。
- profile：`msprof --output=./out --ai-core=on --aic-metrics="PipeUtilization" torchrun ...`，Perfetto 打开 `msprof_*.json`。**timeline 上的重叠不等于墙钟收益**，结论必须回到墙钟数字。

### 8.4 报告模板

每阶段一份，固定包含：环境（CANN 版本、NPU 型号、live AIV 数、拓扑、W、设备数）、shape 集合（`BATCH_SIZE` / `MOE_MICROBATCH_SIZE` / MB / `top_k` / E / E_loc / H / `W_act`）、AIV 分配表、正确性结果、A/B 两组对照数字、§6.6 的限制、失败与已知问题。组织方式参照 [`../dispatch_combine/single_inc/SWEEP_STATUS.md`](../dispatch_combine/single_inc/SWEEP_STATUS.md)。

---

## 9. 故障诊断

- 卡死先看 stage probe：`example/README.md` 的 stage 表（0-8 顶层、20-28 dispatch、40-46 combine、50-52 reduce、70-77 / 170-177 cube、100-108 / 110 v4 计算 rank、120-133 v4 switch）。所有核停在同一 PC 附近几十字节内 → 空转死锁而非崩溃。
- `ASCEND_LAUNCH_BLOCKING=1 ASCEND_PROCESS_LOG_PATH=./logs` 拿日志；`HCCL_EXEC_TIMEOUT=300` 让超时可回栈；`NPU_ASD_ENABLE=1` 查非法内存读。
- `-DMOE_DISABLE_FFN=ON` 隔离通信流水线（仅 v3/v4 路径有此开关）。
- 日志检索：`grep -E "E[0-9]{5}" ./logs/*/plog/plog-*.log`；`grep -iE "aicore timeout|task timeout|Stream Synchronize failed"`。

---

## 10. 交付物与工程约定

**本轮的交付目标是论文 / 实验证据**：重点是收益 A 与 B 的干净归因与可复现，不是产品资格化。这意味着——

- §5 的硬约束照常遵守（它们同时是正确性与可信度的前提），但**不要求**跑完 H1–H22 的全套资格化 sweep；
- §8 的 A/B 两组对照、度量口径与重复性（warmup / measure / CV）是**必交项**，因为结论要经得起审稿追问；设备数归一按 §6.6 第 1 条不做，但拓扑（W+1 张卡）必须在报告里写明；
- P3 做到「有实测支撑的结论」即可，某项优化实测无收益时如实记录，不必强行做完清单。

工程约定：

- **代码落点**：正式原型在 `fusion_kernel/ascend/`；`example/` 保留为旧 v1–v4
  对照，不再承载方案 B。
- **构建边界**：`fusion_kernel/ascend/` 进入 `examples/inc/CMakeLists.txt`，提供 plan、
  prepared C API、计算 probe 和 W2/W4 E2E；独立 D/C 带宽 gate 仍使用原 launcher。
- **每阶段交付**：可运行代码 + 完整复现命令（含全部环境变量）+ §8.4 报告 + 已知问题清单。
- **文档**：本文件是准则的唯一来源；实现落地后把「实际做成什么样」写进 `example/README.md`，不要在 `megamoe/` 下再开一份说明。

### 10.1 远端开发约定（开发在 `910b2c-nb` 上直接进行）

- **代码根**：`/export/home/yinjinrun.montyyin/.cursur/projects/default/shmem`；本地与远端之间只用既有同步脚本，不要在两边各改一份。
- **环境**：`source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh`；确认 `npu-smi` 看到的驱动是 25.0.rc1.1、live AIV 48/卡。跑之前先确认另一批 NPU 不在被别人占用。
- **选卡**：用 `DEVICES=` 显式给物理 id，**不要**用 `ASCEND_VISIBLE_DEVICES` / `ASCEND_RT_VISIBLE_DEVICES`（前者不重映射、后者会把物理 id 对 MemFabric 隐藏，导致 `shmem_init_attr` 报 `svm advise failed`）。所选卡必须同岛（Phy 0–7 或 8–15）。
- **通信库侧的正式数据走 launcher**：`../dispatch_combine/scripts/single_inc/`（带 NPU 锁与空闲检查）。手跑二进制的结果不作为证据。
- **回传**：每次上机跑完，把复现命令、原始日志与 §8.4 报告一并落到仓库里再同步回来；失败的跑次同样要留档（H11 禁止「重跑到过」）。
- **卡死时**：先按 §9 取 stage probe 与 plog，不要直接重跑掩盖。

---

## 11. 代码阅读顺序

1. DeepSeek-V4 技术报告 §3.1（`deepseek-v4技术报告.pdf`，仅在 `fusion_kernel/` 根目录保留这一份）
2. `megamoe/nvidia/deep_gemm/docs/sm90_fp8_mega_moe_warp_report.md`
3. `megamoe/nvidia/deep_gemm/scheduler/mega_moe.cuh`
4. `megamoe/nvidia/deep_gemm/impls/sm90_fp8_mega_moe.cuh` 文件头 PHASE 注释 + Pull/GMM 交叠
5. `example/README.md` + `example/custom_moe_fused_inc.cpp`（v4）+ `example/custom_moe_fused.cpp`（v3）
6. [`../dispatch_combine/single_inc/QUICKSTART.md`](../dispatch_combine/single_inc/QUICKSTART.md)（单 INC 的 Dispatch/Combine 主逻辑与同步点）
7. `megamoe/ascend_related/vllm_dispatch_ffn_combine/dispatch_ffn_combine_kernel_report.md`（知差异，勿混同：它是 per-expert / group 流水，不是 expert-wave，也不是 INC）

## 12. 来源

| 快照 | 原始路径 |
|------|----------|
| NVIDIA kernels | `DeepGEMM_sm90/deep_gemm/include/deep_gemm/{impls,scheduler,layout}/…` |
| NVIDIA heuristics | `DeepGEMM_sm90/csrc/jit_kernels/heuristics/mega_moe.hpp` |
| Ascend 近邻融合核 | `vllm-ascend/csrc/mc2/dispatch_ffn_combine/` |
| 本仓融合样例 | `fusion_kernel/example/` |

相关公开讨论（原理对照，非本仓代码）：FlashMoE / FlashDMoE（单核 + device-initiated）；DySHARP（动态网内计算 + token-centric 流水消解 D/C 不对称）；跨节点 megakernel 通信隐式串行化；Ascend HyperParallel-MoE（AIC/AIV 交错 + device-side SHMEM）；DeepGEMM MegaMoE combine buffer WAR / fence 修复。

---

## 13. English spec

**Goal.** Keep MegaMoE's stage semantics (Dispatch → GMM1 → activation → GMM2 → Combine), replace its *expert-wave* scheduler with a *token-wave* (microbatch) scheduler, and replace peer SHMEM / simulated switch ranks with the real **single INC** transport. Steady state: `INC-Dispatch(mb+1) ∥ FFN(mb) ∥ INC-Combine(mb-1)`; FFN starts when *this microbatch's* rows have landed, not when an expert set is complete.

**Two benefits, measured separately.** **A — compute∥comm** against a serial `D→FFN→C` baseline (v1, optionally v3). **B — D∥C on one INC**: with A already enabled, compare *concurrent INC-D/INC-C* against *forced D/C mutual exclusion*. "Faster than naive HCCL" is not evidence for B. Wall-clock is the primary metric and is **not** normalized by device count: the INC rank's dedicated NPU is a cost of the method, already inside the measured system, and is never credited as extra gain. Compare directly against the `W`-card native baseline and disclose the W+1 topology in the report.

**Increment over the existing baseline.** v3/v4 already provide the three-stage microbatch pipeline on two ping-pong slots, the AIC/AIV split, and per-expert GEMMs — do not rewrite them. The new work is: collapse v4's two simulated switch ranks into one real INC rank (W+1, D and C share it), explicitly enable `INC-D ∥ INC-C` plus a mutual-exclusion switch for attribution, wave-split the activation (`W_act`, default 2, aligned to expert boundaries, one AIC/AIV handoff per wave — never per tile), then tiling work.

**Hard rules (`single_inc_LIVE_STATUS.md`).** Push-only `worker → INC → worker` (H1; no pull, no worker-direct). One INC rank at `W` (H2). Correctness gates performance (H4). Dispatch keeps every assignment but sends one hidden copy per destination rank (H14); Combine reduces in the INC by default (H15). Worker AIV ≤ `floor(live_AIV/2)` per direction, deterministic, never workload-adaptive, cohorts disjoint under D∥C (H16/H17/H22). warmup ≥ 3, measure ≥ 10, CV ≤ 5%; hang/timeout = FAIL (H11/H12). No regression on qualified cases (H18). A/B switches must not bypass constraints (H21).

**Fusion-side invariants.** `MB ≥ 2` (MB=1 is a real, reproduced failure). Slot count must match pipeline depth: today three stages ride on two slots (`slot = mb & 1`); deepening requires `NSLOT = 3` and `mb % NSLOT`. Every produce/consume/reuse transition needs a matching lifetime handshake — "written" ≠ "safe to overwrite". No host-orchestrated collective as a deliverable, no per-microbatch world barrier, no hot-path allocation. Never pass a `__gm__` pointer through a `MOE_NOINLINE` boundary. Strip `TEMP-PROBE` writes before timing; build with `-DMOE_DEBUG_PROBE=ON` to read probes. Do not rely on the metadata copy-to-host to protect peer buffers across layers — provide an explicit buffer-lifetime scheme.

**Phases.** P0 reproduce v1/v3/v4 baselines → P1 swap in the real single INC (W+1), freeze the AIV map → P2 establish A, then enable and attribute B → P3 activation waves, destination-partitioned reduce, expert-per-cube-block mapping, deeper pipeline, grouped matmul, backend selection.

**The INC rank loads no model.** It only relays and reduces, so it needs neither weights nor KV cache — v4's switch ranks load the full model merely because the relay logic was stuffed into ordinary vLLM ranks. Keep the INC out of vLLM's process group entirely: vLLM's world is the W compute ranks, and the INC is a separate process that joins only the fusion kernel's SHMEM symmetric world. Spend the freed HBM on the data path: raise `NSLOT` to 3 to unblock a deeper pipeline, widen the usable `MOE_MICROBATCH_SIZE` range, and give Dispatch and Combine separate non-reused buffers so concurrent D∥C cannot hit ping-pong WAR. Routing/layout metadata must then reach the INC through an explicit path, and the rank arithmetic in `custom_moe.py` (`world = W + 2`) has to change.

**Environment and scale.** Development and qualification happen on **`910b2c-nb`** (16×910B2C, 64 GiB/card, CANN 9.1.0-beta.3, driver 25.0.rc1.1, 48 AIV/card; code root `/export/home/yinjinrun.montyyin/.cursur/projects/default/shmem`). Its two 8-card HCCS islands (Phy 0–7, 8–15) force `W + 1` onto one island, so `W ≤ 7` here and W8 is not measured; main configurations are W=4 and W=2. `910b-yuanmingyu` (ACTIVE, 8×910B) is the fallback environment — re-measure there when it frees up, especially W8, and never mix numbers between machines. **`W` must be a runtime parameter**: no `world = W + 2` style hardcoding, no W-indexed strategy tables (H22), and when `E` is not divisible by `W`, the expert→rank map must tolerate unequal `E_loc` per rank. Use per-machine roofline anchors — on nb, per-HCCS-peer raw ≈ 28 GB/s and measured put-only aggregate is ≈ 42.71 (W2) / 85.45 (W4) GB/s, so the ACTIVE machine's 120/123 GB/s numbers do not apply and the overlap budget is tighter.

**Deliverable is paper-grade evidence**, not product qualification: honour the hard rules in §5, but the required artifacts are the A/B comparisons, the metric discipline (warmup/measure/CV) and reproducible commands — not a full H1–H22 sweep. Record optimizations that turn out not to help.

**Known limitations to state in every report.** The INC process still occupies a whole NPU (W compute ranks need W+1 devices). That card is a cost of the method, not a credit: do **not** normalize by device count — compare wall-clock directly against the `W`-card native baseline and state the W+1 topology. Not loading the model makes co-location memory-feasible, but co-located interference is out of scope this round. The fusion kernel operates in the sub-128 MiB regime where the transport has no bandwidth gate and a known small-size cliff — quantify it in P0. Single node only. Power throttling can make apparent overlap not translate into wall-clock gains.
