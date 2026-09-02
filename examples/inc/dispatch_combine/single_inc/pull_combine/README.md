# 单 INC Push-Dispatch / Pull-Combine 协议 v1（实验分支）

本目录实现新协议，不替换也不调用现有 V1 `SingleInc` 后端。当前公共 API
仍指向已验证的旧实现；只有设备 gate 全部通过后，才会另行接入。

## 协议

```text
count:    A0..An ──PUT counts──> INC ──transpose/PUT reply──> B0..Bn
dispatch: A ──PUT one hidden row + route──> INC ──dedup fan-out PUT──> B
compute:  B local experts ──local reduce per (token, B)──> partial rows
combine:  B ──ready descriptor──> INC ──GET/complete/reduce──> accumulator
egress:   ready token runs ──coalesced PUT──> original A ──completion
```

核心不变量：

- A 对每个 token 只向 INC 上传一份 hidden；同一 B 上的多个 expert 共享该份。
- count 矩阵物理路径只能是 worker→INC→worker。
- Combine 不重新传 token ID；它使用 Dispatch 产生的确定性 packed-row 逆映射。
- INC 只保留当前 generation/wave 的临时状态，跨 wave 不保存路由状态。
- ACK 只在相关 GET 或下游 PUT 完成后发布；失败也发布负 ACK 释放发送槽。
- generation、sequence、semantic digest、范围、字节数、保留字段均 fail-closed。
- 严格路径使用 FP32 partial；性能路径允许 FP16/BF16 partial。

## 当前完成度

- 已完成：ABI、路由编译、count 转置、Dispatch/Combine 主机参考状态机。
- 已完成：endpoint-owned Dispatch packet ABI；hidden、token ID 和 CSR 路由
  metadata 同包上传，INC parser 不依赖预构造 `WavePlan`，并按 destination GPU
  去重 hidden、保留全部 expert/weight/ordinal。
- 已完成：设备 Dispatch correctness qualification：worker AIV PUT packet 后发布
  commit 并返回；INC 在线轮询、转置 count、解析 metadata、去重 fan-out，并发布
  source ACK 与 destination completion。路由没有作为 kernel 参数传入。
- 已完成：设备 Dispatch 多 AIV 数据面。每个 source 的 token/assignment wire view
  以两个连续 PUT 批量转发，destination 在本地筛选；hidden 仍只发给实际命中的
  destination。INC 按 `(source,destination,token-lane)` 分片，lane 写入确定且互不
  重叠的 packed hidden 区间，避免逐 token 远端标量控制写。
- 已完成：设备端严格 metadata 校验。digest、CSR 连续性、count 重算、范围、
  非有限权重和重复 ordinal 均在 fan-out 前 fail-closed；损坏包也发布负 ACK/
  completion，不会永久占住槽位。
- 已完成：动态 per-wave journal 主机参考状态机。它只从各 source 的 endpoint
  Dispatch packet 在线学习 token ID、原 owner/row 和 expected contributor bitmap；
  不接收预构造 token plan。Combine 可按任意跨 rank 时序、任意合法 chunk 大小提交
  `[token_ids, locally-reduced FP32 rows]`，整批先校验再原子消费，收齐 contributor
  后才把结果放入 owner egress 队列。W2–W8 共 500 个随机 wave（含零路由 token）
  已通过。
- 已完成：设备 Dispatch 在 fan-out join 后发布独占 64B `DeviceJournalHeader`；
  INC-owned endpoint packet 不被复制，直接作为本 wave 的不可变 journal。这样
  worker 的发送槽可在 ACK 后复用，同时后续 Combine/index kernel 仍能从 INC
  packet 在线恢复 token ID、owner row 和 contributor bitmap。64 MiB case 未出现
  journal 构建导致的性能回退。
- 已完成：空 wave、零路由 token、重复目的 rank、`topk > worker_count`、乱序到达、
  分块传输、提前 egress、ring 回压、负 ACK 与计划生命周期保护。
- 已完成：910B 上高阶 SHMEM GET 正确性和 W2/W4 聚合带宽锚点。
- 已完成：设备 qualification 路径的 descriptor→pull→严格 FP32 reduce→
  source ACK→按 owner 选择性回传，并在 W2/W4 上逐元素验证。
- 已完成：1536-element UB tiled FP32 reduction，以及 contributor MTE2 与向量 Add
  的 ping/pong 流水。
- 已完成：归约结果从 UB 直接 PUT 到 owner，省去 INC 本地 GM store/read 与独立
  egress；64B timeline 可拆分六个设备阶段。
- 已完成：AIV 固定拥有 owner slice，并错开 source 顺序执行局部
  pull→reduce→push；不再等待整 wave pull 完成。
- 已完成：大消息采用独立 pull producer / owner reducer AIV；producer 每完成
  512 KiB 就发布带 generation 的 cache-line ready，reducer 收齐该 chunk 的所有
  source 后立即严格 FP32 reduce 并从 UB 直接 PUT 回 owner。
- 已完成：chunk 按 owner 条带公平调度，所有 owner 尽早获得首块，避免后半 owner
  在单调地址 pull 中长期空转；小于两个 chunk 时自动回退到零握手 owner-slice。
- 已完成：descriptor 或 ready 超时均 fail-closed；成功 ACK 只在 source 已消费且
  egress 完成后发布，失败 ACK 的 `rows_consumed=0`，不会永久占住发送槽。
- 未完成：持久化设备 server、Dispatch journal 驱动的稀疏 token-ID Combine
  **设备数据面**、
  跨 wave 端到端 Dispatch+Combine gate、公共 API 接入。当前 endpoint Dispatch
  是完整单 wave 算子，但 destination 侧 wire-view 筛选还只在 qualification
  harness 中校验，尚未与正式 expert packing kernel 融合。

设备物理 region 按 64B 向上对齐，但 descriptor 中的 `row_count` 和
`payload_bytes` 始终是真实长度。lane 只在 cache-line 边界切分，最后一个物理
span 最多包含 63B padding；因此真实元素数无需整除 worker、lane 或 cache line，
也不会产生跨 AIV false sharing。

## 主机 gate

```bash
cmake -S . -B /tmp/shmem-pull-combine-v1-build \
  -DUSE_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release -DSOC_TYPE=Ascend910B
cmake --build /tmp/shmem-pull-combine-v1-build --target \
  inc_dc_pull_combine_plan_tests \
  inc_dc_pull_combine_dispatch_tests \
  inc_dc_pull_combine_state_tests \
  inc_dc_pull_combine_fuzz_tests \
  inc_dc_endpoint_dispatch_packet_tests -j4
```

随机 gate 固定种子运行 500 个 W2–W8 联合 wave；开发时另以
`-Wall -Wextra -Werror`、ASan/UBSan 和 10,000-wave soak 通过。
endpoint packet 另以固定种子运行 5,000 个随机 packet，覆盖 W2–W8、空 wave、
0–32 token、每 token 0–12 assignments、重复目标 GPU、三种 dtype、非一致 top-k
及随机 hidden payload；metadata digest、generation/sequence、重复 token/ordinal、
非有限权重、范围、payload 大小和 count 重算均 fail-closed。

设备 qualification target：

```bash
cmake --build /tmp/shmem-pull-combine-v1-build --target \
  inc_dc_pull_combine_device_e2e \
  inc_dc_endpoint_dispatch_device_e2e -j4
```

二进制参数为
`<workers> <pe> <ipport> <first_npu> <真实字节数> <每 worker lane>`；
同一 case 需要并发启动 `workers + 1` 个 PE，最后一个 PE 是 INC。它只用于协议
gate，不是公共 API。lane 传 0 时，从实际 `VECTOR_CORE_NUM` 取一半作为 Combine
预算，再平均分给 worker；不写死 910B 的 40 AIV 或 W2/W4。

设备 Dispatch qualification 参数为
`<workers> <pe> <ipport> <first_npu> <tokens> <hidden> <topk> [aiv] [fault]`。
`aiv=0` 自动取实时 `VECTOR_CORE_NUM` 的一半；`fault=1..4` 分别注入 digest、重复
ordinal、非有限 weight 和 count mismatch。当前使用 BF16
hidden，逐字节检查 A→INC packet、count reply、按 destination 去重后的 hidden、
token/assignment wire view、ACK 和 completion。W2/W4 随机形状覆盖 hidden=1–4096、
top-k=1–8 及 top-k>worker，四种损坏包均正确拒绝。该 target 证明完整单 wave
Dispatch 语义闭环；带宽口径包含 packet upload、在线解析、count transpose、
hidden fan-out、metadata wire view、ACK 和 completion，不是纯链路带宽。

### 当前设备 Dispatch 性能（2026-09-03）

同一 HCCS 平面，自动使用 24/48 AIV。`logical_hidden_gb_s` 只累计一次 worker
hidden ingress 和去重后的 destination hidden egress；metadata/control 虽未计入
字节数，但其时间完整包含在分母中。

| 规模 | 每 worker 输入 | 参数 | 完整 Dispatch | 结果 |
|---|---:|---|---:|---|
| W2 | 32 MiB | 4096 token, hidden 4096, top-k 2 | 22.87 GB/s（3 轮均值，CV 0.150%） | PASS |
| W4 | 16 MiB | 2048 token, hidden 4096, top-k 4 | 37.97 GB/s（3 轮均值，CV 0.246%） | PASS |
| W2 | 64 MiB | 4096 token, hidden 8192, top-k 2 | 29.62 GB/s | PASS |
| W4 | 64 MiB | 2048 token, hidden 16384, top-k 4 | 57.27 GB/s | PASS |

加入设备 journal header 后复测同一 64 MiB case：W2=29.50 GB/s、W4=57.63 GB/s；
数据面与前一稳定点一致，header 发布不在关键路径形成可见回退。

这是当前稳定正确检查点，不代表最终 90% gate 已通过。nb 的 W2/W4 单向 raw
参考分别为 56/112 GB/s，而完整 Dispatch 同时包含 ingress、在线路由和 fan-out，
两者不能直接当作相同口径；后续仍需用同一路由分布的实测 roofline 判定。

## nb-borrow 设备锚点（2026-09-02）

卡 0–4 位于同一 HCCS 平面；每次运行前确认 16 卡均无其他 NPU 进程。
数据逐 worker、逐字节校验。数值是 GET+completion 的 host 计时，包含 warmup
传输字节；不是最终算子带宽。

| 规模 | 每 worker | AIV/worker | 聚合带宽 | 相对 1 AIV |
|---|---:|---:|---:|---:|
| W2+1INC | 64 MiB | 1 | 22.41 GB/s | 1.00x |
| W2+1INC | 64 MiB | 2 | 41.74 GB/s | 1.86x |
| W2+1INC | 64 MiB | 4 | 41.76 GB/s | 1.86x |
| W4+1INC | 64 MiB | 1 | 44.95 GB/s | 1.00x |
| W4+1INC | 64 MiB | 2 | 83.49 GB/s | 1.86x |

结论：当前 910B 纯 MTE pull 在 2 AIV/peer 饱和；完整 Combine 中，多出的 AIV
继续并行 FP32 reduction，因此自动策略使用一半实时 AIV 预算，而不是把 transport
饱和 lane 写死成整个算子的上限。低阶 UDMA 只在 Ascend 950 开启，本协议使用
高阶 SHMEM RMA 自动选择 MTE、SDMA 或 UDMA。

### 设备端正确性 gate

同样使用同一 HCCS 平面的 NPU 0–4；输出按 owner 全量逐元素检查，ACK 字段也
逐项检查。

| 规模 | 真实字节数/worker | 边界 | 结果 |
|---|---:|---|---:|
| W2 | 68 B | 非 64B、非 worker 整分 | PASS |
| W4 | 4 B | 三个 owner 为零元素 | PASS |
| W2/W4 | 1,000,012 B | 非 64B、非 worker 整分 | 全部 PASS |
| W2/W4 | 524,292 / 2,500,012 B | 刚跨流水阈值、非对齐多 chunk | 全部 PASS |
| W2/W4 | 1 MiB | UB 向量路径，各连续 5 次 | 10/10 PASS |
| W2/W4 | 64 MiB | 真 producer/consumer 流水，各 5 次 | 40 个 PE 进程全部 PASS |
| W2 tiny + W4 64 MiB | 两个 HCCS 平面并发作业 | 8 个 PE 进程全部 PASS |

### 当前设备 E2E 性能

计时覆盖一次 kernel 的 descriptor 校验、pull、严格 FP32 reduction、ACK 和
selective push；`logical_rma_gb_s=(W+1)×真实字节数/时间`，不是纯链路带宽。
输出中的六项 `phase_pct` 依次是 descriptor、数据面、order、ACK、release、最终
同步。流水版本的数据面包含 producer pull、generation-ready、严格 FP32 reduce
和 UB direct push；各阶段是交叠的，不能再从该区间拆成可相加的独立耗时。

| 规模 | lane/worker | owner-slice 稳定点 | 512 KiB 流水均值 | 代表单次 | 相对稳定点 |
|---|---:|---:|---:|---:|---:|
| W2×64 MiB | 12（自动） | 4.83 ms / 41.72 GB/s | 52.95 GB/s | 3.806 ms / 52.89 GB/s | +26.9%（均值） |
| W4×64 MiB | 6（自动） | 4.22 ms / 79.43 GB/s | 91.81 GB/s | 3.574 ms / 93.89 GB/s | +15.6%（均值） |

五轮 64 MiB soak 中，W2 均值 52.95 GB/s、CV 0.193%；W4 在另一 HCCS
平面并与 W2 同时运行时均值 91.81 GB/s、CV 2.002%，全部逐元素和 ACK 通过。
单独运行 W4 的代表值为 93.89 GB/s。

纯 pull 锚点分别是 W2=41.74、W4=83.49 GB/s（只计算 `W×bytes` ingress）。
把它换成完整 Combine 的传输受限 logical roof：
`pull_anchor × (W+1)/W`，得到 W2=62.61、W4=104.36 GB/s。最终均值分别达到
该实测 roof 的 84.6% 和 88.0%；W4 单独运行达到 90.0%。这个口径不会把
`pull+push` 的 logical bytes 错当成一条物理链路带宽。W2 仍未达到 90%，所以
不能宣称所有规模都通过最终性能 gate，但当前优化没有以牺牲正确性或稳定性换取
峰值。

nb 运行时报告 48 个 vector core，所以 Dispatch/Combine 各占一半时 Combine
自动预算是 24 AIV；自动 lane 为 W2=12、W4=6，而不是沿用旧 40-AIV 环境的
10/5。W2 将 reducer 并发人为降到 W4 同档后只得到 50.54 GB/s，低于自动策略，
因此没有把机器特定的 reducer 上限写入协议。
