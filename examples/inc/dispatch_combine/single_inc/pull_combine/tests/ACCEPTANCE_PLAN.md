# Pull-Dispatch / Pull-Combine V2 验收计划

本文只定义验收口径和 runner 契约，不改变算子协议、kernel 或既有结果。正式结果必须
写入全新的时间戳目录，不能覆盖 V1 或历史 V2 数据。

## 1. 审查结论

现有测试可复用的部分：

- `test_inc_dc_pull_dispatch_v2.cpp` 已覆盖 5000 个随机 packet 和 500 个随机
  layout，包括 W2--W8、三种 dtype、空 token、可变 top-k、重复 `token_id`、
  乱序 arrival 和 unique-destination 去重。
- `test_inc_dc_pull_combine_v2.cpp` 已覆盖 strict dispatch cookie、注册区越界、
  重复 READY、乱序 READY、状态机和 W2--W8 随机 control plan。
- 历史 regular/random/overlap runner 的拓扑检查、独占输出目录、固定 seed、
  resume、最慢 rank 计时与逐样本统计方式可以复用。

不能直接复用的部分：

- 旧 random runner 生成的是 Host Push-Dispatch token plan；V2 路由仍由各 worker
  的 source slot 生成，但 worker 会把 header/metadata prefix PUT 到 INC inbox，
  publication-last READY 后由 INC 在线解析。
- 旧的双腿相加字节口径不适合作为链路带宽。Dispatch 主分子只计算 INC→Worker
  fan-out 下行 hidden；Combine 主分子只计算 Worker→INC reduction partial 上行。
  GET+PUT 字节之和只能作为 aggregate traffic 诊断字段。
- 当前 GET/relay probe 只能作为链路解释，不能替代完整算子 gate。
- 设备层仍需补齐：完整 Dispatch operator runner、非对称 workload、故障后不重建
  session 的恢复、双 journal slot 的任意 D/C 错峰，以及长时间 soak。

## 2. 不变量与统一口径

所有正式 case 必须同时满足：

1. W2 或 W4 的 worker 与 INC 位于同一 HCCS 平面；运行前检查所用卡无其他进程。
2. INC 将实时查询到的普通 AIV 动态二分给 Dispatch/Combine：每方使用
   `floor(live_aiv / 2)`，两组互不相交。nb-borrow 当前 48 AIV 因而是 24+24，
   40-AIV 机器自动变为 20+20；策略不能随 message、top-k 或 route 分布改变。
3. worker 每个 wave 只发布一个 READY。源 slot 在 `SourceConsumed` 前不可复用；
   destination 只能在 `DestinationCompletion` 成功后消费；journal 必须保留到
   Combine terminal state。
4. `token_id` 不是协议主键；主键为 generation-scoped `(owner_rank, owner_row)`。
5. 每个 case 有有限 timeout。超时、进程异常、少一个 rank 的 PASS、静默错数、
   guard 区损坏或错误后不能恢复，均为 FAIL。
6. 性能样本使用完整 device makespan 的最慢 rank，排除初始化、分配、warmup、
   host oracle 和结果拷回。
7. 正式性能为 `warmup >= 3, measure >= 10`；报告 `min/mean/median/stddev/CV/P50/P95`。
   主带宽必须使用单一物理方向的有效数据量与完整算子时间；CV 必须 `<= 5%`。

### 2.1 Dispatch

```text
dispatch_down_bytes =
    sum(one hidden row for every (token, unique destination))

dispatch_bandwidth = dispatch_down_bytes / full_dispatch_makespan
```

完整时间从 launch 前开始，包含 Worker metadata PUT、READY publication、INC 本地
解析、hidden GET、fan-out PUT、completion 与 ACK，直到最慢 rank 完成。历史
`hidden GET + fan-out PUT` 相加得到的 56/112 GB/s nominal 与 51.52/103.04 gate
不再作为主带宽门限；该 aggregate traffic rate 可保留作流水诊断，但不得与单向
链路峰值比较。

### 2.2 Combine

Combine 主分子只使用实际参与 reduction 的 Worker→INC FP32 partial ingress bytes；
owner egress 不与其相加。READY record PUT、Notice、FP32 reduce、owner PUT、
ACK/completion 的时间全部计入完整分母。历史 `partial ingress + owner egress`
口径只保留为 aggregate traffic 诊断，不作为主带宽或单向峰值对照。

## 3. 固定验收矩阵

矩阵只改变输入，不遍历 tile、lane、channel 或 AIV 等调优参数。

### 3.1 边界与任意长度

对 W2/W4、BF16/FP16/FP32（性能只要求 BF16/FP32 canonical path）覆盖：

| 组 | token/row 或 hidden bytes | 目的 |
|---|---|---|
| empty | 每 rank 0 token、0 assignment | 合法空 wave、ACK/completion/journal 回收 |
| scalar | 1 token，2/4-byte hidden row | 最小合法输入 |
| cache tail | row bytes 62/64/66、126/128/130 | 64B control/cache-line 边界 |
| tile tail | row bytes 8190/8192/8194 | 当前 8 KiB transport tile 前后 1 element |
| reduce tail | hidden elements 2047/2048/2049 | Combine 8 KiB vector tile 尾部 |
| count tail | token count 1/2/3/31/32/33/255/256/257 | 循环、CSR 和分块边界 |
| arithmetic | 最大合法 `uint32` count 的 plan-only checked arithmetic | 64-bit size/offset 溢出必须拒绝，不分配 HBM |

FP16/BF16 的 row bytes 必须为 2 的倍数，FP32 必须为 4 的倍数；不合法字节数应在
host validation 阶段明确拒绝，而不是截断或补成另一个逻辑 shape。物理 region 可
64B 对齐，但 oracle 只比较真实元素并检查 padding/guard 未被越界写入。

### 3.2 数据量阶梯

Dispatch 的 size step 指**每个 worker 的 source hidden payload**；runner 根据
实际 unique destination 精确计算 `dispatch_down_bytes`。其他场景的目标字节仍指
主方向有效字节，不是 padding、metadata 或 allocation bytes：

```text
0, 4 KiB, 64 KiB, 1 MiB, 16 MiB, 64 MiB,
128 MiB, 256 MiB, 512 MiB, 1 GiB, 2 GiB, max_hbm_safe
```

- 除“每 worker 恰好 128 MiB 的 `sym_k2_balanced` Dispatch/Combine”外，其他 size 只要求正确性、
  stable、no hang、bounds，并报告带宽和延迟，不适用固定 raw gate。
- `max_hbm_safe`：runner 从所有参与卡的实时可用 HBM 最小值计算。先按 ABI 和
  workspace 公式精确计算每个候选 shape 的 source/destination/journal/partial/
  accumulator/ring 两槽总量，再保留至少 `max(8 GiB, 20% HBM)` 余量。无法满足余量
  时明确 SKIP_RESOURCE，不得把 OOM 记成协议失败，也不得盲目写死 8 GiB allocation。
- 另跑 4 GiB/8 GiB **logical train**（多个有界 wave/ring slot）验证总数据量扩展；
  这不等价于单个 8 GiB resident batch，报告必须区分两者。

### 3.3 workload 分布

每种分布保留生成 seed 和完整摘要；小 case 保留输入，较大 case仅保留可重放 seed
及 SHA-256。

| 名称 | token 数 | 每 token top-k | destination 分布 |
|---|---|---|---|
| `sym_k1_rr` | 各 rank 相同 | 1 | round-robin |
| `sym_k2_balanced` | 各 rank 相同 | 2 | 相邻目的均衡分布；正式 hard gate |
| `sym_dense` | 各 rank 相同 | W | 每个目的 rank 一项 |
| `same_gpu_multi_expert` | 各 rank 相同 | 2W | 每目的 rank 两个 expert；hidden 仍只发一份 |
| `rank_token_skew` | W2=`1:7`；W4=`1:2:4:8` | W | 均匀目的 |
| `empty_sources` | 仅一个 source 非空 | 1/W | 均匀目的 |
| `hotspot_all1` | 各 rank相同 | 1 或多 expert | 全部到 destination 0 |
| `hotspot_90_10` | 各 rank相同 | 1..W ragged | 90% 到一个目的，其余均匀 |
| `ragged_topk` | token 数和 top-k 都不等 | 每 token 0..2W | 固定 seed 随机 |
| `repeated_token_id` | 对称/非对称各一组 | 0..2W | rank 内及 rank 间重复 ID |

热点和单活跃 source 的物理屋顶不同，不应错误套用 dense raw gate；它们强制正确、
稳定、不挂死，并报告 active-link efficiency。每个非对称 case 还要匹配同 W、同
operator、同 dtype、同 payload 档位的 `sym_k2_balanced` 参考，报告带宽退化比例和延迟
放大比例。仅“每 worker 128 MiB”的 `sym_k2_balanced` 是 W2/W4 92% hard gate
workload；其他 size 和 `sym_dense` 压力 case 不套固定 gate。

### 3.4 READY 时序

- W2 穷举 2 种 READY 顺序；W4 host test 穷举 24 种顺序。
- 设备测试至少覆盖正序、逆序、所有循环移位和 8 个固定 seed 随机排列。
- 每个排列叠加：无延迟、一个低 rank 延迟、一个高 rank 延迟、多个 rank 随机
  `0..1 ms` 延迟。
- INC 必须先处理已 READY 的 source，不能被尚未 READY 的低 rank 头阻塞。
- Combine 用另一组独立排列；Dispatch arrival 顺序不能改变 route identity 或最终值。

## 4. 故障与恢复矩阵

每个故障 case 都必须在**同一 SHMEM session、同一已注册 workspace**中紧接两个合法
generation；只有“错误被拒绝 + 两个后继 wave 正确”才 PASS。

Dispatch 注入：

- READY：magic/version/session/placement/generation/sequence/wave/source/region/slot/
  publication 错误，重复 READY，READY 永不到达。
- header：与 READY identity 不一致，dtype/hidden/worker count/record size 错误，
  非 canonical offset，packet/乘加溢出，metadata digest 错误。
- metadata：CSR gap/overlap/越界，destination/expert 越界，ordinal 重复/缺失，
  NaN/Inf weight，重复 source。
- capacity：source/destination row/assignment/journal/重整 workspace 各自少 1 byte/row。
- 生命周期：提前复用 ring slot、旧 generation 的迟到 READY、completion/ACK
  publication 损坏。

Combine 注入：

- dispatch cookie、generation/sequence/wave/ring slot、region/offset/row count/dtype/
  payload bytes/publication 错误。
- 重复 source、缺少 source timeout、contributor 少一项/多一项、destination row
  越界、旧 journal slot 重放。
- worker 在 partial GET 完成前尝试复用 slot；必须由 ACK/credit 阻止。

失败必须 fail closed：无成功 completion、无部分结果被当作有效结果消费、journal
进入 `ABORTED` 并可回收；禁止依赖进程重启恢复。

## 5. 正确性 oracle

每个设备 case 至少检查：

1. Dispatch 对每个 `(owner_rank, owner_row, unique destination)` 的 hidden 逐字节相等；
   同目的多 expert 只有一份 hidden row。
2. `expert_id/ordinal/weight` 和 `DestinationRow` 引用完整一致；动态 row 顺序用
   `route_key` 对齐，不能假定某个 READY 顺序。
3. journal 中每个 contributor 与 Dispatch 实际 unique destination 一一对应；重复
   `token_id` 不能合并不同 route key。
4. Combine 先由 B 对同 GPU 多 expert 做显式本地加权归并，再由 INC 汇总所有
   contributor。小 case 使用 FP64 host oracle；比较时按实际 dtype/reduction 顺序给出
   误差界，同时保留一组可精确表示的 FP32 数据做逐元素 exact check。
5. source ACK、destination completion、Combine ACK 的 generation/cookie/status/count/
   bytes 均正确；所有前后 guard 和未命中 destination 保持 sentinel。
6. oracle、校验和日志生成必须在计时区间外。

## 6. D+C 任意时序交叠

仅交叠不同 wave：`Combine(N, slot s)` 与 `Dispatch(N+1, slot 1-s)`；同一 wave 的
Combine 仍受 Dispatch journal sealed 的数据依赖约束。

每个 W2/W4 在 64 MiB（正确性）、256 MiB 和 1 GiB（性能）覆盖：

- 同时启动；
- Dispatch 领先 `10 us / 100 us / 0.25*Tc / 0.75*Tc`；
- Combine 领先 `10 us / 100 us / 0.25*Td / 0.75*Td`；
- 双方 READY 各自带固定 seed 的 rank 级抖动；
- 连续 4 个 token wave，D/C 在两个 ring slot 上交替，覆盖 fill/drain 和稳态。

每组先测同 shape 的 solo `Td`、`Tc`，再报告：

```text
serial_us       = Td + Tc
concurrent_us   = first_start 到两个 operator 都完成
actual_speedup  = serial_us / concurrent_us
actual_gain     = 1 - concurrent_us / serial_us
ideal_speedup   = (Td + Tc) / max(Td, Tc)       # 无资源竞争的调度上界
gain_realized   = actual_gain / (1 - max(Td,Tc)/(Td+Tc))
```

正确性、有限完成、动态二分后的两组 AIV 不重叠是 hard gate。中大消息应有真实正收益；在双向
roofline 未冻结前，`ideal_speedup` 只作为解释，不虚构固定的理论收益 gate。

## 7. 稳定性与 soak

- 每个正式性能 case：3 warmup + 10 measure；最佳候选再独立进程重复 3 轮。
- 随机 campaign：W2/W4 各至少 1000 个 mixed-shape wave，固定 campaign seed，
  每个 case 派生独立 seed，支持从 `case_id` 精确重放和断点续跑。
- 长时 soak：至少 10,000 wave 且至少 1 小时；两个 ring slot 交替，混合空、小、
  中、大、对称、ragged、hotspot 和 READY 抖动。每 1000 wave 注入一个可恢复故障。
- soak 期间记录进程 RSS、每卡 HBM、错误计数、P50/P95/P99；不能出现单调内存增长、
  generation/slot 泄漏、timeout、hang 或 mismatch。
- 在 `UINT64_MAX` 邻域做 host-only generation/sequence 边界测试；禁止静默 wrap 到
  可与旧 publication 混淆的值。

## 8. Runner 与产物契约

建议最终 runner：`run_pull_v2_acceptance.py`，支持：

```text
--plan-only                 只生成矩阵，不访问 NPU
--suite host|boundary|perf|fault|overlap|random|soak|all
--workers 2,4
--plane A|B|auto
--output <new-directory>    默认目录必须不存在
--resume                    仅 manifest/binary SHA 完全一致时续跑
--case-id <id>              精确重放
--reference-build <path>    Combine 配对回归
```

设备 binary 每个 measure 应输出一个稳定的 JSONL record，而不是让 runner 猜自由文本：

```json
{
  "schema":"single-inc-pull-v2-sample.v1",
  "case_id":"...",
  "operator":"dispatch",
  "sample":0,
  "workers":4,
  "generation":1,
  "logical_bytes":268435456,
  "physical_get_bytes":67108864,
  "physical_put_bytes":201326592,
  "makespan_us":2500.0,
  "bandwidth_GBps":107.374,
  "correct":true,
  "status":0
}
```

目录结构：

```text
<output>/
  environment.json          # host/CANN/driver/topology/AIV/HBM/git+binary SHA
  plan.json                 # immutable case list、seed、gate、CLI/env
  checkpoint.json           # 原子 rename 更新，仅记录完成 case
  cases/<case-id>/
    meta.json
    pe*.log
    samples.jsonl
    result.json
  summary.json              # 全 case min/mean/CV、正确性、gate、回归、失败原因
```

安全要求：

- 进程启动前获取 plane 级文件锁，并检查所用物理卡无进程；同一 runner 可在两个
  空闲平面并行，但两个子 runner 不能分别执行“全 16 卡必须 idle”的竞争检查。
- 顶层 orchestrator 一次性验证所有目标卡空闲后再并发启动，plane A/B 使用不同
  端口、session id、日志目录和锁。
- 捕获 SIGINT/SIGTERM，先终止本 runner 拥有的 PID，等待卡恢复 idle；禁止按名称
  杀死别人的进程。
- 任何已有输出目录默认拒绝写入；`--resume` 也必须核对 plan、git diff、binary、
  CANN 和 topology digest。

## 9. 推荐执行顺序

1. host protocol tests 与 checked-arithmetic。
2. W2/W4 boundary + oracle。
3. 故障注入及同 session 恢复。
4. 每 worker 128 MiB、`sym_dense` 的 W2/W4 Dispatch 正式 gate，以及 Combine
   配对回归。
5. 主 gate 通过后，再测小消息、其他 size、非对称、热点、ragged、重复 token ID、
   乱序 READY；它们不套固定 raw gate，报告相对对称退化。
6. D+C 任意错峰和四 wave steady-state。
7. max-HBM resident、4/8 GiB logical train。
8. 1000-wave random campaign 和 1 小时/10000-wave soak。

前一步有任何 correctness/hang/fault-recovery FAIL，后续性能数字只可诊断，不可作为
达标证据。
