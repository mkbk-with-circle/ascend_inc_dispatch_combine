# Multi-INC Combine `service_v2` 细粒度流式设计

状态：**Proposed / plan 增补设计**  
用途：记录下一代 Multi-INC Combine 流水线，作为
`paired_multi_inc_combine_service_performance.plan.md` 的设计补充和后续实现对照。  
约束：本文不覆盖原 plan，不删除现有 `packed_epoch_v1`，也不改变“带宽达到
100 GB/s 后才进入 Dispatch/Combine overlap / MC15”的顺序。

## 0. 2026-07-23 协议评审决定

决定是：**替换 ingress/flow-control 协议内核，不重写 combine 归约核心**。

当前 `frame_v3` 已有正确的 `(src_worker, dst_inc, owner)` SPSC ownership、C8/C20
owner 分片、generation/cookie、owner-private context/ready queue、确定性 ordinal reduce、
vector reduce 和 TX/oracle。这些保留为 correctness oracle/fallback。

需要替换的是 per-slot commit/AQ/backoff/signal-hint/channel-seal 闭环。当前 AQ 在初始化时
把逻辑 active channel 全部预置入队，owner 仍需猜测每个 next slot 是否到达；没有单调
committed tail。30-sample frozen 运行虽然 30/30 功能正确，但
`p50=2414.96 us, p95=5369.94 us, max=6390.8 us, min=43.64 GB/s`，证明单次
`130.43 GB/s` 不能代表稳定性能。继续扫描 backoff 参数不能消除这个结构性反馈环。

新路径命名为 **schema4 ingress revision 2 / frame_v4_batch**，独立 feature flag，核心规则：

- 64-bit absolute producer committed-tail / consumer returned-head；
- payload quiet 后发布 64B descriptor，再推进 committed-tail；doorbell 只作提示；
- owner-homogeneous 64--256 KiB microbatch；
- owner 顺序消费 `[head, committed_tail)`，MTE drain 后立即返 input credit；
- window ticket/expected ordinal 是完成真值；不再依赖 final-credit channel seal；
- per-window active-source registry；8 rank 直接 DRR，稀疏大 rank 成本只随 active source
  增长；未经 device atomic gate 不引入共享 MPSC notification；
- generation-tagged DONE tombstone 保留到 window credit，阻止 late duplicate 重开 context；
- TX 按 byte/deadline batch NBI，quiet 后才能 completion/context credit。

FG0 严格 host gate 已在
`docs/inc/report/fg0_service_v2_protocol_gate.json` 记录 `host_contract_pass=true`，
覆盖 2^32 sequence crossing、late duplicate、stale/future、tail-before-payload、
lost doorbell fallback、ring backpressure、window reuse 和 256 active source。它明确记录
`device_correctness_pass=false`、`device_formal_pass=false`；在 FG1/FG2 device gate 前不得
替换默认路径或解冻 P100。

## 1. 当前结论

当前 `packed_epoch_v1` 已证明大量协议与正确性基础，但不适合作为最终性能路径：

- Worker 固定 C8，INC Combine 固定 C20；
- Switch 只有 bid0 消费全部 `(src_worker, lane)` channel；
- bid0 等整个 INC epoch 的 `unique_arrived == expected_arrived`，然后发布一个全局
  `ingress_ready_generation`；
- 其余 AIV 在全局 latch 后才开始归约；
- input ring credit 直到全 epoch reduce/TX/ACK 后才返还；
- TX 仍存在 per-result `putmem + quiet`；
- 当前 group size 的 packed layout 仍有 `max_group_size=8` 限制。

PERF3-V 已从 device 证明 vector 分支启用，但当前大消息约 0.26 GB/s，Worker 约
1 s/epoch，25 epoch ring-wrap 撞到 30 s timeout。这个结果不应通过增加 timeout
包装成 PASS。旧路径应冻结为 correctness oracle/fallback；新路径使用独立 feature flag、
schema 和 gate。

## 2. 设计原则与成熟实现对应

成熟网内归约实现的共同原则是：有限 active slot/pool、generation/sequence、seen
bitmap、credit/self-clocking、确定性 ownership 和流式完成释放，而不是扫描完整路由表或
等待整轮数据全部到齐。

- SwitchML 使用有限 aggregation slot pool、seen bitmap、双 pool/version 和按 BDP
  配置的发送窗口；其 worker 数据面按 slot/tensor chunk 分片，避免共享状态。
- ATP 使用 `<job, sequence>` soft state、bitmap、counter 和 timestamp 管理固定大小的
  aggregator。
- NVIDIA SHARP 强调 streaming aggregation，并用多个 aggregation tree 分散负载。
- NetReduce 保留可靠传输/拥塞语义，并使用 message-level credit 防止 buffer overflow。

本文借鉴这些原则，但针对本项目的 paired INC、固定 AIV、动态 MoE route、FP16/BF16
和 Ascend SHMEM 重新设计。

参考：

- SwitchML NSDI'21: <https://www.usenix.org/system/files/nsdi21-sapio.pdf>
- ATP NSDI'21: <https://www.usenix.org/system/files/nsdi21-lao.pdf>
- NVIDIA SHARP: <https://networking-docs.nvidia.com/sharpum/380>
- NetReduce: <https://arxiv.org/abs/2009.09736>

## 3. 选择：C20 owner-sharded run-to-completion

`service_v2` 首选 **20 个对称 result-owner AIV**，而不是长期固定成
`RX/Reduce/TX` 三个角色池。

原因：

1. 每个 result/context 只有一个 AIV writer，消除共享 pending counter、atomic RMW、
   MPSC ready queue 和跨 AIV state migration；
2. 20 个 AIV 都能同时接收、归约和回传，不产生固定阶段分区失衡；
3. 与 SwitchML 的 run-to-completion、按 slot/chunk 无共享状态分片原则一致；
4. workload、top-k、hidden 或 RX/TX 比例变化时，不需要重新分配 AIV 数量；
5. AIV 数量始终固定为 C20，不随 Worker/group size 增长。

如果后续可信 profile 证明 `quiet` 长时间阻塞 owner event loop，才允许增加一个固定的
TX 子角色 profile；不能在第一版同时引入跨 AIV TX handoff。

## 4. 总体数据流

```text
Home Worker / Dispatch
  为每个 reduction/tile 分配 ticket、slot、ordinal、owner
  route cookie 随 token 到 expert，再随 expert output 返回
                         |
Worker Combine C8        |
  按 (dst INC, owner) bucket
  owner 对应的唯一 lane = owner % 8
  owner-homogeneous microbatch
                         |
                         v
SPSC channel[(src_worker, owner)]
  payload -> quiet -> descriptor/tail commit
                         |
                         v
INC owner AIV[0..19]
  DRR 消费本 owner 的 group_size 个 source channel
  MTE/UB tiled copy -> INC-owned contribution slab
  copy drain 后立即返 input-ring credit
  更新 owner-private generation/seen bitmap
  seen == expected -> owner-local ready FIFO
  ordinal 顺序 vector reduce
  NBI TX microbatch -> one quiet -> completion commit
                         |
                         v
SPSC completion[(owner, worker_lane)] -> paired Worker recv/output
```

这个路径没有：

- bid0 全局 RX；
- 全 epoch `unique_arrived`；
- 全局 `ingress_ready`；
- per-result readiness 扫描；
- 全 epoch `reduce_done` join 才返 input credit；
- 多 AIV 共享 result state。

## 5. 路由合同与有界近无状态 service

INC 不保存完整 assignment route table，也不保存与整次 operation 的 result/token
总数成比例的常驻状态。合同拆成两层：

- **Operation header（Worker 侧持有）**：shape、dtype、总 result 数、output layout 和
  framework cookie；INC 只在接受 window 时校验其 digest；
- **Window ticket（INC 短期持有）**：只描述当前受 credit 约束的 active window，
  其 expected-count exception、owner metadata、context 和 payload slab 总量受
  `window_W/context_pool/BDP` 上限限制，与整次 workload 大小无关。

最低路由合同是 window ticket 和自描述 contribution：

```cpp
struct CombineTicket {
    uint64_t session_id;
    uint64_t generation;
    uint64_t operation_id;
    uint64_t operation_digest;
    uint32_t window_id;
    uint32_t window_base_result;
    uint32_t window_result_count;
    uint32_t dtype;
    uint32_t hidden_elems;
    uint32_t default_expected_count;   // 固定 top-k 时为 O(1)
    uint64_t expected_exceptions_off;  // 仅 active window 内的 zero/drop/动态 top-k
    uint64_t owner_exceptions_off;     // 算法 owner 之外的 window-local exception
    uint64_t recv_layout_off;
    uint64_t completion_layout_off;
    uint64_t plan_digest;
};

struct ContributionEntry {
    uint64_t ticket_generation;
    uint32_t result_id;
    uint16_t ordinal;       // [0, expected_count)
    uint16_t owner;
    uint32_t dst_rank;
    uint32_t dst_token;
    uint32_t hidden_tile;
    uint32_t tile_elems;
    float weight;
};
```

约束：

- ticket 是 expected count、shape、dtype 和 output layout 的唯一真值；entry 不能自行改变
  expected count；
- dispatch 把 `{ticket_generation, result_id, ordinal, owner}` 作为 return cookie 随 token
  传播；expert 不解释 cookie，只在 combine 时原样带回；
- 没有前序 multi-INC dispatch 时，worker runtime 根据正常 combine route 构造同样的
  ticket/cookie；
- uniform top-k 只需要一个 `default_expected_count`；variable top-k 只传 active
  window 内的 count exceptions，不传 O(assignments) 完整表；
- owner 默认由 `{operation_digest, result_id/tile_id, virtual_shard_map}` 确定性生成；
  只有不均匀例外需要 window-local owner exception，禁止常驻整张 owner map；
- generation-tag lazy reuse，正常 epoch 不 memset 整张状态表；
- 首个 contribution 可懒初始化 context，但必须与 ticket 一致，否则 fail-closed。

### 5.1 “近无状态”的可验收定义

INC 允许的常驻状态只有 service config、pair map、固定 AIV 队列池、窗口/通道
池和 telemetry；operation 结束后不保留 route/result 语义。形式谓词为：

```text
persistent_route_bytes == 0
resident_operation_state_bytes <= fixed_service_state
                                + active_window_capacity
                                + measured_BDP_and_burst_margin
active_context_count <= configured_context_pool_capacity
```

把总 result 数放大 2x/4x 而 window/BDP 不变时，INC 峰值状态不得随之线性增长。
如果 metadata 或 expected-count vector 无法装入当前 window，Worker 必须分窗口发布，
而不是扩大 INC 上的整 operation 状态。

## 6. Ownership 与负载均衡

### 6.1 Worker ownership

- Worker 固定最多 C8；
- `worker_lane = owner % 8`；
- 每个 `(src_worker, dst_inc, owner)` 只有一个 producer lane；
- 因此远端 `(src_worker, owner)` channel 是严格 SPSC；
- Worker 对 `(dst_inc, owner)` bucket 使用 byte/cost-aware DRR，不能按 entry 数简单均分。

### 6.2 INC ownership

- 每个 reduction/tile 只有一个 immutable owner；
- owner 独占 context、bitmap、payload slab、ready FIFO、egress 和 TX completion state；
- 同一 ticket 内禁止 work stealing 或半途迁移 owner；
- uniform 大 batch 可 round-robin 或混合 hash；禁止裸 `result_id % 20` 成为唯一长期策略；
- dst/home Worker 是唯一 ticket 分配者，可按
  `cost = tile_bytes * expected_count + tile_bytes(TX)` 做窗口内 greedy/LPT 或 two-choice；
- 可先映射到 256 个 virtual shards，再平衡映射到 C20，owner 写入 return cookie。

### 6.3 Decode/低并发

当 result 数不足以覆盖 C20 时，将 hidden 切成多个 vector-aligned tile，使 ready task 数至少为
活跃 reducer 数的 2--4 倍。每个 tile 是独立 aggregation task；paired Worker 在所有 tile
completion 到达后标记 token 完成。

大 prefill/大 batch 默认按整 result 分片，避免不必要的 tile control 开销。

## 7. Owner-private context

每个 context 独占至少一个 64B cacheline，并只由 owner AIV 修改：

```text
FREE(old generation)
  -> OPEN(current generation)
  -> READY(seen_mask == expected_mask)
  -> REDUCING
  -> TX_PENDING
  -> DONE
  -> FREE(next generation after credit)
```

建议字段：

- generation / state；
- expected count 或 expected mask；
- seen mask（top-k <= 64）或动态 bitset offset；
- per-ordinal contribution slab offset / weight；
- dst rank/token/tile；
- first/last progress timestamp；
- error code / first missing ordinal。

当前正式覆盖 top-k 1/2/4/6/8，可先走 64-bit bitmap；top-k > 64 必须走动态 bitset，不能
静默截断或写死。

归约必须按 ordinal 固定顺序读取 INC-owned slab，不能按到达顺序累加；这样训练结果不会因
网络时序变化而漂移。

## 8. Payload、credit 与内存顺序

### 8.1 为什么先复制到 INC-owned slab

第一版 `service_v2` 不让 packed-ring payload 生命周期跨越 reduce/TX：

- Worker 继续发送 owner-homogeneous 64--256 KiB microbatch，保留大消息效率；
- owner 使用真正的 MTE/UB tiled copy 将每个 entry 搬到稳定的
  `[ticket][result_slot][ordinal][tile]` slab；
- MTE drain 后即可推进 input channel consumer/credit；
- reduce/TX 只引用 INC-owned slab，不再依赖 Worker ring slot；
- INC 显存换来了更短的 ring lifetime、更小 channel cap 和显著更少的 release 竞态。

旧路径约 134 ms 的 ring->ingress copy 是标量 copy，不代表 MTE/UB copy 的性能。新 copy
必须有独立正确性和带宽 gate。

### 8.2 顺序规则

Producer：

```text
gather payload
  -> writer release
  -> putmem/NBI payload
  -> quiet (一次 microbatch)
  -> descriptor + tail commit signal
```

Owner：

```text
acquire committed descriptor
  -> validate ticket/gen/bounds/ordinal
  -> MTE copy payload
  -> MTE drain
  -> advance input credit
  -> update seen bitmap
```

TX：

```text
vector output release
  -> batch putmem_nbi
  -> one quiet per byte threshold/deadline
  -> completion descriptor/doorbell
  -> release output/context credit
```

任何 result-ready/completion 都不能先于 TX quiet；任何 channel credit 都不能先于本地 MTE
完成；任何 context/window reuse 都不能先于对应 completion/abort ACK。

## 9. 滑动窗口与防拥塞

容量按实测 BDP、burst 和合法 shape 动态计算，而不是按整个 workload 或固定 magic size：

```text
pool/window bytes >= measured BDP + max producer burst + safety margin
```

建议初始 window depth W=4/8，并通过 gate 选择；所有乘加使用 checked arithmetic。

分层背压：

1. **Channel credit**：每 `(src, owner)` 独立 head/tail/credit；满时只暂停该 channel；
2. **Owner context credit**：某 owner 的 slot pool 满时，只暂停发往该 owner 的 bucket；
3. **Ticket/window credit**：未收到 generation credit 不得复用 window；
4. **TX credit**：output/context 只在 `NBI -> quiet -> completion` 后释放；
5. **DRR**：owner 对 source channel 使用 deficit round-robin 和 byte quantum；hot/slow source
   不能阻塞其它 source；
6. **Deadline flush**：TX 达到 byte threshold 或 decode latency deadline 即 flush，兼顾吞吐和
   decode 延迟。

所有 producer cursor、consumer cursor、owner ACK、abort ACK、telemetry 使用不同 64B
cacheline；禁止共享 RMW mask。

group size 增大时：

- AIV 仍固定 C20/C8；
- channel/layout/heap 根据 runtime group size 动态增长；
- 不得在 hot path 每轮扫描 `group_size` 个 source channel；每 owner 只消费
  **active-channel registry/notification queue** 中的 channel，空 channel 不产生轮询成本；
- 必须移除 `kCombineBw05MaxGroupSize=8` 的 clamp/fallback；
- 小 group 可以使用 bitmap + `ctz` 查找 active channel；大 group 使用固定分片、
  有界 notification ring 或 hierarchical active-batch ticket；
- 若使用 MPSC notification ring，仅允许在“空->非空”通知层用已上卡证明的
  atomic reservation + per-slot sequence；reduction context/bitmap/credit 仍必须 owner-private，
  禁止共享 pending counter；
- 若 SHMEM atomic 语义未证明，必须回退到确定性分片的 SPSC active-batch queue，
  不得用全量轮询冒充；
- device 正确性覆盖 group 1/2/4/6/8/>8，host/model 覆盖至少 16/32/64/128/256
  以及 checked-overflow 负例。

“workload 增大不明显减慢”指有效 GB/s 和每字节成本保持稳定；总时间随 useful bytes 线性
增长是正常的。正式 scale gate 应要求 2x/4x workload 的 useful GB/s 相对稳定区下降不超过
10%，而不是要求绝对延迟不变。

## 10. Event loop

每个 owner AIV 使用有预算的协作循环，不能在单一阶段无限等待：

```text
consume up to RX_DESC_QUANTUM / RX_BYTE_QUANTUM using DRR
copy/drain completed microbatches and return channel credit
process bounded READY_QUANTUM reductions
issue bounded TX_BYTE_QUANTUM NBI
flush TX on threshold or deadline
publish progress / inspect abort generation periodically
repeat
```

queue 为空不是错误；正常路径不使用固定 spin timeout 判断失败。watchdog 根据 generation 和
live-progress 判断 stall。

## 11. 故障与 fail-closed

- stale generation：丢弃并计数，不修改当前 context；
- future/unaccepted ticket：NACK/隔离，不能覆盖当前 slot；
- duplicate ordinal：owner 通过 bitmap 唯一检测；同一重传可幂等丢弃，冲突元数据 abort；
- bounds/dtype/hidden/owner/ticket digest 冲突：立即 protocol abort；
- incomplete result：绝不 reduce/TX/result-ready；
- stall report 必须精确到
  `{ticket, owner, src_channel, head, tail, result_id, missing_ordinal}`；
- abort generation 通过每 owner 独占 line 发布/确认；20/20 owner quiesce ACK 前不得复用
  window；
- 一个 stalled result 只占自己的 context，不得阻塞其它 ready result。

## 12. 与现有实现的关系

保留并复用：

- C8/C20 资源边界及 Dispatch/Combine 物理 AIV 隔离要求；
- SPSC cursor 的 single-writer cacheline ownership；
- ordered payload-before-doorbell；
- generation/seq、H2 progress/abort、canary/bounds；
- dynamic heap/layout checked arithmetic；
- vector reduce 和 FP32 accumulation/FP16 output oracle；
- BW05-A correctness matrix、device completion、raw lineage/digest。

替换：

- bid0 single RX；
- global `unique_arrived/ingress_ready`；
- hot-path `contrib_seq` 全表 readiness；
- epoch-wide reduce_done join 与 blanket input credit；
- per-result putmem+quiet；
- entry-count-only Worker lane 分配；
- group size 8 的 layout clamp；
- 将 ring payload 长期作为 zero-copy reduction backing store。

## 13. Plan 增补 phases 与 gates

原 plan 保留；在当前 PERF3-V 与继续带宽优化之间插入 FG0--FG6。
不再限制单轮推进的 phase 数量，但仍必须严格遵守依赖 gate：每个 phase 只有在
所有前置谓词通过后才能进入；任一首败必须 fail-closed，停止其依赖链，
但可继续做与该失败无依赖的诊断/工具工作。

### FG0：Host protocol/model

输出：`docs/inc/report/fg0_service_v2_protocol_gate.json`

Gate：

- writer-ownership 矩阵无混写 cacheline；
- ticket/entry ABI host/device digest；
- group 1/2/4/6/8/>8 host model；top-k 1/2/4/6/8/动态；
- group 16/32/64/128/256 的 sparse-active notification 模型，证明空 channel 不被全量
  轮询；
- random reorder/duplicate/stale/future/drop/wrap/backpressure；
- no lost/double reduce、no early credit、no reuse-before-completion；
- 2x/4x total results 下 active state high-water 不超过配置 window/pool 上限，
  `persistent_route_bytes==0`；
- checked-size/overflow/heap negative tests；
- deterministic ordinal reduce model；
- no global ingress latch/reduce_done join in `service_v2` schema。

### FG1：单 owner device

输出：`docs/inc/report/fg1_service_v2_single_owner_gate.json`

Gate：

- 多 source -> 单 owner SPSC；
- owner-homogeneous microbatch；
- MTE copy correctness 和 copy bandwidth；
- per-result early completion：故意延迟 result B，result A 仍先 reduce/TX；
- cap=4 wrap、双 ticket generation、zero/tail/top-k；
- duplicate/future/fault fail-closed；
- input credit 只在 MTE drain 后，TX completion 只在 quiet 后。

### FG2：完整 C20/C8 owner-sharded

输出：`docs/inc/report/fg2_service_v2_c20_gate.json`

Gate：

- C20/C8 数量固定且与 Worker 数无关；
- 每 result/tile exactly one owner；
- 无 shared pending counter；如使用 atomic/MPSC，只能存在于已验证的有界 active
  notification queue，不得进入 reduction state；
- no global latch、no epoch reduce_done join；
- uniform 304 results 的 owner count 15/16；
- variable cost 的 `max_owner_cost/mean <= 1.15`；
- 同 binary BW05-A 5/5、16/16 SUCCESS、VERIFY_STRICT mismatch=0。

### FG3：Window/backpressure/soak/fault

输出：`docs/inc/report/fg3_service_v2_window_gate.json`

Gate：

- W4/W8、queue saturation、cap wrap；
- group 1/2/4/6/8/>8，以及 host/model 16/32/64/128/256；
- 25 epoch ring-wrap、10k valid + fault campaign；
- credit conservation、queue high-water、无 drop/overwrite/rc124；
- 某 owner/source 拥塞时其它 owner/source 继续 progress；
- active source 数固定时，增大 total group size 不得使 owner polling/notification 成本
  按 group size 线性增长；
- 2x/4x total results 下 INC active-state high-water 增长 <= 10%（window/BDP 参数不变）；
- 1x/2x/4x workload 的稳定区 useful GB/s 下降不超过 10%。

### FG4：真实流水线

输出：`docs/inc/report/fg4_service_v2_pipeline_gate.json`

Gate：

- `first_reduce < last_upload`；
- `first_tx < last_reduce`；
- 延迟一个 result 不阻塞其它 ready result；
- trace/progress max-AIV 口径，禁止 AIV cycle 求和冒充 critical path；
- instrumentation overhead <= 5%；
- 分解 RX/MTE/vector/TX issue/TX quiet/completion/credit。

### FG5：冻结大消息 100 GB/s

输出：`docs/inc/report/fg5_service_v2_100gbps_gate.json`

Gate：

- case `mc13_g8_t512_h7168_k8_random`；
- useful_bytes=278921216；
- trace-off，warmup=5，measure>=20；
- random route/weight、FP16、same input/oracle；
- all-rank device-side completion；
- p95/p50 <= 1.15；
- useful bandwidth >= 100 GB/s；
- makespan <= 2789.21216 us；
- 同 binary correctness PASS；
- 不测、不引用 native/naive baseline；
- raw logs、CSV、content digest、exe/SO lineage 保留。

### FG6：Service lifecycle 与 framework adapter contract

输出：`docs/inc/report/fg6_service_v2_framework_gate.json`

Gate：

- 对外 API 只暴露 `query_workspace(shape, dtype, group, topk)` 和
  `combine_async(input, route_cookie, output, stream, process_group)`，不出现 `inc_rank`；
- async ticket/completion、cancel/abort、timeout、multi process group/multi model/session 隔离；
- vLLM 合同：decode/prefill、zero-token rank、dynamic batch/top-k、request cancellation、
  NPU graph 可复用地址/工作区、deadline/throughput flush；
- Megatron 合同：EP process group、forward/backward、autograd 可保存 route cookie、
  microbatch 交错、activation recompute、确定性 ordinal 归约；
- 动态 heap/capacity 与 checked workspace query，normal combine 合法输入均有路径；
- service 常驻，operation 结束/cancel/crash 后 route state 清零，generation/window 可复用；
- mock 不能只验证函数签名；必须至少运行 vLLM decode/prefill 和 Megatron
  forward/backward 的进程组行为模型，证明 Worker 无需感知 INC 拓扑。

FG5 PASS 后才回到 MC15，验证 Multi-INC Dispatch/Combine 的物理 AIV 隔离和时间并发。

## 14. 当前下一步

1. 保留 PERF3-V vector attestation 和 raw log，标记为 diagnostic/failed gate；
2. 不提高旧 BW05-A 的 30 s timeout来获得形式 PASS；
3. 冻结 `packed_epoch_v1` 作为 oracle/fallback；
4. 新增 feature-gated `service_v2` / schema4；
5. 从 FG0 开始按前置 gate 顺序推进；不限制单轮 phase 数，但不得跨过
   未 PASS 的依赖 gate；
6. FG2 同 binary BW05-A 5/5 前，`service_v2` 不得替换默认生产路径；
7. FG5 >=100 GB/s 之前不进入 Dispatch/Combine overlap；除非用户明确要求，
   不再测量或引用 native/naive combine baseline。
