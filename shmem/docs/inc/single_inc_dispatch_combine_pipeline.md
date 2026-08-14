# 单 INC Dispatch / Combine 当前实现与流水线

本文说明当前仓库中单 INC `dispatch` 和单 INC `combine` 的生产实现，重点回答三件事：数据如何经过 worker、INC 和目标 worker；相邻数据如何形成流水；两个算子如何在任意时序下并发。

核心文件：

- Dispatch Host：`examples/inc/dispatch_combine/inc_dc_single_inc_stream_main.cpp`
- Dispatch Kernel：`examples/inc/dispatch_combine/inc_dc_single_inc_stream_kernel.cpp`
- Combine Host：`examples/inc/dispatch_combine/single_inc/combine/inc_dc_combine_launcher.cpp`
- Combine Kernel：`examples/inc/dispatch_combine/single_inc/combine/inc_dc_combine_kernel.cpp`
- 正式 sweep：`examples/inc/dispatch_combine/scripts/single_inc/run_single_inc_operator_sweep.sh`
- 随机 token-plan sweep：`examples/inc/dispatch_combine/scripts/random_token_plan/run_single_inc_os_random_campaign.py`

本文描述的是完整算子的生产路径。`inc_dc_single_inc_roofline_*`、ping-pong 和纯打流程序只测 push 传输上限，不参与生产算子的路由、归约和正确性协议。

## 1. 共同设计原则

两条算子都由 worker 主动向 INC `push` 数据，INC 不反向 pull。数据可见性遵循：

```text
写 payload -> remote completion / quiet -> 发布 generation -> 消费方检查 generation
```

generation 因而既是通知，也是 payload 已可安全读取的提交记录；循环复用缓冲区时也不会把上一轮的 flag 误认为本轮完成。

worker 数、AIV 数、`topk`、行数、hidden 大小、tile 大小、路由、owner 和地址布局均由运行时描述符给出。W2/W4/W8 是验收拓扑，不是 Kernel 上限。Host 使用 checked 64-bit 算术生成对称内存布局，避免用固定规模或 32 位 offset 限制大消息。

## 2. 单 INC Dispatch：U/G/T 三级流水

### 2.1 数据流

每个源 worker 持有输入 token 行及其 `topk` 路由。Dispatch 先把一个源 tile 只上传一次，再由 INC 按路由扩展并发往各目标 worker：

```text
源 worker
  U: 上传 source tile 一次
          |
          v tile_ready
单 INC
  G: direct 引用，或 gather/pack
          |
          v packet_ready
  T: push packet 到各目标 worker
          |
          v completion generation
目标 worker
  在确定的 output offset 得到结果
```

因此 worker 到 INC 的上行只传原始 source tile；`topk` 扩展发生在 INC，扩展后的实际总字节由 INC 下行发送。

### 2.2 Host 路由规划

Host 把路由编译为 `StreamDispatchTask` 和 `StreamRouteEntry`：

- task 表示一个 source tile 到一个目标 worker 的输出 packet；
- route 表示 packet 中某段来自 source tile 的哪些行；
- 连续且无需重排的 task 标记为 direct；
- 任意、不连续路由使用 task 私有的紧凑 staging packet；
- packet 的源、目标、字节范围和 output offset 均在下发前完成边界检查。

启动采用设备侧 gate：worker 报到，INC arm，worker ack，INC 再并行释放 worker。正式 device interval 位于 gate 之后，避免 Host 进程启动先后污染算子耗时。

### 2.3 U：worker 上传

每个 worker 的 `upload_lane_count` 个 AIV 协作处理同一 tile，按行区间分片：

1. lane 把自己的连续行片段 push 到 INC ingress；
2. lane 执行远端完成；
3. lane 发布本地 `upload_chunk_done`；
4. lane 0 收齐全部 lane 后，向 INC 发布该 tile 的 `tile_ready` generation。

ready 发布后，worker 可立即上传下一 tile，不必等待当前 tile 完成下行。因此 `U(n+1)` 能与 INC 上的 `G(n)`、`T(n)` 重叠。

### 2.4 G：INC direct 或 gather

INC 根据路由形态选择两条热路径：

- **Direct zero-copy**：packet 对应 source tile 的连续区域时，TX AIV 直接读取 ingress，不产生 staging 拷贝。全 direct 负载不建立 gather cohort，把更多 AIV 留给发送。
- **Generic gather**：gather AIV 将 route 分片，用 GM2GM MTE 把离散输入行整理到 task 私有的连续 staging packet。各 slice 完成 cache 可见性处理后发布 generation。

存在 generic task 时，运行时根据可用 AIV 和任务量划分 gather/TX cohort；当前策略通常约三分之一用于 gather，其余用于 TX，但不绑定固定物理核编号。

### 2.5 T：INC 下行

TX cohort 按目标 worker 或 task 分片。每个 TX lane 等待 direct tile ready，或等待 generic packet 的 slice ready，然后把连续 packet push 到目标 output offset，并发布 lane 完成 generation。协调 AIV 收齐相关 TX lane 后再向 worker 发布算子完成。

普通 `putmem` 路径仍将单 AIV 的 outstanding window 固定为 1。全 direct 生产路径则使用两个私有 16 KiB UB 和两个独立 sync id 做安全 ping-pong；复用某个 slot 前显式等待该 slot 排空。这既隐藏 MTE 延迟，也避免共享 slot 在散列源/目的地址下静默损坏。

Host 进一步按 TX lane 把 task 物理重排为连续区间。每个 AIV 只顺序读取自己的
task，不再由所有 lane 重复扫描整张任务表，也没有间接 worklist 的随机读取。
当前 48-AIV 平台由统一硬件 policy 得到 16 个 INC TX AIV；worker W2/W4/W8
分别固定使用 8/4/2 个上传 AIV。tiny call 中无工作行的 lane 直接空闲，资源图不变。

### 2.6 稳态时间线

```text
时间 --->

Worker U : [upload tile 0][upload tile 1][upload tile 2]...
INC G    :                [gather 0]    [gather 1]...
INC T    :                     [send 0]      [send 1]...
Target   :                           [use 0]       [use 1]...
```

Direct task 跳过 G 阶段。Generic task 则用 INC 显存换协议简单度：每个 task 有明确的连续 staging 和 generation，发送阶段无需再次解释离散路由。

## 3. 单 INC Combine：P/R/E 波前流水

### 3.1 数据流

worker 持有多个 contribution。INC 按逻辑结果行收齐 contribution，以 FP32 累加，再把最终结果发给该结果指定的 `dst rank / dst row`：

```text
源 workers
  P: push contribution 到 INC ingress
          |
          v ready generation
单 INC owner AIV
  R: 按 CSR 等待、校验、FP32 reduce
          |
          v result ready
  E: push 最终结果到指定 dst
          |
          v completion
目标 worker
```

上行传输 `topk` 展开后的全部 contribution；下行只返回 reduce 后的结果行，从而消除逐 contribution 的重复下行。

### 3.2 Host 的 CSR 与 slot

Host 构造 result-major CSR：

- `result_offsets` 给出每个结果的 contribution 区间；
- contribution 记录 source rank、ordinal、weight、ingress slot、home owner 和 result id；
- result 记录准确的目标 rank 和目标 row；
- ingress slot 按 `(owner, source)` 分组，并生成 source/owner worklist。

传输 slot 只决定数据放在哪里，不代替数学身份。Kernel 另外校验 result id、ordinal、generation 和重复项，防止“地址合法但归约到错误结果”的静默错误。

### 3.3 P：worker producer

当前合格配置启用 device producer、batched ready 和 coalesced group put。Host
使用统一 source/INC-major generation 协议和硬件固定 producer cohort；当前
48-AIV 环境 W2/W4/W8 分别固定使用 24/16/12 个 worker producer lane。K、
message bytes、contribution 数和 route 只改变 worklist 内容，不改变 lane 数。

Producer lane 按 owner/source stream 以 breadth-first wave 调度：先 push 连续 contribution payload，确认远端完成后再发布对应 generation，然后继续下一 stream/wave，不等待前一结果完成 reduce。长包的公共共享 slot 不允许多 outstanding；符合条件的稀疏 K2 train 使用两个私有 UB/event 同时下发，两个 payload 都 remote-complete 并 quiet 后才发布 ready。rank 预归约路径自动退回严格单包完成，避免静默覆盖。

### 3.4 R：INC owner 归约

每个逻辑结果在运行时永久分配给一个 owner AIV。owner 数由共享硬件 policy
确定，不受实际结果数或 workload 约束；当前 48-AIV 环境为 Combine 32，并为
并发 Dispatch 保留不相交的 16。没有结果可处理的 owner 空闲。一个 owner 独占
一个结果，避免跨 AIV 浮点原子累加带来的顺序、精度和性能问题。

Host 会验证单 INC 的 cyclic-owner 映射；验证通过后 owner 直接以 `owner_count`
为步长遍历自己的连续逻辑行，不再由所有 owner 各自扫描完整结果表。任意非
cyclic 映射仍回退到通用过滤路径。

Owner 对自己的 CSR 行执行：

1. 把 contribution 映射到 source/slot 的 ready generation；
2. 等待并检查 generation；
3. 校验 ordinal、result id、slot 和重复 contribution；
4. 收齐该结果后，在 UB 中将 FP16 输入与 FP32 weight 以 FP32 累加；
5. 转回 FP16 结果。

合格路径使用 1536-element wide vector tile，容量条件不满足则回退到 512；`K=1 && weight=1` 可走 identity copy。`topk <= 64` 使用 owner-local 位图，`topk > 64` 自动使用逐 contribution generation marker，因此不存在 64 位位图导致的协议上限。

### 3.5 E：结果返回

当前合格配置启用 `REMOTE_TX=1`：owner 完成结果后，直接把它 push 到 CSR 指定的 `dst rank / dst row`。实测 TX window=4 的最慢-rank 尾部最稳；更深窗口虽有更高峰值，却增加跨运行抖动。

代码还保留可选的独立 TX cohort：owner 发布 `result_tx_ready`，TX lanes 分片发送。该模式适合结果返回成为主要瓶颈的平台，但不是当前默认合格路径。全部 owner 和可选 TX lane 完成后，由 owner 0 汇合统计并发布全局完成。

### 3.6 稳态时间线

```text
时间 --->

Worker P : [push wave 0][push wave 1][push wave 2]...
INC R    :              [reduce row 0][reduce row 1]...
INC E    :                    [return 0]    [return 1]...
Dst      :                           [use 0]       [use 1]...
```

单个结果必须等待它所需的 contribution 全部 ready，这是数学依赖；不同结果彼此独立，所以 producer、多个 owner 的 reduce 和已完成结果的返回可同时推进。

## 4. Dispatch 与 Combine 如何并发

两条算子使用独立的 SHMEM session、descriptor、staging/ingress、generation 和 completion，不读取对方控制变量，也不以对方完成作为推进条件，因此允许任意下发顺序和启动延迟。

当前 48-AIV INC 的不相交配额为 Dispatch 16、Combine 32；W8 overlap 的
worker 侧分别使用 2 个 upload AIV 和 12 个 producer AIV。数量只由 live AIV
和 worker 拓扑规模 W 派生。环境变量只允许在资格化范围内显式减配；越界或
格式非法均 fail closed，不能静默改变生产资源图。

```text
Dispatch : worker upload -> INC gather/direct -> INC downlink
Combine  : worker upload -> INC reduce        -> INC result return
                         同一时间轴并行推进
```

协议上不设置跨算子全局 barrier。真正需要控制的是共享 AIV、MTE 和 HCCS 带宽竞争；独立 cohort 和细粒度 generation 保证一条路径等待某个 tile/source 时，另一条仍能前进。

## 5. 正确性、鲁棒性和可移植性

- Dispatch 的 route 落到预先检查的目标 output offset，测试对完整输出逐字节校验。
- Combine 校验 contribution 的身份、代次和重复项，并按 result 元数据返回正确 dst。
- 地址和字节数使用 checked 64-bit layout；大数据量主要受实际显存/OOM 限制，而非固定 chunk 数或 32 位 offset。
- worker、`topk`、AIV、owner 和 lane 数均为运行时参数。
- 超时、非法元数据或 generation 不一致均 fail closed，不把错误数据当成成功结果。

## 6. 完整算子的测量口径

用设备侧 gate 对齐正式协议区间。令每个 rank 的协议耗时为 `T_rank`：

```text
T_makespan = max(T_rank)

Dispatch BW = rank 去重后的物理下行字节 / T_makespan
Combine  BW = 全部 expert-instance 的物理上行字节 / T_makespan
```

这里使用最慢 rank 的完成时间，但排除 Host 启动先后造成的延长；不能把 rank
带宽相加，也不能使用最快 rank。预热轮不计入统计。Dispatch 的 expert 逻辑
assignment 字节与 Combine 可选 rank-local pre-reduce 的逻辑字节只用于报告，
绝不替代当前默认物理数据面的带宽分子。

性能 gate 的 baseline 来自同拓扑、同方向、push 语义、足够大数据量和充分 AIV 并行的多打一/一打多 roofline。roofline 衡量传输后端上限，完整算子再按相同方向和字节口径计算协议效率。

## 7. 总结

当前 Dispatch 是“source tile 上传一次，INC direct/gather 后并行下发”的 U/G/T 流水；Combine 是“contribution 波前上传，owner 独占结果归约，完成即返回正确 dst”的 P/R/E 流水。两者依靠独立 session、运行时 AIV cohort 和 generation 可见性协议实现任意时序并发，而不是用全局 barrier 串行执行。
