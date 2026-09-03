# 单 INC Pull-Dispatch V2

## 冻结语义

- 每个 worker、每个 token wave 只发布一个 `Ready`。
- `Ready` 只携带 epoch 与注册区域标识；不携带 token-plan、地址或 counts。
- INC GET 固定 header 后，按 tile GET metadata/hidden、在线解析并 fan-out。
- 每个 token 从源 worker 到 INC 只传一份 hidden；对每个唯一目标 GPU 只 PUT
  一份 hidden，同 GPU 多 expert 仅增加 assignment。
- 目标端采用动态 expert-major 布局；兼容重整按 tile 与网络接收交叠。
- INC 保存 generation-scoped journal，Combine 完成后回收。

## 注册内存

```text
Worker source region
└── ring slot
    ├── SlotHeader                 128 B
    ├── TokenRecord[token_count]   32 B/row
    ├── AssignmentRecord[count]    16 B/assignment
    └── Hidden[token_count][H]
```

wire 中只出现 `(session_id, placement_epoch, source_rank, region_id,
ring_slot)`。SHMEM 对称地址、RDMA MR 或真实 INC 物理窗口由 transport registration
解析，不进入协议。

## 生命周期

```text
source slot:
FREE -> FILLING -> READY -> GET_ACTIVE -> SOURCE_CONSUMED -> FREE

journal slot:
FREE -> DISPATCH_OPEN -> DISPATCH_SEALED -> COMBINE_ACTIVE
     -> COMPLETE/ABORTED -> FREE
```

`SOURCE_CONSUMED` 只表示源 slot 已完成全部 GET；destination completion 表示目标
数据和必要重整可消费；journal 必须保留到 Combine completion。三种事件不能合并。

## 数据流水

```text
INC Dispatch AIV 0..23

GET meta 0 -> parse/reserve 0 -> GET hidden 0 -> PUT destinations 0
               GET meta 1 -> parse/reserve 1 -> GET hidden 1 -> PUT destinations 1
                              ...
```

最终实现采用 per-peer channel 与 2–3 个有界 tile credit；禁止跨 AIV 轮询共享
cache line 传递大数据所有权。控制 publication 独占 64 B cache line。未完整校验前
允许预取或写入未提交的目标 slot，但只有成功 completion 才允许 worker 消费。

## 动态布局与重整

INC 为每个 `(destination, expert)` 动态分配 row。一个 `DestinationRow` 对应
`(token, unique destination)`，多个 `ExpertAssignment` 可引用同一行。兼容路径在
目标 worker 本地展开到 expert-major；融合路径让 grouped GEMM 直接消费 compact
hidden 与 gather index。

## Combine 协同

Dispatch 在线生成：

```text
JournalTokenEntry(owner rank/row, contributor range)
JournalContributor(B rank, destination row)
```

Combine 默认 canonical partial 布局：B 把同 GPU 多 expert 输出按 weight 本地归并回
`partial[destination_row]`，发布一个 READY；INC 直接 GET、tile reduce、PUT owner。
现有 1536-element FP32 tile、MTE2 ping/pong、UB Add、直接 PUT 和24 AIV reducer保留。

## AIV 与并发

```text
INC 48 AIV:
  0..23   Pull-Dispatch
  24..47  Pull-Combine

worker:
  READY/completion 常驻控制最多 1–2 AIV
  重整/local-reduce 按需或与计算融合
  其余留给 grouped GEMM/SwiGLU
```

必须支持 `Combine(wave N, slot s)` 与 `Dispatch(wave N+1, slot 1-s)` 任意错峰并发，
standalone benchmark 不得借用另一半 INC AIV。

## Gate

完整 Dispatch 从 READY publication 计到 destination 重整完成、destination completion
和 source ACK 均发布：

```text
logical bytes = hidden GET once + hidden PUT once per unique destination
W2 min >= 50.4 GB/s
W4 min >= 100.8 GB/s
10 measures 全正确，CV <= 5%
```

另外记录 GET-only、PUT-only、匹配 fan-out 比例的 GET+PUT mixed roofline，以及
重整带宽；旧 Push-Dispatch 数据不覆盖，只作为 legacy 对照。
