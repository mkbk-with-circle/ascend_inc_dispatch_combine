# 单 INC 接入 Megatron / vLLM 的生产契约

原始资格化日期：2026-08-03；公开接口收口：2026-08-14

## 结论与路径不变量

当前业务侧只公开
`examples/inc/dispatch_combine/common/api/inc_dc_single_inc.hpp`。它覆盖
worker process group、caller-owned device tensor、外部 stream、device route 和
Dispatch→expert→Combine 生命周期；调用序列固定为
`create -> dispatch -> compute -> combine -> destroy`。框架不得调用 benchmark
main，也不得逐层调用内部 Framework/Easy/Inference API。

2026-08-03 验证过的 workspace lease、异步 request、多 inflight、取消/超时和
stats 能力仍存在于内部 runtime/tests，但不再作为当前原型的公开契约。需要真实
Megatron/vLLM 集成证明确有必要后，再从最短 API 上增量设计，不能把旧分层入口
重新暴露给业务代码。

生产通信路径只有一种：`worker push -> single INC -> worker`。禁止 pull、worker
direct/bypass 和按 shape 切换替代路径。INC 不预知 K、token plan 或负载分布；
INC 与 worker 的 AIV cohort 只由运行时硬件能力和拓扑规模 W 决定，本机 48 AIV
固定分成 dispatch 16、combine 32，二者互不相交；worker W2/W4/W8 的 D/C 为
`8/24`、`4/16`、`2/12`，均不超过总 AIV 的一半，且不随 workload 改变。

## 任意 token plan 与方案 B

`inc_dc_fw_route_metadata_v1_t` 表达 local expert offsets、source token、expert id、原始 top-k ordinal、assignment weight、capacity/drop mask。允许任意合法目的 rank、K>W、ragged route、零 token rank，以及同一 token 的多个 expert 位于同一 rank。

方案 B 的准确语义是：dispatch 对同一 `(token, destination rank)` 只发送一个 hidden row，同时保留所有 expert/ordinal/weight 元数据供目的 rank展开；combine 默认把各 expert-instance 输出全部 push 到 INC，由 INC 按反向 route 加权归约并返回正确 source token。不能在 worker 默认按 rank 预归约，否则会把 grouped-GEMM 后不同 expert 的独立结果混同。

任意 plan 真机资格样例包含 K>W、同 token 同 rank 多 expert、重复 expert id、负/非单位权重和无规律目的 rank；2026-08-03 在 W2/T2/K4 上 3/3 rank 逐字节通过。日志：`/tmp/single-inc-arbitrary-plan-20260803`。

## 与成熟框架接口的对齐情况

| 能力 | 当前状态 | 接入动作 |
|---|---|---|
| 动态 top-k/CSR、weight、drop mask | ABI 已有 | 把 router device tensor 编译成 route descriptor |
| dispatch 输出供 grouped GEMM 使用 | native layout adapter 已有 | 适配器把 Dispatch 输出映射到 expert-major/padded buffer；业务侧不直接操作 layout extension |
| combine 精确反向映射 | 最短 API 已封装 | `SingleIncBatch` 绑定 generation 和精确 route，Combine 成功或异常退出时自动释放 |
| async overlap、多 inflight | 设备/runtime 有历史证据，当前不公开 | 单 INC 普通调用先走最短同步 API；Fusion overlap 走独立 `inc_fusion_api.h` |
| decode 低延迟 | 传输可运行 | 增加 persistent plan/workspace cache，避免逐 token host 分配 |
| EPLB/冗余 expert/remap | 路由可表达 | 映射和负载策略留在上层，INC 只执行已编译 plan |
| FP16/BF16 | 资格范围 | FP8/量化需增加 scale 元数据和数值 gate |

因此，传输内核与语义基础已经具备，公开调用面也已经收成单一短路径；但真实
router adapter、grouped GEMM、动态 shape 和端到端框架门禁完成前，不应宣称
Megatron/vLLM 生产接入已经完成。

## 可移植性与失败策略

- AIV 数量运行时查询；lane/owner 容量为 128，不依赖本机 48。
- `INC_SINGLE_INC_PHY`、`INC_SINGLE_INC_WORKER_PHYS` 可提供平台 profile；缺省 profile 不适用时，从 live topology 选择同为 `HCCS_SW` 的 peers。
- 每次下发前重新核对选定的 INC→worker 关系全部同类，并等待全卡 idle；拓扑查询或 AIV capability 查询失败时拒绝运行。
- 大消息内部按容量分页，计数使用 64 位；OOM 是调用方容量问题，不改变协议语义。
- 输出顺序由 route metadata 决定，不依赖包完成顺序；route 生命周期由
  `SingleIncBatch` 持有到 Combine 完成。

## 接入层下一步

1. 把 Megatron/vLLM router 的 top-k tensor 编译成 `SingleIncRoute`，同时生成 semantic digest/generation。
2. 用 native layout adapter 将 Dispatch 输出直接交给 grouped GEMM；同一
   `SingleIncBatch` 随后传给 Combine，不让框架接触内部 route handle。
3. prefill 复用长生命周期 `SingleInc` 和大 workspace；decode 支持 zero-token
   rank，并在有测量证据后再增加必要的缓存或异步接口。
4. TP×EP 只注册 EP worker group；global rank 到 worker/Phy-ID 的映射由 backend 注入，模型代码看不到 INC rank。
5. capacity/drop、EPLB 和 redundant-expert 策略由框架决定；通信层只执行并验证 plan。

公开入口：`examples/inc/dispatch_combine/common/api/inc_dc_single_inc.hpp`。
设备实现：`examples/inc/dispatch_combine/single_inc/{dispatch,combine,planning,runtime}/`。
完整示例：`examples/inc/dispatch_combine/common/examples/single_inc_api/`。
