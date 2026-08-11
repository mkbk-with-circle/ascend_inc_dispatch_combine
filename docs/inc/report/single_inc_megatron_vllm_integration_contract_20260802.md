# 单 INC 接入 Megatron / vLLM 的生产契约

日期：2026-08-03（最新版）

## 结论与路径不变量

现有 C ABI 已覆盖 worker-only process group、caller-owned device tensor、外部 stream、workspace query/lease、异步 request、多 inflight、取消/超时、动态 token 数、device route 描述和 dispatch/combine 并发。框架不得调用 benchmark main，文本 plan 仅用于资格验证。

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
| dispatch 输出供 grouped GEMM 使用 | **版本化 layout ABI 已有** | `inc_dc_fw_expert_layout_v1_t` 提供 permutation/inverse、logical/padded offsets、tokens-per-expert 和 2 的幂 alignment；native backend 待填充 |
| combine 精确反向映射 | **opaque route handle ABI 已有** | 只能从活跃 dispatch request 创建，绑定 generation；combine 使用期间 handle 不可释放，native backend 待消费 |
| async overlap、多 inflight | ABI 已有 | 框架 backend 接入外部 stream/event，不做 host sync |
| decode 低延迟 | 传输可运行 | 增加 persistent plan/workspace cache，避免逐 token host 分配 |
| EPLB/冗余 expert/remap | 路由可表达 | 映射和负载策略留在上层，INC 只执行已编译 plan |
| FP16/BF16 | 资格范围 | FP8/量化需增加 scale 元数据和数值 gate |

因此，传输内核与语义基础已经具备，expert alignment/permutation 与 opaque
route handle 的稳定 ABI 也已补齐；但在 device route compiler、layout/handle
的 native 消费和真实框架 backend 完成前，不应宣称
Megatron/vLLM 生产接入已经完成。

## 可移植性与失败策略

- AIV 数量运行时查询；lane/owner 容量为 128，不依赖本机 48。
- `INC_SINGLE_INC_PHY`、`INC_SINGLE_INC_WORKER_PHYS` 可提供平台 profile；缺省 profile 不适用时，从 live topology 选择同为 `HCCS_SW` 的 peers。
- 每次下发前重新核对选定的 INC→worker 关系全部同类，并等待全卡 idle；拓扑查询或 AIV capability 查询失败时拒绝运行。
- 大消息内部按容量分页，计数使用 64 位；OOM 是调用方容量问题，不改变协议语义。
- 输出顺序由 route metadata 决定，不依赖包完成顺序；workspace、route 和扩展链在 request release 前保持有效。

## 接入层下一步

1. 在 device 上把 Megatron/vLLM router 的 top-k tensor 编译成 dense/CSR plan，同时生成 semantic digest/generation。
2. 在 native backend 填充已定义的 permutation/expert-alignment layout，并消费已绑定 generation 的 opaque route handle。
3. prefill 使用流式大 workspace；decode 缓存 plan/workspace/handle，支持 zero-token rank。
4. TP×EP 只注册 EP worker group；global rank 到 worker/Phy-ID 的映射由 backend 注入，模型代码看不到 INC rank。
5. capacity/drop、EPLB 和 redundant-expert 策略由框架决定；通信层只执行并验证 plan。

相关实现：`inc_dc_framework_c_api.{h,cpp}`、`inc_dc_single_inc_stream_{main,kernel}.cpp`、`inc_dc_sv2_dyn_csr_combine_{main,kernel}.cpp`。
