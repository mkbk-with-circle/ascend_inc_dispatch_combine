# INC dispatch/combine 框架接入 API

## 结论与边界

框架接入使用
`examples/inc/dispatch_combine/inc_dc_framework_c_api.h` 中的稳定 C ABI。
它统一提供 plan、workspace query、外部 stream 异步 enqueue 和 request
生命周期，面向 PyTorch 扩展、Megatron MoE、vLLM worker/plugin，以及后续
融合算子。

该层不修改当前冻结的 dispatch/combine 内核，也不把 CLI benchmark 当成
框架 API。设备 backend 必须显式注册已晋升的 kernel lineage；backend 未
绑定并完成 NPU gate 前，不得宣称 Megatron/vLLM production ready。

## 为什么单独提供 C ABI

- 不向框架暴露 C++ STL、SHMEM rank、INC owner 或内部 queue。
- `struct_size + abi_version` 支持向后兼容扩展。
- `uint64_t stream` 直接承载调用方 `aclrtStream`，enqueue 禁止设备级同步。
- plan 保存 process group、拓扑代数和静态 route；invocation 保存动态
  token、device route、tensor、workspace 和 stream。
- backend vtable 隔离协议实现，未来融合算子可直接复用 device plan，而
  不必再走 host route generation。
- context 在创建时预分配 request/workspace lease 槽；enqueue 热路径不做
  host malloc。

## 最小调用序列

1. runtime/plugin 初始化 backend vtable，并调用
   `inc_dc_fw_context_create`。
2. 每个 EP process group/rank 创建并缓存 `inc_dc_plan_t`。
3. shape 首次出现时分别对 dispatch/combine 调用
   `inc_dc_fw_query_workspace`，框架分配对齐的 device workspace。
4. 填充 device tensor/route descriptor，传入当前 framework stream，
   调用 `inc_dc_fw_dispatch_async` 或 `inc_dc_fw_combine_async`。
5. 用 request query 或 stream/event 驱动的 backend 完成通知；只有显式
   需要 host 结果时才调用 wait。
6. request 完成/取消后 release；shape cache 淘汰时 release workspace
   token；process group 销毁时 release plan/context。

调用方必须保证 tensor、route、workspace 与 stream 的生命周期至少持续到
request completion。取消只终止该 generation，不得污染后续 invocation。

## Megatron 接入需要绑定的语义

- plan key 至少包含 model、EP process group、rank、world size、dtype、
  hidden、max tokens/top-k、topology generation。
- dispatch 输入对应 token permutation 前的 hidden states；输出 descriptor
  对应 expert-local buffer，并保留 combine 所需的 CSR/assignment identity。
- combine 必须使用同一 microbatch/generation 的 route；activation
  checkpoint/recompute 不能复用已经 stale 的 request。
- FP16/BF16、forward/backward、capacity/padding、zero-token rank 和
  auxiliary routing weight 都需要由 backend capability 与 correctness gate
  明确覆盖。
- process-group teardown、elastic/reconfiguration 必须递增 topology
  generation 并重建 plan，不能沿用旧 rank map。

## vLLM 接入需要绑定的语义

- decode 的 0/1/小 token 与 prefill 的动态 token 使用同一 API，不为固定
  batch 重新编译协议。
- scheduler request churn 通过 operation generation 隔离；超时/取消后释放
  request，但 device buffer 只能在 backend 确认终态后复用。
- workspace query 按 plan/op/shape 缓存且可回收，避免动态 shape 长跑耗尽。
- graph capture 只有 backend capability 声明
  `GRAPH_SAFE_ENQUEUE` 后才能启用；capture 内不得 route host round-trip、
  malloc、环境变量解析、busy wait 或隐式 device synchronize。

## 融合算子预留

融合的 permutation+dispatch、expert+combine 或 dispatch+GEMM 后续可直接
内嵌 `inc_dc_fw_plan_desc_t` 和 device route descriptor。稳定边界是：

- 外部 stream 和调用方 workspace；
- device pointer、shape/stride/dtype；
- operation/topology generation；
- async backend ticket。

因此融合只替换 backend enqueue，不改变上层 plan/request ABI。建议 device
route 最终采用自描述 header + CSR offsets/indices + semantic digest，并让
kernel 在提交前或首包 fail-close 校验 generation、bytes 和 bounds。

## Production 晋升门

接入层还需完成以下设备证据后才能进入默认路径：

- 把锁定 dispatch 与 dyn persistent combine 封装成同一个 backend vtable；
- 证明 enqueue 没有 `aclrtSynchronizeDevice`、进程级 barrier 或 host busy
  wait；
- W2/W4/W8、K1/2/4/6/8 和 K>W，zero/ragged/skew/extreme/fault/
  multi-epoch；
- 同 context 多 stream、多 inflight、dispatch/combine 任意交错；
- cancel/timeout/fault 后下一 generation 恢复；
- Megatron forward/backward/recompute 与 vLLM decode/prefill/churn smoke；
- 30 秒单 case、3+20 formal、100-window soak，且任何 timeout/error/
  `>192 GB/s` 均 fail-close。

在这些 gate 完成前保持 `p100_frozen` 和 `default_path_replaced=false`。
