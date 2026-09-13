# vLLM-Ascend 接入与基线契约

> 当前 nb-borrow ABI 13 的可执行环境、构建命令、发布候选跑法与结果索引见
> [`RUNBOOK_NB_VLLM.md`](RUNBOOK_NB_VLLM.md)。本文后续保留了开发过程中的设计契约；
> 其中 ABI v3/v6/v9 的段落属于演进记录；当前发布路径以 ABI 13 和
> `fusion_kernel_release_20260810` 报告为准。

## 已确认的接入点

目标版本是当前容器中的 vLLM 0.19.1 / vLLM-Ascend 0.19.1。正式接入应新增
`MoECommMethod`，在它的 `fused_experts(MoEFusedExpertsInput)` 中消费 vLLM 已经生成的：

- `hidden_states`
- `topk_ids` / `topk_weights`
- 本 rank 的 `w1` / `w2`
- `expert_map` / `log2phy`

不要替换整个 `FusedMoE.forward`，也不要沿用 `example/custom_moe.py` 的 host
`all_gather + .cpu() + argsort` 路由。后者只保留为旧功能原型。

`single_inc_comm_method.py` 已按这个接口提供适配层，并且有意在 Torch op 未注册、
非 BF16、量化权重或非 SwiGLU 时显式拒绝执行。它当前不自动 monkey-patch vLLM；待
device op 完成后应在 vLLM-Ascend 的 `setup_moe_comm_method` 中正式注册新枚举，而不是
借用/覆盖 `FUSED_MC2`。

`torch_registration.py` 定义图可见的 `inc_fusion::moe` 与 fake 实现，但仅当已加载的
C++ bridge 提供 `inc_fusion_native::worker_moe_out` 时才允许注册。output 是 setup 时
预分配的 mutable alias，forward 不调用 `empty_like`。

`native/` 已提供并真机验证 `inc_fusion_native::route_count_out`、标量
`route_pack_out` 与确定性多 AIV `route_pack_parallel_out`。`torch_route_runtime.py` 为每个 capacity bucket 预分配协议 buffer，
热路径只执行 count、一次 EP metadata all-gather 和 pack；all-gather 的最后一列同时
携带每个 rank 的 active-token 数，并在设备上整理成 ABI v3 引入、ABI v6 继续兼容的
连续 source-length 向量。
prepared worker executor 已完成并通过 W2/W4 E2E：其参数环不会在热路径分配或全局同步。
跨进程 INC command ring 已闭环：单个连续 wire record + ready-last 提交、
全 worker readiness、精确 packet credit 和无清零 ring 回卷均已在 nb 的 W2/W4 ring=2
各 100 次通过。worker 侧 Torch 生命周期提供显式 `worker_prepare`、allocation-free
`worker_moe_out` 和有序 `worker_destroy`；ProcessGroup 外 INC sidecar/bootstrap 已接入
独立 runner，并在 vLLM-Ascend 0.19.1 容器完成 W2/W4、16–1024 token 真机运行。

单 INC 进程不加入 vLLM 的 TP/DP/EP process group，也不加载权重/KV cache；它只加入
Fusion SHMEM world。worker group 是 W，Fusion SHMEM world 是 W+1。

## Engine 生命周期顺序

launcher 生成一次 ACLSHMEM UID，并通过独立 rendezvous 分发给 W 个 model worker 与一个
INC sidecar。每个进程先选定自己的 NPU，再创建 `FusionShmemSession` 和同尺寸 symmetric
tensor。worker 创建 `PreparedTorchRoute + PreparedTorchExecutor`；sidecar 不加载模型，
只创建 `PreparedIncService`。随后所有 W+1 PE 调用一次 `session.barrier()`，INC 返回后调用
`service.start()`，worker 即可进入 forward。启动时序为：

在 SHMEM init 之前，launcher 对每个 worker rank 调用 CPU `query_plan_info()`，校验
`symmetric_bytes/wave_count/ABI/remote_service_bytes` 一致，再通过
`SymmetricHeapPolicy` 算出所有 PE 相同的 heap 大小。rank-local expert 数和 worker
workspace 可以不同。`FusionPeMapping` 显式保存 PE→NPU 映射，不把 nb 的 0/1/2/3/4
编号写入 kernel 或 framework adapter。

```text
launcher: UID ────────────────┬──────── W workers
                              └──────── INC sidecar
all PE:    SHMEM init → symmetric alloc/zero → prepare mirror → barrier
INC:                                                        └→ service.start
worker:                                                      └→ route → moe_out
```

teardown 先停止接收新 batch，等待 engine 在途 forward 收齐；workers 关闭 executor，INC
关闭 service（有序 stop），全部 PE 再做一次 SHMEM barrier，最后释放 symmetric tensor 并
finalize。异常中途不能把 barrier 当成 cancel；push-only 请求一旦发布就必须收齐。

`inc_sidecar.py` 提供不依赖 vLLM ProcessGroup 的 spawn 入口和 parent controller。控制协议
固定为 `PREPARED → READY → STOPPING → STOPPED`：parent 收到 PREPARED 后让所有 worker
完成自己的 prepare/setup barrier，收到 READY 后才放行推理；停止时必须先停止 scheduler
接收新 batch、排空在途 forward，再发送 STOP 并让 workers 进入 teardown barrier。
controller 超时只报错，不会擅自 terminate 常驻 kernel 进程。

当前明确不支持 NPU graph capture/replay。普通 `torch.compile` opaque custom-op 每轮仍会进入
host `worker_moe_out`，可以生成新 ticket；但 capture/replay 会重放旧 generation/ticket，
native bridge 因此设置了硬拒绝。首轮 vLLM 资格化必须关闭 NPU graph，待 ticket 改为
graph-replay-aware device counter 后再单独开放。

`torch_weight_runtime.py` 提供 setup-only `PreparedWeightCache`。它显式区分 kernel ND
`w13[E,2I,H]/w2[E,H,I]` 与逻辑转置布局，只接受 storage format ND(2) 或
FRACTAL_NZ(29)；NZ 在模型加载期一次性转 ND，转置也只发生一次。forward resolver 校验
原 vLLM weight 的 device/data pointer 没有被替换后直接返回缓存，不做 format-cast、
transpose 或 allocation。其他 storage format 一律拒绝。

setup 端的核心调用关系如下（UID 分发与进程创建仍由实际 launcher 负责）：

```python
infos = query_all_worker_plan_info(key, config, owner, local)
heap_bytes = SymmetricHeapPolicy().heap_bytes(infos[0])
session = FusionShmemSession(
    uid, pe, W + 1, heap_bytes, infos[0].symmetric_bytes, device)

# worker: route = PreparedTorchRoute(...)
# worker: executor = PreparedTorchExecutor.from_session(
#                       route, config, worker_pes_tensor, session)
# INC:    service = PreparedIncService(key, config, owner, local,
#                                     worker_pes, session)
session.barrier()
# 仅 INC 调用 service.start()；worker 随后可进入 forward。
```

## 五种模式

| mode | 通信 | 调度 | 用途 |
|---|---|---|---|
| `native_vllm` | vLLM-Ascend 原生 | 原生 | 端到端外部基线 |
| `serial_shmem` | worker-direct SHMEM | 严格 D→FFN→C | 2×2 单元格 |
| `serial_inc` | worker→单 INC→worker | 严格 D→FFN→C | 2×2 单元格 |
| `fused_shmem` | worker-direct SHMEM | token-wave | 2×2 单元格 |
| `fused_inc` | worker→单 INC→worker | token-wave | 交付实现 |

前四种必须通过 `inc_fusion_benchmark_validate_factorial_pair`。原生 vLLM 可以报告端到端
加速，但不得混入 INC/融合收益的因子归因。

资源口径需要同时报两列：主表固定 W 个计算 worker（INC 模式额外使用 1 张不加载模型
的卡），用于隔离通信机制；系统表按总 NPU 数归一 throughput，明确比较 W 与 W+1 的
硬件成本。不能只报前者后宣称“同成本”推理加速，也不能让原生 vLLM 偷换成不同 W 后
仍声称逐算子 shape 等价。

## 正式跑数前的硬 gate

1. 四种模式共享同一份 `topk_ids/topk_weights`、expert placement 和路由 digest。
2. 四种模式共享同一个 GMM1/SwiGLU/GMM2 实现与权重布局 digest。不能用 vLLM
   grouped-GMM 对比另一行的 Catlass kernel，并把差值归因给通信。
3. 路由打包在设备上完成并计入每轮耗时；setup/JIT/权重重排/workspace 分配不计入。
4. BF16 ND 权重由 kernel 原生消费；W8A8/NZ 量化语义尚未实现，必须 fail-closed，不能
   每轮 transpose/format-cast 或把量化 baseline 混入 BF16 资格化。
5. engine setup、独立 INC sidecar、常驻 server 和有序 teardown 已闭环；异常路径仍不得
   把 barrier 当 cancel，失败后需要清理本次 control 目录并确认 NPU context 已释放。
6. 计时样本必须取所有 worker 的 makespan；INC 模式另报 INC service window。

算子 microbenchmark 固定并重放同一份 router 输出；真实 vLLM 运行则保留 router，但固定
模型、prompt、seed、调度策略、EP placement、KV cache 和采样参数。Prefill 报 input
tokens/s 与 TTFT，decode 报 output tokens/s 与 TPOT，并分别给 P50/P95/P99。真实模型的
hidden size 由 checkpoint 固定；跨 hidden 的扫描只属于合成算子测试或不同模型配置。

## 动态 token 数

`inc_moe_runtime.py` 的一般 prepared API可使用 capacity bucket；但一个跨进程长驻 INC
service 必须使用 engine 最大 token 数对应的单一固定容量，任意 0..capacity token 数通过
active-token 向量复用，不再要求整除 microbatch。只有显式 HBM budget 可以拒绝请求。`prepare()`
只在 warmup/setup 调用，计时 forward 只允许 `lookup()`，从而不会把首次分配/JIT 混入
某个 backend。

实际 token 数小于容量时，尾部 wave 由设备 pack 成零 token 的合法描述符。不同 worker
的 token 数也可以不同：同步 wave 取全组最大 active-token 数，而 INC ABI v6 使用每个
source 的真实长度清零、校验、归约和回传，不要求为短 rank 分配伪 output padding。

路由的并行 lane 数在 prepared 阶段从设备 `vector_core_num` 获取。forward 只用整数协议
工作量选择标量/并行路径：小输入与 top-k=1 回退标量，足够大的 regroup 工作进入
analyze/prefix/emit；不维护模型 shape 调参表。nb 的 48 AIV 回归中，被策略选中的 20-case
标准矩阵没有性能倒退，W4/T8192/K8 相对标量 route 从 10.19ms 降至 3.75ms。

## 发布候选状态

- 已完成：device route-pack、Torch NPU `out` 桥接、prepared worker executor、跨进程
  persistent INC、runner sidecar、setup/out/teardown 和 W2/W4 真机 sweep。
- 已冻结：ABI 13 BF16 `fused_inc` 是当前自定义最优路径；性能配置不再运行时遍历。
- 未完成：与 native 相同 prompt 的数值等价、W8A8、replay-safe native graph、当前
  direct-SHMEM vLLM lifecycle。细节和数据见
  [`fusion_kernel_release_20260810`](../../../../../docs/inc/report/nb-borrow/fusion_kernel_release_20260810/README.md)。

---

## English

# vLLM-Ascend integration and baseline contract

> For the executable nb-borrow ABI 13 environment, build commands, release-candidate
> runs, and result index, see [`RUNBOOK_NB_VLLM.md`](RUNBOOK_NB_VLLM.md).
> The rest of this file keeps design contracts from development. ABI v3/v6/v9
> paragraphs are evolution notes; the current release path is ABI 13 and the
> `fusion_kernel_release_20260810` report.

## Confirmed integration point

The target is vLLM 0.19.1 / vLLM-Ascend 0.19.1 in the current container. Formal
integration should add a `MoECommMethod` and consume, in
`fused_experts(MoEFusedExpertsInput)`, what vLLM already produced:

- `hidden_states`
- `topk_ids` / `topk_weights`
- this rank's `w1` / `w2`
- `expert_map` / `log2phy`

Do not replace all of `FusedMoE.forward`, and do not keep using
`example/custom_moe.py`'s host `all_gather + .cpu() + argsort` route. That path
stays only as an old functional prototype.

`single_inc_comm_method.py` already adapts this interface and intentionally
fails closed when the Torch op is unregistered, dtype is not BF16, weights are
quantized, or the activation is not SwiGLU. It does not auto monkey-patch vLLM;
after the device op lands, register a new enum in vLLM-Ascend
`setup_moe_comm_method` instead of borrowing/overriding `FUSED_MC2`.

`torch_registration.py` defines graph-visible `inc_fusion::moe` and a fake impl,
but only registers when the loaded C++ bridge provides
`inc_fusion_native::worker_moe_out`. Output is a mutable alias allocated at
setup; forward does not call `empty_like`.

`native/` provides and device-validates `inc_fusion_native::route_count_out`,
scalar `route_pack_out`, and deterministic multi-AIV `route_pack_parallel_out`.
`torch_route_runtime.py` preallocates protocol buffers per capacity bucket; the
hot path only runs count, one EP metadata all-gather, and pack. The last
all-gather column also carries each rank's active-token count and is packed on
device into the contiguous source-length vector introduced in ABI v3 and still
compatible in ABI v6. The prepared worker executor is done and passed W2/W4
E2E: its argument ring allocates nothing and does no global sync on the hot
path. The cross-process INC command ring is closed: one contiguous wire record
+ ready-last submit, all-worker readiness, exact packet credits, and wrap
without zeroing all passed 100 times each on nb W2/W4 ring=2. Worker-side Torch
lifetime exposes explicit `worker_prepare`, allocation-free `worker_moe_out`,
and ordered `worker_destroy`. The ProcessGroup-external INC sidecar/bootstrap
is wired through a standalone runner and ran on device in the vLLM-Ascend
0.19.1 container for W2/W4 and 16–1024 tokens.

A single-INC process does not join vLLM's TP/DP/EP process group and does not
load weights/KV cache; it only joins the Fusion SHMEM world. The worker group
is W; the Fusion SHMEM world is W+1.

## Engine lifetime order

The launcher mints one ACLSHMEM UID and distributes it over an independent
rendezvous to W model workers plus one INC sidecar. Each process first selects
its NPU, then creates a `FusionShmemSession` and same-size symmetric tensors.
Workers create `PreparedTorchRoute + PreparedTorchExecutor`; the sidecar loads
no model and only creates `PreparedIncService`. Then all W+1 PEs call
`session.barrier()` once; after INC returns it calls `service.start()`, and
workers may enter forward. Before SHMEM init, the launcher calls CPU
`query_plan_info()` for each worker rank, checks
`symmetric_bytes/wave_count/ABI/remote_service_bytes` match, then
`SymmetricHeapPolicy` computes one heap size for every PE. Rank-local expert
count and worker workspace may differ. `FusionPeMapping` stores PE→NPU
explicitly and does not bake nb's 0/1/2/3/4 ids into the kernel or framework
adapter.

```text
launcher: UID ────────────────┬──────── W workers
                              └──────── INC sidecar
all PE:    SHMEM init → symmetric alloc/zero → prepare mirror → barrier
INC:                                                        └→ service.start
worker:                                                      └→ route → moe_out
```

Teardown first stops accepting new batches and waits for in-flight engine
forwards; workers close the executor, INC closes the service (ordered stop),
all PEs barrier once more, then free symmetric tensors and finalize. Mid-failure
must not treat barrier as cancel; a push-only request that was published must
complete.

`inc_sidecar.py` is a spawn entry and parent controller that does not depend on
vLLM ProcessGroup. The control protocol is
`PREPARED → READY → STOPPING → STOPPED`: after PREPARED the parent lets all
workers finish prepare/setup barrier; after READY it allows inference; on stop
it must stop the scheduler from taking new batches, drain in-flight forwards,
then send STOP and let workers enter teardown barrier. Controller timeout only
errors; it will not terminate the resident kernel process on its own.

NPU graph capture/replay is explicitly unsupported. Ordinary `torch.compile`
opaque custom-ops still enter host `worker_moe_out` each step and can mint a
new ticket; capture/replay would replay old generation/ticket, so the native
bridge hard-rejects it. First vLLM qualification must disable NPU graph until
tickets become a graph-replay-aware device counter.

`torch_weight_runtime.py` is a setup-only `PreparedWeightCache`. It distinguishes
kernel ND `w13[E,2I,H]/w2[E,H,I]` from the logical transposed layout, and only
accepts storage format ND(2) or FRACTAL_NZ(29). NZ is converted to ND once at
model load; transpose also happens once. The forward resolver checks that the
original vLLM weight device/data pointer was not replaced, then returns the
cache with no format-cast, transpose, or allocation. Any other storage format
is rejected.

Core setup call graph (UID distribution and process creation stay with the
real launcher):

```python
infos = query_all_worker_plan_info(key, config, owner, local)
heap_bytes = SymmetricHeapPolicy().heap_bytes(infos[0])
session = FusionShmemSession(
    uid, pe, W + 1, heap_bytes, infos[0].symmetric_bytes, device)

# worker: route = PreparedTorchRoute(...)
# worker: executor = PreparedTorchExecutor.from_session(
#                       route, config, worker_pes_tensor, session)
# INC:    service = PreparedIncService(key, config, owner, local,
#                                     worker_pes, session)
session.barrier()
# INC only: service.start(); workers may then enter forward.
```

## Five modes

| mode | Communication | Schedule | Use |
|---|---|---|---|
| `native_vllm` | vLLM-Ascend native | native | end-to-end external baseline |
| `serial_shmem` | worker-direct SHMEM | strict D→FFN→C | 2×2 cell |
| `serial_inc` | worker→single INC→worker | strict D→FFN→C | 2×2 cell |
| `fused_shmem` | worker-direct SHMEM | token-wave | 2×2 cell |
| `fused_inc` | worker→single INC→worker | token-wave | delivery implementation |

The first four must pass `inc_fusion_benchmark_validate_factorial_pair`. Native
vLLM may report end-to-end speedup but must not mix into INC/fusion factor
attribution.

Resource accounting needs two columns: the main table fixes W compute workers
(INC mode uses one extra card that loads no model) to isolate the communication
mechanism; the system table normalizes throughput by total NPU count so W vs
W+1 hardware cost is explicit. Do not report only the former and claim
“same-cost” inference speedup, and do not swap native vLLM to a different W
while claiming per-operator shape equivalence.

## Hard gates before official numbers

1. All four modes share the same `topk_ids/topk_weights`, expert placement, and
   route digest.
2. All four modes share the same GMM1/SwiGLU/GMM2 implementation and weight
   layout digest. Do not compare vLLM grouped-GMM against another row's Catlass
   kernel and attribute the delta to communication.
3. Route packing happens on device and is included in per-step time;
   setup/JIT/weight permute/workspace alloc are not.
4. BF16 ND weights are consumed natively by the kernel. W8A8/NZ quant semantics
   are unimplemented and must fail closed; do not transpose/format-cast every
   step or mix a quant baseline into BF16 qualification.
5. Engine setup, standalone INC sidecar, resident server, and ordered teardown
   are closed; exception paths still must not treat barrier as cancel. After
   failure, clean this control directory and confirm the NPU context is freed.
6. Timing samples must take every worker's makespan; INC mode also reports the
   INC service window.

Operator microbenchmarks freeze and replay one router output. Real vLLM runs
keep the router but freeze model, prompt, seed, schedule policy, EP placement,
KV cache, and sampling. Prefill reports input tokens/s and TTFT; decode reports
output tokens/s and TPOT, each with P50/P95/P99. Real-model hidden size is
fixed by the checkpoint; cross-hidden scans belong to synthetic operator tests
or different model configs.

## Dynamic token counts

`inc_moe_runtime.py`'s general prepared API may use capacity buckets, but one
cross-process long-lived INC service must use a single fixed capacity equal to
the engine max token count. Any 0..capacity token count reuses the
active-token vector; it need not divide a microbatch. Only an explicit HBM
budget may reject a request. `prepare()` runs only at warmup/setup; timed
forward may only `lookup()`, so first alloc/JIT is not mixed into a backend.

When actual tokens are below capacity, the tail wave is packed as a legal
zero-token descriptor. Workers may also differ in token count: sync waves take
the group's max active-token count, and INC ABI v6 uses each source's true
length to zero, check, reduce, and write back, without fake output padding for
short ranks.

Route parallel lane count is taken from device `vector_core_num` at prepare.
Forward only uses integer protocol work to pick scalar vs parallel: small
inputs and top-k=1 fall back to scalar; large enough regroup work enters
analyze/prefix/emit. There is no model-shape tuning table. On nb's 48-AIV
regression, the 20-case matrix chosen by the policy had no perf regression;
W4/T8192/K8 dropped from 10.19 ms scalar route to 3.75 ms.

## Release-candidate status

- Done: device route-pack, Torch NPU `out` bridge, prepared worker executor,
  cross-process persistent INC, runner sidecar, setup/out/teardown, and W2/W4
  device sweeps.
- Frozen: ABI 13 BF16 `fused_inc` is the current custom best path; perf knobs
  are no longer walked at runtime.
- Not done: numeric match on the same prompts as native, W8A8, replay-safe
  native graph, and the current direct-SHMEM vLLM lifecycle. Details and data:
  [`fusion_kernel_release_20260810`](../../../../../docs/inc/report/nb-borrow/fusion_kernel_release_20260810/README.md).
