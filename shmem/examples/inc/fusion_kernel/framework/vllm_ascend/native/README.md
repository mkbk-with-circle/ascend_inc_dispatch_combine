# 原生 Torch 路由与 worker 生命周期桥接

基础构建注册三个只写预分配输出的路由 NPU op：

- `inc_fusion_native::route_count_out`
- `inc_fusion_native::route_pack_out`
- `inc_fusion_native::route_pack_parallel_out`

三者直接使用当前 PyTorch NPU stream，不分配 tensor、不做 host 同步。桥接必须调用
`NPUStream::stream()`，使 torch_npu 软件 task queue 先进入同一 ACL stream；禁止使用
`stream(false)` 绕过前序 PyTorch producer。`local_counts`
为 `[wave_capacity, expert_count+1]`；最后一列在 wave 0 保存 active-token 数。框架随后
在 vLLM EP device group 内只做一次固定容量 `all_gather_into_tensor`，同时得到全局专家
计数和各 rank 长度；payload 不经过 HCCL。`route_pack_out` 产生 Fusion ABI 的
rows/assignments/group-lists/waves。

提供 `FUSION_API_LIBRARY` 后，同一个共享库额外注册：

- `inc_fusion_native::plan_info`：CPU setup op，在 SHMEM init 前返回精确 allocation/ABI；
- `inc_fusion_native::worker_prepare`：setup 时创建固定容量 plan/executor；
- `inc_fusion_native::worker_moe_out`：只向当前 NPU stream enqueue，输出写入调用方的
  prepared tensor，不分配、不同步；
- `inc_fusion_native::worker_destroy`：engine teardown 时等待本 executor 的 event ring 并
  释放资源。
- `inc_fusion_native::service_prepare/start/destroy`：供不加载模型的 INC sidecar 创建、在
  W+1 PE setup barrier 后启动并有序停止常驻 service。

`worker_prepare` 必须由 engine setup 显式传入 SHMEM bootstrap 已建立的
`symmetric_base/symmetric_bytes/ffts_addr`；桥接不会在 forward 内隐式初始化 SHMEM。
当前 native executor 只接受 `serial_inc=2` 和 `fused_inc=4`，SHMEM 两个基线在其独立的
worker-direct backend 完成前会明确拒绝。一个长驻 INC service 使用一个按 engine 最大
token 数建立的固定 plan；0 到 capacity 的动态/不均匀 batch 复用它，不能为不同 capacity
各建一个从 ticket 1 开始的 executor 去共享同一 service。
Python ACLSHMEM binding 同时导出 `aclshmem_barrier_all`；它只用于 setup/teardown，不进入
逐请求热路径。

一个 worker handle 只允许绑定一个 ACL stream，因为 prepared route/output 是单份复用
buffer；同流顺序可保护它们，跨流并发则会被明确拒绝。NPU graph capture 同样被拒绝：
当前 generation/ticket 由 host 每轮构造，直接 replay 捕获图会重复旧 ticket。这里的拒绝
是正确性 gate，不是性能回退开关。

`route_pack_out` 是单 AIV 保守回退；`route_pack_parallel_out` 按
analyze → deterministic prefix → emit 三个阶段使用设备 live AIV。每个 lane 的 histogram
和前缀状态按 64B 独占，32B row/assignment 先写 UB 再由 MTE3 输出，避免 lane 边界发生
scalar-cache false sharing。两条路径输出逐字节相同。框架根据协议工作量
`T*K*(K-1)` 和 live AIV 数选择路径，不保存 W/T/K case 表，也不在 forward 查询硬件。

route-pack kernel 必须在与 vLLM 容器相同的 CANN 版本下构建，不能直接复用宿主机另一套
CANN 生成的二进制。示例：

```bash
cmake -S /path/to/shmem -B /tmp/shmem-build-vllm \
  -DUSE_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release -DSOC_TYPE=Ascend910B
cmake --build /tmp/shmem-build-vllm --target inc_fusion_route_pack_kernel -j8

cmake -S native -B /tmp/inc-fusion-torch \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DROUTE_PACK_LIBRARY=/tmp/shmem-build-vllm/lib/libinc_fusion_route_pack_kernel.so \
  -DFUSION_API_LIBRARY=/tmp/shmem-build-vllm/lib/libinc_fusion_api.so \
  -DASCEND_INCLUDE_DIR="$ASCEND_HOME_PATH/include"
cmake --build /tmp/inc-fusion-torch -j8
```

加载使用 `torch.ops.load_library(...)`。`torch_fusion_runtime.PreparedTorchExecutor`
封装 setup/teardown，`torch_registration` 再发布 graph-visible、带 prepared output alias
的 `inc_fusion::moe`。route kernel、fusion API 与 Torch-NPU 必须用同一目标 CANN 构建；
禁止把宿主 CANN 9.1 二进制加载进 CANN 8.x 的 vLLM 进程。

---

## English

# Native Torch route and worker-lifetime bridge

The base build registers three write-into-preallocated-output route NPU ops:

- `inc_fusion_native::route_count_out`
- `inc_fusion_native::route_pack_out`
- `inc_fusion_native::route_pack_parallel_out`

They use the current PyTorch NPU stream, allocate no tensors, and do no host
sync. The bridge must call `NPUStream::stream()` so the torch_npu software task
queue enters the same ACL stream first; `stream(false)` must not skip a prior
PyTorch producer. `local_counts` is `[wave_capacity, expert_count+1]`; the last
column stores the active-token count on wave 0. The framework then does one
fixed-capacity `all_gather_into_tensor` inside the vLLM EP device group, getting
global expert counts and per-rank lengths together; payload does not go through
HCCL. `route_pack_out` produces Fusion ABI rows/assignments/group-lists/waves.

When `FUSION_API_LIBRARY` is provided, the same shared library also registers:

- `inc_fusion_native::plan_info`: CPU setup op returning exact allocation/ABI
  before SHMEM init;
- `inc_fusion_native::worker_prepare`: create a fixed-capacity plan/executor at
  setup;
- `inc_fusion_native::worker_moe_out`: enqueue onto the current NPU stream into
  the caller's prepared tensor; no alloc, no sync;
- `inc_fusion_native::worker_destroy`: wait this executor's event ring and free
  at engine teardown;
- `inc_fusion_native::service_prepare/start/destroy`: for a model-free INC
  sidecar to create, start after the W+1 PE setup barrier, and stop in order.

`worker_prepare` must receive `symmetric_base/symmetric_bytes/ffts_addr` already
established by SHMEM bootstrap. The bridge will not implicitly init SHMEM on
the forward path. The native executor currently accepts only `serial_inc=2` and
`fused_inc=4`; the two SHMEM baselines are rejected until their independent
worker-direct backends exist. One long-lived INC service uses one fixed plan
sized to the engine max token count; dynamic/uneven batches from 0 to capacity
reuse it. Do not build one executor per capacity starting at ticket 1 against
the same service. The Python ACLSHMEM binding also exports
`aclshmem_barrier_all`; it is setup/teardown only, not on the per-request hot
path.

A worker handle may bind only one ACL stream, because prepared route/output are
single reused buffers. Same-stream order protects them; cross-stream concurrency
is rejected. NPU graph capture is also rejected: generation/ticket are built on
the host each step, and replaying a captured graph would repeat old tickets.
That reject is a correctness gate, not a performance fallback.

`route_pack_out` is the conservative single-AIV fallback;
`route_pack_parallel_out` uses live device AIVs in analyze → deterministic
prefix → emit. Each lane's histogram and prefix state is 64B exclusive; 32B
row/assignment data is written to UB then emitted by MTE3 to avoid scalar-cache
false sharing at lane boundaries. Both paths are byte-identical. The framework
picks a path from protocol work `T*K*(K-1)` and live AIV count; it stores no
W/T/K case table and does not query hardware on forward.

The route-pack kernel must be built against the same CANN version as the vLLM
container. Do not reuse a host binary from another CANN. Example:

```bash
cmake -S /path/to/shmem -B /tmp/shmem-build-vllm \
  -DUSE_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release -DSOC_TYPE=Ascend910B
cmake --build /tmp/shmem-build-vllm --target inc_fusion_route_pack_kernel -j8

cmake -S native -B /tmp/inc-fusion-torch \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DROUTE_PACK_LIBRARY=/tmp/shmem-build-vllm/lib/libinc_fusion_route_pack_kernel.so \
  -DFUSION_API_LIBRARY=/tmp/shmem-build-vllm/lib/libinc_fusion_api.so \
  -DASCEND_INCLUDE_DIR="$ASCEND_HOME_PATH/include"
cmake --build /tmp/inc-fusion-torch -j8
```

Load with `torch.ops.load_library(...)`.
`torch_fusion_runtime.PreparedTorchExecutor` wraps setup/teardown;
`torch_registration` then publishes graph-visible `inc_fusion::moe` with a
prepared output alias. Route kernel, fusion API, and Torch-NPU must be built
for the same target CANN. Do not load a host CANN 9.1 binary into a CANN 8.x
vLLM process.
