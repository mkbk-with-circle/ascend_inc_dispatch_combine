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
