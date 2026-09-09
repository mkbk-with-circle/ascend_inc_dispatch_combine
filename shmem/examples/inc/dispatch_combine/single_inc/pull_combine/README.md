# Single-INC Pull Dispatch / Combine V2

本目录只保留当前 Pull V2 实现。历史 V1、旧 endpoint、调优 probe 和 Fusion
Kernel 已从当前树删除，可通过远端归档标签完整恢复：

```bash
git switch --detach archive/pre-minimal-v1-fusion-20260904
```

当前最小树之前的完整状态即该归档标签；无需在工作树中保留重复源码。

## 最短应用接口

公共头文件：[`inc_dc_pull_v2_api.h`](inc_dc_pull_v2_api.h)。

```cpp
using namespace inc::dc::pull_v2::api;

SingleIncSession session;
single_inc_create(config, backend_ops, backend_context, &session);

BatchHandle batch;
Completion dispatch_done;

shmem_dispatch_alltoall_inc<DataType::BF16>(
    &session,
    wave,
    token_hidden,
    token_ids,
    topk_destination_gpus,
    topk_expert_ids,
    topk_expert_weights,
    token_count,
    topk,
    &dispatch_output,
    stream,
    &batch,
    &dispatch_done);

completion_wait(&session, dispatch_done);
// Grouped GEMM / Expert FFN，并在每个 B 上先做 local weighted reduce。
run_experts_and_local_reduce(dispatch_output, fp32_partials, stream);

Completion combine_done;
shmem_combine_alltoall_inc<DataType::FP32>(
    &session,
    &batch,
    fp32_partials,
    partial_count,
    partial_token_ids,
    fp32_token_output,
    token_capacity,
    &output_count,
    stream,
    &combine_done);

completion_wait(&session, combine_done);
single_inc_destroy(&session);
```

完整可编译示例：
[`inc_dc_pull_v2_api_example.cpp`](inc_dc_pull_v2_api_example.cpp)。示例覆盖初始化、
Dispatch、Expert 计算、Combine、结果打印和资源释放。

### API 能力边界

- Dispatch 输入支持 FP16、BF16、FP32 Hidden。
- Route Weight 使用 FP32 Metadata。
- 当前正式 Combine 只接受 FP32 Local Partial，并执行 FP32 Reduction；模板入口
  `shmem_combine_alltoall_inc<DataType::FP32>` 会在编译期拒绝其他类型，避免隐式
  精度转换。
- 同一 GPU 上多个 Expert 的 Weight 应用与 Local Reduce 由调用方在 Combine 前完成。
- `BatchHandle` 持有 generation-scoped Journal 生命周期；不可跨 Session、重复消费或
  在 Combine 前复用 Ring Slot。
- API 是异步 enqueue 接口；设备错误通过 `Completion` 查询或等待。
- 成功的 query/wait 消费 Completion；重复或跨 Session 使用返回 STALE_HANDLE。
  Dispatch 完成后才可提交对应 Combine/release；Combine 完成前 Ring Slot 仍被占用，
  destroy 返回 BUSY。Timeout 保留资源，不能视为释放 ACK。
- 同一 Session 的 Host 调用需由调用方串行化；不同 Slot 的设备请求仍可并发。
- `DispatchOutput` 可提供 `recv_rows`、`recv_assignments`、`recv_expert_counts`
  三个设备结果视图，供专家计算和本地加权归约使用。后端必须填充调用方请求的视图。
- 当前仓库提供稳定 Frontend、Host Reference Example 和完整设备 qualification；
  真实推理热路径还需要把框架 Router 的 Device Arrays 绑定到 Pull V2 Device Pack
  Adapter。API 不会静默把 Device Route 拷回 CPU。

### 实际接通状态

当前 NPU E2E 程序直接启动 kernel；公开 API 尚无随库提供的 NPU BackendOps。
Combine 测试的 Pull Index 由 Host 参考布局构造，尚未接到真实 Dispatch 产生的
设备 Journal。参考示例仅在 CPU 模拟所有目标 GPU：按唯一目标去重 Hidden，专家
阶段应用一次 Weight 并本地归约，Combine 只累加 partial，并执行 golden 检查。
该示例不测量网络，不是公开 API 的真机端到端验证。

## 协议概要

```text
Dispatch：A READY → INC GET(metadata + 一份 hidden) → unique-destination PUT → B
Compute： B Expert FFN → same-GPU local weighted reduce → FP32 partial
Combine： B Notice → INC GET READY/partial → reduce → selective owner PUT → A
```

详细的两张独立流程图见 [`FLOW.md`](FLOW.md)。

关键不变量：

- Worker 每个 Wave 只发布一次 READY/Notice。
- Dispatch 中每个 Token Hidden 从源 Worker 只 GET 一份。
- 同一目标 GPU 上多个 Expert 共享该 Hidden Row。
- Combine READY Record 保留在 B 端；INC 收到64B Notice 后主动 GET。
- Journal 使用 `(owner_rank, owner_row)` 作为主键，不以 Token ID 作为唯一身份。
- Dispatch/Combine 使用互不重叠的动态半 AIV；并发时可对 Combine 启用 transport
  lane governor，但 Solo 默认不变。

## 当前源码

| 文件 | 用途 |
|---|---|
| `inc_dc_pull_dispatch_v2.{h,cpp}` | Dispatch Host 协议、Slot 与 Journal 编译 |
| `inc_dc_pull_dispatch_v2_abi.h` | Dispatch/Journal 设备 ABI |
| `inc_dc_pull_dispatch_v2_device_kernel.cpp` | READY→GET→Parse→Fan-out 数据面 |
| `inc_dc_pull_combine_v2.{h,cpp}` | Combine Notice、READY 和 Pull Index 协议 |
| `inc_dc_pull_combine_v2_device_kernel.cpp` | Partial GET→Reduction→Owner PUT 数据面 |
| `inc_dc_pull_v2_api.{h,cpp}` | 最短应用 Frontend API |
| `inc_dc_pull_*_e2e.cpp` | Host/真机资格测试，不是应用调用路径 |
| `tests/*.py` | Gate、Overlap、非对称和压力矩阵 Runner |

## 构建与运行

```bash
cmake -S . -B /tmp/shmem-pull-v2-build \
  -DUSE_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release -DSOC_TYPE=Ascend910B

cmake --build /tmp/shmem-pull-v2-build -j8 --target \
  inc_dc_pull_dispatch_v2_tests \
  inc_dc_pull_combine_v2_tests \
  inc_dc_pull_v2_api_tests \
  inc_dc_pull_v2_api_example \
  inc_dc_pull_dispatch_v2_device_e2e \
  inc_dc_pull_combine_v2_npu_e2e

/tmp/shmem-pull-v2-build/bin/inc_dc_pull_v2_api_example
```

## 结果

### Dispatch 统一源段流水

均匀和随机路由共用 `RelaySourceRuns`：去重目标 GPU，确定各目标行偏移，
拉取一份源 hidden，再向所有唯一目标发送。均匀路由合并为连续长段；随机路由
按 token 生成短段并查询已生成的目标行。移除固定 K1/K2/K4 fan-out 函数以及
整批拉齐后逐目标标量重整的独立数据路径。

对齐段使用同一个 ping/pong 搬运循环。非对齐行采用单写入者的精确宽度原语，
避免多 AIV 对同一缓存行写入；统一主流程并不意味着取消必要的尾块处理。
READY、完整路由校验、Journal、成功 Completion / Source ACK 的语义未改变。

relay 时间戳先保存在 AIV 本地，所有布局生产者汇合后才写回，防止覆盖同一
缓存行上的布局完成标志。移植时仍须遵守 API 的容量和 ABI worker 上限；
当前设备验证范围仅 W2/W4。

性能按 fan-out 下行字节 / 完整调用时间，使用同平面基线 A/B 检查128 MiB正式
矩阵的均值和最低值下降均小于1%；小消息与错误输入单独检查正确性和有限完成。
测试入口为 `tests/pull_v2_dispatch_unified.py`。
结果见 [Dispatch统一对照](../../../../../docs/inc/report/nb-borrow/dispatch_unified_20260909/README.md)。

### Combine 统一流水候选

当前开发分支将 K2、K4 和一般贡献数合并为同一个设备执行函数。
每个 token 从 Journal 读取贡献数（0..worker_count），校验贡献链和输出地址，
然后复用缓存按 hidden tile 拉取、累加、回传。同批 token 可有不同 K；
同 GPU 多专家仍先在端侧合并为一份 partial。

所有情况使用两块输入和两块交替输出缓冲，每块 6 KiB。奇数 K 的最后一份
贡献单独累加，零贡献输出零。当前仍保留 ABI 的 128 worker 上限；真机测试
范围仅为 nb 的 W2/W4，不能把 Host 的较大规模校验视为真机扩展验证。

候选版尚未达到 raw gate，K2 对照存在小幅性能回退；尚未替换远程发布版本。
见 [统一流水实验](../../../../../docs/inc/report/nb-borrow/combine_unified_candidate/README.md)。

- 当前单方向有效带宽、拓扑与稳健性：
  [`pull_v2_directional_20260909`](../../../../../docs/inc/report/nb-borrow/pull_v2_directional_20260909/README.md)

旧 GET+PUT 相加带宽口径已删除。当前报告同时列出 raw gate 与实测 transport roof；
Ragged/Hotspot 路由走正确性优先的安全重整路径，性能仍是后续优化项。
