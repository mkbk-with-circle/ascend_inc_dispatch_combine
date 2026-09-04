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
- 当前仓库提供稳定 Frontend、Host Reference Example 和完整设备 qualification；
  真实推理热路径还需要把框架 Router 的 Device Arrays 绑定到 Pull V2 Device Pack
  Adapter。API 不会静默把 Device Route 拷回 CPU。

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

- 正式性能与稳定性：
  [`pull_v2_qualified_20260904`](../../../../../docs/inc/report/nb-borrow/pull_v2_qualified_20260904/README.md)
- 非对称压力与交叠调优：
  [`pull_v2_overlap_stress_20260905`](../../../../../docs/inc/report/nb-borrow/pull_v2_overlap_stress_20260905/README.md)

当前 nb-borrow 的 W2/W4 128 MiB、top-k2 Dispatch/Combine 均通过既定 Gate；任意
Ragged/Hotspot 路由走正确性优先的安全重整路径，性能仍是后续优化项。
