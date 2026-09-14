# Single-INC Pull V2 快速接入

业务热路径只需要保留 `Dispatch → Expert → Combine`。所有数组均为设备指针，
每个 token 的 Top-k GPU、Expert 与 Weight 和 Hidden 一起属于该 Wave 的 Source
Slot；INC 事前不知道 token-plan。

```cpp
#include "pull_combine/inc_dc_pull_v2_api.h"

using namespace inc::dc::pull_v2::api;

SingleIncSession op{};
single_inc_create(config, native_backend, backend_context, &op);

DispatchInput dispatch{};
dispatch.send_buffer = token_hidden;
dispatch.send_token_ids = token_ids;
dispatch.destination_gpus = topk_destination_gpus;
dispatch.expert_ids = topk_expert_ids;
dispatch.expert_weights = topk_expert_weights;
dispatch.token_count = token_count;
dispatch.assignment_count = token_count * topk;
dispatch.fixed_topk = topk;

BatchHandle batch{};
Completion dispatch_done{};
shmem_dispatch_alltoall_inc<DataType::BF16>(
    &op, wave, &dispatch, &dispatch_output,
    stream, &batch, &dispatch_done);

// 如当前 stream/依赖关系不能表达消费顺序，再显式等待 Completion。
completion_wait(&op, dispatch_done);
run_grouped_gemm_and_local_reduce(dispatch_output, fp32_partials, stream);

Completion combine_done{};
shmem_combine_alltoall_inc<DataType::FP32>(
    &op, &batch,
    fp32_partials, partial_count, partial_token_ids,
    token_output, token_capacity, &output_count,
    stream, &combine_done);
completion_wait(&op, combine_done);

single_inc_destroy(&op);
```

`native_backend` 在部署时绑定 transport/device launcher，不进入逐 token 路由规划。
应用热路径只接触 Session、输入/输出、BatchHandle 和 Completion。

完整、可独立运行并打印 fan-out/reduction 结果的参考程序见
[`pull_combine/inc_dc_pull_v2_api_example.cpp`](pull_combine/inc_dc_pull_v2_api_example.cpp)。
协议细节与两张流程图见 [`pull_combine/FLOW.md`](pull_combine/FLOW.md)。
