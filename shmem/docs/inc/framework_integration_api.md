# 单 INC 框架接入接口

## 当前结论

当前原型只保留一个面向业务和框架适配层的入口：
[`inc_dc_single_inc.hpp`](../../examples/inc/dispatch_combine/common/api/inc_dc_single_inc.hpp)。
接入方不应逐层调用 Framework / Easy / Inference，也不应调用 benchmark main。

```cpp
using namespace inc::dc;

auto op = single_inc_create(config);
auto batch = op.dispatch(token_input, expert_input, route, stream);
run_grouped_gemm(expert_input, expert_output, stream);
op.combine(batch, expert_output, token_output, stream);
single_inc_destroy(op);
```

公开流程只有：

```text
create -> dispatch -> expert compute -> combine -> destroy
```

完整可运行示例见
[`inc_dc_single_inc_api_example.cpp`](../../examples/inc/dispatch_combine/common/examples/single_inc_api/inc_dc_single_inc_api_example.cpp)。

## 一次性初始化与热路径

框架适配层在 worker 初始化阶段负责：

1. 按当前 process group、拓扑和固定 shape 填好 `inc_dc_single_inc_config_t`；
2. 绑定 native composite backend、device allocator 和常驻 INC service；
3. 调用一次 `single_inc_create(config)`，并在模型生命周期内复用返回的 `SingleInc`。

每个 MoE batch 的热路径只负责：

1. 将 router 结果编译为本 rank 的 `SingleIncRoute`；
2. `dispatch()` 将 token fan-out 到 expert buffer，并返回 `SingleIncBatch`；
3. 在调用方 stream 上运行 grouped GEMM/activation；
4. `combine()` 复用该 batch 捕获的精确 route，完成加权归约与回传。

`SingleIncBatch` 自动持有 generation 和 Dispatch route；Combine 成功后自动释放，
异常退出时由析构回收。业务代码不管理 request、workspace lease 或 route handle。

## 当前边界

- 当前公开 API 是简单的同步生命周期封装；async request、query/cancel、plan/stats
  暂不公开，需要真实框架集成证明有必要后再增加。
- `inc_dc_single_inc_api.h` 是 C++ 薄封装和 native controller 的内部 C 桥接，
  不是另一套用户 API。
- `inc_dc_framework_c_api.*`、`inc_dc_easy_api.*`、`inc_dc_inference_api.*`
  仍是已验证 V1 runtime 的内部实现和回归测试依赖，不是接入路径。
- Fusion Kernel 使用自己的 prepared API：
  [`inc_fusion_api.h`](../../examples/inc/fusion_kernel/ascend/inc_fusion_api.h)，
  不应经由旧 Framework/Easy 层调用。

## Megatron / vLLM 适配建议

适配器只需要持有一个长生命周期 `SingleInc`，并完成三类张量转换：

- router 输出 -> `SingleIncRoute`；
- Dispatch 输出 -> expert-major/padded grouped-GEMM 输入；
- expert 输出 -> Combine 输入，输出写回原 token 顺序。

TP×EP 场景只把 EP worker 注册到该 communicator；逻辑 rank 到物理 NPU、唯一
INC 以及 AIV cohort 由 native runtime/profile 决定，不应写进模型代码。动态 shape、
graph capture 和公开异步接口仍需分别通过正确性与生命周期 gate 后再扩展。

更底层的协议、AIV 和 workspace 说明见
[`single_inc/QUICKSTART.md`](../../examples/inc/dispatch_combine/single_inc/QUICKSTART.md)；
它们不属于业务调用步骤。
