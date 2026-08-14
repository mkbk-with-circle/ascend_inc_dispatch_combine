# Single-INC API / 单 INC API

## 中文

当前原型只提供一个学习和业务入口：
[`inc_dc_single_inc.hpp`](inc_dc_single_inc.hpp)。

```cpp
auto op = single_inc_create(config);
auto batch = op.dispatch(token_input, expert_input, route, stream);
run_grouped_gemm(expert_input, expert_output, stream);
op.combine(batch, expert_output, token_output, stream);
single_inc_destroy(op);
```

- `route` 由端侧 planner 生成，包含 device token-plan 和本 rank 的 D/C 行数。
- `batch` 保存 generation 和精确 Dispatch route；Combine 后自动释放。
- 当前没有公开 async request、query/cancel、plan/stats 等高级接口；需要时再添加。

[`inc_dc_single_inc_api.h`](inc_dc_single_inc_api.h) 是 C++ 薄封装与 native
controller 使用的内部桥接头，不是新人调用入口。`Framework/Easy/Inference` 和两个
planner 目前只是实现及回归测试依赖，阅读算子时无需沿这些层逐级进入。

完整可运行示例：
[`../examples/single_inc_api/inc_dc_single_inc_api_example.cpp`](../examples/single_inc_api/inc_dc_single_inc_api_example.cpp)。

## English

The current prototype has one application and learning entry point:
[`inc_dc_single_inc.hpp`](inc_dc_single_inc.hpp).

```cpp
auto op = single_inc_create(config);
auto batch = op.dispatch(token_input, expert_input, route, stream);
run_grouped_gemm(expert_input, expert_output, stream);
op.combine(batch, expert_output, token_output, stream);
single_inc_destroy(op);
```

The public flow is only `create -> dispatch -> compute -> combine -> destroy`.
`SingleIncBatch` owns generation and route lifetime. Async requests, query /
cancel, plan, and stats APIs are intentionally not public yet.

`inc_dc_single_inc_api.h` is an internal bridge used by the thin C++ wrapper
and native controller. Framework/Easy/Inference and the host planners remain
implementation and regression-test dependencies; users do not call through
them.
