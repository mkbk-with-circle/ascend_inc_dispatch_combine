# INC Dispatch / Combine

当前维护的独立算子是 **Single-INC Pull V2**：多个 Worker 与唯一 INC 建链，
Dispatch 和 Combine 的 payload 都由 INC 主动 GET，再由 INC PUT 到目标 Worker。

新人只需要沿这一条路径阅读：

1. [`single_inc/QUICKSTART.md`](single_inc/QUICKSTART.md)：最短调用流程；
2. [`single_inc/pull_combine/README.md`](single_inc/pull_combine/README.md)：API、构建与门禁；
3. [`single_inc/pull_combine/FLOW.md`](single_inc/pull_combine/FLOW.md)：Dispatch/Combine 两张流程图；
4. [`single_inc/pull_combine/inc_dc_pull_v2_api_example.cpp`](single_inc/pull_combine/inc_dc_pull_v2_api_example.cpp)：完整可编译示例。

历史 Push V1、endpoint 与诊断 probe 不在当前源码树中。回退入口：

```text
single-inc-v1-legacy
single-inc-pull-v2-qualified-pre-api
```

`common/` 中的框架契约以及 `fusion_kernel/` 的融合算子是独立模块，不是调用 Pull
V2 独立算子的必经层。

## English

The maintained standalone operator is **Single-INC Pull V2**. The INC pulls
both Dispatch and Combine payloads, then publishes results to destination
workers. Start with `single_inc/QUICKSTART.md`; the authoritative API, build
commands, diagrams, and complete example live under `single_inc/pull_combine/`.
Legacy Push V1 and diagnostics are recoverable from the tags above and are not
part of the current source path.
