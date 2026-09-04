# Single-INC Pull V2 Dispatch / Combine

当前唯一实现位于 [`single_inc/pull_combine/`](single_inc/pull_combine/README.md)。

```text
Dispatch: Worker READY → INC GET token+route → parse/reorder → INC PUT B
Combine : Worker Notice → INC GET FP32 partials → reduce → INC PUT owner
```

- [快速接入](single_inc/QUICKSTART.md)
- [API](single_inc/pull_combine/inc_dc_pull_v2_api.h)
- [完整示例](single_inc/pull_combine/inc_dc_pull_v2_api_example.cpp)
- [流程图](single_inc/pull_combine/FLOW.md)
- [Sweep 状态](single_inc/SWEEP_STATUS.md)

旧 API、V1 backend、inline-route 和 Fusion Kernel 已移入 Git 历史。
