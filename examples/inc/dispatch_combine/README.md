# Single-INC Pull V2 Dispatch / Combine

当前维护路径位于 [single_inc/pull_combine](single_inc/pull_combine/README.md)。

```text
Dispatch: Worker PUT Header/metadata → READY
          → INC local parse → GET hidden → fan-out PUT → Completion/ACK
Combine : Worker PUT READY descriptor → Notice
          → INC local validation → GET partial → FP32 reduce → owner PUT
```

入口：

- [快速接入](single_inc/QUICKSTART.md)
- [公共 API](single_inc/pull_combine/inc_dc_pull_v2_api.h)
- [完整示例](single_inc/pull_combine/inc_dc_pull_v2_api_example.cpp)
- [协议流程](single_inc/pull_combine/FLOW.md)
- [测试与口径](single_inc/pull_combine/tests/ACCEPTANCE_PLAN.md)

旧实现可从 `archive/pre-api-cleanup-20260913` 恢复。

## English

The maintained path is Single-INC Pull V2. Workers push control metadata before
READY/Notice; INC then validates local control state and pulls payload. The
links above are the API, example, protocol, and qualification entry points.
