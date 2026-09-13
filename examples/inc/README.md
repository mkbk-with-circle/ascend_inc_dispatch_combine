# INC Examples

当前只维护 Single-INC Pull V2 Dispatch / Combine：

```text
dispatch_combine/
├── common/platform/         # 必需的 AICore 公共头
├── single_inc/pull_combine/ # 协议、kernel、API、示例和设备测试
└── tests/single_inc/        # Host 协议与 API 测试
```

从 [快速接入](dispatch_combine/single_inc/QUICKSTART.md) 开始。历史 V1、
native stream、inline-route 和 Fusion 实验可从
`archive/pre-api-cleanup-20260913` 恢复，不参与当前构建。

## English

The maintained INC example is Single-INC Pull V2 Dispatch / Combine. Start at
[Quickstart](dispatch_combine/single_inc/QUICKSTART.md). Legacy and experimental
implementations are recoverable from `archive/pre-api-cleanup-20260913` and
are not built.
