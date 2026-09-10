# INC Examples

当前开发重点是Single-INC源rank独立分区；旧紧凑布局入口保留作回归对照。
新协议与已完成验证见[源分区说明](dispatch_combine/single_inc/pull_combine/SOURCE_PARTITIONS.md)。

```text
dispatch_combine/
├── common/platform/        # Pull V2 使用的最小 AICore 公共头
├── single_inc/pull_combine # 协议、kernel、API、示例和 qualification runner
└── tests/single_inc/       # Pull V2 Host/API 测试
```

从 [`dispatch_combine/single_inc/QUICKSTART.md`](dispatch_combine/single_inc/QUICKSTART.md)
开始阅读。历史 V1 与 Fusion Kernel 仅存在于归档标签中，不属于当前构建。
