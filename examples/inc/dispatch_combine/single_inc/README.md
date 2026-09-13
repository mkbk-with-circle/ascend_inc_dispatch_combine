# Single-INC Pull V2

```text
Dispatch: Worker PUT Header/metadata → READY → INC GET hidden → INC PUT B
Compute : B expert compute + same-GPU local weighted reduce
Combine : B PUT READY descriptor → Notice → INC GET partials
          → FP32 reduce → INC PUT original A
```

- [快速接入](QUICKSTART.md)
- [实现、API 与构建](pull_combine/README.md)
- [协议流程图](pull_combine/FLOW.md)
- [Dispatch 设计](pull_combine/PULL_DISPATCH_V2_DESIGN.md)

当前源码只保留这一套 Dispatch/Combine 实现。归档标签
`archive/pre-api-cleanup-20260913` 保存清理前代码。

## English

This directory contains the sole maintained Single-INC Pull V2 implementation.
Use [QUICKSTART.md](QUICKSTART.md) for integration and
[pull_combine/README.md](pull_combine/README.md) for build and validation.
