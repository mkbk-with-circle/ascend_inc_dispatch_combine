# Single-INC Pull V2 Dispatch/Combine

本仓库保留一套基于 SHMEM 的昇腾 Single-INC Pull V2 实现。`shmem/` 包含基础库、当前 Dispatch/Combine 设备路径、公共 API、示例和验证测试。

Dispatch 使用 `metadata PUT → READY → INC 本地解析 → hidden GET → fan-out`；Combine 使用 `READY 描述 PUT → Notice → 本地校验 → partial GET → FP32 归约 → owner PUT`。

从 [快速接入](shmem/examples/inc/dispatch_combine/single_inc/QUICKSTART.md)、[API 与构建说明](shmem/examples/inc/dispatch_combine/single_inc/pull_combine/README.md) 和 [当前验证报告](shmem/docs/inc/report/nb-borrow/pull_v2_current_20260913/README.md) 开始。当前报告只将与最终 Dispatch 库匹配的 W2/W4 真机数据列为现行结果；Combine 的新设备带宽尚待真机复测。

旧实验及历史结果可通过 Git 历史查阅；当前主线仅展示交付所需文件。构建产物和原始运行日志不纳入 Git。

## English

This repository contains one maintained Ascend Single-INC Pull V2 implementation built on SHMEM. The `shmem/` directory includes the base library, current Dispatch and Combine device paths, public API, examples, and validation tests.

Dispatch follows `metadata PUT → READY → local INC parse → hidden GET → fan-out`. Combine follows `READY descriptor PUT → Notice → local validation → partial GET → FP32 reduction → owner PUT`.

Start with the [quickstart](shmem/examples/inc/dispatch_combine/single_inc/QUICKSTART.md), [API and build guide](shmem/examples/inc/dispatch_combine/single_inc/pull_combine/README.md), and [current validation report](shmem/docs/inc/report/nb-borrow/pull_v2_current_20260913/README.md). The current report treats only W2/W4 device measurements matching the final Dispatch library as current results; device bandwidth for the new Combine path still needs to be measured.

Earlier experiments and results remain available in Git history. The current main branch shows only delivery files; build artifacts and raw runtime logs are excluded.
