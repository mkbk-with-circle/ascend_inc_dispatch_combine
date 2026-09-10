# Single-INC 源分区 Dispatch / Combine

当前源分区设备入口位于[`single_inc/pull_combine/`](single_inc/pull_combine/README.md)，
旧紧凑布局设备入口保留作对照。各origin位置由预留容量决定，独立完成。

```text
Dispatch: A_s READY → INC解析本源 → PUT到B_s固定分区 → 本源Completion/ACK
Combine : B_s Notice → 真实D Journal → GET本分区partial → reduce → PUT A_s
```

- [快速接入](single_inc/QUICKSTART.md)
- [源分区协议与设备入口](single_inc/pull_combine/SOURCE_PARTITIONS.md)
- [API](single_inc/pull_combine/inc_dc_pull_v2_api.h)
- [完整示例](single_inc/pull_combine/inc_dc_pull_v2_api_example.cpp)
- [流程图](single_inc/pull_combine/FLOW.md)
- [Sweep 状态](single_inc/SWEEP_STATUS.md)

旧 API、V1 backend、inline-route 和 Fusion Kernel 已移入 Git 历史。
