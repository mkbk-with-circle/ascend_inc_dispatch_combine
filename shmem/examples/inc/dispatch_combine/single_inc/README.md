# Single INC：源rank独立分区

当前开发入口位于[pull_combine/](pull_combine/README.md)。
Dispatch按源origin预留独立目标空间，Combine直接读取本源真实设备Journal。
旧紧凑布局入口与旧报告保留作回归对照。

- [源分区协议与内存布局](pull_combine/SOURCE_PARTITIONS.md)
- [详细流程图](pull_combine/FLOW.md)
- [新设备入口与真机示例](pull_combine/README.md)
- [框架前端适配说明](QUICKSTART.md)
- [测试状态](SWEEP_STATUS.md)

历史V1/Fusion从archive/pre-minimal-v1-fusion-20260904恢复。
源分区之前的代码从archive/pre-source-partitions-a230a1b恢复。

## English

The development path uses fixed per-origin destination partitions and a
Combine kernel that consumes the actual device Dispatch Journal.
See the linked protocol and device example. Legacy compact-layout entries
are retained for regression comparisons; frontend backend binding remains
separate from the qualified device test path.
