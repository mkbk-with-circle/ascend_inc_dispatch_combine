# INC 当前报告索引

当前 Single-INC Pull V2 只保留 nb-borrow 的可复现汇总证据：

- [`single_inc_LIVE_STATUS.md`](single_inc_LIVE_STATUS.md)：当前协议、Gate、正式结果和开放项；
- [`ACTIVE_HW_PROFILE.md`](ACTIVE_HW_PROFILE.md)：唯一活跃环境指针；
- [`nb-borrow/pull_v2_qualified_20260904`](nb-borrow/pull_v2_qualified_20260904/README.md)：
  W2/W4 正式带宽、正确性、稳定性与扩展性；
- [`nb-borrow/pull_v2_overlap_stress_20260905`](nb-borrow/pull_v2_overlap_stress_20260905/README.md)：
  D+C 交叠、不同 size 与非对称压力矩阵；
- [`nb-borrow/FUSION_KERNEL_RESULTS.md`](nb-borrow/FUSION_KERNEL_RESULTS.md)：Fusion Kernel 的 nb 结果索引。

当前实现没有其他集群的测试结论。仓库不保存逐 PE JSONL、原始日志、profiler trace
或重复中间结果；这些留在实验机本地，Git 只保留汇总表、测试口径和复现入口。
