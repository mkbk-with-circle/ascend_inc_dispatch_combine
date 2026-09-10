# Single-INC当前状态：源rank独立分区

当前开发分支codex/source-partitioned-protocol，回退点archive/pre-source-partitions-a230a1b。
当前按用户要求先更新PPT与文档，内核优化暂停。

## 当前协议

D的目标buffer、metadata、Journal和完成通知按原始源origin隔离。
固定容量决定各源位置，本源局部布局不等待其他源的READY或实际行数。
B可逐源分区进行专家计算、本地归约并发布Notice。
新C直接消费该源的真实设备D Journal，等待实际贡献后归约并回传原始owner。

[协议及内存布局](../../../examples/inc/dispatch_combine/single_inc/pull_combine/SOURCE_PARTITIONS.md) |
[抽象流程](../../../examples/inc/dispatch_combine/single_inc/pull_combine/FLOW.md) |
[API接通状态](../API_COMPLETION_STATUS.md)

## 最近已完成结果

W4、H=8192、K2对称，同卡组平面A：

| 算子 | 旧平均 GB/s（3+10） | 分区最低 GB/s（1+2） | 分区平均 GB/s（1+2） |
|---|---:|---:|---:|
| Dispatch | 69.596 | 73.872 | 73.921 |
| Combine | 77.303 | 71.376 | 71.483 |

D短测提高，C仍回退，尚未通过完整无回退验收。
W2/W4基本真实D→C、空源、尾部、3轮ring复用和延迟READY独立性检查已通过。
后续Combine并行Journal校验改动尚未完成设备复测。

## 链路参照

当前讲解不再使用旧raw百分比gate。已有多打一put-only定标均值：
W2=42.734、W4=85.478 GB/s。它们是链路实测峰值参照，完整算子有额外处理与同步开销。
D按fan-out下行字节、C按归约上行字节除以完整算子时间，不相加上下行伪造带宽。

[最新分区测量、链路定标来源与验证范围](nb-borrow/source_partitions_20260910/README.md)。
其他日期报告为历史对照，不作为当前分区版本的全部资格证明。
