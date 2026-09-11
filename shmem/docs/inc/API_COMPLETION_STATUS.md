# Single-INC API与设备链路状态（2026-09-10）

当前开发分支为codex/source-partitioned-protocol，采用源rank独立分区。
详细协议见[源分区说明](../../examples/inc/dispatch_combine/single_inc/pull_combine/SOURCE_PARTITIONS.md)。

## 已接通的设备链路

- D按origin使用固定目标分区，独立解析、发送、封存Journal并发布完成通知。
- B按分区消费D结果并生成FP32 partial，发布本B/分区的Notice和READY。
- 新C直接读取真实设备D的JournalTokenEntry/JournalContributor，不再用Host构造Pull Index。
- 并行路由记录校验已通过设备及故障拒绝/恢复验证。
- NPU资格程序已通过W2/W4基本D→C、空源、非对齐尾部和3轮ring复用检查。
- rank 0 READY延迟100ms时，其他非空origin已先完成D分区。
- 测试的专家计算是确定性替身，用于验证通信链路，不代表正式FFN或推理加速。

设备入口为launch_inc_dc_partitioned_dispatch与launch_inc_dc_partitioned_combine；
容量及地址来自SourcePartitionLayout。当前以每origin的kernel执行组/stream运行，
尚非长期驻留交换机服务。

## 原前端已有能力

原SingleIncSession/BackendOps前端已有Batch lease、实例/epoch身份、异步Completion和
slot复用检查；CPU参考示例验证去重、assignment、weight与归约语义。
这些前端组件保留，但仍需适配新的origin分区视图、控制邮箱和设备launcher。

## 尚需完成

1. 通用Frontend随库NPU后端及框架device-pack适配；不应把CPU示例当作新设备入口。
2. 分区版完整性能无回退与更大随机/故障/交叠矩阵。
3. 任意token-ID输出顺序的框架适配，以及FP32以外partial精度的设备资格。
4. 真机验证当前仅W2/W4；其他规模和集群需重新验证资源预算与性能。

最新正式测量、版本差异及本机链路参照见[当前结果](report/nb-borrow/random_pipeline_20260910/CURRENT_RESULTS.md)。

2026-09-11：分区Combine改为dst先PUT128B描述到INC，再发布64B Notice；INC读取本地描述。
LaunchArgs布局与调用参数不变，ready_staging保留兼容；同版库须用于所有rank。见[回归报告](report/nb-borrow/ready_push_20260911/README.md)。
当前PPT保留8页详细流程，已去掉基础原理总览和独立“完成条件与当前边界”页。
