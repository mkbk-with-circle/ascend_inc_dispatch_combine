# 当前Single-INC：源rank独立分区

开发分支 `codex/source-partitioned-protocol`，本轮优化前回退点 `feca99b`。
D布局与通知按origin隔离；C直接消费真实D Journal，并行校验及段间检查通过后拉取所需贡献。

[协议与内存](../../../examples/inc/dispatch_combine/single_inc/pull_combine/SOURCE_PARTITIONS.md) · [抽象流程](../../../examples/inc/dispatch_combine/single_inc/pull_combine/FLOW.md) · [API状态](../API_COMPLETION_STATUS.md) · [最新正式结果及CSV](nb-borrow/random_pipeline_20260910/CURRENT_RESULTS.md)

平面A随机expert路由，H8192、64 experts、每源128 MiB BF16 hidden，每配置3种子/30测量样本：

| 算子 | W2/K2平均 GB/s | W4/K2平均 GB/s | W4/K4平均 GB/s |
|---|---:|---:|---:|
| Dispatch | 34.227 | 63.285 | 66.628 |
| Combine | 38.607 | 71.305 | 72.286 |

19个case全部PASS。C按实际FP32 partial字节计量，不固定为128 MiB/worker。
空源、尾部、K8、延迟READY及错误Journal拒绝/恢复已有验证。
全workload性能目标与无回退尚未全部满足；平面B规则W4/K4有明显带宽波动。

D默认每源1 producer+1 publisher，其余搬运。C两输入/两输出各16 KiB，另有本地校验摘要。
旧24 KiB是软件预算，本机后端UB上限192 KiB。Frontend仍需绑定分区入口，不表示框架接入已完成。

历史多打一put-only均值W2=42.734、W4=85.478 GB/s仅作链路实测参照。
D只计fan-out下行、C只计归约上行，均除以完整方向调用时间。
