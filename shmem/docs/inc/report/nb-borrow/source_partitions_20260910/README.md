# 源rank独立分区：协议与最近已完成测量

本页对应开发工作树`codex/source-partitioned-protocol`，回退点
`archive/pre-source-partitions-a230a1b`。当前按用户要求暂停内核优化，先更新PPT与文档。
Combine并行Journal校验仍是待完成接入/设备复测的开发改动；下表不是该优化的新测结果。

## 当前协议

各B为每个origin预留固定区域：`[ring][origin][R][H]`。
INC分别计算各源内部行号，独立发送、封存Journal并发布分区Completion/ACK。
未READY的其他源不再决定这个源的数据落点。

新C直接消费真实D生成的JournalTokenEntry/JournalContributor，按本源需要的B分区GET、
FP32归约并PUT原始owner。Host只规划固定容量并准备本地数据，不生成归约Pull Index。
前端SingleIncSession/BackendOps仍需绑定新分区launcher，通用框架API尚未自动完成适配。

资格程序完成真实D→确定性专家计算替身→C；此结果不是正式FFN或vLLM推理测试。
整体测试轮次收尾仍同步等待全部源，B可以根据独立分区Completion提前消费。

## 环境和多打一链路峰值参照

nb-borrow：16×910B2C，64 GiB HBM/卡，CANN 9.1.0-beta.3。
NPU 0–7和8–15各为8卡HCCS平面，跨平面为PIX/PHB/SYS。
本次对照仅用平面A：W2为worker0/1→INC2；W4为worker0–3→INC4。
INC各算子总预算为live AIV的一半，本机24；W4每origin6 AIV，W2每origin12 AIV。

按用户最新要求，当前PPT与当前验收说明撤去旧raw百分比gate。
改列已有“多个Worker同时向一个INC发送”的put-only峰值定标：

| 规模 / 方向 | 峰值测试均值 GB/s | 最低 GB/s | CV |
|---|---:|---:|---:|
| W2→INC | 42.734 | 42.714 | 0.0221% |
| W4→INC | 85.478 | 85.445 | 0.0244% |

来源：[nb硬件环境记录](../../../hardware_profiles/910b2c-nb/single_inc_ENV_STATUS.md)。
“约42.7/85.5 GB/s”表示本机已有实测峰值参照；这里列的是该定标的均值和最低值，
不是新测单样本最大值或严格物理理论上限。本轮没有重跑链路测试。
该汇总没有完整保留原put-only定标的payload/轮数，不应把下方算子的128 MiB/3+10参数套给它。

另有历史W4 INC主动pull同向测试：128 MiB/worker、3 lanes/worker，mean 83.130 GB/s；
其计时仍需单独审计，不能和put-only或完整算子混算。
完整算子还包含metadata解析、归约、控制通知和同步；峰值表用于识别链路能力，不另设新的百分比门槛。
该表为多打一上行方向。Dispatch计下行，若计算严格的方向利用率，需使用单独的INC→all定标。

## W4 / K2：最近已完成的算子对照

K2指每token的两个唯一GPU贡献。所有带宽采用十进制GB/s。

| 算子 | 旧均值 GB/s | 分区最低 GB/s | 分区均值 GB/s | 分区 CV | 均值变化 |
|---|---:|---:|---:|---:|---:|
| Dispatch | 69.596 | 73.872 | 73.921 | 0.067% | +6.21% |
| Combine | 77.303 | 71.376 | 71.483 | 0.150% | −7.53% |

- 旧版：3次预热+10次测量。新分区：1次预热+2次测量，仅为短测。
- D：H=8192，BF16，每源8192 token，即128 MiB hidden；有效fan-out总量1 GiB。
- C：H=8192，FP32，每B实际128 MiB partial，有效ingress总量512 MiB。
  为准备这组C，先真实运行每origin2048 token的D，再生成partial；这些准备不计入C时间。
- D分子只计fan-out下行hidden；C只计参与归约的上行partial；分母为完整算子所有origin
  的主机提交到同步返回。数据准备、专家替身和正确性检查在计时之外。
- D与C的计时样本是独立模式；不能把D+C字节相加当作链路带宽。

D短测提高，C仍回退。当前没有通过“所有case性能不下降”的完整验收。
旧协议的其他K、其他shape与压力测试数量不能转记为新分区协议结果。

## 已完成的正确性与独立性检查

无延迟smoke：W2的32/32 token、H=64；W4的31/0/7/19 token、H=33，各3轮。
延迟READY检查：W2 H=64、W4 H=64、W4 H=33，每组3轮；首轮rank0延迟100ms。
测试按自然rank顺序提交，并断言所有其他非空origin的D分区Completion早于rank0的READY。
以上真实D→C输出、路由metadata、guard及ring复用检查均通过。
延迟在输入准备及测试barrier之后注入；这是kernel/分区布局独立性的证据，
不表示当前测试程序已经取消每轮所有Host barrier。

这些有限样本证明了已测场景的分区独立性；大规模、全面故障、更多随机种子和正式性能矩阵仍待完成。

## 证据与入口

- 旧D/C对照：`/tmp/inc-source-partition-baseline-20260910/{results,combine_results}`。
- 分区短测：`/tmp/inc-source-partition-perf-probe-20260910/{dispatch,combine}`。
- 无延迟链路：`/tmp/inc-source-partition-smoke1-20260910`。
- 延迟READY：`/tmp/inc-source-partition-early-20260910`。
- [协议/布局](../../../../../examples/inc/dispatch_combine/single_inc/pull_combine/SOURCE_PARTITIONS.md)
  与[运行入口](../../../../../examples/inc/dispatch_combine/single_inc/pull_combine/README.md)。

原始日志留在实验机独立目录，未覆盖历史记录。PPT只引用此处已经完成的测量。
