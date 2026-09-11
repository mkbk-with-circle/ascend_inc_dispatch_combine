# 源分区Dispatch / Combine验收计划

这是待执行矩阵与判定原则；不是全部测试已经完成的声明。
当前已完成结果见[源分区报告](../../../../../../docs/inc/report/nb-borrow/ready_push_20260911/README.md)。

## 性能口径

- D有效带宽 = 实际fan-out下行hidden字节 / 完整Dispatch时间。
- C有效带宽 = 实际参与归约的上行partial字节 / 完整Combine时间。
- metadata/control只计入时间，不计入有效字节；padding、预留空洞和D的GET字节不加入D分子。
- 初始化、预分配、输入准备、专家计算/替身和数值验证放在计时之外。
- 独立D/C测试记录所有origin的完整提交到同步返回时间；链路串联模式报告总延迟，
  不将D+C字节合计后标为链路带宽。
- 正式性能至少3次预热、10次测量，保留全部样本的min/mean/CV和实际字节数。
- 按用户最新要求撤去旧raw百分比gate。已有多打一峰值定标均值为W2=42.734、
  W4=85.478 GB/s，仅作链路实测参照，不自动变成新的百分比阈值。
- 性能无回退与同卡组旧算子比较，必须对齐shape、dtype、实际路由分布和计时范围。
  样本不足或波动较大时应重复A/B，不裁剪低值或用减少数据量宣称提高。

## 分区与时序不变量

1. 目标数据、metadata、partial、Journal和完成邮箱按origin隔离，位置只由固定容量决定。
2. D每个源/wave一次READY；C每个B/origin/wave一次Notice。
3. 自然rank顺序提交且低rank延迟时，其他非空源仍能完成D分区。
   正序、逆序及随机到达都必须验证，不能通过把已知慢rank放到提交末尾掩盖阻塞。
4. 同源内部布局与数据依赖保留；C只消费自己的真实D Journal，并等待实际贡献B。
5. 完成状态、cookie、generation和ring lease按origin管理；失败源不撤销已完成的其他源。
6. 每算子AIV预算来自live核心数的一半，origin组之和不超过预算。
7. 运行前检查整个目标HCCS平面空闲；输出目录独立，只清理自身进程。

## 待覆盖输入矩阵

| 维度 | 计划覆盖 |
|---|---|
| 卡数 | 同平面W2/W4；其他规模单独验证，不能外推 |
| token数量 | 0、1、31/32/33、255/256/257及配置容量边界 |
| hidden尾部 | 行字节跨64B边界、tile边界；C元素跨1536边界 |
| 大小 | 小消息至128 MiB，再按实时HBM/固定分区公式逐级扩大 |
| 路由 | 固定GPU集合、全expert随机、同GPU多expert、热点、不同top-k |
| 数量偏斜 | 空源、仅一源非空、各源不同比例，每轮轮换实际数量 |
| READY时序 | 延迟低rank/高rank、多源延迟，D与C使用独立到达顺序 |
| 复用 | 多轮、多ring slot；只复用已释放的分区 |

Host-only大数算术测试要验证溢出拒绝，不能据此声称大规模NPU运行通过。
超出实际HBM资源应明确报告资源不足，不截断输入或悄悄换成多wave总量。

## 故障与恢复

计划注入身份、cookie、publication、shape、索引/贡献范围、source_offset跨origin、
重复贡献、缺失READY、容量不足和提前slot复用。
必须验证失败分区无成功Completion，其他独立分区仍完成；
同一会话中后继合法generation可恢复。异常退出、timeout、guard损坏或数值错误均不得算通过。

## 当前可复用入口

- tests/pull_v2_source_partitions.py：early、smoke、performance。
- inc_dc_source_partition_device_e2e.cpp：真实D Journal驱动C，支持独立计时和链路检查。
- tests/test_inc_dc_source_partition_layout.cpp：容量/偏移/对齐/不相交的Host属性检查。

旧紧凑协议的Host/API、压力runner可以辅助回归，但测试数量不自动转记为新协议证明。
