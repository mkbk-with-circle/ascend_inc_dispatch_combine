# Dispatch当前设计入口：源rank独立分区

当前开发采用源rank固定预留分区，详细约定见[SOURCE_PARTITIONS.md](SOURCE_PARTITIONS.md)，
图示见[FLOW.md](FLOW.md)。本页保留旧文件名，供已有链接继续使用。

## 地址与独立性

目标B的接收区域为[ring][origin][R][H]，metadata和计数同样按origin隔离。
R来自初始化容量约定；其他rank的实际数量不参与本源分区基址计算。
每个origin独立拉取、校验、局部布局、fan-out、封存Journal和发布Completion/ACK。

worker准备本wave整个Source Slot后发布一次READY；INC仍需要处理本源metadata，
但不等待无关origin的READY或计数。分区内部的局部行r不等于整个B的全局行号。

## 搬运与Combine衔接

INC验证当前源的uniform提示；满足时连续分tile，否则逐token查询本源row map。
双缓冲/三缓冲允许GET下一块与PUT当前块交叠；非对齐尾部保留精确宽度处理。

每个本源Journal记录原始owner/token与各B局部行的关系。
新Combine直接读取真实设备Journal；B按源分区准备FP32 partial并发布Notice。
不再通过Host预生成Pull Index实现这条新链路。

## 性能与测试

D只计fan-out下行有效字节，C只计参与归约的上行有效字节，均除以完整算子时间。
按用户最新要求移除旧raw百分比gate，当前采用已有多打一实测峰值参照：
W2约42.7、W4约85.5 GB/s。峰值参照不替代完整算子同卡组无回退比较。

最新已完成测量和待完成项见[源分区报告](../../../../../docs/inc/report/nb-borrow/source_partitions_20260910/README.md)。
之前按全局紧凑前缀发号的实现/设计可由archive/pre-source-partitions-a230a1b恢复。
