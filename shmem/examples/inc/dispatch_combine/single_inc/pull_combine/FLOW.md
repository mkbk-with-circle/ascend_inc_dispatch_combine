# 源rank独立分区：抽象流程

s是原始源rank；B_s是专家Worker B为源s预留的分区。位置由容量决定，
实际数量只影响本分区使用范围。以下每个流程都对应一个origin。

## Dispatch

```text
A_s准备本wave的hidden与metadata
  ↓ 固定Source Slot
发布一次64B READY ─────────────────→ INC轮询本源READY
                                      ↓
本源Header/metadata被GET ←────────── GET并校验身份、容量、路由
                                      ↓
                                   计算本源局部行号r，建立本源Journal
                                      ↓
本源hidden tile被GET ←────────────── GET一份hidden，循环缓冲
                                      ↓
B.hidden[ring][s][r] ←─────────────── PUT到各唯一目标B的固定s分区
B_s的路由metadata   ←─────────────── 保留全部expert assignment
                                      ↓ 本执行组校验与传输完成
                                   本源Journal：DISPATCH_SEALED
B_s可以消费         ←────────────── 本分区Destination Completion
A_s源slot可复用     ←────────────── Source ACK
```

同源目标集合一致可连续搬运，变化时逐token定位；对齐主路径分别采用双缓冲/三缓冲。
INC验证Worker提示。其他源可以尚未READY，先到源仍可完成自己的PUT与通知。

## Combine

```text
B消费D已完成的源s分区                 INC等待本源真实Journal封存
  ↓ 专家计算、本地加权归约              ↓ 分段并行校验token及贡献，合并检查段间边界
准备partial[ring][s][r]和128B READY
  ↓
发布64B Notice到[s][ring][B] ───────→ 轮询本Journal实际需要的B
                                      ↓ GET本B_s的128B READY
                                   校验cookie、行数、dtype、分区地址
                                      ↓ 标记该B_s可读
B_s partial tile被GET ←──────────── 按token贡献记录，等所需B_s并GET
                                      ↓ 两份一批FP32累加
A_s原始token输出行  ←────────────── PUT已收齐贡献的结果tile
                                      ↓ 本执行组quiet / 汇合
                                   本源Journal：COMPLETE或ABORTED
B_s partial可复用   ←────────────── 本分区Source ACK
A_s结果可消费       ←────────────── Owner Completion
```

Notice先触发GET READY，partial按归约任务拉取。真实贡献未到齐仍须等待。
Journal校验仅在本origin组内并行，全部检查通过后才接受所需Notice；摘要留在INC本地。
每AIV为两输入/两输出各16 KiB，同token下一tile的GET可提前；复用和错误退出均排空事件。
ACK与Owner Completion可交错到达。分区C直接消费设备D的Journal，Host不构造归约索引。

不同origin或不同可用ring slot可使用独立stream；同一origin的C依赖其D Journal封存。
ring slot按本分区生命周期复用。验证范围见[SOURCE_PARTITIONS.md](SOURCE_PARTITIONS.md)。
