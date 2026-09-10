# 按源 rank 独立分区的 Dispatch / Combine

对应开发分支`codex/source-partitioned-protocol`。回退点为`archive/pre-source-partitions-a230a1b`。
源分区D→C链路已通过基本真机检查；最近完成的带宽仍是1次预热、2次测量的短测，
尚未完成无回退验收。当前按用户要求暂停优化，先同步讲解和文档。

## 固定容量决定位置，实际数量决定使用范围

s是原始源rank（origin），B是目标专家Worker，R是每源预留行容量。
R按配置最大token数向32个token取整，assignment容量向4项取整；
分区边界64B对齐，容量和偏移计算检查整数溢出。

```text
每个目标 B 的 hidden：  [ring][s][R][H]
每个目标 B 的路由：     [ring][s][DestinationRow / ExpertAssignment]
每个 B 的 FP32 partial：[ring][s][R][H]
每个 A_s 的输出：       [ring][R][H]
独立控制邮箱：          [s][ring][B]
INC工作区 / Journal：   [s][ring][私有数据]

hidden地址 = B.hidden_base + ring * hidden_ring_stride
           + s * hidden_partition_stride + r * H * dtype_bytes
```

r只在s分区中有意义，不能直接当作B整个缓冲区的全局行号。分区基址不依赖其他源
的实际token数；未使用的预留空间不计入带宽。对称heap根据所有预留arena及开销计算，
不再固定为2 GiB。

## Dispatch

1. A_s准备本wave全部hidden、token ID和路由metadata，固定Source Slot。
2. 每个A_s/wave发布一次64B READY，INC按注册区和ring定位。
3. INC仅在本源执行组内拉取、校验metadata并计算局部前缀，不等其他origin的READY或行数。
4. INC生成本源row map和Journal，并重新验证Worker提供的目标集合一致提示。
5. 本源目标集合一致时连续分tile；变化时逐token定位。对齐主路径分别采用双缓冲和三缓冲。
   每个需要分发的hidden字节只从源拉一份，再向唯一目标GPU各发送一份。
6. 本源数据和metadata全部可见且校验成功后，封存本源Journal，发布B_s的Completion，
   最后向A_s发布Source ACK。这里只汇合本origin的执行组。

同GPU多个expert共享一行网络hidden，所有assignment和weight保留。
非对齐行保留精确宽度搬运，同一源组内仍有必要同步。
当前通过每origin独立kernel执行组和stream隔离；本机D总预算24 AIV，W4每组6、W2每组12，
由live核心数计算。当前并非长期驻留的交换机服务。
资格程序仍为初始化、计时对齐和轮次复用使用整组Host barrier。延迟READY测试在输入准备
和barrier之后注入延迟，证明分区/kernel内部无跨源布局等待；业务调用层还需按分区驱动消费。

## Combine

B完成某个源分区s的专家计算与本地加权归约后，准备本分区partial和128B READY，
再向[s][ring][B]邮箱发布一次64B Notice，无需等待B的其他源分区一起算完。

INC首先等待本源的真实Dispatch Journal封存并校验。
新设备入口直接读取JournalTokenEntry和JournalContributor，
不依赖Host预生成PullPlan、heads、pull_next或results。
dispatch_cookie=0可从身份匹配的sealed Journal取得cookie，之后READY必须匹配它。

仅轮询Journal实际需要的B。Notice校验后GET 128B READY，再检查cookie、行数、dtype、
分区偏移、容量和publication。partial在归约任务执行到相应token/tile时按需GET：

```text
partial tile地址 = B.partials_base + READY.source_offset
                 + contributor.destination_row * H * 4 + tile_byte_offset

READY.source_offset = ring * registered_slot_stride
                    + s * partial_partition_stride
```

同token的各B贡献逐tile累加，收齐本tile即PUT到原始A_s的对应行。
输入ping/pong和交替输出共4×6 KiB/AIV；奇数尾贡献单独处理，零贡献输出零。
本组完成后更新本源Journal，发布B分区ACK和A_s的Owner Completion；
两类通知可以交错到达，不保证所有B的ACK先于Owner Completion。

真实贡献未就绪仍须等待。移除的是跨origin布局/提交依赖，归约的数据依赖仍然存在。

```text
FREE → DISPATCH_OPEN → DISPATCH_SEALED → COMBINE_ACTIVE → COMPLETE
失败分区 → ABORTED
```

D Source ACK只释放A的输入slot。目标分区、partial和Journal继续按对应消费/完成状态复用；
不能因另一个origin完成或timeout就覆盖这些区域。

## 当前入口与资格状态

- 容量与偏移：inc_dc_source_partition_layout.{h,cpp}。
- D入口：launch_inc_dc_partitioned_dispatch，参数见inc_dc_partitioned_dispatch.h。
- C入口：launch_inc_dc_partitioned_combine，参数见inc_dc_partitioned_combine.h。
- 真机完整示例：inc_dc_source_partition_device_e2e.cpp，包含初始化、真实D、确定性专家计算替身、
  分区Notice、真实Journal驱动的C，以及输出/metadata/guard检查。
- 原SingleIncSession/BackendOps前端仍需适配新分区布局与launcher；原CPU参考示例尚未绑定
  新NPU入口。旧紧凑布局入口保留用于回归对照。

已完成W2/W4基本链路、W4空源/H=33尾部/3轮ring复用。
自然rank顺序提交、rank 0 READY延迟100ms时，其余非空源的D Completion先于rank 0 READY。
最近W4/K2短测：D平均73.921 GB/s、C平均71.4827 GB/s；C低于旧均值77.303 GB/s。
Combine并行Journal校验的接口/工作区改动尚待设备数据面及新一轮验证。
大规模、全故障矩阵与性能无回退仍待完成。

当前讲解撤去旧raw百分比gate，改列已有“多打一”链路峰值参照：
W2约42.7、W4约85.5 GB/s，来自历史put-only峰值定标的均值。
这是实测参照，不是严格物理理论上限；本轮未重跑定标，也未另设百分比门槛。
算子仍按D下行/C上行有效字节除以完整算子时间，与同卡组旧算子比较回退。

详细数据与来源见[分区报告](../../../../../docs/inc/report/nb-borrow/source_partitions_20260910/README.md)。
