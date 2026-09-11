# Single-INC：源rank独立分区 Dispatch / Combine

当前开发路径按原始源rank预留独立目标分区。先READY的源可以独立计算局部行号、
发送并发布自己的Completion，不再等其他源报告实际数量。
新Combine直接消费该源的真实Dispatch Journal；基本真机D→C链路已接通。

当前分支：`codex/source-partitioned-protocol`。
旧紧凑布局入口保留为回归对照；回退点`archive/pre-source-partitions-a230a1b`。
历史V1/Fusion可从更早归档标签恢复。

## 先读这些

- [分区协议与生命周期](SOURCE_PARTITIONS.md)
- [Dispatch / Combine抽象流程](FLOW.md)
- [就绪描述随Notice发布与带宽回归](../../../../../docs/inc/report/nb-borrow/ready_push_20260911/README.md)
- [前轮数据面正式矩阵](../../../../../docs/inc/report/nb-borrow/random_pipeline_20260910/CURRENT_RESULTS.md)
- [API接通状态](../../../../../docs/inc/API_COMPLETION_STATUS.md)

## 当前设备入口

| 文件 | 用途 |
|---|---|
| [inc_dc_source_partition_layout.h](inc_dc_source_partition_layout.h) | 固定容量、分区stride、控制邮箱和INC工作区偏移 |
| [inc_dc_partitioned_dispatch.h](inc_dc_partitioned_dispatch.h) | 每origin的Dispatch启动参数 |
| [inc_dc_pull_dispatch_v2_device_kernel.cpp](inc_dc_pull_dispatch_v2_device_kernel.cpp) | 新分区特化及旧紧凑对照，共用搬运实现 |
| [inc_dc_partitioned_combine.h](inc_dc_partitioned_combine.h) | 每origin的Combine启动参数 |
| [inc_dc_partitioned_combine.cpp](inc_dc_partitioned_combine.cpp) | 直接读取设备Journal的FP32 GET/归约/PUT |
| [inc_dc_source_partition_device_e2e.cpp](inc_dc_source_partition_device_e2e.cpp) | 初始化、真实D→计算替身→C、输出验证和释放 |
| [tests/pull_v2_source_partitions.py](tests/pull_v2_source_partitions.py) | 独立性、smoke、完整算子性能测试 |

传入的目标数据基址按origin偏移，ring stride覆盖全部origin。
C控制邮箱指针先定位到[origin][ring]，kernel按B索引；准确指针约定见相应LaunchArgs注释。
调用方必须通过布局工具计算容量和偏移，并按分区Completion/ACK保持buffer生命周期。

## 协议与内存

```text
D：A_s准备本wave所有token → 一次READY → INC校验本源metadata/局部布局
   → GET一份hidden → PUT到各B的[ring][s]分区 → 本源Completion/ACK

B：消费已完成的s分区 → 专家计算、本地加权归约
   → 写FP32 partial → PUT128B描述到INC → quiet → 发布64B Notice到[s][ring][B]

C：等待真实Dispatch路由记录封存 → 校验 → 轮询Notice并读取INC本地就绪描述
   → 按token/tile GET partial、求和、PUT A_s → 本分区ACK/Owner Completion
```

目标与partial为[ring][origin][R][H]。R由最大容量决定，局部行号r不依赖其他origin数量。
D每个源/wave一次READY；C每个B/源分区/wave一次Notice，不再是所有分区一起准备才通知。
没有该源贡献的B不会被拉取；真实归约贡献未到齐仍须等待。
128B描述和64B Notice按顺序一起发布，省去GET READY往返；原结构、接口和数据面算法保持不变。
发送端和INC须使用同一版本；历史带宽数据的版本说明及新控制路径回归见报告。

本机每算子预算为live AIV数的一半，按origin执行组分配。规则/变化路由的对齐D路径
分别采用双缓冲和三缓冲；C为两输入、两输出各16 KiB，共64 KiB，低于本机后端192 KiB UB上限。
D默认每源留1个producer和1个publisher，其余搬运；C分段并行校验Journal，合并检查边界后再GET。
C的本地校验摘要区由planner分配，须传入validation_scratch及容量。非对齐尾部及身份、容量、cookie检查保留。

## 构建与运行

在仓库根目录、已配置的CANN/SHMEM构建环境中：

```bash
cmake --build "$INC_BUILD" -j8 --target \
  inc_dc_source_partition_layout_tests \
  inc_dc_source_partition_device_e2e

"$INC_BUILD/bin/inc_dc_source_partition_layout_tests"

python3 shmem/examples/inc/dispatch_combine/single_inc/pull_combine/tests/pull_v2_source_partitions.py \
  --build-dir "$INC_BUILD" --output /tmp/source-partition-new-early \
  --first-npu 0 --suite early
```

也可选择smoke/performance/random-stress/combine-regression套件。输出目录须不存在，runner检查整张目标HCCS平面空闲。
并行校验已有设备验证，包括错误Journal拒绝及合法输入恢复；完整性能目标仍未全部满足。

## 框架前端适配状态

`inc_dc_pull_v2_api.{h,cpp}`中的SingleIncSession/BackendOps、BatchHandle和Completion
前端仍需绑定新的分区launcher及分区接收视图；它不会自动切换到上述设备路径。
`inc_dc_pull_v2_api_example.cpp`仍是CPU参考示例。
已经接通的是新的NPU资格程序中“真实D Journal→设备C”，不是通用Frontend后端或vLLM接入。

## 结果怎么读

已有多打一put-only峰值定标：W2平均42.734、W4平均85.478 GB/s，作为本机链路能力参照。
当前讲解已撤去旧raw百分比gate；不把峰值测试均值标成严格物理理论上限。

完整算子D只计fan-out下行字节，C只计参与归约的上行字节，均除以完整调用时间。
未修改的D数据面保留30样本矩阵；新通知流程C另有同卡组A/B及平面A每配置10次复测。前轮规则W4/K4有明显波动，
不宣称全workload性能目标满足。历史短测与旧版本压力测试数量不转作当前资格证明。
