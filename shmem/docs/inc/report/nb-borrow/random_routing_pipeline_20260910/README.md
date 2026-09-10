# 随机路由流水候选：尚未全部达标

> 本页保留源分区改动前的试验数据。当前讲解已撤去旧raw百分比gate；
> 新协议及最近完成测量见[源分区报告](../source_partitions_20260910/README.md)。

本报告不宣布“完全随机路由已达到固定路由带宽”。保留内核对应
`0ccba16`（与 `5278688` 相同）；后续补充卡号、种子、实际字节数等日志字段。
试验分支为 `codex/random-routing-pipeline`，未推送替换远程发布版本。

## 环境与测量口径

- nb-borrow，16×910B2C，64 GiB HBM/NPU，CANN 9.1.0-beta.3。
- 2026-09-10复查：0–7、8–15分别为8卡HCCS平面；跨平面为PIX/PHB/SYS。
- 平面A：W2=worker 0/1、INC 2；W4=worker 0–3、INC 4。
  平面B对应worker 8/9、INC 10，或worker 8–11、INC 12。
- INC 4/12均报告7条224 Gbit/s HCCS、4 lane，health OK，error/retry=0；
  核频率查询均1800 MHz。W2/W4的指定raw参照仍为56/112 GB/s，未降低gate。
  对应92% raw gate仍为51.52/103.04 GB/s，本轮未达到；图中的应用带宽参照另行比较。
- D：每个worker实际128 MiB BF16 hidden，H=8192。带宽为**fan-out hidden字节 / 完整调用时间**。
- C：FP32 partial、H=8192，带宽为**用于归约的ingress字节 / 完整调用时间**。
  固定两贡献测试的总ingress为W×128 MiB，随机分配后各worker不保证恰好128 MiB。
  `mixed_k`本次总ingress为540,082,176字节，不能标成各worker精确128 MiB。
- 时间包括提交、通信、处理和同步，排除输入准备与数值验证。D/C独立测量；
  **不是实际Dispatch→专家计算→Combine串联，也不是vLLM端到端数据**。
- INC仍各使用半数AIV。仅验证W2/W4；不能将此写成任意规模真机已验证。

## “随机”的两种含义

`random_k*_gpu*`固定唯一目标GPU数，再随机选择GPU及其本地expert。
`random_expert_k2/k4/k8`从全部64个expert中无放回随机选择top-k，随后按GPU去重，
每个token的唯一目标GPU数可以变化。两者都不提前向INC上传token plan。

### 全expert随机：平面A、两个种子

每组1次预热+3次测量，种子17、104729。以下为两个种子的均值范围，
是短测/正确性压力结果，**不是3+10正式达标认证**。

| 规模 | expert top-k | 最低GB/s（两种子） | 各种子平均GB/s | CV范围 |
|---|---:|---:|---:|---:|
| W2 | 2 | 32.991 | 33.013–33.081 | 0.022%–0.229% |
| W2 | 4 | 34.374 | 34.438–34.542 | 0.069%–0.134% |
| W4 | 2 | 60.472 | 60.621–60.682 | 0.154%–0.243% |
| W4 | 4 | 63.792 | 63.900–63.955 | 0.172%–0.179% |

全部样本数值/metadata/协议检查通过。不能用“W2固定两个GPU约38 GB/s”
代替此表中的全expert随机结果；也不能把K4动态GPU数与固定GPU2/GPU4混为同一负载。

### 固定两个目标GPU：平面A隔离对照

W4、3次预热+5次测量，两个平面之间串行测试，均使用完整D时间。

| 路由 | 最低GB/s | 平均GB/s | CV |
|---|---:|---:|---:|
| 原有`sym_k2_balanced`对照 | 69.284 | 69.752 | 0.495% |
| `random_k2_gpu2` | 61.453 | 61.659 | 0.217% |

固定路由对照复现了图中69.249/69.747 GB/s；随机路由均值仍低约11.6%，**未达标**。
其余旧case尚需同卡组全面回归，不能据这一项声称所有旧case都没有回退。

### Combine正式复测：平面B

3次预热+10次测量，种子17；归约内核未采用本轮无收益的地址计算试验。

| 规模/贡献模式 | 最低GB/s | 平均GB/s | CV |
|---|---:|---:|---:|
| W2、随机固定两贡献 | 38.942 | 39.024 | 0.166% |
| W4、随机固定两贡献 | 68.952 | 69.268 | 0.374% |
| W4、固定四贡献对照 | 78.495 | 78.885 | 0.238% |
| W4、随机变贡献数（含零） | 68.619 | 68.797 | 0.233% |

全部数值和状态检查通过。W4随机两贡献仍低于图中约77 GB/s参照；
W2接近约39 GB/s，但不把微小均值差异写成严格阈值已通过。

## 平面差异：未解决，不能混算

平面B的W4随机K2/GPU2在双平面测试期间均值53.792 GB/s；
随后确保另一平面空闲，复测仍为53.912 GB/s，固定路由也只有54.812 GB/s。
平面A同样隔离条件下分别为61.659、69.752 GB/s。

因此并发不是这一差异的唯一解释。板型、逻辑/物理ID映射、链路速率与健康检查
未发现直接解释；**不能仅凭HCCS标签断言两个平面的应用带宽等价**。
平面B四目标正式样本还出现5%–7%以上CV，相关结果完整保留在CSV中，不隐藏异常。

## 保留的修改与已拒绝试验

保留：INC内部生成row map、并行布局生产、共享缓存行边界记录单写入者修复、
按已发布区间读取描述符、跨token持续搬运及有界缓冲复用。
源READY、INC GET/解析/fan-out、Journal、Destination Completion与Source ACK语义不变。
Combine内核保持原有统一归约路径；修复测试程序同时提供fault和seed时忽略fault的问题。

以下试验未带来净收益，已撤回工作树，仅留Git历史：

- `f57e9de`：更深GET窗口、源GET批量化、相邻PUT合并；约59–61 GB/s。
- `7de6ba4`：INC本地暂存、目标定向发送、拉取核完成后加入发送；约49–59 GB/s。
- `e90ccdb`：INC内部预计算远端地址；约61–64 GB/s，无显著提升。

暂存试验的打点显示拉取约8.3 ms完成、发送约19.2 ms完成；核复用仍未获得净收益。
这不是协议修改，也不是最终启用路径。

## 性能剖析与验证

`msprof`在平面A采集的第二个样本如下（采集态，仅用于诊断，不参与gate）：

| 指标 | 固定K2 | 随机GPU2 |
|---|---:|---:|
| kernel时间 | 15.417 ms | 17.357 ms |
| AIV scalar time | 2.922 ms | 2.956 ms |
| AIV MTE2 time | 4.621 ms | 3.805 ms |
| AIV MTE3 time | 6.527 ms | 7.784 ms |

这些是profiler聚合指标，**不能相加成总时间或直接解释成链路利用率**。
它们指向随机写出/搬运流水仍需进一步分析，而非仅靠减少路由解析即可解决。
系统Python 3.9可正常导出；默认Python 3.10缺`_ctypes`，本轮仅在导出命令中局部
选择系统Python，没有改全局环境或安装包。

保留候选的D安全回归：23组、287个wave，全通过；覆盖空输入、非对齐尾部、
W2/W4各100次复用、5类错误注入、token/READY偏斜。
C安全回归：20组、262个wave，全通过，并使用seed=17验证fault参数仍生效。
三个Host协议/API测试通过，CMake开启`-UNDEBUG`；10个Python runner测试通过。
另一次W4、128 MiB、固定K2的同会话双流D/C smoke通过，各使用24 AIV；
这是独立两组输入的并发检查，不是同一批token串联或全随机交叠资格认证。
补充日志smoke验证了`first_npu`、seed以及C的逐源字节数之和，fault+seed联合输入亦通过。
有限测试不构成“任意输入、任意集群绝不出错”的证明，也不构成随机性能达标证明。

## 数据与复现

- [summary.csv](summary.csv)：21组汇总，包含卡组、seed、实测字节数与原始日志位置。
- [samples.csv](samples.csv)：181个预热/测量样本，完整保留低值，不筛选最快样本。
- 原始日志位于CSV中的独立`/tmp/random-*`目录；未覆盖历史结果，未把大量JSON/二进制加入仓库。

在仓库根目录，沿用现有构建环境：

```bash
cmake --build "$INC_BUILD" --target inc_dc_pull_dispatch_v2_device_e2e inc_dc_pull_combine_v2_npu_e2e -j8
python3 shmem/examples/inc/dispatch_combine/single_inc/pull_combine/tests/pull_v2_random_dispatch.py \
  --build-dir "$INC_BUILD" --output /tmp/inc-random-new-run \
  --first-npu 0 --workers 2 4 --seeds 17 104729 \
  --routes random_expert_k2 random_expert_k4 --warmup 1 --measure 3
```

输出目录必须不存在。正式复测应改用`--warmup 3 --measure 10`。
Runner逐组检查整张HCCS平面空闲、各rank退出状态、逐样本正确性与实际源数据量。
Combine新增负载名为`random_k2`；H=8192时W2的rows=4096，W4的rows=8192，
可复现总ingress=W×128 MiB；尾部参数为`fault=0, route_seed=17`。

仍需完成：随机写出瓶颈定位与优化、平面B差异定位、全旧case同卡组性能回归、
更多种子及数据量的正式资格矩阵。目前不要发布“完全随机路由已达标”的结论。
