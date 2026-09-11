# Combine就绪描述随Notice发布（2026-09-11）

已移除INC在Notice之后发起的远端GET READY。dst先将128B就绪描述PUT到INC的原有对称区域，
quiet确认完成后发布原64B Notice。INC看到匹配Notice后失效本地描述缓存、原样校验所有字段，随后按路由记录GET partial。

这是两段有序PUT组成的一次就绪发布，**不是**一条192B原子报文。保留原结构和LaunchArgs布局，
`ready_staging`仅保留兼容空间。无需修改调用参数，所有rank使用同一版本库即可。
partial的GET、归约、结果PUT、AIV分配和buffer分配均未修改。

## 发布与生命周期

1. dst计算结束，partial及描述准备完成。
2. 将128B描述写到INC；quiet建立描述先于Notice可见的顺序。
3. 发布64B Notice；INC按原批次身份、cookie、偏移、行数、dtype和publication校验。
4. INC直接读取本地描述，按原数据循环拉取partial、归约并回传。
5. ACK前不覆盖partial和对应描述。按原src/ring隔离及复用规则运行。

## 同卡组带宽A/B

nb-borrow/910B2C，H8192、64 experts、每src128 MiB BF16 hidden，random_expert seed17。
每case 3 warmup+10 measure；C带宽为实际FP32 partial ingress字节/完整C调用时间，
真实Dispatch和partial准备在C计时之前。两版驱动和Dispatch动态库的SHA256完全一致。

| 配置 | 平面 | 改动前平均 GB/s | 改动后平均 GB/s | 平均变化 | 最低变化 |
|---|---|---:|---:|---:|---:|
| W2/K2 | A，workers0/1、INC2 | 38.5954 | 38.5924 | -0.008% | +0.137% |
| W4/K2 | B，workers8–11、INC12 | 71.0847 | 71.1104 | +0.036% | +0.292% |
| W4/K4 | B，workers8–11、INC12 | 72.3397 | 72.2701 | -0.096% | -0.166% |

均通过“不下降超过1%”的平均及最低带宽检查，未观察到明显带宽回退。
本改动省掉控制读取，未声称大数据量吞吐显著提高。

## PPT使用的当前平面A结果

为避免混用平面，另在平面A重测新Combine W4；每case 3 warmup+10 measure：

| 配置 | 最低 GB/s | 平均 GB/s | CV |
|---|---:|---:|---:|
| W2/K2 | 38.5014 | 38.5924 | 0.104873% |
| W4/K2 | 70.9719 | 71.2381 | 0.215227% |
| W4/K4 | 72.0904 | 72.2977 | 0.191833% |

PPT的Dispatch沿用未修改的数据面30样本矩阵，Combine使用本轮10样本。样本数量分别标注。

## 正确性和证据

12个设备case、116轮含预热检查全部通过：包括W2/W4空输入、非对齐H33、空源、
src0 READY延迟100ms、ring复用，以及完整D→C输出/metadata/guard检查。已有READY协议单元测试通过。
未重跑所有历史压力矩阵，不将这些检查视为任意故障的穷尽证明。

[W2平面A](w2_planeA.csv) · [W4平面B A/B](w4_planeB.csv) · [W4平面A新版本](w4_planeA.csv) · [构建指纹](builds.json)

回退提交：`cfeb30f`；基准运行库：`/tmp/inc-inline-ready-baseline-20260911`。
新Combine SHA256：`7c8b0db2a642e010f187c27df5e989641629157a08c24cd6d6acc7f9e9d2c683`。
原始日志：`/tmp/inc-push-ready-w2-20260911`、`/tmp/inc-push-ready-w4-20260911`、`/tmp/inc-push-ready-w4-planeA-20260911`。

复跑入口：`tests/pull_v2_push_ready_check.py`，传入baseline/candidate构建目录、全新输出目录、workers和first-npu。
