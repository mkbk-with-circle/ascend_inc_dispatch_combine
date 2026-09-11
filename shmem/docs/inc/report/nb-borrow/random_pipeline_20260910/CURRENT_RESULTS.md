# 当前源分区实现：正式结果（2026-09-10）

本文为2026-09-10数据面记录。2026-09-11就绪描述随Notice发布及新版本带宽见[通知优化报告](../ready_push_20260911/README.md)。

源rank独立分区、INC pull、真实D Journal驱动Combine。以下不混合两平面数据，不宣称全workload性能无回退。

## 环境与口径

nb-borrow：16×910B2C、64 GiB HBM/卡、CANN 9.1.0-beta.3。
HCCS平面A=0–7、B=8–15；跨平面PIX/PHB/SYS不纳入测试。
平面A：W2 workers0/1+INC2，W4 workers0–3+INC4；平面B分别为8/9+10、8–11+12。
每卡48 AIV，D/C各半，按origin分配。CANN的910B2C配置及SHMEM后端UB上限为192 KiB。

历史多打一put-only定标均值42.734/85.478 GB/s仅作链路实测参照，不是严格物理上限。
D带宽=实际fan-out hidden字节/完整D时间；C=实际FP32 ingress字节/完整C时间。
输入准备、专家计算替身和验证在方向计时之外；C使用真实D Journal，但D不计入C时间。

## 平面A随机路由

H8192、64 experts、每源8192 tokens=128 MiB BF16 hidden。同GPU多expert去重，C输入不固定为128 MiB/worker。
每配置3种子、各3 warmup+10 measure；最低、平均、总体CV直接统计30个非预热样本。

| 算子 | W / expert K | 最低 GB/s | 平均 GB/s | CV |
|---|---|---:|---:|---:|
| Dispatch | 2 / 2 | 34.078 | 34.227 | 0.225% |
| Dispatch | 4 / 2 | 62.859 | 63.285 | 0.340% |
| Dispatch | 4 / 4 | 66.291 | 66.628 | 0.195% |
| Combine | 2 / 2 | 38.443 | 38.607 | 0.189% |
| Combine | 4 / 2 | 70.942 | 71.305 | 0.240% |
| Combine | 4 / 4 | 71.999 | 72.286 | 0.196% |

[每case汇总](current_random.csv)，原始日志 `/tmp/inc-tenpct-final-random-20260910`。
18个方向case加1个非对称真D→C case全部PASS，共244轮含预热检查。

## 平面B规则路由：不与平面A直接作A/B

每case 3 warmup+10 measure；D每worker128 MiB BF16 hidden；C每worker128 MiB FP32 partial，故C规模与上表不同。

| 算子 | W / expert K | 最低 GB/s | 平均 GB/s | CV |
|---|---|---:|---:|---:|
| Dispatch | 2 / 2 | 38.468 | 38.520 | 0.088% |
| Dispatch | 4 / 2 | 74.054 | 74.336 | 0.244% |
| Dispatch | 4 / 4 | 54.329 | 57.427 | 5.064% |
| Combine | 2 / 2 | 39.827 | 39.898 | 0.070% |
| Combine | 4 / 2 | 78.875 | 78.960 | 0.076% |
| Combine | 4 / 4 | 66.325 | 67.394 | 0.851% |

[规则回归汇总](current_regular.csv)。数值检查通过，但W4/K4尤其Dispatch有明显带宽波动。
尚未证明全面无回退，原因须同平面同版本对照诊断，不能由跨平面对照直接归因。

## 实现和验证边界

D默认每源1 producer+1 publisher，其余搬运。C分段并行校验Journal，再合并检查段间row/assignment连续性。
C两输入/两输出各16 KiB，总64 KiB；校验摘要每origin/ring为block_dim×W×64B，由planner分配，不增加网络消息。
延迟READY、空源、非对齐H、K8和ring复用已有通过记录；错误owner注入返回InvalidJournal=2，随后合法D→C恢复通过。
不等于穷尽故障或所有性能目标。进一步+10%目标未全部满足，见[过程记录](TEN_PERCENT.md)。

平面A Combine库SHA256为 `7e82ff141daadd7a03eb64e333442847862a8b2d5ecaa644dd8773be090842c7`。
开发树另有带溢出检查的行偏移优化，平面B库为 `c9bea135304df5e8b265a8a4d329e2ef81cc678714b3dacd7a4955571b6e2fee`，未混入平面A表。
Dispatch库为 `e1141bff51ed4c81b595c5c2984bafb8be6aa7a40d29e4dfa6e90f3265964ce8`。
