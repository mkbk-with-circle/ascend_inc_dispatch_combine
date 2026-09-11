# 随机路由再提升 10%：候选记录

状态：尚未全部达标，不宣称已通过最终无回退验收。改动前代码保存在 `feca99b`，对应运行库保存在 `/tmp/inc-tenpct-baseline-20260910`；没有推送候选。

## 固定目标与协议

沿用上轮每配置 3 种子、30 个测量样本的均值，目标不随候选结果下调：

| 算子 | W / expert K | 基准均值 GB/s | +10% 目标 GB/s |
|---|---|---:|---:|
| Dispatch | 2 / 2 | 33.558 | 36.914 |
| Dispatch | 4 / 2 | 61.455 | 67.601 |
| Dispatch | 4 / 4 | 64.712 | 71.183 |
| Combine | 2 / 2 | 34.547 | 38.002 |
| Combine | 4 / 2 | 65.377 | 71.915 |
| Combine | 4 / 4 | 67.882 | 74.670 |

源 rank 独立分区；每源每 wave READY；INC GET 输入和路由、解析、按唯一目标 GPU fan-out；B 本地归并后 Notice；INC 读取真实 Journal、GET partial、reduce、PUT owner、ACK/Completion，均保持不变。D 分子仍为实际 fan-out hidden，C 为实际 FP32 ingress，时间仍包含完整算子提交和同步。

## 候选改动

- Dispatch 自动分工保留一个 metadata producer 和一个 publisher，其余 AIV 搬运，不增加 INC 的半数 AIV 总额。取消旧的每源 5 个搬运 AIV 上限。
- Combine 以实际后端 UB 上限校验输入/输出缓冲，保留 16 KiB 分块、两输入和双输出，合计 64 KiB。
- CANN 的 `x86_64-linux/data/platform_config/Ascend910B2C.ini` 明确 `ub_size=196608`，SHMEM 的 `src/device/shmemi_device_common.h` 也为 192 KiB。旧 24 KiB 是软件 tile 预算，不能称为硬件 UB 容量。普通 AIV 数量、D/C 分配不变。
- Journal 校验按连续 token 区间并行；每 AIV/worker 一条 64B 摘要。校验全部 token 字段、贡献集合、计数、边界后，再合并检查跨区间 dense row 和 assignment 连续性。任何错误均在拉取 partial 前拒绝。
- 增加 INC 本地 workspace：`block_dim × workers × 64B`，按 origin/ring 隔离，planner 和示例同步分配，不增加任何网络 metadata。
- 行偏移为二次幂时采用仍带溢出检查的移位；其他 hidden 保留精确乘法回退。

已撤回：相邻 token 合并 GET（无明确额外收益）；32 KiB 大分块及其试验游标修改（W4 回退）。不以更多复杂路径保留无收益试验。

## 已完成证据与未完成项

- 候选 smoke：W2/W4 空输入、随机输入、W4 空源及 H33 尾部通过。
- W2/W4 延迟 source 0 的 READY 100ms，其他非空源仍提前完成；真实 D→C 和环形复用通过。
- 故障注入：将 Journal 中间 token 的 owner 改成非法 rank，返回 `InvalidJournal=2`；所有本次测试进程退出后设备空闲，随后合法随机 D→C 恢复测试通过。此项不是带宽样本。
- host layout 单测通过；已更新新 scratch 区域的偏移、大小和隔离检查。
- W2 Combine 同卡组正式测量（3 warmup+10 measure，3 seeds）均值分别 38.5857 / 38.6494 / 38.5861 GB/s，超过 +10% 均值目标。
- Dispatch 三种子正式均值 W2/K2=34.2268、W4/K2=63.2847、W4/K4=66.6285 GB/s，未达+10%目标。
- Combine 三种子正式均值 W4/K2=71.3052、W4/K4=72.2863 GB/s，均仍低于对应+10%目标。
- 行偏移候选额外通过 K8、H=32/4096/16384/32768、源行数 [1,7,8,9] 的真实 D→C，各10轮；规则回归数值通过，但W4/K4带宽波动明显。

本轮正式矩阵已完成，汇总与CSV见[当前结果](CURRENT_RESULTS.md)；不能把短测或平面B数据混成平面A的30样本结果。原始记录：

```text
/tmp/inc-tenpct-final-random-20260910/summary.csv  # 平面 A：D 分工 + C 并行校验/64KiB
/tmp/inc-tenpct-fault-20260910/                  # 错误 owner 拒绝与恢复
/tmp/inc-tenpct-offset-probe-20260910/           # 平面 B：增加精确行偏移快速计算
/tmp/inc-tenpct-edge-20260910/                  # K8、不同 H、短尾与跨块边界
/tmp/inc-tenpct-regular-20260910/                # 规则路由 D/C 回归
```

候选库 SHA256：

```text
7e82ff141daadd7a03eb64e333442847862a8b2d5ecaa644dd8773be090842c7  平面A Combine
e1141bff51ed4c81b595c5c2984bafb8be6aa7a40d29e4dfa6e90f3265964ce8  Dispatch
c9bea135304df5e8b265a8a4d329e2ef81cc678714b3dacd7a4955571b6e2fee  平面B Combine（含行偏移优化）
```
