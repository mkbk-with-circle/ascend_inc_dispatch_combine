# 单 INC — 910b2c-nb 环境状态

> 候选环境；远端 live 探测与性能数据未完成前不作为 ACTIVE，也不用于 PASS 判定。
> 共享硬约束与 gate 见 [`../../report/single_inc_LIVE_STATUS.md`](../../report/single_inc_LIVE_STATUS.md)。

| 字段 | 值 |
|:---|:---|
| profile 名 | `910b2c-nb` |
| 上次更新 | 2026-08-06 |
| 更新者 | Codex |
| 对应共享文档 | `docs/inc/report/single_inc_LIVE_STATUS.md` |

## 1. 本机身份与软件

| 项 | 值 |
|:---|:---|
| 主机 / SSH | `A03-R40-I48-253-0046364.JD.LOCAL` / Codex remote project `npu-borrow` |
| 代码根 | `/export/home/yinjinrun.montyyin/.cursur/projects/default/shmem` |
| NPU / 型号 | live：16× Ascend 910B2C、65536 MiB HBM/卡 |
| 本次实测 CANN | `/usr/local/Ascend/cann-9.1.0-beta.3` |
| npu-smi / 驱动显示版本 | `25.0.rc1.1` |
| env 配置 | `docs/inc/configs/910b2c-nb.env` |

## 2. 本机拓扑与 AIV

| 项 | 值 |
|:---|:---|
| Live AIV 总数 | **48/卡**（`ACL_DEV_ATTR_VECTOR_CORE_NUM`，roofline binary 实时查询） |
| Live 拓扑 | 两个 8-card HCCS 岛：0–7、8–15；0↔8=PIX，矩阵已由 `npu-smi info -t topo` 复核 |
| W2 / W4 实测映射 | INC Phy0；workers `1 2` / `1 2 3 4`，逐边均 live 验证为 HCCS |
| W8 | **本环境不测试**：8-card 岛容纳不了 8W+1INC，无法构造等效链路 |
| 当前 D/C AIV | policy v1 live 验证：INC base D/C=`16/32`；worker W2 D/C=`8/24`、W4=`4/16`；identity-K1 relay owner 子 cohort W2/W4=`24/12` |
| NPU 空闲 / 独占 | 每次运行前后 16/16 无进程；持有 `/tmp/inc_single_inc_npu.lock` |

### 本环境规模口径

正式测试只覆盖 W2+1INC 与 W4+1INC，所有角色必须位于 0–7 同一 HCCS 平面。
W8 的跨平面混合结果不属于本环境验收数据。

## 3. 本机性能锚点

| 项 | 值 |
|:---|:---|
| W2 多打一 put-only | 128 MiB/peer×4 inner iterations；10 measure：min/mean/max=`42.714/42.734/42.741 GB/s`，CV=0.0221% |
| W2 一打多 put-only | min/mean/max=`42.804/42.815/42.822 GB/s`，CV=0.0136% |
| W2 duplex put-only | ingress min=37.615，egress min=37.611，aggregate min/mean/max=`75.223/75.246/75.263 GB/s`，CV=0.0176% |
| W4 多打一 put-only | min/mean/max=`85.445/85.478/85.527 GB/s`，CV=0.0244% |
| W4 一打多 put-only | min/mean/max=`83.258/84.038/85.650 GB/s`，CV=1.2484% |
| W4 duplex put-only | ingress min=74.932，egress min=73.139，aggregate min/mean/max=`146.278/146.461/146.553 GB/s`，CV=0.0631% |
| 正式口径 | warmup≥3、measure≥10、逐样本 CV≤5%、大消息≥128MiB |
| 原始报告 | `roofline_put_20260806/roofline_results.json`、`roofline_put_duplex_peak_20260806/roofline_results.json` |
| shared gate 锚点已更新 | **NO**（本轮仅验证 W2/W4 峰值，不修改 shared gate） |

### 链路口径结论

`224 Gbit/s = 28 GB/s raw` 是一条 worker↔INC HCCS peer link。W2/W4 只同时
使用 2/4 条，因此 raw 聚合上限分别约 56/112 GB/s；约196 GB/s是7条 link
全部参与时的 INC 总出口档，不是单 pair、W2 或 W4 的上限。实测每条约
21.36–21.41 GB/s，W2/W4均约为 raw 的76%。

为排除软件假峰值，另扫了 generic put lanes=1..48、chunk=64 KiB..32 MiB、
quiet=1..128，并用 contiguous/ping-pong direct-MTE 交叉诊断；所有 lanes≥4
路径都收敛在同一单边约21.4 GB/s档，direct-MTE最佳W2聚合42.747 GB/s。

## 4. 本机复测红绿灯

| Case | 正确性 | 带宽或 ratio | vs Gate | 报告路径 |
|:---|:---|---:|:---|:---|
| put roofline W2/W4 ingress/egress/duplex | PASS | 见 §3 | H11 PASS；shared gate 尚未改写 | `roofline_put_20260806/`、`roofline_put_duplex_peak_20260806/` |
| put roofline W8 | N/A | | 本环境明确排除混合链路 W8 | |
| Legacy W4/K8 D/C representative replay | 正确性 PASS | D=56.974 mean / C=73.518 GB/s | PERF GAP（均低于本机 roofline 90%） | `operator_replay_w4_k8_20260806/` |
| Random-route W×K×bytes campaign | UNVERIFIED | | | |
| Portability refactor W2/W4 sweep | 40/40 PASS | gated 29/30；唯一 W4/K1 Combine 128MiB=62.676<65.944；独立3跑 mean=66.205, CV=1.462% | 无系统性回退；边界 case 原 FAIL 保留 | `portability_regression_20260807/` |

本机结论：W2/W4全HCCS put-only峰值与H11稳定性已验证；W8暂按用户要求不进入主线。

### W4/K8 旧环境达标 case 复跑

| Case | 正确性 | 910b2c-nb | 本机 put roofline / 效率 | 旧环境对照 |
|:---|:---|---:|---:|---:|
| Dispatch W4/K8 @256MiB | 5/5 rank PASS | mean 56.974，max 57.760 GB/s，CV=0.885% | 85.650 / 66.52% mean | 910b-yuanmingyu 130.57 |
| Combine W4/K8 @128MiB | 5/5 rank PASS | 73.518 GB/s | 85.527 / 85.96% | 910b-yuanmingyu 128.37 |

原始日志、机器可读汇总和旧报告校验和见
`operator_replay_w4_k8_20260806/`。两项均正确且确认使用 single-INC 路径，
但相对当前环境 put-only roofline 均未达到 90%。

## 5. 本机迁移 Checklist

- [x] 当前 Codex project 直接运行于远端，代码版本确认
- [x] 明确 source CANN 9.1.0-beta.3
- [x] `npu-smi` 空闲且取得独占锁
- [x] live 16×16 topo
- [x] live AIV
- [x] W2/W4 全 HCCS 验证
- [x] W2/W4 put-only ingress / egress / duplex roofline
- [x] legacy W4/K8 Dispatch/Combine representative replay
- [ ] reproducible random-route campaign
- [ ] 仅在成为 ACTIVE 后更新 `ACTIVE_HW_PROFILE.md`

## 6. 本机 Changelog

| 日期 | 谁 | 变更 |
|:---|:---|:---|
| 2026-08-06 | Codex | 从模板创建候选 profile；记录 8-card 岛下 W8+1INC 不可能全等效；远端认证阻塞，未填伪数据 |
| 2026-08-06 | Codex | 确认会话直接位于 npu-borrow；完成 W2/W4 put-only 参数扫描与H11峰值验证，记录每-link物理口径 |
| 2026-08-06 | Codex | 明确本环境只验收同平面 W2/W4；复跑旧环境 W4/K8 D/C 达标 case，结果独立归档且保留旧报告校验和 |
| 2026-08-07 | Codex | 完成低风险可移植性重构后 W2/W4 40-case sweep：40/40 正确、29/30 gated PASS；统一 capability/policy/map fingerprint，旧 yuanmingyu 与 nb 历史数据均未覆盖 |
