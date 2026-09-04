# Single-INC Pull V2 环境状态模板

复制到 `docs/inc/hardware_profiles/<profile>/single_inc_ENV_STATUS.md`。新环境在完成
以下项目之前必须标记为 UNVERIFIED，不能复用 nb 的绝对带宽数字。

## 环境

| 项 | 值 |
|---|---|
| NPU/数量/HBM | |
| CANN/驱动 | |
| 普通 AIV | 运行时查询 |
| HCCS/PCIe 拓扑 | |
| INC 与 Worker placement | |
| 可等效测试规模 | |

## 链路与 Gate

| 规模 | ingress roofline | egress roofline | raw 聚合口径 | gate 比例/数值 |
|---|---:|---:|---:|---:|
| | | | | |

必须同时说明完整算子的计时区间和 logical-byte 分子，不能只用单程链路代替算子。

## 正式矩阵

| 算子 | W/top-k/bytes | 正确性 | Min/Mean/CV | Gate | 结果 |
|---|---|---|---|---|---|
| Dispatch | | | | | UNVERIFIED |
| Combine | | | | | UNVERIFIED |

## 必做检查

- [ ] 运行前后所有目标 NPU 空闲并取得独占锁；
- [ ] live topology 与 placement 等效；
- [ ] publication/cookie/digest/capacity/guard 正确；
- [ ] warmup≥3、measure≥10、CV≤5%；
- [ ] 空输入、尾块、不同 top-k、重复目的、ragged 和 skew；
- [ ] 连续 wave 与 ring-slot 复用；
- [ ] Dispatch/Combine 并发与错峰；
- [ ] 原始日志本地归档，Git 只提交汇总和复现命令。
