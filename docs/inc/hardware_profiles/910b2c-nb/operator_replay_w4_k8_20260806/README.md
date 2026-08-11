# 910b2c-nb：W4/K8 旧达标 case 复跑

本目录只包含 `910b2c-nb`（npu-borrow，16×910B2C，CANN
9.1.0-beta.3）的新测数据。旧环境 `910b-yuanmingyu` 的报告保持原路径和
原内容；`results.json` 记录了三份旧环境真源的 SHA-256，便于检查未被覆盖。

## 同口径对照

| Case | 910b-yuanmingyu 历史值 | 910b2c-nb 本次值 | 当前 put-only roofline | roofline 效率 |
|:---|---:|---:|---:|---:|
| Dispatch W4/K8，256 MiB | 130.57 GB/s | min/mean/max = 56.126/56.974/57.760 GB/s | W4 INC→all max 85.650 GB/s | mean 66.52% |
| Combine W4/K8，128 MiB | 128.37 GB/s | 73.518 GB/s | W4 all→INC max 85.527 GB/s | 85.96% |

两项均使用 INC Phy0、worker Phy1/2/3/4，四条边由运行时再次验证为 HCCS；
warmup=3、measure=10，5/5 rank 正确。Dispatch 的每个 measure 单独以最慢
rank 计时并报告 CV；Combine 是一个覆盖 10 个预排队 epoch 的 persistent
device interval，因此只有整体 makespan，不伪造逐 epoch CV。

## 判定

- 正确性与 single-INC 路径约束通过。
- 当前实现没有保持旧环境的带宽等级。
- 相对本机 put-only roofline，Dispatch 约 66.5%，Combine 约 86.0%；两者均
  未达到 90%，其中 Dispatch 缺口更大。
- 本环境后续正式矩阵只测同平面 W2+1INC 和 W4+1INC；不把混合链路的
  W8+1INC 纳入该环境验收。

完整机器可读结果见 `results.json`，原始 rank 日志保留在两个 case 子目录。
