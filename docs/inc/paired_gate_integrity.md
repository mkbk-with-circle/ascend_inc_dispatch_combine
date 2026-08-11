# PAIRED Gate 诚信约束

## 规则

1. 阶段 todo **不得**标为 `completed`，除非对应产物文件存在且含要求字段。
2. 产物必须来自真实跑数（本机 NPU / 本 build），禁止手写假 PASS。
3. `functional_pass` / `correctness_pass` / `pass` 必须以日志与 JSON 一致为准。

## 各阶段最低证据

| 阶段 | 必需产物 | 必需字段 |
|------|----------|----------|
| MP0 | `d03_paired_mp0_baseline.json/.csv` + `d03_paired_mp0_analysis.md` | `pass`, `cases[].functional_pass`, `paired_e2e_gbps`, `paired_worker_p50_gbps`, `paired_switch_p50_gbps` |
| MP0b | `d03_paired_mp0b_traffic_model_gate.json/.csv` | `pass`, 守恒 checks |
| MP1 | `d03_paired_mp1_placement_gate.json` + `d03_paired_mp1_placement_matrix.csv` | `pass`, `maps[]`, `DC_PLACEMENT` 证据 |
| MP2+ | 计划中对应 `d03_paired_mp*` JSON | 各阶段 `pass` / correctness / perf |

## 禁止

- Gate FAIL 时因“性能数字好看”而 completed。
- 混用不同 build/环境的 gate 到同一 tag。
- 在未跑同配置 baseline 时宣称接近 SHMEM。
