# Single-INC Top-K / Rank 去重压测（CANN 9.1，2026-08-04）

## 结论

- 生产主线 `single_inc_stream` 已真机通过 `K>W` 且 `K>8`：Dispatch 覆盖 K=9/16/64，Combine 覆盖 K=16/64，所有样本无 mismatch / hang。
- Dispatch 保留所有 expert assignment，hidden payload 按 `(token,destination rank)` 去重。W8/K64 balanced 是 8:1，all-to-one 是 64:1；每 token 物理份数上界是 W，不是 `8*W`。
- Dispatch W8/K64 100 次连续下发：balanced 132.743 GB/s，CV=0.74%；all-to-one 115.205 GB/s，CV=0.71%，全部正确。
- Combine 默认仍传输全部 expert-instance，`rank_dedup=0`。256 MiB 下 W8/K8/K16/K64=126.863/127.053/127.767 GB/s，大 top-k 无回退。
- Combine 128 MiB 下 W8/K8/K16/K64 约 116.16–116.41 GB/s，是已有 128 MiB 容量边界缺口，不是 K>W 导致。100 epoch W8/K64=117.818 GB/s，mismatch=0。
- Combine all-to-one=60.108 GB/s，约为单 worker 上行屋顶 62.624 GB/s 的 96%；因此固定 120 GB/s 聚合 gate 对单源热点在物理上不可达。Ragged 在 8 worker 均参与时仅 109.959 GB/s，是真实的调度/元数据开销缺口。
- W8/K64 256 MiB Dispatch∥Combine 真实交叠全 rank 正确；balanced makespan=2422.06 us，相对 solo 串行 4089.32 us 加速 1.688×，距理论下界 15.28%；all-to-one 加速 1.267×，距理论下界 16.32%。
- 8190-byte 非对齐 row 和 2-byte 极小 row 均正确；前者 Dispatch/Combine=121.623/126.430 GB/s。8 个 route/CSR/framework host gate 全部 PASS（其中 C header 为 syntax-only 编译 gate）。

## 口径

- CANN: `/opt/Ascend-9.1/cann-9.1.0-beta.1`
- 硬件：48 AIV，INC Phy0，worker 全为 live `HCCS_SW`。
- Dispatch: 128 MiB 物理 INC→worker 字节/sample，warmup=3，measure=10 或 100；每 sample 取最慢 rank device interval，再统计带宽/CV。
- Combine: 128/256 MiB 物理 worker→INC 字节/epoch，warmup=3，10 或 100 个 device-queued epoch；带宽分母是最慢 rank 的整个 persistent service span。
- 资源在 K/字节/路由间不变：INC D/C=16/32；worker W2/W4/W8 D/C=8/24、4/16、2/12，均 `<=24`。

## Dispatch 关键带宽

| Case | 路由 | 逻辑/物理行去重 | GB/s | CV | 正确性 |
|:---|:---|---:|---:|---:|:---:|
| W2/K9 | balanced | 4.5× | 122.929 | 0.62% | PASS |
| W2/K16 | balanced | 8× | 122.845 | 0.66% | PASS |
| W4/K16 | balanced | 4× | 133.708 | 1.21% | PASS |
| W8/K8 | balanced baseline | 1× | 133.265 | 0.52% | PASS |
| W8/K16 | balanced | 2× | 132.782 | 0.62% | PASS |
| W8/K16 | half-rank | 4× | 132.029 | 0.84% | PASS |
| W8/K64 | balanced | 8× | 132.561 | 0.63% | PASS |
| W8/K64 | all-to-one | 64× | 115.885 | 0.61% | PASS |
| W8/K64, 100 samples | balanced | 8× | 132.743 | 0.74% | PASS |
| W8/K64, 100 samples | all-to-one | 64× | 115.205 | 0.71% | PASS |

W8/K64 all-to-one 每 sample 保留 1,048,576 个 assignment（8 GiB 逻辑 expert payload），但只有 16,384 个去重物理行（128 MiB）。

## Combine 关键带宽

| Case | 物理字节/epoch | GB/s | 120 gate | 正确性 |
|:---|---:|---:|:---:|:---:|
| W2/K16 balanced | 128 MiB | 125.248 | PASS | PASS |
| W4/K16 balanced | 128 MiB | 123.926 | PASS | PASS |
| W8/K8 balanced | 128 MiB | 116.177 | FAIL | PASS |
| W8/K16 balanced | 128 MiB | 116.413 | FAIL | PASS |
| W8/K64 balanced | 128 MiB | 116.161 | FAIL | PASS |
| W8/K64 balanced, 100 epoch | 128 MiB | 117.818 | FAIL | PASS |
| W8/K8 balanced | 256 MiB | 126.863 | PASS | PASS |
| W8/K16 balanced | 256 MiB | 127.053 | PASS | PASS |
| W8/K64 balanced | 256 MiB | 127.767 | PASS | PASS |
| W8/K16 half-rank | 128 MiB | 119.379 | FAIL | PASS |
| W8/K16 ragged | 136 MiB | 109.959 | FAIL | PASS |
| W8/K16 two-rank/zero-rank | 128 MiB | 63.405 | FAIL* | PASS |
| W8/K16 all-to-one | 128 MiB | 60.108 | FAIL* | PASS |

`FAIL*` 表示未过当前固定 120 gate，但受 active-source 物理屋顶限制。以 W2/K16 balanced 的聚合 125.248 GB/s 估算单 worker 上行屋顶为 62.624 GB/s：all-to-one 效率约 96.0%；two-rank 案例的 rank0 承担 93.75% 字节，其拓扑上界约 66.8 GB/s，实测约 94.9% 上界。这不改变当前 formal gate 的 FAIL 记录。

## K64 交叠与边界压力

| 路由 | D solo | C solo | 理论下界 | concurrent makespan | ratio / speedup | 高于下界 |
|:---|---:|---:|---:|---:|---:|---:|
| balanced | 1988.34 us | 2100.98 us | 2100.98 us | 2422.06 us | 0.5923 / 1.688× | 15.28% |
| all-to-one | 2320.37 us | 4899.59 us | 4899.59 us | 5699.04 us | 0.7893 / 1.267× | 16.32% |

两组都使用 live hardware/topology 固定资源：INC D/C=16/32，worker D/C=2/12，`resource_contract_pass=true`。balanced 并发方向带宽 D/C=115.322/114.452 GB/s；all-to-one=95.380/47.102 GB/s。

边界样本：

- W2/K16, hidden row=8190 bytes：Dispatch 121.623 GB/s，CV=0.83%；Combine 126.430 GB/s，正确。
- W2/K16, hidden row=2 bytes：Dispatch/Combine 均正确完成；小于 128 MiB，不做带宽 gate。
- Host gates: contract/reference/logical-plan/CSR-reduce/dyn2r-repair/framework-route-binding/framework-C-API 运行 PASS，C header syntax-only 编译 PASS。

## 回归判断与开放项

1. **Top-k 扩展 PASS**：K=64 不截断 assignment，Dispatch 去重正确，Combine 默认完整 INC 归约。
2. **无 top-k 性能回退**：Dispatch W8/K64 vs K8=-0.53%；Combine 256 MiB W8/K64 vs K8=+0.71%，均在噪声范围内。
3. **OPEN-C128**：W8 balanced 128 MiB 容量边界稳定低于 120，与 K 无关。
4. **OPEN-RAGGED**：全 worker 参与的 ragged 路由仅 109.959 GB/s，需分解 scheduler/ready/metadata 开销。
5. **GATE-MODEL**：热点 case 应额外报告 `min(INC roofline, active-source/load-share roofline)` 效率；在用户批准前不替换现有 120 formal gate。
6. **OPEN-K64-OVERLAP**：K64 balanced/all1 真实交叠均无死锁且资源合规，但 makespan 距理论下界仍有 15.28%/16.32%。

可机读摘要：`single_inc_topk_rank_stress_cann91_20260804.json`。本轮原始日志：`/tmp/single-inc-topk-rank-stress-20260804-r1`。
