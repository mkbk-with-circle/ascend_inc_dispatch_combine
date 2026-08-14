# Single-INC 非 2 次幂 token sweep（CANN 9.1，2026-08-05）

## 结论

- 覆盖 Dispatch/Combine × W2/W4/W8 × 约 128/256 MiB，共 **24 组**；所有
  `tokens_per_worker` 都不是 2 的幂，warmup=3、measure=10。
- 精确重跑后 **24/24 全 rank 正确**，正式 protocol telemetry **1360/1360**；
  每组 10 个带宽样本的 CV 为 **0.24%–1.37%**。
- 性能 hard gate 为 **11/24 PASS**：Combine K>1 6/6 PASS；Combine K1 0/6
  PASS；Dispatch 非 K1 5/6 PASS，唯一失败是约 128 MiB W2/K2；Dispatch
  K1 0/6 PASS。
- 本轮暴露 native Dispatch 的两个真实性能缺口：K1 在两个尺度都只有
  **68.30–70.21 GB/s**；W2/K2 从约 128 MiB 的 **82.93 GB/s** 上升到约
  256 MiB 的 **124.48 GB/s**，说明前者是明显的小尺度/固定成本悬崖。
- 本次未修改 device kernel、传输语义或 AIV map。实测资源仍为 Dispatch
  INC/worker=`16/{8,4,2}`，Combine=`32/{24,16,12}`（W2/W4/W8）；worker
  始终 `<=24` 且 `<=live_AIV/2`。

## 带宽结果

口径：对每个 sample 取所有 worker + INC rank 的
`max(protocol_rank_us)`，然后计算
`physical_direction_bytes / max_rank_time`；表中单位为 GB/s。

| Op | Tier | W | tokens/worker | K | physical bytes | mean | min | max | CV | Gate |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| Dispatch | 128M | 2 | 4099 | 1 | 134316032 | 68.464 | 67.371 | 68.968 | 0.69% | FAIL |
| Dispatch | 128M | 2 | 2053 | 2 | 134545408 | 82.933 | 81.469 | 84.728 | 1.31% | FAIL |
| Dispatch | 128M | 4 | 2053 | 1 | 134545408 | 68.329 | 67.106 | 68.906 | 0.78% | FAIL |
| Dispatch | 128M | 4 | 521 | 4 | 136577024 | 129.980 | 127.733 | 131.385 | 0.77% | PASS |
| Dispatch | 128M | 8 | 1031 | 1 | 135135232 | 68.304 | 67.666 | 68.654 | 0.45% | FAIL |
| Dispatch | 128M | 8 | 131 | 8 | 137363456 | 128.978 | 126.848 | 130.628 | 0.92% | PASS |
| Dispatch | 256M | 2 | 8201 | 1 | 268730368 | 69.562 | 68.617 | 69.932 | 0.53% | FAIL |
| Dispatch | 256M | 2 | 4099 | 2 | 268632064 | 124.482 | 123.639 | 125.468 | 0.47% | PASS |
| Dispatch | 256M | 4 | 4099 | 1 | 268632064 | 70.212 | 69.196 | 70.820 | 0.68% | FAIL |
| Dispatch | 256M | 4 | 1031 | 4 | 270270464 | 132.499 | 130.181 | 133.454 | 0.70% | PASS |
| Dispatch | 256M | 8 | 2053 | 1 | 269090816 | 70.090 | 68.685 | 70.743 | 0.92% | FAIL |
| Dispatch | 256M | 8 | 257 | 8 | 269484032 | 129.918 | 128.060 | 132.269 | 0.83% | PASS |
| Combine | 128M | 2 | 4099 | 1 | 134316032 | 113.646 | 112.964 | 114.360 | 0.42% | FAIL |
| Combine | 128M | 2 | 2053 | 2 | 134545408 | 135.037 | 132.829 | 136.630 | 0.81% | PASS |
| Combine | 128M | 4 | 2053 | 1 | 134545408 | 115.231 | 113.713 | 116.504 | 0.72% | FAIL |
| Combine | 128M | 4 | 521 | 4 | 136577024 | 144.087 | 142.974 | 146.228 | 0.65% | PASS |
| Combine | 128M | 8 | 1031 | 1 | 135135232 | 117.404 | 116.855 | 118.380 | 0.37% | FAIL |
| Combine | 128M | 8 | 131 | 8 | 137363456 | 140.196 | 138.530 | 143.350 | 1.16% | PASS |
| Combine | 256M | 2 | 8201 | 1 | 268730368 | 117.339 | 116.151 | 118.224 | 0.61% | FAIL |
| Combine | 256M | 2 | 4099 | 2 | 268632064 | 140.481 | 138.377 | 143.345 | 1.07% | PASS |
| Combine | 256M | 4 | 4099 | 1 | 268632064 | 112.805 | 110.253 | 115.505 | 1.37% | FAIL |
| Combine | 256M | 4 | 1031 | 4 | 270270464 | 144.689 | 144.087 | 145.716 | 0.35% | PASS |
| Combine | 256M | 8 | 2053 | 1 | 269090816 | 115.302 | 114.790 | 115.674 | 0.24% | FAIL |
| Combine | 256M | 8 | 257 | 8 | 269484032 | 138.010 | 136.694 | 139.362 | 0.58% | PASS |

Dispatch 使用 ACTIVE CANN 9.1 down roofline 123.339 GB/s，因此
R=1/2/4/8 gate 分别为 101.779/107.856/111.175/112.913 GB/s。Combine
K1/K>1 均使用 120 GB/s hard gate。

## 启动稳定性与计时完整性

- 本轮正式 telemetry campaign 首次为 **23/24**：W2/T4099/H16384/K2
  Dispatch 的 PE0 在 ACLSHMEM init 阶段退出，另两 rank 未进入算子。保留
  `dispatch-w2-t4099-h16384-k2-m256-startup-fail`；同 shape 立即独立重跑后
  3/3 rank、30/30 protocol、20/20 event PASS。按 H11/H12，首次启动失败不被
  重跑抹掉，仍是一个环境/启动稳定性问题。
- 更早一次预跑中，同 shape Combine 也出现过一次 init 退出，失败日志保留为
  `combine-w2-t4099-h16384-k2-m256-startup-fail`；本轮 3/3 rank 成功。两次故障
  都发生在 operator protocol 之前，不是 mismatch 或 device-kernel timeout。
- ACL event 是辅助数据：Dispatch 548/560，Combine 540/560。缺失样本均是
  CANN 返回 elapsed=0；对应 protocol telemetry 完整且正确性通过。正式 gate
  不再被这个辅助 event 假失败影响。

## 可复现证据

- sweep 脚本：`examples/inc/dispatch_combine/scripts/run_single_inc_native_nonpow2_sweep.sh`
- raw logs：`/tmp/inc-dc-native-nonpow2-sweep-20260805`
- CANN：`/opt/Ascend-9.1/cann-9.1.0-beta.1`
- Host product gate：PASS。

