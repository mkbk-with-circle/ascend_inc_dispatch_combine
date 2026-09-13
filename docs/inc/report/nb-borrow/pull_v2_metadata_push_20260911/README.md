# Pull-Dispatch V2 Metadata Push A/B（2026-09-11）

本报告验证控制面从 INC GET 改为 Worker PUT 后的正确性、启动延迟与完整算子带宽。
测试位于 nb-borrow 同一 HCCS 平面 A，使用 CANN 9.1.0-beta.3；W2 为 NPU
0/1 + INC 2，W4 为 NPU 0--3 + INC 4。

最终新协议原始日志：`/tmp/pull-v2-metadata-push-directional-formal-20260911`；
旧协议 A/B 日志：`/tmp/pull-v2-header-metadata-20260911`（W2）与
`/tmp/pull-v2-header-metadata-w4-rerun-20260911`（W4）。

## 协议变化

旧协议：

```text
READY -> INC GET Header/metadata -> parse -> GET hidden -> fan-out
```

新协议：

```text
Worker PUT Header/metadata -> remote completion -> publication-last READY
-> INC local parse -> GET hidden -> fan-out
```

`Ready` 仍为 64 B，只携带身份、注册区、ring 和 publication。Header/metadata
写入按 source 分片的紧凑对称 INC inbox；完整 source slot 和 general-path hidden
staging 不会因此变成对称大内存。Source ACK 前，Worker source slot 保持不可变。

## 正确带宽口径

主带宽只计算单一数据方向：

```text
Dispatch bandwidth = INC -> Worker fan-out hidden bytes / full Dispatch time
Combine bandwidth  = Worker -> INC reduction partial bytes / full Combine time
```

完整 Dispatch 时间从 launch 前开始，包含 metadata PUT、READY publication、INC
解析、hidden GET、fan-out、completion 和 Source ACK，直到最慢 rank 完成。
`hidden GET + fan-out PUT` 两腿相加的速率只作为 aggregate traffic 诊断，不再称为
链路或算子主带宽。这样得到的单向结果均低于同机历史 W2/W4 多打一参考
42.734/85.478 GB/s。

## 128 MiB/worker 正式 A/B

固定 H=8192、BF16、top-k2 balanced、3 warmup + 10 measure；所有 rank 与全部
样本正确，CV < 0.5%。

| 规模 | 协议 | mean time | max time | mean 下行 | min 下行 | CV |
|---|---|---:|---:|---:|---:|---:|
| W2 | Header/metadata GET | 14.228 ms | 14.325 ms | 37.735 GB/s | 37.478 GB/s | 0.396% |
| W2 | Metadata PUT | 14.244 ms | 14.297 ms | 37.691 GB/s | 37.552 GB/s | 0.357% |
| W4 | Header/metadata GET | 15.302 ms | 15.389 ms | 70.170 GB/s | 69.774 GB/s | 0.328% |
| W4 | Metadata PUT | 15.339 ms | 15.481 ms | 70.001 GB/s | 69.360 GB/s | 0.534% |

完整时间变化：W2 +0.12%，W4 +0.24%；控制面方向切换在完整算子口径下基本无损。

READY 到 metadata 解析完成的平均时间：

| 规模 | 旧协议 | Metadata PUT | 缩短 |
|---|---:|---:|---:|
| W2 | 7.397 ms | 7.176 ms | 0.221 ms（2.99%） |
| W4 | 7.523 ms | 7.346 ms | 0.177 ms（2.36%） |

该指标只解释 notify 后启动阶段；最终是否无损仍以上表完整算子时间为准。

## 补充正确性

- W2 1 MiB balanced smoke：PASS。
- W2 空 balanced / 空 ragged：PASS。
- W2 1 MiB ragged general path：PASS。
- W2 1 MiB D/C simultaneous overlap：两侧正确且设备 cycle 证明真实重叠；Dispatch
  日志主分子为 4,194,304 B fan-out 下行，Combine 日志主分子为 2,097,152 B
  partial 上行，aggregate traffic 另列且不混入主带宽。
- Host Pull-Dispatch V2 单测：PASS。
- Device kernel、Dispatch E2E、same-session probe target：编译通过。
