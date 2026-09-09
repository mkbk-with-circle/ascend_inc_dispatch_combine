# Pull V2 W4 固定 Expert Top-k4 / Top-k8 正式测试（2026-09-09）

## 范围

- 代码：`cb94ed0`（测试 workload/runner），依赖运行时修复 `cc09de2`；通信 kernel
  本体未改。
- 硬件：nb-borrow，W4+1INC；平面 A 使用 NPU0--4，平面 B 使用 NPU8--12，
  每组5个rank均在同一HCCS平面。
- 尺度：H=8192；Dispatch source hidden=128 MiB/worker（BF16）；Combine local
  partial=128 MiB/worker（FP32）。
- 每case：3 warmup + 10 measure；所有5个rank必须正常退出，每个设备样本必须
  `correct=true`；CV≤5%。
- D/C 使用完全相同的固定路由规则，但仍是两个独立设备算子测试，不是同一批token
  的公开API整链。

## 固定路由

| Workload | Expert K | Unique GPU K | 每GPU expert数 | Combine后Partial K | Dispatch路径 |
|---|---:|---:|---:|---:|---|
| `sym_k4_gpu4` | 4 | 4 | 1 | 4 | uniform快路径 |
| `sym_k4_gpu2` | 4 | 2 | 2 | 2 | 安全重整路径 |
| `sym_k8_gpu4` | 8 | 4 | 2 | 4 | 安全重整路径 |

每个token的expert ID和weight都是固定、确定性的。K4权重均为0.25；K8权重均为
0.125。相同GPU上的多个expert先在B端按weight本地归约，再形成一份FP32 partial。

## 结果

| Operator | Workload | Min GB/s | Mean GB/s | CV | 正确性/稳定性 | vs 103.04参考值 |
|---|---|---:|---:|---:|---|---|
| Dispatch | K4 / GPU4 | 90.227 | 91.532 | 0.848% | PASS | FAIL |
| Combine | K4 / GPU4 | 84.401 | 84.577 | 0.173% | PASS | FAIL |
| Dispatch | K4 / GPU2 | 0.376 | 0.383 | 1.164% | PASS | FAIL |
| Combine | K4 / GPU2 | 107.229 | 107.478 | 0.141% | PASS | PASS |
| Dispatch | K8 / GPU4 | 0.311 | 0.323 | 2.509% | PASS | FAIL |
| Combine | K8 / GPU4 | 81.797 | 84.402 | 1.064% | PASS | FAIL |

合计6个case、78个operator wave；全部数值正确、无超时、无非零rank退出，测试后
16张NPU均无进程。所有case CV≤5%，因此正确性和重复稳定性通过。

`103.04 GB/s` 是既有 W4 nominal 参考值。原验收计划只把它定义为
`sym_k2_balanced` 的hard gate；本报告对K4/K8只给出参考比较，不擅自扩展既有
hard-gate适用范围。

## 结论

1. K4/GPU4 Dispatch 满足uniform条件，能使用分块ping/pong路径，但fanout从2增至4
   后最低为90.227 GB/s，低于103.04参考值。
2. K4/GPU2和K8/GPU4都含“同token、同GPU多个assignment”，当前uniform判定拒绝
   该布局，Dispatch退到先拉齐source、再逐目标标量打包的安全路径，带宽降至
   0.31--0.38 GB/s；这是明确的性能缺口，不是正确性失败。
3. K4/GPU2经过本地归约后仍只有2份partial，Combine维持107.229 GB/s最低值。
4. K4/GPU4与K8/GPU4在本地归约后都形成4份partial，网络payload相同，因此两者
   Combine平均带宽接近84.5 GB/s；增加同GPU expert数没有增加hidden/partial副本。

## 复现

```bash
python3 examples/inc/dispatch_combine/single_inc/pull_combine/tests/\
pull_v2_k4_formal.py \
  --build-dir /tmp/inc-k4-build-20260909 \
  --output /tmp/pull-v2-w4-fixed-topk-new \
  --first-npu 0
```

Runner拒绝覆盖已有输出目录，逐平面检查目标NPU是否空闲，并为每个iteration统一
释放5个rank。原始逐PE日志保存在实验机 `/tmp/pull-v2-formal-20260909-*`，不进入Git。
