# Dispatch 统一源段执行器（nb，2026-09-09）

## 实现与边界

均匀路由、随机路由、同 GPU 多 expert assignment 共用 `RelaySourceRuns`。
唯一目标集合一致的输入合并为连续长段；变化的路由生成 token 短段，从已生成的
目标行表定位偏移。随后同一段执行器 GET 一份源 hidden，再向唯一目标 PUT。
删除固定 K1/K2/K4 fan-out 函数和整批暂存后逐目标标量重整的数据路径。

保留两项必要的条件处理：连续段合并优化，以及非对齐行的单写入者精确宽度搬运。
因此“统一”指公共段编排/搬运算法，不是每个输入执行完全相同的机器指令。
tile 大小沿用原传输策略，不改变 API、路由语义、Journal 或 ACK/completion。

测试中暴露了小消息竞态：布局生产者发布 Pass2Ready，relay 写时间戳，两者共享
缓存行，独立写回可能覆盖进度标志。现在先在 AIV 本地保存时间戳，生产者汇合后
再写回；修复后同一小消息工况累计400个wave未再超时。

## 正式128 MiB性能对照

nb 的同一 HCCS 平面 NPU0–7；W2用worker0/1、INC2，W4用worker0–3、INC4。
每个worker源hidden为128 MiB BF16，H=8192。每组3 warmup + 10 measure。
带宽仅为 **fan-out下行hidden字节 / 完整Dispatch调用时间**。

基线D源码与 `065f55a`（D与发布版b37a0fd相同）SHA256一致，独立重新编译，
与候选使用同一个harness/SHMEM依赖、编译参数及placement。
两个版本都使用相同的主机rank CPU亲和组：
`16,17:18,19:20,21:22,23:64,65`。计时包含READY等待，未减去调度延迟。

| 规模/路由 | 基线mean | 统一mean | mean变化 | 统一min | min变化 | 统一CV |
|---|---:|---:|---:|---:|---:|---:|
| W2 K2/GPU2 | 37.802 | 37.766 | -0.094% | 37.562 | -0.531% | 0.197% |
| W4 K2/GPU2 | 69.607 | 69.747 | +0.202% | 69.249 | -0.265% | 0.366% |
| W4 K4/GPU4 | 77.105 | 77.058 | -0.061% | 76.676 | -0.143% | 0.199% |
| W4 K4/GPU2 | 68.936 | 68.796 | -0.204% | 68.291 | -0.408% | 0.459% |
| W4 K8/GPU4 | 75.550 | 75.667 | +0.155% | 75.391 | -0.021% | 0.194% |

单位GB/s。五组均值与最低值的下降都小于1%，全部数值/协议/guard检查通过。
此结论仅限本机上述矩阵；不是对所有规模和时序的性能保证。
**51.52/103.04 GB/s绝对raw gate仍未达到，不能与“下降小于1%”混淆。**

保留未绑定CPU的失败对照：K4/GPU2 mean下降0.14%，min下降1.19%；最慢样本中
INC启动到all-READY约198µs，其余样本明显更短。此样本没有删除，也未通过裁剪
计时改写结果。固定相同亲和设置后重新对照，结果如上表。

## 安全验证

- 完整安全矩阵21组281wave，全通过：空输入、H=3/32/33/8192、随机路由、
  hotspot、非对齐尾块、W2/W4各100次连续ring复用，以及五类错误注入各3次。
- 另加W2/W4各3wave的75% token skew + 1000µs READY skew，均通过。
- 独立的小消息竞态回归W2/W4各100wave通过；与完整矩阵合计400个重复wave。
- 错误注入检查预期失败状态，不把失败wave计为性能样本。
- Host协议与API测试通过。真机只验证W2/W4；ABI最多128worker，实际launch还受
  `W × (channels + 1) <= AIV预算` 等资源约束，不声称已验证更大集群。
- W4、1MiB统一D/C同时启动冒烟通过：两算子数值正确，设备时间线确认重叠。
  仅一个小消息case，不作为大消息交叠性能或任意并发时序已覆盖的证据。

随机W4、1MiB的探索样本由约0.28提升到约7.78GB/s；这不是3+10正式A/B，
仅用于说明移除标量重整的效果，不作为性能gate证据。

## 复现

从shmem目录执行，输出目录必须不存在：

```bash
python3 examples/inc/dispatch_combine/single_inc/pull_combine/tests/pull_v2_dispatch_unified.py \
  --build-dir /tmp/inc-k4-build-20260909 --baseline-dir /tmp/dispatch-baseline-build \
  --output /tmp/dispatch-ab-new --mode performance \
  --cpu-groups 16,17:18,19:20,21:22,23:64,65

python3 examples/inc/dispatch_combine/single_inc/pull_combine/tests/pull_v2_dispatch_unified.py \
  --build-dir /tmp/inc-k4-build-20260909 --output /tmp/dispatch-safety-new --mode safety
```

baseline-dir应是基线源码的独立构建目录，具有bin和lib；不要覆盖候选库。
runner检查整个目标HCCS平面空闲，超时后仅回收自身子进程。

原始日志在本机 `/tmp/dispatch-unified-performance-pinned`、
`/tmp/dispatch-unified-performance-fixed`（包含失败样本）、
`/tmp/dispatch-unified-safety-fixed`、`/tmp/dispatch-unified-skew-fixed`、
`/tmp/dispatch-ready-race-regression`；不提交逐rank JSON/log。
