# nb-borrow 单 INC Fusion Kernel 阶段报告（2026-08-09）

本报告只记录 `910B2C-nb` 上的新 fusion kernel 数据，不覆盖、不改写此前
`single_inc_overlap_*`、`roofline_put_*` 或 yuanmingyu 环境的任何结果。

## 环境与映射

- CANN：9.1.0-beta.3；驱动：25.0.rc1.1；每卡 48 AIV / 24 AIC。
- 单 HCCS 平面：NPU 0–7。
- W2：worker NPU 1/2，INC NPU 0。
- W4：worker NPU 1/2/3/4，INC NPU 0。
- 测试前 `npu-smi info` 显示全部 NPU 无运行进程。

## 已完成的协议修复

1. header 的 metadata 与 ready 原先一次 64B put，接收端可看到新的 ready/generation 和旧 token。ABI v2 改为 metadata 完成后再单独发布 64-bit commit。
2. `FusionReceivedAssignment` 原为 32B，相邻条目会被不同 RX AIV 并发写同一 cacheline。现改为每条 64B 独占 cacheline。
3. 消费远端 payload 前补齐 DCCI acquire，覆盖 INC 转发、worker Dispatch RX、INC reduction、worker result RX 四条路径。
4. INC Combine 从 producer-major 阻塞接收改为 producer round-robin try-receive，消除固定深度 owner stream 的队头环路。
5. BF16 reduction 复用 UB 的 1024-element tile 边界补 `PIPE_ALL`；修复 hidden≥2048 后第二 tile 权重错误。

## 稳态 W2/W4（tokens=32, H=256, I=512, top-k=2, activation waves=2）

每个进程内 warmup=3、measure=10；表中时间取每次所有 worker 与 INC 的最大值。

| 规模 | 模式 | mean (us) | median (us) | min–max (us) | CV | 正确性 |
|---|---:|---:|---:|---:|---:|---|
| W2 | D/C 并发 | 604.480 | 606.240 | 585.080–622.080 | 1.591% | PASS，max abs 0.667 |
| W2 | 严格交替压力对照 | 1105.838 | 1113.280 | 1074.860–1121.920 | 1.343% | PASS |
| W4 | D/C 并发 | 896.928 | 898.320 | 886.180–912.040 | 0.765% | PASS，max abs 0.667 |
| W4 | 严格交替压力对照 | 1916.448 | 1918.660 | 1873.400–1945.200 | 0.959% | PASS |

严格交替会引入 cohort barrier 和流水气泡，因此它不是纯 D∥C 收益数字；尤其 W4 的
端到端比值超过 2×，不能解释成通信交叠突破理论上限。

## D∥C 服务窗口：理论收益与真实收益

定义：

```text
理论最大加速 = (Td + Tc) / max(Td, Tc) ≤ 2
真实窗口加速 = (Td + Tc) / (Td + Tc - overlap)
交叠实现度   = overlap / min(Td, Tc)
```

这个 2× 上限来自两段工作是否等长，与 W2/W4 本身无直接关系；rank 增多只可能改变
Td/Tc 的比例，使本 case 更接近或远离 2×。

| case | Td cycles | Tc cycles | overlap cycles | 理论最大 | 真实窗口加速 | 交叠实现度 |
|---|---:|---:|---:|---:|---:|---:|
| W2 K2 | 20,377 | 28,462 | 20,082 | 1.7159× | 1.6983× | 98.55% |
| W4 K2 | 35,583 | 44,792 | 35,353 | 1.7944× | 1.7852× | 99.35% |
| W2 K1 tail | 19,270 | 34,177 | 19,038 | 1.5638× | 1.5533× | 98.80% |
| W2 K4 tail | 23,167 | 37,146 | 22,938 | 1.6237× | 1.6137× | 99.01% |
| W4 K1 tail | 21,118 | 35,367 | 20,937 | 1.5971× | 1.5890× | 99.14% |
| W4 K4 tail | 44,045 | 59,561 | 43,824 | 1.7395× | 1.7331× | 99.50% |
| W4 K6 tail | 53,296 | 66,077 | 53,089 | 1.8066× | 1.8009× | 99.61% |
| W4 K8 tail | 64,194 | 77,170 | 64,003 | 1.8319× | 1.8273× | 99.70% |

## 鲁棒性覆盖

- 非整齐尾行：tokens=17, H=192, I=320。
- W2：K1、K4；W4：K1、K4、K6、K8；均连续调用 3 次（首轮 warmup）通过。
- 最大相对/绝对误差均在 BF16 累加预期范围；K8 最大绝对误差 0.403。
- 大行多 packet：tokens=256, H=16384, I=128, K2，32KiB/row；W2/W4 各连续 3 次通过。
- 大行 case 只对 1 个 token 做 CPU golden，并对全输出做 finite checksum；这是扩展性 smoke，不替代全量 golden。

大行结果：W2 makespan 约 7.17ms、最大相对误差 0.462%；W4 makespan 约
13.52ms、最大相对误差 0.460%。设备 trace 的 D/C 交叠实现度分别为 99.87% 与
99.96%。

## 当前结论

- 完整 D→GMM1→SwiGLU→GMM2→C 数据路径已经存在，结果不是由上下行带宽外推得到。
- 单个 MIX worker kernel 内，Dispatch/Combine AIV cohort 与 AIC/AIV 计算 cohort 独立推进；INC 的 24/24 D/C cohort 可任意时序并发。
- prepared API 可重复 enqueue；同一 SHMEM 队列跨 generation 复用已用 warmup+10 measure 验证。
- 当前证据支持 W2/W4、小/中 shape、非整齐 top-k 和 32KiB 多 packet 行；尚不能宣称完成 8GiB 全量 fusion sweep、常驻跨请求 server 或 90% 独立链路带宽 gate。

## GMM2 slice 就绪即 Combine（后续增量）

对照 MegaMoE 的 tile-level signaling 和昇腾融合核的 GMM tile epilogue 后，移除了
worker 侧残留的整波 `GMM2 -> Combine` 屏障。当前按
`expert -> activation slice -> row` 的确定顺序消费 GMM2：每个 slice 的全部 AIC
完成后，该 slice 立即进入单 INC Combine；后续 slice 仍可继续计算。owner stream 的
packet 顺序与旧实现相同，因此不增加 credit 深度，也不引入 shape 特化参数。

稳态仍为 warmup=3、measure=10，时间为同一 iteration 所有 PE 的最大值：

| case | mean (us) | median (us) | CV | D/C 理论最大 | 真实窗口加速 | 交叠实现度 | 正确性 |
|---|---:|---:|---:|---:|---:|---:|---|
| W2, T32 H256 I512 K2 | 595.736 | 595.920 | 1.286% | 1.7329× | 1.7189× | 98.89% | PASS，全量 golden |
| W4, T32 H256 I512 K2 | 887.828 | 888.770 | 0.951% | 1.7933× | 1.7836× | 99.31% | PASS，全量 golden |

相对前一版同 shape 的 makespan mean，W2 从 604.480us 降至 595.736us（1.45%），
W4 从 896.928us 降至 887.828us（1.01%）。这不是参数遍历所得，而是协议流水边界
缩小后的结构性收益。

附加鲁棒性回归：

| case | measure | mean (us) | CV | 真实/理论 D/C 窗口 | 正确性 |
|---|---:|---:|---:|---:|---|
| W2, T17 H192 I320 K4 | 2 | 730.680 | 0.036% | 1.5887× / 1.5971× | PASS，全量 golden |
| W4, T17 H192 I320 K8 | 2 | 1488.760 | 0.282% | 1.8388× / 1.8437× | PASS，全量 golden |
| W2, T8 H16384 I128 K2 | 2 | 1536.920 | 0.731% | 1.3540× / 1.3576× | PASS，全量 golden |
| W4, T8 H16384 I128 K2 | 2 | 1684.220 | 0.500% | 1.4045× / 1.4081× | PASS，全量 golden |

最后两项每行 32KiB、跨两个 transport packet，证明 slice 流式发送没有破坏多 packet
排序、credit 或 INC 归约。测试结束后 `npu-smi info` 再次确认无残留 NPU 进程。

## Shape 自由度与第二轮 sweep

算子没有写死模型 FFN 形状：`token_count / hidden / intermediate / topk /
tokens_per_wave / activation_waves` 均来自运行时 plan。固定的是 910B 的底层 AIC tile
和 INC vector tile；尾块使用实际 shape/精确字节搬运。新增的
`run_inc_fusion_nb_sweep.sh` 会在每个 case 前验证 HCCS 映射和全机 NPU 空闲，并为每个
case 单独保存日志。

第二轮覆盖 W2/W4、H=192–16384、I=128–1024、K1/K2/K4/K8 共 10 个 shape，全部
PASS；CV 范围 0.331%–2.490%。完整表见 `sweep_round2.csv`。其中 D/C 交叠实现度随
中大 shape 提升到 99.47%–99.96%，真实窗口加速最高为 W4/H2048/K8 的 1.9552×。

## INC reduction tile ping-pong

大 hidden 的旧实现每 1024 个 BF16 元素执行一次 `PIPE_ALL`。现改成两套 UB bank，
通过独立 `MTE3_MTE2` 生命周期事件重叠下一 tile 搬入、当前 tile FP32 归约和上一 tile
回写；单 tile packet 继续走原来的低开销路径，避免为了大行优化拖慢小行。

全量 golden 回归均通过。5 次 measure 的代表结果：

| case | ping-pong mean | 改动前 mean | 变化 | CV | 正确性 |
|---|---:|---:|---:|---:|---|
| W2, T8 H16384 I128 K2 | 1534.360us | 1536.920us | -0.17% | 0.519% | PASS |
| W4, T8 H16384 I128 K2 | 1673.104us | 1684.220us | -0.66% | 0.611% | PASS |
| W2, T32 H256 I512 K2 | 598.776us | 595.736us | +0.51% | 1.521% | PASS，噪声范围 |
| W4, T32 H256 I512 K2 | 894.276us | 887.828us | +0.73% | 1.063% | PASS，噪声范围 |

中等 shape 的变化低于此前 CV/noise，不判定为回退；大行获得小幅但稳定的结构性收益。
后续若继续扩大 reduction tile，需要先证明不会增加 packet 首包延迟。

## 常驻 INC descriptor-ring service

新增独立于一次请求路径的常驻服务：INC 的 vector kernel 只启动一次，随后轮询本地
descriptor ring。host 对每个 descriptor 先发布 metadata，最后单独发布 64-bit ticket；
48 个 AIV 各自写一条独占 cacheline，只有 lane0 观察到全部 Dispatch/Combine lane
完成后才发布 descriptor completion。ring slot 在 `complete == previous ticket` 后才能
复用，满载时 API 返回 `INC_FUSION_BUSY`，不会覆盖未完成请求。

一个重要的调度约束是：persistent kernel 占满 INC 的 48 个 AIV，启动后不能再在 INC
上发起 device-side barrier。E2E 因此只在启动前做最后一次全 rank barrier；后续请求
完全依赖 generation/ticket/credit，可接受 worker 的任意到达时序。

验证使用最小 ring=2，覆盖 W2/W4、K2/K8、32KiB 多 packet 行和多次 wrap：

| case | measure | ring wrap 下界 | mean (us) | CV | 正确性 |
|---|---:|---:|---:|---:|---|
| W2 T32 H256 I512 K2 | 50 | 26 | 637.741 | 1.563% | PASS |
| W4 T32 H256 I512 K2 | 50 | 26 | 913.282 | 0.995% | PASS |
| W4 T17 H192 I320 K8 | 10 | 6 | 1549.560 | 0.960% | PASS |
| W2 T8 H16384 I128 K2 | 10 | 6 | 1560.480 | 0.792% | PASS |
| W4 T8 H16384 I128 K2 | 10 | 6 | 1690.823 | 0.614% | PASS |

表中时间包含当前 E2E 的同步 D2H completion query，不是纯设备服务时间；生产接入应通过
`inc_fusion_persistent_service_device_control()` 把 completion 接到 device event/框架调度，
避免把 host polling 开销计入算子。完整数据见 `persistent_service.csv`。

一次请求的旧入口仍保留，并在常驻功能加入后重新通过 W2/W4 回归；新增路径没有改变
原 kernel ABI 的数据面语义。

## 原始临时日志

本轮原始日志仍位于执行机 `/tmp`：

- `/tmp/fusion-steady-ab.lTj48e`
- `/tmp/fusion-steady-w4-ab.3unISq`
- `/tmp/fusion-trace.XFOhEH`
- `/tmp/fusion-trace-w4.yRz1nj`
- `/tmp/fusion-topk-tail.igQyRm`
- `/tmp/fusion-multipacket-fixed.8qIN3h`
- `/tmp/fusion-stream-combine-w2.d7j7KR`
- `/tmp/fusion-stream-combine-w4.GlUITq`
- `/tmp/fusion-stream-tail-w2-k4.XtbU1a`
- `/tmp/fusion-stream-tail-w4-k8.2esCG7`
- `/tmp/fusion-stream-packet-w2.gYmXye`
- `/tmp/fusion-stream-packet-w4.ijQGo8`
- `/tmp/inc-fusion-next-sweep-20260809`
- `/tmp/inc-fusion-reduce-pingpong-20260809`
- `/tmp/inc-fusion-pingpong-fastpath-20260809`
- `/tmp/inc-fusion-persistent-w2-v2-20260809`
- `/tmp/inc-fusion-persistent-matrix-20260809`
- `/tmp/inc-fusion-persistent-stability50-20260809`
- `/tmp/inc-fusion-fallback-after-service-20260809`

上述目录不是长期归档；本文件中的表是本轮纳入仓库的不可覆盖摘要。
