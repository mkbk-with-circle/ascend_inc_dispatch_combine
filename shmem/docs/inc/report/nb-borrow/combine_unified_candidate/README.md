# Combine 统一执行路径候选（nb，2026-09-09）

本候选将固定 K2、固定 K4 和通用归约合并为 `ReduceContributorTaskRange`，
移除 all-two/all-four 分支；K 只控制同一循环的迭代次数。
保持 READY/INC GET/reduce/owner PUT 协议、端侧同 GPU 本地归约、唯一 INC 路径。
所有 K 使用 2 输入 + 2 输出缓冲，每块 6 KiB。

目标仍为 W2/W4 的 51.52/103.04 GB/s；目前未达到。该候选未替换远程 main
`b37a0fd`，可以从本地 `codex/combine-pre-unified-b37a0fd` 回退。

## 同平面对照

H=8192，每 worker 128 MiB FP32 partial，24 AIV，同一 HCCS 平面 NPU 0–4，
INC=NPU4。每组 3 warmup + 10 measure。基线从 b37a0fd 的 kernel 源文件
使用相同编译选项重新构建，复用同一个 harness、SHMEM 与库依赖。交替测量
基线与候选，以下列出初版统一实现的两轮 mean，以区分抖动和稳定差异。

| 贡献数 | 基线 mean 两轮 GB/s | 统一版 mean 两轮 GB/s | 变化 |
|---|---|---|---|
| K2（expert K4/GPU2） | 77.607 / 77.610 | 77.260 / 77.278 | 约 -0.4% |
| K4（expert K4/GPU4） | 78.546 / 78.650 | 78.740 / 78.748 | 约 +0.2% |

随后将每 tile 内动态取模改为游标递增与回绕，最终采用的统一 6 KiB 版本：

| 规模 / 贡献数 | min GB/s | mean GB/s | CV | raw gate |
|---|---:|---:|---:|---|
| W4 / K2（expert K4/GPU2） | 76.990 | 77.169 | 0.144% | 未达 103.04 |
| W4 / K4（expert K4/GPU4） | 78.686 | 78.874 | 0.124% | 未达 103.04 |
| W2 / K2 | 38.939 | 39.061 | 0.154% | 未达 51.52 |

K2 的小幅回退尚未消除；不能宣称“无性能回退”或达到 gate。
4 KiB tile 的对照更慢，未保留。上表只统计归约上行字节 / 完整调用时间；
输入生成和结果验证在计时外。

## 正确性与可扩展性

同一循环处理 0..worker_count 个唯一 source：先检查链头、单调 next、精确链长度、
唯一 source、token 上界、行偏移及 owner，再访问 payload。全局 count 总和与
分 source/owner 偏移仍校验；每 token 的链验证与其后续 tile 共用缓存。
错误波次不发布成功 ACK/completion，输出缓冲在成功通知之前不可消费。

ABI 当前最多 128 worker；缓冲容量和 source 位图按此边界检查，不再有 W<=8
或 W=4 才能使用的算法分支。真机仅验证 W2/W4，不能据此声称已验证128卡。

安全脚本涵盖 K2/K4 的链头、链环、token/owner 越界及最后一个 owner 错误；
各100次 ring 复用；单 AIV 服务4 worker；空输入、短尾；K1、K3、随机混合 K
与非对称路由。混合 K 的 FP32 输入含正负数和不同数量级，使用 FP64 oracle
及基于贡献数和绝对值总和的舍入误差界，不以宽松固定误差掩盖错行。
最终配置共20组、262个wave全部通过；错误注入按预期失败验收，不计性能样本。

```bash
python3 examples/inc/dispatch_combine/single_inc/pull_combine/tests/pull_v2_combine_safety.py \
  --build-dir /tmp/inc-k4-build-20260909 --output /tmp/unified-safety-new
```

原始数据：`/tmp/inc-unified-ab-same-plane`、`/tmp/inc-unified-no-modulo`、
`/tmp/inc-unified-w2-final`、`/tmp/inc-unified-safety-final`。
这些实验数据与前一版本的单向结果均保留，未上传原始逐 rank 日志。
