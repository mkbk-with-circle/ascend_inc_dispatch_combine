# nb-borrow Pull V2 单方向有效带宽（2026-09-09）

> 本页是旧紧凑布局的历史记录，保留当时数据和门槛。当前源分区讲解已撤去旧raw百分比gate，
> 改用多打一实测峰值参照；当前协议与资格范围见[源分区报告](../ready_push_20260911/README.md)。

本报告采用当前约定的唯一性能口径：

- Dispatch：`fan-out 下行 hidden 总字节 / 完整 Dispatch 时间`；
- Combine：`参与归约的上行 partial 总字节 / 完整 Combine 时间`；
- READY、解析、重整、归约、completion 与 ACK 均计入时间；metadata/control
  字节不计入有效数据量。

性能目标仍为 W2/W4 的 51.52/103.04 GB/s；本报告所有 case 尚未达到目标。

## 硬件与物理参考

`nb-borrow` 有 16 张 Ascend 910B2C，分成 NPU 0--7 与 NPU 8--15 两个独立
HCCS 平面。正式 W4 使用同一平面的 NPU 0--3 作为 worker、NPU 4 作为 INC；
本轮 W2 使用 NPU 8--9 作为 worker、NPU 10 作为 INC；后续 Combine W2 复测
使用 NPU 0--1 与 INC NPU 2。每条 worker--INC peer link 的
nominal raw 为 224 Gbit/s（约 28 GB/s）：W2/W4 raw 聚合分别为 56/112 GB/s，
其 92% raw gate 分别为 51.52/103.04 GB/s。

本机 put-only 实测单向参考为：

| 规模 | all→INC min | INC→all min |
|---|---:|---:|
| W2 | 42.714 | 42.804 |
| W4 | 85.445 | 83.258 |

此外，W4、128 MiB/worker、3 lanes/worker 的 INC pull-only transport probe 为
mean 83.130 GB/s、conservative 63.946 GB/s。这些是现有 benchmark 的实测参照，
不是其他调度或实现的严格上界，不能据此证明 raw gate 不可能达到，也不用于降低
目标。历史二进制 probe 的计时口径仍需独立审计，不能作为正式算子验收依据。

## 正式结果

固定 H=8192、128 MiB/worker，3 warmup + 10 measure。全部样本数值正确，且
guard、completion/ACK 和协议状态均通过。

| 算子 | 规模/路由 | min GB/s | mean GB/s | CV | raw 92% gate | 相对实测单向 roof |
|---|---|---:|---:|---:|---:|---:|
| Dispatch | W2, top-k2/GPU2 | 37.618 | 37.849 | 0.509% | 51.52 | 87.9% |
| Combine | W2, top-k2/GPU2 | 39.289 | 39.429 | 0.197% | 51.52 | 92.0% |
| Dispatch | W4, top-k2/GPU2 | 69.117 | 69.705 | 0.533% | 103.04 | 83.0% |
| Combine | W4, top-k2/GPU2 | 77.468 | 77.730 | 0.190% | 103.04 | 90.7% |
| Dispatch | W4, expert-k4/GPU2 | 68.374 | 68.844 | 0.376% | 103.04 | 82.1% |
| Combine | W4, expert-k4/GPU2 | 77.536 | 77.757 | 0.140% | 103.04 | 90.7% |
| Dispatch | W4, expert-k4/GPU4 | 76.632 | 76.970 | 0.215% | 103.04 | 92.0% |
| Combine | W4, expert-k4/GPU4 | 78.405 | 78.600 | 0.154% | 103.04 | 91.8% |
| Dispatch | W4, expert-k8/GPU4 | 75.369 | 75.555 | 0.145% | 103.04 | 90.5% |
| Combine | W4, expert-k8/GPU4 | 78.192 | 78.616 | 0.255% | 103.04 | 91.5% |

Dispatch 的 roof 使用同规模 INC→all put-only min；Combine 使用 all→INC
put-only min。这里的效率只用于定位软件开销，不替代 raw gate。

## 本轮实现变化

- 重复 expert 落在同一 GPU 时按 unique destination 只发送一份 hidden；
- Dispatch 的 fanout4 使用 8 KiB 单读多发流水，任意 expert-k4/k8 元数据仍完整保留；
- Combine 增加固定四贡献者的 6 KiB 双输出流水；
- Combine top-k2 缓存 source READY、payload offset 与 accumulator 地址；
- Combine 的各 source ACK / owner completion 改为按 rank 并行发布；
- 固定 K2/K4 将同等强度的 sealed-plan 校验融合到每个 accumulator 的首次
  数据处理，避免校验预扫描和归约路径重复读取整张 plan；
- 正式 JSON 新增 `downlink_*` / `uplink_*` 字段；旧 `logical_*` 字段仅为兼容。

所有改动保持协议不变：worker 只发布 READY/Notice，数据只能经过
`worker → INC → worker`；Dispatch 每个 source hidden 只被 INC 拉取一份，失败波次
不发布成功 completion/ACK。

固定路由 Combine 相对 pull-only mean 83.130 GB/s 的 mean 效率为 93.5%--94.6%；
W2/W4 相对各自 put-only min 也均超过 90%。

## 设备安全回归与复现

最新修补增加融合校验的 journal token 上界检查，并让完成通知在 AIV 少于
worker 时通过跨步循环覆盖所有 rank。修补后 K4/GPU2 和 K4/GPU4 重新通过
3 warmup + 10 measure，上表这两行已更新；其余行来自此前版本，尚未全部重跑。

设备安全回归通过 14 组、236 个 wave：固定 K2/K4 各自的非法链头、链环、
token 越界、owner 越界各重复 3 次；K2/K4 各 100 次 ring 复用；1 AIV 服务
4 worker、空输入和短尾块。错误注入检查 ABORTED、失败 ACK/completion
及 publication；所有 case 检查 guard。有限矩阵不等同于任意输入的证明。

```bash
python3 examples/inc/dispatch_combine/single_inc/pull_combine/tests/pull_v2_combine_safety.py \
  --build-dir /tmp/inc-k4-build-20260909 \
  --output /tmp/combine-safety-new-run --first-npu 0
```

输出目录必须不存在，脚本检查目标设备空闲，对每组设置超时并只清理自身进程。
