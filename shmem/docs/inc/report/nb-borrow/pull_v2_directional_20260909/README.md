# nb-borrow Pull V2 单方向有效带宽（2026-09-09）

本报告采用当前约定的唯一性能口径：

- Dispatch：`fan-out 下行 hidden 总字节 / 完整 Dispatch 时间`；
- Combine：`参与归约的上行 partial 总字节 / 完整 Combine 时间`；
- READY、解析、重整、归约、completion 与 ACK 均计入时间；metadata/control
  字节不计入有效数据量。

旧报告中的 `GET+PUT logical bytes / device makespan` 仅是历史诊断口径，不能用于
本报告的 gate 判定。

## 硬件与物理参考

`nb-borrow` 有 16 张 Ascend 910B2C，分成 NPU 0--7 与 NPU 8--15 两个独立
HCCS 平面。正式 W4 使用同一平面的 NPU 0--3 作为 worker、NPU 4 作为 INC；
W2 使用 NPU 0--1 作为 worker、NPU 2 作为 INC。每条 worker--INC peer link 的
nominal raw 为 224 Gbit/s（约 28 GB/s）：W2/W4 raw 聚合分别为 56/112 GB/s，
其 92% raw gate 分别为 51.52/103.04 GB/s。

本机 put-only 实测单向参考为：

| 规模 | all→INC min | INC→all min |
|---|---:|---:|
| W2 | 42.714 | 42.804 |
| W4 | 85.445 | 83.258 |

此外，W4、128 MiB/worker、3 lanes/worker 的 INC pull-only transport probe 为
mean 83.130 GB/s、conservative 63.946 GB/s。由于 W4 的 103.04 GB/s raw gate
高于当前 SHMEM transport 自身的实测屋顶，完整 pull/parse/reduce/push 算子在
当前传输层上无法达到该 raw gate；下表仍保留它以明确展示差距。

## 正式结果

固定 H=8192、128 MiB/worker，3 warmup + 10 measure。全部样本数值正确，且
guard、completion/ACK 和协议状态均通过。

| 算子 | 规模/路由 | min GB/s | mean GB/s | CV | raw 92% gate | 相对实测单向 roof |
|---|---|---:|---:|---:|---:|---:|
| Dispatch | W2, top-k2/GPU2 | 37.618 | 37.849 | 0.509% | 51.52 | 87.9% |
| Combine | W2, top-k2/GPU2 | 37.800 | 37.858 | 0.084% | 51.52 | 88.5% |
| Dispatch | W4, top-k2/GPU2 | 69.117 | 69.705 | 0.533% | 103.04 | 83.0% |
| Combine | W4, top-k2/GPU2 | 71.744 | 71.946 | 0.131% | 103.04 | 84.0% |
| Dispatch | W4, expert-k4/GPU2 | 68.374 | 68.844 | 0.376% | 103.04 | 82.1% |
| Combine | W4, expert-k4/GPU2 | 71.676 | 71.871 | 0.159% | 103.04 | 83.9% |
| Dispatch | W4, expert-k4/GPU4 | 76.632 | 76.970 | 0.215% | 103.04 | 92.0% |
| Combine | W4, expert-k4/GPU4 | 72.622 | 72.841 | 0.155% | 103.04 | 85.0% |
| Dispatch | W4, expert-k8/GPU4 | 75.369 | 75.555 | 0.145% | 103.04 | 90.5% |
| Combine | W4, expert-k8/GPU4 | 72.420 | 72.790 | 0.186% | 103.04 | 84.8% |

Dispatch 的 roof 使用同规模 INC→all put-only min；Combine 使用 all→INC
put-only min。这里的效率只用于定位软件开销，不替代 raw gate。

## 本轮实现变化

- 重复 expert 落在同一 GPU 时按 unique destination 只发送一份 hidden；
- Dispatch 的 fanout4 使用 8 KiB 单读多发流水，任意 expert-k4/k8 元数据仍完整保留；
- Combine 增加固定四贡献者的 6 KiB 双输出流水；
- Combine top-k2 缓存 source READY、payload offset 与 accumulator 地址；
- Combine 的各 source ACK / owner completion 改为按 rank 并行发布；
- 正式 JSON 新增 `downlink_*` / `uplink_*` 字段；旧 `logical_*` 字段仅为兼容。

所有改动保持协议不变：worker 只发布 READY/Notice，数据只能经过
`worker → INC → worker`；Dispatch 每个 source hidden 只被 INC 拉取一份，失败波次
不发布成功 completion/ACK。

