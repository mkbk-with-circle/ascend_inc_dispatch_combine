# nb-borrow 单 INC Pull V2 资格报告（2026-09-04）

本目录只记录 `nb-borrow`（Ascend 910B2C）上的 Pull-Dispatch V2 与
Pull-Combine V2；不包含旧协议或其他集群的历史实验。正式性能固定使用
同一 HCCS 平面的 NPU 0--4（0--3 worker，4 为 INC）。运行前确认 16 张卡空闲。

## 硬件拓扑、Rank 映射与 Gate 依据

### 双 HCCS 平面

本机共有 16 张物理 NPU，分成两个彼此独立的 8-card HCCS 平面：

```text
HCCS 平面 A：NPU 0--7    平面内任意两卡均为 HCCS
HCCS 平面 B：NPU 8--15   平面内任意两卡均为 HCCS

跨平面 0--7 ↔ 8--15：不是 HCCS；根据位置走 PIX / PHB / SYS
```

每张卡在本平面内连接另外 7 张卡，每条 peer link 的 live speed 为
224 Gbit/s，即约 28 GB/s raw 单向。平面内是 full mesh，不代表一个 rank 可以把
7×28 GB/s 全部用于本测试：Single-INC 的有效物理边是每个 Worker 与唯一 INC
之间的 `W` 条链路。

跨平面关系仅用于说明为什么不测 W8：典型配对 `i ↔ i+8` 为 PIX，其他跨平面
路径还可能经过 PHB 或 SYS。当前所有正式样本都未使用跨平面链路。

### 逻辑 Rank 到物理 NPU

协议 world size 恒为 `W+1`：逻辑 rank `[0,W)` 是 Worker，rank `W` 是唯一 INC。
测试程序使用连续映射 `physical_npu = first_npu + rank`，本轮 `first_npu=0`：

| 规模 | World | Worker Rank → 物理 NPU | INC Rank → 物理 NPU | Worker↔INC 关系 |
|---|---:|---|---|---|
| W2 | 3 ranks | `0→NPU0`, `1→NPU1` | `2→NPU2` | 2 条均为 HCCS |
| W4 | 5 ranks | `0→NPU0` … `3→NPU3` | `4→NPU4` | 4 条均为 HCCS |

W4 的直观布局如下；NPU5--7 在该平面内空闲，平面 B 全部不参与：

```text
Worker0/NPU0 ─HCCS─┐
Worker1/NPU1 ─HCCS─┤
Worker2/NPU2 ─HCCS─┼─ INC rank4 / NPU4
Worker3/NPU3 ─HCCS─┘

NPU5--7：unused                 NPU8--15：另一 HCCS 平面，unused
```

Worker 之间虽然也由 HCCS full mesh 相连，但协议禁止 Worker 直达 Worker；payload
必须沿 `Worker → INC → Worker` 的唯一路径。8 Worker + 1 INC 需要 9 张卡，无法
放入任一 8-card 平面，所以本机 W8 没有等效链路配置，明确不进入 gate。

### Baseline 与 92% Gate

每个 Worker 向 INC 提供一条独立的 28 GB/s raw peer link，因此同方向 nominal raw
聚合为：

```text
W2 raw = 2 × 28 = 56 GB/s    → gate = 56 × 0.92 = 51.52 GB/s
W4 raw = 4 × 28 = 112 GB/s   → gate = 112 × 0.92 = 103.04 GB/s
```

同时保留单向 put-only 实测，用于描述 SHMEM transport 的物理腿，而不用于下调
正式完整算子 gate：

| 规模 | all→INC min | INC→all min | nominal raw | 完整算子 gate |
|---|---:|---:|---:|---:|
| W2 | 42.714 GB/s | 42.804 GB/s | 56 GB/s | 51.52 GB/s |
| W4 | 85.445 GB/s | 83.258 GB/s | 112 GB/s | 103.04 GB/s |

这里有两个不同口径：

1. put-only 是单一物理方向的 payload / 单腿时间；
2. Pull V2 算子带宽是实际 logical GET+PUT bytes / 从 READY 到全部 completion/ACK
   的完整 device makespan。

由于 INC 会把 GET、解析、重整和 PUT 做流水交叠，完整算子分子同时累计两个方向的
logical bytes；因此算子 GB/s 可以高于某一条单向 put-only 数字。这不表示单条
224-Gbit/s 链路被突破，也不能把 `measured/112` 直接解释成电气链路利用率。92%
gate 是本轮对称 128 MiB 完整算子的 nominal roof contract；小消息、hotspot、
ragged 或非对称 workload 只要求正确、稳定并单独报告性能，不套用该 gate。

## 协议闭环

- Dispatch：worker 准备完整 source slot 后只发一个 READY；INC 主动 GET
  header/metadata/hidden，在线解析路由，每个 source hidden 只拉一份，再向每个
  unique destination fan-out。
- Combine：worker 本地归约后只发一个 64B Notice；128B READY 留在 worker 注册
  ring，INC 收到 Notice 后主动 GET READY 和 partial，归约完成后选择性 PUT 回 owner。
- Dispatch journal 在 seal 时生成 Combine 的不可变 pull index；Combine 不在关键
  路径重复建链。
- INC 的实时普通 AIV 数动态二分给 Dispatch/Combine：`floor(live_aiv / 2)`；没有
  写死 24 或某一 SKU 的 AIV 总数。
- publication-last、generation/sequence/wave/ring-slot、cookie、digest、capacity、
  guard 和有限 timeout 全部 fail-closed。

## 正式 128 MiB/worker gate

工作负载均为 top-k=2 balanced，3 warmup + 10 measure，所有样本全量正确。
带宽为完整算子 logical bytes / READY-to-completion device makespan；metadata/control
字节不计入分子，但时间完整计入分母。

| 算子 | 规模 | 92% gate | min | mean | CV | 判定 |
|---|---:|---:|---:|---:|---:|---|
| Dispatch | W2 | 51.52 GB/s | 56.505 | 56.906 | 0.237% | PASS |
| Dispatch | W4 | 103.04 GB/s | 104.907 | 105.627 | 0.476% | PASS |
| Combine | W2 | 51.52 GB/s | 56.641 | 56.750 | 0.105% | PASS |
| Combine | W4 | 103.04 GB/s | 107.254 | 107.663 | 0.194% | PASS |

W4 Dispatch 的主机完整调用计时最慢样本仍为 104.446 GB/s，高于 gate。纯 GET/relay
只用于解释物理上限，未用于降低 gate。

额外的 256 MiB/worker 单 wave 扩展测试也全量通过：W2 为 57.268 GB/s，W4 为
106.690 GB/s（协议口径）。它们只证明更大 resident message 的正确性和扩展性，
不替代上面的 128 MiB H11 gate。

## D+C 交叠：理论收益与真实收益

schema v2 先独占运行同规格 D-only/C-only，再运行并发 case：

```text
theoretical_speedup = (D_solo + C_solo) / max(D_solo, C_solo)
real_speedup        = (D_solo + C_solo) / concurrent_makespan
real_saved          = 1 - concurrent_makespan / (D_solo + C_solo)
```

并发时长之和计算的几何重叠率单独记录，不能冒充真实端到端收益。

| 规模 | 数据 | D solo | C solo | 理论上限 | 真实加速范围 | 真实节时范围 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 16 MiB/worker | 1.974 ms | 0.941 ms | 1.4768x | 1.1205x--1.1681x | 10.76%--14.39% |
| W4 | 128 MiB/worker | 15.196 ms | 7.468 ms | 1.4914x | 1.1290x--1.1446x | 11.43%--12.63% |

同时启动是两组中的最好点：W2 为 1.1681x / 14.39%，W4 为 1.1446x / 12.63%。
所有同时、D 提前 500 us、C 提前 500 us、固定 seed 随机 skew 均由 INC 设备 cycle
证明真实相交，且两边 oracle/guard/status 全通过。W4 并发时 D 膨胀约
1.28--1.31x、C 膨胀约 1.33--1.35x，这解释了真实收益低于理论值。

## 稳健性与正确性

已真机覆盖 W2/W4：

- 0 token、1-byte 请求、hidden=1537/2049 尾块；
- top-k1 round-robin、top-k2 balanced、top-k=all、重复目的 hotspot；
- token 数和 top-k 同时非对称的 ragged，固定 seed 7/131/65537；
- Combine asymmetric 与 READY skew；
- 两个 ring slot 连续复用；
- digest、非法 assignment、缺失 READY、READY identity、busy journal 五类故障。

另外，W2/W4 的 Dispatch 与 Combine 各完成 100 个连续 device wave；generation 到
1100、wave 到 110、两个 ring slot 交替复用，四组均 100/100 正确，无 timeout、
guard 损坏或状态泄漏。小消息 soak 的延迟离群点不用于性能 gate。

未完整验证为 uniform 的 Dispatch 不再使用推测性的多 AIV general MTE relay。
它先将每个 source hidden 只 GET 一次到 INC，再按 destination 重整并执行一次对齐
bulk PUT。该安全路径明显慢于 uniform fast path，但保证任意合法长度、ragged 和
重复目的的正确性、边界与有限完成。

主机测试另覆盖 5000 个随机 packet、500 个 W2--W8 随机 layout/control plan、
三种 dtype、空 token、可变 top-k、重复 token ID、乱序 READY、越界和溢出拒绝。

## 原始数据（本地归档）

协作仓库只保留本报告中的汇总表、测试口径和复现命令；下列逐 PE JSONL 与日志保存在
实验机本地，不随 Git 上传：

- `raw/formal/`：四个正式 H11 case 的逐 PE JSONL。
- `raw/overlap/`：schema v2 的 solo baseline、四种并发时序及原始日志。
- `raw/robustness/`：tiny/odd/ragged/random/READY-skew 等功能矩阵。

结果目录只追加，不允许 runner 覆盖。交叠 runner 运行示例：

```bash
python3 examples/inc/dispatch_combine/single_inc/pull_combine/tests/\
pull_v2_overlap_qualification.py \
  --workers 4 --first-npu 0 --plane-size 8 --channels 3 \
  --payload-bytes 134217728 --random-cases 1 \
  --output-dir /tmp/pull-v2-overlap-new
```

`--plane-size` 和 NPU placement 只属于当前机器的 qualification guard；生产协议与
kernel 不包含 16 卡、8 卡平面或 24 AIV 常量。
