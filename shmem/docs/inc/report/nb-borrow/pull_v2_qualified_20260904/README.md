# nb-borrow 单 INC Pull V2 资格报告（2026-09-04）

本目录只记录 `nb-borrow`（Ascend 910B2C）上的 Pull-Dispatch V2 与
Pull-Combine V2；不覆盖 yuanmingyu、Push V1 或历史实验。正式性能固定使用
同一 HCCS 平面的 NPU 0--4（0--3 worker，4 为 INC）。运行前确认 16 张卡空闲。

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
