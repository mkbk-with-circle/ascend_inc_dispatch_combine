# Pull V2 交叠优化与非对称压力报告（nb-borrow，2026-09-05）

本报告在 [`pull_v2_qualified_20260904`](../pull_v2_qualified_20260904/README.md)
的正式 W2/W4 gate 之后继续验证两个问题：D+C 交叠收益为什么低，以及多数据量、
多非对称程度下是否稳定。机器仍为 nb-borrow 910B2C；性能结果独占 NPU0--4，
功能矩阵每 case 前后检查整机无其他 NPU 进程。

## 1. 交叠瓶颈

W4 128 MiB、top-k2 的设备 timeline 表明，损失来自数据面而不是控制面：

| 阶段 | solo | 同时运行 | 膨胀 |
|---|---:|---:|---:|
| Dispatch pre-relay | 0.593 ms | 0.601 ms | 1.014x |
| Dispatch metadata 完成 | 7.568 ms | 7.806 ms | 1.032x |
| Dispatch relay | 14.559 ms | 19.155 ms | 1.316x |
| Combine plan | 0.723 ms | 0.695 ms | 0.961x |
| Combine data | 6.745 ms | 9.358 ms | 1.387x |

Dispatch 的物理流量为 512 MiB GET + 1 GiB PUT，Combine 为 512 MiB GET +
256 MiB PUT。两者同时运行时每个方向的总需求约 106--107 GB/s，而本机 W4
SHMEM 单向实测屋顶约 83.7 GB/s。因此至少约 25% 的数据段膨胀是共享链路/transport
credit 的物理结果；当前约 30%--35% 膨胀中，可由调度减少的部分有限。

## 2. AIV/transport lane governor

Solo 始终使用动态半 AIV（本机 24）；只有已知 D+C 并发时改变 Combine active
AIV，Dispatch 保持 qualified channels=3。首次 W4 同时启动 sweep：

| C active AIV | concurrent union | 真实加速 | 真实节时 | C inflation |
|---:|---:|---:|---:|---:|
| 8 | 20.114 ms | 1.126x | 11.17% | 2.395x |
| 12 | 19.892 ms | 1.142x | 12.47% | 2.011x |
| 16 | 19.789 ms | 1.152x | 13.19% | 1.669x |
| 20 | **19.669 ms** | **1.157x** | **13.60%** | 1.478x |
| 24 | 19.758 ms | 1.150x | 13.06% | 1.350x |

重复与错峰测试确认 W4 C=20 的改进虽小但稳定：

| 时序 | C=20 真实加速 | 真实节时 |
|---|---:|---:|
| 同时启动 | 1.1541x | 13.35% |
| Dispatch 提前 500 us | 1.1515x | 13.16% |
| Combine 提前 500 us | 1.1368x | 12.04% |

W2 128 MiB 中，C=22 的 union=19.006 ms、1.1269x/11.26%；C=24 的
union=19.133 ms、1.1203x/10.73%。因此当前可移植的 qualification 建议是：

```text
solo Combine active AIV   = floor(live_aiv / 2)
overlap Combine active AIV = floor(live_aiv / 2) - worker_count
```

该策略未设为生产默认；需要在任何其他硬件/拓扑上重新资格化。把 Dispatch channels 从3
降到2会令 W4 union 增至20.5--20.7 ms，已明确拒绝。

## 3. 单 session 对照

新增 qualification-only same-session probe：每个 PE 只初始化一次 SHMEM context，
INC 用两个 stream back-to-back 提交 D/C，仍各 launch 动态半 AIV。平面 A 的严格
同二进制结果：

| 项 | 时间 |
|---|---:|
| D solo | 15.342 ms |
| C solo | 7.451 ms |
| both 中 D | 19.790 ms |
| both 中 C | 10.063 ms |
| concurrent union | 19.790 ms |

真实加速约1.152x、节时约13.18%，与 dual-session 24-AIV 结果基本一致。因此双
session 不是主要损失来源，生产 resident session 仍值得采用，但不能单独解决共享
transport roof。

## 4. 压力矩阵

新增 `tests/pull_v2_dispatch_matrix.py`。完整默认矩阵为1200 case，不遍历 AIV、tile
或其他调优参数。本轮实际保留并执行：

- 主矩阵36 case：W2/W4 × 4 KiB/1 MiB/128 MiB × top-k2/hotspot ×
  token skew 0/50/100% × READY skew 1 ms；
- ragged矩阵24 case：W2/W4 × 1 MiB/64 MiB × token skew 0/50/100% ×
  READY skew 0/1 ms；
- 64 KiB、75% token skew 的 W2/W4 两个边界回归。

最终保留矩阵 **62/62 PASS**，无 timeout、进程异常、silent corruption 或 guard
损坏。结合上一报告，覆盖的数据量为0、1 byte、4 KiB、64 KiB、1 MiB、16 MiB、
64 MiB、128 MiB、256 MiB；token skew覆盖0/25/50/75/100%，READY skew覆盖
0/100/1000 us。

### 4.1 带宽随非对称程度变化

下表为 READY skew=1 ms 的协议带宽，三列依次为 token skew 0/50/100%：

| 规模/数据量 | top-k2 balanced | hotspot |
|---|---|---|
| W2 1 MiB | 25.67 / 21.54 / 0.41 | 0.51 / 0.52 / 0.49 |
| W2 128 MiB | 57.12 / 56.28 / 53.16* | 0.50 / 0.50 / 0.52 |
| W4 1 MiB | 40.94 / 28.14 / 0.41 | 0.52 / 0.51 / 0.49 |
| W4 128 MiB | 106.34 / 80.55 / 37.99* | 0.49 / 0.50 / 0.53 |

`*` 100% skew 数字来自空 source uniform 修复后的定点复测。修复前空 source 被错误
分类为 nonuniform，只有约0.4 GB/s；现在 W2/W4 分别达到53.16/37.99 GB/s。

Ragged 安全路径在不同 size/skew/READY 时序下约0.37--0.47 GB/s。它保证每个 source
只 GET 一份、INC 按 destination 重整后一次 bulk PUT；目前优先正确性，性能仍是
明确开放项。

## 5. 本轮发现并修复的问题

1. 5-row metadata 的最后8B未发布：`FlushRange` 未对非对齐 slice 向下/向上覆盖
   真实cacheline。已改为 align-down(begin)/align-up(end)。
2. 空 source 被判为 nonuniform：改为 vacuously uniform，并在所有 fixed relay
   入口先检查 `token_count==0`。
3. W4 空 source 的 sideband 等待：prefix coordinator 对空 source 的全部 parser
   chunk 幂等发布 `Pass2Ready`。
4. same-session probe 的 Dispatch Validate stride 接错：元素 capacity 改为字节 stride。

所有修复后，正式对称回归仍为 W2 56.82、W4 105.10 GB/s，高于原 gate。

## 6. 原始数据（本地归档）

协作仓库只保留本报告中的汇总数据和复现口径；下列逐 case JSON、PE 日志和中间
summary 保存在实验机本地，不随 Git 上传：

- `raw/stress/main36/`、`raw/stress/ragged24/`：逐 case JSON、PE 日志和总 summary；
- `raw/overlap/combine_aiv_sweep/`：C active AIV 8/12/16/20/24；
- `raw/overlap/w4_c20_schedules/`：C=20 三种时序；
- `raw/overlap/w2_c22/`、`w2_c24/`：W2 governor 对照；
- `raw/overlap/same_session_*`：同 session D/C/both；
- `raw/asym/`：100% token skew 修复后样本。

独立 mixed-roof 探针两版均未能复现生产 kernel 的流水效率，已删除且未纳入报告。
