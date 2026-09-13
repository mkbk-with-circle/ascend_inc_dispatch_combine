# 单 INC Dispatch / Combine 架构（2026-08-01）

> U/G/T、P/R/E 各阶段的逐条数据流、generation 可见性协议和双算子并发方式，详见
> `docs/inc/single_inc_dispatch_combine_pipeline.md`。

本页只描述当前单 INC 后端及其正确性、性能口径。

业务侧唯一入口是
`examples/inc/dispatch_combine/common/api/inc_dc_single_inc.hpp`；本页以下内容是
设备实现说明，不是额外的调用 API。

## 1. 拓扑与安全生命周期

逻辑 worker 为 `[0,W)`，唯一 INC 为 rank `W`。W2/W4/W8 是跨环境验收规模，
不是 kernel 或 ABI 的固定上限；物理 NPU 编号也不是协议的一部分。

每次真机运行必须从当前 hardware profile 和 live topology 解析逻辑 rank 到物理
NPU 的映射，并确认所选 worker→INC 链路满足该环境的等效性要求。`nb-borrow`
是两个 8 卡 HCCS 平面，因此当前只资格化同一平面内的 W2/W4；8W+1INC 无法
放进单个 8 卡平面，不应用跨平面结果冒充等效 W8。`yuanmingyu` 的星型映射和
历史 W8 数据仍保存在它自己的 hardware profile 中。

具体映射、roofline 和环境 gate 以
`docs/inc/report/ACTIVE_HW_PROFILE.md` 指向的 `single_inc_ENV_STATUS.md` 为准。
launcher 下发前必须检查目标平面无其他进程并持有进程锁；拓扑或空闲检查失败
时 fail closed，不能把任何环境的 Phy 列表写死到共享 API 或 kernel。

## 2. 单 INC dispatch

资格化入口为 `single_inc/dispatch/inc_dc_single_inc_stream_main.cpp`，kernel 为
`single_inc/dispatch/inc_dc_single_inc_stream_kernel.cpp`。

1. worker 将每个 source tile 仅上传一次到 INC staging；
2. plan 按 `(source tile, destination)` 编排，tile-ready 即可触发下行，同时下一个
   tile 继续上行；
3. 连续 source rows 走 zero-copy direct task；任意 route/topk 走 gather cohort 和
   task-private staging；
4. 每个 destination 只等待自己的 TX lanes，独立收到完成 generation；INC 另有
   finish coordinator 提供完整算子完成时间；
5. destination 逐字节验证 source rank、row 和 payload。

generic gather AIV 在发布 completion 前执行 payload 可见性操作。当前公开 MTE
后端每 AIV 只有一个安全 NBI 同步槽，因此 direct 与 generic 都强制每 lane 单
outstanding；不同 AIV 仍完全并发。这个约束修复了大 generic packet 在
512/1536-byte vector 边界的间歇损坏。

所有 task、route、tile、packet、topk 和 workspace 均由运行时 plan 推导。
direct task 不分配 gather staging；generic task 使用紧凑 slot。H2D/D2H host copy
按 64 MiB 分块，layout 使用 checked 64-bit offset，因此协议不限制 16 GiB 或更大
逻辑消息，实际容量只受 allocator/OOM 约束。

## 3. 单 INC combine

入口为 `inc_dc_combine_launcher.cpp`，kernel 为
`inc_dc_combine_kernel.cpp`。

- result CSR 保存每个 contribution 的 source、ordinal、weight、ingress slot、
  home owner、destination rank/row；transport placement 与数值身份分离。
- worker producer 按 owner/source 打包 staging，并以逐行 ordered
  `putmem_signal` 形成细粒度 wavefront；16 producer lanes 是本机 W8 最优点。
- 48 个 owner AIV 独占 result rows，无跨 AIV 浮点原子。FP16 输入在 UB 中转为
  FP32 累加，最终 round 回 FP16；K1 identity 可直接 copy。
- 1536-element wide UB tile 占用 21 KiB（3 个 FP16 row + 2 个 FP32 row），低于
  24 KiB AIV UB，将 vector event/barrier 次数降到原来的约三分之一。容量谓词
  不满足时回退到 512-element tile。
- `topk<=64` 使用 owner-local bitmap；更大 topk 使用 per-contribution generation
  marker，不存在 64-bit shift 上限。重复 ordinal、stale generation、坏 slot、
  不完整 contribution 都 fail closed。
- reduce 结果按 CSR 的动态 destination rank/row 返回正确 worker，并在所有 rank
  做完整数值 oracle。

INC 显存还实现了可选的 result-private arrival counter、head-tile/tail-chunk 和
split reducer/TX cohort，用于不同平台调优；本机上共享原子 counter 和批量 TX
更慢，因此默认保留逐行 generation stream。所有实验路径均可显式关闭，不改变
稳定路径语义。

## 4. 任意时序并发

`run_single_inc_stream_dyn_overlap_case.sh` 同时启动上述两个生产后端的独立
SHMEM session。两个外部 gate 使用绝对 `CLOCK_MONOTONIC` deadline，排除两套
session 初始化差异；支持 `dispatch_first`、`combine_first` 和任意 `DELAY_US`。

并发时 AIV 由共享 hardware/topology policy 固定生成，不绑定物理 core-id，
也不读取 K、bytes、token/result 数或 route。当前 48-AIV 环境的 INC 分区是
Dispatch 16 + Combine 32；worker W2/W4/W8 的 D/C 分别为 `8/24`、`4/16`、
`2/12`，均不超过 live AIV 的一半。两个后端拥有独立 descriptor、generation、
staging、completion 和 SHMEM session，不会互相覆盖或等待对方协议。

## 5. 正式测量口径

启动、session 建立、内存初始化、JIT、host 校验均在计时窗外。每个 session 先
完成 rank rendezvous，再在 device 两阶段 gate 或 calibrated persistent deadline
之后计时。禁止用 host 最早提交到最晚结束，也禁止把各 rank 时间相加。

```text
dispatch_bytes = sum(unique(token, destination_rank)) * hidden_bytes
combine_bytes  = contribution_count * hidden * element_bytes
makespan_us    = max(rank_device_protocol_us)
useful_GB/s    = operator_bytes / makespan_us / 1000
formal_result  = min(useful_GB/s over repeated samples)
```

dispatch 只以按 `(token,destination rank)` 去重后的真实下行字节为分子；combine
只以每个 expert-instance 独立上行的 contributions 为分子。
正式 sweep 要求所有 rank 正确、统一 timing source，以及
`startup_in_timing=setup_in_timing=verify_in_timing=0`，否则 fail closed。

## 6. 一键入口

| 目的 | 脚本 |
|---|---|
| 正式 W2/W4/W8 体积与 top-k sweep | `examples/inc/dispatch_combine/scripts/single_inc/run_single_inc_operator_sweep.sh` |
| 完全随机 token-plan campaign | `examples/inc/dispatch_combine/scripts/random_token_plan/run_single_inc_os_random_campaign.py` |
| 可复现随机 token-plan 矩阵 | `examples/inc/dispatch_combine/scripts/random_token_plan/run_single_inc_nb_random_sweep.py` |

实机数字和复现命令见
`docs/inc/report/single_inc_LIVE_STATUS.md`。
