# 单 INC native API 闭环（CANN 9.1，2026-08-04）

> 这是 2026-08-04 的内部 runtime 资格化快照。文中的 Framework/Easy request
> 名称保留当时的实验语境和原始性能证据，不是当前推荐调用路径。2026-08-14
> 起唯一公开入口为 `examples/inc/dispatch_combine/common/api/inc_dc_single_inc.hpp`，
> 业务流程为 `create -> dispatch -> compute -> combine -> destroy`。

## 结论

Dispatch 和 Combine 均已有真实 `inc_dc_fw_backend_ops_t` provider：
Framework/Easy request 直接下发已资格化的 stream Dispatch 和 DYN-CSR
Combine kernel，没有 mock、fork 或 benchmark 子进程。整链已由
INC 侧持久 proxy service 收集 worker generation mailbox 并启动真实
kernel，worker 不再调用 INC enqueue 或逐代 W+1 barrier。当前控制
transport 资格化范围为同机 IPv4 loopback。

## 实现链路

- Dispatch：canonical dense token plan → source compiler → collective merge →
  hardware-fixed workspace → worker upload/INC relay/worker output。
- Combine：Dispatch 的 route/assignment 表 → reverse LogicalPlan →
  single-INC execution compiler → mode-6 DYN-CSR workspace → worker producer/
  INC full reduction/remote result TX。
- 同 rank 多 expert：Dispatch 物理去重，Combine 保留所有逻辑
  expert-instance/ordinal/FP32 weight。
- 资源：只读 live AIV 和 worker topology、48 AIV 下 Dispatch
  INC/worker=`16/8,4,2`（W2/4/8），Combine=`32/24,16,12`；
  tokens/top-k/bytes/route skew 不参与 AIV 分配。
- 完成证据：ACL event 外，Dispatch 校验全活跃 AIV
  `StreamLaneStat.error`；Combine 校验全 producer/owner lane 的
  generation/fail code 及 global stats，禁止 event-complete 假成功。

## 真机正确性/复用

| 项 | case | 结果 |
|:---|:---|:---|
| Dispatch 100 代 | W2/T16/Hbytes16384/K2 | 3/3 rank，200/200 timing，逐字节 PASS |
| Combine 100 代 | W2/T4/Hbytes16384/K2 | 3/3 rank，200/200 timing，逐元素 PASS |
| Dispatch 扩展 | W4/K4，W8/K8 tiny | 5/5、9/9 rank PASS |
| Combine 扩展 | W4/K4，W8/K8 tiny | 5/5、9/9 rank PASS |
| Host product gate | API/ABI/lifecycle/numeric/portability + D/C plan compiler | PASS |

tiny W8 上 ACL event 偶发因计时分辨率返回 0；该样本不进入带宽
统计，但 request 必须通过所有设备协议 telemetry 和数值校验才计
correctness PASS。

## 带宽对照

以 registered/symmetric view（`INC_NATIVE_ZERO_COPY=1`）做 kernel 同口径比较。
`protocol_rank_us` 从 kernel telemetry 取值，排除 W+1 example 的 host
launch-armed barrier；`event_rank_us` 另行保留 API 完整区间。

| 算子 | 代表 case | native protocol | canonical 对照 | 结论 |
|:---|:---|:---|:---|:---|
| Dispatch | W2/T4096/Hbytes16384/K2，256MiB physical | 124.288–128.047 GB/s | 同 case 约 122–126 GB/s | 无回退（样本波动内） |
| Combine | W2/results5462/H8192/K2，178,978,816 B ingress | slowest-rank 141.09–143.79 GB/s | 143.09 GB/s | 无回退（约 98.6%–100.5%） |

Combine 带宽仍按项目现有口径
`physical_ingress_bytes / max(protocol_rank_us) / 1000`，不把 result egress
重复加到分子。Dispatch 分子为 global physical dispatch bytes。

caller-owned `aclrtMalloc` 兼容路径因本地 D2D staging 有端到端成本；
registered view 通过 `NativeDispatchWorkerBuffers()` /
`NativeCombineWorkerBuffers()` 跳过拷贝。两条路径共用同一协议与
数值验证。

## 本轮修复的真实问题

1. 多代 Dispatch 快 rank 在慢 rank 退出前清理 generation cacheline：
   改为 previous-generation retire + clear/publish + launch-armed 握手。
2. INC 在 worker D2D staging 前启动：worker 先异步 enqueue，握手后
   INC 再启动有限 spin kernel。
3. 只看 ACL event 会将协议错误当成成功：所有活跃 lane
   telemetry 成为 request 终态 gate。
4. Combine 逐 expert-row D2D 将 256MiB case 降到 16–24 GB/s：
   reverse layout 改为稳定 result/ordinal 顺序，mode-6 ingress 整 rank
   连续 staging；不规则/padded layout 仍 fail-safe 退回分段 copy。
5. native Dispatch 固定 256KiB tile 比 canonical 2MiB tile 多 8 倍
   readiness/drain 成本：接入 canonical 的最多 32 tile-epoch 边界。

## 开放项

- 跨主机 persistent service transport adapter；同机 TCP mailbox 已取代
  整链中的逐代 W+1 host barrier。
- 生产 grouped GEMM 框架 adapter；单 communicator 的
  Dispatch→device identity expert expansion→Combine 真机串联已通过，
  但 identity 阶段不冒充 GEMM 性能实现。
- 真实 kernel hang/link fault 下的 cancel/timeout 跨 rank 恢复；device
  telemetry 故障 fail-closed 与 256 代整链 soak 已完成。
- Megatron/vLLM/PyTorch 的框架绑定层与真实 grouped GEMM；native
  expert-major/padded layout compiler 和 registered-buffer 整链消费已完成。

## 单 communicator 整链验证

- 新增 `NativeCompositeBackend`，按 operation 将 workspace/enqueue/
  ticket lifecycle 转发到独立 Dispatch/Combine provider，两套 fixed
  AIV policy 不被重写。
- reverse compiler 显式导出 `contributor_dispatch_rows`，不再由示例
  猜测 physical-row 顺序。
- `inc_dc_native_full_example` 复用 dispatch route handle，在同一
  stream 上做 device-to-device identity expert expansion，再下发 Combine。
- 真机：`/tmp/inc-dc-native-full-w2-g3-r2` 为 W2 3 代 3/3 rank
  PASS；`/tmp/inc-dc-native-full-w4-g1` 为 W4 1 代 5/5 rank PASS。
- 同时修复 Release host test 中带副作用 `assert()` 被 `NDEBUG`
  删除的门禁漏洞；D/C plan compiler 测试现在显式 `-UNDEBUG`。

## 故障和长时复用

- 一次性 qualification API 在 completion event 完成后、强制 telemetry
  回读前改写 device stats；因此验证的是真实 device-memory
  fail-closed，不扰动有限 spin 通信 kernel，正常路径无额外命令。
- `/tmp/inc-dc-native-full-w2-fault-recovery-r2`：W2，generation 1
  的 Dispatch lane error 和 generation 2 的 Combine producer error 均被
  `BACKEND_ERROR` 捕获，ticket/route 正常释放；generation 3 全 rank
  恢复并数值 PASS。
- `/tmp/inc-dc-native-full-w2-soak256`：同一 D/C sessions、composite
  communicator、两个 workspace leases 连续 256 代，3/3 rank 与
  最终数值 PASS。
- 尚未完成的是“运行中 kernel 永不完成或 link 中断”这类破坏
  collective 进度的 fault；它需要 persistent service 的跨 rank abort
  mailbox，不应由单 rank 假装可恢复。

## Persistent INC proxy

- `NativeIncService` 在 INC 进程长驻，worker 只持有
  `NativeIncServiceClient`。固定消息含 magic/version/op/rank/generation，
  service 拒绝重复 rank、错 op 和越代消息。
- Dispatch 必须先收齐 `DISPATCH_PREPARE`，service 才清理 INC 的
  generation cacheline；再收齐 worker kernel 已 enqueue 的
  `DISPATCH_READY` 后启动 INC。这关闭了“worker 已发 readiness，
  INC 后 clear 又抹掉”的实测竞态。
- Combine 收齐 `COMBINE_READY` 后启动 INC；每个 response 携带同
  op/generation 和真实 INC terminal status。socket 有有界 I/O timeout。
- 真机：`/tmp/inc-dc-native-full-service-w2-g3-r2` 3/3 rank，
  `/tmp/inc-dc-native-full-service-w4-g1` 5/5 rank，
  `/tmp/inc-dc-native-full-service-w2-soak256` 256 代 3/3 rank，
  `/tmp/inc-dc-native-full-service-w2-fault` 两次 fail-closed 后恢复；
  均不使用逐代 `aclshmem_barrier_all`。

## Expert layout adapter

- `BuildNativeExpertLayout()` 以模型配置的 rank-major expert ID 全集为
  输入，不从 workload 推断 expert 数；零 token expert 在 logical/padded
  offsets 和 `tokens_per_expert` 中显式保留。
- compiler 同时交付 physical Dispatch row→expert-major padded row 和
  Combine contributor row→padded row inverse map；同 rank 多 expert 会复用一条
  物理 Dispatch row，但仍产生多个独立 expert rows。
- host gate 覆盖 alignment=8、zero-token expert 和非 2 的幂 alignment
  fail-closed。真机 `inc_dc_native_full_example` 实际经过 padded device
  buffer：`/tmp/inc-dc-native-full-layout-w2-g3` 3/3 rank，
  `/tmp/inc-dc-native-full-layout-w4-g1` 5/5 rank，
  `/tmp/inc-dc-native-full-layout-service-w8-g1` 9/9 rank。

## 最终无回退代表复测

新增 service/composite/layout 均未修改 D/C device 数据 kernel。仍以
registered view 单算子 runner 复测大消息：

| 算子 | 日志 | 最终结果 |
|:---|:---|:---|
| Dispatch | `/tmp/inc-dc-native-dispatch-w2-256m-postservice` | 256 MiB，10/10 timing，protocol 123.399–126.917 GB/s；当前 anchor 123.339，PASS |
| Combine | `/tmp/inc-dc-native-combine-w2-postservice` | 178,978,816 B ingress，每 sample 最慢 rank 141.251–143.215 GB/s，PASS |

Dispatch 该次低值较前一组 124.288 低 0.71%，高值较前一组
128.047 低 0.88%；protocol cycles 在未变 kernel 内采集，幅度属当前
环境样本波动，未观察到系统性回退。Combine 与原区间重合。
