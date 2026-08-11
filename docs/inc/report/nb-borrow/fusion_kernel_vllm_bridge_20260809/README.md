# nb-borrow Fusion vLLM 桥接 / ABI v3–v6 增量（2026-08-09）

本目录是新数据集，不覆盖 `fusion_kernel_20260809`、single-INC 带宽报告或
yuanmingyu 环境的历史数据。

## 本轮代码增量

- 设备 `route_count/route_pack` 与 CPU golden 逐字节等价，支持 int32/int64 route ID、
  非均匀 expert placement、非整齐 token 尾部和 prepared capacity 空 wave。
- Torch NPU `out` 桥接在 vLLM-Ascend 0.19.1 容器的 CANN 8.5 下独立构建并真机通过。
- 每个 worker 只 all-gather 一份 `[wave_capacity,E+1]` int32 元数据；最后一列携带
  active-token 数，不做 `.cpu()`、`.item()` 或第二次 collective。
- Fusion ABI 升至 v3；INC 可按每个 source 的真实 active-token 数清零、校验、归约和
  回传。未绑定该向量的旧调用保持统一 `token_count` 兼容语义。

## ABI v3 全链路回归

映射仍为同一 HCCS 平面：W2 worker=NPU1/2、INC=NPU0；W4
worker=NPU1/2/3/4、INC=NPU0。每个 case warmup=1、measure=2；这是 ABI smoke，
不替代此前 10/50 次稳定性数据。

| case | makespan mean | CV | D/C 理论最大收益 | D/C 真实收益 | 交叠实现度 | 正确性 |
|---|---:|---:|---:|---:|---:|---|
| W2 T32 H256 I512 K2 | 605.970 us | 0.437% | 1.67072× | 1.65981× | 99.0203% | PASS，全量 golden |
| W4 T32 H256 I512 K2 | 848.610 us | 1.337% | 1.79172× | 1.78103× | 99.2419% | PASS，全量 golden |

理论收益仍定义为 `(Td+Tc)/max(Td,Tc)`；真实收益为
`(Td+Tc)/(Td+Tc-overlap)`，不是用 rank 数直接推导。测试结束后 16 张 NPU 均无残留
进程。原始日志位于 `/tmp/inc-fusion-abi-v3-smoke-20260809`，归档摘要见
`abi_v3_smoke.csv`。

## 非均匀 worker token 数

进一步以 32-token prepared capacity 运行不同 source 长度。每个 worker 只分配并验证
自己的真实 output 行；没有为短 rank 分配 32 行 output 来掩盖越界。

| case | 各 worker active tokens | makespan mean | CV | 理论收益 | 真实收益 | 交叠实现度 | 正确性 |
|---|---|---:|---:|---:|---:|---:|---|
| W2 | 17, 32 | 610.260 us | 0.302% | 1.58360× | 1.57259× | 98.7997% | PASS，各 rank 全量 golden |
| W4 | 11, 17, 23, 32 | 720.380 us | 1.383% | 1.65585× | 1.64633× | 99.1176% | PASS，各 rank 全量 golden |

这验证了 ABI v3 的按-source清零、route 校验和回传长度；短 rank 的 checksum/verified
tokens 分别对应 17，以及 11/17/23 行。完整摘要见 `abi_v3_uneven_tokens.csv`。

## ABI v3 当时尚未完成（历史快照）

- `inc_fusion_native::moe` 尚未注册；当前完成的是其设备路由和 prepared buffer 前置层。
- worker prepared executor 与跨进程、设备热路径的 INC command ring 仍需连接。
- 完成上述两项后，才会执行统一四后端 2×2 大 sweep 与真实 vLLM
  prefill/decode/TTFT/TPOT 对比。

## Torch 当前流顺序回归

原生桥接最初使用 `NPUStream::stream(false)` 直接提交 ACL kernel。该参数不会先把
torch_npu 的软件 task queue 排入当前 ACL stream，因此 route kernel 偶尔会越过前序
`arange/floor_divide/copy_`，表现为随机 `BadArgs/BadPlacement`。这不是 route 协议或
allocator 容量问题。桥接现改用 `stream()`：它只排空 task queue、保持同流顺序，不做
host/device 全局同步。

修复后用 3 个全新 Python 进程重复 W2/W4 × T32/128/512/2048/8192 × K2/K8，
每 case 5 次 warmup、20 次 measure：60/60 case `status_error=0`，row/assignment 数均与
解析结果一致。代表性中大规模如下（这里只计 device count+pack，不含 metadata HCCL）：

| case | 三进程 mean 范围 | 三进程 CV 范围 |
|---|---:|---:|
| W2 T8192 K2 | 3.853–3.875 ms | 0.900%–1.021% |
| W2 T8192 K8 | 10.357–10.377 ms | 0.986%–1.153% |
| W4 T8192 K2 | 4.157–4.173 ms | 0.894%–1.374% |
| W4 T8192 K8 | 12.638–12.647 ms | 0.825%–0.903% |

完整 60 行原始数据见 `native_route_stream_order_stability.csv`。该结果证明顺序与稳定性，
不代表 route-pack 已达到最终性能目标；当前单 AIV 标量 pack 仍是大 token/top-k 下的
明显前处理开销，后续需要按确定性前缀分配做多 AIV 并行。

## Prepared worker executor

新增的 worker executor 在 setup 时一次性准备 workspace、pinned-host/device 参数环和每槽
event；热路径不会分配或全局同步。每次调用只构造固定 ABI、异步复制一个参数记录、启动
worker kernel 并记录槽位 event；跨 stream 连续调用若追上尚未完成的槽位会返回 `BUSY`，
不会覆盖仍在执行的参数。

用 `FUSION_PREPARED_WORKER=1` 走真实 E2E 路径做 smoke：W2/W4 均为全量 golden PASS，
INC 的 D/C 交叠实现度分别为 99.6322% 与 99.2981%。结果见
`prepared_worker_executor_smoke.csv`。当前计时把 executor 的参数 H2D 也纳入 event window，
因此不能与旧的“参数复制在 start event 前”数字直接作性能回归；后续四后端统一计时会采用
同一边界。

## Route-pack 前缀优化

在保持单 AIV 与完全相同输出 ABI 的前提下，去掉两类重复工作：expert/source 的
destination-row 前缀由“每 assignment 重算”提升为每 wave 预计算；每 token 的
destination match 由每个目的 rank 各自扫描，改为一次 top-k 扫描后复用。该改动不包含
shape 特调或参数遍历。

同一 3 进程 × 20 case 回归再次 60/60 PASS。相对 stream 修复后的基线，全部 case 延迟
下降 13.07%–30.20%；T8192 的代表结果为：

| case | old | new | 降幅 |
|---|---:|---:|---:|
| W2 K2 | 3.863 ms | 2.766 ms | 28.39% |
| W2 K8 | 10.367 ms | 8.850 ms | 14.64% |
| W4 K2 | 4.166 ms | 2.908 ms | 30.20% |
| W4 K8 | 12.642 ms | 10.160 ms | 19.63% |

20 个 case 的三进程聚合数据见 `native_route_prefix_optimization.csv`。中大规模 CV 最大
约 1.52%；小规模受 host/launch 抖动影响，最高 4.53%。该结果随后作为下节确定性多 AIV
analyze/prefix/emit 的标量基线，没有对单 AIV 参数做逐 shape 搜索。

## 确定性多 AIV route-pack

已完成 analyze → prefix → emit 三阶段 pack。48 个 AIV 各自处理每个 wave 的连续 token
区间并写独占 64B histogram/state；单 AIV prefix 唯一决定 row、assignment、expert-row
顺序；emit 通过 UB 缓冲和 MTE3 写紧凑 32B 协议记录，避免相邻 lane 的 cache-line
覆盖。原单 AIV pack 保留为小输入和故障回退。

正确性资格化包括：宿主 CANN 9.1 的 CPU/scalar/parallel 三方逐字节比较，以及 vLLM
CANN 8.5 容器中的 W2/W4、T37/513/8192、K2/K8、int32/int64 共 24 个大规模/尾波
逐字节 case；另注入越界 expert 与 NaN weight，两类错误均精确返回 token/ordinal，26/26
PASS。

性能选择不遍历 shape 参数：prepared 阶段读取 live AIV；forward 以协议的 destination
regroup 工作量 `T*K*(K-1)` 与固定 64-record UB 批次决定 scalar/parallel。标准 20-case
矩阵中策略选中项均不低于当前 scalar，原始对照见 `native_route_parallel_protocol.csv`。
代表结果：

| case | scalar | parallel | 加速 |
|---|---:|---:|---:|
| W2 T8192 K8 | 8.833 ms | 3.612 ms | 2.445× |
| W4 T8192 K8 | 10.194 ms | 3.751 ms | 2.717× |
| W2 T8192 K2 | 2.771 ms | 2.430 ms | 1.140× |
| W4 T8192 K2 | 2.905 ms | 2.479 ms | 1.172× |

小输入如 T32 仍由 scalar 执行，因此不会承担并行路径多出的两次 launch。这里仍只计
device count+pack，不含 EP metadata HCCL；完整融合算子仍需把 prepared executor 和跨进程
INC command ring 接进 Torch，之后再按统一四后端计时边界做端到端 sweep。

## ABI v6：跨进程 INC 服务闭环

prepared worker executor 与跨进程 persistent INC command ring 已在 C E2E 路径闭环。
INC 进程只在 setup 阶段创建并启动一次 48-AIV 常驻 kernel；每个 worker 请求以 ticket
发布，INC 收齐全部 worker readiness 后启动同一请求，32 个 Dispatch AIV 与 16 个
Combine AIV 可按任意到达时序并发执行，完成结果再按 ticket 回传。测试映射固定为同一
HCCS 平面：W2 worker=NPU1/2、INC=NPU0；W4 worker=NPU1/2/3/4、INC=NPU0。

本轮修复了两个会在多请求下随机挂起的协议问题：

1. packet header 的 producer `ready/metadata` 与 consumer `credit` 原在同一 64B cache
   line，不同设备上的 DCCI 可能把对端 8B 更新用旧 cache line 覆盖。ABI v5 起把两类
   writer 分到两个独占 cache line，header 为 128B；发布和回收都只写自己拥有的 line。
2. 原远程命令由 waves、active-token、generation 等多笔小 SDMA NBI 组成；实测失败请求的
   `active_token_counts` 变成 `110,0`，恰好是该 ticket 的 64-bit generation，证明连续小
   SQE 发生污染。ABI v6 将 generation/ticket/request-id/flags、waves 与 active-token
   合成一个连续 wire record，单次提交完成后再独立发布 descriptor `ready`。

精确 `ready/credit` 代际已经足以保护 queue slot 复用，因此同时删除 service ring 回卷时
对四个方向全部 packet header 的全量清零。旧实现前 8 个 ticket 约 0.60–0.88 ms，第 9
个起因清零约 4×1536 个 header 固定升到约 2.4 ms；删除后用最小 ring=2 反复回卷，延迟
保持稳定，也比默认 ring 更强地覆盖复用协议。

### 长稳与正确性

| case | ring | warmup/measure | makespan mean | CV | 理论收益 | 真实收益 | 交叠实现度 | 结果 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| W2 T32 H256 I512 K2 | 2 | 5/100 | 620.232 us | 1.970% | 1.54090× | 1.54051× | 99.9532% | 100/100 PASS |
| W4 T32 H256 I512 K2 | 2 | 5/100 | 709.542 us | 2.182% | 1.59890× | 1.59820× | 99.9260% | 100/100 PASS |

这里的理论收益仍为 `(Td+Tc)/max(Td,Tc)`，真实收益为
`(Td+Tc)/(Td+Tc-overlap)`；它由本次请求真实 D/C 时间窗决定，不由 worker 数量直接推导。
100 次测试每次都复用同两个 service slot、worker arg slot 和数据 queue segment，且所有
worker 的 BF16 结果都通过 CPU golden 误差门槛。

## Torch worker 生命周期 API（ABI v6）

在 C API 长稳闭环之上新增了可选编译的 Torch runtime bridge：

- `worker_prepare` 在 engine setup 将固定容量 plan、executor 参数/event ring、expert
  placement、worker PE 表和外部 SHMEM symmetric heap 绑定成 opaque handle；
- `worker_moe_out` 只消费 prepared route/weight/input/output，并在当前 NPU stream 调用
  `inc_fusion_worker_executor_enqueue`；generation 由 handle 单调管理，成功 enqueue 后才推进；
- `worker_destroy` 从 registry 摘除 handle，再等待本 executor 的在途 event 并释放资源；
- INC sidecar 对称地提供 `service_prepare/start/destroy`，Python SHMEM binding 新增包含
  sidecar 的 `aclshmem_barrier_all`；service start 明确位于 W+1 PE setup barrier 之后；
- graph-visible `inc_fusion::moe` 使用 prepared output alias，fake 路径也返回同一个 alias，
  不再在 forward 中 `empty_like`；
- native 热路径仅发布已完成的 `serial_inc=2` / `fused_inc=4`，SHMEM 两个 attribution
  backend 未完成前显式报错。

随后补齐了启动前的 allocation 查询：CPU `plan_info` 在尚未初始化 SHMEM 时返回
`symmetric/worker/INC workspace`、wave 数、local expert 数、ABI 与 remote-service 大小。
W 个 rank 必须拥有相同 service layout，rank-local workspace 允许不同；C API 单测已覆盖
逐 rank 比较。heap minimum/reserve/alignment 和 PE→NPU 映射被提升为显式 deployment
policy，不再把 nb 的设备编号或 512MiB 测试 heap 当成协议常量。host-only 单测现为
8/8 PASS，CPU dispatcher schema 与错误状态传播也已验证。

并发审计进一步确认：prepared route/output 是单份 buffer，同一 NPU stream 的设备顺序足以
安全复用，但跨 stream 使用同一 handle 会发生覆盖，因此 native 层现绑定首个 stream 并
拒绝后续不同 stream。NPU graph capture/replay 也暂不支持，因为它会重放 host capture 时的
旧 generation/ticket；bridge 使用 torch_npu capture-status 硬 gate，普通逐轮 custom-op
调用不受影响。新增 `inc_sidecar.py` 将独立 sidecar 控制固化为
PREPARED/READY/STOPPING/STOPPED 协议，超时不会破坏性 terminate。host-only 单测现为
9/9 PASS。

权重接入新增 setup-time `PreparedWeightCache`：支持 fusion ND、逻辑转置 ND，以及明确的
FRACTAL_NZ(29)→ND(2) 加载期副本；forward 只做源 tensor pointer/device 身份校验。尝试在
宿主唯一 Torch-NPU 2.7.1/CANN 8.2 venv 做极小真机格式回归，但该临时环境先后缺少 TBE
Python 路径和 `decorator`，两次均在 `torch.npu.set_device` 阶段失败，未执行 tensor 或
kernel。因此这里只记录源码/语法验证，不把它标为真机 PASS；目标 vLLM 容器需重新执行
ND、转置 ND、NZ 三组 gate。

host-only lifecycle/policy 单测 9/9 PASS，Python 文件全部通过 `py_compile`。原生桥用
Torch-NPU 2.7.1 头文件完整编译通过，并用无设备 stub 成功加载；worker 的
prepare/destroy/moe_out 与 sidecar 的 prepare/start/destroy schema 均被 PyTorch
dispatcher 接受。宿主正式 fusion 库由
CANN 9.1 构建，而现存 Torch-NPU 环境是 CANN 8.2，二者混载在
`rtFunctionGetMetaInfo` 处失败，因此本节没有伪报 Torch 真机执行结果；需要在目标 vLLM
CANN 版本内重建 route/fusion/API 后做 engine 联调。原先的 W2/W4 C API 100 次真机结果
不受影响。

### shape、top-k 与动态 token 覆盖

| 覆盖点 | W2 | W4 | 结果 |
|---|---:|---:|---|
| 非对齐尾部 T17/H192/I320/K1 | 724.885 us | 798.356 us | PASS |
| 中等 GMM：T64/H1024/I1024 | K4 1148.796 us | K8 4166.370 us | PASS |
| H2048：T128/I1024/K2 | 1621.560 us | 2258.322 us | PASS |
| 不均匀 active-token | `37,11` 720.033 us | `37,29,17,5` 852.248 us | PASS |
| 多包宽行：T64/H16384/I128/K2 | 1921.467 us | 2350.417 us | PASS |

上述正式 measure 的 CV 为 0.407%–2.778%，交叠实现度为
99.8709%–100%。大 shape 只减少 CPU golden 的 token 数，设备端仍处理表中全部 token；
小 shape 与 100 次长稳均执行相同设备协议。聚合原始数字保存在
`abi_v6_remote_service_results.csv`，对应运行日志目录记录在该 CSV 最后一列；没有覆盖
ABI v3、旧 nb 或 yuanmingyu 数据。

ABI v6 目前证明的是跨进程 C API 的完整服务生命周期与融合执行路径。完整 Torch
`inc_fusion_native::moe` 仍未注册，因为还需要把各 rank 的 executor/service 生命周期
绑定进 ProcessGroup/engine 初始化与 teardown；在这层完成前，不把 C E2E 冒充成可直接
用于 vLLM 的 Python 算子。
