# Single-INC Dispatch / Combine — 快速了解

面向 **完全没接触过本仓库**、但需要在半天内建立正确心智模型的读者。  
本文只讲当前保留的 **单 INC 星型拓扑** 主逻辑。
所有结论均可在下列路径核对（以源码与 CMake 为准，不以历史文件名为准）。

## 最快接入：业务侧只保留 5 个动作

新代码只包含
[`inc_dc_single_inc_api.h`](../common/api/inc_dc_single_inc_api.h)，不要逐层调用
Inference / Easy / Framework。部署阶段把 composite backend、设备 allocator、
固定 shape 和 device token-plan 填入一次 `config`；真实 native worker 再调用一次
`BindNativeSingleIncWorkerControl`，它会隐藏常驻 INC service 的 generation 握手。

```cpp
NativeSingleIncWorkerControl control{dispatch_session, service_client};
CHECK(BindNativeSingleIncWorkerControl(&control, &cfg));
// control、dispatch_session、service_client 的生命周期覆盖 op。
```

```c
inc_dc_single_inc_t *op = NULL;
inc_dc_single_inc_route_t route = {0};
inc_dc_single_inc_config_t cfg;
inc_dc_single_inc_config_init(&cfg);
/* 一次性填写 cfg 的 world/rank/shape/backend/allocator/static_route。 */
CHECK(inc_dc_single_inc_create(&cfg, &op));

inc_dc_single_inc_io_t d, c;
inc_dc_single_inc_io_init(&d);
d.input = token_hbm; d.output = expert_input_hbm;
d.stream = stream; d.operation_generation = generation;
CHECK(inc_dc_single_inc_dispatch(op, &d, &route));

run_grouped_gemm(expert_input_hbm, expert_output_hbm, stream);

inc_dc_single_inc_io_init(&c);
c.input = expert_output_hbm; c.output = token_output_hbm;
c.stream = stream; c.operation_generation = generation;
CHECK(inc_dc_single_inc_combine(op, &route, &c));
CHECK(inc_dc_single_inc_route_release(op, &route));
CHECK(inc_dc_single_inc_destroy(op));
```

完整可运行、会打印 fan-out / reduction 结果并做 golden 检查的例子是
[`../common/examples/inference_api/inc_dc_inference_api_example.cpp`](../common/examples/inference_api/inc_dc_inference_api_example.cpp)。
这层只是零拷贝生命周期封装，不改变 V1 kernel、token-plan 语义或性能路径。

| 角色 | 权威路径 |
|---|---|
| 目录总览 | `README.md`（本目录） |
| Dispatch ABI / kernel | `dispatch/inc_dc_single_inc_stream_abi.h`、`dispatch/inc_dc_single_inc_stream_kernel.cpp` |
| Combine 逻辑计划 / 编译 / 设备 | `combine/inc_dc_combine_logical_plan.h`、`combine/inc_dc_combine_plan_compiler.h`、`combine/inc_dc_combine_kernel.cpp`、`combine/inc_dc_combine_runtime_abi.h` |
| D↔C 正反向计划 | `planning/` |
| 框架接线 | `../common/api/inc_dc_single_inc_api.h`（推荐）+ `runtime/`（部署适配） |
| 真机入口 | `../scripts/single_inc/` |
| **当前 sweep 进度 / 环境 / baseline** | [`SWEEP_STATUS.md`](SWEEP_STATUS.md) |

---

## 中文

### 0. 一句话

单 INC 把 MoE 的 **Dispatch / Combine 两段通信** 做成 **一次（或少数几次）设备 launch 的星型数据面**：

- **Dispatch**：各 worker 把 token 按路由发送到「专家所在的目的 worker」——中间经 **唯一 INC** 做 gather / fan-out。
- **Combine**：各 worker 把专家贡献推到 INC；INC 按计划做 **加权归约**，再把结果 **fan-back** 到源 token 所在 worker。

拓扑固定为：

```text
逻辑 PE:  0 .. W-1     →  workers
逻辑 PE:  W            →  唯一 INC
世界大小: W + 1
```

资格化脚本目前只跑 **W ∈ {2,4,8}**。逻辑 PE 与物理 NPU 编号解耦（由 `INC_PE_TO_NPU_MAP` / `inc_single_pe_map` 映射）。

---

### 1. 先建立三张「地图」

#### 1.1 代码地图（不要从文件名猜产品路径）

```text
dispatch_combine/
  common/          ← 公开 C ABI、协议、平台原语（框架只应依赖这里）
  single_inc/      ← 本文主角：星型 Dispatch + Combine + runtime
  scripts/         ← 真机唯一推荐入口
  tests/           ← host 门禁（可不占 NPU）
```

`single_inc/` 内部：

| 子目录 | 你要它回答的问题 |
|---|---|
| `dispatch/` | 设备上 Dispatch 怎么传？ |
| `combine/` | 设备上 Combine 怎么归约？ |
| `planning/` | host 如何把 route 编成有界 workspace，且 D/C 正反向一致？ |
| `runtime/` | Framework/Easy 如何接到真实 ACLSHMEM kernel？ |
| `tools/` | 随机计划从哪来？（非运行库） |

#### 1.2 调用地图（框架视角）

Framework 只看到 **合成后的** backend vtable；两个 native provider 先合成再注入：

```text
SingleInc API → Inference API → Easy API → Framework C API (backend vtable)
                                      ↓
                    native_composite_backend（合成后的单一 communicator）
                         ↙                         ↘
          native_dispatch_backend          native_combine_backend
                         ↘                         ↙
              常驻 NativeIncService（INC 侧） + worker 侧 enqueue kernel
```

公开符号前缀是 `inc_dc_`。**不要**把内部的 `bw03` / `bw05` / `sv2` / `c0` 当成公开 API——它们是历史/二进制兼容名（CMake **target/exe** 仍可能叫 `sv2`，但编译源用稳定名，见 `combine/README.md`）。

#### 1.3 数据地图（MoE 语义 ↔ Dispatch / Combine）

```text
token × topk 路由
        │
        ▼
   ┌─────────┐     上传/打包      ┌─────┐     fan-out      ┌──────────────┐
   │ workers │ ───────────────► │ INC │ ───────────────► │ dest workers │
   └─────────┘                  └─────┘                   └──────┬───────┘
        ▲                                                        │
        │                     fan-back 结果                       │ expert 计算
        │                                                        ▼
        │      ┌─────┐     加权归约      ┌─────────────────────────┐
        └──────│ INC │ ◄────────────── │ workers（贡献 upload）  │
               └─────┘                  └─────────────────────────┘
                 Combine
```

Dispatch 解决「token 去哪算」；Combine 解决「算完怎么按权重合回来」。

---

### 2. Dispatch：主逻辑（可信到函数名）

#### 2.1 入口怎么分角色

同一份 kernel 二进制，按 `StreamDispatchDesc::pe` 分叉
（`inc_dc_single_inc_stream_dispatch_kernel`）：

```text
pe <  workers  →  StreamWorker(...)
pe == workers  →  StreamInc(...)
```

Host 侧计划编译链（概念顺序）：

`CompileStreamSourcePlan` → `MergeStreamSourcePlans` → `BuildStreamPreparedWorkspace`
→ 填满 `StreamDispatchDesc` + 对称 heap。

#### 2.2 你必须认识的几个 ABI 对象

定义见 `dispatch/inc_dc_single_inc_stream_abi.h`（`namespace inc::dc::single_stream`）：

| 对象 | 作用 |
|---|---|
| `StreamDispatchDesc` | 整次 op 的布局真相源：offsets、lane 数、hidden、topk、generation… |
| `StreamDispatchTask` | 「从一个已发布的 source tile，打一包连续数据，发给某一个 destination worker」 |
| `StreamRouteEntry` | rank-local 的 `source_row` + assignment 切片；同目的地多专家可压成一行传 hidden |
| `StreamExpertAssignment` | `expert_id / route_ordinal / weight_bits` |
| `generation` | 本次 op 的 cacheline 信号值；`StreamWaitGeneration` 自旋等待 |
| lanes | worker：`upload_lane_count`；INC：`gather_lane_count` + `tx_lane_count`（多为 live AIV 派生，不是写死 W2/W4 表） |

对称 heap 约定：描述符在 `kStreamDescOff=0`，数据区从 `kStreamDataOff=4096` 起。魔数 `kStreamMagic = 'STRM'`。

#### 2.3 设备时间线（按同步点理解，而不是按「函数列表」）

**阶段 A — Start-gate（没有 go 就不上传）**

1. 各 worker 的 lane0：向 INC 发 **arrival**；等 INC 的 **arm**；再回 **ack**；全体 upload lane 等 **go**。
2. INC 的 start coordinator：收齐所有 worker arrival → 发 arm → 等齐 ack → 发本地 `local_go`，再给各 worker 发 go。

对应字段在 `start_gate_off` 一带（arrival / arm / ack / go / local_go）。

**阶段 B — Worker 上传（`StreamWorker`）**

资格化 / 产品路径始终是 **`worker → INC → worker` push-only**（共享硬约束 **H1**，见  
`docs/inc/report/single_inc_LIVE_STATUS.md`）。由 flags 选择的两种**经 INC** 上传方式：

1. **Tile / `HasDirect`（仍经 INC）**：多 upload lane 把本 worker 的 `input_off[pe]` **推到 INC**；lane 本地写 `upload_chunk_done`；lane0 收齐后把 `tile_ready_off[worker,tile] = generation` 通知 INC。这里的 “direct” 指传输实现（如 DCCI / private-MTE），**不是**绕过 INC。
2. **`WorkerPack`（经 INC staging）**：按本源任务表只打包需要的行（`StreamPackGenericTask` + MTE gather）→ 送到 **INC staging**，再由 INC TX 出去。

内核里仍可能残留 **`kStreamFlagWorkerDirect`（worker 直达目的 worker、bypass INC）** 分支；  
资格化 launcher 将其固定为关闭（`sparse_worker_direct = false`）。**交付/扫带宽时禁止当作产品路径**——违反 H1。

传输大量使用 `aclshmem_putmem_nbi` / `aclshmem_putmem_signal` / private-MTE pingpong；gather 原语复用  
`../common/platform/inc_dc_gather_mte_aicore.h`（禁止逐字节 AIV 循环）。

**阶段 C — INC fan-out（`StreamInc`）**

1. **Gather lanes**（`lane < gather_lane_count`）：等 `tile_ready` → 按 task/route 从已上传 input 做 MTE gather 填 staging → cache 可见性处理 → 在对应 chunk completion 行发布 generation。
2. **TX lanes**：等 chunk completion（或经 INC 的 ready）→ 把 packet RMA 到 `destination_rank` 的 `output_off + output_byte_offset`。

**阶段 D — Completion**

- INC finish coordinator 收齐 TX `lane_done` 后，向每个 worker 的 `completion_off[worker]` 发 `generation`。
- Worker lane0：先收齐本 rank upload `lane_done`，再等自己的 completion 行。

**不变量（读代码时用它们当「断言」）**

1. 没有 start-gate 的 go，worker 不得开始有效上传。
2. INC 在 `tile_ready` / chunk completion 之前不得认为 payload 可读。
3. Gather→TX 有明确依赖；generic 路径 gather 后有 cache 可见性动作。
4. 结束以 worker 侧看到 completion generation 为准（再叠加本 rank lane_done）。

---

### 3. Combine：主逻辑（可信到函数名）

#### 3.1 先分清「产品路径」和「历史名」

| 你应该读的稳定名（CMake 实际编译） | 仍可能见到的历史/CMake 名 | 说明 |
|---|---|---|
| `inc_dc_combine_kernel.cpp` | **target** 名 `inc_dc_sv2_dyn_csr_combine_kernel` | **产品数据面 = dyn-CSR**；target 名仅为兼容保留 |
| `inc_dc_combine_launcher.cpp` | **exe** 名 `inc_dc_sv2_dyn_csr_combine` | `run_single_inc_dyn_case.sh` 跑的是该 exe |
| `inc_dc_combine_runtime_abi.h` | `DynCsrCtrl`，magic `'DYCS'` | 唯一 host/device ABI 头 |
| `inc_dc_combine_logical_plan.*` | 当前结构即 V2 字段 | 唯一逻辑计划实现 |

构建真相：打开 `examples/inc/CMakeLists.txt` 搜文件名；详表见 [`combine/README.md`](combine/README.md)。

#### 3.2 Host：逻辑计划 → 拓扑 → 可执行计划

概念流水线：

```text
IncDcCombineLogicalPlanV2          （与物理拓扑无关的 contribution/result）
        │
        ▼
IncDcTopologyDescriptor            （worker↔唯一 INC 的显式可达性）
        │
        ▼
CompileLogicalPlanToExecution      → IncDcCompiledExecutionPlan
        │                              （owner、ingress channel/slot、worklist CSR…）
        ▼
BuildNativeCombinePreparedWorkspace → DynCsrCtrl + 有界对称 heap
```

与 Dispatch **严格反向** 的框架路径在  
`planning/inc_dc_single_inc_combine_plan_compiler.h`：  
`BuildCombineReverseLayout(StreamCompiledGlobalPlan)` ——  
contribution 顺序跟 Dispatch 物理行 / assignment 对齐，避免「正反向漂了但各自还能跑」的假绿色。

逻辑计划关键字段（v2）：

- `IncDcLogicalResultV2`：结果落在哪个 `dst_rank/row`，需要多少 contribution。
- `IncDcLogicalContributionV2`：谁贡献（`contributor_rank/row`）、ordinal、weight、uid。

非法拓扑（worker 无法到达唯一 INC）在 compile 阶段 **fail-closed**，不会静默错路由。

#### 3.3 设备：三个阶段

入口同样按 PE 分叉（native backend / kernel 内）：

```text
pe <  W  →  producer（上传贡献）
pe == W  →  INC reducer +（可拆分的）TX fan-back lanes
```

**阶段 1 — Producer（worker）**  
`inc_dc_sv2_dyn_csr_producer_kernel`（实现落在 combine kernel 编译单元内）：

- 可选：先驻留，等 `DynCsrPersistentTriggerLine`（把 device launch 调度踢出计时区）。
- 按 `ready_mode`（native 产品路径固定一类 packed stream chunk 模式）把本 worker 的 contribution 组 RMA 到 INC `ingress_off`。
- **先保证 payload 远程可见，再发 ready generation**（`ready_generation_off`）。
- top-k=1 且可证明「单贡献、权重 1」时可走 identity 直传核（拷贝而非归约）。

**阶段 2 — INC reduce**  
`inc_dc_sv2_dyn_csr_combine_kernel` 中 `bid < owner_count` 的 owner：

- 按 **result CSR / owner worklist** 扫自己该管的 result（注释强调：**不要**按 `source=0..N` 盲扫）。
- 等齐 expected contributions → FP16→FP32 **加权向量归约**  
  （`inc_dc_combine_vector_reduce_aicore.h`；fail-closed，禁止静默标量回退）。
- 写到 `output_off`，并置 `result_tx_ready` 一类信号供 TX。

**阶段 3 — Fan-back TX**  
`bid ≥ owner_count` 的 split TX lane：

- 等 `result_tx_ready` → 把结果 RMA 到 `result_dst_rank` 对应 worker。
- 可选按 rank 打包连续推送以减 RMA 次数。

**Completion**  
若启用 `device_completion`：INC owner0 向各 worker 发 collective completion generation；producer lane0 轮询后 ACK。  
Launcher 也可依赖 host/stream 同步——读脚本/环境变量时注意两种模式。

**不变量**

1. Producer：payload 完成前不得发 ready。  
2. INC：expected contributions 未齐不得宣称该 result reduce 完成。  
3. Owner 只处理 `result_home_owner` 匹配的 result。
4. TX 必须看到 `result_tx_ready` 再 fan-back。  
5. 重复/过期 contribution：`fail_closed_on_dup` 等选项，避免脏累加。

---

### 4. Runtime：框架如何接到上述 Dispatch / Combine 路径

| 组件 | 文件 | 一句话 |
|---|---|---|
| Dispatch provider | `runtime/inc_dc_native_dispatch_backend.*` | enqueue → `launch_inc_dc_single_inc_stream_dispatch_kernel` |
| Combine provider | `runtime/inc_dc_native_combine_backend.*` | worker→producer / INC→combine |
| 合成 communicator | `runtime/inc_dc_native_composite_backend.*` | Easy/Inference 只持一个 backend |
| 常驻 INC 服务 | `runtime/inc_dc_native_inc_service.*` | INC 侧 TCP proxy + generation rendezvous；避免每 op 做笨重的 W+1 host barrier |
| Combine workspace | `runtime/inc_dc_native_combine_workspace.*` | 有界 dyn-CSR heap + 不可变 metadata |
| 整链示例 | `runtime/inc_dc_native_full_example_main.cpp` | Easy + composite + service 冒烟 |

典型 worker 热路径次序：

1. （可选）`DISPATCH_PREPARE` / clear  
2. 本地 enqueue Dispatch 或 Combine kernel  
3. `NativeIncServiceSubmitAndWait(...)` 与 INC 对齐 generation  
4. wait / 查 request 状态  

热路径 **不分配**：大输入由 `common/api` 的 chunk planner + 本目录 planning 预先切好。

---

### 5. 建议阅读顺序（约半天）

1. 本文 → `README.md` → `dispatch/README.md` → `combine/README.md` → `runtime/README.md`  
2. **整文件**读 `dispatch/inc_dc_single_inc_stream_abi.h`  
3. 在 `inc_dc_single_inc_stream_kernel.cpp` 里按符号跳读：  
   `StreamWaitGeneration` → `StreamWorker` → `StreamInc` → `inc_dc_single_inc_stream_dispatch_kernel`  
4. `planning/inc_dc_single_inc_stream_plan_compiler.h` + `inc_dc_single_inc_combine_plan_compiler.h`（正反向）  
5. `combine/inc_dc_combine_logical_plan.h` → `inc_dc_combine_topology.h` → `inc_dc_combine_plan_compiler.h`  
6. `combine/inc_dc_combine_runtime_abi.h`（`DynCsrCtrl` / `kDynCsrOpt*`）  
7. `combine/inc_dc_combine_kernel.cpp`：先抓 producer → owner reduce → split TX 控制流，再看优化分支  
8. `runtime/inc_dc_native_inc_service.h` + 浏览 `inc_dc_native_full_example_main.cpp`  
9. `../common/api/inc_dc_framework_c_api.h` / `inc_dc_easy_api.h`（你对外该承诺的边界）  
10. `scripts/single_inc/` 的拓扑与空闲门禁

---

### 6. 最小可跑通（真机）

```bash
# 在 shmem 的父目录（例如 ascend-样机/）执行：
cmake -S shmem -B build -DCMAKE_BUILD_TYPE=Release -DUSE_EXAMPLES=ON
# 若已在 shmem/ 内，则用：cmake -S . -B build ...
cmake --build build -j --target inc_dc_single_inc_stream inc_dc_sv2_dyn_csr_combine
```

**资格化 / 正式门禁要求走脚本**（拓扑校验、NPU 空闲锁、多 rank 结果检查）。  
二进制仍可被直接调用，但会绕过上述保护，**结果不作交付证据**：

```bash
cd shmem/examples/inc/dispatch_combine/scripts/single_inc

# Dispatch：W=4, tokens/worker=64, hidden=8192, topk=2
./run_single_inc_stream_dispatch_case.sh 4 64 8192 2 /tmp/dc_log

# Combine dyn-CSR：W=4, topk=2, results=256, hidden=4096, mode=0
./run_single_inc_dyn_case.sh 4 2 256 4096 0 /tmp/dyn_log
```

成功时 Dispatch 日志应出现类似 `STREAM_DISPATCH_RESULT ... pass=1`。  
更多矩阵用 `run_single_inc_operator_sweep.sh`；随机覆盖见 `../random_token_plan/`。

Host 逻辑回归（不占 NPU）：`../tests/common/`、`../tests/single_inc/`。

---

### 7. 常见误读（请避开）

1. **以为本树还包含独立的「SwiGLU / expert 计算」阶段**——这里是通信库；expert GEMM 在框架或 `fusion_kernel/`（**不进** `examples/inc/CMakeLists.txt`）参考集成里。  
2. **根据旧报告里的 target 名寻找第二套源文件**——当前 Combine 只有稳定名 dyn-CSR 源；`sv2` 仅保留在 target/符号兼容层。
3. **把 `WorkerDirect` / bypass INC 当成可交付 Dispatch 路径**——违反 H1；资格化固定关闭。
4. **混淆多层 generation**：ABI 常量 / 每 op cacheline 信号 / Framework `operation_generation` 不是同一个变量。
5. **跳过 start-gate / ready 语义直接看 putmem**——同步点才是正确性核心。
6. **手跑二进制绕过 `scripts/single_inc/` 却把结果当正式证据**——会绕过拓扑与空闲门禁。

---

## English

### 0. One sentence

Single-INC implements MoE’s **Dispatch and Combine communication stages** as a
**star-topology data plane** launched in one (or a few) device kernel(s):

- **Dispatch**: workers send tokens to destination workers that own the experts,
  via **one INC** (gather / fan-out).
- **Combine**: workers upload contributions to the INC; the INC **weighted-reduces**
  and **fans results back**.

```text
Logical PE 0 .. W-1  → workers
Logical PE W         → the single INC
World size           → W + 1
```

Qualification scripts currently allow **W ∈ {2,4,8}**. Logical PE ≠ physical NPU id.

### 1. Three maps

**Code**: `common/` (public ABI) · `single_inc/` (this doc) · `scripts/` (device entry) · `tests/` (host gates).

**Call chain**: `Inference → Easy → Framework vtable → composite(native_dispatch + native_combine) → NativeIncService + kernels`.

**Data**: Dispatch moves tokens to experts; Combine returns weighted outputs to source-token layout.

### 2. Dispatch (code-backed)

Kernel entry `inc_dc_single_inc_stream_dispatch_kernel`:

- `pe < workers` → `StreamWorker`
- `pe == workers` → `StreamInc`

Learn ABI objects in `inc_dc_single_inc_stream_abi.h`: `StreamDispatchDesc`, `StreamDispatchTask`, `StreamRouteEntry`, `StreamExpertAssignment`, `generation`, lane counts.

Timeline: **start-gate** → **worker upload via INC** (`HasDirect` tile upload or `WorkerPack`; **not** `WorkerDirect` bypass—H1) → **INC gather + TX** → **completion**. Invariants: no work before `go`; INC must see `tile_ready`/chunk completion before reading; TX depends on gather; workers finish on completion generation.

### 3. Combine (code-backed)

Product path is **dyn-CSR** (`inc_dc_combine_kernel.cpp` / `DynCsrCtrl`). Names like `sv2` / `bw05` are legacy/binary labels, not the public API.

Host pipeline: `IncDcCombineLogicalPlanV2` → topology → `CompileLogicalPlanToExecution` → `DynCsrCtrl` workspace. Framework reverse layout: `BuildCombineReverseLayout` in `planning/`.

Device: producer upload (ready only after payload visibility) → INC owner CSR reduce (vector weighted FP16→FP32, fail-closed) → split TX fan-back → optional device completion.

### 4. Runtime

`native_*_backend` launches the kernels above; `NativeIncService` keeps the INC side resident and rendezvous on generations so workers are not forced through a heavy W+1 host barrier every op. Hot path does not allocate.

### 5. Reading order

This doc → READMEs → stream ABI (full) → `StreamWorker`/`StreamInc` → planning reverse layout → logical plan/topology/compiler → `DynCsrCtrl` → combine kernel control flow → native service/example → Framework/Easy headers. Treat packed/BW05 as background only.

### 6. Minimal device run

From the parent of `shmem/` run `cmake -S shmem -B build …` (or `cmake -S .`
inside `shmem/`). Build targets `inc_dc_single_inc_stream` and
`inc_dc_sv2_dyn_csr_combine`, then qualify **via launchers** (raw binaries
bypass topology/idle gates):

- `scripts/single_inc/run_single_inc_stream_dispatch_case.sh`
- `scripts/single_inc/run_single_inc_dyn_case.sh`

### 7. Misreadings to avoid

No expert GEMM in this tree; the separately built `fusion_kernel/` owns fused
communication and FFN. Do not
treat bw05 or unwired `inc_dc_sv2_*` copies as the Easy hot path. Do not treat
`WorkerDirect` as deliverable (H1). Do not conflate generation layers. Do not
skip start-gate/ready. Do not present raw-bin runs as qualification evidence.
