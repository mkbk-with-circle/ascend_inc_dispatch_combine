# Fusion Kernel API 示例

两个示例都只包含公开头文件 `inc_fusion_api.h`：

| 文件 | 能否直接运行 | 用途 |
|---|---:|---|
| `inc_fusion_plan_smoke.cpp` | 是，不需要 NPU | 创建 prepared plan、读取动态内存预算与 AIV 分组、打印结果并释放 |
| `inc_fusion_runtime_skeleton.cpp` | 默认打印接入提示；真机需实现 `PlatformAdapter` | Worker enqueue、远端 INC 常驻服务、结果读取与释放的完整骨架 |

`PlatformAdapter` 集中承接应用相关的 ACL/SHMEM bootstrap、设备映射、路由、权重和 stream；
示例不构造假设备地址，也不把 Host smoke 当作真机结果。

## 构建与运行 Host 示例

Host smoke 不占用 NPU，但构建仍需 CANN/BiSheng toolchain：

```bash
cmake --build <build-dir> --target \
  inc_fusion_plan_smoke inc_fusion_runtime_skeleton -j

<build-dir>/bin/inc_fusion_plan_smoke
<build-dir>/bin/inc_fusion_runtime_skeleton
```

`inc_fusion_plan_smoke` 的输出形如：

```text
INC Fusion plan ready
  ABI version       : 13
  token waves       : 5
  ...
  worker AIV D/C/FFN: 8/16/24
  INC AIV D/C       : 32/16
  smoke result      : PASS
```

内存和 AIV 数字由当前 ABI、shape 与 live core 数动态计算，应用不得写死。

## 真机接入：完整流程

W+1 个进程使用同一份 `ModelConfig` 和 expert placement；每个进程调用一次
`RunFusionRole()`。

### 1. 实现 `PlatformAdapter::Initialize`

1. 选择当前 PE 对应的 NPU，初始化 ACL device。
2. 初始化 ACLSHMEM world；所有 PE 必须采用相同的 heap capacity 和分配顺序。
3. 创建 Worker stream 或 INC service stream。
4. 查询可用 AIV/AIC 数，写回 `live_aiv`、`live_aic`。

### 2. 实现 `PlatformAdapter::Prepare`

根据 `inc_fusion_prepared_plan_info()` 动态分配：

| 角色 | 必须准备的资源 |
|---|---|
| 所有 PE | `aclshmem_malloc(symmetric_bytes)`、`worker_pes`、expert owner/local index、active token counts、FFTS 地址 |
| Worker | input/output、本地 W13/W2、packed dispatch rows、assignments、waves、group lists；workspace 可由 executor 自动分配 |
| INC | `inc_workspace_bytes` 大小的 workspace、service stream；不持有模型 input/weights |

权重默认是 row-major `[E_local,H,2I]` / `[E_local,I,H]`。对称堆分配后、setup
barrier 前必须将 `symmetric_bytes` 全部清零。

### 3. 两阶段启动 INC

`inc_fusion_remote_service_create()` 只准备 ring。所有对象创建后：

1. 全部 PE 执行一次 `BarrierAll()`；
2. INC 调用 `inc_fusion_remote_service_start()`；
3. Worker 才开始 `inc_fusion_worker_executor_enqueue()`。

顺序不能颠倒：常驻 INC kernel 会占用全部 INC AIV。

### 4. Worker 热路径

每次 forward 只更新动态 bindings。相邻请求的 generation 至少相差
`wave_count + 1`，避免不同请求的 wave generation 别名：

```cpp
inc_fusion_device_bindings_t dynamic{};
dynamic.input = input;
dynamic.output = output;
dynamic.w13 = w13;
dynamic.w2 = w2;
dynamic.dispatch_rows = packed_rows;
dynamic.assignments = packed_assignments;
dynamic.waves = packed_waves;
dynamic.active_token_counts = active_tokens;
dynamic.group_lists = group_lists;

inc_fusion_worker_executor_enqueue(executor, generation, &dynamic, stream);
```

`INC_FUSION_BUSY` 表示 ring slot 仍在途，调用方必须背压而不能覆盖。

### 5. INC 生命周期与结果

INC 通过 descriptor ring 接收请求，执行 Dispatch、等待 FFN、Combine，再处理下一 ticket。
示例按 ticket 查询完成状态；线上通常持续运行到 engine shutdown。

Worker stream 完成后读取 output。checksum 只说明结果可读，正式验证必须对比 BF16 golden。

### 6. 释放顺序

`stop service → destroy service/executor → teardown barrier → destroy plan →
free device/symmetric memory → finalize SHMEM/device`。异常路径应使用框架的 peer-failure
机制，不能等待已经失联的 PE。

## 真实资格化驱动

包含 ACLSHMEM bootstrap、BF16 数据、golden 和多 PE 启动的真机驱动位于：

```text
../ascend/tests/inc_fusion_e2e_main.cpp
../ascend/tests/run_inc_fusion_nb_sweep.sh
```

该驱动比 API 示例长很多，因为它同时承担测试数据生成、正确性校验、性能采样与故障诊断；生产应用通常已有这些上层能力。

---

## English

# Fusion Kernel API examples

Both examples include only the public header `inc_fusion_api.h`:

| File | Runnable as-is | Purpose |
|---|---:|---|
| `inc_fusion_plan_smoke.cpp` | Yes, no NPU | Create a prepared plan, read dynamic memory budget and AIV grouping, print, destroy |
| `inc_fusion_runtime_skeleton.cpp` | Prints integration hints by default; device use needs a `PlatformAdapter` | Full skeleton for worker enqueue, remote INC resident service, result read, and teardown |

`PlatformAdapter` owns application ACL/SHMEM bootstrap, device mapping, routes,
weights, and streams. The examples do not invent fake device addresses, and
host smoke is not a device result.

## Build and run the host examples

Host smoke does not occupy an NPU, but the build still needs CANN/BiSheng:

```bash
cmake --build <build-dir> --target \
  inc_fusion_plan_smoke inc_fusion_runtime_skeleton -j

<build-dir>/bin/inc_fusion_plan_smoke
<build-dir>/bin/inc_fusion_runtime_skeleton
```

`inc_fusion_plan_smoke` prints something like:

```text
INC Fusion plan ready
  ABI version       : 13
  token waves       : 5
  ...
  worker AIV D/C/FFN: 8/16/24
  INC AIV D/C       : 32/16
  smoke result      : PASS
```

Memory and AIV numbers are computed from the current ABI, shape, and live core
count. Applications must not hard-code them.

## Device integration: full flow

W+1 processes share one `ModelConfig` and expert placement; each process calls
`RunFusionRole()` once.

### 1. Implement `PlatformAdapter::Initialize`

1. Select the NPU for this PE and init the ACL device.
2. Init the ACLSHMEM world; every PE must use the same heap capacity and
   allocation order.
3. Create the worker stream or INC service stream.
4. Query live AIV/AIC counts and write back `live_aiv` / `live_aic`.

### 2. Implement `PlatformAdapter::Prepare`

Allocate from `inc_fusion_prepared_plan_info()`:

| Role | Required resources |
|---|---|
| All PEs | `aclshmem_malloc(symmetric_bytes)`, `worker_pes`, expert owner/local index, active token counts, FFTS address |
| Worker | input/output, local W13/W2, packed dispatch rows, assignments, waves, group lists; workspace may be allocated by the executor |
| INC | workspace of `inc_workspace_bytes`, service stream; does not hold model input/weights |

Default weight layout is row-major `[E_local,H,2I]` / `[E_local,I,H]`. After
symmetric-heap allocation and before the setup barrier, zero all
`symmetric_bytes`.

### 3. Two-phase INC start

`inc_fusion_remote_service_create()` only prepares the ring. After all objects
exist:

1. Every PE runs `BarrierAll()` once;
2. INC calls `inc_fusion_remote_service_start()`;
3. Workers may then `inc_fusion_worker_executor_enqueue()`.

Do not reverse this: the resident INC kernel occupies all INC AIVs.

### 4. Worker hot path

Each forward only updates dynamic bindings. Adjacent requests must differ in
generation by at least `wave_count + 1` to avoid wave-generation aliasing:

```cpp
inc_fusion_device_bindings_t dynamic{};
dynamic.input = input;
dynamic.output = output;
dynamic.w13 = w13;
dynamic.w2 = w2;
dynamic.dispatch_rows = packed_rows;
dynamic.assignments = packed_assignments;
dynamic.waves = packed_waves;
dynamic.active_token_counts = active_tokens;
dynamic.group_lists = group_lists;

inc_fusion_worker_executor_enqueue(executor, generation, &dynamic, stream);
```

`INC_FUSION_BUSY` means a ring slot is still in flight. The caller must
back-pressure, not overwrite.

### 5. INC lifetime and results

INC receives requests from the descriptor ring, runs Dispatch, waits for FFN,
Combines, then takes the next ticket. The example queries completion by ticket;
production usually runs until engine shutdown.

Read output after the worker stream completes. A checksum only shows the result
is readable; formal checks must compare a BF16 golden.

### 6. Teardown order

`stop service → destroy service/executor → teardown barrier → destroy plan →
free device/symmetric memory → finalize SHMEM/device`. Exception paths should
use the framework peer-failure mechanism and must not wait for a lost PE.

## Real qualification driver

The device driver with ACLSHMEM bootstrap, BF16 data, golden, and multi-PE
launch lives at:

```text
../ascend/tests/inc_fusion_e2e_main.cpp
../ascend/tests/run_inc_fusion_nb_sweep.sh
```

That driver is much longer than the API examples because it also generates
test data, checks correctness, samples performance, and diagnoses faults.
Production apps usually already have those layers.
