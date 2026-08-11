# 多 INC dispatch/combine 跨昇腾环境移植

## 当前结论

当前锁定实现不能直接宣称在任意昇腾硬件上无缝运行。数据协议已经能够表达
动态 W/K/route，但启动与资源层仍混有当前机器假设：

- 历史 AIV profile 默认 switch/worker 各有 40 AIV；
- 部分 overlap profile 固定划分 12/24 或 16/24；
- dispatch pipeline ABI 固定了 8 upload lane、8 receive lane，以及
  10/17/20-block specialization；
- dyn combine 当前限制 owner≤20、hidden≤7168；
- 通用类型仍有 W≤16、TopK≤16 的历史上限；
- 192 GB/s 是当前环境审计线，不是所有昇腾硬件的物理上限；
- kernel fat binary、CANN runtime symbol、SHMEM transport 和内存对齐能力会
  随 SoC/CANN 变化。

因此正确策略不是根据产品名散落 `if (910B)`，而是把“硬件能力”和“编译后
kernel 能力”分别建模，再求两者交集。

## 已补充的 portability 层

公共 ABI：

`examples/inc/dispatch_combine/portability/inc_dc_portability.h`

它定义三类对象：

1. `inc_dc_hardware_caps_t`
   - SoC/CANN、AIV/AIC、UB、block dim、stream 数；
   - device event、persistent kernel、dynamic block dim、RDMA/UDMA 等；
   - 当前硬件自己的物理带宽上限和 profile generation。
2. `inc_dc_kernel_portability_t`
   - 某个具体 kernel artifact 的最小/首选/最大 block 数；
   - 是否允许弹性缩容；
   - min/max world、max top-k/owner/producer lanes；
   - 最低 CANN、UB 和必需 runtime feature。
3. `inc_dc_portable_launch_plan_t`
   - 在本机真正可启动的 dispatch/combine block slice；
   - owner、producer lane 和 topology/hardware generation；
   - 是否相对首选配置降级。

planner 的 fail-close 规则：

- 设备 AIV 多，不代表固定 40-block kernel 自动支持更多 block；
- 设备 AIV 少，只有声明 elastic 的 kernel 才能缩容；
- 同时请求 dispatch+combine、但不要求 overlap 时，声明
  `INC_DC_KERNEL_TIME_MULTIPLEX_BLOCKS` 的 artifact 可以让两条路径复用同一
  AIV block pool；吞吐会降低，但不因两套最小 block 数相加而失去功能；
- 要求 overlap 时始终使用空间隔离的 block slice，AIV 不足会明确返回
  `RESOURCE_EXHAUSTED`，不会用时分复用冒充并发；
- overlap 必须有 external stream 和 device event；
- CANN、UB、world 上下界、top-k、owner 或 producer lane 不满足时返回明确原因；
- 未知物理上限的 profile 可以跑 correctness，但不能晋升性能；
- 带宽审计读取 profile 上限，不再把 192 GB/s 写成跨设备常量。

`inc_dc_acl_discovery.h/.cpp` 还提供了一个不直接链接特定 CANN ABI 的 ACL
discovery provider。它通过 `dlopen/dlsym` 查询 device 数、AIV/AIC、UB、HBM、
SoC、CANN 和 BF16/event/stream symbol，不调用 `aclInit`、不设置 device，也
不分配或启动 kernel。若宿主进程尚未初始化 ACL，或旧 CANN 不提供相应查询，
它返回 `UNSUPPORTED`，要求使用有证据的 profile overlay；不会回退到猜测值。
`inc_dc_merge_hardware_profile` 负责合并两类证据：runtime 已读出的 AIV/AIC/
device 数不能被 profile 覆盖，只有 UB、SoC、CANN、transport 等缺失字段可以
补齐；任一已知字段冲突都会使 plan 创建失败。

硬件 profile schema：

- `docs/inc/configs/inc_dc_hardware_profile.schema.json`
- `docs/inc/configs/inc_dc_hardware_profile.template.json`

模板中的 AIV/UB/transport/带宽不能按产品名猜测，必须来自 runtime、厂商规格
或本机 qualification gate。

## 接入 device backend 的顺序

### 1. 初始化期发现硬件

框架 plugin 初始化时读取 capability profile，并用 runtime 结果核对：

- 可见 device 数和实际 device ID；
- CANN 版本与动态 symbol；
- AIV/AIC/UB 和最大 block dim；
- stream/event、graph capture；
- SHMEM RDMA/UDMA/fabric；
- HBM、large-page 和 workspace alignment；
- rank/NPU/NUMA/link topology。

关键字段缺失时不得使用默认 40 AIV。profile 与 runtime 不一致时必须递增
hardware generation、使旧 plan stale。

### 2. 每个 artifact 注册 kernel metadata

dispatch、combine 和 overlap SO 的 manifest 必须携带
`inc_dc_kernel_portability_t` 等价信息。当前冻结 SHA 只能绑定当前已验证 profile；
换 SoC 或 CANN 后应从 artifact registry 选择匹配的 SO，不能复用文件名猜测。

### 3. plan 创建时求交集

Megatron/vLLM 创建 EP plan 时调用 portability planner。生成的 block slice、
owner 和 producer lanes 写入 device plan；enqueue 只消费已验证 plan，不在热
路径解析环境变量或重新发现硬件。

### 4. 分层 qualification

新机器第一次接入依次执行：

1. C ABI/profile/schema host gate；
2. 单 kernel device event、memory visibility、signal/atomic smoke；
3. W2/W4/W8 与 K>W correctness；
4. 多 stream dispatch/combine overlap；
5. 本机 native baseline；
6. 3+20 formal 和 100-window soak；
7. 生成该 profile 独立的 physical ceiling 和 performance floor。

旧机器的 P100、baseline 和 192 GB/s 上限不得复制为新机器的 PASS 证据。

## 还没有完成的设备工作

本轮完成的是 capability ABI、资源 planner、profile schema 和 host 合同测试，
没有修改冻结内核，也没有在另一型号 NPU 上执行。要达到真正“快速接入”，仍需：

- 在统一 device backend 初始化中接入 hardware discovery provider；
- 给锁定 dispatch/combine SO 补 artifact-level kernel metadata；
- 把历史 `kIncDcDefault*40` 路径迁到 planner，旧 profile 仅保留兼容模式；
- 将 W/TopK/owner/lane 静态数组改为 descriptor + workspace sizing，才能扩到
  W>16；
- 按 SoC arch 生成独立 fatbin，做 CANN symbol/version negotiation；
- 对不同 atomic/signal、cache visibility、RDMA/UDMA 路径分别跑 device gate；
- 在目标机器重新建立 baseline、物理上限和性能曲线。

在这些设备门通过前，portability 状态应为
`host_contract_pass=true, cross_device_proven=false`。

## 通信库初始化模型

`inc_dc_runtime_api.h` 提供可由业务代码直接调用的显式生命周期：

```text
process start
  └─ inc_dc_runtime_create()          非 collective，发现硬件/合并 profile
       └─ inc_dc_runtime_register_kernel()  注册可用 dispatch/combine SO
            └─ inc_dc_session_create()      每个 EP group 一次，control plane
                 ├─ dispatch_async()        每次操作，仅 enqueue
                 ├─ combine_async()         每次操作，仅 enqueue
                 └─ inc_dc_session_destroy()
       └─ inc_dc_runtime_destroy()
```

显式初始化是合理且必要的，但应用不应在每次操作或为每个 INC owner 手工初始化：

- runtime 通常每个进程创建一次；
- session 每个 model/EP process group/topology generation 创建一次；
- `session_create` 的 backend callback 内部完成 SHMEM bootstrap、worker/INC
  endpoint 映射、persistent service 启动和 readiness；
- worker-facing Megatron/vLLM API 只看到 process group/session，不看到内部
  INC PE；
- topology 改变时销毁并重建 session，普通动态 token/top-k 不重建 runtime；
- enqueue 热路径不得初始化、分配、解析 profile 或进行设备级同步。

runtime 有 artifact registry。每个 SO 注册自己的 SoC、SHA、优先级和 kernel
portability metadata；同一业务二进制在 40 AIV 环境可选择固定高性能
specialization，在 32 AIV 环境会拒绝该 artifact 并选择已验证的 elastic
fallback。在 20 AIV 环境，若不要求 overlap，还可选择经过验证的
time-multiplex fallback。planner 本身没有隐藏的 40-AIV 下限：若注册了
min-block=1 的 scalar fallback，单 AIV 也能生成正确的功能 plan。找不到合法
artifact 时返回 `UNSUPPORTED/RESOURCE_EXHAUSTED`。

这一区分很重要：portability planner 能为任意实际 AIV 数量选择合法 plan，
不等于现有冻结 kernel 已自动变成弹性 kernel。真正的跨设备发行包至少应包含：

1. 当前硬件上的高性能固定 specialization；
2. 覆盖常见 20/24/32/40/64 AIV 的 elastic specialization；
3. 一个低性能但功能完整的 scalar/time-multiplex fallback。

每个 fallback 都必须经过对应 SoC/CANN 的 device correctness gate；仅修改
metadata 不能证明 kernel 支持缩容。

当前 910B 锁定 lineage 已由
`inc_dc_register_qualified_910b_artifacts()` 注册为角色隔离的 specialization：

- Worker dispatch：C3–C17 弹性布局；
- INC dispatch：W8 优先 P5 C20；W2/W4 自动选择已验证的 legacy C10；
  低 AIV 同样回退 legacy C10；
- Worker combine producer：C1–C8；
- INC combine owner：C1–C20，首选 C14；
- session 同时选择独立 dispatch/combine SO，并联合求解 AIV 预算；若首选
  artifact 组合放不下，会回溯选择次优组合，而不是直接失败。

注册信息绑定当前锁定 SHA，且明确限定 `dav-2201`、CANN 9.0、W≤8/K≤8。
它不会把当前 artifact 误标为 W>8 或任意 SoC 可用。

当前已完成 runtime/session C ABI、锁定 kernel metadata 注册和 mock backend
生命周期测试；metadata 尚未由 production backend 在设备上消费和验证，
production backend callback 也尚未绑定，因此不会误宣称已经启动当前 INC
服务。

业务侧的最小调用方式如下（省略错误处理和 artifact 字段填充）：

```c
inc_dc_runtime_t *runtime = NULL;
inc_dc_runtime_config_t runtime_cfg = {0};
runtime_cfg.struct_size = sizeof(runtime_cfg);
runtime_cfg.abi_version = INC_DC_RUNTIME_ABI_VERSION;
runtime_cfg.device_id = local_device_id;
runtime_cfg.qualified_profile = &qualified_profile;
inc_dc_runtime_create(&runtime_cfg, &runtime);

inc_dc_runtime_register_kernel(runtime, &dispatch_combine_artifact);

inc_dc_session_t *session = NULL;
inc_dc_session_create(runtime, &ep_group_desc, &device_backend, &session);
/* 后续 dispatch/combine 只通过 session 异步 enqueue。 */

inc_dc_session_destroy(session);
inc_dc_runtime_destroy(runtime);
```

若业务希望由通信库统一管理上述对象，可直接使用
`inc_dc_client_create()`：它一次完成 runtime 创建、artifact 注册、双路径
session 初始化以及 framework context 创建；随后从
`inc_dc_client_get_framework_context()` 取得 context，正常创建 plan 并调用
`inc_dc_fw_dispatch_async/inc_dc_fw_combine_async`。销毁顺序由
`inc_dc_client_destroy()` 保证，仍有 live plan/request 时返回 `BUSY`。

framework enqueue 已改为先预留 request slot，再在不持有 context 全局锁的
情况下调用 backend。一个 stream 的 backend enqueue 变慢时，不会阻塞另一
stream 的 dispatch/combine enqueue；失败会回滚 slot、plan inflight 和
session 生命周期计数。

因此从通信库使用习惯看，用户需要显式创建 runtime/session，但不需要显式
初始化每个内部 INC rank，也不应在每次 dispatch/combine 前初始化。Megatron
或 vLLM adapter 应在 process-group 创建/销毁时封装 `session_create/destroy`。

## 构建与复现

该模块使用独立 CMake，避免影响正在清理的历史 target：

```bash
cmake -S examples/inc/dispatch_combine/portability \
      -B build/inc_dc_portability
cmake --build build/inc_dc_portability -j4
timeout 30s build/inc_dc_portability/inc_dc_portability_tests
timeout 30s build/inc_dc_portability/inc_dc_runtime_api_tests
timeout 30s build/inc_dc_portability/inc_dc_client_api_tests
timeout 30s build/inc_dc_portability/inc_dc_portability_c_header_tests
```

工程内可通过 `inc_dc::portability` target 链接；也可用
`cmake --install build/inc_dc_portability --prefix <prefix>` 安装
`libinc_dc_portability.so.1` 与 `include/inc_dc/*.h`，供业务 C/C++ adapter
调用。

测试覆盖 1/20/24/32/40/64 AIV、固定/弹性/time-multiplex kernel、
CANN/UB/feature 缺失、W2/K8、W32/K16、W8-only specialization 在 W2/W4
自动回退，以及 192 与 400 GB/s 两种设备上限。
