# SHMEM 初始化与终止流程

> 本文说明 SHMEM 初始化与终止的流程、参数与源码实现。除下文列出的对外接口外，其余函数为内部实现，用户不应直接调用。

**阅读建议**：快速了解读第一、二章；实现细节读第三、四章并对照 `src/host/init/shmem_init.cpp`；Default/UID **Config Store 控制面**（TCP 建链、KV 协议、socket 绑定）详见 [config_store_bootstrap.md](config_store_bootstrap.md)；查函数落点见第五章；排障见第六章。

---

## 一、概述

### 1.1 流程概要

各 PE 对称调用 `aclshmemx_init_attr` 完成初始化，对称调用 `aclshmem_finalize` / `aclshmemx_finalize` 完成终止。须满足：① 每个 PE 均参与 init/finalize；② 除 `my_pe` 外 init 参数一致；③ 各 rank 的 `aclshmem_malloc` / `aclshmem_free` 同序同大小。

初始化依次完成：

1. **Bootstrap（控制面，CPU）**：建立 rank 间连接（Config Store/TCP 或 MPI），通过内部 `barrier` / `allgather` 交换建堆元数据并同步进度。
2. **HYBM 建堆（数据面，NPU）**：`reserve_heap` 预留 GVA，`setup_heap` 分配内存、交换 slice/entity 并 `mmap`。此后同序同大小 `malloc` 时堆内偏移一致，可进行 `aclshmem_putmem` / `aclshmem_getmem` 等 RMA。
3. **子模块就绪**：堆分配器、team、signal、平台同步等完成，应用可调用 SHMEM API。

终止按相反顺序释放：子模块 → remove/release 对称堆 → `finalize_device_state` → `release_aclshmem_entity` → ACL stream → **`aclshmemi_control_barrier_all`（控制面集合同步）** → Bootstrap → `init_manager`。

Bootstrap 内部 `barrier` 用于 init/finalize 控制流程，不同于运行期 `aclshmem_barrier`（team 级业务同步）。多 instance 时 `init_manager`、HYBM 全局初始化、Config Store 由引用计数共享，最后一个 instance finalize 后释放。

### 1.2 核心概念

| 概念 | 说明 |
|------|------|
| **PE / rank** | 参与 SHMEM 的进程；`my_pe` 为本 PE 编号（从 0 起），`n_pes` 为 PE 总数。 |
| **对称堆** | 各 PE 各有一块等长堆；同序同大小的 `malloc` 得到相同堆内偏移。 |
| **Bootstrap** | init 阶段主机侧通信层，用于建链、交换元数据、保证 `malloc` 顺序一致。 |
| **HYBM** | 底层混合内存库；**entity** 为内存域句柄，**slice** 为物理内存，**GVA** 为预留虚拟地址。 |
| **instance** | 同一进程内多套 SHMEM 上下文（`instance_id`，默认 0）。 |
| **host / device state** | host 维护 `g_state`；算子读同步到 NPU 元数据区的 device 副本。 |

### 1.3 典型用法

**UID 模式**（无 MPI、不写 `ip_port`）：

```bash
rank 0: aclshmemx_get_uniqueid(&uid)          // 约定 rank 0 生成，见 examples/init
rank 0: 广播 uid                              // 库不负责广播
各 rank: aclshmemx_set_attr_uniqueid_args(my_pe, n_pes, heap_size, &uid, &attr)
各 rank: aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr)
各 rank: aclshmem_finalize()
```

**MPI 模式**：`aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_MPI, &attr)`；`comm_args` 为 `MPI_Comm*`（`nullptr` 即 `MPI_COMM_WORLD`）。`attr` 中的 `my_pe` / `n_pes` 须与 `MPI_Comm_rank` / `MPI_Comm_size` 一致（二者分别写入 `g_state` 与 Bootstrap 句柄，不一致会导致后续行为异常）。

**Default 模式**（已知 rank 0 地址）：

```bash
各 rank: attr.ip_port = "tcp://rank0_host:port"，并填 my_pe / n_pes / local_mem_size
各 rank: aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr)
各 rank: aclshmem_finalize()
```

完整样例见 `examples/init`。多 instance 见 [multi_instance.md](multi_instance.md)。

### 1.4 初始化状态

| `aclshmemx_init_status()` | 含义 |
|---------------------------|------|
| `ACLSHMEM_STATUS_NOT_INITIALIZED` | `is_aclshmem_created == false` |
| `ACLSHMEM_STATUS_SHM_CREATED` | 堆已建，子模块初始化中（中间态） |
| `ACLSHMEM_STATUS_IS_INITIALIZED` | 可调用 `aclshmem_malloc`、RMA、team 等 |

同一 `instance_id` 不可重复 init：`setup_heap` 置 `is_aclshmem_created = true` 后不得再次调用 `aclshmemx_init_attr`。

### 1.5 终止约束

1. 每个 PE 均须 finalize。
2. finalize 前等待 device kernel / stream 完成；各 PE 须对称进入 finalize，以便 Bootstrap 拆除前经 `aclshmemi_control_barrier_all` 在 Config Store 上对齐（见 §4.2）。
3. finalize 后不得再调用 SHMEM API。

---

## 二、对外接口速览

### 初始化

| 接口 | 说明 |
|------|------|
| `aclshmemx_get_uniqueid` | 约定 rank 0 调用；生成 UID（含 `addr`、`magic`）。 |
| `aclshmemx_set_attr_uniqueid_args` | `(my_pe, n_pes, local_mem_size, uid, attr)`。 |
| `aclshmemx_init_attr` | 主入口，所有 PE 对称调用。 |
| `aclshmemx_init_status` | 查询初始化阶段。 |
| `aclshmemx_instance_ctx_get` / `set` | 多 instance 获取/切换上下文。 |
| `aclshmem_info_get_version` / `get_name` | 库版本与名称。 |

### 终止

| 接口 | 说明 |
|------|------|
| `aclshmem_finalize` | 释放当前活动 instance。 |
| `aclshmemx_finalize` | 按 `instance_id` 释放。 |

### 算子内辅助

| 接口 | 说明 |
|------|------|
| `util_get_ffts_config` / `util_set_ffts_config` | A2/A3 算子内 FFTS 基址；950 可不调用（§3.3.7.4）。 |

### 初始化前可选配置

| 接口 | 说明 |
|------|------|
| `aclshmemx_set_config_store_tls_key` / `aclshmemx_set_conf_store_tls` | Config Store TLS 配置。 |

---

## 三、初始化详解

### 3.1 总览

入口为 `aclshmemx_init_attr`（`shmem_init.cpp`，互斥锁保护）。下图与下表对应 §3.3 各步；细节见 §3.2（参数）与 §3.3（分步）。

![image](images/initialization/init_attr_detailed_flow.png)

| 步骤 | 源码调用 | 文件 |
|------|----------|------|
| 1 多实例 | `aclshmemi_instance_ctx_create` → `aclshmemx_instance_ctx_set_impl` | `shmem_init.cpp` |
| 2 校验 | `aclshmemx_init_status`（须 `NOT_INITIALIZED`）→ `set_log_level` → `check_attr` → `version_compatible` | `shmem_init.cpp` |
| 3 Bootstrap | `aclshmemi_bootstrap_init` | `shmemi_bootstrap.cpp` |
| 4 State | `aclshmemi_state_init_attr` | `shmem_init.cpp` |
| 5 Backend | `new aclshmemi_init_backend()`（`g_init_manager_count++`） | `shmem_init.cpp` |
| 6 堆 | `bind_aclshmem_entity` → `init_device_state` → `reserve_heap` → `setup_heap` | `shmem_init_backend.cpp` |
| 7 子模块 | `memory_manager_initialize` →（可选 `reserve_heap(HOST)`）→ `aclshmemi_signal_init` → `aclshmemi_team_init` → `aclshmemi_sync_init` | `shmem_init.cpp` 等 |
| 8 收尾 | `is_aclshmem_initialized = true` → `prof_util_init` → `update_device_state` → `aclshmemi_control_barrier_all` | `shmem_init.cpp` |

`setup_heap` 末尾 `is_aclshmem_created = true`；步骤 8 开头 `is_aclshmem_initialized = true`。`instance_id != 0` 首次创建失败时，scope guard 调用 `aclshmemi_instance_ctx_destroy(id)`；进入 init 时状态须为 `ACLSHMEM_STATUS_NOT_INITIALIZED`。

### 3.2 参数与 Bootstrap

各 PE 在 `aclshmemx_init_attr_t` 中填写 `my_pe`、`n_pes`、`local_mem_size`，并选择 Bootstrap 模式。库实际映射 `local_mem_size + ACLSHMEM_EXTRA_SIZE`（`g_state.heap_size`）。

**Bootstrap 模式**

| 模式 | 需准备 |
|------|--------|
| `ACLSHMEMX_INIT_WITH_DEFAULT` | 合法 `ip_port`，或 fallback 到 UID |
| `ACLSHMEMX_INIT_WITH_UNIQUEID` | rank 0 生成 UID 并广播 |
| `ACLSHMEMX_INIT_WITH_MPI` | `MPI_Comm*`（仅 `MPI_COMM_WORLD` / `MPI_COMM_SELF`）；`attr.my_pe`/`n_pes` 与 MPI rank/size 一致 |

Default/UID 走 `aclshmem_bootstrap_config_store.so`；MPI 走 `aclshmem_bootstrap_mpi.so`。

![image](images/initialization/bootstrap_modes_comparison.png)

*图：`aclshmemi_bootstrap_init` 按 `bootstrap_flags` 三分支选择插件；Default 先校验 `ip_port`，无效时 fallback 到 UID。*

Default / UID 选定 `aclshmem_bootstrap_config_store.so` 后，各 PE 的 CPU 进程以 PE 0 为中心建立星型连接。PE 0 运行 `AccStoreServer` 维护全局键值表，各 PE（含 PE 0）经同一 `ConnectToPeerServer` 建 client；`SmemNetGroupEngine` 在键值表上实现 init 阶段的 `barrier` / `allgather`。该控制面建链与后续 NPU 对称堆（HYBM `exchange_slice`）相互独立。

> **深入阅读**：[config_store_bootstrap.md](config_store_bootstrap.md) 基于源码详解 Config Store 分层、**监听/连接的 IP:port**、Acclink 握手、KV 消息帧格式与 barrier/allgather 键协议（仅 CPU 控制面，不涉及 NPU）。

![image](images/initialization/connection_between_ranks.png)

*图：Default/UID 模式下以 PE 0 为中心的星型拓扑；**含 PE 0 client** 在内的各 PE 均经 `ConnectToPeerServer` 连向 PE 0 listener。*

UID 的 `addr` 来自 `SHMEM_UID_SESSION_ID` 或 `SHMEM_UID_SOCK_IFNAME`；`magic` 为随机通信域标识符。全体 rank 的 UID 须字节级一致。

**超时参数**

| 参数 | 含义 | 单位 / 说明 |
|------|------|-------------|
| `shm_init_timeout` | Default 且 `ip_port` 合法时，**非 PE 0** client 连 PE 0 的上限 | API 字段名为超时（秒），默认 120；Config Store **实现**将其作为 `ConnectToPeerServer` **重试次数**（每次失败 `sleep(1)`），最坏约 N 秒，详见 [config_store_bootstrap.md §10.1](config_store_bootstrap.md#101-超时汇总) |
| `control_operation_timeout` | barrier/allgather 等待上限 | 秒 |
| UID / Default→UID `timeOut` | 同 `shm_init_timeout` 的建链语义 | 固定 `DEFAULT_TIMEOUT`（120），作重试次数 |
| UID / Default→UID `timeControlOut` | 控制面集合通信超时 | 秒（×1000 → `timeoutMs`） |
| Acclink 收包 | `ACC_LINK_RECV_TIMEOUT` | 秒（1800，常量） |

**`aclshmemx_init_attr_t` 关键字段**

| 字段 | 说明 |
|------|------|
| `my_pe` / `n_pes` | 本 PE 编号与 PE 总数，`0 <= my_pe < n_pes` |
| `local_mem_size` | 每 PE 用户堆大小 |
| `ip_port` | Default：`tcp://host:port` 或 `tcp6://[host]:port`，端口 > 1024 |
| `comm_args` | Default/UID：`aclshmemx_uniqueid_t*`；MPI：`MPI_Comm*` |
| `instance_id` | 多 instance 标识，默认 0 |
| `option_attr.data_op_engine_type` | 传输通路位或（见下表） |
| `option_attr.shm_init_timeout` | 传入 `CreateStore` 的 `connMaxRetry`（实现为重试次数，见 [config_store_bootstrap.md §10.1](config_store_bootstrap.md#101-超时汇总)） |
| `option_attr.control_operation_timeout` | 转为 `SmemGroupOption.timeoutMs` |
| `option_attr.sockFd` | rank 0 预绑定 socket |

**传输通路标志**

| 标志 | 通路 |
|------|------|
| `ACLSHMEM_DATA_OP_MTE` | MTE |
| `ACLSHMEM_DATA_OP_SDMA` | SDMA |
| `ACLSHMEM_DATA_OP_ROCE` | RDMA |
| `ACLSHMEM_DATA_OP_UDMA` | UDMA |

`setup_heap` 后 `reach_info_init` 写入 `topo_list`，供 RMA 选路。

**Config Store 结构**（Default/UID，概要）

`aclshmemi_bootstrap_init` → `dlopen` → `plugin_init` → `init_config_store` → `init_group_engine`，填充 `g_boot_handle.barrier` / `allgather`。socket 绑定、连接目标、消息校验等细节见 [config_store_bootstrap.md](config_store_bootstrap.md)。

| 层次 | 职责 |
|------|------|
| `TcpConfigStore` | 各 rank client，`Get`/`Set`/`Add`/`Append` |
| `AccStoreServer`（rank 0） | 全局键值表 |
| `Acclink` | TCP 建链、握手（`AccConnReq` 含 `rankId`，magic/version 校验）、可选 TLS |
| `SmemNetGroupEngine` | 键值上实现 `GroupBarrier` / `GroupAllGather` |

rank 0 `CreateStore(..., isServer=true)` 监听；各 PE client 的 `connMaxRetry`：PE 0 固定 60 次，非 PE 0 为 `shm_init_timeout`（重试次数）。多 instance 共享 `g_store_ref` 引用计数。Bootstrap `barrier` 用于 `exchange_slice` 同步与 `aclshmem_malloc` 同序约束。

**运行时产物**：`libshmem.so`、`libshmem_utils.so`、`aclshmem_bootstrap_config_store.so`、`aclshmem_bootstrap_mpi.so`。

### 3.3 分步说明

以下按 §3.1 调用序展开。

#### 3.3.1 多实例上下文

同一进程可通过 `instance_id` 维护多套 SHMEM 上下文。多 instance 共享 `init_manager` 与 HYBM 全局初始化，各有独立堆与 Bootstrap 句柄。

- `aclshmemi_instance_ctx_create`（仅 `instance_id != 0`）
- `instance_id != 0` 且（`ip_port` 非空 **或** `comm_args == nullptr`）时调用 `aclshmemi_instance_port_selection`：URL 模板端口须为 **0**，按 `SHMEM_INSTANCE_PORT_RANGE` 填入 `start_port + instance_id`；纯 UID（`ip_port` 为空且 `comm_args` 有效）跳过
- `aclshmemx_instance_ctx_set_impl` 切换 `g_state`、`g_boot_handle` 等

详见 [multi_instance.md](multi_instance.md)。

#### 3.3.2 配置校验

- `aclshmemx_init_status()` 须为 `ACLSHMEM_STATUS_NOT_INITIALIZED`（`is_aclshmem_created == false`），禁止重复 init
- `aclshmemx_set_log_level(ERROR_LEVEL)`；`SHMEM_LOG_LEVEL` 可覆盖
- `check_attr`：PE 范围、`local_mem_size`、超时非 0、`data_op_engine_type` bitmask
- `version_compatible`

#### 3.3.3 Bootstrap 加载

- `aclshmemi_bootstrap_init`：`dlopen` → `plugin_init` → `is_bootstraped = true`
- 模式分支见 §3.2

#### 3.3.4 State 初始化

- `aclshmemi_state_init_attr`：`g_state.mype` / `npes` / `heap_size`；`aclrtCreateStream` → `default_stream`

#### 3.3.5 Backend

- `g_init_manager_count++`；`init_manager == nullptr` 时 `new aclshmemi_init_backend()`，否则复用已有实例
- `-mssanitizer` 构建且以 `mssanitizer --` 运行时，可向检测工具注册元数据区与堆 GVA

#### 3.3.6 Entity 与对称堆

调用链：`bind_aclshmem_entity` → `init_device_state` → `reserve_heap` → `setup_heap`。

![image](images/initialization/memory_heap_initialization.png)

##### 3.3.6.1 `bind_aclshmem_entity`

在 `entity_map_` 登记 `entity_member`，绑定 init 参数、`g_state`、Bootstrap 句柄、device state 镜像。

##### 3.3.6.2 `init_device_state`

- `hybm_init_count == 0` 时 `hybm_init`，映射元数据区（约 32MB，非用户堆）
- `hybm_init_count++`；最后一个 finalize 时 `hybm_uninit`
- `aclshmemi_control_barrier_all()`

##### 3.3.6.3 `reserve_heap`

1. `create_entity` → `hybm_create_entity`（`HYBM_ROLE_PEER`）
2. `hybm_reserve_mem_space` → `hbm_gva`
3. （可选）MSTX 注册 GVA

##### 3.3.6.4 `setup_heap`

1. `hybm_alloc_local_memory` → slice
2. `exchange_slice`：`hybm_export` → `allgather` → `hybm_import` → `barrier`
3. `exchange_entity`：流程同 `exchange_slice`；`hybm_export` 返回 `descLen == 0` 时跳过 allgather/import
4. `hybm_mmap`
5. 分配 `p2p_*` / `rdma_*` / `sdma_*` 堆基址数组（UDMA 复用 P2P）
6. `reach_info_init` → `topo_list`
7. `heap_base = hbm_gva + ALIGN_UP(heap_size, ACLSHMEM_HEAP_ALIGNMENT_SIZE) * my_pe`
8. `is_aclshmem_created = true`

##### 3.3.6.5 host/device state 同步

算子读 device 元数据区副本。`update_device_state()` 拷贝 `g_state` → `entity_device_state`，保留 heap 基址指针，再 `hybm_set_extra_context`。

![image](images/initialization/host_device_state_sync.png)

#### 3.3.7 子模块

##### 3.3.7.1 共享内存管理器

在 `heap_base` 上 first-fit 管理对称堆；每次 `aclshmem_malloc` / `aclshmem_free` 后 `aclshmemi_control_barrier_all`。

- `memory_manager_initialize(heap_base, heap_size)`；DEBUG 模式检 `is_alloc_size_symmetric`
- d2h：init 时 `reserve_heap(HOST_SIDE)`；首次 host malloc 时 lazy `setup`

![image](images/initialization/memory_manager_initialization.png)

*图：对称堆上的 first-fit 分配器；`aclshmem_malloc` 按 First Fit 切分空闲块，`aclshmem_free` 归还并合并相邻空闲块。*

各 rank 还须保持**相同顺序、相同大小**的 `malloc`/`free`，才能保证堆内偏移对称。下图对比合法与非法的分配模式：

![image](images/initialization/malloc.png)

*图：对称堆内偏移对齐示意（正确 vs 错误），非 malloc 调用序列图。*

##### 3.3.7.2 Signal

- `aclshmemi_signal_init`：`aclshmem_malloc(512)` → `signal_addr`；清零 `ACLSHMEM_SIGNAL_SIZE`（8 字节）

##### 3.3.7.3 Team

- `aclshmemi_team_init`：`ACLSHMEM_TEAM_WORLD`；`team_pool`；`sync_pool` / `sync_counter`；`core_sync_*`

![image](images/initialization/team_management_initialization.png)

*图：卡间/核间 Sync Counter 与 Sync Pool 状态结构；`team_pool` 等见上文列表。*

详见 [team.md](team.md)。

##### 3.3.7.4 同步（FFTS）

| 平台 | 行为 |
|------|------|
| A2/A3 | init 时 `rtGetC2cCtrlAddr`；算子入口 `util_set_ffts_config(util_get_ffts_config())` |
| Ascend950 | 不定义 `FFTS_CONFIG`；由 team 同步池完成 barrier |

![image](images/initialization/sync_management_initialization.png)

##### 3.3.7.5 收尾

`is_aclshmem_initialized = true` → `prof_util_init` → `update_device_state` → `aclshmemi_control_barrier_all` → 释放 scope guard

---

## 四、终止（Finalize）详解

### 4.1 总览

`aclshmem_finalize` / `aclshmemx_finalize` 按与 init 相反的顺序释放资源。多 instance 时 `init_manager`、`hybm_init`、`g_store_ref` 由引用计数管理。

| 接口 | 行为 |
|------|------|
| `aclshmem_finalize()` | finalize 当前活动 instance |
| `aclshmemx_finalize(id)` | 目标非当前活动时先 `aclshmemx_instance_ctx_set_impl(id)` |

![image](images/initialization/shmem_finalize_flow.png)

*图：`aclshmemi_finalize_impl` 顺序；在 `aclshmemi_bootstrap_finalize` 前经 `aclshmemi_control_barrier_all` 做 Config Store 控制面同步。*

### 4.2 执行顺序

经 `aclshmemi_finalize_impl`（mutex 保护）：

1. （必要时）`aclshmemx_instance_ctx_set_impl`
2. `aclshmemi_team_finalize` → `aclshmemi_signal_finalize` → `memory_manager_destroy`
3. `remove_heap` → `release_heap` → `finalize_device_state` →（可选 host）→ `release_aclshmem_entity`
4. `aclrtSynchronizeStream` → `aclrtDestroyStream`
5. **`aclshmemi_control_barrier_all`** — 经 Config Store `GroupBarrier` 同步，**全体 PE 均进入 finalize 并到达此点后**，才允许拆 Bootstrap（见 [config_store_bootstrap.md §11](config_store_bootstrap.md#十一finalize)）
6. `aclshmemi_bootstrap_finalize`（`g_store_ref--`，归零时 `DestroyStore`）
7. `g_init_manager_count--`，归零 `delete init_manager`
8. `is_aclshmem_initialized = false`；`aclshmemi_instance_ctx_destroy`（`instance_id == 0` 为 no-op）

`init_manager == nullptr` 时直接清标志并销毁 instance ctx。MPI 模式仅 SHMEM 内部调用了 `MPI_Init` 时 finalize 才 `MPI_Finalize`。

### 4.3 多 instance

- 各 instance 独立 finalize；全局资源由最后一个 instance 释放
- 非 0 instance 的 team 能力受限（如无 split）
- 推荐使用 `aclshmemx_finalize(instance_id)`

---

## 五、函数与文件索引

### 5.1 对外 API

| API | 文件 |
|-----|------|
| `aclshmemx_init_attr` / `aclshmem_finalize` / `aclshmemx_finalize` | `src/host/init/shmem_init.cpp` |
| `aclshmemx_get_uniqueid` / `set_attr_uniqueid_args` / `init_status` / `instance_ctx_*` | `src/host/init/shmem_init.cpp` |
| `aclshmem_malloc` / `aclshmem_free` | `src/host/mem/shmem_mm.cpp` |

### 5.2 初始化关键内部函数

| 函数 | 文件 |
|------|------|
| `check_attr` / `aclshmemi_state_init_attr` | `shmem_init.cpp` |
| `aclshmemi_bootstrap_init` | `shmemi_bootstrap.cpp` |
| `bind_aclshmem_entity` / `init_device_state` / `reserve_heap` / `setup_heap` | `shmem_init_backend.cpp` |
| `exchange_slice` / `exchange_entity` / `update_device_state` | `shmem_init_backend.cpp` |
| `memory_manager_initialize` | `shmem_mm.cpp` |
| `aclshmemi_signal_init` / `aclshmemi_team_init` / `aclshmemi_sync_init` | `shmem_init.cpp` 等 |
| `aclshmemi_control_barrier_all` | `shmem_init_backend.cpp` |

### 5.3 Finalize 关键内部函数

| 函数 | 文件 |
|------|------|
| `aclshmemi_finalize_impl` | `shmem_init.cpp` |
| `aclshmemi_control_barrier_all`（finalize 中，Bootstrap 前） | `shmem_init.cpp` / `shmem_init_backend.cpp` |
| `aclshmemi_team_finalize` / `aclshmemi_signal_finalize` | `shmem_team.cpp` / `shmem_init.cpp` |
| `memory_manager_destroy` | `shmem_mm.cpp` |
| `remove_heap` / `release_heap` / `finalize_device_state` / `release_aclshmem_entity` | `shmem_init_backend.cpp` |
| `aclshmemi_bootstrap_finalize` | `shmemi_bootstrap.cpp` |

### 5.4 模块级文件

| 模块 | 文件 |
|------|------|
| Bootstrap 插件 | `src/host/bootstrap/shmemi_bootstrap_config_store.cpp`、`shmemi_bootstrap_mpi.cpp` |
| Config Store / Acclink | 见 [config_store_bootstrap.md §十三](config_store_bootstrap.md) |
| HYBM meta | `src/host/entity/mem_entity_def.h` |

---

## 六、常见问题

| 现象 | 处理 |
|------|------|
| init 卡在连 rank 0 | 检查 `ip_port`/UID `addr` 连通性；增大 `shm_init_timeout`（建链重试上限，默认 120≈约 120s）；详见 [config_store_bootstrap.md](config_store_bootstrap.md) §10.1 |
| barrier/allgather 超时 | 增大 `control_operation_timeout`（秒）；确认所有 rank 已进入 init；详见 [config_store_bootstrap.md](config_store_bootstrap.md) §九 |
| UID 握手失败 | 确认 UID 字节级一致、`magic` 已广播 |
| finalize barrier 超时 / 卡住 | 确认 **全体 PE 均已调用** `aclshmem_finalize`；增大 `control_operation_timeout`；详见 [config_store_bootstrap.md §11](config_store_bootstrap.md#十一finalize) |
| 不可重复 init | 先 finalize 再 init |
| RMA 地址异常 | 各 rank `malloc`/`free` 同序同大小（见 `malloc.png`） |
| 仅 rank 0 调 init | 每个 PE 均须 `aclshmemx_init_attr` |
| 多 instance 端口冲突 | `SHMEM_INSTANCE_PORT_RANGE`，模板端口写 **0** |
| A2/A3 barrier 异常 | 算子入口 `util_set_ffts_config(util_get_ffts_config())` |
