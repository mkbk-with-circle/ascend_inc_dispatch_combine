# ACLSHMEM Atomic 原子操作接口用法语义总结

> 本文档基于实际代码实现编写，覆盖各层接口：标准接口 `aclshmem_<type>_atomic_<op>`、MTE 后端 `aclshmemx_mte_atomic_<op>`、RDMA 后端 `aclshmemx_roce_atomic_<op>`。

---

## 一、接口层次关系

atomic 接口分为两层：

```text
aclshmem_<type>_atomic_<op>()          ← 标准接口
    │
    ├── topo == MTE  → aclshmemx_mte_atomic_<op>()   ← MTE 后端
    └── topo == ROCE → aclshmemx_roce_atomic_<op>()  ← RDMA 后端
```

标准接口根据 `device_state->topo_list[pe]` 自动分派到 MTE 或 RDMA 后端，并在后端调用之上额外添加同步保证。

---

## 二、MTE 后端接口：`aclshmemx_mte_atomic_<op>`

> 头文件：`include/device/gm2gm/engine/shmem_device_mte.h`
> 实现文件：`src/device/gm2gm/engine/shmem_device_mte.hpp`

**核心特征：所有 MTE 后端接口均为异步**，不论是否带返回值。接口仅发起硬件原子操作即返回，无 quiet/fence。

同步原语：`aclshmemx_mte_quiet()` 或 Kernel Runtime 级 `SetFlag/WaitFlag`。

### 2.1 接口清单

| 接口完整名 | 返回值 | 支持类型 | 硬件平台 |
|---|---|---|---|
| `aclshmemx_mte_atomic_fetch` | T | int32, uint32, float, int64, uint64 | 950 |
| `aclshmemx_mte_atomic_set` | void | uint32, uint64 | 950 |
| `aclshmemx_mte_atomic_swap` | T | uint32, uint64 | 950 |
| `aclshmemx_mte_atomic_compare_swap` | T | uint32, uint64 | 950 |
| `aclshmemx_mte_atomic_inc` | void | int32, uint32, float, int64, uint64 | 950 |
| `aclshmemx_mte_atomic_add` | void | int8, int16, bf16, half, int32, float (A2/A3/950) + uint32, uint64, int64 (950) | A2/A3/950 |
| `aclshmemx_mte_atomic_fetch_inc` | T | int32, uint32, float, int64, uint64 | 950 |
| `aclshmemx_mte_atomic_fetch_add` | T | int32, uint32, float, int64, uint64 | 950 |

### 2.2 同步要求

所有 MTE 后端接口调用后，如果存在**标量计算单元（Scalar）与搬运单元（MTE2/MTE3）之间的数据依赖**，开发者需根据实际数据流插入同步：

| 时机 | 场景 | 同步方式 |
|---|---|---|
| 调用前 | 其他单元（如 MTE2）正在写同一 UB 区域 | `SetFlag/WaitFlag` on `MTE2_S` |
| 调用后 | 后续操作依赖原子操作在 GM 中的结果 | `SetFlag/WaitFlag` on `MTE3_MTE2` |
| 调用后 | 需要复用同一 UB 区域写入新数据 | `SetFlag/WaitFlag` on `MTE3_S` |

### 2.3 `aclshmemx_mte_atomic_add` 的类型差异（重要）

`aclshmemx_mte_atomic_add` 在 Ascend950 上按类型有两种实现路径：

| 类型组 | 支持类型 | 实现方式 | 同步要点 |
|---|---|---|---|
| UB + MTE3 路径 | int8, int16, bf16, half, int32, float | Scalar 写操作数到 UB → MTE3 读 UB → MTE3 atomic add 写 GM | 需关注 UB 相关 fence（MTE2_S、MTE3_S）和 GM fence（MTE3_MTE2） |
| Scalar AtomicAdd 路径 | uint32, uint64, int64 | Scalar 单元直接执行标量 AtomicAdd 写 GM，不经过 UB 和 MTE3 | 不需要 UB/MTE3 相关 fence；仅在 Scalar↔MTE2/MTE3 有 GM 数据依赖时按通用规则同步 |

---

## 三、RDMA 后端接口：`aclshmemx_roce_atomic_<op>`

> 头文件：`include/device/gm2gm/engine/shmem_device_rdma.h`
> 实现文件：`src/device/gm2gm/engine/shmem_device_rdma.hpp`

**核心特征：有返回值的接口为同步接口（内部 quiet），无返回值的接口为异步接口（需调用者 quiet）。**

同步原语：`aclshmemx_roce_quiet(pe, buf, sync_id)`。

RDMA AMO 的 target/source 参数会作为本地对称地址换算到指定 PE，因此必须指向对称内存；不能把任意
本地 GM 地址直接用作该参数。可通过 `aclshmem_malloc` 等对称内存分配接口准备目标对象。

### 3.1 接口清单（同步接口——内部完成 quiet）

| 接口完整名 | 返回值 | 支持类型 | 硬件平台 |
|---|---|---|---|
| `aclshmemx_roce_atomic_fetch` | T | 32/64 位整数 | 950 |
| `aclshmemx_roce_atomic_fetch_add` | T | 32/64 位整数 | 950 |
| `aclshmemx_roce_atomic_fetch_inc` | T | 32/64 位整数 | 950 |
| `aclshmemx_roce_atomic_fetch_and` | T | int32, uint32, int64, uint64 | 950 |
| `aclshmemx_roce_atomic_fetch_or` | T | int32, uint32, int64, uint64 | 950 |
| `aclshmemx_roce_atomic_fetch_xor` | T | int32, uint32, int64, uint64 | 950 |
| `aclshmemx_roce_atomic_compare_swap` | T | 32/64 位整数 | 950 |
| `aclshmemx_roce_atomic_swap` | T | 32/64 位整数 | 950 |

> 以上接口在返回前内部调用 `aclshmemx_roce_quiet`，**调用者无需额外同步**。

### 3.2 接口清单（异步接口——需调用者 quiet）

| 接口完整名 | 返回值 | 支持类型 | 硬件平台 |
|---|---|---|---|
| `aclshmemx_roce_atomic_set` | void | 32/64 位整数 | 950 |
| `aclshmemx_roce_atomic_add` | void | 32/64 位整数 | 950 |
| `aclshmemx_roce_atomic_inc` | void | 32/64 位整数 | 950 |
| `aclshmemx_roce_atomic_and` | void | int32, uint32, int64, uint64 | 950 |
| `aclshmemx_roce_atomic_or` | void | int32, uint32, int64, uint64 | 950 |
| `aclshmemx_roce_atomic_xor` | void | int32, uint32, int64, uint64 | 950 |

> 以上接口仅发起操作即返回，**调用者必须在依赖结果前调用 `aclshmemx_roce_quiet(pe, buf, sync_id)`**。

### 3.3 并发限制

> **警告**：使用 RDMA 作为底层传输时，不支持对同一 PE 的并发 RMA/AMO 操作。请使用 `device_state.rdma_config` 中的 `sync_id` 进行流水线同步。

---

## 四、标准接口：`aclshmem_<type>_atomic_<op>`

> 头文件：`include/device/gm2gm/shmem_device_amo.h`
> 实现文件：`src/device/gm2gm/shmem_device_amo.hpp`

标准接口根据拓扑自动分派到 MTE 或 RDMA 后端，并在后端调用之上**额外添加同步保证**。

标准 AMO 接口会将用户传入的 `dst`、`dest` 或 `source` 交给 `aclshmem_ptr(..., pe)` 换算为指定 PE 上的
目标地址，因此这些参数必须指向对称内存。该要求来自标准接口的地址换算方式，并不表示 MTE 引擎本身
只能访问对称内存；传输层自动分派不会改变这一接口语义。

```text
aclshmem_<type>_atomic_<op>(dst, val, pe)
    │
    ├── topo_list[pe] & MTE
    │   ├── fetch 变体:    调用 aclshmemx_mte_atomic_fetch_<op>() + PipeBarrier<PIPE_ALL>
    │   ├── swap/cas:      调用 aclshmemx_mte_atomic_swap/compare_swap() + PipeBarrier<PIPE_ALL>
    │   ├── add/inc:       透传 aclshmemx_mte_atomic_add/inc()，未加 PipeBarrier
    │   └── set/and/or/xor: 透传底层异步接口，未加 PipeBarrier
    │
    └── topo_list[pe] & ROCE
        ├── fetch 变体:    透传 aclshmemx_roce_atomic_fetch_<op>()（后端已 quiet）
        ├── swap/cas:      透传 aclshmemx_roce_atomic_swap/compare_swap()（后端已 quiet）
        ├── add/inc:       透传 aclshmemx_roce_atomic_add/inc()（后端异步，未加 quiet）
        └── set/and/or/xor: 透传底层异步接口（未加 quiet）
```

### 4.1 同步接口（带返回值）

函数内部完成同步后才返回，**调用者无需额外同步**。

| 接口完整名（以 int32 为例） | 返回值 | MTE 路径同步方式 | RDMA 路径同步方式 |
|---|---|---|---|
| `aclshmem_int32_atomic_fetch_add` | int32_t | `aclshmemx_mte_atomic_fetch_add` + `PipeBarrier<PIPE_ALL>` | `aclshmemx_roce_atomic_fetch_add`（内部 quiet） |
| `aclshmem_int32_atomic_fetch_inc` | int32_t | `aclshmemx_mte_atomic_fetch_inc` + `PipeBarrier<PIPE_ALL>` | `aclshmemx_roce_atomic_fetch_inc`（内部 quiet） |
| `aclshmem_int32_atomic_fetch_and` | int32_t | ✗ 不支持 MTE | `aclshmemx_roce_atomic_fetch_and`（内部 quiet） |
| `aclshmem_int32_atomic_fetch_or` | int32_t | ✗ 不支持 MTE | `aclshmemx_roce_atomic_fetch_or`（内部 quiet） |
| `aclshmem_int32_atomic_fetch_xor` | int32_t | ✗ 不支持 MTE | `aclshmemx_roce_atomic_fetch_xor`（内部 quiet） |
| `aclshmem_int32_atomic_fetch` | int32_t | `aclshmemx_mte_atomic_fetch` + `PipeBarrier<PIPE_ALL>` | `aclshmemx_roce_atomic_fetch`（内部 quiet） |
| `aclshmem_uint32_atomic_swap` | uint32_t | `aclshmemx_mte_atomic_swap` + `PipeBarrier<PIPE_ALL>` | `aclshmemx_roce_atomic_swap`（内部 quiet） |
| `aclshmem_uint32_atomic_compare_swap` | uint32_t | `aclshmemx_mte_atomic_compare_swap` + `PipeBarrier<PIPE_ALL>` | `aclshmemx_roce_atomic_compare_swap`（内部 quiet） |

### 4.2 异步接口（不带返回值，必须显式同步）

| 接口完整名（以 int32 为例） | 返回值 | MTE 路径 | RDMA 路径 | 调用者必须的同步 |
|---|---|---|---|---|
| `aclshmem_int32_atomic_add` | void | `aclshmemx_mte_atomic_add`（异步） | `aclshmemx_roce_atomic_add`（异步） | MTE: SetFlag/WaitFlag（按需）；RDMA: `aclshmemx_roce_quiet` |
| `aclshmem_int32_atomic_inc` | void | `aclshmemx_mte_atomic_inc`（异步） | `aclshmemx_roce_atomic_inc`（异步） | MTE: SetFlag/WaitFlag（按需）；RDMA: `aclshmemx_roce_quiet` |
| `aclshmem_int32_atomic_and` | void | ✗ 不支持 MTE | `aclshmemx_roce_atomic_and`（异步） | RDMA: `aclshmemx_roce_quiet` |
| `aclshmem_int32_atomic_or` | void | ✗ 不支持 MTE | `aclshmemx_roce_atomic_or`（异步） | RDMA: `aclshmemx_roce_quiet` |
| `aclshmem_int32_atomic_xor` | void | ✗ 不支持 MTE | `aclshmemx_roce_atomic_xor`（异步） | RDMA: `aclshmemx_roce_quiet` |
| `aclshmem_uint32_atomic_set` | void | `aclshmemx_mte_atomic_set`（异步） | `aclshmemx_roce_atomic_set`（异步） | MTE: SetFlag/WaitFlag（按需）；RDMA: `aclshmemx_roce_quiet` |

> `atomic_add` 的 MTE 路径同步细节：按数据类型分两条路径，参见 [第二节 2.3](#23-aclshmemx_mte_atomic_add-的类型差异重要)。

---

## 五、同步规则速查（针对MTE和RDMA通路）

```text
使用 aclshmem_<type>_atomic_<op> 接口
│
├── fetch_add / fetch_inc / fetch_and / fetch_or / fetch_xor / fetch / swap / compare_swap
│   └── 同步接口（带返回值），调用返回即完成，返回值可靠
│       MTE: fetch_add/fetch_inc/fetch/swap/compare_swap 有 PipeBarrier
│       RDMA: 全部内部 quiet
│       （MTE 路径 PipeBarrier 已保证传输完成；调用前如有其他单元写 UB 仍需 SetFlag/WaitFlag）
│
├── add / inc / and / or / xor / set
│   └── 异步接口（不带返回值），必须显式同步：
│       RDMA: aclshmemx_roce_quiet(pe, buf, sync_id)
│       MTE: 根据实际数据依赖插入 SetFlag/WaitFlag（and/or/xor 不支持 MTE）
```

---

## 六、各标准接口支持的类型与硬件平台

### 6.1 `aclshmem_<type>_atomic_add`

| 传输通路 | 支持类型 | MTE 实现方式 | 硬件平台 |
|---|---|---|---|
| MTE | int8, int16, bf16, half, int32, float | UB + MTE3 | A2/A3/950 |
| MTE | uint32, uint64, int64 | Scalar AtomicAdd | 950 |
| RDMA | int32, uint32, int64, uint64 | — | 950 |

### 6.2 `aclshmem_<type>_atomic_fetch_add`

| 传输通路 | 支持类型 | 硬件平台 |
|---|---|---|
| MTE | int32, uint32, uint64, int64, float | 950 |
| RDMA | int32, uint32, int64, uint64 | 950 |

### 6.3 `aclshmem_<type>_atomic_inc`

| 传输通路 | 支持类型 | 硬件平台 |
|---|---|---|
| MTE | int32, uint32, uint64, int64 | 950 |
| RDMA | int32, uint32, int64, uint64, float | 950 |

### 6.4 `aclshmem_<type>_atomic_fetch_inc`

| 传输通路 | 支持类型 | 硬件平台 |
|---|---|---|
| MTE | int32, uint32, uint64, int64, float | 950 |
| RDMA | int32, uint32, int64, uint64 | 950 |

### 6.5 位运算系列 (`atomic_and/or/xor` 及 fetch 变体)

| 传输通路 | 支持类型 | 硬件平台 |
|---|---|---|
| RDMA | int32, uint32, int64, uint64 | 950 |

> **注意**：位运算接口不支持 MTE 通路。

### 6.6 `aclshmem_<type>_atomic_fetch`

| 传输通路 | 支持类型 | 硬件平台 |
|---|---|---|
| MTE | uint32, uint64, int32, int64, float | 950 |
| RDMA | uint32, uint64, int32, int64, float | 950 |

### 6.7 `aclshmem_<type>_atomic_set`

| 传输通路 | 支持类型 | 硬件平台 |
|---|---|---|
| MTE | uint32, uint64, int32, int64, float | 950 |
| RDMA | uint32, uint64, int32, int64, float | 950 |

> CAST 变体（`ACLSHMEM_TYPE_FUNC_ATOMIC_SWAP_CAST`）：int32→uint32、int64→uint64、float→uint32 通过类型转换实现。

### 6.8 `aclshmem_<type>_atomic_swap`

| 传输通路 | 支持类型 | 硬件平台 |
|---|---|---|
| MTE | uint32, uint64, int32, int64, float | 950 |
| RDMA | uint32, uint64, int32, int64, float | 950 |

> CAST 变体同上。

### 6.9 `aclshmem_<type>_atomic_compare_swap`

| 传输通路 | 支持类型 | 硬件平台 |
|---|---|---|
| MTE | uint32, uint64, int32, int64 | 950 |
| RDMA | uint32, uint64, int32, int64 | 950 |

> CAST 变体（`ACLSHMEM_TYPE_FUNC_ATOMIC_CAS_CAST`）：int32→uint32、int64→uint64。

---

## 七、跨 PCIe（跨节点）限制

**MTE 通路不支持跨 PCIe**。跨节点场景请使用 RDMA 通路。

---

## 八、代码示例

以下示例中的 `dst` 必须来自对称分配。Host 侧所有 PE 以相同大小集合调用分配接口，并把返回地址传给
执行 Device AMO 的 kernel：

```c
void *symmetric_target = aclshmem_malloc(sizeof(uint64_t));
// Pass symmetric_target to the kernel and use the typed pointer as dst.
```

不能用 `aclrtMalloc` 分配的普通本地 GM 地址替代标准 AMO 的 target；该限制来自远端地址换算，而不是
MTE 或 RDMA 引擎对 GM 地址的固有限制。

### 8.1 异步接口 + RDMA（`aclshmem_int32_atomic_add`）

```c
// 异步 atomic add，RDMA 通路
aclshmem_int32_atomic_add(dst, 42, pe);

// ... 其他不依赖 dst 的计算 ...

// 读取结果前必须 quiet
aclshmemx_roce_quiet(pe, ub_buf, sync_id);

// 现在可以安全使用 dst 的值
int32_t result = *dst;
```

### 8.2 同步接口（`aclshmem_int32_atomic_fetch_add`）

```c
// 同步接口，调用返回即可直接使用返回值
int32_t old_val = aclshmem_int32_atomic_fetch_add(dst, 42, pe);
// old_val 是可靠的旧值，无需额外 quiet
```

### 8.3 MTE 路径 + UB 复用（`aclshmem_int32_atomic_add`，UB+MTE3 路径）

```c
aclshmemx_set_mte_config(0, 256, sync_id);

// fence MTE2 写 UB 完成（如果 MTE2 之前写入了同一 UB 区域）
SetFlag<HardEvent::MTE2_S>(sync_id);
WaitFlag<HardEvent::MTE2_S>(sync_id);

// 异步 atomic add（Scalar 写 UB → MTE3 读 UB → MTE3 写 GM）
aclshmem_int32_atomic_add(dst, 1, pe);

// fence MTE3 写 GM 完成——后续需要读 GM 结果
SetFlag<HardEvent::MTE3_MTE2>(sync_id);
WaitFlag<HardEvent::MTE3_MTE2>(sync_id);

// fence MTE3 读 UB 完成——如果要复用 UB
SetFlag<HardEvent::MTE3_S>(sync_id);
WaitFlag<HardEvent::MTE3_S>(sync_id);

// 现在可以安全读 GM 结果和复用 UB
```

### 8.4 同步接口 + MTE 路径（`aclshmem_int32_atomic_swap`）

```c
// atomic_swap 标准接口 MTE 路径已加 PipeBarrier，返回即完成
int32_t old = aclshmem_int32_atomic_swap(dst, new_val, pe);
// old 是可靠的旧值，无需额外同步

// 如果调用前有其他单元正在写同一 UB 区域，仍需 SetFlag/WaitFlag：
SetFlag<HardEvent::MTE2_S>(sync_id);
WaitFlag<HardEvent::MTE2_S>(sync_id);
```

### 8.5 异步接口 + RDMA（`aclshmem_int32_atomic_and`，RDMA 通路）

```c
// atomic_and 不带返回值，标准接口透传 roce_atomic_and（异步）
aclshmem_int32_atomic_and(dst, mask, pe);

// 读取结果前必须 quiet
aclshmemx_roce_quiet(pe, ub_buf, sync_id);

// 现在可以安全读取
int32_t val = *dst;
```

### 8.6 `aclshmem_uint32_atomic_set`（需显式同步）

```c
// atomic_set 底层为异步（无论 MTE 还是 RDMA）
aclshmem_uint32_atomic_set(dst, 99, pe);

// RDMA 通路需 quiet
aclshmemx_roce_quiet(pe, ub_buf, sync_id);

// 或 MTE 通路需按数据依赖插入 SetFlag/WaitFlag
```

---

## 九、函数名映射

标准 SHMEM 风格的宏定义映射：

```c
#define shmem_int8_atomic_add      aclshmem_int8_atomic_add
#define shmem_int16_atomic_add     aclshmem_int16_atomic_add
#define shmem_int32_atomic_add     aclshmem_int32_atomic_add
#define shmem_half_atomic_add      aclshmem_half_atomic_add
#define shmem_bfloat16_atomic_add  aclshmem_bfloat16_atomic_add
#define shmem_float_atomic_add     aclshmem_float_atomic_add
```

其他类型同理，遵循 `shmem_<type>_atomic_<op>` → `aclshmem_<type>_atomic_<op>` 的映射规则。

---

## 十、各接口层同步性总表

### MTE 后端（`aclshmemx_mte_atomic_<op>`）

| 接口 | 返回值 | 同步性 | 同步方式 |
|---|---|---|---|
| `mte_atomic_fetch` | T | 异步 | 需 SetFlag/WaitFlag 或 `aclshmemx_mte_quiet` |
| `mte_atomic_set` | void | 异步 | 同上 |
| `mte_atomic_swap` | T | 异步 | 同上 |
| `mte_atomic_compare_swap` | T | 异步 | 同上 |
| `mte_atomic_inc` | void | 异步 | 同上 |
| `mte_atomic_add` | void | 异步 | 同上（按类型有两条路径，见 2.3） |
| `mte_atomic_fetch_inc` | T | 异步 | 同上 |
| `mte_atomic_fetch_add` | T | 异步 | 同上 |

### RDMA 后端（`aclshmemx_roce_atomic_<op>`）

| 接口 | 返回值 | 同步性 | 同步方式 |
|---|---|---|---|
| `roce_atomic_fetch` | T | 同步 | 内部 `roce_quiet` |
| `roce_atomic_fetch_add` | T | 同步 | 内部 `roce_quiet` |
| `roce_atomic_fetch_inc` | T | 同步 | 内部 `roce_quiet` |
| `roce_atomic_fetch_and` | T | 同步 | 内部 `roce_quiet` |
| `roce_atomic_fetch_or` | T | 同步 | 内部 `roce_quiet` |
| `roce_atomic_fetch_xor` | T | 同步 | 内部 `roce_quiet` |
| `roce_atomic_compare_swap` | T | 同步 | 内部 `roce_quiet` |
| `roce_atomic_swap` | T | 同步 | 内部 `roce_quiet` |
| `roce_atomic_set` | void | 异步 | 需 `aclshmemx_roce_quiet` |
| `roce_atomic_add` | void | 异步 | 需 `aclshmemx_roce_quiet` |
| `roce_atomic_inc` | void | 异步 | 需 `aclshmemx_roce_quiet` |
| `roce_atomic_and` | void | 异步 | 需 `aclshmemx_roce_quiet` |
| `roce_atomic_or` | void | 异步 | 需 `aclshmemx_roce_quiet` |
| `roce_atomic_xor` | void | 异步 | 需 `aclshmemx_roce_quiet` |

### 标准接口（`aclshmem_<type>_atomic_<op>`）

| 接口 | 返回值 | MTE 路径 | RDMA 路径 | 调用者需同步？ |
|---|---|---|---|---|
| `atomic_fetch_add` | T | 同步（+PipeBarrier） | 同步（透传） | 否 |
| `atomic_fetch_inc` | T | 同步（+PipeBarrier） | 同步（透传） | 否 |
| `atomic_fetch` | T | 同步（+PipeBarrier） | 同步（透传） | 否 |
| `atomic_fetch_and` | T | ✗ 不支持 | 同步（透传） | 否 |
| `atomic_fetch_or` | T | ✗ 不支持 | 同步（透传） | 否 |
| `atomic_fetch_xor` | T | ✗ 不支持 | 同步（透传） | 否 |
| `atomic_swap` | T | 同步（+PipeBarrier） | 同步（透传） | 否 |
| `atomic_compare_swap` | T | 同步（+PipeBarrier） | 同步（透传） | 否 |
| `atomic_and` | void | ✗ 不支持 | 异步（透传） | **是** |
| `atomic_or` | void | ✗ 不支持 | 异步（透传） | **是** |
| `atomic_xor` | void | ✗ 不支持 | 异步（透传） | **是** |
| `atomic_add` | void | 异步 | 异步 | **是** |
| `atomic_inc` | void | 异步 | 异步 | **是** |
| `atomic_set` | void | 异步 | 异步 | **是** |

---

## 十一、源码参考

| 层级 | 头文件 | 实现文件 |
|---|---|---|
| 标准接口声明 | `include/device/gm2gm/shmem_device_amo.h` | — |
| 标准接口实现 | — | `src/device/gm2gm/shmem_device_amo.hpp` |
| MTE 后端接口 | `include/device/gm2gm/engine/shmem_device_mte.h` | `src/device/gm2gm/engine/shmem_device_mte.hpp` |
| RDMA 后端接口 | `include/device/gm2gm/engine/shmem_device_rdma.h` | `src/device/gm2gm/engine/shmem_device_rdma.hpp` |
