# Platform policy / 平台策略

## 中文

### 这个目录是干什么的？

回答「这台机器 **能开多少资源**、**PE 怎么映到物理 NPU**、**设备上怎么安全搬/归约数据**」。  
只放硬件/拓扑策略与 AIC 原语，**不放** 业务 shape 查表（W/K/负载表禁止出现在 `resource_policy`）。

### 为什么要有这一层？

- 单 INC Dispatch/Combine 共享同一套能力上限与 gather 原语，避免各写一套 UB/MTE 约定。
- AIV 分配策略与具体 case 解耦：换拓扑或换卡型时只改策略，不改每个 kernel 的魔法常数。
- `external_start_gate` 服务 overlap 资格化，默认路径几乎无开销，正式安装包可裁剪。

### 文件一览

| 文件 | 用途 | 为什么要有 | 备注 |
|---|---|---|---|
| `inc_dc_platform_capabilities.h` | UB、private-MTE、ABI 容量上限 | 可移植性底座；所有 workspace 尺寸推算依赖它 | 核心 |
| `inc_dc_physical_map.h` | logical PE → 物理 NPU 运行时映射 | 逻辑拓扑与真实设备编号解耦 | 核心 |
| `inc_dc_resource_policy.h` | 仅基于硬件/拓扑的 D/C AIV 分配（如 1:2 权重） | 禁止 workload 查表；保证分配可复现 | 核心 |
| `inc_dc_external_start_gate.h` | 可选外部启动门禁；默认一次环境查询 | overlap 资格化需要可控启动点 | 测试辅助，源码保留 |
| `inc_dc_fp16_host.h` | Host FP16 转换 | host 侧构造/校验 FP16 数据 | 原语 |
| `inc_dc_fp16_aicore.h` | Device FP16 转换 | kernel 内 cast/量化路径共用 | 原语 |
| `inc_dc_ub_tile_aicore.h` | UB 分块辅助 | 统一 tile 吃 UB，避免各 kernel 手算 | 原语 |
| `inc_dc_vector_reduce_aicore.h` | 通用向量归约原语 | Combine 加权归约底座 | 原语 |
| `inc_dc_gather_mte_aicore.h` | MTE gather（GM→UB→GM） | Dispatch/Combine 共用；禁止逐字节 AIV 循环 | 原语 |

---

## English

### What is this directory?

Hardware/topology policy and AIC primitives: **how many resources**, **how PE maps
to physical NPUs**, and **how to gather/reduce safely on device**. No workload
shape lookup tables belong in `resource_policy`.

### Why this layer exists

- Single-INC Dispatch and Combine share one capability ceiling and gather
  contract instead of inventing per-kernel UB/MTE rules.
- AIV allocation is topology/hardware-driven so cases stay reproducible.
- The optional start gate supports overlap qualification with near-zero cost
  on the default path.

### Files

| File | Purpose | Why it exists | Notes |
|---|---|---|---|
| `inc_dc_platform_capabilities.h` | UB, private-MTE, ABI capacity ceilings | Portability base for workspace math | Core |
| `inc_dc_physical_map.h` | Logical-PE → physical-NPU map | Decouple logical topology from device ids | Core |
| `inc_dc_resource_policy.h` | Hardware/topology-only D/C AIV split | No workload tables; reproducible allocation | Core |
| `inc_dc_external_start_gate.h` | Optional external launch gate | Overlap qualification control point | Test helper; keep in source |
| `inc_dc_fp16_host.h` | Host FP16 conversion | Host construct/validate FP16 | Primitive |
| `inc_dc_fp16_aicore.h` | Device FP16 conversion | Shared cast path in kernels | Primitive |
| `inc_dc_ub_tile_aicore.h` | UB tiling helpers | One tiling convention | Primitive |
| `inc_dc_vector_reduce_aicore.h` | Vector reduction engine base | Combine weighted reduce | Primitive |
| `inc_dc_gather_mte_aicore.h` | MTE gather GM→UB→GM | Shared by Dispatch/Combine; no byte loops | Primitive |
