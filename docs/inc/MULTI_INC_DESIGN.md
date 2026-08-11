# Multi-INC 拓扑与 ABI 设计

## 概述

在保留单 INC（8W+1S，`PE_SIZE=9`）路径不变的前提下，扩展 **1 worker ↔ 1 INC** 的 paired 拓扑（8W+8I，`PE_SIZE=16`），支持 2/4/8 rank 灵活绑定，并为 Megatron / vLLM 接入预留 C API。

状态追踪与单 INC `migration_status.json`（43/43）**分离**，使用：

`docs/inc/hardware_profiles/910b-yuanmingyu/multi_inc/multi_inc_status.json`

## Paired 生产数据面：dc_ll 基本路径（2026-07）

**默认 paired dispatch 数据面**为「单 INC 扩展多 INC 基本路径」，引擎 `INC_DC_DISPATCH_ENGINE=dc_ll`：

```
Worker w → paired INC (SwitchPeForWorker(w)) → True-MTE fanout → Destination d
Destination d → reclaim → source_inc_pe（descriptor 携带）
```

关键契约：

- `DcLlExpectedRecvDesc.source_inc_pe`：SINGLE=`group_size`；PAIRED=`SwitchPeForWorker(source_rank)`
- PAIRED egress **单写**：`DcLlRemapPairedDestinationSlots` 按 `source_inc_pe` 划分 `max_recv_slots` 子区间（`local % slots_per_inc` + ring 复用）
- Destination reclaim 按 `source_inc_pe` 路由；`DcLlReclaimBatchState` 按 PE 批量回传
- Switch 热路径 **原样** True-MTE / fifo / 多 AIV；仅 `paired_ingress_source` 过滤 owned source

`dc_ab v2` 保留为显式 fallback（`INC_DC_DISPATCH_ENGINE=dc_ab`），**不得**作为 paired 生产默认；其 ~0.000033 GB/s 带宽探测不代表本路径。

预检与 sweep：

| 脚本 | 说明 |
|------|------|
| `run_d03_dc_ll_paired_preflight_gate.sh` | 1MiB + 4MiB/rank vs live native 85%；产出 `d03_dc_ll_paired_preflight_latest.json` |
| `run_d03_dc_ll_paired_sweep_gated.py` | 读取 preflight，`sweep_allowed=true` 才允许 P5/sweep |

## 数据面路由（无 INC↔INC）

### Dispatch（按 source-rank 配对 INC）

`worker i → INC i → putmem 直发 destination worker PE`

- Switch ingress 仅轮询 `paired_ingress_source = i`（`inc_dispatch_dc_ll_switch_kernel` / fifo）
- Fanout 保持 `putmem*(..., dst_worker)`，不经其他 INC

### Combine（按 **dst-rank** 配对 INC）

对 token 的 top-k 贡献：

`source worker → dst-rank 的 paired INC → 归约 → 直发 dst-rank worker`

- `BuildCombineUploadPlan` 写入 `target_switch_pe = SwitchPeForWorker(dst_rank)`
- Worker / legacy kernel 上传到 `target_switch_pe`
- 每个 INC 仅归约 `result_tokens[rt].source_rank == paired_worker_rank` 的 token

## Baseline 双口径

| 模式 | pass_line 公式 | 说明 |
|------|----------------|------|
| 单 INC (`topology=single`) | `native_comm_only / N × 0.9` | N=`group_size`；单 switch 只吃一条机内路径份额 |
| 多 INC paired | `native_comm_only × 0.9` | **不除以 N**；与 SHMEM `comm_only` 同口径直比 |

实现：`inc_dc_baseline.py` → `inc_dc_p5_shape.py` / `inc_c12_baseline_helper.py`；gate 脚本通过 `--topology single|paired` 传参。

Overlap 交叠收益单独验收：`BW_inc_step ≥ 1.2 × BW_shmem_step`；并发非阻塞见 `run_overlap_nonblock_gate.sh`。

## 拓扑

| 模式 | NPE | Worker PE | INC (switch) PE | 环境变量 |
|------|-----|-----------|-----------------|----------|
| single | group_size + 1 | 0 .. N-1 | N | `9rank.env` |
| paired | group_size × 2 | 0 .. N-1 | N .. 2N-1 | `16rank.env` / `INC_DC_TOPOLOGY=paired` |

映射：`INC_PAIRED_MAP=0:8,1:9,...` 覆盖默认 `worker_i → pe (N+i)`。

实现入口：

- `inc_dc_topology.h` / `inc_dc_topology.cpp` — `TopologyFromEnv`, `SwitchPeForWorker`, `IsSwitchPe`
- `inc_dc_group.h` — `IncDcGroupConfig` 扩展字段：`switch_rank`, `inc_pes`, `num_inc`, `topology`
- `dc_launch_ranks` — 根据 topology 计算 NPE

## 同步语义

- **禁止** paired 热路径上的全局 `aclshmem_barrier_all`
- 默认：`INC_FIRST_PACKET`（pair 内前后同步）
- 计划 v1：epoch token / sequence，允许不同 pair 处于不同阶段以支持 overlap 流水

## AIV 资源（overlap 预留）

单 INC 全量块：dispatch `[0,16)`，combine `[16,40)`。

Overlap 模式（`INC_DC_OVERLAP_MODE=1`）每 INC NPU（连续布局，无 guard gap）：

- dispatch `[0,12)`
- combine `[12,36)`（`switch_block_base=12` 写入 combine trace header）

`BuildOverlapSwitchResourceMap()` / `ApplyOverlapModeToAivProfile()` 在 `inc_dc_resource_map.cpp` / `inc_dc_aiv_profile.cpp`。

## C API / ABI

`inc_dc_api.h` 中 `inc_dc_dispatch_async` / `inc_dc_combine_async` **签名不变**。

`IncDcGroupConfig` 新增字段（`abi_version` 仍为 1，向后兼容）：

```c
int32_t switch_rank;      // worker 侧：配对 INC 的 SHMEM PE
const int32_t *inc_pes;   // 可选，length=group_size
uint32_t num_inc;         // 1 或 group_size
uint32_t topology;        // 0=SINGLE, 1=PAIRED
```

生命周期：由框架在 `inc_dc_create_group` 时填充；handle cache 按 `(group_id, topology)` 区分。

### Megatron / vLLM 对接预留

1. **Rank 映射**：`torch.distributed` rank → `worker_pes[]` + `switch_rank`
2. **ProcessGroup**：paired 模式创建 16-rank SHMEM team；单 INC 仍为 9-rank
3. **Wrapper 层**：读取 `INC_DC_TOPOLOGY` / `IncDcGroupConfig.topology`，不修改核心 kernel
4. **Overlap**：`inc_dc_overlap` binary + `run_overlap_paired_bench.sh` / `run_overlap_nonblock_gate.sh` — 同 session 内 `LaunchOverlapConcurrentAsync`，`OVERLAP_BENCH_SAMPLE` 含 `interval_overlap` 与 Td/Tc 分解

## 实现状态（诚实标注）

| 组件 | 状态 |
|------|------|
| 拓扑 C++ (`inc_dc_topology`) | 已实现 |
| Dispatch paired ingress | 已实现（`paired_ingress_source`） |
| Combine dst-inc routing | 已实现（device + legacy host 热路径） |
| Paired 热路径无 barrier | 已实现（runner bench 循环 + combine roofline） |
| 双口径 baseline | 已实现（`inc_dc_baseline.py` + gate 传参 + c15b paired pass_line） |
| Overlap 真并发 | **代码已实现**（`inc_dc_overlap`）；910c gate 待跑通 |

## 配置

| 文件 | 用途 |
|------|------|
| `docs/inc/configs/2rank.env` | 1W+1I smoke |
| `docs/inc/configs/4rank.env` | 2W+2I smoke |
| `docs/inc/configs/16rank.env` | 8W+8I 生产 |

## Gate / Runner

| 脚本 | 说明 |
|------|------|
| `run_d03_paired_production_traversal_gate.sh` | paired dispatch 遍历（`INC_DC_DISPATCH_ENGINE=dc_ll`） |
| `run_d03_dc_ll_paired_preflight_gate.sh` | paired dc_ll 双 case 带宽预检 |
| `run_c15b_paired_formal_perf_gate.sh` | paired combine formal |
| `run_overlap_paired_bench.sh` | overlap v1 benchmark（`inc_dc_overlap`） |
| `run_overlap_nonblock_gate.sh` | 并发非阻塞门禁 T_both + interval_overlap |
| `run_p7_multi_inc_smoke.sh` | 2/4/8 rank smoke |
| `compare_multi_inc_status.py` | 汇总 `multi_inc_status.json` |

## SHMEM baseline（overlap）

- v1：`inc_dc_overlap` 同 session 并发 launch；`T_both` 对比 solo dispatch/combine
- 门禁：`T_both ≤ 1.15×max(Td,Tc)` 且 `T_both ≤ 0.85×(Td+Tc)`；BW ≥ 1.2× max(solo dispatch, solo combine)
- 单算子对比仍分别测 inc-dispatch vs SHMEM、inc-combine vs SHMEM
