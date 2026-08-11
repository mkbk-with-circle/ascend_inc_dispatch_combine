# DYN0：Combine RoutePlan 语义（Logical vs Compiled）

状态：审计与设计冻结文档。实现从 DYN1A 起；不覆盖 FG7 冻结 P100 lineage。

## 1. 必须区分的维度

| 符号 | 名称 | 含义 |
|------|------|------|
| W / `worker_world_size` | Worker 世界大小 | process group 内 worker rank 数（2/4/8…） |
| K | Top-k（见 §2 三元） | 路由贡献语义，**不是**参与通信的 worker 数 |
| I / `inc_count` | INC PE 数 | 承担归约的 INC 设备数 |
| O / `owner_count` | 每 INC owner/AIV 数 | 调度单元；可为 specialization（如 20），非正确性前提 |
| `active_sources` | 活跃源集合 | 某 window/result 实际有数据的 contributor rank **集合** |

### 禁止的替代（证据不足 / 语义错误）

- `active_sources == topk` 后关闭其余 worker
- `n_pes = topk + 1`
- 假定每个 worker 对每个 result 恰贡献一次
- `ordinal == source_rank` / `worker rank == source ordinal`
- `expected_count == worker_count` 或 `== active_sources`
- 用 FG3A / as-sweep / cart-soak 宣称「动态 top-k 已证明」

### 正确要求

- 固定 Wn 时 **n 个 worker 全部 launch、register、完成协议**（含零 contribution）
- 同 worker 可持有多个不同 ordinal；某 worker 可整轮零 contribution
- `K > W` 必须通过同 worker 多 contribution 实现，不得加 worker
- contribution 到达顺序与 ordinal 无关

## 2. Top-k 三元

```text
declared_max_topk       // workspace / capability 上限（shape.topk 上限）
result_expected_count   // 某 result 实际 contribution 数（LogicalPlan）
uniform_topk            // 可选；仅当全部 result 的 expected_count 相等时有效
```

禁止把 `shape.topk` 一律当成每个 result 的绝对 expected count（padding / drop / capacity clip / ragged）。

Ordinal：每个 result 内唯一且 ∈ `[0, expected_count)`。

## 3. 两层计划（禁止混写）

```text
LogicalRoutePlan（稳定语义，框架可见）
        │ compile(topology, shape, capabilities)
        ▼
CompiledExecutionPlan（物理调度，上层不可见 INC）
        │ launch
        ▼
Device CSR Worklist / Descriptor
```

### LogicalContribution / LogicalResult / LogicalRoutePlan

见实施计划：含 `contribution_uid`、`result_id`、`ordinal`、`contributor_rank`、`contributor_local_row`、`weight`；result 侧 CSR `contribution_begin/count`（=`result_expected_count`）。

`semantic_digest` **不含** specialization 选择与 topology。

**禁止**写入逻辑层：`owner`、`inc_slot`、`payload_offset`、`ingress_channel`。

### CompiledContribution / CompiledExecutionPlan

含 `inc_index`、`owner_index`、`ingress_channel/slot`、`payload_offset`；以及 owner CSR worklist；`semantic_digest` + `topology_digest` + `execution_digest`。

换 W/I/O/非连续 PE 时只重 compile，不改 LogicalPlan。

## 4. 与既有 CombinePlan 的关系（DYN1A 锁定）

现有 [`inc_combine_plan.h`](../examples/inc/dispatch_combine/inc_combine_plan.h)：

- 已有 CSR：`contrib_begin` / `contrib_count`
- 字段混杂：`source_rank` 实为 dst；`assignment_id` 兼 UID/ordinal/slot；`ingress_slot` 属物理层

决策：

1. 升级为 `IncDcCombineLogicalPlanV2`
2. `LegacyPlanToLogicalPlanV2()` 适配旧 `BuildCombineReducePlan`
3. **禁止**第三套长期并存的「DYN RoutePlan」与旧语义并行

## 5. TopologyDescriptor

至少：`worker_pe_ids`、`inc_pe_ids`、worker→可达 INC 的 CSR（`worker_inc_offsets/indices`）、`topology_generation/digest`。

隔离假设备设：`total_pes==2*W`、`inc_pe=W+slot`、`rank==pe`、一一配对、`I==W`、owner=20 才能正确。

## 6. Device CSR reduce

```text
result_offsets[R+1]
contribution_entries[C]
owner_result_offsets[O+1]
owner_result_ids[R]
```

到达：小 K → ordinal bitmap；大 K → 计数+去重；generation+UID 防 stale/dup。

禁止 dense `for source in 0..active_sources` 与 `ordinal==source` 校验作为正确性前提。

## 7. Dispatcher

```text
W8/K8 + FP16 + h7168 + uniform → 冻结 direct-reduce specialization（只读至资格化）
W2/W4/W8 + supported K/shape → compiled CSR specialization
其他合法 → generic plan-driven
非法 → fail-closed（禁止静默退化 active_sources）
```

快路径输出 ≡ LogicalPlan；快路径选择不进 `semantic_digest`。

## 8. Framework route descriptor

仅 `route_cookie` 不足。须 `IncCombineFrameworkRoute{logical_plan, bytes, abi_version, memory_location, semantic_digest}`，submit 绑定 workspace lease + shape/dtype + PG + digests + topology generation + stream + operation generation。

## 9. 带宽口径

禁止 `278921216 * active_sources / 8` 冒充正式/物理带宽。

报告：`input_payload_bytes`、`output_payload_bytes`、`logical_useful_bytes`、`wire_or_ingress_bytes`、`metadata_bytes`、`makespan_us`、`logical_useful_gbps`、`physical_ingress_gbps`；历史对照另列 `legacy_p100_useful_gbps`。物理 >192 → 测量无效。

makespan = `max(rank_us)`（global_begin 包络）；跨 sample = `min(gbps)`。

## 10. 旧证据降级

| 产物 | 重新分类 |
|------|----------|
| FG7 formal/soak/correctness/P100 | 冻结 **W8/K8 uniform** 性能基准；**非**动态 top-k 证明 |
| `fg7_*as*diag*` / cart soak | **active_source / 扇入覆盖** only |
| FG3A extreme matrix | `n_pes=topk+1` 模拟；**非**固定 W 动态 K 证明 |

`dynamic_topk_proven = false`，直至 DYN2+ 设备 gate 出具「全员在场」强制字段。

## 11. 阶段门

DYN0（本文档 + audit）→ DYN1A LogicalPlanV2 → DYN1B Topology+Compiler → DYN1C FrameworkRoute ABI → DYN2 device → DYN3 CSR direct → DYN4 binding → DYN5 formal/soak/promotion。
