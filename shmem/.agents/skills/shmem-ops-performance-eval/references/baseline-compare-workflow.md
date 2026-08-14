# Baseline 接入与性能对比工作流

本文规定 `shmem-ops-performance-eval` 在采集 SHMEM 算子性能时，**如何接入 HCCL/aclnn baseline、如何跑对比、如何在聊天与报告中输出对比结果**。以 `custom-ops/alltoallv` 为参考实现。

**命令代码段唯一参考**：[perf-workflow.md](perf-workflow.md)（Skill 内禁止引用独立 shell 脚本文件名）。

性能评估唯一标准见 [timing-and-metrics-standard.md](timing-and-metrics-standard.md)。

## 1. 核心要求

- 有 HCCL/aclnn baseline 时 **MUST** 实现 C++ baseline 可执行文件，**NEVER** 仅用 Python/torch 脚本代替
- **MUST** 用与 SHMEM 相同的 case（同一 `gen_data.py` 输出、同一 sendcounts/shape/dtype/PE 数）
- **MUST** 在聊天界面输出 **SHMEM vs baseline 对比表**（不得只报 `[PERF]` 单边数据）
- **MUST** 写入 `docs/performance_report.md` §3.5 性能对比表
- 达标线见 [timing-and-metrics-standard.md](timing-and-metrics-standard.md)；规模要求见 [performance-eval-guide.md](performance-eval-guide.md)

分阶段采集（**MUST — HCCL 不得在 SHMEM init 之后同会话调用**）：按 [perf-workflow.md §1](perf-workflow.md) 执行阶段 A（baseline）→ 阶段 B（SHMEM）→ 阶段 C（离线 `perf_compare.py`）。

## 2. 目录与产物（custom-ops）

```
custom-ops/
├── scripts/lib/perf_compare.py       # 带宽离线对比
├── scripts/perf_compare_latency.py   # 时延离线对比（可选）
└── <op_name>/
    ├── baseline/scripts/run_baseline.sh   # HCCL baseline 入口
    ├── scripts/perf.sh                  # SHMEM perf 入口
    └── data/perf/                    # baseline_*.log / shmem_*.log
```

### 2.0 HCCL 与 SHMEM 隔离（Hard Gate）

HCCL baseline 与 SHMEM perf **NEVER** 同 shell / 同 `docker exec` 混跑。分阶段执行与隔离规则见 [perf-workflow.md](perf-workflow.md)。

### 2.1 CMake 集成

算子根 `CMakeLists.txt` 末尾：

```cmake
add_subdirectory(baseline)
```

baseline 的 `CMakeLists.txt` 将 `CMAKE_RUNTIME_OUTPUT_DIRECTORY` 设为 `${CMAKE_BINARY_DIR}/../bin`，与算子主程序同目录。

### 2.2 日志标签约定

| 实现 | 输出前缀 | 必含字段 |
| --- | --- | --- |
| SHMEM 算子 | `[PERF]` | `e2e_us`, `kernel_us`, `e2e_bus_bandwidth_GBps`, `kernel_bus_bandwidth_GBps`, `payload_bytes` |
| HCCL/aclnn baseline | `[BASELINE_PERF]` | 同上；额外 `api=HcclAlltoAllV` 等 |

- **`kernel_bus_bandwidth_GBps`** — **达标对比 MUST 用此字段**
- **`e2e_bus_bandwidth_GBps`** 仅作参考

对比通过 grep 上述两行 + `perf_compare.py` 解析，**禁止**手工抄数。

## 3. 分阶段性能采集与离线对比

三阶段采集流程（A baseline → B SHMEM → C 离线对比）的命令代码段与隔离规则见 [perf-workflow.md](perf-workflow.md)。流程概览见 [performance-eval-guide.md](performance-eval-guide.md)。Docker 内执行见 [docker-exec-contract.md](../../shmem-ops-dev/references/docker-exec-contract.md)。

## 4. 聊天界面输出

完整规范见 [perf-chat-output-spec.md](perf-chat-output-spec.md)。**NEVER** 只输出 SHMEM 单边 `[PERF]`。

## 5. AllToAllV baseline 选型（参考）

1. 优先 `HcclAlltoAllV` / `HcclAlltoAllVC`
2. baseline 复用算子 `op_host_plan.cpp` 解析 `sendcounts.bin`
3. HCCL comm：`HcclGetRootInfo` + 文件广播 + `HcclCommInitRootInfo`

## 6. performance_report.md 必填对比节

§3.5 **MUST** 含 kernel_bus_bandwidth_GBps 对比；§7 **MUST** 说明达标/未达标原因。

## 7. 反模式

- ❌ 同一 shell / docker exec 内先 SHMEM 再 HCCL
- ❌ 同一流程内嵌启动 baseline + shmem
- ❌ baseline 用 Python dist 代替 C++ HCCL
- ❌ 对比表只写文件路径，聊天无表格
- ❌ 在宿主机无 NPU 环境伪造 perf 数据
