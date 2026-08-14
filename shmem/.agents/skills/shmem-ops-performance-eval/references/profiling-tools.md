# SHMEM 性能采集工具与方法

本文说明 SHMEM 算子可用的性能采集工具、使用方法和指标计算规则。

## 1. 采集工具

### 1.1 msprof（推荐）

MindStudio Profiler，昇腾官方性能采集工具。采集 kernel 级 task_time、通信时间和硬件计数器。

**使用方式**：在 `scripts/run.sh` 中包裹可执行程序：

```bash
msprof --application="${EXEC_BIN} ${ARGS}" --output=./output/
```

**产出**：
- `output/PROF_*/mindstudio_profiler_output/task_time_*.csv`：kernel 维度执行时间
- 主要字段：`kernel_type`、`task_time(us)`

**解析方法**：参考 `data_statistic.py` 模式：
```python
df = pd.read_csv(task_time_file)
df = df[df['kernel_type'].str.contains("CORE", na=False)]
# 跳过 warmup 轮，取 perf 轮平均
avg_time = df.iloc[warmup:warmup+perf_cycles]["task_time(us)"].mean()
```

### 1.2 SHMEM Cycle Profiling

SHMEM 内置的 Device 侧 cycle 级打点，基于 `SHMEMI_PROF_START(id)` / `SHMEMI_PROF_END(id)` 宏对。

> **完整操作指南**：[device-profiling-guide.md](device-profiling-guide.md) —— 涵盖宏原理、kernel 侧 5 类 frame 打点规范、Host 侧 `aclshmemx_get_prof` 调用模式、数据填入性能报告的方法。本节仅提供概要。

**使用方式**：在 kernel 代码中插入打点：
```cpp
#include "utils/prof/shmemi_prof.h"

SHMEMI_PROF_START(0);  // 开始计时 frame 0
// ... 通信或计算代码 ...
SHMEMI_PROF_END(0);    // 结束计时 frame 0
```

**原理**：通过 `AscendC::GetSystemCycle()` 获取硬件 cycle 计数，累加到 `device_state->profs->block_prof[block_idx].cycles[frame_id]`。每个 frame 同时记录调用次数 `ccount`。

**适用场景**：
- 定位 kernel 内某段代码的 cycle 开销
- 分析通信/计算/同步各段占比
- 对比优化前后特定段的 cycle 变化

**限制**：
- frame 数量受 `ACLSHMEM_CYCLE_PROF_FRAME_CNT` 限制
- 只在指定 PE（`device_state->profs->pe_id == shmem_my_pe()`）的前几个 block 上采集
- `pipe_barrier(PIPE_ALL)` 会引入额外同步开销，影响绝对数值但不影响对比结论

### 1.3 mssanitizer

内存访问检查工具，不直接用于性能采集，但可用于排除因越界导致的性能异常。

```bash
mssanitizer --log-level=error ${EXEC_BIN} ${ARGS}
```

## 2. 性能指标计算

指标定义和计算公式见 [timing-and-metrics-standard.md](timing-and-metrics-standard.md) §4。此处仅补充工具特有的注意事项：

- msprof 采集时，`latency_us` 从 `task_time(us)` CSV 字段直接获取，多 PE 取最大值
- `logical_payload_bytes` 口径和 `bus_factor` 推导详见 timing-and-metrics-standard.md §4.1 和 §4.3
- `peak_bandwidth_GBps` 来源和 `bandwidth_utilization_percent` 计算详见 timing-and-metrics-standard.md §4.4

## 3. 采集规范

### 3.1 Warmup

- msprof 采集：前 10 次 kernel 调用为 warmup，不计入统计
- 手动 repeat：前 10 轮为 warmup，后续 40 轮取平均

### 3.2 多卡平均

多 PE 运行时，每个 PE 产出独立的 profiling 数据。性能指标取所有 PE 的平均值（延迟取最大值）。

### 3.3 结果记录格式

```
op_name: allgather
dtype: int32
shape: M=1024, N=8
n_pes: 2
data_bytes: 32768
latency_us: 45.2
algo_bandwidth_GBps: 0.725
kernel_bus_bandwidth_GBps: 0.363
baseline: HCCL AllGather
baseline_latency_us: 52.1
ratio: 115.3%
```
