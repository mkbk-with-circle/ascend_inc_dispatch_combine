# SHMEM 性能 Baseline 选择策略

性能采集需要明确的 baseline 作为对比基准。硬规则：

- 有 HCCL/aclnn baseline 时，**MUST** 接入 baseline
- 有 baseline 时，达标线默认 current ≥ baseline 的 **80%**
- 有 baseline 但未达标时，进入性能优化（最多 5 轮）
- 无直接 baseline 时，**MUST** 记录搜索过程，并使用 metric_only 指标验收；通信算子能计算带宽利用率时 **NEVER** 低于 20%

按以下优先级选择 baseline。

## 1. 优先级 1：HCCL / aclnn 算子库

**适用场景**：算子功能与 HCCL 集合通信算子完全或部分对应。

> **重要**：声称"无 baseline"前 **MUST** 逐一排查以下 HCCL 清单。

### 1.1 HCCL 集合通信算子清单

通信算子在此清单中查找 baseline：

| SHMEM 算子 | HCCL Baseline | HCCL API | 说明 |
| --- | --- | --- | --- |
| AllGather | HcclAllGather | `HcclAllGather(sendBuf, recvBuf, count, dataType, comm, stream)` | 每 PE 数据收集到全部 PE |
| AllReduce | HcclAllReduce | `HcclAllReduce(sendBuf, recvBuf, count, dataType, op, comm, stream)` | 全局归约 |
| ReduceScatter | HcclReduceScatter | `HcclReduceScatter(sendBuf, recvBuf, count, dataType, op, comm, stream)` | 归约后分片 |
| AllToAll | HcclAlltoAll | `HcclAlltoAll(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, comm, stream)` | 全交换 |
| Broadcast | HcclBroadcast | `HcclBroadcast(buf, count, dataType, root, comm, stream)` | 广播 |
| Reduce | HcclReduce | `HcclReduce(sendBuf, recvBuf, count, dataType, op, root, comm, stream)` | 归约到 root |
| Scatter | HcclScatter | `HcclScatter(sendBuf, recvBuf, count, dataType, root, comm, stream)` | root 分发 |
| Gather | HcclGather | `HcclGather(sendBuf, recvBuf, count, dataType, root, comm, stream)` | 收集到 root |

### 1.2 aclnn 扩展算子

| SHMEM 算子 | aclnn Baseline | API | 说明 |
| --- | --- | --- | --- |
| AllToAll（V 版本） | aclnnAlltoAllV | `aclnnAlltoAllV(...)` | 变长全交换 |
| AllToAllV（HCCL 回退） | HcclAlltoAllV | `HcclAlltoAllV(...)` | CANN 无独立 aclnn 头文件时的 **MUST** 回退 |

> **查找方法**：在 CANN 安装目录下搜索 `aclnn` 开头的头文件，按算子语义匹配关键词（如 AllToAllV 搜索 `alltoall`）。也可查阅 CANN 在线文档的「aclnn 算子接口」章节。
>
> **AllToAllV 回退**：若 `${ASCEND_HOME_PATH}/include` 无独立 `aclnnAlltoAllV` 头文件，**MUST** 使用 `HcclAlltoAllV`（`hccl/hccl.h`），参考 `custom-ops/alltoallv/baseline/` 与 [baseline-compare-workflow.md](baseline-compare-workflow.md)。

### 1.3 Baseline 接入方式（C++ 可执行文件）

HCCL 和 aclnn baseline **MUST** 以 C++ 可执行文件方式接入（**NEVER** 使用 Python 脚本），在算子目录的 `baseline/` 子目录下编写独立的 baseline 测试程序和 `CMakeLists.txt`，**NEVER** 将 baseline 源码或编译 target 放在算子 `src/` 或根目录 `CMakeLists.txt` 中。

**HCCL baseline 示例**（以 AllToAll 为例）：

```cpp
#include <hccl/hccl.h>
#include <acl/acl.h>

bool LaunchHcclBaseline(void *sendBuf, void *recvBuf,
                         int64_t count, HcclComm comm, aclrtStream stream) {
    HcclResult ret = HcclAlltoAll(
        sendBuf, static_cast<uint64_t>(count), HCCL_DATA_TYPE_FP16,
        recvBuf, static_cast<uint64_t>(count), HCCL_DATA_TYPE_FP16,
        comm, stream);
    return (ret == HCCL_SUCCESS);
}

aclrtEvent startEvent, endEvent;
aclrtCreateEvent(&startEvent);
aclrtCreateEvent(&endEvent);

for (int i = 0; i < warmup; i++) {
    LaunchHcclBaseline(sendBuf, recvBuf, count, comm, stream);
}
aclrtSynchronizeStream(stream);

aclrtRecordEvent(startEvent, stream);
for (int i = 0; i < iters; i++) {
    LaunchHcclBaseline(sendBuf, recvBuf, count, comm, stream);
}
aclrtRecordEvent(endEvent, stream);
aclrtSynchronizeStream(stream);

float elapsed_ms;
aclrtEventElapsedTime(&elapsed_ms, startEvent, endEvent);
float avg_us = elapsed_ms * 1000.0f / iters;
printf("[BASELINE_PERF] pe=%d e2e_us=%.2f kernel_us=%.2f algo_bandwidth_GBps=%.2f e2e_bus_bandwidth_GBps=%.2f kernel_bus_bandwidth_GBps=%.2f\n", my_pe, avg_us, avg_us, ...);
```

**CMake 集成**（baseline 的 CMakeLists.txt 和源码 **MUST** 放在 `baseline/` 目录下，**NEVER** 混入算子 `src/`）：

```cmake
add_executable(${OP_NAME}_hccl_baseline baseline/src/${OP_NAME}_hccl.cpp)
target_link_libraries(${OP_NAME}_hccl_baseline PRIVATE hccl ascendcl)
```

**aclnn baseline 示例**（以 AllToAllV 为例）：

```cpp
#include <aclnn/aclnn_alltoallv.h>
#include <acl/acl.h>

uint64_t workspaceSize = 0;
aclnnAlltoAllVGetWorkspaceSize(...);
aclnnAlltoAllV(workspace, workspaceSize, executor, stream);
```

**记录格式**：

```yaml
baseline:
  type: "cann_operator"
  source: "HCCL"
  api: "HcclAlltoAll"
  latency_us: 125.3
  command: "./alltoall_hccl --n_pes 8 --count 1048576 --dtype fp16 --iters 100"
  environment:
    cann_version: "8.0.RC1"
    device: "Atlas 800T A2"
    n_pes: 8
```

**AllToAllV 对比命令**（含 SHMEM + baseline；完整参数见 [perf-workflow.md §1](perf-workflow.md)）：

```bash
OP=alltoallv
DEVICE_LIST=0,1,2,3,4,5,6,7
BASE_COUNT=8388608
DTYPE=float16
WARMUP=10
MEASURE=40
ITERS=$((WARMUP + MEASURE))  # 50 total

# 阶段 A（独立 shell）：baseline
bash "${SHMEM_REPO}/custom-ops/${OP}/baseline/scripts/run_baseline.sh" \
  "${DEVICE_LIST}" "${BASE_COUNT}" "${DTYPE}" "${ITERS}"

# 阶段 B（新 shell，间隔 ≥30s）：SHMEM
bash "${SHMEM_REPO}/custom-ops/${OP}/scripts/perf.sh" \
  "${DEVICE_LIST}" "${BASE_COUNT}" "${DTYPE}" "${ITERS}"

# 阶段 C（离线）
python3 "${SHMEM_REPO}/custom-ops/scripts/lib/perf_compare.py" \
  --shmem-log "${SHMEM_REPO}/custom-ops/${OP}/data/perf/shmem_${BASE_COUNT}_${DTYPE}.log" \
  --baseline-log "${SHMEM_REPO}/custom-ops/${OP}/data/perf/baseline_${BASE_COUNT}_${DTYPE}.log"
```

完整工作流见 [baseline-compare-workflow.md](baseline-compare-workflow.md)。

## 2. 优先级 2：指标测试（无直接 baseline）

**适用场景**：既找不到对应的 HCCL 算子，也找不到 aclnn 扩展算子。

使用 metric_only 前 **MUST** 说明已逐一检查过以下来源且均无对应实现：HCCL 清单（§1.1）、aclnn 扩展（§1.2）、已有 SHMEM example、用户参考实现。metric_only 不是跳过性能采集或指标验收的理由。

**测试策略**：

1. **时延测试**：测量端到端时延，分析是否存在明显的性能瓶颈，与理论时延对比（如果可计算）
2. **带宽测试**：计算算法带宽和总线带宽，与硬件峰值带宽对比；对通信算子，能计算带宽利用率时 **MUST** 输出 utilization，且 **NEVER** 低于 20%
3. **带宽利用率**：对于通信算子，测量带宽利用率，计算理论带宽与实际带宽的比值

**判断标准**：

- 时延明显过高（简单搬运操作耗时超过设计或理论阈值）
- 通信算子带宽利用率低于 20%
- 通信算子的带宽利用率低于 20%，或 wait/sync 占比显示明显瓶颈
- 存在明显的空闲等待时间

**指标计算公式**（来自 [timing-and-metrics-standard.md](timing-and-metrics-standard.md)，此处仅引用，不作为独立实现标准）：

```text
algo_bandwidth_GBps = logical_payload_bytes / latency_s / 1e9
kernel_bus_bandwidth_GBps  = algo_bandwidth_GBps × bus_factor   # kernel 口径 — 达标主指标
e2e_bus_bandwidth_GBps     = algo_bandwidth_GBps(e2e) × bus_factor  # e2e 口径 — 仅参考
bandwidth_utilization_percent = kernel_bus_bandwidth_GBps / peak_bandwidth_GBps × 100
```

> 实际性能采集 **MUST** 通过算子 `--perf` 输出 `[PERF]` 行获取指标值，**NEVER** 手工计算或使用 Python 脚本替代 C++ 性能测试程序。

**记录格式**：

```yaml
baseline:
  type: "metric_only"
  current_latency_us: 245.8
  logical_payload_bytes: 4.0 MB
  algo_bandwidth_GBps: 16.3
  kernel_bus_bandwidth_GBps: 8.15
  peak_bandwidth_GBps: 300
  bandwidth_utilization_percent: 2.7
  bottleneck_analysis: 带宽利用率极低
```

## 3. Baseline 选择决策树

```
Step 1: 查找 HCCL 集合通信算子清单（§1.1）
        ├─ 有完全对应 → 使用 HCCL 算子作为 baseline
        └─ 无 → Step 2

Step 2: 查找 aclnn 扩展算子（§1.2）
        ├─ 有对应算子 → 使用 aclnn 算子作为 baseline
        └─ 无 → Step 3

Step 3: 使用指标测试（§2，metric_only）
        ├─ 测量时延和带宽利用率
        └─ 带宽利用率 ≥ 20%
```

**关键提醒**：

- 声称"无 baseline"前 **MUST** 走完 Step 1-2，并在报告中记录每步的搜索结果
- 很多集合通信算子在 HCCL 中有对应实现，不要遗漏 Step 1
- HCCL 库包含 AllToAll 等算子，不要误认为 HCCL 只有 AllGather/AllReduce/ReduceScatter

## 4. Baseline 记录要求

无论使用哪种 baseline，都 **MUST** 记录：

```yaml
baseline:
  type: "hccl" | "aclnn" | "metric_only"
  source: "HCCL" | "aclnn" | "metric_only"
  api: "HcclAlltoAll"
  measurement:
    latency_us: 210.5
    algo_bandwidth_GBps: 19.0
    kernel_bus_bandwidth_GBps: 28.5
    peak_bandwidth_GBps: 28.0
    bandwidth_utilization_percent: 9.5
    effective_flops: 123000000000000.0
    compute_utilization_percent: 42.0
  command: "./baseline_test --n_pes 8 --count 1048576"
  environment:
    cann_version: "8.0.RC1"
    device: "Atlas 800T A2"
    soc: "Ascend910B3"
    n_pes: 8
  search_process: "已检查 HCCL 清单、aclnn 清单，选定 ..."
```

## 附录 A：可选 stitched baseline（非默认流程）

> **注意**：当前标准流程（design / performance-eval）的 `baseline_search` **不要求**排查拼接方案。仅当 HCCL/aclnn 均无对应且用户明确要求 stitched 对比时，可参考本节实现 C++ 拼接 baseline。

| 通信算子 | 拼接方案 | 组件 |
| --- | --- | --- |
| KV Shuffle | Put + Get + Barrier | `aclshmemx_mte_put_nbi` + `aclshmemx_mte_get_nbi` + `aclshmem_barrier_all` |

拼接 baseline **MUST** 用 C++ 可执行文件实现，**NEVER** 使用 Python 脚本。
