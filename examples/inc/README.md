# INC examples

## 中文

本目录只保留两套正式交付：

- `dispatch_combine/`：单 INC Dispatch/Combine、公开 API、测试和资格化脚本；
- `fusion_kernel/`：ABI 13 单 INC fusion kernel、prepared API、vLLM-Ascend
  接入层、测试与运行手册。

两者都由本目录的 `CMakeLists.txt` 构建。共享 FP16、UB、vector-reduction、gather、
拓扑和资源策略位于 `dispatch_combine/common/platform/`。历史 AG/RS、`inc_s09`、
MegaMoE/NVIDIA 参考源码、临时探针、构建产物和 launcher 日志均不属于当前依赖闭包。

### API 快速入口

| 功能 | 公开 API | 完整示例 |
|---|---|---|
| 单 INC Dispatch/Combine | [`inc_dc_single_inc.hpp`](dispatch_combine/common/api/inc_dc_single_inc.hpp) | [`inc_dc_single_inc_api_example.cpp`](dispatch_combine/common/examples/single_inc_api/inc_dc_single_inc_api_example.cpp) |
| 单 INC Fusion Kernel | [`inc_fusion_api.h`](fusion_kernel/ascend/inc_fusion_api.h) | [`fusion_kernel/examples/`](fusion_kernel/examples/README.md) |

Fusion trace 解析入口见 [`fusion_kernel/tools/`](fusion_kernel/tools/README.md)。

最新结构化结果和硬件 profile 位于 `docs/inc/`。Fusion 对比数据可从
[`FUSION_KERNEL_RESULTS.md`](../../docs/inc/report/nb-borrow/FUSION_KERNEL_RESULTS.md)
直接进入：

- Fusion 最终资格化：[报告](../../docs/inc/report/nb-borrow/fusion_kernel_qualified_path_20260811/README.md) · [CSV](../../docs/inc/report/nb-borrow/fusion_kernel_qualified_path_20260811/results.csv)；
- ABI 13 sweep 与 vLLM 对照：[报告](../../docs/inc/report/nb-borrow/fusion_kernel_release_20260810/README.md) · [CSV](../../docs/inc/report/nb-borrow/fusion_kernel_release_20260810/results.csv)；
- 单 INC 状态：[`single_inc_LIVE_STATUS.md`](../../docs/inc/report/single_inc_LIVE_STATUS.md)。

### 新人从这里开始

如果从未接触过这棵树，需要先建立单 INC Dispatch/Combine 的正确心智模型
（拓扑、设备时间线、planning、runtime 接线、如何跑一个 case），请读：

**[`dispatch_combine/single_inc/QUICKSTART.md`](dispatch_combine/single_inc/QUICKSTART.md)**

中文主文 + English summary；结论均可对照源码符号（`StreamWorker` /
`StreamInc` / `DynCsrCtrl` / `NativeIncService` 等）。

**当前单 INC sweep 进度 / 环境 / baseline gates：**
[`dispatch_combine/single_inc/SWEEP_STATUS.md`](dispatch_combine/single_inc/SWEEP_STATUS.md)

然后浏览 `dispatch_combine/README.md`，看 `common/`、`single_inc/`、`scripts/`、
`tests/` 如何划分所有权。

Fusion 的协议、token-wave timeline、调用接口和 nb 复现命令分别见
[`fusion_kernel/PRINCIPLES.md`](fusion_kernel/PRINCIPLES.md)、
[`fusion_kernel/README.md`](fusion_kernel/README.md) 和
[`fusion_kernel/framework/vllm_ascend/RUNBOOK_NB_VLLM.md`](fusion_kernel/framework/vllm_ascend/RUNBOOK_NB_VLLM.md)。

---

## English

This directory keeps two formal deliveries:

- `dispatch_combine/`: single-INC Dispatch/Combine, public API, tests, and
  qualification scripts;
- `fusion_kernel/`: ABI 13 single-INC fusion kernel, prepared API, vLLM-Ascend
  integration, tests, and runbooks.

Both are built from this directory's `CMakeLists.txt`. Shared FP16, UB,
vector-reduction, gather, topology, and resource policy live in
`dispatch_combine/common/platform/`. Historical AG/RS, `inc_s09`, MegaMoE/NVIDIA
reference sources, temporary probes, build artifacts, and launcher logs are
not in the current dependency closure.

### API quick entry

| Feature | Public API | Complete example |
|---|---|---|
| Single-INC Dispatch/Combine | [`inc_dc_single_inc.hpp`](dispatch_combine/common/api/inc_dc_single_inc.hpp) | [`inc_dc_single_inc_api_example.cpp`](dispatch_combine/common/examples/single_inc_api/inc_dc_single_inc_api_example.cpp) |
| Single-INC Fusion Kernel | [`inc_fusion_api.h`](fusion_kernel/ascend/inc_fusion_api.h) | [`fusion_kernel/examples/`](fusion_kernel/examples/README.md) |

Fusion trace parsing starts at [`fusion_kernel/tools/`](fusion_kernel/tools/README.md).

Latest structured results and hardware profiles live under `docs/inc/`. Fusion
comparison data is indexed from
[`FUSION_KERNEL_RESULTS.md`](../../docs/inc/report/nb-borrow/FUSION_KERNEL_RESULTS.md):

- Fusion final qualification: [report](../../docs/inc/report/nb-borrow/fusion_kernel_qualified_path_20260811/README.md) · [CSV](../../docs/inc/report/nb-borrow/fusion_kernel_qualified_path_20260811/results.csv);
- ABI 13 sweep vs vLLM: [report](../../docs/inc/report/nb-borrow/fusion_kernel_release_20260810/README.md) · [CSV](../../docs/inc/report/nb-borrow/fusion_kernel_release_20260810/results.csv);
- Single-INC status: [`single_inc_LIVE_STATUS.md`](../../docs/inc/report/single_inc_LIVE_STATUS.md).

### New here?

If you have never touched this tree and need a trustworthy mental model of
**single-INC Dispatch and Combine** (topology, device timelines, planning,
runtime wiring, how to run one case), read:

**[`dispatch_combine/single_inc/QUICKSTART.md`](dispatch_combine/single_inc/QUICKSTART.md)**

Chinese main text plus an English summary; claims can be checked against source
symbols (`StreamWorker` / `StreamInc` / `DynCsrCtrl` / `NativeIncService`, etc.).

**Current single-INC sweep progress / environments / baseline gates:**
[`dispatch_combine/single_inc/SWEEP_STATUS.md`](dispatch_combine/single_inc/SWEEP_STATUS.md)

Then skim `dispatch_combine/README.md` for how `common/`, `single_inc/`,
`scripts/`, and `tests/` split ownership.

Fusion protocol, token-wave timeline, call interface, and nb reproduction
commands are in [`fusion_kernel/PRINCIPLES.md`](fusion_kernel/PRINCIPLES.md),
[`fusion_kernel/README.md`](fusion_kernel/README.md), and
[`fusion_kernel/framework/vllm_ascend/RUNBOOK_NB_VLLM.md`](fusion_kernel/framework/vllm_ascend/RUNBOOK_NB_VLLM.md).
