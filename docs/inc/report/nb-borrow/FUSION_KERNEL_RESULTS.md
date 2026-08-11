# Fusion Kernel 对比实验入口

以下数据均来自 `nb-borrow`（Ascend 910B2C）。当前代码对应 ABI 13；旧数据保留但不会覆盖当前结果。

| 数据集 | 版本与用途 | 报告 | 机读数据 |
|---|---|---|---|
| 当前端到端对比 | ABI 13；`fused_inc`、`serial_inc`、native eager/graph，W2/W4、16–1024 token | [发布报告](fusion_kernel_release_20260810/README.md) | [results.csv](fusion_kernel_release_20260810/results.csv) · [W4 fused JSON](fusion_kernel_release_20260810/raw/fused_inc_w4_extended.json) · [完整性清单](fusion_kernel_release_20260810/MANIFEST.sha256) |
| 当前算子资格化 | ABI 13；W2/W4、全量 golden、多 packet、D/C overlap | [资格化报告](fusion_kernel_qualified_path_20260811/README.md) | [results.csv](fusion_kernel_qualified_path_20260811/results.csv) |
| 五路径端到端矩阵 | 历史 ABI 9；native、serial/fused × SHMEM/INC，prefill 128 | [五路径报告](fusion_kernel_vllm_e2e_comparison_20260809/README.md) | [summary.csv](fusion_kernel_vllm_e2e_comparison_20260809/summary.csv) · [W4 fused INC JSON](fusion_kernel_vllm_e2e_comparison_20260809/raw/fused_inc_w4_prefill128/result.json) |
| 早期算子基线 | 算子级 Dispatch/FFN/Combine 对照 | [基线说明](fusion_kernel_baselines_20260809/README.md) | [operator_comparison.csv](fusion_kernel_baselines_20260809/operator_comparison.csv) |

当前 ABI 13 的 W4 端到端均值摘要：

| input token | fused INC | native eager | native graph |
|---:|---:|---:|---:|
| 32 | 72.573 ms | 71.582 ms | 31.109 ms |
| 128 | 73.497 ms | 68.411 ms | 34.279 ms |
| 512 | 81.819 ms | 69.268 ms | 62.998 ms |

因此当前结论是：Fusion 已有完整对比数据，但 ABI 13 端到端尚未稳定超过原生 vLLM；不要把历史五路径结果或算子 D/C 窗口收益当成当前端到端加速结论。
