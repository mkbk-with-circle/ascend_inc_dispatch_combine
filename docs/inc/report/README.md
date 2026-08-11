# INC 最新报告索引

`docs/inc/report` 只保留当前仍有效的最小证据集。历史调优快照、rank 日志、中间 CSV/JSON、已失效 gate、`examples/inc/report` 旧阶段 gate 和重复可视化已于 2026-08-04 清理。

## Single-INC

- `single_inc_LIVE_STATUS.md`：硬 gate、当前状态和开放项。
- `single_inc_native_api_closure_cann91_20260804.md`：真实 Dispatch/Combine Framework/Easy provider、registered view、复用与同口径带宽证据。
- `single_inc_overlap_theory_cann91_20260804.md`：CANN 9.1 双向理论与 overlap 结论。
- [`nb-borrow/`](nb-borrow/README.md)：nb-borrow 专属结果；与 yuanmingyu 历史 sweep 分目录保存。
- [Fusion Kernel 对比实验总入口](nb-borrow/FUSION_KERNEL_RESULTS.md)：当前 ABI 13、算子资格化、历史五路径和原始数据的直接链接。
- [`fusion_kernel_release_20260810`](nb-borrow/fusion_kernel_release_20260810/README.md)：ABI 13 当前最优 `fused_inc` 的 W2/W4、16–1024 token sweep、native graph/eager、W8A8 参考、四路径状态和原始 JSON。
- `single_inc_final_stress_cann91_20260804.md`：最新 sweep/连续下发压测汇总。
- `single_inc_nonpow2_sweep_cann91_20260805.md`：native D/C 非 2 次幂 token、128/256 MiB sweep，含启动失败复测。
- `single_inc_final_regular_sweep_cann91_20260804.json`：最新可机读正则 sweep。
- `single_inc_topk_rank_stress_cann91_20260804.{md,json}`：K>W/K>8、按 rank 去重、热点/ragged 与 100 epoch 压测。
- `single_inc_megatron_vllm_integration_contract_20260802.md`：当前框架集成契约。
- `ACTIVE_HW_PROFILE.md` 与 `env/ENV_STATUS_TEMPLATE.md`：环境指针/模板。
- 仓库只保留结构化结果、资格化摘要和复现入口；launcher 日志、PID/READY 文件、
  profiler trace 与已失效的中间调参结果不进入当前交付。

## Multi-INC

总结文档位于 `../multi_inc_dispatch_combine_delivery_report.md`。本目录保留：

- `multi_inc_delivery_validation_gate.json`：最新完整交付 gate。
- `multi_inc_delivery_conditional_lock_status.json`：当前 conditional lock 状态。
- `dispatch_transferred_bytes_metric_v2_gate.json`：最新 Dispatch 带宽分子口径。
- `dispatch_150_160_candidate_gate.json`：最新 150–160 GB/s Dispatch 候选数据。

清理后如需历史过程，应从测试脚本重新生成，不应再将中间快照当作当前结论。
