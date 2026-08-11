# nb-borrow 实验结果

本目录只保存 16 卡 910B2C `nb-borrow` 环境产生的数据。原有
`910b-yuanmingyu` 环境的报告和扫描结果保持原样，不在这里覆盖或缩放。

- `single_inc_overlap_20260808T113214Z/`：W2/W4、单算子 128 MiB 的单 INC
  交叠资格化测试，包含单独运行、同时启动和错峰 500 微秒三种时序。
- `fusion_kernel_20260809/`：Fusion kernel 算子级与 persistent-service sweep。
- `fusion_kernel_baselines_20260809/`：算子基线和早期原生 vLLM 对照。
- `fusion_kernel_vllm_bridge_20260809/`：Torch/vLLM bridge、路由和 ABI v3/v6
  历史资格化数据。
- `fusion_kernel_vllm_e2e_comparison_20260809/`：保留的 ABI v9 W2/W4 × 五路径
  历史正式 vLLM 端到端结果和原始 JSON。
- `fusion_kernel_release_20260810/`：当前 ABI 13 发布候选；冻结的 `fused_inc`
  最优路径、16–1024 token 扩展 sweep、native graph/eager、不同权重模式和失败边界。
- `fusion_kernel_qualified_path_20260811/`：记录原始资格化源码快照 `780334b` / ABI 13
  `fused_inc`，以及重新构建后的 W2/W4 全量 golden、多 packet 与稳定性复核结果。
  同一源码已在当前分支按 INC 核心、fusion 核心、框架接入和构建接线拆成逻辑提交。

当前容器、镜像 digest、构建参数和完整复现命令见
`examples/inc/fusion_kernel/framework/vllm_ascend/RUNBOOK_NB_VLLM.md`。
