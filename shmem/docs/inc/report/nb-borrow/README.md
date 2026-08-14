# nb-borrow 实验结果

本目录只保存 16 卡 910B2C `nb-borrow` 环境产生的数据。原有
`910b-yuanmingyu` 环境的报告和扫描结果保持原样，不在这里覆盖或缩放。

Fusion 数据统一入口：[`FUSION_KERNEL_RESULTS.md`](FUSION_KERNEL_RESULTS.md)。

- [`single_inc_inline_v2_dispatch_20260814`](single_inc_inline_v2_dispatch_20260814/README.md)：
  暂停开发的 V2 complete-token inline-route Dispatch 实验数据；INC 无 token-plan
  预知、24/48 AIV，W2/W4 大消息 20 次稳定性与通用路由回归。它不是当前公开
  `SingleInc` V1 API 的默认 backend，也不覆盖/替代 V1 结果。
- `single_inc_overlap_20260808T113214Z/`：W2/W4、单算子 128 MiB 的单 INC
  交叠资格化测试，包含单独运行、同时启动和错峰 500 微秒三种时序。
- [`fusion_kernel_release_20260810`](fusion_kernel_release_20260810/README.md)：当前 ABI 13 发布候选；冻结的 `fused_inc`
  最优路径、16–1024 token 扩展 sweep、当前 native graph/eager 对照、不同权重模式和失败边界。
- [`fusion_kernel_qualified_path_20260811`](fusion_kernel_qualified_path_20260811/README.md)：记录原始资格化源码快照 `780334b` / ABI 13
  `fused_inc`，以及重新构建后的 W2/W4 全量 golden、多 packet 与稳定性复核结果。
  同一源码已在当前分支按 INC 核心、fusion 核心、框架接入和构建接线拆成逻辑提交。
- [`release_validation_20260811`](release_validation_20260811/README.md)：清理后 fresh build、100 轮稳定性和 W2/W4 真机验证。

当前容器、镜像 digest、构建参数和完整复现命令见
`examples/inc/fusion_kernel/framework/vllm_ascend/RUNBOOK_NB_VLLM.md`。
