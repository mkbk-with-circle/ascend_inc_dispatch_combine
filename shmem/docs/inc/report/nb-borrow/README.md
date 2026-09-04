# nb-borrow 实验结果

本目录是当前 Single-INC Pull V2 和 Fusion Kernel 的唯一结果环境：16× Ascend
910B2C、两个 8-card HCCS 平面、CANN 9.1.0-beta.3。

## Single-INC Pull V2

- [`pull_v2_qualified_20260904`](pull_v2_qualified_20260904/README.md)：W2/W4
  128 MiB 正式 gate、256 MiB 扩展、100-wave soak 与随机正确性；
- [`pull_v2_overlap_stress_20260905`](pull_v2_overlap_stress_20260905/README.md)：
  Dispatch/Combine 交叠、62-case size/skew/hotspot/ragged 压力矩阵；
- [`pull_many_get_peak_20260903`](pull_many_get_peak_20260903/README.md)：GET roofline
  的历史诊断口径，仅用于解释链路，不用于降低正式 gate。

## Fusion Kernel

- [`FUSION_KERNEL_RESULTS.md`](FUSION_KERNEL_RESULTS.md)：当前结果总入口；
- [`fusion_kernel_release_20260810`](fusion_kernel_release_20260810/README.md)：ABI 13
  发布候选与 W2/W4 sweep；
- [`release_validation_20260811`](release_validation_20260811/README.md)：fresh build、
  100 轮稳定性和真机验证。

逐 case JSON、PE 日志、build 产物和 profiler trace 不进入 Git。
