# Pull V2 device primitives

当前只保留 Combine device kernel 的直接依赖：

| 文件 | 用途 |
|---|---|
| `inc_dc_platform_capabilities.h` | AIV UB 与平台容量常量 |
| `inc_dc_ub_tile_aicore.h` | UB tile 尺寸计算 |
| `inc_dc_vector_reduce_aicore.h` | FP32 partial 向量归约 |

这些头文件由 `inc_dc_pull_combine_v2_device_kernel.cpp` 直接或传递包含。
