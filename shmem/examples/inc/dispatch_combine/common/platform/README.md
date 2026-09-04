# Pull V2 AICore Platform Headers

本目录仅保留 Pull V2 Combine kernel 直接依赖的三个硬件无关辅助头：

- `inc_dc_platform_capabilities.h`
- `inc_dc_ub_tile_aicore.h`
- `inc_dc_vector_reduce_aicore.h`

这里没有公开应用 API；应用入口位于 `single_inc/pull_combine/`。
