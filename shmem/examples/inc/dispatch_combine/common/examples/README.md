# C API 接入示例

这些示例不进入运行库；其中 `.c` 文件同时作为公开 C ABI 的 C11 编译门禁。

| 目录 | 适用场景 |
|---|---|
| `inference_api/` | **推荐**的 `inc_dc_single_inc_api` 完整示例；可运行且含数值检查 |
| `easy_api/` | 仅供需要自定义底层生命周期的高级接入者 |

首次接入只需阅读并运行
[`inference_api/inc_dc_inference_api_example.cpp`](inference_api/inc_dc_inference_api_example.cpp)。
