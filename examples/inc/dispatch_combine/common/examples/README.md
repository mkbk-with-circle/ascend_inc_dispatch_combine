# C API 接入示例

这些示例不进入运行库；其中 `.c` 文件同时作为公开 C ABI 的 C11 编译门禁。

| 目录 | 适用场景 |
|---|---|
| `easy_api/` | 分步骤学习 init → route plan → Dispatch → Combine |
| `inference_api/` | prepare-once 的推理热路径；含可运行、带正确性检查的完整 CPU mock 示例 |

首次接入建议直接阅读并运行
[`inference_api/inc_dc_inference_api_example.cpp`](inference_api/inc_dc_inference_api_example.cpp)。
