# ascend_inc_dispatch_combine

昇腾 Single-INC **Pull V2 Dispatch/Combine** 最小交付仓库。

当前树只保留：

- SHMEM 基础库；
- Pull V2 Dispatch/Combine 协议和设备 kernel；
- 公共 API 与完整调用示例；
- Host、设备 E2E、交叠和压力测试代码；
- nb-borrow W2/W4 汇总资格证据。

V1、inline-route、旧 framework/runtime、Fusion Kernel 和历史结果均已从当前树删除，
可通过 Git 标签 `archive/pre-minimal-v1-fusion-20260904` 完整恢复。

## 快速入口

- [最短调用流程](shmem/examples/inc/dispatch_combine/single_inc/QUICKSTART.md)
- [Pull V2 README](shmem/examples/inc/dispatch_combine/single_inc/pull_combine/README.md)
- [公共 API](shmem/examples/inc/dispatch_combine/single_inc/pull_combine/inc_dc_pull_v2_api.h)
- [完整示例](shmem/examples/inc/dispatch_combine/single_inc/pull_combine/inc_dc_pull_v2_api_example.cpp)
- [Dispatch/Combine 流程图](shmem/examples/inc/dispatch_combine/single_inc/pull_combine/FLOW.md)
- [当前 nb 结果](shmem/docs/inc/report/single_inc_LIVE_STATUS.md)

构建产物、原始 JSONL、PE 日志和 profiler trace 不进入 Git。
