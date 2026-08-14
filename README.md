# ascend_inc_dispatch_combine

昇腾 910B 上的 **单 INC Dispatch / Combine** 与 **Fusion Kernel** 工作区。

当前正式交付都在 `shmem/examples/inc/`：

| 路径 | 内容 |
|---|---|
| [`shmem/examples/inc/dispatch_combine/`](shmem/examples/inc/dispatch_combine/README.md) | 单 INC Dispatch/Combine、公开 API、测试与资格化脚本 |
| [`shmem/examples/inc/fusion_kernel/`](shmem/examples/inc/fusion_kernel/README.md) | ABI 13 fusion kernel、prepared API、vLLM-Ascend 接入 |

新人先读 [`single_inc/QUICKSTART.md`](shmem/examples/inc/dispatch_combine/single_inc/QUICKSTART.md)。  
应用入口：[`inc_dc_single_inc_api.h`](shmem/examples/inc/dispatch_combine/common/api/inc_dc_single_inc_api.h)。

`cann-packages/`、构建产物和日志默认不入库。
