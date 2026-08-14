# INC Dispatch / Combine

## 中文

### 这个目录是干什么的？

这里是当前维护的 **INC Dispatch/Combine（简称 INC DC）** 实现树。  
语义上对应 MoE 的 **Dispatch / Combine 两段通信**：

1. **Dispatch**：按路由把 token 从 worker 发送到专家所在位置（经 INC 中转）。
2. **Combine**：把各专家贡献收集并加权归约，再写回源 token 布局。

历史 AG/RS、阶段 gate、未进入产品闭包的实验实现已经移除。  
目录根部故意不放源码，避免「靠文件名前缀猜所有权」；所有权以物理路径为准。

**新人快速了解单 INC Dispatch/Combine 主逻辑（推荐首读）：**
[`single_inc/QUICKSTART.md`](single_inc/QUICKSTART.md)

**应用直接调用（推荐唯一入口）：**
[`common/api/inc_dc_single_inc.hpp`](common/api/inc_dc_single_inc.hpp)

**可运行的完整调用例子：**
[`common/examples/single_inc_api/inc_dc_single_inc_api_example.cpp`](common/examples/single_inc_api/inc_dc_single_inc_api_example.cpp)

**当前 sweep 进度 / 环境 / baseline：**
[`single_inc/SWEEP_STATUS.md`](single_inc/SWEEP_STATUS.md)

### 目录边界

- 把 **稳定公开 ABI**（`common/`）与 **单 INC 实现**（`single_inc/`）分开，框架只依赖前者。
- 把 **资格化脚本/测试** 与运行库分开，二进制安装包可以裁剪，但源码仓保留门禁。

### 子目录

| 目录 | 用途 | 为什么要有 | 交付属性 |
|---|---|---|---|
| `common/` | 稳定 C API、共享协议、硬件策略、C 接入示例 | 框架与单 INC 的契约层；改 ABI 必须可控 | 生产必需 |
| `single_inc/` | 单 INC Dispatch、Combine、常驻服务、计划编译 | 当前正式星型产品数据面 | 生产必需 |
| `scripts/` | 真机 case launcher 与随机 token-plan sweep | 强制拓扑/空闲保护，避免手跑绕过门禁 | 资格化工具 |
| `tests/` | Host 侧 API / planner / backend 回归 | 交付门禁；不依赖真机也能挡 ABI 漂移 | 交付门禁 |

### 构建入口

正式构建入口在 `examples/inc/CMakeLists.txt`（由上层 `shmem` 构建系统拉入）：

```bash
# 在 shmem 的父目录执行；若已在 shmem/ 内则用 cmake -S .
cmake -S shmem -B build -DCMAKE_BUILD_TYPE=Release -DUSE_EXAMPLES=ON
cmake --build build -j --target \
  inc_dc_single_inc_stream inc_dc_sv2_dyn_csr_combine
```

真机资格化请走 `scripts/single_inc/`（裸跑 bin 会绕过拓扑/空闲门禁，不作交付证据）。  
注意：exe/target 名可能仍带 `sv2`，**源文件以 CMake 列表中的稳定名为准**（见 `single_inc/combine/README.md`）。
