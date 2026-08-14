# SHMEM 对外头文件与库文件说明

> 范围：对外公开接口头文件，以及安装产物中的共享库（`.so`）。

## 1. 库文件总览

| 库文件 | 类型 | 用途 | 依赖版本信息 |
|--------|------|------|--------------|
| `libshmem.so` | 核心运行时 | Host 侧主库：初始化、堆、Team、Host RMA/同步/集合通信等 | 依赖 `ASCEND_HOME_PATH` 下 CANN 库（`runtime`/`ascendcl`/`tiling_api`/`platform`/`c_sec` 等）；链接 `libshmem_utils.so`；`SOC_TYPE=Ascend950` 时另链 `hcomm` |
| `libshmem_utils.so` | 工具/支撑库 | 日志、内部工具与公共支撑符号，供主库与 bootstrap 插件链接 | 无独立对外版本号；UDMA 开启时编译期依赖 nlohmann/json 头文件 |
| `aclshmem_bootstrap_config_store.so` | Bootstrap 插件（无 `lib` 前缀） | Default / UniqueID 初始化控制面（Config Store）。`ACLSHMEMX_INIT_WITH_DEFAULT` 与 `ACLSHMEMX_INIT_WITH_UNIQUEID` 均加载本库 | 运行时由 init 动态加载；链接 `libshmem_utils.so`；无独立版本号 |
| `aclshmem_bootstrap_uid.so` | Bootstrap 插件 | UniqueID bootstrap 相关库 | 链接 `libshmem_utils.so`；无独立版本号 |
| `aclshmem_bootstrap_mpi.so` | Bootstrap 插件（可选） | MPI 模式 bootstrap | CMake `find_package(MPI)` 成功时才编译；链接 `MPI::MPI_CXX` 与 `libshmem_utils.so` |
| `_pyshmem*.so` | Python 扩展（可选） | pybind11 绑定，供 `import shmem` | 构建：`find_package(pybind11)`；链接 `libshmem.so` 等；运行：`torch-npu`（`setup.py`） |

说明：

- Device 侧接口多为头文件内联；Host 侧符号由 `libshmem.so` 提供。
- 应用侧通常 `#include "shmem.h"`，链接 `-lshmem`，并保证 `libshmem_utils.so` 与实际使用的 bootstrap 插件可被动态链接器解析。

## 2. 对外公开接口头文件列表

统一入口：`shmem.h`（按 Host / Device / SIMT 条件包含下列头文件）。

### 2.1 总入口与公共定义

| 对外公开接口文件 | 对应 so 库文件 | 文件用途 | 依赖版本信息 |
|------------------|----------------|----------|--------------|
| `shmem.h` | Host 部分对应 `libshmem.so`（及 utils/bootstrap）；Device/SIMT 部分无 | 对外总头文件：按编译宏聚合 Host / Device / SIMT 公开 API | Device/SIMT 分支依赖编译宏（如 `__CCE_AICORE__`、`USE_SIMT`）；无独立外部版本号 |
| `host_device/shmem_common_types.h` | 无 | Host/Device 共用类型、枚举、`ACLSHMEM_DEVICE` 等宏 | 依赖 CANN 头文件 `acl/acl_rt.h` |
| `host_device/shmem_common_macros.h` | 无 | Host/Device 共用宏 | 无 |
| `host/shmem_host_def.h` | `libshmem.so` | Host 侧常量、版本宏、`ACLSHMEM_HOST_API`、公共结构/枚举 | 无外部版本约束（仅定义本库 API 宏） |
| `device/shmem_def.h` | 无 | Device 侧公共定义与非连续拷贝等结构 | 依赖 CANN；`USE_SIMT` 时有条件分支 |

### 2.2 Host 侧公开接口

| 对外公开接口文件 | 对应 so 库文件 | 文件用途 | 依赖版本信息 |
|------------------|----------------|----------|--------------|
| `host/init/shmem_host_init.h` | `libshmem.so` + 运行时加载 bootstrap 插件 | 初始化/终结、UniqueId、init 状态查询、`aclshmemx_init_attr` 等 | Default/UniqueID 加载 `aclshmem_bootstrap_config_store.so`；MPI 模式加载 `aclshmem_bootstrap_mpi.so`（需系统 MPI） |
| `host/mem/shmem_host_heap.h` | `libshmem.so` | 对称堆分配：`malloc`/`calloc`/`align`/`free` 及扩展内存类型接口 | 无 |
| `host/team/shmem_host_team.h` | `libshmem.so` | Team 创建/拆分/销毁、PE 翻译、`my_pe`/`n_pes` 等 | 无 |
| `host/data_plane/shmem_host_rma.h` | `libshmem.so` | Host 侧 RMA（put/get 等）与 `aclshmem_ptr` | 无 |
| `host/data_plane/shmem_host_so.h` | `libshmem.so` | Host 侧 put+signal 同步/异步接口 | 无 |
| `host/data_plane/shmem_host_p2p_sync.h` | `libshmem.so` | Host 侧 P2P 同步、`handle_wait`、signal wait、ffts 配置获取 | 无 |
| `host/data_plane/shmem_host_cc.h` | `libshmem.so` | Host 侧集合通信：barrier/sync 及 on_stream 变体 | 无 |
| `host/utils/shmem_log.h` | `libshmem.so` / `libshmem_utils.so` | 外部 logger、日志级别、profiling 打印/获取 | 无 |

### 2.3 Device 侧公开接口（GM↔GM / UB↔GM）

| 对外公开接口文件 | 对应 so 库文件 | 文件用途 | 依赖版本信息 |
|------------------|----------------|----------|--------------|
| `device/gm2gm/shmem_device_rma.h` | 无 | Device GM↔GM 标准 RMA（多 dtype put/get 等） | 依赖 CANN；底层可走 MTE/RDMA/SDMA/UDMA（由拓扑与编译选项决定）；远端操作数用于对称地址换算；未使能 RDMA 时本地 Put 源端或 Get 目的端可使用任意有效的本地 GM 地址，使能 RDMA 后两端均须指向对称内存，调用者无需判断单次实际引擎 |
| `device/gm2gm/shmem_device_so.h` | 无 | Device 侧 put+signal 同步接口 | 依赖 CANN |
| `device/gm2gm/shmem_device_amo.h` | 无 | Device 原子操作（atomic_add 等） | 依赖 CANN；头文件注明各引擎支持类型随 SoC（如 UDMA 面向 Ascend950） |
| `device/gm2gm/shmem_device_mo.h` | 无 | Device 内存序：`quiet` / `fence` | 依赖 CANN |
| `device/gm2gm/shmem_device_p2p_sync.h` | 无 | Device P2P 同步：signal_op、wait_until 等 | 依赖 CANN |
| `device/gm2gm/shmem_device_cc.h` | 无 | Device 集合通信：barrier（含 vector-only 等变体）、ffts 地址设置 | 依赖 CANN；头文件限制主要用于 MIX Kernel |
| `device/gm2gm/engine/shmem_device_mte.h` | 无 | MTE 引擎：对称地址翻译、连续/非连续异步 put/get | 依赖 CANN |
| `device/gm2gm/engine/shmem_device_rdma.h` | 无 | RDMA 引擎：本地和远端操作数均须指向对称内存，且完整传输范围不得越过各自内存分配的异步 RMA | 依赖 CANN；Host 侧需开启 `ACLSHMEM_RDMA_SUPPORT`（Ascend950 另需指定 RDMA backend） |
| `device/gm2gm/engine/shmem_device_sdma.h` | 无 | SDMA 引擎：参数设置与异步 put/get | 依赖 CANN |
| `device/gm2gm/engine/shmem_device_udma.h` | 无 | UDMA 引擎：低阶异步 RMA | 依赖 CANN；编译需 `ACLSHMEM_UDMA_SUPPORT` 且 Hcomm 提供所需 API（CMake `try_compile` 检测）；面向 Ascend950 |
| `device/ub2gm/shmem_device_rma.h` | 无 | UB↔GM 标准 RMA（UB 侧 get/put 等） | 依赖 CANN |
| `device/ub2gm/engine/shmem_device_mte.h` | 无 | UB↔GM 的 MTE 引擎异步拷贝 | 依赖 CANN |
| `device/team/shmem_device_team.h` | 无 | Device 侧 Team/PE 查询与 PE 翻译 | 依赖 CANN |

### 2.4 Device SIMT 公开接口（需开启 SIMT 支持）

| 对外公开接口文件 | 对应 so 库文件 | 文件用途 | 依赖版本信息 |
|------------------|----------------|----------|--------------|
| `device_simt/shmem_simt_common_types.h` | 无 | SIMT 公共状态/类型定义 | 需编译定义 `USE_SIMT`（`ACLSHMEM_SIMT_SUPPORT`） |
| `device_simt/gm2gm/shmem_device_simt_rma.h` | 无 | SIMT 作用域下的 GM↔GM RMA | 需 `USE_SIMT`；依赖 CANN |
| `device_simt/gm2gm/engine/shmem_device_simt_mte.h` | 无 | SIMT MTE：thread/block/warp 作用域异步拷贝 | 需 `USE_SIMT`；依赖 CANN |
| `device_simt/team/shmem_device_simt_team.h` | 无 | SIMT 侧 Team 相关接口 | 需 `USE_SIMT`；依赖 CANN |
