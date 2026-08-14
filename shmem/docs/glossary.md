# SHMEM 术语表

本文面向使用 SHMEM 的开发者，整理 README、docs、include 和 examples 中常见的缩写与名词。

说明：

- 本项目中的 `SHMEM` 主要围绕“对称内存”通信模型：各 PE 按相同规则分配可远程访问的对称数据对象或对称堆内存。文档中出现“共享内存”时，通常指 SHMEM 管理的可跨 PE 访问内存区域或基于该区域实现的通信能力。

- 除 `SHMEM` 项目特有术语，其他华为/昇腾/CANN 相关术语说明以[昇腾](https://www.hiascend.com/document/detail/zh/Glossary/gls/gls_0001.html)、[CANN](https://www.hiascend.com/cann)、[灵衢 UB](https://www.unifiedbus.com/zh) 官网为准。

## 编程模型与接口

| 术语 | 英文展开 | 中文说明 | 项目语境 |
| --- | --- | --- | --- |
| `SHMEM` | Symmetric Hierarchical Memory / Shared Memory | 对称内存/共享内存通信库 | 本仓项目名，表示面向昇腾平台的多机多卡内存通信库。 |
| `OpenSHMEM` | OpenSHMEM Application Programming Interface | OpenSHMEM 应用编程接口规范 | 标准化 SHMEM 编程模型的 API 规范，提供 PE、对称数据对象、RMA、AMO、Signal、集合通信等概念。 |
| `Host API` | Host-side API | Host 侧接口 | CPU/Host 侧调用的接口，负责初始化、内存管理、通信域管理、同步和任务下发。 |
| `Device API` | Device-side API | Device 侧接口 | AICore/kernel 上下文调用的接口，负责远程访存、同步和通信域操作。 |
| `PE` | Processing Element | 处理单元 | SHMEM 通信参与者编号，通常对应一个进程、rank 或设备参与方。 |
| `Team` | Team | 通信域/分组 | 一组 PE 组成的通信域，用于限定同步和通信操作范围。`ACLSHMEM_TEAM_WORLD` 是初始化后的默认全局通信域。 |
| `Symmetric Memory` | Symmetric Memory | 对称内存 | 各 PE 按相同规则分配、可通过 SHMEM 接口远程访问的内存区域。 |
| `Symmetric Heap` | Symmetric Heap | 对称堆 | `aclshmem_malloc` / `aclshmem_free` 管理的对称内存堆。 |

## 通信语义与同步

| 术语 | 英文展开 | 中文说明 | 项目语境 |
| --- | --- | --- | --- |
| `RMA` | Remote Memory Access | 远程内存访问 | 一侧 PE 主动访问另一侧 PE 的对称内存，包含 put/get 等单边访问操作。 |
| `AMO` | Atomic Memory Operation | 原子内存操作 | 对远端或本端对称内存执行 atomic add、fetch、swap、compare-swap 等原子语义操作。 |
| `SO` | Signal Operation | 信号变量/信号操作 | SHMEM 信号变量或信号操作，用于 put-with-signal、signal wait/op 等同步场景。 |
| `P2P Sync` | Point-to-Point Synchronization | 点对点同步 | 点对点等待、测试和同步接口语义。 |
| `CC` | Collective Communication | 集合通信 | 多个 PE 共同参与的通信操作，例如 Barrier、Allgather、Allreduce。 |
| `MO` | Memory Ordering | 内存保序 | 同步机制，用于保障跨处理单元的内存读写及阻塞/非阻塞接口操作的执行顺序，并确保操作完成状态可靠送达。 |
| `put` | Put operation | 写远端操作 | 本地 PE 主动将数据写入远端 PE 的对称地址。 |
| `get` | Get operation | 读远端操作 | 本地 PE 主动从远端 PE 的对称地址读取数据。 |
| `NBI` | Non-Blocking Immediate | 非阻塞立即操作 | `*_nbi` API 后缀，表示非阻塞数据操作：调用后立即返回，不等待操作完成；必须配合 `quiet()` 或 `fence()` 等内存序屏障，才能确保操作完成状态的可靠送达与全局顺序。 |
| `Blocking` | Blocking | 阻塞操作 | 表示阻塞数据操作：调用会阻塞当前线程，返回时即保证操作完成状态的可靠送达。 |
| `Barrier` | Barrier Synchronization | 屏障同步 | Team 内集合同步。参与 PE 到达同步点后才继续执行，并按接口语义保证此前相关操作完成或可见。 |
| `Sync` | Synchronization | 同步 | 与 Barrier 类似，但 `Sync` 只保证此前 memory store 的完成与可见性，不保证 SHMEM 远程更新全部完成。 |
| `Quiet` | Quiet | 完成等待 | 等待调用 PE 先前发出的对称数据对象操作完成。 |
| `Fence` | Fence | 顺序栅栏 | OpenSHMEM 语义中用于约束 Put、AMO 和 store 的投递顺序；本项目当前实现与 `quiet` 行为接近，同时保证顺序和完成。 |

## 通信引擎、协议与数据通路

| 术语 | 英文展开 | 中文说明 | 项目语境 |
| --- | --- | --- | --- |
| `MTE` | Memory Transfer Engine | 内存传输引擎 | 昇腾 AICore 侧内存搬运引擎，用于 D2D、D2H、H2D、D2rH、rH2D 等通路。 |
| `DMA` | Direct Memory Access | 直接内存访问 | 不经由通用计算单元逐字节搬运的内存访问机制。 |
| `xDMA` | Direct Memory Access family | DMA 类引擎统称 | 对高速 DMA 类传输引擎的统称。 |
| `SDMA` | System Direct Memory Access | 系统直接内存访问 | 数据传输引擎，作为 RMA/AMO 的底层传输后端之一。 |
| `UDMA` | Unified Direct Memory Access | 统一直接内存访问 | 数据传输引擎，作为 RMA/AMO 的底层传输后端之一。 |
| `RDMA` | Remote Direct Memory Access | 远程直接内存访问 | 数据传输引擎，允许通过网络或设备互联直接访问远端内存的机制，作为 RMA/AMO 的底层传输后端之一。 |
| `HCCS` | Huawei Cache Coherent System | 华为缓存一致性系统 | 昇腾服务器内 CPU/NPU 间高速互联总线。 |
| `UB` 协议 | Unified Bus | 灵衢总线 | 华为新一代互联协议，旨在统一数据中心内部所有互联，提供低时延、高带宽的通信。 |
| `UBMEM` / `UBmem` | UBMEM | 内存语义 | 华为新一代互联协议，提供内存语义通信。 |
| `URMA` | Unified Remote Memory Access | 统一远程内存访问 | 华为新一代互联协议，基于 UB 总线，提供类似本地访问语义的远程内存操作能力。 |
| `PCIe` | Peripheral Component Interconnect Express | 高速外设互联总线 | CPU 与 NPU 等其他外设的连接标准，也用于部分拓扑中的跨组互联。 |
| `SIO` | **S**erial **I**/**O** Link | SIO 链路/通路 | 昇腾 A3 处理器片内，两个物理 Die （芯片裸片）之间的高速互联通路。 |
| `RoCE` / `ROCE` | RDMA over Converged Ethernet | 基于融合以太网的 RDMA | RDMA 引擎使用的网络协议，常用于跨节点通信。 |
| `D2D` | Device to Device | 设备到设备 | 本端 NPU Device 内存与对端 NPU Device 内存之间的数据访问或搬运。 |
| `D2H` | Device to Host | 设备到主机 | 本端 NPU Device 内存到本端 Host 内存的数据通路。 |
| `H2D` | Host to Device | 主机到设备 | 本端 Host 内存到本端 NPU Device 内存的数据通路。 |
| `D2rH` | Device to remote Host | 设备到远端主机 | 本端 NPU Device 内存到远端 PE 所在 Host 内存的数据通路。 |
| `rH2D` | remote Host to Device | 远端主机到设备 | 远端 PE 所在 Host 内存到本端 NPU Device 内存的数据通路。 |
| `gm2gm` / `GM2GM` | Global Memory to Global Memory | 全局内存到全局内存 | Device 侧 AICore 驱动的 GM 到 GM 数据面接口/通路，常用于跨 PE 对称内存访问。 |
| `ub2gm` / `UB2GM` | Unified Buffer to Global Memory | 统一缓冲区到全局内存 | AICore 本地 UB 到远端设备 GM 的数据面接口/通路。 |
| `CMO` | Cache Maintenance Operation | 缓存维护操作 | SDMA 侧公开的缓存维护接口，用于显式控制缓存预取/维护操作，如 GM2L2。 |

## 内存、硬件与执行单元

| 术语 | 英文展开 | 中文说明 | 项目语境 |
| --- | --- | --- | --- |
| `Host` | Host | 主机侧 | CPU 侧控制面，负责初始化、内存管理、bootstrap、team/sync 管理和 host API 下发。 |
| `Device` | Device | 设备侧 | NPU/AICore 数据面，执行 RMA、同步和通信 kernel。 |
| `NPU` | Neural Processing Unit | 神经网络处理器 | 昇腾 AI 处理器设备。 |
| `AICore` / `AI Core` | AICore | 昇腾 AICore | Device 侧执行自定义算子和通信数据面代码的核心。 |
| `AIV` | AIV / Vector Core | 昇腾向量核 | 分离架构下 Vector Core 的官方名称，承担向量计算与低时延通信。 |
| `AIC` | AIC / Cube Core | 昇腾立方核 | 分离架构下 Cube Core 的官方名称，负责矩阵运算。 |
| `VEC` | Vector Core | 向量核 | 与 VECTOR_CORE 同义，指代向量计算单元。 |
| `CUBE` | Cube Core | Cube 核 | 与 Cube Unit 同义，专门执行高算力的矩阵乘加运算。 |
| `GM` | Global Memory | 全局内存 | Device 侧全局可寻址内存，接口中常以 `__gm__` 标识。 |
| `UB` | Unified Buffer | 统一缓冲区 | AICore 片上临时缓冲区，接口中常以 `__ubuf__` 或 `LocalTensor` 标识。 |
| `HBM` | High Bandwidth Memory | 高带宽内存 | NPU 侧高带宽设备内存。 |
| `DRAM` | Dynamic Random Access Memory | 动态随机访问内存 | Host 侧内存。 |
| `L2` | Level-2 Cache | 二级缓存 | CMO、缓存预取和示例中出现的缓存层级。 |

## 平台、环境变量与工具

| 术语 | 英文展开 | 中文说明 | 项目语境 |
| --- | --- | --- | --- |
| `CANN` | Compute Architecture for Neural Networks | 昇腾异构计算架构软件栈 | SHMEM 依赖 CANN 编译、运行和设备 API。 |
| `ACL` / `AscendCL` | Ascend Computing Language | 昇腾计算语言/运行时 API | 初始化、内存、stream、Graph 等 Host 侧运行时 API 依赖。 |
| `MPI` | Message Passing Interface | 消息传递接口 | 可作为 bootstrap 控制通道后端之一，用于 rank 间 allgather/barrier。 |
| `TLS` | Transport Layer Security | 传输层安全协议 | SHMEM 默认启用的通信加密机制。 |
| `SSL` | Secure Sockets Layer | 安全套接字层 | 安全连接相关依赖和测试中常与 TLS 相关。 |
| `UID` | Unique ID | 唯一标识 | 初始化/bootstrap 通信标识和相关环境变量命名。 |
| `IP` / `IPv4` / `IPv6` | Internet Protocol / Internet Protocol version 4 / Internet Protocol version 6 | 网际协议/IPv4/IPv6 | bootstrap 地址、网口协议和网络配置中使用。 |
| `HCCL` | Huawei Collective Communication Library | 华为集合通信库 | RDMA 环境变量 `HCCL_RDMA_TC`、`HCCL_RDMA_SL` 复用 HCCL 配置。 |
| `TC` | Traffic Class | 流量类别 | RDMA 网络 QoS 参数，对应 `HCCL_RDMA_TC`。 |
| `SL` | Service Level | 服务等级 | RDMA 网络服务等级参数，对应 `HCCL_RDMA_SL`。 |
| `PFC` | Priority Flow Control | 优先级流控 | 与 RDMA 场景 SL/网络配置相关。 |
| `SOC_TYPE` | System-on-Chip Type | 芯片型号类型 | 构建脚本和工具文档中的硬件型号选择参数。 |
| `ASCEND_HOME_PATH` | Ascend home path | Ascend 安装路径 | CANN/Ascend 安装路径环境变量。 |
| `ASCEND_TOOLKIT_HOME` | Ascend Toolkit home | Ascend Toolkit 安装路径 | Toolkit 安装路径环境变量。 |
| `ASCEND_RT_VISIBLE_DEVICES` | Ascend Runtime Visible Devices | 昇腾运行时可见设备 | 控制当前进程可见 Device 的环境变量。 |
| `LD_LIBRARY_PATH` | Library load path | 动态库搜索路径 | 运行示例和脚本时配置 SHMEM/CANN 动态库搜索路径。 |
| `SHMEM_UID_SESSION_ID` | SHMEM Unique ID Session ID | SHMEM UID 会话标识 | 指定 PE 0 bootstrap 监听 IP 和端口。 |
| `SHMEM_UID_SOCK_IFNAME` | SHMEM Unique ID Socket Interface Name | SHMEM UID 网口名 | 指定 PE 0 bootstrap 使用的网口名和网络层协议。 |
| `SHMEM_INSTANCE_PORT_RANGE` | SHMEM Instance Port Range | 多实例端口范围 | 多实例场景中为各实例 bootstrap 指定可用端口范围。 |
| `SHMEM_CYCLE_PROF_PE` | SHMEM Cycle Profiling PE | SHMEM cycle 采集 PE | 指定 profiling 采集的 PE。 |
| `SHMEM_LOG_LEVEL` | SHMEM Log Level | SHMEM 日志级别 | 控制 SHMEM 日志输出级别。 |
| `ENABLE_ASCENDC_DUMP` | Enable AscendC Dump | 启用 AscendC Dump | 编译期开关，用于启用 AscendC dump 调试逻辑。 |

## 示例、调测与性能术语

| 术语 | 英文展开 | 中文说明 | 项目语境 |
| --- | --- | --- | --- |
| [`SIMT`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10064.html) | Single Instruction Multiple Threads | 单指令多线程 | 示例中的指令/执行模式。 |
| [`SIMD`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0015.html) | Single Instruction Multiple Data | 单指令多数据 | 示例中的指令/执行模式。 |
| [`VF_TYPE`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10052.html) | Vector Function Type | 向量功能类型参数 | SIMT 与 SIMD 混合编程模式，整个执行过程以 VF 为基本调度单位。 |
| `DFX` | Design for X | 可诊断性/可维护性等工程能力 | log、sanitizer、profiling、debug 等能力的统称。 |
| [`mssanitizer`](https://gitcode.com/Ascend/mssanitizer/blob/master/docs/zh/install_guide/mssanitizer_install_guide.md) | MindStudio Sanitizer | MindStudio 内存检测工具 | 内存检测工具。 |
| [`MSTX`](https://gitcode.com/Ascend/mstx/blob/master/docs/zh/install_guide/mstx_install_guide.md) | MindStudio Tools Extension | MindStudio 工具扩展（性能打点 API） | `mstx` 工具安装和 mssanitizer 相关流程中出现；CANN 性能分析 API，支持自定义打点定位性能瓶颈。 |
| `PROF` / `Profiling` | Profiling | 性能采集/打点 | cycle profiling、perftest 等性能采集相关术语。 |
| `DUMP` | Dump | 数据/运行信息转储 | AscendC dump 调试能力。 |
| `PERF` / `perftest` | Performance Test | 性能测试 | 性能测试示例、脚本和输出指标。 |
| `throughput` | Throughput | 吞吐 | 性能测试输出指标。 |
| `bandwidth` | Bandwidth | 带宽 | 性能测试输出指标。 |
| `latency` | Latency | 时延 | 性能测试输出指标。 |
| `KB` / `MB` / `GB` | Kilobyte / Megabyte / Gigabyte | 千字节/兆字节/吉字节 | 文档、示例参数和性能输出中的容量单位。 |
| `低阶接口` | Low-level API (engine-specific API) | 直接暴露底层传输引擎（UDMA、RDMA、SDMA、MTE 等）通信语义的接口 | SHMEM 中位于 `engine/` 目录下的引擎专用接口（如 `aclshmemx_udma_*`、`aclshmemx_roce_*`），需调用方自行管理 buffer 配置与同步，支持并发操作。代码中常以 lowlevel 指代。 |
| `高阶接口` | High-level API | 屏蔽底层引擎细节、提供统一编程模型的高层抽象接口 | SHMEM 中位于 `gm2gm/` 或 `host/data_plane/` 下的封装接口（如 `aclshmem_put_*`、`aclshmem_amo_*`），内部自动管理 buffer 与同步，使用默认 buffer 不支持并发；UDMA/RDMA/SDMA 高阶接口需通过 `aclshmemx_*_config` 预配置引擎参数。代码中常以 highlevel 指代。 |
