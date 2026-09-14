<!-- 中文 / Chinese -->

# <div align="center">SHMEM</div>

<h4><div align="center">基于对称内存的昇腾分布式内存通信加速库</div></h4>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-SHMEM-blue)](https://shmem-doc.pages.dev/)[![Release](https://img.shields.io/badge/Release-v1.3.0-brightgreen)](https://gitcode.com/cann/shmem/releases/v1.3.0)[![Platform](https://img.shields.io/badge/Platform-Ascend%20NPU-red)](https://www.hiascend.com/)[![SIG](https://img.shields.io/badge/SIG-shmem-lightgrey)](https://gitcode.com/cann/community/tree/master/CANN/sigs/shmem)

</div>

## 最新动态

🚀 [2026/04] [SHMEM v1.3.0](https://gitcode.com/cann/shmem/releases/v1.3.0) 版本发布，欢迎下载体验。

   - 新增 AICore 直驱能力： A2/A3 SDMA 访存和预取，950 MTE 访存，覆盖更多通信引擎。
   - 新增 40+ 接口，覆盖 RMA、Signal、P2P 同步、Barrier 等，完善通信编程语义。
   - 增强 log、sanitizer、profiling、debug 等 DFX 能力，提升问题定位效率。

🔥 [2025/12] SHMEM 项目首次上线。

## ⚡️快速入门

### 安装

SHMEM 提供三种安装方式，按需选择：

**方式一：pip 安装**

```bash
pip install cann-shmem -i https://ascend.devcloud.huaweicloud.com/cann/pypi/simple/
shmem-config --version   # 查询安装版本
shmem-config --diagnose  # 检查 native 加载和包完整性
```

> 注意：
> - 运行环境 glibc 版本需 ≥ 2.34，否则可能因缺少符号导致 `libshmem.so` 加载失败。可通过 `ldd --version` 查看本地 glibc 版本。

**方式二：二进制包安装**

```bash
# 获取软件包 SHMEM_{version}_linux-{arch}.run（见 release 页面或本地构建）
chmod +x 软件包名.run
./软件包名.run --install
source /usr/local/Ascend/shmem/latest/set_env.sh
```

**方式三：源码编译**

```bash
git clone https://gitcode.com/cann/shmem.git
cd shmem
bash scripts/build.sh             # A2/A3 平台
# bash scripts/build.sh -soc_type Ascend950  # 950 平台
source install/set_env.sh
```

> 完整安装步骤（含 CANN 环境准备、依赖说明、Docker 容器、编译执行和本地验证等）详见 [快速入门文档](docs/quickstart.md)。

安装 Python wheel 后，可通过 [shmem-config 命令参考](docs/tools/shmem_config_guide.md)查询后端和安装路径，执行环境检查及问题诊断。

若您需要了解文档和接口中出现的缩写与名词，请参考[术语表](docs/glossary.md)。

## 一、项目简介

SHMEM 是面向昇腾平台的多机多卡内存通信库，通过封装 Host 侧与 Device 侧接口，实现跨设备的高效内存访问与数据同步。其核心价值在于：

- 支持 AICore 直驱 MTE、xDMA 使能 D2D/D2H/H2D/D2rH/rH2D 通信
- 简化分布式场景下的卡间通信逻辑，降低算子开发门槛
- 与 CANN 生态深度适配，支持通算融合类算子的快速部署
- 更多详细资料请参考[SHMEM](https://shmem-doc.pages.dev/)

## 二、核心功能

![核心功能](docs/images/readme-features.png)

**1. 双侧接口体系**

- Host 侧：负责初始化、内存堆管理、通信域（Team）创建及全局同步
- Device 侧：提供远程内存访问（RMA）、设备级同步及通信域操作

接口设计贴合昇腾算子开发范式，支持 Host 与 Device 协同工作流。

**2. 高性能通信优化**

- 内置 MTE、xDMA 引擎支持，实现远程内存直接读写，减少数据传输延迟
- 兼容 MPI 通信框架，支持集合通信（Allgather/Allreduce 等）场景
- 针对昇腾硬件特性优化数据传输路径，提升多卡协同效率

**3. 安全通信机制**

- 默认启用 TLS 加密保护跨设备数据传输，支持接口级关闭控制：

   ```c
   int32_t ret = aclshmemx_set_conf_store_tls(false, NULL, 0);
   ```

- 提供安全加固指南，包括权限配置、加密套件选择等企业级安全策略

**4. 通信通路覆盖**

下图展示了 SHMEM 支持的全链路通信通路（以910A3为例），覆盖 Host 侧与 Device 侧的不同传输引擎：

<img src="docs/images/dma.png" width="800"/>

如图所示，SHMEM 支持多种通信引擎和通路：

- **MTE Engine**：芯片级内存传输引擎，支持 D2D、D2H、H2D、D2rH、rH2D 多种通路
- **xDMA Engine**：高速直接内存访问引擎，支撑主机内/主机间高效数据传输

**5. 多语言与扩展支持**

- 提供 C++ 原生接口与 Python 封装，满足不同开发场景需求
- 模块化设计支持通信后端（MTE/xDMA）动态切换，便于功能扩展

**6. 丰富场景样例**
覆盖基础通信到复杂算子融合场景：

- rdma_demo：RDMA 协议通信演示
- [CATCCOS](https://gitcode.com/cann/catccos)：通算融合算子库，包含矩阵乘法、ReduceScatter 等常用算子

## 三、代码结构

```text
shmem/                                  # 项目根目录
├── docs/                               # 文档与说明
├── examples/                           # 示例工程集合
├── include/                            # 对外头文件
│   ├── shmem.h                         # SHMEM 所有对外 API 汇总
│   ├── device/                         # Device 侧头文件
│   │   ├── gm2gm/                      # AICore 驱动 gm2gm 数据面接口
│   │   │   └── engine/                 # AICore 直驱 gm2gm 低阶接口
│   │   ├── team/                       # Device 侧通信域管理
│   │   └── ub2gm/                      # AICore 驱动 ub2gm 数据面接口
│   │       └── engine/                 # AICore 直驱 ub2gm 低阶接口
│   ├── host/                           # Host 侧头文件
│   │   ├── data_plane/                 # Host 侧数据面接口
│   │   ├── init/                       # Host 侧初始化接口
│   │   ├── mem/                        # Host 侧内存管理接口
│   │   ├── team/                       # Host 侧通信域管理接口
│   │   └── utils/                      # 工具与通用辅助代码
│   └── host_device/                    # 共用目录
├── scripts/                            # 示例脚本（编译/运行）
├── src/                                # 源码实现
│   ├── device/                         # Device 侧实现
│   │   ├── gm2gm/                      # AICore 直驱 gm2gm 数据面接口
│   │   │   └── engine/                 # AICore 直驱 gm2gm 低阶接口
│   │   ├── team/                       # Device 侧通信域管理
│   │   └── ub2gm/                      # AICore 驱动 ub2gm 数据面接口
│   │       └── mte/                    # AICore 直驱 ub2gm 低阶接口
│   ├── host/                           # Host 侧实现
│   │   ├── bootstrap/                  # bootstrap
│   │   ├── hybm/                       # Hybrid Memory 实现
│   │   ├── init/                       # 初始化
│   │   ├── mem/                        # 内存管理相关
│   │   ├── python_wrapper/             # Python 封装/绑定
│   │   ├── sync/                       # 同步原语（barrier/p2p/order）
│   │   ├── team/                       # team（通信域）相关
│   │   ├── transport/                  # 传输层实现（如 RDMA\SDMA\UDMA）
│   │   └── utils/                      # 工具与通用辅助代码
│   ├── host_device/                    # 共用目录
│   └── python/                         # Python 相关目录
└── tests/                              # 测试用例集合（UT/功能测试）
```

## 四、典型使用场景

**1. 通算融合类算子开发**：基于 Device 侧内存直接访问接口，开发融合「计算+通信」的自定义算子（如 matmul+allreduce），减少卡间数据拷贝，提升算子执行效率。

**2. 多机多卡数据同步**：通过 Host 侧通信域管理接口，快速搭建多机多卡集群的内存共享通道，实现跨节点数据同步，适配分布式训练场景。

**3. 低延迟卡间通信**：利用 RDMA 优化的 Device 侧接口，实现卡间毫秒级数据传输，满足实时性要求高的 AI 推理场景。

**4. Python 分布式训练适配**：通过 Python 扩展接口，将 SHMEM 集成到 PyTorch 分布式训练流程中，替代传统 MPI 通信，降低训练通信开销。

## 五、常见问题（FAQ）

**Q1：编译时报「CANN 环境未找到」？**

A：确认已安装 CANN toolkit，并已执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。项目编译默认使用 CANN toolkit 提供的 `bisheng` 编译器，CANN 版本需满足 [CANN 版本说明](docs/quickstart.md#43-cann)。

**Q2：运行示例时报「卡间通信超时」？**

A：检查 RDMA 网卡是否可用、节点间网络是否连通、防火墙是否放行初始化通信端口（默认 8666）、交换机无损网络配置是否正确，以及各节点时钟是否同步。RDMA 建链监听端口由系统自动分配，无需手工配置，RDMA 端口使用规则见 [Troubleshooting - RDMA 端口分配规则](docs/debug/Troubleshooting_FAQs.md#RDMA-端口管理规则)。

**Q3：Python 导入 shmem 时报「找不到模块」？**

A：确认已安装 wheel 包，且 `source` 了 install 目录下的 `set_env.sh`，环境变量 `PYTHONPATH` 包含 shmem 路径。

**Q4：关闭 TLS 后仍提示加密失败？**

A：需在 `aclshmemx_init_attr` 前调用 `aclshmemx_set_conf_store_tls`，初始化后无法修改 TLS 配置。

**Q5：googletest、nlohmann/json 这些依赖执行 `build.sh` 时提示 git 失败？**

A：确认 git 配置是否可以访问 GitCode。`googletest v1.14.x` 用于 UT 构建，`nlohmann/json v3.11.3` 用于 Ascend950 平台构建，默认由 `scripts/build.sh` 自动拉取；离线环境可提前准备到 `3rdparty/googletest` 和 `3rdparty/json`。详情见 [第三方源码依赖](docs/quickstart.md#47-第三方源码依赖).

**Q6：CANN 包安装失败怎么办？**

A：查看[常见问题](https://www.hiascend.com/document/detail/zh/AscendFAQ/CommuFunc/resdl/rdl_011.html)

**Q7：仓库根目录没有 Dockerfile，如何准备容器环境？**

A：本仓当前不维护独立 Dockerfile。建议根据芯片、系统架构和 CANN 版本从[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub)选择 CANN 容器镜像，再按 [Docker 容器环境](docs/quickstart.md#49-docker-容器环境) 和快速开始步骤完成源码构建。

**Q8：无 NPU 环境能否运行 UT 和 examples？**

A：可以做依赖检查和编译验证，但 `scripts/run.sh`、`scripts/run_examples.sh` 需要可用 NPU、驱动和 CANN 运行时，运行结果需在硬件环境或项目 CI 中验证。

> 更多故障排查见：[Troubleshooting](docs/debug/Troubleshooting_FAQs.md)

## 六、贡献

### 贡献者列表

- [华南理工大学 陆璐教授团队](https://www2.scut.edu.cn/cs/2017/0629/c22284a328108/page.htm)

### 参与贡献指南

欢迎订阅 [SHMEM SIG 会议](https://mailweb.cann.osinfra.cn/mailman3/lists/shmem.cann.osinfra.cn/)，参与社区例会和议题讨论，与社区成员共同交流方案设计、接口规划和使用问题。

**1. 提 Issue**

- 提交 bug：明确环境（硬件/软件版本）、复现步骤、错误日志；
- 提功能需求：说明场景、预期效果、适配的硬件/软件版本。

**2. 提 PR**

- 分支规范：功能开发用 `feature/xxx`，bug 修复用 `bugfix/xxx`；
- 代码规范：遵循项目代码规范，新增代码需补充单元测试；
- PR 描述：说明修改目的、核心逻辑、测试验证结果。

**3. 代码审核**

- PR 需通过 CI 自动测试（编译、单元测试、代码规范检查）；
- 至少 1 名维护者审核通过后，方可合并。

详细步骤可参考[贡献指南](CONTRIBUTING.md)

## 七、安全声明

- 通信安全：默认启用 TLS 加密，支持自定义加密套件
- 公网依赖：依赖的开源仓库与工具地址参见[公网地址清单](SECURITY.md#公网地址声明)
- 加固指南：参考[安全加固建议](SECURITY.md#安全加固)配置系统权限与防火墙

## 八、版权与许可

Copyright (c) 2025 Huawei Technologies Co., Ltd.
本项目基于 CANN Open Software License Agreement Version 2.0 授权，仅允许用于昇腾处理器相关开发。

## 九、注意事项

1. 本项目仅适配昇腾平台，不支持其他硬件架构（如 x86 通用服务器、NVIDIA GPU）；
2. 示例代码仅供学习参考，生产环境使用前需完成充分的功能和性能测试；
3. CANN 版本升级可能导致接口兼容问题，建议锁定文档指定的 CANN 版本；
4. 关闭 TLS 加密后，需确保通信网络为可信内网，避免数据泄露风险。

---

<!-- English -->
<div align="center">

# SHMEM

<h4>Symmetric Memory-based Ascend Distributed Memory Communication Acceleration Library</h4>

[![Documentation](https://img.shields.io/badge/Documentation-SHMEM-blue)](https://shmem-doc.pages.dev/)
[![Release](https://img.shields.io/badge/Release-v1.3.0-brightgreen)](https://gitcode.com/cann/shmem/releases/v1.3.0)
[![Platform](https://img.shields.io/badge/Platform-Ascend%20NPU-red)](https://www.hiascend.com/)
[![SIG](https://img.shields.io/badge/SIG-shmem-lightgrey)](https://gitcode.com/cann/community/tree/master/CANN/sigs/shmem)

</div>

## What's New
🚀 [April 2026] [SHMEM v1.3.0 release](https://gitcode.com/cann/shmem/releases/v1.3.0). Download and experience it now.
   - Introduced AI Core direct-driven capabilities: 910B/910C SDMA memory access and prefetch, 950 MTE memory access, and more communication engines.
   - Added 40+ new APIs, spanning RMA, signaling, P2P synchronization, and barrier operations, enriching communication semantics.
   - Enhanced DFX capabilities, such as logging, sanitizers, profiling, and debugging, for faster problem pinpointing.

🔥 [December 2025] Initial launch of the SHMEM project

## 1. Project Introduction
SHMEM is a multi-server, multi-device memory communication library designed for the Ascend platform. By abstracting host APIs and device APIs, SHMEM enables fast cross-device memory access and data synchronization. Its core benefits include:
- Support for AI Core direct-driven MTE/xDMA, enabling D2D, D2H, H2D, D2rH, and rH2D communication paths
- Simplified inter-device communication logic in distributed workloads, lowering operator development complexity
- Deep integration with the CANN ecosystem, accelerating deployment of MC2 operators
- For more information, see [SHMEM](https://shmem-doc.pages.dev/).


## 2. Core Functions
![Core Functions](docs/images/readme-features_en.png)

**Dual-side APIs**
- Host APIs: initialization, memory heap management, communicator (often called team) creation, and global synchronization
- Device APIs: remote memory access (RMA), device-level synchronization, and team operations

The API design follows the Ascend operator development rules and supports host-device collaboration.

**High-performance communication optimization**
- Built-in MTE and xDMA engines for direct remote memory read/write, minimizing latency
- MPI interoperability, supporting collective communication primitives such as AllGather and AllReduce
- Optimized data transfer paths for Ascend hardware features to improve the multi-device collaboration efficiency

**Secure communication mechanism**
- TLS encryption is enabled by default to protect cross-device data transfer and can be disabled for specific APIs:
   ```c
   int32_t ret = aclshmemx_set_conf_store_tls(false, NULL, 0);
   ```

- Enterprise-grade security guidelines are provided, covering permission configurations and cipher suite selection.

**Comprehensive communication channels**

The following figure shows the full-link communication channels supported by SHMEM (using Ascend 910A3 as an example), covering different transmission engines on the host and device.

<img src="docs/images/dma_en.png" width="800"/>

As shown in the figure, SHMEM supports diverse communication engines and channels:
- **MTE engine**: chip-level memory transfer, supporting D2D, D2H, H2D, D2rH, and rH2D
- **xDMA Engine**: high-speed direct memory access (DMA), supporting intra-host and inter-host fast data transfer

**Multi-language and extensibility**
- Native C++ APIs and Python bindings
- Modular backend design enabling dynamic switching between MTE and xDMA

**Extensive use cases**
Use cases from basic communication to complex operator fusion:
- rdma_demo: RDMA communication demonstration
- matmul_allreduce: implementation of MC2 operators (matrix multiplication + AllReduce)


## 3. Environment Setup

### 3.1 Hardware Requirements
- Atlas series: 800I A2/A3 and 800T A2/A3
- Architecture compatibility: AArch64 and x86

### 3.2 Software Dependencies
#### 3.2.1 CANN Version Description
| Driver/Firmware| CANN Version| D2D | D2H/H2D | D2rH/rH2D | Other Dependencies|
| --- | --- | --- | --- | --- | --- |
| Ascend HDK 25.0.RC1.1 | 9.0.0-beta.2 or later<br>[Community Edition Resources](https://www.hiascend.com/developer/download/community/result?module=cann)| MTE<br>RDMA<br>SDMA | MTE | MTE | To enable SDMA, download the community edition of [ops-legacy package](https://www.hiascend.com/developer/download/community/result?module=cann).|
| Ascend HDK 25.0.RC1.1 | 9.0.0 or later<br>Toolkit package for the trial version: [x86_64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260305000326487/x86_64/Ascend-cann-toolkit_9.0.0_linux-x86_64.run) / [aarch64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260305000326487/aarch64/Ascend-cann-toolkit_9.0.0_linux-aarch64.run)| MTE<br>RDMA<br>SDMA | MTE | MTE | To enable SDMA, download the ops-legacy package (based on the hardware platform): [A2 x86_64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-910b-ops-legacy_9.0.0_linux-x86_64.run) / [A2 aarch64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-910b-ops-legacy_9.0.0_linux-aarch64.run) / [A3 x86_64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-A3-ops-legacy_9.0.0_linux-x86_64.run) / [A3 aarch64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-A3-ops-legacy_9.0.0_linux-aarch64.run)|
| Ascend HDK 25.0.RC1.1 | 8.5.0 or later<br>[Community Edition Resources](https://www.hiascend.com/developer/download/community/result?module=cann)| MTE<br>RDMA | MTE | MTE | Enabling A3 D2rH/rH2D: LingQu Computing Network [1.5.0](https://support.huawei.com/enterprise/en/ascend-computing/lingqu-computing-network-pid-258003841/software)<br>Upgrade guide: [Installation Guide](https://support.huawei.com/enterprise/en/ascend-computing/lingqu-computing-network-pid-258003841)|
| Ascend HDK 25.0.RC1.1 | 8.3.RC1 or later<br>[Community Edition Resources](https://www.hiascend.com/developer/download/community/result?module=cann)| MTE<br>RDMA |  |  | |

#### 3.2.2 CANN Package Installation
See [CANN Quick Installation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=openEuler&Software=cannToolKit).

Configure CANN environment variables (using a default installation path):
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Configure CANN environment variables (using a custom installation path):
```bash
source ${install_path}/ascend-toolkit/set_env.sh
```

### 3.3 Other Software Dependencies
- Install the PyTorch framework and torch_npu plugin.
  - The package must be installed to build and run PyTorch operators with input and output tensors.
  - Select the version to be installed based on the actual environment. For details, see [Ascend Extension for PyTorch](https://www.hiascend.com/document/detail/en/Pytorch/720/configandinstg/instg/insg_0004.html).
- Toolchains:
  - CMake 3.19 or later
  - GLIBC 2.28 or later

### 3.4 Python Dependency Installation
```bash
# Hard dependencies (required for build + runtime)
python3 -m pip install -r requirements.txt

# Additional dependencies (optional) for running Python examples
python3 -m pip install -r requirements-examples.txt
```

### 3.5 Optional Dependencies
- MPI: Open MPI 4.0+ (distributed communication)
- Python: 3.7+ (used for Python APIs)
- PyTorch: 1.12+ (used for running Python samples)

## 4. Quick Start

### 4.1 Installation Methods
#### 4.1.1 Method 1: Source Code Build
```bash
# Clone the code repository.
git clone https://gitcode.com/cann/shmem.git
cd shmem

# Build the core library (excluding the xDMA capability, examples, and tests by default).
bash scripts/build.sh

# Configure environment variables.
source install/set_env.sh
```
Note: For details about the parameters of build.sh, see [compilation_build_guide_en.md](./docs/compilation_build_guide_en.md).

#### 4.1.2 Method 2: Binary Package Installation
How to obtain: `bash scripts/build.sh -package`

Software package format: `SHMEM_{version}_linux-{arch}.run`
```bash
# Go to the corresponding directory.
cd {project_root}/package/{arch}/
# Configure and verify permissions.
chmod +x SHMEM_{version}_linux-{arch}.run
./SHMEM_{version}_linux-{arch}.run --check

# Install the package (default path: /usr/local/Ascend/shmem).
./SHMEM_{version}_linux-{arch}.run --install

# Configure environment variables.
# (Default path: /usr/local/Ascend/shmem)
source /usr/local/Ascend/shmem/latest/set_env.sh
# (Custom path: ${install_path}/shmem)
source ${install_path}/shmem/latest/set_env.sh
```

### 4.2 Installation Verification
Use `matmul_allreduce` as an example to verify core functions.

1. Build in the `shmem/` source directory:

   ```sh
   bash scripts/build.sh -examples
   ```

2. Run the demo in the `shmem/examples/matmul_allreduce` directory:

   ```sh
   bash scripts/run.sh -ranks 2 -M 1024 -K 2048 -N 8192
   ```

Note: The examples and other sample code are for reference only. Exercise caution when using them in the production environment.

### 4.3 Debug Mode
Build in the `shmem/` source directory:

   ```sh
   bash scripts/build.sh -examples -debug
   ```

Note: The `-examples` parameter is optional. For details, see [Usage Reference](./docs/debug/Troubleshooting_FAQs_en.md#shmem-faqs).

### 4.4 Python APIs
Note: For details about the Python API list, see [Python API List](./docs/api/pythonAPI_en.md).

1. Build the Python extension in the root directory of the repository:

   ```sh
   bash scripts/build.sh -python_extension
   ```

2. Source the environment setup script in the installation directory to configure environment variables:

   ```sh
   source install/set_env.sh
   ```

3. Set whether to enable TLS authentication. By default, TLS authentication is enabled. To disable TLS authentication, use the following API:

   ```python
   import shmem as shm
   shm.set_conf_store_tls(False, "")   # Disable TLS authentication
   ```

   ```python
   import shmem as shm
   tls_info = "xxx"
   shm.set_conf_store_tls(True, tls_info) # Enable TLS authentication
   ```

4. Run the Python extension test demo.

   ```sh
   bash examples/python_extension/run.sh
   ```

If `test.py running success!` is printed in the log, the demo is running.


## 5. Code Structure
```
shmem/                                 # Project root directory
├── docs/                              # Documentation and description
├── examples/                          # A collection of examples
├── include/                           # External header files
│   ├── shmem.h                        # All SHMEM external APIs
│   ├── device/                        # Device-side header file
│   │   ├── gm2gm/                     # Data plane API gm2gm driven by AI Core
│   │   │   └── engine/                # Low-level API of gm2mm, directly driven by AI Core
│   │   ├── team/                       # Device-side team management
│   │   └── ub2gm/                     # Data plane API ub2gm driven by AI Core
│   │       └── engine/                 # Low-level API of ub2gm, directly driven by AI Core
│   ├── host/                          # Host-side header file
│   │   ├── data_plane/                # Host-side data plane API
│   │   ├── init/                      # Host-side initialization API
│   │   ├── mem/                       # Host-side memory management API
│   │   ├── team/                      # Host-side team management API
│   │   └── utils/                     # Tools and general auxiliary code
│   └── host_device/                   # Shared directory
├── scripts/                           # Sample scripts (build/run)
├── src/                               # Source code implementation
│   ├── device/                        # Device-side implementation
│   │   ├── gm2gm/                     # Data plane API gm2gm directly driven by AI Core
│   │   │   └── engine/                # Low-level API of gm2mm, directly driven by AI Core
│   │   ├── team/                       # Device-side team management
│   │   └── ub2gm/                     # Data plane API ub2gm driven by AI Core
│   │       └── mte/                    # Low-level API of ub2gm, directly driven by AI Core
│   ├── host/                          # Host-side implementation
│   │   ├── bootstrap/                  # bootstrap
│   │   ├── hybm/                      # Hybrid Memory implementation
│   │   ├── init/                      # Initialization
│   │   ├── mem/                       # Memory management
│   │   ├── python_wrapper/            # Python encapsulation/bindings
│   │   ├── sync/                      # Synchronization primitives (barrier/p2p/order)
│   │   ├── team/                       # Team (communicator)
│   │   ├── transport/                  # Transport layer implementation (such as RDMA, SDMA, and UDMA)
│   │   └── utils/                     # Tools and general auxiliary code
│   ├── host_device/                   # Shared directory
│   └── python/                         # Python-related directory
└── tests/                              # Test case set (UTs/functional tests)
```

## 6. Typical Use Cases
**MC2 operator development**: Develop custom operators (Matmul + AllReduce) that merge compute and communication by leveraging device-side direct memory access APIs. This reduces inter-device data copies and improves operator execution efficiency.

**Multi-server, multi-device data synchronization**: Use host-side team management APIs to quickly establish shared memory channels for a multi-server, multi-device cluster, enabling efficient data synchronization across servers for distributed training workloads.

**Low-latency inter-device communication**: Employ RDMA-optimized device-side APIs to transfer data among devices in milliseconds, meeting the real-time requirements of latency-sensitive AI inference scenarios.

**Python distributed training adaptation**: Integrate SHMEM into PyTorch distributed workflows via Python extension APIs, replacing traditional MPI communication to reduce training communication overhead.

## 7. Test Framework
- **Unit tests**: designed for core APIs (such as initialization, memory operations, and synchronization), located in `tests/unittest/`.

- **Operator generalization tests**: dynamic generation of test data and accuracy checks for examples such as `matmul_allreduce`.

### 7.1 Running Unit Tests

```bash
# Build and run unit tests.
bash scripts/build.sh -uttests
bash scripts/run.sh
```
The run.sh script provides parameters, such as `-ranks` and `-test_filter`, to customize the number of devices used and to apply `gtest_filter` for executing tests. Example:
```bash
# Run all *Init* test cases on 8 devices.
bash scripts/run.sh -ranks 8 -test_filter Init
```
For details about the parameters, see [Related Scripts-run.sh](docs/compilation_build_guide_en.md#Key-SHMEM-Files).

### 7.2 Building and Running Examples

```bash
bash scripts/build.sh -examples
bash scripts/run_examples.sh
```

### 7.3 Running an Example Independently
You can view the README file in the corresponding example directory under the `examples` directory.

### 7.4 Running Python Test Cases
```bash
# Build Python extensions.
bash scripts/build.sh -python_extension
# Install a wheel package generated by the build script.
pip3 install dist/shmem-xxx.whl --force-reinstall
```

You can also manually build and install the wheel package in the root directory of the repository.

```bash
python3 setup.py bdist_wheel
pip3 install dist/shmem-xxx.whl --force-reinstall
```

```bash
# Run the Python test on two devices.
torchrun --nproc-per-node=2 examples/python_extension/test/init_test.py
```

### 7.5 Custom Tests
Based on the GTest framework in the `tests/` directory, new test cases must comply with the following rules:
- Test file naming: `{module}_test.cc`
- Test case naming: `{FunctionName}_{Scenario}_Test`

## 8. Configuration and Tuning (Optional, Advanced Usage)
**Disabling TLS encryption (to improve communication performance)**

TLS encryption is enabled by default. In a trusted intranet environment, it can be disabled to accelerate communication:
```c
int32_t ret = aclshmemx_set_conf_store_tls(false, NULL, 0);
```
```python
import shmem as shm
shm.set_conf_store_tls(False, "")
```

**Adjusting the shared memory size**

Specify the shared memory pool size during initialization (default: 16 GB) to support large-memory scenarios:
```c
aclshmemx_init_attr_t attr;
attr.local_mem_size = 32 * 1024 * 1024 * 1024; // 32GB
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
```

**Performance tuning suggestions**
- Prefer device APIs to minimize host-device interactions.
- Group teams by physical node to reduce cross-node communication.
- For large-batch data transfers, enable the RDMA protocol (by adding `-DSHMEM_RDMA=ON` during the build).

## 9. FAQs
**Question 1: What do I do If a message is displayed indicating that the CANN environment is not found during a build?**

Answer: Ensure that `source /usr/local/Ascend/ascend-toolkit/set_env.sh` has been executed and the CANN version meets the requirements in [Environment Dependencies](#32-software-dependencies).

**Question 2: What do I do if a message is displayed indicating that inter-device communication timed out during example running?**

Answer: Check whether RDMA is enabled on the NIC, whether the firewall allows the communication port (8666 by default), and whether the clocks of all nodes are synchronized.

**Question 3: What do I do if a message is displayed indicating that the module cannot be found when I import shmem to Python?**

Answer: Ensure that the wheel package has been installed, the `set_env.sh` file in the `install` directory has been sourced, and the environment variable `PYTHONPATH` contains the `shmem` path.

****Question 4: What do I do if an encryption failure message is still displayed after TLS is disabled?****

Answer: Call the `aclshmemx_set_conf_store_tls` function before `aclshmemx_init_attr`. After the initialization, the TLS configuration cannot be modified.

**Question 5: What do I do if a Git failure message is displayed when `build.sh` is executed using the GoogleTest and Catlass plugins?**

Answer: Check whether your Git configuration can access external websites. If the environment cannot connect to the websites, manually the required packages and place them under the `3rdparty` directory.

**Question 6: What can I do if the CANN package fails to be installed?**

Answer: See [FAQs](https://www.hiascend.com/document/detail/en/AscendFAQ/CommuFunc/resdl/rdl_011.html).

> For more troubleshooting information, see [Troubleshooting](docs/debug/Troubleshooting_FAQs_en.md).

## 10. Contributions
### Contributors
- [Professor Lu Lu, South China University of Technology](https://www2.scut.edu.cn/cs/2017/0629/c22284a328108/page.htm)

### Contribution
Subscribe to [SHMEM SIG Meetings](https://mailweb.cann.osinfra.cn/mailman3/lists/shmem.cann.osinfra.cn/) and join regular community meetings and discussions. Share your ideas on solution design, API planning, and usage with other members.

**1. Submitting issues**

- Submit bug reports to specify the environment (hardware/software versions), reproduction steps, and error logs.
- Submit feature requests by describing use cases, expected outcome, and supported hardware/software versions.

**2. Submitting a PR**

- Branch naming convention: Use `feature/xxx` for new features and `bugfix/xxx` for bug fixes.
- Coding standards: Follow project coding guidelines. New code must include unit tests.
- PR description: Explain the purpose of the changes, core logic, and test validation results.

**3. Reviewing code**

- PRs must pass CI automated checks (builds, unit tests, code style validation).
- At least one maintainer's approval is required before merging.

For details, see [Contribution Guide](CONTRIBUTING_en.md).

## 11. Security Statement
- Communication security: TLS encryption is enabled by default, and custom cipher suites are supported.
- Public network dependencies: For details about required open-source repositories and tools, see [Public Network Address List](SECURITY_en.md#public-network-address-statement).
- Security hardening guide: Configure system permissions and firewalls by referring to [Security Hardening Suggestions](SECURITY_en.md#security-hardening).

## 12. Copyright and License
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This project, licensed under CANN Open Software License Agreement Version 2.0, is exclusively for Ascend processor development.

## 13. Precautions
1. This project works only on the Ascend platform and does not support other hardware like x86 servers or NVIDIA GPUs.
2. The example code is for learning and reference. Test its functionality and performance before using it in a production environment.
3. Upgrading the CANN version can cause API compatibility issues. Use the CANN versions specified in this document.
4. After TLS encryption is disabled, ensure that the communication network is a trusted intranet to prevent data leakage.
