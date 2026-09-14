<!-- 中文 / Chinese -->
# 样例介绍

## 环境要求

同[rdma_demo](../../rdma_demo/README.md)中的环境要求。

## 使用方式

1. 在shmem/目录编译。RDMA 编译参数（A2/A3，以及 Ascend950 的 `XSCALE` / `HNS_1825` 后端）详见 [编译与构建 - RDMA 参数使用说明](../../../docs/compilation_build_guide.md#rdma参数使用说明)。

2. 直接在`examples/rdma_handlewait_test/unuse_handlewait`目录下执行`bash run.sh`；或者在shmem/目录运行:
    > 注：Ascend950 平台需设置 `IBV_EXTEND_DRIVERS` 环境变量，参见[环境变量说明](../../rdma_demo/README.md#ibv_extend_drivers-环境变量)。
    - 单机2卡执行命令

    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export IBV_EXTEND_DRIVERS=<path_to_plugin.so>  # 仅Ascend950平台需要根据网卡类型进行环境变量设置，详见环境变量说明
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    ./build/bin/unuse_handlewait 2 0 tcp://127.0.0.1:8765 2 0 0 & # PE 0
    ./build/bin/unuse_handlewait 2 1 tcp://127.0.0.1:8765 2 0 0 & # PE 1
    ```

    > 注：\<shmem-root-directory\>为SHMEM项目的根目录，\<path_to_plugin.so\>为根据网卡类型设置的插件库路径。
    - 跨机2卡执行命令

    假设机器A的ip为ip1，机器B的ip为ip2。
    在机器A执行如下命令：

    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export IBV_EXTEND_DRIVERS=<path_to_plugin.so>  # 仅Ascend950平台需要根据网卡类型进行环境变量设置，详见环境变量说明
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    ./build/bin/unuse_handlewait 2 0 tcp://ip1:8765 1 0 0 # PE 0
    ```

    同时，在机器B执行如下命令：

    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export IBV_EXTEND_DRIVERS=<path_to_plugin.so>  # 仅Ascend950平台需要根据网卡类型进行环境变量设置，详见环境变量说明
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    ./build/bin/unuse_handlewait 2 1 tcp://ip1:8765 1 1 0 # PE 1
    ```

    > 注：\<shmem-root-directory\>为SHMEM项目的根目录，\<path_to_plugin.so\>为根据网卡类型设置的插件库路径。

3. 命令行参数说明

    ```bash
    ./unuse_handlewait <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>
    ```

    - n_pes: 全局PE数量。
    - pe_id: 当前进程的PE号。
    - ipport: SHMEM初始化需要的IP及端口号，格式为`tcp://<IP>:<端口号>`。如果执行跨机测试，需要将IP设为PE0所在Host的IP。
    - g_npus: 当前机器上启动的NPU卡的数量。
    - f_pe: 当前机器上使用的第一个PE号。
    - f_npu: 当前机器执行本样例使用的第一张NPU卡的卡号。

---

<!-- English -->
## Environment Requirements
See the environment requirements in [rdma_demo](../../rdma_demo/README_en.md).

## Instructions
1. Build in the `shmem/` directory.
```bash
bash scripts/build.sh -enable_rdma -examples
```
2. Run the `bash run.sh` command in the `examples/rdma_handlewait_test/unuse_handlewait` directory or run the following commands in the `shmem/` directory:
- For single-server dual-device, run the following commands:
    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    ./build/bin/unuse_handlewait 2 0 tcp://127.0.0.1:8765 2 0 0 & # PE 0
    ./build/bin/unuse_handlewait 2 1 tcp://127.0.0.1:8765 2 0 0 & # PE 1
    ```
    > Note: \<shmem-root-directory\> is the root directory of the SHMEM project.
- For cross-server dual-device, run the following commands:

    Assume that the IP address of server A is `ip1` and that of server B is `ip2`.
    Run the following commands on server A:
    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    ./build/bin/unuse_handlewait 2 0 tcp://ip1:8765 1 0 0 # PE 0
    ```
    At the same time, run the following commands on server B:
    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    ./build/bin/unuse_handlewait 2 1 tcp://ip1:8765 1 1 0 # PE 1
    ```
    > Note: \<shmem-root-directory\> is the root directory of the SHMEM project.

3. Parameters in the command line
```bash
    ./unuse_handlewait <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>
```
- n_pes: number of global PEs.
- pe_id: PE ID of the current process.
- ipport: IP address and port number required for SHMEM initialization. The format is `tcp://<IP address>:<port number>`. To perform a cross-server test, set the IP address to the IP address of the host where PE0 is located.
- g_npus: number of NPUs started on the current server.
- f_pe: ID of the first PE used on the current server.
- f_npu: ID of the first NPU used to run this sample on the current server.
