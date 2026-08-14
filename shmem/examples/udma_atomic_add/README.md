# 样例介绍

## 版本和平台支持说明

- UDMA 原子加样例仅支持 Ascend950 平台（`__NPU_ARCH__ == 3510`），非 Ascend950 平台不支持运行该样例。
- CANN 需提供 Ascend950 UDMA/HCOMM 资源接口，并安装与 Ascend950 匹配的 toolkit 和 ops 包，建议使用 CANN 9.1.0 及以上版本。低版本 CANN 如果缺少 `HcommEndpointCreate`、`HcommMemReg`、`HcommChannelCreate` 等 HCOMM UDMA API，构建阶段会关闭 `ACLSHMEM_UDMA_SUPPORT`，该样例无法运行。
- 编译前需完成 CANN 环境配置，例如执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，并使用 `-soc_type Ascend950` 编译。

使用方式:

1. 在shmem/目录编译:

    ```bash
    bash scripts/build.sh -examples -soc_type Ascend950
    ```

2. 在shmem/目录运行:

    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    export SHMEM_UID_SESSION_ID=127.0.0.1:8899
    ./build/bin/udma_atomic_add 2 0 tcp://127.0.0.1:8899 2 0 0 & # PE 0
    ./build/bin/udma_atomic_add 2 1 tcp://127.0.0.1:8899 2 0 0 & # PE 1
    ```

3. 命令行参数说明

    ```bash
    ./udma_atomic_add <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>
    ```

    - n_pes: 全局PE数量。
    - pe_id: 当前进程的PE号。
    - ipport: SHMEM初始化需要的IP及端口号，格式为`tcp://<IP>:<端口号>`。
    - g_npus: 当前机器上启动的NPU卡的数量。
    - f_pe: 当前机器上使用的第一个PE号。
    - f_npu: 当前机器执行本样例使用的第一张NPU卡的卡号
