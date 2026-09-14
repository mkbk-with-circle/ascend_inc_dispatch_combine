<!-- 中文 / Chinese -->
# 样例介绍

## 示例场景

aclGraph图结构如下：
![image.png](https://raw.gitcode.com/user-images/assets/8546182/b9370686-7b23-4b69-b18e-606167315795/image.png 'image.png')
由于rdma allgather只发送Write，因此在rdma allgather算子中添加`aclshmemx_roce_barrier_all`接口进行同步，确保所有PE上的图均已执行到相应阶段，确保aclGraph图（model）的精度正常。

## 环境要求

同[rdma_demo](../rdma_demo/README.md)中的环境要求。

## 使用方式

1. 在shmem/目录编译。
    RDMA 编译参数（A2/A3，以及 Ascend950 的 `XSCALE` / `HNS_1825` 后端）详见 [编译与构建 - RDMA 参数使用说明](../../docs/compilation_build_guide.md#rdma参数使用说明)。

2. 在shmem/rdma_aclgraph_demo目录运行。
    > 注：Ascend950 平台需设置 `IBV_EXTEND_DRIVERS` 环境变量，参见[环境变量说明](../rdma_demo/README.md#ibv_extend_drivers-环境变量)。

    ```bash
    bash run.sh # 单机双卡用例
    ```

3. 双机8卡示例，修改对应的run.sh

    ```bash
    # 1机
    pids=()
    msprof --application="./build/bin/rdma_aclgraph_demo 8 0 tcp://{1机IP}:{端口号} 4 0 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 0
    pid=$!
    pids+=("$pid")

    msprof --application="./build/bin/rdma_aclgraph_demo 8 1 tcp://{1机IP}:{端口号} 4 0 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 1
    pid=$!
    pids+=("$pid")

    msprof --application="./build/bin/rdma_aclgraph_demo 8 2 tcp://{1机IP}:{端口号} 4 0 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 2
    pid=$!
    pids+=("$pid")

    msprof --application="./build/bin/rdma_aclgraph_demo 8 3 tcp://{1机IP}:{端口号} 4 0 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 3
    pid=$!
    pids+=("$pid")

    # 2机
    pids=()
    msprof --application="./build/bin/rdma_aclgraph_demo 8 4 tcp://{1机IP}:{端口号} 4 4 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 4
    pid=$!
    pids+=("$pid")

    msprof --application="./build/bin/rdma_aclgraph_demo 8 5 tcp://{1机IP}:{端口号} 4 4 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 5
    pid=$!
    pids+=("$pid")

    msprof --application="./build/bin/rdma_aclgraph_demo 8 6 tcp://{1机IP}:{端口号} 4 4 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 6
    pid=$!
    pids+=("$pid")

    msprof --application="./build/bin/rdma_aclgraph_demo 8 7 tcp://{1机IP}:{端口号} 4 4 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 7
    pid=$!
    pids+=("$pid")
    ```

4. 命令行参数说明

    ```bash
    ./rdma_aclgraph_demo <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>
    ```

    - n_pes: 全局Pe数量。
    - pe_id: 当前Pe号。
    - ipport: SHMEM初始化需要的IP及端口号，格式为`tcp://<IP>:<端口号>`。如果执行跨机测试，需要将IP设为pe0所在Host的IP。
    - g_npus: 当前机器上启动的NPU数量。
    - f_pe: 当前机器上使用的第一个Pe号。
    - f_npu: 当前机器上使用的第一个NPU卡号。

---

<!-- English -->
## Example Scenario
The ACLGraph structure is as follows:
![image.png](https://raw.gitcode.com/user-images/assets/8546182/091ff732-56c0-431e-be32-21ab6f725de4/image.png 'image.png')
As `rdma_allGather` only issues write operations, it is necessary to add the `aclshmemx_handle_wait` API both before and after `rdma_allGather` during graph construction. This ensures synchronization so that all PEs have executed the graph up to the corresponding stage, thereby guaranteeing the correctness of precision in the ACLGraph model.

## Environment Requirements

- Before running this example, ensure that the RDMA environment is available (the RDMA NIC and driver have been correctly installed and configured).

### Checking the RDMA Environment
```bash
lspci | grep -i RDMA
for i in {0..7}; do hccn_tool -i $i -net_health -g; done
```
If the following command output is displayed, the environment is available:
![](../../docs/images/rdma_env.png)

## Instructions
1. Build in the `shmem/` directory:
```bash
bash scripts/build.sh -enable_rdma -examples
```
2.1 Run in the `shmem/rdma_aclgraph_demo` directory:
```bash
bash run.sh # Single-server dual-device example
```
2.2 For a dual-server 8-device example, modify the corresponding `run.sh` file.
```bash
# Server 1
pids=()
msprof --application="./build/bin/rdma_aclgraph_demo 8 0 tcp://{IP address of server 1}:{Port number } 4 0 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 0
pid=$!
pids+=("$pid")

msprof --application="./build/bin/rdma_aclgraph_demo 8 1 tcp://{IP address of server 1}:{Port number } 4 0 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 1
pid=$!
pids+=("$pid")

msprof --application="./build/bin/rdma_aclgraph_demo 8 2 tcp://{IP address of server 1}:{Port number } 4 0 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 2
pid=$!
pids+=("$pid")

msprof --application="./build/bin/rdma_aclgraph_demo 8 3 tcp://{IP address of server 1}:{Port number } 4 0 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 3
pid=$!
pids+=("$pid")

# server 2
pids=()
msprof --application="./build/bin/rdma_aclgraph_demo 8 4 tcp://{IP address of server 1}:{Port number } 4 4 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 4
pid=$!
pids+=("$pid")

msprof --application="./build/bin/rdma_aclgraph_demo 8 5 tcp://{IP address of server 1}:{Port number } 4 4 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 5
pid=$!
pids+=("$pid")

msprof --application="./build/bin/rdma_aclgraph_demo 8 6 tcp://{IP address of server 1}:{Port number } 4 4 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 6
pid=$!
pids+=("$pid")

msprof --application="./build/bin/rdma_aclgraph_demo 8 7 tcp://{IP address of server 1}:{Port number}} 4 4 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 7
pid=$!
pids+=("$pid")
```
3. Parameters in the command line
    `./rdma_aclgraph_demo <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>`

- n_pes: number of global PEs.
- pe_id: PE ID of the current process.
- ipport: IP address and port number required for SHMEM initialization. The format is tcp://`<IP_address>:<port_number>`. To perform a cross-server test, set the IP address to the IP address of the host where PE0 is located.
- g_npus: number of NPUs started on the current server.
- f_pe: ID of the first PE used on the current server.
- f_npu: ID of the first NPU used on the current server.
