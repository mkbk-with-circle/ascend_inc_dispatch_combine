<!-- 中文 / Chinese -->
# 样例介绍

## 使用方式

### 1. 编译项目

**default以外的流程执行需要自行安装并导入MPI环境变量**[安装参考](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/hccltool/HCCLpertest_16_0002.html)
该用例基于mpich实现，安装Open MPI或其他MPI如果遇到报错可能需要自行调整脚本中mpi相关参数。

```bash
# mpich安装在默认路径时可参考，需根据实际情况替换。
export PATH=/usr/local/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH
```

在 `shmem/` 根目录下执行编译脚本：

- A2/A3 平台:

```bash
bash scripts/build.sh
source install/set_env.sh
```

- Ascend950 平台:

```bash
bash scripts/build.sh -soc_type Ascend950
source install/set_env.sh
```

### 2. 运行 init 用例

用例目录

```bash
cd examples/init
```

用例接收两个参数，第一个参数指定流程，第二个参数指定pe数量（不能超过卡数）。不指定时默认`mode = default`,`pesize = 2`。
执行 default 流程，2 pe

```bash
bash run.sh -mode default -pesize 2
```

执行 mpi 流程，2 pe

```bash
bash run.sh -mode mpi -pesize 2
```

执行 uid 流程，2 pe

```bash
bash run.sh -mode uid -pesize 2
```

执行 uid_multi 流程，4 pe

```bash
# enpxxx需要替换为ip addr指令获得的网卡
export SHMEM_UID_SOCK_IFNAME=enpxxxxxxx:inet4

bash run.sh -mode uid_multi -pesize 4

unset SHMEM_UID_SOCK_IFNAME
```

执行 uid_multi_stress 流程，4 pe（uid_multi 循环压测）

```bash
# 编译时定义 STRESS_LOOP_COUNT=20，循环执行 20 次 uid_multi 创建/销毁流程
# enpxxx需要替换为ip addr指令获得的网卡
export SHMEM_UID_SOCK_IFNAME=enpxxxxxxx:inet4

bash run.sh -mode uid_multi_stress -pesize 4

unset SHMEM_UID_SOCK_IFNAME

```

如需修改循环次数，编辑 CMakeLists.txt 中 RUN_MODE=6 对应的 `STRESS_LOOP_COUNT` 值后重新运行。

执行 uid_default 流程，2 pe

```bash
bash run.sh -mode uid_default -pesize 2
```

### 3. 跨机运行

> **注**：`-ipport` 只传 `host:port`（脚本会自动拼接 `tcp://` 前缀）
>
> - IPv4 地址：`192.168.1.100:8666`
> - **主机名**：`my-server:8666`（主机名通过系统 DNS 解析为实际 IP 地址）

执行 default 流程，双机，每个机器2 pe，在两台机器分别执行如下命令。

```bash
# pe0所在机器（${该机器的ip:port}也可使用主机名，如my-server:8666）
bash run.sh -mode default -pesize 4 -fpe 0 -gnpus 2 -ipport ${该机器的ip:port}
# 其他机器
bash run.sh -mode default -pesize 4 -fpe 2 -gnpus 2 -ipport ${pe0机器的ip:port}
```

执行 mpi/uid 流程，双机，每个机器2 pe。
这两个流程依赖mpi能力，跨机需自行配置hostfile文件，以mpich配置文件为例

```sh
# 请替换成实际机器的ip，且保证pe0机器在最前面,冒号后配置每个机器支持启动的pe数，pe数需小于卡数，建议每台机器启动pe数量一致。
0.0.0.1:2
0.0.0.2:2
```

**mpirun要求多机可执行文件位置一致，请先用相同mode编译所有机器上的样例。如果mode不一致会导致报错或者卡死。**

```sh
bash run.sh -build -mode mpi #uid流程则讲mode参数改为uid
```

然后在pe0执行
mpi流程

```bash
# gnpu参数用来指定每台设备自身的pe数，在所有机器启动卡数一致时，设置为单机pe数量可以保证都从0卡开始启动。
bash run.sh -mode mpi -gnpus 2
```

uid流程

```bash
# pe0所在机器（${该机器的ip:port}也可使用主机名，如my-server:8666）
bash run.sh -mode uid -gnpus 2 -ipport ${pe0机器的ip:port}
```

**如果想再次以 mpi/uid 模式执行单机用例推荐删除hostfile文件。**

---

<!-- English -->
### Instructions

#### 1. Building a Project

**To run non-default workflows, you must manually install MPI and import the corresponding environment variables.** [Installation Reference](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/hccltool/HCCLpertest_16_0002.html)
This test case is implemented based on MPICH. If an error is reported during the installation of Open MPI or any other MPI, you may need to adjust the MPI-related parameters in the script.

```bash
# MPICH is installed in the default path. Replace the path with the actual path as needed.
export PATH=/usr/local/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH
```

Run the `build.sh` script in the root (`shmem/`) directory.

```bash
bash scripts/build.sh
source install/set_env.sh
```

##### 2. Running the init Case

Case directory
```bash
cd examples/init
```

The case accepts two parameters. The first parameter specifies the workflow, and the second parameter specifies the number of PEs (which cannot exceed the number of devices). If not specified, `mode = default` and `pesize = 2` are used by default.
Execute the default workflow with 2 PEs.
```bash
bash run.sh -mode default -pesize 2
```

Execute the MPI workflow with 2 PEs.
```bash
bash run.sh -mode mpi -pesize 2
```

Execute the UID workflow with 2 PEs.
```bash
bash run.sh -mode uid -pesize 2
```

Execute the uid_multi workflow with 4 PEs.
```bash
# Replace enpxxx with the NIC obtained by running the ip addr command.
export SHMEM_UID_SOCK_IFNAME=enpxxxxxxx:inet4

bash run.sh -mode uid_multi -pesize 4

unset SHMEM_UID_SOCK_IFNAME
```

Execute the uid_default workflow with 2 PEs.
```bash
bash run.sh -mode uid_default -pesize 2
```

##### 3. Cross-Server Running
Execute the default workflow on two servers, with 2 PEs on each server. Run the following commands on both servers:
```bash
# Server where PE0 is located
bash run.sh -mode default -pesize 4 -fpe 0 -gnpus 2 -ipport ${IP address:port number of the server}
# Any other server
bash run.sh -mode default -pesize 4 -fpe 2 -gnpus 2 -ipport ${IP address:port number of the server where PE0 is located}
```

Execute the MPI/UID workflow on two servers, with 2 PEs on each server.
The two workflows depend on the MPI capability. For cross-server execution, you need to configure the hostfile file. The following uses the MPICH configuration file as an example.
```sh
# Replace the IP addresses with the actual ones and ensure that the server where PE0 is located is placed at the very top. After the colon (:), configure the number of PEs that can be started on each server. The number of PEs must be less than the number of devices. It is recommended that the number of PEs started on each server be the same.
0.0.0.1:2
0.0.0.2:2
```
**mpirun requires that the locations of the executable files on multiple servers be the same. Compile the samples on all servers in the same mode first. If the modes are inconsistent, an error will be reported or the system will be suspended.**
```sh
bash run.sh -build -mode mpi # For the UID workflow, change the mode parameter value to uid.
```
Then, run the following commands on PE0:
MPI workflow
```bash
# The gnpu parameter specifies the number of PEs on each server. When the number of devices started on all servers is the same, set this parameter to the number of PEs on a single server to ensure that execution always starts from device 0 on every server.
bash run.sh -mode mpi -gnpus 2
```
UID workflow
```bash
bash run.sh -mode uid -gnpus 2 -ipport ${IP address:port number of the server where PE0 is located}
```
**If you want to execute a single-device test case again in MPI or UID mode, you are advised to delete the hostfile file.**
