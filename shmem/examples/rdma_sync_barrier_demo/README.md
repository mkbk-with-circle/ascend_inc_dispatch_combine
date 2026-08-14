# 示例场景

本示例演示通过 RDMA 传输通路，使用低阶接口 `aclshmemx_roce_put_nbi` 配合同步/栅栏原语完成多 PE 之间的全交换（all-gather）数据通信，并验证数据正确性。

具体包含以下 8 种同步模式：

- **sync_all**：使用 `aclshmemx_roce_put_nbi` + `aclshmemx_roce_quiet` + `aclshmemx_roce_sync_all` 完成全局同步。
- **sync_all_buf**：使用 `aclshmemx_roce_put_nbi` + `aclshmemx_roce_quiet` + `aclshmemx_roce_sync_all(buf, sync_id)` 显式传入 UB buffer 和 sync_id。
- **barrier_all**：使用 `aclshmemx_roce_put_nbi` + `aclshmemx_roce_barrier_all` 完成全局同步（barrier 内部自动 quiet）。
- **barrier_all_buf**：使用 `aclshmemx_roce_put_nbi` + `aclshmemx_roce_barrier_all(buf, sync_id)` 显式传入 UB buffer 和 sync_id。
- **sync_team**：使用 `aclshmemx_roce_put_nbi` + `aclshmemx_roce_quiet` + `aclshmemx_roce_team_sync(team)` 在 team 内完成同步。
- **sync_team_buf**：使用 `aclshmemx_roce_put_nbi` + `aclshmemx_roce_quiet` + `aclshmemx_roce_team_sync(team, buf, sync_id)` 在 team 内显式传入参数。
- **barrier_team**：使用 `aclshmemx_roce_put_nbi` + `aclshmemx_roce_barrier(team)` 在 team 内完成同步。
- **barrier_team_buf**：使用 `aclshmemx_roce_put_nbi` + `aclshmemx_roce_barrier(team, buf, sync_id)` 显式传入参数。

## 环境要求

同[rdma_demo](../rdma_demo/README.md)中的环境要求。

## 使用方式

### 编译

在shmem/目录编译。RDMA 编译参数（A2/A3，以及 Ascend950 的 `XSCALE` / `HNS_1825` 后端）详见 [编译与构建 - RDMA 参数使用说明](../../docs/compilation_build_guide.md#rdma参数使用说明)。

### 运行

> 注：Ascend950 平台需设置 `IBV_EXTEND_DRIVERS` 环境变量，参见[环境变量说明](../rdma_demo/README.md#ibv_extend_drivers-环境变量)。

#### 方式一：在 `examples/rdma_sync_barrier_demo` 目录下执行 `bash run.sh`

`run.sh` 支持通过参数指定 demo 类型，默认为 `sync_all`。

```bash
bash run.sh sync_all           # 运行 sync_all demo（默认）
bash run.sh sync_all_buf       # 运行 sync_all(buf, sync_id) demo
bash run.sh barrier_all        # 运行 barrier_all demo
bash run.sh barrier_all_buf    # 运行 barrier_all(buf, sync_id) demo
bash run.sh sync_team          # 运行 sync_team demo
bash run.sh sync_team_buf      # 运行 sync_team(buf, sync_id) demo
bash run.sh barrier_team       # 运行 barrier_team demo
bash run.sh barrier_team_buf   # 运行 barrier_team(buf, sync_id) demo
```

#### 方式二：在 `shmem/` 目录手动运行命令

- 单机 2 卡执行命令

```bash
export PROJECT_ROOT=<shmem-root-directory>
export IBV_EXTEND_DRIVERS=<path_to_plugin.so> # 仅 Ascend950 平台需要根据网卡类型进行环境变量设置，详见环境变量说明
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
./build/bin/rdma_sync_barrier_demo 2 0 tcp://127.0.0.1:8899 2 0 0 sync_all & # PE 0
./build/bin/rdma_sync_barrier_demo 2 1 tcp://127.0.0.1:8899 2 0 0 sync_all & # PE 1
```

> 注：\<shmem-root-directory\> 为 SHMEM 项目的根目录，\<path_to_plugin.so\> 为根据网卡类型设置的插件库路径。

- 跨机 2 卡执行命令

假设机器 A 的 IP 为 ip1，机器 B 的 IP 为 ip2。
在机器 A 执行如下命令：

```bash
export PROJECT_ROOT=<shmem-root-directory>
export IBV_EXTEND_DRIVERS=<path_to_plugin.so> # 仅 Ascend950 平台需要根据网卡类型进行环境变量设置，详见环境变量说明
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
./build/bin/rdma_sync_barrier_demo 2 0 tcp://ip1:8765 1 0 0 sync_all # PE 0
```

同时，在机器 B 执行如下命令：

```bash
export PROJECT_ROOT=<shmem-root-directory>
export IBV_EXTEND_DRIVERS=<path_to_plugin.so> # 仅 Ascend950 平台需要根据网卡类型进行环境变量设置，详见环境变量说明
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
./build/bin/rdma_sync_barrier_demo 2 1 tcp://ip1:8765 1 1 0 sync_all # PE 1
```

> 注：\<shmem-root-directory\> 为 SHMEM 项目的根目录，\<path_to_plugin.so\> 为根据网卡类型设置的插件库路径。
>
> 如需在容器中运行跨机测试，启动容器时指定 `--net=host` 模式即可。

- 双机 16 卡（每机 8 卡）执行命令示例

假设机器 A 的 IP 为 ip1，机器 B 的 IP 为 ip2。
在机器 A 执行如下命令：

```bash
export PROJECT_ROOT=<shmem-root-directory>
export IBV_EXTEND_DRIVERS=<path_to_plugin.so> # 仅 Ascend950 平台需要根据网卡类型进行环境变量设置，详见环境变量说明
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
pids=()
for pe in $(seq 0 7); do
    ./build/bin/rdma_sync_barrier_demo 16 $pe tcp://ip1:8765 8 0 0 sync_all &
    pids+=($!)
done
for pid in ${pids[@]}; do wait $pid; done
```

在机器 B 执行如下命令：

```bash
export PROJECT_ROOT=<shmem-root-directory>
export IBV_EXTEND_DRIVERS=<path_to_plugin.so> # 仅 Ascend950 平台需要根据网卡类型进行环境变量设置，详见环境变量说明
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
pids=()
for pe in $(seq 8 15); do
    ./build/bin/rdma_sync_barrier_demo 16 $pe tcp://ip1:8765 8 8 0 sync_all &
    pids+=($!)
done
for pid in ${pids[@]}; do wait $pid; done
```

### 命令行参数说明

```bash
./rdma_sync_barrier_demo <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu> [demo_type]
```

| 参数 | 说明 |
|------|------|
| n_pes | 全局 PE 数量 |
| pe_id | 当前进程的 PE 号 |
| ipport | SHMEM 初始化需要的 IP 及端口号，格式为 `tcp://<IP>:<端口号>`。若执行跨机测试，需将 IP 设为 PE 0 所在 Host 的 IP |
| g_npus | 当前机器上启动的 NPU 卡数量 |
| f_pe | 当前机器上使用的第一个 PE 号 |
| f_npu | 当前机器执行本样例使用的第一张 NPU 卡的卡号 |
| demo_type | 可选，指定 demo 类型，默认为 `sync_all`。支持的值见下方表格 |

**demo_type 取值：**

| 值 | 说明 |
|------|------|
| `sync_all` | 使用 `aclshmemx_roce_sync_all` 完成全局同步 |
| `sync_all_buf` | 使用 `aclshmemx_roce_sync_all(buf, sync_id)` 完成全局同步 |
| `barrier_all` | 使用 `aclshmemx_roce_barrier_all` 完成全局同步 |
| `barrier_all_buf` | 使用 `aclshmemx_roce_barrier_all(buf, sync_id)` 完成全局同步 |
| `sync_team` | 使用 `aclshmemx_roce_team_sync(team)` 在 team 内完成同步 |
| `sync_team_buf` | 使用 `aclshmemx_roce_team_sync(team, buf, sync_id)` 在 team 内完成同步 |
| `barrier_team` | 使用 `aclshmemx_roce_barrier(team)` 在 team 内完成同步 |
| `barrier_team_buf` | 使用 `aclshmemx_roce_barrier(team, buf, sync_id)` 在 team 内完成同步 |

### 预期输出

每个 PE 向所有其他 PE 发送自己的数据（值为 `pe_id + 10`），同步完成后各 PE 校验收到的数据是否正确。校验通过后输出：

```sh
[PASS] check success, pe=<pe_id>
```

若校验失败，会打印具体的不匹配信息：

```sh
[FAIL] pe=<pe_id> offset=<offset> got=<actual> expected=<expected>
```
