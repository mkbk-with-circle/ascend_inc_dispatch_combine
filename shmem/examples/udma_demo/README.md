# 样例介绍

## 版本和平台支持说明

- UDMA 样例仅支持 Ascend950 平台（`__NPU_ARCH__ == 3510`），非 Ascend950 平台不支持运行该样例。
- CANN 需提供 Ascend950 UDMA/HCOMM 资源接口，并安装与 Ascend950 匹配的 toolkit 和 ops 包，建议使用 CANN 9.1.0 及以上版本。低版本 CANN 如果缺少 `HcommEndpointCreate`、`HcommMemReg`、`HcommChannelCreate` 等 HCOMM UDMA API，构建阶段会关闭 `ACLSHMEM_UDMA_SUPPORT`，该样例无法运行。
- 编译前需完成 CANN 环境配置，例如执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，并使用 `-soc_type Ascend950` 编译。

使用方式:

1.在shmem/目录编译:
```bash
bash scripts/build.sh -examples -soc_type Ascend950
```

2.在shmem/目录运行:
```bash
bash examples/udma_demo/run.sh 0 # allgather测试
bash examples/udma_demo/run.sh 1 # put signal 测试
```
默认按单机8卡启动，脚本依次拉起`PE 0`到`PE 7`，并等待所有进程退出。

`run.sh`会自动设置`PROJECT_ROOT`、`LD_LIBRARY_PATH`，并根据`ipport`参数设置`SHMEM_UID_SESSION_ID`，无需手动导出这些环境变量。

UDMA 高阶 RMA 接口默认使用 `PIPE_MTE3` 下发 WQE，需要一段 UB
scratch。默认配置为 `offset = 189 * 1024`、`ub_size = 128` 字节、
`sync_id = 0`；如果样例或业务 kernel 需要复用这段 UB，可通过
`aclshmemx_set_udma_config(offset, ub_size, sync_id)` 调整。`ub_size`
必须不小于 128 字节，用于容纳当前 UDMA 数据搬移操作的一块完整
WQE staging block。

3.run.sh脚本命令行参数说明

脚本按位置解析参数，所有参数均可选，缺省时使用默认值。
```bash
bash examples/udma_demo/run.sh <test_type> <n_pes> <g_npus> <ipport> <f_pe> <local_pes> <f_npu>
```

- test_type: 测试类型，0表示运行all-gather测试（默认），1表示运行put signal测试。
- n_pes: 跨所有机器的全局PE总数（默认8）。
- g_npus: 本机使用的NPU卡数量（默认8）。
- ipport: bootstrap节点地址，格式为`IP:PORT`（默认`127.0.0.1:8899`）。必须是node0的IP，且所有节点均可达。
- f_pe: 本机使用的第一个全局PE号（默认0）。
- local_pes: 本机要拉起的PE进程数（默认等于n_pes）。
- f_npu: 本机使用的第一张NPU卡的卡号（默认0）。

多机启动示例（2机 × 8卡 = 16 PE，假设node0的IP为192.168.1.10）：
```bash
# node0：f_pe=0，拉起全局 PE 0~7
bash examples/udma_demo/run.sh 0 16 8 192.168.1.10:8899 0 8 0
# node1：f_pe=8，拉起全局 PE 8~15
bash examples/udma_demo/run.sh 0 16 8 192.168.1.10:8899 8 8 0
```

mssanitizer 检测：当前实验在 MTE3 路径恢复结构体字段填充，并由 all-gather 与 put signal 两个 kernel 在首次调用前对完整 128 字节 scratch 做一次性清零（`udma_demo_kernel.cpp` 中的 `init_udma_wqe_scratch`）。这既初始化结构体位域读改写可能读取的原值，也覆盖 put signal 的 32 字节尾部填充，用于验证调用方全量预初始化能否消除 initcheck 的未初始化读。检测运行方式：

```bash
mssanitizer -- bash examples/udma_demo/run.sh 0
mssanitizer -- bash examples/udma_demo/run.sh 1
```

4.底层二进制命令行参数说明
```bash
./udma_demo <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu> [test_type]
```

- n_pes: 全局PE数量。
- pe_id: 当前进程对应的PE号。
- ipport: SHMEM初始化需要的IP及端口号，格式为tcp://<IP>:<端口号>。
- g_npus: 当前机器上启动的NPU卡的数量。
- f_pe: 当前机器上使用的第一个PE号。
- f_npu: 当前机器执行本样例使用的第一张NPU卡的卡号。
- test_type: 测试类型（可选），0表示运行all-gather测试（默认），1表示运行put signal测试。
