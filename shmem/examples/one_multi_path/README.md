# One Path + Multi Path 单核分片搬运样例

## 样例介绍

本样例用于验证 Ascend950 上一个 AIV 核通过 `one_path` 和 `multi_path` 两种路径分别访问两块独立的
远端 HBM，重点展示 fabric handle 的交换、双路径映射以及单核分片搬运流程。

每个 PE 申请并导出两块本地物理内存，通过 TCP allgather 交换两个 fabric handle，
并按 ring 关系选择下一个 PE：

```text
peer_pe = (pe_id + 1) % n_pes
```

每个 PE 将 peer 的两个 fabric handle 各导入一次，分别使用固定的 `one_path` `linkType=2` 和
`multi_path` `linkType=3`，映射得到两个 remote VA。两块远端内存保存相同的测试数据，kernel 按
32 字节边界从中间分成两半，使用一次 kernel launch 和一个 AIV block 完成搬运：

```text
同一个 AIV block
├── one_path remote VA -> data[0, split)
└── multi_path remote VA -> data[split, size)
```

两段搬运在同一个 kernel 中顺序执行。本样例只验证两种链路映射的数据访问正确性，不验证
`one_path` 和 `multi_path` 并发，也不采集或比较带宽。

## 支持平台与前置条件

- Ascend950 运行环境。
- CANN 提供 `aclrtMemExportToShareableHandleV2`、`aclrtMemImportFromShareableHandleV2` 和
  `aclrtMemMapSetLink`。
- 跨机运行时，`ipport` 后的两个连续端口可达；`port+1` 用于交换 fabric handle，`port+2` 用于交换状态。
- ring 中每一对相邻 PE 都支持 `one_path` `linkType=2` 和 `multi_path` `linkType=3`。

本样例不支持 A2/A3。`linkType` 的实际路由由 CANN 运行时和硬件组网决定，运行前应确认所用
CANN 与驱动版本支持上述接口及对应链路类型。

## 目录结构

```text
examples/one_multi_path/
├── CMakeLists.txt                    # 样例构建配置
├── main.cpp                          # 初始化、fabric handle 交换、映射与结果校验
├── one_multi_path_kernel.cpp         # 单 AIV block 双路径分片搬运 kernel
├── one_multi_path_kernel.h           # kernel 启动接口
├── README.md                         # 使用说明
└── run.sh                            # 单机或多机多进程启动脚本
```

## 调用流程

1. 每个 PE 完成 `aclInit`、`aclrtSetDevice` 和 `aclshmemx_init_attr`。
2. 每个 PE 使用 `aclrtMallocPhysical` 申请两块本地 HBM，并将两块内存的测试数据填充为 `pe_id + 10`。
3. 所有 PE 通过 TCP gather/broadcast 交换两个 fabric handle。
4. 每个 PE 选择 ring 下一 PE，并将其两个 fabric handle 各导入一次。
5. 两个 imported handle 分别调用 `aclrtMemMapSetLink(2)` 和 `aclrtMemMapSetLink(3)`。
6. 两个 imported handle 分别映射为 `one_path` remote VA 和 `multi_path` remote VA。
7. 数据按 32 字节边界分成两半，启动一个 AIV block；单次搬运超过 16 KB 时在 kernel 内分块循环。
8. 同一个 kernel 先通过 `one_path` remote VA 搬运前半段，再通过 `multi_path` remote VA 搬运后半段。
9. 将结果复制到 Host，逐个 `int32_t` 元素校验后交换最终状态并释放资源。

## 编译

```bash
bash scripts/build.sh -cann -examples -soc_type Ascend950
```

编译产物：

```text
build/bin/one_multi_path
```

样例在启用 examples 且 `SOC_TYPE=Ascend950` 时参与构建。

`-cann` 用于启用 CANN 开放接口构建，`-examples` 用于构建样例，`-soc_type Ascend950` 用于选择
Ascend950 后端。完整编译参数见[编译与构建说明](../../docs/compilation_build_guide.md)。

## 运行

单机两卡：

```bash
cd examples/one_multi_path
bash run.sh -pes 2 -gnpus 2 -fpe 0 -fnpu 0 -size 2048
```

单机四卡：

```bash
bash run.sh -pes 4 -gnpus 4 -fpe 0 -fnpu 0 -size 2048
```

两台机器各使用两张卡时，两边必须使用相同的 bootstrap 地址。

机器 A：

```bash
bash run.sh -pes 4 -ipport tcp://<A_IP>:8899 -gnpus 2 -fpe 0 -fnpu 0 -size 2048
```

机器 B：

```bash
bash run.sh -pes 4 -ipport tcp://<A_IP>:8899 -gnpus 2 -fpe 2 -fnpu 0 -size 2048
```

脚本要求每台机器负责一段连续 PE，并将本机 PE 按顺序映射到连续的逻辑 NPU：

```text
local_index = pe_id - fpe
device_id   = fnpu + local_index
```

多机运行时，各机器的 `-pes` 和 `-ipport` 必须一致，`[fpe, fpe + gnpus)` 区间不能重叠，且所有
机器的 PE 区间合并后应覆盖 `[0, pes)`。`ipport` 中的 IP 应为启动 PE 0 的机器可被其他机器访问的地址。

## 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-pes` | `2` | PE 总数，至少为 2 |
| `-ipport` | `tcp://127.0.0.1:8899` | SHMEM bootstrap 地址 |
| `-gnpus` | `2` | 当前机器启动的 PE/NPU 数量 |
| `-fpe` | `0` | 当前机器第一个 PE ID |
| `-fnpu` / `-dev` | `0` | 当前机器第一个逻辑 NPU ID |
| `-size` | `2048` | 实际校验的数据大小，单位 KB，必须为正整数 |

`one_path` `linkType=2` 和 `multi_path` `linkType=3` 固定在样例内部，不需要通过命令行传入。
样例将每块物理内存的映射大小按 2 MB 大页向上取整，`-size` 仍表示实际搬运和校验的数据量。

## 预期输出

```text
PE 0: path=one_path, offset=0, size=1048576
PE 0: path=multi_path, offset=1048576, size=1048576
PE 0: single-core one_path/multi_path copy from PE 1 PASSED
PE 1: single-core one_path/multi_path copy from PE 0 PASSED
```

校验通过只能证明两个 remote VA 均可正确访问。物理流量是否按 `linkType` 经过预期链路，仍由
CANN 运行时和实际组网决定，必要时应结合平台链路计数器确认。

## 常见问题

- 编译后没有 `build/bin/one_multi_path`：确认同时传入 `-examples` 和 `-soc_type Ascend950`，并检查
  CMake 输出中的 `SOC_TYPE`、`USE_EXAMPLES` 和 `ACLSHMEM_ONE_MULTI_PATH`。只有 CANN 头文件和
  `libascendcl.so` 同时提供 Fabric Handle API 与 `aclrtMemMapSetLink` 时，CMake 才会编译本样例。
- 初始化失败或进程持续等待：确认所有 PE 使用相同且可达的 `-ipport`，PE 区间完整无重叠，并确认
  bootstrap 端口未被其他任务占用。
- `aclrtMemMapSetLink` 或 remote handle 映射失败：确认 CANN、驱动和组网支持目标 PE 间的
  `linkType=2/3`，并检查 ring 相邻 PE 的实际拓扑。
- 数据校验通过但无法确认物理链路：样例不读取链路计数器，需结合平台侧链路观测工具进一步确认。
