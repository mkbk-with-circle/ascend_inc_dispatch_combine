# 示例介绍

shmem_perftest是用于测试AscendC::DataCopy、shmem MTE/UDMA/RDMA引擎以及SIMT RMA接口性能的参数化测试示例集合，包含五个子示例：

- **ascendc_perftest**：测试AscendC::DataCopy性能（不支持Ascend950）
- **mte_perftest**：测试shmem MTE引擎性能
- **udma_perftest**：测试shmem UDMA低阶接口性能（仅Ascend950）
- **rdma_perftest**：测试shmem RDMA低阶接口性能（需开启RDMA支持编译）
- **simt_rma_perftest**：测试SIMT RMA gm2gm接口性能（仅Ascend950，需开启SIMT支持编译）

该示例可以帮助用户对比多种数据传输方式的性能表现。**该脚本测试结果仅做参考，性能以实际场景为准**

## Python依赖

如果需要生成性能图表和Markdown报告，需要安装以下Python依赖：

```bash
pip install pandas matplotlib seaborn numpy tabulate
```

## 使用说明

### 快速开始

在shmem根目录下编译并运行：

```bash
# 编译示例
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# A2/A3 平台
bash scripts/build.sh -examples
# Ascend950 平台
bash scripts/build.sh -soc_type Ascend950 -examples

# 如需运行 simt_rma_perftest，需启用SIMT支持并指定Ascend950
bash scripts/build.sh -examples -enable_simt -soc_type Ascend950

# 如需运行 rdma_perftest，需启用RDMA支持（RDMA参数详见docs/compilation_build_guide.md）
# A2/A3 平台
bash scripts/build.sh -examples -enable_rdma
# Ascend950 平台（需指定RDMA后端：XSCALE 或 HNS_1825）
bash scripts/build.sh -soc_type Ascend950 -examples -enable_rdma -rdma_backend HNS_1825

# 如需运行 --memory-type dram，需启用CANN模式编译
# A2/A3 平台
bash scripts/build.sh -examples -cann
# Ascend950 平台
bash scripts/build.sh -soc_type Ascend950 -examples -cann

# 运行已编译示例（默认模式 all，仅生成CSV）
# 注：all 模式包含 ascendc/mte/udma/simt，不包含 rdma_perftest（需单独运行）
cd examples/shmem_perftest
bash run.sh -m all -t put -d float -fpe 0

# 运行已编译示例，并生成性能图表
bash run.sh -m all -t put -d float -fpe 0 -a plot

# 运行已编译示例，并生成性能图表和Markdown报告
bash run.sh -m all -t put -d float -fpe 0 -a md
```

### 外层run.sh参数说明

下表列出外层 `run.sh` 支持的参数及其默认值。**各子示例对同一参数的支持范围和默认值可能不同**（例如 `-t` 可选值、`-b` 是否强制为 1、`--ub-size` 最小值等），具体以各子示例 README 为准：

| 参数 | 说明 | 外层默认值 |
|------|------|---------|
| `-t\|--test-type <type>` | 测试类型，可选值因引擎而异（见下方说明） | put |
| `-d\|--datatype <type>` | 数据类型 (float\|int8\|int16\|int32\|int64\|uint8\|uint16\|uint32\|uint64\|char\|all) | float |
| `-b\|--block-size <size>` | 设置核数（udma/rdma 强制为 1） | 32 |
| `--block-range <min> <max>` | 设置核数范围（udma/rdma 强制为 1） | 32-32 |
| `-e\|--exponent <exponent>` | 设置数据量的幂数 | - |
| `--exponent-range <min> <max>` | 设置数据量的幂数范围 | 3-20 |
| `--loop-count <count>` | 设置循环次数 | 1000 |
| `--ub-size <size>` | 设置UB size(KB)；rdma_perftest 至少需要 192B，具体以子示例 README 为准 | 16 |
| `--memory-type <hbm\|dram>` | 设置mte_perftest使用的SHMEM内存类型（仅mte生效） | hbm |
| `--batch <N>` | BW 路径下每 N 次 `*_nbi` 后 `quiet`；UDMA/RDMA 的具体限制以子示例 README 为准。RDMA XSCALE 要求 `batch < 1024`，非法值由 `rdma_perftest/run.sh` 自动改为 100 | 0 |
| `-pes <size>` | 设置PE大小 | 2 |
| `-ipport <ip:port>` | 设置IP端口 | tcp://127.0.0.1:8760 |
| `-gnpus <num>` | 设置NPU数量 | 2 |
| `-fnpu <id>` | 设置首个NPU ID | 0 |
| `-fpe <id>` | 设置首个PE ID | 0 |
| `-m\|--mode <ascendc\|mte\|udma\|simt\|all>` | 设置运行模式 (ascendc=只跑ascendc, mte=只跑mte, udma=只跑udma, simt=只跑simt, all=全跑), 默认all |
| `-a\|--analyse <none\|plot\|md>` | 设置分析模式 (none=不生成, plot=只生成图, md=同时生成图和md), 默认none |

> **`-t/--test-type` 各引擎可选值差异**：
>
> - **ascendc_perftest**：`put` / `get` / `ub2gm_local` / `ub2gm_remote` / `gm2ub_local` / `gm2ub_remote` / `all`
> - **mte_perftest / rdma_perftest**：`put` / `bi_put` / `get` / `bi_get` / `all`
> - **udma_perftest**：`put` / `bi_put` / `get` / `bi_get` / `put_signal` / `all`
> - **simt_rma_perftest**：`put` / `get`（由编译期常量 `OP_TYPE` 决定，`-t` 仅做一致性校验）
>
> 若指定的 `-t` 不被某引擎支持，外层脚本会跳过该引擎并打印 WARN。各参数的详细说明、约束及子示例独有参数（如 udma 的 `--metric`、rdma 的 `--sync-id` / `-q` 等）请参考对应子示例 README。

### 运行模式

- **all（默认）**：依次运行ascendc_perftest（不支持Ascend950，950上运行会报错）、mte_perftest、udma_perftest（udma仅在Ascend950上有效）、simt_rma_perftest（需SIMT编译，否则跳过）
- **ascendc**：只运行ascendc_perftest（不支持Ascend950，950上运行会报错）
- **mte**：只运行mte_perftest
- **udma**：只运行udma_perftest
- **simt**：只运行simt_rma_perftest

> 注：`rdma_perftest` 不包含在上述运行模式中，需单独进入 `rdma_perftest/` 目录运行，且编译时需启用 `-enable_rdma` 参数，详见 [rdma_perftest/README.md](./rdma_perftest/README.md)。

### SIMT RMA测试约束

`simt_rma_perftest` 固定使用两卡进行性能测试，`-pes` 和 `-gnpus` 必须为 2。该子示例的 `OP_TYPE` 和 `DATA_SIZE` 由 `simt_rma_perftest/main.cpp` 中的编译期常量决定；外层 `-t` / `-d` 只有在用户显式传入时才会转发给SIMT子脚本进行一致性校验，不会改变实际测试的操作类型或数据位宽。

因此，`-t all` 或 `-d all` 不会展开运行SIMT矩阵，顶层脚本会跳过 `simt_rma_perftest` 并打印提示。如需测试其他SIMT RMA操作或数据位宽，请修改 `simt_rma_perftest/main.cpp` 中的编译期常量后重新编译。

### DRAM内存测试约束

`--memory-type dram` 仅作用于 mte_perftest，会使用 `aclshmemx_malloc(..., HOST_SIDE)` 分配Host侧DRAM内存。该功能依赖CANN模式，编译时必须使用：

```bash
# A2/A3 平台
bash scripts/build.sh -examples -cann
# Ascend950 平台
bash scripts/build.sh -soc_type Ascend950 -examples -cann
```

DRAM测试需要运行环境支持Host侧DRAM内存访问，相关硬件和可用内存约束可参考 [rma_d2h_demo](../rma_d2h_demo/README.md) 的“约束限制”章节。mte_perftest默认配置1GB本地内存；当测试参数需要更大本地内存时，程序会按数据量自动上调，运行前需确保可用DRAM空间大于实际本地内存配置。

## 输出结果

运行完成后，结果会统一放在 `examples/shmem_perftest/output/` 目录下：

```text
examples/shmem_perftest/output/
├── ascendc_perftest/    # ascendc_perftest测试结果
│   ├── 0_put_float.csv
│   └── picture/          # （使用--plot时生成）性能图表
│       └── 0_put_float/
│           ├── 0_put_float_UBsize_compare.png
│           ├── 0_put_float_Core_compare.png
│           ├── 0_put_float_bandwidth_max_heatmap.png
│           └── 0_put_float_bandwidth_mean_heatmap.png
├── mte_perftest/         # mte_perftest测试结果
│   ├── put_float_0.csv
│   └── picture/          # （使用--plot时生成）性能图表
│       └── put_float_0/
│           ├── put_float_0_UBsize_compare.png
│           ├── put_float_0_Core_compare.png
│           ├── put_float_0_bandwidth_max_heatmap.png
│           └── put_float_0_bandwidth_mean_heatmap.png
├── udma_perftest/        # udma_perftest测试结果
│   └── udma_bw_put_float_0.csv
├── rdma_perftest/        # rdma_perftest测试结果（单独运行rdma_perftest后拷贝至此）
│   └── rdma_put_float_0.csv
├── simt_rma_perftest/    # simt_rma_perftest测试结果
│   └── 32_32-32_put_simt_3-20_l1000_t1024_.csv
└── performance_report.md  # （使用--markdown时生成）性能测试报告
```

## 单独测试子示例

如果需要单独测试某个子示例，请进入对应子目录查看详细README：

- **ascendc_perftest**：请参考 [ascendc_perftest/README.md](./ascendc_perftest/README.md)
- **mte_perftest**：请参考 [mte_perftest/README.md](./mte_perftest/README.md)
- **udma_perftest**：请参考 [udma_perftest/README.md](./udma_perftest/README.md)
- **rdma_perftest**：请参考 [rdma_perftest/README.md](./rdma_perftest/README.md)
- **simt_rma_perftest**：请参考 [simt_rma_perftest/README.md](./simt_rma_perftest/README.md)
