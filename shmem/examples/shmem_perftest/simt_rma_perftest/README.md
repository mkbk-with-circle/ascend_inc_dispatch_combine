# 样例介绍

本样例展示在 SIMD 与 SIMT 混合编译模式下，如何对 SIMT 远程内存访问（RMA）接口进行性能测试（Performance Test）。
测试用于评估单机两卡间 Device-to-Device 的 `gm2gm`（global memory 到 global memory）数据传输能力，覆盖**单向**的 `put` 与 `get` 操作，并输出带宽与时延统计。

## 测试模型

测试固定使用两张卡，PE 号分别为 0 和 1：

- **Active PE（PE 0）**：所有通信操作（`put` / `get`）的发起方。
- **Passive PE（PE 1）**：通信的对端，不主动发起操作。

两种操作的数据流向与校验方分别为：

| 操作 | 数据流向 | 校验方 |
| --- | --- | --- |
| `put` | Active PE 的对称内存 → Passive PE 的对称内存 | Passive PE |
| `get` | Passive PE 的对称内存 → Active PE 的对称内存 | Active PE |

## 性能测试方法

为获得贴近真实的带宽与时延、避免数据缓存（Data Cache）命中导致结果虚高，以及方便实现，本样例采用以下设计：

- **仅 Active PE 发起、源与目的同址**：`put` / `get` 是单边（one-sided）操作，全部由 Active PE 发起，Passive PE 仅作为远端对象、不参与计算（两卡通过 host 侧 barrier 同步）。每次传输的源地址与目的地址是**同一个对称内存偏移**，数据在两卡的相同偏移之间搬运。
- **预热与多轮取均值**：单次测量会先执行固定 100 次传输进行预热（不计入统计），再执行 `loops` 次传输进行采样。计时方式为：预热结束后，对这 `loops` 次传输整体计时（仅在首次采样前打点开始、末次采样后打点结束），统计得到的总时长再除以 `loops` 得到单次耗时，从而摊薄打点开销、排除冷启动的影响。
- **逻辑段与逻辑环**：每个 Core 物理上分配一块 1MB 的对称内存。设单次传输大小为 $X$ 字节、使用 $N$ 个 Core，定义每个 Core 的**逻辑段**大小为 $L = \min(1\text{MB},\ (100 + loops) \times X)$（且不小于 $X$，并保证为 $X$ 的整数倍）。$N$ 个逻辑段首尾相接，构成一个大小为 $N \times L$ 的**逻辑环**，Core $j$ 的逻辑段起始于环内偏移 $j \times L$。
- **滑动窗口遍历**：第 $i$ 次迭代时，所有 Core 共同构成一个滑动窗口，窗口整体在环内的起点为 $(i \times X) \bmod (N \times L)$。Core $j$ 本次传输的目标偏移为 $\big(i \times X + j \times L\big) \bmod (N \times L)$，即各 Core 在环上彼此错开一个逻辑段、各自传输一段长 $X$ 的数据。窗口每轮前进 $X$ 字节并在环内回绕。由于 $X \le L$，同一轮内各 Core 的数据段互不重叠；迭代之间由 `SyncAll` 保证不冲突。
- **避免缓存命中**：连续传输的地址不断前移，使其尽量落在不同缓存行上；当总传输量较小时逻辑环只占用刚好够用的空间，较大时按 1MB 上限循环复用，从而避免数据缓存命中抬高带宽读数。校验时按同样的逻辑段布局逐段比对（每段前 $L$ 字节），与实际写入的区域完全一致。

## 源文件宏定义配置

部分测试维度通过 `main.cpp` 开头的常量定义控制。在编译前可直接修改这些常量来改变底层接口的调用行为或数据位宽：

| 常量定义 | 含义与作用 | 可选值 / 默认值 |
| --- | --- | --- |
| `VF_TYPE` | 指令计算模式框架。 | `SIMT`（默认）、`SIMD` |
| `OP_TYPE` | 性能测试执行的具体操作。 | `PUT`（默认）、`GET`、`NONE`（仅以 `count = 0` 调用接口，测量接口调用开销而不传输实际数据） |
| `DATA_SIZE` | 所调用 RMA 底层接口的数据位宽（单位：bit）。修改后会切换到对应位宽的读写接口（如 `aclshmemx_get32_block` 切换为 `aclshmemx_get64_block`）。 | `8`、`16`、`32`（默认）、`64` |
| `THREAD_COUNT` | SIMT 模式下单 Core 启动的线程数，决定向量指令流的并发规模。 | 默认 `1024` |

> **提示**：修改上述常量后，需重新回到根目录执行编译（见下文），新配置才会生效。
> 由于部分原因，目前同一个编译单元中，两个仅调用相似simt rma接口的vf函数会导致编译错误（尽管编译不会报错，运行时会有错误），本样例通过修改源码以测试不同的simt RMA接口，并提供了常量定义以方便修改。

## 支持的设备

- Ascend950

## 使用方式

1. **配置 CANN 环境变量**
   编译前需先加载 CANN 的环境变量（按实际安装路径选择其一）：

   ```bash
   # 默认安装路径
   source /usr/local/Ascend/cann/bin/set_env.sh
   # 自定义安装路径
   source ${install_path}/cann/bin/set_env.sh
   ```

2. **编译项目**
   在 `shmem/` 根目录下执行编译脚本：

   ```bash
   bash scripts/build.sh -examples -enable_simt -soc_type Ascend950
   ```

3. **运行示例程序**
   进入示例目录并执行运行脚本：

   ```bash
   cd examples/shmem_perftest/simt_rma_perftest
   bash run.sh [options]
   ```

### 脚本参数说明

`run.sh` 支持以下参数，用于调节测试规模与条件：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `-pes <int>` | PE 总数。本测试固定两卡模型，必须为 2。 | 2 |
| `-ipport <ip:port>` | ACLSHMEM 初始化通信地址。 | `tcp://127.0.0.1:8760` |
| `-gnpus <int>` | 本机启动进程数 / NPU 数量。本测试固定为 2，传入其他值会报错。 | 2 |
| `-fnpu <int>` | 首个 NPU ID，实际 device id 为 `pe_id % gnpus + fnpu`。 | 0 |
| `-fpe <int>` | 首个 PE ID。保留为与 shmem_perftest 参数兼容，当前不参与 rank 或 device 计算。 | 0 |
| `-t`/`--test-type <put\|get\|none>` | 可选校验项；若指定，必须与源码编译期 `OP_TYPE` 一致，否则二进制报错。 | - |
| `-d`/`--datatype <type>` | 可选校验项；类型名会映射到 bit 位宽，必须与源码编译期 `DATA_SIZE` 一致。 | - |
| `-b`/`--block-size` | 每个 PE 使用的 Core（Block）数量。 | 32 |
| `--block-range <min> <max>` | Core（Block）数量扫描范围，每个核数各产出统计结果。 | 32 32 |
| `--block-list <b1,b2,...>` | 以逗号分隔显式指定要测试的核数（如 `2,4,8`）。指定后优先于 `-b`/`--block-range`。 | - |
| `--loop-count` | 正式采样的循环次数。 | 1000 |
| `-e`/`--exponent <exp>` | 单次传输数据量的指数（单值），取值为 2 的指数（例如 `10` 表示 $2^{10} = 1024$ 字节）。 | - |
| `--exponent-range <min> <max>` | 单次传输数据量的指数范围。 | 3 20 |
| `--ub-size` | 每个 Core 使用的 Unified Buffer 大小，单位 KB。**仅在 SIMD 模式下生效；默认的 SIMT 模式不使用该参数。** | 16 |

> 本测试为固定的两卡（Active PE0 / Passive PE1）模型，启动进程数与程序内 PE 数均固定为 2。`-pes` 和 `-gnpus` 保留为与 shmem_perftest 参数兼容，但传入非 2 的值会报错。
> 测试会从 `--exponent-range` 的 min 到 max 逐个指数遍历单次传输数据量（即 $2^{min}, 2^{min+1}, \dots, 2^{max}$ 字节），并在 `--block-range`（或 `--block-list`）指定的核数集合上逐个核数遍历，每个（核数, 数据量）组合各产出一行统计结果。

例如，测试 `4` 个 Core 在传输 $2^8$ 到 $2^{12}$ 字节数据时的性能表现：

```bash
bash run.sh -b 4 --exponent-range 8 12
```

### 性能输出说明

测试正常结束后，Active PE（PE 0）会在示例目录下的 `output/` 子目录中输出一个 `.csv` 性能统计文件，文件名格式为：

```bash
[DATA_SIZE]_[blocks]_[OpType]_[VfType]_[minExp]-[maxExp]_l[loop_count]_t[THREAD_COUNT]_.csv
```

其中 `[blocks]` 段反映本次实际测试的核数：连续区间（如 `--block-range 1 4`）记为 `1-4`；通过 `--block-list` 指定的离散核数（如 `2,4,8`）按测试顺序以 `_` 连接，记为 `2_4_8`。

`.csv` 文件中各列含义如下：

| 列名 | 说明 |
| --- | --- |
| `DataSize/B` | 单次 RMA 通信传输的数据量（字节），对应本行采样的 $2^{exp}$ 取值。 |
| `Npus` | 参与测试的 PE 数量，两卡测试下为 2。 |
| `Blocks` | 参与通信的 Core（Block）数量，即 `-b`/`--block-size`。 |
| `UBsize/KB` | 每个 Core 使用的 Unified Buffer 大小（KB），即 `--ub-size`。 |
| `Bandwidth/GB/s (1000)` | 本组参数测得的跨卡平均传输带宽，按十进制单位换算（除以 $1000^3$）。`none` 操作下恒为 0。 |
| `Bandwidth/GiB/s (1024)` | 同一带宽按二进制单位换算（除以 $1024^3$）。`none` 操作下恒为 0。 |
| `CoreMaxTime/us` | 多个 Core 中，单次操作平均耗时最长的那个 Core 的时延（微秒），即带宽计算所用的时间。 |
| `SingleCoreTime/us` | 各 Core 单次操作的平均时延（微秒），由该 Core `loops` 次传输的总时长除以 `loops` 得到，每个 Core 一列。 |
