<!-- 中文 / Chinese -->
# shmem算子接入torch样例

该目录提供了shmem部分算子接入torch后的使用效果，仅做展示使用，不建议生产环境使用！

Torch 扩展与对应 C++ 算子共用同一套 kernel 实现，**运行约束、支持的数据类型与 PE 数限制与 C++ 样例一致**。使用前请先查阅下表中的 C++ 样例文档。

## 样例脚本与算子对应关系

| 脚本 | Torch 算子类 | 对应 C++ 样例文档 |
|------|-------------|-------------------|
| `allgather.py` | `torch.classes.ShmemOps.AllGather` | [examples/allgather/README.md](../../allgather/README.md) |
| `kv_shuffle.py` | `torch.classes.ShmemOps.KVShuffle` | [examples/kv_shuffle/README.md](../../kv_shuffle/README.md) |

### allgather.py 运行约束

与 [allgather C++ 样例](../../allgather/README.md) 一致，关键限制如下：

- **PEs**：仅支持 `[2, 4, 8]`；
- **dtype（算子支持）**：`torch.int32`、`torch.float16`、`torch.bfloat16`；样例脚本固定使用 `float16`，不提供数据类型切换选项

### kv_shuffle.py 运行约束

与 [kv_shuffle C++ 样例](../../kv_shuffle/README.md) 一致，关键限制如下：

- **PEs**：由 `--pes` 指定，不超过可用 device 数量即可，无其它额外限制；样例默认 8 卡
- **dtype（算子支持）**：KV cache 为 `torch.int8`；`global_shuffle_tensor`、`src_block_tensor`、`dst_block_tensor` 为 `torch.int64`；样例脚本固定上述类型，不提供数据类型切换选项

## 编译运行

在shmem根目录执行以下命令：

```sh
# 编译example算子用例及其torch扩展
# A2/A3 平台
bash scripts/build.sh -python_example
# Ascend950 平台
bash scripts/build.sh -soc_type Ascend950 -python_example
source install/set_env.sh
cd examples/python_extension/torch_test
python xxx.py # 默认拉起八卡用例
python xxx.py --pes 2 # --pes可以指定卡数，拉起两卡用例
```

注：xxx表示文件名，需要根据实际情况替换。

## 参数说明

### --tool 参数

- `--tool 0`：直接运行，不使用性能分析工具（默认值）
- `--tool 1`：使用 msprof 性能分析工具

**注意**：内存检测工具（如 mssanitizer）和性能采集工具（如 msprof）不能同时使用！

### --pes 参数

- `--pes <N>`：指定使用的卡数（PE 数）
- **限制与对应 C++ 算子一致**，使用前请查阅上文「样例脚本与算子对应关系」及链接的 C++ 样例文档

## 内存检测工具使用说明

如需使用内存检测工具（mssanitizer），需要：

1. **编译时添加编译选项**：

   ```sh
   # A2/A3 平台
   bash scripts/build.sh -python_example -mssanitizer
   # Ascend950 平台
   bash scripts/build.sh -soc_type Ascend950 -python_example -mssanitizer
   ```

2. **运行时使用 mssanitizer 拉起**：

   ```sh
   mssanitizer -- python xxx.py --pes 2
   ```

   注意：加完编译选项后，不能直接用 python xx.py 拉起，必须使用 mssanitizer -- python xx.py 方式运行！

---

<!-- English -->
# Example of SHMEM Operator Integration with PyTorch
This directory provides an example of integrating selected SHMEM operators with PyTorch. This example is intended for demonstration purposes only and are not recommended for production environments.
## Build and Run
Run the following commands in the root directory of SHMEM:
```sh
# Build example operators and their PyTorch extensions.
bash scripts/build.sh -python_example
source install/set_env.sh
cd examples/python_extension/torch_test
python xxx.py # Run the 8-device example by default.
python xxx.py --pes 2 # Run the 2-device example (--pes specifies the number of devices).
```
Note: Replace xxx with an actual file name.
## Parameter Description
### --tool
- `--tool 0`: Run directly without using the performance profiling tool. (This is the default value.)
- `--tool 1`: The msprof profiling tool is used.

**Note**: A memory check tool (for example, msSanitizer) and a profiling tool (for example, msprof) cannot be used simultaneously.

### --pes
- `--pes <N>`: the number of devices to be used. For example, `--pes 2` indicates that two devices are to be used.

## Memory Check Tool Usage
To use the memory check tool (msSanitizer), perform the following operations:
1. **Add build options during the build**:
   ```sh
   bash scripts/build.sh -python_example -mssanitizer
   ```

2. **Run the example using msSanitizer**:
   ```sh
   mssanitizer -- python xxx.py --pes 2
   ```
   Note: After adding the build options, you cannot directly run the example via `python xx.py`. You must run it via `mssanitizer -- python xx.py`.
