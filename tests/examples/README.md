<!-- 中文 / Chinese -->
# 算子泛化性测试框架说明

<!-- English -->
# Operator Generalization Test Framework

<!-- 中文 / Chinese -->
## 1. 简介

本测试框架旨在为 [SHMEM](https://gitcode.com/cann/shmem) 系列内核提供一个基于 `pytest` 的自动化、可扩展的泛化精度测试环境。框架的核心是通过随机生成覆盖各种数据类型、张量形状和数值分布的测试用例，调用C++内核执行计算，并与 `numpy` 在CPU上计算的真值进行对比，以验证内核的正确性和精度。

<!-- English -->
## 1. Overview
This test framework is designed to provide an automated and extensible generalization precision testing environment for the [SHMEM](https://gitcode.com/cann/shmem) kernel series, built on `pytest`. The framework randomly generates test cases covering diverse data types, tensor shapes, and value distributions. These test cases are executed by the C++ kernels, and the results are compared against ground truth values computed with `NumPy` on the CPU. This ensures rigorous validation of kernel correctness and numerical precision.

<!-- 中文 / Chinese -->
## 2. 框架结构

```text
tests/examples/
├── config.py # 全局测试配置文件
├── utils.py # 通用工具函数（动态误差计算等）
├── np_uniform_generator.py # 均匀分布随机数生成器
├── np_normal_generator.py # 正态分布随机数生成器
├── <kernel_name>/
│ ├── test_<kernel_name>.py # 指定内核的 pytest 测试脚本
│ └── test_data/ # 持久化的测试数据缓存目录
├── readme.md # 本文档
└── ... # 其他内核的测试目录
```

- `config.py`: 定义了整个测试框架的通用配置，如张量形状约束、数值分布参数、数据类型、精度阈值等。
- `utils.py`: 包含通用工具函数，特别是 `get_rtol()` 函数，用于根据数据类型和计算量动态计算误差阈值。
- `np_uniform_generator.py` / `np_normal_generator.py`: 分别用于生成均匀分布和正态分布的测试数据。
- `<kernel_name>/test_<kernel_name>.py`: 针对特定内核的测试脚本。
- `<kernel_name>/test_data/`: 用于存储生成的测试数据，以备调试和分析。

<!-- English -->
## 2. Framework Structure
```
tests/examples/
├── config.py # Global test configuration file
├── utils.py # Common utility functions (for example, dynamic error tolerance calculation)
├── np_uniform_generator.py # Uniform distribution random number generator
├── np_normal_generator.py # Normal distribution random number generator
├── matmul_allreduce/
│ ├── test_fusion_matmul_allreduce.py # Pytest script for the matmul_allreduce kernel
│ └── test_data/ # Persistent cache directory for test data
├── readme.md # This document
└──... # Other kernel-specific test directories
```
-   `config.py`: defines global configuration for the entire test framework, including tensor shape constraints, distribution parameters, data types, and precision thresholds.
-   `utils.py`: contains common utility functions, especially the `get_rtol()` function, which dynamically computes the error tolerance based on the data type and computational workload.
-   `np_uniform_generator.py` or `np_normal_generator.py`: generates test data following uniform or normal distributions.
-   `<kernel_name>/test_<kernel_name>.py`: Pytest scripts for specific kernels.
-   `<kernel_name>/test_data/`: stores generated test data for debugging and analysis.

<!-- 中文 / Chinese -->
## 3. 核心逻辑

测试脚本 (`test_<kernel_name>.py`) 的执行流程如下：

### 3.1. 测试用例生成

- **参数组合与分类**: `get_test_cases` 函数会生成两类测试用例：
    - **正确性 (`correctness`) 测试**: 用于验证基本功能，输入数据范围较小。
    - **数值稳定性 (`stability`) 测试**: 用于验证在极端或特殊数值下的表现，输入数据更复杂。
    该函数根据 `config.py` 中定义的参数（如数据类型、`rank`数量、各类测试用例数）生成一系列 `pytest.param`，用于驱动测试。
- **形状生成**: `generate_shapes` 函数根据 `config.py` 中的约束（如 `SHAPE_DIM_VALUES`, `SHAPE_DIM_RANDOM_RANGE` 等）随机生成合法的张量形状。

### 3.2. 数据与真值生成

- **张量生成**:
    - 对于 **正确性** 测试，使用 `NPUniformGenerator` 在一个预设的、较小的范围内生成均匀分布的张量。
    - 对于 **数值稳定性** 测试，使用 `NPNormalGenerator` 生成符合正态分布的张量，可能包含更具挑战性的数值。
- **真值计算**: 在 CPU 端使用 `numpy` 计算真值。为了保证精度，中间累加过程在 `fp32` 下进行，最后再转换到目标精度。
- **溢出处理**: 在生成真值后，会检查是否存在 `inf` 或 `NaN`。如果存在，会通过 `pytest.skip()` 跳过当前测试用例，避免因输入数据本身导致的问题而标记测试为失败。

### 3.3. 数据持久化

- 为每个测试用例的参数组合计算一个唯一的MD5哈希值。
- 以该哈希值创建一个目录（位于 `test_data/<hash_value>`），并将该测试用例的所有输入张量（`rank_<i>_a.bin`, `rank_<i>_b.bin`）保存到此目录中。这便于后续的复现和调试。

### 3.4. C++ 内核执行

- 使用 Python 的 `multiprocessing` 模块，为每个 `rank` 创建一个独立的进程。
- 每个进程会调用预编译的 C++ 可执行文件 (`EXECUTABLE_PATH`)。
- 测试所需的参数，包括 `rank`、`world_size`、网络地址以及**数据持久化目录的路径**，都会通过命令行参数传递给 C++ 程序。
- C++ 程序从指定的数据目录中读取输入，并将计算结果 `aclshmem_output.bin` 写回到同一目录。

### 3.5. 结果验证

- 所有 C++ 进程执行完毕后，主进程会从数据目录中读取 `aclshmem_output.bin`。
- **动态精度验证**:
    1. **计算操作量**: 根据 `m, k, n, world_size` 计算出总的浮点运算次数。
    2. **获取容忍度**: 调用 `utils.py` 中的 `get_rtol()` 函数，传入数据类型和操作量，获得一个动态计算出的误差容忍度 `err`。
    3. **双重标准验证**:
        - 对真值中绝对值大于等于 `1.0` 的部分，进行 **相对误差** 比较 (`|act - gt| / |gt| <= err`)。
        - 对真值中绝对值小于 `1.0` 的部分，进行 **绝对误差** 比较 (`|act - gt| <= err`)。

<!-- English -->
## 3. Core Logic

The execution process of the test script (`test_fusion_matmul_allreduce.py`) is as follows:

### 3.1. Test Case Generation
-   **Parameter combinations and classification**: The `get_test_cases` function generates two categories of test cases:
    -   **Correctness tests**: Validate basic functionality with relatively small input ranges.
    -   **Numerical stability tests**: Validate behavior under extreme or special values with more complex inputs.
    A series of `pytest.param` objects are generated based on parameters defined in `config.py` (for example, data types, `rank` count, and the number of test cases per category). They are used for driver tests.
-   **Shape generation**: The `generate_shapes` function randomly generates valid tensor shapes based on the constraints (such as `SHAPE_DIM_VALUES` and `SHAPE_DIM_RANDOM_RANGE`) in `config.py`.

### 3.2. Data and Ground Truth Generation
-   **Tensor generation**:
    -   For **correctness** tests, use `NPUniformGenerator` to generate uniformly distributed tensors within a predefined small range.
    -   For **numerical stability** tests, use `NPNormalGenerator` to generate tensors following a normal distribution, potentially containing more challenging values.
-   **Ground truth computation**: `NumPy` is used on the CPU to compute ground truth values. Intermediate accumulation is performed in `fp32` for precision and finally converted to the target precision.
-   **Overflow handling**: After ground truth generation, the framework checks for `inf` or `NaN`. If detected, the test case is skipped via `pytest.skip()` to avoid false failures caused by invalid input data.

### 3.3. Data Persistence
-   Each test case's parameter combination is hashed into a unique MD5 value.
-   A directory (located in `test_data/<hash_value>`) is created using the hash value, storing all input tensors (`rank_<i>_a.bin` and `rank_<i>_b.bin`) of the test case. This facilitates subsequent reproduction and debugging.

### 3.4. C++ Kernel Execution
-   The `multiprocessing` module of Python spawns a separate process for each `rank`.
-   Each process calls a precompiled C++ executable file (`EXECUTABLE_PATH`).
-   The parameters required for the test, including the `rank`, `world_size`, network address, and **the data persistence directory**, are passed to the C++ program through command-line arguments.
-   The C++ program reads the input from the specified data directory and writes the computation result `aclshmem_output.bin` back to the same directory as the input data.

### 3.5. Result Verification
-   After all C++ processes complete, the main process reads `aclshmem_output.bin` from the data directory.
-   **Dynamic precision validation**:
    1.  **Compute the operation count**: Compute the total number of floating-point operations (FLOPs) based on the `m, k, n, and world_size`.
    2.  **Obtain the tolerance**: Call the `get_rtol()` function in `utils.py`, passing the data type and operation count to obtain a dynamically computed error tolerance `err`.
    3.  **Verify the result with dual standards**:
        -   For ground truth values with absolute magnitude ≥ `1.0`, perform a **relative error** check: `|act - gt| / |gt| <= err`.
        -   For ground truth values with absolute magnitude < `1.0`, perform an **absolute error** check: `|act - gt| <= err`.

<!-- 中文 / Chinese -->
## 4. 如何运行测试

1. **编译核函数**: 确保目标核函数的 C++ 可执行文件已经编译。例如，对于 `<kernel_name>`，需要先执行：

    ```bash
    bash examples/<kernel_name>/scripts/build.sh
    ```

2. **设置可执行文件路径**: 在 `test_<kernel_name>.py` 脚本的顶部，确认 `EXECUTABLE_PATH` 变量指向了正确的 C++ 可执行文件路径。
3. **运行 Pytest**: 在项目根目录下，直接运行 `pytest` 命令。

    ```bash
    export LD_LIBRARY_PATH=<path_to_aclshmem_lib>:$LD_LIBRARY_PATH
    pytest -sv tests/examples/<kernel_name>/
    ```

    *请将 `<path_to_...>` 替换为实际的库路径。*

4. **输出示例** <br/>
![本地图片](./assets/test_mm_ar_output_example.png)
<br/>说明：带“correctness"的用例是正确性测试用例，带”stability"是用例是数值稳定性测试用例。

<!-- English -->
## 4. Running a Test
1.  **Build the kernel function**: Ensure that the C++ executable file of the target kernel function has been built. For example, for `matmul_allreduce`, run the following command first:
    ```bash
    bash examples/matmul_allreduce/scripts/build.sh
    ```
2.  **Set the executable file path**: At the top of the `test_fusion_matmul_allreduce.py` script, ensure that the `EXECUTABLE_PATH` variable points to the correct C++ executable file path.
3.  **Run pytest**: In the root directory of the project, run the `pytest` command.
    ```bash
    export LD_LIBRARY_PATH=<path_to_aclshmem_lib>:$LD_LIBRARY_PATH
    pytest -sv tests/examples/matmul_allreduce/
    ```
    *Replace `<path_to_...>` with the actual library path.*

4. **Output example**<br>
![](./assets/test_mm_ar_output_example.png)
<br>Note: Cases with `correctness` are correctness test cases, and cases with `stability` are numerical stability test cases.

<!-- 中文 / Chinese -->
## 5. 如何扩展

要为新的内核（例如 `allgather`）添加测试，可以遵循以下步骤：

1. 在 `tests/examples/` 目录下创建一个新的子目录，例如 `allgather`。
2. 在该目录中创建一个新的测试脚本，例如 `test_allgather.py`。
3. 参考 `test_<kernel_name>.py` 的结构，实现新内核的测试逻辑：
    - 实现数据生成逻辑（可复用 `NP*Generator` 或创建新的生成器）和真值计算。
    - 实现调用 C++ 内核的辅助函数。
    - 实现结果验证逻辑，可复用 `get_rtol` 和双重标准验证方法。
4. 如果需要新的通用配置，可以将其添加到 `tests/examples/config.py` 中。

<!-- English -->
## 5. Extension
To add a test for a new kernel (for example, `allgather`), perform the following steps:
1.  Create a new subdirectory, for example, `allgather`, in the `tests/examples/`.
2.  Create a new test script in this directory, for example, `test_allgather.py`.
3.  Implement the test logic of the new kernel by referring to the structure of `test_matmul_allreduce.py`.
    -   Implement the data generation logic (by reusing `NP*Generator` or creating a new generator) and ground truth computation.
    -   Call the auxiliary function of the C++ kernel.
    -   Implement the result verification logic. You can reuse the `get_rtol` and dual-standard verification method.
4.  If a new general configuration is required, add it to `tests/examples/config.py`.
