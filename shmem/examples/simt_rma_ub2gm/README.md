# 样例介绍

本样例旨在展示 SIMD 与 SIMT 混合编译模式下，SIMT 远程内存访问（RMA）接口使用 UB 作为中转缓冲区进行数据搬运的典型方法。样例代码通过 `__simt_vf__` 函数在 Device 侧申请 UB 数组，并调用 SIMT RMA NBI 接口在 GM 与 UB 之间完成数据传输。

本样例主要演示以下接口形式：

```cpp
simt::aclshmem_int32_get_nbi(__ubuf__ int32_t *dst, __gm__ int32_t *src, uint32_t elem_size, int32_t pe);
simt::aclshmem_int32_put_nbi(__gm__ int32_t *dst, __ubuf__ int32_t *src, uint32_t elem_size, int32_t pe);
```

其中：

| 参数 | 说明 |
| --- | --- |
| `dst` | 目标地址，`get_nbi` 中为 UB 地址，`put_nbi` 中为 GM 地址 |
| `src` | 源地址，`get_nbi` 中为 GM 地址，`put_nbi` 中为 UB 地址 |
| `elem_size` | 传输的 `int32_t` 元素个数 |
| `pe` | 目标或源 PE 编号 |

## 样例执行流程

本样例通过以下流程演示 UB 到 GM 的 RMA 数据路径：

1. **环境初始化**：每个 PE 初始化 3 块大小相同的对称内存。其中，`origin` 数据初始化为 `[my_pe + 0, ..., my_pe + size - 1]`，`res_prev` 和 `res_next` 初始化为 `-1`。
2. **本地 GM 到 UB**：每个 PE 使用 `aclshmem_int32_get_nbi` 将自身 `origin` 中的数据读取到 UB 缓冲区。
3. **UB 到远端 GM**：每个 PE 使用 `aclshmem_int32_put_nbi` 将 UB 缓冲区中的数据写入逻辑上属于下一个 PE 的 `res_next`。
4. **远端 GM 到 UB**：每个 PE 使用 `aclshmem_int32_get_nbi` 将逻辑上属于上一个 PE 的 `origin` 数据读取到 UB 缓冲区。
5. **UB 到本地 GM**：每个 PE 使用 `aclshmem_int32_put_nbi` 将 UB 缓冲区中的数据写入自身 `res_prev`。
6. **结果校验**：通信操作完成后，各 PE 将数据拷贝回 Host 并自动校验传输结果。

## 支持的设备

- Ascend950

## 目录结构

```text
examples/simt_rma_ub2gm/
├── CMakeLists.txt
├── README.md
├── main.cpp
└── run.sh
```

## 使用方式

1. **编译项目**

   在 `shmem/` 根目录下执行编译脚本：

   ```bash
   bash scripts/build.sh -examples -enable_simt -soc_type Ascend950
   ```

2. **运行 simt_rma_ub2gm 示例程序**

   进入示例目录并执行运行脚本：

   ```bash
   cd examples/simt_rma_ub2gm
   bash run.sh
   ```

   `run.sh` 默认启动 2 个独立进程，每个进程对应一个 PE，并使用 `build/bin/simt_rma_ub2gm` 执行样例。

3. **查看结果**

   样例运行结束后会打印各 PE 的 `origin`、`res_prev`、`res_next` 数据摘要。若校验通过，会输出类似如下日志：

   ```text
   [SUCCESS] PE 0: Verification passed for RMA transfers.
   [SUCCESS] PE 1: Verification passed for RMA transfers.
   ```
