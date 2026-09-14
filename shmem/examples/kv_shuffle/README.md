<!-- 中文 / Chinese -->
# 样例介绍

## 使用方式

1. **编译项目**
   在 `shmem/` 根目录下执行编译脚本：
   - A2/A3 平台:

   ```bash
   bash scripts/build.sh -examples
   ```

   - Ascend950 平台:

   ```bash
   bash scripts/build.sh -soc_type Ascend950 -examples
   ```

2. **运行KV_Shuffle示例程序**
   进入示例目录并执行运行脚本：

   ```bash
   cd examples/kv_shuffle
   bash scripts/run.sh [pe_size]
   ```

   **参数说明**：
   - `pe_size`：指定算子运行的pe个数。
   - 示例：使用第0和第1个NPU设备运行2卡kv_shuffle示例：

       ```bash
       bash scripts/run.sh 2
       ```

## 算子介绍

KV Shuffle 算子核心功能是实现 KV Cache（键值缓存）的跨设备 / 跨 PE 数据重排与远程拷贝，适配大模型训练 / 推理中 KV Cache 的分布式调度需求。

在大模型分布式训练 / 推理场景中，KV Cache 会按 Block（块）管理，不同计算 PE 之间需要根据调度策略（如 shuffle table）对 KV Cache 的 Block 进行重排、迁移，本算子为该场景提供高效 KV Block 跨 PE 拷贝与重映射，相比传统主机侧调度，大幅降低 KV Cache 迁移的延迟和带宽开销。

### cpp接口

```cpp
class KVShuffleOps {
public:
    // 默认构造函数
    KVShuffleOps(uint32_t block_dims, void* stream);

    ~KVShuffleOps();

    // 接受 Tensor 的函数
    void compute(
        uint8_t* k_cache,
        uint8_t* v_cache,
        uint8_t* global_shuffle_table,
        uint8_t* src_block_table,
        uint8_t* dst_block_table,
        int64_t block_nums,
        int64_t kv_head_num, int64_t page_size, int64_t head_dim);
private:
    void* sync_ptr_;
    int32_t count_;
    uint32_t block_dims_;
    void* stream_;
    uint64_t fftsAddr_;
};
```

**接口参数说明**

| 参数名      | 输入/输出 | 描述|
| :------------ | :---------- | :-------- |
| `uint8_t*k_cache`| 输入/输出      | 指向键缓存全局内存的指针，存储需要进行shuffle操作的键数据块，连续内存，按块组织，每个块的大小为`kv_head_num * page_size * head_dim * sizeof(data_type)` |
| `uint8_t* v_cache` | 输入/输出      | 指向值缓存全局内存的指针，存储需要进行shuffle操作的值数据块，与k_cache相同的连续内存布局 |
| `uint8_t* global_shuffle_table` | 输入      | 全局shuffle表，存储每个进程的配对信息和操作类型，实际存储int64_t类型数据，内存布局 ：数组结构，每个PE对应2个int64_t条目：`[pair_rank_0, operation_0, pair_rank_1, operation_1, ..., pair_rank_n, operation_n]` 数据限制 ：大小必须为`2 * n_pes * sizeof(int64_t)`，其中`n_pes`是进程总数，operation只能是0或1（0表示发送，1表示接收）配对关系必须是双向的（A的`pair_rank`是B，则B的`pair_rank`必须是A）|
| `uint8_t* src_block_table` | 输入      | 源块索引表，指示每个shuffle操作的源块ID，实际存储int64_t类型数据，一维数组，长度为`block_nums`，每个元素的值必须是有效的块`ID(0 ≤ src_block_id < block_nums)` |
| `uint8_t* dst_block_table` | 输入      |目标块索引表，指示每个shuffle操作的目标块ID，实际存储int64_t类型数据，每个元素的值必须是有效的块`ID(0 ≤ dst_block_id < block_nums)`|
| `int64_t block_nums` | 输入      | 需要进行shuffle操作的块数量 |
| `int64_t kv_head_num` | 输入      |键值数据的头数量 |
| `int64_t page_size` | 输入      | KV缓存中每个页面的大小|
| `int64_t head_dim` | 输入      | 每个头的维度 |

### torch接口

```py
# 创建算子
kv_shuffle = torch.classes.ShmemOps.KVShuffle()
# 计算
kv_shuffle.compute(global_shuffle_tensor, aclshmem_k_cache_tensor,
                                aclshmem_v_cache_tensor, src_block_tensor, dst_block_tensor)
```

**接口参数说明**

#### 1. global_shuffle_tensor

- 含义 ：全局shuffle表，存储每个进程的配对信息和操作类型
- 数据类型 ：PyTorch张量， torch.int64 类型
- 形状 ：二维数组，形状为 [n_pes, 2] ，其中n_pes是进程总数
- 内容结构 ：

  ```py
  [
    [pair_rank_0, operation_0],
    [pair_rank_1, operation_1],
    ...,
    [pair_rank_n, operation_n]
  ]
  ```

- 数据限制 ：
  - 必须在NPU设备上（使用 .npu() 方法转换）
  - operation 只能是0或1（0表示发送，1表示接收）
  - 配对关系必须是双向的（A的pair_rank是B，则B的pair_rank必须是A）
  - 数据类型必须是 int64

#### 2. aclshmem_k_cache_tensor

- 含义 ：指向键缓存全局内存的张量，存储需要进行shuffle操作的键数据块
- 数据类型 ：PyTorch张量，torch.int8 类型
- 形状 ：四维数组，形状为 [block_nums, kv_head_num, page_size, head_dim]
- 数据限制 ：
  - 通过 aclshmem_common.malloc_like() 创建的ACL SHMEM共享内存张量
  - 维度顺序必须严格为 [块数, 头数, 页大小, 头维度]
  - 块数必须与 block_nums 参数匹配

#### 3. aclshmem_v_cache_tensor

- 含义 ：指向值缓存全局内存的张量，存储需要进行shuffle操作的值数据块
- 数据类型 ：PyTorch张量，与 aclshmem_k_cache_tensor 相同
- 形状 ：四维数组，形状与 aclshmem_k_cache_tensor 相同： [block_nums, kv_head_num, page_size, head_dim]
- 数据限制 ：
  - 通过 aclshmem_common.malloc_like() 创建的ACL SHMEM共享内存张量
  - 在NPU设备上
  - 数据类型必须与 aclshmem_k_cache_tensor 一致
  - 形状必须与 aclshmem_k_cache_tensor 完全匹配

#### 4. src_block_tensor

- 含义 ：源块索引表，指示每个shuffle操作的源块ID
- 数据类型 ：PyTorch张量， torch.int64 类型
- 形状 ：一维数组，长度为 block_nums
- 数据限制 ：
  - 必须在NPU设备上
  - 数据类型必须是 int64
  - 每个元素的值必须是有效的块ID（0 ≤ src_block_id < block_nums）
  - 数组长度必须与当前进程需要处理的块数匹配

#### 5. dst_block_tensor

- 含义 ：目标块索引表，指示每个shuffle操作的目标块ID
- 数据类型 ：PyTorch张量， torch.int64 类型
- 形状 ：一维数组，长度为 block_nums
- 数据限制 ：
  - 必须在NPU设备上
  - 数据类型必须是 int64
  - 每个元素的值必须是有效的块ID（0 ≤ dst_block_id < block_nums）
  - 数组长度必须与 src_block_tensor 完全匹配

#### 关键参数推导

在PyTorch扩展的C++实现中，以下参数从输入张量中推导出来：

- block_nums ：从 dst_block_tensor.size(0) 获取，表示需要处理的块数量
- kv_head_num ：从 KeyCache.size(1) 获取，表示键值数据的头数量
- page_size ：从 KeyCache.size(2) 获取，表示每个页面的token数量
- head_dim ：从 KeyCache.size(3) 获取，表示每个头的维度

### KVShuffle算子数据流转效果说明

本文档通过具体示例展示KVShuffle算子的数据流转过程，包括初始数据状态、传输策略计算、块表生成和最终数据变换结果。

#### 1. 示例配置

为了清晰展示数据流转，我们使用以下简化配置：

| 参数 | 值 | 说明 |
|------|-----|------|
| RANKS | 2 | 进程总数 |
| INIT_BATCH | 2 | 每个进程的初始批次数 |
| PAGE_SIZE | 4 | 页面大小（每个页面包含的token数量） |
| KV_HEAD_NUM | 1 | 头数量（简化为1便于展示） |
| HEAD_DIM | 2 | 头维度（简化为2便于展示） |
| MAX_SEQLEN | 8 | 最大序列长度 |

#### 2. Batch Token与KV缓存的关系

在理解数据流转之前，需要先明确Batch Token与KV缓存之间的核心关系：

##### 2.1 基本概念

| 概念 | 含义 |
|------|------|
| **Batch Token** | 一个batch中包含的token数量，即序列长度（seqlen） |
| **KV缓存** | 存储键值对的缓存结构，用于注意力机制的高效计算 |
| **Page** | KV缓存的基本存储单位，每个page包含固定数量的token（PAGE_SIZE） |
| **Block** | 一个batch在KV缓存中占用的连续pages集合 |

##### 2.2 关系公式

1. **块数计算**：一个batch需要的块数 = batch token数 ÷ 页面大小 + 1（向上取整）

   ```python
   block_num = seqlen // PAGE_SIZE + 1
   ```

2. **KV缓存总大小**：

   ```python
   total_cache_size = max_block_nums × kv_head_num × page_size × head_dim × data_type_size
   ```

##### 2.3 映射关系示例

以进程0的Batch 0为例：

- Batch Token数：6
- 页面大小（PAGE_SIZE）：4
- 需要的块数：6 ÷ 4 + 1 = 2块
- 这2块对应KV缓存中的Block 0和Block 1

##### 2.4 直观理解

```text
Batch Token (seqlen=6) → 映射到 → KV缓存的2个块
┌─────────────────┐     ┌────────────────────────────────────────────┐
│ Token 0-5       │     │ Block 0 (PAGE_SIZE=4): 存储Token 0-3       │
│ (6个token)      │     │ Block 1 (PAGE_SIZE=4): 存储Token 4-5       │
└─────────────────┘     └────────────────────────────────────────────┘
```

这种映射关系确保了即使不同batch的token长度不同，也能在KV缓存中高效存储和访问。

#### 3. 初始数据状态

##### 3.1 进程0的初始数据

**KV缓存形状**：(block_num, kv_head_num, page_size, head_dim) = (4, 1, 4, 2)

**K缓存数据**：

```text
# Block 0
[[[1.1, 1.2], [1.3, 1.4], [1.5, 1.6], [1.7, 1.8]]]

# Block 1
[[[2.1, 2.2], [2.3, 2.4], [2.5, 2.6], [2.7, 2.8]]]

# Block 2
[[[3.1, 3.2], [3.3, 3.4], [3.5, 3.6], [3.7, 3.8]]]

# Block 3
[[[4.1, 4.2], [4.3, 4.4], [4.5, 4.6], [4.7, 4.8]]]
```

**V缓存数据**：

```text
# Block 0
[[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]]

# Block 1
[[[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]]]

# Block 2
[[[1.7, 1.8], [1.9, 2.0], [2.1, 2.2], [2.3, 2.4]]]

# Block 3
[[[2.5, 2.6], [2.7, 2.8], [2.9, 3.0], [3.1, 3.2]]]
```

##### 3.2 进程1的初始数据

**KV缓存形状**：(block_num, kv_head_num, page_size, head_dim) = (2, 1, 4, 2)

**K缓存数据**：

```text
# Block 0
[[[5.1, 5.2], [5.3, 5.4], [5.5, 5.6], [5.7, 5.8]]]

# Block 1
[[[6.1, 6.2], [6.3, 6.4], [6.5, 6.6], [6.7, 6.8]]]
```

**V缓存数据**：

```text
# Block 0
[[[3.3, 3.4], [3.5, 3.6], [3.7, 3.8], [3.9, 4.0]]]

# Block 1
[[[4.1, 4.2], [4.3, 4.4], [4.5, 4.6], [4.7, 4.8]]]
```

#### 3. 负载均衡与传输策略计算

##### 3.1 Batch token长度

假设生成的batch token长度如下：

| 进程 | Batch 0 | Batch 1 | 总token数 |
|------|---------|---------|----------|
| 0 | 6 | 7 | 13 |
| 1 | 3 | 3 | 6 |

##### 3.2 块数计算

每个batch的块数计算公式：`block_num = seqlen // PAGE_SIZE + 1`

| 进程 | Batch 0 | Batch 1 | 总块数 |
|------|---------|---------|--------|
| 0 | 2 (6//4+1) | 2 (7//4+1) | 4 |
| 1 | 1 (3//4+1) | 1 (3//4+1) | 2 |

**注意**：由于 token 是按块管理的，最后一个块可能没有填满（例如 Batch 0 的 Block 1 只有 2 个 token）。

##### 3.3 Batch块映射

**进程0的batch_blocks_list**：

```python
[  # batch_blocks_list[0]
    (0, [0, 1]),  # Batch 0使用块0和块1
    (1, [2, 3])   # Batch 1使用块2和块3
]
```

**进程1的batch_blocks_list**：

```python
[  # batch_blocks_list[1]
    (0, [0]),     # Batch 0使用块0
    (1, [1])      # Batch 1使用块1
]
```

##### 3.4 负载均衡计算

- 平均token数：`(13 + 6) / 2 = 9.5`
- 进程0需要传输的token数：`13 - 9.5 = 3.5`（取整为3）
- 进程1需要接收的token数：`9.5 - 6 = 3.5`（取整为3）

##### 3.5 传输batch选择

由于token是按块管理的，选择传输Batch 0（6个token）来尽可能接近理想负载：

**transfer_tokens_list**：

```python
[  # transfer_tokens_list
    (6, [0]),  # 进程0传输Batch 0的6个token
    (-1, [])   # 进程1不需要传输
]
```

#### 4. 块表生成

##### 4.1 src_block_table

进程0需要传输Batch 0对应的块ID [0, 1]：

**src_block_table**：

```python
[  # src_block_table
    [0, 1],  # 进程0的源块表
    []       # 进程1的源块表
]
```

##### 4.2 dst_block_table

进程1当前使用了2个块（0和1），因此目标块ID从2开始：

**dst_block_table**：

```python
[  # dst_block_table
    [2, 3],  # 进程0的目标块表（传输到进程1的块2和3）
    []       # 进程1的目标块表
]
```

##### 4.3 配对关系

**pair_list**：

```python
[  # pair_list
    [1, 0],  # 进程0与进程1配对，角色为发送方(0)
    [0, 1]   # 进程1与进程0配对，角色为接收方(1)
]
```

#### 5. KVShuffle数据变换

##### 5.1 数据传输过程

| 源进程 | 源块ID | 目标进程 | 目标块ID | 传输的数据 |
|--------|--------|----------|----------|------------|
| 0 | 0 | 1 | 2 | 进程0的K块0、V块0 |
| 0 | 1 | 1 | 3 | 进程0的K块1、V块1 |

##### 5.2 关于源进程块的清理说明

**为什么进程0的块没有被清理？**

- KVShuffle算子默认执行的是**数据复制**而非数据移动
- 这是因为在分布式训练场景中，源进程可能仍然需要这些数据用于后续的计算或其他batch处理
- 如果应用层确实需要清理源进程的数据，可以在KVShuffle操作完成后，手动释放或标记这些块为可用
- 清理操作通常由应用层根据具体业务逻辑决定，而不是由KVShuffle算子自动执行

##### 5.3 变换后数据状态

###### 进程0的最终数据（不变）

**K缓存**：

```text
# Block 0
[[[1.1, 1.2], [1.3, 1.4], [1.5, 1.6], [1.7, 1.8]]]

# Block 1
[[[2.1, 2.2], [2.3, 2.4], [2.5, 2.6], [2.7, 2.8]]]

# Block 2
[[[3.1, 3.2], [3.3, 3.4], [3.5, 3.6], [3.7, 3.8]]]

# Block 3
[[[4.1, 4.2], [4.3, 4.4], [4.5, 4.6], [4.7, 4.8]]]
```

**V缓存**：

```text
# Block 0
[[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]]

# Block 1
[[[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]]]

# Block 2
[[[1.7, 1.8], [1.9, 2.0], [2.1, 2.2], [2.3, 2.4]]]

# Block 3
[[[2.5, 2.6], [2.7, 2.8], [2.9, 3.0], [3.1, 3.2]]]
```

###### 进程1的最终数据（新增块2和3）

**K缓存**：

```text
# Block 0（原有）
[[[5.1, 5.2], [5.3, 5.4], [5.5, 5.6], [5.7, 5.8]]]

# Block 1（原有）
[[[6.1, 6.2], [6.3, 6.4], [6.5, 6.6], [6.7, 6.8]]]

# Block 2（新增，来自进程0的块0）
[[[1.1, 1.2], [1.3, 1.4], [1.5, 1.6], [1.7, 1.8]]]

# Block 3（新增，来自进程0的块1）
[[[2.1, 2.2], [2.3, 2.4], [2.5, 2.6], [2.7, 2.8]]]
```

**V缓存**：

```text
# Block 0（原有）
[[[3.3, 3.4], [3.5, 3.6], [3.7, 3.8], [3.9, 4.0]]]

# Block 1（原有）
[[[4.1, 4.2], [4.3, 4.4], [4.5, 4.6], [4.7, 4.8]]]

# Block 2（新增，来自进程0的块0）
[[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]]

# Block 3（新增，来自进程0的块1）
[[[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]]]
```

#### 6. 数据验证

##### 6.1 传输前后数据一致性

- 进程1的K块2与进程0的K块0完全相同
- 进程1的K块3与进程0的K块1完全相同
- 进程1的V块2与进程0的V块0完全相同
- 进程1的V块3与进程0的V块1完全相同

##### 6.2 负载均衡效果

传输前后的token分布：

| 进程 | 传输前token数 | 传输后token数 | 均衡度 |
|------|--------------|--------------|--------|
| 0 | 13 | 13 - 6 = 7 | 更接近平均值9.5 |
| 1 | 6 | 6 + 6 = 12 | 更接近平均值9.5 |

#### 7. 数据流转总结

```text
┌─────────────────────────┐     ┌────────────────────────┐
│        进程0初始数据     │     │        进程1初始数据    │
│  K块0: [1.1, 1.2, ...]  │     │  K块0: [5.1, 5.2, ...]  │
│  K块1: [2.1, 2.2, ...]  │     │  K块1: [6.1, 6.2, ...]  │
│  K块2: [3.1, 3.2, ...]  │     │  V块0: [3.3, 3.4, ...]  │
│  K块3: [4.1, 4.2, ...]  │     │  V块1: [4.1, 4.2, ...]  │
│  V块0: [0.1, 0.2, ...]  │     └─────────────────────────┘
│  V块1: [0.9, 1.0, ...]  │               ▲
│  V块2: [1.7, 1.8, ...]  │               │
│  V块3: [2.5, 2.6, ...]  │               │
└────────────┬────────────┘               │
             │                            │
             │ 传输Batch 0的块0和块1       │
             │                            │
             ▼                            │
┌─────────────────────────┐     ┌─────────┴─────────┐
│      生成块表和策略      │     │       数据传输    │
│  src_block_table: [0, 1]│───▶│ K块0 → 进程1的K块2 │
│  dst_block_table: [2, 3]│     │ K块1 → 进程1的K块3 │
│  pair_list: [1, 0]      │     │ V块0 → 进程1的V块2 │
└─────────────────────────┘     │ V块1 → 进程1的V块3 │
                                └───────────────────┘
                                          │
                                          ▼
┌─────────────────────────┐     ┌────────────────────────┐
│      进程0最终数据       │     │      进程1最终数据      │
│  K块0: [1.1, 1.2, ...]  │     │  K块0: [5.1, 5.2, ...]  │
│  K块1: [2.1, 2.2, ...]  │     │  K块1: [6.1, 6.2, ...]  │
│  K块2: [3.1, 3.2, ...]  │     │  K块2: [1.1, 1.2, ...]  │
│  K块3: [4.1, 4.2, ...]  │     │  K块3: [2.1, 2.2, ...]  │
│  V块0: [0.1, 0.2, ...]  │     │  V块0: [3.3, 3.4, ...]  │
│  V块1: [0.9, 1.0, ...]  │     │  V块1: [4.1, 4.2, ...]  │
│  V块2: [1.7, 1.8, ...]  │     │  V块2: [0.1, 0.2, ...]  │
│  V块3: [2.5, 2.6, ...]  │     │  V块3: [0.9, 1.0, ...]  │
└─────────────────────────┘     └─────────────────────────┘
```

#### 8. 关键数据结构示例

##### 8.1 global_shuffle_tensor

```text
[[1, 0],  # 进程0与进程1配对，角色为发送方
 [0, 1]]  # 进程1与进程0配对，角色为接收方
```

##### 8.2 aclshmem_k_cache_tensor（进程0）

```text
# 形状: (4, 1, 4, 2)
[[[[1.1, 1.2], [1.3, 1.4], [1.5, 1.6], [1.7, 1.8]]],  # Block 0
 [[[2.1, 2.2], [2.3, 2.4], [2.5, 2.6], [2.7, 2.8]]],  # Block 1
 [[[3.1, 3.2], [3.3, 3.4], [3.5, 3.6], [3.7, 3.8]]],  # Block 2
 [[[4.1, 4.2], [4.3, 4.4], [4.5, 4.6], [4.7, 4.8]]]]  # Block 3
```

##### 8.3 src_block_tensor（进程0）

```text
[0, 1]  # 要传输的源块ID
```

##### 8.4 dst_block_tensor（进程0）

```text
[2, 3]  # 传输到目标进程的块ID
```

#### 9. 性能指标示例

| 指标 | 值 | 说明 |
|------|-----|------|
| 传输块数 | 2 | 本次传输了2个块 |
| 传输token数 | 6 | 从进程0传输2个块（共8个token存储空间，实际有用6个token：1个满块4个token+1个半块2个token）到进程1 |
| 传输数据量 | 2×1×4×2×2=32字节 | K和V各16字节（float16类型） |
| 负载均衡度 | 从13:6变为7:12 | 更接近理想的9.5:9.5 |

#### 10. 应用场景说明

通过这个具体的数据流示例，我们可以看到KVShuffle算子：

1. **解决了负载不均衡问题**：将数据从负载高的进程传输到负载低的进程
2. **保持了数据完整性**：传输前后的数据内容完全一致
3. **高效利用了内存**：只传输必要的块，避免了不必要的数据移动
4. **支持动态批处理**：可以根据实际batch大小动态调整传输策略

这种数据流转机制特别适合于分布式训练中的KV缓存管理，可以有效提高训练效率和资源利用率。

---

<!-- English -->
### Instructions

1. **Build a project**.
   Run the `build.sh` script in the root (`shmem/`) directory.
   ```bash
   bash scripts/build.sh -examples
   ```

2. **Run the KV_Shuffle sample program**.
   Go to the `examples` directory and run the `run.sh` script.
   ```bash
   cd examples/kv_shuffle
   bash scripts/run.sh [pe_size]
   ```

   - **Parameter description**:
     - `pe_size`: Specifies the number of PEs on which the operator runs.
     - Example: Use NPUs 0 and 1 to run the kv_shuffle sample that needs two devices.
       ```bash
       bash scripts/run.sh 2
       ```
### Operator Introduction
The core function of the KV Shuffle operator is to implement cross-device or cross-PE data reshuffling and remote copying of the KV Cache, adapting to the distributed scheduling requirements of the KV Cache in large model training and inference.

In distributed training and inference scenarios of large models, the KV Cache is managed by block. Blocks in the KV Cache need to be reshuffled and migrated between different compute PEs based on scheduling policies (such as shuffle tables). This operator provides efficient cross-PE copying and remapping of KV blocks, significantly reducing the latency and bandwidth overhead of KV Cache migration compared to traditional host-side scheduling.

#### C++ API

```cpp
class KVShuffleOps {
public:
    // Default constructor
    KVShuffleOps(uint32_t block_dims, void* stream);

    ~KVShuffleOps();

    // Function for receiving tensors
    void compute(
        uint8_t* k_cache,
        uint8_t* v_cache,
        uint8_t* global_shuffle_table,
        uint8_t* src_block_table,
        uint8_t* dst_block_table,
        int64_t block_nums,
        int64_t kv_head_num, int64_t page_size, int64_t head_dim);
private:
    void* sync_ptr_;
    int32_t count_;
    uint32_t block_dims_;
    void* stream_;
    uint64_t fftsAddr_;
};
```

**API Parameters**

| Parameter     | Input/Output| Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| :------------ | :---------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| uint8_t* k_cache| Input/Output     | Pointer to the Key Cache global memory, which stores the key data blocks for the shuffle operation. The memory is contiguous and organized by block. The size of each block is kv_head_num * page_size * head_dim * sizeof(data_type).|
| uint8_t* v_cache | Input/Output     | Pointer to the Value Cache global memory, which stores the value data blocks for the shuffle operation. It shares the same contiguous memory layout as `k_cache`.|
| uint8_t* global_shuffle_table | Input     | Global shuffle table, which stores the pairing information and operation type of each rank. It actually stores data of the `int64_t` type. The memory layout is an array structure, with each PE corresponding to two `int64_t` entries: `[pair_rank_0, operation_0, pair_rank_1, operation_1, ..., pair_rank_n, operation_n]`. Data restrictions: The size must be `2 * n_pes * sizeof(int64_t)`, where `n_pes` is the total number of ranks. The value of `operation` can only be `0` or `1` (`0` for sending and `1` for receiving). The pairing relationship must be bidirectional (if A's `pair_rank` is B, then B's `pair_rank` must be A).|
| uint8_t* src_block_table | Input     | Source block index table, which indicates the source block ID of each shuffle operation. The actual data is of the `int64_t` type and stored in a one-dimensional array with a length of `block_nums`. The value of each element must be a valid block ID (0 ≤ src_block_id < block_nums).|
| uint8_t* dst_block_table | Input     |Destination block index table, which indicates the destination block ID of each shuffle operation. The actual data is of the `int64_t` type. The value of each element must be a valid block ID (0 ≤ dst_block_id < block_nums).|
| int64_t block_nums | Input     | Number of blocks to be shuffled.|
| int64_t kv_head_num | Input     |Number of key-value data heads.|
| int64_t page_size | Input     | Size of each page in the KV Cache.|
| int64_t head_dim | Input     | Dimension of each head.|

#### PyTorch API

```py
# Create an operator.
kv_shuffle = torch.classes.ShmemOps.KVShuffle()
# Computation
kv_shuffle.compute(global_shuffle_tensor, aclshmem_k_cache_tensor,
                                aclshmem_v_cache_tensor, src_block_tensor, dst_block_tensor)
```

**API Parameters**

##### 1. global_shuffle_tensor

- Description: global shuffle table, which stores the pairing information and operation type of each rank.
- Data type: PyTorch tensor of the `torch.int64` type.
- Shape: two-dimensional array with a shape of `[n_pes, 2]`, where `n_pes` is the total number of ranks.
- Content structure:
  ```
  [
    [pair_rank_0, operation_0],
    [pair_rank_1, operation_1],
    ...,
    [pair_rank_n, operation_n]
  ]
  ```
- Data restrictions:
  - The tensor must reside on the NPU device (converted via the `.npu()` method).
  - The value of `operation` must be `0` or `1` (`0` for sending and `1` for receiving).
  - The pairing relationship must be bidirectional (if the `pair_rank` of A is B, the `pair_rank` of B must be A).
  - The data type must be `int64`.

##### 2. aclshmem_k_cache_tensor

- Description: tensor pointing to the Key Cache global memory, which stores the key data blocks to be shuffled.
- Data type: PyTorch tensor of the `torch.int8` type
- Shape: four-dimensional array with a shape of [block_nums, kv_head_num, page_size, head_dim]
- Data restrictions:
  - ACL SHMEM shared memory tensor created using `aclshmem_common.malloc_like()`
  - The dimension sequence must strictly be [number of blocks, number of heads, page size, head dimension].
  - The number of blocks must match the `block_nums` parameter.

##### 3. aclshmem_v_cache_tensor

- Description: tensor pointing to the Value Cache global memory, which stores the value data blocks to be shuffled.
- Data type: PyTorch tensor, same as `aclshmem_k_cache_tensor`
- Shape: four-dimensional array, with a shape identical to `aclshmem_k_cache_tensor`: [block_nums, kv_head_num, page_size, head_dim]
- Data restrictions:
  - ACL SHMEM shared memory tensor created using `aclshmem_common.malloc_like()`
  - On the NPU device
  - The data type must be the same as that of `aclshmem_k_cache_tensor`.
  - The shape must exactly match that of `aclshmem_k_cache_tensor`.

##### 4. src_block_tensor

- Description: source block index table, which indicates the source block ID of each shuffle operation.
- Data type: PyTorch tensor of the `torch.int64` type.
- Shape: one-dimensional array with a length of `block_nums`
- Data restrictions:
  - The tensor must reside on the NPU device.
  - The data type must be `int64`.
  - The value of each element must be a valid block ID (0 ≤ src_block_id < block_nums).
  - The array length must match the number of blocks to be processed by the current rank.

##### 5. dst_block_tensor

- Description: destination block index table, which indicates the destination block ID of each shuffle operation.
- Data type: PyTorch tensor of the `torch.int64` type.
- Shape: one-dimensional array with a length of `block_nums`
- Data restrictions:
  - The tensor must reside on the NPU device.
  - The data type must be `int64`.
  - The value of each element must be a valid block ID (0 ≤ dst_block_id < block_nums).
  - The array length must exactly match that of `src_block_tensor`.

##### Key Parameter Derivation

In the C++ implementation of the PyTorch extension, the following parameters are derived from the input tensor:

- `block_nums`: obtained from `dst_block_tensor.size(0)`, representing the number of blocks to be processed
- `kv_head_num`: obtained from `KeyCache.size(1)`, representing the number of key-value data heads
- `page_size`: obtained from `KeyCache.size(2)`, representing the number of tokens per page
- `head_dim`: obtained from `KeyCache.size(3)`, representing the dimension of each head

### Data Flow Effect of the KVShuffle Operator

This document uses a specific example to demonstrate the data flow process of the KVShuffle operator, including the initial data state, transmission policy computation, block table generation, and final data transformation result.

#### 1. Example Configuration

To clearly demonstrate data flow, we use the following simplified configuration:

| Parameter| Value| Description|
|------|-----|------|
| RANKS | 2 | Total number of ranks|
| INIT_BATCH | 2 | Initial batch size of each rank|
| PAGE_SIZE | 4 | Page size (number of tokens on each page)|
| KV_HEAD_NUM | 1 | Number of heads (simplified to 1 for demonstration)|
| HEAD_DIM | 2 | Head dimension (simplified to 2 for demonstration)|
| MAX_SEQLEN | 8 | Maximum sequence length|

#### 2. Relationship Between Batch Token and KV Cache

Before understanding data flow, it is important to clarify the core relationship between Batch Token and KV Cache.

##### 2.1 Basic Concepts

| Concept| Definition|
|------|------|
| **Batch Token** | Number of tokens contained in a batch, that is, the sequence length (seqlen)|
| **KV Cache**| A cache structure that stores key-value pairs, used for efficient computation of the attention mechanism.|
| **Page** | Basic storage unit of the KV Cache. Each page contains a fixed number of tokens (PAGE_SIZE).|
| **Block** | A set of consecutive pages occupied by a batch in the KV Cache.|

##### 2.2 Relationship Formulas

1. **Block count calculation**: Number of blocks required by a batch = Number of batch tokens/Page size + 1 (rounded up)

   ```python
   block_num = seqlen // PAGE_SIZE + 1
   ```
2. **Total size of the KV Cache**:

   ```python
   total_cache_size = max_block_nums × kv_head_num × page_size × head_dim × data_type_size
   ```

##### 2.3 Mapping Example

Take Batch 0 of Rank 0 as an example:

- Number of batch tokens: 6
- Page size (PAGE_SIZE): 4
- Number of required blocks: 6/4 + 1 = 2
- These two blocks correspond to Block 0 and Block 1 in the KV Cache.

##### 2.4 Intuitive Understanding

```
Batch Token (seqlen=6) → mapped to → 2 blocks in the KV Cache
┌─────────────────┐     ┌────────────────────────────────────────────┐
│ Token 0-5       │     │ Block 0 (PAGE_SIZE=4): Stores Token 0-3       │
│ (6 tokens)      │     │ Block 1 (PAGE_SIZE=4): Stores Token 4-5       │
└─────────────────┘     └────────────────────────────────────────────┘
```

This mapping ensures that tokens can be efficiently stored and accessed in the KV Cache even if the token lengths of different batches are different.

#### 3. Initial Data State

##### 3.1 Initial Data of Rank 0

**KV Cache shape**: (block_num, kv_head_num, page_size, head_dim) = (4, 1, 4, 2)

**K Cache data**:

```
# Block 0
[[[1.1, 1.2], [1.3, 1.4], [1.5, 1.6], [1.7, 1.8]]]

# Block 1
[[[2.1, 2.2], [2.3, 2.4], [2.5, 2.6], [2.7, 2.8]]]

# Block 2
[[[3.1, 3.2], [3.3, 3.4], [3.5, 3.6], [3.7, 3.8]]]

# Block 3
[[[4.1, 4.2], [4.3, 4.4], [4.5, 4.6], [4.7, 4.8]]]
```

**V Cache data**:

```
# Block 0
[[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]]

# Block 1
[[[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]]]

# Block 2
[[[1.7, 1.8], [1.9, 2.0], [2.1, 2.2], [2.3, 2.4]]]

# Block 3
[[[2.5, 2.6], [2.7, 2.8], [2.9, 3.0], [3.1, 3.2]]]
```

##### 3.2 Initial Data of Rank 1

**KV Cache shape**: (block_num, kv_head_num, page_size, head_dim) = (2, 1, 4, 2)

**K Cache data**:

```
# Block 0
[[[5.1, 5.2], [5.3, 5.4], [5.5, 5.6], [5.7, 5.8]]]

# Block 1
[[[6.1, 6.2], [6.3, 6.4], [6.5, 6.6], [6.7, 6.8]]]
```

**V Cache data**:

```
# Block 0
[[[3.3, 3.4], [3.5, 3.6], [3.7, 3.8], [3.9, 4.0]]]

# Block 1
[[[4.1, 4.2], [4.3, 4.4], [4.5, 4.6], [4.7, 4.8]]]
```

#### 3. Load Balancing and Transmission Policy Computation

##### 3.1 Batch Token Length

Assume that the generated batch token lengths are as follows:

| Rank| Batch 0 | Batch 1 | Total Number of Tokens|
|------|---------|---------|----------|
| 0 | 6 | 7 | 13 |
| 1 | 3 | 3 | 6 |

##### 3.2 Calculating the Number of Blocks

Formula for calculating the number of blocks in each batch: `block_num = seqlen // PAGE_SIZE + 1`

| Rank| Batch 0 | Batch 1 | Total Number of Blocks|
|------|---------|---------|--------|
| 0 | 2 (6//4+1) | 2 (7//4+1) | 4 |
| 1 | 1 (3//4+1) | 1 (3//4+1) | 2 |

**Note**: Tokens are managed by block, so the last block may not be fully filled (for example, Block 1 of Batch 0 has only two tokens).

##### 3.3 Batch-to-Block Mapping

**batch_blocks_list of Rank 0**:

```python
[  # batch_blocks_list[0]
    (0, [0, 1]),  # Batch 0 uses Blocks 0 and 1.
    (1, [2, 3])   # Batch 1 uses Blocks 2 and 3.
]
```

**batch_blocks_list of Rank 1**:

```python
[  # batch_blocks_list[1]
    (0, [0]),     # Batch 0 uses Block 0.
    (1, [1])      # Batch 1 uses Block 1.
]
```

##### 3.4 Load Balancing Computation

- Average number of tokens: `(13 + 6) / 2 = 9.5`
- Number of tokens to be transmitted by Rank 0: `13 - 9.5 = 3.5` (rounded down to 3)
- Number of tokens to be received by Rank 1: `9.5 - 6 = 3.5` (rounded down to 3)

##### 3.5 Selection of Batches for Transmission

Since tokens are managed by block, Batch 0 (with 6 tokens) is selected for transmission to approach the ideal load as closely as possible:

**transfer_tokens_list**:

```python
[  # transfer_tokens_list
    (6, [0]),  # Rank 0 transmits 6 tokens of Batch 0.
    (-1, [])   # Rank 1 does not need to transmit.
]
```

#### 4. Block Table Generation

##### 4.1 src_block_table

Rank 0 needs to transmit the block IDs [0, 1] corresponding to Batch 0:

**src_block_table**:

```python
[  # src_block_table
    [0, 1],  # Source block table of Rank 0
    []       # Source block table of Rank 1
]
```

##### 4.2 dst_block_table

Rank 1 currently uses two blocks (0 and 1), so the destination block IDs start from 2:

**dst_block_table**:

```python
[  # dst_block_table
    [2, 3],  # Destination block table of Rank 0 (Blocks 2 and 3 transmitted to Rank 1)
    []       # Destination block table of Rank 1
]
```

##### 4.3 Pairing Relationship

**pair_list**:

```python
[  # pair_list
    [1, 0],  # Rank 0 is paired with Rank 1 and acts as the sender (0).
    [0, 1]   # Rank 1 is paired with Rank 0 and acts as the receiver (1).
]
```

#### 5. KVShuffle Data Transformation

##### 5.1 Data Transmission Process

| Source Rank| Source Block ID| Destination Rank| Destination Block ID| Transmitted Data|
|--------|--------|----------|----------|------------|
| 0 | 0 | 1 | 2 | K Block 0 and V Block 0 of Rank 0|
| 0 | 1 | 1 | 3 | K Block 1 and V Block 1 of Rank 0|

##### 5.2 Notes on Clearing Source Rank Blocks

**Why are the blocks of Rank 0 not cleared?**

- By default, the KVShuffle operator performs **data replication** instead of data movement.
- This is because in distributed training scenarios, the source rank may still need the data for subsequent computation or other batch processing.
- If the application layer indeed needs to clear the data of the source rank, these blocks can be manually released or marked as available after the KVShuffle operation is complete.
- The clearing operation is usually determined by the application layer based on the specific service logic, rather than being automatically performed by the KVShuffle operator.

##### 5.3 Data Status After Transformation

###### Final Data of Rank 0 (Unchanged)

**K Cache**:

```
# Block 0
[[[1.1, 1.2], [1.3, 1.4], [1.5, 1.6], [1.7, 1.8]]]

# Block 1
[[[2.1, 2.2], [2.3, 2.4], [2.5, 2.6], [2.7, 2.8]]]

# Block 2
[[[3.1, 3.2], [3.3, 3.4], [3.5, 3.6], [3.7, 3.8]]]

# Block 3
[[[4.1, 4.2], [4.3, 4.4], [4.5, 4.6], [4.7, 4.8]]]
```

**V Cache**:

```
# Block 0
[[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]]

# Block 1
[[[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]]]

# Block 2
[[[1.7, 1.8], [1.9, 2.0], [2.1, 2.2], [2.3, 2.4]]]

# Block 3
[[[2.5, 2.6], [2.7, 2.8], [2.9, 3.0], [3.1, 3.2]]]
```

###### Final Data of Rank 1 (Blocks 2 and 3 Added)

**K Cache**:

```
# Block 0 (Original)
[[[5.1, 5.2], [5.3, 5.4], [5.5, 5.6], [5.7, 5.8]]]

# Block 1 (Original)
[[[6.1, 6.2], [6.3, 6.4], [6.5, 6.6], [6.7, 6.8]]]

# Block 2 (New, from Block 0 of Rank 0)
[[[1.1, 1.2], [1.3, 1.4], [1.5, 1.6], [1.7, 1.8]]]

# Block 3 (New, from Block 1 of Rank 0)
[[[2.1, 2.2], [2.3, 2.4], [2.5, 2.6], [2.7, 2.8]]]
```

**V Cache**:

```
# Block 0 (Original)
[[[3.3, 3.4], [3.5, 3.6], [3.7, 3.8], [3.9, 4.0]]]

# Block 1 (Original)
[[[4.1, 4.2], [4.3, 4.4], [4.5, 4.6], [4.7, 4.8]]]

# Block 2 (New, from Block 0 of Rank 0)
[[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]]

# Block 3 (New, from Block 1 of Rank 0)
[[[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]]]
```

#### 6. Data Verification

##### 6.1 Data Consistency Before and After Transmission

- K Block 2 of Rank 1 is identical to K Block 0 of Rank 0.
- K Block 3 of Rank 1 is identical to K Block 1 of Rank 0.
- V Block 2 of Rank 1 is identical to V Block 0 of Rank 0.
- V Block 3 of Rank 1 is identical to V Block 1 of Rank 0.

##### 6.2 Load Balancing Effect

Token distribution before and after transmission:

| Rank| Number of Tokens Before Transmission| Number of Tokens After Transmission| Balancing Effect|
|------|--------------|--------------|--------|
| 0 | 13 | 13 - 6 = 7 | Closer to the average value 9.5|
| 1 | 6 | 6 + 6 = 12 | Closer to the average value 9.5|

#### 7. Data Flow Summary

```
┌─────────────────────────┐     ┌────────────────────────┐
│        Initial data of Rank 0     │     │        Initial data of Rank 1    │
│  K Block 0: [1.1, 1.2, ...]  │     │  K Block 0: [5.1, 5.2, ...]  │
│  K Block 1: [2.1, 2.2, ...]  │     │  K Block 1: [6.1, 6.2, ...]  │
│  K Block 2: [3.1, 3.2, ...]  │     │  V Block 0: [3.3, 3.4, ...]  │
│  K Block 3: [4.1, 4.2, ...]  │     │  V Block 1: [4.1, 4.2, ...]  │
│  V Block 0: [0.1, 0.2, ...]  │     └─────────────────────────┘
│  V Block 1: [0.9, 1.0, ...]  │               ▲
│  V Block 2: [1.7, 1.8, ...]  │               │
│  V Block 3: [2.5, 2.6, ...]  │               │
└────────────┬────────────┘               │
             │                            │
             │ Transmitting Blocks 0 and 1 of Batch 0       │
             │                            │
             ▼                            │
┌─────────────────────────┐     ┌─────────┴─────────┐
│      Generating block tables and policies      │     │       Data transmission    │
│  src_block_table: [0, 1]│───▶│ K Block 0 → K Block 2 of Rank 1 │
│  dst_block_table: [2, 3]│     │ K Block 1 → K Block 3 of Rank 1 │
│  pair_list: [1, 0]      │     │ V Block 0 → V Block 2 of Rank 1 │
└─────────────────────────┘     │ V Block 1 → V Block 3 of Rank 1 │
                                └───────────────────┘
                                          │
                                          ▼
┌─────────────────────────┐     ┌────────────────────────┐
│      Final data of Rank 0       │     │      Final data of Rank 1      │
│  K Block 0: [1.1, 1.2, ...]  │     │  K Block 0: [5.1, 5.2, ...]  │
│  K Block 1: [2.1, 2.2, ...]  │     │  K Block 1: [6.1, 6.2, ...]  │
│  K Block 2: [3.1, 3.2, ...]  │     │  K Block 2: [1.1, 1.2, ...]  │
│  K Block 3: [4.1, 4.2, ...]  │     │  K Block 3: [2.1, 2.2, ...]  │
│  V Block 0: [0.1, 0.2, ...]  │     │  V Block 0: [3.3, 3.4, ...]  │
│  V Block 1: [0.9, 1.0, ...]  │     │  V Block 1: [4.1, 4.2, ...]  │
│  V Block 2: [1.7, 1.8, ...]  │     │  V Block 2: [0.1, 0.2, ...]  │
│  V Block 3: [2.5, 2.6, ...]  │     │  V Block 3: [0.9, 1.0, ...]  │
└─────────────────────────┘     └─────────────────────────┘
```

#### 8. Key Data Structure Examples

##### 8.1 global_shuffle_tensor

```
[1, 0],  # Rank 0 is paired with Rank 1 and acts as the sender.
 [0, 1]   # Rank 1 is paired with Rank 0 and acts as the receiver.
```

##### 8.2 aclshmem_k_cache_tensor (Rank 0)

```
# Shape: (4, 1, 4, 2)
[[[[1.1, 1.2], [1.3, 1.4], [1.5, 1.6], [1.7, 1.8]]],  # Block 0
 [[[2.1, 2.2], [2.3, 2.4], [2.5, 2.6], [2.7, 2.8]]],  # Block 1
 [[[3.1, 3.2], [3.3, 3.4], [3.5, 3.6], [3.7, 3.8]]],  # Block 2
 [[[4.1, 4.2], [4.3, 4.4], [4.5, 4.6], [4.7, 4.8]]]]  # Block 3
```

##### 8.3 src_block_tensor (Rank 0)

```
[0, 1]  # ID of the source block to be transmitted
```

##### 8.4 dst_block_tensor (Rank 0)

```
[2, 3]  # ID of the block transmitted to the destination rank
```

#### 9. Performance Metric Examples

| Metric| Value| Description|
|------|-----|------|
| Transmitted Blocks| 2 | Two blocks are transmitted in this example.|
| Transmitted Tokens| 6 | Two blocks (total capacity of 8 tokens, with 6 being utilized tokens: 1 fully occupied block with 4 tokens + 1 partially occupied block with 2 tokens) are transmitted from Rank 0 to Rank 1.|
| Transmitted Data Volume| 2 x 1 x 4 x 2 x 2 = 32 bytes| 16 bytes for K and V each (float16 type)|
| Load Balancing Effect| From 13:6 to 7:12| Closer to the ideal 9.5:9.5|

#### 10. Application Scenarios

Through this specific data flow example, we can see that the KVShuffle operator:

1. **Solves the load imbalance problem**: Data is transmitted from ranks with high load to ranks with low load.
2. **Maintains data integrity**: The data content remains unchanged before and after transmission.
3. **Efficiently utilizes memory**: Only necessary blocks are transmitted, avoiding unnecessary data movement.
4. **Supports dynamic batch processing**: The transmission policy can be dynamically adjusted based on the actual batch size.

This data flow mechanism is particularly suitable for KV Cache management in distributed training, effectively improving training efficiency and resource utilization.
