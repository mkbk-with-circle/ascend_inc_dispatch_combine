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
