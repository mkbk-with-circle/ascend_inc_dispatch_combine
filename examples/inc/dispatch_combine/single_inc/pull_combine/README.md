# 单 INC Push-Dispatch / Pull-Combine 协议 v1（实验分支）

本目录实现新协议，不替换也不调用现有 V1 `SingleInc` 后端。当前公共 API
仍指向已验证的旧实现；只有设备 gate 全部通过后，才会另行接入。

## 协议

```text
count:    A0..An ──PUT counts──> INC ──transpose/PUT reply──> B0..Bn
dispatch: A ──PUT one hidden row + route──> INC ──dedup fan-out PUT──> B
compute:  B local experts ──local reduce per (token, B)──> partial rows
combine:  B ──ready descriptor──> INC ──GET/complete/reduce──> accumulator
egress:   ready token runs ──coalesced PUT──> original A ──completion
```

核心不变量：

- A 对每个 token 只向 INC 上传一份 hidden；同一 B 上的多个 expert 共享该份。
- count 矩阵物理路径只能是 worker→INC→worker。
- Combine 不重新传 token ID；它使用 Dispatch 产生的确定性 packed-row 逆映射。
- INC 只保留当前 generation/wave 的临时状态，跨 wave 不保存路由状态。
- ACK 只在相关 GET 或下游 PUT 完成后发布；失败也发布负 ACK 释放发送槽。
- generation、sequence、semantic digest、范围、字节数、保留字段均 fail-closed。
- 严格路径使用 FP32 partial；性能路径允许 FP16/BF16 partial。

## 当前完成度

- 已完成：ABI、路由编译、count 转置、Dispatch/Combine 主机参考状态机。
- 已完成：空 wave、零路由 token、重复目的 rank、`topk > worker_count`、乱序到达、
  分块传输、提前 egress、ring 回压、负 ACK 与计划生命周期保护。
- 已完成：910B 上高阶 SHMEM GET 正确性和 W2/W4 聚合带宽锚点。
- 未完成：持久化设备 server、设备归约/回传、端到端 Dispatch+Combine gate、公共 API 接入。

## 主机 gate

```bash
cmake -S . -B /tmp/shmem-pull-combine-v1-build \
  -DUSE_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release -DSOC_TYPE=Ascend910B
cmake --build /tmp/shmem-pull-combine-v1-build --target \
  inc_dc_pull_combine_plan_tests \
  inc_dc_pull_combine_dispatch_tests \
  inc_dc_pull_combine_state_tests \
  inc_dc_pull_combine_fuzz_tests -j4
```

随机 gate 固定种子运行 500 个 W2–W8 联合 wave；开发时另以
`-Wall -Wextra -Werror`、ASan/UBSan 和 10,000-wave soak 通过。

## nb-borrow 设备锚点（2026-09-02）

卡 0–4 位于同一 HCCS 平面；每次运行前确认 16 卡均无其他 NPU 进程。
数据逐 worker、逐字节校验。数值是 GET+completion 的 host 计时，包含 warmup
传输字节；不是最终算子带宽。

| 规模 | 每 worker | AIV/worker | 聚合带宽 | 相对 1 AIV |
|---|---:|---:|---:|---:|
| W2+1INC | 64 MiB | 1 | 22.41 GB/s | 1.00x |
| W2+1INC | 64 MiB | 2 | 41.74 GB/s | 1.86x |
| W2+1INC | 64 MiB | 4 | 41.76 GB/s | 1.86x |
| W4+1INC | 64 MiB | 1 | 44.95 GB/s | 1.00x |
| W4+1INC | 64 MiB | 2 | 83.49 GB/s | 1.86x |

结论：当前 910B MTE 路径在 2 AIV/peer 饱和。资源策略应根据 worker 数与
实时 AIV 数推导 lane，而不是写死 20/20；W2 Combine 需要 4 AIV，W4 需要 8 AIV。
低阶 UDMA 只在 Ascend 950 开启，本协议使用高阶 SHMEM RMA 自动选择 MTE、SDMA
或 UDMA，避免把 910B 跑到不支持的 engine。
