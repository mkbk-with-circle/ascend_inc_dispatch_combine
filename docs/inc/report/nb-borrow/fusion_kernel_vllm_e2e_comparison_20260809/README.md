# nb-borrow：Fusion Kernel 五路径 vLLM 端到端正式结果

这是 2026-08-09 在隔离容器 `montyyin_inc_vllm_0191` 中得到的可查看副本。原始运行目录
仍保留在宿主机
`/export/home/yinjinrun.montyyin/.borrow/inc-vllm-0191/logs/vllm_factorial_formal_20260809`；
本目录位于代码工作区内，因此 Codex 文件查看器可以直接打开。

## 实验口径

- 机器：nb-borrow，W2/W4 与 INC 全在 HCCS 平面 `0..7`。
- 模型：Qwen3-30B-A3B。
- workload：prefill 128，batch 1，输出 1 token，warmup 1，measure 3。
- 自定义路径：`max_model_len=256`、`max_num_batched_tokens=128`、
  `gpu_memory_utilization=0.55`、eager。
- 原生路径：`max_model_len=256`、默认 `max_num_batched_tokens=8192`、
  `gpu_memory_utilization=0.55`、eager。该请求没有发生 chunk；后续扩大 workload 前应统一
  scheduler cap。
- `output_tokens_per_second` 在本 case 中只有 1 个输出 token，因此主要是完成时延倒数，
  不是稳态 decode throughput。

## 路径定义

本报告一共列出五条端到端路径。其中 `native_vLLM` 是外部参考基线；其余四条是同一套
自定义 MoE 实现组成的 `调度方式 × 通信后端` 2×2 归因矩阵：

| | worker-direct SHMEM | 单 INC |
|---|---|---|
| 严格串行调度 | `serial_SHMEM` | `serial_INC` |
| token-wave 融合调度 | `fused_SHMEM` | `fused_INC` |

四条自定义路径具有相同的计算和接口边界：vLLM 仍负责 router、Top-K 选择并产生
`topk_ids/topk_weights`；随后统一进入 `inc_fusion::moe`，由自定义实现完成 route pack、
Dispatch、BF16 GMM1、SwiGLU、BF16 GMM2、按 `topk_weights` 加权归约和 Combine，最后把
与输入 hidden states 同 shape 的 MoE 输出交回 vLLM。四条路径使用同一份模型权重、route、
expert placement、计算 kernel 和计时边界，区别只有通信经过哪里以及是否开放 token-wave
之间的流水交叠。

### `serial_SHMEM`：worker 直连通信，严格串行

```text
vLLM router / Top-K
        ↓
worker 源 token ──SHMEM put──> 目标 expert 所在 worker
        ↓                         ↓
        └────────── 等待完整 Dispatch
                                  ↓
                         GMM1 → SwiGLU → GMM2
                                  ↓
expert worker ──SHMEM put──> token 原属 worker 本地加权归约
                                  ↓
                              MoE output
```

- 只使用 `W` 张计算卡，不创建 INC sidecar，也没有 INC rank。
- Dispatch 和 Combine 都是 worker-to-worker 的 push-only SHMEM 通信。
- Combine 的各路 expert 输出回到 token 原属 worker，由该 worker 按 routing weight 归约。
- 仍然运行统一 fusion kernel，但打开 strict-serial 依赖门：一波必须按
  `Dispatch → GMM1/SwiGLU/GMM2 → Combine` 完成，不允许跨 token wave 的 D/FFN/C 交叠。
- 用途：给 SHMEM 通信后端提供“无流水交叠”的串行基线。

### `serial_INC`：通信经过单 INC，严格串行

```text
vLLM router / Top-K
        ↓
worker ──put──> 单 INC ──put/fan-out──> 目标 expert worker
                                             ↓
                                    GMM1 → SwiGLU → GMM2
                                             ↓
expert worker ──put──> 单 INC 加权归约 ──put──> token 原属 worker
                                             ↓
                                         MoE output
```

- 使用 `W` 张计算卡，另加 1 张不加载模型和 KV cache 的 INC 卡，即共 `W+1` 张卡。
- Dispatch 固定走 `worker → INC → worker`；源 token 到同一目标 worker 的重复 assignment
  保留逻辑记录，但 hidden payload 按目标 worker 做物理去重后由 INC fan-out。
- Combine 固定走 `worker → INC → worker`；INC 收齐对应 token 的 expert 输出，在 INC 上按
  `topk_weights` 完成加权归约，再把最终 hidden row 回传 token 原属 worker。
- 与 `serial_SHMEM` 相同，strict-serial 依赖门禁止跨 wave 的通信/计算交叠。
- 用途：与 `serial_SHMEM` 对比单 INC 数据面的成本；同时作为 `fused_INC` 的无交叠基线。

### `fused_SHMEM`：worker 直连通信，token-wave 流水

```text
时间向右：
wave n+1:  Dispatch ────────────────
wave n:              GMM1/SwiGLU/GMM2
wave n-1:                         Combine ──────
通信路径：worker ────────────────> worker
```

- 卡数和数据路径与 `serial_SHMEM` 相同，仍是 `W` 张 worker 直接 SHMEM put。
- 关闭 strict-serial 门，以 token wave 为最外层调度；稳态允许
  `Dispatch(n+1) || FFN(n) || Combine(n-1)`。
- GMM1、SwiGLU、GMM2 还会按 expert-ready/activation-wave 的依赖边细粒度推进，但外层调度
  仍是 token wave，并非退回 expert-wave。
- 用途：`serial_SHMEM → fused_SHMEM` 的差值只归因于 token-wave 融合/通信计算交叠，不含
  INC 带来的收益。

### `fused_INC`：单 INC 数据面，token-wave 融合交付路径

```text
时间向右：
INC Dispatch cohort:  D(n+1) ─────────────────
worker AIC/AIV:                 FFN(n) ─────────────
INC Combine cohort:                         C(n-1) ───────

数据路径：worker → 单 INC → worker；Dispatch 和 Combine 使用不相交的 INC AIV cohort。
```

- 使用 `W+1` 张卡；INC 是独立、无模型、长驻的 sidecar/service kernel。
- 通信语义与 `serial_INC` 完全相同，但开放 token-wave 流水，以及同一 INC 上
  Dispatch/Combine 的并发执行。
- INC 的 Dispatch AIV 和 Combine AIV 是互不重叠的固定 cohort，因此 D/C 可以任意时序重叠，
  不会争用同一个 AIV；worker 侧另有固定的 D、C、FFN AIV/AIC 分工。
- 这是目标交付路径。`serial_INC → fused_INC` 表示在相同 INC 数据面下融合流水的真实收益；
  `fused_SHMEM → fused_INC` 才用于观察通信后端从 worker-direct 换成单 INC 后的变化。

### `native_vLLM`：原生外部参考基线

`native_vLLM` 不属于上述四格。它不注册 `SingleIncMoECommMethod`，不进入
`inc_fusion::moe`，也不启动 INC sidecar；router、EP 通信和 Expert 计算均保留
vLLM-Ascend 0.19.1 为 Qwen3-30B-A3B 选择的原生路径。本次日志显示 EP 已启用，使用
unquantized MoE 的 OOT backend。它用于回答“完整自定义方案距离原生 vLLM 端到端性能还有
多少”，但由于计算 kernel、通信实现和运行时开销都可能不同，不能把
`native_vLLM ↔ fused_INC` 的全部差值归因于 INC 或 token-wave 交叠。

因此，四种自定义路径的正确读法是：

- 纵向比较 `serial → fused`：测融合调度和交叠收益；
- 横向比较 `SHMEM → INC`：测通信数据面的变化；
- 与 `native_vLLM` 比较：只作端到端竞争力评估，不作单因素归因。

## 汇总

| 路径 | W | 平均完成时间 (ms) | 中位数 (ms) | CV | 输出 token/s |
|---|---:|---:|---:|---:|---:|
| native_vLLM | 2 | 69.293 | 68.415 | 2.956% | 14.432 |
| serial_SHMEM | 2 | 781.976 | 781.469 | 0.097% | 1.279 |
| serial_INC | 2 | 838.835 | 838.705 | 0.071% | 1.192 |
| fused_SHMEM | 2 | 657.242 | 657.832 | 0.167% | 1.522 |
| fused_INC | 2 | 686.750 | 686.029 | 0.149% | 1.456 |
| native_vLLM | 4 | 72.371 | 70.985 | 4.187% | 13.818 |
| serial_SHMEM | 4 | 445.322 | 445.241 | 0.165% | 2.246 |
| serial_INC | 4 | 505.322 | 504.283 | 0.304% | 1.979 |
| fused_SHMEM | 4 | 385.259 | 385.130 | 0.066% | 2.596 |
| fused_INC | 4 | 442.039 | 442.139 | 0.132% | 2.262 |

同一通信后端内，融合相对串行的真实收益为：W2 SHMEM `1.190×`，W2 INC `1.221×`，
W4 SHMEM `1.156×`，W4 INC `1.143×`。这是当前数据能支持的交叠收益。原生 vLLM 明显
更快，说明自定义 engine 路径仍有较大的非通信开销，不能宣称当前 Fusion 已取得端到端
推理加速。

机器与容器的完整复现方法见
[`RUNBOOK_NB_VLLM.md`](../../../../../examples/inc/fusion_kernel/framework/vllm_ascend/RUNBOOK_NB_VLLM.md)。

## 原始结果

- [W2 native vLLM](raw/native_vllm_w2_prefill128/result.json)
- [W2 serial SHMEM](raw/serial_shmem_w2_prefill128/result.json)
- [W2 serial INC](raw/serial_inc_w2_prefill128/result.json)
- [W2 fused SHMEM](raw/fused_shmem_w2_prefill128/result.json)
- [W2 fused INC](raw/fused_inc_w2_prefill128/result.json)
- [W4 native vLLM](raw/native_vllm_w4_prefill128/result.json)
- [W4 serial SHMEM](raw/serial_shmem_w4_prefill128/result.json)
- [W4 serial INC](raw/serial_inc_w4_prefill128/result.json)
- [W4 fused SHMEM](raw/fused_shmem_w4_prefill128/result.json)
- [W4 fused INC](raw/fused_inc_w4_prefill128/result.json)

每个 raw 子目录还保留对应 `launcher.log`，用于复核 setup、sidecar 生命周期和退出状态。
