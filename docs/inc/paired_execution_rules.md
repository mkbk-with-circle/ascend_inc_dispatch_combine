# PAIRED DC-LL 执行约束（MP plan execution rules）

本文件固化 `paired-dcll-performance` 计划的执行约束，**不修改 plan 文件本身**。

## 子 Agent

- 若使用 Cursor Task/子 agent 推进本计划：仅允许 **composer 非 fast** 型号。
- 禁止用 fast 子 agent 跑 NPU gate / 改 kernel。

## NPU 下发

1. 下发前：`pgrep -af 'inc_dc_dispatch|run_d03|run_realworld'` + `timeout 10 npu-smi info`。
2. 禁止叠跑第二套 16-rank PAIRED。
3. 本会话僵尸可 `pkill -9 -f 'inc_dc_dispatch'`；他人训练只等待。
4. 所有命令带 `timeout`；smoke wall 默认 60–90s。

## 构建

kernel `.so` 更新后必须：

```bash
rm -f build/bin/inc_dc_dispatch
cmake --build build -j8 --target inc_dc_dispatch
```

确认 `bin` 与 `libinc_dc_dispatch_kernel.so` 时间戳接近；运行时 `LD_LIBRARY_PATH=.../build/lib:...`。

## 默认 PAIRED 参数（功能正确优先）

| 变量 | 默认 |
|------|------|
| `INC_DC_LL_QUEUE_DEPTH` / `INC_DC_RING_DEPTH` | 8 |
| `INC_DC_LL_FIFO_PUBLISH_MODE` | `ordered_v2` |
| `INC_DC_AIV_PROFILE` | `balanced_d9_rx8` |
| `INC_DC_LL_FANOUT_ENGINE` | `mte_pingpong` |
| `INC_DC_LL_VERIFY_MODE` | `sampled`（structural drain 即 `payload_verify_pass=1`） |

`depth=2` 仅用于有意压力测试；环复用需依赖 switch 侧 `token_seq`/`pending!=0` 守卫与“完成时不写回 pending=0”。

## Gate 诚信

见同目录 `paired_gate_integrity.md`：无真实 JSON/CSV 证据不得将 todo 标为 completed。
