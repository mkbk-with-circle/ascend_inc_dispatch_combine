# 单 INC / Fusion 清理后发布验证（nb-borrow，2026-08-11）

本报告验证“只保留单 INC 与 Fusion Kernel”后的代码闭包。它是功能与稳定性
发布门禁，不替代同目录已有的大消息性能资格化报告。

## 环境与保护

- 设备：16× Ascend 910B2C；验证只使用 HCCS 平面 0–7。
- 工具链：CANN 9.1.0 beta 3、BiSheng 15.0.5、`Ascend910B` Release。
- W2 映射：worker NPU 1/2，INC NPU 0；W4 映射：worker NPU 1/2/3/4，INC NPU 0。
- 每个设备 case 前后均经过 launcher 的 live-topology、全机空闲和文件锁门禁；
  收尾 `npu-smi info` 显示 16 张卡均无运行进程。

## 多层验证结果

| 层次 | 覆盖 | 结果 |
|---|---|---|
| 残留审计 | 已删除实现的文件名、CMake target、文档关键词与旧配置 | 无残留 |
| 从零构建 | 最短 C++ API、内部 C bridge、单 INC D/C、native runtime、Fusion plan/API/compute/e2e | 全部通过 |
| Host 正确性 | 单 INC API/plan/runtime 内部门禁、唯一公开示例、Fusion plan/benchmark/API | 全部通过 |
| 设备计算正确性 | BF16 grouped GEMM、融合 FFN、独立 activation | 全部通过 |
| 重复稳定性 | 单 INC API/完整示例/D+C plan、Fusion plan/API，100 轮 | 100/100 通过 |
| 真机端到端 | W2/W4 Dispatch、Combine、Fusion 2-wave/4-wave | 全部 PE 通过 |
| 脚本/工具 | shell 语法、timeline parser 8 个单测、Python 字节码编译 | 全部通过 |

### 单 INC 真机 case

| 算子 | 规模 | 非整齐输入 | 通过 PE |
|---|---:|---|---:|
| Dispatch | W2, K2 | tokens=31, hidden_bytes=512 | 3/3 |
| Combine | W2, K2 | results=63, hidden=257, ragged mode | 3/3 |
| Dispatch | W4, K2 | tokens=31, hidden_bytes=512 | 5/5 |
| Combine | W4, K2 | results=63, hidden=257, ragged mode | 5/5 |

### Fusion 真机 case

下表时间只用于证明清理后路径能稳定执行；每项为 1 次 warmup + 2 次 measure。
理论收益是该次 D/C 窗口的可交叠上限，真实收益是设备 trace 重算的
`dc_window_speedup`。

| 规模 | shape `(T,H,I,K)` | token waves | makespan 均值 | 理论收益 | 真实收益 | 交叠实现率 | 结果 |
|---|---|---:|---:|---:|---:|---:|---|
| W2 | 17,192,320,2 | 2 | 388.030 µs | 1.7010× | 1.6753× | 97.81% | PASS |
| W4 | 17,192,320,2 | 2 | 406.360 µs | 1.6928× | 1.6666× | 97.73% | PASS |
| W2 | 31,256,513,2 | 4 | 504.990 µs | 1.6281× | 1.6110× | 98.31% | PASS |
| W4 | 31,256,513,2 | 4 | 532.540 µs | 1.6435× | 1.6238× | 98.11% | PASS |

机读数据见 [`fusion_results.csv`](fusion_results.csv)。原始 rank 日志按仓库清理政策不提交；
可由 `run_single_inc_*_case.sh` 和 `run_inc_fusion_nb_sweep.sh` 重新生成。

## 复现入口

```bash
cmake -S . -B /tmp/shmem-release -DUSE_EXAMPLES=ON \
  -DCMAKE_BUILD_TYPE=Release -DSOC_TYPE=Ascend910B
cmake --build /tmp/shmem-release --target \
  inc_dc_inference_api_tests inc_dc_single_inc_c_header_tests \
  inc_dc_single_inc_api_example \
  inc_dc_single_inc_stream inc_dc_sv2_dyn_csr_combine \
  inc_fusion_plan_tests inc_fusion_api_tests inc_fusion_compute_tests \
  inc_fusion_e2e inc_fusion_plan_smoke inc_fusion_runtime_skeleton -j4
```

真机必须走：

- `examples/inc/dispatch_combine/scripts/single_inc/run_single_inc_stream_dispatch_case.sh`
- `examples/inc/dispatch_combine/scripts/single_inc/run_single_inc_dyn_case.sh`
- `examples/inc/fusion_kernel/ascend/tests/run_inc_fusion_nb_sweep.sh`
