# Fusion timeline 解析器

`parse_fusion_timeline.py` 解析 Fusion/vLLM-Ascend trace，并重算 Dispatch/Combine
聚合窗口收益；仅依赖 Python 3 标准库。

## 支持的日志

- `FUSION_INC_TRACE`
- `FUSION_WORKER_TRACE`
- `FUSION_INC_DISPATCH_CHECKPOINTS`
- `FUSION_INC_COMBINE_CHECKPOINTS`
- vLLM 的 `INC_FUSION_DEVICE_TRACE ... records=[{...}]`

> 当前 trace 只有整次 kernel/角色的聚合跨度和相对 checkpoint，无法还原逐 wave
> 绝对 timeline；脚本不会伪造这类事件。

## 快速使用

解析一个 case，默认生成 Markdown：

```bash
python3 examples/inc/fusion_kernel/tools/parse_fusion_timeline.py \
  /path/to/case/pe*.log --strict -o timeline.md
```

也可直接传目录，或输出 JSON/CSV：

```bash
python3 examples/inc/fusion_kernel/tools/parse_fusion_timeline.py /path/to/case -o timeline.md
python3 examples/inc/fusion_kernel/tools/parse_fusion_timeline.py /path/to/case --format json -o timeline.json
python3 examples/inc/fusion_kernel/tools/parse_fusion_timeline.py /path/to/case --format csv -o timeline.csv
```

`--clock-mhz` 显式启用 cycle→µs 换算；脚本不会猜频率。`--strict` 遇到缺字段、
矛盾或损坏 trace 时返回非零。全部参数见：

```bash
python3 examples/inc/fusion_kernel/tools/parse_fusion_timeline.py --help
```

## 指标口径

令 `D`、`C` 为 INC Dispatch/Combine 聚合 span，`O` 为两者交集：

```text
串行 D+C 窗口       = D + C
实际并发窗口         = D + C - O
overlap 实现率       = O / min(D, C)
理论 window 上限     = (D + C) / max(D, C)
实际 window speedup  = (D + C) / (D + C - O)
达到理论比例         = 实际 window speedup / 理论 window 上限
```

`actual_vs_theoretical_pct` 是 **D/C 窗口效率**，不是算子或 vLLM 端到端收益。

`FUSION_WORKER_TRACE` 的角色编号为：

| role | 含义 |
|---:|---|
| 3 | Worker Dispatch 上传 |
| 4 | Worker Dispatch 接收/打包 |
| 5 | Worker FFN 计算 |
| 6 | Worker Combine |

不同角色只有相对 checkpoint，不能直接横向对齐。Markdown 默认每类最多展示 200 行；
`--detail-limit 0` 可取消限制，JSON/CSV 始终保留全部结果。

## 样例与测试

仓库内的 `tests/fixtures/sample_case.log` 同时覆盖独立 E2E trace 和同一行拼接的 vLLM
Python records。运行测试：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s examples/inc/fusion_kernel/tools/tests -p 'test_*.py' -v
```
