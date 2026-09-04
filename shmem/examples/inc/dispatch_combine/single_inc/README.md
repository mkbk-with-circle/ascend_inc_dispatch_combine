# Single INC

当前独立 Dispatch/Combine 的唯一主路径位于
[`pull_combine/`](pull_combine/README.md)，协议版本为 Pull V2。

```text
Dispatch: Worker READY → INC GET token+route → parse/reorder → INC PUT B
Compute : B expert compute + same-GPU local weighted reduce
Combine : B Notice → INC GET partials → FP32 reduce → INC PUT original A
```

- 最短接入：[`QUICKSTART.md`](QUICKSTART.md)
- 公共头文件：[`pull_combine/inc_dc_pull_v2_api.h`](pull_combine/inc_dc_pull_v2_api.h)
- 完整示例：[`pull_combine/inc_dc_pull_v2_api_example.cpp`](pull_combine/inc_dc_pull_v2_api_example.cpp)
- 两张详细流程图：[`pull_combine/FLOW.md`](pull_combine/FLOW.md)
- 设计与门禁：[`pull_combine/PULL_DISPATCH_V2_DESIGN.md`](pull_combine/PULL_DISPATCH_V2_DESIGN.md)

历史 V1/Fusion 完整树从 `archive/pre-minimal-v1-fusion-20260904` 标签读取，
不应与当前 Pull V2 混用。

## English

The sole maintained standalone Dispatch/Combine path is `pull_combine/` (Pull
V2). Use `QUICKSTART.md` for the shortest integration path and `FLOW.md` for
the protocol diagrams. Recover legacy implementations from
`archive/pre-minimal-v1-fusion-20260904`, rather than mixing them into the
current build.
