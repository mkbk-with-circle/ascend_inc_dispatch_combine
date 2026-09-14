<!-- 中文 / Chinese -->
# Pull V2 current validation（更新于 2026-09-14）

本报告只描述当前源码树最终保留的 Single-INC Pull V2。

<!-- English -->
> **English:** This README documents the retained Pull V2 protocol, API, build, tests, binary fingerprints, and reproducible measurements.

## 当前协议

```text
Dispatch:
  Worker immutable Source Slot
  → PUT Header/metadata to INC
  → publication-last 64B READY
  → INC local parse
  → GET hidden once
  → fan-out PUT
  → Destination Completion / Source ACK

Combine:
  Worker FP32 partials
  → PUT 128B READY descriptor to INC
  → publication-last 64B Notice
  → INC local READY validation
  → GET partials
  → FP32 reduction
  → owner PUT / ACK / Completion
```

<!-- English -->
### English — Current protocol

The Chinese section defines the retained Dispatch and Combine control and payload sequence, publication ordering, and completion semantics.

## 公共 API

唯一入口为
[`inc_dc_pull_v2_api.h`](../../../../../examples/inc/dispatch_combine/single_inc/pull_combine/inc_dc_pull_v2_api.h)。
应用流程为：

```text
create → dispatch → expert compute/local reduce → combine → wait → destroy
```

API 使用 `BatchHandle` 保存 Dispatch Journal 生命周期，Combine 消费同一 handle；
`BackendOps` 只在 session 创建时绑定 transport/device launcher。

<!-- English -->
### English — Public API

The Chinese section defines the supported application API and object lifecycle. Function names, types, templates, and examples remain unchanged.

## Fresh build 与 host 验证

环境：CANN 9.1.0-beta.3，SOC_TYPE=Ascend910B。

```bash
source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh
cmake -S . -B /tmp/shmem-final-clean-build-20260913 \
  -DUSE_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release -DSOC_TYPE=Ascend910B
cmake --build /tmp/shmem-final-clean-build-20260913 -j8

/tmp/shmem-final-clean-build-20260913/bin/inc_dc_pull_dispatch_v2_tests
/tmp/shmem-final-clean-build-20260913/bin/inc_dc_pull_combine_v2_tests
/tmp/shmem-final-clean-build-20260913/bin/inc_dc_pull_v2_api_tests
/tmp/shmem-final-clean-build-20260913/bin/inc_dc_pull_combine_v2_contract_tests
/tmp/shmem-final-clean-build-20260913/bin/inc_dc_pull_v2_api_example
python3 -m unittest \
  examples/inc/dispatch_combine/single_inc/pull_combine/tests/test_pull_v2_overlap_qualification.py
```

结果：以上命令全部退出 0。Combine contract 覆盖 W2/W4、零行、小规模、
非对称和 hidden=2049 尾部；API 示例完成 Dispatch、模拟 expert compute、
Combine、数值检查和销毁。

构建指纹：

| 对象 | SHA256 |
|---|---|
| Dispatch source | `4c910356f75af0502ca8a80c78c4d6e050634eb538d9c2a698a06106ee58174c` |
| Dispatch device library | `76c4f3b731d7669d9515e09537f80a7464429f1523d4f899fc1dbffb87041feb` |
| Combine source | `1afbe5d84eda5264aa591025f88b2544c35545027453af2d019dd578b47780be` |
| Combine device library | `63e2e6a991899db4c8b83597f54f0261fa9939ca0d6734230f86c30d20ef4526` |

<!-- English -->
### English — ## Fresh build 与 host 验证

This section covers ## Fresh build 与 host 验证. Commands, paths, code blocks, tables, values, and constraints in the Chinese section above apply unchanged.

## 当前可复现的设备性能

Dispatch 的 fresh-build library 与 2026-09-11 真机测量 library SHA256 完全一致。
配置为 nb-borrow 同一 HCCS 平面 A、CANN 9.1.0-beta.3、H=8192、
每 Worker 128 MiB BF16、top-k2 balanced、3 warmup + 10 measure。

| 规模 | mean time | max time | mean 下行 | min 下行 | CV |
|---|---:|---:|---:|---:|---:|
| W2 | 14.244 ms | 14.297 ms | 37.691 GB/s | 37.552 GB/s | 0.357% |
| W4 | 15.339 ms | 15.481 ms | 70.001 GB/s | 69.360 GB/s | 0.534% |

原始日志：`/tmp/pull-v2-metadata-push-directional-formal-20260911`。
带宽分子只含 fan-out 下行 hidden 字节，metadata PUT 位于 READY 之前且包含在完整时间内。

### Combine 上行带宽（2026-09-14 真机复测）

本次使用上表所列的 Combine source 与 device library 指纹，在 nb-borrow 的同一 HCCS 平面 A 测试：W2 使用 NPU 0、1 和 INC 2；W4 使用 NPU 0–3 和 INC 4。CANN 9.1.0-beta.3，H=8192，`sym_k2_balanced`，每个 Worker 提供 128 MiB FP32 partial，`active_aiv=24`。每个规模独立运行两轮，每轮 3 次预热、10 次正式测量；40/40 个正式样本及所有参与 rank 均正确退出。测试前确认 16 张 NPU 均无运行进程。

主带宽严格采用 **所有 Worker→INC 的 FP32 partial 上行有效字节之和 ÷ INC 端完整 Combine 调用时间之和**。计时从 INC 端 kernel launch 前开始，到 `aclrtSynchronizeStream` 返回结束；owner PUT 字节不计入分子，计时范围包含其开销。每次 W2 调用的上行字节为 268,435,456，W4 为 536,870,912。下表的合并带宽按两轮共 20 个样本的总字节数除以总时间计算，非挑选最佳单轮。

| 规模 | 两轮平均完整时间 | 两轮合并上行带宽 | 最低单次带宽 | 单轮带宽（第 1 / 2 轮） |
|---|---:|---:|---:|---:|
| W2 / K2 | 7.144 ms | **37.575 GB/s** | 32.960 GB/s | 37.303 / 37.852 GB/s |
| W4 / K2 | 7.476 ms | **71.812 GB/s** | 71.497 GB/s | 71.851 / 71.773 GB/s |

W2 第一轮有一次主机端完整时间升至 8.144 ms，使其最低带宽为 32.960 GB/s；第二轮 10 个样本的最低值为 37.757 GB/s。该样本保留在合并统计中。[40 条逐次测量](combine_w2_w4_20260914.csv)记录了字节数、完整耗时、单次带宽、正确性和 AIV 数。原始 rank 日志位于 nb-borrow 的 `/tmp/combine-main-w2-formal.xPSBKP`、`/tmp/combine-main-w2-rerun.1wEx4h`、`/tmp/combine-main-w4-formal.Myj2Qx` 和 `/tmp/combine-main-w4-rerun.6wBtND`。

复现时，在 `source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh` 后，为每轮设置唯一的 `SHMEM_UID_SESSION_ID` 和空闲 TCP endpoint，同时启动 `workers+1` 个 rank：

```text
inc_dc_pull_combine_v2_npu_e2e <workers> <pe> <endpoint> 0 8192 <rows> sym_k2_balanced 3 10
W2: workers=2, pe=0..2, rows=4096
W4: workers=4, pe=0..4, rows=8192
```

这些是 Combine 独立算子测试，Dispatch 的 128 MiB BF16 输入采用不同 token 数；两列 GB/s 不应直接解释为相同工作量下的速度比较。旧 Combine、source-partition 分支或不同 workload 的数值不作为本报告的当前结果。

<!-- English -->
### English — Device performance and metrics

The Dispatch table above reports fan-out BF16 egress bytes divided by the full Dispatch call time. Its device-library SHA256 matches the measured 2026-09-11 build.

The Combine measurements were taken on 2026-09-14 using the Combine source and device-library SHA256 values listed above. Both W2 and W4 used H=8192, balanced K2 routing, 128 MiB of FP32 partials per worker, 24 active AIVs, three warmups, and ten measured calls in each of two independent runs. All 40 measured calls passed correctness checks. W2 used NPUs 0–2 and W4 used NPUs 0–4 on HCCS plane A.

**Combine ingress bandwidth = total Worker-to-INC FP32 partial bytes / total full Combine call time** across the 20 measured calls for each configuration. Timing starts before the INC kernel launch and ends when `aclrtSynchronizeStream` returns. Owner PUT traffic is excluded from the numerator but remains within the timed call. W2 transferred 268,435,456 ingress bytes per call and achieved **37.575 GB/s** across the two runs; W4 transferred 536,870,912 bytes per call and achieved **71.812 GB/s**. The corresponding mean call times were 7.144 ms and 7.476 ms. The lowest single-call values were 32.960 GB/s (W2) and 71.497 GB/s (W4). One W2 call had an 8.144 ms host-side delay; it is retained in the pooled result. The [sample CSV](combine_w2_w4_20260914.csv) contains all 40 measurements, and the Chinese section records the exact invocation and raw-log locations.

These are standalone Combine measurements. Dispatch used a different BF16 token count, so the two operators' GB/s figures do not by themselves compare latency under an identical workload.
