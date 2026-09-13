# Pull V2 current validation（2026-09-13）

本报告只描述当前源码树最终保留的 Single-INC Pull V2。

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

## 公共 API

唯一入口为
[`inc_dc_pull_v2_api.h`](../../../../../examples/inc/dispatch_combine/single_inc/pull_combine/inc_dc_pull_v2_api.h)。
应用流程为：

```text
create → dispatch → expert compute/local reduce → combine → wait → destroy
```

API 使用 `BatchHandle` 保存 Dispatch Journal 生命周期，Combine 消费同一 handle；
`BackendOps` 只在 session 创建时绑定 transport/device launcher。

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

当前执行环境没有暴露 `/dev/davinci*`，`npu-smi` 返回 DCMI -8005，因此
2026-09-13 fresh-build Combine device library尚无新的设备带宽样本。本报告不引用
旧 Combine、source-partition 分支或不同 workload 的性能数字。设备恢复后应使用：

```bash
python3 examples/inc/dispatch_combine/single_inc/pull_combine/tests/\
pull_v2_overlap_qualification.py \
  --build-dir /tmp/shmem-final-clean-build-20260913 \
  --output-dir /tmp/pull-v2-current-final \
  --workers 2 --first-npu 0 --plane-size 8 \
  --payload-bytes 134217728 --channels 5 \
  --schedules simultaneous --random-cases 0
```

W4 使用 `--workers 4 --channels 3` 和新的输出目录。
