# 单 INC Dispatch V2（inline route）

本报告只记录 `nb-borrow`（Ascend 910B2C）上当前稳定最优的 V2 Dispatch
路径。旧环境、V1 和其他历史实验数据未被覆盖。

## 协议与资源约束

- 每个 worker 上传完整的 `header | logical routes | padding | hidden_state |
  commit`；metadata 与 hidden state 同一批到达。
- INC 不接收或预知 token plan。它只持有静态 expert placement，在观察到
  committed token record 后在线校验、解析、写 journal 并 fan-out。
- 数据路径始终为 `worker -> INC -> worker`。
- live AIV 为 48；Dispatch 在 INC 和 worker 上均只启动一半，即 24 AIV。
- 中、大消息使用由两个 16 KiB 私有 MTE credit 推导的 32 KiB payload
  microbatch；不使用 W/K 参数查表。
- dense all-destination batch 由 INC 从 token metadata 在线识别。hidden payload
  在 INC 本地只 gather 一次，然后从同一只读 staging fan-out；每个目的 rank
  的 route key、assignment 和 journal 仍逐 token 在线生成。

## 大消息稳定性结果

计量口径为 `fanout payload bytes / 所有 rank 的最慢完整 device makespan`。
每个 case `warmup=3, measure=20`，最后一轮执行完整 record、assignment 和
payload-byte 校验。

| Case | 上传 payload | Fan-out payload | INC/worker AIV | min | mean | max | CV | 结果 |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| W2/K2，2048 token/worker，hidden=16 KiB | 64 MiB | 128 MiB | 24/24 | 37.924 GB/s | 38.168 GB/s | 38.347 GB/s | 0.300% | PASS，3/3 ranks |
| W4/K4，1024 token/worker，hidden=16 KiB | 64 MiB | 256 MiB | 24/24 | 55.639 GB/s | 57.404 GB/s | 59.266 GB/s | 1.568% | PASS，5/5 ranks |

W4 已回到此前 V1 的约 57 GB/s 带宽档位。相对本轮优化前的 V2
（W2 mean 37.558 GB/s、W4 mean 47.437 GB/s），当前 mean 分别提升约
1.6% 和 21.0%。

另外完成以下通用路由回归，均逐 rank PASS：

- W2/K1，64 token/worker，hidden=16 KiB；
- W4/K1，64 token/worker，hidden=16 KiB；
- W4/K2，64 token/worker，hidden=16 KiB。

## 复现

```bash
cd /export/home/yinjinrun.montyyin/.cursur/projects/default/shmem
source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh

export INC_BUILD_DIR=/tmp/shmem-single-inc-build.R7lTK6
export INC_INLINE_V2_INC_LANES=24
export INC_INLINE_V2_WORKER_LANES=24
export INC_INLINE_V2_WARMUP=3
export INC_INLINE_V2_MEASURE=20

examples/inc/dispatch_combine/scripts/single_inc/\
run_single_inc_inline_v2_dispatch_case.sh \
  2 2048 16384 2 /tmp/v2-w2-k2-128m 240

examples/inc/dispatch_combine/scripts/single_inc/\
run_single_inc_inline_v2_dispatch_case.sh \
  4 1024 16384 4 /tmp/v2-w4-k4-256m 240
```

launcher 会先验证同一 HCCS 平面并等待 NPU 空闲；任一 rank 超时、协议错误或
校验失败都会返回非零。

