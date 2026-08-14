# 910b-yuanmingyu — 活跃 profile

远程 SSH：`yuanmingyu-ymy` → `10.140.158.149:49285`，用户 `ymy`。

## 机器指纹（初检 2026-07-08）

| 项 | 值 |
|----|-----|
| 主机 | `probing-real-worker-0`（K8s worker 容器） |
| NPU | 8× Ascend910B（`IT22HMDA_1_S`，PCI `0xD803`） |
| 逻辑 PE | 16 |
| HBM | 65536 MiB / die |
| CANN | 优先 `/home/ymy/Ascend/cann-9.0.0`，fallback `/usr/local/Ascend/cann-8.5.0` |
| 驱动 | npu-smi 25.3.rc1.2 |

## 当前证据

- `single_inc_ENV_STATUS.md`：本机 CANN/topology/AIV 与最新单 INC 结果。
- 单 INC 可机读报告统一收敛在 `docs/inc/report/README.md` 列出的最小证据集。
- 历史 `gates/` / `p5/` / `snapshots/` 和递归复制的 profile 已清理。

## 环境

```bash
source docs/inc/configs/910b-yuanmingyu.env
source scripts/inc_env_910b.sh   # 在 shmem/ 下
```
