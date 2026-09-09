# 当前活跃硬件 Profile

当前且唯一有 Pull V2 资格证据的环境是 **910b2c-nb（nb-borrow）**。

| 项 | 路径 |
|---|---|
| 环境状态 | `docs/inc/hardware_profiles/910b2c-nb/single_inc_ENV_STATUS.md` |
| 环境配置 | `docs/inc/configs/910b2c-nb.env` |
| 正式结果 | `docs/inc/report/nb-borrow/pull_v2_directional_20260909/` |

当前实现尚未在其他集群上测试。换机器、CANN 或拓扑时必须新建独立 profile，重新
探测 roofline、运行正确性/稳定性矩阵并设定本机 gate，不能复用 nb 的绝对数字。
