# 当前活跃硬件 Profile

当前资格化环境为 **910b2c-nb**；`910b-yuanmingyu` 的星型拓扑数据保留为独立历史
profile，不与 nb 的双 HCCS 平面结果混写或覆盖。

| 层 | nb 当前资格化 | yuanmingyu 保留结果 |
|:---|:---|:---|
| 环境状态 | `docs/inc/hardware_profiles/910b2c-nb/single_inc_ENV_STATUS.md` | `docs/inc/hardware_profiles/910b-yuanmingyu/single_inc_ENV_STATUS.md` |
| 配置 | `docs/inc/configs/910b2c-nb.env` | `docs/inc/configs/910b-yuanmingyu.env` |
| 结构化数据 | `docs/inc/hardware_profiles/910b2c-nb/` | `docs/inc/hardware_profiles/910b-yuanmingyu/` |

共享状态与硬 gate 位于 `docs/inc/report/single_inc_LIVE_STATUS.md`。换机器、CANN 或拓扑时，
应新建/更新对应 hardware profile，再据该环境的实测 roofline 判定，不能直接复用另一台机器的
绝对带宽 gate。
