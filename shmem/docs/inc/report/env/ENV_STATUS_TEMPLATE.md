# 单 INC — 环境状态（TEMPLATE）

> 克隆到：`docs/inc/hardware_profiles/<profile-name>/single_inc_ENV_STATUS.md`  
> 只写**本环境**事实与复测。硬约束 / gate 公式 / 全局队列 →  
> [`../../single_inc_LIVE_STATUS.md`](../../single_inc_LIVE_STATUS.md)

| 字段 | 值 |
|:---|:---|
| profile 名 | `<profile-name>` |
| 上次更新 | YYYY-MM-DD |
| 更新者 | |
| 对应共享文档 | `docs/inc/report/single_inc_LIVE_STATUS.md` |

---

## 1. 本机身份与软件

| 项 | 值 |
|:---|:---|
| 主机 / SSH | |
| 代码根 | `…/ascend-样机/shmem` |
| NPU / 型号 | （`npu-smi info`） |
| CANN（可多版本并存时逐行写） | 路径 + 版本号；标明**本次实测用哪一个** |
| 其它本机 CANN（若有） | 例：系统 8.5 / 用户目录 9.0；**勿与别的环境混用数字** |
| 驱动 | |
| env 配置文件 | `docs/inc/configs/<profile>.env`（若有） |

---

## 2. 本机拓扑与 AIV

| 项 | 值 |
|:---|:---|
| Live AIV 总数 | （查询，勿抄别的环境） |
| INC rank / Phy | |
| Worker Phy 集合 | |
| 禁用链路 | |
| 当前 D/C AIV 分配（实测所用） | 例：动态 / 临时 16+32 |
| NPU 空闲 / 独占 | |

---

## 3. 本机性能锚点

| 项 | 值 |
|:---|:---|
| 单向 PushRoofline | ____ GB/s（供 shared `gate_dispatch`） |
| 锚点测量命令/报告 | |
| 套用 shared gate 时是否已更新锚点 | YES / NO |

---

## 4. 本机复测红绿灯

图例：`PASS` / `FAIL` / `GAP` / `UNVERIFIED`。  
对照峰值：shared 文档 §5；gate：shared §2（用**本机** PushRoofline）。

| Case | 正确性 | 带宽或 ratio | vs Gate | vs 共享峰值榜 | 报告路径 |
|:---|:---|---:|:---|:---|:---|
| D W8/K8 @128MiB | UNVERIFIED | | | | |
| D W8/K8 @256MiB | UNVERIFIED | | | | |
| D W8/K1 @128MiB | UNVERIFIED | | | | |
| D W2/K8 @256MiB | UNVERIFIED | | | | |
| C W8/K8 @128MiB | UNVERIFIED | | | | |
| C W8/K8 @256MiB | UNVERIFIED | | | | |
| C W8/K1 @128MiB | UNVERIFIED | | | | |
| C W8/K1 @256MiB | UNVERIFIED | | | | |
| Overlap W8 bal 256M/K8 | UNVERIFIED | | | | |

本机结论（一句话）：

---

## 5. 本机迁移 Checklist

- [ ] 代码 / CANN env  
- [ ] `npu-smi` 空闲  
- [ ] live topo → §2  
- [ ] live AIV → §2  
- [ ] 测 PushRoofline → §3  
- [ ] 跑 §4 最小集  
- [ ] 若刷新全局峰值或关队列项 → **回写 shared LIVE_STATUS**  
- [ ] 更新 `ACTIVE_HW_PROFILE.md`（若本环境成为 ACTIVE）

---

## 6. 本机 Changelog

| 日期 | 谁 | 变更 |
|:---|:---|:---|
| | | 从模板创建 |
