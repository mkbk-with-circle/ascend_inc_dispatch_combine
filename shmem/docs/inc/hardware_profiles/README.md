# INC 硬件 Profile

不同机器的数据必须进入各自 profile，禁止用新机结果覆盖旧环境报告。

| Profile | 目录 | 用途 |
|---------|------|------|
| **910b-yuanmingyu** | `910b-yuanmingyu/` | 旧环境 910B / CANN 9.1 beta1 的完整历史结果与迁移对照 |
| **910b2c-nb** | `910b2c-nb/` | 当前 npu-borrow，16×910B2C、双 8-card HCCS 平面；仅验收同平面 W2/W4 |

## 活跃 profile

历史 ACTIVE 指针仍是 **910b-yuanmingyu**；在 npu-borrow 上运行时必须显式选择
`910b2c-nb`，不能把候选环境数据写回旧 profile：

```bash
source docs/inc/configs/910b2c-nb.env
```

pass line 以活跃 profile 的 `single_inc_ENV_STATUS.md` 和 `docs/inc/report/README.md` 列出的最新报告为准。
