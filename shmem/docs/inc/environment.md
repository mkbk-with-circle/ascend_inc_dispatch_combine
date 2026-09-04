# Single-INC Pull V2 开发环境

当前唯一完成资格测试的环境是 **nb-borrow / 910B2C**。

## 软件与代码

| 项 | 值 |
|---|---|
| 代码根 | `/export/home/yinjinrun.montyyin/.cursur/projects/default/shmem` |
| CANN | `/usr/local/Ascend/cann-9.1.0-beta.3` |
| 驱动 | `25.0.rc1.1` |
| NPU | 16× Ascend 910B2C，64 GiB HBM/卡 |

```bash
source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh
source docs/inc/configs/910b2c-nb.env

cmake -S . -B /tmp/shmem-pull-v2-build \
  -DUSE_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release -DSOC_TYPE=Ascend910B
```

## 拓扑约束

- NPU 0–7、8–15 是两个独立 HCCS 平面；
- 正式测试只使用一个平面内的 W2/W4 + 1 INC；
- 本机无法构造同平面 W8+1INC，因此 W8 不属于当前资格范围；
- 每次真机测试前后必须确认所有 NPU 空闲并取得独占锁。

完整环境状态见
[`hardware_profiles/910b2c-nb/single_inc_ENV_STATUS.md`](hardware_profiles/910b2c-nb/single_inc_ENV_STATUS.md)。
当前实现未在其他机器或 CANN 组合上完成验证。
