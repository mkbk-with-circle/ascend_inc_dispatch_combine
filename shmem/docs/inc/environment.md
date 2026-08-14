# 16 NPU SHMEM-only INC 样机环境

记录本机 INC 开发与验证环境。敏感账号/路径使用环境变量占位。

## 机器与 NPU

- OS: openEuler / Linux aarch64
- NPU: 8× Ascend910（16 PE），`npu-smi info` 可查
- 显存: 每 die ~64 GiB HBM

## CANN / SHMEM

```bash
export ASCEND_HOME_PATH=/opt/Ascend-9.1/cann-9.1.0-beta.1
source ${ASCEND_HOME_PATH}/set_env.sh
```

- CANN: 9.1.0-beta.1（当前资格化环境）
- SHMEM: 仓库 `version.info`；单 INC 当前状态见 `docs/inc/report/single_inc_LIVE_STATUS.md`

## 构建

```bash
cd shmem
bash scripts/build.sh -examples
```

产物：`build/bin/inc`、`build/bin/allgather`、`build/lib/libshmem.so`

## 配置文件

| 文件 | 用途 |
|------|------|
| `docs/inc/configs/9rank.env` | 8W+1INC（需按目标机器 profile 选择物理卡） |
| `docs/inc/configs/910b-yuanmingyu.env` | 远程 910B profile（`INC_HW_PROFILE=910b-yuanmingyu`） |

## 多硬件 Profile

Gate / baseline 按机器隔离，见 `docs/inc/hardware_profiles/`：

| Profile | 目录 | 说明 |
|---------|------|------|
| **910b-yuanmingyu** | `hardware_profiles/910b-yuanmingyu/` | 当前活跃 profile |

```bash
source docs/inc/configs/910b-yuanmingyu.env
source scripts/inc_env_910b.sh
```

本地同步远程：`../scripts/sync-yuanmingyu.sh push|pull`

## S00 脚本

```bash
bash scripts/inc_env_check.sh
bash scripts/inc_run_existing_smoke.sh
```

## INC 运行示例

```bash
source docs/inc/configs/9rank.env
export LD_LIBRARY_PATH=$PWD/build/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
# 8W+1S AG (rank 0..8)
for i in $(seq 0 8); do ./build/bin/inc 9 $i $INC_IPPORT 9 0 0 ag & done; wait
```
