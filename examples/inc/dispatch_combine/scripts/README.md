# Qualification scripts / 资格化脚本

## 中文

### 这个目录是干什么的？

真机与随机覆盖的 **资格化启动脚本**。  
**不是** 通信库运行时；二进制安装包可排除，但源码仓应保留。

### 为什么资格化要求走脚本而不是直接跑二进制？

- 脚本强制拓扑检查、NPU 空闲锁、多 rank 结果汇总；裸跑 bin **技术上可行**，但会绕过门禁，结果不作数。
- 把「怎么安全启动」集中在一处，kernel 目录不掺 shell 逻辑。
- 单 INC 进度 / 环境 / baseline 口径：[`../single_inc/SWEEP_STATUS.md`](../single_inc/SWEEP_STATUS.md)。

### 子目录

| 子目录 | 用途 | 为什么要有 |
|---|---|---|
| `single_inc/` | 正式 case launcher 与确定性 operator sweep | 日常真机门禁入口 |
| `random_token_plan/` | 可恢复确定性矩阵 + OS 真随机 campaign | 覆盖长尾 route/shape；调用正式 launcher，不复制数据面 |

---

## English

### What is this directory?

**Qualification launch scripts** for device cases and random coverage.  
Not runtime library code; binary packages may omit them, but source trees should keep them.

### Why qualification requires scripts instead of raw binaries

- Scripts enforce topology checks, idle-NPU locks, and multi-rank aggregation.
  Raw binaries still run, but bypass those gates and are not delivery evidence.
- Keep “how to launch safely” out of kernel directories.
- Progress board: [`../single_inc/SWEEP_STATUS.md`](../single_inc/SWEEP_STATUS.md).

### Subdirectories

| Subdirectory | Role | Why it exists |
|---|---|---|
| `single_inc/` | Formal case launchers + deterministic operator sweep | Daily device gate entry |
| `random_token_plan/` | Resumable deterministic matrix + OS-random campaign | Long-tail route/shape coverage via formal launchers |
