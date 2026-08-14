# shmem-config 命令参考

`shmem-config` 是 SHMEM Python 包自带的配置查询工具，安装 `cann-shmem` 后即可使用。用于查询安装路径、后端类型、编译链接参数、环境检测和诊断信息。A2/A3/Ascend950 的完整环境准备和安装步骤参见[快速开始](../quickstart.md#5-快速上手)。

## 命令概览

| 命令 | 用途 | 输出示例 |
|---|---|---|
| `shmem-config --version` | 查询 SHMEM 版本号 | `1.0.0` |
| `shmem-config --backend` | 当前 SoC 后端选择结果 | `910` 或 `950` |
| `shmem-config --include` | C/C++ 头文件 include 路径 | `/usr/local/lib/python3.x/.../shmem/include` |
| `shmem-config --lib` | 后端 .so 库目录路径 | `/usr/local/lib/python3.x/.../shmem/backends/910` |
| `shmem-config --ldflags` | 推荐链接参数（含 `-I`、`-L`、`-l`） | `-I.../include -L.../backends/910 -lshmem` |
| `shmem-config --rpath` | 推荐 rpath 链接参数 | `-Wl,-rpath,.../backends/910` |
| `shmem-config --root` | SHMEM 包根目录路径 | `/usr/local/lib/python3.x/.../shmem` |
| `shmem-config --runtime-root` | 运行时加载的 libshmem.so 根路径 | `/usr/local/lib/python3.x/.../shmem` |
| `shmem-config --diagnose` | 结构化诊断信息（JSON） | 见下方示例 |
| `shmem-config --check` | 运行环境与包检测脚本 | 终端输出，含芯片/卡数/MTE/SDMA/UDMA/RDMA 状态 |

## 各命令详解

### --version

输出 SHMEM 包的版本号，数据来源为 `shmem/version.info` 的第一行。

```bash
$ shmem-config --version
1.0.0
```

### --backend

通过 `libascendcl.so` 的 `aclrtGetSocName()` 接口自动检测当前 SoC 型号，返回后端标识。

- `Ascend910` / `DAV-220*` 系列 → `910`
- `Ascend950` / `DAV-3510` 系列 → `950`
- 检测失败时兜底返回 `910`

```bash
$ shmem-config --backend
910
```

### --include / --lib / --root

查询 SHMEM 安装目录下的关键路径，常用于在 CMake 或 Makefile 中引用。

```bash
$ shmem-config --include
/usr/local/lib/python3.11/site-packages/shmem/include

$ shmem-config --lib
/usr/local/lib/python3.11/site-packages/shmem/backends/910

$ shmem-config --root
/usr/local/lib/python3.11/site-packages/shmem
```

### --ldflags / --rpath

输出 SHMEM 的编译链接 flags，可用于 `gcc` / `g++` 命令行或 CMake 变量。SHMEM 公共头文件依赖 CANN 头文件；编译前需加载 CANN 环境，并额外传入 `${ASCEND_HOME_PATH}/include` 等当前 CANN 安装要求的路径。

```bash
$ shmem-config --ldflags
-I/usr/local/lib/python3.11/site-packages/shmem/include \
-L/usr/local/lib/python3.11/site-packages/shmem/backends/910 -lshmem

$ shmem-config --rpath
-Wl,-rpath,/usr/local/lib/python3.11/site-packages/shmem/backends/910
```

为兼容内部共享库未写入 `$ORIGIN` 的包版本，运行外部 C/C++ 程序前还需将 `shmem-config --lib` 输出的目录加入 `LD_LIBRARY_PATH`。

### --runtime-root

查询当前进程实际加载的 `libshmem.so` 所在包根路径。命令先读取 `/proc/self/maps`，再从 `LD_LIBRARY_PATH` 查找；若两处均没有 `libshmem.so`，输出为空。

```bash
$ shmem-config --runtime-root
/usr/local/lib/python3.11/site-packages/shmem
```

### --diagnose

以 JSON 格式输出结构化诊断信息，适用于 CI 集成或自动化巡检。全新安装首次加载包时会先执行环境检查；需要直接解析 JSON 的场景，应先单独执行一次 `shmem-config --check`，再采集 `--diagnose` 的标准输出。输出字段包括：

| 字段 | 说明 |
|---|---|
| `version` | SHMEM 包版本 |
| `backend.selected` | 当前选择的后端 |
| `backend.auto_detected_soc` | 自动检测的 SoC 型号 |
| `release_build` | 是否为 Release 构建 |
| `native_load` | 本地扩展及依赖库是否加载成功 |
| `multi_so_conflict` | 是否存在多版本 libshmem.so 冲突 |
| `multi_so_conflict.loaded_paths` | 已加载的 libshmem.so 路径列表 |
| `backend_artifacts` | 后端 .so 文件完整性检查结果 |
| `runtime_root` | 运行时加载路径与包路径是否一致 |
| `degraded` | 是否存在降级运行 |
| `next_steps` | 诊断结论和建议操作 |

示例输出：

```json
{
  "version": "9.0.0.beta.2",
  "backend": {
    "selected": "910",
    "auto_detected_soc": "Ascend910"
  },
  "release_build": true,
  "native_load": {
    "ok": true,
    "error": null
  },
  "multi_so_conflict": {
    "detected": false,
    "loaded_paths": []
  },
  "backend_artifacts": {
    "backend": "910",
    "complete": true,
    "missing": []
  },
  "runtime_root": {
    "path": "/usr/local/lib/python3.11/site-packages/shmem",
    "matches_package_root": true,
    "package_root": "/usr/local/lib/python3.11/site-packages/shmem"
  },
  "degraded": false,
  "next_steps": [
    "No issues detected. SHMEM is ready."
  ]
}
```

### --check [--package <路径>]

运行随 wheel 打包的环境与包检测脚本 `preinstall_check.sh`，检测内容包括：

1. 芯片平台识别
2. CANN / HDK 版本基线
3. 拓扑链路与 MTE 支持
4. SDMA 支持（910B/C 平台）
5. UDMA 支持（Ascend950 平台）
6. RDMA 网卡与网络健康状态
7. 包内容完整性（含 `--package` 时）

```bash
# 检测当前环境
shmem-config --check

# 检测当前环境和指定的软件包目录
shmem-config --check --package /absolute/path/to/package/root
```

检查结果中的 `WARN` 表示对应能力需要确认或不满足条件。当前检查用于诊断和给出处理建议，即使存在能力告警也可能返回成功；请以最终汇总和各检查项输出作为判断依据。

各平台的预期结果如下：

| 检查项 | A2/A3 | Ascend950 |
|---|---|---|
| 芯片识别 | `910B/C` | `950` |
| MTE | 根据拓扑检查 HCCS/SIO 链路 | 默认支持 |
| SDMA | 根据 CANN 版本和环境判断 | 显示 `N/A` |
| UDMA | 显示 `N/A` | 检查 UB 组网和包内 UDMA 产物 |
| RDMA | 检查 RoCE 网络 | 检查 XSCALE 或 HNS_1825 网卡及包后端 |

## 首次导入自动检查

wheel 安装后，当前用户第一次加载 `shmem` 包时会自动运行一次环境检查；执行 `import shmem` 或多数 `shmem-config` 子命令都会触发包加载：

```bash
python3 -c "import shmem; print('import shmem success')"
```

检查结果按 SHMEM 版本记录在 `~/.cache/shmem/.env_checked`，同一版本后续导入不再重复检查。自动检查超过 15 秒时会跳过，不阻塞模块导入。需要重新检查时直接执行 `shmem-config --check`。

如需重新验证首次导入流程，可删除当前用户的检查标记后再次导入：

```bash
rm -f ~/.cache/shmem/.env_checked
python3 -c "import shmem"
```

## 安装验收与问题诊断

安装完成后执行：

```bash
shmem-config --version
shmem-config --backend
shmem-config --diagnose
```

满足以下条件表示 Python wheel 基础安装正常：

- `native_load.ok` 为 `true`，且 `degraded` 为 `false`。
- A2/A3 的 `--backend` 输出 `910`，Ascend950 输出 `950`。
- `backend_artifacts.complete` 为 `true`。
- `multi_so_conflict.detected` 为 `false`。

若 Ascend950 被选择为 `910`，请确认驱动已加载且设备对当前用户可见；自动识别失败时会降级到默认的 910 后端。

若 `multi_so_conflict.detected` 为 `true`，请根据 `loaded_paths` 检查 `LD_LIBRARY_PATH`，移除旧安装目录或重复的 `libshmem.so` 路径，再重新启动进程。Release 包也可能在启动检查阶段直接报告 `Multiple libshmem.so instances detected` 并退出。

若找不到 `shmem-config`，请确认 wheel 已安装到当前 Python 环境，并确认该环境的可执行文件目录位于 `PATH`：

```bash
python3 -m pip show cann-shmem
command -v shmem-config
```

## 使用场景

- **编译外部工程**：用 `--ldflags` 和 `--rpath` 拼装编译命令或 CMake 变量
- **CI 巡检**：用 `--diagnose` 输出 JSON 做自动化健康检查
- **问题排查**：用 `--runtime-root` 确认实际加载的 SHMEM 版本，用 `--diagnose` 检查 .so 冲突和完整性
- **部署前验证**：用 `--check` 一次性检测环境是否满足运行条件
