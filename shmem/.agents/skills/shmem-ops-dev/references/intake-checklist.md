# 启动门禁与 Intake 清单

Agent 在 **任何工具调用之前**（含 Read/Grep/Shell/子 skill 调用）必须完成本清单。未完成 → **停止**，不得进入 Phase 1 或执行环境探测命令。

> 唯一例外：为完成 Step 0.1 自动检测而执行的 **只读** 环境变量检查（如 `echo $ASCEND_HOME_PATH`、`echo $CONDA_DEFAULT_ENV`、`which bisheng`），且不得在此阶段读取源码、编译或调用子 skill。

---

## 启动门禁（Hard Gate）

```
收到 SHMEM 算子开发任务
        │
        ▼
┌───────────────────────────────────────┐
│ Phase 0 Intake（本清单）               │
│  • 五项必问 — 全部 AskQuestion         │
│  • 禁止因用户消息/环境检测而跳过任一项  │
│  • 记录 intake 摘要                    │
└───────────────────────────────────────┘
        │ 五项全部经 AskQuestion 确认
        ▼
   Phase 1 及后续（允许工具调用）
```

**NEVER** 在 Phase 0 未完成时：读 design.md、调用子 skill、执行 [custom-ops-entrypoints.md](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §1/§2、创建算子目录、写代码。

---

## 五项必问（Fresh Session — 全部 MUST，零跳过）

**触发**：用户 **@ 指定** `.agents/skills/shmem-ops-dev/SKILL.md` 后执行（见 [`askquestion-template.md`](askquestion-template.md)）。

用户 @ 引用 skill 时，**MUST** 从被引用文件路径上溯记录 `skills_root`（含 `shmem-ops-dev/` 的目录）；与 `SHMEM_REPO` 分离部署时二者可不同（见 [shmem-repo-resolution.md](shmem-repo-resolution.md)）。

**Fresh Session（新聊天窗口）** 中，以下 **五项 MUST 全部** 通过 `AskQuestion`（或平台等价结构化提问 UI）向用户确认。

**禁止跳过**，包括但不限于以下理由：

- 用户消息已写 `custom-ops`、`独立编译`、`examples`、`in-tree`
- 用户消息已写「要/不要 Torch」「要/不要性能」「全流程」「端到端」
- 用户消息已给出 `set_env.sh` 绝对路径
- shell 已 `source` CANN、`ASCEND_HOME_PATH` 已设置、`bisheng` 可用
- Docker/容器内存在 `/usr/local/Ascend/ascend-toolkit`
- Agent 认为某选项是「合理默认」（如 custom-ops、不要 Torch）

> **唯一不重问**：**同一会话内** 用户已通过 AskQuestion 确认过某项，且 Agent 已记录 `phase0_intake`；**新窗口一律作废，五项全部重问**。

| # | 字段 | AskQuestion 标题（建议） | 选项 |
| --- | --- | --- | --- |
| 1 | **CANN 路径** | CANN 环境来源？ | 见下 **「CANN #1 提问方式」** — **禁止**用无法输入文字的固定选项 C 冒充「自定义路径」 |
| 2 | **构建模式** | 算子放在哪个目录？ | A. `custom-ops/<op_name>/` 独立工程 / B. `examples/<op_name>/` in-tree |
| 3 | **Torch 接入** | 是否需要 PyTorch 接入（Phase 5.5）？ | 需要 / 不需要 |
| 4 | **性能采集** | 是否需要性能采集（Phase 6）？ | 需要 / 不需要 |
| 5 | **自动性能优化** | Phase 6 未达标时是否自动进入优化（Phase 6.5）？ | 是 / 否 |

**执行顺序**：

1. **MUST** Read [`askquestion-template.md`](askquestion-template.md)
2. 按模板 **verbatim** 调用 `AskQuestion`（5 题；CANN 仅 2 options + Other）
3. CANN 自定义路径未收齐时 **禁止** Phase 1

### CANN #1 提问方式（AskQuestion 平台限制）

`AskQuestion` **只有选项按钮 + Other 自由输入**，选项标签内 **无法** 让用户键入路径。自定义 CANN **MUST** 用下列两种方式之一：

#### 方式 A（推荐）：Other 直接填路径

CANN 题 **仅两个固定选项** + 依赖 Other：

| 选项 | 含义 |
| --- | --- |
| A. 沿用当前 shell 已设置的 CANN | 展示 `ASCEND_HOME_PATH`，用户确认 |
| B. 默认系统安装 | 展示候选 `set_env.sh`，用户确认 |

**AskQuestion 题干 MUST 写明**：

> 若 CANN 不在上述两项，请选择 **Other**，并在输入框粘贴 `set_env.sh` 的**完整绝对路径**。
> 样例：`/home/<用户名>/CANN/8.5.0/ascend-toolkit/set_env.sh`

Agent 收到 Other 内容后 **MUST** 校验为非空且路径存在（CANN set_env.sh），写入 `cann_set_env`，`cann_source_mode: user_custom`。

#### 方式 B：两步 — 先选「自定义」，再聊天粘贴

若表单必须提供第三项按钮，第三项 **MUST** 写：

> C. 自定义路径（选后请在**下一条聊天消息**粘贴 `set_env.sh` 绝对路径）

用户选 C 后，Agent **MUST**：

1. **停止**，不进入 Phase 1
2. 回复：「请粘贴 `set_env.sh` 的绝对路径（例：`/home/xxx/CANN/8.5.0/ascend-toolkit/set_env.sh`）」
3. 等待用户**下一条消息**中的路径文本
4. 校验后写入 `cann_set_env`

**禁止**：在选项 C 的标签里写样例路径却 **不** 追问，就假定用户已提供路径。

### CANN 追问细节（#1 选 B / Other / #1b 收到路径后）

- 选 **B 默认系统**：展示探测到的候选路径（见 `cann-env-resolution.md`），用户确认后才 `source`
- **Other 或 #1b 自定义路径**：用户给出的路径 **MUST** 为 `set_env.sh` 绝对路径；Agent 在 Phase 0 内 **MUST** `test -f` 或等价检查（只读），不存在则请用户更正
- 选 **A 沿用当前**：展示 `ASCEND_HOME_PATH` 与 `bisheng`，用户确认后才记录

详细流程见 `shmem-ops-compile-debug/references/cann-env-resolution.md`。

### 记录格式（Phase 0 结束前输出给用户）

```yaml
phase0_intake:
  op_name: <op_name>
  build_mode: independent_project | in_tree_example
  torch_required: true | false
  performance_required: true | false
  performance_auto_optim: true | false   # 仅 performance_required:true 时生效；#4 为 false 时强制 false
  cann_set_env: <绝对路径>
  cann_source_mode: session_preconfigured | default_system | user_custom
  conda_env: <name或 pending>
  shmem_repo: <path或 pending>
  skills_root: <skill树绝对路径>        # MUST；用户 @ skill 时从上溯路径写入
  askquestion_completed: [1, 2, 3, 4, 5]   # MUST 五项齐全；CANN 自定义时 #1 含 #1b 路径已收齐
  cann_set_env_pending: false             # 选 C/Other 后路径未收到时为 true，禁止进 Phase 1
  docker_container: shmem_test | null     # 用户消息指明容器名时写入；见 docker-exec-contract.md
  docker_workdir: <容器内 SHMEM_REPO 绝对路径>
  docker_exec_required: true | false    # docker_container 非空时为 true
```

**`torch_required: true`** → 编排器 **MUST 自动** Phase 5.5（全部算子 `torch_test_*.py` 8PE），不得等用户说「继续」。
**`performance_required: true`** → Phase 6 **MUST** 输出带宽表+时延表。
**`performance_auto_optim: true`**（且 `performance_required: true`）→ Phase 6 未达标后 **MUST 立即** 进入 Phase 6.5，**禁止**等用户说「继续」。
**`performance_auto_optim: false`** → Phase 6 未达标仅报告差距，**禁止**擅自改 kernel 进入 Phase 6.5。

**`docker_container` 非空** → 全部 build/run/perf/Torch **MUST** `docker exec`（[docker-exec-contract.md](docker-exec-contract.md)）。

---

## 条件追问（检测后决定，非五项必问）

| 条件 | 动作 | 工具 |
| --- | --- | --- |
| 用户消息已写明 Docker 容器名（如 `shmem_test`） | 写入 `docker_container`，后续 Shell **全部** `docker exec` | 见 [docker-exec-contract.md](docker-exec-contract.md) |
| 用户未写 Docker 且宿主机无 `npu-smi`/`bisheng` | **MUST** 询问容器名或确认本机 NPU | 对话 |
| `CONDA_DEFAULT_ENV` 为空或为 `base` | MUST 询问 conda 环境名 | 对话或 AskQuestion |
| 工作目录无法自动识别 SHMEM 仓库（无 `include/`+`src/`） | MUST 询问 `SHMEM_REPO` 路径 | 对话或 AskQuestion |
| `install/set_env.sh` 不存在 | 提示先 `bash scripts/build.sh -examples` 或询问已安装路径 | 对话 |
| `import torch/torch_npu` 失败 **且** `torch_required: true` | MUST 询问是否安装或切换环境 | 对话 |

> **注意**：`torch_required` / `performance_required` / `performance_auto_optim` 的 **业务决策** 仅来自 AskQuestion #3/#4/#5，与用户消息措辞或 `import torch` 是否成功无关。

---

## 需求最小集（Step 0.2，可对话确认）

| 字段 | 来源 | 未提供时 |
| --- | --- | --- |
| `op_name` | 用户消息或 AskQuestion | MUST 询问 |
| 功能语义 | 用户消息 | 集合通信类可从 op 名推断，仍须在 intake 摘要中复述确认 |
| SoC | 用户消息 | 可默认当前设备，Phase 1 设计前确认 |
| dtype | 用户消息 | Phase 1 设计门禁前 MUST 确认 |
| `docker_container` | 用户消息（如「shmem_test 容器」） | 未写且需 NPU 时 MUST 询问 |

---

## 示例：「基于 shmem 在 custom-ops 下实现 alltoallv…」

即使用户消息 **已包含** `custom-ops`、dtype、拓扑描述，**五项仍全部 MUST AskQuestion**：

| 项 | Fresh Session |
| --- | --- |
| CANN 路径 #1 | **MUST AskQuestion**（不可因消息或 shell 环境跳过） |
| 构建模式 #2 | **MUST AskQuestion**（不可因消息含 custom-ops 跳过） |
| Torch 接入 #3 | **MUST AskQuestion** |
| 性能采集 #4 | **MUST AskQuestion** |
| 自动优化 #5 | **MUST AskQuestion** |
| op_name | 可从消息推断为 `alltoallv`，不必单独 AskQuestion |

**预期**：一次或连续五次 AskQuestion（或平台单次 5 题表单），五项全部有用户选择后，再进入 Phase 1。

---

## 反模式

- ❌ 用户消息含 `custom-ops` 就跳过 AskQuestion #2
- ❌ 用户消息含 `set_env.sh` 路径就跳过 AskQuestion #1
- ❌ 用户说「全流程/端到端」就跳过 #3/#4 或擅自设 `torch_required: true`
- ❌ 用户只说「开发算子」就默认 `custom-ops` 且不发起 AskQuestion #2
- ❌ `ASCEND_HOME_PATH` 已设置或 Docker 有默认 CANN 就跳过 AskQuestion #1
- ❌ CANN「自定义」仅用 AskQuestion 固定选项 C，不引导 **Other 填路径** 或 **#1b 下一条消息追问**
- ❌ Phase 0 未完成就 `Read` 参考算子、`Grep` 代码库或调用 `shmem-ops-design`
- ❌ 遇失败自行停止等用户说「继续」（见 [agent-execution-contract.md](agent-execution-contract.md) §1）
- ❌ 用户已指定 Docker 却在宿主机跑 [custom-ops-entrypoints.md](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §1/§2/perf（见 [docker-exec-contract.md](docker-exec-contract.md)）
