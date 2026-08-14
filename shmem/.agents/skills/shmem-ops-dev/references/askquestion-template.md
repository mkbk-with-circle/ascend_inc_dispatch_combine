# AskQuestion 固定模板（启动 Intake — MUST verbatim 使用）

新窗口 Agent **禁止**自行编造 Phase 0 表单。调用 `AskQuestion` 时 **MUST** 与本文件一致（仅可将 `<用户名>` 等占位符替换为检测到的实际值）。

> **触发方式**：用户 **主动 @ 指定** `.agents/skills/shmem-ops-dev/SKILL.md`（或对话中明确引用该 skill）时生效；**须在对话中引用后才会加载**，不会在未引用时自动注入。
> **Agent 收到 @ 指定后 MUST**：① Read 本 skill 的 `SKILL.md` → ② Read 本文件 → ③ **verbatim** 调用 `AskQuestion`（禁止自行改 CANN 为第三个固定「自定义」按钮）。
> 若只读 `SKILL.md` 摘要、未读本模板，表单会与预期不一致。

---

## 调用前

1. Read 本文件
2. **MUST NOT** 在此之前 Read 源码 / Grep / Shell 编译（Hard Gate）
3. 若用户消息已写明 Docker 容器名（如 `shmem_test`），在 AskQuestion 前记录 `docker_container`（见 [docker-exec-contract.md](docker-exec-contract.md)）
4. 调用 `AskQuestion`，参数 **与下方 JSON 一致**（**5 题**）

---

## AskQuestion 参数（复制使用）

**title**：

```text
Phase 0 启动确认（SHMEM 算子开发）
```

**questions**（5 题；CANN **仅 2 个 options**，自定义走 **Other**）：

```json
[
  {
    "id": "cann",
    "prompt": "1. CANN 环境来源？\n\n若 CANN 不在上述两项，请选择 Other，并在输入框粘贴 set_env.sh 的完整绝对路径。\n样例：/home/<用户名>/CANN/8.5.0/ascend-toolkit/set_env.sh",
    "options": [
      {"id": "session", "label": "A. 沿用当前 shell 已设置的 CANN"},
      {"id": "default", "label": "B. 默认系统安装（如 /usr/local/Ascend/ascend-toolkit/set_env.sh）"}
    ]
  },
  {
    "id": "build_mode",
    "prompt": "2. 算子放在哪个目录？",
    "options": [
      {"id": "custom_ops", "label": "A. custom-ops/<op_name>/ 独立工程"},
      {"id": "in_tree", "label": "B. examples/<op_name>/ in-tree example"}
    ]
  },
  {
    "id": "torch",
    "prompt": "3. 是否需要 PyTorch 接入（Phase 5.5）？",
    "options": [
      {"id": "yes", "label": "需要"},
      {"id": "no", "label": "不需要"}
    ]
  },
  {
    "id": "perf",
    "prompt": "4. 是否需要性能采集（Phase 6）？\n\n采集平台规格带宽表+时延表，与 HCCL baseline 对比。",
    "options": [
      {"id": "yes", "label": "需要"},
      {"id": "no", "label": "不需要"}
    ]
  },
  {
    "id": "perf_auto_optim",
    "prompt": "5. Phase 6 未达标时，是否自动进入性能优化（Phase 6.5，最多 5 轮）？\n\n选「是」：Agent 未达标后 MUST 立即改 kernel 并复测，禁止等用户说「继续」。\n选「否」：仅输出差距表，不擅自优化。\n若第 4 题选「不需要」，本题选「是」无效。",
    "options": [
      {"id": "yes", "label": "是，未达标自动优化"},
      {"id": "no", "label": "否，仅采集报告"}
    ]
  }
]
```

---

## 禁止的 CANN 问法（NEVER）

- ❌ 第三个固定选项 `C. 自定义 set_env.sh`（用户无法在选项内输入路径）
- ❌ 在 `options` 里写「选后在对话提供路径」却不配 Other 说明
- ❌ 拆成 6 题或合并成 1 题（除非平台不支持 5 题表单，则按 #1→#5 **逐题**调用，每题仍用本模板对应条目）

---

## 用户选 CANN = Other 或两步 C 之后

- Other 内容 → `cann_set_env`，`cann_source_mode: user_custom`
- 若用户仅选「自定义」但未填路径 → 追问下一条消息，**禁止** Phase 1

---

## 结果映射


| 字段                      | 映射                                            |
| ----------------------- | --------------------------------------------- |
| `cann=session`          | `cann_source_mode: session_preconfigured`     |
| `cann=default`          | `cann_source_mode: default_system`            |
| `cann=Other`            | `cann_source_mode: user_custom`，路径 = Other 文本 |
| `build_mode=custom_ops` | `independent_project`                         |
| `build_mode=in_tree`    | `in_tree_example`                             |
| `torch=yes/no`          | `torch_required: true/false`                  |
| `perf=yes/no`           | `performance_required: true/false`            |
| `perf_auto_optim=yes`   | `performance_auto_optim: true`（须 `performance_required:true`，否则强制 `false`） |
| `perf_auto_optim=no`    | `performance_auto_optim: false`               |
