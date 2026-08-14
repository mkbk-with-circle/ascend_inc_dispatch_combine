# SHMEM Agent Assets

AI Agent 配置资产，用于辅助 SHMEM 通信算子与通算融合算子的开发。

> **声明**：本目录下的 Skills 仅用于辅助 AI Agent 生成 SHMEM 算子代码与文档，**不作性能保证、泛化性保证或生产可用性承诺**。生成的代码需经人工审查、编译验证和充分测试后方可用于生产环境。

## 目录结构

```
.agents/
├── README.md           ← 本文件
└── skills/             ← SHMEM 算子开发 Skill 工具集
    ├── README.md       ← Skill 全景文档（工作流 / Phase / 产物 / 布局）
    ├── shmem-ops-dev/          ← 编排器：Phase 0-7 全流程调度（含 5.5 / 6.5）
    ├── shmem-ops-design/       ← Phase 1：算子设计 → design.md
    ├── shmem-ops-testcase-gen/  ← Phase 2：用例生成 → case matrix + golden/checker
    ├── shmem-ops-code-gen/      ← Phase 3：代码生成 → src/ + CMakeLists.txt
    ├── shmem-ops-compile-debug/ ← Phase 4：编译与调试
    ├── shmem-ops-correctness-eval/ ← Phase 4：正确性验证
    ├── shmem-ops-code-review/     ← Phase 5：设计-代码一致性走读
    ├── shmem-ops-torch-bind/      ← Phase 5.5：Torch 接入与验证（条件性）
    ├── shmem-ops-performance-eval/   ← Phase 6：性能基线采集
    └── shmem-ops-performance-optim/  ← Phase 6.5：性能优化迭代（5 轮，条件性）
```

## 工作流

10 个 Skill 按 Phase 0-7 状态机串行编排，覆盖算子从需求到交付的全流程：

```
Phase 0 需求确认 → 1 设计 → 2 用例 → 3 代码 → 4 编译+正确性 → 5 走读 → 5.5 Torch 接入（条件性） → 6 性能基线 → 6.5 性能优化（条件性） → 7 最终交付
```

阶段门控：前一阶段检查点全部通过后才进入下一阶段。详见 [skills/README.md](skills/README.md)。

## 快速开始

Skills **由用户在对话中主动 @ 引用**；须在对话中引用后才会加载，不会在未引用时自动注入。

在消息中 @ 编排器并描述需求，例如：

> @.agents/skills/shmem-ops-dev/SKILL.md 基于 shmem 在 custom-ops 下实现 alltoallv…

Agent 首步：Read 该 `SKILL.md` → Read `skills/shmem-ops-dev/references/askquestion-template.md` → Read `skills/shmem-ops-dev/references/agent-execution-contract.md` → Read `skills/shmem-ops-dev/references/shmem-repo-resolution.md` → AskQuestion 五项 → 记录 `phase0_intake` 后再按 Phase 1–7 调度子 skill。

也可结合伪代码：

> @.agents/skills/shmem-ops-dev/SKILL.md 基于下列 moe dispatch 伪代码在 custom-ops 实现…

## 边界声明

| 项目 | 说明 |
|------|------|
| **用途** | 仅辅助 Agent 生成 SHMEM 算子代码与文档，不替代人工开发 |
| **性能保证** | 不承诺生成代码的性能指标（带宽、延迟等）达到任何特定标准 |
| **正确性保证** | 不承诺生成代码一次性通过正确性验证，需人工审查与迭代 |
| **Torch 接入** | 仅在需求中启用 `meta.torch_required: true` 时执行，生成的绑定和测试仍需独立验证 |
| **生产可用性** | 生成代码仅供参考与学习，生产部署前须完成完整测试与审查 |
| **影响范围** | Skills 不修改 SHMEM 核心库（`src/`、`include/`），生成内容应限定在用户指定的算子工程或示例目录 |
| **构建/运行** | Skills 不参与 SHMEM 库本身的构建、打包或运行流程 |

本目录内容为 Markdown 工作流资产，不属于 SHMEM 构建、打包、运行或公共 API 的一部分。
