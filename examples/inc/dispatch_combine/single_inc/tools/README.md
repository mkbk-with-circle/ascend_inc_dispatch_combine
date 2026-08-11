# Single-INC tools / 单 INC 工具

## 中文

### 这个目录是干什么的？

单 INC 的 **开发/资格化工具**，不链接进运行库。

### 为什么要有？

随机 token plan 是鲁棒性与覆盖率 sweep 的输入；必须带版本与校验，且与正式
wire/compiler 对齐。放在 tools 而不是 runtime，避免把随机源依赖带进产品库。

### 文件一览

| 文件 | 用途 | 为什么要有 |
|---|---|---|
| `inc_dc_os_random_plan.cpp` | 从 OS 随机源生成带版本/校验的 Dispatch 文本计划或 Combine 二进制计划 | 供 `scripts/random_token_plan/` 与手工复现；可单独构建 |

---

## English

### What is this directory?

Single-INC **developer/qualification utilities** that are not linked into the
runtime library.

### Why it exists

Random token plans feed robustness/coverage sweeps. They must be versioned and
checksummed and stay aligned with formal wire/compilers—without pulling OS
randomness dependencies into the product library.

### Files

| File | Purpose | Why it exists |
|---|---|---|
| `inc_dc_os_random_plan.cpp` | OS-random versioned Dispatch text / Combine binary plans | Used by random sweeps; build as a utility |
