# SHMEM Coding Style Guide

本指南面向 SHMEM 开源仓的贡献者，定义公共 API、公共头文件、实现文件、示例代码、注释、错误处理、内存对齐、Device 侧检查、环境变量、命名空间使用和可验证检查的基本要求。

本指南可作为代码评审和自动化检查的规则输入。人工评审或自动化工具应优先检查第 1 章的目标、命名总表和硬性约束，再按后续章节逐项复核。

所有新增代码应优先满足本指南。修改既有代码时，在不破坏源码兼容性、ABI 兼容性和用户迁移路径的前提下逐步对齐。公共 API、公共类型、公共宏和公共结构体字段一旦发布，不应随意改名、改变参数含义、改变返回码语义或改变同步完成语义。

> 本指南的约束对象是 SHMEM 仓（`include/`、`src/`、`examples/`、`tests/`、`tools/`、`scripts/` 及对外文档中的代码片段）。文中路径如无特殊说明均相对仓库根目录。

---

## 1. 目标、命名总表与硬性约束

### 1.1 目标

本指南的核心目标是明确 SHMEM 对外接口边界，避免内部实现泄漏到用户可见区域。

对外 API 前缀规则：

- 标准接口使用 `aclshmem_*`。
- 扩展接口使用 `aclshmemx_*`。

明确禁止：

- 对外接口中出现 `aclshmemi_*`。
- 对外数据结构中出现 `aclshmemi_*`。
- 示例代码中调用 `aclshmemi_*`。
- 用户文档中将 `aclshmemi_*` 描述为用户可依赖接口。
- 公共头文件中暴露 `ACLSHMEMI_*` 公共宏。

`aclshmemi_*` 与 `ACLSHMEMI_*` 只能作为内部实现符号使用，且应限制在 `src/`、内部测试或未安装的内部实现文件中。

### 1.2 命名规则总表

下表是命名规则的快速检查入口，详细说明见第 4 章。

| 对象 | 规则 | 允许示例 | 禁止或不推荐示例 |
|---|---|---|---|
| 标准公共函数 | `snake_case`，使用 `aclshmem_*` | `aclshmem_malloc`, `aclshmem_putmem_nbi` | `aclshmemInit`, `shmem_put` |
| 扩展公共函数 | `snake_case`，使用 `aclshmemx_*` | `aclshmemx_barrier_on_stream` | `aclshmemi_barrier_on_stream` |
| 内部函数 | `snake_case`，可使用 `aclshmemi_*`，仅限内部实现 | `aclshmemi_prepare_and_post_rma` | 出现在 `include/**` 或 `examples/**` 中 |
| 公共类型 | `snake_case` + `_t`，使用 `aclshmem_*_t` 或 `aclshmemx_*_t` | `aclshmem_team_t`, `aclshmemx_init_attr_t` | `aclshmemi_context_t` |
| 公共结构体/联合体 | `snake_case` + `_t`，字段使用 `snake_case` | `aclshmem_handle_t` | 字段类型暴露 `aclshmemi_*` |
| 枚举类型 | `snake_case` + `_t` | `aclshmem_error_code_t` | `AclshmemErrorCode` |
| 公共枚举值 | 全大写下划线，使用 `ACLSHMEM_*` 或 `ACLSHMEMX_*` | `ACLSHMEM_SUCCESS` | `SHMEM_SUCCESS`, `AclshmemSuccess` |
| 标准公共宏 | `ACLSHMEM_<模块/能力>_<语义名>` | `ACLSHMEM_MAX_TEAMS`, `ACLSHMEM_SYNCBIT_SIZE` | `CANN_SHMEM_TEAM_MAX_SIZE` |
| 扩展公共宏 | `ACLSHMEMX_<模块/能力>_<语义名>` | 新增扩展宏按该模板命名 | `SHMEMX_STREAM_MAX_EVENTS` |
| 内部宏 | `ACLSHMEMI_<模块/能力>_<语义名>`，仅限内部实现 | 内部实现文件中的 `ACLSHMEMI_*` | 出现在公共头文件中 |
| 环境变量 | `SHMEM_<模块/能力>_<语义名>` | `SHMEM_LOG_LEVEL`, `SHMEM_UID_SESSION_ID` | `ACLSHMEM_LOG_LEVEL` |
| 局部变量 | `snake_case` | `local_mem_size` | `localMemSize` |
| 布尔变量 | `is_` / `has_` / `can_` 前缀 | `is_initialized`, `has_error` | `initialized`, `errorFlag` |
| 文件名 | `snake_case` | `shmem_host_rma.h` | `ShmemHostRma.h` |
| 头文件保护宏 | 全大写下划线，与文件路径对应 | `SHMEM_API_H`, `SHMEM_HOST_RMA_H` | `#pragma once`（不统一） |
| C++ 命名空间使用 | 公共头文件禁止全局 `using namespace` | `std::vector<int>` | `using namespace std;` |
| Device 侧防御性检查 | 非必要兜底检查只允许在 debug 模式 | 使用项目已引入的统一 debug check 机制 | release 热路径中反复 `if (ptr == NULL) return;` |

### 1.3 硬性约束

#### 1.3.1 include 目录约束

新增或修改的 `include/**` 不得引入任何 `aclshmemi_*` 接口或数据结构。

该规则覆盖以下路径（含未来新增的类似目录）：

- `include/*.h`
- `include/host/**/*.h`
- `include/device/**/*.h`
- `include/host_device/**/*.h`
- `include/internal/**/*.h`
- `include/detail/**/*.h`
- `include/private/**/*.h`

不允许通过新增 `include/internal`、`include/detail`、`include/private` 等目录规避公共头文件检查。只要位于 `include/**` 下，就视为用户可见头文件。历史兼容别名应按兼容性策略逐步清理，但不得作为新增代码的依据。

#### 1.3.2 样例约束

新增或修改的 `examples/**` 只允许使用对外 API。

示例代码不得：

- 调用 `aclshmemi_*`。
- 声明或依赖 `aclshmemi_*` 数据结构。
- 使用 `ACLSHMEMI_*` 宏。
- 包含内部头文件。
- 依赖未文档化环境变量。
- 依赖内部调试入口。
- 依赖内部 allocator、内部 transport 或内部同步实现。

示例代码必须展示推荐用法，包括初始化、错误处理、通信完成和资源释放流程。

#### 1.3.3 宏定义约束

公共头文件中的宏定义使用：

- `ACLSHMEM_<模块/能力>_<语义名>`
- `ACLSHMEMX_<模块/能力>_<语义名>`

内部宏定义使用：

- `ACLSHMEMI_<模块/能力>_<语义名>`

约束：

- `ACLSHMEM_*` 用于标准公共能力。
- `ACLSHMEMX_*` 用于扩展公共能力。
- `ACLSHMEMI_*` 只能用于内部实现。
- `ACLSHMEMI_*` 不得作为公共宏出现在 `include/**` 中。
- `ACLSHMEMI_*` 不得出现在 `examples/**` 推荐用法中。
- `ACLSHMEMI_*` 不得作为用户文档中的可依赖宏。

现有公共宏可参考 `ACLSHMEM_MAX_TEAMS`、`ACLSHMEM_SYNCBIT_SIZE` 等命名形态。新增扩展公共宏使用 `ACLSHMEMX_*` 模板命名；新增内部宏使用 `ACLSHMEMI_*` 模板命名，并限制在内部实现文件中。

#### 1.3.4 环境变量约束

SHMEM 环境变量使用 `SHMEM_*` 前缀，不使用 `ACLSHMEM_*`。

推荐：

```bash
SHMEM_LOG_LEVEL=INFO
SHMEM_UID_SESSION_ID=127.0.0.1:8899
SHMEM_CYCLE_PROF_PE=0
```

不推荐：

```bash
ACLSHMEM_LOG_LEVEL=INFO
ACLSHMEM_UID_SESSION_ID=127.0.0.1:8899
```

环境变量属于运行时配置，不属于 C API 符号体系。API、类型、枚举和宏使用 `aclshmem_*`、`aclshmemx_*`、`ACLSHMEM_*`、`ACLSHMEMX_*`；环境变量使用 `SHMEM_*`。

#### 1.3.5 内存对齐约束

SHMEM 通信路径中的内存必须满足对应场景的对齐要求。

涉及以下对象时，必须明确对齐粒度、分配来源、生命周期和失败处理：

- 对称内存。
- Device 内存。
- Host 锁页内存。
- UB workspace。
- LocalTensor。
- signal、sync、flag、doorbell、notify 等同步对象。
- 跨 PE 访问的共享元数据。
- 通信缓冲区和传输引擎 workspace。

不得只写"按 cache line 对齐"或"按硬件要求对齐"这类泛化描述。

#### 1.3.6 内存检测约束

对于非 `aclrt` 申请的地址，如果该地址会进入 SHMEM 通信路径、Device 访问路径或跨 PE 内存访问路径，新增代码必须评估内存检测工具（msSanitizer）的支持范围，并通过插桩编译和工具拉起验证。

重点检查场景包括：

- `malloc` / `calloc` / `realloc` 申请的 Host 地址。
- `new` / `new[]` 申请的 Host 地址。
- `mmap` 或其他系统接口申请的地址。
- 项目自定义 allocator 返回的地址。
- 非 `aclrtMalloc`、`aclrtMallocHost` 等 `aclrt` 接口申请，但后续会被通信或 Device 访问的地址。

要求：

- 非 `aclrt` 地址进入异步通信、RMA、DMA 或 Device 访问路径的新增逻辑，应纳入 msSanitizer 验证范围。
- 提交 PR 前，应使用 msSanitizer 拉起本地测试，覆盖新增或修改的内存路径。

#### 1.3.7 Device 侧检查约束

Device 侧代码处于性能敏感路径。非必要不做兜底检查。

发布路径中的 Device 侧代码应满足：

- 不做非必要参数兜底检查。
- 不做非必要状态兜底检查。
- 不做非必要平台兜底分支。
- 不做非必要慢路径 fallback。
- 不在性能路径中加入可由 Host 侧提前完成的合法性检查。
- 不在 Device 侧通过复杂 `if-else` 兜底处理用户错误输入。

Device 侧所有防御性或兜底性质的 `if-else` 检查应放在 debug 模式下，包括：

- 空指针检查。
- PE 范围检查。
- size 范围检查。
- alignment 检查。
- enum 合法性检查。
- 地址空间检查。
- workspace size 检查。
- sync_id、event_id、notify 地址合法性检查。
- 不支持芯片、不支持传输引擎、不支持线程域的兜底检查。

说明：业务语义必需的分支不属于兜底检查。例如按 block、warp、vector、AIV、AIC 或传输引擎做不同执行路径选择，可以保留；但应优先使用编译期分派或轻量分支，避免在热路径中反复做参数防御。

推荐做法：

- Host API、Host wrapper 或 kernel launch 前完成参数校验。
- Device release 路径只保留业务语义必需分支。
- Device debug 路径使用项目已引入并验证过的统一 debug check 机制。
- 对编译期可确定的能力使用 `#if`、`if constexpr`、模板特化或编译期常量分派。
- 对不支持能力应在 Host 侧返回错误，而不是在 Device 热路径中 fallback。

Device release 路径中不推荐反复做空指针、PE 范围、对齐或能力兜底检查后直接返回或进入慢路径。新增这类检查时，应优先迁移到 Host 入口、Host wrapper 或 kernel launch 前；确需在 Device 侧保留的，应说明业务语义必要性。项目尚未统一引入并验证 debug check 封装前，对外规范不预置具体 debug 宏名。

#### 1.3.8 命名空间约束

禁止在公共头文件中使用全局 `using namespace`，包括但不限于：

```cpp
using namespace std;
using namespace shmem;
using namespace acl;
```

禁止范围包括：

- `include/**/*.h`
- `include/**/*.hpp`
- `examples/**/include/*.h`
- `examples/**/include/*.hpp`
- 用户文档中建议复制使用的公共头文件代码片段

公共头文件中应使用显式完全限定名：

```cpp
std::vector<int> values;
std::string name;
```

源文件 `.cpp` 中也应避免文件级 `using namespace`。优先使用显式完全限定名或局部命名空间别名。

需要缩短较长限定名时，可在函数或局部作用域内使用命名空间别名；不要在公共头文件全局作用域引入别名或 `using namespace`。

### 1.4 整体规范

1. 对外接口稳定优先。公开头文件中的函数、类型、宏、枚举值和结构体字段会被用户代码直接依赖，新增或修改前必须考虑兼容性。

2. 内部实现不得外露。内部前缀、内部类型、内部文件路径、内部调试入口和实验能力不应出现在公共头文件、示例代码和用户文档中。

3. `include/**` 全部按用户可见区域处理。即使目录名包含 `internal`、`detail` 或 `private`，也不得在其中放置 `aclshmemi_*` 对外接口或数据结构。

4. 示例只展示推荐用法。示例代码应使用正式发布的公共 API，不依赖内部符号、内部头文件或未文档化环境变量。

5. Host API 与 Device API 边界清晰。Host 侧资源管理、初始化、同步和任务下发，与 Device 侧远端访存、数据搬运、原子和同步操作，应在命名、注释、头文件路径和执行域说明中保持区分。

6. Device 侧 release 路径保持轻量。能在 Host 侧完成的参数检查、状态检查、能力检查，不应在 Device 热路径中重复兜底。Device 侧防御性 `if-else` 检查默认只在 debug 模式开启。

7. 异步语义必须写清楚。提交成功、入队成功、本地完成、远端可见和通信完成是不同概念，接口注释、示例和测试必须区分。

8. 内存与对齐要求必须具体。涉及对称内存、Device 内存、Host 锁页内存、UB workspace、cache line 或 page 的代码，必须说明分配来源、对齐粒度、单位、生命周期和失败处理。

9. 非 `aclrt` 申请且会进入通信或 Device 访问路径的内存，必须纳入 msSanitizer 验证范围。

10. 环境变量使用 `SHMEM_*` 前缀，不使用 `ACLSHMEM_*`。

11. 公共头文件不得污染用户命名空间。禁止全局 `using namespace`。

12. 最小必要抽象。新增封装应服务于稳定 ABI、跨模块复用、清晰边界或可测试性，不为局部代码制造额外层级。

---

## 2. 适用范围

本指南适用于 SHMEM 仓中的 C/C++ 代码、公共头文件、示例代码、测试代码和对外文档中的代码片段。

重点约束包括：

- 对外 API、对外类型、公共宏和公共枚举值的命名。
- `include/` 与 `src/` 的边界。
- Host API 与 Device API 的声明、注释和执行域说明。
- Device 侧 release/debug 检查边界。
- 环境变量命名。
- C++ 命名空间使用。
- 错误码、日志、资源释放和失败路径处理。
- 异步提交、stream 入队、本地完成、远端可见和通信完成语义。
- 对称内存、AscendCL 内存、Ascend C LocalTensor/UB、cache line 和 page 对齐。
- msSanitizer 内存检测。
- 示例代码和 CI 风格检查。

Python 封装（`src/host/python_wrapper/`、`src/python/`）遵循 PEP 8，4 空格，120 列，并保持与 C/C++ 接口命名一致；本指南不展开 Python 细节。

---

## 3. 文件组织

### 3.1 目录边界

SHMEM 仓目录布局如下：

- `include/`：公共声明，面向用户安装和引用。
- `include/host/`：Host 侧公共声明。
- `include/device/`：Device 侧公共声明。
- `include/host_device/`：Host 与 Device 共用的公共类型、宏和声明。
- `src/`：实现文件和内部实现细节。
- `examples/`：推荐用法示例，只使用公共 API。
- `tests/`：单元测试、系统测试、回归测试和性能测试。
- `docs/`：用户文档、开发文档和说明材料。
- `scripts/`、`tools/`：构建、检查、生成和辅助工具。

`include/**` 全部视为用户可见区域。不得将内部接口、内部数据结构或内部宏放入 `include/internal`、`include/detail`、`include/private` 等目录。

公共入口头文件 `include/shmem.h` 只聚合稳定头文件，不直接或间接导出内部实现符号。

### 3.2 头文件与实现文件

- 公共 C/C++ 声明优先使用 `.h`。
- C++ 实现使用 `.cpp`。
- Device 侧模板或内联实现可使用 `.hpp`，但应放在明确的实现边界内（如 `src/**/*.hpp`），不作为用户稳定 ABI 承诺。
- 头文件应保持自包含：用户包含任一公共头文件时，不应依赖额外的隐式 include 顺序。
- 新增头文件使用 include guard（见 3.4），保持全仓一致。
- 公共头文件不得 include `src/` 下的文件。
- 示例代码和用户文档中的代码片段不得 include 内部头文件。
- 公共头文件中不得定义非 `inline` 的函数实体或非 `constexpr` 的全局对象，避免 ODR 和重复定义问题。
- 公共头文件应避免引入不必要的重量级依赖；可前置声明时优先前置声明。
- 公共头文件禁止全局 `using namespace`。

### 3.3 版权头

新增源码文件必须在文件头部添加版权声明。C/C++/头文件使用块注释格式，Python/Shell 使用行注释格式。版权头模板与 [`CONTRIBUTING.md`](https://gitcode.com/cann/shmem/blob/master/CONTRIBUTING.md) 保持一致。

C/C++/头文件模板：

```c
/**
 * Copyright (c) [Name of the copyright owner]. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
```

Python/Shell 文件模板：

```bash
# Copyright (c) [Name of the copyright owner]. 2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ================================================================================================================
```

说明：

- `[Name of the copyright owner]`：个人贡献填个人署名；代表雇主贡献填雇主名称。版权归属有疑问时咨询法律顾问或雇主法务。
- `2026` 为创建或修改该文件的年份，按实际时间修改。
- 公共头文件中的版权头可使用 Doxygen `@cond IGNORE_COPYRIGHT` / `@endcond` 包裹，避免干扰 API 文档生成。

### 3.4 头文件保护宏

新增头文件使用 include guard（`#ifndef` / `#define` / `#endif`），命名与文件路径对应的全大写下划线形式，不使用 `#pragma once`，以保持全仓一致。

```c
#ifndef SHMEM_HOST_RMA_H
#define SHMEM_HOST_RMA_H

/* ... declarations ... */

#endif // SHMEM_HOST_RMA_H
```

命名规则：

- 宏名 = 文件路径去除扩展名，转大写，分隔符用 `_`。
- 例：`include/host/data_plane/shmem_host_rma.h` 在仓库中使用 `SHMEM_HOST_RMA_H`。
- 入口头 `include/shmem.h` 使用 `SHMEM_API_H`。

---

## 4. 命名规范

### 4.1 公共 API 前缀

SHMEM 公共 API 分为标准接口和扩展接口：

- 标准接口使用 `aclshmem_*`。
- 扩展接口使用 `aclshmemx_*`。
- 内部实现可使用 `aclshmemi_*`，但只能存在于内部实现范围内。

公共头文件、示例代码和用户使用文档中不得出现 `aclshmemi_*` 作为用户可调用接口或用户可依赖类型。

不得通过以下方式规避：

```c
/* 不允许 */
typedef struct aclshmemi_context aclshmem_context_t;

/* 不允许 */
#define aclshmem_create_context aclshmemi_create_context

/* 不允许 */
static inline int aclshmem_do_something(...)
{
    return aclshmemi_do_something(...);
}
```

### 4.2 宏定义命名

新增宏定义应区分公共可见范围和内部实现范围。

公共标准宏使用：

```c
ACLSHMEM_<模块/能力>_<语义名>
```

公共扩展宏使用：

```c
ACLSHMEMX_<模块/能力>_<语义名>
```

内部宏使用：

```c
ACLSHMEMI_<模块/能力>_<语义名>
```

要求：

- 公共头文件中的新增公共宏使用 `ACLSHMEM_*` 或 `ACLSHMEMX_*`。
- 内部新增宏使用 `ACLSHMEMI_*`。
- `ACLSHMEMI_*` 不得出现在 `include/**` 中。
- `ACLSHMEMI_*` 不得出现在 `examples/**` 推荐用法中。
- 公共宏和内部宏均使用全大写下划线，不使用驼峰或无项目前缀命名。
- 仅供内部实现使用的调试开关、路径宏、传输细节宏不得放入公共头文件。
- 本规则强制约束增量宏定义。存量兼容宏若已完成整改，应从公共头文件、示例和用户文档中移除。

现有公共宏可参考 `ACLSHMEM_MAX_TEAMS`、`ACLSHMEM_SYNCBIT_SIZE` 等命名形态。新增宏示例不应使用仓库中不存在的具体名称伪装成可用接口；需要说明命名规则时，使用 `ACLSHMEM_<模块/能力>_<语义名>`、`ACLSHMEMX_<模块/能力>_<语义名>`、`ACLSHMEMI_<模块/能力>_<语义名>` 这类模板表达。

### 4.3 环境变量命名

环境变量使用 `SHMEM_*` 前缀。

推荐：

```bash
SHMEM_LOG_LEVEL=INFO
SHMEM_UID_SESSION_ID=127.0.0.1:8899
SHMEM_CYCLE_PROF_PE=0
```

不推荐：

```bash
ACLSHMEM_LOG_LEVEL=INFO
ACLSHMEM_UID_SESSION_ID=127.0.0.1:8899
```

环境变量新增要求：

- 使用 `SHMEM_*` 前缀。
- 名称使用全大写下划线。
- 不使用驼峰。
- 不使用 `ACLSHMEM_*`。
- 不使用内部实现前缀。
- 必须在用户文档中说明默认值、取值范围、单位和生效时机。
- 涉及性能、内存大小、传输路径或调试行为的环境变量必须说明兼容性和安全影响。

示例说明格式：

```markdown
### SHMEM_LOG_LEVEL

Controls SHMEM runtime log level.

Values:
- ERROR
- WARN
- INFO
- DEBUG

Default: WARN.
Effective time: read during SHMEM initialization, for example
`aclshmemx_init_attr`.
```

### 4.4 标准接口与扩展接口

标准接口 `aclshmem_*` 用于稳定、通用、跨平台的 SHMEM 能力，例如：

- 初始化、结束和基础信息查询。
- 对称内存管理。
- PE 和 team 查询。
- RMA、原子、同步和信号操作。

扩展接口 `aclshmemx_*` 用于平台相关、执行域相关或增强能力，例如：

- stream 语义接口。
- block / warp / vector / AIV / AIC 等设备线程域接口。
- 特定传输引擎配置或优化接口。
- 非标准初始化和上下文扩展接口。
- 依赖特定芯片、特定 CANN 版本或实验能力的接口。

新增接口时，如果能力属于通用 SHMEM 语义，优先设计为 `aclshmem_*`；如果能力依赖特定执行模型、后端、stream、传输引擎或实验性扩展，应设计为 `aclshmemx_*`。

---

## 5. C/C++ 格式与命名空间

SHMEM C/C++ 代码采用 Google C++ 风格的基本约束，并结合项目既有习惯。仓库根目录提供 `.clang-format`，以仓库配置为准；新增代码不应通过手工排版对抗 clang-format。

`.clang-format` 关键配置：

- `BasedOnStyle: Google`
- `ColumnLimit: 120`
- `IndentWidth: 4`，`UseTab: Never`
- `PointerAlignment: Left`（指针左对齐，例如 `int *ptr`）
- `BreakBeforeBraces: Custom`，`AfterFunction: true`（函数左大括号换行，其余紧凑）
- `SortIncludes: false`（include 顺序由开发者按语义组织，不强制字母序）

基本要求：

- 缩进使用 4 个空格。
- 单行不超过 120 列。
- 指针左对齐，例如 `int *ptr`。
- 函数名、变量名、文件名使用 `snake_case`。
- 枚举值、公共宏和公共常量使用全大写下划线。
- 不使用 tab。
- 函数参数较多时按语义分组换行，避免强行挤在一行。
- 有意未使用的参数应按项目既有方式显式标记，例如使用 `(void)param;`。
- 确需例外时应先讨论是否调整格式配置。

示例：

```c
ACLSHMEM_HOST_API void aclshmemx_putmem_on_stream(
    void *dst,
    void *src,
    size_t elem_size,
    int32_t pe,
    aclrtStream stream);
```

### 5.1 命名空间

公共头文件中禁止使用：

```cpp
using namespace std;
```

不推荐：

```cpp
// public header
using namespace std;

class Foo {
public:
    vector<int> values;
};
```

推荐：

```cpp
// public header
class Foo {
public:
    std::vector<int> values;
};
```

源文件中也应避免文件级 `using namespace`。

不推荐：

```cpp
using namespace std;

void foo()
{
    vector<int> values;
}
```

推荐：

```cpp
void foo()
{
    std::vector<int> values;
}
```

允许函数内部局部命名空间别名：

```cpp
void create_endpoint()
{
    namespace local_alias = fully::qualified::namespace_name;
    local_alias::Endpoint endpoint;
}
```

---

## 6. 公共类型与 ABI

公共结构体、枚举和类型别名必须保持可读、稳定、可演进。

### 6.1 公共类型要求

- 类型名使用 `aclshmem_*_t` 或 `aclshmemx_*_t`。
- 枚举值使用 `ACLSHMEM_*` 或 `ACLSHMEMX_*`。
- 公共结构体字段使用 `snake_case`。
- 公共结构体字段应有注释，说明用途、生命周期、单位和合法范围。
- 面向 ABI 的结构体建议包含 `version`、`size` 或等价版本字段。
- 对外结构体新增字段时优先尾部追加，不重排已有字段。
- 指针字段必须说明所有权、是否可为空、是否要求对称内存、是否要求对齐。
- enum 字段必须说明非法值处理方式。
- 结构体初始化应提供初始化宏、初始化函数或明确的零初始化约定。
- 公共数据结构中不得出现 `aclshmemi_*` 字段类型、内部句柄或内部上下文指针。

仓库中的公共初始化属性结构体示例：

```cpp
typedef struct aclshmemx_init_attr {
    int my_pe;
    int n_pes;
    char ip_port[ACLSHMEM_MAX_IP_PORT_LEN] = {};
    uint64_t local_mem_size;
    aclshmem_init_optional_attr_t option_attr = {
        (1 << 16) + sizeof(aclshmem_init_optional_attr_t),
        ACLSHMEM_DATA_OP_MTE,
        DEFAULT_TIMEOUT,
        DEFAULT_TIMEOUT,
        DEFAULT_TIMEOUT};
    void *comm_args = nullptr;
    uint64_t instance_id = 0;
} aclshmemx_init_attr_t;
```

### 6.2 内部类型要求

内部类型不得成为用户需要声明、传入、保存或解释的对象。

不允许：

```c
typedef struct aclshmemi_context aclshmem_context_t;
```

确有必要暴露状态或句柄时，应封装成公共 opaque 类型。形式如下，该片段是命名模板，不是仓库现有接口：

```c
typedef struct <public_context> *<public_context_t>;
```

并在文档中明确：

- 创建接口。
- 销毁接口。
- 是否可复制。
- 是否可跨线程使用。
- 是否可跨 PE 使用。
- finalize 后是否失效。
- 失败后句柄是否有效。

用户不应依赖公共句柄的内部布局。句柄的创建、销毁、复制、并发访问和失效规则必须由公共 API 文档说明。

### 6.3 存量兼容策略

本节规则强制约束新增公共 API、公共类型和公共宏。存量不符合规则的公共符号不得扩大使用范围。

确需保留的历史兼容符号，应满足：

- 不在新增代码中继续使用。
- 不在示例和新文档中推荐使用。
- 有明确替代接口。
- 有迁移说明。
- 有删除计划或兼容性说明。

若历史内部前缀符号已经完成整改，例如由 `aclshmemi_*` 迁移到 `aclshmemx_*`，则公共头文件、示例和用户文档中不应再保留旧内部前缀别名。

---

## 7. Host API 与 Device API

### 7.1 Host API

Host API 指由 Host CPU 调用的接口，负责初始化、配置、资源管理、同步和任务下发。

Host API 应满足：

- 使用 `ACLSHMEM_HOST_API` 或项目统一导出宏。
- 命名使用 `aclshmem_*` 或 `aclshmemx_*`。
- 注释中明确执行域为 Host。
- 明确是否线程安全。
- 明确是否阻塞。
- 对异步提交接口说明完成条件。
- 对 stream 接口说明入队成功、本地完成和 stream 同步要求。
- 对分配、释放和查询接口说明特殊返回值。
- 对将进入 Device 路径的参数做必要合法性检查，避免 Device release 路径重复兜底。

Host API 不强制全部返回 `int`。规则如下：

- 管理类、配置类、提交类和新增功能类接口原则上返回 `int` 或 `aclshmem_error_code_t`。
- 内存分配类接口可以返回指针，但必须明确失败返回 `NULL` 的语义。
- 释放类接口可以返回 `void`，但必须明确空指针行为和调用前置条件。
- 查询类接口可以通过返回值或输出参数返回结果，但必须明确失败处理方式。
- 公共 API 不抛异常。
- 公共 API 不返回未文档化的特殊值。

示例：

```c
ACLSHMEM_HOST_API int aclshmemx_init_attr(
    aclshmemx_bootstrap_t bootstrap_flags,
    aclshmemx_init_attr_t *attributes);
ACLSHMEM_HOST_API void *aclshmem_malloc(size_t size);
ACLSHMEM_HOST_API int aclshmem_finalize(void);
ACLSHMEM_HOST_API void aclshmem_free(void *ptr);
```

### 7.2 Device API

Device API 指在设备执行上下文中调用的接口，负责数据搬运、远端访存、原子和同步。

Device API 应满足：

- 使用 `ACLSHMEM_DEVICE` 或项目统一 device 标记。
- 注释中明确调用域，例如 device-only、block、warp、vector、AIV、AIC 或 host/device 皆可。
- 不引入发布路径上的非必要 runtime 分支。
- 不依赖未文档化的编译宏组合。
- 对内存可见性、同步需求和线程域限制写清楚。
- 对传输引擎、芯片型号、CANN 版本依赖写清楚。
- 对 UB workspace、sync_id、event_id、notify 地址等资源写清楚生命周期和复用规则。
- Device release 路径不做非必要兜底检查。
- Device 侧防御性 `if-else` 检查应放在 debug 模式。

Device API 注释应明确：

- 支持的数据类型。
- 指针地址空间要求，例如 global memory、UB、LocalTensor。
- 对称地址要求。
- PE 参数合法范围。
- 是否同步返回。
- 是否需要 `quiet`、`fence`、`barrier`、`sync` 或 engine-specific quiet。
- 并发限制，例如同一 PE、同一 remote address、同一 sync_id 是否支持并发。
- 使用不支持类型、平台或线程域时的行为。
- 哪些错误应由 Host 侧检查，哪些 debug 检查仅在 Device debug 模式启用。

### 7.3 Device 侧 release/debug 检查边界

Device 侧代码应区分三类分支。

| 类型 | release 路径是否允许 | 说明 |
|---|---:|---|
| 业务语义分支 | 允许 | 例如按线程域、数据类型、传输引擎选择不同实现 |
| 编译期能力分派 | 允许 | 例如 `#if`、`if constexpr`、模板特化 |
| 防御性参数检查 | 默认不允许 | 应放入 debug 模式或 Host 侧检查 |
| 状态兜底检查 | 默认不允许 | 应在 Host 侧、初始化阶段或提交阶段处理 |
| 慢路径 fallback | 默认不允许 | 除非已证明是必要语义且有性能评估 |
| 平台兼容 fallback | 默认不允许 | 应在 Host 侧能力查询或初始化阶段处理 |

Device release 路径不推荐：

```c
if (ptr == NULL) {
    return;
}

if (!is_supported_engine(engine)) {
    fallback_to_other_engine(...);
}

if (!is_aligned(ptr, 32)) {
    slow_path(...);
}
```

Device debug 路径应通过项目已发布的统一检查机制覆盖空指针、对齐、PE 合法性等防御性条件。对外规范不预置具体 debug 宏名。

Device 侧检查宏要求：

- 若宏位于公共头文件，使用 `ACLSHMEM_*` 或 `ACLSHMEMX_*` 前缀。
- 若宏只位于内部实现文件，使用 `ACLSHMEMI_*` 前缀。
- debug check 在 release 编译中必须为空实现或被编译器完全优化掉。
- debug check 不得改变 release 行为。
- debug check 不得引入用户可依赖的错误返回语义。

---

## 8. API 注释

公共 API 使用 Doxygen 风格注释。新增或修改公共 API 时，注释至少包含：

- `@brief`：一句话说明接口行为。
- 参数说明：每个参数标注 `[in]`、`[out]` 或 `[in,out]`。
- 返回值：说明成功值和失败错误码。
- 执行域：Host、Device、stream、block、warp、vector、AIV 或 AIC。
- 阻塞语义：blocking、non-blocking、入队或异步提交。
- 完成语义：何时本地完成，何时远端可见，是否需要 `quiet`、`barrier`、`sync`、engine-specific quiet 或 stream 同步。
- 线程安全：是否可并发调用，是否要求外部串行化。
- 内存要求：是否要求对称内存、地址空间、对齐、大小单位和生命周期。
- Device 检查边界：哪些合法性由 Host 侧保证，哪些 debug 检查仅在 debug 模式启用。
- 失败行为：失败后输出参数、句柄、stream 任务和内存状态是否有效。

注释中避免重复代码字面含义。例如不要把 `count` 注释成 "the count"。应说明单位、范围和行为影响。

推荐模板：

```c
/**
 * @brief Submit a non-blocking put operation to a remote PE.
 *
 * @param[in] dst Destination symmetric address on the remote PE. Must not be NULL.
 * @param[in] src Source buffer on the local PE. Must not be NULL.
 * @param[in] size Number of bytes to transfer.
 * @param[in] pe Remote PE number. Must be in the valid PE range.
 *
 * @note Address requirements: dst points to symmetric memory on pe. src is
 *       used as supplied and may be any valid local device GM buffer.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting
 *       ACLSHMEM_DATA_OP_ROCE, complete transfer ranges for both operands must
 *       remain within their respective symmetric memory allocations. Runtime
 *       dispatch may select RDMA
 *       for the target PE; callers need not determine the per-operation engine.
 * @note Execution domain: Host.
 * @note Blocking semantics: non-blocking. Returning from this void API only
 *       means the submission path has returned; it is not a completion signal.
 * @note Completion semantics: the caller must call aclshmem_quiet(),
 *       aclshmem_barrier(), aclshmem_sync(), an engine-specific quiet API,
 *       or synchronize the associated stream as documented before reusing
 *       resources or assuming remote visibility.
 * @note Failure behavior: this existing API does not return a status code.
 *       Implementations must document diagnostics and error propagation
 *       through the established project mechanism.
 * @note Thread safety: thread-safe after successful initialization unless
 *       otherwise documented.
 */
ACLSHMEM_HOST_API void aclshmem_putmem_nbi(void *dst, void *src, size_t elem_size, int32_t pe);
```

Device API 注释示例：

```c
/**
 * @brief Put data from the current device execution context to a remote PE.
 *
 * @param[in] dst Destination symmetric address on the remote PE.
 * @param[in] src Source address in device global memory.
 * @param[in] size Number of bytes to transfer.
 * @param[in] pe Remote PE number.
 *
 * @note Address requirements: dst points to symmetric memory on pe. src is
 *       used as supplied and may be any valid address in device global memory.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting
 *       ACLSHMEM_DATA_OP_ROCE, complete transfer ranges for both operands must
 *       remain within their respective symmetric memory allocations. Runtime
 *       dispatch may select RDMA
 *       for the target PE; callers need not determine the per-operation engine.
 * @note Execution domain: Device.
 * @note The caller must ensure that dst and src are non-NULL, properly aligned,
 *       and that pe is valid. These conditions may be checked only in debug
 *       builds. Release device code does not perform defensive fallback checks.
 * @note Completion semantics: the caller must use the documented quiet or sync
 *       operation before assuming remote visibility.
 */
/* device-side public API declaration */
```

---

## 9. 错误处理与日志

### 9.1 返回码

- 成功返回 `0` 或 `ACLSHMEM_SUCCESS`。
- 失败返回稳定的负值错误码，或返回文档明确的失败值。
- 公共 API 不抛异常。
- 公共 API 不返回未文档化的特殊值。
- 不把底层依赖错误码直接作为公共错误码泄漏，除非该错误码已被纳入公共契约。

错误码应按语义分层：

- 参数错误：空指针、越界、非法枚举值、非法 size、非法 alignment。
- 状态错误：未初始化、重复初始化、finalize 后调用、上下文不匹配。
- 资源错误：内存不足、句柄失效、sync_id 不足、event 不足、底层依赖不可用。
- 运行错误：超时、远端不可达、设备执行失败、stream 提交失败。
- 能力错误：当前平台、芯片、CANN 版本、传输引擎或后端不支持。
- 内部错误：不变量破坏或不应发生的状态，应记录日志并返回稳定错误码。

### 9.2 调用侧处理

调用有状态返回值的公共 API 后应检查返回值。失败路径应及时退出或进入统一错误处理逻辑，不应继续使用：

- 无效指针。
- 无效句柄。
- 无效 team。
- 无效 stream。
- 未完成初始化的上下文。
- 分配失败返回的 `NULL`。
- 提交失败但仍假定存在的异步任务。

示例：

```c
int ret = aclshmemx_set_sdma_config(offset, ub_size, sync_id);
if (ret != ACLSHMEM_SUCCESS) {
    SHM_LOG_ERROR("aclshmemx_set_sdma_config failed, ret=" << ret);
    return ret;
}

void *buf = aclshmem_malloc(size);
if (buf == NULL) {
    SHM_LOG_ERROR("aclshmem_malloc failed, size=" << size);
    aclshmem_finalize();
    return ACLSHMEM_MALLOC_FAILED;
}
```

### 9.3 日志

- 对外日志应使用项目统一 logger 或 logger 回调。
- 错误日志应包含 API 名称、错误码和关键上下文。
- 日志不得泄漏敏感信息、密钥、用户数据内容、完整内存 dump 或未经脱敏的网络地址。
- 同一失败路径避免重复打印，防止日志噪声掩盖根因。
- 公共 API 默认不应输出大量 INFO 日志。
- 性能路径中的日志必须可关闭，或按日志级别控制。
- Device 侧调试输出不得进入默认发布路径。
- Device 侧 release 路径不得通过日志替代错误处理。
- 日志中的错误码必须与返回码一致，避免误导排障。

推荐格式：

```c
SHM_LOG_ERROR("aclshmem_team_split_strided failed, ret=" << ret
              << ", start=" << start
              << ", stride=" << stride
              << ", size=" << size);
```

---

## 10. 异步与同步语义

带有 `nbi`、`on_stream` 或其他异步语义的接口必须明确：

- 成功返回是否仅表示提交成功。
- 成功返回是否表示任务已入队。
- 数据何时本地可复用。
- 远端数据何时可见。
- 用户需要调用哪个完成性接口。
- Host 发起和 Device 发起的操作是否属于同一完成域。
- stream 同步、device 同步、SHMEM quiet、barrier、sync 之间的关系。

### 10.1 必须区分的状态

| 状态 | 含义 |
|---|---|
| 提交成功 | API 参数检查通过，任务已提交到运行时、内部队列或传输引擎 |
| 入队成功 | 任务已进入 stream、SQ、WQ 或 engine 队列 |
| 本地完成 | 本地 buffer 可复用，或本地写入已完成 |
| 远端可见 | 远端 PE 可以观察到数据更新 |
| 通信完成 | 当前执行域发起的通信操作已完成 |
| 全局同步完成 | 指定 team 或所有 PE 达到同步点 |

### 10.2 完成性接口

异步接口必须说明需要哪个完成性接口，例如：

- `aclshmem_quiet`
- `aclshmem_fence`
- `aclshmem_barrier`
- `aclshmem_sync`
- stream 同步接口
- device 同步接口
- engine-specific quiet，例如 MTE、SDMA、RDMA、UDMA 对应的 quiet 或 wait 接口

### 10.3 示例代码要求

示例代码中，如果使用异步接口，必须显式展示完成性操作，例如：

```c
aclshmem_putmem_nbi(dst, src, size, pe);

/* Required before reusing src or assuming remote visibility. */
aclshmem_quiet();
```

stream 示例必须展示 stream 同步或明确交给调用者同步：

```c
aclshmemx_putmem_on_stream(dst, src, size, pe, stream);

aclError acl_ret = aclrtSynchronizeStream(stream);
if (acl_ret != ACL_SUCCESS) {
    return ACLSHMEM_INNER_ERROR;
}
```

---

## 11. 内存与对齐

SHMEM 通信路径对内存布局、对称性、cache line、page、UB workspace 和硬件 DMA 对齐敏感。新增公共结构体、共享状态、通信缓冲区、对称堆分配和内存池切分逻辑时，应按分配来源和使用场景分别说明对齐要求。

不得只写"按 cache line 对齐"或"按硬件要求对齐"这类泛化描述。

### 11.1 对称堆分配

- 用户/API 语义统一使用“对称内存（symmetric memory）”；仅在描述底层预留堆、`heap_base`、`heap_size`
  或分配器实现时使用“对称堆（symmetric heap）”。
- 用户可通信的对称内存应通过公共内存 API 获取，例如 `aclshmem_malloc`、`aclshmem_calloc`、`aclshmem_align`、`aclshmemx_malloc`、`aclshmemx_calloc` 或 `aclshmemx_align`。
- 示例代码和用户文档不得要求用户依赖内部堆实现。
- 使用 `aclshmem_align(alignment, size)` 时，`alignment` 必须为 2 的幂。
- 调用侧必须处理分配失败返回 `NULL` 的情况。
- `size == 0`、分配失败和返回 `NULL` 的语义必须在接口注释或调用侧处理逻辑中体现。
- 各 PE 上的对称内存必须保持一致的分配关系。所有 PE 应按相同顺序、相同大小或文档明确允许的方式分配。
- 不得把普通 `malloc/new` 的 Host 地址作为 SHMEM 对称地址传入 RMA、AMO、signal 或同步 API。
- 不得在公共 ABI 中暴露内部 allocator 布局、chunk header、free list、页表或远端地址翻译细节。

### 11.2 初始化属性与对称内存大小

涉及 `local_mem_size`、heap size、page size 或内部映射大小时：

- 单位必须写明为 byte。
- 是否要求所有 PE 一致必须写清楚。
- 是否按 page 向上对齐必须写清楚。
- 不得在业务代码中散落裸数字。
- 应使用统一宏或 helper 做对齐，例如 `ALIGN_TO(size, page)` 或项目统一封装。
- 对齐失败、整数溢出和 size 超过上限必须有错误处理。
- 公共结构体中表示内存大小的字段应使用 `size_t`、`uint64_t` 或项目统一类型，不使用不明确宽度的 `int`。

### 11.3 AscendCL Device/Host 内存二次分配

基于 `aclrtMalloc`、`aclrtMallocHost` 或其他 AscendCL 内存申请接口获取大块内存并自行切分时，每个子段必须满足对应 AscendCL API 的 size 和起始地址对齐要求。

Device 内存二次分配应遵守：

- 子段大小按 32 字节向上对齐后额外保留 32 字节，即类似 `ALIGN_UP(len, 32) + 32` 模式。
- 子段起始地址满足 64 字节对齐。
- 大块内存切分逻辑必须集中封装，不得在业务代码中手写不同版本的 `ALIGN_UP`。
- 不得假设普通 `malloc/new` 返回的 Host 地址可直接满足 Device DMA、RMA 或异步拷贝路径要求。

对齐与溢出检查要求：

- `alignment` 必须为 2 的幂，否则视为参数错误。
- 对齐计算（align_up）必须检查 `alignment` 合法性，非法时返回错误或 0。
- 指针对齐检查必须同时验证 `alignment` 合法性和地址对齐。
- 涉及 size 加法或乘法时必须检查整数溢出。
- 对齐与溢出检查 helper 属于内部实现，不得放入 `include/**`；应使用项目统一封装，不得在业务代码中手写不同版本。

### 11.4 Ascend C LocalTensor、UB 和数据搬运

Ascend C LocalTensor、UB workspace 和数据搬运相关代码应明确以下要求：

- LocalTensor/UB 相关缓冲区默认按 32 字节粒度设计和检查。
- 如果具体 API 要求 512B、64B、128B 或其他粒度，必须在 API 注释、config 结构和示例中写明。
- UB workspace 字段必须说明：
  - 地址空间。
  - size 单位。
  - 最小容量。
  - 对齐要求。
  - 生命周期。
  - 是否可跨调用复用。
  - 是否可被多个 core 并发使用。
- `sync_id`、`event_id`、notify 地址、flag 地址等同步资源必须说明分配来源和复用规则。
- 不同 TPosition、不同传输引擎、不同芯片型号的对齐要求可能不同，不得把某一接口的 32B 或 512B 约束推广成全仓唯一规则。

### 11.5 cache line、L2 cache line 和 page

- 需要 scalar cache line 对齐时，使用项目统一常量，不硬编码 `64`。
- 需要 L2 cache line 对齐时，使用项目统一常量，不硬编码 `512`。
- 需要 SHMEM page 对齐时，使用项目统一常量，不硬编码 `2 * 1024 * 1024`。
- 共享状态、同步 flag、signal、doorbell、notify 地址和跨 PE 访问的元数据结构必须避免伪共享。
- 固定布局结构体必须配合 `static_assert(sizeof(...))`、字段 offset 检查或 ABI 测试。
- 公共 ABI 不得依赖未文档化的编译器 padding、临时结构体布局或内部 allocator 实现。

示例：

```c
static_assert(SCALAR_DATA_CACHELINE_SIZE == 64, "Unexpected scalar cache line size");
static_assert(L2_CACHELINE_SIZE == 512, "Unexpected L2 cache line size");
```

### 11.6 对齐检查

涉及对齐的公共 Host API 应在入口处检查：

- alignment 是否为 2 的幂。
- pointer 是否满足 alignment。
- size 是否满足最小粒度。
- size 对齐计算是否溢出。
- 对称地址是否来自合法对称堆。
- UB workspace 是否满足 size 和地址空间要求。

检查失败应返回参数错误，不得继续提交通信操作。

Device API release 路径不应重复做上述非必要兜底检查。Device 侧只在 debug 模式下保留对应 `DCHECK` 或断言。

---

## 12. msSanitizer 约束

### 12.1 检查范围

新增代码涉及以下路径时，必须评估是否需要使用 msSanitizer 做内存检测或竞争检测：

- SHMEM RMA 路径。
- SHMEM AMO 路径。
- signal / wait / sync 路径。
- Device kernel 访问路径。
- 工具支持范围内的 DMA / SDMA / RDMA / UDMA / MTE 等传输路径。
- 跨 PE 远端访问路径。
- runtime 内部异步拷贝路径。

后续新增 DMA / SDMA / RDMA / UDMA / MTE 等传输路径或相关接口时，必须主动拉起 msSanitizer 支持范围检查，确认工具是否覆盖该路径，并在 PR 中记录检查结论、验证命令和结果。若工具暂不支持，应说明原因并补充等效验证依据或后续验证计划。

### 12.2 使用方式

SHMEM 的 msSanitizer 验证通过编译插桩和运行时工具拉起完成。

新增代码需要参考以下使用方式：

- 使用 `bash scripts/build.sh -mssanitizer` 或组合样例参数，例如 `bash scripts/build.sh -examples -mssanitizer`。
- `USE_MSSANITIZER=ON` 时，CMake 会按 `SOC_TYPE` 与 bisheng 版本设置 sanitizer 编译选项：非 Ascend950 固定为 `-g --cce-enable-sanitizer`；Ascend950 在新版本 CANN 下同样启用该选项，旧版本仅加 `-g`（此时 AscendC API 相关内存检测不可用，详见 cmake 配置日志）。
- 运行时必须使用 mssanitizer 拉起目标程序，例如：

```sh
mssanitizer -- application parameter1 parameter2 ...
```

样例验证可参考 allgather 方式：

```sh
bash scripts/build.sh -examples -mssanitizer
cd examples/allgather
mssanitizer -- bash run.sh -pes 2
```

### 12.3 PR 要求

涉及以下变更时，PR 说明中应注明 msSanitizer 验证结果：

- 新增 allocator。
- 修改对称堆。
- 修改 Host buffer 管理。
- 修改 Device buffer 管理。
- 修改通信 buffer 生命周期。
- 修改 RMA、AMO、signal、sync、DMA 路径。
- 修改非 `aclrt` 地址进入 Device 或通信路径的逻辑。

PR 提交前应使用 msSanitizer 拉起本地测试。无法由 msSanitizer 覆盖的路径应在 PR 中说明原因，并提供等效验证依据或后续验证计划。

---

## 13. 示例代码

示例代码是用户学习 API 的第一入口，应遵守比普通测试更严格的对外边界。

示例代码必须满足：

- 只使用 `aclshmem_*` 和 `aclshmemx_*` 公共 API。
- 不包含内部头文件。
- 不调用内部函数。
- 不使用 `aclshmemi_*` 或 `ACLSHMEMI_*`。
- 不依赖未文档化环境变量。
- 环境变量示例使用 `SHMEM_*`。
- 展示完整的初始化流程。
- 展示返回值检查。
- 展示通信完成操作。
- 展示资源释放流程。
- 异步示例必须包含完成性操作。
- Device 示例必须说明执行域、启动方式和 Host 侧参数检查。
- 多 PE 示例必须说明运行命令、PE 数量、预期输出和常见失败排查入口。

推荐示例结构：

```c
int main(int argc, char **argv)
{
    /* Build attributes with aclshmemx_set_attr_uniqueid_args or project helper first. */
    int ret = aclshmemx_init_attr(bootstrap_flags, &attributes);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }

    int my_pe = aclshmem_my_pe();
    int npes = aclshmem_n_pes();

    void *buf = aclshmem_malloc(kSize);
    if (buf == NULL) {
        aclshmem_finalize();
        return ACLSHMEM_MALLOC_FAILED;
    }

    if (my_pe == 0 && npes > 1) {
        aclshmem_putmem_nbi(remote_dst, buf, kSize, 1);
        aclshmem_quiet();
    }

    aclshmem_barrier_all();
    aclshmem_free(buf);
    aclshmem_finalize();
    return 0;
}
```

Device 示例中，参数合法性应在 Host 侧检查：

```c
if (dst == NULL || src == NULL || size == 0) {
    return ACLSHMEM_INVALID_PARAM;
}

if (!is_aligned(dst, 32) || !is_aligned(src, 32)) {
    return ACLSHMEM_INVALID_PARAM;
}

/* Launch kernel after validation. */
```

不推荐在 Device kernel 中重复兜底：

```c
if (dst == NULL || src == NULL) {
    return;
}
```

---

## 14. 测试与验证

新增代码至少应包含与风险匹配的验证：

- 命名或头文件边界变更：增加静态检查或脚本检查。
- 公共 API 行为变更：增加单元测试或系统测试。
- 异步语义变更：覆盖提交成功、入队成功、完成、远端可见和异常路径。
- Device API 变更：覆盖不同线程域、编译宏组合和芯片能力组合。
- Device release/debug 检查变更：确认 release 路径不引入非必要 `if-else` 兜底检查。
- 内存分配或对齐变更：覆盖 size 为 0、非法 alignment、未对齐地址、溢出、分配失败和释放路径。
- msSanitizer 相关变更：覆盖非 `aclrt` 地址进入通信或 Device 访问路径的验证场景。
- ABI 相关变更：检查结构体大小、字段顺序、字段 offset 和初始化兼容性。
- 示例变更：确保示例可编译、可运行，并有可验证输出。
- 传输引擎变更：覆盖 MTE、SDMA、RDMA、UDMA 等对应路径。
- 性能路径变更：除正确性测试外，应保留性能基线对比。

测试代码可以使用内部 helper，但公共示例和用户文档不得使用内部符号。

Device 侧性能路径测试应至少确认：

- release 编译下 debug check 被消除。
- release 路径无新增非必要防御性分支。
- debug 编译下非法参数能被 debug check 捕获。
- Host 侧参数检查覆盖 Device release 路径依赖的前置条件。

---

## 15. 发布前检查清单

提交公共 API、公共头文件、Device 侧性能路径或示例前，至少检查以下项：

- 新增或修改的 `include/**` 不引入 `aclshmemi_*` 接口或数据结构。
- 新增或修改的 `include/**` 不引入 `ACLSHMEMI_*` 公共宏。
- 新增或修改的 `examples/**` 和对外文档中无内部前缀推荐用法。
- 新增公共 API 使用 `aclshmem_*` 或 `aclshmemx_*`。
- 新增公共类型使用 `aclshmem_*_t` 或 `aclshmemx_*_t`。
- 枚举值使用 `ACLSHMEM_*` 或 `ACLSHMEMX_*`。
- 新增公共宏使用 `ACLSHMEM_*` 或 `ACLSHMEMX_*`。
- 新增内部宏使用 `ACLSHMEMI_*`，且不得进入公共头文件、示例和用户文档。
- 环境变量使用 `SHMEM_*`，不使用 `ACLSHMEM_*`。
- 公共头文件和 examples 公共 include 头文件无全局 `using namespace`。
- 新增头文件使用 include guard（`#ifndef`/`#define`/`#endif`），命名与文件路径对应。
- 新增源码文件含版权头，年份与署名正确。
- 示例代码只包含公共头文件，只调用公共 API。
- 异步接口注释和示例写明完成性操作。
- Host API 与 Device API 的调用域写清楚。
- Device release 路径不包含非必要兜底 `if-else` 检查。
- Device 防御性检查已放入 debug 模式。
- 返回值和错误码语义可被用户稳定依赖。
- 公共结构体的版本、生命周期、字段单位和 ABI 影响已说明。
- 内存对齐要求已写明分配来源、对齐粒度和失败处理。
- 非 `aclrt` 地址进入通信或 Device 访问路径的新增逻辑已完成 msSanitizer 验证。
- 相关测试、静态检查或等效验证已执行。

### 15.1 增量提交检查命令

本节命令用于检查本次新增或修改的代码。历史代码中可能存在兼容别名、内部示例或迁移中的存量用法，不应把全仓命中直接等同于新增代码违规。对普通 PR 或补丁，优先检查 staged diff 或 review diff。

```bash
# 新增公共区域不得引入 aclshmemi_* 或 ACLSHMEMI_* 推荐用法。
git diff --cached -U0 -- include examples docs \
  | rg '^\+.*\baclshmemi_|\+.*\bACLSHMEMI_'

# 新增公共 API 避免驼峰命名，需要结合上下文人工判断。
git diff --cached -U0 -- include \
  | rg '^\+.*\b[A-Za-z0-9]+[A-Z][A-Za-z0-9_]*\s*\('

# 示例和对外文档不得新增内部路径 include。
git diff --cached -U0 -- examples docs \
  | rg '^\+.*#\s*include\s*[<"].*(internal|detail|private|src)/'

# 新增环境变量不得使用 ACLSHMEM_*。
git diff --cached -U0 -- src examples docs \
  | rg '^\+.*(getenv\("ACLSHMEM_|ACLSHMEM_[A-Z0-9_]+=)'

# 公共头文件和 examples 公共 include 头文件不得新增全局 using namespace。
git diff --cached -U0 -- include 'examples/**/include' \
  | rg '^\+.*\busing\s+namespace\b'

# 异步示例需要人工复核完成性操作。
git diff --cached -U0 -- examples docs \
  | rg '^\+.*(nbi|on_stream|quiet|barrier|sync|Synchronize)'
```

### 15.2 全量基线辅助命令

本节命令只用于评估仓库历史基线和清理范围。若当前基线仍包含历史兼容符号或存量示例命中，需要先形成清理计划或 allowlist；不得直接用全量命中否定普通新增代码。

```bash
rg -n "\baclshmemi_" include examples docs \
  --glob '!**/coding_style_guide.md'

rg -n "\bACLSHMEMI_" include examples docs \
  --glob '!**/coding_style_guide.md'

rg -n "\busing\s+namespace\b" include 'examples/**/include'
```

### 15.3 Device 侧检查命令

Device 侧 release/debug 检查可使用辅助命令发现新增分支，并要求人工复核：

```bash
# 辅助发现 Device 侧新增 if 分支，需要人工判断是否为业务语义分支。
git diff --cached -U0 -- include/device src \
  | rg '^\+.*\bif\s*\('

# 辅助发现新增 fallback 语义。
git diff --cached -U0 -- include/device src \
  | rg '^\+.*\b(fallback|slow_path|return;|return\s+0;)\b'
```

检查要求：

- Device 侧新增 `if` 如果属于参数检查、状态检查、对齐检查或能力兜底，应放入 debug check 或迁移到 Host 侧。
- Device 侧新增 `if` 如果属于业务语义分支，应在代码或 PR 中说明必要性。
- Device 侧新增 fallback 必须说明是否属于功能语义必需路径，并提供性能影响评估。

### 15.4 自动化检查提示

可将下面内容和本文件一起提供给自动化检查工具作为检查要求：

```text
请按 docs/coding_style_guide.md 检查本次改动，优先关注：
1. 新增或修改的 include/** 是否出现 aclshmemi_* 接口、数据结构或 ACLSHMEMI_* 公共宏。
2. 公共 API 是否只使用 aclshmem_* 或 aclshmemx_* 前缀。
3. 公共宏是否使用 ACLSHMEM_* 或 ACLSHMEMX_*，内部宏是否使用 ACLSHMEMI_*。
4. 新增或修改的 examples/** 是否只使用公共 API，是否包含内部头文件或内部符号。
5. 环境变量是否使用 SHMEM_*，是否错误使用 ACLSHMEM_*。
6. 公共头文件是否存在 using namespace。
7. Device release 路径是否新增非必要 if-else 兜底检查；防御性检查是否在 debug 模式。
8. 内存路径是否说明对齐要求，非 aclrt 地址进入通信或 Device 路径是否纳入 msSanitizer 验证范围。
9. 异步接口和示例是否明确 quiet、barrier、sync、stream synchronize 或 engine-specific quiet 等完成性操作。
10. 公共结构体、公共类型、错误码和 ABI 影响是否有注释和兼容性说明。
```

---

## 16. 兼容性策略

历史兼容别名可以在必要时保留，但新增能力应优先以正式公共前缀发布。确需废弃旧接口时，应提供迁移说明、替代接口和足够的过渡期。

公共 API 一旦发布，不应随意改名、改变参数含义或改变返回码语义。必须破坏兼容性时，应通过版本说明和发布说明明确告知用户。

兼容性处理原则：

- 新增代码不得继续使用 deprecated 接口。
- 新增示例不得展示 deprecated 接口。
- 新增文档不得推荐 deprecated 接口。
- 删除 deprecated 符号前，应确认影响范围并提供迁移说明。
- 源码兼容破坏、ABI 破坏和行为语义变化应分别说明。
- 对用户可见的同步完成语义变化必须作为兼容性风险处理。
- 对公共结构体字段顺序、大小、对齐和初始化方式的修改必须评估 ABI 影响。
- Device release 路径行为变化必须评估性能和兼容性影响。
- 将 Device 侧兜底检查迁移到 Host 侧时，必须确认 Host 侧覆盖全部用户入口。

若内部前缀历史别名已经整改为 `aclshmemx_*`、`ACLSHMEM_*` 或 `ACLSHMEMX_*`，则公共头文件、示例代码和用户文档中不应继续保留 `aclshmemi_*` 或 `ACLSHMEMI_*` 兼容入口。

---

## 17. 参考

- 仓库根目录：[`SHMEM`](https://gitcode.com/cann/shmem)
- 贡献指南：[`CONTRIBUTING.md`](https://gitcode.com/cann/shmem/blob/master/CONTRIBUTING.md)
- 许可协议：`LICENSE`（CANN Open Software License Agreement Version 2.0）
- 格式配置：仓库根目录 `.clang-format`（`BasedOnStyle: Google`，120 列，4 空格，指针左对齐）
- 对标参考：NVSHMEM API 命名约定（`nvshmem_*` / `nvshmemx_*` / `NVSHMEM_*`，按 host-only / gpu-only / host+gpu 区分调用域）

> 本指南随版本演进，修订请通过 Issue 或 PR 提出。新增规则应附可验证的检查命令或评审步骤。
