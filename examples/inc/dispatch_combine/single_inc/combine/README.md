# Single-INC Combine / 单 INC Combine

> **构建真相以** `examples/inc/CMakeLists.txt` **为准。**  
> CMake target 仍可能叫 `inc_dc_sv2_dyn_csr_*`（二进制/报告兼容），但**编译进库的源文件是稳定名** `inc_dc_combine_*`。  
> 产品热路径 = **dyn-CSR**（`DynCsrCtrl`）；新人主逻辑见 [`../QUICKSTART.md`](../QUICKSTART.md) §3。

## 中文

### 这个目录是干什么的？

单 INC **Combine 数据面**：逻辑计划 → 拓扑校验 → 可执行计划 → 对称 workspace →  
设备 kernel（producer / INC reduce / fan-back）。

### 为什么文件比 Dispatch 多？

- 要处理变长 CSR 贡献、加权归约、结果回传与挂死诊断。
- 逻辑计划与物理拓扑解耦；同一逻辑计划可编译到不同 channel/lane。
- CMake target/符号保留少量 `sv2` 名称以兼容已有脚本和报告；源文件只保留稳定名。

### 当前产品构建闭包（读这个就够）

| 文件 | 用途 | 为什么要有 | CMake |
|---|---|---|---|
| `inc_dc_combine_kernel.cpp` | producer、INC reduce、result fan-back 的设备 kernel | Combine 数据面本体 | **是**（target 名仍为 `inc_dc_sv2_dyn_csr_combine_kernel`） |
| `inc_dc_combine_launcher.cpp` | standalone 资格化 launcher | `run_single_inc_dyn_case.sh` 启动的 exe 源 | **是**（exe：`inc_dc_sv2_dyn_csr_combine`） |
| `inc_dc_combine_runtime_abi.h` | `DynCsrCtrl` 等 host/device ABI（magic `'DYCS'`） | 设备控制块真相源 | 被上述源 `#include` |
| `inc_dc_combine_vector_reduce_aicore.h` | FP16→FP32 加权向量归约 | 正式归约引擎；fail-closed | 被 kernel include |
| `inc_dc_combine_logical_plan.h` / `.cpp` | 拓扑无关逻辑计划（结构即 **V2**：uid/ordinal/weight…） | 计划语义与设备布局解耦 | **是** |
| `inc_dc_combine_plan_wire.h` / `.cpp` | 版本化 wire 编解码 | 工具/落盘/重放 | **是**（host exe / tools） |
| `inc_dc_combine_topology.h` / `.cpp` | worker→INC 可达性 / channel 校验 | 非法拓扑 fail-closed | **是** |
| `inc_dc_combine_plan_compiler.h` / `.cpp` | 逻辑计划 → 可执行 INC/channel 计划 | 物理绑定 | **是** |

### 产品路径澄清（避免误读）

| 说法 | 是否成立 |
|---|---|
| 当前 Framework / Easy / native Combine = **dyn-CSR** | **是** |
| BW05 packed 是当前 Easy 热路径 | **否** |
| CMake target 名叫 `sv2_dyn_csr` 所以源文件必须叫 sv2 | **否**（target 名遗留；源用稳定名） |
| 资格化必须走 `../scripts/single_inc/run_single_inc_dyn_case.sh` | **政策要求**（裸跑 bin 会绕过拓扑/空闲门禁，结果不作数） |

---

## English

### What is this directory?

Single-INC **Combine data path**: logical plan → topology → executable plan →
symmetric workspace → device kernels (producer / INC reduce / fan-back).

### Build truth

- **Compiled sources** use stable names `inc_dc_combine_*` (see
  `examples/inc/CMakeLists.txt`).
- **CMake target / binary names** may still say `inc_dc_sv2_dyn_csr_*` for
  report compatibility.
- Product hot path is **dyn-CSR** (`DynCsrCtrl`).
- See [`../QUICKSTART.md`](../QUICKSTART.md) §3 for the newcomer narrative.

### Product closure (abbreviated)

| File | Role | In CMake? |
|---|---|---|
| `inc_dc_combine_kernel.cpp` | Device data path | Yes (target still named `*_sv2_*_kernel`) |
| `inc_dc_combine_launcher.cpp` | Qual launcher → exe `inc_dc_sv2_dyn_csr_combine` | Yes |
| `inc_dc_combine_runtime_abi.h` | `DynCsrCtrl` ABI | Included by product sources |
| `inc_dc_combine_vector_reduce_aicore.h` | Weighted vector reduce | Included by kernel |
| `inc_dc_combine_logical_plan.*` | Topology-independent plan (V2 fields) | Yes |
| `inc_dc_combine_plan_wire.*` | Versioned wire codec | Yes (host/tools) |
| `inc_dc_combine_topology.*` | Reachability/channel checks | Yes |
| `inc_dc_combine_plan_compiler.*` | Logical → executable plan | Yes |
