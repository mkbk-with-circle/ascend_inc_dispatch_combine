# Single-INC Combine / 单 INC Combine

> **构建真相以** `examples/inc/CMakeLists.txt` **为准。**  
> CMake target 仍可能叫 `inc_dc_sv2_dyn_csr_*`（二进制/报告兼容），但**编译进库的源文件是稳定名** `inc_dc_combine_*`。  
> 磁盘上若还有 `inc_dc_sv2_*` / `*_logical_plan_v2*` 等近副本，**默认未挂接构建**——不要当成第二套「必需数据面」。  
> 产品热路径 = **dyn-CSR**（`DynCsrCtrl`）；新人主逻辑见 [`../QUICKSTART.md`](../QUICKSTART.md) §3。

## 中文

### 这个目录是干什么的？

单 INC **Combine 数据面**：逻辑计划 → 拓扑校验 → 可执行计划 → 对称 workspace →  
设备 kernel（producer / INC reduce / fan-back），并附带活性/谱系诊断 sidecar。

### 为什么文件比 Dispatch 多？

- 要处理变长 CSR 贡献、加权归约、结果回传与挂死诊断。
- 逻辑计划与物理拓扑解耦；同一逻辑计划可编译到不同 channel/lane。
- sidecar 把进度/谱系放在正式 schema ownership 之外，避免污染布局。
- 历史实验曾留下 bw03/bw05/sv2 命名；稳定名重构后，**部分旧文件名仍留在磁盘**。

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
| `inc_dc_combine_symmetric_layout.h` | standalone launcher 有界对称 workspace | 跨 rank 缓冲摆放 | host 路径依赖 |
| `inc_dc_combine_upload_transport.h` | upload lane/ring 基础定义 | 历史上行契约；部分常量仍被引用 | 间接 |
| `inc_dc_combine_liveness_sidecar.h` | 活性 / abort / 故障归因（稳定命名） | 挂死诊断入口 | 资格化保留 |
| `inc_dc_combine_lineage_sidecar.h` | top-k=1 谱系诊断（稳定命名） | 定位哪一跳写坏 | 可编译选项；资格化保留 |
| `inc_combine_trace.h` | timing 窗外 host-visible 计数 | launcher/报告依赖 | 资格化依赖 |
| `inc_combine_bw03.h` | 固定 upload lane / batch·ready ring 等常量 | **间接依赖底座**（非独立产品路径） | 被其它头 include |
| `inc_combine_bw05.h` / `inc_dc_combine_packed_transport.h` | 历史 packed SPSC 布局 | **不是** Framework/native 热路径；native 走 dyn-CSR | 间接/历史 |
| `inc_combine_plan.h` | 含 legacy device meta / 转换辅助 | host 侧 legacy→逻辑计划等；**设备热路径读的是 `DynCsrCtrl`** | 间接 |
| `inc_combine_bw05_h2_sidecar.h` | live-progress + 可控 abort（magic `B2H2`） | 实现细节；稳定入口见 `inc_dc_combine_liveness_sidecar.h` | 诊断 |
| `inc_combine_bw05_k1_diag_sidecar.h` | top-k=1 多阶段谱系采样 | debug；稳定入口见 lineage sidecar | 诊断 |

### 磁盘上存在、但默认未进当前 CMake 的近副本

下列文件**可能仍在目录里**，用于对照或尚未删除的镜像；**不要**当作第二套必需运行时：

| 文件 | 说明 |
|---|---|
| `inc_dc_sv2_dyn_csr_combine_kernel.cpp` | 相对 `inc_dc_combine_kernel.cpp` 的历史近副本；**CMake 不编它** |
| `inc_dc_sv2_dyn_csr_combine_main.cpp` | 相对 `inc_dc_combine_launcher.cpp` 的历史近副本；**脚本跑的是 launcher 编出的 exe** |
| `inc_dc_sv2_dyn_csr_combine.h` | 相对 `inc_dc_combine_runtime_abi.h` 的历史镜像 |
| `inc_dc_sv2_c0_vector_reduce_aicore.h` | 相对 `inc_dc_combine_vector_reduce_aicore.h` 的历史镜像 |
| `inc_dc_combine_logical_plan_v2.*` / `inc_dc_combine_logical_plan_wire_v2.*` | 未挂接的 v2 分文件镜像；**当前构建入口已是** `inc_dc_combine_logical_plan.*`（内容即 V2） |

若不确定：在 `examples/inc/CMakeLists.txt` 里搜文件名——搜不到就不在交付构建里。

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
symmetric workspace → device kernels (producer / INC reduce / fan-back), plus
liveness/lineage sidecars.

### Build truth

- **Compiled sources** use stable names `inc_dc_combine_*` (see
  `examples/inc/CMakeLists.txt`).
- **CMake target / binary names** may still say `inc_dc_sv2_dyn_csr_*` for
  report compatibility.
- On-disk `inc_dc_sv2_*` / `*_v2*` near-copies are **not** the default build
  inputs unless listed in CMake.
- Product hot path is **dyn-CSR** (`DynCsrCtrl`). BW05/packed headers are
  historical/indirect—not the Framework/native hot path.
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
| `inc_combine_bw05.h` / packed transport | Legacy packed layout | Indirect only |
| `inc_dc_sv2_*` / `*_v2*` siblings | Unwired near-copies | **No** (unless CMake lists them) |
