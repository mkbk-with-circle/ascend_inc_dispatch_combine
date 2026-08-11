# INC examples

This directory has two explicit ownership domains:

- `dispatch_combine/` is the maintained, CMake-built Single-INC and Multi-INC
  communication delivery, including public APIs, tests, and qualification
  scripts.
- `fusion_kernel/` is reference-only integration research and is not part of
  the INC transport build.

`CMakeLists.txt` intentionally builds only `dispatch_combine/` — **`fusion_kernel/`
is not a build target here**. Shared FP16, UB, vector-reduction, gather,
topology, and resource-policy primitives live in
`dispatch_combine/common/platform/`; there is no legacy source layer in this
directory root.

Historical AG/RS and `inc_s09` prototypes are not part of the maintained
dependency closure. Current evidence and hardware profiles live under
`docs/inc/`.

## New here? / 新人从这里开始

If you have **never touched this tree** and need a trustworthy mental model of
**single-INC Dispatch and Combine** (topology, device timelines, planning,
runtime wiring, how to run one case), read:

**[`dispatch_combine/single_inc/QUICKSTART.md`](dispatch_combine/single_inc/QUICKSTART.md)**

中文主文 + English summary；结论均可对照源码符号（`StreamWorker` /
`StreamInc` / `DynCsrCtrl` / `NativeIncService` 等）。

**Current single-INC sweep progress / environments / baseline gates:**  
[`dispatch_combine/single_inc/SWEEP_STATUS.md`](dispatch_combine/single_inc/SWEEP_STATUS.md)

Then skim `dispatch_combine/README.md` for how `common/`, `single_inc/`,
`multi_inc/`, `scripts/`, and `tests/` split ownership.
