# Yuan single-INC + current fusion delivery

> 当前唯一源码真源是本仓库的 `shmem` worktree。历史目录
> `shmem-hybrid-yuan-single-current-fusion` 只是等价提交的旧 worktree，
> `build-hybrid-yuan-single-current-fusion` 只是由该旧 worktree 生成的 CMake
> 构建缓存；二者都不是运行或继续开发所需的源码入口。

This branch intentionally combines these qualified components:

| Component | Source |
|---|---|
| Single-INC dispatch | yuanmingyu run `20260812T174225`, commit `e06ce80d875a3509c9660d20b7a2ec0f6ad68c2f` |
| Single-INC combine | yuanmingyu run `20260812T174225`, commit `e06ce80d875a3509c9660d20b7a2ec0f6ad68c2f` |
| Fusion kernel/API/runtime | current nb branch at assembly time, commit `e06ce80d875a3509c9660d20b7a2ec0f6ad68c2f` |
| CATLASS base | `7d4c8401ae2b2aeb8a5786671e4fe7f53ca96c18` |
| CATLASS fusion fix | tracked `catlass_grouped_matmul_zero_m.patch` in this repository |

The main-repository commits happen to be the same today, so no source-level
cherry-pick is required. The important assembly fix is making the previously
implicit CATLASS dependency and its sparse zero-M guard reproducible.

## Correctness-only validation

On an idle nb host:

```bash
source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh
examples/inc/run_yuan_single_current_fusion_smoke.sh /tmp/hybrid-smoke
```

The smoke builds the three delivery targets and runs five small W2 cases on
one HCCS plane:

- dispatch K1 and K2;
- combine K1 and K2;
- fusion `T17/H192/I320/K1/A2`.

It does not run a throughput sweep and does not update performance gates.
The yuanmingyu Phase-A data remains the performance reference for single-INC;
nb only establishes buildability, topology placement, launch, and all-rank
correctness for this assembled source tree.

The recorded nb smoke result is under
`docs/inc/report/nb-borrow/hybrid_yuan_single_current_fusion_smoke_20260813/`.

## Fresh-clone preparation

The CATLASS checkout is intentionally not committed to this repository.  A
fresh clone is prepared from the pinned upstream commit and the tracked patch:

```bash
examples/inc/fusion_kernel/tools/prepare_catlass_dependency.sh
```

The command is idempotent and verifies both the patch and resulting header by
SHA-256.  The correctness smoke invokes it automatically before configuring
CMake.

Only Git-tracked files need to be transferred to another host.  Do not copy
`3rdparty/`, build directories, test output, device logs, or local Git bundles;
they are generated or machine-local.  After transfer, verify the checkout with:

```bash
git status --short
git rev-parse HEAD
```

The expected result for a transferred delivery is a clean canonical `shmem`
worktree.  A separate `hybrid-yuan-single-current-fusion-20260812` worktree or
its build directory is not required.  Use the repository preparation script
on the destination before building the fusion target.
