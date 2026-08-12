# Yuan single-INC + current fusion delivery

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
