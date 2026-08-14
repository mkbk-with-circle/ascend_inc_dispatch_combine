# nb-borrow single-INC scaled-gate requalification (2026-08-06)

This directory contains **only** measurements from `nb-borrow` (16 x
Ascend 910B2C, two 8-card HCCS planes).  It does not replace or modify the
`yuanmingyu` evidence under `docs/inc/hardware_profiles/910b-yuanmingyu`.

## Scope and topology contract

- INC: Phy 0.
- W2 workers: Phy 1,2.  W4 workers: Phy 1,2,3,4.
- Every selected INC/worker pair was rechecked live as HCCS before launch.
- W8 is deliberately unconfigured on this profile because 8W+1INC cannot fit
  in one 8-card HCCS plane.
- Every launcher held `/tmp/inc_single_inc_npu.lock` and checked all NPUs idle
  before and after the run.

The production target remains the `yuanmingyu`-like star: equivalent worker
links meet one switch and the INC endpoint link is the bottleneck.  The nb
56/112-GB/s values below are local qualification ceilings only; no kernel or
protocol reads those numbers or assumes per-peer 28 GB/s links.

## Frozen gate conversion

Old roofline convention: 192 GB/s.  The user-specified nb raw ceilings are
56 GB/s (W2) and 112 GB/s (W4), hence:

```
r_W2 = 56 / 192 = 0.2916666667
r_W4 = 112 / 192 = 0.5833333333
```

Dispatch preserves the old active-roofline formula
`0.93 * 123.339 / (1 + 0.127 / R)`, where `R=min(topk, workers)`, and then
multiplies by the scale factor.  General Combine preserves the old formal
120-GB/s gate and scales it.  Identity K1 is a duplex relay (one output byte
for every input byte), so it is compared with 90% of the simultaneously
measured ingress/egress roofline rather than the reduction-only ingress gate.

| Case | Old gate (GB/s) | nb gate (GB/s) |
|---|---:|---:|
| W2 Dispatch R1 | 101.779 | 29.686 |
| W2 Dispatch R2 | 107.856 | 31.458 |
| W4 Dispatch R1 | 101.779 | 59.371 |
| W4 Dispatch R2 | 107.856 | 62.916 |
| W4 Dispatch R4 | 111.175 | 64.852 |
| W2 Combine K>=2 | 120.000 | 35.000 |
| W4 Combine K>=2 | 120.000 | 70.000 |

## Independent put-only rooflines

Formal single-direction runs use 128 MiB/peer, H3+H10.  Values are aggregate
physical-direction bandwidth using the slow-rank makespan.

| Scale | Direction | min | mean | CV |
|---|---|---:|---:|---:|
| W2 | all -> INC | 42.721 | 42.733 | 0.0189% |
| W2 | INC -> all | 42.802 | 42.813 | 0.0148% |
| W4 | all -> INC | 85.442 | 85.475 | 0.0201% |
| W4 | INC -> all | 83.218 | 83.305 | 0.0619% |

Source: `roofline/roofline_results.json`.

The simultaneous duplex probe measured 37.626 GB/s per direction for W2 and
73.271 GB/s (the slower direction) for W4.  Repeating it with the operator's
AIV cohorts (W2 I24/E24, W4 I16/E12) produced the same ceilings.  Therefore
the strict K1 gates are 33.864 GB/s (W2) and 65.944 GB/s (W4).  Evidence is
under `duplex_roofline/` and `duplex_roofline_matched_aiv/`.

## Accepted protocol changes

1. Full-rank Dispatch chooses resident epochs from the live AIV cohort, the
   source/destination stream grid, and the bounded 2-upload x 2-TX credits.
   It contains no W2/W4 table and no nb bandwidth constant.
2. When that grid exactly occupies the TX cohort, one full runtime packet is
   resident in every credit state.  This removed short-train fill/drain
   overhead: W4/K4/128 MiB rose from 57.012 to 69.277 GB/s mean.
3. Identity-K1 producer pair-ready is valid for every worker count; the old
   `worker_count==2` restriction was removed.
4. K1 relay slots split the compiler-qualified 24-KiB AIV UB budget evenly,
   rather than fixing two 8-KiB slots from an old machine.
5. Identity-K1 owner count is derived as `ceil(live_AIV/workers)`.
   Weighted K1 and K>=2 retain the generic arbitrary-CSR reducer.  The
   source-group worklist remains diagnostic-only: its kernel branch is not a
   production generation-credit path, so the final binary no longer reports
   it as active merely because a plan is identity K1.

Rejected candidates and their logs are retained under `tuning/` and the
`protocol_*_check` directories.  In particular, extra K4 upload credits,
asymmetric K1 UB slots, four-row delayed ready, packed two-phase K1, and
mode-5 ordered streaming all regressed at least one case and are not defaults.

## Final-matrix evidence and row-policy correction

`final_matrix_v17/` contains 24/24 correct, rc=0 executions (W2/W4,
K1/K2/K4, 128/256 MiB).  It is retained as robustness evidence.  It also
exposed a sweep-planner bug: after selecting the transport-qualified 16-KiB
identity-K1 row, the general short-train rule could overwrite W4/K1 with a
32-KiB row.  The planner now requires `K > 1` before applying that general
rule.  No operator path is selected from this qualification row policy.

The corrected W4/K1 formal cases are in `formal_k1_row_policy_v18/`.  The
32-KiB results remain in v17 and are not deleted or relabelled as formal.

| Operator | Scale/K | Bytes | Measured GB/s | Scaled gate | Result |
|---|---|---:|---:|---:|---|
| Dispatch | W2/K1 | 128 / 256 MiB | 33.441 / 34.902 | 29.686 | PASS / PASS |
| Dispatch | W2/K2 | 128 / 256 MiB | 39.464 / 39.593 | 31.458 | PASS / PASS |
| Dispatch | W4/K1 | 128 / 256 MiB | 62.178 / 64.316 | 59.371 | PASS / PASS |
| Dispatch | W4/K2 | 128 / 256 MiB | 65.090 / 62.915 | 62.916 | PASS / **FAIL** |
| Dispatch | W4/K4 | 128 / 256 MiB | 74.132 / 78.068 | 64.852 | PASS / PASS |
| Combine | W2/K1 | 128 / 256 MiB | 36.391 / 36.874 | 33.864 duplex | PASS / PASS |
| Combine | W2/K2 | 128 / 256 MiB | 37.426 / 39.056 | 35.000 | PASS / PASS |
| Combine | W4/K1 | 128 / 256 MiB | 64.448 / 69.511 | 65.944 duplex | **FAIL** / PASS |
| Combine | W4/K2 | 128 / 256 MiB | 70.967 / 74.936 | 70.000 | PASS / PASS |
| Combine | W4/K4 | 128 / 256 MiB | 73.813 / 77.904 | 70.000 | PASS / PASS |

K4 on W2 is an additional K>worker stress case, not part of the requested
`topk <= rank` matrix; it is nevertheless correct and above the K>=2 gate at
both sizes.  W4 Dispatch K2/256 is deliberately classified FAIL even though
it misses by only about 0.002 GB/s.  An independent rerun measured 62.518
GB/s, so the result is not rounded or cherry-picked into a pass.  A
live-lane-derived 16-epoch candidate regressed further to 61.664 GB/s and was
reverted; its log is retained under `protocol_live_epoch_bound_v20/`.

Two protocol hypotheses for 32-KiB K1 (runtime-row pair-ready and a single
full-budget UB credit) were also correct but slower, and were reverted.  Their
logs remain under `protocol_k1_runtime_row_pair_ready_v14/` and
`protocol_k1_full_budget_credit_v15/`.

The final sources build successfully and pass the host-only stream-plan
compiler, single-INC-combine plan, resource-policy, dynamic-plan, and
CSR-reduction tests in `build-nb-cann91`.  The sweep script passes `bash -n`.

## Current decision

- Correctness/process stability is 24/24 for the retained matrix; W2 and all
  W4 K>=2 Combine cases meet the scaled gates.
- The campaign is **not fully gate-complete**: corrected W4 Combine K1 at
  128 MiB is 1.496 GB/s below its strict duplex gate, and W4 Dispatch K2 at
  256 MiB has a small but reproducible shortfall.
- The remaining protocol work is streaming coalescence across contiguous K1
  source/destination runs without the failed two-phase pack, plus reducing
  W4/K2 Dispatch repeat variance without sacrificing its 32-epoch overlap.
- `final_matrix_v12/` remains preserved as the earlier safely interrupted
  run; no old yuanmingyu or nb result was overwritten.
