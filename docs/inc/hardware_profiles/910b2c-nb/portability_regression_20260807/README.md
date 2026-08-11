# single-INC portability refactor regression — 910b2c-nb (2026-08-07)

This directory contains only `npu-borrow` / `910b2c-nb` measurements.  It
does not replace or rewrite the retained `910b-yuanmingyu` evidence.

## Change boundary

The refactor deliberately left the qualified data protocol unchanged:

- Dispatch now uses the shared strict PE-to-physical-NPU resolver.
- 24-KiB AIV UB, 16-KiB private-MTE packet, and ABI lane/owner ceilings have
  one compile-time capability source.  Current values compile to the same
  constants as before.
- resource policy version 1 preserves the qualified `1:2` Dispatch/Combine
  INC split; optional weights/reserved AIV are control-plane inputs only.
- registration logs contain capability version, resource-policy fingerprint,
  physical-map digest, UB/packet capabilities, and ABI-cap diagnostics.
- the sweep accepts an explicit hardware-profile gate, a configurable gate
  size, and a legal-top-k-only matrix.  No gate value is visible to a kernel.

The complete product host gate and both real device targets built and passed
before this sweep.

## Sweep contract

- topology: INC Phy0; W2 workers Phy1/2; W4 workers Phy1/2/3/4; every edge
  revalidated as HCCS before every case;
- matrix: Dispatch + Combine, W2 `{K1,K2}`, W4 `{K1,K2,K4}`;
- logical sizes: 64, 128, 256 MiB and 1 GiB;
- warmup 3, repeat 10; `K>W` is explicitly skipped;
- 128 MiB and above are performance-gated; 64 MiB is correctness/stability;
- all launches hold `/tmp/inc_single_inc_npu.lock` and check NPU idle before
  and after each case.

The nb gates are the frozen scaled gates from `scaled_gate_requal_20260806`:
Dispatch uses the old 123.339-GB/s anchor times 56/192 (W2) or 112/192 (W4);
Combine K>=2 is 35/70 GB/s; K1 duplex is 33.864/65.944 GB/s.

## Result

| Scope | Result |
|---|---:|
| Executions | **40/40 correct, rc=0** |
| Non-gated 64-MiB rows | **10/10 correct** |
| Performance-gated rows | **29/30 PASS** |
| Dispatch gated rows | **15/15 PASS** |
| Combine gated rows | **14/15 PASS** |
| Max Dispatch gated repeat CV | **2.608%** (H11 PASS) |
| Timeout / hang / mismatch | **0 / 0 / 0** |

Representative retained values:

| Operator | Case | 128 MiB | 256 MiB | 1 GiB | Gate |
|---|---|---:|---:|---:|---:|
| Dispatch | W2/K1 | 33.467 | 34.779 | 34.868 | 29.686 |
| Dispatch | W2/K2 | 39.462 | 39.591 | 39.594 | 31.458 |
| Dispatch | W4/K1 | 62.179 | 64.020 | 64.411 | 59.371 |
| Dispatch | W4/K2 | 66.216 | 63.258 | 63.812 | 62.916 |
| Dispatch | W4/K4 | 74.182 | 78.064 | 78.120 | 64.852 |
| Combine | W2/K1 | 36.464 | 36.863 | 36.883 | 33.864 |
| Combine | W2/K2 | 37.530 | 39.097 | 39.106 | 35.000 |
| Combine | W4/K1 | **62.676 FAIL** | 69.371 | 69.636 | 65.944 |
| Combine | W4/K2 | 71.295 | 75.085 | 76.189 | 70.000 |
| Combine | W4/K4 | 74.037 | 78.084 | 78.251 | 70.000 |

The sole performance failure is the already-known W4/K1 128-MiB marginal
case.  Three independent follow-up executions are retained under
`recheck_W4K1_128/`: 66.147, 67.419, 65.050 GB/s (mean 66.205, CV 1.462%).
Including the original 62.676 sample gives mean 65.323 and CV 2.668%.  The
original failure is not erased or relabelled; the evidence says there is no
systematic refactor regression, while this boundary remains marginal.

Every Dispatch and Combine registration within a scale reported one common
capability/policy/map identity:

| W | policy fingerprint | physical-map digest | UB | private MTE packet |
|---|---:|---:|---:|---:|
| 2 | 7961044054078796256 | 11570871483800784260 | 24576 | 16384 |
| 4 | 5872072434342665794 | 4733234321614906753 | 24576 | 16384 |

Machine-readable truth is `report.json`; raw per-rank logs are below
`dispatch/`, `combine/`, and `recheck_W4K1_128/`.
