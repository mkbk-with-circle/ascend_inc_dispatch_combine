# Hybrid yuan single-INC + current fusion smoke

- Date: 2026-08-13
- Host: `nb-borrow`
- Tested main-repository commit: `92c9959f4cc6685873a3d6722a8ecf116c9bb3e7`
- Single-INC source: `e06ce80d875a3509c9660d20b7a2ec0f6ad68c2f`
- Fusion source: `e06ce80d875a3509c9660d20b7a2ec0f6ad68c2f`
- CATLASS source: `7d4c8401ae2b2aeb8a5786671e4fe7f53ca96c18`
- CANN: `/usr/local/Ascend/cann-9.1.0-beta.3`
- Placement: INC on NPU 0; workers on NPU 1 and 2; both links reported HCCS
- Final status: `PASS`
- Performance qualified: no

The smoke rebuilt `inc_dc_single_inc_stream`,
`inc_dc_sv2_dyn_csr_combine`, and `inc_fusion_e2e`, then ran five small
correctness cases. Dispatch K1/K2 and combine K1/K2 each completed with all
three expected ranks passing and process return code 0. The W2 fusion case
also reported `PASS`.

Any timing printed incidentally by the smoke is not a performance result and
must not replace the qualified yuanmingyu single-INC sweep data.
