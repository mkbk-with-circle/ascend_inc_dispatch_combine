#ifndef INC_COMBINE_TRACE_H
#define INC_COMBINE_TRACE_H

#include <cstdint>
#include <cstdlib>

namespace inc {
namespace dc {

// Host-visible combine bench config at sym trace region base (128B).
struct DcCombineTraceHeader {
    uint32_t magic = 0x43425448u; // CBTH
    uint32_t spin_cap = 2000000u;
    uint64_t expected_seq = 1u;
    uint32_t combine_flags = 0;
    uint32_t bench_warmup = 0;
    uint32_t bench_measure = 0;
    uint32_t use_seq_credit = 0;
    uint32_t poll_prefilled = 0;
    uint32_t token_batch = 1;
    uint32_t tile_bytes_kib = 0;
    uint32_t switch_block_base = 0;
    uint32_t pipeline_trace_enable = 0; // MC09 same-run stage ring
    uint32_t stage_ring_capacity = 0;   // per-AIV sub-ring capacity
    uint32_t switch_probe_level = 6;    // 0=formal-only, 1..5=A..E, 6=F
    // Worker ll phase: 0=upload+drain, 1=upload-only, 2=drain-only (paired MC09 staging).
    uint32_t worker_ll_phase = 0;
    // BW02 isolation on LL path (kCombineRoofline* values; 0/5 = e2e).
    uint32_t isolation_stage = 0;
    // BW03: env-gated upload lanes / ready queue (see INC_DC_COMBINE_UPLOAD_LANES / READY_QUEUE).
    uint32_t upload_lanes_enable = 0;
    uint32_t ready_queue_enable = 0;
    uint32_t bw03_meta_off = 0; // byte offset from poll_meta_base to IncDcCombineBw03DeviceLayout region
    uint32_t packed_channel_enable = 0; // INC_DC_COMBINE_PACKED_CHANNEL=1
    uint32_t bw05_meta_off = 0;         // byte offset from poll_meta_base to IncDcCombineBw05DeviceLayout region
    // pad[0]: device FUTURE_SEQ_COUNT (kernel increments; host must not repurpose).
    // pad[3]: BW05-K1 diag enable (1 when sidecar present).
    // pad[4]: BW05-K1 diag byte offset from poll_meta_base.
    // pad[5]: BW05-H2 progress enable (1 when sidecar present).
    // pad[6]: BW05-H2 sidecar byte offset from poll_meta_base.
    uint32_t pad[7] = {};
};

// Per-switch-block / per-rank summary (128B) — host reads outside timing window.
// BW02: summary counters only (O(1) per AIV per stage); no per-token writes here.
struct DcCombineDevTraceCell {
    uint64_t wait_ready_cycles = 0;
    uint64_t reduce_cycles = 0;
    uint64_t tx_cycles = 0;
    uint32_t spin_timeout_count = 0;
    uint32_t max_spin_observed = 0;
    uint32_t first_timeout_assignment = 0xFFFFFFFFu;
    uint32_t iterations_completed = 0;
    uint64_t last_makespan_us = 0;
    // BW02 summary (fixed writes, not in token hot loop):
    uint32_t payload_put_count = 0;
    uint32_t control_put_count = 0;
    uint32_t results_reduced = 0;
    uint32_t stage_event_writes = 0; // must be 0 when pipeline_trace_enable=0
    uint64_t payload_put_bytes = 0;
    uint64_t epoch_timing_filled = 0; // bid0: number of epoch slots written
    // PERF2-M breakdown counters (device-side; host sums across AIVs).
    uint64_t dcci_cycles = 0;
    uint32_t dcci_calls = 0;
    // PERF3-V: per-AIV path attestation (replaces future_seq_count; future stays in conservation).
    uint16_t vector_result_count = 0;
    uint16_t scalar_result_count = 0;
    uint64_t dcci_bytes = 0;
    uint64_t ring_ingress_copy_cycles = 0;
    uint64_t ring_ingress_copy_bytes = 0;
    uint64_t reduce_arith_cycles = 0;
    // Exactly 128B.
};
static_assert(sizeof(DcCombineDevTraceCell) == 128, "DevTraceCell must stay 128B");

// MC09: compact stage begin/end/progress event (32B) in per-AIV sub-ring.
struct DcCombineStageEventDev {
    uint32_t cycle_lo = 0;       // GetSystemCycle low 32, relative to kernel_start
    uint16_t stage = 0;          // 0..4 IncCombinePipelineStage
    uint16_t begin = 0;          // 1=begin, 0=end/progress
    uint32_t tile_id = 0;
    uint32_t op_seq_lo = 0;      // bench iter / epoch
    uint16_t global_block = 0;
    uint16_t physical_core_id = 0;
    uint32_t progress = 0;
    uint32_t result_slot = 0;
    uint32_t pad = 0; // keep 32B
};

// MC09 Phase C: per-AIV sub-ring header (16B each, indexed by global_block).
struct DcCombineStageSubRingHeader {
    uint32_t write_idx = 0;
    uint32_t overflow = 0;
    uint32_t dropped = 0;
    uint32_t corrupt = 0;
};

// MC09 Phase C: global ring header — aggregate invalidation + clock rendezvous.
struct DcCombineStageRingHeader {
    uint32_t magic = 0x43535248u; // CSRH
    uint32_t subring_capacity = 0; // slots per AIV / global_block
    uint32_t num_subrings = 0;     // max global_block + 1
    uint32_t enable = 0;
    uint32_t kernel_start_lo = 0;  // block-0 epoch for relative cycle_lo
    uint32_t pe_role = 0;          // 1=worker heap, 2=switch heap
    uint32_t overflow_count = 0;   // aggregate; any non-zero → trace_invalid
    uint32_t dropped_count = 0;
    uint32_t corrupt_count = 0;
    uint32_t clock_epoch_lo = 0;   // pre-measure rendezvous cycle (block 0)
    uint32_t clock_calibrated = 0; // 1 after epoch sampled
    uint32_t capacity = 0;         // legacy flat-ring alias (device kernels)
    uint32_t write_idx = 0;        // legacy flat-ring monotonic index
    uint32_t pad[3] = {};
};

constexpr uint16_t kDcCombinePhysicalCoreIdUnknown = 0xFFFFu;
constexpr uint64_t kDcCombineRecvHiddenPoisonHalf = 0xDEADu;

constexpr uint64_t kDcCombineTraceHeaderBytes = 128;
constexpr uint64_t kDcCombineTraceCellBytes = 128;
constexpr uint32_t kDcCombineMaxTraceBlocks = 40;

// BW04: per-PE per-AIV per-epoch start/end (device cycles). Not stage-event ring.
constexpr uint32_t kDcCombineMaxBenchEpochs = 64;
constexpr uint32_t kDcCombineMaxAivTimingBlocks = kDcCombineMaxTraceBlocks; // >= C20
struct DcCombineAivEpochRecord {
    uint64_t start_cycle = 0;
    uint64_t end_cycle = 0;
    uint32_t generation = 0;
    uint16_t block_id = 0;
    uint16_t active = 0; // 1 if this AIV participated in the epoch
};
static_assert(sizeof(DcCombineAivEpochRecord) == 24, "AivEpochRecord size");

struct DcCombineEpochTimingHeader {
    uint32_t magic = 0x43455448u; // CETH
    uint32_t warmup = 0;
    uint32_t measure = 0;
    uint32_t filled = 0; // epochs written (0..warmup+measure)
    uint64_t aggregate_elapsed_cycles = 0; // sum over measured of pe_max(end-start)
    uint32_t pipeline_trace_enable = 0;
    uint32_t pe_role = 0; // 1=worker, 2=switch
    uint32_t block_num = 0;
    uint32_t max_blocks = kDcCombineMaxAivTimingBlocks;
    uint32_t calib_cycles = 0; // independent busy-loop cycles (device)
    uint32_t schema = 2;      // 2 = per-AIV start/end records
    // Followed by DcCombineAivEpochRecord[max_blocks][kDcCombineMaxBenchEpochs] at +64.
};
constexpr uint64_t kDcCombineStageRingHeaderBytes = 64;
constexpr uint64_t kDcCombineStageSubRingHeaderBytes = 16;
constexpr uint64_t kDcCombineStageEventBytes = 32;
// Sized for staged MC09: worker upload+drain (~1.5k) and switch rx/reduce/tx (~1.5k) without overflow.
constexpr uint32_t kDcCombineStageSubRingCapacity = 2048;
// Legacy alias: total event slots across all sub-rings.
constexpr uint32_t kDcCombineStageRingCapacity =
    kDcCombineMaxTraceBlocks * kDcCombineStageSubRingCapacity;

constexpr uint64_t kDcCombineStageRingHeaderOff =
    kDcCombineTraceHeaderBytes +
    static_cast<uint64_t>(kDcCombineMaxTraceBlocks) * kDcCombineTraceCellBytes;
constexpr uint64_t kDcCombineStageSubRingHeadersOff =
    kDcCombineStageRingHeaderOff + kDcCombineStageRingHeaderBytes;
constexpr uint64_t kDcCombineStageEventBaseOff =
    kDcCombineStageSubRingHeadersOff +
    static_cast<uint64_t>(kDcCombineMaxTraceBlocks) * kDcCombineStageSubRingHeaderBytes;

// MC09 Phase B: independent switch AIV probe slot (after trace region, 128B).
constexpr uint32_t kDcCombineSwitchProbeMagic = 0x44435350u; // DCSP
constexpr uint32_t kDcCombineSwitchProbeAccepted = 0xACCEu;
constexpr uint32_t kDcCombineSwitchProbeAivMarker = 0xA1A0u;
constexpr uint32_t kDcCombineSwitchProbeAicMarker = 0xA1C0u;

struct DcCombineSwitchProbe {
    uint32_t magic = 0;
    uint32_t level = 0;
    int32_t pe = -1;
    int32_t block_idx = -1;
    int32_t block_num = -1;
    uint32_t group_size = 0;
    uint32_t switch_rank = 0;
    uint32_t local_rank = 0;
    uint32_t result_n = 0;
    uint32_t contribution_n = 0;
    uint32_t accepted_magic = 0;
    uint32_t write_idx = 0;
    uint32_t overflow = 0;
    uint32_t aiv_marker = 0;
    uint32_t aic_marker = 0;
    uint32_t pad[18] = {};
};

constexpr uint64_t kDcCombineSwitchProbeBytes = 128;

// Device-safe layout offsets (avoid host-only inline helpers in __aicore__).
constexpr uint64_t kDcCombineSwitchProbeOff =
    kDcCombineStageEventBaseOff +
    static_cast<uint64_t>(kDcCombineMaxTraceBlocks) * kDcCombineStageSubRingCapacity * kDcCombineStageEventBytes;
constexpr uint64_t kDcCombineEpochTimingOff = kDcCombineSwitchProbeOff + kDcCombineSwitchProbeBytes;
constexpr uint64_t kDcCombineEpochTimingHeaderBytes = 64;
constexpr uint64_t kDcCombineEpochTimingSlotsBytes =
    static_cast<uint64_t>(kDcCombineMaxAivTimingBlocks) * kDcCombineMaxBenchEpochs *
    sizeof(DcCombineAivEpochRecord);

inline uint64_t DcCombineAivEpochRecordOffset(uint32_t block_idx, uint32_t epoch_idx)
{
    return kDcCombineEpochTimingOff + kDcCombineEpochTimingHeaderBytes +
           (static_cast<uint64_t>(block_idx) * kDcCombineMaxBenchEpochs + epoch_idx) *
               sizeof(DcCombineAivEpochRecord);
}

inline uint64_t DcCombineStageRingHeaderOffset()
{
    return kDcCombineStageRingHeaderOff;
}

inline uint64_t DcCombineStageSubRingHeaderOffset(uint32_t block_idx)
{
    return kDcCombineStageSubRingHeadersOff +
           static_cast<uint64_t>(block_idx) * kDcCombineStageSubRingHeaderBytes;
}

inline uint64_t DcCombineStageEventOffset(uint32_t block_idx, uint32_t idx_in_subring)
{
    return kDcCombineStageEventBaseOff +
           (static_cast<uint64_t>(block_idx) * kDcCombineStageSubRingCapacity + idx_in_subring) *
               kDcCombineStageEventBytes;
}

inline uint64_t DcCombineStageEventOffset(uint32_t idx)
{
    return DcCombineStageEventOffset(0u, idx);
}

inline uint64_t DcCombineSwitchProbeOffset()
{
    return kDcCombineSwitchProbeOff;
}

inline uint64_t DcCombineEpochTimingOffset()
{
    return kDcCombineEpochTimingOff;
}

// Legacy alias (BW02 single elapsed slot); prefer DcCombineAivEpochRecordOffset.
inline uint64_t DcCombineEpochElapsedOffset(uint32_t epoch_idx)
{
    return DcCombineAivEpochRecordOffset(0, epoch_idx);
}

inline uint64_t DcCombineTraceRegionBytes()
{
    return kDcCombineEpochTimingOff + kDcCombineEpochTimingHeaderBytes + kDcCombineEpochTimingSlotsBytes;
}

// Host-side MC09 ring read metadata (overlap gate / trace_invalid).
struct DcCombineStageRingMeta {
    bool trace_invalid = false;
    uint32_t overflow_count = 0;
    uint32_t dropped_count = 0;
    uint32_t corrupt_count = 0;
    uint32_t kernel_start_lo = 0;
    uint32_t clock_epoch_lo = 0;
    uint32_t switch_rx_event_count = 0;
    uint32_t switch_reduce_event_count = 0;
    uint32_t switch_tx_event_count = 0;
    bool worker_drain_from_completion_doorbell = false;
};

// Parse INC_DC_SWITCH_PROBE_LEVEL: 0=formal-only, A..F=ladder step, default F.
inline uint32_t ParseSwitchProbeLevelFromEnv()
{
    const char *raw = std::getenv("INC_DC_SWITCH_PROBE_LEVEL");
    if (raw == nullptr || raw[0] == '\0' || (raw[0] == 'F' && raw[1] == '\0') || raw[0] == 'f') {
        return 6u;
    }
    if (raw[0] == '0' && raw[1] == '\0') {
        return 0u;
    }
    if (raw[1] == '\0') {
        const char c = raw[0] >= 'a' && raw[0] <= 'z' ? static_cast<char>(raw[0] - 'a' + 'A') : raw[0];
        if (c >= 'A' && c <= 'F') {
            return static_cast<uint32_t>(c - 'A' + 1u);
        }
    }
    return 6u;
}

inline uint64_t DcCombineTraceCellOffset(uint32_t block_idx)
{
    return kDcCombineTraceHeaderBytes + static_cast<uint64_t>(block_idx) * kDcCombineTraceCellBytes;
}

inline bool DcCombineTraceBlockIndexValid(uint32_t block_idx)
{
    return block_idx < kDcCombineMaxTraceBlocks;
}

inline bool DcCombineStageRingTraceInvalid(const DcCombineStageRingHeader &rh)
{
    return rh.overflow_count != 0u || rh.dropped_count != 0u || rh.corrupt_count != 0u;
}

// Host-testable: aggregate per-switch-block trace cells (C12+).
inline DcCombineDevTraceCell AggregateCombineTraceCells(const DcCombineDevTraceCell *cells, int block_dim)
{
    DcCombineDevTraceCell out{};
    if (cells == nullptr || block_dim <= 0) {
        return out;
    }
    const int n = block_dim > static_cast<int>(kDcCombineMaxTraceBlocks) ? static_cast<int>(kDcCombineMaxTraceBlocks)
                                                                          : block_dim;
    for (int i = 0; i < n; ++i) {
        const DcCombineDevTraceCell &c = cells[i];
        out.wait_ready_cycles += c.wait_ready_cycles;
        out.reduce_cycles += c.reduce_cycles;
        out.tx_cycles += c.tx_cycles;
        out.spin_timeout_count += c.spin_timeout_count;
        if (c.max_spin_observed > out.max_spin_observed) {
            out.max_spin_observed = c.max_spin_observed;
        }
        if (c.first_timeout_assignment != 0xFFFFFFFFu) {
            if (out.first_timeout_assignment == 0xFFFFFFFFu ||
                c.first_timeout_assignment < out.first_timeout_assignment) {
                out.first_timeout_assignment = c.first_timeout_assignment;
            }
        }
        out.iterations_completed += c.iterations_completed;
        if (c.last_makespan_us > out.last_makespan_us) {
            out.last_makespan_us = c.last_makespan_us;
        }
        out.payload_put_count += c.payload_put_count;
        out.control_put_count += c.control_put_count;
        out.results_reduced += c.results_reduced;
        out.stage_event_writes += c.stage_event_writes;
        out.payload_put_bytes += c.payload_put_bytes;
        if (c.epoch_timing_filled > out.epoch_timing_filled) {
            out.epoch_timing_filled = c.epoch_timing_filled;
        }
        out.dcci_cycles += c.dcci_cycles;
        out.dcci_calls += c.dcci_calls;
        {
            const uint32_t vv = static_cast<uint32_t>(out.vector_result_count) + c.vector_result_count;
            out.vector_result_count = static_cast<uint16_t>(vv > 0xFFFFu ? 0xFFFFu : vv);
            const uint32_t ss = static_cast<uint32_t>(out.scalar_result_count) + c.scalar_result_count;
            out.scalar_result_count = static_cast<uint16_t>(ss > 0xFFFFu ? 0xFFFFu : ss);
        }
        out.dcci_bytes += c.dcci_bytes;
        out.ring_ingress_copy_cycles += c.ring_ingress_copy_cycles;
        out.ring_ingress_copy_bytes += c.ring_ingress_copy_bytes;
        out.reduce_arith_cycles += c.reduce_arith_cycles;
    }
    return out;
}

// Host-testable: result-token shard indices for switch block bid (C12+).
inline void EnumerateSwitchResultShard(uint32_t result_n, int bid, int bnum, uint32_t *out_indices,
                                       uint32_t *out_count, uint32_t max_out)
{
    if (out_count != nullptr) {
        *out_count = 0;
    }
    if (out_indices == nullptr || out_count == nullptr || bid < 0 || bnum <= 0 || max_out == 0) {
        return;
    }
    uint32_t cnt = 0;
    for (uint32_t rt = static_cast<uint32_t>(bid); rt < result_n; rt += static_cast<uint32_t>(bnum)) {
        if (cnt >= max_out) {
            break;
        }
        out_indices[cnt++] = rt;
    }
    *out_count = cnt;
}

inline bool ValidateSwitchResultShardCover(uint32_t result_n, int bnum)
{
    if (bnum <= 0 || static_cast<uint32_t>(bnum) > kDcCombineMaxTraceBlocks) {
        return false;
    }
    uint32_t total = 0;
    for (int b = 0; b < bnum; ++b) {
        uint32_t dummy[8192];
        uint32_t cnt = 0;
        EnumerateSwitchResultShard(result_n, b, bnum, dummy, &cnt, 8192);
        total += cnt;
    }
    return total == result_n;
}

inline uint64_t DcCombineEncodeSeq(uint32_t epoch, uint32_t step)
{
    return (static_cast<uint64_t>(epoch + 1u) << 32) | (static_cast<uint64_t>(step + 1u) & 0xFFFFFFFFu);
}

} // namespace dc
} // namespace inc

#endif
