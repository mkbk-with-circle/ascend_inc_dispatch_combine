#ifndef INC_DC_COMBINE_RUNTIME_ABI_H
#define INC_DC_COMBINE_RUNTIME_ABI_H

#include <cstdint>

#include "inc_dc_platform_capabilities.h"

namespace inc {
namespace dc {

constexpr uint32_t kDynCsrMagic = 0x44594353u; // 'DYCS'
constexpr uint32_t kDynCsrCtrlBytes = 448u;
// Layout capacity only.  The host queries ACL_DEV_ATTR_VECTOR_CORE_NUM and
// selects no more owners than the current device actually provides.
// Protocol storage capacity.  owner_count remains a live-hardware-derived
// runtime value; this ceiling only bounds ABI/workspace arrays.
constexpr uint32_t kDynCsrMaxOwners = kIncDcMaxCombineOwners;
constexpr uint32_t kDynCsrOptLocalOrdinalBitmap = 1u << 0;
constexpr uint32_t kDynCsrOptK1IdentityCopy = 1u << 1;
constexpr uint32_t kDynCsrOptFirstContributionInit = 1u << 2;
constexpr uint32_t kDynCsrOptSourceGroupWorklist = 1u << 3;
constexpr uint32_t kDynCsrOptRemoteResultTx = 1u << 4;
// Identity K1 is a routed copy, not a reduction.  The host enables this only
// after proving every result has exactly one contribution with weight 1.
// Weighted K1 and every non-K1 plan stay on the general INC CSR path.
constexpr uint32_t kDynCsrOptK1DirectResultTx = 1u << 5;
// Runtime ingress slots are packed by (INC, owner, source), allowing one
// contiguous RMA per ready group instead of one RMA per contribution.
constexpr uint32_t kDynCsrOptCoalescedGroupPut = 1u << 6;
// Benchmark/framework path: the operation kernel is resident before inputs
// are released and waits on a local generation cacheline.  This removes
// per-call device launch scheduling from the protocol critical path.
constexpr uint32_t kDynCsrOptPersistentLocalTrigger = 1u << 7;
// Fine-grained single-tile stream protocol.  Every ordered payload transfer
// atomically increments a result-private cacheline.  Reducers poll once per
// result instead of once per contribution; the cumulative generation target
// avoids a reset race when the same INC staging allocation is reused.
constexpr uint32_t kDynCsrOptResultArrivalCounter = 1u << 8;
// Release one head tile per owner/source before switching to the configured
// multi-tile steady-state chunk.  This is a two-level gate: reducers start at
// row latency while the bulk train retains signal/RMA amortization.
constexpr uint32_t kDynCsrOptHeadTileStream = 1u << 9;
// Use a wider UB tile only after runtime qualification.  This halves vector
// event/barrier frequency while remaining within the queried platform's
// 24-KiB AIV UB budget for the five live buffers.
constexpr uint32_t kDynCsrOptWideVectorTile = 1u << 10;
constexpr uint32_t kDynCsrOptBatchResultTx = 1u << 11;
// The single-INC compiler assigns results to owners in cyclic order.  When
// host validation proves that layout, each owner walks only its own rows
// instead of every owner redundantly scanning the complete result table.
constexpr uint32_t kDynCsrOptCyclicOwnerResults = 1u << 12;
constexpr uint32_t kDynCsrOptLocalRankPrereduce = 1u << 13;
// Results are stored in (destination rank, destination-local row) order on
// the INC.  Dedicated TX lanes may consequently push one contiguous stripe
// instead of issuing one RMA per result row.
constexpr uint32_t kDynCsrOptRankPackedResultTx = 1u << 14;
// Run the local rank pre-reduction as one batch behind a cross-lane generation
// barrier instead of reducing each contribution inside the transport loop.
// Staging serializes all vector work ahead of all transport, so it only pays
// off when a producer lane cannot push what it just reduced.
constexpr uint32_t kDynCsrOptStagedRankPrereduce = 1u << 15;
// A rank-local regular 2--4-way reducer can reserve one disjoint UB slice for
// private-MTE transport.  The next chunk is reduced while the previous chunk
// is in flight; a one-credit drain precedes every cache visibility publish.
constexpr uint32_t kDynCsrOptAsyncRankPrereducePush = 1u << 16;
// Write a regular local pre-reduce tile from vector UB directly into the
// symmetric INC ingress address, eliminating the worker-GM staging roundtrip.
constexpr uint32_t kDynCsrOptUbDirectRankPush = 1u << 17;
// Publish the INC owner-0 completion generation to workers with two private
// MTE credits.  Payload/result traffic is already fully drained at this
// point; the private UB slots only remove the per-worker public-NBI quiet
// bubble while preserving one remotely completed cacheline per worker.
constexpr uint32_t kDynCsrOptCompletionMteFanout = 1u << 18;
// Keep every identity-K1 row as an independent 16-KiB transfer, but give two
// rows disjoint private UB/event credits.  This candidate avoids the shared
// public-put staging slot without joining adjacent rows into one packet.
constexpr uint32_t kDynCsrOptK1PrivateMtePush = 1u << 19;
// A private-MTE K1 pair is remotely complete as one credit.  One ready
// generation at the lane-strided pair head can therefore release both rows.
constexpr uint32_t kDynCsrOptK1PairReady = 1u << 20;
struct alignas(64) DynCsrPersistentTriggerLine {
    int32_t generation = 0;
    uint32_t reserved = 0;
    uint64_t target_cycle = 0;
    uint64_t service_start_cycle = 0;
    uint64_t service_end_cycle = 0;
    uint32_t last_generation = 0;
    uint32_t status = 0;
    uint8_t padding[24]{};
};
static_assert(sizeof(DynCsrPersistentTriggerLine) == 64,
              "persistent trigger line must own one cacheline");

struct alignas(64) DynCsrCtrl {
    uint32_t magic = kDynCsrMagic;
    uint32_t result_count = 0;
    uint32_t contribution_count = 0;
    uint32_t hidden = 0;
    uint32_t tile_bytes = 0;
    uint32_t element_bytes = 2; // FP16
    uint32_t owner_count = 1;
    uint32_t inc_pe = 0;
    uint32_t generation = 1;
    uint32_t fail_closed_on_dup = 1;
    uint64_t result_offsets_off = 0;
    uint64_t result_home_owner_off = 0;
    uint64_t contrib_slot_off = 0;
    uint64_t contrib_weight_off = 0;
    uint64_t contrib_uid_lo_off = 0;
    uint64_t contrib_uid_hi_off = 0;
    uint64_t contrib_ordinal_off = 0;
    uint64_t contrib_gen_off = 0;
    uint64_t ingress_off = 0;
    uint64_t output_off = 0;
    // R compact bitmap words followed by C generation-tagged ordinal words.
    // The second region removes the old K<=64 protocol limit.
    uint64_t arrival_off = 0;
    uint64_t stats_off = 0;
    uint64_t owner_stats_off = 0;
    uint64_t contrib_source_rank_off = 0;
    uint64_t ready_generation_off = 0;
    uint32_t max_ingress_slots = 0;
    uint32_t this_worker_rank = 0;
    uint32_t producer_lane_count = 1;
    uint32_t producer_quiet_window = 1;
    uint32_t overlap_enable = 0;
    uint32_t ready_stride_bytes = 64;
    uint32_t ready_spin_cap = 40000000;
    // Sparse owner-source publication.  A group is
    // (owner * worker_count + source).  The variable-length source bitmap
    // deliberately avoids a W<=64 protocol limit.
    uint64_t contrib_owner_off = 0;
    uint64_t group_offsets_off = 0;
    uint64_t group_entries_off = 0;
    uint64_t owner_source_bitmap_off = 0;
    uint32_t worker_count = 0;
    uint32_t group_count = 0;
    uint32_t source_bitmap_words = 0;
    // 0=barrier, 1=per-slot, 2=owner/source batch, 3=INC/source batch,
    // 4=lazy diagnostic, 5=packed owner/source stream chunks,
    // 6=packed source/INC global stream chunks.
    uint32_t ready_mode = 0;
    // kDynCsrOpt*: independently reversible device fast paths.
    uint32_t optimization_flags = 0;
    // 0=legacy full-output DCCI; 1=only DCCI outputs <=512B.  Large outputs
    // are produced by MTE3 and consumed only after stream completion.
    uint32_t output_dcci_small_only = 0;
    // Device completion replaces only the timed-path host barrier.  Reducer
    // owner 0 publishes one generation record per worker after every owner
    // has finished; producer lane 0 waits for every INC record.
    uint32_t device_completion = 0;
    uint64_t worker_pe_off = 0;
    // Coarser source-contribution CSR used by ready_mode=3/6.  It preserves
    // per-result top-k semantics while amortizing one ready signal across all
    // owners on the single INC.
    uint64_t source_contribution_offsets_off = 0;
    uint64_t source_contribution_entries_off = 0;
    uint64_t waited_source_bitmap_off = 0;
    uint64_t source_group_offsets_off = 0;
    uint64_t source_group_entries_off = 0;
    uint64_t result_dst_rank_off = 0;
    uint64_t result_dst_row_off = 0;
    uint32_t tx_quiet_window = 1;
    uint32_t abort_generation = 0;
    // Maximum bytes in one contiguous producer RMA.  The host rounds this
    // down to a whole number of tiles; zero means one tile.
    uint32_t coalesced_chunk_bytes = 0;
    // Device-resident launch rendezvous.  Each participating PE owns one
    // 64-byte arrival record followed by one local release record.  A zero
    // offset disables the gate for compatibility/correctness baselines.
    uint64_t start_gate_off = 0;
    // Optional split INC pipeline. Reducers publish result-ready lines in GM;
    // a disjoint TX cohort drains them to the dynamic destination ranks.
    uint32_t tx_lane_count = 0;
    uint32_t reserved_split = 0;
    uint64_t result_tx_ready_off = 0;
    uint64_t tx_done_off = 0;
    uint64_t contrib_result_off = 0;
    uint64_t result_arrival_counter_off = 0;
    // Scheme-B local expert fan-in.  Logical expert rows are packed by
    // logical contribution index.  Each physical contribution owns one slice
    // in offsets/entries/weights and is reduced on its worker before push.
    uint64_t logical_input_off = 0;
    uint64_t local_reduce_offsets_off = 0;
    uint64_t local_reduce_entries_off = 0;
    uint64_t local_reduce_weights_off = 0;
    uint32_t logical_contribution_count = 0;
    uint32_t local_rank_prereduce = 0;
    uint64_t result_tx_rank_offsets_off = 0;
    uint64_t packed_result_ids_off = 0;
};
static_assert(sizeof(DynCsrCtrl) == kDynCsrCtrlBytes, "DynCsrCtrl 448B");

struct alignas(64) DynCsrStats {
    uint32_t magic = 0;
    uint32_t done = 0;
    uint32_t fail_code = 0;
    uint32_t reduced = 0;
    uint32_t mismatch_hint = 0;
    uint32_t dup_rejected = 0;
    uint32_t stale_rejected = 0;
    uint32_t owned_results = 0;
    uint64_t reduce_cycles = 0;
    uint32_t reserved[20]{};
};
static_assert(sizeof(DynCsrStats) == 128, "DynCsrStats 128B");

struct alignas(64) DynCsrOwnerStats {
    uint32_t owner = 0;
    uint32_t done = 0;
    uint32_t fail_code = 0;
    uint32_t reduced = 0;
    uint64_t reduce_cycles = 0;
    // Device-only stage telemetry.  Cycle values are compared only within
    // one device; clocks from different PEs are never assumed synchronized.
    uint64_t ready_wait_cycles = 0;
    uint64_t first_ready_cycle = 0;
    uint64_t first_reduce_cycle = 0;
    uint64_t last_reduce_cycle = 0;
    uint32_t reserved[2]{};
};
static_assert(sizeof(DynCsrOwnerStats) == 64, "DynCsrOwnerStats 64B");

// Workers reuse the owner-stats allocation as lane-private producer
// telemetry.  Reducers and producers live on disjoint PEs, so this adds no
// shared ownership or fixed worker-count limit.
struct alignas(64) DynCsrProducerStats {
    uint32_t lane = 0;
    uint32_t done = 0;
    uint32_t issued = 0;
    uint32_t ready_signals = 0;
    uint64_t first_issue_cycle = 0;
    uint64_t last_quiet_cycle = 0;
    uint64_t last_ready_cycle = 0;
    uint64_t kernel_cycles = 0;
    // reserved[0] is DynCsrFail; reserved[1] is payload-stage generation.
    uint32_t reserved[4]{};
};
static_assert(sizeof(DynCsrProducerStats) == 64,
              "DynCsrProducerStats 64B");

enum DynCsrFail : uint32_t {
    kDynCsrFailNone = 0,
    kDynCsrFailMagic = 1,
    kDynCsrFailHidden = 2,
    kDynCsrFailCsr = 3,
    kDynCsrFailHome = 4,
    kDynCsrFailDup = 5,
    kDynCsrFailStale = 6,
    kDynCsrFailMissing = 7,
    kDynCsrFailSlot = 8,
    kDynCsrFailVector = 9,
    kDynCsrFailCancelled = 10,
};

} // namespace dc
} // namespace inc

#endif
