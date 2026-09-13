#ifndef INC_DC_SINGLE_INC_INLINE_BENCH_ABI_H
#define INC_DC_SINGLE_INC_INLINE_BENCH_ABI_H

#include <cstdint>

#include "../../common/protocol/inc_dc_inline_route_protocol.h"
#include "../../common/platform/inc_dc_platform_capabilities.h"

namespace inc::dc::single_inline {

// Self-contained device benchmark ABI for the V2 inline-route data path.
// The descriptor lives at symmetric_heap + 0 on every PE.  Endpoint inputs
// contain router output only (expert id, ordinal and weight); notably there is
// no destination rank, output row, SHMEM PE, address or AIV schedule.
constexpr uint32_t kInlineDispatchBenchMagic = 0x49444232u; // IDB2
constexpr uint32_t kInlineDispatchBenchVersion = 2u;
constexpr uint32_t kInlineDispatchMaxLanes = kIncDcMaxDispatchLanes;

enum InlineDispatchBenchError : uint32_t {
    kInlineDispatchOk = 0u,
    kInlineDispatchBadDescriptor = 1u,
    kInlineDispatchTimeout = 2u,
    kInlineDispatchBadBatch = 3u,
    kInlineDispatchBadToken = 4u,
    kInlineDispatchBadRoute = 5u,
    kInlineDispatchBadPlacement = 6u,
    kInlineDispatchCapacity = 7u,
};

struct alignas(64) InlineDispatchBenchStatV2 {
    uint64_t start_cycle = 0u;
    uint64_t end_cycle = 0u;
    uint64_t upload_bytes = 0u;
    uint64_t fanout_bytes = 0u;
    uint32_t token_records = 0u;
    uint32_t assignments = 0u;
    uint32_t batches = 0u;
    uint32_t error = kInlineDispatchOk;
    uint32_t done_generation = 0u;
    uint32_t reserved32 = 0u;
    uint64_t reserved64 = 0u;
};
static_assert(sizeof(InlineDispatchBenchStatV2) == 64u,
              "inline dispatch stat ABI");

// Device Dispatch journal entry.  Combine must present route_key and the
// physical contributor rank; all other routing/weight fields remain
// authoritative here and are never recomputed by the endpoint.
struct alignas(64) InlineDispatchJournalEntryV2 {
    InlineRouteKeyV2 route_key{};
    uint64_t source_token = 0u;
    uint32_t source_rank = 0u;
    uint32_t contributor_rank = 0u;
    uint32_t expert_id = 0u;
    uint32_t local_expert = 0u;
    uint32_t destination_row = 0u;
    uint32_t route_ordinal = 0u;
    uint32_t weight_bits = 0u;
    uint32_t wave = 0u;
    uint32_t generation = 0u;
    uint32_t valid_generation = 0u;
};
static_assert(sizeof(InlineDispatchJournalEntryV2) == 64u,
              "inline dispatch journal ABI");

// All offsets are symmetric-heap-relative. source_* regions are local router
// output on each worker PE. ingress/fanout regions are symmetric staging and
// receive regions. Worker-done control contains, in order, W upload lines,
// inc_lane_count parser lines and W fanout-complete lines (64 bytes each).
struct alignas(256) InlineDispatchBenchDescV2 {
    uint32_t magic = kInlineDispatchBenchMagic;
    uint32_t version = kInlineDispatchBenchVersion;
    uint32_t pe = 0u;
    uint32_t inc_pe = 0u;
    uint32_t worker_count = 0u;
    uint32_t expert_count = 0u;
    uint32_t hidden_bytes = 0u;
    uint32_t max_topk = 0u;
    uint32_t tokens_per_worker = 0u;
    uint32_t batch_tokens = 0u;
    uint32_t worker_lane_count = 0u;
    uint32_t inc_lane_count = 0u;
    uint32_t spin_cap = 0u;
    uint32_t generation = 0u;
    uint32_t slot_count = 0u;
    uint32_t reserved32 = 0u;
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t request_id = 0u;
    uint64_t source_hidden_off = 0u;
    uint64_t source_route_count_off = 0u;
    uint64_t source_route_entry_off = 0u;
    uint64_t ingress_frame_off = 0u;
    uint64_t ingress_frame_stride = 0u;
    uint64_t fanout_record_off = 0u;
    uint64_t fanout_record_stride = 0u;
    // Dense online-resolved batches use transport segmentation: metadata and
    // payload are placed in two contiguous arrays and one batch-ready ticket
    // publishes the complete logical fanout records.  This avoids copying a
    // large hidden row through padded per-token staging while preserving the
    // same protocol header/assignment semantics at the receiver adapter.
    uint64_t dense_fanout_metadata_off = 0u;
    uint64_t dense_fanout_metadata_stride = 0u;
    uint64_t dense_fanout_payload_off = 0u;
    uint64_t expert_owner_off = 0u;
    uint64_t expert_local_index_off = 0u;
    uint64_t journal_off = 0u;
    // One 64-byte release ticket per (slot, source, batch, destination).
    // It is published only after every complete fanout record in that
    // microbatch is remotely visible.
    uint64_t fanout_ready_off = 0u;
    uint64_t worker_done_off = 0u;
    uint64_t stats_off = 0u;
    uint64_t total_bytes = 0u;
    uint64_t reserved64[3]{};
};
static_assert(sizeof(InlineDispatchBenchDescV2) == 256u,
              "inline dispatch descriptor ABI");

} // namespace inc::dc::single_inline

#endif
