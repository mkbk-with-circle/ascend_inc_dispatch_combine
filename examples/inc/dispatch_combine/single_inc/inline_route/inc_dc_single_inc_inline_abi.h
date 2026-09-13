#ifndef INC_DC_SINGLE_INC_INLINE_ABI_H
#define INC_DC_SINGLE_INC_INLINE_ABI_H

#include <cstdint>

#include "inc_dc_inline_route_protocol.h"
#include "inc_dc_platform_capabilities.h"

namespace inc::dc::single_inline {

constexpr uint32_t kSingleInlineMagic = 0x53495232u; // SIR2
constexpr uint32_t kSingleInlineVersion = 2u;
constexpr uint64_t kSingleInlineCacheLine = 64u;
constexpr uint64_t kSingleInlineWorkspaceAlignment = 256u;

enum SingleInlineGenerationState : uint32_t {
    kSingleInlineFree = 0u,
    kSingleInlineDispatchOpen = 1u,
    kSingleInlineDispatchSealed = 2u,
    kSingleInlineDraining = 3u,
    kSingleInlineDone = 4u,
    kSingleInlineAborted = 5u,
};

// One authoritative INC-side entry per logical top-k assignment.  It is
// published before the matching fanout assignment becomes visible.
struct alignas(64) SingleInlineJournalEntry {
    InlineRouteKeyV2 route_key{};
    uint64_t source_token = 0u;
    uint32_t source_rank = 0u;
    uint32_t destination_rank = 0u;
    uint32_t expert_id = 0u;
    uint32_t local_expert = 0u;
    uint32_t destination_row = 0u;
    uint32_t route_ordinal = 0u;
    uint32_t weight_bits = 0u;
    uint32_t wave = 0u;
    uint32_t arrived_generation = 0u;
    uint32_t result_index = 0u;
};
static_assert(sizeof(SingleInlineJournalEntry) == 64u,
              "single INC journal entry ABI");

// One result per source token.  Combine AIVs update arrived only after the
// corresponding weighted payload tile is committed to the accumulator.
struct alignas(64) SingleInlineResultState {
    uint64_t source_token = 0u;
    uint64_t accumulator_byte_offset = 0u;
    uint64_t hidden_bytes = 0u;
    uint32_t source_rank = 0u;
    uint32_t wave = 0u;
    uint32_t expected = 0u;
    uint32_t arrived = 0u;
    uint32_t state = 0u;
    uint32_t reserved32 = 0u;
    uint64_t reserved64[2]{};
};
static_assert(sizeof(SingleInlineResultState) == 64u,
              "single INC result state ABI");

// INC-resolved worker metadata.  Multiple assignments may reference one
// payload_row, so a token routed to several experts on one worker sends its
// hidden bytes once.
struct alignas(64) SingleInlineFanoutToken {
    uint64_t source_token = 0u;
    uint64_t payload_row = 0u;
    uint64_t assignment_begin = 0u;
    uint32_t assignment_count = 0u;
    uint32_t source_rank = 0u;
    uint32_t destination_rank = 0u;
    uint32_t wave = 0u;
    uint64_t reserved64[3]{};
};
static_assert(sizeof(SingleInlineFanoutToken) == 64u,
              "single INC fanout token ABI");

// INC-local producer/consumer descriptor.  Parser lane publishes ready only
// after the complete fanout record is built; TX lanes then send it to the
// resolved worker slot without reparsing endpoint routes.
struct alignas(64) SingleInlineFanoutTask {
    uint64_t local_record_index = 0u;
    uint64_t remote_record_index = 0u;
    uint64_t record_bytes = 0u;
    uint32_t destination_rank = 0u;
    uint32_t wave = 0u;
    uint32_t ready_generation = 0u;
    uint32_t reserved32 = 0u;
    uint64_t reserved64[3]{};
};
static_assert(sizeof(SingleInlineFanoutTask) == 64u,
              "single INC fanout task ABI");

struct alignas(64) SingleInlineQueueControl {
    uint64_t generation = 0u;
    uint64_t published = 0u;
    uint64_t final_count = 0u;
    uint64_t consumed = 0u;
    uint32_t sealed = 0u;
    uint32_t error = 0u;
    uint64_t reserved64[3]{};
};
static_assert(sizeof(SingleInlineQueueControl) == 64u,
              "single INC queue control ABI");

struct alignas(64) SingleInlineGenerationControl {
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t request_id = 0u;
    uint64_t slot_epoch = 0u;
    uint32_t state = kSingleInlineFree;
    uint32_t error = 0u;
    uint32_t dispatch_sources_sealed = 0u;
    uint32_t combine_sources_sealed = 0u;
    uint64_t outstanding_results = 0u;
};
static_assert(sizeof(SingleInlineGenerationControl) == 64u,
              "single INC generation control ABI");

// All addresses are symmetric-heap-relative.  SHMEM PE numbers are confined
// to the transport adapter and are deliberately absent from route records.
struct alignas(64) SingleInlineRuntimeDesc {
    uint32_t magic = kSingleInlineMagic;
    uint32_t version = kSingleInlineVersion;
    uint32_t pe = 0u;
    uint32_t inc_pe = 0u;
    uint32_t worker_count = 0u;
    uint32_t expert_count = 0u;
    uint32_t hidden_bytes = 0u;
    uint32_t max_topk = 0u;
    uint32_t tokens_per_wave = 0u;
    uint32_t batch_tokens = 0u;
    uint32_t payload_tile_rows = 0u;
    uint32_t slot_count = 0u;
    uint32_t batches_per_wave = 0u;
    uint32_t payload_tiles_per_wave = 0u;
    uint32_t dispatch_lane_count = 0u;
    uint32_t combine_lane_count = 0u;
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t request_id = 0u;
    uint64_t ingress_frame_stride = 0u;
    uint64_t fanout_record_stride = 0u;
    uint64_t combine_record_stride = 0u;
    uint64_t result_record_stride = 0u;
    uint64_t generation_control_off = 0u;
    uint64_t expert_owner_off = 0u;
    uint64_t expert_local_index_off = 0u;
    uint64_t worker_endpoint_off = 0u;
    uint64_t ingress_frame_off = 0u;
    uint64_t journal_result_off = 0u;
    uint64_t journal_entry_off = 0u;
    uint64_t expert_row_counter_off = 0u;
    uint64_t fanout_task_off = 0u;
    uint64_t fanout_queue_off = 0u;
    uint64_t fanout_record_off = 0u;
    uint64_t fanout_count_off = 0u;
    uint64_t combine_record_off = 0u;
    uint64_t accumulator_off = 0u;
    uint64_t result_record_off = 0u;
    uint64_t stats_off = 0u;
    uint64_t total_bytes = 0u;
};
static_assert(sizeof(SingleInlineRuntimeDesc) % 64u == 0u,
              "single INC runtime descriptor alignment");

} // namespace inc::dc::single_inline

#endif
