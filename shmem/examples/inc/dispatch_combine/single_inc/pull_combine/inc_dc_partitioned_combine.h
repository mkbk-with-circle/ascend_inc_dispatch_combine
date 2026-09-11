#ifndef INC_DC_PARTITIONED_COMBINE_H
#define INC_DC_PARTITIONED_COMBINE_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "inc_dc_pull_combine_v2.h"

namespace inc::dc::pull_v2 {

constexpr uint32_t kPartitionedCombineSourceScratchBytes = 64u;
constexpr uint32_t kPartitionedCombineValidationSummaryBytes = 64u;

// Per-origin Combine keeps the existing READY wire ABI.  ACK/completion are
// exposed here because the host owns one independent [origin][ring][B] arena
// for each record type.
struct alignas(64) PartitionedCombineSourceAck {
    uint32_t magic = kPullCombineV2Magic;
    uint16_t abi_version = kPullCombineV2AbiVersion;
    uint16_t struct_bytes = sizeof(PartitionedCombineSourceAck);
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint64_t dispatch_cookie = 0u;
    uint32_t wave = 0u;
    uint32_t source_rank = 0u;
    uint32_t source_region_id = 0u;
    uint32_t status = 0u;
    uint16_t ring_slot = 0u;
    uint16_t flags = 0u;
    uint32_t row_count = 0u;
    uint64_t bytes_consumed = 0u;
    uint64_t reserved[5]{};
    uint64_t publication = 0u;
};
static_assert(sizeof(PartitionedCombineSourceAck) == 128u,
              "partitioned Combine source ACK ABI drift");
static_assert(offsetof(PartitionedCombineSourceAck, publication) +
                  sizeof(uint64_t) ==
              sizeof(PartitionedCombineSourceAck),
              "partitioned Combine source ACK publication must be last");

struct alignas(64) PartitionedCombineOwnerCompletion {
    uint32_t magic = kPullCombineV2Magic;
    uint16_t abi_version = kPullCombineV2AbiVersion;
    uint16_t struct_bytes = sizeof(PartitionedCombineOwnerCompletion);
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint64_t dispatch_cookie = 0u;
    uint32_t wave = 0u;
    uint32_t owner_rank = 0u;
    uint32_t status = 0u;
    uint32_t row_count = 0u;
    uint16_t ring_slot = 0u;
    uint16_t flags = 0u;
    uint32_t reserved0 = 0u;
    uint64_t bytes_produced = 0u;
    uint64_t reserved[5]{};
    uint64_t publication = 0u;
};
static_assert(sizeof(PartitionedCombineOwnerCompletion) == 128u,
              "partitioned Combine owner completion ABI drift");
static_assert(offsetof(PartitionedCombineOwnerCompletion, publication) +
                  sizeof(uint64_t) ==
              sizeof(PartitionedCombineOwnerCompletion),
              "partitioned Combine owner publication must be last");

struct alignas(64) PartitionedCombineTimeline {
    uint32_t status = 0u;
    uint32_t ready_sources = 0u;
    uint64_t kernel_start = 0u;
    uint64_t journal_validated = 0u;
    uint64_t first_ready = 0u;
    uint64_t all_required_ready = 0u;
    uint64_t first_get = 0u;
    uint64_t last_get = 0u;
    uint64_t last_reduce = 0u;
    uint64_t last_owner_put = 0u;
    uint64_t source_acks_done = 0u;
    uint64_t owner_completion_done = 0u;
    uint64_t kernel_done = 0u;
    uint64_t reserved[4]{};
};
static_assert(sizeof(PartitionedCombineTimeline) == 128u,
              "partitioned Combine timeline ABI drift");

// The four control arenas are already offset by the caller to this
// [origin][ring] slice. The kernel therefore indexes them by worker B only.
// symmetric_partials and owner_output are symmetric-region bases; their
// ring/origin offsets are checked and formed on device.
struct PartitionedCombineLaunchArgs {
    uint8_t *symmetric_partials;
    uint8_t *ready_records;       // Symmetric CombineReadyV2[B]: dst pushes to INC before Notice
    uint8_t *ready_notices;       // CombineReadyNoticeV2[B]
    uint8_t *ready_staging;       // Reserved for ABI compatibility; no control GET staging needed
    uint8_t *registrations;       // CombineRegionRegistration[B]
    uint8_t *source_acks;          // already [origin][ring], then [B]
    uint8_t *owner_output;         // symmetric [ring][row][hidden]
    uint8_t *owner_completions;    // [origin][ring] base; publish [origin_rank]
    uint8_t *journal_header;       // exact [origin][ring] slot
    uint8_t *journal_tokens;       // JournalTokenEntry[N]
    uint8_t *journal_contributors; // JournalContributor[M]
    uint8_t *destination_row_counts; // uint32_t[B]
    uint8_t *source_state;         // INC-local, scratch bytes per B
    uint8_t *source_payload_offsets; // same stride, one u64 per B
    uint8_t *validation_scratch;   // INC-local [block_dim][B][64]
    uint8_t *status_line;          // PartitionedCombineTimeline

    uint64_t ffts_addr;
    uint64_t session_id;
    uint64_t placement_epoch;
    uint64_t generation;
    uint64_t sequence;
    // Zero acquires the non-zero cookie from the matching sealed Journal.
    // Non-zero requires an exact match.
    uint64_t dispatch_cookie;
    uint64_t owner_output_slot_stride;
    uint64_t journal_token_capacity;
    uint64_t journal_contributor_capacity;
    uint64_t journal_assignment_capacity;
    uint64_t source_scratch_capacity; // number of B entries in both caches
    uint64_t validation_scratch_capacity_bytes;
    uint64_t spin_cap;

    uint32_t worker_count;
    uint32_t hidden;
    uint32_t origin_rank;
    uint32_t origin_row_capacity;
    int32_t inc_pe;
    uint32_t wave;
    uint32_t ring_slot;
    uint32_t slot_count;
};
static_assert(std::is_standard_layout<PartitionedCombineLaunchArgs>::value,
              "partitioned Combine launch args must remain standard layout");
static_assert(std::is_trivial<PartitionedCombineLaunchArgs>::value,
              "partitioned Combine launch args must remain POD");

} // namespace inc::dc::pull_v2

extern "C" void launch_inc_dc_partitioned_combine(
    uint32_t block_dim, void *stream,
    const inc::dc::pull_v2::PartitionedCombineLaunchArgs *args);

#endif
