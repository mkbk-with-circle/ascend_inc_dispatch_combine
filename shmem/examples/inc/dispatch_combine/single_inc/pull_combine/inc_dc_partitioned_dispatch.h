#ifndef INC_DC_PARTITIONED_DISPATCH_H
#define INC_DC_PARTITIONED_DISPATCH_H
#include <cstdint>
namespace inc::dc::pull_v2 {
// Each launch serves one origin. Destination arena pointers already select
// its fixed partition; ring strides span all origins. INC workspaces/Journal
// are private to this launch. READY/ACK arrays retain all worker entries.
struct PartitionedDispatchLaunchArgs {
    uint32_t origin_rank = 0u;
    uint8_t *source_region = nullptr;
    uint8_t *ready_mailbox = nullptr;
    uint8_t *inc_slots = nullptr;
    uint8_t *source_acks = nullptr;
    uint8_t *destination_hidden = nullptr;
    uint8_t *destination_rows = nullptr;
    uint8_t *destination_assignments = nullptr;
    uint8_t *destination_expert_counts = nullptr;
    uint8_t *destination_completions = nullptr;
    uint8_t *inc_destination_rows = nullptr;
    uint8_t *inc_destination_assignments = nullptr;
    uint8_t *journal_header = nullptr;
    uint8_t *journal_tokens = nullptr;
    uint8_t *journal_contributors = nullptr;
    uint8_t *journal_assignments = nullptr;
    uint8_t *row_map = nullptr;
    uint8_t *source_token_prefix = nullptr;
    uint8_t *source_destination_prefix = nullptr;
    uint8_t *destination_row_counts = nullptr;
    uint8_t *destination_assignment_counts = nullptr;
    uint8_t *expert_counts = nullptr;
    uint8_t *parser_scratch = nullptr;
    uint8_t *status_line = nullptr;
    uint64_t ffts_addr = 0;
    uint64_t session_id = 0;
    uint64_t placement_epoch = 0;
    uint64_t generation = 0;
    uint64_t sequence = 0;
    uint64_t source_slot_stride = 0;
    uint64_t destination_hidden_slot_stride = 0;
    uint64_t destination_rows_slot_stride = 0;
    uint64_t destination_assignments_slot_stride = 0;
    uint64_t destination_expert_counts_slot_stride = 0;
    uint64_t journal_token_capacity = 0;
    uint64_t journal_contributor_capacity = 0;
    uint64_t journal_assignment_capacity = 0;
    uint64_t destination_row_capacity = 0;
    uint64_t destination_assignment_capacity = 0;
    uint64_t inc_destination_rows_stride_bytes = 0;
    uint64_t inc_destination_assignments_stride_bytes = 0;
    uint64_t row_map_capacity_entries = 0;
    uint64_t source_destination_prefix_capacity_entries = 0;
    uint64_t expert_counts_capacity_entries = 0;
    uint64_t parser_scratch_capacity_entries = 0;
    uint32_t worker_count = 0;
    uint32_t expert_count = 0;
    uint32_t hidden = 0;
    uint32_t dtype = 0;
    int32_t inc_pe = 0;
    uint32_t region_id = 0;
    uint32_t wave = 0;
    uint32_t ring_slot = 0;
    uint32_t slot_count = 0;
    uint32_t channels_per_source = 0;
    uint64_t spin_cap = 0;
};
}
extern "C" void launch_inc_dc_partitioned_dispatch(uint32_t block_dim, void *stream,
    const inc::dc::pull_v2::PartitionedDispatchLaunchArgs *args);
#endif
