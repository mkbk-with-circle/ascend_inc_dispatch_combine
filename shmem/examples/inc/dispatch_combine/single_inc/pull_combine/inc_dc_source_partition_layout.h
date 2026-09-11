#ifndef INC_DC_SOURCE_PARTITION_LAYOUT_H
#define INC_DC_SOURCE_PARTITION_LAYOUT_H

#include <cstdint>
#include <string>

#include "inc_dc_partitioned_combine.h"

namespace inc::dc::pull_v2 {

// Fixed-capacity source partitions used by both Dispatch and Combine.  The
// planner runs during session setup, before routes or live token counts exist;
// none of its offsets depend on READY arrival order or wave contents.
constexpr uint32_t kSourcePartitionAlignment = 64u;
constexpr uint32_t kSourcePartitionRowQuantum = 32u;
constexpr uint32_t kSourcePartitionAssignmentQuantum = 4u;
constexpr uint32_t kSourcePartitionExpertCountQuantum = 16u;

enum class SourcePartitionLayoutStatus : uint32_t {
    OK = 0u,
    INVALID_ARGUMENT,
    SIZE_OVERFLOW,
    CAPACITY_EXCEEDED,
};

struct SourcePartitionConfig {
    uint32_t worker_count = 0u;
    uint32_t ring_count = 0u;
    uint32_t max_source_tokens = 0u;
    uint32_t max_source_assignments = 0u;
    uint32_t hidden = 0u;
    uint32_t expert_count = 0u;
    // Ordinary Dispatch AIVs assigned to each independent origin launch.
    // Source-partitioned Dispatch has one parser cohort of this size.
    uint32_t dispatch_aiv_per_origin = 0u;
    DataType dtype = DataType::BF16;
};

struct SourcePacketLayout {
    uint64_t tokens_offset = 0u;
    uint64_t assignments_offset = 0u;
    uint64_t hidden_offset = 0u;
    uint64_t packet_stride = 0u;
    uint64_t worker_stride = 0u;
    uint64_t arena_bytes = 0u; // [worker][ring][packet_stride]
};

// One instance of each destination arena is allocated on every worker B.
// Its fixed addressing is [ring][origin][capacity], never a live-count prefix.
struct DispatchDestinationLayout {
    uint64_t hidden_partition_stride = 0u;
    uint64_t hidden_arena_bytes = 0u;
    uint64_t rows_partition_stride = 0u;
    uint64_t rows_arena_bytes = 0u;
    uint64_t assignments_partition_stride = 0u;
    uint64_t assignments_arena_bytes = 0u;
    uint64_t expert_counts_partition_stride = 0u;
    uint64_t expert_counts_arena_bytes = 0u;
};

// One partial arena is allocated on every worker B and one output arena on
// every owner A.  Combine storage is always FP32 regardless of Dispatch dtype.
struct CombineDataLayout {
    uint64_t partial_partition_stride = 0u;
    uint64_t partial_arena_bytes = 0u; // [ring][origin][R][H]
    uint64_t output_ring_stride = 0u;
    uint64_t output_arena_bytes = 0u; // per owner: [ring][R][H]
};

// All offsets below are relative to one per-origin/per-ring INC partition.
// Runtime code populates them after parsing source metadata; the host does not
// build or upload a route plan.
struct IncDispatchWorkspaceLayout {
    uint64_t source_packet_offset = 0u;
    uint64_t destination_rows_offset = 0u;
    uint64_t destination_assignments_offset = 0u;
    uint64_t row_map_offset = 0u;
    uint64_t source_token_prefix_offset = 0u;
    uint64_t source_destination_prefix_offset = 0u;
    uint64_t destination_row_counts_offset = 0u;
    uint64_t destination_assignment_counts_offset = 0u;
    uint64_t expert_counts_offset = 0u;
    uint64_t parser_scratch_offset = 0u;
    uint64_t timeline_offset = 0u;
    uint64_t row_map_capacity_entries = 0u;
    uint64_t source_destination_prefix_capacity_entries = 0u;
    uint64_t expert_counts_capacity_entries = 0u;
    uint64_t parser_scratch_capacity_entries = 0u;
    uint64_t partition_stride = 0u;
    uint64_t arena_bytes = 0u; // [origin][ring][partition_stride]
};

struct IncJournalLayout {
    uint64_t header_offset = 0u;
    uint64_t tokens_offset = 0u;
    uint64_t contributors_offset = 0u;
    uint64_t assignments_offset = 0u;
    uint64_t partition_stride = 0u;
    uint64_t arena_bytes = 0u; // [origin][ring][partition_stride]
};

struct IncCombineWorkspaceLayout {
    uint64_t validation_scratch_offset = 0u;
    uint64_t validation_scratch_bytes = 0u;
    uint64_t ready_staging_offset = 0u;
    uint64_t source_state_offset = 0u;
    uint64_t source_payload_offsets_offset = 0u;
    uint64_t timeline_offset = 0u;
    uint64_t partition_stride = 0u;
    uint64_t arena_bytes = 0u; // [origin][ring][partition_stride]
};

struct SourcePartitionLayout {
    SourcePartitionConfig config{};
    uint32_t dtype_bytes = 0u;
    uint32_t row_capacity = 0u;
    uint32_t assignment_capacity = 0u;
    uint32_t expert_count_capacity = 0u;
    // At most one contributor exists for each (source token, destination),
    // and each contributor consumes at least one assignment.
    uint32_t contributor_capacity = 0u;
    uint64_t dispatch_row_bytes = 0u;
    uint64_t combine_row_bytes = 0u;

    SourcePacketLayout source{};
    DispatchDestinationLayout destination{};
    CombineDataLayout combine{};
    IncDispatchWorkspaceLayout inc_dispatch{};
    IncJournalLayout inc_journal{};
    IncCombineWorkspaceLayout inc_combine{};

    // Convenient totals for distinct cache-line control arenas.  Origin
    // controls are [origin][ring][worker B]; Dispatch Ready/SourceConsumed
    // can retain one separate [ring] arena per worker A.
    uint64_t worker_ring_control64_bytes = 0u;
    uint64_t worker_ring_control128_bytes = 0u;
    uint64_t origin_ring_control64_bytes = 0u;
    uint64_t origin_ring_control128_bytes = 0u;
};

SourcePartitionLayoutStatus BuildSourcePartitionLayout(
    const SourcePartitionConfig &config, SourcePartitionLayout *layout,
    std::string *error = nullptr);

// Validates exact counts from one source READY against configured capacities.
// It deliberately does not calculate an address or a cross-source prefix.
SourcePartitionLayoutStatus ValidateSourcePartitionLiveCounts(
    const SourcePartitionLayout &layout, uint32_t token_count,
    uint32_t assignment_count, std::string *error = nullptr);

// Checked byte-offset helpers intended for the host launcher/main driver.
// Data arena helpers use [ring][origin]; INC helpers use [origin][ring] and
// control helpers use [origin][ring][worker B], keeping each origin's
// lifecycle independently addressable.
struct SourcePartitionOffsets final {
    static bool SourcePacket(const SourcePartitionLayout &layout,
                             uint32_t worker, uint32_t ring,
                             uint64_t *offset);
    static bool DestinationHidden(const SourcePartitionLayout &layout,
                                  uint32_t ring, uint32_t origin,
                                  uint32_t row, uint32_t hidden_column,
                                  uint64_t *offset);
    static bool DestinationRow(const SourcePartitionLayout &layout,
                               uint32_t ring, uint32_t origin, uint32_t row,
                               uint64_t *offset);
    static bool DestinationAssignment(const SourcePartitionLayout &layout,
                                      uint32_t ring, uint32_t origin,
                                      uint32_t assignment, uint64_t *offset);
    static bool DestinationExpertCount(const SourcePartitionLayout &layout,
                                       uint32_t ring, uint32_t origin,
                                       uint32_t expert, uint64_t *offset);
    static bool CombinePartial(const SourcePartitionLayout &layout,
                               uint32_t ring, uint32_t origin, uint32_t row,
                               uint32_t hidden_column, uint64_t *offset);
    static bool CombineOutput(const SourcePartitionLayout &layout,
                              uint32_t ring, uint32_t row,
                              uint32_t hidden_column, uint64_t *offset);
    static bool OriginRingControl(const SourcePartitionLayout &layout,
                                  uint32_t origin, uint32_t ring,
                                  uint32_t worker_b, uint32_t record_bytes,
                                  uint64_t *offset);
    static bool WorkerRingControl(const SourcePartitionLayout &layout,
                                  uint32_t ring, uint32_t record_bytes,
                                  uint64_t *offset);
    static bool OriginRingControlArenaBytes(
        const SourcePartitionLayout &layout, uint32_t record_bytes,
        uint64_t *bytes);
    static bool WorkerRingControlArenaBytes(
        const SourcePartitionLayout &layout, uint32_t record_bytes,
        uint64_t *bytes);
    static bool IncDispatchWorkspace(const SourcePartitionLayout &layout,
                                     uint32_t origin, uint32_t ring,
                                     uint64_t *offset);
    static bool IncJournal(const SourcePartitionLayout &layout,
                           uint32_t origin, uint32_t ring,
                           uint64_t *offset);
    static bool IncCombineWorkspace(const SourcePartitionLayout &layout,
                                    uint32_t origin, uint32_t ring,
                                    uint64_t *offset);
};

const char *SourcePartitionLayoutStatusString(
    SourcePartitionLayoutStatus status);

} // namespace inc::dc::pull_v2

#endif // INC_DC_SOURCE_PARTITION_LAYOUT_H
