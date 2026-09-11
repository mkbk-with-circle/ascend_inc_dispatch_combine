#include "inc_dc_source_partition_layout.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace inc::dc::pull_v2 {
namespace {

constexpr uint64_t kU64Max = std::numeric_limits<uint64_t>::max();
constexpr uint32_t kU32Max = std::numeric_limits<uint32_t>::max();

SourcePartitionLayoutStatus Fail(SourcePartitionLayoutStatus status,
                                 const char *message, std::string *error)
{
    if (error != nullptr) *error = message;
    return status;
}

bool Add(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr || b > kU64Max - a) return false;
    *out = a + b;
    return true;
}

bool Mul(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr || (a != 0u && b > kU64Max / a)) return false;
    *out = a * b;
    return true;
}

bool Align(uint64_t value, uint64_t alignment, uint64_t *out)
{
    if (out == nullptr || alignment == 0u ||
        (alignment & (alignment - 1u)) != 0u)
        return false;
    const uint64_t mask = alignment - 1u;
    if (value > kU64Max - mask) return false;
    *out = (value + mask) & ~mask;
    return true;
}

bool Capacity(uint32_t value, uint32_t quantum, uint32_t *out)
{
    if (out == nullptr || quantum == 0u ||
        value > kU32Max - (quantum - 1u))
        return false;
    *out = ((value + quantum - 1u) / quantum) * quantum;
    return true;
}

bool DataTypeBytes(DataType dtype, uint32_t *bytes)
{
    if (bytes == nullptr) return false;
    switch (dtype) {
        case DataType::FP16:
        case DataType::BF16: *bytes = 2u; return true;
        case DataType::FP32: *bytes = 4u; return true;
    }
    return false;
}

bool AlignedPartitionStride(uint64_t bytes, uint64_t *stride)
{
    // Keep even a zero-capacity origin distinct and cache-line addressable.
    return Align(std::max<uint64_t>(bytes, kSourcePartitionAlignment),
                 kSourcePartitionAlignment, stride);
}

bool Append(uint64_t bytes, uint64_t *cursor, uint64_t *offset)
{
    uint64_t aligned = 0u;
    uint64_t end = 0u;
    if (cursor == nullptr || offset == nullptr ||
        !Align(*cursor, kSourcePartitionAlignment, &aligned) ||
        !Add(aligned, bytes, &end))
        return false;
    *offset = aligned;
    *cursor = end;
    return true;
}

bool FinishPartition(uint64_t cursor, uint64_t *stride)
{
    return AlignedPartitionStride(cursor, stride);
}

bool PartitionArenaBytes(const SourcePartitionConfig &config,
                         uint64_t partition_stride, uint64_t *bytes)
{
    uint64_t partitions = 0u;
    return Mul(config.worker_count, config.ring_count, &partitions) &&
        Mul(partitions, partition_stride, bytes);
}

bool Bytes(uint64_t count, uint64_t element_bytes, uint64_t *bytes)
{
    return Mul(count, element_bytes, bytes);
}

bool AppendArray(uint64_t count, uint64_t element_bytes, uint64_t *cursor,
                 uint64_t *offset)
{
    uint64_t bytes = 0u;
    return Bytes(count, element_bytes, &bytes) &&
        Append(bytes, cursor, offset);
}

bool ParserScratchEntries(uint32_t active_blocks, uint32_t workers,
                          uint32_t experts, uint64_t *entries)
{
    uint64_t cursor = 0u;
    uint64_t extent = 0u;
    uint64_t expert_values = 0u;
    uint64_t row_stride = 0u;
    uint64_t expert_stride = 0u;
    uint64_t boundary_stride = 0u;
    uint64_t boundary_terms = 0u;
    if (entries == nullptr || active_blocks < 2u || workers == 0u ||
        experts == 0u || !Mul(workers, experts, &expert_values) ||
        !Align(workers, 16u, &row_stride) ||
        !Align(expert_values, 16u, &expert_stride))
        return false;

    // Keep this checked host formula identical to BuildParserScratchLayout in
    // the source-partitioned device kernel (all units are uint32_t entries).
    if (!Mul(16u, workers, &extent) || !Add(cursor, extent, &cursor) ||
        !Add(cursor, extent, &cursor) ||
        !Add(cursor, static_cast<uint64_t>(workers) + 1u, &cursor) ||
        !Align(cursor, 16u, &cursor) ||
        !Mul(16u, active_blocks, &extent) ||
        !Add(cursor, extent, &cursor) || !Add(cursor, extent, &cursor) ||
        !Mul(row_stride, active_blocks, &extent) ||
        !Add(cursor, extent, &cursor) || !Add(cursor, extent, &cursor) ||
        !Mul(expert_stride, active_blocks, &extent) ||
        !Add(cursor, extent, &cursor) ||
        !Mul(16u, workers, &extent) || !Add(cursor, extent, &cursor) ||
        !Mul(row_stride, active_blocks, &extent) ||
        !Add(cursor, extent, &cursor) || !Add(cursor, extent, &cursor) ||
        !Mul(16u, active_blocks, &extent) ||
        !Add(cursor, extent, &cursor) ||
        !Mul(workers, 2u, &boundary_terms) ||
        !Add(boundary_terms, 2u, &boundary_terms) ||
        !Mul(boundary_terms, 8u * 32u, &boundary_stride) ||
        !Mul(boundary_stride, active_blocks, &extent) ||
        !Add(cursor, extent, &cursor))
        return false;
    *entries = cursor;
    return true;
}

bool RingOriginBase(const SourcePartitionLayout &layout,
                    uint64_t partition_stride, uint32_t ring,
                    uint32_t origin, uint64_t *offset)
{
    uint64_t index = 0u;
    uint64_t ring_base = 0u;
    return offset != nullptr && ring < layout.config.ring_count &&
        origin < layout.config.worker_count &&
        Mul(ring, layout.config.worker_count, &ring_base) &&
        Add(ring_base, origin, &index) &&
        Mul(index, partition_stride, offset);
}

bool OriginRingBase(const SourcePartitionLayout &layout,
                    uint64_t partition_stride, uint32_t origin,
                    uint32_t ring, uint64_t *offset)
{
    uint64_t index = 0u;
    uint64_t origin_base = 0u;
    return offset != nullptr && origin < layout.config.worker_count &&
        ring < layout.config.ring_count &&
        Mul(origin, layout.config.ring_count, &origin_base) &&
        Add(origin_base, ring, &index) &&
        Mul(index, partition_stride, offset);
}

bool AddElement(uint64_t base, uint64_t index, uint64_t element_bytes,
                uint64_t *offset)
{
    uint64_t relative = 0u;
    return Mul(index, element_bytes, &relative) &&
        Add(base, relative, offset);
}

bool AddMatrixElement(uint64_t base, uint32_t row, uint32_t hidden_column,
                      uint32_t hidden, uint32_t element_bytes,
                      uint64_t *offset)
{
    uint64_t index = 0u;
    return Mul(row, hidden, &index) && Add(index, hidden_column, &index) &&
        AddElement(base, index, element_bytes, offset);
}

bool ControlStride(uint32_t record_bytes, uint64_t *stride)
{
    return record_bytes != 0u &&
        Align(record_bytes, kSourcePartitionAlignment, stride);
}

} // namespace

SourcePartitionLayoutStatus BuildSourcePartitionLayout(
    const SourcePartitionConfig &config, SourcePartitionLayout *layout,
    std::string *error)
{
    if (layout == nullptr || config.worker_count == 0u ||
        config.worker_count > kPullDispatchMaxWorkers ||
        config.ring_count == 0u ||
        config.ring_count > std::numeric_limits<uint16_t>::max() ||
        config.hidden == 0u || config.expert_count == 0u ||
        config.dispatch_aiv_per_origin < 2u)
        return Fail(SourcePartitionLayoutStatus::INVALID_ARGUMENT,
                    "invalid source partition configuration", error);

    SourcePartitionLayout built{};
    built.config = config;
    if (!DataTypeBytes(config.dtype, &built.dtype_bytes))
        return Fail(SourcePartitionLayoutStatus::INVALID_ARGUMENT,
                    "invalid Dispatch data type", error);
    if (!Capacity(config.max_source_tokens, kSourcePartitionRowQuantum,
                  &built.row_capacity) ||
        !Capacity(config.max_source_assignments,
                  kSourcePartitionAssignmentQuantum,
                  &built.assignment_capacity) ||
        !Capacity(config.expert_count, kSourcePartitionExpertCountQuantum,
                  &built.expert_count_capacity))
        return Fail(SourcePartitionLayoutStatus::SIZE_OVERFLOW,
                    "rounded capacity exceeds uint32 range", error);

    uint64_t token_destinations = 0u;
    if (!Mul(config.max_source_tokens, config.worker_count,
             &token_destinations))
        return Fail(SourcePartitionLayoutStatus::SIZE_OVERFLOW,
                    "contributor capacity overflow", error);
    built.contributor_capacity = static_cast<uint32_t>(
        std::min<uint64_t>(config.max_source_assignments,
                           token_destinations));
    if (!Mul(config.hidden, built.dtype_bytes, &built.dispatch_row_bytes) ||
        !Mul(config.hidden, sizeof(float), &built.combine_row_bytes))
        return Fail(SourcePartitionLayoutStatus::SIZE_OVERFLOW,
                    "hidden row size overflow", error);

    uint64_t cursor = sizeof(SlotHeader);
    uint64_t bytes = 0u;
    if (!Align(cursor, kSourcePartitionAlignment,
               &built.source.tokens_offset) ||
        !Bytes(config.max_source_tokens, sizeof(TokenRecord), &bytes) ||
        !Add(built.source.tokens_offset, bytes, &cursor) ||
        !Align(cursor, kSourcePartitionAlignment,
               &built.source.assignments_offset) ||
        !Bytes(config.max_source_assignments, sizeof(AssignmentRecord),
               &bytes) ||
        !Add(built.source.assignments_offset, bytes, &cursor) ||
        !Align(cursor, kSourcePartitionAlignment,
               &built.source.hidden_offset) ||
        !Bytes(config.max_source_tokens, built.dispatch_row_bytes, &bytes) ||
        !Add(built.source.hidden_offset, bytes, &cursor) ||
        !FinishPartition(cursor, &built.source.packet_stride) ||
        !Mul(config.ring_count, built.source.packet_stride,
             &built.source.worker_stride) ||
        !Mul(config.worker_count, built.source.worker_stride,
             &built.source.arena_bytes))
        return Fail(SourcePartitionLayoutStatus::SIZE_OVERFLOW,
                    "source packet arena size overflow", error);

    uint64_t partition_count = 0u;
    if (!Mul(config.ring_count, config.worker_count, &partition_count) ||
        !Bytes(built.row_capacity, built.dispatch_row_bytes, &bytes) ||
        !AlignedPartitionStride(
            bytes, &built.destination.hidden_partition_stride) ||
        !Mul(partition_count, built.destination.hidden_partition_stride,
             &built.destination.hidden_arena_bytes) ||
        !Bytes(built.row_capacity, sizeof(inc::dc::pull_v2::DestinationRow),
               &bytes) ||
        !AlignedPartitionStride(bytes,
            &built.destination.rows_partition_stride) ||
        !Mul(partition_count, built.destination.rows_partition_stride,
             &built.destination.rows_arena_bytes) ||
        !Bytes(built.assignment_capacity, sizeof(ExpertAssignment), &bytes) ||
        !AlignedPartitionStride(
            bytes, &built.destination.assignments_partition_stride) ||
        !Mul(partition_count,
             built.destination.assignments_partition_stride,
             &built.destination.assignments_arena_bytes) ||
        !Bytes(built.expert_count_capacity, sizeof(uint32_t), &bytes) ||
        !AlignedPartitionStride(
            bytes, &built.destination.expert_counts_partition_stride) ||
        !Mul(partition_count,
             built.destination.expert_counts_partition_stride,
             &built.destination.expert_counts_arena_bytes))
        return Fail(SourcePartitionLayoutStatus::SIZE_OVERFLOW,
                    "Dispatch destination arena size overflow", error);

    if (!Bytes(built.row_capacity, built.combine_row_bytes, &bytes) ||
        !AlignedPartitionStride(bytes,
            &built.combine.partial_partition_stride) ||
        !Mul(partition_count, built.combine.partial_partition_stride,
             &built.combine.partial_arena_bytes) ||
        !AlignedPartitionStride(bytes, &built.combine.output_ring_stride) ||
        !Mul(config.ring_count, built.combine.output_ring_stride,
             &built.combine.output_arena_bytes))
        return Fail(SourcePartitionLayoutStatus::SIZE_OVERFLOW,
                    "Combine data arena size overflow", error);

    // Per-origin Dispatch scratch.  These are capacities for data generated
    // by INC parsing; none of them is a host-generated route plan.
    cursor = 0u;
    uint64_t worker_rows = 0u;
    uint64_t worker_assignments = 0u;
    uint64_t row_map_entries = 0u;
    uint64_t worker_expert_entries = 0u;
    uint64_t source_destination_prefix_entries = 0u;
    if (!Mul(config.worker_count, built.row_capacity, &worker_rows) ||
        !Mul(config.worker_count, built.assignment_capacity,
             &worker_assignments) ||
        !Mul(config.worker_count, built.row_capacity, &row_map_entries) ||
        !Mul(config.worker_count, config.expert_count,
             &worker_expert_entries) ||
        !Mul(config.worker_count, 2u,
             &source_destination_prefix_entries) ||
        !ParserScratchEntries(config.dispatch_aiv_per_origin,
                              config.worker_count, config.expert_count,
                              &built.inc_dispatch
                                   .parser_scratch_capacity_entries) ||
        !Append(built.source.packet_stride, &cursor,
                &built.inc_dispatch.source_packet_offset) ||
        !AppendArray(worker_rows, sizeof(inc::dc::pull_v2::DestinationRow),
                     &cursor,
                     &built.inc_dispatch.destination_rows_offset) ||
        !AppendArray(worker_assignments, sizeof(ExpertAssignment), &cursor,
                     &built.inc_dispatch.destination_assignments_offset) ||
        !AppendArray(row_map_entries, sizeof(uint32_t), &cursor,
                     &built.inc_dispatch.row_map_offset) ||
        !AppendArray(2u, sizeof(uint32_t), &cursor,
                     &built.inc_dispatch.source_token_prefix_offset) ||
        !AppendArray(
            source_destination_prefix_entries, sizeof(uint32_t), &cursor,
            &built.inc_dispatch.source_destination_prefix_offset) ||
        !AppendArray(config.worker_count, sizeof(uint32_t), &cursor,
                     &built.inc_dispatch.destination_row_counts_offset) ||
        !AppendArray(
            config.worker_count, sizeof(uint32_t), &cursor,
            &built.inc_dispatch.destination_assignment_counts_offset) ||
        !AppendArray(worker_expert_entries, sizeof(uint32_t), &cursor,
                     &built.inc_dispatch.expert_counts_offset) ||
        !AppendArray(
            built.inc_dispatch.parser_scratch_capacity_entries,
            sizeof(uint32_t), &cursor,
            &built.inc_dispatch.parser_scratch_offset) ||
        !Append(sizeof(PullTimeline), &cursor,
                &built.inc_dispatch.timeline_offset) ||
        !FinishPartition(cursor, &built.inc_dispatch.partition_stride) ||
        !PartitionArenaBytes(config, built.inc_dispatch.partition_stride,
                             &built.inc_dispatch.arena_bytes))
        return Fail(SourcePartitionLayoutStatus::SIZE_OVERFLOW,
                    "INC Dispatch workspace size overflow", error);
    built.inc_dispatch.row_map_capacity_entries = row_map_entries;
    built.inc_dispatch.source_destination_prefix_capacity_entries =
        source_destination_prefix_entries;
    built.inc_dispatch.expert_counts_capacity_entries =
        worker_expert_entries;

    cursor = 0u;
    if (!Append(sizeof(JournalSlotHeader), &cursor,
                &built.inc_journal.header_offset) ||
        !AppendArray(built.row_capacity, sizeof(JournalTokenEntry), &cursor,
                     &built.inc_journal.tokens_offset) ||
        !AppendArray(built.contributor_capacity,
                     sizeof(JournalContributor), &cursor,
                     &built.inc_journal.contributors_offset) ||
        !AppendArray(built.assignment_capacity, sizeof(AssignmentRecord),
                     &cursor, &built.inc_journal.assignments_offset) ||
        !FinishPartition(cursor, &built.inc_journal.partition_stride) ||
        !PartitionArenaBytes(config, built.inc_journal.partition_stride,
                             &built.inc_journal.arena_bytes))
        return Fail(SourcePartitionLayoutStatus::SIZE_OVERFLOW,
                    "INC journal arena size overflow", error);

    // Direct Journal consumer state.  The partitioned kernel builds no host
    // pull plan: it stages READY records and derives contributor traversal
    // from this origin's sealed Journal on device.
    cursor = 0u;
    built.inc_combine.validation_scratch_bytes =
        static_cast<uint64_t>(config.dispatch_aiv_per_origin) * config.worker_count * 64u;
    if (!Append(built.inc_combine.validation_scratch_bytes, &cursor,
                &built.inc_combine.validation_scratch_offset) ||
        !AppendArray(config.worker_count, sizeof(CombineReadyV2), &cursor,
                     &built.inc_combine.ready_staging_offset) ||
        !AppendArray(config.worker_count,
                     kPartitionedCombineSourceScratchBytes, &cursor,
                     &built.inc_combine.source_state_offset) ||
        !AppendArray(config.worker_count,
                     kPartitionedCombineSourceScratchBytes, &cursor,
                     &built.inc_combine.source_payload_offsets_offset) ||
        !Append(sizeof(PartitionedCombineTimeline), &cursor,
                &built.inc_combine.timeline_offset) ||
        !FinishPartition(cursor, &built.inc_combine.partition_stride) ||
        !PartitionArenaBytes(config, built.inc_combine.partition_stride,
                             &built.inc_combine.arena_bytes))
        return Fail(SourcePartitionLayoutStatus::SIZE_OVERFLOW,
                    "INC Combine workspace size overflow", error);

    if (!Mul(config.ring_count, 64u,
             &built.worker_ring_control64_bytes) ||
        !Mul(config.ring_count, 128u,
             &built.worker_ring_control128_bytes) ||
        !Mul(partition_count, config.worker_count, &bytes) ||
        !Mul(bytes, 64u,
             &built.origin_ring_control64_bytes) ||
        !Mul(partition_count, config.worker_count, &bytes) ||
        !Mul(bytes, 128u,
             &built.origin_ring_control128_bytes))
        return Fail(SourcePartitionLayoutStatus::SIZE_OVERFLOW,
                    "control arena size overflow", error);

    *layout = std::move(built);
    if (error != nullptr) error->clear();
    return SourcePartitionLayoutStatus::OK;
}

SourcePartitionLayoutStatus ValidateSourcePartitionLiveCounts(
    const SourcePartitionLayout &layout, uint32_t token_count,
    uint32_t assignment_count, std::string *error)
{
    if (layout.config.worker_count == 0u || layout.config.ring_count == 0u)
        return Fail(SourcePartitionLayoutStatus::INVALID_ARGUMENT,
                    "source partition layout is not initialized", error);
    if (token_count == 0u && assignment_count != 0u)
        return Fail(SourcePartitionLayoutStatus::INVALID_ARGUMENT,
                    "assignments require at least one source token", error);
    if (token_count > layout.config.max_source_tokens ||
        assignment_count > layout.config.max_source_assignments)
        return Fail(SourcePartitionLayoutStatus::CAPACITY_EXCEEDED,
                    "source live count exceeds configured capacity", error);
    if (error != nullptr) error->clear();
    return SourcePartitionLayoutStatus::OK;
}

bool SourcePartitionOffsets::SourcePacket(
    const SourcePartitionLayout &layout, uint32_t worker, uint32_t ring,
    uint64_t *offset)
{
    uint64_t worker_base = 0u;
    uint64_t ring_offset = 0u;
    return offset != nullptr && worker < layout.config.worker_count &&
        ring < layout.config.ring_count &&
        Mul(worker, layout.source.worker_stride, &worker_base) &&
        Mul(ring, layout.source.packet_stride, &ring_offset) &&
        Add(worker_base, ring_offset, offset);
}

bool SourcePartitionOffsets::DestinationHidden(
    const SourcePartitionLayout &layout, uint32_t ring, uint32_t origin,
    uint32_t row, uint32_t hidden_column, uint64_t *offset)
{
    uint64_t base = 0u;
    return row < layout.row_capacity &&
        hidden_column < layout.config.hidden &&
        RingOriginBase(layout, layout.destination.hidden_partition_stride,
                       ring, origin, &base) &&
        AddMatrixElement(base, row, hidden_column, layout.config.hidden,
                         layout.dtype_bytes, offset);
}

bool SourcePartitionOffsets::DestinationRow(
    const SourcePartitionLayout &layout, uint32_t ring, uint32_t origin,
    uint32_t row, uint64_t *offset)
{
    uint64_t base = 0u;
    return row < layout.row_capacity &&
        RingOriginBase(layout, layout.destination.rows_partition_stride,
                       ring, origin, &base) &&
        AddElement(base, row, sizeof(inc::dc::pull_v2::DestinationRow),
                   offset);
}

bool SourcePartitionOffsets::DestinationAssignment(
    const SourcePartitionLayout &layout, uint32_t ring, uint32_t origin,
    uint32_t assignment, uint64_t *offset)
{
    uint64_t base = 0u;
    return assignment < layout.assignment_capacity &&
        RingOriginBase(layout,
                       layout.destination.assignments_partition_stride,
                       ring, origin, &base) &&
        AddElement(base, assignment, sizeof(ExpertAssignment), offset);
}

bool SourcePartitionOffsets::DestinationExpertCount(
    const SourcePartitionLayout &layout, uint32_t ring, uint32_t origin,
    uint32_t expert, uint64_t *offset)
{
    uint64_t base = 0u;
    return expert < layout.config.expert_count &&
        RingOriginBase(layout,
                       layout.destination.expert_counts_partition_stride,
                       ring, origin, &base) &&
        AddElement(base, expert, sizeof(uint32_t), offset);
}

bool SourcePartitionOffsets::CombinePartial(
    const SourcePartitionLayout &layout, uint32_t ring, uint32_t origin,
    uint32_t row, uint32_t hidden_column, uint64_t *offset)
{
    uint64_t base = 0u;
    return row < layout.row_capacity &&
        hidden_column < layout.config.hidden &&
        RingOriginBase(layout, layout.combine.partial_partition_stride,
                       ring, origin, &base) &&
        AddMatrixElement(base, row, hidden_column, layout.config.hidden,
                         sizeof(float), offset);
}

bool SourcePartitionOffsets::CombineOutput(
    const SourcePartitionLayout &layout, uint32_t ring, uint32_t row,
    uint32_t hidden_column, uint64_t *offset)
{
    uint64_t base = 0u;
    return offset != nullptr && ring < layout.config.ring_count &&
        row < layout.row_capacity &&
        hidden_column < layout.config.hidden &&
        Mul(ring, layout.combine.output_ring_stride, &base) &&
        AddMatrixElement(base, row, hidden_column, layout.config.hidden,
                         sizeof(float), offset);
}

bool SourcePartitionOffsets::OriginRingControl(
    const SourcePartitionLayout &layout, uint32_t origin, uint32_t ring,
    uint32_t worker_b, uint32_t record_bytes, uint64_t *offset)
{
    uint64_t stride = 0u;
    uint64_t slice_stride = 0u;
    uint64_t base = 0u;
    return worker_b < layout.config.worker_count &&
        ControlStride(record_bytes, &stride) &&
        Mul(layout.config.worker_count, stride, &slice_stride) &&
        OriginRingBase(layout, slice_stride, origin, ring, &base) &&
        AddElement(base, worker_b, stride, offset);
}

bool SourcePartitionOffsets::WorkerRingControl(
    const SourcePartitionLayout &layout, uint32_t ring,
    uint32_t record_bytes, uint64_t *offset)
{
    uint64_t stride = 0u;
    return offset != nullptr && ring < layout.config.ring_count &&
        ControlStride(record_bytes, &stride) && Mul(ring, stride, offset);
}

bool SourcePartitionOffsets::OriginRingControlArenaBytes(
    const SourcePartitionLayout &layout, uint32_t record_bytes,
    uint64_t *bytes)
{
    uint64_t stride = 0u;
    uint64_t records = 0u;
    uint64_t slices = 0u;
    return bytes != nullptr && ControlStride(record_bytes, &stride) &&
        Mul(layout.config.worker_count, layout.config.ring_count, &records) &&
        Mul(records, layout.config.worker_count, &slices) &&
        Mul(slices, stride, bytes);
}

bool SourcePartitionOffsets::WorkerRingControlArenaBytes(
    const SourcePartitionLayout &layout, uint32_t record_bytes,
    uint64_t *bytes)
{
    uint64_t stride = 0u;
    return bytes != nullptr && ControlStride(record_bytes, &stride) &&
        Mul(layout.config.ring_count, stride, bytes);
}

bool SourcePartitionOffsets::IncDispatchWorkspace(
    const SourcePartitionLayout &layout, uint32_t origin, uint32_t ring,
    uint64_t *offset)
{
    return OriginRingBase(layout, layout.inc_dispatch.partition_stride,
                          origin, ring, offset);
}

bool SourcePartitionOffsets::IncJournal(
    const SourcePartitionLayout &layout, uint32_t origin, uint32_t ring,
    uint64_t *offset)
{
    return OriginRingBase(layout, layout.inc_journal.partition_stride,
                          origin, ring, offset);
}

bool SourcePartitionOffsets::IncCombineWorkspace(
    const SourcePartitionLayout &layout, uint32_t origin, uint32_t ring,
    uint64_t *offset)
{
    return OriginRingBase(layout, layout.inc_combine.partition_stride,
                          origin, ring, offset);
}

const char *SourcePartitionLayoutStatusString(
    SourcePartitionLayoutStatus status)
{
    switch (status) {
        case SourcePartitionLayoutStatus::OK: return "ok";
        case SourcePartitionLayoutStatus::INVALID_ARGUMENT:
            return "invalid_argument";
        case SourcePartitionLayoutStatus::SIZE_OVERFLOW:
            return "size_overflow";
        case SourcePartitionLayoutStatus::CAPACITY_EXCEEDED:
            return "capacity_exceeded";
    }
    return "unknown";
}

} // namespace inc::dc::pull_v2
