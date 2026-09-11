#include "inc_dc_source_partition_layout.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace inc::dc::pull_v2;

int failures = 0;

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::cerr << __FILE__ << ':' << __LINE__                         \
                      << ": CHECK failed: " #condition << '\n';             \
            ++failures;                                                      \
        }                                                                    \
    } while (false)

bool Aligned64(uint64_t value)
{
    return value % kSourcePartitionAlignment == 0u;
}

SourcePartitionConfig StandardConfig(DataType dtype = DataType::BF16)
{
    SourcePartitionConfig config{};
    config.worker_count = 3u;
    config.ring_count = 3u;
    config.max_source_tokens = 33u;
    config.max_source_assignments = 5u;
    config.hidden = 7u;
    config.expert_count = 17u;
    config.dispatch_aiv_per_origin = 6u;
    config.dtype = dtype;
    return config;
}

SourcePartitionLayout MakeLayout(
    const SourcePartitionConfig &config = StandardConfig())
{
    SourcePartitionLayout layout{};
    std::string error;
    const auto status = BuildSourcePartitionLayout(config, &layout, &error);
    CHECK(status == SourcePartitionLayoutStatus::OK);
    CHECK(error.empty());
    return layout;
}

void TestConfiguredCapacitiesAndStrides()
{
    const SourcePartitionLayout layout = MakeLayout();
    CHECK(layout.inc_combine.validation_scratch_bytes == 6u * 3u * 64u);
    CHECK(Aligned64(layout.inc_combine.validation_scratch_offset));
    CHECK(layout.inc_combine.validation_scratch_offset +
              layout.inc_combine.validation_scratch_bytes <=
          layout.inc_combine.ready_staging_offset);
    CHECK(layout.dtype_bytes == 2u);
    CHECK(layout.row_capacity == 64u);
    CHECK(layout.assignment_capacity == 8u);
    CHECK(layout.expert_count_capacity == 32u);
    CHECK(layout.contributor_capacity == 5u);
    CHECK(layout.dispatch_row_bytes == 14u);
    CHECK(layout.combine_row_bytes == 28u);

    CHECK(layout.source.tokens_offset == 128u);
    CHECK(layout.source.assignments_offset == 1216u);
    CHECK(layout.source.hidden_offset == 1344u);
    CHECK(layout.source.packet_stride == 1856u);
    CHECK(layout.source.worker_stride ==
          layout.config.ring_count * layout.source.packet_stride);
    CHECK(layout.source.arena_bytes ==
          static_cast<uint64_t>(layout.config.worker_count) *
              layout.source.worker_stride);

    CHECK(layout.destination.hidden_partition_stride == 896u);
    CHECK(layout.destination.rows_partition_stride == 2560u);
    CHECK(layout.destination.assignments_partition_stride == 256u);
    CHECK(layout.destination.expert_counts_partition_stride == 128u);
    CHECK(layout.combine.partial_partition_stride == 1792u);
    CHECK(layout.combine.output_ring_stride == 1792u);

    const uint64_t partitions =
        static_cast<uint64_t>(layout.config.worker_count) *
        layout.config.ring_count;
    CHECK(layout.destination.hidden_arena_bytes ==
          partitions * layout.destination.hidden_partition_stride);
    CHECK(layout.destination.rows_arena_bytes ==
          partitions * layout.destination.rows_partition_stride);
    CHECK(layout.destination.assignments_arena_bytes ==
          partitions * layout.destination.assignments_partition_stride);
    CHECK(layout.destination.expert_counts_arena_bytes ==
          partitions * layout.destination.expert_counts_partition_stride);
    CHECK(layout.combine.partial_arena_bytes ==
          partitions * layout.combine.partial_partition_stride);
    CHECK(layout.combine.output_arena_bytes ==
          layout.config.ring_count * layout.combine.output_ring_stride);

    CHECK(layout.worker_ring_control64_bytes == 3u * 64u);
    CHECK(layout.worker_ring_control128_bytes == 3u * 128u);
    CHECK(layout.origin_ring_control64_bytes ==
          partitions * layout.config.worker_count * 64u);
    CHECK(layout.origin_ring_control128_bytes ==
          partitions * layout.config.worker_count * 128u);

    const uint64_t aligned_values[] = {
        layout.source.tokens_offset,
        layout.source.assignments_offset,
        layout.source.hidden_offset,
        layout.source.packet_stride,
        layout.source.worker_stride,
        layout.destination.hidden_partition_stride,
        layout.destination.rows_partition_stride,
        layout.destination.assignments_partition_stride,
        layout.destination.expert_counts_partition_stride,
        layout.combine.partial_partition_stride,
        layout.combine.output_ring_stride,
        layout.inc_dispatch.partition_stride,
        layout.inc_journal.partition_stride,
        layout.inc_combine.partition_stride,
    };
    for (uint64_t value : aligned_values) CHECK(Aligned64(value));

    CHECK(layout.inc_dispatch.arena_bytes ==
          partitions * layout.inc_dispatch.partition_stride);
    CHECK(layout.inc_journal.arena_bytes ==
          partitions * layout.inc_journal.partition_stride);
    CHECK(layout.inc_combine.arena_bytes ==
          partitions * layout.inc_combine.partition_stride);
    CHECK(layout.inc_combine.validation_scratch_offset == 0u);
    CHECK(layout.inc_combine.ready_staging_offset ==
          layout.inc_combine.validation_scratch_bytes);
    CHECK(layout.inc_combine.source_state_offset ==
          layout.inc_combine.ready_staging_offset +
              layout.config.worker_count * sizeof(CombineReadyV2));
    CHECK(layout.inc_combine.source_payload_offsets_offset ==
          layout.inc_combine.source_state_offset +
              layout.config.worker_count *
                  kPartitionedCombineSourceScratchBytes);
    CHECK(layout.inc_combine.timeline_offset ==
          layout.inc_combine.source_payload_offsets_offset +
              layout.config.worker_count *
                  kPartitionedCombineSourceScratchBytes);
    CHECK(layout.inc_dispatch.row_map_capacity_entries == 3u * 64u);
    CHECK(layout.inc_dispatch.source_destination_prefix_capacity_entries ==
          2u * 3u);
    CHECK(layout.inc_dispatch.expert_counts_capacity_entries == 3u * 17u);
    CHECK(layout.inc_dispatch.parser_scratch_capacity_entries == 13504u);
    CHECK(Aligned64(layout.inc_dispatch.source_token_prefix_offset));
    CHECK(Aligned64(
        layout.inc_dispatch.source_destination_prefix_offset));
    CHECK(Aligned64(layout.inc_dispatch.parser_scratch_offset));
}

void TestDTypePartitionBoundaries()
{
    for (DataType dtype :
         {DataType::FP16, DataType::BF16, DataType::FP32}) {
        SourcePartitionConfig config = StandardConfig(dtype);
        config.max_source_tokens = 1u;
        const SourcePartitionLayout layout = MakeLayout(config);
        CHECK(layout.row_capacity == 32u);
        CHECK(Aligned64(layout.source.packet_stride));
        CHECK(Aligned64(layout.destination.hidden_partition_stride));
        CHECK(layout.destination.hidden_partition_stride ==
              static_cast<uint64_t>(layout.row_capacity) * config.hidden *
                  layout.dtype_bytes);
    }
}

void TestKeyOffsetsAndBounds()
{
    const SourcePartitionLayout layout = MakeLayout();
    uint64_t offset = 0u;
    CHECK(SourcePartitionOffsets::SourcePacket(layout, 2u, 1u, &offset));
    CHECK(offset == 2u * layout.source.worker_stride +
                        layout.source.packet_stride);
    CHECK(!SourcePartitionOffsets::SourcePacket(layout, 3u, 0u, &offset));
    CHECK(!SourcePartitionOffsets::SourcePacket(layout, 0u, 3u, &offset));

    CHECK(SourcePartitionOffsets::DestinationHidden(
        layout, 1u, 2u, 63u, 6u, &offset));
    const uint64_t hidden_partition =
        (1u * layout.config.worker_count + 2u) *
        layout.destination.hidden_partition_stride;
    CHECK(offset == hidden_partition +
                        (63u * layout.config.hidden + 6u) *
                            layout.dtype_bytes);
    CHECK(offset + layout.dtype_bytes <=
          hidden_partition + layout.destination.hidden_partition_stride);
    CHECK(!SourcePartitionOffsets::DestinationHidden(
        layout, 1u, 2u, 64u, 0u, &offset));
    CHECK(!SourcePartitionOffsets::DestinationHidden(
        layout, 1u, 2u, 0u, 7u, &offset));

    CHECK(SourcePartitionOffsets::DestinationRow(
        layout, 2u, 1u, 63u, &offset));
    CHECK(offset + sizeof(DestinationRow) <=
          layout.destination.rows_arena_bytes);
    CHECK(!SourcePartitionOffsets::DestinationRow(
        layout, 2u, 1u, 64u, &offset));

    CHECK(SourcePartitionOffsets::DestinationAssignment(
        layout, 2u, 1u, 7u, &offset));
    CHECK(offset + sizeof(ExpertAssignment) <=
          layout.destination.assignments_arena_bytes);
    CHECK(!SourcePartitionOffsets::DestinationAssignment(
        layout, 2u, 1u, 8u, &offset));

    CHECK(SourcePartitionOffsets::DestinationExpertCount(
        layout, 2u, 1u, 16u, &offset));
    CHECK(!SourcePartitionOffsets::DestinationExpertCount(
        layout, 2u, 1u, 17u, &offset));

    CHECK(SourcePartitionOffsets::CombinePartial(
        layout, 2u, 2u, 63u, 6u, &offset));
    CHECK(offset + sizeof(float) <= layout.combine.partial_arena_bytes);
    CHECK(!SourcePartitionOffsets::CombinePartial(
        layout, 3u, 0u, 0u, 0u, &offset));

    CHECK(SourcePartitionOffsets::CombineOutput(
        layout, 2u, 63u, 6u, &offset));
    CHECK(offset + sizeof(float) <= layout.combine.output_arena_bytes);
    CHECK(!SourcePartitionOffsets::CombineOutput(
        layout, 2u, 64u, 0u, &offset));
}

std::vector<uint64_t> AddressesForScenario(
    const SourcePartitionLayout &layout,
    const std::vector<uint32_t> &arrival_order,
    const std::vector<std::pair<uint32_t, uint32_t>> &live_counts)
{
    std::vector<uint64_t> addresses(layout.config.worker_count, 0u);
    for (uint32_t origin : arrival_order) {
        CHECK(origin < live_counts.size());
        CHECK(ValidateSourcePartitionLiveCounts(
                  layout, live_counts[origin].first,
                  live_counts[origin].second) ==
              SourcePartitionLayoutStatus::OK);
        CHECK(SourcePartitionOffsets::DestinationHidden(
            layout, 1u, origin, 0u, 0u, &addresses[origin]));
    }
    return addresses;
}

void TestArrivalAndCountsCannotMovePartitions()
{
    const SourcePartitionLayout layout = MakeLayout();
    const auto first = AddressesForScenario(
        layout, {2u, 0u, 1u}, {{1u, 1u}, {33u, 5u}, {0u, 0u}});
    const auto second = AddressesForScenario(
        layout, {1u, 2u, 0u}, {{32u, 4u}, {2u, 3u}, {17u, 5u}});
    CHECK(first == second);
    for (uint32_t origin = 0u; origin < layout.config.worker_count;
         ++origin) {
        CHECK(first[origin] ==
              (layout.config.worker_count + origin) *
                  layout.destination.hidden_partition_stride);
    }

    CHECK(ValidateSourcePartitionLiveCounts(layout, 0u, 0u) ==
          SourcePartitionLayoutStatus::OK);
    CHECK(ValidateSourcePartitionLiveCounts(layout, 0u, 1u) ==
          SourcePartitionLayoutStatus::INVALID_ARGUMENT);
    CHECK(ValidateSourcePartitionLiveCounts(layout, 34u, 5u) ==
          SourcePartitionLayoutStatus::CAPACITY_EXCEEDED);
    CHECK(ValidateSourcePartitionLiveCounts(layout, 33u, 6u) ==
          SourcePartitionLayoutStatus::CAPACITY_EXCEEDED);
}

void TestControlAndPrivateOriginOrdering()
{
    const SourcePartitionLayout layout = MakeLayout();
    uint64_t offset = 0u;
    uint64_t bytes = 0u;
    CHECK(SourcePartitionOffsets::OriginRingControl(
        layout, 1u, 2u, 2u, sizeof(CombineReadyNoticeV2), &offset));
    CHECK(offset ==
          ((1u * layout.config.ring_count + 2u) *
               layout.config.worker_count +
           2u) *
              64u);
    CHECK(SourcePartitionOffsets::OriginRingControlArenaBytes(
        layout, sizeof(CombineReadyNoticeV2), &bytes));
    CHECK(bytes == layout.origin_ring_control64_bytes);
    CHECK(SourcePartitionOffsets::OriginRingControlArenaBytes(
        layout, sizeof(CombineReadyV2), &bytes));
    CHECK(bytes == layout.origin_ring_control128_bytes);
    CHECK(SourcePartitionOffsets::OriginRingControlArenaBytes(
        layout, sizeof(DestinationCompletion), &bytes));
    CHECK(bytes == layout.origin_ring_control128_bytes);

    // Dispatch Ready and SourceConsumed remain one [ring] arena per worker.
    CHECK(SourcePartitionOffsets::WorkerRingControl(
        layout, 2u, sizeof(Ready), &offset));
    CHECK(offset == 2u * 64u);
    CHECK(SourcePartitionOffsets::WorkerRingControlArenaBytes(
        layout, sizeof(Ready), &bytes));
    CHECK(bytes == layout.worker_ring_control64_bytes);
    CHECK(SourcePartitionOffsets::WorkerRingControlArenaBytes(
        layout, sizeof(SourceConsumed), &bytes));
    CHECK(bytes == layout.worker_ring_control128_bytes);
    CHECK(!SourcePartitionOffsets::OriginRingControl(
        layout, 0u, 0u, 0u, 0u, &offset));
    CHECK(!SourcePartitionOffsets::OriginRingControl(
        layout, 0u, 0u, layout.config.worker_count,
        sizeof(CombineReadyV2), &offset));

    CHECK(SourcePartitionOffsets::IncDispatchWorkspace(
        layout, 2u, 1u, &offset));
    CHECK(offset == (2u * layout.config.ring_count + 1u) *
                        layout.inc_dispatch.partition_stride);
    CHECK(Aligned64(offset));
    CHECK(SourcePartitionOffsets::IncJournal(layout, 2u, 1u, &offset));
    CHECK(offset == (2u * layout.config.ring_count + 1u) *
                        layout.inc_journal.partition_stride);
    CHECK(Aligned64(offset));
    CHECK(SourcePartitionOffsets::IncCombineWorkspace(
        layout, 2u, 1u, &offset));
    CHECK(offset == (2u * layout.config.ring_count + 1u) *
                        layout.inc_combine.partition_stride);
    CHECK(Aligned64(offset));
    CHECK(!SourcePartitionOffsets::IncJournal(layout, 3u, 0u, &offset));
}

void CheckGuardedDisjointArena(uint64_t arena_bytes, uint64_t stride,
                               uint32_t partitions)
{
    constexpr size_t guard = 64u;
    constexpr uint8_t guard_value = 0xa5u;
    std::vector<uint8_t> storage(
        static_cast<size_t>(arena_bytes) + 2u * guard, guard_value);
    uint8_t *arena = storage.data() + guard;
    for (uint32_t partition = 0u; partition < partitions; ++partition) {
        CHECK(Aligned64(static_cast<uint64_t>(partition) * stride));
        std::memset(arena + static_cast<uint64_t>(partition) * stride,
                    static_cast<int>(partition + 1u),
                    static_cast<size_t>(stride));
    }
    for (uint32_t partition = 0u; partition < partitions; ++partition) {
        const uint8_t expected = static_cast<uint8_t>(partition + 1u);
        const uint64_t begin = static_cast<uint64_t>(partition) * stride;
        for (uint64_t byte = 0u; byte < stride; ++byte)
            CHECK(arena[begin + byte] == expected);
    }
    CHECK(std::all_of(storage.begin(), storage.begin() + guard,
                      [](uint8_t value) { return value == guard_value; }));
    CHECK(std::all_of(storage.end() - guard, storage.end(),
                      [](uint8_t value) { return value == guard_value; }));
}

void TestDisjointPartitionsAndTailPreservation()
{
    const SourcePartitionLayout layout = MakeLayout();
    const uint32_t partitions =
        layout.config.worker_count * layout.config.ring_count;
    CheckGuardedDisjointArena(layout.destination.hidden_arena_bytes,
                              layout.destination.hidden_partition_stride,
                              partitions);
    CheckGuardedDisjointArena(layout.source.arena_bytes,
                              layout.source.packet_stride, partitions);
    CheckGuardedDisjointArena(layout.destination.rows_arena_bytes,
                              layout.destination.rows_partition_stride,
                              partitions);
    CheckGuardedDisjointArena(layout.destination.assignments_arena_bytes,
                              layout.destination.assignments_partition_stride,
                              partitions);
    CheckGuardedDisjointArena(
        layout.destination.expert_counts_arena_bytes,
        layout.destination.expert_counts_partition_stride, partitions);
    CheckGuardedDisjointArena(layout.combine.partial_arena_bytes,
                              layout.combine.partial_partition_stride,
                              partitions);
    CheckGuardedDisjointArena(layout.combine.output_arena_bytes,
                              layout.combine.output_ring_stride,
                              layout.config.ring_count);
    CheckGuardedDisjointArena(layout.inc_dispatch.arena_bytes,
                              layout.inc_dispatch.partition_stride,
                              partitions);
    CheckGuardedDisjointArena(layout.inc_journal.arena_bytes,
                              layout.inc_journal.partition_stride,
                              partitions);
    CheckGuardedDisjointArena(layout.inc_combine.arena_bytes,
                              layout.inc_combine.partition_stride,
                              partitions);
    CheckGuardedDisjointArena(layout.origin_ring_control64_bytes, 64u,
                              partitions * layout.config.worker_count);
    CheckGuardedDisjointArena(layout.origin_ring_control128_bytes, 128u,
                              partitions * layout.config.worker_count);

    constexpr uint8_t poison = 0xccu;
    std::vector<uint8_t> hidden(
        static_cast<size_t>(layout.destination.hidden_arena_bytes), poison);
    uint64_t begin = 0u;
    CHECK(SourcePartitionOffsets::DestinationHidden(
        layout, 0u, 1u, 0u, 0u, &begin));
    const uint32_t live_rows = 5u;
    std::memset(hidden.data() + begin, 0x31,
                static_cast<size_t>(live_rows * layout.dispatch_row_bytes));
    const uint64_t tail_begin =
        begin + live_rows * layout.dispatch_row_bytes;
    const uint64_t partition_end =
        begin + layout.destination.hidden_partition_stride;
    using Difference = std::vector<uint8_t>::difference_type;
    CHECK(std::all_of(hidden.begin() + static_cast<Difference>(tail_begin),
                      hidden.begin() + static_cast<Difference>(partition_end),
                      [](uint8_t value) { return value == poison; }));

    // A zero-row source touches neither its own partition nor its neighbors.
    uint64_t zero_begin = 0u;
    CHECK(SourcePartitionOffsets::DestinationHidden(
        layout, 0u, 2u, 0u, 0u, &zero_begin));
    CHECK(std::all_of(
        hidden.begin() + static_cast<Difference>(zero_begin),
        hidden.begin() + static_cast<Difference>(
                             zero_begin +
                             layout.destination.hidden_partition_stride),
        [](uint8_t value) { return value == poison; }));
}

void TestZeroCapacityStillHasIndependentPartitions()
{
    SourcePartitionConfig config = StandardConfig();
    config.max_source_tokens = 0u;
    config.max_source_assignments = 0u;
    const SourcePartitionLayout layout = MakeLayout(config);
    CHECK(layout.row_capacity == 0u);
    CHECK(layout.assignment_capacity == 0u);
    CHECK(layout.contributor_capacity == 0u);
    CHECK(layout.destination.hidden_partition_stride == 64u);
    CHECK(layout.destination.rows_partition_stride == 64u);
    CHECK(layout.destination.assignments_partition_stride == 64u);
    CHECK(layout.combine.partial_partition_stride == 64u);
    CHECK(ValidateSourcePartitionLiveCounts(layout, 0u, 0u) ==
          SourcePartitionLayoutStatus::OK);
    uint64_t offset = 0u;
    CHECK(!SourcePartitionOffsets::DestinationHidden(
        layout, 0u, 0u, 0u, 0u, &offset));
    CHECK(SourcePartitionOffsets::IncJournal(layout, 1u, 0u, &offset));
    CHECK(offset == layout.config.ring_count *
                        layout.inc_journal.partition_stride);
}

void TestInvalidAndOverflowConfigurations()
{
    SourcePartitionLayout layout{};
    std::string error;
    SourcePartitionConfig config = StandardConfig();
    CHECK(BuildSourcePartitionLayout(config, nullptr, &error) ==
          SourcePartitionLayoutStatus::INVALID_ARGUMENT);
    config.worker_count = 129u;
    CHECK(BuildSourcePartitionLayout(config, &layout, &error) ==
          SourcePartitionLayoutStatus::INVALID_ARGUMENT);
    config = StandardConfig();
    config.ring_count = 0u;
    CHECK(BuildSourcePartitionLayout(config, &layout, &error) ==
          SourcePartitionLayoutStatus::INVALID_ARGUMENT);
    config = StandardConfig();
    config.ring_count =
        static_cast<uint32_t>(std::numeric_limits<uint16_t>::max()) + 1u;
    CHECK(BuildSourcePartitionLayout(config, &layout, &error) ==
          SourcePartitionLayoutStatus::INVALID_ARGUMENT);
    config = StandardConfig();
    config.hidden = 0u;
    CHECK(BuildSourcePartitionLayout(config, &layout, &error) ==
          SourcePartitionLayoutStatus::INVALID_ARGUMENT);
    config = StandardConfig();
    config.dispatch_aiv_per_origin = 1u;
    CHECK(BuildSourcePartitionLayout(config, &layout, &error) ==
          SourcePartitionLayoutStatus::INVALID_ARGUMENT);
    config = StandardConfig();
    config.dtype = static_cast<DataType>(99u);
    CHECK(BuildSourcePartitionLayout(config, &layout, &error) ==
          SourcePartitionLayoutStatus::INVALID_ARGUMENT);

    config = StandardConfig();
    config.max_source_tokens = std::numeric_limits<uint32_t>::max();
    CHECK(BuildSourcePartitionLayout(config, &layout, &error) ==
          SourcePartitionLayoutStatus::SIZE_OVERFLOW);
    config = StandardConfig();
    config.max_source_assignments = std::numeric_limits<uint32_t>::max();
    CHECK(BuildSourcePartitionLayout(config, &layout, &error) ==
          SourcePartitionLayoutStatus::SIZE_OVERFLOW);
    config = StandardConfig();
    config.expert_count = std::numeric_limits<uint32_t>::max();
    CHECK(BuildSourcePartitionLayout(config, &layout, &error) ==
          SourcePartitionLayoutStatus::SIZE_OVERFLOW);

    // Rounded capacities are representable, but the hidden arena product is
    // not.  This exercises checked byte multiplication, not just rounding.
    config.worker_count = 128u;
    config.ring_count = std::numeric_limits<uint16_t>::max();
    config.max_source_tokens =
        std::numeric_limits<uint32_t>::max() - 31u;
    config.max_source_assignments = 4u;
    config.hidden = std::numeric_limits<uint32_t>::max();
    config.expert_count = 16u;
    config.dtype = DataType::FP32;
    CHECK(BuildSourcePartitionLayout(config, &layout, &error) ==
          SourcePartitionLayoutStatus::SIZE_OVERFLOW);
    CHECK(SourcePartitionLayoutStatusString(
              SourcePartitionLayoutStatus::SIZE_OVERFLOW) ==
          std::string("size_overflow"));
}

} // namespace

int main()
{
    TestConfiguredCapacitiesAndStrides();
    TestDTypePartitionBoundaries();
    TestKeyOffsetsAndBounds();
    TestArrivalAndCountsCannotMovePartitions();
    TestControlAndPrivateOriginOrdering();
    TestDisjointPartitionsAndTailPreservation();
    TestZeroCapacityStillHasIndependentPartitions();
    TestInvalidAndOverflowConfigurations();
    if (failures != 0) {
        std::cerr << failures << " source partition layout checks failed\n";
        return 1;
    }
    std::cout << "source partition layout checks passed\n";
    return 0;
}
