#include "inc_dc_native_combine_workspace.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "inc_dc_combine_topology.h"

namespace inc::dc::single_stream {
namespace {

uint64_t Align64(uint64_t value)
{
    return (value + 63u) & ~63ull;
}

bool Reserve(uint64_t bytes, uint64_t *cursor, uint64_t *offset)
{
    if (cursor == nullptr || offset == nullptr ||
        bytes > std::numeric_limits<uint64_t>::max() - *cursor) {
        return false;
    }
    *offset = *cursor;
    const uint64_t next = *cursor + bytes;
    if (next > std::numeric_limits<uint64_t>::max() - 63u) return false;
    *cursor = Align64(next);
    return true;
}

template <class T>
void Store(std::vector<uint8_t> *image, uint64_t offset,
           const std::vector<T> &values)
{
    if (!values.empty()) {
        std::memcpy(image->data() + offset, values.data(),
                    values.size() * sizeof(T));
    }
}

uint32_t FloatBits(float value)
{
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

} // namespace

bool BuildNativeCombinePreparedWorkspace(
    const IncDcCombineLogicalPlanV2 &logical,
    const std::vector<uint64_t> &contributor_rows,
    uint32_t hidden, uint32_t live_aiv,
    NativeCombinePreparedWorkspace *workspace)
{
    if (workspace == nullptr || hidden == 0u || live_aiv == 0u ||
        logical.worker_world_size < 2u ||
        contributor_rows.size() != logical.worker_world_size) {
        return false;
    }
    IncDcLogicalPlanValidateReport validation{};
    if (ValidateLogicalPlanV2(logical, &validation) != IncDcStatus::OK)
        return false;
    IncDcAivPolicy resources{};
    if (!IncDcResolveAivPolicy(live_aiv, logical.worker_world_size,
                               kDynCsrMaxOwners, kDynCsrMaxOwners,
                               &resources)) {
        return false;
    }
    const uint32_t workers = logical.worker_world_size;
    const uint32_t owners = resources.combine_inc_aiv;
    IncDcTopologyDescriptor topology{};
    if (BuildSingleIncTopology(workers, owners, 0u, workers, 1u,
                               &topology) != IncDcStatus::OK) {
        return false;
    }
    for (uint32_t rank = 0u; rank < workers; ++rank)
        topology.worker_pe_ids[rank] = rank;
    topology.inc_pe = workers;
    topology.topology_digest = ComputeTopologyDigest(topology);
    IncDcCompiledExecutionPlan execution{};
    IncDcPlanCompileReport compile{};
    if (CompileLogicalPlanToExecution(logical, topology, hidden, 2u,
                                      &execution, &compile) !=
        IncDcStatus::OK) {
        return false;
    }
    uint32_t max_slot = 0u;
    for (const auto &entry : execution.schedule)
        max_slot = std::max(max_slot, entry.ingress_slot);
    const uint32_t slot_count = max_slot + 1u;
    const uint32_t group_count = owners * workers;
    const uint32_t bitmap_words = (workers + 31u) / 32u;
    const uint64_t payload_bytes = static_cast<uint64_t>(hidden) * 2u;
    if (payload_bytes > std::numeric_limits<uint32_t>::max() - 63u)
        return false;
    const uint32_t tile_bytes =
        static_cast<uint32_t>(Align64(payload_bytes));

    DynCsrCtrl control{};
    control.result_count = logical.result_count;
    control.contribution_count = logical.contribution_count;
    control.hidden = hidden;
    control.tile_bytes = tile_bytes;
    control.element_bytes = 2u;
    control.owner_count = owners;
    control.inc_pe = workers;
    control.generation = 1u;
    control.fail_closed_on_dup = 1u;
    control.max_ingress_slots = slot_count;
    control.producer_lane_count = resources.combine_worker_aiv;
    control.producer_quiet_window = 1u;
    control.overlap_enable = 1u;
    control.ready_stride_bytes = 64u;
    control.ready_spin_cap = 40000000u;
    control.worker_count = workers;
    control.group_count = group_count;
    control.source_bitmap_words = bitmap_words;
    control.ready_mode = 6u;
    control.output_dcci_small_only = 1u;
    control.device_completion = 1u;
    control.tx_quiet_window =
        tile_bytes <= 16u * 1024u &&
                validation.expected_count_max <= 2u
            ? 32u : 1u;
    control.optimization_flags =
        kDynCsrOptLocalOrdinalBitmap |
        kDynCsrOptK1IdentityCopy |
        kDynCsrOptFirstContributionInit |
        kDynCsrOptK1PrivateMtePush |
        kDynCsrOptK1PairReady |
        kDynCsrOptRemoteResultTx |
        kDynCsrOptCoalescedGroupPut |
        kDynCsrOptCyclicOwnerResults |
        kDynCsrOptWideVectorTile;
    if (tile_bytes > 16u * 1024u)
        control.optimization_flags |= kDynCsrOptBatchResultTx;
    const bool identity_k1 = validation.expected_count_min == 1u &&
        validation.expected_count_max == 1u;
    const uint32_t owners_per_producer = std::max(
        1u, (owners + control.producer_lane_count - 1u) /
                control.producer_lane_count);
    const uint64_t chunk_tiles = identity_k1 ? 1u :
        (static_cast<uint64_t>(workers) * 3u >= owners ||
         owners_per_producer <= 2u) ? 4u : 64u;
    control.coalesced_chunk_bytes = static_cast<uint32_t>(
        std::max<uint64_t>(tile_bytes,
            std::min<uint64_t>(512u * 1024u,
                static_cast<uint64_t>(tile_bytes) * chunk_tiles)));

    uint64_t cursor = Align64(kDynCsrCtrlBytes);
    uint64_t packed_result_ids_off = 0u;
    uint64_t local_reduce_offsets_off = 0u;
    uint64_t local_reduce_entries_off = 0u;
    uint64_t local_reduce_weights_off = 0u;
    uint64_t logical_input_off = 0u;
    const uint64_t ready_record_count = std::max<uint64_t>(
        static_cast<uint64_t>(slot_count) + workers,
        static_cast<uint64_t>(group_count) + workers +
            static_cast<uint64_t>(workers) * workers);
    bool ok =
        Reserve((static_cast<uint64_t>(logical.result_count) + 1u) * 4u,
                &cursor, &control.result_offsets_off) &&
        Reserve(static_cast<uint64_t>(logical.result_count) * 4u,
                &cursor, &control.result_home_owner_off) &&
        Reserve(static_cast<uint64_t>(logical.result_count) * 4u,
                &cursor, &control.result_dst_rank_off) &&
        Reserve(static_cast<uint64_t>(logical.result_count) * 4u,
                &cursor, &control.result_dst_row_off) &&
        Reserve((static_cast<uint64_t>(workers) + 1u) * 4u,
                &cursor, &control.result_tx_rank_offsets_off) &&
        Reserve(static_cast<uint64_t>(logical.result_count) * 4u,
                &cursor, &packed_result_ids_off) &&
        Reserve(static_cast<uint64_t>(logical.contribution_count) * 4u,
                &cursor, &control.contrib_slot_off) &&
        Reserve(static_cast<uint64_t>(logical.contribution_count) * 4u,
                &cursor, &control.contrib_weight_off) &&
        Reserve(static_cast<uint64_t>(logical.contribution_count) * 4u,
                &cursor, &control.contrib_uid_lo_off) &&
        Reserve(static_cast<uint64_t>(logical.contribution_count) * 4u,
                &cursor, &control.contrib_uid_hi_off) &&
        Reserve(static_cast<uint64_t>(logical.contribution_count) * 4u,
                &cursor, &control.contrib_ordinal_off) &&
        Reserve(static_cast<uint64_t>(logical.contribution_count) * 8u,
                &cursor, &control.contrib_gen_off) &&
        Reserve(static_cast<uint64_t>(logical.contribution_count) * 4u,
                &cursor, &control.contrib_source_rank_off) &&
        Reserve(static_cast<uint64_t>(logical.contribution_count) * 4u,
                &cursor, &control.contrib_owner_off) &&
        Reserve(static_cast<uint64_t>(logical.contribution_count) * 4u,
                &cursor, &control.contrib_result_off) &&
        Reserve((static_cast<uint64_t>(group_count) + 1u) * 4u,
                &cursor, &control.group_offsets_off) &&
        Reserve(static_cast<uint64_t>(logical.contribution_count) * 4u,
                &cursor, &control.group_entries_off) &&
        Reserve((static_cast<uint64_t>(workers) + 1u) * 4u,
                &cursor, &control.source_contribution_offsets_off) &&
        Reserve(static_cast<uint64_t>(logical.contribution_count) * 4u,
                &cursor, &control.source_contribution_entries_off) &&
        Reserve((static_cast<uint64_t>(workers) + 1u) * 4u,
                &cursor, &control.source_group_offsets_off) &&
        Reserve(static_cast<uint64_t>(group_count) * 4u,
                &cursor, &control.source_group_entries_off) &&
        Reserve(static_cast<uint64_t>(owners) * bitmap_words * 4u,
                &cursor, &control.owner_source_bitmap_off) &&
        Reserve(static_cast<uint64_t>(owners) * bitmap_words * 4u,
                &cursor, &control.waited_source_bitmap_off) &&
        Reserve(static_cast<uint64_t>(workers) * 4u,
                &cursor, &control.worker_pe_off) &&
        Reserve((static_cast<uint64_t>(workers) + 2u) * 64u,
                &cursor, &control.start_gate_off) &&
        Reserve(ready_record_count * 64u,
                &cursor, &control.ready_generation_off) &&
        Reserve(static_cast<uint64_t>(logical.result_count) * 64u,
                &cursor, &control.result_arrival_counter_off) &&
        Reserve(0u, &cursor, &local_reduce_offsets_off) &&
        Reserve(0u, &cursor, &local_reduce_entries_off) &&
        Reserve(0u, &cursor, &local_reduce_weights_off) &&
        Reserve(0u, &cursor, &logical_input_off) &&
        Reserve(static_cast<uint64_t>(slot_count) * tile_bytes,
                &cursor, &control.ingress_off) &&
        Reserve(static_cast<uint64_t>(logical.result_count) * tile_bytes,
                &cursor, &control.output_off) &&
        Reserve(static_cast<uint64_t>(logical.result_count) * 64u,
                &cursor, &control.result_tx_ready_off) &&
        Reserve(0u, &cursor, &control.tx_done_off) &&
        Reserve((static_cast<uint64_t>(logical.result_count) +
                 logical.contribution_count) * 8u,
                &cursor, &control.arrival_off) &&
        Reserve(sizeof(DynCsrStats), &cursor, &control.stats_off) &&
        Reserve(static_cast<uint64_t>(kDynCsrMaxOwners) *
                    sizeof(DynCsrOwnerStats),
                &cursor, &control.owner_stats_off);
    (void)packed_result_ids_off;
    control.local_reduce_offsets_off = local_reduce_offsets_off;
    control.local_reduce_entries_off = local_reduce_entries_off;
    control.local_reduce_weights_off = local_reduce_weights_off;
    control.logical_input_off = logical_input_off;
    if (!ok || cursor > std::numeric_limits<size_t>::max() - 4096u)
        return false;
    const uint64_t heap_bytes = Align64(cursor + 4096u);
    std::vector<uint8_t> image(static_cast<size_t>(heap_bytes), 0u);

    std::vector<uint32_t> result_dst_rank(logical.result_count);
    std::vector<uint32_t> result_dst_row(logical.result_count);
    std::vector<uint32_t> contribution_result(logical.contribution_count);
    for (uint32_t result = 0u; result < logical.result_count; ++result) {
        result_dst_rank[result] = logical.results[result].dst_rank;
        result_dst_row[result] = logical.results[result].dst_local_row;
        for (uint32_t si = execution.result_offsets[result];
             si < execution.result_offsets[result + 1u]; ++si)
            contribution_result[si] = result;
    }

    std::vector<std::vector<uint32_t>> grouped(group_count);
    std::vector<std::vector<uint32_t>> source_contributions(workers);
    std::vector<uint32_t> contribution_owner(logical.contribution_count);
    std::vector<uint32_t> owner_bitmap(
        static_cast<size_t>(owners) * bitmap_words, 0u);
    for (uint32_t si = 0u; si < execution.schedule.size(); ++si) {
        const auto &compiled = execution.schedule[si];
        const auto &entry =
            logical.contributions[compiled.logical_contribution_index];
        const uint32_t flat = compiled.owner_index;
        if (flat >= owners || entry.contributor_rank >= workers)
            return false;
        contribution_owner[si] = flat;
        grouped[flat * workers + entry.contributor_rank].push_back(si);
        source_contributions[entry.contributor_rank].push_back(si);
        owner_bitmap[static_cast<size_t>(flat) * bitmap_words +
                     (entry.contributor_rank >> 5u)] |=
            1u << (entry.contributor_rank & 31u);
    }
    std::vector<uint32_t> group_offsets(group_count + 1u);
    std::vector<uint32_t> group_entries;
    std::vector<uint32_t> source_group_offsets(workers + 1u);
    std::vector<uint32_t> source_group_entries;
    for (uint32_t group = 0u; group < group_count; ++group) {
        group_offsets[group] = static_cast<uint32_t>(group_entries.size());
        group_entries.insert(group_entries.end(), grouped[group].begin(),
                             grouped[group].end());
    }
    group_offsets[group_count] = static_cast<uint32_t>(group_entries.size());
    for (uint32_t source = 0u; source < workers; ++source) {
        source_group_offsets[source] =
            static_cast<uint32_t>(source_group_entries.size());
        for (uint32_t owner = 0u; owner < owners; ++owner) {
            if (!grouped[owner * workers + source].empty())
                source_group_entries.push_back(owner);
        }
    }
    source_group_offsets[workers] =
        static_cast<uint32_t>(source_group_entries.size());
    std::vector<uint32_t> source_contribution_offsets(workers + 1u);
    std::vector<uint32_t> source_contribution_entries;
    uint32_t packed_slot = 0u;
    for (uint32_t source = 0u; source < workers; ++source) {
        source_contribution_offsets[source] =
            static_cast<uint32_t>(source_contribution_entries.size());
        for (uint32_t si : source_contributions[source])
            execution.schedule[si].ingress_slot = packed_slot++;
        source_contribution_entries.insert(source_contribution_entries.end(),
            source_contributions[source].begin(),
            source_contributions[source].end());
    }
    source_contribution_offsets[workers] =
        static_cast<uint32_t>(source_contribution_entries.size());
    if (packed_slot != logical.contribution_count ||
        packed_slot > slot_count) return false;

    std::vector<uint32_t> slots(logical.contribution_count);
    std::vector<uint32_t> weights(logical.contribution_count);
    std::vector<uint32_t> uid_lo(logical.contribution_count);
    std::vector<uint32_t> uid_hi(logical.contribution_count);
    std::vector<uint32_t> ordinals(logical.contribution_count);
    std::vector<uint64_t> generations(logical.contribution_count, 0u);
    std::vector<uint32_t> sources(logical.contribution_count);
    for (uint32_t si = 0u; si < execution.schedule.size(); ++si) {
        const auto &compiled = execution.schedule[si];
        const auto &entry =
            logical.contributions[compiled.logical_contribution_index];
        slots[si] = compiled.ingress_slot;
        weights[si] = FloatBits(entry.weight);
        uid_lo[si] = static_cast<uint32_t>(entry.contribution_uid);
        uid_hi[si] = static_cast<uint32_t>(entry.contribution_uid >> 32u);
        ordinals[si] = entry.ordinal;
        sources[si] = entry.contributor_rank;
    }
    std::vector<uint32_t> worker_pes(workers);
    for (uint32_t rank = 0u; rank < workers; ++rank) worker_pes[rank] = rank;

    Store(&image, control.result_offsets_off, execution.result_offsets);
    Store(&image, control.result_home_owner_off, execution.result_home_owner);
    Store(&image, control.result_dst_rank_off, result_dst_rank);
    Store(&image, control.result_dst_row_off, result_dst_row);
    Store(&image, control.contrib_slot_off, slots);
    Store(&image, control.contrib_weight_off, weights);
    Store(&image, control.contrib_uid_lo_off, uid_lo);
    Store(&image, control.contrib_uid_hi_off, uid_hi);
    Store(&image, control.contrib_ordinal_off, ordinals);
    Store(&image, control.contrib_gen_off, generations);
    Store(&image, control.contrib_source_rank_off, sources);
    Store(&image, control.contrib_owner_off, contribution_owner);
    Store(&image, control.contrib_result_off, contribution_result);
    Store(&image, control.group_offsets_off, group_offsets);
    Store(&image, control.group_entries_off, group_entries);
    Store(&image, control.source_contribution_offsets_off,
          source_contribution_offsets);
    Store(&image, control.source_contribution_entries_off,
          source_contribution_entries);
    Store(&image, control.source_group_offsets_off, source_group_offsets);
    Store(&image, control.source_group_entries_off, source_group_entries);
    Store(&image, control.owner_source_bitmap_off, owner_bitmap);
    Store(&image, control.worker_pe_off, worker_pes);
    std::memcpy(image.data(), &control, sizeof(control));

    workspace->control = control;
    workspace->resources = resources;
    workspace->execution = std::move(execution);
    workspace->heap_bytes = heap_bytes;
    workspace->immutable_image = std::move(image);
    workspace->contributor_rows = contributor_rows;
    workspace->input_copies.assign(workers, {});
    for (uint32_t si = 0u; si < workspace->execution.schedule.size(); ++si) {
        const auto &compiled = workspace->execution.schedule[si];
        const auto &entry =
            logical.contributions[compiled.logical_contribution_index];
        workspace->input_copies[entry.contributor_rank].push_back({
            entry.contributor_local_row, compiled.ingress_slot});
    }
    return true;
}

} // namespace inc::dc::single_stream
