#include "inc_dc_native_expert_layout_adapter.h"

#include <algorithm>
#include <limits>
#include <unordered_map>

namespace inc::dc::single_stream {

namespace {

bool PowerOfTwo(uint32_t value)
{
    return value != 0u && (value & (value - 1u)) == 0u;
}

bool AlignUp(uint64_t value, uint32_t alignment, uint64_t *aligned)
{
    if (aligned == nullptr || !PowerOfTwo(alignment) ||
        value > std::numeric_limits<uint64_t>::max() - (alignment - 1u))
        return false;
    *aligned = (value + alignment - 1u) &
        ~static_cast<uint64_t>(alignment - 1u);
    return true;
}

struct Item {
    uint32_t expert_index = 0u;
    uint32_t expert_id = 0u;
    uint32_t dispatch_row = 0u;
    uint32_t combine_row = 0u;
    uint32_t result = 0u;
    uint32_t ordinal = 0u;
};

} // namespace

bool BuildNativeExpertLayout(
    const StreamCompiledGlobalPlan &dispatch,
    const CombineReverseLayout &combine,
    const std::vector<std::vector<uint32_t>> &configured_expert_ids,
    uint32_t expert_alignment, NativeExpertLayout *layout)
{
    if (layout == nullptr || !PowerOfTwo(expert_alignment) ||
        dispatch.worker_world_size < 2u ||
        configured_expert_ids.size() != dispatch.worker_world_size ||
        combine.contributor_rows.size() != dispatch.worker_world_size ||
        combine.contributor_dispatch_rows.size() !=
            dispatch.worker_world_size ||
        combine.logical_plan.contribution_count !=
            dispatch.logical_assignments) return false;

    std::vector<std::unordered_map<uint32_t, uint32_t>> expert_index(
        dispatch.worker_world_size);
    for (uint32_t rank = 0u; rank < dispatch.worker_world_size; ++rank) {
        if (configured_expert_ids[rank].empty()) return false;
        for (uint32_t index = 0u;
             index < configured_expert_ids[rank].size(); ++index) {
            if (!expert_index[rank]
                     .emplace(configured_expert_ids[rank][index], index)
                     .second) return false;
        }
    }

    std::vector<std::vector<Item>> items(dispatch.worker_world_size);
    std::vector<uint32_t> destination_row(dispatch.worker_world_size, 0u);
    for (const auto &task : dispatch.tasks) {
        if (task.destination_rank >= dispatch.worker_world_size ||
            task.route_begin > dispatch.routes.size() ||
            task.route_count > dispatch.routes.size() - task.route_begin)
            return false;
        for (uint32_t ri = 0u; ri < task.route_count; ++ri) {
            const auto &route = dispatch.routes[task.route_begin + ri];
            const uint32_t physical =
                destination_row[task.destination_rank]++;
            if (route.source_rank >= dispatch.worker_world_size ||
                route.source_row >= dispatch.tokens_per_worker ||
                route.assignment_begin > dispatch.assignments.size() ||
                route.assignment_count >
                    dispatch.assignments.size() - route.assignment_begin)
                return false;
            const uint32_t result = route.source_rank *
                dispatch.tokens_per_worker + route.source_row;
            for (uint32_t ai = 0u; ai < route.assignment_count; ++ai) {
                const auto &assignment =
                    dispatch.assignments[route.assignment_begin + ai];
                const auto found = expert_index[task.destination_rank].find(
                    assignment.expert_id);
                if (found == expert_index[task.destination_rank].end() ||
                    assignment.route_ordinal >= dispatch.topk) return false;
                const uint64_t contribution_index =
                    static_cast<uint64_t>(result) * dispatch.topk +
                    assignment.route_ordinal;
                if (contribution_index >=
                    combine.logical_plan.contributions.size()) return false;
                const auto &contribution =
                    combine.logical_plan.contributions[contribution_index];
                if (contribution.result_id != result ||
                    contribution.ordinal != assignment.route_ordinal ||
                    contribution.contributor_rank != task.destination_rank ||
                    contribution.contributor_local_row >=
                        combine.contributor_rows[task.destination_rank] ||
                    combine.contributor_dispatch_rows[task.destination_rank]
                        [contribution.contributor_local_row] != physical)
                    return false;
                items[task.destination_rank].push_back({
                    found->second, assignment.expert_id, physical,
                    contribution.contributor_local_row, result,
                    assignment.route_ordinal});
            }
        }
    }

    NativeExpertLayout built{};
    built.expert_alignment = expert_alignment;
    built.ranks.resize(dispatch.worker_world_size);
    for (uint32_t rank = 0u; rank < dispatch.worker_world_size; ++rank) {
        if (destination_row[rank] != dispatch.destination_physical_rows[rank] ||
            items[rank].size() != combine.contributor_rows[rank]) return false;
        auto &rank_items = items[rank];
        std::sort(rank_items.begin(), rank_items.end(),
                  [](const Item &a, const Item &b) {
            if (a.expert_index != b.expert_index)
                return a.expert_index < b.expert_index;
            if (a.result != b.result) return a.result < b.result;
            return a.ordinal < b.ordinal;
        });
        NativeExpertRankLayout &out = built.ranks[rank];
        out.expert_ids = configured_expert_ids[rank];
        const size_t experts = out.expert_ids.size();
        out.tokens_per_expert.assign(experts, 0u);
        for (const Item &item : rank_items)
            ++out.tokens_per_expert[item.expert_index];
        out.expert_offsets.assign(experts + 1u, 0u);
        out.padded_expert_offsets.assign(experts + 1u, 0u);
        for (size_t expert = 0u; expert < experts; ++expert) {
            out.expert_offsets[expert + 1u] =
                out.expert_offsets[expert] + out.tokens_per_expert[expert];
            uint64_t padded = 0u;
            if (!AlignUp(out.tokens_per_expert[expert], expert_alignment,
                         &padded) ||
                out.padded_expert_offsets[expert] >
                    std::numeric_limits<uint64_t>::max() - padded)
                return false;
            out.padded_expert_offsets[expert + 1u] =
                out.padded_expert_offsets[expert] + padded;
        }
        out.padded_row_count = out.padded_expert_offsets.back();
        out.combine_row_to_padded_row.assign(
            combine.contributor_rows[rank],
            std::numeric_limits<uint64_t>::max());
        std::vector<uint64_t> within(experts, 0u);
        for (const Item &item : rank_items) {
            const uint64_t padded_row =
                out.padded_expert_offsets[item.expert_index] +
                within[item.expert_index]++;
            out.dispatch_rows.push_back(item.dispatch_row);
            out.combine_rows.push_back(item.combine_row);
            out.padded_rows.push_back(padded_row);
            if (out.combine_row_to_padded_row[item.combine_row] !=
                std::numeric_limits<uint64_t>::max()) return false;
            out.combine_row_to_padded_row[item.combine_row] = padded_row;
        }
        for (uint64_t row : out.combine_row_to_padded_row)
            if (row == std::numeric_limits<uint64_t>::max()) return false;
    }
    *layout = std::move(built);
    return true;
}

} // namespace inc::dc::single_stream
