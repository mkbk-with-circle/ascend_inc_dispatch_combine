#include "inc_dc_single_inc_combine_plan_compiler.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace inc::dc::single_stream {

bool BuildCombineReverseLayout(
    const StreamCompiledGlobalPlan &dispatch,
    CombineReverseLayout *combine)
{
    if (combine == nullptr || dispatch.worker_world_size < 2u ||
        dispatch.tokens_per_worker == 0u || dispatch.topk == 0u ||
        dispatch.assignments.size() != dispatch.logical_assignments ||
        dispatch.routes.size() != dispatch.physical_rows) {
        return false;
    }
    const uint64_t result_count64 =
        static_cast<uint64_t>(dispatch.worker_world_size) *
        dispatch.tokens_per_worker;
    const uint64_t contribution_count64 = result_count64 * dispatch.topk;
    if (result_count64 > std::numeric_limits<uint32_t>::max() ||
        contribution_count64 > std::numeric_limits<uint32_t>::max() ||
        contribution_count64 != dispatch.logical_assignments) {
        return false;
    }

    struct Pending {
        uint32_t ordinal = 0u;
        uint32_t contributor = 0u;
        uint32_t contributor_dispatch_row = 0u;
        float weight = 0.0f;
    };
    std::vector<std::vector<Pending>> pending(result_count64);
    std::vector<uint32_t> destination_rows(dispatch.worker_world_size, 0u);
    for (const auto &task : dispatch.tasks) {
        if (task.destination_rank >= dispatch.worker_world_size ||
            task.route_begin > dispatch.routes.size() ||
            task.route_count > dispatch.routes.size() - task.route_begin) {
            return false;
        }
        for (uint32_t ri = 0u; ri < task.route_count; ++ri) {
            const auto &route = dispatch.routes[task.route_begin + ri];
            const uint32_t dispatch_row =
                destination_rows[task.destination_rank]++;
            if (route.source_rank >= dispatch.worker_world_size ||
                route.source_row >= dispatch.tokens_per_worker ||
                route.assignment_begin > dispatch.assignments.size() ||
                route.assignment_count >
                    dispatch.assignments.size() - route.assignment_begin) {
                return false;
            }
            const uint64_t result =
                static_cast<uint64_t>(route.source_rank) *
                    dispatch.tokens_per_worker + route.source_row;
            for (uint32_t ai = 0u; ai < route.assignment_count; ++ai) {
                const auto &assignment =
                    dispatch.assignments[route.assignment_begin + ai];
                float weight = 0.0f;
                std::memcpy(&weight, &assignment.weight_bits,
                            sizeof(weight));
                pending[result].push_back({
                    assignment.route_ordinal, task.destination_rank,
                    dispatch_row, weight});
            }
        }
    }
    for (uint32_t rank = 0u; rank < dispatch.worker_world_size; ++rank) {
        if (destination_rows[rank] !=
            dispatch.destination_physical_rows[rank]) return false;
    }

    IncDcCombineLogicalPlanV2 plan{};
    plan.worker_world_size = dispatch.worker_world_size;
    plan.result_count = static_cast<uint32_t>(result_count64);
    plan.contribution_count =
        static_cast<uint32_t>(contribution_count64);
    plan.declared_max_topk = dispatch.topk;
    plan.uniform_topk_valid = true;
    plan.uniform_topk = dispatch.topk;
    plan.results.reserve(plan.result_count);
    plan.contributions.reserve(plan.contribution_count);
    std::vector<uint64_t> contributor_rows(dispatch.worker_world_size, 0u);
    std::vector<std::vector<uint32_t>> contributor_dispatch_rows(
        dispatch.worker_world_size);
    for (uint32_t result = 0u; result < plan.result_count; ++result) {
        auto &items = pending[result];
        if (items.size() != dispatch.topk) return false;
        std::sort(items.begin(), items.end(), [](const Pending &a,
                                                 const Pending &b) {
            return a.ordinal < b.ordinal;
        });
        IncDcLogicalResultV2 output{};
        output.dst_rank = result / dispatch.tokens_per_worker;
        output.dst_local_row = result % dispatch.tokens_per_worker;
        output.contribution_begin =
            static_cast<uint32_t>(plan.contributions.size());
        output.contribution_count = dispatch.topk;
        for (uint32_t ordinal = 0u; ordinal < dispatch.topk; ++ordinal) {
            if (items[ordinal].ordinal != ordinal) return false;
            IncDcLogicalContributionV2 contribution{};
            contribution.contribution_uid =
                static_cast<uint64_t>(result) * dispatch.topk + ordinal + 1u;
            contribution.result_id = result;
            contribution.ordinal = ordinal;
            contribution.contributor_rank = items[ordinal].contributor;
            if (contributor_rows[items[ordinal].contributor] >
                std::numeric_limits<uint32_t>::max()) return false;
            contribution.contributor_local_row = static_cast<uint32_t>(
                contributor_rows[items[ordinal].contributor]++);
            contribution.weight = items[ordinal].weight;
            plan.contributions.push_back(contribution);
            contributor_dispatch_rows[items[ordinal].contributor].push_back(
                items[ordinal].contributor_dispatch_row);
        }
        plan.results.push_back(output);
    }
    plan.semantic_digest = ComputeLogicalPlanSemanticDigest(plan);
    IncDcLogicalPlanValidateReport report{};
    if (ValidateLogicalPlanV2(plan, &report) != IncDcStatus::OK) return false;
    combine->logical_plan = std::move(plan);
    combine->contributor_rows = std::move(contributor_rows);
    combine->contributor_dispatch_rows =
        std::move(contributor_dispatch_rows);
    return true;
}

} // namespace inc::dc::single_stream
