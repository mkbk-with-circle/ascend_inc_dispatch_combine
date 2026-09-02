#include "inc_dc_pull_combine_plan.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

using namespace inc::dc::pull_combine;

namespace {

WavePlanDesc ValidDesc()
{
    WavePlanDesc desc{};
    desc.worker_count = 4u;
    desc.expert_count = 32u;
    desc.hidden = 16u;
    desc.element_bytes = 2u;
    desc.wave = 3u;
    desc.generation = 17u;
    desc.tokens_per_rank = {2u, 1u, 0u, 1u};
    desc.routes = {
        {0u, 0u, {{1u, 0u, 0u, 0.25f},
                  {1u, 1u, 1u, 0.25f},
                  {2u, 2u, 2u, 0.50f}}},
        {0u, 1u, {}},
        {1u, 0u, {{2u, 3u, 0u, 1.0f}}},
        {3u, 0u, {{0u, 4u, 0u, 0.5f},
                  {2u, 5u, 1u, 0.5f}}},
    };
    return desc;
}

} // namespace

int main()
{
    WavePlan plan{};
    std::string error;
    WavePlanDesc desc = ValidDesc();
    assert(BuildWavePlan(desc, &plan, &error) == PlanStatus::OK);
    assert(error.empty());
    assert(plan.route_slots.size() == 5u);
    assert(plan.assignments.size() == 6u);
    assert((plan.expected_contributors ==
            std::vector<uint32_t>{2u, 0u, 1u, 2u}));

    auto cell = [&plan](uint32_t a, uint32_t b) -> uint64_t {
        return plan.token_row_counts[a * plan.worker_count + b];
    };
    auto assignments = [&plan](uint32_t a, uint32_t b) -> uint64_t {
        return plan.assignment_counts[a * plan.worker_count + b];
    };
    assert(cell(0u, 1u) == 1u && assignments(0u, 1u) == 2u);
    assert(cell(0u, 2u) == 1u && assignments(0u, 2u) == 1u);
    assert(cell(1u, 2u) == 1u && assignments(1u, 2u) == 1u);
    assert(cell(3u, 0u) == 1u && assignments(3u, 0u) == 1u);
    assert(cell(3u, 2u) == 1u && assignments(3u, 2u) == 1u);
    assert((plan.combine_source_offsets ==
            std::vector<uint64_t>{0u, 1u, 2u, 5u, 5u}));
    assert(plan.combine_rows[0].source_rank == 0u);
    assert(plan.combine_rows[0].destination_rank == 3u);
    assert(plan.combine_rows[1].source_rank == 1u);
    assert(plan.combine_rows[1].destination_rank == 0u);
    assert(plan.combine_rows[2].source_rank == 2u);
    assert(plan.combine_rows[2].destination_rank == 0u);
    assert(plan.dispatch_payload_bytes == 4u * 16u * 2u);
    assert(plan.strict_partial_bytes == 5u * 16u * sizeof(float));
    assert(plan.semantic_digest != 0u);

    // Input order is not a semantic field: the same route must compile to the
    // exact same dense route slots and digest.
    WavePlanDesc shuffled = desc;
    std::reverse(shuffled.routes.begin(), shuffled.routes.end());
    std::reverse(shuffled.routes.back().assignments.begin(),
                 shuffled.routes.back().assignments.end());
    WavePlan same{};
    assert(BuildWavePlan(shuffled, &same, &error) == PlanStatus::OK);
    assert(same.semantic_digest == plan.semantic_digest);
    assert(same.combine_source_offsets == plan.combine_source_offsets);

    WavePlanDesc bad = desc;
    bad.routes.pop_back();
    assert(BuildWavePlan(bad, &same, &error) == PlanStatus::INVALID_ROUTE);
    bad = desc;
    bad.routes[0].assignments[1].ordinal = 0u;
    assert(BuildWavePlan(bad, &same, &error) ==
           PlanStatus::DUPLICATE_ORDINAL);
    bad = desc;
    bad.routes[0].assignments[0].weight =
        std::numeric_limits<float>::quiet_NaN();
    assert(BuildWavePlan(bad, &same, &error) ==
           PlanStatus::NONFINITE_WEIGHT);
    bad = desc;
    bad.routes[0].assignments[0].destination_rank = 4u;
    assert(BuildWavePlan(bad, &same, &error) == PlanStatus::INVALID_ROUTE);
    bad = desc;
    bad.generation = 0u;
    assert(BuildWavePlan(bad, &same, &error) ==
           PlanStatus::INVALID_ARGUMENT);
    return 0;
}
