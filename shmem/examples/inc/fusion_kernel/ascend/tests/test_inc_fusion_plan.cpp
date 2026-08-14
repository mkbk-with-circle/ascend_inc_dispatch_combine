#include "inc_fusion_plan.h"
#include "inc_fusion_route.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace inc::fusion;

namespace {

FusionPlan Build(uint32_t workers, uint32_t tokens, uint32_t wave_tokens,
                 uint32_t experts, uint32_t topk,
                 uint32_t service_ring_size = 4u)
{
    FusionPlanConfig config{};
    config.live_aiv = 48u;
    config.live_aic = 24u;
    config.worker_count = workers;
    config.rank = 0u;
    config.inc_pe = workers;
    config.hidden = 2048u;
    config.intermediate = 8192u;
    config.expert_count = experts;
    config.topk = topk;
    config.token_count = tokens;
    config.tokens_per_wave = wave_tokens;
    config.slot_count = 3u;
    config.service_ring_size = service_ring_size;
    std::vector<uint32_t> owner(experts);
    std::vector<uint32_t> local(experts);
    std::vector<uint32_t> next(workers, 0u);
    for (uint32_t e = 0; e < experts; ++e) {
        owner[e] = (e * 3u + 1u) % workers;
        local[e] = next[owner[e]]++;
    }
    FusionPlan plan{};
    std::string error;
    assert(BuildFusionPlan(config, owner.data(), local.data(), &plan, &error));
    assert(error.empty());
    return plan;
}

void TestResourcesAndWaves()
{
    FusionPlan plan = Build(4u, 4097u, 512u, 64u, 8u);
    assert(plan.resources.inc_dispatch_aiv == 32u);
    assert(plan.resources.inc_combine_aiv == 16u);
    assert(plan.resources.worker_dispatch_aiv == 8u);
    assert(plan.resources.worker_combine_aiv == 16u);
    assert(plan.resources.worker_compute_aiv == 24u);
    assert(plan.waves.size() == 9u);
    assert(plan.waves.back().token_count == 1u);
    assert(plan.waves[3].slot == 0u);
    assert(plan.worker_workspace.total_bytes > 0u);
    assert(plan.inc_workspace.total_bytes > 0u);
    assert(plan.symmetric.total_bytes > 0u);
    assert(plan.remote_service.ring_size == 4u);
    assert(plan.remote_service.control_off <
           plan.remote_service.descriptors_off);
    assert(plan.remote_service.request_off <
           plan.remote_service.waves_off);
    assert(plan.remote_service.waves_off <
           plan.remote_service.active_token_counts_off);
    assert(plan.remote_service.request_stride >=
           plan.remote_service.request_bytes);
    assert(plan.remote_service.request_bytes ==
           sizeof(FusionRemoteRequestHeader) +
               plan.waves.size() * sizeof(FusionWaveDesc) +
               plan.config.worker_count * sizeof(uint32_t));
    assert(plan.remote_service.request_off +
               static_cast<uint64_t>(plan.remote_service.ring_size) *
                   plan.remote_service.request_stride <=
           plan.remote_service.worker_pes_off);
    assert(plan.remote_service.worker_ready_off <
           plan.remote_service.lane_progress_off);
    assert(plan.remote_service.lane_progress_off <
           plan.symmetric.total_bytes);
    assert(plan.max_source_dispatch_rows_per_wave == 2048u);
    assert(plan.max_source_assignments_per_wave == 4096u);
    assert(plan.max_dispatch_rows_per_wave == 2048u);
    assert(plan.max_assignments_per_wave == 16384u);
    const uint64_t symmetric_small = plan.symmetric.total_bytes;
    FusionPlan larger = Build(4u, 8193u, 512u, 64u, 8u);
    assert(larger.symmetric.total_bytes > symmetric_small);
    FusionPlan shallower_ring = Build(4u, 4097u, 512u, 64u, 8u, 2u);
    assert(shallower_ring.remote_service.ring_size == 2u);
    assert(shallower_ring.symmetric.total_bytes < symmetric_small);
    FusionKernelArgs args = MakeFusionKernelArgs(plan, kFusionWorker, 17u);
    assert(args.operation_generation == 17u);
    assert(args.flags & kFusionConcurrentDispatchCombine);
    assert(args.remote_service.ring_size == 4u);
}

void TestAdversarialReceiveCapacity()
{
    FusionPlan w2 = Build(2u, 64u, 64u, 8u, 1u);
    assert(w2.max_source_dispatch_rows_per_wave == 64u);
    assert(w2.max_source_assignments_per_wave == 64u);
    assert(w2.max_dispatch_rows_per_wave == 128u);
    assert(w2.max_assignments_per_wave == 128u);
    assert(w2.worker_workspace.dispatch_slot_bytes ==
           static_cast<uint64_t>(128u) * 2048u * 2u);
    assert(w2.worker_workspace.assignment_slot_bytes ==
           static_cast<uint64_t>(128u) * 2048u * 2u);

    FusionPlan w4 = Build(4u, 64u, 64u, 8u, 2u);
    assert(w4.max_source_dispatch_rows_per_wave == 128u);
    assert(w4.max_source_assignments_per_wave == 128u);
    assert(w4.max_dispatch_rows_per_wave == 256u);
    assert(w4.max_assignments_per_wave == 512u);
}

void TestNonUniformPlacementAndRoute()
{
    FusionPlan plan = Build(3u, 16u, 8u, 10u, 4u);
    FusionDispatchRow row{};
    row.source_rank = 0u;
    row.source_token = 3u;
    row.destination_rank = plan.expert_owner[0u];
    row.assignment_begin = 0u;
    row.assignment_count = 1u;
    row.wave = 0u;
    FusionExpertAssignment assignment{};
    assignment.dispatch_row = 0u;
    assignment.expert_id = 0u;
    assignment.local_expert = plan.expert_local_index[0u];
    assignment.route_ordinal = 2u;
    assignment.destination_token = 3u;
    assignment.wave = 0u;
    std::string error;
    assert(ValidateFusionRoute(plan, &row, 1u, &assignment, 1u, &error));
    assignment.local_expert++;
    assert(!ValidateFusionRoute(plan, &row, 1u, &assignment, 1u, &error));
}

void TestInvalidConfigurations()
{
    FusionPlanConfig config{};
    config.live_aiv = 48u;
    config.live_aic = 24u;
    config.worker_count = 4u;
    config.rank = 0u;
    config.inc_pe = 4u;
    config.hidden = 2048u;
    config.intermediate = 8192u;
    config.expert_count = 4u;
    config.topk = 2u;
    config.token_count = 128u;
    config.tokens_per_wave = 64u;
    config.slot_count = 2u;
    uint32_t owner[4] = {0u, 1u, 2u, 3u};
    uint32_t local[4] = {0u, 0u, 0u, 0u};
    FusionPlan plan{};
    std::string error;
    assert(!BuildFusionPlan(config, owner, local, &plan, &error));
    config.slot_count = 3u;
    owner[3] = 4u;
    assert(!BuildFusionPlan(config, owner, local, &plan, &error));
}

void TestLiveAivBound()
{
    FusionPlanConfig config{};
    config.live_aiv = kFusionMaxAiv + 1u;
    config.live_aic = 24u;
    config.worker_count = 2u;
    config.rank = 0u;
    config.inc_pe = 2u;
    config.hidden = 256u;
    config.intermediate = 512u;
    config.expert_count = 4u;
    config.topk = 2u;
    config.token_count = 32u;
    config.tokens_per_wave = 8u;
    config.slot_count = 3u;
    const uint32_t owner[4] = {0u, 0u, 1u, 1u};
    const uint32_t local[4] = {0u, 1u, 0u, 1u};
    FusionPlan plan{};
    std::string error;
    assert(!BuildFusionPlan(config, owner, local, &plan, &error));
    assert(!error.empty());
}

void TestDynamicRouteCompilation()
{
    FusionPlan plan = Build(4u, 5u, 3u, 8u, 2u);
    const size_t routes = static_cast<size_t>(4u) * 5u * 2u;
    std::vector<uint32_t> expert_ids(routes);
    std::vector<float> weights(routes);
    for (size_t i = 0u; i < routes; i += 2u) {
        // In Build(), experts 0 and 4 are both owned by worker 1. Each token
        // therefore has two logical assignments but one physical hidden row.
        expert_ids[i] = 0u;
        expert_ids[i + 1u] = 4u;
        weights[i] = 0.25f;
        weights[i + 1u] = 0.75f;
    }
    FusionRouteBundle bundle{};
    std::string error;
    assert(CompileFusionRoute(plan, expert_ids.data(), weights.data(),
                              &bundle, &error));
    assert(error.empty());
    assert(bundle.ranks.size() == 4u);
    for (uint32_t source = 0u; source < 4u; ++source) {
        const FusionRankRoute &rank = bundle.ranks[source];
        assert(rank.dispatch_rows.size() == 5u);
        assert(rank.assignments.size() == 10u);
        assert(rank.waves.size() == 2u);
        assert(rank.waves[0].dispatch_row_count == 3u);
        assert(rank.waves[0].assignment_count == 6u);
        assert(ValidateFusionRoute(plan, rank.dispatch_rows.data(),
                                   static_cast<uint32_t>(rank.dispatch_rows.size()),
                                   rank.assignments.data(),
                                   static_cast<uint32_t>(rank.assignments.size()),
                                   &error));
    }
    const uint32_t local0 = plan.expert_local_index[0u];
    const uint32_t local4 = plan.expert_local_index[4u];
    const auto &groups = bundle.group_lists[1u][0u];
    assert(groups[local0] == 12u);
    assert(groups[local4] == 24u);
}

void TestVariableActiveTokenCompilation()
{
    FusionPlan plan = Build(4u, 5u, 3u, 8u, 2u);
    const size_t routes = static_cast<size_t>(4u) * 5u * 2u;
    std::vector<uint32_t> expert_ids(routes);
    std::vector<float> weights(routes, 0.5f);
    for (size_t i = 0u; i < routes; ++i)
        expert_ids[i] = static_cast<uint32_t>(i % 8u);
    const uint32_t active[4] = {5u, 3u, 0u, 4u};
    FusionRouteBundle bundle{};
    std::string error;
    assert(CompileFusionRouteActive(plan, expert_ids.data(), weights.data(),
                                    active, &bundle, &error));
    assert(error.empty());
    assert(bundle.ranks[0].assignments.size() == 10u);
    assert(bundle.ranks[1].assignments.size() == 6u);
    assert(bundle.ranks[2].assignments.empty());
    assert(bundle.ranks[2].dispatch_rows.empty());
    assert(bundle.ranks[3].assignments.size() == 8u);
    for (const FusionRankRoute &rank : bundle.ranks) {
        assert(rank.waves.size() == 2u);
        assert(rank.waves[0].token_count == 3u);
        assert(rank.waves[1].token_count == 2u);
    }
}

} // namespace

int main()
{
    TestResourcesAndWaves();
    TestAdversarialReceiveCapacity();
    TestNonUniformPlacementAndRoute();
    TestInvalidConfigurations();
    TestLiveAivBound();
    TestDynamicRouteCompilation();
    TestVariableActiveTokenCompilation();
    std::cout << "inc fusion plan tests passed\n";
    return 0;
}
