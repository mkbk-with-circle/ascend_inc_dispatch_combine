#include "inc_dc_easy_api.h"
#include "inc_dc_native_combine_workspace.h"
#include "inc_dc_single_inc_combine_plan_compiler.h"

#include <cassert>
#include <cmath>
#include <vector>

using namespace inc::dc::single_stream;

int main()
{
    constexpr uint32_t world = 2u;
    constexpr uint32_t tokens = 3u;
    constexpr uint32_t topk = 4u;
    const int32_t experts[tokens * topk] = {
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    const float weights[tokens * topk] = {
        .1f, .2f, .3f, .4f, .5f, .6f,
        .7f, .8f, .9f, 1.f, 1.1f, 1.2f};
    std::vector<StreamCompiledSourcePlan> sources;
    for (uint32_t rank = 0u; rank < world; ++rank) {
        inc_dc_easy_token_plan_desc_t description{};
        inc_dc_easy_token_plan_desc_init(&description);
        description.tokens = tokens;
        description.topk = topk;
        description.worker_world_size = world;
        description.worker_rank = rank;
        description.experts_per_worker = 2u;
        description.expert_ids = experts;
        description.weights = weights;
        description.generation = 9u;
        uint64_t bytes = 0u;
        assert(inc_dc_easy_token_plan_query(&description, &bytes) ==
               INC_DC_FW_OK);
        std::vector<uint8_t> wire(bytes);
        inc_dc_easy_token_plan_info_t info{};
        assert(inc_dc_easy_token_plan_build(
                   &description, wire.data(), wire.size(), &info) ==
               INC_DC_FW_OK);
        StreamPlanCompileInput input{};
        input.source_rank = rank;
        input.worker_world_size = world;
        input.hidden_bytes = 16384u;
        input.tile_rows = 16u;
        input.max_routes_per_packet = 64u;
        input.host_token_plan = wire.data();
        input.host_token_plan_bytes = wire.size();
        StreamCompiledSourcePlan source{};
        assert(CompileStreamSourcePlan(input, &source));
        sources.push_back(std::move(source));
    }
    StreamCompiledGlobalPlan dispatch{};
    assert(MergeStreamSourcePlans(sources, &dispatch));
    CombineReverseLayout reverse{};
    assert(BuildCombineReverseLayout(dispatch, &reverse));
    assert(reverse.logical_plan.result_count == world * tokens);
    assert(reverse.logical_plan.contribution_count ==
           world * tokens * topk);
    // Each destination gets two expert instances per token from each source.
    assert(reverse.contributor_rows[0] == world * tokens * 2u);
    assert(reverse.contributor_rows[1] == world * tokens * 2u);
    assert(reverse.contributor_dispatch_rows.size() == world);
    for (uint32_t rank = 0u; rank < world; ++rank) {
        assert(reverse.contributor_dispatch_rows[rank].size() ==
               reverse.contributor_rows[rank]);
        for (uint32_t physical : reverse.contributor_dispatch_rows[rank])
            assert(physical < dispatch.destination_physical_rows[rank]);
    }
    for (uint32_t result = 0u;
         result < reverse.logical_plan.result_count; ++result) {
        const auto &row = reverse.logical_plan.results[result];
        assert(row.dst_rank == result / tokens);
        assert(row.dst_local_row == result % tokens);
        assert(row.contribution_count == topk);
        for (uint32_t ordinal = 0u; ordinal < topk; ++ordinal) {
            const auto &contribution = reverse.logical_plan.contributions[
                row.contribution_begin + ordinal];
            assert(contribution.result_id == result);
            assert(contribution.ordinal == ordinal);
            assert(contribution.contributor_rank == ordinal / 2u);
            assert(std::fabs(
                       contribution.weight -
                       weights[(result % tokens) * topk + ordinal]) < 1e-6f);
        }
    }
    NativeCombinePreparedWorkspace prepared{};
    assert(BuildNativeCombinePreparedWorkspace(
        reverse.logical_plan, reverse.contributor_rows, 8192u, 48u,
        &prepared));
    assert(prepared.resources.combine_inc_aiv == 32u);
    assert(prepared.resources.combine_worker_aiv == 24u);
    assert(prepared.control.owner_count == 32u);
    assert(prepared.control.inc_pe == world);
    assert(prepared.control.producer_lane_count == 24u);
    assert(prepared.control.group_count ==
           prepared.control.owner_count * world);
    assert(prepared.control.ready_mode == 6u);
    assert((prepared.control.optimization_flags &
            inc::dc::kDynCsrOptRemoteResultTx) != 0u);
    assert(prepared.immutable_image.size() == prepared.heap_bytes);
    assert(prepared.heap_bytes > prepared.control.output_off);
    assert(prepared.execution.topology.worker_count == world);
    assert(prepared.execution.topology.owner_count ==
           prepared.control.owner_count);
    assert(prepared.execution.topology.inc_pe == world);
    for (const auto &compiled : prepared.execution.schedule) {
        assert(compiled.owner_index < prepared.control.owner_count);
        const auto &logical = reverse.logical_plan.contributions[
            compiled.logical_contribution_index];
        uint32_t channel = UINT32_MAX;
        assert(inc::dc::LookupIngressChannel(
            prepared.execution.topology, logical.contributor_rank,
            &channel));
        assert(channel == compiled.ingress_channel);
    }

    // The specialized topology remains fail-closed: the unique INC cannot
    // overlap a worker, ingress resources remain one-per-worker, and callers
    // cannot mutate a descriptor without updating its digest.
    {
        auto invalid = prepared.execution.topology;
        invalid.inc_pe = invalid.worker_pe_ids[0];
        invalid.topology_digest = inc::dc::ComputeTopologyDigest(invalid);
        inc::dc::IncDcTopologyValidateReport report{};
        assert(inc::dc::ValidateTopologyDescriptor(invalid, &report) !=
               inc::dc::IncDcStatus::OK);
    }
    {
        auto invalid = prepared.execution.topology;
        invalid.worker_ingress_channels[1] =
            invalid.worker_ingress_channels[0];
        invalid.topology_digest = inc::dc::ComputeTopologyDigest(invalid);
        inc::dc::IncDcTopologyValidateReport report{};
        assert(inc::dc::ValidateTopologyDescriptor(invalid, &report) !=
               inc::dc::IncDcStatus::OK);
    }
    {
        auto invalid = prepared.execution.topology;
        invalid.topology_digest ^= 1u;
        inc::dc::IncDcTopologyValidateReport report{};
        assert(inc::dc::ValidateTopologyDescriptor(invalid, &report) !=
               inc::dc::IncDcStatus::OK);
    }
    for (uint32_t rank = 0u; rank < world; ++rank) {
        const auto &copies = prepared.input_copies[rank];
        assert(copies.size() == reverse.contributor_rows[rank]);
        for (uint32_t row = 0u; row < copies.size(); ++row) {
            assert(copies[row].source_row == row);
            if (row != 0u) {
                assert(copies[row].ingress_slot ==
                       copies[row - 1u].ingress_slot + 1u);
            }
        }
    }
    return 0;
}
