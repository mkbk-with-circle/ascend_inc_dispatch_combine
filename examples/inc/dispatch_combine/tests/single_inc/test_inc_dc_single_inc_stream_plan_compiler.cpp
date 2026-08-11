#include "inc_dc_easy_api.h"
#include "inc_dc_single_inc_stream_plan_compiler.h"

#include <cassert>
#include <cstdint>
#include <vector>

using namespace inc::dc::single_stream;

int main()
{
    assert(ResolveStreamPacketRows(16u * 1024u) == 64u);
    assert(ResolveStreamTileRows(2053u, 16u * 1024u) == 192u);
    assert(ResolveStreamTileRows(1031u, 16u * 1024u) == 128u);
    constexpr uint64_t tokens = 4u;
    constexpr uint32_t topk = 4u;
    constexpr uint32_t world = 2u;
    int32_t experts[tokens * topk] = {
        0, 2, 1, 3, 0, 2, 1, 3,
        0, 2, 1, 3, 0, 2, 1, 3};
    float weights[tokens * topk];
    for (uint32_t i = 0u; i < tokens * topk; ++i) {
        weights[i] = static_cast<float>(i + 1u) / 32.0f;
    }
    inc_dc_easy_token_plan_desc_t description{};
    inc_dc_easy_token_plan_desc_init(&description);
    description.tokens = tokens;
    description.topk = topk;
    description.worker_world_size = world;
    description.worker_rank = 0u;
    description.experts_per_worker = 2u;
    description.expert_ids = experts;
    description.weights = weights;
    description.generation = 17u;
    uint64_t bytes = 0u;
    assert(inc_dc_easy_token_plan_query(&description, &bytes) ==
           INC_DC_FW_OK);
    std::vector<uint8_t> plan(bytes);
    inc_dc_easy_token_plan_info_t info{};
    assert(inc_dc_easy_token_plan_build(
               &description, plan.data(), plan.size(), &info) ==
           INC_DC_FW_OK);
    assert(info.logical_assignments == tokens * topk);
    assert(info.global_physical_rows == tokens * world);

    StreamPlanCompileInput input{};
    input.source_rank = 0u;
    input.worker_world_size = world;
    input.hidden_bytes = 16u * 1024u;
    input.tile_rows = 2u;
    input.max_routes_per_packet = 8u;
    input.host_token_plan = plan.data();
    input.host_token_plan_bytes = plan.size();
    StreamCompiledSourcePlan compiled{};
    assert(CompileStreamSourcePlan(input, &compiled));
    assert(compiled.logical_assignments == tokens * topk);
    assert(compiled.physical_rows == tokens * world);
    assert(compiled.assignments.size() == tokens * topk);
    assert(compiled.routes.size() == tokens * world);
    assert(compiled.tasks.size() == 4u); // two tiles x two destinations
    assert(compiled.physical_output_bytes ==
           tokens * world * input.hidden_bytes);
    for (const auto &route : compiled.routes) {
        assert(route.source_rank == 0u);
        assert(route.assignment_count == 2u);
    }

    description.worker_rank = 1u;
    std::vector<uint8_t> plan1(bytes);
    assert(inc_dc_easy_token_plan_build(
               &description, plan1.data(), plan1.size(), &info) ==
           INC_DC_FW_OK);
    input.source_rank = 1u;
    input.host_token_plan = plan1.data();
    StreamCompiledSourcePlan compiled1{};
    assert(CompileStreamSourcePlan(input, &compiled1));
    std::vector<StreamCompiledSourcePlan> sources;
    sources.push_back(compiled1); // deliberately reversed arrival order
    sources.push_back(compiled);
    StreamCompiledGlobalPlan global{};
    assert(MergeStreamSourcePlans(sources, &global));
    assert(global.logical_assignments == tokens * topk * world);
    assert(global.physical_rows == tokens * world * world);
    assert(global.assignments.size() == tokens * topk * world);
    assert(global.routes.size() == tokens * world * world);
    assert(global.tasks.size() == 8u);
    assert(global.destination_output_offsets.size() == world);
    assert(global.destination_output_offsets[0] == 0u);
    assert(global.destination_physical_rows[0] == tokens * world);
    assert(global.destination_physical_rows[1] == tokens * world);
    for (size_t i = 1u; i < global.tasks.size(); ++i) {
        assert(global.tasks[i - 1u].source_tile <=
               global.tasks[i].source_tile);
    }
    StreamWorkspaceBuildInput workspace_input{};
    workspace_input.live_aiv = 48u;
    workspace_input.tile_rows = 2u;
    StreamPreparedWorkspace prepared{};
    assert(BuildStreamPreparedWorkspace(
        global, workspace_input, &prepared));
    assert(prepared.resources.dispatch_inc_aiv == 16u);
    assert(prepared.resources.dispatch_worker_aiv == 8u);
    assert(prepared.descriptor.lane_count == 16u);
    assert(prepared.descriptor.upload_lane_count == 8u);
    assert(prepared.descriptor.direct_task_count == global.tasks.size());
    assert(prepared.descriptor.gather_chunk_count == 0u);
    assert((prepared.descriptor.reserved32 & kStreamFlagDirectDcci) == 0u);
    assert(prepared.descriptor.tx_pingpong == 1u);
    assert(prepared.descriptor.tx_lane_tasks_contiguous == 2u);
    assert(prepared.descriptor.total_bytes >
           prepared.descriptor.output_off + global.physical_output_bytes);
    assert(prepared.tx_lane_task_offsets.size() == 17u);
    assert(prepared.tx_lane_task_indices.size() == global.tasks.size());

    return 0;
}
