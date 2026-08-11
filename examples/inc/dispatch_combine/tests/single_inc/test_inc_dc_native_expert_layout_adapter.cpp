#include "inc_dc_easy_api.h"
#include "inc_dc_native_expert_layout_adapter.h"

#include <cassert>
#include <vector>

using namespace inc::dc::single_stream;

int main()
{
    constexpr uint32_t world = 2u;
    constexpr uint32_t tokens = 3u;
    constexpr uint32_t topk = 4u;
    const int32_t experts[tokens * topk] = {
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<StreamCompiledSourcePlan> sources;
    for (uint32_t rank = 0u; rank < world; ++rank) {
        inc_dc_easy_token_plan_desc_t desc{};
        inc_dc_easy_token_plan_desc_init(&desc);
        desc.tokens = tokens;
        desc.topk = topk;
        desc.worker_world_size = world;
        desc.worker_rank = rank;
        desc.experts_per_worker = 2u;
        desc.expert_ids = experts;
        desc.generation = 1u;
        uint64_t bytes = 0u;
        assert(inc_dc_easy_token_plan_query(&desc, &bytes) == INC_DC_FW_OK);
        std::vector<uint8_t> wire(bytes);
        inc_dc_easy_token_plan_info_t info{};
        assert(inc_dc_easy_token_plan_build(
                   &desc, wire.data(), wire.size(), &info) == INC_DC_FW_OK);
        StreamPlanCompileInput input{};
        input.source_rank = rank;
        input.worker_world_size = world;
        input.hidden_bytes = 16384u;
        input.tile_rows = 4u;
        input.max_routes_per_packet = 64u;
        input.host_token_plan = wire.data();
        input.host_token_plan_bytes = wire.size();
        StreamCompiledSourcePlan source{};
        assert(CompileStreamSourcePlan(input, &source));
        sources.push_back(std::move(source));
    }
    StreamCompiledGlobalPlan dispatch{};
    CombineReverseLayout reverse{};
    assert(MergeStreamSourcePlans(sources, &dispatch));
    assert(BuildCombineReverseLayout(dispatch, &reverse));
    const std::vector<std::vector<uint32_t>> configured = {
        {0u, 1u, 99u}, {2u, 3u, 100u}};
    NativeExpertLayout layout{};
    assert(BuildNativeExpertLayout(
        dispatch, reverse, configured, 8u, &layout));
    assert(layout.ranks.size() == world);
    for (uint32_t rank = 0u; rank < world; ++rank) {
        const auto &r = layout.ranks[rank];
        assert(r.expert_offsets.back() == reverse.contributor_rows[rank]);
        assert(r.tokens_per_expert.back() == 0u); // configured zero-token expert
        assert(r.padded_row_count % 8u == 0u);
        assert(r.dispatch_rows.size() == reverse.contributor_rows[rank]);
        assert(r.combine_rows.size() == reverse.contributor_rows[rank]);
        for (uint64_t row : r.combine_row_to_padded_row)
            assert(row < r.padded_row_count);
    }
    NativeExpertLayout invalid{};
    assert(!BuildNativeExpertLayout(
        dispatch, reverse, configured, 3u, &invalid));
    return 0;
}
