#include "inc_dc_single_inc_inline_plan.h"

#include <cassert>
#include <cstdint>

using namespace inc::dc::single_inline;

int main()
{
    SingleInlinePlanDesc desc{};
    desc.worker_count = 4u;
    desc.expert_count = 16u;
    desc.hidden_bytes = 16384u;
    desc.max_topk = 4u;
    desc.tokens_per_wave = 32u;
    desc.batch_tokens = 8u;
    desc.payload_tile_rows = 4u;
    desc.slot_count = 3u;
    desc.dispatch_lane_count = 20u;
    desc.combine_lane_count = 20u;
    desc.session_id = 7u;
    desc.placement_epoch = 11u;

    SingleInlinePlan plan{};
    assert(BuildSingleInlinePlan(desc, &plan) == SingleInlinePlanStatus::OK);
    assert(plan.runtime.batches_per_wave == 4u);
    assert(plan.runtime.payload_tiles_per_wave == 8u);
    assert(plan.results_per_slot == 128u);
    assert(plan.contributions_per_slot == 512u);
    assert(plan.max_token_record_bytes >= 16384u);
    assert(plan.stored_token_record_stride % 64u == 0u);
    assert(plan.runtime.ingress_frame_stride >=
           plan.max_batch_frame_bytes);
    assert(plan.runtime.fanout_record_stride % 64u == 0u);
    assert(plan.runtime.combine_record_stride % 64u == 0u);
    assert(plan.runtime.result_record_stride ==
           plan.runtime.combine_record_stride);
    assert(plan.runtime.total_bytes > plan.runtime.stats_off);
    assert(plan.runtime.total_bytes % kSingleInlineWorkspaceAlignment == 0u);

    // Larger total requests are represented as more waves and do not enlarge
    // the bounded slot workspace.
    SingleInlinePlan same = plan;
    assert(BuildSingleInlinePlan(desc, &same) == SingleInlinePlanStatus::OK);
    assert(same.runtime.total_bytes == plan.runtime.total_bytes);

    SingleInlinePlanDesc bad = desc;
    bad.slot_count = 1u;
    assert(BuildSingleInlinePlan(bad, &same) ==
           SingleInlinePlanStatus::INVALID_ARGUMENT);
    bad = desc;
    bad.max_topk = 17u;
    assert(BuildSingleInlinePlan(bad, &same) ==
           SingleInlinePlanStatus::INVALID_ARGUMENT);
    return 0;
}
