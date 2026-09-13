#ifndef INC_DC_SINGLE_INC_INLINE_PLAN_H
#define INC_DC_SINGLE_INC_INLINE_PLAN_H

#include <cstdint>

#include "inc_dc_single_inc_inline_abi.h"

namespace inc::dc::single_inline {

enum class SingleInlinePlanStatus : uint32_t {
    OK = 0u,
    INVALID_ARGUMENT,
    CAPACITY_EXCEEDED,
};

struct SingleInlinePlanDesc {
    uint32_t worker_count = 0u;
    uint32_t expert_count = 0u;
    uint32_t hidden_bytes = 0u;
    uint32_t max_topk = 0u;
    uint32_t tokens_per_wave = 0u;
    uint32_t batch_tokens = 0u;
    uint32_t payload_tile_rows = 0u;
    uint32_t slot_count = 0u;
    uint32_t dispatch_lane_count = 0u;
    uint32_t combine_lane_count = 0u;
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
};

struct SingleInlinePlan {
    SingleInlineRuntimeDesc runtime{};
    uint64_t max_token_record_bytes = 0u;
    uint64_t stored_token_record_stride = 0u;
    uint64_t max_batch_frame_bytes = 0u;
    uint64_t results_per_slot = 0u;
    uint64_t contributions_per_slot = 0u;
    uint64_t fanout_tokens_per_worker_slot = 0u;
    uint64_t fanout_assignments_per_worker_slot = 0u;
};

SingleInlinePlanStatus BuildSingleInlinePlan(
    const SingleInlinePlanDesc &desc, SingleInlinePlan *plan);
const char *SingleInlinePlanStatusString(SingleInlinePlanStatus status);

} // namespace inc::dc::single_inline

#endif
