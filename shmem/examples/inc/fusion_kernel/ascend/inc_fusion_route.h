#ifndef INC_FUSION_ROUTE_H
#define INC_FUSION_ROUTE_H

#include <cstdint>
#include <string>
#include <vector>

#include "inc_fusion_plan.h"

namespace inc::fusion {

struct FusionRankRoute {
    std::vector<FusionDispatchRow> dispatch_rows;
    std::vector<FusionExpertAssignment> assignments;
    std::vector<FusionWaveDesc> waves;
};

struct FusionRouteBundle {
    std::vector<FusionRankRoute> ranks;
    // [worker][wave][local expert] cumulative grouped-GEMM row boundaries.
    std::vector<std::vector<std::vector<int64_t>>> group_lists;
};

// Compile one invocation from the framework-standard dense top-k arrays.
// expert_ids and weights are [worker_count, token_count, topk]. The compiler
// keeps one hidden row per (source token, destination rank), while preserving
// every expert instance as an independent Combine contribution.
bool CompileFusionRoute(const FusionPlan &plan,
                        const uint32_t *expert_ids,
                        const float *weights,
                        FusionRouteBundle *bundle,
                        std::string *error);

// Variable-length reference used by dynamic-batch qualification. Arrays keep
// the fixed-capacity [worker, plan.token_count, topk] stride, but only
// active_token_counts[worker] rows are interpreted. Every rank receives the
// same wave envelope derived from max(active_token_counts), while its source
// rows stop at its own active length.
bool CompileFusionRouteActive(const FusionPlan &plan,
                              const uint32_t *expert_ids,
                              const float *weights,
                              const uint32_t *active_token_counts,
                              FusionRouteBundle *bundle,
                              std::string *error);

} // namespace inc::fusion

#endif
