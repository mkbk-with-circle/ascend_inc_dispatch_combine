#ifndef INC_DC_SINGLE_INC_COMBINE_PLAN_COMPILER_H
#define INC_DC_SINGLE_INC_COMBINE_PLAN_COMPILER_H

#include <cstdint>
#include <vector>

#include "inc_dc_combine_logical_plan.h"
#include "inc_dc_single_inc_stream_plan_compiler.h"

namespace inc::dc::single_stream {

struct CombineReverseLayout {
    IncDcCombineLogicalPlanV2 logical_plan{};
    // Number of expanded expert-instance rows consumed from every worker.
    std::vector<uint64_t> contributor_rows;
    // For each contributor rank/local row, identifies the rank-local physical
    // Dispatch output row that must be expanded before expert computation.
    // This is the bridge consumed by grouped-GEMM/layout adapters; it is not a
    // workload-dependent transport decision.
    std::vector<std::vector<uint32_t>> contributor_dispatch_rows;
};

// Converts the exact Dispatch physical-row/assignment order into the reverse
// Combine plan.  Every logical expert assignment remains one contribution;
// rank-deduplicated Dispatch rows are expanded in route/assignment order.
bool BuildCombineReverseLayout(
    const StreamCompiledGlobalPlan &dispatch,
    CombineReverseLayout *combine);

} // namespace inc::dc::single_stream

#endif
