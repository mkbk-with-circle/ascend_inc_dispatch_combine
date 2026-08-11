#ifndef INC_DC_NATIVE_COMBINE_WORKSPACE_H
#define INC_DC_NATIVE_COMBINE_WORKSPACE_H

#include <cstdint>
#include <vector>

#include "inc_dc_combine_plan_compiler.h"
#include "inc_dc_resource_policy.h"
#include "inc_dc_combine_runtime_abi.h"

namespace inc::dc::single_stream {

struct NativeCombineInputCopy {
    uint32_t source_row = 0u;
    uint32_t ingress_slot = 0u;
};

struct NativeCombinePreparedWorkspace {
    DynCsrCtrl control{};
    IncDcAivPolicy resources{};
    IncDcCompiledExecutionPlan execution{};
    uint64_t heap_bytes = 0u;
    std::vector<uint8_t> immutable_image;
    std::vector<uint64_t> contributor_rows;
    std::vector<std::vector<NativeCombineInputCopy>> input_copies;
};

// Builds the qualified single-INC, mode-6 streaming layout.  Resource counts
// depend only on live hardware and worker topology; route/shape only size and
// populate metadata/storage.
bool BuildNativeCombinePreparedWorkspace(
    const IncDcCombineLogicalPlanV2 &logical,
    const std::vector<uint64_t> &contributor_rows,
    uint32_t hidden, uint32_t live_aiv,
    NativeCombinePreparedWorkspace *workspace);

} // namespace inc::dc::single_stream

#endif
