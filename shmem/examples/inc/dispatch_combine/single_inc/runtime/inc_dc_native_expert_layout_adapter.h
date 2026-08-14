#ifndef INC_DC_NATIVE_EXPERT_LAYOUT_ADAPTER_H
#define INC_DC_NATIVE_EXPERT_LAYOUT_ADAPTER_H

#include <cstdint>
#include <vector>

#include "inc_dc_single_inc_combine_plan_compiler.h"

namespace inc::dc::single_stream {

struct NativeExpertRankLayout {
    std::vector<uint32_t> expert_ids;
    std::vector<uint64_t> expert_offsets;
    std::vector<uint64_t> padded_expert_offsets;
    std::vector<uint64_t> tokens_per_expert;
    // Logical expert-major row -> rank-local physical Dispatch row.
    std::vector<uint32_t> dispatch_rows;
    // Logical expert-major row -> rank-local Combine contributor row.
    std::vector<uint32_t> combine_rows;
    // Logical expert-major row -> padded grouped-GEMM row.
    std::vector<uint64_t> padded_rows;
    // Inverse permutation required by Combine.
    std::vector<uint64_t> combine_row_to_padded_row;
    uint64_t padded_row_count = 0u;
};

struct NativeExpertLayout {
    uint32_t expert_alignment = 1u;
    std::vector<NativeExpertRankLayout> ranks;
};

// configured_expert_ids is rank-major and includes zero-token experts. It is
// the framework/model contract, not inferred from this workload's active set.
bool BuildNativeExpertLayout(
    const StreamCompiledGlobalPlan &dispatch,
    const CombineReverseLayout &combine,
    const std::vector<std::vector<uint32_t>> &configured_expert_ids,
    uint32_t expert_alignment, NativeExpertLayout *layout);

} // namespace inc::dc::single_stream

#endif
