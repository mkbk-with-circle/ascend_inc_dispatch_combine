#ifndef INC_DC_COMBINE_LOGICAL_PLAN_H
#define INC_DC_COMBINE_LOGICAL_PLAN_H

#include <cstdint>
#include <string>
#include <vector>

#include "inc_dc_types.h"

namespace inc {
namespace dc {

constexpr uint32_t kIncDcCombineLogicalPlanAbiV2 = 2u;

struct IncDcLogicalContributionV2 {
    uint64_t contribution_uid = 0;
    uint32_t result_id = 0;
    uint32_t ordinal = 0;
    uint32_t contributor_rank = 0;
    uint32_t contributor_local_row = 0;
    float weight = 1.f;
};

struct IncDcLogicalResultV2 {
    uint32_t dst_rank = 0;
    uint32_t dst_local_row = 0;
    uint32_t contribution_begin = 0;
    uint32_t contribution_count = 0; // result_expected_count
};

struct IncDcCombineLogicalPlanV2 {
    uint32_t abi_version = kIncDcCombineLogicalPlanAbiV2;
    uint32_t worker_world_size = 0;
    uint32_t result_count = 0;
    uint32_t contribution_count = 0;
    uint32_t declared_max_topk = 0;
    bool uniform_topk_valid = false;
    uint32_t uniform_topk = 0;
    std::vector<IncDcLogicalResultV2> results;
    std::vector<IncDcLogicalContributionV2> contributions;
    uint64_t semantic_digest = 0;
};

struct IncDcLogicalPlanValidateReport {
    bool ok = false;
    std::string first_error;
    uint32_t expected_count_min = 0;
    uint32_t expected_count_max = 0;
    std::vector<uint32_t> contributions_per_worker;
    std::vector<uint32_t> zero_contribution_workers;
};

// Build a synthetic LogicalPlan for W x K matrices (host tests / compilers).
// mode:
//   0 = round-robin contributors (may concentrate on few ranks when K>W)
//   1 = force multi-ordinal on rank0 + zero on rank W-1 when W>=2
//   2 = ragged expected_count (result i gets 1+(i%K) capped by declared_max)
IncDcStatus BuildSyntheticLogicalPlanV2(uint32_t worker_world_size,
                                        uint32_t declared_max_topk,
                                        uint32_t result_count, uint32_t mode,
                                        IncDcCombineLogicalPlanV2 *out);

uint64_t ComputeLogicalPlanSemanticDigest(const IncDcCombineLogicalPlanV2 &plan);

IncDcStatus ValidateLogicalPlanV2(const IncDcCombineLogicalPlanV2 &plan,
                                  IncDcLogicalPlanValidateReport *report);

uint64_t LogicalUsefulBytes(const IncDcCombineLogicalPlanV2 &plan,
                            uint32_t hidden, uint32_t input_elem_bytes,
                            uint32_t output_elem_bytes);

} // namespace dc
} // namespace inc

#endif
