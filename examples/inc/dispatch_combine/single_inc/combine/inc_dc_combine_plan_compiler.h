#ifndef INC_DC_COMBINE_PLAN_COMPILER_H
#define INC_DC_COMBINE_PLAN_COMPILER_H

#include <cstdint>
#include <string>
#include <vector>

#include "inc_dc_combine_logical_plan.h"
#include "inc_dc_combine_topology.h"
#include "inc_dc_types.h"

namespace inc {
namespace dc {

struct IncDcCompiledContribution {
    uint32_t logical_contribution_index = 0;
    uint32_t owner_index = 0;     // == result home_owner
    uint32_t ingress_channel = 0; // from topology edge map
    uint32_t ingress_slot = 0;
    uint64_t payload_offset = 0; // uint64; no silent truncation
    uint32_t buffer_id = 0;      // currently 0 = default ingress buffer
    uint32_t element_bytes = 0;
};

struct IncDcCompiledExecutionPlan {
    IncDcTopologyDescriptor topology{};
    std::vector<IncDcCompiledContribution> schedule;
    // Per-result unique owner in the single INC's reduce cohort.
    std::vector<uint32_t> result_home_owner;
    // CSR over the single INC's owners.
    std::vector<uint32_t> owner_worklist_offsets; // size owners_total+1
    std::vector<uint32_t> owner_worklist_entries; // indices into schedule
    // Device CSR views for results
    std::vector<uint32_t> result_offsets; // size R+1
    std::vector<uint32_t> contribution_entry_indices; // size C -> schedule idx
    std::vector<uint32_t> owner_result_offsets;
    std::vector<uint32_t> owner_result_ids;
    uint32_t hidden = 0;
    uint32_t element_bytes = 0;
    uint64_t row_bytes = 0;
    uint64_t semantic_digest = 0;
    uint64_t topology_digest = 0;
    uint64_t execution_digest = 0;
};

struct IncDcPlanCompileReport {
    bool ok = false;
    std::string first_error;
};

// Compile LogicalPlan → single-INC execution plan.
// element_bytes = sizeof(input element); row_bytes = checked_mul(hidden, element_bytes).
// Fail-closed when any contributor cannot reach the single INC.
IncDcStatus CompileLogicalPlanToExecution(
    const IncDcCombineLogicalPlanV2 &logical,
    const IncDcTopologyDescriptor &topology, uint32_t hidden,
    uint32_t element_bytes, IncDcCompiledExecutionPlan *out,
    IncDcPlanCompileReport *report);

uint64_t ComputeExecutionDigest(const IncDcCompiledExecutionPlan &plan);

// Full hard validator (requires logical for contribution identity checks).
IncDcStatus ValidateCompiledExecutionPlan(
    const IncDcCombineLogicalPlanV2 &logical,
    const IncDcCompiledExecutionPlan &plan, IncDcPlanCompileReport *report);

} // namespace dc
} // namespace inc

#endif
