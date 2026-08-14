#ifndef INC_FUSION_PLAN_H
#define INC_FUSION_PLAN_H

#include <cstdint>
#include <string>
#include <vector>

#include "inc_fusion_abi.h"

namespace inc::fusion {

struct FusionPlanConfig {
    uint32_t live_aiv = 0u;
    uint32_t live_aic = 0u;
    uint32_t worker_count = 0u;
    uint32_t rank = 0u;
    uint32_t inc_pe = 0u;
    uint32_t hidden = 0u;
    uint32_t intermediate = 0u;
    uint32_t expert_count = 0u;
    uint32_t topk = 0u;
    uint32_t token_count = 0u;
    uint32_t tokens_per_wave = 0u;
    uint32_t slot_count = kFusionMinSlots;
    uint32_t service_ring_size = 4u;
    uint32_t activation_waves = 2u;
    uint32_t element_bytes = 2u;
    // Zero means wait until generation/cancellation. Production arbitrary
    // timing must not fail merely because another cohort was scheduled late.
    uint32_t spin_cap = 0u;
};

struct FusionPlan {
    FusionPlanConfig config{};
    FusionResourcePlan resources{};
    FusionSymmetricLayout symmetric{};
    FusionRemoteServiceLayout remote_service{};
    FusionWorkspaceLayout worker_workspace{};
    FusionWorkspaceLayout inc_workspace{};
    std::vector<uint32_t> expert_owner;
    std::vector<uint32_t> expert_local_index;
    std::vector<uint32_t> local_expert_counts;
    std::vector<FusionWaveDesc> waves;
    uint32_t max_source_dispatch_rows_per_wave = 0u;
    uint32_t max_source_assignments_per_wave = 0u;
    uint32_t max_dispatch_rows_per_wave = 0u;
    uint32_t max_assignments_per_wave = 0u;
    uint32_t transport_tile_bytes = 0u;
    uint32_t compute_tile_rows = 0u;
};

bool BuildFusionPlan(const FusionPlanConfig &config,
                     const uint32_t *expert_owner,
                     const uint32_t *expert_local_index,
                     FusionPlan *plan, std::string *error);

bool ValidateFusionRoute(const FusionPlan &plan,
                         const FusionDispatchRow *rows, uint32_t row_count,
                         const FusionExpertAssignment *assignments,
                         uint32_t assignment_count, std::string *error);

FusionKernelArgs MakeFusionKernelArgs(const FusionPlan &plan,
                                      uint32_t role,
                                      uint64_t generation);

} // namespace inc::fusion

#endif
