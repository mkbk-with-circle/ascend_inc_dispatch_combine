#ifndef INC_COMBINE_PLAN_H
#define INC_COMBINE_PLAN_H

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "inc_dc_combine_upload_transport.h"
#include "inc_dc_combine_packed_transport.h"
#include "inc_dc_combine_liveness_sidecar.h"
#include "inc_dc_combine_lineage_sidecar.h"
#include "inc_dc_handle.h"
#include "inc_dc_combine_symmetric_layout.h"

namespace inc {
namespace dc {

// P3.0：按 (source_rank, source_token) 排序的 contiguous reduce plan
struct IncDcResultTokenDesc {
    int32_t source_rank = -1;
    int32_t source_token = -1;
    int32_t contrib_begin = 0;
    int32_t contrib_count = 0;
};

struct IncDcContributionDesc {
    int32_t contributor_rank = -1;
    uint32_t ingress_slot = 0;
    uint32_t assignment_id = 0;
    uint32_t expert_output_offset = 0;
    float weight = 1.f;  // S0: 从 IncCombineInverseEntry.weight 填入，S2 device kernel 读取
};

struct IncDcCombineReducePlan {
    std::vector<IncDcResultTokenDesc> result_tokens;
    std::vector<IncDcContributionDesc> contributions;
};

struct IncDcCombineDeviceMetaLayout {
    uint64_t upload_plan = 0;
    uint64_t contributions = 0;
    uint64_t result_tokens = 0;
    uint64_t expected_contrib = 0;
    uint64_t arrived_contrib = 0;
    uint64_t contrib_arrived = 0;
    uint64_t contrib_seq = 0;
    uint64_t result_expected_count = 0;
    uint64_t result_arrived_count = 0;
    uint64_t result_ready_seq = 0;
    uint64_t done_bitmap = 0;
    uint64_t total_bytes = 0;
    bool use_seq_credit = false;
    // BW03 optional region (after done_bitmap); offsets relative to recv_meta base.
    IncDcCombineBw03DeviceLayout bw03{};
    // BW05 packed channel (formal; do not alias BW03 shared-tail rings).
    IncDcCombineBw05DeviceLayout bw05{};
    // BW05-H2 live-progress/abort sidecar (outside schema3 ownership; env-gated).
    Bw05H2SidecarLayout h2{};
    bool h2_progress_enable = false;
    // BW05-K1 data-lineage diagnostic sidecar (debug-only; env-gated).
    Bw05K1DiagSidecarLayout k1_diag{};
    bool k1_diag_enable = false;
};

inline bool CombineBw03UploadLanesEnabled()
{
    const char *raw = std::getenv("INC_DC_COMBINE_UPLOAD_LANES");
    return raw != nullptr && raw[0] == '1' && raw[1] == '\0';
}

inline bool CombineBw03ReadyQueueEnabled()
{
    const char *raw = std::getenv("INC_DC_COMBINE_READY_QUEUE");
    return raw != nullptr && raw[0] == '1' && raw[1] == '\0';
}

// CPU：由 combine_inverse 生成 result_tokens + contributions（assignment/weight/ingress_slot）。
IncDcStatus BuildCombineReducePlan(const IncDispatchHandle &handle, IncDcCombineReducePlan *out);

// Paired INC: keep only result tokens for dst_rank; remaps contrib indices.
IncDcStatus FilterCombineReducePlanForDstRank(const IncDcCombineReducePlan &in, int32_t dst_rank,
                                              IncDcCombineReducePlan *out);

void ApplyResultMajorIngressSlots(IncDcCombineReducePlan *plan, uint32_t max_contrib_per_token);

// Paired INC: remaps ingress_slot to destination-local result-major indices so each
// destination rank independently starts at slot 0. assignment_id stays global identity.
void ApplyDestinationLocalIngressSlots(IncDcCombineReducePlan *plan, uint32_t max_contrib_per_token);

// Capacity = max over destinations of (local_result_count * max_contrib_per_token).
uint32_t MaxDestinationLocalIngressSlots(const IncDcCombineReducePlan &plan, uint32_t max_contrib_per_token);

// Global result-major capacity (single-INC): result_tokens * max_contrib_per_token.
uint32_t MaxGlobalResultMajorIngressSlots(const IncDcCombineReducePlan &plan, uint32_t max_contrib_per_token);

// True iff all contribution ingress_slots are unique (required for single-INC shared ingress).
bool IngressSlotsGloballyUnique(const IncDcCombineReducePlan &plan);

IncDcCombineDeviceMetaLayout ComputeCombineDeviceMetaLayout(uint32_t upload_count, uint32_t contrib_count,
                                                            uint32_t result_count, bool use_seq_credit = false,
                                                            bool bw03_upload_lanes = false,
                                                            bool bw03_ready_queue = false,
                                                            bool bw05_packed_channel = false,
                                                            uint32_t hidden_bytes = 0,
                                                            uint32_t bw05_channel_capacity = 0);

} // namespace dc
} // namespace inc

#endif
