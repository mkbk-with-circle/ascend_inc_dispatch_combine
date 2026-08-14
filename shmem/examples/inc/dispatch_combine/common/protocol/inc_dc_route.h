#ifndef INC_DC_ROUTE_H
#define INC_DC_ROUTE_H

#include <cstdint>

#include "inc_dc_group.h"
#include "inc_dc_types.h"

namespace inc {
namespace dc {

struct IncDcTensorDesc {
    void *data = nullptr;
    uint64_t num_tokens = 0;
    uint32_t hidden_size = 0;
    IncDcDType dtype = IncDcDType::FP16;
    uint64_t stride_token_bytes = 0;
};

struct IncDcRouteSpec {
    IncDcRouteFormat format = IncDcRouteFormat::TOPK_DENSE;
    const int32_t *topk_expert_ids = nullptr; // [T,K], -1 invalid
    const float *topk_weights = nullptr;      // optional [T,K]
    const int32_t *csr_offsets = nullptr;   // optional [T+1]
    const int32_t *csr_expert_ids = nullptr;
    uint32_t topk = 0;
    uint64_t route_epoch = 0;
    uint64_t num_tokens = 0; // local token count for this rank
};

IncDcStatus ValidateRouteSpec(const IncDcGroupConfig *group, const IncDcRouteSpec *route, IncDcErrorInfo *err = nullptr);

// 单 token 的有效 assignment 数（展开 CSR/topk）
uint32_t CountTokenAssignments(const IncDcGroupConfig *group, const IncDcRouteSpec *route, uint32_t token_idx);

// destination rank bitmap（至少 32bit，按 group_size 使用低位）
uint32_t TokenDestinationRankMask(const IncDcGroupConfig *group, const IncDcRouteSpec *route, uint32_t token_idx);

} // namespace dc
} // namespace inc

#endif
