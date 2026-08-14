#ifndef INC_DC_LAYOUT_H
#define INC_DC_LAYOUT_H

#include <cstdint>
#include <vector>

#include "inc_dc_group.h"
#include "inc_dc_route.h"
#include "inc_dc_types.h"

namespace inc {
namespace dc {

struct IncDispatchLayout {
    uint32_t layout_version = kLayoutVersion;
    uint32_t group_size = 0;
    uint64_t route_epoch = 0;
    uint64_t op_seq = 0;
    IncDcDType dtype = IncDcDType::FP16;
    uint32_t hidden_size = 0;

    std::vector<uint32_t> num_tokens_per_rank;   // 每 rank 本地 token 数（send 侧）
    std::vector<uint32_t> num_recv_per_rank;     // 每 rank 收到的 assignment 数（recv 侧）
    std::vector<uint32_t> num_tokens_per_expert;
    std::vector<uint8_t> is_token_in_rank; // flattened [T, group_size]

    std::vector<int32_t> send_token_ids;
    std::vector<uint32_t> send_rank_offsets; // group_size + 1

    std::vector<int32_t> recv_src_rank;
    std::vector<int32_t> recv_src_token;
    std::vector<int32_t> recv_expert_ids;
    std::vector<uint32_t> recv_expert_offsets;
    std::vector<float> recv_assignment_weights;

    std::vector<uint32_t> local_expert_token_offsets;

    // 流量守恒
    uint64_t logical_assignments = 0;
    uint64_t network_token_copies = 0;
    uint64_t network_payload_bytes = 0;
    uint64_t metadata_bytes = 0;
    uint64_t local_bytes = 0;
    uint64_t heap_bytes_required = 0;

    int retain_count = 0;
};

IncDcStatus IncDispatchLayoutRetain(IncDispatchLayout *layout);
IncDcStatus IncDispatchLayoutRelease(IncDispatchLayout **layout);

} // namespace dc
} // namespace inc

#endif
