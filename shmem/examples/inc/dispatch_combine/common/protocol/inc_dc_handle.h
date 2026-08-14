#ifndef INC_DC_HANDLE_H
#define INC_DC_HANDLE_H

#include <cstdint>
#include <vector>

#include "inc_dc_group.h"
#include "inc_dc_layout.h"
#include "inc_dc_types.h"

namespace inc {
namespace dc {

// 稳定 token identity：(task_id, source_rank, source_token_id, route_epoch)
struct IncDcTokenId {
    uint64_t task_id = 0;
    int32_t source_rank = -1;
    int32_t source_token_id = -1;
    uint64_t route_epoch = 0;
};

struct IncDcAssignmentMeta {
    IncDcTokenId token_id{};
    int32_t destination_rank = -1;
    int32_t expert_id = -1;
    int32_t route_slot = 0;
    uint32_t recv_expert_offset = 0;
    float weight = 1.f;
    bool is_local = false;
};

struct IncCombineInverseEntry {
    IncDcTokenId token_id{};
    int32_t source_rank = -1;
    int32_t source_token_id = -1;
    int32_t assignment_id = -1;
    int32_t expert_id = -1;
    int32_t contributor_rank = -1;
    uint32_t expert_output_offset = 0;
    float weight = 1.f;
    uint32_t expected_contribution_seq = 0;
};

struct IncDispatchHandle {
    uint32_t handle_version = kHandleVersion;
    uint32_t abi_version = kAbiVersion;
    uint32_t group_id = 0;
    uint32_t group_size = 0;
    uint64_t route_epoch = 0;
    uint64_t op_seq = 0;
    IncDcDType dtype = IncDcDType::FP16;
    uint32_t hidden_size = 0;
    uint32_t layout_version = kLayoutVersion;
    uint32_t packet_version = kPacketVersion;

    std::vector<IncDcAssignmentMeta> assignments;
    std::vector<IncCombineInverseEntry> combine_inverse;
    std::vector<uint32_t> expected_contrib_per_token; // per source token on owning rank

    int retain_count = 0;
    bool stale = false;
};

struct IncDispatchResult {
    void *recv_hidden = nullptr;
    void *recv_topk_expert_ids = nullptr;
    void *recv_topk_weights = nullptr;
    const uint32_t *num_recv_tokens_per_expert = nullptr;
    uint64_t num_recv_assignments = 0;
};

struct IncCombineResult {
    void *combined_hidden = nullptr;
    void *combined_topk_weights = nullptr;
};

IncDcStatus IncDispatchHandleRetain(IncDispatchHandle *handle);
IncDcStatus IncDispatchHandleRelease(IncDispatchHandle **handle);
IncDcStatus IncDispatchHandleMarkStale(IncDispatchHandle *handle);

} // namespace dc
} // namespace inc

#endif
