#ifndef INC_DC_COMBINE_SYMMETRIC_LAYOUT_H
#define INC_DC_COMBINE_SYMMETRIC_LAYOUT_H

#include <cstdint>

#include "inc_dc_packet.h"
#include "inc_dc_types.h"
#include "inc_combine_trace.h"

namespace inc {
namespace dc {

// BW00 mc13_g8_t512_h7168_k8_random requires tokens/rank=512; was 257.
constexpr uint32_t kDcMaxTokensPerRank = 512;
constexpr uint32_t kDcMaxHidden = 8192;
constexpr uint32_t kDcAlign = 64;

inline uint64_t DcAlignUp(uint64_t v, uint64_t a = kDcAlign)
{
    return (v + a - 1) / a * a;
}

inline bool DcCheckedAddU64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr) {
        return false;
    }
    if (a > ~uint64_t{0} - b) {
        return false;
    }
    *out = a + b;
    return true;
}

inline bool DcCheckedMulU64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr) {
        return false;
    }
    if (a != 0 && b > (~uint64_t{0}) / a) {
        return false;
    }
    *out = a * b;
    return true;
}

// 设备 job 描述（host 写入对称堆）
struct IncDcDeviceJobDesc {
    uint32_t magic = 0x44434A42u; // DCJB
    uint32_t group_size = 0;
    uint32_t switch_rank = 0;
    uint32_t hidden_size = 0;
    uint32_t dtype_bytes = 2;
    uint32_t presync_mode = 0; // IncDcPresyncMode
    uint32_t ring_depth = 1;
    uint32_t token_block_tokens = 1;
    uint64_t route_epoch = 0;
    uint64_t op_seq = 0;
    uint32_t num_global_experts = 0;
    uint32_t local_rank = 0;
    uint32_t num_local_tokens = 0;
    uint32_t recv_assignment_count = 0;
    uint32_t send_network_copies = 0;
    uint32_t remote_plan_count = 0;
    uint32_t recv_count_per_rank[16] = {};
    uint32_t macro_block_count = 0;  // P5c persistent: token-blocks per epoch
    uint32_t epoch_seq = 0;          // P5c presync v2 monotonic seq
    uint32_t presync_protocol = 0;   // 0=legacy, 2=v2
    uint32_t pipeline_mode = 0;      // 0=wave, 1=persistent
    uint32_t local_hidden_rows = 0;  // combine source capacity (contributor-local rows)
    uint32_t max_tokens_per_rank = 0; // recv/egress row stride per rank
};

struct IncDcDeviceRecvEntry {
    int32_t dst_rank = -1;
    int32_t dst_slot = -1;
    uint32_t ingress_slot = 0; // src_rank * kDcMaxTokensPerRank + src_token
    int32_t expert_id = -1;
};

struct IncDcDeviceTokenDesc {
    int32_t source_token = -1;
    uint32_t destination_mask = 0;
    uint32_t num_assignments = 0;
    uint32_t hidden_offset_elems = 0; // 相对 local hidden staging
    int32_t expert_ids[kMaxTopk] = {};
    int32_t route_slots[kMaxTopk] = {};
    float weights[kMaxTopk] = {};
};

struct IncDcCombineUploadEntry {
    int32_t contributor_rank = -1;
    uint32_t assignment_id = 0;
    uint32_t expert_output_offset = 0;
    uint32_t result_token_idx = 0;
    uint32_t ingress_slot = 0;
    uint32_t target_switch_pe = 0; // dst-rank paired INC SHMEM PE (paired combine routing)
};

struct IncDcSymmetricLayout {
    uint64_t job_desc = 0;
    uint64_t control = 0; // phase, error_latch, first_bitmap, service_count
    uint64_t token_desc = 0;
    uint64_t local_hidden = 0;
    uint64_t switch_ingress = 0;
    uint64_t recv_hidden = 0;
    uint64_t recv_meta = 0;
    uint64_t slot_ring = 0;
    uint64_t trace = 0;
    uint64_t switch_probe = 0;
    uint64_t total_bytes = 0;
    // Capacity metadata (bytes/rows) for fail-closed bounds — not offsets.
    uint32_t group_size = 0;
    uint32_t max_tokens_per_rank = 0;
    uint32_t local_hidden_rows = 0;   // source capacity
    uint32_t ingress_slots = 0;
    uint32_t recv_hidden_rows = 0;    // egress/recv capacity (= max_tokens * group)
    uint32_t max_recv_assignments = 0;
    uint64_t hidden_stride = 0;
    uint64_t pkt_stride = 0;
    uint64_t local_hidden_bytes = 0;
    uint64_t switch_ingress_bytes = 0;
    uint64_t recv_hidden_bytes = 0;
};

struct IncDcDeviceSlotDesc {
    uint32_t state = 0;
    uint32_t epoch_seq = 0;
    uint32_t token_begin = 0;
    uint32_t token_count = 0;
    uint32_t bytes = 0;
    uint32_t pad[3] = {};
};

// ingress_slot_count: 0 → default max_tokens_per_rank * group_size * ring_depth.
// Combine may index ingress by assignment_id (up to total assignments); pass an explicit
// count so putmem cannot walk past ingress into recv_meta/trace.
// local_hidden_rows: 0 → max_tokens_per_rank (dispatch default). Combine must pass
// max contributor-local compact rows so expert_output_offset cannot OOB into ingress.
// recv_hidden_rows: 0 → max_tokens_per_rank * group_size (single-INC global egress).
// Paired combine should pass max(actual tokens_per_rank) for destination-local egress.
inline IncDcSymmetricLayout ComputeDispatchSymmetricLayout(uint32_t group_size, uint32_t hidden_size,
                                                           uint32_t dtype_bytes, uint32_t max_tokens_per_rank,
                                                           uint32_t max_recv_assignments, uint32_t ring_depth,
                                                           uint32_t ingress_slot_count = 0,
                                                           uint32_t local_hidden_rows = 0,
                                                           uint32_t recv_hidden_rows = 0)
{
    IncDcSymmetricLayout lo{};
    uint64_t off = 0;
    off = DcAlignUp(off + sizeof(IncDcDeviceJobDesc));
    lo.job_desc = 0;
    lo.control = off;
    off += DcAlignUp(512); // control + v2 presync seq arrays
    lo.token_desc = off;
    off += DcAlignUp(static_cast<uint64_t>(max_tokens_per_rank) * sizeof(IncDcDeviceTokenDesc) * group_size);
    lo.local_hidden = off;
    const uint64_t hidden_stride = DcAlignUp(static_cast<uint64_t>(hidden_size) * dtype_bytes);
    const uint32_t lh_rows = local_hidden_rows > 0 ? local_hidden_rows : max_tokens_per_rank;
    uint64_t local_hidden_bytes = 0;
    if (!DcCheckedMulU64(hidden_stride, lh_rows, &local_hidden_bytes)) {
        lo.total_bytes = 0;
        return lo;
    }
    local_hidden_bytes = DcAlignUp(local_hidden_bytes);
    off += local_hidden_bytes;
    off += 64; // MEM1 guard canary pad (local_hidden | switch_ingress)
    lo.switch_ingress = off;
    const uint64_t pkt_stride =
        DcAlignUp(kDcPacketHdrBytes + hidden_stride + kMaxTopk * sizeof(IncDcAssignmentPkt));
    const uint32_t rd = ring_depth > 0 ? ring_depth : 1;
    const uint32_t default_ingress = max_tokens_per_rank * group_size * rd;
    const uint32_t ingress_n = ingress_slot_count > 0 ? ingress_slot_count : default_ingress;
    uint64_t ingress_bytes = 0;
    if (!DcCheckedMulU64(pkt_stride, ingress_n, &ingress_bytes)) {
        lo.total_bytes = 0;
        return lo;
    }
    off += ingress_bytes;
    off += 64; // MEM1 guard canary pad (switch_ingress | recv_hidden)
    lo.recv_hidden = off;
    // Paired: destination-local rows; single-INC: rank*stride+tok uniqueness.
    const uint32_t recv_rows =
        recv_hidden_rows > 0 ? recv_hidden_rows : (max_tokens_per_rank * group_size);
    uint64_t recv_bytes = 0;
    if (!DcCheckedMulU64(hidden_stride, recv_rows, &recv_bytes)) {
        lo.total_bytes = 0;
        return lo;
    }
    off += recv_bytes;
    off += 64; // MEM1 guard canary pad (recv_hidden | recv_meta)
    lo.recv_meta = off;
    off += DcAlignUp(static_cast<uint64_t>(max_recv_assignments) * sizeof(IncDcDeviceRecvEntry));
    off += 64; // MEM1 guard canary pad (recv_meta | slot_ring)
    lo.slot_ring = off;
    off += DcAlignUp(static_cast<uint64_t>(rd) * sizeof(IncDcDeviceSlotDesc));
    lo.trace = off;
    // header(128) + cells(40*128) + MC09 stage ring header(64) + events(2048*32) + switch probe(128)
    off += DcAlignUp(inc::dc::DcCombineTraceRegionBytes());
    lo.switch_probe = off - inc::dc::kDcCombineSwitchProbeBytes;
    lo.total_bytes = DcAlignUp(off);
    lo.group_size = group_size;
    lo.max_tokens_per_rank = max_tokens_per_rank;
    lo.local_hidden_rows = lh_rows;
    lo.ingress_slots = ingress_n;
    lo.recv_hidden_rows = recv_rows;
    lo.max_recv_assignments = max_recv_assignments;
    lo.hidden_stride = hidden_stride;
    lo.pkt_stride = pkt_stride;
    lo.local_hidden_bytes = local_hidden_bytes;
    lo.switch_ingress_bytes = ingress_bytes;
    lo.recv_hidden_bytes = recv_bytes;
    return lo;
}

// P5c v2: ring_depth token-block slots (not full 257 expansion for ingress)
inline IncDcSymmetricLayout ComputeDispatchSymmetricLayoutV2(uint32_t group_size, uint32_t hidden_size,
                                                             uint32_t dtype_bytes, uint32_t max_tokens_per_rank,
                                                             uint32_t max_recv_assignments, uint32_t ring_depth,
                                                             uint32_t local_hidden_rows = 0)
{
    IncDcSymmetricLayout lo{};
    uint64_t off = 0;
    off = DcAlignUp(off + sizeof(IncDcDeviceJobDesc));
    lo.job_desc = 0;
    lo.control = off;
    off += DcAlignUp(512); // v2 presync seq arrays
    lo.token_desc = off;
    off += DcAlignUp(static_cast<uint64_t>(max_tokens_per_rank) * sizeof(IncDcDeviceTokenDesc) * group_size);
    lo.local_hidden = off;
    const uint64_t hidden_stride = DcAlignUp(static_cast<uint64_t>(hidden_size) * dtype_bytes);
    const uint32_t lh_rows = local_hidden_rows > 0 ? local_hidden_rows : max_tokens_per_rank;
    uint64_t local_hidden_bytes = 0;
    if (!DcCheckedMulU64(hidden_stride, lh_rows, &local_hidden_bytes)) {
        lo.total_bytes = 0;
        return lo;
    }
    local_hidden_bytes = DcAlignUp(local_hidden_bytes);
    off += local_hidden_bytes;
    off += 64;
    lo.switch_ingress = off;
    const uint64_t pkt_stride =
        DcAlignUp(kDcPacketHdrBytes + hidden_stride + kMaxTopk * sizeof(IncDcAssignmentPkt));
    const uint32_t rd = ring_depth > 0 ? ring_depth : 1;
    const uint32_t ingress_n = rd * group_size;
    uint64_t ingress_bytes = 0;
    if (!DcCheckedMulU64(pkt_stride, ingress_n, &ingress_bytes)) {
        lo.total_bytes = 0;
        return lo;
    }
    off += ingress_bytes;
    off += 64;
    lo.recv_hidden = off;
    const uint32_t recv_rows = max_tokens_per_rank * group_size;
    uint64_t recv_bytes = 0;
    if (!DcCheckedMulU64(hidden_stride, recv_rows, &recv_bytes)) {
        lo.total_bytes = 0;
        return lo;
    }
    off += recv_bytes;
    off += 64;
    lo.recv_meta = off;
    off += DcAlignUp(static_cast<uint64_t>(max_recv_assignments) * sizeof(IncDcDeviceRecvEntry));
    off += 64;
    lo.slot_ring = off;
    off += DcAlignUp(static_cast<uint64_t>(rd) * sizeof(IncDcDeviceSlotDesc));
    lo.trace = off;
    off += DcAlignUp(inc::dc::DcCombineTraceRegionBytes());
    lo.switch_probe = off - inc::dc::kDcCombineSwitchProbeBytes;
    lo.total_bytes = DcAlignUp(off);
    lo.group_size = group_size;
    lo.max_tokens_per_rank = max_tokens_per_rank;
    lo.local_hidden_rows = lh_rows;
    lo.ingress_slots = ingress_n;
    lo.recv_hidden_rows = recv_rows;
    lo.max_recv_assignments = max_recv_assignments;
    lo.hidden_stride = hidden_stride;
    lo.pkt_stride = pkt_stride;
    lo.local_hidden_bytes = local_hidden_bytes;
    lo.switch_ingress_bytes = ingress_bytes;
    lo.recv_hidden_bytes = recv_bytes;
    return lo;
}

} // namespace dc
} // namespace inc

#endif
