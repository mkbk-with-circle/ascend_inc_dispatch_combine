#ifndef INC_DC_COMBINE_UPLOAD_TRANSPORT_H
#define INC_DC_COMBINE_UPLOAD_TRANSPORT_H

#include <cstdint>
#include <vector>

#include "inc_dc_combine_symmetric_layout.h"

namespace inc {
namespace dc {

// BW03: fixed worker upload AIV count (8 lanes).
constexpr uint32_t kCombineFixedUploadLaneCount = 8u;
constexpr uint32_t kCombineBw03MinPayloadPutBytes = 64u * 1024u;
constexpr uint32_t kCombineBw03MaxPayloadPutBytes = 256u * 1024u;
constexpr uint32_t kCombineBw03BatchRingCapacity = 256u;
constexpr uint32_t kCombineBw03ReadyRingCapacity = 512u;

struct IncDcCombineUploadLaneRange {
    uint32_t plan_begin = 0;
    uint32_t plan_end = 0; // exclusive index into upload plan
};

struct IncDcCombineBatchRingHeader {
    uint32_t head = 0;
    uint32_t tail = 0;
    uint32_t capacity = kCombineBw03BatchRingCapacity;
    uint32_t overflow = 0;
};

struct IncDcCombineBatchRingEntry {
    uint64_t publish_seq = 0;
    uint32_t plan_begin = 0;
    uint32_t plan_end = 0; // exclusive
    uint32_t worker_rank = 0;
    uint32_t lane_id = 0;
};

struct IncDcCombineReadyRingHeader {
    uint32_t head = 0;
    uint32_t tail = 0;
    uint32_t capacity = kCombineBw03ReadyRingCapacity;
    uint32_t overflow = 0;
};

struct IncDcCombineBw03DeviceLayout {
    uint64_t upload_lane_ranges = 0; // IncDcCombineUploadLaneRange[kCombineFixedUploadLaneCount]
    uint64_t batch_ring_hdr = 0;
    uint64_t batch_ring_entries = 0;
    uint64_t ready_ring_hdr = 0;
    uint64_t ready_ring_entries = 0; // uint32_t result_token_idx per slot
    uint64_t result_enqueued_bitmap = 0;
    uint64_t total_bytes = 0;
};

struct IncDcCombineUploadLanePartition {
    IncDcCombineUploadLaneRange lanes[kCombineFixedUploadLaneCount]{};
    uint32_t lane_count = kCombineFixedUploadLaneCount;
};

// Host: after sorting upload plan by locality, split into equal contiguous segments.
IncDcCombineUploadLanePartition PartitionUploadPlanIntoLanes(
    const std::vector<IncDcCombineUploadEntry> &plan, int32_t contributor_rank);

// Sort plan entries for BW03 locality (target_switch_pe, ingress_slot, expert_output_offset).
void SortCombineUploadPlanForBw03(std::vector<IncDcCombineUploadEntry> *plan);

IncDcCombineBw03DeviceLayout ComputeCombineBw03DeviceLayout(uint64_t base_off, uint32_t result_count);

} // namespace dc
} // namespace inc

#endif
