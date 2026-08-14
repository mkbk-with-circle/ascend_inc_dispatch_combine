#ifndef INC_FUSION_ROUTE_PACK_H
#define INC_FUSION_ROUTE_PACK_H

#include <cstdint>

#include "inc_fusion_abi.h"

namespace inc::fusion {

enum FusionRouteIdType : uint32_t {
    kFusionRouteInt32 = 1u,
    kFusionRouteInt64 = 2u,
};

enum FusionRoutePackError : uint32_t {
    kFusionRoutePackOk = 0u,
    kFusionRoutePackBadArgs = 1u,
    kFusionRoutePackBadExpert = 2u,
    kFusionRoutePackBadPlacement = 3u,
    kFusionRoutePackCapacity = 4u,
    kFusionRoutePackBadWeight = 5u,
};

struct alignas(64) FusionRoutePackStatus {
    uint32_t error = kFusionRoutePackOk;
    uint32_t error_token = 0u;
    uint32_t error_ordinal = 0u;
    uint32_t error_expert = 0u;
    uint32_t dispatch_row_count = 0u;
    uint32_t assignment_count = 0u;
    uint32_t wave_count = 0u;
    uint32_t reserved32 = 0u;
    uint64_t reserved64[4] = {};
};
static_assert(sizeof(FusionRoutePackStatus) == 64u,
              "route-pack status ABI");

// [wave_capacity, expert_count + 1] uint32_t elements.  The last column is
// metadata: row zero stores this worker's active-token count and inactive tail
// rows store zero.  One fixed-shape all-gather therefore exchanges both expert
// counts and dynamic rank lengths.
constexpr uint64_t FusionRouteLocalCountsBytes(uint32_t wave_count,
                                               uint32_t expert_count)
{
    return static_cast<uint64_t>(wave_count) * (expert_count + 1u) *
           sizeof(uint32_t);
}

// Two expert_count-sized uint32_t vectors: expert starts and local seen count.
constexpr uint64_t FusionRoutePackScratchBytes(uint32_t expert_count)
{
    return static_cast<uint64_t>(expert_count) * 2u * sizeof(uint32_t);
}

// Deterministic multi-AIV pack uses one private, cache-line-aligned expert
// histogram for every (wave,lane), one cache line of lane prefix state, one
// expert-base vector per wave and one error line per lane.  The layout depends
// only on prepared capacity and the hardware-derived lane count; no workload
// shape is tuned at run time.
constexpr uint64_t FusionRoutePackParallelScratchWords(
    uint32_t wave_capacity, uint32_t expert_count, uint32_t lane_count)
{
    const uint64_t line_words = kFusionCacheLineBytes / sizeof(uint32_t);
    const uint64_t expert_stride =
        (static_cast<uint64_t>(expert_count) + line_words - 1u) /
        line_words * line_words;
    return static_cast<uint64_t>(wave_capacity) * lane_count * expert_stride +
        static_cast<uint64_t>(wave_capacity) * lane_count * line_words +
        static_cast<uint64_t>(wave_capacity) * expert_stride +
        static_cast<uint64_t>(lane_count) * line_words;
}

constexpr uint64_t FusionRoutePackParallelScratchBytes(
    uint32_t wave_capacity, uint32_t expert_count, uint32_t lane_count)
{
    return FusionRoutePackParallelScratchWords(
        wave_capacity, expert_count, lane_count) * sizeof(uint32_t);
}

} // namespace inc::fusion

extern "C" void launch_inc_fusion_route_count_kernel(
    const void *topk_ids, uint32_t id_type,
    uint32_t token_count, uint32_t topk, uint32_t expert_count,
    uint32_t tokens_per_wave, uint32_t wave_capacity,
    uint32_t *local_counts,
    inc::fusion::FusionRoutePackStatus *status, void *stream);

// global_counts layout is [worker_count, wave_capacity, expert_count + 1]. The
// caller obtains it by an on-device all-gather of local_counts.  This metadata
// collective is common to all four baselines and is included in timed route
// packing; payload data never uses it.
extern "C" void launch_inc_fusion_route_pack_kernel(
    const void *topk_ids, uint32_t id_type, const float *topk_weights,
    const uint32_t *global_counts, const uint32_t *expert_owner,
    const uint32_t *expert_local_index, uint32_t worker_count,
    uint32_t rank, uint32_t token_count, uint32_t topk,
    uint32_t hidden, uint32_t element_bytes, uint32_t expert_count,
    uint32_t local_expert_count, uint32_t tokens_per_wave,
    uint32_t slot_count, uint32_t activation_waves,
    inc::fusion::FusionDispatchRow *dispatch_rows,
    uint32_t dispatch_row_capacity,
    inc::fusion::FusionExpertAssignment *assignments,
    uint32_t assignment_capacity, int64_t *group_lists,
    inc::fusion::FusionWaveDesc *waves, uint32_t wave_capacity,
    uint32_t *scratch, inc::fusion::FusionRoutePackStatus *status,
    void *stream);

// Analyze/prefix/emit implementation. `lane_count` must come from the live
// device profile and is capped by kFusionMaxAiv.  It produces exactly the same
// byte order as launch_inc_fusion_route_pack_kernel.  The scalar entry point
// remains available as a conservative fallback.
extern "C" void launch_inc_fusion_route_pack_parallel_kernel(
    const void *topk_ids, uint32_t id_type, const float *topk_weights,
    const uint32_t *global_counts, const uint32_t *expert_owner,
    const uint32_t *expert_local_index, uint32_t worker_count,
    uint32_t rank, uint32_t token_count, uint32_t topk,
    uint32_t hidden, uint32_t element_bytes, uint32_t expert_count,
    uint32_t local_expert_count, uint32_t tokens_per_wave,
    uint32_t slot_count, uint32_t activation_waves,
    inc::fusion::FusionDispatchRow *dispatch_rows,
    uint32_t dispatch_row_capacity,
    inc::fusion::FusionExpertAssignment *assignments,
    uint32_t assignment_capacity, int64_t *group_lists,
    inc::fusion::FusionWaveDesc *waves, uint32_t wave_capacity,
    uint32_t lane_count, uint32_t *parallel_scratch,
    inc::fusion::FusionRoutePackStatus *status, void *stream);

#endif
