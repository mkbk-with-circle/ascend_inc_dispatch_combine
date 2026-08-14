#ifndef INC_DC_SINGLE_INC_STREAM_PLAN_COMPILER_H
#define INC_DC_SINGLE_INC_STREAM_PLAN_COMPILER_H

#include <cstdint>
#include <vector>

#include "inc_dc_easy_api.h"
#include "inc_dc_resource_policy.h"
#include "inc_dc_single_inc_stream_abi.h"

namespace inc::dc::single_stream {

// Hardware-artifact tiling policy shared by every native entry point.  Keep
// at most about sixteen readiness epochs and align an epoch to the 1-MiB
// transport packet capacity, avoiding a systematic tiny tail on every tile.
inline uint32_t ResolveStreamPacketRows(uint32_t hidden_bytes)
{
    return hidden_bytes == 0u
        ? 0u : (hidden_bytes >= 1048576u ? 1u : 1048576u / hidden_bytes);
}

inline uint32_t ResolveStreamTileRows(
    uint32_t tokens, uint32_t hidden_bytes)
{
    const uint32_t packet_rows = ResolveStreamPacketRows(hidden_bytes);
    if (tokens == 0u || packet_rows == 0u) return 0u;
    const uint64_t epoch_rows = (static_cast<uint64_t>(tokens) + 15u) / 16u;
    const uint64_t aligned_epoch_rows =
        (epoch_rows + packet_rows - 1u) / packet_rows * packet_rows;
    const uint32_t base_rows = hidden_bytes >= 262144u
        ? 1u : 262144u / hidden_bytes;
    const uint64_t tile_rows =
        aligned_epoch_rows > base_rows ? aligned_epoch_rows : base_rows;
    return tile_rows > UINT32_MAX ? 0u : static_cast<uint32_t>(tile_rows);
}

struct StreamPlanCompileInput {
    uint32_t source_rank = 0u;
    uint32_t worker_world_size = 0u;
    uint32_t hidden_bytes = 0u;
    uint32_t tile_rows = 0u;
    uint32_t max_routes_per_packet = 0u;
    const void *host_token_plan = nullptr;
    uint64_t host_token_plan_bytes = 0u;
};

struct StreamCompiledSourcePlan {
    uint32_t source_rank = 0u;
    uint32_t worker_world_size = 0u;
    uint32_t tokens = 0u;
    uint32_t topk = 0u;
    uint32_t hidden_bytes = 0u;
    uint64_t semantic_digest = 0u;
    uint64_t generation = 0u;
    uint64_t logical_assignments = 0u;
    uint64_t physical_rows = 0u;
    uint64_t physical_output_bytes = 0u;
    std::vector<StreamDispatchTask> tasks;
    std::vector<StreamRouteEntry> routes;
    std::vector<StreamExpertAssignment> assignments;
};

struct StreamCompiledGlobalPlan {
    uint32_t worker_world_size = 0u;
    uint32_t tokens_per_worker = 0u;
    uint32_t topk = 0u;
    uint32_t hidden_bytes = 0u;
    uint64_t semantic_digest = 0u;
    uint64_t generation = 0u;
    uint64_t logical_assignments = 0u;
    uint64_t physical_rows = 0u;
    uint64_t physical_output_bytes = 0u;
    std::vector<StreamDispatchTask> tasks;
    std::vector<StreamRouteEntry> routes;
    std::vector<StreamExpertAssignment> assignments;
    std::vector<uint64_t> destination_output_offsets;
    std::vector<uint64_t> destination_physical_rows;
    std::vector<uint64_t> source_semantic_digests;
};

struct StreamWorkspaceBuildInput {
    uint32_t live_aiv = 0u;
    uint32_t tile_rows = 0u;
    // Direct HCCS MTE reads the peer-published GM source without L2.  Avoid a
    // whole-cache DCCI per packet and use two private UB/sync credits so the
    // fixed TX cohort can keep one packet in flight while preparing the next.
    uint32_t direct_dcci = 0u;
    uint32_t tx_pingpong = 1u;
};

struct StreamPreparedWorkspace {
    StreamDispatchDesc descriptor{};
    IncDcAivPolicy resources{};
    std::vector<StreamDispatchTask> tasks;
    std::vector<StreamRouteEntry> routes;
    std::vector<StreamExpertAssignment> assignments;
    std::vector<uint32_t> tx_lane_task_offsets;
    std::vector<uint32_t> tx_lane_task_indices;
    std::vector<uint32_t> worker_task_offsets;
    std::vector<uint32_t> worker_task_indices;
    std::vector<uint64_t> destination_output_offsets;
    std::vector<uint64_t> destination_physical_rows;
    std::vector<uint64_t> source_semantic_digests;
};

/* Pure host control-plane compilation. No allocation, stream, or device sync. */
bool CompileStreamSourcePlan(
    const StreamPlanCompileInput &input,
    StreamCompiledSourcePlan *compiled);

/* Collective control-plane merge; inputs may arrive in any rank order. */
bool MergeStreamSourcePlans(
    const std::vector<StreamCompiledSourcePlan> &sources,
    StreamCompiledGlobalPlan *compiled);

/* Builds the exact symmetric-heap descriptor/table layout used by kernel. */
bool BuildStreamPreparedWorkspace(
    const StreamCompiledGlobalPlan &plan,
    const StreamWorkspaceBuildInput &input,
    StreamPreparedWorkspace *workspace);

} // namespace inc::dc::single_stream

#endif
