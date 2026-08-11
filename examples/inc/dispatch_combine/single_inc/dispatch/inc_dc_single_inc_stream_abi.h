#ifndef INC_DC_SINGLE_INC_STREAM_ABI_H
#define INC_DC_SINGLE_INC_STREAM_ABI_H

#include <cstdint>

#include "inc_dc_platform_capabilities.h"

namespace inc::dc::single_stream {

constexpr uint32_t kStreamMagic = 0x5354524du; // STRM
constexpr uint32_t kStreamVersion = 1u;
constexpr uint32_t kStreamGeneration = 1u;
// ABI storage capacity, not the launch width.  The active width is queried
// from the live device and partitioned at runtime.  Keep headroom for newer
// products with more AIVs instead of silently capping them at this machine's
// 48-core count.
constexpr uint32_t kStreamMaxLanes = kIncDcMaxDispatchLanes;
constexpr uint64_t kStreamDescOff = 0u;
constexpr uint64_t kStreamDataOff = 4096u;
constexpr uint32_t kStreamFlagDirectDcci = 1u << 0;
constexpr uint32_t kStreamFlagWorkerPack = 1u << 1;
constexpr uint32_t kStreamFlagHasDirect = 1u << 2;
constexpr uint32_t kStreamFlagWorkerDirect = 1u << 3;

inline constexpr uint64_t StreamAlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1u) / alignment * alignment;
}

// One task packs a set of rows from one already-published source tile and
// sends the resulting contiguous packet to one destination worker.
struct alignas(64) StreamDispatchTask {
    uint32_t source_rank = 0u;
    uint32_t destination_rank = 0u;
    uint32_t source_tile = 0u;
    uint32_t route_begin = 0u;
    uint32_t route_count = 0u;
    uint32_t reserved0 = 0u;
    uint64_t output_byte_offset = 0u;
    uint64_t packet_bytes = 0u;
    uint64_t reserved1[2] = {};
};
static_assert(sizeof(StreamDispatchTask) == 64u, "stream task ABI");

// source_row is rank-local.  Tasks guarantee every referenced row belongs to
// task.source_rank/source_tile; keeping the wire entry small improves cache
// behaviour for arbitrary top-k route tables.
struct StreamRouteEntry {
    uint32_t source_rank = 0u;
    uint32_t source_row = 0u;
    // One physical row may represent several expert instances when they live
    // on the same destination rank.  The destination expands this compact
    // row using the assignment slice; HCCS carries the hidden payload once.
    uint32_t assignment_begin = 0u;
    uint32_t assignment_count = 0u;
};
static_assert(sizeof(StreamRouteEntry) == 16u, "stream route ABI");

struct StreamExpertAssignment {
    uint32_t expert_id = 0u;
    uint32_t route_ordinal = 0u;
    uint32_t weight_bits = 0u;
    uint32_t reserved = 0u;
};
static_assert(sizeof(StreamExpertAssignment) == 16u,
              "stream expert assignment ABI");

struct alignas(64) StreamLaneStat {
    uint64_t start_cycle = 0u;
    uint64_t end_cycle = 0u;
    uint64_t input_bytes = 0u;
    uint64_t output_bytes = 0u;
    uint64_t gather_cycles = 0u;
    uint64_t transport_cycles = 0u;
    uint32_t tasks = 0u;
    uint32_t error = 0u;
};
static_assert(sizeof(StreamLaneStat) == 64u, "stream stat ABI");

struct alignas(64) StreamDispatchDesc {
    uint32_t magic = kStreamMagic;
    uint32_t version = kStreamVersion;
    uint32_t generation = kStreamGeneration;
    uint32_t pe = 0u;
    uint32_t workers = 0u;
    uint32_t lane_count = 0u;
    uint32_t gather_lane_count = 0u;
    uint32_t tx_lane_count = 0u;
    uint32_t upload_lane_count = 0u;
    uint32_t hidden_bytes = 0u;
    uint32_t tokens_per_worker = 0u;
    uint32_t topk = 0u;
    uint32_t tile_rows = 0u;
    uint32_t tiles_per_worker = 0u;
    uint32_t task_count = 0u;
    uint32_t route_count = 0u;
    uint32_t task_route_capacity = 0u;
    uint32_t spin_cap = 0u;
    uint32_t tx_window = 0u;
    uint32_t gather_chunk_routes = 0u;
    uint32_t gather_chunk_count = 0u;
    uint32_t direct_task_count = 0u;
    uint32_t reserved32 = 0u;
    // Direct packets stage through two private UB buffers with distinct sync
    // ids instead of the shared per-AIV slot the public putmem path owns.
    uint32_t tx_pingpong = 0u;
    // Sparse-source uploads may use the same two-credit private-MTE scheme.
    // This is selected from runtime work density and the live AIV budget; it
    // is deliberately independent of a W2/W4/W8 product table.
    uint32_t upload_pingpong = 0u;
    uint64_t input_stride = 0u;
    uint64_t tile_bytes = 0u;
    uint64_t max_packet_bytes = 0u;
    uint64_t logical_input_bytes = 0u;
    uint64_t logical_output_bytes = 0u;
    uint64_t input_off = 0u;
    uint64_t tile_ready_off = 0u;
    uint64_t direct_ready_off = 0u;
    uint64_t task_off = 0u;
    uint64_t route_off = 0u;
    uint64_t staging_off = 0u;
    uint64_t output_off = 0u;
    uint64_t upload_chunk_done_off = 0u;
    uint64_t lane_done_off = 0u;
    uint64_t completion_off = 0u;
    uint64_t start_gate_off = 0u;
    uint64_t stats_off = 0u;
    uint64_t total_bytes = 0u;
    // Optional host-compiled, lane-major task layout.  Offsets contains
    // tx_lane_count + 1 uint32_t entries and removes the O(lanes*tasks)
    // device-side ownership scan without introducing an indirect worklist.
    uint64_t tx_lane_task_offsets_off = 0u;
    // When tx_lane_tasks_contiguous == 2, offsets slice this task-index
    // array.  It removes the O(TX lanes * tasks) ownership scan while the
    // canonical task table retains producer/pipeline order.
    uint64_t tx_lane_task_indices_off = 0u;
    uint64_t tx_lane_tasks_contiguous = 0u;
    uint64_t expert_assignment_off = 0u;
    uint64_t expert_assignment_count = 0u;
    // Host-compiled per-source slices into a task-index array. Worker AIVs
    // traverse only their own tasks while the task table remains tile-major
    // for prompt INC consumption.
    uint64_t worker_task_offsets_off = 0u;
    uint64_t worker_task_indices_off = 0u;
    // Qualification/framework-managed absolute device-cycle release.  Zero
    // preserves the ordinary production launch path.
    uint64_t start_target_cycle = 0u;
};
static_assert(sizeof(StreamDispatchDesc) % 64u == 0u, "stream desc alignment");

} // namespace inc::dc::single_stream

#endif
