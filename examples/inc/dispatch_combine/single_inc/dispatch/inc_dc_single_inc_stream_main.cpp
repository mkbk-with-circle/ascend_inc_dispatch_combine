#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <sched.h>

#include "acl/acl.h"
#include "param.h"
#include "shmem.h"
#include "utils.h"

#include "inc_dc_single_inc_stream_abi.h"
#include "inc_dc_external_start_gate.h"
#include "inc_dc_physical_map.h"
#include "inc_dc_resource_policy.h"

using namespace inc::dc::single_stream;

extern "C" void launch_inc_dc_single_inc_stream_dispatch_kernel(
    uint8_t *sym, int block_dim, void *stream);
extern "C" void launch_inc_dc_single_inc_stream_cycle_probe_kernel(
    uint8_t *sym, uint64_t cycle_off, void *stream);

int g_npus = 16;
const char *ipport = "tcp://127.0.0.1:8969";
int f_npu = 0;
aclshmemx_uniqueid_t default_flag_uid;

static int ResolvePhysicalNpuForPe(int pe)
{
    return inc::dc::ResolvePhysicalNpuForPe(pe, g_npus, f_npu);
}

static uint8_t PayloadByte(uint32_t source, uint32_t row, uint32_t byte)
{
    return static_cast<uint8_t>((source * 29u + row * 17u + byte * 13u + 7u) &
                                0xffu);
}

static int InitShmem(int pe, int npe, uint64_t heap_bytes, int32_t *dev,
                     aclrtStream *stream)
{
    *dev = ResolvePhysicalNpuForPe(pe);
    int st = aclInit(nullptr);
    if (st == 0) st = aclrtSetDevice(*dev);
    if (st == 0) st = aclrtCreateStream(stream);
    if (st != 0) return st;
    aclshmemx_init_attr_t attr;
    test_set_attr(pe, npe, heap_bytes, ipport, default_flag_uid, &attr);
    return aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
}

// Planning precedes symmetric-heap sizing, but its bounded credit window must
// use the same live AIV policy as the eventual launch.  Query the device in a
// short ACL-only session; no NPU memory or SHMEM endpoint is created here.
// The measured interval starts much later, after the normal InitShmem path.
static bool QueryLiveAivForPlanning(int pe, uint32_t *aiv)
{
    if (aiv == nullptr) return false;
    const int32_t dev = ResolvePhysicalNpuForPe(pe);
    int st = aclInit(nullptr);
    if (st == 0) st = aclrtSetDevice(dev);
    int64_t value = 0;
    if (st == 0) {
        st = aclrtGetDeviceInfo(dev, ACL_DEV_ATTR_VECTOR_CORE_NUM, &value);
    }
    if (st == 0) st = aclrtResetDevice(dev);
    const int finalize_st = aclFinalize();
    if (st == 0 && finalize_st != 0) st = finalize_st;
    if (st != 0 || value <= 0 ||
        value > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    *aiv = static_cast<uint32_t>(value);
    return true;
}

static void Cleanup(int32_t dev, aclrtStream stream, uint8_t *sym)
{
    if (sym != nullptr) aclshmem_free(sym);
    aclshmem_finalize();
    if (stream != nullptr) aclrtDestroyStream(stream);
    aclrtResetDevice(dev);
    aclFinalize();
}

static bool AddRegion(uint64_t bytes, uint64_t *cursor, uint64_t *off)
{
    *cursor = StreamAlignUp(*cursor, 4096u);
    *off = *cursor;
    if (bytes > std::numeric_limits<uint64_t>::max() - *cursor) return false;
    *cursor += bytes;
    return true;
}

static int CopyH2D(uint8_t *dst, const uint8_t *src, uint64_t bytes)
{
    constexpr uint64_t chunk = 64ull * 1024ull * 1024ull;
    for (uint64_t off = 0u; off < bytes; off += chunk) {
        const uint64_t n = std::min(chunk, bytes - off);
        const int st = aclrtMemcpy(dst + off, n, src + off, n,
                                   ACL_MEMCPY_HOST_TO_DEVICE);
        if (st != 0) return st;
    }
    return 0;
}

static int CopyD2H(uint8_t *dst, const uint8_t *src, uint64_t bytes)
{
    constexpr uint64_t chunk = 64ull * 1024ull * 1024ull;
    for (uint64_t off = 0u; off < bytes; off += chunk) {
        const uint64_t n = std::min(chunk, bytes - off);
        const int st = aclrtMemcpy(dst + off, n, src + off, n,
                                   ACL_MEMCPY_DEVICE_TO_HOST);
        if (st != 0) return st;
    }
    return 0;
}

static uint32_t RouteDestination(uint32_t workers, uint32_t source,
                                 uint32_t row, uint32_t ordinal,
                                 uint32_t route_mode)
{
    uint32_t destination = (source * 3u + row + ordinal) % workers;
    if (route_mode == 3u) return 0u;
    if (route_mode == 4u) {
        const uint32_t active = workers <= 1u ? 1u : (workers + 1u) / 2u;
        return destination % active;
    }
    if (route_mode >= 5u && route_mode <= 8u) {
        static constexpr uint32_t redirected_eighths[] = {1u, 3u, 5u, 7u};
        uint64_t mixed =
            (static_cast<uint64_t>(source) << 40u) ^
            (static_cast<uint64_t>(row) << 8u) ^ ordinal ^
            0x9e3779b97f4a7c15ull;
        mixed = (mixed ^ (mixed >> 30u)) * 0xbf58476d1ce4e5b9ull;
        mixed = (mixed ^ (mixed >> 27u)) * 0x94d049bb133111ebull;
        mixed ^= mixed >> 31u;
        if (static_cast<uint32_t>(mixed >> 61u) <
            redirected_eighths[route_mode - 5u]) {
            destination = 0u;
        }
    }
    return destination;
}

struct HostTokenAssignment {
    uint32_t destination_rank = 0u;
    uint32_t expert_id = 0u;
    float weight = 1.f;
};

static uint32_t FloatBits(float value)
{
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static bool BuildTokenPlan(uint32_t workers, uint32_t tokens, uint32_t topk,
                           uint32_t route_mode,
                           std::vector<HostTokenAssignment> *plan,
                           std::string *plan_source)
{
    if (plan == nullptr || plan_source == nullptr || workers == 0u ||
        tokens == 0u || topk == 0u) {
        return false;
    }
    const uint64_t count64 = static_cast<uint64_t>(workers) * tokens * topk;
    if (count64 > std::numeric_limits<size_t>::max() ||
        count64 > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    plan->assign(static_cast<size_t>(count64), HostTokenAssignment{});
    const char *path = std::getenv("INC_STREAM_TOKEN_PLAN_FILE");
    if (path == nullptr || path[0] == '\0') {
        for (uint32_t source = 0u; source < workers; ++source) {
            for (uint32_t row = 0u; row < tokens; ++row) {
                for (uint32_t ordinal = 0u; ordinal < topk; ++ordinal) {
                    const uint64_t index =
                        (static_cast<uint64_t>(source) * tokens + row) * topk +
                        ordinal;
                    const uint32_t destination = RouteDestination(
                        workers, source, row, ordinal, route_mode);
                    // Different ordinals remain different expert instances even
                    // when K > W maps several of them to the same rank.
                    const uint64_t expert =
                        static_cast<uint64_t>(ordinal) * workers + destination;
                    if (expert > std::numeric_limits<uint32_t>::max()) {
                        return false;
                    }
                    (*plan)[index] = HostTokenAssignment{
                        destination, static_cast<uint32_t>(expert), 1.f};
                }
            }
        }
        *plan_source = "synthetic";
        return true;
    }

    std::ifstream input(path);
    if (!input) return false;
    std::string line;
    if (!std::getline(input, line)) return false;
    std::istringstream header(line);
    std::string magic;
    uint32_t file_workers = 0u, file_tokens = 0u, file_topk = 0u;
    if (!(header >> magic >> file_workers >> file_tokens >> file_topk) ||
        magic != "INC_TOKEN_PLAN_V1" || file_workers != workers ||
        file_tokens != tokens || file_topk != topk) {
        return false;
    }
    std::vector<uint8_t> seen(static_cast<size_t>(count64), 0u);
    uint64_t parsed = 0u;
    while (std::getline(input, line)) {
        const size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') continue;
        std::istringstream row_stream(line);
        uint32_t source = 0u, row = 0u, ordinal = 0u, destination = 0u;
        uint64_t expert64 = 0u;
        float weight = 0.f;
        if (!(row_stream >> source >> row >> ordinal >> destination >>
              expert64 >> weight) ||
            source >= workers || row >= tokens || ordinal >= topk ||
            destination >= workers ||
            expert64 > std::numeric_limits<uint32_t>::max() ||
            !std::isfinite(weight)) {
            return false;
        }
        std::string extra;
        if (row_stream >> extra) return false;
        const uint64_t index =
            (static_cast<uint64_t>(source) * tokens + row) * topk + ordinal;
        if (seen[index] != 0u) return false;
        seen[index] = 1u;
        (*plan)[index] = HostTokenAssignment{
            destination, static_cast<uint32_t>(expert64), weight};
        ++parsed;
    }
    if (parsed != count64) return false;
    *plan_source = path;
    return true;
}

static bool BuildTasks(uint32_t workers, uint32_t tokens, uint32_t topk,
                       uint32_t tile_rows, uint32_t route_capacity,
                       uint32_t hidden_bytes,
                       const std::vector<HostTokenAssignment> &plan,
                       bool split_runs,
                       std::vector<StreamDispatchTask> *tasks,
                       std::vector<StreamRouteEntry> *routes,
                       std::vector<StreamExpertAssignment> *assignments)
{
    if (tasks == nullptr || routes == nullptr || assignments == nullptr) {
        return false;
    }
    tasks->clear();
    routes->clear();
    assignments->clear();
    if (workers == 0u || tokens == 0u || topk == 0u || tile_rows == 0u ||
        route_capacity == 0u) {
        return false;
    }
    uint64_t output_cursor = 0u;
    const uint32_t tile_count = (tokens + tile_rows - 1u) / tile_rows;
    for (uint32_t tile = 0u; tile < tile_count; ++tile) {
        const uint32_t row_begin = tile * tile_rows;
        const uint32_t row_end = std::min(tokens, row_begin + tile_rows);
        for (uint32_t source = 0u; source < workers; ++source) {
            for (uint32_t destination = 0u; destination < workers;
                 ++destination) {
                std::vector<StreamRouteEntry> packet;
                for (uint32_t row = row_begin; row < row_end; ++row) {
                    const uint32_t assignment_begin =
                        static_cast<uint32_t>(assignments->size());
                    for (uint32_t ordinal = 0u; ordinal < topk; ++ordinal) {
                        const uint64_t plan_index =
                            (static_cast<uint64_t>(source) * tokens + row) *
                                topk +
                            ordinal;
                        if (plan_index >= plan.size()) return false;
                        const auto &assignment = plan[plan_index];
                        if (assignment.destination_rank == destination) {
                            assignments->push_back(StreamExpertAssignment{
                                assignment.expert_id, ordinal,
                                FloatBits(assignment.weight), 0u});
                        }
                    }
                    const uint32_t assignment_count =
                        static_cast<uint32_t>(assignments->size()) -
                        assignment_begin;
                    if (assignment_count != 0u) {
                        packet.push_back(StreamRouteEntry{
                            source, row, assignment_begin, assignment_count});
                    }
                }
                size_t begin = 0u;
                while (begin < packet.size()) {
                    // Cut the destination slice wherever the source rows stop
                    // being consecutive.  Every emitted task then reads a
                    // contiguous ingress range and can be forwarded straight
                    // from it, so no shape needs the gather/staging stage.
                    // Transport measures the same rate at one row per packet
                    // as at eight, so the coalescing that staging used to buy
                    // is not worth an extra read/write pass on the INC.
                    size_t run_end = begin + 1u;
                    while (run_end < packet.size() &&
                           split_runs &&
                           packet[run_end].source_row ==
                               packet[run_end - 1u].source_row + 1u &&
                           run_end - begin < route_capacity) {
                        ++run_end;
                    }
                    if (!split_runs) {
                        run_end = std::min<size_t>(begin + route_capacity,
                                                   packet.size());
                    }
                    const uint32_t count =
                        static_cast<uint32_t>(run_end - begin);
                    if (routes->size() + count >
                            std::numeric_limits<uint32_t>::max() ||
                        tasks->size() ==
                            std::numeric_limits<uint32_t>::max()) {
                        return false;
                    }
                    StreamDispatchTask task{};
                    task.source_rank = source;
                    task.destination_rank = destination;
                    task.source_tile = tile;
                    task.route_begin =
                        static_cast<uint32_t>(routes->size());
                    task.route_count = count;
                    task.output_byte_offset = output_cursor;
                    task.packet_bytes =
                        static_cast<uint64_t>(count) * hidden_bytes;
                    tasks->push_back(task);
                    routes->insert(routes->end(), packet.begin() + begin,
                                   packet.begin() + begin + count);
                    output_cursor += task.packet_bytes;
                    begin = run_end;
                }
            }
        }
    }
    return true;
}

static bool PartitionTasksByTxLane(
    uint32_t workers, uint32_t tx_lane_count,
    std::vector<StreamDispatchTask> *tasks,
    std::vector<uint32_t> *lane_offsets)
{
    if (workers == 0u || tx_lane_count == 0u || tasks == nullptr ||
        lane_offsets == nullptr) {
        return false;
    }
    std::vector<std::vector<StreamDispatchTask>> lane_tasks(tx_lane_count);
    for (uint32_t task_index = 0u; task_index < tasks->size(); ++task_index) {
        const StreamDispatchTask &task = (*tasks)[task_index];
        if (task.destination_rank >= workers) return false;
        // A fixed INC service must remain fully occupied for arbitrary route
        // skew.  Pinning lanes to peers leaves 14/16 lanes idle in W8
        // all-to-one even though one peer can consume the whole HCCS port
        // group.  Assign the canonical task stream round-robin across the
        // unchanged TX cohort instead.  Every lane may target every peer;
        // correctness and output placement remain task-defined.
        const uint32_t lane = task_index % tx_lane_count;
        if (lane >= tx_lane_count) return false;
        lane_tasks[lane].push_back(task);
    }
    lane_offsets->assign(tx_lane_count + 1u, 0u);
    std::vector<StreamDispatchTask> ordered;
    ordered.reserve(tasks->size());
    for (uint32_t lane = 0u; lane < tx_lane_count; ++lane) {
        (*lane_offsets)[lane] = static_cast<uint32_t>(ordered.size());
        ordered.insert(ordered.end(), lane_tasks[lane].begin(),
                       lane_tasks[lane].end());
    }
    (*lane_offsets)[tx_lane_count] = static_cast<uint32_t>(ordered.size());
    tasks->swap(ordered);
    return true;
}

static bool BuildTxLaneTaskWorklist(
    uint32_t workers, uint32_t tx_lane_count,
    const std::vector<StreamDispatchTask> &tasks,
    std::vector<uint32_t> *lane_offsets,
    std::vector<uint32_t> *lane_task_indices)
{
    if (workers == 0u || tx_lane_count == 0u || lane_offsets == nullptr ||
        lane_task_indices == nullptr) {
        return false;
    }
    std::vector<std::vector<uint32_t>> lane_tasks(tx_lane_count);
    for (uint32_t task_index = 0u; task_index < tasks.size(); ++task_index) {
        const auto &task = tasks[task_index];
        if (task.destination_rank >= workers) return false;
        const uint32_t lane = task_index % tx_lane_count;
        if (lane >= tx_lane_count) return false;
        lane_tasks[lane].push_back(task_index);
    }
    lane_offsets->assign(tx_lane_count + 1u, 0u);
    lane_task_indices->clear();
    lane_task_indices->reserve(tasks.size());
    for (uint32_t lane = 0u; lane < tx_lane_count; ++lane) {
        (*lane_offsets)[lane] =
            static_cast<uint32_t>(lane_task_indices->size());
        lane_task_indices->insert(lane_task_indices->end(),
                                  lane_tasks[lane].begin(),
                                  lane_tasks[lane].end());
    }
    (*lane_offsets)[tx_lane_count] =
        static_cast<uint32_t>(lane_task_indices->size());
    return lane_task_indices->size() == tasks.size();
}

int main(int argc, char **argv)
{
    if (argc < 11) {
        std::cerr
            << "usage: inc_dc_single_inc_stream <npe> <pe> <ipport> <g_npus> <f_npu> "
               "<tokens_per_worker> <hidden_bytes> <topk> <tile_bytes> <max_packet_bytes> "
               "[warmup] [measure] [inc_lanes] [upload_lanes]\n";
        return 2;
    }
    const int npe = std::atoi(argv[1]);
    const int pe = std::atoi(argv[2]);
    ipport = argv[3];
    g_npus = std::atoi(argv[4]);
    f_npu = std::atoi(argv[5]);
    const uint32_t workers = npe > 0 ? static_cast<uint32_t>(npe - 1) : 0u;
    const uint32_t tokens = static_cast<uint32_t>(std::strtoul(argv[6], nullptr, 10));
    const uint32_t hidden_bytes = static_cast<uint32_t>(std::strtoul(argv[7], nullptr, 10));
    const uint32_t topk = static_cast<uint32_t>(std::strtoul(argv[8], nullptr, 10));
    const uint64_t requested_tile_bytes = std::strtoull(argv[9], nullptr, 10);
    const uint64_t requested_packet_bytes = std::strtoull(argv[10], nullptr, 10);
    const uint32_t warmup = argc > 11 ? static_cast<uint32_t>(std::strtoul(argv[11], nullptr, 10)) : 2u;
    const uint32_t measure = argc > 12 ? static_cast<uint32_t>(std::strtoul(argv[12], nullptr, 10)) : 5u;
    // Zero requests a runtime-derived cohort.  This keeps the binary
    // portable across products with a different number of vector cores and
    // reserves at least half of the INC for a concurrent combine kernel.
    uint32_t inc_lanes = argc > 13 ? static_cast<uint32_t>(std::strtoul(argv[13], nullptr, 10)) : 0u;
    uint32_t upload_lanes = argc > 14 ? static_cast<uint32_t>(std::strtoul(argv[14], nullptr, 10)) : 0u;
    uint32_t tx_window = 8u;
    if (const char *raw = std::getenv("INC_STREAM_TX_WINDOW")) {
        tx_window = static_cast<uint32_t>(std::strtoul(raw, nullptr, 10));
    }
    uint32_t direct_dcci = 1u;
    if (const char *raw = std::getenv("INC_STREAM_DIRECT_DCCI")) {
        direct_dcci = static_cast<uint32_t>(std::strtoul(raw, nullptr, 10));
    }
    uint32_t tx_pingpong = 0u;
    if (const char *raw = std::getenv("INC_STREAM_TX_PINGPONG")) {
        tx_pingpong = static_cast<uint32_t>(std::strtoul(raw, nullptr, 10));
    }
    bool split_runs = true;
    if (const char *raw = std::getenv("INC_STREAM_SPLIT_RUNS")) {
        split_runs = std::strtoul(raw, nullptr, 10) != 0u;
    }
    uint32_t gather_chunk_routes = 16u;
    if (const char *raw = std::getenv("INC_STREAM_GATHER_CHUNK_ROUTES")) {
        gather_chunk_routes =
            static_cast<uint32_t>(std::strtoul(raw, nullptr, 10));
    }
    uint32_t requested_tx_lanes = 0u;
    if (const char *raw = std::getenv("INC_STREAM_TX_LANES")) {
        requested_tx_lanes =
            static_cast<uint32_t>(std::strtoul(raw, nullptr, 10));
    }
    uint32_t route_mode = 0u;
    if (const char *raw = std::getenv("INC_STREAM_ROUTE_MODE")) {
        route_mode =
            static_cast<uint32_t>(std::strtoul(raw, nullptr, 10));
    }
    if (workers == 0u ||
        workers > std::numeric_limits<uint32_t>::max() / 4u ||
        pe < 0 || pe >= npe ||
        tokens == 0u || hidden_bytes == 0u || topk == 0u || measure == 0u ||
        requested_tile_bytes < hidden_bytes ||
        requested_packet_bytes < hidden_bytes || tx_window == 0u ||
        tx_window > 64u || direct_dcci > 1u || tx_pingpong > 1u ||
        gather_chunk_routes == 0u ||
        !(route_mode == 0u || route_mode == 3u || route_mode == 4u ||
          (route_mode >= 5u && route_mode <= 8u))) {
        return 3;
    }
    inc::dc::IncDcPhysicalMapValidation physical_map{};
    std::string physical_map_error;
    if (!inc::dc::ValidatePhysicalNpuMap(
            npe, g_npus, f_npu, /*require_unique=*/true, &physical_map,
            &physical_map_error)) {
        std::cerr << "STREAM_PHYSICAL_MAP_FAIL pe=" << pe
                  << " reason=" << physical_map_error << std::endl;
        return 3;
    }

    const uint32_t task_route_capacity = static_cast<uint32_t>(
        std::max<uint64_t>(1u, requested_packet_bytes / hidden_bytes));
    const uint64_t max_packet_bytes =
        static_cast<uint64_t>(task_route_capacity) * hidden_bytes;
    const uint64_t input_stride = static_cast<uint64_t>(tokens) * hidden_bytes;
    const uint64_t input_bytes = static_cast<uint64_t>(workers) * input_stride;
    if (static_cast<uint64_t>(workers) * tokens >
        std::numeric_limits<uint64_t>::max() / topk / hidden_bytes) {
        return 3;
    }
    const uint64_t logical_output_bytes =
        static_cast<uint64_t>(workers) * tokens * topk * hidden_bytes;

    std::vector<HostTokenAssignment> token_plan;
    std::string plan_source;
    if (!BuildTokenPlan(workers, tokens, topk, route_mode, &token_plan,
                        &plan_source)) {
        std::cerr << "STREAM_DISPATCH_TOKEN_PLAN_FAIL pe=" << pe << std::endl;
        return 4;
    }
    // Count the actual rank-deduplicated rows in an arbitrary token plan.
    // This is O(assignments), including for large top-k, and lets transport
    // selection depend on work density instead of worker-count case tables.
    uint64_t estimated_physical_rows = 0u;
    std::vector<uint32_t> destination_epoch(workers, 0u);
    uint32_t epoch = 1u;
    for (uint64_t token = 0u;
         token < static_cast<uint64_t>(workers) * tokens; ++token) {
        if (epoch == 0u) {
            std::fill(destination_epoch.begin(), destination_epoch.end(), 0u);
            epoch = 1u;
        }
        for (uint32_t ordinal = 0u; ordinal < topk; ++ordinal) {
            const auto &assignment =
                token_plan[token * topk + ordinal];
            if (destination_epoch[assignment.destination_rank] != epoch) {
                destination_epoch[assignment.destination_rank] = epoch;
                ++estimated_physical_rows;
            }
        }
        ++epoch;
    }
    const uint64_t estimated_physical_bytes =
        estimated_physical_rows * hidden_bytes;
    const uint64_t logical_assignments =
        static_cast<uint64_t>(workers) * tokens * topk;
    const bool rank_multiplicity =
        estimated_physical_rows != 0u &&
        logical_assignments / estimated_physical_rows >= 2u;
    uint32_t planning_aiv = 0u;
    IncDcAivPolicy planning_policy{};
    if (!QueryLiveAivForPlanning(pe, &planning_aiv) ||
        !IncDcResolveAivPolicy(planning_aiv, workers, kStreamMaxLanes,
                               kStreamMaxLanes, &planning_policy)) {
        std::cerr << "STREAM_PLANNING_AIV_QUERY_FAIL pe=" << pe
                  << " workers=" << workers << std::endl;
        return 10;
    }
    const uint32_t planning_inc_lanes =
        inc_lanes == 0u ? planning_policy.dispatch_inc_aiv : inc_lanes;
    if (planning_inc_lanes == 0u ||
        planning_inc_lanes > planning_policy.dispatch_inc_aiv) {
        std::cerr << "STREAM_PLANNING_AIV_OVERRIDE_INVALID pe=" << pe
                  << " requested_inc_lanes=" << inc_lanes
                  << " qualified_limit="
                  << planning_policy.dispatch_inc_aiv << std::endl;
        return 3;
    }
    bool adaptive_tile = false;
    if (const char *raw = std::getenv("INC_STREAM_ADAPTIVE_TILE")) {
        adaptive_tile = raw[0] == '1' && raw[1] == '\0';
    }
    uint64_t selected_tile_bytes = requested_tile_bytes;
    if (adaptive_tile && rank_multiplicity &&
        estimated_physical_bytes > 64ull * 1024ull * 1024ull) {
        // A half-packet tile amortizes upload-ready synchronization while a
        // second tile remains available for the private-MTE pipeline.  The
        // value therefore scales with the caller's packet contract rather
        // than baking in a device- or rank-specific byte constant.
        selected_tile_bytes = std::max(
            selected_tile_bytes,
            std::max<uint64_t>(hidden_bytes, requested_packet_bytes / 2u));
    }
    if (adaptive_tile && estimated_physical_rows != 0u &&
        estimated_physical_bytes > 64ull * 1024ull * 1024ull) {
        // Keep the average packet produced for one destination at least
        // 64 KiB.  The density is derived from the compiled arbitrary plan:
        // average unique destinations per token = physical_rows/tokens.
        // This raises only sparse-fanout tiles (for example W8/K1) and scales
        // to other worker counts without a W/K lookup table.
        const long double token_count =
            static_cast<long double>(workers) * tokens;
        const long double density_tile =
            (64.0L * 1024.0L) * workers * token_count /
            static_cast<long double>(estimated_physical_rows);
        const uint64_t bounded_density_tile = static_cast<uint64_t>(
            std::min<long double>(requested_packet_bytes,
                                  std::ceil(density_tile)));
        selected_tile_bytes =
            std::max(selected_tile_bytes,
                     std::max<uint64_t>(hidden_bytes,
                                        bounded_density_tile));
    }
    if (adaptive_tile && estimated_physical_bytes >
                             64ull * 1024ull * 1024ull &&
        estimated_physical_rows ==
            static_cast<uint64_t>(workers) * workers * tokens) {
        // A full-rank fanout has one source->destination stream for every
        // pair.  The direct relay protocol has two bounded worker-upload
        // credits and two bounded INC-TX credits.  Keep exactly one resident
        // epoch for each state in that 2x2 credit product: fewer epochs cannot
        // fill both stages, while more epochs only repeat ready/drain control
        // without adding in-flight storage.  The byte size is derived from
        // the runtime input stride, not from W/K or a machine-specific table.
        constexpr uint64_t kUploadCredits = 2u;
        constexpr uint64_t kTxCredits = 2u;
        const uint64_t stream_count =
            static_cast<uint64_t>(workers) * workers;
        // When there are more TX lanes than source/destination streams, each
        // stream must expose multiple epochs or part of the fixed cohort has
        // no work.  Scale the resident window by that live lane cover.  This
        // keeps the rule portable to other AIV counts and non-power-of-two W.
        const uint64_t lane_cover = std::max<uint64_t>(
            1u, static_cast<uint64_t>(planning_inc_lanes) / stream_count);
        const uint64_t lane_grid = static_cast<uint64_t>(std::ceil(
            std::sqrt(static_cast<long double>(planning_inc_lanes))));
        // A worker count narrower than the TX lane grid has a proportionally
        // larger upload leg (D/R) and needs another source-fill wave before
        // the relay reaches steady state.  Wide fanout naturally resolves to
        // one; no scale-specific branch or byte constant is involved.
        const uint64_t source_fill = std::max<uint64_t>(
            1u, (lane_grid + workers - 1u) / workers);
        const uint64_t kResidentEpochs =
            kUploadCredits * kTxCredits * lane_cover * source_fill;
        uint64_t wave_tile = std::max<uint64_t>(
            hidden_bytes, input_stride / kResidentEpochs);
        // If the source/destination grid exactly occupies the live TX cohort,
        // every state in the upload/TX credit product needs one complete
        // transport packet before the first drain.  A shorter message would
        // otherwise create four tiny readiness epochs and spend most of its
        // time filling and closing the same bounded pipeline.  Platforms with
        // spare lane cover or a narrower source grid already get additional
        // epochs from lane_cover/source_fill and must not take this floor.
        if (lane_cover == 1u && source_fill == 1u) {
            wave_tile = std::max<uint64_t>(
                wave_tile, requested_packet_bytes * kResidentEpochs);
        }
        // Tile epochs and transport packets are independent: BuildTasks
        // still splits each destination slice at max_packet_bytes.  Capping
        // the epoch here at one packet would reintroduce the readiness
        // overhead this policy is intended to amortize.
        selected_tile_bytes = std::max(selected_tile_bytes, wave_tile);
    }
    if (split_runs && tokens != 0u) {
        // Each tile epoch ends with a full private-MTE drain on the INC, and
        // that serialization does not shrink with the epoch.  Finer epochs are
        // otherwise good — they overlap worker upload with INC forwarding — so
        // the policy only rules out epoch counts high enough for draining to
        // dominate, and leaves the tile alone below that.  Plans whose rows
        // reach few destinations are the ones that hit the cap, because their
        // per-epoch payload shrinks with the fanout.
        const uint32_t max_tile_epochs = 32u;
        const uint64_t epoch_capped_rows =
            (tokens + max_tile_epochs - 1u) / max_tile_epochs;
        selected_tile_bytes =
            std::max(selected_tile_bytes, epoch_capped_rows * hidden_bytes);
    }
    const uint32_t tile_rows = static_cast<uint32_t>(
        std::max<uint64_t>(1u, selected_tile_bytes / hidden_bytes));
    const uint64_t tile_bytes = static_cast<uint64_t>(tile_rows) * hidden_bytes;
    const uint32_t tiles = (tokens + tile_rows - 1u) / tile_rows;
    std::vector<StreamDispatchTask> tasks;
    std::vector<StreamRouteEntry> routes;
    std::vector<StreamExpertAssignment> expert_assignments;
    if (!BuildTasks(workers, tokens, topk, tile_rows, task_route_capacity,
                    hidden_bytes, token_plan, split_runs, &tasks, &routes,
                    &expert_assignments)) {
        return 4;
    }
    const uint64_t physical_output_bytes =
        static_cast<uint64_t>(routes.size()) * hidden_bytes;
    uint64_t gather_chunk_count64 = 0u;
    uint64_t generic_staging_bytes = 0u;
    uint32_t direct_task_count = 0u;
    uint32_t generic_task_count = 0u;
    // This is the single-INC operator: every legal shape uses the same
    // worker->INC->worker data path.  Worker-direct is deliberately
    // unreachable; neither shape heuristics nor environment variables may
    // change the transport topology.
    constexpr bool sparse_worker_direct = false;
    for (auto &task : tasks) {
        bool direct = !sparse_worker_direct && task.route_count > 0u;
        uint64_t first_linear = 0u;
        if (direct) {
            const auto &first = routes[task.route_begin];
            first_linear = static_cast<uint64_t>(first.source_rank) * tokens +
                           first.source_row;
            for (uint32_t i = 0u; i < task.route_count; ++i) {
                const auto &route = routes[task.route_begin + i];
                const uint64_t linear =
                    static_cast<uint64_t>(route.source_rank) * tokens +
                    route.source_row;
                if (linear != first_linear + i) {
                    direct = false;
                    break;
                }
            }
        }
        const uint32_t chunks = direct
            ? 0u
            : (task.route_count + gather_chunk_routes - 1u) /
                  gather_chunk_routes;
        if (gather_chunk_count64 + chunks >
            std::numeric_limits<uint32_t>::max()) {
            return 4;
        }
        task.reserved0 = static_cast<uint32_t>(gather_chunk_count64);
        task.reserved1[0] = chunks;
        task.reserved1[1] = direct
            ? first_linear * hidden_bytes + 1u
            : 0u;
        if (direct) {
            ++direct_task_count;
        } else {
            if (task.packet_bytes >
                std::numeric_limits<uint64_t>::max() -
                    generic_staging_bytes) {
                return 4;
            }
            // Store a one-based byte offset, not a task index multiplied by
            // max_packet_bytes.  Arbitrary top-k often creates many small
            // generic packets; fixed-stride staging can otherwise amplify a
            // few hundred MiB of useful data into a multi-GiB heap.
            task.reserved1[1] = generic_staging_bytes + 1u;
            generic_staging_bytes += task.packet_bytes;
            ++generic_task_count;
        }
        gather_chunk_count64 += chunks;
    }
    const uint32_t gather_chunk_count =
        static_cast<uint32_t>(gather_chunk_count64);

    uint64_t cursor = kStreamDataOff;
    StreamDispatchDesc desc{};
    desc.workers = workers;
    desc.hidden_bytes = hidden_bytes;
    desc.tokens_per_worker = tokens;
    desc.topk = topk;
    desc.tile_rows = tile_rows;
    desc.tiles_per_worker = tiles;
    desc.task_count = static_cast<uint32_t>(tasks.size());
    desc.route_count = static_cast<uint32_t>(routes.size());
    desc.task_route_capacity = task_route_capacity;
    desc.spin_cap = 2000000000u;
    desc.tx_window = tx_window;
    desc.gather_chunk_routes = gather_chunk_routes;
    desc.gather_chunk_count = gather_chunk_count;
    desc.direct_task_count = direct_task_count;
    // Keep the historically qualified per-packet DCCI as the default while
    // allowing a correctness-gated experiment for the direct non-L2 source
    // path.  Generic staging always retains its producer-side visibility
    // publication and is unaffected by this switch.
    desc.reserved32 = direct_dcci != 0u ? kStreamFlagDirectDcci : 0u;
    // Generic packets are packed by their owning worker and pushed into the
    // INC staging heap.  This distributes irregular gather work across the
    // workers and leaves the INC almost stateless: it only waits for a packet
    // generation and forwards the contiguous payload.  Direct packets retain
    // the existing zero-copy upload path.
    if (generic_task_count != 0u) {
        desc.reserved32 |= kStreamFlagWorkerPack;
    }
    if (sparse_worker_direct) {
        desc.reserved32 |= kStreamFlagWorkerDirect;
    }
    if (direct_task_count != 0u) {
        desc.reserved32 |= kStreamFlagHasDirect;
    }
    // The public MTE backend owns one UB/sync slot per AIV.  This applies to
    // both zero-copy and staged sources: multiple NBI puts from one AIV reuse
    // the slot and corrupt an earlier packet at vector-block boundaries.
    // Different TX AIVs remain fully concurrent, so this is a per-lane credit
    // rather than a global serialization point.  Direct packets reach deeper
    // pipelining through tx_pingpong, which owns a private buffer and sync id
    // per slot instead of sharing one.
    desc.tx_window = 1u;
    desc.tx_pingpong = tx_pingpong;
    desc.input_stride = input_stride;
    desc.tile_bytes = tile_bytes;
    desc.max_packet_bytes = max_packet_bytes;
    desc.logical_input_bytes = input_bytes;
    desc.logical_output_bytes = logical_output_bytes;
    desc.expert_assignment_count = expert_assignments.size();
    bool layout_ok =
        AddRegion(input_bytes, &cursor, &desc.input_off) &&
        AddRegion(static_cast<uint64_t>(workers) * tiles * 64u, &cursor,
                  &desc.tile_ready_off) &&
        AddRegion(static_cast<uint64_t>(workers) * tiles * 64u, &cursor,
                  &desc.direct_ready_off) &&
        AddRegion(tasks.size() * sizeof(StreamDispatchTask), &cursor,
                  &desc.task_off) &&
        AddRegion(routes.size() * sizeof(StreamRouteEntry), &cursor,
                  &desc.route_off) &&
        AddRegion(expert_assignments.size() *
                      sizeof(StreamExpertAssignment),
                  &cursor, &desc.expert_assignment_off) &&
        AddRegion(static_cast<uint64_t>(kStreamMaxLanes + 1u) *
                      sizeof(uint32_t),
                  &cursor, &desc.tx_lane_task_offsets_off) &&
        AddRegion(tasks.size() * sizeof(uint32_t), &cursor,
                  &desc.tx_lane_task_indices_off) &&
        AddRegion((static_cast<uint64_t>(workers) + 1u) * sizeof(uint32_t),
                  &cursor, &desc.worker_task_offsets_off) &&
        AddRegion(tasks.size() * sizeof(uint32_t), &cursor,
                  &desc.worker_task_indices_off) &&
        AddRegion(generic_staging_bytes, &cursor, &desc.staging_off) &&
        AddRegion(physical_output_bytes, &cursor, &desc.output_off) &&
        AddRegion(static_cast<uint64_t>(workers) * tiles *
                      kStreamMaxLanes * 64u,
                  &cursor, &desc.upload_chunk_done_off) &&
        AddRegion(static_cast<uint64_t>(workers + 1u) * kStreamMaxLanes * 64u,
                  &cursor, &desc.lane_done_off) &&
        AddRegion(static_cast<uint64_t>(workers + gather_chunk_count) * 64u,
                  &cursor, &desc.completion_off) &&
        AddRegion(static_cast<uint64_t>(workers * 4u + 1u) * 64u,
                  &cursor, &desc.start_gate_off) &&
        AddRegion(static_cast<uint64_t>(workers + 1u) * kStreamMaxLanes *
                      sizeof(StreamLaneStat),
                  &cursor, &desc.stats_off);
    if (!layout_ok) return 4;
    desc.total_bytes = StreamAlignUp(cursor, 4096u);
    constexpr uint64_t kDeviceLargePageBytes = 2ull * 1024ull * 1024ull;
    constexpr uint64_t kHeapHeadroomBytes = 64ull * 1024ull * 1024ull;
    if (desc.total_bytes >
        std::numeric_limits<uint64_t>::max() - kHeapHeadroomBytes) {
        return 4;
    }
    // ACLSHMEM's device-memory backend allocates the complete symmetric heap
    // in 2 MiB large pages.  Aligning only to the descriptor's 4 KiB ABI
    // boundary works accidentally while the 2 GiB floor is active, but any
    // larger shape whose layout is not itself 2 MiB aligned is rejected by
    // MemEntityDefault before HBM allocation.  Keep descriptor regions at
    // 4 KiB granularity and independently satisfy the allocator contract.
    const uint64_t heap_bytes = std::max<uint64_t>(
        2ull * 1024ull * 1024ull * 1024ull,
        StreamAlignUp(desc.total_bytes + kHeapHeadroomBytes,
                      kDeviceLargePageBytes));

    aclrtStream stream = nullptr;
    aclrtEvent ev0 = nullptr;
    aclrtEvent ev1 = nullptr;
    int32_t dev = -1;
    uint8_t *sym = nullptr;
    int st = InitShmem(pe, npe, heap_bytes, &dev, &stream);
    if (st != 0) {
        std::cerr << "STREAM_SHMEM_INIT_FAIL pe=" << pe
                  << " device=" << dev << " status=" << st << std::endl;
        return 10;
    }
    int64_t aiv_raw = 0;
    st = aclrtGetDeviceInfo(dev, ACL_DEV_ATTR_VECTOR_CORE_NUM, &aiv_raw);
    if (st != 0 || aiv_raw <= 0) {
        std::cerr << "STREAM_AIV_QUERY_FAIL pe=" << pe
                  << " device=" << dev << " status=" << st
                  << " vector_cores=" << aiv_raw << std::endl;
        Cleanup(dev, stream, sym);
        return 10;
    }
    const uint32_t aiv = static_cast<uint32_t>(aiv_raw);
    IncDcAivPolicy aiv_policy{};
    if (!IncDcResolveAivPolicy(aiv, workers, kStreamMaxLanes,
                               kStreamMaxLanes, &aiv_policy)) {
        std::cerr << "STREAM_AIV_POLICY_FAIL pe=" << pe
                  << " available_aiv=" << aiv
                  << " workers=" << workers << std::endl;
        Cleanup(dev, stream, sym);
        return 10;
    }
    if (inc_lanes == 0u) {
        inc_lanes = aiv_policy.dispatch_inc_aiv;
    } else if (inc_lanes > aiv_policy.dispatch_inc_aiv) {
        std::cerr << "STREAM_AIV_OVERRIDE_INVALID pe=" << pe
                  << " requested_inc_lanes=" << inc_lanes
                  << " qualified_limit="
                  << aiv_policy.dispatch_inc_aiv << std::endl;
        Cleanup(dev, stream, sym);
        return 3;
    }
    if (upload_lanes == 0u) {
        upload_lanes = aiv_policy.dispatch_worker_aiv;
    } else if (upload_lanes > aiv_policy.worker_half_limit) {
        std::cerr << "STREAM_WORKER_AIV_OVERRIDE_INVALID pe=" << pe
                  << " requested_upload_lanes=" << upload_lanes
                  << " worker_half_limit="
                  << aiv_policy.worker_half_limit << std::endl;
        Cleanup(dev, stream, sym);
        return 3;
    }
    upload_lanes =
        std::max(1u, std::min(upload_lanes, kStreamMaxLanes));
    // Keep the live hardware AIV budget.  A small number of large peer-owned
    // tasks still needs one TX lane per destination plus ordering/gather
    // lanes; capping by task_count+1 can otherwise collapse TX to one lane.
    // A single large tile must still use multiple worker AIVs.  Capping by
    // Do not cap by tile/row count. Idle lanes are still part of the fixed
    // hardware resource map and simply receive no rows for a tiny call.
    desc.lane_count = inc_lanes;
    // Keep one role map for every token plan.  Every selected INC lane issues
    // MTE traffic; the cohort grows from workload bytes up to the runtime
    // saturation cap.  Live roofline scans show that adding initiators beyond
    // one third of AIVs only contends for the same INC port/MTE resources.
    (void)requested_tx_lanes;
    desc.gather_lane_count = 0u;
    desc.tx_lane_count = std::max(
        1u, std::min(inc_lanes, aiv_policy.dispatch_inc_aiv));
    if (desc.tx_lane_count == 0u) {
        desc.tx_lane_count = 1u;
    }
    desc.upload_lane_count = upload_lanes;
    // The bounded two-credit uploader owns private UB/event slots on each
    // worker AIV; it does not consume an extra INC lane.  The previous strict
    // inequality unnecessarily disabled the protocol at an exact full-rank
    // lane cover (for example four sources x four destinations with 16 INC
    // lanes), forcing every tile through upload-drain-ready serialization.
    // Equality is safe: every source/destination stream still has one INC
    // lane and the two worker-local credits remain bounded and disjoint.
    desc.upload_pingpong =
        adaptive_tile && rank_multiplicity &&
                estimated_physical_bytes > 64ull * 1024ull * 1024ull &&
                static_cast<uint64_t>(workers) * workers <= inc_lanes
            ? 1u
            : 0u;
    desc.pe = static_cast<uint32_t>(pe);

    std::vector<uint32_t> tx_lane_task_offsets;
    std::vector<uint32_t> tx_lane_task_indices;
    std::vector<uint32_t> worker_task_offsets(workers + 1u, 0u);
    std::vector<uint32_t> worker_task_indices;
    if ((desc.reserved32 & kStreamFlagWorkerPack) != 0u) {
        worker_task_indices.reserve(tasks.size());
        for (uint32_t source = 0u; source < workers; ++source) {
            worker_task_offsets[source] =
                static_cast<uint32_t>(worker_task_indices.size());
            for (uint32_t task_index = 0u; task_index < tasks.size();
                 ++task_index) {
                if (tasks[task_index].source_rank == source &&
                    tasks[task_index].reserved1[0] != 0u) {
                    worker_task_indices.push_back(task_index);
                }
            }
        }
        worker_task_offsets[workers] =
            static_cast<uint32_t>(worker_task_indices.size());
    }
    if ((desc.reserved32 & kStreamFlagWorkerPack) == 0u) {
        if (!PartitionTasksByTxLane(workers, desc.tx_lane_count, &tasks,
                                    &tx_lane_task_offsets)) {
            Cleanup(dev, stream, sym);
            return 10;
        }
        desc.tx_lane_tasks_contiguous = 1u;
    } else {
        // Preserve tile/source/destination production order for distributed
        // packing.  Lane-major reordering delays the first packet of later TX
        // lanes until a worker has produced whole earlier lane ranges, which
        // drains the two-hop pipeline.  The deterministic device ownership
        // scan is cheap compared with that bubble and scales with metadata.
        if (!BuildTxLaneTaskWorklist(
                workers, desc.tx_lane_count, tasks,
                &tx_lane_task_offsets, &tx_lane_task_indices)) {
            Cleanup(dev, stream, sym);
            return 10;
        }
        desc.tx_lane_tasks_contiguous = 2u;
    }

    sym = static_cast<uint8_t *>(aclshmem_malloc(desc.total_bytes));
    if (sym == nullptr) {
        Cleanup(dev, stream, sym);
        return 11;
    }
    st = aclrtMemset(sym, desc.total_bytes, 0, desc.total_bytes);
    if (st == 0) {
        st = aclrtMemcpy(sym + kStreamDescOff, sizeof(desc), &desc, sizeof(desc),
                         ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (st == 0 && !tasks.empty()) {
        st = aclrtMemcpy(sym + desc.task_off,
                         tasks.size() * sizeof(StreamDispatchTask), tasks.data(),
                         tasks.size() * sizeof(StreamDispatchTask),
                         ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (st == 0 && !routes.empty()) {
        st = aclrtMemcpy(sym + desc.route_off,
                         routes.size() * sizeof(StreamRouteEntry), routes.data(),
                         routes.size() * sizeof(StreamRouteEntry),
                         ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (st == 0 && !expert_assignments.empty()) {
        st = aclrtMemcpy(
            sym + desc.expert_assignment_off,
            expert_assignments.size() * sizeof(StreamExpertAssignment),
            expert_assignments.data(),
            expert_assignments.size() * sizeof(StreamExpertAssignment),
            ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (st == 0 && !tx_lane_task_offsets.empty()) {
        st = aclrtMemcpy(sym + desc.tx_lane_task_offsets_off,
                         tx_lane_task_offsets.size() * sizeof(uint32_t),
                         tx_lane_task_offsets.data(),
                         tx_lane_task_offsets.size() * sizeof(uint32_t),
                         ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (st == 0 && !tx_lane_task_indices.empty()) {
        st = aclrtMemcpy(sym + desc.tx_lane_task_indices_off,
                         tx_lane_task_indices.size() * sizeof(uint32_t),
                         tx_lane_task_indices.data(),
                         tx_lane_task_indices.size() * sizeof(uint32_t),
                         ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (st == 0) {
        st = aclrtMemcpy(sym + desc.worker_task_offsets_off,
                         worker_task_offsets.size() * sizeof(uint32_t),
                         worker_task_offsets.data(),
                         worker_task_offsets.size() * sizeof(uint32_t),
                         ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (st == 0 && !worker_task_indices.empty()) {
        st = aclrtMemcpy(sym + desc.worker_task_indices_off,
                         worker_task_indices.size() * sizeof(uint32_t),
                         worker_task_indices.data(),
                         worker_task_indices.size() * sizeof(uint32_t),
                         ACL_MEMCPY_HOST_TO_DEVICE);
    }
    std::vector<uint8_t> input(input_stride);
    if (st == 0 && static_cast<uint32_t>(pe) < workers) {
        for (uint32_t row = 0u; row < tokens; ++row) {
            for (uint32_t b = 0u; b < hidden_bytes; ++b) {
                input[static_cast<uint64_t>(row) * hidden_bytes + b] =
                    PayloadByte(static_cast<uint32_t>(pe), row, b);
            }
        }
        st = CopyH2D(sym + desc.input_off +
                         static_cast<uint64_t>(pe) * input_stride,
                     input.data(), input_stride);
    }
    if (st != 0 || aclrtSynchronizeStream(stream) != 0) {
        Cleanup(dev, stream, sym);
        return 12;
    }

    std::cout << "STREAM_DISPATCH_CONFIG pe=" << pe
              << " physical_npu=" << dev
              << " role=" << (static_cast<uint32_t>(pe) == workers ? "inc" : "worker")
              << " workers=" << workers << " aiv=" << aiv
              << " platform_caps_version="
              << inc::dc::kIncDcPlatformCapabilityVersion
              << " resource_policy_version=" << aiv_policy.policy_version
              << " resource_policy_fingerprint="
              << IncDcResourcePolicyFingerprint(aiv_policy)
              << " dispatch_inc_capped="
              << aiv_policy.dispatch_inc_capped
              << " combine_inc_capped="
              << aiv_policy.combine_inc_capped
              << " physical_map_digest=" << physical_map.digest
              << " explicit_map_entries=" << physical_map.explicit_entries
              << " ub_bytes=" << inc::dc::kIncDcAivUbBudgetBytes
              << " private_mte_packet_bytes="
              << inc::dc::kIncDcPrivateMtePacketBytes
              << " inc_lanes=" << inc_lanes
              << " gather_lanes=" << desc.gather_lane_count
              << " tx_lanes=" << desc.tx_lane_count
              << " upload_lanes=" << upload_lanes
              << " upload_pingpong=" << desc.upload_pingpong
              << " tokens=" << tokens << " hidden_bytes=" << hidden_bytes
              << " topk=" << topk << " tile_rows=" << tile_rows
              << " route_mode=" << route_mode
              << " plan_source=" << plan_source
              << " tile_bytes=" << tile_bytes
              << " max_packet_bytes=" << max_packet_bytes
              << " tx_window=" << desc.tx_window
              << " direct_dcci="
              << ((desc.reserved32 & kStreamFlagDirectDcci) != 0u)
              << " worker_pack="
              << ((desc.reserved32 & kStreamFlagWorkerPack) != 0u)
              << " worker_direct="
              << ((desc.reserved32 & kStreamFlagWorkerDirect) != 0u)
              << " tx_pingpong=" << desc.tx_pingpong
              << " lane_major_tasks=" << desc.tx_lane_tasks_contiguous
              << " gather_chunk_routes=" << gather_chunk_routes
              << " gather_chunks=" << gather_chunk_count
              << " direct_tasks=" << direct_task_count
              << " generic_tasks=" << generic_task_count
              << " generic_staging_bytes=" << generic_staging_bytes
              << " tasks=" << tasks.size() << " routes=" << routes.size()
              << " expert_assignments=" << expert_assignments.size()
              << " input_bytes=" << input_bytes
              << " physical_output_bytes=" << physical_output_bytes
              << " logical_output_bytes=" << logical_output_bytes << std::endl;

    // Pin only the already-initialized submit thread when requested by a
    // qualification/framework launcher.  Pinning before ACL/SHMEM init would
    // also trap their helper threads on one CPU and create artificial tails.
    if (const char *raw = std::getenv("INC_STREAM_MAIN_CPU")) {
        char *end = nullptr;
        const long requested = std::strtol(raw, &end, 10);
        if (end == raw || *end != '\0' || requested < 0 ||
            requested >= CPU_SETSIZE) {
            std::cerr << "STREAM_DISPATCH_MAIN_CPU_ENV_FAIL pe=" << pe
                      << std::endl;
            Cleanup(dev, stream, sym);
            return 2;
        }
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(static_cast<int>(requested), &cpuset);
        if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
            std::cerr << "STREAM_DISPATCH_MAIN_CPU_PIN_FAIL pe=" << pe
                      << std::endl;
            Cleanup(dev, stream, sym);
            return 13;
        }
        std::cout << "STREAM_DISPATCH_MAIN_CPU_PIN pe=" << pe
                  << " cpu=" << requested << " late=1" << std::endl;
    }

    if (!inc::dc::IncDcExternalStartGate("dispatch", pe)) {
        std::cerr << "STREAM_DISPATCH_EXTERNAL_START_GATE_FAIL pe=" << pe
                  << std::endl;
        Cleanup(dev, stream, sym);
        return 13;
    }
    // The optional absolute deadline is shared with other independent SHMEM
    // sessions by the validation harness.  Session setup and process wakeup
    // are outside the operator interval; the existing device two-phase gate
    // still aligns all ranks within this dispatch session.
    aclshmem_barrier_all();
    const uint64_t external_start_ns = inc::dc::IncDcExternalStartNs();
    uint64_t device_start_target_cycle = 0u;
    if (external_start_ns != 0u) {
        auto host_now_ns = []() -> uint64_t {
            return static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count());
        };
        auto best_cycle_sample = [&](uint64_t *cycle, uint64_t *host_mid_ns,
                                     uint64_t *best_rtt_ns) -> bool {
            *best_rtt_ns = std::numeric_limits<uint64_t>::max();
            for (uint32_t sample = 0u; sample < 12u; ++sample) {
                const uint64_t before_ns = host_now_ns();
                launch_inc_dc_single_inc_stream_cycle_probe_kernel(
                    sym, desc.start_gate_off, stream);
                uint64_t observed_cycle = 0u;
                if (aclrtSynchronizeStream(stream) != ACL_SUCCESS ||
                    aclrtMemcpy(&observed_cycle, sizeof(observed_cycle),
                                sym + desc.start_gate_off,
                                sizeof(observed_cycle),
                                ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                    return false;
                }
                const uint64_t after_ns = host_now_ns();
                const uint64_t rtt_ns = after_ns - before_ns;
                if (observed_cycle != 0u && rtt_ns < *best_rtt_ns) {
                    *cycle = observed_cycle;
                    *host_mid_ns = before_ns + rtt_ns / 2u;
                    *best_rtt_ns = rtt_ns;
                }
            }
            return *best_rtt_ns != std::numeric_limits<uint64_t>::max();
        };
        uint64_t cycle1 = 0u, cycle2 = 0u, host1_ns = 0u, host2_ns = 0u;
        uint64_t rtt1_ns = 0u, rtt2_ns = 0u;
        if (!best_cycle_sample(&cycle1, &host1_ns, &rtt1_ns)) {
            Cleanup(dev, stream, sym);
            return 13;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (!best_cycle_sample(&cycle2, &host2_ns, &rtt2_ns) ||
            cycle2 <= cycle1 || host2_ns <= host1_ns) {
            Cleanup(dev, stream, sym);
            return 13;
        }
        const double cycles_per_ns =
            static_cast<double>(cycle2 - cycle1) /
            static_cast<double>(host2_ns - host1_ns);
        if (!std::isfinite(cycles_per_ns) || cycles_per_ns < 0.03 ||
            cycles_per_ns > 0.08 || external_start_ns <= host2_ns + 1000000u) {
            std::cerr << "STREAM_DISPATCH_CYCLE_CALIBRATION_FAIL pe=" << pe
                      << " cycles_per_ns=" << cycles_per_ns << std::endl;
            Cleanup(dev, stream, sym);
            return 13;
        }
        device_start_target_cycle =
            cycle2 + static_cast<uint64_t>(
                         static_cast<double>(external_start_ns - host2_ns) *
                         cycles_per_ns);
    }

    aclrtCreateEvent(&ev0);
    aclrtCreateEvent(&ev1);
    bool timing_ok = true;
    for (uint32_t epoch = 0u; epoch < warmup + measure; ++epoch) {
        desc.generation = epoch + 1u;
        desc.start_target_cycle = epoch == 0u ? device_start_target_cycle : 0u;
        const uint64_t tile_ready_bytes =
            static_cast<uint64_t>(workers) * tiles * 64u;
        const uint64_t lane_done_bytes =
            static_cast<uint64_t>(workers + 1u) * kStreamMaxLanes * 64u;
        const uint64_t upload_chunk_done_bytes =
            static_cast<uint64_t>(workers) * tiles * kStreamMaxLanes * 64u;
        const uint64_t completion_bytes =
            static_cast<uint64_t>(workers + gather_chunk_count) * 64u;
        const uint64_t stats_bytes =
            static_cast<uint64_t>(workers + 1u) * kStreamMaxLanes *
            sizeof(StreamLaneStat);
        const uint64_t start_gate_bytes =
            static_cast<uint64_t>(workers * 4u + 1u) * 64u;
        st = aclrtMemset(sym + desc.tile_ready_off, tile_ready_bytes, 0,
                         tile_ready_bytes);
        if (st == 0) {
            st = aclrtMemset(sym + desc.direct_ready_off, tile_ready_bytes, 0,
                             tile_ready_bytes);
        }
        if (st == 0) {
            st = aclrtMemset(sym + desc.upload_chunk_done_off,
                             upload_chunk_done_bytes, 0,
                             upload_chunk_done_bytes);
        }
        if (st == 0) {
            st = aclrtMemset(sym + desc.lane_done_off, lane_done_bytes, 0,
                             lane_done_bytes);
        }
        if (st == 0) {
            st = aclrtMemset(sym + desc.completion_off, completion_bytes, 0,
                             completion_bytes);
        }
        if (st == 0) {
            st = aclrtMemset(sym + desc.start_gate_off, start_gate_bytes, 0,
                             start_gate_bytes);
        }
        if (st == 0) {
            st = aclrtMemset(sym + desc.stats_off, stats_bytes, 0, stats_bytes);
        }
        if (st == 0) {
            st = aclrtMemcpy(sym + kStreamDescOff, sizeof(desc), &desc,
                             sizeof(desc), ACL_MEMCPY_HOST_TO_DEVICE);
        }
        aclrtSynchronizeStream(stream);
        aclshmem_barrier_all();
        aclrtRecordEvent(ev0, stream);
        const int blocks = static_cast<uint32_t>(pe) == workers
                               ? static_cast<int>(inc_lanes)
                               : static_cast<int>(upload_lanes);
        launch_inc_dc_single_inc_stream_dispatch_kernel(sym, blocks, stream);
        aclrtRecordEvent(ev1, stream);
        st = aclrtSynchronizeStream(stream);
        float ms = 0.0f;
        if (st == 0) st = aclrtEventElapsedTime(&ms, ev0, ev1);
        aclshmem_barrier_all();
        const uint32_t timing_lane =
            static_cast<uint32_t>(pe) == workers
                ? (desc.gather_lane_count == 0u &&
                           desc.lane_count > workers
                       ? workers
                       : desc.gather_lane_count)
                : 0u;
        StreamLaneStat protocol_stat{};
        if (st == 0) {
            st = aclrtMemcpy(
                &protocol_stat, sizeof(protocol_stat),
                sym + desc.stats_off +
                    (static_cast<uint64_t>(pe) * kStreamMaxLanes +
                     timing_lane) * sizeof(StreamLaneStat),
                sizeof(protocol_stat), ACL_MEMCPY_DEVICE_TO_HOST);
        }
        const bool protocol_timing_ok =
            protocol_stat.error == 0u &&
            protocol_stat.end_cycle > protocol_stat.start_cycle;
        const double protocol_us = protocol_timing_ok
            ? static_cast<double>(protocol_stat.end_cycle -
                                  protocol_stat.start_cycle) /
                  50.0
            : 0.0;
        if (st != 0 || ms <= 0.0f || !protocol_timing_ok) timing_ok = false;
        if (epoch >= warmup) {
            std::cout << "STREAM_DISPATCH_TIMING pe=" << pe
                      << " sample=" << (epoch - warmup)
                      << " start_cycle=" << protocol_stat.start_cycle
                      << " end_cycle=" << protocol_stat.end_cycle
                      << " rank_us=" << protocol_us
                      << " event_rank_us="
                      << static_cast<double>(ms) * 1000.0
                      << " physical_dispatch_bytes="
                      << physical_output_bytes
                      << " logical_dispatch_bytes="
                      << logical_output_bytes
                      // Backward-compatible field; never use it as a
                      // transport-bandwidth numerator.
                      << " global_dispatch_bytes=" << logical_output_bytes
                      << " logical_input_bytes=" << input_bytes
                      << " timing_source=device_cycle_after_device_two_phase_gate"
                      << " startup_in_timing=0 setup_in_timing=0 verify_in_timing=0"
                      << std::endl;
        }
    }

    bool verify = timing_ok;
    uint64_t verified_rows = 0u;
    uint64_t verified_assignments = 0u;
    uint64_t bad_task = std::numeric_limits<uint64_t>::max();
    uint32_t bad_route = std::numeric_limits<uint32_t>::max();
    uint32_t bad_byte = std::numeric_limits<uint32_t>::max();
    uint32_t bad_expected = 0u;
    uint32_t bad_actual = 0u;
    if (static_cast<uint32_t>(pe) < workers) {
        std::vector<uint8_t> packet(max_packet_bytes);
        for (size_t ti = 0u; ti < tasks.size() && verify; ++ti) {
            const auto &task = tasks[ti];
            if (task.destination_rank != static_cast<uint32_t>(pe)) continue;
            st = CopyD2H(packet.data(),
                         sym + desc.output_off + task.output_byte_offset,
                         task.packet_bytes);
            if (st != 0) {
                verify = false;
                bad_task = ti;
                break;
            }
            for (uint32_t i = 0u; i < task.route_count && verify; ++i) {
                const auto &route = routes[task.route_begin + i];
                if (route.assignment_count == 0u ||
                    static_cast<uint64_t>(route.assignment_begin) +
                            route.assignment_count >
                        expert_assignments.size()) {
                    verify = false;
                    bad_task = ti;
                    bad_route = i;
                    break;
                }
                const uint8_t *row = packet.data() +
                                     static_cast<uint64_t>(i) * hidden_bytes;
                for (uint32_t b = 0u; b < hidden_bytes; ++b) {
                    if (row[b] != PayloadByte(route.source_rank,
                                              route.source_row, b)) {
                        verify = false;
                        bad_task = ti;
                        bad_route = i;
                        bad_byte = b;
                        bad_expected = PayloadByte(route.source_rank,
                                                   route.source_row, b);
                        bad_actual = row[b];
                        break;
                    }
                }
                ++verified_rows;
                verified_assignments += route.assignment_count;
            }
        }
    }
    std::vector<StreamLaneStat> stats(kStreamMaxLanes);
    if (st == 0) {
        st = aclrtMemcpy(
            stats.data(), stats.size() * sizeof(StreamLaneStat),
            sym + desc.stats_off + static_cast<uint64_t>(pe) *
                                       kStreamMaxLanes * sizeof(StreamLaneStat),
            stats.size() * sizeof(StreamLaneStat), ACL_MEMCPY_DEVICE_TO_HOST);
    }
    const uint32_t active_lanes =
        static_cast<uint32_t>(pe) == workers ? inc_lanes : upload_lanes;
    uint64_t gather_cycles_sum = 0u;
    uint64_t transport_cycles_sum = 0u;
    uint64_t gather_cycles_max = 0u;
    uint64_t transport_cycles_max = 0u;
    uint64_t stat_input_bytes = 0u;
    uint64_t stat_output_bytes = 0u;
    uint32_t stat_tasks = 0u;
    for (uint32_t lane = 0u; lane < active_lanes; ++lane) {
        gather_cycles_sum += stats[lane].gather_cycles;
        transport_cycles_sum += stats[lane].transport_cycles;
        gather_cycles_max = std::max(gather_cycles_max, stats[lane].gather_cycles);
        transport_cycles_max =
            std::max(transport_cycles_max, stats[lane].transport_cycles);
        stat_input_bytes += stats[lane].input_bytes;
        stat_output_bytes += stats[lane].output_bytes;
        stat_tasks += stats[lane].tasks;
        if (stats[lane].error != 0u ||
            stats[lane].end_cycle <= stats[lane].start_cycle) {
            verify = false;
            std::cerr << "STREAM_DISPATCH_LANE_FAIL pe=" << pe
                      << " lane=" << lane
                      << " error=" << stats[lane].error
                      << " start=" << stats[lane].start_cycle
                      << " end=" << stats[lane].end_cycle << std::endl;
        }
    }
    std::cout << "STREAM_DISPATCH_STAGE_STATS pe=" << pe
              << " active_lanes=" << active_lanes
              << " tasks=" << stat_tasks
              << " input_bytes=" << stat_input_bytes
              << " output_bytes=" << stat_output_bytes
              << " gather_cycles_sum=" << gather_cycles_sum
              << " gather_cycles_max=" << gather_cycles_max
              << " transport_cycles_sum=" << transport_cycles_sum
              << " transport_cycles_max=" << transport_cycles_max
              << std::endl;
    const bool pass = verify && st == 0;
    std::cout << "STREAM_DISPATCH_RESULT pe=" << pe
              << " pass=" << (pass ? 1 : 0)
              << " verified_rows=" << verified_rows
              << " verified_assignments=" << verified_assignments
              << " bad_task=" << bad_task
              << " bad_route=" << bad_route
              << " bad_byte=" << bad_byte
              << " bad_expected=" << bad_expected
              << " bad_actual=" << bad_actual
              << " physical_dispatch_bytes=" << physical_output_bytes
              << " logical_dispatch_bytes=" << logical_output_bytes
              << " global_dispatch_bytes=" << logical_output_bytes
              << std::endl;

    aclrtDestroyEvent(ev0);
    aclrtDestroyEvent(ev1);
    Cleanup(dev, stream, sym);
    return pass ? 0 : 20;
}
