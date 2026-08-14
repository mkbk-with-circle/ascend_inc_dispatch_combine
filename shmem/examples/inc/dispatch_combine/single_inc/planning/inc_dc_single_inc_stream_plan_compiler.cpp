#include "inc_dc_single_inc_stream_plan_compiler.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace inc::dc::single_stream {
namespace {

bool HeaderValid(const inc_dc_easy_token_plan_header_v1_t *header,
                 uint64_t bytes)
{
    if (header == nullptr || bytes < sizeof(*header) ||
        header->magic != INC_DC_EASY_TOKEN_PLAN_MAGIC ||
        header->abi_major != INC_DC_EASY_TOKEN_PLAN_ABI_VERSION ||
        header->header_bytes < sizeof(*header) ||
        header->assignment_bytes !=
            sizeof(inc_dc_easy_token_assignment_v1_t) ||
        header->total_bytes > bytes ||
        header->total_bytes < header->header_bytes ||
        header->tokens == 0u || header->topk == 0u ||
        header->worker_world_size == 0u ||
        header->tokens >
            std::numeric_limits<uint64_t>::max() / header->topk ||
        header->assignment_count != header->tokens * header->topk) {
        return false;
    }
    const uint64_t payload =
        header->assignment_count * header->assignment_bytes;
    return payload <= header->total_bytes - header->header_bytes;
}

uint32_t WeightBits(uint32_t bits) { return bits; }

bool AddRegion(uint64_t bytes, uint64_t *cursor, uint64_t *offset)
{
    if (cursor == nullptr || offset == nullptr) return false;
    *cursor = StreamAlignUp(*cursor, 4096u);
    *offset = *cursor;
    if (bytes > std::numeric_limits<uint64_t>::max() - *cursor) return false;
    *cursor += bytes;
    return true;
}

bool Multiply(uint64_t a, uint64_t b, uint64_t *result)
{
    if (result == nullptr ||
        (a != 0u && b > std::numeric_limits<uint64_t>::max() / a)) {
        return false;
    }
    *result = a * b;
    return true;
}

} // namespace

bool CompileStreamSourcePlan(
    const StreamPlanCompileInput &input,
    StreamCompiledSourcePlan *compiled)
{
    if (compiled == nullptr || input.host_token_plan == nullptr ||
        input.worker_world_size == 0u ||
        input.source_rank >= input.worker_world_size ||
        input.hidden_bytes == 0u || input.tile_rows == 0u ||
        input.max_routes_per_packet == 0u) {
        return false;
    }
    const auto *header =
        static_cast<const inc_dc_easy_token_plan_header_v1_t *>(
            input.host_token_plan);
    if (!HeaderValid(header, input.host_token_plan_bytes) ||
        header->worker_world_size != input.worker_world_size ||
        header->tokens > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    const auto *wire = reinterpret_cast<
        const inc_dc_easy_token_assignment_v1_t *>(
            static_cast<const uint8_t *>(input.host_token_plan) +
            header->header_bytes);

    StreamCompiledSourcePlan out{};
    out.source_rank = input.source_rank;
    out.worker_world_size = input.worker_world_size;
    out.tokens = static_cast<uint32_t>(header->tokens);
    out.topk = header->topk;
    out.hidden_bytes = input.hidden_bytes;
    out.semantic_digest = header->semantic_digest;
    out.generation = header->generation;
    out.logical_assignments = header->assignment_count;

    uint64_t output_cursor = 0u;
    const uint32_t tile_count =
        (out.tokens + input.tile_rows - 1u) / input.tile_rows;
    for (uint32_t tile = 0u; tile < tile_count; ++tile) {
        const uint32_t row_begin = tile * input.tile_rows;
        const uint32_t row_end =
            std::min(out.tokens, row_begin + input.tile_rows);
        for (uint32_t destination = 0u;
             destination < input.worker_world_size; ++destination) {
            std::vector<StreamRouteEntry> packet;
            for (uint32_t row = row_begin; row < row_end; ++row) {
                if (out.assignments.size() >
                    std::numeric_limits<uint32_t>::max()) {
                    return false;
                }
                const uint32_t assignment_begin =
                    static_cast<uint32_t>(out.assignments.size());
                for (uint32_t ordinal = 0u; ordinal < out.topk; ++ordinal) {
                    const uint64_t index =
                        static_cast<uint64_t>(row) * out.topk + ordinal;
                    const auto &assignment = wire[index];
                    if (assignment.expert_id < 0 ||
                        assignment.destination_rank >=
                        input.worker_world_size) {
                        return false;
                    }
                    if (assignment.destination_rank == destination) {
                        out.assignments.push_back(StreamExpertAssignment{
                            static_cast<uint32_t>(assignment.expert_id),
                            ordinal, WeightBits(assignment.weight_bits), 0u});
                    }
                }
                const uint32_t assignment_count =
                    static_cast<uint32_t>(out.assignments.size()) -
                    assignment_begin;
                if (assignment_count != 0u) {
                    packet.push_back(StreamRouteEntry{
                        input.source_rank, row, assignment_begin,
                        assignment_count});
                }
            }
            size_t begin = 0u;
            while (begin < packet.size()) {
                size_t run_end = begin + 1u;
                while (run_end < packet.size() &&
                       packet[run_end].source_row ==
                           packet[run_end - 1u].source_row + 1u &&
                       run_end - begin < input.max_routes_per_packet) {
                    ++run_end;
                }
                const uint32_t count =
                    static_cast<uint32_t>(run_end - begin);
                if (out.routes.size() + count >
                        std::numeric_limits<uint32_t>::max() ||
                    out.tasks.size() ==
                        std::numeric_limits<uint32_t>::max() ||
                    static_cast<uint64_t>(count) * input.hidden_bytes >
                        std::numeric_limits<uint64_t>::max() -
                            output_cursor) {
                    return false;
                }
                StreamDispatchTask task{};
                task.source_rank = input.source_rank;
                task.destination_rank = destination;
                task.source_tile = tile;
                task.route_begin =
                    static_cast<uint32_t>(out.routes.size());
                task.route_count = count;
                task.output_byte_offset = output_cursor;
                task.packet_bytes =
                    static_cast<uint64_t>(count) * input.hidden_bytes;
                out.tasks.push_back(task);
                out.routes.insert(
                    out.routes.end(), packet.begin() + begin,
                    packet.begin() + run_end);
                output_cursor += task.packet_bytes;
                begin = run_end;
            }
        }
    }
    out.physical_rows = out.routes.size();
    out.physical_output_bytes = output_cursor;
    if (out.assignments.size() != out.logical_assignments ||
        out.physical_rows != header->global_physical_rows) {
        return false;
    }
    *compiled = std::move(out);
    return true;
}

bool MergeStreamSourcePlans(
    const std::vector<StreamCompiledSourcePlan> &sources,
    StreamCompiledGlobalPlan *compiled)
{
    if (compiled == nullptr || sources.empty() ||
        sources.size() > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    const uint32_t world = static_cast<uint32_t>(sources.size());
    std::vector<const StreamCompiledSourcePlan *> by_rank(world, nullptr);
    const auto &first = sources.front();
    if (first.worker_world_size != world || first.tokens == 0u ||
        first.topk == 0u || first.hidden_bytes == 0u ||
        first.generation == 0u) {
        return false;
    }
    for (const auto &source : sources) {
        if (source.worker_world_size != world || source.source_rank >= world ||
            by_rank[source.source_rank] != nullptr ||
            source.tokens != first.tokens || source.topk != first.topk ||
            source.hidden_bytes != first.hidden_bytes ||
            source.generation != first.generation ||
            source.routes.size() != source.physical_rows ||
            source.assignments.size() != source.logical_assignments) {
            return false;
        }
        by_rank[source.source_rank] = &source;
    }

    struct TaskRef {
        uint32_t source_rank;
        const StreamCompiledSourcePlan *source;
        const StreamDispatchTask *task;
    };
    std::vector<TaskRef> order;
    for (uint32_t source_rank = 0u; source_rank < world; ++source_rank) {
        if (by_rank[source_rank] == nullptr) return false;
        for (const auto &task : by_rank[source_rank]->tasks) {
            if (task.source_rank != source_rank ||
                task.route_begin > by_rank[source_rank]->routes.size() ||
                task.route_count >
                    by_rank[source_rank]->routes.size() - task.route_begin) {
                return false;
            }
            order.push_back(TaskRef{source_rank, by_rank[source_rank], &task});
        }
    }
    std::stable_sort(order.begin(), order.end(),
        [](const TaskRef &a, const TaskRef &b) {
            if (a.task->source_tile != b.task->source_tile)
                return a.task->source_tile < b.task->source_tile;
            if (a.source_rank != b.source_rank)
                return a.source_rank < b.source_rank;
            return a.task->destination_rank < b.task->destination_rank;
        });

    StreamCompiledGlobalPlan out{};
    out.worker_world_size = world;
    out.tokens_per_worker = first.tokens;
    out.topk = first.topk;
    out.hidden_bytes = first.hidden_bytes;
    out.generation = first.generation;
    out.semantic_digest = 0x5354524d504c414eull; // STRMPLAN
    out.source_semantic_digests.assign(world, 0u);
    uint64_t output_cursor = 0u;
    for (const TaskRef &ref : order) {
        StreamDispatchTask task = *ref.task;
        if (out.routes.size() > std::numeric_limits<uint32_t>::max())
            return false;
        task.route_begin = static_cast<uint32_t>(out.routes.size());
        task.output_byte_offset = output_cursor;
        for (uint32_t i = 0u; i < ref.task->route_count; ++i) {
            StreamRouteEntry route =
                ref.source->routes[ref.task->route_begin + i];
            if (route.assignment_begin > ref.source->assignments.size() ||
                route.assignment_count >
                    ref.source->assignments.size() - route.assignment_begin ||
                out.assignments.size() >
                    std::numeric_limits<uint32_t>::max()) {
                return false;
            }
            const uint32_t assignment_begin =
                static_cast<uint32_t>(out.assignments.size());
            out.assignments.insert(
                out.assignments.end(),
                ref.source->assignments.begin() + route.assignment_begin,
                ref.source->assignments.begin() + route.assignment_begin +
                    route.assignment_count);
            route.assignment_begin = assignment_begin;
            out.routes.push_back(route);
        }
        if (task.packet_bytes >
            std::numeric_limits<uint64_t>::max() - output_cursor) {
            return false;
        }
        output_cursor += task.packet_bytes;
        out.tasks.push_back(task);
    }
    for (const auto &source : sources) {
        if (source.logical_assignments >
                std::numeric_limits<uint64_t>::max() -
                    out.logical_assignments ||
            source.physical_rows >
                std::numeric_limits<uint64_t>::max() - out.physical_rows) {
            return false;
        }
        out.logical_assignments += source.logical_assignments;
        out.physical_rows += source.physical_rows;
        out.semantic_digest ^= source.semantic_digest +
            0x9e3779b97f4a7c15ull + (out.semantic_digest << 6u) +
            (out.semantic_digest >> 2u);
        out.source_semantic_digests[source.source_rank] =
            source.semantic_digest;
    }
    out.physical_output_bytes = output_cursor;
    out.destination_output_offsets.assign(world, 0u);
    out.destination_physical_rows.assign(world, 0u);
    std::vector<uint64_t> destination_bytes(world, 0u);
    for (const auto &task : out.tasks) {
        if (task.destination_rank >= world ||
            task.packet_bytes >
                std::numeric_limits<uint64_t>::max() -
                    destination_bytes[task.destination_rank]) {
            return false;
        }
        destination_bytes[task.destination_rank] += task.packet_bytes;
    }
    uint64_t destination_cursor = 0u;
    for (uint32_t destination = 0u; destination < world; ++destination) {
        out.destination_output_offsets[destination] = destination_cursor;
        out.destination_physical_rows[destination] =
            destination_bytes[destination] / out.hidden_bytes;
        destination_cursor += destination_bytes[destination];
    }
    std::vector<uint64_t> next_output = out.destination_output_offsets;
    for (auto &task : out.tasks) {
        task.output_byte_offset = next_output[task.destination_rank];
        next_output[task.destination_rank] += task.packet_bytes;
    }
    uint64_t expected_physical_bytes = 0u;
    if (!Multiply(out.physical_rows, out.hidden_bytes,
                  &expected_physical_bytes) ||
        out.routes.size() != out.physical_rows ||
        out.assignments.size() != out.logical_assignments ||
        out.physical_output_bytes != expected_physical_bytes) {
        return false;
    }
    if (out.semantic_digest == 0u) out.semantic_digest = 1u;
    *compiled = std::move(out);
    return true;
}

bool BuildStreamPreparedWorkspace(
    const StreamCompiledGlobalPlan &plan,
    const StreamWorkspaceBuildInput &input,
    StreamPreparedWorkspace *workspace)
{
    if (workspace == nullptr || plan.worker_world_size == 0u ||
        plan.tokens_per_worker == 0u || plan.topk == 0u ||
        plan.hidden_bytes == 0u || input.live_aiv == 0u ||
        input.tile_rows == 0u || input.direct_dcci > 1u ||
        input.tx_pingpong > 1u ||
        plan.tasks.size() > std::numeric_limits<uint32_t>::max() ||
        plan.routes.size() > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    StreamPreparedWorkspace out{};
    if (!IncDcResolveAivPolicy(
            input.live_aiv, plan.worker_world_size, kStreamMaxLanes,
            kStreamMaxLanes, &out.resources)) {
        return false;
    }
    out.tasks = plan.tasks;
    out.routes = plan.routes;
    out.assignments = plan.assignments;
    out.destination_output_offsets = plan.destination_output_offsets;
    out.destination_physical_rows = plan.destination_physical_rows;
    out.source_semantic_digests = plan.source_semantic_digests;
    const uint32_t tiles =
        (plan.tokens_per_worker + input.tile_rows - 1u) / input.tile_rows;

    uint64_t input_stride = 0u;
    uint64_t input_bytes = 0u;
    uint64_t logical_output_rows = 0u;
    uint64_t logical_output_bytes = 0u;
    if (!Multiply(plan.tokens_per_worker, plan.hidden_bytes, &input_stride) ||
        !Multiply(plan.worker_world_size, input_stride, &input_bytes) ||
        !Multiply(plan.worker_world_size, plan.tokens_per_worker,
                  &logical_output_rows) ||
        !Multiply(logical_output_rows, plan.topk, &logical_output_rows) ||
        !Multiply(logical_output_rows, plan.hidden_bytes,
                  &logical_output_bytes)) {
        return false;
    }
    uint64_t max_packet_bytes = 0u;
    for (auto &task : out.tasks) {
        if (task.route_count == 0u ||
            task.route_begin > out.routes.size() ||
            task.route_count > out.routes.size() - task.route_begin) {
            return false;
        }
        const auto &first = out.routes[task.route_begin];
        const uint64_t first_linear =
            static_cast<uint64_t>(first.source_rank) *
                plan.tokens_per_worker + first.source_row;
        for (uint32_t i = 0u; i < task.route_count; ++i) {
            const auto &route = out.routes[task.route_begin + i];
            const uint64_t linear =
                static_cast<uint64_t>(route.source_rank) *
                    plan.tokens_per_worker + route.source_row;
            if (route.source_rank != task.source_rank ||
                linear != first_linear + i) {
                return false;
            }
        }
        if (first_linear >
            (std::numeric_limits<uint64_t>::max() - 1u) /
                plan.hidden_bytes) {
            return false;
        }
        task.reserved0 = 0u;
        task.reserved1[0] = 0u;
        task.reserved1[1] = first_linear * plan.hidden_bytes + 1u;
        max_packet_bytes = std::max(max_packet_bytes, task.packet_bytes);
    }

    const uint32_t tx_lanes = out.resources.dispatch_inc_aiv;
    std::vector<std::vector<uint32_t>> per_lane(tx_lanes);
    for (uint32_t i = 0u; i < out.tasks.size(); ++i) {
        per_lane[i % tx_lanes].push_back(i);
    }
    out.tx_lane_task_offsets.assign(tx_lanes + 1u, 0u);
    out.tx_lane_task_indices.clear();
    out.tx_lane_task_indices.reserve(out.tasks.size());
    for (uint32_t lane = 0u; lane < tx_lanes; ++lane) {
        out.tx_lane_task_offsets[lane] =
            static_cast<uint32_t>(out.tx_lane_task_indices.size());
        out.tx_lane_task_indices.insert(out.tx_lane_task_indices.end(),
                                        per_lane[lane].begin(),
                                        per_lane[lane].end());
    }
    out.tx_lane_task_offsets[tx_lanes] =
        static_cast<uint32_t>(out.tx_lane_task_indices.size());
    out.worker_task_offsets.assign(plan.worker_world_size + 1u, 0u);

    StreamDispatchDesc desc{};
    desc.workers = plan.worker_world_size;
    desc.lane_count = out.resources.dispatch_inc_aiv;
    desc.gather_lane_count = 0u;
    desc.tx_lane_count = tx_lanes;
    desc.upload_lane_count = out.resources.dispatch_worker_aiv;
    desc.hidden_bytes = plan.hidden_bytes;
    desc.tokens_per_worker = plan.tokens_per_worker;
    desc.topk = plan.topk;
    desc.tile_rows = input.tile_rows;
    desc.tiles_per_worker = tiles;
    desc.task_count = static_cast<uint32_t>(out.tasks.size());
    desc.route_count = static_cast<uint32_t>(out.routes.size());
    desc.task_route_capacity = static_cast<uint32_t>(
        max_packet_bytes / plan.hidden_bytes);
    desc.spin_cap = 2000000000u;
    desc.tx_window = 1u;
    desc.gather_chunk_routes = 1u;
    desc.gather_chunk_count = 0u;
    desc.direct_task_count = desc.task_count;
    desc.reserved32 = kStreamFlagHasDirect;
    if (input.direct_dcci != 0u) desc.reserved32 |= kStreamFlagDirectDcci;
    desc.tx_pingpong = input.tx_pingpong;
    desc.upload_pingpong = 0u;
    desc.input_stride = input_stride;
    desc.tile_bytes =
        static_cast<uint64_t>(input.tile_rows) * plan.hidden_bytes;
    desc.max_packet_bytes = max_packet_bytes;
    desc.logical_input_bytes = input_bytes;
    desc.logical_output_bytes = logical_output_bytes;
    desc.expert_assignment_count = out.assignments.size();
    // Keep canonical tile/source/destination order in the task table and use
    // an index worklist for lane ownership.  Reordering the table lane-major
    // delays later TX lanes until whole earlier ranges are produced and
    // drains the two-hop pipeline.
    desc.tx_lane_tasks_contiguous = 2u;

    uint64_t cursor = kStreamDataOff;
    uint64_t scratch = 0u;
    bool ok =
        AddRegion(input_bytes, &cursor, &desc.input_off) &&
        AddRegion(static_cast<uint64_t>(plan.worker_world_size) * tiles * 64u,
                  &cursor, &desc.tile_ready_off) &&
        AddRegion(static_cast<uint64_t>(plan.worker_world_size) * tiles * 64u,
                  &cursor, &desc.direct_ready_off) &&
        AddRegion(out.tasks.size() * sizeof(StreamDispatchTask),
                  &cursor, &desc.task_off) &&
        AddRegion(out.routes.size() * sizeof(StreamRouteEntry),
                  &cursor, &desc.route_off) &&
        AddRegion(out.assignments.size() * sizeof(StreamExpertAssignment),
                  &cursor, &desc.expert_assignment_off) &&
        AddRegion(static_cast<uint64_t>(kStreamMaxLanes + 1u) *
                      sizeof(uint32_t),
                  &cursor, &desc.tx_lane_task_offsets_off) &&
        AddRegion(out.tasks.size() * sizeof(uint32_t),
                  &cursor, &desc.tx_lane_task_indices_off) &&
        AddRegion(static_cast<uint64_t>(plan.worker_world_size + 1u) *
                      sizeof(uint32_t),
                  &cursor, &desc.worker_task_offsets_off) &&
        AddRegion(out.tasks.size() * sizeof(uint32_t),
                  &cursor, &desc.worker_task_indices_off) &&
        AddRegion(0u, &cursor, &desc.staging_off) &&
        AddRegion(plan.physical_output_bytes, &cursor, &desc.output_off) &&
        AddRegion(static_cast<uint64_t>(plan.worker_world_size) * tiles *
                      kStreamMaxLanes * 64u,
                  &cursor, &desc.upload_chunk_done_off) &&
        AddRegion(static_cast<uint64_t>(plan.worker_world_size + 1u) *
                      kStreamMaxLanes * 64u,
                  &cursor, &desc.lane_done_off) &&
        AddRegion(static_cast<uint64_t>(plan.worker_world_size) * 64u,
                  &cursor, &desc.completion_off) &&
        AddRegion((static_cast<uint64_t>(plan.worker_world_size) * 4u + 1u) *
                      64u,
                  &cursor, &desc.start_gate_off) &&
        AddRegion(static_cast<uint64_t>(plan.worker_world_size + 1u) *
                      kStreamMaxLanes * sizeof(StreamLaneStat),
                  &cursor, &desc.stats_off);
    (void)scratch;
    if (!ok) return false;
    desc.total_bytes = StreamAlignUp(cursor, 4096u);
    out.descriptor = desc;
    *workspace = std::move(out);
    return true;
}

} // namespace inc::dc::single_stream
