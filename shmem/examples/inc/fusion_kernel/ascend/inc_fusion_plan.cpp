#include "inc_fusion_plan.h"

#include <algorithm>
#include <limits>
#include <sstream>

namespace inc::fusion {
namespace {

constexpr uint64_t kAlignment = 512u;
constexpr uint32_t kPrivateMteBytes = 16u * 1024u;
constexpr uint32_t kComputeM = 128u;
// Two packets are insufficient once Dispatch payload and route metadata share
// a lane: the producer can wrap before the INC/worker round trip returns a
// credit and form a cyclic backpressure chain with Combine. Eight remains a
// bounded, token-count-independent window while covering the protocol RTT.
constexpr uint32_t kQueueDepth = 8u;

bool Fail(std::string *error, const char *message)
{
    if (error != nullptr) *error = message;
    return false;
}

bool Add(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr || a > std::numeric_limits<uint64_t>::max() - b)
        return false;
    *out = a + b;
    return true;
}

bool Mul(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr || (a != 0u && b > std::numeric_limits<uint64_t>::max() / a))
        return false;
    *out = a * b;
    return true;
}

bool Align(uint64_t value, uint64_t *out)
{
    uint64_t expanded = 0u;
    if (!Add(value, kAlignment - 1u, &expanded)) return false;
    *out = expanded / kAlignment * kAlignment;
    return true;
}

bool Append(uint64_t bytes, uint64_t *cursor, uint64_t *offset)
{
    if (!Align(*cursor, cursor)) return false;
    *offset = *cursor;
    return Add(*cursor, bytes, cursor);
}

uint64_t HashWord(uint64_t hash, uint32_t word)
{
    hash ^= word;
    return hash * 1099511628211ull;
}

FusionResourcePlan ResolveResources(uint32_t live_aiv, uint32_t live_aic)
{
    FusionResourcePlan out{};
    out.policy_version = 3u;
    out.live_aiv = live_aiv;
    out.live_aic = live_aic;
    // INC Dispatch is a pure fanout: it repeats every source payload once per
    // destination, so it moves W times the bytes that Combine returns, and a
    // single AIV reaches only a fraction of an HCCS link.  Two thirds of the
    // INC lanes therefore go to Dispatch, which also lets the kernel stripe
    // each (source, destination) pair across several lanes on a four-worker
    // topology.  Combine keeps a third, enough to stay off the critical path
    // because its reduction is overlapped with the next wave's fanout.
    out.inc_combine_aiv = std::max(1u, live_aiv / 3u);
    out.inc_dispatch_aiv = live_aiv > out.inc_combine_aiv
        ? live_aiv - out.inc_combine_aiv : 1u;
    // Dispatch is primarily an MTE fanout/pack stage, while dense Combine also
    // performs the local weighted reduction.  Give Combine twice Dispatch's
    // cohort and retain half of a 48-AIV worker for activation/scheduling:
    // 8 dispatch / 16 combine / 24 compute.  This remains shape-independent
    // and scales proportionally with the live-AIV hardware profile.
    out.worker_dispatch_aiv = std::max(1u, live_aiv / 6u);
    out.worker_combine_aiv = std::max(1u, live_aiv / 3u);
    const uint32_t communication =
        out.worker_dispatch_aiv + out.worker_combine_aiv;
    out.worker_compute_aiv = live_aiv > communication
        ? live_aiv - communication : 0u;
    uint64_t hash = 1469598103934665603ull;
    hash = HashWord(hash, out.policy_version);
    hash = HashWord(hash, out.live_aiv);
    hash = HashWord(hash, out.inc_dispatch_aiv);
    hash = HashWord(hash, out.inc_combine_aiv);
    hash = HashWord(hash, out.worker_dispatch_aiv);
    hash = HashWord(hash, out.worker_combine_aiv);
    hash = HashWord(hash, out.worker_compute_aiv);
    hash = HashWord(hash, out.live_aic);
    out.fingerprint = hash;
    return out;
}

} // namespace

bool BuildFusionPlan(const FusionPlanConfig &config,
                     const uint32_t *expert_owner,
                     const uint32_t *expert_local_index,
                     FusionPlan *plan, std::string *error)
{
    if (plan == nullptr || expert_owner == nullptr ||
        expert_local_index == nullptr)
        return Fail(error, "null plan or expert placement");
    if (config.live_aiv < 6u || config.live_aiv > kFusionMaxAiv ||
        config.live_aic == 0u ||
        config.worker_count < 2u || config.worker_count > kFusionMaxWorkers ||
        config.rank >= config.worker_count || config.inc_pe < config.worker_count)
        return Fail(error, "invalid hardware or rank configuration");
    if (config.hidden == 0u || config.intermediate == 0u ||
        config.expert_count == 0u || config.topk == 0u ||
        config.topk > config.expert_count || config.token_count == 0u ||
        config.tokens_per_wave == 0u || config.element_bytes != 2u)
        return Fail(error, "invalid MoE shape");
    if (config.slot_count < kFusionMinSlots || config.activation_waves == 0u ||
        config.service_ring_size < kFusionMinServiceRing ||
        config.service_ring_size > kFusionMaxServiceRing)
        return Fail(error, "invalid slot, activation-wave, or service-ring configuration");

    FusionPlan built{};
    built.config = config;
    built.resources = ResolveResources(config.live_aiv, config.live_aic);
    if (built.resources.worker_compute_aiv == 0u)
        return Fail(error, "no worker AIV remains for compute");

    built.expert_owner.assign(expert_owner, expert_owner + config.expert_count);
    built.expert_local_index.assign(
        expert_local_index, expert_local_index + config.expert_count);
    built.local_expert_counts.assign(config.worker_count, 0u);
    for (uint32_t expert = 0u; expert < config.expert_count; ++expert) {
        const uint32_t owner = built.expert_owner[expert];
        if (owner >= config.worker_count)
            return Fail(error, "expert owner is outside worker world");
        const uint32_t local = built.expert_local_index[expert];
        built.local_expert_counts[owner] =
            std::max(built.local_expert_counts[owner], local + 1u);
    }
    std::vector<uint8_t> seen(config.expert_count, 0u);
    for (uint32_t owner = 0u; owner < config.worker_count; ++owner) {
        std::fill(seen.begin(), seen.end(), 0u);
        for (uint32_t expert = 0u; expert < config.expert_count; ++expert) {
            if (built.expert_owner[expert] != owner) continue;
            const uint32_t local = built.expert_local_index[expert];
            if (local >= config.expert_count || seen[local] != 0u)
                return Fail(error, "expert local indices are not unique per owner");
            seen[local] = 1u;
        }
    }

    const uint32_t waves =
        (config.token_count + config.tokens_per_wave - 1u) /
        config.tokens_per_wave;
    built.waves.reserve(waves);
    for (uint32_t wave = 0u; wave < waves; ++wave) {
        FusionWaveDesc desc{};
        desc.generation = static_cast<uint64_t>(wave) + 1u;
        desc.token_begin = wave * config.tokens_per_wave;
        desc.token_count = std::min(config.tokens_per_wave,
                                    config.token_count - desc.token_begin);
        desc.slot = wave % config.slot_count;
        desc.activation_waves = config.activation_waves;
        built.waves.push_back(desc);
    }

    const uint64_t destinations_per_token =
        std::min(config.topk, config.worker_count);
    uint64_t source_rows64 = 0u;
    uint64_t source_assignments64 = 0u;
    uint64_t received_rows64 = 0u;
    uint64_t received_assignments64 = 0u;
    if (!Mul(config.tokens_per_wave, destinations_per_token, &source_rows64) ||
        !Mul(config.tokens_per_wave, config.topk, &source_assignments64) ||
        !Mul(config.tokens_per_wave, config.worker_count, &received_rows64) ||
        !Mul(source_assignments64, config.worker_count,
             &received_assignments64) ||
        source_rows64 > std::numeric_limits<uint32_t>::max() ||
        source_assignments64 > std::numeric_limits<uint32_t>::max() ||
        received_rows64 > std::numeric_limits<uint32_t>::max() ||
        received_assignments64 > std::numeric_limits<uint32_t>::max())
        return Fail(error, "wave route capacity overflow");
    built.max_source_dispatch_rows_per_wave =
        static_cast<uint32_t>(source_rows64);
    built.max_source_assignments_per_wave =
        static_cast<uint32_t>(source_assignments64);
    // A destination must tolerate the adversarial but valid route where every
    // source sends every top-k assignment to experts owned by that worker.
    // These are receive-side capacities, not the smaller per-source route
    // counts. Keeping this distinction prevents silent HBM overwrite while
    // preserving the fixed-depth transport queues.
    const uint64_t max_rows64 = std::max(source_rows64, received_rows64);
    const uint64_t max_assignments64 = received_assignments64;
    built.max_dispatch_rows_per_wave = static_cast<uint32_t>(max_rows64);
    built.max_assignments_per_wave = static_cast<uint32_t>(max_assignments64);

    uint64_t row_bytes = 0u;
    if (!Mul(config.hidden, config.element_bytes, &row_bytes))
        return Fail(error, "hidden row overflow");
    built.transport_tile_bytes = static_cast<uint32_t>(
        std::min<uint64_t>(kPrivateMteBytes,
                           std::max<uint64_t>(512u, row_bytes)));
    built.compute_tile_rows = kComputeM;

    uint64_t dispatch_slot = 0u;
    uint64_t assignment_slot = 0u;
    uint64_t output_slot = 0u;
    uint64_t gate_up_slot = 0u;
    uint64_t activation_slot = 0u;
    if (!Mul(max_rows64, row_bytes, &dispatch_slot) ||
        !Mul(max_assignments64, row_bytes, &assignment_slot) ||
        !Mul(config.tokens_per_wave, row_bytes, &output_slot) ||
        !Mul(max_assignments64,
             static_cast<uint64_t>(config.intermediate) * 2u * config.element_bytes,
             &gate_up_slot) ||
        !Mul(max_assignments64,
             static_cast<uint64_t>(config.intermediate) * config.element_bytes,
             &activation_slot))
        return Fail(error, "workspace shape overflow");

    FusionSymmetricLayout symmetric{};
    const uint32_t direct_dispatch_lanes = config.worker_count *
        std::max(1u, built.resources.worker_dispatch_aiv / 2u);
    const uint32_t direct_combine_lanes = config.worker_count *
        (built.resources.worker_combine_aiv > 1u
             ? built.resources.worker_combine_aiv -
                   built.resources.worker_combine_aiv / 2u
             : 0u);
    symmetric.queue_lanes = std::max(std::max(
        std::max(built.resources.worker_dispatch_aiv,
                 built.resources.worker_combine_aiv),
        std::max(built.resources.inc_dispatch_aiv,
                 built.resources.inc_combine_aiv)),
        std::max(direct_dispatch_lanes, direct_combine_lanes));
    symmetric.queue_depth = kQueueDepth;
    symmetric.packet_bytes = built.transport_tile_bytes;
    uint64_t queue_slots = 0u;
    uint64_t queue_header_bytes = 0u;
    uint64_t queue_payload_bytes = 0u;
    uint64_t symmetric_cursor = 0u;
    // Cross-process requests need the same ring isolation as command
    // descriptors. Reusing one packet header arena immediately on ticket+1
    // permits a fast request to race the previous request's final credit.
    if (!Mul(config.slot_count, config.service_ring_size, &queue_slots) ||
        !Mul(queue_slots, config.worker_count, &queue_slots) ||
        !Mul(queue_slots, symmetric.queue_lanes, &queue_slots) ||
        !Mul(queue_slots, symmetric.queue_depth, &queue_slots) ||
        !Mul(queue_slots, sizeof(FusionPacketHeader), &queue_header_bytes) ||
        !Mul(queue_slots, symmetric.packet_bytes, &queue_payload_bytes) ||
        !Append(queue_header_bytes, &symmetric_cursor,
                &symmetric.dispatch_header_off) ||
        !Append(queue_payload_bytes, &symmetric_cursor,
                &symmetric.dispatch_payload_off) ||
        !Append(queue_header_bytes, &symmetric_cursor,
                &symmetric.combine_header_off) ||
        !Append(queue_payload_bytes, &symmetric_cursor,
                &symmetric.combine_payload_off) ||
        !Append(queue_header_bytes, &symmetric_cursor,
                &symmetric.dispatch_result_header_off) ||
        !Append(queue_payload_bytes, &symmetric_cursor,
                &symmetric.dispatch_result_payload_off) ||
        !Append(queue_header_bytes, &symmetric_cursor,
                &symmetric.combine_result_header_off) ||
        !Append(queue_payload_bytes, &symmetric_cursor,
                &symmetric.combine_result_payload_off) ||
        !Mul(config.slot_count, config.worker_count,
             &queue_header_bytes) ||
        !Mul(queue_header_bytes, kFusionCacheLineBytes,
             &queue_header_bytes) ||
        !Append(queue_header_bytes, &symmetric_cursor,
                &symmetric.control_off) ||
        !Align(symmetric_cursor, &symmetric_cursor))
        return Fail(error, "symmetric queue size overflow");

    // Bulk single-INC transport arenas.  They are request-ring and slot
    // indexed, so an arbitrary number of tokens is handled by bounded waves
    // and a later persistent request cannot overwrite an earlier request.
    // Layout reserved words are ABI-stable extension points:
    //   [0] input/result arena, [1] route arena,
    //   [2] expert-output arena, [3] control arena.
    // A worker can enqueue later tickets before every peer has consumed the
    // previous ticket's result.  The persistent INC executes descriptors in
    // order, but that alone does not make peer-local payload/control arenas
    // safe to reuse.  Ring the bulk data plane by service ticket as well as
    // wave slot so asynchronous forwards cannot overwrite one another.
    uint64_t bulk_request_slots = 0u;
    uint64_t bulk_input_stride = 0u;
    uint64_t bulk_route_stride = 0u;
    uint64_t bulk_expert_stride = 0u;
    uint64_t bulk_control_lines = 0u;
    uint64_t bulk_bytes = 0u;
    uint64_t bulk_rows_bytes = 0u;
    uint64_t bulk_assignments_bytes = 0u;
    if (!Mul(config.slot_count, config.service_ring_size,
             &bulk_request_slots) ||
        !Mul(config.tokens_per_wave, row_bytes, &bulk_input_stride) ||
        !Mul(source_rows64, sizeof(FusionDispatchRow), &bulk_rows_bytes) ||
        !Mul(source_assignments64, sizeof(FusionExpertAssignment),
             &bulk_assignments_bytes) ||
        !Add(bulk_rows_bytes, bulk_assignments_bytes, &bulk_route_stride) ||
        !Align(bulk_route_stride, &bulk_route_stride) ||
        !Mul(max_assignments64,
             row_bytes + sizeof(FusionExpertAssignment),
             &bulk_expert_stride) ||
        !Mul(config.worker_count, config.worker_count, &bulk_control_lines) ||
        !Add(bulk_control_lines,
             static_cast<uint64_t>(4u) * config.worker_count,
             &bulk_control_lines) ||
        !Add(bulk_control_lines,
             static_cast<uint64_t>(2u) *
                 built.resources.inc_combine_aiv,
             &bulk_control_lines) ||
        !Add(bulk_control_lines,
             static_cast<uint64_t>(2u) * config.worker_count,
             &bulk_control_lines) ||
        !Add(bulk_control_lines,
             static_cast<uint64_t>(3u) * config.worker_count,
             &bulk_control_lines) ||
        !Add(bulk_control_lines,
             built.resources.inc_dispatch_aiv,
             &bulk_control_lines) ||
        !Add(bulk_control_lines, config.worker_count,
             &bulk_control_lines) ||
        !Mul(bulk_request_slots, config.worker_count, &bulk_bytes) ||
        !Mul(bulk_bytes, bulk_input_stride, &bulk_bytes) ||
        !Append(bulk_bytes, &symmetric_cursor, &symmetric.reserved64[0]) ||
        !Mul(bulk_request_slots, config.worker_count, &bulk_bytes) ||
        !Mul(bulk_bytes, bulk_route_stride, &bulk_bytes) ||
        !Append(bulk_bytes, &symmetric_cursor, &symmetric.reserved64[1]) ||
        !Mul(bulk_request_slots, config.worker_count, &bulk_bytes) ||
        !Mul(bulk_bytes, bulk_expert_stride, &bulk_bytes) ||
        !Append(bulk_bytes, &symmetric_cursor, &symmetric.reserved64[2]) ||
        !Mul(bulk_request_slots, bulk_control_lines, &bulk_bytes) ||
        !Mul(bulk_bytes, sizeof(FusionBulkControl), &bulk_bytes) ||
        !Append(bulk_bytes, &symmetric_cursor, &symmetric.reserved64[3]) ||
        !Align(symmetric_cursor, &symmetric_cursor))
        return Fail(error, "bulk transport arena size overflow");
    FusionRemoteServiceLayout remote{};
    remote.ring_size = config.service_ring_size;
    const uint64_t remote_begin = symmetric_cursor;
    uint64_t remote_bytes = 0u;
    uint64_t request_bytes = sizeof(FusionRemoteRequestHeader);
    uint64_t request_stride = 0u;
    uint64_t wave_bytes = 0u;
    uint64_t active_bytes = 0u;
    if (!Mul(waves, sizeof(FusionWaveDesc), &wave_bytes) ||
        !Mul(config.worker_count, sizeof(uint32_t), &active_bytes) ||
        !Add(request_bytes, wave_bytes, &request_bytes) ||
        !Add(request_bytes, active_bytes, &request_bytes) ||
        !Align(request_bytes, &request_stride) ||
        request_bytes > std::numeric_limits<uint32_t>::max() ||
        request_stride > std::numeric_limits<uint32_t>::max())
        return Fail(error, "remote request size overflow");
    remote.request_bytes = static_cast<uint32_t>(request_bytes);
    remote.request_stride = static_cast<uint32_t>(request_stride);
    if (!Append(sizeof(FusionServiceControl), &symmetric_cursor,
                &remote.control_off) ||
        !Mul(config.service_ring_size, sizeof(FusionServiceDescriptor),
             &remote_bytes) ||
        !Append(remote_bytes, &symmetric_cursor, &remote.descriptors_off) ||
        !Mul(config.service_ring_size, sizeof(FusionKernelArgs),
             &remote_bytes) ||
        !Append(remote_bytes, &symmetric_cursor, &remote.args_off) ||
        !Mul(config.service_ring_size, request_stride, &remote_bytes) ||
        !Append(remote_bytes, &symmetric_cursor, &remote.request_off) ||
        !Mul(config.worker_count, sizeof(uint32_t), &remote_bytes) ||
        !Append(remote_bytes, &symmetric_cursor, &remote.worker_pes_off) ||
        !Mul(config.service_ring_size, config.worker_count, &remote_bytes) ||
        !Mul(remote_bytes, kFusionCacheLineBytes, &remote_bytes) ||
        !Append(remote_bytes, &symmetric_cursor, &remote.worker_ready_off) ||
        !Mul(config.live_aiv, kFusionCacheLineBytes, &remote_bytes) ||
        !Append(remote_bytes, &symmetric_cursor, &remote.lane_progress_off) ||
        !Align(symmetric_cursor, &symmetric_cursor))
        return Fail(error, "remote service layout overflow");
    remote.waves_off = remote.request_off +
        sizeof(FusionRemoteRequestHeader);
    remote.active_token_counts_off = remote.waves_off + wave_bytes;
    remote.total_bytes = symmetric_cursor - remote_begin;
    symmetric.total_bytes = symmetric_cursor;

    FusionWorkspaceLayout layout{};
    layout.dispatch_slot_bytes = dispatch_slot;
    layout.assignment_slot_bytes = assignment_slot;
    layout.output_slot_bytes = output_slot;
    uint64_t cursor = 0u;
    uint64_t bytes = 0u;
    if (!Mul(config.slot_count, sizeof(FusionSlotState), &bytes) ||
        !Append(bytes, &cursor, &layout.slot_state_off) ||
        !Mul(config.slot_count, dispatch_slot, &bytes) ||
        !Append(bytes, &cursor, &layout.dispatch_ring_off) ||
        !Mul(config.slot_count, assignment_slot, &bytes) ||
        !Append(bytes, &cursor, &layout.grouped_input_off) ||
        !Mul(config.slot_count, gate_up_slot, &bytes) ||
        !Append(bytes, &cursor, &layout.gate_up_off) ||
        !Mul(config.slot_count, activation_slot, &bytes) ||
        !Append(bytes, &cursor, &layout.activation_off) ||
        !Mul(config.slot_count, assignment_slot, &bytes) ||
        !Append(bytes, &cursor, &layout.expert_output_off) ||
        !Mul(config.slot_count, assignment_slot, &bytes) ||
        !Append(bytes, &cursor, &layout.combine_ring_off) ||
        !Mul(config.slot_count,
             max_assignments64 * sizeof(FusionReceivedAssignment), &bytes) ||
        !Append(bytes, &cursor, &layout.assignment_meta_off) ||
        !Mul(config.slot_count, output_slot, &bytes) ||
        !Append(bytes, &cursor, &layout.final_output_off) ||
        !Mul(config.slot_count,
             static_cast<uint64_t>(config.expert_count) * sizeof(int64_t),
             &bytes) ||
        !Append(bytes, &cursor, &layout.expert_ready_off) ||
        !Mul(config.slot_count,
             static_cast<uint64_t>(config.expert_count) *
                 config.activation_waves * 5u *
                 std::max(config.live_aic,
                          built.resources.worker_compute_aiv) *
                 kFusionCacheLineBytes,
             &bytes) ||
        !Append(bytes, &cursor, &layout.tile_ready_off) ||
        !Mul(config.slot_count, 4096u, &bytes) ||
        !Append(bytes, &cursor, &layout.trace_off) || !Align(cursor, &cursor))
        return Fail(error, "workspace byte size overflow");
    layout.total_bytes = cursor;
    built.symmetric = symmetric;
    built.remote_service = remote;
    built.worker_workspace = layout;

    // INC does not allocate GMM scratch. It retains one accumulator and one
    // result row per source token, while packet queues bound transport memory.
    FusionWorkspaceLayout inc_layout{};
    uint64_t inc_cursor = 0u;
    uint64_t global_output_slot = 0u;
    if (!Mul(config.worker_count, output_slot, &global_output_slot) ||
        !Mul(config.slot_count, sizeof(FusionSlotState), &bytes) ||
        !Append(bytes, &inc_cursor, &inc_layout.slot_state_off) ||
        !Mul(config.slot_count, global_output_slot, &bytes) ||
        !Append(bytes, &inc_cursor, &inc_layout.combine_ring_off) ||
        !Mul(config.slot_count, global_output_slot, &bytes) ||
        !Append(bytes, &inc_cursor, &inc_layout.final_output_off) ||
        !Mul(config.slot_count, 4096u, &bytes) ||
        !Append(bytes, &inc_cursor, &inc_layout.trace_off) ||
        !Align(inc_cursor, &inc_cursor))
        return Fail(error, "INC workspace size overflow");
    inc_layout.output_slot_bytes = global_output_slot;
    inc_layout.total_bytes = inc_cursor;
    built.inc_workspace = inc_layout;
    *plan = std::move(built);
    if (error != nullptr) error->clear();
    return true;
}

bool ValidateFusionRoute(const FusionPlan &plan,
                         const FusionDispatchRow *rows, uint32_t row_count,
                         const FusionExpertAssignment *assignments,
                         uint32_t assignment_count, std::string *error)
{
    if ((row_count != 0u && rows == nullptr) ||
        (assignment_count != 0u && assignments == nullptr))
        return Fail(error, "null route array");
    const uint32_t wave_count = static_cast<uint32_t>(plan.waves.size());
    for (uint32_t row = 0u; row < row_count; ++row) {
        const FusionDispatchRow &entry = rows[row];
        if (entry.source_rank >= plan.config.worker_count ||
            entry.source_token >= plan.config.token_count ||
            entry.destination_rank >= plan.config.worker_count ||
            entry.wave >= wave_count ||
            entry.assignment_begin > assignment_count ||
            entry.assignment_count > assignment_count - entry.assignment_begin)
            return Fail(error, "invalid dispatch row");
        for (uint32_t i = 0u; i < entry.assignment_count; ++i) {
            const FusionExpertAssignment &assignment =
                assignments[entry.assignment_begin + i];
            if (assignment.dispatch_row != row ||
                assignment.expert_id >= plan.config.expert_count ||
                assignment.wave != entry.wave ||
                assignment.destination_row >=
                    plan.max_assignments_per_wave ||
                plan.expert_owner[assignment.expert_id] !=
                    entry.destination_rank ||
                plan.expert_local_index[assignment.expert_id] !=
                    assignment.local_expert ||
                assignment.route_ordinal >= plan.config.topk)
                return Fail(error, "dispatch/assignment mismatch");
        }
    }
    for (uint32_t i = 0u; i < assignment_count; ++i) {
        if (assignments[i].dispatch_row >= row_count ||
            assignments[i].destination_token >= plan.config.token_count)
            return Fail(error, "invalid combine assignment");
    }
    if (error != nullptr) error->clear();
    return true;
}

FusionKernelArgs MakeFusionKernelArgs(const FusionPlan &plan,
                                      uint32_t role,
                                      uint64_t generation)
{
    FusionKernelArgs args{};
    args.role = role;
    args.flags = kFusionPushOnly | kFusionDynamicRoute |
                 kFusionTileReadyCompute |
                 kFusionConcurrentDispatchCombine |
                 kFusionWeightedIncReduce |
                 kFusionBulkWaveTransport;
    args.rank = plan.config.rank;
    args.worker_count = plan.config.worker_count;
    args.inc_pe = plan.config.inc_pe;
    args.hidden = plan.config.hidden;
    args.intermediate = plan.config.intermediate;
    args.expert_count = plan.config.expert_count;
    args.local_expert_count = plan.local_expert_counts[plan.config.rank];
    args.topk = plan.config.topk;
    args.token_count = plan.config.token_count;
    args.tokens_per_wave = plan.config.tokens_per_wave;
    args.wave_count = static_cast<uint32_t>(plan.waves.size());
    args.slot_count = plan.config.slot_count;
    args.activation_waves = plan.config.activation_waves;
    args.transport_tile_bytes = plan.transport_tile_bytes;
    args.compute_tile_rows = plan.compute_tile_rows;
    args.ready_producers = std::max(
        plan.config.live_aic, plan.resources.worker_compute_aiv);
    args.spin_cap = plan.config.spin_cap;
    args.operation_generation = generation;
    args.resources = plan.resources;
    args.symmetric_layout = plan.symmetric;
    args.remote_service = plan.remote_service;
    args.layout = role == kFusionInc
        ? plan.inc_workspace : plan.worker_workspace;
    return args;
}

} // namespace inc::fusion
