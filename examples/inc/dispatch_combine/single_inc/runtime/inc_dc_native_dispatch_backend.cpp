#include "inc_dc_native_dispatch_backend.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <new>
#include <thread>

#include "acl/acl.h"
#include "shmem.h"

namespace inc::dc::single_stream {

extern "C" void launch_inc_dc_single_inc_stream_dispatch_kernel(
    uint8_t *sym, int block_dim, void *stream);

namespace {

struct NativeTicket {
    bool live = false;
    uint64_t generation = 0u;
    uint32_t state = INC_DC_FW_REQUEST_FREE;
    aclrtEvent event = nullptr;
    StreamDispatchDesc descriptor{};
};

bool CopyH2D(void *destination, uint64_t capacity, const void *source,
             uint64_t bytes)
{
    return bytes == 0u ||
           aclrtMemcpy(destination, capacity, source, bytes,
                       ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

bool Contiguous2d(const inc_dc_fw_tensor_desc_t &tensor, uint64_t rows,
                  uint32_t hidden)
{
    return tensor.data != nullptr && tensor.ndim == 2u &&
           tensor.dims[0] >= 0 &&
           static_cast<uint64_t>(tensor.dims[0]) >= rows &&
           tensor.dims[1] == static_cast<int64_t>(hidden) &&
           tensor.strides[0] == static_cast<int64_t>(hidden) &&
           tensor.strides[1] == 1;
}

} // namespace

struct NativeDispatchSession {
    std::mutex mutex;
    uint8_t *sym = nullptr;
    uint64_t sym_bytes = 0u;
    uint32_t pe = 0u;
    StreamPreparedWorkspace prepared{};
    NativeTicket ticket{};
    uint64_t next_ticket_generation = 1u;
    uint64_t last_protocol_cycles = 0u;
    bool fault_armed = false;
    uint32_t fault_lane = 0u;
    uint32_t fault_error = 0u;
};

namespace {

NativeTicket *FindTicket(
    NativeDispatchSession *session, inc_dc_fw_backend_ticket_t ticket)
{
    return session != nullptr && ticket.value == 1u &&
                   session->ticket.live &&
                   session->ticket.generation == ticket.generation
               ? &session->ticket
               : nullptr;
}

inc_dc_fw_status_t EnqueueKernel(
    NativeDispatchSession *session, uint64_t stream_value,
    uint64_t operation_generation,
    const inc_dc_fw_invocation_t *invocation,
    inc_dc_fw_backend_ticket_t *ticket)
{
    if (session == nullptr || stream_value == 0u ||
        operation_generation == 0u || ticket == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->ticket.live) return INC_DC_FW_CAPACITY_EXCEEDED;
    const uint32_t world = session->prepared.descriptor.workers;
    const bool worker = session->pe < world;
    const bool inc = session->pe == world;
    if (!worker && !inc) return INC_DC_FW_INVALID_ARGUMENT;
    if (worker) {
        const uint64_t local_rows =
            session->prepared.destination_physical_rows[session->pe];
        if (invocation == nullptr ||
            invocation->shape.tokens !=
                session->prepared.descriptor.tokens_per_worker ||
            invocation->shape.topk != session->prepared.descriptor.topk ||
            invocation->shape.hidden_size * 2u !=
                session->prepared.descriptor.hidden_bytes ||
            invocation->route.semantic_digest !=
                session->prepared.source_semantic_digests[session->pe] ||
            invocation->route.generation != operation_generation ||
            !Contiguous2d(
                invocation->input,
                session->prepared.descriptor.tokens_per_worker,
                invocation->shape.hidden_size) ||
            !Contiguous2d(
                invocation->output, local_rows,
                invocation->shape.hidden_size)) {
            return INC_DC_FW_LAYOUT_MISMATCH;
        }
    }

    NativeTicket &entry = session->ticket;
    entry = {};
    entry.generation = session->next_ticket_generation++;
    if (entry.generation == 0u) entry.generation =
        session->next_ticket_generation++;
    entry.state = INC_DC_FW_REQUEST_SUBMITTED;
    entry.descriptor = session->prepared.descriptor;
    entry.descriptor.pe = session->pe;
    if (operation_generation > std::numeric_limits<uint32_t>::max()) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    entry.descriptor.generation =
        static_cast<uint32_t>(operation_generation);
    auto stream = reinterpret_cast<aclrtStream>(
        static_cast<uintptr_t>(stream_value));
    if (aclrtCreateEvent(&entry.event) != ACL_SUCCESS) {
        entry = {};
        return INC_DC_FW_BACKEND_ERROR;
    }
    auto fail = [&]() {
        (void)aclrtSynchronizeStream(stream);
        (void)aclrtDestroyEvent(entry.event);
        entry = {};
        return INC_DC_FW_BACKEND_ERROR;
    };
    if (aclrtMemcpyAsync(
            session->sym + kStreamDescOff, sizeof(entry.descriptor),
            &entry.descriptor, sizeof(entry.descriptor),
            ACL_MEMCPY_HOST_TO_DEVICE, stream) != ACL_SUCCESS) {
        return fail();
    }
    if (worker) {
        const uint64_t input_bytes =
            session->prepared.descriptor.input_stride;
        uint8_t *input_destination =
            session->sym + session->prepared.descriptor.input_off +
            static_cast<uint64_t>(session->pe) * input_bytes;
        if (invocation->input.data != input_destination && aclrtMemcpyAsync(
                input_destination, input_bytes, invocation->input.data,
                input_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream) !=
            ACL_SUCCESS) {
            return fail();
        }
    }
    const int block_dim = static_cast<int>(
        worker ? session->prepared.resources.dispatch_worker_aiv
               : session->prepared.resources.dispatch_inc_aiv);
    launch_inc_dc_single_inc_stream_dispatch_kernel(
        session->sym, block_dim, stream);
    if (worker) {
        const uint64_t output_rows =
            session->prepared.destination_physical_rows[session->pe];
        const uint64_t output_bytes =
            output_rows * session->prepared.descriptor.hidden_bytes;
        const uint64_t output_offset =
            session->prepared.destination_output_offsets[session->pe];
        uint8_t *native_output =
            session->sym + session->prepared.descriptor.output_off +
                output_offset;
        if (invocation->output.data != native_output && aclrtMemcpyAsync(
                invocation->output.data, output_bytes,
                native_output,
                output_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream) !=
            ACL_SUCCESS) {
            return fail();
        }
    }
    if (aclrtRecordEvent(entry.event, stream) != ACL_SUCCESS) return fail();
    entry.live = true;
    ticket->value = 1u;
    ticket->generation = entry.generation;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t GetCapabilities(
    void *opaque, inc_dc_fw_capabilities_t *capabilities)
{
    auto *session = static_cast<NativeDispatchSession *>(opaque);
    if (session == nullptr || capabilities == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    capabilities->feature_bits =
        INC_DC_FW_FEATURE_EXTERNAL_STREAM |
        INC_DC_FW_FEATURE_DEVICE_ROUTE;
    capabilities->max_world_size =
        session->prepared.descriptor.workers;
    capabilities->max_topk = session->prepared.descriptor.topk;
    capabilities->max_inflight = 1u;
    capabilities->workspace_alignment = 256u;
    capabilities->dtype_bits =
        (1ull << INC_DC_FW_DTYPE_FP16) |
        (1ull << INC_DC_FW_DTYPE_BF16);
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t QueryWorkspace(
    void *opaque, const inc_dc_fw_plan_desc_t *, uint32_t operation,
    const inc_dc_fw_shape_t *shape, inc_dc_fw_workspace_t *workspace)
{
    auto *session = static_cast<NativeDispatchSession *>(opaque);
    if (session == nullptr || shape == nullptr || workspace == nullptr ||
        operation != INC_DC_FW_OP_DISPATCH ||
        shape->tokens != session->prepared.descriptor.tokens_per_worker ||
        shape->topk != session->prepared.descriptor.topk ||
        shape->hidden_size * 2u !=
            session->prepared.descriptor.hidden_bytes) {
        return INC_DC_FW_UNSUPPORTED;
    }
    workspace->struct_size = sizeof(*workspace);
    workspace->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    workspace->bytes = 256u;
    workspace->alignment = 256u;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Enqueue(
    void *opaque, const inc_dc_fw_plan_desc_t *plan, uint32_t operation,
    const inc_dc_fw_invocation_t *invocation,
    inc_dc_fw_backend_ticket_t *ticket)
{
    if (operation != INC_DC_FW_OP_DISPATCH || invocation == nullptr ||
        plan == nullptr) {
        return INC_DC_FW_UNSUPPORTED;
    }
    inc_dc_fw_invocation_t bound = *invocation;
    if (bound.route.data == nullptr) {
        bound.route = plan->static_route;
        bound.route.generation = bound.operation_generation;
    }
    return EnqueueKernel(
        static_cast<NativeDispatchSession *>(opaque), bound.stream,
        bound.operation_generation, &bound, ticket);
}

inc_dc_fw_status_t Query(
    void *opaque, inc_dc_fw_backend_ticket_t ticket, uint32_t *state)
{
    auto *session = static_cast<NativeDispatchSession *>(opaque);
    if (state == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(session->mutex);
    NativeTicket *entry = FindTicket(session, ticket);
    if (entry == nullptr) return INC_DC_FW_STALE_HANDLE;
    aclrtEventRecordedStatus event_status =
        ACL_EVENT_RECORDED_STATUS_NOT_READY;
    if (aclrtQueryEventStatus(entry->event, &event_status) != ACL_SUCCESS) {
        entry->state = INC_DC_FW_REQUEST_FAILED;
        *state = entry->state;
        return INC_DC_FW_BACKEND_ERROR;
    }
    if (event_status == ACL_EVENT_RECORDED_STATUS_COMPLETE) {
        std::array<StreamLaneStat, kStreamMaxLanes> stats{};
        const uint64_t stats_offset = entry->descriptor.stats_off +
            static_cast<uint64_t>(session->pe) * kStreamMaxLanes *
                sizeof(StreamLaneStat);
        const uint32_t active_lanes = session->pe <
                entry->descriptor.workers
            ? session->prepared.resources.dispatch_worker_aiv
            : session->prepared.resources.dispatch_inc_aiv;
        if (session->fault_armed) {
            const uint64_t fault_offset = stats_offset +
                static_cast<uint64_t>(session->fault_lane) *
                    sizeof(StreamLaneStat) + offsetof(StreamLaneStat, error);
            if (aclrtMemcpy(
                    session->sym + fault_offset, sizeof(uint32_t),
                    &session->fault_error, sizeof(uint32_t),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                entry->state = INC_DC_FW_REQUEST_FAILED;
                *state = entry->state;
                return INC_DC_FW_BACKEND_ERROR;
            }
            session->fault_armed = false;
        }
        if (aclrtMemcpy(stats.data(), stats.size() * sizeof(StreamLaneStat),
                        session->sym + stats_offset,
                        stats.size() * sizeof(StreamLaneStat),
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            entry->state = INC_DC_FW_REQUEST_FAILED;
            *state = entry->state;
            return INC_DC_FW_BACKEND_ERROR;
        }
        entry->state = INC_DC_FW_REQUEST_COMPLETED;
        if (stats[0].end_cycle > stats[0].start_cycle) {
            session->last_protocol_cycles =
                stats[0].end_cycle - stats[0].start_cycle;
        } else {
            session->last_protocol_cycles = 0u;
        }
        for (uint32_t lane = 0u; lane < active_lanes; ++lane) {
            if (stats[lane].error == 0u) continue;
            std::cerr << "NATIVE_DISPATCH_PROTOCOL_ERROR pe="
                      << session->pe << " generation="
                      << entry->descriptor.generation << " lane=" << lane
                      << " error=" << stats[lane].error << std::endl;
            entry->state = INC_DC_FW_REQUEST_FAILED;
        }
    }
    *state = entry->state;
    if (entry->state == INC_DC_FW_REQUEST_SUBMITTED)
        return INC_DC_FW_NOT_READY;
    return entry->state == INC_DC_FW_REQUEST_COMPLETED
               ? INC_DC_FW_OK : INC_DC_FW_BACKEND_ERROR;
}

inc_dc_fw_status_t Wait(
    void *opaque, inc_dc_fw_backend_ticket_t ticket, uint64_t timeout_ns)
{
    auto *session = static_cast<NativeDispatchSession *>(opaque);
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
        uint32_t state = INC_DC_FW_REQUEST_SUBMITTED;
        const inc_dc_fw_status_t status = Query(opaque, ticket, &state);
        if (status == INC_DC_FW_OK) return INC_DC_FW_OK;
        if (status != INC_DC_FW_NOT_READY) return status;
        if (timeout_ns != 0u) {
            const auto elapsed = std::chrono::duration_cast<
                std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start).count();
            if (elapsed >= 0 &&
                static_cast<uint64_t>(elapsed) >= timeout_ns) {
                return INC_DC_FW_TIMEOUT;
            }
        }
        std::this_thread::yield();
    }
}

inc_dc_fw_status_t Cancel(void *, inc_dc_fw_backend_ticket_t)
{
    return INC_DC_FW_UNSUPPORTED;
}

inc_dc_fw_status_t Release(
    void *opaque, inc_dc_fw_backend_ticket_t ticket)
{
    auto *session = static_cast<NativeDispatchSession *>(opaque);
    std::lock_guard<std::mutex> lock(session->mutex);
    NativeTicket *entry = FindTicket(session, ticket);
    if (entry == nullptr) return INC_DC_FW_STALE_HANDLE;
    if (entry->state == INC_DC_FW_REQUEST_SUBMITTED) {
        return INC_DC_FW_NOT_READY;
    }
    if (aclrtDestroyEvent(entry->event) != ACL_SUCCESS) {
        return INC_DC_FW_BACKEND_ERROR;
    }
    *entry = {};
    return INC_DC_FW_OK;
}

} // namespace

inc_dc_fw_status_t CreateNativeDispatchSession(
    const NativeDispatchSessionConfig &config,
    NativeDispatchSession **session)
{
    if (session == nullptr || config.symmetric_heap == nullptr ||
        config.prepared == nullptr ||
        config.local_pe > config.prepared->descriptor.workers ||
        config.symmetric_heap_bytes <
            config.prepared->descriptor.total_bytes ||
        config.prepared->destination_output_offsets.size() !=
            config.prepared->descriptor.workers ||
        config.prepared->destination_physical_rows.size() !=
            config.prepared->descriptor.workers ||
        config.prepared->source_semantic_digests.size() !=
            config.prepared->descriptor.workers) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    auto *created = new (std::nothrow) NativeDispatchSession();
    if (created == nullptr) return INC_DC_FW_INTERNAL;
    created->sym = config.symmetric_heap;
    created->sym_bytes = config.symmetric_heap_bytes;
    created->pe = config.local_pe;
    created->prepared = *config.prepared;
    StreamDispatchDesc descriptor = created->prepared.descriptor;
    descriptor.pe = created->pe;
    bool ok = aclrtMemset(
                  created->sym, descriptor.total_bytes, 0,
                  descriptor.total_bytes) == ACL_SUCCESS &&
              CopyH2D(created->sym + kStreamDescOff, sizeof(descriptor),
                      &descriptor, sizeof(descriptor)) &&
              CopyH2D(created->sym + descriptor.task_off,
                      created->prepared.tasks.size() *
                          sizeof(StreamDispatchTask),
                      created->prepared.tasks.data(),
                      created->prepared.tasks.size() *
                          sizeof(StreamDispatchTask)) &&
              CopyH2D(created->sym + descriptor.route_off,
                      created->prepared.routes.size() *
                          sizeof(StreamRouteEntry),
                      created->prepared.routes.data(),
                      created->prepared.routes.size() *
                          sizeof(StreamRouteEntry)) &&
              CopyH2D(created->sym + descriptor.expert_assignment_off,
                      created->prepared.assignments.size() *
                          sizeof(StreamExpertAssignment),
                      created->prepared.assignments.data(),
                      created->prepared.assignments.size() *
                          sizeof(StreamExpertAssignment)) &&
              CopyH2D(created->sym + descriptor.tx_lane_task_offsets_off,
                      (kStreamMaxLanes + 1u) * sizeof(uint32_t),
                      created->prepared.tx_lane_task_offsets.data(),
                      created->prepared.tx_lane_task_offsets.size() *
                          sizeof(uint32_t)) &&
              CopyH2D(created->sym + descriptor.tx_lane_task_indices_off,
                      created->prepared.tx_lane_task_indices.size() *
                          sizeof(uint32_t),
                      created->prepared.tx_lane_task_indices.data(),
                      created->prepared.tx_lane_task_indices.size() *
                          sizeof(uint32_t)) &&
              CopyH2D(created->sym + descriptor.worker_task_offsets_off,
                      (descriptor.workers + 1u) * sizeof(uint32_t),
                      created->prepared.worker_task_offsets.data(),
                      created->prepared.worker_task_offsets.size() *
                          sizeof(uint32_t)) &&
              CopyH2D(created->sym + descriptor.worker_task_indices_off,
                      created->prepared.worker_task_indices.size() *
                          sizeof(uint32_t),
                      created->prepared.worker_task_indices.data(),
                      created->prepared.worker_task_indices.size() *
                          sizeof(uint32_t));
    if (!ok) {
        delete created;
        return INC_DC_FW_BACKEND_ERROR;
    }
    *session = created;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t DestroyNativeDispatchSession(
    NativeDispatchSession *session)
{
    if (session == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        if (session->ticket.live) return INC_DC_FW_BUSY;
    }
    delete session;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t NativeDispatchBackendOps(
    NativeDispatchSession *session, inc_dc_fw_backend_ops_t *ops)
{
    if (session == nullptr || ops == nullptr ||
        session->pe >= session->prepared.descriptor.workers) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    *ops = {};
    ops->struct_size = sizeof(*ops);
    ops->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    ops->backend_context = session;
    ops->get_capabilities = GetCapabilities;
    ops->query_workspace = QueryWorkspace;
    ops->enqueue = Enqueue;
    ops->query = Query;
    ops->wait = Wait;
    ops->cancel = Cancel;
    ops->release = Release;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t NativeDispatchWorkerBuffers(
    NativeDispatchSession *session, void **input, uint64_t *input_bytes,
    void **output, uint64_t *output_bytes)
{
    if (session == nullptr || input == nullptr || input_bytes == nullptr ||
        output == nullptr || output_bytes == nullptr ||
        session->pe >= session->prepared.descriptor.workers) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    *input_bytes = session->prepared.descriptor.input_stride;
    *input = session->sym + session->prepared.descriptor.input_off +
        static_cast<uint64_t>(session->pe) * *input_bytes;
    *output_bytes =
        session->prepared.destination_physical_rows[session->pe] *
        session->prepared.descriptor.hidden_bytes;
    *output = session->sym + session->prepared.descriptor.output_off +
        session->prepared.destination_output_offsets[session->pe];
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t NativeDispatchLastProtocolCycles(
    NativeDispatchSession *session, uint64_t *cycles)
{
    if (session == nullptr || cycles == nullptr)
        return INC_DC_FW_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->last_protocol_cycles == 0u) return INC_DC_FW_NOT_READY;
    *cycles = session->last_protocol_cycles;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t NativeDispatchArmLaneError(
    NativeDispatchSession *session, uint32_t lane, uint32_t error)
{
    if (session == nullptr || error == 0u)
        return INC_DC_FW_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(session->mutex);
    const uint32_t active = session->pe < session->prepared.descriptor.workers
        ? session->prepared.resources.dispatch_worker_aiv
        : session->prepared.resources.dispatch_inc_aiv;
    if (lane >= active) return INC_DC_FW_INVALID_ARGUMENT;
    if (session->ticket.live || session->fault_armed) return INC_DC_FW_BUSY;
    session->fault_lane = lane;
    session->fault_error = error;
    session->fault_armed = true;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t NativeDispatchPrepareGeneration(
    NativeDispatchSession *session, uint64_t operation_generation)
{
    if (session == nullptr || operation_generation == 0u ||
        operation_generation > std::numeric_limits<uint32_t>::max()) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->ticket.live) return INC_DC_FW_BUSY;
    auto &desc = session->prepared.descriptor;
    const uint64_t tile_ready_bytes =
        static_cast<uint64_t>(desc.workers) * desc.tiles_per_worker * 64u;
    const uint64_t upload_done_bytes =
        static_cast<uint64_t>(desc.workers) * desc.tiles_per_worker *
        kStreamMaxLanes * 64u;
    const uint64_t lane_done_bytes =
        static_cast<uint64_t>(desc.workers + 1u) * kStreamMaxLanes * 64u;
    const uint64_t completion_bytes =
        static_cast<uint64_t>(desc.workers + desc.gather_chunk_count) * 64u;
    const uint64_t start_gate_bytes =
        static_cast<uint64_t>(desc.workers * 4u + 1u) * 64u;
    const uint64_t stats_bytes =
        static_cast<uint64_t>(desc.workers + 1u) * kStreamMaxLanes *
        sizeof(StreamLaneStat);
    StreamDispatchDesc local = desc;
    local.pe = session->pe;
    local.generation = static_cast<uint32_t>(operation_generation);
    const bool ok =
        aclrtMemset(session->sym + desc.tile_ready_off, tile_ready_bytes, 0,
                    tile_ready_bytes) == ACL_SUCCESS &&
        aclrtMemset(session->sym + desc.direct_ready_off, tile_ready_bytes, 0,
                    tile_ready_bytes) == ACL_SUCCESS &&
        aclrtMemset(session->sym + desc.upload_chunk_done_off,
                    upload_done_bytes, 0, upload_done_bytes) == ACL_SUCCESS &&
        aclrtMemset(session->sym + desc.lane_done_off, lane_done_bytes, 0,
                    lane_done_bytes) == ACL_SUCCESS &&
        aclrtMemset(session->sym + desc.completion_off, completion_bytes, 0,
                    completion_bytes) == ACL_SUCCESS &&
        aclrtMemset(session->sym + desc.start_gate_off, start_gate_bytes, 0,
                    start_gate_bytes) == ACL_SUCCESS &&
        aclrtMemset(session->sym + desc.stats_off, stats_bytes, 0,
                    stats_bytes) == ACL_SUCCESS &&
        aclrtMemcpy(session->sym + kStreamDescOff, sizeof(local), &local,
                    sizeof(local), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    return ok ? INC_DC_FW_OK : INC_DC_FW_BACKEND_ERROR;
}

inc_dc_fw_status_t NativeDispatchIncEnqueue(
    NativeDispatchSession *session, uint64_t stream,
    uint64_t operation_generation, inc_dc_fw_backend_ticket_t *ticket)
{
    if (session == nullptr ||
        session->pe != session->prepared.descriptor.workers) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    return EnqueueKernel(
        session, stream, operation_generation, nullptr, ticket);
}

inc_dc_fw_status_t NativeDispatchIncWaitAndRelease(
    NativeDispatchSession *session, inc_dc_fw_backend_ticket_t ticket,
    uint64_t timeout_ns)
{
    const inc_dc_fw_status_t status = Wait(session, ticket, timeout_ns);
    return status == INC_DC_FW_OK ? Release(session, ticket) : status;
}

} // namespace inc::dc::single_stream
