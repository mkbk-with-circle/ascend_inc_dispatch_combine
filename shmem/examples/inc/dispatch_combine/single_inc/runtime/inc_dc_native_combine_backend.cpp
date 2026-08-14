#include "inc_dc_native_combine_backend.h"

#include <algorithm>
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

namespace inc::dc::single_stream {

extern "C" void launch_inc_dc_sv2_dyn_csr_combine_kernel(
    uint8_t *sym, uint64_t ctrl_off, int block_dim, void *stream);
extern "C" void launch_inc_dc_sv2_dyn_csr_producer_kernel(
    uint8_t *sym, uint64_t ctrl_off, int block_dim, void *stream);

namespace {

struct NativeCombineTicket {
    bool live = false;
    bool protocol_checked = false;
    uint64_t ticket_generation = 0u;
    uint32_t operation_generation = 0u;
    uint32_t state = INC_DC_FW_REQUEST_FREE;
    aclrtEvent event = nullptr;
};

bool Contiguous2d(const inc_dc_fw_tensor_desc_t &tensor, uint64_t rows,
                  uint32_t hidden)
{
    return tensor.data != nullptr && tensor.ndim == 2u &&
           tensor.dims[0] >= 0 &&
           static_cast<uint64_t>(tensor.dims[0]) >= rows &&
           tensor.dims[1] == static_cast<int64_t>(hidden) &&
           tensor.strides[0] == static_cast<int64_t>(hidden) &&
           tensor.strides[1] == 1 &&
           tensor.memory_location == INC_DC_FW_MEMORY_DEVICE;
}

} // namespace

struct NativeCombineSession {
    std::mutex mutex;
    uint8_t *sym = nullptr;
    uint64_t sym_bytes = 0u;
    uint32_t pe = 0u;
    uint32_t tokens = 0u;
    uint32_t topk = 0u;
    NativeCombinePreparedWorkspace prepared{};
    std::vector<uint64_t> source_digests;
    NativeCombineTicket ticket{};
    uint64_t next_ticket_generation = 1u;
    uint64_t last_protocol_cycles = 0u;
    bool producer_fault_armed = false;
    uint32_t producer_fault_lane = 0u;
    uint32_t producer_fault_error = 0u;
};

namespace {

NativeCombineTicket *FindTicket(
    NativeCombineSession *session, inc_dc_fw_backend_ticket_t ticket)
{
    return session != nullptr && ticket.value == 1u &&
                   session->ticket.live &&
                   session->ticket.ticket_generation == ticket.generation
               ? &session->ticket : nullptr;
}

inc_dc_fw_status_t EnqueueKernel(
    NativeCombineSession *session, uint64_t stream_value,
    uint64_t operation_generation,
    const inc_dc_fw_invocation_t *invocation,
    inc_dc_fw_backend_ticket_t *ticket)
{
    if (session == nullptr || stream_value == 0u || ticket == nullptr ||
        operation_generation == 0u ||
        operation_generation > std::numeric_limits<uint32_t>::max()) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->ticket.live) return INC_DC_FW_CAPACITY_EXCEEDED;
    const uint32_t workers = session->prepared.control.worker_count;
    const bool worker = session->pe < workers;
    const bool inc = session->pe == workers;
    if (!worker && !inc) return INC_DC_FW_INVALID_ARGUMENT;
    const uint32_t hidden = session->prepared.control.hidden;
    if (worker) {
        const uint64_t input_rows =
            session->prepared.contributor_rows[session->pe];
        if (invocation == nullptr || invocation->shape.tokens != session->tokens ||
            invocation->shape.topk != session->topk ||
            invocation->shape.hidden_size != hidden ||
            invocation->shape.dtype != INC_DC_FW_DTYPE_FP16 ||
            invocation->weights.data != nullptr ||
            invocation->route.semantic_digest !=
                session->source_digests[session->pe] ||
            invocation->route.generation != operation_generation ||
            !Contiguous2d(invocation->input, input_rows, hidden) ||
            !Contiguous2d(invocation->output, session->tokens, hidden)) {
            return INC_DC_FW_LAYOUT_MISMATCH;
        }
    }

    NativeCombineTicket &entry = session->ticket;
    entry = {};
    entry.ticket_generation = session->next_ticket_generation++;
    if (entry.ticket_generation == 0u)
        entry.ticket_generation = session->next_ticket_generation++;
    entry.operation_generation = static_cast<uint32_t>(operation_generation);
    entry.state = INC_DC_FW_REQUEST_SUBMITTED;
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
            session->sym + offsetof(DynCsrCtrl, generation),
            sizeof(uint32_t), &entry.operation_generation,
            sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE, stream) !=
        ACL_SUCCESS) {
        return fail();
    }
    if (worker) {
        const uint64_t row_bytes = static_cast<uint64_t>(hidden) * 2u;
        const auto *input = static_cast<const uint8_t *>(invocation->input.data);
        const auto &copies = session->prepared.input_copies[session->pe];
        const uint8_t *native_input = copies.empty() ? nullptr :
            session->sym + session->prepared.control.ingress_off +
            static_cast<uint64_t>(copies[0].ingress_slot) *
                session->prepared.control.tile_bytes;
        const bool input_is_native = input == native_input;
        for (size_t begin = 0u; begin < copies.size();) {
            size_t end = begin + 1u;
            if (row_bytes == session->prepared.control.tile_bytes) {
                while (end < copies.size() &&
                       copies[end].source_row ==
                           copies[begin].source_row + end - begin &&
                       copies[end].ingress_slot ==
                           copies[begin].ingress_slot + end - begin) {
                    ++end;
                }
            }
            const uint64_t bytes = (end - begin) * row_bytes;
            const auto &copy = copies[begin];
            uint8_t *destination = session->sym +
                session->prepared.control.ingress_off +
                static_cast<uint64_t>(copy.ingress_slot) *
                    session->prepared.control.tile_bytes;
            if (!input_is_native && aclrtMemcpyAsync(destination, bytes,
                                 input + static_cast<uint64_t>(copy.source_row) *
                                     row_bytes,
                                 bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                 stream) != ACL_SUCCESS) {
                return fail();
            }
            begin = end;
        }
        launch_inc_dc_sv2_dyn_csr_producer_kernel(
            session->sym, 0u,
            static_cast<int>(
                session->prepared.resources.combine_worker_aiv),
            stream);
        const uint64_t output_bytes =
            static_cast<uint64_t>(session->tokens) * row_bytes;
        uint8_t *native_output =
            session->sym + session->prepared.control.output_off;
        if (invocation->output.data != native_output && aclrtMemcpyAsync(
                invocation->output.data, output_bytes,
                native_output,
                output_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream) !=
            ACL_SUCCESS) {
            return fail();
        }
    } else {
        launch_inc_dc_sv2_dyn_csr_combine_kernel(
            session->sym, 0u,
            static_cast<int>(session->prepared.resources.combine_inc_aiv),
            stream);
    }
    if (aclrtRecordEvent(entry.event, stream) != ACL_SUCCESS) return fail();
    entry.live = true;
    ticket->value = 1u;
    ticket->generation = entry.ticket_generation;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t GetCapabilities(
    void *opaque, inc_dc_fw_capabilities_t *capabilities)
{
    auto *session = static_cast<NativeCombineSession *>(opaque);
    if (session == nullptr || capabilities == nullptr)
        return INC_DC_FW_INVALID_ARGUMENT;
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    capabilities->feature_bits = INC_DC_FW_FEATURE_EXTERNAL_STREAM |
                                 INC_DC_FW_FEATURE_DEVICE_ROUTE;
    capabilities->max_world_size = session->prepared.control.worker_count;
    capabilities->max_topk = session->topk;
    capabilities->max_inflight = 1u;
    capabilities->workspace_alignment = 256u;
    capabilities->dtype_bits = 1ull << INC_DC_FW_DTYPE_FP16;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t QueryWorkspace(
    void *opaque, const inc_dc_fw_plan_desc_t *, uint32_t operation,
    const inc_dc_fw_shape_t *shape, inc_dc_fw_workspace_t *workspace)
{
    auto *session = static_cast<NativeCombineSession *>(opaque);
    if (session == nullptr || shape == nullptr || workspace == nullptr ||
        operation != INC_DC_FW_OP_COMBINE || shape->tokens != session->tokens ||
        shape->topk != session->topk ||
        shape->hidden_size != session->prepared.control.hidden ||
        shape->dtype != INC_DC_FW_DTYPE_FP16) {
        return INC_DC_FW_LAYOUT_MISMATCH;
    }
    workspace->bytes = 256u;
    workspace->alignment = 256u;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Enqueue(
    void *opaque, const inc_dc_fw_plan_desc_t *plan, uint32_t operation,
    const inc_dc_fw_invocation_t *invocation,
    inc_dc_fw_backend_ticket_t *ticket)
{
    if (operation != INC_DC_FW_OP_COMBINE || invocation == nullptr ||
        plan == nullptr) return INC_DC_FW_UNSUPPORTED;
    inc_dc_fw_invocation_t bound = *invocation;
    if (bound.route.data == nullptr) {
        bound.route = plan->static_route;
        bound.route.generation = bound.operation_generation;
    }
    return EnqueueKernel(static_cast<NativeCombineSession *>(opaque),
                         bound.stream, bound.operation_generation,
                         &bound, ticket);
}

inc_dc_fw_status_t Query(
    void *opaque, inc_dc_fw_backend_ticket_t ticket, uint32_t *state)
{
    auto *session = static_cast<NativeCombineSession *>(opaque);
    if (session == nullptr || state == nullptr)
        return INC_DC_FW_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(session->mutex);
    NativeCombineTicket *entry = FindTicket(session, ticket);
    if (entry == nullptr) return INC_DC_FW_STALE_HANDLE;
    aclrtEventRecordedStatus event_status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
    if (aclrtQueryEventStatus(entry->event, &event_status) != ACL_SUCCESS) {
        entry->state = INC_DC_FW_REQUEST_FAILED;
        *state = entry->state;
        return INC_DC_FW_BACKEND_ERROR;
    }
    if (event_status == ACL_EVENT_RECORDED_STATUS_COMPLETE &&
        !entry->protocol_checked) {
        entry->protocol_checked = true;
        entry->state = INC_DC_FW_REQUEST_COMPLETED;
        if (session->pe == session->prepared.control.worker_count) {
            DynCsrStats stats{};
            std::array<DynCsrOwnerStats, kDynCsrMaxOwners> owners{};
            if (aclrtMemcpy(&stats, sizeof(stats),
                            session->sym + session->prepared.control.stats_off,
                            sizeof(stats), ACL_MEMCPY_DEVICE_TO_HOST) !=
                    ACL_SUCCESS ||
                aclrtMemcpy(
                    owners.data(), owners.size() * sizeof(DynCsrOwnerStats),
                    session->sym +
                        session->prepared.control.owner_stats_off,
                    owners.size() * sizeof(DynCsrOwnerStats),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                stats.magic != kDynCsrMagic ||
                stats.done != 1u ||
                stats.fail_code != 0u) {
                std::cerr << "NATIVE_COMBINE_PROTOCOL_ERROR pe=" << session->pe
                          << " generation=" << entry->operation_generation
                          << " magic=" << stats.magic << " done=" << stats.done
                          << " fail=" << stats.fail_code << std::endl;
                entry->state = INC_DC_FW_REQUEST_FAILED;
            }
            for (uint32_t owner = 0u;
                 owner < session->prepared.resources.combine_inc_aiv;
                 ++owner) {
                if (owners[owner].done == entry->operation_generation &&
                    owners[owner].fail_code == 0u) continue;
                std::cerr << "NATIVE_COMBINE_OWNER_ERROR pe=" << session->pe
                          << " generation=" << entry->operation_generation
                          << " owner=" << owner
                          << " done=" << owners[owner].done
                          << " fail=" << owners[owner].fail_code << std::endl;
                entry->state = INC_DC_FW_REQUEST_FAILED;
            }
            session->last_protocol_cycles = 0u;
            for (uint32_t owner = 0u;
                 owner < session->prepared.resources.combine_inc_aiv;
                 ++owner) {
                session->last_protocol_cycles = std::max(
                    session->last_protocol_cycles,
                    owners[owner].reduce_cycles);
            }
        } else {
            std::array<DynCsrProducerStats, kDynCsrMaxOwners> producers{};
            if (session->producer_fault_armed) {
                const uint64_t fault_offset =
                    session->prepared.control.owner_stats_off +
                    static_cast<uint64_t>(session->producer_fault_lane) *
                        sizeof(DynCsrProducerStats) +
                    offsetof(DynCsrProducerStats, reserved);
                if (aclrtMemcpy(
                        session->sym + fault_offset, sizeof(uint32_t),
                        &session->producer_fault_error, sizeof(uint32_t),
                        ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                    entry->state = INC_DC_FW_REQUEST_FAILED;
                }
                session->producer_fault_armed = false;
            }
            if (aclrtMemcpy(
                    producers.data(),
                    producers.size() * sizeof(DynCsrProducerStats),
                    session->sym + session->prepared.control.owner_stats_off,
                    producers.size() * sizeof(DynCsrProducerStats),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                entry->state = INC_DC_FW_REQUEST_FAILED;
            }
            for (uint32_t lane = 0u;
                 lane < session->prepared.resources.combine_worker_aiv;
                 ++lane) {
                if (producers[lane].done == entry->operation_generation &&
                    producers[lane].reserved[0] == 0u) continue;
                std::cerr << "NATIVE_COMBINE_PRODUCER_ERROR pe=" << session->pe
                          << " generation=" << entry->operation_generation
                          << " lane=" << lane
                          << " done=" << producers[lane].done
                          << " fail=" << producers[lane].reserved[0]
                          << std::endl;
                entry->state = INC_DC_FW_REQUEST_FAILED;
            }
            session->last_protocol_cycles = producers[0].kernel_cycles;
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
            if (elapsed >= 0 && static_cast<uint64_t>(elapsed) >= timeout_ns) {
                auto *session = static_cast<NativeCombineSession *>(opaque);
                DynCsrCtrl control{};
                DynCsrProducerStats producer{};
                DynCsrOwnerStats owner{};
                (void)aclrtMemcpy(&control, sizeof(control), session->sym,
                                  sizeof(control),
                                  ACL_MEMCPY_DEVICE_TO_HOST);
                (void)aclrtMemcpy(
                    &producer, sizeof(producer),
                    session->sym + session->prepared.control.owner_stats_off,
                    sizeof(producer), ACL_MEMCPY_DEVICE_TO_HOST);
                (void)aclrtMemcpy(
                    &owner, sizeof(owner),
                    session->sym + session->prepared.control.owner_stats_off,
                    sizeof(owner), ACL_MEMCPY_DEVICE_TO_HOST);
                std::cerr << "NATIVE_COMBINE_TIMEOUT pe=" << session->pe
                          << " expected_generation="
                          << session->ticket.operation_generation
                          << " control_generation=" << control.generation
                          << " lane_or_owner=" << producer.lane
                          << " done=" << producer.done
                          << " issued=" << producer.issued
                          << " ready=" << producer.ready_signals
                          << " fail=" << producer.reserved[0]
                          << " owner_done=" << owner.done
                          << " owner_fail=" << owner.fail_code << std::endl;
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
    auto *session = static_cast<NativeCombineSession *>(opaque);
    if (session == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(session->mutex);
    NativeCombineTicket *entry = FindTicket(session, ticket);
    if (entry == nullptr) return INC_DC_FW_STALE_HANDLE;
    if (entry->state == INC_DC_FW_REQUEST_SUBMITTED)
        return INC_DC_FW_NOT_READY;
    if (aclrtDestroyEvent(entry->event) != ACL_SUCCESS)
        return INC_DC_FW_BACKEND_ERROR;
    *entry = {};
    return INC_DC_FW_OK;
}

} // namespace

inc_dc_fw_status_t CreateNativeCombineSession(
    const NativeCombineSessionConfig &config,
    NativeCombineSession **session)
{
    if (session == nullptr || config.symmetric_heap == nullptr ||
        config.prepared == nullptr ||
        config.source_semantic_digests == nullptr ||
        config.tokens_per_worker == 0u || config.topk == 0u ||
        config.local_pe > config.prepared->control.worker_count ||
        config.symmetric_heap_bytes < config.prepared->heap_bytes ||
        config.source_semantic_digests->size() !=
            config.prepared->control.worker_count) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    auto *created = new (std::nothrow) NativeCombineSession();
    if (created == nullptr) return INC_DC_FW_INTERNAL;
    created->sym = config.symmetric_heap;
    created->sym_bytes = config.symmetric_heap_bytes;
    created->pe = config.local_pe;
    created->tokens = config.tokens_per_worker;
    created->topk = config.topk;
    created->prepared = *config.prepared;
    created->source_digests = *config.source_semantic_digests;
    std::vector<uint8_t> image = created->prepared.immutable_image;
    DynCsrCtrl control = created->prepared.control;
    control.this_worker_rank = config.local_pe < control.worker_count
        ? config.local_pe : 0u;
    std::memcpy(image.data(), &control, sizeof(control));
    if (aclrtMemcpy(created->sym, created->sym_bytes, image.data(),
                    image.size(), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        delete created;
        return INC_DC_FW_BACKEND_ERROR;
    }
    *session = created;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t DestroyNativeCombineSession(NativeCombineSession *session)
{
    if (session == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        if (session->ticket.live) return INC_DC_FW_BUSY;
    }
    delete session;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t NativeCombineBackendOps(
    NativeCombineSession *session, inc_dc_fw_backend_ops_t *ops)
{
    if (session == nullptr || ops == nullptr ||
        session->pe >= session->prepared.control.worker_count)
        return INC_DC_FW_INVALID_ARGUMENT;
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

inc_dc_fw_status_t NativeCombineWorkerBuffers(
    NativeCombineSession *session, void **input, uint64_t *input_bytes,
    void **output, uint64_t *output_bytes)
{
    if (session == nullptr || input == nullptr || input_bytes == nullptr ||
        output == nullptr || output_bytes == nullptr ||
        session->pe >= session->prepared.control.worker_count) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    const auto &copies = session->prepared.input_copies[session->pe];
    if (copies.empty() ||
        session->prepared.control.tile_bytes !=
            static_cast<uint64_t>(session->prepared.control.hidden) * 2u) {
        return INC_DC_FW_UNSUPPORTED;
    }
    for (size_t index = 0u; index < copies.size(); ++index) {
        if (copies[index].source_row != index ||
            copies[index].ingress_slot !=
                copies[0].ingress_slot + index) {
            return INC_DC_FW_UNSUPPORTED;
        }
    }
    *input_bytes = copies.size() * session->prepared.control.tile_bytes;
    *input = session->sym + session->prepared.control.ingress_off +
        static_cast<uint64_t>(copies[0].ingress_slot) *
            session->prepared.control.tile_bytes;
    *output_bytes = static_cast<uint64_t>(session->tokens) *
        session->prepared.control.tile_bytes;
    *output = session->sym + session->prepared.control.output_off;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t NativeCombineLastProtocolCycles(
    NativeCombineSession *session, uint64_t *cycles)
{
    if (session == nullptr || cycles == nullptr)
        return INC_DC_FW_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->last_protocol_cycles == 0u) return INC_DC_FW_NOT_READY;
    *cycles = session->last_protocol_cycles;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t NativeCombineArmProducerError(
    NativeCombineSession *session, uint32_t lane, uint32_t error)
{
    if (session == nullptr || error == 0u)
        return INC_DC_FW_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->pe >= session->prepared.control.worker_count ||
        lane >= session->prepared.resources.combine_worker_aiv) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    if (session->ticket.live || session->producer_fault_armed)
        return INC_DC_FW_BUSY;
    session->producer_fault_lane = lane;
    session->producer_fault_error = error;
    session->producer_fault_armed = true;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t NativeCombineIncEnqueue(
    NativeCombineSession *session, uint64_t stream,
    uint64_t operation_generation, inc_dc_fw_backend_ticket_t *ticket)
{
    if (session == nullptr ||
        session->pe != session->prepared.control.worker_count)
        return INC_DC_FW_INVALID_ARGUMENT;
    return EnqueueKernel(session, stream, operation_generation, nullptr,
                         ticket);
}

inc_dc_fw_status_t NativeCombineIncWaitAndRelease(
    NativeCombineSession *session, inc_dc_fw_backend_ticket_t ticket,
    uint64_t timeout_ns)
{
    const inc_dc_fw_status_t status = Wait(session, ticket, timeout_ns);
    return status == INC_DC_FW_OK ? Release(session, ticket) : status;
}

} // namespace inc::dc::single_stream
