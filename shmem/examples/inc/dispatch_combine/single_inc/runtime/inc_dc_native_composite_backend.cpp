#include "inc_dc_native_composite_backend.h"

#include <algorithm>
#include <array>
#include <limits>
#include <mutex>
#include <new>

namespace inc::dc::single_stream {

namespace {

bool ValidChild(const inc_dc_fw_backend_ops_t &ops)
{
    return ops.struct_size >= sizeof(inc_dc_fw_backend_ops_t) &&
           ops.abi_version == INC_DC_FRAMEWORK_ABI_VERSION &&
           ops.backend_context != nullptr && ops.get_capabilities != nullptr &&
           ops.query_workspace != nullptr && ops.enqueue != nullptr &&
           ops.query != nullptr && ops.wait != nullptr &&
           ops.cancel != nullptr && ops.release != nullptr;
}

struct TicketSlot {
    bool live = false;
    uint64_t generation = 0u;
    inc_dc_fw_backend_ticket_t child{};
};

} // namespace

struct NativeCompositeBackend {
    std::mutex mutex;
    inc_dc_fw_backend_ops_t dispatch{};
    inc_dc_fw_backend_ops_t combine{};
    std::array<TicketSlot, 2u> tickets{};
    uint64_t next_generation = 1u;
};

namespace {

const inc_dc_fw_backend_ops_t *Select(
    NativeCompositeBackend *backend, uint32_t operation)
{
    if (backend == nullptr) return nullptr;
    if (operation == INC_DC_FW_OP_DISPATCH) return &backend->dispatch;
    if (operation == INC_DC_FW_OP_COMBINE) return &backend->combine;
    return nullptr;
}

bool Resolve(NativeCompositeBackend *backend,
             inc_dc_fw_backend_ticket_t ticket, uint32_t *index,
             inc_dc_fw_backend_ticket_t *child)
{
    if (backend == nullptr || index == nullptr || child == nullptr ||
        ticket.value == 0u || ticket.value > backend->tickets.size()) {
        return false;
    }
    const uint32_t slot = static_cast<uint32_t>(ticket.value - 1u);
    std::lock_guard<std::mutex> lock(backend->mutex);
    const TicketSlot &entry = backend->tickets[slot];
    if (!entry.live || entry.generation != ticket.generation) return false;
    *index = slot;
    *child = entry.child;
    return true;
}

inc_dc_fw_status_t GetCapabilities(
    void *opaque, inc_dc_fw_capabilities_t *capabilities)
{
    auto *backend = static_cast<NativeCompositeBackend *>(opaque);
    if (backend == nullptr || capabilities == nullptr)
        return INC_DC_FW_INVALID_ARGUMENT;
    inc_dc_fw_capabilities_t d{};
    inc_dc_fw_capabilities_t c{};
    inc_dc_fw_status_t status = backend->dispatch.get_capabilities(
        backend->dispatch.backend_context, &d);
    if (status != INC_DC_FW_OK) return status;
    status = backend->combine.get_capabilities(
        backend->combine.backend_context, &c);
    if (status != INC_DC_FW_OK) return status;
    *capabilities = {};
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    capabilities->feature_bits = d.feature_bits & c.feature_bits;
    capabilities->max_world_size = std::min(d.max_world_size, c.max_world_size);
    capabilities->max_topk = std::min(d.max_topk, c.max_topk);
    capabilities->max_inflight = 1u;
    capabilities->workspace_alignment =
        std::max(d.workspace_alignment, c.workspace_alignment);
    capabilities->dtype_bits = d.dtype_bits & c.dtype_bits;
    return capabilities->max_world_size == 0u ||
                   capabilities->max_topk == 0u ||
                   capabilities->dtype_bits == 0u
               ? INC_DC_FW_UNSUPPORTED : INC_DC_FW_OK;
}

inc_dc_fw_status_t QueryWorkspace(
    void *opaque, const inc_dc_fw_plan_desc_t *plan, uint32_t operation,
    const inc_dc_fw_shape_t *shape, inc_dc_fw_workspace_t *workspace)
{
    auto *backend = static_cast<NativeCompositeBackend *>(opaque);
    const auto *child = Select(backend, operation);
    return child == nullptr ? INC_DC_FW_UNSUPPORTED :
        child->query_workspace(
            child->backend_context, plan, operation, shape, workspace);
}

inc_dc_fw_status_t Enqueue(
    void *opaque, const inc_dc_fw_plan_desc_t *plan, uint32_t operation,
    const inc_dc_fw_invocation_t *invocation,
    inc_dc_fw_backend_ticket_t *ticket)
{
    auto *backend = static_cast<NativeCompositeBackend *>(opaque);
    const auto *child = Select(backend, operation);
    if (child == nullptr || ticket == nullptr) return INC_DC_FW_UNSUPPORTED;
    const uint32_t slot = operation == INC_DC_FW_OP_DISPATCH ? 0u : 1u;
    {
        std::lock_guard<std::mutex> lock(backend->mutex);
        if (backend->tickets[slot].live) return INC_DC_FW_CAPACITY_EXCEEDED;
    }
    inc_dc_fw_backend_ticket_t child_ticket{};
    const inc_dc_fw_status_t status = child->enqueue(
        child->backend_context, plan, operation, invocation, &child_ticket);
    if (status != INC_DC_FW_OK) return status;
    if (child_ticket.value == 0u || child_ticket.generation == 0u) {
        return INC_DC_FW_BACKEND_ERROR;
    }
    std::lock_guard<std::mutex> lock(backend->mutex);
    TicketSlot &entry = backend->tickets[slot];
    if (entry.live) {
        (void)child->cancel(child->backend_context, child_ticket);
        (void)child->release(child->backend_context, child_ticket);
        return INC_DC_FW_CAPACITY_EXCEEDED;
    }
    entry.live = true;
    entry.child = child_ticket;
    entry.generation = backend->next_generation++;
    if (entry.generation == 0u) entry.generation = backend->next_generation++;
    ticket->value = slot + 1u;
    ticket->generation = entry.generation;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Query(
    void *opaque, inc_dc_fw_backend_ticket_t ticket, uint32_t *state)
{
    auto *backend = static_cast<NativeCompositeBackend *>(opaque);
    uint32_t slot = 0u;
    inc_dc_fw_backend_ticket_t child_ticket{};
    if (state == nullptr || !Resolve(backend, ticket, &slot, &child_ticket))
        return INC_DC_FW_STALE_HANDLE;
    const auto &child = slot == 0u ? backend->dispatch : backend->combine;
    return child.query(child.backend_context, child_ticket, state);
}

inc_dc_fw_status_t Wait(
    void *opaque, inc_dc_fw_backend_ticket_t ticket, uint64_t timeout_ns)
{
    auto *backend = static_cast<NativeCompositeBackend *>(opaque);
    uint32_t slot = 0u;
    inc_dc_fw_backend_ticket_t child_ticket{};
    if (!Resolve(backend, ticket, &slot, &child_ticket))
        return INC_DC_FW_STALE_HANDLE;
    const auto &child = slot == 0u ? backend->dispatch : backend->combine;
    return child.wait(child.backend_context, child_ticket, timeout_ns);
}

inc_dc_fw_status_t Cancel(
    void *opaque, inc_dc_fw_backend_ticket_t ticket)
{
    auto *backend = static_cast<NativeCompositeBackend *>(opaque);
    uint32_t slot = 0u;
    inc_dc_fw_backend_ticket_t child_ticket{};
    if (!Resolve(backend, ticket, &slot, &child_ticket))
        return INC_DC_FW_STALE_HANDLE;
    const auto &child = slot == 0u ? backend->dispatch : backend->combine;
    return child.cancel(child.backend_context, child_ticket);
}

inc_dc_fw_status_t Release(
    void *opaque, inc_dc_fw_backend_ticket_t ticket)
{
    auto *backend = static_cast<NativeCompositeBackend *>(opaque);
    uint32_t slot = 0u;
    inc_dc_fw_backend_ticket_t child_ticket{};
    if (!Resolve(backend, ticket, &slot, &child_ticket))
        return INC_DC_FW_STALE_HANDLE;
    const auto &child = slot == 0u ? backend->dispatch : backend->combine;
    const inc_dc_fw_status_t status =
        child.release(child.backend_context, child_ticket);
    if (status != INC_DC_FW_OK) return status;
    std::lock_guard<std::mutex> lock(backend->mutex);
    TicketSlot &entry = backend->tickets[slot];
    if (!entry.live || entry.generation != ticket.generation)
        return INC_DC_FW_STALE_HANDLE;
    entry = {};
    return INC_DC_FW_OK;
}

} // namespace

inc_dc_fw_status_t CreateNativeCompositeBackend(
    const inc_dc_fw_backend_ops_t &dispatch,
    const inc_dc_fw_backend_ops_t &combine,
    NativeCompositeBackend **backend)
{
    if (backend == nullptr || *backend != nullptr || !ValidChild(dispatch) ||
        !ValidChild(combine)) return INC_DC_FW_INVALID_ARGUMENT;
    auto *created = new (std::nothrow) NativeCompositeBackend();
    if (created == nullptr) return INC_DC_FW_INTERNAL;
    created->dispatch = dispatch;
    created->combine = combine;
    *backend = created;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t DestroyNativeCompositeBackend(
    NativeCompositeBackend *backend)
{
    if (backend == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    {
        std::lock_guard<std::mutex> lock(backend->mutex);
        for (const auto &ticket : backend->tickets)
            if (ticket.live) return INC_DC_FW_BUSY;
    }
    delete backend;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t NativeCompositeBackendOps(
    NativeCompositeBackend *backend, inc_dc_fw_backend_ops_t *ops)
{
    if (backend == nullptr || ops == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    *ops = {};
    ops->struct_size = sizeof(*ops);
    ops->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    ops->backend_context = backend;
    ops->get_capabilities = GetCapabilities;
    ops->query_workspace = QueryWorkspace;
    ops->enqueue = Enqueue;
    ops->query = Query;
    ops->wait = Wait;
    ops->cancel = Cancel;
    ops->release = Release;
    return INC_DC_FW_OK;
}

} // namespace inc::dc::single_stream
