#include "inc_dc_framework_c_api.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <vector>

namespace {

constexpr uint32_t kImplementationVersion = 2u;
constexpr uint32_t kDefaultMaxInflight = 64u;
constexpr uint32_t kDefaultMaxWorkspaceQueries = 256u;
constexpr uint32_t kMaxConfiguredSlots = 1u << 20;

bool HeaderValid(uint32_t size, uint32_t expected, uint32_t version)
{
    return size >= expected && version == INC_DC_FRAMEWORK_ABI_VERSION;
}

bool OperationValid(uint32_t op)
{
    return op == INC_DC_FW_OP_DISPATCH || op == INC_DC_FW_OP_COMBINE;
}

bool DTypeValid(uint32_t dtype)
{
    return dtype == INC_DC_FW_DTYPE_FP16 ||
           dtype == INC_DC_FW_DTYPE_BF16;
}

bool RouteValid(const inc_dc_fw_route_desc_t &route, bool optional)
{
    if (route.data == nullptr) {
        return optional && route.bytes == 0u;
    }
    return HeaderValid(route.struct_size, sizeof(route), route.abi_version) &&
           route.bytes != 0u && route.semantic_digest != 0u &&
           route.generation != 0u &&
           route.memory_location <= INC_DC_FW_MEMORY_DEVICE &&
           route.format <= INC_DC_FW_ROUTE_OPAQUE_DEVICE_PLAN;
}

bool ShapeValid(const inc_dc_fw_plan_desc_t &plan,
                const inc_dc_fw_shape_t &shape)
{
    return HeaderValid(shape.struct_size, sizeof(shape), shape.abi_version) &&
           shape.hidden_size == plan.hidden_size && shape.topk != 0u &&
           shape.topk <= plan.max_topk &&
           shape.tokens <= static_cast<uint64_t>(
                               std::numeric_limits<int64_t>::max()) &&
           shape.dtype == plan.dtype && DTypeValid(shape.dtype);
}

bool TensorValid(const inc_dc_fw_tensor_desc_t &tensor, uint32_t hidden,
                 bool allow_null)
{
    if (tensor.data == nullptr) {
        return allow_null;
    }
    if (!HeaderValid(tensor.struct_size, sizeof(tensor),
                     tensor.abi_version) ||
        !DTypeValid(tensor.dtype) ||
        tensor.memory_location != INC_DC_FW_MEMORY_DEVICE ||
        tensor.ndim == 0u || tensor.ndim > INC_DC_FRAMEWORK_MAX_DIMS) {
        return false;
    }
    for (uint32_t i = 0; i < tensor.ndim; ++i) {
        if (tensor.dims[i] <= 0 || tensor.strides[i] < 0) {
            return false;
        }
    }
    return hidden == 0u ||
           tensor.dims[tensor.ndim - 1u] ==
               static_cast<int64_t>(hidden);
}

bool WeightTensorValid(const inc_dc_fw_tensor_desc_t &tensor,
                       bool allow_null)
{
    if (tensor.data == nullptr) return allow_null;
    if (!HeaderValid(tensor.struct_size, sizeof(tensor),
                     tensor.abi_version) ||
        (tensor.dtype != INC_DC_FW_DTYPE_FP16 &&
         tensor.dtype != INC_DC_FW_DTYPE_BF16 &&
         tensor.dtype != INC_DC_FW_DTYPE_FP32) ||
        tensor.memory_location != INC_DC_FW_MEMORY_DEVICE ||
        tensor.ndim != 1u || tensor.dims[0] <= 0 ||
        tensor.strides[0] != 1) {
        return false;
    }
    return true;
}

bool MetadataTensorValid(const inc_dc_fw_tensor_desc_t &tensor,
                         uint32_t dtype, uint64_t elements, bool optional)
{
    if (tensor.data == nullptr) {
        return optional;
    }
    if (!HeaderValid(tensor.struct_size, sizeof(tensor),
                     tensor.abi_version) ||
        tensor.dtype != dtype ||
        tensor.memory_location != INC_DC_FW_MEMORY_DEVICE ||
        tensor.ndim != 1u || tensor.dims[0] < 0 ||
        tensor.strides[0] != 1) {
        return false;
    }
    return static_cast<uint64_t>(tensor.dims[0]) >= elements;
}

inc_dc_fw_status_t ValidateExtensions(
    uint64_t feature_bits,
    const inc_dc_fw_invocation_t &invocation)
{
    const bool has_extensions =
        (invocation.flags & INC_DC_FW_INVOCATION_HAS_EXTENSIONS) != 0u;
    if (!has_extensions) {
        return invocation.reserved[0] == 0u ? INC_DC_FW_OK
                                            : INC_DC_FW_INVALID_ARGUMENT;
    }
    auto *header = reinterpret_cast<const inc_dc_fw_extension_header_t *>(
        static_cast<uintptr_t>(invocation.reserved[0]));
    if (header == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    uint32_t seen_types = 0u;
    for (uint32_t depth = 0u; header != nullptr; ++depth) {
        if (depth >= 16u ||
            !HeaderValid(header->struct_size, sizeof(*header),
                         header->abi_version)) {
            return INC_DC_FW_INVALID_ARGUMENT;
        }
        if (header->type == INC_DC_FW_EXT_ROUTE_METADATA_V1) {
            if ((seen_types & 1u) != 0u) return INC_DC_FW_INVALID_ARGUMENT;
            seen_types |= 1u;
            if ((feature_bits &
                 INC_DC_FW_FEATURE_ROUTE_METADATA) == 0u) {
                return INC_DC_FW_UNSUPPORTED;
            }
            auto *meta =
                reinterpret_cast<const inc_dc_fw_route_metadata_v1_t *>(
                    header);
            if (!HeaderValid(meta->header.struct_size, sizeof(*meta),
                             meta->header.abi_version) ||
                meta->logical_assignments < meta->physical_rows ||
                meta->local_expert_count ==
                    std::numeric_limits<uint64_t>::max() ||
                !MetadataTensorValid(
                    meta->expert_offsets, INC_DC_FW_DTYPE_INT64,
                    meta->local_expert_count + 1u, true) ||
                !MetadataTensorValid(
                    meta->source_token_indices, INC_DC_FW_DTYPE_INT64,
                    meta->physical_rows, true) ||
                !MetadataTensorValid(
                    meta->expert_ids, INC_DC_FW_DTYPE_INT32,
                    meta->logical_assignments, true) ||
                !MetadataTensorValid(
                    meta->route_ordinals, INC_DC_FW_DTYPE_INT32,
                    meta->logical_assignments, true) ||
                !MetadataTensorValid(
                    meta->dropped_mask, INC_DC_FW_DTYPE_UINT8,
                    meta->logical_assignments, true)) {
                return INC_DC_FW_LAYOUT_MISMATCH;
            }
            const auto &weights = meta->assignment_weights;
            if (weights.data != nullptr &&
                weights.dtype != INC_DC_FW_DTYPE_FP16 &&
                weights.dtype != INC_DC_FW_DTYPE_BF16 &&
                weights.dtype != INC_DC_FW_DTYPE_FP32) {
                return INC_DC_FW_LAYOUT_MISMATCH;
            }
            if (weights.data != nullptr &&
                (!HeaderValid(weights.struct_size, sizeof(weights),
                              weights.abi_version) ||
                 weights.memory_location != INC_DC_FW_MEMORY_DEVICE ||
                 weights.ndim != 1u || weights.strides[0] != 1 ||
                 weights.dims[0] < 0 ||
                 static_cast<uint64_t>(weights.dims[0]) <
                     meta->logical_assignments)) {
                return INC_DC_FW_LAYOUT_MISMATCH;
            }
        } else if (header->type == INC_DC_FW_EXT_EXPERT_LAYOUT_V1) {
            if ((seen_types & 2u) != 0u) return INC_DC_FW_INVALID_ARGUMENT;
            seen_types |= 2u;
            if ((feature_bits & INC_DC_FW_FEATURE_EXPERT_LAYOUT) == 0u) {
                return INC_DC_FW_UNSUPPORTED;
            }
            auto *layout =
                reinterpret_cast<const inc_dc_fw_expert_layout_v1_t *>(
                    header);
            const uint32_t alignment = layout->expert_alignment;
            if (!HeaderValid(layout->header.struct_size, sizeof(*layout),
                             layout->header.abi_version) ||
                layout->local_expert_count ==
                    std::numeric_limits<uint64_t>::max() ||
                alignment == 0u || (alignment & (alignment - 1u)) != 0u ||
                !MetadataTensorValid(
                    layout->permutation_indices, INC_DC_FW_DTYPE_INT64,
                    layout->logical_rows, true) ||
                !MetadataTensorValid(
                    layout->inverse_permutation, INC_DC_FW_DTYPE_INT64,
                    layout->logical_rows, true) ||
                !MetadataTensorValid(
                    layout->expert_offsets, INC_DC_FW_DTYPE_INT64,
                    layout->local_expert_count + 1u, true) ||
                !MetadataTensorValid(
                    layout->padded_expert_offsets, INC_DC_FW_DTYPE_INT64,
                    layout->local_expert_count + 1u, true) ||
                !MetadataTensorValid(
                    layout->tokens_per_expert, INC_DC_FW_DTYPE_INT64,
                    layout->local_expert_count, true)) {
                return INC_DC_FW_LAYOUT_MISMATCH;
            }
        } else {
            return INC_DC_FW_UNSUPPORTED;
        }
        header = header->next;
    }
    return INC_DC_FW_OK;
}

uint64_t NextGeneration(uint64_t current)
{
    ++current;
    return current == 0u ? 1u : current;
}

struct WorkspaceLease {
    bool live = false;
    uint64_t generation = 0;
    uint32_t active_requests = 0u;
    const inc_dc_plan_t *plan = nullptr;
    uint32_t operation = 0;
    inc_dc_fw_shape_t shape{};
    inc_dc_fw_workspace_t workspace{};
};

struct RequestSlot {
    bool live = false;
    bool backend_call_active = false;
    uint64_t generation = 0;
    inc_dc_plan_t *plan = nullptr;
    uint64_t workspace_query_token = 0u;
    uint64_t route_handle_token = 0u;
    uint64_t operation_generation = 0u;
    uint32_t operation = 0u;
    inc_dc_fw_backend_ticket_t ticket{};
    uint32_t state = INC_DC_FW_REQUEST_FREE;
    bool terminal_accounted = false;
};

struct RouteHandleSlot {
    bool live = false;
    uint64_t generation = 0u;
    uint32_t active_requests = 0u;
    inc_dc_plan_t *plan = nullptr;
    inc_dc_fw_route_desc_t route{};
};

} // namespace

struct inc_dc_context {
    mutable std::mutex mutex;
    inc_dc_fw_backend_ops_t backend{};
    inc_dc_fw_capabilities_t capabilities{};
    std::vector<RequestSlot> requests;
    std::vector<WorkspaceLease> leases;
    std::vector<RouteHandleSlot> route_handles;
    uint32_t live_plans = 0;
    uint32_t live_requests = 0;
    uint32_t peak_live_requests = 0;
    uint64_t dispatch_enqueued = 0u;
    uint64_t combine_enqueued = 0u;
    uint64_t completed = 0u;
    uint64_t cancelled = 0u;
    uint64_t failed = 0u;
    uint64_t wait_timeouts = 0u;
    uint64_t enqueue_failures = 0u;
};

struct inc_dc_plan {
    inc_dc_context_t *context = nullptr;
    inc_dc_fw_plan_desc_t desc{};
    std::atomic<uint32_t> refs{1u};
    std::atomic<uint32_t> inflight{0u};
    std::atomic<uint32_t> live_route_handles{0u};
};

namespace {

bool BackendValid(const inc_dc_fw_backend_ops_t &backend)
{
    return HeaderValid(backend.struct_size, sizeof(backend),
                       backend.abi_version) &&
           backend.get_capabilities != nullptr &&
           backend.query_workspace != nullptr &&
           backend.enqueue != nullptr && backend.query != nullptr &&
           backend.cancel != nullptr && backend.release != nullptr;
}

RequestSlot *FindRequestLocked(inc_dc_context_t *context,
                               inc_dc_fw_request_t request)
{
    if (request.slot == 0u || request.slot > context->requests.size()) {
        return nullptr;
    }
    RequestSlot &slot = context->requests[request.slot - 1u];
    return slot.live && slot.generation == request.generation
               ? &slot
               : nullptr;
}

WorkspaceLease *FindLeaseLocked(inc_dc_context_t *context, uint64_t token)
{
    const uint32_t index = static_cast<uint32_t>(token & 0xffffffffu);
    const uint32_t generation = static_cast<uint32_t>(token >> 32u);
    if (index == 0u || index > context->leases.size() ||
        generation == 0u) {
        return nullptr;
    }
    WorkspaceLease &lease = context->leases[index - 1u];
    return lease.live &&
                   static_cast<uint32_t>(lease.generation) == generation
               ? &lease
               : nullptr;
}

RouteHandleSlot *FindRouteHandleLocked(
    inc_dc_context_t *context, uint64_t token)
{
    const uint32_t index = static_cast<uint32_t>(token & 0xffffffffu);
    const uint32_t generation = static_cast<uint32_t>(token >> 32u);
    if (index == 0u || index > context->route_handles.size() ||
        generation == 0u) {
        return nullptr;
    }
    RouteHandleSlot &slot = context->route_handles[index - 1u];
    return slot.live &&
                   static_cast<uint32_t>(slot.generation) == generation
               ? &slot
               : nullptr;
}

void AccountTerminalLocked(inc_dc_context_t *context, RequestSlot *slot)
{
    if (slot == nullptr || slot->terminal_accounted) return;
    if (slot->state == INC_DC_FW_REQUEST_COMPLETED) {
        ++context->completed;
    } else if (slot->state == INC_DC_FW_REQUEST_CANCELLED) {
        ++context->cancelled;
    } else if (slot->state == INC_DC_FW_REQUEST_FAILED) {
        ++context->failed;
    } else {
        return;
    }
    slot->terminal_accounted = true;
}

bool SameShape(const inc_dc_fw_shape_t &a, const inc_dc_fw_shape_t &b)
{
    return a.tokens == b.tokens && a.hidden_size == b.hidden_size &&
           a.topk == b.topk && a.dtype == b.dtype && a.flags == b.flags;
}

inc_dc_fw_status_t ValidateInvocationLocked(
    inc_dc_plan_t *plan, uint32_t operation,
    const inc_dc_fw_invocation_t *invocation)
{
    if (invocation == nullptr ||
        !HeaderValid(invocation->struct_size, sizeof(*invocation),
                     invocation->abi_version) ||
        !ShapeValid(plan->desc, invocation->shape) ||
        invocation->stream == 0u ||
        invocation->operation_generation == 0u) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    const bool empty = invocation->shape.tokens == 0u;
    if (!TensorValid(invocation->input, invocation->shape.hidden_size,
                     empty) ||
        !TensorValid(invocation->output, invocation->shape.hidden_size,
                     empty) ||
        !WeightTensorValid(invocation->weights, true)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    if (invocation->weights.data != nullptr &&
        invocation->weights.dtype == INC_DC_FW_DTYPE_FP32 &&
        (plan->context->capabilities.feature_bits &
         INC_DC_FW_FEATURE_FP32_WEIGHTS) == 0u) {
        return INC_DC_FW_UNSUPPORTED;
    }
    if (invocation->input.data != nullptr &&
        invocation->input.dtype != invocation->shape.dtype) {
        return INC_DC_FW_LAYOUT_MISMATCH;
    }
    if (invocation->output.data != nullptr &&
        invocation->output.dtype != invocation->shape.dtype) {
        return INC_DC_FW_LAYOUT_MISMATCH;
    }
    if (!RouteValid(invocation->route, true) &&
        !RouteValid(plan->desc.static_route, false)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    if (invocation->route.data != nullptr &&
        invocation->route.generation != invocation->operation_generation) {
        return INC_DC_FW_STALE_HANDLE;
    }
    const inc_dc_fw_status_t extension_status =
        ValidateExtensions(plan->context->capabilities.feature_bits,
                           *invocation);
    if (extension_status != INC_DC_FW_OK) {
        return extension_status;
    }
    WorkspaceLease *lease = FindLeaseLocked(
        plan->context, invocation->workspace_query_token);
    if (lease == nullptr || lease->plan != plan ||
        lease->operation != operation ||
        !SameShape(lease->shape, invocation->shape)) {
        return INC_DC_FW_LAYOUT_MISMATCH;
    }
    if (invocation->workspace == nullptr ||
        invocation->workspace_bytes < lease->workspace.bytes ||
        (reinterpret_cast<uintptr_t>(invocation->workspace) %
             lease->workspace.alignment) != 0u) {
        return INC_DC_FW_CAPACITY_EXCEEDED;
    }
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Enqueue(inc_dc_plan_t *plan, uint32_t operation,
                           const inc_dc_fw_invocation_t *invocation,
                           inc_dc_fw_request_t *request,
                           uint64_t route_handle_token = 0u)
{
    if (plan == nullptr || request == nullptr || !OperationValid(operation)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    inc_dc_context_t *context = plan->context;
    std::unique_lock<std::mutex> lock(context->mutex);
    inc_dc_fw_status_t status =
        ValidateInvocationLocked(plan, operation, invocation);
    if (status != INC_DC_FW_OK) {
        return status;
    }
    WorkspaceLease *workspace_lease = FindLeaseLocked(
        context, invocation->workspace_query_token);
    if (workspace_lease == nullptr || workspace_lease->plan != plan) {
        return INC_DC_FW_STALE_HANDLE;
    }
    RouteHandleSlot *route_handle = nullptr;
    if (route_handle_token != 0u) {
        route_handle = FindRouteHandleLocked(context, route_handle_token);
        if (route_handle == nullptr || route_handle->plan != plan) {
            return INC_DC_FW_STALE_HANDLE;
        }
    }
    auto free_it = std::find_if(
        context->requests.begin(), context->requests.end(),
        [](const RequestSlot &slot) { return !slot.live; });
    if (free_it == context->requests.end()) {
        return INC_DC_FW_CAPACITY_EXCEEDED;
    }

    /*
     * Reserve the framework slot before entering the device backend, then
     * drop the context lock.  A slow dispatch enqueue must not prevent a
     * combine enqueue on another stream (or vice versa).  The reservation
     * also pins both plan and context until success or rollback.
     */
    const size_t slot_index =
        static_cast<size_t>(free_it - context->requests.begin());
    free_it->generation = NextGeneration(free_it->generation);
    const uint64_t slot_generation = free_it->generation;
    free_it->live = true;
    free_it->backend_call_active = true;
    free_it->plan = plan;
    free_it->workspace_query_token = invocation->workspace_query_token;
    free_it->route_handle_token = route_handle_token;
    free_it->operation_generation = invocation->operation_generation;
    free_it->operation = operation;
    free_it->ticket = {};
    free_it->state = INC_DC_FW_REQUEST_SUBMITTED;
    free_it->terminal_accounted = false;
    plan->inflight.fetch_add(1u, std::memory_order_relaxed);
    ++workspace_lease->active_requests;
    if (route_handle != nullptr) ++route_handle->active_requests;
    ++context->live_requests;
    context->peak_live_requests =
        std::max(context->peak_live_requests, context->live_requests);

    const inc_dc_fw_backend_ops_t backend = context->backend;
    const inc_dc_fw_plan_desc_t plan_desc = plan->desc;
    inc_dc_fw_backend_ticket_t ticket{};
    lock.unlock();
    status = backend.enqueue(
        backend.backend_context, &plan_desc, operation, invocation,
        &ticket);
    const bool valid_ticket =
        ticket.value != 0u && ticket.generation != 0u;
    if (status == INC_DC_FW_OK && !valid_ticket) {
        (void)backend.cancel(backend.backend_context, ticket);
        (void)backend.release(backend.backend_context, ticket);
        status = INC_DC_FW_BACKEND_ERROR;
    }
    lock.lock();

    RequestSlot &reserved = context->requests[slot_index];
    if (!reserved.live ||
        reserved.generation != slot_generation ||
        reserved.plan != plan) {
        return INC_DC_FW_INTERNAL;
    }
    reserved.backend_call_active = false;
    if (status != INC_DC_FW_OK) {
        reserved.live = false;
        reserved.plan = nullptr;
        WorkspaceLease *rollback_lease = FindLeaseLocked(
            context, reserved.workspace_query_token);
        if (rollback_lease != nullptr &&
            rollback_lease->active_requests != 0u) {
            --rollback_lease->active_requests;
        }
        reserved.workspace_query_token = 0u;
        RouteHandleSlot *rollback_route = FindRouteHandleLocked(
            context, reserved.route_handle_token);
        if (rollback_route != nullptr &&
            rollback_route->active_requests != 0u) {
            --rollback_route->active_requests;
        }
        reserved.route_handle_token = 0u;
        reserved.operation_generation = 0u;
        reserved.operation = 0u;
        reserved.ticket = {};
        reserved.state = INC_DC_FW_REQUEST_FREE;
        reserved.terminal_accounted = false;
        plan->inflight.fetch_sub(1u, std::memory_order_release);
        --context->live_requests;
        ++context->enqueue_failures;
        return status;
    }
    reserved.ticket = ticket;
    if (operation == INC_DC_FW_OP_DISPATCH) {
        ++context->dispatch_enqueued;
    } else {
        ++context->combine_enqueued;
    }
    request->slot = static_cast<uint64_t>(slot_index) + 1u;
    request->generation = slot_generation;
    return INC_DC_FW_OK;
}

} // namespace

extern "C" {

inc_dc_fw_status_t inc_dc_fw_get_version(inc_dc_fw_version_t *version)
{
    if (version == nullptr ||
        version->struct_size < sizeof(*version)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    version->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    version->implementation_version = kImplementationVersion;
    return INC_DC_FW_OK;
}

const char *inc_dc_fw_status_string(inc_dc_fw_status_t status)
{
    switch (status) {
    case INC_DC_FW_OK: return "OK";
    case INC_DC_FW_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
    case INC_DC_FW_UNSUPPORTED: return "UNSUPPORTED";
    case INC_DC_FW_CAPACITY_EXCEEDED: return "CAPACITY_EXCEEDED";
    case INC_DC_FW_LAYOUT_MISMATCH: return "LAYOUT_MISMATCH";
    case INC_DC_FW_STALE_HANDLE: return "STALE_HANDLE";
    case INC_DC_FW_NOT_READY: return "NOT_READY";
    case INC_DC_FW_TIMEOUT: return "TIMEOUT";
    case INC_DC_FW_CANCELLED: return "CANCELLED";
    case INC_DC_FW_BUSY: return "BUSY";
    case INC_DC_FW_BACKEND_ERROR: return "BACKEND_ERROR";
    default: return "INTERNAL";
    }
}

inc_dc_fw_status_t inc_dc_fw_context_create(
    const inc_dc_fw_context_config_t *config, inc_dc_context_t **context)
{
    if (config == nullptr || context == nullptr ||
        !HeaderValid(config->struct_size, sizeof(*config),
                     config->abi_version) ||
        !BackendValid(config->backend)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    const uint32_t max_inflight =
        config->max_inflight == 0u ? kDefaultMaxInflight
                                   : config->max_inflight;
    const uint32_t max_queries =
        config->max_workspace_queries == 0u
            ? kDefaultMaxWorkspaceQueries
            : config->max_workspace_queries;
    if (max_inflight > kMaxConfiguredSlots ||
        max_queries > kMaxConfiguredSlots) {
        return INC_DC_FW_CAPACITY_EXCEEDED;
    }
    auto *created = new (std::nothrow) inc_dc_context_t();
    if (created == nullptr) {
        return INC_DC_FW_INTERNAL;
    }
    try {
        created->requests.resize(max_inflight);
        created->leases.resize(max_queries);
        created->route_handles.resize(max_inflight);
    } catch (...) {
        delete created;
        return INC_DC_FW_INTERNAL;
    }
    created->backend = config->backend;
    created->capabilities.struct_size = sizeof(created->capabilities);
    created->capabilities.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    inc_dc_fw_status_t status = created->backend.get_capabilities(
        created->backend.backend_context, &created->capabilities);
    if (status != INC_DC_FW_OK ||
        !HeaderValid(created->capabilities.struct_size,
                     sizeof(created->capabilities),
                     created->capabilities.abi_version) ||
        created->capabilities.workspace_alignment == 0u ||
        (created->capabilities.workspace_alignment &
         (created->capabilities.workspace_alignment - 1u)) != 0u) {
        delete created;
        return status == INC_DC_FW_OK ? INC_DC_FW_BACKEND_ERROR : status;
    }
    created->capabilities.max_inflight =
        created->capabilities.max_inflight == 0u
            ? max_inflight
            : std::min(created->capabilities.max_inflight, max_inflight);
    *context = created;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_fw_context_get_capabilities(
    const inc_dc_context_t *context,
    inc_dc_fw_capabilities_t *capabilities)
{
    if (context == nullptr || capabilities == nullptr ||
        capabilities->struct_size < sizeof(*capabilities)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(context->mutex);
    *capabilities = context->capabilities;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_fw_context_get_stats(
    const inc_dc_context_t *context, inc_dc_fw_context_stats_t *stats)
{
    if (context == nullptr || stats == nullptr ||
        stats->struct_size < sizeof(*stats)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(context->mutex);
    const uint32_t live_leases = static_cast<uint32_t>(std::count_if(
        context->leases.begin(), context->leases.end(),
        [](const WorkspaceLease &lease) { return lease.live; }));
    const uint32_t caller_size = stats->struct_size;
    *stats = {};
    stats->struct_size = caller_size;
    stats->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    stats->dispatch_enqueued = context->dispatch_enqueued;
    stats->combine_enqueued = context->combine_enqueued;
    stats->completed = context->completed;
    stats->cancelled = context->cancelled;
    stats->failed = context->failed;
    stats->wait_timeouts = context->wait_timeouts;
    stats->enqueue_failures = context->enqueue_failures;
    stats->live_requests = context->live_requests;
    stats->peak_live_requests = context->peak_live_requests;
    stats->live_plans = context->live_plans;
    stats->live_workspace_leases = live_leases;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_fw_context_destroy(inc_dc_context_t *context)
{
    if (context == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    {
        std::lock_guard<std::mutex> lock(context->mutex);
        if (context->live_requests != 0u || context->live_plans != 0u) {
            return INC_DC_FW_BUSY;
        }
    }
    delete context;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_fw_plan_create(
    inc_dc_context_t *context, const inc_dc_fw_plan_desc_t *desc,
    inc_dc_plan_t **plan)
{
    if (context == nullptr || desc == nullptr || plan == nullptr ||
        !HeaderValid(desc->struct_size, sizeof(*desc),
                     desc->abi_version) ||
        desc->session_id == 0u || desc->model_id == 0u ||
        desc->process_group_id == 0u ||
        desc->topology_generation == 0u ||
        desc->worker_world_size < 2u ||
        desc->worker_rank >= desc->worker_world_size ||
        desc->max_tokens == 0u || desc->hidden_size == 0u ||
        desc->max_topk == 0u || !DTypeValid(desc->dtype) ||
        !RouteValid(desc->static_route, true)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    {
        std::lock_guard<std::mutex> lock(context->mutex);
        const auto &caps = context->capabilities;
        if ((caps.max_world_size != 0u &&
             desc->worker_world_size > caps.max_world_size) ||
            (caps.max_topk != 0u && desc->max_topk > caps.max_topk) ||
            (caps.dtype_bits & (1ull << desc->dtype)) == 0u) {
            return INC_DC_FW_UNSUPPORTED;
        }
    }
    auto *created = new (std::nothrow) inc_dc_plan_t();
    if (created == nullptr) {
        return INC_DC_FW_INTERNAL;
    }
    created->context = context;
    created->desc = *desc;
    {
        std::lock_guard<std::mutex> lock(context->mutex);
        ++context->live_plans;
    }
    *plan = created;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_fw_plan_retain(inc_dc_plan_t *plan)
{
    if (plan == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    uint32_t refs = plan->refs.load(std::memory_order_relaxed);
    do {
        if (refs == std::numeric_limits<uint32_t>::max()) {
            return INC_DC_FW_CAPACITY_EXCEEDED;
        }
    } while (!plan->refs.compare_exchange_weak(
        refs, refs + 1u, std::memory_order_relaxed));
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_fw_plan_release(inc_dc_plan_t *plan)
{
    if (plan == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    if (plan->inflight.load(std::memory_order_acquire) != 0u ||
        plan->live_route_handles.load(std::memory_order_acquire) != 0u) {
        return INC_DC_FW_BUSY;
    }
    const uint32_t old =
        plan->refs.fetch_sub(1u, std::memory_order_acq_rel);
    if (old == 0u) {
        plan->refs.fetch_add(1u, std::memory_order_relaxed);
        return INC_DC_FW_STALE_HANDLE;
    }
    if (old == 1u) {
        inc_dc_context_t *context = plan->context;
        {
            std::lock_guard<std::mutex> lock(context->mutex);
            for (auto &lease : context->leases) {
                if (lease.live && lease.plan == plan) {
                    lease.live = false;
                }
            }
            --context->live_plans;
        }
        delete plan;
    }
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_fw_query_workspace(
    inc_dc_plan_t *plan, uint32_t operation,
    const inc_dc_fw_shape_t *shape, inc_dc_fw_workspace_t *workspace)
{
    if (plan == nullptr || shape == nullptr || workspace == nullptr ||
        workspace->struct_size < sizeof(*workspace) ||
        !OperationValid(operation) || !ShapeValid(plan->desc, *shape)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    inc_dc_context_t *context = plan->context;
    std::lock_guard<std::mutex> lock(context->mutex);
    const auto cached = std::find_if(
        context->leases.begin(), context->leases.end(),
        [plan, operation, shape](const WorkspaceLease &lease) {
            return lease.live && lease.plan == plan &&
                   lease.operation == operation &&
                   SameShape(lease.shape, *shape);
        });
    if (cached != context->leases.end()) {
        *workspace = cached->workspace;
        return INC_DC_FW_OK;
    }
    auto free_it = std::find_if(
        context->leases.begin(), context->leases.end(),
        [](const WorkspaceLease &lease) { return !lease.live; });
    if (free_it == context->leases.end()) {
        return INC_DC_FW_CAPACITY_EXCEEDED;
    }
    inc_dc_fw_workspace_t result{};
    result.struct_size = sizeof(result);
    result.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    inc_dc_fw_status_t status = context->backend.query_workspace(
        context->backend.backend_context, &plan->desc, operation, shape,
        &result);
    if (status != INC_DC_FW_OK) {
        return status;
    }
    if (!HeaderValid(result.struct_size, sizeof(result),
                     result.abi_version) ||
        result.bytes == 0u || result.alignment == 0u ||
        (result.alignment & (result.alignment - 1u)) != 0u ||
        result.alignment < context->capabilities.workspace_alignment) {
        return INC_DC_FW_BACKEND_ERROR;
    }
    free_it->generation = NextGeneration(free_it->generation);
    const uint64_t index =
        static_cast<uint64_t>(free_it - context->leases.begin()) + 1u;
    result.query_token =
        (static_cast<uint64_t>(
             static_cast<uint32_t>(free_it->generation))
         << 32u) |
        index;
    free_it->live = true;
    free_it->active_requests = 0u;
    free_it->plan = plan;
    free_it->operation = operation;
    free_it->shape = *shape;
    free_it->workspace = result;
    *workspace = result;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_fw_workspace_release(
    inc_dc_plan_t *plan, uint64_t workspace_query_token)
{
    if (plan == nullptr || workspace_query_token == 0u) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    inc_dc_context_t *context = plan->context;
    std::lock_guard<std::mutex> lock(context->mutex);
    WorkspaceLease *lease =
        FindLeaseLocked(context, workspace_query_token);
    if (lease == nullptr || lease->plan != plan) {
        return INC_DC_FW_STALE_HANDLE;
    }
    if (lease->active_requests != 0u) {
        return INC_DC_FW_BUSY;
    }
    lease->live = false;
    lease->plan = nullptr;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_fw_dispatch_async(
    inc_dc_plan_t *plan, const inc_dc_fw_invocation_t *invocation,
    inc_dc_fw_request_t *request)
{
    return Enqueue(plan, INC_DC_FW_OP_DISPATCH, invocation, request);
}

inc_dc_fw_status_t inc_dc_fw_combine_async(
    inc_dc_plan_t *plan, const inc_dc_fw_invocation_t *invocation,
    inc_dc_fw_request_t *request)
{
    return Enqueue(plan, INC_DC_FW_OP_COMBINE, invocation, request);
}

inc_dc_fw_status_t inc_dc_fw_route_handle_create(
    inc_dc_plan_t *plan, inc_dc_fw_request_t dispatch_request,
    const inc_dc_fw_route_desc_t *route,
    inc_dc_fw_route_handle_t *route_handle)
{
    if (plan == nullptr || route_handle == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    inc_dc_context_t *context = plan->context;
    std::lock_guard<std::mutex> lock(context->mutex);
    RequestSlot *request_slot = FindRequestLocked(context, dispatch_request);
    if (request_slot == nullptr || request_slot->plan != plan ||
        request_slot->operation != INC_DC_FW_OP_DISPATCH) {
        return INC_DC_FW_STALE_HANDLE;
    }
    inc_dc_fw_route_desc_t captured =
        route == nullptr ? plan->desc.static_route : *route;
    if (!RouteValid(captured, false)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    if (route != nullptr &&
        captured.generation != request_slot->operation_generation) {
        return INC_DC_FW_STALE_HANDLE;
    }
    // Static routes are generation-independent at plan level; the opaque
    // handle makes this dispatch instance generation-specific.
    captured.generation = request_slot->operation_generation;
    auto free_it = std::find_if(
        context->route_handles.begin(), context->route_handles.end(),
        [](const RouteHandleSlot &slot) { return !slot.live; });
    if (free_it == context->route_handles.end()) {
        return INC_DC_FW_CAPACITY_EXCEEDED;
    }
    free_it->generation = NextGeneration(free_it->generation);
    free_it->live = true;
    free_it->active_requests = 0u;
    free_it->plan = plan;
    free_it->route = captured;
    plan->live_route_handles.fetch_add(1u, std::memory_order_relaxed);
    route_handle->slot =
        static_cast<uint64_t>(free_it - context->route_handles.begin()) + 1u;
    route_handle->generation = free_it->generation;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_fw_combine_with_route_async(
    inc_dc_plan_t *plan, inc_dc_fw_route_handle_t route_handle,
    const inc_dc_fw_invocation_t *invocation,
    inc_dc_fw_request_t *request)
{
    if (plan == nullptr || invocation == nullptr || route_handle.slot == 0u ||
        route_handle.generation == 0u ||
        route_handle.slot > std::numeric_limits<uint32_t>::max()) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    const uint64_t token =
        (static_cast<uint64_t>(
             static_cast<uint32_t>(route_handle.generation))
         << 32u) |
        static_cast<uint32_t>(route_handle.slot);
    inc_dc_fw_invocation_t bound = *invocation;
    {
        std::lock_guard<std::mutex> lock(plan->context->mutex);
        RouteHandleSlot *slot =
            FindRouteHandleLocked(plan->context, token);
        if (slot == nullptr || slot->plan != plan) {
            return INC_DC_FW_STALE_HANDLE;
        }
        bound.route = slot->route;
    }
    return Enqueue(
        plan, INC_DC_FW_OP_COMBINE, &bound, request, token);
}

inc_dc_fw_status_t inc_dc_fw_route_handle_release(
    inc_dc_plan_t *plan, inc_dc_fw_route_handle_t route_handle)
{
    if (plan == nullptr || route_handle.slot == 0u ||
        route_handle.generation == 0u ||
        route_handle.slot > std::numeric_limits<uint32_t>::max()) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    const uint64_t token =
        (static_cast<uint64_t>(
             static_cast<uint32_t>(route_handle.generation))
         << 32u) |
        static_cast<uint32_t>(route_handle.slot);
    std::lock_guard<std::mutex> lock(plan->context->mutex);
    RouteHandleSlot *slot = FindRouteHandleLocked(plan->context, token);
    if (slot == nullptr || slot->plan != plan) {
        return INC_DC_FW_STALE_HANDLE;
    }
    if (slot->active_requests != 0u) return INC_DC_FW_BUSY;
    slot->live = false;
    slot->plan = nullptr;
    slot->route = {};
    plan->live_route_handles.fetch_sub(1u, std::memory_order_release);
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_fw_request_query(
    inc_dc_context_t *context, inc_dc_fw_request_t request,
    uint32_t *state)
{
    if (context == nullptr || state == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(context->mutex);
    RequestSlot *slot = FindRequestLocked(context, request);
    if (slot == nullptr) {
        return INC_DC_FW_STALE_HANDLE;
    }
    if (slot->state == INC_DC_FW_REQUEST_SUBMITTED) {
        uint32_t backend_state = INC_DC_FW_REQUEST_SUBMITTED;
        inc_dc_fw_status_t status = context->backend.query(
            context->backend.backend_context, slot->ticket,
            &backend_state);
        if (status != INC_DC_FW_OK && status != INC_DC_FW_NOT_READY) {
            slot->state = INC_DC_FW_REQUEST_FAILED;
            AccountTerminalLocked(context, slot);
            *state = slot->state;
            return status;
        }
        if (status == INC_DC_FW_OK &&
            backend_state != INC_DC_FW_REQUEST_SUBMITTED) {
            if (backend_state != INC_DC_FW_REQUEST_COMPLETED &&
                backend_state != INC_DC_FW_REQUEST_CANCELLED &&
                backend_state != INC_DC_FW_REQUEST_FAILED) {
                slot->state = INC_DC_FW_REQUEST_FAILED;
                AccountTerminalLocked(context, slot);
                *state = slot->state;
                return INC_DC_FW_BACKEND_ERROR;
            }
            slot->state = backend_state;
            AccountTerminalLocked(context, slot);
        }
    }
    *state = slot->state;
    return slot->state == INC_DC_FW_REQUEST_SUBMITTED
               ? INC_DC_FW_NOT_READY
               : INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_fw_request_wait(
    inc_dc_context_t *context, inc_dc_fw_request_t request,
    uint64_t timeout_ns)
{
    if (context == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    std::unique_lock<std::mutex> lock(context->mutex);
    RequestSlot *slot = FindRequestLocked(context, request);
    if (slot == nullptr) {
        return INC_DC_FW_STALE_HANDLE;
    }
    if (slot->state != INC_DC_FW_REQUEST_SUBMITTED) {
        return slot->state == INC_DC_FW_REQUEST_COMPLETED
                   ? INC_DC_FW_OK
                   : INC_DC_FW_CANCELLED;
    }
    if (context->backend.wait == nullptr) {
        return INC_DC_FW_UNSUPPORTED;
    }
    if (slot->backend_call_active) {
        return INC_DC_FW_BUSY;
    }
    slot->backend_call_active = true;
    const inc_dc_fw_backend_ticket_t ticket = slot->ticket;
    void *backend_context = context->backend.backend_context;
    auto wait_fn = context->backend.wait;
    lock.unlock();
    const inc_dc_fw_status_t status =
        wait_fn(backend_context, ticket, timeout_ns);
    lock.lock();
    slot = FindRequestLocked(context, request);
    if (slot == nullptr) {
        return INC_DC_FW_STALE_HANDLE;
    }
    slot->backend_call_active = false;
    if (status == INC_DC_FW_OK) {
        slot->state = INC_DC_FW_REQUEST_COMPLETED;
        AccountTerminalLocked(context, slot);
    } else if (status == INC_DC_FW_TIMEOUT) {
        ++context->wait_timeouts;
    } else if (status != INC_DC_FW_TIMEOUT &&
               status != INC_DC_FW_NOT_READY) {
        slot->state = INC_DC_FW_REQUEST_FAILED;
        AccountTerminalLocked(context, slot);
    }
    return status;
}

inc_dc_fw_status_t inc_dc_fw_request_cancel(
    inc_dc_context_t *context, inc_dc_fw_request_t request)
{
    if (context == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(context->mutex);
    RequestSlot *slot = FindRequestLocked(context, request);
    if (slot == nullptr) {
        return INC_DC_FW_STALE_HANDLE;
    }
    if (slot->state != INC_DC_FW_REQUEST_SUBMITTED) {
        return INC_DC_FW_BUSY;
    }
    if (slot->backend_call_active) {
        return INC_DC_FW_BUSY;
    }
    const inc_dc_fw_status_t status = context->backend.cancel(
        context->backend.backend_context, slot->ticket);
    if (status == INC_DC_FW_OK) {
        slot->state = INC_DC_FW_REQUEST_CANCELLED;
        AccountTerminalLocked(context, slot);
    }
    return status;
}

inc_dc_fw_status_t inc_dc_fw_request_release(
    inc_dc_context_t *context, inc_dc_fw_request_t request)
{
    if (context == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(context->mutex);
    RequestSlot *slot = FindRequestLocked(context, request);
    if (slot == nullptr) {
        return INC_DC_FW_STALE_HANDLE;
    }
    if (slot->state == INC_DC_FW_REQUEST_SUBMITTED) {
        return INC_DC_FW_NOT_READY;
    }
    if (slot->backend_call_active) {
        return INC_DC_FW_BUSY;
    }
    const inc_dc_fw_status_t status = context->backend.release(
        context->backend.backend_context, slot->ticket);
    if (status != INC_DC_FW_OK) {
        return status;
    }
    WorkspaceLease *lease = FindLeaseLocked(
        context, slot->workspace_query_token);
    if (lease == nullptr || lease->active_requests == 0u) {
        return INC_DC_FW_INTERNAL;
    }
    --lease->active_requests;
    if (slot->route_handle_token != 0u) {
        RouteHandleSlot *route_handle = FindRouteHandleLocked(
            context, slot->route_handle_token);
        if (route_handle == nullptr || route_handle->active_requests == 0u) {
            return INC_DC_FW_INTERNAL;
        }
        --route_handle->active_requests;
    }
    slot->plan->inflight.fetch_sub(1u, std::memory_order_release);
    slot->live = false;
    slot->backend_call_active = false;
    slot->plan = nullptr;
    slot->workspace_query_token = 0u;
    slot->route_handle_token = 0u;
    slot->operation_generation = 0u;
    slot->operation = 0u;
    slot->ticket = {};
    slot->state = INC_DC_FW_REQUEST_FREE;
    slot->terminal_accounted = false;
    --context->live_requests;
    return INC_DC_FW_OK;
}

} // extern "C"
