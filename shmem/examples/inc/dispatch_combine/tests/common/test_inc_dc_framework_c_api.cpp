#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>

#include "inc_dc_framework_c_api.h"

#define CHECK(x)                                                          \
    do {                                                                  \
        if (!(x)) {                                                       \
            std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #x,       \
                         __FILE__, __LINE__);                              \
            std::abort();                                                 \
        }                                                                 \
    } while (0)

namespace {

struct Ticket {
    bool live = false;
    uint64_t generation = 0;
    uint32_t state = INC_DC_FW_REQUEST_FREE;
    uint64_t stream = 0;
    uint32_t operation = 0;
};

struct Backend {
    std::array<Ticket, 64> tickets{};
    uint64_t next_generation = 1;
    uint32_t enqueue_count = 0;
    uint32_t cancel_count = 0;
    uint32_t release_count = 0;
    bool synchronized_device = false;
    std::mutex wait_mutex;
    std::condition_variable wait_cv;
    bool wait_entered = false;
    bool unblock_wait = false;
    std::mutex enqueue_mutex;
    std::condition_variable enqueue_cv;
    bool block_combine_enqueue = false;
    bool combine_enqueue_entered = false;
    bool unblock_combine_enqueue = false;
    inc_dc_fw_status_t next_enqueue_status = INC_DC_FW_OK;
    bool next_enqueue_invalid_ticket = false;
};

inc_dc_fw_status_t GetCapabilities(
    void *, inc_dc_fw_capabilities_t *caps)
{
    if (caps == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    caps->struct_size = sizeof(*caps);
    caps->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    caps->feature_bits =
        INC_DC_FW_FEATURE_EXTERNAL_STREAM |
        INC_DC_FW_FEATURE_MULTI_INFLIGHT |
        INC_DC_FW_FEATURE_DEVICE_ROUTE |
        INC_DC_FW_FEATURE_DYNAMIC_TOKENS |
        INC_DC_FW_FEATURE_DISPATCH_COMBINE_OVERLAP |
        INC_DC_FW_FEATURE_GRAPH_SAFE_ENQUEUE |
        INC_DC_FW_FEATURE_CANCEL |
        INC_DC_FW_FEATURE_FUSED_OPERATOR_READY |
        INC_DC_FW_FEATURE_ROUTE_METADATA |
        INC_DC_FW_FEATURE_LOCAL_EXPERT_EXPAND |
        INC_DC_FW_FEATURE_FP32_WEIGHTS |
        INC_DC_FW_FEATURE_EXPERT_LAYOUT;
    caps->max_world_size = 128;
    caps->max_topk = 64;
    caps->max_inflight = 64;
    caps->workspace_alignment = 256;
    caps->dtype_bits =
        (1ull << INC_DC_FW_DTYPE_FP16) |
        (1ull << INC_DC_FW_DTYPE_BF16);
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t QueryWorkspace(
    void *, const inc_dc_fw_plan_desc_t *, uint32_t operation,
    const inc_dc_fw_shape_t *shape, inc_dc_fw_workspace_t *workspace)
{
    if (shape == nullptr || workspace == nullptr ||
        (operation != INC_DC_FW_OP_DISPATCH &&
         operation != INC_DC_FW_OP_COMBINE)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    workspace->struct_size = sizeof(*workspace);
    workspace->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    workspace->bytes =
        ((4096ull + static_cast<uint64_t>(shape->tokens) *
                        shape->hidden_size * 2ull +
          255ull) /
         256ull) *
        256ull;
    workspace->alignment = 256;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Enqueue(
    void *opaque, const inc_dc_fw_plan_desc_t *, uint32_t operation,
    const inc_dc_fw_invocation_t *invocation,
    inc_dc_fw_backend_ticket_t *ticket)
{
    auto *backend = static_cast<Backend *>(opaque);
    if (backend == nullptr || invocation == nullptr || ticket == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    if (operation == INC_DC_FW_OP_COMBINE) {
        std::unique_lock<std::mutex> lock(backend->enqueue_mutex);
        if (backend->block_combine_enqueue) {
            backend->combine_enqueue_entered = true;
            backend->enqueue_cv.notify_all();
            backend->enqueue_cv.wait(lock, [backend] {
                return backend->unblock_combine_enqueue;
            });
        }
    }
    std::lock_guard<std::mutex> lock(backend->enqueue_mutex);
    if (backend->next_enqueue_status != INC_DC_FW_OK) {
        const inc_dc_fw_status_t status = backend->next_enqueue_status;
        backend->next_enqueue_status = INC_DC_FW_OK;
        return status;
    }
    if (backend->next_enqueue_invalid_ticket) {
        backend->next_enqueue_invalid_ticket = false;
        *ticket = {};
        return INC_DC_FW_OK;
    }
    for (size_t i = 0; i < backend->tickets.size(); ++i) {
        auto &entry = backend->tickets[i];
        if (!entry.live) {
            entry.live = true;
            entry.generation = backend->next_generation++;
            entry.state = INC_DC_FW_REQUEST_SUBMITTED;
            entry.stream = invocation->stream;
            entry.operation = operation;
            ticket->value = i + 1u;
            ticket->generation = entry.generation;
            ++backend->enqueue_count;
            return INC_DC_FW_OK;
        }
    }
    return INC_DC_FW_CAPACITY_EXCEEDED;
}

Ticket *Find(Backend *backend, inc_dc_fw_backend_ticket_t ticket)
{
    if (backend == nullptr || ticket.value == 0u ||
        ticket.value > backend->tickets.size()) {
        return nullptr;
    }
    Ticket &entry = backend->tickets[ticket.value - 1u];
    return entry.live && entry.generation == ticket.generation
               ? &entry
               : nullptr;
}

inc_dc_fw_status_t Query(
    void *opaque, inc_dc_fw_backend_ticket_t ticket, uint32_t *state)
{
    auto *entry = Find(static_cast<Backend *>(opaque), ticket);
    if (entry == nullptr || state == nullptr) {
        return INC_DC_FW_STALE_HANDLE;
    }
    *state = entry->state;
    return entry->state == INC_DC_FW_REQUEST_SUBMITTED
               ? INC_DC_FW_NOT_READY
               : INC_DC_FW_OK;
}

inc_dc_fw_status_t Wait(
    void *opaque, inc_dc_fw_backend_ticket_t ticket, uint64_t timeout_ns)
{
    auto *entry = Find(static_cast<Backend *>(opaque), ticket);
    if (entry == nullptr) {
        return INC_DC_FW_STALE_HANDLE;
    }
    if (timeout_ns == 1u) {
        return INC_DC_FW_TIMEOUT;
    }
    if (timeout_ns == 42u) {
        std::unique_lock<std::mutex> lock(
            static_cast<Backend *>(opaque)->wait_mutex);
        static_cast<Backend *>(opaque)->wait_entered = true;
        static_cast<Backend *>(opaque)->wait_cv.notify_all();
        static_cast<Backend *>(opaque)->wait_cv.wait(
            lock, [opaque] {
                return static_cast<Backend *>(opaque)->unblock_wait;
            });
    }
    entry->state = INC_DC_FW_REQUEST_COMPLETED;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Cancel(
    void *opaque, inc_dc_fw_backend_ticket_t ticket)
{
    auto *backend = static_cast<Backend *>(opaque);
    auto *entry = Find(backend, ticket);
    if (entry == nullptr) {
        return INC_DC_FW_STALE_HANDLE;
    }
    entry->state = INC_DC_FW_REQUEST_CANCELLED;
    ++backend->cancel_count;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Release(
    void *opaque, inc_dc_fw_backend_ticket_t ticket)
{
    auto *backend = static_cast<Backend *>(opaque);
    auto *entry = Find(backend, ticket);
    if (entry == nullptr ||
        entry->state == INC_DC_FW_REQUEST_SUBMITTED) {
        return INC_DC_FW_NOT_READY;
    }
    entry->live = false;
    ++backend->release_count;
    return INC_DC_FW_OK;
}

inc_dc_fw_backend_ops_t Ops(Backend *backend)
{
    inc_dc_fw_backend_ops_t ops{};
    ops.struct_size = sizeof(ops);
    ops.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    ops.backend_context = backend;
    ops.get_capabilities = GetCapabilities;
    ops.query_workspace = QueryWorkspace;
    ops.enqueue = Enqueue;
    ops.query = Query;
    ops.wait = Wait;
    ops.cancel = Cancel;
    ops.release = Release;
    return ops;
}

inc_dc_fw_route_desc_t Route(const void *data, uint64_t generation)
{
    inc_dc_fw_route_desc_t route{};
    route.struct_size = sizeof(route);
    route.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    route.format = INC_DC_FW_ROUTE_CSR;
    route.memory_location = INC_DC_FW_MEMORY_DEVICE;
    route.data = data;
    route.bytes = 4096;
    route.semantic_digest = 0x12340000ull + generation;
    route.generation = generation;
    return route;
}

inc_dc_fw_shape_t Shape(uint32_t tokens, uint32_t hidden, uint32_t topk)
{
    inc_dc_fw_shape_t shape{};
    shape.struct_size = sizeof(shape);
    shape.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    shape.tokens = tokens;
    shape.hidden_size = hidden;
    shape.topk = topk;
    shape.dtype = INC_DC_FW_DTYPE_FP16;
    return shape;
}

inc_dc_fw_tensor_desc_t Tensor(void *data, int64_t rows, int64_t hidden)
{
    inc_dc_fw_tensor_desc_t tensor{};
    tensor.struct_size = sizeof(tensor);
    tensor.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    tensor.data = data;
    tensor.dtype = INC_DC_FW_DTYPE_FP16;
    tensor.memory_location = INC_DC_FW_MEMORY_DEVICE;
    tensor.ndim = 2;
    tensor.dims[0] = rows;
    tensor.dims[1] = hidden;
    tensor.strides[0] = hidden;
    tensor.strides[1] = 1;
    return tensor;
}

inc_dc_fw_tensor_desc_t MetaTensor(
    void *data, uint32_t dtype, int64_t elements)
{
    inc_dc_fw_tensor_desc_t tensor{};
    tensor.struct_size = sizeof(tensor);
    tensor.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    tensor.data = data;
    tensor.dtype = dtype;
    tensor.memory_location = INC_DC_FW_MEMORY_DEVICE;
    tensor.ndim = 1;
    tensor.dims[0] = elements;
    tensor.strides[0] = 1;
    return tensor;
}

inc_dc_fw_invocation_t Invocation(
    const inc_dc_fw_shape_t &shape, void *storage,
    const inc_dc_fw_workspace_t &workspace, uint64_t stream,
    uint64_t generation)
{
    inc_dc_fw_invocation_t invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    invocation.shape = shape;
    invocation.input = Tensor(
        storage, shape.tokens == 0u ? 1 : shape.tokens,
        shape.hidden_size);
    invocation.output = Tensor(
        storage, shape.tokens == 0u ? 1 : shape.tokens,
        shape.hidden_size);
    invocation.route = Route(storage, generation);
    invocation.workspace = storage;
    invocation.workspace_bytes = workspace.bytes;
    invocation.workspace_query_token = workspace.query_token;
    invocation.stream = stream;
    invocation.operation_generation = generation;
    return invocation;
}

inc_dc_plan_t *CreatePlan(
    inc_dc_context_t *context, uint32_t world, uint32_t rank,
    uint32_t max_topk)
{
    static uint8_t route_storage[4096];
    inc_dc_fw_plan_desc_t desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    desc.session_id = 10u + world;
    desc.model_id = 20u;
    desc.process_group_id = 30u;
    desc.topology_generation = 1u;
    desc.worker_world_size = world;
    desc.worker_rank = rank;
    desc.max_tokens = 4096;
    desc.hidden_size = 7168;
    desc.max_topk = max_topk;
    desc.dtype = INC_DC_FW_DTYPE_FP16;
    desc.static_route = Route(route_storage, 1u);
    inc_dc_plan_t *plan = nullptr;
    CHECK(inc_dc_fw_plan_create(context, &desc, &plan) ==
          INC_DC_FW_OK);
    return plan;
}

} // namespace

int main()
{
    inc_dc_fw_version_t version{};
    version.struct_size = sizeof(version);
    CHECK(inc_dc_fw_get_version(&version) == INC_DC_FW_OK);
    CHECK(version.abi_version == INC_DC_FRAMEWORK_ABI_VERSION);
    CHECK(inc_dc_fw_get_version(nullptr) ==
          INC_DC_FW_INVALID_ARGUMENT);

    Backend backend{};
    inc_dc_fw_context_config_t config{};
    config.struct_size = sizeof(config);
    config.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    config.max_inflight = 8;
    config.max_workspace_queries = 32;
    config.backend = Ops(&backend);
    inc_dc_context_t *context = nullptr;
    CHECK(inc_dc_fw_context_create(&config, &context) == INC_DC_FW_OK);

    inc_dc_fw_capabilities_t caps{};
    caps.struct_size = sizeof(caps);
    CHECK(inc_dc_fw_context_get_capabilities(context, &caps) ==
          INC_DC_FW_OK);
    CHECK(caps.max_inflight == 8);
    CHECK((caps.feature_bits &
           INC_DC_FW_FEATURE_DISPATCH_COMBINE_OVERLAP) != 0u);

    // W2/W4/W8 and arbitrary legal K, including K > W.
    for (uint32_t world : {2u, 4u, 8u}) {
        for (uint32_t topk : {1u, 2u, 4u, 6u, 8u, 16u}) {
            inc_dc_plan_t *plan = CreatePlan(context, world, 0u, topk);
            CHECK(inc_dc_fw_plan_release(plan) == INC_DC_FW_OK);
        }
    }

    inc_dc_plan_t *plan = CreatePlan(context, 8u, 3u, 16u);
    const auto shape = Shape(32u, 7168u, 6u);
    inc_dc_fw_workspace_t dispatch_ws{};
    dispatch_ws.struct_size = sizeof(dispatch_ws);
    inc_dc_fw_workspace_t combine_ws{};
    combine_ws.struct_size = sizeof(combine_ws);
    CHECK(inc_dc_fw_query_workspace(
              plan, INC_DC_FW_OP_DISPATCH, &shape, &dispatch_ws) ==
          INC_DC_FW_OK);
    CHECK(inc_dc_fw_query_workspace(
              plan, INC_DC_FW_OP_COMBINE, &shape, &combine_ws) ==
          INC_DC_FW_OK);

    void *storage = nullptr;
    const uint64_t allocation =
        dispatch_ws.bytes > combine_ws.bytes ? dispatch_ws.bytes
                                             : combine_ws.bytes;
    CHECK(posix_memalign(&storage, 256u, allocation) == 0);
    auto dispatch =
        Invocation(shape, storage, dispatch_ws, 0x1000u, 101u);
    auto combine =
        Invocation(shape, storage, combine_ws, 0x2000u, 102u);
    inc_dc_fw_request_t dispatch_request{}, combine_request{};
    CHECK(inc_dc_fw_dispatch_async(plan, &dispatch, &dispatch_request) ==
          INC_DC_FW_OK);
    CHECK(inc_dc_fw_combine_async(plan, &combine, &combine_request) ==
          INC_DC_FW_OK);
    CHECK(dispatch_request.slot != combine_request.slot);
    CHECK(inc_dc_fw_workspace_release(
              plan, dispatch_ws.query_token) == INC_DC_FW_BUSY);
    CHECK(inc_dc_fw_workspace_release(
              plan, combine_ws.query_token) == INC_DC_FW_BUSY);
    CHECK(backend.enqueue_count == 2u);
    CHECK(!backend.synchronized_device);
    CHECK(inc_dc_fw_plan_release(plan) == INC_DC_FW_BUSY);

    uint32_t state = INC_DC_FW_REQUEST_FREE;
    CHECK(inc_dc_fw_request_query(context, dispatch_request, &state) ==
          INC_DC_FW_NOT_READY);
    CHECK(state == INC_DC_FW_REQUEST_SUBMITTED);
    CHECK(inc_dc_fw_request_wait(context, dispatch_request, 1u) ==
          INC_DC_FW_TIMEOUT);
    CHECK(inc_dc_fw_request_wait(context, dispatch_request, 1000u) ==
          INC_DC_FW_OK);
    CHECK(inc_dc_fw_request_release(context, dispatch_request) ==
          INC_DC_FW_OK);
    CHECK(inc_dc_fw_request_query(context, dispatch_request, &state) ==
          INC_DC_FW_STALE_HANDLE);

    CHECK(inc_dc_fw_request_cancel(context, combine_request) ==
          INC_DC_FW_OK);
    CHECK(inc_dc_fw_request_release(context, combine_request) ==
          INC_DC_FW_OK);

    // A cancelled operation must not poison the next generation.
    inc_dc_fw_request_t recovered{};
    combine.operation_generation++;
    combine.route.generation++;
    CHECK(inc_dc_fw_combine_async(plan, &combine, &recovered) ==
          INC_DC_FW_OK);
    CHECK(inc_dc_fw_request_wait(context, recovered, 1000u) ==
          INC_DC_FW_OK);
    CHECK(inc_dc_fw_request_release(context, recovered) ==
          INC_DC_FW_OK);

    // Dispatch returns one opaque reverse-route identity.  It survives
    // dispatch request release, pins the plan, and cannot be released while
    // a combine request is consuming it.
    dispatch.operation_generation = 200u;
    dispatch.route.generation = 200u;
    inc_dc_fw_request_t routed_dispatch{};
    CHECK(inc_dc_fw_dispatch_async(
              plan, &dispatch, &routed_dispatch) == INC_DC_FW_OK);
    inc_dc_fw_route_handle_t route_handle{};
    CHECK(inc_dc_fw_route_handle_create(
              plan, routed_dispatch, &dispatch.route, &route_handle) ==
          INC_DC_FW_OK);
    CHECK(inc_dc_fw_plan_release(plan) == INC_DC_FW_BUSY);
    CHECK(inc_dc_fw_request_wait(
              context, routed_dispatch, 1000u) == INC_DC_FW_OK);
    CHECK(inc_dc_fw_request_release(context, routed_dispatch) ==
          INC_DC_FW_OK);
    combine.operation_generation = 200u;
    combine.route.generation = 999u; // ignored in favour of opaque handle
    inc_dc_fw_request_t routed_combine{};
    CHECK(inc_dc_fw_combine_with_route_async(
              plan, route_handle, &combine, &routed_combine) ==
          INC_DC_FW_OK);
    CHECK(inc_dc_fw_route_handle_release(plan, route_handle) ==
          INC_DC_FW_BUSY);
    CHECK(inc_dc_fw_request_wait(
              context, routed_combine, 1000u) == INC_DC_FW_OK);
    CHECK(inc_dc_fw_request_release(context, routed_combine) ==
          INC_DC_FW_OK);
    CHECK(inc_dc_fw_route_handle_release(plan, route_handle) ==
          INC_DC_FW_OK);
    inc_dc_fw_request_t stale_route_request{};
    CHECK(inc_dc_fw_combine_with_route_async(
              plan, route_handle, &combine, &stale_route_request) ==
          INC_DC_FW_STALE_HANDLE);
    combine.route.generation = combine.operation_generation;

    // An explicit wait on one request must not serialize enqueue on another
    // stream.  This is required for dispatch/combine overlap.
    inc_dc_fw_request_t blocked{}, concurrent{};
    combine.operation_generation++;
    combine.route.generation++;
    CHECK(inc_dc_fw_combine_async(plan, &combine, &blocked) ==
          INC_DC_FW_OK);
    std::thread waiter([&] {
        CHECK(inc_dc_fw_request_wait(context, blocked, 42u) ==
              INC_DC_FW_OK);
    });
    {
        std::unique_lock<std::mutex> lock(backend.wait_mutex);
        backend.wait_cv.wait(lock, [&] { return backend.wait_entered; });
    }
    dispatch.operation_generation++;
    dispatch.route.generation++;
    CHECK(inc_dc_fw_dispatch_async(plan, &dispatch, &concurrent) ==
          INC_DC_FW_OK);
    {
        std::lock_guard<std::mutex> lock(backend.wait_mutex);
        backend.unblock_wait = true;
    }
    backend.wait_cv.notify_all();
    waiter.join();
    CHECK(inc_dc_fw_request_wait(context, concurrent, 1000u) ==
          INC_DC_FW_OK);
    CHECK(inc_dc_fw_request_release(context, blocked) ==
          INC_DC_FW_OK);
    CHECK(inc_dc_fw_request_release(context, concurrent) ==
          INC_DC_FW_OK);

    // A backend enqueue that is delayed on one stream must not hold the
    // framework context lock or block an enqueue on another stream.
    backend.block_combine_enqueue = true;
    backend.combine_enqueue_entered = false;
    backend.unblock_combine_enqueue = false;
    inc_dc_fw_request_t slow_enqueue{}, parallel_enqueue{};
    combine.operation_generation++;
    combine.route.generation++;
    std::thread enqueue_thread([&] {
        CHECK(inc_dc_fw_combine_async(
                  plan, &combine, &slow_enqueue) == INC_DC_FW_OK);
    });
    {
        std::unique_lock<std::mutex> lock(backend.enqueue_mutex);
        backend.enqueue_cv.wait(
            lock, [&] { return backend.combine_enqueue_entered; });
    }
    dispatch.operation_generation++;
    dispatch.route.generation++;
    CHECK(inc_dc_fw_dispatch_async(
              plan, &dispatch, &parallel_enqueue) == INC_DC_FW_OK);
    {
        std::lock_guard<std::mutex> lock(backend.enqueue_mutex);
        backend.unblock_combine_enqueue = true;
    }
    backend.enqueue_cv.notify_all();
    enqueue_thread.join();
    CHECK(inc_dc_fw_request_wait(
              context, slow_enqueue, 1000u) == INC_DC_FW_OK);
    CHECK(inc_dc_fw_request_wait(
              context, parallel_enqueue, 1000u) == INC_DC_FW_OK);
    CHECK(inc_dc_fw_request_release(context, slow_enqueue) ==
          INC_DC_FW_OK);
    CHECK(inc_dc_fw_request_release(context, parallel_enqueue) ==
          INC_DC_FW_OK);
    backend.block_combine_enqueue = false;

    // Backend failure and malformed-success ticket both roll back the
    // reserved framework slot and workspace pin.  The next generation must
    // remain usable rather than inheriting a poisoned reservation.
    dispatch.operation_generation++;
    dispatch.route.generation++;
    backend.next_enqueue_status = INC_DC_FW_BACKEND_ERROR;
    inc_dc_fw_request_t failed_enqueue{};
    CHECK(inc_dc_fw_dispatch_async(
              plan, &dispatch, &failed_enqueue) == INC_DC_FW_BACKEND_ERROR);
    backend.next_enqueue_invalid_ticket = true;
    CHECK(inc_dc_fw_dispatch_async(
              plan, &dispatch, &failed_enqueue) == INC_DC_FW_BACKEND_ERROR);
    CHECK(inc_dc_fw_dispatch_async(
              plan, &dispatch, &failed_enqueue) == INC_DC_FW_OK);
    CHECK(inc_dc_fw_request_wait(
              context, failed_enqueue, 1000u) == INC_DC_FW_OK);
    CHECK(inc_dc_fw_request_release(context, failed_enqueue) ==
          INC_DC_FW_OK);

    // Repeated dynamic-shape query is cached and query leases are recyclable.
    inc_dc_fw_workspace_t cached_ws{};
    cached_ws.struct_size = sizeof(cached_ws);
    CHECK(inc_dc_fw_query_workspace(
              plan, INC_DC_FW_OP_DISPATCH, &shape, &cached_ws) ==
          INC_DC_FW_OK);
    CHECK(cached_ws.query_token == dispatch_ws.query_token);
    CHECK(inc_dc_fw_workspace_release(
              plan, dispatch_ws.query_token) == INC_DC_FW_OK);
    CHECK(inc_dc_fw_workspace_release(
              plan, dispatch_ws.query_token) ==
          INC_DC_FW_STALE_HANDLE);

    // Fail closed on wrong workspace binding, bad stream, and bad shape.
    auto bad = dispatch;
    bad.workspace_query_token = combine_ws.query_token;
    inc_dc_fw_request_t unused{};
    CHECK(inc_dc_fw_dispatch_async(plan, &bad, &unused) ==
          INC_DC_FW_LAYOUT_MISMATCH);
    bad = combine;
    bad.stream = 0u;
    CHECK(inc_dc_fw_dispatch_async(plan, &bad, &unused) ==
          INC_DC_FW_INVALID_ARGUMENT);
    bad = combine;
    bad.shape.topk = 17u;
    CHECK(inc_dc_fw_dispatch_async(plan, &bad, &unused) ==
          INC_DC_FW_INVALID_ARGUMENT);
    bad = combine;
    bad.workspace_bytes = combine_ws.bytes - 1u;
    CHECK(inc_dc_fw_combine_async(plan, &bad, &unused) ==
          INC_DC_FW_CAPACITY_EXCEEDED);

    // Dynamic routes are generation-bound, and framework-facing expert
    // metadata fails on the host before an invalid descriptor reaches the
    // asynchronous backend.
    bad = combine;
    ++bad.operation_generation;
    CHECK(inc_dc_fw_combine_async(plan, &bad, &unused) ==
          INC_DC_FW_STALE_HANDLE);

    inc_dc_fw_route_metadata_v1_t metadata{};
    metadata.header.struct_size = sizeof(metadata);
    metadata.header.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    metadata.header.type = INC_DC_FW_EXT_ROUTE_METADATA_V1;
    metadata.local_expert_count = 2u;
    metadata.physical_rows = 4u;
    metadata.logical_assignments = 6u;
    metadata.expert_offsets = MetaTensor(
        storage, INC_DC_FW_DTYPE_INT64, 3);
    metadata.source_token_indices = MetaTensor(
        storage, INC_DC_FW_DTYPE_INT64, 4);
    metadata.expert_ids = MetaTensor(
        storage, INC_DC_FW_DTYPE_INT32, 6);
    metadata.route_ordinals = MetaTensor(
        storage, INC_DC_FW_DTYPE_INT32, 6);
    metadata.assignment_weights = MetaTensor(
        storage, INC_DC_FW_DTYPE_FP32, 6);
    metadata.dropped_mask = MetaTensor(
        storage, INC_DC_FW_DTYPE_UINT8, 6);
    auto extended = combine;
    extended.flags |= INC_DC_FW_INVOCATION_HAS_EXTENSIONS;
    extended.reserved[0] = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(&metadata.header));
    inc_dc_fw_request_t extended_request{};
    CHECK(inc_dc_fw_combine_async(
              plan, &extended, &extended_request) == INC_DC_FW_OK);
    CHECK(inc_dc_fw_request_wait(
              context, extended_request, 1000u) == INC_DC_FW_OK);
    CHECK(inc_dc_fw_request_release(context, extended_request) ==
          INC_DC_FW_OK);

    inc_dc_fw_expert_layout_v1_t expert_layout{};
    expert_layout.header.struct_size = sizeof(expert_layout);
    expert_layout.header.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    expert_layout.header.type = INC_DC_FW_EXT_EXPERT_LAYOUT_V1;
    expert_layout.permutation_indices = MetaTensor(
        storage, INC_DC_FW_DTYPE_INT64, 6);
    expert_layout.inverse_permutation = MetaTensor(
        storage, INC_DC_FW_DTYPE_INT64, 6);
    expert_layout.expert_offsets = MetaTensor(
        storage, INC_DC_FW_DTYPE_INT64, 3);
    expert_layout.padded_expert_offsets = MetaTensor(
        storage, INC_DC_FW_DTYPE_INT64, 3);
    expert_layout.tokens_per_expert = MetaTensor(
        storage, INC_DC_FW_DTYPE_INT64, 2);
    expert_layout.local_expert_count = 2u;
    expert_layout.logical_rows = 6u;
    expert_layout.expert_alignment = 16u;
    metadata.header.next = &expert_layout.header;
    extended.operation_generation++;
    extended.route.generation++;
    CHECK(inc_dc_fw_combine_async(
              plan, &extended, &extended_request) == INC_DC_FW_OK);
    CHECK(inc_dc_fw_request_wait(
              context, extended_request, 1000u) == INC_DC_FW_OK);
    CHECK(inc_dc_fw_request_release(context, extended_request) ==
          INC_DC_FW_OK);
    expert_layout.expert_alignment = 3u;
    CHECK(inc_dc_fw_combine_async(plan, &extended, &unused) ==
          INC_DC_FW_LAYOUT_MISMATCH);
    expert_layout.expert_alignment = 16u;
    metadata.header.next = nullptr;
    metadata.expert_ids.dtype = INC_DC_FW_DTYPE_FP16;
    CHECK(inc_dc_fw_combine_async(plan, &extended, &unused) ==
          INC_DC_FW_LAYOUT_MISMATCH);
    metadata.expert_ids.dtype = INC_DC_FW_DTYPE_INT32;
    extended.flags &= ~INC_DC_FW_INVOCATION_HAS_EXTENSIONS;
    CHECK(inc_dc_fw_combine_async(plan, &extended, &unused) ==
          INC_DC_FW_INVALID_ARGUMENT);

    inc_dc_fw_context_stats_t stats{};
    stats.struct_size = sizeof(stats);
    CHECK(inc_dc_fw_context_get_stats(context, &stats) == INC_DC_FW_OK);
    CHECK(stats.abi_version == INC_DC_FRAMEWORK_ABI_VERSION);
    CHECK(stats.dispatch_enqueued >= 3u);
    CHECK(stats.combine_enqueued >= 5u);
    CHECK(stats.completed >= 7u);
    CHECK(stats.cancelled == 1u);
    CHECK(stats.failed == 0u);
    CHECK(stats.wait_timeouts == 1u);
    CHECK(stats.enqueue_failures == 2u);
    CHECK(stats.live_requests == 0u);
    CHECK(stats.peak_live_requests >= 2u);
    CHECK(stats.live_plans == 1u);
    CHECK(stats.live_workspace_leases == 1u);

    CHECK(inc_dc_fw_context_destroy(context) == INC_DC_FW_BUSY);
    CHECK(inc_dc_fw_plan_release(plan) == INC_DC_FW_OK);
    CHECK(inc_dc_fw_context_destroy(context) == INC_DC_FW_OK);
    std::free(storage);
    std::puts("inc_dc_framework_c_api: PASS");
    return 0;
}
