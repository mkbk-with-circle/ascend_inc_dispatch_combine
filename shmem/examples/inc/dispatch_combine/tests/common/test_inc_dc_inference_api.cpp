#include "inc_dc_single_inc_api.h"

#include <cassert>
#include <cstdlib>
#include <vector>

namespace {

struct Ticket { uint64_t generation = 0u; uint32_t state = INC_DC_FW_REQUEST_FREE; };
struct Backend {
    std::vector<Ticket> tickets = std::vector<Ticket>(8);
    uint64_t next_generation = 1u;
    uint64_t allocations = 0u;
    uint64_t frees = 0u;
    uint64_t enqueues = 0u;
};

struct Control {
    uint64_t before_dispatch = 0u;
    uint64_t after_dispatch = 0u;
    uint64_t before_combine = 0u;
    uint64_t after_combine = 0u;
};

inc_dc_fw_status_t Before(void *opaque, uint32_t operation,
                          uint64_t generation)
{
    auto *control = static_cast<Control *>(opaque);
    assert(control != nullptr && generation != 0u);
    if (operation == INC_DC_FW_OP_DISPATCH) ++control->before_dispatch;
    else if (operation == INC_DC_FW_OP_COMBINE) ++control->before_combine;
    else return INC_DC_FW_INVALID_ARGUMENT;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t After(void *opaque, uint32_t operation,
                         uint64_t generation)
{
    auto *control = static_cast<Control *>(opaque);
    assert(control != nullptr && generation != 0u);
    if (operation == INC_DC_FW_OP_DISPATCH) ++control->after_dispatch;
    else if (operation == INC_DC_FW_OP_COMBINE) ++control->after_combine;
    else return INC_DC_FW_INVALID_ARGUMENT;
    return INC_DC_FW_OK;
}

Ticket *Find(Backend *backend, inc_dc_fw_backend_ticket_t ticket)
{
    if (ticket.value == 0u || ticket.value > backend->tickets.size()) return nullptr;
    Ticket &found = backend->tickets[ticket.value - 1u];
    return found.generation == ticket.generation ? &found : nullptr;
}

inc_dc_fw_status_t Capabilities(void *, inc_dc_fw_capabilities_t *caps)
{
    caps->struct_size = sizeof(*caps);
    caps->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    caps->feature_bits = INC_DC_FW_FEATURE_EXTERNAL_STREAM |
                         INC_DC_FW_FEATURE_MULTI_INFLIGHT |
                         INC_DC_FW_FEATURE_DISPATCH_COMBINE_OVERLAP |
                         INC_DC_FW_FEATURE_DYNAMIC_TOKENS;
    caps->max_world_size = 16u;
    caps->max_topk = 8u;
    caps->max_inflight = 8u;
    caps->workspace_alignment = 256u;
    caps->dtype_bits = 1ull << INC_DC_FW_DTYPE_FP16;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Workspace(void *, const inc_dc_fw_plan_desc_t *, uint32_t op,
                             const inc_dc_fw_shape_t *shape,
                             inc_dc_fw_workspace_t *workspace)
{
    workspace->struct_size = sizeof(*workspace);
    workspace->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    workspace->bytes = shape->tokens * 2u + op * 256u;
    workspace->alignment = 256u;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Enqueue(void *opaque, const inc_dc_fw_plan_desc_t *, uint32_t,
                           const inc_dc_fw_invocation_t *invocation,
                           inc_dc_fw_backend_ticket_t *ticket)
{
    auto *backend = static_cast<Backend *>(opaque);
    assert(invocation->workspace != nullptr);
    assert(invocation->workspace_query_token != 0u);
    for (size_t i = 0; i < backend->tickets.size(); ++i) {
        Ticket &entry = backend->tickets[i];
        if (entry.state == INC_DC_FW_REQUEST_FREE) {
            entry.generation = backend->next_generation++;
            entry.state = INC_DC_FW_REQUEST_SUBMITTED;
            ticket->value = i + 1u;
            ticket->generation = entry.generation;
            ++backend->enqueues;
            return INC_DC_FW_OK;
        }
    }
    return INC_DC_FW_CAPACITY_EXCEEDED;
}

inc_dc_fw_status_t Query(void *opaque, inc_dc_fw_backend_ticket_t ticket,
                         uint32_t *state)
{
    Ticket *found = Find(static_cast<Backend *>(opaque), ticket);
    if (found == nullptr) return INC_DC_FW_STALE_HANDLE;
    *state = found->state;
    return found->state == INC_DC_FW_REQUEST_SUBMITTED ? INC_DC_FW_NOT_READY
                                                       : INC_DC_FW_OK;
}

inc_dc_fw_status_t Wait(void *opaque, inc_dc_fw_backend_ticket_t ticket,
                        uint64_t timeout_ns)
{
    Ticket *found = Find(static_cast<Backend *>(opaque), ticket);
    if (found == nullptr) return INC_DC_FW_STALE_HANDLE;
    if (timeout_ns == 1u) return INC_DC_FW_TIMEOUT;
    found->state = INC_DC_FW_REQUEST_COMPLETED;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Cancel(void *opaque, inc_dc_fw_backend_ticket_t ticket)
{
    Ticket *found = Find(static_cast<Backend *>(opaque), ticket);
    if (found == nullptr) return INC_DC_FW_STALE_HANDLE;
    found->state = INC_DC_FW_REQUEST_CANCELLED;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Release(void *opaque, inc_dc_fw_backend_ticket_t ticket)
{
    Ticket *found = Find(static_cast<Backend *>(opaque), ticket);
    if (found == nullptr) return INC_DC_FW_STALE_HANDLE;
    if (found->state == INC_DC_FW_REQUEST_SUBMITTED) return INC_DC_FW_NOT_READY;
    found->state = INC_DC_FW_REQUEST_FREE;
    return INC_DC_FW_OK;
}

void *Allocate(uint64_t bytes, uint32_t alignment, void *opaque)
{
    auto *backend = static_cast<Backend *>(opaque);
    void *pointer = nullptr;
    ++backend->allocations;
    return posix_memalign(&pointer, alignment, bytes) == 0 ? pointer : nullptr;
}

void Free(void *pointer, void *opaque)
{
    ++static_cast<Backend *>(opaque)->frees;
    std::free(pointer);
}

inc_dc_fw_backend_ops_t Ops(Backend *backend)
{
    inc_dc_fw_backend_ops_t ops{};
    ops.struct_size = sizeof(ops);
    ops.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    ops.backend_context = backend;
    ops.get_capabilities = Capabilities;
    ops.query_workspace = Workspace;
    ops.enqueue = Enqueue;
    ops.query = Query;
    ops.wait = Wait;
    ops.cancel = Cancel;
    ops.release = Release;
    return ops;
}

} // namespace

int main()
{
    Backend backend{};
    uint8_t static_route[64]{};
    inc_dc_infer_config_t config{};
    inc_dc_infer_config_init(&config);
    config.communicator.session_id = 1u;
    config.communicator.model_id = 2u;
    config.communicator.process_group_id = 3u;
    config.communicator.topology_generation = 4u;
    config.communicator.worker_world_size = 4u;
    config.communicator.worker_rank = 0u;
    config.communicator.max_tokens_per_chunk = 4096u;
    config.communicator.hidden_size = 7168u;
    config.communicator.max_topk = 8u;
    config.communicator.max_inflight = 8u;
    config.communicator.backend = Ops(&backend);
    inc_dc_easy_route_device_init(&config.communicator.static_route,
        INC_DC_FW_ROUTE_OPAQUE_DEVICE_PLAN, static_route,
        sizeof(static_route), 42u, 1u);
    config.allocate_device = Allocate;
    config.free_device = Free;
    config.allocator_context = &backend;

    inc_dc_infer_session_t *session = nullptr;
    assert(inc_dc_infer_session_create(&config, &session) == INC_DC_FW_OK);
    inc_dc_infer_plan_desc_t desc{};
    inc_dc_infer_plan_desc_init(&desc);
    desc.tokens = 128u;
    desc.topk = 2u;
    inc_dc_infer_plan_t *plan = nullptr;
    assert(inc_dc_infer_plan_create(session, &desc, &plan) == INC_DC_FW_OK);
    inc_dc_infer_plan_t *sibling = nullptr;
    assert(inc_dc_infer_plan_create(session, &desc, &sibling) == INC_DC_FW_OK);
    assert(backend.allocations == 4u);
    assert(inc_dc_infer_session_destroy(session) == INC_DC_FW_BUSY);

    inc_dc_infer_plan_info_t info{};
    assert(inc_dc_infer_plan_get_info(plan, &info) == INC_DC_FW_OK);
    assert(info.tokens == 128u && info.topk == 2u);
    assert(info.dispatch_workspace_bytes != info.combine_workspace_bytes);

    uint8_t input[64]{}, output[64]{};
    inc_dc_infer_io_t io{};
    inc_dc_infer_io_init(&io);
    io.input = input;
    io.output = output;
    io.stream = 1u;
    io.operation_generation = 7u;

    // Same-shape plans share the framework lease but own separate storage.
    // Releasing one must not invalidate the other pipeline slot.
    inc_dc_infer_request_t sibling_request{};
    assert(inc_dc_infer_dispatch_async(sibling, &io, &sibling_request) == INC_DC_FW_OK);
    assert(inc_dc_infer_request_wait_and_release(
               sibling, &sibling_request, 0u) == INC_DC_FW_OK);
    assert(inc_dc_infer_plan_destroy(sibling) == INC_DC_FW_OK);
    assert(backend.frees == 2u);

    // The two independent slots are the required arbitrary D/C overlap.
    inc_dc_infer_request_t dispatch{}, combine{}, duplicate{};
    assert(inc_dc_infer_dispatch_async(plan, &io, &dispatch) == INC_DC_FW_OK);
    assert(inc_dc_infer_combine_async(plan, &io, &combine) == INC_DC_FW_OK);
    assert(inc_dc_infer_dispatch_async(plan, &io, &duplicate) == INC_DC_FW_BUSY);
    assert(backend.allocations == 4u); // no hot-path allocations
    assert(inc_dc_infer_plan_destroy(plan) == INC_DC_FW_BUSY);

    inc_dc_infer_route_t route{};
    assert(inc_dc_infer_route_capture(plan, &dispatch, &io, &route) == INC_DC_FW_OK);
    assert(inc_dc_infer_session_destroy(session) == INC_DC_FW_BUSY);
    assert(inc_dc_infer_request_wait_and_release(plan, &dispatch, 0u) == INC_DC_FW_OK);
    assert(inc_dc_infer_request_wait_and_release(plan, &combine, 0u) == INC_DC_FW_OK);

    inc_dc_infer_request_t routed{};
    assert(inc_dc_infer_combine_with_route_async(plan, &route, &io, &routed) == INC_DC_FW_OK);
    assert(inc_dc_infer_request_wait(plan, &routed, 1u) == INC_DC_FW_TIMEOUT);
    assert(inc_dc_infer_request_wait_and_release(plan, &routed, 0u) == INC_DC_FW_OK);
    assert(inc_dc_infer_request_query(plan, &routed, nullptr) == INC_DC_FW_INVALID_ARGUMENT);
    assert(inc_dc_infer_route_release(session, &route) == INC_DC_FW_OK);

    // Reuse the prepared plan many times without allocator traffic.
    for (uint64_t generation = 8u; generation < 4104u; ++generation) {
        io.operation_generation = generation;
        inc_dc_infer_request_t request{};
        assert(inc_dc_infer_dispatch_async(plan, &io, &request) == INC_DC_FW_OK);
        assert(inc_dc_infer_request_wait_and_release(plan, &request, 0u) == INC_DC_FW_OK);
    }
    assert(backend.allocations == 4u);
    assert(inc_dc_infer_plan_destroy(plan) == INC_DC_FW_OK);
    assert(backend.frees == 4u);
    assert(inc_dc_infer_session_destroy(session) == INC_DC_FW_OK);

    // Recommended facade: exact-shape setup and one overlapping D/C pair.
    Backend facade_backend{};
    Control control{};
    inc_dc_single_inc_config_t facade_config{};
    inc_dc_single_inc_config_init(&facade_config);
    facade_config.worker_world_size = 4u;
    facade_config.worker_rank = 0u;
    facade_config.hidden_size = 7168u;
    facade_config.tokens = 128u;
    facade_config.topk = 2u;
    facade_config.backend = Ops(&facade_backend);
    inc_dc_easy_route_device_init(
        &facade_config.static_route,
        INC_DC_FW_ROUTE_OPAQUE_DEVICE_PLAN, static_route,
        sizeof(static_route), 42u, 1u);
    facade_config.allocate_device = Allocate;
    facade_config.free_device = Free;
    facade_config.allocator_context = &facade_backend;
    facade_config.before_enqueue = Before;
    facade_config.after_enqueue = After;
    facade_config.control_context = &control;

    inc_dc_single_inc_t *facade = nullptr;
    assert(inc_dc_single_inc_create(&facade_config, &facade) == INC_DC_FW_OK);
    inc_dc_single_inc_io_t facade_io{};
    inc_dc_single_inc_io_init(&facade_io);
    facade_io.input = input;
    facade_io.output = output;
    facade_io.stream = 1u;
    facade_io.operation_generation = 1u;
    inc_dc_single_inc_request_t facade_dispatch{}, facade_combine{};
    inc_dc_single_inc_route_t facade_route{};
    assert(inc_dc_single_inc_dispatch_async(
               facade, &facade_io, &facade_dispatch, &facade_route) ==
           INC_DC_FW_OK);
    assert(inc_dc_single_inc_combine_async(
               facade, &facade_route, &facade_io, &facade_combine) ==
           INC_DC_FW_OK);
    assert(inc_dc_single_inc_destroy(facade) == INC_DC_FW_BUSY);
    assert(inc_dc_single_inc_request_wait_and_release(
               facade, &facade_dispatch, 0u) == INC_DC_FW_OK);
    assert(inc_dc_single_inc_request_wait_and_release(
               facade, &facade_combine, 0u) == INC_DC_FW_OK);
    assert(inc_dc_single_inc_destroy(facade) == INC_DC_FW_BUSY);
    assert(inc_dc_single_inc_route_release(facade, &facade_route) ==
           INC_DC_FW_OK);
    assert(inc_dc_single_inc_destroy(facade) == INC_DC_FW_OK);
    assert(control.before_dispatch == 1u && control.after_dispatch == 1u);
    assert(control.before_combine == 1u && control.after_combine == 1u);
    assert(facade_backend.allocations == facade_backend.frees);
    return 0;
}
