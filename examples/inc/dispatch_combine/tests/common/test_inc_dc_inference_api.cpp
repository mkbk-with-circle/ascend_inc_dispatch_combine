#include "inc_dc_single_inc_api.h"
#include "inc_dc_single_inc.hpp"

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

    // The only public facade: create -> dispatch -> combine -> destroy.
    Backend short_backend{};
    inc_dc_single_inc_config_t short_config{};
    inc_dc_single_inc_config_init(&short_config);
    short_config.worker_world_size = 4u;
    short_config.worker_rank = 0u;
    short_config.hidden_size = 7168u;
    short_config.tokens = 128u;
    short_config.topk = 2u;
    short_config.backend = Ops(&short_backend);
    short_config.allocate_device = Allocate;
    short_config.free_device = Free;
    short_config.allocator_context = &short_backend;

    inc::dc::SingleIncRoute short_route{};
    inc_dc_easy_route_device_init(
        &short_route.device, INC_DC_FW_ROUTE_OPAQUE_DEVICE_PLAN,
        static_route, sizeof(static_route), 42u, 9u);
    short_route.dispatch_output_rows = 256u;
    short_route.combine_input_rows = 256u;

    auto short_op = inc::dc::single_inc_create(short_config);
    auto batch = short_op.dispatch(input, output, short_route, 1u);
    short_op.combine(batch, input, output, 2u);
    inc::dc::single_inc_destroy(short_op);
    assert(short_backend.enqueues == 2u);
    assert(short_backend.allocations == short_backend.frees);
    return 0;
}
