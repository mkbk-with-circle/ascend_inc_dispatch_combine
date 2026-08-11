#include "inc_dc_easy_api.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CHECK(x)                                                        \
    do {                                                                \
        if (!(x)) {                                                     \
            std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #x,     \
                         __FILE__, __LINE__);                            \
            std::abort();                                               \
        }                                                               \
    } while (0)

namespace {

struct Ticket {
    uint64_t generation = 0u;
    uint32_t state = INC_DC_FW_REQUEST_FREE;
};

struct Backend {
    std::vector<Ticket> tickets = std::vector<Ticket>(8);
    uint64_t generation = 1u;
    uint32_t dispatches = 0u;
    uint32_t combines = 0u;
};

Ticket *Find(Backend *backend, inc_dc_fw_backend_ticket_t ticket)
{
    if (backend == nullptr || ticket.value == 0u ||
        ticket.value > backend->tickets.size()) return nullptr;
    Ticket &entry = backend->tickets[ticket.value - 1u];
    return entry.generation == ticket.generation ? &entry : nullptr;
}

inc_dc_fw_status_t Capabilities(
    void *, inc_dc_fw_capabilities_t *capabilities)
{
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    capabilities->feature_bits =
        INC_DC_FW_FEATURE_EXTERNAL_STREAM |
        INC_DC_FW_FEATURE_MULTI_INFLIGHT |
        INC_DC_FW_FEATURE_DYNAMIC_TOKENS |
        INC_DC_FW_FEATURE_INTERNAL_CHUNKING |
        INC_DC_FW_FEATURE_ROUTE_METADATA |
        INC_DC_FW_FEATURE_LOCAL_EXPERT_EXPAND |
        INC_DC_FW_FEATURE_FP32_WEIGHTS;
    capabilities->max_world_size = 16u;
    capabilities->max_topk = 64u;
    capabilities->max_inflight = 8u;
    capabilities->workspace_alignment = 256u;
    capabilities->dtype_bits = 1ull << INC_DC_FW_DTYPE_FP16;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Workspace(
    void *, const inc_dc_fw_plan_desc_t *, uint32_t,
    const inc_dc_fw_shape_t *shape, inc_dc_fw_workspace_t *workspace)
{
    workspace->struct_size = sizeof(*workspace);
    workspace->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    workspace->alignment = 256u;
    workspace->bytes =
        ((shape->tokens * shape->hidden_size * 2ull + 4095ull + 255ull) /
         256ull) * 256ull;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Enqueue(
    void *opaque, const inc_dc_fw_plan_desc_t *, uint32_t operation,
    const inc_dc_fw_invocation_t *invocation,
    inc_dc_fw_backend_ticket_t *ticket)
{
    auto *backend = static_cast<Backend *>(opaque);
    if (invocation->stream == 0u) return INC_DC_FW_INVALID_ARGUMENT;
    for (size_t i = 0; i < backend->tickets.size(); ++i) {
        Ticket &entry = backend->tickets[i];
        if (entry.state == INC_DC_FW_REQUEST_FREE) {
            entry.generation = backend->generation++;
            entry.state = INC_DC_FW_REQUEST_SUBMITTED;
            ticket->value = i + 1u;
            ticket->generation = entry.generation;
            if (operation == INC_DC_FW_OP_DISPATCH) ++backend->dispatches;
            if (operation == INC_DC_FW_OP_COMBINE) ++backend->combines;
            return INC_DC_FW_OK;
        }
    }
    return INC_DC_FW_CAPACITY_EXCEEDED;
}

inc_dc_fw_status_t Query(
    void *opaque, inc_dc_fw_backend_ticket_t ticket, uint32_t *state)
{
    Ticket *entry = Find(static_cast<Backend *>(opaque), ticket);
    if (entry == nullptr) return INC_DC_FW_STALE_HANDLE;
    *state = entry->state;
    return entry->state == INC_DC_FW_REQUEST_SUBMITTED
               ? INC_DC_FW_NOT_READY : INC_DC_FW_OK;
}

inc_dc_fw_status_t Wait(
    void *opaque, inc_dc_fw_backend_ticket_t ticket, uint64_t timeout_ns)
{
    Ticket *entry = Find(static_cast<Backend *>(opaque), ticket);
    if (entry == nullptr) return INC_DC_FW_STALE_HANDLE;
    if (timeout_ns == 1u) return INC_DC_FW_TIMEOUT;
    entry->state = INC_DC_FW_REQUEST_COMPLETED;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Cancel(
    void *opaque, inc_dc_fw_backend_ticket_t ticket)
{
    Ticket *entry = Find(static_cast<Backend *>(opaque), ticket);
    if (entry == nullptr) return INC_DC_FW_STALE_HANDLE;
    entry->state = INC_DC_FW_REQUEST_CANCELLED;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Release(
    void *opaque, inc_dc_fw_backend_ticket_t ticket)
{
    Ticket *entry = Find(static_cast<Backend *>(opaque), ticket);
    if (entry == nullptr) return INC_DC_FW_STALE_HANDLE;
    if (entry->state == INC_DC_FW_REQUEST_SUBMITTED)
        return INC_DC_FW_NOT_READY;
    entry->state = INC_DC_FW_REQUEST_FREE;
    return INC_DC_FW_OK;
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
    const int32_t expert_ids[8] = {0, 8, 1, 2, 9, 10, 3, 11};
    const float route_weights[8] = {
        0.75f, 0.25f, 0.60f, 0.40f,
        0.55f, 0.45f, 0.80f, 0.20f};
    inc_dc_easy_token_plan_desc_t token_desc{};
    inc_dc_easy_token_plan_desc_init(&token_desc);
    token_desc.tokens = 4u;
    token_desc.topk = 2u;
    token_desc.worker_world_size = 2u;
    token_desc.worker_rank = 0u;
    token_desc.experts_per_worker = 8u;
    token_desc.expert_ids = expert_ids;
    token_desc.weights = route_weights;
    token_desc.generation = 7u;
    uint64_t token_plan_bytes = 0u;
    CHECK(inc_dc_easy_token_plan_query(&token_desc, &token_plan_bytes) ==
          INC_DC_FW_OK);
    std::vector<uint8_t> token_plan(token_plan_bytes);
    inc_dc_easy_token_plan_info_t token_info{};
    CHECK(inc_dc_easy_token_plan_build(
              &token_desc, token_plan.data(), token_plan.size(),
              &token_info) == INC_DC_FW_OK);
    CHECK(token_info.logical_assignments == 8u);
    CHECK(token_info.global_physical_rows == 6u);
    CHECK(token_info.local_assignments == 4u);
    CHECK(token_info.local_physical_rows == 3u);
    CHECK(token_info.semantic_digest != 0u);
    const auto *token_header =
        reinterpret_cast<const inc_dc_easy_token_plan_header_v1_t *>(
            token_plan.data());
    CHECK(token_header->magic == INC_DC_EASY_TOKEN_PLAN_MAGIC);
    CHECK(token_header->total_bytes == token_plan_bytes);
    CHECK(token_header->semantic_digest == token_info.semantic_digest);
    inc_dc_fw_route_desc_t dense_route{};
    inc_dc_easy_token_plan_device_route_init(
        &dense_route, token_plan.data(), &token_info);
    CHECK(dense_route.format == INC_DC_FW_ROUTE_TOPK_DENSE);
    CHECK(dense_route.bytes == token_plan_bytes);
    CHECK(dense_route.generation == 7u);

    int32_t duplicate_experts[2] = {1, 1};
    token_desc.tokens = 1u;
    token_desc.expert_ids = duplicate_experts;
    CHECK(inc_dc_easy_token_plan_query(&token_desc, &token_plan_bytes) ==
          INC_DC_FW_OK);
    token_plan.resize(token_plan_bytes);
    CHECK(inc_dc_easy_token_plan_build(
              &token_desc, token_plan.data(), token_plan.size(),
              &token_info) == INC_DC_FW_INVALID_ARGUMENT);

    // Routing weights cross the public ABI as FP32.  Reject non-finite
    // values while the plan is still host-visible instead of allowing NaN
    // or Inf to contaminate an asynchronous reduction.
    int32_t finite_experts[2] = {1, 2};
    float invalid_weights[2] = {
        std::numeric_limits<float>::quiet_NaN(), 1.0f};
    token_desc.expert_ids = finite_experts;
    token_desc.weights = invalid_weights;
    CHECK(inc_dc_easy_token_plan_build(
              &token_desc, token_plan.data(), token_plan.size(),
              &token_info) == INC_DC_FW_INVALID_ARGUMENT);
    invalid_weights[0] = std::numeric_limits<float>::infinity();
    CHECK(inc_dc_easy_token_plan_build(
              &token_desc, token_plan.data(), token_plan.size(),
              &token_info) == INC_DC_FW_INVALID_ARGUMENT);

    Backend backend{};
    uint8_t route_storage[256]{};
    inc_dc_easy_comm_config_t config{};
    inc_dc_easy_comm_config_init(&config);
    config.session_id = 1u;
    config.model_id = 2u;
    config.process_group_id = 3u;
    config.topology_generation = 4u;
    config.worker_world_size = 8u;
    config.worker_rank = 3u;
    config.max_tokens_per_chunk = 4096u;
    config.hidden_size = 7168u;
    config.max_topk = 64u;
    config.max_inflight = 8u;
    config.backend = Ops(&backend);
    inc_dc_easy_route_device_init(
        &config.static_route, INC_DC_FW_ROUTE_OPAQUE_DEVICE_PLAN,
        route_storage, sizeof(route_storage), 0x1234u, 1u);

    inc_dc_easy_comm_t *comm = nullptr;
    CHECK(inc_dc_easy_comm_create(&config, &comm) == INC_DC_FW_OK);
    inc_dc_fw_capabilities_t caps{};
    caps.struct_size = sizeof(caps);
    CHECK(inc_dc_easy_comm_get_capabilities(comm, &caps) == INC_DC_FW_OK);
    CHECK(caps.max_topk == 64u);

    inc_dc_fw_workspace_t dispatch_ws{};
    CHECK(inc_dc_easy_workspace_query(
              comm, INC_DC_FW_OP_DISPATCH, 32u, 16u, &dispatch_ws) ==
          INC_DC_FW_OK);
    inc_dc_fw_workspace_t combine_ws{};
    CHECK(inc_dc_easy_workspace_query(
              comm, INC_DC_FW_OP_COMBINE, 32u, 16u, &combine_ws) ==
          INC_DC_FW_OK);
    const size_t bytes = static_cast<size_t>(
        dispatch_ws.bytes > combine_ws.bytes
            ? dispatch_ws.bytes : combine_ws.bytes);
    void *storage = nullptr;
    CHECK(posix_memalign(&storage, 256u, bytes) == 0);

    inc_dc_easy_op_t op{};
    inc_dc_easy_op_init(&op);
    CHECK(op.weight_dtype == INC_DC_FW_DTYPE_FP32);
    op.tokens = 32u;
    op.topk = 16u;
    op.input = storage;
    op.output = storage;
    op.workspace = storage;
    op.workspace_bytes = dispatch_ws.bytes;
    op.workspace_query_token = dispatch_ws.query_token;
    op.stream = 0x1000u;
    op.operation_generation = 1u; /* static route is generation-independent */

    inc_dc_fw_request_t dispatch{};
    CHECK(inc_dc_easy_dispatch_async(comm, &op, &dispatch) == INC_DC_FW_OK);
    inc_dc_fw_route_handle_t easy_route_handle{};
    CHECK(inc_dc_easy_route_handle_create(
              comm, dispatch, &op, &easy_route_handle) == INC_DC_FW_OK);
    CHECK(backend.dispatches == 1u);
    CHECK(inc_dc_easy_comm_destroy(comm) == INC_DC_FW_BUSY);
    CHECK(inc_dc_easy_workspace_release(
              comm, dispatch_ws.query_token) == INC_DC_FW_BUSY);
    CHECK(inc_dc_easy_request_wait(comm, dispatch, 1u) == INC_DC_FW_TIMEOUT);
    CHECK(inc_dc_easy_request_wait_and_release(
              comm, dispatch, 1000u) == INC_DC_FW_OK);

    op.workspace_bytes = combine_ws.bytes;
    op.workspace_query_token = combine_ws.query_token;
    op.stream = 0x2000u;
    op.weights = storage;
    op.weight_elements = op.tokens * op.topk;
    op.weight_dtype = INC_DC_FW_DTYPE_FP32;
    inc_dc_fw_request_t combine{};
    CHECK(inc_dc_easy_combine_with_route_async(
              comm, easy_route_handle, &op, &combine) == INC_DC_FW_OK);
    CHECK(backend.combines == 1u);
    CHECK(inc_dc_easy_workspace_release(
              comm, combine_ws.query_token) == INC_DC_FW_BUSY);
    CHECK(inc_dc_easy_route_handle_release(
              comm, easy_route_handle) == INC_DC_FW_BUSY);
    uint32_t state = INC_DC_FW_REQUEST_FREE;
    CHECK(inc_dc_easy_request_query(comm, combine, &state) ==
          INC_DC_FW_NOT_READY);
    CHECK(state == INC_DC_FW_REQUEST_SUBMITTED);
    CHECK(inc_dc_easy_request_cancel(comm, combine) == INC_DC_FW_OK);
    CHECK(inc_dc_easy_request_release(comm, combine) == INC_DC_FW_OK);
    CHECK(inc_dc_easy_route_handle_release(
              comm, easy_route_handle) == INC_DC_FW_OK);

    // Reuse one communicator and the same two workspace leases for many
    // generations.  This catches request-slot leaks, stale backend tickets,
    // and leases that remain pinned after wait-and-release.
    op.topk = 16u;
    for (uint64_t generation = 3u; generation < 4099u; ++generation) {
        const bool is_dispatch = (generation & 1u) != 0u;
        op.workspace_bytes = is_dispatch ? dispatch_ws.bytes
                                         : combine_ws.bytes;
        op.workspace_query_token =
            is_dispatch ? dispatch_ws.query_token : combine_ws.query_token;
        op.stream = is_dispatch ? 0x1000u : 0x2000u;
        op.operation_generation = generation;
        inc_dc_fw_request_t request{};
        const inc_dc_fw_status_t submit =
            is_dispatch ? inc_dc_easy_dispatch_async(comm, &op, &request)
                        : inc_dc_easy_combine_async(comm, &op, &request);
        CHECK(submit == INC_DC_FW_OK);
        CHECK(inc_dc_easy_request_wait_and_release(
                  comm, request, 1000u) == INC_DC_FW_OK);
    }

    op.topk = 65u;
    CHECK(inc_dc_easy_dispatch_async(comm, &op, &dispatch) ==
          INC_DC_FW_INVALID_ARGUMENT);
    inc_dc_fw_context_stats_t easy_stats{};
    easy_stats.struct_size = sizeof(easy_stats);
    CHECK(inc_dc_easy_comm_get_stats(comm, &easy_stats) == INC_DC_FW_OK);
    CHECK(easy_stats.dispatch_enqueued == 2049u);
    CHECK(easy_stats.combine_enqueued == 2049u);
    CHECK(easy_stats.completed == 4097u);
    CHECK(easy_stats.cancelled == 1u);
    CHECK(easy_stats.wait_timeouts == 1u);
    CHECK(easy_stats.live_requests == 0u);
    CHECK(inc_dc_easy_workspace_release(
              comm, dispatch_ws.query_token) == INC_DC_FW_OK);
    CHECK(inc_dc_easy_workspace_release(
              comm, combine_ws.query_token) == INC_DC_FW_OK);
    CHECK(inc_dc_easy_comm_destroy(comm) == INC_DC_FW_OK);

    // A portability/session owner may create the framework context first;
    // Easy API borrows it and must not destroy it with the communicator.
    inc_dc_fw_context_config_t borrowed_config{};
    borrowed_config.struct_size = sizeof(borrowed_config);
    borrowed_config.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    borrowed_config.max_inflight = 8u;
    borrowed_config.max_workspace_queries = 8u;
    borrowed_config.backend = Ops(&backend);
    inc_dc_context_t *borrowed_context = nullptr;
    CHECK(inc_dc_fw_context_create(
              &borrowed_config, &borrowed_context) == INC_DC_FW_OK);
    inc_dc_easy_comm_t *borrowed_comm = nullptr;
    CHECK(inc_dc_easy_comm_create_from_context(
              &config, borrowed_context, &borrowed_comm) == INC_DC_FW_OK);
    CHECK(inc_dc_fw_context_destroy(borrowed_context) == INC_DC_FW_BUSY);
    CHECK(inc_dc_easy_comm_destroy(borrowed_comm) == INC_DC_FW_OK);
    CHECK(inc_dc_fw_context_destroy(borrowed_context) == INC_DC_FW_OK);
    std::free(storage);
    std::puts("inc_dc_easy_api: PASS");
    return 0;
}
