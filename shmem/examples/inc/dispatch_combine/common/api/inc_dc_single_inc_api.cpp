#include "inc_dc_single_inc_api.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <new>

struct inc_dc_single_inc {
    inc_dc_infer_session_t *session = nullptr;
    inc_dc_infer_plan_t *plan = nullptr;
    uint64_t default_timeout_ns = 0u;
    std::atomic<uint64_t> live_routes{0u};
    inc_dc_single_inc_control_fn before_enqueue = nullptr;
    inc_dc_single_inc_control_fn after_enqueue = nullptr;
    void *control_context = nullptr;
};

namespace {

bool HeaderValid(const inc_dc_single_inc_config_t *config)
{
    return config != nullptr &&
        config->struct_size >= sizeof(*config) &&
        config->abi_version == INC_DC_SINGLE_INC_ABI_VERSION;
}

inc_dc_fw_status_t Create(
    const inc_dc_single_inc_config_t *config, inc_dc_context_t *context,
    inc_dc_single_inc_t **single_inc)
{
    if (!HeaderValid(config) || single_inc == nullptr ||
        config->worker_world_size == 0u ||
        config->worker_rank >= config->worker_world_size ||
        config->hidden_size == 0u || config->tokens == 0u ||
        config->topk == 0u || config->allocate_device == nullptr ||
        config->free_device == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    *single_inc = nullptr;
    auto *created = new (std::nothrow) inc_dc_single_inc_t();
    if (created == nullptr) return INC_DC_FW_INTERNAL;

    inc_dc_infer_config_t infer{};
    inc_dc_infer_config_init(&infer);
    infer.communicator.session_id = config->session_id;
    infer.communicator.model_id = config->model_id;
    infer.communicator.process_group_id = config->process_group_id;
    infer.communicator.topology_generation = config->topology_generation;
    infer.communicator.worker_world_size = config->worker_world_size;
    infer.communicator.worker_rank = config->worker_rank;
    infer.communicator.max_tokens_per_chunk =
        config->max_tokens_per_chunk == 0u
            ? static_cast<uint32_t>(std::min<uint64_t>(
                  config->tokens, std::numeric_limits<uint32_t>::max()))
            : config->max_tokens_per_chunk;
    infer.communicator.hidden_size = config->hidden_size;
    infer.communicator.max_topk = config->topk;
    infer.communicator.dtype = config->dtype;
    infer.communicator.max_inflight =
        config->max_inflight == 0u ? 2u : config->max_inflight;
    infer.communicator.max_workspace_queries =
        config->max_workspace_queries == 0u
            ? 2u : config->max_workspace_queries;
    infer.communicator.static_route = config->static_route;
    infer.communicator.backend = config->backend;
    infer.allocate_device = config->allocate_device;
    infer.free_device = config->free_device;
    infer.allocator_context = config->allocator_context;
    infer.flags = config->flags;

    inc_dc_fw_status_t status = context == nullptr
        ? inc_dc_infer_session_create(&infer, &created->session)
        : inc_dc_infer_session_create_from_context(
              &infer, context, &created->session);
    if (status != INC_DC_FW_OK) {
        delete created;
        return status;
    }

    inc_dc_infer_plan_desc_t plan{};
    inc_dc_infer_plan_desc_init(&plan);
    plan.tokens = config->tokens;
    plan.topk = config->topk;
    status = inc_dc_infer_plan_create(
        created->session, &plan, &created->plan);
    if (status != INC_DC_FW_OK) {
        (void)inc_dc_infer_session_destroy(created->session);
        delete created;
        return status;
    }
    created->default_timeout_ns = config->default_timeout_ns;
    created->before_enqueue = config->before_enqueue;
    created->after_enqueue = config->after_enqueue;
    created->control_context = config->control_context;
    *single_inc = created;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t AbortAndRelease(
    inc_dc_single_inc_t *single_inc,
    inc_dc_single_inc_request_t *request,
    inc_dc_fw_status_t primary_status)
{
    const inc_dc_fw_status_t cancel = inc_dc_infer_request_cancel(
        single_inc->plan, request);
    if (cancel != INC_DC_FW_OK && cancel != INC_DC_FW_CANCELLED &&
        cancel != INC_DC_FW_UNSUPPORTED) {
        return primary_status;
    }
    /* A cancelled request must be terminal before its backend ticket is freed. */
    (void)inc_dc_infer_request_wait(single_inc->plan, request, 0u);
    (void)inc_dc_infer_request_release(single_inc->plan, request);
    return primary_status;
}

inc_dc_fw_status_t FinishBlocking(
    inc_dc_single_inc_t *single_inc,
    inc_dc_single_inc_request_t *request)
{
    const inc_dc_fw_status_t status = inc_dc_infer_request_wait(
        single_inc->plan, request, single_inc->default_timeout_ns);
    if (status != INC_DC_FW_OK)
        return AbortAndRelease(single_inc, request, status);
    return inc_dc_infer_request_release(single_inc->plan, request);
}

} // namespace

extern "C" {

void inc_dc_single_inc_config_init(inc_dc_single_inc_config_t *config)
{
    if (config == nullptr) return;
    *config = {};
    config->struct_size = sizeof(*config);
    config->abi_version = INC_DC_SINGLE_INC_ABI_VERSION;
    config->session_id = 1u;
    config->model_id = 1u;
    config->process_group_id = 1u;
    config->topology_generation = 1u;
    config->dtype = INC_DC_FW_DTYPE_FP16;
}

void inc_dc_single_inc_io_init(inc_dc_single_inc_io_t *io)
{
    inc_dc_infer_io_init(io);
}

inc_dc_fw_status_t inc_dc_single_inc_create(
    const inc_dc_single_inc_config_t *config,
    inc_dc_single_inc_t **single_inc)
{
    return Create(config, nullptr, single_inc);
}

inc_dc_fw_status_t inc_dc_single_inc_create_from_context(
    const inc_dc_single_inc_config_t *config, inc_dc_context_t *context,
    inc_dc_single_inc_t **single_inc)
{
    if (context == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    return Create(config, context, single_inc);
}

inc_dc_fw_status_t inc_dc_single_inc_dispatch_async(
    inc_dc_single_inc_t *single_inc, const inc_dc_single_inc_io_t *io,
    inc_dc_single_inc_request_t *request,
    inc_dc_single_inc_route_t *route)
{
    if (single_inc == nullptr || io == nullptr || request == nullptr ||
        route == nullptr)
        return INC_DC_FW_INVALID_ARGUMENT;
    inc_dc_fw_status_t status = INC_DC_FW_OK;
    if (single_inc->before_enqueue != nullptr) {
        status = single_inc->before_enqueue(
            single_inc->control_context, INC_DC_FW_OP_DISPATCH,
            io->operation_generation);
        if (status != INC_DC_FW_OK) return status;
    }
    status = inc_dc_infer_dispatch_async(
        single_inc->plan, io, request);
    if (status != INC_DC_FW_OK) return status;
    status = inc_dc_infer_route_capture(
        single_inc->plan, request, io, route);
    if (status != INC_DC_FW_OK)
        return AbortAndRelease(single_inc, request, status);
    single_inc->live_routes.fetch_add(1u, std::memory_order_relaxed);
    if (single_inc->after_enqueue != nullptr) {
        status = single_inc->after_enqueue(
            single_inc->control_context, INC_DC_FW_OP_DISPATCH,
            io->operation_generation);
        if (status != INC_DC_FW_OK) {
            (void)inc_dc_single_inc_route_release(single_inc, route);
            return AbortAndRelease(single_inc, request, status);
        }
    }
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_single_inc_combine_async(
    inc_dc_single_inc_t *single_inc,
    const inc_dc_single_inc_route_t *dispatch_route,
    const inc_dc_single_inc_io_t *io,
    inc_dc_single_inc_request_t *request)
{
    if (single_inc == nullptr || io == nullptr)
        return INC_DC_FW_INVALID_ARGUMENT;
    inc_dc_fw_status_t status = INC_DC_FW_OK;
    if (single_inc->before_enqueue != nullptr) {
        status = single_inc->before_enqueue(
            single_inc->control_context, INC_DC_FW_OP_COMBINE,
            io->operation_generation);
        if (status != INC_DC_FW_OK) return status;
    }
    status = inc_dc_infer_combine_with_route_async(
        single_inc->plan, dispatch_route, io, request);
    if (status != INC_DC_FW_OK) return status;
    if (single_inc->after_enqueue != nullptr) {
        status = single_inc->after_enqueue(
            single_inc->control_context, INC_DC_FW_OP_COMBINE,
            io->operation_generation);
        if (status != INC_DC_FW_OK)
            return AbortAndRelease(single_inc, request, status);
    }
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_single_inc_dispatch(
    inc_dc_single_inc_t *single_inc, const inc_dc_single_inc_io_t *io,
    inc_dc_single_inc_route_t *route)
{
    inc_dc_single_inc_request_t request{};
    inc_dc_fw_status_t status = inc_dc_single_inc_dispatch_async(
        single_inc, io, &request, route);
    if (status != INC_DC_FW_OK) return status;
    status = FinishBlocking(single_inc, &request);
    if (status != INC_DC_FW_OK)
        (void)inc_dc_single_inc_route_release(single_inc, route);
    return status;
}

inc_dc_fw_status_t inc_dc_single_inc_combine(
    inc_dc_single_inc_t *single_inc,
    const inc_dc_single_inc_route_t *dispatch_route,
    const inc_dc_single_inc_io_t *io)
{
    inc_dc_single_inc_request_t request{};
    inc_dc_fw_status_t status = inc_dc_single_inc_combine_async(
        single_inc, dispatch_route, io, &request);
    return status == INC_DC_FW_OK ? FinishBlocking(single_inc, &request)
                                  : status;
}

inc_dc_fw_status_t inc_dc_single_inc_request_query(
    inc_dc_single_inc_t *single_inc,
    const inc_dc_single_inc_request_t *request, uint32_t *state)
{
    return single_inc == nullptr ? INC_DC_FW_INVALID_ARGUMENT
        : inc_dc_infer_request_query(single_inc->plan, request, state);
}

inc_dc_fw_status_t inc_dc_single_inc_request_wait(
    inc_dc_single_inc_t *single_inc,
    const inc_dc_single_inc_request_t *request, uint64_t timeout_ns)
{
    return single_inc == nullptr ? INC_DC_FW_INVALID_ARGUMENT
        : inc_dc_infer_request_wait(single_inc->plan, request, timeout_ns);
}

inc_dc_fw_status_t inc_dc_single_inc_request_cancel(
    inc_dc_single_inc_t *single_inc,
    const inc_dc_single_inc_request_t *request)
{
    return single_inc == nullptr ? INC_DC_FW_INVALID_ARGUMENT
        : inc_dc_infer_request_cancel(single_inc->plan, request);
}

inc_dc_fw_status_t inc_dc_single_inc_request_release(
    inc_dc_single_inc_t *single_inc, inc_dc_single_inc_request_t *request)
{
    return single_inc == nullptr ? INC_DC_FW_INVALID_ARGUMENT
        : inc_dc_infer_request_release(single_inc->plan, request);
}

inc_dc_fw_status_t inc_dc_single_inc_request_wait_and_release(
    inc_dc_single_inc_t *single_inc, inc_dc_single_inc_request_t *request,
    uint64_t timeout_ns)
{
    return single_inc == nullptr ? INC_DC_FW_INVALID_ARGUMENT
        : inc_dc_infer_request_wait_and_release(
              single_inc->plan, request, timeout_ns);
}

inc_dc_fw_status_t inc_dc_single_inc_route_release(
    inc_dc_single_inc_t *single_inc, inc_dc_single_inc_route_t *route)
{
    if (single_inc == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    const inc_dc_fw_status_t status = inc_dc_infer_route_release(
        single_inc->session, route);
    if (status == INC_DC_FW_OK) {
        const uint64_t before = single_inc->live_routes.fetch_sub(
            1u, std::memory_order_relaxed);
        if (before == 0u) {
            single_inc->live_routes.store(0u, std::memory_order_relaxed);
            return INC_DC_FW_INTERNAL;
        }
    }
    return status;
}

inc_dc_fw_status_t inc_dc_single_inc_get_plan_info(
    const inc_dc_single_inc_t *single_inc, inc_dc_infer_plan_info_t *info)
{
    return single_inc == nullptr ? INC_DC_FW_INVALID_ARGUMENT
        : inc_dc_infer_plan_get_info(single_inc->plan, info);
}

inc_dc_fw_status_t inc_dc_single_inc_get_stats(
    const inc_dc_single_inc_t *single_inc,
    inc_dc_fw_context_stats_t *stats)
{
    return single_inc == nullptr ? INC_DC_FW_INVALID_ARGUMENT
        : inc_dc_infer_session_get_stats(single_inc->session, stats);
}

inc_dc_fw_status_t inc_dc_single_inc_destroy(
    inc_dc_single_inc_t *single_inc)
{
    if (single_inc == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    if (single_inc->live_routes.load(std::memory_order_relaxed) != 0u)
        return INC_DC_FW_BUSY;
    inc_dc_fw_status_t status = inc_dc_infer_plan_destroy(single_inc->plan);
    if (status != INC_DC_FW_OK) return status;
    single_inc->plan = nullptr;
    status = inc_dc_infer_session_destroy(single_inc->session);
    if (status != INC_DC_FW_OK) return status;
    single_inc->session = nullptr;
    delete single_inc;
    return INC_DC_FW_OK;
}

} // extern "C"
