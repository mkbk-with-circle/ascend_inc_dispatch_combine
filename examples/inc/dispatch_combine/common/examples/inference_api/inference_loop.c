#include "inc_dc_inference_api.h"

/*
 * Minimal framework-side ownership pattern.  aclrtMalloc/aclrtFree wrappers
 * can be passed directly through these callback signatures in production.
 */
typedef struct model_moe_api {
    inc_dc_infer_session_t *session;
    inc_dc_infer_plan_t *shape_plan;
} model_moe_api_t;

inc_dc_fw_status_t model_moe_create(
    const inc_dc_fw_backend_ops_t *backend,
    uint32_t world_size, uint32_t rank, uint32_t hidden_size,
    uint32_t max_tokens, uint32_t max_topk,
    uint64_t prepared_tokens, uint32_t prepared_topk,
    inc_dc_infer_allocate_fn allocate_device,
    inc_dc_infer_free_fn free_device, void *allocator_context,
    model_moe_api_t *api)
{
    inc_dc_infer_config_t config;
    inc_dc_infer_plan_desc_t plan;
    inc_dc_fw_status_t status;
    if (backend == NULL || api == NULL) return INC_DC_FW_INVALID_ARGUMENT;
    *api = (model_moe_api_t){0};

    inc_dc_infer_config_init(&config);
    config.communicator.session_id = 1u;
    config.communicator.model_id = 1u;
    config.communicator.process_group_id = 1u;
    config.communicator.topology_generation = 1u;
    config.communicator.worker_world_size = world_size;
    config.communicator.worker_rank = rank;
    config.communicator.max_tokens_per_chunk = max_tokens;
    config.communicator.hidden_size = hidden_size;
    config.communicator.max_topk = max_topk;
    config.communicator.backend = *backend;
    config.allocate_device = allocate_device;
    config.free_device = free_device;
    config.allocator_context = allocator_context;
    status = inc_dc_infer_session_create(&config, &api->session);
    if (status != INC_DC_FW_OK) return status;

    inc_dc_infer_plan_desc_init(&plan);
    plan.tokens = prepared_tokens;
    plan.topk = prepared_topk;
    status = inc_dc_infer_plan_create(api->session, &plan, &api->shape_plan);
    if (status != INC_DC_FW_OK) {
        (void)inc_dc_infer_session_destroy(api->session);
        api->session = NULL;
    }
    return status;
}

/* Hot path: no workspace query, malloc, free, or host synchronization. */
inc_dc_fw_status_t model_moe_dispatch_async(
    model_moe_api_t *api, void *tokens, void *expert_rows,
    const inc_dc_fw_route_desc_t *device_route, uint64_t stream,
    uint64_t generation, inc_dc_infer_request_t *request)
{
    inc_dc_infer_io_t io;
    inc_dc_infer_io_init(&io);
    io.input = tokens;
    io.output = expert_rows;
    if (device_route != NULL) io.route = *device_route;
    io.stream = stream;
    io.operation_generation = generation;
    return inc_dc_infer_dispatch_async(api->shape_plan, &io, request);
}

inc_dc_fw_status_t model_moe_combine_async(
    model_moe_api_t *api, void *expert_output, void *token_output,
    void *weights, uint64_t weight_elements,
    const inc_dc_infer_route_t *dispatch_route, uint64_t stream,
    uint64_t generation, inc_dc_infer_request_t *request)
{
    inc_dc_infer_io_t io;
    inc_dc_infer_io_init(&io);
    io.input = expert_output;
    io.output = token_output;
    io.weights = weights;
    io.weight_elements = weight_elements;
    io.stream = stream;
    io.operation_generation = generation;
    return dispatch_route == NULL
        ? inc_dc_infer_combine_async(api->shape_plan, &io, request)
        : inc_dc_infer_combine_with_route_async(
              api->shape_plan, dispatch_route, &io, request);
}

inc_dc_fw_status_t model_moe_destroy(model_moe_api_t *api)
{
    inc_dc_fw_status_t status;
    if (api == NULL) return INC_DC_FW_INVALID_ARGUMENT;
    status = inc_dc_infer_plan_destroy(api->shape_plan);
    if (status != INC_DC_FW_OK) return status;
    api->shape_plan = NULL;
    status = inc_dc_infer_session_destroy(api->session);
    if (status == INC_DC_FW_OK) api->session = NULL;
    return status;
}
