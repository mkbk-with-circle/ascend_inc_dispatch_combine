#include "single_inc_example.h"

#include <stdlib.h>

inc_dc_fw_status_t example_token_plan_create(
    uint64_t tokens, uint32_t topk, uint32_t worker_world_size,
    uint32_t worker_rank, uint32_t experts_per_worker,
    const int32_t *expert_ids, const uint32_t *destination_ranks,
    const float *weights, uint64_t generation, uint64_t stream,
    example_device_alloc_fn allocate, example_device_free_fn release,
    example_device_copy_h2d_fn copy_h2d, void *allocator_context,
    example_token_plan_t *plan)
{
    uint64_t host_bytes = 0u;
    inc_dc_easy_token_plan_desc_t description;
    inc_dc_fw_status_t status;
    if (allocate == NULL || release == NULL || copy_h2d == NULL ||
        plan == NULL) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    *plan = (example_token_plan_t){0};
    inc_dc_easy_token_plan_desc_init(&description);
    description.tokens = tokens;
    description.topk = topk;
    description.worker_world_size = worker_world_size;
    description.worker_rank = worker_rank;
    description.experts_per_worker = experts_per_worker;
    description.expert_ids = expert_ids;
    description.destination_ranks = destination_ranks;
    description.weights = weights;
    description.generation = generation;

    status = inc_dc_easy_token_plan_query(&description, &host_bytes);
    if (status != INC_DC_FW_OK) return status;
    plan->host_plan = malloc((size_t)host_bytes);
    if (plan->host_plan == NULL) return INC_DC_FW_CAPACITY_EXCEEDED;
    status = inc_dc_easy_token_plan_build(
        &description, plan->host_plan, host_bytes, &plan->info);
    if (status != INC_DC_FW_OK) goto fail;

    plan->device_plan = allocate(host_bytes, 64u, allocator_context);
    if (plan->device_plan == NULL) {
        status = INC_DC_FW_CAPACITY_EXCEEDED;
        goto fail;
    }
    status = copy_h2d(
        plan->device_plan, plan->host_plan, host_bytes, stream,
        allocator_context);
    if (status != INC_DC_FW_OK) goto fail;
    inc_dc_easy_token_plan_device_route_init(
        &plan->route, plan->device_plan, &plan->info);
    plan->tokens = tokens;
    plan->topk = topk;
    plan->worker_world_size = worker_world_size;
    plan->worker_rank = worker_rank;
    return INC_DC_FW_OK;

fail:
    if (plan->device_plan != NULL) {
        release(plan->device_plan, allocator_context);
    }
    free(plan->host_plan);
    *plan = (example_token_plan_t){0};
    return status;
}

void example_token_plan_destroy(
    example_token_plan_t *plan, example_device_free_fn release,
    void *allocator_context)
{
    if (plan == NULL) return;
    if (plan->device_plan != NULL && release != NULL) {
        release(plan->device_plan, allocator_context);
    }
    free(plan->host_plan);
    *plan = (example_token_plan_t){0};
}
