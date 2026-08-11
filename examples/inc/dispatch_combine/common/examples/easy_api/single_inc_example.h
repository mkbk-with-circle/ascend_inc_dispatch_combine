#ifndef INC_DC_SINGLE_INC_EXAMPLE_H
#define INC_DC_SINGLE_INC_EXAMPLE_H

#include "inc_dc_easy_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *(*example_device_alloc_fn)(
    uint64_t bytes, uint32_t alignment, void *context);
typedef void (*example_device_free_fn)(void *pointer, void *context);
typedef inc_dc_fw_status_t (*example_device_copy_h2d_fn)(
    void *device_destination, const void *host_source, uint64_t bytes,
    uint64_t stream, void *context);

typedef struct example_token_plan {
    void *host_plan;
    void *device_plan;
    inc_dc_easy_token_plan_info_t info;
    inc_dc_fw_route_desc_t route;
    uint64_t tokens;
    uint32_t topk;
    uint32_t worker_world_size;
    uint32_t worker_rank;
} example_token_plan_t;

inc_dc_fw_status_t example_single_inc_init(
    const inc_dc_fw_backend_ops_t *native_backend,
    uint32_t worker_world_size, uint32_t worker_rank,
    uint32_t hidden_size, uint32_t max_topk,
    inc_dc_easy_comm_t **comm);
inc_dc_fw_status_t example_single_inc_init_from_context(
    inc_dc_context_t *framework_context,
    uint32_t worker_world_size, uint32_t worker_rank,
    uint32_t hidden_size, uint32_t max_topk,
    inc_dc_easy_comm_t **comm);

inc_dc_fw_status_t example_token_plan_create(
    uint64_t tokens, uint32_t topk, uint32_t worker_world_size,
    uint32_t worker_rank, uint32_t experts_per_worker,
    const int32_t *expert_ids, const uint32_t *destination_ranks,
    const float *weights, uint64_t generation, uint64_t stream,
    example_device_alloc_fn allocate, example_device_free_fn release,
    example_device_copy_h2d_fn copy_h2d, void *allocator_context,
    example_token_plan_t *plan);
void example_token_plan_destroy(
    example_token_plan_t *plan, example_device_free_fn release,
    void *allocator_context);

inc_dc_fw_status_t example_single_inc_dispatch(
    inc_dc_easy_comm_t *comm, void *input, void *output,
    uint64_t physical_output_rows, const example_token_plan_t *plan,
    uint64_t stream,
    example_device_alloc_fn allocate, example_device_free_fn release,
    void *allocator_context, uint64_t timeout_ns);

inc_dc_fw_status_t example_single_inc_combine(
    inc_dc_easy_comm_t *comm, void *expert_output, void *combined_output,
    void *weights, uint64_t expert_instance_rows,
    const example_token_plan_t *plan, uint64_t stream,
    example_device_alloc_fn allocate, example_device_free_fn release,
    void *allocator_context, uint64_t timeout_ns);

/* Complete symmetric test flow; every worker calls it with its own rank. */
inc_dc_fw_status_t example_run_dispatch_combine(
    const inc_dc_fw_backend_ops_t *native_backend,
    uint32_t worker_world_size, uint32_t worker_rank,
    uint32_t hidden_size, uint64_t stream,
    void *dispatch_input, void *dispatch_output,
    void *expert_output, void *combined_output,
    example_device_alloc_fn allocate, example_device_free_fn release,
    example_device_copy_h2d_fn copy_h2d, void *allocator_context);

#ifdef __cplusplus
}
#endif

#endif
