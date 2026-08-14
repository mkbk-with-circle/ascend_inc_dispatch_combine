#include "single_inc_example.h"

inc_dc_fw_status_t example_run_dispatch_combine(
    const inc_dc_fw_backend_ops_t *native_backend,
    uint32_t worker_world_size, uint32_t worker_rank,
    uint32_t hidden_size, uint64_t stream,
    void *dispatch_input, void *dispatch_output,
    void *expert_output, void *combined_output,
    example_device_alloc_fn allocate, example_device_free_fn release,
    example_device_copy_h2d_fn copy_h2d, void *allocator_context)
{
    /*
     * Four local tokens, top-k=2, eight experts/rank.  Every worker uses the
     * same symmetric test plan. Duplicate destinations are intentional:
     * token 1 selects two experts on rank 0 and crosses the wire only once.
     */
    static const int32_t expert_ids[8] = {
        0, 8,  /* token 0 -> rank 0, rank 1 */
        1, 2,  /* token 1 -> two experts on rank 0 */
        9, 10, /* token 2 -> two experts on rank 1 */
        3, 11  /* token 3 -> rank 0, rank 1 */
    };
    static const float route_weights[8] = {
        0.75f, 0.25f, 0.60f, 0.40f,
        0.55f, 0.45f, 0.80f, 0.20f
    };
    const uint64_t tokens = 4u;
    const uint32_t topk = 2u;
    const uint32_t experts_per_worker = 8u;
    const uint64_t generation = 1u;
    const uint64_t timeout_ns = 30ull * 1000ull * 1000ull * 1000ull;
    inc_dc_easy_comm_t *comm = NULL;
    example_token_plan_t plan = {0};
    inc_dc_fw_status_t status;

    if (worker_world_size != 2u) {
        /* The literal plan above names experts on exactly two workers. */
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    status = example_single_inc_init(
        native_backend, worker_world_size, worker_rank, hidden_size,
        topk, &comm);
    if (status != INC_DC_FW_OK) return status;

    status = example_token_plan_create(
        tokens, topk, worker_world_size, worker_rank, experts_per_worker,
        expert_ids,
        NULL, /* derive destination = expert_id / experts_per_worker */
        route_weights, generation, stream, allocate, release, copy_h2d,
        allocator_context, &plan);
    if (status != INC_DC_FW_OK) goto destroy_comm;

    /*
     * Because all workers use the same symmetric plan, each destination gets
     * the same local physical rows from every source worker. A real backend
     * normally obtains this receive capacity during its route exchange.
     */
    status = example_single_inc_dispatch(
        comm, dispatch_input, dispatch_output,
        plan.info.local_physical_rows * worker_world_size, &plan, stream,
        allocate, release, allocator_context, timeout_ns);
    if (status != INC_DC_FW_OK) goto destroy_plan;

    /* Local expert expansion turns physical rows into expert instances. */
    status = example_single_inc_combine(
        comm, expert_output, combined_output,
        NULL, /* weights are already present in the token plan */
        plan.info.local_assignments * worker_world_size, &plan, stream,
        allocate, release, allocator_context, timeout_ns);

destroy_plan:
    example_token_plan_destroy(&plan, release, allocator_context);
destroy_comm:
    {
        const inc_dc_fw_status_t destroy_status =
            inc_dc_easy_comm_destroy(comm);
        return status == INC_DC_FW_OK ? destroy_status : status;
    }
}
