#include "single_inc_example.h"

inc_dc_fw_status_t example_single_inc_dispatch(
    inc_dc_easy_comm_t *comm, void *input, void *output,
    uint64_t physical_output_rows, const example_token_plan_t *plan,
    uint64_t stream,
    example_device_alloc_fn allocate, example_device_free_fn release,
    void *allocator_context, uint64_t timeout_ns)
{
    if (comm == NULL || plan == NULL || plan->device_plan == NULL ||
        allocate == NULL || release == NULL || physical_output_rows == 0u) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    inc_dc_fw_workspace_t workspace = {0};
    inc_dc_fw_status_t status = inc_dc_easy_workspace_query(
        comm, INC_DC_FW_OP_DISPATCH, plan->tokens, plan->topk, &workspace);
    if (status != INC_DC_FW_OK) return status;
    void *workspace_ptr = workspace.bytes == 0u ? NULL : allocate(
        workspace.bytes, workspace.alignment, allocator_context);
    if (workspace.bytes != 0u && workspace_ptr == NULL) {
        (void)inc_dc_easy_workspace_release(comm, workspace.query_token);
        return INC_DC_FW_CAPACITY_EXCEEDED;
    }

    inc_dc_easy_op_t operation;
    inc_dc_easy_op_init(&operation);
    operation.tokens = plan->tokens;
    operation.input_rows = plan->tokens;
    operation.output_rows = physical_output_rows;
    operation.topk = plan->topk;
    operation.input = input;
    operation.output = output;
    operation.workspace = workspace_ptr;
    operation.workspace_bytes = workspace.bytes;
    operation.workspace_query_token = workspace.query_token;
    operation.stream = stream;
    operation.operation_generation = plan->info.generation;
    operation.route = plan->route;

    inc_dc_fw_request_t request = {0};
    status = inc_dc_easy_dispatch_async(comm, &operation, &request);
    if (status == INC_DC_FW_OK) {
        status = inc_dc_easy_request_wait_and_release(
            comm, request, timeout_ns);
    }
    if (workspace_ptr != NULL) release(workspace_ptr, allocator_context);
    (void)inc_dc_easy_workspace_release(comm, workspace.query_token);
    return status;
}
