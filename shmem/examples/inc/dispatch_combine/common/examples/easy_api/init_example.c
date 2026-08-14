#include "single_inc_example.h"

static void fill_config(
    uint32_t worker_world_size, uint32_t worker_rank,
    uint32_t hidden_size, uint32_t max_topk,
    inc_dc_easy_comm_config_t *config)
{
    inc_dc_easy_comm_config_init(config);
    config->session_id = 1u;
    config->model_id = 1u;
    config->process_group_id = 1u;
    config->topology_generation = 1u;
    config->worker_world_size = worker_world_size;
    config->worker_rank = worker_rank;
    config->max_tokens_per_chunk = 32768u;
    config->hidden_size = hidden_size;
    config->max_topk = max_topk;
    config->dtype = INC_DC_FW_DTYPE_FP16;
    config->max_inflight = 64u;
    config->max_workspace_queries = 64u;
}

inc_dc_fw_status_t example_single_inc_init(
    const inc_dc_fw_backend_ops_t *native_backend,
    uint32_t worker_world_size, uint32_t worker_rank,
    uint32_t hidden_size, uint32_t max_topk,
    inc_dc_easy_comm_t **comm)
{
    if (native_backend == NULL || comm == NULL) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    inc_dc_easy_comm_config_t config;
    fill_config(worker_world_size, worker_rank, hidden_size, max_topk,
                &config);
    config.backend = *native_backend;
    return inc_dc_easy_comm_create(&config, comm);
}

inc_dc_fw_status_t example_single_inc_init_from_context(
    inc_dc_context_t *framework_context,
    uint32_t worker_world_size, uint32_t worker_rank,
    uint32_t hidden_size, uint32_t max_topk,
    inc_dc_easy_comm_t **comm)
{
    if (framework_context == NULL || comm == NULL) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    inc_dc_easy_comm_config_t config;
    fill_config(worker_world_size, worker_rank, hidden_size, max_topk,
                &config);
    return inc_dc_easy_comm_create_from_context(
        &config, framework_context, comm);
}
