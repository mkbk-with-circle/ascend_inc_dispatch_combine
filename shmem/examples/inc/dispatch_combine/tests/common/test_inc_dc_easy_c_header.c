#include "inc_dc_easy_api.h"

static void compile_only(void)
{
    inc_dc_easy_comm_config_t config;
    inc_dc_easy_op_t operation;
    inc_dc_easy_token_plan_desc_t token_plan;
    inc_dc_easy_comm_config_init(&config);
    inc_dc_easy_op_init(&operation);
    inc_dc_easy_token_plan_desc_init(&token_plan);
}
