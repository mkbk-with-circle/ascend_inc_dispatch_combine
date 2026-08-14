#include "inc_dc_inference_api.h"

static void compile_only(void)
{
    inc_dc_infer_config_t config;
    inc_dc_infer_plan_desc_t plan;
    inc_dc_infer_io_t io;
    inc_dc_infer_request_t request = {0};
    inc_dc_infer_route_t route = {0};
    inc_dc_infer_config_init(&config);
    inc_dc_infer_plan_desc_init(&plan);
    inc_dc_infer_io_init(&io);
    (void)request;
    (void)route;
}
