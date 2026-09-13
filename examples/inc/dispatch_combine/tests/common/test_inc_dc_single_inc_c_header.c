#include "inc_dc_single_inc_api.h"

int main(void)
{
    inc_dc_single_inc_config_t config;
    inc_dc_single_inc_io_t io;
    inc_dc_single_inc_config_init(&config);
    inc_dc_single_inc_io_init(&io);
    return config.abi_version == INC_DC_SINGLE_INC_ABI_VERSION ? 0 : 1;
}
