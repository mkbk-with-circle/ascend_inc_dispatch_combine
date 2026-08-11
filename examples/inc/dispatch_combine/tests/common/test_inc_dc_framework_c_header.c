#include "inc_dc_framework_c_api.h"

#if UINTPTR_MAX == UINT64_MAX
_Static_assert(sizeof(inc_dc_fw_version_t) == 16, "version ABI drift");
_Static_assert(sizeof(inc_dc_fw_capabilities_t) == 72, "capability ABI drift");
_Static_assert(sizeof(inc_dc_fw_route_desc_t) == 64, "route ABI drift");
_Static_assert(sizeof(inc_dc_fw_tensor_desc_t) == 112, "tensor ABI drift");
_Static_assert(sizeof(inc_dc_fw_plan_desc_t) == 168, "plan ABI drift");
_Static_assert(sizeof(inc_dc_fw_shape_t) == 48, "shape ABI drift");
_Static_assert(sizeof(inc_dc_fw_workspace_t) == 56, "workspace ABI drift");
_Static_assert(sizeof(inc_dc_fw_invocation_t) == 544, "invocation ABI drift");
_Static_assert(sizeof(inc_dc_fw_request_t) == 16, "request ABI drift");
_Static_assert(sizeof(inc_dc_fw_backend_ops_t) == 104, "backend ABI drift");
_Static_assert(sizeof(inc_dc_fw_context_config_t) == 160, "config ABI drift");
#endif

int main(void)
{
    inc_dc_fw_version_t version = {0};
    version.struct_size = sizeof(version);
    if (inc_dc_fw_get_version(&version) != INC_DC_FW_OK) {
        return 1;
    }
    return version.abi_version == INC_DC_FRAMEWORK_ABI_VERSION ? 0 : 2;
}
