#include "inc_dc_native_composite_backend.h"

#include <cassert>
#include <cstdint>

using namespace inc::dc::single_stream;

namespace {

struct Fake {
    uint32_t expected_operation = 0u;
    uint64_t generation = 0u;
    bool live = false;
};

inc_dc_fw_status_t Caps(void *, inc_dc_fw_capabilities_t *caps)
{
    *caps = {};
    caps->struct_size = sizeof(*caps);
    caps->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    caps->feature_bits = INC_DC_FW_FEATURE_EXTERNAL_STREAM |
                         INC_DC_FW_FEATURE_DEVICE_ROUTE;
    caps->max_world_size = 8u;
    caps->max_topk = 64u;
    caps->max_inflight = 1u;
    caps->workspace_alignment = 256u;
    caps->dtype_bits = 1ull << INC_DC_FW_DTYPE_FP16;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Workspace(void *opaque, const inc_dc_fw_plan_desc_t *,
                             uint32_t operation, const inc_dc_fw_shape_t *,
                             inc_dc_fw_workspace_t *workspace)
{
    auto *fake = static_cast<Fake *>(opaque);
    if (operation != fake->expected_operation) return INC_DC_FW_UNSUPPORTED;
    workspace->bytes = operation * 256u;
    workspace->alignment = 256u;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Enqueue(void *opaque, const inc_dc_fw_plan_desc_t *,
                           uint32_t operation,
                           const inc_dc_fw_invocation_t *,
                           inc_dc_fw_backend_ticket_t *ticket)
{
    auto *fake = static_cast<Fake *>(opaque);
    if (operation != fake->expected_operation || fake->live)
        return INC_DC_FW_UNSUPPORTED;
    fake->live = true;
    ticket->value = 11u + operation;
    ticket->generation = ++fake->generation;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Query(void *opaque, inc_dc_fw_backend_ticket_t,
                         uint32_t *state)
{
    auto *fake = static_cast<Fake *>(opaque);
    if (!fake->live) return INC_DC_FW_STALE_HANDLE;
    *state = INC_DC_FW_REQUEST_COMPLETED;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Wait(void *opaque, inc_dc_fw_backend_ticket_t, uint64_t)
{
    return static_cast<Fake *>(opaque)->live ? INC_DC_FW_OK :
        INC_DC_FW_STALE_HANDLE;
}

inc_dc_fw_status_t Cancel(void *opaque, inc_dc_fw_backend_ticket_t)
{
    return static_cast<Fake *>(opaque)->live ? INC_DC_FW_OK :
        INC_DC_FW_STALE_HANDLE;
}

inc_dc_fw_status_t Release(void *opaque, inc_dc_fw_backend_ticket_t)
{
    auto *fake = static_cast<Fake *>(opaque);
    if (!fake->live) return INC_DC_FW_STALE_HANDLE;
    fake->live = false;
    return INC_DC_FW_OK;
}

inc_dc_fw_backend_ops_t Ops(Fake *fake)
{
    inc_dc_fw_backend_ops_t ops{};
    ops.struct_size = sizeof(ops);
    ops.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    ops.backend_context = fake;
    ops.get_capabilities = Caps;
    ops.query_workspace = Workspace;
    ops.enqueue = Enqueue;
    ops.query = Query;
    ops.wait = Wait;
    ops.cancel = Cancel;
    ops.release = Release;
    return ops;
}

} // namespace

int main()
{
    Fake dispatch{INC_DC_FW_OP_DISPATCH};
    Fake combine{INC_DC_FW_OP_COMBINE};
    NativeCompositeBackend *backend = nullptr;
    assert(CreateNativeCompositeBackend(
               Ops(&dispatch), Ops(&combine), &backend) == INC_DC_FW_OK);
    inc_dc_fw_backend_ops_t composite{};
    assert(NativeCompositeBackendOps(backend, &composite) == INC_DC_FW_OK);
    inc_dc_fw_capabilities_t caps{};
    assert(composite.get_capabilities(composite.backend_context, &caps) ==
           INC_DC_FW_OK);
    assert(caps.max_world_size == 8u && caps.max_topk == 64u);

    inc_dc_fw_workspace_t workspace{};
    assert(composite.query_workspace(
               composite.backend_context, nullptr, INC_DC_FW_OP_DISPATCH,
               nullptr, &workspace) == INC_DC_FW_OK);
    assert(workspace.bytes == INC_DC_FW_OP_DISPATCH * 256u);
    assert(composite.query_workspace(
               composite.backend_context, nullptr, INC_DC_FW_OP_COMBINE,
               nullptr, &workspace) == INC_DC_FW_OK);
    assert(workspace.bytes == INC_DC_FW_OP_COMBINE * 256u);

    inc_dc_fw_backend_ticket_t d{};
    inc_dc_fw_backend_ticket_t c{};
    assert(composite.enqueue(composite.backend_context, nullptr,
                             INC_DC_FW_OP_DISPATCH, nullptr, &d) ==
           INC_DC_FW_OK);
    assert(composite.enqueue(composite.backend_context, nullptr,
                             INC_DC_FW_OP_COMBINE, nullptr, &c) ==
           INC_DC_FW_OK);
    assert(d.value != c.value);
    assert(DestroyNativeCompositeBackend(backend) == INC_DC_FW_BUSY);
    assert(composite.wait(composite.backend_context, d, 1u) == INC_DC_FW_OK);
    assert(composite.release(composite.backend_context, d) == INC_DC_FW_OK);
    assert(composite.query(composite.backend_context, d, nullptr) ==
           INC_DC_FW_STALE_HANDLE);
    assert(composite.cancel(composite.backend_context, c) == INC_DC_FW_OK);
    assert(composite.release(composite.backend_context, c) == INC_DC_FW_OK);
    assert(DestroyNativeCompositeBackend(backend) == INC_DC_FW_OK);
    return 0;
}
