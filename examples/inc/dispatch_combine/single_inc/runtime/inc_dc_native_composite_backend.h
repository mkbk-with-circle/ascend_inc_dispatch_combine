#ifndef INC_DC_NATIVE_COMPOSITE_BACKEND_H
#define INC_DC_NATIVE_COMPOSITE_BACKEND_H

#include "inc_dc_framework_c_api.h"

namespace inc::dc::single_stream {

struct NativeCompositeBackend;

// Combines independently qualified Dispatch and Combine providers behind one
// Framework/Easy communicator. Ownership of both child vtables remains with
// the caller and their sessions must outlive the composite.
inc_dc_fw_status_t CreateNativeCompositeBackend(
    const inc_dc_fw_backend_ops_t &dispatch,
    const inc_dc_fw_backend_ops_t &combine,
    NativeCompositeBackend **backend);
inc_dc_fw_status_t DestroyNativeCompositeBackend(
    NativeCompositeBackend *backend);
inc_dc_fw_status_t NativeCompositeBackendOps(
    NativeCompositeBackend *backend, inc_dc_fw_backend_ops_t *ops);

} // namespace inc::dc::single_stream

#endif
