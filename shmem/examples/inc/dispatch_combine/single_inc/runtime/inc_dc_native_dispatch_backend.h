#ifndef INC_DC_NATIVE_DISPATCH_BACKEND_H
#define INC_DC_NATIVE_DISPATCH_BACKEND_H

#include <cstdint>

#include "inc_dc_framework_c_api.h"
#include "inc_dc_single_inc_stream_plan_compiler.h"

namespace inc::dc::single_stream {

struct NativeDispatchSession;

struct NativeDispatchSessionConfig {
    uint8_t *symmetric_heap = nullptr;
    uint64_t symmetric_heap_bytes = 0u;
    uint32_t local_pe = 0u;
    const StreamPreparedWorkspace *prepared = nullptr;
};

/* Control-plane creation uploads immutable tables to an initialized heap. */
inc_dc_fw_status_t CreateNativeDispatchSession(
    const NativeDispatchSessionConfig &config,
    NativeDispatchSession **session);
inc_dc_fw_status_t DestroyNativeDispatchSession(
    NativeDispatchSession *session);

/* Worker ranks inject this vtable into Framework/Easy API. */
inc_dc_fw_status_t NativeDispatchBackendOps(
    NativeDispatchSession *session, inc_dc_fw_backend_ops_t *ops);

/* Optional registered-buffer fast path.  Returned pointers remain owned by
 * the session/symmetric heap and are valid until session destruction. */
inc_dc_fw_status_t NativeDispatchWorkerBuffers(
    NativeDispatchSession *session, void **input, uint64_t *input_bytes,
    void **output, uint64_t *output_bytes);
inc_dc_fw_status_t NativeDispatchLastProtocolCycles(
    NativeDispatchSession *session, uint64_t *cycles);

/* Qualification-only, one-shot device telemetry fault. After completion is
 * observed, Query writes device stats before the mandatory telemetry read;
 * normal launches execute no extra operation. */
inc_dc_fw_status_t NativeDispatchArmLaneError(
    NativeDispatchSession *session, uint32_t lane, uint32_t error);

/* Transitional SPMD control plane: reset generation-owned cachelines before
 * all ranks enter the enqueue barrier. Persistent INC service replaces it. */
inc_dc_fw_status_t NativeDispatchPrepareGeneration(
    NativeDispatchSession *session, uint64_t operation_generation);

/* Dedicated INC rank calls the same one-shot kernel for each generation. */
inc_dc_fw_status_t NativeDispatchIncEnqueue(
    NativeDispatchSession *session, uint64_t stream,
    uint64_t operation_generation, inc_dc_fw_backend_ticket_t *ticket);
inc_dc_fw_status_t NativeDispatchIncWaitAndRelease(
    NativeDispatchSession *session, inc_dc_fw_backend_ticket_t ticket,
    uint64_t timeout_ns);

} // namespace inc::dc::single_stream

#endif
