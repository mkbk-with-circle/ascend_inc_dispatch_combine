#ifndef INC_DC_NATIVE_COMBINE_BACKEND_H
#define INC_DC_NATIVE_COMBINE_BACKEND_H

#include <cstdint>
#include <vector>

#include "inc_dc_framework_c_api.h"
#include "inc_dc_native_combine_workspace.h"

namespace inc::dc::single_stream {

struct NativeCombineSession;

struct NativeCombineSessionConfig {
    uint8_t *symmetric_heap = nullptr;
    uint64_t symmetric_heap_bytes = 0u;
    uint32_t local_pe = 0u;
    uint32_t tokens_per_worker = 0u;
    uint32_t topk = 0u;
    const NativeCombinePreparedWorkspace *prepared = nullptr;
    const std::vector<uint64_t> *source_semantic_digests = nullptr;
};

inc_dc_fw_status_t CreateNativeCombineSession(
    const NativeCombineSessionConfig &config,
    NativeCombineSession **session);
inc_dc_fw_status_t DestroyNativeCombineSession(NativeCombineSession *session);

inc_dc_fw_status_t NativeCombineBackendOps(
    NativeCombineSession *session, inc_dc_fw_backend_ops_t *ops);

inc_dc_fw_status_t NativeCombineWorkerBuffers(
    NativeCombineSession *session, void **input, uint64_t *input_bytes,
    void **output, uint64_t *output_bytes);
inc_dc_fw_status_t NativeCombineLastProtocolCycles(
    NativeCombineSession *session, uint64_t *cycles);

/* Qualification-only one-shot producer telemetry fault. Query writes device
 * stats after completion is observed and before mandatory telemetry read. */
inc_dc_fw_status_t NativeCombineArmProducerError(
    NativeCombineSession *session, uint32_t lane, uint32_t error);

inc_dc_fw_status_t NativeCombineIncEnqueue(
    NativeCombineSession *session, uint64_t stream,
    uint64_t operation_generation, inc_dc_fw_backend_ticket_t *ticket);
inc_dc_fw_status_t NativeCombineIncWaitAndRelease(
    NativeCombineSession *session, inc_dc_fw_backend_ticket_t ticket,
    uint64_t timeout_ns);

} // namespace inc::dc::single_stream

#endif
