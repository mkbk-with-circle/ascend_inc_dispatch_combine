#ifndef INC_DC_NATIVE_INC_SERVICE_H
#define INC_DC_NATIVE_INC_SERVICE_H

#include <cstdint>

#include "inc_dc_framework_c_api.h"

namespace inc::dc::single_stream {

struct NativeDispatchSession;
struct NativeCombineSession;
struct NativeIncService;
struct NativeIncServiceClient;

enum class NativeIncServiceOp : uint32_t {
    DISPATCH = 1u,
    COMBINE = 2u,
    DISPATCH_PREPARE = 3u,
};

struct NativeIncServiceConfig {
    uint32_t worker_world_size = 0u;
    uint16_t tcp_port = 0u;
    uint64_t stream = 0u;
    uint64_t timeout_ns = 0u;
    NativeDispatchSession *dispatch = nullptr;
    NativeCombineSession *combine = nullptr;
};

/* INC-side persistent host proxy. It owns only control sockets; native
 * sessions/stream remain caller-owned and must outlive the service. */
inc_dc_fw_status_t CreateNativeIncService(
    const NativeIncServiceConfig &config, NativeIncService **service);
inc_dc_fw_status_t NativeIncServiceRun(
    NativeIncService *service, uint32_t generations);
inc_dc_fw_status_t DestroyNativeIncService(NativeIncService *service);

/* Worker-side mailbox endpoint. Submit is called only after that worker's
 * native kernel has been enqueued, except DISPATCH_PREPARE which is submitted
 * after local clear and before enqueue. It replaces per-generation W+1 host
 * barriers with generation-checked service rendezvous. */
inc_dc_fw_status_t CreateNativeIncServiceClient(
    const char *host, uint16_t port, uint32_t worker_rank,
    uint64_t timeout_ns, NativeIncServiceClient **client);
inc_dc_fw_status_t NativeIncServiceSubmitAndWait(
    NativeIncServiceClient *client, NativeIncServiceOp operation,
    uint64_t generation);
inc_dc_fw_status_t DestroyNativeIncServiceClient(
    NativeIncServiceClient *client);

} // namespace inc::dc::single_stream

#endif
