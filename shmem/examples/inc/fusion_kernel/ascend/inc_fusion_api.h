#ifndef INC_FUSION_API_H
#define INC_FUSION_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inc_fusion_prepared_plan inc_fusion_prepared_plan_t;
typedef struct inc_fusion_persistent_service inc_fusion_persistent_service_t;
typedef struct inc_fusion_worker_executor inc_fusion_worker_executor_t;

typedef enum inc_fusion_status {
    INC_FUSION_OK = 0,
    INC_FUSION_INVALID_ARGUMENT = 1,
    INC_FUSION_INVALID_PLAN = 2,
    INC_FUSION_BUFFER_TOO_SMALL = 3,
    INC_FUSION_BUSY = 4,
    INC_FUSION_RUNTIME_ERROR = 5,
} inc_fusion_status_t;

typedef enum inc_fusion_role {
    INC_FUSION_ROLE_WORKER = 1,
    INC_FUSION_ROLE_INC = 2,
} inc_fusion_role_t;

typedef enum inc_fusion_execution_flag {
    INC_FUSION_EXEC_DEFAULT = 0,
    // Diagnostic/attribution baseline: alternate INC Dispatch and Combine
    // token waves instead of allowing the two disjoint AIV cohorts to overlap.
    INC_FUSION_EXEC_SERIALIZE_INC_DC = 1u << 0,
    INC_FUSION_EXEC_REMOTE_INC_SERVICE = 1u << 1,
    // Measured serial_inc baseline.  It preserves the production kernel,
    // compute implementation, memory layout and launch boundary while gating
    // token waves to D -> FFN -> C order.  This intentionally avoids charging
    // the baseline extra kernel launches or setup allocations.
    INC_FUSION_EXEC_STRICT_SERIAL_PIPELINE = 1u << 2,
    INC_FUSION_EXEC_WORKER_DIRECT_SHMEM = 1u << 3,
    // w13/w2 are contiguous [E,H,2I]/[E,I,H] (Catlass RowMajor B).
    INC_FUSION_EXEC_WEIGHT_B_ROW_MAJOR = 1u << 4,
    // The persistent INC service also relays the tiny per-wave route-count
    // histogram before each request. Workers upload one disjoint block to the
    // INC and the INC fans the complete table back out. This keeps routing on
    // the same star data plane as Dispatch/Combine and avoids a worker HCCL
    // collective on every MoE layer.
    INC_FUSION_EXEC_INC_ROUTE_RELAY = 1u << 5,
    // Return the complete rank-ordered token output to every worker through
    // the INC instead of a rank-local slice.
    INC_FUSION_EXEC_GLOBAL_OUTPUT_FANOUT = 1u << 6,
} inc_fusion_execution_flag_t;

typedef enum inc_fusion_service_request_status {
    INC_FUSION_SERVICE_PENDING = 0,
    INC_FUSION_SERVICE_SUCCESS = 1,
    INC_FUSION_SERVICE_BAD_ARGS = 2,
    INC_FUSION_SERVICE_OPERATOR_ERROR = 3,
} inc_fusion_service_request_status_t;

typedef struct inc_fusion_plan_desc {
    uint32_t live_aiv;
    uint32_t live_aic;
    uint32_t worker_count;
    uint32_t rank;
    uint32_t inc_pe;
    uint32_t hidden;
    uint32_t intermediate;
    uint32_t expert_count;
    uint32_t topk;
    uint32_t token_count;
    uint32_t tokens_per_wave;
    uint32_t slot_count;
    uint32_t service_ring_size;
    uint32_t activation_waves;
    uint32_t spin_cap;
} inc_fusion_plan_desc_t;

typedef struct inc_fusion_plan_info {
    uint64_t symmetric_bytes;
    uint64_t worker_workspace_bytes;
    uint64_t inc_workspace_bytes;
    uint64_t resource_fingerprint;
    uint32_t wave_count;
    uint32_t local_expert_count;
    uint32_t worker_dispatch_aiv;
    uint32_t worker_combine_aiv;
    uint32_t worker_compute_aiv;
    uint32_t inc_dispatch_aiv;
    uint32_t inc_combine_aiv;
    uint32_t kernel_args_bytes;
    uint32_t remote_service_ring_size;
    uint32_t fusion_abi_version;
    uint64_t remote_service_bytes;
} inc_fusion_plan_info_t;

// Scratch addresses inside the prepared symmetric heap that may be reused by
// the worker-only route-count exchange before a fusion request is published.
// The exchange completes on every worker before Dispatch may overwrite these
// bulk arenas, so it adds no persistent allocation to the plan.
typedef struct inc_fusion_route_exchange_info {
    uint64_t counts_off;
    uint64_t doorbells_off;
} inc_fusion_route_exchange_info_t;

// All pointers are device addresses. The prepared API performs no allocation
// and no route interpretation on the enqueue path.
typedef struct inc_fusion_device_bindings {
    void *symmetric_base;
    void *input;
    void *output;
    void *w13;
    void *w2;
    void *dispatch_rows;
    void *assignments;
    void *waves;
    void *expert_owner;
    void *expert_local_index;
    void *worker_pes;
    // [worker_count] uint32_t on device. Optional for uniform legacy calls;
    // production dynamic-batch integrations should always bind it.
    void *active_token_counts;
    void *group_lists;
    void *workspace;
    void *system_workspace;
    uint64_t ffts_addr;
    uint32_t execution_flags;
    uint32_t reserved32;
} inc_fusion_device_bindings_t;

typedef struct inc_fusion_service_result {
    uint64_t ticket;
    uint64_t request_id;
    uint32_t complete;
    uint32_t status;
    uint32_t operator_error;
    uint32_t reserved32;
} inc_fusion_service_result_t;

typedef struct inc_fusion_worker_executor_info {
    void *workspace;
    void *device_args;
    const void *host_args;
    uint64_t workspace_bytes;
    uint32_t ring_size;
    uint32_t owns_workspace;
    uint64_t submitted;
    uint32_t kernel_args_bytes;
    uint32_t reserved32;
} inc_fusion_worker_executor_info_t;

void inc_fusion_plan_desc_init(inc_fusion_plan_desc_t *desc);

inc_fusion_status_t inc_fusion_prepared_plan_create(
    const inc_fusion_plan_desc_t *desc,
    const uint32_t *expert_owner,
    const uint32_t *expert_local_index,
    inc_fusion_prepared_plan_t **out);

void inc_fusion_prepared_plan_destroy(inc_fusion_prepared_plan_t *plan);

inc_fusion_status_t inc_fusion_prepared_plan_info(
    const inc_fusion_prepared_plan_t *plan,
    inc_fusion_plan_info_t *info);

inc_fusion_status_t inc_fusion_prepared_route_exchange_info(
    const inc_fusion_prepared_plan_t *plan,
    inc_fusion_route_exchange_info_t *info);

// Fills the stable device ABI into caller-owned pinned/host memory. The
// caller may copy it to a preallocated device argument record asynchronously.
inc_fusion_status_t inc_fusion_prepared_build_args(
    const inc_fusion_prepared_plan_t *plan,
    inc_fusion_role_t role,
    uint64_t generation,
    const inc_fusion_device_bindings_t *bindings,
    void *args_buffer,
    size_t args_buffer_bytes);

// Enqueue only: no allocation, plan compilation, topology query or host
// synchronization. device_args must contain the record built above.
inc_fusion_status_t inc_fusion_prepared_enqueue(
    const inc_fusion_prepared_plan_t *plan,
    inc_fusion_role_t role,
    void *symmetric_base,
    void *device_args,
    void *stream);

// Prepared worker-side hot-path executor. `static_bindings` supplies
// symmetric_base, expert placement, worker_pes, optional active-token vector,
// system_workspace and FFTS address. If workspace is null, the executor owns
// one allocation sized by the plan; otherwise caller-owned workspace is used.
// The plan and all caller-owned static buffers must outlive the executor.
inc_fusion_status_t inc_fusion_worker_executor_create(
    const inc_fusion_prepared_plan_t *plan,
    const inc_fusion_device_bindings_t *static_bindings,
    uint32_t ring_size,
    inc_fusion_worker_executor_t **out);

// Allocation-free, synchronization-free enqueue. Dynamic bindings provide
// input/output/weights/route/waves/group-lists for this request. Static fields
// are taken from create(); active_token_counts may be overridden per request.
// A ring slot still in use returns INC_FUSION_BUSY instead of being overwritten.
inc_fusion_status_t inc_fusion_worker_executor_enqueue(
    inc_fusion_worker_executor_t *executor,
    uint64_t generation,
    const inc_fusion_device_bindings_t *dynamic_bindings,
    void *stream);

inc_fusion_status_t inc_fusion_worker_executor_info(
    const inc_fusion_worker_executor_t *executor,
    inc_fusion_worker_executor_info_t *info);

// Destroy waits only for this executor's in-flight ring events, then releases
// executor-owned storage. Call outside the timed forward path.
void inc_fusion_worker_executor_destroy(
    inc_fusion_worker_executor_t *executor);

// Launch one long-lived INC vector kernel. Creation may allocate control
// memory and an internal stop stream; submit performs no allocation and does
// not synchronize. `plan` and `symmetric_base` must outlive the service.
inc_fusion_status_t inc_fusion_persistent_service_create(
    const inc_fusion_prepared_plan_t *plan,
    void *symmetric_base,
    uint32_t ring_size,
    void *service_stream,
    inc_fusion_persistent_service_t **out);

// Cross-process form used by vLLM. Control/descriptors/args/metadata are
// placed at prepared offsets inside symmetric_base. `static_bindings` must
// provide INC workspace, worker_pes and optional system workspace/FFTS.
// Worker rank zero publishes each request from inside its worker kernel; no
// host call crosses process boundaries on the forward hot path.
inc_fusion_status_t inc_fusion_remote_service_create(
    const inc_fusion_prepared_plan_t *plan,
    const inc_fusion_device_bindings_t *static_bindings,
    void *service_stream,
    inc_fusion_persistent_service_t **out);

// Complete the two-phase remote startup after every PE has initialized its
// symmetric mirror and the application has executed one setup barrier.
inc_fusion_status_t inc_fusion_remote_service_start(
    inc_fusion_persistent_service_t *service);

// Publish one already-built INC FusionKernelArgs record. Use the same stream
// that prepared/copied device_args, or establish an event dependency first.
// The returned ticket is strictly increasing and defines completion order.
inc_fusion_status_t inc_fusion_persistent_service_submit(
    inc_fusion_persistent_service_t *service,
    void *device_args,
    uint64_t request_id,
    void *submit_stream,
    uint64_t *ticket);

// Non-destructive completion query. It copies one cache line from the INC and
// may synchronize that tiny D2H copy; it never synchronizes worker streams.
inc_fusion_status_t inc_fusion_persistent_service_query(
    inc_fusion_persistent_service_t *service,
    uint64_t ticket,
    inc_fusion_service_result_t *result);

// Request an orderly exit while the server is waiting for a descriptor.
// Requests already inside the push-only protocol must finish before stop can
// return. Destroy performs stop automatically when needed.
inc_fusion_status_t inc_fusion_persistent_service_stop(
    inc_fusion_persistent_service_t *service);

void inc_fusion_persistent_service_destroy(
    inc_fusion_persistent_service_t *service);

// Advanced integration hook for device-side completion/event chaining.
void *inc_fusion_persistent_service_device_control(
    const inc_fusion_persistent_service_t *service);

#ifdef __cplusplus
}
#endif

#endif
