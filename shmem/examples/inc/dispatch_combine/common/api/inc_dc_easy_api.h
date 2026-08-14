#ifndef INC_DC_EASY_API_H
#define INC_DC_EASY_API_H

/*
 * Small, stable C facade for applications that want to call the single-INC
 * Dispatch/Combine service without constructing framework descriptors.
 *
 * The deployment layer supplies one initialized native backend vtable to
 * inc_dc_easy_comm_create().  Model/operator code only uses workspace query,
 * dispatch/combine, request completion, and destroy below.
 */

#include "inc_dc_framework_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define INC_DC_EASY_ABI_VERSION 1u
#define INC_DC_EASY_TOKEN_PLAN_MAGIC 0x31505445u /* 'ETP1' */
#define INC_DC_EASY_TOKEN_PLAN_ABI_VERSION 1u

typedef struct inc_dc_easy_comm inc_dc_easy_comm_t;

typedef struct inc_dc_easy_comm_config {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t session_id;
    uint64_t model_id;
    uint64_t process_group_id;
    uint64_t topology_generation;
    uint32_t worker_world_size;
    uint32_t worker_rank;
    uint32_t max_tokens_per_chunk;
    uint32_t hidden_size;
    uint32_t max_topk;
    uint32_t dtype;
    uint32_t max_inflight;
    uint32_t max_workspace_queries;
    uint64_t flags;
    inc_dc_fw_route_desc_t static_route; /* optional */

    /*
     * Internal deployment boundary.  A framework/backend adapter fills this
     * once during communicator initialization.  Per-op application code must
     * not replace or mutate it.
     */
    inc_dc_fw_backend_ops_t backend;
    uint64_t reserved[4];
} inc_dc_easy_comm_config_t;

typedef struct inc_dc_easy_op {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t tokens;
    uint64_t input_rows;  /* zero => tokens */
    uint64_t output_rows; /* zero => tokens */
    uint32_t topk;
    uint32_t weight_dtype; /* FP32 default; ignored when weights == NULL */
    void *input;
    void *output;
    void *weights;        /* optional contiguous [weight_elements] */
    uint64_t weight_elements;
    inc_dc_fw_route_desc_t route; /* optional if communicator has static route */
    void *workspace;
    uint64_t workspace_bytes;
    uint64_t workspace_query_token;
    uint64_t stream; /* reinterpret_cast<uint64_t>(aclrtStream) */
    uint64_t operation_generation;
    uint64_t deadline_ns;
    inc_dc_fw_route_metadata_v1_t *metadata; /* optional caller-owned header */
    uint64_t flags;
    uint64_t reserved[4];
} inc_dc_easy_op_t;

/*
 * Canonical dense token plan consumed by Dispatch and reused by Combine.
 * Assignment (token, topk_slot) is stored at token * topk + topk_slot.
 * The blob is built in host memory, copied byte-for-byte to device memory,
 * and passed with route format INC_DC_FW_ROUTE_TOPK_DENSE.
 */
typedef struct inc_dc_easy_token_assignment_v1 {
    int32_t expert_id;
    uint32_t destination_rank;
    uint32_t weight_bits; /* IEEE-754 float32 bits */
    uint32_t flags;
} inc_dc_easy_token_assignment_v1_t;

typedef struct inc_dc_easy_token_plan_header_v1 {
    uint32_t magic;
    uint16_t abi_major;
    uint16_t abi_minor;
    uint32_t header_bytes;
    uint32_t assignment_bytes;
    uint64_t total_bytes;
    uint64_t tokens;
    uint64_t assignment_count;
    uint64_t global_physical_rows;
    uint32_t topk;
    uint32_t worker_world_size;
    uint32_t experts_per_worker;
    uint32_t flags;
    uint64_t generation;
    uint64_t semantic_digest;
    uint64_t reserved[4];
} inc_dc_easy_token_plan_header_v1_t;

typedef struct inc_dc_easy_token_plan_desc {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t tokens;
    uint32_t topk;
    uint32_t worker_world_size;
    uint32_t worker_rank;
    uint32_t experts_per_worker;
    const int32_t *expert_ids;        /* HOST [tokens * topk] */
    const uint32_t *destination_ranks;/* HOST optional [tokens * topk] */
    const float *weights;             /* HOST optional; NULL means 1.0 */
    uint64_t generation;
    uint64_t flags;
    uint64_t reserved[4];
} inc_dc_easy_token_plan_desc_t;

typedef struct inc_dc_easy_token_plan_info {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t bytes;
    uint64_t semantic_digest;
    uint64_t generation;
    uint64_t logical_assignments;
    uint64_t global_physical_rows;
    uint64_t local_assignments;
    uint64_t local_physical_rows;
    uint64_t reserved[4];
} inc_dc_easy_token_plan_info_t;

/* Fill versioned structures with safe defaults. */
void inc_dc_easy_comm_config_init(inc_dc_easy_comm_config_t *config);
void inc_dc_easy_op_init(inc_dc_easy_op_t *operation);
void inc_dc_easy_route_device_init(
    inc_dc_fw_route_desc_t *route, uint32_t format, const void *data,
    uint64_t bytes, uint64_t semantic_digest, uint64_t generation);

/*
 * Explicit test/integration plan construction. Query bytes first, allocate a
 * host buffer, build, then copy exactly info.bytes to caller-owned device
 * memory. destination_ranks may be NULL only when contiguous expert
 * placement can be derived as expert_id / experts_per_worker.
 */
void inc_dc_easy_token_plan_desc_init(
    inc_dc_easy_token_plan_desc_t *description);
inc_dc_fw_status_t inc_dc_easy_token_plan_query(
    const inc_dc_easy_token_plan_desc_t *description, uint64_t *bytes);
inc_dc_fw_status_t inc_dc_easy_token_plan_build(
    const inc_dc_easy_token_plan_desc_t *description, void *host_plan,
    uint64_t host_plan_capacity, inc_dc_easy_token_plan_info_t *info);
void inc_dc_easy_token_plan_device_route_init(
    inc_dc_fw_route_desc_t *route, const void *device_plan,
    const inc_dc_easy_token_plan_info_t *info);

/* Collective/control-plane initialization; call once per worker process. */
inc_dc_fw_status_t inc_dc_easy_comm_create(
    const inc_dc_easy_comm_config_t *config, inc_dc_easy_comm_t **comm);
/*
 * Bind Easy API to a caller-owned framework context that was initialized by
 * the portability/session layer.  The communicator creates/releases only its
 * plan; context lifetime remains with the caller.
 */
inc_dc_fw_status_t inc_dc_easy_comm_create_from_context(
    const inc_dc_easy_comm_config_t *config, inc_dc_context_t *context,
    inc_dc_easy_comm_t **comm);
inc_dc_fw_status_t inc_dc_easy_comm_get_capabilities(
    const inc_dc_easy_comm_t *comm,
    inc_dc_fw_capabilities_t *capabilities);
inc_dc_fw_status_t inc_dc_easy_comm_get_stats(
    const inc_dc_easy_comm_t *comm, inc_dc_fw_context_stats_t *stats);

/* Query once per shape and allocate caller-owned aligned device workspace. */
inc_dc_fw_status_t inc_dc_easy_workspace_query(
    inc_dc_easy_comm_t *comm, uint32_t operation, uint64_t tokens,
    uint32_t topk, inc_dc_fw_workspace_t *workspace);
inc_dc_fw_status_t inc_dc_easy_workspace_release(
    inc_dc_easy_comm_t *comm, uint64_t workspace_query_token);

/* Asynchronous: success means enqueued on operation->stream, not completed. */
inc_dc_fw_status_t inc_dc_easy_dispatch_async(
    inc_dc_easy_comm_t *comm, const inc_dc_easy_op_t *operation,
    inc_dc_fw_request_t *request);
inc_dc_fw_status_t inc_dc_easy_combine_async(
    inc_dc_easy_comm_t *comm, const inc_dc_easy_op_t *operation,
    inc_dc_fw_request_t *request);
inc_dc_fw_status_t inc_dc_easy_route_handle_create(
    inc_dc_easy_comm_t *comm, inc_dc_fw_request_t dispatch_request,
    const inc_dc_easy_op_t *dispatch_operation,
    inc_dc_fw_route_handle_t *route_handle);
inc_dc_fw_status_t inc_dc_easy_combine_with_route_async(
    inc_dc_easy_comm_t *comm, inc_dc_fw_route_handle_t route_handle,
    const inc_dc_easy_op_t *operation, inc_dc_fw_request_t *request);
inc_dc_fw_status_t inc_dc_easy_route_handle_release(
    inc_dc_easy_comm_t *comm, inc_dc_fw_route_handle_t route_handle);

inc_dc_fw_status_t inc_dc_easy_request_query(
    inc_dc_easy_comm_t *comm, inc_dc_fw_request_t request,
    uint32_t *state);
inc_dc_fw_status_t inc_dc_easy_request_wait(
    inc_dc_easy_comm_t *comm, inc_dc_fw_request_t request,
    uint64_t timeout_ns);
inc_dc_fw_status_t inc_dc_easy_request_cancel(
    inc_dc_easy_comm_t *comm, inc_dc_fw_request_t request);
inc_dc_fw_status_t inc_dc_easy_request_release(
    inc_dc_easy_comm_t *comm, inc_dc_fw_request_t request);

/* Wait then release.  A timeout leaves the request live and queryable. */
inc_dc_fw_status_t inc_dc_easy_request_wait_and_release(
    inc_dc_easy_comm_t *comm, inc_dc_fw_request_t request,
    uint64_t timeout_ns);

/* Returns BUSY while requests are live; never destroys a borrowed context. */
inc_dc_fw_status_t inc_dc_easy_comm_destroy(inc_dc_easy_comm_t *comm);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
