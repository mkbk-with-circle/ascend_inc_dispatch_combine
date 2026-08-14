#ifndef INC_DC_FRAMEWORK_C_API_H
#define INC_DC_FRAMEWORK_C_API_H

/*
 * Stable C ABI for framework and fused-operator integration.
 *
 * The ABI intentionally does not expose C++ objects, SHMEM ranks, INC owner
 * ranks, or aclrt headers.  A stream is an opaque caller-owned handle and all
 * tensors/routes/workspaces are caller-owned device descriptors.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INC_DC_FRAMEWORK_ABI_VERSION 2u
#define INC_DC_FRAMEWORK_MAX_DIMS 4u

typedef struct inc_dc_context inc_dc_context_t;
typedef struct inc_dc_plan inc_dc_plan_t;

typedef enum inc_dc_fw_status {
    INC_DC_FW_OK = 0,
    INC_DC_FW_INVALID_ARGUMENT = 1,
    INC_DC_FW_UNSUPPORTED = 2,
    INC_DC_FW_CAPACITY_EXCEEDED = 3,
    INC_DC_FW_LAYOUT_MISMATCH = 4,
    INC_DC_FW_STALE_HANDLE = 5,
    INC_DC_FW_NOT_READY = 6,
    INC_DC_FW_TIMEOUT = 7,
    INC_DC_FW_CANCELLED = 8,
    INC_DC_FW_BUSY = 9,
    INC_DC_FW_BACKEND_ERROR = 10,
    INC_DC_FW_INTERNAL = 11
} inc_dc_fw_status_t;

typedef enum inc_dc_fw_dtype {
    INC_DC_FW_DTYPE_FP16 = 0,
    INC_DC_FW_DTYPE_BF16 = 1,
    INC_DC_FW_DTYPE_FP32 = 2,
    INC_DC_FW_DTYPE_INT32 = 3,
    INC_DC_FW_DTYPE_INT64 = 4,
    INC_DC_FW_DTYPE_UINT8 = 5
} inc_dc_fw_dtype_t;

typedef enum inc_dc_fw_memory {
    INC_DC_FW_MEMORY_HOST = 0,
    INC_DC_FW_MEMORY_DEVICE = 1
} inc_dc_fw_memory_t;

typedef enum inc_dc_fw_route_format {
    INC_DC_FW_ROUTE_TOPK_DENSE = 0,
    INC_DC_FW_ROUTE_CSR = 1,
    INC_DC_FW_ROUTE_OPAQUE_DEVICE_PLAN = 2
} inc_dc_fw_route_format_t;

typedef enum inc_dc_fw_operation {
    INC_DC_FW_OP_DISPATCH = 1,
    INC_DC_FW_OP_COMBINE = 2
} inc_dc_fw_operation_t;

typedef enum inc_dc_fw_request_state {
    INC_DC_FW_REQUEST_FREE = 0,
    INC_DC_FW_REQUEST_SUBMITTED = 1,
    INC_DC_FW_REQUEST_COMPLETED = 2,
    INC_DC_FW_REQUEST_CANCELLED = 3,
    INC_DC_FW_REQUEST_FAILED = 4
} inc_dc_fw_request_state_t;

enum {
    INC_DC_FW_FEATURE_EXTERNAL_STREAM = 1ull << 0,
    INC_DC_FW_FEATURE_MULTI_INFLIGHT = 1ull << 1,
    INC_DC_FW_FEATURE_DEVICE_ROUTE = 1ull << 2,
    INC_DC_FW_FEATURE_DYNAMIC_TOKENS = 1ull << 3,
    INC_DC_FW_FEATURE_DISPATCH_COMBINE_OVERLAP = 1ull << 4,
    INC_DC_FW_FEATURE_GRAPH_SAFE_ENQUEUE = 1ull << 5,
    INC_DC_FW_FEATURE_CANCEL = 1ull << 6,
    INC_DC_FW_FEATURE_FUSED_OPERATOR_READY = 1ull << 7,
    /*
     * One public invocation may contain more rows than the device protocol
     * can hold at once.  The backend must page it through a bounded
     * workspace and preserve global route/result ordinals.
     */
    INC_DC_FW_FEATURE_INTERNAL_CHUNKING = 1ull << 8,
    INC_DC_FW_FEATURE_ROUTE_METADATA = 1ull << 9,
    INC_DC_FW_FEATURE_LOCAL_EXPERT_EXPAND = 1ull << 10,
    INC_DC_FW_FEATURE_SPARSE_DIRECT = 1ull << 11,
    /* Optional invocation.weights may use FP32 while payload remains FP16/BF16. */
    INC_DC_FW_FEATURE_FP32_WEIGHTS = 1ull << 12,
    /* Grouped-GEMM permutation, inverse map, and padded expert offsets. */
    INC_DC_FW_FEATURE_EXPERT_LAYOUT = 1ull << 13
};

enum {
    INC_DC_FW_INVOCATION_HAS_EXTENSIONS = 1ull << 0
};

typedef enum inc_dc_fw_extension_type {
    INC_DC_FW_EXT_ROUTE_METADATA_V1 = 1,
    INC_DC_FW_EXT_EXPERT_LAYOUT_V1 = 2
} inc_dc_fw_extension_type_t;

typedef struct inc_dc_fw_version {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t implementation_version;
    uint32_t reserved;
} inc_dc_fw_version_t;

typedef struct inc_dc_fw_capabilities {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t feature_bits;
    uint32_t max_world_size;
    uint32_t max_topk;
    uint32_t max_inflight;
    uint32_t workspace_alignment;
    uint64_t dtype_bits;
    uint64_t reserved[4];
} inc_dc_fw_capabilities_t;

typedef struct inc_dc_fw_route_desc {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t format;
    uint32_t memory_location;
    const void *data;
    uint64_t bytes;
    uint64_t semantic_digest;
    uint64_t generation;
    uint64_t reserved[2];
} inc_dc_fw_route_desc_t;

typedef struct inc_dc_fw_tensor_desc {
    uint32_t struct_size;
    uint32_t abi_version;
    void *data;
    uint32_t dtype;
    uint32_t memory_location;
    uint32_t ndim;
    uint32_t reserved0;
    int64_t dims[INC_DC_FRAMEWORK_MAX_DIMS];
    int64_t strides[INC_DC_FRAMEWORK_MAX_DIMS]; /* element strides */
    uint64_t reserved[2];
} inc_dc_fw_tensor_desc_t;

/*
 * Forward-compatible invocation extension chain.  invocation.reserved[0]
 * points to the first caller-owned HOST header when
 * INC_DC_FW_INVOCATION_HAS_EXTENSIONS is set.  The descriptors may point at
 * device tensors and must remain live until the request is released.
 */
typedef struct inc_dc_fw_extension_header {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t type;
    uint32_t reserved0;
    const struct inc_dc_fw_extension_header *next;
    uint64_t reserved[2];
} inc_dc_fw_extension_header_t;

/*
 * Metadata required by Megatron/vLLM-style MoE consumers.  Every tensor is
 * optional (data == NULL).  Dispatch fills requested outputs; combine accepts
 * the same mapping so duplicate experts on one rank expand locally and reduce
 * back to the original source token without another wire copy.
 *
 * expert_offsets:       [local_expert_count + 1], int64 semantics
 * source_token_indices: [physical_rows], int64 semantics
 * expert_ids:           [logical_assignments], int32 semantics
 * route_ordinals:       [logical_assignments], int32 semantics
 * assignment_weights:   [logical_assignments], plan dtype or fp32 backend ABI
 * dropped_mask:         [logical_assignments], uint8 semantics
 */
typedef struct inc_dc_fw_route_metadata_v1 {
    inc_dc_fw_extension_header_t header;
    inc_dc_fw_tensor_desc_t expert_offsets;
    inc_dc_fw_tensor_desc_t source_token_indices;
    inc_dc_fw_tensor_desc_t expert_ids;
    inc_dc_fw_tensor_desc_t route_ordinals;
    inc_dc_fw_tensor_desc_t assignment_weights;
    inc_dc_fw_tensor_desc_t dropped_mask;
    uint64_t local_expert_count;
    uint64_t physical_rows;
    uint64_t logical_assignments;
    uint64_t flags;
    uint64_t reserved[4];
} inc_dc_fw_route_metadata_v1_t;

/*
 * Optional dispatch output layout for Megatron/vLLM grouped GEMM.  All
 * tensors are caller-owned DEVICE buffers and may be omitted independently.
 * permutation_indices maps expert-major rows to source-token rows;
 * inverse_permutation maps those rows back for combine.  Padded offsets use
 * expert_alignment rows and never alter logical_rows.
 */
typedef struct inc_dc_fw_expert_layout_v1 {
    inc_dc_fw_extension_header_t header;
    inc_dc_fw_tensor_desc_t permutation_indices; /* int64 [logical_rows] */
    inc_dc_fw_tensor_desc_t inverse_permutation;  /* int64 [logical_rows] */
    inc_dc_fw_tensor_desc_t expert_offsets;       /* int64 [experts + 1] */
    inc_dc_fw_tensor_desc_t padded_expert_offsets;/* int64 [experts + 1] */
    inc_dc_fw_tensor_desc_t tokens_per_expert;    /* int64 [experts] */
    uint64_t local_expert_count;
    uint64_t logical_rows;
    uint32_t expert_alignment;
    uint32_t flags;
    uint64_t reserved[4];
} inc_dc_fw_expert_layout_v1_t;

typedef struct inc_dc_fw_plan_desc {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t session_id;
    uint64_t model_id;
    uint64_t process_group_id;
    uint64_t topology_generation;
    uint32_t worker_world_size;
    uint32_t worker_rank;
    /*
     * Device chunk capacity, not a public shape limit.  A backend advertising
     * INTERNAL_CHUNKING accepts larger logical shapes and pages them.
     */
    uint32_t max_tokens;
    uint32_t hidden_size;
    uint32_t max_topk;
    uint32_t dtype;
    uint64_t flags;
    inc_dc_fw_route_desc_t static_route;
    uint64_t reserved[4];
} inc_dc_fw_plan_desc_t;

typedef struct inc_dc_fw_shape {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t tokens;                 /* logical rows in the public operation */
    uint32_t hidden_size;
    uint32_t topk;
    uint32_t dtype;
    uint32_t reserved0;
    uint64_t flags;
    uint64_t reserved[1];
} inc_dc_fw_shape_t;

typedef struct inc_dc_fw_workspace {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t bytes;
    uint32_t alignment;
    uint32_t reserved0;
    uint64_t query_token;
    uint64_t reserved[3];
} inc_dc_fw_workspace_t;

typedef struct inc_dc_fw_invocation {
    uint32_t struct_size;
    uint32_t abi_version;
    inc_dc_fw_shape_t shape;
    inc_dc_fw_tensor_desc_t input;
    inc_dc_fw_tensor_desc_t output;
    inc_dc_fw_tensor_desc_t weights; /* data may be NULL */
    inc_dc_fw_route_desc_t route;    /* data may be NULL when plan is static */
    void *workspace;
    uint64_t workspace_bytes;
    uint64_t workspace_query_token;
    uint64_t stream;                 /* reinterpret_cast<uint64_t>(aclrtStream) */
    uint64_t operation_generation;
    uint64_t deadline_ns;            /* absolute monotonic deadline, 0 = none */
    uint64_t flags;
    /* reserved[0] is the optional inc_dc_fw_extension_header_t pointer. */
    uint64_t reserved[4];
} inc_dc_fw_invocation_t;

typedef struct inc_dc_fw_request {
    uint64_t slot;
    uint64_t generation;
} inc_dc_fw_request_t;

typedef struct inc_dc_fw_route_handle {
    uint64_t slot;
    uint64_t generation;
} inc_dc_fw_route_handle_t;

/* Monotonic context counters plus a consistent live-state snapshot. */
typedef struct inc_dc_fw_context_stats {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t dispatch_enqueued;
    uint64_t combine_enqueued;
    uint64_t completed;
    uint64_t cancelled;
    uint64_t failed;
    uint64_t wait_timeouts;
    uint64_t enqueue_failures;
    uint32_t live_requests;
    uint32_t peak_live_requests;
    uint32_t live_plans;
    uint32_t live_workspace_leases;
    uint64_t reserved[4];
} inc_dc_fw_context_stats_t;

typedef struct inc_dc_fw_backend_ticket {
    uint64_t value;
    uint64_t generation;
} inc_dc_fw_backend_ticket_t;

typedef struct inc_dc_fw_backend_ops {
    uint32_t struct_size;
    uint32_t abi_version;
    void *backend_context;
    inc_dc_fw_status_t (*get_capabilities)(
        void *backend_context, inc_dc_fw_capabilities_t *capabilities);
    inc_dc_fw_status_t (*query_workspace)(
        void *backend_context, const inc_dc_fw_plan_desc_t *plan,
        uint32_t operation, const inc_dc_fw_shape_t *shape,
        inc_dc_fw_workspace_t *workspace);
    inc_dc_fw_status_t (*enqueue)(
        void *backend_context, const inc_dc_fw_plan_desc_t *plan,
        uint32_t operation, const inc_dc_fw_invocation_t *invocation,
        inc_dc_fw_backend_ticket_t *ticket);
    inc_dc_fw_status_t (*query)(
        void *backend_context, inc_dc_fw_backend_ticket_t ticket,
        uint32_t *state);
    inc_dc_fw_status_t (*wait)(
        void *backend_context, inc_dc_fw_backend_ticket_t ticket,
        uint64_t timeout_ns);
    inc_dc_fw_status_t (*cancel)(
        void *backend_context, inc_dc_fw_backend_ticket_t ticket);
    inc_dc_fw_status_t (*release)(
        void *backend_context, inc_dc_fw_backend_ticket_t ticket);
    uint64_t reserved[4];
} inc_dc_fw_backend_ops_t;

/*
 * Backend callback contract:
 * - get_capabilities/query_workspace are control-plane calls.
 * - enqueue must only enqueue on invocation->stream; it must not synchronize
 *   the stream/device, allocate hot-path host memory, or wait for another op.
 *   Backends advertising MULTI_INFLIGHT must accept concurrent enqueue calls
 *   for different streams; the framework does not hold its context lock
 *   while enqueue runs.
 * - query/cancel/release must be non-blocking and must not re-enter this API.
 * - wait is the only callback allowed to block.  The framework lock is not
 *   held while wait runs, so unrelated streams remain enqueueable.
 * - query writes an inc_dc_fw_request_state_t value to state.
 */

typedef struct inc_dc_fw_context_config {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t max_inflight;
    uint32_t max_workspace_queries;
    uint64_t flags;
    inc_dc_fw_backend_ops_t backend;
    uint64_t reserved[4];
} inc_dc_fw_context_config_t;

inc_dc_fw_status_t inc_dc_fw_get_version(inc_dc_fw_version_t *version);
const char *inc_dc_fw_status_string(inc_dc_fw_status_t status);

inc_dc_fw_status_t inc_dc_fw_context_create(
    const inc_dc_fw_context_config_t *config, inc_dc_context_t **context);
inc_dc_fw_status_t inc_dc_fw_context_get_capabilities(
    const inc_dc_context_t *context, inc_dc_fw_capabilities_t *capabilities);
inc_dc_fw_status_t inc_dc_fw_context_get_stats(
    const inc_dc_context_t *context, inc_dc_fw_context_stats_t *stats);
inc_dc_fw_status_t inc_dc_fw_context_destroy(inc_dc_context_t *context);

inc_dc_fw_status_t inc_dc_fw_plan_create(
    inc_dc_context_t *context, const inc_dc_fw_plan_desc_t *desc,
    inc_dc_plan_t **plan);
inc_dc_fw_status_t inc_dc_fw_plan_retain(inc_dc_plan_t *plan);
inc_dc_fw_status_t inc_dc_fw_plan_release(inc_dc_plan_t *plan);

inc_dc_fw_status_t inc_dc_fw_query_workspace(
    inc_dc_plan_t *plan, uint32_t operation, const inc_dc_fw_shape_t *shape,
    inc_dc_fw_workspace_t *workspace);
inc_dc_fw_status_t inc_dc_fw_workspace_release(
    inc_dc_plan_t *plan, uint64_t workspace_query_token);

inc_dc_fw_status_t inc_dc_fw_dispatch_async(
    inc_dc_plan_t *plan, const inc_dc_fw_invocation_t *invocation,
    inc_dc_fw_request_t *request);
inc_dc_fw_status_t inc_dc_fw_combine_async(
    inc_dc_plan_t *plan, const inc_dc_fw_invocation_t *invocation,
    inc_dc_fw_request_t *request);

/*
 * Capture the exact dispatch route as an opaque, generation-checked handle.
 * route may be NULL to capture the plan's static route.  Caller-owned route
 * storage remains live until the handle is released.  A handle pinned by an
 * in-flight combine request returns BUSY on release.
 */
inc_dc_fw_status_t inc_dc_fw_route_handle_create(
    inc_dc_plan_t *plan, inc_dc_fw_request_t dispatch_request,
    const inc_dc_fw_route_desc_t *route,
    inc_dc_fw_route_handle_t *route_handle);
inc_dc_fw_status_t inc_dc_fw_combine_with_route_async(
    inc_dc_plan_t *plan, inc_dc_fw_route_handle_t route_handle,
    const inc_dc_fw_invocation_t *invocation,
    inc_dc_fw_request_t *request);
inc_dc_fw_status_t inc_dc_fw_route_handle_release(
    inc_dc_plan_t *plan, inc_dc_fw_route_handle_t route_handle);

inc_dc_fw_status_t inc_dc_fw_request_query(
    inc_dc_context_t *context, inc_dc_fw_request_t request,
    uint32_t *state);
inc_dc_fw_status_t inc_dc_fw_request_wait(
    inc_dc_context_t *context, inc_dc_fw_request_t request,
    uint64_t timeout_ns);
inc_dc_fw_status_t inc_dc_fw_request_cancel(
    inc_dc_context_t *context, inc_dc_fw_request_t request);
inc_dc_fw_status_t inc_dc_fw_request_release(
    inc_dc_context_t *context, inc_dc_fw_request_t request);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
