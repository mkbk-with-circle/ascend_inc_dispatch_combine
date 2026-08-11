#ifndef INC_DC_INFERENCE_API_H
#define INC_DC_INFERENCE_API_H

/*
 * Prepared, allocation-free hot-path API for inference integrations.
 *
 * A session owns the Easy communicator.  A plan owns one persistent Dispatch
 * workspace and one persistent Combine workspace for an exact (tokens, topk)
 * shape.  Consequently one Dispatch and one Combine may be in flight on the
 * same plan at the same time.  Create a small pool of plans when more than one
 * request of the same operation must overlap.
 */

#include "inc_dc_easy_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define INC_DC_INFERENCE_ABI_VERSION 1u

typedef struct inc_dc_infer_session inc_dc_infer_session_t;
typedef struct inc_dc_infer_plan inc_dc_infer_plan_t;

typedef void *(*inc_dc_infer_allocate_fn)(
    uint64_t bytes, uint32_t alignment, void *context);
typedef void (*inc_dc_infer_free_fn)(void *pointer, void *context);

typedef struct inc_dc_infer_config {
    uint32_t struct_size;
    uint32_t abi_version;
    inc_dc_easy_comm_config_t communicator;
    inc_dc_infer_allocate_fn allocate_device;
    inc_dc_infer_free_fn free_device;
    void *allocator_context;
    uint64_t flags;
    uint64_t reserved[4];
} inc_dc_infer_config_t;

typedef struct inc_dc_infer_plan_desc {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t tokens;
    uint32_t topk;
    uint32_t reserved0;
    uint64_t flags;
    uint64_t reserved[4];
} inc_dc_infer_plan_desc_t;

typedef struct inc_dc_infer_io {
    uint32_t struct_size;
    uint32_t abi_version;
    void *input;
    void *output;
    void *weights;
    uint64_t input_rows;       /* zero => plan tokens */
    uint64_t output_rows;      /* zero => plan tokens */
    uint64_t weight_elements;
    uint32_t weight_dtype;     /* FP32 default */
    uint32_t reserved0;
    inc_dc_fw_route_desc_t route; /* optional when a static route is configured */
    inc_dc_fw_route_metadata_v1_t *metadata;
    uint64_t stream;           /* reinterpret_cast<uint64_t>(aclrtStream) */
    uint64_t operation_generation;
    uint64_t deadline_ns;
    uint64_t flags;
    uint64_t reserved[4];
} inc_dc_infer_io_t;

typedef struct inc_dc_infer_request {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t operation;
    uint32_t reserved0;
    inc_dc_fw_request_t framework_request;
    uint64_t plan_generation;
    uint64_t reserved[4];
} inc_dc_infer_request_t;

typedef struct inc_dc_infer_route {
    uint32_t struct_size;
    uint32_t abi_version;
    inc_dc_fw_route_handle_t framework_route;
    uint64_t session_generation;
    uint64_t reserved[3];
} inc_dc_infer_route_t;

typedef struct inc_dc_infer_plan_info {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t tokens;
    uint32_t topk;
    uint32_t reserved0;
    uint64_t dispatch_workspace_bytes;
    uint64_t combine_workspace_bytes;
    uint32_t dispatch_workspace_alignment;
    uint32_t combine_workspace_alignment;
    uint64_t reserved[4];
} inc_dc_infer_plan_info_t;

void inc_dc_infer_config_init(inc_dc_infer_config_t *config);
void inc_dc_infer_plan_desc_init(inc_dc_infer_plan_desc_t *description);
void inc_dc_infer_io_init(inc_dc_infer_io_t *io);

inc_dc_fw_status_t inc_dc_infer_session_create(
    const inc_dc_infer_config_t *config, inc_dc_infer_session_t **session);
inc_dc_fw_status_t inc_dc_infer_session_create_from_context(
    const inc_dc_infer_config_t *config, inc_dc_context_t *context,
    inc_dc_infer_session_t **session);
inc_dc_fw_status_t inc_dc_infer_session_get_capabilities(
    const inc_dc_infer_session_t *session,
    inc_dc_fw_capabilities_t *capabilities);
inc_dc_fw_status_t inc_dc_infer_session_get_stats(
    const inc_dc_infer_session_t *session, inc_dc_fw_context_stats_t *stats);

/* Setup path: queries and allocates both persistent device workspaces. */
inc_dc_fw_status_t inc_dc_infer_plan_create(
    inc_dc_infer_session_t *session,
    const inc_dc_infer_plan_desc_t *description,
    inc_dc_infer_plan_t **plan);
inc_dc_fw_status_t inc_dc_infer_plan_get_info(
    const inc_dc_infer_plan_t *plan, inc_dc_infer_plan_info_t *info);

/* Allocation-free hot path. */
inc_dc_fw_status_t inc_dc_infer_dispatch_async(
    inc_dc_infer_plan_t *plan, const inc_dc_infer_io_t *io,
    inc_dc_infer_request_t *request);
inc_dc_fw_status_t inc_dc_infer_combine_async(
    inc_dc_infer_plan_t *plan, const inc_dc_infer_io_t *io,
    inc_dc_infer_request_t *request);

/* Preserve the exact Dispatch route and consume it from Combine. */
inc_dc_fw_status_t inc_dc_infer_route_capture(
    inc_dc_infer_plan_t *plan,
    const inc_dc_infer_request_t *dispatch_request,
    const inc_dc_infer_io_t *dispatch_io,
    inc_dc_infer_route_t *route);
inc_dc_fw_status_t inc_dc_infer_combine_with_route_async(
    inc_dc_infer_plan_t *plan, const inc_dc_infer_route_t *route,
    const inc_dc_infer_io_t *io, inc_dc_infer_request_t *request);
inc_dc_fw_status_t inc_dc_infer_route_release(
    inc_dc_infer_session_t *session, inc_dc_infer_route_t *route);

inc_dc_fw_status_t inc_dc_infer_request_query(
    inc_dc_infer_plan_t *plan, const inc_dc_infer_request_t *request,
    uint32_t *state);
inc_dc_fw_status_t inc_dc_infer_request_wait(
    inc_dc_infer_plan_t *plan, const inc_dc_infer_request_t *request,
    uint64_t timeout_ns);
inc_dc_fw_status_t inc_dc_infer_request_cancel(
    inc_dc_infer_plan_t *plan, const inc_dc_infer_request_t *request);
inc_dc_fw_status_t inc_dc_infer_request_release(
    inc_dc_infer_plan_t *plan, inc_dc_infer_request_t *request);
inc_dc_fw_status_t inc_dc_infer_request_wait_and_release(
    inc_dc_infer_plan_t *plan, inc_dc_infer_request_t *request,
    uint64_t timeout_ns);

/* Both return BUSY instead of invalidating live work. */
inc_dc_fw_status_t inc_dc_infer_plan_destroy(inc_dc_infer_plan_t *plan);
inc_dc_fw_status_t inc_dc_infer_session_destroy(
    inc_dc_infer_session_t *session);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
