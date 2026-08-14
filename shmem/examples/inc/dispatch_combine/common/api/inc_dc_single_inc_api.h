#ifndef INC_DC_SINGLE_INC_API_H
#define INC_DC_SINGLE_INC_API_H

/*
 * Recommended application-facing API for the qualified single-INC V1 path.
 *
 * Deployment code creates one exact-shape operator.  Model code then only
 * supplies tensors/stream/generation to Dispatch and Combine.  The lower
 * Inference -> Easy -> Framework layers stay private implementation details.
 */

#include "inc_dc_inference_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define INC_DC_SINGLE_INC_ABI_VERSION 1u

typedef struct inc_dc_single_inc inc_dc_single_inc_t;
typedef inc_dc_infer_io_t inc_dc_single_inc_io_t;
typedef inc_dc_infer_request_t inc_dc_single_inc_request_t;
typedef inc_dc_infer_route_t inc_dc_single_inc_route_t;

/* Optional deployment hooks, used by the native persistent INC controller. */
typedef inc_dc_fw_status_t (*inc_dc_single_inc_control_fn)(
    void *context, uint32_t operation, uint64_t generation);

/* One-time deployment and exact-shape preparation. */
typedef struct inc_dc_single_inc_config {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t session_id;
    uint64_t model_id;
    uint64_t process_group_id;
    uint64_t topology_generation;
    uint32_t worker_world_size;
    uint32_t worker_rank;
    uint32_t hidden_size;
    uint32_t dtype;
    uint64_t tokens;
    uint32_t topk;
    uint32_t max_tokens_per_chunk; /* zero => min(tokens, UINT32_MAX) */
    uint32_t max_inflight;         /* zero => 2 (one Dispatch + one Combine) */
    uint32_t max_workspace_queries;/* zero => 2 */
    inc_dc_fw_route_desc_t static_route; /* optional device token plan */
    inc_dc_fw_backend_ops_t backend;     /* native composite backend */
    inc_dc_infer_allocate_fn allocate_device;
    inc_dc_infer_free_fn free_device;
    void *allocator_context;
    inc_dc_single_inc_control_fn before_enqueue;
    inc_dc_single_inc_control_fn after_enqueue;
    void *control_context;
    uint64_t default_timeout_ns; /* zero is passed through to backend wait */
    uint64_t flags;
    uint64_t reserved[4];
} inc_dc_single_inc_config_t;

void inc_dc_single_inc_config_init(inc_dc_single_inc_config_t *config);
void inc_dc_single_inc_io_init(inc_dc_single_inc_io_t *io);

inc_dc_fw_status_t inc_dc_single_inc_create(
    const inc_dc_single_inc_config_t *config,
    inc_dc_single_inc_t **single_inc);
inc_dc_fw_status_t inc_dc_single_inc_create_from_context(
    const inc_dc_single_inc_config_t *config, inc_dc_context_t *context,
    inc_dc_single_inc_t **single_inc);

/*
 * Simplest blocking path:
 *   dispatch -> expert compute -> combine -> route_release.
 * Dispatch returns the exact route that Combine must reuse.
 */
inc_dc_fw_status_t inc_dc_single_inc_dispatch(
    inc_dc_single_inc_t *single_inc, const inc_dc_single_inc_io_t *io,
    inc_dc_single_inc_route_t *route);
inc_dc_fw_status_t inc_dc_single_inc_combine(
    inc_dc_single_inc_t *single_inc,
    const inc_dc_single_inc_route_t *dispatch_route,
    const inc_dc_single_inc_io_t *io);

/* Allocation-free asynchronous path for Dispatch/Combine overlap. */
inc_dc_fw_status_t inc_dc_single_inc_dispatch_async(
    inc_dc_single_inc_t *single_inc, const inc_dc_single_inc_io_t *io,
    inc_dc_single_inc_request_t *request,
    inc_dc_single_inc_route_t *route);
inc_dc_fw_status_t inc_dc_single_inc_combine_async(
    inc_dc_single_inc_t *single_inc,
    const inc_dc_single_inc_route_t *dispatch_route,
    const inc_dc_single_inc_io_t *io,
    inc_dc_single_inc_request_t *request);

inc_dc_fw_status_t inc_dc_single_inc_request_query(
    inc_dc_single_inc_t *single_inc,
    const inc_dc_single_inc_request_t *request, uint32_t *state);
inc_dc_fw_status_t inc_dc_single_inc_request_wait(
    inc_dc_single_inc_t *single_inc,
    const inc_dc_single_inc_request_t *request, uint64_t timeout_ns);
inc_dc_fw_status_t inc_dc_single_inc_request_cancel(
    inc_dc_single_inc_t *single_inc,
    const inc_dc_single_inc_request_t *request);
inc_dc_fw_status_t inc_dc_single_inc_request_release(
    inc_dc_single_inc_t *single_inc, inc_dc_single_inc_request_t *request);
inc_dc_fw_status_t inc_dc_single_inc_request_wait_and_release(
    inc_dc_single_inc_t *single_inc, inc_dc_single_inc_request_t *request,
    uint64_t timeout_ns);
inc_dc_fw_status_t inc_dc_single_inc_route_release(
    inc_dc_single_inc_t *single_inc, inc_dc_single_inc_route_t *route);

inc_dc_fw_status_t inc_dc_single_inc_get_plan_info(
    const inc_dc_single_inc_t *single_inc, inc_dc_infer_plan_info_t *info);
inc_dc_fw_status_t inc_dc_single_inc_get_stats(
    const inc_dc_single_inc_t *single_inc,
    inc_dc_fw_context_stats_t *stats);

/* Returns BUSY while a request or captured route is still live. */
inc_dc_fw_status_t inc_dc_single_inc_destroy(
    inc_dc_single_inc_t *single_inc);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
