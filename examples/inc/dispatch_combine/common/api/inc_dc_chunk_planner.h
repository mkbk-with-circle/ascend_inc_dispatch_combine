#ifndef INC_DC_CHUNK_PLANNER_H
#define INC_DC_CHUNK_PLANNER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Pure host-side planner shared by Dispatch and Combine backends.
 *
 * The public operation is uint64-sized.  protocol_capacity is only the
 * largest row count representable by one device epoch.  workspace_bytes is
 * a bounded reusable ring; it never scales with logical_rows.
 */
typedef struct inc_dc_chunk_plan_desc {
    uint64_t logical_rows;
    uint64_t workspace_bytes;
    uint64_t fixed_workspace_bytes;
    uint64_t workspace_bytes_per_row;
    uint32_t protocol_capacity;
    uint32_t topk;
} inc_dc_chunk_plan_desc_t;

typedef struct inc_dc_chunk {
    uint64_t chunk_id;
    uint64_t global_row_begin;
    uint64_t row_count;
    uint64_t global_route_begin;
    uint64_t route_count;
    uint64_t workspace_bytes;
    uint32_t generation;
    uint32_t is_last;
} inc_dc_chunk_t;

typedef enum inc_dc_chunk_status {
    INC_DC_CHUNK_OK = 0,
    INC_DC_CHUNK_DONE = 1,
    INC_DC_CHUNK_INVALID_ARGUMENT = 2,
    INC_DC_CHUNK_OUT_OF_MEMORY = 3,
    INC_DC_CHUNK_OVERFLOW = 4
} inc_dc_chunk_status_t;

/*
 * Returns one deterministic chunk. generation is never zero and is suitable
 * for the operation_id/chunk_id/generation device tuple.
 */
inc_dc_chunk_status_t inc_dc_plan_chunk(
    const inc_dc_chunk_plan_desc_t *desc, uint64_t chunk_id,
    uint64_t operation_generation, inc_dc_chunk_t *chunk);

#ifdef __cplusplus
}
#endif

#endif
