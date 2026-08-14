#ifndef INC_DC_DISPATCH_ROUTE_PLAN_H
#define INC_DC_DISPATCH_ROUTE_PLAN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * One bounded device chunk of a logical Dispatch operation.
 *
 * logical_row_begin is 64-bit and may exceed 2^32.  row_count is deliberately
 * per-chunk: callers page an arbitrarily large operation through this
 * compiler and reuse the same output buffers.
 */
typedef struct inc_dc_dispatch_route_plan_desc {
    uint64_t logical_row_begin;
    uint32_t row_count;
    uint32_t topk;
    uint32_t worker_count;
    uint32_t experts_per_worker;
    const int32_t *expert_ids;          /* row_count * topk */
    const uint64_t *cell_ordinal_base;  /* worker_count * experts_per_worker */
} inc_dc_dispatch_route_plan_desc_t;

typedef struct inc_dc_dispatch_route_entry {
    uint64_t logical_row;
    uint32_t chunk_row;
    uint32_t source_segment_offset;
    uint32_t topk_slot;
    uint32_t expert_id;
} inc_dc_dispatch_route_entry_t;

typedef struct inc_dc_dispatch_route_plan_buffers {
    uint64_t *cell_offsets;
    uint64_t cell_offsets_capacity;
    inc_dc_dispatch_route_entry_t *entries;
    uint64_t entries_capacity;
    uint64_t *cell_counts;
    uint64_t cell_counts_capacity;
} inc_dc_dispatch_route_plan_buffers_t;

typedef enum inc_dc_dispatch_route_plan_status {
    INC_DC_DISPATCH_ROUTE_PLAN_OK = 0,
    INC_DC_DISPATCH_ROUTE_PLAN_INVALID_ARGUMENT = 1,
    INC_DC_DISPATCH_ROUTE_PLAN_CAPACITY = 2,
    INC_DC_DISPATCH_ROUTE_PLAN_OVERFLOW = 3,
    INC_DC_DISPATCH_ROUTE_PLAN_BAD_EXPERT = 4
} inc_dc_dispatch_route_plan_status_t;

/*
 * Query mode: buffers may be NULL; required sizes are returned.
 * Compile mode: all buffers must meet the returned capacities.
 */
inc_dc_dispatch_route_plan_status_t inc_dc_dispatch_route_plan_compile(
    const inc_dc_dispatch_route_plan_desc_t *desc,
    inc_dc_dispatch_route_plan_buffers_t *buffers,
    uint64_t *required_offsets, uint64_t *required_entries,
    uint64_t *required_counts);

#ifdef __cplusplus
}
#endif

#endif
