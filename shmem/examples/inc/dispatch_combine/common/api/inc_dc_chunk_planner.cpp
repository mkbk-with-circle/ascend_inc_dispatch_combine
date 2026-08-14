#include "inc_dc_chunk_planner.h"

#include <algorithm>
#include <limits>

namespace {

bool MulOverflow(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr) {
        return true;
    }
    if (a != 0u && b > std::numeric_limits<uint64_t>::max() / a) {
        return true;
    }
    *out = a * b;
    return false;
}

bool AddOverflow(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr ||
        b > std::numeric_limits<uint64_t>::max() - a) {
        return true;
    }
    *out = a + b;
    return false;
}

} // namespace

extern "C" inc_dc_chunk_status_t inc_dc_plan_chunk(
    const inc_dc_chunk_plan_desc_t *desc, uint64_t chunk_id,
    uint64_t operation_generation, inc_dc_chunk_t *chunk)
{
    if (desc == nullptr || chunk == nullptr ||
        desc->protocol_capacity == 0u || desc->topk == 0u ||
        operation_generation == 0u) {
        return INC_DC_CHUNK_INVALID_ARGUMENT;
    }

    uint64_t workspace_capacity =
        static_cast<uint64_t>(desc->protocol_capacity);
    if (desc->workspace_bytes_per_row != 0u) {
        if (desc->workspace_bytes <= desc->fixed_workspace_bytes) {
            return INC_DC_CHUNK_OUT_OF_MEMORY;
        }
        workspace_capacity =
            (desc->workspace_bytes - desc->fixed_workspace_bytes) /
            desc->workspace_bytes_per_row;
        workspace_capacity =
            std::min(workspace_capacity,
                     static_cast<uint64_t>(desc->protocol_capacity));
    } else if (desc->workspace_bytes < desc->fixed_workspace_bytes) {
        return INC_DC_CHUNK_OUT_OF_MEMORY;
    }
    if (workspace_capacity == 0u) {
        return INC_DC_CHUNK_OUT_OF_MEMORY;
    }

    uint64_t begin = 0u;
    if (MulOverflow(chunk_id, workspace_capacity, &begin)) {
        return INC_DC_CHUNK_OVERFLOW;
    }
    if (begin >= desc->logical_rows) {
        return INC_DC_CHUNK_DONE;
    }
    const uint64_t rows =
        std::min(workspace_capacity, desc->logical_rows - begin);

    uint64_t route_begin = 0u;
    uint64_t route_count = 0u;
    uint64_t variable_workspace = 0u;
    uint64_t used_workspace = 0u;
    if (MulOverflow(begin, desc->topk, &route_begin) ||
        MulOverflow(rows, desc->topk, &route_count) ||
        MulOverflow(rows, desc->workspace_bytes_per_row,
                    &variable_workspace) ||
        AddOverflow(desc->fixed_workspace_bytes, variable_workspace,
                    &used_workspace)) {
        return INC_DC_CHUNK_OVERFLOW;
    }

    *chunk = {};
    chunk->chunk_id = chunk_id;
    chunk->global_row_begin = begin;
    chunk->row_count = rows;
    chunk->global_route_begin = route_begin;
    chunk->route_count = route_count;
    chunk->workspace_bytes = used_workspace;
    chunk->generation = static_cast<uint32_t>(
        (operation_generation + chunk_id) & 0xffffffffu);
    if (chunk->generation == 0u) {
        chunk->generation = 1u;
    }
    chunk->is_last = (begin + rows == desc->logical_rows) ? 1u : 0u;
    return INC_DC_CHUNK_OK;
}
