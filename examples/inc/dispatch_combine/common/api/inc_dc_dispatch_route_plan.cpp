#include "inc_dc_dispatch_route_plan.h"

#include <limits>
#include <vector>

namespace {

bool MulOverflow(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr ||
        (a != 0u && b > std::numeric_limits<uint64_t>::max() / a)) {
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

extern "C" inc_dc_dispatch_route_plan_status_t
inc_dc_dispatch_route_plan_compile(
    const inc_dc_dispatch_route_plan_desc_t *desc,
    inc_dc_dispatch_route_plan_buffers_t *buffers,
    uint64_t *required_offsets, uint64_t *required_entries,
    uint64_t *required_counts)
{
    if (desc == nullptr || required_offsets == nullptr ||
        required_entries == nullptr || required_counts == nullptr ||
        desc->topk == 0u || desc->worker_count == 0u ||
        desc->experts_per_worker == 0u ||
        (desc->row_count != 0u && desc->expert_ids == nullptr)) {
        return INC_DC_DISPATCH_ROUTE_PLAN_INVALID_ARGUMENT;
    }

    uint64_t cells = 0u;
    uint64_t route_slots = 0u;
    if (MulOverflow(desc->worker_count, desc->experts_per_worker,
                    &cells) ||
        MulOverflow(desc->row_count, desc->topk, &route_slots) ||
        cells == std::numeric_limits<uint64_t>::max()) {
        return INC_DC_DISPATCH_ROUTE_PLAN_OVERFLOW;
    }
    *required_offsets = cells + 1u;
    *required_entries = route_slots;
    *required_counts = cells;

    if (buffers == nullptr || buffers->cell_offsets == nullptr ||
        buffers->entries == nullptr || buffers->cell_counts == nullptr) {
        return INC_DC_DISPATCH_ROUTE_PLAN_OK;
    }
    if (buffers->cell_offsets_capacity < cells + 1u ||
        buffers->entries_capacity < route_slots ||
        buffers->cell_counts_capacity < cells) {
        return INC_DC_DISPATCH_ROUTE_PLAN_CAPACITY;
    }

    std::vector<uint64_t> cursor;
    try {
        cursor.assign(static_cast<size_t>(cells), 0u);
    } catch (...) {
        return INC_DC_DISPATCH_ROUTE_PLAN_CAPACITY;
    }
    for (uint64_t cell = 0u; cell < cells; ++cell) {
        buffers->cell_counts[cell] = 0u;
    }

    for (uint64_t route = 0u; route < route_slots; ++route) {
        const int32_t expert = desc->expert_ids[route];
        if (expert < 0 || static_cast<uint64_t>(expert) >= cells) {
            return INC_DC_DISPATCH_ROUTE_PLAN_BAD_EXPERT;
        }
        ++buffers->cell_counts[static_cast<uint32_t>(expert)];
    }

    uint64_t prefix = 0u;
    for (uint64_t cell = 0u; cell < cells; ++cell) {
        buffers->cell_offsets[cell] = prefix;
        cursor[cell] = prefix;
        if (AddOverflow(prefix, buffers->cell_counts[cell], &prefix)) {
            return INC_DC_DISPATCH_ROUTE_PLAN_OVERFLOW;
        }
    }
    buffers->cell_offsets[cells] = prefix;
    if (prefix != route_slots) {
        return INC_DC_DISPATCH_ROUTE_PLAN_OVERFLOW;
    }

    for (uint64_t route = 0u; route < route_slots; ++route) {
        const uint32_t expert =
            static_cast<uint32_t>(desc->expert_ids[route]);
        const uint64_t out_index = cursor[expert]++;
        const uint64_t chunk_row = route / desc->topk;
        uint64_t logical_row = 0u;
        if (AddOverflow(desc->logical_row_begin, chunk_row,
                        &logical_row)) {
            return INC_DC_DISPATCH_ROUTE_PLAN_OVERFLOW;
        }
        uint64_t ordinal =
            out_index - buffers->cell_offsets[expert];
        if (desc->cell_ordinal_base != nullptr &&
            AddOverflow(ordinal, desc->cell_ordinal_base[expert],
                        &ordinal)) {
            return INC_DC_DISPATCH_ROUTE_PLAN_OVERFLOW;
        }
        if (ordinal > std::numeric_limits<uint32_t>::max()) {
            return INC_DC_DISPATCH_ROUTE_PLAN_OVERFLOW;
        }
        inc_dc_dispatch_route_entry_t &entry =
            buffers->entries[out_index];
        entry.logical_row = logical_row;
        entry.chunk_row = static_cast<uint32_t>(chunk_row);
        entry.source_segment_offset = static_cast<uint32_t>(ordinal);
        entry.topk_slot =
            static_cast<uint32_t>(route % desc->topk);
        entry.expert_id = expert;
    }
    return INC_DC_DISPATCH_ROUTE_PLAN_OK;
}
