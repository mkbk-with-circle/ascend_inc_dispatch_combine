#include "inc_dc_chunk_planner.h"

#include <cassert>
#include <cstdint>
#include <limits>

int main()
{
    inc_dc_chunk_plan_desc_t d{};
    d.logical_rows = 4097;
    d.workspace_bytes = 4096 + 1024 * 64;
    d.fixed_workspace_bytes = 4096;
    d.workspace_bytes_per_row = 64;
    d.protocol_capacity = 4096;
    d.topk = 8;

    inc_dc_chunk_t c{};
    assert(inc_dc_plan_chunk(&d, 0, 7, &c) == INC_DC_CHUNK_OK);
    assert(c.global_row_begin == 0 && c.row_count == 1024);
    assert(c.global_route_begin == 0 && c.route_count == 8192);
    assert(c.is_last == 0);
    assert(inc_dc_plan_chunk(&d, 4, 7, &c) == INC_DC_CHUNK_OK);
    assert(c.global_row_begin == 4096 && c.row_count == 1);
    assert(c.global_route_begin == 32768 && c.route_count == 8);
    assert(c.is_last == 1);
    assert(inc_dc_plan_chunk(&d, 5, 7, &c) == INC_DC_CHUNK_DONE);

    d.logical_rows = (uint64_t{1} << 32) + 17;
    d.workspace_bytes = 4096 + 4096 * 64;
    assert(inc_dc_plan_chunk(&d, uint64_t{1} << 20, 9, &c) ==
           INC_DC_CHUNK_OK);
    assert(c.global_row_begin == (uint64_t{1} << 32));
    assert(c.row_count == 17 && c.is_last == 1);

    d.workspace_bytes = d.fixed_workspace_bytes;
    assert(inc_dc_plan_chunk(&d, 0, 1, &c) ==
           INC_DC_CHUNK_OUT_OF_MEMORY);

    d.workspace_bytes = std::numeric_limits<uint64_t>::max();
    d.workspace_bytes_per_row = 0;
    d.logical_rows = std::numeric_limits<uint64_t>::max();
    d.protocol_capacity = std::numeric_limits<uint32_t>::max();
    d.topk = std::numeric_limits<uint32_t>::max();
    assert(inc_dc_plan_chunk(&d, 2, 1, &c) ==
           INC_DC_CHUNK_OVERFLOW);
    return 0;
}
