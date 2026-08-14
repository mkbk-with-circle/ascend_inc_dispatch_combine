#include "inc_dc_dispatch_route_plan.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

static void Run(uint32_t workers, uint32_t rows, uint32_t topk,
                bool single_hot, uint64_t global_begin)
{
    const uint32_t epp = 8u;
    const uint64_t cells = static_cast<uint64_t>(workers) * epp;
    std::vector<int32_t> experts(static_cast<size_t>(rows) * topk);
    for (uint64_t i = 0; i < experts.size(); ++i) {
        experts[i] = single_hot ? 0 : static_cast<int32_t>(i % cells);
    }
    std::vector<uint64_t> ordinal_base(cells, 17u);
    inc_dc_dispatch_route_plan_desc_t desc{
        global_begin, rows, topk, workers, epp, experts.data(),
        ordinal_base.data()};
    uint64_t no = 0, ne = 0, nc = 0;
    const auto query_status = inc_dc_dispatch_route_plan_compile(
        &desc, nullptr, &no, &ne, &nc);
    assert(query_status == INC_DC_DISPATCH_ROUTE_PLAN_OK);
    assert(no == cells + 1u);
    assert(ne == static_cast<uint64_t>(rows) * topk);
    assert(nc == cells);

    std::vector<uint64_t> offsets(no), counts(nc);
    std::vector<inc_dc_dispatch_route_entry_t> entries(ne);
    inc_dc_dispatch_route_plan_buffers_t buffers{
        offsets.data(), offsets.size(), entries.data(), entries.size(),
        counts.data(), counts.size()};
    const auto compile_status = inc_dc_dispatch_route_plan_compile(
        &desc, &buffers, &no, &ne, &nc);
    assert(compile_status == INC_DC_DISPATCH_ROUTE_PLAN_OK);
    assert(offsets.front() == 0u && offsets.back() == ne);
    uint64_t sum = 0u;
    for (uint64_t cell = 0; cell < cells; ++cell) {
        assert(offsets[cell] <= offsets[cell + 1u]);
        assert(offsets[cell + 1u] - offsets[cell] == counts[cell]);
        sum += counts[cell];
        uint32_t prior = 0u;
        for (uint64_t i = offsets[cell]; i < offsets[cell + 1u]; ++i) {
            assert(entries[i].expert_id == cell);
            assert(entries[i].logical_row >= global_begin);
            assert(entries[i].logical_row <
                   global_begin + rows);
            assert(entries[i].topk_slot < topk);
            assert(entries[i].source_segment_offset >= 17u);
            assert(i == offsets[cell] ||
                   entries[i].source_segment_offset == prior + 1u);
            prior = entries[i].source_segment_offset;
        }
    }
    assert(sum == ne);
}

int main()
{
    for (uint32_t workers : {2u, 4u, 8u}) {
        for (uint32_t topk : {1u, 2u, 4u, 6u, 8u, 16u}) {
            Run(workers, 0u, topk, false, 0u);
            Run(workers, 1u, topk, false, 0u);
            Run(workers, 2048u, topk, false,
                (uint64_t{1} << 32) + 13u);
            Run(workers, 257u, topk, true, 91u);
        }
    }

    int32_t bad = 16;
    inc_dc_dispatch_route_plan_desc_t desc{
        0u, 1u, 1u, 2u, 8u, &bad, nullptr};
    uint64_t offsets[17]{}, counts[16]{};
    inc_dc_dispatch_route_entry_t entry{};
    inc_dc_dispatch_route_plan_buffers_t buffers{
        offsets, 17u, &entry, 1u, counts, 16u};
    uint64_t no = 0, ne = 0, nc = 0;
    auto status = inc_dc_dispatch_route_plan_compile(
        &desc, &buffers, &no, &ne, &nc);
    assert(status == INC_DC_DISPATCH_ROUTE_PLAN_BAD_EXPERT);

    desc.logical_row_begin = std::numeric_limits<uint64_t>::max();
    bad = 0;
    status = inc_dc_dispatch_route_plan_compile(
        &desc, &buffers, &no, &ne, &nc);
    assert(status == INC_DC_DISPATCH_ROUTE_PLAN_OK);
    desc.row_count = 2u;
    int32_t two[2]{0, 0};
    desc.expert_ids = two;
    inc_dc_dispatch_route_entry_t entries[2]{};
    buffers.entries = entries;
    buffers.entries_capacity = 2u;
    status = inc_dc_dispatch_route_plan_compile(
        &desc, &buffers, &no, &ne, &nc);
    assert(status == INC_DC_DISPATCH_ROUTE_PLAN_OVERFLOW);
    return 0;
}
