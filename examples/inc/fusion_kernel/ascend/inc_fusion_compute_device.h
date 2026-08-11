#ifndef INC_FUSION_COMPUTE_DEVICE_H
#define INC_FUSION_COMPUTE_DEVICE_H

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_m.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_operator.h"
#include "lib/activation/swiglu.h"
#include "shmem.h"

namespace inc::fusion::compute {

__aicore__ inline bool WaitGeneration(__gm__ uint8_t *line,
                                      uint64_t generation,
                                      uint32_t spin_cap)
{
    __gm__ uint64_t *value = reinterpret_cast<__gm__ uint64_t *>(line);
    uint32_t spins = 0u;
    while (*value != generation) {
        if ((spins & (spins < 64u ? 0u : 31u)) == 0u)
            dcci_cacheline(line);
        ++spins;
        if (spin_cap != 0u && spins >= spin_cap) return false;
    }
    return true;
}

__aicore__ inline void PublishGeneration(__gm__ uint8_t *line,
                                         uint64_t generation)
{
    *reinterpret_cast<__gm__ uint64_t *>(line) = generation;
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(line);
}

__aicore__ inline bool WaitAllGenerations(__gm__ uint8_t *lines,
                                          uint32_t producers,
                                          uint64_t generation,
                                          uint32_t spin_cap)
{
    for (uint32_t producer = 0u; producer < producers; ++producer) {
        if (!WaitGeneration(lines +
                static_cast<uint64_t>(producer) * 64u,
                generation, spin_cap))
            return false;
    }
    return true;
}

// BF16 grouped GEMM with an expert-granular readiness gate. The outer token
// wave is owned by the caller; this routine starts each expert as soon as the
// Dispatch AIV cohort publishes its packed rows.
__aicore__ inline bool RunGroupedBf16(
    __gm__ bfloat16_t *a, __gm__ bfloat16_t *b,
    __gm__ bfloat16_t *c, __gm__ int64_t *group_list,
    __gm__ uint8_t *input_ready, __gm__ uint8_t *output_ready,
    uint32_t input_ready_stride, uint32_t input_ready_producers,
    uint32_t output_ready_stride,
    uint32_t experts, uint32_t n, uint32_t k,
    uint64_t generation, uint32_t spin_cap)
{
    using namespace Catlass;
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::ColumnMajor;
    using LayoutC = layout::RowMajor;
    using ArchTag = Arch::AtlasA2;
    constexpr bool kUnitFlag = true;
    constexpr bool kShuffleK = true;
    using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsync<
        1, 2, 2, 2, 1, kUnitFlag, kShuffleK>;
    using L1Tile = GemmShape<128, 256, 256>;
    using L0Tile = GemmShape<128, 256, 64>;
    using AType = Gemm::GemmType<bfloat16_t, LayoutA>;
    using BType = Gemm::GemmType<bfloat16_t, LayoutB>;
    using CType = Gemm::GemmType<bfloat16_t, LayoutC>;
    using BlockMmad = Gemm::Block::BlockMmad<
        DispatchPolicy, L1Tile, L0Tile, AType, BType, CType>;
    using Scheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

    Arch::Resource<ArchTag> resource;
    BlockMmad mmad(resource);
    Scheduler scheduler;
    AscendC::GlobalTensor<bfloat16_t> gm_a;
    AscendC::GlobalTensor<bfloat16_t> gm_c;
    gm_a.SetGlobalBuffer(a);
    gm_c.SetGlobalBuffer(c);
    const uint32_t core = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t cores = static_cast<uint32_t>(AscendC::GetBlockNum());
    int64_t a_off = 0;
    int64_t b_off = 0;
    int64_t c_off = 0;
    int64_t previous = 0;
    uint32_t start_core = 0u;

    for (uint32_t expert = 0u; expert < experts; ++expert) {
        __gm__ uint8_t *ready = input_ready +
            static_cast<uint64_t>(expert) * input_ready_stride * 64u;
        if (!WaitAllGenerations(ready, input_ready_producers,
                                generation, spin_cap))
            return false;
        const int64_t end = group_list[expert];
        if (end < previous) return false;
        const uint32_t m = static_cast<uint32_t>(end - previous);
        const GemmCoord shape{m, n, k};
        LayoutA layout_a{m, k};
        LayoutB layout_b{k, n};
        LayoutC layout_c{m, n};
        scheduler.Update(shape, MakeCoord(L1Tile::M, L1Tile::N));
        const uint32_t loops = scheduler.GetCoreLoops();
        AscendC::GlobalTensor<bfloat16_t> gm_b;
        gm_b.SetGlobalBuffer(b + b_off);
        if (m <= L1Tile::M)
            gm_b.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        const uint32_t first = core < start_core
            ? core + cores - start_core : core - start_core;
        for (uint32_t loop = first; loop < loops; loop += cores) {
            const GemmCoord block = scheduler.GetBlockCoord(loop);
            const GemmCoord actual = scheduler.GetActualBlockShape(block);
            const MatrixCoord ao{block.m() * L1Tile::M,
                                 block.k() * L1Tile::K};
            const MatrixCoord bo{block.k() * L1Tile::K,
                                 block.n() * L1Tile::N};
            const MatrixCoord co{block.m() * L1Tile::M,
                                 block.n() * L1Tile::N};
            mmad(gm_a[a_off + layout_a.GetOffset(ao)], layout_a,
                 gm_b[layout_b.GetOffset(bo)], layout_b,
                 gm_c[c_off + layout_c.GetOffset(co)], layout_c, actual);
        }
        if constexpr (BlockMmad::DispatchPolicy::ASYNC)
            mmad.SynchronizeBlock();
        PublishGeneration(output_ready +
            (static_cast<uint64_t>(expert) * output_ready_stride + core) * 64u,
            generation);
        a_off += static_cast<int64_t>(m) * k;
        b_off += static_cast<int64_t>(k) * n;
        c_off += static_cast<int64_t>(m) * n;
        previous = end;
        start_core = loops == 0u ? start_core : (start_core + loops) % cores;
    }
    return true;
}

// One expert-row slice on all Cube cores. The caller owns readiness and uses
// this primitive to interleave GMM1/GMM2 slices with AIV SwiGLU work while the
// outer scheduler remains token-wave based.
template <class LayoutB>
__aicore__ inline void RunBf16SliceLayout(
    __gm__ bfloat16_t *a, __gm__ bfloat16_t *b,
    __gm__ bfloat16_t *c, uint32_t m, uint32_t n, uint32_t k,
    uint32_t start_core)
{
    using namespace Catlass;
    using LayoutA = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    using ArchTag = Arch::AtlasA2;
    constexpr bool kUnitFlag = true;
    constexpr bool kShuffleK = true;
    using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsync<
        1, 2, 2, 2, 1, kUnitFlag, kShuffleK>;
    using L1Tile = GemmShape<128, 256, 256>;
    using L0Tile = GemmShape<128, 256, 64>;
    using AType = Gemm::GemmType<bfloat16_t, LayoutA>;
    using BType = Gemm::GemmType<bfloat16_t, LayoutB>;
    using CType = Gemm::GemmType<bfloat16_t, LayoutC>;
    using BlockMmad = Gemm::Block::BlockMmad<
        DispatchPolicy, L1Tile, L0Tile, AType, BType, CType>;
    using Scheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

    Arch::Resource<ArchTag> resource;
    BlockMmad block_mmad(resource);
    Scheduler scheduler;
    const GemmCoord shape{m, n, k};
    LayoutA layout_a{m, k};
    LayoutB layout_b{k, n};
    LayoutC layout_c{m, n};
    scheduler.Update(shape, MakeCoord(L1Tile::M, L1Tile::N));
    const uint32_t core = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t cores = static_cast<uint32_t>(AscendC::GetBlockNum());
    const uint32_t loops = scheduler.GetCoreLoops();
    const uint32_t first = core < start_core
        ? core + cores - start_core : core - start_core;
    AscendC::GlobalTensor<bfloat16_t> gm_a;
    AscendC::GlobalTensor<bfloat16_t> gm_b;
    AscendC::GlobalTensor<bfloat16_t> gm_c;
    gm_a.SetGlobalBuffer(a);
    gm_b.SetGlobalBuffer(b);
    gm_c.SetGlobalBuffer(c);
    if (m <= L1Tile::M)
        gm_b.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    for (uint32_t loop = first; loop < loops; loop += cores) {
        const GemmCoord block = scheduler.GetBlockCoord(loop);
        const GemmCoord actual = scheduler.GetActualBlockShape(block);
        const MatrixCoord ao{block.m() * L1Tile::M,
                             block.k() * L1Tile::K};
        const MatrixCoord bo{block.k() * L1Tile::K,
                             block.n() * L1Tile::N};
        const MatrixCoord co{block.m() * L1Tile::M,
                             block.n() * L1Tile::N};
        block_mmad(gm_a[layout_a.GetOffset(ao)], layout_a,
                   gm_b[layout_b.GetOffset(bo)], layout_b,
                   gm_c[layout_c.GetOffset(co)], layout_c, actual);
    }
    if constexpr (BlockMmad::DispatchPolicy::ASYNC)
        block_mmad.SynchronizeBlock();
}

__aicore__ inline void RunBf16Slice(
    __gm__ bfloat16_t *a, __gm__ bfloat16_t *b,
    __gm__ bfloat16_t *c, uint32_t m, uint32_t n, uint32_t k,
    uint32_t start_core, bool weight_b_row_major = false)
{
    if (weight_b_row_major) {
        RunBf16SliceLayout<Catlass::layout::RowMajor>(
            a, b, c, m, n, k, start_core);
    } else {
        RunBf16SliceLayout<Catlass::layout::ColumnMajor>(
            a, b, c, m, n, k, start_core);
    }
}

// True cross-expert grouped GEMM for sparse MoE waves.  Unlike
// RunBf16Slice, the CATLASS kernel keeps one BlockMmad instance across the
// complete local expert list and synchronizes it once.  This is essential
// when local_experts exceeds the Cube count: synchronizing once per tiny
// expert/slice otherwise dominates the useful matrix work.
template <class LayoutB>
__aicore__ inline void RunGroupedBf16AllReadyLayout(
    __gm__ bfloat16_t *a, __gm__ bfloat16_t *b,
    __gm__ bfloat16_t *c, __gm__ int64_t *group_list,
    uint32_t rows, uint32_t experts, uint32_t n, uint32_t k)
{
    using namespace Catlass;
    using LayoutA = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    using ArchTag = Arch::AtlasA2;
    constexpr bool kUnitFlag = true;
    constexpr bool kShuffleK = true;
    using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsync<
        1, 2, 2, 2, 1, kUnitFlag, kShuffleK>;
    using L1Tile = GemmShape<128, 256, 256>;
    using L0Tile = GemmShape<128, 256, 64>;
    using AType = Gemm::GemmType<bfloat16_t, LayoutA>;
    using BType = Gemm::GemmType<bfloat16_t, LayoutB>;
    using CType = Gemm::GemmType<bfloat16_t, LayoutC>;
    using BlockMmad = Gemm::Block::BlockMmad<
        DispatchPolicy, L1Tile, L0Tile, AType, BType, CType>;
    using Scheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
    using Kernel = Gemm::Kernel::GroupedMatmulSliceM<
        BlockMmad, void, Scheduler, int64_t>;

    typename Kernel::Params params{
        GemmCoord{rows, n, k}, experts,
        reinterpret_cast<GM_ADDR>(group_list),
        reinterpret_cast<GM_ADDR>(a), LayoutA{rows, k},
        reinterpret_cast<GM_ADDR>(b), LayoutB{k, n},
        reinterpret_cast<GM_ADDR>(c), LayoutC{rows, n}};
    Kernel kernel;
    kernel.template operator()<AscendC::AIC>(params);
}

__aicore__ inline void RunGroupedBf16AllReady(
    __gm__ bfloat16_t *a, __gm__ bfloat16_t *b,
    __gm__ bfloat16_t *c, __gm__ int64_t *group_list,
    uint32_t rows, uint32_t experts, uint32_t n, uint32_t k,
    bool weight_b_row_major)
{
    if (weight_b_row_major) {
        RunGroupedBf16AllReadyLayout<Catlass::layout::RowMajor>(
            a, b, c, group_list, rows, experts, n, k);
    } else {
        RunGroupedBf16AllReadyLayout<Catlass::layout::ColumnMajor>(
            a, b, c, group_list, rows, experts, n, k);
    }
}

// BF16 SwiGLU for the row-major [gate | up] output of GMM1. AIVs split each
// expert's logical [M, intermediate] activation into independent column
// segments. Exact-byte DataCopyPad keeps arbitrary tail sizes correct; no
// hidden alignment contract leaks into the public API.
__aicore__ inline bool RunBf16SwiGlu(
    __gm__ bfloat16_t *gate_up, __gm__ bfloat16_t *activation,
    __gm__ int64_t *group_list, __gm__ uint8_t *input_ready,
    __gm__ uint8_t *output_ready, uint32_t input_ready_stride,
    uint32_t input_ready_producers, uint32_t output_ready_stride,
    uint32_t output_producer, uint32_t output_producers,
    uint32_t experts, uint32_t intermediate, uint32_t activation_waves,
    uint64_t generation, uint32_t spin_cap)
{
    // Atlas A2 FP32 vector math executes 64 elements per repeat.  Keep the
    // scalar-count high-level SwiGLU call to one repeat; larger tiles require
    // explicit repeat/stride programming (added by the optimized path), not
    // a larger calCount that CANN 8.5 only partially evaluates.
    constexpr uint32_t kTileElements = 1024u;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gate_bf16_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> up_bf16_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> out_bf16_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gate_fp32_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> up_fp32_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> temp_fp32_buf;
    pipe.InitBuffer(gate_bf16_buf, kTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(up_bf16_buf, kTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(out_bf16_buf, kTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(gate_fp32_buf, kTileElements * sizeof(float));
    pipe.InitBuffer(up_fp32_buf, kTileElements * sizeof(float));
    pipe.InitBuffer(temp_fp32_buf, kTileElements * sizeof(float));
    auto gate_bf16 = gate_bf16_buf.Get<bfloat16_t>();
    auto up_bf16 = up_bf16_buf.Get<bfloat16_t>();
    auto out_bf16 = out_bf16_buf.Get<bfloat16_t>();
    auto gate_fp32 = gate_fp32_buf.Get<float>();
    auto up_fp32 = up_fp32_buf.Get<float>();
    auto temp_fp32 = temp_fp32_buf.Get<float>();
    AscendC::GlobalTensor<bfloat16_t> gm_gate_up;
    AscendC::GlobalTensor<bfloat16_t> gm_activation;
    gm_gate_up.SetGlobalBuffer(gate_up);
    gm_activation.SetGlobalBuffer(activation);

    int64_t previous = 0;
    for (uint32_t expert = 0u; expert < experts; ++expert) {
        const int64_t end = group_list[expert];
        if (end < previous) return false;
        const uint64_t rows = static_cast<uint64_t>(end - previous);
        for (uint32_t activation_wave = 0u;
             activation_wave < activation_waves; ++activation_wave) {
            const uint32_t flat = expert * activation_waves + activation_wave;
            __gm__ uint8_t *ready = input_ready +
                static_cast<uint64_t>(flat) * input_ready_stride * 64u;
            if (!WaitAllGenerations(ready, input_ready_producers,
                                    generation, spin_cap))
                return false;
            const uint64_t row_begin = rows * activation_wave /
                                       activation_waves;
            const uint64_t row_end = rows * (activation_wave + 1u) /
                                     activation_waves;
            const uint64_t tiles_per_row =
                (intermediate + kTileElements - 1u) / kTileElements;
            const uint64_t tiles = (row_end - row_begin) * tiles_per_row;
            for (uint64_t tile = output_producer;
                 tile < tiles; tile += output_producers) {
            const uint64_t local_row = tile / tiles_per_row;
            const uint64_t row = row_begin + local_row;
            const uint32_t column = static_cast<uint32_t>(
                tile - local_row * tiles_per_row) * kTileElements;
            const uint32_t available = intermediate - column;
            const uint32_t count = available < kTileElements
                ? available : kTileElements;
            const uint64_t packed_row = static_cast<uint64_t>(previous) + row;
            const uint64_t gate_offset =
                packed_row * (2u * intermediate) + column;
            const uint64_t up_offset = gate_offset + intermediate;
            const uint64_t out_offset = packed_row * intermediate + column;
            AscendC::DataCopyExtParams copy_in(
                1u, count * sizeof(bfloat16_t), 0u, 0u, 0u);
            // DataCopyPadExtParams has no value-initializing default
            // constructor on the CANN 8.x toolchain.  Leaving it
            // uninitialized makes the padding switch/value depend on UB
            // stack contents and causes identical SwiGLU invocations to
            // produce different results.
            AscendC::DataCopyPadExtParams<bfloat16_t> pad(false, 0, 0, 0);
            AscendC::DataCopyPad(gate_bf16, gm_gate_up[gate_offset],
                                 copy_in, pad);
            AscendC::DataCopyPad(up_bf16, gm_gate_up[up_offset],
                                 copy_in, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast(gate_fp32, gate_bf16,
                          AscendC::RoundMode::CAST_NONE, count);
            AscendC::Cast(up_fp32, up_bf16,
                          AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(temp_fp32, gate_fp32, -1.0f, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(temp_fp32, temp_fp32, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(temp_fp32, temp_fp32, 1.0f, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(temp_fp32, gate_fp32, temp_fp32, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(temp_fp32, temp_fp32, up_fp32, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(out_bf16, temp_fp32,
                          AscendC::RoundMode::CAST_RINT, count);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyExtParams copy_out(
                1u, count * sizeof(bfloat16_t), 0u, 0u, 0u);
            AscendC::DataCopyPad(gm_activation[out_offset], out_bf16,
                                 copy_out);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            }
            PublishGeneration(output_ready +
                (static_cast<uint64_t>(flat) * output_ready_stride +
                 output_producer) * 64u, generation);
        }
        previous = end;
    }
    return true;
}

// Geometry of one SwiGLU tile.  A tile spans several whole intermediate rows
// when the row is short, because the eight vector instructions and the six
// pipe barriers that separate them cost the same whether they run over one
// row or over several, and the barriers dominate at row granularity.  Rows
// longer than the staging budget fall back to column tiles.
struct SwiGluTiling {
    uint32_t rows_per_tile;
    uint32_t columns_per_tile;
    uint64_t column_tiles;
    uint64_t tiles;
};

__aicore__ inline SwiGluTiling SwiGluPlanTiles(
    uint64_t rows, uint32_t intermediate, uint32_t budget)
{
    SwiGluTiling out{};
    out.columns_per_tile = intermediate < budget ? intermediate : budget;
    out.rows_per_tile = out.columns_per_tile == 0u
        ? 1u : budget / out.columns_per_tile;
    if (out.rows_per_tile == 0u) out.rows_per_tile = 1u;
    out.column_tiles = out.columns_per_tile == 0u ? 1u :
        (intermediate + out.columns_per_tile - 1u) / out.columns_per_tile;
    out.tiles = ((rows + out.rows_per_tile - 1u) / out.rows_per_tile) *
        out.column_tiles;
    return out;
}

// Stages one SwiGLU tile of the [gate | up] GMM1 output into the given half of
// the double buffer and arms its arrival flag.  Returns the element count,
// which an edge tile may shorten.
__aicore__ inline uint32_t SwiGluStageLoad(
    AscendC::GlobalTensor<bfloat16_t> &gm_gate_up,
    AscendC::LocalTensor<bfloat16_t> &gate_bf16,
    AscendC::LocalTensor<bfloat16_t> &up_bf16,
    const SwiGluTiling &tiling, uint32_t budget, uint64_t tile,
    uint64_t rows, uint32_t intermediate, uint32_t stage)
{
    const uint64_t row_group = tile / tiling.column_tiles;
    const uint32_t column = static_cast<uint32_t>(
        tile - row_group * tiling.column_tiles) * tiling.columns_per_tile;
    const uint64_t first_row = row_group * tiling.rows_per_tile;
    const uint64_t remaining_rows = rows - first_row;
    const uint32_t tile_rows = remaining_rows < tiling.rows_per_tile
        ? static_cast<uint32_t>(remaining_rows) : tiling.rows_per_tile;
    const uint32_t available = intermediate - column;
    const uint32_t tile_columns = available < tiling.columns_per_tile
        ? available : tiling.columns_per_tile;
    const uint32_t block_bytes = tile_columns * sizeof(bfloat16_t);
    const uint64_t gate_offset =
        first_row * (2u * intermediate) + column;
    const uint32_t offset = stage * budget;
    // Consecutive gate blocks are separated by the row's up half, and vice
    // versa, so one strided copy per half packs the tile contiguously.
    AscendC::DataCopyExtParams copy(
        static_cast<uint16_t>(tile_rows), block_bytes,
        (2u * intermediate - tile_columns) * sizeof(bfloat16_t), 0u, 0u);
    AscendC::DataCopyPadExtParams<bfloat16_t> pad(false, 0, 0, 0);
    AscendC::DataCopyPad(gate_bf16[offset], gm_gate_up[gate_offset],
                         copy, pad);
    AscendC::DataCopyPad(up_bf16[offset],
                         gm_gate_up[gate_offset + intermediate], copy, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
        static_cast<event_t>(stage));
    return tile_rows * tile_columns;
}

// Stage-wide variant for the bulk grouped-GMM path.  Grouped GMM1 exposes no
// expert-granular completion edge: all rows are ready together.  Running the
// same elementwise math over the contiguous packed rows therefore needs one
// wave barrier, not E separate producer/consumer barriers.
//
// The tile loop is a two-stage software pipeline: staging the loads one tile
// ahead keeps MTE2 busy while the vector unit reduces the tile that arrived
// before it.
__aicore__ inline void RunBf16SwiGluAllReady(
    __gm__ bfloat16_t *gate_up, __gm__ bfloat16_t *activation,
    uint64_t rows, uint32_t intermediate,
    uint32_t producer, uint32_t producers)
{
    constexpr uint32_t kTileElements = 3072u;
    constexpr uint32_t kStages = 2u;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gate_bf16_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> up_bf16_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> out_bf16_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gate_fp32_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> up_fp32_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> temp_fp32_buf;
    pipe.InitBuffer(gate_bf16_buf,
                    kStages * kTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(up_bf16_buf,
                    kStages * kTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(out_bf16_buf,
                    kStages * kTileElements * sizeof(bfloat16_t));
    // The vector stage is serial, so the float temporaries need no staging.
    pipe.InitBuffer(gate_fp32_buf, kTileElements * sizeof(float));
    pipe.InitBuffer(up_fp32_buf, kTileElements * sizeof(float));
    pipe.InitBuffer(temp_fp32_buf, kTileElements * sizeof(float));
    auto gate_bf16 = gate_bf16_buf.Get<bfloat16_t>();
    auto up_bf16 = up_bf16_buf.Get<bfloat16_t>();
    auto out_bf16 = out_bf16_buf.Get<bfloat16_t>();
    auto gate_fp32 = gate_fp32_buf.Get<float>();
    auto up_fp32 = up_fp32_buf.Get<float>();
    auto temp_fp32 = temp_fp32_buf.Get<float>();
    AscendC::GlobalTensor<bfloat16_t> gm_gate_up;
    AscendC::GlobalTensor<bfloat16_t> gm_activation;
    gm_gate_up.SetGlobalBuffer(gate_up);
    gm_activation.SetGlobalBuffer(activation);

    const SwiGluTiling tiling =
        SwiGluPlanTiles(rows, intermediate, kTileElements);
    if (producer >= tiling.tiles) return;
    const uint64_t my_tiles =
        (tiling.tiles - producer + producers - 1u) / producers;
    uint32_t counts[kStages] = {0u, 0u};
    for (uint64_t index = 0u; index < my_tiles; ++index) {
        const uint32_t stage = static_cast<uint32_t>(index & 1u);
        const event_t event = static_cast<event_t>(stage);
        if (index == 0u)
            counts[stage] = SwiGluStageLoad(
                gm_gate_up, gate_bf16, up_bf16, tiling, kTileElements,
                producer, rows, intermediate, stage);
        if (index + 1u < my_tiles) {
            const uint32_t next_stage =
                static_cast<uint32_t>((index + 1u) & 1u);
            // The tile two slots back released this buffer when its cast
            // finished; anything earlier has long drained.
            if (index >= 1u)
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                    static_cast<event_t>(next_stage));
            counts[next_stage] = SwiGluStageLoad(
                gm_gate_up, gate_bf16, up_bf16, tiling, kTileElements,
                producer + (index + 1u) * producers, rows,
                intermediate, next_stage);
        }
        const uint64_t tile = producer + index * producers;
        const uint64_t row_group = tile / tiling.column_tiles;
        const uint32_t column = static_cast<uint32_t>(
            tile - row_group * tiling.column_tiles) * tiling.columns_per_tile;
        const uint64_t first_row = row_group * tiling.rows_per_tile;
        const uint32_t tile_columns =
            intermediate - column < tiling.columns_per_tile
                ? intermediate - column : tiling.columns_per_tile;
        const uint32_t count = counts[stage];
        const uint32_t offset = stage * kTileElements;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        AscendC::Cast(gate_fp32, gate_bf16[offset],
                      AscendC::RoundMode::CAST_NONE, count);
        AscendC::Cast(up_fp32, up_bf16[offset],
                      AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
        // gate and up are dead once they are in float, so the next load may
        // overwrite this stage's staging buffers from here on.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::Muls(temp_fp32, gate_fp32, -1.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(temp_fp32, temp_fp32, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(temp_fp32, temp_fp32, 1.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(temp_fp32, gate_fp32, temp_fp32, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(temp_fp32, temp_fp32, up_fp32, count);
        AscendC::PipeBarrier<PIPE_V>();
        if (index >= kStages)
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        AscendC::Cast(out_bf16[offset], temp_fp32,
                      AscendC::RoundMode::CAST_RINT, count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        const uint32_t tile_rows = tile_columns == 0u
            ? 0u : count / tile_columns;
        AscendC::DataCopyExtParams store(
            static_cast<uint16_t>(tile_rows),
            tile_columns * sizeof(bfloat16_t), 0u,
            (intermediate - tile_columns) * sizeof(bfloat16_t), 0u);
        AscendC::DataCopyPad(
            gm_activation[first_row * intermediate + column],
            out_bf16[offset], store);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
    }
    // Retire the stores that the loop deliberately left in flight.
    for (uint64_t index = my_tiles > kStages ? my_tiles - kStages : 0u;
         index < my_tiles; ++index) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
            static_cast<event_t>(index & 1u));
    }
    if (my_tiles >= 1u) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
            static_cast<event_t>((my_tiles - 1u) & 1u));
        if (my_tiles >= 2u)
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                static_cast<event_t>(my_tiles & 1u));
    }
}

} // namespace inc::fusion::compute

#endif
