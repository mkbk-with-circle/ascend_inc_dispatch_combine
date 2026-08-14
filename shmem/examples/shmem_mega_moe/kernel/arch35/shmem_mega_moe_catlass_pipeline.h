/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_CATLASS_PIPELINE_H
#define SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_CATLASS_PIPELINE_H

#include "shmem_ascendc_compat.h"

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "shmem_mega_moe_swiglu_epilogue.h"
#include "shmem_mega_moe_types.h"

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "shmem_mega_moe_combine_pipeline.h"

namespace ShmemMegaMoeKernel {

using ShmemMegaMoeCatlassArch = Catlass::Arch::Ascend950;
using ShmemMegaMoeScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

constexpr uint32_t ALIGN32 = 32U;
constexpr uint32_t L1_TILE_M_256 = 256U;
constexpr uint32_t L1_TILE_M_128 = 128U;
constexpr uint32_t L1_TILE_N = 256U;
constexpr uint32_t L1_TILE_K = 256U;
constexpr uint32_t L0_TILE_K = 128U;
constexpr uint32_t SCALE_K_L1_RATE = 2U;
constexpr uint32_t SWIGLU_N_HALF = 2U;
constexpr uint32_t MAX_SINGLE_MN_256_256 = L1_TILE_M_256 * L1_TILE_N;
constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_256 = (MAX_SINGLE_MN_256_256 + ALIGN32 - 1U) / ALIGN32 * ALIGN32;
constexpr uint32_t MAX_SINGLE_MN_128_256 = L1_TILE_M_128 * L1_TILE_N;
constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_128 = (MAX_SINGLE_MN_128_256 + ALIGN32 - 1U) / ALIGN32 * ALIGN32;
constexpr uint32_t ROUTING_METADATA_UB_OFFSET = 200U * 1024U;

template <
    typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
    uint32_t TileM, bool OutputToUb>
struct ShmemMegaMoeMatmulTraits {
    using LayoutTagA = Catlass::layout::RowMajor;
    using LayoutTagB = Catlass::layout::ColumnMajor;
    using LayoutTagC = Catlass::layout::RowMajor;
    using LayoutMxScaleA = decltype(tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagA, false>(1, 1));
    using LayoutMxScaleB = decltype(tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagB, true>(1, 1));
    using L1TileShape = tla::Shape<tla::Int<TileM>, tla::Int<L1_TILE_N>, tla::Int<L1_TILE_K>>;
    using L0TileShape = tla::Shape<tla::Int<TileM>, tla::Int<L1_TILE_N>, tla::Int<L0_TILE_K>>;
    using DispatchPolicy = Catlass::Gemm::MmadMx<ShmemMegaMoeCatlassArch, true, SCALE_K_L1_RATE>;
    using TileCopyToGm = Catlass::Gemm::Tile::PackedMxTileCopyTla<
        ShmemMegaMoeCatlassArch, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMxScaleA, LayoutMxScaleA,
        ElementMxScaleB, LayoutMxScaleB, ElementC, LayoutTagC, void>;
    using TileCopyToUb = Catlass::Gemm::Tile::PackedMxTileCopyTlaToUB<
        ShmemMegaMoeCatlassArch, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMxScaleA, LayoutMxScaleA,
        ElementMxScaleB, LayoutMxScaleB, ElementC, LayoutTagC, void>;
    using TileCopy = std::conditional_t<OutputToUb, TileCopyToUb, TileCopyToGm>;
    using Type = Catlass::Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
};

__aicore__ inline int64_t CalculateRowGroupStride(uint32_t tokenCount, uint32_t rankCount, uint32_t vectorCoreCount)
{
    int64_t maxTokensPerExpert = static_cast<int64_t>(tokenCount) * rankCount;
    int64_t maxRowGroupsPerExpert =
        ShmemMegaMoeKernel::CeilDiv(maxTokensPerExpert, static_cast<int64_t>(L1_TILE_M_256));
    int64_t perCoreRowGroupsStride =
        ShmemMegaMoeKernel::AlignUp(static_cast<int64_t>(vectorCoreCount) * INT_CACHELINE, ALIGN_128);
    return ShmemMegaMoeKernel::AlignUp(maxRowGroupsPerExpert * perCoreRowGroupsStride, ALIGN_128);
}

__aicore__ inline void ResolveCoreGroup(
    uint32_t coreId, uint32_t numGroups, uint32_t totalCores, uint32_t& myGroup, uint32_t& myIdxInGrp,
    uint32_t& myGrpSize)
{
    uint32_t baseSize = totalCores / numGroups;
    uint32_t remainder = totalCores % numGroups;
    uint32_t boundary = remainder * (baseSize + 1U);
    if (coreId < boundary) {
        myGroup = coreId / (baseSize + 1U);
        myIdxInGrp = coreId % (baseSize + 1U);
        myGrpSize = baseSize + 1U;
    } else {
        uint32_t adjusted = coreId - boundary;
        myGroup = remainder + adjusted / baseSize;
        myIdxInGrp = adjusted % baseSize;
        myGrpSize = baseSize;
    }
}

__aicore__ inline void ResolveGroupCoreRange(
    uint32_t expertIndex, uint32_t numGroups, uint32_t totalCores, uint32_t& firstGroupCore, uint32_t& groupCoreCount)
{
    uint32_t baseSize = totalCores / numGroups;
    uint32_t remainder = totalCores % numGroups;
    if (expertIndex < remainder) {
        groupCoreCount = baseSize + 1U;
        firstGroupCore = expertIndex * (baseSize + 1U);
    } else {
        groupCoreCount = baseSize;
        firstGroupCore = remainder * (baseSize + 1U) + (expertIndex - remainder) * baseSize;
    }
}

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void RunFirstProjection(
    ShmemMegaMoeSwiGluEpilogue<ElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true>& epilogueOp,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const ShmemMegaMoeMatmulBuffers& matmulBuffers, uint32_t& schedulerStartBlock, int32_t& vectorSyncState)
{
    if constexpr (g_coreType == AscendC::AIV) {
        if (GetSubBlockIdx() != 0) {
            return;
        }
    }

    uint32_t m = Get<SHAPE_ROW_INDEX>(problemShape);
    uint32_t n = Get<SHAPE_COLUMN_INDEX>(problemShape);
    uint32_t k = Get<SHAPE_REDUCTION_INDEX>(problemShape);
    uint32_t activationWidth = n / SWIGLU_N_HALF;
    epilogueOp.ConfigureTile({m, activationWidth, k, 0});

    uint32_t cubeBlockCount = GetBlockNum();
    uint32_t cubeBlockIndex = GetBlockIdx() / GetTaskRation();
    ShmemMegaMoeScheduler scheduler(
        Catlass::GemmCoord{m, activationWidth, k}, Catlass::MatrixCoord{L1_TILE_M_256, L1_TILE_N});
    uint32_t scheduledTileCount = scheduler.GetCoreLoops();
    uint32_t firstTileIndex =
        (cubeBlockIndex < schedulerStartBlock ? cubeBlockIndex + cubeBlockCount : cubeBlockIndex) - schedulerStartBlock;

    if constexpr (g_coreType == AscendC::AIC) {
        using MatmulTraits = ShmemMegaMoeMatmulTraits<
            ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, L1_TILE_M_256, true>;
        using BlockMmad = typename MatmulTraits::Type;

        uint32_t scaleK = ::CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(k);
        auto layoutA = tla::MakeLayout<ElementA, typename MatmulTraits::LayoutTagA>(m, k);
        auto layoutB = tla::MakeLayout<ElementB, typename MatmulTraits::LayoutTagB>(k, n);
        auto layoutScaleA =
            tla::MakeMxScaleLayout<ElementMxScaleA, typename MatmulTraits::LayoutTagA, false>(m, scaleK);
        auto layoutScaleB = tla::MakeMxScaleLayout<ElementMxScaleB, typename MatmulTraits::LayoutTagB, true>(scaleK, n);
        auto layoutC = tla::MakeLayout<ElementC, typename MatmulTraits::LayoutTagC>(L1_TILE_M_256, L1_TILE_N);

        GlobalTensor<ElementA> inputGlobal;
        GlobalTensor<ElementB> weightGlobal;
        GlobalTensor<ElementMxScaleA> inputScaleGlobal;
        GlobalTensor<ElementMxScaleB> weightScaleGlobal;
        inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(matmulBuffers.lhs));
        weightGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(matmulBuffers.rhs));
        inputScaleGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleA*>(matmulBuffers.lhsScale));
        weightScaleGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleB*>(matmulBuffers.rhsScale));

        auto inputTensor = tla::MakeTensor(inputGlobal, layoutA, Catlass::Arch::PositionGM{});
        auto weightTensor = tla::MakeTensor(weightGlobal, layoutB, Catlass::Arch::PositionGM{});
        auto inputScaleTensor = tla::MakeTensor(inputScaleGlobal, layoutScaleA, Catlass::Arch::PositionGM{});
        auto weightScaleTensor = tla::MakeTensor(weightScaleGlobal, layoutScaleB, Catlass::Arch::PositionGM{});

        LocalTensor<ElementC> outputTileBuffer0(TPosition::VECIN, 0, MAX_SINGLE_MN_ALIGN32_NUM_256);
        LocalTensor<ElementC> outputTileBuffer1(
            TPosition::VECIN, MAX_SINGLE_MN_ALIGN32_NUM_256 * sizeof(ElementC), MAX_SINGLE_MN_ALIGN32_NUM_256);
        auto outputTileTensor0 = tla::MakeTensor(outputTileBuffer0, layoutC, Catlass::Arch::PositionUB{});
        auto outputTileTensor1 = tla::MakeTensor(outputTileBuffer1, layoutC, Catlass::Arch::PositionUB{});

        Catlass::Arch::Resource<ShmemMegaMoeCatlassArch> resource;
        BlockMmad blockMmad(resource);
        uint32_t lastReadyWave = static_cast<uint32_t>(-1);

        for (uint32_t tileIndex = firstTileIndex; tileIndex < scheduledTileCount; tileIndex += cubeBlockCount) {
            auto blockCoord = scheduler.GetBlockCoord(tileIndex);
            auto actualShape = scheduler.GetActualBlockShape(blockCoord);
            uint32_t rowOffset = blockCoord.m() * L1_TILE_M_256;
            uint32_t columnOffset = blockCoord.n() * L1_TILE_N;
            uint32_t waveIndex = blockCoord.m();
            if (waveIndex != lastReadyWave) {
                uint32_t targetValue = min(L1_TILE_M_256, m - rowOffset);
                __gm__ int32_t* readinessFlag = matmulBuffers.waveReadyFlags + waveIndex;
                while (targetValue != AscendC::ReadGmByPassDCache(readinessFlag)) {
                    int64_t startCycle = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - startCycle < 100) {
                    }
                }
                lastReadyWave = waveIndex;
            }
            if (vectorSyncState) {
                WaitForVector();
            }

            auto tensorBlockA = tla::GetTile(
                inputTensor, tla::MakeCoord(rowOffset, 0), tla::MakeShape(actualShape.m(), actualShape.k()));
            auto tensorBlockScaleA = tla::GetTile(
                inputScaleTensor, tla::MakeCoord(rowOffset, 0),
                tla::MakeShape(actualShape.m(), ::CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(actualShape.k())));
            auto tensorBlockUbFirst =
                tla::GetTile(outputTileTensor0, tla::MakeCoord(0, 0), tla::MakeShape(actualShape.m(), actualShape.n()));
            auto tensorBlockUbSecond =
                tla::GetTile(outputTileTensor1, tla::MakeCoord(0, 0), tla::MakeShape(actualShape.m(), actualShape.n()));

            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                auto tensorBlockB = tla::GetTile(
                    weightTensor, tla::MakeCoord(0, columnOffset + weightBlock * activationWidth),
                    tla::MakeShape(actualShape.k(), actualShape.n()));
                auto tensorBlockScaleB = tla::GetTile(
                    weightScaleTensor, tla::MakeCoord(0, columnOffset + weightBlock * activationWidth),
                    tla::MakeShape(::CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(actualShape.k()), actualShape.n()));
                Catlass::GemmCoord singleShape{actualShape.m(), actualShape.n(), actualShape.k()};
                if (weightBlock == 0) {
                    blockMmad(
                        tensorBlockA, tensorBlockB, tensorBlockUbFirst, singleShape, tensorBlockScaleA,
                        tensorBlockScaleB);
                } else {
                    blockMmad(
                        tensorBlockA, tensorBlockB, tensorBlockUbSecond, singleShape, tensorBlockScaleA,
                        tensorBlockScaleB);
                }
            }
            NotifyVector();
            vectorSyncState = 1;
        }
    } else {
        for (uint32_t tileIndex = firstTileIndex; tileIndex < scheduledTileCount; tileIndex += cubeBlockCount) {
            auto blockCoord = scheduler.GetBlockCoord(tileIndex);
            auto actualShape = scheduler.GetActualBlockShape(blockCoord);
            uint32_t rowOffset = blockCoord.m() * L1_TILE_M_256;
            uint32_t columnOffset = blockCoord.n() * L1_TILE_N;
            Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{actualShape.m(), actualShape.n(), 0, 0};
            Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                rowOffset * activationWidth + columnOffset,
                rowOffset * CeilDiv(activationWidth, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                    CeilDiv(columnOffset, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                0,
                0,
                0,
                0};
            WaitForCube();
            AscendC::SetCtrlSpr<60, 60>(0);
            epilogueOp(epilogueShape, epilogueOffset);
            NotifyCube();
            vectorSyncState = 1;
        }
    }
    schedulerStartBlock = (schedulerStartBlock + scheduledTileCount) % cubeBlockCount;
}

template <
    uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
    typename ElementMxScaleB>
__aicore__ inline void RunSecondProjection(
    const ShmemMegaMoeKernelParams& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const ShmemMegaMoeMatmulBuffers& matmulBuffers, uint32_t& schedulerStartBlock, int32_t& completedOutputTiles,
    uint32_t processedRows, uint16_t& bufferIndex, uint32_t expertIndex, int64_t expertCounterStride)
{
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        if (GetSubBlockIdx() != 0) {
            return;
        }
    }

    uint32_t m = Get<SHAPE_ROW_INDEX>(problemShape);
    uint32_t n = Get<SHAPE_REDUCTION_INDEX>(problemShape);
    uint32_t k = Get<SHAPE_COLUMN_INDEX>(problemShape) / SWIGLU_N_HALF;
    uint32_t cubeBlockCount = GetBlockNum();
    uint32_t cubeBlockIndex = GetBlockIdx() / GetTaskRation();
    constexpr bool outputInUb = CombineQuantMode == COMBINE_NO_QUANT;
    constexpr uint32_t tileRowCount = outputInUb ? L1_TILE_M_128 : L1_TILE_M_256;

    ShmemMegaMoeScheduler scheduler(Catlass::GemmCoord{m, n, k}, Catlass::MatrixCoord{tileRowCount, L1_TILE_N});
    uint32_t scheduledTileCount = scheduler.GetCoreLoops();
    uint32_t firstTileIndex =
        (cubeBlockIndex < schedulerStartBlock ? cubeBlockIndex + cubeBlockCount : cubeBlockIndex) - schedulerStartBlock;

    if constexpr (g_coreType == AscendC::AIC) {
        using MatmulTraits = ShmemMegaMoeMatmulTraits<
            ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, tileRowCount, outputInUb>;
        using BlockMmad = typename MatmulTraits::Type;

        uint32_t scaleK = ::CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(k);
        auto layoutA = tla::MakeLayout<ElementA, typename MatmulTraits::LayoutTagA>(m, k);
        auto layoutB = tla::MakeLayout<ElementB, typename MatmulTraits::LayoutTagB>(k, n);
        auto layoutScaleA =
            tla::MakeMxScaleLayout<ElementMxScaleA, typename MatmulTraits::LayoutTagA, false>(m, scaleK);
        auto layoutScaleB = tla::MakeMxScaleLayout<ElementMxScaleB, typename MatmulTraits::LayoutTagB, true>(scaleK, n);
        auto layoutC = tla::MakeLayout<ElementC, typename MatmulTraits::LayoutTagC>(
            outputInUb ? tileRowCount : m, outputInUb ? L1_TILE_N : n);

        GlobalTensor<ElementA> inputGlobal;
        GlobalTensor<ElementB> weightGlobal;
        GlobalTensor<ElementMxScaleA> inputScaleGlobal;
        GlobalTensor<ElementMxScaleB> weightScaleGlobal;
        GlobalTensor<ElementC> outputGlobal;
        inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(matmulBuffers.lhs));
        weightGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(matmulBuffers.rhs));
        inputScaleGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleA*>(matmulBuffers.lhsScale));
        weightScaleGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleB*>(matmulBuffers.rhsScale));
        if constexpr (!outputInUb) {
            outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(matmulBuffers.output));
        }

        auto inputTensor = tla::MakeTensor(inputGlobal, layoutA, Catlass::Arch::PositionGM{});
        auto weightTensor = tla::MakeTensor(weightGlobal, layoutB, Catlass::Arch::PositionGM{});
        auto inputScaleTensor = tla::MakeTensor(inputScaleGlobal, layoutScaleA, Catlass::Arch::PositionGM{});
        auto weightScaleTensor = tla::MakeTensor(weightScaleGlobal, layoutScaleB, Catlass::Arch::PositionGM{});
        LocalTensor<ElementC> outputTileBuffer0(TPosition::VECIN, 0, MAX_SINGLE_MN_ALIGN32_NUM_128);
        LocalTensor<ElementC> outputTileBuffer1(
            TPosition::VECIN, MAX_SINGLE_MN_ALIGN32_NUM_128 * sizeof(ElementC), MAX_SINGLE_MN_ALIGN32_NUM_128);
        auto outputTileTensor0 = tla::MakeTensor(outputTileBuffer0, layoutC, Catlass::Arch::PositionUB{});
        auto outputTileTensor1 = tla::MakeTensor(outputTileBuffer1, layoutC, Catlass::Arch::PositionUB{});

        Catlass::Arch::Resource<ShmemMegaMoeCatlassArch> resource;
        BlockMmad blockMmad(resource);
        uint32_t vectorCoreCount = cubeBlockCount * 2U;

        for (uint32_t tileIndex = firstTileIndex; tileIndex < scheduledTileCount; tileIndex += cubeBlockCount) {
            auto blockCoord = scheduler.GetBlockCoord(tileIndex);
            auto actualShape = scheduler.GetActualBlockShape(blockCoord);
            uint32_t rowOffset = blockCoord.m() * tileRowCount;
            uint32_t columnOffset = blockCoord.n() * L1_TILE_N;

            if (tileIndex == firstTileIndex) {
                ShmemMegaMoeScheduler firstProjectionScheduler(
                    Catlass::GemmCoord{m, k, n}, Catlass::MatrixCoord{L1_TILE_M_256, L1_TILE_N});
                uint32_t targetLoops = firstProjectionScheduler.GetCoreLoops();
                __gm__ int32_t* readinessFlag = matmulBuffers.readyFlags;
                while (targetLoops != AscendC::ReadGmByPassDCache(readinessFlag)) {
                    int64_t startCycle = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - startCycle < 100) {
                    }
                }
            }

            auto tensorBlockA = tla::GetTile(
                inputTensor, tla::MakeCoord(rowOffset, 0), tla::MakeShape(actualShape.m(), actualShape.k()));
            auto tensorBlockB = tla::GetTile(
                weightTensor, tla::MakeCoord(0, columnOffset), tla::MakeShape(actualShape.k(), actualShape.n()));
            auto tensorBlockScaleA = tla::GetTile(
                inputScaleTensor, tla::MakeCoord(rowOffset, 0),
                tla::MakeShape(actualShape.m(), ::CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(actualShape.k())));
            auto tensorBlockScaleB = tla::GetTile(
                weightScaleTensor, tla::MakeCoord(0, columnOffset),
                tla::MakeShape(::CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(actualShape.k()), actualShape.n()));
            Catlass::GemmCoord singleShape{actualShape.m(), actualShape.n(), actualShape.k()};

            if constexpr (outputInUb) {
                if (completedOutputTiles >= 2) {
                    WaitForVector(bufferIndex);
                }
                if (bufferIndex == 0) {
                    auto tensorBlockC = tla::GetTile(
                        outputTileTensor0, tla::MakeCoord(0, 0), tla::MakeShape(actualShape.m(), actualShape.n()));
                    blockMmad(
                        tensorBlockA, tensorBlockB, tensorBlockC, singleShape, tensorBlockScaleA, tensorBlockScaleB);
                } else {
                    auto tensorBlockC = tla::GetTile(
                        outputTileTensor1, tla::MakeCoord(0, 0), tla::MakeShape(actualShape.m(), actualShape.n()));
                    blockMmad(
                        tensorBlockA, tensorBlockB, tensorBlockC, singleShape, tensorBlockScaleA, tensorBlockScaleB);
                }
                NotifyVector(bufferIndex);
            } else {
                auto tensorC = tla::MakeTensor(outputGlobal, layoutC, Catlass::Arch::PositionGM{});
                auto tensorBlockC = tla::GetTile(
                    tensorC, tla::MakeCoord(rowOffset, columnOffset), tla::MakeShape(actualShape.m(), actualShape.n()));
                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, singleShape, tensorBlockScaleA, tensorBlockScaleB);
                SetFlag<HardEvent::FIX_S>(0);
                WaitFlag<HardEvent::FIX_S>(0);

                uint32_t rowGroupIndex = blockCoord.m();
                uint32_t rowGroupsThisExpert = ShmemMegaMoeKernel::CeilDiv(m, tileRowCount);
                uint32_t firstGroupCore;
                uint32_t groupCoreCount;
                ResolveGroupCoreRange(
                    rowGroupIndex, rowGroupsThisExpert, vectorCoreCount, firstGroupCore, groupCoreCount);
                int64_t baseOffset = static_cast<int64_t>(expertIndex) * expertCounterStride +
                                     static_cast<int64_t>(rowGroupIndex) * vectorCoreCount * INT_CACHELINE;
                for (uint32_t vectorCoreIndex = firstGroupCore; vectorCoreIndex < firstGroupCore + groupCoreCount;
                     ++vectorCoreIndex) {
                    AscendC::AtomicAdd(
                        reinterpret_cast<__gm__ int32_t*>(params.workspace.rowGroupReadyFlags) + baseOffset +
                            vectorCoreIndex * INT_CACHELINE,
                        int32_t(1));
                }
            }
            if constexpr (outputInUb) {
                ++completedOutputTiles;
                bufferIndex = 1U - bufferIndex;
            }
        }
    } else if constexpr (outputInUb) {
        LocalTensor<ElementC> secondProjectionOutputFirst(TPosition::VECIN, 0, MAX_SINGLE_MN_ALIGN32_NUM_128);
        LocalTensor<ElementC> secondProjectionOutputSecond(
            TPosition::VECIN, MAX_SINGLE_MN_ALIGN32_NUM_128 * sizeof(ElementC), MAX_SINGLE_MN_ALIGN32_NUM_128);

        for (uint32_t tileIndex = firstTileIndex; tileIndex < scheduledTileCount; tileIndex += cubeBlockCount) {
            auto blockCoord = scheduler.GetBlockCoord(tileIndex);
            auto actualShape = scheduler.GetActualBlockShape(blockCoord);
            uint32_t rowOffset = blockCoord.m() * tileRowCount;
            uint32_t columnOffset = blockCoord.n() * L1_TILE_N;
            LocalTensor<ElementC> secondProjectionOutput =
                bufferIndex == 0 ? secondProjectionOutputFirst : secondProjectionOutputSecond;
            WaitForCube(bufferIndex);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::GlobalTensor<int32_t> routingMetadataGlobal;
            int32_t tileRowCount = actualShape.m();
            LocalTensor<int32_t> routingMetadata(
                TPosition::VECCALC, ROUTING_METADATA_UB_OFFSET, tileRowCount * ROUTING_RECORD_WIDTH);
            routingMetadataGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
                params.workspace.routingMetadata +
                (processedRows + rowOffset) * ROUTING_RECORD_WIDTH * sizeof(int32_t)));
            AscendC::DataCopy(routingMetadata, routingMetadataGlobal, tileRowCount * ROUTING_RECORD_WIDTH);
            AscendC::Shape<int64_t, int64_t, int64_t, int64_t> combineShape{
                actualShape.m(), actualShape.n(), actualShape.k(), 0};
            ShmemMegaMoeKernel::CombineTokens<ElementC, decltype(combineShape)>(
                rowOffset, columnOffset, n, routingMetadata, secondProjectionOutput, combineShape, params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            NotifyCube(bufferIndex);
            ++completedOutputTiles;
            bufferIndex = 1U - bufferIndex;
        }
    }
    schedulerStartBlock = (schedulerStartBlock + scheduledTileCount) % cubeBlockCount;
}

} // namespace ShmemMegaMoeKernel

#endif // SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_CATLASS_PIPELINE_H
