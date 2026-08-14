/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file shmem_mega_moe_memory_layout.h
 * \brief
 */

#ifndef SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_MEMORY_LAYOUT_H
#define SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_MEMORY_LAYOUT_H

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__) || defined(SHMEM_MEGA_MOE_EXAMPLE_ARCH35)
#include "shmem_ascendc_compat.h"
#include "shmem_mega_moe_tiling.h"
#include "common/shmem_mega_moe_math.h"
#define SHMEM_MEGA_MOE_LAYOUT_HOST_DEVICE __forceinline__[host, aicore]
#else
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif
#define SHMEM_MEGA_MOE_LAYOUT_HOST_DEVICE
#endif

constexpr uint64_t SHAPE_ROW_INDEX = 0UL;
constexpr uint64_t SHAPE_COLUMN_INDEX = 1UL;
constexpr uint64_t SHAPE_REDUCTION_INDEX = 2UL;
constexpr uint64_t TOKEN_SCALE_SIZE = 8U * 1024U;
constexpr uint64_t FIRST_INPUT_OFFSET_INDEX = 0UL;
constexpr uint64_t FIRST_WEIGHT_OFFSET_INDEX = 1UL;
constexpr uint64_t FIRST_INPUT_SCALE_OFFSET_INDEX = 2UL;
constexpr uint64_t FIRST_WEIGHT_SCALE_OFFSET_INDEX = 3UL;
constexpr uint64_t ACTIVATION_OUTPUT_OFFSET_INDEX = 5UL;
constexpr uint64_t ACTIVATION_SCALE_OFFSET_INDEX = 6UL;
constexpr uint64_t EXPERT_FLAG_OFFSET_INDEX = 7UL;
constexpr uint64_t SECOND_WEIGHT_OFFSET_INDEX = 8UL;
constexpr uint64_t SECOND_WEIGHT_SCALE_OFFSET_INDEX = 9UL;
constexpr uint64_t SECOND_OUTPUT_OFFSET_INDEX = 10UL;
constexpr uint64_t TOKEN_PREFIX_OFFSET_INDEX = 11UL;
constexpr uint64_t INT32_PER_256B = 8U;
constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr uint32_t SWIGLU_N_HALF = 2U;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_SCALE_GROUP_NUM = 32;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int64_t SHMEM_CONTROL_BYTES = 1024 * 60LL;
constexpr int64_t ALIGN_32 = 32LL;
constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_256 = 256LL;
constexpr int64_t ALIGN_512 = 512LL;
constexpr int32_t INT_CACHELINE = 16;
// wave-grain dispatch flag: must match ShmemMegaMoeKernel::L1_TILE_M
constexpr int64_t DISPATCH_WAVE_TILE_M = 256LL;
// second projection tile size for row group calculation
constexpr int64_t L1_TILE_M_128 = 128LL;
constexpr uint32_t COMBINE_INPUT_LIST_COUNT = 3U;
constexpr uint32_t DOUBLE_BUFFER = 2U;
constexpr uint32_t RANK_ID = 0U;
constexpr uint32_t TOKEN_ID = 1U;
constexpr uint32_t ROUTE_SLOT_INDEX = 2U;
constexpr uint32_t SYNC_EVENT_ID1 = 1;
constexpr uint32_t SYNC_EVENT_ID2 = 2;
constexpr uint32_t SYNC_EVENT_ID3 = 3;
constexpr uint32_t SYNC_EVENT_ID4 = 4;
constexpr uint32_t SYNC_EVENT_ID5 = 5;
constexpr int64_t SIZE_INT_8 = 1U;
constexpr int64_t SIZE_INT_32 = 4U;
constexpr int64_t SIZE_BF_16 = 2U;
constexpr int64_t E5M2_QUANT = 3U;
constexpr int64_t E4M3_QUANT = 4U;
constexpr int64_t E2M1_QUANT = 5U;
constexpr int64_t HALF_TO_FP32 = 2U;
constexpr int64_t OVERFLOW_MODE_CTRL = 60U;
// Combine quantization modes
constexpr uint8_t COMBINE_NO_QUANT = 0;
constexpr uint8_t MXFP8_E5M2_COMM_QUANT = 3;
constexpr uint8_t MXFP8_E4M3_COMM_QUANT = 4;
// Combine buffer constants
constexpr uint32_t ROUTING_RECORD_WIDTH = 8U;

struct ShmemMegaMoeWorkspaceLayout {
    GM_ADDR receivedTokenData;
    GM_ADDR receivedTokenScales;
    GM_ADDR activatedTokenData;
    GM_ADDR activatedTokenScales;
    GM_ADDR receivedTokenCounts;
    GM_ADDR routingMetadata;
    GM_ADDR activationReadyFlags;
    GM_ADDR inputReadyFlags;
    GM_ADDR matmulOutput;
    GM_ADDR rowGroupReadyFlags;

    int64_t totalBytes;
    SHMEM_MEGA_MOE_LAYOUT_HOST_DEVICE ShmemMegaMoeWorkspaceLayout() = default;
    SHMEM_MEGA_MOE_LAYOUT_HOST_DEVICE ShmemMegaMoeWorkspaceLayout(GM_ADDR base, const ShmemMegaMoeTilingData* config)
    {
        totalBytes = 0;
        receivedTokenData = base;

        totalBytes += ShmemMegaMoeKernel::AlignUp(SIZE_INT_8 * config->maxReceivedTokens * config->modelDim, ALIGN_512);
        receivedTokenScales = base + totalBytes;

        totalBytes += ShmemMegaMoeKernel::AlignUp(
            SIZE_INT_8 * config->maxReceivedTokens * config->modelDim / MXFP_SCALE_GROUP_NUM, ALIGN_512);

        activatedTokenData = base + totalBytes;
        totalBytes += ShmemMegaMoeKernel::AlignUp(
            SIZE_INT_8 * config->maxReceivedTokens * config->ffnDim / SWIGLU_N_HALF, ALIGN_512);

        activatedTokenScales = base + totalBytes;
        totalBytes += ShmemMegaMoeKernel::AlignUp(
            SIZE_INT_8 * config->maxReceivedTokens * config->ffnDim / SWIGLU_N_HALF / MXFP_SCALE_GROUP_NUM, ALIGN_512);

        receivedTokenCounts = base + totalBytes;
        totalBytes +=
            ShmemMegaMoeKernel::AlignUp(config->localExpertCount * ALIGN_32 * config->cubeCoreCount, ALIGN_512);

        routingMetadata = base + totalBytes;
        totalBytes += ShmemMegaMoeKernel::AlignUp(config->maxReceivedTokens * ALIGN_32, ALIGN_512);

        activationReadyFlags = base + totalBytes;
        totalBytes += SIZE_INT_32 * config->localExpertCount * INT_CACHELINE;

        inputReadyFlags = base + totalBytes;
        // wave-grain dispatch flag: per expert allocate one slot per wave (L1_TILE_M=256 rows),
        // aligned up to INT_CACHELINE so each expert's slot block stays cache-line clean.
        int64_t maxWavesPerExpert =
            ShmemMegaMoeKernel::CeilDiv(static_cast<int64_t>(config->maxReceivedTokens), DISPATCH_WAVE_TILE_M);
        int64_t dispatchFlagSlotsPerExpert =
            ShmemMegaMoeKernel::AlignUp(maxWavesPerExpert, static_cast<int64_t>(INT_CACHELINE));
        totalBytes += SIZE_INT_32 * config->localExpertCount * dispatchFlagSlotsPerExpert;

        // Quantization-only workspace buffers
        matmulOutput = nullptr;
        rowGroupReadyFlags = nullptr;
        if (config->combineQuantMode != COMBINE_NO_QUANT) {
            // second projection output buffer (bf16)
            matmulOutput = base + totalBytes;
            totalBytes += ShmemMegaMoeKernel::AlignUp(
                static_cast<int64_t>(SIZE_BF_16 * config->maxReceivedTokens * config->modelDim), ALIGN_512);

            // Row group completion counters
            rowGroupReadyFlags = base + totalBytes;
            int64_t maxTokensPerExpert = static_cast<int64_t>(config->tokenCount) * config->rankCount;
            int64_t maxRowGroupsPerExpert = ShmemMegaMoeKernel::CeilDiv(maxTokensPerExpert, DISPATCH_WAVE_TILE_M);
            int64_t perCoreRowGroupsStride =
                ShmemMegaMoeKernel::AlignUp(static_cast<int64_t>(config->vectorCoreCount) * INT_CACHELINE, ALIGN_128);
            int64_t maxRowGroupsStride =
                ShmemMegaMoeKernel::AlignUp(maxRowGroupsPerExpert * perCoreRowGroupsStride, ALIGN_128);
            totalBytes += SIZE_INT_32 * config->localExpertCount * maxRowGroupsStride;
        }
    }
};
#undef SHMEM_MEGA_MOE_LAYOUT_HOST_DEVICE
#endif // SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_MEMORY_LAYOUT_H
