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
 * \file shmem_mega_moe_types.h
 * \brief
 */

#ifndef SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_TYPES_H
#define SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_TYPES_H

// Some AscendC lib/std headers use ASCENDC_HOST_AICORE before defining it.
#ifndef ASCENDC_HOST_AICORE
#if defined(__CCE__)
#define ASCENDC_HOST_AICORE [ host, aicore ]
#else
#define ASCENDC_HOST_AICORE
#endif
#endif

#include "shmem_ascendc_compat.h"
#if __has_include("basic_api/kernel_operator_scalar_intf.h")
#include "basic_api/kernel_operator_scalar_intf.h"
#else
#include "kernel_operator_scalar_intf.h"
#endif
#include "lib/std/tuple.h"
#include "shmem.h"
#include "shmem_mega_moe_tiling.h"
#include "common/shmem_mega_moe_math.h"
#include "shmem_mega_moe_memory_layout.h"

namespace AscendC {
#if !defined(__NPU_ARCH__)
template <typename T>
__aicore__ inline T ReadGmByPassDCache(__gm__ T* addr)
{
    return *addr;
}

template <typename T>
__aicore__ inline void WriteGmByPassDCache(__gm__ T* addr, T value)
{
    *addr = value;
}
#endif
} // namespace AscendC

using AscendC::AIC;
using AscendC::AIV;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::GetBlockIdx;
using AscendC::GetBlockNum;
using AscendC::ReadGmByPassDCache;
using AscendC::WriteGmByPassDCache;

#if defined(__CCE__)
#define SHMEM_MEGA_MOE_HOST_DEVICE __forceinline__[host, aicore]
#else
#define SHMEM_MEGA_MOE_HOST_DEVICE inline
#endif

struct ShmemMegaMoeMatmulBuffers {
    GM_ADDR lhs;
    GM_ADDR rhs;
    GM_ADDR lhsScale;
    GM_ADDR rhsScale;
    __gm__ int32_t* readyFlags;
    __gm__ int32_t* waveReadyFlags;
    GM_ADDR output;
};

struct ShmemMegaMoeSymmetricMemoryLayout {
    GM_ADDR localBase;
    GM_ADDR inboundMask;
    GM_ADDR quantizedTokens;
    GM_ADDR combinedTokens;
    SHMEM_MEGA_MOE_HOST_DEVICE ShmemMegaMoeSymmetricMemoryLayout() = default;
    SHMEM_MEGA_MOE_HOST_DEVICE ShmemMegaMoeSymmetricMemoryLayout(
        GM_ADDR base, const ShmemMegaMoeTilingData* config, uint32_t elemsPerByte = 1)
    {
        localBase = base;
        uint64_t offset = static_cast<uint64_t>(SHMEM_CONTROL_BYTES);
        inboundMask = base + offset;

        uint64_t routeCount = static_cast<uint64_t>(config->tokenCount) * config->expertsPerToken;
        uint64_t compareCount =
            ShmemMegaMoeKernel::AlignUp(routeCount * sizeof(int32_t), static_cast<uint64_t>(ALIGN_256)) /
            sizeof(int32_t);
        uint64_t maskAlignSize = ShmemMegaMoeKernel::AlignUp(compareCount / 8, static_cast<uint64_t>(ALIGN_32));
        uint64_t maskSlotSize = maskAlignSize + static_cast<uint64_t>(ALIGN_32); // mask + 32B count

        offset += ShmemMegaMoeKernel::AlignUp(
            static_cast<uint64_t>(config->localExpertCount) * config->rankCount * maskSlotSize,
            static_cast<uint64_t>(ALIGN_512));

        quantizedTokens = base + offset;
        uint64_t scaleCount =
            ShmemMegaMoeKernel::CeilDiv(static_cast<uint64_t>(config->modelDim), static_cast<uint64_t>(ALIGN_32));
        uint64_t dataBytes = ShmemMegaMoeKernel::AlignUp(
            static_cast<uint64_t>(config->modelDim) / elemsPerByte, static_cast<uint64_t>(ALIGN_256));
        uint64_t scaleBytes = scaleCount * sizeof(int8_t);
        uint64_t tokenBytes = ShmemMegaMoeKernel::AlignUp(dataBytes + scaleBytes, static_cast<uint64_t>(ALIGN_32));
        offset += ShmemMegaMoeKernel::AlignUp(
            static_cast<uint64_t>(config->tokenCount) * tokenBytes, static_cast<uint64_t>(ALIGN_512));
        combinedTokens = base + offset;
    }
};

#undef SHMEM_MEGA_MOE_HOST_DEVICE

struct ShmemMegaMoeKernelParams {
    GM_ADDR input;
    GM_ADDR expertIds;
    GM_ADDR weight1;
    GM_ADDR weight1Scales;
    GM_ADDR weight2;
    GM_ADDR weight2Scales;
    GM_ADDR routingWeights;
    GM_ADDR output;
    GM_ADDR expertTokenCountsOut;
    ShmemMegaMoeWorkspaceLayout workspace;
    ShmemMegaMoeSymmetricMemoryLayout symmetricMemory;
    ShmemMegaMoeTilingData* config;
};

enum class ShmemMegaMoeMatmulStage : int32_t { kFirstProjection, kSecondProjection };

__aicore__ inline void NotifyCube(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void WaitForVector(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void NotifyVector(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
}

__aicore__ inline void WaitForCube(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + value);
}

__aicore__ inline void FinishFirstProjectionSync(int32_t& vectorSyncState)
{
    if (vectorSyncState == 0) {
        return;
    }
    if constexpr (g_coreType == AIC) {
        WaitForVector();
    }
}

__aicore__ inline void FinishSecondProjectionSync(int32_t& vectorSyncState, uint16_t secondProjectionBufferIndex)
{
    if constexpr (g_coreType == AIV) {
        return;
    }
    if (vectorSyncState <= 0) {
        return;
    } else if (vectorSyncState == 1) {
        WaitForVector();
    } else {
        WaitForVector(secondProjectionBufferIndex);
        WaitForVector(1 - secondProjectionBufferIndex);
    }
}

__aicore__ inline void SplitWorkAcrossVectorCores(
    int32_t totalLen, int32_t& coreLen, int32_t& coreOffset, int32_t align = ALIGN_32)
{
    int32_t coreIndex = GetBlockIdx();
    int32_t coreCount = GetBlockNum() * 2;
    int32_t lenPerCore = ShmemMegaMoeKernel::CeilDiv(static_cast<uint32_t>(totalLen), static_cast<uint32_t>(coreCount));
    int32_t lenPerCoreAlign =
        ShmemMegaMoeKernel::AlignUp(static_cast<uint32_t>(lenPerCore), static_cast<uint32_t>(align));
    coreLen = lenPerCoreAlign;
    coreOffset = coreIndex * lenPerCoreAlign;
    if (coreOffset + coreLen >= totalLen) {
        coreLen = totalLen - coreOffset;
    }
    if (coreOffset >= totalLen) {
        coreLen = 0;
    }
}

__aicore__ inline void WaitForRankGeneration(__gm__ int32_t* signal, int32_t generation)
{
    do {
        // The same slot is reused by successive barrier generations. A faster rank may publish
        // the next generation before a slower rank samples the current one, so equality can wait
        // forever after the value has advanced past generation.
        if (ReadGmByPassDCache(signal) >= generation) {
            return;
        }
    } while (true);
}

__aicore__ inline GM_ADDR GetRemoteAddress(GM_ADDR localBase, uint32_t rankId, uint64_t offset)
{
    return reinterpret_cast<GM_ADDR>(aclshmem_ptr(localBase, rankId)) + offset;
}

// Base template: handles single-index case
template <size_t I, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return AscendC::Std::get<I>(AscendC::Std::forward<T>(t));
}

// Recursive template: handles multiple index cases
template <size_t First, size_t Second, size_t... Rest, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return Get<Second, Rest...>(AscendC::Std::get<First>(AscendC::Std::forward<T>(t)));
}

template <AscendC::HardEvent event, int32_t eventId>
__aicore__ inline void SyncFuncStatic()
{
    // static_assert(eventId >= 0 && eventId <= 5, "SyncFuncStatic eventId must be [0, 5]!");
    AscendC::SetFlag<event>(static_cast<event_t>(eventId));
    AscendC::WaitFlag<event>(static_cast<event_t>(eventId));
}
#endif // SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_TYPES_H
