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
 * \file shmem_mega_moe_tiling.h
 * \brief
 */

#ifndef SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_TILING_H
#define SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_TILING_H

#include <algorithm>
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

template <typename T, typename TilingPtr>
#if defined(__NPU_TILING__) || defined(__CCE_AICORE__) || defined(__AICORE__)
__aicore__ inline void InitShmemMegaMoeTilingData(TilingPtr tiling, T* config)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)config;
    for (uint32_t i = 0; i < sizeof(T) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
}
#else
inline void InitShmemMegaMoeTilingData(TilingPtr tiling, T* config)
{
    const auto* src = reinterpret_cast<const uint8_t*>(tiling);
    auto* dst = reinterpret_cast<uint8_t*>(config);
    std::copy_n(src, sizeof(T), dst);
}
#endif

struct ShmemMegaMoeTilingData {
    uint32_t localExpertCount;
    uint32_t tokenCount;
    uint32_t modelDim;
    uint32_t ffnDim;
    uint32_t rankCount;
    uint32_t cubeBlocksPerRank;
    uint32_t maxReceivedTokens;
    uint32_t expertsPerToken;
    uint32_t cubeCoreCount;
    uint32_t vectorCoreCount;
    int64_t combineQuantMode;
};
#endif // SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_TILING_H
