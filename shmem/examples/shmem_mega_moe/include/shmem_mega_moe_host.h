/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHMEM_EXAMPLES_MEGA_MOE_HOST_H
#define SHMEM_EXAMPLES_MEGA_MOE_HOST_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

#ifndef OP_LOGD
#define OP_LOGD(...)
#endif

#include "shmem_mega_moe_tiling.h"
#include "shmem_mega_moe_types.h"
#include "shmem_mega_moe_memory_layout.h"

namespace ShmemMegaMoeExample {

constexpr uint64_t MiB = 1024ULL * 1024ULL;
constexpr uint64_t DefaultLocalMemSize = 1024ULL * MiB;

enum class Mode {
    Arch35E4M3,
    Arch35E5M2,
};

struct RunConfig {
    Mode mode = Mode::Arch35E4M3;
    uint32_t rankCount = 1;
    uint32_t rankId = 0;
    uint32_t tokenCount = 1;
    uint32_t modelDim = 1024;
    uint32_t ffnDim = 1024;
    uint32_t expertsPerToken = 1;
    uint32_t localExpertCount = 1;
    uint32_t maxReceivedTokens = 0;
    uint32_t vectorCoreCount = 20;
    uint32_t cubeCoreCount = 20;
    int loopCount = 1;
    int warmupCount = 0;
};

inline size_t AlignUp(size_t value, size_t alignment)
{
    return alignment == 0 ? value : ((value + alignment - 1) / alignment) * alignment;
}

inline bool ParseMode(const std::string& mode, Mode& out)
{
    if (mode == "arch35_e4m3") {
        out = Mode::Arch35E4M3;
        return true;
    }
    if (mode == "arch35_e5m2") {
        out = Mode::Arch35E5M2;
        return true;
    }
    return false;
}

inline uint32_t DefaultMaxReceivedTokens(const RunConfig& cfg)
{
    if (cfg.maxReceivedTokens != 0) {
        return cfg.maxReceivedTokens;
    }
    const uint64_t defaultCount =
        static_cast<uint64_t>(cfg.tokenCount) * cfg.rankCount * std::min(cfg.expertsPerToken, cfg.localExpertCount);
    return defaultCount <= std::numeric_limits<uint32_t>::max() ? static_cast<uint32_t>(defaultCount) : 0;
}

inline ShmemMegaMoeTilingData BuildArch35Tiling(const RunConfig& cfg)
{
    ShmemMegaMoeTilingData tiling{};
    tiling.localExpertCount = cfg.localExpertCount;
    tiling.tokenCount = cfg.tokenCount;
    tiling.modelDim = cfg.modelDim;
    tiling.ffnDim = cfg.ffnDim;
    tiling.rankCount = cfg.rankCount;
    tiling.cubeBlocksPerRank = std::max(1U, cfg.cubeCoreCount / std::max(1U, cfg.rankCount));
    tiling.maxReceivedTokens = DefaultMaxReceivedTokens(cfg);
    tiling.expertsPerToken = cfg.expertsPerToken;
    tiling.cubeCoreCount = cfg.cubeCoreCount;
    tiling.vectorCoreCount = cfg.vectorCoreCount;
    tiling.combineQuantMode = COMBINE_NO_QUANT;
    return tiling;
}

inline size_t CalcArch35WorkspaceSize(const ShmemMegaMoeTilingData& tiling)
{
    ShmemMegaMoeWorkspaceLayout info(reinterpret_cast<GM_ADDR>(0), &tiling);
    return AlignUp(static_cast<size_t>(info.totalBytes) + 64ULL * MiB, 512);
}

inline size_t CalcArch35SymmetricMemorySize(const ShmemMegaMoeTilingData& tiling, uint32_t elemsPerByte)
{
    constexpr uintptr_t dummyBase = 0x10000000UL;
    ShmemMegaMoeSymmetricMemoryLayout info(reinterpret_cast<GM_ADDR>(dummyBase), &tiling, elemsPerByte);
    const size_t routeCount = static_cast<size_t>(tiling.tokenCount) * tiling.expertsPerToken;
    const size_t combineSendSize = routeCount * static_cast<size_t>(tiling.modelDim) * sizeof(uint16_t);
    const size_t combineSendOffset = reinterpret_cast<uintptr_t>(info.combinedTokens) - dummyBase;
    return AlignUp(combineSendOffset + combineSendSize + 64ULL * MiB, 512);
}
} // namespace ShmemMegaMoeExample

#endif // SHMEM_EXAMPLES_MEGA_MOE_HOST_H
