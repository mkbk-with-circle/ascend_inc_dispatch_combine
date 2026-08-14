/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHMEM_EXAMPLES_MEGA_MOE_MAIN_ARCH35_H
#define SHMEM_EXAMPLES_MEGA_MOE_MAIN_ARCH35_H

#include "shmem_mega_moe_tiling.h"
#include "shmem_mega_moe_tiling_key.h"
#include "shmem_ascendc_compat.h"
#include "shmem_mega_moe_types.h"

#include "shmem_mega_moe_kernel.h"

namespace ShmemMegaMoeExample {

template <
    typename XType, typename YType, typename RoutingWeightType, uint8_t DispatchQuantMode, uint8_t DispatchQuantOutType,
    uint8_t CombineQuantOutType>
__global__ __mix__(1, 2) __schedmode__(1) void shmem_mega_moe_arch35_kernel(
    uint64_t fftsAddr, GM_ADDR symmetricMemoryBase, GM_ADDR x, GM_ADDR expertIds, GM_ADDR routingWeights,
    GM_ADDR weight1, GM_ADDR weight2, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR xActiveMask, GM_ADDR scales,
    GM_ADDR yOut, GM_ADDR expertTokenCountsOut, GM_ADDR workspaceGM, ShmemMegaMoeTilingData config)
{
    AscendC::InitSocState();
    AscendC::SetSyncBaseAddr(fftsAddr);

    if constexpr (DispatchQuantMode == DISPATCH_QUANT_MODE_MXFP) {
        ShmemMegaMoeKernel::ShmemMegaMoePipeline<
            XType, YType, RoutingWeightType, DispatchQuantOutType, CombineQuantOutType>
            op;
        op.Init(
            symmetricMemoryBase, x, expertIds, routingWeights, weight1, weight2, xActiveMask, weightScales1,
            weightScales2, scales, yOut, expertTokenCountsOut, workspaceGM, &config);
        op.Process();
    }
}

} // namespace ShmemMegaMoeExample

#endif // SHMEM_EXAMPLES_MEGA_MOE_MAIN_ARCH35_H
