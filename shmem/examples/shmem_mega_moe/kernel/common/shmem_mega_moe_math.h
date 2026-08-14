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
 * \file shmem_mega_moe_math.h
 * \brief Host/device integer helpers used by the MegaMoE example.
 */
#ifndef SHMEM_EXAMPLES_MEGA_MOE_KERNEL_COMMON_MATH_H
#define SHMEM_EXAMPLES_MEGA_MOE_KERNEL_COMMON_MATH_H

#include <cstdint>

#if defined(__CCE__)
#define SHMEM_MEGA_MOE_HOST_DEVICE __forceinline__[host, aicore]
#else
#define SHMEM_MEGA_MOE_HOST_DEVICE inline
#endif

namespace ShmemMegaMoeKernel {
template <typename T, typename U>
SHMEM_MEGA_MOE_HOST_DEVICE T CeilDiv(T a, U b)
{
    if (a <= 0 || b <= 0) {
        return 0;
    }
    uint64_t calcA = static_cast<uint64_t>(a);
    uint64_t calcB = static_cast<uint64_t>(b);
    return static_cast<T>((calcA + calcB - 1) / calcB);
}

template <typename T, typename U>
SHMEM_MEGA_MOE_HOST_DEVICE T FloorDiv(T a, U b)
{
    if (a <= 0 || b <= 0) {
        return 0;
    }
    return static_cast<T>(a / b);
}

template <typename T, typename U>
SHMEM_MEGA_MOE_HOST_DEVICE T AlignUp(T a, U b)
{
    return static_cast<T>(CeilDiv(a, b) * b);
}

template <typename T, typename U>
SHMEM_MEGA_MOE_HOST_DEVICE T AlignDown(T a, U b)
{
    if (a <= 0 || b <= 0) {
        return 0;
    }
    return static_cast<T>(a / b * b);
}
} // namespace ShmemMegaMoeKernel

#undef SHMEM_MEGA_MOE_HOST_DEVICE

#endif // SHMEM_EXAMPLES_MEGA_MOE_KERNEL_COMMON_MATH_H
