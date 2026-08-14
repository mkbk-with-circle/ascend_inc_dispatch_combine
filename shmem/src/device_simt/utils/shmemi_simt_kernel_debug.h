/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEMI_SIMT_KERNEL_DEBUG_H
#define SHMEMI_SIMT_KERNEL_DEBUG_H
#include "utils/debug/asc_printf.h"
#include "utils/debug/asc_assert.h"
#include "utils/shmemi_kernel_debug.h"

namespace simt
{
    template<class... Args>
    __simt_callee__ inline void aclshmemi_kernel_abort(__gm__ const char* fmt, Args&&... args)
    {
        printf(fmt, args...);
        __trap();
    }
    
    template<class... Args>
    __simt_callee__ inline void aclshmemi_kernel_printf(__gm__ const char* fmt, Args&&... args)
    {
        printf(fmt, args...);
    }
}


#endif

