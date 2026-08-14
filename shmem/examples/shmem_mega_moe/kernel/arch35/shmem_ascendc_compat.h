/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_ASCENDC_COMPAT_H
#define SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_ASCENDC_COMPAT_H

#if __has_include("basic_api/kernel_basic_intf.h")
#include "basic_api/kernel_basic_intf.h"
#elif __has_include("kernel_basic_intf.h")
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#endif // SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_ASCENDC_COMPAT_H
