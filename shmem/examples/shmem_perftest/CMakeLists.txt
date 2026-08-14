# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (USE_EXAMPLES)
    add_subdirectory(ascendc_perftest)
    add_subdirectory(mte_perftest)
    add_subdirectory(udma_perftest)
    if (ACLSHMEM_RDMA_SUPPORT)
        add_subdirectory(rdma_perftest)
    endif()
    if (ACLSHMEM_SIMT_SUPPORT)
        add_subdirectory(simt_rma_perftest)
    endif()
endif()
