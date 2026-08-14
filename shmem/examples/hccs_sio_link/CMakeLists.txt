# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
aclshmem_add_fusion_example(hccs_sio_link main.cpp kernels/hccs_sio_link_kernel.cpp kernels/hccs_sio_link_perf_kernel.cpp)
if(USE_EXAMPLES)
    target_include_directories(hccs_sio_link PRIVATE
        ${PROJECT_SOURCE_DIR}/examples/hccs_sio_link/kernels
        ${PROJECT_SOURCE_DIR}/examples/hccs_sio_link/utils
    )
endif()
