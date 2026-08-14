/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "shmem.h"

extern "C" ACLSHMEM_GLOBAL void signal_rdma_set(uint64_t config, GM_ADDR addr, int pe_id, int n_pes)
{
    util_set_ffts_config(config);
    auto sig_addr = (__gm__ int32_t*)addr;
    int next = (pe_id + 1) % n_pes;
    int32_t my_signal = next + 1;
    int32_t expected_signal = pe_id + 1;

    aclshmemx_signal_op(sig_addr, my_signal, ACLSHMEM_SIGNAL_SET, next);
    aclshmem_signal_wait_until(sig_addr, ACLSHMEM_CMP_EQ, expected_signal);
}

void signal_rdma_set_do(void* stream, uint64_t config, uint8_t* addr, int pe_id, int n_pes)
{
    signal_rdma_set<<<1, nullptr, stream>>>(config, addr, pe_id, n_pes);
}
