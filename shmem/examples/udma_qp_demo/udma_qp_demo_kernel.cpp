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

constexpr uint32_t UDMA_WQE_SCRATCH_BYTES = ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE;

enum class DemoOperation : int32_t {
    PUT = 0,
    GET = 1,
    PUT_SIGNAL = 2,
};

/*
 * 参数说明：
 *   symmetric    本 PE 的 SHMEM 对称地址，Put 时作为远端目标，Get 时作为远端数据源。
 *   local_buffer 本 PE 的普通 Device 地址，Put 时作为本地数据源，Get 时作为本地目标。
 *   elements     uint8_t 元素总数，也就是总字节数。
 *   peer         直连 UDMA 的远端 PE，本样例固定为环上的下一个 PE。
 *   operation    DemoOperation 的整数编码，决定提交 Put、Get 或 PutSignal WQE。
 *
 * Host 启动的 block 数必须等于 QP 数，使每个 QP 只有一个 AIV 提交者。
 */
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void udma_qp_demo_kernel(
    GM_ADDR symmetric, GM_ADDR local_buffer, GM_ADDR signal_words, uint64_t elements, int32_t peer, int32_t operation)
{
    const uint32_t qp_idx = AscendC::GetBlockIdx();
    const uint32_t qp_num = AscendC::GetBlockNum();

    /*
     * Host 已保证元素总数能被 QP 数整除，因此每个 QP 处理等长的连续分片。
     * slice_count 表示单个 QP 的元素数，slice_offset 表示该分片在完整缓冲区中的起始位置。
     * 所有 QP 的区间首尾相接，正好覆盖 [0, elements)，不需要额外的分片分支。
     */
    const uint64_t slice_count = elements / qp_num;
    const uint64_t slice_offset = qp_idx * slice_count;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> scratch;
    pipe.InitBuffer(scratch, UDMA_WQE_SCRATCH_BYTES);
    __ubuf__ uint8_t* wqe_scratch = reinterpret_cast<__ubuf__ uint8_t*>(scratch.Get<uint8_t>().GetPhyAddr());

    if (operation == static_cast<int32_t>(DemoOperation::PUT)) {
        aclshmemx_udma_qp_put_nbi<uint8_t>(
            symmetric + slice_offset, local_buffer + slice_offset, wqe_scratch, static_cast<uint32_t>(slice_count),
            peer, qp_idx, 0);
    } else if (operation == static_cast<int32_t>(DemoOperation::GET)) {
        aclshmemx_udma_qp_get_nbi<uint8_t>(
            local_buffer + slice_offset, symmetric + slice_offset, wqe_scratch, static_cast<uint32_t>(slice_count),
            peer, qp_idx, 0);
    } else {
        const uint64_t signal = (static_cast<uint64_t>(aclshmem_my_pe() + 1) << 32) | (qp_idx + 1);
        aclshmemx_udma_qp_put_signal_nbi<uint8_t>(
            symmetric + slice_offset, local_buffer + slice_offset, static_cast<uint32_t>(slice_count),
            reinterpret_cast<__gm__ uint64_t*>(signal_words) + qp_idx, signal, peer, qp_idx, wqe_scratch, 0);
    }
    aclshmemx_udma_qp_quiet(peer, qp_idx);
}

void launch_udma_qp_demo(
    uint32_t block_dim, void* stream, uint8_t* symmetric, uint8_t* local_buffer, uint64_t* signal_words,
    uint64_t elements, int32_t peer, int32_t operation)
{
    udma_qp_demo_kernel<<<block_dim, nullptr, stream>>>(
        symmetric, local_buffer, reinterpret_cast<uint8_t*>(signal_words), elements, peer, operation);
}
