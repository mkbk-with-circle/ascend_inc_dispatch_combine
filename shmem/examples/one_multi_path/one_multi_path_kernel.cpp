/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "one_multi_path_kernel.h"

#include "device/shmem_def.h"
#include "kernel_operator.h"

namespace {

// The single AIV block exclusively uses UB [0, 16 KB) as its staging buffer.
constexpr uint32_t UB_SCRATCH_OFFSET = 0U;
constexpr uint32_t UB_SCRATCH_SIZE_BYTES = 16U * 1024U;

__aicore__ inline void copy_gm_to_ub(__ubuf__ uint8_t* dst_ub, __gm__ uint8_t* src_gm, uint32_t size)
{
    AscendC::LocalTensor<uint8_t> ub_tensor;
    AscendC::GlobalTensor<uint8_t> gm_tensor;
    AscendC::DataCopyExtParams params(1, size, 0, 0, 0);
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(dst_ub);
    ub_tensor.address_.dataLen = ALIGN_UP(size, UB_ALIGN_SIZE);
    gm_tensor.SetGlobalBuffer(src_gm);
    AscendC::DataCopyPadExtParams<uint8_t> pad_params;
    AscendC::DataCopyPad(ub_tensor, gm_tensor, params, pad_params);
}

__aicore__ inline void copy_ub_to_gm(__gm__ uint8_t* dst_gm, __ubuf__ uint8_t* src_ub, uint32_t size)
{
    AscendC::LocalTensor<uint8_t> ub_tensor;
    AscendC::GlobalTensor<uint8_t> gm_tensor;
    AscendC::DataCopyExtParams params(1, size, 0, 0, 0);
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(src_ub);
    ub_tensor.address_.dataLen = ALIGN_UP(size, UB_ALIGN_SIZE);
    gm_tensor.SetGlobalBuffer(dst_gm);
    AscendC::DataCopyPad(gm_tensor, ub_tensor, params);
}

__aicore__ inline void copy_range(__gm__ uint8_t* dst_gm, __gm__ uint8_t* src_gm, uint64_t size)
{
    if (size == 0) {
        return;
    }

    __ubuf__ uint8_t* ub = reinterpret_cast<__ubuf__ uint8_t*>(static_cast<uint64_t>(UB_SCRATCH_OFFSET));
    constexpr uint32_t sync_id = 0;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(sync_id);

    uint64_t offset = 0;
    while (offset < size) {
        uint64_t remain = size - offset;
        uint32_t copy_bytes = static_cast<uint32_t>(remain > UB_SCRATCH_SIZE_BYTES ? UB_SCRATCH_SIZE_BYTES : remain);
        copy_gm_to_ub(ub, src_gm + offset, copy_bytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(sync_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(sync_id);
        copy_ub_to_gm(dst_gm + offset, ub, copy_bytes);
        offset += copy_bytes;
        if (offset < size) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(sync_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(sync_id);
        }
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(sync_id);
}

[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void one_multi_path_copy_kernel(
    GM_ADDR dst, GM_ADDR one_path_src, GM_ADDR multi_path_src, uint64_t split_size, uint64_t total_size)
{
    __gm__ uint8_t* dst_gm = reinterpret_cast<__gm__ uint8_t*>(dst);
    __gm__ uint8_t* one_path_src_gm = reinterpret_cast<__gm__ uint8_t*>(one_path_src);
    __gm__ uint8_t* multi_path_src_gm = reinterpret_cast<__gm__ uint8_t*>(multi_path_src);

    copy_range(dst_gm, one_path_src_gm, split_size);
    copy_range(dst_gm + split_size, multi_path_src_gm + split_size, total_size - split_size);
}

} // namespace

void launch_one_multi_path_copy(
    void* stream, uint8_t* dst, uint8_t* one_path_src, uint8_t* multi_path_src, uint64_t split_size,
    uint64_t total_size)
{
    one_multi_path_copy_kernel<<<1, nullptr, stream>>>(dst, one_path_src, multi_path_src, split_size, total_size);
}
