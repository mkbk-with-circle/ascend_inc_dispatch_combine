/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccs_sio_link_perf_kernel.h"
#include "hccs_sio_link_config.h"
#include "kernel_operator.h"
#include "shmem.h"
#include "host_device/shmem_common_types.h"

template <typename T>
__aicore__ inline void perf_local_copy_gm2ub(__ubuf__ T *dst_ub, __gm__ T *src_gm, uint32_t size)
{
    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> gm_tensor;
    AscendC::DataCopyExtParams params(1, size, 0, 0, 0);
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(dst_ub);
    ub_tensor.address_.dataLen = ALIGN_UP(size, UB_ALIGN_SIZE);
    gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src_gm));
    AscendC::DataCopyPadExtParams<T> pad_params;
    AscendC::DataCopyPad(ub_tensor, gm_tensor, params, pad_params);
}

template <typename T>
__aicore__ inline void perf_local_copy_ub2gm(__gm__ T *dst_gm, __ubuf__ T *src_ub, uint32_t size)
{
    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> gm_tensor;
    AscendC::DataCopyExtParams params(1, size, 0, 0, 0);
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(src_ub);
    ub_tensor.address_.dataLen = ALIGN_UP(size, UB_ALIGN_SIZE);
    gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst_gm));
    AscendC::DataCopyPad(gm_tensor, ub_tensor, params);
}

template <typename T>
__aicore__ inline void perf_local_mte_get_hccs(__gm__ T *hccs_src, __gm__ T *local_dst,
                                                  __ubuf__ T *buf, uint32_t ub_size_bytes,
                                                  uint32_t elem_size, uint32_t sync_id)
{
    uint64_t block_size = ub_size_bytes / sizeof(T) * sizeof(T);
    if (block_size == 0) {
        return;
    }
    uint64_t elem_bytes = static_cast<uint64_t>(elem_size) * sizeof(T);
    uint64_t remain = elem_bytes % block_size;
    uint64_t repeat_times = elem_bytes / block_size;
    uint64_t repeat_elem = block_size / sizeof(T);

    for (uint64_t i = 0; i < repeat_times; i++) {
        perf_local_copy_gm2ub(buf, hccs_src + i * repeat_elem, block_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(sync_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(sync_id);
        perf_local_copy_ub2gm(local_dst + i * repeat_elem, buf, block_size);
        if (i != repeat_times - 1 || remain > 0) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(sync_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(sync_id);
        }
    }
    if (remain > 0) {
        perf_local_copy_gm2ub(buf, hccs_src + repeat_times * repeat_elem, remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(sync_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(sync_id);
        perf_local_copy_ub2gm(local_dst + repeat_times * repeat_elem, buf, remain);
    }
}

template <typename T>
[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void mte_get_mixed_perf_kernel(
    GM_ADDR sio_dst_gva, GM_ADDR sio_src_gva, int sio_elements, int peer_pe,
    GM_ADDR hccs_dst_gva, GM_ADDR hccs_remote_gva, int hccs_elements,
    int ub_size_kb, int32_t frame_id, int64_t prof_pe_val, int warmup, int loop_count, bool is_unilateral)
{
    int64_t pe = aclshmem_my_pe();
    if (is_unilateral && pe != prof_pe_val) {
        aclshmemx_barrier_all_vec();
        return;
    }

    int32_t current_block_index = AscendC::GetBlockIdx();
    int32_t total_blocks = AscendC::GetBlockNum();
    int ub_size_bytes = ub_size_kb * 1024;

    int32_t sio_block_count = total_blocks * HCCS_SIO_RATIO_NUM / HCCS_SIO_RATIO_DEN;
    if (sio_block_count == 0)
        sio_block_count = 1;
    if (sio_block_count > total_blocks)
        sio_block_count = total_blocks;
    int32_t hccs_block_count = total_blocks - sio_block_count;

    constexpr uint32_t sync_id = 0;

    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    int32_t block_idx = AscendC::GetBlockIdx();
    bool do_prof = (device_state != nullptr && device_state->profs != nullptr &&
                    frame_id < ACLSHMEM_CYCLE_PROF_FRAME_CNT && block_idx < ACLSHMEM_CYCLE_PROF_MAX_BLOCK &&
                    device_state->profs->pe_id == pe);

    AscendC::PipeBarrier<PIPE_ALL>();

    int64_t total_cycles = 0;
    int64_t total_count = 0;

    for (int i = 0; i < warmup + loop_count; i++) {
        uint64_t prof_start_cycle = 0;
        if (i >= warmup && do_prof) {
            pipe_barrier(PIPE_ALL);
            prof_start_cycle = AscendC::GetSystemCycle();
        }

        if (current_block_index < sio_block_count && sio_elements > 0) {
            __gm__ T *sio_dst = (__gm__ T *)sio_dst_gva;
            __gm__ T *sio_src = (__gm__ T *)sio_src_gva;

            int32_t sio_block_idx = current_block_index;
            int32_t sio_offset_elem = sio_elements / sio_block_count;
            int32_t sio_offset = sio_offset_elem * sio_block_idx;
            int32_t sio_block_elements =
                (sio_block_idx == sio_block_count - 1) ? sio_elements - sio_offset : sio_offset_elem;

            if (sio_block_elements > 0) {
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(sync_id);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(sync_id);

                aclshmemx_mte_get_nbi(
                    sio_dst + sio_offset, sio_src + sio_offset, reinterpret_cast<__ubuf__ T *>(0), ub_size_bytes,
                    sio_block_elements, peer_pe, sync_id);

                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(sync_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(sync_id);
            }
        } else if (hccs_block_count > 0 && hccs_elements > 0) {
            __gm__ T *hccs_dst = (__gm__ T *)hccs_dst_gva;
            __gm__ T *hccs_remote = (__gm__ T *)hccs_remote_gva;

            int32_t hccs_block_idx = current_block_index - sio_block_count;
            int32_t hccs_offset_elem = hccs_elements / hccs_block_count;
            int32_t hccs_offset = hccs_offset_elem * hccs_block_idx;
            int32_t hccs_block_elements =
                (hccs_block_idx == hccs_block_count - 1) ? hccs_elements - hccs_offset : hccs_offset_elem;

            if (hccs_block_elements > 0) {
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(sync_id);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(sync_id);

                perf_local_mte_get_hccs(
                    hccs_remote + hccs_offset, hccs_dst + hccs_offset, reinterpret_cast<__ubuf__ T *>(0), ub_size_bytes,
                    hccs_block_elements, sync_id);

                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(sync_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(sync_id);
            }
        }

        if (i >= warmup && do_prof) {
            pipe_barrier(PIPE_ALL);
            auto prof_end_cycle = AscendC::GetSystemCycle();
            total_cycles += (prof_end_cycle - prof_start_cycle);
            total_count += 1;
        }
    }

    if (do_prof) {
        device_state->profs->block_prof[block_idx].cycles[frame_id] += total_cycles;
        device_state->profs->block_prof[block_idx].ccount[frame_id] += total_count;
    }

    aclshmemx_barrier_all_vec();
}

template <typename T>
void launch_mte_get_mixed_perf(uint32_t block_dim, void *stream,
                                uint8_t *sio_dst, uint8_t *sio_src, int sio_elements, int peer_pe,
                                uint8_t *hccs_dst, uint8_t *hccs_remote, int hccs_elements,
                                int ub_size_kb, int32_t frame_id, int64_t prof_pe_val, int warmup, int loop_count, bool is_unilateral)
{
    mte_get_mixed_perf_kernel<T><<<block_dim, nullptr, stream>>>(
        sio_dst, sio_src, sio_elements, peer_pe,
        hccs_dst, hccs_remote, hccs_elements,
        ub_size_kb, frame_id, prof_pe_val, warmup, loop_count, is_unilateral);
}

template void launch_mte_get_mixed_perf<int>(uint32_t, void *, uint8_t *, uint8_t *, int, int, uint8_t *, uint8_t *, int, int, int32_t, int64_t, int, int, bool);
template void launch_mte_get_mixed_perf<int64_t>(uint32_t, void *, uint8_t *, uint8_t *, int, int, uint8_t *, uint8_t *, int, int, int32_t, int64_t, int, int, bool);
template void launch_mte_get_mixed_perf<float>(uint32_t, void *, uint8_t *, uint8_t *, int, int, uint8_t *, uint8_t *, int, int, int32_t, int64_t, int, int, bool);

template <typename T>
__aicore__ inline void perf_local_mte_put_hccs(__gm__ T *local_src, __gm__ T *hccs_dst,
                                                  __ubuf__ T *buf, uint32_t ub_size_bytes,
                                                  uint32_t elem_size, uint32_t sync_id)
{
    // Precondition: ub_size_bytes >= sizeof(T). If violated, block_size becomes 0 and
    // the function silently returns without transferring any data.
    uint64_t block_size = ub_size_bytes / sizeof(T) * sizeof(T);
    if (block_size == 0) {
        return;
    }
    uint64_t elem_bytes = static_cast<uint64_t>(elem_size) * sizeof(T);
    uint64_t remain = elem_bytes % block_size;
    uint64_t repeat_times = elem_bytes / block_size;
    uint64_t repeat_elem = block_size / sizeof(T);

    for (uint64_t i = 0; i < repeat_times; i++) {
        perf_local_copy_gm2ub(buf, local_src + i * repeat_elem, block_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(sync_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(sync_id);
        perf_local_copy_ub2gm(hccs_dst + i * repeat_elem, buf, block_size);
        if (i != repeat_times - 1 || remain > 0) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(sync_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(sync_id);
        }
    }
    if (remain > 0) {
        perf_local_copy_gm2ub(buf, local_src + repeat_times * repeat_elem, remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(sync_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(sync_id);
        perf_local_copy_ub2gm(hccs_dst + repeat_times * repeat_elem, buf, remain);
    }
}

template <typename T>
[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void mte_put_mixed_perf_kernel(
    GM_ADDR sio_dst_gva, GM_ADDR sio_src_gva, int sio_elements, int peer_pe,
    GM_ADDR hccs_remote_gva, GM_ADDR hccs_src_gva, int hccs_elements,
    int ub_size_kb, int32_t frame_id, int64_t prof_pe_val, int warmup, int loop_count, bool is_unilateral)
{
    int64_t pe = aclshmem_my_pe();
    if (is_unilateral && pe != prof_pe_val) {
        aclshmemx_barrier_all_vec();
        return;
    }

    int32_t current_block_index = AscendC::GetBlockIdx();
    int32_t total_blocks = AscendC::GetBlockNum();
    int ub_size_bytes = ub_size_kb * 1024;

    int32_t sio_block_count = total_blocks * HCCS_SIO_RATIO_NUM / HCCS_SIO_RATIO_DEN;
    if (sio_block_count == 0)
        sio_block_count = 1;
    if (sio_block_count > total_blocks)
        sio_block_count = total_blocks;
    int32_t hccs_block_count = total_blocks - sio_block_count;

    constexpr uint32_t sync_id = 0;

    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    int32_t block_idx = AscendC::GetBlockIdx();
    bool do_prof = (device_state != nullptr && device_state->profs != nullptr &&
                    frame_id < ACLSHMEM_CYCLE_PROF_FRAME_CNT && block_idx < ACLSHMEM_CYCLE_PROF_MAX_BLOCK &&
                    device_state->profs->pe_id == pe);

    AscendC::PipeBarrier<PIPE_ALL>();

    int64_t total_cycles = 0;
    int64_t total_count = 0;

    for (int i = 0; i < warmup + loop_count; i++) {
        uint64_t prof_start_cycle = 0;
        if (i >= warmup && do_prof) {
            pipe_barrier(PIPE_ALL);
            prof_start_cycle = AscendC::GetSystemCycle();
        }

        if (current_block_index < sio_block_count && sio_elements > 0) {
            __gm__ T *sio_dst = (__gm__ T *)sio_dst_gva;
            __gm__ T *sio_src = (__gm__ T *)sio_src_gva;

            int32_t sio_block_idx = current_block_index;
            int32_t sio_offset_elem = sio_elements / sio_block_count;
            int32_t sio_offset = sio_offset_elem * sio_block_idx;
            int32_t sio_block_elements =
                (sio_block_idx == sio_block_count - 1) ? sio_elements - sio_offset : sio_offset_elem;

            if (sio_block_elements > 0) {
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(sync_id);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(sync_id);

                aclshmemx_mte_put_nbi(
                    sio_dst + sio_offset, sio_src + sio_offset, reinterpret_cast<__ubuf__ T *>(0), ub_size_bytes,
                    sio_block_elements, peer_pe, sync_id);

                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(sync_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(sync_id);
            }
        } else if (hccs_block_count > 0 && hccs_elements > 0) {
            __gm__ T *hccs_remote = (__gm__ T *)hccs_remote_gva;
            __gm__ T *hccs_src = (__gm__ T *)hccs_src_gva;

            int32_t hccs_block_idx = current_block_index - sio_block_count;
            int32_t hccs_offset_elem = hccs_elements / hccs_block_count;
            int32_t hccs_offset = hccs_offset_elem * hccs_block_idx;
            int32_t hccs_block_elements =
                (hccs_block_idx == hccs_block_count - 1) ? hccs_elements - hccs_offset : hccs_offset_elem;

            if (hccs_block_elements > 0) {
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(sync_id);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(sync_id);

                perf_local_mte_put_hccs(
                    hccs_src + hccs_offset, hccs_remote + hccs_offset, reinterpret_cast<__ubuf__ T *>(0), ub_size_bytes,
                    hccs_block_elements, sync_id);

                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(sync_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(sync_id);
            }
        }

        if (i >= warmup && do_prof) {
            pipe_barrier(PIPE_ALL);
            auto prof_end_cycle = AscendC::GetSystemCycle();
            total_cycles += (prof_end_cycle - prof_start_cycle);
            total_count += 1;
        }
    }

    if (do_prof) {
        device_state->profs->block_prof[block_idx].cycles[frame_id] += total_cycles;
        device_state->profs->block_prof[block_idx].ccount[frame_id] += total_count;
    }

    aclshmemx_barrier_all_vec();
}

template <typename T>
void launch_mte_put_mixed_perf(uint32_t block_dim, void *stream,
                                uint8_t *sio_dst, uint8_t *sio_src, int sio_elements, int peer_pe,
                                uint8_t *hccs_remote, uint8_t *hccs_src, int hccs_elements,
                                int ub_size_kb, int32_t frame_id, int64_t prof_pe_val, int warmup, int loop_count, bool is_unilateral)
{
    mte_put_mixed_perf_kernel<T><<<block_dim, nullptr, stream>>>(
        sio_dst, sio_src, sio_elements, peer_pe,
        hccs_remote, hccs_src, hccs_elements,
        ub_size_kb, frame_id, prof_pe_val, warmup, loop_count, is_unilateral);
}

template void launch_mte_put_mixed_perf<int>(uint32_t, void *, uint8_t *, uint8_t *, int, int, uint8_t *, uint8_t *, int, int, int32_t, int64_t, int, int, bool);
template void launch_mte_put_mixed_perf<int64_t>(uint32_t, void *, uint8_t *, uint8_t *, int, int, uint8_t *, uint8_t *, int, int, int32_t, int64_t, int, int, bool);
template void launch_mte_put_mixed_perf<float>(uint32_t, void *, uint8_t *, uint8_t *, int, int, uint8_t *, uint8_t *, int, int, int32_t, int64_t, int, int, bool);
