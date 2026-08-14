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
constexpr uint64_t MESSAGE_SIZE = 64;
constexpr uint32_t UDMA_ACTION_BATCH_COUNT = 2;
constexpr uint32_t UDMA_ACTION_UB_SIZE = MESSAGE_SIZE * UDMA_ACTION_BATCH_COUNT;

ACLSHMEM_DEVICE GM_ADDR get_udma_action_slot(GM_ADDR gva, int64_t rank, uint32_t batch_idx)
{
    return gva + (rank * UDMA_ACTION_BATCH_COUNT + batch_idx) * MESSAGE_SIZE;
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAGetTest(GM_ADDR gva)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE * 2, 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    GM_ADDR dest_addr;

    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        dest_addr = gva + peer * MESSAGE_SIZE;
        aclshmemx_udma_get_nbi<uint8_t, PIPE_S>(
            dest_addr, dest_addr, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), MESSAGE_SIZE, peer);
        aclshmemx_udma_quiet(peer);
    }
}

void test_udma_get(uint32_t block_dim, void* stream, uint8_t* gva) { UDMAGetTest<<<block_dim, nullptr, stream>>>(gva); }

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAGetActionPointerTest(GM_ADDR gva)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UDMA_ACTION_UB_SIZE);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UDMA_ACTION_UB_SIZE, 0);
    __ubuf__ uint8_t* ub_ptr = (__ubuf__ uint8_t*)ubLocal.GetPhyAddr();

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        aclshmemx_submit_state_t state{};
        aclshmemx_defer_t defer_action(state);
        aclshmemx_submit_t submit_action(state);
        for (uint32_t i = 0; i < UDMA_ACTION_BATCH_COUNT; ++i) {
            GM_ADDR dest_addr = get_udma_action_slot(gva, peer, i);
            if (i + 1 == UDMA_ACTION_BATCH_COUNT) {
                aclshmemx_udma_get_nbi<uint8_t, PIPE_MTE3>(
                    dest_addr, dest_addr, ub_ptr, MESSAGE_SIZE, peer, 0, submit_action);
            } else {
                aclshmemx_udma_get_nbi<uint8_t, PIPE_MTE3>(
                    dest_addr, dest_addr, ub_ptr, MESSAGE_SIZE, peer, 0, defer_action);
            }
        }
        aclshmemx_udma_quiet(peer);
    }
}

void test_udma_get_action_pointer(uint32_t block_dim, void* stream, uint8_t* gva)
{
    UDMAGetActionPointerTest<<<block_dim, nullptr, stream>>>(gva);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAGetActionTensorTest(GM_ADDR gva)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UDMA_ACTION_UB_SIZE);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UDMA_ACTION_UB_SIZE, 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        aclshmemx_submit_state_t state{};
        aclshmemx_defer_t defer_action(state);
        aclshmemx_submit_t submit_action(state);
        for (uint32_t i = 0; i < UDMA_ACTION_BATCH_COUNT; ++i) {
            GM_ADDR dest_addr = get_udma_action_slot(gva, peer, i);
            AscendC::GlobalTensor<uint8_t> dst_tensor;
            AscendC::GlobalTensor<uint8_t> src_tensor;
            dst_tensor.SetGlobalBuffer((__gm__ uint8_t*)dest_addr, MESSAGE_SIZE);
            src_tensor.SetGlobalBuffer((__gm__ uint8_t*)dest_addr, MESSAGE_SIZE);
            if (i + 1 == UDMA_ACTION_BATCH_COUNT) {
                aclshmemx_udma_get_nbi<uint8_t, PIPE_MTE3>(
                    dst_tensor, src_tensor, ubLocal, MESSAGE_SIZE, peer, 0, submit_action);
            } else {
                aclshmemx_udma_get_nbi<uint8_t, PIPE_MTE3>(
                    dst_tensor, src_tensor, ubLocal, MESSAGE_SIZE, peer, 0, defer_action);
            }
        }
        aclshmemx_udma_quiet(peer);
    }
}

void test_udma_get_action_tensor(uint32_t block_dim, void* stream, uint8_t* gva)
{
    UDMAGetActionTensorTest<<<block_dim, nullptr, stream>>>(gva);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAPutTest(GM_ADDR gva)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE * 2, 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    GM_ADDR src_addr;

    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        src_addr = gva + rank * MESSAGE_SIZE;
        aclshmemx_udma_put_nbi<uint8_t, PIPE_S>(
            src_addr, src_addr, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), MESSAGE_SIZE, peer);
        aclshmemx_udma_quiet(peer);
    }
}

void test_udma_put(uint32_t block_dim, void* stream, uint8_t* gva) { UDMAPutTest<<<block_dim, nullptr, stream>>>(gva); }

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAPutActionPointerTest(GM_ADDR gva)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UDMA_ACTION_UB_SIZE);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UDMA_ACTION_UB_SIZE, 0);
    __ubuf__ uint8_t* ub_ptr = (__ubuf__ uint8_t*)ubLocal.GetPhyAddr();

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        aclshmemx_submit_state_t state{};
        aclshmemx_defer_t defer_action(state);
        aclshmemx_submit_t submit_action(state);
        for (uint32_t i = 0; i < UDMA_ACTION_BATCH_COUNT; ++i) {
            GM_ADDR src_addr = get_udma_action_slot(gva, rank, i);
            if (i + 1 == UDMA_ACTION_BATCH_COUNT) {
                aclshmemx_udma_put_nbi<uint8_t, PIPE_MTE3>(
                    src_addr, src_addr, ub_ptr, MESSAGE_SIZE, peer, 0, submit_action);
            } else {
                aclshmemx_udma_put_nbi<uint8_t, PIPE_MTE3>(
                    src_addr, src_addr, ub_ptr, MESSAGE_SIZE, peer, 0, defer_action);
            }
        }
        aclshmemx_udma_quiet(peer);
    }
}

void test_udma_put_action_pointer(uint32_t block_dim, void* stream, uint8_t* gva)
{
    UDMAPutActionPointerTest<<<block_dim, nullptr, stream>>>(gva);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAPutActionTensorTest(GM_ADDR gva)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UDMA_ACTION_UB_SIZE);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UDMA_ACTION_UB_SIZE, 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        aclshmemx_submit_state_t state{};
        aclshmemx_defer_t defer_action(state);
        aclshmemx_submit_t submit_action(state);
        for (uint32_t i = 0; i < UDMA_ACTION_BATCH_COUNT; ++i) {
            GM_ADDR src_addr = get_udma_action_slot(gva, rank, i);
            AscendC::GlobalTensor<uint8_t> dst_tensor;
            AscendC::GlobalTensor<uint8_t> src_tensor;
            dst_tensor.SetGlobalBuffer((__gm__ uint8_t*)src_addr, MESSAGE_SIZE);
            src_tensor.SetGlobalBuffer((__gm__ uint8_t*)src_addr, MESSAGE_SIZE);
            if (i + 1 == UDMA_ACTION_BATCH_COUNT) {
                aclshmemx_udma_put_nbi<uint8_t, PIPE_MTE3>(
                    dst_tensor, src_tensor, ubLocal, MESSAGE_SIZE, peer, 0, submit_action);
            } else {
                aclshmemx_udma_put_nbi<uint8_t, PIPE_MTE3>(
                    dst_tensor, src_tensor, ubLocal, MESSAGE_SIZE, peer, 0, defer_action);
            }
        }
        aclshmemx_udma_quiet(peer);
    }
}

void test_udma_put_action_tensor(uint32_t block_dim, void* stream, uint8_t* gva)
{
    UDMAPutActionTensorTest<<<block_dim, nullptr, stream>>>(gva);
}

#ifdef ACLSHMEM_RELAY_SUPPORT
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMARelayActionCompileTest(GM_ADDR gva)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UDMA_ACTION_UB_SIZE);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UDMA_ACTION_UB_SIZE, 0);
    __ubuf__ uint8_t* ub_ptr = (__ubuf__ uint8_t*)ubLocal.GetPhyAddr();

    int my_pe = static_cast<int>(aclshmem_my_pe());
    int npes = static_cast<int>(aclshmem_n_pes());
    if (npes < 3) {
        return;
    }
    int pe = (my_pe + 1) % npes;
    int relay_pe = (my_pe + 2) % npes;
    GM_ADDR addr = get_udma_action_slot(gva, my_pe, 0);
    AscendC::GlobalTensor<uint8_t> dst_tensor;
    AscendC::GlobalTensor<uint8_t> src_tensor;
    dst_tensor.SetGlobalBuffer((__gm__ uint8_t*)addr, MESSAGE_SIZE);
    src_tensor.SetGlobalBuffer((__gm__ uint8_t*)addr, MESSAGE_SIZE);

    aclshmemx_submit_state_t put_ptr_state{};
    aclshmemx_defer_t put_ptr_defer(put_ptr_state);
    aclshmemx_submit_t put_ptr_submit(put_ptr_state);
    aclshmemx_udma_relay_put_nbi<uint8_t, PIPE_MTE3>(addr, addr, ub_ptr, MESSAGE_SIZE, pe, relay_pe, 0, put_ptr_defer);
    aclshmemx_udma_relay_put_nbi<uint8_t, PIPE_MTE3>(addr, addr, ub_ptr, MESSAGE_SIZE, pe, relay_pe, 0, put_ptr_submit);

    aclshmemx_submit_state_t put_tensor_state{};
    aclshmemx_defer_t put_tensor_defer(put_tensor_state);
    aclshmemx_submit_t put_tensor_submit(put_tensor_state);
    aclshmemx_udma_relay_put_nbi<uint8_t, PIPE_MTE3>(
        dst_tensor, src_tensor, ubLocal, MESSAGE_SIZE, pe, relay_pe, 0, put_tensor_defer);
    aclshmemx_udma_relay_put_nbi<uint8_t, PIPE_MTE3>(
        dst_tensor, src_tensor, ubLocal, MESSAGE_SIZE, pe, relay_pe, 0, put_tensor_submit);

    aclshmemx_submit_state_t get_ptr_state{};
    aclshmemx_defer_t get_ptr_defer(get_ptr_state);
    aclshmemx_submit_t get_ptr_submit(get_ptr_state);
    aclshmemx_udma_relay_get_nbi<uint8_t, PIPE_MTE3>(addr, addr, ub_ptr, MESSAGE_SIZE, pe, relay_pe, 0, get_ptr_defer);
    aclshmemx_udma_relay_get_nbi<uint8_t, PIPE_MTE3>(addr, addr, ub_ptr, MESSAGE_SIZE, pe, relay_pe, 0, get_ptr_submit);

    aclshmemx_submit_state_t get_tensor_state{};
    aclshmemx_defer_t get_tensor_defer(get_tensor_state);
    aclshmemx_submit_t get_tensor_submit(get_tensor_state);
    aclshmemx_udma_relay_get_nbi<uint8_t, PIPE_MTE3>(
        dst_tensor, src_tensor, ubLocal, MESSAGE_SIZE, pe, relay_pe, 0, get_tensor_defer);
    aclshmemx_udma_relay_get_nbi<uint8_t, PIPE_MTE3>(
        dst_tensor, src_tensor, ubLocal, MESSAGE_SIZE, pe, relay_pe, 0, get_tensor_submit);
}
#endif

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAPutSignalTest(GM_ADDR gva, GM_ADDR sig_addr)
{
    int64_t my_pe = aclshmem_my_pe();
    int64_t npes = aclshmem_n_pes();
    GM_ADDR src_addr;

    for (int64_t peer = 0; peer < npes; peer++) {
        if (peer == my_pe) {
            continue;
        }
        src_addr = gva + my_pe * MESSAGE_SIZE;
        uint64_t signal = 1000;
        auto dst_sig_addr = sig_addr + sizeof(uint64_t) * my_pe;
        aclshmemx_udma_put_signal_nbi(src_addr, src_addr, MESSAGE_SIZE, (__gm__ uint64_t*)dst_sig_addr, signal, peer);
        aclshmemx_udma_quiet(peer);
#ifndef ACLSHMEM_RELAY_SUPPORT
        aclshmemx_udma_qp_put_signal_nbi(
            src_addr, src_addr, MESSAGE_SIZE, (__gm__ uint64_t*)dst_sig_addr, signal, peer, 0);
        aclshmemx_udma_qp_quiet(peer, 0);
#endif
    }
}

void test_udma_put_signal(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr)
{
    UDMAPutSignalTest<<<block_dim, nullptr, stream>>>(gva, sig_addr);
}

#ifndef ACLSHMEM_RELAY_SUPPORT
enum class UdmaQpTestOperation : int32_t {
    PUT = 0,
    GET = 1,
    PUT_SIGNAL = 2,
};

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UDMAQpDataPathTest(
    GM_ADDR symmetric, GM_ADDR local_buffer, GM_ADDR signal_words, uint32_t slice_size, int32_t peer, int32_t operation)
{
    const uint32_t qp_idx = AscendC::GetBlockIdx();
    const uint64_t slice_offset = static_cast<uint64_t>(qp_idx) * slice_size;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> scratch;
    pipe.InitBuffer(scratch, ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE);
    __ubuf__ uint8_t* wqe_scratch = reinterpret_cast<__ubuf__ uint8_t*>(scratch.Get<uint8_t>().GetPhyAddr());

    if (operation == static_cast<int32_t>(UdmaQpTestOperation::PUT)) {
        aclshmemx_udma_qp_put_nbi<uint8_t>(
            symmetric + slice_offset, local_buffer + slice_offset, wqe_scratch, slice_size, peer, qp_idx);
    } else if (operation == static_cast<int32_t>(UdmaQpTestOperation::GET)) {
        aclshmemx_udma_qp_get_nbi<uint8_t>(
            local_buffer + slice_offset, symmetric + slice_offset, wqe_scratch, slice_size, peer, qp_idx);
    } else {
        const uint64_t signal = (static_cast<uint64_t>(aclshmem_my_pe() + 1) << 32) | (qp_idx + 1);
        aclshmemx_udma_qp_put_signal_nbi<uint8_t>(
            symmetric + slice_offset, local_buffer + slice_offset, slice_size,
            reinterpret_cast<__gm__ uint64_t*>(signal_words) + qp_idx, signal, peer, qp_idx, wqe_scratch);
    }
    aclshmemx_udma_qp_quiet(peer, qp_idx);
}

void test_udma_qp_data_path(
    uint32_t block_dim, void* stream, uint8_t* symmetric, uint8_t* local_buffer, uint64_t* signal_words,
    uint32_t slice_size, int32_t peer, int32_t operation)
{
    UDMAQpDataPathTest<<<block_dim, nullptr, stream>>>(
        symmetric, local_buffer, reinterpret_cast<uint8_t*>(signal_words), slice_size, peer, operation);
}
#endif
