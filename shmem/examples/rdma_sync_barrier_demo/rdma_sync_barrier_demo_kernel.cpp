/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RDMA_SYNC_BARRIER_DEMO_KERNEL_
#define _RDMA_SYNC_BARRIER_DEMO_KERNEL_

#include "kernel_operator.h"
#include "shmem.h"

// ---- roce_sync_all demo ----
// 每个 PE 向所有其他 PE 发送数据，quiet 确保数据到达，sync_all 确保同步完成
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void device_roce_sync_all_demo(
    GM_ADDR gva, int message_length)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);
    int64_t my_rank = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();
    AscendC::PipeBarrier<PIPE_ALL>();

    aclshmemx_roce_barrier_all();

    for (int i = 0; i < pe_size; i++) {
        if (i == my_rank)
            continue;
        aclshmemx_roce_put_nbi(
            gva + my_rank * message_length, gva + my_rank * message_length, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(),
            message_length, i, 0);
    }
    for (int i = 0; i < pe_size; i++) {
        if (i == my_rank)
            continue;
        aclshmemx_roce_quiet(i, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
    }
    aclshmemx_roce_sync_all();
}

// ---- roce_barrier_all demo ----
// 每个 PE 向所有其他 PE 发送数据，barrier_all 确保 RDMA 完成并同步
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void device_roce_barrier_all_demo(
    GM_ADDR gva, int message_length)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);
    int64_t my_rank = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();
    AscendC::PipeBarrier<PIPE_ALL>();

    aclshmemx_roce_barrier_all();

    for (int i = 0; i < pe_size; i++) {
        if (i == my_rank)
            continue;
        aclshmemx_roce_put_nbi(
            gva + my_rank * message_length, gva + my_rank * message_length, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(),
            message_length, i, 0);
    }
    aclshmemx_roce_barrier_all();
}

// ---- roce_barrier_all(buf, sync_id) demo ----
// 每个 PE 向所有其他 PE 发送数据，barrier_all(buf, sync_id) 确保 RDMA 完成并同步
// 使用显式传入的 UB buffer 和 sync_id，避免与 rdma_config 默认资源冲突
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void device_roce_barrier_all_with_buf_demo(
    GM_ADDR gva, int message_length)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);
    int64_t my_rank = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();
    AscendC::PipeBarrier<PIPE_ALL>();

    aclshmemx_roce_barrier_all();

    for (int i = 0; i < pe_size; i++) {
        if (i == my_rank)
            continue;
        aclshmemx_roce_put_nbi(
            gva + my_rank * message_length, gva + my_rank * message_length, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(),
            message_length, i, 0);
    }
    aclshmemx_roce_barrier_all((__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
}

// ---- roce_sync_all(buf, sync_id) demo ----
// 每个 PE 向所有其他 PE 发送数据，quiet 确保数据到达，sync_all(buf, sync_id) 确保同步完成
// 使用显式传入的 UB buffer 和 sync_id，避免与 rdma_config 默认资源冲突
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void device_roce_sync_all_with_buf_demo(
    GM_ADDR gva, int message_length)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);
    int64_t my_rank = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();
    AscendC::PipeBarrier<PIPE_ALL>();

    aclshmemx_roce_barrier_all();

    for (int i = 0; i < pe_size; i++) {
        if (i == my_rank)
            continue;
        aclshmemx_roce_put_nbi(
            gva + my_rank * message_length, gva + my_rank * message_length, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(),
            message_length, i, 0);
    }
    for (int i = 0; i < pe_size; i++) {
        if (i == my_rank)
            continue;
        aclshmemx_roce_quiet(i, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
    }
    aclshmemx_roce_sync_all((__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
}

// ---- roce_team_sync demo ----
// team 内每个 PE 向所有其他 team 成员发送数据，quiet 确保数据到达，team_sync 确保同步完成
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void device_roce_sync_team_demo(
    GM_ADDR gva, int message_length, aclshmem_team_t team_id)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);
    int64_t my_rank = aclshmem_my_pe();
    AscendC::PipeBarrier<PIPE_ALL>();

    aclshmemx_team_t* team = aclshmemi_get_state()->team_pools[team_id];
    int start = team->start;
    int stride = team->stride;
    int size = team->size;
    for (int i = 0; i < size; i++) {
        int peer = start + i * stride;
        if (peer == my_rank)
            continue;
        aclshmemx_roce_put_nbi(
            gva + my_rank * message_length, gva + my_rank * message_length, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(),
            message_length, peer, 0);
    }
    aclshmemx_roce_barrier(team);
    for (int i = 0; i < size; i++) {
        int peer = start + i * stride;
        if (peer == my_rank)
            continue;
        aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
    }
    aclshmemx_roce_team_sync(team);
}

// ---- roce_barrier(team) demo ----
// team 内每个 PE 向所有其他 team 成员发送数据，barrier 确保 RDMA 完成并同步
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void device_roce_barrier_team_demo(
    GM_ADDR gva, int message_length, aclshmem_team_t team_id)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);
    int64_t my_rank = aclshmem_my_pe();
    AscendC::PipeBarrier<PIPE_ALL>();

    aclshmemx_team_t* team = aclshmemi_get_state()->team_pools[team_id];
    int start = team->start;
    int stride = team->stride;
    int size = team->size;

    aclshmemx_roce_barrier(team);

    for (int i = 0; i < size; i++) {
        int peer = start + i * stride;
        if (peer == my_rank)
            continue;
        aclshmemx_roce_put_nbi(
            gva + my_rank * message_length, gva + my_rank * message_length, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(),
            message_length, peer, 0);
    }
    aclshmemx_roce_barrier(team);
}

// ---- roce_sync(team, buf, sync_id) demo ----
// team 内每个 PE 向所有其他 team 成员发送数据，quiet 确保数据到达，team_sync(buf, sync_id) 确保同步完成
// 使用显式传入的 UB buffer 和 sync_id，避免与 rdma_config 默认资源冲突
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void device_roce_sync_team_with_buf_demo(
    GM_ADDR gva, int message_length, aclshmem_team_t team_id)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);
    int64_t my_rank = aclshmem_my_pe();
    AscendC::PipeBarrier<PIPE_ALL>();

    aclshmemx_team_t* team = aclshmemi_get_state()->team_pools[team_id];
    int start = team->start;
    int stride = team->stride;
    int size = team->size;
    aclshmemx_roce_barrier(team);
    for (int i = 0; i < size; i++) {
        int peer = start + i * stride;
        if (peer == my_rank)
            continue;
        aclshmemx_roce_put_nbi(
            gva + my_rank * message_length, gva + my_rank * message_length, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(),
            message_length, peer, 0);
    }
    for (int i = 0; i < size; i++) {
        int peer = start + i * stride;
        if (peer == my_rank)
            continue;
        aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
    }
    aclshmemx_roce_sync(team, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
}

// ---- roce_barrier(team, buf, sync_id) demo ----
// team 内每个 PE 向所有其他 team 成员发送数据，barrier(team, buf, sync_id) 确保 RDMA 完成并同步
// 使用显式传入的 UB buffer 和 sync_id，避免与 rdma_config 默认资源冲突
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void device_roce_barrier_team_with_buf_demo(
    GM_ADDR gva, int message_length, aclshmem_team_t team_id)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);
    int64_t my_rank = aclshmem_my_pe();
    AscendC::PipeBarrier<PIPE_ALL>();

    aclshmemx_team_t* team = aclshmemi_get_state()->team_pools[team_id];
    int start = team->start;
    int stride = team->stride;
    int size = team->size;

    aclshmemx_roce_barrier(team);

    for (int i = 0; i < size; i++) {
        int peer = start + i * stride;
        if (peer == my_rank)
            continue;
        aclshmemx_roce_put_nbi(
            gva + my_rank * message_length, gva + my_rank * message_length, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(),
            message_length, peer, 0);
    }
    aclshmemx_roce_barrier(team, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
}

// ---- Host 端启动函数 ----
void roce_sync_all_demo(uint32_t block_dim, void* stream, uint8_t* gva, int message_length)
{
    device_roce_sync_all_demo<<<block_dim, nullptr, stream>>>(gva, message_length);
}

void roce_sync_all_with_buf_demo(uint32_t block_dim, void* stream, uint8_t* gva, int message_length)
{
    device_roce_sync_all_with_buf_demo<<<block_dim, nullptr, stream>>>(gva, message_length);
}

void roce_barrier_all_demo(uint32_t block_dim, void* stream, uint8_t* gva, int message_length)
{
    device_roce_barrier_all_demo<<<block_dim, nullptr, stream>>>(gva, message_length);
}

void roce_barrier_all_with_buf_demo(uint32_t block_dim, void* stream, uint8_t* gva, int message_length)
{
    device_roce_barrier_all_with_buf_demo<<<block_dim, nullptr, stream>>>(gva, message_length);
}

void roce_sync_team_demo(uint32_t block_dim, void* stream, uint8_t* gva, int message_length, aclshmem_team_t team_id)
{
    device_roce_sync_team_demo<<<block_dim, nullptr, stream>>>(gva, message_length, team_id);
}

void roce_sync_team_with_buf_demo(
    uint32_t block_dim, void* stream, uint8_t* gva, int message_length, aclshmem_team_t team_id)
{
    device_roce_sync_team_with_buf_demo<<<block_dim, nullptr, stream>>>(gva, message_length, team_id);
}

void roce_barrier_team_demo(uint32_t block_dim, void* stream, uint8_t* gva, int message_length, aclshmem_team_t team_id)
{
    device_roce_barrier_team_demo<<<block_dim, nullptr, stream>>>(gva, message_length, team_id);
}

void roce_barrier_team_with_buf_demo(
    uint32_t block_dim, void* stream, uint8_t* gva, int message_length, aclshmem_team_t team_id)
{
    device_roce_barrier_team_with_buf_demo<<<block_dim, nullptr, stream>>>(gva, message_length, team_id);
}

#endif // _RDMA_SYNC_BARRIER_DEMO_KERNEL_
