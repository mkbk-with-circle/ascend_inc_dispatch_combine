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

// ---- aclshmemx_roce_sync_all 测试 ----
// 每个 PE 向所有其他 PE 发送 message_length 大小的数据，quiet 确保数据到达，sync_all 确保同步完成
extern "C" ACLSHMEM_GLOBAL_VECTOR void roce_sync_all_data(
    uint64_t config, GM_ADDR addr, int rank_id, int rank_size, int message_length)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);
    AscendC::PipeBarrier<PIPE_ALL>();

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    aclshmemx_roce_barrier_all();
    for (int i = 0; i < rank_size; i++) {
        if (i == rank_id)
            continue;
        aclshmemx_roce_put_nbi(
            addr + rank_id * message_length, addr + rank_id * message_length, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(),
            message_length, i, 0);
    }
    for (int i = 0; i < rank_size; i++) {
        if (i == rank_id)
            continue;
        aclshmemx_roce_quiet(i, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
    }
    aclshmemx_roce_sync_all();
#endif
}

// ---- aclshmemx_roce_sync_all(buf, sync_id) 测试 ----
// 每个 PE 向所有其他 PE 发送 message_length 大小的数据，quiet 确保数据到达，sync_all(buf, sync_id) 确保同步完成
// 使用显式传入的 UB buffer 和 sync_id，避免与 rdma_config 默认资源冲突
extern "C" ACLSHMEM_GLOBAL_VECTOR void roce_sync_all_with_buf_data(
    uint64_t config, GM_ADDR addr, int rank_id, int rank_size, int message_length)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);
    AscendC::PipeBarrier<PIPE_ALL>();

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    aclshmemx_roce_barrier_all();
    for (int i = 0; i < rank_size; i++) {
        if (i == rank_id)
            continue;
        aclshmemx_roce_put_nbi(
            addr + rank_id * message_length, addr + rank_id * message_length, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(),
            message_length, i, 0);
    }
    for (int i = 0; i < rank_size; i++) {
        if (i == rank_id)
            continue;
        aclshmemx_roce_quiet(i, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
    }
    aclshmemx_roce_sync_all((__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
#endif
}

// ---- aclshmemx_roce_barrier_all 测试 ----
extern "C" ACLSHMEM_GLOBAL_VECTOR void roce_barrier_all_data(
    uint64_t config, GM_ADDR addr, int rank_id, int rank_size, int message_length)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);
    AscendC::PipeBarrier<PIPE_ALL>();

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    aclshmemx_roce_barrier_all();
    for (int i = 0; i < rank_size; i++) {
        if (i == rank_id)
            continue;
        aclshmemx_roce_put_nbi(
            addr + rank_id * message_length, addr + rank_id * message_length, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(),
            message_length, i, 0);
    }
    aclshmemx_roce_barrier_all();
#endif
}

// ---- aclshmemx_roce_barrier_all(buf, sync_id) 测试 ----
extern "C" ACLSHMEM_GLOBAL_VECTOR void roce_barrier_all_with_buf_data(
    uint64_t config, GM_ADDR addr, int rank_id, int rank_size, int message_length)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);
    AscendC::PipeBarrier<PIPE_ALL>();

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    aclshmemx_roce_barrier_all();
    for (int i = 0; i < rank_size; i++) {
        if (i == rank_id)
            continue;
        aclshmemx_roce_put_nbi(
            addr + rank_id * message_length, addr + rank_id * message_length, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(),
            message_length, i, 0);
    }
    aclshmemx_roce_barrier_all((__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
#endif
}

// ---- aclshmemx_roce_team_sync 测试 ----
// team 内每个 PE 向所有其他 team 成员发送数据，quiet 确保数据到达，team_sync 确保同步完成
extern "C" ACLSHMEM_GLOBAL_VECTOR void roce_sync_team_data(
    uint64_t config, GM_ADDR addr, int rank_id, int rank_size, int message_length, aclshmem_team_t team_id)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);
    AscendC::PipeBarrier<PIPE_ALL>();

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    aclshmemx_team_t* team = aclshmemi_get_state()->team_pools[team_id];
    int start = team->start;
    int stride = team->stride;
    int size = team->size;
    aclshmemx_roce_barrier(team);
    for (int i = 0; i < size; i++) {
        int peer = start + i * stride;
        if (peer == rank_id)
            continue;
        aclshmemx_roce_put_nbi(
            addr + rank_id * message_length, addr + rank_id * message_length, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(),
            message_length, peer, 0);
    }
    for (int i = 0; i < size; i++) {
        int peer = start + i * stride;
        if (peer == rank_id)
            continue;
        aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
    }
    aclshmemx_roce_sync(team);
#endif
}

// ---- aclshmemx_roce_team_sync(buf, sync_id) 测试 ----
// team 内每个 PE 向所有其他 team 成员发送数据，quiet 确保数据到达，team_sync(buf, sync_id) 确保同步完成
// 使用显式传入的 UB buffer 和 sync_id，避免与 rdma_config 默认资源冲突
extern "C" ACLSHMEM_GLOBAL_VECTOR void roce_sync_team_with_buf_data(
    uint64_t config, GM_ADDR addr, int rank_id, int rank_size, int message_length, aclshmem_team_t team_id)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);
    AscendC::PipeBarrier<PIPE_ALL>();

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    aclshmemx_team_t* team = aclshmemi_get_state()->team_pools[team_id];
    int start = team->start;
    int stride = team->stride;
    int size = team->size;
    aclshmemx_roce_barrier(team);
    for (int i = 0; i < size; i++) {
        int peer = start + i * stride;
        if (peer == rank_id)
            continue;
        aclshmemx_roce_put_nbi(
            addr + rank_id * message_length, addr + rank_id * message_length, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(),
            message_length, peer, 0);
    }
    for (int i = 0; i < size; i++) {
        int peer = start + i * stride;
        if (peer == rank_id)
            continue;
        aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
    }
    aclshmemx_roce_sync(team, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
#endif
}

// ---- aclshmemx_roce_barrier(team) 测试 ----
extern "C" ACLSHMEM_GLOBAL_VECTOR void roce_barrier_team_data(
    uint64_t config, GM_ADDR addr, int rank_id, int rank_size, int message_length, aclshmem_team_t team_id)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);
    AscendC::PipeBarrier<PIPE_ALL>();

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    aclshmemx_team_t* team = aclshmemi_get_state()->team_pools[team_id];
    int start = team->start;
    int stride = team->stride;
    int size = team->size;
    aclshmemx_roce_barrier(team);
    for (int i = 0; i < size; i++) {
        int peer = start + i * stride;
        if (peer == rank_id)
            continue;
        aclshmemx_roce_put_nbi(
            addr + rank_id * message_length, addr + rank_id * message_length, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(),
            message_length, peer, 0);
    }
    aclshmemx_roce_barrier(team);
#endif
}

// ---- aclshmemx_roce_barrier(team, buf, sync_id) 测试 ----
extern "C" ACLSHMEM_GLOBAL_VECTOR void roce_barrier_team_with_buf_data(
    uint64_t config, GM_ADDR addr, int rank_id, int rank_size, int message_length, aclshmem_team_t team_id)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);
    AscendC::PipeBarrier<PIPE_ALL>();

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    aclshmemx_team_t* team = aclshmemi_get_state()->team_pools[team_id];
    int start = team->start;
    int stride = team->stride;
    int size = team->size;
    aclshmemx_roce_barrier(team);
    for (int i = 0; i < size; i++) {
        int peer = start + i * stride;
        if (peer == rank_id)
            continue;
        aclshmemx_roce_put_nbi(
            addr + rank_id * message_length, addr + rank_id * message_length, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(),
            message_length, peer, 0);
    }
    aclshmemx_roce_barrier(team, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
#endif
}

// ---- aclshmemx_roce_quiet 测试 ----
// PE 0 先 put message_length 大小的数据到 PE 1，再 quiet，确保数据到达
extern "C" ACLSHMEM_GLOBAL_VECTOR void roce_quiet_data_test(
    uint64_t config, GM_ADDR addr, int rank_id, int rank_size, int message_length)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);
    AscendC::PipeBarrier<PIPE_ALL>();

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    if (rank_id == 0) {
        int next_pe = (rank_id + 1) % rank_size;
        aclshmemx_roce_put_nbi(
            addr + rank_id * message_length, addr + rank_id * message_length, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(),
            message_length, next_pe, 0);
        aclshmemx_roce_quiet(next_pe, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
    }
    aclshmemx_roce_sync_all();
#endif
}

// ---- Host 端启动函数 ----
void roce_sync_all_data_do(void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, int message_length)
{
    roce_sync_all_data<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size, message_length);
}

void roce_sync_all_with_buf_data_do(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, int message_length)
{
    roce_sync_all_with_buf_data<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size, message_length);
}

void roce_barrier_all_data_do(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, int message_length)
{
    roce_barrier_all_data<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size, message_length);
}

void roce_barrier_all_with_buf_data_do(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, int message_length)
{
    roce_barrier_all_with_buf_data<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size, message_length);
}

void roce_sync_team_data_do(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, int message_length,
    aclshmem_team_t team_id)
{
    roce_sync_team_data<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size, message_length, team_id);
}

void roce_sync_team_with_buf_data_do(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, int message_length,
    aclshmem_team_t team_id)
{
    roce_sync_team_with_buf_data<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size, message_length, team_id);
}

void roce_barrier_team_data_do(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, int message_length,
    aclshmem_team_t team_id)
{
    roce_barrier_team_data<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size, message_length, team_id);
}

void roce_barrier_team_with_buf_data_do(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, int message_length,
    aclshmem_team_t team_id)
{
    roce_barrier_team_with_buf_data<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size, message_length, team_id);
}

void roce_quiet_data_test_do(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, int message_length)
{
    roce_quiet_data_test<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size, message_length);
}
