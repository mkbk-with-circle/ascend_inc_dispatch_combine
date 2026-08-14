/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RDMAPERF_KERNEL_RDMA_PERFTEST_
#define _RDMAPERF_KERNEL_RDMA_PERFTEST_

#include "kernel_operator.h"
#include "shmem.h"
#include "perftest_common_types.h"

template <typename T>
__aicore__ inline void rdma_perf_test_put_impl(
    uint64_t fftsAddr, GM_ADDR dst_gva, GM_ADDR src_gva, int elements, perftest::rdma_mode_t test_mode, int ub_size_b,
    int loop_count, int metric, int batch, uint32_t sync_id, GM_ADDR timing_out_gva)
{
    util_set_ffts_config(fftsAddr);
    int64_t pe = aclshmem_my_pe();
    int peer_pe = (pe + 1) % aclshmem_n_pes();
    __gm__ int64_t* timing_out = reinterpret_cast<__gm__ int64_t*>(timing_out_gva);

    bool is_bidir = (test_mode == perftest::TEST_MODE_RDMA_BI_PUT);

    // Ensure both PEs are synchronized before any RDMA operations begin
    aclshmemx_roce_barrier_all();

    if (!is_bidir && pe != 0) {
        if (timing_out != nullptr) {
            timing_out[1] = 0;
        }
        aclshmemx_roce_barrier_all();
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, ub_size_b);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(ub_size_b, 0);
    __ubuf__ T* ub_ptr = reinterpret_cast<__ubuf__ T*>(ubLocal.GetPhyAddr());

    __gm__ T* dst_gm = reinterpret_cast<__gm__ T*>(dst_gva);
    __gm__ T* src_gm = reinterpret_cast<__gm__ T*>(src_gva);

    int warmup = perftest::PERFTEST_WARMUP_ITERS;
    int loop_test = loop_count;
    int batch_size = (batch <= 0 || batch > loop_test) ? loop_test : batch;

    const uint64_t msg_size = static_cast<uint64_t>(elements) * sizeof(T);
    const bool use_aggregate =
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
        (msg_size < (64 * 1024));
#else
        false;
#endif

    AscendC::PipeBarrier<PIPE_ALL>();

    if (metric == static_cast<int>(perftest::PERF_METRIC_LAT)) {
        // Latency: time loop_count nbi submits in a single window, quiet outside.
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
        if (use_aggregate && warmup > 1) {
            aclshmemx_submit_state_t submit_state;
            aclshmemx_defer_t defer_action(submit_state);
            aclshmemx_submit_t submit_action(submit_state);
            for (int i = 0; i < warmup - 1; ++i) {
                aclshmemx_roce_put_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, defer_action);
            }
            aclshmemx_roce_put_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, submit_action);
        } else
#endif
        {
            for (int i = 0; i < warmup; ++i) {
                aclshmemx_roce_put_nbi(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id);
            }
        }
        aclshmemx_roce_quiet(peer_pe, ub_ptr, sync_id);
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t api_time_start = AscendC::GetSystemCycle();
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
        if (use_aggregate && loop_test > 1) {
            aclshmemx_submit_state_t submit_state;
            aclshmemx_defer_t defer_action(submit_state);
            aclshmemx_submit_t submit_action(submit_state);
            for (int i = 0; i < loop_test - 1; ++i) {
                aclshmemx_roce_put_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, defer_action);
            }
            aclshmemx_roce_put_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, submit_action);
        } else
#endif
        {
            for (int i = 0; i < loop_test; ++i) {
                aclshmemx_roce_put_nbi(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t api_time_end = AscendC::GetSystemCycle();
        aclshmemx_roce_quiet(peer_pe, ub_ptr, sync_id);
        if (timing_out != nullptr) {
            int64_t api_total_time = api_time_end - api_time_start;
            if (pe == 0) {
                timing_out[0] = api_total_time;
            } else {
                timing_out[1] = api_total_time;
            }
            dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(timing_out), sizeof(uint64_t) * 2);
            if (is_bidir) {
                aclshmemx_roce_barrier_all();
                __gm__ T* slot = reinterpret_cast<__gm__ T*>(&timing_out[pe]);
                aclshmemx_roce_put_nbi(slot, slot, ub_ptr, sizeof(int64_t) / sizeof(T), peer_pe, sync_id);
                aclshmemx_roce_barrier_all();
            }
        }

    } else {
        // Bandwidth: submit NBIs in groups of `batch_size`, quiet at each group boundary.
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
        if (use_aggregate && warmup > 1) {
            aclshmemx_submit_state_t submit_state;
            aclshmemx_defer_t defer_action(submit_state);
            aclshmemx_submit_t submit_action(submit_state);
            for (int i = 0; i < warmup - 1; ++i) {
                aclshmemx_roce_put_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, defer_action);
            }
            aclshmemx_roce_put_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, submit_action);
        } else
#endif
        {
            for (int i = 0; i < warmup; ++i) {
                aclshmemx_roce_put_nbi(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id);
            }
        }
        aclshmemx_roce_quiet(peer_pe, ub_ptr, sync_id);
        int full_groups = loop_test / batch_size;
        int remainder = loop_test - full_groups * batch_size;
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t waiting_time_start = AscendC::GetSystemCycle();
        for (int g = 0; g < full_groups; ++g) {
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
            if (use_aggregate && batch_size > 1) {
                aclshmemx_submit_state_t submit_state;
                aclshmemx_defer_t defer_action(submit_state);
                aclshmemx_submit_t submit_action(submit_state);
                for (int j = 0; j < batch_size - 1; ++j) {
                    aclshmemx_roce_put_nbi(
                        dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, defer_action);
                }
                aclshmemx_roce_put_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, submit_action);
            } else
#endif
            {
                for (int j = 0; j < batch_size; ++j) {
                    aclshmemx_roce_put_nbi(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id);
                }
            }
            aclshmemx_roce_quiet(peer_pe, ub_ptr, sync_id);
        }
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
        if (use_aggregate && remainder > 1) {
            aclshmemx_submit_state_t submit_state;
            aclshmemx_defer_t defer_action(submit_state);
            aclshmemx_submit_t submit_action(submit_state);
            for (int j = 0; j < remainder - 1; ++j) {
                aclshmemx_roce_put_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, defer_action);
            }
            aclshmemx_roce_put_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, submit_action);
        } else
#endif
        {
            for (int j = 0; j < remainder; ++j) {
                aclshmemx_roce_put_nbi(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id);
            }
        }
        if (remainder > 0) {
            aclshmemx_roce_quiet(peer_pe, ub_ptr, sync_id);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t waiting_time_end = AscendC::GetSystemCycle();

        // Output: PE0 -> timing_out[0], PE1 -> timing_out[1]
        if (timing_out != nullptr) {
            int64_t waiting_total_time = waiting_time_end - waiting_time_start;
            if (pe == 0) {
                timing_out[0] = waiting_total_time;
            } else {
                timing_out[1] = waiting_total_time;
            }
            dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(timing_out), sizeof(uint64_t) * 2);
            if (is_bidir) {
                aclshmemx_roce_barrier_all();
                __gm__ T* slot = reinterpret_cast<__gm__ T*>(&timing_out[pe]);
                aclshmemx_roce_put_nbi(slot, slot, ub_ptr, sizeof(int64_t) / sizeof(T), peer_pe, sync_id);
                aclshmemx_roce_barrier_all();
            }
        }
    }
    aclshmemx_roce_barrier_all();
}

template <typename T>
__aicore__ inline void rdma_perf_test_get_impl(
    uint64_t fftsAddr, GM_ADDR dst_gva, GM_ADDR src_gva, int elements, perftest::rdma_mode_t test_mode, int ub_size_b,
    int loop_count, int metric, int batch, uint32_t sync_id, GM_ADDR timing_out_gva)
{
    util_set_ffts_config(fftsAddr);
    int64_t pe = aclshmem_my_pe();
    int peer_pe = (pe + 1) % aclshmem_n_pes();
    __gm__ int64_t* timing_out = reinterpret_cast<__gm__ int64_t*>(timing_out_gva);

    bool is_bidir = (test_mode == perftest::TEST_MODE_RDMA_BI_GET);

    // Ensure both PEs are synchronized before any RDMA operations begin
    aclshmemx_roce_barrier_all();

    if (!is_bidir && pe != 0) {
        if (timing_out != nullptr) {
            timing_out[1] = 0;
        }
        aclshmemx_roce_barrier_all();
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, ub_size_b);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(ub_size_b, 0);
    __ubuf__ T* ub_ptr = reinterpret_cast<__ubuf__ T*>(ubLocal.GetPhyAddr());

    __gm__ T* dst_gm = reinterpret_cast<__gm__ T*>(dst_gva);
    __gm__ T* src_gm = reinterpret_cast<__gm__ T*>(src_gva);

    int warmup = perftest::PERFTEST_WARMUP_ITERS;
    int loop_test = loop_count;
    int batch_size = (batch <= 0 || batch > loop_test) ? loop_test : batch;

    const uint64_t msg_size = static_cast<uint64_t>(elements) * sizeof(T);
    const bool use_aggregate =
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
        (msg_size < (64 * 1024));
#else
        false;
#endif

    AscendC::PipeBarrier<PIPE_ALL>();

    if (metric == static_cast<int>(perftest::PERF_METRIC_LAT)) {
        // Latency: time loop_count get_nbi submits in a single window, quiet outside.
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
        if (use_aggregate && warmup > 1) {
            aclshmemx_submit_state_t submit_state;
            aclshmemx_defer_t defer_action(submit_state);
            aclshmemx_submit_t submit_action(submit_state);
            for (int i = 0; i < warmup - 1; ++i) {
                aclshmemx_roce_get_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, defer_action);
            }
            aclshmemx_roce_get_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, submit_action);
        } else
#endif
        {
            for (int i = 0; i < warmup; ++i) {
                aclshmemx_roce_get_nbi(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id);
            }
        }
        aclshmemx_roce_quiet(peer_pe, ub_ptr, sync_id);
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t api_time_start = AscendC::GetSystemCycle();
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
        if (use_aggregate && loop_test > 1) {
            aclshmemx_submit_state_t submit_state;
            aclshmemx_defer_t defer_action(submit_state);
            aclshmemx_submit_t submit_action(submit_state);
            for (int i = 0; i < loop_test - 1; ++i) {
                aclshmemx_roce_get_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, defer_action);
            }
            aclshmemx_roce_get_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, submit_action);
        } else
#endif
        {
            for (int i = 0; i < loop_test; ++i) {
                aclshmemx_roce_get_nbi(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t api_time_end = AscendC::GetSystemCycle();
        aclshmemx_roce_quiet(peer_pe, ub_ptr, sync_id);
        // dcci removed — relying on host-side 64B-aligned allocation instead
        if (timing_out != nullptr) {
            int64_t api_total_time = api_time_end - api_time_start;
            if (pe == 0) {
                timing_out[0] = api_total_time;
            } else {
                timing_out[1] = api_total_time;
            }
            dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(timing_out), sizeof(uint64_t) * 2);
            if (is_bidir) {
                aclshmemx_roce_barrier_all();
                __gm__ T* slot = reinterpret_cast<__gm__ T*>(&timing_out[pe]);
                aclshmemx_roce_put_nbi(slot, slot, ub_ptr, sizeof(int64_t) / sizeof(T), peer_pe, sync_id);
                aclshmemx_roce_barrier_all();
            }
        }

    } else {
        // Bandwidth: submit NBIs in groups of `batch_size`, quiet at each group boundary.
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
        if (use_aggregate && warmup > 1) {
            aclshmemx_submit_state_t submit_state;
            aclshmemx_defer_t defer_action(submit_state);
            aclshmemx_submit_t submit_action(submit_state);
            for (int i = 0; i < warmup - 1; ++i) {
                aclshmemx_roce_get_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, defer_action);
            }
            aclshmemx_roce_get_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, submit_action);
        } else
#endif
        {
            for (int i = 0; i < warmup; ++i) {
                aclshmemx_roce_get_nbi(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id);
            }
        }
        aclshmemx_roce_quiet(peer_pe, ub_ptr, sync_id);
        int full_groups = loop_test / batch_size;
        int remainder = loop_test - full_groups * batch_size;
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t get_time_start = AscendC::GetSystemCycle();
        for (int g = 0; g < full_groups; ++g) {
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
            if (use_aggregate && batch_size > 1) {
                aclshmemx_submit_state_t submit_state;
                aclshmemx_defer_t defer_action(submit_state);
                aclshmemx_submit_t submit_action(submit_state);
                for (int j = 0; j < batch_size - 1; ++j) {
                    aclshmemx_roce_get_nbi(
                        dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, defer_action);
                }
                aclshmemx_roce_get_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, submit_action);
            } else
#endif
            {
                for (int j = 0; j < batch_size; ++j) {
                    aclshmemx_roce_get_nbi(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id);
                }
            }
            aclshmemx_roce_quiet(peer_pe, ub_ptr, sync_id);
        }
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
        if (use_aggregate && remainder > 1) {
            aclshmemx_submit_state_t submit_state;
            aclshmemx_defer_t defer_action(submit_state);
            aclshmemx_submit_t submit_action(submit_state);
            for (int j = 0; j < remainder - 1; ++j) {
                aclshmemx_roce_get_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, defer_action);
            }
            aclshmemx_roce_get_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id, submit_action);
        } else
#endif
        {
            for (int j = 0; j < remainder; ++j) {
                aclshmemx_roce_get_nbi(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, sync_id);
            }
        }
        if (remainder > 0) {
            aclshmemx_roce_quiet(peer_pe, ub_ptr, sync_id);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t get_time_end = AscendC::GetSystemCycle();

        // dcci removed — relying on host-side 64B-aligned allocation instead
        // Output: PE0 -> timing_out[0], PE1 -> timing_out[1]
        if (timing_out != nullptr) {
            int64_t get_total_time = get_time_end - get_time_start;
            if (pe == 0) {
                timing_out[0] = get_total_time;
            } else {
                timing_out[1] = get_total_time;
            }
            dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(timing_out), sizeof(uint64_t) * 2);
            if (is_bidir) {
                aclshmemx_roce_barrier_all();
                __gm__ T* slot = reinterpret_cast<__gm__ T*>(&timing_out[pe]);
                aclshmemx_roce_put_nbi(slot, slot, ub_ptr, sizeof(int64_t) / sizeof(T), peer_pe, sync_id);
                aclshmemx_roce_barrier_all();
            }
        }
    }
    aclshmemx_roce_barrier_all();
}

#define DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(type_name, cpp_type)                                               \
    extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void rdma_perf_test_##type_name##_put(   \
        uint64_t fftsAddr, GM_ADDR dst_gva, GM_ADDR src_gva, int elements, perftest::rdma_mode_t test_mode, \
        int ub_size_b, int loop_count, int metric, int batch, uint32_t sync_id, GM_ADDR timing_out_gva)     \
    {                                                                                                       \
        rdma_perf_test_put_impl<cpp_type>(                                                                  \
            fftsAddr, dst_gva, src_gva, elements, test_mode, ub_size_b, loop_count, metric, batch, sync_id, \
            timing_out_gva);                                                                                \
    }                                                                                                       \
    extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void rdma_perf_test_##type_name##_get(   \
        uint64_t fftsAddr, GM_ADDR dst_gva, GM_ADDR src_gva, int elements, perftest::rdma_mode_t test_mode, \
        int ub_size_b, int loop_count, int metric, int batch, uint32_t sync_id, GM_ADDR timing_out_gva)     \
    {                                                                                                       \
        rdma_perf_test_get_impl<cpp_type>(                                                                  \
            fftsAddr, dst_gva, src_gva, elements, test_mode, ub_size_b, loop_count, metric, batch, sync_id, \
            timing_out_gva);                                                                                \
    }

DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(float, float)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(int8, int8_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(int16, int16_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(int32, int32_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(int64, int64_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(uint8, uint8_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(uint16, uint16_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(uint32, uint32_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(uint64, uint64_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(char, char)

#define DISPATCH_RDMA_PERF_PUT(type_name, cpp_type)                   \
    rdma_perf_test_##type_name##_put<<<block_dim, nullptr, stream>>>( \
        fftsAddr, dst_gva, src_gva, elements, t_mode, ub_size_b, loop_count, metric, batch, sync_id, timing_out_gva)

#define DISPATCH_RDMA_PERF_GET(type_name, cpp_type)                   \
    rdma_perf_test_##type_name##_get<<<block_dim, nullptr, stream>>>( \
        fftsAddr, dst_gva, src_gva, elements, t_mode, ub_size_b, loop_count, metric, batch, sync_id, timing_out_gva)

#define DISPATCH_RDMA_PERF_FOR_ALL_TYPES(MACRO) \
    switch (d_type) {                           \
        case perftest::DATA_TYPE_FLOAT:         \
            MACRO(float, float);                \
            break;                              \
        case perftest::DATA_TYPE_INT8:          \
            MACRO(int8, int8_t);                \
            break;                              \
        case perftest::DATA_TYPE_INT16:         \
            MACRO(int16, int16_t);              \
            break;                              \
        case perftest::DATA_TYPE_INT32:         \
            MACRO(int32, int32_t);              \
            break;                              \
        case perftest::DATA_TYPE_INT64:         \
            MACRO(int64, int64_t);              \
            break;                              \
        case perftest::DATA_TYPE_UINT8:         \
            MACRO(uint8, uint8_t);              \
            break;                              \
        case perftest::DATA_TYPE_UINT16:        \
            MACRO(uint16, uint16_t);            \
            break;                              \
        case perftest::DATA_TYPE_UINT32:        \
            MACRO(uint32, uint32_t);            \
            break;                              \
        case perftest::DATA_TYPE_UINT64:        \
            MACRO(uint64, uint64_t);            \
            break;                              \
        case perftest::DATA_TYPE_CHAR:          \
            MACRO(char, char);                  \
            break;                              \
        default:                                \
            MACRO(float, float);                \
            break;                              \
    }

extern "C" void launch_rdma_perf_kernel(
    uint32_t block_dim, void* stream, uint64_t fftsAddr, uint8_t* dst_gva, uint8_t* src_gva, int elements,
    int test_mode, int data_type, int ub_size_b, int loop_count, int metric, int batch, int sync_id,
    uint8_t* timing_out_gva)
{
    perftest::rdma_mode_t t_mode = static_cast<perftest::rdma_mode_t>(test_mode);
    perftest::perf_data_type_t d_type = static_cast<perftest::perf_data_type_t>(data_type);

    switch (t_mode) {
        case perftest::TEST_MODE_RDMA_PUT:
        case perftest::TEST_MODE_RDMA_BI_PUT:
            DISPATCH_RDMA_PERF_FOR_ALL_TYPES(DISPATCH_RDMA_PERF_PUT);
            break;
        case perftest::TEST_MODE_RDMA_GET:
        case perftest::TEST_MODE_RDMA_BI_GET:
            DISPATCH_RDMA_PERF_FOR_ALL_TYPES(DISPATCH_RDMA_PERF_GET);
            break;
        default:
            break;
    }
}

#endif // _RDMAPERF_KERNEL_RDMA_PERFTEST_
