/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _UDMAPERF_KERNEL_UDMA_PERFTEST_
#define _UDMAPERF_KERNEL_UDMA_PERFTEST_

#include <cstring>
#include "kernel_operator.h"
#include "shmem.h"
#include "utils/prof/shmemi_prof.h"
#include "perftest_common_types.h"

template <typename T>
__aicore__ inline void udma_perf_test_put_impl(GM_ADDR dst_gva, GM_ADDR src_gva, int elements, int32_t frame_id,
                                                perftest::udma_mode_t test_mode, int ub_size_kb,
                                                int64_t prof_pe_val, int loop_count, int metric, int batch)
{
    int64_t pe = aclshmem_my_pe();
    int peer_pe = (pe + 1) % aclshmem_n_pes();

    bool is_unilateral = (test_mode == perftest::TEST_MODE_UDMA_PUT);
    if (is_unilateral && pe != prof_pe_val) {
        aclshmemx_barrier_all_vec();
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, ub_size_kb * 1024);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(ub_size_kb * 1024, 0);
    __ubuf__ T *ub_ptr = reinterpret_cast<__ubuf__ T *>(ubLocal.GetPhyAddr());

    __gm__ T *dst_gm = reinterpret_cast<__gm__ T *>(dst_gva);
    __gm__ T *src_gm = reinterpret_cast<__gm__ T *>(src_gva);

    int warmup = perftest::PERFTEST_WARMUP_ITERS;
    int loop_test = loop_count;
    int batch_size = (batch <= 0 || batch > loop_test) ? loop_test : batch;
    AscendC::PipeBarrier<PIPE_ALL>();

    if (metric == static_cast<int>(perftest::PERF_METRIC_LAT)) {
        // Latency: time loop_count put_nbi submits in a single window, quiet outside.
        // SHMEMI_PROF_START/END has its own pipe_barrier overhead, so we keep ONE pair
        // around the whole batch and divide by loop_count instead of timing each iter.
        for (int i = 0; i < warmup; ++i) {
            aclshmemx_udma_put_nbi(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe);
        }
        aclshmemx_udma_quiet(peer_pe);
        SHMEMI_PROF_START(frame_id);
        for (int i = 0; i < loop_test; ++i) {
            aclshmemx_udma_put_nbi(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe);
        }
        SHMEMI_PROF_END(frame_id);
        aclshmemx_udma_quiet(peer_pe);
    } else {
        // Bandwidth: submit NBIs in groups of `batch_size`, quiet at each group boundary.
        // Nested loop avoids a per-iter `%` inside the timed window:
        //   - batch_size == loop_test (default) -> 1 group, 1 trailing quiet (full async);
        //   - batch_size == 1 -> loop_test groups of 1 (synchronous);
        //   - mid sizes split into full_groups + remainder, with one extra quiet for the tail.
        for (int i = 0; i < warmup; ++i) {
            aclshmemx_udma_put_nbi(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe);
        }
        aclshmemx_udma_quiet(peer_pe);
        int full_groups = loop_test / batch_size;
        int remainder = loop_test - full_groups * batch_size;
        SHMEMI_PROF_START(frame_id);
        for (int g = 0; g < full_groups; ++g) {
            for (int j = 0; j < batch_size; ++j) {
                aclshmemx_udma_put_nbi(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe);
            }
            aclshmemx_udma_quiet(peer_pe);
        }
        for (int j = 0; j < remainder; ++j) {
            aclshmemx_udma_put_nbi(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe);
        }
        if (remainder > 0) {
            aclshmemx_udma_quiet(peer_pe);
        }
        SHMEMI_PROF_END(frame_id);
    }
    aclshmemx_barrier_all_vec();
}

template <typename T>
__aicore__ inline void udma_perf_test_get_impl(GM_ADDR dst_gva, GM_ADDR src_gva, int elements, int32_t frame_id,
                                                perftest::udma_mode_t test_mode, int ub_size_kb,
                                                int64_t prof_pe_val, int loop_count, int batch)
{
    int64_t pe = aclshmem_my_pe();
    int peer_pe = (pe + 1) % aclshmem_n_pes();

    bool is_unilateral = (test_mode == perftest::TEST_MODE_UDMA_GET);
    if (is_unilateral && pe != prof_pe_val) {
        aclshmemx_barrier_all_vec();
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, ub_size_kb * 1024);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(ub_size_kb * 1024, 0);
    __ubuf__ T *ub_ptr = reinterpret_cast<__ubuf__ T *>(ubLocal.GetPhyAddr());

    __gm__ T *dst_gm = reinterpret_cast<__gm__ T *>(dst_gva);
    __gm__ T *src_gm = reinterpret_cast<__gm__ T *>(src_gva);

    int warmup = perftest::PERFTEST_WARMUP_ITERS;
    int loop_test = loop_count;
    int batch_size = (batch <= 0 || batch > loop_test) ? loop_test : batch;
    AscendC::PipeBarrier<PIPE_ALL>();
    for (int i = 0; i < warmup; ++i) {
        aclshmemx_udma_get_nbi(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe);
    }
    aclshmemx_udma_quiet(peer_pe);
    int full_groups = loop_test / batch_size;
    int remainder = loop_test - full_groups * batch_size;
    SHMEMI_PROF_START(frame_id);
    for (int g = 0; g < full_groups; ++g) {
        for (int j = 0; j < batch_size; ++j) {
            aclshmemx_udma_get_nbi(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe);
        }
        aclshmemx_udma_quiet(peer_pe);
    }
    for (int j = 0; j < remainder; ++j) {
        aclshmemx_udma_get_nbi(dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe);
    }
    if (remainder > 0) {
        aclshmemx_udma_quiet(peer_pe);
    }
    SHMEMI_PROF_END(frame_id);
    aclshmemx_barrier_all_vec();
}

template <typename T>
__aicore__ inline void udma_perf_test_put_signal_impl(GM_ADDR dst_gva, GM_ADDR src_gva, GM_ADDR sig_gva,
                                                      int elements, int32_t frame_id,
                                                      int64_t prof_pe_val, int loop_count, uint64_t signal_base,
                                                      int batch)
{
    int64_t pe = aclshmem_my_pe();
    int peer_pe = (pe + 1) % aclshmem_n_pes();

    if (pe != prof_pe_val) {
        aclshmemx_barrier_all_vec();
        return;
    }

    __gm__ T *dst_gm = reinterpret_cast<__gm__ T *>(dst_gva);
    __gm__ T *src_gm = reinterpret_cast<__gm__ T *>(src_gva);
    __gm__ uint64_t *sig_addr = reinterpret_cast<__gm__ uint64_t *>(sig_gva) + pe;

    int warmup = perftest::PERFTEST_WARMUP_ITERS;
    int loop_test = loop_count;
    int batch_size = (batch <= 0 || batch > loop_test) ? loop_test : batch;
    AscendC::PipeBarrier<PIPE_ALL>();
    for (int i = 0; i < warmup; ++i) {
        aclshmemx_udma_put_signal_nbi(dst_gm, src_gm, static_cast<uint32_t>(elements), sig_addr,
                                       signal_base + static_cast<uint64_t>(i), peer_pe);
    }
    aclshmemx_udma_quiet(peer_pe);
    int full_groups = loop_test / batch_size;
    int remainder = loop_test - full_groups * batch_size;
    uint64_t sig_value = signal_base + static_cast<uint64_t>(warmup);
    SHMEMI_PROF_START(frame_id);
    for (int g = 0; g < full_groups; ++g) {
        for (int j = 0; j < batch_size; ++j) {
            aclshmemx_udma_put_signal_nbi(dst_gm, src_gm, static_cast<uint32_t>(elements), sig_addr,
                                           sig_value++, peer_pe);
        }
        aclshmemx_udma_quiet(peer_pe);
    }
    for (int j = 0; j < remainder; ++j) {
        aclshmemx_udma_put_signal_nbi(dst_gm, src_gm, static_cast<uint32_t>(elements), sig_addr,
                                       sig_value++, peer_pe);
    }
    if (remainder > 0) {
        aclshmemx_udma_quiet(peer_pe);
    }
    SHMEMI_PROF_END(frame_id);
    aclshmemx_barrier_all_vec();
}

#define DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(type_name, cpp_type) \
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void udma_perf_test_##type_name##_put( \
    GM_ADDR dst_gva, GM_ADDR src_gva, int elements, int32_t frame_id, perftest::udma_mode_t test_mode, \
    int ub_size_kb, int64_t prof_pe_val, int loop_count, int metric, int batch) \
{ \
    udma_perf_test_put_impl<cpp_type>(dst_gva, src_gva, elements, frame_id, test_mode, ub_size_kb, prof_pe_val, loop_count, metric, batch); \
} \
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void udma_perf_test_##type_name##_get( \
    GM_ADDR dst_gva, GM_ADDR src_gva, int elements, int32_t frame_id, perftest::udma_mode_t test_mode, \
    int ub_size_kb, int64_t prof_pe_val, int loop_count, int batch) \
{ \
    udma_perf_test_get_impl<cpp_type>(dst_gva, src_gva, elements, frame_id, test_mode, ub_size_kb, prof_pe_val, loop_count, batch); \
} \
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void udma_perf_test_##type_name##_put_signal( \
    GM_ADDR dst_gva, GM_ADDR src_gva, GM_ADDR sig_gva, int elements, int32_t frame_id, \
    int64_t prof_pe_val, int loop_count, uint64_t signal_base, int batch) \
{ \
    udma_perf_test_put_signal_impl<cpp_type>(dst_gva, src_gva, sig_gva, elements, frame_id, prof_pe_val, loop_count, signal_base, batch); \
}

DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(float, float)
DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(int8, int8_t)
DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(int16, int16_t)
DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(int32, int32_t)
DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(int64, int64_t)
DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(uint8, uint8_t)
DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(uint16, uint16_t)
DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(uint32, uint32_t)
DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(uint64, uint64_t)
DEFINE_UDMA_PERF_KERNEL_FOR_TYPE(char, char)

#define DISPATCH_UDMA_PERF_PUT(type_name, cpp_type) \
    udma_perf_test_##type_name##_put<<<block_dim, nullptr, stream>>>( \
        dst_gva, src_gva, elements, frame_id, t_mode, ub_size_kb, prof_pe_val, loop_count, metric, batch)

#define DISPATCH_UDMA_PERF_GET(type_name, cpp_type) \
    udma_perf_test_##type_name##_get<<<block_dim, nullptr, stream>>>( \
        dst_gva, src_gva, elements, frame_id, t_mode, ub_size_kb, prof_pe_val, loop_count, batch)

#define DISPATCH_UDMA_PERF_PUT_SIGNAL(type_name, cpp_type) \
    udma_perf_test_##type_name##_put_signal<<<block_dim, nullptr, stream>>>( \
        dst_gva, src_gva, sig_gva, elements, frame_id, prof_pe_val, loop_count, signal_base, batch)

#define DISPATCH_UDMA_PERF_FOR_ALL_TYPES(MACRO) \
    switch (d_type) { \
        case perftest::DATA_TYPE_FLOAT:  MACRO(float, float);   break; \
        case perftest::DATA_TYPE_INT8:   MACRO(int8, int8_t);   break; \
        case perftest::DATA_TYPE_INT16:  MACRO(int16, int16_t); break; \
        case perftest::DATA_TYPE_INT32:  MACRO(int32, int32_t); break; \
        case perftest::DATA_TYPE_INT64:  MACRO(int64, int64_t); break; \
        case perftest::DATA_TYPE_UINT8:  MACRO(uint8, uint8_t); break; \
        case perftest::DATA_TYPE_UINT16: MACRO(uint16, uint16_t); break; \
        case perftest::DATA_TYPE_UINT32: MACRO(uint32, uint32_t); break; \
        case perftest::DATA_TYPE_UINT64: MACRO(uint64, uint64_t); break; \
        case perftest::DATA_TYPE_CHAR:   MACRO(char, char);     break; \
        default:                          MACRO(float, float);   break; \
    }

extern "C" void launch_udma_perf_kernel(uint32_t block_dim, void *stream, uint8_t *dst_gva, uint8_t *src_gva,
                                         uint8_t *sig_gva, int elements, int32_t frame_id, int test_mode,
                                         int data_type, int ub_size_kb, int64_t prof_pe_val, int loop_count,
                                         uint64_t signal_base, int metric, int batch)
{
    perftest::udma_mode_t t_mode = static_cast<perftest::udma_mode_t>(test_mode);
    perftest::perf_data_type_t d_type = static_cast<perftest::perf_data_type_t>(data_type);

    switch (t_mode) {
        case perftest::TEST_MODE_UDMA_PUT:
        case perftest::TEST_MODE_UDMA_BI_PUT:
            DISPATCH_UDMA_PERF_FOR_ALL_TYPES(DISPATCH_UDMA_PERF_PUT);
            break;
        case perftest::TEST_MODE_UDMA_GET:
        case perftest::TEST_MODE_UDMA_BI_GET:
            DISPATCH_UDMA_PERF_FOR_ALL_TYPES(DISPATCH_UDMA_PERF_GET);
            break;
        case perftest::TEST_MODE_UDMA_PUT_SIGNAL:
            DISPATCH_UDMA_PERF_FOR_ALL_TYPES(DISPATCH_UDMA_PERF_PUT_SIGNAL);
            break;
        default:
            break;
    }
}

#endif  // _UDMAPERF_KERNEL_UDMA_PERFTEST_
