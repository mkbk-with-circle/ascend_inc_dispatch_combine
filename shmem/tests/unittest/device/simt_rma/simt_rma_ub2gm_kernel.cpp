/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * @file simt_rma_ub2gm_kernel.cpp
 * @brief Device implementation of SIMT RMA ub2gm unit tests.
 */

#if defined(USE_SIMT)

#include "kernel_operator.h"
#include "shmem.h"
#include "utils/debug/asc_printf.h"

constexpr size_t MEM_BYTES = 4096;

template <typename T>
__simt_callee__ inline void local_gm2ub(__ubuf__ T* dst, __gm__ T* src, size_t elem_size)
{
    asc_syncthreads();
    size_t tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    size_t stride = blockDim.x * blockDim.y * blockDim.z;
    for (size_t i = tid; i < elem_size; i += stride) {
        dst[i] = src[i];
    }
    asc_syncthreads();
}

template <typename T>
__simt_callee__ inline void local_ub2gm(__gm__ T* dst, __ubuf__ T* src, size_t elem_size)
{
    asc_syncthreads();
    size_t tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    size_t stride = blockDim.x * blockDim.y * blockDim.z;
    for (size_t i = tid; i < elem_size; i += stride) {
        dst[i] = src[i];
    }
    asc_syncthreads();
}

// Warning: DO NOT REMOVE the printf function invocation, such as printf("Testing " #FUNCNAME "\n");
// Doing so may cause mysterious compiler errors.

#define SIMT_RMA_PUT_TYPED_TEST(FUNC) \
    FUNC(aclshmem_half_put, half) \
    FUNC(aclshmemx_half_put_warp, half) \
    FUNC(aclshmemx_half_put_block, half) \
    FUNC(aclshmem_half_put_nbi, half) \
    FUNC(aclshmemx_half_put_nbi_warp, half) \
    FUNC(aclshmemx_half_put_nbi_block, half) \
    FUNC(aclshmem_float_put, float) \
    FUNC(aclshmemx_float_put_warp, float) \
    FUNC(aclshmemx_float_put_block, float) \
    FUNC(aclshmem_float_put_nbi, float) \
    FUNC(aclshmemx_float_put_nbi_warp, float) \
    FUNC(aclshmemx_float_put_nbi_block, float) \
    FUNC(aclshmem_int8_put, int8_t) \
    FUNC(aclshmemx_int8_put_warp, int8_t) \
    FUNC(aclshmemx_int8_put_block, int8_t) \
    FUNC(aclshmem_int8_put_nbi, int8_t) \
    FUNC(aclshmemx_int8_put_nbi_warp, int8_t) \
    FUNC(aclshmemx_int8_put_nbi_block, int8_t) \
    FUNC(aclshmem_int16_put, int16_t) \
    FUNC(aclshmemx_int16_put_warp, int16_t) \
    FUNC(aclshmemx_int16_put_block, int16_t) \
    FUNC(aclshmem_int16_put_nbi, int16_t) \
    FUNC(aclshmemx_int16_put_nbi_warp, int16_t) \
    FUNC(aclshmemx_int16_put_nbi_block, int16_t) \
    FUNC(aclshmem_int32_put, int32_t) \
    FUNC(aclshmemx_int32_put_warp, int32_t) \
    FUNC(aclshmemx_int32_put_block, int32_t) \
    FUNC(aclshmem_int32_put_nbi, int32_t) \
    FUNC(aclshmemx_int32_put_nbi_warp, int32_t) \
    FUNC(aclshmemx_int32_put_nbi_block, int32_t) \
    FUNC(aclshmem_int64_put, int64_t) \
    FUNC(aclshmemx_int64_put_warp, int64_t) \
    FUNC(aclshmemx_int64_put_block, int64_t) \
    FUNC(aclshmem_int64_put_nbi, int64_t) \
    FUNC(aclshmemx_int64_put_nbi_warp, int64_t) \
    FUNC(aclshmemx_int64_put_nbi_block, int64_t) \
    FUNC(aclshmem_uint8_put, uint8_t) \
    FUNC(aclshmemx_uint8_put_warp, uint8_t) \
    FUNC(aclshmemx_uint8_put_block, uint8_t) \
    FUNC(aclshmem_uint8_put_nbi, uint8_t) \
    FUNC(aclshmemx_uint8_put_nbi_warp, uint8_t) \
    FUNC(aclshmemx_uint8_put_nbi_block, uint8_t) \
    FUNC(aclshmem_uint16_put, uint16_t) \
    FUNC(aclshmemx_uint16_put_warp, uint16_t) \
    FUNC(aclshmemx_uint16_put_block, uint16_t) \
    FUNC(aclshmem_uint16_put_nbi, uint16_t) \
    FUNC(aclshmemx_uint16_put_nbi_warp, uint16_t) \
    FUNC(aclshmemx_uint16_put_nbi_block, uint16_t) \
    FUNC(aclshmem_uint32_put, uint32_t) \
    FUNC(aclshmemx_uint32_put_warp, uint32_t) \
    FUNC(aclshmemx_uint32_put_block, uint32_t) \
    FUNC(aclshmem_uint32_put_nbi, uint32_t) \
    FUNC(aclshmemx_uint32_put_nbi_warp, uint32_t) \
    FUNC(aclshmemx_uint32_put_nbi_block, uint32_t) \
    FUNC(aclshmem_uint64_put, uint64_t) \
    FUNC(aclshmemx_uint64_put_warp, uint64_t) \
    FUNC(aclshmemx_uint64_put_block, uint64_t) \
    FUNC(aclshmem_uint64_put_nbi, uint64_t) \
    FUNC(aclshmemx_uint64_put_nbi_warp, uint64_t) \
    FUNC(aclshmemx_uint64_put_nbi_block, uint64_t) \
    FUNC(aclshmem_char_put, unsigned char) \
    FUNC(aclshmemx_char_put_warp, unsigned char) \
    FUNC(aclshmemx_char_put_block, unsigned char) \
    FUNC(aclshmem_char_put_nbi, unsigned char) \
    FUNC(aclshmemx_char_put_nbi_warp, unsigned char) \
    FUNC(aclshmemx_char_put_nbi_block, unsigned char) \
    FUNC(aclshmem_bfloat16_put, bfloat16_t) \
    FUNC(aclshmemx_bfloat16_put_warp, bfloat16_t) \
    FUNC(aclshmemx_bfloat16_put_block, bfloat16_t) \
    FUNC(aclshmem_bfloat16_put_nbi, bfloat16_t) \
    FUNC(aclshmemx_bfloat16_put_nbi_warp, bfloat16_t) \
    FUNC(aclshmemx_bfloat16_put_nbi_block, bfloat16_t)

#define SIMT_RMA_GET_TYPED_TEST(FUNC) \
    FUNC(aclshmem_half_get, half) \
    FUNC(aclshmemx_half_get_warp, half) \
    FUNC(aclshmemx_half_get_block, half) \
    FUNC(aclshmem_half_get_nbi, half) \
    FUNC(aclshmemx_half_get_nbi_warp, half) \
    FUNC(aclshmemx_half_get_nbi_block, half) \
    FUNC(aclshmem_float_get, float) \
    FUNC(aclshmemx_float_get_warp, float) \
    FUNC(aclshmemx_float_get_block, float) \
    FUNC(aclshmem_float_get_nbi, float) \
    FUNC(aclshmemx_float_get_nbi_warp, float) \
    FUNC(aclshmemx_float_get_nbi_block, float) \
    FUNC(aclshmem_int8_get, int8_t) \
    FUNC(aclshmemx_int8_get_warp, int8_t) \
    FUNC(aclshmemx_int8_get_block, int8_t) \
    FUNC(aclshmem_int8_get_nbi, int8_t) \
    FUNC(aclshmemx_int8_get_nbi_warp, int8_t) \
    FUNC(aclshmemx_int8_get_nbi_block, int8_t) \
    FUNC(aclshmem_int16_get, int16_t) \
    FUNC(aclshmemx_int16_get_warp, int16_t) \
    FUNC(aclshmemx_int16_get_block, int16_t) \
    FUNC(aclshmem_int16_get_nbi, int16_t) \
    FUNC(aclshmemx_int16_get_nbi_warp, int16_t) \
    FUNC(aclshmemx_int16_get_nbi_block, int16_t) \
    FUNC(aclshmem_int32_get, int32_t) \
    FUNC(aclshmemx_int32_get_warp, int32_t) \
    FUNC(aclshmemx_int32_get_block, int32_t) \
    FUNC(aclshmem_int32_get_nbi, int32_t) \
    FUNC(aclshmemx_int32_get_nbi_warp, int32_t) \
    FUNC(aclshmemx_int32_get_nbi_block, int32_t) \
    FUNC(aclshmem_int64_get, int64_t) \
    FUNC(aclshmemx_int64_get_warp, int64_t) \
    FUNC(aclshmemx_int64_get_block, int64_t) \
    FUNC(aclshmem_int64_get_nbi, int64_t) \
    FUNC(aclshmemx_int64_get_nbi_warp, int64_t) \
    FUNC(aclshmemx_int64_get_nbi_block, int64_t) \
    FUNC(aclshmem_uint8_get, uint8_t) \
    FUNC(aclshmemx_uint8_get_warp, uint8_t) \
    FUNC(aclshmemx_uint8_get_block, uint8_t) \
    FUNC(aclshmem_uint8_get_nbi, uint8_t) \
    FUNC(aclshmemx_uint8_get_nbi_warp, uint8_t) \
    FUNC(aclshmemx_uint8_get_nbi_block, uint8_t) \
    FUNC(aclshmem_uint16_get, uint16_t) \
    FUNC(aclshmemx_uint16_get_warp, uint16_t) \
    FUNC(aclshmemx_uint16_get_block, uint16_t) \
    FUNC(aclshmem_uint16_get_nbi, uint16_t) \
    FUNC(aclshmemx_uint16_get_nbi_warp, uint16_t) \
    FUNC(aclshmemx_uint16_get_nbi_block, uint16_t) \
    FUNC(aclshmem_uint32_get, uint32_t) \
    FUNC(aclshmemx_uint32_get_warp, uint32_t) \
    FUNC(aclshmemx_uint32_get_block, uint32_t) \
    FUNC(aclshmem_uint32_get_nbi, uint32_t) \
    FUNC(aclshmemx_uint32_get_nbi_warp, uint32_t) \
    FUNC(aclshmemx_uint32_get_nbi_block, uint32_t) \
    FUNC(aclshmem_uint64_get, uint64_t) \
    FUNC(aclshmemx_uint64_get_warp, uint64_t) \
    FUNC(aclshmemx_uint64_get_block, uint64_t) \
    FUNC(aclshmem_uint64_get_nbi, uint64_t) \
    FUNC(aclshmemx_uint64_get_nbi_warp, uint64_t) \
    FUNC(aclshmemx_uint64_get_nbi_block, uint64_t) \
    FUNC(aclshmem_char_get, unsigned char) \
    FUNC(aclshmemx_char_get_warp, unsigned char) \
    FUNC(aclshmemx_char_get_block, unsigned char) \
    FUNC(aclshmem_char_get_nbi, unsigned char) \
    FUNC(aclshmemx_char_get_nbi_warp, unsigned char) \
    FUNC(aclshmemx_char_get_nbi_block, unsigned char) \
    FUNC(aclshmem_bfloat16_get, bfloat16_t) \
    FUNC(aclshmemx_bfloat16_get_warp, bfloat16_t) \
    FUNC(aclshmemx_bfloat16_get_block, bfloat16_t) \
    FUNC(aclshmem_bfloat16_get_nbi, bfloat16_t) \
    FUNC(aclshmemx_bfloat16_get_nbi_warp, bfloat16_t) \
    FUNC(aclshmemx_bfloat16_get_nbi_block, bfloat16_t)

#define SIMT_RMA_PUT_RAW_TEST(FUNC) \
    FUNC(aclshmem_put8, int8_t) \
    FUNC(aclshmemx_put8_warp, int8_t) \
    FUNC(aclshmemx_put8_block, int8_t) \
    FUNC(aclshmem_put8_nbi, int8_t) \
    FUNC(aclshmemx_put8_nbi_warp, int8_t) \
    FUNC(aclshmemx_put8_nbi_block, int8_t) \
    FUNC(aclshmem_put16, int16_t) \
    FUNC(aclshmemx_put16_warp, int16_t) \
    FUNC(aclshmemx_put16_block, int16_t) \
    FUNC(aclshmem_put16_nbi, int16_t) \
    FUNC(aclshmemx_put16_nbi_warp, int16_t) \
    FUNC(aclshmemx_put16_nbi_block, int16_t) \
    FUNC(aclshmem_put32, int32_t) \
    FUNC(aclshmemx_put32_warp, int32_t) \
    FUNC(aclshmemx_put32_block, int32_t) \
    FUNC(aclshmem_put32_nbi, int32_t) \
    FUNC(aclshmemx_put32_nbi_warp, int32_t) \
    FUNC(aclshmemx_put32_nbi_block, int32_t) \
    FUNC(aclshmem_put64, int64_t) \
    FUNC(aclshmemx_put64_warp, int64_t) \
    FUNC(aclshmemx_put64_block, int64_t) \
    FUNC(aclshmem_put64_nbi, int64_t) \
    FUNC(aclshmemx_put64_nbi_warp, int64_t) \
    FUNC(aclshmemx_put64_nbi_block, int64_t) \
    FUNC(aclshmem_put128, int4) \
    FUNC(aclshmemx_put128_warp, int4) \
    FUNC(aclshmemx_put128_block, int4) \
    FUNC(aclshmem_put128_nbi, int4) \
    FUNC(aclshmemx_put128_nbi_warp, int4) \
    FUNC(aclshmemx_put128_nbi_block, int4) \
    FUNC(aclshmem_putmem, char) \
    FUNC(aclshmemx_putmem_warp, char) \
    FUNC(aclshmemx_putmem_block, char) \
    FUNC(aclshmem_putmem_nbi, char) \
    FUNC(aclshmemx_putmem_nbi_warp, char) \
    FUNC(aclshmemx_putmem_nbi_block, char)

#define SIMT_RMA_GET_RAW_TEST(FUNC) \
    FUNC(aclshmem_get8, int8_t) \
    FUNC(aclshmemx_get8_warp, int8_t) \
    FUNC(aclshmemx_get8_block, int8_t) \
    FUNC(aclshmem_get8_nbi, int8_t) \
    FUNC(aclshmemx_get8_nbi_warp, int8_t) \
    FUNC(aclshmemx_get8_nbi_block, int8_t) \
    FUNC(aclshmem_get16, int16_t) \
    FUNC(aclshmemx_get16_warp, int16_t) \
    FUNC(aclshmemx_get16_block, int16_t) \
    FUNC(aclshmem_get16_nbi, int16_t) \
    FUNC(aclshmemx_get16_nbi_warp, int16_t) \
    FUNC(aclshmemx_get16_nbi_block, int16_t) \
    FUNC(aclshmem_get32, int32_t) \
    FUNC(aclshmemx_get32_warp, int32_t) \
    FUNC(aclshmemx_get32_block, int32_t) \
    FUNC(aclshmem_get32_nbi, int32_t) \
    FUNC(aclshmemx_get32_nbi_warp, int32_t) \
    FUNC(aclshmemx_get32_nbi_block, int32_t) \
    FUNC(aclshmem_get64, int64_t) \
    FUNC(aclshmemx_get64_warp, int64_t) \
    FUNC(aclshmemx_get64_block, int64_t) \
    FUNC(aclshmem_get64_nbi, int64_t) \
    FUNC(aclshmemx_get64_nbi_warp, int64_t) \
    FUNC(aclshmemx_get64_nbi_block, int64_t) \
    FUNC(aclshmem_get128, int4) \
    FUNC(aclshmemx_get128_warp, int4) \
    FUNC(aclshmemx_get128_block, int4) \
    FUNC(aclshmem_get128_nbi, int4) \
    FUNC(aclshmemx_get128_nbi_warp, int4) \
    FUNC(aclshmemx_get128_nbi_block, int4) \
    FUNC(aclshmem_getmem, char) \
    FUNC(aclshmemx_getmem_warp, char) \
    FUNC(aclshmemx_getmem_block, char) \
    FUNC(aclshmem_getmem_nbi, char) \
    FUNC(aclshmemx_getmem_nbi_warp, char) \
    FUNC(aclshmemx_getmem_nbi_block, char)

#define SIMT_RMA_PUT_TYPED_KERNEL(FUNCNAME, TYPE)                                                                                              \
__simt_vf__ __launch_bounds__(1024) inline void simt_ub2gm_callee_call_##FUNCNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)  \
{                                                                                                                                              \
    __ubuf__ TYPE ub_buf[MEM_BYTES / sizeof(TYPE)];                                                                                            \
    if (blockIdx.x == 0 && threadIdx.x == 0) {                                                                                                  \
        printf("Testing " #FUNCNAME "\n");                                                                                                     \
    }                                                                                                                                          \
    asc_syncthreads();                                                                                                                         \
    local_gm2ub<TYPE>(ub_buf, (__gm__ TYPE*)src, elem_size);                                                                                   \
    simt::FUNCNAME((__gm__ TYPE*)dst, ub_buf, elem_size, pe);                                                                                  \
}                                                                                                                                              \
__global__ __vector__ void kernel_call_ub2gm_##FUNCNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)                            \
{                                                                                                                                              \
    asc_vf_call<simt_ub2gm_callee_call_##FUNCNAME>(dim3(1024), dst, src, elem_size, pe);                                                             \
}                                                                                                                                              \
void test_ub2gm_##FUNCNAME(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe)                                                   \
{                                                                                                                                              \
    kernel_call_ub2gm_##FUNCNAME<<<1, 0, stream>>>(dst, src, elem_size, pe);                                                                         \
}

#define SIMT_RMA_GET_TYPED_KERNEL(FUNCNAME, TYPE)                                                                                              \
__simt_vf__ __launch_bounds__(1024) inline void simt_ub2gm_callee_call_##FUNCNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)  \
{                                                                                                                                              \
    __ubuf__ TYPE ub_buf[MEM_BYTES / sizeof(TYPE)];                                                                                            \
    if (blockIdx.x == 0 && threadIdx.x == 0) {                                                                                                  \
        printf("Testing " #FUNCNAME "\n");                                                                                                     \
    }                                                                                                                                          \
    asc_syncthreads();                                                                                                                         \
    simt::FUNCNAME(ub_buf, (__gm__ TYPE*)src, elem_size, pe);                                                                                  \
    local_ub2gm<TYPE>((__gm__ TYPE*)dst, ub_buf, elem_size);                                                                                   \
}                                                                                                                                              \
__global__ __vector__ void kernel_call_ub2gm_##FUNCNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)                            \
{                                                                                                                                              \
    asc_vf_call<simt_ub2gm_callee_call_##FUNCNAME>(dim3(1024), dst, src, elem_size, pe);                                                             \
}                                                                                                                                              \
void test_ub2gm_##FUNCNAME(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe)                                                   \
{                                                                                                                                              \
    kernel_call_ub2gm_##FUNCNAME<<<1, 0, stream>>>(dst, src, elem_size, pe);                                                                         \
}

#define SIMT_RMA_PUT_RAW_KERNEL(FUNCNAME, TYPE)                                                                                                \
__simt_vf__ __launch_bounds__(1024) inline void simt_ub2gm_callee_call_##FUNCNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)  \
{                                                                                                                                              \
    __ubuf__ TYPE ub_buf[MEM_BYTES / sizeof(TYPE)];                                                                                            \
    if (blockIdx.x == 0 && threadIdx.x == 0) {                                                                                                  \
        printf("Testing " #FUNCNAME "\n");                                                                                                     \
    }                                                                                                                                          \
    asc_syncthreads();                                                                                                                         \
    local_gm2ub<TYPE>(ub_buf, (__gm__ TYPE*)src, elem_size);                                                                                   \
    simt::FUNCNAME((__gm__ void*)dst, (__ubuf__ void*)ub_buf, elem_size, pe);                                                                  \
}                                                                                                                                              \
__global__ __vector__ void kernel_call_ub2gm_##FUNCNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)                            \
{                                                                                                                                              \
    asc_vf_call<simt_ub2gm_callee_call_##FUNCNAME>(dim3(1024), dst, src, elem_size, pe);                                                             \
}                                                                                                                                              \
void test_ub2gm_##FUNCNAME(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe)                                                   \
{                                                                                                                                              \
    kernel_call_ub2gm_##FUNCNAME<<<1, 0, stream>>>(dst, src, elem_size, pe);                                                                         \
}

#define SIMT_RMA_GET_RAW_KERNEL(FUNCNAME, TYPE)                                                                                                \
__simt_vf__ __launch_bounds__(1024) inline void simt_ub2gm_callee_call_##FUNCNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)  \
{                                                                                                                                              \
    __ubuf__ TYPE ub_buf[MEM_BYTES / sizeof(TYPE)];                                                                                            \
    if (blockIdx.x == 0 && threadIdx.x == 0) {                                                                                                  \
        printf("Testing " #FUNCNAME "\n");                                                                                                     \
    }                                                                                                                                          \
    asc_syncthreads();                                                                                                                         \
    simt::FUNCNAME((__ubuf__ void*)ub_buf, (__gm__ void*)src, elem_size, pe);                                                                  \
    local_ub2gm<TYPE>((__gm__ TYPE*)dst, ub_buf, elem_size);                                                                                   \
}                                                                                                                                              \
__global__ __vector__ void kernel_call_ub2gm_##FUNCNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)                            \
{                                                                                                                                              \
    asc_vf_call<simt_ub2gm_callee_call_##FUNCNAME>(dim3(1024), dst, src, elem_size, pe);                                                             \
}                                                                                                                                              \
void test_ub2gm_##FUNCNAME(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe)                                                   \
{                                                                                                                                              \
    kernel_call_ub2gm_##FUNCNAME<<<1, 0, stream>>>(dst, src, elem_size, pe);                                                                         \
}

SIMT_RMA_PUT_TYPED_TEST(SIMT_RMA_PUT_TYPED_KERNEL)
SIMT_RMA_GET_TYPED_TEST(SIMT_RMA_GET_TYPED_KERNEL)
SIMT_RMA_PUT_RAW_TEST(SIMT_RMA_PUT_RAW_KERNEL)
SIMT_RMA_GET_RAW_TEST(SIMT_RMA_GET_RAW_KERNEL)

#undef SIMT_RMA_GET_RAW_KERNEL
#undef SIMT_RMA_PUT_RAW_KERNEL
#undef SIMT_RMA_GET_TYPED_KERNEL
#undef SIMT_RMA_PUT_TYPED_KERNEL

#define SIMT_RMA_PUT_SCHAR_TEST(FUNC) \
    FUNC(aclshmem_char_put, aclshmem_schar_put, signed char) \
    FUNC(aclshmemx_char_put_warp, aclshmemx_schar_put_warp, signed char) \
    FUNC(aclshmemx_char_put_block, aclshmemx_schar_put_block, signed char) \
    FUNC(aclshmem_char_put_nbi, aclshmem_schar_put_nbi, signed char) \
    FUNC(aclshmemx_char_put_nbi_warp, aclshmemx_schar_put_nbi_warp, signed char) \
    FUNC(aclshmemx_char_put_nbi_block, aclshmemx_schar_put_nbi_block, signed char)

#define SIMT_RMA_GET_SCHAR_TEST(FUNC) \
    FUNC(aclshmem_char_get, aclshmem_schar_get, signed char) \
    FUNC(aclshmemx_char_get_warp, aclshmemx_schar_get_warp, signed char) \
    FUNC(aclshmemx_char_get_block, aclshmemx_schar_get_block, signed char) \
    FUNC(aclshmem_char_get_nbi, aclshmem_schar_get_nbi, signed char) \
    FUNC(aclshmemx_char_get_nbi_warp, aclshmemx_schar_get_nbi_warp, signed char) \
    FUNC(aclshmemx_char_get_nbi_block, aclshmemx_schar_get_nbi_block, signed char)

#define SIMT_RMA_PUT_SCHAR_KERNEL(FUNCNAME, REPRNAME, TYPE)                                                                                     \
__simt_vf__ __launch_bounds__(1024) inline void simt_ub2gm_callee_call_##REPRNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)  \
{                                                                                                                                              \
    __ubuf__ TYPE ub_buf[MEM_BYTES / sizeof(TYPE)];                                                                                            \
    if (blockIdx.x == 0 && threadIdx.x == 0) {                                                                                                  \
        printf("Testing " #REPRNAME "\n");                                                                                                     \
    }                                                                                                                                          \
    asc_syncthreads();                                                                                                                         \
    local_gm2ub<TYPE>(ub_buf, (__gm__ TYPE*)src, elem_size);                                                                                   \
    simt::FUNCNAME((__gm__ TYPE*)dst, ub_buf, elem_size, pe);                                                                                  \
}                                                                                                                                              \
__global__ __vector__ void kernel_call_ub2gm_##REPRNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)                            \
{                                                                                                                                              \
    asc_vf_call<simt_ub2gm_callee_call_##REPRNAME>(dim3(1024), dst, src, elem_size, pe);                                                             \
}                                                                                                                                              \
void test_ub2gm_##REPRNAME(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe)                                                   \
{                                                                                                                                              \
    kernel_call_ub2gm_##REPRNAME<<<1, 0, stream>>>(dst, src, elem_size, pe);                                                                         \
}

#define SIMT_RMA_GET_SCHAR_KERNEL(FUNCNAME, REPRNAME, TYPE)                                                                                     \
__simt_vf__ __launch_bounds__(1024) inline void simt_ub2gm_callee_call_##REPRNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)  \
{                                                                                                                                              \
    __ubuf__ TYPE ub_buf[MEM_BYTES / sizeof(TYPE)];                                                                                            \
    if (blockIdx.x == 0 && threadIdx.x == 0) {                                                                                                  \
        printf("Testing " #REPRNAME "\n");                                                                                                     \
    }                                                                                                                                          \
    asc_syncthreads();                                                                                                                         \
    simt::FUNCNAME(ub_buf, (__gm__ TYPE*)src, elem_size, pe);                                                                                  \
    local_ub2gm<TYPE>((__gm__ TYPE*)dst, ub_buf, elem_size);                                                                                   \
}                                                                                                                                              \
__global__ __vector__ void kernel_call_ub2gm_##REPRNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)                            \
{                                                                                                                                              \
    asc_vf_call<simt_ub2gm_callee_call_##REPRNAME>(dim3(1024), dst, src, elem_size, pe);                                                             \
}                                                                                                                                              \
void test_ub2gm_##REPRNAME(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe)                                                   \
{                                                                                                                                              \
    kernel_call_ub2gm_##REPRNAME<<<1, 0, stream>>>(dst, src, elem_size, pe);                                                                         \
}

SIMT_RMA_PUT_SCHAR_TEST(SIMT_RMA_PUT_SCHAR_KERNEL)
SIMT_RMA_GET_SCHAR_TEST(SIMT_RMA_GET_SCHAR_KERNEL)

#undef SIMT_RMA_GET_SCHAR_KERNEL
#undef SIMT_RMA_PUT_SCHAR_KERNEL
#undef SIMT_RMA_GET_SCHAR_TEST
#undef SIMT_RMA_PUT_SCHAR_TEST
#undef SIMT_RMA_GET_RAW_TEST
#undef SIMT_RMA_PUT_RAW_TEST
#undef SIMT_RMA_GET_TYPED_TEST
#undef SIMT_RMA_PUT_TYPED_TEST

#endif // defined(USE_SIMT)
