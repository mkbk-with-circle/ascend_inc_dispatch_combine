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
#include "opdev/bfloat16.h"
#include "opdev/fp16_t.h"
#include "unittest/utils/func_type.h"
constexpr uint64_t MESSAGE_SIZE = 64;

#define ATOMIC_ADD_TEST_KERNEL(NAME, TYPE)                                                              \
    extern "C" __global__ __aicore__ void test_atomic_add_##NAME##_kernel(GM_ADDR gva, uint64_t config) \
    {                                                                                                   \
        util_set_ffts_config(config);                                                                   \
        int64_t pe = aclshmem_my_pe();                                                                  \
        int64_t pe_size = aclshmem_n_pes();                                                             \
        GM_ADDR dst_addr;                                                                               \
                                                                                                        \
        for (int64_t peer = 0; peer < pe_size; peer++) {                                                \
            if (peer == pe) {                                                                           \
                continue;                                                                               \
            }                                                                                           \
            dst_addr = gva + pe * MESSAGE_SIZE;                                                         \
            if ASCEND_IS_AIV {                                                                          \
                if (AscendC::GetSubBlockIdx() == 0) {                                                   \
                    aclshmem_##NAME##_atomic_add((__gm__ TYPE*)dst_addr, 1, peer);                      \
                }                                                                                       \
            }                                                                                           \
        }                                                                                               \
        aclshmem_barrier_all();                                                                         \
    }
ACLSHMEM_ATOMIC_ADD_FUNC_TYPE_KERNEL(ATOMIC_ADD_TEST_KERNEL);
ACLSHMEM_ATOMIC_ADD_910_FUNC_TYPE(ATOMIC_ADD_TEST_KERNEL);
ACLSHMEM_ATOMIC_ADD_EXT_FUNC_TYPE(ATOMIC_ADD_TEST_KERNEL);

#define ATOMIC_ADD_TEST(NAME, TYPE)                                                                   \
    void test_atomic_add_##NAME##_do(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config) \
    {                                                                                                 \
        test_atomic_add_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);                 \
    }
ACLSHMEM_ATOMIC_ADD_FUNC_TYPE_KERNEL(ATOMIC_ADD_TEST);
ACLSHMEM_ATOMIC_ADD_910_FUNC_TYPE(ATOMIC_ADD_TEST);
ACLSHMEM_ATOMIC_ADD_EXT_FUNC_TYPE(ATOMIC_ADD_TEST);

/**
 * @brief atomic_fetch_add kernel test
 *        Each pe fetches and adds to other pes' memory
 *        Verifies return value is in [0, block_dim) range
 */
#define ATOMIC_FETCH_ADD_TEST_KERNEL(NAME, TYPE)                                                   \
    extern "C" __global__ __aicore__ void test_atomic_fetch_add_##NAME##_kernel(                   \
        __gm__ TYPE* gva, __gm__ int* error_flag, uint32_t block_dim, uint64_t config)             \
    {                                                                                              \
        util_set_ffts_config(config);                                                              \
        int64_t pe = aclshmem_my_pe();                                                             \
        int64_t pe_size = aclshmem_n_pes();                                                        \
        __gm__ TYPE* dst_addr;                                                                     \
        TYPE old_val;                                                                              \
                                                                                                   \
        for (int64_t peer = 0; peer < pe_size; peer++) {                                           \
            if (peer == pe) {                                                                      \
                continue;                                                                          \
            }                                                                                      \
            dst_addr = gva + pe * MESSAGE_SIZE / sizeof(TYPE);                                     \
            if ASCEND_IS_AIV {                                                                     \
                if (AscendC::GetSubBlockIdx() == 0) {                                              \
                    old_val = aclshmem_##NAME##_atomic_fetch_add((__gm__ TYPE*)dst_addr, 1, peer); \
                    if (old_val < 0 || old_val >= static_cast<TYPE>(block_dim)) {                  \
                        *error_flag = 1;                                                           \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        aclshmem_barrier_all();                                                                    \
    }
ACLSHMEM_ATOMIC_ADD_950_FUNC_TYPE(ATOMIC_FETCH_ADD_TEST_KERNEL);

#define ATOMIC_FETCH_ADD_TEST(NAME, TYPE)                                                                          \
    void test_atomic_fetch_add_##NAME##_do(                                                                        \
        uint32_t block_dim, void* stream, TYPE* gva, int* error_flag, uint64_t config)                             \
    {                                                                                                              \
        test_atomic_fetch_add_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, error_flag, block_dim, config); \
    }
ACLSHMEM_ATOMIC_ADD_950_FUNC_TYPE(ATOMIC_FETCH_ADD_TEST);

/*****************************************************************************
 *                    atomic_inc test                                        *
 *****************************************************************************/
/**
 * @brief atomic_inc kernel test
 *        Each pe asks other pes to increment its own slot
 */
#define ATOMIC_INC_TEST_KERNEL(NAME, TYPE)                                                                   \
    extern "C" __global__ __aicore__ void test_atomic_inc_##NAME##_kernel(__gm__ TYPE* gva, uint64_t config) \
    {                                                                                                        \
        util_set_ffts_config(config);                                                                        \
        int64_t pe = aclshmem_my_pe();                                                                       \
        int64_t pe_size = aclshmem_n_pes();                                                                  \
        __gm__ TYPE* dst_addr;                                                                               \
                                                                                                             \
        for (int64_t peer = 0; peer < pe_size; peer++) {                                                     \
            if (peer == pe) {                                                                                \
                continue;                                                                                    \
            }                                                                                                \
            dst_addr = gva + pe * MESSAGE_SIZE / sizeof(TYPE);                                               \
            if ASCEND_IS_AIV {                                                                               \
                if (AscendC::GetSubBlockIdx() == 0) {                                                        \
                    aclshmem_##NAME##_atomic_inc((__gm__ TYPE*)dst_addr, peer);                              \
                }                                                                                            \
            }                                                                                                \
        }                                                                                                    \
        aclshmem_barrier_all();                                                                              \
    }
ACLSHMEM_ATOMIC_ADD_950_FUNC_TYPE(ATOMIC_INC_TEST_KERNEL);

#define ATOMIC_INC_TEST(NAME, TYPE)                                                                \
    void test_atomic_inc_##NAME##_do(uint32_t block_dim, void* stream, TYPE* gva, uint64_t config) \
    {                                                                                              \
        test_atomic_inc_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);              \
    }
ACLSHMEM_ATOMIC_ADD_950_FUNC_TYPE(ATOMIC_INC_TEST);

/*****************************************************************************
 *                    atomic_fetch_inc test                                  *
 *****************************************************************************/
/**
 * @brief atomic_fetch_inc kernel test
 *        Each pe asks other pes to increment its own slot and returns old value
 *        Verifies return value is in [0, block_dim) range
 */
#define ATOMIC_FETCH_INC_TEST_KERNEL(NAME, TYPE)                                                \
    extern "C" __global__ __aicore__ void test_atomic_fetch_inc_##NAME##_kernel(                \
        __gm__ TYPE* gva, __gm__ int* error_flag, uint32_t block_dim, uint64_t config)          \
    {                                                                                           \
        util_set_ffts_config(config);                                                           \
        int64_t pe = aclshmem_my_pe();                                                          \
        int64_t pe_size = aclshmem_n_pes();                                                     \
        __gm__ TYPE* dst_addr;                                                                  \
        TYPE old_val;                                                                           \
                                                                                                \
        for (int64_t peer = 0; peer < pe_size; peer++) {                                        \
            if (peer == pe) {                                                                   \
                continue;                                                                       \
            }                                                                                   \
            dst_addr = gva + pe * MESSAGE_SIZE / sizeof(TYPE);                                  \
            if ASCEND_IS_AIV {                                                                  \
                if (AscendC::GetSubBlockIdx() == 0) {                                           \
                    old_val = aclshmem_##NAME##_atomic_fetch_inc((__gm__ TYPE*)dst_addr, peer); \
                    if (old_val < 0 || old_val >= static_cast<TYPE>(block_dim)) {               \
                        *error_flag = 1;                                                        \
                    }                                                                           \
                }                                                                               \
            }                                                                                   \
        }                                                                                       \
        aclshmem_barrier_all();                                                                 \
    }
ACLSHMEM_ATOMIC_ADD_950_FUNC_TYPE(ATOMIC_FETCH_INC_TEST_KERNEL);

#define ATOMIC_FETCH_INC_TEST(NAME, TYPE)                                                                          \
    void test_atomic_fetch_inc_##NAME##_do(                                                                        \
        uint32_t block_dim, void* stream, TYPE* gva, int* error_flag, uint64_t config)                             \
    {                                                                                                              \
        test_atomic_fetch_inc_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, error_flag, block_dim, config); \
    }
ACLSHMEM_ATOMIC_ADD_950_FUNC_TYPE(ATOMIC_FETCH_INC_TEST);