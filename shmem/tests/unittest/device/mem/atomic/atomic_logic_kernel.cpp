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
#include "unittest/utils/func_type.h"
constexpr uint64_t MESSAGE_SIZE = 64;

/*****************************************************************************
 *                    atomic_and test                                         *
 *****************************************************************************/
/**
 * @brief atomic_and kernel test
 *        Each PE clears its own bit in other PEs' slots
 */
#define ATOMIC_AND_TEST_KERNEL(NAME, TYPE)                                                                   \
    extern "C" __global__ __aicore__ void test_atomic_and_##NAME##_kernel(__gm__ TYPE* gva, uint64_t config) \
    {                                                                                                        \
        util_set_ffts_config(config);                                                                        \
        int64_t pe = aclshmem_my_pe();                                                                       \
        int64_t pe_size = aclshmem_n_pes();                                                                  \
        __gm__ TYPE* dst_addr;                                                                               \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                           \
        uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;                                            \
        uint32_t sync_id = device_state->rdma_config.sync_id;                                                \
                                                                                                             \
        for (int64_t peer = 0; peer < pe_size; peer++) {                                                     \
            if (peer == pe) {                                                                                \
                continue;                                                                                    \
            }                                                                                                \
            dst_addr = gva + pe * MESSAGE_SIZE / sizeof(TYPE);                                               \
            if ASCEND_IS_AIV {                                                                               \
                if (AscendC::GetSubBlockIdx() == 0) {                                                        \
                    aclshmem_##NAME##_atomic_and((__gm__ TYPE*)dst_addr, ~(TYPE)(1LLU << pe), peer);         \
                    aclshmemx_roce_quiet(peer, reinterpret_cast<__ubuf__ char*>(copy_ub), sync_id);          \
                }                                                                                            \
            }                                                                                                \
        }                                                                                                    \
        aclshmem_barrier_all();                                                                              \
    }
ACLSHMEM_ATOMIC_LOGIC_FUNC_TYPE(ATOMIC_AND_TEST_KERNEL);

#define ATOMIC_AND_TEST(NAME, TYPE)                                                                \
    void test_atomic_and_##NAME##_do(uint32_t block_dim, void* stream, TYPE* gva, uint64_t config) \
    {                                                                                              \
        test_atomic_and_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);              \
    }
ACLSHMEM_ATOMIC_LOGIC_FUNC_TYPE(ATOMIC_AND_TEST);

/*****************************************************************************
 *                    atomic_fetch_and test                                   *
 *****************************************************************************/
/**
 * @brief atomic_fetch_and kernel test
 *        Each PE clears its own bit in other PEs' slots and returns old value
 *        Verifies return value has own bit still set (not yet cleared)
 */
#define ATOMIC_FETCH_AND_TEST_KERNEL(NAME, TYPE)                                                                     \
    extern "C" __global__ __aicore__ void test_atomic_fetch_and_##NAME##_kernel(                                     \
        __gm__ TYPE* gva, __gm__ int* error_flag, uint64_t pe_size, uint64_t config)                                 \
    {                                                                                                                \
        util_set_ffts_config(config);                                                                                \
        int64_t pe = aclshmem_my_pe();                                                                               \
        int64_t pe_size_val = aclshmem_n_pes();                                                                      \
        __gm__ TYPE* dst_addr;                                                                                       \
        TYPE old_val;                                                                                                \
                                                                                                                     \
        for (int64_t peer = 0; peer < pe_size_val; peer++) {                                                         \
            if (peer == pe) {                                                                                        \
                continue;                                                                                            \
            }                                                                                                        \
            dst_addr = gva + pe * MESSAGE_SIZE / sizeof(TYPE);                                                       \
            if ASCEND_IS_AIV {                                                                                       \
                if (AscendC::GetSubBlockIdx() == 0) {                                                                \
                    old_val = aclshmem_##NAME##_atomic_fetch_and((__gm__ TYPE*)dst_addr, ~(TYPE)(1LLU << pe), peer); \
                    /* Verify: return value should have own bit still set */                                         \
                    if ((old_val & ((TYPE)(1LLU << pe))) == 0) {                                                     \
                        *error_flag = 1;                                                                             \
                    }                                                                                                \
                }                                                                                                    \
            }                                                                                                        \
        }                                                                                                            \
        aclshmem_barrier_all();                                                                                      \
    }
ACLSHMEM_ATOMIC_LOGIC_FUNC_TYPE(ATOMIC_FETCH_AND_TEST_KERNEL);

#define ATOMIC_FETCH_AND_TEST(NAME, TYPE)                                                                        \
    void test_atomic_fetch_and_##NAME##_do(                                                                      \
        uint32_t block_dim, void* stream, TYPE* gva, int* error_flag, uint64_t pe_size, uint64_t config)         \
    {                                                                                                            \
        test_atomic_fetch_and_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, error_flag, pe_size, config); \
    }
ACLSHMEM_ATOMIC_LOGIC_FUNC_TYPE(ATOMIC_FETCH_AND_TEST);

/*****************************************************************************
 *                    atomic_or test                                          *
 *****************************************************************************/
/**
 * @brief atomic_or kernel test
 *        Each PE sets its own bit in other PEs' slots
 */
#define ATOMIC_OR_TEST_KERNEL(NAME, TYPE)                                                                   \
    extern "C" __global__ __aicore__ void test_atomic_or_##NAME##_kernel(__gm__ TYPE* gva, uint64_t config) \
    {                                                                                                       \
        util_set_ffts_config(config);                                                                       \
        int64_t pe = aclshmem_my_pe();                                                                      \
        int64_t pe_size = aclshmem_n_pes();                                                                 \
        __gm__ TYPE* dst_addr;                                                                              \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                          \
        uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;                                           \
        uint32_t sync_id = device_state->rdma_config.sync_id;                                               \
                                                                                                            \
        for (int64_t peer = 0; peer < pe_size; peer++) {                                                    \
            if (peer == pe) {                                                                               \
                continue;                                                                                   \
            }                                                                                               \
            dst_addr = gva + pe * MESSAGE_SIZE / sizeof(TYPE);                                              \
            if ASCEND_IS_AIV {                                                                              \
                if (AscendC::GetSubBlockIdx() == 0) {                                                       \
                    aclshmem_##NAME##_atomic_or((__gm__ TYPE*)dst_addr, (TYPE)(1LLU << pe), peer);          \
                    aclshmemx_roce_quiet(peer, reinterpret_cast<__ubuf__ char*>(copy_ub), sync_id);         \
                }                                                                                           \
            }                                                                                               \
        }                                                                                                   \
        aclshmem_barrier_all();                                                                             \
    }
ACLSHMEM_ATOMIC_LOGIC_FUNC_TYPE(ATOMIC_OR_TEST_KERNEL);

#define ATOMIC_OR_TEST(NAME, TYPE)                                                                \
    void test_atomic_or_##NAME##_do(uint32_t block_dim, void* stream, TYPE* gva, uint64_t config) \
    {                                                                                             \
        test_atomic_or_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);              \
    }
ACLSHMEM_ATOMIC_LOGIC_FUNC_TYPE(ATOMIC_OR_TEST);

/*****************************************************************************
 *                    atomic_fetch_or test                                    *
 *****************************************************************************/
/**
 * @brief atomic_fetch_or kernel test
 *        Each PE sets its own bit in other PEs' slots and returns old value
 *        Verifies return value has own bit not yet set
 */
#define ATOMIC_FETCH_OR_TEST_KERNEL(NAME, TYPE)                                                                    \
    extern "C" __global__ __aicore__ void test_atomic_fetch_or_##NAME##_kernel(                                    \
        __gm__ TYPE* gva, __gm__ int* error_flag, uint64_t pe_size, uint64_t config)                               \
    {                                                                                                              \
        util_set_ffts_config(config);                                                                              \
        int64_t pe = aclshmem_my_pe();                                                                             \
        int64_t pe_size_val = aclshmem_n_pes();                                                                    \
        __gm__ TYPE* dst_addr;                                                                                     \
        TYPE old_val;                                                                                              \
                                                                                                                   \
        for (int64_t peer = 0; peer < pe_size_val; peer++) {                                                       \
            if (peer == pe) {                                                                                      \
                continue;                                                                                          \
            }                                                                                                      \
            dst_addr = gva + pe * MESSAGE_SIZE / sizeof(TYPE);                                                     \
            if ASCEND_IS_AIV {                                                                                     \
                if (AscendC::GetSubBlockIdx() == 0) {                                                              \
                    old_val = aclshmem_##NAME##_atomic_fetch_or((__gm__ TYPE*)dst_addr, (TYPE)(1LLU << pe), peer); \
                    /* Verify: return value should have own bit not yet set */                                     \
                    if ((old_val & ((TYPE)(1LLU << pe))) != 0) {                                                   \
                        *error_flag = 1;                                                                           \
                    }                                                                                              \
                }                                                                                                  \
            }                                                                                                      \
            aclshmem_barrier_all();                                                                                \
        }                                                                                                          \
    }
ACLSHMEM_ATOMIC_LOGIC_FUNC_TYPE(ATOMIC_FETCH_OR_TEST_KERNEL);

#define ATOMIC_FETCH_OR_TEST(NAME, TYPE)                                                                        \
    void test_atomic_fetch_or_##NAME##_do(                                                                      \
        uint32_t block_dim, void* stream, TYPE* gva, int* error_flag, uint64_t pe_size, uint64_t config)        \
    {                                                                                                           \
        test_atomic_fetch_or_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, error_flag, pe_size, config); \
    }
ACLSHMEM_ATOMIC_LOGIC_FUNC_TYPE(ATOMIC_FETCH_OR_TEST);

/*****************************************************************************
 *                    atomic_xor test                                         *
 *****************************************************************************/
/**
 * @brief atomic_xor kernel test
 *        Each PE toggles its own bit in other PEs' slots (1→0)
 *        Initial: all bits = 1, final: lower pe_size bits = 0
 */
#define ATOMIC_XOR_TEST_KERNEL(NAME, TYPE)                                                                   \
    extern "C" __global__ __aicore__ void test_atomic_xor_##NAME##_kernel(__gm__ TYPE* gva, uint64_t config) \
    {                                                                                                        \
        util_set_ffts_config(config);                                                                        \
        int64_t pe = aclshmem_my_pe();                                                                       \
        int64_t pe_size = aclshmem_n_pes();                                                                  \
        __gm__ TYPE* dst_addr;                                                                               \
        __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();                           \
        uint64_t copy_ub = device_state->rdma_config.aclshmem_ub;                                            \
        uint32_t sync_id = device_state->rdma_config.sync_id;                                                \
                                                                                                             \
        for (int64_t peer = 0; peer < pe_size; peer++) {                                                     \
            if (peer == pe) {                                                                                \
                continue;                                                                                    \
            }                                                                                                \
            dst_addr = gva + pe * MESSAGE_SIZE / sizeof(TYPE);                                               \
            if ASCEND_IS_AIV {                                                                               \
                if (AscendC::GetSubBlockIdx() == 0) {                                                        \
                    aclshmem_##NAME##_atomic_xor((__gm__ TYPE*)dst_addr, (TYPE)(1LLU << pe), peer);          \
                    aclshmemx_roce_quiet(peer, reinterpret_cast<__ubuf__ char*>(copy_ub), sync_id);          \
                }                                                                                            \
            }                                                                                                \
        }                                                                                                    \
        aclshmem_barrier_all();                                                                              \
    }
ACLSHMEM_ATOMIC_LOGIC_FUNC_TYPE(ATOMIC_XOR_TEST_KERNEL);

#define ATOMIC_XOR_TEST(NAME, TYPE)                                                                \
    void test_atomic_xor_##NAME##_do(uint32_t block_dim, void* stream, TYPE* gva, uint64_t config) \
    {                                                                                              \
        test_atomic_xor_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);              \
    }
ACLSHMEM_ATOMIC_LOGIC_FUNC_TYPE(ATOMIC_XOR_TEST);

/*****************************************************************************
 *                    atomic_fetch_xor test                                   *
 *****************************************************************************/
/**
 * @brief atomic_fetch_xor kernel test
 *        Each PE toggles its own bit in other PEs' slots and returns old value
 *        Verifies return value has own bit still set (not yet toggled)
 */
#define ATOMIC_FETCH_XOR_TEST_KERNEL(NAME, TYPE)                                                                    \
    extern "C" __global__ __aicore__ void test_atomic_fetch_xor_##NAME##_kernel(                                    \
        __gm__ TYPE* gva, __gm__ int* error_flag, uint64_t pe_size, uint64_t config)                                \
    {                                                                                                               \
        util_set_ffts_config(config);                                                                               \
        int64_t pe = aclshmem_my_pe();                                                                              \
        int64_t pe_size_val = aclshmem_n_pes();                                                                     \
        __gm__ TYPE* dst_addr;                                                                                      \
        TYPE old_val;                                                                                               \
                                                                                                                    \
        for (int64_t peer = 0; peer < pe_size_val; peer++) {                                                        \
            if (peer == pe) {                                                                                       \
                continue;                                                                                           \
            }                                                                                                       \
            dst_addr = gva + pe * MESSAGE_SIZE / sizeof(TYPE);                                                      \
            if ASCEND_IS_AIV {                                                                                      \
                if (AscendC::GetSubBlockIdx() == 0) {                                                               \
                    old_val = aclshmem_##NAME##_atomic_fetch_xor((__gm__ TYPE*)dst_addr, (TYPE)(1LLU << pe), peer); \
                    /* Verify: return value should have own bit still set (1) */                                    \
                    if ((old_val & ((TYPE)(1LLU << pe))) == 0) {                                                    \
                        *error_flag = 1;                                                                            \
                    }                                                                                               \
                }                                                                                                   \
            }                                                                                                       \
        }                                                                                                           \
        aclshmem_barrier_all();                                                                                     \
    }
ACLSHMEM_ATOMIC_LOGIC_FUNC_TYPE(ATOMIC_FETCH_XOR_TEST_KERNEL);

#define ATOMIC_FETCH_XOR_TEST(NAME, TYPE)                                                                        \
    void test_atomic_fetch_xor_##NAME##_do(                                                                      \
        uint32_t block_dim, void* stream, TYPE* gva, int* error_flag, uint64_t pe_size, uint64_t config)         \
    {                                                                                                            \
        test_atomic_fetch_xor_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, error_flag, pe_size, config); \
    }
ACLSHMEM_ATOMIC_LOGIC_FUNC_TYPE(ATOMIC_FETCH_XOR_TEST);
