/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "opdev/bfloat16.h"
#include "opdev/fp16_t.h"
#include "func_type.h"
#include "unittest_main_test.h"
#include "atomic.h"

#define TEST_ATOMIC_ADD_FUNC(NAME, TYPE) \
    extern void test_atomic_add_##NAME##_do(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)

ACLSHMEM_ATOMIC_ADD_FUNC_TYPE_HOST(TEST_ATOMIC_ADD_FUNC);
ACLSHMEM_ATOMIC_ADD_910_FUNC_TYPE(TEST_ATOMIC_ADD_FUNC);
ACLSHMEM_ATOMIC_ADD_EXT_FUNC_TYPE(TEST_ATOMIC_ADD_FUNC);

#define TEST_ACLSHMEM_ATOMIC_ADD_HOST(NAME, TYPE)                                                                   \
    static void test_atomic_add_##NAME##_host(aclrtStream stream, uint32_t pe_id, uint32_t pe_size)                 \
    {                                                                                                               \
        size_t messageSize = 64;                                                                                    \
        TYPE* xHost;                                                                                                \
        size_t totalSize = messageSize * pe_size;                                                                   \
                                                                                                                    \
        ASSERT_EQ(aclrtMallocHost((void**)(&xHost), totalSize), 0);                                                 \
        for (uint32_t i = 0; i < messageSize / sizeof(TYPE); i++) {                                                 \
            xHost[i] = pe_id + 1;                                                                                   \
        }                                                                                                           \
                                                                                                                    \
        uint8_t* ptr = (uint8_t*)aclshmem_malloc(totalSize);                                                        \
        ASSERT_EQ(                                                                                                  \
            aclrtMemcpy(ptr + pe_id * messageSize, messageSize, xHost, messageSize, ACL_MEMCPY_HOST_TO_DEVICE), 0); \
                                                                                                                    \
        uint32_t block_dim = 1;                                                                                     \
        test_atomic_add_##NAME##_do(block_dim, stream, ptr, util_get_ffts_config());                                \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                               \
                                                                                                                    \
        std::string p_name = "[Process " + std::to_string(pe_id) + "] ";                                            \
        std::cout << p_name;                                                                                        \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                     \
        for (uint32_t i = 0; i < pe_size; i++) {                                                                    \
            if (i == pe_id) {                                                                                       \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(pe_id + 1));                     \
            } else {                                                                                                \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(block_dim));                     \
            }                                                                                                       \
        }                                                                                                           \
    }
ACLSHMEM_ATOMIC_ADD_FUNC_TYPE_HOST(TEST_ACLSHMEM_ATOMIC_ADD_HOST);
ACLSHMEM_ATOMIC_ADD_910_FUNC_TYPE(TEST_ACLSHMEM_ATOMIC_ADD_HOST);
ACLSHMEM_ATOMIC_ADD_EXT_FUNC_TYPE(TEST_ACLSHMEM_ATOMIC_ADD_HOST);

#define TEST_CLEANUP_AND_EXIT(pe_id, stream, device_id)                            \
    do {                                                                           \
        std::cout << "[TEST] begin to exit...... pe_id: " << (pe_id) << std::endl; \
        test_finalize(stream, device_id);                                          \
    } while (0)

#define TEST_ACLSHMEM_ATOMIC_ADD(NAME, TYPE)                                                  \
    void test_aclshmem_atomic_add_##NAME##_mem(int pe_id, int n_pes, uint64_t local_mem_size) \
    {                                                                                         \
        int32_t device_id = pe_id % test_gnpu_num + test_first_npu;                           \
        aclrtStream stream;                                                                   \
        test_init(pe_id, n_pes, local_mem_size, &stream);                                     \
        ASSERT_NE(stream, nullptr);                                                           \
        test_atomic_add_##NAME##_host(stream, pe_id, n_pes);                                  \
        std::cout << "[TEST] begin to exit...... pe_id: " << pe_id << std::endl;              \
        test_finalize(stream, device_id);                                                     \
    }
ACLSHMEM_ATOMIC_ADD_FUNC_TYPE_HOST(TEST_ACLSHMEM_ATOMIC_ADD);
ACLSHMEM_ATOMIC_ADD_910_FUNC_TYPE(TEST_ACLSHMEM_ATOMIC_ADD);
ACLSHMEM_ATOMIC_ADD_EXT_FUNC_TYPE(TEST_ACLSHMEM_ATOMIC_ADD);

#define TEST_ATOMIC_ADD_API(NAME, TYPE)                                                       \
    TEST(TestMemApi, TestShmemAtomicAdd##NAME##Mem)                                           \
    {                                                                                         \
        const int processCount = test_gnpu_num;                                               \
        uint64_t local_mem_size = 1024UL * 1024UL * 64;                                       \
        test_mutil_task(test_aclshmem_atomic_add_##NAME##_mem, local_mem_size, processCount); \
    }

ACLSHMEM_ATOMIC_ADD_FUNC_TYPE_HOST(TEST_ATOMIC_ADD_API);
ACLSHMEM_ATOMIC_ADD_910_FUNC_TYPE(TEST_ATOMIC_ADD_API);

#define TEST_ATOMIC_ADD_EXT_API(NAME, TYPE)                                                       \
    TEST(TestMemApi, TestShmemAtomicAdd##NAME##Mem)                                               \
    {                                                                                             \
        const int processCount = test_gnpu_num;                                                   \
        uint64_t local_mem_size = 1024UL * 1024UL * 64;                                           \
        if (is_hardware_atomic_supported()) {                                                     \
            test_mutil_task(test_aclshmem_atomic_add_##NAME##_mem, local_mem_size, processCount); \
        } else {                                                                                  \
            std::cout << "SKIP TestShmemAtomicAdd" << #NAME << "Mem" << std::endl;                \
        }                                                                                         \
    }

ACLSHMEM_ATOMIC_ADD_EXT_FUNC_TYPE(TEST_ATOMIC_ADD_EXT_API);

/*****************************************************************************
 *                    atomic_fetch_add test                                  *
 *****************************************************************************/
#define TEST_ATOMIC_FETCH_ADD_FUNC(NAME, TYPE)     \
    extern void test_atomic_fetch_add_##NAME##_do( \
        uint32_t block_dim, void* stream, TYPE* gva, int* error_flag, uint64_t config)

ACLSHMEM_ATOMIC_ADD_950_FUNC_TYPE(TEST_ATOMIC_FETCH_ADD_FUNC);

#define TEST_ACLSHMEM_ATOMIC_FETCH_ADD_HOST(NAME, TYPE)                                                                \
    static void test_atomic_fetch_add_##NAME##_host(aclrtStream stream, uint32_t pe_id, uint32_t pe_size)              \
    {                                                                                                                  \
        size_t messageSize = 64;                                                                                       \
        TYPE* xHost;                                                                                                   \
        size_t totalSize = messageSize * pe_size;                                                                      \
                                                                                                                       \
        ASSERT_EQ(aclrtMallocHost((void**)(&xHost), totalSize), 0);                                                    \
        for (uint32_t i = 0; i < messageSize / sizeof(TYPE); i++) {                                                    \
            xHost[i] = pe_id + 1;                                                                                      \
        }                                                                                                              \
                                                                                                                       \
        TYPE* ptr = (TYPE*)aclshmem_malloc(totalSize);                                                                 \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(                                                                                               \
                ptr + pe_id * messageSize / sizeof(TYPE), messageSize, xHost, messageSize, ACL_MEMCPY_HOST_TO_DEVICE), \
            0);                                                                                                        \
                                                                                                                       \
        uint32_t block_dim = 1;                                                                                        \
        /* Allocate error_flag buffer on device */                                                                     \
        int* error_flag_dev;                                                                                           \
        int* error_flag_host;                                                                                          \
        ASSERT_EQ(aclrtMalloc((void**)(&error_flag_dev), 512, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                         \
        ASSERT_EQ(aclrtMallocHost((void**)(&error_flag_host), sizeof(int)), 0);                                        \
        *error_flag_host = 0;                                                                                          \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(error_flag_dev, sizeof(int), error_flag_host, sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE), 0);     \
                                                                                                                       \
        test_atomic_fetch_add_##NAME##_do(block_dim, stream, (TYPE*)ptr, error_flag_dev, util_get_ffts_config());      \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
                                                                                                                       \
        std::string p_name = "[Process " + std::to_string(pe_id) + "] ";                                               \
        std::cout << p_name;                                                                                           \
                                                                                                                       \
        /* Verify return value range via error_flag */                                                                 \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(error_flag_host, sizeof(int), error_flag_dev, sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST), 0);     \
        ASSERT_EQ(*error_flag_host, 0);                                                                                \
        aclrtFree(error_flag_dev);                                                                                     \
        aclrtFreeHost(error_flag_host);                                                                                \
                                                                                                                       \
        /* Verify data results */                                                                                      \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                        \
        for (uint32_t i = 0; i < pe_size; i++) {                                                                       \
            if (i == pe_id) {                                                                                          \
                /* Own slot: initialized to pe_id+1, not modified by others */                                         \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(pe_id + 1));                        \
            } else {                                                                                                   \
                /* Other slots: initialized to 0, modified block_dim times by other pes */                             \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(block_dim));                        \
            }                                                                                                          \
        }                                                                                                              \
    }
ACLSHMEM_ATOMIC_ADD_950_FUNC_TYPE(TEST_ACLSHMEM_ATOMIC_FETCH_ADD_HOST);

#define TEST_ACLSHMEM_ATOMIC_FETCH_ADD(NAME, TYPE)                                                  \
    void test_aclshmem_atomic_fetch_add_##NAME##_mem(int pe_id, int n_pes, uint64_t local_mem_size) \
    {                                                                                               \
        int32_t device_id = pe_id % test_gnpu_num + test_first_npu;                                 \
        aclrtStream stream;                                                                         \
        test_init(pe_id, n_pes, local_mem_size, &stream);                                           \
        ASSERT_NE(stream, nullptr);                                                                 \
        test_atomic_fetch_add_##NAME##_host(stream, pe_id, n_pes);                                  \
        std::cout << "[TEST] begin to exit...... pe_id: " << pe_id << std::endl;                    \
        test_finalize(stream, device_id);                                                           \
    }
ACLSHMEM_ATOMIC_ADD_950_FUNC_TYPE(TEST_ACLSHMEM_ATOMIC_FETCH_ADD);

#define TEST_ATOMIC_FETCH_ADD_API(NAME, TYPE)                                                           \
    TEST(TestMemApi, TestShmemAtomicFetchAdd##NAME##Mem)                                                \
    {                                                                                                   \
        const int processCount = test_gnpu_num;                                                         \
        uint64_t local_mem_size = 1024UL * 1024UL * 64;                                                 \
        if (is_hardware_atomic_supported()) {                                                           \
            test_mutil_task(test_aclshmem_atomic_fetch_add_##NAME##_mem, local_mem_size, processCount); \
        } else {                                                                                        \
            std::cout << "SKIP TestShmemAtomicFetchAdd" << #NAME << "Mem" << std::endl;                 \
        }                                                                                               \
    }
ACLSHMEM_ATOMIC_ADD_950_FUNC_TYPE(TEST_ATOMIC_FETCH_ADD_API);

/*****************************************************************************
 *                    atomic_inc test                                        *
 *****************************************************************************/
#define TEST_ATOMIC_INC_FUNC(NAME, TYPE) \
    extern void test_atomic_inc_##NAME##_do(uint32_t block_dim, void* stream, TYPE* gva, uint64_t config)

ACLSHMEM_ATOMIC_ADD_950_FUNC_TYPE(TEST_ATOMIC_INC_FUNC);

#define TEST_ACLSHMEM_ATOMIC_INC_HOST(NAME, TYPE)                                                                      \
    static void test_atomic_inc_##NAME##_host(aclrtStream stream, uint32_t pe_id, uint32_t pe_size)                    \
    {                                                                                                                  \
        size_t messageSize = 64;                                                                                       \
        TYPE* xHost;                                                                                                   \
        size_t totalSize = messageSize * pe_size;                                                                      \
                                                                                                                       \
        ASSERT_EQ(aclrtMallocHost((void**)(&xHost), totalSize), 0);                                                    \
        for (uint32_t i = 0; i < messageSize / sizeof(TYPE); i++) {                                                    \
            xHost[i] = pe_id + 1;                                                                                      \
        }                                                                                                              \
                                                                                                                       \
        TYPE* ptr = (TYPE*)aclshmem_malloc(totalSize);                                                                 \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(                                                                                               \
                ptr + pe_id * messageSize / sizeof(TYPE), messageSize, xHost, messageSize, ACL_MEMCPY_HOST_TO_DEVICE), \
            0);                                                                                                        \
                                                                                                                       \
        uint32_t block_dim = 1;                                                                                        \
        test_atomic_inc_##NAME##_do(block_dim, stream, (TYPE*)ptr, util_get_ffts_config());                            \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
                                                                                                                       \
        std::string p_name = "[Process " + std::to_string(pe_id) + "] ";                                               \
        std::cout << p_name;                                                                                           \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                        \
        for (uint32_t i = 0; i < pe_size; i++) {                                                                       \
            if (i == pe_id) {                                                                                          \
                /* Own slot: initialized to pe_id+1, incremented by (pe_size-1) other pes */                           \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(pe_id + 1));                        \
            } else {                                                                                                   \
                /* Other slots: initialized to 0 by self, incremented by one pe */                                     \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(block_dim));                        \
            }                                                                                                          \
        }                                                                                                              \
    }
ACLSHMEM_ATOMIC_ADD_950_FUNC_TYPE(TEST_ACLSHMEM_ATOMIC_INC_HOST);

#define TEST_ACLSHMEM_ATOMIC_INC(NAME, TYPE)                                                  \
    void test_aclshmem_atomic_inc_##NAME##_mem(int pe_id, int n_pes, uint64_t local_mem_size) \
    {                                                                                         \
        int32_t device_id = pe_id % test_gnpu_num + test_first_npu;                           \
        aclrtStream stream;                                                                   \
        test_init(pe_id, n_pes, local_mem_size, &stream);                                     \
        ASSERT_NE(stream, nullptr);                                                           \
        test_atomic_inc_##NAME##_host(stream, pe_id, n_pes);                                  \
        std::cout << "[TEST] begin to exit...... pe_id: " << pe_id << std::endl;              \
        test_finalize(stream, device_id);                                                     \
    }
ACLSHMEM_ATOMIC_ADD_950_FUNC_TYPE(TEST_ACLSHMEM_ATOMIC_INC);

#define TEST_ATOMIC_INC_API(NAME, TYPE)                                                           \
    TEST(TestMemApi, TestShmemAtomicInc##NAME##Mem)                                               \
    {                                                                                             \
        const int processCount = test_gnpu_num;                                                   \
        uint64_t local_mem_size = 1024UL * 1024UL * 64;                                           \
        if (is_hardware_atomic_supported()) {                                                     \
            test_mutil_task(test_aclshmem_atomic_inc_##NAME##_mem, local_mem_size, processCount); \
        } else {                                                                                  \
            std::cout << "SKIP TestShmemAtomicInc" << #NAME << "Mem" << std::endl;                \
        }                                                                                         \
    }
ACLSHMEM_ATOMIC_ADD_950_FUNC_TYPE(TEST_ATOMIC_INC_API);

/*****************************************************************************
 *                    atomic_fetch_inc test                                  *
 *****************************************************************************/
#define TEST_ATOMIC_FETCH_INC_FUNC(NAME, TYPE)     \
    extern void test_atomic_fetch_inc_##NAME##_do( \
        uint32_t block_dim, void* stream, TYPE* gva, int* error_flag, uint64_t config)

ACLSHMEM_ATOMIC_ADD_950_FUNC_TYPE(TEST_ATOMIC_FETCH_INC_FUNC);

#define TEST_ACLSHMEM_ATOMIC_FETCH_INC_HOST(NAME, TYPE)                                                                \
    static void test_atomic_fetch_inc_##NAME##_host(aclrtStream stream, uint32_t pe_id, uint32_t pe_size)              \
    {                                                                                                                  \
        size_t messageSize = 64;                                                                                       \
        TYPE* xHost;                                                                                                   \
        size_t totalSize = messageSize * pe_size;                                                                      \
                                                                                                                       \
        ASSERT_EQ(aclrtMallocHost((void**)(&xHost), totalSize), 0);                                                    \
        for (uint32_t i = 0; i < messageSize / sizeof(TYPE); i++) {                                                    \
            xHost[i] = pe_id + 1;                                                                                      \
        }                                                                                                              \
                                                                                                                       \
        TYPE* ptr = (TYPE*)aclshmem_malloc(totalSize);                                                                 \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(                                                                                               \
                ptr + pe_id * messageSize / sizeof(TYPE), messageSize, xHost, messageSize, ACL_MEMCPY_HOST_TO_DEVICE), \
            0);                                                                                                        \
                                                                                                                       \
        uint32_t block_dim = 1;                                                                                        \
        /* Allocate error_flag buffer on device */                                                                     \
        int* error_flag_dev;                                                                                           \
        int* error_flag_host;                                                                                          \
        ASSERT_EQ(aclrtMalloc((void**)(&error_flag_dev), 512, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                         \
        ASSERT_EQ(aclrtMallocHost((void**)(&error_flag_host), sizeof(int)), 0);                                        \
        *error_flag_host = 0;                                                                                          \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(error_flag_dev, sizeof(int), error_flag_host, sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE), 0);     \
                                                                                                                       \
        test_atomic_fetch_inc_##NAME##_do(block_dim, stream, (TYPE*)ptr, error_flag_dev, util_get_ffts_config());      \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
                                                                                                                       \
        std::string p_name = "[Process " + std::to_string(pe_id) + "] ";                                               \
        std::cout << p_name;                                                                                           \
                                                                                                                       \
        /* Verify return value range via error_flag */                                                                 \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(error_flag_host, sizeof(int), error_flag_dev, sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST), 0);     \
        ASSERT_EQ(*error_flag_host, 0);                                                                                \
        aclrtFree(error_flag_dev);                                                                                     \
        aclrtFreeHost(error_flag_host);                                                                                \
                                                                                                                       \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                        \
        for (uint32_t i = 0; i < pe_size; i++) {                                                                       \
            if (i == pe_id) {                                                                                          \
                /* Own slot: initialized to pe_id+1, incremented by (pe_size-1) other pes */                           \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(pe_id + 1));                        \
            } else {                                                                                                   \
                /* Other slots: initialized to 0 by self, incremented by one pe */                                     \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(block_dim));                        \
            }                                                                                                          \
        }                                                                                                              \
    }
ACLSHMEM_ATOMIC_ADD_950_FUNC_TYPE(TEST_ACLSHMEM_ATOMIC_FETCH_INC_HOST);

#define TEST_ACLSHMEM_ATOMIC_FETCH_INC(NAME, TYPE)                                                  \
    void test_aclshmem_atomic_fetch_inc_##NAME##_mem(int pe_id, int n_pes, uint64_t local_mem_size) \
    {                                                                                               \
        int32_t device_id = pe_id % test_gnpu_num + test_first_npu;                                 \
        aclrtStream stream;                                                                         \
        test_init(pe_id, n_pes, local_mem_size, &stream);                                           \
        ASSERT_NE(stream, nullptr);                                                                 \
        test_atomic_fetch_inc_##NAME##_host(stream, pe_id, n_pes);                                  \
        std::cout << "[TEST] begin to exit...... pe_id: " << pe_id << std::endl;                    \
        test_finalize(stream, device_id);                                                           \
    }
ACLSHMEM_ATOMIC_ADD_950_FUNC_TYPE(TEST_ACLSHMEM_ATOMIC_FETCH_INC);

#define TEST_ATOMIC_FETCH_INC_API(NAME, TYPE)                                                           \
    TEST(TestMemApi, TestShmemAtomicFetchInc##NAME##Mem)                                                \
    {                                                                                                   \
        const int processCount = test_gnpu_num;                                                         \
        uint64_t local_mem_size = 1024UL * 1024UL * 64;                                                 \
        if (is_hardware_atomic_supported()) {                                                           \
            test_mutil_task(test_aclshmem_atomic_fetch_inc_##NAME##_mem, local_mem_size, processCount); \
        } else {                                                                                        \
            std::cout << "SKIP TestShmemAtomicFetchInc" << #NAME << "Mem" << std::endl;                 \
        }                                                                                               \
    }
ACLSHMEM_ATOMIC_ADD_950_FUNC_TYPE(TEST_ATOMIC_FETCH_INC_API);