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
#include "unittest/utils/scope_env.h"

const int INSTANCE_NUM = 3;
const int input_length = 256;
const int ut_magic = 10;

#define TEST_FUNC(NAME, TYPE)                                                                           \
    extern void multi_instance_##NAME##_test_put(uint32_t block_dim, void *stream, uint64_t config,     \
            uint8_t *src, uint8_t *dst, uint8_t *shmem_addrs_device, int inst_count)

ACLSHMEM_FUNC_TYPE_HOST(TEST_FUNC);

#define TEST_PUT_GET(NAME, TYPE)                                                                                    \
    static void multi_instance_test_##NAME##_put(                                                                   \
            aclrtStream stream, uint32_t rank_id, uint32_t rank_size, int inst_count)                               \
    {                                                                                                               \
        /* Each Instance Memcpy Num */                                                                              \
        int single_num = input_length;                                                                              \
        size_t input_size = single_num * sizeof(TYPE);                                                              \
                                                                                                                    \
        /* Total Instance Memcpy Num */                                                                             \
        int total_num = input_length * inst_count;                                                                  \
        size_t total_size = total_num * sizeof(TYPE);                                                               \
                                                                                                                    \
        std::vector<TYPE> input(total_size, 0);                                                                     \
        for (int i = 0; i < total_num; i++) {                                                                       \
            input[i] = static_cast<TYPE>(i + ut_magic);                                                             \
        }                                                                                                           \
                                                                                                                    \
        void *src;                                                                                                  \
        ASSERT_EQ(aclrtMalloc(&src, total_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                                    \
        ASSERT_EQ(aclrtMemcpy(src, total_size, input.data(), total_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);            \
                                                                                                                    \
        void *dst;                                                                                                  \
        ASSERT_EQ(aclrtMalloc(&dst, total_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                                    \
        ASSERT_EQ(aclrtMemset(dst, total_size, 0, total_size), 0);                                                  \
                                                                                                                    \
        /* Store each instance's shmem addr */                                                                      \
        void *shmem_addrs_device;                                                                                   \
        ASSERT_EQ(aclrtMalloc(&shmem_addrs_device, inst_count * sizeof(uint64_t), ACL_MEM_MALLOC_NORMAL_ONLY), 0);  \
                                                                                                                    \
        /* aclshmem_malloc each instance's shmem addr */                                                            \
        std::vector<uint64_t> shmem_addrs_host = {};                                                                \
        for (int i = 0; i < inst_count; i++) {                                                                      \
            uint64_t instance_offset = 1;                                                                           \
            uint64_t instance_id = (instance_offset + i);                                                           \
            aclshmemx_instance_ctx_set(instance_id);                                                                \
            uint64_t shmem_addr = (uint64_t)aclshmem_malloc(input_size);                                            \
            ASSERT_NE((void *)shmem_addr, nullptr);                                                                 \
            shmem_addrs_host.push_back(shmem_addr);                                                                 \
        }                                                                                                           \
                                                                                                                    \
        /* Copy shmem addr list to device */                                                                        \
        ASSERT_EQ(aclrtMemcpy(shmem_addrs_device, inst_count * sizeof(uint64_t),                                    \
                    shmem_addrs_host.data(), inst_count * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE), 0);         \
                                                                                                                    \
        uint32_t block_dim = 1;                                                                                     \
        multi_instance_##NAME##_test_put(block_dim, stream,                                                         \
            util_get_ffts_config(), (uint8_t *)src, (uint8_t *)dst, (uint8_t *)shmem_addrs_device, inst_count);     \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                               \
                                                                                                                    \
        /* result check */                                                                                          \
        std::vector<TYPE> output_gm(single_num, 0);                                                                 \
        std::vector<TYPE> output_shmem(single_num, 0);                                                              \
        for (int loop = 0; loop < inst_count; loop++) {                                                             \
            ASSERT_EQ(aclrtMemcpy(output_gm.data(), input_size,                                                     \
                (char *)dst + loop * input_size, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);                        \
            ASSERT_EQ(aclrtMemcpy(output_shmem.data(), input_size,                                                  \
                (void *)(shmem_addrs_host[loop]), input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);                       \
                                                                                                                    \
            /* Check if shmem addr get same result with ordinary gm */                                              \
            int32_t flag_gm = 0;                                                                                    \
            for (int i = 0; i < single_num; i++) {                                                                  \
                if (output_shmem[i] != output_gm[i])                                                                \
                    flag_gm = 1;                                                                                    \
            }                                                                                                       \
            ASSERT_EQ(flag_gm, 0);                                                                                  \
            /* Check if result is expected */                                                                       \
            int32_t flag_value = 0;                                                                                 \
            for (int i = 0; i < single_num; i++) {                                                                  \
                if (output_shmem[i] != static_cast<TYPE>(loop * single_num + i + ut_magic))                         \
                    flag_value = 1;                                                                                 \
            }                                                                                                       \
            ASSERT_EQ(flag_value, 0);                                                                               \
            uint64_t instance_offset = 1;                                                                           \
            aclshmemx_instance_ctx_set(instance_offset + loop);                                                     \
            aclshmem_free((void *)shmem_addrs_host[loop]);                                                          \
        }                                                                                                           \
    }

ACLSHMEM_FUNC_TYPE_HOST(TEST_PUT_GET);

#define TEST_ACLSHMEM_MEM(NAME, TYPE)                                                                           \
    void test_multi_instance_##NAME##_aclshmem_mem(int rank_id, int n_ranks, uint64_t local_mem_size)           \
    {                                                                                                           \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                           \
        int inst_count = INSTANCE_NUM;                                                                          \
        aclrtStream stream;                                                                                     \
        test_multi_instance_init(rank_id, n_ranks, local_mem_size, &stream, inst_count);                        \
        ASSERT_NE(stream, nullptr);                                                                             \
                                                                                                                \
        multi_instance_test_##NAME##_put(stream, rank_id, n_ranks, inst_count);                                 \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                            \
        test_multi_instance_finalize(stream, device_id, inst_count);                                            \
    }

ACLSHMEM_FUNC_TYPE_HOST(TEST_ACLSHMEM_MEM);

#define TESTAPI(NAME, TYPE)                                                                         \
    TEST(TestMemApi, TestShmemMultiInstanceGM##NAME##Mem)                                           \
    {                                                                                               \
        const char* fix_port_range = "20000:21023";                                                 \
        ScopedEnv env_port("SHMEM_INSTANCE_PORT_RANGE", fix_port_range);                            \
                                                                                                    \
        const int process_count = test_gnpu_num;                                                    \
        uint64_t local_mem_size = 64 * 1024UL * 1024;                                               \
        test_mutil_task(test_multi_instance_##NAME##_aclshmem_mem, local_mem_size, process_count);  \
    }

ACLSHMEM_FUNC_TYPE_HOST(TESTAPI);
