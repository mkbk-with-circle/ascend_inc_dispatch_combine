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
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmem.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"
#include "rdma_sync_barrier_kernel.h"

constexpr int32_t ROCE_SYNC_BARRIER_TEST_NUM = 3;
static const int MESSAGE_LENGTH = 1024 * 256;

static void fill_segment_to_device(uint8_t *addr_dev, int offset, int message_length, uint64_t pattern)
{
    int num_elements = message_length / (int)sizeof(uint64_t);
    uint8_t *host_buf;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&host_buf), message_length), 0);
    uint64_t *data = reinterpret_cast<uint64_t *>(host_buf);
    for (int i = 0; i < num_elements; i++) {
        data[i] = pattern;
    }
    ASSERT_EQ(aclrtMemcpy(addr_dev + offset, message_length, host_buf, message_length, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtFreeHost(host_buf), 0);
}

static void verify_segment_from_device(uint8_t *addr_dev, int offset, int message_length, uint64_t expected)
{
    int num_elements = message_length / (int)sizeof(uint64_t);
    uint8_t *host_buf;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&host_buf), message_length), 0);
    ASSERT_EQ(aclrtMemcpy(host_buf, message_length, addr_dev + offset, message_length, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    uint64_t *data = reinterpret_cast<uint64_t *>(host_buf);
    for (int i = 0; i < num_elements; i++) {
        ASSERT_EQ(data[i], expected);
    }
    ASSERT_EQ(aclrtFreeHost(host_buf), 0);
}

// ---- roce_sync_all 测试 ----
static void test_roce_sync_all(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    auto status = test_rdma_init(rank_id, n_ranks, local_mem_size, &stream);
    if (status != 0) { return; }
    ASSERT_NE(stream, nullptr);

    size_t total_size = (size_t)n_ranks * MESSAGE_LENGTH;
    uint8_t *addr_dev = static_cast<uint8_t *>(aclshmem_malloc(total_size));
    ASSERT_NE(addr_dev, nullptr);
    ASSERT_EQ(aclrtMemset(addr_dev, total_size, 0, total_size), 0);

    fill_segment_to_device(addr_dev, rank_id * MESSAGE_LENGTH, MESSAGE_LENGTH, (uint64_t)(rank_id + 1));

    for (int32_t i = 1; i <= ROCE_SYNC_BARRIER_TEST_NUM; i++) {
        std::cout << "[TEST] roce_sync_all rank_id: " << rank_id
                  << " msg_len: " << MESSAGE_LENGTH << " time: " << i << std::endl;
        roce_sync_all_data_do(stream, util_get_ffts_config(), addr_dev, rank_id, n_ranks, MESSAGE_LENGTH);
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

        for (int pe = 0; pe < n_ranks; pe++) {
            if (pe == rank_id) continue;
            verify_segment_from_device(addr_dev, pe * MESSAGE_LENGTH, MESSAGE_LENGTH, (uint64_t)(pe + 1));
        }
    }

    aclshmem_free(addr_dev);

    test_finalize(stream, device_id);
}

// ---- roce_barrier_all 测试 ----
static void test_roce_sync_all_with_buf(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    auto status = test_rdma_init(rank_id, n_ranks, local_mem_size, &stream);
    if (status != 0) { return; }
    ASSERT_NE(stream, nullptr);

    size_t total_size = (size_t)n_ranks * MESSAGE_LENGTH;
    uint8_t *addr_dev = static_cast<uint8_t *>(aclshmem_malloc(total_size));
    ASSERT_NE(addr_dev, nullptr);
    ASSERT_EQ(aclrtMemset(addr_dev, total_size, 0, total_size), 0);

    fill_segment_to_device(addr_dev, rank_id * MESSAGE_LENGTH, MESSAGE_LENGTH, (uint64_t)(rank_id + 1));

    for (int32_t i = 1; i <= ROCE_SYNC_BARRIER_TEST_NUM; i++) {
        std::cout << "[TEST] roce_sync_all_with_buf rank_id: " << rank_id
                  << " msg_len: " << MESSAGE_LENGTH << " time: " << i << std::endl;
        roce_sync_all_with_buf_data_do(stream, util_get_ffts_config(), addr_dev, rank_id, n_ranks, MESSAGE_LENGTH);
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

        for (int pe = 0; pe < n_ranks; pe++) {
            if (pe == rank_id) continue;
            verify_segment_from_device(addr_dev, pe * MESSAGE_LENGTH, MESSAGE_LENGTH, (uint64_t)(pe + 1));
        }
    }

    aclshmem_free(addr_dev);

    test_finalize(stream, device_id);
}

// ---- roce_barrier_all 测试 ----
static void test_roce_barrier_all(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    auto status = test_rdma_init(rank_id, n_ranks, local_mem_size, &stream);
    if (status != 0) { return; }
    ASSERT_NE(stream, nullptr);

    size_t total_size = (size_t)n_ranks * MESSAGE_LENGTH;
    uint8_t *addr_dev = static_cast<uint8_t *>(aclshmem_malloc(total_size));
    ASSERT_NE(addr_dev, nullptr);
    ASSERT_EQ(aclrtMemset(addr_dev, total_size, 0, total_size), 0);

    fill_segment_to_device(addr_dev, rank_id * MESSAGE_LENGTH, MESSAGE_LENGTH, (uint64_t)(rank_id + 1));

    for (int32_t i = 1; i <= ROCE_SYNC_BARRIER_TEST_NUM; i++) {
        std::cout << "[TEST] roce_barrier_all rank_id: " << rank_id
                  << " msg_len: " << MESSAGE_LENGTH << " time: " << i << std::endl;
        roce_barrier_all_data_do(stream, util_get_ffts_config(), addr_dev, rank_id, n_ranks, MESSAGE_LENGTH);
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

        for (int pe = 0; pe < n_ranks; pe++) {
            if (pe == rank_id) continue;
            verify_segment_from_device(addr_dev, pe * MESSAGE_LENGTH, MESSAGE_LENGTH, (uint64_t)(pe + 1));
        }
    }

    aclshmem_free(addr_dev);

    test_finalize(stream, device_id);
}

// ---- roce_barrier_all(buf, sync_id) 测试 ----
static void test_roce_barrier_all_with_buf(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    auto status = test_rdma_init(rank_id, n_ranks, local_mem_size, &stream);
    if (status != 0) { return; }
    ASSERT_NE(stream, nullptr);

    size_t total_size = (size_t)n_ranks * MESSAGE_LENGTH;
    uint8_t *addr_dev = static_cast<uint8_t *>(aclshmem_malloc(total_size));
    ASSERT_NE(addr_dev, nullptr);
    ASSERT_EQ(aclrtMemset(addr_dev, total_size, 0, total_size), 0);

    fill_segment_to_device(addr_dev, rank_id * MESSAGE_LENGTH, MESSAGE_LENGTH, (uint64_t)(rank_id + 1));

    for (int32_t i = 1; i <= ROCE_SYNC_BARRIER_TEST_NUM; i++) {
        std::cout << "[TEST] roce_barrier_all_with_buf rank_id: " << rank_id
                  << " msg_len: " << MESSAGE_LENGTH << " time: " << i << std::endl;
        roce_barrier_all_with_buf_data_do(stream, util_get_ffts_config(), addr_dev, rank_id, n_ranks, MESSAGE_LENGTH);
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

        for (int pe = 0; pe < n_ranks; pe++) {
            if (pe == rank_id) continue;
            verify_segment_from_device(addr_dev, pe * MESSAGE_LENGTH, MESSAGE_LENGTH, (uint64_t)(pe + 1));
        }
    }

    aclshmem_free(addr_dev);

    test_finalize(stream, device_id);
}

// ---- roce_sync(team) 测试 ----
static void test_roce_sync_team_with_buf(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    auto status = test_rdma_init(rank_id, n_ranks, local_mem_size, &stream);
    if (status != 0) { return; }
    ASSERT_NE(stream, nullptr);

    aclshmem_team_t team_odd;
    int start = 1;
    int stride = 2;
    int team_size = n_ranks / 2;
    aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);

    size_t total_size = (size_t)n_ranks * MESSAGE_LENGTH;
    uint8_t *addr_dev = static_cast<uint8_t *>(aclshmem_malloc(total_size));
    ASSERT_NE(addr_dev, nullptr);
    ASSERT_EQ(aclrtMemset(addr_dev, total_size, 0, total_size), 0);

    if (rank_id & 1) {
        fill_segment_to_device(addr_dev, rank_id * MESSAGE_LENGTH, MESSAGE_LENGTH, (uint64_t)(rank_id + 1));
    }

    for (int32_t i = 1; i <= ROCE_SYNC_BARRIER_TEST_NUM; i++) {
        if (rank_id & 1) {
            std::cout << "[TEST] roce_sync_team_with_buf rank_id: " << rank_id
                      << " msg_len: " << MESSAGE_LENGTH << " time: " << i << std::endl;
            roce_sync_team_with_buf_data_do(stream, util_get_ffts_config(), addr_dev, rank_id, n_ranks, MESSAGE_LENGTH, team_odd);
            ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

            for (int j = 0; j < team_size; j++) {
                int pe = start + j * stride;
                if (pe == rank_id) continue;
                verify_segment_from_device(addr_dev, pe * MESSAGE_LENGTH, MESSAGE_LENGTH, (uint64_t)(pe + 1));
            }
        }
    }

    aclshmem_free(addr_dev);

    aclshmem_team_destroy(team_odd);
    test_finalize(stream, device_id);
}

// ---- roce_sync(team) 测试 ----
static void test_roce_sync_team(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    auto status = test_rdma_init(rank_id, n_ranks, local_mem_size, &stream);
    if (status != 0) { return; }
    ASSERT_NE(stream, nullptr);

    aclshmem_team_t team_odd;
    int start = 1;
    int stride = 2;
    int team_size = n_ranks / 2;
    aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);

    size_t total_size = (size_t)n_ranks * MESSAGE_LENGTH;
    uint8_t *addr_dev = static_cast<uint8_t *>(aclshmem_malloc(total_size));
    ASSERT_NE(addr_dev, nullptr);
    ASSERT_EQ(aclrtMemset(addr_dev, total_size, 0, total_size), 0);

    if (rank_id & 1) {
        fill_segment_to_device(addr_dev, rank_id * MESSAGE_LENGTH, MESSAGE_LENGTH, (uint64_t)(rank_id + 1));
    }

    for (int32_t i = 1; i <= ROCE_SYNC_BARRIER_TEST_NUM; i++) {
        if (rank_id & 1) {
            std::cout << "[TEST] roce_sync_team rank_id: " << rank_id
                      << " msg_len: " << MESSAGE_LENGTH << " time: " << i << std::endl;
            roce_sync_team_data_do(stream, util_get_ffts_config(), addr_dev, rank_id, n_ranks, MESSAGE_LENGTH, team_odd);
            ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

            for (int j = 0; j < team_size; j++) {
                int pe = start + j * stride;
                if (pe == rank_id) continue;
                verify_segment_from_device(addr_dev, pe * MESSAGE_LENGTH, MESSAGE_LENGTH, (uint64_t)(pe + 1));
            }
        }
    }

    aclshmem_free(addr_dev);

    aclshmem_team_destroy(team_odd);
    test_finalize(stream, device_id);
}

// ---- roce_barrier(team) 测试 ----
static void test_roce_barrier_team(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    auto status = test_rdma_init(rank_id, n_ranks, local_mem_size, &stream);
    if (status != 0) { return; }
    ASSERT_NE(stream, nullptr);

    aclshmem_team_t team_odd;
    int start = 1;
    int stride = 2;
    int team_size = n_ranks / 2;
    aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);

    size_t total_size = (size_t)n_ranks * MESSAGE_LENGTH;
    uint8_t *addr_dev = static_cast<uint8_t *>(aclshmem_malloc(total_size));
    ASSERT_NE(addr_dev, nullptr);
    ASSERT_EQ(aclrtMemset(addr_dev, total_size, 0, total_size), 0);

    if (rank_id & 1) {
        fill_segment_to_device(addr_dev, rank_id * MESSAGE_LENGTH, MESSAGE_LENGTH, (uint64_t)(rank_id + 1));
    }

    for (int32_t i = 1; i <= ROCE_SYNC_BARRIER_TEST_NUM; i++) {
        if (rank_id & 1) {
            std::cout << "[TEST] roce_barrier_team rank_id: " << rank_id
                      << " msg_len: " << MESSAGE_LENGTH << " time: " << i << std::endl;
            roce_barrier_team_data_do(stream, util_get_ffts_config(), addr_dev, rank_id, n_ranks, MESSAGE_LENGTH, team_odd);
            ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

            for (int j = 0; j < team_size; j++) {
                int pe = start + j * stride;
                if (pe == rank_id) continue;
                verify_segment_from_device(addr_dev, pe * MESSAGE_LENGTH, MESSAGE_LENGTH, (uint64_t)(pe + 1));
            }
        }
    }

    aclshmem_free(addr_dev);

    aclshmem_team_destroy(team_odd);
    test_finalize(stream, device_id);
}

// ---- roce_barrier(team, buf, sync_id) 测试 ----
static void test_roce_barrier_team_with_buf(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    auto status = test_rdma_init(rank_id, n_ranks, local_mem_size, &stream);
    if (status != 0) { return; }
    ASSERT_NE(stream, nullptr);

    aclshmem_team_t team_odd;
    int start = 1;
    int stride = 2;
    int team_size = n_ranks / 2;
    aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);

    size_t total_size = (size_t)n_ranks * MESSAGE_LENGTH;
    uint8_t *addr_dev = static_cast<uint8_t *>(aclshmem_malloc(total_size));
    ASSERT_NE(addr_dev, nullptr);
    ASSERT_EQ(aclrtMemset(addr_dev, total_size, 0, total_size), 0);

    if (rank_id & 1) {
        fill_segment_to_device(addr_dev, rank_id * MESSAGE_LENGTH, MESSAGE_LENGTH, (uint64_t)(rank_id + 1));
    }

    for (int32_t i = 1; i <= ROCE_SYNC_BARRIER_TEST_NUM; i++) {
        if (rank_id & 1) {
            std::cout << "[TEST] roce_barrier_team_with_buf rank_id: " << rank_id
                      << " msg_len: " << MESSAGE_LENGTH << " time: " << i << std::endl;
            roce_barrier_team_with_buf_data_do(stream, util_get_ffts_config(), addr_dev, rank_id, n_ranks, MESSAGE_LENGTH, team_odd);
            ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

            for (int j = 0; j < team_size; j++) {
                int pe = start + j * stride;
                if (pe == rank_id) continue;
                verify_segment_from_device(addr_dev, pe * MESSAGE_LENGTH, MESSAGE_LENGTH, (uint64_t)(pe + 1));
            }
        }
    }

    aclshmem_free(addr_dev);

    aclshmem_team_destroy(team_odd);
    test_finalize(stream, device_id);
}

// ---- roce_quiet 测试 ----
static void test_roce_quiet(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    auto status = test_rdma_init(rank_id, n_ranks, local_mem_size, &stream);
    if (status != 0) { return; }
    ASSERT_NE(stream, nullptr);

    size_t total_size = (size_t)n_ranks * MESSAGE_LENGTH;
    uint8_t *addr_dev = static_cast<uint8_t *>(aclshmem_malloc(total_size));
    ASSERT_NE(addr_dev, nullptr);
    ASSERT_EQ(aclrtMemset(addr_dev, total_size, 0, total_size), 0);

    if (rank_id == 0) {
        fill_segment_to_device(addr_dev, 0, MESSAGE_LENGTH, (uint64_t)1);
    }

    for (int32_t i = 1; i <= ROCE_SYNC_BARRIER_TEST_NUM; i++) {
        std::cout << "[TEST] roce_quiet rank_id: " << rank_id
                  << " msg_len: " << MESSAGE_LENGTH << " time: " << i << std::endl;
        roce_quiet_data_test_do(stream, util_get_ffts_config(), addr_dev, rank_id, n_ranks, MESSAGE_LENGTH);
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

        if (rank_id == 1) {
            verify_segment_from_device(addr_dev, 0, MESSAGE_LENGTH, (uint64_t)1);
        }
    }

    aclshmem_free(addr_dev);

    test_finalize(stream, device_id);
}

// ---- TEST 宏注册 ----
TEST(TEST_SYNC_BARRIER_API, test_roce_sync_all)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_roce_sync_all, local_mem_size, process_count);
}

TEST(TEST_SYNC_BARRIER_API, test_roce_sync_all_with_buf)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_roce_sync_all_with_buf, local_mem_size, process_count);
}

TEST(TEST_SYNC_BARRIER_API, test_roce_barrier_all)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_roce_barrier_all, local_mem_size, process_count);
}

TEST(TEST_SYNC_BARRIER_API, test_roce_barrier_all_with_buf)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_roce_barrier_all_with_buf, local_mem_size, process_count);
}

TEST(TEST_SYNC_BARRIER_API, test_roce_sync_team)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_roce_sync_team, local_mem_size, process_count);
}

TEST(TEST_SYNC_BARRIER_API, test_roce_sync_team_with_buf)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_roce_sync_team_with_buf, local_mem_size, process_count);
}

TEST(TEST_SYNC_BARRIER_API, test_roce_barrier_team)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_roce_barrier_team, local_mem_size, process_count);
}

TEST(TEST_SYNC_BARRIER_API, test_roce_barrier_team_with_buf)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_roce_barrier_team_with_buf, local_mem_size, process_count);
}

TEST(TEST_SYNC_BARRIER_API, test_roce_quiet)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_roce_quiet, local_mem_size, process_count);
}
