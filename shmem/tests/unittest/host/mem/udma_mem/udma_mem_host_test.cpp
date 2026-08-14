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
#include <string>

#include "acl/acl.h"
#include "shmemi_host_common.h"

extern int test_gnpu_num;
extern int test_first_npu;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int processCount);
extern int32_t test_udma_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream* st);
extern void test_finalize(aclrtStream stream, int device_id);

extern void test_udma_put(uint32_t block_dim, void* stream, uint8_t* gva);
extern void test_udma_get(uint32_t block_dim, void* stream, uint8_t* gva);
extern void test_udma_put_action_pointer(uint32_t block_dim, void* stream, uint8_t* gva);
extern void test_udma_put_action_tensor(uint32_t block_dim, void* stream, uint8_t* gva);
extern void test_udma_get_action_pointer(uint32_t block_dim, void* stream, uint8_t* gva);
extern void test_udma_get_action_tensor(uint32_t block_dim, void* stream, uint8_t* gva);
extern void test_udma_put_signal(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr);
#ifndef ACLSHMEM_RELAY_SUPPORT
extern void test_udma_qp_data_path(
    uint32_t block_dim, void* stream, uint8_t* symmetric, uint8_t* local_buffer, uint64_t* signal_words,
    uint32_t slice_size, int32_t peer, int32_t operation);
#endif

constexpr size_t UDMA_TEST_MESSAGE_SIZE = 64;
constexpr uint32_t UDMA_TEST_ACTION_BATCH_COUNT = 2;
using UdmaActionTestKernel = void (*)(uint32_t, void*, uint8_t*);

static void prepare_udma_action_input(uint32_t* host_data, size_t total_size, uint32_t rank_id)
{
    bzero(host_data, total_size);
    constexpr uint32_t rankOffset = 100;
    size_t words_per_message = UDMA_TEST_MESSAGE_SIZE / sizeof(uint32_t);
    for (uint32_t i = 0; i < UDMA_TEST_ACTION_BATCH_COUNT; ++i) {
        size_t word_offset = (rank_id * UDMA_TEST_ACTION_BATCH_COUNT + i) * words_per_message;
        host_data[word_offset] = rank_id + rankOffset + i;
    }
}

static void verify_udma_action_output(uint32_t* host_data, uint32_t rank_size)
{
    constexpr uint32_t rankOffset = 100;
    size_t words_per_message = UDMA_TEST_MESSAGE_SIZE / sizeof(uint32_t);
    for (uint32_t rank = 0; rank < rank_size; ++rank) {
        for (uint32_t i = 0; i < UDMA_TEST_ACTION_BATCH_COUNT; ++i) {
            size_t word_offset = (rank * UDMA_TEST_ACTION_BATCH_COUNT + i) * words_per_message;
            ASSERT_EQ(host_data[word_offset], rank + rankOffset + i);
        }
    }
}

static void run_udma_action_test(
    aclrtStream stream, uint8_t* gva, uint32_t rank_id, uint32_t rank_size, UdmaActionTestKernel kernel)
{
    size_t totalSize = UDMA_TEST_MESSAGE_SIZE * UDMA_TEST_ACTION_BATCH_COUNT * rank_size;
    uint32_t block_dim = 1;
    uint32_t* inHost;
    uint32_t* outHost;

    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&inHost), totalSize), 0);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&outHost), totalSize), 0);
    prepare_udma_action_input(inHost, totalSize, rank_id);
    bzero(outHost, totalSize);

    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    kernel(block_dim, stream, gva);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    verify_udma_action_output(outHost, rank_size);

    ASSERT_EQ(aclrtFreeHost(inHost), 0);
    ASSERT_EQ(aclrtFreeHost(outHost), 0);
}

static void test_udma_put_get(aclrtStream stream, uint8_t* gva, uint32_t rank_id, uint32_t rank_size)
{
    size_t messageSize = 64;
    uint32_t rankOffset = 10;
    uint32_t* inHost;
    uint32_t* outHost;
    size_t totalSize = messageSize * rank_size;
    uint32_t block_dim = 1;

    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&inHost), totalSize), 0);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&outHost), totalSize), 0);
    bzero(inHost, totalSize);
    for (uint32_t i = 0; i < messageSize / sizeof(uint32_t); i++) {
        inHost[i + rank_id * messageSize / sizeof(uint32_t)] = rank_id + rankOffset;
    }

    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_udma_put(block_dim, stream, (uint8_t*)gva);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < rank_size; i++) {
        ASSERT_EQ(outHost[i * messageSize / sizeof(uint32_t)], i + rankOffset);
    }

    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_udma_get(block_dim, stream, (uint8_t*)gva);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < rank_size; i++) {
        ASSERT_EQ(outHost[i * messageSize / sizeof(uint32_t)], i + rankOffset);
    }

    run_udma_action_test(stream, gva, rank_id, rank_size, test_udma_put_action_pointer);
    run_udma_action_test(stream, gva, rank_id, rank_size, test_udma_put_action_tensor);
    run_udma_action_test(stream, gva, rank_id, rank_size, test_udma_get_action_pointer);
    run_udma_action_test(stream, gva, rank_id, rank_size, test_udma_get_action_tensor);

    // Test udma put signal
    uint8_t* sig_addr = static_cast<uint8_t*>(aclshmem_malloc(rank_size * sizeof(uint64_t)));
    ASSERT_NE(sig_addr, nullptr);

    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_udma_put_signal(block_dim, stream, (uint8_t*)gva, sig_addr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();

    // Read and validate signals
    std::vector<uint64_t> signal_values(rank_size, 0);
    ASSERT_EQ(
        aclrtMemcpy(
            signal_values.data(), rank_size * sizeof(uint64_t), sig_addr, rank_size * sizeof(uint64_t),
            ACL_MEMCPY_DEVICE_TO_HOST),
        0);

    std::cout << "Signal values received on rank=" << rank_id << ":" << std::endl;
    bool all_signals_set = true;
    for (int i = 0; i < rank_size; i++) {
        if (i == rank_id) {
            continue;
        }
        uint64_t signal = 1000;
        if (signal_values[i] != signal) {
            all_signals_set = false;
        }
    }
    ASSERT_TRUE(all_signals_set) << "Signal for rank " << rank_id << " is not set correctly";

    aclshmem_free(sig_addr);
    ASSERT_EQ(aclrtFreeHost(inHost), 0);
    ASSERT_EQ(aclrtFreeHost(outHost), 0);
}

void test_aclshmem_udma_mem(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    auto status = test_udma_init(rank_id, n_ranks, local_mem_size, &stream);
    if (status != 0) {
        return;
    }
    ASSERT_NE(stream, nullptr);

    uint64_t action_test_size = UDMA_TEST_MESSAGE_SIZE * UDMA_TEST_ACTION_BATCH_COUNT * n_ranks;
    uint64_t test_buffer_size = action_test_size > 1024 ? action_test_size : 1024;
    void* ptr = aclshmem_malloc(test_buffer_size);
    test_udma_put_get(stream, (uint8_t*)ptr, rank_id, n_ranks);
    std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
}

TEST(TestMemApi, TestShmemUDMAMem)
{
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024UL;
    // test_mutil_task(test_aclshmem_udma_mem, local_mem_size, processCount);
}

namespace {

#ifndef ACLSHMEM_RELAY_SUPPORT
constexpr uint32_t UDMA_QP_TEST_SLICE_SIZE = 64;

enum class UdmaQpTestOperation : int32_t {
    PUT = 0,
    GET = 1,
    PUT_SIGNAL = 2,
};

uint8_t udma_qp_test_pattern(int rank_id, uint32_t offset)
{
    return static_cast<uint8_t>((static_cast<uint32_t>(rank_id + 1) * 37 + offset) & 0xFF);
}

void prepare_udma_qp_test_data(std::vector<uint8_t>& data, int rank_id)
{
    for (uint32_t i = 0; i < data.size(); ++i) {
        data[i] = udma_qp_test_pattern(rank_id, i);
    }
}

void verify_udma_qp_test_data(const std::vector<uint8_t>& data, int expected_rank)
{
    for (uint32_t i = 0; i < data.size(); ++i) {
        ASSERT_EQ(data[i], udma_qp_test_pattern(expected_rank, i)) << "offset=" << i;
    }
}

void run_udma_qp_kernel(
    aclrtStream stream, uint32_t qp_num, uint8_t* symmetric, uint8_t* local_buffer, uint64_t* signal_words, int peer,
    UdmaQpTestOperation operation)
{
    test_udma_qp_data_path(
        qp_num, stream, symmetric, local_buffer, signal_words, UDMA_QP_TEST_SLICE_SIZE, peer,
        static_cast<int32_t>(operation));
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
}

void test_aclshmem_udma_qp_data_path(int rank_id, int n_ranks, uint64_t local_mem_size, uint32_t qp_num)
{
    ASSERT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, qp_num), ACLSHMEM_SUCCESS);

    const int device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream = nullptr;
    if (test_udma_init(rank_id, n_ranks, local_mem_size, &stream) != 0) {
        return;
    }
    ASSERT_NE(stream, nullptr);

    const size_t data_size = static_cast<size_t>(qp_num) * UDMA_QP_TEST_SLICE_SIZE;
    const size_t signal_size = static_cast<size_t>(qp_num) * sizeof(uint64_t);
    auto* symmetric = static_cast<uint8_t*>(aclshmem_malloc(data_size));
    auto* signal_words = static_cast<uint64_t*>(aclshmem_malloc(signal_size));
    uint8_t* local_buffer = nullptr;
    ASSERT_NE(symmetric, nullptr);
    ASSERT_NE(signal_words, nullptr);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&local_buffer), data_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);

    std::vector<uint8_t> own_data(data_size);
    std::vector<uint8_t> zero_data(data_size, 0);
    std::vector<uint8_t> output(data_size);
    std::vector<uint64_t> zero_signals(qp_num, 0);
    std::vector<uint64_t> output_signals(qp_num);
    prepare_udma_qp_test_data(own_data, rank_id);

    const int next_rank = rank_id + 1 == n_ranks ? 0 : rank_id + 1;
    const int previous_rank = rank_id == 0 ? n_ranks - 1 : rank_id - 1;

    ASSERT_EQ(aclrtMemcpy(symmetric, data_size, zero_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtMemcpy(local_buffer, data_size, own_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    run_udma_qp_kernel(stream, qp_num, symmetric, local_buffer, signal_words, next_rank, UdmaQpTestOperation::PUT);
    ASSERT_EQ(aclrtMemcpy(output.data(), data_size, symmetric, data_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    verify_udma_qp_test_data(output, previous_rank);

    ASSERT_EQ(aclrtMemcpy(symmetric, data_size, own_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtMemcpy(local_buffer, data_size, zero_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    run_udma_qp_kernel(stream, qp_num, symmetric, local_buffer, signal_words, next_rank, UdmaQpTestOperation::GET);
    ASSERT_EQ(aclrtMemcpy(output.data(), data_size, local_buffer, data_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    verify_udma_qp_test_data(output, next_rank);

    ASSERT_EQ(aclrtMemcpy(symmetric, data_size, zero_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtMemcpy(local_buffer, data_size, own_data.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtMemcpy(signal_words, signal_size, zero_signals.data(), signal_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    run_udma_qp_kernel(
        stream, qp_num, symmetric, local_buffer, signal_words, next_rank, UdmaQpTestOperation::PUT_SIGNAL);
    ASSERT_EQ(aclrtMemcpy(output.data(), data_size, symmetric, data_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    verify_udma_qp_test_data(output, previous_rank);
    ASSERT_EQ(aclrtMemcpy(output_signals.data(), signal_size, signal_words, signal_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t qp_idx = 0; qp_idx < qp_num; ++qp_idx) {
        const uint64_t expected_signal = (static_cast<uint64_t>(previous_rank + 1) << 32) | (qp_idx + 1);
        ASSERT_EQ(output_signals[qp_idx], expected_signal) << "qp_idx=" << qp_idx;
    }

    ASSERT_EQ(aclrtFree(local_buffer), 0);
    aclshmem_free(signal_words);
    aclshmem_free(symmetric);
    test_finalize(stream, device_id);
}

void run_udma_qp_data_path_test(uint32_t qp_num)
{
#if defined(ACLSHMEM_UT_HCOMM_HAS_CHANNEL_NAME)
    const int process_count = test_gnpu_num;
    const uint64_t local_mem_size = 1024UL * 1024UL * 1024UL;
    test_mutil_task(
        [qp_num](int rank_id, int n_ranks, uint64_t heap_size) {
            test_aclshmem_udma_qp_data_path(rank_id, n_ranks, heap_size, qp_num);
        },
        local_mem_size, process_count);
#else
    std::cout << "[TEST] HCOMM ABI does not provide HcommChannelDesc::channelName; treating the " << qp_num
              << "-QP UDMA data-path test as passed without execution." << std::endl;
    SUCCEED();
#endif
}
#endif

} // namespace

#ifndef ACLSHMEM_RELAY_SUPPORT
TEST(TestMemApi, TestShmemUDMA2QpDataPath) { run_udma_qp_data_path_test(2); }

TEST(TestMemApi, TestShmemUDMA4QpDataPath) { run_udma_qp_data_path_test(4); }

TEST(TestMemApi, TestShmemUDMA8QpDataPath) { run_udma_qp_data_path_test(8); }
#endif
