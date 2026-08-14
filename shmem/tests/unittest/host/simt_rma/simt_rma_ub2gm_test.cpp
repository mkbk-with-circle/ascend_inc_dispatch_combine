/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file simt_rma_ub2gm_test.cpp
 * @brief SIMT RMA ub2gm unit tests host implementation.
 */

#if defined(USE_SIMT)

#include <cstring>
#include <iostream>
#include <string>
#include <tuple>
#include <variant>
#include <vector>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "simt_rma_ub2gm_definitions.h"
#include "unittest_main_test.h"

using InputParams = std::tuple<void*, void*, size_t, int32_t>; // void *dst, void *src, size_t elem_size, int32_t pe

struct InputParamsTransform {
    InputParams operator()(TypedSig sig, InputParams input_params)
    {
        auto [dst, src, elem_size, pe] = input_params;
        return InputParams(dst, src, elem_size / get_type_size(sig.name), pe);
    }

    InputParams operator()(SizedSig sig, InputParams input_params)
    {
        auto [dst, src, elem_size, pe] = input_params;
        return InputParams(dst, src, elem_size / get_bits_size(sig.bits), pe);
    }

    InputParams operator()(MemorySig sig, InputParams input_params)
    {
        return input_params;
    }
};

static void init_mem(void* mem, size_t size, int8_t pe)
{
    int8_t* mem_ptr = static_cast<int8_t*>(mem);
    for (size_t i = 0; i < size; ++i) {
        mem_ptr[i] = pe;
    }
}

static bool check_mem(void* mem, size_t size, int8_t target_pe)
{
    int8_t* mem_ptr = static_cast<int8_t*>(mem);
    for (size_t i = 0; i < size; ++i) {
        if (mem_ptr[i] != target_pe) {
            return false;
        }
    }
    return true;
}

static int32_t get_peer_pe(int32_t my_pe, int32_t n_pes)
{
    return (my_pe + 1) % n_pes;
}

static int32_t get_checking_pe(Sigs sig)
{
    OpType op = std::visit([](const auto& s) { return s.op; }, sig);
    if (op == OpType::GET) {
        return 0;
    }
    return 1;
}

static inline void print_array_bytes(const void* arr, size_t len, const std::string& prefix = "")
{
    const unsigned char* p = static_cast<const unsigned char*>(arr);
    std::cout << prefix;
    for (size_t i = 0; i < len; ++i) {
        if (i) {
            std::cout << ' ';
        }
        std::cout << static_cast<int>(p[i]);
    }
    std::cout << '\n';
}

static void invoke_test(FunctionInterface* func, Sigs sig, InputParams input_params, aclrtStream stream)
{
    auto input_param = std::visit([&](auto&& s) {
        return InputParamsTransform()(s, input_params);
    }, sig);
    auto [dst, src, elem_size, pe] = input_param;
    func(stream, dst, src, elem_size, pe);
}

static inline std::ostream& operator<<(std::ostream& os, const TestCaseEntry& t)
{
    os << "{func, ";
    const Sigs& sig = std::get<1>(t);
    std::visit([&os](auto const& s) { os << s; }, sig);
    os << "}";
    return os;
}

class SimtRMAUb2GmTest : public ::testing::TestWithParam<TestCaseEntry> {};

static TestCaseEntry g_current_test_case_entry = Entries[0];

void test_simt_rma_ub2gm(int my_pe, int n_pes, uint64_t local_mem_size)
{
    aclrtStream stream = nullptr;
    int32_t device_id = my_pe % test_gnpu_num + test_first_npu;
    test_init(my_pe, n_pes, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    constexpr int32_t mem_size = 4096 * sizeof(int8_t); // bytes

    void* host_mem = nullptr;
    void* device_mem = nullptr;

    ASSERT_EQ(aclrtMallocHost(&host_mem, mem_size), ACL_SUCCESS);
    device_mem = aclshmem_malloc(mem_size);
    ASSERT_NE(device_mem, nullptr);

    init_mem(host_mem, mem_size, my_pe);
    ASSERT_EQ(aclrtMemcpy(device_mem, mem_size, host_mem, mem_size, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    auto [func, sig] = g_current_test_case_entry;
    InputParams input_params = std::make_tuple(device_mem, device_mem, mem_size, 1);

    aclshmem_barrier_all();
    if (my_pe == 0) {
        invoke_test(func, sig, input_params, stream);
    }

    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmem_barrier_all();
    EXPECT_EQ(aclrtMemcpy(host_mem, mem_size, device_mem, mem_size, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);

    if (my_pe == get_checking_pe(sig)) {
        EXPECT_TRUE(check_mem(host_mem, mem_size, get_peer_pe(my_pe, n_pes)));
    }

    aclshmem_free(device_mem);
    aclrtFreeHost(host_mem);

    test_finalize(stream, device_id);
}

TEST_P(SimtRMAUb2GmTest, SimtRMAUb2GmTest)
{
    constexpr int32_t process_count = 2;
    constexpr uint64_t local_mem_size = 1024UL * 1024UL * 1024UL;
    g_current_test_case_entry = GetParam();
    test_mutil_task(test_simt_rma_ub2gm, local_mem_size, process_count);
}

INSTANTIATE_TEST_SUITE_P(SimtRMAUb2GmTestCases, SimtRMAUb2GmTest, ::testing::ValuesIn(Entries));

#endif // defined(USE_SIMT)
