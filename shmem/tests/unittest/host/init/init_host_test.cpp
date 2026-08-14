/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <acl/acl.h>
#include <netinet/in.h>
#include <errno.h>
#include <unistd.h>
#include <algorithm>
#include <gtest/gtest.h>
#include "shmemi_host_common.h"
#include "utils/exception/shmem_exception_report.h"
#include "host/init/shmem_host_init.h"
#include "host/shmem_host_def.h"
#include "init/shmemi_init.h"
#include "unittest_main_test.h"
#include "unittest/utils/scope_env.h"

namespace shm {
constexpr uint32_t timeout = 50;
void logger_test_example(int level, const char* msg)
{
    // do print here
}

void exception_report_test_callback(void* exception_info) { (void)exception_info; }
} // namespace shm

static_assert(sizeof(aclshmem_init_optional_attr_t) == 24, "Unexpected init optional attr ABI change");

void test_aclshmem_init(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 2), ACLSHMEM_NOT_SUPPORTED);
    EXPECT_EQ(g_state.mype, rank_id);
    EXPECT_EQ(g_state.npes, n_ranks);
    EXPECT_NE(g_state.heap_base, nullptr);
    EXPECT_NE(g_state.p2p_device_heap_base[rank_id], nullptr);
    EXPECT_EQ(g_state.heap_size, local_mem_size + ACLSHMEM_EXTRA_SIZE);
    EXPECT_NE(g_state.team_pools[0], nullptr);
    status = aclshmemx_init_status();
    EXPECT_EQ(status, ACLSHMEM_STATUS_IS_INITIALIZED);
    status = aclshmem_finalize();
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 2), ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_qp_reconfiguration_after_last_finalize(int rank_id, int /*n_ranks*/, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(aclInit(nullptr), ACL_SUCCESS);
    EXPECT_EQ(aclrtSetDevice(device_id), ACL_SUCCESS);
    EXPECT_EQ(aclshmemx_set_conf_store_tls(false, nullptr, 0), ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 2), ACLSHMEM_SUCCESS);

    aclshmemx_init_attr_t attributes;
    test_set_attr(0, 1, local_mem_size, test_global_ipport, &attributes);
    EXPECT_EQ(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes), ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 4), ACLSHMEM_NOT_SUPPORTED);
    EXPECT_EQ(aclshmem_finalize(), ACLSHMEM_SUCCESS);

    EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 4), ACLSHMEM_SUCCESS);
    test_set_attr(0, 1, local_mem_size, test_global_ipport, &attributes);
    EXPECT_EQ(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes), ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 2), ACLSHMEM_NOT_SUPPORTED);
    EXPECT_EQ(aclshmem_finalize(), ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 2), ACLSHMEM_SUCCESS);

    EXPECT_EQ(aclrtResetDevice(device_id), ACL_SUCCESS);
    EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
}

void test_aclshmem_init_invalid_rank_id(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int erank_id = -1;
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);

    aclshmemx_init_attr_t attributes;
    test_set_attr(erank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);

    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    EXPECT_EQ(status, ACLSHMEM_INVALID_VALUE);
    status = aclshmemx_init_status();
    EXPECT_EQ(status, ACLSHMEM_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 2), ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_aclshmem_init_invalid_n_ranks(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int en_ranks = ACLSHMEM_MAX_PES + 1;
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    aclshmemx_uniqueid_t uid;

    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, en_ranks, local_mem_size, test_global_ipport, &attributes);

    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    EXPECT_EQ(status, ACLSHMEM_INVALID_VALUE);
    status = aclshmemx_init_status();
    EXPECT_EQ(status, ACLSHMEM_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_aclshmem_init_rank_id_over_size(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);

    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id + n_ranks, n_ranks, local_mem_size, test_global_ipport, &attributes);

    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_INVALID_PARAM);
    status = aclshmemx_init_status();
    EXPECT_EQ(status, ACLSHMEM_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_aclshmem_init_zero_mem(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    // local_mem_size = 0
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);

    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_INVALID_VALUE);
    status = aclshmemx_init_status();
    EXPECT_EQ(status, ACLSHMEM_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_shmem_init_huge_pool(int pe_id, int n_pes, uint64_t local_mem_size)
{
    uint32_t device_id = pe_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    size_t hbm_free, hbm_total;
    EXPECT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &hbm_free, &hbm_total), 0);
    if (local_mem_size > hbm_total) {
        constexpr uint64_t hbm_ratio_percent = 60;
        local_mem_size = ALIGN_UP(hbm_total * hbm_ratio_percent / 100, 1024ULL * 1024 * 1024);
    }
    shmem_set_conf_store_tls(false, nullptr, 0);
    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, test_global_ipport, &attributes);

    status = shmem_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(g_state.mype, pe_id);
    EXPECT_EQ(g_state.npes, n_pes);
    EXPECT_NE(g_state.heap_base, nullptr);
    EXPECT_NE(g_state.p2p_device_heap_base[pe_id], nullptr);
    EXPECT_EQ(g_state.heap_size, local_mem_size + ACLSHMEM_EXTRA_SIZE);
    EXPECT_NE(g_state.team_pools[0], nullptr);
    status = shmem_init_status();
    EXPECT_EQ(status, ACLSHMEM_STATUS_IS_INITIALIZED);
    status = shmem_finalize();
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_shmem_init(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    shmem_set_conf_store_tls(false, nullptr, 0);
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);

    status = shmem_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(g_state.mype, rank_id);
    EXPECT_EQ(g_state.npes, n_ranks);
    EXPECT_NE(g_state.heap_base, nullptr);
    EXPECT_NE(g_state.p2p_device_heap_base[rank_id], nullptr);
    EXPECT_EQ(g_state.heap_size, local_mem_size + ACLSHMEM_EXTRA_SIZE);
    EXPECT_NE(g_state.team_pools[0], nullptr);
    status = shmem_init_status();
    EXPECT_EQ(status, ACLSHMEM_STATUS_IS_INITIALIZED);
    status = shmem_finalize();
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}
void aclshmem_test_logger(int level, const char* msg)
{
    if (level < 1) {
        return;
    }
    if (msg == NULL) {
        printf("extern log set :(null)\n");
        return;
    }
    const char* level_name = "unknown";
    switch (level) {
        case 0:
            level_name = "debug";
            break;
        case 1:
            level_name = "info";
            break;
        case 2:
            level_name = "warn";
            break;
        case 3:
            level_name = "error";
            break;
        case 4:
            level_name = "fatal";
            break;
        default:
            break;
    }

    printf("extern log set [%s]: %s\n", level_name, msg);
}

void test_aclshmem_extern_logger(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    EXPECT_EQ(aclshmemx_set_extern_logger(aclshmem_test_logger), ACLSHMEM_SUCCESS);
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(g_state.mype, rank_id);
    EXPECT_EQ(g_state.npes, n_ranks);
    EXPECT_NE(g_state.heap_base, nullptr);
    EXPECT_NE(g_state.p2p_device_heap_base[rank_id], nullptr);
    EXPECT_EQ(g_state.heap_size, local_mem_size + ACLSHMEM_EXTRA_SIZE);
    EXPECT_NE(g_state.team_pools[0], nullptr);
    status = aclshmemx_init_status();
    EXPECT_EQ(status, ACLSHMEM_STATUS_IS_INITIALIZED);
    status = aclshmem_finalize();
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()) {
        exit(1);
    }
}

void test_aclshmem_init_invalid_ip(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, "tcp://123.45.67.89:2345", &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_INNER_ERROR);
    status = aclshmem_finalize();
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_aclshmem_repeatedly_init(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_INNER_ERROR);
    status = aclshmem_finalize();
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    sleep(2);
    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(g_state.mype, rank_id);
    EXPECT_EQ(g_state.npes, n_ranks);
    EXPECT_NE(g_state.heap_base, nullptr);
    EXPECT_NE(g_state.p2p_device_heap_base[rank_id], nullptr);
    EXPECT_EQ(g_state.heap_size, local_mem_size + ACLSHMEM_EXTRA_SIZE);
    EXPECT_NE(g_state.team_pools[0], nullptr);
    status = aclshmemx_init_status();
    EXPECT_EQ(status, ACLSHMEM_STATUS_IS_INITIALIZED);
    status = aclshmem_finalize();
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

std::vector<std::string> split_env(const std::string& s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

void test_aclshmem_init_cant_access(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    const std::string env_name = "ASCEND_RT_VISIBLE_DEVICES";
    std::string original_value;
    bool is_original_empty = false;
    const char* original_env_value = std::getenv(env_name.c_str());
    std::string target_device;
    if (original_env_value) {
        original_value = original_env_value;
        std::vector<std::string> devices = split_env(original_value, ',');
        if (devices.size() >= n_ranks) {
            target_device = devices[rank_id];
        } else {
            return;
        }
    } else {
        target_device = std::to_string(rank_id);
        is_original_empty = true;
    }
    setenv(env_name.c_str(), target_device.c_str(), 1);

    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(0), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_NE(g_state.heap_base, nullptr);
    status = aclshmem_finalize();
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(0), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (is_original_empty) {
        unsetenv(env_name.c_str());
    } else {
        setenv(env_name.c_str(), original_value.c_str(), 1);
    }
}

// =============================== Multi-Instance UT ===============================
// 多实例初始化公共逻辑：
// 返回 0 表示当前 pe 在 dev_list 中且初始化成功; 返回 -1 表示当前 pe 不参与该实例
static int multi_instance_init(
    int rank_id, uint64_t local_mem_size, uint64_t instance_id, std::vector<int>& dev_list, aclshmemx_init_attr_t& attr,
    bool& joined)
{
    auto it = std::find(dev_list.begin(), dev_list.end(), rank_id);
    if (it == dev_list.end()) {
        joined = false;
        return ACLSHMEM_SUCCESS; // 当前 pe 不在该实例的设备列表中, 不做初始化
    }
    joined = true;

    // 默认模式下 port 必须为 0
    const char* ipport = "tcp://127.0.0.1:0";
    int local_pe_id = static_cast<int>(std::distance(dev_list.begin(), it));

    int status = test_set_attr(local_pe_id, static_cast<int>(dev_list.size()), local_mem_size, ipport, &attr);
    // 多实例默认模式要求 comm_args 为 nullptr
    attr.instance_id = instance_id;
    attr.comm_args = nullptr;
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
    return status;
}

// 全部 rank 单实例创建 -> 校验状态 -> 销毁
void test_aclshmem_multi_instance_single(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);

    uint64_t INSTANCE_ID = 1;
    aclshmemx_init_attr_t attr;
    std::vector<int> dev_list;
    for (int i = 0; i < n_ranks; ++i) {
        dev_list.push_back(i); // {0,1,...,n_ranks-1}
    }

    bool joined = false;
    int ret = multi_instance_init(rank_id, local_mem_size, INSTANCE_ID, dev_list, attr, joined);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);

    if (joined) {
        EXPECT_EQ(aclshmemx_init_status(), ACLSHMEM_STATUS_IS_INITIALIZED);
        EXPECT_EQ(attr.instance_id, INSTANCE_ID);
        EXPECT_EQ(aclshmemx_finalize(attr.instance_id), ACLSHMEM_SUCCESS);
    }

    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

// 实例内 malloc/free (设备子集 {max(2, n_ranks/2)})
void test_aclshmem_multi_instance_malloc(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);

    uint64_t INSTANCE_ID = 1;
    aclshmemx_init_attr_t attr;
    int subset = std::max(2, n_ranks / 2); // n_ranks=2->2, 4->2, 8->4
    std::vector<int> dev_list;
    for (int i = 0; i < subset; ++i)
        dev_list.push_back(i);

    bool joined = false;
    int ret = multi_instance_init(rank_id, local_mem_size, INSTANCE_ID, dev_list, attr, joined);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);

    if (joined) {
        void* ptr = aclshmem_malloc(1024);
        EXPECT_NE(ptr, nullptr);
        std::vector<uint64_t> host_buf(1024 / sizeof(uint64_t), INSTANCE_ID);
        EXPECT_EQ(aclrtMemcpy(ptr, 1024, host_buf.data(), 1024, ACL_MEMCPY_HOST_TO_DEVICE), 0);
        aclshmem_free(ptr);

        EXPECT_EQ(aclshmemx_finalize(attr.instance_id), ACLSHMEM_SUCCESS);
    }

    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

// 同一进程同时归属两个设备重叠的实例
// instance 2 -> 首尾, instance 4 -> 末两个, pe n_ranks-1 同时属于这两个实例
void test_aclshmem_multi_instance_overlap(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    // 若总rank小于1, 不执行该UT
    if (n_ranks <= 1) {
        return;
    }

    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);

    uint64_t INSTANCE_ID2 = 2;
    aclshmemx_init_attr_t inst2_attr;
    std::vector<int> inst2_dev = {0, n_ranks - 1}; // 首 + 尾
    bool joined2 = false;
    int ret2 = multi_instance_init(rank_id, local_mem_size, INSTANCE_ID2, inst2_dev, inst2_attr, joined2);
    EXPECT_EQ(ret2, ACLSHMEM_SUCCESS);
    if (joined2) {
        EXPECT_EQ(inst2_attr.instance_id, INSTANCE_ID2);
    }

    uint64_t INSTANCE_ID4 = 4;
    aclshmemx_init_attr_t inst4_attr;
    std::vector<int> inst4_dev = {n_ranks - 2, n_ranks - 1}; // 末尾两个
    bool joined4 = false;
    int ret4 = multi_instance_init(rank_id, local_mem_size, INSTANCE_ID4, inst4_dev, inst4_attr, joined4);
    EXPECT_EQ(ret4, ACLSHMEM_SUCCESS);
    if (joined4) {
        EXPECT_EQ(inst4_attr.instance_id, INSTANCE_ID4);
    }

    // pe(n_ranks-1) 此时同时持有 instance 2 与 instance 4，分别做一次内存操作后再销毁
    if (joined2) {
        // 需要先切换context为instance 2
        int status = aclshmemx_instance_ctx_set(INSTANCE_ID2);
        EXPECT_EQ(status, ACLSHMEM_SUCCESS);

        void* ptr = aclshmem_malloc(1024);
        EXPECT_NE(ptr, nullptr);
        aclshmem_free(ptr);
        EXPECT_EQ(aclshmemx_finalize(inst2_attr.instance_id), ACLSHMEM_SUCCESS);
    }

    if (joined4) {
        // 需要先切换context为instance 4
        int status = aclshmemx_instance_ctx_set(INSTANCE_ID4);
        EXPECT_EQ(status, ACLSHMEM_SUCCESS);

        void* ptr = aclshmem_malloc(1024);
        EXPECT_NE(ptr, nullptr);
        aclshmem_free(ptr);
        EXPECT_EQ(aclshmemx_finalize(inst4_attr.instance_id), ACLSHMEM_SUCCESS);
    }

    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

// 错误用例 —— 多实例下传入非法 instance_id（0 视为非法 / 未占用）
void test_aclshmem_multi_instance_invalid_id(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);

    aclshmemx_init_attr_t attr;
    std::vector<int> dev_list;
    for (int i = 0; i < n_ranks; ++i)
        dev_list.push_back(i); // {0,1,...,n_ranks-1}
    auto it = std::find(dev_list.begin(), dev_list.end(), rank_id);
    if (it != dev_list.end()) {
        const char* ipport = "tcp://127.0.0.1:0";
        int local_pe_id = static_cast<int>(std::distance(dev_list.begin(), it));
        attr.instance_id = 1025; // 非法 instance_id
        test_set_attr(local_pe_id, static_cast<int>(dev_list.size()), local_mem_size, ipport, &attr);
        attr.comm_args = nullptr;
        int status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
        EXPECT_EQ(status, ACLSHMEM_INVALID_VALUE);
        EXPECT_EQ(aclshmemx_init_status(), ACLSHMEM_STATUS_NOT_INITIALIZED);
    }

    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

TEST(TestInitAPI, TestSetUdmaQpNumBeforeInit)
{
    EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_ROCE, 1), ACLSHMEM_NOT_SUPPORTED);

    for (uint32_t configured_qp_num : {1U, 2U, 4U, 8U, ACLSHMEM_MAX_QP_NUM}) {
        EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, configured_qp_num), ACLSHMEM_SUCCESS);
    }

    EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 0), ACLSHMEM_INVALID_VALUE);
    EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, ACLSHMEM_MAX_QP_NUM + 1), ACLSHMEM_INVALID_VALUE);
    EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 1), ACLSHMEM_SUCCESS);
}

TEST(TestInitAPI, TestShmemInit)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_init, local_mem_size, process_count);
}

TEST(TestInitAPI, TestQpReconfigurationAfterLastFinalize)
{
    constexpr int process_count = 1;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_qp_reconfiguration_after_last_finalize, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInitHugePool)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 38 * 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_init_huge_pool, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInitErrorInvalidRankId)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_init_invalid_rank_id, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInitErrorInvalidNRanks)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_init_invalid_n_ranks, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInitErrorRankIdOversize)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_init_rank_id_over_size, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInitErrorZeroMem)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 0;
    test_mutil_task(test_aclshmem_init_zero_mem, local_mem_size, process_count);
}

TEST(TestInitAPI, TestInfoGetVersion)
{
    int major = 0;
    int minor = 0;
    aclshmem_info_get_version(&major, &minor);
    EXPECT_EQ(major, ACLSHMEM_MAJOR_VERSION);
    EXPECT_EQ(minor, ACLSHMEM_MINOR_VERSION);
}

TEST(TestInitAPI, TestInfoGetVersionNull)
{
    int major = 0;
    aclshmem_info_get_version(&major, nullptr);
    EXPECT_EQ(major, 0);
}

TEST(TestInitAPI, TestInfoGetName)
{
    char name[256];
    aclshmem_info_get_name(name);
    EXPECT_TRUE(strlen(name) > 0);

    std::string expect = "ACLSHMEM v" + std::to_string(ACLSHMEM_VENDOR_MAJOR_VER) + "." +
                         std::to_string(ACLSHMEM_VENDOR_MINOR_VER) + "." +
                         std::to_string(ACLSHMEM_VENDOR_PATCH_VER).c_str();
    for (size_t i = 0; i < expect.length(); i++) {
        EXPECT_EQ(expect[i], name[i]);
    }
}

TEST(TestInitAPI, TestInfoGetNameNull)
{
    char* input = nullptr;
    aclshmem_info_get_name(input);
    EXPECT_EQ(input, nullptr);
}

TEST(TestInitAPI, TestShmemSetLogLevel)
{
    auto ret = aclshmemx_set_log_level(aclshmem_log::DEBUG_LEVEL);
    EXPECT_EQ(ret, 0);

    char* original_log_level = NULL;
    const char* env_val = getenv("SHMEM_LOG_LEVEL");
    if (env_val != NULL) {
        original_log_level = strdup(env_val);
    }

    setenv("SHMEM_LOG_LEVEL", "DEBUG", 1);
    EXPECT_EQ(aclshmemx_set_log_level(-1), 0);

    setenv("SHMEM_LOG_LEVEL", "INFO", 1);
    EXPECT_EQ(aclshmemx_set_log_level(-1), 0);

    setenv("SHMEM_LOG_LEVEL", "WARN", 1);
    EXPECT_EQ(aclshmemx_set_log_level(-1), 0);

    setenv("SHMEM_LOG_LEVEL", "ERROR", 1);
    EXPECT_EQ(aclshmemx_set_log_level(-1), 0);

    setenv("SHMEM_LOG_LEVEL", "FATAL", 1);
    EXPECT_EQ(aclshmemx_set_log_level(-1), 0);

    unsetenv("SHMEM_LOG_LEVEL");
    if (original_log_level != NULL) {
        setenv("SHMEM_LOG_LEVEL", original_log_level, 1);
        free(original_log_level);
        original_log_level = NULL;
    }
}

TEST(TestExceptionReportAPI, TestReportNoPending)
{
    EXPECT_FALSE(aclshmemi_exception_report_pending());
    EXPECT_EQ(aclshmemx_report_exception(), ACLSHMEM_SUCCESS);
}

TEST(TestExceptionReportAPI, TestEnableReport)
{
    EXPECT_EQ(aclshmemx_enable_exception_report(nullptr, ACLSHMEMX_EXCEPTION_REPORT_INFO), ACLSHMEM_SUCCESS);
    EXPECT_FALSE(aclshmemi_exception_report_pending());
    aclshmemi_exception_report_finalize();

    EXPECT_EQ(
        aclshmemx_enable_exception_report(shm::exception_report_test_callback, ACLSHMEMX_EXCEPTION_REPORT_DEBUG),
        ACLSHMEM_SUCCESS);
    EXPECT_FALSE(aclshmemi_exception_report_pending());
    aclshmemi_exception_report_finalize();

    EXPECT_EQ(
        aclshmemx_enable_exception_report(
            nullptr, static_cast<aclshmemx_exception_report_level_t>(ACLSHMEMX_EXCEPTION_REPORT_DEBUG + 1)),
        ACLSHMEM_INVALID_PARAM);
}

TEST(TestExceptionReportAPI, TestReportSnapshotWatermark)
{
    aclshmemi_exception_report_finalize();
    EXPECT_FALSE(aclshmemi_exception_report_record_snapshot(1, 2, 3, 4, 5));
    EXPECT_FALSE(aclshmemi_exception_report_pending());

    EXPECT_EQ(aclshmemx_enable_exception_report(nullptr, ACLSHMEMX_EXCEPTION_REPORT_INFO), ACLSHMEM_SUCCESS);
    EXPECT_TRUE(aclshmemi_exception_report_record_snapshot(10, 20, 30, 40, 50));
    EXPECT_TRUE(aclshmemi_exception_report_pending());
    EXPECT_EQ(aclshmemx_report_exception(), ACLSHMEM_SUCCESS);
    EXPECT_FALSE(aclshmemi_exception_report_pending());

    EXPECT_TRUE(aclshmemi_exception_report_record_snapshot(11, 21, 31, 41, 51));
    EXPECT_TRUE(aclshmemi_exception_report_pending());
    aclshmemi_exception_report_finalize();
}

TEST(TestExceptionReportAPI, TestReportKeepsUdmaReporterForCompositeMasks)
{
    aclshmemi_exception_report_finalize();
    const int old_npes = g_state.npes;
    g_state.npes = 2;

    const data_op_engine_type_t masks_with_udma[] = {
        ACLSHMEM_DATA_OP_UDMA,
        static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_UDMA | ACLSHMEM_DATA_OP_SDMA),
        static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_UDMA | ACLSHMEM_DATA_OP_ROCE),
        static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_UDMA | ACLSHMEM_DATA_OP_SDMA | ACLSHMEM_DATA_OP_ROCE),
    };

    for (auto mask : masks_with_udma) {
        EXPECT_EQ(aclshmemi_exception_report_apply_deferred_config(mask), ACLSHMEM_SUCCESS);
        aclshmemi_exception_report_context_t context{};
        aclshmemi_exception_report_save_context(context);
        EXPECT_EQ(context.enabled_engines, 0U);
        aclshmemi_exception_report_finalize();
    }

    EXPECT_EQ(aclshmemx_enable_exception_report(nullptr, ACLSHMEMX_EXCEPTION_REPORT_INFO), ACLSHMEM_SUCCESS);
    EXPECT_EQ(
        aclshmemi_exception_report_apply_deferred_config(
            static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_UDMA | ACLSHMEM_DATA_OP_ROCE)),
        ACLSHMEM_SUCCESS);
    aclshmemi_exception_report_context_t context{};
    aclshmemi_exception_report_save_context(context);
    EXPECT_EQ(context.enabled_engines, 0U);
    aclshmemi_exception_report_finalize();

    EXPECT_EQ(aclshmemx_enable_exception_report(nullptr, ACLSHMEMX_EXCEPTION_REPORT_DEBUG), ACLSHMEM_SUCCESS);
    for (auto mask : masks_with_udma) {
        EXPECT_EQ(aclshmemi_exception_report_apply_deferred_config(mask), ACLSHMEM_SUCCESS);
        aclshmemi_exception_report_context_t debug_context{};
        aclshmemi_exception_report_save_context(debug_context);
        EXPECT_EQ(debug_context.enabled_engines, static_cast<uint32_t>(ACLSHMEM_DATA_OP_UDMA));
    }
    aclshmemi_exception_report_finalize();

    EXPECT_EQ(aclshmemx_enable_exception_report(nullptr, ACLSHMEMX_EXCEPTION_REPORT_DEBUG), ACLSHMEM_SUCCESS);
    EXPECT_EQ(
        aclshmemi_exception_report_apply_deferred_config(
            static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_SDMA | ACLSHMEM_DATA_OP_ROCE)),
        ACLSHMEM_SUCCESS);
    aclshmemi_exception_report_context_t debug_context{};
    aclshmemi_exception_report_save_context(debug_context);
    EXPECT_EQ(debug_context.enabled_engines, 0U);

    g_state.npes = old_npes;
    aclshmemi_exception_report_finalize();
}

TEST(TestExceptionReportAPI, TestReportSkipsUdmaReporterForSinglePeDebug)
{
    aclshmemi_exception_report_finalize();
    const int old_npes = g_state.npes;
    g_state.npes = 1;

    EXPECT_EQ(aclshmemx_enable_exception_report(nullptr, ACLSHMEMX_EXCEPTION_REPORT_DEBUG), ACLSHMEM_SUCCESS);
    EXPECT_EQ(
        aclshmemi_exception_report_apply_deferred_config(
            static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_UDMA | ACLSHMEM_DATA_OP_ROCE)),
        ACLSHMEM_SUCCESS);
    aclshmemi_exception_report_context_t context{};
    aclshmemi_exception_report_save_context(context);
    EXPECT_EQ(context.enabled_engines, 0U);

    g_state.npes = old_npes;
    aclshmemi_exception_report_finalize();
}

TEST(TestExceptionReportAPI, TestReportSavesAndRestoresUdmaInfoAddress)
{
    aclshmemi_exception_report_finalize();

    constexpr uint64_t udma_info_address = 0x12345000ULL;
    aclshmemi_exception_report_set_udma_info_address(udma_info_address);
    aclshmemi_exception_report_context_t context{};
    aclshmemi_exception_report_save_context(context);
    EXPECT_EQ(context.udma_info_address, udma_info_address);

    aclshmemi_exception_report_set_udma_info_address(0);
    EXPECT_EQ(aclshmemi_exception_report_udma_info_address(), 0U);

    aclshmemi_exception_report_restore_context(context);
    EXPECT_EQ(aclshmemi_exception_report_udma_info_address(), udma_info_address);

    aclshmemi_exception_report_finalize();
}

TEST(TestInitAPI, TestShmemInitAlias)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_init, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemExternLog)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_extern_logger, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemRepeatInit)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_repeatedly_init, local_mem_size, process_count);
}

void test_shmem_get_uniqueid_and_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclshmemx_uniqueid_t& uid)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;

    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(aclrtSetDevice(device_id), 0);

    aclshmemx_init_attr_t attr;
    int ret = aclshmemx_set_attr_uniqueid_args(rank_id, n_ranks, (int64_t)local_mem_size, &uid, &attr);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);

    EXPECT_EQ(attr.my_pe, rank_id);
    EXPECT_EQ(attr.n_pes, n_ranks);
    EXPECT_EQ(attr.local_mem_size, local_mem_size);
    EXPECT_NE(attr.comm_args, nullptr);

    ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);

    EXPECT_EQ(aclshmem_finalize(), ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

TEST(TestInitAPI, TestShmemGetUniqueIdAndInit)
{
    const uint64_t local_mem_size = 64UL * 1024UL * 1024UL;
    const int process_count = test_gnpu_num;

    test_mutil_task_with_uid(test_shmem_get_uniqueid_and_init, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInvalidIP)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_init_invalid_ip, local_mem_size, 1);
}

TEST(TestInitAPI, TestShmemCantAccess)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_init_cant_access, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemMultiInstanceSingle)
{
    // 多实例默认模式所需环境变量，作用域结束自动还原
    const char* fix_port_range = "20000:21023";
    ScopedEnv env_port("SHMEM_INSTANCE_PORT_RANGE", fix_port_range);

    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_multi_instance_single, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemMultiInstanceMalloc)
{
    const char* fix_port_range = "20000:21023";
    ScopedEnv env_port("SHMEM_INSTANCE_PORT_RANGE", fix_port_range);

    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_multi_instance_malloc, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemMultiInstanceOverlap)
{
    const char* fix_port_range = "20000:21023";
    ScopedEnv env_port("SHMEM_INSTANCE_PORT_RANGE", fix_port_range);

    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_multi_instance_overlap, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemMultiInstanceInvalidId)
{
    const char* fix_port_range = "20000:21023";
    ScopedEnv env_port("SHMEM_INSTANCE_PORT_RANGE", fix_port_range);

    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_multi_instance_invalid_id, local_mem_size, process_count);
}

void test_aclshmemx_finalize_active(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);

    uint64_t INSTANCE_ID = 1;
    aclshmemx_init_attr_t attr;
    std::vector<int> dev_list;
    for (int i = 0; i < n_ranks; ++i)
        dev_list.push_back(i);

    bool joined = false;
    int ret = multi_instance_init(rank_id, local_mem_size, INSTANCE_ID, dev_list, attr, joined);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);

    if (joined) {
        EXPECT_EQ(aclshmemx_init_status(), ACLSHMEM_STATUS_IS_INITIALIZED);
        EXPECT_EQ(attr.instance_id, INSTANCE_ID);

        EXPECT_EQ(aclshmemx_finalize(INSTANCE_ID), ACLSHMEM_SUCCESS);
    }

    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_aclshmemx_finalize_nonactive(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    if (n_ranks <= 1) {
        return;
    }

    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);

    uint64_t INSTANCE_ID2 = 2;
    aclshmemx_init_attr_t inst2_attr;
    std::vector<int> inst2_dev = {0, n_ranks - 1};
    bool joined2 = false;
    int ret2 = multi_instance_init(rank_id, local_mem_size, INSTANCE_ID2, inst2_dev, inst2_attr, joined2);
    EXPECT_EQ(ret2, ACLSHMEM_SUCCESS);

    uint64_t INSTANCE_ID4 = 4;
    aclshmemx_init_attr_t inst4_attr;
    std::vector<int> inst4_dev = {n_ranks - 2, n_ranks - 1};
    bool joined4 = false;
    int ret4 = multi_instance_init(rank_id, local_mem_size, INSTANCE_ID4, inst4_dev, inst4_attr, joined4);
    EXPECT_EQ(ret4, ACLSHMEM_SUCCESS);

    if (joined2 && joined4) {
        int status = aclshmemx_instance_ctx_set(INSTANCE_ID2);
        EXPECT_EQ(status, ACLSHMEM_SUCCESS);

        void* ptr = aclshmem_malloc(1024);
        EXPECT_NE(ptr, nullptr);
        aclshmem_free(ptr);

        // 当前活动实例为 INSTANCE_ID2，直接释放非活跃实例 INSTANCE_ID4
        EXPECT_EQ(aclshmemx_finalize(INSTANCE_ID4), ACLSHMEM_SUCCESS);
        EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 2), ACLSHMEM_NOT_SUPPORTED);

        // 活动实例 INSTANCE_ID2 不受影响
        EXPECT_EQ(aclshmemx_instance_ctx_set(INSTANCE_ID2), ACLSHMEM_SUCCESS);

        ptr = aclshmem_malloc(1024);
        EXPECT_NE(ptr, nullptr);
        aclshmem_free(ptr);

        // 释放当前活动实例
        EXPECT_EQ(aclshmemx_finalize(INSTANCE_ID2), ACLSHMEM_SUCCESS);
        EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 2), ACLSHMEM_SUCCESS);
    }

    if (joined2 && !joined4) {
        EXPECT_EQ(aclshmemx_finalize(INSTANCE_ID2), ACLSHMEM_SUCCESS);
        EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 2), ACLSHMEM_SUCCESS);
    }

    if (joined4 && !joined2) {
        EXPECT_EQ(aclshmemx_finalize(INSTANCE_ID4), ACLSHMEM_SUCCESS);
        EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 2), ACLSHMEM_SUCCESS);
    }

    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_aclshmemx_finalize_nonexist(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);

    uint64_t INSTANCE_ID = 1;
    aclshmemx_init_attr_t attr;
    std::vector<int> dev_list;
    for (int i = 0; i < n_ranks; ++i)
        dev_list.push_back(i);

    bool joined = false;
    int ret = multi_instance_init(rank_id, local_mem_size, INSTANCE_ID, dev_list, attr, joined);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);

    if (joined) {
        EXPECT_EQ(aclshmemx_init_status(), ACLSHMEM_STATUS_IS_INITIALIZED);

        // 释放不存在的 instance_id，应返回错误且不影响当前活动实例
        EXPECT_NE(aclshmemx_finalize(999), ACLSHMEM_SUCCESS);
        EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 2), ACLSHMEM_NOT_SUPPORTED);

        // 当前活动实例仍可正常操作
        void* ptr = aclshmem_malloc(1024);
        EXPECT_NE(ptr, nullptr);
        aclshmem_free(ptr);

        // 正常释放当前实例
        EXPECT_EQ(aclshmemx_finalize(INSTANCE_ID), ACLSHMEM_SUCCESS);
        EXPECT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 2), ACLSHMEM_SUCCESS);
    }

    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

TEST(TestInitAPI, TestShmemxFinalizeActive)
{
    const char* fix_port_range = "20000:21023";
    ScopedEnv env_port("SHMEM_INSTANCE_PORT_RANGE", fix_port_range);

    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmemx_finalize_active, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemxFinalizeNonactive)
{
    const char* fix_port_range = "20000:21023";
    ScopedEnv env_port("SHMEM_INSTANCE_PORT_RANGE", fix_port_range);

    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmemx_finalize_nonactive, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemxFinalizeNonexist)
{
    const char* fix_port_range = "20000:21023";
    ScopedEnv env_port("SHMEM_INSTANCE_PORT_RANGE", fix_port_range);

    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmemx_finalize_nonexist, local_mem_size, process_count);
}
