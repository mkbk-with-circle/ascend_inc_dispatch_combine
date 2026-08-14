/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

#include "acl/acl.h"
#include "utils.h"
#include "rdma_sync_barrier_demo_kernel.h"

int g_npus = 8;
const char *ipport;
int f_pe = 0;
int f_npu = 0;


aclshmemx_uniqueid_t default_flag_uid;

static int init_env(int pe_id, int n_pes, uint64_t local_mem_size, int32_t &device_id, aclrtStream &stream)
{
    device_id = pe_id % g_npus + f_npu;
    int status = 0;
    status |= aclInit(nullptr);
    status |= aclrtSetDevice(device_id);
    status |= aclrtCreateStream(&stream);

    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    return status;
}

static int finalize_env(int32_t device_id, aclrtStream stream)
{
    int status = 0;
    status |= aclshmem_finalize();
    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();
    return status;
}

static bool verify_all_gather_result(uint8_t *ptr, int n_pes, int trans_size, int pe_id)
{
    int num_elements = trans_size / (int)sizeof(int32_t);
    size_t total_size = (size_t)n_pes * trans_size;
    int32_t *y_host;
    if (aclrtMallocHost(reinterpret_cast<void**>(&y_host), total_size) != ACL_SUCCESS) {
        std::cout << "[ERROR] aclrtMallocHost failed" << std::endl;
        return false;
    }
    if (aclrtMemcpy(y_host, total_size, ptr, total_size, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        std::cout << "[ERROR] aclrtMemcpy failed" << std::endl;
        aclrtFreeHost(y_host);
        return false;
    }

    bool success = true;
    for (int i = 0; i < n_pes; i++) {
        int32_t expected = i + 10;
        for (int j = 0; j < num_elements; j++) {
            if (y_host[i * num_elements + j] != expected) {
                std::cout << "[FAIL] pe=" << pe_id << " offset=" << i * num_elements + j
                          << " got=" << y_host[i * num_elements + j] << " expected=" << expected << std::endl;
                success = false;
                break;
            }
        }
    }
    if (success) {
        std::cout << "[PASS] check success, pe=" << pe_id << std::endl;
    }
    if (aclrtFreeHost(y_host) != ACL_SUCCESS) {
        std::cout << "[ERROR] aclrtFreeHost failed" << std::endl;
        return false;
    }
    return success;
}

// ---- Device 侧 sync_all demo ----
int test_roce_sync_all(int pe_id, int n_pes, uint64_t local_mem_size)
{
    int32_t device_id;
    aclrtStream stream = nullptr;
    int status = init_env(pe_id, n_pes, local_mem_size, device_id, stream);
    if (status != 0) { return -1; }

    int trans_size = 1024 * 256;
    size_t total_size = (size_t)n_pes * trans_size;
    uint8_t *ptr = static_cast<uint8_t*>(aclshmem_malloc(total_size));
    if (aclrtMemset(ptr, total_size, 0, total_size) != ACL_SUCCESS) {
        std::cout << "[ERROR] aclrtMemset failed" << std::endl;
        aclshmem_free(ptr);
        finalize_env(device_id, stream);
        return -1;
    }

    int num_elements = trans_size / (int)sizeof(int32_t);
    std::vector<int32_t> input(num_elements, pe_id + 10);
    if (aclrtMemcpy(ptr + pe_id * trans_size, trans_size,
        input.data(), trans_size, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        std::cout << "[ERROR] aclrtMemcpy failed" << std::endl;
        aclshmem_free(ptr);
        finalize_env(device_id, stream);
        return -1;
    }

    roce_sync_all_demo(1, stream, ptr, trans_size);
    status |= aclrtSynchronizeStream(stream);

    bool success = verify_all_gather_result(ptr, n_pes, trans_size, pe_id);

    aclshmem_free(ptr);
    finalize_env(device_id, stream);
    return success ? 0 : -1;
}

// ---- Device 侧 sync_all(buf, sync_id) demo ----
int test_roce_sync_all_with_buf(int pe_id, int n_pes, uint64_t local_mem_size)
{
    int32_t device_id;
    aclrtStream stream = nullptr;
    int status = init_env(pe_id, n_pes, local_mem_size, device_id, stream);
    if (status != 0) { return -1; }

    int trans_size = 1024 * 256;
    size_t total_size = (size_t)n_pes * trans_size;
    uint8_t *ptr = static_cast<uint8_t*>(aclshmem_malloc(total_size));
    if (aclrtMemset(ptr, total_size, 0, total_size) != ACL_SUCCESS) {
        std::cout << "[ERROR] aclrtMemset failed" << std::endl;
        aclshmem_free(ptr);
        finalize_env(device_id, stream);
        return -1;
    }

    int num_elements = trans_size / (int)sizeof(int32_t);
    std::vector<int32_t> input(num_elements, pe_id + 10);
    if (aclrtMemcpy(ptr + pe_id * trans_size, trans_size,
        input.data(), trans_size, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        std::cout << "[ERROR] aclrtMemcpy failed" << std::endl;
        aclshmem_free(ptr);
        finalize_env(device_id, stream);
        return -1;
    }

    roce_sync_all_with_buf_demo(1, stream, ptr, trans_size);
    status |= aclrtSynchronizeStream(stream);

    bool success = verify_all_gather_result(ptr, n_pes, trans_size, pe_id);

    aclshmem_free(ptr);
    finalize_env(device_id, stream);
    return success ? 0 : -1;
}

// ---- Device 侧 barrier_all demo ----
int test_roce_barrier_all(int pe_id, int n_pes, uint64_t local_mem_size)
{
    int32_t device_id;
    aclrtStream stream = nullptr;
    int status = init_env(pe_id, n_pes, local_mem_size, device_id, stream);
    if (status != 0) { return -1; }

    int trans_size = 1024 * 256;
    size_t total_size = (size_t)n_pes * trans_size;
    uint8_t *ptr = static_cast<uint8_t*>(aclshmem_malloc(total_size));
    if (aclrtMemset(ptr, total_size, 0, total_size) != ACL_SUCCESS) {
        std::cout << "[ERROR] aclrtMemset failed" << std::endl;
        aclshmem_free(ptr);
        finalize_env(device_id, stream);
        return -1;
    }

    int num_elements = trans_size / (int)sizeof(int32_t);
    std::vector<int32_t> input(num_elements, pe_id + 10);
    if (aclrtMemcpy(ptr + pe_id * trans_size, trans_size,
        input.data(), trans_size, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        std::cout << "[ERROR] aclrtMemcpy failed" << std::endl;
        aclshmem_free(ptr);
        finalize_env(device_id, stream);
        return -1;
    }

    roce_barrier_all_demo(1, stream, ptr, trans_size);
    status |= aclrtSynchronizeStream(stream);

    bool success = verify_all_gather_result(ptr, n_pes, trans_size, pe_id);

    aclshmem_free(ptr);
    finalize_env(device_id, stream);
    return success ? 0 : -1;
}

// ---- Device 侧 barrier_all(buf, sync_id) demo ----
int test_roce_barrier_all_with_buf(int pe_id, int n_pes, uint64_t local_mem_size)
{
    int32_t device_id;
    aclrtStream stream = nullptr;
    int status = init_env(pe_id, n_pes, local_mem_size, device_id, stream);
    if (status != 0) { return -1; }

    int trans_size = 1024 * 256;
    size_t total_size = (size_t)n_pes * trans_size;
    uint8_t *ptr = static_cast<uint8_t*>(aclshmem_malloc(total_size));
    if (aclrtMemset(ptr, total_size, 0, total_size) != ACL_SUCCESS) {
        std::cout << "[ERROR] aclrtMemset failed" << std::endl;
        aclshmem_free(ptr);
        finalize_env(device_id, stream);
        return -1;
    }

    int num_elements = trans_size / (int)sizeof(int32_t);
    std::vector<int32_t> input(num_elements, pe_id + 10);
    if (aclrtMemcpy(ptr + pe_id * trans_size, trans_size,
        input.data(), trans_size, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        std::cout << "[ERROR] aclrtMemcpy failed" << std::endl;
        aclshmem_free(ptr);
        finalize_env(device_id, stream);
        return -1;
    }

    roce_barrier_all_with_buf_demo(1, stream, ptr, trans_size);
    status |= aclrtSynchronizeStream(stream);

    bool success = verify_all_gather_result(ptr, n_pes, trans_size, pe_id);

    aclshmem_free(ptr);
    finalize_env(device_id, stream);
    return success ? 0 : -1;
}

// ---- Device 侧 sync(team) demo ----
int test_roce_sync_team(int pe_id, int n_pes, uint64_t local_mem_size)
{
    int32_t device_id;
    aclrtStream stream = nullptr;
    int status = init_env(pe_id, n_pes, local_mem_size, device_id, stream);
    if (status != 0) { return -1; }

    aclshmem_team_t team_odd;
    int start = 1;
    int stride = 2;
    int team_size = n_pes / 2;
    aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);

    int trans_size = 1024 * 256;
    size_t total_size = (size_t)n_pes * trans_size;
    uint8_t *ptr = static_cast<uint8_t*>(aclshmem_malloc(total_size));
    if (aclrtMemset(ptr, total_size, 0, total_size) != ACL_SUCCESS) {
        std::cout << "[ERROR] aclrtMemset failed" << std::endl;
        aclshmem_free(ptr);
        aclshmem_team_destroy(team_odd);
        finalize_env(device_id, stream);
        return -1;
    }

    if (pe_id & 1) {
        int num_elements = trans_size / (int)sizeof(int32_t);
        std::vector<int32_t> input(num_elements, pe_id + 10);
        if (aclrtMemcpy(ptr + pe_id * trans_size, trans_size,
            input.data(), trans_size, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            std::cout << "[ERROR] aclrtMemcpy failed" << std::endl;
            aclshmem_free(ptr);
            aclshmem_team_destroy(team_odd);
            finalize_env(device_id, stream);
            return -1;
        }
    }

    if (pe_id & 1) {
        roce_sync_team_demo(1, stream, ptr, trans_size, team_odd);
        status |= aclrtSynchronizeStream(stream);
    }

    bool success = true;
    if (pe_id & 1) {
        int num_elements = trans_size / (int)sizeof(int32_t);
        int32_t *y_host;
        if (aclrtMallocHost(reinterpret_cast<void**>(&y_host), total_size) != ACL_SUCCESS) {
            std::cout << "[ERROR] aclrtMallocHost failed" << std::endl;
            aclshmem_free(ptr);
            aclshmem_team_destroy(team_odd);
            finalize_env(device_id, stream);
            return -1;
        }
        if (aclrtMemcpy(y_host, total_size, ptr, total_size, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            std::cout << "[ERROR] aclrtMemcpy failed" << std::endl;
            aclrtFreeHost(y_host);
            aclshmem_free(ptr);
            aclshmem_team_destroy(team_odd);
            finalize_env(device_id, stream);
            return -1;
        }
        for (int j = 0; j < team_size; j++) {
            int pe = start + j * stride;
            int32_t expected = pe + 10;
            for (int k = 0; k < num_elements; k++) {
                if (y_host[pe * num_elements + k] != expected) {
                    std::cout << "[FAIL] pe=" << pe_id << " got=" << y_host[pe * num_elements + k]
                              << " expected=" << expected << std::endl;
                    success = false;
                    break;
                }
            }
        }
        if (success) {
            std::cout << "[PASS] roce_sync_team check success, pe=" << pe_id << std::endl;
        }
        if (aclrtFreeHost(y_host) != ACL_SUCCESS) {
            std::cout << "[ERROR] aclrtFreeHost failed" << std::endl;
            aclshmem_free(ptr);
            aclshmem_team_destroy(team_odd);
            finalize_env(device_id, stream);
            return -1;
        }
    }

    aclshmem_free(ptr);
    aclshmem_team_destroy(team_odd);
    finalize_env(device_id, stream);
    return success ? 0 : -1;
}

// ---- Device 侧 sync(team, buf, sync_id) demo ----
int test_roce_sync_team_with_buf(int pe_id, int n_pes, uint64_t local_mem_size)
{
    int32_t device_id;
    aclrtStream stream = nullptr;
    int status = init_env(pe_id, n_pes, local_mem_size, device_id, stream);
    if (status != 0) { return -1; }

    aclshmem_team_t team_odd;
    int start = 1;
    int stride = 2;
    int team_size = n_pes / 2;
    aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);

    int trans_size = 1024 * 256;
    size_t total_size = (size_t)n_pes * trans_size;
    uint8_t *ptr = static_cast<uint8_t*>(aclshmem_malloc(total_size));
    if (aclrtMemset(ptr, total_size, 0, total_size) != ACL_SUCCESS) {
        std::cout << "[ERROR] aclrtMemset failed" << std::endl;
        aclshmem_free(ptr);
        aclshmem_team_destroy(team_odd);
        finalize_env(device_id, stream);
        return -1;
    }

    if (pe_id & 1) {
        int num_elements = trans_size / (int)sizeof(int32_t);
        std::vector<int32_t> input(num_elements, pe_id + 10);
        if (aclrtMemcpy(ptr + pe_id * trans_size, trans_size,
            input.data(), trans_size, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            std::cout << "[ERROR] aclrtMemcpy failed" << std::endl;
            aclshmem_free(ptr);
            aclshmem_team_destroy(team_odd);
            finalize_env(device_id, stream);
            return -1;
        }
    }

    if (pe_id & 1) {
        roce_sync_team_with_buf_demo(1, stream, ptr, trans_size, team_odd);
        status |= aclrtSynchronizeStream(stream);
    }

    bool success = true;
    if (pe_id & 1) {
        int num_elements = trans_size / (int)sizeof(int32_t);
        int32_t *y_host;
        if (aclrtMallocHost(reinterpret_cast<void**>(&y_host), total_size) != ACL_SUCCESS) {
            std::cout << "[ERROR] aclrtMallocHost failed" << std::endl;
            aclshmem_free(ptr);
            aclshmem_team_destroy(team_odd);
            finalize_env(device_id, stream);
            return -1;
        }
        if (aclrtMemcpy(y_host, total_size, ptr, total_size, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            std::cout << "[ERROR] aclrtMemcpy failed" << std::endl;
            aclrtFreeHost(y_host);
            aclshmem_free(ptr);
            aclshmem_team_destroy(team_odd);
            finalize_env(device_id, stream);
            return -1;
        }
        for (int j = 0; j < team_size; j++) {
            int pe = start + j * stride;
            int32_t expected = pe + 10;
            for (int k = 0; k < num_elements; k++) {
                if (y_host[pe * num_elements + k] != expected) {
                    std::cout << "[FAIL] pe=" << pe_id << " got=" << y_host[pe * num_elements + k]
                              << " expected=" << expected << std::endl;
                    success = false;
                    break;
                }
            }
        }
        if (success) {
            std::cout << "[PASS] roce_sync_team_with_buf check success, pe=" << pe_id << std::endl;
        }
        if (aclrtFreeHost(y_host) != ACL_SUCCESS) {
            std::cout << "[ERROR] aclrtFreeHost failed" << std::endl;
            aclshmem_free(ptr);
            aclshmem_team_destroy(team_odd);
            finalize_env(device_id, stream);
            return -1;
        }
    }

    aclshmem_free(ptr);
    aclshmem_team_destroy(team_odd);
    finalize_env(device_id, stream);
    return success ? 0 : -1;
}

// ---- Device 侧 barrier(team) demo ----
int test_roce_barrier_team(int pe_id, int n_pes, uint64_t local_mem_size)
{
    int32_t device_id;
    aclrtStream stream = nullptr;
    int status = init_env(pe_id, n_pes, local_mem_size, device_id, stream);
    if (status != 0) { return -1; }

    aclshmem_team_t team_odd;
    int start = 1;
    int stride = 2;
    int team_size = n_pes / 2;
    aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);

    int trans_size = 1024 * 256;
    size_t total_size = (size_t)n_pes * trans_size;
    uint8_t *ptr = static_cast<uint8_t*>(aclshmem_malloc(total_size));
    if (aclrtMemset(ptr, total_size, 0, total_size) != ACL_SUCCESS) {
        std::cout << "[ERROR] aclrtMemset failed" << std::endl;
        aclshmem_free(ptr);
        aclshmem_team_destroy(team_odd);
        finalize_env(device_id, stream);
        return -1;
    }

    if (pe_id & 1) {
        int num_elements = trans_size / (int)sizeof(int32_t);
        std::vector<int32_t> input(num_elements, pe_id + 10);
        if (aclrtMemcpy(ptr + pe_id * trans_size, trans_size,
            input.data(), trans_size, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            std::cout << "[ERROR] aclrtMemcpy failed" << std::endl;
            aclshmem_free(ptr);
            aclshmem_team_destroy(team_odd);
            finalize_env(device_id, stream);
            return -1;
        }
    }

    if (pe_id & 1) {
        roce_barrier_team_demo(1, stream, ptr, trans_size, team_odd);
        status |= aclrtSynchronizeStream(stream);
    }

    bool success = true;
    if (pe_id & 1) {
        int num_elements = trans_size / (int)sizeof(int32_t);
        int32_t *y_host;
        if (aclrtMallocHost(reinterpret_cast<void**>(&y_host), total_size) != ACL_SUCCESS) {
            std::cout << "[ERROR] aclrtMallocHost failed" << std::endl;
            aclshmem_free(ptr);
            aclshmem_team_destroy(team_odd);
            finalize_env(device_id, stream);
            return -1;
        }
        if (aclrtMemcpy(y_host, total_size, ptr, total_size, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            std::cout << "[ERROR] aclrtMemcpy failed" << std::endl;
            aclrtFreeHost(y_host);
            aclshmem_free(ptr);
            aclshmem_team_destroy(team_odd);
            finalize_env(device_id, stream);
            return -1;
        }
        for (int j = 0; j < team_size; j++) {
            int pe = start + j * stride;
            int32_t expected = pe + 10;
            for (int k = 0; k < num_elements; k++) {
                if (y_host[pe * num_elements + k] != expected) {
                    std::cout << "[FAIL] pe=" << pe_id << " got=" << y_host[pe * num_elements + k]
                              << " expected=" << expected << std::endl;
                    success = false;
                    break;
                }
            }
        }
        if (success) {
            std::cout << "[PASS] roce_barrier_team check success, pe=" << pe_id << std::endl;
        }
        if (aclrtFreeHost(y_host) != ACL_SUCCESS) {
            std::cout << "[ERROR] aclrtFreeHost failed" << std::endl;
            aclshmem_free(ptr);
            aclshmem_team_destroy(team_odd);
            finalize_env(device_id, stream);
            return -1;
        }
    }

    aclshmem_free(ptr);
    aclshmem_team_destroy(team_odd);
    finalize_env(device_id, stream);
    return success ? 0 : -1;
}

// ---- Device 侧 barrier(team, buf, sync_id) demo ----
int test_roce_barrier_team_with_buf(int pe_id, int n_pes, uint64_t local_mem_size)
{
    int32_t device_id;
    aclrtStream stream = nullptr;
    int status = init_env(pe_id, n_pes, local_mem_size, device_id, stream);
    if (status != 0) { return -1; }

    aclshmem_team_t team_odd;
    int start = 1;
    int stride = 2;
    int team_size = n_pes / 2;
    aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);

    int trans_size = 1024 * 256;
    size_t total_size = (size_t)n_pes * trans_size;
    uint8_t *ptr = static_cast<uint8_t*>(aclshmem_malloc(total_size));
    if (aclrtMemset(ptr, total_size, 0, total_size) != ACL_SUCCESS) {
        std::cout << "[ERROR] aclrtMemset failed" << std::endl;
        aclshmem_free(ptr);
        aclshmem_team_destroy(team_odd);
        finalize_env(device_id, stream);
        return -1;
    }

    if (pe_id & 1) {
        int num_elements = trans_size / (int)sizeof(int32_t);
        std::vector<int32_t> input(num_elements, pe_id + 10);
        if (aclrtMemcpy(ptr + pe_id * trans_size, trans_size,
            input.data(), trans_size, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            std::cout << "[ERROR] aclrtMemcpy failed" << std::endl;
            aclshmem_free(ptr);
            aclshmem_team_destroy(team_odd);
            finalize_env(device_id, stream);
            return -1;
        }
    }

    if (pe_id & 1) {
        roce_barrier_team_with_buf_demo(1, stream, ptr, trans_size, team_odd);
        status |= aclrtSynchronizeStream(stream);
    }

    bool success = true;
    if (pe_id & 1) {
        int num_elements = trans_size / (int)sizeof(int32_t);
        int32_t *y_host;
        if (aclrtMallocHost(reinterpret_cast<void**>(&y_host), total_size) != ACL_SUCCESS) {
            std::cout << "[ERROR] aclrtMallocHost failed" << std::endl;
            aclshmem_free(ptr);
            aclshmem_team_destroy(team_odd);
            finalize_env(device_id, stream);
            return -1;
        }
        if (aclrtMemcpy(y_host, total_size, ptr, total_size, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            std::cout << "[ERROR] aclrtMemcpy failed" << std::endl;
            aclrtFreeHost(y_host);
            aclshmem_free(ptr);
            aclshmem_team_destroy(team_odd);
            finalize_env(device_id, stream);
            return -1;
        }
        for (int j = 0; j < team_size; j++) {
            int pe = start + j * stride;
            int32_t expected = pe + 10;
            for (int k = 0; k < num_elements; k++) {
                if (y_host[pe * num_elements + k] != expected) {
                    std::cout << "[FAIL] pe=" << pe_id << " got=" << y_host[pe * num_elements + k]
                              << " expected=" << expected << std::endl;
                    success = false;
                    break;
                }
            }
        }
        if (success) {
            std::cout << "[PASS] roce_barrier_team_with_buf check success, pe=" << pe_id << std::endl;
        }
        if (aclrtFreeHost(y_host) != ACL_SUCCESS) {
            std::cout << "[ERROR] aclrtFreeHost failed" << std::endl;
            aclshmem_free(ptr);
            aclshmem_team_destroy(team_odd);
            finalize_env(device_id, stream);
            return -1;
        }
    }

    aclshmem_free(ptr);
    aclshmem_team_destroy(team_odd);
    finalize_env(device_id, stream);
    return success ? 0 : -1;
}

static void print_usage(const char *prog)
{
    std::cout << "Usage: " << prog << " <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu> [demo_type]" << std::endl;
    std::cout << "  demo_type:" << std::endl;
    std::cout << "    sync_all       - Device侧 aclshmemx_roce_sync_all demo" << std::endl;
    std::cout << "    sync_all_buf   - Device侧 aclshmemx_roce_sync_all(buf, sync_id) demo" << std::endl;
    std::cout << "    barrier_all    - Device侧 aclshmemx_roce_barrier_all demo" << std::endl;
    std::cout << "    barrier_all_buf - Device侧 aclshmemx_roce_barrier_all(buf, sync_id) demo" << std::endl;
    std::cout << "    sync_team      - Device侧 aclshmemx_roce_team_sync(team) demo" << std::endl;
    std::cout << "    sync_team_buf  - Device侧 aclshmemx_roce_team_sync(team, buf, sync_id) demo" << std::endl;
    std::cout << "    barrier_team   - Device侧 aclshmemx_roce_barrier(team) demo" << std::endl;
    std::cout << "    barrier_team_buf - Device侧 aclshmemx_roce_barrier(team, buf, sync_id) demo" << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc < 7) {
        print_usage(argv[0]);
        return -1;
    }

    int argIdx = 1;
    int n_pes = atoi(argv[argIdx++]);
    int pe_id = atoi(argv[argIdx++]);
    ipport = argv[argIdx++];
    g_npus = atoi(argv[argIdx++]);
    f_pe = atoi(argv[argIdx++]);
    f_npu = atoi(argv[argIdx++]);
    std::string demo_type = (argIdx < argc) ? argv[argIdx++] : "sync_all";

    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    int status = 0;

    if (demo_type == "sync_all") {
        status = test_roce_sync_all(pe_id, n_pes, local_mem_size);
    } else if (demo_type == "sync_all_buf") {
        status = test_roce_sync_all_with_buf(pe_id, n_pes, local_mem_size);
    } else if (demo_type == "barrier_all") {
        status = test_roce_barrier_all(pe_id, n_pes, local_mem_size);
    } else if (demo_type == "barrier_all_buf") {
        status = test_roce_barrier_all_with_buf(pe_id, n_pes, local_mem_size);
    } else if (demo_type == "sync_team") {
        status = test_roce_sync_team(pe_id, n_pes, local_mem_size);
    } else if (demo_type == "sync_team_buf") {
        status = test_roce_sync_team_with_buf(pe_id, n_pes, local_mem_size);
    } else if (demo_type == "barrier_team") {
        status = test_roce_barrier_team(pe_id, n_pes, local_mem_size);
    } else if (demo_type == "barrier_team_buf") {
        status = test_roce_barrier_team_with_buf(pe_id, n_pes, local_mem_size);
    } else {
        std::cout << "[ERROR] unknown demo_type: " << demo_type << std::endl;
        print_usage(argv[0]);
        return -1;
    }

    if (status == 0) {
        std::cout << "[SUCCESS] " << demo_type << " demo run success in pe " << pe_id << std::endl;
    } else {
        std::cout << "[FAILED] " << demo_type << " demo run failed in pe " << pe_id << std::endl;
    }
    return status;
}
