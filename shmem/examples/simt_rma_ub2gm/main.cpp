/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdlib>
#include <unistd.h>

#include "acl/acl.h"
#include "kernel_operator.h"

#include "shmem.h"
#include "../utils/utils.h"

const char* ipport = "tcp://127.0.0.1:8998";
aclshmemx_uniqueid_t default_flag_uid;

// 定义传输大小(传输COPY_SIZE个int32_t的数据)
constexpr int32_t COPY_SIZE = 4096;

__simt_vf__ __launch_bounds__(1024) inline void transfer_data(
    __gm__ int32_t* origin, __gm__ int32_t* res_prev, __gm__ int32_t* res_next, uint32_t elem_size, int32_t prev_pe,
    int32_t next_pe, int32_t mype)
{
    __ubuf__ int32_t ub_buf[COPY_SIZE];
    simt::aclshmemx_int32_get_nbi_warp(ub_buf, origin, elem_size, mype);
    simt::aclshmemx_int32_put_nbi_warp(res_next, ub_buf, elem_size, next_pe);
    simt::aclshmemx_int32_get_nbi_warp(ub_buf, origin, elem_size, prev_pe);
    simt::aclshmemx_int32_put_nbi_warp(res_prev, ub_buf, elem_size, mype);
}

__global__ __vector__ void demo_call(__gm__ int32_t* origin, __gm__ int32_t* res_prev, __gm__ int32_t* res_next)
{
    int32_t mype = aclshmem_my_pe();
    int32_t npes = aclshmem_n_pes();

    int32_t prev_pe = (mype - 1 + npes) % npes;
    int32_t next_pe = (mype + 1) % npes;

    asc_vf_call<transfer_data>(dim3(32), origin, res_prev, res_next, COPY_SIZE, prev_pe, next_pe, mype);
}

void run_demo_mem(void* stream, int32_t* origin, int32_t* res_prev, int32_t* res_next)
{
    demo_call<<<1, 0, stream>>>(origin, res_prev, res_next);
}

/**
 * @brief 打印 origin, res_prev 和 res_next 数组的内容
 * @param my_pe 当前节点的 ID，用于提示
 * @param print_all 是否打印全部元素。若为 false 且长度超过 20，则只打印首尾
 */
void print_buffers(
    int my_pe, int32_t* origin, int32_t* res_prev, int32_t* res_next, int32_t size, bool print_all = true)
{
    auto print_array = [&](const char* name, int32_t* arr) {
        printf("[PE %d] %s: [", my_pe, name);
        if (size <= 20 || print_all) {
            for (int i = 0; i < size; ++i) {
                printf("%d%s", arr[i], (i == size - 1 ? "" : ", "));
            }
        } else {
            // 长度较长时只打印首尾
            for (int i = 0; i < 10; ++i)
                printf("%d, ", arr[i]);
            printf("... , ");
            for (int i = size - 5; i < size; ++i) {
                printf("%d%s", arr[i], (i == size - 1 ? "" : ", "));
            }
        }
        printf("]\n");
    };

    printf("\n[PE %d] ======= Data Report (Size: %d) =======\n", my_pe, size);
    print_array("origin ", origin);
    print_array("res_prev", res_prev);
    print_array("res_next", res_next);
    printf("[PE %d] ======================================\n\n", my_pe);
}

int32_t test_aclshmem_rma_mem(int32_t my_pe, int32_t n_pes)
{
    int32_t ret = 0;
    aclrtStream stream = nullptr;

    ACL_CHECK_WITH_RET(aclInit(nullptr), ERROR_LOG("aclInit failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtSetDevice(my_pe), ERROR_LOG("aclrtSetDevice failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtCreateStream(&stream), ERROR_LOG("aclrtCreateStream failed"), return -1);

    // 1. 准备 Host 端的初始化和校验缓冲区
    size_t data_bytes = COPY_SIZE * sizeof(int32_t);
    int32_t *origin_host, *res_prev_host, *res_next_host;
    ACL_CHECK_WITH_RET(
        aclrtMallocHost(reinterpret_cast<void**>(&origin_host), data_bytes), ERROR_LOG("malloc origin_host failed"),
        return -1);
    ACL_CHECK_WITH_RET(
        aclrtMallocHost(reinterpret_cast<void**>(&res_prev_host), data_bytes), ERROR_LOG("malloc res_prev_host failed"),
        return -1);
    ACL_CHECK_WITH_RET(
        aclrtMallocHost(reinterpret_cast<void**>(&res_next_host), data_bytes), ERROR_LOG("malloc res_next_host failed"),
        return -1);

    // 初始化 origin: Range[my_pe, my_pe + COPY_SIZE)
    for (int i = 0; i < COPY_SIZE; ++i) {
        origin_host[i] = my_pe + i;
        res_prev_host[i] = -1;
        res_next_host[i] = -1;
    }

    // 2. 初始化 ACLSHMEM 并申请对称堆空间
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    aclshmemx_init_attr_t attributes;
    test_set_attr(my_pe, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);
    ACL_CHECK_WITH_RET(
        aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes), ERROR_LOG("aclshmemx_init failed"), return -1);

    int32_t* origin_device = (int32_t*)aclshmemx_malloc(data_bytes);
    int32_t* res_prev_device = (int32_t*)aclshmemx_malloc(data_bytes);
    int32_t* res_next_device = (int32_t*)aclshmemx_malloc(data_bytes);

    // 将初始化数据拷贝到 Device 对称堆
    ACL_CHECK_WITH_RET(
        aclrtMemcpy(origin_device, data_bytes, origin_host, data_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
        ERROR_LOG("memcpy origin to device failed"), return -1);
    ACL_CHECK_WITH_RET(
        aclrtMemcpy(res_prev_device, data_bytes, res_prev_host, data_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
        ERROR_LOG("memcpy res_prev to device failed"), return -1);
    ACL_CHECK_WITH_RET(
        aclrtMemcpy(res_next_device, data_bytes, res_next_host, data_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
        ERROR_LOG("memcpy res_next to device failed"), return -1);

    // 3. 执行同步与计算
    aclshmem_barrier_all();
    run_demo_mem(stream, origin_device, res_prev_device, res_next_device);
    ACL_CHECK_WITH_RET(aclrtSynchronizeStream(stream), ERROR_LOG("stream sync failed"), return -1);
    aclshmem_barrier_all();

    // 4. 拷贝回 Host 进行校验
    ACL_CHECK_WITH_RET(
        aclrtMemcpy(res_prev_host, data_bytes, res_prev_device, data_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
        ERROR_LOG("memcpy res_prev back failed"), return -1);
    ACL_CHECK_WITH_RET(
        aclrtMemcpy(res_next_host, data_bytes, res_next_device, data_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
        ERROR_LOG("memcpy res_next back failed"), return -1);

    sleep(my_pe + 1); // prevent multi-process interference(printing)
    print_buffers(my_pe, origin_host, res_prev_host, res_next_host, COPY_SIZE, false);

    // 5. 校验逻辑
    bool success = true;
    int32_t prev_pe = (my_pe - 1 + n_pes) % n_pes;
    int32_t next_pe = (my_pe + 1) % n_pes;

    for (int i = 0; i < COPY_SIZE; ++i) {
        if (res_prev_host[i] != (prev_pe + i)) {
            printf("[ERROR] PE %d: res_prev[%d] expected %d, got %d\n", my_pe, i, prev_pe + i, res_prev_host[i]);
            success = false;
            break;
        }
        if (res_next_host[i] != (next_pe + i)) {
            printf("[ERROR] PE %d: res_next[%d] expected %d, got %d\n", my_pe, i, next_pe + i, res_next_host[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("[SUCCESS] PE %d: Verification passed for RMA transfers.\n", my_pe);
        ret = 0;
    } else {
        printf("[FAILURE] PE %d: Verification failed for RMA transfers.\n", my_pe);
        ret = -1;
    }

    aclshmemx_free(origin_device);
    aclshmemx_free(res_prev_device);
    aclshmemx_free(res_next_device);
    aclshmem_finalize();

    aclrtFreeHost(origin_host);
    aclrtFreeHost(res_prev_host);
    aclrtFreeHost(res_next_host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(my_pe);
    aclFinalize();
    return ret;
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        ERROR_LOG("Usage: %s <n_pes> <my_pe>", argv[0]);
        return -1;
    }
    int n_pes = atoi(argv[1]);
    int my_pe = atoi(argv[2]);

    int32_t ret = test_aclshmem_rma_mem(my_pe, n_pes);
    INFO_LOG("[INFO] demo run end in pe %d.", my_pe);
    return ret;
}
