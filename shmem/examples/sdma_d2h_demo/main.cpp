/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "../utils/utils.h"
#include "kernel_operator.h"
#include "param.h"
#include "shmem.h"

/*
 * 本样例演示 device kernel 侧使用 SDMA 在普通 Device 内存和 HOST_SIDE SHMEM 之间搬运数据。
 *
 * 支持的 op：
 *   put  验证 aclshmemx_sdma_put_nbi：
 *        本 PE device 内存 -> 本 PE/目标 PE HOST_SIDE SHMEM。
 *   get  验证 aclshmemx_sdma_get_nbi：
 *        本 PE/目标 PE HOST_SIDE SHMEM -> 本 PE device 内存。
 *   all  依次跑 put 和 get。
 */

// 检查返回值，失败时打印错误并直接返回。
#define CHECK_RET(func)                                                                 \
    do {                                                                                \
        int ret = (func);                                                               \
        if (ret != 0) {                                                                 \
            std::cerr << __FILE__ << ":" << __LINE__ << " error: " << ret << std::endl; \
            return ret;                                                                 \
        }                                                                               \
    } while (0)

// 调用公共 ACL_CHECK 打印错误，同时记录首个错误，不改变当前控制流。
#define RECORD_ACL_RET(func, error_ret) \
    do {                                \
        int acl_ret = (func);           \
        ACL_CHECK(acl_ret);             \
        if (acl_ret != 0) {             \
            if ((error_ret) == 0) {     \
                (error_ret) = acl_ret;  \
            }                           \
        }                               \
    } while (0)

const char* ipport = "tcp://127.0.0.1:8766";
aclshmemx_uniqueid_t default_flag_uid;

// 批量模式下启动 20 个 block，并按实际 block 数切分每段 payload。
constexpr uint32_t SDMA_BLOCK_NUM = 20;
// SDMA 接口需要一小段 UB 临时空间。这里固定使用 UB 上 offset=1024 的 64B 空间。
constexpr uint32_t SDMA_UB_OFFSET = 1024;
constexpr uint32_t SDMA_UB_SIZE = 64;
// 默认每个 PE 准备 1024 * 1024 个元素。
constexpr size_t DEFAULT_ELEM_COUNT = 1024 * 1024;
// 默认HOST_SIDE heap只需覆盖本样例数据区；缩小默认值可避免1GB级VMM授权先于SDMA数据面验证失败。
constexpr size_t DEFAULT_HEAP_MB = 16;
constexpr uint64_t BYTES_PER_MB = 1024UL * 1024UL;
// 每个 PE 写入 pe + 10，方便肉眼区分 PE0/PE1/PE2 的数据。
constexpr int VALUE_OFFSET = 10;
constexpr float FLOAT_EPS = 1e-5f;

struct sdma_elem_range_t {
    uint64_t offset;
    uint64_t count;
};

__aicore__ inline bool get_sdma_elem_range(
    int elem_count, uint32_t worker_idx, uint32_t worker_num, sdma_elem_range_t* range)
{
    if (elem_count <= 0 || worker_num == 0 || worker_idx >= worker_num || range == nullptr) {
        return false;
    }

    uint64_t total_elems = static_cast<uint64_t>(elem_count);
    uint64_t base_elems = total_elems / worker_num;
    uint64_t extra_elems = total_elems % worker_num;
    if (worker_idx < extra_elems) {
        range->offset = worker_idx * (base_elems + 1);
        range->count = base_elems + 1;
    } else {
        range->offset = extra_elems * (base_elems + 1) + (worker_idx - extra_elems) * base_elems;
        range->count = base_elems;
    }
    return range->count != 0;
}

bool parse_int_arg(const char* name, const char* value, int min_value, int max_value, int* out)
{
    if (value == nullptr || value[0] == '\0') {
        std::cerr << "Invalid " << name << ": empty value" << std::endl;
        return false;
    }

    errno = 0;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (errno == ERANGE || end == value || *end != '\0' || parsed < min_value || parsed > max_value) {
        std::cerr << "Invalid " << name << ": " << value << ", expected [" << min_value << ", " << max_value << "]"
                  << std::endl;
        return false;
    }

    *out = static_cast<int>(parsed);
    return true;
}

bool parse_size_arg(const char* name, const char* value, size_t min_value, size_t max_value, size_t* out)
{
    if (value == nullptr || value[0] == '\0' || value[0] == '-') {
        std::cerr << "Invalid " << name << ": " << (value == nullptr ? "null" : value) << std::endl;
        return false;
    }

    errno = 0;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno == ERANGE || end == value || *end != '\0' || parsed < min_value ||
        parsed > static_cast<unsigned long long>(max_value)) {
        std::cerr << "Invalid " << name << ": " << value << ", expected [" << min_value << ", " << max_value << "]"
                  << std::endl;
        return false;
    }

    *out = static_cast<size_t>(parsed);
    return true;
}

bool is_supported_op(const std::string& op) { return op == "put" || op == "get" || op == "all"; }

bool is_supported_type(const std::string& data_type)
{
    return data_type == "int" || data_type == "uint8" || data_type == "int64" || data_type == "fp32";
}

template <typename T>
bool validate_workload(int n_pes, size_t elem_count, uint64_t local_mem_size)
{
    if (n_pes <= 0) {
        std::cerr << "Invalid n_pes: " << n_pes << std::endl;
        return false;
    }

    size_t pe_count = static_cast<size_t>(n_pes);
    if (elem_count > std::numeric_limits<size_t>::max() / pe_count) {
        std::cerr << "Invalid elem_count: total element count overflow" << std::endl;
        return false;
    }

    size_t total_elems = pe_count * elem_count;
    if (total_elems > std::numeric_limits<size_t>::max() / sizeof(T)) {
        std::cerr << "Invalid elem_count: total byte size overflow" << std::endl;
        return false;
    }

    size_t total_bytes = total_elems * sizeof(T);
    if (total_bytes > local_mem_size) {
        std::cerr << "Invalid elem_count: required " << total_bytes << " bytes, but SHMEM heap is " << local_mem_size
                  << " bytes" << std::endl;
        return false;
    }

    return true;
}

template <typename T>
T pattern_value(int pe)
{
    return static_cast<T>(pe + VALUE_OFFSET);
}

template <typename T>
bool value_equal(T actual, T expected)
{
    if constexpr (std::is_floating_point_v<T>) {
        return std::fabs(static_cast<double>(actual) - static_cast<double>(expected)) < FLOAT_EPS;
    } else {
        return actual == expected;
    }
}

int32_t set_attr(
    int32_t my_pe, int32_t n_pes, uint64_t local_mem_size, const char* ip_port, aclshmemx_init_attr_t* attributes)
{
    // attributes 是 ACLSHMEM 初始化参数。每个 PE 都会调用一次，并填入自己的 my_pe。
    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), static_cast<size_t>(ACLSHMEM_MAX_IP_PORT_LEN - 1));
        std::copy_n(ip_port, ip_len, attributes->ip_port);
        if (attributes->ip_port[0] == '\0') {
            return ACLSHMEM_INVALID_VALUE;
        }
    }

    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = my_pe;
    attributes->n_pes = n_pes;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_SDMA, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void*>(&default_flag_uid);
    default_flag_uid.my_pe = my_pe;
    default_flag_uid.n_pes = n_pes;
    return ACLSHMEM_SUCCESS;
}

template <typename T>
__global__ __aicore__ void sdma_d2h_put_kernel(GM_ADDR host_dst_gm, GM_ADDR device_src_gm, int elem_count)
{
    if ASCEND_IS_AIV {
        // 当前SDMA channel按block初始化，subblock1不单独提交，避免访问未初始化channel。
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        int my_pe = aclshmem_my_pe();
        int n_pes = aclshmem_n_pes();

        /*
         * host_dst:
         *   HOST_SIDE SHMEM 对称内存。device kernel 通过
         *   SHMEM/SDMA 地址转换后可以把数据写到本 PE 或远端 PE 的 host_dst。
         *
         * device_src:
         *   本 PE 的普通 device 内存，里面存放当前 PE 要发送的数据。
         *
         * tmp_buff:
         *   SDMA 接口提交任务和 quiet 时使用的 UB 临时空间。
         */
        __gm__ T* host_dst = reinterpret_cast<__gm__ T*>(host_dst_gm);
        __gm__ T* device_src = reinterpret_cast<__gm__ T*>(device_src_gm);
        __ubuf__ uint8_t* tmp_buff = reinterpret_cast<__ubuf__ uint8_t*>(uint64_t(SDMA_UB_OFFSET));

        const auto subblock_num = AscendC::GetSubBlockNum();
        const auto worker_num = AscendC::GetBlockNum();
        if (elem_count <= 0 || subblock_num == 0 || worker_num == 0) {
            return;
        }
        const auto worker_idx = AscendC::GetBlockIdx() / subblock_num;

        /*
         * 多个 block 并行搬运同一段数据。这里按 T 元素切分，而不是按字节切分，
         * 避免 int/float 等类型生成非元素对齐的 SDMA 地址和长度。
         */
        sdma_elem_range_t range;
        if (!get_sdma_elem_range(elem_count, worker_idx, worker_num, &range)) {
            return;
        }

        /*
         * src 是本 PE device_src 中当前 block 负责的小段。
         *
         * dst 是 HOST_SIDE SHMEM 中“当前 PE 对应 segment”的本地对称地址：
         *   host_dst + elem_count * my_pe + elem_offset
         *
         * 它表面上是本 PE 的地址，但 aclshmemx_sdma_put_nbi 会根据 pe 参数把它转成：
         *   目标 PE 的 host_dst + elem_count * my_pe + elem_offset
         *
         * 所以 PE i 会把自己的 device_src 写到本 PE 和其他 PE 的 host_dst 第 i 段：
         *   pe == my_pe 验证本地 D2H；
         *   pe != my_pe 验证 D2rH。
         */
        __gm__ T* src = device_src + range.offset;
        __gm__ T* dst = host_dst + static_cast<uint64_t>(elem_count) * my_pe + range.offset;
        for (int pe = 0; pe < n_pes; ++pe) {
            aclshmemx_sdma_put_nbi(
                dst, src, reinterpret_cast<__ubuf__ T*>(tmp_buff), SDMA_UB_SIZE, range.count, pe, EVENT_ID0);
        }
        // put_nbi 只是提交任务，不等待完成；quiet 确保当前 block 提交的 SDMA 都完成。
        aclshmemx_sdma_quiet(tmp_buff, SDMA_UB_SIZE, EVENT_ID0);
    }
}

template <typename T>
__global__ __aicore__ void sdma_d2h_get_kernel(GM_ADDR device_dst_gm, GM_ADDR host_src_gm, int elem_count)
{
    if ASCEND_IS_AIV {
        // 当前SDMA channel按block初始化，subblock1不单独提交，避免访问未初始化channel。
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        int my_pe = aclshmem_my_pe();
        int n_pes = aclshmem_n_pes();

        /*
         * host_src:
         *   HOST_SIDE SHMEM 对称内存。每个 PE 的 host_src 布局都一样。
         *   本 kernel 会从本 PE 或远端 PE 的 host_src 中读数据。
         *
         * device_dst:
         *   本 PE 的普通 device 内存，用来保存 get 拉回来的数据。
         *
         * tmp_buff:
         *   SDMA 接口提交任务和 quiet 时使用的 UB 临时空间。
         */
        __gm__ T* device_dst = reinterpret_cast<__gm__ T*>(device_dst_gm);
        __gm__ T* host_src = reinterpret_cast<__gm__ T*>(host_src_gm);
        __ubuf__ uint8_t* tmp_buff = reinterpret_cast<__ubuf__ uint8_t*>(uint64_t(SDMA_UB_OFFSET));

        const auto subblock_num = AscendC::GetSubBlockNum();
        const auto worker_num = AscendC::GetBlockNum();
        if (elem_count <= 0 || subblock_num == 0 || worker_num == 0) {
            return;
        }
        const auto worker_idx = AscendC::GetBlockIdx() / subblock_num;

        /*
         * 和 put kernel 一样，多个 block 合力搬运一段 payload。
         * 每个 block 只负责一段完整的 T 元素，避免生成非元素对齐的 SDMA 请求。
         */
        sdma_elem_range_t range;
        if (!get_sdma_elem_range(elem_count, worker_idx, worker_num, &range)) {
            return;
        }

        /*
         * 本 PE 会从本 PE 和每个远端 PE 的 host_src 第 pe 段拉数据，
         * 放到本 PE device_dst 第 pe 段：
         *   pe == my_pe 验证本地 H2D；
         *   pe != my_pe 验证 rH2D。
         */
        for (int pe = 0; pe < n_pes; ++pe) {
            __gm__ T* dst = device_dst + static_cast<uint64_t>(elem_count) * pe + range.offset;
            __gm__ T* src = host_src + static_cast<uint64_t>(elem_count) * pe + range.offset;
            aclshmemx_sdma_get_nbi(
                dst, src, reinterpret_cast<__ubuf__ T*>(tmp_buff), SDMA_UB_SIZE, range.count, pe, EVENT_ID0);
        }
        // get_nbi 也只是提交任务；quiet 确保当前 block 发起的 get 都已经完成。
        aclshmemx_sdma_quiet(tmp_buff, SDMA_UB_SIZE, EVENT_ID0);
    }
}

template <typename T>
int validate_segments(const std::vector<T>& actual, int my_pe, int n_pes, size_t elem_count)
{
    /*
     * 校验规则：
     *   第 pe 段应该全部等于 pe + 10。
     *
     * elem_count 是每个 PE 的元素个数。
     */
    for (int pe = 0; pe < n_pes; ++pe) {
        T expected = pattern_value<T>(pe);
        for (size_t idx = 0; idx < elem_count; ++idx) {
            T value = actual[static_cast<size_t>(pe) * elem_count + idx];
            if (!value_equal(value, expected)) {
                std::cerr << "PE " << my_pe << " validate failed at segment " << pe << ", index " << idx << std::endl;
                return -1;
            }
        }
    }
    return 0;
}

template <typename T>
void print_value(T value)
{
    if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
        std::cout << static_cast<int>(value);
    } else {
        std::cout << value;
    }
}

template <typename T>
void print_segment_first_values(
    const std::string& op, const std::vector<T>& result, int my_pe, int n_pes, size_t elem_count)
{
    size_t print_count = std::min<size_t>(8, elem_count);
    for (int pe = 0; pe < n_pes; ++pe) {
        size_t segment_offset = static_cast<size_t>(pe) * elem_count;
        std::cout << "[RESULT] pe " << my_pe << " " << op << " segment " << pe << " expected ";
        print_value(pattern_value<T>(pe));
        std::cout << " first " << print_count << " values:";
        for (size_t idx = 0; idx < print_count && segment_offset + idx < result.size(); ++idx) {
            std::cout << " ";
            print_value(result[segment_offset + idx]);
        }
        std::cout << std::endl;
    }
}

template <typename T>
int run_put_mode(T* host_dst, int prepare_ret, int my_pe, int n_pes, size_t elem_count, size_t total_bytes)
{
    // error_ret记录数据面执行过程中是否已经出现错误
    // 后续通过error_ret 是否等于 0，决定是否继续执行依赖前序资源的操作
    int error_ret = prepare_ret;
    // validate_ret是函数最终返回值。它主要承接两类结果：
    // 1. 如果数据面已经失败，`validate_ret = error_ret`。
    // 2. 如果数据面成功，则执行结果校验，`validate_ret = validate_segments(...)`。
    int validate_ret = 0;

    // stream 是 host 提交 device kernel 的队列。
    aclrtStream stream = nullptr;
    // 记录 stream 创建错误，不在资源申请后提前退出，统一走到函数末尾释放本地资源。
    if (error_ret == 0) {
        RECORD_ACL_RET(aclrtCreateStream(&stream), error_ret);
    }

    // 每个 PE 一段 payload；host_dst 保存所有 PE 的 payload，所以总大小是 n_pes 段。
    size_t segment_bytes = elem_count * sizeof(T);
    size_t total_elems = static_cast<size_t>(n_pes) * elem_count;

    /*
     * host_dst 是 HOST_SIDE SHMEM 对称内存：
     *   - 每个 PE 都有自己的 host_dst。
     *   - 所有 PE 的 host_dst 布局一样。
     *   - device kernel 通过 SDMA 可以写到本 PE 或远端 PE 的 host_dst。
     *
     * 布局：
     *   [PE0 segment][PE1 segment]...[PEn segment]
     */
    // device_src 是本 PE 的普通 device 内存，只存放当前 PE 自己要发送的一段数据。
    T* device_src = nullptr;
    // 前序资源不可用时跳过 device 申请，但仍进入后续统一 cleanup。
    if (error_ret == 0 && host_dst != nullptr) {
        RECORD_ACL_RET(
            aclrtMalloc(reinterpret_cast<void**>(&device_src), segment_bytes, ACL_MEM_MALLOC_HUGE_FIRST), error_ret);
    }

    // host_dst 初始为全 0；所有 segment 都必须由 SDMA put 写入，包括本 PE 的 D2H segment。
    std::vector<T> init_host(total_elems, static_cast<T>(0));

    // 普通 device 初始化。
    std::vector<T> device_src_host(elem_count, pattern_value<T>(my_pe));
    // 已有错误时跳过后续 memcpy，避免继续使用无效 device_src。
    if (error_ret == 0) {
        RECORD_ACL_RET(
            aclrtMemcpy(device_src, segment_bytes, device_src_host.data(), segment_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
            error_ret);
    }
    // Host -> HOST_SIDE SHMEM：清空本 PE 的 host_dst 结果区。
    // 只在前序初始化成功时写 HOST_SIDE，失败 PE 仍会进入后续 cleanup。
    if (error_ret == 0) {
        RECORD_ACL_RET(
            aclrtMemcpy(host_dst, total_bytes, init_host.data(), total_bytes, ACL_MEMCPY_HOST_TO_DEVICE), error_ret);
    }

    // 确保所有 PE 都完成 host_dst/device_src 初始化，再开始写本地和远端 HOST_SIDE 内存。
    aclshmem_barrier_all();
    if (error_ret == 0 && host_dst != nullptr) {
        sdma_d2h_put_kernel<T><<<SDMA_BLOCK_NUM, nullptr, stream>>>(
            reinterpret_cast<uint8_t*>(host_dst), reinterpret_cast<uint8_t*>(device_src), static_cast<int>(elem_count));
        // 等当前 PE 的 kernel 完成。
        RECORD_ACL_RET(aclrtSynchronizeStream(stream), error_ret);
    }
    // 再等所有 PE 的 kernel 都完成，确保本地 D2H 和其他 PE 写到本 PE host_dst 的数据都已完成。
    aclshmem_barrier_all();

    /*
     * Device/SHMEM -> Host：这里是为了校验结果，把 HOST_SIDE SHMEM 内容拷回 vector。
     */
    std::vector<T> result(total_elems);
    // 只在数据面无错误时回拷校验结果；失败 PE 继续参与 cleanup。
    if (error_ret == 0) {
        RECORD_ACL_RET(
            aclrtMemcpy(result.data(), total_bytes, host_dst, total_bytes, ACL_MEMCPY_DEVICE_TO_HOST), error_ret);
    }

    if (error_ret == 0) {
        print_segment_first_values("put", result, my_pe, n_pes, elem_count);
        validate_ret = validate_segments(result, my_pe, n_pes, elem_count);
        if (validate_ret == 0) {
            std::cout << "[SUCCESS] put op pass in pe " << my_pe << std::endl;
        }
    } else {
        validate_ret = error_ret;
    }

    // 所有 PE 走到 cleanup 点；HOST_SIDE SHMEM 由 run_op 处理。
    aclshmem_barrier_all();

    // 本地资源释放失败会更新最终返回值。
    if (device_src != nullptr) {
        RECORD_ACL_RET(aclrtFree(device_src), validate_ret);
    }
    if (stream != nullptr) {
        RECORD_ACL_RET(aclrtDestroyStream(stream), validate_ret);
    }
    return validate_ret;
}

template <typename T>
int run_get_mode(T* host_src, int prepare_ret, int my_pe, int n_pes, size_t elem_count, size_t total_bytes)
{
    int error_ret = prepare_ret;
    int validate_ret = 0;

    aclrtStream stream = nullptr;
    // 记录 stream 创建错误，不在资源申请后提前退出，统一走到函数末尾释放本地资源。
    if (error_ret == 0) {
        RECORD_ACL_RET(aclrtCreateStream(&stream), error_ret);
    }

    // 每个 PE 一段 payload；host_src/device_dst 都保存所有 PE 的 payload，所以总大小是 n_pes 段。
    size_t total_elems = static_cast<size_t>(n_pes) * elem_count;

    /*
     * host_src 是 HOST_SIDE SHMEM 对称内存：
     *   - 每个 PE 都有自己的 host_src。
     *   - 每个 PE 只把自己的 segment 初始化成 my_pe + 10。
     *   - 本 PE 和其他 PE 通过 get_nbi 从这里读取对应 segment。
     *
     * 布局：
     *   [PE0 segment][PE1 segment]...[PEn segment]
     */
    // device_dst 是本 PE 的普通 device 内存，保存从本 PE 和各个远端 PE 拉回来的结果。
    T* device_dst = nullptr;
    // 前序资源不可用时跳过 device 申请，但仍进入后续统一 cleanup。
    if (error_ret == 0 && host_src != nullptr) {
        RECORD_ACL_RET(
            aclrtMalloc(reinterpret_cast<void**>(&device_dst), total_bytes, ACL_MEM_MALLOC_HUGE_FIRST), error_ret);
    }

    /*
     * init_host 是本 PE host_src 的初始内容。
     * 只有第 my_pe 段放真实数据，因为本 PE 和远端 PE get 本 PE 数据时只会读这一段。
     */
    std::vector<T> init_host(total_elems, static_cast<T>(0));
    std::fill(
        init_host.begin() + static_cast<size_t>(my_pe) * elem_count,
        init_host.begin() + static_cast<size_t>(my_pe + 1) * elem_count, pattern_value<T>(my_pe));

    // Host -> HOST_SIDE SHMEM：把本 PE 可被本 PE 和其他 PE 读取的数据放进 host_src。
    // 只在前序初始化成功时写 HOST_SIDE，失败 PE 仍会进入后续 cleanup。
    if (error_ret == 0) {
        RECORD_ACL_RET(
            aclrtMemcpy(host_src, total_bytes, init_host.data(), total_bytes, ACL_MEMCPY_HOST_TO_DEVICE), error_ret);
    }
    // 普通 device 结果区直接用 Host -> Device aclrtMemcpy 清零。
    // 所有 segment 都必须由 SDMA get 写入，包括本 PE 的 H2D segment。
    std::vector<T> device_dst_host(total_elems, static_cast<T>(0));
    // 已有错误时跳过后续 memcpy，避免继续使用无效 device_dst。
    if (error_ret == 0) {
        RECORD_ACL_RET(
            aclrtMemcpy(device_dst, total_bytes, device_dst_host.data(), total_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
            error_ret);
    }

    // 确保所有 PE 都完成 host_src/device_dst 初始化，再开始从本地和远端 HOST_SIDE 拉数据。
    aclshmem_barrier_all();
    if (error_ret == 0 && host_src != nullptr) {
        sdma_d2h_get_kernel<T><<<SDMA_BLOCK_NUM, nullptr, stream>>>(
            reinterpret_cast<uint8_t*>(device_dst), reinterpret_cast<uint8_t*>(host_src), static_cast<int>(elem_count));
        RECORD_ACL_RET(aclrtSynchronizeStream(stream), error_ret);
    }
    // 等所有 PE 的 get 都结束，避免有 PE 仍在读取某个 HOST_SIDE host_src。
    aclshmem_barrier_all();

    /*
     * Device -> Host：把本 PE device_dst 拷回校验。
     */
    std::vector<T> result(total_elems);
    // 只在数据面无错误时回拷校验结果；失败 PE 继续参与 cleanup。
    if (error_ret == 0) {
        RECORD_ACL_RET(
            aclrtMemcpy(result.data(), total_bytes, device_dst, total_bytes, ACL_MEMCPY_DEVICE_TO_HOST), error_ret);
    }

    if (error_ret == 0) {
        print_segment_first_values("get", result, my_pe, n_pes, elem_count);
        validate_ret = validate_segments(result, my_pe, n_pes, elem_count);
        if (validate_ret == 0) {
            std::cout << "[SUCCESS] get op pass in pe " << my_pe << std::endl;
        }
    } else {
        validate_ret = error_ret;
    }

    // 所有 PE 走到 cleanup 点；HOST_SIDE SHMEM 由 run_op 处理。
    aclshmem_barrier_all();

    // 本地资源释放失败会更新最终返回值。
    if (device_dst != nullptr) {
        RECORD_ACL_RET(aclrtFree(device_dst), validate_ret);
    }
    if (stream != nullptr) {
        RECORD_ACL_RET(aclrtDestroyStream(stream), validate_ret);
    }
    return validate_ret;
}

template <typename T>
int run_op(const std::string& op, int my_pe, int n_pes, size_t elem_count)
{
    size_t total_elems = static_cast<size_t>(n_pes) * elem_count;
    size_t total_bytes = total_elems * sizeof(T);

    T* host_side_buf = reinterpret_cast<T*>(aclshmemx_malloc(total_bytes, HOST_SIDE));
    int malloc_error = 0;
    if (host_side_buf == nullptr) {
        std::cerr << "aclshmemx_malloc HOST_SIDE failed in pe " << my_pe << ", required bytes: " << total_bytes
                  << std::endl;
        malloc_error = ACLSHMEM_MALLOC_FAILED;
    }

    int ret = 0;
    if (op == "put") {
        ret = run_put_mode<T>(host_side_buf, malloc_error, my_pe, n_pes, elem_count, total_bytes);
    } else if (op == "get") {
        ret = run_get_mode<T>(host_side_buf, malloc_error, my_pe, n_pes, elem_count, total_bytes);
    } else if (op == "all") {
        ret = run_put_mode<T>(host_side_buf, malloc_error, my_pe, n_pes, elem_count, total_bytes);
        int get_ret = run_get_mode<T>(host_side_buf, malloc_error, my_pe, n_pes, elem_count, total_bytes);
        if (ret == 0) {
            ret = get_ret;
        }
    } else {
        std::cerr << "Unsupported op: " << op << std::endl;
        ret = -1;
    }

    if (host_side_buf != nullptr) {
        aclshmemx_free(host_side_buf, HOST_SIDE);
    }
    aclshmem_barrier_all();
    return ret;
}

struct demo_args_t {
    int n_pes = 0;
    int my_pe = 0;
    std::string op;
    std::string data_type;
    size_t elem_count = DEFAULT_ELEM_COUNT;
    uint64_t local_mem_size = 0;
};

bool validate_workload_by_type(const demo_args_t& args)
{
    if (args.data_type == "int") {
        return validate_workload<int>(args.n_pes, args.elem_count, args.local_mem_size);
    }
    if (args.data_type == "uint8") {
        return validate_workload<uint8_t>(args.n_pes, args.elem_count, args.local_mem_size);
    }
    if (args.data_type == "int64") {
        return validate_workload<int64_t>(args.n_pes, args.elem_count, args.local_mem_size);
    }
    if (args.data_type == "fp32") {
        return validate_workload<float>(args.n_pes, args.elem_count, args.local_mem_size);
    }
    std::cerr << "Unsupported type: " << args.data_type << std::endl;
    return false;
}

int run_op_by_type(const demo_args_t& args)
{
    if (args.data_type == "int") {
        return run_op<int>(args.op, args.my_pe, args.n_pes, args.elem_count);
    }
    if (args.data_type == "uint8") {
        return run_op<uint8_t>(args.op, args.my_pe, args.n_pes, args.elem_count);
    }
    if (args.data_type == "int64") {
        return run_op<int64_t>(args.op, args.my_pe, args.n_pes, args.elem_count);
    }
    if (args.data_type == "fp32") {
        return run_op<float>(args.op, args.my_pe, args.n_pes, args.elem_count);
    }
    std::cerr << "Unsupported type: " << args.data_type << std::endl;
    return -1;
}

void print_usage(const char* program)
{
    std::cerr << "Usage: " << program << " <n_pes> <my_pe> <ipport> <op> <type> [elem_count] [heap_mb]" << std::endl;
    std::cerr << "  op: put | get | all" << std::endl;
    std::cerr << "  type: int | uint8 | int64 | fp32" << std::endl;
}

bool parse_and_validate_args(int argc, char* argv[], demo_args_t* args)
{
    if (args == nullptr) {
        std::cerr << "Invalid args output: nullptr" << std::endl;
        return false;
    }
    if (argc < INDEX6 || argc > INDEX8) {
        print_usage(argv[INDEX0]);
        return false;
    }

    constexpr int max_int = std::numeric_limits<int>::max();
    constexpr size_t max_kernel_elem_count = static_cast<size_t>(std::numeric_limits<int>::max());
    constexpr size_t max_heap_mb = std::numeric_limits<uint64_t>::max() / BYTES_PER_MB;
    size_t heap_mb = DEFAULT_HEAP_MB;

    if (!parse_int_arg("n_pes", argv[INDEX1], 1, max_int, &args->n_pes) ||
        !parse_int_arg("my_pe", argv[INDEX2], 0, max_int, &args->my_pe)) {
        print_usage(argv[INDEX0]);
        return false;
    }
    if (args->my_pe >= args->n_pes) {
        std::cerr << "Invalid my_pe: " << args->my_pe << ", expected [0, " << (args->n_pes - 1) << "]" << std::endl;
        return false;
    }

    ipport = argv[INDEX3];
    if (ipport == nullptr || ipport[0] == '\0' || strlen(ipport) >= ACLSHMEM_MAX_IP_PORT_LEN) {
        std::cerr << "Invalid ipport: empty or too long" << std::endl;
        return false;
    }

    args->op = argv[INDEX4];
    args->data_type = argv[INDEX5];
    if (!is_supported_op(args->op)) {
        std::cerr << "Unsupported op: " << args->op << std::endl;
        print_usage(argv[INDEX0]);
        return false;
    }
    if (!is_supported_type(args->data_type)) {
        std::cerr << "Unsupported type: " << args->data_type << std::endl;
        print_usage(argv[INDEX0]);
        return false;
    }
    if (argc == INDEX7 && !parse_size_arg("elem_count", argv[INDEX6], 1, max_kernel_elem_count, &args->elem_count)) {
        print_usage(argv[INDEX0]);
        return false;
    }
    if (argc == INDEX8 && (!parse_size_arg("elem_count", argv[INDEX6], 1, max_kernel_elem_count, &args->elem_count) ||
                           !parse_size_arg("heap_mb", argv[INDEX7], 1, max_heap_mb, &heap_mb))) {
        print_usage(argv[INDEX0]);
        return false;
    }

    args->local_mem_size = static_cast<uint64_t>(heap_mb) * BYTES_PER_MB;

    if (!validate_workload_by_type(*args)) {
        return false;
    }

    return true;
}

int main(int argc, char* argv[])
{
    demo_args_t args;
    if (!parse_and_validate_args(argc, argv, &args)) {
        return -1;
    }
    // 简单按 my_pe 映射 NPU：PE0->NPU0、PE1->NPU1。
    int32_t device_id = args.my_pe;
    // 先初始化 ACL，再设置当前进程绑定的 NPU。
    CHECK_RET(aclInit(nullptr));
    CHECK_RET(aclrtSetDevice(device_id));

    aclshmemx_init_attr_t attributes;
    CHECK_RET(set_attr(args.my_pe, args.n_pes, args.local_mem_size, ipport, &attributes));
    CHECK_RET(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes));

    // 根据 -type 选择模板类型，再根据 -op 选择 put/get/all。
    int ret = run_op_by_type(args);
    // 按 SHMEM -> device -> ACL 的顺序释放全局资源。
    int finalize_ret = aclshmem_finalize();
    int reset_ret = aclrtResetDevice(device_id);
    int acl_finalize_ret = aclFinalize();
    if (ret != 0 || finalize_ret != 0 || reset_ret != 0 || acl_finalize_ret != 0) {
        std::cerr << "sdma_d2h_demo failed in pe " << args.my_pe << ": op_ret=" << ret
                  << ", aclshmem_finalize=" << finalize_ret << ", aclrtResetDevice=" << reset_ret
                  << ", aclFinalize=" << acl_finalize_ret << std::endl;
        return -1;
    }

    std::cout << "[SUCCESS] sdma_d2h_demo run success in pe " << args.my_pe << std::endl;
    return 0;
}
