/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "acl/acl.h"
#include "one_multi_path_kernel.h"
#include "shmem.h"
#include "utils.h"

namespace {

constexpr int MIN_PE_COUNT = 2;
constexpr size_t DEVICE_LARGE_PAGE_SIZE = 2UL * 1024UL * 1024UL;
constexpr size_t PATH_SLICE_ALIGNMENT = 32UL;
constexpr uint64_t RESERVE_FLAGS_ASCEND950 = 1UL;
constexpr uint32_t ONE_PATH_LINK_TYPE = 2U;
constexpr uint32_t MULTI_PATH_LINK_TYPE = 3U;
constexpr int32_t DATA_OFFSET = 10;
constexpr int TCP_CONNECT_TIMEOUT_SECONDS = 60;
constexpr int TCP_CONNECT_RETRY_INTERVAL_MS = 100;
constexpr int TCP_IO_TIMEOUT_SECONDS = 60;
constexpr int TCP_FABRIC_HANDLES_PORT_OFFSET = 1;
constexpr int TCP_STATUS_PORT_OFFSET = 2;

bool parse_int(const char* value, int& out)
{
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || end == nullptr || *end != '\0' || parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}

bool parse_size_kb(const char* value, size_t& out)
{
    if (value == nullptr || value[0] == '\0' || value[0] == '-') {
        return false;
    }
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || end == nullptr || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max() / 1024UL) {
        return false;
    }
    out = static_cast<size_t>(parsed) * 1024UL;
    return true;
}

bool align_up(size_t value, size_t align, size_t& aligned)
{
    if (align == 0 || value > std::numeric_limits<size_t>::max() - (align - 1)) {
        return false;
    }
    aligned = (value + align - 1) / align * align;
    return true;
}

bool send_all(int fd, const void* data, size_t size)
{
    const auto* bytes = static_cast<const uint8_t*>(data);
    while (size > 0) {
        ssize_t sent = send(fd, bytes, size, MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR) {
            continue;
        }
        if (sent <= 0) {
            return false;
        }
        bytes += sent;
        size -= static_cast<size_t>(sent);
    }
    return true;
}

bool recv_all(int fd, void* data, size_t size)
{
    auto* bytes = static_cast<uint8_t*>(data);
    while (size > 0) {
        ssize_t received = recv(fd, bytes, size, 0);
        if (received < 0 && errno == EINTR) {
            continue;
        }
        if (received <= 0) {
            return false;
        }
        bytes += received;
        size -= static_cast<size_t>(received);
    }
    return true;
}

bool parse_tcp_endpoint(const char* endpoint, in_addr& address, int& port)
{
    constexpr const char* prefix = "tcp://";
    if (endpoint == nullptr) {
        return false;
    }
    std::string value(endpoint);
    size_t separator = value.rfind(':');
    if (value.compare(0, std::strlen(prefix), prefix) != 0 || separator == std::string::npos ||
        separator <= std::strlen(prefix) || separator + 1 >= value.size()) {
        return false;
    }
    std::string host = value.substr(std::strlen(prefix), separator - std::strlen(prefix));
    return parse_int(value.substr(separator + 1).c_str(), port) && port > 0 &&
           port <= std::numeric_limits<uint16_t>::max() && inet_pton(AF_INET, host.c_str(), &address) == 1;
}

bool wait_for_tcp_connect(int fd, const std::chrono::steady_clock::time_point& deadline)
{
    while (std::chrono::steady_clock::now() < deadline) {
        auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
        int timeout_ms = static_cast<int>(remaining.count() > 0 ? remaining.count() : 1);
        pollfd event{fd, POLLOUT, 0};
        int ret = poll(&event, 1, timeout_ms);
        if (ret < 0 && errno == EINTR) {
            continue;
        }
        if (ret <= 0 || (event.revents & POLLNVAL) != 0) {
            return false;
        }

        int socket_error = 0;
        socklen_t error_size = sizeof(socket_error);
        return getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error, &error_size) == 0 && socket_error == 0;
    }
    return false;
}

int connect_tcp(const sockaddr_in& address)
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(TCP_CONNECT_TIMEOUT_SECONDS);
    while (std::chrono::steady_clock::now() < deadline) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd >= 0) {
            int flags = fcntl(fd, F_GETFL, 0);
            bool is_nonblocking = flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
            int ret = is_nonblocking ? connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) : -1;
            int connect_error = errno;
            bool is_connected =
                is_nonblocking && (ret == 0 || (ret < 0 &&
                                                (connect_error == EINPROGRESS || connect_error == EALREADY ||
                                                 connect_error == EWOULDBLOCK || connect_error == EINTR) &&
                                                wait_for_tcp_connect(fd, deadline)));
            if (is_connected && fcntl(fd, F_SETFL, flags) == 0) {
                return fd;
            }
            close(fd);
        }

        auto now = std::chrono::steady_clock::now();
        if (now < deadline) {
            auto retry_interval = std::chrono::milliseconds(TCP_CONNECT_RETRY_INTERVAL_MS);
            std::this_thread::sleep_for(deadline - now < retry_interval ? deadline - now : retry_interval);
        }
    }
    return -1;
}

bool tcp_allgather(
    const void* local_data, void* all_data, size_t size, int pe_id, int n_pes, const char* endpoint, int port_offset,
    const char* tag)
{
    in_addr server_address{};
    int base_port = 0;
    if (local_data == nullptr || all_data == nullptr || size == 0 || n_pes <= 0 || pe_id < 0 || pe_id >= n_pes ||
        size > std::numeric_limits<size_t>::max() / static_cast<size_t>(n_pes) || port_offset <= 0 ||
        port_offset > std::numeric_limits<uint16_t>::max() ||
        !parse_tcp_endpoint(endpoint, server_address, base_port) ||
        base_port > std::numeric_limits<uint16_t>::max() - port_offset) {
        std::cerr << "invalid TCP allgather arguments for " << tag << std::endl;
        return false;
    }

    size_t total_size = size * static_cast<size_t>(n_pes);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr = server_address;
    address.sin_port = htons(static_cast<uint16_t>(base_port + port_offset));
    timeval timeout{TCP_IO_TIMEOUT_SECONDS, 0};

    if (pe_id != 0) {
        int fd = connect_tcp(address);
        uint32_t rank = htonl(static_cast<uint32_t>(pe_id));
        bool success = fd >= 0 && setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0 &&
                       setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0 &&
                       send_all(fd, &rank, sizeof(rank)) && send_all(fd, local_data, size) &&
                       recv_all(fd, all_data, total_size);
        if (fd >= 0) {
            close(fd);
        }
        if (!success) {
            std::cerr << "TCP allgather " << tag << " failed on PE " << pe_id << std::endl;
        }
        return success;
    }

    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    int reuse_address = 1;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    bool success =
        listen_fd >= 0 && setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse_address, sizeof(reuse_address)) == 0 &&
        setsockopt(listen_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0 &&
        bind(listen_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0 && listen(listen_fd, n_pes) == 0;
    std::vector<int> peer_fds(static_cast<size_t>(n_pes), -1);
    auto* gathered = static_cast<uint8_t*>(all_data);
    std::memcpy(gathered, local_data, size);
    for (int connected = 1; success && connected < n_pes; ++connected) {
        int fd = accept(listen_fd, nullptr, nullptr);
        uint32_t rank = 0;
        success = fd >= 0 && setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0 &&
                  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0 &&
                  recv_all(fd, &rank, sizeof(rank));
        int peer = success ? static_cast<int>(ntohl(rank)) : -1;
        success = success && peer > 0 && peer < n_pes && peer_fds[peer] < 0 &&
                  recv_all(fd, gathered + static_cast<size_t>(peer) * size, size);
        if (success) {
            peer_fds[peer] = fd;
        } else if (fd >= 0) {
            close(fd);
        }
    }
    for (int peer = 1; success && peer < n_pes; ++peer) {
        success = send_all(peer_fds[peer], gathered, total_size);
    }
    for (int fd : peer_fds) {
        if (fd >= 0) {
            close(fd);
        }
    }
    if (listen_fd >= 0) {
        close(listen_fd);
    }
    if (!success) {
        std::cerr << "TCP allgather " << tag << " failed on PE 0" << std::endl;
    }
    return success;
}

int get_peer_pe(int pe_id, int n_pes) { return (pe_id + 1) % n_pes; }

struct HandleExchangeInfo {
    int32_t status = 0;
    aclrtMemFabricHandle one_path_handle{};
    aclrtMemFabricHandle multi_path_handle{};
};

class MemMapping {
public:
    MemMapping() = default;
    MemMapping(const MemMapping&) = delete;
    MemMapping& operator=(const MemMapping&) = delete;
    ~MemMapping() { reset(); }

    int alloc_local(size_t size, int device_id)
    {
        size_ = size;
        aclrtPhysicalMemProp prop{};
        prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
        prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
        prop.memAttr = ACL_HBM_MEM_HUGE;
        prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = static_cast<uint32_t>(device_id);

        aclError ret = aclrtMallocPhysical(&handle_, size_, &prop, 0);
        if (ret != ACL_SUCCESS || handle_ == nullptr) {
            std::cerr << "aclrtMallocPhysical failed, size=" << size_ << ", ret=" << ret << std::endl;
            return -1;
        }
        return reserve_and_map();
    }

    int import_remote(const aclrtMemFabricHandle& share_handle)
    {
        aclrtMemFabricHandle handle = share_handle;
        aclError ret = aclrtMemImportFromShareableHandleV2(&handle, ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, 0, &handle_);
        if (ret != ACL_SUCCESS || handle_ == nullptr) {
            std::cerr << "aclrtMemImportFromShareableHandleV2 failed, ret=" << ret << std::endl;
            return -1;
        }
        return 0;
    }

    int map_imported(size_t size)
    {
        size_ = size;
        return reserve_and_map();
    }

    aclrtDrvMemHandle handle() const { return handle_; }
    void* address() const { return va_; }

private:
    int reserve_and_map()
    {
        aclError ret = aclrtReserveMemAddress(&va_, size_, 0, nullptr, RESERVE_FLAGS_ASCEND950);
        if (ret != ACL_SUCCESS || va_ == nullptr) {
            std::cerr << "aclrtReserveMemAddress failed, size=" << size_ << ", ret=" << ret << std::endl;
            return -1;
        }
        reserved_ = true;
        ret = aclrtMapMem(va_, size_, 0, handle_, 0);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtMapMem failed, ret=" << ret << std::endl;
            return -1;
        }
        mapped_ = true;
        return 0;
    }

    void reset()
    {
        if (mapped_ && va_ != nullptr) {
            aclError ret = aclrtUnmapMem(va_);
            if (ret != ACL_SUCCESS) {
                std::cerr << "aclrtUnmapMem failed, ret=" << ret << std::endl;
            }
            mapped_ = false;
        }
        if (reserved_ && va_ != nullptr) {
            aclError ret = aclrtReleaseMemAddress(va_);
            if (ret != ACL_SUCCESS) {
                std::cerr << "aclrtReleaseMemAddress failed, ret=" << ret << std::endl;
            }
            reserved_ = false;
            va_ = nullptr;
        }
        if (handle_ != nullptr) {
            aclError ret = aclrtFreePhysical(handle_);
            if (ret != ACL_SUCCESS) {
                std::cerr << "aclrtFreePhysical failed, ret=" << ret << std::endl;
            }
            handle_ = nullptr;
        }
    }

    aclrtDrvMemHandle handle_ = nullptr;
    void* va_ = nullptr;
    size_t size_ = 0;
    bool reserved_ = false;
    bool mapped_ = false;
};

bool validate_result(const int32_t* data, size_t count, int peer_pe)
{
    int32_t expected = peer_pe + DATA_OFFSET;
    for (size_t i = 0; i < count; ++i) {
        if (data[i] != expected) {
            std::cerr << "data mismatch at index " << i << ": got " << data[i] << ", expected " << expected
                      << std::endl;
            return false;
        }
    }
    return true;
}

int export_fabric_handle(aclrtDrvMemHandle local_handle, aclrtMemFabricHandle& share_handle)
{
    aclError ret = aclrtMemExportToShareableHandleV2(
        local_handle, ACL_RT_VMM_EXPORT_FLAG_DISABLE_PID_VALIDATION, ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, &share_handle);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtMemExportToShareableHandleV2 failed, ret=" << ret << std::endl;
        return -1;
    }
    return 0;
}

int create_remote_mapping(
    const aclrtMemFabricHandle& peer_handle, const char* path_name, uint32_t link_type, size_t map_size,
    MemMapping& mapping)
{
    if (mapping.import_remote(peer_handle) != 0) {
        std::cerr << "path " << path_name << ": import remote handle failed" << std::endl;
        return -1;
    }
    aclError ret = aclrtMemMapSetLink(mapping.handle(), static_cast<aclrtMemLinkType>(link_type));
    if (ret != ACL_SUCCESS) {
        std::cerr << "path " << path_name << ": aclrtMemMapSetLink failed, link_type=" << link_type << ", ret=" << ret
                  << std::endl;
        return -1;
    }
    if (mapping.map_imported(map_size) != 0) {
        std::cerr << "path " << path_name << ": map imported handle failed" << std::endl;
        return -1;
    }
    std::cout << "path " << path_name << ": link_type=" << link_type << ", remote_va=" << mapping.address()
              << std::endl;
    return 0;
}

bool calculate_split_size(size_t data_size, size_t& split_size)
{
    if (data_size < 2 * PATH_SLICE_ALIGNMENT || data_size % (2 * PATH_SLICE_ALIGNMENT) != 0) {
        return false;
    }
    split_size = data_size / 2;
    return true;
}

int copy_and_validate(void* result_buffer, int32_t* result_host, size_t data_size, int peer_pe)
{
    aclError ret = aclrtMemcpy(result_host, data_size, result_buffer, data_size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtMemcpy result to host failed, ret=" << ret << std::endl;
        return -1;
    }
    if (!validate_result(result_host, data_size / sizeof(int32_t), peer_pe)) {
        return -1;
    }
    return 0;
}

int run_single_core_copy(
    int pe_id, int peer_pe, MemMapping& one_path_mapping, MemMapping& multi_path_mapping, size_t data_size,
    void* result_buffer, int32_t* result_host)
{
    size_t split_size = 0;
    if (!calculate_split_size(data_size, split_size)) {
        std::cerr << "failed to split data across one_path and multi_path" << std::endl;
        return -1;
    }
    std::cout << "PE " << pe_id << ": path=one_path, offset=0, size=" << split_size << std::endl;
    std::cout << "PE " << pe_id << ": path=multi_path, offset=" << split_size << ", size=" << data_size - split_size
              << std::endl;

    aclrtStream stream = nullptr;
    aclError ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS || stream == nullptr) {
        std::cerr << "aclrtCreateStream failed, ret=" << ret << std::endl;
        return -1;
    }
    launch_one_multi_path_copy(
        stream, reinterpret_cast<uint8_t*>(result_buffer), reinterpret_cast<uint8_t*>(one_path_mapping.address()),
        reinterpret_cast<uint8_t*>(multi_path_mapping.address()), static_cast<uint64_t>(split_size),
        static_cast<uint64_t>(data_size));

    int status = 0;
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtSynchronizeStream failed, ret=" << ret << std::endl;
        status = -1;
    }
    if (status == 0 && copy_and_validate(result_buffer, result_host, data_size, peer_pe) != 0) {
        std::cerr << "PE " << pe_id << ": single-core one_path/multi_path verification FAILED" << std::endl;
        status = -1;
    }
    if (status == 0) {
        std::cout << "PE " << pe_id << ": single-core one_path/multi_path copy from PE " << peer_pe << " PASSED"
                  << std::endl;
    }

    aclError destroy_ret = aclrtDestroyStream(stream);
    if (destroy_ret != ACL_SUCCESS) {
        std::cerr << "aclrtDestroyStream failed, ret=" << destroy_ret << std::endl;
    }
    return status;
}

void print_usage(const char* program)
{
    std::cerr << "Usage: " << program
              << " <n_pes:>=2> <pe_id:0..n_pes-1> <ipport:tcp://ip:port> <g_npus> <f_pe> <f_npu> <size_kb>"
              << std::endl;
}

int init_acl_shmem(int pe_id, int n_pes, int device_id, uint64_t local_mem_size, const char* ip_port)
{
    int status = 0;
    status |= aclInit(nullptr);
    status |= aclrtSetDevice(device_id);

    aclshmemx_uniqueid_t default_flag_uid{};
    aclshmemx_init_attr_t attributes{};
    status |= test_set_attr(pe_id, n_pes, local_mem_size, ip_port, default_flag_uid, &attributes);
    status |= aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    return status;
}

int test_one_multi_path(int n_pes, int pe_id, int device_id, const char* ip_port, size_t data_size, size_t map_size)
{
    int status = 0;
    MemMapping one_path_local_mapping;
    MemMapping multi_path_local_mapping;
    if (one_path_local_mapping.alloc_local(map_size, device_id) != 0) {
        std::cerr << "allocate one_path local mapping failed" << std::endl;
        status = -1;
    }
    if (status == 0 && multi_path_local_mapping.alloc_local(map_size, device_id) != 0) {
        std::cerr << "allocate multi_path local mapping failed" << std::endl;
        status = -1;
    }

    int32_t* init_data = nullptr;
    if (status == 0) {
        aclError ret = aclrtMallocHost(reinterpret_cast<void**>(&init_data), data_size);
        if (ret != ACL_SUCCESS || init_data == nullptr) {
            std::cerr << "aclrtMallocHost init data failed, size=" << data_size << ", ret=" << ret << std::endl;
            status = -1;
        } else {
            for (size_t idx = 0; idx < data_size / sizeof(int32_t); ++idx) {
                init_data[idx] = pe_id + DATA_OFFSET;
            }
            ret = aclrtMemcpy(
                one_path_local_mapping.address(), map_size, init_data, data_size, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS) {
                std::cerr << "aclrtMemcpy host to one_path local mapping failed, ret=" << ret << std::endl;
                status = -1;
            }
            if (status == 0) {
                ret = aclrtMemcpy(
                    multi_path_local_mapping.address(), map_size, init_data, data_size, ACL_MEMCPY_HOST_TO_DEVICE);
                if (ret != ACL_SUCCESS) {
                    std::cerr << "aclrtMemcpy host to multi_path local mapping failed, ret=" << ret << std::endl;
                    status = -1;
                }
            }
        }
    }
    if (init_data != nullptr) {
        aclError ret = aclrtFreeHost(init_data);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtFreeHost init data failed, ret=" << ret << std::endl;
        }
    }

    HandleExchangeInfo local_exchange{};
    if (status == 0 && export_fabric_handle(one_path_local_mapping.handle(), local_exchange.one_path_handle) != 0) {
        std::cerr << "export one_path fabric handle failed" << std::endl;
        status = -1;
    }
    if (status == 0 && export_fabric_handle(multi_path_local_mapping.handle(), local_exchange.multi_path_handle) != 0) {
        std::cerr << "export multi_path fabric handle failed" << std::endl;
        status = -1;
    }

    local_exchange.status = status;
    std::vector<HandleExchangeInfo> all_exchange(n_pes);
    if (!tcp_allgather(
            &local_exchange, all_exchange.data(), sizeof(HandleExchangeInfo), pe_id, n_pes, ip_port,
            TCP_FABRIC_HANDLES_PORT_OFFSET, "fabric handles")) {
        status = -1;
    }

    int peer_pe = get_peer_pe(pe_id, n_pes);
    if (status == 0 && all_exchange[peer_pe].status != 0) {
        std::cerr << "PE " << pe_id << ": peer handles export failed: " << all_exchange[peer_pe].status << std::endl;
        status = -1;
    }

    MemMapping one_path_mapping;
    MemMapping multi_path_mapping;
    if (status == 0 &&
        create_remote_mapping(
            all_exchange[peer_pe].one_path_handle, "one_path", ONE_PATH_LINK_TYPE, map_size, one_path_mapping) != 0) {
        status = -1;
    }
    if (status == 0 && create_remote_mapping(
                           all_exchange[peer_pe].multi_path_handle, "multi_path", MULTI_PATH_LINK_TYPE, map_size,
                           multi_path_mapping) != 0) {
        status = -1;
    }

    void* result_buffer = nullptr;
    if (status == 0) {
        aclError ret = aclrtMalloc(&result_buffer, data_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS || result_buffer == nullptr) {
            std::cerr << "aclrtMalloc failed, size=" << data_size << ", ret=" << ret << std::endl;
            status = -1;
        }
    }

    int32_t* result_host = nullptr;
    if (status == 0) {
        aclError ret = aclrtMallocHost(reinterpret_cast<void**>(&result_host), data_size);
        if (ret != ACL_SUCCESS || result_host == nullptr) {
            std::cerr << "aclrtMallocHost failed, size=" << data_size << ", ret=" << ret << std::endl;
            status = -1;
        }
    }

    if (status == 0 &&
        run_single_core_copy(
            pe_id, peer_pe, one_path_mapping, multi_path_mapping, data_size, result_buffer, result_host) != 0) {
        status = -1;
    }

    int32_t local_status = static_cast<int32_t>(status);
    std::vector<int32_t> all_status(n_pes, -1);
    int final_status = status;
    if (!tcp_allgather(
            &local_status, all_status.data(), sizeof(local_status), pe_id, n_pes, ip_port, TCP_STATUS_PORT_OFFSET,
            "status")) {
        final_status = -1;
    } else {
        for (int idx = 0; idx < n_pes; ++idx) {
            if (all_status[idx] != 0) {
                std::cerr << "PE " << pe_id << ": PE " << idx << " reported failure: " << all_status[idx] << std::endl;
                final_status = -1;
                break;
            }
        }
    }

    if (result_host != nullptr) {
        aclError ret = aclrtFreeHost(result_host);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtFreeHost failed, ret=" << ret << std::endl;
        }
    }
    if (result_buffer != nullptr) {
        aclError ret = aclrtFree(result_buffer);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtFree failed, ret=" << ret << std::endl;
        }
    }
    return final_status;
}

int run_one_multi_path(int n_pes, int pe_id, int device_id, const char* ip_port, size_t data_size, size_t map_size)
{
    int status = init_acl_shmem(pe_id, n_pes, device_id, map_size, ip_port);
    if (status != ACLSHMEM_SUCCESS) {
        std::cerr << "init_acl_shmem failed, ret=" << status << std::endl;
        return status;
    }

    status = test_one_multi_path(n_pes, pe_id, device_id, ip_port, data_size, map_size);

    ACL_CHECK(aclshmem_finalize());
    ACL_CHECK(aclrtResetDevice(device_id));
    ACL_CHECK(aclFinalize());
    return status;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc != 8) {
        print_usage(argv[0]);
        return -1;
    }

    int n_pes = 0;
    int pe_id = 0;
    const char* ip_port = argv[3];
    int g_npus = 0;
    int first_pe = 0;
    int first_npu = 0;
    size_t data_size = 0;

    if (!parse_int(argv[1], n_pes) || !parse_int(argv[2], pe_id) || ip_port == nullptr || ip_port[0] == '\0' ||
        !parse_int(argv[4], g_npus) || !parse_int(argv[5], first_pe) || !parse_int(argv[6], first_npu) ||
        !parse_size_kb(argv[7], data_size) || n_pes < MIN_PE_COUNT || pe_id < 0 || pe_id >= n_pes || g_npus <= 0 ||
        g_npus > n_pes || first_pe < 0 || first_npu < 0 || first_pe > n_pes - g_npus) {
        std::cerr << "Invalid arguments: require n_pes >= " << MIN_PE_COUNT
                  << ", valid PE/NPU mapping, non-empty ipport, and size_kb > 0" << std::endl;
        return -1;
    }

    int local_pe = pe_id - first_pe;
    if (local_pe < 0 || local_pe >= g_npus) {
        std::cerr << "Invalid local PE mapping, pe_id=" << pe_id << ", f_pe=" << first_pe << ", g_npus=" << g_npus
                  << std::endl;
        return -1;
    }
    if (first_npu > std::numeric_limits<int>::max() - local_pe) {
        std::cerr << "Invalid NPU mapping, f_npu=" << first_npu << ", local_pe=" << local_pe << std::endl;
        return -1;
    }
    int device_id = local_pe + first_npu;

    size_t map_size = 0;
    if (!align_up(data_size, DEVICE_LARGE_PAGE_SIZE, map_size)) {
        std::cerr << "Invalid data size for mapping: " << data_size << std::endl;
        return -1;
    }

    std::cout << "PE " << pe_id << ": device=" << device_id << ", ipport=" << ip_port
              << ", peer=" << get_peer_pe(pe_id, n_pes) << ", one_path_link_type=" << ONE_PATH_LINK_TYPE
              << ", multi_path_link_type=" << MULTI_PATH_LINK_TYPE << ", data_size=" << data_size
              << ", map_size=" << map_size << std::endl;

    return run_one_multi_path(n_pes, pe_id, device_id, ip_port, data_size, map_size);
}
