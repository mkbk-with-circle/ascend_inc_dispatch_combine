/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "pcie_nic.h"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <arpa/inet.h>
#include <dirent.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <securec.h>

#include "aclshmemi_hal.h"
#include "pcie_nic_matcher.h"
#include "utils/shmemi_logger.h"

constexpr int MAX_NIC_PATH = 256 + 32;
constexpr int MAX_NIC_COUNT = 32;
constexpr int MAX_PCIE_BUS_ID_LEN = 32;

typedef struct {
    char path[MAX_NIC_PATH];
    char pcie_path[MAX_NIC_PATH];
    size_t comon_prefix_len;
} nic;

typedef struct _stNPU {
    int id = 0;
    char bus_id[MAX_PCIE_BUS_ID_LEN] = {0};
    char path[MAX_NIC_PATH] = {0};
    char nic_path[MAX_NIC_PATH] = {0};
    char nic_name[MAX_NIC_PATH] = {0};
} NPU;

static int get_abs_path(const char* path, char* abs_path, size_t abs_path_len)
{
    char resolved_path[PATH_MAX] = {0};
    char* p = realpath(path, resolved_path);
    if (p == NULL) {
        return -1;
    }
    return strcpy_s(abs_path, abs_path_len, resolved_path);
}

int get_nics(nic* nics, size_t* nic_len)
{
    DIR* dir = opendir("/sys/class/net");
    if (dir == NULL) {
        return -1;
    }

    int loop = 0;
    struct dirent* entry = NULL;
    while ((entry = readdir(dir)) != NULL) {
        if (loop >= MAX_NIC_COUNT) {
            break;
        }
        if (entry->d_name[0] == '.') {
            continue;
        }
        if (strcmp(entry->d_name, "lo") == 0) {
            continue;
        }

        int ret = sprintf_s(nics[loop].path, MAX_NIC_PATH, "/sys/class/net/%s", entry->d_name);
        if (ret < 0) {
            closedir(dir);
            SHM_LOG_ERROR("get_nics failed: sprintf_s failed for NIC " << entry->d_name);
            return -1;
        }

        get_abs_path(nics[loop].path, nics[loop].pcie_path, MAX_NIC_PATH);
        loop++;
    }

    *nic_len = loop;
    closedir(dir);
    return 0;
}

// Scan /sys/class/infiniband for RDMA HCA devices and get their ethernet interface names.
// For each HCA, looks in <hca>/device/net/* to find the associated ethernet device.
// Returns the PCIe path of the HCA device.
static int get_rdma_nics(nic* nics, size_t* nic_len)
{
    DIR* dir = opendir("/sys/class/infiniband");
    if (dir == NULL) {
        return -1;
    }

    int loop = 0;
    struct dirent* entry = NULL;
    while ((entry = readdir(dir)) != NULL) {
        if (loop >= MAX_NIC_COUNT) {
            break;
        }
        if (entry->d_name[0] == '.') {
            continue;
        }

        // Get the HCA device PCIe path
        char device_path[MAX_NIC_PATH] = {0};
        int ret = sprintf_s(device_path, MAX_NIC_PATH, "/sys/class/infiniband/%s/device", entry->d_name);
        if (ret < 0) {
            continue;
        }

        char pcie_device_path[MAX_NIC_PATH] = {0};
        if (get_abs_path(device_path, pcie_device_path, MAX_NIC_PATH) != 0) {
            continue;
        }

        // Find the ethernet interface name under device/net/
        char net_dir[MAX_NIC_PATH] = {0};
        ret = sprintf_s(net_dir, MAX_NIC_PATH, "/sys/class/infiniband/%s/device/net", entry->d_name);
        if (ret < 0) {
            continue;
        }

        DIR* net_dir_handle = opendir(net_dir);
        if (net_dir_handle == NULL) {
            continue;
        }

        struct dirent* net_entry = NULL;
        char eth_name[MAX_NIC_PATH] = {0};
        while ((net_entry = readdir(net_dir_handle)) != NULL) {
            if (net_entry->d_name[0] == '.') {
                continue;
            }
            // Use the first non-dot entry as the ethernet device name
            (void)strcpy_s(eth_name, sizeof(eth_name), net_entry->d_name);
            break;
        }
        closedir(net_dir_handle);

        if (eth_name[0] == '\0') {
            continue;
        }

        // Store the PCIe path of the HCA device
        (void)strcpy_s(nics[loop].pcie_path, sizeof(nics[loop].pcie_path), pcie_device_path);
        // Store the ethernet interface name (which is what get_ip_by_if_name needs)
        ret = sprintf_s(nics[loop].path, MAX_NIC_PATH, "/sys/class/net/%s", eth_name);
        if (ret < 0) {
            continue;
        }
        loop++;
    }

    *nic_len = loop;
    closedir(dir);
    return 0;
}

// Helper: get the serial number of the PCIe switch that a device is behind.
// Equivalent to Python: lspci -vv -s {bdf}:00 | grep Serial
static std::string get_pcie_switch_serial(const char* pcie_path)
{
    std::string path(pcie_path);
    if (path.size() <= shm::topo::PCI_PATH_TAIL_LEN) {
        return "";
    }
    std::string parent = path.substr(0, path.size() - shm::topo::PCI_PATH_TAIL_LEN);
    size_t last_slash = parent.rfind('/');
    if (last_slash == std::string::npos) {
        return "";
    }
    std::string bdf = parent.substr(last_slash + 1);
    std::string addr = bdf + ":00";

    // BDF must only contain hex digits, colons, and dots — reject anything else
    for (char c : addr) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') || c == ':' || c == '.')) {
            SHM_LOG_DEBUG("get_pcie_switch_serial: invalid BDF address " << addr);
            return "";
        }
    }

    char cmd[256] = {0};
    int ret = snprintf_s(cmd, sizeof(cmd), sizeof(cmd) - 1, "lspci -vv -s %s 2>/dev/null", addr.c_str());
    if (ret < 0) {
        return "";
    }

    FILE* fp = popen(cmd, "r");
    if (fp == nullptr) {
        SHM_LOG_DEBUG("get_pcie_switch_serial: popen failed for " << addr);
        return "";
    }

    char line[256] = {0};
    std::string result;
    while (fgets(line, sizeof(line), fp) != nullptr) {
        if (strstr(line, "Serial") != nullptr) {
            result = line;
            break;
        }
    }
    pclose(fp);

    // trim trailing whitespace
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
        result.pop_back();
    }

    if (result.empty()) {
        SHM_LOG_DEBUG("get_pcie_switch_serial: no Serial found for " << addr);
    }
    return result;
}

namespace shm {
namespace topo {

bool is_pix_topo(const char* path_a, const char* path_b)
{
    if (path_a == nullptr || path_b == nullptr) {
        return false;
    }
    size_t len_a = strlen(path_a);
    size_t len_b = strlen(path_b);
    if (len_a <= PCI_PATH_TAIL_LEN || len_b <= PCI_PATH_TAIL_LEN) {
        return false;
    }
    size_t prefix_a = len_a - PCI_PATH_TAIL_LEN;
    size_t prefix_b = len_b - PCI_PATH_TAIL_LEN;

    // Step 1: same prefix → under the same PCIe switch
    if (prefix_a == prefix_b && strncmp(path_a, path_b, prefix_a) == 0) {
        return true;
    }

    // Step 2: different switches → compare switch serials (Python lspci fallback)
    std::string serial_a = get_pcie_switch_serial(path_a);
    std::string serial_b = get_pcie_switch_serial(path_b);
    return !serial_a.empty() && serial_a == serial_b;
}

} // namespace topo
} // namespace shm

// Round-robin PIX topology matching, equivalent to Python's match_npu_hca.
// NPUs under the same PCIe switch share the same NIC (NDR affinity).
// Hardware constraint: each PCIe switch has at most one RDMA HCA, so round-robin
// naturally produces the correct one-to-one switch-to-HCA mapping. If future
// hardware adds multiple HCAs per switch, this logic would need revision.
// If no PIX match is found, returns -1 (no fallback).
static int get_pcie_nics_round_robin(NPU* npu, size_t pos, const nic* nics, size_t nic_len, int& cur)
{
    if (nic_len == 0) {
        return -1;
    }
    char dev_path[MAX_NIC_PATH] = {0};
    char abs_path[MAX_NIC_PATH] = {0};
    size_t abs_path_len = sizeof(abs_path);
    int ret = sprintf_s(dev_path, sizeof(dev_path), "/sys/bus/pci/devices/%s", npu[pos].bus_id);
    if (ret < 0) {
        return -1;
    }
    if (get_abs_path(dev_path, abs_path, abs_path_len) != 0) {
        return -1;
    }

    const char* nic_pcie_paths[MAX_NIC_COUNT] = {0};
    for (size_t i = 0; i < nic_len && i < MAX_NIC_COUNT; ++i) {
        nic_pcie_paths[i] = nics[i].pcie_path;
    }

    const int nic_count = static_cast<int>(nic_len);
    for (int j = cur; j < nic_count + cur; ++j) {
        int p = j % nic_count;
        if (nic_pcie_paths[p] == nullptr || nic_pcie_paths[p][0] == '\0') {
            continue;
        }
        if (shm::topo::is_pix_topo(abs_path, nic_pcie_paths[p])) {
            (void)strcpy_s(npu[pos].nic_path, sizeof(npu[pos].nic_path), nics[p].pcie_path);
            (void)strcpy_s(npu[pos].nic_name, sizeof(npu[pos].nic_name), shm::topo::get_path_basename(nics[p].path));
            cur = j + 1;
            return 0;
        }
    }

    SHM_LOG_DEBUG("get_npu_roce_ip: no PIX match for npu_id=" << npu[pos].id);
    return -1;
}

static void init_npus(NPU* npu, size_t npu_count)
{
    memset_s(npu, sizeof(NPU) * npu_count, 0, sizeof(NPU) * npu_count);
    auto& hal = shm::topo::aclshmemi_hal_t::instance();

    for (size_t i = 0; i < npu_count; ++i) {
        npu[i].id = i;

        struct dcmi_pcie_info_all pcie_info;
        memset_s(&pcie_info, sizeof(pcie_info), 0, sizeof(pcie_info));
        int ret = hal.get_device_pcie_info(i, &pcie_info);
        if (ret != 0) {
            SHM_LOG_DEBUG("init_npus: get_device_pcie_info failed for npu_id=" << i << ", skipping");
            continue;
        }

        ret = sprintf_s(
            npu[i].bus_id, MAX_PCIE_BUS_ID_LEN, "%04x:%02x:%02x.%x", pcie_info.domain, pcie_info.bdf_busid,
            pcie_info.bdf_deviceid, pcie_info.bdf_funcid);
        if (ret < 0) {
            SHM_LOG_ERROR("init_npus failed: sprintf_s bus_id failed for npu_id=" << i);
            continue;
        }
    }
}

int get_ip_by_if_name(const char* ifname, char* ip, size_t addr_len)
{
    struct ifaddrs* ifaddr = NULL;
    if (getifaddrs(&ifaddr) == -1) {
        return -1;
    }
    int found_interface = 0;
    for (struct ifaddrs* ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr || !ifa->ifa_name) {
            continue;
        }
        if (strcmp(ifname, ifa->ifa_name) != 0) {
            continue;
        }
        found_interface = 1;
        if (ifa->ifa_addr->sa_family == AF_INET) {
            struct sockaddr_in* sin = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
            if (inet_ntop(AF_INET, &sin->sin_addr, ip, addr_len) != NULL) {
                freeifaddrs(ifaddr);
                return 0;
            }
            freeifaddrs(ifaddr);
            return -1;
        }
    }
    freeifaddrs(ifaddr);
    errno = found_interface ? ENOENT : ENODEV;
    return -1;
}

int get_npu_roce_ip(int npu_id, char* ip_addr, size_t ip_addr_len)
{
    auto& hal = shm::topo::aclshmemi_hal_t::instance();
    auto npu_count_opt = hal.get_npu_count();
    if (!npu_count_opt) {
        return -1;
    }

    int npu_count = *npu_count_opt;

    // 边界校验：防止负值或超大值导致内存过度分配
    if (npu_count <= 0 || npu_count > ACLSHMEMI_MAX_NPU_COUNT) {
        SHM_LOG_ERROR(
            "get_npu_roce_ip failed: invalid npu_count=" << npu_count << ", expected range [1, "
                                                         << ACLSHMEMI_MAX_NPU_COUNT << "]");
        return -1;
    }

    std::vector<NPU> npus(npu_count);
    init_npus(npus.data(), npu_count);

    nic nics[MAX_NIC_COUNT];
    memset_s(nics, sizeof(nics), 0, sizeof(nics));
    size_t nic_len = 0;

    // Phase 1: try to find RDMA HCAs via /sys/class/infiniband for more accurate matching
    if (get_rdma_nics(nics, &nic_len) != 0 || nic_len == 0) {
        SHM_LOG_DEBUG("get_npu_roce_ip: no RDMA HCA found via /sys/class/infiniband, falling back to /sys/class/net");
        // Phase 2: fall back to generic /sys/class/net scan
        memset_s(nics, sizeof(nics), 0, sizeof(nics));
        nic_len = 0;
        if (get_nics(nics, &nic_len) != 0) {
            SHM_LOG_DEBUG("get_npu_roce_ip: get_nics failed");
            return -1;
        }
        if (nic_len == 0) {
            SHM_LOG_DEBUG("get_npu_roce_ip: no NIC found");
            return -1;
        }
    } else {
        SHM_LOG_DEBUG("get_npu_roce_ip: found " << nic_len << " RDMA HCA device(s) via /sys/class/infiniband");
    }

    int cur = 0;
    for (int i = 0; i < npu_count; ++i) {
        if (npus[i].bus_id[0] == '\0') {
            SHM_LOG_DEBUG("get_npu_roce_ip: skipping npu_id=" << i << " (no PCIe bus ID)");
            continue;
        }
        int ret = get_pcie_nics_round_robin(npus.data(), i, nics, nic_len, cur);
        if (ret != 0) {
            SHM_LOG_DEBUG("get_npu_roce_ip: get_pcie_nics failed for npu_id=" << i);
            continue;
        }
    }

    for (int i = 0; i < npu_count; ++i) {
        if (npus[i].id == npu_id) {
            if (npus[i].nic_name[0] == '\0') {
                SHM_LOG_DEBUG("get_npu_roce_ip: no matched PCIe NIC for npu_id=" << npu_id);
                errno = ENODEV;
                return -1;
            }
            return get_ip_by_if_name(npus[i].nic_name, ip_addr, ip_addr_len);
        }
    }

    errno = ENODEV;
    return -1;
}
