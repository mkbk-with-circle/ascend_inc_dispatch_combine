/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclshmemi_npu_nic_affinity.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cctype>
#include <dirent.h>
#include <ifaddrs.h>
#include <sys/socket.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include <securec.h>

#include "aclshmemi_hal.h"
#include "aclshmemi_xml_parser.h"
#include "utils/shmemi_logger.h"

namespace shm {
namespace topo {

namespace {

constexpr int MAX_AFFINITY_GROUP_COUNT = 16;
constexpr int MAX_NIC_COUNT = 64;
constexpr int MAX_NIC_NAME_LENGTH = 64;
constexpr int MAX_IP_LENGTH = INET_ADDRSTRLEN;
constexpr const char* HCA_NET_PATH_PREFIX = "/sys/class/infiniband/";
constexpr const char* HCA_NET_PATH_SUFFIX = "/device/net";

struct affinity_group_t {
    std::vector<int> npu_ids;
    std::vector<size_t> nic_indices;
};

struct affinity_info_t {
    std::vector<std::string> nic_names;
    std::vector<affinity_group_t> groups;
    std::vector<std::vector<bool>> affined;
};

bool is_group_tag(const aclshmemi_xml_tag_t& tag) { return tag.name == "pci" || tag.name == "ub"; }

bool has_group_ancestor(const std::vector<aclshmemi_xml_tag_t>& tags, const aclshmemi_xml_tag_t& tag)
{
    int parent = tag.parent_index;
    while (parent >= 0) {
        const auto& parent_tag = tags[static_cast<size_t>(parent)];
        if (is_group_tag(parent_tag)) {
            return true;
        }
        parent = parent_tag.parent_index;
    }
    return false;
}

bool is_descendant_or_self(const std::vector<aclshmemi_xml_tag_t>& tags, size_t tag_index, size_t root_index)
{
    int current = static_cast<int>(tag_index);
    while (current >= 0) {
        if (static_cast<size_t>(current) == root_index) {
            return true;
        }
        current = tags[static_cast<size_t>(current)].parent_index;
    }
    return false;
}

bool normalize_bdf(const std::string& value, std::string& normalized)
{
    constexpr size_t BDF_LENGTH = 12;
    if (value.size() != BDF_LENGTH || value[4] != ':' || value[7] != ':' || value[10] != '.') {
        return false;
    }
    for (size_t i = 0; i < value.size(); ++i) {
        if (i == 4 || i == 7 || i == 10) {
            continue;
        }
        if (std::isxdigit(static_cast<unsigned char>(value[i])) == 0) {
            return false;
        }
    }
    normalized = value;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return true;
}

bool parse_phy_id(const std::string& value, int& phy_id)
{
    if (value.empty()) {
        return false;
    }

    constexpr int MAX_PHY_ID = ACLSHMEMI_MAX_NPU_COUNT - 1;
    int parsed = 0;
    for (char ch : value) {
        if (ch < '0' || ch > '9') {
            return false;
        }
        const int digit = ch - '0';
        if (parsed > (MAX_PHY_ID - digit) / 10) {
            return false;
        }
        parsed = parsed * 10 + digit;
    }

    phy_id = parsed;
    return true;
}

bool is_safe_device_name(const std::string& name)
{
    if (name.empty() || name.size() >= static_cast<size_t>(MAX_NIC_NAME_LENGTH) || name == "." || name == "..") {
        return false;
    }
    for (char ch : name) {
        const auto value = static_cast<unsigned char>(ch);
        if (std::isalnum(value) == 0 && ch != '_' && ch != '-' && ch != '.') {
            return false;
        }
    }
    return true;
}

std::optional<std::string> get_interface_ipv4(const std::string& interface_name)
{
    struct ifaddrs* interfaces = nullptr;
    if (getifaddrs(&interfaces) != 0) {
        return std::nullopt;
    }

    std::optional<std::string> result;
    for (struct ifaddrs* current = interfaces; current != nullptr; current = current->ifa_next) {
        if (current->ifa_name == nullptr || current->ifa_addr == nullptr || interface_name != current->ifa_name ||
            current->ifa_addr->sa_family != AF_INET) {
            continue;
        }
        const auto* address = reinterpret_cast<const struct sockaddr_in*>(current->ifa_addr);
        char buffer[MAX_IP_LENGTH] = {0};
        if (inet_ntop(AF_INET, &address->sin_addr, buffer, sizeof(buffer)) != nullptr) {
            result = std::string(buffer);
            break;
        }
    }
    freeifaddrs(interfaces);
    return result;
}

std::vector<std::string> get_hca_interfaces(const std::string& hca_name)
{
    const std::string path = std::string(HCA_NET_PATH_PREFIX) + hca_name + HCA_NET_PATH_SUFFIX;
    DIR* directory = opendir(path.c_str());
    if (directory == nullptr) {
        return {};
    }

    std::vector<std::string> interfaces;
    struct dirent* entry = nullptr;
    while ((entry = readdir(directory)) != nullptr) {
        if (entry->d_name[0] == '.') {
            continue;
        }
        const std::string interface_name(entry->d_name);
        if (is_safe_device_name(interface_name)) {
            interfaces.push_back(interface_name);
        }
    }
    closedir(directory);
    std::sort(interfaces.begin(), interfaces.end());
    return interfaces;
}

std::optional<std::string> format_npu_bdf(int phy_id)
{
    struct dcmi_pcie_info_all pcie_info;
    if (memset_s(&pcie_info, sizeof(pcie_info), 0, sizeof(pcie_info)) != EOK) {
        return std::nullopt;
    }
    auto& hal = aclshmemi_hal_t::instance();
    if (hal.get_device_pcie_info(phy_id, &pcie_info) != 0) {
        return std::nullopt;
    }

    char bdf[32] = {0};
    const int ret = snprintf_s(
        bdf, sizeof(bdf), sizeof(bdf) - 1, "%04x:%02x:%02x.%x", pcie_info.domain, pcie_info.bdf_busid,
        pcie_info.bdf_deviceid, pcie_info.bdf_funcid);
    if (ret < 0) {
        return std::nullopt;
    }
    std::string normalized;
    if (!normalize_bdf(bdf, normalized)) {
        return std::nullopt;
    }
    return normalized;
}

void add_unique_npu(affinity_group_t& group, int phy_id)
{
    if (std::find(group.npu_ids.begin(), group.npu_ids.end(), phy_id) == group.npu_ids.end()) {
        group.npu_ids.push_back(phy_id);
    }
}

bool add_unique_nic(affinity_info_t& info, affinity_group_t& group, const std::string& name, std::string& error_message)
{
    if (!is_safe_device_name(name)) {
        error_message = "invalid NIC/HCA name '" + name + "'";
        return false;
    }

    auto it = std::find(info.nic_names.begin(), info.nic_names.end(), name);
    size_t nic_index = 0;
    if (it == info.nic_names.end()) {
        if (info.nic_names.size() >= static_cast<size_t>(MAX_NIC_COUNT)) {
            error_message = "virtual topology contains too many NICs";
            return false;
        }
        nic_index = info.nic_names.size();
        info.nic_names.push_back(name);
    } else {
        nic_index = static_cast<size_t>(std::distance(info.nic_names.begin(), it));
    }
    if (std::find(group.nic_indices.begin(), group.nic_indices.end(), nic_index) == group.nic_indices.end()) {
        group.nic_indices.push_back(nic_index);
    }
    return true;
}

bool build_bdf_table(std::unordered_map<std::string, int>& bdf_to_npu)
{
    auto& hal = aclshmemi_hal_t::instance();
    for (int phy_id = 0; phy_id < ACLSHMEMI_MAX_NPU_COUNT; ++phy_id) {
        if (!hal.get_user_id_from_phy_id(static_cast<uint32_t>(phy_id))) {
            continue;
        }
        auto bdf = format_npu_bdf(phy_id);
        if (!bdf) {
            continue;
        }
        bdf_to_npu.emplace(*bdf, phy_id);
    }
    return !bdf_to_npu.empty();
}

bool build_affinity_info(
    const std::vector<aclshmemi_xml_tag_t>& tags, affinity_info_t& info, std::string& error_message)
{
    if (tags.empty() || tags.front().name != "system") {
        error_message = "virtual topology root element must be <system>";
        return false;
    }

    std::vector<size_t> group_roots;
    bool has_pcie_group = false;
    for (size_t i = 0; i < tags.size(); ++i) {
        if (!is_group_tag(tags[i]) || tags[i].self_closing || has_group_ancestor(tags, tags[i])) {
            continue;
        }
        if (group_roots.size() >= static_cast<size_t>(MAX_AFFINITY_GROUP_COUNT)) {
            error_message = "virtual topology contains too many affinity groups";
            return false;
        }
        group_roots.push_back(i);
        has_pcie_group = has_pcie_group || tags[i].name == "pci";
    }
    if (group_roots.empty()) {
        error_message = "virtual topology contains no <pci> or <ub> affinity group";
        return false;
    }

    std::unordered_map<std::string, int> bdf_to_npu;
    if (has_pcie_group && !build_bdf_table(bdf_to_npu)) {
        error_message = "failed to obtain PCI BDF information for visible NPUs";
        return false;
    }

    for (size_t root_index : group_roots) {
        affinity_group_t group;
        bool has_npu_descriptor = false;
        bool has_nic_descriptor = false;
        const bool pcie_group = tags[root_index].name == "pci";

        for (size_t i = root_index; i < tags.size(); ++i) {
            if (!is_descendant_or_self(tags, i, root_index)) {
                continue;
            }
            const auto& tag = tags[i];
            if (pcie_group && tag.name == "pci") {
                const std::string* bus_id = tag.find_attr("busid");
                if (bus_id == nullptr || bus_id->empty()) {
                    if (i == root_index) {
                        continue;
                    }
                    error_message = "<pci> in an affinity group is missing busid";
                    return false;
                }
                std::string normalized;
                if (!normalize_bdf(*bus_id, normalized)) {
                    error_message = "invalid PCI busid '" + *bus_id + "'";
                    return false;
                }
                auto npu = bdf_to_npu.find(normalized);
                if (npu != bdf_to_npu.end()) {
                    has_npu_descriptor = true;
                    add_unique_npu(group, npu->second);
                }
            } else if (!pcie_group && tag.name == "npu") {
                const std::string* chip_phy_id = tag.find_attr("chipphyid");
                int npu_id = -1;
                if (chip_phy_id == nullptr || !parse_phy_id(*chip_phy_id, npu_id)) {
                    error_message = "<npu> in a UB affinity group has invalid chipphyid";
                    return false;
                }
                has_npu_descriptor = true;
                if (aclshmemi_hal_t::instance().get_user_id_from_phy_id(static_cast<uint32_t>(npu_id))) {
                    add_unique_npu(group, npu_id);
                }
            }
            if (tag.name == "net") {
                const std::string* name = tag.find_attr("name");
                if (name == nullptr || name->empty()) {
                    error_message = "<net> in an affinity group is missing name";
                    return false;
                }
                has_nic_descriptor = true;
                if (!add_unique_nic(info, group, *name, error_message)) {
                    return false;
                }
            }
        }

        if (!has_npu_descriptor) {
            SHM_LOG_DEBUG("Affinity group has no NPU visible to the current process");
        }
        if (!has_nic_descriptor) {
            SHM_LOG_DEBUG("Affinity group has no NIC descriptor");
        }
        info.groups.push_back(std::move(group));
    }

    if (info.nic_names.empty()) {
        error_message = "virtual topology contains no NIC";
        return false;
    }
    info.affined.assign(static_cast<size_t>(ACLSHMEMI_MAX_NPU_COUNT), std::vector<bool>(info.nic_names.size(), false));
    for (const auto& group : info.groups) {
        for (int npu_id : group.npu_ids) {
            if (npu_id < 0 || npu_id >= ACLSHMEMI_MAX_NPU_COUNT) {
                continue;
            }
            for (size_t nic_index : group.nic_indices) {
                info.affined[static_cast<size_t>(npu_id)][nic_index] = true;
            }
        }
    }
    return true;
}

std::optional<std::string> select_ip_round_robin(
    int phy_id, const affinity_info_t& info, const aclshmemi_nic_ip_resolver_t& resolver)
{
    std::vector<std::optional<std::string>> nic_ips;
    nic_ips.reserve(info.nic_names.size());
    for (const auto& name : info.nic_names) {
        auto ip = resolver.resolve(name);
        if (!ip) {
            SHM_LOG_WARN("Cannot resolve IPv4 address for topology NIC/HCA " << name);
        }
        nic_ips.push_back(std::move(ip));
    }

    std::vector<std::optional<std::string>> assignment(static_cast<size_t>(ACLSHMEMI_MAX_NPU_COUNT));
    std::vector<std::optional<size_t>> assignment_nic_index(static_cast<size_t>(ACLSHMEMI_MAX_NPU_COUNT));
    size_t current = 0;
    for (int npu_id = 0; npu_id < ACLSHMEMI_MAX_NPU_COUNT; ++npu_id) {
        for (size_t offset = 0; offset < info.nic_names.size(); ++offset) {
            const size_t nic_index = (current + offset) % info.nic_names.size();
            if (!nic_ips[nic_index] || !info.affined[static_cast<size_t>(npu_id)][nic_index]) {
                continue;
            }
            assignment[static_cast<size_t>(npu_id)] = nic_ips[nic_index];
            assignment_nic_index[static_cast<size_t>(npu_id)] = nic_index;
            current = (nic_index + 1) % info.nic_names.size();
            break;
        }
    }

    if (phy_id < 0 || phy_id >= ACLSHMEMI_MAX_NPU_COUNT || !assignment[static_cast<size_t>(phy_id)]) {
        return std::nullopt;
    }
    const size_t nic_index = *assignment_nic_index[static_cast<size_t>(phy_id)];
    SHM_LOG_INFO(
        "Resolved virtual-topology RoCE IPv4 for phy_id=" << phy_id << ", nic=" << info.nic_names[nic_index]
                                                          << ", ip=" << *assignment[static_cast<size_t>(phy_id)]);
    return assignment[static_cast<size_t>(phy_id)];
}

} // namespace

std::optional<std::string> aclshmemi_system_nic_ip_resolver_t::resolve(const std::string& name) const
{
    if (!is_safe_device_name(name)) {
        return std::nullopt;
    }
    auto ip = get_interface_ipv4(name);
    if (ip) {
        return ip;
    }
    for (const auto& interface_name : get_hca_interfaces(name)) {
        ip = get_interface_ipv4(interface_name);
        if (ip) {
            return ip;
        }
    }
    return std::nullopt;
}

std::optional<std::string> aclshmemi_get_roce_ip_from_xml(
    int phy_id, const std::string& xml_path, const aclshmemi_nic_ip_resolver_t& resolver)
{
    SHM_LOG_INFO("Resolve RoCE IPv4 from virtual topology, phy_id=" << phy_id << ", xml_path=" << xml_path);
    if (phy_id < 0 || phy_id >= ACLSHMEMI_MAX_NPU_COUNT) {
        SHM_LOG_ERROR("Invalid physical NPU id for virtual topology, phy_id=" << phy_id);
        return std::nullopt;
    }
    if (!aclshmemi_hal_t::instance().get_user_id_from_phy_id(static_cast<uint32_t>(phy_id))) {
        SHM_LOG_ERROR("Physical NPU id is not visible to the current process, phy_id=" << phy_id);
        return std::nullopt;
    }

    std::vector<aclshmemi_xml_tag_t> tags;
    std::string error_message;
    if (!aclshmemi_xml_parser_t::parse_file(xml_path, tags, error_message)) {
        SHM_LOG_ERROR("Parse virtual topology XML failed: " << error_message);
        return std::nullopt;
    }

    affinity_info_t info;
    if (!build_affinity_info(tags, info, error_message)) {
        SHM_LOG_ERROR("Build NPU-NIC affinity from virtual topology failed: " << error_message);
        return std::nullopt;
    }

    auto ip = select_ip_round_robin(phy_id, info, resolver);
    if (!ip) {
        SHM_LOG_ERROR("No usable affined RoCE IPv4 address for phy_id=" << phy_id);
    }
    return ip;
}

std::optional<std::string> aclshmemi_get_roce_ip_from_xml(int phy_id)
{
    aclshmemi_system_nic_ip_resolver_t resolver;
    return aclshmemi_get_roce_ip_from_xml(phy_id, ACLSHMEMI_VIRTUAL_TOPOLOGY_XML_PATH, resolver);
}

} // namespace topo
} // namespace shm
