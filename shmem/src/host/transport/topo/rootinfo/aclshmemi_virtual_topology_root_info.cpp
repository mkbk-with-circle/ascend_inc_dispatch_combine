/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclshmemi_virtual_topology_root_info.h"

#include <arpa/inet.h>
#include <iomanip>

#include "aclshmemi_hal.h"
#include "aclshmemi_product_strategy.h"
#include "utils/shmemi_logger.h"

namespace shm {
namespace topo {

namespace {

std::optional<aclshmemi_net_layer_t> build_xml_roce_layer(const std::string& ip)
{
    struct in_addr address;
    if (inet_pton(AF_INET, ip.c_str(), &address) != 1) {
        SHM_LOG_ERROR("Invalid RoCE IPv4 address resolved from virtual topology: " << ip);
        return std::nullopt;
    }

    aclshmemi_net_layer_t layer;
    aclshmemi_net_layer_init(layer, 3, "cluster");
    aclshmemi_net_layer_set_net_type(layer, "CLOS");

    aclshmemi_addr_t addr;
    aclshmemi_addr_set_ip(addr, ip);
    aclshmemi_addr_set_plane_id(addr, "plane0");
    aclshmemi_addr_add_port(addr, "d2h");
    aclshmemi_net_layer_add_addr(layer, addr);
    return layer;
}

} // namespace

std::optional<aclshmemi_root_info_t> aclshmemi_generate_virtual_topology_root_info(
    int phy_id, const std::string& xml_path, const aclshmemi_nic_ip_resolver_t& resolver)
{
    auto& hal = aclshmemi_hal_t::instance();
    auto mainboard_id = hal.get_mainboard_id(phy_id);
    if (!mainboard_id) {
        SHM_LOG_ERROR("Generate virtual-topology rootinfo failed: cannot get mainboard id for phy_id=" << phy_id);
        return std::nullopt;
    }
    SHM_LOG_INFO(
        "Generate virtual-topology rootinfo fallback, phy_id=" << phy_id << ", mainboard_id=0x" << std::hex
                                                               << *mainboard_id << std::dec);

    auto strategy = aclshmemi_product_strategy_t::create(*mainboard_id);
    if (!strategy) {
        SHM_LOG_ERROR(
            "Generate virtual-topology rootinfo failed: unsupported mainboard id 0x" << std::hex << *mainboard_id);
        return std::nullopt;
    }
    auto root_info = strategy->get_root_info_without_roce(phy_id, *mainboard_id);
    if (!root_info) {
        SHM_LOG_ERROR(
            "Generate virtual-topology rootinfo failed: base rootinfo generation failed for phy_id=" << phy_id);
        return std::nullopt;
    }

    auto ip = aclshmemi_get_roce_ip_from_xml(phy_id, xml_path, resolver);
    if (!ip) {
        SHM_LOG_ERROR(
            "Generate virtual-topology rootinfo failed: virtual topology has no usable IP for phy_id=" << phy_id);
        return std::nullopt;
    }
    auto roce_layer = build_xml_roce_layer(*ip);
    if (!roce_layer) {
        return std::nullopt;
    }

    for (auto& rank : root_info->ranks) {
        if (rank.device_id != phy_id) {
            continue;
        }
        aclshmemi_rank_add_net_layer(rank, *roce_layer);
        SHM_LOG_INFO("Generate virtual-topology rootinfo fallback success, phy_id=" << phy_id << ", RoCE IPv4=" << *ip);
        return root_info;
    }

    SHM_LOG_ERROR("Generate virtual-topology rootinfo failed: generated rank is missing for phy_id=" << phy_id);
    return std::nullopt;
}

std::optional<std::string> aclshmemi_generate_virtual_topology_root_info_json(int phy_id)
{
    aclshmemi_system_nic_ip_resolver_t resolver;
    auto root_info =
        aclshmemi_generate_virtual_topology_root_info(phy_id, ACLSHMEMI_VIRTUAL_TOPOLOGY_XML_PATH, resolver);
    if (!root_info) {
        return std::nullopt;
    }
    return aclshmemi_root_info_to_string(*root_info);
}

} // namespace topo
} // namespace shm
