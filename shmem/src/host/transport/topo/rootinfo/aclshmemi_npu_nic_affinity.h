/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSHMEMI_NPU_NIC_AFFINITY_HPP
#define ACLSHMEMI_NPU_NIC_AFFINITY_HPP

#include <optional>
#include <string>

namespace shm {
namespace topo {

constexpr const char* ACLSHMEMI_VIRTUAL_TOPOLOGY_XML_PATH = "/var/run/ascend-topologyd/virtualTopology.xml";

class aclshmemi_nic_ip_resolver_t {
public:
    virtual ~aclshmemi_nic_ip_resolver_t() = default;
    virtual std::optional<std::string> resolve(const std::string& name) const = 0;
};

class aclshmemi_system_nic_ip_resolver_t : public aclshmemi_nic_ip_resolver_t {
public:
    std::optional<std::string> resolve(const std::string& name) const override;
};

std::optional<std::string> aclshmemi_get_roce_ip_from_xml(
    int phy_id, const std::string& xml_path, const aclshmemi_nic_ip_resolver_t& resolver);
std::optional<std::string> aclshmemi_get_roce_ip_from_xml(int phy_id);

} // namespace topo
} // namespace shm

#endif // ACLSHMEMI_NPU_NIC_AFFINITY_HPP
