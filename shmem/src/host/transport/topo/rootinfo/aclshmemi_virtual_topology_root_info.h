/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSHMEMI_VIRTUAL_TOPOLOGY_ROOT_INFO_HPP
#define ACLSHMEMI_VIRTUAL_TOPOLOGY_ROOT_INFO_HPP

#include <optional>
#include <string>

#include "aclshmemi_npu_nic_affinity.h"
#include "aclshmemi_types.h"

namespace shm {
namespace topo {

std::optional<aclshmemi_root_info_t> aclshmemi_generate_virtual_topology_root_info(
    int phy_id, const std::string& xml_path, const aclshmemi_nic_ip_resolver_t& resolver);
std::optional<std::string> aclshmemi_generate_virtual_topology_root_info_json(int phy_id);

} // namespace topo
} // namespace shm

#endif // ACLSHMEMI_VIRTUAL_TOPOLOGY_ROOT_INFO_HPP
