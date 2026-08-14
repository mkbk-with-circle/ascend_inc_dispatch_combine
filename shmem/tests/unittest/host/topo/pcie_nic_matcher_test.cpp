/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "pcie_nic_matcher.h"

namespace shm {
namespace topo {

class PcieNicMatcherTest : public testing::Test {};

TEST_F(PcieNicMatcherTest, GetPathBasenameHandlesConstPath)
{
    EXPECT_STREQ(get_path_basename("/sys/class/net/enp1s0f0"), "enp1s0f0");
    EXPECT_STREQ(get_path_basename("enp1s0f0"), "enp1s0f0");
    EXPECT_STREQ(get_path_basename(nullptr), "");
}

TEST_F(PcieNicMatcherTest, IsPixTopoSamePrefixReturnsTrue)
{
    const char* path_a = "/sys/devices/pci0000:80/0000:80:01.0/0000:81:00.0";
    const char* path_b = "/sys/devices/pci0000:80/0000:80:01.0/0000:81:00.1";
    EXPECT_TRUE(is_pix_topo(path_a, path_b));
}

TEST_F(PcieNicMatcherTest, IsPixTopoDifferentPrefixReturnsFalse)
{
    const char* path_a = "/sys/devices/pci0000:80/0000:80:01.0/0000:81:00.0";
    const char* path_b = "/sys/devices/pci0000:80/0000:80:02.0/0000:82:00.0";
    // Different switches and lspci not available in unit test → false
    EXPECT_FALSE(is_pix_topo(path_a, path_b));
}

TEST_F(PcieNicMatcherTest, IsPixTopoPathTooShortReturnsFalse)
{
    const char* short_path = "/sys/123456789012";
    const char* normal_path = "/sys/devices/pci0000:80/0000:80:01.0/0000:81:00.0";
    EXPECT_FALSE(is_pix_topo(short_path, normal_path));
}

TEST_F(PcieNicMatcherTest, IsPixTopoNullptrReturnsFalse)
{
    const char* valid = "/sys/devices/pci0000:80/0000:80:01.0/0000:81:00.0";
    EXPECT_FALSE(is_pix_topo(nullptr, valid));
    EXPECT_FALSE(is_pix_topo(valid, nullptr));
    EXPECT_FALSE(is_pix_topo(nullptr, nullptr));
}

} // namespace topo
} // namespace shm
