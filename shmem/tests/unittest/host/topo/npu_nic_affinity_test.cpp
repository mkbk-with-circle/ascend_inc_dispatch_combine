/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <unistd.h>
#include <vector>

#include "aclshmemi_npu_nic_affinity.h"
#include "mock_hal.h"

namespace shm {
namespace topo {

namespace {

class fake_nic_ip_resolver_t : public aclshmemi_nic_ip_resolver_t {
public:
    explicit fake_nic_ip_resolver_t(std::unordered_map<std::string, std::string> addresses)
        : addresses_(std::move(addresses))
    {}

    std::optional<std::string> resolve(const std::string& name) const override
    {
        auto it = addresses_.find(name);
        return it == addresses_.end() ? std::nullopt : std::optional<std::string>(it->second);
    }

private:
    std::unordered_map<std::string, std::string> addresses_;
};

} // namespace

class NpuNicAffinityTest : public testing::Test {
protected:
    void SetUp() override
    {
        testing::Mock::AllowLeak(&MockHal::instance());
        char path[] = "/tmp/aclshmem_affinity_XXXXXX";
        const int fd = mkstemp(path);
        ASSERT_GE(fd, 0);
        close(fd);
        xml_path_ = path;
    }

    void TearDown() override
    {
        if (!xml_path_.empty()) {
            std::remove(xml_path_.c_str());
        }
        testing::Mock::VerifyAndClearExpectations(&MockHal::instance());
    }

    void WriteXml(const std::string& xml)
    {
        std::ofstream file(xml_path_, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(file.is_open());
        file << xml;
        ASSERT_TRUE(file.good());
    }

    void ExpectVisibleNpus(const std::vector<uint32_t>& phy_ids)
    {
        EXPECT_CALL(MockHal::instance(), get_user_id_from_phy_id(testing::_))
            .Times(testing::AnyNumber())
            .WillRepeatedly(testing::Invoke([phy_ids](uint32_t phy_id) -> std::optional<uint32_t> {
                auto it = std::find(phy_ids.begin(), phy_ids.end(), phy_id);
                if (it == phy_ids.end()) {
                    return std::nullopt;
                }
                return static_cast<uint32_t>(std::distance(phy_ids.begin(), it));
            }));
    }

    void ExpectVisibleNpus(int count)
    {
        std::vector<uint32_t> phy_ids;
        phy_ids.reserve(static_cast<size_t>(count));
        for (int phy_id = 0; phy_id < count; ++phy_id) {
            phy_ids.push_back(static_cast<uint32_t>(phy_id));
        }
        ExpectVisibleNpus(phy_ids);
    }

    std::string xml_path_;
};

TEST_F(NpuNicAffinityTest, SelectsIpFromUbAffinityGroup)
{
    ExpectVisibleNpus(4);
    WriteXml("<system version=\"1.0\"><cpu numaid=\"0\">"
             "<ub><nic><net name=\"hrn5_0\"/></nic><npu chipphyid=\"0\"/><npu chipphyid=\"1\"/></ub>"
             "<ub><nic><net name=\"hrn5_1\"/></nic><npu chipphyid=\"2\"/><npu chipphyid=\"3\"/></ub>"
             "</cpu></system>");
    fake_nic_ip_resolver_t resolver(
        std::unordered_map<std::string, std::string>{{"hrn5_0", "10.0.0.1"}, {"hrn5_1", "10.0.0.2"}});

    auto ip = aclshmemi_get_roce_ip_from_xml(2, xml_path_, resolver);
    ASSERT_TRUE(ip.has_value());
    EXPECT_EQ(*ip, "10.0.0.2");
}

TEST_F(NpuNicAffinityTest, SkipsUnresolvedNicWithinSameGroup)
{
    ExpectVisibleNpus(1);
    WriteXml("<system><ub><nic><net name=\"hrn5_bad\"/><net name=\"hrn5_good\"/></nic>"
             "<npu chipphyid=\"0\"/></ub></system>");
    fake_nic_ip_resolver_t resolver(std::unordered_map<std::string, std::string>{{"hrn5_good", "192.168.10.8"}});

    auto ip = aclshmemi_get_roce_ip_from_xml(0, xml_path_, resolver);
    ASSERT_TRUE(ip.has_value());
    EXPECT_EQ(*ip, "192.168.10.8");
}

TEST_F(NpuNicAffinityTest, MatchesPcieBusId)
{
    ExpectVisibleNpus(2);
    EXPECT_CALL(MockHal::instance(), get_device_pcie_info(testing::_, testing::_))
        .Times(2)
        .WillRepeatedly(testing::Invoke([](int phy_id, struct dcmi_pcie_info_all* info) {
            *info = {};
            info->domain = 0;
            info->bdf_busid = static_cast<unsigned int>(phy_id + 3);
            return 0;
        }));
    WriteXml("<system><cpu><pci busid=\"0000:01:00.0\"><nic><net name=\"eth0\"/></nic>"
             "<pci busid=\"0000:03:00.0\"/><pci busid=\"0000:04:00.0\"/>"
             "</pci></cpu></system>");
    fake_nic_ip_resolver_t resolver(std::unordered_map<std::string, std::string>{{"eth0", "172.16.0.4"}});

    auto ip = aclshmemi_get_roce_ip_from_xml(1, xml_path_, resolver);
    ASSERT_TRUE(ip.has_value());
    EXPECT_EQ(*ip, "172.16.0.4");
}

TEST_F(NpuNicAffinityTest, SupportsPcieGroupContainerWithoutBusId)
{
    ExpectVisibleNpus(2);
    EXPECT_CALL(MockHal::instance(), get_device_pcie_info(testing::_, testing::_))
        .Times(2)
        .WillRepeatedly(testing::Invoke([](int phy_id, struct dcmi_pcie_info_all* info) {
            *info = {};
            info->domain = 0;
            info->bdf_busid = static_cast<unsigned int>(phy_id + 3);
            return 0;
        }));
    WriteXml("<system><cpu><pci>"
             "<pci busid=\"0000:03:00.0\"/><pci busid=\"0000:04:00.0\"/>"
             "<pci busid=\"0000:05:00.0\"><nic><net name=\"ens100f0\"/></nic></pci>"
             "</pci></cpu></system>");
    fake_nic_ip_resolver_t resolver(std::unordered_map<std::string, std::string>{{"ens100f0", "172.16.0.5"}});

    auto ip = aclshmemi_get_roce_ip_from_xml(1, xml_path_, resolver);
    ASSERT_TRUE(ip.has_value());
    EXPECT_EQ(*ip, "172.16.0.5");
}

TEST_F(NpuNicAffinityTest, SupportsNonZeroVisiblePhyIdInUbGroup)
{
    ExpectVisibleNpus(std::vector<uint32_t>{4});
    WriteXml("<system><ub><nic><net name=\"hrn5_4\"/></nic>"
             "<npu chipphyid=\"4\"/></ub></system>");
    fake_nic_ip_resolver_t resolver(std::unordered_map<std::string, std::string>{{"hrn5_4", "10.0.0.4"}});

    auto ip = aclshmemi_get_roce_ip_from_xml(4, xml_path_, resolver);
    ASSERT_TRUE(ip.has_value());
    EXPECT_EQ(*ip, "10.0.0.4");
}

TEST_F(NpuNicAffinityTest, SupportsNonContiguousVisiblePhyIdsInPcieGroup)
{
    ExpectVisibleNpus(std::vector<uint32_t>{1, 4});
    EXPECT_CALL(MockHal::instance(), get_device_pcie_info(testing::_, testing::_))
        .Times(2)
        .WillRepeatedly(testing::Invoke([](int phy_id, struct dcmi_pcie_info_all* info) {
            *info = {};
            info->domain = 0;
            info->bdf_busid = static_cast<unsigned int>(phy_id + 3);
            return 0;
        }));
    WriteXml("<system><cpu><pci>"
             "<pci busid=\"0000:04:00.0\"/><pci busid=\"0000:07:00.0\"/>"
             "<pci busid=\"0000:05:00.0\"><nic><net name=\"eth0\"/><net name=\"eth1\"/></nic></pci>"
             "</pci></cpu></system>");
    fake_nic_ip_resolver_t resolver(
        std::unordered_map<std::string, std::string>{{"eth0", "172.16.0.1"}, {"eth1", "172.16.0.4"}});

    auto ip = aclshmemi_get_roce_ip_from_xml(4, xml_path_, resolver);
    ASSERT_TRUE(ip.has_value());
    EXPECT_EQ(*ip, "172.16.0.4");
}

TEST_F(NpuNicAffinityTest, FailsWhenCurrentNpuHasNoAffinity)
{
    ExpectVisibleNpus(2);
    WriteXml("<system><ub><nic><net name=\"hrn5_0\"/></nic><npu chipphyid=\"0\"/></ub></system>");
    fake_nic_ip_resolver_t resolver(std::unordered_map<std::string, std::string>{{"hrn5_0", "10.0.0.1"}});

    EXPECT_FALSE(aclshmemi_get_roce_ip_from_xml(1, xml_path_, resolver).has_value());
}

TEST_F(NpuNicAffinityTest, FailsOnMalformedTopologyContent)
{
    ExpectVisibleNpus(1);
    WriteXml("<system><ub><net/><npu chipphyid=\"0\"/></ub></system>");
    fake_nic_ip_resolver_t resolver(std::unordered_map<std::string, std::string>{});

    EXPECT_FALSE(aclshmemi_get_roce_ip_from_xml(0, xml_path_, resolver).has_value());
}

TEST_F(NpuNicAffinityTest, RejectsInvalidUbChipPhyId)
{
    ExpectVisibleNpus(1);
    fake_nic_ip_resolver_t resolver(std::unordered_map<std::string, std::string>{{"hrn5_0", "10.0.0.1"}});
    const std::vector<std::string> invalid_values = {"", "-1", "+1", " 1", "1x", "64", "999999999999999999999"};

    for (const auto& value : invalid_values) {
        SCOPED_TRACE(value);
        WriteXml("<system><ub><nic><net name=\"hrn5_0\"/></nic><npu chipphyid=\"" + value + "\"/></ub></system>");
        EXPECT_FALSE(aclshmemi_get_roce_ip_from_xml(0, xml_path_, resolver).has_value());
    }
}

} // namespace topo
} // namespace shm
