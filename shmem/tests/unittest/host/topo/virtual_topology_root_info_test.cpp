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

#include <cstdio>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <unistd.h>

#include "aclshmemi_virtual_topology_root_info.h"
#include "aclshmemi_product_strategy.h"
#include "mock_hal.h"

namespace shm {
namespace topo {

namespace {

class fixed_nic_ip_resolver_t : public aclshmemi_nic_ip_resolver_t {
public:
    explicit fixed_nic_ip_resolver_t(std::string ip) : ip_(std::move(ip)) {}

    std::optional<std::string> resolve(const std::string& name) const override
    {
        return name == "hrn5_0" ? std::optional<std::string>(ip_) : std::nullopt;
    }

private:
    std::string ip_;
};

void FillUbxUeList(UEList* ue_list)
{
    *ue_list = {};
    ue_list->ueNum = 2;

    auto& mesh = ue_list->ueList[0];
    mesh.eidNum = 2;
    mesh.eidList[0].eid.raw[5] = 0x41;
    mesh.eidList[0].eid.raw[6] = 3;
    mesh.eidList[1].eid.raw[5] = 0x42;
    mesh.eidList[1].eid.raw[6] = 3;

    auto& clos = ue_list->ueList[1];
    clos.eidNum = 3;
    clos.eidList[0].eid.raw[5] = 0x44;
    clos.eidList[0].eid.raw[6] = 2;
    clos.eidList[1].eid.raw[5] = 0x45;
    clos.eidList[1].eid.raw[6] = 2;
    clos.eidList[2].eid.raw[5] = 0x7F;
    clos.eidList[2].eid.raw[6] = 2;
}

} // namespace

class VirtualTopologyRootInfoTest : public testing::Test {
protected:
    void SetUp() override
    {
        testing::Mock::AllowLeak(&MockHal::instance());
        char path[] = "/tmp/aclshmem_virtual_topology_rootinfo_XXXXXX";
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

    void WriteValidXml()
    {
        WriteXml("<system><ub><nic><net name=\"hrn5_0\"/></nic><npu chipphyid=\"0\"/></ub></system>");
    }

    void ExpectVisibleNpu()
    {
        EXPECT_CALL(MockHal::instance(), get_user_id_from_phy_id(testing::_))
            .Times(testing::AnyNumber())
            .WillRepeatedly(testing::Return(std::optional<uint32_t>(0)));
    }

    void ExpectCardBaseGeneration()
    {
        EXPECT_CALL(MockHal::instance(), get_mainboard_id(0))
            .WillOnce(testing::Return(std::optional<uint32_t>(ACLSHMEMI_MAIN_BOARD_ID_CARD_NOMESH)));
        EXPECT_CALL(MockHal::instance(), get_driver_install_path()).WillOnce(testing::Return("/usr/local/Ascend"));
        EXPECT_CALL(MockHal::instance(), get_ue_list(0, testing::_)).WillOnce(testing::Invoke([](int, UEList* ue_list) {
            *ue_list = {};
            return 0;
        }));
        ExpectVisibleNpu();
    }

    std::string xml_path_;
};

TEST_F(VirtualTopologyRootInfoTest, CardUsesOnlyXmlRoceLayer)
{
    ExpectCardBaseGeneration();
    WriteValidXml();
    fixed_nic_ip_resolver_t resolver("10.20.30.40");

    auto root_info = aclshmemi_generate_virtual_topology_root_info(0, xml_path_, resolver);
    ASSERT_TRUE(root_info.has_value());
    ASSERT_EQ(root_info->ranks.size(), 1U);
    ASSERT_EQ(root_info->ranks[0].level_list.size(), 1U);
    const auto& layer = root_info->ranks[0].level_list[0];
    EXPECT_EQ(layer.level, 3);
    EXPECT_EQ(layer.net_type, "CLOS");
    ASSERT_EQ(layer.addr_list.size(), 1U);
    EXPECT_EQ(layer.addr_list[0].addr_type, "IPV4");
    EXPECT_EQ(layer.addr_list[0].addr, "10.20.30.40");
    EXPECT_EQ(layer.addr_list[0].plane_id, "plane0");
    ASSERT_EQ(layer.addr_list[0].ports.size(), 1U);
    EXPECT_EQ(layer.addr_list[0].ports[0], "d2h");
}

TEST_F(VirtualTopologyRootInfoTest, XmlFailureFailsWholeGeneration)
{
    ExpectCardBaseGeneration();
    WriteXml("<system><ub><net name=\"hrn5_0\"></system>");
    fixed_nic_ip_resolver_t resolver("10.20.30.40");

    EXPECT_FALSE(aclshmemi_generate_virtual_topology_root_info(0, xml_path_, resolver).has_value());
}

TEST_F(VirtualTopologyRootInfoTest, ServerKeepsBaseLayersAndAppendsXmlRoceLayer)
{
    EXPECT_CALL(MockHal::instance(), get_mainboard_id(0))
        .WillOnce(testing::Return(std::optional<uint32_t>(ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH)));
    EXPECT_CALL(MockHal::instance(), get_driver_install_path()).WillOnce(testing::Return("/usr/local/Ascend"));
    EXPECT_CALL(MockHal::instance(), get_ue_list(0, testing::_)).WillOnce(testing::Invoke([](int, UEList* ue_list) {
        *ue_list = {};
        return 0;
    }));
    EXPECT_CALL(MockHal::instance(), get_spod_info(0, testing::_))
        .WillOnce(testing::Invoke([](int, dcmi_spod_info* spod_info) {
            *spod_info = {};
            return 0;
        }));
    ExpectVisibleNpu();
    WriteValidXml();
    fixed_nic_ip_resolver_t resolver("10.20.30.40");

    auto root_info = aclshmemi_generate_virtual_topology_root_info(0, xml_path_, resolver);
    ASSERT_TRUE(root_info.has_value());
    ASSERT_EQ(root_info->ranks[0].level_list.size(), 3U);
    EXPECT_EQ(root_info->ranks[0].level_list[0].net_type, "MESH");
    EXPECT_EQ(root_info->ranks[0].level_list[1].net_type, "CLOS");
    EXPECT_EQ(root_info->ranks[0].level_list[2].addr_list[0].addr, "10.20.30.40");
}

TEST_F(VirtualTopologyRootInfoTest, PodKeepsBaseLayersAndAppendsXmlRoceLayer)
{
    EXPECT_CALL(MockHal::instance(), get_mainboard_id(0))
        .WillOnce(testing::Return(std::optional<uint32_t>(ACLSHMEMI_MAIN_BOARD_ID_POD)));
    EXPECT_CALL(MockHal::instance(), get_driver_install_path()).WillOnce(testing::Return("/usr/local/Ascend"));
    EXPECT_CALL(MockHal::instance(), get_ue_list(0, testing::_)).WillOnce(testing::Invoke([](int, UEList* ue_list) {
        *ue_list = {};
        return 0;
    }));
    EXPECT_CALL(MockHal::instance(), get_spod_info(0, testing::_))
        .WillOnce(testing::Invoke([](int, dcmi_spod_info* spod_info) {
            *spod_info = {};
            return 0;
        }));
    ExpectVisibleNpu();
    WriteValidXml();
    fixed_nic_ip_resolver_t resolver("10.20.30.40");

    auto root_info = aclshmemi_generate_virtual_topology_root_info(0, xml_path_, resolver);
    ASSERT_TRUE(root_info.has_value());
    ASSERT_EQ(root_info->ranks[0].level_list.size(), 3U);
    EXPECT_EQ(root_info->ranks[0].level_list[0].net_type, "MESH");
    EXPECT_EQ(root_info->ranks[0].level_list[1].net_type, "CLOS");
    EXPECT_EQ(root_info->ranks[0].level_list[2].addr_list[0].addr, "10.20.30.40");
}

TEST_F(VirtualTopologyRootInfoTest, UbxKeepsBaseLayerAndAppendsXmlRoceLayer)
{
    EXPECT_CALL(MockHal::instance(), get_mainboard_id(0))
        .WillOnce(testing::Return(std::optional<uint32_t>(ACLSHMEMI_MAIN_BOARD_ID_SERVER_UBX)));
    EXPECT_CALL(MockHal::instance(), get_driver_install_path()).WillOnce(testing::Return("/usr/local/Ascend"));
    EXPECT_CALL(MockHal::instance(), get_server_id()).WillOnce(testing::Return("server-id"));
    EXPECT_CALL(MockHal::instance(), get_ue_list(0, testing::_)).WillOnce(testing::Invoke([](int, UEList* ue_list) {
        FillUbxUeList(ue_list);
        return 0;
    }));
    ExpectVisibleNpu();
    WriteValidXml();
    fixed_nic_ip_resolver_t resolver("10.20.30.40");

    auto root_info = aclshmemi_generate_virtual_topology_root_info(0, xml_path_, resolver);
    ASSERT_TRUE(root_info.has_value());
    ASSERT_EQ(root_info->ranks[0].level_list.size(), 2U);
    EXPECT_EQ(root_info->ranks[0].level_list[0].net_type, "TOPO_FILE_DESC");
    EXPECT_EQ(root_info->ranks[0].level_list[0].addr_list.size(), 5U);
    EXPECT_EQ(root_info->ranks[0].level_list[1].addr_list[0].addr, "10.20.30.40");
}

TEST_F(VirtualTopologyRootInfoTest, UbxWithoutBaseEidFailsBeforeXmlLayer)
{
    EXPECT_CALL(MockHal::instance(), get_mainboard_id(0))
        .WillOnce(testing::Return(std::optional<uint32_t>(ACLSHMEMI_MAIN_BOARD_ID_SERVER_UBX)));
    EXPECT_CALL(MockHal::instance(), get_driver_install_path()).WillOnce(testing::Return("/usr/local/Ascend"));
    EXPECT_CALL(MockHal::instance(), get_server_id()).WillOnce(testing::Return("server-id"));
    EXPECT_CALL(MockHal::instance(), get_ue_list(0, testing::_)).WillOnce(testing::Invoke([](int, UEList* ue_list) {
        *ue_list = {};
        return 0;
    }));
    fixed_nic_ip_resolver_t resolver("10.20.30.40");

    EXPECT_FALSE(aclshmemi_generate_virtual_topology_root_info(0, xml_path_, resolver).has_value());
}

} // namespace topo
} // namespace shm
