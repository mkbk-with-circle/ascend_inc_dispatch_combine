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
#include <map>
#include <string>
#include <securec.h>
#include "aclshmemi_eid_parser.h"
#include "dl_dcmi_types.h"

namespace shm {
namespace topo {

class EidParserTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

static void hex32_to_bin16(const char* hex_str, uint8_t* bin_out)
{
    for (int i = 0; i < 16; i++) {
        char c1 = hex_str[2 * i];
        char c2 = hex_str[2 * i + 1];
        int h1 = (c1 >= '0' && c1 <= '9') ? c1 - '0' : (c1 >= 'a' && c1 <= 'f') ? 10 + c1 - 'a' : 10 + c1 - 'A';
        int h2 = (c2 >= '0' && c2 <= '9') ? c2 - '0' : (c2 >= 'a' && c2 <= 'f') ? 10 + c2 - 'a' : 10 + c2 - 'A';
        bin_out[i] = (h1 << 4) | h2;
    }
}

static void set_ub_eid(dcmi_urma_eid_t& eid, uint8_t fe, uint8_t die, uint8_t port)
{
    int ret = memset_s(eid.raw, DCMI_URMA_EID_SIZE, 0, DCMI_URMA_EID_SIZE);
    EXPECT_EQ(ret, EOK);
    eid.raw[5] = static_cast<uint8_t>((die << 6) | (port & 0x3f));
    eid.raw[6] = fe;
}

TEST_F(EidParserTest, GetPortIdFromHexString)
{
    std::map<std::string, int> port_map = {
        {"000000000000002000100000dfdf0020", 4}, {"000000000000002000100000dfdf0028", 5},
        {"000000000000002000100000dfdf0030", 6}, {"000000000000002000100000dfdf0021", 4},
        {"000000000000002000100000dfdf0029", 5}, {"000000000000002000100000dfdf0031", 6},
        {"000000000000002000100000dfdf0022", 4}, {"000000000000002000100000dfdf002a", 5},
        {"000000000000002000100000dfdf0032", 6}, {"000000000000002000100000dfdf0023", 4},
        {"000000000000002000100000dfdf002b", 5}, {"000000000000002000100000dfdf0033", 6},
    };

    for (auto& it : port_map) {
        auto port = aclshmemi_eid_parser_t::get_port_id(it.first);
        EXPECT_TRUE(port.has_value());
        EXPECT_EQ(port.value(), it.second);
    }
}

TEST_F(EidParserTest, GetPortIdFromBinary)
{
    dcmi_urma_eid_t eid;
    hex32_to_bin16("000000000000002000100000dfdf0020", eid.raw);
    int port = aclshmemi_eid_parser_t::get_port_id(eid);
    EXPECT_EQ(port, 4);

    hex32_to_bin16("000000000000002000100000dfdf0028", eid.raw);
    port = aclshmemi_eid_parser_t::get_port_id(eid);
    EXPECT_EQ(port, 5);

    hex32_to_bin16("000000000000002000100000dfdf0030", eid.raw);
    port = aclshmemi_eid_parser_t::get_port_id(eid);
    EXPECT_EQ(port, 6);
}

TEST_F(EidParserTest, GetFeIdFromBinary)
{
    dcmi_urma_eid_t eid;
    hex32_to_bin16("000000000000002000100000dfdf0020", eid.raw);
    int fe_id = aclshmemi_eid_parser_t::get_fe_id(eid);
    EXPECT_EQ(fe_id, 1);

    hex32_to_bin16("000000000000004000100000dfdf0020", eid.raw);
    fe_id = aclshmemi_eid_parser_t::get_fe_id(eid);
    EXPECT_EQ(fe_id, 2);

    hex32_to_bin16("000000000000006000100000dfdf0020", eid.raw);
    fe_id = aclshmemi_eid_parser_t::get_fe_id(eid);
    EXPECT_EQ(fe_id, 3);
}

TEST_F(EidParserTest, GetDieIdFromBinary)
{
    dcmi_urma_eid_t eid;
    hex32_to_bin16("000000000000002000100000dfdf0020", eid.raw);
    int die_id = aclshmemi_eid_parser_t::get_die_id(eid);
    EXPECT_EQ(die_id, 0);

    hex32_to_bin16("000000000000002000100000dfdf0024", eid.raw);
    die_id = aclshmemi_eid_parser_t::get_die_id(eid);
    EXPECT_EQ(die_id, 1);
}

TEST_F(EidParserTest, GetMaxFeId)
{
    dcmi_urma_eid_info_t eid_list[4];
    hex32_to_bin16("000000000000002000100000dfdf0020", eid_list[0].eid.raw);
    hex32_to_bin16("000000000000004000100000dfdf0028", eid_list[1].eid.raw);
    hex32_to_bin16("000000000000006000100000dfdf0030", eid_list[2].eid.raw);
    hex32_to_bin16("00000000000000a000100000dfdf0038", eid_list[3].eid.raw);

    auto max_fe = aclshmemi_eid_parser_t::get_max_fe_id(eid_list, 4);
    EXPECT_TRUE(max_fe.has_value());
    EXPECT_EQ(max_fe.value(), 5);
}

TEST_F(EidParserTest, GetUBEntityId)
{
    UBEntity ue;
    ue.eidNum = 1;
    set_ub_eid(ue.eidList[0].eid, 1, 0, 4);

    int entity_id = aclshmemi_eid_parser_t::get_ub_entity_id(ue);
    EXPECT_EQ(entity_id, 1);

    ue.eidNum = 2;
    set_ub_eid(ue.eidList[1].eid, 2, 0, 5);
    entity_id = aclshmemi_eid_parser_t::get_ub_entity_id(ue);
    EXPECT_EQ(entity_id, 1);
}

TEST_F(EidParserTest, GetServerPortGroupIdx)
{
    UBEntity ue;
    ue.eidNum = 4;
    set_ub_eid(ue.eidList[0].eid, 2, 0, 4);
    set_ub_eid(ue.eidList[1].eid, 2, 0, 5);
    set_ub_eid(ue.eidList[2].eid, 2, 0, 0x3f);
    set_ub_eid(ue.eidList[3].eid, 2, 0, 6);

    int idx = aclshmemi_eid_parser_t::get_server_port_group_idx(ue);
    EXPECT_EQ(idx, 2);
}

TEST_F(EidParserTest, GetMaxEntityId)
{
    UEList ue_list;
    ue_list.ueNum = 3;

    ue_list.ueList[0].eidNum = 1;
    set_ub_eid(ue_list.ueList[0].eidList[0].eid, 1, 0, 4);

    ue_list.ueList[1].eidNum = 1;
    set_ub_eid(ue_list.ueList[1].eidList[0].eid, 3, 0, 5);

    ue_list.ueList[2].eidNum = 1;
    set_ub_eid(ue_list.ueList[2].eidList[0].eid, 5, 1, 6);

    int max_id = aclshmemi_eid_parser_t::get_max_entity_id(ue_list);
    EXPECT_EQ(max_id, 5);

    max_id = aclshmemi_eid_parser_t::get_max_entity_id(ue_list, 0);
    EXPECT_EQ(max_id, 3);
}

TEST_F(EidParserTest, GetPodDieId)
{
    dcmi_urma_eid_t eid;
    set_ub_eid(eid, 2, 0, 4);
    int pod_die = aclshmemi_eid_parser_t::get_pod_die_id(eid);
    EXPECT_EQ(pod_die, 0);

    set_ub_eid(eid, 2, 1, 4);
    pod_die = aclshmemi_eid_parser_t::get_pod_die_id(eid);
    EXPECT_EQ(pod_die, 1);
}

TEST_F(EidParserTest, GetServerDieId)
{
    dcmi_urma_eid_t eid;
    set_ub_eid(eid, 2, 0, 4);
    int server_die = aclshmemi_eid_parser_t::get_server_die_id(eid);
    EXPECT_EQ(server_die, 0);

    set_ub_eid(eid, 2, 1, 4);
    server_die = aclshmemi_eid_parser_t::get_server_die_id(eid);
    EXPECT_EQ(server_die, 1);
}

TEST_F(EidParserTest, GetUBPortId)
{
    dcmi_urma_eid_t eid;
    set_ub_eid(eid, 2, 1, 6);
    EXPECT_EQ(aclshmemi_eid_parser_t::get_ub_port_id(eid), 6);
    EXPECT_FALSE(aclshmemi_eid_parser_t::is_ub_port_group(eid));

    set_ub_eid(eid, 2, 1, 0x3f);
    EXPECT_TRUE(aclshmemi_eid_parser_t::is_ub_port_group(eid));
}

TEST_F(EidParserTest, GetLowBitPort)
{
    dcmi_urma_eid_t eid;
    hex32_to_bin16("000000000000002000100000dfdf0020", eid.raw);
    int low_port = aclshmemi_eid_parser_t::get_low_bit_port(eid);
    EXPECT_EQ(low_port, 0x20);

    hex32_to_bin16("000000000000002000100000dfdf0028", eid.raw);
    low_port = aclshmemi_eid_parser_t::get_low_bit_port(eid);
    EXPECT_EQ(low_port, 0x28);
}

TEST_F(EidParserTest, GetUbFields)
{
    dcmi_urma_eid_t eid = {};
    eid.raw[5] = 0x45;
    eid.raw[6] = 0x82;

    EXPECT_EQ(aclshmemi_eid_parser_t::get_ub_die_id(eid), 1);
    EXPECT_EQ(aclshmemi_eid_parser_t::get_ub_port_id(eid), 5);
    EXPECT_EQ(aclshmemi_eid_parser_t::get_ub_fe_id(eid), 2);

    UBEntity entity = {};
    entity.eidNum = 2;
    entity.eidList[0].eid = eid;
    entity.eidList[1].eid.raw[5] = 0x7F;
    EXPECT_EQ(aclshmemi_eid_parser_t::get_server_port_group_idx(entity), 1);
}

TEST_F(EidParserTest, InvalidHexString)
{
    auto port = aclshmemi_eid_parser_t::get_port_id("invalid");
    EXPECT_FALSE(port.has_value());

    auto fe = aclshmemi_eid_parser_t::get_fe_id("short");
    EXPECT_FALSE(fe.has_value());

    auto die = aclshmemi_eid_parser_t::get_die_id("short");
    EXPECT_FALSE(die.has_value());
}

} // namespace topo
} // namespace shm
