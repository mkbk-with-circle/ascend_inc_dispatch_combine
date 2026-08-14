/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstring>
#include <memory>
#include "aclshmemi_product_strategy.h"
#include "mock_hal.h"

namespace shm {
namespace topo {

class ProductStrategyTest : public testing::Test {
protected:
    void SetUp() override { testing::Mock::AllowLeak(&MockHal::instance()); }
    void TearDown() override { testing::Mock::VerifyAndClearExpectations(&MockHal::instance()); }
};

TEST_F(ProductStrategyTest, CreateCardNoMesh)
{
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_CARD_NOMESH);
    EXPECT_TRUE(strategy != nullptr);
    EXPECT_EQ(strategy->get_root_info_size(), 4096);
}

TEST_F(ProductStrategyTest, CreateCard2PMesh)
{
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_CARD_2PMESH);
    EXPECT_TRUE(strategy != nullptr);
    EXPECT_EQ(strategy->get_root_info_size(), 4096);
}

TEST_F(ProductStrategyTest, CreateCard4PMesh)
{
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_CARD_4PMESH);
    EXPECT_TRUE(strategy != nullptr);
    EXPECT_EQ(strategy->get_root_info_size(), 4096);
}

TEST_F(ProductStrategyTest, CreateServer8PMesh)
{
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH);
    EXPECT_TRUE(strategy != nullptr);
    EXPECT_EQ(strategy->get_root_info_size(), 4096);
}

TEST_F(ProductStrategyTest, CreatePod)
{
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_POD);
    EXPECT_TRUE(strategy != nullptr);
    EXPECT_EQ(strategy->get_root_info_size(), 4096);
}

TEST_F(ProductStrategyTest, CreatePod2D)
{
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_POD_2D);
    EXPECT_TRUE(strategy != nullptr);
    EXPECT_EQ(strategy->get_root_info_size(), 4096);
}

TEST_F(ProductStrategyTest, CreateServerUbx)
{
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_SERVER_UBX);
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->get_root_info_size(), ACLSHMEMI_ROOT_INFO_SIZE);
    EXPECT_NE(dynamic_cast<aclshmemi_ubx_product_t*>(strategy.get()), nullptr);
}

TEST_F(ProductStrategyTest, CreateInvalidMainboardId)
{
    auto strategy = aclshmemi_product_strategy_t::create(0xFFFFFFFF);
    EXPECT_TRUE(strategy == nullptr);
}

TEST_F(ProductStrategyTest, CreateUnknownMainboardId)
{
    auto strategy = aclshmemi_product_strategy_t::create(0x99);
    EXPECT_TRUE(strategy == nullptr);
}

TEST_F(ProductStrategyTest, RootInfoSizeAllTypes)
{
    uint32_t mainboard_ids[] = {ACLSHMEMI_MAIN_BOARD_ID_CARD_NOMESH, ACLSHMEMI_MAIN_BOARD_ID_CARD_2PMESH,
                                ACLSHMEMI_MAIN_BOARD_ID_CARD_4PMESH, ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH,
                                ACLSHMEMI_MAIN_BOARD_ID_SERVER_UBX,  ACLSHMEMI_MAIN_BOARD_ID_POD,
                                ACLSHMEMI_MAIN_BOARD_ID_POD_2D};

    for (auto id : mainboard_ids) {
        auto strategy = aclshmemi_product_strategy_t::create(id);
        if (strategy) {
            EXPECT_EQ(strategy->get_root_info_size(), ACLSHMEMI_ROOT_INFO_SIZE);
        }
    }
}

TEST_F(ProductStrategyTest, CardProductTypeCheck)
{
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_CARD_4PMESH);
    EXPECT_TRUE(strategy != nullptr);

    aclshmemi_card_product_t* card_strategy = dynamic_cast<aclshmemi_card_product_t*>(strategy.get());
    EXPECT_TRUE(card_strategy != nullptr);
}

TEST_F(ProductStrategyTest, ServerProductTypeCheck)
{
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH);
    EXPECT_TRUE(strategy != nullptr);

    aclshmemi_server_product_t* server_strategy = dynamic_cast<aclshmemi_server_product_t*>(strategy.get());
    EXPECT_TRUE(server_strategy != nullptr);
}

TEST_F(ProductStrategyTest, PodProductTypeCheck)
{
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_POD);
    EXPECT_TRUE(strategy != nullptr);

    aclshmemi_pod_product_t* pod_strategy = dynamic_cast<aclshmemi_pod_product_t*>(strategy.get());
    EXPECT_TRUE(pod_strategy != nullptr);
}

TEST_F(ProductStrategyTest, UbxProductBuildsHcommCompatibleBaseLayer)
{
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_SERVER_UBX);
    ASSERT_NE(strategy, nullptr);

    EXPECT_CALL(MockHal::instance(), get_driver_install_path()).WillOnce(testing::Return("/usr/local/Ascend"));
    EXPECT_CALL(MockHal::instance(), get_server_id()).WillOnce(testing::Return("server-id"));
    EXPECT_CALL(MockHal::instance(), get_ue_list(0, testing::_)).WillOnce(testing::Invoke([](int, UEList* ue_list) {
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
        return 0;
    }));

    auto root_info = strategy->get_root_info(0, ACLSHMEMI_MAIN_BOARD_ID_SERVER_UBX);
    ASSERT_TRUE(root_info.has_value());
    EXPECT_EQ(root_info->topo_file_path, "/usr/local/Ascend/driver/topo/950/atlas_850_3.json");
    ASSERT_EQ(root_info->ranks.size(), 1U);
    ASSERT_EQ(root_info->ranks[0].level_list.size(), 1U);

    const auto& layer = root_info->ranks[0].level_list[0];
    EXPECT_EQ(layer.level, 0);
    EXPECT_EQ(layer.instance_id, "server-id");
    EXPECT_EQ(layer.net_type, "TOPO_FILE_DESC");
    ASSERT_EQ(layer.addr_list.size(), 5U);
    EXPECT_EQ(layer.addr_list[0].plane_id, "plane_1");
    EXPECT_EQ(layer.addr_list[2].plane_id, "plane_clos_1_4");
    EXPECT_EQ(layer.addr_list[3].plane_id, "plane_clos_1_5");
    EXPECT_EQ(layer.addr_list[4].plane_id, "plane_clos_1");
    ASSERT_EQ(layer.addr_list[4].ports.size(), 2U);
    EXPECT_EQ(layer.addr_list[4].ports[0], "1/4");
    EXPECT_EQ(layer.addr_list[4].ports[1], "1/5");
}

} // namespace topo
} // namespace shm
