/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclshmemi_product_strategy.h"

#include <algorithm>
#include <utility>

#include <securec.h>

#include "aclshmemi_eid_parser.h"
#include "aclshmemi_hal.h"
#include "host_rdma.h"
#include "utils/shmemi_logger.h"

namespace shm {
namespace topo {

static constexpr const char* TOPO_FILE_DIR_PATH = "driver/topo/950";
static constexpr const char* ROCE_PLANE_ID = "roce";

static std::string build_topo_file_path(const std::string& driver_path, const std::string& topo_filename)
{
    return driver_path + "/" + TOPO_FILE_DIR_PATH + "/" + topo_filename;
}

template <typename... Args>
static bool format_rootinfo_string(
    const char* context, char* buffer, size_t buffer_size, const char* format, Args... args)
{
    int ret = snprintf_s(buffer, buffer_size, buffer_size - 1, format, args...);
    if (ret < 0) {
        SHM_LOG_ERROR(context << " failed: snprintf_s ret=" << ret);
        return false;
    }
    return true;
}

static const char* card_topo_filename(uint32_t mainboard_id)
{
    switch (mainboard_id) {
        case ACLSHMEMI_MAIN_BOARD_ID_CARD_NOMESH:
            return "atlas_350_1.json";
        case ACLSHMEMI_MAIN_BOARD_ID_CARD_2PMESH:
            return "atlas_350_2.json";
        case ACLSHMEMI_MAIN_BOARD_ID_CARD_4PMESH:
            return "atlas_350_3.json";
        default:
            return "atlas_350_1.json";
    }
}

static const aclshmemi_ub_entity_rule_t g_ub_rules[] = {
    {ACLSHMEMI_MAIN_BOARD_ID_SERVER_TYPE1, 1, 0, 3, {4, 5, 6, 7}},
    {ACLSHMEMI_MAIN_BOARD_ID_SERVER_TYPE1, 1, 1, 2, {5, 6}},
    {ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH, 1, 0, 3, {1, 2, 3, 4, 5, 6, 7, 8}},
};

static const aclshmemi_pod_ub_entity_rule_t g_pod_ub_rules[] = {
    {ACLSHMEMI_MAIN_BOARD_ID_POD, 1, 1, 2, {0, 1, 2, 3, 5, 6}, {0, 1, 2, 3}, "plane_pg_0"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD, 1, 0, 2, {1, 2}, {0, 1, 2, 3}, "plane_pg_1"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD, 1, 0, 2, {0, 1, 2, 3, 4, 5}, {4, 5, 6, 7}, "plane_pg_0"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD, 1, 1, 2, {1, 2}, {4, 5, 6, 7}, "plane_pg_1"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD_2D, 1, 1, 2, {0, 1, 2, 3, 5, 6}, {0, 1, 2, 3}, "plane_pg_0"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD_2D, 1, 0, 2, {1, 2}, {0, 1, 2, 3}, "plane_pg_1"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD_2D, 1, 0, 2, {0, 1, 2, 3, 4, 5}, {4, 5, 6, 7}, "plane_pg_0"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD_2D, 1, 1, 2, {1, 2}, {4, 5, 6, 7}, "plane_pg_1"},
};

std::optional<aclshmemi_root_info_t> aclshmemi_product_strategy_t::get_root_info_without_roce(
    int phy_id, uint32_t mainboard_id)
{
    return get_root_info(phy_id, mainboard_id);
}

std::unique_ptr<aclshmemi_product_strategy_t> aclshmemi_product_strategy_t::create(uint32_t mainboard_id)
{
    switch (mainboard_id) {
        case ACLSHMEMI_MAIN_BOARD_ID_CARD_NOMESH:
        case ACLSHMEMI_MAIN_BOARD_ID_CARD_2PMESH:
        case ACLSHMEMI_MAIN_BOARD_ID_CARD_4PMESH:
            return std::make_unique<aclshmemi_card_product_t>();
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH:
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_TYPE1:
            return std::make_unique<aclshmemi_server_product_t>();
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_UBX:
            return std::make_unique<aclshmemi_ubx_product_t>();
        case ACLSHMEMI_MAIN_BOARD_ID_POD:
        case ACLSHMEMI_MAIN_BOARD_ID_POD_2D:
            return std::make_unique<aclshmemi_pod_product_t>();
        default:
            return nullptr;
    }
}
std::optional<aclshmemi_root_info_t> aclshmemi_card_product_t::get_root_info(int phy_id, uint32_t mainboard_id)
{
    return build_root_info(phy_id, mainboard_id, true);
}

std::optional<aclshmemi_root_info_t> aclshmemi_card_product_t::get_root_info_without_roce(
    int phy_id, uint32_t mainboard_id)
{
    return build_root_info(phy_id, mainboard_id, false);
}

std::optional<aclshmemi_root_info_t> aclshmemi_card_product_t::build_root_info(
    int phy_id, uint32_t mainboard_id, bool include_roce)
{
    auto& hal = aclshmemi_hal_t::instance();

    aclshmemi_root_info_t root_info;
    aclshmemi_root_info_init(root_info);
    std::string driver_path = hal.get_driver_install_path();
    aclshmemi_root_info_set_topo_file_path(
        root_info, build_topo_file_path(driver_path, card_topo_filename(mainboard_id)));

    UEList ue_list;
    memset_s(&ue_list, sizeof(ue_list), 0, sizeof(ue_list));
    int ret = hal.get_ue_list(phy_id, &ue_list);
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmemi_card_product_t::get_root_info failed: get_ue_list failed for phy_id=" << phy_id);
        return std::nullopt;
    }

    aclshmemi_rank_t rank;
    aclshmemi_rank_init(rank, phy_id, phy_id);

    dcmi_urma_eid_info_t eid_list[MAX_EID_NUM];
    memset_s(eid_list, sizeof(eid_list), 0, sizeof(eid_list));
    size_t eid_cnt = MAX_EID_NUM;
    for (uint32_t i = 0; i < ue_list.ueNum; ++i) {
        for (uint32_t j = 0; j < ue_list.ueList[i].eidNum && eid_cnt > 0; ++j) {
            if (eid_cnt > 0) {
                eid_list[MAX_EID_NUM - eid_cnt] = ue_list.ueList[i].eidList[j];
                eid_cnt--;
            }
        }
    }
    eid_cnt = MAX_EID_NUM - eid_cnt;

    if (mainboard_id == ACLSHMEMI_MAIN_BOARD_ID_CARD_2PMESH) {
        auto mesh_layer = process_mesh2p_layer(phy_id, eid_list, eid_cnt);
        if (mesh_layer) {
            aclshmemi_rank_add_net_layer(rank, *mesh_layer);
        }
    } else if (mainboard_id == ACLSHMEMI_MAIN_BOARD_ID_CARD_4PMESH) {
        auto mesh_layer = process_mesh_layer(phy_id, eid_list, eid_cnt);
        if (mesh_layer) {
            aclshmemi_rank_add_net_layer(rank, *mesh_layer);
        }
    }
    if (include_roce) {
        auto roce_layer = process_roce_layer(phy_id);
        if (roce_layer) {
            aclshmemi_rank_add_net_layer(rank, *roce_layer);
        }
    }

    aclshmemi_root_info_add_rank(root_info, rank);
    return root_info;
}

std::optional<aclshmemi_net_layer_t> aclshmemi_card_product_t::process_mesh_layer(
    int phy_id, const dcmi_urma_eid_info_t* eid_list, size_t eid_cnt)
{
    if (eid_cnt == 0) {
        return std::nullopt;
    }

    auto& hal = aclshmemi_hal_t::instance();
    std::string server_id = hal.get_server_id();

    char instance_id[64] = {0};
    snprintf_s(instance_id, sizeof(instance_id), sizeof(instance_id) - 1, "%s", server_id.c_str());

    aclshmemi_net_layer_t layer;
    aclshmemi_net_layer_init(layer, 0, instance_id);
    aclshmemi_net_layer_set_net_type(layer, "TOPO_FILE_DESC");

    for (size_t i = 0; i < eid_cnt; ++i) {
        int port_id = aclshmemi_eid_parser_t::get_port_id(eid_list[i].eid);
        if (port_id > 9) {
            continue;
        }
        int die_id = aclshmemi_eid_parser_t::get_die_id(eid_list[i].eid);

        aclshmemi_addr_t addr;
        aclshmemi_addr_set_eid(addr, eid_list[i].eid);

        char port[16] = {0};
        char plane_id[16] = {0};
        snprintf_s(port, sizeof(port), sizeof(port) - 1, "%d/%d", die_id, port_id);
        snprintf_s(plane_id, sizeof(plane_id), sizeof(plane_id) - 1, "plane_%d", die_id);
        aclshmemi_addr_add_port(addr, port);
        aclshmemi_addr_set_plane_id(addr, plane_id);

        aclshmemi_net_layer_add_addr(layer, addr);
    }

    return layer;
}

std::optional<aclshmemi_net_layer_t> aclshmemi_card_product_t::process_mesh2p_layer(
    int phy_id, const dcmi_urma_eid_info_t* eid_list, size_t eid_cnt)
{
    if (eid_cnt == 0) {
        return std::nullopt;
    }

    auto& hal = aclshmemi_hal_t::instance();
    std::string server_id = hal.get_server_id();

    char instance_id[64] = {0};
    snprintf_s(instance_id, sizeof(instance_id), sizeof(instance_id) - 1, "%s_%d", server_id.c_str(), phy_id / 2);

    aclshmemi_net_layer_t layer;
    aclshmemi_net_layer_init(layer, 0, instance_id);
    aclshmemi_net_layer_set_net_type(layer, "TOPO_FILE_DESC");

    for (size_t i = 0; i < eid_cnt; ++i) {
        int port_id = aclshmemi_eid_parser_t::get_port_id(eid_list[i].eid);
        if (port_id <= 9) {
            continue;
        }

        aclshmemi_addr_t addr;
        aclshmemi_addr_set_eid(addr, eid_list[i].eid);

        const int ports[] = {4, 5, 6, 8};
        for (int j = 0; j < 4; ++j) {
            char port[16] = {0};
            snprintf_s(port, sizeof(port), sizeof(port) - 1, "0/%d", ports[j]);
            aclshmemi_addr_add_port(addr, port);
        }
        aclshmemi_addr_set_plane_id(addr, "plane_0");
        aclshmemi_net_layer_add_addr(layer, addr);
    }

    return layer;
}

std::optional<aclshmemi_net_layer_t> aclshmemi_card_product_t::process_roce_layer(int phy_id)
{
    aclshmemi_net_layer_t layer;
    aclshmemi_net_layer_init(layer, 3, "cluster");
    aclshmemi_net_layer_set_net_type(layer, "CLOS");

    aclshmemi_addr_t addr;
    char ip_addr[64] = {0};
    if (get_npu_host_rdma_ip(phy_id, ip_addr, sizeof(ip_addr)) != 0) {
        return std::nullopt;
    }
    aclshmemi_addr_set_ip(addr, ip_addr);
    aclshmemi_addr_set_plane_id(addr, ROCE_PLANE_ID);
    aclshmemi_addr_add_port(addr, "d2h");

    aclshmemi_net_layer_add_addr(layer, addr);
    return layer;
}

std::optional<aclshmemi_net_layer_t> aclshmemi_server_product_t::process_mesh_layer(
    int phy_id, const UEList& ue_list, const dcmi_spod_info& spod_info)
{
    aclshmemi_net_layer_t layer;
    char instance_id[64] = {0};
    if (!format_rootinfo_string(
            "aclshmemi_server_product_t::process_mesh_layer instance_id", instance_id, sizeof(instance_id),
            "sp%u_srv%u", spod_info.super_pod_id, spod_info.server_index)) {
        return std::nullopt;
    }
    aclshmemi_net_layer_init(layer, 0, instance_id);
    aclshmemi_net_layer_set_net_type(layer, "MESH");

    constexpr int MESH_DIE = 1;
    int mesh_entity_id = aclshmemi_eid_parser_t::get_max_entity_id(ue_list, MESH_DIE);

    for (uint32_t i = 0; i < ue_list.ueNum; ++i) {
        int fe = aclshmemi_eid_parser_t::get_ub_entity_id(ue_list.ueList[i]);
        if (ue_list.ueList[i].eidNum == 0) {
            continue;
        }

        int die = aclshmemi_eid_parser_t::get_ub_die_id(ue_list.ueList[i].eidList[0].eid);
        if (fe != mesh_entity_id || die != MESH_DIE) {
            continue;
        }

        for (uint32_t j = 0; j < ue_list.ueList[i].eidNum; ++j) {
            if (aclshmemi_eid_parser_t::is_ub_port_group(ue_list.ueList[i].eidList[j].eid)) {
                continue;
            }

            aclshmemi_addr_t addr;
            aclshmemi_addr_set_eid(addr, ue_list.ueList[i].eidList[j].eid);

            char port[16] = {0};
            char plane_id[16] = {0};
            int phy_port_id = aclshmemi_eid_parser_t::get_ub_port_id(ue_list.ueList[i].eidList[j].eid);
            if (!format_rootinfo_string(
                    "aclshmemi_server_product_t::process_mesh_layer port", port, sizeof(port), "%d/%d", die,
                    phy_port_id)) {
                return std::nullopt;
            }
            if (!format_rootinfo_string(
                    "aclshmemi_server_product_t::process_mesh_layer plane_id", plane_id, sizeof(plane_id), "plane_%d",
                    die)) {
                return std::nullopt;
            }

            aclshmemi_addr_add_port(addr, port);
            aclshmemi_addr_set_plane_id(addr, plane_id);
            aclshmemi_net_layer_add_addr(layer, addr);
        }
    }

    return layer;
}

std::optional<aclshmemi_net_layer_t> aclshmemi_server_product_t::process_clos_layer(
    int phy_id, uint32_t mainboard_id, const UEList& ue_list, const dcmi_spod_info& spod_info)
{
    aclshmemi_net_layer_t layer;
    char instance_id[64] = {0};
    if (!format_rootinfo_string(
            "aclshmemi_server_product_t::process_clos_layer instance_id", instance_id, sizeof(instance_id),
            "superpod_%u", spod_info.super_pod_id)) {
        return std::nullopt;
    }
    aclshmemi_net_layer_init(layer, 1, instance_id);
    aclshmemi_net_layer_set_net_type(layer, "CLOS");

    constexpr size_t RULE_COUNT = sizeof(g_ub_rules) / sizeof(g_ub_rules[0]);

    for (uint32_t i = 0; i < ue_list.ueNum; ++i) {
        int fe = aclshmemi_eid_parser_t::get_ub_entity_id(ue_list.ueList[i]);
        int port_group_idx = aclshmemi_eid_parser_t::get_server_port_group_idx(ue_list.ueList[i]);
        if (port_group_idx < 0) {
            continue;
        }

        int die = aclshmemi_eid_parser_t::get_server_die_id(ue_list.ueList[i].eidList[port_group_idx].eid);

        for (size_t r = 0; r < RULE_COUNT; ++r) {
            if (mainboard_id == g_ub_rules[r].mainboard_id && die == g_ub_rules[r].die_id &&
                fe == g_ub_rules[r].ue_id) {
                aclshmemi_addr_t addr;
                aclshmemi_addr_set_eid(addr, ue_list.ueList[i].eidList[port_group_idx].eid);

                for (uint32_t j = 0; j < ue_list.ueList[i].eidNum; ++j) {
                    if (aclshmemi_eid_parser_t::is_ub_port_group(ue_list.ueList[i].eidList[j].eid)) {
                        continue;
                    }
                    int port_id = aclshmemi_eid_parser_t::get_ub_port_id(ue_list.ueList[i].eidList[j].eid);
                    char port[16] = {0};
                    if (!format_rootinfo_string(
                            "aclshmemi_server_product_t::process_clos_layer port", port, sizeof(port), "%d/%d", die,
                            port_id)) {
                        return std::nullopt;
                    }
                    aclshmemi_addr_add_port(addr, port);
                }

                char plane_id[16] = {0};
                if (!format_rootinfo_string(
                        "aclshmemi_server_product_t::process_clos_layer plane_id", plane_id, sizeof(plane_id),
                        "plane_clos_%d", die)) {
                    return std::nullopt;
                }
                aclshmemi_addr_set_plane_id(addr, plane_id);

                aclshmemi_net_layer_add_addr(layer, addr);
            }
        }
    }

    return layer;
}
std::optional<aclshmemi_root_info_t> aclshmemi_server_product_t::get_root_info(int phy_id, uint32_t mainboard_id)
{
    auto& hal = aclshmemi_hal_t::instance();

    aclshmemi_root_info_t root_info;
    aclshmemi_root_info_init(root_info);
    std::string driver_path = hal.get_driver_install_path();
    aclshmemi_root_info_set_topo_file_path(root_info, build_topo_file_path(driver_path, "atlas_850_1.json"));

    UEList ue_list;
    memset_s(&ue_list, sizeof(ue_list), 0, sizeof(ue_list));
    int ret = hal.get_ue_list(phy_id, &ue_list);
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmemi_server_product_t::get_root_info failed: get_ue_list failed for phy_id=" << phy_id);
        return std::nullopt;
    }

    dcmi_spod_info spod_info;
    memset_s(&spod_info, sizeof(spod_info), 0, sizeof(spod_info));
    ret = hal.get_spod_info(phy_id, &spod_info);
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmemi_server_product_t::get_root_info failed: get_spod_info failed for phy_id=" << phy_id);
        return std::nullopt;
    }

    aclshmemi_rank_t rank;
    aclshmemi_rank_init(rank, phy_id, phy_id);

    auto mesh_layer = process_mesh_layer(phy_id, ue_list, spod_info);
    if (!mesh_layer) {
        SHM_LOG_ERROR("aclshmemi_server_product_t::get_root_info failed: process_mesh_layer failed");
        return std::nullopt;
    }
    aclshmemi_rank_add_net_layer(rank, *mesh_layer);

    auto clos_layer = process_clos_layer(phy_id, mainboard_id, ue_list, spod_info);
    if (!clos_layer) {
        SHM_LOG_ERROR("aclshmemi_server_product_t::get_root_info failed: process_clos_layer failed");
        return std::nullopt;
    }
    aclshmemi_rank_add_net_layer(rank, *clos_layer);

    aclshmemi_root_info_add_rank(root_info, rank);
    return root_info;
}

std::optional<aclshmemi_net_layer_t> aclshmemi_pod_product_t::process_mesh_layer(
    int phy_id, const UEList& ue_list, const dcmi_spod_info& spod_info)
{
    aclshmemi_net_layer_t layer;
    char instance_id[64] = {0};
    if (!format_rootinfo_string(
            "aclshmemi_pod_product_t::process_mesh_layer instance_id", instance_id, sizeof(instance_id), "sp%u_srv%u",
            spod_info.super_pod_id, spod_info.server_index)) {
        return std::nullopt;
    }
    aclshmemi_net_layer_init(layer, 0, instance_id);
    aclshmemi_net_layer_set_net_type(layer, "MESH");

    constexpr int NPU_NUM = 8;
    constexpr int MAX_MESH_PORT_ID = 9;
    const int mesh_die_id = (phy_id % NPU_NUM) < 4 ? 0 : 1;
    int mesh_entity_id = aclshmemi_eid_parser_t::get_max_entity_id(ue_list, mesh_die_id);

    for (uint32_t i = 0; i < ue_list.ueNum; ++i) {
        int fe = aclshmemi_eid_parser_t::get_ub_entity_id(ue_list.ueList[i]);
        if (ue_list.ueList[i].eidNum == 0) {
            continue;
        }

        int die = aclshmemi_eid_parser_t::get_ub_die_id(ue_list.ueList[i].eidList[0].eid);
        if (fe != mesh_entity_id || die != mesh_die_id) {
            continue;
        }

        for (uint32_t j = 0; j < ue_list.ueList[i].eidNum; ++j) {
            int phy_port_id = aclshmemi_eid_parser_t::get_ub_port_id(ue_list.ueList[i].eidList[j].eid);
            if (phy_port_id > MAX_MESH_PORT_ID) {
                continue;
            }

            aclshmemi_addr_t addr;
            aclshmemi_addr_set_eid(addr, ue_list.ueList[i].eidList[j].eid);

            char port[16] = {0};
            if (!format_rootinfo_string(
                    "aclshmemi_pod_product_t::process_mesh_layer port", port, sizeof(port), "%d/%d", die,
                    phy_port_id)) {
                return std::nullopt;
            }
            aclshmemi_addr_add_port(addr, port);
            aclshmemi_addr_set_plane_id(addr, "plane_0");

            aclshmemi_net_layer_add_addr(layer, addr);
        }
    }

    return layer;
}

static const aclshmemi_pod_ub_entity_rule_t* get_pod_ub_rule(
    int npu_id, int level, uint32_t mainboard_id, int die, int fe_id)
{
    constexpr size_t RULE_COUNT = sizeof(g_pod_ub_rules) / sizeof(g_pod_ub_rules[0]);
    constexpr int NPU_NUM = 8;

    for (size_t i = 0; i < RULE_COUNT; ++i) {
        if (g_pod_ub_rules[i].level != level) {
            continue;
        }

        int npu_in_rule = 0;
        for (size_t j = 0; j < g_pod_ub_rules[i].npus.size(); ++j) {
            if (g_pod_ub_rules[i].npus[j] == (npu_id % NPU_NUM)) {
                npu_in_rule = 1;
                break;
            }
        }
        if (npu_in_rule == 0) {
            continue;
        }

        if (mainboard_id == g_pod_ub_rules[i].mainboard_id && die == g_pod_ub_rules[i].die_id &&
            fe_id == g_pod_ub_rules[i].ue_id) {
            return &g_pod_ub_rules[i];
        }
    }
    return nullptr;
}

std::optional<aclshmemi_net_layer_t> aclshmemi_pod_product_t::process_clos_layer(
    int phy_id, uint32_t mainboard_id, const UEList& ue_list, const dcmi_spod_info& spod_info)
{
    aclshmemi_net_layer_t layer;
    char instance_id[64] = {0};
    if (!format_rootinfo_string(
            "aclshmemi_pod_product_t::process_clos_layer instance_id", instance_id, sizeof(instance_id), "superpod_%u",
            spod_info.super_pod_id)) {
        return std::nullopt;
    }
    aclshmemi_net_layer_init(layer, 1, instance_id);
    aclshmemi_net_layer_set_net_type(layer, "CLOS");

    std::optional<aclshmemi_addr_t> primary_addr;
    std::optional<aclshmemi_addr_t> secondary_addr;
    for (uint32_t i = 0; i < ue_list.ueNum; ++i) {
        int fe = aclshmemi_eid_parser_t::get_ub_entity_id(ue_list.ueList[i]);
        int port_group_idx = aclshmemi_eid_parser_t::get_server_port_group_idx(ue_list.ueList[i]);
        if (port_group_idx < 0) {
            continue;
        }

        int die = aclshmemi_eid_parser_t::get_pod_die_id(ue_list.ueList[i].eidList[port_group_idx].eid);

        const aclshmemi_pod_ub_entity_rule_t* rule = get_pod_ub_rule(phy_id, 1, mainboard_id, die, fe);
        if (rule == nullptr) {
            continue;
        }

        aclshmemi_addr_t addr;
        aclshmemi_addr_set_eid(addr, ue_list.ueList[i].eidList[port_group_idx].eid);

        int port_num = 0;
        for (uint32_t j = 0; j < ue_list.ueList[i].eidNum; ++j) {
            if (aclshmemi_eid_parser_t::is_ub_port_group(ue_list.ueList[i].eidList[j].eid)) {
                continue;
            }
            int port_id = aclshmemi_eid_parser_t::get_ub_port_id(ue_list.ueList[i].eidList[j].eid);
            char port[16] = {0};
            if (!format_rootinfo_string(
                    "aclshmemi_pod_product_t::process_clos_layer port", port, sizeof(port), "%d/%d", die, port_id)) {
                return std::nullopt;
            }
            aclshmemi_addr_add_port(addr, port);
            port_num++;
        }

        constexpr int PRIMARY_PORT_NUM = 6;
        if (port_num >= PRIMARY_PORT_NUM) {
            aclshmemi_addr_set_plane_id(addr, "plane_0");
            primary_addr = addr;
        } else {
            aclshmemi_addr_set_plane_id(addr, "plane_1");
            secondary_addr = addr;
        }
    }
    if (primary_addr) {
        aclshmemi_net_layer_add_addr(layer, *primary_addr);
    }
    if (secondary_addr) {
        aclshmemi_net_layer_add_addr(layer, *secondary_addr);
    }

    return layer;
}
std::optional<aclshmemi_root_info_t> aclshmemi_pod_product_t::get_root_info(int phy_id, uint32_t mainboard_id)
{
    auto& hal = aclshmemi_hal_t::instance();

    aclshmemi_root_info_t root_info;
    aclshmemi_root_info_init(root_info);
    std::string driver_path = hal.get_driver_install_path();
    aclshmemi_root_info_set_topo_file_path(root_info, build_topo_file_path(driver_path, "atlas_950_1.json"));

    UEList ue_list;
    memset_s(&ue_list, sizeof(ue_list), 0, sizeof(ue_list));
    int ret = hal.get_ue_list(phy_id, &ue_list);
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmemi_pod_product_t::get_root_info failed: get_ue_list failed for phy_id=" << phy_id);
        return std::nullopt;
    }

    dcmi_spod_info spod_info;
    memset_s(&spod_info, sizeof(spod_info), 0, sizeof(spod_info));
    ret = hal.get_spod_info(phy_id, &spod_info);
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmemi_pod_product_t::get_root_info failed: get_spod_info failed for phy_id=" << phy_id);
        return std::nullopt;
    }

    constexpr int NPU_NUM = 8;
    int local_id = (phy_id % NPU_NUM) + ((spod_info.server_index % NPU_NUM) * NPU_NUM);

    aclshmemi_rank_t rank;
    aclshmemi_rank_init(rank, phy_id, local_id);

    auto mesh_layer = process_mesh_layer(phy_id, ue_list, spod_info);
    if (!mesh_layer) {
        SHM_LOG_ERROR("aclshmemi_pod_product_t::get_root_info failed: process_mesh_layer failed");
        return std::nullopt;
    }
    aclshmemi_rank_add_net_layer(rank, *mesh_layer);

    auto clos_layer = process_clos_layer(phy_id, mainboard_id, ue_list, spod_info);
    if (!clos_layer) {
        SHM_LOG_ERROR("aclshmemi_pod_product_t::get_root_info failed: process_clos_layer failed");
        return std::nullopt;
    }
    aclshmemi_rank_add_net_layer(rank, *clos_layer);

    aclshmemi_root_info_add_rank(root_info, rank);
    return root_info;
}

namespace {

const UBEntity* find_ubx_entity(const UEList& ue_list, int die_id, int fe_id)
{
    const uint32_t ue_num = std::min<uint32_t>(ue_list.ueNum, MAX_UE_PER_NPU);
    for (uint32_t i = 0; i < ue_num; ++i) {
        const auto& entity = ue_list.ueList[i];
        if (entity.eidNum == 0) {
            continue;
        }
        if (aclshmemi_eid_parser_t::get_ub_die_id(entity.eidList[0].eid) == die_id &&
            aclshmemi_eid_parser_t::get_ub_fe_id(entity.eidList[0].eid) == fe_id) {
            return &entity;
        }
    }
    return nullptr;
}

bool add_ubx_port_addresses(const UBEntity& entity, bool clos_ports, aclshmemi_net_layer_t& layer)
{
    const uint32_t eid_num = std::min<uint32_t>(entity.eidNum, MAX_EID_PER_UE);
    const int port_group_index = aclshmemi_eid_parser_t::get_server_port_group_idx(entity);
    for (uint32_t i = 0; i < eid_num; ++i) {
        const auto& eid = entity.eidList[i].eid;
        if (port_group_index == static_cast<int>(i)) {
            continue;
        }
        const int die_id = aclshmemi_eid_parser_t::get_ub_die_id(eid);
        const int port_id = aclshmemi_eid_parser_t::get_ub_port_id(eid);

        aclshmemi_addr_t addr;
        aclshmemi_addr_set_eid(addr, eid);
        char port[16] = {0};
        char plane_id[32] = {0};
        if (!format_rootinfo_string("add_ubx_port_addresses port", port, sizeof(port), "%d/%d", die_id, port_id)) {
            return false;
        }
        if (clos_ports) {
            if (!format_rootinfo_string(
                    "add_ubx_port_addresses CLOS plane", plane_id, sizeof(plane_id), "plane_clos_%d_%d", die_id,
                    port_id)) {
                return false;
            }
        } else {
            if (!format_rootinfo_string(
                    "add_ubx_port_addresses MESH plane", plane_id, sizeof(plane_id), "plane_%d", die_id)) {
                return false;
            }
        }
        aclshmemi_addr_add_port(addr, port);
        aclshmemi_addr_set_plane_id(addr, plane_id);
        aclshmemi_net_layer_add_addr(layer, addr);
    }
    return true;
}

bool add_ubx_clos_address(const UBEntity& entity, aclshmemi_net_layer_t& layer)
{
    const int port_group_index = aclshmemi_eid_parser_t::get_server_port_group_idx(entity);
    if (port_group_index < 0) {
        return true;
    }
    const auto& port_group_eid = entity.eidList[static_cast<size_t>(port_group_index)].eid;
    const int die_id = aclshmemi_eid_parser_t::get_ub_die_id(port_group_eid);

    aclshmemi_addr_t addr;
    aclshmemi_addr_set_eid(addr, port_group_eid);
    const uint32_t eid_num = std::min<uint32_t>(entity.eidNum, MAX_EID_PER_UE);
    for (uint32_t i = 0; i < eid_num; ++i) {
        if (static_cast<int>(i) == port_group_index) {
            continue;
        }
        char port[16] = {0};
        if (!format_rootinfo_string(
                "add_ubx_clos_address port", port, sizeof(port), "%d/%d", die_id,
                aclshmemi_eid_parser_t::get_ub_port_id(entity.eidList[i].eid))) {
            return false;
        }
        aclshmemi_addr_add_port(addr, port);
    }
    char plane_id[32] = {0};
    if (!format_rootinfo_string("add_ubx_clos_address plane", plane_id, sizeof(plane_id), "plane_clos_%d", die_id)) {
        return false;
    }
    aclshmemi_addr_set_plane_id(addr, plane_id);
    aclshmemi_net_layer_add_addr(layer, addr);
    return true;
}

} // namespace

std::optional<aclshmemi_net_layer_t> aclshmemi_ubx_product_t::process_base_layer(const UEList& ue_list)
{
    aclshmemi_net_layer_t layer;
    aclshmemi_net_layer_init(layer, 0, aclshmemi_hal_t::instance().get_server_id());
    aclshmemi_net_layer_set_net_type(layer, "TOPO_FILE_DESC");

    constexpr int UBX_DIE_ID = 1;
    constexpr int UBX_MESH_FE_ID = 3;
    constexpr int UBX_CLOS_FE_ID = 2;
    const UBEntity* mesh_entity = find_ubx_entity(ue_list, UBX_DIE_ID, UBX_MESH_FE_ID);
    if (mesh_entity != nullptr && !add_ubx_port_addresses(*mesh_entity, false, layer)) {
        return std::nullopt;
    }

    const UBEntity* clos_entity = find_ubx_entity(ue_list, UBX_DIE_ID, UBX_CLOS_FE_ID);
    if (clos_entity != nullptr) {
        if (!add_ubx_port_addresses(*clos_entity, true, layer) || !add_ubx_clos_address(*clos_entity, layer)) {
            return std::nullopt;
        }
    }

    return layer.addr_list.empty() ? std::nullopt : std::optional<aclshmemi_net_layer_t>(std::move(layer));
}

std::optional<aclshmemi_root_info_t> aclshmemi_ubx_product_t::get_root_info(int phy_id, uint32_t mainboard_id)
{
    if (mainboard_id != ACLSHMEMI_MAIN_BOARD_ID_SERVER_UBX) {
        return std::nullopt;
    }

    auto& hal = aclshmemi_hal_t::instance();
    UEList ue_list;
    if (memset_s(&ue_list, sizeof(ue_list), 0, sizeof(ue_list)) != EOK || hal.get_ue_list(phy_id, &ue_list) != 0) {
        SHM_LOG_ERROR("aclshmemi_ubx_product_t::get_root_info failed: get_ue_list failed for phy_id=" << phy_id);
        return std::nullopt;
    }

    aclshmemi_root_info_t root_info;
    aclshmemi_root_info_init(root_info);
    aclshmemi_root_info_set_topo_file_path(
        root_info, build_topo_file_path(hal.get_driver_install_path(), "atlas_850_3.json"));

    aclshmemi_rank_t rank;
    aclshmemi_rank_init(rank, phy_id, phy_id);
    auto base_layer = process_base_layer(ue_list);
    if (!base_layer) {
        SHM_LOG_ERROR(
            "aclshmemi_ubx_product_t::get_root_info failed: no usable UBX base-layer EID for phy_id=" << phy_id);
        return std::nullopt;
    }
    aclshmemi_rank_add_net_layer(rank, *base_layer);
    aclshmemi_root_info_add_rank(root_info, rank);
    return root_info;
}

} // namespace topo
} // namespace shm
