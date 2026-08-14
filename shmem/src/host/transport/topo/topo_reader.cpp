/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <arpa/inet.h>
#include <cctype>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
#include "rootinfo/aclshmemi_virtual_topology_root_info.h"
#endif
#include "rootinfo/topo_addr_info.h"
#include "shmemi_file_util.h"
#include "shmemi_host_common.h"
#include "topo_reader.h"

namespace shm {
namespace transport {

namespace {

bool CheckTopoFilePath(const std::string& path, std::string& realPath)
{
    realPath = path;
    if (utils::FileUtil::IsSymlink(realPath) || !utils::FileUtil::Realpath(realPath) ||
        !utils::FileUtil::IsFile(realPath) || !utils::FileUtil::CheckFileSize(realPath)) {
        SHM_LOG_ERROR("Topo file path check failed: " << path);
        return false;
    }
    return true;
}

bool NetTypeFromString(const std::string& netType, NetType& out)
{
    if (netType == "TOPO_FILE_DESC") {
        out = NetType::TopoFileDesc;
        return true;
    }
    if (netType == "MESH") {
        out = NetType::Mesh;
        return true;
    }
    if (netType == "CLOS") {
        out = NetType::Clos;
        return true;
    }
    return false;
}

// addrType defaults to EID when the field is absent; IPV4 / IPV6 select the dotted-string
// converters. Matching is case-insensitive: the caller upper-cases before calling.
bool RankAddrTypeFromString(const std::string& addrType, RankAddrType& out)
{
    if (addrType == "EID") {
        out = RankAddrType::EID;
        return true;
    }
    if (addrType == "IPV4") {
        out = RankAddrType::IPV4;
        return true;
    }
    if (addrType == "IPV6") {
        out = RankAddrType::IPV6;
        return true;
    }
    return false;
}

const char* AddrTypeName(RankAddrType addrType)
{
    switch (addrType) {
        case RankAddrType::EID:
            return "EID";
        case RankAddrType::IPV4:
            return "IPV4";
        case RankAddrType::IPV6:
            return "IPV6";
        default:
            return "UNKNOWN";
    }
}

// TryGet: extract a typed value from a json node. Returns true and writes out only when
// the node holds a compatible value; never logs. One overload per supported field type.
bool TryGet(const nlohmann::json& value, std::string& out)
{
    if (!value.is_string()) {
        return false;
    }
    out = value.get<std::string>();
    return true;
}

bool TryGet(const nlohmann::json& value, uint32_t& out)
{
    try {
        if (value.is_string()) {
            out = static_cast<uint32_t>(std::stoul(value.get<std::string>()));
            return true;
        }
        if (value.is_number()) {
            out = value.get<uint32_t>();
            return true;
        }
    } catch (const std::exception&) {
        // fall through to the failure return below
    }
    return false;
}

bool TryGet(const nlohmann::json& value, std::vector<std::string>& out)
{
    if (!value.is_array()) {
        return false;
    }
    out.clear();
    for (const auto& element : value) {
        if (element.is_string()) {
            out.push_back(element.get<std::string>());
        }
    }
    return true;
}

// GetRequiredField: field must exist and hold a value of type T, else log and fail.
// context carries caller-side locators (localId/phyId/eidIndex) for diagnostics.
template <typename T>
bool GetRequiredField(const nlohmann::json& obj, const char* key, T& out, const std::string& context = "")
{
    const auto item = obj.find(key);
    if (item == obj.end() || !TryGet(*item, out)) {
        SHM_LOG_ERROR("Rootinfo/Topo: missing or invalid field '" << key << "'. " << context);
        return false;
    }
    return true;
}

// GetOptionalField: write out only when the field exists and matches type T, else leave
// out untouched. Silent by design.
template <typename T>
bool GetOptionalField(const nlohmann::json& obj, const char* key, T& out)
{
    const auto item = obj.find(key);
    return item != obj.end() && TryGet(*item, out);
}

// GetRequiredArray: returns a pointer to the array node for iteration, or nullptr (with a
// log) when the field is missing, not an array, or empty while requireNonEmpty is set.
const nlohmann::json* GetRequiredArray(
    const nlohmann::json& obj, const char* key, bool requireNonEmpty, const std::string& context = "")
{
    const auto item = obj.find(key);
    if (item == obj.end() || !item->is_array() || (requireNonEmpty && item->empty())) {
        SHM_LOG_ERROR("Rootinfo/Topo: missing or invalid array '" << key << "'. " << context);
        return nullptr;
    }
    return &(*item);
}

} // namespace

bool TopoReader::LoadRootInfoJson(uint32_t phyId, nlohmann::json& out)
{
    bool shouldFallback = false;
    std::ifstream rootInfoFile(ROOTINFO_PATH);
    if (!rootInfoFile.is_open()) {
        SHM_LOG_WARN(
            "Rootinfo file not found at " << ROOTINFO_PATH << ", fallback to generated rootinfo for phyId " << phyId);
        shouldFallback = true;
    } else {
        try {
            rootInfoFile >> out;
        } catch (const std::exception& ex) {
            SHM_LOG_ERROR("Parse rootinfo file failed, the path is " << ROOTINFO_PATH << ", error: " << ex.what());
            SHM_LOG_WARN("Rootinfo file parse failed, fallback to generated rootinfo for phyId " << phyId);
            shouldFallback = true;
        }
    }

    if (!shouldFallback) {
        SHM_LOG_INFO("Load rootinfo from file " << ROOTINFO_PATH);
        return true;
    }

    std::string generatedRootInfo;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    auto virtualTopologyRootInfo =
        shm::topo::aclshmemi_generate_virtual_topology_root_info_json(static_cast<int>(phyId));
    if (!virtualTopologyRootInfo) {
        SHM_LOG_ERROR("Failed to generate virtual-topology rootinfo for phyId " << phyId);
        return false;
    }
    generatedRootInfo = std::move(*virtualTopologyRootInfo);
#else
    size_t rootInfoSize = 0;
    int ret = topo_addr_info_get_size(static_cast<int>(phyId), &rootInfoSize);
    if (ret != 0 || rootInfoSize == 0) {
        SHM_LOG_ERROR(
            "Failed to get generated rootinfo size for phyId " << phyId << ", ret = " << ret
                                                               << ", size = " << rootInfoSize);
        return false;
    }
    SHM_LOG_INFO("Generated rootinfo size for phyId " << phyId << " is " << rootInfoSize);

    std::vector<char> rootInfoBuffer(rootInfoSize + 1, '\0');
    size_t actualSize = rootInfoSize;
    ret = topo_addr_info_get(static_cast<int>(phyId), rootInfoBuffer.data(), &actualSize);
    if (ret != 0 || actualSize == 0) {
        SHM_LOG_ERROR(
            "Failed to get generated rootinfo for phyId " << phyId << ", ret = " << ret
                                                          << ", actualSize = " << actualSize);
        return false;
    }
    if (actualSize > rootInfoBuffer.size() - 1) {
        SHM_LOG_ERROR(
            "Generated rootinfo size overflow, actualSize " << actualSize << ", capacity " << rootInfoBuffer.size());
        return false;
    }
    generatedRootInfo.assign(rootInfoBuffer.data(), actualSize);
#endif

    try {
        out = nlohmann::json::parse(generatedRootInfo);
#ifdef DEBUG_MODE
        SHM_LOG_DEBUG("Generated rootinfo json for phyId " << phyId << ":\n" << out.dump(2));
#endif
    } catch (const std::exception& ex) {
        SHM_LOG_ERROR("Failed to parse generated rootinfo json for phyId " << phyId << ", error: " << ex.what());
        return false;
    }

    SHM_LOG_INFO("Use generated rootinfo fallback for phyId " << phyId);
    return true;
}

bool TopoReader::ParseRootInfo(uint32_t phyId, RootInfo& out)
{
    out = RootInfo{};
    nlohmann::json rootInfoJson;
    if (!LoadRootInfoJson(phyId, rootInfoJson)) {
        SHM_LOG_ERROR("Failed to load rootinfo for phyId " << phyId);
        return false;
    }

    if (!ParseRootInfoJson(rootInfoJson, phyId, out)) {
        out = RootInfo{};
        SHM_LOG_ERROR("Rootinfo content is unusable for phyId " << phyId);
        return false;
    }
    return true;
}

bool TopoReader::ParseLevelInfo(const nlohmann::json& levelJson, LevelInfo& out)
{
    std::string netType;
    if (!GetRequiredField(levelJson, "net_type", netType)) {
        return false;
    }
    if (!NetTypeFromString(netType, out.netType)) {
        SHM_LOG_ERROR("Rootinfo: unsupported net_type " << netType);
        return false;
    }

    out.netLayer = 0;
    if (!GetRequiredField(levelJson, "net_layer", out.netLayer) ||
        !GetRequiredField(levelJson, "net_instance_id", out.netInstanceId)) {
        return false;
    }
    GetOptionalField(levelJson, "net_attr", out.netAttr);
    return true;
}

bool TopoReader::ParseRankAddr(
    const nlohmann::json& rankAddrJson, uint32_t localId, uint32_t eidIndex,
    const std::shared_ptr<LevelInfo>& levelInfo, RankAddr& out)
{
    const std::string context = "localId " + std::to_string(localId) + ", eidIndex " + std::to_string(eidIndex);
    if (!rankAddrJson.contains("addr")) {
        SHM_LOG_ERROR("Rootinfo rank_addr entry missing addr, " << context);
        return false;
    }

    out.levelInfo = levelInfo;

    std::string addrTypeStr = "EID";
    if (rankAddrJson.contains("addr_type") && !TryGet(rankAddrJson["addr_type"], addrTypeStr)) {
        SHM_LOG_ERROR("Rootinfo addr_type format is unsupported, " << context);
        return false;
    }
    std::transform(addrTypeStr.begin(), addrTypeStr.end(), addrTypeStr.begin(), [](unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
    });
    if (!RankAddrTypeFromString(addrTypeStr, out.addrType)) {
        SHM_LOG_ERROR("Rootinfo addr_type is unsupported: " << addrTypeStr << ", " << context);
        return false;
    }

    if (!EidConverter::Convert(out.addrType, rankAddrJson["addr"], out.eidData)) {
        SHM_LOG_ERROR("Rootinfo: failed to parse addr, " << context);
        return false;
    }

    if (!GetRequiredField(rankAddrJson, "plane_id", out.planeId, context)) {
        return false;
    }
    GetOptionalField(rankAddrJson, "ports", out.ports);
    return true;
}

bool TopoReader::ParseRootInfoJson(const nlohmann::json& rootInfoJson, uint32_t phyId, RootInfo& out)
{
    if (!GetRequiredField(rootInfoJson, "topo_file_path", out.topoFilePath)) {
        return false;
    }

    const auto* rankList = GetRequiredArray(rootInfoJson, "rank_list", false);
    if (rankList == nullptr) {
        return false;
    }

    for (const auto& rankJson : *rankList) {
        uint32_t deviceId = 0;
        uint32_t localId = 0;
        if (!GetRequiredField(rankJson, "device_id", deviceId) || !GetRequiredField(rankJson, "local_id", localId)) {
            return false;
        }
        if (deviceId != phyId) {
            continue;
        }

        out.deviceId = deviceId;
        out.localId = localId;
        const std::string context = "phyId " + std::to_string(phyId) + ", localId " + std::to_string(localId);

        const auto* levelList = GetRequiredArray(rankJson, "level_list", true, context);
        if (levelList == nullptr) {
            return false;
        }

        uint32_t eidIndex = 0;
        for (const auto& levelJson : *levelList) {
            auto levelInfo = std::make_shared<LevelInfo>();
            if (!ParseLevelInfo(levelJson, *levelInfo)) {
                return false;
            }
            out.levels.push_back(levelInfo);

            const auto* rankAddrList = GetRequiredArray(levelJson, "rank_addr_list", false, context);
            if (rankAddrList == nullptr) {
                return false;
            }

            for (const auto& rankAddrJson : *rankAddrList) {
                auto rankAddr = std::make_shared<RankAddr>();
                if (!ParseRankAddr(rankAddrJson, localId, eidIndex, levelInfo, *rankAddr)) {
                    SHM_LOG_WARN("Rootinfo: skip invalid rank_addr, localId " << localId << ", eidIndex " << eidIndex);
                    ++eidIndex;
                    continue;
                }
                out.eidIndexToRankAddr[eidIndex] = rankAddr;
                for (const auto& port : rankAddr->ports) {
                    out.portsToRankAddr[port] = rankAddr;
                }
                ++eidIndex;
            }
        }

        out.totalEidCount = static_cast<uint32_t>(out.eidIndexToRankAddr.size());
        if (out.totalEidCount == 0) {
            SHM_LOG_ERROR("Rootinfo: invalid eid count for phyId " << phyId << ", localId " << localId);
            return false;
        }

        SHM_LOG_INFO(
            "Parse rootinfo success for phyId " << phyId << ", localId " << localId << ", eidCount "
                                                << out.totalEidCount);
        return true;
    }

    SHM_LOG_ERROR("Rootinfo: no rank entry found for phyId " << phyId);
    return false;
}

bool TopoReader::ParseTopoInfo(const std::string& path, TopoInfo& out)
{
    out = TopoInfo{};
    std::string realPath;
    if (!CheckTopoFilePath(path, realPath)) {
        return false;
    }

    std::ifstream topoFile(realPath);
    if (!topoFile.is_open()) {
        SHM_LOG_ERROR("Failed to open topo file: " << realPath);
        return false;
    }
    nlohmann::json topoInfoJson;
    try {
        topoFile >> topoInfoJson;
    } catch (const nlohmann::json::exception& e) {
        SHM_LOG_ERROR("Topo parse failed: " << e.what());
        return false;
    }
    SHM_LOG_INFO("Read topo json from " << realPath);

    const auto* edgeList = GetRequiredArray(topoInfoJson, "edge_list", false);
    if (edgeList == nullptr) {
        return false;
    }

    for (const auto& edgeObj : *edgeList) {
        if (!edgeObj.contains("local_a")) {
            continue;
        }

        // CLOS edges describe a scale-out attachment (single endpoint, no peer local_b);
        // every other edge is a point-to-point mesh link between local_a and local_b.
        bool isClos = false;
        std::string topoType;
        if (GetOptionalField(edgeObj, "topo_type", topoType)) {
            isClos = topoType == "CLOS";
        }
        if (!edgeObj.contains("local_b")) {
            isClos = true;
        }

        if (isClos) {
            ClosTopoEdge edge;
            edge.netLayer = 0;
            edge.topoInstanceId = 0;
            if (!GetRequiredField(edgeObj, "local_a", edge.localA) ||
                !GetRequiredField(edgeObj, "net_layer", edge.netLayer) ||
                !GetRequiredField(edgeObj, "topo_instance_id", edge.topoInstanceId)) {
                return false;
            }
            GetOptionalField(edgeObj, "local_a_ports", edge.ports);
            out.closTopoEdges.push_back(std::move(edge));
            continue;
        }

        MeshTopoEdge edge;
        if (!GetRequiredField(edgeObj, "local_a", edge.localA) || !GetRequiredField(edgeObj, "local_b", edge.localB)) {
            return false;
        }
        GetOptionalField(edgeObj, "local_a_ports", edge.localAPorts);
        GetOptionalField(edgeObj, "local_b_ports", edge.localBPorts);
        out.meshTopoEdges.push_back(std::move(edge));
    }

    if (out.meshTopoEdges.empty() && out.closTopoEdges.empty()) {
        SHM_LOG_ERROR("Topo parse failed: no valid edge entries parsed.");
        return false;
    }

    // Keep CLOS edges ordered by netLayer ascending so route lookup probes the
    // closest scale-out layer first.
    std::stable_sort(
        out.closTopoEdges.begin(), out.closTopoEdges.end(),
        [](const ClosTopoEdge& lhs, const ClosTopoEdge& rhs) { return lhs.netLayer < rhs.netLayer; });
    return true;
}

// Resolves the local route used to connect from myLocalId to peerLocalId. Only the
// outbound (local) eid index can be derived here; the peer must supply its own remote index.
// The mesh fabric is probed first; if no direct mesh link exists, the CLOS planes are
// probed from the lowest netLayer up, matching a plane shared by both endpoints.
bool TopoReader::GetLocalEidRouteForPeer(
    const RootInfo& root, const TopoInfo& topo, uint32_t myLocalId, uint32_t peerLocalId, uint32_t& localEidIndex,
    EidData& localEidRaw)
{
    std::string localPort;
    for (const auto& edge : topo.meshTopoEdges) {
        if (edge.localA == myLocalId && edge.localB == peerLocalId && !edge.localAPorts.empty()) {
            localPort = edge.localAPorts[0];
            break;
        }
        if (edge.localB == myLocalId && edge.localA == peerLocalId && !edge.localBPorts.empty()) {
            localPort = edge.localBPorts[0];
            break;
        }
    }

    if (localPort.empty()) {
        for (const auto& myEdge : topo.closTopoEdges) {
            if (myEdge.localA != myLocalId || myEdge.ports.empty()) {
                continue;
            }
            for (const auto& peerEdge : topo.closTopoEdges) {
                if (peerEdge.localA == peerLocalId && peerEdge.netLayer == myEdge.netLayer &&
                    peerEdge.topoInstanceId == myEdge.topoInstanceId) {
                    localPort = myEdge.ports[0];
                    break;
                }
            }
            if (!localPort.empty()) {
                break;
            }
        }
    }

    if (localPort.empty()) {
        SHM_LOG_ERROR(
            "Failed to get local eid route, no usable edge between localId " << myLocalId << " and " << peerLocalId);
        return false;
    }

    const auto portItem = root.portsToRankAddr.find(localPort);
    if (portItem == root.portsToRankAddr.end() || portItem->second == nullptr) {
        SHM_LOG_ERROR("Failed to get local eid index, port " << localPort << " not found for localId " << myLocalId);
        return false;
    }
    const auto& rankAddr = portItem->second;

    bool found = false;
    for (const auto& item : root.eidIndexToRankAddr) {
        if (item.second == rankAddr) {
            localEidIndex = item.first;
            found = true;
            break;
        }
    }
    if (!found) {
        SHM_LOG_ERROR("Failed to get local eid index, rank addr for port " << localPort << " is not indexed");
        return false;
    }

    localEidRaw = rankAddr->eidData;
    SHM_LOG_INFO(
        "Get local eid route success, myLocalId: " << myLocalId << ", peerLocalId: " << peerLocalId << ", localPort: "
                                                   << localPort << ", localEidIndex: " << localEidIndex);
    return true;
}

bool TopoReader::GetLocalId(const RootInfo& root, uint32_t deviceId, uint32_t& localId)
{
    if (!root.eidIndexToRankAddr.empty() && root.deviceId == deviceId) {
        localId = root.localId;
        return true;
    }
    SHM_LOG_ERROR("RootInfo is invalid or incomplete, failed to find localId for deviceId " << deviceId);
    return false;
}

bool TopoReader::GetEidCount(const RootInfo& root, uint32_t& count)
{
    if (root.totalEidCount > 0) {
        count = root.totalEidCount;
        return true;
    }
    SHM_LOG_ERROR("RootInfo is invalid or incomplete, failed to find a valid eid count.");
    return false;
}

bool TopoReader::ParseRdmaNetAddr(uint32_t phyId, net_addr_t& outIp)
{
    nlohmann::json rootInfoJson;
    if (!LoadRootInfoJson(phyId, rootInfoJson)) {
        SHM_LOG_ERROR("Failed to load rootinfo for phyId " << phyId);
        return false;
    }

    if (!rootInfoJson.contains("rank_list") || !rootInfoJson["rank_list"].is_array()) {
        SHM_LOG_ERROR("Rootinfo: missing or invalid rank_list, phyId " << phyId);
        return false;
    }

    bool found = false;
    for (const auto& rankJson : rootInfoJson["rank_list"]) {
        if (!rankJson.contains("device_id") || !rankJson.contains("local_id")) {
            continue;
        }
        uint32_t rankDeviceId = 0;
        if (!ParseUint(rankJson["device_id"], rankDeviceId) || rankDeviceId != phyId) {
            continue;
        }

        if (!rankJson.contains("level_list") || !rankJson["level_list"].is_array() || rankJson["level_list"].empty()) {
            SHM_LOG_WARN("Rootinfo: missing level_list for phyId " << phyId);
            return false;
        }

        for (const auto& levelJson : rankJson["level_list"]) {
            if (!levelJson.contains("net_type") || !levelJson["net_type"].is_string()) {
                continue;
            }
            if (levelJson["net_type"].get<std::string>() != "CLOS") {
                continue;
            }
            if (!levelJson.contains("rank_addr_list") || !levelJson["rank_addr_list"].is_array()) {
                SHM_LOG_WARN("Rootinfo: missing rank_addr_list for phyId " << phyId);
                return false;
            }

            for (const auto& rankAddrJson : levelJson["rank_addr_list"]) {
                if (!rankAddrJson.contains("addr_type")) {
                    continue;
                }
                if (!rankAddrJson["addr_type"].is_string()) {
                    SHM_LOG_ERROR("Rootinfo addr_type format is unsupported, phyId " << phyId);
                    return false;
                }
                if (!rankAddrJson.contains("addr") || !rankAddrJson["addr"].is_string()) {
                    SHM_LOG_WARN("Rootinfo: missing or invalid addr field in CLOS entry, phyId " << phyId);
                    continue;
                }
                std::string addrType = rankAddrJson["addr_type"].get<std::string>();
                std::string addrStr = rankAddrJson["addr"].get<std::string>();
                if (addrType == "IPV4") {
                    if (inet_pton(AF_INET, addrStr.c_str(), &outIp.ip.ipv4) != 1) {
                        SHM_LOG_WARN(
                            "Rootinfo: invalid IPv4 addr in CLOS entry, phyId " << phyId << ", addr " << addrStr);
                        continue;
                    }
                    outIp.type = IpV4;
                    found = true;
                } else if (addrType == "IPV6") {
                    if (inet_pton(AF_INET6, addrStr.c_str(), &outIp.ip.ipv6) != 1) {
                        SHM_LOG_WARN(
                            "Rootinfo: invalid IPv6 addr in CLOS entry, phyId " << phyId << ", addr " << addrStr);
                        continue;
                    }
                    outIp.type = IpV6;
                    found = true;
                }
            }
        }
        break;
    }

    if (!found) {
        SHM_LOG_ERROR("Rootinfo: no valid CLOS address found for phyId " << phyId);
    }
    return found;
}

bool EidConverter::Convert(RankAddrType addrType, const nlohmann::json& addr, EidData& raw)
{
    bool converted = false;
    switch (addrType) {
        case RankAddrType::EID:
            converted = FromPlainEid(addrType, addr, raw);
            break;
        case RankAddrType::IPV4:
            converted = FromIpv4(addrType, addr, raw);
            break;
        case RankAddrType::IPV6:
            converted = FromIpv6(addrType, addr, raw);
            break;
    }
    if (!converted) {
        SHM_LOG_ERROR("Rootinfo: failed to parse addr as " << AddrTypeName(addrType));
        return false;
    }
    return true;
}

bool EidConverter::FromPlainEid(RankAddrType addrType, const nlohmann::json& addr, EidData& raw)
{
    raw.fill(0);
    if (addr.is_array()) {
        if (addr.size() != raw.size()) {
            SHM_LOG_ERROR("Rootinfo " << AddrTypeName(addrType) << " addr array size is invalid: " << addr.size());
            return false;
        }
        for (size_t i = 0; i < raw.size(); ++i) {
            if (!addr[i].is_number_unsigned() && !addr[i].is_number_integer()) {
                SHM_LOG_ERROR("Rootinfo addr array contains non-numeric value at index " << i);
                return false;
            }
            auto value = addr[i].get<int32_t>();
            if (value < 0 || value > 0xff) {
                SHM_LOG_ERROR("Rootinfo addr array value out of byte range at index " << i << ", value " << value);
                return false;
            }
            raw[i] = static_cast<uint8_t>(value);
        }
        return true;
    }

    if (!addr.is_string()) {
        SHM_LOG_ERROR("Rootinfo addr format is unsupported.");
        return false;
    }

    const std::string& jsonString = addr.get_ref<const std::string&>();
    std::string normalized;
    normalized.reserve(jsonString.size());
    for (const char ch : jsonString) {
        if (std::isxdigit(static_cast<unsigned char>(ch))) {
            normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
    }

    // Tolerate odd-length hex by left-padding to the full EID width.
    if (normalized.size() < URMA_EID_HEX_SIZE) {
        normalized.insert(normalized.begin(), URMA_EID_HEX_SIZE - normalized.size(), '0');
    }
    if (normalized.size() != URMA_EID_HEX_SIZE) {
        SHM_LOG_ERROR("Rootinfo addr hex length is invalid: " << normalized.size());
        return false;
    }

    try {
        for (size_t i = 0; i < raw.size(); ++i) {
            const std::string byteStr = normalized.substr(i * 2, 2);
            raw[i] = static_cast<uint8_t>(std::stoul(byteStr, nullptr, 16));
        }
    } catch (const std::exception& ex) {
        SHM_LOG_ERROR("Failed to parse rootinfo addr hex string, error: " << ex.what());
        return false;
    }
    return true;
}

bool EidConverter::FromIpv4(RankAddrType addrType, const nlohmann::json& addr, EidData& raw)
{
    if (!addr.is_string()) {
        SHM_LOG_ERROR("Rootinfo " << AddrTypeName(addrType) << " addr must be a string.");
        return false;
    }
    const std::string& addrStr = addr.get_ref<const std::string&>();

    in_addr ipv4Addr{};
    if (inet_pton(AF_INET, addrStr.c_str(), &ipv4Addr) != 1) {
        SHM_LOG_ERROR("Rootinfo IPV4 addr is invalid: " << addrStr);
        return false;
    }

    raw.fill(0);
    raw[10] = static_cast<uint8_t>((URMA_EID_IPV4_PREFIX >> 8) & 0xff);
    raw[11] = static_cast<uint8_t>(URMA_EID_IPV4_PREFIX & 0xff);

    const auto* ipv4Bytes = reinterpret_cast<const uint8_t*>(&ipv4Addr.s_addr);
    std::copy_n(ipv4Bytes, sizeof(ipv4Addr), raw.begin() + 12);
    SHM_LOG_INFO("Rootinfo IPV4 addr converted to EID, addr " << addrStr);
    return true;
}

bool EidConverter::FromIpv6(RankAddrType addrType, const nlohmann::json& addr, EidData& raw)
{
    if (!addr.is_string()) {
        SHM_LOG_ERROR("Rootinfo " << AddrTypeName(addrType) << " addr must be a string.");
        return false;
    }
    const std::string& addrStr = addr.get_ref<const std::string&>();

    raw.fill(0);
    if (inet_pton(AF_INET6, addrStr.c_str(), raw.data()) != 1) {
        SHM_LOG_ERROR("Rootinfo IPV6 addr is invalid: " << addrStr);
        return false;
    }
    SHM_LOG_INFO("Rootinfo IPV6 addr converted to EID, addr " << addrStr);
    return true;
}

bool TopoReader::ParseUint(const nlohmann::json& jsonValue, uint32_t& value)
{
    if (!TryGet(jsonValue, value)) {
        SHM_LOG_ERROR("Failed to parse uint value.");
        return false;
    }
    return true;
}

namespace {

std::string NetInstanceIdAt(
    const std::vector<std::vector<SyncEndpoint>>& rankIdxToSyncEndpoint, uint32_t rank, uint32_t netLayer)
{
    if (rank >= rankIdxToSyncEndpoint.size()) {
        return {};
    }
    for (const auto& endpoint : rankIdxToSyncEndpoint[rank]) {
        if (endpoint.netLayer == netLayer) {
            return endpoint.netInstanceId;
        }
    }
    return {};
}

bool ResolveEidIndexByPort(
    const std::vector<SyncEndpoint>& endpoints, const std::string& port, uint32_t netLayer, uint32_t& eidIndex)
{
    for (const auto& endpoint : endpoints) {
        if (endpoint.netLayer != netLayer) {
            continue;
        }
        for (const auto& ownedPort : endpoint.ports) {
            if (ownedPort == port) {
                eidIndex = endpoint.eidIndex;
                return true;
            }
        }
    }
    return false;
}

bool ResolveLocalEidByPort(
    const RootInfo& root, const std::string& localPort, uint32_t& localEidIndex, EidData& localEidRaw)
{
    const auto portItem = root.portsToRankAddr.find(localPort);
    if (portItem == root.portsToRankAddr.end() || portItem->second == nullptr) {
        SHM_LOG_ERROR("Failed to get local eid index, port " << localPort << " not found.");
        return false;
    }
    const auto& rankAddr = portItem->second;

    for (const auto& item : root.eidIndexToRankAddr) {
        if (item.second == rankAddr) {
            localEidIndex = item.first;
            localEidRaw = rankAddr->eidData;
            return true;
        }
    }
    SHM_LOG_ERROR("Failed to get local eid index, rank addr for port " << localPort << " is not indexed.");
    return false;
}

bool ResolveLocalEidByIndex(const RootInfo& root, uint32_t eidIndex, EidData& localEidRaw)
{
    const auto item = root.eidIndexToRankAddr.find(eidIndex);
    if (item == root.eidIndexToRankAddr.end() || item->second == nullptr) {
        SHM_LOG_ERROR("Failed to get local eid raw, eidIndex " << eidIndex << " not indexed.");
        return false;
    }
    localEidRaw = item->second->eidData;
    return true;
}

} // namespace

bool TopoQuerier::GetEidRouteMesh1D(
    uint32_t targetRank, uint32_t& localEidIndex, EidData& localEidRaw, uint32_t& remoteEidIndex)
{
    constexpr uint32_t MESH_NET_LAYER = 0;
    const std::string myInstance = NetInstanceIdAt(rankIdxToSyncEndpoint_, myRank_, MESH_NET_LAYER);
    const std::string targetInstance = NetInstanceIdAt(rankIdxToSyncEndpoint_, targetRank, MESH_NET_LAYER);
    if (myInstance.empty() || myInstance != targetInstance) {
        SHM_LOG_DEBUG(
            "Mesh1D route skipped, layer0 net_instance mismatch for rank "
            << myRank_ << " (\"" << myInstance << "\") and target rank " << targetRank << " (\"" << targetInstance
            << "\").");
        return false;
    }

    if (myRank_ >= rankToLocalId_.size() || targetRank >= rankToLocalId_.size()) {
        SHM_LOG_ERROR("Mesh1D route failed, rank out of range, myRank " << myRank_ << ", targetRank " << targetRank);
        return false;
    }
    const uint32_t myLocalId = rankToLocalId_[myRank_];
    const uint32_t peerLocalId = rankToLocalId_[targetRank];

    std::string localPort;
    std::string remotePort;
    for (const auto& edge : topoInfo_.meshTopoEdges) {
        if (edge.localA == myLocalId && edge.localB == peerLocalId && !edge.localAPorts.empty() &&
            !edge.localBPorts.empty()) {
            localPort = edge.localAPorts[0];
            remotePort = edge.localBPorts[0];
            break;
        }
        if (edge.localB == myLocalId && edge.localA == peerLocalId && !edge.localBPorts.empty() &&
            !edge.localAPorts.empty()) {
            localPort = edge.localBPorts[0];
            remotePort = edge.localAPorts[0];
            break;
        }
    }
    if (localPort.empty() || remotePort.empty()) {
        SHM_LOG_DEBUG(
            "Mesh1D route not found, no direct mesh link between localId " << myLocalId << " and " << peerLocalId);
        return false;
    }

    if (!ResolveLocalEidByPort(rootInfo_, localPort, localEidIndex, localEidRaw)) {
        return false;
    }
    if (targetRank >= rankIdxToSyncEndpoint_.size() ||
        !ResolveEidIndexByPort(rankIdxToSyncEndpoint_[targetRank], remotePort, MESH_NET_LAYER, remoteEidIndex)) {
        SHM_LOG_ERROR(
            "Mesh1D route failed, peer port " << remotePort << " not found in target rank " << targetRank
                                              << " endpoints.");
        return false;
    }
    SHM_LOG_INFO(
        "Mesh1D route success, myRank " << myRank_ << ", targetRank " << targetRank << ", localPort " << localPort
                                        << ", localEidIndex " << localEidIndex << ", remotePort " << remotePort
                                        << ", remoteEidIndex " << remoteEidIndex);
    return true;
}

bool TopoQuerier::GetEidRouteClos(
    uint32_t targetRank, uint32_t& localEidIndex, EidData& localEidRaw, uint32_t& remoteEidIndex)
{
    if (myRank_ >= rankIdxToSyncEndpoint_.size() || targetRank >= rankIdxToSyncEndpoint_.size()) {
        SHM_LOG_ERROR("Clos route failed, rank out of range, myRank " << myRank_ << ", targetRank " << targetRank);
        return false;
    }
    const auto& myEndpoints = rankIdxToSyncEndpoint_[myRank_];
    const auto& peerEndpoints = rankIdxToSyncEndpoint_[targetRank];

    for (const auto& mine : myEndpoints) {
        for (const auto& peer : peerEndpoints) {
            if (mine.netLayer != peer.netLayer || mine.netInstanceId != peer.netInstanceId ||
                mine.planeId != peer.planeId || mine.netInstanceId.empty()) {
                continue;
            }
            if (!ResolveLocalEidByIndex(rootInfo_, mine.eidIndex, localEidRaw)) {
                return false;
            }
            localEidIndex = mine.eidIndex;
            remoteEidIndex = peer.eidIndex;
            SHM_LOG_INFO(
                "Clos route success, myRank "
                << myRank_ << ", targetRank " << targetRank << ", netLayer " << mine.netLayer << ", netInstanceId \""
                << mine.netInstanceId << "\", planeId \"" << mine.planeId << "\", localEidIndex " << localEidIndex
                << ", remoteEidIndex " << remoteEidIndex);
            return true;
        }
    }

    SHM_LOG_DEBUG("Clos route not found, no shared plane between rank " << myRank_ << " and " << targetRank);
    return false;
}

bool TopoQuerier::GetEidRoute(
    uint32_t targetRank, uint32_t& localEidIndex, EidData& localEidRaw, uint32_t& remoteEidIndex)
{
    return GetEidRouteMesh1D(targetRank, localEidIndex, localEidRaw, remoteEidIndex) ||
           GetEidRouteClos(targetRank, localEidIndex, localEidRaw, remoteEidIndex);
}

} // namespace transport
} // namespace shm
