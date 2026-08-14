/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBRID_TRANSPORT_TOPO_READER_H
#define MF_HYBRID_TRANSPORT_TOPO_READER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "sotre_net.h"

namespace shm {
namespace transport {

constexpr std::size_t URMA_EID_RAW_SIZE = 16;
constexpr std::size_t URMA_EID_HEX_SIZE = URMA_EID_RAW_SIZE * 2;
constexpr uint32_t URMA_EID_IPV4_PREFIX = 0x0000;

using EidData = std::array<uint8_t, URMA_EID_RAW_SIZE>;
using EidPort = std::string;
using EidIndex = uint32_t;

enum class NetType { Mesh = 0, Clos = 1, TopoFileDesc = 2 };

enum class RankAddrType { EID = 0, IPV4 = 1, IPV6 = 2 };

struct LevelInfo {
    uint32_t netLayer{};
    std::string netInstanceId{};
    NetType netType{};
    std::string netAttr{};
};

struct RankAddr {
    RankAddrType addrType{};
    EidData eidData{};
    std::string planeId{};

    std::vector<EidPort> ports{};
    std::shared_ptr<LevelInfo> levelInfo{};
};

struct RootInfo {
    std::string topoFilePath{};
    uint32_t deviceId{};
    uint32_t localId{};
    uint32_t totalEidCount{};

    std::vector<std::shared_ptr<LevelInfo>> levels{};
    std::map<EidIndex, std::shared_ptr<RankAddr>> eidIndexToRankAddr{};
    std::map<EidPort, std::shared_ptr<RankAddr>> portsToRankAddr{};
};

struct MeshTopoEdge {
    uint32_t localA{};
    uint32_t localB{};
    std::vector<EidPort> localAPorts{};
    std::vector<EidPort> localBPorts{};
};

struct ClosTopoEdge {
    uint32_t netLayer{};
    uint32_t topoInstanceId{};

    uint32_t localA{};
    std::vector<EidPort> ports{};
};

struct TopoInfo {
    std::vector<MeshTopoEdge> meshTopoEdges{};
    std::vector<ClosTopoEdge> closTopoEdges{};
};

// Per-rank-addr view exchanged across ranks during PrepareOpenDevice. One SyncEndpoint
// is emitted for every rank_addr on every level (including netLayer 0). Two ranks are
// peers on a given plane when their (netLayer, netInstanceId, planeId) all match; the
// peer's eidIndex/ports then identify the remote route. eidData is intentionally NOT
// carried: the local eid raw is fetched from the local rootInfo by eidIndex.
struct SyncEndpoint {
    uint32_t netLayer{};
    std::string netInstanceId{};
    std::string planeId{};
    std::vector<EidPort> ports{}; // used to reverse-map a peer port to its eidIndex
    uint32_t eidIndex{};

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(SyncEndpoint, netLayer, netInstanceId, planeId, ports, eidIndex)
};

class EidConverter {
public:
    static bool Convert(RankAddrType addrType, const nlohmann::json& addr, EidData& raw);

private:
    static bool FromPlainEid(RankAddrType addrType, const nlohmann::json& addr, EidData& raw);
    static bool FromIpv4(RankAddrType addrType, const nlohmann::json& addr, EidData& raw);
    static bool FromIpv6(RankAddrType addrType, const nlohmann::json& addr, EidData& raw);
};

class TopoReader {
public:
    static constexpr const char* ROOTINFO_PATH = "/etc/hccl_rootinfo.json";
    static bool ParseRootInfo(uint32_t phyId, RootInfo& out);
    static bool ParseTopoInfo(const std::string& path, TopoInfo& out);
    static bool GetLocalEidRouteForPeer(
        const RootInfo& root, const TopoInfo& topo, uint32_t myLocalId, uint32_t peerLocalId, uint32_t& localEidIndex,
        EidData& localEidRaw);
    static bool GetLocalId(const RootInfo& root, uint32_t deviceId, uint32_t& localId);

    static bool GetEidCount(const RootInfo& root, uint32_t& count);

    static bool ParseRdmaNetAddr(uint32_t phyId, net_addr_t& outIp);

private:
    static bool LoadRootInfoJson(uint32_t phyId, nlohmann::json& out);
    static bool ParseRootInfoJson(const nlohmann::json& rootInfoJson, uint32_t phyId, RootInfo& out);
    static bool ParseLevelInfo(const nlohmann::json& levelJson, LevelInfo& out);
    static bool ParseRankAddr(
        const nlohmann::json& rankAddrJson, uint32_t localId, uint32_t eidIndex,
        const std::shared_ptr<LevelInfo>& levelInfo, RankAddr& out);
    static bool ParseUint(const nlohmann::json& jsonValue, uint32_t& value);
};

class TopoQuerier {
public:
    explicit TopoQuerier(
        const RootInfo& r, const TopoInfo& t, uint32_t myRank, const std::vector<uint32_t>& rankToLocalId,
        const std::vector<std::vector<SyncEndpoint>>& rankIdxToSyncEndpoint)
        : rootInfo_(r),
          topoInfo_(t),
          myRank_(myRank),
          rankToLocalId_(rankToLocalId),
          rankIdxToSyncEndpoint_(rankIdxToSyncEndpoint)
    {}

    bool GetEidRouteMesh1D(
        uint32_t targetRank, uint32_t& localEidIndex, EidData& localEidRaw, uint32_t& remoteEidIndex);
    bool GetEidRouteClos(uint32_t targetRank, uint32_t& localEidIndex, EidData& localEidRaw, uint32_t& remoteEidIndex);

    bool GetEidRoute(uint32_t targetRank, uint32_t& localEidIndex, EidData& localEidRaw, uint32_t& remoteEidIndex);

private:
    const RootInfo& rootInfo_;
    const TopoInfo& topoInfo_;
    uint32_t myRank_;

    const std::vector<uint32_t>& rankToLocalId_;
    const std::vector<std::vector<SyncEndpoint>>& rankIdxToSyncEndpoint_;
};

} // namespace transport
} // namespace shm

#endif // MF_HYBRID_TRANSPORT_TOPO_READER_H
