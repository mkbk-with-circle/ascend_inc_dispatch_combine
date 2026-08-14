/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHMEM_HCOMM_ENTITY_COMPAT_H
#define SHMEM_HCOMM_ENTITY_COMPAT_H

#include <cstring>
#include <netinet/in.h>

#include <cstddef>
#include <cstdint>

/*
 * Do not include CANN HCOMM headers here. HCOMM is loaded with dlopen/dlsym,
 * so host compilation uses the ABI-shaped declarations below, copied from
 * hcomm-master/include/hcomm_res_defs.h.
 */

constexpr uint32_t COMM_ADDR_EID_LEN = 16U;
constexpr uint32_t HCOMM_CHANNEL_MAGIC_WORD = 0x0fcf0f0fU;
constexpr uint32_t HCOMM_CHANNEL_VERSION_ONE = 1U;
constexpr uint32_t HCOMM_CHANNEL_VERSION = 3U;
constexpr uint32_t HCOMM_CHANNEL_NAME_MAX_LEN = 191U;

enum class CommEngine : int32_t {
    RESERVED = -1,
    CPU = 0,
    CPU_TS = 1,
    AICPU = 2,
    AICPU_TS = 3,
    AIV = 4,
    CCU = 5,
};
inline constexpr CommEngine COMM_ENGINE_RESERVED = CommEngine::RESERVED;
inline constexpr CommEngine COMM_ENGINE_CPU = CommEngine::CPU;
inline constexpr CommEngine COMM_ENGINE_CPU_TS = CommEngine::CPU_TS;
inline constexpr CommEngine COMM_ENGINE_AICPU = CommEngine::AICPU;
inline constexpr CommEngine COMM_ENGINE_AICPU_TS = CommEngine::AICPU_TS;
inline constexpr CommEngine COMM_ENGINE_AIV = CommEngine::AIV;
inline constexpr CommEngine COMM_ENGINE_CCU = CommEngine::CCU;

enum class CommProtocol : int32_t {
    RESERVED = -1,
    HCCS = 0,
    ROCE = 1,
    PCIE = 2,
    SIO = 3,
    UBC_CTP = 4,
    UBC_TP = 5,
    UB_MEM = 6,
    UBOE = 7,
    HCCS_ONLY = 8,
    UBG = 9,
};
inline constexpr CommProtocol COMM_PROTOCOL_RESERVED = CommProtocol::RESERVED;
inline constexpr CommProtocol COMM_PROTOCOL_HCCS = CommProtocol::HCCS;
inline constexpr CommProtocol COMM_PROTOCOL_ROCE = CommProtocol::ROCE;
inline constexpr CommProtocol COMM_PROTOCOL_PCIE = CommProtocol::PCIE;
inline constexpr CommProtocol COMM_PROTOCOL_SIO = CommProtocol::SIO;
inline constexpr CommProtocol COMM_PROTOCOL_UBC_CTP = CommProtocol::UBC_CTP;
inline constexpr CommProtocol COMM_PROTOCOL_UBC_TP = CommProtocol::UBC_TP;
inline constexpr CommProtocol COMM_PROTOCOL_UB_MEM = CommProtocol::UB_MEM;
inline constexpr CommProtocol COMM_PROTOCOL_UBOE = CommProtocol::UBOE;
inline constexpr CommProtocol COMM_PROTOCOL_HCCS_ONLY = CommProtocol::HCCS_ONLY;
inline constexpr CommProtocol COMM_PROTOCOL_UBG = CommProtocol::UBG;

enum class CommAddrType : int32_t {
    RESERVED = -1,
    IP_V4 = 0,
    IP_V6 = 1,
    ID = 2,
    EID = 3,
};
inline constexpr CommAddrType COMM_ADDR_TYPE_RESERVED = CommAddrType::RESERVED;
inline constexpr CommAddrType COMM_ADDR_TYPE_IP_V4 = CommAddrType::IP_V4;
inline constexpr CommAddrType COMM_ADDR_TYPE_IP_V6 = CommAddrType::IP_V6;
inline constexpr CommAddrType COMM_ADDR_TYPE_ID = CommAddrType::ID;
inline constexpr CommAddrType COMM_ADDR_TYPE_EID = CommAddrType::EID;

struct CommAddr {
    CommAddrType type;
    union {
        uint8_t raws[36];
        struct in_addr addr;
        struct in6_addr addr6;
        uint32_t id;
        uint8_t eid[16];
    };
};

enum class EndpointLocType : int32_t {
    RESERVED = -1,
    DEVICE = 0,
    HOST = 1,
};
inline constexpr EndpointLocType ENDPOINT_LOC_TYPE_RESERVED = EndpointLocType::RESERVED;
inline constexpr EndpointLocType ENDPOINT_LOC_TYPE_DEVICE = EndpointLocType::DEVICE;
inline constexpr EndpointLocType ENDPOINT_LOC_TYPE_HOST = EndpointLocType::HOST;

struct EndpointLoc {
    EndpointLocType locType;
    union {
        uint8_t raws[60];
        struct {
            uint32_t devPhyId;
            uint32_t superDevId;
            uint32_t serverIdx;
            uint32_t superPodIdx;
        } device;
        struct {
            uint32_t id;
        } host;
    };
};
struct EndpointDesc {
    CommProtocol protocol;
    CommAddr commAddr;
    EndpointLoc loc;
    union {
        uint8_t raws[52];
    };
};
using ChannelHandle = uint64_t;
using HcclResult = int32_t;
constexpr HcclResult HCCL_SUCCESS = 0;
constexpr HcclResult HCCL_E_PTR = 2;

inline HcclResult EndpointDescInit(EndpointDesc* endpoint, uint32_t num)
{
    if (endpoint == nullptr) {
        return HCCL_E_PTR;
    }
    for (uint32_t idx = 0; idx < num; ++idx) {
        std::memset(endpoint, 0, sizeof(EndpointDesc));
        endpoint->protocol = COMM_PROTOCOL_RESERVED;
        endpoint->commAddr.type = COMM_ADDR_TYPE_RESERVED;
        endpoint->loc.locType = ENDPOINT_LOC_TYPE_RESERVED;
        ++endpoint;
    }
    return HCCL_SUCCESS;
}

using EndpointHandle = void*;
using HcommMemHandle = void*;
using HcommResult = int32_t;
using HcommSocket = void*;

enum class CommMemType : int32_t {
    INVALID = -1,
    DEVICE = 0,
    HOST = 1,
};
struct CommMem {
    CommMemType type;
    void* addr;
    uint64_t size;
};

struct CommAbiHeader {
    uint32_t version;
    uint32_t magicWord;
    uint32_t size;
    uint32_t reserved;
};

enum class HcommSocketRole : int32_t {
    RESERVED = -1,
    CLIENT = 0,
    SERVER = 1,
};
inline constexpr HcommSocketRole HCOMM_SOCKET_ROLE_RESERVED = HcommSocketRole::RESERVED;
inline constexpr HcommSocketRole HCOMM_SOCKET_ROLE_CLIENT = HcommSocketRole::CLIENT;
inline constexpr HcommSocketRole HCOMM_SOCKET_ROLE_SERVER = HcommSocketRole::SERVER;

struct HcommChannelDesc {
    CommAbiHeader header{};
    EndpointDesc remoteEndpoint{};
    uint32_t notifyNum{0};
    bool exchangeAllMems{false};
    HcommMemHandle* memHandles{nullptr};
    uint32_t memHandleNum{0};
    HcommSocket socket{nullptr};
    HcommSocketRole role{HcommSocketRole::RESERVED};
    uint16_t port{0};
    union {
        uint8_t raws[128];
        struct {
            uint32_t queueNum;
            uint32_t retryCnt;
            uint32_t retryInterval;
            uint8_t tc;
            uint8_t sl;
            uint32_t qpThreshold;
        } roceAttr;
        struct {
            uint32_t qos;
        } hccsAttr;
        struct {
            uint32_t sqDepth;
        } ubAttr;
    };
    uint32_t qos{0};
    const char* channelName{nullptr};
};

#define HCOMM_CHANNEL_DESC_ABI_V1_SIZE (offsetof(HcommChannelDesc, qos))

using HcommMem = CommMem;

// Keep the HCOM memory-registration ABI used by SHMEM independent from the
// CANN header spelling. The runtime implementation is loaded by dlsym and
// follows the pointer-form ABI used by CANN 9.0.0 and later.
namespace shm {
using HcommResult = int32_t;
using HcommMemHandle = void*;
enum class CommMemType : int32_t {
    INVALID = -1,
    DEVICE = 0,
    HOST = 1,
};
struct CommMem {
    CommMemType type;
    void* addr;
    uint64_t size;
};

namespace detail {

template <typename Desc>
auto InitializeHcommChannelDesc(Desc* channelDesc, uint32_t num, int)
    -> decltype(HcommChannelDescInit(channelDesc, num))
{
    return HcommChannelDescInit(channelDesc, num);
}

template <typename Desc>
HcommResult InitializeHcommChannelDesc(Desc* channelDesc, uint32_t num, long)
{
    if (channelDesc == nullptr) {
        return 2;
    }
    for (uint32_t i = 0; i < num; ++i) {
        std::memset(channelDesc, 0xFF, sizeof(HcommChannelDesc));
        channelDesc->header.version = HCOMM_CHANNEL_VERSION;
        channelDesc->header.magicWord = HCOMM_CHANNEL_MAGIC_WORD;
        channelDesc->header.size = sizeof(HcommChannelDesc);
        channelDesc->header.reserved = 0;
        channelDesc->notifyNum = 0;
        channelDesc->exchangeAllMems = false;
        channelDesc->memHandles = nullptr;
        channelDesc->memHandleNum = 0;
        channelDesc->socket = nullptr;
        channelDesc->role = HCOMM_SOCKET_ROLE_RESERVED;
        channelDesc->port = 0;
        channelDesc->remoteEndpoint.protocol = COMM_PROTOCOL_RESERVED;
        channelDesc->remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_RESERVED;
        channelDesc->remoteEndpoint.loc.locType = ENDPOINT_LOC_TYPE_RESERVED;
        channelDesc->channelName = nullptr;
        ++channelDesc;
    }
    return 0;
}

} // namespace detail

// Keep a project-scoped initializer available without colliding with CANN's
// global declaration. Use CANN's initializer whenever the visible headers
// declare it; otherwise use the ABI-shaped fallback implementation.
inline HcommResult ShmemHcommChannelDescInit(HcommChannelDesc* channelDesc, uint32_t num)
{
    return detail::InitializeHcommChannelDesc(channelDesc, num, 0);
}

} // namespace shm

#endif // SHMEM_HCOMM_ENTITY_COMPAT_H
