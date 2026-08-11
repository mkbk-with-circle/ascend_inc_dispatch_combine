#ifndef INC_DC_PACKET_H
#define INC_DC_PACKET_H

#include <cstdint>

#include "inc_dc_types.h"

namespace inc {
namespace dc {

constexpr uint32_t kDcPacketMagic = 0x4443504Bu; // 'DCPK'
constexpr uint32_t kDcPacketHdrBytes = 64;
constexpr uint32_t kDcMaxTokenBlock = 32;
constexpr uint32_t kDcMaxAssignmentsPerToken = 16;

// 设备/主机共享 packet header（64B 对齐）
struct IncDcPacketHeader {
    uint32_t magic = kDcPacketMagic;
    uint32_t version = kPacketVersion;
    uint64_t op_seq = 0;
    uint64_t route_epoch = 0;
    uint32_t group_id = 0;
    uint32_t group_size = 0;
    int32_t source_rank = -1;
    int32_t source_token = -1;
    uint32_t destination_mask = 0; // 低位 group_size
    uint32_t hidden_bytes = 0;
    uint32_t metadata_bytes = 0;
    uint32_t flags = 0; // EMPTY_FIRST=1, FIRST_DATA=2
    uint32_t num_assignments = 0;
    uint32_t crc = 0;
};

// per-assignment metadata 紧随 hidden payload
struct IncDcAssignmentPkt {
    int32_t expert_id = -1;
    int32_t route_slot = 0;
    float weight = 1.f;
    int32_t reserved = 0;
};

static_assert(sizeof(IncDcPacketHeader) == kDcPacketHdrBytes, "IncDcPacketHeader must be 64B");

enum IncDcPacketFlags : uint32_t {
    DC_PKT_EMPTY_FIRST = 1u,
    DC_PKT_FIRST_DATA = 2u,
};

} // namespace dc
} // namespace inc

#endif
