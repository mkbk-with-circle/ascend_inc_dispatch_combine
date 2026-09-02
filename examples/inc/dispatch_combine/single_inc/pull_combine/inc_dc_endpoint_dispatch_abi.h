#ifndef INC_DC_ENDPOINT_DISPATCH_ABI_H
#define INC_DC_ENDPOINT_DISPATCH_ABI_H

#include <cstdint>

namespace inc::dc::pull_combine {

constexpr uint32_t kEndpointDispatchMagic = 0x44503245u; // 'DP2E'
constexpr uint16_t kEndpointDispatchAbiVersion = 1u;
constexpr uint32_t kEndpointDispatchAlignment = 64u;

enum class EndpointDataType : uint32_t {
    FP16 = 0u,
    BF16 = 1u,
    FP32 = 2u,
};

// The worker writes the complete packet before publishing its separate
// cache-line commit.  Offsets are relative to the beginning of this header.
struct alignas(64) EndpointDispatchPacketHeader {
    uint32_t magic = kEndpointDispatchMagic;
    uint16_t abi_version = kEndpointDispatchAbiVersion;
    uint16_t header_bytes = sizeof(EndpointDispatchPacketHeader);
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint32_t wave = 0u;
    uint32_t source_rank = 0u;
    uint32_t worker_count = 0u;
    uint32_t token_count = 0u;
    uint32_t hidden = 0u;
    uint32_t dtype = static_cast<uint32_t>(EndpointDataType::BF16);
    uint32_t assignment_count = 0u;
    uint32_t flags = 0u;
    uint64_t counts_offset = 0u;
    uint64_t tokens_offset = 0u;
    uint64_t assignments_offset = 0u;
    uint64_t hidden_offset = 0u;
    uint64_t packet_bytes = 0u;
    uint64_t metadata_digest = 0u;
    uint64_t reserved[3]{};
};
static_assert(sizeof(EndpointDispatchPacketHeader) == 128u,
              "endpoint Dispatch header ABI drift");

// Hidden row i is at hidden_offset + source_token * hidden * dtype_bytes.
struct EndpointDispatchTokenRecord {
    uint64_t token_id = 0u;
    uint32_t source_token = 0u;
    uint32_t assignment_begin = 0u;
    uint32_t assignment_count = 0u;
    uint32_t reserved0 = 0u;
    uint64_t reserved1 = 0u;
};
static_assert(sizeof(EndpointDispatchTokenRecord) == 32u,
              "endpoint token record ABI drift");

struct EndpointDispatchAssignmentRecord {
    uint32_t destination_rank = 0u;
    uint32_t expert_id = 0u;
    uint32_t ordinal = 0u;
    float weight = 1.0f;
};
static_assert(sizeof(EndpointDispatchAssignmentRecord) == 16u,
              "endpoint assignment record ABI drift");

// Publication is the only ready flag polled by the INC.  It is written after
// the packet and count vectors are visible in the INC-owned ring slot.
struct alignas(64) EndpointDispatchCommit {
    uint32_t magic = kEndpointDispatchMagic;
    uint16_t abi_version = kEndpointDispatchAbiVersion;
    uint16_t struct_bytes = sizeof(EndpointDispatchCommit);
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint32_t wave = 0u;
    uint32_t source_rank = 0u;
    uint32_t slot = 0u;
    uint32_t flags = 0u;
    uint64_t packet_bytes = 0u;
    uint64_t metadata_digest = 0u;
    uint64_t reserved = 0u;
};
static_assert(sizeof(EndpointDispatchCommit) == 64u,
              "endpoint Dispatch commit must own one cache line");

} // namespace inc::dc::pull_combine

#endif
