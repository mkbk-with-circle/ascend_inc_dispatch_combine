#ifndef INC_DC_ENDPOINT_DISPATCH_PACKET_H
#define INC_DC_ENDPOINT_DISPATCH_PACKET_H

#include <cstdint>
#include <string>
#include <vector>

#include "inc_dc_endpoint_dispatch_abi.h"

namespace inc::dc::pull_combine {

enum class EndpointDispatchStatus : uint32_t {
    OK = 0u,
    INVALID_ARGUMENT,
    INVALID_HEADER,
    STALE_GENERATION,
    SEQUENCE_MISMATCH,
    OUT_OF_RANGE,
    SIZE_OVERFLOW,
    PAYLOAD_SIZE_MISMATCH,
    COUNT_MISMATCH,
    DUPLICATE_TOKEN,
    DUPLICATE_ORDINAL,
    NONFINITE_WEIGHT,
    DIGEST_MISMATCH,
};

struct EndpointDispatchConfig {
    uint32_t worker_count = 0u;
    uint32_t expert_count = 0u;
    uint32_t hidden = 0u;
    EndpointDataType dtype = EndpointDataType::BF16;
    uint32_t wave = 0u;
    uint32_t source_rank = 0u;
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
};

// assignment_offsets is CSR with token_ids.size()+1 entries.  This is the
// internal variable-top-k form; a fixed-top-k API synthesizes i*topk offsets.
struct EndpointDispatchInput {
    EndpointDispatchConfig config{};
    std::vector<uint64_t> token_ids;
    std::vector<uint32_t> assignment_offsets;
    std::vector<EndpointDispatchAssignmentRecord> assignments;
    std::vector<uint8_t> hidden_payload;
};

struct EndpointFanoutRow {
    uint64_t token_id = 0u;
    uint32_t source_rank = 0u;
    uint32_t source_token = 0u;
    std::vector<EndpointDispatchAssignmentRecord> assignments;
    std::vector<uint8_t> hidden;
};

struct ParsedEndpointDispatch {
    EndpointDispatchPacketHeader header{};
    std::vector<uint32_t> token_counts;
    std::vector<uint32_t> assignment_counts;
    std::vector<std::vector<EndpointFanoutRow>> destination_rows;
};

EndpointDispatchStatus BuildEndpointDispatchPacket(
    const EndpointDispatchInput &input, std::vector<uint8_t> *packet,
    EndpointDispatchCommit *commit, std::string *error = nullptr);

EndpointDispatchStatus ParseEndpointDispatchPacket(
    const uint8_t *packet, uint64_t packet_bytes,
    const EndpointDispatchConfig &expected, ParsedEndpointDispatch *parsed,
    std::string *error = nullptr);

const char *EndpointDispatchStatusString(EndpointDispatchStatus status);

} // namespace inc::dc::pull_combine

#endif
