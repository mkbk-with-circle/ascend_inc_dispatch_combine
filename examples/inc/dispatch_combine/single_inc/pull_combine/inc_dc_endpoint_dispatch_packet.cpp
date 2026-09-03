#include "inc_dc_endpoint_dispatch_packet.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace inc::dc::pull_combine {
namespace {

bool Mul(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr || (a != 0u &&
        b > std::numeric_limits<uint64_t>::max() / a))
        return false;
    *out = a * b;
    return true;
}

bool Add(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr || b > std::numeric_limits<uint64_t>::max() - a)
        return false;
    *out = a + b;
    return true;
}

bool Align(uint64_t value, uint64_t *out)
{
    uint64_t expanded = 0u;
    if (!Add(value, kEndpointDispatchAlignment - 1u, &expanded))
        return false;
    *out = expanded / kEndpointDispatchAlignment *
        kEndpointDispatchAlignment;
    return true;
}

uint32_t DTypeBytes(EndpointDataType dtype)
{
    switch (dtype) {
        case EndpointDataType::FP16:
        case EndpointDataType::BF16: return 2u;
        case EndpointDataType::FP32: return 4u;
    }
    return 0u;
}

EndpointDispatchStatus Fail(EndpointDispatchStatus status,
                            const char *message, std::string *error)
{
    if (error != nullptr) *error = message;
    return status;
}

void HashBytes(uint64_t *hash, const void *data, uint64_t bytes)
{
    constexpr uint64_t kPrime = 1099511628211ull;
    const uint8_t *p = static_cast<const uint8_t *>(data);
    for (uint64_t i = 0u; i < bytes; ++i) {
        *hash ^= p[i];
        *hash *= kPrime;
    }
}

uint64_t MetadataDigest(const EndpointDispatchPacketHeader &header,
                        const uint8_t *packet)
{
    constexpr uint64_t kOffset = 1469598103934665603ull;
    EndpointDispatchPacketHeader canonical = header;
    canonical.metadata_digest = 0u;
    uint64_t hash = kOffset;
    HashBytes(&hash, &canonical, sizeof(canonical));
    const uint64_t count_bytes =
        static_cast<uint64_t>(header.worker_count) * sizeof(uint32_t) * 2u;
    HashBytes(&hash, packet + header.counts_offset, count_bytes);
    HashBytes(&hash, packet + header.tokens_offset,
              static_cast<uint64_t>(header.token_count) *
                  sizeof(EndpointDispatchTokenRecord));
    HashBytes(&hash, packet + header.assignments_offset,
              static_cast<uint64_t>(header.assignment_count) *
                  sizeof(EndpointDispatchAssignmentRecord));
    return hash;
}

bool ConfigValid(const EndpointDispatchConfig &config)
{
    return config.worker_count >= 2u &&
        config.worker_count <= kEndpointDispatchMaxWorkers &&
        config.expert_count != 0u &&
        config.hidden != 0u && DTypeBytes(config.dtype) != 0u &&
        config.source_rank < config.worker_count &&
        config.generation != 0u && config.sequence != 0u;
}

} // namespace

EndpointDispatchStatus BuildEndpointDispatchPacket(
    const EndpointDispatchInput &input, std::vector<uint8_t> *packet,
    EndpointDispatchCommit *commit, std::string *error)
{
    if (packet == nullptr || commit == nullptr ||
        !ConfigValid(input.config) ||
        input.token_ids.size() > std::numeric_limits<uint32_t>::max() ||
        input.assignments.size() > std::numeric_limits<uint32_t>::max() ||
        input.assignment_offsets.size() != input.token_ids.size() + 1u ||
        input.assignment_offsets.empty() ||
        input.assignment_offsets.front() != 0u ||
        input.assignment_offsets.back() != input.assignments.size() ||
        !std::is_sorted(input.assignment_offsets.begin(),
                        input.assignment_offsets.end())) {
        return Fail(EndpointDispatchStatus::INVALID_ARGUMENT,
                    "invalid Dispatch input shape", error);
    }

    uint64_t hidden_elements = 0u;
    uint64_t hidden_bytes = 0u;
    if (!Mul(input.token_ids.size(), input.config.hidden, &hidden_elements) ||
        !Mul(hidden_elements, DTypeBytes(input.config.dtype), &hidden_bytes)) {
        return Fail(EndpointDispatchStatus::SIZE_OVERFLOW,
                    "hidden payload size overflow", error);
    }
    if (input.hidden_payload.size() != hidden_bytes) {
        return Fail(EndpointDispatchStatus::PAYLOAD_SIZE_MISMATCH,
                    "hidden payload has the wrong size", error);
    }

    std::unordered_set<uint64_t> token_ids;
    std::vector<uint32_t> token_counts(input.config.worker_count, 0u);
    std::vector<uint32_t> assignment_counts(input.config.worker_count, 0u);
    std::vector<EndpointDispatchTokenRecord> tokens(input.token_ids.size());
    for (uint32_t token = 0u; token < input.token_ids.size(); ++token) {
        if (!token_ids.insert(input.token_ids[token]).second) {
            return Fail(EndpointDispatchStatus::DUPLICATE_TOKEN,
                        "duplicate token ID", error);
        }
        const uint32_t begin = input.assignment_offsets[token];
        const uint32_t end = input.assignment_offsets[token + 1u];
        std::unordered_set<uint32_t> ordinals;
        std::vector<uint8_t> destination_seen(input.config.worker_count, 0u);
        for (uint32_t i = begin; i < end; ++i) {
            const EndpointDispatchAssignmentRecord &assignment =
                input.assignments[i];
            if (assignment.destination_rank >= input.config.worker_count ||
                assignment.expert_id >= input.config.expert_count) {
                return Fail(EndpointDispatchStatus::OUT_OF_RANGE,
                            "assignment destination or expert out of range",
                            error);
            }
            if (!std::isfinite(assignment.weight)) {
                return Fail(EndpointDispatchStatus::NONFINITE_WEIGHT,
                            "non-finite routing weight", error);
            }
            if (!ordinals.insert(assignment.ordinal).second) {
                return Fail(EndpointDispatchStatus::DUPLICATE_ORDINAL,
                            "duplicate assignment ordinal", error);
            }
            if (assignment_counts[assignment.destination_rank] ==
                std::numeric_limits<uint32_t>::max()) {
                return Fail(EndpointDispatchStatus::SIZE_OVERFLOW,
                            "assignment count overflow", error);
            }
            ++assignment_counts[assignment.destination_rank];
            if (destination_seen[assignment.destination_rank] == 0u) {
                destination_seen[assignment.destination_rank] = 1u;
                if (token_counts[assignment.destination_rank] ==
                    std::numeric_limits<uint32_t>::max()) {
                    return Fail(EndpointDispatchStatus::SIZE_OVERFLOW,
                                "token count overflow", error);
                }
                ++token_counts[assignment.destination_rank];
            }
        }
        tokens[token].token_id = input.token_ids[token];
        tokens[token].source_token = token;
        tokens[token].assignment_begin = begin;
        tokens[token].assignment_count = end - begin;
    }

    uint64_t count_bytes = 0u;
    uint64_t token_bytes = 0u;
    uint64_t assignment_bytes = 0u;
    if (!Mul(input.config.worker_count, sizeof(uint32_t) * 2u,
             &count_bytes) ||
        !Mul(tokens.size(), sizeof(EndpointDispatchTokenRecord),
             &token_bytes) ||
        !Mul(input.assignments.size(),
             sizeof(EndpointDispatchAssignmentRecord), &assignment_bytes)) {
        return Fail(EndpointDispatchStatus::SIZE_OVERFLOW,
                    "metadata size overflow", error);
    }

    EndpointDispatchPacketHeader header{};
    header.generation = input.config.generation;
    header.sequence = input.config.sequence;
    header.wave = input.config.wave;
    header.source_rank = input.config.source_rank;
    header.worker_count = input.config.worker_count;
    header.token_count = static_cast<uint32_t>(tokens.size());
    header.hidden = input.config.hidden;
    header.dtype = static_cast<uint32_t>(input.config.dtype);
    header.assignment_count =
        static_cast<uint32_t>(input.assignments.size());
    header.counts_offset = sizeof(header);
    uint64_t end = 0u;
    if (!Add(header.counts_offset, count_bytes, &end) ||
        !Align(end, &header.tokens_offset) ||
        !Add(header.tokens_offset, token_bytes, &end) ||
        !Align(end, &header.assignments_offset) ||
        !Add(header.assignments_offset, assignment_bytes, &end) ||
        !Align(end, &header.hidden_offset) ||
        !Add(header.hidden_offset, hidden_bytes, &end) ||
        !Align(end, &header.packet_bytes) ||
        header.packet_bytes > std::numeric_limits<size_t>::max()) {
        return Fail(EndpointDispatchStatus::SIZE_OVERFLOW,
                    "packet size overflow", error);
    }

    packet->assign(static_cast<size_t>(header.packet_bytes), 0u);
    std::memcpy(packet->data(), &header, sizeof(header));
    std::memcpy(packet->data() + header.counts_offset, token_counts.data(),
                static_cast<size_t>(input.config.worker_count) *
                    sizeof(uint32_t));
    std::memcpy(packet->data() + header.counts_offset +
                    static_cast<uint64_t>(input.config.worker_count) *
                        sizeof(uint32_t),
                assignment_counts.data(),
                static_cast<size_t>(input.config.worker_count) *
                    sizeof(uint32_t));
    if (!tokens.empty())
        std::memcpy(packet->data() + header.tokens_offset, tokens.data(),
                    static_cast<size_t>(token_bytes));
    if (!input.assignments.empty())
        std::memcpy(packet->data() + header.assignments_offset,
                    input.assignments.data(),
                    static_cast<size_t>(assignment_bytes));
    if (!input.hidden_payload.empty())
        std::memcpy(packet->data() + header.hidden_offset,
                    input.hidden_payload.data(), input.hidden_payload.size());

    header.metadata_digest = MetadataDigest(header, packet->data());
    std::memcpy(packet->data(), &header, sizeof(header));
    EndpointDispatchCommit built_commit{};
    built_commit.generation = header.generation;
    built_commit.sequence = header.sequence;
    built_commit.wave = header.wave;
    built_commit.source_rank = header.source_rank;
    built_commit.slot = input.config.ring_slot;
    built_commit.packet_bytes = header.packet_bytes;
    built_commit.metadata_digest = header.metadata_digest;
    *commit = built_commit;
    if (error != nullptr) error->clear();
    return EndpointDispatchStatus::OK;
}

EndpointDispatchStatus ParseEndpointDispatchPacket(
    const uint8_t *packet, uint64_t packet_bytes,
    const EndpointDispatchConfig &expected, ParsedEndpointDispatch *parsed,
    std::string *error)
{
    if (packet == nullptr || parsed == nullptr || !ConfigValid(expected) ||
        packet_bytes < sizeof(EndpointDispatchPacketHeader)) {
        return Fail(EndpointDispatchStatus::INVALID_ARGUMENT,
                    "invalid parser arguments", error);
    }
    EndpointDispatchPacketHeader header{};
    std::memcpy(&header, packet, sizeof(header));
    for (uint64_t value : header.reserved) {
        if (value != 0u)
            return Fail(EndpointDispatchStatus::INVALID_HEADER,
                        "nonzero reserved header field", error);
    }
    if (header.magic != kEndpointDispatchMagic ||
        header.abi_version != kEndpointDispatchAbiVersion ||
        header.header_bytes != sizeof(header) || header.flags != 0u ||
        header.worker_count != expected.worker_count ||
        header.source_rank != expected.source_rank ||
        header.hidden != expected.hidden ||
        header.dtype != static_cast<uint32_t>(expected.dtype)) {
        return Fail(EndpointDispatchStatus::INVALID_HEADER,
                    "header/config mismatch", error);
    }
    if (header.generation != expected.generation ||
        header.wave != expected.wave) {
        return Fail(EndpointDispatchStatus::STALE_GENERATION,
                    "stale generation or wave", error);
    }
    if (header.sequence != expected.sequence) {
        return Fail(EndpointDispatchStatus::SEQUENCE_MISMATCH,
                    "unexpected packet sequence", error);
    }

    const uint64_t dtype_bytes = DTypeBytes(expected.dtype);
    uint64_t count_bytes = 0u;
    uint64_t token_bytes = 0u;
    uint64_t assignment_bytes = 0u;
    uint64_t hidden_elements = 0u;
    uint64_t hidden_bytes = 0u;
    uint64_t canonical_tokens = 0u;
    uint64_t canonical_assignments = 0u;
    uint64_t canonical_hidden = 0u;
    uint64_t canonical_packet = 0u;
    uint64_t end = 0u;
    if (!Mul(header.worker_count, sizeof(uint32_t) * 2u, &count_bytes) ||
        !Mul(header.token_count, sizeof(EndpointDispatchTokenRecord),
             &token_bytes) ||
        !Mul(header.assignment_count,
             sizeof(EndpointDispatchAssignmentRecord), &assignment_bytes) ||
        !Mul(header.token_count, header.hidden, &hidden_elements) ||
        !Mul(hidden_elements, dtype_bytes, &hidden_bytes) ||
        !Add(sizeof(header), count_bytes, &end) ||
        !Align(end, &canonical_tokens) ||
        !Add(canonical_tokens, token_bytes, &end) ||
        !Align(end, &canonical_assignments) ||
        !Add(canonical_assignments, assignment_bytes, &end) ||
        !Align(end, &canonical_hidden) ||
        !Add(canonical_hidden, hidden_bytes, &end) ||
        !Align(end, &canonical_packet)) {
        return Fail(EndpointDispatchStatus::SIZE_OVERFLOW,
                    "packet metadata size overflow", error);
    }
    if (header.counts_offset != sizeof(header) ||
        header.tokens_offset != canonical_tokens ||
        header.assignments_offset != canonical_assignments ||
        header.hidden_offset != canonical_hidden ||
        header.packet_bytes != canonical_packet ||
        header.packet_bytes != packet_bytes) {
        return Fail(EndpointDispatchStatus::PAYLOAD_SIZE_MISMATCH,
                    "noncanonical packet offsets or size", error);
    }
    if (MetadataDigest(header, packet) != header.metadata_digest) {
        return Fail(EndpointDispatchStatus::DIGEST_MISMATCH,
                    "metadata digest mismatch", error);
    }

    ParsedEndpointDispatch built{};
    built.header = header;
    built.token_counts.resize(header.worker_count);
    built.assignment_counts.resize(header.worker_count);
    built.source_tokens.reserve(header.token_count);
    built.destination_rows.resize(header.worker_count);
    std::memcpy(built.token_counts.data(), packet + header.counts_offset,
                static_cast<size_t>(header.worker_count) * sizeof(uint32_t));
    std::memcpy(built.assignment_counts.data(),
                packet + header.counts_offset +
                    static_cast<uint64_t>(header.worker_count) *
                        sizeof(uint32_t),
                static_cast<size_t>(header.worker_count) * sizeof(uint32_t));

    std::vector<uint32_t> actual_tokens(header.worker_count, 0u);
    std::vector<uint32_t> actual_assignments(header.worker_count, 0u);
    std::unordered_set<uint64_t> token_ids;
    uint32_t expected_assignment_begin = 0u;
    for (uint32_t token = 0u; token < header.token_count; ++token) {
        EndpointDispatchTokenRecord token_record{};
        std::memcpy(&token_record,
                    packet + header.tokens_offset +
                        static_cast<uint64_t>(token) * sizeof(token_record),
                    sizeof(token_record));
        if (token_record.source_token != token ||
            token_record.assignment_begin != expected_assignment_begin ||
            token_record.assignment_count > header.assignment_count -
                expected_assignment_begin ||
            token_record.reserved0 != 0u || token_record.reserved1 != 0u) {
            return Fail(EndpointDispatchStatus::INVALID_HEADER,
                        "invalid token record range", error);
        }
        if (!token_ids.insert(token_record.token_id).second) {
            return Fail(EndpointDispatchStatus::DUPLICATE_TOKEN,
                        "duplicate token ID", error);
        }
        built.source_tokens.push_back(token_record);
        std::vector<std::vector<EndpointDispatchAssignmentRecord>> grouped(
            header.worker_count);
        std::unordered_set<uint32_t> ordinals;
        for (uint32_t local = 0u; local < token_record.assignment_count;
             ++local) {
            EndpointDispatchAssignmentRecord assignment{};
            const uint32_t assignment_index =
                token_record.assignment_begin + local;
            std::memcpy(
                &assignment,
                packet + header.assignments_offset +
                    static_cast<uint64_t>(assignment_index) *
                        sizeof(assignment),
                sizeof(assignment));
            if (assignment.destination_rank >= header.worker_count ||
                assignment.expert_id >= expected.expert_count) {
                return Fail(EndpointDispatchStatus::OUT_OF_RANGE,
                            "assignment destination or expert out of range",
                            error);
            }
            if (!std::isfinite(assignment.weight)) {
                return Fail(EndpointDispatchStatus::NONFINITE_WEIGHT,
                            "non-finite routing weight", error);
            }
            if (!ordinals.insert(assignment.ordinal).second) {
                return Fail(EndpointDispatchStatus::DUPLICATE_ORDINAL,
                            "duplicate assignment ordinal", error);
            }
            grouped[assignment.destination_rank].push_back(assignment);
            ++actual_assignments[assignment.destination_rank];
        }
        const uint64_t hidden_row_bytes =
            static_cast<uint64_t>(header.hidden) * dtype_bytes;
        for (uint32_t destination = 0u; destination < header.worker_count;
             ++destination) {
            if (grouped[destination].empty()) continue;
            EndpointFanoutRow row{};
            row.token_id = token_record.token_id;
            row.source_rank = header.source_rank;
            row.source_token = token_record.source_token;
            row.assignments = std::move(grouped[destination]);
            row.hidden.resize(static_cast<size_t>(hidden_row_bytes));
            std::memcpy(
                row.hidden.data(),
                packet + header.hidden_offset +
                    static_cast<uint64_t>(token) * hidden_row_bytes,
                static_cast<size_t>(hidden_row_bytes));
            built.destination_rows[destination].push_back(std::move(row));
            ++actual_tokens[destination];
        }
        expected_assignment_begin += token_record.assignment_count;
    }
    if (expected_assignment_begin != header.assignment_count ||
        actual_tokens != built.token_counts ||
        actual_assignments != built.assignment_counts) {
        return Fail(EndpointDispatchStatus::COUNT_MISMATCH,
                    "count vector does not match token metadata", error);
    }
    *parsed = std::move(built);
    if (error != nullptr) error->clear();
    return EndpointDispatchStatus::OK;
}

const char *EndpointDispatchStatusString(EndpointDispatchStatus status)
{
    switch (status) {
        case EndpointDispatchStatus::OK: return "OK";
        case EndpointDispatchStatus::INVALID_ARGUMENT:
            return "INVALID_ARGUMENT";
        case EndpointDispatchStatus::INVALID_HEADER: return "INVALID_HEADER";
        case EndpointDispatchStatus::STALE_GENERATION:
            return "STALE_GENERATION";
        case EndpointDispatchStatus::SEQUENCE_MISMATCH:
            return "SEQUENCE_MISMATCH";
        case EndpointDispatchStatus::OUT_OF_RANGE: return "OUT_OF_RANGE";
        case EndpointDispatchStatus::SIZE_OVERFLOW: return "SIZE_OVERFLOW";
        case EndpointDispatchStatus::PAYLOAD_SIZE_MISMATCH:
            return "PAYLOAD_SIZE_MISMATCH";
        case EndpointDispatchStatus::COUNT_MISMATCH: return "COUNT_MISMATCH";
        case EndpointDispatchStatus::DUPLICATE_TOKEN:
            return "DUPLICATE_TOKEN";
        case EndpointDispatchStatus::DUPLICATE_ORDINAL:
            return "DUPLICATE_ORDINAL";
        case EndpointDispatchStatus::NONFINITE_WEIGHT:
            return "NONFINITE_WEIGHT";
        case EndpointDispatchStatus::DIGEST_MISMATCH:
            return "DIGEST_MISMATCH";
    }
    return "UNKNOWN";
}

} // namespace inc::dc::pull_combine
