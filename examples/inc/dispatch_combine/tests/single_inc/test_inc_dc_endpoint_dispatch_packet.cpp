#include "inc_dc_endpoint_dispatch_packet.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace inc::dc::pull_combine;

namespace {

uint32_t DTypeBytes(EndpointDataType dtype)
{
    return dtype == EndpointDataType::FP32 ? 4u : 2u;
}

EndpointDispatchInput BasicInput()
{
    EndpointDispatchInput input{};
    input.config.worker_count = 4u;
    input.config.expert_count = 8u;
    input.config.hidden = 3u;
    input.config.dtype = EndpointDataType::BF16;
    input.config.wave = 2u;
    input.config.source_rank = 1u;
    input.config.generation = 7u;
    input.config.sequence = 3u;
    input.config.ring_slot = 5u;
    input.token_ids = {100u, 101u};
    input.assignment_offsets = {0u, 3u, 4u};
    input.assignments = {
        {0u, 0u, 0u, 0.25f},
        {0u, 1u, 1u, 0.50f},
        {2u, 4u, 2u, 0.75f},
        {1u, 3u, 0u, 1.00f},
    };
    input.hidden_payload.resize(12u);
    for (uint32_t i = 0u; i < input.hidden_payload.size(); ++i)
        input.hidden_payload[i] = static_cast<uint8_t>(11u + i);
    return input;
}

void TestBasic()
{
    const EndpointDispatchInput input = BasicInput();
    std::vector<uint8_t> packet;
    EndpointDispatchCommit commit{};
    assert(BuildEndpointDispatchPacket(input, &packet, &commit) ==
           EndpointDispatchStatus::OK);
    assert(commit.generation == input.config.generation);
    assert(commit.sequence == input.config.sequence);
    assert(commit.slot == input.config.ring_slot);
    assert(commit.packet_bytes == packet.size());
    assert(commit.metadata_digest != 0u);

    ParsedEndpointDispatch parsed{};
    assert(ParseEndpointDispatchPacket(packet.data(), packet.size(),
                                       input.config, &parsed) ==
           EndpointDispatchStatus::OK);
    assert((parsed.token_counts == std::vector<uint32_t>{1u, 1u, 1u, 0u}));
    assert((parsed.assignment_counts ==
            std::vector<uint32_t>{2u, 1u, 1u, 0u}));
    assert(parsed.destination_rows[0].size() == 1u);
    assert(parsed.destination_rows[0][0].assignments.size() == 2u);
    assert(parsed.destination_rows[0][0].hidden ==
           std::vector<uint8_t>(input.hidden_payload.begin(),
                                input.hidden_payload.begin() + 6));
    assert(parsed.destination_rows[1].size() == 1u);
    assert(parsed.destination_rows[1][0].token_id == 101u);
    assert(parsed.destination_rows[2].size() == 1u);
    assert(parsed.destination_rows[2][0].token_id == 100u);
    assert(parsed.destination_rows[3].empty());

    EndpointDispatchConfig stale = input.config;
    ++stale.generation;
    assert(ParseEndpointDispatchPacket(packet.data(), packet.size(), stale,
                                       &parsed) ==
           EndpointDispatchStatus::STALE_GENERATION);

    std::vector<uint8_t> corrupt = packet;
    corrupt[sizeof(EndpointDispatchPacketHeader)] ^= 1u;
    assert(ParseEndpointDispatchPacket(corrupt.data(), corrupt.size(),
                                       input.config, &parsed) ==
           EndpointDispatchStatus::DIGEST_MISMATCH);
}

void TestRejectedInputs()
{
    std::vector<uint8_t> packet;
    EndpointDispatchCommit commit{};

    EndpointDispatchInput duplicate_token = BasicInput();
    duplicate_token.token_ids[1] = duplicate_token.token_ids[0];
    assert(BuildEndpointDispatchPacket(duplicate_token, &packet, &commit) ==
           EndpointDispatchStatus::DUPLICATE_TOKEN);

    EndpointDispatchInput duplicate_ordinal = BasicInput();
    duplicate_ordinal.assignments[1].ordinal =
        duplicate_ordinal.assignments[0].ordinal;
    assert(BuildEndpointDispatchPacket(duplicate_ordinal, &packet, &commit) ==
           EndpointDispatchStatus::DUPLICATE_ORDINAL);

    EndpointDispatchInput nonfinite = BasicInput();
    nonfinite.assignments[0].weight =
        std::numeric_limits<float>::infinity();
    assert(BuildEndpointDispatchPacket(nonfinite, &packet, &commit) ==
           EndpointDispatchStatus::NONFINITE_WEIGHT);

    EndpointDispatchInput bad_payload = BasicInput();
    bad_payload.hidden_payload.pop_back();
    assert(BuildEndpointDispatchPacket(bad_payload, &packet, &commit) ==
           EndpointDispatchStatus::PAYLOAD_SIZE_MISMATCH);
}

void TestRandom()
{
    constexpr uint64_t kSeed = 0xd15ca7c2a11ull;
    std::mt19937_64 rng(kSeed);
    for (uint32_t iteration = 0u; iteration < 5000u; ++iteration) {
        EndpointDispatchInput input{};
        input.config.worker_count = 2u + static_cast<uint32_t>(rng() % 7u);
        input.config.expert_count = 1u + static_cast<uint32_t>(rng() % 64u);
        input.config.hidden = 1u + static_cast<uint32_t>(rng() % 33u);
        input.config.dtype = static_cast<EndpointDataType>(rng() % 3u);
        input.config.wave = iteration % 17u;
        input.config.source_rank = static_cast<uint32_t>(
            rng() % input.config.worker_count);
        input.config.generation = 100u + iteration;
        input.config.sequence = 1u + iteration;
        const uint32_t tokens = static_cast<uint32_t>(rng() % 33u);
        input.assignment_offsets.push_back(0u);
        std::vector<std::vector<uint32_t>> expected_token_counts(
            tokens, std::vector<uint32_t>(input.config.worker_count, 0u));
        for (uint32_t token = 0u; token < tokens; ++token) {
            input.token_ids.push_back(
                (static_cast<uint64_t>(iteration) << 32u) + token + 1u);
            const uint32_t topk = static_cast<uint32_t>(rng() % 13u);
            for (uint32_t k = 0u; k < topk; ++k) {
                EndpointDispatchAssignmentRecord assignment{};
                assignment.destination_rank = static_cast<uint32_t>(
                    rng() % input.config.worker_count);
                assignment.expert_id = static_cast<uint32_t>(
                    rng() % input.config.expert_count);
                assignment.ordinal = k;
                assignment.weight =
                    static_cast<float>(static_cast<int32_t>(rng() % 2001u) -
                                       1000) /
                    1000.0f;
                input.assignments.push_back(assignment);
                expected_token_counts[token][assignment.destination_rank] =
                    1u;
            }
            input.assignment_offsets.push_back(
                static_cast<uint32_t>(input.assignments.size()));
        }
        const uint64_t hidden_bytes = static_cast<uint64_t>(tokens) *
            input.config.hidden * DTypeBytes(input.config.dtype);
        input.hidden_payload.resize(static_cast<size_t>(hidden_bytes));
        for (uint8_t &value : input.hidden_payload)
            value = static_cast<uint8_t>(rng());

        std::vector<uint8_t> packet;
        EndpointDispatchCommit commit{};
        assert(BuildEndpointDispatchPacket(input, &packet, &commit) ==
               EndpointDispatchStatus::OK);
        ParsedEndpointDispatch parsed{};
        assert(ParseEndpointDispatchPacket(packet.data(), packet.size(),
                                           input.config, &parsed) ==
               EndpointDispatchStatus::OK);

        std::vector<uint32_t> expected_rows(input.config.worker_count, 0u);
        std::vector<uint32_t> expected_assignments(
            input.config.worker_count, 0u);
        for (uint32_t token = 0u; token < tokens; ++token) {
            for (uint32_t destination = 0u;
                 destination < input.config.worker_count; ++destination) {
                expected_rows[destination] +=
                    expected_token_counts[token][destination];
            }
        }
        for (const auto &assignment : input.assignments)
            ++expected_assignments[assignment.destination_rank];
        assert(parsed.token_counts == expected_rows);
        assert(parsed.assignment_counts == expected_assignments);
        for (uint32_t destination = 0u;
             destination < input.config.worker_count; ++destination) {
            assert(parsed.destination_rows[destination].size() ==
                   expected_rows[destination]);
            for (const EndpointFanoutRow &row :
                 parsed.destination_rows[destination]) {
                assert(row.hidden.size() == static_cast<size_t>(
                    input.config.hidden * DTypeBytes(input.config.dtype)));
                assert(!row.assignments.empty());
                for (const auto &assignment : row.assignments)
                    assert(assignment.destination_rank == destination);
            }
        }
    }
}

} // namespace

int main()
{
    TestBasic();
    TestRejectedInputs();
    TestRandom();
    return 0;
}
