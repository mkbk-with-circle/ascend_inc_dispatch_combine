#include "inc_dc_pull_dispatch_v2.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace inc::dc::pull_v2;

namespace {

uint32_t DTypeBytes(DataType dtype)
{
    return dtype == DataType::FP32 ? 4u : 2u;
}

SessionConfig Session(uint32_t workers = 4u, uint32_t experts = 8u,
                      uint32_t hidden = 3u,
                      DataType dtype = DataType::BF16)
{
    SessionConfig session{};
    session.session_id = 0x12345678u;
    session.placement_epoch = 9u;
    session.worker_count = workers;
    session.expert_count = experts;
    session.hidden = hidden;
    session.dtype = dtype;
    session.ring_slots = 2u;
    return session;
}

SourceInput BasicInput(uint32_t rank = 1u)
{
    SourceInput input{};
    input.session = Session();
    input.generation = 17u;
    input.sequence = 3u;
    input.wave = 5u;
    input.source_rank = rank;
    input.source_region_id = 11u + rank;
    input.ring_slot = 1u;
    // IDs deliberately repeat across ranks; owner rank/row is the identity.
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
        input.hidden_payload[i] = static_cast<uint8_t>(rank * 17u + i);
    return input;
}

RegionRegistration Registration(const SourceInput &input,
                                uint64_t slot_stride)
{
    RegionRegistration registration{};
    registration.session_id = input.session.session_id;
    registration.placement_epoch = input.session.placement_epoch;
    registration.source_rank = input.source_rank;
    registration.region_id = input.source_region_id;
    registration.slot_count = input.session.ring_slots;
    registration.region_bytes = slot_stride * registration.slot_count;
    registration.slot_stride = slot_stride;
    return registration;
}

ParsedSource BuildAndParse(const SourceInput &input)
{
    std::vector<uint8_t> slot;
    SlotHeader header{};
    Ready ready{};
    assert(BuildSlot(input, &slot, &header, &ready) == Status::OK);
    const RegionRegistration registration = Registration(input, slot.size());
    assert(ValidateRegistration(registration, input.session) == Status::OK);
    ParsedSource parsed{};
    assert(ParseReadyAndSlot(registration, input.session, ready,
                             slot.data(), slot.size(), &parsed) ==
           Status::OK);
    assert(parsed.header.metadata_digest == header.metadata_digest);
    assert(parsed.hidden_payload == input.hidden_payload);
    return parsed;
}

void TestBasicProtocol()
{
    const SourceInput input = BasicInput();
    std::vector<uint8_t> slot;
    SlotHeader header{};
    Ready ready{};
    assert(BuildSlot(input, &slot, &header, &ready) == Status::OK);
    assert(ready.publication == ReadyPublication(ready));
    assert(header.token_count == 2u);
    assert(header.assignment_count == 4u);
    assert(header.hidden_offset % kPullDispatchAlignment == 0u);

    RegionRegistration registration = Registration(input, slot.size());
    ParsedSource parsed{};
    assert(ParseReadyAndSlot(registration, input.session, ready,
                             slot.data(), slot.size(), &parsed) ==
           Status::OK);
    assert(parsed.tokens.size() == 2u);
    assert(parsed.assignments.size() == 4u);

    Ready bad_ready = ready;
    ++bad_ready.sequence;
    assert(ParseReadyAndSlot(registration, input.session, bad_ready,
                             slot.data(), slot.size(), &parsed) ==
           Status::INVALID_READY);

    std::vector<uint8_t> corrupt = slot;
    corrupt[header.tokens_offset] ^= 1u;
    assert(ParseReadyAndSlot(registration, input.session, ready,
                             corrupt.data(), corrupt.size(), &parsed) ==
           Status::DIGEST_MISMATCH);

    RegionRegistration bad_registration = registration;
    bad_registration.slot_stride -= 1u;
    assert(ValidateRegistration(bad_registration, input.session) ==
           Status::INVALID_REGISTRATION);

    SourceInput bad_assignment = input;
    bad_assignment.assignments[1].ordinal = 0u;
    assert(BuildSlot(bad_assignment, &slot, &header, &ready) ==
           Status::INVALID_ASSIGNMENT);
}

void TestLayoutAndJournal()
{
    std::vector<ParsedSource> sources;
    for (uint32_t rank = 0u; rank < 4u; ++rank)
        sources.push_back(BuildAndParse(BasicInput(rank)));
    LayoutConfig config{};
    config.worker_count = 4u;
    config.expert_count = 8u;
    config.destination_row_capacity = 16u;
    config.destination_assignment_capacity = 32u;
    CompiledLayout layout{};
    assert(CompileLayout(sources, {3u, 1u, 0u, 2u}, config, &layout) ==
           Status::OK);
    assert(layout.journal_tokens.size() == 8u);
    assert(layout.journal_header.token_count == 8u);
    assert(layout.journal_header.dispatch_cookie != 0u);
    assert(static_cast<JournalSlotState>(layout.journal_header.state) ==
           JournalSlotState::DISPATCH_SEALED);
    // Token 0 has two local experts on destination 0 but only one network row.
    assert(layout.destination_rows[0].size() == 4u);
    assert(layout.destination_rows[0][0].assignments_count == 2u);
    assert(layout.expert_assignments[0].size() == 8u);
    // Token IDs repeat across ranks without aliasing route keys.
    assert(layout.journal_tokens[0].token_id ==
           layout.journal_tokens[2].token_id);
    assert(layout.journal_tokens[0].route_key !=
           layout.journal_tokens[2].route_key);
    for (uint32_t rank = 0u; rank < 4u; ++rank) {
        for (uint32_t row = 0u; row < 2u; ++row) {
            const JournalTokenEntry &entry =
                layout.journal_tokens[rank * 2u + row];
            assert(entry.route_key == RouteKey(rank, row));
            assert(entry.owner_rank == rank && entry.owner_row == row);
        }
    }

    LayoutConfig too_small = config;
    too_small.destination_row_capacity = 1u;
    assert(CompileLayout(sources, {0u, 1u, 2u, 3u}, too_small,
                         &layout) == Status::CAPACITY_EXCEEDED);
    assert(CompileLayout(sources, {0u, 1u, 1u, 3u}, config,
                         &layout) == Status::DUPLICATE_SOURCE);
}

void TestJournalStateMachine()
{
    JournalSlotHeader header{};
    assert(TransitionJournal(&header, JournalSlotState::DISPATCH_OPEN) ==
           Status::OK);
    assert(TransitionJournal(&header, JournalSlotState::COMBINE_ACTIVE) ==
           Status::INVALID_STATE_TRANSITION);
    assert(TransitionJournal(&header, JournalSlotState::DISPATCH_SEALED) ==
           Status::OK);
    assert(TransitionJournal(&header, JournalSlotState::COMBINE_ACTIVE) ==
           Status::OK);
    assert(TransitionJournal(&header, JournalSlotState::COMPLETE) ==
           Status::OK);
    assert(TransitionJournal(&header, JournalSlotState::FREE) == Status::OK);
}

SourceInput UniformOrEmptyInput(uint32_t rank, bool active)
{
    SourceInput input{};
    input.session = Session(4u, 8u, 3u, DataType::BF16);
    input.generation = 91u;
    input.sequence = 27u;
    input.wave = 6u;
    input.source_rank = rank;
    input.source_region_id = rank + 1u;
    input.ring_slot = 0u;
    input.assignment_offsets = {0u};
    if (!active) return input;
    input.token_ids = {700u, 701u};
    input.assignment_offsets = {0u, 2u, 4u};
    input.assignments = {
        {0u, 0u, 0u, 0.4f}, {2u, 1u, 1u, 0.6f},
        {0u, 2u, 0u, 0.3f}, {2u, 3u, 1u, 0.7f},
    };
    input.hidden_payload.resize(2u * 3u * 2u, 0x5au);
    return input;
}

void TestEmptySourcesAreUniform()
{
    std::vector<ParsedSource> one_active;
    for (uint32_t rank = 0u; rank < 4u; ++rank) {
        const ParsedSource source = BuildAndParse(
            UniformOrEmptyInput(rank, rank == 0u));
        assert((source.header.flags & kSlotFlagUniformDestinations) != 0u);
        one_active.push_back(source);
    }
    LayoutConfig config{};
    config.worker_count = 4u;
    config.expert_count = 8u;
    config.destination_row_capacity = 8u;
    config.destination_assignment_capacity = 8u;
    CompiledLayout layout{};
    assert(CompileLayout(one_active, {0u, 1u, 2u, 3u}, config, &layout) ==
           Status::OK);
    assert((layout.journal_header.flags &
            kJournalFlagUniformDestinations) != 0u);
    assert(layout.journal_header.token_count == 2u);
    assert(layout.destination_rows[0].size() == 2u);
    assert(layout.destination_rows[2].size() == 2u);

    std::vector<ParsedSource> all_empty;
    for (uint32_t rank = 0u; rank < 4u; ++rank) {
        const ParsedSource source = BuildAndParse(
            UniformOrEmptyInput(rank, false));
        assert((source.header.flags & kSlotFlagUniformDestinations) != 0u);
        all_empty.push_back(source);
    }
    assert(CompileLayout(all_empty, {3u, 2u, 1u, 0u}, config, &layout) ==
           Status::OK);
    assert((layout.journal_header.flags &
            kJournalFlagUniformDestinations) != 0u);
    assert(layout.journal_header.token_count == 0u);
    assert(layout.journal_header.contributor_count == 0u);
}

void TestRandomPackets()
{
    constexpr uint64_t kSeed = 0x50444c4c7632ull;
    std::mt19937_64 rng(kSeed);
    for (uint32_t iteration = 0u; iteration < 5000u; ++iteration) {
        const uint32_t workers = 2u + static_cast<uint32_t>(rng() % 7u);
        const uint32_t experts = 1u + static_cast<uint32_t>(rng() % 32u);
        const uint32_t hidden = 1u + static_cast<uint32_t>(rng() % 17u);
        const DataType dtype = static_cast<DataType>(rng() % 3u);
        SourceInput input{};
        input.session = Session(workers, experts, hidden, dtype);
        input.generation = 100u + iteration;
        input.sequence = 1u + iteration;
        input.wave = iteration % 31u;
        input.source_rank = static_cast<uint32_t>(rng() % workers);
        input.source_region_id = 1u + input.source_rank;
        input.ring_slot = static_cast<uint16_t>(iteration & 1u);
        const uint32_t tokens = static_cast<uint32_t>(rng() % 17u);
        input.assignment_offsets.push_back(0u);
        for (uint32_t token = 0u; token < tokens; ++token) {
            // Deliberately small ID space verifies it is not a global key.
            input.token_ids.push_back(rng() % 8u);
            const uint32_t topk = static_cast<uint32_t>(rng() % 9u);
            for (uint32_t k = 0u; k < topk; ++k) {
                input.assignments.push_back(AssignmentRecord{
                    static_cast<uint32_t>(rng() % workers),
                    static_cast<uint32_t>(rng() % experts), k,
                    static_cast<float>(static_cast<int32_t>(rng() % 2001u) -
                                       1000) /
                        1000.0f});
            }
            input.assignment_offsets.push_back(input.assignments.size());
        }
        input.hidden_payload.resize(
            static_cast<size_t>(tokens) * hidden * DTypeBytes(dtype));
        for (uint8_t &byte : input.hidden_payload)
            byte = static_cast<uint8_t>(rng());
        ParsedSource parsed = BuildAndParse(input);
        assert(parsed.tokens.size() == tokens);
        assert(parsed.assignments.size() == input.assignments.size());
        assert(parsed.hidden_payload == input.hidden_payload);
    }
}

void TestRandomLayouts()
{
    constexpr uint64_t kSeed = 0x4c41594f55547632ull;
    std::mt19937_64 rng(kSeed);
    for (uint32_t iteration = 0u; iteration < 500u; ++iteration) {
        const uint32_t workers = 2u + static_cast<uint32_t>(rng() % 7u);
        const uint32_t experts = 1u + static_cast<uint32_t>(rng() % 16u);
        const uint32_t hidden = 1u + static_cast<uint32_t>(rng() % 9u);
        const DataType dtype = static_cast<DataType>(rng() % 3u);
        const uint32_t tokens = static_cast<uint32_t>(rng() % 17u);
        const SessionConfig session = Session(workers, experts, hidden, dtype);
        std::vector<ParsedSource> sources;
        uint64_t total_tokens = 0u;
        uint64_t total_assignments = 0u;
        for (uint32_t rank = 0u; rank < workers; ++rank) {
            SourceInput input{};
            input.session = session;
            input.generation = 1000u + iteration;
            input.sequence = 500u + iteration;
            input.wave = iteration % 13u;
            input.source_rank = rank;
            input.source_region_id = rank + 1u;
            input.ring_slot = static_cast<uint16_t>(iteration & 1u);
            input.assignment_offsets.push_back(0u);
            for (uint32_t token = 0u; token < tokens; ++token) {
                input.token_ids.push_back(token % 3u);
                const uint32_t topk = static_cast<uint32_t>(rng() % 9u);
                for (uint32_t k = 0u; k < topk; ++k) {
                    input.assignments.push_back(AssignmentRecord{
                        static_cast<uint32_t>(rng() % workers),
                        static_cast<uint32_t>(rng() % experts), k,
                        static_cast<float>(k + 1u) /
                            static_cast<float>(topk + 1u)});
                }
                input.assignment_offsets.push_back(input.assignments.size());
            }
            input.hidden_payload.resize(
                static_cast<size_t>(tokens) * hidden * DTypeBytes(dtype));
            for (uint8_t &byte : input.hidden_payload)
                byte = static_cast<uint8_t>(rng());
            total_tokens += input.token_ids.size();
            total_assignments += input.assignments.size();
            sources.push_back(BuildAndParse(input));
        }
        std::vector<uint32_t> arrival(workers);
        for (uint32_t rank = 0u; rank < workers; ++rank)
            arrival[rank] = rank;
        std::shuffle(arrival.begin(), arrival.end(), rng);
        LayoutConfig config{};
        config.worker_count = workers;
        config.expert_count = experts;
        config.destination_row_capacity =
            std::max<uint64_t>(1u, total_tokens);
        config.destination_assignment_capacity =
            std::max<uint64_t>(1u, total_assignments);
        CompiledLayout layout{};
        assert(CompileLayout(sources, arrival, config, &layout) ==
               Status::OK);
        assert(layout.journal_tokens.size() == total_tokens);
        assert(layout.journal_assignments.size() == total_assignments);
        assert(layout.journal_header.token_count == total_tokens);
        assert(layout.journal_header.contributor_count ==
               layout.contributors.size());
        for (const JournalTokenEntry &entry : layout.journal_tokens) {
            assert(entry.route_key ==
                   RouteKey(entry.owner_rank, entry.owner_row));
            assert(entry.contributors_begin + entry.contributors_count <=
                   layout.contributors.size());
            for (uint32_t i = 0u; i < entry.contributors_count; ++i) {
                const JournalContributor &contributor = layout.contributors[
                    entry.contributors_begin + i];
                assert(contributor.worker_rank < workers);
                assert(contributor.destination_row <
                       layout.destination_rows[
                           contributor.worker_rank].size());
                const DestinationRow &row = layout.destination_rows[
                    contributor.worker_rank][contributor.destination_row];
                assert(row.route_key == entry.route_key);
                assert(row.assignments_begin ==
                       contributor.assignment_begin);
                assert(row.assignments_count ==
                       contributor.assignment_count);
            }
        }
        for (uint32_t destination = 0u; destination < workers;
             ++destination) {
            for (uint32_t row = 0u;
                 row < layout.destination_rows[destination].size(); ++row) {
                const DestinationRow &destination_row =
                    layout.destination_rows[destination][row];
                assert(destination_row.destination_row == row);
                assert(destination_row.assignments_begin +
                           destination_row.assignments_count <=
                       layout.expert_assignments[destination].size());
                for (uint32_t i = 0u;
                     i < destination_row.assignments_count; ++i) {
                    const ExpertAssignment &assignment =
                        layout.expert_assignments[destination][
                            destination_row.assignments_begin + i];
                    assert(assignment.destination_row == row);
                    assert(assignment.expert_id < experts);
                }
            }
        }
    }
}

} // namespace

int main()
{
    TestBasicProtocol();
    TestLayoutAndJournal();
    TestJournalStateMachine();
    TestEmptySourcesAreUniform();
    TestRandomPackets();
    TestRandomLayouts();
    return 0;
}
