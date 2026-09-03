#include "inc_dc_pull_combine_v2.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

using namespace inc::dc::pull_v2;

namespace {

uint32_t Bytes(PartialDataType dtype)
{
    return dtype == PartialDataType::FP32 ? 4u : 2u;
}

uint64_t Align64(uint64_t value)
{
    return (value + 63u) / 64u * 64u;
}

SessionConfig DispatchSession(uint32_t workers, uint32_t experts,
                              uint32_t hidden)
{
    SessionConfig session{};
    session.session_id = 0x12340000u + workers;
    session.placement_epoch = 19u;
    session.worker_count = workers;
    session.expert_count = experts;
    session.hidden = hidden;
    session.dtype = DataType::BF16;
    session.ring_slots = 2u;
    return session;
}

ParsedSource BuildSource(const SessionConfig &session, uint32_t rank,
                         uint64_t generation, uint64_t sequence,
                         uint32_t wave, uint16_t slot,
                         uint32_t tokens, std::mt19937_64 *rng)
{
    SourceInput input{};
    input.session = session;
    input.generation = generation;
    input.sequence = sequence;
    input.wave = wave;
    input.source_rank = rank;
    input.source_region_id = 100u + rank;
    input.ring_slot = slot;
    input.assignment_offsets.push_back(0u);
    for (uint32_t token = 0u; token < tokens; ++token) {
        // Deliberately repeated both within a rank and across ranks. The
        // protocol identity is (owner_rank, owner_row), never token_id.
        input.token_ids.push_back(token % 3u);
        const uint32_t topk = rng == nullptr ? 4u :
            static_cast<uint32_t>((*rng)() % 9u);
        for (uint32_t ordinal = 0u; ordinal < topk; ++ordinal) {
            uint32_t destination = rng == nullptr ?
                (token + ordinal / 2u) % session.worker_count :
                static_cast<uint32_t>((*rng)() % session.worker_count);
            input.assignments.push_back(AssignmentRecord{
                destination,
                rng == nullptr ? ordinal % session.expert_count :
                    static_cast<uint32_t>((*rng)() % session.expert_count),
                ordinal, 1.0f / static_cast<float>(ordinal + 1u)});
        }
        input.assignment_offsets.push_back(input.assignments.size());
    }
    input.hidden_payload.resize(static_cast<size_t>(tokens) * session.hidden *
                                2u, static_cast<uint8_t>(rank + 1u));

    std::vector<uint8_t> bytes;
    SlotHeader header{};
    Ready ready{};
    assert(BuildSlot(input, &bytes, &header, &ready) == Status::OK);
    RegionRegistration registration{};
    registration.session_id = session.session_id;
    registration.placement_epoch = session.placement_epoch;
    registration.source_rank = rank;
    registration.region_id = input.source_region_id;
    registration.slot_count = session.ring_slots;
    registration.slot_stride = bytes.size();
    registration.region_bytes = bytes.size() * session.ring_slots;
    ParsedSource parsed{};
    assert(ParseReadyAndSlot(registration, session, ready, bytes.data(),
                             bytes.size(), &parsed) == Status::OK);
    return parsed;
}

CompiledLayout BuildLayout(uint32_t workers, uint32_t experts,
                           uint32_t hidden, uint32_t tokens,
                           uint64_t generation, uint64_t sequence,
                           uint32_t wave, uint16_t slot,
                           std::mt19937_64 *rng = nullptr)
{
    const SessionConfig session = DispatchSession(workers, experts, hidden);
    std::vector<ParsedSource> sources;
    uint64_t total_tokens = 0u;
    uint64_t total_assignments = 0u;
    for (uint32_t rank = 0u; rank < workers; ++rank) {
        const uint32_t rank_tokens = tokens + (rank & 1u);
        sources.push_back(BuildSource(session, rank, generation, sequence,
                                      wave, slot, rank_tokens, rng));
        total_tokens += sources.back().tokens.size();
        total_assignments += sources.back().assignments.size();
    }
    std::vector<uint32_t> arrival(workers);
    for (uint32_t rank = 0u; rank < workers; ++rank) arrival[rank] = rank;
    if (rng != nullptr) std::shuffle(arrival.begin(), arrival.end(), *rng);
    LayoutConfig config{};
    config.worker_count = workers;
    config.expert_count = experts;
    config.destination_row_capacity = std::max<uint64_t>(1u, total_tokens);
    config.destination_assignment_capacity =
        std::max<uint64_t>(1u, total_assignments);
    CompiledLayout layout{};
    assert(CompileLayout(sources, arrival, config, &layout) == Status::OK);
    return layout;
}

CombineV2Config Config(uint32_t workers, uint32_t hidden,
                       PartialDataType dtype = PartialDataType::FP32)
{
    CombineV2Config config{};
    config.session_id = 0xabc00000u + workers;
    config.placement_epoch = 31u;
    config.worker_count = workers;
    config.hidden = hidden;
    config.partial_dtype = dtype;
    config.ring_slots = 2u;
    return config;
}

std::vector<CombineRegionRegistration> Registrations(
    const CombinePullPlan &plan, const CombineV2Config &config)
{
    std::vector<CombineRegionRegistration> registrations;
    for (uint32_t rank = 0u; rank < config.worker_count; ++rank) {
        const uint64_t payload = static_cast<uint64_t>(
            plan.source_row_counts[rank]) * config.hidden *
            Bytes(config.partial_dtype);
        CombineRegionRegistration registration{};
        registration.session_id = config.session_id;
        registration.placement_epoch = config.placement_epoch;
        registration.source_rank = rank;
        registration.region_id = 700u + rank;
        registration.slot_count = config.ring_slots;
        registration.slot_stride = Align64(payload + 128u);
        registration.region_bytes = registration.slot_stride *
            registration.slot_count;
        registrations.push_back(registration);
    }
    return registrations;
}

CombineReadyV2 MakeReady(const CombinePullPlan &plan,
                         const CombineV2Config &config,
                         const CombineRegionRegistration &registration)
{
    CombineReadyV2 ready{};
    ready.session_id = config.session_id;
    ready.placement_epoch = config.placement_epoch;
    ready.generation = plan.generation;
    ready.sequence = plan.sequence;
    ready.dispatch_cookie = plan.dispatch_cookie;
    ready.wave = plan.wave;
    ready.source_rank = registration.source_rank;
    ready.source_region_id = registration.region_id;
    ready.ring_slot = plan.ring_slot;
    ready.row_count = plan.source_row_counts[ready.source_rank];
    ready.hidden = config.hidden;
    ready.partial_dtype = static_cast<uint32_t>(config.partial_dtype);
    ready.source_offset = static_cast<uint64_t>(ready.ring_slot) *
        registration.slot_stride;
    ready.payload_bytes = static_cast<uint64_t>(ready.row_count) *
        config.hidden * Bytes(config.partial_dtype);
    ready.publication = CombineReadyPublication(ready);
    return ready;
}

void Republish(CombineReadyV2 *ready)
{
    ready->publication = 0u;
    ready->publication = CombineReadyPublication(*ready);
}

void TestCanonicalDirectPlan()
{
    CompiledLayout layout = BuildLayout(4u, 16u, 8u, 5u,
                                        101u, 7u, 3u, 1u);
    const CombineV2Config config = Config(4u, 8u);
    CombinePullPlan plan{};
    assert(CompileCombinePullPlan(layout, config, &plan) ==
           CombineV2Status::OK);
    assert(plan.dispatch_cookie == layout.journal_header.dispatch_cookie);
    assert(plan.pulls.size() == layout.contributors.size());
    assert(plan.results.size() == layout.journal_tokens.size());

    // The data plane consumes each B as one contiguous canonical array and
    // indexes it directly by source_row; no token-id lookup is involved.
    for (uint32_t source = 0u; source < config.worker_count; ++source) {
        assert(plan.source_offsets[source + 1u] -
                   plan.source_offsets[source] ==
               plan.source_row_counts[source]);
        for (uint32_t row = 0u; row < plan.source_row_counts[source]; ++row) {
            const CombinePullOp &op =
                plan.pulls[plan.source_offsets[source] + row];
            assert(op.source_rank == source && op.source_row == row);
        }
    }

    // Emulate reduction using only the compiled direct pull operations.
    std::vector<uint32_t> reduced(plan.accumulator_count, 0u);
    std::vector<uint32_t> received(plan.accumulator_count, 0u);
    for (const CombinePullOp &op : plan.pulls) {
        reduced[op.accumulator_index] += 1000u * op.source_rank +
                                         op.source_row + 1u;
        ++received[op.accumulator_index];
    }
    for (const CombineResultOp &result : plan.results) {
        assert(received[result.accumulator_index] ==
               result.expected_contributors);
        assert(result.owner_rank < config.worker_count);
        (void)reduced[result.accumulator_index];
    }

    // Repeated framework token ids remain independent journal entries.
    bool repeated = false;
    for (size_t i = 0u; i < layout.journal_tokens.size(); ++i)
        for (size_t j = i + 1u; j < layout.journal_tokens.size(); ++j)
            repeated |= layout.journal_tokens[i].token_id ==
                            layout.journal_tokens[j].token_id &&
                        layout.journal_tokens[i].route_key !=
                            layout.journal_tokens[j].route_key;
    assert(repeated);

    // Multiple experts on the same B produce one contributor/partial.
    bool saw_local_reduce = false;
    for (const JournalContributor &contributor : layout.contributors)
        saw_local_reduce |= contributor.assignment_count > 1u;
    assert(saw_local_reduce);
}

void TestReadyAndStateMachine()
{
    CompiledLayout layout = BuildLayout(4u, 8u, 16u, 4u,
                                        202u, 8u, 9u, 1u);
    const CombineV2Config config = Config(4u, 16u);
    CombinePullPlan preplan{};
    assert(CompileCombinePullPlan(layout, config, &preplan) ==
           CombineV2Status::OK);
    std::vector<CombineRegionRegistration> registrations =
        Registrations(preplan, config);
    std::reverse(registrations.begin(), registrations.end());

    CombineV2Coordinator coordinator;
    assert(coordinator.Initialize(&layout, config, registrations) ==
           CombineV2Status::OK);
    assert(coordinator.state() == JournalSlotState::COMBINE_ACTIVE);
    assert(coordinator.Finish() == CombineV2Status::NOT_READY);

    const std::vector<uint32_t> order{2u, 0u, 3u, 1u};
    for (uint32_t rank : order) {
        const CombineRegionRegistration &registration =
            registrations[config.worker_count - 1u - rank];
        CombineReadyV2 ready = MakeReady(coordinator.plan(), config,
                                         registration);
        assert(coordinator.NotifyReady(ready) == CombineV2Status::OK);
        assert(coordinator.NotifyReady(ready) ==
               CombineV2Status::DUPLICATE_SOURCE);
    }
    assert(coordinator.all_ready());
    assert(coordinator.Finish() == CombineV2Status::OK);
    assert(coordinator.state() == JournalSlotState::COMPLETE);
    assert(coordinator.Finish() ==
           CombineV2Status::INVALID_STATE_TRANSITION);

    CompiledLayout aborted = BuildLayout(2u, 4u, 8u, 2u,
                                         203u, 9u, 10u, 0u);
    const CombineV2Config config2 = Config(2u, 8u);
    CombinePullPlan plan2{};
    assert(CompileCombinePullPlan(aborted, config2, &plan2) ==
           CombineV2Status::OK);
    CombineV2Coordinator abort_coordinator;
    assert(abort_coordinator.Initialize(
               &aborted, config2, Registrations(plan2, config2)) ==
           CombineV2Status::OK);
    assert(abort_coordinator.Abort() == CombineV2Status::ABORTED);
    assert(abort_coordinator.state() == JournalSlotState::ABORTED);
}

void TestCookieAndBoundsFaults()
{
    CompiledLayout layout = BuildLayout(2u, 8u, 32u, 3u,
                                        303u, 11u, 6u, 1u);
    const CombineV2Config config = Config(2u, 32u);
    CombinePullPlan plan{};
    assert(CompileCombinePullPlan(layout, config, &plan) ==
           CombineV2Status::OK);
    std::vector<CombineRegionRegistration> registrations =
        Registrations(plan, config);
    CombineReadyV2 good = MakeReady(plan, config, registrations[0]);
    assert(ValidateCombineReady(good, registrations[0], config, plan) ==
           CombineV2Status::OK);

    CombineReadyV2 bad = good;
    ++bad.dispatch_cookie;
    Republish(&bad);
    assert(ValidateCombineReady(bad, registrations[0], config, plan) ==
           CombineV2Status::COOKIE_MISMATCH);

    bad = good;
    ++bad.generation;
    Republish(&bad);
    assert(ValidateCombineReady(bad, registrations[0], config, plan) ==
           CombineV2Status::STALE_EPOCH);

    bad = good;
    ++bad.row_count;
    bad.payload_bytes += static_cast<uint64_t>(config.hidden) * 4u;
    Republish(&bad);
    assert(ValidateCombineReady(bad, registrations[0], config, plan) ==
           CombineV2Status::INVALID_READY);

    bad = good;
    bad.source_offset = registrations[0].region_bytes;
    Republish(&bad);
    assert(ValidateCombineReady(bad, registrations[0], config, plan) ==
           CombineV2Status::CAPACITY_EXCEEDED);

    bad = good;
    ++bad.payload_bytes;
    Republish(&bad);
    assert(ValidateCombineReady(bad, registrations[0], config, plan) ==
           CombineV2Status::CAPACITY_EXCEEDED);

    bad = good;
    bad.publication ^= 1u;
    assert(ValidateCombineReady(bad, registrations[0], config, plan) ==
           CombineV2Status::INVALID_READY);

    CombineRegionRegistration invalid = registrations[0];
    invalid.region_bytes = invalid.slot_stride - 1u;
    assert(ValidateCombineRegistration(invalid, config) ==
           CombineV2Status::INVALID_REGISTRATION);

    CompiledLayout bad_journal = layout;
    bad_journal.journal_header.dispatch_cookie = 0u;
    assert(CompileCombinePullPlan(bad_journal, config, &plan) ==
           CombineV2Status::INVALID_JOURNAL);
}

void TestRandomW2ToW8()
{
    constexpr uint64_t kSeed = 0x434f4d42494e4532ull;
    std::mt19937_64 rng(kSeed);
    uint32_t iteration = 0u;
    for (uint32_t workers = 2u; workers <= 8u; ++workers) {
        for (uint32_t case_index = 0u; case_index < 150u;
             ++case_index, ++iteration) {
            const uint32_t hidden = 1u +
                static_cast<uint32_t>(rng() % 129u);
            const uint32_t tokens = static_cast<uint32_t>(rng() % 17u);
            const PartialDataType dtype = static_cast<PartialDataType>(
                rng() % 3u);
            CompiledLayout layout = BuildLayout(
                workers, 1u + static_cast<uint32_t>(rng() % 32u), hidden,
                tokens, 1000u + iteration, 5000u + iteration,
                iteration % 23u, static_cast<uint16_t>(iteration & 1u),
                &rng);
            const CombineV2Config config = Config(workers, hidden, dtype);
            CombinePullPlan plan{};
            assert(CompileCombinePullPlan(layout, config, &plan) ==
                   CombineV2Status::OK);
            std::vector<CombineRegionRegistration> registrations =
                Registrations(plan, config);
            std::shuffle(registrations.begin(), registrations.end(), rng);
            CombineV2Coordinator coordinator;
            assert(coordinator.Initialize(&layout, config, registrations) ==
                   CombineV2Status::OK);
            std::vector<uint32_t> order(workers);
            for (uint32_t rank = 0u; rank < workers; ++rank)
                order[rank] = rank;
            std::shuffle(order.begin(), order.end(), rng);
            for (uint32_t rank : order) {
                const auto registration = std::find_if(
                    registrations.begin(), registrations.end(),
                    [rank](const CombineRegionRegistration &candidate) {
                        return candidate.source_rank == rank;
                    });
                assert(registration != registrations.end());
                CombineReadyV2 ready = MakeReady(coordinator.plan(), config,
                                                  *registration);
                assert(coordinator.NotifyReady(ready) == CombineV2Status::OK);
            }
            assert(coordinator.all_ready());
            assert(coordinator.Finish() == CombineV2Status::OK);
            assert(coordinator.state() == JournalSlotState::COMPLETE);
        }
    }
}

} // namespace

int main()
{
    TestCanonicalDirectPlan();
    TestReadyAndStateMachine();
    TestCookieAndBoundsFaults();
    TestRandomW2ToW8();
    return 0;
}
