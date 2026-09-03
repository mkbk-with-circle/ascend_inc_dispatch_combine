#include "inc_dc_pull_combine_v2.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Host-side end-to-end contract harness for the device Combine V2 kernel.
// It deliberately has no ACL/NPU dependency: the test compiles a real
// Dispatch journal and Combine pull plan, validates READY records, emulates
// the indexed FP32 reduction, publishes ACK/completion records, and verifies
// every output element against an independent journal-derived oracle.

using namespace inc::dc::pull_v2;

namespace {

constexpr uint64_t kHashOffset = 1469598103934665603ull;

struct alignas(64) SourceAck {
    uint32_t magic = kPullCombineV2Magic;
    uint16_t abi_version = kPullCombineV2AbiVersion;
    uint16_t struct_bytes = sizeof(SourceAck);
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint64_t dispatch_cookie = 0u;
    uint32_t wave = 0u;
    uint32_t source_rank = 0u;
    uint32_t source_region_id = 0u;
    uint32_t status = 0u;
    uint16_t ring_slot = 0u;
    uint16_t flags = 0u;
    uint32_t row_count = 0u;
    uint64_t bytes_consumed = 0u;
    uint64_t reserved[5]{};
    uint64_t publication = 0u;
};
static_assert(sizeof(SourceAck) == 128u);
static_assert(offsetof(SourceAck, publication) + sizeof(uint64_t) ==
              sizeof(SourceAck));

struct alignas(64) OwnerCompletion {
    uint32_t magic = kPullCombineV2Magic;
    uint16_t abi_version = kPullCombineV2AbiVersion;
    uint16_t struct_bytes = sizeof(OwnerCompletion);
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint64_t dispatch_cookie = 0u;
    uint32_t wave = 0u;
    uint32_t owner_rank = 0u;
    uint32_t status = 0u;
    uint32_t row_count = 0u;
    uint16_t ring_slot = 0u;
    uint16_t flags = 0u;
    uint32_t reserved0 = 0u;
    uint64_t bytes_produced = 0u;
    uint64_t reserved[5]{};
    uint64_t publication = 0u;
};
static_assert(sizeof(OwnerCompletion) == 128u);
static_assert(offsetof(OwnerCompletion, publication) + sizeof(uint64_t) ==
              sizeof(OwnerCompletion));

void Check(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

uint64_t Align64(uint64_t value)
{
    return (value + 63u) / 64u * 64u;
}

uint64_t Publication(const void *record, size_t bytes)
{
    const auto *p = static_cast<const uint8_t *>(record);
    uint64_t hash = kHashOffset;
    for (size_t i = 0u; i < bytes; ++i) {
        const uint64_t x = hash ^ p[i];
        hash = (x << 40u) + (x << 8u) + (x << 7u) + (x << 5u) +
            (x << 4u) + (x << 1u) + x;
    }
    return hash == 0u ? 1u : hash;
}

float Partial(uint32_t source, uint32_t source_row, uint32_t element)
{
    const int32_t signed_value =
        static_cast<int32_t>((source * 131u + source_row * 17u +
                              element * 7u) % 1009u) - 504;
    return static_cast<float>(signed_value) / 32.0f;
}

ParsedSource BuildSource(const SessionConfig &session, uint32_t source,
                         uint32_t token_count, uint64_t generation,
                         uint64_t sequence, uint32_t wave,
                         uint16_t ring_slot)
{
    SourceInput input{};
    input.session = session;
    input.generation = generation;
    input.sequence = sequence;
    input.wave = wave;
    input.source_rank = source;
    input.source_region_id = 100u + source;
    input.ring_slot = ring_slot;
    input.assignment_offsets.push_back(0u);
    for (uint32_t token = 0u; token < token_count; ++token) {
        input.token_ids.push_back(token % 2u); // not a global identity
        const uint32_t first = (source + token) % session.worker_count;
        const uint32_t second = (first + 1u) % session.worker_count;
        // Two experts on first B exercise B-local reduction and must still
        // compile to one network contributor for that destination.
        input.assignments.push_back(AssignmentRecord{
            first, (token * 3u) % session.expert_count, 0u, 0.75f});
        input.assignments.push_back(AssignmentRecord{
            first, (token * 3u + 1u) % session.expert_count, 1u, 0.25f});
        input.assignments.push_back(AssignmentRecord{
            second, (token * 3u + 2u) % session.expert_count, 2u, 1.0f});
        input.assignment_offsets.push_back(input.assignments.size());
    }
    input.hidden_payload.resize(
        static_cast<size_t>(token_count) * session.hidden * 2u,
        static_cast<uint8_t>(source + 1u));

    std::vector<uint8_t> slot;
    SlotHeader header{};
    Ready dispatch_ready{};
    Check(BuildSlot(input, &slot, &header, &dispatch_ready) == Status::OK,
          "BuildSlot failed");
    RegionRegistration registration{};
    registration.session_id = session.session_id;
    registration.placement_epoch = session.placement_epoch;
    registration.source_rank = source;
    registration.region_id = input.source_region_id;
    registration.slot_count = session.ring_slots;
    registration.slot_stride = Align64(slot.size());
    registration.region_bytes =
        registration.slot_stride * registration.slot_count;
    ParsedSource parsed{};
    Check(ParseReadyAndSlot(registration, session, dispatch_ready,
                            slot.data(), slot.size(), &parsed) == Status::OK,
          "ParseReadyAndSlot failed");
    return parsed;
}

void RunCase(uint32_t workers, uint32_t hidden,
             const std::vector<uint32_t> &tokens_per_source,
             const char *label)
{
    constexpr uint64_t generation = 41u;
    constexpr uint64_t sequence = 9u;
    constexpr uint32_t wave = 3u;
    constexpr uint16_t ring_slot = 1u;
    Check(tokens_per_source.size() == workers, "bad test shape");

    SessionConfig dispatch_config{};
    dispatch_config.session_id = 0x12345000u + workers;
    dispatch_config.placement_epoch = 17u;
    dispatch_config.worker_count = workers;
    dispatch_config.expert_count = 16u;
    dispatch_config.hidden = hidden;
    dispatch_config.dtype = DataType::BF16;
    dispatch_config.ring_slots = 2u;
    std::vector<ParsedSource> sources;
    uint64_t total_tokens = 0u;
    uint64_t total_assignments = 0u;
    for (uint32_t source = 0u; source < workers; ++source) {
        sources.push_back(BuildSource(dispatch_config, source,
                                      tokens_per_source[source], generation,
                                      sequence, wave, ring_slot));
        total_tokens += sources.back().tokens.size();
        total_assignments += sources.back().assignments.size();
    }
    std::vector<uint32_t> arrival(workers);
    for (uint32_t source = 0u; source < workers; ++source)
        arrival[source] = workers - source - 1u;
    LayoutConfig layout_config{};
    layout_config.worker_count = workers;
    layout_config.expert_count = dispatch_config.expert_count;
    layout_config.destination_row_capacity =
        std::max<uint64_t>(1u, total_tokens * workers);
    layout_config.destination_assignment_capacity =
        std::max<uint64_t>(1u, total_assignments);
    CompiledLayout layout{};
    Check(CompileLayout(sources, arrival, layout_config, &layout) ==
              Status::OK,
          "CompileLayout failed");

    CombineV2Config config{};
    config.session_id = dispatch_config.session_id;
    config.placement_epoch = dispatch_config.placement_epoch;
    config.worker_count = workers;
    config.hidden = hidden;
    config.partial_dtype = PartialDataType::FP32;
    config.ring_slots = dispatch_config.ring_slots;
    CombinePullPlan plan{};
    Check(CompileCombinePullPlan(layout, config, &plan) ==
              CombineV2Status::OK,
          "CompileCombinePullPlan failed");

    std::vector<CombineRegionRegistration> registrations(workers);
    for (uint32_t source = 0u; source < workers; ++source) {
        const uint64_t bytes = static_cast<uint64_t>(
            plan.source_row_counts[source]) * hidden * sizeof(float);
        auto &registration = registrations[source];
        registration.session_id = config.session_id;
        registration.placement_epoch = config.placement_epoch;
        registration.source_rank = source;
        registration.region_id = 700u + source;
        registration.slot_count = config.ring_slots;
        registration.slot_stride = Align64(bytes + 64u);
        registration.region_bytes =
            registration.slot_stride * registration.slot_count;
    }

    CombineV2Coordinator coordinator;
    Check(coordinator.Initialize(&layout, config, registrations) ==
              CombineV2Status::OK,
          "coordinator Initialize failed");
    Check(coordinator.state() == JournalSlotState::COMBINE_ACTIVE,
          "journal did not enter COMBINE_ACTIVE");
    for (uint32_t ordinal = 0u; ordinal < workers; ++ordinal) {
        const uint32_t source = (ordinal * 3u + 1u) % workers;
        CombineReadyV2 ready{};
        ready.session_id = config.session_id;
        ready.placement_epoch = config.placement_epoch;
        ready.generation = plan.generation;
        ready.sequence = plan.sequence;
        ready.dispatch_cookie = plan.dispatch_cookie;
        ready.wave = plan.wave;
        ready.source_rank = source;
        ready.source_region_id = registrations[source].region_id;
        ready.ring_slot = plan.ring_slot;
        ready.row_count = plan.source_row_counts[source];
        ready.hidden = hidden;
        ready.partial_dtype = static_cast<uint32_t>(PartialDataType::FP32);
        ready.source_offset = static_cast<uint64_t>(plan.ring_slot) *
            registrations[source].slot_stride;
        ready.payload_bytes = static_cast<uint64_t>(ready.row_count) *
            hidden * sizeof(float);
        ready.publication = CombineReadyPublication(ready);
        Check(ValidateCombineReady(ready, registrations[source], config,
                                   plan) == CombineV2Status::OK,
              "READY validation failed");
        Check(coordinator.NotifyReady(ready) == CombineV2Status::OK,
              "READY notification failed");
    }
    Check(coordinator.all_ready(), "not all READY records were accepted");

    std::vector<std::vector<float>> partials(workers);
    for (uint32_t source = 0u; source < workers; ++source) {
        partials[source].resize(
            static_cast<size_t>(plan.source_row_counts[source]) * hidden);
        for (uint32_t row = 0u; row < plan.source_row_counts[source]; ++row)
            for (uint32_t element = 0u; element < hidden; ++element)
                partials[source][static_cast<size_t>(row) * hidden +
                                 element] = Partial(source, row, element);
    }

    std::vector<float> accumulators(
        static_cast<size_t>(plan.accumulator_count) * hidden, 0.0f);
    std::vector<uint32_t> received(plan.accumulator_count, 0u);
    for (const CombinePullOp &op : plan.pulls) {
        for (uint32_t element = 0u; element < hidden; ++element) {
            accumulators[static_cast<size_t>(op.accumulator_index) * hidden +
                         element] +=
                partials[op.source_rank][static_cast<size_t>(op.source_row) *
                                         hidden + element];
        }
        ++received[op.accumulator_index];
    }

    std::vector<std::vector<float>> output(workers);
    for (uint32_t owner = 0u; owner < workers; ++owner)
        output[owner].assign(
            static_cast<size_t>(plan.owner_offsets[owner + 1u] -
                                plan.owner_offsets[owner]) * hidden,
            0.0f);
    for (const CombineResultOp &result : plan.results) {
        Check(received[result.accumulator_index] ==
                  result.expected_contributors,
              "contributor count mismatch");
        std::copy_n(
            accumulators.begin() +
                static_cast<size_t>(result.accumulator_index) * hidden,
            hidden,
            output[result.owner_rank].begin() +
                static_cast<size_t>(result.owner_row) * hidden);
    }

    // Independent oracle walks the Dispatch journal rather than the compiled
    // pull/result arrays, catching plan index and owner placement errors.
    for (const JournalTokenEntry &token : layout.journal_tokens) {
        for (uint32_t element = 0u; element < hidden; ++element) {
            float expected = 0.0f;
            for (uint32_t i = 0u; i < token.contributors_count; ++i) {
                const JournalContributor &contributor = layout.contributors[
                    token.contributors_begin + i];
                expected += Partial(contributor.worker_rank,
                                    contributor.destination_row, element);
            }
            const float actual = output[token.owner_rank][
                static_cast<size_t>(token.owner_row) * hidden + element];
            Check(std::fabs(actual - expected) <= 1.0e-6f,
                  "FP32 oracle mismatch");
        }
    }

    std::vector<SourceAck> acks(workers);
    std::vector<OwnerCompletion> completions(workers);
    for (uint32_t source = 0u; source < workers; ++source) {
        SourceAck &ack = acks[source];
        ack.session_id = config.session_id;
        ack.placement_epoch = config.placement_epoch;
        ack.generation = plan.generation;
        ack.sequence = plan.sequence;
        ack.dispatch_cookie = plan.dispatch_cookie;
        ack.wave = plan.wave;
        ack.source_rank = source;
        ack.source_region_id = registrations[source].region_id;
        ack.ring_slot = plan.ring_slot;
        ack.row_count = plan.source_row_counts[source];
        ack.bytes_consumed = static_cast<uint64_t>(ack.row_count) * hidden *
            sizeof(float);
        ack.publication = Publication(&ack, offsetof(SourceAck, publication));
        Check(ack.publication != 0u && ack.status == 0u,
              "source ACK publication failed");

        OwnerCompletion &completion = completions[source];
        completion.session_id = config.session_id;
        completion.placement_epoch = config.placement_epoch;
        completion.generation = plan.generation;
        completion.sequence = plan.sequence;
        completion.dispatch_cookie = plan.dispatch_cookie;
        completion.wave = plan.wave;
        completion.owner_rank = source;
        completion.ring_slot = plan.ring_slot;
        completion.row_count = static_cast<uint32_t>(
            plan.owner_offsets[source + 1u] - plan.owner_offsets[source]);
        completion.bytes_produced =
            static_cast<uint64_t>(completion.row_count) * hidden *
            sizeof(float);
        completion.publication = Publication(
            &completion, offsetof(OwnerCompletion, publication));
        Check(completion.publication != 0u && completion.status == 0u,
              "owner completion publication failed");
    }

    Check(coordinator.Finish() == CombineV2Status::OK,
          "coordinator Finish failed");
    Check(coordinator.state() == JournalSlotState::COMPLETE,
          "journal did not enter COMPLETE");
    std::cout << "PASS " << label << " W" << workers
              << " hidden=" << hidden << " tokens=" << total_tokens
              << " pulls=" << plan.pulls.size() << '\n';
}

} // namespace

int main()
{
    try {
        RunCase(2u, 7u, {0u, 0u}, "zero");
        RunCase(2u, 1u, {1u, 2u}, "small");
        RunCase(2u, 1537u, {3u, 1u}, "tail");
        RunCase(4u, 7u, {0u, 0u, 0u, 0u}, "zero");
        RunCase(4u, 3u, {1u, 2u, 3u, 1u}, "small-asymmetric");
        RunCase(4u, 1537u, {3u, 1u, 4u, 2u}, "tail-asymmetric");
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
