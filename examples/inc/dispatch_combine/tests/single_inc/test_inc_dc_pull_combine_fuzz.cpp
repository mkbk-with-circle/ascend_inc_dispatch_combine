#include "inc_dc_pull_combine_dispatch.h"
#include "inc_dc_pull_combine_state.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

using namespace inc::dc::pull_combine;

namespace {

CombineReadyDescriptor Ready(const WavePlan &plan, PartialDType dtype,
                             uint32_t source, uint64_t sequence,
                             uint64_t begin, uint32_t rows)
{
    CombineReadyDescriptor descriptor{};
    descriptor.generation = plan.generation;
    descriptor.sequence = sequence;
    descriptor.wave = plan.wave;
    descriptor.source_rank = source;
    descriptor.combine_row_begin = begin;
    descriptor.row_count = rows;
    descriptor.partial_dtype = static_cast<uint32_t>(dtype);
    descriptor.source_region_id = 0x1000u + source;
    descriptor.source_offset = sequence * 4096u;
    descriptor.payload_bytes = static_cast<uint64_t>(rows) * plan.hidden *
        (dtype == PartialDType::FP32 ? 4u : 2u);
    descriptor.semantic_digest = plan.semantic_digest;
    return descriptor;
}

CountCommitDescriptor CountDescriptor(const WavePlan &plan, uint32_t source)
{
    CountCommitDescriptor descriptor{};
    descriptor.generation = plan.generation;
    descriptor.wave = plan.wave;
    descriptor.source_rank = source;
    descriptor.worker_count = plan.worker_count;
    descriptor.source_region_id = 0x2000u + source;
    descriptor.token_counts_offset = source * 4096u;
    descriptor.assignment_counts_offset = source * 4096u + 64u;
    descriptor.semantic_digest = plan.semantic_digest;
    return descriptor;
}

DispatchIngressDescriptor Ingress(const WavePlan &plan, uint32_t source,
                                  uint64_t sequence, uint32_t token_begin,
                                  uint32_t tokens)
{
    DispatchIngressDescriptor descriptor{};
    descriptor.generation = plan.generation;
    descriptor.sequence = sequence;
    descriptor.wave = plan.wave;
    descriptor.source_rank = source;
    descriptor.source_token_begin = token_begin;
    descriptor.token_count = tokens;
    descriptor.element_bytes = plan.element_bytes;
    descriptor.inc_region_id = 0x3000u + source;
    descriptor.inc_offset = sequence * 4096u;
    descriptor.payload_bytes = static_cast<uint64_t>(tokens) * plan.hidden *
        plan.element_bytes;
    descriptor.semantic_digest = plan.semantic_digest;
    return descriptor;
}

float TokenValue(uint32_t result, uint32_t h)
{
    return static_cast<float>((result + 1u) * 37u + h) * 0.25f;
}

float PartialValue(const CombineRow &row, uint32_t h)
{
    return static_cast<float>((row.source_rank + 1u) * 1000u +
                              row.result_id * 13u + h) * 0.125f;
}

} // namespace

int main(int argc, char **argv)
{
    constexpr uint64_t kDefaultSeed = 0x50434d425631ull;
    const uint64_t requested_iterations = argc > 1
        ? std::strtoull(argv[1], nullptr, 0)
        : 500u;
    const uint64_t seed = argc > 2
        ? std::strtoull(argv[2], nullptr, 0)
        : kDefaultSeed;
    if (argc > 3 || requested_iterations == 0u ||
        requested_iterations > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "usage: " << argv[0]
                  << " [positive_iterations] [seed]\n";
        return 2;
    }
    const uint32_t iterations =
        static_cast<uint32_t>(requested_iterations);
    std::mt19937_64 rng(seed);
    for (uint32_t iteration = 0u; iteration < iterations; ++iteration) {
        WavePlanDesc desc{};
        desc.worker_count = 2u + static_cast<uint32_t>(rng() % 7u);
        desc.expert_count = 8u + static_cast<uint32_t>(rng() % 57u);
        desc.hidden = 1u + static_cast<uint32_t>(rng() % 17u);
        desc.element_bytes = 2u;
        desc.wave = iteration % 11u;
        desc.generation = 1000u + iteration;
        desc.tokens_per_rank.resize(desc.worker_count);
        for (uint32_t &tokens : desc.tokens_per_rank)
            tokens = static_cast<uint32_t>(rng() % 7u);

        for (uint32_t source = 0u; source < desc.worker_count; ++source) {
            for (uint32_t token = 0u;
                 token < desc.tokens_per_rank[source]; ++token) {
                TokenRoute route{};
                route.source_rank = source;
                route.source_token = token;
                const uint32_t topk = static_cast<uint32_t>(rng() % 13u);
                route.assignments.reserve(topk);
                for (uint32_t k = 0u; k < topk; ++k) {
                    RouteAssignment assignment{};
                    assignment.destination_rank = static_cast<uint32_t>(
                        rng() % desc.worker_count);
                    assignment.expert_id = static_cast<uint32_t>(
                        rng() % desc.expert_count);
                    assignment.ordinal = k;
                    assignment.weight =
                        (static_cast<int32_t>(rng() % 2001u) - 1000) /
                        1000.0f;
                    route.assignments.push_back(assignment);
                }
                std::shuffle(route.assignments.begin(),
                             route.assignments.end(), rng);
                desc.routes.push_back(std::move(route));
            }
        }
        std::shuffle(desc.routes.begin(), desc.routes.end(), rng);

        WavePlan plan{};
        assert(BuildWavePlan(desc, &plan) == PlanStatus::OK);

        // Exercise the same random plan through count transpose and Dispatch
        // before feeding its inverse rows to Combine.
        PullDispatchCoordinator dispatch;
        assert(dispatch.Initialize(plan, 16u) == DispatchStatus::OK);
        std::vector<uint32_t> source_order(desc.worker_count);
        std::iota(source_order.begin(), source_order.end(), 0u);
        std::shuffle(source_order.begin(), source_order.end(), rng);
        for (uint32_t source : source_order) {
            std::vector<uint64_t> token_counts(desc.worker_count);
            std::vector<uint64_t> assignment_counts(desc.worker_count);
            for (uint32_t destination = 0u;
                 destination < desc.worker_count; ++destination) {
                const size_t index = static_cast<size_t>(source) *
                    desc.worker_count + destination;
                token_counts[destination] = plan.token_row_counts[index];
                assignment_counts[destination] =
                    plan.assignment_counts[index];
            }
            assert(dispatch.CommitCounts(CountDescriptor(plan, source),
                                         token_counts, assignment_counts) ==
                   DispatchStatus::OK);
        }
        assert(dispatch.counts_ready());
        for (uint32_t destination = 0u;
             destination < desc.worker_count; ++destination) {
            CountReply reply{};
            assert(dispatch.PopCountReply(destination, &reply) ==
                   DispatchStatus::OK);
        }

        std::vector<uint32_t> token_cursor(desc.worker_count, 0u);
        std::vector<uint64_t> dispatch_sequence(desc.worker_count, 1u);
        std::vector<uint64_t> dispatch_descriptor_count(
            desc.worker_count, 0u);
        std::vector<uint32_t> ingress_active;
        for (uint32_t source = 0u; source < desc.worker_count; ++source) {
            if (desc.tokens_per_rank[source] != 0u)
                ingress_active.push_back(source);
        }
        while (!ingress_active.empty()) {
            const size_t pick = static_cast<size_t>(
                rng() % ingress_active.size());
            const uint32_t source = ingress_active[pick];
            const uint32_t tokens = std::min<uint32_t>(
                1u + static_cast<uint32_t>(rng() % 3u),
                desc.tokens_per_rank[source] - token_cursor[source]);
            std::vector<float> hidden_values;
            hidden_values.reserve(static_cast<size_t>(tokens) * plan.hidden);
            for (uint32_t local = token_cursor[source];
                 local < token_cursor[source] + tokens; ++local) {
                const uint32_t result = static_cast<uint32_t>(
                    plan.result_rank_offsets[source] + local);
                for (uint32_t h = 0u; h < plan.hidden; ++h)
                    hidden_values.push_back(TokenValue(result, h));
            }
            assert(dispatch.NotifyIngress(
                       Ingress(plan, source, dispatch_sequence[source],
                               token_cursor[source], tokens), hidden_values) ==
                   DispatchStatus::OK);
            ++dispatch_sequence[source];
            ++dispatch_descriptor_count[source];
            token_cursor[source] += tokens;
            if (token_cursor[source] == desc.tokens_per_rank[source])
                ingress_active.erase(ingress_active.begin() + pick);
        }

        std::vector<uint8_t> slot_seen(plan.route_slots.size(), 0u);
        uint64_t dispatch_rows = 0u;
        for (uint32_t destination = 0u;
             destination < desc.worker_count; ++destination) {
            for (;;) {
                DispatchFanoutChunk chunk{};
                const DispatchStatus status = dispatch.PopFanoutChunk(
                    destination,
                    1u + static_cast<uint32_t>(rng() % 5u), &chunk);
                if (status == DispatchStatus::NOT_READY) break;
                assert(status == DispatchStatus::OK);
                assert(chunk.assignment_offsets.size() ==
                       chunk.route_slot_ids.size() + 1u);
                for (size_t row = 0u;
                     row < chunk.route_slot_ids.size(); ++row) {
                    const uint32_t slot_id = chunk.route_slot_ids[row];
                    assert(slot_id < plan.route_slots.size());
                    assert(slot_seen[slot_id] == 0u);
                    slot_seen[slot_id] = 1u;
                    const RouteSlot &slot = plan.route_slots[slot_id];
                    for (uint32_t h = 0u; h < plan.hidden; ++h) {
                        assert(std::fabs(chunk.hidden_values[
                            row * plan.hidden + h] -
                            TokenValue(slot.result_id, h)) < 1e-5f);
                    }
                    assert(chunk.assignment_offsets[row + 1u] -
                               chunk.assignment_offsets[row] ==
                           slot.assignment_count);
                    for (uint32_t i = 0u; i < slot.assignment_count; ++i) {
                        const PackedAssignment &actual = chunk.assignments[
                            chunk.assignment_offsets[row] + i];
                        const PackedAssignment &expected_assignment =
                            plan.assignments[slot.assignment_begin + i];
                        assert(actual.route_slot ==
                               expected_assignment.route_slot);
                        assert(actual.expert_id ==
                               expected_assignment.expert_id);
                        assert(actual.ordinal == expected_assignment.ordinal);
                        assert(actual.weight == expected_assignment.weight);
                    }
                    ++dispatch_rows;
                }
            }
        }
        assert(dispatch_rows == plan.route_slots.size());
        for (uint32_t source = 0u; source < desc.worker_count; ++source) {
            for (uint64_t expected_sequence = 1u;
                 expected_sequence <= dispatch_descriptor_count[source];
                 ++expected_sequence) {
                DispatchAck ack{};
                assert(dispatch.PopAck(source, &ack) == DispatchStatus::OK);
                assert(ack.sequence == expected_sequence);
                assert(ack.status == static_cast<uint32_t>(
                    DispatchStatus::OK));
            }
        }
        assert(dispatch.complete());

        const PartialDType dtype = iteration % 2u == 0u
            ? PartialDType::FP32 : PartialDType::BF16;
        PullCombineCoordinator coordinator;
        assert(coordinator.Initialize(plan, 4u, dtype) ==
               CoordinatorStatus::OK);

        const uint32_t result_count = static_cast<uint32_t>(
            plan.expected_contributors.size());
        std::vector<float> expected(
            static_cast<size_t>(result_count) * plan.hidden, 0.0f);
        for (const CombineRow &row : plan.combine_rows) {
            for (uint32_t h = 0u; h < plan.hidden; ++h) {
                expected[static_cast<size_t>(row.result_id) * plan.hidden + h]
                    += PartialValue(row, h);
            }
        }

        std::vector<uint64_t> cursor = plan.combine_source_offsets;
        std::vector<uint64_t> sequence(desc.worker_count, 1u);
        std::vector<uint32_t> active;
        for (uint32_t source = 0u; source < desc.worker_count; ++source) {
            if (cursor[source] < plan.combine_source_offsets[source + 1u])
                active.push_back(source);
        }
        while (!active.empty()) {
            const size_t pick = static_cast<size_t>(rng() % active.size());
            const uint32_t source = active[pick];
            const uint64_t end = plan.combine_source_offsets[source + 1u];
            const uint32_t rows = static_cast<uint32_t>(std::min<uint64_t>(
                1u + rng() % 3u, end - cursor[source]));
            const CombineReadyDescriptor descriptor = Ready(
                plan, dtype, source, sequence[source], cursor[source], rows);
            assert(coordinator.Notify(descriptor) == CoordinatorStatus::OK);
            FetchTask task{};
            assert(coordinator.BeginFetch(source, &task) ==
                   CoordinatorStatus::OK);
            std::vector<float> partials;
            partials.reserve(static_cast<size_t>(rows) * plan.hidden);
            for (uint64_t packed = cursor[source];
                 packed < cursor[source] + rows; ++packed) {
                const CombineRow &row = plan.combine_rows[packed];
                for (uint32_t h = 0u; h < plan.hidden; ++h)
                    partials.push_back(PartialValue(row, h));
            }
            assert(coordinator.CompleteFetch(
                       source, sequence[source], partials) ==
                   CoordinatorStatus::OK);
            CombineAck ack{};
            assert(coordinator.PopAck(source, &ack) == CoordinatorStatus::OK);
            assert(ack.sequence == sequence[source]);
            ++sequence[source];
            cursor[source] += rows;
            if (cursor[source] == end)
                active.erase(active.begin() + pick);
        }

        std::vector<float> actual(expected.size(), 0.0f);
        uint64_t drained = 0u;
        for (uint32_t destination = 0u;
             destination < desc.worker_count; ++destination) {
            for (;;) {
                EgressChunk chunk{};
                const CoordinatorStatus status = coordinator.PopEgressChunk(
                    destination, 1u + static_cast<uint32_t>(rng() % 5u),
                    &chunk);
                if (status == CoordinatorStatus::NOT_READY) break;
                assert(status == CoordinatorStatus::OK);
                for (size_t row = 0u; row < chunk.result_ids.size(); ++row) {
                    const uint32_t result = chunk.result_ids[row];
                    for (uint32_t h = 0u; h < plan.hidden; ++h) {
                        actual[static_cast<size_t>(result) * plan.hidden + h]
                            = chunk.values[row * plan.hidden + h];
                    }
                    ++drained;
                }
            }
        }
        assert(drained == result_count);
        assert(coordinator.complete());
        assert(actual.size() == expected.size());
        for (size_t i = 0u; i < actual.size(); ++i)
            assert(std::fabs(actual[i] - expected[i]) < 1e-5f);
    }
    std::cout << "[PASS] randomized waves=" << iterations
              << " seed=" << seed << '\n';
    return 0;
}
