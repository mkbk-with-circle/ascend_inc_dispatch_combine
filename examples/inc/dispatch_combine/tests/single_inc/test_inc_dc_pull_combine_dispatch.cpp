#include "inc_dc_pull_combine_dispatch.h"

#include <cassert>
#include <cmath>
#include <vector>

using namespace inc::dc::pull_combine;

namespace {

WavePlan BuildPlan()
{
    WavePlanDesc desc{};
    desc.worker_count = 4u;
    desc.expert_count = 16u;
    desc.hidden = 2u;
    desc.element_bytes = 2u;
    desc.wave = 5u;
    desc.generation = 29u;
    desc.tokens_per_rank = {2u, 1u, 0u, 1u};
    desc.routes = {
        {0u, 0u, {{1u, 0u, 0u, 0.5f}, {2u, 1u, 1u, 0.5f}}},
        {0u, 1u, {}},
        {1u, 0u, {{2u, 2u, 0u, 1.0f}}},
        {3u, 0u, {{0u, 3u, 0u, 0.5f}, {2u, 4u, 1u, 0.5f}}},
    };
    WavePlan plan{};
    assert(BuildWavePlan(desc, &plan) == PlanStatus::OK);
    return plan;
}

CountCommitDescriptor CountDescriptor(const WavePlan &plan, uint32_t source)
{
    CountCommitDescriptor descriptor{};
    descriptor.generation = plan.generation;
    descriptor.wave = plan.wave;
    descriptor.source_rank = source;
    descriptor.worker_count = plan.worker_count;
    descriptor.source_region_id = 100u + source;
    descriptor.token_counts_offset = source * 4096u;
    descriptor.assignment_counts_offset = source * 4096u + 64u;
    descriptor.semantic_digest = plan.semantic_digest;
    return descriptor;
}

DispatchIngressDescriptor Ingress(const WavePlan &plan, uint32_t source,
                                  uint64_t sequence, uint32_t token_begin,
                                  uint32_t token_count)
{
    DispatchIngressDescriptor descriptor{};
    descriptor.generation = plan.generation;
    descriptor.sequence = sequence;
    descriptor.wave = plan.wave;
    descriptor.source_rank = source;
    descriptor.source_token_begin = token_begin;
    descriptor.token_count = token_count;
    descriptor.element_bytes = plan.element_bytes;
    descriptor.inc_region_id = 1000u + source;
    descriptor.inc_offset = sequence * 4096u;
    descriptor.payload_bytes = static_cast<uint64_t>(token_count) *
        plan.hidden * plan.element_bytes;
    descriptor.semantic_digest = plan.semantic_digest;
    return descriptor;
}

void ExpectValues(const DispatchFanoutChunk &chunk,
                  const std::vector<float> &expected)
{
    assert(chunk.hidden_values.size() == expected.size());
    for (size_t i = 0u; i < expected.size(); ++i)
        assert(std::fabs(chunk.hidden_values[i] - expected[i]) < 1e-6f);
}

} // namespace

int main()
{
    const WavePlan plan = BuildPlan();
    PullDispatchCoordinator dispatch;
    assert(dispatch.Initialize(plan, 2u) == DispatchStatus::OK);

    // No payload may enter the wave before the all-worker count transpose.
    assert(dispatch.NotifyIngress(Ingress(plan, 0u, 1u, 0u, 2u),
                                  {1.0f, 2.0f, 3.0f, 4.0f}) ==
           DispatchStatus::COUNTS_NOT_READY);

    for (uint32_t source = 0u; source < plan.worker_count; ++source) {
        std::vector<uint64_t> token_counts(plan.worker_count);
        std::vector<uint64_t> assignment_counts(plan.worker_count);
        for (uint32_t destination = 0u;
             destination < plan.worker_count; ++destination) {
            const size_t index = static_cast<size_t>(source) *
                plan.worker_count + destination;
            token_counts[destination] = plan.token_row_counts[index];
            assignment_counts[destination] = plan.assignment_counts[index];
        }
        if (source == 0u) {
            std::vector<uint64_t> bad = token_counts;
            ++bad[1];
            assert(dispatch.CommitCounts(CountDescriptor(plan, source), bad,
                                         assignment_counts) ==
                   DispatchStatus::COUNT_MISMATCH);
        }
        assert(dispatch.CommitCounts(CountDescriptor(plan, source),
                                     token_counts, assignment_counts) ==
               DispatchStatus::OK);
    }
    assert(dispatch.counts_ready());

    CountReply reply{};
    assert(dispatch.PopCountReply(2u, &reply) == DispatchStatus::OK);
    assert((reply.token_rows_from_source ==
            std::vector<uint64_t>{1u, 1u, 0u, 1u}));
    assert((reply.receive_row_offsets ==
            std::vector<uint64_t>{0u, 1u, 2u, 2u}));
    assert(dispatch.PopCountReply(2u, &reply) == DispatchStatus::NOT_READY);

    // Source 3 is allowed to arrive first.  B0 can consume it immediately;
    // B2 can also skip its missing rows 0..1 and consume ready row 2.
    assert(dispatch.NotifyIngress(Ingress(plan, 3u, 1u, 0u, 1u),
                                  {7.0f, 8.0f}) == DispatchStatus::OK);
    DispatchFanoutChunk chunk{};
    assert(dispatch.PopFanoutChunk(0u, 4u, &chunk) == DispatchStatus::OK);
    assert(chunk.destination_row_begin == 0u);
    ExpectValues(chunk, {7.0f, 8.0f});
    DispatchAck ack{};
    assert(dispatch.PopAck(3u, &ack) == DispatchStatus::NOT_READY);
    assert(dispatch.PopFanoutChunk(2u, 4u, &chunk) == DispatchStatus::OK);
    assert(chunk.destination_row_begin == 2u);
    ExpectValues(chunk, {7.0f, 8.0f});
    assert(chunk.assignment_offsets == std::vector<uint32_t>({0u, 1u}));
    assert(chunk.assignments[0].expert_id == 4u);
    assert(dispatch.PopAck(3u, &ack) == DispatchStatus::OK);
    assert(ack.sequence == 1u && ack.tokens_consumed == 1u);

    assert(dispatch.NotifyIngress(Ingress(plan, 0u, 1u, 0u, 2u),
                                  {1.0f, 2.0f, 3.0f, 4.0f}) ==
           DispatchStatus::OK);
    assert(dispatch.PopFanoutChunk(1u, 4u, &chunk) == DispatchStatus::OK);
    ExpectValues(chunk, {1.0f, 2.0f});
    assert(dispatch.PopAck(0u, &ack) == DispatchStatus::NOT_READY);
    assert(dispatch.PopFanoutChunk(2u, 1u, &chunk) == DispatchStatus::OK);
    assert(chunk.destination_row_begin == 0u);
    ExpectValues(chunk, {1.0f, 2.0f});
    // The second source-0 token has no route.  It adds no fan-out dependency,
    // so the descriptor ACK is now safe.
    assert(dispatch.PopAck(0u, &ack) == DispatchStatus::OK);
    assert(ack.tokens_consumed == 2u);

    assert(dispatch.NotifyIngress(Ingress(plan, 1u, 1u, 0u, 1u),
                                  {5.0f, 6.0f}) == DispatchStatus::OK);
    assert(dispatch.PopFanoutChunk(2u, 4u, &chunk) == DispatchStatus::OK);
    assert(chunk.destination_row_begin == 1u);
    ExpectValues(chunk, {5.0f, 6.0f});
    assert(dispatch.PopAck(1u, &ack) == DispatchStatus::OK);
    assert(dispatch.complete());
    assert(dispatch.received_tokens() == 4u);
    assert(dispatch.sent_route_slots() == 5u);
    assert(dispatch.terminal_status() == DispatchStatus::OK);

    // Fail-closed aborts also release accepted ingress slots with negative
    // ACKs, matching the Combine-side buffer-lifetime rule.
    PullDispatchCoordinator aborted;
    assert(aborted.Initialize(plan, 2u) == DispatchStatus::OK);
    for (uint32_t source = 0u; source < plan.worker_count; ++source) {
        std::vector<uint64_t> tokens(plan.worker_count);
        std::vector<uint64_t> assignments(plan.worker_count);
        for (uint32_t destination = 0u;
             destination < plan.worker_count; ++destination) {
            const size_t index = static_cast<size_t>(source) *
                plan.worker_count + destination;
            tokens[destination] = plan.token_row_counts[index];
            assignments[destination] = plan.assignment_counts[index];
        }
        assert(aborted.CommitCounts(CountDescriptor(plan, source), tokens,
                                    assignments) == DispatchStatus::OK);
    }
    assert(aborted.NotifyIngress(Ingress(plan, 0u, 1u, 0u, 2u),
                                  {1.0f, 2.0f, 3.0f, 4.0f}) ==
           DispatchStatus::OK);
    assert(aborted.Abort(DispatchStatus::PAYLOAD_SIZE_MISMATCH) ==
           DispatchStatus::ABORTED);
    assert(aborted.PopAck(0u, &ack) == DispatchStatus::OK);
    assert(ack.tokens_consumed == 0u);
    assert(ack.status == static_cast<uint32_t>(
        DispatchStatus::PAYLOAD_SIZE_MISMATCH));
    return 0;
}
