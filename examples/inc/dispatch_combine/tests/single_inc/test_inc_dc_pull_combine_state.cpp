#include "inc_dc_pull_combine_state.h"

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

CombineReadyDescriptor Ready(const WavePlan &plan, uint32_t source,
                             uint64_t sequence, uint64_t row_begin,
                             uint32_t rows)
{
    CombineReadyDescriptor desc{};
    desc.generation = plan.generation;
    desc.sequence = sequence;
    desc.wave = plan.wave;
    desc.source_rank = source;
    desc.combine_row_begin = row_begin;
    desc.row_count = rows;
    desc.partial_dtype = static_cast<uint32_t>(PartialDType::FP32);
    desc.source_region_id = 100u + source;
    desc.source_offset = sequence * 4096u;
    desc.payload_bytes = static_cast<uint64_t>(rows) * plan.hidden *
                         sizeof(float);
    desc.semantic_digest = plan.semantic_digest;
    return desc;
}

void ExpectValues(const EgressChunk &chunk,
                  const std::vector<float> &expected)
{
    assert(chunk.values.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
        assert(std::fabs(chunk.values[i] - expected[i]) < 1e-6f);
}

} // namespace

int main()
{
    const WavePlan plan = BuildPlan();

    // Empty waves are legal and complete immediately.  They are common when
    // dynamic routing leaves a token wave empty on every rank.
    WavePlanDesc empty_desc{};
    empty_desc.worker_count = 2u;
    empty_desc.expert_count = 4u;
    empty_desc.hidden = 8u;
    empty_desc.element_bytes = 2u;
    empty_desc.wave = 0u;
    empty_desc.generation = 1u;
    empty_desc.tokens_per_rank = {0u, 0u};
    WavePlan empty_plan{};
    assert(BuildWavePlan(empty_desc, &empty_plan) == PlanStatus::OK);
    PullCombineCoordinator empty;
    assert(empty.Initialize(empty_plan, 2u, PartialDType::BF16) ==
           CoordinatorStatus::OK);
    assert(empty.complete());
    assert(empty.terminal_status() == CoordinatorStatus::OK);

    PullCombineCoordinator coordinator;
    assert(coordinator.Initialize(plan, 2u, PartialDType::FP32) ==
           CoordinatorStatus::OK);

    CombineReadyDescriptor stale = Ready(plan, 1u, 1u, 1u, 1u);
    --stale.generation;
    assert(coordinator.Notify(stale) == CoordinatorStatus::STALE_GENERATION);
    CombineReadyDescriptor bad_digest = Ready(plan, 1u, 1u, 1u, 1u);
    ++bad_digest.semantic_digest;
    assert(coordinator.Notify(bad_digest) ==
           CoordinatorStatus::DIGEST_MISMATCH);
    CombineReadyDescriptor nonzero_reserved =
        Ready(plan, 1u, 1u, 1u, 1u);
    nonzero_reserved.reserved[2] = 1u;
    assert(coordinator.Notify(nonzero_reserved) ==
           CoordinatorStatus::INVALID_DESCRIPTOR);

    // B2 owns three packed rows.  A depth-2 SPSC ring rejects the third
    // publication until the first ACK is consumed by B.
    assert(coordinator.Notify(Ready(plan, 2u, 1u, 2u, 1u)) ==
           CoordinatorStatus::OK);
    assert(coordinator.Notify(Ready(plan, 2u, 2u, 3u, 1u)) ==
           CoordinatorStatus::OK);
    assert(coordinator.Notify(Ready(plan, 2u, 3u, 4u, 1u)) ==
           CoordinatorStatus::RING_FULL);

    FetchTask task{};
    assert(coordinator.BeginFetch(2u, &task) == CoordinatorStatus::OK);
    assert(coordinator.BeginFetch(2u, &task) == CoordinatorStatus::PEER_BUSY);
    assert(coordinator.CompleteFetch(2u, 1u, {100.0f, 200.0f}) ==
           CoordinatorStatus::OK);
    // ACK publication alone protects the buffer, but the host model requires
    // the producer to observe/pop it before reusing the physical ring slot.
    assert(coordinator.Notify(Ready(plan, 2u, 3u, 4u, 1u)) ==
           CoordinatorStatus::RING_FULL);
    CombineAck ack{};
    assert(coordinator.PopAck(2u, &ack) == CoordinatorStatus::OK);
    assert(ack.sequence == 1u && ack.rows_consumed == 1u);
    assert(coordinator.Notify(Ready(plan, 2u, 3u, 4u, 1u)) ==
           CoordinatorStatus::OK);

    assert(coordinator.BeginFetch(2u, &task) == CoordinatorStatus::OK);
    assert(coordinator.CompleteFetch(2u, 2u, {3.0f, 4.0f}) ==
           CoordinatorStatus::OK);
    assert(coordinator.PopAck(2u, &ack) == CoordinatorStatus::OK);
    assert(coordinator.BeginFetch(2u, &task) == CoordinatorStatus::OK);
    assert(coordinator.CompleteFetch(2u, 3u, {5.0f, 6.0f}) ==
           CoordinatorStatus::OK);
    assert(coordinator.PopAck(2u, &ack) == CoordinatorStatus::OK);

    // B1 arrives later.  Result A0/token0 becomes ready immediately after
    // this GET completion; the zero-contributor A0/token1 was ready at init.
    assert(coordinator.Notify(Ready(plan, 1u, 1u, 1u, 1u)) ==
           CoordinatorStatus::OK);
    assert(coordinator.BeginFetch(1u, &task) == CoordinatorStatus::OK);
    assert(coordinator.CompleteFetch(1u, 1u, {10.0f, 20.0f}) ==
           CoordinatorStatus::OK);
    assert(coordinator.PopAck(1u, &ack) == CoordinatorStatus::OK);

    EgressChunk egress{};
    assert(coordinator.PopEgressChunk(0u, 2u, &egress) ==
           CoordinatorStatus::OK);
    assert(egress.destination_row_begin == 0u);
    assert((egress.result_ids == std::vector<uint32_t>{0u, 1u}));
    ExpectValues(egress, {110.0f, 220.0f, 0.0f, 0.0f});
    assert(coordinator.PopEgressChunk(1u, 4u, &egress) ==
           CoordinatorStatus::OK);
    ExpectValues(egress, {3.0f, 4.0f});

    // B0 is the globally last source, but only A3 waits for it.  Other
    // destinations have already received useful output.
    assert(coordinator.Notify(Ready(plan, 0u, 1u, 0u, 1u)) ==
           CoordinatorStatus::OK);
    assert(coordinator.BeginFetch(0u, &task) == CoordinatorStatus::OK);
    assert(coordinator.CompleteFetch(0u, 1u, {1.0f, 2.0f}) ==
           CoordinatorStatus::OK);
    assert(coordinator.PopAck(0u, &ack) == CoordinatorStatus::OK);
    assert(coordinator.PopEgressChunk(3u, 1u, &egress) ==
           CoordinatorStatus::OK);
    ExpectValues(egress, {6.0f, 8.0f});
    assert(coordinator.complete());
    assert(coordinator.completed_rows() == 5u);
    assert(coordinator.ready_results() == 4u);
    assert(coordinator.sent_results() == 4u);

    // Wrong GET completion size aborts the wave and all subsequent calls
    // fail closed instead of exposing a partial result.
    PullCombineCoordinator broken;
    assert(broken.Initialize(plan, 2u, PartialDType::FP32) ==
           CoordinatorStatus::OK);
    assert(broken.Notify(Ready(plan, 1u, 1u, 1u, 1u)) ==
           CoordinatorStatus::OK);
    assert(broken.Notify(Ready(plan, 1u, 2u, 1u, 1u)) ==
           CoordinatorStatus::DUPLICATE_ROW);
    assert(broken.BeginFetch(1u, &task) == CoordinatorStatus::OK);
    assert(broken.CompleteFetch(1u, 1u, {1.0f}) ==
           CoordinatorStatus::ABORTED);
    assert(broken.aborted());
    assert(broken.terminal_status() ==
           CoordinatorStatus::PAYLOAD_SIZE_MISMATCH);
    assert(broken.Notify(Ready(plan, 0u, 1u, 0u, 1u)) ==
           CoordinatorStatus::ABORTED);
    // The accepted in-flight descriptor still receives a negative ACK, so B
    // can safely release its registered send slot after a generation abort.
    assert(broken.PopAck(1u, &ack) == CoordinatorStatus::OK);
    assert(ack.sequence == 1u && ack.rows_consumed == 0u);
    assert(ack.status == static_cast<uint32_t>(
        CoordinatorStatus::PAYLOAD_SIZE_MISMATCH));

    // Initialize owns its plan rather than retaining caller storage.
    PullCombineCoordinator owned;
    WavePlan temporary = BuildPlan();
    assert(owned.Initialize(temporary, 2u, PartialDType::FP32) ==
           CoordinatorStatus::OK);
    temporary = {};
    assert(owned.Notify(Ready(plan, 0u, 1u, 0u, 1u)) ==
           CoordinatorStatus::OK);
    return 0;
}
