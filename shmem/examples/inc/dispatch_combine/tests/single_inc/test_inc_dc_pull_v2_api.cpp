#include "inc_dc_pull_v2_api.h"

#include <cassert>
#include <cstdint>

using namespace inc::dc::pull_v2::api;

namespace {

struct Mock {
    uint32_t dispatches = 0u;
    uint32_t combines = 0u;
    uint32_t releases = 0u;
};

StatusCode Create(void *, const SessionConfig &) { return StatusCode::OK; }
StatusCode Destroy(void *) { return StatusCode::OK; }
StatusCode Dispatch(
    void *raw, DataType dtype, const WaveId &id, const DispatchInput *,
    DispatchOutput *, Stream, BackendTicket *batch, BackendTicket *completion)
{
    auto *mock = static_cast<Mock *>(raw);
    ++mock->dispatches;
    batch->words[0] = id.generation;
    batch->words[1] = static_cast<uint32_t>(dtype);
    completion->words[0] = 1u;
    return StatusCode::OK;
}
StatusCode Combine(
    void *raw, const WaveId &id, const BackendTicket &batch,
    const CombineInput *, CombineOutput *, Stream,
    BackendTicket *completion)
{
    if (batch.words[0] != id.generation) return StatusCode::STALE_HANDLE;
    ++static_cast<Mock *>(raw)->combines;
    completion->words[0] = 2u;
    return StatusCode::OK;
}
StatusCode Query(void *, const BackendTicket &) { return StatusCode::OK; }
StatusCode Wait(void *, const BackendTicket &, uint64_t) { return StatusCode::OK; }
StatusCode Release(void *raw, const BackendTicket &)
{
    ++static_cast<Mock *>(raw)->releases;
    return StatusCode::OK;
}

BackendOps Ops()
{
    return BackendOps{Create, Destroy, Dispatch, Combine, Query, Wait, Release};
}

SessionConfig Config(uint32_t rank = 0u)
{
    SessionConfig config{};
    config.session_id = 77u;
    config.placement_epoch = 3u;
    config.rank = rank;
    config.worker_count = 2u;
    config.inc_rank = 2u;
    config.hidden = 4u;
    config.expert_count = 4u;
    config.max_tokens_per_rank = 8u;
    config.max_assignments_per_rank = 16u;
    return config;
}

void TestWorkerLifecycle()
{
    Mock mock;
    SingleIncSession session;
    assert(single_inc_create(Config(), Ops(), &mock, &session));
    BatchHandle overflow_batch;
    Completion overflow_done;
    Status overflow = shmem_dispatch_alltoall_inc<DataType::BF16>(
        &session, WaveId{1u, 1u, 0u, 0u, 0u}, nullptr, nullptr, nullptr,
        nullptr, nullptr, std::numeric_limits<uint32_t>::max(), 2u, nullptr,
        reinterpret_cast<Stream>(1u), &overflow_batch, &overflow_done);
    assert(overflow.code == StatusCode::INVALID_ARGUMENT);
    assert(mock.dispatches == 0u && !overflow_batch.live &&
           !overflow_done.live);
    float input[8]{};
    uint64_t token_ids[2]{};
    uint32_t destinations[4]{};
    uint32_t experts[4]{};
    float weights[4]{1, 1, 1, 1};
    float expert_rows[16]{};
    uint32_t rows = 0u, assignments = 0u;
    DispatchOutput output{
        expert_rows, sizeof(expert_rows), 4u, 4u, &rows, &assignments};
    BatchHandle batch;
    Completion dispatch_done;
    WaveId wave{1u, 2u, 3u, 0u, 0u};
    assert(shmem_dispatch_alltoall_inc<DataType::BF16>(
        &session, wave, input, token_ids, destinations, experts, weights,
        2u, 2u, &output, reinterpret_cast<Stream>(1u), &batch,
        &dispatch_done));
    assert(mock.dispatches == 1u && batch.live && dispatch_done.live);
    assert(completion_query(&session, dispatch_done));
    float partials[16]{};
    float recv[8]{};
    uint64_t partial_ids[4]{};
    uint32_t recv_count = 0u;
    Completion combine_done;
    assert(shmem_combine_alltoall_inc<DataType::FP32>(
        &session, &batch, partials, 4u, partial_ids, recv, 2u,
        &recv_count, reinterpret_cast<Stream>(1u), &combine_done));
    assert(mock.combines == 1u && !batch.live && combine_done.live);
    assert(completion_wait(&session, combine_done));
    Status stale = shmem_combine_alltoall_inc<DataType::FP32>(
        &session, &batch, partials, 4u, partial_ids, recv, 2u,
        &recv_count, reinterpret_cast<Stream>(1u), &combine_done);
    assert(stale.code == StatusCode::STALE_HANDLE);
    assert(single_inc_destroy(&session));
}

void TestReleaseAndIncRank()
{
    Mock mock;
    SingleIncSession worker;
    assert(single_inc_create(Config(), Ops(), &mock, &worker));
    float input[4]{};
    uint64_t ids[1]{};
    uint32_t destinations[1]{}, experts[1]{};
    float weights[1]{1};
    float recv[4]{};
    uint32_t rows = 0u, assignments = 0u;
    DispatchOutput output{recv, sizeof(recv), 1u, 1u, &rows, &assignments};
    BatchHandle batch;
    Completion done;
    DispatchOutput too_small = output;
    too_small.recv_capacity_bytes = 1u;
    Status capacity = shmem_dispatch_alltoall_inc<DataType::FP32>(
        &worker, WaveId{8u, 8u, 0u, 0u, 0u}, input, ids, destinations,
        experts, weights, 1u, 1u, &too_small,
        reinterpret_cast<Stream>(1u), &batch, &done);
    assert(capacity.code == StatusCode::CAPACITY_EXCEEDED);
    assert(!batch.live && !done.live);
    assert(shmem_dispatch_alltoall_inc<DataType::FP32>(
        &worker, WaveId{9u, 9u, 0u, 0u, 0u}, input, ids, destinations,
        experts, weights, 1u, 1u, &output, reinterpret_cast<Stream>(1u),
        &batch, &done));
    assert(single_inc_destroy(&worker).code == StatusCode::BUSY_SLOT);
    assert(batch_release(&worker, &batch));
    assert(mock.releases == 1u);
    assert(single_inc_destroy(&worker));

    SingleIncSession inc;
    assert(single_inc_create(Config(2u), Ops(), &mock, &inc));
    BatchHandle inc_batch;
    Completion inc_done;
    assert(shmem_dispatch_alltoall_inc<DataType::BF16>(
        &inc, WaveId{10u, 10u, 0u, 0u, 0u}, nullptr, nullptr,
        reinterpret_cast<Stream>(1u), &inc_batch, &inc_done));
    assert(batch_release(&inc, &inc_batch));
    assert(single_inc_destroy(&inc));
}

} // namespace

int main()
{
    TestWorkerLifecycle();
    TestReleaseAndIncRank();
    return 0;
}
