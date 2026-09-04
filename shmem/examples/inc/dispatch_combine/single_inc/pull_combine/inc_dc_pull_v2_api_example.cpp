/*
 * Pull V2 shortest API example.
 *
 * This example uses an in-process reference backend so it can show the full
 * application lifecycle without reserving NPUs. The application-facing calls
 * are identical for a native backend:
 *
 *   create -> dispatch -> local expert compute/reduce -> combine -> destroy
 */

#include "inc_dc_pull_v2_api.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace inc::dc::pull_v2::api;

namespace {

struct Route {
    uint32_t token;
    uint32_t destination;
    uint32_t expert;
    float weight;
};

struct ReferenceBackend {
    SessionConfig config{};
    std::vector<Route> routes;
    uint64_t next_ticket = 1u;
};

StatusCode Create(void *raw, const SessionConfig &config)
{
    static_cast<ReferenceBackend *>(raw)->config = config;
    return StatusCode::OK;
}

StatusCode Destroy(void *) { return StatusCode::OK; }

StatusCode Dispatch(
    void *raw, DataType dtype, const WaveId &id, const DispatchInput *input,
    DispatchOutput *output, Stream, BackendTicket *batch,
    BackendTicket *completion)
{
    auto *backend = static_cast<ReferenceBackend *>(raw);
    if (dtype != DataType::FP32 || input == nullptr || output == nullptr)
        return StatusCode::UNSUPPORTED;
    const auto *tokens = static_cast<const float *>(input->send_buffer);
    auto *expert_rows = static_cast<float *>(output->recv_buffer);
    backend->routes.clear();
    uint32_t row = 0u;
    for (uint32_t token = 0u; token < input->token_count; ++token) {
        const uint32_t begin = input->fixed_topk == 0u
            ? input->assignment_offsets[token]
            : token * input->fixed_topk;
        const uint32_t end = input->fixed_topk == 0u
            ? input->assignment_offsets[token + 1u]
            : begin + input->fixed_topk;
        for (uint32_t assignment = begin; assignment < end; ++assignment) {
            backend->routes.push_back(Route{
                token, input->destination_gpus[assignment],
                input->expert_ids[assignment],
                input->expert_weights[assignment]});
            std::copy_n(
                tokens + static_cast<uint64_t>(token) * backend->config.hidden,
                backend->config.hidden,
                expert_rows + static_cast<uint64_t>(row) *
                    backend->config.hidden);
            ++row;
        }
    }
    *output->recv_row_count = row;
    *output->recv_assignment_count = input->assignment_count;
    batch->words[0] = id.generation;
    completion->words[0] = backend->next_ticket++;
    return StatusCode::OK;
}

StatusCode Combine(
    void *raw, const WaveId &id, const BackendTicket &batch,
    const CombineInput *input, CombineOutput *output, Stream,
    BackendTicket *completion)
{
    auto *backend = static_cast<ReferenceBackend *>(raw);
    if (input == nullptr || output == nullptr ||
        batch.words[0] != id.generation ||
        input->send_count != backend->routes.size())
        return StatusCode::INVALID_ARGUMENT;
    std::fill_n(
        output->recv_buffer,
        static_cast<uint64_t>(output->recv_capacity) * backend->config.hidden,
        0.0f);
    for (uint32_t row = 0u; row < input->send_count; ++row) {
        const Route &route = backend->routes[row];
        for (uint32_t h = 0u; h < backend->config.hidden; ++h) {
            output->recv_buffer[
                static_cast<uint64_t>(route.token) * backend->config.hidden +
                h] += input->send_buffer[
                    static_cast<uint64_t>(row) * backend->config.hidden + h] *
                route.weight;
        }
    }
    *output->recv_count = output->recv_capacity;
    completion->words[0] = backend->next_ticket++;
    return StatusCode::OK;
}

StatusCode Query(void *, const BackendTicket &) { return StatusCode::OK; }
StatusCode Wait(void *, const BackendTicket &, uint64_t) { return StatusCode::OK; }
StatusCode Release(void *, const BackendTicket &) { return StatusCode::OK; }

BackendOps Ops()
{
    BackendOps ops{};
    ops.create = Create;
    ops.destroy = Destroy;
    ops.dispatch = Dispatch;
    ops.combine = Combine;
    ops.query = Query;
    ops.wait = Wait;
    ops.release_batch = Release;
    return ops;
}

bool Check(Status status, const char *where)
{
    if (status) return true;
    std::cerr << where << ": " << status.message << '\n';
    return false;
}

} // namespace

int main()
{
    constexpr uint32_t kTokens = 2u;
    constexpr uint32_t kHidden = 4u;
    constexpr uint32_t kTopK = 2u;
    constexpr uint32_t kRows = kTokens * kTopK;

    ReferenceBackend backend;
    SingleIncSession session;
    SessionConfig config{};
    config.session_id = 1001u;
    config.placement_epoch = 1u;
    config.rank = 0u;
    config.worker_count = 2u;
    config.inc_rank = 2u;
    config.hidden = kHidden;
    config.expert_count = 4u;
    config.max_tokens_per_rank = 16u;
    config.max_assignments_per_rank = 32u;
    if (!Check(single_inc_create(config, Ops(), &backend, &session), "create"))
        return 1;
    std::cout << "[1/5] session initialized\n";

    std::array<float, kTokens * kHidden> token_input{
        1, 2, 3, 4,
        10, 20, 30, 40};
    std::array<uint64_t, kTokens> token_ids{101u, 102u};
    std::array<uint32_t, kRows> destinations{0u, 1u, 1u, 0u};
    std::array<uint32_t, kRows> experts{0u, 2u, 3u, 1u};
    std::array<float, kRows> weights{0.25f, 0.75f, 0.5f, 0.5f};
    std::array<float, kRows * kHidden> expert_input{};
    uint32_t recv_rows = 0u;
    uint32_t recv_assignments = 0u;
    DispatchOutput dispatch_output{};
    dispatch_output.recv_buffer = expert_input.data();
    dispatch_output.recv_capacity_bytes = sizeof(expert_input);
    dispatch_output.row_capacity = kRows;
    dispatch_output.assignment_capacity = kRows;
    dispatch_output.recv_row_count = &recv_rows;
    dispatch_output.recv_assignment_count = &recv_assignments;

    WaveId wave{1u, 1u, 0u, 0u, 0u};
    BatchHandle batch;
    Completion dispatch_done;
    Stream stream = reinterpret_cast<Stream>(1u);
    Status status = shmem_dispatch_alltoall_inc<DataType::FP32>(
        &session, wave, token_input.data(), token_ids.data(),
        destinations.data(), experts.data(), weights.data(), kTokens, kTopK,
        &dispatch_output, stream, &batch, &dispatch_done);
    if (!Check(status, "dispatch") ||
        !Check(completion_wait(&session, dispatch_done), "dispatch wait"))
        return 1;
    std::cout << "[2/5] dispatch rows=" << recv_rows << '\n';

    // Stand-in for grouped GEMM + same-GPU local weighted reduction.
    std::array<float, kRows * kHidden> local_partials{};
    for (uint32_t row = 0u; row < kRows; ++row) {
        const float expert_scale = static_cast<float>(experts[row] + 1u);
        for (uint32_t h = 0u; h < kHidden; ++h)
            local_partials[row * kHidden + h] =
                expert_input[row * kHidden + h] * expert_scale;
    }
    std::cout << "[3/5] local expert compute complete\n";

    std::array<uint64_t, kRows> partial_token_ids{101u, 101u, 102u, 102u};
    std::array<float, kTokens * kHidden> token_output{};
    uint32_t output_rows = 0u;
    Completion combine_done;
    status = shmem_combine_alltoall_inc<DataType::FP32>(
        &session, &batch, local_partials.data(), kRows,
        partial_token_ids.data(), token_output.data(), kTokens, &output_rows,
        stream, &combine_done);
    if (!Check(status, "combine") ||
        !Check(completion_wait(&session, combine_done), "combine wait"))
        return 1;
    std::cout << "[4/5] combine output rows=" << output_rows << '\n';
    for (uint32_t token = 0u; token < kTokens; ++token) {
        std::cout << "token " << token << ':';
        for (uint32_t h = 0u; h < kHidden; ++h)
            std::cout << ' ' << token_output[token * kHidden + h];
        std::cout << '\n';
    }

    if (!Check(single_inc_destroy(&session), "destroy")) return 1;
    std::cout << "[5/5] session destroyed\n";
    return 0;
}
