#include "inc_dc_pull_v2_api.h"

#include <limits>

namespace inc::dc::pull_v2::api {

namespace {

Status MakeStatus(StatusCode code)
{
    return Status{code, StatusString(code)};
}

bool ValidSessionConfig(const SessionConfig &config)
{
    return config.session_id != 0u && config.placement_epoch != 0u &&
        config.worker_count >= 2u && config.worker_count <= 128u &&
        config.inc_rank == config.worker_count &&
        config.rank <= config.inc_rank && config.hidden != 0u &&
        config.expert_count != 0u && config.ring_slots != 0u &&
        config.max_tokens_per_rank != 0u &&
        config.max_assignments_per_rank != 0u;
}

bool Worker(const SingleIncSession &session)
{
    return session.config.rank < session.config.worker_count;
}

uint32_t DataTypeBytes(DataType dtype)
{
    switch (dtype) {
        case DataType::FP16:
        case DataType::BF16: return 2u;
        case DataType::FP32: return 4u;
    }
    return 0u;
}

} // namespace

const char *StatusString(StatusCode code)
{
    switch (code) {
        case StatusCode::OK: return "ok";
        case StatusCode::INVALID_ARGUMENT: return "invalid argument";
        case StatusCode::CAPACITY_EXCEEDED: return "capacity exceeded";
        case StatusCode::BUSY_SLOT: return "ring slot busy";
        case StatusCode::STALE_HANDLE: return "stale batch handle";
        case StatusCode::BACKEND_ERROR: return "backend error";
        case StatusCode::TIMEOUT: return "timeout";
        case StatusCode::ABORTED: return "aborted";
        case StatusCode::UNSUPPORTED: return "unsupported";
    }
    return "unknown";
}

Status single_inc_create(
    const SessionConfig &config, const BackendOps &backend,
    void *backend_context, SingleIncSession *session)
{
    if (session == nullptr || session->initialized ||
        !ValidSessionConfig(config) || backend.create == nullptr ||
        backend.destroy == nullptr || backend.dispatch == nullptr ||
        backend.combine == nullptr || backend.query == nullptr ||
        backend.wait == nullptr || backend.release_batch == nullptr)
        return MakeStatus(StatusCode::INVALID_ARGUMENT);
    const StatusCode code = backend.create(backend_context, config);
    if (code != StatusCode::OK) return MakeStatus(code);
    session->config = config;
    session->backend = backend;
    session->backend_context = backend_context;
    session->initialized = true;
    return MakeStatus(StatusCode::OK);
}

Status single_inc_destroy(SingleIncSession *session)
{
    if (session == nullptr || !session->initialized)
        return MakeStatus(StatusCode::INVALID_ARGUMENT);
    if (session->live_batches != 0u)
        return MakeStatus(StatusCode::BUSY_SLOT);
    const StatusCode code = session->backend.destroy(session->backend_context);
    if (code != StatusCode::OK) return MakeStatus(code);
    *session = SingleIncSession{};
    return MakeStatus(StatusCode::OK);
}

Status DispatchAsync(
    SingleIncSession *session, DataType dtype, const WaveId &id,
    const DispatchInput *input, DispatchOutput *output, Stream stream,
    BatchHandle *batch, Completion *completion)
{
    if (session == nullptr || !session->initialized || batch == nullptr ||
        completion == nullptr || batch->live || completion->live ||
        id.generation == 0u || id.sequence == 0u ||
        id.ring_slot >= session->config.ring_slots || stream == nullptr ||
        DataTypeBytes(dtype) == 0u)
        return MakeStatus(StatusCode::INVALID_ARGUMENT);

    if (Worker(*session)) {
        if (input == nullptr || output == nullptr ||
            (input->token_count != 0u &&
             (input->send_buffer == nullptr ||
              input->send_token_ids == nullptr)) ||
            input->token_count > session->config.max_tokens_per_rank ||
            input->assignment_count >
                session->config.max_assignments_per_rank ||
            (input->fixed_topk == 0u &&
             input->assignment_offsets == nullptr) ||
            (input->assignment_count != 0u &&
             (input->destination_gpus == nullptr ||
              input->expert_ids == nullptr ||
              input->expert_weights == nullptr)) ||
            output->recv_buffer == nullptr ||
            output->recv_capacity_bytes == 0u ||
            output->row_capacity == 0u ||
            output->assignment_capacity == 0u ||
            output->recv_row_count == nullptr ||
            output->recv_assignment_count == nullptr)
            return MakeStatus(StatusCode::INVALID_ARGUMENT);
        if (input->fixed_topk != 0u &&
            (input->token_count >
                 std::numeric_limits<uint32_t>::max() /
                     input->fixed_topk ||
             input->assignment_count !=
                 input->token_count * input->fixed_topk))
            return MakeStatus(StatusCode::INVALID_ARGUMENT);
        const uint64_t elements =
            static_cast<uint64_t>(output->row_capacity) *
            session->config.hidden;
        const uint32_t dtype_bytes = DataTypeBytes(dtype);
        if (elements > std::numeric_limits<uint64_t>::max() / dtype_bytes ||
            elements * dtype_bytes > output->recv_capacity_bytes ||
            input->assignment_count > output->assignment_capacity)
            return MakeStatus(StatusCode::CAPACITY_EXCEEDED);
    } else if (input != nullptr || output != nullptr) {
        return MakeStatus(StatusCode::INVALID_ARGUMENT);
    }

    BackendTicket batch_ticket{};
    BackendTicket completion_ticket{};
    const StatusCode code = session->backend.dispatch(
        session->backend_context, dtype, id, input, output, stream,
        &batch_ticket, &completion_ticket);
    if (code != StatusCode::OK) return MakeStatus(code);
    batch->session_id = session->config.session_id;
    batch->id = id;
    batch->ticket = batch_ticket;
    batch->live = true;
    ++session->live_batches;
    completion->ticket = completion_ticket;
    completion->live = true;
    return MakeStatus(StatusCode::OK);
}

Status CombineAsync(
    SingleIncSession *session, BatchHandle *batch,
    const CombineInput *input, CombineOutput *output, Stream stream,
    Completion *completion)
{
    if (session == nullptr || !session->initialized || batch == nullptr)
        return MakeStatus(StatusCode::INVALID_ARGUMENT);
    if (!batch->live || batch->session_id != session->config.session_id)
        return MakeStatus(StatusCode::STALE_HANDLE);
    if (completion == nullptr || completion->live || stream == nullptr)
        return MakeStatus(StatusCode::INVALID_ARGUMENT);
    if (Worker(*session)) {
        if (input == nullptr || output == nullptr ||
            (input->send_count != 0u &&
             (input->send_buffer == nullptr ||
              input->send_token_ids == nullptr)) ||
            output->recv_buffer == nullptr || output->recv_count == nullptr ||
            output->recv_capacity == 0u ||
            input->send_count > session->config.max_assignments_per_rank ||
            output->recv_capacity > session->config.max_tokens_per_rank)
            return MakeStatus(StatusCode::INVALID_ARGUMENT);
    } else if (input != nullptr || output != nullptr) {
        return MakeStatus(StatusCode::INVALID_ARGUMENT);
    }

    BackendTicket completion_ticket{};
    const StatusCode code = session->backend.combine(
        session->backend_context, batch->id, batch->ticket, input, output,
        stream, &completion_ticket);
    if (code != StatusCode::OK) return MakeStatus(code);
    batch->live = false;
    if (session->live_batches == 0u)
        return MakeStatus(StatusCode::BACKEND_ERROR);
    --session->live_batches;
    completion->ticket = completion_ticket;
    completion->live = true;
    return MakeStatus(StatusCode::OK);
}

Status CombineFp32Async(
    SingleIncSession *session, BatchHandle *batch,
    const float *send_buffer, uint32_t send_count,
    const uint64_t *send_token_ids, float *recv_buffer,
    uint32_t recv_capacity, uint32_t *recv_count, Stream stream,
    Completion *completion)
{
    CombineInput input{};
    input.send_buffer = send_buffer;
    input.send_token_ids = send_token_ids;
    input.send_count = send_count;
    CombineOutput output{};
    output.recv_buffer = recv_buffer;
    output.recv_capacity = recv_capacity;
    output.recv_count = recv_count;
    const bool worker = session != nullptr && session->initialized &&
        Worker(*session);
    return CombineAsync(
        session, batch, worker ? &input : nullptr,
        worker ? &output : nullptr, stream, completion);
}

Status batch_release(SingleIncSession *session, BatchHandle *batch)
{
    if (session == nullptr || !session->initialized || batch == nullptr ||
        !batch->live || batch->session_id != session->config.session_id)
        return MakeStatus(StatusCode::STALE_HANDLE);
    const StatusCode code = session->backend.release_batch(
        session->backend_context, batch->ticket);
    if (code != StatusCode::OK) return MakeStatus(code);
    batch->live = false;
    if (session->live_batches == 0u)
        return MakeStatus(StatusCode::BACKEND_ERROR);
    --session->live_batches;
    return MakeStatus(StatusCode::OK);
}

Status completion_query(
    SingleIncSession *session, const Completion &completion)
{
    if (session == nullptr || !session->initialized || !completion.live)
        return MakeStatus(StatusCode::INVALID_ARGUMENT);
    return MakeStatus(session->backend.query(
        session->backend_context, completion.ticket));
}

Status completion_wait(
    SingleIncSession *session, const Completion &completion,
    uint64_t timeout_ns)
{
    if (session == nullptr || !session->initialized || !completion.live)
        return MakeStatus(StatusCode::INVALID_ARGUMENT);
    return MakeStatus(session->backend.wait(
        session->backend_context, completion.ticket, timeout_ns));
}

} // namespace inc::dc::pull_v2::api
