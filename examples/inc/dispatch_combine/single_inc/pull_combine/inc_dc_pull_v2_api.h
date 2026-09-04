#ifndef INC_DC_PULL_V2_API_H
#define INC_DC_PULL_V2_API_H

#include <cstdint>

namespace inc::dc::pull_v2::api {

using Stream = void *;

enum class DataType : uint32_t {
    FP16 = 0u,
    BF16 = 1u,
    FP32 = 2u,
};

enum class StatusCode : uint32_t {
    OK = 0u,
    INVALID_ARGUMENT,
    CAPACITY_EXCEEDED,
    BUSY_SLOT,
    STALE_HANDLE,
    BACKEND_ERROR,
    TIMEOUT,
    ABORTED,
    UNSUPPORTED,
};

struct Status {
    StatusCode code = StatusCode::OK;
    const char *message = "ok";

    explicit operator bool() const { return code == StatusCode::OK; }
};

const char *StatusString(StatusCode code);

struct WaveId {
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint32_t wave = 0u;
    uint16_t ring_slot = 0u;
    uint16_t reserved = 0u;
};

struct SessionConfig {
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint32_t rank = 0u;
    uint32_t worker_count = 0u;
    uint32_t inc_rank = 0u;
    uint32_t hidden = 0u;
    uint32_t expert_count = 0u;
    uint32_t ring_slots = 2u;
    uint32_t max_tokens_per_rank = 0u;
    uint32_t max_assignments_per_rank = 0u;
};

// Canonical variable-top-k input. All pointers are device pointers. Route
// assignments are flattened; assignment_offsets has token_count + 1 entries.
struct DispatchInput {
    const void *send_buffer = nullptr;             // [token_count, hidden]
    const uint64_t *send_token_ids = nullptr;      // [token_count]
    const uint32_t *assignment_offsets = nullptr;  // [token_count + 1]
    const uint32_t *destination_gpus = nullptr;    // [assignment_count]
    const uint32_t *expert_ids = nullptr;           // [assignment_count]
    const uint32_t *ordinals = nullptr;             // optional
    const float *expert_weights = nullptr;          // [assignment_count]
    uint32_t token_count = 0u;
    uint32_t assignment_count = 0u;
    uint32_t fixed_topk = 0u; // nonzero permits assignment_offsets == nullptr
};

struct DispatchOutput {
    void *recv_buffer = nullptr;
    uint64_t recv_capacity_bytes = 0u;
    uint32_t row_capacity = 0u;
    uint32_t assignment_capacity = 0u;
    uint32_t *recv_row_count = nullptr;
    uint32_t *recv_assignment_count = nullptr;
};

// Current device-qualified Combine contract is FP32 partial/reduction/output.
// The caller has already reduced all experts local to the same destination GPU.
struct CombineInput {
    const float *send_buffer = nullptr;        // [send_count, hidden]
    const uint64_t *send_token_ids = nullptr;  // [send_count]
    uint32_t send_count = 0u;
};

struct CombineOutput {
    float *recv_buffer = nullptr;
    uint32_t recv_capacity = 0u;
    uint32_t *recv_count = nullptr;
};

struct BackendTicket {
    uint64_t words[4]{};
};

struct BatchHandle {
    uint64_t session_id = 0u;
    WaveId id{};
    BackendTicket ticket{};
    bool live = false;
};

struct Completion {
    BackendTicket ticket{};
    bool live = false;
};

struct BackendOps {
    StatusCode (*create)(void *context, const SessionConfig &config) = nullptr;
    StatusCode (*destroy)(void *context) = nullptr;
    StatusCode (*dispatch)(
        void *context, DataType dtype, const WaveId &id,
        const DispatchInput *input, DispatchOutput *output, Stream stream,
        BackendTicket *batch, BackendTicket *completion) = nullptr;
    StatusCode (*combine)(
        void *context, const WaveId &id, const BackendTicket &batch,
        const CombineInput *input, CombineOutput *output, Stream stream,
        BackendTicket *completion) = nullptr;
    StatusCode (*query)(void *context, const BackendTicket &completion) = nullptr;
    StatusCode (*wait)(
        void *context, const BackendTicket &completion,
        uint64_t timeout_ns) = nullptr;
    StatusCode (*release_batch)(
        void *context, const BackendTicket &batch) = nullptr;
};

struct SingleIncSession {
    SessionConfig config{};
    BackendOps backend{};
    void *backend_context = nullptr;
    uint32_t live_batches = 0u;
    bool initialized = false;
};

Status single_inc_create(
    const SessionConfig &config, const BackendOps &backend,
    void *backend_context, SingleIncSession *session);
Status single_inc_destroy(SingleIncSession *session);
Status batch_release(SingleIncSession *session, BatchHandle *batch);
Status completion_query(
    SingleIncSession *session, const Completion &completion);
Status completion_wait(
    SingleIncSession *session, const Completion &completion,
    uint64_t timeout_ns = 0u);

Status DispatchAsync(
    SingleIncSession *session, DataType dtype, const WaveId &id,
    const DispatchInput *input, DispatchOutput *output, Stream stream,
    BatchHandle *batch, Completion *completion);
Status CombineAsync(
    SingleIncSession *session, BatchHandle *batch,
    const CombineInput *input, CombineOutput *output, Stream stream,
    Completion *completion);

template <DataType DT>
Status shmem_dispatch_alltoall_inc(
    SingleIncSession *session, const WaveId &id,
    const DispatchInput *input, DispatchOutput *output, Stream stream,
    BatchHandle *batch, Completion *completion)
{
    static_assert(
        DT == DataType::FP16 || DT == DataType::BF16 ||
        DT == DataType::FP32);
    return DispatchAsync(
        session, DT, id, input, output, stream, batch, completion);
}

// Convenience overload matching a dense [tokens, topk] router output.
// ordinal is implicitly the top-k slot index.
template <DataType DT>
Status shmem_dispatch_alltoall_inc(
    SingleIncSession *session, const WaveId &id,
    const void *send_buffer, const uint64_t *send_token_ids,
    const uint32_t *send_k_gpus, const uint32_t *send_k_expert_ids,
    const float *send_k_expert_weights, uint32_t token_count,
    uint32_t topk, DispatchOutput *output, Stream stream,
    BatchHandle *batch, Completion *completion)
{
    DispatchInput input{};
    input.send_buffer = send_buffer;
    input.send_token_ids = send_token_ids;
    input.destination_gpus = send_k_gpus;
    input.expert_ids = send_k_expert_ids;
    input.expert_weights = send_k_expert_weights;
    input.token_count = token_count;
    input.assignment_count = token_count * topk;
    input.fixed_topk = topk;
    return shmem_dispatch_alltoall_inc<DT>(
        session, id, &input, output, stream, batch, completion);
}

// Non-template implementation used by the typed public wrapper below.
Status CombineFp32Async(
    SingleIncSession *session, BatchHandle *batch,
    const float *send_buffer, uint32_t send_count,
    const uint64_t *send_token_ids, float *recv_buffer,
    uint32_t recv_capacity, uint32_t *recv_count, Stream stream,
    Completion *completion);

// The public Combine spelling mirrors Dispatch. The currently qualified
// reduction path is deliberately restricted to FP32 local partials. Keeping
// that constraint at compile time avoids a silent precision conversion while
// leaving the API shape ready for future FP16/BF16 specializations.
template <DataType DT>
Status shmem_combine_alltoall_inc(
    SingleIncSession *session, BatchHandle *batch,
    const float *send_buffer, uint32_t send_count,
    const uint64_t *send_token_ids, float *recv_buffer,
    uint32_t recv_capacity, uint32_t *recv_count, Stream stream,
    Completion *completion)
{
    static_assert(
        DT == DataType::FP32,
        "Pull V2 Combine currently supports FP32 local partials only");
    return CombineFp32Async(
        session, batch, send_buffer, send_count, send_token_ids,
        recv_buffer, recv_capacity, recv_count, stream, completion);
}

} // namespace inc::dc::pull_v2::api

#endif // INC_DC_PULL_V2_API_H
