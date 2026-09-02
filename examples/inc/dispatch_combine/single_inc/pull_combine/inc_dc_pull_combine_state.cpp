#include "inc_dc_pull_combine_state.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace inc::dc::pull_combine {
namespace {

uint32_t PartialBytes(PartialDType dtype)
{
    switch (dtype) {
        case PartialDType::FP32: return 4u;
        case PartialDType::FP16:
        case PartialDType::BF16: return 2u;
    }
    return 0u;
}

bool Mul(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr || (a != 0u &&
        b > std::numeric_limits<uint64_t>::max() / a))
        return false;
    *out = a * b;
    return true;
}

bool ReservedFieldsAreZero(const CombineReadyDescriptor &descriptor)
{
    if (descriptor.reserved0 != 0u) return false;
    for (uint64_t value : descriptor.reserved) {
        if (value != 0u) return false;
    }
    return true;
}

bool IsNondecreasing(const std::vector<uint64_t> &offsets)
{
    return std::is_sorted(offsets.begin(), offsets.end());
}

bool ValidateCompiledPlan(const WavePlan &plan)
{
    if (plan.worker_count == 0u ||
        plan.worker_count > kPullCombineMaxWorkers ||
        plan.generation == 0u || plan.hidden == 0u ||
        plan.combine_source_offsets.size() != plan.worker_count + 1u ||
        plan.result_rank_offsets.size() != plan.worker_count + 1u ||
        plan.combine_source_offsets.front() != 0u ||
        plan.result_rank_offsets.front() != 0u ||
        !IsNondecreasing(plan.combine_source_offsets) ||
        !IsNondecreasing(plan.result_rank_offsets) ||
        plan.combine_source_offsets.back() != plan.combine_rows.size() ||
        plan.result_rank_offsets.back() !=
            plan.expected_contributors.size()) {
        return false;
    }

    std::vector<uint32_t> observed(plan.expected_contributors.size(), 0u);
    for (uint32_t source = 0u; source < plan.worker_count; ++source) {
        for (uint64_t packed = plan.combine_source_offsets[source];
             packed < plan.combine_source_offsets[source + 1u]; ++packed) {
            const CombineRow &row = plan.combine_rows[packed];
            if (row.source_rank != source ||
                row.result_id >= observed.size() ||
                observed[row.result_id] ==
                    std::numeric_limits<uint32_t>::max()) {
                return false;
            }
            ++observed[row.result_id];
        }
    }
    return observed == plan.expected_contributors;
}

} // namespace

CoordinatorStatus PullCombineCoordinator::Initialize(
    const WavePlan &plan, uint32_t ring_depth, PartialDType partial_dtype)
{
    const uint32_t partial_bytes = PartialBytes(partial_dtype);
    if (!ValidateCompiledPlan(plan) || ring_depth < 2u ||
        partial_bytes == 0u) {
        return CoordinatorStatus::INVALID_ARGUMENT;
    }
    uint64_t accumulator_elements = 0u;
    if (!Mul(plan.expected_contributors.size(), plan.hidden,
             &accumulator_elements) ||
        accumulator_elements > std::numeric_limits<size_t>::max()) {
        return CoordinatorStatus::INVALID_ARGUMENT;
    }

    plan_ = plan;
    ring_depth_ = ring_depth;
    partial_dtype_ = partial_dtype;
    initialized_ = true;
    aborted_ = false;
    terminal_status_ = CoordinatorStatus::NOT_READY;
    sources_.assign(plan.worker_count, SourceState{});
    combine_row_reserved_.assign(plan.combine_rows.size(), 0u);
    combine_row_completed_.assign(plan.combine_rows.size(), 0u);
    received_contributors_.assign(plan.expected_contributors.size(), 0u);
    result_ready_.assign(plan.expected_contributors.size(), 0u);
    result_sent_.assign(plan.expected_contributors.size(), 0u);
    accumulators_.assign(accumulator_elements, 0.0f);
    completed_rows_ = 0u;
    ready_results_ = 0u;
    sent_results_ = 0u;
    for (uint32_t result = 0u;
         result < plan.expected_contributors.size(); ++result) {
        if (plan.expected_contributors[result] == 0u) {
            result_ready_[result] = 1u;
            ++ready_results_;
        }
    }
    if (complete()) terminal_status_ = CoordinatorStatus::OK;
    return CoordinatorStatus::OK;
}

CoordinatorStatus PullCombineCoordinator::Notify(
    const CombineReadyDescriptor &descriptor)
{
    if (!initialized_) return CoordinatorStatus::NOT_READY;
    if (aborted_) return CoordinatorStatus::ABORTED;
    if (descriptor.magic != kPullCombineMagic ||
        descriptor.abi_version != kPullCombineAbiVersion ||
        descriptor.struct_bytes != sizeof(CombineReadyDescriptor) ||
        descriptor.source_rank >= plan_.worker_count ||
        descriptor.row_count == 0u || descriptor.sequence == 0u ||
        descriptor.source_region_id == 0u ||
        descriptor.source_offset % kPullCombineCacheLine != 0u ||
        !ReservedFieldsAreZero(descriptor) ||
        (descriptor.flags & ~(ToBits(DescriptorFlags::FINAL_FOR_WAVE) |
                              ToBits(DescriptorFlags::DEBUG_TOKEN_KEYS))) != 0u) {
        return CoordinatorStatus::INVALID_DESCRIPTOR;
    }
    if (descriptor.generation != plan_.generation ||
        descriptor.wave != plan_.wave)
        return CoordinatorStatus::STALE_GENERATION;
    if (descriptor.semantic_digest != plan_.semantic_digest)
        return CoordinatorStatus::DIGEST_MISMATCH;
    if (descriptor.partial_dtype != static_cast<uint32_t>(partial_dtype_))
        return CoordinatorStatus::DTYPE_MISMATCH;

    SourceState &source = sources_[descriptor.source_rank];
    if (descriptor.sequence != source.next_publish_sequence)
        return CoordinatorStatus::SEQUENCE_MISMATCH;
    if (source.next_publish_sequence ==
        std::numeric_limits<uint64_t>::max())
        return Abort(CoordinatorStatus::SEQUENCE_MISMATCH);
    if (source.queued.size() + source.acks.size() +
            (source.fetch_inflight ? 1u : 0u) >=
        ring_depth_)
        return CoordinatorStatus::RING_FULL;

    uint64_t row_end = 0u;
    if (descriptor.combine_row_begin >
            std::numeric_limits<uint64_t>::max() - descriptor.row_count) {
        return CoordinatorStatus::OUT_OF_RANGE;
    }
    row_end = descriptor.combine_row_begin + descriptor.row_count;
    if (descriptor.combine_row_begin <
            plan_.combine_source_offsets[descriptor.source_rank] ||
        row_end > plan_.combine_source_offsets[descriptor.source_rank + 1u] ||
        row_end > plan_.combine_rows.size()) {
        return CoordinatorStatus::OUT_OF_RANGE;
    }

    uint64_t expected_bytes = 0u;
    if (!Mul(descriptor.row_count, plan_.hidden, &expected_bytes) ||
        !Mul(expected_bytes, PartialBytes(partial_dtype_), &expected_bytes) ||
        descriptor.payload_bytes != expected_bytes) {
        return CoordinatorStatus::PAYLOAD_SIZE_MISMATCH;
    }
    for (uint64_t row = descriptor.combine_row_begin; row < row_end; ++row) {
        if (combine_row_reserved_[row] != 0u)
            return CoordinatorStatus::DUPLICATE_ROW;
    }
    for (uint64_t row = descriptor.combine_row_begin; row < row_end; ++row)
        combine_row_reserved_[row] = 1u;

    source.queued.push_back(descriptor);
    ++source.next_publish_sequence;
    return CoordinatorStatus::OK;
}

CoordinatorStatus PullCombineCoordinator::BeginFetch(uint32_t source_rank,
                                                       FetchTask *task)
{
    if (!initialized_ || task == nullptr || source_rank >= sources_.size())
        return CoordinatorStatus::INVALID_ARGUMENT;
    if (aborted_) return CoordinatorStatus::ABORTED;
    SourceState &source = sources_[source_rank];
    if (source.fetch_inflight) return CoordinatorStatus::PEER_BUSY;
    if (source.queued.empty()) return CoordinatorStatus::NOT_READY;
    if (source.queued.front().sequence != source.next_fetch_sequence)
        return Abort(CoordinatorStatus::SEQUENCE_MISMATCH);
    source.inflight = source.queued.front();
    source.queued.pop_front();
    source.fetch_inflight = true;
    task->descriptor = source.inflight;
    ++source.next_fetch_sequence;
    return CoordinatorStatus::OK;
}

CoordinatorStatus PullCombineCoordinator::CompleteFetch(
    uint32_t source_rank, uint64_t sequence,
    const std::vector<float> &partial_rows)
{
    if (!initialized_ || source_rank >= sources_.size())
        return CoordinatorStatus::INVALID_ARGUMENT;
    if (aborted_) return CoordinatorStatus::ABORTED;
    SourceState &source = sources_[source_rank];
    if (!source.fetch_inflight) return CoordinatorStatus::NOT_READY;
    const CombineReadyDescriptor descriptor = source.inflight;
    if (sequence != descriptor.sequence ||
        sequence != source.next_ack_sequence)
        return Abort(CoordinatorStatus::SEQUENCE_MISMATCH);
    uint64_t expected_values = 0u;
    if (!Mul(descriptor.row_count, plan_.hidden, &expected_values) ||
        partial_rows.size() != expected_values) {
        return Abort(CoordinatorStatus::PAYLOAD_SIZE_MISMATCH);
    }

    const uint64_t row_end = descriptor.combine_row_begin +
                             descriptor.row_count;
    // Validate the whole descriptor before changing any accumulator.  Device
    // code follows the same two-phase rule so a malformed range can never
    // leave a partially committed result behind.
    std::vector<uint32_t> contributions_in_descriptor(
        received_contributors_.size(), 0u);
    for (uint64_t packed_row = descriptor.combine_row_begin;
         packed_row < row_end; ++packed_row) {
        if (combine_row_completed_[packed_row] != 0u)
            return Abort(CoordinatorStatus::DUPLICATE_ROW);
        const CombineRow &row = plan_.combine_rows[packed_row];
        const uint32_t result = row.result_id;
        if (result >= received_contributors_.size())
            return Abort(CoordinatorStatus::OUT_OF_RANGE);
        ++contributions_in_descriptor[result];
        if (contributions_in_descriptor[result] >
                plan_.expected_contributors[result] -
                    received_contributors_[result]) {
            return Abort(CoordinatorStatus::DUPLICATE_ROW);
        }
    }

    size_t input = 0u;
    for (uint64_t packed_row = descriptor.combine_row_begin;
         packed_row < row_end; ++packed_row) {
        const CombineRow &row = plan_.combine_rows[packed_row];
        const uint32_t result = row.result_id;
        float *accumulator = accumulators_.data() +
            static_cast<size_t>(result) * plan_.hidden;
        if (received_contributors_[result] == 0u) {
            std::copy_n(partial_rows.data() + input, plan_.hidden,
                        accumulator);
        } else {
            for (uint32_t h = 0u; h < plan_.hidden; ++h)
                accumulator[h] += partial_rows[input + h];
        }
        input += plan_.hidden;
        combine_row_completed_[packed_row] = 1u;
        ++completed_rows_;
        ++received_contributors_[result];
        if (received_contributors_[result] ==
            plan_.expected_contributors[result]) {
            result_ready_[result] = 1u;
            ++ready_results_;
        }
    }

    CombineAck ack{};
    ack.generation = plan_.generation;
    ack.sequence = sequence;
    ack.source_rank = source_rank;
    ack.status = static_cast<uint32_t>(CoordinatorStatus::OK);
    ack.rows_consumed = descriptor.row_count;
    source.acks.push_back(ack);
    source.fetch_inflight = false;
    source.inflight = {};
    ++source.next_ack_sequence;
    return CoordinatorStatus::OK;
}

CoordinatorStatus PullCombineCoordinator::PopAck(uint32_t source_rank,
                                                   CombineAck *ack)
{
    if (!initialized_ || ack == nullptr || source_rank >= sources_.size())
        return CoordinatorStatus::INVALID_ARGUMENT;
    SourceState &source = sources_[source_rank];
    if (source.acks.empty()) return CoordinatorStatus::NOT_READY;
    *ack = source.acks.front();
    source.acks.pop_front();
    return CoordinatorStatus::OK;
}

CoordinatorStatus PullCombineCoordinator::PopEgressChunk(
    uint32_t destination_rank, uint32_t max_rows, EgressChunk *chunk)
{
    if (!initialized_ || chunk == nullptr || max_rows == 0u ||
        destination_rank >= plan_.worker_count)
        return CoordinatorStatus::INVALID_ARGUMENT;
    if (aborted_) return CoordinatorStatus::ABORTED;
    const uint32_t begin = static_cast<uint32_t>(
        plan_.result_rank_offsets[destination_rank]);
    const uint32_t end = static_cast<uint32_t>(
        plan_.result_rank_offsets[destination_rank + 1u]);
    uint32_t first = end;
    for (uint32_t result = begin; result < end; ++result) {
        if (result_ready_[result] != 0u && result_sent_[result] == 0u) {
            first = result;
            break;
        }
    }
    if (first == end) return CoordinatorStatus::NOT_READY;

    EgressChunk built{};
    built.destination_rank = destination_rank;
    built.destination_row_begin = first - begin;
    for (uint32_t result = first; result < end &&
         built.result_ids.size() < max_rows; ++result) {
        if (result_ready_[result] == 0u || result_sent_[result] != 0u)
            break;
        built.result_ids.push_back(result);
        const float *values = accumulators_.data() +
            static_cast<size_t>(result) * plan_.hidden;
        built.values.insert(built.values.end(), values,
                            values + plan_.hidden);
        result_sent_[result] = 1u;
        ++sent_results_;
    }
    *chunk = std::move(built);
    if (complete()) terminal_status_ = CoordinatorStatus::OK;
    return CoordinatorStatus::OK;
}

CoordinatorStatus PullCombineCoordinator::Abort(CoordinatorStatus reason)
{
    if (!initialized_) return CoordinatorStatus::NOT_READY;
    if (reason == CoordinatorStatus::OK ||
        reason == CoordinatorStatus::NOT_READY)
        return CoordinatorStatus::INVALID_ARGUMENT;
    if (aborted_) return CoordinatorStatus::ABORTED;

    // Every descriptor accepted before the failure receives exactly one ACK.
    // This releases the worker's registered source slots even on a fail-closed
    // generation abort.  Successful ACKs already queued are preserved.
    for (uint32_t source_rank = 0u; source_rank < sources_.size();
         ++source_rank) {
        SourceState &source = sources_[source_rank];
        auto publish_failure = [&](const CombineReadyDescriptor &descriptor) {
            CombineAck ack{};
            ack.generation = plan_.generation;
            ack.sequence = descriptor.sequence;
            ack.source_rank = source_rank;
            ack.status = static_cast<uint32_t>(reason);
            ack.rows_consumed = 0u;
            source.acks.push_back(ack);
        };
        if (source.fetch_inflight) {
            publish_failure(source.inflight);
            source.fetch_inflight = false;
            source.inflight = {};
        }
        while (!source.queued.empty()) {
            publish_failure(source.queued.front());
            source.queued.pop_front();
        }
    }
    aborted_ = true;
    terminal_status_ = reason;
    return CoordinatorStatus::ABORTED;
}

bool PullCombineCoordinator::complete() const
{
    if (!initialized_ || aborted_ ||
        completed_rows_ != combine_row_completed_.size() ||
        sent_results_ != result_sent_.size())
        return false;
    for (const SourceState &source : sources_) {
        if (source.fetch_inflight || !source.queued.empty()) return false;
    }
    return true;
}

const char *CoordinatorStatusString(CoordinatorStatus status)
{
    switch (status) {
        case CoordinatorStatus::OK: return "OK";
        case CoordinatorStatus::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case CoordinatorStatus::INVALID_DESCRIPTOR: return "INVALID_DESCRIPTOR";
        case CoordinatorStatus::STALE_GENERATION: return "STALE_GENERATION";
        case CoordinatorStatus::SEQUENCE_MISMATCH: return "SEQUENCE_MISMATCH";
        case CoordinatorStatus::RING_FULL: return "RING_FULL";
        case CoordinatorStatus::PEER_BUSY: return "PEER_BUSY";
        case CoordinatorStatus::OUT_OF_RANGE: return "OUT_OF_RANGE";
        case CoordinatorStatus::DUPLICATE_ROW: return "DUPLICATE_ROW";
        case CoordinatorStatus::DTYPE_MISMATCH: return "DTYPE_MISMATCH";
        case CoordinatorStatus::DIGEST_MISMATCH: return "DIGEST_MISMATCH";
        case CoordinatorStatus::PAYLOAD_SIZE_MISMATCH:
            return "PAYLOAD_SIZE_MISMATCH";
        case CoordinatorStatus::NOT_READY: return "NOT_READY";
        case CoordinatorStatus::ABORTED: return "ABORTED";
    }
    return "UNKNOWN";
}

} // namespace inc::dc::pull_combine
