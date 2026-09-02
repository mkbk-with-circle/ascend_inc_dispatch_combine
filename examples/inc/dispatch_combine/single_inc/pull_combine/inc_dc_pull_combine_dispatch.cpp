#include "inc_dc_pull_combine_dispatch.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace inc::dc::pull_combine {
namespace {

bool Mul(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr || (a != 0u &&
        b > std::numeric_limits<uint64_t>::max() / a))
        return false;
    *out = a * b;
    return true;
}

bool ReservedFieldsAreZero(const DispatchIngressDescriptor &descriptor)
{
    for (uint64_t value : descriptor.reserved) {
        if (value != 0u) return false;
    }
    return true;
}

uint64_t MatrixIndex(uint32_t row, uint32_t column, uint32_t width)
{
    return static_cast<uint64_t>(row) * width + column;
}

} // namespace

DispatchStatus PullDispatchCoordinator::Initialize(const WavePlan &plan,
                                                     uint32_t ring_depth)
{
    if (plan.worker_count == 0u ||
        plan.worker_count > kPullCombineMaxWorkers ||
        plan.generation == 0u || plan.hidden == 0u ||
        (plan.element_bytes != 2u && plan.element_bytes != 4u) ||
        ring_depth < 2u ||
        plan.tokens_per_rank.size() != plan.worker_count ||
        plan.result_rank_offsets.size() != plan.worker_count + 1u ||
        plan.result_rank_offsets.front() != 0u ||
        !std::is_sorted(plan.result_rank_offsets.begin(),
                        plan.result_rank_offsets.end()) ||
        plan.result_rank_offsets.back() !=
            plan.expected_contributors.size() ||
        plan.token_row_counts.size() !=
            static_cast<size_t>(plan.worker_count) * plan.worker_count ||
        plan.assignment_counts.size() != plan.token_row_counts.size() ||
        plan.dispatch_receive_offsets.size() !=
            plan.token_row_counts.size()) {
        return DispatchStatus::INVALID_ARGUMENT;
    }

    uint64_t token_elements = 0u;
    if (!Mul(plan.expected_contributors.size(), plan.hidden,
             &token_elements) ||
        token_elements > std::numeric_limits<size_t>::max()) {
        return DispatchStatus::INVALID_ARGUMENT;
    }

    std::vector<std::vector<uint32_t>> destination_slots(plan.worker_count);
    for (uint32_t destination = 0u; destination < plan.worker_count;
         ++destination) {
        uint64_t rows = 0u;
        for (uint32_t source = 0u; source < plan.worker_count; ++source)
            rows += plan.token_row_counts[MatrixIndex(
                source, destination, plan.worker_count)];
        if (rows > std::numeric_limits<size_t>::max())
            return DispatchStatus::INVALID_ARGUMENT;
        destination_slots[destination].assign(
            static_cast<size_t>(rows), std::numeric_limits<uint32_t>::max());
    }
    for (uint32_t slot_id = 0u; slot_id < plan.route_slots.size(); ++slot_id) {
        const RouteSlot &slot = plan.route_slots[slot_id];
        if (slot.result_id >= plan.expected_contributors.size() ||
            slot.source_rank >= plan.worker_count ||
            slot.destination_rank >= plan.worker_count ||
            slot.source_token >= plan.tokens_per_rank[slot.source_rank] ||
            slot.destination_row >=
                destination_slots[slot.destination_rank].size() ||
            destination_slots[slot.destination_rank][slot.destination_row] !=
                std::numeric_limits<uint32_t>::max() ||
            static_cast<uint64_t>(slot.assignment_begin) +
                slot.assignment_count > plan.assignments.size()) {
            return DispatchStatus::INVALID_ARGUMENT;
        }
        destination_slots[slot.destination_rank][slot.destination_row] =
            slot_id;
    }
    for (const auto &slots : destination_slots) {
        if (std::find(slots.begin(), slots.end(),
                      std::numeric_limits<uint32_t>::max()) != slots.end())
            return DispatchStatus::INVALID_ARGUMENT;
    }

    plan_ = plan;
    workers_ = plan.worker_count;
    ring_depth_ = ring_depth;
    initialized_ = true;
    aborted_ = false;
    terminal_status_ = DispatchStatus::NOT_READY;
    committed_count_sources_ = 0u;
    count_committed_.assign(workers_, 0u);
    count_reply_sent_.assign(workers_, 0u);
    sources_.assign(workers_, SourceState{});
    token_reserved_.assign(plan.expected_contributors.size(), 0u);
    token_received_.assign(plan.expected_contributors.size(), 0u);
    token_sequence_.assign(plan.expected_contributors.size(), 0u);
    token_values_.assign(static_cast<size_t>(token_elements), 0.0f);
    destination_slots_ = std::move(destination_slots);
    route_slot_sent_.assign(plan.route_slots.size(), 0u);
    received_tokens_ = 0u;
    sent_route_slots_ = 0u;
    if (plan.expected_contributors.empty() && plan.route_slots.empty())
        terminal_status_ = DispatchStatus::OK;
    return DispatchStatus::OK;
}

DispatchStatus PullDispatchCoordinator::CommitCounts(
    const CountCommitDescriptor &descriptor,
    const std::vector<uint64_t> &token_counts,
    const std::vector<uint64_t> &assignment_counts)
{
    if (!initialized_) return DispatchStatus::NOT_READY;
    if (aborted_) return DispatchStatus::ABORTED;
    if (descriptor.magic != kPullCombineMagic ||
        descriptor.abi_version != kPullCombineAbiVersion ||
        descriptor.struct_bytes != sizeof(CountCommitDescriptor) ||
        descriptor.source_rank >= workers_ ||
        descriptor.worker_count != workers_ ||
        descriptor.reserved0 != 0u || descriptor.source_region_id == 0u ||
        descriptor.token_counts_offset % sizeof(uint64_t) != 0u ||
        descriptor.assignment_counts_offset % sizeof(uint64_t) != 0u ||
        token_counts.size() != workers_ ||
        assignment_counts.size() != workers_) {
        return DispatchStatus::INVALID_DESCRIPTOR;
    }
    if (descriptor.generation != plan_.generation ||
        descriptor.wave != plan_.wave)
        return DispatchStatus::STALE_GENERATION;
    if (descriptor.semantic_digest != plan_.semantic_digest)
        return DispatchStatus::DIGEST_MISMATCH;
    if (count_committed_[descriptor.source_rank] != 0u)
        return DispatchStatus::COUNT_MISMATCH;
    for (uint32_t destination = 0u; destination < workers_; ++destination) {
        const uint64_t index = MatrixIndex(
            descriptor.source_rank, destination, workers_);
        if (token_counts[destination] != plan_.token_row_counts[index] ||
            assignment_counts[destination] !=
                plan_.assignment_counts[index]) {
            return DispatchStatus::COUNT_MISMATCH;
        }
    }
    count_committed_[descriptor.source_rank] = 1u;
    ++committed_count_sources_;
    return DispatchStatus::OK;
}

DispatchStatus PullDispatchCoordinator::PopCountReply(
    uint32_t destination_rank, CountReply *reply)
{
    if (!initialized_ || reply == nullptr || destination_rank >= workers_)
        return DispatchStatus::INVALID_ARGUMENT;
    if (aborted_) return DispatchStatus::ABORTED;
    if (!counts_ready()) return DispatchStatus::COUNTS_NOT_READY;
    if (count_reply_sent_[destination_rank] != 0u)
        return DispatchStatus::NOT_READY;

    CountReply built{};
    built.destination_rank = destination_rank;
    built.token_rows_from_source.resize(workers_);
    built.assignments_from_source.resize(workers_);
    built.receive_row_offsets.resize(workers_);
    for (uint32_t source = 0u; source < workers_; ++source) {
        built.token_rows_from_source[source] = plan_.token_row_counts[
            MatrixIndex(source, destination_rank, workers_)];
        built.assignments_from_source[source] = plan_.assignment_counts[
            MatrixIndex(source, destination_rank, workers_)];
        built.receive_row_offsets[source] = plan_.dispatch_receive_offsets[
            MatrixIndex(destination_rank, source, workers_)];
    }
    count_reply_sent_[destination_rank] = 1u;
    *reply = std::move(built);
    return DispatchStatus::OK;
}

DispatchStatus PullDispatchCoordinator::NotifyIngress(
    const DispatchIngressDescriptor &descriptor,
    const std::vector<float> &hidden_values)
{
    if (!initialized_) return DispatchStatus::NOT_READY;
    if (aborted_) return DispatchStatus::ABORTED;
    if (!counts_ready()) return DispatchStatus::COUNTS_NOT_READY;
    if (descriptor.magic != kPullCombineMagic ||
        descriptor.abi_version != kPullCombineAbiVersion ||
        descriptor.struct_bytes != sizeof(DispatchIngressDescriptor) ||
        descriptor.source_rank >= workers_ || descriptor.sequence == 0u ||
        descriptor.token_count == 0u || descriptor.inc_region_id == 0u ||
        descriptor.inc_offset % kPullCombineCacheLine != 0u ||
        descriptor.element_bytes != plan_.element_bytes ||
        !ReservedFieldsAreZero(descriptor) ||
        (descriptor.flags & ~(ToBits(DescriptorFlags::FINAL_FOR_WAVE) |
                              ToBits(DescriptorFlags::DEBUG_TOKEN_KEYS))) !=
            0u) {
        return DispatchStatus::INVALID_DESCRIPTOR;
    }
    if (descriptor.generation != plan_.generation ||
        descriptor.wave != plan_.wave)
        return DispatchStatus::STALE_GENERATION;
    if (descriptor.semantic_digest != plan_.semantic_digest)
        return DispatchStatus::DIGEST_MISMATCH;

    SourceState &source = sources_[descriptor.source_rank];
    if (descriptor.sequence != source.next_publish_sequence)
        return DispatchStatus::SEQUENCE_MISMATCH;
    if (source.next_publish_sequence ==
        std::numeric_limits<uint64_t>::max())
        return Abort(DispatchStatus::SEQUENCE_MISMATCH);
    if (source.pending.size() + source.acks.size() >= ring_depth_)
        return DispatchStatus::RING_FULL;

    const uint64_t token_end =
        static_cast<uint64_t>(descriptor.source_token_begin) +
        descriptor.token_count;
    if (token_end > plan_.tokens_per_rank[descriptor.source_rank])
        return DispatchStatus::OUT_OF_RANGE;
    uint64_t expected_values = 0u;
    uint64_t expected_bytes = 0u;
    if (!Mul(descriptor.token_count, plan_.hidden, &expected_values) ||
        !Mul(expected_values, plan_.element_bytes, &expected_bytes) ||
        hidden_values.size() != expected_values ||
        descriptor.payload_bytes != expected_bytes) {
        return DispatchStatus::PAYLOAD_SIZE_MISMATCH;
    }

    const uint64_t result_begin =
        plan_.result_rank_offsets[descriptor.source_rank] +
        descriptor.source_token_begin;
    for (uint64_t result = result_begin;
         result < result_begin + descriptor.token_count; ++result) {
        if (token_reserved_[result] != 0u)
            return DispatchStatus::DUPLICATE_TOKEN;
    }

    PendingIngress pending{};
    pending.descriptor = descriptor;
    for (uint64_t result = result_begin;
         result < result_begin + descriptor.token_count; ++result) {
        token_reserved_[result] = 1u;
        token_received_[result] = 1u;
        token_sequence_[result] = descriptor.sequence;
        pending.remaining_route_slots += plan_.expected_contributors[result];
        const size_t local = static_cast<size_t>(result - result_begin);
        std::copy_n(hidden_values.data() + local * plan_.hidden,
                    plan_.hidden,
                    token_values_.data() +
                        static_cast<size_t>(result) * plan_.hidden);
        ++received_tokens_;
    }
    source.pending.push_back(pending);
    ++source.next_publish_sequence;
    PublishCompletedAcks(descriptor.source_rank);
    if (complete()) terminal_status_ = DispatchStatus::OK;
    return DispatchStatus::OK;
}

DispatchStatus PullDispatchCoordinator::PopFanoutChunk(
    uint32_t destination_rank, uint32_t max_rows,
    DispatchFanoutChunk *chunk)
{
    if (!initialized_ || chunk == nullptr || max_rows == 0u ||
        destination_rank >= workers_)
        return DispatchStatus::INVALID_ARGUMENT;
    if (aborted_) return DispatchStatus::ABORTED;
    if (!counts_ready()) return DispatchStatus::COUNTS_NOT_READY;

    const std::vector<uint32_t> &slots =
        destination_slots_[destination_rank];
    size_t first = slots.size();
    for (size_t row = 0u; row < slots.size(); ++row) {
        const uint32_t slot_id = slots[row];
        const RouteSlot &slot = plan_.route_slots[slot_id];
        if (token_received_[slot.result_id] != 0u &&
            route_slot_sent_[slot_id] == 0u) {
            first = row;
            break;
        }
    }
    if (first == slots.size()) return DispatchStatus::NOT_READY;

    DispatchFanoutChunk built{};
    built.destination_rank = destination_rank;
    built.destination_row_begin = static_cast<uint32_t>(first);
    built.assignment_offsets.push_back(0u);
    std::vector<uint32_t> touched_sources;
    for (size_t row = first; row < slots.size() &&
         built.route_slot_ids.size() < max_rows; ++row) {
        const uint32_t slot_id = slots[row];
        const RouteSlot &slot = plan_.route_slots[slot_id];
        if (token_received_[slot.result_id] == 0u ||
            route_slot_sent_[slot_id] != 0u)
            break;
        built.route_slot_ids.push_back(slot_id);
        const float *values = token_values_.data() +
            static_cast<size_t>(slot.result_id) * plan_.hidden;
        built.hidden_values.insert(built.hidden_values.end(), values,
                                   values + plan_.hidden);
        for (uint32_t i = 0u; i < slot.assignment_count; ++i) {
            built.assignments.push_back(
                plan_.assignments[slot.assignment_begin + i]);
        }
        built.assignment_offsets.push_back(static_cast<uint32_t>(
            built.assignments.size()));
        route_slot_sent_[slot_id] = 1u;
        ++sent_route_slots_;

        SourceState &source = sources_[slot.source_rank];
        const uint64_t sequence = token_sequence_[slot.result_id];
        auto pending = std::find_if(
            source.pending.begin(), source.pending.end(),
            [sequence](const PendingIngress &entry) {
                return entry.descriptor.sequence == sequence;
            });
        if (pending == source.pending.end() ||
            pending->remaining_route_slots == 0u)
            return Abort(DispatchStatus::SEQUENCE_MISMATCH);
        --pending->remaining_route_slots;
        touched_sources.push_back(slot.source_rank);
    }
    std::sort(touched_sources.begin(), touched_sources.end());
    touched_sources.erase(std::unique(touched_sources.begin(),
                                      touched_sources.end()),
                          touched_sources.end());
    for (uint32_t source : touched_sources) PublishCompletedAcks(source);
    *chunk = std::move(built);
    if (complete()) terminal_status_ = DispatchStatus::OK;
    return DispatchStatus::OK;
}

void PullDispatchCoordinator::PublishCompletedAcks(uint32_t source_rank)
{
    SourceState &source = sources_[source_rank];
    while (!source.pending.empty() &&
           source.pending.front().remaining_route_slots == 0u) {
        const DispatchIngressDescriptor descriptor =
            source.pending.front().descriptor;
        DispatchAck ack{};
        ack.generation = plan_.generation;
        ack.sequence = descriptor.sequence;
        ack.source_rank = source_rank;
        ack.status = static_cast<uint32_t>(DispatchStatus::OK);
        ack.tokens_consumed = descriptor.token_count;
        source.acks.push_back(ack);
        source.pending.pop_front();
    }
}

DispatchStatus PullDispatchCoordinator::PopAck(uint32_t source_rank,
                                                DispatchAck *ack)
{
    if (!initialized_ || ack == nullptr || source_rank >= workers_)
        return DispatchStatus::INVALID_ARGUMENT;
    SourceState &source = sources_[source_rank];
    if (source.acks.empty()) return DispatchStatus::NOT_READY;
    *ack = source.acks.front();
    source.acks.pop_front();
    return DispatchStatus::OK;
}

DispatchStatus PullDispatchCoordinator::Abort(DispatchStatus reason)
{
    if (!initialized_) return DispatchStatus::NOT_READY;
    if (reason == DispatchStatus::OK || reason == DispatchStatus::NOT_READY)
        return DispatchStatus::INVALID_ARGUMENT;
    if (aborted_) return DispatchStatus::ABORTED;
    for (uint32_t source_rank = 0u; source_rank < workers_; ++source_rank) {
        SourceState &source = sources_[source_rank];
        while (!source.pending.empty()) {
            const DispatchIngressDescriptor descriptor =
                source.pending.front().descriptor;
            DispatchAck ack{};
            ack.generation = plan_.generation;
            ack.sequence = descriptor.sequence;
            ack.source_rank = source_rank;
            ack.status = static_cast<uint32_t>(reason);
            ack.tokens_consumed = 0u;
            source.acks.push_back(ack);
            source.pending.pop_front();
        }
    }
    aborted_ = true;
    terminal_status_ = reason;
    return DispatchStatus::ABORTED;
}

bool PullDispatchCoordinator::complete() const
{
    if (!initialized_ || aborted_ || !counts_ready() ||
        received_tokens_ != token_received_.size() ||
        sent_route_slots_ != route_slot_sent_.size())
        return false;
    for (const SourceState &source : sources_) {
        if (!source.pending.empty()) return false;
    }
    return true;
}

const char *DispatchStatusString(DispatchStatus status)
{
    switch (status) {
        case DispatchStatus::OK: return "OK";
        case DispatchStatus::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case DispatchStatus::INVALID_DESCRIPTOR: return "INVALID_DESCRIPTOR";
        case DispatchStatus::STALE_GENERATION: return "STALE_GENERATION";
        case DispatchStatus::SEQUENCE_MISMATCH: return "SEQUENCE_MISMATCH";
        case DispatchStatus::RING_FULL: return "RING_FULL";
        case DispatchStatus::OUT_OF_RANGE: return "OUT_OF_RANGE";
        case DispatchStatus::DUPLICATE_TOKEN: return "DUPLICATE_TOKEN";
        case DispatchStatus::DIGEST_MISMATCH: return "DIGEST_MISMATCH";
        case DispatchStatus::PAYLOAD_SIZE_MISMATCH:
            return "PAYLOAD_SIZE_MISMATCH";
        case DispatchStatus::COUNT_MISMATCH: return "COUNT_MISMATCH";
        case DispatchStatus::COUNTS_NOT_READY: return "COUNTS_NOT_READY";
        case DispatchStatus::NOT_READY: return "NOT_READY";
        case DispatchStatus::ABORTED: return "ABORTED";
    }
    return "UNKNOWN";
}

} // namespace inc::dc::pull_combine
