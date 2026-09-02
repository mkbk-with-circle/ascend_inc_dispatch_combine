#include "inc_dc_dynamic_journal.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace inc::dc::pull_combine {
namespace {

bool SameShape(const ParsedEndpointDispatch &dispatch,
               const DynamicJournalConfig &config)
{
    return dispatch.header.worker_count == config.worker_count &&
        dispatch.header.hidden == config.hidden &&
        dispatch.header.wave == config.wave &&
        dispatch.header.generation == config.generation &&
        dispatch.header.source_rank < config.worker_count &&
        dispatch.destination_rows.size() == config.worker_count;
}

bool BitSet(const uint64_t bits[2], uint32_t rank)
{
    return (bits[rank >> 6u] & (1ull << (rank & 63u))) != 0u;
}

void SetBit(uint64_t bits[2], uint32_t rank)
{
    bits[rank >> 6u] |= 1ull << (rank & 63u);
}

bool Complete(const uint64_t expected[2], const uint64_t received[2])
{
    return expected[0] == received[0] && expected[1] == received[1];
}

} // namespace

DynamicJournalStatus DynamicWaveJournal::Initialize(
    const DynamicJournalConfig &config)
{
    if (config.worker_count < 2u || config.worker_count > 128u ||
        config.hidden == 0u || config.generation == 0u)
        return DynamicJournalStatus::INVALID_ARGUMENT;
    config_ = config;
    initialized_ = true;
    sealed_ = false;
    dispatch_sources_.assign(config.worker_count, 0u);
    next_sequence_.assign(config.worker_count, 1u);
    entries_.clear();
    token_index_.clear();
    accumulators_.clear();
    ready_by_owner_.assign(config.worker_count, {});
    ready_count_ = 0u;
    sent_count_ = 0u;
    return DynamicJournalStatus::OK;
}

DynamicJournalStatus DynamicWaveJournal::AddDispatchSource(
    const ParsedEndpointDispatch &dispatch)
{
    if (!initialized_) return DynamicJournalStatus::INVALID_ARGUMENT;
    if (sealed_) return DynamicJournalStatus::DISPATCH_ALREADY_SEALED;
    if (!SameShape(dispatch, config_))
        return DynamicJournalStatus::SHAPE_MISMATCH;
    const uint32_t source = dispatch.header.source_rank;
    if (dispatch_sources_[source] != 0u)
        return DynamicJournalStatus::SOURCE_ALREADY_PUBLISHED;

    // A token appears once in every selected destination view.  First build a
    // source-local map and validate it transactionally before changing the
    // journal, so duplicate global IDs cannot leave a partial source behind.
    struct Pending {
        uint64_t token_id = 0u;
        uint32_t owner_row = 0u;
        uint64_t expected[2]{0u, 0u};
    };
    if (dispatch.source_tokens.size() != dispatch.header.token_count)
        return DynamicJournalStatus::SHAPE_MISMATCH;
    std::unordered_map<uint64_t, Pending> pending;
    for (const EndpointDispatchTokenRecord &token : dispatch.source_tokens) {
        if (token.source_token >= dispatch.header.token_count ||
            !pending.emplace(token.token_id,
                             Pending{token.token_id, token.source_token,
                                     {0u, 0u}}).second)
            return DynamicJournalStatus::DUPLICATE_TOKEN_ID;
    }
    for (uint32_t destination = 0u; destination < config_.worker_count;
         ++destination) {
        for (const EndpointFanoutRow &row :
             dispatch.destination_rows[destination]) {
            if (row.source_rank != source ||
                row.source_token >= dispatch.header.token_count ||
                row.assignments.empty())
                return DynamicJournalStatus::SHAPE_MISMATCH;
            auto it = pending.find(row.token_id);
            if (it == pending.end() ||
                it->second.owner_row != row.source_token)
                return DynamicJournalStatus::SHAPE_MISMATCH;
            if (BitSet(it->second.expected, destination))
                return DynamicJournalStatus::SHAPE_MISMATCH;
            SetBit(it->second.expected, destination);
        }
    }
    if (pending.size() > std::numeric_limits<uint32_t>::max() -
            entries_.size())
        return DynamicJournalStatus::CAPACITY_EXCEEDED;
    for (const auto &[token_id, item] : pending) {
        if (token_index_.find(token_id) != token_index_.end())
            return DynamicJournalStatus::DUPLICATE_TOKEN_ID;
        (void)item;
    }

    std::vector<Pending> ordered;
    ordered.reserve(pending.size());
    for (const auto &[token_id, item] : pending) {
        (void)token_id;
        ordered.push_back(item);
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const Pending &a, const Pending &b) {
                  return a.owner_row < b.owner_row;
              });
    for (const Pending &item : ordered) {
        const uint32_t index = static_cast<uint32_t>(entries_.size());
        Entry entry{};
        entry.token_id = item.token_id;
        entry.owner_rank = source;
        entry.owner_row = item.owner_row;
        entry.expected[0] = item.expected[0];
        entry.expected[1] = item.expected[1];
        entries_.push_back(entry);
        token_index_.emplace(item.token_id, index);
    }
    dispatch_sources_[source] = 1u;
    return DynamicJournalStatus::OK;
}

DynamicJournalStatus DynamicWaveJournal::SealDispatch()
{
    if (!initialized_) return DynamicJournalStatus::INVALID_ARGUMENT;
    if (sealed_) return DynamicJournalStatus::DISPATCH_ALREADY_SEALED;
    if (std::find(dispatch_sources_.begin(), dispatch_sources_.end(), 0u) !=
        dispatch_sources_.end())
        return DynamicJournalStatus::DISPATCH_NOT_SEALED;
    if (entries_.size() > std::numeric_limits<size_t>::max() /
            config_.hidden)
        return DynamicJournalStatus::CAPACITY_EXCEEDED;
    accumulators_.assign(entries_.size() * config_.hidden, 0.0f);
    sealed_ = true;
    // A future packet view will include zero-route tokens.  Keeping this
    // readiness path here makes their semantics explicit: output is zero.
    for (uint32_t i = 0u; i < entries_.size(); ++i) {
        if (entries_[i].expected[0] == 0u && entries_[i].expected[1] == 0u) {
            entries_[i].ready = true;
            ready_by_owner_[entries_[i].owner_rank].push_back(i);
            ++ready_count_;
        }
    }
    return DynamicJournalStatus::OK;
}

DynamicJournalStatus DynamicWaveJournal::ConsumeCombine(
    const SparseCombineBatch &batch, SparseCombineAck *ack)
{
    if (ack == nullptr || !sealed_ ||
        batch.source_rank >= config_.worker_count || batch.sequence == 0u) {
        return !sealed_ ? DynamicJournalStatus::DISPATCH_NOT_SEALED
                        : DynamicJournalStatus::INVALID_ARGUMENT;
    }
    ack->source_rank = batch.source_rank;
    ack->sequence = batch.sequence;
    ack->rows_consumed = 0u;
    if (batch.sequence != next_sequence_[batch.source_rank]) {
        ack->status = DynamicJournalStatus::SOURCE_SEQUENCE_MISMATCH;
        return ack->status;
    }
    if (batch.token_ids.size() > std::numeric_limits<size_t>::max() /
            config_.hidden ||
        batch.values.size() != batch.token_ids.size() * config_.hidden) {
        ack->status = DynamicJournalStatus::PAYLOAD_SIZE_MISMATCH;
        return ack->status;
    }

    std::vector<uint32_t> indices;
    indices.reserve(batch.token_ids.size());
    std::unordered_set<uint64_t> in_batch;
    for (uint64_t token_id : batch.token_ids) {
        auto found = token_index_.find(token_id);
        if (found == token_index_.end()) {
            ack->status = DynamicJournalStatus::UNKNOWN_TOKEN_ID;
            return ack->status;
        }
        if (!in_batch.insert(token_id).second) {
            ack->status = DynamicJournalStatus::DUPLICATE_CONTRIBUTOR;
            return ack->status;
        }
        const Entry &entry = entries_[found->second];
        if (!BitSet(entry.expected, batch.source_rank)) {
            ack->status = DynamicJournalStatus::UNEXPECTED_CONTRIBUTOR;
            return ack->status;
        }
        if (BitSet(entry.received, batch.source_rank)) {
            ack->status = DynamicJournalStatus::DUPLICATE_CONTRIBUTOR;
            return ack->status;
        }
        indices.push_back(found->second);
    }

    for (uint32_t row = 0u; row < indices.size(); ++row) {
        Entry &entry = entries_[indices[row]];
        float *accumulator = accumulators_.data() +
            static_cast<uint64_t>(indices[row]) * config_.hidden;
        const float *partial = batch.values.data() +
            static_cast<uint64_t>(row) * config_.hidden;
        for (uint32_t element = 0u; element < config_.hidden; ++element)
            accumulator[element] += partial[element];
        SetBit(entry.received, batch.source_rank);
        if (!entry.ready && Complete(entry.expected, entry.received)) {
            entry.ready = true;
            ready_by_owner_[entry.owner_rank].push_back(indices[row]);
            ++ready_count_;
        }
    }
    ++next_sequence_[batch.source_rank];
    ack->status = DynamicJournalStatus::OK;
    ack->rows_consumed = batch.token_ids.size();
    return DynamicJournalStatus::OK;
}

DynamicJournalStatus DynamicWaveJournal::PopEgress(
    uint32_t owner_rank, uint32_t max_rows, DynamicEgressBatch *batch)
{
    if (!sealed_) return DynamicJournalStatus::DISPATCH_NOT_SEALED;
    if (batch == nullptr || owner_rank >= config_.worker_count ||
        max_rows == 0u)
        return DynamicJournalStatus::INVALID_ARGUMENT;
    auto &queue = ready_by_owner_[owner_rank];
    if (queue.empty()) return DynamicJournalStatus::NOT_READY;
    *batch = {};
    batch->owner_rank = owner_rank;
    const uint32_t rows = std::min<uint32_t>(max_rows, queue.size());
    batch->owner_rows.reserve(rows);
    batch->token_ids.reserve(rows);
    batch->values.reserve(static_cast<size_t>(rows) * config_.hidden);
    for (uint32_t row = 0u; row < rows; ++row) {
        const uint32_t index = queue.front();
        queue.pop_front();
        Entry &entry = entries_[index];
        entry.sent = true;
        batch->owner_rows.push_back(entry.owner_row);
        batch->token_ids.push_back(entry.token_id);
        const float *values = accumulators_.data() +
            static_cast<uint64_t>(index) * config_.hidden;
        batch->values.insert(batch->values.end(), values,
                             values + config_.hidden);
        ++sent_count_;
    }
    return DynamicJournalStatus::OK;
}

bool DynamicWaveJournal::combine_complete() const
{
    return sealed_ && sent_count_ == entries_.size();
}

const char *DynamicJournalStatusString(DynamicJournalStatus status)
{
    switch (status) {
        case DynamicJournalStatus::OK: return "OK";
        case DynamicJournalStatus::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case DynamicJournalStatus::SHAPE_MISMATCH: return "SHAPE_MISMATCH";
        case DynamicJournalStatus::SOURCE_ALREADY_PUBLISHED:
            return "SOURCE_ALREADY_PUBLISHED";
        case DynamicJournalStatus::DUPLICATE_TOKEN_ID:
            return "DUPLICATE_TOKEN_ID";
        case DynamicJournalStatus::DISPATCH_NOT_SEALED:
            return "DISPATCH_NOT_SEALED";
        case DynamicJournalStatus::DISPATCH_ALREADY_SEALED:
            return "DISPATCH_ALREADY_SEALED";
        case DynamicJournalStatus::SOURCE_SEQUENCE_MISMATCH:
            return "SOURCE_SEQUENCE_MISMATCH";
        case DynamicJournalStatus::UNKNOWN_TOKEN_ID: return "UNKNOWN_TOKEN_ID";
        case DynamicJournalStatus::UNEXPECTED_CONTRIBUTOR:
            return "UNEXPECTED_CONTRIBUTOR";
        case DynamicJournalStatus::DUPLICATE_CONTRIBUTOR:
            return "DUPLICATE_CONTRIBUTOR";
        case DynamicJournalStatus::PAYLOAD_SIZE_MISMATCH:
            return "PAYLOAD_SIZE_MISMATCH";
        case DynamicJournalStatus::CAPACITY_EXCEEDED:
            return "CAPACITY_EXCEEDED";
        case DynamicJournalStatus::NOT_READY: return "NOT_READY";
    }
    return "UNKNOWN";
}

} // namespace inc::dc::pull_combine
