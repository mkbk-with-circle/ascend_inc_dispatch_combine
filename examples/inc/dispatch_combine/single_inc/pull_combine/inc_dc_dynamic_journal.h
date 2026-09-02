#ifndef INC_DC_DYNAMIC_JOURNAL_H
#define INC_DC_DYNAMIC_JOURNAL_H

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "inc_dc_endpoint_dispatch_packet.h"

namespace inc::dc::pull_combine {

enum class DynamicJournalStatus : uint32_t {
    OK = 0u,
    INVALID_ARGUMENT,
    SHAPE_MISMATCH,
    SOURCE_ALREADY_PUBLISHED,
    DUPLICATE_TOKEN_ID,
    DISPATCH_NOT_SEALED,
    DISPATCH_ALREADY_SEALED,
    SOURCE_SEQUENCE_MISMATCH,
    UNKNOWN_TOKEN_ID,
    UNEXPECTED_CONTRIBUTOR,
    DUPLICATE_CONTRIBUTOR,
    PAYLOAD_SIZE_MISMATCH,
    CAPACITY_EXCEEDED,
    NOT_READY,
};

struct DynamicJournalConfig {
    uint32_t worker_count = 0u;
    uint32_t hidden = 0u;
    uint32_t wave = 0u;
    uint64_t generation = 0u;
};

// Combine input is the logical content of one worker-owned registered slot.
// The device ABI carries the same token_ids + contiguous FP32 row layout;
// this owning form keeps the host reference model independent of transport.
struct SparseCombineBatch {
    uint32_t source_rank = 0u;
    uint64_t sequence = 0u;
    std::vector<uint64_t> token_ids;
    std::vector<float> values;
};

struct SparseCombineAck {
    uint32_t source_rank = 0u;
    uint64_t sequence = 0u;
    DynamicJournalStatus status = DynamicJournalStatus::OK;
    uint64_t rows_consumed = 0u;
};

struct DynamicEgressBatch {
    uint32_t owner_rank = 0u;
    std::vector<uint32_t> owner_rows;
    std::vector<uint64_t> token_ids;
    std::vector<float> values;
};

// Per-wave state learned only from endpoint-owned Dispatch packets.  There is
// no pre-uploaded token plan: AddDispatchSource is called as each source
// packet is parsed by the INC.  State is discarded after the wave completes.
class DynamicWaveJournal {
public:
    DynamicJournalStatus Initialize(const DynamicJournalConfig &config);
    DynamicJournalStatus AddDispatchSource(
        const ParsedEndpointDispatch &dispatch);
    DynamicJournalStatus SealDispatch();

    // Validate the whole batch before mutating any accumulator.  A malformed
    // batch therefore produces a negative ACK without partially consuming a
    // worker slot.
    DynamicJournalStatus ConsumeCombine(const SparseCombineBatch &batch,
                                        SparseCombineAck *ack);
    DynamicJournalStatus PopEgress(uint32_t owner_rank, uint32_t max_rows,
                                   DynamicEgressBatch *batch);

    bool dispatch_sealed() const { return sealed_; }
    bool combine_complete() const;
    uint64_t token_count() const { return entries_.size(); }
    uint64_t ready_count() const { return ready_count_; }
    uint64_t sent_count() const { return sent_count_; }

private:
    struct Entry {
        uint64_t token_id = 0u;
        uint32_t owner_rank = 0u;
        uint32_t owner_row = 0u;
        uint64_t expected[2]{0u, 0u};
        uint64_t received[2]{0u, 0u};
        bool ready = false;
        bool sent = false;
    };

    DynamicJournalConfig config_{};
    bool initialized_ = false;
    bool sealed_ = false;
    std::vector<uint8_t> dispatch_sources_;
    std::vector<uint64_t> next_sequence_;
    std::vector<Entry> entries_;
    std::unordered_map<uint64_t, uint32_t> token_index_;
    std::vector<float> accumulators_;
    std::vector<std::deque<uint32_t>> ready_by_owner_;
    uint64_t ready_count_ = 0u;
    uint64_t sent_count_ = 0u;
};

const char *DynamicJournalStatusString(DynamicJournalStatus status);

} // namespace inc::dc::pull_combine

#endif
