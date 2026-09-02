#ifndef INC_DC_PULL_COMBINE_DISPATCH_H
#define INC_DC_PULL_COMBINE_DISPATCH_H

#include <cstdint>
#include <deque>
#include <vector>

#include "inc_dc_pull_combine_abi.h"
#include "inc_dc_pull_combine_plan.h"

namespace inc::dc::pull_combine {

enum class DispatchStatus : uint32_t {
    OK = 0u,
    INVALID_ARGUMENT,
    INVALID_DESCRIPTOR,
    STALE_GENERATION,
    SEQUENCE_MISMATCH,
    RING_FULL,
    OUT_OF_RANGE,
    DUPLICATE_TOKEN,
    DIGEST_MISMATCH,
    PAYLOAD_SIZE_MISMATCH,
    COUNT_MISMATCH,
    COUNTS_NOT_READY,
    NOT_READY,
    ABORTED,
};

struct CountReply {
    uint32_t destination_rank = 0u;
    std::vector<uint64_t> token_rows_from_source;
    std::vector<uint64_t> assignments_from_source;
    std::vector<uint64_t> receive_row_offsets;
};

struct DispatchFanoutChunk {
    uint32_t destination_rank = 0u;
    uint32_t destination_row_begin = 0u;
    std::vector<uint32_t> route_slot_ids;
    std::vector<float> hidden_values;
    // assignment_offsets has route_slot_ids.size()+1 entries.  The expert
    // metadata for row i is assignments[offsets[i]..offsets[i+1]).
    std::vector<uint32_t> assignment_offsets;
    std::vector<PackedAssignment> assignments;
};

// Transport-neutral reference state machine for count exchange and Dispatch.
// NotifyIngress represents the publication after A's PUT into INC HBM.  A
// dispatch ACK is generated only when all fan-out PUTs using that range have
// completed (modeled by PopFanoutChunk).
class PullDispatchCoordinator {
public:
    DispatchStatus Initialize(const WavePlan &plan, uint32_t ring_depth);

    DispatchStatus CommitCounts(const CountCommitDescriptor &descriptor,
                                const std::vector<uint64_t> &token_counts,
                                const std::vector<uint64_t> &assignment_counts);
    DispatchStatus PopCountReply(uint32_t destination_rank,
                                 CountReply *reply);
    DispatchStatus NotifyIngress(const DispatchIngressDescriptor &descriptor,
                                 const std::vector<float> &hidden_values);
    DispatchStatus PopFanoutChunk(uint32_t destination_rank,
                                  uint32_t max_rows,
                                  DispatchFanoutChunk *chunk);
    DispatchStatus PopAck(uint32_t source_rank, DispatchAck *ack);
    DispatchStatus Abort(DispatchStatus reason);

    bool counts_ready() const { return committed_count_sources_ == workers_; }
    bool complete() const;
    bool aborted() const { return aborted_; }
    DispatchStatus terminal_status() const { return terminal_status_; }
    uint64_t received_tokens() const { return received_tokens_; }
    uint64_t sent_route_slots() const { return sent_route_slots_; }

private:
    struct PendingIngress {
        DispatchIngressDescriptor descriptor{};
        uint64_t remaining_route_slots = 0u;
    };

    struct SourceState {
        uint64_t next_publish_sequence = 1u;
        std::deque<PendingIngress> pending;
        std::deque<DispatchAck> acks;
    };

    void PublishCompletedAcks(uint32_t source_rank);

    WavePlan plan_{};
    uint32_t workers_ = 0u;
    uint32_t ring_depth_ = 0u;
    bool initialized_ = false;
    bool aborted_ = false;
    DispatchStatus terminal_status_ = DispatchStatus::NOT_READY;
    uint32_t committed_count_sources_ = 0u;
    std::vector<uint8_t> count_committed_;
    std::vector<uint8_t> count_reply_sent_;
    std::vector<SourceState> sources_;
    std::vector<uint8_t> token_reserved_;
    std::vector<uint8_t> token_received_;
    std::vector<uint64_t> token_sequence_;
    std::vector<float> token_values_;
    std::vector<std::vector<uint32_t>> destination_slots_;
    std::vector<uint8_t> route_slot_sent_;
    uint64_t received_tokens_ = 0u;
    uint64_t sent_route_slots_ = 0u;
};

const char *DispatchStatusString(DispatchStatus status);

} // namespace inc::dc::pull_combine

#endif
