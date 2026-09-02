#ifndef INC_DC_PULL_COMBINE_STATE_H
#define INC_DC_PULL_COMBINE_STATE_H

#include <cstdint>
#include <deque>
#include <vector>

#include "inc_dc_pull_combine_abi.h"
#include "inc_dc_pull_combine_plan.h"

namespace inc::dc::pull_combine {

enum class CoordinatorStatus : uint32_t {
    OK = 0u,
    INVALID_ARGUMENT,
    INVALID_DESCRIPTOR,
    STALE_GENERATION,
    SEQUENCE_MISMATCH,
    RING_FULL,
    PEER_BUSY,
    OUT_OF_RANGE,
    DUPLICATE_ROW,
    DTYPE_MISMATCH,
    DIGEST_MISMATCH,
    PAYLOAD_SIZE_MISMATCH,
    NOT_READY,
    ABORTED,
};

struct FetchTask {
    CombineReadyDescriptor descriptor{};
};

struct EgressChunk {
    uint32_t destination_rank = 0u;
    uint32_t destination_row_begin = 0u;
    std::vector<uint32_t> result_ids;
    std::vector<float> values;
};

// Host reference model for the persistent INC protocol.  It deliberately
// models GET completion as a separate step: Notify/BeginFetch never make a
// partial visible to reducers.  Only CompleteFetch (the device quiet point)
// may update accumulators and publish ACK.
class PullCombineCoordinator {
public:
    CoordinatorStatus Initialize(const WavePlan &plan, uint32_t ring_depth,
                                 PartialDType partial_dtype);

    CoordinatorStatus Notify(const CombineReadyDescriptor &descriptor);
    CoordinatorStatus BeginFetch(uint32_t source_rank, FetchTask *task);
    CoordinatorStatus CompleteFetch(uint32_t source_rank, uint64_t sequence,
                                    const std::vector<float> &partial_rows);
    CoordinatorStatus PopAck(uint32_t source_rank, CombineAck *ack);
    CoordinatorStatus PopEgressChunk(uint32_t destination_rank,
                                     uint32_t max_rows,
                                     EgressChunk *chunk);
    CoordinatorStatus Abort(CoordinatorStatus reason);

    bool complete() const;
    bool aborted() const { return aborted_; }
    CoordinatorStatus terminal_status() const { return terminal_status_; }
    uint64_t completed_rows() const { return completed_rows_; }
    uint64_t ready_results() const { return ready_results_; }
    uint64_t sent_results() const { return sent_results_; }

private:
    struct SourceState {
        uint64_t next_publish_sequence = 1u;
        uint64_t next_fetch_sequence = 1u;
        uint64_t next_ack_sequence = 1u;
        std::deque<CombineReadyDescriptor> queued;
        bool fetch_inflight = false;
        CombineReadyDescriptor inflight{};
        std::deque<CombineAck> acks;
    };

    // Own the compiled plan: callers may free or reuse their planner storage
    // immediately after Initialize returns.
    WavePlan plan_{};
    uint32_t ring_depth_ = 0u;
    PartialDType partial_dtype_ = PartialDType::FP32;
    bool initialized_ = false;
    bool aborted_ = false;
    CoordinatorStatus terminal_status_ = CoordinatorStatus::NOT_READY;
    std::vector<SourceState> sources_;
    std::vector<uint8_t> combine_row_reserved_;
    std::vector<uint8_t> combine_row_completed_;
    std::vector<uint32_t> received_contributors_;
    std::vector<uint8_t> result_ready_;
    std::vector<uint8_t> result_sent_;
    std::vector<float> accumulators_;
    uint64_t completed_rows_ = 0u;
    uint64_t ready_results_ = 0u;
    uint64_t sent_results_ = 0u;
};

const char *CoordinatorStatusString(CoordinatorStatus status);

} // namespace inc::dc::pull_combine

#endif
