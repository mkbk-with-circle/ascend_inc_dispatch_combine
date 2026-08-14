#ifndef INC_DC_INLINE_ROUTE_JOURNAL_H
#define INC_DC_INLINE_ROUTE_JOURNAL_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "inc_dc_inline_route_protocol.h"

namespace inc::dc {

// Host reference model for the route state that a real INC creates while it
// parses inline Dispatch metadata.  The journal has no SHMEM, ACL or transport
// dependency; device/ring implementations are expected to preserve the same
// state transitions and validation rules.
enum class IncDcInlineJournalStatus : uint32_t {
    OK = 0u,
    INVALID_ARGUMENT,
    INVALID_CONFIG,
    NO_MEMORY,
    CAPACITY_EXCEEDED,
    NO_FREE_GENERATION_SLOT,
    GENERATION_ALREADY_OPEN,
    STALE_HANDLE,
    INVALID_STATE,
    DISPATCH_SEALED,
    PLACEMENT_NOT_FOUND,
    DUPLICATE_TOKEN,
    DUPLICATE_ORDINAL,
    INVALID_ROUTE_KEY,
    STALE_ROUTE_KEY,
    WRONG_CONTRIBUTOR,
    DUPLICATE_CONTRIBUTION,
    GENERATION_ABORTED,
    INCOMPLETE,
};

enum class IncDcInlineGenerationState : uint32_t {
    FREE = 0u,
    DISPATCH_OPEN,
    DISPATCH_SEALED,
    ABORTED,
};

struct IncDcInlineTokenRecordV2 {
    uint32_t origin_rank = 0u;
    uint64_t origin_token = 0u;
    uint32_t wave = 0u;
    uint64_t hidden_bytes = 0u;
    const InlineRouteEntryV2 *endpoints = nullptr;
    uint64_t endpoint_count = 0u;
};

struct IncDcInlineResolvedRouteV2 {
    InlineRouteKeyV2 route_key{};
    uint32_t contributor_rank = 0u;
    uint32_t expert_id = 0u;
    uint32_t local_expert = 0u;
    uint32_t ordinal = 0u;
    uint32_t weight_bits = 0u;
    // Online row ordinal within
    // (generation, request, wave, contributor_rank, local_expert).
    uint64_t destination_row = 0u;
};

struct IncDcInlineGenerationHandleV2 {
    uint64_t generation = 0u;
    uint64_t request_id = 0u;
    uint64_t slot_epoch = 0u;
    uint64_t authenticator = 0u;
    uint32_t slot_index = 0u;
    uint32_t reserved = 0u;
};

using IncDcInlineExpertPlacementFn = bool (*)(
    void *context, uint32_t expert_id, uint32_t *destination_rank,
    uint32_t *local_expert);

struct IncDcInlineRouteJournalConfig {
    uint32_t worker_count = 0u;
    uint32_t expert_count = 0u;
    uint32_t generation_slot_count = 0u;
    uint32_t max_results_per_slot = 0u;
    uint32_t max_contributions_per_slot = 0u;
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    // Per-session non-zero cookie.  It prevents a key from another journal
    // instance from becoming valid after slot/generation reuse.
    uint64_t session_cookie = 0u;
    IncDcInlineExpertPlacementFn lookup_placement = nullptr;
    void *placement_context = nullptr;
};

struct IncDcInlineGenerationOpenV2 {
    uint64_t generation = 0u;
    uint64_t request_id = 0u;
    uint32_t result_capacity = 0u;
    uint32_t contribution_capacity = 0u;
};

struct IncDcInlineCombineReceiptV2 {
    uint32_t origin_rank = 0u;
    uint64_t origin_token = 0u;
    uint32_t ordinal = 0u;
    uint32_t weight_bits = 0u;
    uint32_t expected = 0u;
    uint32_t arrived = 0u;
    uint32_t wave = 0u;
    uint64_t hidden_bytes = 0u;
    bool result_complete = false;
};

struct IncDcInlineGenerationSummaryV2 {
    IncDcInlineGenerationState state = IncDcInlineGenerationState::FREE;
    uint64_t generation = 0u;
    uint64_t request_id = 0u;
    uint64_t slot_epoch = 0u;
    uint32_t result_count = 0u;
    uint32_t contribution_count = 0u;
    uint32_t completed_result_count = 0u;
    uint32_t arrived_contribution_count = 0u;
};

class IncDcInlineRouteJournal {
public:
    static IncDcInlineJournalStatus Create(
        const IncDcInlineRouteJournalConfig &config,
        std::unique_ptr<IncDcInlineRouteJournal> *journal);

    IncDcInlineJournalStatus OpenGeneration(
        const IncDcInlineGenerationOpenV2 &open,
        IncDcInlineGenerationHandleV2 *handle);

    // Transactional: on error neither journal state nor routes is changed.
    IncDcInlineJournalStatus RecordDispatchToken(
        const IncDcInlineGenerationHandleV2 &handle,
        const IncDcInlineTokenRecordV2 &token,
        std::vector<IncDcInlineResolvedRouteV2> *routes);

    // Combine needs only the opaque key and the physical contributor port.
    // It may arrive while Dispatch is still open; publication ordering is a
    // transport/device concern, represented here by RecordDispatchToken
    // completing before the key is returned.
    IncDcInlineJournalStatus RecordCombineContribution(
        const InlineRouteKeyV2 &route_key,
        uint32_t contributor_rank,
        IncDcInlineCombineReceiptV2 *receipt);

    IncDcInlineJournalStatus SealDispatch(
        const IncDcInlineGenerationHandleV2 &handle);
    IncDcInlineJournalStatus AbortGeneration(
        const IncDcInlineGenerationHandleV2 &handle);
    IncDcInlineJournalStatus ReleaseGeneration(
        const IncDcInlineGenerationHandleV2 &handle);
    IncDcInlineJournalStatus QueryGeneration(
        const IncDcInlineGenerationHandleV2 &handle,
        IncDcInlineGenerationSummaryV2 *summary) const;

    IncDcInlineRouteJournal(const IncDcInlineRouteJournal &) = delete;
    IncDcInlineRouteJournal &operator=(
        const IncDcInlineRouteJournal &) = delete;

private:
    struct ResultState {
        uint32_t origin_rank = 0u;
        uint64_t origin_token = 0u;
        uint32_t first_contribution = 0u;
        uint32_t expected = 0u;
        uint32_t arrived = 0u;
        uint32_t wave = 0u;
        uint64_t hidden_bytes = 0u;
        bool complete = false;
    };

    struct ContributionState {
        uint32_t result_index = 0u;
        uint32_t contributor_rank = 0u;
        uint32_t expert_id = 0u;
        uint32_t local_expert = 0u;
        uint32_t ordinal = 0u;
        uint32_t weight_bits = 0u;
        uint32_t wave = 0u;
        uint64_t destination_row = 0u;
        bool arrived = false;
    };

    struct GenerationSlot {
        IncDcInlineGenerationState state = IncDcInlineGenerationState::FREE;
        uint64_t generation = 0u;
        uint64_t request_id = 0u;
        uint64_t epoch = 0u;
        uint32_t result_capacity = 0u;
        uint32_t contribution_capacity = 0u;
        uint32_t completed_results = 0u;
        uint32_t arrived_contributions = 0u;
        std::vector<ResultState> results;
        std::vector<ContributionState> contributions;
    };

    explicit IncDcInlineRouteJournal(
        const IncDcInlineRouteJournalConfig &config);

    uint64_t HandleAuthenticator(uint32_t slot_index, uint64_t generation,
                                 uint64_t request_id, uint64_t epoch) const;
    uint64_t RouteAuthenticator(uint32_t slot_index,
                                uint32_t contribution_index,
                                uint64_t generation, uint64_t request_id,
                                uint64_t epoch) const;
    InlineRouteKeyV2 MakeRouteKey(
        uint32_t slot_index, uint32_t contribution_index,
        const GenerationSlot &slot) const;
    GenerationSlot *FindSlot(const IncDcInlineGenerationHandleV2 &handle);
    const GenerationSlot *FindSlot(
        const IncDcInlineGenerationHandleV2 &handle) const;

    IncDcInlineRouteJournalConfig config_{};
    mutable std::mutex mutex_;
    std::vector<GenerationSlot> slots_;
};

const char *IncDcInlineJournalStatusString(
    IncDcInlineJournalStatus status);

} // namespace inc::dc

#endif
