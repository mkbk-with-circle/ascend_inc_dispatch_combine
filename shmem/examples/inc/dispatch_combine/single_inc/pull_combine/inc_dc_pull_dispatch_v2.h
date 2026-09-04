#ifndef INC_DC_PULL_DISPATCH_V2_H
#define INC_DC_PULL_DISPATCH_V2_H

#include <cstdint>
#include <string>
#include <vector>

#include "inc_dc_pull_dispatch_v2_abi.h"

namespace inc::dc::pull_v2 {

enum class Status : uint32_t {
    OK = 0u,
    INVALID_ARGUMENT,
    INVALID_REGISTRATION,
    INVALID_READY,
    INVALID_HEADER,
    STALE_EPOCH,
    SIZE_OVERFLOW,
    CAPACITY_EXCEEDED,
    DIGEST_MISMATCH,
    INVALID_TOKEN,
    INVALID_ASSIGNMENT,
    DUPLICATE_SOURCE,
    INVALID_STATE_TRANSITION,
};

struct SessionConfig {
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint32_t worker_count = 0u;
    uint32_t expert_count = 0u;
    uint32_t hidden = 0u;
    DataType dtype = DataType::BF16;
    uint32_t ring_slots = 0u;
};

struct SourceInput {
    SessionConfig session{};
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint32_t wave = 0u;
    uint32_t source_rank = 0u;
    uint32_t source_region_id = 0u;
    uint16_t ring_slot = 0u;
    std::vector<uint64_t> token_ids;
    std::vector<uint32_t> assignment_offsets;
    std::vector<AssignmentRecord> assignments;
    std::vector<uint8_t> hidden_payload;
};

struct ParsedSource {
    SlotHeader header{};
    std::vector<TokenRecord> tokens;
    std::vector<AssignmentRecord> assignments;
    std::vector<uint8_t> hidden_payload;
};

struct LayoutConfig {
    uint32_t worker_count = 0u;
    uint32_t expert_count = 0u;
    uint64_t destination_row_capacity = 0u;
    uint64_t destination_assignment_capacity = 0u;
};

struct CompiledLayout {
    JournalSlotHeader journal_header{};
    std::vector<JournalTokenEntry> journal_tokens;
    std::vector<JournalContributor> contributors;
    std::vector<AssignmentRecord> journal_assignments;
    std::vector<std::vector<DestinationRow>> destination_rows;
    std::vector<std::vector<ExpertAssignment>> expert_assignments;
    // Flattened [destination][expert].
    std::vector<uint32_t> expert_counts;
};

Status ValidateRegistration(const RegionRegistration &registration,
                            const SessionConfig &session,
                            std::string *error = nullptr);

Status BuildSlot(const SourceInput &input, std::vector<uint8_t> *slot,
                 SlotHeader *header, Ready *ready,
                 std::string *error = nullptr);

Status ParseReadyAndSlot(const RegionRegistration &registration,
                         const SessionConfig &session, const Ready &ready,
                         const uint8_t *slot, uint64_t slot_capacity,
                         ParsedSource *parsed,
                         std::string *error = nullptr);

// arrival_order determines dynamic destination-row allocation.  Journal token
// order remains source-rank/owner-row order and is therefore deterministic.
Status CompileLayout(const std::vector<ParsedSource> &sources,
                     const std::vector<uint32_t> &arrival_order,
                     const LayoutConfig &config, CompiledLayout *layout,
                     std::string *error = nullptr);

bool ValidJournalTransition(JournalSlotState from, JournalSlotState to);
Status TransitionJournal(JournalSlotHeader *header, JournalSlotState to,
                         std::string *error = nullptr);

uint64_t ReadyPublication(const Ready &ready);
uint64_t MetadataDigest(const SlotHeader &header, const uint8_t *slot);
const char *StatusString(Status status);

} // namespace inc::dc::pull_v2

#endif
