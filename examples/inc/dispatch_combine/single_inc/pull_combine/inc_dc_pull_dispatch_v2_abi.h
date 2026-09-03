#ifndef INC_DC_PULL_DISPATCH_V2_ABI_H
#define INC_DC_PULL_DISPATCH_V2_ABI_H

#include <cstdint>

namespace inc::dc::pull_v2 {

constexpr uint32_t kPullDispatchMagic = 0x50443249u; // 'PD2I'
constexpr uint16_t kPullDispatchAbiVersion = 1u;
constexpr uint32_t kPullDispatchAlignment = 64u;
constexpr uint32_t kPullDispatchMaxWorkers = 128u;

enum class DataType : uint32_t {
    FP16 = 0u,
    BF16 = 1u,
    FP32 = 2u,
};

// Installed once by the transport.  Per-wave records carry only region_id
// and ring_slot, never a process virtual address or a SHMEM pointer.
struct alignas(64) RegionRegistration {
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint32_t source_rank = 0u;
    uint32_t region_id = 0u;
    uint32_t slot_count = 0u;
    uint32_t alignment = kPullDispatchAlignment;
    uint64_t region_bytes = 0u;
    uint64_t slot_stride = 0u;
    uint64_t reserved[2]{};
};
static_assert(sizeof(RegionRegistration) == 64u,
              "pull Dispatch registration ABI drift");

// A worker publishes exactly one READY per source/wave after its complete
// slot is immutable and remotely readable.  publication is written last.
struct alignas(64) Ready {
    uint32_t magic = kPullDispatchMagic;
    uint16_t abi_version = kPullDispatchAbiVersion;
    uint16_t struct_bytes = sizeof(Ready);
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint32_t wave = 0u;
    uint32_t source_rank = 0u;
    uint32_t source_region_id = 0u;
    uint16_t ring_slot = 0u;
    uint16_t flags = 0u;
    uint64_t publication = 0u;
};
static_assert(sizeof(Ready) == 64u, "pull Dispatch READY ABI drift");

// Canonical source slot.  The INC first GETs this fixed header, then pulls
// metadata and hidden tiles from offsets derived and checked from the counts.
struct alignas(64) SlotHeader {
    uint32_t magic = kPullDispatchMagic;
    uint16_t abi_version = kPullDispatchAbiVersion;
    uint16_t header_bytes = sizeof(SlotHeader);
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint32_t wave = 0u;
    uint32_t source_rank = 0u;
    uint32_t source_region_id = 0u;
    uint16_t ring_slot = 0u;
    uint16_t flags = 0u;
    uint32_t worker_count = 0u;
    uint32_t token_count = 0u;
    uint32_t assignment_count = 0u;
    uint32_t hidden = 0u;
    uint32_t dtype = static_cast<uint32_t>(DataType::BF16);
    uint32_t token_record_bytes = 0u;
    uint32_t assignment_record_bytes = 0u;
    uint64_t tokens_offset = 0u;
    uint64_t assignments_offset = 0u;
    uint64_t hidden_offset = 0u;
    uint64_t packet_bytes = 0u;
    uint64_t metadata_digest = 0u;
};
static_assert(sizeof(SlotHeader) == 128u,
              "pull Dispatch slot header ABI drift");

struct TokenRecord {
    uint64_t token_id = 0u;
    uint32_t source_token = 0u;
    uint32_t assignment_begin = 0u;
    uint32_t assignment_count = 0u;
    uint32_t reserved0 = 0u;
    uint64_t reserved1 = 0u;
};
static_assert(sizeof(TokenRecord) == 32u,
              "pull Dispatch token record ABI drift");

struct AssignmentRecord {
    uint32_t destination_rank = 0u;
    uint32_t expert_id = 0u;
    uint32_t ordinal = 0u;
    float weight = 1.0f;
};
static_assert(sizeof(AssignmentRecord) == 16u,
              "pull Dispatch assignment ABI drift");

enum class JournalSlotState : uint32_t {
    FREE = 0u,
    DISPATCH_OPEN,
    DISPATCH_SEALED,
    COMBINE_ACTIVE,
    COMPLETE,
    ABORTED,
};

struct alignas(64) JournalSlotHeader {
    uint32_t magic = kPullDispatchMagic;
    uint16_t abi_version = kPullDispatchAbiVersion;
    uint16_t struct_bytes = sizeof(JournalSlotHeader);
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint64_t dispatch_cookie = 0u;
    uint32_t wave = 0u;
    uint16_t ring_slot = 0u;
    uint16_t state = static_cast<uint16_t>(JournalSlotState::FREE);
    uint32_t token_count = 0u;
    uint32_t contributor_count = 0u;
    uint32_t status = 0u;
    uint32_t flags = 0u;
    uint64_t reserved[1]{};
};
static_assert(sizeof(JournalSlotHeader) == 64u,
              "pull Dispatch journal header ABI drift");

// route_key is generation-scoped.  The stable identity is owner rank/row;
// token_id is retained only for framework compatibility and diagnostics.
struct alignas(64) JournalTokenEntry {
    uint64_t route_key = 0u;
    uint64_t token_id = 0u;
    uint32_t owner_rank = 0u;
    uint32_t owner_row = 0u;
    uint32_t contributors_begin = 0u;
    uint32_t contributors_count = 0u;
    uint32_t assignments_begin = 0u;
    uint32_t assignments_count = 0u;
    uint32_t accumulator_index = 0u;
    uint32_t flags = 0u;
    uint64_t reserved[2]{};
};
static_assert(sizeof(JournalTokenEntry) == 64u,
              "pull Dispatch journal token ABI drift");

struct JournalContributor {
    uint32_t worker_rank = 0u;
    uint32_t destination_row = 0u;
    uint32_t assignment_begin = 0u;
    uint32_t assignment_count = 0u;
};
static_assert(sizeof(JournalContributor) == 16u,
              "pull Dispatch contributor ABI drift");

// One hidden row is sent per unique destination.  Multiple local experts
// reference this row and are expanded only by the destination-side adapter.
struct DestinationRow {
    uint64_t route_key = 0u;
    uint64_t token_id = 0u;
    uint32_t source_rank = 0u;
    uint32_t source_token = 0u;
    uint32_t destination_row = 0u;
    uint32_t assignments_begin = 0u;
    uint32_t assignments_count = 0u;
    uint32_t reserved = 0u;
};
static_assert(sizeof(DestinationRow) == 40u,
              "pull Dispatch destination row ABI drift");

struct ExpertAssignment {
    uint32_t destination_row = 0u;
    uint32_t expert_id = 0u;
    uint32_t expert_row = 0u;
    uint32_t ordinal = 0u;
    float weight = 1.0f;
    uint32_t reserved[3]{};
};
static_assert(sizeof(ExpertAssignment) == 32u,
              "pull Dispatch expert assignment ABI drift");

constexpr uint64_t RouteKey(uint32_t owner_rank, uint32_t owner_row)
{
    return (static_cast<uint64_t>(owner_rank) << 32u) | owner_row;
}

} // namespace inc::dc::pull_v2

#endif
