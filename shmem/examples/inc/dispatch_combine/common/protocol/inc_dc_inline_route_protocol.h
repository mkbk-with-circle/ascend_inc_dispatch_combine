#ifndef INC_DC_INLINE_ROUTE_PROTOCOL_H
#define INC_DC_INLINE_ROUTE_PROTOCOL_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace inc::dc {

// Standalone single-INC V2 wire contract.
//
// The endpoint owns only the logical router output carried by
// InlineTokenRecordHeaderV2 + InlineRouteEntryV2[].  It must never send a
// destination rank, destination row, combine owner, SHMEM PE/address or AIV
// id.  The INC resolves physical placement, allocates route keys and records
// the authoritative Dispatch journal used by Combine.
//
// All integral fields use canonical little-endian wire order.  The Ascend
// emulator is little-endian and may use the layout-gated structs directly;
// other transports must encode/decode the fields explicitly.  Reserved bytes
// must be zero and unknown flags fail closed.
constexpr uint32_t kInlineRouteTokenMagic = 0x49525432u;    // IRT2
constexpr uint32_t kInlineRouteBatchMagic = 0x49524232u;    // IRB2
constexpr uint32_t kInlineRouteFanoutMagic = 0x49524632u;   // IRF2
constexpr uint32_t kInlineRouteCombineMagic = 0x49524332u;  // IRC2
constexpr uint32_t kInlineRouteResultMagic = 0x49525232u;   // IRR2
constexpr uint32_t kInlineRouteControlMagic = 0x49524532u;  // IRE2
constexpr uint16_t kInlineRouteVersion = 2u;
constexpr uint16_t kInlineRouteHeaderBytes = 128u;
constexpr uint64_t kInlineRouteWireAlignment = 64u;

enum InlineRouteRecordFlags : uint32_t {
    kInlineRouteRecordNone = 0u,
    // A retry is legal only when the complete canonical record is identical
    // to the first record with the same logical identity.
    kInlineRouteRecordRetry = 1u << 0,
};
constexpr uint32_t kInlineRouteRecordAllowedFlags =
    kInlineRouteRecordRetry;

enum InlineRouteControlKind : uint32_t {
    kInlineRouteDispatchWaveSeal = 1u,
    kInlineRouteCombineWaveSeal = 2u,
    kInlineRouteAbortGeneration = 3u,
    kInlineRouteReleaseGeneration = 4u,
};

enum InlineRouteControlFlags : uint32_t {
    kInlineRouteControlNone = 0u,
    // digest is FNV-1a-64 over canonical token identity and route entries;
    // hidden payload bytes are intentionally excluded from this cheap gate.
    kInlineRouteControlHasDigest = 1u << 0,
};
constexpr uint32_t kInlineRouteControlAllowedFlags =
    kInlineRouteControlHasDigest;

// Opaque identity allocated by the INC Dispatch journal.  The slot epoch
// prevents stale keys from matching after a bounded generation slot is reused.
struct alignas(16) InlineRouteKeyV2 {
    uint64_t journal_locator = 0u;
    uint64_t authenticator = 0u;
};

// One endpoint-generated logical top-k choice.  route_ordinal must be unique
// and contiguous [0, route_count) within a token.  weight_bits is the bitwise
// IEEE-754 float32 representation.  There is deliberately no destination.
struct alignas(16) InlineRouteEntryV2 {
    uint32_t expert_id = 0u;
    uint32_t route_ordinal = 0u;
    uint32_t weight_bits = 0u;
    uint32_t reserved32 = 0u;
};

// Logical record layout:
//   header -> route entries -> zero padding -> hidden payload
// payload_offset and record_bytes make extensions and checked streaming
// parsing unambiguous.  Transport segmentation/reassembly is below this
// protocol; records from different ordered ingress queues are never
// interleaved inside one logical record.
struct alignas(64) InlineTokenRecordHeaderV2 {
    uint32_t magic = kInlineRouteTokenMagic;
    uint16_t version = kInlineRouteVersion;
    uint16_t header_bytes = kInlineRouteHeaderBytes;
    uint32_t flags = kInlineRouteRecordNone;
    uint32_t source_rank = 0u;
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t request_id = 0u;
    uint64_t source_token = 0u;
    uint64_t hidden_bytes = 0u;
    uint64_t payload_offset = 0u;
    uint64_t record_bytes = 0u;
    uint32_t route_count = 0u;
    uint32_t route_entry_bytes = sizeof(InlineRouteEntryV2);
    uint32_t wave = 0u;
    uint32_t reserved32 = 0u;
    // Zero while any byte of this record may still change.  The transport
    // release-publishes generation here only after header, routes and hidden
    // payload are all remotely complete.  The INC must acquire this field
    // before reading any token metadata.
    uint64_t commit = 0u;
    uint64_t reserved64[3]{};
};

// Optional doorbell batching.  The offset table contains record_count+1
// little-endian uint64_t offsets relative to the frame start.  Every member
// is still a complete header+routes+hidden+commit token record; batching may
// not split metadata from payload or expose either one early.  A transport
// can write the directory once and then release-publish each token commit as
// soon as that complete record is remotely visible.
struct alignas(64) InlineTokenBatchHeaderV2 {
    uint32_t magic = kInlineRouteBatchMagic;
    uint16_t version = kInlineRouteVersion;
    uint16_t header_bytes = kInlineRouteHeaderBytes;
    uint32_t flags = kInlineRouteRecordNone;
    uint32_t source_rank = 0u;
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t request_id = 0u;
    uint32_t wave = 0u;
    uint32_t record_count = 0u;
    uint64_t offset_table_bytes = 0u;
    uint64_t records_offset = 0u;
    uint64_t frame_bytes = 0u;
    // Directory-only commit.  It authorizes the INC to read offsets, never
    // token routes; each enclosed token has its own later commit.
    uint64_t directory_commit = 0u;
    uint64_t reserved64[5]{};
};

// INC-generated metadata for one logical assignment delivered to a resolved
// destination worker.  A fanout record can contain several assignments for
// experts on that worker while transmitting the hidden payload only once.
struct alignas(16) InlineFanoutAssignmentV2 {
    InlineRouteKeyV2 route_key{};
    uint32_t expert_id = 0u;
    uint32_t local_expert = 0u;
    uint32_t route_ordinal = 0u;
    uint32_t flags = 0u;
    uint64_t destination_row = 0u;
    uint64_t reserved64 = 0u;
};

// Layout: header -> assignments -> zero padding -> hidden payload.
struct alignas(64) InlineFanoutRecordHeaderV2 {
    uint32_t magic = kInlineRouteFanoutMagic;
    uint16_t version = kInlineRouteVersion;
    uint16_t header_bytes = kInlineRouteHeaderBytes;
    uint32_t flags = kInlineRouteRecordNone;
    uint32_t resolved_destination_rank = 0u;
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t request_id = 0u;
    uint64_t source_token = 0u;
    uint64_t hidden_bytes = 0u;
    uint64_t payload_offset = 0u;
    uint64_t record_bytes = 0u;
    uint32_t source_rank = 0u;
    uint32_t assignment_count = 0u;
    uint32_t wave = 0u;
    uint32_t reserved32 = 0u;
    uint64_t commit = 0u;
    uint64_t reserved64[3]{};
};

// One worker -> INC expert contribution.  Combine supplies only the opaque
// route key and payload; origin token, expert, route ordinal, destination and
// weight are authoritative in the INC journal.  Transport segmentation is
// transparent, as it is for Dispatch and fanout.
struct alignas(64) InlineCombineRecordHeaderV2 {
    uint32_t magic = kInlineRouteCombineMagic;
    uint16_t version = kInlineRouteVersion;
    uint16_t header_bytes = kInlineRouteHeaderBytes;
    uint32_t flags = kInlineRouteRecordNone;
    uint32_t contributor_rank = 0u;
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t request_id = 0u;
    InlineRouteKeyV2 route_key{};
    uint64_t payload_offset = kInlineRouteHeaderBytes;
    uint64_t payload_bytes = 0u;
    uint64_t record_bytes = 0u;
    uint32_t wave = 0u;
    uint32_t reserved32 = 0u;
    uint64_t commit = 0u;
    uint64_t reserved64[3]{};
};

// INC -> origin worker final reduced result.
struct alignas(64) InlineResultRecordHeaderV2 {
    uint32_t magic = kInlineRouteResultMagic;
    uint16_t version = kInlineRouteVersion;
    uint16_t header_bytes = kInlineRouteHeaderBytes;
    uint32_t flags = kInlineRouteRecordNone;
    uint32_t origin_rank = 0u;
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t request_id = 0u;
    uint64_t source_token = 0u;
    uint64_t payload_offset = kInlineRouteHeaderBytes;
    uint64_t payload_bytes = 0u;
    uint64_t record_bytes = 0u;
    uint32_t wave = 0u;
    uint32_t status = 0u;
    uint64_t commit = 0u;
    uint64_t reserved64[4]{};
};

// Source-owned wave seal or generation lifecycle record. token_count is the
// number of unique endpoint token records and assignment_count is the sum of
// their route_count values.  Physical destination grouping does not change
// either count.
struct alignas(64) InlineRouteControlV2 {
    uint32_t magic = kInlineRouteControlMagic;
    uint16_t version = kInlineRouteVersion;
    uint16_t header_bytes = kInlineRouteHeaderBytes;
    uint32_t kind = 0u;
    uint32_t source_rank = 0u;
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t request_id = 0u;
    uint32_t wave = 0u;
    uint32_t flags = kInlineRouteControlNone;
    uint64_t token_count = 0u;
    uint64_t assignment_count = 0u;
    uint64_t digest = 0u;
    uint64_t reserved64[6]{};
};

static_assert(sizeof(InlineRouteKeyV2) == 16u, "inline route key ABI");
static_assert(sizeof(InlineRouteEntryV2) == 16u, "inline route entry ABI");
static_assert(sizeof(InlineTokenRecordHeaderV2) == 128u,
              "inline token header ABI");
static_assert(sizeof(InlineTokenBatchHeaderV2) == 128u,
              "inline token batch ABI");
static_assert(sizeof(InlineFanoutAssignmentV2) == 48u,
              "inline fanout assignment ABI");
static_assert(sizeof(InlineFanoutRecordHeaderV2) == 128u,
              "inline fanout header ABI");
static_assert(sizeof(InlineCombineRecordHeaderV2) == 128u,
              "inline combine header ABI");
static_assert(sizeof(InlineResultRecordHeaderV2) == 128u,
              "inline result header ABI");
static_assert(sizeof(InlineRouteControlV2) == 128u,
              "inline route control ABI");

#define INC_DC_INLINE_LAYOUT_GATE(type)                                      \
    static_assert(std::is_standard_layout<type>::value, #type " layout");   \
    static_assert(std::is_trivially_copyable<type>::value, #type " copy")
INC_DC_INLINE_LAYOUT_GATE(InlineRouteKeyV2);
INC_DC_INLINE_LAYOUT_GATE(InlineRouteEntryV2);
INC_DC_INLINE_LAYOUT_GATE(InlineTokenRecordHeaderV2);
INC_DC_INLINE_LAYOUT_GATE(InlineTokenBatchHeaderV2);
INC_DC_INLINE_LAYOUT_GATE(InlineFanoutAssignmentV2);
INC_DC_INLINE_LAYOUT_GATE(InlineFanoutRecordHeaderV2);
INC_DC_INLINE_LAYOUT_GATE(InlineCombineRecordHeaderV2);
INC_DC_INLINE_LAYOUT_GATE(InlineResultRecordHeaderV2);
INC_DC_INLINE_LAYOUT_GATE(InlineRouteControlV2);
#undef INC_DC_INLINE_LAYOUT_GATE

static_assert(offsetof(InlineTokenRecordHeaderV2, session_id) == 16u,
              "inline token session offset");
static_assert(offsetof(InlineTokenRecordHeaderV2, hidden_bytes) == 56u,
              "inline token hidden offset");
static_assert(offsetof(InlineTokenRecordHeaderV2, payload_offset) == 64u,
              "inline token payload offset field");
static_assert(offsetof(InlineTokenRecordHeaderV2, route_count) == 80u,
              "inline token route count offset");

inline bool InlineCheckedAdd(uint64_t left, uint64_t right, uint64_t *out)
{
    if (out == nullptr || right > std::numeric_limits<uint64_t>::max() - left)
        return false;
    *out = left + right;
    return true;
}

inline bool InlineCheckedMul(uint64_t left, uint64_t right, uint64_t *out)
{
    if (out == nullptr ||
        (left != 0u && right > std::numeric_limits<uint64_t>::max() / left))
        return false;
    *out = left * right;
    return true;
}

inline bool InlineCheckedAlignUp(uint64_t value, uint64_t alignment,
                                 uint64_t *out)
{
    if (out == nullptr || alignment == 0u ||
        (alignment & (alignment - 1u)) != 0u)
        return false;
    const uint64_t mask = alignment - 1u;
    if (value > std::numeric_limits<uint64_t>::max() - mask) return false;
    *out = (value + mask) & ~mask;
    return true;
}

inline bool InlineTokenRecordLayout(uint32_t route_count,
                                    uint64_t hidden_bytes,
                                    uint64_t *payload_offset,
                                    uint64_t *record_bytes)
{
    if (payload_offset == nullptr || record_bytes == nullptr) return false;
    uint64_t route_bytes = 0u;
    uint64_t metadata_bytes = 0u;
    uint64_t payload = 0u;
    return InlineCheckedMul(route_count, sizeof(InlineRouteEntryV2),
                            &route_bytes) &&
           InlineCheckedAdd(sizeof(InlineTokenRecordHeaderV2), route_bytes,
                            &metadata_bytes) &&
           InlineCheckedAlignUp(metadata_bytes, kInlineRouteWireAlignment,
                                &payload) &&
           InlineCheckedAdd(payload, hidden_bytes, record_bytes) &&
           ((*payload_offset = payload), true);
}

inline bool InlineFanoutRecordLayout(uint32_t assignment_count,
                                     uint64_t hidden_bytes,
                                     uint64_t *payload_offset,
                                     uint64_t *record_bytes)
{
    if (payload_offset == nullptr || record_bytes == nullptr) return false;
    uint64_t assignment_bytes = 0u;
    uint64_t metadata_bytes = 0u;
    uint64_t payload = 0u;
    return InlineCheckedMul(assignment_count,
                            sizeof(InlineFanoutAssignmentV2),
                            &assignment_bytes) &&
           InlineCheckedAdd(sizeof(InlineFanoutRecordHeaderV2),
                            assignment_bytes, &metadata_bytes) &&
           InlineCheckedAlignUp(metadata_bytes, kInlineRouteWireAlignment,
                                &payload) &&
           InlineCheckedAdd(payload, hidden_bytes, record_bytes) &&
           ((*payload_offset = payload), true);
}

inline bool InlinePayloadRecordLayout(uint64_t payload_bytes,
                                      uint64_t *record_bytes)
{
    return InlineCheckedAdd(kInlineRouteHeaderBytes, payload_bytes,
                            record_bytes);
}

inline bool InlineBatchPreambleLayout(uint32_t record_count,
                                      uint64_t *offset_table_bytes,
                                      uint64_t *records_offset)
{
    if (offset_table_bytes == nullptr || records_offset == nullptr) return false;
    uint64_t entries = 0u;
    uint64_t table = 0u;
    uint64_t preamble = 0u;
    return InlineCheckedAdd(record_count, 1u, &entries) &&
           InlineCheckedMul(entries, sizeof(uint64_t), &table) &&
           InlineCheckedAdd(sizeof(InlineTokenBatchHeaderV2), table,
                            &preamble) &&
           InlineCheckedAlignUp(preamble, kInlineRouteWireAlignment,
                                records_offset) &&
           ((*offset_table_bytes = table), true);
}

inline bool InlineStoredRecordEnd(uint64_t record_offset,
                                  uint64_t logical_record_bytes,
                                  uint64_t *next_record_offset)
{
    uint64_t end = 0u;
    return InlineCheckedAdd(record_offset, logical_record_bytes, &end) &&
           InlineCheckedAlignUp(end, kInlineRouteWireAlignment,
                                next_record_offset);
}

} // namespace inc::dc

#endif
