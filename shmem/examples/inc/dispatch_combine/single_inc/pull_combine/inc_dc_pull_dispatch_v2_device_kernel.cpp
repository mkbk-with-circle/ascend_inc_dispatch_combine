#include "kernel_operator.h"
#include "shmem.h"

#include "inc_dc_pull_dispatch_v2_abi.h"

using namespace inc::dc::pull_v2;

// Pull-Dispatch V2 data-plane kernel.
//
// Workspace contract (all sizes are bytes unless stated otherwise):
//
//   source_region
//     Symmetric worker source region.  Slot `ring_slot` starts at
//     ring_slot * source_slot_stride on every worker PE.
//   ready_mailbox
//     Symmetric Ready[worker_count].  Worker r owns element r; INC polls the
//     corresponding local elements after the workers publish them remotely.
//   inc_slots
//     INC-only metadata staging, worker_count * source_slot_stride.  Only the
//     header and [tokens_offset, hidden_offset) metadata are normally
//     populated; non-cacheline row widths use the hidden region for the
//     single-writer exact-width transfer primitive.
//   source_acks
//     Symmetric SourceConsumed[worker_count].
//   destination_hidden / destination_rows / destination_assignments /
//   destination_expert_counts
//     Symmetric destination rings.  The supplied strides are per ring slot.
//     Every destination consumes its own PE-local slot base.
//   destination_completions
//     Symmetric DestinationCompletion[slot_count][worker_count].  Element
//     [slot][destination] is used as INC staging and as the destination's
//     completion mailbox.
//   inc_destination_rows
//     INC-only worker_count slices using an explicit 64B-aligned byte stride;
//     each slice holds destination_row_capacity rows.
//   inc_destination_assignments
//     Same, with destination_assignment_capacity assignments per slice.
//   journal_header / journal_tokens / journal_contributors /
//   journal_assignments
//     Caller-selected INC journal slot.  Contributor ranges are compact and
//     contiguous in source/token order, matching the host Combine V2 model.
//   row_map
//     INC-only uint32_t[journal_token_capacity][worker_count].  Only routed
//     destinations are initialized.
//   source_token_prefix
//     INC-only uint32_t[worker_count + 1].
//   source_destination_prefix
//     INC-only uint32_t[(worker_count + 1) * worker_count].
//   destination_row_counts / destination_assignment_counts
//     INC-only uint32_t[worker_count].
//   expert_counts
//     INC-only uint32_t[worker_count][expert_count].
//   parser_scratch
//     INC-only checked uint32_t workspace.  It holds source-cohort state and
//     per-AIV contributor/row/assignment/expert counts, then prefix bases.
//   status_line
//     PullTimeline (128 bytes).
//
// A valid launch uses worker_count * channels_per_source AIVs, never more
// than the caller-supplied block_dim.  The runtime launches Dispatch with
// half of the device's discovered AIV count (24 on this 48-AIV 910B, 20 on
// a 40-AIV device), leaving the other half available to Combine.  Worker PEs
// use block zero only to publish READY; all payload movement starts at INC.

namespace {

constexpr uint32_t kDefaultChannelsPerSource = 5u;
constexpr uint32_t kUbSlotBytes = 12u * 1024u;
constexpr uint32_t kMaxHiddenTileBytes = 8u * 1024u;
constexpr uint32_t kInvalidRow = ~0u;
constexpr uint32_t kOrdinalVisited = 0x80000000u;

// Device status values intentionally preserve the public Status numbering.
constexpr uint32_t kStatusOk = 0u;
constexpr uint32_t kStatusInvalidArgument = 1u;
constexpr uint32_t kStatusInvalidReady = 3u;
constexpr uint32_t kStatusInvalidHeader = 4u;
constexpr uint32_t kStatusSizeOverflow = 6u;
constexpr uint32_t kStatusCapacityExceeded = 7u;
constexpr uint32_t kStatusDigestMismatch = 8u;
constexpr uint32_t kStatusInvalidToken = 9u;
constexpr uint32_t kStatusInvalidAssignment = 10u;
constexpr uint32_t kStatusInvalidState = 12u;
constexpr uint32_t kStatusReadyTimeout = 13u;
constexpr uint32_t kParserPass2Ready = 0x80000000u;
constexpr uint32_t kParserMetadataReady = 0x40000000u;
constexpr uint32_t kSourceHeaderReady = 1u;
constexpr uint32_t kSourceMetadataReady = 2u;
constexpr uint32_t kDestinationMetadataPublished = 0x4d455441u; // 'META'

constexpr uint64_t kHashOffset = 1469598103934665603ull;

static_assert(kUbSlotBytes * 2u <= 24u * 1024u,
              "Pull Dispatch ping/pong exceeds ordinary AIV UB");
static_assert(kMaxHiddenTileBytes <= kUbSlotBytes,
              "Pull Dispatch transfer exceeds one UB slot");
static_assert(kMaxHiddenTileBytes % kPullDispatchAlignment == 0u,
              "Pull Dispatch tile cap must preserve transport alignment");

__aicore__ inline uint32_t DTypeBytes(uint32_t dtype)
{
    if (dtype == static_cast<uint32_t>(DataType::FP16) ||
        dtype == static_cast<uint32_t>(DataType::BF16))
        return 2u;
    if (dtype == static_cast<uint32_t>(DataType::FP32)) return 4u;
    return 0u;
}

__aicore__ inline bool AddU64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (b > ~0ull - a) return false;
    *out = a + b;
    return true;
}

__aicore__ inline uint64_t MulU32Exact(uint32_t a, uint32_t b)
{
    const uint32_t a0 = a & 0xffffu;
    const uint32_t a1 = a >> 16u;
    const uint32_t b0 = b & 0xffffu;
    const uint32_t b1 = b >> 16u;
    // Each 16x16 product fits in uint32_t, so no generic wide multiply is
    // emitted by Bisheng.  The additions are widened explicitly.
    volatile uint32_t p0_native = a0 * b0;
    volatile uint32_t p1_native = a0 * b1;
    volatile uint32_t p2_native = a1 * b0;
    volatile uint32_t p3_native = a1 * b1;
    const uint64_t p0 = p0_native;
    const uint64_t p1 = p1_native;
    const uint64_t p2 = p2_native;
    const uint64_t p3 = p3_native;
    return p0 + ((p1 + p2) << 16u) + (p3 << 32u);
}

// The dav-2201 compiler lowers a generic 64x64 multiply to compiler-rt's
// unavailable __multi3.  All hot address products here have a 32-bit index;
// split both operands into native 16-bit limbs.
__aicore__ inline uint64_t MulU64ByU32(uint64_t value, uint32_t factor)
{
    return MulU32Exact(static_cast<uint32_t>(value), factor) +
        (MulU32Exact(static_cast<uint32_t>(value >> 32u), factor) << 32u);
}

__attribute__((optnone)) __aicore__ inline bool CheckedMulU64ByU32(
    uint64_t value, uint32_t factor, uint64_t *out)
{
    if (factor != 0u && value > ~0ull / factor) return false;
    *out = MulU64ByU32(value, factor);
    return true;
}

__aicore__ inline bool AlignU64(uint64_t value, uint64_t alignment,
                                uint64_t *out)
{
    uint64_t expanded = 0u;
    if (alignment == 0u || (alignment & (alignment - 1u)) != 0u ||
        !AddU64(value, alignment - 1u, &expanded))
        return false;
    *out = expanded & ~(alignment - 1u);
    return true;
}

__aicore__ inline uint64_t HashByte(uint64_t hash, uint8_t value)
{
    // FNV prime = 2^40 + 0x1b3.  Spell the modular multiplication as
    // shifts/adds because the dav-2201 AIC runtime does not provide the
    // compiler's generic wide-multiply helper (__multi3).
    const uint64_t x = hash ^ value;
    return (x << 40u) + (x << 8u) + (x << 7u) + (x << 5u) +
        (x << 4u) + (x << 1u) + x;
}

__aicore__ inline uint64_t DeviceRouteKey(uint32_t owner_rank,
                                           uint32_t owner_row)
{
    return (static_cast<uint64_t>(owner_rank) << 32u) | owner_row;
}

__aicore__ inline uint64_t HashGmBytes(uint64_t hash,
                                       __gm__ const uint8_t *data,
                                       uint64_t bytes)
{
    for (uint64_t i = 0u; i < bytes; ++i)
        hash = HashByte(hash, data[i]);
    return hash;
}

__aicore__ inline uint64_t ReadyHash(__gm__ const Ready *ready)
{
    // Host ReadyPublication hashes the complete canonical descriptor with
    // publication replaced by zero, including the final eight zero bytes.
    uint64_t hash = HashGmBytes(
        kHashOffset, reinterpret_cast<__gm__ const uint8_t *>(ready),
        __builtin_offsetof(Ready, publication));
    for (uint32_t i = 0u; i < sizeof(uint64_t); ++i)
        hash = HashByte(hash, 0u);
    return hash == 0u ? 1u : hash;
}

__aicore__ inline bool IsFinite(float value)
{
    union Bits {
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    return (bits.u & 0x7f800000u) != 0x7f800000u;
}

__aicore__ inline void FlushRange(__gm__ uint8_t *base, uint64_t bytes)
{
    if (bytes == 0u) return;
    const uint64_t address = reinterpret_cast<uint64_t>(base);
    const uint64_t begin = address &
        ~(static_cast<uint64_t>(kPullDispatchAlignment) - 1u);
    uint64_t end = 0u;
    uint64_t aligned_end = 0u;
    // All callers pass validated HBM ranges.  Keep overflow handling local so
    // a malformed range can never wrap the dcci loop across address zero.
    if (!AddU64(address, bytes, &end) ||
        !AlignU64(end, kPullDispatchAlignment, &aligned_end))
        return;
    for (uint64_t line = begin; line < aligned_end;
         line += kPullDispatchAlignment)
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(line));
}

__aicore__ inline void PutGmRange(__gm__ uint8_t *remote,
                                  __gm__ uint8_t *local, uint64_t bytes,
                                  int32_t destination)
{
    // Keep intermediate calls cache-line aligned.  Only the final call may
    // contain a short tail, matching the canonical wire ABI.
    constexpr uint32_t kMaxChunk = 0x7fffffc0u;
    uint64_t offset = 0u;
    while (bytes != 0u) {
        const uint32_t chunk = static_cast<uint32_t>(
            bytes > kMaxChunk ? kMaxChunk : bytes);
        aclshmem_putmem(remote + offset, local + offset, chunk, destination);
        offset += chunk;
        bytes -= chunk;
    }
}

__aicore__ inline void PutGmRangeAligned(__gm__ uint8_t *remote,
                                         __gm__ uint8_t *local,
                                         uint64_t bytes,
                                         __gm__ uint8_t *tail,
                                         int32_t destination)
{
    const uint64_t full = bytes & ~(static_cast<uint64_t>(
        kPullDispatchAlignment) - 1u);
    if (full != 0u) PutGmRange(remote, local, full, destination);
    const uint32_t remainder = static_cast<uint32_t>(bytes - full);
    if (remainder == 0u) return;
    for (uint32_t byte = 0u; byte < kPullDispatchAlignment; ++byte)
        tail[byte] = byte < remainder ? local[full + byte] : 0u;
    FlushRange(tail, kPullDispatchAlignment);
    PutGmRange(remote + full, tail, kPullDispatchAlignment, destination);
    // The caller may immediately reuse its one cacheline for another metadata
    // stream; close local and remote completion before that reuse.
    aclshmem_quiet();
}

__aicore__ inline bool ReadyValid(
    __gm__ Ready *ready, uint32_t source, uint32_t workers,
    uint64_t session_id, uint64_t placement_epoch, uint64_t generation,
    uint64_t sequence, uint32_t wave, uint32_t region_id,
    uint16_t ring_slot, uint32_t slot_count)
{
    const uint64_t publication = ready->publication;
    return ready->magic == kPullDispatchMagic &&
        ready->abi_version == kPullDispatchAbiVersion &&
        ready->struct_bytes == sizeof(Ready) &&
        ready->session_id == session_id &&
        ready->placement_epoch == placement_epoch &&
        ready->generation == generation && ready->sequence == sequence &&
        ready->wave == wave && ready->source_rank == source &&
        ready->source_rank < workers &&
        ready->source_region_id == region_id &&
        ready->ring_slot == ring_slot && ready->ring_slot < slot_count &&
        ready->flags == 0u && publication != 0u &&
        publication == ReadyHash(ready);
}

__aicore__ inline uint32_t ValidateHeader(
    __gm__ SlotHeader *header, uint32_t source, uint32_t workers,
    uint32_t expert_count, uint32_t expected_hidden,
    uint32_t expected_dtype, uint64_t session_id,
    uint64_t placement_epoch, uint64_t generation, uint64_t sequence,
    uint32_t wave, uint32_t region_id, uint16_t ring_slot,
    uint64_t source_slot_stride)
{
    const uint32_t dtype_bytes = DTypeBytes(header->dtype);
    if (header->magic != kPullDispatchMagic ||
        header->abi_version != kPullDispatchAbiVersion ||
        header->header_bytes != sizeof(SlotHeader) ||
        header->session_id != session_id ||
        header->placement_epoch != placement_epoch ||
        header->generation != generation || header->sequence != sequence ||
        header->wave != wave || header->source_rank != source ||
        header->source_region_id != region_id ||
        header->ring_slot != ring_slot ||
        (header->flags & ~kSlotFlagUniformDestinations) != 0u ||
        header->worker_count != workers || expert_count == 0u ||
        header->hidden != expected_hidden || header->dtype != expected_dtype ||
        dtype_bytes == 0u ||
        header->token_record_bytes != sizeof(TokenRecord) ||
        header->assignment_record_bytes != sizeof(AssignmentRecord) ||
        header->reserved0 != 0u)
        return kStatusInvalidHeader;

    uint64_t token_bytes = 0u;
    uint64_t assignment_bytes = 0u;
    uint64_t hidden_elements = 0u;
    uint64_t hidden_bytes = 0u;
    uint64_t canonical_tokens = 0u;
    uint64_t canonical_assignments = 0u;
    uint64_t canonical_hidden = 0u;
    uint64_t canonical_packet = 0u;
    uint64_t end = 0u;
    if (!CheckedMulU64ByU32(sizeof(TokenRecord), header->token_count,
                            &token_bytes) ||
        !CheckedMulU64ByU32(sizeof(AssignmentRecord),
                            header->assignment_count, &assignment_bytes) ||
        !CheckedMulU64ByU32(header->token_count, header->hidden,
                            &hidden_elements) ||
        !CheckedMulU64ByU32(hidden_elements, dtype_bytes, &hidden_bytes) ||
        !AlignU64(sizeof(SlotHeader), kPullDispatchAlignment,
                  &canonical_tokens) ||
        !AddU64(canonical_tokens, token_bytes, &end) ||
        !AlignU64(end, kPullDispatchAlignment, &canonical_assignments) ||
        !AddU64(canonical_assignments, assignment_bytes, &end) ||
        !AlignU64(end, kPullDispatchAlignment, &canonical_hidden) ||
        !AddU64(canonical_hidden, hidden_bytes, &end) ||
        !AlignU64(end, kPullDispatchAlignment, &canonical_packet))
        return kStatusSizeOverflow;
    if (header->tokens_offset != canonical_tokens ||
        header->assignments_offset != canonical_assignments ||
        header->hidden_offset != canonical_hidden ||
        header->packet_bytes != canonical_packet ||
        canonical_packet > source_slot_stride)
        return kStatusInvalidHeader;
    return kStatusOk;
}

__aicore__ inline uint64_t MetadataDigest(__gm__ SlotHeader *header,
                                           __gm__ uint8_t *slot)
{
    // Hash the canonical header with metadata_digest replaced by zero.  The
    // field is the last uint64_t, so hashing the prefix plus eight zero bytes
    // avoids mutating remotely supplied metadata.
    uint64_t hash = HashGmBytes(
        kHashOffset, reinterpret_cast<__gm__ uint8_t *>(header),
        __builtin_offsetof(SlotHeader, metadata_digest));
    for (uint32_t i = 0u; i < sizeof(uint64_t); ++i)
        hash = HashByte(hash, 0u);
    hash = HashGmBytes(hash, slot + header->tokens_offset,
                      static_cast<uint64_t>(header->token_count) *
                          sizeof(TokenRecord));
    return HashGmBytes(hash, slot + header->assignments_offset,
                       static_cast<uint64_t>(header->assignment_count) *
                           sizeof(AssignmentRecord));
}

__aicore__ inline bool HiddenPaddingZero(__gm__ SlotHeader *header,
                                         __gm__ uint8_t *slot)
{
    const uint64_t padding_begin = header->assignments_offset +
        MulU64ByU32(sizeof(AssignmentRecord), header->assignment_count);
    for (uint64_t offset = padding_begin; offset < header->hidden_offset;
         ++offset)
        if (slot[offset] != 0u) return false;
    return true;
}

__aicore__ inline uint64_t AckPublication(uint64_t generation,
                                           uint64_t sequence,
                                           uint32_t source,
                                           uint32_t status)
{
    uint64_t value = generation ^ (sequence << 1u) ^
        (static_cast<uint64_t>(source) << 48u) ^
        (static_cast<uint64_t>(status) << 24u) ^ 0xa55aa55aa55aa55aull;
    return value == 0u ? 1u : value;
}

__aicore__ inline uint64_t CompletionPublication(uint64_t generation,
                                                  uint64_t sequence,
                                                  uint32_t destination,
                                                  uint32_t status)
{
    uint64_t value = generation ^ (sequence << 3u) ^
        (static_cast<uint64_t>(destination) << 44u) ^
        (static_cast<uint64_t>(status) << 20u) ^ 0x5aa55aa55aa55aa5ull;
    return value == 0u ? 1u : value;
}

__aicore__ inline void PublishSourceAck(
    __gm__ SourceConsumed *ack, uint32_t source, uint32_t status,
    uint64_t bytes_consumed, uint64_t session_id, uint64_t generation,
    uint64_t sequence, uint64_t placement_epoch, uint64_t dispatch_cookie,
    uint32_t region_id, uint32_t wave, uint16_t ring_slot)
{
    ack->magic = kPullDispatchMagic;
    ack->abi_version = kPullDispatchAbiVersion;
    ack->struct_bytes = sizeof(SourceConsumed);
    ack->session_id = session_id;
    ack->placement_epoch = placement_epoch;
    ack->generation = generation;
    ack->sequence = sequence;
    ack->dispatch_cookie = status == kStatusOk ? dispatch_cookie : 0u;
    ack->wave = wave;
    ack->source_rank = source;
    ack->source_region_id = region_id;
    ack->status = status;
    ack->ring_slot = ring_slot;
    ack->flags = 0u;
    ack->reserved0 = 0u;
    ack->bytes_consumed = status == kStatusOk ? bytes_consumed : 0u;
    ack->publication = 0u;
    for (uint32_t i = 0u; i < 5u; ++i) ack->reserved[i] = 0u;
    FlushRange(reinterpret_cast<__gm__ uint8_t *>(ack),
               sizeof(SourceConsumed));
    aclshmem_uint64_p(&ack->publication, 0u,
                      static_cast<int32_t>(source));
    aclshmem_quiet();
    aclshmem_putmem(ack, ack,
                    __builtin_offsetof(SourceConsumed, publication),
                    static_cast<int32_t>(source));
    aclshmem_quiet();
    aclshmem_uint64_p(
        &ack->publication,
        AckPublication(generation, sequence, source, status),
        static_cast<int32_t>(source));
    aclshmem_quiet();
}

__aicore__ inline void PublishDestinationCompletion(
    __gm__ DestinationCompletion *completion, uint32_t destination,
    uint32_t status, uint32_t rows, uint32_t assignments,
    uint64_t session_id, uint64_t placement_epoch, uint64_t generation,
    uint64_t sequence, uint64_t dispatch_cookie, uint32_t wave,
    uint16_t ring_slot)
{
    completion->magic = kPullDispatchMagic;
    completion->abi_version = kPullDispatchAbiVersion;
    completion->struct_bytes = sizeof(DestinationCompletion);
    completion->session_id = session_id;
    completion->placement_epoch = placement_epoch;
    completion->generation = generation;
    completion->sequence = sequence;
    completion->dispatch_cookie = status == kStatusOk ? dispatch_cookie : 0u;
    completion->wave = wave;
    completion->destination_rank = destination;
    completion->status = status;
    completion->row_count = status == kStatusOk ? rows : 0u;
    completion->assignment_count = status == kStatusOk ? assignments : 0u;
    completion->ring_slot = ring_slot;
    completion->flags = 0u;
    completion->publication = 0u;
    for (uint32_t i = 0u; i < 6u; ++i) completion->reserved[i] = 0u;
    FlushRange(reinterpret_cast<__gm__ uint8_t *>(completion),
               sizeof(DestinationCompletion));
    aclshmem_uint64_p(&completion->publication, 0u,
                      static_cast<int32_t>(destination));
    aclshmem_quiet();
    aclshmem_putmem(completion, completion,
                    __builtin_offsetof(DestinationCompletion, publication),
                    static_cast<int32_t>(destination));
    aclshmem_quiet();
    aclshmem_uint64_p(
        &completion->publication,
        CompletionPublication(generation, sequence, destination, status),
        static_cast<int32_t>(destination));
    aclshmem_quiet();
}

__aicore__ inline void SetGetReady(uint32_t slot)
{
    if (slot == 0u)
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0u);
    else if (slot == 1u)
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(1u);
    else
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(2u);
}

__aicore__ inline void WaitGetReady(uint32_t slot)
{
    if (slot == 0u)
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0u);
    else if (slot == 1u)
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(1u);
    else
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(2u);
}

__aicore__ inline void SetPutDone(uint32_t slot)
{
    if (slot == 0u)
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
    else if (slot == 1u)
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
    else
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(2u);
}

__aicore__ inline void WaitPutDone(uint32_t slot)
{
    if (slot == 0u)
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
    else if (slot == 1u)
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
    else
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(2u);
}

__aicore__ inline void PullHiddenToUb(__ubuf__ uint8_t *ub,
                                      __gm__ uint8_t *remote,
                                      uint32_t bytes, int32_t source,
                                      uint32_t ping)
{
    aclshmemx_mte_get_nbi(
        reinterpret_cast<__ubuf__ int8_t *>(ub),
        reinterpret_cast<__gm__ int8_t *>(remote), bytes, source, ping);
    SetGetReady(ping);
}

__aicore__ inline void WaitHiddenPull(uint32_t ping)
{
    WaitGetReady(ping);
}

__aicore__ inline void PutHiddenFromUb(__gm__ uint8_t *remote,
                                       __ubuf__ uint8_t *ub,
                                       uint32_t bytes, int32_t destination,
                                       uint32_t ping)
{
    aclshmemx_mte_put_nbi(
        reinterpret_cast<__gm__ int8_t *>(remote),
        reinterpret_cast<__ubuf__ int8_t *>(ub), bytes, destination, ping);
}

// All parser scratch is uint32_t-addressed and caller-owned.  Pass one writes
// per-AIV counts; the single, bounded prefix phase replaces those counts with
// deterministic source-major bases; pass two therefore writes disjoint final
// ranges without atomics.  In particular, compact contributors keep the exact
// host-oracle order: source, token, destination rank.
struct ParserScratchLayout {
    uint64_t source_state;
    uint64_t source_error;
    uint64_t source_assignment_prefix;
    uint64_t block_status;
    uint64_t block_contributors;
    uint64_t block_rows;
    uint64_t block_assignments;
    uint64_t block_experts;
    uint64_t metadata_tails;
    uint64_t block_row_ends;
    uint64_t block_assignment_ends;
    uint64_t block_contributor_ends;
    uint64_t boundary_records;
    uint64_t boundary_block_stride;
    uint64_t source_stride;
    uint64_t block_stride;
    uint64_t row_stride;
    uint64_t expert_stride;
    uint64_t entries;
};

__aicore__ inline bool BuildParserScratchLayout(
    uint32_t active_blocks, uint32_t workers, uint32_t experts,
    uint64_t capacity, ParserScratchLayout *layout)
{
    uint64_t cursor = 0u;
    uint64_t extent = 0u;
    uint64_t expert_values = 0u;
    if (layout == nullptr || active_blocks == 0u || workers == 0u ||
        experts == 0u ||
        !CheckedMulU64ByU32(workers, experts, &expert_values) ||
        !AlignU64(workers, 16u, &layout->row_stride) ||
        !AlignU64(expert_values, 16u, &layout->expert_stride))
        return false;
    layout->source_stride = 16u;
    layout->block_stride = 16u;
    layout->source_state = cursor;
    if (!CheckedMulU64ByU32(layout->source_stride, workers, &extent) ||
        !AddU64(cursor, extent, &cursor)) return false;
    layout->source_error = cursor;
    if (!AddU64(cursor, extent, &cursor)) return false;
    layout->source_assignment_prefix = cursor;
    if (!AddU64(cursor, static_cast<uint64_t>(workers) + 1u, &cursor))
        return false;
    if (!AlignU64(cursor, 16u, &cursor)) return false;
    layout->block_status = cursor;
    if (!CheckedMulU64ByU32(layout->block_stride, active_blocks, &extent) ||
        !AddU64(cursor, extent, &cursor)) return false;
    layout->block_contributors = cursor;
    if (!AddU64(cursor, extent, &cursor)) return false;
    layout->block_rows = cursor;
    if (!CheckedMulU64ByU32(layout->row_stride, active_blocks, &extent) ||
        !AddU64(cursor, extent, &cursor)) return false;
    layout->block_assignments = cursor;
    if (!AddU64(cursor, extent, &cursor)) return false;
    layout->block_experts = cursor;
    if (!CheckedMulU64ByU32(layout->expert_stride, active_blocks, &extent) ||
        !AddU64(cursor, extent, &cursor)) return false;
    layout->metadata_tails = cursor;
    if (!CheckedMulU64ByU32(16u, workers, &extent) ||
        !AddU64(cursor, extent, &cursor)) return false;
    layout->block_row_ends = cursor;
    if (!CheckedMulU64ByU32(layout->row_stride, active_blocks, &extent) ||
        !AddU64(cursor, extent, &cursor)) return false;
    layout->block_assignment_ends = cursor;
    if (!AddU64(cursor, extent, &cursor)) return false;
    layout->block_contributor_ends = cursor;
    if (!CheckedMulU64ByU32(16u, active_blocks, &extent) ||
        !AddU64(cursor, extent, &cursor)) return false;
    layout->boundary_records = cursor;
    layout->boundary_block_stride = (2u + static_cast<uint64_t>(workers) * 2u) * 8u * 32u;
    if (!CheckedMulU64ByU32(layout->boundary_block_stride, active_blocks, &extent) ||
        !AddU64(cursor, extent, &cursor)) return false;
    layout->entries = cursor;
    return cursor <= capacity;
}

// Parallel producers own full records, but not necessarily full cachelines.
// Redirect only boundary records to private 128B slots. One repair owner later
// copies those records after all producers have flushed their interiors.
__aicore__ inline __gm__ uint8_t *BoundaryRecord(
    __gm__ uint8_t *record, __gm__ uint8_t *begin, __gm__ uint8_t *end,
    uint32_t bytes, __gm__ uint8_t *private_slots)
{
    const uint64_t a = reinterpret_cast<uint64_t>(record);
    const uint64_t b = reinterpret_cast<uint64_t>(begin);
    const uint64_t e = reinterpret_cast<uint64_t>(end);
    uint64_t slot = 8u;
    if ((b & 63u) != 0u && a < ((b + 63u) & ~63ull))
        slot = (a - b) / bytes;
    else if ((e & 63u) != 0u && a + bytes > (e & ~63ull))
        slot = 4u + (e - a - bytes) / bytes;
    if (slot == 8u) return record;
    __gm__ uint8_t *entry = private_slots + slot * 128u;
    *reinterpret_cast<__gm__ uint64_t *>(entry) = a;
    *reinterpret_cast<__gm__ uint32_t *>(entry + 8u) = bytes;
    return entry + 64u;
}

__aicore__ inline uint32_t RelayTileBytes(uint32_t contributors,
                                          uint32_t channels_per_source)
{
    if (contributors == 0u) return kMaxHiddenTileBytes;
    // Qualified rule: up to four channels use a 16-KiB aggregate window;
    // denser schedules use 12 KiB.  Every individual transfer is capped at
    // the empirically stable 8-KiB point even though each UB slot is 12 KiB.
    // The low-channel fanout-2 transport point uses a smaller aggregate
    // window while metadata work is active on the remaining AIVs.
    if (channels_per_source <= 3u && contributors == 2u)
        return 6u * 1024u;
    // One UB tile is reused for every destination. Fanout four therefore
    // does not consume four independent UB windows; keep each transfer at the
    // qualified 8-KiB MTE ceiling instead of shrinking it to 4 KiB.
    if (contributors == 4u && channels_per_source <= 4u)
        return kMaxHiddenTileBytes;
    const uint32_t in_flight_budget = channels_per_source <= 4u
        ? 16u * 1024u : kUbSlotBytes;
    uint32_t bytes = in_flight_budget / contributors;
    bytes &= ~(kPullDispatchAlignment - 1u);
    if (bytes > kMaxHiddenTileBytes) bytes = kMaxHiddenTileBytes;
    // At kPullDispatchMaxWorkers=128 this remains nonzero (64 bytes).
    return bytes < kPullDispatchAlignment ? kPullDispatchAlignment : bytes;
}

// Keep parser boundaries on eight-token groups.  For the canonical dense
// layout this aligns contributor, destination-row, assignment and row-map
// block boundaries to 64 B, avoiding scalar-cache false sharing between AIVs.
__aicore__ inline uint32_t ParserTokenBoundary(
    uint32_t token_count, uint32_t lane, uint32_t cohort)
{
    const uint64_t groups =
        (static_cast<uint64_t>(token_count) + 7u) / 8u;
    const uint64_t group = MulU64ByU32(groups, lane) / cohort;
    const uint64_t token = group * 8u;
    return token < token_count ? static_cast<uint32_t>(token) : token_count;
}

__aicore__ inline bool WaitParserReady(
    __gm__ uint32_t *block_status, uint64_t block_stride,
    uint32_t parser, uint64_t spin_cap)
{
    __gm__ uint32_t *cell = block_status +
        static_cast<uint64_t>(parser) * block_stride;
    for (uint64_t spin = 0u; spin < spin_cap; ++spin) {
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(cell));
        if (*cell == kParserPass2Ready) return true;
    }
    return false;
}

struct RelayBuffers {
    bool put_busy[2]{false, false};
    uint32_t next_ping = 0u;
};

__aicore__ inline void DrainRelay(RelayBuffers *buffers)
{
    for (uint32_t slot = 0u; slot < 2u; ++slot) {
        if (buffers->put_busy[slot]) WaitPutDone(slot);
        buffers->put_busy[slot] = false;
    }
    aclshmemx_mte_quiet();
}

__aicore__ inline void RelayRun(
    __gm__ uint8_t *source_hidden, __gm__ uint8_t *destination_hidden_base,
    uint64_t source_bytes, uint32_t source, uint32_t channel,
    uint32_t channels_per_source, uint32_t destination_count,
    uint32_t *destinations, uint64_t *destination_byte_base, RelayBuffers *buffers,
    uint32_t requested_tile_bytes)
{
    if (destination_count == 0u || source_bytes == 0u) return;
    const uint32_t tile_bytes = requested_tile_bytes != 0u ? requested_tile_bytes :
        RelayTileBytes(destination_count, channels_per_source);
    const uint64_t tiles = (source_bytes + tile_bytes - 1u) / tile_bytes;
    __ubuf__ uint8_t *ub[2]{
        reinterpret_cast<__ubuf__ uint8_t *>(0),
        reinterpret_cast<__ubuf__ uint8_t *>(kUbSlotBytes)};
    bool *put_busy = buffers->put_busy;
    uint32_t ping = buffers->next_ping;
    uint64_t tile = channel;
    if (tile < tiles) {
        if (put_busy[ping]) {
            WaitPutDone(ping);
            put_busy[ping] = false;
        }
        uint64_t byte_offset = tile * tile_bytes;
        uint32_t bytes = static_cast<uint32_t>(
            source_bytes - byte_offset < tile_bytes
                ? source_bytes - byte_offset : tile_bytes);
        PullHiddenToUb(ub[ping],
            source_hidden + byte_offset,
            bytes, static_cast<int32_t>(source), ping);
        while (tile < tiles) {
            WaitHiddenPull(ping);
            const uint64_t next_tile = tile + channels_per_source;
            const uint32_t next_ping = ping ^ 1u;
            uint64_t next_offset = 0u;
            uint32_t next_bytes = 0u;
            if (next_tile < tiles) {
                if (put_busy[next_ping]) {
                    WaitPutDone(next_ping);
                    put_busy[next_ping] = false;
                }
                next_offset = next_tile * tile_bytes;
                next_bytes = static_cast<uint32_t>(
                    source_bytes - next_offset < tile_bytes
                        ? source_bytes - next_offset : tile_bytes);
                PullHiddenToUb(ub[next_ping],
                    source_hidden + next_offset,
                    next_bytes, static_cast<int32_t>(source), next_ping);
            }
            for (uint32_t local = 0u; local < destination_count; ++local) {
                PutHiddenFromUb(
                    destination_hidden_base +
                        destination_byte_base[local] + byte_offset,
                    ub[ping], bytes,
                    static_cast<int32_t>(destinations[local]), ping);
            }
            SetPutDone(ping);
            put_busy[ping] = true;
            if (next_tile >= tiles) break;
            tile = next_tile;
            ping = next_ping;
            byte_offset = next_offset;
            bytes = next_bytes;
        }
    }
    buffers->next_ping = ping ^ 1u;
}


// Keep exact wide addressing without running the compiler-workaround limb
// multiply on every row of a power-of-two hidden size. Arbitrary sizes retain
// the exact multiply; no shape is rejected to obtain this optimization.
__aicore__ inline uint64_t RelayRowOffset(uint64_t row_bytes, uint32_t row,
                                       uint32_t shift)
{
    return shift < 64u ? static_cast<uint64_t>(row) << shift
                       : MulU64ByU32(row_bytes, row);
}

struct RelayTileTask {
    uint64_t offset;
    uint64_t sub;
    uint32_t token;
    uint32_t bytes;
    bool valid;
};

__aicore__ inline uint32_t RelayBatchBytes(uint64_t row_bytes, uint32_t token,
    uint64_t sub, uint32_t tokens, uint32_t tile_bytes, uint32_t row_shift)
{
    const uint32_t block_left = 8u - token % 8u;
    const uint32_t rows = tokens - token < block_left ? tokens - token : block_left;
    const uint64_t remaining = RelayRowOffset(row_bytes, rows, row_shift) - sub;
    return static_cast<uint32_t>(remaining < tile_bytes ? remaining : tile_bytes);
}

__aicore__ inline void AdvanceRelayTask(RelayTileTask &task, bool coalesced,
    uint64_t row_bytes, uint64_t source_bytes, uint32_t tokens,
    uint32_t channels, uint32_t tile_bytes, uint32_t row_shift)
{
    if (!task.valid) return;
    if (coalesced) {
        const uint64_t step = static_cast<uint64_t>(channels) * tile_bytes;
        task.valid = step < source_bytes - task.offset;
        if (task.valid) task.offset += step;
    } else {
        uint64_t remaining = task.bytes;
        while (remaining >= row_bytes - task.sub) {
            remaining -= row_bytes - task.sub;
            task.sub = 0u;
            ++task.token;
        }
        task.sub += remaining;
        if (task.sub == 0u && task.token % 8u == 0u) {
            const uint32_t skip = (channels - 1u) * 8u;
            task.valid = task.token < tokens && tokens - task.token > skip;
            if (task.valid) task.token += skip;
        } else {
            task.valid = task.token < tokens;
        }
        if (task.valid)
            task.offset = RelayRowOffset(row_bytes, task.token, row_shift) + task.sub;
    }
    if (task.valid) {
        const uint64_t remaining = source_bytes - task.offset;
        task.bytes = coalesced
            ? static_cast<uint32_t>(remaining < tile_bytes ? remaining : tile_bytes)
            : RelayBatchBytes(row_bytes, task.token, task.sub, tokens, tile_bytes, row_shift);
    }
}

// One tile loop crosses token boundaries. Source prefetch does not depend on
// the next destination descriptor: READY/header validation already proves its
// source range. Destination resolution is performed while that GET is live.
__attribute__((noinline)) __aicore__ void RelayMappedTasks(
    __gm__ uint8_t *source_base, __gm__ uint8_t *destination_base,
    __gm__ uint32_t *prefix, __gm__ SlotHeader *header,
    uint64_t row_bytes, uint32_t source, uint32_t workers,
    uint32_t channel, uint32_t channels, __gm__ uint32_t *error,
    __gm__ uint32_t *row_map, __gm__ uint32_t *token_prefix,
    __gm__ uint32_t *block_status, uint64_t block_stride,
    uint32_t parser_cohort, uint64_t spin_cap)
{
    if (header->token_count == 0u) return;
    const bool coalesced = (header->flags & kSlotFlagUniformDestinations) != 0u;
    uint32_t row_shift = 0u;
    uint64_t stride_bits = row_bytes;
    while (stride_bits > 1u && (stride_bits & 1u) == 0u) {
        stride_bits >>= 1u;
        ++row_shift;
    }
    if (stride_bits != 1u) row_shift = 64u;
    __gm__ uint8_t *mapped_source = reinterpret_cast<__gm__ uint8_t *>(
        aclshmem_ptr(source_base + header->hidden_offset, static_cast<int32_t>(source)));
    __gm__ uint8_t *mapped_destinations[kPullDispatchMaxWorkers];
    for (uint32_t destination = 0u; destination < workers; ++destination)
        mapped_destinations[destination] = reinterpret_cast<__gm__ uint8_t *>(
            aclshmem_ptr(destination_base, static_cast<int32_t>(destination)));
    __gm__ uint8_t *slot = reinterpret_cast<__gm__ uint8_t *>(header);
    __gm__ TokenRecord *tokens = reinterpret_cast<__gm__ TokenRecord *>(slot + header->tokens_offset);
    __gm__ AssignmentRecord *assignments = reinterpret_cast<__gm__ AssignmentRecord *>(slot + header->assignments_offset);
    uint32_t destinations[kPullDispatchMaxWorkers];
    uint64_t offsets[kPullDispatchMaxWorkers];
    uint64_t run_remote[kPullDispatchMaxWorkers];
    uint32_t run_local[kPullDispatchMaxWorkers];
    uint32_t run_bytes[kPullDispatchMaxWorkers];
    uint32_t count = 0u;
    if (coalesced) {
        uint32_t bits[4]{0u, 0u, 0u, 0u};
        for (uint32_t i = 0u; i < tokens[0].assignment_count; ++i) {
            const uint32_t destination = assignments[tokens[0].assignment_begin + i].destination_rank;
            const uint32_t mask = 1u << (destination & 31u);
            if (bits[destination >> 5u] & mask) continue;
            bits[destination >> 5u] |= mask;
            destinations[count] = destination;
            offsets[count++] = RelayRowOffset(row_bytes, prefix[static_cast<uint64_t>(source) * workers + destination], row_shift);
        }
        if (count == 0u) return;
    }
    constexpr uint32_t kTokenTileBytes = 64u * 1024u;
    static_assert(2u * kTokenTileBytes <= ub_limit,
                  "streaming relay exceeds the SHMEM backend UB capacity");
    const uint32_t tile_bytes = coalesced ? RelayTileBytes(count, channels) : kTokenTileBytes;
    const uint32_t ub_stride = coalesced ? kUbSlotBytes : kTokenTileBytes;
    const uint32_t slots = 2u;
    const uint64_t source_bytes = RelayRowOffset(row_bytes, header->token_count, row_shift);
    // Keep a short contiguous source block on one relay. Token-at-a-time
    // striping creates unnecessary source and destination address jumps.
    constexpr uint32_t kTokensPerRelayBlock = 8u;
    uint32_t token = coalesced ? 0u : channel * kTokensPerRelayBlock;
    uint64_t sub = 0u;
    uint64_t source_offset = coalesced ? static_cast<uint64_t>(channel) * tile_bytes :
                                        RelayRowOffset(row_bytes, token, row_shift);
    if (source_offset >= source_bytes) return;
    uint32_t bytes = static_cast<uint32_t>(
        (coalesced ? source_bytes - source_offset : row_bytes) < tile_bytes
        ? (coalesced ? source_bytes - source_offset : row_bytes) : tile_bytes);
    if (!coalesced)
        bytes = RelayBatchBytes(row_bytes, token, sub, header->token_count, tile_bytes, row_shift);
    __ubuf__ uint8_t *ub[3]{reinterpret_cast<__ubuf__ uint8_t *>(0),
                           reinterpret_cast<__ubuf__ uint8_t *>(ub_stride),
                           reinterpret_cast<__ubuf__ uint8_t *>(ub_stride * 2u)};
    bool busy[3]{false, false, false};
    bool queued[3]{false, false, false};
    RelayTileTask tasks[3];
    RelayTileTask pending{source_offset, sub, token, bytes, true};
    uint32_t ping = 0u, cached_token = kInvalidRow;
    uint32_t ready_until = 0u, lane = 0u;
    // Populate the entire bounded GET window before the first fan-out.
    // Recycling a slot is ordered after its own PUTs, not after every PUT.
    for (uint32_t i = 0u; i < slots && pending.valid; ++i) {
        tasks[i] = pending;
        aclshmemi_copy_gm2ub(reinterpret_cast<__ubuf__ int8_t *>(ub[i]),
            reinterpret_cast<__gm__ int8_t *>(mapped_source + pending.offset), pending.bytes);
        SetGetReady(i);
        queued[i] = true;
        AdvanceRelayTask(pending, coalesced, row_bytes, source_bytes,
            header->token_count, channels, tile_bytes, row_shift);
    }
    for (;;) {
        source_offset = tasks[ping].offset;
        sub = tasks[ping].sub;
        token = tasks[ping].token;
        bytes = tasks[ping].bytes;
        WaitHiddenPull(ping);
        queued[ping] = false;
        bool valid = true;
        uint32_t slice = 0u;
        if (!coalesced)
            for (uint32_t destination = 0u; destination < workers; ++destination)
                run_bytes[destination] = 0u;
        while (slice < bytes && valid) {
        const uint32_t slice_bytes = coalesced ? bytes : static_cast<uint32_t>(
            row_bytes - sub < bytes - slice ? row_bytes - sub : bytes - slice);
        if (!coalesced && cached_token != token) {
            if (token >= ready_until) {
                while (lane + 1u < parser_cohort &&
                       token >= ParserTokenBoundary(header->token_count, lane + 1u, parser_cohort))
                    ++lane;
                __gm__ uint32_t *progress = block_status +
                    static_cast<uint64_t>(source * parser_cohort + lane) * block_stride + 4u;
                uint32_t observed = 0u;
                for (uint64_t spin = 0u; spin < spin_cap; ++spin) {
                    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(progress));
                    observed = *reinterpret_cast<__gm__ volatile uint32_t *>(progress);
                    if (observed > token) break;
                }
                valid = observed > token;
                if (!valid) *error = kStatusReadyTimeout;
                if (valid) {
                    // A progress publication releases this immutable range.
                    // Invalidate it once, not once per token (several tokens
                    // share each scalar cacheline). Never touch unpublished
                    // rows, which their producer may still be modifying.
                    __gm__ uint32_t *published = row_map +
                        (static_cast<uint64_t>(token_prefix[source]) + token) * workers;
                    FlushRange(reinterpret_cast<__gm__ uint8_t *>(published),
                        static_cast<uint64_t>(observed - token) * workers * sizeof(uint32_t));
                }
                ready_until = observed;
            }
            if (valid) {
                __gm__ uint32_t *rows = row_map +
                    (static_cast<uint64_t>(token_prefix[source]) + token) * workers;
                count = 0u;
                uint32_t destination = (source + channel) % workers;
                for (uint32_t local = 0u; local < workers; ++local) {
                    const uint32_t row = rows[destination];
                    if (row != kInvalidRow) {
                        if (row < prefix[static_cast<uint64_t>(source) * workers + destination] ||
                            row >= prefix[static_cast<uint64_t>(source + 1u) * workers + destination]) {
                            *error = kStatusInvalidState;
                            valid = false;
                            break;
                        }
                        destinations[count] = destination;
                        offsets[count++] = RelayRowOffset(row_bytes, row, row_shift);
                    }
                    if (++destination == workers) destination = 0u;
                }
                cached_token = token;
            }
        }
        if (!valid) break;
        for (uint32_t i = 0u; i < count; ++i) {
            const uint32_t destination = destinations[i];
            if (coalesced) {
                aclshmemi_copy_ub2gm(
                    reinterpret_cast<__gm__ int8_t *>(mapped_destinations[destination] + offsets[i] + source_offset),
                    reinterpret_cast<__ubuf__ int8_t *>(ub[ping]), slice_bytes);
                continue;
            }
            // Merge only if both local and remote byte ranges are adjacent.
            // Different expert assignments do not prevent hidden coalescing;
            // destination metadata still retains every assignment separately.
            const uint64_t remote = offsets[i] + sub;
            if (run_bytes[destination] != 0u &&
                run_remote[destination] + run_bytes[destination] == remote &&
                run_local[destination] + run_bytes[destination] == slice) {
                run_bytes[destination] += slice_bytes;
            } else {
                if (run_bytes[destination] != 0u)
                    aclshmemi_copy_ub2gm(
                        reinterpret_cast<__gm__ int8_t *>(mapped_destinations[destination] + run_remote[destination]),
                        reinterpret_cast<__ubuf__ int8_t *>(ub[ping] + run_local[destination]), run_bytes[destination]);
                run_remote[destination] = remote;
                run_local[destination] = slice;
                run_bytes[destination] = slice_bytes;
            }
        }
        slice += slice_bytes;
        sub += slice_bytes;
        if (!coalesced && sub == row_bytes) {
            ++token;
            sub = 0u;
        }
        }
        if (!valid) {
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(error));
            // Consume every speculative event, including error paths.
            for (uint32_t i = 0u; i < slots; ++i)
                if (queued[i]) WaitHiddenPull(i);
            break;
        }
        if (!coalesced)
            for (uint32_t destination = 0u; destination < workers; ++destination)
                if (run_bytes[destination] != 0u)
                    aclshmemi_copy_ub2gm(
                        reinterpret_cast<__gm__ int8_t *>(mapped_destinations[destination] + run_remote[destination]),
                        reinterpret_cast<__ubuf__ int8_t *>(ub[ping] + run_local[destination]), run_bytes[destination]);
        // Includes every token slice in this source GET batch.
        SetPutDone(ping);
        busy[ping] = true;
        if (pending.valid) {
            WaitPutDone(ping);
            busy[ping] = false;
            tasks[ping] = pending;
            aclshmemi_copy_gm2ub(reinterpret_cast<__ubuf__ int8_t *>(ub[ping]),
                reinterpret_cast<__gm__ int8_t *>(mapped_source + pending.offset), pending.bytes);
            SetGetReady(ping);
            queued[ping] = true;
            AdvanceRelayTask(pending, coalesced, row_bytes, source_bytes,
                header->token_count, channels, tile_bytes, row_shift);
        }
        ping = ping + 1u == slots ? 0u : ping + 1u;
        if (!queued[ping]) break;
    }
    for (uint32_t i = 0u; i < slots; ++i)
        if (busy[i]) WaitPutDone(i);
    aclshmemx_mte_quiet();
}

// A run is contiguous source hidden plus one contiguous interval per unique
// destination. Verified uniform input is a single run; arbitrary routing is
// a sequence of token runs. All fanouts share the same relay executor.
__attribute__((noinline)) __aicore__ void RelaySourceRuns(
    __gm__ uint8_t *source_base, __gm__ uint8_t *destination_hidden_base,
    __gm__ uint32_t *prefix, __gm__ SlotHeader *header,
    __gm__ uint8_t *destination_rows, uint64_t rows_stride,
    uint64_t row_bytes, uint32_t source, uint32_t workers,
    uint32_t channel, uint32_t channels, __gm__ uint32_t *error,
    __gm__ uint32_t *row_map, __gm__ uint32_t *token_prefix,
    __gm__ uint32_t *block_status, uint64_t block_stride,
    uint32_t parser_cohort, uint64_t spin_cap)
{
    if (row_bytes % kPullDispatchAlignment == 0u &&
        (row_map != nullptr || (header->flags & kSlotFlagUniformDestinations) != 0u)) {
        RelayMappedTasks(source_base, destination_hidden_base, prefix, header,
            row_bytes, source, workers, channel, channels, error, row_map,
            token_prefix, block_status, block_stride, parser_cohort, spin_cap);
        return;
    }
    RelayBuffers buffers{};
    uint32_t ready_until = 0u, parser_lane = 0u;
    __gm__ uint8_t *slot = reinterpret_cast<__gm__ uint8_t *>(header);
    __gm__ TokenRecord *tokens = reinterpret_cast<__gm__ TokenRecord *>(slot + header->tokens_offset);
    __gm__ AssignmentRecord *assignments =
        reinterpret_cast<__gm__ AssignmentRecord *>(slot + header->assignments_offset);
    const bool coalesced = (header->flags & kSlotFlagUniformDestinations) != 0u;
    const bool aligned = row_bytes % kPullDispatchAlignment == 0u;
    for (uint32_t token = coalesced ? 0u : channel; token < header->token_count;) {
        if (!coalesced && row_map != nullptr && token >= ready_until) {
            while (parser_lane + 1u < parser_cohort &&
                   token >= ParserTokenBoundary(header->token_count, parser_lane + 1u, parser_cohort))
                ++parser_lane;
            __gm__ uint32_t *progress = block_status +
                static_cast<uint64_t>(source * parser_cohort + parser_lane) * block_stride + 4u;
            uint32_t observed = 0u;
            for (uint64_t spin = 0u; spin < spin_cap; ++spin) {
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(progress));
                observed = *reinterpret_cast<__gm__ volatile uint32_t *>(progress);
                if (observed > token) break;
            }
            if (observed <= token) {
                *error = kStatusReadyTimeout;
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(error));
                DrainRelay(&buffers);
                return;
            }
            ready_until = observed;
        }
        uint32_t destinations[kPullDispatchMaxWorkers];
        uint64_t offsets[kPullDispatchMaxWorkers];
        uint32_t bits[4]{0u, 0u, 0u, 0u};
        uint32_t count = 0u;
        if (!coalesced && row_map != nullptr) {
            __gm__ uint32_t *rows = row_map +
                (static_cast<uint64_t>(token_prefix[source]) + token) * workers;
            FlushRange(reinterpret_cast<__gm__ uint8_t *>(rows), workers * sizeof(uint32_t));
            uint32_t destination = (source + channel) % workers;
            for (uint32_t local = 0u; local < workers; ++local) {
                const uint32_t row = rows[destination];
                if (row != kInvalidRow) {
                    if (row < prefix[static_cast<uint64_t>(source) * workers + destination] ||
                        row >= prefix[static_cast<uint64_t>(source + 1u) * workers + destination]) {
                        *error = kStatusInvalidState;
                        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(error));
                        DrainRelay(&buffers);
                        return;
                    }
                    destinations[count] = destination;
                    offsets[count++] = MulU64ByU32(row_bytes, row);
                }
                if (++destination == workers) destination = 0u;
            }
        } else {
        for (uint32_t i = 0u; i < tokens[token].assignment_count; ++i) {
            const uint32_t destination =
                assignments[tokens[token].assignment_begin + i].destination_rank;
            const uint32_t mask = 1u << (destination & 31u);
            if (bits[destination >> 5u] & mask) continue;
            bits[destination >> 5u] |= mask;
            uint32_t row = prefix[static_cast<uint64_t>(source) * workers + destination] + token;
            if (!coalesced) {
                uint32_t lo = prefix[static_cast<uint64_t>(source) * workers + destination];
                uint32_t hi = prefix[static_cast<uint64_t>(source + 1u) * workers + destination];
                const uint32_t limit = hi;
                __gm__ DestinationRow *layout = reinterpret_cast<__gm__ DestinationRow *>(
                    destination_rows + MulU64ByU32(rows_stride, destination));
                while (lo < hi) {
                    const uint32_t mid = lo + (hi - lo) / 2u;
                    FlushRange(reinterpret_cast<__gm__ uint8_t *>(layout + mid), sizeof(DestinationRow));
                    if (layout[mid].source_token < token) lo = mid + 1u;
                    else hi = mid;
                }
                if (lo == limit || layout[lo].source_token != token ||
                    layout[lo].source_rank != source) {
                    *error = kStatusInvalidState;
                    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(error));
                    DrainRelay(&buffers);
                    return;
                }
                row = lo;
            }
            destinations[count] = destination;
            offsets[count++] = MulU64ByU32(row_bytes, row);
        }
        }
        const uint64_t begin = MulU64ByU32(row_bytes, token);
        const uint64_t bytes = MulU64ByU32(row_bytes, coalesced ? header->token_count : 1u);
        if (aligned && bytes >= kPullDispatchAlignment) {
            RelayRun(source_base + header->hidden_offset + begin,
                destination_hidden_base, bytes, source,
                coalesced ? channel : 0u, coalesced ? channels : 1u,
                count, destinations, offsets, &buffers,
                coalesced ? 0u : kMaxHiddenTileBytes);
        } else if (count != 0u) {
            // Exact-width tail primitive. One AIV owns every destination on
            // unaligned launches, preventing concurrent cacheline writers.
            __gm__ uint8_t *staging = slot + header->hidden_offset + begin;
            for (uint64_t done = 0u; done < bytes;) {
                const uint32_t size = static_cast<uint32_t>(
                    bytes - done > 0x7fffffc0ull ? 0x7fffffc0ull : bytes - done);
                aclshmem_getmem(staging + done,
                    source_base + header->hidden_offset + begin + done, size,
                    static_cast<int32_t>(source));
                done += size;
            }
            aclshmem_quiet();
            FlushRange(staging, bytes);
            for (uint32_t i = 0u; i < count; ++i)
                PutGmRange(destination_hidden_base + offsets[i], staging, bytes,
                           static_cast<int32_t>(destinations[i]));
            aclshmem_quiet();
        }
        if (coalesced) break;
        if (header->token_count - token <= channels) break;
        token += channels;
    }
    DrainRelay(&buffers);
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void inc_dc_pull_dispatch_v2_device_kernel(
    GM_ADDR source_region, GM_ADDR ready_mailbox, GM_ADDR inc_slots,
    GM_ADDR source_acks, GM_ADDR destination_hidden,
    GM_ADDR destination_rows, GM_ADDR destination_assignments,
    GM_ADDR destination_expert_counts, GM_ADDR destination_completions,
    GM_ADDR inc_destination_rows, GM_ADDR inc_destination_assignments,
    GM_ADDR journal_header, GM_ADDR journal_tokens,
    GM_ADDR journal_contributors, GM_ADDR journal_assignments,
    GM_ADDR row_map, GM_ADDR source_token_prefix,
    GM_ADDR source_destination_prefix, GM_ADDR destination_row_counts,
    GM_ADDR destination_assignment_counts, GM_ADDR expert_counts,
    GM_ADDR parser_scratch, GM_ADDR status_line, uint64_t ffts_addr,
    uint64_t session_id,
    uint64_t placement_epoch, uint64_t generation, uint64_t sequence,
    uint64_t source_slot_stride, uint64_t destination_hidden_slot_stride,
    uint64_t destination_rows_slot_stride,
    uint64_t destination_assignments_slot_stride,
    uint64_t destination_expert_counts_slot_stride,
    uint64_t journal_token_capacity, uint64_t journal_contributor_capacity,
    uint64_t journal_assignment_capacity,
    uint64_t destination_row_capacity,
    uint64_t destination_assignment_capacity,
    uint64_t inc_destination_rows_stride_bytes,
    uint64_t inc_destination_assignments_stride_bytes,
    uint64_t row_map_capacity_entries,
    uint64_t source_destination_prefix_capacity_entries,
    uint64_t expert_counts_capacity_entries,
    uint64_t parser_scratch_capacity_entries, uint32_t worker_count,
    uint32_t expert_count, uint32_t hidden, uint32_t dtype, int32_t inc_pe,
    uint32_t region_id, uint32_t wave, uint32_t ring_slot,
    uint32_t slot_count, uint32_t channels_per_source, uint64_t spin_cap)
{
    shmemx_set_ffts_config(ffts_addr);
    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t blocks = AscendC::GetBlockNum();
    const int32_t pe = aclshmem_my_pe();
    __gm__ PullTimeline *timeline =
        reinterpret_cast<__gm__ PullTimeline *>(status_line);
    __gm__ uint32_t *status = reinterpret_cast<__gm__ uint32_t *>(
        status_line);

    const uint32_t dtype_bytes = DTypeBytes(dtype);
    uint64_t row_bytes = 0u;
    uint64_t relay_done_cycle = 0u;
    uint64_t required_channels = 0u;
    uint64_t required_expert_counts = 0u;
    uint64_t required_prefix_entries = 0u;
    uint64_t required_hidden_slot = 0u;
    uint64_t required_rows_slot = 0u;
    uint64_t required_assignments_slot = 0u;
    uint64_t required_expert_slot = 0u;
    const uint32_t parser_cohort = worker_count == 0u
        ? 0u : blocks / worker_count;
    const uint32_t active_parser_blocks = parser_cohort * worker_count;
    ParserScratchLayout scratch_layout{};
    const bool launch_valid =
        worker_count >= 2u && worker_count <= kPullDispatchMaxWorkers &&
        expert_count != 0u && hidden != 0u && dtype_bytes != 0u &&
        inc_pe == static_cast<int32_t>(worker_count) && region_id != 0u &&
        generation != 0u && sequence != 0u && ring_slot < slot_count &&
        source_slot_stride >= sizeof(SlotHeader) &&
        source_slot_stride % kPullDispatchAlignment == 0u &&
        channels_per_source != 0u && spin_cap != 0u &&
        parser_cohort != 0u &&
        active_parser_blocks > worker_count &&
        journal_token_capacity != 0u &&
        journal_token_capacity <= 0xffffffffull &&
        journal_contributor_capacity <= 0xffffffffull &&
        journal_assignment_capacity <= 0xffffffffull &&
        destination_row_capacity != 0u &&
        destination_row_capacity <= 0xffffffffull &&
        destination_assignment_capacity != 0u &&
        destination_assignment_capacity <= 0xffffffffull &&
        inc_destination_rows_stride_bytes % kPullDispatchAlignment == 0u &&
        inc_destination_assignments_stride_bytes %
                kPullDispatchAlignment == 0u &&
        CheckedMulU64ByU32(worker_count, channels_per_source,
                           &required_channels) &&
        required_channels <= active_parser_blocks - worker_count &&
        CheckedMulU64ByU32(hidden, dtype_bytes, &row_bytes) &&
        CheckedMulU64ByU32(worker_count, expert_count,
                           &required_expert_counts) &&
        required_expert_counts <= expert_counts_capacity_entries &&
        CheckedMulU64ByU32(static_cast<uint64_t>(worker_count) + 1u,
                           worker_count, &required_prefix_entries) &&
        required_prefix_entries <=
            source_destination_prefix_capacity_entries &&
        CheckedMulU64ByU32(row_bytes,
                           static_cast<uint32_t>(destination_row_capacity),
                           &required_hidden_slot) &&
        required_hidden_slot <= destination_hidden_slot_stride &&
        CheckedMulU64ByU32(sizeof(DestinationRow),
                           static_cast<uint32_t>(destination_row_capacity),
                           &required_rows_slot) &&
        required_rows_slot <= destination_rows_slot_stride &&
        required_rows_slot <= inc_destination_rows_stride_bytes &&
        CheckedMulU64ByU32(
            sizeof(ExpertAssignment),
            static_cast<uint32_t>(destination_assignment_capacity),
            &required_assignments_slot) &&
        required_assignments_slot <= destination_assignments_slot_stride &&
        required_assignments_slot <=
            inc_destination_assignments_stride_bytes &&
        CheckedMulU64ByU32(sizeof(uint32_t), expert_count,
                           &required_expert_slot) &&
        required_expert_slot <= destination_expert_counts_slot_stride &&
        BuildParserScratchLayout(active_parser_blocks, worker_count,
                                 expert_count,
                                 parser_scratch_capacity_entries,
                                 &scratch_layout);

    if (!launch_valid) {
        if (pe == inc_pe && block == 0u) {
            *status = kStatusInvalidArgument;
            dcci_cacheline(status_line);
        }
        return;
    }

    // A worker publishes exactly one descriptor and then leaves the kernel;
    // its immutable source slot remains leased until SourceConsumed arrives.
    if (pe != inc_pe) {
        if (block == 0u && pe >= 0 &&
            static_cast<uint32_t>(pe) < worker_count) {
            __gm__ Ready *local =
                reinterpret_cast<__gm__ Ready *>(ready_mailbox) + pe;
            __gm__ Ready *remote = local;
            aclshmem_uint64_p(&remote->publication, 0u, inc_pe);
            aclshmem_quiet();
            aclshmem_putmem(remote, local,
                            __builtin_offsetof(Ready, publication), inc_pe);
            aclshmem_quiet();
            aclshmem_uint64_p(&remote->publication, local->publication,
                              inc_pe);
            aclshmem_quiet();
        }
        return;
    }

    __gm__ uint8_t *source_base = source_region +
        MulU64ByU32(source_slot_stride, ring_slot);
    __gm__ uint8_t *destination_hidden_base = destination_hidden +
        MulU64ByU32(destination_hidden_slot_stride, ring_slot);
    __gm__ uint8_t *destination_rows_base = destination_rows +
        MulU64ByU32(destination_rows_slot_stride, ring_slot);
    __gm__ uint8_t *destination_assignments_base =
        destination_assignments + MulU64ByU32(
            destination_assignments_slot_stride, ring_slot);
    __gm__ uint8_t *destination_expert_counts_base =
        destination_expert_counts + MulU64ByU32(
            destination_expert_counts_slot_stride, ring_slot);

    __gm__ uint32_t *scratch =
        reinterpret_cast<__gm__ uint32_t *>(parser_scratch);
    __gm__ uint32_t *source_state = scratch + scratch_layout.source_state;
    __gm__ uint32_t *block_row_ends = scratch + scratch_layout.block_row_ends;
    __gm__ uint32_t *block_assignment_ends = scratch + scratch_layout.block_assignment_ends;
    __gm__ uint32_t *block_contributor_ends = scratch + scratch_layout.block_contributor_ends;
    __gm__ uint32_t *source_errors = scratch + scratch_layout.source_error;
    __gm__ uint32_t *source_assignment_prefix =
        scratch + scratch_layout.source_assignment_prefix;
    __gm__ uint32_t *block_status = scratch + scratch_layout.block_status;
    __gm__ uint32_t *block_contributors =
        scratch + scratch_layout.block_contributors;
    __gm__ uint32_t *block_rows = scratch + scratch_layout.block_rows;
    __gm__ uint32_t *block_assignments =
        scratch + scratch_layout.block_assignments;
    __gm__ uint32_t *block_experts =
        scratch + scratch_layout.block_experts;
    __gm__ uint8_t *metadata_tails = reinterpret_cast<__gm__ uint8_t *>(
        scratch + scratch_layout.metadata_tails);

    // Phase 0: reserve the journal and initialize only bounded coordination
    // state.  Each source cohort polls its own READY below, so a late rank
    // cannot head-of-line block parsing of an early rank.
    if (block == 0u) {
        *status = kStatusOk;
        timeline->ready_sources = 0u;
        timeline->kernel_start = AscendC::GetSystemCycle();
        timeline->all_ready = 0u;
        timeline->headers_pulled = 0u;
        timeline->metadata_parse_begin = timeline->kernel_start;
        timeline->metadata_parse_done = 0u;
        timeline->journal_reserved = 0u;
        timeline->hidden_get_begin = 0u;
        timeline->hidden_get_done = 0u;
        timeline->fanout_put_begin = 0u;
        timeline->fanout_put_done = 0u;
        timeline->reorg_done = 0u;
        timeline->destination_completions_done = 0u;
        timeline->source_acks_done = 0u;
        timeline->kernel_done = 0u;
        timeline->reserved[0] = 0u;
        for (uint32_t source = 0u; source < worker_count; ++source) {
            __gm__ uint32_t *state = source_state +
                static_cast<uint64_t>(source) *
                    scratch_layout.source_stride;
            state[0] = 0u;
            *reinterpret_cast<__gm__ uint64_t *>(state + 2u) = 0u;
            *reinterpret_cast<__gm__ uint64_t *>(state + 4u) = 0u;
            state[6] = 0u;
            state[7] = 0u;
            state[12] = 0u;
            for (uint32_t word = 8u; word < 12u; ++word) state[word] = 0u;
            source_errors[static_cast<uint64_t>(source) *
                          scratch_layout.source_stride] = 0u;
        }
        for (uint64_t index = 0u; index < required_expert_counts; ++index)
            reinterpret_cast<__gm__ uint32_t *>(expert_counts)[index] = 0u;
        for (uint32_t parser = 0u; parser < active_parser_blocks; ++parser) {
            __gm__ uint32_t *parser_state = block_status +
                static_cast<uint64_t>(parser) *
                    scratch_layout.block_stride;
            parser_state[0] = kStatusOk;
            parser_state[1] = 0u;
            parser_state[4] = 0u;
            *reinterpret_cast<__gm__ uint64_t *>(parser_state + 2u) = 0u;
        }

        __gm__ JournalSlotHeader *jheader =
            reinterpret_cast<__gm__ JournalSlotHeader *>(journal_header);
        if (jheader->magic != 0u &&
            (jheader->magic != kPullDispatchMagic ||
             jheader->abi_version != kPullDispatchAbiVersion ||
             jheader->struct_bytes != sizeof(JournalSlotHeader) ||
             (jheader->state != static_cast<uint16_t>(
                  JournalSlotState::FREE) &&
              jheader->state != static_cast<uint16_t>(
                  JournalSlotState::COMPLETE) &&
              jheader->state != static_cast<uint16_t>(
                  JournalSlotState::ABORTED)))) {
            *status = kStatusInvalidState;
        } else {
            jheader->magic = kPullDispatchMagic;
            jheader->abi_version = kPullDispatchAbiVersion;
            jheader->struct_bytes = sizeof(JournalSlotHeader);
            jheader->generation = generation;
            jheader->sequence = sequence;
            jheader->dispatch_cookie = 0u;
            jheader->wave = wave;
            jheader->ring_slot = static_cast<uint16_t>(ring_slot);
            jheader->state = static_cast<uint16_t>(
                JournalSlotState::DISPATCH_OPEN);
            jheader->token_count = 0u;
            jheader->contributor_count = 0u;
            jheader->status = 0u;
            jheader->flags = 0u;
            jheader->reserved[0] = 0u;
            timeline->journal_reserved = AscendC::GetSystemCycle();
            dcci_cacheline(journal_header);
        }
        FlushRange(parser_scratch,
                   scratch_layout.block_status * sizeof(uint32_t) +
                       static_cast<uint64_t>(active_parser_blocks) *
                           scratch_layout.block_stride *
                           sizeof(uint32_t));
        dcci_cacheline(status_line);
    }
    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);

    // Phase 1: source-local READY/GET and parallel validation/counting.  The
    // parser blocks are grouped source-major, with a contiguous token slice
    // per lane.  This order is what later makes a tiny deterministic prefix
    // sufficient for atomics-free final writes.
    if (*status == kStatusOk && block < active_parser_blocks) {
        const uint32_t source = block / parser_cohort;
        const uint32_t lane = block % parser_cohort;
        uint32_t local_status = kStatusOk;
        __gm__ uint8_t *boundary = reinterpret_cast<__gm__ uint8_t *>(scratch +
            scratch_layout.boundary_records + static_cast<uint64_t>(block) *
            scratch_layout.boundary_block_stride);
        for (uint32_t slot = 0u; slot < (2u + 2u * worker_count) * 8u; ++slot) {
            *reinterpret_cast<__gm__ uint64_t *>(boundary + slot * 128u) = 0u;
            dcci_cacheline(boundary + slot * 128u);
        }

        __gm__ uint32_t *my_rows = block_rows +
            static_cast<uint64_t>(block) * scratch_layout.row_stride;
        __gm__ uint32_t *my_assignments =
            block_assignments +
            static_cast<uint64_t>(block) * scratch_layout.row_stride;
        __gm__ uint32_t *my_experts = block_experts +
            static_cast<uint64_t>(block) * scratch_layout.expert_stride;
        __gm__ uint32_t *my_contributors = block_contributors +
            static_cast<uint64_t>(block) * scratch_layout.block_stride;
        *my_contributors = 0u;
        for (uint32_t destination = 0u; destination < worker_count;
             ++destination) {
            my_rows[destination] = 0u;
            my_assignments[destination] = 0u;
        }
        for (uint64_t index = 0u;
             index < static_cast<uint64_t>(worker_count) * expert_count;
             ++index)
            my_experts[index] = 0u;

        if (lane == 0u) {
            bool terminal = false;
            for (uint64_t spin = 0u; spin < spin_cap && !terminal; ++spin) {
                __gm__ Ready *ready =
                    reinterpret_cast<__gm__ Ready *>(ready_mailbox) + source;
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(ready));
                if (ready->publication == 0u) continue;
                if (ready->generation != generation ||
                    ready->sequence != sequence || ready->wave != wave ||
                    ready->ring_slot != ring_slot)
                    continue;
                if (!ReadyValid(ready, source, worker_count, session_id,
                                placement_epoch, generation, sequence, wave,
                                region_id, static_cast<uint16_t>(ring_slot),
                                slot_count)) {
                    local_status = kStatusInvalidReady;
                    terminal = true;
                    break;
                }
                *reinterpret_cast<__gm__ uint64_t *>(
                    source_state + static_cast<uint64_t>(source) *
                        scratch_layout.source_stride + 2u) =
                    AscendC::GetSystemCycle();
                __gm__ uint8_t *local_slot = inc_slots +
                    MulU64ByU32(source_slot_stride, source);
                aclshmem_getmem(local_slot, source_base, sizeof(SlotHeader),
                                static_cast<int32_t>(source));
                dcci_cacheline(local_slot);
                __gm__ SlotHeader *header =
                    reinterpret_cast<__gm__ SlotHeader *>(local_slot);
                local_status = ValidateHeader(
                    header, source, worker_count, expert_count, hidden, dtype,
                    session_id, placement_epoch, generation, sequence, wave,
                    region_id, static_cast<uint16_t>(ring_slot),
                    source_slot_stride);
                if (local_status != kStatusOk) {
                    terminal = true;
                    break;
                }
                terminal = true;
            }
            if (!terminal) local_status = kStatusReadyTimeout;
            source_errors[static_cast<uint64_t>(source) *
                          scratch_layout.source_stride] = local_status;
            source_state[static_cast<uint64_t>(source) *
                         scratch_layout.source_stride] =
                local_status == kStatusOk
                ? kSourceHeaderReady
                : (kOrdinalVisited | local_status);
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
                source_errors + static_cast<uint64_t>(source) *
                    scratch_layout.source_stride));
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
                source_state + static_cast<uint64_t>(source) *
                    scratch_layout.source_stride));
        } else {
            uint32_t observed = 0u;
            for (uint64_t spin = 0u; spin < spin_cap; ++spin) {
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
                    source_state + static_cast<uint64_t>(source) *
                        scratch_layout.source_stride));
                observed = source_state[static_cast<uint64_t>(source) *
                                        scratch_layout.source_stride];
                if (observed != 0u) break;
            }
            if (observed == 0u)
                local_status = kStatusReadyTimeout;
            else if ((observed & kOrdinalVisited) != 0u)
                local_status = observed & ~kOrdinalVisited;
        }

        // Header publication releases a source cohort immediately.  Each lane
        // pulls a disjoint cache-line range of [tokens_offset, hidden_offset),
        // so metadata bandwidth scales with parser_cohort and the header can
        // never be overwritten.  A second source publication below prevents
        // any parser from consuming metadata until every range is visible and
        // the padding invariant has been checked.
        __gm__ uint32_t *metadata_state = block_status +
            static_cast<uint64_t>(block) * scratch_layout.block_stride + 1u;
        if (local_status == kStatusOk) {
            __gm__ uint8_t *slot = inc_slots +
                MulU64ByU32(source_slot_stride, source);
            dcci_cacheline(slot);
            __gm__ SlotHeader *header =
                reinterpret_cast<__gm__ SlotHeader *>(slot);
            const uint64_t metadata_begin = header->tokens_offset;
            const uint64_t metadata_lines =
                (header->hidden_offset - metadata_begin) /
                    kPullDispatchAlignment;
            const uint64_t first_line =
                MulU64ByU32(metadata_lines, lane) / parser_cohort;
            const uint64_t last_line =
                MulU64ByU32(metadata_lines, lane + 1u) / parser_cohort;
            uint64_t offset = metadata_begin +
                first_line * kPullDispatchAlignment;
            uint64_t left =
                (last_line - first_line) * kPullDispatchAlignment;
            const uint64_t local_bytes = left;
            while (left != 0u) {
                const uint32_t chunk = static_cast<uint32_t>(
                    left > 0x7fffffc0ull ? 0x7fffffc0ull : left);
                aclshmem_getmem(slot + offset, source_base + offset, chunk,
                                static_cast<int32_t>(source));
                offset += chunk;
                left -= chunk;
            }
            if (local_bytes != 0u)
                FlushRange(slot + metadata_begin +
                               first_line * kPullDispatchAlignment,
                           local_bytes);
            *metadata_state = kParserMetadataReady;
        } else {
            *metadata_state = kOrdinalVisited | local_status;
        }
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(metadata_state));

        if (local_status == kStatusOk) {
            for (uint32_t peer_lane = 0u; peer_lane < parser_cohort;
                 ++peer_lane) {
                __gm__ uint32_t *peer_state = block_status +
                    static_cast<uint64_t>(source * parser_cohort + peer_lane) *
                        scratch_layout.block_stride + 1u;
                uint32_t observed = 0u;
                for (uint64_t spin = 0u; spin < spin_cap; ++spin) {
                    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
                        peer_state));
                    observed = *peer_state;
                    if (observed != 0u) break;
                }
                if (observed == 0u) {
                    local_status = kStatusReadyTimeout;
                    break;
                }
                if (observed != kParserMetadataReady) {
                    local_status = observed & ~kOrdinalVisited;
                    break;
                }
            }
        }

        __gm__ uint32_t *cohort_state = source_state +
            static_cast<uint64_t>(source) * scratch_layout.source_stride;
        if (lane == 0u && local_status == kStatusOk) {
            __gm__ uint8_t *slot = inc_slots +
                MulU64ByU32(source_slot_stride, source);
            __gm__ SlotHeader *header =
                reinterpret_cast<__gm__ SlotHeader *>(slot);
            if (!HiddenPaddingZero(header, slot))
                local_status = kStatusInvalidHeader;
            source_errors[static_cast<uint64_t>(source) *
                          scratch_layout.source_stride] = local_status;
            *cohort_state = local_status == kStatusOk
                ? kSourceMetadataReady
                : (kOrdinalVisited | local_status);
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
                source_errors + static_cast<uint64_t>(source) *
                    scratch_layout.source_stride));
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(cohort_state));
        } else if (lane != 0u && local_status == kStatusOk) {
            uint32_t observed = kSourceHeaderReady;
            for (uint64_t spin = 0u; spin < spin_cap; ++spin) {
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
                    cohort_state));
                observed = *cohort_state;
                if (observed != kSourceHeaderReady) break;
            }
            if (observed == kSourceHeaderReady)
                local_status = kStatusReadyTimeout;
            else if ((observed & kOrdinalVisited) != 0u)
                local_status = observed & ~kOrdinalVisited;
            else if (observed != kSourceMetadataReady)
                local_status = kStatusInvalidState;
        }

        if (local_status == kStatusOk) {
            __gm__ uint8_t *slot = inc_slots +
                MulU64ByU32(source_slot_stride, source);
            __gm__ SlotHeader *header =
                reinterpret_cast<__gm__ SlotHeader *>(slot);
            __gm__ TokenRecord *tokens =
                reinterpret_cast<__gm__ TokenRecord *>(
                    slot + header->tokens_offset);
            __gm__ AssignmentRecord *assignments =
                reinterpret_cast<__gm__ AssignmentRecord *>(
                    slot + header->assignments_offset);
            const uint32_t token_begin = ParserTokenBoundary(
                header->token_count, lane, parser_cohort);
            const uint32_t token_end = ParserTokenBoundary(
                header->token_count, lane + 1u, parser_cohort);

            // Verify the uniform hint in parallel. Lane zero publishes the
            // first token's exact destination bitmap; every parser lane then
            // compares only its own token slice. This retains independent INC
            // validation without a serial O(tokens*topk) prefix bottleneck.
            if ((header->flags & kSlotFlagUniformDestinations) != 0u) {
                if (lane == 0u) {
                    uint32_t reference[4]{0u, 0u, 0u, 0u};
                    if (header->token_count != 0u) {
                        __gm__ TokenRecord *first = tokens;
                        uint64_t first_end = 0u;
                        if (first->assignment_begin != 0u ||
                            !AddU64(first->assignment_begin,
                                    first->assignment_count, &first_end) ||
                            first_end > header->assignment_count) {
                            local_status = kStatusInvalidToken;
                        } else {
                            for (uint32_t local = 0u;
                                 local < first->assignment_count; ++local) {
                                const uint32_t destination = assignments[
                                    first->assignment_begin + local]
                                    .destination_rank;
                                if (destination >= worker_count) {
                                    local_status = kStatusInvalidAssignment;
                                    break;
                                }
                                reference[destination >> 5u] |=
                                    1u << (destination & 31u);
                            }
                        }
                    }
                    for (uint32_t word = 0u; word < 4u; ++word)
                        cohort_state[8u + word] = reference[word];
                    cohort_state[7] = local_status == kStatusOk
                        ? 1u : (kOrdinalVisited | local_status);
                    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
                        cohort_state));
                } else {
                    uint32_t observed = 0u;
                    for (uint64_t spin = 0u; spin < spin_cap; ++spin) {
                        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
                            cohort_state));
                        observed = cohort_state[7];
                        if (observed != 0u) break;
                    }
                    if (observed == 0u)
                        local_status = kStatusReadyTimeout;
                    else if ((observed & kOrdinalVisited) != 0u)
                        local_status = observed & ~kOrdinalVisited;
                }
            }

            if (lane == 0u && header->token_count == 0u &&
                header->assignment_count != 0u)
                local_status = kStatusInvalidToken;

            for (uint32_t token = token_begin;
                 token < token_end && local_status == kStatusOk; ++token) {
                __gm__ TokenRecord *record = tokens + token;
                uint64_t assignment_end = 0u;
                uint64_t expected_begin = 0u;
                if (token != 0u &&
                    !AddU64(tokens[token - 1u].assignment_begin,
                            tokens[token - 1u].assignment_count,
                            &expected_begin))
                    local_status = kStatusInvalidToken;
                if (!AddU64(record->assignment_begin,
                            record->assignment_count, &assignment_end) ||
                    local_status != kStatusOk ||
                    record->source_token != token ||
                    record->assignment_begin != expected_begin ||
                    assignment_end > header->assignment_count ||
                    (token + 1u == header->token_count &&
                     assignment_end != header->assignment_count) ||
                    record->reserved0 != 0u || record->reserved1 != 0u) {
                    local_status = kStatusInvalidToken;
                    break;
                }

                // First linear scan validates every field before the exact
                // in-place ordinal bitmap below touches the staging copy.
                for (uint32_t local = 0u;
                     local < record->assignment_count; ++local) {
                    __gm__ AssignmentRecord *assignment = assignments +
                        record->assignment_begin + local;
                    if (assignment->destination_rank >= worker_count ||
                        assignment->expert_id >= expert_count ||
                        !IsFinite(assignment->weight) ||
                        (assignment->ordinal & kOrdinalVisited) != 0u ||
                        assignment->ordinal >= record->assignment_count) {
                        local_status = kStatusInvalidAssignment;
                        break;
                    }
                }
                if (local_status != kStatusOk) break;
                uint32_t ordinal_bits[4]{0u, 0u, 0u, 0u};
                if (record->assignment_count <= 128u) {
                    for (uint32_t local = 0u;
                         local < record->assignment_count; ++local) {
                        const uint32_t ordinal = assignments[
                            record->assignment_begin + local].ordinal;
                        const uint32_t word = ordinal >> 5u;
                        const uint32_t mask = 1u << (ordinal & 31u);
                        if ((ordinal_bits[word] & mask) != 0u) {
                            local_status = kStatusInvalidAssignment;
                            break;
                        }
                        ordinal_bits[word] |= mask;
                    }
                } else {
                    // Very large top-k remains legal.  Keep a bounded-memory
                    // exact fallback instead of rejecting it or allocating a
                    // variable-sized UB bitmap.
                    for (uint32_t local = 0u;
                         local < record->assignment_count; ++local) {
                        const uint32_t ordinal = assignments[
                            record->assignment_begin + local].ordinal;
                        for (uint32_t previous = 0u; previous < local;
                             ++previous) {
                            if (assignments[record->assignment_begin +
                                            previous].ordinal == ordinal) {
                                local_status = kStatusInvalidAssignment;
                                break;
                            }
                        }
                        if (local_status != kStatusOk) break;
                    }
                }
                if (local_status != kStatusOk) break;

                uint32_t destination_bits[4]{0u, 0u, 0u, 0u};
                for (uint32_t local = 0u;
                     local < record->assignment_count; ++local) {
                    __gm__ AssignmentRecord *assignment = assignments +
                        record->assignment_begin + local;
                    const uint32_t destination = assignment->destination_rank;
                    const uint32_t word = destination >> 5u;
                    const uint32_t mask = 1u << (destination & 31u);
                    if ((destination_bits[word] & mask) == 0u) {
                        destination_bits[word] |= mask;
                        if (my_rows[destination] == 0xffffffffu ||
                            *my_contributors == 0xffffffffu) {
                            local_status = kStatusCapacityExceeded;
                            break;
                        }
                        ++my_rows[destination];
                        ++(*my_contributors);
                    }
                    if (my_assignments[destination] == 0xffffffffu) {
                        local_status = kStatusCapacityExceeded;
                        break;
                    }
                    ++my_assignments[destination];
                    const uint64_t expert_index =
                        static_cast<uint64_t>(destination) * expert_count +
                        assignment->expert_id;
                    if (my_experts[expert_index] == 0xffffffffu) {
                        local_status = kStatusCapacityExceeded;
                        break;
                    }
                    ++my_experts[expert_index];
                }
                if (local_status == kStatusOk &&
                    (header->flags & kSlotFlagUniformDestinations) != 0u) {
                    for (uint32_t word = 0u; word < 4u; ++word) {
                        if (destination_bits[word] !=
                            cohort_state[8u + word]) {
                            local_status = kStatusInvalidAssignment;
                            break;
                        }
                    }
                }
            }
        }
        block_status[static_cast<uint64_t>(block) *
                     scratch_layout.block_stride] = local_status;
        FlushRange(reinterpret_cast<__gm__ uint8_t *>(block_status +
                       static_cast<uint64_t>(block) *
                           scratch_layout.block_stride),
                   sizeof(uint32_t));
        FlushRange(reinterpret_cast<__gm__ uint8_t *>(
                       my_contributors), sizeof(uint32_t));
        FlushRange(reinterpret_cast<__gm__ uint8_t *>(my_rows),
                   static_cast<uint64_t>(worker_count) * sizeof(uint32_t));
        FlushRange(reinterpret_cast<__gm__ uint8_t *>(my_assignments),
                   static_cast<uint64_t>(worker_count) * sizeof(uint32_t));
        FlushRange(reinterpret_cast<__gm__ uint8_t *>(my_experts),
                   static_cast<uint64_t>(worker_count) * expert_count *
                       sizeof(uint32_t));
    }
    AscendC::SyncAll<true>();

    // Prefix: the only serial planning work is bounded by
    // active_AIVs*(workers + workers*experts), never token_count*topk.  Count
    // cells become per-block bases in place.
    if (block == 0u && *status == kStatusOk) {
        uint64_t total_tokens = 0u;
        uint64_t total_assignments = 0u;
        uint64_t row_map_entries = 0u;
        uint64_t cookie = kHashOffset;
        uint64_t last_ready_cycle = 0u;
        bool all_uniform_hints = true;
        reinterpret_cast<__gm__ uint32_t *>(source_token_prefix)[0] = 0u;
        source_assignment_prefix[0] = 0u;
        for (uint32_t source = 0u; source < worker_count; ++source) {
            const uint32_t source_error =
                source_errors[static_cast<uint64_t>(source) *
                              scratch_layout.source_stride];
            if (source_error != kStatusOk) {
                *status = source_error;
                break;
            }
            const uint64_t ready_cycle =
                *reinterpret_cast<__gm__ uint64_t *>(
                    source_state + static_cast<uint64_t>(source) *
                        scratch_layout.source_stride + 2u);
            if (ready_cycle > last_ready_cycle)
                last_ready_cycle = ready_cycle;
            __gm__ SlotHeader *header =
                reinterpret_cast<__gm__ SlotHeader *>(inc_slots +
                    MulU64ByU32(source_slot_stride, source));
            if ((header->flags & kSlotFlagUniformDestinations) == 0u)
                all_uniform_hints = false;
            if (!AddU64(total_tokens, header->token_count, &total_tokens) ||
                !AddU64(total_assignments, header->assignment_count,
                        &total_assignments) ||
                total_tokens > journal_token_capacity ||
                total_tokens > 0xffffffffull ||
                (journal_assignment_capacity != 0u &&
                 total_assignments > journal_assignment_capacity) ||
                total_assignments > 0xffffffffull) {
                *status = kStatusCapacityExceeded;
                break;
            }
            reinterpret_cast<__gm__ uint32_t *>(
                source_token_prefix)[source + 1u] =
                static_cast<uint32_t>(total_tokens);
            source_assignment_prefix[source + 1u] =
                static_cast<uint32_t>(total_assignments);
            cookie = HashGmBytes(cookie,
                reinterpret_cast<__gm__ const uint8_t *>(
                    &header->metadata_digest),
                sizeof(header->metadata_digest));
        }
        if (*status == kStatusOk && row_map_capacity_entries != 0u &&
            (!CheckedMulU64ByU32(total_tokens, worker_count,
                                &row_map_entries) ||
             row_map_entries > row_map_capacity_entries))
            *status = kStatusCapacityExceeded;
        for (uint32_t parser = 0u;
             parser < active_parser_blocks && *status == kStatusOk; ++parser)
            if (block_status[static_cast<uint64_t>(parser) *
                             scratch_layout.block_stride] != kStatusOk)
                *status = block_status[static_cast<uint64_t>(parser) *
                                       scratch_layout.block_stride];

        uint64_t contributor_cursor = 0u;
        uint32_t row_cursor[kPullDispatchMaxWorkers]{};
        uint32_t assignment_cursor[kPullDispatchMaxWorkers]{};
        uint32_t source_assignment_start[kPullDispatchMaxWorkers]{};
        if (*status == kStatusOk) {
            for (uint32_t source = 0u; source < worker_count; ++source) {
                for (uint32_t destination = 0u;
                     destination < worker_count; ++destination) {
                    __gm__ uint32_t *prefix_cell =
                        reinterpret_cast<__gm__ uint32_t *>(
                            source_destination_prefix) +
                        static_cast<uint64_t>(source) * worker_count +
                        destination;
                    *prefix_cell = row_cursor[destination];
                    source_assignment_start[destination] =
                        assignment_cursor[destination];
                }
                for (uint32_t lane = 0u; lane < parser_cohort; ++lane) {
                    const uint32_t parser = source * parser_cohort + lane;
                    __gm__ uint32_t *parser_contributors =
                        block_contributors +
                        static_cast<uint64_t>(parser) *
                            scratch_layout.block_stride;
                    const uint32_t contributor_count =
                        *parser_contributors;
                    *parser_contributors =
                        static_cast<uint32_t>(contributor_cursor);
                    contributor_cursor += contributor_count;
                    block_contributor_ends[static_cast<uint64_t>(parser) *
                        scratch_layout.block_stride] = static_cast<uint32_t>(contributor_cursor);
                    if (contributor_cursor > journal_contributor_capacity ||
                        contributor_cursor > 0xffffffffull) {
                        *status = kStatusCapacityExceeded;
                        break;
                    }
                    for (uint32_t destination = 0u;
                         destination < worker_count; ++destination) {
                        const uint64_t cell =
                            static_cast<uint64_t>(parser) *
                                scratch_layout.row_stride + destination;
                        const uint32_t rows = block_rows[cell];
                        const uint32_t assignments = block_assignments[cell];
                        block_rows[cell] = row_cursor[destination];
                        block_assignments[cell] =
                            assignment_cursor[destination];
                        if (rows > destination_row_capacity -
                                       row_cursor[destination] ||
                            assignments > destination_assignment_capacity -
                                              assignment_cursor[destination]) {
                            *status = kStatusCapacityExceeded;
                            break;
                        }
                        row_cursor[destination] += rows;
                        assignment_cursor[destination] += assignments;
                        block_row_ends[cell] = row_cursor[destination];
                        block_assignment_ends[cell] = assignment_cursor[destination];
                        for (uint32_t expert = 0u; expert < expert_count;
                             ++expert) {
                            const uint64_t expert_cell =
                                static_cast<uint64_t>(parser) *
                                    scratch_layout.expert_stride +
                                static_cast<uint64_t>(destination) *
                                    expert_count + expert;
                            const uint32_t count = block_experts[expert_cell];
                            const uint64_t final_index =
                                static_cast<uint64_t>(destination) *
                                    expert_count + expert;
                            const uint32_t base =
                                reinterpret_cast<__gm__ uint32_t *>(
                                    expert_counts)[final_index];
                            block_experts[expert_cell] = base;
                            if (count > 0xffffffffu - base) {
                                *status = kStatusCapacityExceeded;
                                break;
                            }
                            reinterpret_cast<__gm__ uint32_t *>(
                                expert_counts)[final_index] = base + count;
                        }
                        if (*status != kStatusOk) break;
                    }
                    if (*status != kStatusOk) break;
                }
                __gm__ SlotHeader *source_header =
                    reinterpret_cast<__gm__ SlotHeader *>(inc_slots +
                        MulU64ByU32(source_slot_stride, source));
                const bool source_hint =
                    (source_header->flags &
                     kSlotFlagUniformDestinations) != 0u;
                // Empty sources contribute the empty set and are vacuously
                // uniform; every destination delta must remain 0/0 below.
                bool source_uniform = true;
                for (uint32_t destination = 0u;
                     destination < worker_count; ++destination) {
                    const uint32_t row_begin =
                        reinterpret_cast<__gm__ uint32_t *>(
                            source_destination_prefix)[
                                static_cast<uint64_t>(source) * worker_count +
                                destination];
                    const uint32_t source_rows =
                        row_cursor[destination] - row_begin;
                    const uint32_t source_assignments =
                        assignment_cursor[destination] -
                            source_assignment_start[destination];
                    if ((source_rows == 0u && source_assignments == 0u) ||
                        (source_rows == source_header->token_count &&
                         source_assignments >= source_header->token_count))
                        continue;
                    source_uniform = false;
                }
                if (source_hint && !source_uniform)
                    *status = kStatusInvalidAssignment;
                if (!source_hint) all_uniform_hints = false;
                if (*status != kStatusOk) break;
            }
        }
        if (*status == kStatusOk) {
            for (uint32_t destination = 0u; destination < worker_count;
                 ++destination) {
                __gm__ uint32_t *final_prefix =
                    reinterpret_cast<__gm__ uint32_t *>(
                        source_destination_prefix) +
                    static_cast<uint64_t>(worker_count) * worker_count +
                    destination;
                *final_prefix = row_cursor[destination];
                reinterpret_cast<__gm__ uint32_t *>(
                    destination_row_counts)[destination] =
                    row_cursor[destination];
                reinterpret_cast<__gm__ uint32_t *>(
                    destination_assignment_counts)[destination] =
                    assignment_cursor[destination];
            }
            if (*status != kStatusOk) all_uniform_hints = false;
        }
        if (*status == kStatusOk) {
            __gm__ JournalSlotHeader *jheader =
                reinterpret_cast<__gm__ JournalSlotHeader *>(journal_header);
            jheader->token_count = static_cast<uint32_t>(total_tokens);
            jheader->contributor_count =
                static_cast<uint32_t>(contributor_cursor);
            jheader->dispatch_cookie = cookie == 0u ? 1u : cookie;
            jheader->flags = all_uniform_hints
                ? kJournalFlagUniformDestinations : 0u;
            timeline->all_ready = last_ready_cycle;
            timeline->ready_sources = worker_count;
            timeline->headers_pulled = AscendC::GetSystemCycle();
            // Empty parser chunks have no phase-2 writes.  Publish them from
            // the prefix coordinator so sideband progress never depends on an
            // otherwise work-free producer being scheduled.  A producer may
            // later repeat the same idempotent publication.
            for (uint32_t source = 0u; source < worker_count; ++source) {
                __gm__ SlotHeader *source_header =
                    reinterpret_cast<__gm__ SlotHeader *>(inc_slots +
                        MulU64ByU32(source_slot_stride, source));
                if (source_header->token_count != 0u) continue;
                for (uint32_t lane = 0u; lane < parser_cohort; ++lane) {
                    __gm__ uint32_t *parser_state = block_status +
                        static_cast<uint64_t>(source * parser_cohort + lane) *
                            scratch_layout.block_stride;
                    *parser_state = kParserPass2Ready;
                    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
                        parser_state));
                }
            }
        }
        FlushRange(source_token_prefix,
                   (static_cast<uint64_t>(worker_count) + 1u) *
                       sizeof(uint32_t));
        FlushRange(source_destination_prefix,
                   required_prefix_entries * sizeof(uint32_t));
        FlushRange(destination_row_counts,
                   static_cast<uint64_t>(worker_count) * sizeof(uint32_t));
        FlushRange(destination_assignment_counts,
                   static_cast<uint64_t>(worker_count) * sizeof(uint32_t));
        FlushRange(expert_counts,
                   required_expert_counts * sizeof(uint32_t));
        FlushRange(parser_scratch,
                   scratch_layout.boundary_records * sizeof(uint32_t));
        dcci_cacheline(status_line);
        dcci_cacheline(journal_header);
    }
    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);
    // Block zero selected the semantic journal flag in the prefix phase.
    // Every AIV must invalidate its private scalar-cache copy before deriving
    // producer/relay/metadata roles; otherwise different AIVs may disagree on
    // uniform_fast (relay can take fast while sideband skips both metadata
    // publication paths).
    dcci_cacheline(journal_header);

    // Pass one uses all parser AIVs. Pass two separates metadata production
    // from hidden relay, for both uniform and random routing:
    //   [0, W*channels)             hidden relay
    //   [W*channels, ...producer)    preferably two producers per source
    //   [active-W, active)          dedicated FNV/metadata sideband
    // If two/source does not fit, select the largest integral non-overlapping
    // cohort (at least one/source); otherwise retain the general schedule.
    __gm__ JournalSlotHeader *phase2_header =
        reinterpret_cast<__gm__ JournalSlotHeader *>(journal_header);
    const bool uniform_layout =
        (phase2_header->flags & kJournalFlagUniformDestinations) != 0u;
    const uint32_t uniform_sideband_begin =
        active_parser_blocks - worker_count;
    const uint32_t uniform_producer_begin =
        static_cast<uint32_t>(required_channels);
    const uint32_t uniform_available_producers =
        uniform_producer_begin <= uniform_sideband_begin
        ? uniform_sideband_begin - uniform_producer_begin : 0u;
    const uint32_t uniform_available_per_source = worker_count == 0u
        ? 0u : uniform_available_producers / worker_count;
    const uint32_t uniform_producers_per_source =
        uniform_available_per_source > 2u
        ? 2u : uniform_available_per_source;
    const uint32_t uniform_producer_count =
        uniform_producers_per_source * worker_count;
    const uint32_t uniform_producer_end =
        uniform_producer_begin + uniform_producer_count;
    const bool uniform_fast = uniform_layout &&
        uniform_producers_per_source != 0u;
    bool parallel_uniform_phase2 = uniform_producers_per_source != 0u;
    if (row_map_capacity_entries != 0u &&
        (8u * sizeof(uint32_t) * worker_count) % kPullDispatchAlignment != 0u)
        parallel_uniform_phase2 = false;
    if (parallel_uniform_phase2)
        for (uint32_t source = 0u; source < worker_count; ++source) {
            __gm__ SlotHeader *source_header =
                reinterpret_cast<__gm__ SlotHeader *>(inc_slots +
                    MulU64ByU32(source_slot_stride, source));
            if ((source_header->token_count & 7u) != 0u) {
                parallel_uniform_phase2 = false;
                break;
            }
        }
    // Eight-token boundaries align the token-indexed maps. Variable-length
    // output streams stage their shared boundary cachelines privately; a
    // single sideband writer repairs them before metadata publication.
    // Unaligned/ragged partitions retain serial pass two on the last producer.
    const bool phase2_worker = parallel_uniform_phase2
        ? block >= uniform_producer_begin && block < uniform_producer_end
        : block == uniform_sideband_begin - 1u;
    if (*status == kStatusOk && phase2_worker) {
      for (uint32_t work = 0u; work < active_parser_blocks; ++work) {
        // Round-robin sources so every relay gets its first descriptors early.
        const uint32_t parser_index = parallel_uniform_phase2 ? work :
            (work % worker_count) * parser_cohort + work / worker_count;
        if (parallel_uniform_phase2) {
            const uint32_t producer = block - uniform_producer_begin;
            const uint32_t producer_source =
                producer / uniform_producers_per_source;
            const uint32_t producer_lane =
                producer % uniform_producers_per_source;
            if (parser_index / parser_cohort != producer_source ||
                parser_index % parser_cohort %
                        uniform_producers_per_source != producer_lane)
                continue;
        }
        // Serialized block zero deliberately walks every parser chunk and
        // publishes every Pass2Ready cell; all other AIVs remain consumers.
        const uint32_t source = parser_index / parser_cohort;
        const uint32_t lane = parser_index % parser_cohort;
        __gm__ uint8_t *slot = inc_slots +
            MulU64ByU32(source_slot_stride, source);
        __gm__ SlotHeader *header =
            reinterpret_cast<__gm__ SlotHeader *>(slot);
        __gm__ TokenRecord *tokens =
            reinterpret_cast<__gm__ TokenRecord *>(slot +
                                                   header->tokens_offset);
        __gm__ AssignmentRecord *assignments =
            reinterpret_cast<__gm__ AssignmentRecord *>(slot +
                                                        header->assignments_offset);
        const uint32_t token_begin = ParserTokenBoundary(
            header->token_count, lane, parser_cohort);
        const uint32_t token_end = ParserTokenBoundary(
            header->token_count, lane + 1u, parser_cohort);
        uint32_t row_map_published = token_begin;
        uint32_t contributor_cursor = block_contributors[
            static_cast<uint64_t>(parser_index) *
                scratch_layout.block_stride];
        const uint32_t contributor_begin = contributor_cursor;
        const uint32_t contributor_end = block_contributor_ends[
            static_cast<uint64_t>(parser_index) * scratch_layout.block_stride];
        __gm__ uint8_t *boundary = reinterpret_cast<__gm__ uint8_t *>(scratch +
            scratch_layout.boundary_records + static_cast<uint64_t>(parser_index) *
            scratch_layout.boundary_block_stride);
        __gm__ uint32_t *my_rows = block_rows +
            static_cast<uint64_t>(parser_index) * scratch_layout.row_stride;
        __gm__ uint32_t *my_assignments =
            block_assignments +
            static_cast<uint64_t>(parser_index) * scratch_layout.row_stride;
        __gm__ uint32_t *my_experts = block_experts +
            static_cast<uint64_t>(parser_index) *
                scratch_layout.expert_stride;
        uint32_t row_begin[kPullDispatchMaxWorkers];
        uint32_t assignment_begin_by_destination[
            kPullDispatchMaxWorkers];
        for (uint32_t destination = 0u; destination < worker_count;
             ++destination) {
            row_begin[destination] = my_rows[destination];
            assignment_begin_by_destination[destination] =
                my_assignments[destination];
        }

        for (uint32_t token = token_begin; token < token_end; ++token) {
            __gm__ TokenRecord *record = tokens + token;
            const uint32_t global_token =
                reinterpret_cast<__gm__ uint32_t *>(
                    source_token_prefix)[source] + token;
            __gm__ JournalTokenEntry *jtoken =
                reinterpret_cast<__gm__ JournalTokenEntry *>(
                    journal_tokens) + global_token;
            jtoken->route_key = DeviceRouteKey(source, token);
            jtoken->token_id = record->token_id;
            jtoken->owner_rank = source;
            jtoken->owner_row = token;
            jtoken->contributors_begin = contributor_cursor;
            jtoken->contributors_count = 0u;
            jtoken->assignments_begin =
                source_assignment_prefix[source] + record->assignment_begin;
            jtoken->assignments_count = record->assignment_count;
            jtoken->accumulator_index = global_token;
            jtoken->flags = 0u;
            jtoken->reserved[0] = 0u;
            jtoken->reserved[1] = 0u;

            if (journal_assignment_capacity != 0u)
                for (uint32_t local = 0u;
                     local < record->assignment_count; ++local) {
                    __gm__ AssignmentRecord *input = assignments +
                        record->assignment_begin + local;
                    __gm__ AssignmentRecord *output =
                        reinterpret_cast<__gm__ AssignmentRecord *>(
                            journal_assignments) +
                        jtoken->assignments_begin + local;
                    if (parallel_uniform_phase2)
                        output = reinterpret_cast<__gm__ AssignmentRecord *>(BoundaryRecord(
                            reinterpret_cast<__gm__ uint8_t *>(output),
                            journal_assignments + static_cast<uint64_t>(source_assignment_prefix[source] +
                                tokens[token_begin].assignment_begin) * sizeof(AssignmentRecord),
                            journal_assignments + static_cast<uint64_t>(source_assignment_prefix[source] +
                                tokens[token_end - 1u].assignment_begin + tokens[token_end - 1u].assignment_count) * sizeof(AssignmentRecord),
                            sizeof(AssignmentRecord), boundary + 8u * 128u));
                    output->destination_rank = input->destination_rank;
                    output->expert_id = input->expert_id;
                    output->ordinal = input->ordinal;
                    output->weight = input->weight;
                }

            const bool emit_row_map = row_map_capacity_entries != 0u;
            __gm__ uint32_t *token_row_map =
                reinterpret_cast<__gm__ uint32_t *>(row_map) +
                static_cast<uint64_t>(global_token) * worker_count;
            uint32_t token_assignment_counts[kPullDispatchMaxWorkers];
            uint32_t token_destination_rows[kPullDispatchMaxWorkers];
            uint32_t destination_bits[4]{0u, 0u, 0u, 0u};
            if (emit_row_map)
                for (uint32_t destination = 0u;
                     destination < worker_count; ++destination)
                    token_row_map[destination] = kInvalidRow;
            for (uint32_t local = 0u; local < record->assignment_count;
                 ++local) {
                __gm__ AssignmentRecord *assignment = assignments +
                    record->assignment_begin + local;
                const uint32_t destination = assignment->destination_rank;
                const uint32_t word = destination >> 5u;
                const uint32_t mask = 1u << (destination & 31u);
                if ((destination_bits[word] & mask) == 0u) {
                    token_assignment_counts[destination] = 1u;
                    destination_bits[destination >> 5u] |=
                        1u << (destination & 31u);
                } else {
                    ++token_assignment_counts[destination];
                }
            }

            uint32_t contributor_count = 0u;
            {
                for (uint32_t destination = 0u; destination < worker_count; ++destination) {
                    const uint32_t word = destination >> 5u;
                    const uint32_t bit = destination & 31u;
                    if ((destination_bits[word] & (1u << bit)) == 0u)
                        continue;
                    if (destination >= worker_count) continue;
                    const uint32_t destination_assignment_count =
                        token_assignment_counts[destination];
                    const uint32_t destination_row = my_rows[destination]++;
                    const uint32_t destination_assignment_begin =
                        my_assignments[destination];
                    token_destination_rows[destination] = destination_row;
                    if (emit_row_map)
                        token_row_map[destination] = destination_row;

                    __gm__ DestinationRow *out_row =
                        reinterpret_cast<__gm__ DestinationRow *>(
                            inc_destination_rows +
                            MulU64ByU32(
                                inc_destination_rows_stride_bytes,
                                destination)) + destination_row;
                    if (parallel_uniform_phase2) {
                        __gm__ uint8_t *base = inc_destination_rows +
                            MulU64ByU32(inc_destination_rows_stride_bytes, destination);
                        out_row = reinterpret_cast<__gm__ DestinationRow *>(BoundaryRecord(
                            reinterpret_cast<__gm__ uint8_t *>(out_row),
                            base + static_cast<uint64_t>(row_begin[destination]) * sizeof(DestinationRow),
                            base + static_cast<uint64_t>(block_row_ends[
                                static_cast<uint64_t>(parser_index) * scratch_layout.row_stride + destination]) * sizeof(DestinationRow),
                            sizeof(DestinationRow), boundary + (2u + destination * 2u) * 8u * 128u));
                    }
                    out_row->route_key = DeviceRouteKey(source, token);
                    out_row->token_id = record->token_id;
                    out_row->source_rank = source;
                    out_row->source_token = token;
                    out_row->destination_row = destination_row;
                    out_row->assignments_begin =
                        destination_assignment_begin;
                    out_row->assignments_count =
                        destination_assignment_count;
                    out_row->reserved = 0u;

                    __gm__ JournalContributor *contributor =
                        reinterpret_cast<__gm__ JournalContributor *>(
                            journal_contributors) + contributor_cursor++;
                    if (parallel_uniform_phase2)
                        contributor = reinterpret_cast<__gm__ JournalContributor *>(BoundaryRecord(
                            reinterpret_cast<__gm__ uint8_t *>(contributor),
                            journal_contributors + static_cast<uint64_t>(contributor_begin) * sizeof(JournalContributor),
                            journal_contributors + static_cast<uint64_t>(contributor_end) * sizeof(JournalContributor),
                            sizeof(JournalContributor), boundary));
                    contributor->worker_rank = destination;
                    contributor->destination_row = destination_row;
                    contributor->assignment_begin =
                        destination_assignment_begin;
                    contributor->assignment_count =
                        destination_assignment_count;
                    ++contributor_count;
                }
            }

            // A single assignment scan now writes every destination's
            // already-reserved range, preserving the host oracle's filtered
            // source order without a destination x top-k nested scan.
            for (uint32_t local = 0u; local < record->assignment_count;
                 ++local) {
                __gm__ AssignmentRecord *assignment = assignments +
                    record->assignment_begin + local;
                const uint32_t destination = assignment->destination_rank;
                const uint64_t expert_index =
                    static_cast<uint64_t>(destination) * expert_count +
                    assignment->expert_id;
                __gm__ ExpertAssignment *out =
                    reinterpret_cast<__gm__ ExpertAssignment *>(
                        inc_destination_assignments +
                        MulU64ByU32(
                            inc_destination_assignments_stride_bytes,
                            destination)) + my_assignments[destination]++;
                if (parallel_uniform_phase2) {
                    __gm__ uint8_t *base = inc_destination_assignments +
                        MulU64ByU32(inc_destination_assignments_stride_bytes, destination);
                    out = reinterpret_cast<__gm__ ExpertAssignment *>(BoundaryRecord(
                        reinterpret_cast<__gm__ uint8_t *>(out),
                        base + static_cast<uint64_t>(assignment_begin_by_destination[destination]) * sizeof(ExpertAssignment),
                        base + static_cast<uint64_t>(block_assignment_ends[
                            static_cast<uint64_t>(parser_index) * scratch_layout.row_stride + destination]) * sizeof(ExpertAssignment),
                        sizeof(ExpertAssignment), boundary + (3u + destination * 2u) * 8u * 128u));
                }
                out->destination_row = token_destination_rows[destination];
                out->expert_id = assignment->expert_id;
                out->expert_row = my_experts[expert_index]++;
                out->ordinal = assignment->ordinal;
                out->weight = assignment->weight;
                out->reserved[0] = 0u;
                out->reserved[1] = 0u;
                out->reserved[2] = 0u;
            }
            jtoken->contributors_count = contributor_count;
            if (emit_row_map &&
                (header->flags & kSlotFlagUniformDestinations) == 0u &&
                (token + 1u == token_end || token + 1u - row_map_published >= 32u)) {
                const uint64_t first = static_cast<uint64_t>(
                    reinterpret_cast<__gm__ uint32_t *>(source_token_prefix)[source]) + row_map_published;
                FlushRange(row_map + first * worker_count * sizeof(uint32_t),
                    static_cast<uint64_t>(token + 1u - row_map_published) * worker_count * sizeof(uint32_t));
                AscendC::PipeBarrier<PIPE_ALL>();
                __gm__ uint32_t *progress = block_status +
                    static_cast<uint64_t>(parser_index) * scratch_layout.block_stride + 4u;
                *progress = token + 1u;
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(progress));
                row_map_published = token + 1u;
            }
        }
        if (token_begin < token_end) {
            const uint32_t assignment_begin =
                tokens[token_begin].assignment_begin;
            const uint32_t assignment_end =
                tokens[token_end - 1u].assignment_begin +
                tokens[token_end - 1u].assignment_count;
            const uint32_t global_token_begin =
                reinterpret_cast<__gm__ uint32_t *>(
                    source_token_prefix)[source] + token_begin;
            FlushRange(journal_tokens +
                    static_cast<uint64_t>(global_token_begin) *
                        sizeof(JournalTokenEntry),
                static_cast<uint64_t>(token_end - token_begin) *
                    sizeof(JournalTokenEntry));
            if (row_map_capacity_entries != 0u)
                FlushRange(row_map +
                        static_cast<uint64_t>(global_token_begin) *
                            worker_count * sizeof(uint32_t),
                    static_cast<uint64_t>(token_end - token_begin) *
                        worker_count * sizeof(uint32_t));
            if (journal_assignment_capacity != 0u)
                FlushRange(journal_assignments +
                        static_cast<uint64_t>(
                            source_assignment_prefix[source] +
                            assignment_begin) * sizeof(AssignmentRecord),
                    static_cast<uint64_t>(assignment_end - assignment_begin) *
                        sizeof(AssignmentRecord));
        }
        FlushRange(journal_contributors +
                static_cast<uint64_t>(contributor_begin) *
                    sizeof(JournalContributor),
            static_cast<uint64_t>(contributor_cursor - contributor_begin) *
                sizeof(JournalContributor));
        for (uint32_t destination = 0u; destination < worker_count;
             ++destination) {
            FlushRange(inc_destination_rows +
                    MulU64ByU32(inc_destination_rows_stride_bytes,
                                destination) +
                    static_cast<uint64_t>(row_begin[destination]) *
                        sizeof(DestinationRow),
                static_cast<uint64_t>(my_rows[destination] -
                                      row_begin[destination]) *
                    sizeof(DestinationRow));
            FlushRange(inc_destination_assignments +
                    MulU64ByU32(inc_destination_assignments_stride_bytes,
                                destination) +
                    static_cast<uint64_t>(
                        assignment_begin_by_destination[destination]) *
                        sizeof(ExpertAssignment),
                static_cast<uint64_t>(my_assignments[destination] -
                    assignment_begin_by_destination[destination]) *
                    sizeof(ExpertAssignment));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint32_t slot = 0u; slot < (2u + 2u * worker_count) * 8u; ++slot) {
            __gm__ uint8_t *entry = boundary + slot * 128u;
            if (*reinterpret_cast<__gm__ uint64_t *>(entry) != 0u)
                FlushRange(entry, 128u);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        block_status[static_cast<uint64_t>(parser_index) *
                     scratch_layout.block_stride] = kParserPass2Ready;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
            block_status + static_cast<uint64_t>(parser_index) *
                scratch_layout.block_stride));
      }
    }
    if (block == 0u && *status == kStatusOk) {
        timeline->metadata_parse_done = AscendC::GetSystemCycle();
        timeline->hidden_get_begin = timeline->metadata_parse_done;
        timeline->fanout_put_begin = timeline->metadata_parse_done;
        dcci_cacheline(status_line);
    }
    dcci_cacheline(status_line);

    // Exact source FNV and destination metadata publication use the final
    // worker_count AIVs.  There is deliberately no barrier between this
    // sideband work and relay: hidden GET/fanout starts immediately on the
    // transport AIVs.  A bad FNV is collected only after the relay join, so
    // speculative bytes can never acquire a successful completion.
    if (*status == kStatusOk &&
        block >= uniform_sideband_begin &&
        block < active_parser_blocks) {
        const uint32_t side =
            block - uniform_sideband_begin;
        bool plans_ready = true;
        for (uint32_t parser = 0u; parser < active_parser_blocks; ++parser) {
            if (!WaitParserReady(block_status, scratch_layout.block_stride,
                                 parser, spin_cap)) {
                plans_ready = false;
                break;
            }
        }
        if (!plans_ready) {
            source_errors[static_cast<uint64_t>(side) *
                          scratch_layout.source_stride] =
                kStatusReadyTimeout;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
                source_errors + static_cast<uint64_t>(side) *
                    scratch_layout.source_stride));
        }
        if (plans_ready) {
            __gm__ uint32_t *repaired = source_state + 12u;
            if (side == 0u) {
                for (uint32_t parser = 0u; parser < active_parser_blocks; ++parser) {
                    __gm__ uint8_t *records = reinterpret_cast<__gm__ uint8_t *>(scratch +
                        scratch_layout.boundary_records + static_cast<uint64_t>(parser) *
                        scratch_layout.boundary_block_stride);
                    for (uint32_t index = 0u; index < (2u + worker_count * 2u) * 8u; ++index) {
                        __gm__ uint8_t *entry = records + index * 128u;
                        dcci_cacheline(entry);
                        const uint64_t target = *reinterpret_cast<__gm__ uint64_t *>(entry);
                        if (target == 0u) continue;
                        const uint32_t bytes = *reinterpret_cast<__gm__ uint32_t *>(entry + 8u);
                        __gm__ uint32_t *destination = reinterpret_cast<__gm__ uint32_t *>(target);
                        __gm__ uint32_t *values = reinterpret_cast<__gm__ uint32_t *>(entry + 64u);
                        dcci_cacheline(entry + 64u);
                        for (uint32_t word = 0u; word < bytes / 4u; ++word)
                            destination[word] = values[word];
                        FlushRange(reinterpret_cast<__gm__ uint8_t *>(destination), bytes);
                    }
                }
                *repaired = 1u;
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(repaired));
            } else {
                uint32_t observed = 0u;
                for (uint64_t spin = 0u; spin < spin_cap; ++spin) {
                    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(repaired));
                    observed = *repaired;
                    if (observed == 1u) break;
                }
                if (observed != 1u) plans_ready = false;
            }
        }
        if (plans_ready) {
        __gm__ uint8_t *slot = inc_slots +
            MulU64ByU32(source_slot_stride, side);
        __gm__ SlotHeader *header =
            reinterpret_cast<__gm__ SlotHeader *>(slot);
        if (MetadataDigest(header, slot) != header->metadata_digest)
            source_errors[static_cast<uint64_t>(side) *
                          scratch_layout.source_stride] =
                kStatusDigestMismatch;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
            source_errors + static_cast<uint64_t>(side) *
                scratch_layout.source_stride));

        // On the qualified uniform path all layout producers have already
        // published their disjoint ranges.  Send destination metadata now,
        // while relay AIVs are still moving hidden bytes.  Completion remains
        // the sole remote visibility point, so even a later FNV failure cannot
        // expose these speculative writes as successful output.
        {
            dcci_cacheline(destination_row_counts +
                           static_cast<uint64_t>(side) * sizeof(uint32_t));
            dcci_cacheline(destination_assignment_counts +
                           static_cast<uint64_t>(side) * sizeof(uint32_t));
            const uint32_t rows =
                reinterpret_cast<__gm__ uint32_t *>(
                    destination_row_counts)[side];
            const uint32_t assignment_count =
                reinterpret_cast<__gm__ uint32_t *>(
                    destination_assignment_counts)[side];
            __gm__ uint8_t *row_source = inc_destination_rows +
                MulU64ByU32(inc_destination_rows_stride_bytes, side);
            __gm__ uint8_t *assignment_source =
                inc_destination_assignments +
                MulU64ByU32(inc_destination_assignments_stride_bytes, side);
            __gm__ uint8_t *expert_source = expert_counts +
                static_cast<uint64_t>(side) * expert_count *
                    sizeof(uint32_t);
            __gm__ uint8_t *tail = metadata_tails +
                static_cast<uint64_t>(side) * kPullDispatchAlignment;
            FlushRange(row_source,
                       static_cast<uint64_t>(rows) *
                           sizeof(DestinationRow));
            FlushRange(assignment_source,
                       static_cast<uint64_t>(assignment_count) *
                           sizeof(ExpertAssignment));
            FlushRange(expert_source,
                       static_cast<uint64_t>(expert_count) *
                           sizeof(uint32_t));
            if (rows != 0u)
                PutGmRange(destination_rows_base, row_source,
                    (static_cast<uint64_t>(rows) * sizeof(DestinationRow) +
                     kPullDispatchAlignment - 1u) &
                        ~(static_cast<uint64_t>(kPullDispatchAlignment) - 1u),
                    static_cast<int32_t>(side));
            if (assignment_count != 0u)
                PutGmRange(destination_assignments_base, assignment_source,
                    (static_cast<uint64_t>(assignment_count) *
                         sizeof(ExpertAssignment) +
                     kPullDispatchAlignment - 1u) &
                        ~(static_cast<uint64_t>(kPullDispatchAlignment) - 1u),
                    static_cast<int32_t>(side));
            PutGmRangeAligned(destination_expert_counts_base, expert_source,
                static_cast<uint64_t>(expert_count) * sizeof(uint32_t), tail,
                static_cast<int32_t>(side));
            aclshmem_quiet();
            __gm__ uint32_t *published = source_state +
                static_cast<uint64_t>(side) *
                    scratch_layout.source_stride + 6u;
            *published = kDestinationMetadataPublished;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(published));
        }

        }
        __gm__ uint32_t *side_state = source_state +
            static_cast<uint64_t>(side) * scratch_layout.source_stride;
        *reinterpret_cast<__gm__ uint64_t *>(side_state + 4u) =
            AscendC::GetSystemCycle();
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(side_state));
    }

    // All routes use source runs. Nonuniform runs need completed row
    // descriptors; uniform runs use the already validated prefix immediately.
    const bool byte_aligned = row_bytes % kPullDispatchAlignment == 0u;
    if (*status == kStatusOk &&
        (byte_aligned ? block < required_channels : block == 0u)) {
        bool rows_ready = true;
        if (!uniform_fast && row_map_capacity_entries == 0u)
            for (uint32_t parser = 0u; parser < active_parser_blocks; ++parser)
                if (!WaitParserReady(block_status, scratch_layout.block_stride,
                                     parser, spin_cap)) {
                    rows_ready = false;
                    break;
                }
        if (rows_ready && !uniform_fast && row_map_capacity_entries == 0u) {
            uint32_t observed = 0u;
            for (uint64_t spin = 0u; spin < spin_cap; ++spin) {
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(source_state + 12u));
                observed = source_state[12];
                if (observed == 1u) break;
            }
            rows_ready = observed == 1u;
        }
        if (!rows_ready) {
            *status = kStatusReadyTimeout;
            dcci_cacheline(status_line);
        } else {
            const uint32_t first_source = byte_aligned ? block % worker_count : 0u;
            const uint32_t last_source = byte_aligned ? first_source + 1u : worker_count;
            for (uint32_t source = first_source; source < last_source; ++source) {
                __gm__ SlotHeader *header = reinterpret_cast<__gm__ SlotHeader *>(
                    inc_slots + MulU64ByU32(source_slot_stride, source));
                RelaySourceRuns(source_base, destination_hidden_base,
                    reinterpret_cast<__gm__ uint32_t *>(source_destination_prefix),
                    header, inc_destination_rows, inc_destination_rows_stride_bytes,
                    row_bytes, source, worker_count,
                    byte_aligned ? block / worker_count : 0u,
                    byte_aligned ? channels_per_source : 1u, status,
                    row_map_capacity_entries == 0u ? nullptr : reinterpret_cast<__gm__ uint32_t *>(row_map),
                    reinterpret_cast<__gm__ uint32_t *>(source_token_prefix),
                    block_status, scratch_layout.block_stride, parser_cohort, spin_cap);
            }
        }
        relay_done_cycle = AscendC::GetSystemCycle();
    }

    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);
    // Parser readiness and relay timing occupy the same cacheline. Never
    // publish the timestamp while a different AIV may publish Pass2Ready:
    // a scalar cacheline writeback can otherwise erase that progress flag.
    if (relay_done_cycle != 0u) {
        __gm__ uint32_t *relay_state = block_status +
            static_cast<uint64_t>(block) * scratch_layout.block_stride;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(relay_state));
        *reinterpret_cast<__gm__ uint64_t *>(relay_state + 2u) = relay_done_cycle;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(relay_state));
    }
    AscendC::SyncAll<true>();

    if (block == 0u && *status == kStatusOk) {
        uint64_t relay_done = 0u;
        for (uint32_t relay = 0u; relay < required_channels; ++relay) {
            __gm__ uint32_t *relay_state = block_status +
                static_cast<uint64_t>(relay) *
                    scratch_layout.block_stride;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(relay_state));
            const uint64_t cycle =
                *reinterpret_cast<__gm__ uint64_t *>(relay_state + 2u);
            if (cycle > relay_done) relay_done = cycle;
        }
        uint64_t sideband_done = 0u;
        for (uint32_t source = 0u; source < worker_count; ++source) {
            __gm__ uint32_t *state = source_state +
                static_cast<uint64_t>(source) *
                    scratch_layout.source_stride;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(state));
            const uint64_t side_cycle =
                *reinterpret_cast<__gm__ uint64_t *>(state + 4u);
            if (side_cycle > sideband_done) sideband_done = side_cycle;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
                source_errors + static_cast<uint64_t>(source) *
                    scratch_layout.source_stride));
            const uint32_t source_error =
                source_errors[static_cast<uint64_t>(source) *
                              scratch_layout.source_stride];
            if (source_error != kStatusOk) {
                *status = source_error;
                break;
            }
        }
        timeline->metadata_parse_done = sideband_done;
        timeline->hidden_get_done = relay_done;
        timeline->fanout_put_done = timeline->hidden_get_done;
        dcci_cacheline(status_line);
    }
    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);

    AscendC::SyncAll<true>();
    if (block == 0u) {
        if (*status == kStatusOk) {
            for (uint32_t destination = 0u;
                 destination < worker_count; ++destination) {
                __gm__ uint32_t *published = source_state +
                    static_cast<uint64_t>(destination) *
                        scratch_layout.source_stride + 6u;
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(published));
                if (*published != kDestinationMetadataPublished) {
                    *status = kStatusInvalidState;
                    break;
                }
            }
        }
        timeline->reorg_done = AscendC::GetSystemCycle();
        dcci_cacheline(status_line);
    }
    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);

    if (block == 0u) {
        __gm__ JournalSlotHeader *jheader =
            reinterpret_cast<__gm__ JournalSlotHeader *>(journal_header);
        if (*status == kStatusOk) {
            jheader->status = kStatusOk;
            jheader->state = static_cast<uint16_t>(
                JournalSlotState::DISPATCH_SEALED);
        } else if (jheader->magic == kPullDispatchMagic &&
                   jheader->generation == generation &&
                   jheader->sequence == sequence) {
            jheader->status = *status;
            jheader->state = static_cast<uint16_t>(
                JournalSlotState::ABORTED);
        }
        dcci_cacheline(journal_header);
        if (timeline->hidden_get_done == 0u) {
            timeline->hidden_get_done = AscendC::GetSystemCycle();
            timeline->fanout_put_done = timeline->hidden_get_done;
            timeline->reorg_done = timeline->hidden_get_done;
        }

        // Completion is the destination-side publication barrier.  No worker
        // may consume rows, assignments or hidden before this final word.
        for (uint32_t destination = 0u;
             destination < worker_count; ++destination) {
            __gm__ DestinationCompletion *completion =
                reinterpret_cast<__gm__ DestinationCompletion *>(
                    destination_completions) +
                static_cast<uint64_t>(ring_slot) * worker_count +
                destination;
            const uint32_t rows = *status == kStatusOk
                ? reinterpret_cast<__gm__ uint32_t *>(
                      destination_row_counts)[destination]
                : 0u;
            const uint32_t assignments = *status == kStatusOk
                ? reinterpret_cast<__gm__ uint32_t *>(
                      destination_assignment_counts)[destination]
                : 0u;
            PublishDestinationCompletion(
                completion, destination, *status, rows, assignments,
                session_id, placement_epoch, generation, sequence,
                *status == kStatusOk ? jheader->dispatch_cookie : 0u,
                wave, static_cast<uint16_t>(ring_slot));
        }
        timeline->destination_completions_done =
            AscendC::GetSystemCycle();

        // SourceConsumed is deliberately last: it is the source slot lease
        // release and therefore cannot precede any hidden GET.
        for (uint32_t source = 0u; source < worker_count; ++source) {
            __gm__ SourceConsumed *ack =
                reinterpret_cast<__gm__ SourceConsumed *>(source_acks) +
                source;
            uint64_t consumed = 0u;
            if (*status == kStatusOk) {
                __gm__ SlotHeader *header =
                    reinterpret_cast<__gm__ SlotHeader *>(
                        inc_slots + MulU64ByU32(source_slot_stride, source));
                consumed = header->packet_bytes;
            }
            PublishSourceAck(ack, source, *status, consumed, session_id,
                             generation, sequence, placement_epoch,
                             *status == kStatusOk
                                 ? jheader->dispatch_cookie
                                 : 0u,
                             region_id, wave,
                             static_cast<uint16_t>(ring_slot));
        }
        timeline->source_acks_done = AscendC::GetSystemCycle();
        timeline->kernel_done = timeline->source_acks_done;
        FlushRange(status_line, sizeof(PullTimeline));
    }
}

extern "C" void launch_inc_dc_pull_dispatch_v2_device(
    uint32_t block_dim, void *stream, uint8_t *source_region,
    uint8_t *ready_mailbox, uint8_t *inc_slots, uint8_t *source_acks,
    uint8_t *destination_hidden, uint8_t *destination_rows,
    uint8_t *destination_assignments, uint8_t *destination_expert_counts,
    uint8_t *destination_completions, uint8_t *inc_destination_rows,
    uint8_t *inc_destination_assignments, uint8_t *journal_header,
    uint8_t *journal_tokens, uint8_t *journal_contributors,
    uint8_t *journal_assignments, uint8_t *row_map,
    uint8_t *source_token_prefix, uint8_t *source_destination_prefix,
    uint8_t *destination_row_counts,
    uint8_t *destination_assignment_counts, uint8_t *expert_counts,
    uint8_t *parser_scratch, uint8_t *status_line, uint64_t ffts_addr,
    uint64_t session_id,
    uint64_t placement_epoch, uint64_t generation, uint64_t sequence,
    uint64_t source_slot_stride, uint64_t destination_hidden_slot_stride,
    uint64_t destination_rows_slot_stride,
    uint64_t destination_assignments_slot_stride,
    uint64_t destination_expert_counts_slot_stride,
    uint64_t journal_token_capacity, uint64_t journal_contributor_capacity,
    uint64_t journal_assignment_capacity,
    uint64_t destination_row_capacity,
    uint64_t destination_assignment_capacity,
    uint64_t inc_destination_rows_stride_bytes,
    uint64_t inc_destination_assignments_stride_bytes,
    uint64_t row_map_capacity_entries,
    uint64_t source_destination_prefix_capacity_entries,
    uint64_t expert_counts_capacity_entries,
    uint64_t parser_scratch_capacity_entries, uint32_t worker_count,
    uint32_t expert_count, uint32_t hidden, uint32_t dtype, int32_t inc_pe,
    uint32_t region_id, uint32_t wave, uint32_t ring_slot,
    uint32_t slot_count, uint32_t channels_per_source, uint64_t spin_cap)
{
    uint32_t selected_channels = channels_per_source;
    if (selected_channels == 0u) {
        const uint32_t portable_max = worker_count == 0u
            ? 0u
            : (block_dim > worker_count
                ? (block_dim - worker_count) / worker_count : 0u);
        selected_channels = portable_max < kDefaultChannelsPerSource
            ? portable_max
            : kDefaultChannelsPerSource;
    }
    inc_dc_pull_dispatch_v2_device_kernel<<<block_dim, nullptr, stream>>>(
        source_region, ready_mailbox, inc_slots, source_acks,
        destination_hidden, destination_rows, destination_assignments,
        destination_expert_counts, destination_completions,
        inc_destination_rows, inc_destination_assignments, journal_header,
        journal_tokens, journal_contributors, journal_assignments, row_map,
        source_token_prefix, source_destination_prefix,
        destination_row_counts, destination_assignment_counts,
        expert_counts, parser_scratch, status_line, ffts_addr, session_id,
        placement_epoch, generation, sequence, source_slot_stride,
        destination_hidden_slot_stride, destination_rows_slot_stride,
        destination_assignments_slot_stride,
        destination_expert_counts_slot_stride, journal_token_capacity,
        journal_contributor_capacity, journal_assignment_capacity,
        destination_row_capacity, destination_assignment_capacity,
        inc_destination_rows_stride_bytes,
        inc_destination_assignments_stride_bytes,
        row_map_capacity_entries,
        source_destination_prefix_capacity_entries,
        expert_counts_capacity_entries, parser_scratch_capacity_entries,
        worker_count, expert_count, hidden, dtype, inc_pe, region_id, wave,
        ring_slot, slot_count, selected_channels, spin_cap);
}
