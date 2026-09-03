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
//     header and [tokens_offset, hidden_offset) metadata are populated; hidden
//     rows are never staged in INC HBM.
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
//     INC-only [worker_count][destination_row_capacity].
//   inc_destination_assignments
//     INC-only [worker_count][destination_assignment_capacity].
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
constexpr uint32_t kHiddenTileBytes = 8u * 1024u;
constexpr uint32_t kUbSlotBytes = 12u * 1024u;
constexpr uint32_t kInvalidRow = ~0u;

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

constexpr uint64_t kHashOffset = 1469598103934665603ull;

static_assert(kUbSlotBytes * 2u <= 24u * 1024u,
              "Pull Dispatch ping/pong exceeds ordinary AIV UB");
static_assert(kHiddenTileBytes <= kUbSlotBytes,
              "Pull Dispatch transfer exceeds one UB slot");

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
    for (uint64_t offset = 0u; offset < bytes;
         offset += kPullDispatchAlignment)
        dcci_cacheline(base + offset);
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
        header->ring_slot != ring_slot || header->flags != 0u ||
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
    else
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(1u);
}

__aicore__ inline void WaitGetReady(uint32_t slot)
{
    if (slot == 0u)
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0u);
    else
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(1u);
}

__aicore__ inline void SetPutDone(uint32_t slot)
{
    if (slot == 0u)
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
    else
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
}

__aicore__ inline void WaitPutDone(uint32_t slot)
{
    if (slot == 0u)
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
    else
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
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
    GM_ADDR status_line, uint64_t ffts_addr, uint64_t session_id,
    uint64_t placement_epoch, uint64_t generation, uint64_t sequence,
    uint64_t source_slot_stride, uint64_t destination_hidden_slot_stride,
    uint64_t destination_rows_slot_stride,
    uint64_t destination_assignments_slot_stride,
    uint64_t destination_expert_counts_slot_stride,
    uint64_t journal_token_capacity, uint64_t journal_contributor_capacity,
    uint64_t journal_assignment_capacity,
    uint64_t destination_row_capacity,
    uint64_t destination_assignment_capacity,
    uint64_t row_map_capacity_entries,
    uint64_t source_destination_prefix_capacity_entries,
    uint64_t expert_counts_capacity_entries, uint32_t worker_count,
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
    uint64_t required_channels = 0u;
    uint64_t required_expert_counts = 0u;
    uint64_t required_prefix_entries = 0u;
    uint64_t required_hidden_slot = 0u;
    uint64_t required_rows_slot = 0u;
    uint64_t required_assignments_slot = 0u;
    uint64_t required_expert_slot = 0u;
    const bool launch_valid =
        worker_count >= 2u && worker_count <= kPullDispatchMaxWorkers &&
        expert_count != 0u && hidden != 0u && dtype_bytes != 0u &&
        inc_pe == static_cast<int32_t>(worker_count) && region_id != 0u &&
        generation != 0u && sequence != 0u && ring_slot < slot_count &&
        source_slot_stride >= sizeof(SlotHeader) &&
        source_slot_stride % kPullDispatchAlignment == 0u &&
        channels_per_source != 0u && spin_cap != 0u &&
        journal_token_capacity != 0u &&
        journal_token_capacity <= 0xffffffffull &&
        journal_contributor_capacity <= 0xffffffffull &&
        journal_assignment_capacity <= 0xffffffffull &&
        destination_row_capacity != 0u &&
        destination_row_capacity <= 0xffffffffull &&
        destination_assignment_capacity != 0u &&
        destination_assignment_capacity <= 0xffffffffull &&
        CheckedMulU64ByU32(worker_count, channels_per_source,
                           &required_channels) &&
        required_channels <= blocks &&
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
        CheckedMulU64ByU32(
            sizeof(ExpertAssignment),
            static_cast<uint32_t>(destination_assignment_capacity),
            &required_assignments_slot) &&
        required_assignments_slot <= destination_assignments_slot_stride &&
        CheckedMulU64ByU32(sizeof(uint32_t), expert_count,
                           &required_expert_slot) &&
        required_expert_slot <= destination_expert_counts_slot_stride;

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

    if (block == 0u) {
        *status = kStatusOk;
        timeline->ready_sources = 0u;
        timeline->kernel_start = AscendC::GetSystemCycle();
        timeline->all_ready = 0u;
        timeline->headers_pulled = 0u;
        timeline->metadata_parse_begin = 0u;
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
        dcci_cacheline(status_line);

        __gm__ uint32_t *ready_state =
            reinterpret_cast<__gm__ uint32_t *>(source_token_prefix);
        for (uint32_t source = 0u; source < worker_count; ++source)
            ready_state[source] = 0u;
        uint32_t remaining = worker_count;
        // Arrival order is deliberately unconstrained.  Once a READY is
        // observed, block zero immediately pulls and validates its fixed
        // header and then pulls only metadata, never hidden payload.
        for (uint64_t spin = 0u;
             spin < spin_cap && remaining != 0u && *status == kStatusOk;
            ++spin) {
            for (uint32_t source = 0u; source < worker_count; ++source) {
                if (ready_state[source] != 0u) continue;
                __gm__ Ready *ready =
                    reinterpret_cast<__gm__ Ready *>(ready_mailbox) + source;
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(ready));
                if (ready->publication == 0u) continue;
                // A terminal mailbox entry from a previous wave is normal:
                // ignore it until this wave's descriptor replaces it.  Once
                // the strong wave identity matches, any remaining mismatch
                // is corruption and must fail closed rather than time out.
                if (ready->generation != generation ||
                    ready->sequence != sequence || ready->wave != wave ||
                    ready->ring_slot != ring_slot)
                    continue;
                if (!ReadyValid(ready, source, worker_count, session_id,
                                placement_epoch, generation, sequence, wave,
                                region_id, static_cast<uint16_t>(ring_slot),
                                slot_count)) {
                    *status = kStatusInvalidReady;
                    break;
                }
                __gm__ uint8_t *local_slot = inc_slots +
                    MulU64ByU32(source_slot_stride, source);
                aclshmem_getmem(local_slot, source_base,
                                sizeof(SlotHeader),
                                static_cast<int32_t>(source));
                dcci_cacheline(local_slot);
                __gm__ SlotHeader *header =
                    reinterpret_cast<__gm__ SlotHeader *>(local_slot);
                *status = ValidateHeader(
                    header, source, worker_count, expert_count, hidden, dtype,
                    session_id, placement_epoch, generation, sequence, wave,
                    region_id, static_cast<uint16_t>(ring_slot),
                    source_slot_stride);
                if (*status != kStatusOk) break;
                const uint64_t metadata_bytes =
                    header->hidden_offset - sizeof(SlotHeader);
                if (metadata_bytes != 0u) {
                    // Header validation proved the byte count fits both the
                    // source slot and uint32_t RMA chunks.  Pull in bounded
                    // calls so metadata can exceed one transport call.
                    uint64_t offset = sizeof(SlotHeader);
                    uint64_t left = metadata_bytes;
                    while (left != 0u) {
                        const uint32_t chunk = static_cast<uint32_t>(
                            left > 0x7fffffc0ull ? 0x7fffffc0ull : left);
                        aclshmem_getmem(local_slot + offset,
                                        source_base + offset, chunk,
                                        static_cast<int32_t>(source));
                        offset += chunk;
                        left -= chunk;
                    }
                    FlushRange(local_slot + sizeof(SlotHeader),
                               metadata_bytes);
                }
                if (MetadataDigest(header, local_slot) !=
                    header->metadata_digest) {
                    *status = kStatusDigestMismatch;
                    break;
                }
                ready_state[source] = 1u;
                --remaining;
                ++timeline->ready_sources;
            }
        }
        if (remaining != 0u && *status == kStatusOk)
            *status = kStatusReadyTimeout;
        timeline->all_ready = AscendC::GetSystemCycle();
        timeline->headers_pulled = timeline->all_ready;
        timeline->metadata_parse_begin = timeline->all_ready;

        uint64_t total_tokens = 0u;
        uint64_t total_assignments = 0u;
        uint64_t fixed_contributors = 0u;
        if (*status == kStatusOk) {
            for (uint32_t destination = 0u;
                 destination < worker_count; ++destination) {
                reinterpret_cast<__gm__ uint32_t *>(
                    destination_row_counts)[destination] = 0u;
                reinterpret_cast<__gm__ uint32_t *>(
                    destination_assignment_counts)[destination] = 0u;
            }
            for (uint64_t i = 0u; i < required_expert_counts; ++i)
                reinterpret_cast<__gm__ uint32_t *>(expert_counts)[i] = 0u;

            reinterpret_cast<__gm__ uint32_t *>(source_token_prefix)[0] = 0u;
            for (uint32_t source = 0u; source < worker_count; ++source) {
                __gm__ SlotHeader *header =
                    reinterpret_cast<__gm__ SlotHeader *>(
                        inc_slots + MulU64ByU32(source_slot_stride, source));
                if (!AddU64(total_tokens, header->token_count,
                            &total_tokens) ||
                    !AddU64(total_assignments, header->assignment_count,
                            &total_assignments) ||
                    total_tokens > journal_token_capacity ||
                    total_assignments > journal_assignment_capacity ||
                    total_tokens > 0xffffffffull ||
                    total_assignments > 0xffffffffull) {
                    *status = kStatusCapacityExceeded;
                    break;
                }
                reinterpret_cast<__gm__ uint32_t *>(
                    source_token_prefix)[source + 1u] =
                    static_cast<uint32_t>(total_tokens);
            }
            if (!CheckedMulU64ByU32(total_tokens, worker_count,
                                    &fixed_contributors) ||
                fixed_contributors > row_map_capacity_entries ||
                fixed_contributors > 0xffffffffull)
                *status = kStatusCapacityExceeded;
        }

        __gm__ JournalSlotHeader *jheader =
            reinterpret_cast<__gm__ JournalSlotHeader *>(journal_header);
        if (*status == kStatusOk) {
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
            }
        }

        uint64_t cookie = kHashOffset;
        uint32_t journal_index = 0u;
        uint32_t journal_contributor_cursor = 0u;
        uint32_t journal_assignment_cursor = 0u;

        // Deterministic source-major planning.  It validates canonical CSR,
        // exact ordinal permutations, bounds and finite weights before any
        // destination receives a completion.  Each token allocates at most
        // one row per unique destination.
        for (uint32_t source = 0u;
             source < worker_count && *status == kStatusOk; ++source) {
            __gm__ uint8_t *slot = inc_slots +
                MulU64ByU32(source_slot_stride, source);
            __gm__ SlotHeader *header =
                reinterpret_cast<__gm__ SlotHeader *>(slot);
            cookie = HashGmBytes(
                cookie,
                reinterpret_cast<__gm__ const uint8_t *>(
                    &header->metadata_digest),
                sizeof(header->metadata_digest));
            for (uint32_t destination = 0u;
                 destination < worker_count; ++destination) {
                reinterpret_cast<__gm__ uint32_t *>(
                    source_destination_prefix)[
                        static_cast<uint64_t>(source) * worker_count +
                        destination] =
                    reinterpret_cast<__gm__ uint32_t *>(
                        destination_row_counts)[destination];
            }
            __gm__ TokenRecord *tokens =
                reinterpret_cast<__gm__ TokenRecord *>(
                    slot + header->tokens_offset);
            __gm__ AssignmentRecord *assignments =
                reinterpret_cast<__gm__ AssignmentRecord *>(
                    slot + header->assignments_offset);
            uint32_t expected_assignment = 0u;
            for (uint32_t token = 0u;
                 token < header->token_count && *status == kStatusOk;
                 ++token, ++journal_index) {
                __gm__ TokenRecord *record = tokens + token;
                if (record->source_token != token ||
                    record->assignment_begin != expected_assignment ||
                    record->assignment_count >
                        header->assignment_count - expected_assignment ||
                    record->reserved0 != 0u || record->reserved1 != 0u) {
                    *status = kStatusInvalidToken;
                    break;
                }
                for (uint32_t local = 0u;
                     local < record->assignment_count; ++local) {
                    __gm__ AssignmentRecord *assignment =
                        assignments + record->assignment_begin + local;
                    if (assignment->destination_rank >= worker_count ||
                        assignment->expert_id >= expert_count ||
                        !IsFinite(assignment->weight) ||
                        assignment->ordinal >= record->assignment_count) {
                        *status = kStatusInvalidAssignment;
                        break;
                    }
                    // With all ordinals in [0,k), pairwise uniqueness proves
                    // an exact ordinal permutation without a variable UB
                    // scratch allocation.
                    for (uint32_t previous = 0u; previous < local;
                         ++previous) {
                        if (assignments[
                                record->assignment_begin + previous]
                                .ordinal == assignment->ordinal) {
                            *status = kStatusInvalidAssignment;
                            break;
                        }
                    }
                    if (*status != kStatusOk) break;
                }
                if (*status != kStatusOk) break;

                __gm__ JournalTokenEntry *jtoken =
                    reinterpret_cast<__gm__ JournalTokenEntry *>(
                        journal_tokens) + journal_index;
                jtoken->route_key = DeviceRouteKey(source, token);
                jtoken->token_id = record->token_id;
                jtoken->owner_rank = source;
                jtoken->owner_row = token;
                jtoken->contributors_begin = journal_contributor_cursor;
                jtoken->contributors_count = 0u;
                jtoken->assignments_begin = journal_assignment_cursor;
                jtoken->assignments_count = record->assignment_count;
                jtoken->accumulator_index = journal_index;
                jtoken->flags = 0u;
                jtoken->reserved[0] = 0u;
                jtoken->reserved[1] = 0u;

                for (uint32_t local = 0u;
                     local < record->assignment_count; ++local) {
                    __gm__ AssignmentRecord *source_assignment =
                        assignments + record->assignment_begin + local;
                    __gm__ AssignmentRecord *journal_assignment =
                        reinterpret_cast<__gm__ AssignmentRecord *>(
                            journal_assignments) +
                        journal_assignment_cursor++;
                    journal_assignment->destination_rank =
                        source_assignment->destination_rank;
                    journal_assignment->expert_id =
                        source_assignment->expert_id;
                    journal_assignment->ordinal =
                        source_assignment->ordinal;
                    journal_assignment->weight = source_assignment->weight;
                }

                uint32_t contributor_count = 0u;
                for (uint32_t local = 0u;
                     local < record->assignment_count; ++local) {
                    __gm__ AssignmentRecord *first =
                        assignments + record->assignment_begin + local;
                    const uint32_t destination =
                        first->destination_rank;
                    bool first_for_destination = true;
                    for (uint32_t previous = 0u; previous < local;
                         ++previous) {
                        if (assignments[
                                record->assignment_begin + previous]
                                .destination_rank == destination) {
                            first_for_destination = false;
                            break;
                        }
                    }
                    if (!first_for_destination) continue;
                    if (journal_contributor_cursor >=
                            journal_contributor_capacity) {
                        *status = kStatusCapacityExceeded;
                        break;
                    }
                    __gm__ uint32_t *row_counts =
                        reinterpret_cast<__gm__ uint32_t *>(
                            destination_row_counts);
                    __gm__ uint32_t *assignment_counts =
                        reinterpret_cast<__gm__ uint32_t *>(
                            destination_assignment_counts);
                    if (row_counts[destination] >=
                            destination_row_capacity) {
                        *status = kStatusCapacityExceeded;
                        break;
                    }
                    const uint32_t destination_row =
                        row_counts[destination]++;
                    const uint32_t destination_assignment_begin =
                        assignment_counts[destination];
                    uint32_t destination_assignment_count = 0u;
                    for (uint32_t member = 0u;
                         member < record->assignment_count; ++member) {
                        __gm__ AssignmentRecord *assignment =
                            assignments + record->assignment_begin + member;
                        if (assignment->destination_rank != destination)
                            continue;
                        if (assignment_counts[destination] >=
                                destination_assignment_capacity) {
                            *status = kStatusCapacityExceeded;
                            break;
                        }
                        const uint64_t expert_index =
                            static_cast<uint64_t>(destination) *
                                expert_count + assignment->expert_id;
                        __gm__ ExpertAssignment *out =
                            reinterpret_cast<__gm__ ExpertAssignment *>(
                                inc_destination_assignments) +
                            static_cast<uint64_t>(destination) *
                                destination_assignment_capacity +
                            assignment_counts[destination]++;
                        out->destination_row = destination_row;
                        out->expert_id = assignment->expert_id;
                        out->expert_row =
                            reinterpret_cast<__gm__ uint32_t *>(
                                expert_counts)[expert_index]++;
                        out->ordinal = assignment->ordinal;
                        out->weight = assignment->weight;
                        out->reserved[0] = 0u;
                        out->reserved[1] = 0u;
                        out->reserved[2] = 0u;
                        ++destination_assignment_count;
                    }
                    if (*status != kStatusOk) break;

                    __gm__ DestinationRow *out_row =
                        reinterpret_cast<__gm__ DestinationRow *>(
                            inc_destination_rows) +
                        static_cast<uint64_t>(destination) *
                            destination_row_capacity +
                        destination_row;
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

                    reinterpret_cast<__gm__ uint32_t *>(row_map)[
                        static_cast<uint64_t>(journal_index) * worker_count +
                        destination] = destination_row;
                    __gm__ JournalContributor *contributor =
                        reinterpret_cast<__gm__ JournalContributor *>(
                            journal_contributors) +
                        journal_contributor_cursor++;
                    contributor->worker_rank = destination;
                    contributor->destination_row = destination_row;
                    contributor->assignment_begin =
                        destination_assignment_begin;
                    contributor->assignment_count =
                        destination_assignment_count;
                    ++contributor_count;
                }
                jtoken->contributors_count = contributor_count;
                expected_assignment += record->assignment_count;
            }
            if (*status == kStatusOk &&
                expected_assignment != header->assignment_count)
                *status = kStatusInvalidToken;
        }

        if (*status == kStatusOk) {
            for (uint32_t destination = 0u;
                 destination < worker_count; ++destination) {
                reinterpret_cast<__gm__ uint32_t *>(
                    source_destination_prefix)[
                        static_cast<uint64_t>(worker_count) * worker_count +
                        destination] =
                    reinterpret_cast<__gm__ uint32_t *>(
                        destination_row_counts)[destination];
            }
            jheader->token_count = static_cast<uint32_t>(total_tokens);
            jheader->contributor_count = journal_contributor_cursor;
            jheader->dispatch_cookie = cookie == 0u ? 1u : cookie;
            timeline->metadata_parse_done = AscendC::GetSystemCycle();

            // Publish all layout/control arrays before data AIVs read row_map
            // and before remote workers can observe their metadata.
            FlushRange(row_map,
                       MulU64ByU32(total_tokens, worker_count) *
                           sizeof(uint32_t));
            FlushRange(source_token_prefix,
                       (static_cast<uint64_t>(worker_count) + 1u) *
                           sizeof(uint32_t));
            FlushRange(source_destination_prefix,
                       required_prefix_entries * sizeof(uint32_t));
            FlushRange(destination_row_counts,
                       static_cast<uint64_t>(worker_count) *
                           sizeof(uint32_t));
            FlushRange(destination_assignment_counts,
                       static_cast<uint64_t>(worker_count) *
                           sizeof(uint32_t));
            FlushRange(expert_counts,
                       required_expert_counts * sizeof(uint32_t));
            FlushRange(journal_tokens,
                       total_tokens * sizeof(JournalTokenEntry));
            FlushRange(journal_contributors,
                       static_cast<uint64_t>(journal_contributor_cursor) *
                           sizeof(JournalContributor));
            FlushRange(journal_assignments,
                       total_assignments * sizeof(AssignmentRecord));
            dcci_cacheline(journal_header);

            // Metadata is transferred before payload, but becomes consumable
            // only after the final DestinationCompletion publication.
            for (uint32_t destination = 0u;
                 destination < worker_count; ++destination) {
                const uint32_t rows =
                    reinterpret_cast<__gm__ uint32_t *>(
                        destination_row_counts)[destination];
                const uint32_t assignments =
                    reinterpret_cast<__gm__ uint32_t *>(
                        destination_assignment_counts)[destination];
                __gm__ uint8_t *row_source = inc_destination_rows +
                    MulU64ByU32(destination_row_capacity, destination) *
                        sizeof(DestinationRow);
                __gm__ uint8_t *assignment_source =
                    inc_destination_assignments +
                    MulU64ByU32(destination_assignment_capacity,
                                destination) *
                        sizeof(ExpertAssignment);
                __gm__ uint8_t *expert_source = expert_counts +
                    static_cast<uint64_t>(destination) * expert_count *
                        sizeof(uint32_t);
                FlushRange(row_source,
                           static_cast<uint64_t>(rows) *
                               sizeof(DestinationRow));
                FlushRange(assignment_source,
                           static_cast<uint64_t>(assignments) *
                               sizeof(ExpertAssignment));
                FlushRange(expert_source,
                           static_cast<uint64_t>(expert_count) *
                               sizeof(uint32_t));
                if (rows != 0u)
                    PutGmRange(destination_rows_base, row_source,
                               static_cast<uint64_t>(rows) *
                                   sizeof(DestinationRow),
                               static_cast<int32_t>(destination));
                if (assignments != 0u)
                    PutGmRange(destination_assignments_base,
                               assignment_source,
                               static_cast<uint64_t>(assignments) *
                                   sizeof(ExpertAssignment),
                               static_cast<int32_t>(destination));
                PutGmRange(destination_expert_counts_base, expert_source,
                           static_cast<uint64_t>(expert_count) *
                               sizeof(uint32_t),
                           static_cast<int32_t>(destination));
                aclshmem_quiet();
            }
            timeline->reorg_done = AscendC::GetSystemCycle();
            timeline->hidden_get_begin = timeline->reorg_done;
            timeline->fanout_put_begin = timeline->reorg_done;
        }
        dcci_cacheline(status_line);
    }

    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);

    // Every (source, channel) owns a deterministic set of hidden row tiles.
    // The next remote GET is issued into the other 12-KiB UB slot before the
    // current tile is fanned out, overlapping worker->INC MTE2 with
    // INC->worker MTE3.  The hidden row is pulled once regardless of top-k.
    if (*status == kStatusOk && block < required_channels) {
        const uint32_t source = block / channels_per_source;
        const uint32_t channel = block % channels_per_source;
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
        const uint64_t tiles_per_row =
            (row_bytes + kHiddenTileBytes - 1u) / kHiddenTileBytes;
        const uint64_t total_tasks =
            MulU64ByU32(tiles_per_row, header->token_count);
        __ubuf__ uint8_t *ub[2]{
            reinterpret_cast<__ubuf__ uint8_t *>(0),
            reinterpret_cast<__ubuf__ uint8_t *>(kUbSlotBytes)};
        bool put_busy[2]{false, false};

        uint64_t task = channel;
        if (task < total_tasks) {
            uint32_t ping = 0u;
            uint32_t token = static_cast<uint32_t>(task / tiles_per_row);
            uint64_t tile = task % tiles_per_row;
            uint64_t byte_offset = tile * kHiddenTileBytes;
            uint32_t bytes = static_cast<uint32_t>(
                row_bytes - byte_offset < kHiddenTileBytes
                    ? row_bytes - byte_offset
                    : kHiddenTileBytes);
            PullHiddenToUb(
                ub[ping], source_base + header->hidden_offset +
                    MulU64ByU32(row_bytes, token) + byte_offset,
                bytes, static_cast<int32_t>(source), ping);

            while (task < total_tasks) {
                WaitHiddenPull(ping);
                const uint64_t next_task = task + channels_per_source;
                const uint32_t next_ping = ping ^ 1u;
                if (next_task < total_tasks) {
                    if (put_busy[next_ping]) {
                        WaitPutDone(next_ping);
                        put_busy[next_ping] = false;
                    }
                    const uint32_t next_token = static_cast<uint32_t>(
                        next_task / tiles_per_row);
                    const uint64_t next_tile = next_task % tiles_per_row;
                    const uint64_t next_offset =
                        next_tile * kHiddenTileBytes;
                    const uint32_t next_bytes = static_cast<uint32_t>(
                        row_bytes - next_offset < kHiddenTileBytes
                            ? row_bytes - next_offset
                            : kHiddenTileBytes);
                    PullHiddenToUb(
                        ub[next_ping], source_base + header->hidden_offset +
                            MulU64ByU32(row_bytes, next_token) +
                            next_offset,
                        next_bytes, static_cast<int32_t>(source), next_ping);
                }

                __gm__ TokenRecord *record = tokens + token;
                const uint32_t global_token =
                    reinterpret_cast<__gm__ uint32_t *>(
                        source_token_prefix)[source] + token;
                bool issued_put = false;
                for (uint32_t local = 0u;
                     local < record->assignment_count; ++local) {
                    const uint32_t destination =
                        assignments[record->assignment_begin + local]
                            .destination_rank;
                    bool first_for_destination = true;
                    for (uint32_t previous = 0u; previous < local;
                         ++previous) {
                        if (assignments[
                                record->assignment_begin + previous]
                                .destination_rank == destination) {
                            first_for_destination = false;
                            break;
                        }
                    }
                    if (!first_for_destination) continue;
                    const uint32_t destination_row =
                        reinterpret_cast<__gm__ uint32_t *>(row_map)[
                            static_cast<uint64_t>(global_token) *
                                worker_count + destination];
                    if (destination_row == kInvalidRow ||
                        destination_row >= destination_row_capacity) {
                        *status = kStatusInvalidToken;
                        dcci_cacheline(status_line);
                        break;
                    }
                    PutHiddenFromUb(
                        destination_hidden_base +
                            MulU64ByU32(row_bytes, destination_row) +
                                byte_offset,
                        ub[ping], bytes,
                        static_cast<int32_t>(destination), ping);
                    issued_put = true;
                }
                if (issued_put) {
                    SetPutDone(ping);
                    put_busy[ping] = true;
                }
                if (*status != kStatusOk) break;
                task = next_task;
                if (task >= total_tasks) break;
                ping = next_ping;
                token = static_cast<uint32_t>(task / tiles_per_row);
                tile = task % tiles_per_row;
                byte_offset = tile * kHiddenTileBytes;
                bytes = static_cast<uint32_t>(
                    row_bytes - byte_offset < kHiddenTileBytes
                        ? row_bytes - byte_offset
                        : kHiddenTileBytes);
            }
        }
        for (uint32_t slot_index = 0u; slot_index < 2u; ++slot_index)
            if (put_busy[slot_index]) WaitPutDone(slot_index);
        // Event completion releases both UB slots, while mte_quiet closes
        // remote visibility before the cross-AIV join and completion.
        aclshmemx_mte_quiet();
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
        timeline->hidden_get_done = AscendC::GetSystemCycle();
        timeline->fanout_put_done = timeline->hidden_get_done;

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
    uint8_t *status_line, uint64_t ffts_addr, uint64_t session_id,
    uint64_t placement_epoch, uint64_t generation, uint64_t sequence,
    uint64_t source_slot_stride, uint64_t destination_hidden_slot_stride,
    uint64_t destination_rows_slot_stride,
    uint64_t destination_assignments_slot_stride,
    uint64_t destination_expert_counts_slot_stride,
    uint64_t journal_token_capacity, uint64_t journal_contributor_capacity,
    uint64_t journal_assignment_capacity,
    uint64_t destination_row_capacity,
    uint64_t destination_assignment_capacity,
    uint64_t row_map_capacity_entries,
    uint64_t source_destination_prefix_capacity_entries,
    uint64_t expert_counts_capacity_entries, uint32_t worker_count,
    uint32_t expert_count, uint32_t hidden, uint32_t dtype, int32_t inc_pe,
    uint32_t region_id, uint32_t wave, uint32_t ring_slot,
    uint32_t slot_count, uint32_t channels_per_source, uint64_t spin_cap)
{
    uint32_t selected_channels = channels_per_source;
    if (selected_channels == 0u) {
        const uint32_t portable_max = worker_count == 0u
            ? 0u
            : block_dim / worker_count;
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
        expert_counts, status_line, ffts_addr, session_id, placement_epoch,
        generation, sequence, source_slot_stride,
        destination_hidden_slot_stride, destination_rows_slot_stride,
        destination_assignments_slot_stride,
        destination_expert_counts_slot_stride, journal_token_capacity,
        journal_contributor_capacity, journal_assignment_capacity,
        destination_row_capacity, destination_assignment_capacity,
        row_map_capacity_entries,
        source_destination_prefix_capacity_entries,
        expert_counts_capacity_entries, worker_count, expert_count, hidden,
        dtype, inc_pe, region_id, wave, ring_slot, slot_count,
        selected_channels, spin_cap);
}
