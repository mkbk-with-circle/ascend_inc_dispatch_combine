#include "kernel_operator.h"
#include "shmem.h"

#include "inc_dc_pull_combine_v2.h"
#include "inc_dc_vector_reduce_aicore.h"

using namespace inc::dc::pull_v2;

// Pull-Combine V2 device data plane.
//
// The source payload is the canonical array
// partial[source_rank][destination_row][hidden].  The source-major pull plan
// is compiled once per Dispatch journal and consumed directly: this kernel
// never pulls token ids and never builds a worker-by-token row map.  A compact
// O(contributor) linked index is built in caller-provided INC scratch so an
// accumulator lane can visit only its actual contributors.
//
// A production launch uses half of the INC's live ordinary AIVs. Pull-
// Dispatch owns the other half, so D(N+1) and C(N) can overlap without
// changing either kernel's resource allocation. The host derives block_dim
// from the queried vector-core count; this file does not encode one SKU's
// 24-AIV half. Worker PEs use block zero only to publish their immutable
// READY record.

namespace {

constexpr uint32_t kTileElements = 1536u;
constexpr uint32_t kTileBytes = kTileElements * sizeof(float);
constexpr uint32_t kInvalidIndex = ~0u;
constexpr uint64_t kHashOffset = 1469598103934665603ull;
constexpr uint64_t kSourceScratchStride = kPullCombineV2Alignment;
constexpr uint64_t kSourceAcceptedCycleOffset = 8u;

constexpr uint32_t kVecPingMte2V = 0u;
constexpr uint32_t kVecPingVMte2 = 1u;
constexpr uint32_t kVecPongMte2V = 2u;
constexpr uint32_t kVecPongVMte2 = 3u;
constexpr uint32_t kVecMte3V = 4u;
constexpr uint32_t kVecVMte3 = 5u;

static_assert(kTileBytes * 4u <= INC_VEC_UB_BUDGET_BYTES,
              "Pull-Combine V2 input/output double buffers exceed AIV UB");

// Values intentionally match CombineV2Status so a device failure can be
// passed through the thin host API without translation.
constexpr uint32_t kStatusOk = 0u;
constexpr uint32_t kStatusInvalidArgument = 1u;
constexpr uint32_t kStatusInvalidJournal = 2u;
constexpr uint32_t kStatusInvalidRegistration = 3u;
constexpr uint32_t kStatusInvalidReady = 4u;
constexpr uint32_t kStatusCookieMismatch = 5u;
constexpr uint32_t kStatusStaleEpoch = 6u;
constexpr uint32_t kStatusSizeOverflow = 7u;
constexpr uint32_t kStatusCapacityExceeded = 8u;
constexpr uint32_t kStatusDuplicateSource = 9u;
constexpr uint32_t kStatusNotReady = 10u;
constexpr uint32_t kStatusInvalidState = 11u;

// Both completion records use a final publication word.  Consumers validate
// the complete wave identity and publication before releasing a ring slot.
struct alignas(64) CombineSourceAckV2 {
    uint32_t magic = kPullCombineV2Magic;
    uint16_t abi_version = kPullCombineV2AbiVersion;
    uint16_t struct_bytes = sizeof(CombineSourceAckV2);
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint64_t dispatch_cookie = 0u;
    uint32_t wave = 0u;
    uint32_t source_rank = 0u;
    uint32_t source_region_id = 0u;
    uint32_t status = 0u;
    uint16_t ring_slot = 0u;
    uint16_t flags = 0u;
    uint32_t row_count = 0u;
    uint64_t bytes_consumed = 0u;
    uint64_t reserved[5]{};
    uint64_t publication = 0u;
};
static_assert(sizeof(CombineSourceAckV2) == 128u,
              "Pull-Combine V2 source ACK ABI drift");
static_assert(__builtin_offsetof(CombineSourceAckV2, publication) +
                  sizeof(uint64_t) == sizeof(CombineSourceAckV2),
              "source ACK publication must be last");

struct alignas(64) CombineOwnerCompletionV2 {
    uint32_t magic = kPullCombineV2Magic;
    uint16_t abi_version = kPullCombineV2AbiVersion;
    uint16_t struct_bytes = sizeof(CombineOwnerCompletionV2);
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint64_t dispatch_cookie = 0u;
    uint32_t wave = 0u;
    uint32_t owner_rank = 0u;
    uint32_t status = 0u;
    uint32_t row_count = 0u;
    uint16_t ring_slot = 0u;
    uint16_t flags = 0u;
    uint32_t reserved0 = 0u;
    uint64_t bytes_produced = 0u;
    uint64_t reserved[5]{};
    uint64_t publication = 0u;
};
static_assert(sizeof(CombineOwnerCompletionV2) == 128u,
              "Pull-Combine V2 owner completion ABI drift");
static_assert(__builtin_offsetof(CombineOwnerCompletionV2, publication) +
                  sizeof(uint64_t) == sizeof(CombineOwnerCompletionV2),
              "owner completion publication must be last");

struct alignas(64) CombineDeviceTimelineV2 {
    uint32_t status = 0u;
    uint32_t ready_sources = 0u;
    uint64_t kernel_start = 0u;
    uint64_t plan_index_done = 0u;
    uint64_t first_ready = 0u;
    uint64_t all_ready = 0u;
    uint64_t first_get = 0u;
    uint64_t last_get = 0u;
    uint64_t last_reduce = 0u;
    uint64_t last_owner_put = 0u;
    uint64_t source_acks_done = 0u;
    uint64_t owner_completions_done = 0u;
    uint64_t kernel_done = 0u;
    uint64_t reserved[4]{};
};
static_assert(sizeof(CombineDeviceTimelineV2) == 128u,
              "Pull-Combine V2 timeline ABI drift");

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

__aicore__ inline uint64_t HashByte(uint64_t hash, uint8_t value)
{
    const uint64_t x = hash ^ value;
    return (x << 40u) + (x << 8u) + (x << 7u) + (x << 5u) +
        (x << 4u) + (x << 1u) + x;
}

__aicore__ inline uint64_t HashGmBytes(uint64_t hash,
                                       __gm__ const uint8_t *data,
                                       uint64_t bytes)
{
    for (uint64_t i = 0u; i < bytes; ++i)
        hash = HashByte(hash, data[i]);
    return hash;
}

__aicore__ inline uint64_t HashU16(uint64_t hash, uint16_t value)
{
    for (uint32_t i = 0u; i < sizeof(value); ++i)
        hash = HashByte(hash, static_cast<uint8_t>(value >> (8u * i)));
    return hash;
}

__aicore__ inline uint64_t HashU32(uint64_t hash, uint32_t value)
{
    for (uint32_t i = 0u; i < sizeof(value); ++i)
        hash = HashByte(hash, static_cast<uint8_t>(value >> (8u * i)));
    return hash;
}

__aicore__ inline uint64_t HashU64(uint64_t hash, uint64_t value)
{
    for (uint32_t i = 0u; i < sizeof(value); ++i)
        hash = HashByte(hash, static_cast<uint8_t>(value >> (8u * i)));
    return hash;
}

__aicore__ inline uint64_t ReadyPublication(
    __gm__ const CombineReadyV2 *ready)
{
    uint64_t hash = kHashOffset;
    hash = HashU32(hash, ready->magic);
    hash = HashU16(hash, ready->abi_version);
    hash = HashU16(hash, ready->struct_bytes);
    hash = HashU64(hash, ready->session_id);
    hash = HashU64(hash, ready->placement_epoch);
    hash = HashU64(hash, ready->generation);
    hash = HashU64(hash, ready->sequence);
    hash = HashU64(hash, ready->dispatch_cookie);
    hash = HashU32(hash, ready->wave);
    hash = HashU32(hash, ready->source_rank);
    hash = HashU32(hash, ready->source_region_id);
    hash = HashU16(hash, ready->ring_slot);
    hash = HashU16(hash, ready->flags);
    hash = HashU32(hash, ready->row_count);
    hash = HashU32(hash, ready->hidden);
    hash = HashU32(hash, ready->partial_dtype);
    hash = HashU64(hash, ready->source_offset);
    hash = HashU64(hash, ready->payload_bytes);
    for (uint32_t i = 0u; i < 3u; ++i)
        hash = HashU64(hash, ready->reserved[i]);
    return hash == 0u ? 1u : hash;
}

__aicore__ inline uint64_t ReadyNoticePublication(
    __gm__ const CombineReadyNoticeV2 *notice)
{
    uint64_t hash = kHashOffset;
    hash = HashU32(hash, notice->magic);
    hash = HashU16(hash, notice->abi_version);
    hash = HashU16(hash, notice->struct_bytes);
    hash = HashU64(hash, notice->session_id);
    hash = HashU64(hash, notice->placement_epoch);
    hash = HashU64(hash, notice->generation);
    hash = HashU64(hash, notice->sequence);
    hash = HashU32(hash, notice->wave);
    hash = HashU32(hash, notice->source_rank);
    hash = HashU16(hash, notice->ring_slot);
    hash = HashU16(hash, notice->flags);
    hash = HashU32(hash, notice->reserved0);
    return hash == 0u ? 1u : hash;
}

__aicore__ inline bool ReadyNoticeValid(
    __gm__ const CombineReadyNoticeV2 *notice, uint32_t source,
    uint32_t workers, uint32_t slots, uint64_t session_id,
    uint64_t placement_epoch, uint64_t generation, uint64_t sequence,
    uint32_t wave, uint16_t ring_slot)
{
    return source < workers && notice->magic == kPullCombineV2Magic &&
        notice->abi_version == kPullCombineV2AbiVersion &&
        notice->struct_bytes == sizeof(CombineReadyNoticeV2) &&
        notice->session_id == session_id &&
        notice->placement_epoch == placement_epoch &&
        notice->generation == generation && notice->sequence == sequence &&
        notice->wave == wave && notice->source_rank == source &&
        notice->ring_slot == ring_slot && notice->ring_slot < slots &&
        notice->flags == 0u && notice->reserved0 == 0u &&
        notice->publication != 0u &&
        notice->publication == ReadyNoticePublication(notice);
}

__aicore__ inline void FlushRange(__gm__ uint8_t *base, uint64_t bytes)
{
    for (uint64_t offset = 0u; offset < bytes;
         offset += kPullCombineV2Alignment)
        dcci_cacheline(base + offset);
}

__aicore__ inline __gm__ uint32_t *SourceReadyAddress(
    __gm__ uint8_t *source_ready, uint32_t source)
{
    return reinterpret_cast<__gm__ uint32_t *>(
        source_ready + MulU64ByU32(kSourceScratchStride, source));
}

__aicore__ inline __gm__ uint64_t *SourceAcceptedCycleAddress(
    __gm__ uint8_t *source_ready, uint32_t source)
{
    return reinterpret_cast<__gm__ uint64_t *>(
        source_ready + MulU64ByU32(kSourceScratchStride, source) +
        kSourceAcceptedCycleOffset);
}

__aicore__ inline __gm__ uint64_t *SourcePayloadOffsetAddress(
    __gm__ uint8_t *source_payload_offsets, uint32_t source)
{
    return reinterpret_cast<__gm__ uint64_t *>(
        source_payload_offsets +
        MulU64ByU32(kSourceScratchStride, source));
}

__aicore__ inline uint64_t LoadSourcePayloadOffset(
    __gm__ uint8_t *source_payload_offsets, uint32_t source)
{
    __gm__ uint64_t *offset =
        SourcePayloadOffsetAddress(source_payload_offsets, source);
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(offset));
    return *reinterpret_cast<__gm__ volatile uint64_t *>(offset);
}

__aicore__ inline void SetFailure(__gm__ uint32_t *status,
                                   uint32_t failure)
{
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(status));
    if (*status == kStatusOk) {
        *status = failure;
        AscendC::PipeBarrier<PIPE_ALL>();
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(status));
    }
}

__aicore__ inline bool RegistrationValid(
    __gm__ const CombineRegionRegistration *registration, uint32_t source,
    uint32_t workers, uint32_t slots, uint64_t session_id,
    uint64_t placement_epoch)
{
    uint64_t required = 0u;
    return source < workers && registration->session_id == session_id &&
        registration->placement_epoch == placement_epoch &&
        registration->source_rank == source &&
        registration->region_id != 0u &&
        registration->slot_count >= slots &&
        registration->alignment >= kPullCombineV2Alignment &&
        (registration->alignment & (registration->alignment - 1u)) == 0u &&
        registration->slot_stride != 0u &&
        registration->slot_stride % registration->alignment == 0u &&
        CheckedMulU64ByU32(registration->slot_stride,
                           registration->slot_count, &required) &&
        required <= registration->region_bytes &&
        registration->reserved[0] == 0u &&
        registration->reserved[1] == 0u;
}

__aicore__ inline uint32_t ValidateReady(
    __gm__ const CombineReadyV2 *ready,
    __gm__ const CombineRegionRegistration *registration,
    __gm__ const uint64_t *source_offsets, uint32_t source,
    uint32_t workers, uint32_t hidden, uint32_t slots,
    uint64_t session_id, uint64_t placement_epoch, uint64_t generation,
    uint64_t sequence, uint64_t dispatch_cookie, uint32_t wave,
    uint16_t ring_slot)
{
    if (ready->magic != kPullCombineV2Magic ||
        ready->abi_version != kPullCombineV2AbiVersion ||
        ready->struct_bytes != sizeof(CombineReadyV2) ||
        ready->session_id != session_id ||
        ready->placement_epoch != placement_epoch ||
        ready->source_rank != source || source >= workers ||
        ready->source_region_id != registration->region_id ||
        ready->ring_slot >= slots ||
        ready->ring_slot >= registration->slot_count ||
        ready->flags != kPullCombineV2CanonicalRows ||
        ready->hidden != hidden ||
        ready->partial_dtype !=
            static_cast<uint32_t>(PartialDataType::FP32) ||
        ready->reserved[0] != 0u || ready->reserved[1] != 0u ||
        ready->reserved[2] != 0u || ready->publication == 0u ||
        ready->publication != ReadyPublication(ready))
        return kStatusInvalidReady;
    if (ready->dispatch_cookie != dispatch_cookie)
        return kStatusCookieMismatch;
    if (ready->generation != generation || ready->sequence != sequence ||
        ready->wave != wave || ready->ring_slot != ring_slot)
        return kStatusStaleEpoch;

    const uint64_t expected_rows =
        source_offsets[source + 1u] - source_offsets[source];
    uint64_t expected_bytes = 0u;
    uint64_t slot_base = 0u;
    uint64_t slot_end = 0u;
    uint64_t payload_end = 0u;
    if (expected_rows > 0xffffffffull ||
        !CheckedMulU64ByU32(expected_rows, hidden, &expected_bytes) ||
        !CheckedMulU64ByU32(expected_bytes, sizeof(float),
                            &expected_bytes) ||
        !CheckedMulU64ByU32(registration->slot_stride, ring_slot,
                            &slot_base) ||
        !AddU64(slot_base, registration->slot_stride, &slot_end) ||
        !AddU64(ready->source_offset, ready->payload_bytes, &payload_end))
        return kStatusSizeOverflow;
    if (ready->row_count != expected_rows ||
        ready->payload_bytes != expected_bytes ||
        ready->source_offset % registration->alignment != 0u ||
        ready->source_offset < slot_base || payload_end > slot_end ||
        payload_end > registration->region_bytes)
        return kStatusCapacityExceeded;
    return kStatusOk;
}

// Publish a 128-byte control record whose commit word lives in its second
// cacheline.  A scalar remote store is not a reliable cacheline publication
// primitive on the target platform: a polling peer can continue observing an
// old mapped line long after the scalar operation completed.  Use full-line
// RMA plus quiet boundaries instead.
//
// The first transfer revokes any previous generation before line zero is
// replaced.  This prevents a consumer from combining a new identity line
// with an old, non-zero commit word.  The final transfer carries the complete
// second line and the non-zero commit word together.
__aicore__ inline void PublishControlRecord128(
    __gm__ uint8_t *local, __gm__ uint8_t *remote,
    uint32_t publication_offset, uint64_t publication, int32_t remote_pe)
{
    __gm__ uint64_t *local_publication =
        reinterpret_cast<__gm__ uint64_t *>(local + publication_offset);
    constexpr uint32_t kSecondLineOffset = kPullCombineV2Alignment;

    *local_publication = 0u;
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(local + kSecondLineOffset);
    aclshmem_putmem(remote + kSecondLineOffset,
                    local + kSecondLineOffset,
                    kPullCombineV2Alignment, remote_pe);
    aclshmem_quiet();

    *local_publication = publication;
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(local);
    dcci_cacheline(local + kSecondLineOffset);
    aclshmem_putmem(remote, local, kPullCombineV2Alignment, remote_pe);
    aclshmem_quiet();
    aclshmem_putmem(remote + kSecondLineOffset,
                    local + kSecondLineOffset,
                    kPullCombineV2Alignment, remote_pe);
    aclshmem_quiet();
}

__aicore__ inline void PublishReadyNoticeToInc(
    __gm__ CombineReadyNoticeV2 *local,
    __gm__ CombineReadyNoticeV2 *remote,
    int32_t inc_pe)
{
    // A notice is exactly one cacheline and is the worker's sole remote
    // control transfer for this wave.  READY and payload remain local until
    // the INC observes this notice and actively pulls them.
    __gm__ uint8_t *local_line =
        reinterpret_cast<__gm__ uint8_t *>(local);
    dcci_cacheline(local_line);
    aclshmem_putmem(remote, local, sizeof(*local), inc_pe);
    aclshmem_quiet();
}

__aicore__ inline void PublishSourceAck(
    __gm__ CombineSourceAckV2 *ack, uint32_t source, uint32_t status,
    uint32_t rows, uint64_t bytes, uint32_t region_id,
    uint64_t session_id, uint64_t placement_epoch, uint64_t generation,
    uint64_t sequence, uint64_t dispatch_cookie, uint32_t wave,
    uint16_t ring_slot)
{
    ack->magic = kPullCombineV2Magic;
    ack->abi_version = kPullCombineV2AbiVersion;
    ack->struct_bytes = sizeof(CombineSourceAckV2);
    ack->session_id = session_id;
    ack->placement_epoch = placement_epoch;
    ack->generation = generation;
    ack->sequence = sequence;
    ack->dispatch_cookie = dispatch_cookie;
    ack->wave = wave;
    ack->source_rank = source;
    ack->source_region_id = region_id;
    ack->status = status;
    ack->ring_slot = ring_slot;
    ack->flags = 0u;
    ack->row_count = status == kStatusOk ? rows : 0u;
    ack->bytes_consumed = status == kStatusOk ? bytes : 0u;
    for (uint32_t i = 0u; i < 5u; ++i) ack->reserved[i] = 0u;
    ack->publication = 0u;
    FlushRange(reinterpret_cast<__gm__ uint8_t *>(ack), sizeof(*ack));
    uint64_t publication = HashGmBytes(
        kHashOffset, reinterpret_cast<__gm__ uint8_t *>(ack),
        __builtin_offsetof(CombineSourceAckV2, publication));
    if (publication == 0u) publication = 1u;
    PublishControlRecord128(
        reinterpret_cast<__gm__ uint8_t *>(ack),
        reinterpret_cast<__gm__ uint8_t *>(ack),
        __builtin_offsetof(CombineSourceAckV2, publication), publication,
        static_cast<int32_t>(source));
}

__aicore__ inline void PublishOwnerCompletion(
    __gm__ CombineOwnerCompletionV2 *completion, uint32_t owner,
    uint32_t status, uint32_t rows, uint64_t bytes,
    uint64_t session_id, uint64_t placement_epoch, uint64_t generation,
    uint64_t sequence, uint64_t dispatch_cookie, uint32_t wave,
    uint16_t ring_slot)
{
    completion->magic = kPullCombineV2Magic;
    completion->abi_version = kPullCombineV2AbiVersion;
    completion->struct_bytes = sizeof(CombineOwnerCompletionV2);
    completion->session_id = session_id;
    completion->placement_epoch = placement_epoch;
    completion->generation = generation;
    completion->sequence = sequence;
    completion->dispatch_cookie = dispatch_cookie;
    completion->wave = wave;
    completion->owner_rank = owner;
    completion->status = status;
    completion->row_count = status == kStatusOk ? rows : 0u;
    completion->ring_slot = ring_slot;
    completion->flags = 0u;
    completion->reserved0 = 0u;
    completion->bytes_produced = status == kStatusOk ? bytes : 0u;
    for (uint32_t i = 0u; i < 5u; ++i) completion->reserved[i] = 0u;
    completion->publication = 0u;
    FlushRange(reinterpret_cast<__gm__ uint8_t *>(completion),
               sizeof(*completion));
    uint64_t publication = HashGmBytes(
        kHashOffset, reinterpret_cast<__gm__ uint8_t *>(completion),
        __builtin_offsetof(CombineOwnerCompletionV2, publication));
    if (publication == 0u) publication = 1u;
    PublishControlRecord128(
        reinterpret_cast<__gm__ uint8_t *>(completion),
        reinterpret_cast<__gm__ uint8_t *>(completion),
        __builtin_offsetof(CombineOwnerCompletionV2, publication),
        publication, static_cast<int32_t>(owner));
}

__aicore__ inline void PullFp32ToUb(
    __ubuf__ uint8_t *destination, __gm__ float *source,
    uint32_t elements, int32_t source_pe, uint32_t ping)
{
    const uint32_t ready = ping == 0u ? kVecPingVMte2 : kVecPongVMte2;
    const uint32_t done = ping == 0u ? kVecPingMte2V : kVecPongMte2V;
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ready);
    aclshmemx_mte_get_nbi(
        reinterpret_cast<__ubuf__ float *>(destination), source,
        elements, source_pe, ping);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(done);
}

__aicore__ inline void PutFp32ToOwner(
    __ubuf__ uint8_t *source, __gm__ float *destination,
    uint32_t elements, int32_t owner)
{
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kVecVMte3);
    aclshmemx_mte_put_nbi(destination,
                          reinterpret_cast<__ubuf__ float *>(source),
                          elements, owner, 0u);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kVecMte3V);
}

__aicore__ inline bool SourceReady(__gm__ uint8_t *source_ready,
                                    uint32_t source)
{
    __gm__ uint32_t *state = SourceReadyAddress(source_ready, source);
    __gm__ uint8_t *address = reinterpret_cast<__gm__ uint8_t *>(state);
    dcci_cacheline(address);
    return *reinterpret_cast<__gm__ volatile uint32_t *>(state) == 1u;
}

__aicore__ inline bool SeenSource(uint64_t low, uint64_t high,
                                   uint32_t source)
{
    const uint64_t bit = 1ull << (source & 63u);
    return source < 64u ? (low & bit) != 0u : (high & bit) != 0u;
}

__aicore__ inline void MarkSource(uint64_t *low, uint64_t *high,
                                  uint32_t source)
{
    const uint64_t bit = 1ull << (source & 63u);
    if (source < 64u)
        *low |= bit;
    else
        *high |= bit;
}

__aicore__ inline bool WaitSourceReady(
    __gm__ uint8_t *source_ready, uint32_t source, uint64_t spin_cap,
    __gm__ uint32_t *status)
{
    const uint64_t wait_begin = AscendC::GetSystemCycle();
    while (AscendC::GetSystemCycle() - wait_begin < spin_cap) {
        if (SourceReady(source_ready, source)) return true;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(status));
        if (*status != kStatusOk) return false;
    }
    SetFailure(status, kStatusNotReady);
    return false;
}

// All contributor counts use the same bounded token validator and tile loop.
// Two input buffers and alternating output buffers fit 24 KiB. The previous
// PUT overlaps the next tile's GET/reduction without changing wire semantics.
__aicore__ inline bool ReduceContributorTaskRange(
    __gm__ uint8_t *symmetric_partials, __gm__ uint8_t *owner_output_base,
    __gm__ const CombinePullOp *pulls, __gm__ const uint32_t *pull_next,
    __gm__ const uint32_t *heads, __gm__ const uint32_t *counts,
    __gm__ const uint32_t *result_indices,
    __gm__ const CombineResultOp *results, __gm__ uint8_t *source_ready,
    __gm__ uint8_t *payload_offsets,
    __gm__ const uint64_t *source_offsets,
    __gm__ const uint64_t *owner_offsets,
    uint64_t pull_count, uint64_t result_count,
    uint64_t task_begin, uint64_t task_end, uint64_t tiles_per_row,
    uint32_t hidden, uint64_t row_bytes, uint64_t output_slot_offset,
    uint32_t worker_count, uint64_t spin_cap, __gm__ uint32_t *status)
{
    __ubuf__ uint8_t *input0_ub = reinterpret_cast<__ubuf__ uint8_t *>(0);
    __ubuf__ uint8_t *input1_ub = input0_ub + kTileBytes;
    __ubuf__ uint8_t *output0_ub = input1_ub + kTileBytes;
    __ubuf__ uint8_t *output1_ub = output0_ub + kTileBytes;
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kVecPingVMte2);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kVecPongVMte2);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kVecMte3V);
    uint32_t sources[kPullDispatchMaxWorkers];
    uint64_t row_offsets[kPullDispatchMaxWorkers];
    uint64_t registered_offsets[kPullDispatchMaxWorkers];
    uint64_t ready_low = 0u, ready_high = 0u;
    uint32_t cached_accumulator = kInvalidIndex;
    uint32_t count = 0u, owner = 0u, output_ping = 0u, rotation = 0u;
    __gm__ float *owner_output = nullptr;
    bool ok = true;
    // row_bytes is fixed for the whole launch. Hoist its overflow bound and
    // power-of-two classification out of the per-token address validation.
    const uint64_t max_row_index = ~0ull / row_bytes;
    uint32_t row_shift = 0u;
    uint64_t stride_bits = row_bytes;
    while (stride_bits > 1u && (stride_bits & 1u) == 0u) {
        stride_bits >>= 1u;
        ++row_shift;
    }
    if (stride_bits != 1u) row_shift = 64u;

    for (uint64_t task = task_begin; task < task_end; ++task) {
        if (((task - task_begin) & 63u) == 0u) {
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(status));
            if (*status != kStatusOk) { ok = false; break; }
        }
        const uint32_t accumulator = static_cast<uint32_t>(task / tiles_per_row);
        if (cached_accumulator != accumulator) {
            count = counts[accumulator];
            const uint32_t result_index = result_indices[accumulator];
            if (count > worker_count || result_index >= result_count) {
                SetFailure(status, kStatusInvalidJournal); ok = false; break;
            }
            __gm__ const CombineResultOp *result = results + result_index;
            owner = result->owner_rank;
            const uint32_t token = result->journal_token;
            if (result->accumulator_index != accumulator ||
                token >= result_count || result->expected_contributors != count ||
                owner >= worker_count || result_index < owner_offsets[owner] ||
                result_index >= owner_offsets[owner + 1u] ||
                result->owner_row != result_index - owner_offsets[owner] ||
                result->reserved[0] || result->reserved[1] || result->reserved[2]) {
                SetFailure(status, kStatusInvalidJournal); ok = false; break;
            }
            uint64_t owner_row_offset = 0u;
            if (result->owner_row > max_row_index) {
                SetFailure(status, kStatusSizeOverflow); ok = false; break;
            }
            owner_row_offset = row_shift < 64u
                ? static_cast<uint64_t>(result->owner_row) << row_shift
                : MulU64ByU32(row_bytes, result->owner_row);
            owner_output = reinterpret_cast<__gm__ float *>(
                owner_output_base + output_slot_offset + owner_row_offset);
            uint32_t current = heads[accumulator];
            uint64_t seen_low = 0u, seen_high = 0u;
            for (uint32_t i = 0u; i < count; ++i) {
                if (current >= pull_count) {
                    SetFailure(status, kStatusInvalidJournal); ok = false; break;
                }
                const uint32_t source = pulls[current].source_rank;
                if (source >= worker_count || SeenSource(seen_low, seen_high, source) ||
                    pulls[current].accumulator_index != accumulator ||
                    pulls[current].journal_token != token ||
                    current < source_offsets[source] ||
                    current >= source_offsets[source + 1u] ||
                    pulls[current].source_row != current - source_offsets[source]) {
                    SetFailure(status, kStatusInvalidJournal); ok = false; break;
                }
                sources[i] = source;
                MarkSource(&seen_low, &seen_high, source);
                if (pulls[current].source_row > max_row_index) {
                    SetFailure(status, kStatusSizeOverflow); ok = false; break;
                }
                row_offsets[i] = row_shift < 64u
                    ? static_cast<uint64_t>(pulls[current].source_row) << row_shift
                    : MulU64ByU32(row_bytes, pulls[current].source_row);
                const uint32_t next = pull_next[current];
                if (next != kInvalidIndex && (next <= current || next >= pull_count)) {
                    SetFailure(status, kStatusInvalidJournal); ok = false; break;
                }
                current = next;
            }
            if (!ok) break;
            if (current != kInvalidIndex) {
                SetFailure(status, kStatusInvalidJournal); ok = false; break;
            }
            // Validate every address before issuing a payload operation. A
            // source's region remains immutable until final Source ACK.
            for (uint32_t i = 0u; i < count; ++i) {
                const uint32_t source = sources[i];
                if (!SeenSource(ready_low, ready_high, source)) {
                    if (!WaitSourceReady(source_ready, source, spin_cap, status)) {
                        ok = false; break;
                    }
                    registered_offsets[source] = LoadSourcePayloadOffset(payload_offsets, source);
                    MarkSource(&ready_low, &ready_high, source);
                }
            }
            if (!ok) break;
            cached_accumulator = accumulator;
            rotation = count == 0u ? 0u : AscendC::GetBlockIdx() % count;
        }
        const uint64_t element_begin = (task % tiles_per_row) * kTileElements;
        const uint32_t elements = static_cast<uint32_t>(
            hidden - element_begin < kTileElements ? hidden - element_begin : kTileElements);
        __ubuf__ uint8_t *output_ub = output_ping == 0u ? output0_ub : output1_ub;
        AscendC::LocalTensor<float> output = IncVecBindFloatUb(output_ub, elements * sizeof(float));
        if (count == 0u) {
            AscendC::Duplicate(output, 0.0f, elements);
            AscendC::PipeBarrier<PIPE_V>();
        }
        uint32_t cursor = rotation;
        for (uint32_t done = 0u; done < count; done += 2u) {
            const uint32_t first = cursor;
            if (++cursor == count) cursor = 0u;
            const uint32_t source0 = sources[first];
            __gm__ float *remote0 = reinterpret_cast<__gm__ float *>(
                symmetric_partials + registered_offsets[source0] + row_offsets[first]);
            PullFp32ToUb(input0_ub, remote0 + element_begin, elements,
                          static_cast<int32_t>(source0), 0u);
            const bool has_second = done + 1u < count;
            if (has_second) {
                const uint32_t second = cursor;
                if (++cursor == count) cursor = 0u;
                const uint32_t source1 = sources[second];
                __gm__ float *remote1 = reinterpret_cast<__gm__ float *>(
                    symmetric_partials + registered_offsets[source1] + row_offsets[second]);
                PullFp32ToUb(input1_ub, remote1 + element_begin, elements,
                              static_cast<int32_t>(source1), 1u);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kVecPingMte2V);
            if (has_second)
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kVecPongMte2V);
            AscendC::LocalTensor<float> input0 = IncVecBindFloatUb(input0_ub, elements * sizeof(float));
            AscendC::LocalTensor<float> input1 = IncVecBindFloatUb(input1_ub, elements * sizeof(float));
            if (done == 0u) {
                if (has_second)
                    AscendC::Add(output, input0, input1, elements);
                else
                    AscendC::Adds(output, input0, 0.0f, elements);
            } else {
                AscendC::Add(output, output, input0, elements);
                if (has_second) {
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(output, output, input1, elements);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kVecPingVMte2);
            if (has_second)
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kVecPongVMte2);
        }
        // The alternate output was released by the wait before the previous
        // PUT. This wait only serializes PUT submissions, not current compute.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kVecMte3V);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kVecVMte3);
        PutFp32ToOwner(output_ub, owner_output + element_begin, elements, static_cast<int32_t>(owner));
        output_ping ^= 1u;
        if (++rotation >= count) rotation = 0u;
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kVecMte3V);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kVecPingVMte2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kVecPongVMte2);
    aclshmemx_mte_quiet();
    return ok && *status == kStatusOk;
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void inc_dc_pull_combine_v2_device_kernel(
    GM_ADDR symmetric_partials, GM_ADDR ready_records,
    GM_ADDR ready_notices, GM_ADDR ready_staging, GM_ADDR registrations,
    GM_ADDR source_acks, GM_ADDR owner_output, GM_ADDR owner_completions,
    GM_ADDR source_offsets, GM_ADDR pulls, GM_ADDR owner_offsets,
    GM_ADDR results, GM_ADDR journal_header, GM_ADDR pull_next,
    GM_ADDR accumulator_heads, GM_ADDR accumulator_contributor_counts,
    GM_ADDR accumulator_result_index, GM_ADDR source_ready_state,
    GM_ADDR source_payload_offsets, GM_ADDR status_line, uint64_t ffts_addr,
    uint64_t session_id, uint64_t placement_epoch, uint64_t generation,
    uint64_t sequence, uint64_t dispatch_cookie,
    uint64_t owner_output_slot_stride, uint64_t pull_count,
    uint64_t result_count, uint64_t pull_index_capacity,
    uint64_t accumulator_index_capacity,
    uint64_t source_scratch_capacity, uint32_t accumulator_count,
    uint32_t worker_count, uint32_t hidden, uint32_t partial_dtype,
    int32_t inc_pe, uint32_t wave, uint32_t ring_slot,
    uint32_t slot_count, uint64_t spin_cap)
{
    shmemx_set_ffts_config(ffts_addr);
    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t blocks = AscendC::GetBlockNum();
    const int32_t pe = aclshmem_my_pe();
    __gm__ CombineDeviceTimelineV2 *timeline =
        reinterpret_cast<__gm__ CombineDeviceTimelineV2 *>(status_line);
    __gm__ uint32_t *status = &timeline->status;

    uint64_t row_bytes = 0u;
    uint64_t output_slot_offset = 0u;
    uint64_t total_tasks = 0u;
    const uint64_t tiles_per_row =
        hidden == 0u ? 0u :
        (static_cast<uint64_t>(hidden) + kTileElements - 1u) /
            kTileElements;
    const bool launch_valid =
        // block_dim is derived by the host from half the live vector cores.
        // Avoid encoding one SKU's 24-AIV partition into this protocol.
        blocks != 0u && worker_count >= 2u &&
        worker_count <= kPullDispatchMaxWorkers &&
        inc_pe == static_cast<int32_t>(worker_count) &&
        session_id != 0u && placement_epoch != 0u && generation != 0u &&
        sequence != 0u && dispatch_cookie != 0u && hidden != 0u &&
        partial_dtype == static_cast<uint32_t>(PartialDataType::FP32) &&
        ring_slot < slot_count && slot_count != 0u && spin_cap != 0u &&
        pull_count <= 0xffffffffull && result_count <= 0xffffffffull &&
        result_count == accumulator_count &&
        pull_count <= pull_index_capacity &&
        accumulator_count <= accumulator_index_capacity &&
        worker_count <= source_scratch_capacity &&
        CheckedMulU64ByU32(sizeof(float), hidden, &row_bytes) &&
        CheckedMulU64ByU32(owner_output_slot_stride, ring_slot,
                           &output_slot_offset) &&
        CheckedMulU64ByU32(tiles_per_row, accumulator_count, &total_tasks);

    // Each worker publishes one cacheline notice. READY and payload stay in
    // its registered local region and are pulled by the INC.
    if (pe != inc_pe) {
        if (launch_valid && block == 0u && pe >= 0 &&
            static_cast<uint32_t>(pe) < worker_count) {
            __gm__ CombineReadyV2 *local_ready =
                reinterpret_cast<__gm__ CombineReadyV2 *>(ready_records) +
                static_cast<uint64_t>(ring_slot * worker_count) + pe;
            dcci_cacheline(
                reinterpret_cast<__gm__ uint8_t *>(local_ready));
            dcci_cacheline(
                reinterpret_cast<__gm__ uint8_t *>(local_ready) + 64u);
            AscendC::PipeBarrier<PIPE_ALL>();
            __gm__ CombineReadyNoticeV2 *local =
                reinterpret_cast<__gm__ CombineReadyNoticeV2 *>(
                    ready_notices) +
                static_cast<uint64_t>(ring_slot * worker_count) + pe;
            PublishReadyNoticeToInc(local, local, inc_pe);
        }
        return;
    }

    if (block == 0u) {
        *status = launch_valid ? kStatusOk : kStatusInvalidArgument;
        timeline->ready_sources = 0u;
        timeline->kernel_start = AscendC::GetSystemCycle();
        timeline->plan_index_done = 0u;
        timeline->first_ready = 0u;
        timeline->all_ready = 0u;
        timeline->first_get = 0u;
        timeline->last_get = 0u;
        timeline->last_reduce = 0u;
        timeline->last_owner_put = 0u;
        timeline->source_acks_done = 0u;
        timeline->owner_completions_done = 0u;
        timeline->kernel_done = 0u;
        for (uint32_t i = 0u; i < 4u; ++i) timeline->reserved[i] = 0u;

        __gm__ JournalSlotHeader *header =
            reinterpret_cast<__gm__ JournalSlotHeader *>(journal_header);
        bool journal_identity = false;
        if (launch_valid) {
            dcci_cacheline(journal_header);
            journal_identity =
                header->magic == kPullDispatchMagic &&
                header->abi_version == kPullDispatchAbiVersion &&
                header->struct_bytes == sizeof(JournalSlotHeader) &&
                header->generation == generation &&
                header->sequence == sequence &&
                header->dispatch_cookie == dispatch_cookie &&
                header->wave == wave && header->ring_slot == ring_slot &&
                header->state == static_cast<uint16_t>(
                    JournalSlotState::DISPATCH_SEALED) &&
                header->token_count == accumulator_count &&
                header->contributor_count == pull_count &&
                header->status == 0u &&
                (header->flags & ~kJournalFlagDenseAllDestinations) == 0u &&
                header->reserved[0] == 0u;
            if (!journal_identity)
                *status = kStatusInvalidJournal;
            else {
                header->state = static_cast<uint16_t>(
                    JournalSlotState::COMBINE_ACTIVE);
                dcci_cacheline(journal_header);
            }
        }

        __gm__ uint64_t *by_source =
            reinterpret_cast<__gm__ uint64_t *>(source_offsets);
        __gm__ uint64_t *by_owner =
            reinterpret_cast<__gm__ uint64_t *>(owner_offsets);
        __gm__ uint32_t *counts = reinterpret_cast<__gm__ uint32_t *>(
            accumulator_contributor_counts);
        if (*status == kStatusOk) {
            for (uint32_t source = 0u; source < worker_count; ++source) {
                __gm__ CombineRegionRegistration *registration =
                    reinterpret_cast<__gm__ CombineRegionRegistration *>(
                        registrations) + source;
                if (!RegistrationValid(registration, source, worker_count,
                                       slot_count, session_id,
                                       placement_epoch)) {
                    *status = kStatusInvalidRegistration;
                    break;
                }
                *SourceReadyAddress(source_ready_state, source) = 0u;
                *SourceAcceptedCycleAddress(source_ready_state, source) =
                    0u;
                *SourcePayloadOffsetAddress(source_payload_offsets, source) =
                    0u;
            }
        }
        if (*status == kStatusOk &&
            (by_source[0] != 0u ||
             by_source[worker_count] != pull_count ||
             by_owner[0] != 0u ||
             by_owner[worker_count] != result_count))
            *status = kStatusInvalidJournal;

        // Prove that the immutable per-accumulator chains cover every pull.
        // The detailed structural validation is distributed across all AIVs
        // after the first synchronization below.
        uint64_t indexed_pulls = 0u;
        for (uint32_t accumulator = 0u;
             accumulator < accumulator_count && *status == kStatusOk;
             ++accumulator) {
            if (counts[accumulator] > worker_count ||
                !AddU64(indexed_pulls, counts[accumulator],
                        &indexed_pulls) || indexed_pulls > pull_count)
                *status = kStatusInvalidJournal;
        }
        if (*status == kStatusOk && indexed_pulls != pull_count)
            *status = kStatusInvalidJournal;

        uint64_t maximum_owner_bytes = 0u;
        for (uint32_t owner = 0u;
             owner < worker_count && *status == kStatusOk; ++owner) {
            if (by_source[owner] > by_source[owner + 1u] ||
                by_owner[owner] > by_owner[owner + 1u]) {
                *status = kStatusInvalidJournal;
                break;
            }
            uint64_t owner_bytes = 0u;
            const uint64_t owner_rows =
                by_owner[owner + 1u] - by_owner[owner];
            if (owner_rows > 0xffffffffull ||
                !CheckedMulU64ByU32(row_bytes,
                                     static_cast<uint32_t>(owner_rows),
                                     &owner_bytes)) {
                *status = kStatusSizeOverflow;
                break;
            }
            if (owner_bytes > maximum_owner_bytes)
                maximum_owner_bytes = owner_bytes;
        }
        if (*status == kStatusOk &&
            maximum_owner_bytes > owner_output_slot_stride)
            *status = kStatusCapacityExceeded;

        AscendC::PipeBarrier<PIPE_ALL>();
        FlushRange(source_ready_state,
                   MulU64ByU32(kSourceScratchStride, worker_count));
        FlushRange(source_payload_offsets,
                   MulU64ByU32(kSourceScratchStride, worker_count));
        dcci_cacheline(status_line);
    }

    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);
    if (*status != kStatusOk) goto finalize;

    if (block == 0u) {
        timeline->plan_index_done = AscendC::GetSystemCycle();
        dcci_cacheline(status_line);
    }
    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);
    if (*status != kStatusOk) goto finalize;

    // Each source has one polling owner.  Owners with several sources scan
    // all of them every pass, so a ready high-rank source is not hidden behind
    // a late low-rank source.  Other AIVs can already reduce ready sources.
    {
        uint32_t owned = 0u;
        for (uint32_t source = block; source < worker_count;
             source += blocks)
            ++owned;
        uint32_t remaining = owned;
        const uint64_t wait_begin = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - wait_begin < spin_cap &&
               remaining != 0u && *status == kStatusOk) {
            for (uint32_t source = block; source < worker_count;
                 source += blocks) {
                __gm__ uint32_t *ready_flag =
                    SourceReadyAddress(source_ready_state, source);
                dcci_cacheline(
                    reinterpret_cast<__gm__ uint8_t *>(ready_flag));
                if (*reinterpret_cast<__gm__ volatile uint32_t *>(
                        ready_flag) != 0u)
                    continue;
                __gm__ CombineReadyNoticeV2 *notice =
                    reinterpret_cast<__gm__ CombineReadyNoticeV2 *>(
                        ready_notices) +
                    static_cast<uint64_t>(ring_slot * worker_count) +
                    source;
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(notice));
                const uint64_t observed_publication =
                    *reinterpret_cast<__gm__ volatile uint64_t *>(
                        reinterpret_cast<__gm__ uint8_t *>(notice) +
                        __builtin_offsetof(CombineReadyNoticeV2,
                                           publication));
                if (observed_publication == 0u) continue;
                AscendC::PipeBarrier<PIPE_ALL>();
                // Ignore an old notice in a reused ring slot. A notice with
                // the current identity but a bad digest fails closed.
                if (notice->generation != generation ||
                    notice->sequence != sequence || notice->wave != wave ||
                    notice->ring_slot != ring_slot)
                    continue;
                if (!ReadyNoticeValid(
                        notice, source, worker_count, slot_count, session_id,
                        placement_epoch, generation, sequence, wave,
                        static_cast<uint16_t>(ring_slot))) {
                    SetFailure(status, kStatusInvalidReady);
                    break;
                }

                __gm__ CombineReadyV2 *remote_ready =
                    reinterpret_cast<__gm__ CombineReadyV2 *>(
                        ready_records) +
                    static_cast<uint64_t>(ring_slot * worker_count) + source;
                __gm__ CombineReadyV2 *ready =
                    reinterpret_cast<__gm__ CombineReadyV2 *>(
                        ready_staging) + source;
                aclshmem_getmem(ready, remote_ready, sizeof(*ready),
                                static_cast<int32_t>(source));
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(ready));
                dcci_cacheline(
                    reinterpret_cast<__gm__ uint8_t *>(ready) + 64u);
                __gm__ CombineRegionRegistration *registration =
                    reinterpret_cast<__gm__ CombineRegionRegistration *>(
                        registrations) + source;
                const uint32_t ready_status = ValidateReady(
                    ready, registration,
                    reinterpret_cast<__gm__ uint64_t *>(source_offsets),
                    source, worker_count, hidden, slot_count, session_id,
                    placement_epoch, generation, sequence, dispatch_cookie,
                    wave, static_cast<uint16_t>(ring_slot));
                if (ready_status != kStatusOk) {
                    SetFailure(status, ready_status);
                    break;
                }
                *SourcePayloadOffsetAddress(source_payload_offsets, source) =
                    ready->source_offset;
                AscendC::PipeBarrier<PIPE_ALL>();
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
                    SourcePayloadOffsetAddress(source_payload_offsets,
                                               source)));
                *SourceAcceptedCycleAddress(source_ready_state, source) =
                    AscendC::GetSystemCycle();
                *ready_flag = 1u;
                AscendC::PipeBarrier<PIPE_ALL>();
                dcci_cacheline(
                    reinterpret_cast<__gm__ uint8_t *>(ready_flag));
                --remaining;
            }
            dcci_cacheline(status_line);
        }
        if (remaining != 0u && *status == kStatusOk)
            SetFailure(status, kStatusNotReady);
    }

    if (*status == kStatusOk) {
        const uint64_t tasks_per_block = total_tasks / blocks;
        const uint64_t extra = total_tasks % blocks;
        const uint64_t task_begin = MulU64ByU32(tasks_per_block, block) +
            (block < extra ? block : extra);
        const uint64_t task_end = task_begin + tasks_per_block +
            (block < extra ? 1u : 0u);
        ReduceContributorTaskRange(
            symmetric_partials, owner_output,
            reinterpret_cast<__gm__ CombinePullOp *>(pulls),
            reinterpret_cast<__gm__ uint32_t *>(pull_next),
            reinterpret_cast<__gm__ uint32_t *>(accumulator_heads),
            reinterpret_cast<__gm__ uint32_t *>(accumulator_contributor_counts),
            reinterpret_cast<__gm__ uint32_t *>(accumulator_result_index),
            reinterpret_cast<__gm__ CombineResultOp *>(results),
            source_ready_state, source_payload_offsets,
            reinterpret_cast<__gm__ uint64_t *>(source_offsets),
            reinterpret_cast<__gm__ uint64_t *>(owner_offsets),
            pull_count, result_count, task_begin, task_end, tiles_per_row,
            hidden, row_bytes, output_slot_offset, worker_count, spin_cap, status);
    }

finalize:
    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);
    if (block == 0u) {
        uint32_t final_status = *status;
        __gm__ JournalSlotHeader *header =
            reinterpret_cast<__gm__ JournalSlotHeader *>(journal_header);
        dcci_cacheline(journal_header);
        if (header->magic == kPullDispatchMagic &&
            header->generation == generation &&
            header->sequence == sequence &&
            header->dispatch_cookie == dispatch_cookie &&
            header->state == static_cast<uint16_t>(
                JournalSlotState::COMBINE_ACTIVE)) {
            header->status = final_status;
            header->state = static_cast<uint16_t>(
                final_status == kStatusOk ? JournalSlotState::COMPLETE :
                                           JournalSlotState::ABORTED);
            dcci_cacheline(journal_header);
        } else if (final_status == kStatusOk) {
            final_status = kStatusInvalidState;
            *status = final_status;
            dcci_cacheline(status_line);
        }

        uint32_t ready_sources = 0u;
        uint64_t first_ready = 0u;
        uint64_t all_ready = 0u;
        if (launch_valid) {
            for (uint32_t source = 0u; source < worker_count; ++source) {
                __gm__ uint32_t *ready =
                    SourceReadyAddress(source_ready_state, source);
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(ready));
                if (*reinterpret_cast<__gm__ volatile uint32_t *>(ready) !=
                    1u)
                    continue;
                const uint64_t accepted =
                    *SourceAcceptedCycleAddress(source_ready_state, source);
                ++ready_sources;
                if (first_ready == 0u || accepted < first_ready)
                    first_ready = accepted;
                if (accepted > all_ready) all_ready = accepted;
            }
        }
        timeline->ready_sources = ready_sources;
        timeline->first_ready = first_ready;
        timeline->all_ready = all_ready;
        dcci_cacheline(status_line);
    }

    // ACK and owner completion are independent per rank. Publish both records
    // on one AIV per worker instead of serializing 24 remote quiet operations
    // through block zero. The publication-last record format and fail-closed
    // status are unchanged.
    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);
    for (uint32_t rank = block; rank < worker_count; rank += blocks) {
        const uint32_t final_status = *status;
        __gm__ uint64_t *by_source =
            reinterpret_cast<__gm__ uint64_t *>(source_offsets);
        __gm__ uint64_t *by_owner =
            reinterpret_cast<__gm__ uint64_t *>(owner_offsets);
        const uint32_t source_rows = static_cast<uint32_t>(
            by_source[rank + 1u] - by_source[rank]);
        uint64_t source_bytes = 0u;
        CheckedMulU64ByU32(row_bytes, source_rows, &source_bytes);
        __gm__ CombineRegionRegistration *registration =
            reinterpret_cast<__gm__ CombineRegionRegistration *>(
                registrations) + rank;
        __gm__ CombineSourceAckV2 *ack =
            reinterpret_cast<__gm__ CombineSourceAckV2 *>(source_acks) +
            static_cast<uint64_t>(ring_slot * worker_count) + rank;
        PublishSourceAck(
            ack, rank, final_status, source_rows, source_bytes,
            registration->region_id, session_id, placement_epoch,
            generation, sequence, dispatch_cookie, wave,
            static_cast<uint16_t>(ring_slot));

        const uint32_t owner_rows = static_cast<uint32_t>(
            by_owner[rank + 1u] - by_owner[rank]);
        uint64_t owner_bytes = 0u;
        CheckedMulU64ByU32(row_bytes, owner_rows, &owner_bytes);
        __gm__ CombineOwnerCompletionV2 *completion =
            reinterpret_cast<__gm__ CombineOwnerCompletionV2 *>(
                owner_completions) +
            static_cast<uint64_t>(ring_slot * worker_count) + rank;
        PublishOwnerCompletion(
            completion, rank, final_status, owner_rows, owner_bytes,
            session_id, placement_epoch, generation, sequence,
            dispatch_cookie, wave, static_cast<uint16_t>(ring_slot));
    }
    AscendC::SyncAll<true>();
    if (block == 0u) {
        timeline->source_acks_done = AscendC::GetSystemCycle();
        timeline->owner_completions_done = timeline->source_acks_done;
        timeline->last_get = timeline->owner_completions_done;
        timeline->last_reduce = timeline->owner_completions_done;
        timeline->last_owner_put = timeline->owner_completions_done;
        timeline->kernel_done = timeline->owner_completions_done;
        FlushRange(status_line, sizeof(*timeline));
    }
}

extern "C" void launch_inc_dc_pull_combine_v2_device(
    uint32_t block_dim, void *stream, uint8_t *symmetric_partials,
    uint8_t *ready_records, uint8_t *ready_notices,
    uint8_t *ready_staging, uint8_t *registrations,
    uint8_t *source_acks, uint8_t *owner_output,
    uint8_t *owner_completions, uint8_t *source_offsets, uint8_t *pulls,
    uint8_t *owner_offsets, uint8_t *results, uint8_t *journal_header,
    uint8_t *pull_next, uint8_t *accumulator_heads,
    uint8_t *accumulator_contributor_counts,
    uint8_t *accumulator_result_index, uint8_t *source_ready_state,
    uint8_t *source_payload_offsets, uint8_t *status_line,
    uint64_t ffts_addr, uint64_t session_id, uint64_t placement_epoch,
    uint64_t generation, uint64_t sequence, uint64_t dispatch_cookie,
    uint64_t owner_output_slot_stride, uint64_t pull_count,
    uint64_t result_count, uint64_t pull_index_capacity,
    uint64_t accumulator_index_capacity,
    uint64_t source_scratch_capacity, uint32_t accumulator_count,
    uint32_t worker_count, uint32_t hidden, uint32_t partial_dtype,
    int32_t inc_pe, uint32_t wave, uint32_t ring_slot,
    uint32_t slot_count, uint64_t spin_cap)
{
    inc_dc_pull_combine_v2_device_kernel<<<block_dim, nullptr, stream>>>(
        symmetric_partials, ready_records, ready_notices, ready_staging,
        registrations, source_acks, owner_output, owner_completions,
        source_offsets, pulls, owner_offsets, results, journal_header,
        pull_next, accumulator_heads, accumulator_contributor_counts,
        accumulator_result_index, source_ready_state,
        source_payload_offsets, status_line, ffts_addr, session_id,
        placement_epoch, generation, sequence, dispatch_cookie,
        owner_output_slot_stride, pull_count, result_count,
        pull_index_capacity, accumulator_index_capacity,
        source_scratch_capacity, accumulator_count, worker_count, hidden,
        partial_dtype, inc_pe, wave, ring_slot, slot_count, spin_cap);
}
