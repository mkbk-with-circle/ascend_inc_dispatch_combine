#include "kernel_operator.h"
#include "shmem.h"

#include "inc_dc_partitioned_combine.h"
#include "inc_dc_vector_reduce_aicore.h"

using namespace inc::dc::pull_v2;

// One launch consumes exactly one origin's sealed Dispatch journal.  The
// worker payload remains in the registered symmetric layout
// [ring][origin][origin_row_capacity][hidden].  No address in this kernel is
// derived from another origin's actual row count.
namespace {

// Use one backend transport packet per input/output buffer. Four buffers
// use 64 KiB on this backend, below its 192-KiB UB limit. The old 24-KiB budget is a software
// tile budget, not this SHMEM backend's physical UB bound (ub_limit).
// The launch still uses only its assigned ordinary AIVs.
constexpr uint32_t kTileElements = inc::dc::kIncDcPrivateMtePacketBytes / sizeof(float);
constexpr uint32_t kTileBytes = kTileElements * sizeof(float);
constexpr uint64_t kHashOffset = 1469598103934665603ull;
constexpr uint64_t kSourceScratchStride =
    kPartitionedCombineSourceScratchBytes;
constexpr uint64_t kSourceAcceptedCycleOffset = 8u;
constexpr uint64_t kSourceRequiredOffset = 16u;

constexpr uint32_t kVecPingMte2V = 0u;
constexpr uint32_t kVecPingVMte2 = 1u;
constexpr uint32_t kVecPongMte2V = 2u;
constexpr uint32_t kVecPongVMte2 = 3u;
constexpr uint32_t kVecMte3V = 4u;
constexpr uint32_t kVecVMte3 = 5u;


// Numeric values deliberately match CombineV2Status.
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

__aicore__ inline uint64_t DeviceRouteKey(uint32_t owner, uint32_t row)
{
    return (static_cast<uint64_t>(owner) << 32u) | row;
}

// Keep overflow validation, but avoid optnone software 64-bit divide/multiply
// for power-of-two row strides. Non-power-of-two H retains the exact fallback.
__aicore__ inline bool CheckedRowOffset(uint64_t bytes, uint32_t row,
                                      uint32_t shift, uint64_t *offset)
{
    if (shift < 64u) {
        if (static_cast<uint64_t>(row) > (~0ull >> shift)) return false;
        *offset = static_cast<uint64_t>(row) << shift;
        return true;
    }
    return CheckedMulU64ByU32(bytes, row, offset);
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

__aicore__ inline uint64_t NoticePublication(
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

__aicore__ inline bool NoticeValid(
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
        notice->publication == NoticePublication(notice);
}

__aicore__ inline void FlushRange(__gm__ uint8_t *base, uint64_t bytes)
{
    for (uint64_t offset = 0u; offset < bytes;
         offset += kPullCombineV2Alignment)
        dcci_cacheline(base + offset);
}

__aicore__ inline __gm__ uint32_t *SourceReadyAddress(
    __gm__ uint8_t *source_state, uint32_t source)
{
    return reinterpret_cast<__gm__ uint32_t *>(
        source_state + MulU64ByU32(kSourceScratchStride, source));
}

__aicore__ inline __gm__ uint64_t *SourceAcceptedCycleAddress(
    __gm__ uint8_t *source_state, uint32_t source)
{
    return reinterpret_cast<__gm__ uint64_t *>(
        source_state + MulU64ByU32(kSourceScratchStride, source) +
        kSourceAcceptedCycleOffset);
}

__aicore__ inline __gm__ uint32_t *SourceRequiredAddress(
    __gm__ uint8_t *source_state, uint32_t source)
{
    return reinterpret_cast<__gm__ uint32_t *>(
        source_state + MulU64ByU32(kSourceScratchStride, source) +
        kSourceRequiredOffset);
}

__aicore__ inline __gm__ uint64_t *SourcePayloadOffsetAddress(
    __gm__ uint8_t *payload_offsets, uint32_t source)
{
    return reinterpret_cast<__gm__ uint64_t *>(
        payload_offsets + MulU64ByU32(kSourceScratchStride, source));
}

__aicore__ inline uint64_t LoadSourcePayloadOffset(
    __gm__ uint8_t *payload_offsets, uint32_t source)
{
    __gm__ uint64_t *offset =
        SourcePayloadOffsetAddress(payload_offsets, source);
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

__aicore__ inline bool RegistrationValid(
    __gm__ const CombineRegionRegistration *registration, uint32_t source,
    uint32_t workers, uint32_t slots, uint64_t expected_slot_stride,
    uint64_t session_id, uint64_t placement_epoch)
{
    uint64_t required = 0u;
    return source < workers && registration->session_id == session_id &&
        registration->placement_epoch == placement_epoch &&
        registration->source_rank == source && registration->region_id != 0u &&
        registration->slot_count >= slots &&
        registration->alignment >= kPullCombineV2Alignment &&
        (registration->alignment & (registration->alignment - 1u)) == 0u &&
        registration->slot_stride == expected_slot_stride &&
        registration->slot_stride % registration->alignment == 0u &&
        CheckedMulU64ByU32(registration->slot_stride,
                           registration->slot_count, &required) &&
        required <= registration->region_bytes &&
        registration->reserved[0] == 0u &&
        registration->reserved[1] == 0u;
}

// Dispatch and Combine use independent streams.  A reused slot may still
// expose the preceding origin generation when this kernel starts, so only a
// matching identity is interpreted.  DISPATCH_OPEN for the matching identity
// is a normal in-flight state; SEALED is the publication consumed below.
__aicore__ inline bool WaitForSealedJournal(
    __gm__ JournalSlotHeader *header, uint64_t generation,
    uint64_t sequence, uint64_t expected_cookie, uint32_t wave,
    uint16_t ring_slot, uint64_t spin_cap, uint64_t *resolved_cookie,
    __gm__ uint32_t *status)
{
    const uint64_t wait_begin = AscendC::GetSystemCycle();
    while (AscendC::GetSystemCycle() - wait_begin < spin_cap) {
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(header));
        const bool wave_identity =
            header->generation == generation &&
            header->sequence == sequence &&
            header->wave == wave && header->ring_slot == ring_slot;
        if (!wave_identity) continue;
        const uint16_t state = header->state;
        if (state == static_cast<uint16_t>(JournalSlotState::DISPATCH_OPEN))
            continue;
        if (state == static_cast<uint16_t>(
                JournalSlotState::DISPATCH_SEALED)) {
            if (header->magic != kPullDispatchMagic ||
                header->abi_version != kPullDispatchAbiVersion ||
                header->struct_bytes != sizeof(JournalSlotHeader) ||
                header->dispatch_cookie == 0u) {
                *status = kStatusInvalidJournal;
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(status));
                return false;
            }
            if (expected_cookie != 0u &&
                header->dispatch_cookie != expected_cookie) {
                *status = kStatusCookieMismatch;
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(status));
                return false;
            }
            *resolved_cookie = header->dispatch_cookie;
            return true;
        }
        *status = kStatusInvalidJournal;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(status));
        return false;
    }
    *status = kStatusNotReady;
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(status));
    return false;
}

__aicore__ inline uint32_t ValidateReady(
    __gm__ const CombineReadyV2 *ready,
    __gm__ const CombineRegionRegistration *registration,
    uint32_t expected_rows, uint64_t expected_offset,
    uint64_t expected_bytes, uint32_t source, uint32_t workers,
    uint32_t hidden, uint32_t slots, uint64_t session_id,
    uint64_t placement_epoch, uint64_t generation, uint64_t sequence,
    uint64_t dispatch_cookie, uint32_t wave, uint16_t ring_slot)
{
    uint64_t payload_end = 0u;
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
        ready->partial_dtype != static_cast<uint32_t>(PartialDataType::FP32) ||
        ready->reserved[0] != 0u || ready->reserved[1] != 0u ||
        ready->reserved[2] != 0u || ready->publication == 0u ||
        ready->publication != ReadyPublication(ready))
        return kStatusInvalidReady;
    if (ready->dispatch_cookie != dispatch_cookie)
        return kStatusCookieMismatch;
    if (ready->generation != generation || ready->sequence != sequence ||
        ready->wave != wave || ready->ring_slot != ring_slot)
        return kStatusStaleEpoch;
    if (!AddU64(expected_offset, expected_bytes, &payload_end))
        return kStatusSizeOverflow;
    if (ready->row_count != expected_rows ||
        ready->payload_bytes != expected_bytes ||
        ready->source_offset != expected_offset ||
        ready->source_offset % registration->alignment != 0u ||
        payload_end > registration->region_bytes)
        return kStatusCapacityExceeded;
    return kStatusOk;
}

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
    aclshmem_putmem(remote + kSecondLineOffset, local + kSecondLineOffset,
                    kPullCombineV2Alignment, remote_pe);
    aclshmem_quiet();
    *local_publication = publication;
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(local);
    dcci_cacheline(local + kSecondLineOffset);
    aclshmem_putmem(remote, local, kPullCombineV2Alignment, remote_pe);
    aclshmem_quiet();
    aclshmem_putmem(remote + kSecondLineOffset, local + kSecondLineOffset,
                    kPullCombineV2Alignment, remote_pe);
    aclshmem_quiet();
}

__aicore__ inline void PublishReadyNoticeToInc(
    __gm__ CombineReadyNoticeV2 *local, int32_t inc_pe)
{
    __gm__ uint8_t *line = reinterpret_cast<__gm__ uint8_t *>(local);
    dcci_cacheline(line);
    aclshmem_putmem(local, local, sizeof(*local), inc_pe);
    aclshmem_quiet();
}

__aicore__ inline void PublishSourceAck(
    __gm__ PartitionedCombineSourceAck *ack, uint32_t source,
    uint32_t status, uint32_t rows, uint64_t bytes, uint32_t region_id,
    uint64_t session_id, uint64_t placement_epoch, uint64_t generation,
    uint64_t sequence, uint64_t dispatch_cookie, uint32_t wave,
    uint16_t ring_slot)
{
    ack->magic = kPullCombineV2Magic;
    ack->abi_version = kPullCombineV2AbiVersion;
    ack->struct_bytes = sizeof(PartitionedCombineSourceAck);
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
        __builtin_offsetof(PartitionedCombineSourceAck, publication));
    if (publication == 0u) publication = 1u;
    PublishControlRecord128(
        reinterpret_cast<__gm__ uint8_t *>(ack),
        reinterpret_cast<__gm__ uint8_t *>(ack),
        __builtin_offsetof(PartitionedCombineSourceAck, publication),
        publication, static_cast<int32_t>(source));
}

__aicore__ inline void PublishOwnerCompletion(
    __gm__ PartitionedCombineOwnerCompletion *completion, uint32_t owner,
    uint32_t status, uint32_t rows, uint64_t bytes,
    uint64_t session_id, uint64_t placement_epoch, uint64_t generation,
    uint64_t sequence, uint64_t dispatch_cookie, uint32_t wave,
    uint16_t ring_slot)
{
    completion->magic = kPullCombineV2Magic;
    completion->abi_version = kPullCombineV2AbiVersion;
    completion->struct_bytes = sizeof(PartitionedCombineOwnerCompletion);
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
        __builtin_offsetof(PartitionedCombineOwnerCompletion, publication));
    if (publication == 0u) publication = 1u;
    PublishControlRecord128(
        reinterpret_cast<__gm__ uint8_t *>(completion),
        reinterpret_cast<__gm__ uint8_t *>(completion),
        __builtin_offsetof(PartitionedCombineOwnerCompletion, publication),
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

__aicore__ inline bool SourceReady(__gm__ uint8_t *source_state,
                                    uint32_t source)
{
    __gm__ uint32_t *state = SourceReadyAddress(source_state, source);
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(state));
    return *reinterpret_cast<__gm__ volatile uint32_t *>(state) == 1u;
}

__aicore__ inline bool WaitSourceReady(
    __gm__ uint8_t *source_state, uint32_t source, uint64_t spin_cap,
    __gm__ uint32_t *status)
{
    const uint64_t wait_begin = AscendC::GetSystemCycle();
    while (AscendC::GetSystemCycle() - wait_begin < spin_cap) {
        if (SourceReady(source_state, source)) return true;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(status));
        if (*status != kStatusOk) return false;
    }
    SetFailure(status, kStatusNotReady);
    return false;
}

// A source is accepted only by its source%blocks polling owner, so the single
// READY staging record per B never has concurrent writers.  Polling covers
// only workers that occur in this origin's real contributor journal.
__aicore__ inline bool PollRequiredSources(
    __gm__ uint8_t *ready_records, __gm__ uint8_t *ready_notices,
    __gm__ uint8_t *ready_staging,
    __gm__ CombineRegionRegistration *registrations,
    __gm__ uint32_t *destination_row_counts, __gm__ uint8_t *source_state,
    __gm__ uint8_t *payload_offsets, uint32_t block, uint32_t blocks,
    uint32_t workers, uint32_t hidden, uint32_t slots, uint32_t origin,
    uint32_t origin_row_capacity, uint64_t row_bytes,
    uint64_t expected_slot_stride, uint64_t session_id,
    uint64_t placement_epoch, uint64_t generation, uint64_t sequence,
    uint64_t dispatch_cookie, uint32_t wave, uint16_t ring_slot,
    uint64_t spin_cap, __gm__ uint32_t *status)
{
    uint32_t remaining = 0u;
    for (uint32_t source = block; source < workers; source += blocks)
        if (*SourceRequiredAddress(source_state, source) != 0u) ++remaining;
    const uint64_t wait_begin = AscendC::GetSystemCycle();
    while (remaining != 0u &&
           AscendC::GetSystemCycle() - wait_begin < spin_cap) {
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(status));
        if (*status != kStatusOk) return false;
        for (uint32_t source = block; source < workers; source += blocks) {
            if (*SourceRequiredAddress(source_state, source) == 0u ||
                SourceReady(source_state, source))
                continue;
            __gm__ CombineReadyNoticeV2 *notice =
                reinterpret_cast<__gm__ CombineReadyNoticeV2 *>(
                    ready_notices) + source;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(notice));
            const uint64_t observed =
                *reinterpret_cast<__gm__ volatile uint64_t *>(
                    reinterpret_cast<__gm__ uint8_t *>(notice) +
                    __builtin_offsetof(CombineReadyNoticeV2, publication));
            if (observed == 0u) continue;
            AscendC::PipeBarrier<PIPE_ALL>();
            // A stale record in a reused mailbox is ignored. A record with
            // this origin wave's identity but a bad digest fails closed.
            if (notice->generation != generation ||
                notice->sequence != sequence || notice->wave != wave ||
                notice->ring_slot != ring_slot)
                continue;
            if (!NoticeValid(notice, source, workers, slots, session_id,
                             placement_epoch, generation, sequence, wave,
                             ring_slot)) {
                SetFailure(status, kStatusInvalidReady);
                return false;
            }
            // dst pushed this descriptor before publishing Notice. Read the
            // INC-local symmetric copy; no remote GET of control metadata.
            __gm__ CombineReadyV2 *ready =
                reinterpret_cast<__gm__ CombineReadyV2 *>(ready_records) +
                source;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(ready));
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(ready) +
                           kPullCombineV2Alignment);
            const uint32_t rows = destination_row_counts[source];
            uint64_t expected_bytes = 0u;
            uint64_t slot_offset = 0u;
            uint64_t partition_offset = 0u;
            uint64_t expected_offset = 0u;
            if (!CheckedMulU64ByU32(row_bytes, rows, &expected_bytes) ||
                !CheckedMulU64ByU32(expected_slot_stride, ring_slot,
                                    &slot_offset) ||
                !CheckedMulU64ByU32(row_bytes, origin_row_capacity,
                                    &partition_offset) ||
                !CheckedMulU64ByU32(partition_offset, origin,
                                    &partition_offset) ||
                !AddU64(slot_offset, partition_offset, &expected_offset)) {
                SetFailure(status, kStatusSizeOverflow);
                return false;
            }
            __gm__ CombineRegionRegistration *registration =
                registrations + source;
            const uint32_t ready_status = ValidateReady(
                ready, registration, rows, expected_offset, expected_bytes,
                source, workers, hidden, slots, session_id, placement_epoch,
                generation, sequence, dispatch_cookie, wave, ring_slot);
            if (ready_status != kStatusOk) {
                SetFailure(status, ready_status);
                return false;
            }
            *SourcePayloadOffsetAddress(payload_offsets, source) =
                ready->source_offset;
            AscendC::PipeBarrier<PIPE_ALL>();
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
                SourcePayloadOffsetAddress(payload_offsets, source)));
            *SourceAcceptedCycleAddress(source_state, source) =
                AscendC::GetSystemCycle();
            *SourceReadyAddress(source_state, source) = 1u;
            AscendC::PipeBarrier<PIPE_ALL>();
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
                SourceReadyAddress(source_state, source)));
            --remaining;
        }
    }
    if (remaining != 0u) {
        SetFailure(status, kStatusNotReady);
        return false;
    }
    return true;
}

// This is the same 1536-FP32 pair-input/two-output reduction loop as Pull V2.
// Only the metadata source changes: contiguous real JournalContributor ranges
// replace a host-compiled PullPlan.
__aicore__ inline bool ReduceJournalTaskRange(
    __gm__ uint8_t *symmetric_partials, __gm__ uint8_t *owner_output_base,
    __gm__ JournalTokenEntry *tokens,
    __gm__ JournalContributor *contributors,
    __gm__ uint32_t *destination_row_counts,
    __gm__ uint8_t *source_state, __gm__ uint8_t *payload_offsets,
    uint64_t token_count, uint64_t contributor_count,
    uint64_t task_begin, uint64_t task_end, uint64_t tiles_per_row,
    uint32_t hidden, uint64_t row_bytes, uint64_t output_slot_offset,
    uint32_t worker_count, uint32_t origin, uint64_t spin_cap,
    __gm__ uint32_t *status)
{
    __ubuf__ uint8_t *input0_ub = reinterpret_cast<__ubuf__ uint8_t *>(0);
    uint32_t row_shift = 0u;
    uint64_t stride = row_bytes;
    while (stride > 1u && (stride & 1u) == 0u) { ++row_shift; stride >>= 1u; }
    if (stride != 1u) row_shift = 64u;
    static_assert(kTileBytes * 4u <= ub_limit,
                  "partitioned Combine buffers exceed backend AIV UB");
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
    uint32_t cached_token = ~0u;
    uint32_t count = 0u, rotation = 0u, output_ping = 0u;
    bool prefetched = false;
    bool prefetched_second = false;
    __gm__ float *owner_output = nullptr;
    bool ok = true;

    for (uint64_t task = task_begin; task < task_end; ++task) {
        if (((task - task_begin) & 63u) == 0u) {
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(status));
            if (*status != kStatusOk) { ok = false; break; }
        }
        const uint32_t token_index =
            static_cast<uint32_t>(task / tiles_per_row);
        if (cached_token != token_index) {
            if (token_index >= token_count) {
                SetFailure(status, kStatusInvalidJournal);
                ok = false;
                break;
            }
            __gm__ JournalTokenEntry *token = tokens + token_index;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(token));
            count = token->contributors_count;
            const uint64_t begin = token->contributors_begin;
            uint64_t end = 0u;
            if (token->route_key != DeviceRouteKey(origin, token_index) ||
                token->owner_rank != origin ||
                token->owner_row != token_index ||
                token->accumulator_index != token_index ||
                count > worker_count || !AddU64(begin, count, &end) ||
                end > contributor_count || token->flags != 0u ||
                token->reserved[0] != 0u || token->reserved[1] != 0u) {
                SetFailure(status, kStatusInvalidJournal);
                ok = false;
                break;
            }
            uint64_t owner_row_offset = 0u;
            if (!CheckedRowOffset(row_bytes, token->owner_row, row_shift,
                                    &owner_row_offset)) {
                SetFailure(status, kStatusSizeOverflow);
                ok = false;
                break;
            }
            owner_output = reinterpret_cast<__gm__ float *>(
                owner_output_base + output_slot_offset + owner_row_offset);
            uint64_t seen_low = 0u, seen_high = 0u;
            for (uint32_t i = 0u; i < count; ++i) {
                __gm__ JournalContributor *contributor =
                    contributors + begin + i;
                const uint32_t source = contributor->worker_rank;
                if (source >= worker_count) {
                    SetFailure(status, kStatusInvalidJournal);
                    ok = false;
                    break;
                }
                if (SeenSource(seen_low, seen_high, source)) {
                    SetFailure(status, kStatusDuplicateSource);
                    ok = false;
                    break;
                }
                if (contributor->destination_row >=
                    destination_row_counts[source]) {
                    SetFailure(status, kStatusInvalidJournal);
                    ok = false;
                    break;
                }
                sources[i] = source;
                MarkSource(&seen_low, &seen_high, source);
                if (!CheckedRowOffset(row_bytes,
                        contributor->destination_row, row_shift, &row_offsets[i])) {
                    SetFailure(status, kStatusSizeOverflow);
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
            for (uint32_t i = 0u; i < count; ++i) {
                const uint32_t source = sources[i];
                if (!SeenSource(ready_low, ready_high, source)) {
                    if (!WaitSourceReady(source_state, source, spin_cap,
                                         status)) {
                        ok = false;
                        break;
                    }
                    registered_offsets[source] =
                        LoadSourcePayloadOffset(payload_offsets, source);
                    MarkSource(&ready_low, &ready_high, source);
                }
            }
            if (!ok) break;
            cached_token = token_index;
            rotation = count == 0u ? 0u : AscendC::GetBlockIdx() % count;
        }

        const uint64_t element_begin =
            (task % tiles_per_row) * kTileElements;
        const uint32_t elements = static_cast<uint32_t>(
            hidden - element_begin < kTileElements ?
                hidden - element_begin : kTileElements);
        __ubuf__ uint8_t *output_ub = output_ping == 0u ? output0_ub : output1_ub;
        AscendC::LocalTensor<float> output =
            IncVecBindFloatUb(output_ub, elements * sizeof(float));
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
                symmetric_partials + registered_offsets[source0] +
                row_offsets[first]);
            const bool use_prefetch = done == 0u && prefetched;
            if (!use_prefetch)
                PullFp32ToUb(input0_ub, remote0 + element_begin, elements,
                             static_cast<int32_t>(source0), 0u);
            const bool has_second = done + 1u < count;
            if (has_second) {
                const uint32_t second = cursor;
                if (++cursor == count) cursor = 0u;
                const uint32_t source1 = sources[second];
                __gm__ float *remote1 = reinterpret_cast<__gm__ float *>(
                    symmetric_partials + registered_offsets[source1] +
                    row_offsets[second]);
                if (!use_prefetch)
                    PullFp32ToUb(input1_ub, remote1 + element_begin, elements,
                                 static_cast<int32_t>(source1), 1u);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kVecPingMte2V);
            if (has_second)
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                    kVecPongMte2V);
            if (use_prefetch) prefetched = false;
            AscendC::LocalTensor<float> input0 =
                IncVecBindFloatUb(input0_ub, elements * sizeof(float));
            AscendC::LocalTensor<float> input1 =
                IncVecBindFloatUb(input1_ub, elements * sizeof(float));
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
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                    kVecPongVMte2);
        }
        // All sources and row offsets for this token are already validated
        // and READY. Start the next tile before waiting on the output queue;
        // never speculate across a token or this AIV's task-range boundary.
        if (count != 0u && task + 1u < task_end &&
            (task + 1u) / tiles_per_row == token_index) {
            const uint64_t next_element = element_begin + kTileElements;
            const uint32_t next_elements = static_cast<uint32_t>(
                hidden - next_element < kTileElements ?
                    hidden - next_element : kTileElements);
            uint32_t next_cursor = rotation + 1u == count ? 0u : rotation + 1u;
            const uint32_t first = next_cursor;
            const uint32_t source0 = sources[first];
            __gm__ float *remote0 = reinterpret_cast<__gm__ float *>(
                symmetric_partials + registered_offsets[source0] + row_offsets[first]);
            PullFp32ToUb(input0_ub, remote0 + next_element, next_elements,
                         static_cast<int32_t>(source0), 0u);
            prefetched_second = count > 1u;
            if (prefetched_second) {
                if (++next_cursor == count) next_cursor = 0u;
                const uint32_t source1 = sources[next_cursor];
                __gm__ float *remote1 = reinterpret_cast<__gm__ float *>(
                    symmetric_partials + registered_offsets[source1] + row_offsets[next_cursor]);
                PullFp32ToUb(input1_ub, remote1 + next_element, next_elements,
                             static_cast<int32_t>(source1), 1u);
            }
            prefetched = true;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kVecMte3V);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kVecVMte3);
        PutFp32ToOwner(output_ub, owner_output + element_begin, elements,
                       static_cast<int32_t>(origin));
        output_ping ^= 1u;
        if (++rotation >= count) rotation = 0u;
    }
    // An asynchronous error can stop the next iteration after its GET was
    // issued. Consume those events before returning the input buffers.
    if (prefetched) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kVecPingMte2V);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kVecPingVMte2);
        if (prefetched_second) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kVecPongMte2V);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kVecPongVMte2);
        }
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kVecMte3V);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kVecPingVMte2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kVecPongVMte2);
    aclshmemx_mte_quiet();
    return ok && *status == kStatusOk;
}

// Independent immutable token ranges; block-zero merges all row boundaries.
__aicore__ inline void ValidateJournalRange(
    __gm__ uint8_t *journal_tokens, __gm__ uint8_t *journal_contributors,
    __gm__ uint8_t *destination_row_counts, __gm__ uint8_t *scratch,
    uint32_t token_count, uint32_t contributor_count,
    uint32_t workers, uint32_t origin, uint32_t block, uint32_t blocks)
{
        uint32_t local_status = kStatusOk;
        FlushRange(destination_row_counts, MulU64ByU32(sizeof(uint32_t), workers));
        uint32_t *status = &local_status;
        const uint32_t token_begin = static_cast<uint32_t>(MulU64ByU32(token_count, block) / blocks);
        const uint32_t token_end = static_cast<uint32_t>(MulU64ByU32(token_count, block + 1u) / blocks);
        __gm__ uint32_t *summary = reinterpret_cast<__gm__ uint32_t *>(
            scratch + MulU64ByU32(MulU64ByU32(64u, workers), block));
        {
            __gm__ JournalTokenEntry *tokens =
                reinterpret_cast<__gm__ JournalTokenEntry *>(journal_tokens);
            __gm__ JournalContributor *contributors =
                reinterpret_cast<__gm__ JournalContributor *>(
                    journal_contributors);
            __gm__ uint32_t *row_counts =
                reinterpret_cast<__gm__ uint32_t *>(destination_row_counts);
            uint32_t next_row[kPullDispatchMaxWorkers];
            uint32_t next_assignment[kPullDispatchMaxWorkers];
            for (uint32_t source = 0u; source < workers; ++source) {
                next_row[source] = 0xffffffffu;
                next_assignment[source] = 0u;
                for (uint32_t f = 0u; f < 16u; ++f) summary[source * 16u + f] = 0u;
                summary[source * 16u + 1u] = 0xffffffffu;
            }
            uint64_t contributor_cursor = 0u;
            uint64_t source_assignment_cursor = 0u;
            if (token_begin != 0u) {
                __gm__ JournalTokenEntry *previous = tokens + token_begin - 1u;
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(previous));
                if (!AddU64(previous->contributors_begin, previous->contributors_count, &contributor_cursor) ||
                    contributor_cursor > contributor_count ||
                    !AddU64(previous->assignments_begin, previous->assignments_count, &source_assignment_cursor) ||
                    source_assignment_cursor > 0xffffffffull)
                    *status = kStatusInvalidJournal;
            }
            if (contributor_cursor < contributor_count)
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(contributors + contributor_cursor));
            for (uint32_t t = token_begin;
                 t < token_end && *status == kStatusOk; ++t) {
                __gm__ JournalTokenEntry *token = tokens + t;
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(token));
                uint64_t contributor_end = 0u;
                uint64_t source_assignment_end = 0u;
                if (token->route_key != DeviceRouteKey(origin, t) ||
                    token->owner_rank != origin ||
                    token->owner_row != t || token->accumulator_index != t ||
                    token->contributors_begin != contributor_cursor ||
                    token->contributors_count > workers ||
                    !AddU64(contributor_cursor,
                            token->contributors_count, &contributor_end) ||
                    contributor_end > contributor_count ||
                    token->assignments_begin != source_assignment_cursor ||
                    !AddU64(source_assignment_cursor,
                            token->assignments_count,
                            &source_assignment_end) ||
                    source_assignment_end > 0xffffffffull ||
                    token->flags != 0u || token->reserved[0] != 0u ||
                    token->reserved[1] != 0u) {
                    *status = kStatusInvalidJournal;
                    break;
                }
                uint64_t seen_low = 0u, seen_high = 0u;
                uint64_t token_assignments = 0u;
                for (uint32_t i = 0u; i < token->contributors_count; ++i) {
                    __gm__ JournalContributor *contributor =
                        contributors + contributor_cursor + i;
                    if (((contributor_cursor + i) & 3u) == 0u)
                        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
                            contributor));
                    const uint32_t source = contributor->worker_rank;
                    uint64_t assignment_end = 0u;
                    if (source >= workers ||
                        SeenSource(seen_low, seen_high, source)) {
                        *status = source < workers ?
                            kStatusDuplicateSource : kStatusInvalidJournal;
                        break;
                    }
                    if (next_row[source] == 0xffffffffu) {
                        next_row[source] = contributor->destination_row;
                        next_assignment[source] = contributor->assignment_begin;
                        summary[source * 16u + 1u] = next_row[source];
                        summary[source * 16u + 3u] = next_assignment[source];
                    }
                    if (contributor->destination_row != next_row[source] ||
                        contributor->destination_row >= row_counts[source] ||
                        contributor->assignment_count == 0u ||
                        contributor->assignment_begin !=
                            next_assignment[source] ||
                        !AddU64(contributor->assignment_begin,
                                contributor->assignment_count,
                                &assignment_end) ||
                        assignment_end > 0xffffffffull ||
                        !AddU64(token_assignments,
                                contributor->assignment_count,
                                &token_assignments) ||
                        token_assignments > token->assignments_count) {
                        *status = kStatusInvalidJournal;
                        break;
                    }
                    MarkSource(&seen_low, &seen_high, source);
                    ++next_row[source];
                    next_assignment[source] =
                        static_cast<uint32_t>(assignment_end);

                }
                if (*status != kStatusOk) break;
                if (token_assignments != token->assignments_count) {
                    *status = kStatusInvalidJournal;
                    break;
                }
                contributor_cursor = contributor_end;
                source_assignment_cursor = source_assignment_end;
            }
            if (*status == kStatusOk && token_end == token_count &&
                contributor_cursor != contributor_count)
                *status = kStatusInvalidJournal;
            for (uint32_t source = 0u; source < workers; ++source) {
                summary[source * 16u + 2u] = next_row[source];
                summary[source * 16u + 4u] = next_assignment[source];
            }
            summary[0] = local_status;
            AscendC::PipeBarrier<PIPE_ALL>();
            FlushRange(reinterpret_cast<__gm__ uint8_t *>(summary), MulU64ByU32(64u, workers));
        }

}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void inc_dc_partitioned_combine_kernel(
    GM_ADDR symmetric_partials, GM_ADDR ready_records,
    GM_ADDR ready_notices, GM_ADDR ready_staging, GM_ADDR registrations,
    GM_ADDR source_acks, GM_ADDR owner_output, GM_ADDR owner_completions,
    GM_ADDR journal_header, GM_ADDR journal_tokens,
    GM_ADDR journal_contributors, GM_ADDR destination_row_counts,
    GM_ADDR source_state, GM_ADDR source_payload_offsets, GM_ADDR validation_scratch,
    GM_ADDR status_line, uint64_t ffts_addr, uint64_t session_id,
    uint64_t placement_epoch, uint64_t generation, uint64_t sequence,
    uint64_t dispatch_cookie, uint64_t owner_output_slot_stride,
    uint64_t journal_token_capacity,
    uint64_t journal_contributor_capacity,
    uint64_t source_scratch_capacity, uint64_t validation_scratch_capacity_bytes, uint64_t spin_cap,
    uint32_t worker_count, uint32_t hidden, uint32_t origin_rank,
    uint32_t origin_row_capacity, int32_t inc_pe, uint32_t wave,
    uint32_t ring_slot, uint32_t slot_count)
{
    shmemx_set_ffts_config(ffts_addr);
    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t blocks = AscendC::GetBlockNum();
    const int32_t pe = aclshmem_my_pe();
    __gm__ PartitionedCombineTimeline *timeline =
        reinterpret_cast<__gm__ PartitionedCombineTimeline *>(status_line);
    __gm__ uint32_t *status = &timeline->status;

    uint64_t row_bytes = 0u;
    uint64_t partition_bytes = 0u;
    uint64_t expected_slot_stride = 0u;
    uint64_t output_slot_offset = 0u;
    uint64_t effective_cookie = dispatch_cookie;
    const bool launch_valid =
        blocks != 0u && worker_count >= 2u &&
        worker_count <= kPullDispatchMaxWorkers &&
        origin_rank < worker_count &&
        inc_pe == static_cast<int32_t>(worker_count) &&
        session_id != 0u && placement_epoch != 0u && generation != 0u &&
        sequence != 0u && hidden != 0u &&
        origin_row_capacity != 0u &&
        (origin_row_capacity & 31u) == 0u && ring_slot < slot_count &&
        slot_count != 0u && spin_cap != 0u &&
        owner_output_slot_stride != 0u &&
        source_scratch_capacity >= worker_count &&
        validation_scratch != nullptr &&
        validation_scratch_capacity_bytes >= MulU64ByU32(MulU64ByU32(64u, worker_count), blocks) &&
        journal_token_capacity <= 0xffffffffull &&
        journal_contributor_capacity <= 0xffffffffull &&
        CheckedMulU64ByU32(sizeof(float), hidden, &row_bytes) &&
        CheckedMulU64ByU32(row_bytes, origin_row_capacity,
                           &partition_bytes) &&
        CheckedMulU64ByU32(partition_bytes, worker_count,
                           &expected_slot_stride) &&
        CheckedMulU64ByU32(owner_output_slot_stride, ring_slot,
                           &output_slot_offset);

    // Every worker has one origin-specific READY/notice mailbox. Publishing
    // the notice is harmless for a zero-row worker; the INC deliberately does
    // not wait for it unless the journal contains that worker.
    if (pe != inc_pe) {
        if (launch_valid && block == 0u && pe >= 0 &&
            static_cast<uint32_t>(pe) < worker_count) {
            __gm__ CombineReadyV2 *ready =
                reinterpret_cast<__gm__ CombineReadyV2 *>(ready_records) + pe;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(ready));
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(ready) +
                           kPullCombineV2Alignment);
            AscendC::PipeBarrier<PIPE_ALL>();
            // Keep the wire records and payload path unchanged. Release the
            // complete READY descriptor to INC before the 64B notification.
            aclshmem_putmem(ready, ready, sizeof(*ready), inc_pe);
            aclshmem_quiet();
            __gm__ CombineReadyNoticeV2 *notice =
                reinterpret_cast<__gm__ CombineReadyNoticeV2 *>(
                    ready_notices) + pe;
            PublishReadyNoticeToInc(notice, inc_pe);
        }
        return;
    }

    uint64_t token_count = 0u;
    uint64_t contributor_count = 0u;
    uint64_t tiles_per_row =
        hidden == 0u ? 0u :
        (static_cast<uint64_t>(hidden) + kTileElements - 1u) /
            kTileElements;
    uint64_t total_tasks = 0u;

    if (block == 0u) {
        *status = launch_valid ? kStatusOk : kStatusInvalidArgument;
        timeline->ready_sources = 0u;
        timeline->kernel_start = AscendC::GetSystemCycle();
        timeline->journal_validated = 0u;
        timeline->first_ready = 0u;
        timeline->all_required_ready = 0u;
        timeline->first_get = 0u;
        timeline->last_get = 0u;
        timeline->last_reduce = 0u;
        timeline->last_owner_put = 0u;
        timeline->source_acks_done = 0u;
        timeline->owner_completion_done = 0u;
        timeline->kernel_done = 0u;
        for (uint32_t i = 0u; i < 4u; ++i) timeline->reserved[i] = 0u;

        if (launch_valid) {
            for (uint32_t source = 0u; source < worker_count; ++source) {
                *SourceReadyAddress(source_state, source) = 0u;
                *SourceAcceptedCycleAddress(source_state, source) = 0u;
                *SourceRequiredAddress(source_state, source) = 0u;
                *SourcePayloadOffsetAddress(source_payload_offsets, source) =
                    0u;
            }
        }

        __gm__ JournalSlotHeader *header =
            reinterpret_cast<__gm__ JournalSlotHeader *>(journal_header);
        if (*status == kStatusOk) {
            WaitForSealedJournal(
                header, generation, sequence, dispatch_cookie, wave,
                static_cast<uint16_t>(ring_slot), spin_cap,
                &effective_cookie, status);
        }
        if (*status == kStatusOk) {
            header->state = static_cast<uint16_t>(
                JournalSlotState::COMBINE_ACTIVE);
            dcci_cacheline(journal_header);
            if (header->status != 0u ||
                (header->flags & ~kJournalFlagDenseAllDestinations) != 0u ||
                header->reserved[0] != 0u ||
                header->token_count > journal_token_capacity ||
                header->contributor_count > journal_contributor_capacity) {
                *status = kStatusInvalidJournal;
            } else {
                token_count = header->token_count;
                contributor_count = header->contributor_count;
                uint64_t owner_bytes = 0u;
                if (!CheckedMulU64ByU32(row_bytes, header->token_count,
                                        &owner_bytes) ||
                    owner_bytes > owner_output_slot_stride ||
                    !CheckedMulU64ByU32(tiles_per_row,
                                        header->token_count,
                                        &total_tasks)) {
                    *status = owner_bytes > owner_output_slot_stride ?
                        kStatusCapacityExceeded : kStatusSizeOverflow;
                }
            }
        }

        if (*status == kStatusOk) {
            for (uint32_t source = 0u; source < worker_count; ++source) {
                __gm__ CombineRegionRegistration *registration =
                    reinterpret_cast<__gm__ CombineRegionRegistration *>(
                        registrations) + source;
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
                    registration));
                if (!RegistrationValid(
                        registration, source, worker_count, slot_count,
                        expected_slot_stride, session_id, placement_epoch)) {
                    *status = kStatusInvalidRegistration;
                    break;
                }
                dcci_cacheline(destination_row_counts +
                               static_cast<uint64_t>(source) *
                                   sizeof(uint32_t));
                if (reinterpret_cast<__gm__ uint32_t *>(
                        destination_row_counts)[source] >
                    origin_row_capacity) {
                    *status = kStatusCapacityExceeded;
                    break;
                }
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        FlushRange(source_state,
                   MulU64ByU32(kSourceScratchStride, worker_count));
        FlushRange(source_payload_offsets,
                   MulU64ByU32(kSourceScratchStride, worker_count));
        timeline->journal_validated = AscendC::GetSystemCycle();
        dcci_cacheline(status_line);
    }

    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);
    if (*status != kStatusOk) goto finalize;

    // Each AIV validates its immutable token range; then merge dense row and
    // assignment ranges before accepting any source Notice or pulling payload.
    {
        __gm__ JournalSlotHeader *header = reinterpret_cast<__gm__ JournalSlotHeader *>(journal_header);
        dcci_cacheline(journal_header);
        ValidateJournalRange(journal_tokens, journal_contributors,
            destination_row_counts, validation_scratch, header->token_count,
            header->contributor_count, worker_count, origin_rank, block, blocks);
    }
    AscendC::SyncAll<true>();
    if (block == 0u) {
        for (uint32_t source = 0u; source < worker_count; ++source) {
            uint32_t next_row = 0u, next_assignment = 0u;
            for (uint32_t lane = 0u; lane < blocks; ++lane) {
                __gm__ uint32_t *summary = reinterpret_cast<__gm__ uint32_t *>(
                    validation_scratch + MulU64ByU32(MulU64ByU32(64u, worker_count), lane) +
                        MulU64ByU32(64u, source));
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(summary));
                if (source == 0u && summary[0] != kStatusOk) *status = summary[0];
                if (summary[1] == 0xffffffffu) continue;
                if (summary[1] != next_row || summary[3] != next_assignment ||
                    summary[2] < summary[1] || summary[4] < summary[3])
                    *status = kStatusInvalidJournal;
                next_row = summary[2];
                next_assignment = summary[4];
            }
            if (next_row != reinterpret_cast<__gm__ uint32_t *>(destination_row_counts)[source])
                *status = kStatusInvalidJournal;
            *SourceRequiredAddress(source_state, source) = next_row != 0u ? 1u : 0u;
        }
        FlushRange(source_state, MulU64ByU32(kSourceScratchStride, worker_count));
        timeline->journal_validated = AscendC::GetSystemCycle();
        dcci_cacheline(status_line);
    }
    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);
    if (*status != kStatusOk) goto finalize;

    // Reload sealed counts on every AIV after validation. They
    // are immutable until this origin journal reaches a terminal state.
    {
        __gm__ JournalSlotHeader *header =
            reinterpret_cast<__gm__ JournalSlotHeader *>(journal_header);
        dcci_cacheline(journal_header);
        token_count = header->token_count;
        contributor_count = header->contributor_count;
        effective_cookie = header->dispatch_cookie;
        if (!CheckedMulU64ByU32(tiles_per_row, header->token_count,
                                &total_tasks)) {
            SetFailure(status, kStatusSizeOverflow);
            goto finalize;
        }
    }

    PollRequiredSources(
        ready_records, ready_notices, ready_staging,
        reinterpret_cast<__gm__ CombineRegionRegistration *>(registrations),
        reinterpret_cast<__gm__ uint32_t *>(destination_row_counts),
        source_state, source_payload_offsets, block, blocks, worker_count,
        hidden, slot_count, origin_rank, origin_row_capacity, row_bytes,
        expected_slot_stride, session_id, placement_epoch, generation,
        sequence, effective_cookie, wave, static_cast<uint16_t>(ring_slot),
        spin_cap, status);

    dcci_cacheline(status_line);
    if (*status == kStatusOk) {
        const uint64_t tasks_per_block = total_tasks / blocks;
        const uint64_t extra = total_tasks % blocks;
        const uint64_t task_begin = MulU64ByU32(tasks_per_block, block) +
            (block < extra ? block : extra);
        const uint64_t task_end = task_begin + tasks_per_block +
            (block < extra ? 1u : 0u);
        ReduceJournalTaskRange(
            symmetric_partials, owner_output,
            reinterpret_cast<__gm__ JournalTokenEntry *>(journal_tokens),
            reinterpret_cast<__gm__ JournalContributor *>(
                journal_contributors),
            reinterpret_cast<__gm__ uint32_t *>(destination_row_counts),
            source_state, source_payload_offsets, token_count,
            contributor_count, task_begin, task_end, tiles_per_row, hidden,
            row_bytes, output_slot_offset, worker_count, origin_rank,
            spin_cap, status);
    }

finalize:
    // Every successful payload PUT is quiet before the local grid joins and
    // block zero commits this origin journal. There is no other-origin join.
    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);
    if (effective_cookie == 0u && launch_valid) {
        __gm__ JournalSlotHeader *cookie_header =
            reinterpret_cast<__gm__ JournalSlotHeader *>(journal_header);
        dcci_cacheline(journal_header);
        if (cookie_header->generation == generation &&
            cookie_header->sequence == sequence &&
            cookie_header->wave == wave &&
            cookie_header->ring_slot == ring_slot)
            effective_cookie = cookie_header->dispatch_cookie;
    }
    if (block == 0u) {
        uint32_t final_status = *status;
        __gm__ JournalSlotHeader *header =
            reinterpret_cast<__gm__ JournalSlotHeader *>(journal_header);
        if (launch_valid) {
            dcci_cacheline(journal_header);
            if (header->magic == kPullDispatchMagic &&
                header->generation == generation &&
                header->sequence == sequence &&
                header->dispatch_cookie == effective_cookie &&
                header->state == static_cast<uint16_t>(
                    JournalSlotState::COMBINE_ACTIVE)) {
                header->status = final_status;
                header->state = static_cast<uint16_t>(
                    final_status == kStatusOk ? JournalSlotState::COMPLETE :
                                               JournalSlotState::ABORTED);
                AscendC::PipeBarrier<PIPE_ALL>();
                dcci_cacheline(journal_header);
            } else if (final_status == kStatusOk) {
                final_status = kStatusInvalidState;
                *status = final_status;
                dcci_cacheline(status_line);
            }
        }

        uint32_t ready_sources = 0u;
        uint64_t first_ready = 0u;
        uint64_t all_ready = 0u;
        if (launch_valid) {
            for (uint32_t source = 0u; source < worker_count; ++source) {
                if (*SourceRequiredAddress(source_state, source) == 0u)
                    continue;
                __gm__ uint32_t *ready =
                    SourceReadyAddress(source_state, source);
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(ready));
                if (*reinterpret_cast<__gm__ volatile uint32_t *>(ready) !=
                    1u)
                    continue;
                const uint64_t accepted =
                    *SourceAcceptedCycleAddress(source_state, source);
                ++ready_sources;
                if (first_ready == 0u || accepted < first_ready)
                    first_ready = accepted;
                if (accepted > all_ready) all_ready = accepted;
            }
        }
        const uint64_t joined = AscendC::GetSystemCycle();
        timeline->ready_sources = ready_sources;
        timeline->first_ready = first_ready;
        timeline->all_required_ready = all_ready;
        timeline->last_get = joined;
        timeline->last_reduce = joined;
        timeline->last_owner_put = joined;
        dcci_cacheline(status_line);
    }

    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);
    if (launch_valid) {
        for (uint32_t source = block; source < worker_count;
             source += blocks) {
            const uint32_t final_status = *status;
            __gm__ uint32_t *row_counts =
                reinterpret_cast<__gm__ uint32_t *>(destination_row_counts);
            const uint32_t rows = row_counts[source];
            uint64_t bytes = 0u;
            if (!CheckedMulU64ByU32(row_bytes, rows, &bytes)) bytes = 0u;
            __gm__ CombineRegionRegistration *registration =
                reinterpret_cast<__gm__ CombineRegionRegistration *>(
                    registrations) + source;
            __gm__ PartitionedCombineSourceAck *ack =
                reinterpret_cast<__gm__ PartitionedCombineSourceAck *>(
                    source_acks) + source;
            PublishSourceAck(
                ack, source, final_status, rows, bytes,
                registration->region_id, session_id, placement_epoch,
                generation, sequence, effective_cookie, wave,
                static_cast<uint16_t>(ring_slot));
        }
        if (block == 0u) {
            __gm__ JournalSlotHeader *header =
                reinterpret_cast<__gm__ JournalSlotHeader *>(journal_header);
            const uint32_t rows = header->token_count;
            uint64_t bytes = 0u;
            if (!CheckedMulU64ByU32(row_bytes, rows, &bytes)) bytes = 0u;
            __gm__ PartitionedCombineOwnerCompletion *completion =
                reinterpret_cast<__gm__ PartitionedCombineOwnerCompletion *>(
                    owner_completions) + origin_rank;
            PublishOwnerCompletion(
                completion, origin_rank, *status, rows, bytes, session_id,
                placement_epoch, generation, sequence, effective_cookie, wave,
                static_cast<uint16_t>(ring_slot));
        }
    }
    AscendC::SyncAll<true>();
    if (block == 0u) {
        const uint64_t done = AscendC::GetSystemCycle();
        timeline->source_acks_done = done;
        timeline->owner_completion_done = done;
        timeline->kernel_done = done;
        FlushRange(status_line, sizeof(*timeline));
    }
}

extern "C" void launch_inc_dc_partitioned_combine(
    uint32_t block_dim, void *stream,
    const PartitionedCombineLaunchArgs *args)
{
    if (args == nullptr) return;
    inc_dc_partitioned_combine_kernel<<<block_dim, nullptr, stream>>>(
        args->symmetric_partials, args->ready_records,
        args->ready_notices, args->ready_staging, args->registrations,
        args->source_acks, args->owner_output, args->owner_completions,
        args->journal_header, args->journal_tokens,
        args->journal_contributors, args->destination_row_counts,
        args->source_state, args->source_payload_offsets, args->validation_scratch, args->status_line,
        args->ffts_addr, args->session_id, args->placement_epoch,
        args->generation, args->sequence, args->dispatch_cookie,
        args->owner_output_slot_stride, args->journal_token_capacity,
        args->journal_contributor_capacity, args->source_scratch_capacity, args->validation_scratch_capacity_bytes,
        args->spin_cap, args->worker_count, args->hidden, args->origin_rank,
        args->origin_row_capacity, args->inc_pe, args->wave,
        args->ring_slot, args->slot_count);
}
