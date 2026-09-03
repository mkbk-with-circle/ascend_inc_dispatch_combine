#include "kernel_operator.h"
#include "shmem.h"

#include "inc_dc_endpoint_dispatch_abi.h"
#include "inc_dc_device_journal_abi.h"

using namespace inc::dc::pull_combine;

namespace {

constexpr uint32_t kStatusOk = 0u;
constexpr uint32_t kStatusCommitTimeout = 1u;
constexpr uint32_t kStatusInvalidCommit = 2u;
constexpr uint32_t kStatusInvalidPacket = 3u;
constexpr uint64_t kCommitSpinLimit = 1000000000ull;
constexpr uint32_t kLocalPackUbBytes = 16u * 1024u;
constexpr uint64_t kFnvOffset = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

__aicore__ inline uint32_t DTypeBytes(uint32_t dtype)
{
    return dtype == static_cast<uint32_t>(EndpointDataType::FP32) ? 4u : 2u;
}

__aicore__ inline bool DTypeValid(uint32_t dtype)
{
    return dtype == static_cast<uint32_t>(EndpointDataType::FP16) ||
        dtype == static_cast<uint32_t>(EndpointDataType::BF16) ||
        dtype == static_cast<uint32_t>(EndpointDataType::FP32);
}

__aicore__ inline void FlushRange(__gm__ uint8_t *pointer, uint64_t bytes)
{
    AscendC::PipeBarrier<PIPE_ALL>();
    for (uint64_t offset = 0u; offset < bytes; offset += 64u)
        dcci_cacheline(pointer + offset);
    AscendC::PipeBarrier<PIPE_ALL>();
}

// Exact local GM->GM copy used while packing destination tiles.  Calling the
// public SHMEM GET path for the INC's own PE performs address translation and
// transport selection for every token; this direct MTE path keeps those
// operations out of the per-token loop while retaining bounded UB usage.
__aicore__ inline void CopyLocalGm(__gm__ uint8_t *destination,
                                   __gm__ uint8_t *source, uint64_t bytes)
{
    __ubuf__ uint8_t *ub = reinterpret_cast<__ubuf__ uint8_t *>(0u);
    uint64_t offset = 0u;
    while (offset < bytes) {
        const uint32_t chunk = static_cast<uint32_t>(
            bytes - offset < kLocalPackUbBytes
                ? bytes - offset : kLocalPackUbBytes);
        aclshmemi_copy_gm2ub(ub, source + offset, chunk);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0u);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0u);
        aclshmemi_copy_ub2gm(destination + offset, ub, chunk);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
        offset += chunk;
    }
}

__aicore__ inline bool HeaderValid(
    __gm__ const EndpointDispatchPacketHeader *header, uint32_t source,
    uint32_t worker_count, uint32_t hidden, uint32_t dtype,
    uint64_t generation, uint32_t wave, uint64_t slot_bytes)
{
    const uint64_t token_bytes = static_cast<uint64_t>(header->token_count) *
        sizeof(EndpointDispatchTokenRecord);
    const uint64_t assignment_bytes =
        static_cast<uint64_t>(header->assignment_count) *
        sizeof(EndpointDispatchAssignmentRecord);
    const uint64_t hidden_elements =
        static_cast<uint64_t>(header->token_count) * hidden;
    const uint32_t dtype_bytes = DTypeBytes(dtype);
    const uint64_t max_hidden_elements =
        dtype_bytes == 4u ? 0x3fffffffffffffffull
                          : 0x7fffffffffffffffull;
    if (worker_count < 2u ||
        worker_count > kEndpointDispatchMaxWorkers ||
        !DTypeValid(dtype) ||
        hidden_elements > max_hidden_elements)
        return false;
    const uint64_t hidden_bytes = hidden_elements * dtype_bytes;
    return header->magic == kEndpointDispatchMagic &&
        header->abi_version == kEndpointDispatchAbiVersion &&
        header->header_bytes == sizeof(EndpointDispatchPacketHeader) &&
        header->generation == generation && header->sequence == 1u &&
        header->wave == wave && header->source_rank == source &&
        header->worker_count == worker_count && header->hidden == hidden &&
        header->dtype == dtype && header->flags == 0u &&
        header->packet_bytes != 0u && header->packet_bytes <= slot_bytes &&
        header->packet_bytes % kEndpointDispatchAlignment == 0u &&
        header->counts_offset == sizeof(EndpointDispatchPacketHeader) &&
        header->tokens_offset % kEndpointDispatchAlignment == 0u &&
        header->assignments_offset % kEndpointDispatchAlignment == 0u &&
        header->hidden_offset % kEndpointDispatchAlignment == 0u &&
        header->tokens_offset >= header->counts_offset &&
        static_cast<uint64_t>(worker_count) * sizeof(uint32_t) * 2u <=
            header->tokens_offset - header->counts_offset &&
        header->assignments_offset >= header->tokens_offset &&
        token_bytes <= header->assignments_offset - header->tokens_offset &&
        header->hidden_offset >= header->assignments_offset &&
        assignment_bytes <=
            header->hidden_offset - header->assignments_offset &&
        header->hidden_offset <= header->packet_bytes &&
        hidden_bytes <= header->packet_bytes - header->hidden_offset &&
        header->reserved[0] == 0u && header->reserved[1] == 0u &&
        header->reserved[2] == 0u;
}

__aicore__ inline uint64_t HashRange(__gm__ const uint8_t *data,
                                     uint64_t bytes, uint64_t hash)
{
    for (uint64_t i = 0u; i < bytes; ++i) {
        hash ^= data[i];
        hash *= kFnvPrime;
    }
    return hash;
}

__aicore__ inline uint64_t MetadataDigest(
    __gm__ uint8_t *packet, __gm__ EndpointDispatchPacketHeader *header)
{
    uint64_t hash = kFnvOffset;
    constexpr uint64_t digest_offset =
        __builtin_offsetof(EndpointDispatchPacketHeader, metadata_digest);
    hash = HashRange(packet, digest_offset, hash);
    for (uint32_t i = 0u; i < sizeof(uint64_t); ++i) {
        hash ^= 0u;
        hash *= kFnvPrime;
    }
    hash = HashRange(packet + digest_offset + sizeof(uint64_t),
                     sizeof(EndpointDispatchPacketHeader) - digest_offset -
                         sizeof(uint64_t),
                     hash);
    hash = HashRange(packet + header->counts_offset,
                     static_cast<uint64_t>(header->worker_count) *
                         sizeof(uint32_t) * 2u,
                     hash);
    hash = HashRange(packet + header->tokens_offset,
                     static_cast<uint64_t>(header->token_count) *
                         sizeof(EndpointDispatchTokenRecord),
                     hash);
    return HashRange(packet + header->assignments_offset,
                     static_cast<uint64_t>(header->assignment_count) *
                         sizeof(EndpointDispatchAssignmentRecord),
                     hash);
}

__aicore__ inline bool WeightFinite(
    __gm__ EndpointDispatchAssignmentRecord *assignment)
{
    const uint32_t bits =
        reinterpret_cast<__gm__ uint32_t *>(assignment)[3];
    return (bits & 0x7f800000u) != 0x7f800000u;
}

__aicore__ inline uint32_t ReadCount(
    __gm__ uint8_t *packet, uint64_t counts_offset, uint32_t index);

__aicore__ inline bool ValidatePacketMetadata(
    __gm__ uint8_t *packet, __gm__ EndpointDispatchPacketHeader *header,
    uint32_t expert_count, __gm__ uint32_t *cursor, uint32_t source)
{
    const uint32_t workers = header->worker_count;
    for (uint32_t destination = 0u; destination < workers; ++destination) {
        __gm__ uint32_t *slot = cursor +
            (static_cast<uint64_t>(source) * workers + destination) * 16u;
        slot[8] = 0u;
        slot[9] = 0u;
    }
    uint32_t expected_assignment_begin = 0u;
    for (uint32_t token = 0u; token < header->token_count; ++token) {
        __gm__ EndpointDispatchTokenRecord *record =
            reinterpret_cast<__gm__ EndpointDispatchTokenRecord *>(
                packet + header->tokens_offset) + token;
        if (record->source_token != token ||
            record->assignment_begin != expected_assignment_begin ||
            record->assignment_count >
                header->assignment_count - expected_assignment_begin ||
            record->reserved0 != 0u || record->reserved1 != 0u)
            return false;
        uint64_t seen[2]{0u, 0u};
        for (uint32_t local = 0u; local < record->assignment_count; ++local) {
            __gm__ EndpointDispatchAssignmentRecord *assignment =
                reinterpret_cast<__gm__ EndpointDispatchAssignmentRecord *>(
                    packet + header->assignments_offset) +
                record->assignment_begin + local;
            if (assignment->destination_rank >= workers ||
                assignment->expert_id >= expert_count ||
                !WeightFinite(assignment))
                return false;
            for (uint32_t prior = 0u; prior < local; ++prior) {
                __gm__ EndpointDispatchAssignmentRecord *other =
                    reinterpret_cast<
                        __gm__ EndpointDispatchAssignmentRecord *>(
                            packet + header->assignments_offset) +
                    record->assignment_begin + prior;
                if (other->ordinal == assignment->ordinal) return false;
            }
            const uint32_t destination = assignment->destination_rank;
            __gm__ uint32_t *slot = cursor +
                (static_cast<uint64_t>(source) * workers + destination) *
                    16u;
            ++slot[9];
            const uint32_t word = destination >> 6u;
            const uint64_t bit = 1ull << (destination & 63u);
            if ((seen[word] & bit) == 0u) {
                seen[word] |= bit;
                ++slot[8];
            }
        }
        expected_assignment_begin += record->assignment_count;
    }
    if (expected_assignment_begin != header->assignment_count) return false;
    for (uint32_t destination = 0u; destination < workers; ++destination) {
        __gm__ uint32_t *slot = cursor +
            (static_cast<uint64_t>(source) * workers + destination) * 16u;
        if (slot[8] != ReadCount(packet, header->counts_offset,
                                 destination) ||
            slot[9] != ReadCount(packet, header->counts_offset,
                                 workers + destination))
            return false;
    }
    return true;
}

__aicore__ inline uint32_t ReadCount(
    __gm__ uint8_t *packet, uint64_t counts_offset, uint32_t index)
{
    __gm__ uint32_t *value = reinterpret_cast<__gm__ uint32_t *>(
        packet + counts_offset) + index;
    return *value;
}

__aicore__ inline void PublishAck(
    __gm__ EndpointDispatchAck *ack, uint32_t source, uint32_t status,
    uint64_t tokens_consumed, uint64_t generation)
{
    const int32_t source_pe = static_cast<int32_t>(source);
    aclshmem_uint32_p(&ack->magic, kEndpointDispatchMagic, source_pe);
    aclshmem_uint16_p(&ack->abi_version, kEndpointDispatchAbiVersion,
                      source_pe);
    aclshmem_uint16_p(&ack->struct_bytes, sizeof(EndpointDispatchAck),
                      source_pe);
    aclshmem_uint64_p(&ack->sequence, 1u, source_pe);
    aclshmem_uint32_p(&ack->source_rank, source, source_pe);
    aclshmem_uint32_p(&ack->status, status, source_pe);
    aclshmem_uint64_p(&ack->tokens_consumed, tokens_consumed, source_pe);
    aclshmem_uint64_p(&ack->reserved[0], 0u, source_pe);
    aclshmem_uint64_p(&ack->reserved[1], 0u, source_pe);
    aclshmem_uint64_p(&ack->reserved[2], 0u, source_pe);
    aclshmem_quiet();
    // Generation is the publication word and is written last.
    aclshmem_uint64_p(&ack->generation, generation, source_pe);
    aclshmem_quiet();
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void inc_dc_endpoint_dispatch_device_e2e_kernel(
    GM_ADDR source_packet, GM_ADDR inc_packets, GM_ADDR commit_mailbox,
    GM_ADDR recv_hidden, GM_ADDR recv_rows, GM_ADDR recv_assignments,
    GM_ADDR recv_counts, GM_ADDR ack_mailbox, GM_ADDR completion_mailbox,
    GM_ADDR cursors, GM_ADDR hidden_staging, GM_ADDR journal_header,
    GM_ADDR status_line,
    uint64_t ffts_addr,
    uint64_t slot_bytes,
    uint64_t staging_bytes_per_destination,
    uint64_t row_capacity, uint64_t assignment_capacity,
    uint32_t hidden,
    uint32_t dtype, uint32_t expert_count, uint32_t worker_count,
    int32_t inc_pe, uint64_t generation, uint32_t wave)
{
    shmemx_set_ffts_config(ffts_addr);
    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t blocks = AscendC::GetBlockNum();
    const int32_t pe = aclshmem_my_pe();
    __gm__ uint32_t *global_status =
        reinterpret_cast<__gm__ uint32_t *>(status_line);

    if (worker_count < 2u || worker_count > kEndpointDispatchMaxWorkers ||
        slot_bytes < sizeof(EndpointDispatchPacketHeader) ||
        slot_bytes % kEndpointDispatchAlignment != 0u ||
        hidden == 0u || !DTypeValid(dtype)) {
        if (pe == inc_pe && block == 0u) {
            *global_status = kStatusInvalidCommit;
            dcci_cacheline(status_line);
        }
        return;
    }

    if (pe != inc_pe) {
        __gm__ EndpointDispatchPacketHeader *local_header =
            reinterpret_cast<__gm__ EndpointDispatchPacketHeader *>(
                source_packet);
        FlushRange(source_packet, sizeof(EndpointDispatchPacketHeader));
        // Never trust a caller-owned packet length before the INC parser has
        // validated it.  A corrupt length still publishes its commit and is
        // rejected remotely, but cannot make the sender read beyond its slot.
        const bool local_size_valid = local_header->packet_bytes >=
                sizeof(EndpointDispatchPacketHeader) &&
            local_header->packet_bytes <= slot_bytes &&
            local_header->packet_bytes % kEndpointDispatchAlignment == 0u;
        const uint64_t bytes_to_copy = local_size_valid
            ? local_header->packet_bytes
            : sizeof(EndpointDispatchPacketHeader);
        const uint64_t cache_lines = bytes_to_copy / 64u;
        const uint64_t first_line = cache_lines * block / blocks;
        const uint64_t last_line = cache_lines * (block + 1u) / blocks;
        if (first_line < last_line) {
            aclshmem_putmem(
                inc_packets + static_cast<uint64_t>(pe) * slot_bytes +
                    first_line * 64u,
                source_packet + first_line * 64u,
                (last_line - first_line) * 64u, inc_pe);
        }
        AscendC::SyncAll<true>();
        if (block == 0u) {
            __gm__ EndpointDispatchCommit *local_commit =
                reinterpret_cast<__gm__ EndpointDispatchCommit *>(
                    commit_mailbox) + pe;
            aclshmem_putmem(local_commit, local_commit,
                            sizeof(EndpointDispatchCommit), inc_pe);
        }
        return;
    }

    if (block == 0u) {
        *global_status = kStatusOk;
        dcci_cacheline(status_line);
        bool acquired[kEndpointDispatchMaxWorkers]{};
        uint32_t remaining = worker_count;
        // Poll every uncommitted source instead of blocking on rank order.  A
        // late rank therefore cannot prevent validation of packets that are
        // already visible, and the same loop works for arbitrary arrival
        // timing without maintaining source-order state.
        for (uint64_t spin = 0u;
             spin < kCommitSpinLimit && remaining != 0u &&
                 *global_status == kStatusOk;
             ++spin) {
            for (uint32_t source = 0u; source < worker_count; ++source) {
                if (acquired[source]) continue;
                __gm__ EndpointDispatchCommit *commit =
                    reinterpret_cast<__gm__ EndpointDispatchCommit *>(
                        commit_mailbox) + source;
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(commit));
                if (commit->generation != generation ||
                    commit->wave != wave)
                    continue;
                if (commit->magic != kEndpointDispatchMagic ||
                    commit->abi_version != kEndpointDispatchAbiVersion ||
                    commit->struct_bytes != sizeof(EndpointDispatchCommit) ||
                    commit->source_rank != source ||
                    commit->sequence != 1u || commit->slot != 0u ||
                    commit->flags != 0u || commit->packet_bytes == 0u ||
                    commit->packet_bytes > slot_bytes ||
                    commit->reserved != 0u) {
                    *global_status = kStatusInvalidCommit;
                    break;
                }
                __gm__ uint8_t *packet = inc_packets +
                    static_cast<uint64_t>(source) * slot_bytes;
                FlushRange(packet, sizeof(EndpointDispatchPacketHeader));
                __gm__ EndpointDispatchPacketHeader *header =
                    reinterpret_cast<__gm__ EndpointDispatchPacketHeader *>(
                        packet);
                if (!HeaderValid(header, source, worker_count, hidden, dtype,
                                 generation, wave, slot_bytes) ||
                    header->packet_bytes != commit->packet_bytes ||
                    header->metadata_digest != commit->metadata_digest) {
                    *global_status = kStatusInvalidPacket;
                    break;
                }
                // Commit publication makes the packet immutable.  Invalidate
                // compact metadata once rather than in each destination loop.
                FlushRange(packet, header->hidden_offset);
                acquired[source] = true;
                --remaining;
            }
        }
        if (remaining != 0u && *global_status == kStatusOk)
            *global_status = kStatusCommitTimeout;
        AscendC::PipeBarrier<PIPE_ALL>();
        dcci_cacheline(status_line);
    }
    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);

    __gm__ uint32_t *cursor = reinterpret_cast<__gm__ uint32_t *>(cursors);
    const uint64_t row_bytes = static_cast<uint64_t>(hidden) *
        DTypeBytes(dtype);

    if (*global_status == kStatusOk) {
        for (uint32_t source = block; source < worker_count;
             source += blocks) {
            __gm__ uint8_t *packet =
                inc_packets + static_cast<uint64_t>(source) * slot_bytes;
            __gm__ EndpointDispatchPacketHeader *header =
                reinterpret_cast<__gm__ EndpointDispatchPacketHeader *>(
                    packet);
            if (MetadataDigest(packet, header) != header->metadata_digest ||
                !ValidatePacketMetadata(packet, header, expert_count, cursor,
                                        source)) {
                *global_status = kStatusInvalidPacket;
                AscendC::PipeBarrier<PIPE_ALL>();
                dcci_cacheline(status_line);
                break;
            }
        }
    }
    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);

    if (block == 0u && *global_status == kStatusOk) {
        // Count transpose and source-major receive offsets.  The complete
        // reply is published before any fan-out payload for that destination.
        for (uint32_t destination = 0u; destination < worker_count;
             ++destination) {
            uint32_t row_offset = 0u;
            uint32_t assignment_offset = 0u;
            for (uint32_t source = 0u; source < worker_count; ++source) {
                __gm__ uint8_t *packet =
                    inc_packets + static_cast<uint64_t>(source) * slot_bytes;
                __gm__ EndpointDispatchPacketHeader *header =
                    reinterpret_cast<__gm__ EndpointDispatchPacketHeader *>(
                        packet);
                const uint32_t token_count = ReadCount(
                    packet, header->counts_offset, destination);
                const uint32_t assignment_count = ReadCount(
                    packet, header->counts_offset,
                    worker_count + destination);
                __gm__ uint32_t *remote_counts =
                    reinterpret_cast<__gm__ uint32_t *>(recv_counts);
                aclshmem_uint32_p(remote_counts + source, token_count,
                                  static_cast<int32_t>(destination));
                aclshmem_uint32_p(remote_counts + worker_count + source,
                                  assignment_count,
                                  static_cast<int32_t>(destination));
                aclshmem_uint32_p(remote_counts + worker_count * 2u + source,
                                  row_offset,
                                  static_cast<int32_t>(destination));
                aclshmem_uint32_p(remote_counts + worker_count * 3u + source,
                                  assignment_offset,
                                  static_cast<int32_t>(destination));
                row_offset += token_count;
                assignment_offset += assignment_count;
            }
            if (row_offset > row_capacity ||
                assignment_offset > assignment_capacity) {
                *global_status = kStatusInvalidPacket;
                break;
            }
            aclshmem_quiet();
        }
    }
    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);

    const uint32_t data_blocks = blocks;
    if (*global_status == kStatusOk) {
        const uint32_t pair_count = worker_count * worker_count;
        const bool multiple_lanes = data_blocks >= pair_count;
        const uint32_t first_pair = multiple_lanes ? block % pair_count
                                                   : block;
        const uint32_t pair_stride = multiple_lanes ? pair_count : data_blocks;
        const uint32_t pair_limit = multiple_lanes ? first_pair + 1u
                                                   : pair_count;
        for (uint32_t pair = first_pair; pair < pair_limit;
             pair += pair_stride) {
            const uint32_t source = pair / worker_count;
            const uint32_t destination = pair % worker_count;
            const uint32_t lane = multiple_lanes ? block / pair_count : 0u;
            const uint32_t lanes = multiple_lanes
                ? (data_blocks + pair_count - 1u - pair) / pair_count : 1u;
            __gm__ uint8_t *packet = inc_packets +
                static_cast<uint64_t>(source) * slot_bytes;
            __gm__ EndpointDispatchPacketHeader *header =
                reinterpret_cast<__gm__ EndpointDispatchPacketHeader *>(
                    packet);
            const uint32_t token_begin =
                static_cast<uint64_t>(header->token_count) * lane / lanes;
            const uint32_t token_end =
                static_cast<uint64_t>(header->token_count) * (lane + 1u) /
                lanes;
            uint32_t row_index = 0u;
            uint32_t assignment_index = 0u;
            for (uint32_t previous = 0u; previous < source; ++previous) {
                __gm__ uint8_t *prior_packet = inc_packets +
                    static_cast<uint64_t>(previous) * slot_bytes;
                __gm__ EndpointDispatchPacketHeader *prior_header =
                    reinterpret_cast<__gm__ EndpointDispatchPacketHeader *>(
                        prior_packet);
                row_index += ReadCount(prior_packet,
                    prior_header->counts_offset, destination);
                assignment_index += ReadCount(prior_packet,
                    prior_header->counts_offset,
                    worker_count + destination);
            }
            // Derive the lane's deterministic output prefix by parsing only
            // the preceding token slice for this source/destination pair.
            for (uint32_t token = 0u; token < token_begin; ++token) {
                __gm__ EndpointDispatchTokenRecord *record =
                    reinterpret_cast<__gm__ EndpointDispatchTokenRecord *>(
                        packet + header->tokens_offset) + token;
                uint32_t hits = 0u;
                for (uint32_t local = 0u; local < record->assignment_count;
                     ++local) {
                    __gm__ EndpointDispatchAssignmentRecord *assignment =
                        reinterpret_cast<
                            __gm__ EndpointDispatchAssignmentRecord *>(
                                packet + header->assignments_offset) +
                        record->assignment_begin + local;
                    hits += assignment->destination_rank == destination;
                }
                row_index += hits != 0u;
                assignment_index += hits;
            }
            if (lane == 0u) {
                const uint64_t token_capacity_per_source =
                    row_capacity / worker_count;
                const uint64_t assignment_capacity_per_source =
                    assignment_capacity / worker_count;
                if (header->token_count > token_capacity_per_source ||
                    header->assignment_count >
                        assignment_capacity_per_source) {
                    *global_status = kStatusInvalidPacket;
                    break;
                }
                aclshmem_putmem(
                    recv_rows + static_cast<uint64_t>(source) *
                        token_capacity_per_source *
                        sizeof(EndpointDispatchTokenRecord),
                    packet + header->tokens_offset,
                    static_cast<uint64_t>(header->token_count) *
                        sizeof(EndpointDispatchTokenRecord),
                    static_cast<int32_t>(destination));
                aclshmem_putmem(
                    recv_assignments + static_cast<uint64_t>(source) *
                        assignment_capacity_per_source *
                        sizeof(EndpointDispatchAssignmentRecord),
                    packet + header->assignments_offset,
                    static_cast<uint64_t>(header->assignment_count) *
                        sizeof(EndpointDispatchAssignmentRecord),
                    static_cast<int32_t>(destination));
            }
            const uint64_t tile_rows =
                staging_bytes_per_destination / row_bytes;
            uint32_t tile_begin = row_index;
            uint32_t staged_rows = 0u;
            __gm__ uint8_t *staging = hidden_staging +
                static_cast<uint64_t>(block) *
                    staging_bytes_per_destination;
            for (uint32_t token = token_begin; token < token_end; ++token) {
                __gm__ EndpointDispatchTokenRecord *record =
                    reinterpret_cast<__gm__ EndpointDispatchTokenRecord *>(
                        packet + header->tokens_offset) + token;
                uint32_t hits = 0u;
                for (uint32_t local = 0u; local < record->assignment_count;
                     ++local) {
                    __gm__ EndpointDispatchAssignmentRecord *assignment =
                        reinterpret_cast<
                            __gm__ EndpointDispatchAssignmentRecord *>(
                                packet + header->assignments_offset) +
                        record->assignment_begin + local;
                    hits += assignment->destination_rank == destination;
                }
                if (hits == 0u) continue;
                if (row_index >= row_capacity ||
                    assignment_index + hits > assignment_capacity) {
                    *global_status = kStatusInvalidPacket;
                    break;
                }
                __gm__ uint8_t *hidden_source = packet +
                    header->hidden_offset +
                    static_cast<uint64_t>(token) * row_bytes;
                if (tile_rows == 0u) {
                    aclshmem_putmem(
                        recv_hidden + static_cast<uint64_t>(row_index) *
                            row_bytes,
                        hidden_source, row_bytes,
                        static_cast<int32_t>(destination));
                } else {
                    CopyLocalGm(staging +
                        static_cast<uint64_t>(staged_rows) * row_bytes,
                        hidden_source, row_bytes);
                    ++staged_rows;
                    if (staged_rows == tile_rows) {
                        aclshmem_putmem(
                            recv_hidden + static_cast<uint64_t>(tile_begin) *
                                row_bytes,
                            staging, tile_rows * row_bytes,
                            static_cast<int32_t>(destination));
                        tile_begin += staged_rows;
                        staged_rows = 0u;
                    }
                }
                ++row_index;
                assignment_index += hits;
            }
            if (staged_rows != 0u)
                aclshmem_putmem(
                    recv_hidden + static_cast<uint64_t>(tile_begin) *
                        row_bytes,
                    staging, static_cast<uint64_t>(staged_rows) * row_bytes,
                    static_cast<int32_t>(destination));
            if (*global_status != kStatusOk) break;
        }
    }

    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);

    if (block == 0u) {
        // The immutable INC-owned endpoint packets are the dynamic journal.
        // Publish a compact header only after fan-out has joined; a later
        // Combine/index kernel can parse the retained packets during expert
        // compute without any pre-uploaded token plan.
        uint64_t journal_tokens = 0u;
        if (*global_status == kStatusOk) {
            for (uint32_t source = 0u; source < worker_count; ++source) {
                __gm__ uint8_t *packet = inc_packets +
                    static_cast<uint64_t>(source) * slot_bytes;
                __gm__ EndpointDispatchPacketHeader *packet_header =
                    reinterpret_cast<__gm__
                        EndpointDispatchPacketHeader *>(packet);
                journal_tokens += packet_header->token_count;
            }
        }
        __gm__ DeviceJournalHeader *header =
            reinterpret_cast<__gm__ DeviceJournalHeader *>(journal_header);
        header->magic = kDeviceJournalMagic;
        header->abi_version = kDeviceJournalAbiVersion;
        header->struct_bytes = sizeof(DeviceJournalHeader);
        header->generation = generation;
        header->wave = wave;
        header->worker_count = worker_count;
        header->token_count = journal_tokens;
        header->status = *global_status;
        header->flags = 0u;
        header->reserved[0] = 0u;
        header->reserved[1] = 0u;
        header->reserved[2] = 0u;
        dcci_cacheline(journal_header);

        // ACK publication is centralized after all destination owners finish,
        // so a source slot cannot be recycled while another AIV still reads
        // that source packet.
        for (uint32_t source = 0u; source < worker_count; ++source) {
            __gm__ EndpointDispatchAck *ack =
                reinterpret_cast<__gm__ EndpointDispatchAck *>(ack_mailbox) +
                source;
            uint64_t tokens_consumed = 0u;
            if (*global_status == kStatusOk) {
                __gm__ uint8_t *packet = inc_packets +
                    static_cast<uint64_t>(source) * slot_bytes;
                __gm__ EndpointDispatchPacketHeader *header =
                    reinterpret_cast<__gm__ EndpointDispatchPacketHeader *>(
                        packet);
                tokens_consumed = header->token_count;
            }
            PublishAck(ack, source, *global_status, tokens_consumed,
                       generation);
        }
    }

    if (block == 0u) {
        for (uint32_t destination = 0u; destination < worker_count;
             ++destination) {
        uint32_t rows = 0u;
        uint32_t assignments = 0u;
        if (*global_status == kStatusOk) {
            for (uint32_t source = 0u; source < worker_count; ++source) {
                __gm__ uint8_t *packet =
                    inc_packets + static_cast<uint64_t>(source) * slot_bytes;
                __gm__ EndpointDispatchPacketHeader *header =
                    reinterpret_cast<__gm__ EndpointDispatchPacketHeader *>(
                        packet);
                rows += ReadCount(packet, header->counts_offset, destination);
                assignments += ReadCount(packet, header->counts_offset,
                                         worker_count + destination);
            }
        }
        __gm__ EndpointDispatchReceiveCompletion *completion =
            reinterpret_cast<__gm__ EndpointDispatchReceiveCompletion *>(
                completion_mailbox) + destination;
        const int32_t destination_pe = static_cast<int32_t>(destination);
        aclshmem_uint32_p(&completion->magic, kEndpointDispatchMagic,
                          destination_pe);
        aclshmem_uint16_p(&completion->abi_version,
                          kEndpointDispatchAbiVersion, destination_pe);
        aclshmem_uint16_p(&completion->struct_bytes,
                          sizeof(EndpointDispatchReceiveCompletion),
                          destination_pe);
        aclshmem_uint32_p(&completion->wave, wave, destination_pe);
        aclshmem_uint32_p(&completion->destination_rank, destination,
                          destination_pe);
        aclshmem_uint32_p(&completion->status, *global_status,
                          destination_pe);
        aclshmem_uint32_p(
            &completion->row_count,
            *global_status == kStatusOk ? rows : 0u, destination_pe);
        aclshmem_uint32_p(
            &completion->assignment_count,
            *global_status == kStatusOk ? assignments : 0u,
            destination_pe);
        aclshmem_uint32_p(&completion->worker_count, worker_count,
                          destination_pe);
        aclshmem_uint64_p(&completion->reserved[0], 0u, destination_pe);
        aclshmem_uint64_p(&completion->reserved[1], 0u, destination_pe);
        aclshmem_uint64_p(&completion->reserved[2], 0u, destination_pe);
        aclshmem_quiet();
        // Generation publishes the complete receive record.
        aclshmem_uint64_p(&completion->generation, generation,
                          destination_pe);
        aclshmem_quiet();
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(status_line);
}

extern "C" void launch_inc_dc_endpoint_dispatch_device_e2e(
    uint32_t block_dim, void *stream, uint8_t *source_packet,
    uint8_t *inc_packets, uint8_t *commit_mailbox, uint8_t *recv_hidden,
    uint8_t *recv_rows, uint8_t *recv_assignments, uint8_t *recv_counts,
    uint8_t *ack_mailbox, uint8_t *completion_mailbox, uint8_t *cursors,
    uint8_t *hidden_staging, uint8_t *journal_header, uint8_t *status_line,
    uint64_t ffts_addr,
    uint64_t slot_bytes, uint64_t staging_bytes_per_destination,
    uint64_t row_capacity, uint64_t assignment_capacity,
    uint32_t hidden,
    uint32_t dtype, uint32_t expert_count, uint32_t worker_count,
    int32_t inc_pe, uint64_t generation, uint32_t wave)
{
    inc_dc_endpoint_dispatch_device_e2e_kernel<<<
        block_dim, nullptr, stream>>>(
        source_packet, inc_packets, commit_mailbox, recv_hidden, recv_rows,
        recv_assignments, recv_counts, ack_mailbox, completion_mailbox,
        cursors, hidden_staging, journal_header, status_line,
        ffts_addr,
        slot_bytes,
        staging_bytes_per_destination, row_capacity,
        assignment_capacity, hidden, dtype,
        expert_count, worker_count, inc_pe, generation, wave);
}
