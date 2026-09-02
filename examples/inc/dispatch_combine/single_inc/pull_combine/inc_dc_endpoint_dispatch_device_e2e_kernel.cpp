#include "kernel_operator.h"
#include "shmem.h"

#include "inc_dc_endpoint_dispatch_abi.h"

using namespace inc::dc::pull_combine;

namespace {

constexpr uint32_t kStatusOk = 0u;
constexpr uint32_t kStatusCommitTimeout = 1u;
constexpr uint32_t kStatusInvalidCommit = 2u;
constexpr uint32_t kStatusInvalidPacket = 3u;
constexpr uint64_t kCommitSpinLimit = 1000000000ull;

__aicore__ inline uint32_t DTypeBytes(uint32_t dtype)
{
    return dtype == static_cast<uint32_t>(EndpointDataType::FP32) ? 4u : 2u;
}

__aicore__ inline void FlushRange(__gm__ uint8_t *pointer, uint64_t bytes)
{
    AscendC::PipeBarrier<PIPE_ALL>();
    for (uint64_t offset = 0u; offset < bytes; offset += 64u)
        dcci_cacheline(pointer + offset);
}

__aicore__ inline bool WaitCommit(
    __gm__ EndpointDispatchCommit *commit, uint32_t source,
    uint64_t generation, uint32_t wave)
{
    for (uint64_t spin = 0u; spin < kCommitSpinLimit; ++spin) {
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(commit));
        if (commit->generation == generation && commit->wave == wave &&
            commit->source_rank == source)
            return true;
    }
    return false;
}

__aicore__ inline bool HeaderValid(
    __gm__ const EndpointDispatchPacketHeader *header, uint32_t source,
    uint32_t worker_count, uint32_t hidden, uint32_t dtype,
    uint64_t generation, uint32_t wave, uint64_t slot_bytes)
{
    return header->magic == kEndpointDispatchMagic &&
        header->abi_version == kEndpointDispatchAbiVersion &&
        header->header_bytes == sizeof(EndpointDispatchPacketHeader) &&
        header->generation == generation && header->sequence == 1u &&
        header->wave == wave && header->source_rank == source &&
        header->worker_count == worker_count && header->hidden == hidden &&
        header->dtype == dtype && header->flags == 0u &&
        header->packet_bytes != 0u && header->packet_bytes <= slot_bytes &&
        header->counts_offset == sizeof(EndpointDispatchPacketHeader) &&
        header->tokens_offset % kEndpointDispatchAlignment == 0u &&
        header->assignments_offset % kEndpointDispatchAlignment == 0u &&
        header->hidden_offset % kEndpointDispatchAlignment == 0u &&
        header->tokens_offset >= header->counts_offset +
            static_cast<uint64_t>(worker_count) * sizeof(uint32_t) * 2u &&
        header->assignments_offset >= header->tokens_offset +
            static_cast<uint64_t>(header->token_count) *
                sizeof(EndpointDispatchTokenRecord) &&
        header->hidden_offset >= header->assignments_offset +
            static_cast<uint64_t>(header->assignment_count) *
                sizeof(EndpointDispatchAssignmentRecord) &&
        header->hidden_offset +
            static_cast<uint64_t>(header->token_count) * hidden *
                DTypeBytes(dtype) <= header->packet_bytes &&
        header->reserved[0] == 0u && header->reserved[1] == 0u &&
        header->reserved[2] == 0u;
}

__aicore__ inline uint32_t ReadCount(
    __gm__ uint8_t *packet, uint64_t counts_offset, uint32_t index)
{
    __gm__ uint32_t *value = reinterpret_cast<__gm__ uint32_t *>(
        packet + counts_offset) + index;
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(value));
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
    GM_ADDR cursors, GM_ADDR status_line, uint64_t ffts_addr,
    uint64_t slot_bytes,
    uint64_t row_capacity, uint64_t assignment_capacity, uint32_t hidden,
    uint32_t dtype, uint32_t expert_count, uint32_t worker_count,
    int32_t inc_pe, uint64_t generation, uint32_t wave)
{
    shmemx_set_ffts_config(ffts_addr);
    if (AscendC::GetBlockIdx() != 0u) return;
    const int32_t pe = aclshmem_my_pe();
    __gm__ uint32_t *global_status =
        reinterpret_cast<__gm__ uint32_t *>(status_line);

    if (pe != inc_pe) {
        __gm__ EndpointDispatchPacketHeader *local_header =
            reinterpret_cast<__gm__ EndpointDispatchPacketHeader *>(
                source_packet);
        FlushRange(source_packet, sizeof(EndpointDispatchPacketHeader));
        aclshmem_putmem(
            inc_packets + static_cast<uint64_t>(pe) * slot_bytes,
            source_packet, local_header->packet_bytes, inc_pe);
        __gm__ EndpointDispatchCommit *local_commit =
            reinterpret_cast<__gm__ EndpointDispatchCommit *>(
                commit_mailbox) + pe;
        aclshmem_putmem(local_commit, local_commit,
                        sizeof(EndpointDispatchCommit), inc_pe);
        return;
    }

    *global_status = kStatusOk;
    dcci_cacheline(status_line);
    for (uint32_t source = 0u; source < worker_count; ++source) {
        __gm__ EndpointDispatchCommit *commit =
            reinterpret_cast<__gm__ EndpointDispatchCommit *>(
                commit_mailbox) + source;
        if (!WaitCommit(commit, source, generation, wave)) {
            *global_status = kStatusCommitTimeout;
            break;
        }
        if (commit->magic != kEndpointDispatchMagic ||
            commit->abi_version != kEndpointDispatchAbiVersion ||
            commit->struct_bytes != sizeof(EndpointDispatchCommit) ||
            commit->sequence != 1u || commit->slot != 0u ||
            commit->flags != 0u || commit->packet_bytes == 0u ||
            commit->packet_bytes > slot_bytes || commit->reserved != 0u) {
            *global_status = kStatusInvalidCommit;
            break;
        }
        __gm__ uint8_t *packet =
            inc_packets + static_cast<uint64_t>(source) * slot_bytes;
        FlushRange(packet, sizeof(EndpointDispatchPacketHeader));
        __gm__ EndpointDispatchPacketHeader *header =
            reinterpret_cast<__gm__ EndpointDispatchPacketHeader *>(packet);
        if (!HeaderValid(header, source, worker_count, hidden, dtype,
                         generation, wave, slot_bytes) ||
            header->packet_bytes != commit->packet_bytes ||
            header->metadata_digest != commit->metadata_digest) {
            *global_status = kStatusInvalidPacket;
            break;
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(status_line);

    __gm__ uint32_t *cursor = reinterpret_cast<__gm__ uint32_t *>(cursors);
    const uint64_t row_bytes = static_cast<uint64_t>(hidden) *
        DTypeBytes(dtype);

    if (*global_status == kStatusOk) {
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

    if (*global_status == kStatusOk) {
        for (uint32_t source = 0u; source < worker_count; ++source) {
            __gm__ uint8_t *packet =
                inc_packets + static_cast<uint64_t>(source) * slot_bytes;
            __gm__ EndpointDispatchPacketHeader *header =
                reinterpret_cast<__gm__ EndpointDispatchPacketHeader *>(
                    packet);
            for (uint32_t destination = 0u; destination < worker_count;
                 ++destination) {
                uint32_t row_begin = 0u;
                uint32_t assignment_begin = 0u;
                for (uint32_t previous = 0u; previous < source; ++previous) {
                    __gm__ uint8_t *previous_packet =
                        inc_packets +
                        static_cast<uint64_t>(previous) * slot_bytes;
                    __gm__ EndpointDispatchPacketHeader *previous_header =
                        reinterpret_cast<
                            __gm__ EndpointDispatchPacketHeader *>(
                                previous_packet);
                    row_begin += ReadCount(
                        previous_packet, previous_header->counts_offset,
                        destination);
                    assignment_begin += ReadCount(
                        previous_packet, previous_header->counts_offset,
                        worker_count + destination);
                }
                cursor[destination] = row_begin;
                cursor[worker_count + destination] = assignment_begin;
            }

            for (uint32_t token = 0u; token < header->token_count; ++token) {
                __gm__ EndpointDispatchTokenRecord *token_record =
                    reinterpret_cast<__gm__ EndpointDispatchTokenRecord *>(
                        packet + header->tokens_offset) + token;
                dcci_cacheline(
                    reinterpret_cast<__gm__ uint8_t *>(token_record));
                if (token_record->source_token != token ||
                    token_record->assignment_begin >
                        header->assignment_count ||
                    token_record->assignment_count >
                        header->assignment_count -
                            token_record->assignment_begin ||
                    token_record->reserved0 != 0u ||
                    token_record->reserved1 != 0u) {
                    *global_status = kStatusInvalidPacket;
                    break;
                }
                for (uint32_t destination = 0u;
                     destination < worker_count; ++destination) {
                    uint32_t assignments_for_destination = 0u;
                    for (uint32_t local = 0u;
                         local < token_record->assignment_count; ++local) {
                        __gm__ EndpointDispatchAssignmentRecord *assignment =
                            reinterpret_cast<
                                __gm__ EndpointDispatchAssignmentRecord *>(
                                    packet + header->assignments_offset) +
                            token_record->assignment_begin + local;
                        dcci_cacheline(
                            reinterpret_cast<__gm__ uint8_t *>(assignment));
                        if (assignment->destination_rank >= worker_count ||
                            assignment->expert_id >= expert_count) {
                            *global_status = kStatusInvalidPacket;
                            break;
                        }
                        if (assignment->destination_rank == destination)
                            ++assignments_for_destination;
                    }
                    if (*global_status != kStatusOk) break;
                    if (assignments_for_destination == 0u) continue;

                    const uint32_t row_index = cursor[destination]++;
                    const uint32_t output_assignment_begin =
                        cursor[worker_count + destination];
                    if (row_index >= row_capacity ||
                        output_assignment_begin +
                            assignments_for_destination >
                            assignment_capacity) {
                        *global_status = kStatusInvalidPacket;
                        break;
                    }
                    __gm__ EndpointDispatchFanoutRecord *row =
                        reinterpret_cast<
                            __gm__ EndpointDispatchFanoutRecord *>(
                                recv_rows) + row_index;
                    const int32_t destination_pe =
                        static_cast<int32_t>(destination);
                    aclshmem_uint32_p(&row->source_rank, source,
                                      destination_pe);
                    aclshmem_uint32_p(&row->source_token, token,
                                      destination_pe);
                    aclshmem_uint32_p(&row->assignment_begin,
                                      output_assignment_begin,
                                      destination_pe);
                    aclshmem_uint32_p(&row->assignment_count,
                                      assignments_for_destination,
                                      destination_pe);
                    aclshmem_uint64_p(&row->reserved, 0u, destination_pe);
                    aclshmem_quiet();
                    aclshmem_uint64_p(&row->token_id, token_record->token_id,
                                      destination_pe);
                    aclshmem_quiet();
                    aclshmem_putmem(
                        recv_hidden + static_cast<uint64_t>(row_index) *
                            row_bytes,
                        packet + header->hidden_offset +
                            static_cast<uint64_t>(token) * row_bytes,
                        row_bytes, destination_pe);

                    uint32_t output_assignment = output_assignment_begin;
                    for (uint32_t local = 0u;
                         local < token_record->assignment_count; ++local) {
                        __gm__ EndpointDispatchAssignmentRecord *assignment =
                            reinterpret_cast<
                                __gm__ EndpointDispatchAssignmentRecord *>(
                                    packet + header->assignments_offset) +
                            token_record->assignment_begin + local;
                        if (assignment->destination_rank != destination)
                            continue;
                        aclshmem_putmem(
                            reinterpret_cast<__gm__ uint8_t *>(
                                recv_assignments) +
                                static_cast<uint64_t>(output_assignment) *
                                    sizeof(*assignment),
                            reinterpret_cast<__gm__ uint8_t *>(assignment),
                            sizeof(*assignment),
                            static_cast<int32_t>(destination));
                        ++output_assignment;
                    }
                    cursor[worker_count + destination] = output_assignment;
                }
                if (*global_status != kStatusOk) break;
            }

            __gm__ EndpointDispatchAck *ack =
                reinterpret_cast<__gm__ EndpointDispatchAck *>(ack_mailbox) +
                source;
            PublishAck(
                ack, source, *global_status,
                *global_status == kStatusOk ? header->token_count : 0u,
                generation);
            if (*global_status != kStatusOk) break;
        }
    }

    if (*global_status != kStatusOk) {
        // Publish a negative ACK for every source, including sources not yet
        // visited by the data loop, so no ring slot remains permanently busy.
        for (uint32_t source = 0u; source < worker_count; ++source) {
            __gm__ EndpointDispatchAck *ack =
                reinterpret_cast<__gm__ EndpointDispatchAck *>(ack_mailbox) +
                source;
            PublishAck(ack, source, *global_status, 0u, generation);
        }
    }

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
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(status_line);
}

extern "C" void launch_inc_dc_endpoint_dispatch_device_e2e(
    uint32_t block_dim, void *stream, uint8_t *source_packet,
    uint8_t *inc_packets, uint8_t *commit_mailbox, uint8_t *recv_hidden,
    uint8_t *recv_rows, uint8_t *recv_assignments, uint8_t *recv_counts,
    uint8_t *ack_mailbox, uint8_t *completion_mailbox, uint8_t *cursors,
    uint8_t *status_line, uint64_t ffts_addr, uint64_t slot_bytes,
    uint64_t row_capacity, uint64_t assignment_capacity, uint32_t hidden,
    uint32_t dtype, uint32_t expert_count, uint32_t worker_count,
    int32_t inc_pe, uint64_t generation, uint32_t wave)
{
    inc_dc_endpoint_dispatch_device_e2e_kernel<<<
        block_dim, nullptr, stream>>>(
        source_packet, inc_packets, commit_mailbox, recv_hidden, recv_rows,
        recv_assignments, recv_counts, ack_mailbox, completion_mailbox,
        cursors, status_line, ffts_addr, slot_bytes, row_capacity,
        assignment_capacity, hidden, dtype,
        expert_count, worker_count, inc_pe, generation, wave);
}
