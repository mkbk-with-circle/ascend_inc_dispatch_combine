#include "kernel_operator.h"
#include "shmem.h"

#include "inc_dc_pull_dispatch_v2_abi.h"

using namespace inc::dc::pull_v2;

namespace {

constexpr uint32_t kPullOk = 0u;
constexpr uint32_t kPullReadyTimeout = 1u;
constexpr uint32_t kPullInvalidReady = 2u;
constexpr uint32_t kPullInvalidHeader = 3u;

__aicore__ inline bool DTypeValid(uint32_t dtype)
{
    return dtype == static_cast<uint32_t>(DataType::FP16) ||
        dtype == static_cast<uint32_t>(DataType::BF16) ||
        dtype == static_cast<uint32_t>(DataType::FP32);
}

__aicore__ inline bool ReadyValid(
    __gm__ Ready *ready, uint32_t source, uint32_t workers,
    uint64_t session_id, uint64_t placement_epoch, uint64_t generation,
    uint64_t sequence, uint32_t wave, uint32_t region_id,
    uint16_t ring_slot, uint32_t slot_count)
{
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
        ready->flags == 0u && ready->publication != 0u;
}

__aicore__ inline bool HeaderValid(
    __gm__ SlotHeader *header, uint32_t source, uint32_t workers,
    uint64_t session_id, uint64_t placement_epoch, uint64_t generation,
    uint64_t sequence, uint32_t wave, uint32_t region_id,
    uint16_t ring_slot, uint64_t slot_stride)
{
    return header->magic == kPullDispatchMagic &&
        header->abi_version == kPullDispatchAbiVersion &&
        header->header_bytes == sizeof(SlotHeader) &&
        header->session_id == session_id &&
        header->placement_epoch == placement_epoch &&
        header->generation == generation && header->sequence == sequence &&
        header->wave == wave && header->source_rank == source &&
        header->worker_count == workers &&
        header->source_region_id == region_id &&
        header->ring_slot == ring_slot && header->flags == 0u &&
        DTypeValid(header->dtype) && header->hidden != 0u &&
        header->token_record_bytes == sizeof(TokenRecord) &&
        header->assignment_record_bytes == sizeof(AssignmentRecord) &&
        header->tokens_offset >= sizeof(SlotHeader) &&
        header->assignments_offset >= header->tokens_offset &&
        header->hidden_offset >= header->assignments_offset &&
        header->packet_bytes >= header->hidden_offset &&
        header->packet_bytes <= slot_stride &&
        header->tokens_offset % kPullDispatchAlignment == 0u &&
        header->assignments_offset % kPullDispatchAlignment == 0u &&
        header->hidden_offset % kPullDispatchAlignment == 0u &&
        header->packet_bytes % kPullDispatchAlignment == 0u;
}

__aicore__ inline uint64_t AckPublication(uint64_t generation,
                                           uint64_t sequence,
                                           uint32_t source)
{
    uint64_t value = generation ^ (sequence << 1u) ^
        (static_cast<uint64_t>(source) << 48u) ^ 0xa55aa55aa55aa55aull;
    return value == 0u ? 1u : value;
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void inc_dc_pull_dispatch_v2_get_kernel(
    GM_ADDR source_region, GM_ADDR ready_mailbox, GM_ADDR inc_slots,
    GM_ADDR source_acks, GM_ADDR status_line, uint64_t ffts_addr,
    uint64_t session_id, uint64_t placement_epoch, uint64_t generation,
    uint64_t sequence, uint64_t slot_stride, uint32_t worker_count,
    int32_t inc_pe, uint32_t region_id, uint32_t wave,
    uint32_t ring_slot, uint32_t slot_count, uint32_t pull_lanes_per_source,
    uint64_t spin_cap)
{
    shmemx_set_ffts_config(ffts_addr);
    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t blocks = AscendC::GetBlockNum();
    const int32_t pe = aclshmem_my_pe();
    __gm__ PullTimeline *timeline =
        reinterpret_cast<__gm__ PullTimeline *>(status_line);
    __gm__ uint32_t *status = reinterpret_cast<__gm__ uint32_t *>(
        status_line);

    if (worker_count < 2u || worker_count > kPullDispatchMaxWorkers ||
        inc_pe != static_cast<int32_t>(worker_count) ||
        ring_slot >= slot_count || pull_lanes_per_source == 0u ||
        static_cast<uint64_t>(worker_count) * pull_lanes_per_source >
            blocks ||
        slot_stride < sizeof(SlotHeader) ||
        slot_stride % kPullDispatchAlignment != 0u || spin_cap == 0u) {
        if (pe == inc_pe && block == 0u) {
            *status = kPullInvalidHeader;
            dcci_cacheline(status_line);
        }
        return;
    }

    if (pe != inc_pe) {
        if (block == 0u) {
            __gm__ Ready *local = reinterpret_cast<__gm__ Ready *>(
                ready_mailbox) + pe;
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

    if (block == 0u) {
        *status = kPullOk;
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

        bool acquired[kPullDispatchMaxWorkers]{};
        uint32_t remaining = worker_count;
        for (uint64_t spin = 0u;
             spin < spin_cap && remaining != 0u; ++spin) {
            for (uint32_t source = 0u; source < worker_count; ++source) {
                if (acquired[source]) continue;
                __gm__ Ready *ready =
                    reinterpret_cast<__gm__ Ready *>(ready_mailbox) + source;
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(ready));
                if (ready->publication == 0u) continue;
                if (!ReadyValid(ready, source, worker_count, session_id,
                                placement_epoch, generation, sequence, wave,
                                region_id, static_cast<uint16_t>(ring_slot),
                                slot_count)) {
                    *status = kPullInvalidReady;
                    remaining = 0u;
                    break;
                }
                acquired[source] = true;
                --remaining;
                ++timeline->ready_sources;
            }
        }
        if (remaining != 0u && *status == kPullOk)
            *status = kPullReadyTimeout;
        timeline->all_ready = AscendC::GetSystemCycle();

        if (*status == kPullOk) {
            for (uint32_t source = 0u; source < worker_count; ++source) {
                __gm__ uint8_t *local_header = inc_slots +
                    static_cast<uint64_t>(source) * slot_stride;
                __gm__ uint8_t *remote_header = source_region +
                    static_cast<uint64_t>(ring_slot) * slot_stride;
                aclshmem_getmem(local_header, remote_header,
                                sizeof(SlotHeader), source);
                __gm__ SlotHeader *header =
                    reinterpret_cast<__gm__ SlotHeader *>(local_header);
                dcci_cacheline(local_header);
                if (!HeaderValid(
                        header, source, worker_count, session_id,
                        placement_epoch, generation, sequence, wave,
                        region_id, static_cast<uint16_t>(ring_slot),
                        slot_stride)) {
                    *status = kPullInvalidHeader;
                    break;
                }
            }
        }
        timeline->headers_pulled = AscendC::GetSystemCycle();
        timeline->hidden_get_begin = timeline->headers_pulled;
        dcci_cacheline(status_line);
    }
    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);

    if (*status == kPullOk) {
        const uint32_t source = block / pull_lanes_per_source;
        const uint32_t lane = block % pull_lanes_per_source;
        if (source < worker_count) {
            __gm__ uint8_t *local_slot = inc_slots +
                static_cast<uint64_t>(source) * slot_stride;
            __gm__ SlotHeader *header =
                reinterpret_cast<__gm__ SlotHeader *>(local_slot);
            dcci_cacheline(local_slot);
            const uint64_t payload_bytes =
                header->packet_bytes - sizeof(SlotHeader);
            const uint64_t begin = sizeof(SlotHeader) +
                payload_bytes * lane / pull_lanes_per_source;
            const uint64_t end = sizeof(SlotHeader) +
                payload_bytes * (lane + 1u) / pull_lanes_per_source;
            if (begin < end) {
                __gm__ uint8_t *remote_slot = source_region +
                    static_cast<uint64_t>(ring_slot) * slot_stride;
                aclshmem_getmem(local_slot + begin, remote_slot + begin,
                                static_cast<uint32_t>(end - begin), source);
            }
        }
    }
    AscendC::SyncAll<true>();

    if (block == 0u) {
        timeline->hidden_get_done = AscendC::GetSystemCycle();
        for (uint32_t source = 0u; source < worker_count; ++source) {
            __gm__ SourceConsumed *ack =
                reinterpret_cast<__gm__ SourceConsumed *>(source_acks) +
                source;
            ack->magic = kPullDispatchMagic;
            ack->abi_version = kPullDispatchAbiVersion;
            ack->struct_bytes = sizeof(SourceConsumed);
            ack->session_id = session_id;
            ack->placement_epoch = placement_epoch;
            ack->generation = generation;
            ack->sequence = sequence;
            ack->dispatch_cookie = 0u;
            ack->wave = wave;
            ack->source_rank = source;
            ack->source_region_id = region_id;
            ack->status = *status;
            ack->ring_slot = static_cast<uint16_t>(ring_slot);
            ack->flags = 0u;
            ack->reserved0 = 0u;
            ack->bytes_consumed = 0u;
            if (*status == kPullOk) {
                __gm__ SlotHeader *header =
                    reinterpret_cast<__gm__ SlotHeader *>(
                        inc_slots + static_cast<uint64_t>(source) *
                            slot_stride);
                ack->bytes_consumed = header->packet_bytes;
            }
            ack->publication = 0u;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(ack));
            aclshmem_uint64_p(&ack->publication, 0u,
                              static_cast<int32_t>(source));
            aclshmem_quiet();
            aclshmem_putmem(ack, ack,
                            __builtin_offsetof(SourceConsumed, publication),
                            source);
            aclshmem_quiet();
            aclshmem_uint64_p(
                &ack->publication,
                AckPublication(generation, sequence, source), source);
            aclshmem_quiet();
        }
        timeline->source_acks_done = AscendC::GetSystemCycle();
        timeline->kernel_done = timeline->source_acks_done;
        dcci_cacheline(status_line);
    }
}

extern "C" void launch_inc_dc_pull_dispatch_v2_get(
    uint32_t block_dim, void *stream, uint8_t *source_region,
    uint8_t *ready_mailbox, uint8_t *inc_slots, uint8_t *source_acks,
    uint8_t *status_line, uint64_t ffts_addr, uint64_t session_id,
    uint64_t placement_epoch, uint64_t generation, uint64_t sequence,
    uint64_t slot_stride, uint32_t worker_count, int32_t inc_pe,
    uint32_t region_id, uint32_t wave, uint32_t ring_slot,
    uint32_t slot_count, uint32_t pull_lanes_per_source, uint64_t spin_cap)
{
    inc_dc_pull_dispatch_v2_get_kernel<<<block_dim, nullptr, stream>>>(
        source_region, ready_mailbox, inc_slots, source_acks, status_line,
        ffts_addr, session_id, placement_epoch, generation, sequence,
        slot_stride, worker_count, inc_pe, region_id, wave, ring_slot,
        slot_count, pull_lanes_per_source, spin_cap);
}
