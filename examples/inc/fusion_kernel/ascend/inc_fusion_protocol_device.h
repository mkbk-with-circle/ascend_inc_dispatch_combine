#ifndef INC_FUSION_PROTOCOL_DEVICE_H
#define INC_FUSION_PROTOCOL_DEVICE_H

#include "kernel_operator.h"
#include "shmem.h"

#include "inc_fusion_abi.h"

namespace inc::fusion::device {

constexpr uint32_t kFusionHeaderReadyBytes = sizeof(uint64_t);
constexpr uint32_t kFusionHeaderProducerLineBytes = 64u;

__aicore__ inline __gm__ const FusionRemoteRequestHeader *RemoteRequest(
    __gm__ const FusionKernelArgs *args)
{
    return args->remote_request == 0u
        ? nullptr
        : reinterpret_cast<__gm__ const FusionRemoteRequestHeader *>(
              args->remote_request);
}

__aicore__ inline uint64_t OperationGeneration(
    __gm__ const FusionKernelArgs *args)
{
    __gm__ const FusionRemoteRequestHeader *request = RemoteRequest(args);
    return request == nullptr ? args->operation_generation
                              : request->operation_generation;
}

__aicore__ inline uint64_t ServiceTicket(
    __gm__ const FusionKernelArgs *args)
{
    __gm__ const FusionRemoteRequestHeader *request = RemoteRequest(args);
    return request == nullptr ? args->service_ticket
                              : request->service_ticket;
}

__aicore__ inline uint64_t RequestId(
    __gm__ const FusionKernelArgs *args)
{
    __gm__ const FusionRemoteRequestHeader *request = RemoteRequest(args);
    return request == nullptr ? args->request_id : request->request_id;
}

__aicore__ inline uint32_t RequestFlags(
    __gm__ const FusionKernelArgs *args)
{
    __gm__ const FusionRemoteRequestHeader *request = RemoteRequest(args);
    return request == nullptr ? args->flags : request->flags;
}

__aicore__ inline uint64_t PacketCommit(uint64_t generation,
                                        uint32_t sequence)
{
    return (generation << 32u) |
           (static_cast<uint64_t>(sequence) + 1u);
}

__aicore__ inline void Dcci(__gm__ uint8_t *line, uint64_t bytes)
{
    const uint64_t count = (bytes + 63u) / 64u;
    for (uint64_t i = 0u; i < count; ++i)
        dcci_cacheline(line + i * 64u);
}

// dcci publishes to the device point of coherency, so a multi-MiB Cube/Vector
// handoff buffer can be acquired by the whole cohort in parallel.  One core
// walking the buffer alone puts tens of thousands of serialised cache-line
// operations on the FFN critical path.  Callers must fence producers before
// the stripe and join all stripes before any consumer reads.
__aicore__ inline void DcciStripe(__gm__ uint8_t *base, uint64_t bytes,
                                  uint32_t worker, uint32_t workers)
{
    if (bytes == 0u || workers == 0u) return;
    const uint64_t count = (bytes + 63u) / 64u;
    const uint64_t begin = count * worker / workers;
    const uint64_t end = count * (worker + 1u) / workers;
    for (uint64_t i = begin; i < end; ++i)
        dcci_cacheline(base + i * 64u);
}

// Whole-cache clean and invalidate.  A ranged sweep issues one serialised
// cache operation per 64 B line, so publishing a multi-MiB handoff buffer that
// way costs tens of thousands of them; this is a single instruction covering
// every line the core holds.  It is strictly stronger than any ranged sweep by
// the same core, so it is a safe substitute wherever a range is published, at
// the cost of also dropping unrelated clean lines.
__aicore__ inline void DcciAll(__gm__ uint8_t *base)
{
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(base);
    AscendC::DataCacheCleanAndInvalid<
        uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
        AscendC::DcciDst::CACHELINE_OUT>(global);
}

__aicore__ inline uint64_t QueueIndex(__gm__ const FusionKernelArgs *args,
                                      uint32_t slot, uint32_t worker,
                                      uint32_t lane, uint32_t sequence)
{
    uint64_t request_slot = slot;
    const uint64_t service_ticket = ServiceTicket(args);
    if ((RequestFlags(args) & kFusionRemoteIncService) != 0u &&
        service_ticket != 0u &&
        args->remote_service.ring_size >= kFusionMinServiceRing &&
        args->remote_service.ring_size <= kFusionMaxServiceRing) {
        request_slot +=
            ((service_ticket - 1u) %
                 args->remote_service.ring_size) * args->slot_count;
    }
    return (((request_slot * args->worker_count + worker) *
              args->symmetric_layout.queue_lanes + lane) *
             args->symmetric_layout.queue_depth) +
           sequence % args->symmetric_layout.queue_depth;
}

__aicore__ inline uint64_t BulkRequestSlot(
    __gm__ const FusionKernelArgs *args, uint32_t slot)
{
    uint64_t request_slot = slot;
    const uint64_t service_ticket = ServiceTicket(args);
    if ((RequestFlags(args) & kFusionRemoteIncService) != 0u &&
        service_ticket != 0u &&
        args->remote_service.ring_size >= kFusionMinServiceRing &&
        args->remote_service.ring_size <= kFusionMaxServiceRing) {
        request_slot += ((service_ticket - 1u) %
            args->remote_service.ring_size) * args->slot_count;
    }
    return request_slot;
}

__aicore__ inline uint64_t BulkInputStride(
    __gm__ const FusionKernelArgs *args)
{
    return static_cast<uint64_t>(args->tokens_per_wave) * args->hidden *
           sizeof(bfloat16_t);
}

__aicore__ inline uint64_t BulkSourceRows(
    __gm__ const FusionKernelArgs *args)
{
    const uint32_t destinations = args->topk < args->worker_count
        ? args->topk : args->worker_count;
    return static_cast<uint64_t>(args->tokens_per_wave) * destinations;
}

__aicore__ inline uint64_t BulkSourceAssignments(
    __gm__ const FusionKernelArgs *args)
{
    return static_cast<uint64_t>(args->tokens_per_wave) * args->topk;
}

__aicore__ inline uint64_t BulkRouteStride(
    __gm__ const FusionKernelArgs *args)
{
    uint64_t bytes = BulkSourceRows(args) * sizeof(FusionDispatchRow) +
        BulkSourceAssignments(args) * sizeof(FusionExpertAssignment);
    return (bytes + 511u) / 512u * 512u;
}

__aicore__ inline uint64_t BulkExpertStride(
    __gm__ const FusionKernelArgs *args)
{
    return static_cast<uint64_t>(args->tokens_per_wave) *
        args->worker_count * args->topk *
        (static_cast<uint64_t>(args->hidden) * sizeof(bfloat16_t) +
         sizeof(FusionExpertAssignment));
}

__aicore__ inline uint64_t BulkControlLines(
    __gm__ const FusionKernelArgs *args)
{
    return static_cast<uint64_t>(4u) * args->worker_count +
        static_cast<uint64_t>(args->worker_count) * args->worker_count +
        static_cast<uint64_t>(2u) * args->resources.inc_combine_aiv +
        static_cast<uint64_t>(6u) * args->worker_count +
        args->resources.inc_dispatch_aiv;
}

__aicore__ inline uint64_t BulkReleaseBase(
    __gm__ const FusionKernelArgs *args)
{
    return static_cast<uint64_t>(4u) * args->worker_count +
        static_cast<uint64_t>(args->worker_count) * args->worker_count +
        static_cast<uint64_t>(2u) * args->resources.inc_combine_aiv;
}

__aicore__ inline uint64_t BulkHandshakeBase(
    __gm__ const FusionKernelArgs *args)
{
    return BulkReleaseBase(args) +
        static_cast<uint64_t>(2u) * args->worker_count;
}

__aicore__ inline uint64_t BulkIncPrepBase(
    __gm__ const FusionKernelArgs *args)
{
    return BulkHandshakeBase(args) +
        static_cast<uint64_t>(3u) * args->worker_count;
}

__aicore__ inline uint64_t BulkDispatchSourceBase(
    __gm__ const FusionKernelArgs *args)
{
    return BulkIncPrepBase(args) + args->resources.inc_dispatch_aiv;
}

__aicore__ inline __gm__ uint8_t *BulkArena(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint64_t base, uint64_t stride, uint32_t slot, uint32_t worker)
{
    return sym + base +
        (BulkRequestSlot(args, slot) * args->worker_count + worker) * stride;
}

__aicore__ inline __gm__ FusionBulkControl *BulkControl(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint32_t slot, uint64_t line)
{
    return reinterpret_cast<__gm__ FusionBulkControl *>(
        sym + args->symmetric_layout.reserved64[3]) +
        BulkRequestSlot(args, slot) * BulkControlLines(args) + line;
}

__aicore__ inline bool WaitBulkGeneration(
    __gm__ FusionBulkControl *control, uint64_t generation,
    uint32_t spin_cap)
{
    uint32_t spins = 0u;
    while (true) {
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(control));
        if (control->generation == generation) {
            // `generation` is the commit word for the complete cache line.
            // Re-acquire after observing it so count/byte fields cannot be
            // consumed from an older cached generation (the same rule used
            // by WaitReady for packet metadata).
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(control));
            if (control->generation == generation) return true;
        }
        ++spins;
        if (spin_cap != 0u && spins >= spin_cap) return false;
    }
}

__aicore__ inline void PublishBulkLocal(
    __gm__ FusionBulkControl *control, uint64_t generation,
    uint32_t count0, uint32_t count1, uint64_t bytes0, uint64_t bytes1)
{
    control->count0 = count0;
    control->count1 = count1;
    control->bytes0 = bytes0;
    control->bytes1 = bytes1;
    control->generation = generation;
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(control));
}

__aicore__ inline void PublishBulkRemote(
    __gm__ FusionBulkControl *control, uint64_t generation,
    uint32_t count0, uint32_t count1, uint64_t bytes0, uint64_t bytes1,
    int32_t remote_pe)
{
    PublishBulkLocal(control, generation, count0, count1, bytes0, bytes1);
    aclshmem_putmem(control, control, sizeof(FusionBulkControl), remote_pe);
}

__aicore__ inline __gm__ FusionPacketHeader *Header(
    __gm__ uint8_t *sym, uint64_t header_off, uint64_t index)
{
    return reinterpret_cast<__gm__ FusionPacketHeader *>(
        sym + header_off + index * sizeof(FusionPacketHeader));
}

__aicore__ inline __gm__ uint8_t *Payload(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint64_t payload_off, uint64_t index)
{
    return sym + payload_off +
           index * args->symmetric_layout.packet_bytes;
}

__aicore__ inline bool WaitReady(__gm__ FusionPacketHeader *header,
                                 uint64_t generation, uint32_t sequence,
                                 uint32_t spin_cap)
{
    uint32_t spins = 0u;
    const uint64_t commit = PacketCommit(generation, sequence);
    // Always invalidate before the first commit load. A remote 8-byte commit
    // can become visible while the rest of this locally cached header is
    // stale; accepting it without this acquire step produces a torn packet.
    while (true) {
        Dcci(reinterpret_cast<__gm__ uint8_t *>(header),
             kFusionHeaderProducerLineBytes);
        for (uint32_t poll = 0u; poll < 32u; ++poll) {
            if (header->ready == commit) {
                // This second invalidate is the acquire edge for metadata if
                // the coherent ready load changed between polling refreshes.
                Dcci(reinterpret_cast<__gm__ uint8_t *>(header),
                     kFusionHeaderProducerLineBytes);
                if (header->ready == commit) return true;
                break;
            }
            ++spins;
            if (spin_cap != 0u && spins >= spin_cap) return false;
        }
    }
}

__aicore__ inline bool WaitCredit(__gm__ FusionPacketHeader *header,
                                  uint64_t previous_commit,
                                  uint32_t spin_cap)
{
    if (previous_commit == 0u) return true;
    uint32_t spins = 0u;
    while (true) {
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
            &header->credit));
        for (uint32_t poll = 0u; poll < 32u; ++poll) {
            if (header->credit == previous_commit) return true;
            ++spins;
            if (spin_cap != 0u && spins >= spin_cap) return false;
        }
    }
}

// Publish payload first and the ready cacheline last. The remote consumer can
// never observe a ready generation whose payload is still in flight.
__aicore__ inline bool Publish(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint64_t header_off, uint64_t payload_off, uint64_t index,
    __gm__ const uint8_t *source, uint32_t payload_bytes,
    const FusionPacketHeader &metadata, uint64_t generation,
    uint32_t sequence, int32_t remote_pe)
{
    if (payload_bytes > args->symmetric_layout.packet_bytes) return false;
    __gm__ FusionPacketHeader *header = Header(sym, header_off, index);
    // The queue entry can be reused both within one wave and by a later wave
    // mapped to the same slot. Its previous ready ticket is authoritative:
    // do not overwrite the entry until that exact publication was credited.
    // This avoids needing the previous stream length, which is dynamic under
    // MoE routing.
    // A later kernel launch, or the next request of a persistent kernel, may
    // retain a pre-publication cache line for this queue entry.  `ready` is
    // the authoritative reuse ticket, so acquire it before deciding that the
    // slot has no predecessor.  Without this acquire a stale zero can skip
    // credit backpressure and overwrite an unconsumed packet.
    Dcci(reinterpret_cast<__gm__ uint8_t *>(header),
         kFusionHeaderProducerLineBytes);
    const uint64_t previous = header->ready;
    if (!WaitCredit(header, previous, args->spin_cap)) return false;
    __gm__ uint8_t *packet = Payload(sym, args, payload_off, index);
    if (payload_bytes != 0u) {
        aclshmem_putmem_nbi(packet,
            const_cast<__gm__ uint8_t *>(source), payload_bytes, remote_pe);
    }
    header->wave = metadata.wave;
    header->source_rank = metadata.source_rank;
    header->destination_rank = metadata.destination_rank;
    header->source_token = metadata.source_token;
    header->destination_token = metadata.destination_token;
    header->expert_id = metadata.expert_id;
    header->route_ordinal = metadata.route_ordinal;
    header->weight_bits = metadata.weight_bits;
    header->payload_bytes = metadata.payload_bytes;
    header->payload_offset = metadata.payload_offset;
    header->kind = metadata.kind;
    header->reserved = 0u;
    AscendC::PipeBarrier<PIPE_ALL>();
    Dcci(reinterpret_cast<__gm__ uint8_t *>(header),
         kFusionHeaderProducerLineBytes);
    // Payload and metadata share one NBI batch. The following quiet completes
    // both before the independent commit, avoiding a per-payload fence while
    // preserving the acquire/release protocol.
    aclshmem_putmem_nbi(
        reinterpret_cast<__gm__ uint8_t *>(header) +
            kFusionHeaderReadyBytes,
        reinterpret_cast<__gm__ uint8_t *>(header) +
            kFusionHeaderReadyBytes,
        kFusionHeaderProducerLineBytes - kFusionHeaderReadyBytes,
        remote_pe);
    aclshmem_quiet();
    header->ready = PacketCommit(generation, sequence);
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(header));
    aclshmem_putmem_nbi(
        reinterpret_cast<__gm__ uint8_t *>(&header->ready),
        reinterpret_cast<__gm__ uint8_t *>(&header->ready),
        sizeof(header->ready), remote_pe);
    aclshmem_quiet();
    return true;
}

// Credit has a different writer from ready and owns a separate cache line.
// Only its eight-byte word is returned to the producer.
__aicore__ inline void Release(__gm__ FusionPacketHeader *header,
                               uint32_t sequence, int32_t producer_pe)
{
    header->credit = header->ready;
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
        &header->credit));
    aclshmem_putmem_nbi(
        reinterpret_cast<__gm__ uint8_t *>(&header->credit),
        reinterpret_cast<__gm__ uint8_t *>(&header->credit),
        sizeof(header->credit), producer_pe);
    aclshmem_quiet();
}

__aicore__ inline int32_t WorkerPe(__gm__ const FusionKernelArgs *args,
                                   uint32_t worker)
{
    if (args->worker_pes == 0u) return static_cast<int32_t>(worker);
    __gm__ uint32_t *pes = reinterpret_cast<__gm__ uint32_t *>(
        args->worker_pes);
    return static_cast<int32_t>(pes[worker]);
}

} // namespace inc::fusion::device

#endif
