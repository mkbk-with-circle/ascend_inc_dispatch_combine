#include "kernel_operator.h"
#include "shmem.h"

#include "inc_dc_single_inc_inline_bench_abi.h"
#include "../../common/platform/inc_dc_gather_mte_aicore.h"

using namespace inc::dc;
using namespace inc::dc::single_inline;

__aicore__ inline uint64_t InlineAlign64Device(uint64_t value)
{
    return (value + 63u) & ~63ull;
}

__aicore__ inline uint32_t InlineCeilDivDevice(uint32_t a, uint32_t b)
{
    return a / b + (a % b != 0u ? 1u : 0u);
}

__aicore__ inline void InlineZeroDevice(__gm__ uint8_t *dst, uint64_t bytes)
{
    uint64_t i = 0u;
    for (; i + sizeof(uint64_t) <= bytes; i += sizeof(uint64_t))
        *reinterpret_cast<__gm__ uint64_t *>(dst + i) = 0u;
    for (; i < bytes; ++i) dst[i] = 0u;
}

__aicore__ inline void InlineFlushLinesDevice(__gm__ uint8_t *base,
                                               uint64_t bytes)
{
    for (uint64_t offset = 0u; offset < bytes; offset += 64u)
        dcci_cacheline(base + offset);
}

// Copy a regular group of complete-token payloads between two record arrays.
// Both sides are strided because protocol metadata stays adjacent to each
// token.  One MTE2 gather + one MTE3 scatter replaces the per-token 8-KiB
// bounce loop while preserving the complete-record wire layout.
__aicore__ inline void InlineCopyStridedPayloadRows(
    GM_ADDR dst, GM_ADDR src, uint32_t row_bytes, uint32_t rows,
    uint32_t src_gap_bytes, uint32_t dst_gap_bytes,
    __ubuf__ uint8_t *ub, uint32_t event)
{
    AscendC::LocalTensor<uint8_t> ub_tensor;
    AscendC::GlobalTensor<uint8_t> gm_src;
    AscendC::GlobalTensor<uint8_t> gm_dst;
    AscendC::DataCopyPadExtParams<uint8_t> pad;
    AscendC::DataCopyExtParams gather(
        static_cast<uint16_t>(rows), row_bytes, src_gap_bytes, 0u, 0u);
    ub_tensor.address_.logicPos =
        static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ub);
    ub_tensor.address_.dataLen = DcLlGatherAlign32(row_bytes * rows);
    gm_src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(src));
    gm_dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(dst));
    AscendC::DataCopyPad(ub_tensor, gm_src, gather, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
    AscendC::DataCopyExtParams scatter(
        static_cast<uint16_t>(rows), row_bytes, 0u, dst_gap_bytes, 0u);
    AscendC::DataCopyPad(gm_dst, ub_tensor, scatter);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
}

__aicore__ inline bool InlineWait32Device(__gm__ uint8_t *line,
                                          uint32_t generation,
                                          uint32_t spin_cap)
{
    __gm__ volatile uint32_t *value =
        reinterpret_cast<__gm__ volatile uint32_t *>(line);
    uint32_t spins = 0u;
    while (*value != generation) {
        dcci_cacheline(line);
        if (++spins >= spin_cap) return false;
    }
    return true;
}

__aicore__ inline void InlinePublishLocal32Device(__gm__ uint8_t *line,
                                                   uint32_t generation)
{
    *reinterpret_cast<__gm__ uint32_t *>(line) = generation;
    dcci_cacheline(line);
}

__aicore__ inline uint64_t InlineMixDevice(uint64_t x)
{
    x ^= x >> 30u;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27u;
    x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31u);
}

__aicore__ inline __gm__ InlineDispatchBenchStatV2 *InlineBenchStat(
    __gm__ uint8_t *sym, __gm__ InlineDispatchBenchDescV2 *desc,
    uint32_t lane)
{
    const uint64_t index =
        static_cast<uint64_t>(desc->pe) * kInlineDispatchMaxLanes + lane;
    return reinterpret_cast<__gm__ InlineDispatchBenchStatV2 *>(
        sym + desc->stats_off + index * sizeof(InlineDispatchBenchStatV2));
}

__aicore__ inline void InlineClearStatDevice(
    __gm__ InlineDispatchBenchStatV2 *stat)
{
    stat->start_cycle = 0u;
    stat->end_cycle = 0u;
    stat->upload_bytes = 0u;
    stat->fanout_bytes = 0u;
    stat->token_records = 0u;
    stat->assignments = 0u;
    stat->batches = 0u;
    stat->error = kInlineDispatchOk;
    stat->done_generation = 0u;
    stat->reserved32 = 0u;
    stat->reserved64 = 0u;
}

__aicore__ inline bool InlineDescriptorValidDevice(
    __gm__ InlineDispatchBenchDescV2 *d)
{
    return d->magic == kInlineDispatchBenchMagic &&
           d->version == kInlineDispatchBenchVersion &&
           d->worker_count != 0u && d->inc_pe == d->worker_count &&
           d->expert_count != 0u && d->hidden_bytes != 0u &&
           d->max_topk != 0u && d->tokens_per_worker != 0u &&
           d->batch_tokens != 0u && d->worker_lane_count != 0u &&
           d->inc_lane_count != 0u &&
           d->worker_lane_count <= kInlineDispatchMaxLanes &&
           d->inc_lane_count <= kInlineDispatchMaxLanes &&
           d->spin_cap != 0u && d->generation != 0u &&
           d->slot_count != 0u && d->total_bytes >= sizeof(*d);
}

__aicore__ inline uint64_t InlineStoredTokenStrideDevice(
    __gm__ InlineDispatchBenchDescV2 *d)
{
    return InlineAlign64Device(
        InlineAlign64Device(sizeof(InlineTokenRecordHeaderV2) +
                            static_cast<uint64_t>(d->max_topk) *
                                sizeof(InlineRouteEntryV2)) +
        d->hidden_bytes);
}

__aicore__ inline uint64_t InlineBatchRecordsOffsetDevice(uint32_t count)
{
    return InlineAlign64Device(sizeof(InlineTokenBatchHeaderV2) +
                               (static_cast<uint64_t>(count) + 1u) *
                                   sizeof(uint64_t));
}

__aicore__ inline void InlineBuildWorkerBatch(
    __gm__ uint8_t *sym, __gm__ InlineDispatchBenchDescV2 *d,
    uint32_t batch, __ubuf__ uint8_t *ub,
    __gm__ InlineDispatchBenchStatV2 *stat)
{
    const uint32_t begin = batch * d->batch_tokens;
    uint32_t count = d->tokens_per_worker - begin;
    if (count > d->batch_tokens) count = d->batch_tokens;
    const uint32_t batches =
        InlineCeilDivDevice(d->tokens_per_worker, d->batch_tokens);
    const uint32_t slot = d->generation % d->slot_count;
    const uint64_t frame_index =
        (static_cast<uint64_t>(slot) * d->worker_count + d->pe) * batches +
        batch;
    __gm__ uint8_t *frame =
        sym + d->ingress_frame_off + frame_index * d->ingress_frame_stride;
    const uint64_t records_off = InlineBatchRecordsOffsetDevice(count);
    const uint64_t stored_stride = InlineStoredTokenStrideDevice(d);
    const uint64_t frame_bytes = records_off + count * stored_stride;
    if (frame_bytes > d->ingress_frame_stride) {
        stat->error = kInlineDispatchCapacity;
        return;
    }

    __gm__ InlineTokenBatchHeaderV2 *bh =
        reinterpret_cast<__gm__ InlineTokenBatchHeaderV2 *>(frame);
    bh->magic = kInlineRouteBatchMagic;
    bh->version = kInlineRouteVersion;
    bh->header_bytes = kInlineRouteHeaderBytes;
    bh->flags = kInlineRouteRecordNone;
    bh->source_rank = d->pe;
    bh->session_id = d->session_id;
    bh->placement_epoch = d->placement_epoch;
    bh->generation = d->generation;
    bh->request_id = d->request_id;
    bh->wave = batch;
    bh->record_count = count;
    bh->offset_table_bytes = (static_cast<uint64_t>(count) + 1u) * 8u;
    bh->records_offset = records_off;
    bh->frame_bytes = frame_bytes;
    bh->directory_commit = 0u;
    for (uint32_t r = 0u; r < 5u; ++r) bh->reserved64[r] = 0u;
    __gm__ uint64_t *offsets = reinterpret_cast<__gm__ uint64_t *>(
        frame + sizeof(InlineTokenBatchHeaderV2));
    for (uint32_t i = 0u; i <= count; ++i)
        offsets[i] = records_off + static_cast<uint64_t>(i) * stored_stride;
    const uint64_t directory_bytes =
        sizeof(InlineTokenBatchHeaderV2) +
        (static_cast<uint64_t>(count) + 1u) * sizeof(uint64_t);
    InlineZeroDevice(frame + directory_bytes, records_off - directory_bytes);

    for (uint32_t i = 0u; i < count; ++i) {
        const uint32_t token = begin + i;
        const uint32_t routes = *reinterpret_cast<__gm__ uint32_t *>(
            sym + d->source_route_count_off +
            static_cast<uint64_t>(token) * sizeof(uint32_t));
        if (routes == 0u || routes > d->max_topk) {
            stat->error = kInlineDispatchBadRoute;
            return;
        }
        __gm__ uint8_t *record = frame + offsets[i];
        const uint64_t payload_off = InlineAlign64Device(
            sizeof(InlineTokenRecordHeaderV2) +
            static_cast<uint64_t>(routes) * sizeof(InlineRouteEntryV2));
        const uint64_t record_bytes = payload_off + d->hidden_bytes;
        __gm__ InlineTokenRecordHeaderV2 *th =
            reinterpret_cast<__gm__ InlineTokenRecordHeaderV2 *>(record);
        th->magic = kInlineRouteTokenMagic;
        th->version = kInlineRouteVersion;
        th->header_bytes = kInlineRouteHeaderBytes;
        th->flags = kInlineRouteRecordNone;
        th->source_rank = d->pe;
        th->session_id = d->session_id;
        th->placement_epoch = d->placement_epoch;
        th->generation = d->generation;
        th->request_id = d->request_id;
        th->source_token = token;
        th->hidden_bytes = d->hidden_bytes;
        th->payload_offset = payload_off;
        th->record_bytes = record_bytes;
        th->route_count = routes;
        th->route_entry_bytes = sizeof(InlineRouteEntryV2);
        th->wave = batch;
        th->reserved32 = 0u;
        th->commit = 0u;
        th->reserved64[0] = 0u;
        th->reserved64[1] = 0u;
        th->reserved64[2] = 0u;
        __gm__ InlineRouteEntryV2 *dst_routes =
            reinterpret_cast<__gm__ InlineRouteEntryV2 *>(
                record + sizeof(InlineTokenRecordHeaderV2));
        __gm__ InlineRouteEntryV2 *src_routes =
            reinterpret_cast<__gm__ InlineRouteEntryV2 *>(
                sym + d->source_route_entry_off) +
            static_cast<uint64_t>(token) * d->max_topk;
        for (uint32_t r = 0u; r < routes; ++r) {
            dst_routes[r].expert_id = src_routes[r].expert_id;
            dst_routes[r].route_ordinal = src_routes[r].route_ordinal;
            dst_routes[r].weight_bits = src_routes[r].weight_bits;
            dst_routes[r].reserved32 = 0u;
        }
        const uint64_t metadata_bytes =
            sizeof(InlineTokenRecordHeaderV2) +
            static_cast<uint64_t>(routes) * sizeof(InlineRouteEntryV2);
        InlineZeroDevice(record + metadata_bytes,
                         payload_off - metadata_bytes);
        DcLlGatherCopyGm2GmChunked(
            reinterpret_cast<GM_ADDR>(record + payload_off),
            reinterpret_cast<GM_ADDR>(
                sym + d->source_hidden_off +
                static_cast<uint64_t>(token) * d->hidden_bytes),
            d->hidden_bytes, ub, kDcLlGatherEvent);
        th->commit = d->generation;
        ++stat->token_records;
        stat->assignments += routes;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    // Scalar-written protocol metadata is cache-coherent only after DCCI.
    // Hidden payload was produced by the MTE copy and is already ordered by
    // the PIPE_ALL barrier; invalidating every payload cacheline here would
    // serialize large records for no visibility benefit.
    InlineFlushLinesDevice(frame, records_off);
    for (uint32_t i = 0u; i < count; ++i) {
        __gm__ uint8_t *record = frame + offsets[i];
        __gm__ InlineTokenRecordHeaderV2 *header =
            reinterpret_cast<__gm__ InlineTokenRecordHeaderV2 *>(record);
        InlineFlushLinesDevice(record, header->payload_offset);
    }
    // Do not alias a signal word with the bulk payload.  On the current
    // ACLSHMEM backend that can make the signal visible before the first body
    // cacheline.  Publish the complete frame, drain it, then publish the
    // 64-bit protocol commit as a second ordered put.  The INC never reads
    // token metadata before this commit becomes visible.
    bh->directory_commit = 0u;
    constexpr uint32_t kInlinePrivateMteTile = 16u * 1024u;
    __ubuf__ uint8_t *tx_ub =
        reinterpret_cast<__ubuf__ uint8_t *>(64u * 1024u);
    if (frame_bytes <= 0xffffffffull) {
        aclshmemx_mte_put_nbi(
            frame, frame, tx_ub, kInlinePrivateMteTile,
            static_cast<uint32_t>(frame_bytes), static_cast<int>(d->inc_pe),
            0u);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
        aclshmemx_mte_quiet();
    } else {
        aclshmem_putmem_nbi(frame, frame, frame_bytes,
                            static_cast<int>(d->inc_pe));
        aclshmem_quiet();
    }
    bh->directory_commit = d->generation;
    dcci_cacheline(
        frame + offsetof(InlineTokenBatchHeaderV2, directory_commit));
    aclshmem_putmem_nbi(
        frame + offsetof(InlineTokenBatchHeaderV2, directory_commit),
        frame + offsetof(InlineTokenBatchHeaderV2, directory_commit),
        sizeof(uint64_t), static_cast<int>(d->inc_pe));
    aclshmem_quiet();
    stat->upload_bytes += frame_bytes + sizeof(uint64_t);
    ++stat->batches;
}

__aicore__ inline bool InlineParseAndFanoutToken(
    __gm__ uint8_t *sym, __gm__ InlineDispatchBenchDescV2 *d,
    __gm__ InlineTokenRecordHeaderV2 *th, uint32_t slot,
    __ubuf__ uint8_t *ub, bool defer_transport,
    __gm__ InlineDispatchBenchStatV2 *stat)
{
    if (th->commit != d->generation) return false;
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(th) + 64u);
    if (th->magic != kInlineRouteTokenMagic ||
        th->version != kInlineRouteVersion ||
        th->header_bytes != kInlineRouteHeaderBytes ||
        th->source_rank >= d->worker_count ||
        th->session_id != d->session_id ||
        th->placement_epoch != d->placement_epoch ||
        th->generation != d->generation ||
        th->request_id != d->request_id ||
        th->source_token >= d->tokens_per_worker ||
        th->hidden_bytes != d->hidden_bytes || th->route_count == 0u ||
        th->route_count > d->max_topk ||
        th->route_entry_bytes != sizeof(InlineRouteEntryV2)) return false;
    const uint64_t expected_payload = InlineAlign64Device(
        sizeof(InlineTokenRecordHeaderV2) +
        static_cast<uint64_t>(th->route_count) *
            sizeof(InlineRouteEntryV2));
    if (th->payload_offset != expected_payload ||
        th->record_bytes != expected_payload + d->hidden_bytes) return false;

    const uint64_t global_token =
        static_cast<uint64_t>(th->source_rank) * d->tokens_per_worker +
        th->source_token;
    __gm__ InlineRouteEntryV2 *routes =
        reinterpret_cast<__gm__ InlineRouteEntryV2 *>(
            reinterpret_cast<__gm__ uint8_t *>(th) +
            sizeof(InlineTokenRecordHeaderV2));
    __gm__ uint32_t *owners = reinterpret_cast<__gm__ uint32_t *>(
        sym + d->expert_owner_off);
    __gm__ uint32_t *local_experts = reinterpret_cast<__gm__ uint32_t *>(
        sym + d->expert_local_index_off);

    for (uint32_t r = 0u; r < th->route_count; ++r) {
        if (routes[r].expert_id >= d->expert_count ||
            routes[r].route_ordinal != r || routes[r].reserved32 != 0u ||
            owners[routes[r].expert_id] >= d->worker_count) return false;
    }

    for (uint32_t first = 0u; first < th->route_count; ++first) {
        const uint32_t destination = owners[routes[first].expert_id];
        bool seen = false;
        uint32_t assignment_count = 1u;
        if (!defer_transport) {
            for (uint32_t prior = 0u; prior < first; ++prior)
                if (owners[routes[prior].expert_id] == destination)
                    seen = true;
            if (seen) continue;
            assignment_count = 0u;
            for (uint32_t r = first; r < th->route_count; ++r)
                if (owners[routes[r].expert_id] == destination)
                    ++assignment_count;
        }
        // Destination-major storage makes every dense source microbatch one
        // contiguous remote put per destination.  The routing decision is
        // still made above from this token's metadata; the host supplies no
        // physical fanout plan.
        const uint64_t record_index =
            (static_cast<uint64_t>(slot) * d->worker_count + destination) *
                d->worker_count * d->tokens_per_worker +
            global_token;
        const uint64_t physical_stride = defer_transport
            ? d->dense_fanout_metadata_stride
            : d->fanout_record_stride;
        __gm__ uint8_t *fanout = sym +
            (defer_transport ? d->dense_fanout_metadata_off
                             : d->fanout_record_off) +
            record_index * physical_stride;
        const uint64_t payload_off = InlineAlign64Device(
            sizeof(InlineFanoutRecordHeaderV2) +
            static_cast<uint64_t>(assignment_count) *
                sizeof(InlineFanoutAssignmentV2));
        const uint64_t record_bytes = payload_off + d->hidden_bytes;
        if ((!defer_transport && record_bytes > d->fanout_record_stride) ||
            (defer_transport &&
             payload_off > d->dense_fanout_metadata_stride))
            return false;
        if (!defer_transport) {
            InlineZeroDevice(fanout, payload_off);
        } else {
            // The dense path has exactly one assignment per destination.
            // Initialize every ABI field explicitly and clear only the final
            // alignment tail instead of zeroing the full 192-byte record.
            *reinterpret_cast<__gm__ uint64_t *>(
                fanout + sizeof(InlineFanoutRecordHeaderV2) +
                sizeof(InlineFanoutAssignmentV2)) = 0u;
            *reinterpret_cast<__gm__ uint64_t *>(
                fanout + sizeof(InlineFanoutRecordHeaderV2) +
                sizeof(InlineFanoutAssignmentV2) + sizeof(uint64_t)) = 0u;
        }
        __gm__ InlineFanoutRecordHeaderV2 *fh =
            reinterpret_cast<__gm__ InlineFanoutRecordHeaderV2 *>(fanout);
        fh->magic = kInlineRouteFanoutMagic;
        fh->version = kInlineRouteVersion;
        fh->header_bytes = kInlineRouteHeaderBytes;
        fh->flags = kInlineRouteRecordNone;
        fh->resolved_destination_rank = destination;
        fh->session_id = d->session_id;
        fh->placement_epoch = d->placement_epoch;
        fh->generation = d->generation;
        fh->request_id = d->request_id;
        fh->source_token = th->source_token;
        fh->hidden_bytes = d->hidden_bytes;
        fh->payload_offset = payload_off;
        fh->record_bytes = record_bytes;
        fh->source_rank = th->source_rank;
        fh->assignment_count = assignment_count;
        fh->wave = th->wave;
        fh->reserved32 = 0u;
        fh->commit = 0u;
        fh->reserved64[0] = 0u;
        fh->reserved64[1] = 0u;
        fh->reserved64[2] = 0u;
        __gm__ InlineFanoutAssignmentV2 *assignments =
            reinterpret_cast<__gm__ InlineFanoutAssignmentV2 *>(
                fanout + sizeof(InlineFanoutRecordHeaderV2));
        uint32_t out = 0u;
        const uint32_t route_begin = defer_transport ? first : 0u;
        const uint32_t route_end = defer_transport ? first + 1u
                                                   : th->route_count;
        for (uint32_t r = route_begin; r < route_end; ++r) {
            if (!defer_transport &&
                owners[routes[r].expert_id] != destination)
                continue;
            const uint64_t journal_index =
                (static_cast<uint64_t>(slot) * d->worker_count *
                     d->tokens_per_worker + global_token) * d->max_topk + r;
            const uint64_t locator = journal_index + 1u;
            const uint64_t auth = InlineMixDevice(
                d->session_id ^ d->placement_epoch ^ d->request_id ^
                (static_cast<uint64_t>(d->generation) << 32u) ^ locator);
            const uint32_t destination_row = static_cast<uint32_t>(
                global_token * d->max_topk + r);
            assignments[out].route_key.journal_locator = locator;
            assignments[out].route_key.authenticator = auth;
            assignments[out].expert_id = routes[r].expert_id;
            assignments[out].local_expert =
                local_experts[routes[r].expert_id];
            assignments[out].route_ordinal = r;
            assignments[out].flags = 0u;
            assignments[out].destination_row = destination_row;
            assignments[out].reserved64 = 0u;
            __gm__ InlineDispatchJournalEntryV2 *journal =
                reinterpret_cast<__gm__ InlineDispatchJournalEntryV2 *>(
                    sym + d->journal_off) + journal_index;
            journal->route_key.journal_locator = locator;
            journal->route_key.authenticator = auth;
            journal->source_token = th->source_token;
            journal->source_rank = th->source_rank;
            journal->contributor_rank = destination;
            journal->expert_id = routes[r].expert_id;
            journal->local_expert = assignments[out].local_expert;
            journal->destination_row = destination_row;
            journal->route_ordinal = r;
            journal->weight_bits = routes[r].weight_bits;
            journal->wave = th->wave;
            journal->generation = d->generation;
            journal->valid_generation = d->generation;
            // Dense batches publish the journal and all destination metadata
            // with one whole-cache operation after every token in the batch
            // has been parsed.  Walking one cache line per assignment makes
            // scalar coherence work scale with W and was the dominant W4
            // overhead.  The generic path has no later batch publication, so
            // retain its fail-closed per-entry visibility here.
            if (!defer_transport)
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(journal));
            ++out;
        }
        if (!defer_transport) {
            DcLlGatherCopyGm2GmChunked(
                reinterpret_cast<GM_ADDR>(fanout + payload_off),
                reinterpret_cast<GM_ADDR>(
                    reinterpret_cast<__gm__ uint8_t *>(th) +
                    th->payload_offset),
                d->hidden_bytes, ub, kDcLlGatherEvent);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        if (defer_transport) {
            // Batch-ready is the release publication for all records in the
            // contiguous microbatch, so the copied record may already carry
            // its final commit value.
            fh->commit = d->generation;
        } else {
            InlineFlushLinesDevice(fanout, payload_off);
            aclshmem_putmem_nbi(fanout, fanout, record_bytes,
                                static_cast<int>(destination));
            aclshmem_quiet();
            fh->commit = d->generation;
            dcci_cacheline(fanout +
                           offsetof(InlineFanoutRecordHeaderV2, commit));
            aclshmem_putmem_nbi(
                fanout + offsetof(InlineFanoutRecordHeaderV2, commit),
                fanout + offsetof(InlineFanoutRecordHeaderV2, commit),
                sizeof(uint64_t), static_cast<int>(destination));
            aclshmem_quiet();
            stat->fanout_bytes += record_bytes + sizeof(uint64_t);
        }
    }
    ++stat->token_records;
    stat->assignments += th->route_count;
    return true;
}

__aicore__ inline void InlineDispatchWorkerDevice(
    __gm__ uint8_t *sym, __gm__ InlineDispatchBenchDescV2 *d,
    uint32_t lane)
{
    if (lane >= d->worker_lane_count) return;
    __gm__ InlineDispatchBenchStatV2 *stat = InlineBenchStat(sym, d, lane);
    InlineClearStatDevice(stat);
    stat->start_cycle = get_sys_cnt();
    const uint32_t batches =
        InlineCeilDivDevice(d->tokens_per_worker, d->batch_tokens);
    __ubuf__ uint8_t *ub = reinterpret_cast<__ubuf__ uint8_t *>(0);
    for (uint32_t batch = lane; batch < batches;
         batch += d->worker_lane_count) {
        InlineBuildWorkerBatch(sym, d, batch, ub, stat);
        if (stat->error != kInlineDispatchOk) break;
    }
    aclshmem_quiet();
    // A PE has one kernel cohort but batches are striped over all worker
    // lanes.  Publish every lane locally before lane 0 announces the worker
    // to the INC; otherwise the INC can observe an incomplete microbatch set.
    stat->done_generation = d->generation;
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(stat) +
                   offsetof(InlineDispatchBenchStatV2, done_generation));
    if (lane == 0u) {
        for (uint32_t other = 0u; other < d->worker_lane_count; ++other) {
            __gm__ InlineDispatchBenchStatV2 *other_stat =
                InlineBenchStat(sym, d, other);
            if (!InlineWait32Device(
                    reinterpret_cast<__gm__ uint8_t *>(other_stat) +
                        offsetof(InlineDispatchBenchStatV2, done_generation),
                    d->generation, d->spin_cap)) {
                stat->error = kInlineDispatchTimeout;
                break;
            }
            if (other_stat->error != kInlineDispatchOk)
                stat->error = other_stat->error;
        }
    }
    if (lane == 0u && stat->error == kInlineDispatchOk) {
        aclshmem_putmem_signal(
            sym + d->worker_done_off + static_cast<uint64_t>(d->pe) * 64u + 8u,
            sym + d->worker_done_off + static_cast<uint64_t>(d->pe) * 64u + 8u,
            sizeof(uint64_t),
            reinterpret_cast<__gm__ int32_t *>(
                sym + d->worker_done_off + static_cast<uint64_t>(d->pe) * 64u),
            static_cast<int32_t>(d->generation), ACLSHMEM_SIGNAL_SET,
            static_cast<int>(d->inc_pe));
        const uint64_t complete_index =
            static_cast<uint64_t>(d->worker_count + d->inc_lane_count + d->pe);
        if (!InlineWait32Device(sym + d->worker_done_off +
                                   complete_index * 64u,
                               d->generation, d->spin_cap))
            stat->error = kInlineDispatchTimeout;
    }
    stat->end_cycle = get_sys_cnt();
}

__aicore__ inline void InlineDispatchIncDevice(
    __gm__ uint8_t *sym, __gm__ InlineDispatchBenchDescV2 *d,
    uint32_t lane)
{
    if (lane >= d->inc_lane_count) return;
    __gm__ InlineDispatchBenchStatV2 *stat = InlineBenchStat(sym, d, lane);
    InlineClearStatDevice(stat);
    stat->start_cycle = get_sys_cnt();
    const uint32_t batches =
        InlineCeilDivDevice(d->tokens_per_worker, d->batch_tokens);
    const uint32_t slot = d->generation % d->slot_count;
    __ubuf__ uint8_t *ub = reinterpret_cast<__ubuf__ uint8_t *>(0);
    const uint64_t total_batches =
        static_cast<uint64_t>(d->worker_count) * batches;
    for (uint64_t flat = lane; flat < total_batches;
         flat += d->inc_lane_count) {
        const uint32_t source = static_cast<uint32_t>(flat / batches);
        const uint32_t batch = static_cast<uint32_t>(flat % batches);
        const uint64_t frame_index =
            (static_cast<uint64_t>(slot) * d->worker_count + source) *
                batches + batch;
        __gm__ uint8_t *frame = sym + d->ingress_frame_off +
            frame_index * d->ingress_frame_stride;
        if (!InlineWait32Device(
                frame + offsetof(InlineTokenBatchHeaderV2, directory_commit),
                d->generation, d->spin_cap)) {
            stat->error = kInlineDispatchTimeout;
            break;
        }
        __gm__ InlineTokenBatchHeaderV2 *bh =
            reinterpret_cast<__gm__ InlineTokenBatchHeaderV2 *>(frame);
        dcci_cacheline(frame);
        if (bh->magic != kInlineRouteBatchMagic ||
            bh->version != kInlineRouteVersion ||
            bh->source_rank != source || bh->generation != d->generation ||
            bh->record_count == 0u || bh->record_count > d->batch_tokens ||
            bh->frame_bytes > d->ingress_frame_stride) {
            stat->reserved32 = bh->magic != kInlineRouteBatchMagic
                ? 1u
                : (bh->version != kInlineRouteVersion
                       ? 2u
                       : (bh->source_rank != source
                              ? 3u
                              : (bh->generation != d->generation
                                     ? 4u
                                     : ((bh->record_count == 0u ||
                                         bh->record_count > d->batch_tokens)
                                            ? 5u
                                            : 6u))));
            stat->error = kInlineDispatchBadBatch;
            break;
        }
        __gm__ uint64_t *offsets = reinterpret_cast<__gm__ uint64_t *>(
            frame + sizeof(InlineTokenBatchHeaderV2));
        for (uint32_t i = 0u; i < bh->record_count; ++i) {
            if (offsets[i] < bh->records_offset ||
                offsets[i + 1u] <= offsets[i] ||
                offsets[i + 1u] > bh->frame_bytes) {
                stat->reserved32 = 100u + i;
                stat->error = kInlineDispatchBadBatch;
                break;
            }
        }
        if (stat->error != kInlineDispatchOk) break;
        // Detect the dense all-destination case from the committed token
        // metadata itself.  This is an online protocol decision, not a host
        // plan or a W-specific shortcut.  Dense batches can be relayed as one
        // contiguous frame per destination; arbitrary sparse/skewed routes
        // retain the generic per-record path.
        bool dense_all_destinations = d->worker_count <= 64u;
        __gm__ uint32_t *owners = reinterpret_cast<__gm__ uint32_t *>(
            sym + d->expert_owner_off);
        for (uint32_t i = 0u; i < bh->record_count; ++i) {
            __gm__ InlineTokenRecordHeaderV2 *candidate =
                reinterpret_cast<__gm__ InlineTokenRecordHeaderV2 *>(
                    frame + offsets[i]);
            if (candidate->route_count != d->worker_count) {
                dense_all_destinations = false;
                break;
            }
            __gm__ InlineRouteEntryV2 *candidate_routes =
                reinterpret_cast<__gm__ InlineRouteEntryV2 *>(
                    reinterpret_cast<__gm__ uint8_t *>(candidate) +
                    sizeof(InlineTokenRecordHeaderV2));
            uint64_t destination_mask = 0u;
            for (uint32_t route = 0u;
                 route < candidate->route_count; ++route) {
                if (candidate_routes[route].expert_id >= d->expert_count) {
                    dense_all_destinations = false;
                    break;
                }
                const uint32_t destination =
                    owners[candidate_routes[route].expert_id];
                if (destination >= d->worker_count ||
                    (destination_mask & (1ull << destination)) != 0u) {
                    dense_all_destinations = false;
                    break;
                }
                destination_mask |= 1ull << destination;
            }
            const uint64_t expected_mask = d->worker_count == 64u
                ? ~0ull
                : ((1ull << d->worker_count) - 1ull);
            if (destination_mask != expected_mask)
                dense_all_destinations = false;
            if (!dense_all_destinations) break;
        }
        for (uint32_t i = 0u; i < bh->record_count; ++i) {
            if (offsets[i] < bh->records_offset ||
                offsets[i + 1u] <= offsets[i] ||
                offsets[i + 1u] > bh->frame_bytes) {
                stat->reserved32 = 100u + i;
                stat->error = kInlineDispatchBadBatch;
                break;
            }
            __gm__ InlineTokenRecordHeaderV2 *th =
                reinterpret_cast<__gm__ InlineTokenRecordHeaderV2 *>(
                    frame + offsets[i]);
            if (!InlineParseAndFanoutToken(
                    sym, d, th, slot, ub, dense_all_destinations, stat)) {
                stat->error = kInlineDispatchBadToken;
                break;
            }
        }
        if (stat->error != kInlineDispatchOk) break;
        if (dense_all_destinations) {
            // Online parsing above produced both the authoritative journal
            // and all destination-specific fanout metadata.  Publish that
            // scalar state once per microbatch before any MTE reads it.  This
            // replaces O(tokens * destinations) cache-line operations while
            // preserving the same release boundary and fail-closed parser.
            dcci_entire_cache();
            const uint64_t global_begin =
                static_cast<uint64_t>(source) * d->tokens_per_worker +
                static_cast<uint64_t>(batch) * d->batch_tokens;
            __gm__ InlineTokenRecordHeaderV2 *first_token =
                reinterpret_cast<__gm__ InlineTokenRecordHeaderV2 *>(
                    frame + offsets[0]);
            const uint64_t input_stride = bh->record_count > 1u
                ? offsets[1] - offsets[0]
                : InlineStoredTokenStrideDevice(d);
            constexpr uint32_t kDenseGatherUbBytes = 64u * 1024u;
            uint32_t rows_per_copy = d->hidden_bytes == 0u
                ? 0u
                : kDenseGatherUbBytes / d->hidden_bytes;
            if (rows_per_copy == 0u) rows_per_copy = 1u;
            constexpr uint32_t kTxSlots = 2u;
            constexpr uint32_t kPrivateMteTile = 16u * 1024u;
            bool tx_busy[kTxSlots] = {false, false};
            const uint64_t metadata_put_bytes =
                static_cast<uint64_t>(bh->record_count) *
                d->dense_fanout_metadata_stride;
            const uint64_t payload_put_bytes =
                static_cast<uint64_t>(bh->record_count) * d->hidden_bytes;
            const bool private_supported =
                metadata_put_bytes <= 0xffffffffull &&
                payload_put_bytes <= 0xffffffffull;
            // Every destination receives the same hidden rows in a dense
            // all-destination batch.  Gather the strided wire records once
            // into one immutable INC-local payload staging range, then use it
            // as the source for all remote destinations.  Destination-specific
            // metadata remains separate and was resolved from each token.
            const uint64_t staging_record_index =
                (static_cast<uint64_t>(slot) * d->worker_count) *
                    d->worker_count * d->tokens_per_worker +
                global_begin;
            __gm__ uint8_t *staging_payload =
                sym + d->dense_fanout_payload_off +
                staging_record_index * d->hidden_bytes;
            for (uint32_t row = 0u; row < bh->record_count;
                 row += rows_per_copy) {
                uint32_t group = bh->record_count - row;
                if (group > rows_per_copy) group = rows_per_copy;
                __gm__ uint8_t *src_payload =
                    reinterpret_cast<__gm__ uint8_t *>(first_token) +
                    first_token->payload_offset +
                    static_cast<uint64_t>(row) * input_stride;
                __gm__ uint8_t *dst_payload = staging_payload +
                    static_cast<uint64_t>(row) * d->hidden_bytes;
                if (d->hidden_bytes <= kDenseGatherUbBytes) {
                    InlineCopyStridedPayloadRows(
                        reinterpret_cast<GM_ADDR>(dst_payload),
                        reinterpret_cast<GM_ADDR>(src_payload),
                        d->hidden_bytes, group,
                        static_cast<uint32_t>(input_stride -
                                              d->hidden_bytes),
                        0u, ub, kDcLlGatherEvent);
                } else {
                    DcLlGatherCopyGm2GmChunked(
                        reinterpret_cast<GM_ADDR>(dst_payload),
                        reinterpret_cast<GM_ADDR>(src_payload),
                        d->hidden_bytes, ub, kDcLlGatherEvent);
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            __ubuf__ uint8_t *payload_ub[kTxSlots] = {
                reinterpret_cast<__ubuf__ uint8_t *>(64u * 1024u),
                reinterpret_cast<__ubuf__ uint8_t *>(
                    64u * 1024u + kPrivateMteTile)};
            uint32_t tx_ping = 0u;
            for (uint32_t destination = 0u;
                 destination < d->worker_count; ++destination) {
                const uint64_t record_index =
                    (static_cast<uint64_t>(slot) * d->worker_count +
                     destination) * d->worker_count *
                        d->tokens_per_worker +
                    global_begin;
                __gm__ uint8_t *first =
                    sym + d->dense_fanout_metadata_off +
                    record_index * d->dense_fanout_metadata_stride;
                __gm__ uint8_t *remote_payload =
                    sym + d->dense_fanout_payload_off +
                    record_index * d->hidden_bytes;
                if (private_supported) {
                    // Payload transfers use both private-MTE credits in
                    // ping-pong order.  Small metadata transfers use the
                    // public NBI path and drain once after the cohort, so a
                    // 192-byte header no longer occupies an entire private
                    // credit and serializes the next destination.
                    const uint32_t tx_slot = tx_ping % kTxSlots;
                    if (tx_busy[tx_slot]) {
                        AscendC::WaitFlag<
                            AscendC::HardEvent::MTE3_MTE2>(tx_slot);
                        tx_busy[tx_slot] = false;
                    }
                    aclshmemx_mte_put_nbi(
                        remote_payload, staging_payload, payload_ub[tx_slot],
                        kPrivateMteTile,
                        static_cast<uint32_t>(payload_put_bytes),
                        static_cast<int>(destination), tx_slot);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                        tx_slot);
                    tx_busy[tx_slot] = true;
                    ++tx_ping;
                    aclshmem_putmem_nbi(first, first, metadata_put_bytes,
                                        static_cast<int>(destination));
                } else {
                    aclshmem_putmem_nbi(remote_payload, staging_payload,
                                        payload_put_bytes,
                                        static_cast<int>(destination));
                    aclshmem_putmem_nbi(first, first, metadata_put_bytes,
                                        static_cast<int>(destination));
                    aclshmem_quiet();
                }
                stat->fanout_bytes +=
                    payload_put_bytes + metadata_put_bytes;
            }
            bool any_private = false;
            for (uint32_t tx_slot = 0u; tx_slot < kTxSlots; ++tx_slot) {
                if (tx_busy[tx_slot]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                        tx_slot);
                    tx_busy[tx_slot] = false;
                    any_private = true;
                }
            }
            if (any_private) aclshmemx_mte_quiet();
            if (private_supported) aclshmem_quiet();
            // One ready ticket releases a whole complete-record microbatch.
            // Publish only after every destination transfer above has drained.
            for (uint32_t destination = 0u;
                 destination < d->worker_count; ++destination) {
                const uint64_t ready_index =
                    (((static_cast<uint64_t>(slot) * d->worker_count +
                       source) * batches + batch) * d->worker_count +
                     destination);
                __gm__ uint8_t *ready =
                    sym + d->fanout_ready_off + ready_index * 64u;
                *reinterpret_cast<__gm__ uint64_t *>(ready + 8u) =
                    d->generation;
                dcci_cacheline(ready);
                aclshmem_putmem_signal(
                    ready + 8u, ready + 8u, sizeof(uint64_t),
                    reinterpret_cast<__gm__ int32_t *>(ready),
                    static_cast<int32_t>(d->generation),
                    ACLSHMEM_SIGNAL_SET, static_cast<int>(destination));
            }
        }
        ++stat->batches;
    }
    aclshmem_quiet();
    const uint64_t parser_done =
        static_cast<uint64_t>(d->worker_count + lane);
    InlinePublishLocal32Device(
        sym + d->worker_done_off + parser_done * 64u, d->generation);
    if (lane == 0u) {
        for (uint32_t other = 0u; other < d->inc_lane_count; ++other) {
            const uint64_t index =
                static_cast<uint64_t>(d->worker_count + other);
            if (!InlineWait32Device(sym + d->worker_done_off + index * 64u,
                                   d->generation, d->spin_cap))
                stat->error = kInlineDispatchTimeout;
        }
        if (stat->error == kInlineDispatchOk) {
            for (uint32_t worker = 0u; worker < d->worker_count; ++worker) {
                const uint64_t index = static_cast<uint64_t>(
                    d->worker_count + d->inc_lane_count + worker);
                __gm__ uint8_t *line =
                    sym + d->worker_done_off + index * 64u;
                aclshmem_putmem_signal(
                    line + 8u, line + 8u, sizeof(uint64_t),
                    reinterpret_cast<__gm__ int32_t *>(line),
                    static_cast<int32_t>(d->generation),
                    ACLSHMEM_SIGNAL_SET, static_cast<int>(worker));
            }
            aclshmem_quiet();
        }
    }
    stat->end_cycle = get_sys_cnt();
    stat->done_generation = d->generation;
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
inc_dc_single_inc_inline_dispatch_kernel(GM_ADDR symmetric_heap)
{
#if defined(ASCEND_IS_AIV)
    __gm__ uint8_t *sym = reinterpret_cast<__gm__ uint8_t *>(symmetric_heap);
    __gm__ InlineDispatchBenchDescV2 *desc =
        reinterpret_cast<__gm__ InlineDispatchBenchDescV2 *>(sym);
    const uint32_t lane =
        static_cast<uint32_t>(AscendC::GetBlockIdx());
    if (!InlineDescriptorValidDevice(desc)) {
        if (lane == 0u && desc->stats_off != 0u) {
            __gm__ InlineDispatchBenchStatV2 *stat = InlineBenchStat(
                sym, desc, lane);
            InlineClearStatDevice(stat);
            stat->error = kInlineDispatchBadDescriptor;
        }
        return;
    }
    if (desc->pe < desc->worker_count)
        InlineDispatchWorkerDevice(sym, desc, lane);
    else if (desc->pe == desc->inc_pe)
        InlineDispatchIncDevice(sym, desc, lane);
#endif
}

extern "C" void launch_inc_dc_single_inc_inline_dispatch_kernel(
    uint8_t *sym, int block_dim, void *stream)
{
    inc_dc_single_inc_inline_dispatch_kernel<<<block_dim, nullptr, stream>>>(
        sym);
}
