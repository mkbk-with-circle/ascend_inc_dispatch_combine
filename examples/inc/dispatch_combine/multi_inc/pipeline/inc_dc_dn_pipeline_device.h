/**
 * Pipeline device 辅助：复用 Queue-v2 P6 原语，扩展两跳 tail/head publish。
 */
#pragma once

// 必须先拉 kernel_operator，再打开 namespace inc（与 queue_v2_device 一致）
#include "inc_dc_dn_queue_v2_device.h"
#include "inc_dc_dn_pipeline_abi.h"

namespace inc::dc::dn::pl {

using inc::dc::dn::qv2::Qv2DrainTileDataAndDescriptor;
using inc::dc::dn::qv2::Qv2IssueDescriptorTileP6NoDrain;
using inc::dc::dn::qv2::Qv2IssuePayloadBatchP6NoDrain;
using inc::dc::dn::qv2::Qv2IssuePayloadRangeP6NoDrainGmSrc;
using inc::dc::dn::qv2::Qv2IssuePayloadRangeP6NoDrain;
using inc::dc::dn::qv2::Qv2IssueDescriptorRangeP6NoDrain;
using inc::dc::dn::qv2::Qv2MteUbComplete;
using inc::dc::dn::qv2::Qv2PublishTailOrdered;
using inc::dc::dn::qv2::Qv2CheckPayload;
using inc::dc::dn::qv2::Qv2DcciLine;
using inc::dc::dn::qv2::Qv2DcciRange;

// Candidate2：活跃 upload lane 数（INC gather/reclaim 禁止硬等满 kPlRawUploadLaneCount）
__aicore__ inline uint32_t PlDevUploadLaneCount(__gm__ PlPipelineDesc *desc)
{
    uint32_t U = desc->lane_count;
    if (U == 0u) {
        U = kPlMaxUploadLanes;
    }
    if (U > kPlMaxUploadLanes) {
        U = kPlMaxUploadLanes;
    }
    return U;
}

// Candidate2：与 host PlMakeWorkerServiceLayout 同口径（device 内联，禁止依赖 host-only inline）
__aicore__ inline PlWorkerServiceLayout PlDevWorkerServiceLayout(__gm__ PlPipelineDesc *desc)
{
    PlWorkerServiceLayout L{};
    uint32_t U = desc->lane_count;
    if (U == 0u) {
        U = kPlMaxUploadLanes;
    }
    if (U > kPlMaxUploadLanes) {
        U = kPlMaxUploadLanes;
    }
    uint32_t R = (desc->pipeline_mode & kPlRecvLaneCountMask) >> kPlRecvLaneCountShift;
    if (R == 0u) {
        R = kPlMaxRecvLanes;
    }
    if (R > kPlMaxRecvLanes) {
        R = kPlMaxRecvLanes;
    }
    L.upload_lane_count = U;
    L.recv_lane_count = R;
    L.upload_begin = 0u;
    L.recv_begin = U;
    L.control_block = U + R;
    L.block_dim = U + R + 1u;
    return L;
}

__aicore__ inline void PlDcci(__gm__ uint8_t *p)
{
    dcci_cacheline(p);
}

__aicore__ inline uint64_t PlDevDestinationCsrOffsetsOff(
    uint32_t metadata_bytes)
{
    const uint64_t aligned =
        (static_cast<uint64_t>(metadata_bytes) + 63u) & ~uint64_t{63u};
    return kPlIncRawRouteMetaOff + aligned;
}

__aicore__ inline bool PlDevDestinationCsrFits(
    uint32_t token_count, uint32_t topk, uint32_t worker_count)
{
    const uint64_t route_count =
        static_cast<uint64_t>(token_count) *
        static_cast<uint64_t>(topk);
    const uint64_t metadata_bytes =
        route_count * sizeof(uint32_t);
    const uint64_t offsets_end =
        ((metadata_bytes + 63u) & ~uint64_t{63u}) +
        (static_cast<uint64_t>(worker_count) * 8u + 1u) *
            sizeof(uint32_t);
    return token_count <= 0xffffu &&
           offsets_end <= kPlIncRawRouteMetaBytes &&
           route_count * sizeof(PlDestinationCsrEntry) <=
               kPlIncRawRouteOrdinalBytes;
}

__aicore__ inline void PlDcciUploadCtr(__gm__ PlUploadCounters *c)
{
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(c));
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(c) + 64u);
}

__aicore__ inline void PlDcciForwardCtr(__gm__ PlForwardCounters *c)
{
    __gm__ uint8_t *p = reinterpret_cast<__gm__ uint8_t *>(c);
    for (uint32_t i = 0; i < 3u; ++i) {
        PlDcci(p + static_cast<uint64_t>(i) * 64u);
    }
}

__aicore__ inline void PlDcciRecvCtr(__gm__ PlRecvCounters *c)
{
    __gm__ uint8_t *p = reinterpret_cast<__gm__ uint8_t *>(c);
    for (uint32_t i = 0; i < 3u; ++i) {
        PlDcci(p + static_cast<uint64_t>(i) * 64u);
    }
}

__aicore__ inline void PlTraceDcci(__gm__ PlServiceTraceLine *tr)
{
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(tr));
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(tr) + 64u);
}

__aicore__ inline uint64_t DevPlIngressTailLineOff(uint32_t lane_id)
{
    return kPlIngressTailOff + static_cast<uint64_t>(lane_id) * 64u;
}

__aicore__ inline uint64_t DevPlIngressHeadLineOff(uint32_t lane_id)
{
    return kPlIngressHeadOff + static_cast<uint64_t>(lane_id) * 64u;
}

__aicore__ inline uint64_t DevPlWorkerDescLaneRelOff(uint32_t lane_id)
{
    return static_cast<uint64_t>(lane_id) * kPlWorkerDescLaneStride * kPlDescriptorBytes;
}

__aicore__ inline uint64_t DevPlEgressTailLineOff(uint32_t source_id)
{
    return kPlEgressTailOff + static_cast<uint64_t>(source_id) * 64u;
}

__aicore__ inline uint64_t DevPlEgressHeadLineOff(uint32_t source_id)
{
    return kPlEgressHeadOff + static_cast<uint64_t>(source_id) * 64u;
}

__aicore__ inline uint64_t DevPlIncPayloadLaneOff(uint32_t lane_id)
{
    return kPlIncPayloadOff + static_cast<uint64_t>(lane_id) * kQv2RingDepth * kPlMaxTokenBytes;
}

__aicore__ inline uint64_t DevPlIncDescLaneOff(uint32_t lane_id)
{
    return kPlIncDescOff + static_cast<uint64_t>(lane_id) * kQv2RingDepth * kPlDescriptorBytes;
}

__aicore__ inline uint64_t DevPlDestChannelDescOff(uint32_t source_id)
{
    return kPlDestChannelDescOff +
           static_cast<uint64_t>(source_id) * kQv2RingDepth * kPlDescriptorBytes;
}

__aicore__ inline uint64_t DevPlDestChannelPayloadOff(uint32_t source_id)
{
    return kPlDestChannelPayloadOff +
           static_cast<uint64_t>(source_id) * kQv2RingDepth * kPlMaxTokenBytes;
}

// M4：Worker local input（绝对 GM 指针）batch put
__aicore__ inline void PlIssuePayloadBatchGmSrc(__gm__ uint8_t *sym, __gm__ uint8_t *worker_src_gm,
                                               uint64_t inc_lane_payload_base, uint64_t lane_seq,
                                               uint32_t batch_tokens, uint32_t nbytes, uint32_t depth, int peer,
                                               uint64_t &puts, uint64_t &mte_waits, uint32_t &wrap_split_puts)
{
    const uint32_t slot0 = static_cast<uint32_t>(lane_seq % depth);
    const uint32_t until_wrap = depth - slot0;
    const uint64_t range_bytes = static_cast<uint64_t>(batch_tokens) * nbytes;
    if (batch_tokens <= until_wrap) {
        const uint64_t dst = inc_lane_payload_base + static_cast<uint64_t>(slot0) * nbytes;
        inc::dc::dn::qv2::Qv2IssuePayloadRangeP6NoDrainGmSrc(sym, worker_src_gm, dst, static_cast<uint32_t>(range_bytes), peer, puts,
                                           mte_waits);
    } else {
        const uint32_t bytes1 = until_wrap * nbytes;
        const uint32_t bytes2 = static_cast<uint32_t>(range_bytes) - bytes1;
        const uint64_t dst1 = inc_lane_payload_base + static_cast<uint64_t>(slot0) * nbytes;
        inc::dc::dn::qv2::Qv2IssuePayloadRangeP6NoDrainGmSrc(sym, worker_src_gm, dst1, bytes1, peer, puts, mte_waits);
        inc::dc::dn::qv2::Qv2IssuePayloadRangeP6NoDrainGmSrc(sym, worker_src_gm + bytes1, inc_lane_payload_base, bytes2, peer, puts,
                                          mte_waits);
        ++wrap_split_puts;
    }
}

// Pipeline 128B descriptor tile put（sym+offset 源；禁止误用 qv2 的 64B stride）
__aicore__ inline void PlIssueDescriptorTileP6NoDrainOff(__gm__ uint8_t *sym, uint64_t worker_desc_off,
                                                         uint64_t inc_lane_desc_base, uint64_t lane_seq,
                                                         uint32_t tile_tokens, uint32_t depth, int peer,
                                                         uint64_t &puts, uint64_t &mte_waits, uint32_t &wrap_split_puts)
{
    const uint32_t slot0 = static_cast<uint32_t>(lane_seq % depth);
    const uint32_t until_wrap = depth - slot0;
    const uint32_t desc_bytes = tile_tokens * kPlDescriptorBytes;
    if (tile_tokens <= until_wrap) {
        const uint64_t dst = inc_lane_desc_base + static_cast<uint64_t>(slot0) * kPlDescriptorBytes;
        Qv2IssueDescriptorRangeP6NoDrain(sym, worker_desc_off, dst, desc_bytes, peer, puts, mte_waits);
    } else {
        const uint32_t bytes1 = until_wrap * kPlDescriptorBytes;
        const uint32_t bytes2 = desc_bytes - bytes1;
        const uint64_t dst1 = inc_lane_desc_base + static_cast<uint64_t>(slot0) * kPlDescriptorBytes;
        Qv2IssueDescriptorRangeP6NoDrain(sym, worker_desc_off, dst1, bytes1, peer, puts, mte_waits);
        Qv2IssueDescriptorRangeP6NoDrain(sym, worker_desc_off + bytes1, inc_lane_desc_base, bytes2, peer, puts,
                                         mte_waits);
        ++wrap_split_puts;
    }
}

// Pipeline 128B descriptor tile put（M4：worker_desc_off 可为绝对 GM 指针）
__aicore__ inline void PlIssueDescriptorTileP6NoDrain(__gm__ uint8_t *sym, uint64_t worker_desc_off,
                                                      uint64_t inc_lane_desc_base, uint64_t lane_seq,
                                                      uint32_t tile_tokens, uint32_t depth, int peer,
                                                      uint64_t &puts, uint64_t &mte_waits, uint32_t &wrap_split_puts)
{
    const uint32_t slot0 = static_cast<uint32_t>(lane_seq % depth);
    const uint32_t until_wrap = depth - slot0;
    const uint32_t desc_bytes = tile_tokens * kPlDescriptorBytes;
    __gm__ uint8_t *worker_desc_gm = reinterpret_cast<__gm__ uint8_t *>(worker_desc_off);
    if (tile_tokens <= until_wrap) {
        const uint64_t dst = inc_lane_desc_base + static_cast<uint64_t>(slot0) * kPlDescriptorBytes;
        inc::dc::dn::qv2::Qv2IssueDescriptorRangeP6NoDrainGmSrc(sym, worker_desc_gm, dst, desc_bytes, peer, puts, mte_waits);
    } else {
        const uint32_t bytes1 = until_wrap * kPlDescriptorBytes;
        const uint32_t bytes2 = desc_bytes - bytes1;
        const uint64_t dst1 = inc_lane_desc_base + static_cast<uint64_t>(slot0) * kPlDescriptorBytes;
        inc::dc::dn::qv2::Qv2IssueDescriptorRangeP6NoDrainGmSrc(sym, worker_desc_gm, dst1, bytes1, peer, puts, mte_waits);
        inc::dc::dn::qv2::Qv2IssueDescriptorRangeP6NoDrainGmSrc(sym, worker_desc_gm + bytes1, inc_lane_desc_base, bytes2, peer, puts,
                                              mte_waits);
        ++wrap_split_puts;
    }
}

__aicore__ inline uint64_t DevPlIncEgressCreditLineOff(uint32_t dest_rank)
{
    return kPlIncEgressCreditOff + static_cast<uint64_t>(dest_rank) * 64u;
}

__aicore__ inline uint64_t DevPlIncEgressPublishScratchOff(uint32_t lane_id)
{
    return kPlIncEgressPublishScratchOff + static_cast<uint64_t>(lane_id) * 64u;
}

__aicore__ inline uint64_t DevPlDestCreditPublishScratchOff(uint32_t source_id)
{
    return kPlDestCreditPublishScratchOff + static_cast<uint64_t>(source_id) * 64u;
}

__aicore__ inline uint64_t DevPlLaneTokenBase(uint32_t lane_id, uint32_t tokens_per_lane)
{
    return static_cast<uint64_t>(lane_id) * tokens_per_lane;
}

__aicore__ inline void PlDcciDoneLine(__gm__ uint8_t *p)
{
    PlDcci(p);
}

__aicore__ inline void PlDcciTrace128(__gm__ uint8_t *p)
{
    Qv2DcciRange(p, 128u);
}

__aicore__ inline void PlDcciTrace256(__gm__ uint8_t *p)
{
    Qv2DcciRange(p, 256u);
}

__aicore__ inline void PlDcciTrace384(__gm__ uint8_t *p)
{
    Qv2DcciRange(p, 384u);
}

// 每 epoch 入口：完整清零 128B trace（magic=0），字段填完后再写 magic + PlDcciTrace128
__aicore__ inline void PlClearTrace128(__gm__ uint8_t *p)
{
    for (uint32_t i = 0; i < 128u; i += 8u) {
        *reinterpret_cast<__gm__ uint64_t *>(p + static_cast<uint64_t>(i)) = 0u;
    }
    PlDcciTrace128(p);
}

__aicore__ inline void PlClearTrace256(__gm__ uint8_t *p)
{
    for (uint32_t i = 0; i < 256u; i += 8u) {
        *reinterpret_cast<__gm__ uint64_t *>(p + static_cast<uint64_t>(i)) = 0u;
    }
    PlDcciTrace256(p);
}

__aicore__ inline void PlClearTrace384(__gm__ uint8_t *p)
{
    for (uint32_t i = 0; i < 384u; i += 8u) {
        *reinterpret_cast<__gm__ uint64_t *>(p + static_cast<uint64_t>(i)) = 0u;
    }
    PlDcciTrace384(p);
}

__aicore__ inline void PlPublishAivEpochTiming(__gm__ uint8_t *sym, uint64_t timing_off, uint64_t first_cycle,
                                               uint64_t done_cycle, uint64_t epoch, uint32_t block_id)
{
    __gm__ PlAivEpochTimingLine *line = reinterpret_cast<__gm__ PlAivEpochTimingLine *>(sym + timing_off);
    line->first_issue_cycle = first_cycle;
    line->done_cycle = done_cycle;
    line->done_epoch = epoch;
    line->block_id = block_id;
    line->magic = kPlMagic;
    PlDcciDoneLine(reinterpret_cast<__gm__ uint8_t *>(line));
}

__aicore__ inline uint64_t DevPlUploadAivTimingOff(uint32_t lane_id)
{
    return kPlUploadAivTimingOff + static_cast<uint64_t>(lane_id) * 64u;
}

__aicore__ inline uint64_t DevPlRecvAivTimingOff(uint32_t source_id)
{
    return kPlRecvAivTimingOff + static_cast<uint64_t>(source_id) * 64u;
}

__aicore__ inline uint64_t DevPlForwardDoneLineOff(uint32_t lane_id)
{
    return kPlForwardDoneOff + static_cast<uint64_t>(lane_id) * 64u;
}

__aicore__ inline uint64_t DevPlForwardAivTimingOff(uint32_t lane_id)
{
    return kPlForwardAivTimingOff + static_cast<uint64_t>(lane_id) * 64u;
}

__aicore__ inline uint64_t DevPlFirstHopDoneLineOff(uint32_t lane_id)
{
    return kPlFirstHopDoneOff + static_cast<uint64_t>(lane_id) * 64u;
}

__aicore__ inline uint64_t DevPlTokenReadyLineOff(uint32_t token_index)
{
    return kPlTokenReadyOff + static_cast<uint64_t>(token_index) * 64u;
}

__aicore__ inline uint64_t DevPlIncForwardLaneStageTimingOff(uint32_t lane_id)
{
    return kPlIncForwardLaneStageTimingOff + static_cast<uint64_t>(lane_id) * 256u;
}

__aicore__ inline uint64_t DevPlDestinationCompletionTraceOff()
{
    return kPlDestinationCompletionTraceOff;
}

__aicore__ inline uint64_t DevPlRecvChannelCompletionTraceOff(uint32_t source_id)
{
    return kPlRecvChannelCompletionTraceOff + static_cast<uint64_t>(source_id) * 384u;
}

// PlDescriptor=128B：必须刷两行 cacheline；只 DCCI 首行会导致 ring_slot 等二行字段永久陈旧
__aicore__ inline void PlDcciDescriptor(__gm__ PlDescriptor *d)
{
    Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(d), kPlDescriptorBytes);
}

// generation mismatch：立即 fail-closed（不与 epoch-stale 混计，不空转 64 次）
__aicore__ inline bool PlReadDescriptorWithRetry(__gm__ PlDescriptor *d, uint64_t epoch, uint64_t expect_seq,
                                                 uint32_t expect_slot, uint64_t &stale, uint64_t &dup,
                                                 uint64_t &lost, uint64_t &gen_mismatch)
{
    for (uint32_t di = 0; di < 64u; ++di) {
        AscendC::PipeBarrier<PIPE_ALL>();
        PlDcciDescriptor(d);
        if (d->epoch != epoch) {
            ++stale;
            continue;
        }
        if (d->generation != static_cast<uint32_t>(epoch)) {
            ++gen_mismatch;
            return false;
        }
        if (d->lane_sequence < expect_seq) {
            ++dup;
            continue;
        }
        if (d->lane_sequence > expect_seq) {
            ++lost;
            return false;
        }
        if (d->ring_slot != expect_slot) {
            continue;
        }
        return true;
    }
    return false;
}

// Pipeline 专用 egress tail：整行 putmem_signal，doorbell 对齐 cumulative tail（勿用于 qv2 campaign）
__aicore__ inline void PlPublishEgressTailSignaled(__gm__ Qv2TailLine *tail_line, __gm__ uint8_t *sym,
                                                   uint64_t remote_tail_off, int peer, uint64_t tail_val,
                                                   uint64_t epoch_tag, uint64_t &tail_pub, uint64_t &tail_quiet)
{
    tail_line->tail = tail_val;
    tail_line->epoch_tag = epoch_tag;
    // 本 workload tail≪2^31；超界时 clamp，recv 侧 GE(seen+1) 仍与 tail 单调一致
    constexpr uint64_t kNotifyClamp = 0x7FFFFFFFu;
    const int32_t notify_val =
        (tail_val > kNotifyClamp) ? static_cast<int32_t>(kNotifyClamp) : static_cast<int32_t>(tail_val);
    tail_line->notify_seq = notify_val;
    AscendC::PipeBarrier<PIPE_ALL>();
    Qv2DcciLine(reinterpret_cast<__gm__ uint8_t *>(tail_line));
    __gm__ uint8_t *remote_tail = sym + remote_tail_off;
    __gm__ int32_t *notify_sig =
        reinterpret_cast<__gm__ int32_t *>(remote_tail + inc::dc::dn::qv2::kQv2TailNotifySeqOff);
    aclshmem_putmem_signal_nbi(remote_tail, reinterpret_cast<__gm__ void *>(tail_line), sizeof(Qv2TailLine),
                               notify_sig, notify_val, ACLSHMEM_SIGNAL_SET, peer);
    ++tail_pub;
    aclshmem_quiet();
    ++tail_quiet;
}

__aicore__ inline void PlPublishEgressTail(__gm__ Qv2TailLine *tail_line, __gm__ uint8_t *sym, uint64_t remote_tail_off,
                                          int peer, uint64_t tail_val, uint64_t epoch_tag, uint64_t &tail_pub,
                                          uint64_t &tail_quiet)
{
    PlPublishEgressTailSignaled(tail_line, sym, remote_tail_off, peer, tail_val, epoch_tag, tail_pub, tail_quiet);
}

// Pipeline 专用 destination→INC credit head：整行 putmem_signal；勿用于 ingress head
__aicore__ inline void PlPublishEgressHeadSignaled(__gm__ Qv2HeadLine *head_line, __gm__ uint8_t *sym,
                                                   uint64_t remote_head_off, int peer, uint64_t head_val,
                                                   uint64_t epoch_tag, uint64_t &head_pub, bool quiet_now,
                                                   uint64_t &head_quiet)
{
    head_line->head = head_val;
    head_line->epoch_tag = epoch_tag;
    constexpr uint64_t kNotifyClamp = 0x7FFFFFFFu;
    const int32_t notify_val =
        (head_val > kNotifyClamp) ? static_cast<int32_t>(kNotifyClamp) : static_cast<int32_t>(head_val);
    head_line->notify_seq = notify_val;
    AscendC::PipeBarrier<PIPE_ALL>();
    Qv2DcciLine(reinterpret_cast<__gm__ uint8_t *>(head_line));
    __gm__ uint8_t *remote_head = sym + remote_head_off;
    __gm__ int32_t *notify_sig =
        reinterpret_cast<__gm__ int32_t *>(remote_head + inc::dc::dn::qv2::kQv2HeadNotifySeqOff);
    aclshmem_putmem_signal_nbi(remote_head, reinterpret_cast<__gm__ void *>(head_line), sizeof(Qv2HeadLine),
                               notify_sig, notify_val, ACLSHMEM_SIGNAL_SET, peer);
    ++head_pub;
    if (quiet_now) {
        aclshmem_quiet();
        ++head_quiet;
    }
}

__aicore__ inline void PlPublishEgressHead(__gm__ Qv2HeadLine *head_line, __gm__ uint8_t *sym, uint64_t remote_head_off,
                                           int peer, uint64_t head_val, uint64_t epoch_tag, uint64_t &head_pub,
                                           uint64_t &head_quiet)
{
    head_line->head = head_val;
    head_line->epoch_tag = epoch_tag;
    AscendC::PipeBarrier<PIPE_ALL>();
    Qv2DcciLine(reinterpret_cast<__gm__ uint8_t *>(head_line));
    aclshmem_putmem_nbi(reinterpret_cast<__gm__ uint8_t *>(sym + remote_head_off),
                        reinterpret_cast<__gm__ uint8_t *>(&head_line->head), sizeof(uint64_t), peer);
    ++head_pub;
    aclshmem_quiet();
    ++head_quiet;
}

__aicore__ inline void PlRefreshIngressHeadFromInc(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                   __gm__ Qv2HeadLine *head_line, uint32_t lane_id,
                                                   __gm__ PlUploadCounters *uc)
{
    (void)sym;
    (void)desc;
    (void)lane_id;
    AscendC::PipeBarrier<PIPE_ALL>();
    Qv2DcciLine(reinterpret_cast<__gm__ uint8_t *>(head_line));
    if (head_line->head > uc->cached_remote_head) {
        uc->cached_remote_head = head_line->head;
    }
}

} // namespace inc::dc::dn::pl
