/**
 * Queue-v2 device 辅助：DCCI / tail-head publish / P6 transport（与 D1 指令序一致，勿改语义）。
 */
#pragma once

#include "kernel_operator.h"
#include "shmem.h"
#include "shmemi_device_common.h"
#include "host/shmem_host_def.h"
#include "inc_dc_dn_transport_abi.h"
#include "inc_dc_dn_queue_v2_abi.h"

namespace inc::dc::dn::qv2 {

using inc::dc::dn::kD1PrimP4RangeQuiet;
using inc::dc::dn::kD1PrimP6RangeMteWaitFinalQuiet;
using inc::dc::dn::kD1QuietBatch;

__aicore__ inline uint8_t Qv2PatternByteDevice(uint32_t pair_id, uint32_t token_seq, uint32_t byte_off)
{
    return static_cast<uint8_t>((pair_id * 31u + token_seq * 17u + byte_off) & 0xFFu);
}

__aicore__ inline void Qv2DcciLine(__gm__ uint8_t *p)
{
    dcci_cacheline(p);
}

__aicore__ inline void Qv2DcciRange(__gm__ uint8_t *p, uint32_t nbytes)
{
    const uint32_t lines = (nbytes + 63u) / 64u;
    for (uint32_t i = 0; i < lines; ++i) {
        dcci_cacheline(p + static_cast<uint64_t>(i) * 64u);
    }
}

// Qv2LaneCounters=128B：刷新全部 2 条 cacheline
__aicore__ inline void Qv2DcciLaneCounters(__gm__ camp::Qv2LaneCounters *lc)
{
    __gm__ uint8_t *p = reinterpret_cast<__gm__ uint8_t *>(lc);
    Qv2DcciLine(p);
    Qv2DcciLine(p + 64u);
}

// Qv2RouteCounters=256B：刷新全部 4 条 cacheline
__aicore__ inline void Qv2DcciRouteCounters(__gm__ camp::Qv2RouteCounters *rc)
{
    __gm__ uint8_t *p = reinterpret_cast<__gm__ uint8_t *>(rc);
    for (uint32_t i = 0; i < 4u; ++i) {
        Qv2DcciLine(p + static_cast<uint64_t>(i) * 64u);
    }
}

// 与 D1 ingress transport kernel 相同实现
__aicore__ inline void Qv2MteUbComplete(int peer, uint64_t &mte_waits)
{
    __gm__ aclshmem_device_host_state_t *st = aclshmemi_get_state();
    if ((st->topo_list[peer] & ACLSHMEM_TRANSPORT_MTE) == 0u) {
        return;
    }
    const AscendC::TEventID sync_id = static_cast<AscendC::TEventID>(st->mte_config.sync_id);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(sync_id);
    ++mte_waits;
}

__aicore__ inline void Qv2RunRangeP6(__gm__ uint8_t *sym, uint64_t source_off, uint64_t dest_off, uint32_t token_begin,
                                     uint32_t token_end, uint32_t nbytes, uint32_t batch_tokens, int peer,
                                     uint64_t &puts, uint64_t &quiets, uint64_t &mte_waits)
{
    __gm__ aclshmem_device_host_state_t *st = aclshmemi_get_state();
    const bool is_mte = (st->topo_list[peer] & ACLSHMEM_TRANSPORT_MTE) != 0u;
    const uint32_t batch = batch_tokens > 0 ? batch_tokens : kD1QuietBatch;
    for (uint32_t t = token_begin; t < token_end;) {
        const uint32_t ntok = (t + batch > token_end) ? (token_end - t) : batch;
        const uint64_t off = static_cast<uint64_t>(t) * nbytes;
        const uint32_t range_bytes = ntok * nbytes;
        aclshmem_putmem_nbi(sym + dest_off + off, sym + source_off + off, range_bytes, peer);
        ++puts;
        if (is_mte) {
            Qv2MteUbComplete(peer, mte_waits);
        } else {
            aclshmem_quiet();
            ++quiets;
        }
        t += ntok;
    }
    if (is_mte) {
        aclshmem_quiet();
        ++quiets;
    }
}

__aicore__ inline void Qv2RunRangeP4(__gm__ uint8_t *sym, uint64_t source_off, uint64_t dest_off, uint32_t token_begin,
                                     uint32_t token_end, uint32_t nbytes, uint32_t batch_tokens, int peer,
                                     uint64_t &puts, uint64_t &quiets)
{
    const uint32_t batch = batch_tokens > 0 ? batch_tokens : kD1QuietBatch;
    for (uint32_t t = token_begin; t < token_end;) {
        const uint32_t ntok = (t + batch > token_end) ? (token_end - t) : batch;
        const uint64_t off = static_cast<uint64_t>(t) * nbytes;
        const uint32_t range_bytes = ntok * nbytes;
        aclshmem_putmem_nbi(sym + dest_off + off, sym + source_off + off, range_bytes, peer);
        ++puts;
        aclshmem_quiet();
        ++quiets;
        t += ntok;
    }
}

__aicore__ inline void Qv2PublishTail(__gm__ Qv2TailLine *tail_line, __gm__ uint8_t *sym, uint64_t remote_tail_off,
                                        int peer, uint64_t tail_val, uint64_t epoch_tag, __gm__ Qv2Telemetry *tel)
{
    tail_line->tail = tail_val;
    tail_line->epoch_tag = epoch_tag;
    AscendC::PipeBarrier<PIPE_ALL>();
    Qv2DcciLine(reinterpret_cast<__gm__ uint8_t *>(tail_line));
    aclshmem_putmem_nbi(reinterpret_cast<__gm__ uint8_t *>(sym + remote_tail_off),
                        reinterpret_cast<__gm__ uint8_t *>(&tail_line->tail), sizeof(uint64_t), peer);
    ++tel->tail_publish_count;
    aclshmem_quiet();
    ++tel->tail_completion_count;
}

__aicore__ inline void Qv2PublishHead(__gm__ Qv2HeadLine *head_line, __gm__ uint8_t *sym, uint64_t remote_head_off,
                                      int peer, uint64_t head_val, uint64_t epoch_tag, __gm__ Qv2Telemetry *tel)
{
    head_line->head = head_val;
    head_line->epoch_tag = epoch_tag;
    AscendC::PipeBarrier<PIPE_ALL>();
    Qv2DcciLine(reinterpret_cast<__gm__ uint8_t *>(head_line));
    aclshmem_putmem_nbi(reinterpret_cast<__gm__ uint8_t *>(sym + remote_head_off),
                        reinterpret_cast<__gm__ uint8_t *>(&head_line->head), sizeof(uint64_t), peer);
    ++tel->head_publish_count;
    aclshmem_quiet();
}

__aicore__ inline bool Qv2CheckPayload(__gm__ uint8_t *p, uint32_t nbytes, uint32_t pair_id, uint32_t token_seq)
{
    for (uint32_t i = 0; i < nbytes; ++i) {
        if (p[i] != Qv2PatternByteDevice(pair_id, token_seq, i)) {
            return false;
        }
    }
    return true;
}

__aicore__ inline void Qv2PutPayloadChunk(__gm__ uint8_t *base, uint64_t local_src, uint64_t local_dst,
                                          uint32_t chunk_tokens, uint32_t nbytes, int peer, bool use_p6,
                                          uint64_t &puts, uint64_t &quiets, uint64_t &mte_payload)
{
    if (use_p6) {
        Qv2RunRangeP6(base, local_src, local_dst, 0, chunk_tokens, nbytes, chunk_tokens, peer, puts, quiets,
                      mte_payload);
    } else {
        Qv2RunRangeP4(base, local_src, local_dst, 0, chunk_tokens, nbytes, chunk_tokens, peer, puts, quiets);
    }
}

// ----- D2.1 tile-level issue path（不修改上方 D2.0 reference）-----

__aicore__ inline void Qv2IssuePayloadRangeP6NoDrain(__gm__ uint8_t *sym, uint64_t src_off, uint64_t dst_off,
                                                    uint32_t range_bytes, int peer, uint64_t &puts,
                                                    uint64_t &mte_waits)
{
    aclshmem_putmem_nbi(sym + dst_off, sym + src_off, range_bytes, peer);
    ++puts;
    Qv2MteUbComplete(peer, mte_waits);
}

// M4：Worker local aclrtMalloc 源（绝对 GM 指针，非 sym 偏移）
__aicore__ inline void Qv2IssuePayloadRangeP6NoDrainGmSrc(__gm__ uint8_t *sym, __gm__ uint8_t *src_gm,
                                                         uint64_t dst_off, uint32_t range_bytes, int peer,
                                                         uint64_t &puts, uint64_t &mte_waits)
{
    aclshmem_putmem_nbi(sym + dst_off, src_gm, range_bytes, peer);
    ++puts;
    Qv2MteUbComplete(peer, mte_waits);
}

__aicore__ inline void Qv2IssueDescriptorRangeP6NoDrain(__gm__ uint8_t *sym, uint64_t src_off, uint64_t dst_off,
                                                        uint32_t range_bytes, int peer, uint64_t &puts,
                                                        uint64_t &mte_waits)
{
    aclshmem_putmem_nbi(sym + dst_off, sym + src_off, range_bytes, peer);
    ++puts;
    Qv2MteUbComplete(peer, mte_waits);
}

__aicore__ inline void Qv2IssueDescriptorRangeP6NoDrainGmSrc(__gm__ uint8_t *sym, __gm__ uint8_t *src_gm,
                                                             uint64_t dst_off, uint32_t range_bytes, int peer,
                                                             uint64_t &puts, uint64_t &mte_waits)
{
    aclshmem_putmem_nbi(sym + dst_off, src_gm, range_bytes, peer);
    ++puts;
    Qv2MteUbComplete(peer, mte_waits);
}

__aicore__ inline void Qv2DrainTileDataAndDescriptor(uint64_t &drain_quiet_count)
{
    aclshmem_quiet();
    ++drain_quiet_count;
}

__aicore__ inline void Qv2PublishTailOrdered(__gm__ Qv2TailLine *tail_line, __gm__ uint8_t *sym,
                                             uint64_t remote_tail_off, int peer, uint64_t tail_val, uint64_t epoch_tag,
                                             uint64_t &tail_publish_count, uint64_t &tail_completion_quiet_count)
{
    tail_line->tail = tail_val;
    tail_line->epoch_tag = epoch_tag;
    AscendC::PipeBarrier<PIPE_ALL>();
    Qv2DcciLine(reinterpret_cast<__gm__ uint8_t *>(tail_line));
    aclshmem_putmem_nbi(reinterpret_cast<__gm__ uint8_t *>(sym + remote_tail_off),
                        reinterpret_cast<__gm__ uint8_t *>(&tail_line->tail), sizeof(uint64_t), peer);
    ++tail_publish_count;
    aclshmem_quiet();
    ++tail_completion_quiet_count;
}

__aicore__ inline void Qv2PublishHeadRangeOrdered(__gm__ uint8_t *sym, uint64_t local_heads_off,
                                                  uint64_t remote_heads_off, uint32_t range_bytes, int peer,
                                                  uint64_t &head_publish_count, uint64_t &head_completion_quiet_count)
{
    aclshmem_putmem_nbi(sym + remote_heads_off, sym + local_heads_off, range_bytes, peer);
    ++head_publish_count;
    aclshmem_quiet();
    ++head_completion_quiet_count;
}

// logical batch：支持 ring wrap 拆成最多 2 个 physical put，不 drain
__aicore__ inline void Qv2IssuePayloadBatchP6NoDrain(__gm__ uint8_t *sym, uint64_t worker_src_off,
                                                      uint64_t inc_lane_payload_base, uint64_t lane_seq,
                                                      uint32_t batch_tokens, uint32_t nbytes, uint32_t depth,
                                                      int peer, uint64_t &puts, uint64_t &mte_waits,
                                                      uint32_t &wrap_split_puts)
{
    const uint32_t slot0 = static_cast<uint32_t>(lane_seq % depth);
    const uint32_t until_wrap = depth - slot0;
    const uint64_t range_bytes = static_cast<uint64_t>(batch_tokens) * nbytes;
    if (batch_tokens <= until_wrap) {
        const uint64_t dst = inc_lane_payload_base + static_cast<uint64_t>(slot0) * nbytes;
        Qv2IssuePayloadRangeP6NoDrain(sym, worker_src_off, dst, static_cast<uint32_t>(range_bytes), peer, puts,
                                      mte_waits);
    } else {
        const uint32_t bytes1 = until_wrap * nbytes;
        const uint32_t bytes2 = static_cast<uint32_t>(range_bytes) - bytes1;
        const uint64_t dst1 = inc_lane_payload_base + static_cast<uint64_t>(slot0) * nbytes;
        Qv2IssuePayloadRangeP6NoDrain(sym, worker_src_off, dst1, bytes1, peer, puts, mte_waits);
        Qv2IssuePayloadRangeP6NoDrain(sym, worker_src_off + bytes1, inc_lane_payload_base, bytes2, peer, puts,
                                      mte_waits);
        ++wrap_split_puts;
    }
}

__aicore__ inline void Qv2IssueDescriptorTileP6NoDrain(__gm__ uint8_t *sym, uint64_t worker_desc_off,
                                                       uint64_t inc_lane_desc_base, uint64_t lane_seq,
                                                       uint32_t tile_tokens, uint32_t depth, int peer,
                                                       uint64_t &puts, uint64_t &mte_waits, uint32_t &wrap_split_puts)
{
    const uint32_t slot0 = static_cast<uint32_t>(lane_seq % depth);
    const uint32_t until_wrap = depth - slot0;
    const uint32_t desc_bytes = tile_tokens * kQv2DescriptorBytes;
    if (tile_tokens <= until_wrap) {
        const uint64_t dst = inc_lane_desc_base + static_cast<uint64_t>(slot0) * kQv2DescriptorBytes;
        Qv2IssueDescriptorRangeP6NoDrain(sym, worker_desc_off, dst, desc_bytes, peer, puts, mte_waits);
    } else {
        const uint32_t bytes1 = until_wrap * kQv2DescriptorBytes;
        const uint32_t bytes2 = desc_bytes - bytes1;
        const uint64_t dst1 = inc_lane_desc_base + static_cast<uint64_t>(slot0) * kQv2DescriptorBytes;
        Qv2IssueDescriptorRangeP6NoDrain(sym, worker_desc_off, dst1, bytes1, peer, puts, mte_waits);
        Qv2IssueDescriptorRangeP6NoDrain(sym, worker_desc_off + bytes1, inc_lane_desc_base, bytes2, peer, puts,
                                         mte_waits);
        ++wrap_split_puts;
    }
}

// device 侧布局偏移（abi.h 的 host inline 不可在 AICore 调用）
namespace camp {

__aicore__ inline uint64_t DevCampLaneTokenBase(uint32_t lane_id, uint32_t tokens_per_lane)
{
    return static_cast<uint64_t>(lane_id) * tokens_per_lane;
}

__aicore__ inline uint64_t DevCampIncPayloadLaneOff(uint32_t lane_id)
{
    return kCampIncPayloadOff + static_cast<uint64_t>(lane_id) * kQv2RingDepth * kQv2TokenBytes;
}

__aicore__ inline uint64_t DevCampIncDescLaneOff(uint32_t lane_id)
{
    return kCampIncDescOff + static_cast<uint64_t>(lane_id) * kQv2RingDepth * kQv2DescriptorBytes;
}

__aicore__ inline uint64_t DevCampTailLineOff(uint32_t lane_id)
{
    return kCampTailLineOff + static_cast<uint64_t>(lane_id) * 64u;
}

__aicore__ inline uint64_t DevCampHeadLineOff(uint32_t lane_id)
{
    return kCampHeadLineOff + static_cast<uint64_t>(lane_id) * 64u;
}

__aicore__ inline uint64_t DevCampWorkerTraceOff(uint32_t lane_id)
{
    return kCampWorkerTraceOff + static_cast<uint64_t>(lane_id) * 128u;
}

__aicore__ inline uint64_t DevCampRouteTraceOff(uint32_t route_id)
{
    return kCampRouteTraceOff + static_cast<uint64_t>(route_id) * 128u;
}

__aicore__ inline uint64_t DevCampRouteDoneLineOff(uint32_t route_id)
{
    return kCampRouteDoneLineOff + static_cast<uint64_t>(route_id) * 64u;
}

__aicore__ inline void DevCampTraceDcci(__gm__ Qv2ServiceTraceLine *tr)
{
    Qv2DcciLine(reinterpret_cast<__gm__ uint8_t *>(tr));
}

__aicore__ inline uint64_t DevCampWorkerResidentOff(uint32_t lane_id)
{
    return kCampWorkerResidentOff + static_cast<uint64_t>(lane_id) * 64u;
}

__aicore__ inline uint64_t DevCampRouteResidentOff(uint32_t route_id)
{
    return kCampRouteResidentOff + static_cast<uint64_t>(route_id) * 64u;
}

} // namespace camp

__aicore__ inline bool Qv2ReadDescriptorWithRetry(__gm__ Qv2Descriptor *d, uint64_t expect_seq, uint64_t expect_epoch,
                                                  uint32_t expect_slot, uint64_t &retry_count,
                                                  uint64_t &persistent_stale_count)
{
    for (uint32_t i = 0; i < 64u; ++i) {
        AscendC::PipeBarrier<PIPE_ALL>();
        Qv2DcciLine(reinterpret_cast<__gm__ uint8_t *>(d));
        if (d->lane_sequence == expect_seq && d->epoch == expect_epoch && d->ring_slot == expect_slot) {
            if (i > 0) {
                retry_count += i;
            }
            return true;
        }
        ++retry_count;
    }
    ++persistent_stale_count;
    return false;
}

} // namespace inc::dc::dn::qv2
