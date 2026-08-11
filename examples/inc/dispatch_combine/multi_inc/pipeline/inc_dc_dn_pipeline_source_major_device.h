/**
 * P5.3 Source-Major Ingress — per-lane raw SPSC + route_done_generation + staging SPSC.
 * 正式路径禁止：lane0 汇总 early tail、pending_route_mask 多写 RMW、staging ready-bool、egress 用 n 重发。
 */
#pragma once

#include "inc_dc_dn_pipeline_device.h"
#include "inc_dc_dn_pipeline_full_dispatch_device.h"
#include "inc_dc_gather_mte_aicore.h"

namespace inc::dc::dn::pl {

__aicore__ inline uint64_t DevPlP5CellRouteRecvLedgerLineOff(uint32_t source_rank, uint32_t dest_rank)
{
    return kPlP5CellRouteRecvLedgerOff +
           (static_cast<uint64_t>(source_rank % kPlMaxSources) * kPlMaxSources +
            static_cast<uint64_t>(dest_rank % kPlMaxSources)) *
               64u;
}

__aicore__ inline uint64_t DevPlP5CellGatherLedgerLineOff(uint32_t source_rank, uint32_t dest_rank)
{
    return kPlP5CellGatherLedgerOff +
           (static_cast<uint64_t>(source_rank % kPlMaxSources) * kPlMaxSources +
            static_cast<uint64_t>(dest_rank % kPlMaxSources)) *
               64u;
}

__aicore__ inline uint64_t DevPlP5CellEgressLedgerLineOff(uint32_t source_rank, uint32_t dest_rank)
{
    return kPlP5CellEgressLedgerOff +
           (static_cast<uint64_t>(source_rank % kPlMaxSources) * kPlMaxSources +
            static_cast<uint64_t>(dest_rank % kPlMaxSources)) *
               64u;
}

__aicore__ inline uint64_t DevPlRawLaneTailOff(uint32_t lane)
{
    return kPlRawIngressTailOff + static_cast<uint64_t>(lane % kPlRawUploadLaneCount) * 64u;
}

__aicore__ inline uint64_t DevPlRawLaneHeadOff(uint32_t lane)
{
    return kPlRawIngressHeadOff + static_cast<uint64_t>(lane % kPlRawUploadLaneCount) * 64u;
}

__aicore__ inline uint64_t DevPlRawChunkDescLaneSlotOff(uint32_t lane, uint32_t slot)
{
    const uint32_t L = lane % kPlRawUploadLaneCount;
    const uint32_t s = slot % kPlRawChunkRingDepth;
    return kPlRawChunkDescOff +
           (static_cast<uint64_t>(L) * kPlRawChunkRingDepth + static_cast<uint64_t>(s)) * sizeof(PlRawChunkDesc);
}

__aicore__ inline uint64_t DevPlRouteDoneGenOff(uint32_t upload_lane, uint32_t destination)
{
    const uint32_t L = upload_lane % kPlRawUploadLaneCount;
    const uint32_t d = destination % kPlMaxSources;
    return kPlRouteDoneGenOff + (static_cast<uint64_t>(L) * kPlMaxSources + static_cast<uint64_t>(d)) * 64u;
}

__aicore__ inline uint64_t DevPlIncP5StagingTailOff(uint32_t dest_lane)
{
    return kPlIncP5StagingTailOff + static_cast<uint64_t>(dest_lane % kQv2LaneCount) * 64u;
}

__aicore__ inline uint64_t DevPlIncP5StagingHeadOff(uint32_t dest_lane)
{
    return kPlIncP5StagingHeadOff + static_cast<uint64_t>(dest_lane % kQv2LaneCount) * 64u;
}

__aicore__ inline uint64_t DevPlIncP5EgressDoneOff(uint32_t dest_lane)
{
    return kPlIncP5EgressDoneOff + static_cast<uint64_t>(dest_lane % kQv2LaneCount) * 64u;
}

__aicore__ inline uint64_t DevPlIncP5ExtResidentLineOff(uint32_t block_id)
{
    const uint32_t idx = (block_id >= 10u) ? (block_id - 10u) : 0u;
    return kPlIncP5ExtResidentOff + static_cast<uint64_t>(idx) * 64u;
}

__aicore__ inline uint64_t DevPlIncP5ExtTraceLineOff(uint32_t block_id)
{
    const uint32_t idx = (block_id >= 10u) ? (block_id - 10u) : 0u;
    return kPlIncP5ExtTraceOff + static_cast<uint64_t>(idx) * 128u;
}

__aicore__ inline bool PlIsSourceMajorRaw(__gm__ PlPipelineDesc *desc)
{
    return (desc->pipeline_mode & kPlIngressLayoutSourceMajorRaw) != 0u;
}

// E0：显式 typed epoch read —— 返回 valid；禁止泛化 EpochLocalValue；禁止只返回数值
// valid=false：magic/epoch 不匹配（旧 epoch 或未初始化）→ 调用方逻辑值按 0 + stale++
// 同 epoch 且 head>tail：由 SafeOccupancy / 调用方报协议错误，禁止静默改 0

__aicore__ inline bool ReadRawCtrlEpochValue(__gm__ PlRawLaneCtrlLine *line, uint64_t epoch, uint64_t &out_value)
{
    if (line->magic == kPlRawLaneCtrlMagic && line->epoch == epoch) {
        out_value = line->value;
        return true;
    }
    out_value = 0u;
    return false;
}

__aicore__ inline bool ReadStagingCtrlEpochValue(__gm__ PlStagingCtrlLine *line, uint64_t epoch,
                                                  uint64_t &out_value)
{
    if (line->magic == kPlStagingCtrlMagic && line->epoch == epoch) {
        out_value = line->value;
        return true;
    }
    out_value = 0u;
    return false;
}

__aicore__ inline bool ReadEgressHeadEpochValue(__gm__ Qv2HeadLine *line, uint64_t epoch, uint64_t &out_value)
{
    if (line->magic == inc::dc::dn::qv2::kQv2Magic && line->epoch_tag == epoch) {
        out_value = line->head;
        return true;
    }
    out_value = 0u;
    return false;
}

__aicore__ inline bool ReadEgressTailEpochValue(__gm__ Qv2TailLine *line, uint64_t epoch, uint64_t &out_value)
{
    if (line->magic == inc::dc::dn::qv2::kQv2Magic && line->epoch_tag == epoch) {
        out_value = line->tail;
        return true;
    }
    out_value = 0u;
    return false;
}

// tail>=head → occ=tail-head 且 ok；tail<head → 协议错误（同 epoch 非法），禁止 unsigned wrap
__aicore__ inline bool SafeOccupancy(uint64_t tail, uint64_t head, uint64_t &occ)
{
    if (tail < head) {
        occ = 0u;
        return false;
    }
    occ = tail - head;
    return true;
}

__aicore__ inline void PlSourceMajorPublishStagingRange(
    __gm__ uint8_t *sym, uint64_t payload_base, uint64_t desc_base,
    uint64_t begin_seq, uint64_t end_seq, uint32_t depth,
    uint32_t nbytes)
{
    uint64_t seq = begin_seq;
    while (seq < end_seq) {
        const uint32_t slot = static_cast<uint32_t>(seq % depth);
        uint64_t run = end_seq - seq;
        const uint32_t contig = depth - slot;
        if (run > contig) {
            run = contig;
        }
        Qv2DcciRange(
            sym + payload_base + static_cast<uint64_t>(slot) * nbytes,
            static_cast<uint32_t>(run) * nbytes);
        Qv2DcciRange(
            sym + desc_base +
                static_cast<uint64_t>(slot) * kPlDescriptorBytes,
            static_cast<uint32_t>(run) * kPlDescriptorBytes);
        seq += run;
    }
}

__aicore__ inline void PlSourceMajorFlushGatherGroup(
    __gm__ uint8_t *sym, uint64_t payload_base, uint64_t begin_seq,
    uint32_t depth, uint32_t nbytes, GM_ADDR *sources, uint32_t count,
    __ubuf__ uint8_t *ub0, __ubuf__ uint8_t *ub1)
{
    if (count == 0u) {
        return;
    }
    const uint32_t slot = static_cast<uint32_t>(begin_seq % depth);
    __gm__ uint8_t *dst =
        sym + payload_base + static_cast<uint64_t>(slot) * nbytes;
    if (count == 4u && nbytes <= kDcLlGatherUbTileBytes &&
        slot + count <= depth) {
        DcLlGatherFourChunks(
            reinterpret_cast<GM_ADDR>(dst), sources[0], sources[1],
            sources[2], sources[3], nbytes, ub0, ub1);
        return;
    }
    for (uint32_t i = 0u; i < count; ++i) {
        DcLlGatherCopyGm2GmChunked(
            reinterpret_cast<GM_ADDR>(
                dst + static_cast<uint64_t>(i) * nbytes),
            sources[i], nbytes, ub0, kDcLlGatherEvent);
    }
}

__aicore__ inline uint32_t PlP53ReadInjectFlags(__gm__ uint8_t *sym)
{
    __gm__ PlP53InjectLine *inj = reinterpret_cast<__gm__ PlP53InjectLine *>(sym + kPlP53InjectOff);
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(inj));
    if (inj->magic != kPlP53InjectMagic) {
        return 0u;
    }
    return inj->flags;
}

__aicore__ inline uint32_t PlDestinationCsrLowerBoundToken(
    __gm__ PlDestinationCsrEntry *entries, uint32_t begin, uint32_t end,
    uint32_t token)
{
    while (begin < end) {
        const uint32_t mid = begin + (end - begin) / 2u;
        if (static_cast<uint32_t>(entries[mid].source_token_id) < token) {
            begin = mid + 1u;
        } else {
            end = mid;
        }
    }
    return begin;
}

__aicore__ inline void PlWorkerSourceMajorInitLaneWork(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                       __gm__ PlFullDispatchConfig *cfg, uint32_t lane_id,
                                                       uint64_t epoch)
{
    // DEPRECATED for source-major raw upload path：lane_work_off 属 count-forward 合同。
    // 保留函数仅供遗留 dest-major / 诊断调用；P5 source-major RawUpload 不得调用。
    __gm__ PlLaneWorkLine *lw = reinterpret_cast<__gm__ PlLaneWorkLine *>(
        sym + cfg->lane_work_off + static_cast<uint64_t>(lane_id) * 64u);
    const uint32_t lanes = desc->lane_count == 0u ? 1u : desc->lane_count;
    const uint32_t tokens = desc->tokens_per_epoch;
    uint32_t tokens_lane = desc->tokens_per_lane;
    if (tokens_lane == 0u) {
        tokens_lane = tokens / lanes;
        if (lane_id + 1u == lanes) {
            tokens_lane = tokens - lane_id * (tokens / lanes);
        }
    }
    lw->epoch = epoch;
    lw->token_count = tokens_lane;
    lw->gather_count = tokens_lane;
    lw->upload_count = 0u;
    lw->done_epoch = epoch;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(lw));
}

__aicore__ inline void PlSourceMajorLaneTokenRange(__gm__ PlPipelineDesc *desc, uint32_t lane_id,
                                                   uint32_t &tok_begin, uint32_t &tok_end)
{
    const uint32_t tokens = desc->tokens_per_epoch;
    const uint32_t lanes = desc->lane_count == 0u ? 1u : desc->lane_count;
    uint32_t tpl = desc->tokens_per_lane;
    if (tpl == 0u) {
        tpl = tokens / lanes;
    }
    tok_begin = lane_id * tpl;
    if (lane_id + 1u >= lanes) {
        tok_end = tokens;
    } else {
        tok_end = tok_begin + tpl;
    }
    if (tok_end > tokens) {
        tok_end = tokens;
    }
    if (tok_begin > tok_end) {
        tok_begin = tok_end;
    }
}

__aicore__ inline bool PlRawLaneReachedTerminal(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                uint32_t lane_id, uint64_t tail, uint64_t epoch)
{
    if (tail == 0u) {
        return false;
    }
    const uint32_t slot = static_cast<uint32_t>((tail - 1u) % kPlRawChunkRingDepth);
    __gm__ PlRawChunkDesc *cd =
        reinterpret_cast<__gm__ PlRawChunkDesc *>(sym + DevPlRawChunkDescLaneSlotOff(lane_id, slot));
    Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(cd), sizeof(PlRawChunkDesc));
    if (cd->magic != kPlRawChunkDescMagic || cd->epoch != epoch || cd->generation != tail) {
        return false;
    }
    uint32_t expect_begin = 0u;
    uint32_t expect_end = 0u;
    PlSourceMajorLaneTokenRange(desc, lane_id, expect_begin, expect_end);
    if (expect_begin == expect_end) {
        return cd->token_begin == expect_begin && cd->token_count == 0u;
    }
    return cd->token_begin + cd->token_count >= expect_end;
}

__aicore__ inline void PlPublishRawLaneTail(__gm__ uint8_t *sym, uint32_t lane_id, uint64_t generation,
                                            uint64_t epoch, int peer)
{
    __gm__ PlRawLaneCtrlLine *tl =
        reinterpret_cast<__gm__ PlRawLaneCtrlLine *>(sym + DevPlRawLaneTailOff(lane_id));
    tl->value = generation;
    tl->epoch = epoch;
    tl->magic = kPlRawLaneCtrlMagic;
    tl->lane_id = lane_id;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(tl));
    aclshmem_putmem_nbi(sym + DevPlRawLaneTailOff(lane_id), reinterpret_cast<__gm__ uint8_t *>(tl),
                        sizeof(PlRawLaneCtrlLine), peer);
}

__aicore__ inline void PlPublishRawLaneHead(__gm__ uint8_t *sym, uint32_t lane_id, uint64_t head,
                                            uint64_t epoch, int peer)
{
    __gm__ PlRawLaneCtrlLine *hd =
        reinterpret_cast<__gm__ PlRawLaneCtrlLine *>(sym + DevPlRawLaneHeadOff(lane_id));
    hd->value = head;
    hd->epoch = epoch;
    hd->magic = kPlRawLaneCtrlMagic;
    hd->lane_id = lane_id;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(hd));
    aclshmem_putmem_nbi(sym + DevPlRawLaneHeadOff(lane_id), reinterpret_cast<__gm__ uint8_t *>(hd),
                        sizeof(PlRawLaneCtrlLine), peer);
}

__aicore__ inline void PlPublishRouteDone(__gm__ uint8_t *sym, uint32_t upload_lane, uint32_t dest,
                                          uint64_t done_gen, uint64_t epoch)
{
    __gm__ PlRouteDoneGenLine *rd =
        reinterpret_cast<__gm__ PlRouteDoneGenLine *>(sym + DevPlRouteDoneGenOff(upload_lane, dest));
    rd->done_generation = done_gen;
    rd->epoch = epoch;
    rd->magic = kPlRouteDoneGenMagic;
    rd->upload_lane = upload_lane;
    rd->destination = dest;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(rd));
}

// Worker：每 lane 独立 SPSC；tile 受本 lane head credit；发布本 lane cumulative tail。
// Ownership：禁止写 cfg->lane_work_off（lane_work 属 route-count / count-forward 合同，由
// SourcePublishCounts 维护）。本函数仅写 PlUploadCounters / raw ring。
__aicore__ inline void PlWorkerSourceMajorRawUploadEpoch(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                         __gm__ PlFullDispatchConfig *fdc,
                                                         __gm__ PlUploadCounters *uc,
                                                         __gm__ PlServiceTraceLine *trace, uint32_t lane_id,
                                                         uint64_t epoch)
{
    // 不再调用 PlWorkerSourceMajorInitLaneWork —— 避免覆盖 SourcePublishCounts 写入的 count 元数据

    const int peer = static_cast<int>(desc->peer_pe);
    const uint32_t nbytes = desc->payload_bytes;
    const PlEpochRouteView route = PlResolveEpochRouteView(sym, desc, fdc, epoch);
    const uint32_t tokens = route.token_count;
    const uint32_t topk = route.topk;
    const uint32_t epp = route.expert_per_pe;
    uint32_t tok_begin = 0u;
    uint32_t tok_end = 0u;
    PlSourceMajorLaneTokenRange(desc, lane_id, tok_begin, tok_end);
    const uint32_t my_tokens = (tok_end > tok_begin) ? (tok_end - tok_begin) : 0u;
    const uint64_t need = static_cast<uint64_t>(desc->tokens_per_epoch) * static_cast<uint64_t>(nbytes);
    const uint64_t raw_base = PlFullDispatchRawInputBase(fdc, desc, sym, epoch);
    // 本地只读 expert_ids 生成 dest_mask/checksum；packed meta 由 Worker control 单写者 push。
    __gm__ int32_t *eids = reinterpret_cast<__gm__ int32_t *>(route.eid_base);
    const uint32_t inject = PlP53ReadInjectFlags(sym);
    if (route.ok == 0u) {
        uc->error_code = 16u;
        PlDcciUploadCtr(uc);
        return;
    }

    uint64_t payload_puts = 0;
    uint64_t payload_mte = 0;
    uint64_t up_scalar = 0;
    uint64_t up_range = 0;
    uint64_t quiets = 0;
    uint64_t first_chunk_cyc = 0;
    uint64_t credit_wait = 0;

    if (need > kPlIncRawSlabBytes) {
        uc->error_code = 43u;
        PlDcciUploadCtr(uc);
        return;
    }

    // withhold_credit：强制小 tile，使 ring depth 耗尽后走 credit 超时（error 49）
    // CLI batch_tokens 为上限：batch=1 必须强制 1-token/generation（wrap 硬门依赖）
    uint32_t bw_batch = (nbytes == 0u) ? 1u : (kPlRawChunkTargetBytes / nbytes);
    const uint32_t cli_batch = desc->batch_tokens > 0u ? desc->batch_tokens : 8u;
    if (bw_batch < cli_batch) {
        bw_batch = cli_batch;
    }
    if (cli_batch > 0u && bw_batch > cli_batch) {
        bw_batch = cli_batch; // 允许缩小（旧逻辑只升不降 → wrap case 假 local_tail=1）
    }
    if (bw_batch == 0u) {
        bw_batch = 1u;
    }
    if ((inject & kPlP53InjectWithholdCredit) != 0u) {
        bw_batch = 1u;
    }

    uint64_t local_tail = 0u; // published generation（每 epoch 本地从 0；不读远端旧线继承）
    uint32_t t = 0u;
    uint32_t stale_head_epoch = 0u;
    // E6 负例：同 epoch 注入 head>tail → 立即协议错误
    if ((inject & kPlP53InjectSameEpochHeadGtTail) != 0u) {
        uc->error_code = kPlErrEpochLocalHeadGtTail;
        PlDcciUploadCtr(uc);
        return;
    }
    while (t < my_tokens || (my_tokens == 0u && local_tail == 0u)) {
        // credit：tail-head < depth；超时 fail-closed（含 withhold_credit 负例）
        constexpr uint64_t kWaitUs = 2000000ull;
        const uint64_t tw0 = AscendC::GetSystemCycle();
        bool credit_ok = false;
        while (!PlWaitBudgetExceeded(tw0, kWaitUs)) {
            __gm__ PlRawLaneCtrlLine *hd =
                reinterpret_cast<__gm__ PlRawLaneCtrlLine *>(sym + DevPlRawLaneHeadOff(lane_id));
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(hd));
            uint64_t head = 0u;
            if (!ReadRawCtrlEpochValue(hd, epoch, head)) {
                ++stale_head_epoch;
                head = 0u;
            } else if (head > local_tail) {
                // 同 epoch head>tail → 协议错误，禁止静默改 0
                uc->error_code = kPlErrEpochLocalHeadGtTail;
                uc->credit_wait_cycles += credit_wait;
                PlDcciUploadCtr(uc);
                return;
            }
            uint64_t occ = 0u;
            if (!SafeOccupancy(local_tail, head, occ)) {
                uc->error_code = kPlErrEpochLocalHeadGtTail;
                PlDcciUploadCtr(uc);
                return;
            }
            if (occ < static_cast<uint64_t>(kPlRawChunkRingDepth)) {
                credit_ok = true;
                break;
            }
            ++credit_wait;
        }
        if (!credit_ok) {
            uc->error_code = ((inject & kPlP53InjectWithholdCredit) != 0u) ? 49u : 48u;
            PlDcciUploadCtr(uc);
            return;
        }

        const uint32_t bt = (my_tokens == 0u) ? 0u : ((t + bw_batch > my_tokens) ? (my_tokens - t) : bw_batch);
        const uint32_t tile_tok_begin = tok_begin + t;
        const uint64_t next_gen = local_tail + 1u;
        const uint32_t ring_slot = static_cast<uint32_t>(local_tail % kPlRawChunkRingDepth);

        // inject early_tail：在 desc/payload 前发布 tail（应导致 INC 结构化失败，不得假绿）
        if ((inject & kPlP53InjectEarlyTail) != 0u && local_tail == 0u) {
            PlPublishRawLaneTail(sym, lane_id, next_gen, epoch, peer);
            aclshmem_quiet();
            ++quiets;
            uc->error_code = 44u; // early_tail_injected
            PlDcciUploadCtr(uc);
            return;
        }

        if (bt > 0u) {
            __gm__ uint8_t *src = reinterpret_cast<__gm__ uint8_t *>(
                raw_base + static_cast<uint64_t>(tile_tok_begin) * nbytes);
            const uint64_t dst = kPlIncRawSlabOff + static_cast<uint64_t>(tile_tok_begin) * nbytes;
            aclshmem_putmem_nbi(sym + dst, src, bt * nbytes, peer);
            ++payload_puts;
            Qv2MteUbComplete(peer, payload_mte);
            if (bt <= 1u) {
                ++up_scalar;
            } else {
                ++up_range;
            }
            // Route metadata 由 control 单写者整 epoch put；此处禁止分片 put packed buffer。
            if (first_chunk_cyc == 0u) {
                first_chunk_cyc = AscendC::GetSystemCycle();
            }
        }

        uint32_t dest_mask = 0u;
        uint64_t meta_cs = 0u;
        if (bt > 0u) {
            __gm__ int32_t *tile_meta =
                eids + static_cast<uint64_t>(tile_tok_begin) * topk;
            for (uint32_t i = 0u; i < bt; ++i) {
                for (uint32_t s = 0u; s < topk; ++s) {
                    const int32_t eid = eids[static_cast<uint64_t>(tile_tok_begin + i) * topk + s];
                    if (eid < 0) {
                        continue;
                    }
                    uint32_t dest = 0u;
                    uint32_t local_expert = 0u;
                    PlSplitExpertId(static_cast<uint32_t>(eid), epp, dest,
                                    local_expert);
                    if (dest < 32u) {
                        dest_mask |= (1u << dest);
                    }
                }
            }
            // Count control publishes packed expert+ordinal before MetaReady.
            // Hash that exact wire representation so the single verifier can
            // use the fast contiguous FNV path.
            __gm__ PlRawMetaReadyLine *meta_rdy =
                reinterpret_cast<__gm__ PlRawMetaReadyLine *>(
                    sym + kPlRawMetaReadyOff);
            const uint64_t meta_wait_begin = AscendC::GetSystemCycle();
            bool packed_ready = false;
            while (!PlWaitBudgetExceeded(meta_wait_begin, 2000000ull)) {
                PlDcci(reinterpret_cast<__gm__ uint8_t *>(meta_rdy));
                if (meta_rdy->magic == kPlRawMetaReadyMagic &&
                    meta_rdy->epoch == epoch &&
                    meta_rdy->ready_signal == static_cast<int32_t>(epoch)) {
                    packed_ready = true;
                    break;
                }
            }
            if (!packed_ready) {
                uc->error_code = kPlErrRawMetaReadyWait;
                PlDcciUploadCtr(uc);
                return;
            }
            __gm__ int32_t *tile_wire = tile_meta;
            if (topk == 1u) {
                tile_wire = reinterpret_cast<__gm__ int32_t *>(
                    sym + kPlIncRawRouteMetaOff +
                    static_cast<uint64_t>(tile_tok_begin) *
                        sizeof(int32_t));
                Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(tile_wire),
                            bt * sizeof(int32_t));
            }
            meta_cs = PlFnv1a64Int32Range(tile_wire, bt * topk);
        }

        __gm__ PlRawChunkDesc *cd =
            reinterpret_cast<__gm__ PlRawChunkDesc *>(sym + DevPlRawChunkDescLaneSlotOff(lane_id, ring_slot));
        cd->epoch = epoch;
        cd->generation = next_gen;
        if ((inject & kPlP53InjectStaleGen) != 0u && local_tail == 0u) {
            cd->generation = epoch + 999u; // stale
            uc->error_code = 45u;
        }
        cd->source_rank = desc->pair_id;
        cd->token_begin = tile_tok_begin;
        cd->token_count = bt;
        cd->payload_bytes = bt * nbytes;
        cd->route_metadata_offset =
            kPlIncRawRouteMetaOff + static_cast<uint64_t>(tile_tok_begin) * topk * sizeof(int32_t);
        cd->destination_present_mask = dest_mask;
        cd->deprecated_pending_route_mask = 0u;
        cd->magic = kPlRawChunkDescMagic;
        cd->upload_lane = lane_id;
        cd->expect_payload_checksum64 = 0u;
        cd->expect_meta_checksum64 = meta_cs;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(cd));
        aclshmem_putmem_nbi(sym + DevPlRawChunkDescLaneSlotOff(lane_id, ring_slot),
                            reinterpret_cast<__gm__ uint8_t *>(cd), sizeof(PlRawChunkDesc), peer);
        aclshmem_quiet();
        ++quiets;

        local_tail = next_gen;
        PlPublishRawLaneTail(sym, lane_id, local_tail, epoch, peer);
        aclshmem_quiet();
        ++quiets;
        ++uc->tail_publish_count;
        if (local_tail > 1u && (local_tail % kPlRawChunkRingDepth) == 1u) {
            ++uc->wrap_split_put_count; // raw_wrap_count
        }

        if (my_tokens == 0u) {
            break; // zero-token：仍发布 generation=1 empty tile
        }
        t += bt;
    }

    uc->local_tail = local_tail;
    uc->cached_remote_head = stale_head_epoch; // 复用：stale_head_epoch_count 诊断
    uc->payload_put_count += payload_puts;
    uc->payload_mte_wait_count += payload_mte;
    uc->upload_scalar_put_count += up_scalar;
    uc->upload_range_put_count += up_range;
    uc->data_descriptor_drain_quiet_count += quiets;
    uc->first_tail_publish_cycle = first_chunk_cyc;
    uc->upload_batch_count += (my_tokens > 0u) ? ((my_tokens + bw_batch - 1u) / bw_batch) : 1u;
    uc->upload_batch_tokens_sum += my_tokens;
    (void)credit_wait;
    PlDcciUploadCtr(uc);
    if (trace != nullptr) {
        if (trace->first_issue_cycle == 0u) {
            trace->first_issue_cycle = first_chunk_cyc;
        }
        PlTraceDcci(trace);
    }
}

// INC block16：per-lane reclaim；head[L]=min(route_done[L][*]) 且 ≤ tail[L]
// 必须轮询整 epoch：gather 与 reclaim 并行，单次扫过会在 route_done 未就绪时空退。
__aicore__ inline void PlIncSourceMajorReclaimControlEpoch(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                           uint64_t epoch)
{
    const uint32_t inject = PlP53ReadInjectFlags(sym);
    if ((inject & kPlP53InjectWithholdCredit) != 0u) {
        // 结构化：不推进 head；配合 upload bw_batch=1 触发 credit error 49（禁止仅靠 timeout PASS）
        return;
    }
    const uint32_t dest_n = desc->worker_count == 0u ? kQv2LaneCount : desc->worker_count;
    const uint32_t upload_lanes = PlDevUploadLaneCount(desc);
    const int peer = static_cast<int>(desc->peer_pe);
    constexpr uint64_t kWaitUs = 8000000ull;
    const uint64_t t0 = AscendC::GetSystemCycle();
    while (!PlWaitBudgetExceeded(t0, kWaitUs)) {
        bool all_caught = true;
        bool all_terminal = true;
        for (uint32_t L = 0u; L < upload_lanes; ++L) {
            __gm__ PlRawLaneCtrlLine *tl =
                reinterpret_cast<__gm__ PlRawLaneCtrlLine *>(sym + DevPlRawLaneTailOff(L));
            __gm__ PlRawLaneCtrlLine *hd =
                reinterpret_cast<__gm__ PlRawLaneCtrlLine *>(sym + DevPlRawLaneHeadOff(L));
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(tl));
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(hd));
            uint64_t tail = 0u;
            uint64_t head = 0u;
            (void)ReadRawCtrlEpochValue(tl, epoch, tail);
            if (!ReadRawCtrlEpochValue(hd, epoch, head)) {
                head = 0u;
            } else if (head > tail) {
                // 同 epoch head>tail：协议错误（reclaim 侧）
                return;
            }
            if (!PlRawLaneReachedTerminal(sym, desc, L, tail, epoch)) {
                all_terminal = false;
            }
            if (head < tail) {
                all_caught = false;
            }
            while (head < tail) {
                uint64_t min_done = ~0ull;
                for (uint32_t d = 0u; d < dest_n && d < kPlMaxSources; ++d) {
                    __gm__ PlRouteDoneGenLine *rd =
                        reinterpret_cast<__gm__ PlRouteDoneGenLine *>(sym + DevPlRouteDoneGenOff(L, d));
                    PlDcci(reinterpret_cast<__gm__ uint8_t *>(rd));
                    const uint64_t dg =
                        (rd->magic == kPlRouteDoneGenMagic && rd->epoch == epoch) ? rd->done_generation : 0u;
                    if (dg < min_done) {
                        min_done = dg;
                    }
                }
                if (min_done == ~0ull || min_done <= head) {
                    break;
                }
                const uint64_t new_head = (min_done < tail) ? min_done : tail;
                if (new_head <= head) {
                    break;
                }
                head = new_head;
                PlPublishRawLaneHead(sym, L, head, epoch, peer);
                aclshmem_quiet();
            }
        }
        if (all_caught && all_terminal) {
            break;
        }
    }
}

__aicore__ inline void PlIncSourceMajorGatherProducerEpoch(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                           __gm__ PlFullDispatchConfig *fdc,
                                                           __gm__ PlForwardCounters *fc, uint32_t dest_lane,
                                                           uint64_t epoch)
{
    const uint32_t nbytes = desc->payload_bytes;
    const uint32_t topk = desc->route_topk == 0u ? 1u : desc->route_topk;
    const uint32_t epp = desc->expert_per_pe == 0u ? 8u : desc->expert_per_pe;
    const uint32_t depth = desc->ring_depth == 0u ? kQv2RingDepth : desc->ring_depth;
    const uint32_t inject = PlP53ReadInjectFlags(sym);
    const uint32_t tokens_per_src = desc->tokens_per_epoch;
    const uint32_t dest_n = desc->worker_count == 0u ? 1u : desc->worker_count;
    const uint32_t upload_lanes = PlDevUploadLaneCount(desc);
    uint64_t lane_seen[kPlRawUploadLaneCount];
    for (uint32_t i = 0u; i < kPlRawUploadLaneCount; ++i) {
        lane_seen[i] = 0u;
    }
    // E5：只重置本地协议变量；不从远端继承旧 staging（reader 只认当前 epoch）
    uint64_t staging_seq = 0u;
    uint64_t staging_pub = 0u;
    uint32_t staging_wrap = 0u;
    uint32_t staging_publish_coalesce =
        desc->head_tile_tokens > 0u ? desc->head_tile_tokens : 4u;
    const uint32_t staging_publish_cap = depth > 1u ? depth / 2u : 1u;
    if (staging_publish_coalesce > staging_publish_cap) {
        staging_publish_coalesce = staging_publish_cap;
    }
    if (staging_publish_coalesce == 0u) {
        staging_publish_coalesce = 1u;
    }

    __ubuf__ uint8_t *ub = reinterpret_cast<__ubuf__ uint8_t *>(0);
    __ubuf__ uint8_t *ub_alt =
        reinterpret_cast<__ubuf__ uint8_t *>(kDcLlGatherUbTileBytes);
    GM_ADDR gather_sources[4];
    uint32_t gather_pending = 0u;
    uint64_t gather_begin_seq = 0u;
    constexpr uint64_t kWaitUs = 8000000ull;
    const uint64_t t0 = AscendC::GetSystemCycle();

    // withhold_credit：INC gather 立即 fail-closed，避免 Worker recv 空等满 epoch
    if ((inject & kPlP53InjectWithholdCredit) != 0u) {
        fc->error_code = 49u;
        PlDcciForwardCtr(fc);
        return;
    }

    // 先等 Worker control 单写者 RawMetaReady（epoch 必须严格相等，禁止 >=）
    {
        __gm__ PlP53InjectLine *inj_meta =
            reinterpret_cast<__gm__ PlP53InjectLine *>(sym + kPlP53InjectOff);
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(inj_meta));
        if (inj_meta->magic == kPlP53InjectMagic &&
            (inj_meta->flags & kPlP53InjectDropMetaReady) != 0u) {
            // 负例：结构化 fail-closed，禁止 8s 空等冒充 timeout
            fc->error_code = kPlErrRawMetaReadyWait;
            PlDcciForwardCtr(fc);
            return;
        }
        __gm__ PlRawMetaReadyLine *meta_rdy =
            reinterpret_cast<__gm__ PlRawMetaReadyLine *>(sym + kPlRawMetaReadyOff);
        bool meta_ready = false;
        while (!PlWaitBudgetExceeded(t0, kWaitUs)) {
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(meta_rdy));
            if (meta_rdy->magic == kPlRawMetaReadyMagic && meta_rdy->epoch == epoch &&
                meta_rdy->ready_signal == static_cast<int32_t>(epoch)) {
                meta_ready = true;
                break;
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        if (!meta_ready) {
            fc->error_code = kPlErrRawMetaReadyWait;
            PlDcciForwardCtr(fc);
            return;
        }
    }

    const bool use_destination_csr =
        (desc->pipeline_mode & kPlModeDestinationCsrRoutePlan) != 0u;
    __gm__ PlDestinationCsrEntry *csr_entries =
        reinterpret_cast<__gm__ PlDestinationCsrEntry *>(
            sym + kPlIncRawRouteOrdinalOff);
    uint32_t csr_begin = 0u;
    uint32_t csr_end = 0u;
    uint32_t csr_expert_begin[8];
    uint32_t csr_expert_end[8];
    for (uint32_t le = 0u; le < 8u; ++le) {
        csr_expert_begin[le] = 0u;
        csr_expert_end[le] = 0u;
    }
    if (use_destination_csr && dest_lane < dest_n) {
        const uint32_t metadata_bytes =
            tokens_per_src * topk * sizeof(uint32_t);
        __gm__ uint32_t *csr_offsets =
            reinterpret_cast<__gm__ uint32_t *>(
                sym + PlDevDestinationCsrOffsetsOff(metadata_bytes));
        Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(csr_offsets),
                    (dest_n * 8u + 1u) * sizeof(uint32_t));
        const uint32_t cell_begin = dest_lane * 8u;
        csr_begin = csr_offsets[cell_begin];
        csr_end = csr_offsets[cell_begin + epp];
        const uint32_t route_capacity =
            tokens_per_src * topk;
        if (csr_begin > csr_end || csr_end > route_capacity ||
            static_cast<uint64_t>(csr_end) *
                    sizeof(PlDestinationCsrEntry) >
                kPlIncRawRouteOrdinalBytes) {
            fc->error_code = kPlErrOutputCapacityInsufficient;
            PlDcciForwardCtr(fc);
            return;
        }
        for (uint32_t le = 0u; le < epp; ++le) {
            csr_expert_begin[le] = csr_offsets[cell_begin + le];
            csr_expert_end[le] = csr_offsets[cell_begin + le + 1u];
            if (csr_expert_begin[le] > csr_expert_end[le] ||
                csr_expert_begin[le] < csr_begin ||
                csr_expert_end[le] > csr_end) {
                fc->error_code = kPlErrOutputCapacityInsufficient;
                PlDcciForwardCtr(fc);
                return;
            }
        }
        if (csr_end > csr_begin) {
            Qv2DcciRange(
                reinterpret_cast<__gm__ uint8_t *>(
                    csr_entries + csr_begin),
                (csr_end - csr_begin) *
                    sizeof(PlDestinationCsrEntry));
        }
    }

    // 非本 worker dest：只推进 route_done，避免 8 核并发 MTE/UB 冲突
    if (dest_lane >= dest_n) {
        while (!PlWaitBudgetExceeded(t0, kWaitUs)) {
            bool all_caught = true;
            bool all_terminal = true;
            for (uint32_t L = 0u; L < upload_lanes; ++L) {
                __gm__ PlRawLaneCtrlLine *tl =
                    reinterpret_cast<__gm__ PlRawLaneCtrlLine *>(sym + DevPlRawLaneTailOff(L));
                PlDcci(reinterpret_cast<__gm__ uint8_t *>(tl));
                uint64_t tail = 0u;
                (void)ReadRawCtrlEpochValue(tl, epoch, tail);
                while (lane_seen[L] < tail) {
                    const uint64_t gen = lane_seen[L] + 1u;
                    PlPublishRouteDone(sym, L, dest_lane, gen, epoch);
                    lane_seen[L] = gen;
                }
                if (lane_seen[L] < tail) {
                    all_caught = false;
                }
                if (!PlRawLaneReachedTerminal(sym, desc, L, tail, epoch)) {
                    all_terminal = false;
                }
            }
            if (all_caught && all_terminal) {
                break;
            }
        }
        fc->ingress_seen = 0u;
        PlDcciForwardCtr(fc);
        (void)fdc;
        return;
    }

    while (!PlWaitBudgetExceeded(t0, kWaitUs)) {
        bool all_caught = true;
        for (uint32_t L = 0u; L < upload_lanes; ++L) {
            __gm__ PlRawLaneCtrlLine *tl =
                reinterpret_cast<__gm__ PlRawLaneCtrlLine *>(sym + DevPlRawLaneTailOff(L));
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(tl));
            uint64_t tail = 0u;
            (void)ReadRawCtrlEpochValue(tl, epoch, tail);
            if (lane_seen[L] < tail) {
                all_caught = false;
                const uint64_t gen = lane_seen[L] + 1u;
                const uint32_t slot = static_cast<uint32_t>((gen - 1u) % kPlRawChunkRingDepth);
                __gm__ PlRawChunkDesc *cd =
                    reinterpret_cast<__gm__ PlRawChunkDesc *>(sym + DevPlRawChunkDescLaneSlotOff(L, slot));
                bool desc_ready = false;
                while (!PlWaitBudgetExceeded(t0, kWaitUs)) {
                    Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(cd), sizeof(PlRawChunkDesc));
                    if (cd->magic == kPlRawChunkDescMagic && cd->epoch == epoch) {
                        desc_ready = true;
                        break;
                    }
                }
                if (!desc_ready) {
                    fc->error_code = 46u;
                    PlDcciForwardCtr(fc);
                    return;
                }
                if (cd->generation != gen && cd->generation != 0u) {
                    fc->error_code = 47u;
                    PlDcciForwardCtr(fc);
                    return;
                }
                __gm__ uint32_t *route_meta =
                    reinterpret_cast<__gm__ uint32_t *>(
                        sym + cd->route_metadata_offset);
                __gm__ uint8_t *slab = reinterpret_cast<__gm__ uint8_t *>(
                    sym + kPlIncRawSlabOff + static_cast<uint64_t>(cd->token_begin) * nbytes);
                uint32_t observed_dest_mask = 0u;
                if (cd->token_count > 0u) {
                    if (!use_destination_csr || dest_lane == 0u) {
                        Qv2DcciRange(
                            reinterpret_cast<__gm__ uint8_t *>(
                                route_meta),
                            cd->token_count * topk * sizeof(int32_t));
                    }
                    // All destination gather AIVs consume the same immutable
                    // raw metadata chunk.  One deterministic verifier is
                    // sufficient for end-to-end corruption detection; every
                    // lane still validates the destination mask below.
                    if (dest_lane == 0u && cd->expect_meta_checksum64 != 0u) {
                        const uint64_t obs_cs = PlFnv1a64Int32Range(
                            reinterpret_cast<__gm__ const int32_t *>(
                                route_meta),
                            cd->token_count * topk);
                        if (obs_cs != cd->expect_meta_checksum64) {
                            fc->error_code = kPlErrRawMetaChecksum;
                            PlDcciForwardCtr(fc);
                            return;
                        }
                    }
                    if (dest_lane == 0u) {
                        for (uint32_t i = 0u; i < cd->token_count; ++i) {
                            for (uint32_t s = 0u; s < topk; ++s) {
                                const uint64_t route_idx =
                                    static_cast<uint64_t>(i) * topk + s;
                                const uint32_t word =
                                    route_meta[route_idx];
                                if (word == 0xffffffffu) {
                                    continue;
                                }
                                const uint32_t eid = word & 0xffu;
                                uint32_t dest = 0u;
                                uint32_t local_expert = 0u;
                                PlSplitExpertId(eid, epp, dest,
                                                local_expert);
                                if (dest < 32u) {
                                    observed_dest_mask |= (1u << dest);
                                }
                            }
                        }
                        if (observed_dest_mask !=
                            cd->destination_present_mask) {
                            fc->error_code = kPlErrRawMetaDestMask;
                            PlDcciForwardCtr(fc);
                            return;
                        }
                    }
                    // mask 外 destination 禁止进入 staging（diagonal phantom 在此 fail-closed）
                    if (dest_lane < 32u && (cd->destination_present_mask & (1u << dest_lane)) == 0u) {
                        // 本 dest 无路由：仍推进 route_done，但不 staging
                        PlPublishRouteDone(sym, L, dest_lane, gen, epoch);
                        lane_seen[L] = gen;
                        continue;
                    }
                }

                const uint64_t staging_base = DevPlIncPayloadLaneOff(dest_lane);
                const uint64_t staging_desc_base = DevPlIncDescLaneOff(dest_lane);

                const uint32_t expert_slice_count =
                    use_destination_csr ? epp : 1u;
                for (uint32_t expert_slice = 0u;
                     expert_slice < expert_slice_count; ++expert_slice) {
                    uint32_t route_scan_begin = 0u;
                    uint32_t route_scan_end = cd->token_count * topk;
                    if (use_destination_csr) {
                        route_scan_begin =
                            PlDestinationCsrLowerBoundToken(
                                csr_entries,
                                csr_expert_begin[expert_slice],
                                csr_expert_end[expert_slice],
                                cd->token_begin);
                        route_scan_end =
                            PlDestinationCsrLowerBoundToken(
                                csr_entries, route_scan_begin,
                                csr_expert_end[expert_slice],
                                cd->token_begin + cd->token_count);
                    }
                for (uint32_t route_scan = route_scan_begin;
                     route_scan < route_scan_end; ++route_scan) {
                        uint32_t i = 0u;
                        uint32_t s = 0u;
                        uint32_t eid = 0u;
                        uint32_t seg_off = 0u;
                        if (use_destination_csr) {
                            __gm__ PlDestinationCsrEntry *entry =
                                csr_entries + route_scan;
                            const uint32_t token =
                                entry->source_token_id;
                            if (token < cd->token_begin ||
                                token >= cd->token_begin +
                                             cd->token_count) {
                                fc->error_code =
                                    kPlErrRawMetaDestMask;
                                PlDcciForwardCtr(fc);
                                return;
                            }
                            i = token - cd->token_begin;
                            s = entry->topk_slot;
                            eid = entry->expert_id;
                            seg_off = entry->source_segment_offset;
                        } else {
                            i = route_scan / topk;
                            s = route_scan % topk;
                            const uint32_t word =
                                route_meta[route_scan];
                        if (word == 0xffffffffu) {
                            continue;
                        }
                            eid = word & 0xffu;
                            seg_off = word >> 8u;
                        }
                        uint32_t dest = 0u;
                        uint32_t local_expert = 0u;
                        PlSplitExpertId(eid, epp, dest, local_expert);
                        if (dest != dest_lane) {
                            // 若 dest 不在 descriptor mask 内，上面已 fail；此处仅 skip 非本 lane
                            if (dest < 32u && (cd->destination_present_mask & (1u << dest)) == 0u) {
                                fc->error_code = kPlErrRawMetaDestMask;
                                PlDcciForwardCtr(fc);
                                return;
                            }
                            continue;
                        }
                        {
                            __gm__ PlStagingCtrlLine *sh = reinterpret_cast<__gm__ PlStagingCtrlLine *>(
                                sym + DevPlIncP5StagingHeadOff(dest_lane));
                            PlDcci(reinterpret_cast<__gm__ uint8_t *>(sh));
                            uint64_t head = 0u;
                            if (!ReadStagingCtrlEpochValue(sh, epoch, head)) {
                                head = 0u;
                            } else if (head > staging_seq) {
                                fc->error_code = kPlErrEpochLocalStagingOcc;
                                PlDcciForwardCtr(fc);
                                return;
                            }
                            uint64_t occ = 0u;
                            while ((!SafeOccupancy(staging_seq, head, occ) ||
                                    occ >= static_cast<uint64_t>(depth)) &&
                                   !PlWaitBudgetExceeded(t0, kWaitUs)) {
                                if (!SafeOccupancy(staging_seq, head, occ) && head > staging_seq) {
                                    fc->error_code = kPlErrEpochLocalStagingOcc;
                                    PlDcciForwardCtr(fc);
                                    return;
                                }
                                PlDcci(reinterpret_cast<__gm__ uint8_t *>(sh));
                                if (!ReadStagingCtrlEpochValue(sh, epoch, head)) {
                                    head = 0u;
                                } else if (head > staging_seq) {
                                    fc->error_code = kPlErrEpochLocalStagingOcc;
                                    PlDcciForwardCtr(fc);
                                    return;
                                }
                            }
                            if (!SafeOccupancy(staging_seq, head, occ) ||
                                occ >= static_cast<uint64_t>(depth)) {
                                fc->error_code = 48u;
                                PlDcciForwardCtr(fc);
                                return;
                            }
                        }
                        const uint32_t ring_slot = static_cast<uint32_t>(staging_seq % depth);
                        if (staging_seq > 0u && ring_slot == 0u) {
                            ++staging_wrap;
                        }
                        if (gather_pending > 0u && ring_slot == 0u) {
                            PlSourceMajorFlushGatherGroup(
                                sym, staging_base, gather_begin_seq, depth,
                                nbytes, gather_sources, gather_pending, ub,
                                ub_alt);
                            gather_pending = 0u;
                        }
                        if (gather_pending == 0u) {
                            gather_begin_seq = staging_seq;
                        }
                        gather_sources[gather_pending++] =
                            reinterpret_cast<GM_ADDR>(
                                slab + static_cast<uint64_t>(i) * nbytes);
                        __gm__ PlDescriptor *dd = reinterpret_cast<__gm__ PlDescriptor *>(
                            sym + staging_desc_base + static_cast<uint64_t>(ring_slot) * kPlDescriptorBytes);
                        dd->epoch = epoch;
                        dd->generation = static_cast<uint32_t>(epoch);
                        dd->lane_sequence = staging_seq;
                        dd->source_rank = cd->source_rank;
                        dd->source_token_id = cd->token_begin + i;
                        dd->topk_slot = s;
                        dd->expert_id = eid;
                        dd->destination_rank = dest_lane;
                        dd->local_expert_id = local_expert;
                        if (dd->local_expert_id >= 8u) {
                            fc->error_code = kPlErrOutputCapacityInsufficient;
                            PlDcciForwardCtr(fc);
                            return;
                        }
                        if (seg_off == 0xffffffffu) {
                            fc->error_code = kPlErrOutputCapacityInsufficient;
                            PlDcciForwardCtr(fc);
                            return;
                        }
                        dd->source_segment_offset = seg_off;
                        dd->destination_final_slot = seg_off;
                        dd->destination_slot = seg_off;
                        dd->payload_bytes = nbytes;
                        dd->source_lane = dest_lane;
                        dd->ring_slot = ring_slot;
                        dd->token_sequence =
                            cd->source_rank * tokens_per_src + cd->token_begin + i + 1u;
                        dd->checksum_seed = 0u;
                        ++staging_seq;
                        if (gather_pending == 4u) {
                            PlSourceMajorFlushGatherGroup(
                                sym, staging_base, gather_begin_seq, depth,
                                nbytes, gather_sources, gather_pending, ub,
                                ub_alt);
                            gather_pending = 0u;
                        }
                        // A single raw chunk can contain more routes for one
                        // hot destination than the staging ring can hold.
                        // Publish inside the chunk so egress can return credit
                        // before gather reaches the ring limit.
                        if ((inject & kPlP53InjectDropStagingTail) == 0u &&
                            (staging_seq - staging_pub) >=
                                static_cast<uint64_t>(staging_publish_coalesce)) {
                            PlSourceMajorPublishStagingRange(
                                sym, staging_base, staging_desc_base,
                                staging_pub, staging_seq, depth, nbytes);
                            fc->gather_token_dcci_count +=
                                staging_seq - staging_pub;
                            __gm__ PlStagingCtrlLine *st =
                                reinterpret_cast<__gm__ PlStagingCtrlLine *>(
                                    sym + DevPlIncP5StagingTailOff(dest_lane));
                            st->value = staging_seq;
                            st->epoch = epoch;
                            st->magic = kPlStagingCtrlMagic;
                            st->destination = dest_lane;
                            st->gather_complete = 0u;
                            PlDcci(reinterpret_cast<__gm__ uint8_t *>(st));
                            staging_pub = staging_seq;
                        }
                }
                }
                if (gather_pending > 0u) {
                    PlSourceMajorFlushGatherGroup(
                        sym, staging_base, gather_begin_seq, depth, nbytes,
                        gather_sources, gather_pending, ub, ub_alt);
                    gather_pending = 0u;
                }

                PlPublishRouteDone(sym, L, dest_lane, gen, epoch);
                lane_seen[L] = gen;
            }
        }
        if (all_caught) {
            bool stable = true;
            bool all_terminal = true;
            for (uint32_t L = 0u; L < upload_lanes; ++L) {
                __gm__ PlRawLaneCtrlLine *tl =
                    reinterpret_cast<__gm__ PlRawLaneCtrlLine *>(
                        sym + DevPlRawLaneTailOff(L));
                PlDcci(reinterpret_cast<__gm__ uint8_t *>(tl));
                uint64_t tail = 0u;
                (void)ReadRawCtrlEpochValue(tl, epoch, tail);
                if (lane_seen[L] < tail) {
                    stable = false;
                    break;
                }
                if (!PlRawLaneReachedTerminal(sym, desc, L, tail, epoch)) {
                    all_terminal = false;
                }
            }
            if (stable && all_terminal) {
                break;
            }
        }
    }
    // final flush：gather_complete=1；即使 staging_seq==0 也发布，避免 egress 在 gather 前误退出
    if ((inject & kPlP53InjectDropStagingTail) == 0u) {
        if (staging_seq > staging_pub) {
            PlSourceMajorPublishStagingRange(
                sym, DevPlIncPayloadLaneOff(dest_lane),
                DevPlIncDescLaneOff(dest_lane), staging_pub, staging_seq,
                depth, nbytes);
            fc->gather_token_dcci_count += staging_seq - staging_pub;
            staging_pub = staging_seq;
        }
        __gm__ PlStagingCtrlLine *st =
            reinterpret_cast<__gm__ PlStagingCtrlLine *>(sym + DevPlIncP5StagingTailOff(dest_lane));
        st->value = staging_seq;
        st->epoch = epoch;
        st->magic = kPlStagingCtrlMagic;
        st->destination = dest_lane;
        st->gather_complete = 1u;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(st));
    }
    fc->ingress_seen = staging_seq;
    fc->max_ingress_occupancy = staging_wrap; // 复用：staging_wrap_count 诊断
    __gm__ PlP5CellGatherLedgerLine *gledger =
        reinterpret_cast<__gm__ PlP5CellGatherLedgerLine *>(
            sym + DevPlP5CellGatherLedgerLineOff(desc->pair_id, dest_lane));
    gledger->epoch = epoch;
    gledger->source = desc->pair_id;
    gledger->destination = dest_lane;
    gledger->staging_final_gen = static_cast<uint32_t>(staging_seq);
    gledger->error_code = fc->error_code;
    gledger->magic = kPlP5CellLedgerMagic;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(gledger));
    PlDcciForwardCtr(fc);
    (void)fdc;
}

__aicore__ inline void PlIncSourceMajorEgressSenderEpoch(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                         __gm__ PlForwardCounters *fc, uint32_t dest_lane,
                                                         uint64_t epoch)
{
    const uint32_t dest_n = desc->worker_count == 0u ? 1u : desc->worker_count;
    if (dest_lane >= dest_n) {
        __gm__ PlEgressDoneLine *done =
            reinterpret_cast<__gm__ PlEgressDoneLine *>(sym + DevPlIncP5EgressDoneOff(dest_lane));
        done->done_epoch = epoch;
        done->dest_lane = dest_lane;
        done->magic = kPlEgressDoneMagic;
        done->egress_seq_end = 0u;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(done));
        PlDcciForwardCtr(fc);
        return;
    }
    const uint32_t inject = PlP53ReadInjectFlags(sym);
    if ((inject & kPlP53InjectWithholdCredit) != 0u) {
        fc->error_code = 49u;
        PlDcciForwardCtr(fc);
        return;
    }
    const uint32_t nbytes = desc->payload_bytes;
    const uint32_t depth = desc->ring_depth == 0u ? kQv2RingDepth : desc->ring_depth;
    const uint32_t upload_lanes = PlDevUploadLaneCount(desc);
    const int dest_pe = static_cast<int>(dest_lane);
    const uint32_t source_ch = desc->pair_id;
    uint64_t sent = 0u;
    uint64_t egress_seq = 0u;
    bool seen_epoch_staging = false;
    constexpr uint64_t kWaitUs = 8000000ull;
    const uint64_t t0 = AscendC::GetSystemCycle();
    uint64_t payload_puts = 0;
    uint64_t payload_mte = 0;
    uint64_t desc_puts = 0;
    uint64_t desc_mte = 0;
    // Ledger publish counters stay local: RMW on the shared egress ledger line
    // would reload a stale cacheline and clobber gather's staging_final_gen.
    uint32_t normal_tail_pubs = 0u;
    uint32_t terminal_tail_pubs = 0u;
    uint64_t first_pub_cycle = 0u;
    uint64_t last_pub_cycle = 0u;
    // batch_cap：CLI batch_tokens / egress_publish_target_bytes（默认约 128KiB token 数），且 ≤ depth/2。
    // A/B：CLI batch_tokens=1 即强制 scalar；禁止无条件 batch_cap=1。
    uint32_t batch_cap = desc->batch_tokens > 0u ? desc->batch_tokens : 0u;
    if (desc->egress_publish_target_bytes > 0u && nbytes > 0u) {
        const uint32_t bc = desc->egress_publish_target_bytes / nbytes;
        // Egress and ingress have independent optimal batch sizes.  The old
        // min(upload_batch, egress_bytes) silently capped every remote put at
        // 32 tokens even when the destination ring had more credit.
        if (bc > 0u) {
            batch_cap = bc;
        }
    }
    if (batch_cap == 0u && nbytes > 0u) {
        constexpr uint32_t kDefaultPublishBytes = 128u * 1024u;
        batch_cap = kDefaultPublishBytes / nbytes;
        if (batch_cap == 0u) {
            batch_cap = 1u;
        }
    }
    if (batch_cap == 0u) {
        batch_cap = 32u;
    }
    const uint32_t depth_window = depth > 1u ? (depth - 1u) : 1u;
    if (batch_cap > depth_window) {
        batch_cap = depth_window;
    }

    // destination credit：仅同 epoch_tag 有效；旧 epoch → 0；同 epoch credit>seq → 协议错误
    __gm__ Qv2HeadLine *credit_line =
        reinterpret_cast<__gm__ Qv2HeadLine *>(sym + DevPlIncEgressCreditLineOff(dest_lane));
    uint64_t cached_dest_credit = 0u;
    uint32_t egress_wrap = 0u;
    uint32_t stale_credit_epoch = 0u;

    while (!PlWaitBudgetExceeded(t0, kWaitUs)) {
        __gm__ PlStagingCtrlLine *st =
            reinterpret_cast<__gm__ PlStagingCtrlLine *>(sym + DevPlIncP5StagingTailOff(dest_lane));
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(st));
        uint64_t ready_gen = 0u;
        const bool staging_valid = ReadStagingCtrlEpochValue(st, epoch, ready_gen);
        if (staging_valid) {
            seen_epoch_staging = true;
        }
        if (ready_gen > sent) {
            const uint64_t remaining = ready_gen - sent;
            AscendC::PipeBarrier<PIPE_ALL>();
            Qv2DcciLine(reinterpret_cast<__gm__ uint8_t *>(credit_line));
            uint64_t credit_head = 0u;
            if (!ReadEgressHeadEpochValue(credit_line, epoch, credit_head)) {
                ++stale_credit_epoch;
                credit_head = 0u;
            } else if (credit_head > egress_seq) {
                fc->error_code = kPlErrEpochLocalCreditGtSeq;
                PlDcciForwardCtr(fc);
                return;
            }
            cached_dest_credit = credit_head;
            uint64_t occ = 0u;
            while ((!SafeOccupancy(egress_seq, cached_dest_credit, occ) ||
                    occ >= static_cast<uint64_t>(depth)) &&
                   !PlWaitBudgetExceeded(t0, kWaitUs)) {
                if (!SafeOccupancy(egress_seq, cached_dest_credit, occ) &&
                    cached_dest_credit > egress_seq) {
                    fc->error_code = kPlErrEpochLocalCreditGtSeq;
                    PlDcciForwardCtr(fc);
                    return;
                }
                ++fc->egress_credit_wait_cycles;
                PlSpinWaitSignalLineDcci(reinterpret_cast<__gm__ uint8_t *>(credit_line),
                                         static_cast<uint32_t>(fc->egress_credit_wait_cycles));
                Qv2DcciLine(reinterpret_cast<__gm__ uint8_t *>(credit_line));
                if (!ReadEgressHeadEpochValue(credit_line, epoch, credit_head)) {
                    ++stale_credit_epoch;
                    credit_head = 0u;
                } else if (credit_head > egress_seq) {
                    fc->error_code = kPlErrEpochLocalCreditGtSeq;
                    PlDcciForwardCtr(fc);
                    return;
                }
                cached_dest_credit = credit_head;
            }
            if (!SafeOccupancy(egress_seq, cached_dest_credit, occ)) {
                fc->error_code = kPlErrEpochLocalCreditGtSeq;
                PlDcciForwardCtr(fc);
                return;
            }
            const uint64_t credit =
                (occ >= static_cast<uint64_t>(depth)) ? 0ull : (static_cast<uint64_t>(depth) - occ);
            const uint32_t slot0 = static_cast<uint32_t>(sent % depth);
            const uint32_t staging_contig = depth - slot0;
            const uint32_t egress_slot0 = static_cast<uint32_t>(egress_seq % depth);
            const uint32_t dest_contig = depth - egress_slot0;
            uint32_t run = static_cast<uint32_t>(remaining);
            if (run > staging_contig) {
                run = staging_contig;
            }
            if (run > dest_contig) {
                run = dest_contig;
            }
            if (static_cast<uint64_t>(run) > credit) {
                run = static_cast<uint32_t>(credit);
            }
            if (run > batch_cap) {
                run = batch_cap;
            }
            if (run == 0u) {
                continue;
            }
            const uint64_t ingress_payload_src =
                DevPlIncPayloadLaneOff(dest_lane) + static_cast<uint64_t>(slot0) * nbytes;
            const uint64_t dest_payload_dst =
                DevPlDestChannelPayloadOff(source_ch) + static_cast<uint64_t>(egress_slot0) * nbytes;
            Qv2IssuePayloadRangeP6NoDrain(sym, ingress_payload_src, dest_payload_dst, run * nbytes, dest_pe,
                                          payload_puts, payload_mte);
            const uint64_t ingress_desc_src =
                DevPlIncDescLaneOff(dest_lane) + static_cast<uint64_t>(slot0) * kPlDescriptorBytes;
            const uint64_t dest_desc_dst =
                DevPlDestChannelDescOff(source_ch) + static_cast<uint64_t>(egress_slot0) * kPlDescriptorBytes;
            Qv2IssueDescriptorRangeP6NoDrain(sym, ingress_desc_src, dest_desc_dst, run * kPlDescriptorBytes,
                                             dest_pe, desc_puts, desc_mte);
            uint64_t drain_q = 0;
            Qv2DrainTileDataAndDescriptor(drain_q);
            (void)drain_q;
            egress_seq += run;
            if (egress_seq > 0u && (egress_seq % depth) < run) {
                ++egress_wrap;
            }
            // 必须用 putmem_signal 发布 notify_seq；recv 侧先等 epoch_tag 再 GE
            // 本地 put 源必须按 destination lane 独占。source_ch 在同一 INC 的 8 个
            // egress AIV 间相同；若把正式 remote tail line 当作本地 source scratch，
            // 并发 putmem_signal 会读到其他 lane 覆盖的 tail/epoch。
            __gm__ Qv2TailLine *tail_scratch =
                reinterpret_cast<__gm__ Qv2TailLine *>(sym + DevPlIncEgressPublishScratchOff(dest_lane));
            uint64_t tp = 0;
            uint64_t tq = 0;
            tail_scratch->magic = inc::dc::dn::qv2::kQv2Magic;
            tail_scratch->lane_id = source_ch;
            // remote landing slot is still indexed by source channel.
            PlPublishEgressTail(tail_scratch, sym, DevPlEgressTailLineOff(source_ch), dest_pe, egress_seq, epoch,
                                tp, tq);
            normal_tail_pubs += static_cast<uint32_t>(tp);
            last_pub_cycle = AscendC::GetSystemCycle();
            if (first_pub_cycle == 0u) {
                first_pub_cycle = last_pub_cycle;
            }
            __gm__ PlP5CellEgressLedgerLine *ledger =
                reinterpret_cast<__gm__ PlP5CellEgressLedgerLine *>(
                    sym + DevPlP5CellEgressLedgerLineOff(source_ch, dest_lane));
            // Absolute stores only; never RMW and never touch staging_final_gen
            // (gather is the sole writer of that field).
            ledger->epoch = epoch;
            ledger->source = source_ch;
            ledger->destination = dest_lane;
            ledger->egress_seq_end = static_cast<uint32_t>(egress_seq);
            ledger->last_published_tail = static_cast<uint32_t>(egress_seq);
            ledger->normal_tail_publish_count = normal_tail_pubs;
            ledger->terminal_republish_count = terminal_tail_pubs;
            ledger->first_publish_cycle = first_pub_cycle;
            ledger->last_publish_cycle = last_pub_cycle;
            ledger->magic = kPlP5CellLedgerMagic;
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(ledger));
            if (run > 1u) {
                ++fc->egress_payload_range_put_count;
            } else {
                ++fc->egress_payload_scalar_put_count;
            }
            ++fc->egress_batch_count;
            fc->egress_batch_tokens += run;
            sent += run;
            // 推进 staging_head = sent（egress 唯一 writer）
            __gm__ PlStagingCtrlLine *sh =
                reinterpret_cast<__gm__ PlStagingCtrlLine *>(sym + DevPlIncP5StagingHeadOff(dest_lane));
            sh->value = sent;
            sh->epoch = epoch;
            sh->magic = kPlStagingCtrlMagic;
            sh->destination = dest_lane;
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(sh));
            continue;
        }

        // 退出：gather final 完成 + upload reclaim 完 + staging 排空
        // 禁止仅 raw_done∧sent≥中间 coalesce ready_gen（B/64tok epoch2 早退 → egress_tail=56）
        bool raw_done = true;
        for (uint32_t L = 0u; L < upload_lanes; ++L) {
            __gm__ PlRawLaneCtrlLine *tl =
                reinterpret_cast<__gm__ PlRawLaneCtrlLine *>(sym + DevPlRawLaneTailOff(L));
            __gm__ PlRawLaneCtrlLine *hd =
                reinterpret_cast<__gm__ PlRawLaneCtrlLine *>(sym + DevPlRawLaneHeadOff(L));
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(tl));
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(hd));
            uint64_t tail = 0u;
            uint64_t head = 0u;
            (void)ReadRawCtrlEpochValue(tl, epoch, tail);
            (void)ReadRawCtrlEpochValue(hd, epoch, head);
            if (!PlRawLaneReachedTerminal(sym, desc, L, tail, epoch) || head < tail) {
                raw_done = false;
                break;
            }
        }
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(st));
        uint64_t final_gen = 0u;
        const bool final_valid = ReadStagingCtrlEpochValue(st, epoch, final_gen);
        const bool gather_done = final_valid && (st->gather_complete == 1u);
        // raw_done is diagnostic here.  gather_complete is the authoritative
        // per-destination proof that all terminal raw descriptors were
        // consumed; coupling egress completion to the independent reclaim AIV
        // only extends the INC rank tail.
        (void)raw_done;
        if (seen_epoch_staging && gather_done && sent >= final_gen) {
            break;
        }
    }
    // E5b：仅在完整排空后写 done；超时不完整则报错且不发 done（防 block19/recv 假齐）
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(
        reinterpret_cast<__gm__ PlStagingCtrlLine *>(sym + DevPlIncP5StagingTailOff(dest_lane))));
    {
        __gm__ PlStagingCtrlLine *st_final =
            reinterpret_cast<__gm__ PlStagingCtrlLine *>(sym + DevPlIncP5StagingTailOff(dest_lane));
        uint64_t final_gen = 0u;
        const bool final_ok =
            ReadStagingCtrlEpochValue(st_final, epoch, final_gen) && (st_final->gather_complete == 1u);
        if (!(final_ok && sent >= final_gen)) {
            fc->error_code = kPlErrEpochLocalEgressWait;
            fc->egress_forwarded = egress_seq;
            fc->max_egress_occupancy = egress_wrap;
            fc->first_ingress_tail = stale_credit_epoch;
            PlDcciForwardCtr(fc);
            return;
        }
        // Diagnostic-only causality A/B. The normal protocol does not rely on
        // a terminal re-publish; enabling this must never authorize a pass.
        if ((inject & kPlP53InjectTerminalTailRepublish) != 0u) {
            __gm__ Qv2TailLine *terminal_tail_scratch =
                reinterpret_cast<__gm__ Qv2TailLine *>(sym + DevPlIncEgressPublishScratchOff(dest_lane));
            terminal_tail_scratch->magic = inc::dc::dn::qv2::kQv2Magic;
            terminal_tail_scratch->lane_id = source_ch;
            uint64_t terminal_tail_pub = 0u;
            uint64_t terminal_tail_quiet = 0u;
            PlPublishEgressTail(terminal_tail_scratch, sym, DevPlEgressTailLineOff(source_ch), dest_pe, egress_seq,
                                epoch, terminal_tail_pub, terminal_tail_quiet);
            fc->egress_tail_publish_count += terminal_tail_pub;
            terminal_tail_pubs += static_cast<uint32_t>(terminal_tail_pub);
            last_pub_cycle = AscendC::GetSystemCycle();
        }
    }
    {
        __gm__ PlEgressDoneLine *done =
            reinterpret_cast<__gm__ PlEgressDoneLine *>(sym + DevPlIncP5EgressDoneOff(dest_lane));
        done->done_epoch = epoch;
        done->dest_lane = dest_lane;
        done->magic = kPlEgressDoneMagic;
        done->egress_seq_end = egress_seq;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(done));
    }
    fc->egress_forwarded = egress_seq;
    __gm__ PlP5CellEgressLedgerLine *final_ledger =
        reinterpret_cast<__gm__ PlP5CellEgressLedgerLine *>(
            sym + DevPlP5CellEgressLedgerLineOff(source_ch, dest_lane));
    final_ledger->epoch = epoch;
    final_ledger->source = source_ch;
    final_ledger->destination = dest_lane;
    final_ledger->egress_seq_end = static_cast<uint32_t>(egress_seq);
    final_ledger->last_published_tail = static_cast<uint32_t>(egress_seq);
    final_ledger->normal_tail_publish_count = normal_tail_pubs;
    final_ledger->terminal_republish_count = terminal_tail_pubs;
    final_ledger->first_publish_cycle = first_pub_cycle;
    final_ledger->last_publish_cycle = last_pub_cycle;
    final_ledger->error_code = fc->error_code;
    final_ledger->magic = kPlP5CellLedgerMagic;
    // Do not write staging_final_gen here — gather owns that field exclusively.
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(final_ledger));
    fc->max_egress_occupancy = egress_wrap; // 复用：egress_wrap_count
    fc->first_ingress_tail = stale_credit_epoch; // 复用：stale credit epoch ignored
    (void)payload_puts;
    (void)payload_mte;
    (void)desc_puts;
    (void)desc_mte;
    PlDcciForwardCtr(fc);
}

// first_hop：等每 lane raw_tail≥1 且 desc 可见；校验 unique token 合计
__aicore__ inline void PlIncSourceMajorIngressVerifyEpoch(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                          __gm__ PlFullDispatchConfig *cfg,
                                                          __gm__ PlForwardCounters *fc, uint32_t lane_id,
                                                          uint64_t epoch)
{
    if (lane_id != 0u) {
        return;
    }
    const uint32_t nbytes = desc->payload_bytes;
    const uint32_t tokens = desc->tokens_per_epoch;
    const uint32_t lanes = desc->lane_count == 0u ? 1u : desc->lane_count;
    const uint32_t inject = PlP53ReadInjectFlags(sym);
    constexpr uint64_t kWaitUs = 8000000ull; // 8s：首 epoch 调度抖动；失败仍发 first_hop_done
    const uint64_t t0 = AscendC::GetSystemCycle();
    uint64_t verified_run = 0u;
    bool all_ready = false;

    while (!PlWaitBudgetExceeded(t0, kWaitUs)) {
        verified_run = 0u;
        uint32_t ready_lanes = 0u;
        bool inject_fail = false;
        for (uint32_t L = 0u; L < lanes && L < kPlRawUploadLaneCount; ++L) {
            __gm__ PlRawLaneCtrlLine *tl =
                reinterpret_cast<__gm__ PlRawLaneCtrlLine *>(sym + DevPlRawLaneTailOff(L));
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(tl));
            if (tl->magic != kPlRawLaneCtrlMagic || tl->epoch != epoch || tl->value < 1u) {
                continue;
            }
            // 只看最新 generation 的 desc（单 tile 常见）；多 tile 累加 token_count
            uint32_t lane_tokens = 0u;
            bool lane_ok = true;
            for (uint64_t g = 1u; g <= tl->value; ++g) {
                const uint32_t slot = static_cast<uint32_t>((g - 1u) % kPlRawChunkRingDepth);
                __gm__ PlRawChunkDesc *cd =
                    reinterpret_cast<__gm__ PlRawChunkDesc *>(sym + DevPlRawChunkDescLaneSlotOff(L, slot));
                PlDcci(reinterpret_cast<__gm__ uint8_t *>(cd));
                if (cd->magic != kPlRawChunkDescMagic || cd->epoch != epoch) {
                    if ((inject & kPlP53InjectEarlyTail) != 0u) {
                        inject_fail = true;
                    }
                    lane_ok = false;
                    break;
                }
                if ((inject & kPlP53InjectStaleGen) != 0u && cd->generation != g) {
                    inject_fail = true;
                    break;
                }
                if (cd->generation != g) {
                    lane_ok = false;
                    break;
                }
                lane_tokens += cd->token_count;
            }
            if (inject_fail) {
                break;
            }
            if (lane_ok) {
                verified_run += lane_tokens;
                ++ready_lanes;
            }
        }
        if (inject_fail) {
            const uint32_t ierr = ((inject & kPlP53InjectEarlyTail) != 0u) ? 44u : 45u;
            fc->error_code = ierr;
            PlDcciForwardCtr(fc);
            // fail-closed：必须发 first_hop_done，否则 Worker 永等 → NEG 被 timeout 假杀
            __gm__ PlFirstHopDoneLine *scratch_inj = reinterpret_cast<__gm__ PlFirstHopDoneLine *>(
                sym + DevPlIncEgressPublishScratchOff(0u));
            scratch_inj->epoch = epoch;
            scratch_inj->lane_id = 0u;
            scratch_inj->tokens_received = 0u;
            scratch_inj->payload_bytes_received = 0u;
            scratch_inj->descriptor_bytes_received = 0u;
            scratch_inj->ingress_head = 0u;
            scratch_inj->error_code = ierr;
            scratch_inj->magic = kPlMagic;
            PlDcciDoneLine(reinterpret_cast<__gm__ uint8_t *>(scratch_inj));
            aclshmem_putmem_nbi(sym + cfg->first_hop_done_off, reinterpret_cast<__gm__ uint8_t *>(scratch_inj),
                                sizeof(PlFirstHopDoneLine), static_cast<int>(desc->peer_pe));
            aclshmem_quiet();
            PlDevRankTimingMarkForwardDone(sym, AscendC::GetSystemCycle());
            PlDevRankTimingPublishLocalCompletion(sym, kPlRoleMaskInc);
            return;
        }
        if (ready_lanes == lanes && verified_run == tokens) {
            all_ready = true;
            break;
        }
    }
    if (!all_ready || verified_run != tokens) {
        fc->error_code = 42u;
        PlDcciForwardCtr(fc);
        // fail-closed：仍发布 first_hop_done（error_code!=0），避免 Worker 永等
        __gm__ PlFirstHopDoneLine *scratch_fail = reinterpret_cast<__gm__ PlFirstHopDoneLine *>(
            sym + DevPlIncEgressPublishScratchOff(0u));
        scratch_fail->epoch = epoch;
        scratch_fail->lane_id = 0u;
        scratch_fail->tokens_received = 0u;
        scratch_fail->payload_bytes_received = 0u;
        scratch_fail->descriptor_bytes_received = 0u;
        scratch_fail->ingress_head = 0u;
        scratch_fail->error_code = 42u;
        scratch_fail->magic = kPlMagic;
        PlDcciDoneLine(reinterpret_cast<__gm__ uint8_t *>(scratch_fail));
        aclshmem_putmem_nbi(sym + cfg->first_hop_done_off, reinterpret_cast<__gm__ uint8_t *>(scratch_fail),
                            sizeof(PlFirstHopDoneLine), static_cast<int>(desc->peer_pe));
        aclshmem_quiet();
        PlDevRankTimingMarkForwardDone(sym, AscendC::GetSystemCycle());
        PlDevRankTimingPublishLocalCompletion(sym, kPlRoleMaskInc);
        return;
    }

    // first_hop：直接 reclaim 各 lane head=tail（无 gather）
    // withhold_credit：故意不推进 head，并发 first_hop_done error=49（结构化 fail-closed）
    if ((inject & kPlP53InjectWithholdCredit) != 0u) {
        fc->error_code = 49u;
        PlDcciForwardCtr(fc);
        __gm__ PlFirstHopDoneLine *scratch_wh = reinterpret_cast<__gm__ PlFirstHopDoneLine *>(
            sym + DevPlIncEgressPublishScratchOff(0u));
        scratch_wh->epoch = epoch;
        scratch_wh->lane_id = 0u;
        scratch_wh->tokens_received = 0u;
        scratch_wh->payload_bytes_received = 0u;
        scratch_wh->descriptor_bytes_received = 0u;
        scratch_wh->ingress_head = 0u;
        scratch_wh->error_code = 49u;
        scratch_wh->magic = kPlMagic;
        PlDcciDoneLine(reinterpret_cast<__gm__ uint8_t *>(scratch_wh));
        aclshmem_putmem_nbi(sym + cfg->first_hop_done_off, reinterpret_cast<__gm__ uint8_t *>(scratch_wh),
                            sizeof(PlFirstHopDoneLine), static_cast<int>(desc->peer_pe));
        aclshmem_quiet();
        PlDevRankTimingMarkForwardDone(sym, AscendC::GetSystemCycle());
        PlDevRankTimingPublishLocalCompletion(sym, kPlRoleMaskInc);
        return;
    }
    for (uint32_t L = 0u; L < lanes && L < kPlRawUploadLaneCount; ++L) {
        __gm__ PlRawLaneCtrlLine *tl =
            reinterpret_cast<__gm__ PlRawLaneCtrlLine *>(sym + DevPlRawLaneTailOff(L));
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(tl));
        PlPublishRawLaneHead(sym, L, tl->value, epoch, static_cast<int>(desc->peer_pe));
    }
    aclshmem_quiet();

    __gm__ PlFirstHopDoneLine *scratch = reinterpret_cast<__gm__ PlFirstHopDoneLine *>(
        sym + DevPlIncEgressPublishScratchOff(0u));
    scratch->epoch = epoch;
    scratch->lane_id = 0u;
    scratch->tokens_received = static_cast<uint32_t>(static_cast<uint64_t>(tokens) * epoch);
    scratch->payload_bytes_received = static_cast<uint64_t>(tokens) * nbytes;
    scratch->descriptor_bytes_received = 0u;
    scratch->ingress_head = static_cast<uint64_t>(lanes);
    scratch->error_code = 0u;
    scratch->magic = kPlMagic;
    PlDcciDoneLine(reinterpret_cast<__gm__ uint8_t *>(scratch));
    aclshmem_putmem_nbi(sym + cfg->first_hop_done_off, reinterpret_cast<__gm__ uint8_t *>(scratch),
                        sizeof(PlFirstHopDoneLine), static_cast<int>(desc->peer_pe));
    aclshmem_quiet();
    fc->ingress_seen = tokens;

    {
        const uint32_t topk = desc->route_topk == 0u ? 1u : desc->route_topk;
        __gm__ PlFirstHopVerifyLine *vline =
            reinterpret_cast<__gm__ PlFirstHopVerifyLine *>(sym + kPlFirstHopVerifyOff);
        vline->epoch = epoch;
        vline->source_rank = desc->pair_id;
        vline->tokens_checked = tokens;
        vline->payload_bytes_checked = static_cast<uint64_t>(tokens) * nbytes;
        vline->meta_bytes_checked = static_cast<uint64_t>(tokens) * topk * sizeof(int32_t);
        vline->payload_checksum64_obs = 0u;
        vline->payload_exact = 0u;
        vline->meta_exact = 0u;
        vline->error_code = 0u;
        vline->magic = kPlFirstHopVerifyMagic;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(vline));
        aclshmem_putmem_nbi(sym + kPlFirstHopVerifyOff, reinterpret_cast<__gm__ uint8_t *>(vline),
                            sizeof(PlFirstHopVerifyLine), static_cast<int>(desc->peer_pe));
        aclshmem_quiet();
    }

    PlDevRankTimingMarkForwardDone(sym, AscendC::GetSystemCycle());
    PlDevRankTimingPublishLocalCompletion(sym, kPlRoleMaskInc);
    PlDcciForwardCtr(fc);
}

} // namespace inc::dc::dn::pl
