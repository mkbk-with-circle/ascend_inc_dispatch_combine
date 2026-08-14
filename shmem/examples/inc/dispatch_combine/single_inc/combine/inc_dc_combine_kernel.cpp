/**
 * DYN3D: independent CSR combine kernel (not FG7 frozen SO).
 * Reducer scans result_offsets CSR only — never source=0..active_sources.
 */
#include "kernel_operator.h"
#include "shmem.h"

#include "inc_dc_combine_vector_reduce_aicore.h"
#include "inc_dc_combine_runtime_abi.h"
#include "inc_dc_fp16_aicore.h"
#include "inc_dc_gather_mte_aicore.h"

using namespace AscendC;
using namespace inc::dc;

__aicore__ inline void DynDcci(__gm__ uint8_t *p, uint32_t n)
{
    const uint32_t lines = (n + 63u) / 64u;
    for (uint32_t i = 0; i < lines; ++i) {
        dcci_cacheline(p + static_cast<uint64_t>(i) * 64u);
    }
}

__aicore__ inline float DynBitsToFloat(uint32_t bits)
{
    union {
        uint32_t u;
        float f;
    } x;
    x.u = bits;
    return x.f;
}

__aicore__ inline bool DynAbortRequested(__gm__ DynCsrCtrl *ctrl,
                                         uint32_t generation)
{
    DynDcci(reinterpret_cast<__gm__ uint8_t *>(ctrl), kDynCsrCtrlBytes);
    return ctrl->abort_generation == generation;
}

__aicore__ inline void DynReadyPollDcci(__gm__ int32_t *ready,
                                        uint32_t spins)
{
    // Poll aggressively for short arrivals, then reduce cache-invalidation
    // traffic while a large RMA is still in flight.  This is a bounded
    // visibility backoff, not a sleep: generation, abort and spin-cap
    // semantics remain unchanged.
    const uint32_t mask =
        spins < 64u ? 0u : (spins < 1024u ? 7u : 31u);
    if ((spins & mask) == 0u) {
        DynDcci(reinterpret_cast<__gm__ uint8_t *>(ready),
                sizeof(int32_t));
    }
}

__aicore__ inline int32_t DynParticipantPe(
    uint32_t participant, uint32_t worker_count, uint32_t inc_pe,
    __gm__ uint32_t *worker_pes)
{
    if (participant < worker_count) {
        return static_cast<int32_t>(worker_pes[participant]);
    }
    return participant == worker_count ? static_cast<int32_t>(inc_pe) : -1;
}

// A stream-level SHMEM barrier orders work but, on the target runtime, does
// not guarantee that all streams leave the collective at the same device
// time.  A late stream then makes reducers spin for milliseconds even though
// payload transport itself is fast.  This bounded device rendezvous starts
// every producer/reducer kernel generation only after lane/owner 0 from every
// runtime participant is resident.  Records are cacheline-owned and topology
// sized, so W2/W4/W8 and non-contiguous PE maps use identical semantics.
__aicore__ inline bool DynDeviceGenerationStartGate(
    __gm__ uint8_t *sym, __gm__ DynCsrCtrl *ctrl, uint32_t participant,
    bool coordinator_lane, __gm__ uint32_t *worker_pes)
{
    if (ctrl->start_gate_off == 0u) {
        return true;
    }
    const uint32_t participant_count = ctrl->worker_count + 1u;
    if (participant_count == 0u || participant >= participant_count ||
        ctrl->ready_stride_bytes < 64u || worker_pes == nullptr) {
        return false;
    }
    const int32_t coordinator_pe = DynParticipantPe(
        0u, ctrl->worker_count, ctrl->inc_pe, worker_pes);
    if (coordinator_pe < 0) {
        return false;
    }
    const int32_t generation = static_cast<int32_t>(ctrl->generation);
    __gm__ int32_t *local_go = reinterpret_cast<__gm__ int32_t *>(
        sym + ctrl->start_gate_off +
        static_cast<uint64_t>(participant_count) *
            ctrl->ready_stride_bytes);

    if (coordinator_lane) {
        __gm__ int32_t *arrival = reinterpret_cast<__gm__ int32_t *>(
            sym + ctrl->start_gate_off +
            static_cast<uint64_t>(participant) *
                ctrl->ready_stride_bytes);
        *arrival = generation;
        AscendC::PipeBarrier<PIPE_ALL>();
        DynDcci(reinterpret_cast<__gm__ uint8_t *>(arrival),
                ctrl->ready_stride_bytes);
        aclshmem_putmem_nbi(
            reinterpret_cast<__gm__ void *>(arrival),
            reinterpret_cast<__gm__ void *>(arrival),
            ctrl->ready_stride_bytes, coordinator_pe);
        aclshmem_quiet();

        if (participant == 0u) {
            for (uint32_t peer = 0u; peer < participant_count; ++peer) {
                __gm__ int32_t *peer_arrival =
                    reinterpret_cast<__gm__ int32_t *>(
                        sym + ctrl->start_gate_off +
                        static_cast<uint64_t>(peer) *
                            ctrl->ready_stride_bytes);
                uint32_t spins = 0u;
                while (*peer_arrival != generation &&
                       spins < ctrl->ready_spin_cap) {
                    DynReadyPollDcci(peer_arrival, spins);
                    if ((spins & 1023u) == 0u &&
                        DynAbortRequested(ctrl, ctrl->generation)) {
                        return false;
                    }
                    ++spins;
                }
                if (*peer_arrival != generation) {
                    return false;
                }
            }
            *local_go = generation;
            AscendC::PipeBarrier<PIPE_ALL>();
            DynDcci(reinterpret_cast<__gm__ uint8_t *>(local_go),
                    ctrl->ready_stride_bytes);
            // The release is outside measured work.  Prefer two idempotent
            // blocking cacheline fanout rounds over one NBI burst: the latter
            // has exhibited rare 1--4 ms mapped-cache visibility tails even
            // after quiet.  A peer may leave after round one; round two writes
            // the same generation and cannot regress protocol state.
            for (uint32_t round = 0u; round < 2u; ++round) {
                for (uint32_t peer = 1u; peer < participant_count; ++peer) {
                    const int32_t peer_pe = DynParticipantPe(
                        peer, ctrl->worker_count, ctrl->inc_pe, worker_pes);
                    if (peer_pe < 0) {
                        return false;
                    }
                    aclshmem_putmem(
                        reinterpret_cast<__gm__ void *>(local_go),
                        reinterpret_cast<__gm__ void *>(local_go),
                        ctrl->ready_stride_bytes, peer_pe);
                }
            }
        }
    }

    uint32_t spins = 0u;
    while (*local_go != generation && spins < ctrl->ready_spin_cap) {
        DynReadyPollDcci(local_go, spins);
        if ((spins & 1023u) == 0u &&
            DynAbortRequested(ctrl, ctrl->generation)) {
            return false;
        }
        ++spins;
    }
    return *local_go == generation;
}

__aicore__ inline bool DynWaitPersistentLocalTrigger(
    __gm__ uint8_t *sym, __gm__ DynCsrCtrl *ctrl)
{
    if ((ctrl->optimization_flags &
         kDynCsrOptPersistentLocalTrigger) == 0u) {
        return true;
    }
    const uint32_t participant_count = ctrl->worker_count + 1u;
    if (ctrl->start_gate_off == 0u || participant_count == 0u ||
        ctrl->ready_stride_bytes < 64u) {
        return false;
    }
    __gm__ int32_t *trigger = reinterpret_cast<__gm__ int32_t *>(
        sym + ctrl->start_gate_off +
        static_cast<uint64_t>(participant_count) *
            ctrl->ready_stride_bytes);
    __gm__ uint64_t *target_cycle =
        reinterpret_cast<__gm__ uint64_t *>(
            reinterpret_cast<__gm__ uint8_t *>(trigger) + 8u);
    const int32_t generation = static_cast<int32_t>(ctrl->generation);
    DynDcci(reinterpret_cast<__gm__ uint8_t *>(trigger),
            ctrl->ready_stride_bytes);
    if (*target_cycle != 0u) {
        while (GetSystemCycle() < *target_cycle) {
            // The target is local and immutable.  No fabric polling or DCCI
            // is needed, so host scheduling cannot perturb the release.
        }
        return true;
    }
    uint32_t spins = 0u;
    while (*trigger != generation && spins < ctrl->ready_spin_cap) {
        DynReadyPollDcci(trigger, spins);
        if ((spins & 1023u) == 0u &&
            DynAbortRequested(ctrl, ctrl->generation)) {
            return false;
        }
        ++spins;
    }
    return *trigger == generation;
}

__aicore__ inline __gm__ DynCsrPersistentTriggerLine *
DynPersistentTriggerLine(__gm__ uint8_t *sym, __gm__ DynCsrCtrl *ctrl)
{
    const uint32_t participant_count = ctrl->worker_count + 1u;
    return reinterpret_cast<__gm__ DynCsrPersistentTriggerLine *>(
        sym + ctrl->start_gate_off +
        static_cast<uint64_t>(participant_count) *
            ctrl->ready_stride_bytes);
}

__aicore__ inline void DynPersistentServiceStart(
    __gm__ uint8_t *sym, __gm__ DynCsrCtrl *ctrl, uint64_t cycle)
{
    if ((ctrl->optimization_flags &
         kDynCsrOptPersistentLocalTrigger) == 0u) {
        return;
    }
    __gm__ DynCsrPersistentTriggerLine *line =
        DynPersistentTriggerLine(sym, ctrl);
    if (line->generation == static_cast<int32_t>(ctrl->generation)) {
        line->service_start_cycle = cycle;
        line->service_end_cycle = 0u;
        line->last_generation = ctrl->generation;
        line->status = 0u;
        AscendC::PipeBarrier<PIPE_ALL>();
        DynDcci(reinterpret_cast<__gm__ uint8_t *>(line), sizeof(*line));
    }
}

__aicore__ inline void DynPersistentServiceEnd(
    __gm__ uint8_t *sym, __gm__ DynCsrCtrl *ctrl, uint32_t status)
{
    if ((ctrl->optimization_flags &
         kDynCsrOptPersistentLocalTrigger) == 0u) {
        return;
    }
    __gm__ DynCsrPersistentTriggerLine *line =
        DynPersistentTriggerLine(sym, ctrl);
    DynDcci(reinterpret_cast<__gm__ uint8_t *>(line), sizeof(*line));
    line->service_end_cycle = GetSystemCycle();
    line->last_generation = ctrl->generation;
    // A queued service is one logical operation.  Never let a later healthy
    // epoch erase an earlier descriptor/cancel/protocol failure.
    if (line->status == kDynCsrFailNone &&
        status != kDynCsrFailNone) {
        line->status = status;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    DynDcci(reinterpret_cast<__gm__ uint8_t *>(line), sizeof(*line));
}

__aicore__ inline bool DynWaitProducerPeerLanes(
    __gm__ uint8_t *sym, __gm__ DynCsrCtrl *ctrl, uint32_t generation)
{
    // Lane 0 must not start the collective completion wait until every other
    // producer lane has published all of its owner/source groups.  Otherwise
    // a short worklist on lane 0 can exhaust the INC wait budget while a
    // longer peer lane is still issuing payloads.
    for (uint32_t peer = 1u; peer < ctrl->producer_lane_count; ++peer) {
        __gm__ DynCsrProducerStats *peer_stats =
            reinterpret_cast<__gm__ DynCsrProducerStats *>(
                sym + ctrl->owner_stats_off +
                static_cast<uint64_t>(peer) *
                    sizeof(DynCsrProducerStats));
        uint32_t spins = 0u;
        while (peer_stats->done != generation &&
               spins < ctrl->ready_spin_cap) {
            DynDcci(reinterpret_cast<__gm__ uint8_t *>(peer_stats),
                    sizeof(DynCsrProducerStats));
            ++spins;
        }
        if (peer_stats->done != generation) {
            return false;
        }
    }
    return true;
}

__aicore__ inline bool DynWaitProducerPeerPayloads(
    __gm__ uint8_t *sym, __gm__ DynCsrCtrl *ctrl, uint32_t generation)
{
    // INC-scoped ready publication is performed by lane 0, but every lane
    // owns a disjoint payload slice.  Keep this stage separate from `done`:
    // `done` is the operation-completion credit while reserved[1] is only the
    // local payload visibility credit for the current generation.
    for (uint32_t peer = 1u; peer < ctrl->producer_lane_count; ++peer) {
        __gm__ DynCsrProducerStats *peer_stats =
            reinterpret_cast<__gm__ DynCsrProducerStats *>(
                sym + ctrl->owner_stats_off +
                static_cast<uint64_t>(peer) *
                    sizeof(DynCsrProducerStats));
        uint32_t spins = 0u;
        while (peer_stats->reserved[1] != generation &&
               spins < ctrl->ready_spin_cap) {
            DynDcci(reinterpret_cast<__gm__ uint8_t *>(peer_stats),
                    sizeof(DynCsrProducerStats));
            ++spins;
        }
        if (peer_stats->reserved[1] != generation ||
            peer_stats->reserved[0] != kDynCsrFailNone) {
            return false;
        }
    }
    return true;
}

__aicore__ inline uint64_t DynCollectiveCompletionBase(
    __gm__ DynCsrCtrl *ctrl)
{
    // Stream readiness owns records [0, max_ingress_slots).  Other ready
    // modes own [0, group_count).  Keeping completion beyond the active
    // namespace lets queued epochs safely reuse payload/output buffers.
    return (ctrl->ready_mode == 5u || ctrl->ready_mode == 6u)
               ? static_cast<uint64_t>(ctrl->max_ingress_slots)
               : static_cast<uint64_t>(ctrl->group_count);
}

__aicore__ inline bool DynWaitCollectiveValue(
    __gm__ DynCsrCtrl *ctrl, __gm__ int32_t *line, int32_t expected)
{
    uint32_t spins = 0u;
    while (*line != expected) {
        DynReadyPollDcci(line, spins);
        if ((spins & 1023u) == 0u &&
            DynAbortRequested(ctrl, ctrl->generation)) {
            return false;
        }
        ++spins;
    }
    return true;
}

__aicore__ inline bool DynAckCollectiveCompletion(
    __gm__ uint8_t *sym, __gm__ DynCsrCtrl *ctrl, uint32_t generation)
{
    if (sym == nullptr || ctrl == nullptr || ctrl->owner_count == 0u) {
        return false;
    }
    // Keep ACK TX storage disjoint from the local completion RX cacheline.
    // The INC may publish generation+1 immediately after observing the ACK;
    // sourcing the ACK from that RX line therefore creates a cross-direction
    // DMA race even when the worker subsequently calls quiet.
    __gm__ int32_t *ack_stage = reinterpret_cast<__gm__ int32_t *>(
        sym + ctrl->stats_off);
    *ack_stage = -static_cast<int32_t>(generation);
    AscendC::PipeBarrier<PIPE_ALL>();
    DynDcci(reinterpret_cast<__gm__ uint8_t *>(ack_stage),
            ctrl->ready_stride_bytes);
    const uint64_t completion =
        DynCollectiveCompletionBase(ctrl) + ctrl->this_worker_rank;
    __gm__ int32_t *done = reinterpret_cast<__gm__ int32_t *>(
        sym + ctrl->ready_generation_off +
        completion * ctrl->ready_stride_bytes);
    // Completion is a cacheline protocol, not an atomic arithmetic
    // protocol.  signal_op can complete while a peer still observes the
    // old mapped cacheline for milliseconds on this platform.  Publish
    // the ACK through the same cacheline RMA + quiet sequence as payload
    // readiness so a long persistent train cannot lose its final ACK.
    aclshmem_putmem_nbi(
        reinterpret_cast<__gm__ void *>(done),
        reinterpret_cast<__gm__ void *>(ack_stage),
        ctrl->ready_stride_bytes, static_cast<int32_t>(ctrl->inc_pe));
    aclshmem_quiet();
    return true;
}

__aicore__ inline void DynPutContiguousGroup(
    __gm__ uint8_t *tile, uint32_t tile_count, uint32_t tile_bytes,
    uint32_t chunk_bytes, uint32_t quiet_window, int32_t home_pe,
    __gm__ DynCsrProducerStats *pst)
{
    uint32_t chunk_tiles =
        tile_bytes == 0u ? 1u : chunk_bytes / tile_bytes;
    if (chunk_tiles == 0u) {
        chunk_tiles = 1u;
    }
    // The target NBI transport does not preserve a large same-destination
    // contiguous transfer when that logical transfer is split into multiple
    // chunks and several chunks are left outstanding together.  This showed
    // up as correct first-vector-tile data followed by stale data at the
    // 512 KiB split.  A single-chunk group retains the requested window;
    // multi-chunk groups use an explicit per-chunk completion boundary.
    const uint32_t window =
        tile_count > chunk_tiles
            ? 1u
            : (quiet_window == 0u ? 1u : quiet_window);
    uint32_t pending = 0u;
    for (uint32_t first = 0u; first < tile_count;
         first += chunk_tiles) {
        const uint32_t count =
            tile_count - first < chunk_tiles ? tile_count - first
                                             : chunk_tiles;
        if (pst->issued == 0u) {
            pst->first_issue_cycle = GetSystemCycle();
        }
        __gm__ uint8_t *chunk =
            tile + static_cast<uint64_t>(first) * tile_bytes;
        const uint32_t packet_bytes = count * tile_bytes;
        if (packet_bytes > kIncDcPrivateMtePacketBytes) {
            // The public put path owns one shared staging/sync slot per AIV;
            // long packets can silently overwrite a vector block when many
            // producers target one INC.  The private MTE push backend carries
            // an explicit UB buffer and completion credit, so hidden size is
            // no longer coupled to that public packet limit.
            const uint32_t ub_tile = 8u * 1024u;
            __ubuf__ uint8_t *ub =
                reinterpret_cast<__ubuf__ uint8_t *>(0);
            aclshmemx_mte_put_nbi(chunk, chunk, ub, ub_tile, packet_bytes,
                                  home_pe, 0u);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
            aclshmemx_mte_quiet();
            pst->last_quiet_cycle = GetSystemCycle();
            pending = 0u;
        } else {
            aclshmem_putmem_nbi(
                reinterpret_cast<__gm__ void *>(chunk),
                reinterpret_cast<__gm__ void *>(chunk), packet_bytes,
                home_pe);
            if (++pending >= window) {
                aclshmem_quiet();
                pst->last_quiet_cycle = GetSystemCycle();
                pending = 0u;
            }
        }
        pst->issued += count;
    }
    if (pending != 0u) {
        aclshmem_quiet();
        pst->last_quiet_cycle = GetSystemCycle();
    }
}

__aicore__ inline void DynPublishStreamReady(
    __gm__ uint8_t *sym, __gm__ DynCsrCtrl *ctrl, uint32_t slot,
    int32_t home_pe, __gm__ DynCsrProducerStats *pst)
{
    __gm__ uint8_t *ready_bytes =
        sym + ctrl->ready_generation_off +
        static_cast<uint64_t>(slot) * ctrl->ready_stride_bytes;
    __gm__ int32_t *ready =
        reinterpret_cast<__gm__ int32_t *>(ready_bytes);
    aclshmemx_signal_op(
        ready, static_cast<int32_t>(ctrl->generation), ACLSHMEM_SIGNAL_SET,
        home_pe);
    ++pst->ready_signals;
    pst->last_ready_cycle = GetSystemCycle();
}

__aicore__ inline void DynCsrInitAccHalfUb(
    float weight, int n_elems, __ubuf__ uint8_t *half_ub,
    __ubuf__ uint8_t *fp32_acc_ub, int fp32_nbytes, uint32_t evt_mte2v,
    uint32_t evt_vmte2)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evt_mte2v);
    AscendC::LocalTensor<half> half_t =
        IncVecBindHalfUb(half_ub,
                         n_elems * static_cast<int>(sizeof(uint16_t)));
    AscendC::LocalTensor<float> acc =
        IncVecBindFloatUb(fp32_acc_ub, fp32_nbytes);
    AscendC::Cast(acc, half_t, AscendC::RoundMode::CAST_NONE, n_elems);
    AscendC::PipeBarrier<PIPE_V>();
    if (weight != 1.f) {
        AscendC::Muls(acc, acc, weight, n_elems);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evt_vmte2);
}

__aicore__ inline uint32_t DynCsrVectorReduce(
    __gm__ uint8_t *ingress, __gm__ uint16_t *out_row,
    __gm__ uint32_t *slots, __gm__ uint32_t *weights, uint32_t begin,
    uint32_t end, uint32_t hidden, uint32_t tile_bytes,
    bool first_contribution_init, bool wide_vector_tile,
    bool defer_result_visibility, int32_t remote_output_pe = -1)
{
    if (ingress == nullptr || out_row == nullptr || slots == nullptr ||
        weights == nullptr || begin >= end || hidden == 0u ||
        tile_bytes < hidden * sizeof(uint16_t)) {
        return kDynCsrFailVector;
    }
    // 3 fp16 + 2 fp32 live rows consume 21 KiB at 1536 elements, below the
    // qualified 24-KiB AIV UB budget.  This retains MTE3/output overlap with
    // the next input tile; aliasing the dead fp32 temporary looked denser but
    // serialized that overlap and regressed live bandwidth substantially.
    constexpr int kDynCsrWideTileElems = 1536;
    const int tile_elems =
        wide_vector_tile &&
                IncVecVectorUbBytes(kDynCsrWideTileElems) <=
                    INC_VEC_UB_BUDGET_BYTES
            ? kDynCsrWideTileElems
            : IncVecEffectiveTileElemsAicore(kC0VecTileElemsRequested);
    const int tile_limit = wide_vector_tile ? kDynCsrWideTileElems
                                            : INC_VEC_MAX_REPEAT;
    if (tile_elems < 1 || tile_elems > tile_limit ||
        IncVecVectorUbBytes(tile_elems) > INC_VEC_UB_BUDGET_BYTES) {
        return kDynCsrFailVector;
    }

    const int fp16_row = IncVecFp16RowBytes(tile_elems);
    const int fp32_row = IncVecFp32RowBytes(tile_elems);
    __ubuf__ uint8_t *ub_base = reinterpret_cast<__ubuf__ uint8_t *>(0);
    __ubuf__ uint8_t *ub_ping = ub_base;
    __ubuf__ uint8_t *ub_pong =
        ub_base + static_cast<uint64_t>(fp16_row);
    __ubuf__ uint8_t *ub_fp32_temp =
        ub_pong + static_cast<uint64_t>(fp16_row);
    __ubuf__ uint8_t *ub_fp32_acc =
        ub_fp32_temp + static_cast<uint64_t>(fp32_row);
    __ubuf__ uint8_t *ub_fp16_out =
        ub_fp32_acc + static_cast<uint64_t>(fp32_row);

    C0VecInitTruePingpongEvents();
    for (uint32_t e0 = 0; e0 < hidden;
         e0 += static_cast<uint32_t>(tile_elems)) {
        const int n = static_cast<int>(
            hidden - e0 < static_cast<uint32_t>(tile_elems)
                ? hidden - e0
                : static_cast<uint32_t>(tile_elems));
        if (n <= 0 || n > tile_limit) {
            C0VecDrainTruePingpongEvents();
            return kDynCsrFailVector;
        }
        const int fp16_nbytes = n * static_cast<int>(sizeof(uint16_t));
        const int fp32_nbytes = n * static_cast<int>(sizeof(float));
        if (!first_contribution_init) {
            AscendC::LocalTensor<float> acc =
                IncVecBindFloatUb(ub_fp32_acc, fp32_nbytes);
            AscendC::Duplicate(acc, static_cast<float>(0), n);
            AscendC::PipeBarrier<PIPE_V>();
        }

        GM_ADDR first = reinterpret_cast<GM_ADDR>(
            ingress + static_cast<uint64_t>(slots[begin]) * tile_bytes +
            static_cast<uint64_t>(e0) * sizeof(uint16_t));
        C0VecCopyHalfGmToUbEvt(first, ub_ping, fp16_nbytes,
                              kC0VecEvtPingVMte2, kC0VecEvtPingMte2V,
                              nullptr);

        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t local = i - begin;
            const bool ping = (local & 1u) == 0u;
            __ubuf__ uint8_t *current = ping ? ub_ping : ub_pong;
            const uint32_t current_m2v =
                ping ? kC0VecEvtPingMte2V : kC0VecEvtPongMte2V;
            const uint32_t current_vm2 =
                ping ? kC0VecEvtPingVMte2 : kC0VecEvtPongVMte2;
            if (i + 1u < end) {
                const uint32_t next = i + 1u;
                const bool next_ping = ((local + 1u) & 1u) == 0u;
                GM_ADDR next_src = reinterpret_cast<GM_ADDR>(
                    ingress +
                    static_cast<uint64_t>(slots[next]) * tile_bytes +
                    static_cast<uint64_t>(e0) * sizeof(uint16_t));
                C0VecCopyHalfGmToUbEvt(
                    next_src, next_ping ? ub_ping : ub_pong, fp16_nbytes,
                    next_ping ? kC0VecEvtPingVMte2 : kC0VecEvtPongVMte2,
                    next_ping ? kC0VecEvtPingMte2V : kC0VecEvtPongMte2V,
                    nullptr);
            }
            if (first_contribution_init && local == 0u) {
                DynCsrInitAccHalfUb(DynBitsToFloat(weights[i]), n, current,
                                    ub_fp32_acc, fp32_nbytes, current_m2v,
                                    current_vm2);
            } else {
                C0VecAccumHalfUb(DynBitsToFloat(weights[i]), n, current,
                                 ub_fp32_temp, ub_fp32_acc, fp32_nbytes,
                                 current_m2v, current_vm2, nullptr);
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kC0VecEvtMte3V);
        AscendC::LocalTensor<float> fp32_t =
            IncVecBindFloatUb(ub_fp32_acc, fp32_nbytes);
        AscendC::LocalTensor<half> half_t =
            IncVecBindHalfUb(ub_fp16_out, fp16_nbytes);
        AscendC::Cast(half_t, fp32_t, AscendC::RoundMode::CAST_RINT, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kC0VecEvtVMte3);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kC0VecEvtVMte3);
        if (remote_output_pe >= 0) {
            // The reduced tile already resides in VECOUT.  Push it from UB
            // through the INC directly to its final worker, avoiding the
            // local-GM store, cache publication, GM reread and per-row quiet
            // of the former two-stage egress.  The private MTE event below is
            // the credit, so different rows never reuse an in-flight slot.
            aclshmemx_mte_put_nbi(
                out_row + e0,
                reinterpret_cast<__ubuf__ uint16_t *>(ub_fp16_out),
                static_cast<uint32_t>(n), remote_output_pe, 0u);
        } else {
            AscendC::LocalTensor<uint8_t> ub_tensor;
            AscendC::GlobalTensor<uint8_t> gm_tensor;
            AscendC::DataCopyExtParams params(
                1, static_cast<uint32_t>(fp16_nbytes), 0, 0, 0);
            ub_tensor.address_.logicPos =
                static_cast<uint8_t>(AscendC::TPosition::VECOUT);
            ub_tensor.address_.bufferAddr =
                reinterpret_cast<uint64_t>(ub_fp16_out);
            ub_tensor.address_.dataLen = static_cast<uint32_t>(
                IncVecUbAlignUp(fp16_nbytes, 32));
            gm_tensor.SetGlobalBuffer(
                reinterpret_cast<__gm__ uint8_t *>(out_row) +
                static_cast<uint64_t>(e0) * sizeof(uint16_t));
            AscendC::DataCopyPad(gm_tensor, ub_tensor, params);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kC0VecEvtMte3V);
    }
    C0VecDrainTruePingpongEvents();
    // The event drain makes the local MTE transactions complete, while the
    // full barrier is still required before a following SHMEM result put can
    // observe the freshly written GM row.  Removing it produced intermittent
    // zero rows at remote destinations on live hardware.
    if (remote_output_pe < 0 && !defer_result_visibility) {
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    return kDynCsrFailNone;
}

// Rank-local duplicate experts are commonly laid out at a constant logical
// contribution stride.  Gather 2--4 fp16 fragments with one MTE2 descriptor,
// then preserve the general reducer's ordered FP32 accumulation.  Irregular
// plans and larger multiplicities fall back to the ping-pong implementation.
__aicore__ inline uint32_t DynCsrVectorReduceRegularSmall(
    __gm__ uint8_t *ingress, __gm__ uint16_t *out_row,
    __gm__ uint32_t *entries, __gm__ uint32_t *weights, uint32_t begin,
    uint32_t count, uint32_t entry_step, uint32_t hidden,
    uint32_t tile_bytes, bool defer_result_visibility,
    uint32_t ub_base_offset = 0u,
    uint32_t ub_budget_bytes = INC_VEC_UB_BUDGET_BYTES,
    int32_t remote_output_pe = -1)
{
    if (ingress == nullptr || out_row == nullptr || entries == nullptr ||
        weights == nullptr || count < 2u || count > 4u || entry_step == 0u ||
        hidden == 0u || tile_bytes < hidden * sizeof(uint16_t) ||
        ub_base_offset > INC_VEC_UB_BUDGET_BYTES ||
        ub_budget_bytes > INC_VEC_UB_BUDGET_BYTES - ub_base_offset) {
        return kDynCsrFailVector;
    }
    const int bytes_per_elem =
        static_cast<int>(count) * static_cast<int>(sizeof(uint16_t)) +
        2 * static_cast<int>(sizeof(float));
    int tile_elems =
        (static_cast<int>(ub_budget_bytes) / bytes_per_elem / 64) * 64;
    if (tile_elems < 64) return kDynCsrFailVector;

    const int fp16_row = IncVecFp16RowBytes(tile_elems);
    const int fp32_row = IncVecFp32RowBytes(tile_elems);
    __ubuf__ uint8_t *ub_inputs =
        reinterpret_cast<__ubuf__ uint8_t *>(
            static_cast<uint64_t>(ub_base_offset));
    __ubuf__ uint8_t *ub_fp32_temp =
        ub_inputs + static_cast<uint64_t>(count) * fp16_row;
    __ubuf__ uint8_t *ub_fp32_acc =
        ub_fp32_temp + static_cast<uint64_t>(fp32_row);
    // Once the last input has been accumulated, fp32_temp is dead until the
    // next hidden tile.  Reuse its first half for the rounded fp16 output so
    // the vector tile is not narrowed by a separate output allocation.
    __ubuf__ uint8_t *ub_fp16_out = ub_fp32_temp;

    IncVecInitPipeEvents();
    for (uint32_t e0 = 0u; e0 < hidden;
         e0 += static_cast<uint32_t>(tile_elems)) {
        const int n = static_cast<int>(
            hidden - e0 < static_cast<uint32_t>(tile_elems)
                ? hidden - e0 : static_cast<uint32_t>(tile_elems));
        const uint32_t fp16_nbytes =
            static_cast<uint32_t>(n * static_cast<int>(sizeof(uint16_t)));
        const uint32_t packed_fp16_row =
            static_cast<uint32_t>(IncVecUbAlignUp(fp16_nbytes, 32));
        const int fp32_nbytes = n * static_cast<int>(sizeof(float));
        const uint64_t source_step =
            static_cast<uint64_t>(entry_step) * tile_bytes;
        if (source_step < fp16_nbytes ||
            source_step - fp16_nbytes > 0xffffffffull) {
            IncVecDrainPipeEvents();
            return kDynCsrFailVector;
        }

        AscendC::LocalTensor<uint8_t> ub_tensor;
        AscendC::GlobalTensor<uint8_t> gm_tensor;
        AscendC::DataCopyPadExtParams<uint8_t> pad;
        AscendC::DataCopyExtParams gather(
            static_cast<uint16_t>(count), fp16_nbytes,
            static_cast<uint32_t>(source_step - fp16_nbytes), 0u, 0u);
        ub_tensor.address_.logicPos =
            static_cast<uint8_t>(AscendC::TPosition::VECIN);
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ub_inputs);
        ub_tensor.address_.dataLen = packed_fp16_row * count;
        gm_tensor.SetGlobalBuffer(
            ingress + static_cast<uint64_t>(entries[begin]) * tile_bytes +
            static_cast<uint64_t>(e0) * sizeof(uint16_t));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(INC_VEC_EVT_VMTE2);
        AscendC::DataCopyPad(ub_tensor, gm_tensor, gather, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(INC_VEC_EVT_MTE2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(INC_VEC_EVT_MTE2V);

        for (uint32_t local = 0u; local < count; ++local) {
            __ubuf__ uint8_t *half_ub =
                ub_inputs + static_cast<uint64_t>(local) * packed_fp16_row;
            AscendC::LocalTensor<half> half_t =
                IncVecBindHalfUb(half_ub, fp16_nbytes);
            const float weight = DynBitsToFloat(weights[begin + local]);
            if (local == 0u) {
                AscendC::LocalTensor<float> acc =
                    IncVecBindFloatUb(ub_fp32_acc, fp32_nbytes);
                AscendC::Cast(acc, half_t, AscendC::RoundMode::CAST_NONE, n);
                AscendC::PipeBarrier<PIPE_V>();
                if (weight != 1.f) {
                    AscendC::Muls(acc, acc, weight, n);
                }
            } else {
                if (local == 1u) {
                    // The previous tile's MTE3 store reads the aliased temp /
                    // output buffer.  Current MTE2 gather and first-input cast
                    // are disjoint and may overlap it; wait only at the first
                    // operation that reuses fp32_temp.
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                        INC_VEC_EVT_MTE3V);
                }
                AscendC::LocalTensor<float> temp =
                    IncVecBindFloatUb(ub_fp32_temp, fp32_nbytes);
                AscendC::Cast(temp, half_t, AscendC::RoundMode::CAST_NONE, n);
                AscendC::PipeBarrier<PIPE_V>();
                if (weight != 1.f) {
                    AscendC::Muls(temp, temp, weight, n);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::LocalTensor<float> acc =
                    IncVecBindFloatUb(ub_fp32_acc, fp32_nbytes);
                AscendC::Add(acc, acc, temp, n);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(INC_VEC_EVT_VMTE2);

        AscendC::LocalTensor<float> acc =
            IncVecBindFloatUb(ub_fp32_acc, fp32_nbytes);
        AscendC::LocalTensor<half> half_out =
            IncVecBindHalfUb(ub_fp16_out, fp16_nbytes);
        AscendC::Cast(half_out, acc, AscendC::RoundMode::CAST_RINT, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(INC_VEC_EVT_VMTE3);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(INC_VEC_EVT_VMTE3);
        if (remote_output_pe >= 0) {
            aclshmemx_mte_put_nbi(
                out_row + e0,
                reinterpret_cast<__ubuf__ uint16_t *>(ub_fp16_out),
                static_cast<uint32_t>(n), remote_output_pe, 0u);
        } else {
            AscendC::LocalTensor<uint8_t> out_ub;
            AscendC::GlobalTensor<uint8_t> out_gm;
            AscendC::DataCopyExtParams store(
                1u, fp16_nbytes, 0u, 0u, 0u);
            out_ub.address_.logicPos =
                static_cast<uint8_t>(AscendC::TPosition::VECOUT);
            out_ub.address_.bufferAddr =
                reinterpret_cast<uint64_t>(ub_fp16_out);
            out_ub.address_.dataLen = IncVecUbAlignUp(fp16_nbytes, 32);
            out_gm.SetGlobalBuffer(
                reinterpret_cast<__gm__ uint8_t *>(out_row) +
                static_cast<uint64_t>(e0) * sizeof(uint16_t));
            AscendC::DataCopyPad(out_gm, out_ub, store);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(INC_VEC_EVT_MTE3V);
    }
    IncVecDrainPipeEvents();
    if (!defer_result_visibility) AscendC::PipeBarrier<PIPE_ALL>();
    return kDynCsrFailNone;
}

__aicore__ inline uint32_t DynCsrK2IdentityAdd(
    __gm__ uint8_t *ingress, __gm__ uint16_t *out_row,
    __gm__ uint32_t *slots, uint32_t begin, uint32_t hidden,
    uint32_t tile_bytes, int32_t remote_output_pe = -1);

__aicore__ inline uint32_t DynPrepareRankDedupContribution(
    __gm__ uint8_t *sym, __gm__ DynCsrCtrl *ctrl,
    __gm__ uint32_t *physical_slots, uint32_t physical_contribution,
    bool defer_visibility = false, uint32_t ub_base_offset = 0u,
    uint32_t ub_budget_bytes = INC_VEC_UB_BUDGET_BYTES,
    int32_t remote_output_pe = -1)
{
    if ((ctrl->optimization_flags & kDynCsrOptLocalRankPrereduce) == 0u) {
        return kDynCsrFailNone;
    }
    if (ctrl->local_rank_prereduce == 0u ||
        physical_contribution >= ctrl->contribution_count ||
        ctrl->logical_contribution_count == 0u ||
        physical_slots[physical_contribution] >= ctrl->max_ingress_slots) {
        return kDynCsrFailCsr;
    }
    __gm__ uint32_t *offsets = reinterpret_cast<__gm__ uint32_t *>(
        sym + ctrl->local_reduce_offsets_off);
    __gm__ uint32_t *entries = reinterpret_cast<__gm__ uint32_t *>(
        sym + ctrl->local_reduce_entries_off);
    __gm__ uint32_t *weights = reinterpret_cast<__gm__ uint32_t *>(
        sym + ctrl->local_reduce_weights_off);
    const uint32_t begin = offsets[physical_contribution];
    const uint32_t end = offsets[physical_contribution + 1u];
    if (begin >= end || end > ctrl->logical_contribution_count) {
        return kDynCsrFailCsr;
    }
    for (uint32_t p = begin; p < end; ++p) {
        if (entries[p] >= ctrl->logical_contribution_count) {
            return kDynCsrFailCsr;
        }
    }
    __gm__ uint16_t *physical = reinterpret_cast<__gm__ uint16_t *>(
        sym + ctrl->ingress_off +
        static_cast<uint64_t>(physical_slots[physical_contribution]) *
            ctrl->tile_bytes);
    const uint32_t count = end - begin;
    constexpr uint32_t kFp32OneBits = 0x3f800000u;
    if (count == 2u && weights[begin] == kFp32OneBits &&
        weights[begin + 1u] == kFp32OneBits) {
        return DynCsrK2IdentityAdd(
            sym + ctrl->logical_input_off, physical, entries, begin,
            ctrl->hidden, ctrl->tile_bytes, remote_output_pe);
    }
    // The regular-small kernel drains all of its private pipe events before
    // returning.  A caller may therefore defer only the final GM visibility
    // barrier while it prepares a disjoint two-slot transport batch; the
    // batch-level PIPE_ALL + cache publication still precedes every SHMEM
    // source read.  This keeps arbitrary 2--4-way local multiplicity on the
    // vector fast path without weakening the release ordering.
    if (count >= 2u && count <= 4u) {
        const uint32_t first = entries[begin];
        const uint32_t second = entries[begin + 1u];
        const uint32_t step = second > first ? second - first : 0u;
        bool regular = step != 0u;
        for (uint32_t local = 2u; regular && local < count; ++local) {
            regular = entries[begin + local] == first + local * step;
        }
        if (regular) {
            return DynCsrVectorReduceRegularSmall(
                sym + ctrl->logical_input_off, physical, entries, weights,
                begin, count, step, ctrl->hidden, ctrl->tile_bytes,
                defer_visibility, ub_base_offset, ub_budget_bytes,
                remote_output_pe);
        }
    }
    // A transport MTE owns [0, ub_base_offset).  An irregular or wider local
    // reducer cannot safely fall back to the generic base-zero UB layout
    // while that transfer is in flight; the host predicates this fast path,
    // and this check fails closed if plan metadata ever disagrees.
    if (ub_base_offset != 0u) return kDynCsrFailVector;
    return DynCsrVectorReduce(
        sym + ctrl->logical_input_off, physical, entries, weights, begin, end,
        ctrl->hidden, ctrl->tile_bytes, true,
        (ctrl->optimization_flags & kDynCsrOptWideVectorTile) != 0u,
        defer_visibility);
}

// K1 identity-weight combine is a copy, not a floating-point reduction.
// Keep the predicate exact so arbitrary weighted K1 plans retain the general
// FP32 accumulate path.  Chunking removes any hidden-size/UB-size coupling.
__aicore__ inline uint32_t DynCsrK1IdentityCopy(
    __gm__ uint8_t *ingress, __gm__ uint16_t *out_row,
    __gm__ uint32_t *slots, uint32_t contribution, uint32_t hidden,
    uint32_t tile_bytes)
{
    if (ingress == nullptr || out_row == nullptr || slots == nullptr ||
        hidden == 0u || tile_bytes < hidden * sizeof(uint16_t)) {
        return kDynCsrFailVector;
    }
    __ubuf__ uint8_t *ub = reinterpret_cast<__ubuf__ uint8_t *>(0);
    GM_ADDR src = reinterpret_cast<GM_ADDR>(
        ingress + static_cast<uint64_t>(slots[contribution]) * tile_bytes);
    GM_ADDR dst = reinterpret_cast<GM_ADDR>(out_row);
    DcLlGatherCopyGm2GmChunked(dst, src, tile_bytes, ub, kDcLlGatherEvent);
    AscendC::PipeBarrier<PIPE_ALL>();
    return kDynCsrFailNone;
}

// The exact sum of two binary16 values is representable in binary32.  For
// unit weights, rounding that sum once to binary16 is therefore identical to
// the generic Cast(fp16->fp32), Add(fp32), Cast(fp32->fp16) contract.  Keep
// this predicate at exactly two unit-weight contributions; weighted or K>2
// reductions retain the order-stable FP32 accumulator.
__aicore__ inline uint32_t DynCsrK2IdentityAdd(
    __gm__ uint8_t *ingress, __gm__ uint16_t *out_row,
    __gm__ uint32_t *slots, uint32_t begin, uint32_t hidden,
    uint32_t tile_bytes, int32_t remote_output_pe)
{
    if (ingress == nullptr || out_row == nullptr || slots == nullptr ||
        hidden == 0u || tile_bytes < hidden * sizeof(uint16_t)) {
        return kDynCsrFailVector;
    }
    // Two fp16 inputs are the only simultaneously live UB rows; the sum
    // reuses the first row in place.  Derive the largest 256-element tile
    // from the platform-qualified vector UB budget instead of leaving three
    // quarters of UB idle.  This changes neither AIV roles nor semantics and
    // applies uniformly to every exact two-contribution identity result.
    constexpr int kTileElems =
        (INC_VEC_UB_BUDGET_BYTES /
         (2 * static_cast<int>(sizeof(uint16_t))) / 256) * 256;
    const int fp16_row = IncVecFp16RowBytes(kTileElems);
    if (fp16_row * 2 > INC_VEC_UB_BUDGET_BYTES) {
        return kDynCsrFailVector;
    }
    __ubuf__ uint8_t *ping = reinterpret_cast<__ubuf__ uint8_t *>(0);
    __ubuf__ uint8_t *pong = ping + static_cast<uint64_t>(fp16_row);
    C0VecInitTruePingpongEvents();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
    bool first = true;
    for (uint32_t e0 = 0u; e0 < hidden; e0 += kTileElems) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
        if (!first) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                kC0VecEvtPingVMte2);
        }
        first = false;
        const int n = static_cast<int>(
            hidden - e0 < static_cast<uint32_t>(kTileElems)
                ? hidden - e0 : static_cast<uint32_t>(kTileElems));
        const int bytes = n * static_cast<int>(sizeof(uint16_t));
        GM_ADDR src0 = reinterpret_cast<GM_ADDR>(
            ingress + static_cast<uint64_t>(slots[begin]) * tile_bytes +
            static_cast<uint64_t>(e0) * sizeof(uint16_t));
        GM_ADDR src1 = reinterpret_cast<GM_ADDR>(
            ingress + static_cast<uint64_t>(slots[begin + 1u]) * tile_bytes +
            static_cast<uint64_t>(e0) * sizeof(uint16_t));
        C0VecCopyHalfGmToUbEvt(src0, ping, bytes, kC0VecEvtPingVMte2,
                              kC0VecEvtPingMte2V, nullptr);
        C0VecCopyHalfGmToUbEvt(src1, pong, bytes, kC0VecEvtPongVMte2,
                              kC0VecEvtPongMte2V, nullptr);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            kC0VecEvtPingMte2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            kC0VecEvtPongMte2V);
        AscendC::LocalTensor<half> a = IncVecBindHalfUb(ping, bytes);
        AscendC::LocalTensor<half> b = IncVecBindHalfUb(pong, bytes);
        AscendC::Add(a, a, b, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
            kC0VecEvtPongVMte2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kC0VecEvtVMte3);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kC0VecEvtVMte3);
        if (remote_output_pe >= 0) {
            aclshmemx_mte_put_nbi(
                out_row + e0,
                reinterpret_cast<__ubuf__ uint16_t *>(ping),
                static_cast<uint32_t>(n), remote_output_pe, 0u);
        } else {
            AscendC::LocalTensor<uint8_t> ub_tensor;
            AscendC::GlobalTensor<uint8_t> gm_tensor;
            AscendC::DataCopyExtParams params(
                1, static_cast<uint32_t>(bytes), 0, 0, 0);
            ub_tensor.address_.logicPos =
                static_cast<uint8_t>(AscendC::TPosition::VECOUT);
            ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ping);
            ub_tensor.address_.dataLen =
                static_cast<uint32_t>(IncVecUbAlignUp(bytes, 32));
            gm_tensor.SetGlobalBuffer(
                reinterpret_cast<__gm__ uint8_t *>(out_row) +
                static_cast<uint64_t>(e0) * sizeof(uint16_t));
            AscendC::DataCopyPad(gm_tensor, ub_tensor, params);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kC0VecEvtPingVMte2);
    C0VecDrainTruePingpongEvents();
    return kDynCsrFailNone;
}

__aicore__ inline uint32_t DynCsrK1StrictIncRelayIssue(
    __gm__ uint8_t *ingress, __gm__ uint16_t *remote_out,
    __gm__ uint32_t *slots, uint32_t contribution, uint32_t tile_bytes,
    int32_t remote_output_pe, bool *busy, uint32_t *ping)
{
    if (ingress == nullptr || remote_out == nullptr || slots == nullptr ||
        tile_bytes == 0u || remote_output_pe < 0 || busy == nullptr ||
        ping == nullptr || *ping > 1u) {
        return kDynCsrFailVector;
    }
    const uint32_t bi = *ping;
    if (busy[bi]) {
        if (bi == 0u) AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
        else if (bi == 1u) AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
        else AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
        busy[bi] = false;
    }
    // Split the platform-qualified UB budget evenly between the two private
    // relay credits.  The previous fixed 8-KiB slots left a third of UB idle
    // and coupled this portable protocol to one historical qualification
    // machine.  Alignment is guaranteed by the shared UB budget contract.
    constexpr uint32_t kK1RelaySlotBytes =
        static_cast<uint32_t>(INC_VEC_UB_BUDGET_BYTES / 2u);
    const uint32_t ub_tile = tile_bytes < kK1RelaySlotBytes
        ? tile_bytes : kK1RelaySlotBytes;
    __ubuf__ uint8_t *ub = reinterpret_cast<__ubuf__ uint8_t *>(
        static_cast<uint64_t>(bi) * kK1RelaySlotBytes);
    __gm__ uint8_t *src =
        ingress + static_cast<uint64_t>(slots[contribution]) * tile_bytes;
    aclshmemx_mte_put_nbi(
        reinterpret_cast<__gm__ uint8_t *>(remote_out), src, ub, ub_tile,
        tile_bytes, remote_output_pe, bi);
    if (bi == 0u) AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
    else if (bi == 1u) AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
    else AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
    busy[bi] = true;
    *ping = 1u - bi;
    return kDynCsrFailNone;
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
inc_dc_sv2_dyn_csr_producer_kernel(__gm__ uint8_t *sym, uint64_t ctrl_off)
{
    if ASCEND_IS_AIV {
        const uint32_t lane = static_cast<uint32_t>(GetBlockIdx());
        __gm__ DynCsrCtrl *ctrl = (__gm__ DynCsrCtrl *)(sym + ctrl_off);
        DynDcci((__gm__ uint8_t *)ctrl, kDynCsrCtrlBytes);
        if (ctrl->magic != kDynCsrMagic || ctrl->tile_bytes == 0u ||
            ctrl->producer_lane_count == 0u ||
            lane >= ctrl->producer_lane_count) {
            return;
        }
        __gm__ DynCsrProducerStats *pst =
            (__gm__ DynCsrProducerStats *)(
                sym + ctrl->owner_stats_off +
                static_cast<uint64_t>(lane) * sizeof(DynCsrProducerStats));
        pst->lane = lane;
        pst->done = 0u;
        pst->issued = 0u;
        pst->ready_signals = 0u;
        pst->first_issue_cycle = 0u;
        pst->last_quiet_cycle = 0u;
        pst->last_ready_cycle = 0u;
        pst->reserved[0] = kDynCsrFailNone;
        pst->reserved[1] = 0u;
        uint64_t producer_t0 = 0u;
        __gm__ uint32_t *sources =
            (__gm__ uint32_t *)(sym + ctrl->contrib_source_rank_off);
        __gm__ uint32_t *slots =
            (__gm__ uint32_t *)(sym + ctrl->contrib_slot_off);
        __gm__ uint32_t *worker_pes =
            (__gm__ uint32_t *)(sym + ctrl->worker_pe_off);
        const uint32_t quiet_window =
            ctrl->producer_quiet_window == 0u ? 1u
                                              : ctrl->producer_quiet_window;
        if (!DynWaitPersistentLocalTrigger(sym, ctrl)) {
            pst->done = 0xffffffffu;
            AscendC::PipeBarrier<PIPE_ALL>();
            DynDcci(reinterpret_cast<__gm__ uint8_t *>(pst), sizeof(*pst));
            return;
        }
        producer_t0 = GetSystemCycle();
        if (lane == 0u) {
            DynPersistentServiceStart(sym, ctrl, producer_t0);
        }
        if (ctrl->ready_mode == 2u || ctrl->ready_mode == 3u ||
            ctrl->ready_mode == 4u || ctrl->ready_mode == 5u ||
            ctrl->ready_mode == 6u) {
            if (ctrl->worker_count == 0u ||
                ctrl->this_worker_rank >= ctrl->worker_count ||
                ctrl->owner_count == 0u ||
                ctrl->group_count != ctrl->owner_count * ctrl->worker_count ||
                ctrl->ready_stride_bytes < sizeof(int32_t)) {
                return;
            }
            __gm__ uint32_t *group_offsets =
                (__gm__ uint32_t *)(sym + ctrl->group_offsets_off);
            __gm__ uint32_t *group_entries =
                (__gm__ uint32_t *)(sym + ctrl->group_entries_off);
            __gm__ uint32_t *contrib_result =
                (__gm__ uint32_t *)(sym + ctrl->contrib_result_off);
            __gm__ uint32_t *source_contribution_offsets =
                (__gm__ uint32_t *)(
                    sym + ctrl->source_contribution_offsets_off);
            __gm__ uint32_t *source_contribution_entries =
                (__gm__ uint32_t *)(
                    sym + ctrl->source_contribution_entries_off);

            // Freeze worker-local rank-dedup staging before any SHMEM source
            // read.  Physical contributions own unique slots; the local
            // generation barrier separates the multi-AIV vector-production
            // phase from the multi-AIV transport phase.
            const bool staged_rank_dedup =
                (ctrl->optimization_flags &
                 kDynCsrOptLocalRankPrereduce) != 0u &&
                (ctrl->optimization_flags &
                 kDynCsrOptStagedRankPrereduce) != 0u;
            if (staged_rank_dedup) {
                bool stage_ok = true;
                for (uint32_t i = lane; i < ctrl->contribution_count;
                     i += ctrl->producer_lane_count) {
                    if (sources[i] != ctrl->this_worker_rank) continue;
                    if (DynPrepareRankDedupContribution(sym, ctrl, slots, i,
                                                        false) !=
                        kDynCsrFailNone) {
                        stage_ok = false;
                        break;
                    }
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                dcci_entire_cache();
                pst->reserved[0] =
                    stage_ok ? kDynCsrFailNone : kDynCsrFailVector;
                pst->reserved[1] = ctrl->generation;
                DynDcci(reinterpret_cast<__gm__ uint8_t *>(pst),
                        sizeof(*pst));
                for (uint32_t peer = 0u;
                     stage_ok && peer < ctrl->producer_lane_count; ++peer) {
                    __gm__ DynCsrProducerStats *peer_stats =
                        reinterpret_cast<__gm__ DynCsrProducerStats *>(
                            sym + ctrl->owner_stats_off +
                            static_cast<uint64_t>(peer) *
                                sizeof(DynCsrProducerStats));
                    uint32_t spins = 0u;
                    while (peer_stats->reserved[1] != ctrl->generation &&
                           spins < ctrl->ready_spin_cap) {
                        DynDcci(reinterpret_cast<__gm__ uint8_t *>(peer_stats),
                                sizeof(*peer_stats));
                        ++spins;
                    }
                    if (peer_stats->reserved[1] != ctrl->generation ||
                        peer_stats->reserved[0] != kDynCsrFailNone) {
                        stage_ok = false;
                    }
                }
                dcci_entire_cache();
                if (!stage_ok) {
                    pst->done = 0xffffffffu;
                    pst->reserved[0] = kDynCsrFailVector;
                    AscendC::PipeBarrier<PIPE_ALL>();
                    DynDcci(reinterpret_cast<__gm__ uint8_t *>(pst),
                            sizeof(*pst));
                    return;
                }
            }

            if (ctrl->ready_mode == 3u) {
                // All producer lanes shard every destination INC/source
                // batch.  This matters for the single-INC case: assigning a
                // whole INC to one lane would otherwise leave every lane but
                // lane 0 idle.  Lane 0 publishes the one ready cacheline only
                // after every payload shard has completed remotely.
                bool producer_ok = true;
                do {
                    const uint32_t group = ctrl->this_worker_rank;
                    const uint32_t begin = source_contribution_offsets[group];
                    const uint32_t end =
                        source_contribution_offsets[group + 1u];
                    if (end <= begin || end > ctrl->contribution_count) {
                        continue;
                    }
                    const uint32_t group_size = end - begin;
                    const uint32_t lane_begin =
                        begin + static_cast<uint32_t>(
                                    (static_cast<uint64_t>(group_size) * lane) /
                                    ctrl->producer_lane_count);
                    const uint32_t lane_end =
                        begin + static_cast<uint32_t>(
                                    (static_cast<uint64_t>(group_size) *
                                     (lane + 1u)) /
                                    ctrl->producer_lane_count);
                    const uint32_t home_pe = ctrl->inc_pe;
                    bool valid = true;
                    uint32_t pending = 0u;
                    const bool coalesced =
                        (ctrl->optimization_flags &
                         kDynCsrOptCoalescedGroupPut) != 0u;
                    uint32_t first_slot = 0u;
                    for (uint32_t p = lane_begin; p < lane_end; ++p) {
                        const uint32_t i = source_contribution_entries[p];
                        if (i >= ctrl->contribution_count ||
                            sources[i] != ctrl->this_worker_rank ||
                            slots[i] >= ctrl->max_ingress_slots) {
                            valid = false;
                            break;
                        }
                        if (!staged_rank_dedup &&
                            DynPrepareRankDedupContribution(sym, ctrl, slots,
                                                           i) !=
                            kDynCsrFailNone) {
                            valid = false;
                            break;
                        }
                        if (p == lane_begin) {
                            first_slot = slots[i];
                        } else if (coalesced &&
                                   slots[i] != first_slot + (p - lane_begin)) {
                            valid = false;
                            break;
                        }
                        if (coalesced) {
                            continue;
                        }
                        __gm__ uint8_t *tile =
                            sym + ctrl->ingress_off +
                            static_cast<uint64_t>(slots[i]) *
                                ctrl->tile_bytes;
                        if (pst->issued == 0u) {
                            pst->first_issue_cycle = GetSystemCycle();
                        }
                        aclshmem_putmem_nbi(
                            reinterpret_cast<__gm__ void *>(tile),
                            reinterpret_cast<__gm__ void *>(tile),
                            ctrl->tile_bytes, static_cast<int32_t>(home_pe));
                        ++pst->issued;
                        if (++pending >= quiet_window) {
                            aclshmem_quiet();
                            pst->last_quiet_cycle = GetSystemCycle();
                            pending = 0u;
                        }
                    }
                    if (!valid) {
                        producer_ok = false;
                        continue;
                    }
                    if (coalesced && lane_end > lane_begin) {
                        __gm__ uint8_t *tile =
                            sym + ctrl->ingress_off +
                            static_cast<uint64_t>(first_slot) *
                                ctrl->tile_bytes;
                        DynPutContiguousGroup(
                            tile, lane_end - lane_begin, ctrl->tile_bytes,
                            ctrl->coalesced_chunk_bytes, quiet_window,
                            static_cast<int32_t>(home_pe), pst);
                    }
                    if (!coalesced && pending != 0u) {
                        aclshmem_quiet();
                        pst->last_quiet_cycle = GetSystemCycle();
                    }
                } while (false);
                if (!producer_ok) {
                    pst->reserved[0] = kDynCsrFailSlot;
                }
                // Each lane has quieted every payload slice it owns.  Publish
                // that fact in a lane-private cacheline before lane 0 sends
                // any destination-ready generation.
                pst->reserved[1] = ctrl->generation;
                AscendC::PipeBarrier<PIPE_ALL>();
                DynDcci(reinterpret_cast<__gm__ uint8_t *>(pst),
                        sizeof(*pst));

                if (lane == 0u) {
                    producer_ok = producer_ok &&
                                  DynWaitProducerPeerPayloads(
                                      sym, ctrl, ctrl->generation);
                    do {
                        const uint32_t group = ctrl->this_worker_rank;
                        const uint32_t begin = source_contribution_offsets[group];
                        const uint32_t end =
                            source_contribution_offsets[group + 1u];
                        if (end <= begin || end > ctrl->contribution_count) {
                            continue;
                        }
                        const uint32_t home_pe = ctrl->inc_pe;
                        __gm__ int32_t *ready =
                        reinterpret_cast<__gm__ int32_t *>(
                            sym + ctrl->ready_generation_off +
                            static_cast<uint64_t>(group) *
                                ctrl->ready_stride_bytes);
                    // Publish generation through the same RMA+quiet path as
                    // payload.  On this platform signal_op can return while a
                    // peer keeps observing the old mapped cacheline for
                    // milliseconds; a cacheline RMA has explicit remote
                    // completion and preserves the 64-byte ready ownership.
                        *ready = producer_ok
                                     ? static_cast<int32_t>(ctrl->generation)
                                     : -static_cast<int32_t>(ctrl->generation);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        DynDcci(reinterpret_cast<__gm__ uint8_t *>(ready),
                                ctrl->ready_stride_bytes);
                        aclshmem_putmem_nbi(
                            reinterpret_cast<__gm__ void *>(ready),
                            reinterpret_cast<__gm__ void *>(ready),
                            ctrl->ready_stride_bytes,
                            static_cast<int32_t>(home_pe));
                        aclshmem_quiet();
                        ++pst->ready_signals;
                        pst->last_ready_cycle = GetSystemCycle();
                    } while (false);
                }
                pst->kernel_cycles = GetSystemCycle() - producer_t0;
                if (ctrl->device_completion != 0u && lane == 0u) {
                    const uint64_t completion =
                        DynCollectiveCompletionBase(ctrl) +
                        ctrl->this_worker_rank;
                    __gm__ int32_t *done =
                        reinterpret_cast<__gm__ int32_t *>(
                            sym + ctrl->ready_generation_off +
                            completion * ctrl->ready_stride_bytes);
                    bool completion_ok = producer_ok &&
                        DynWaitCollectiveValue(
                            ctrl, done,
                            static_cast<int32_t>(ctrl->generation));
                    if (completion_ok) {
                        completion_ok = DynAckCollectiveCompletion(
                            sym, ctrl, ctrl->generation);
                    }
                    pst->done = completion_ok ? ctrl->generation
                                              : 0xffffffffu;
                } else {
                    pst->done = ctrl->generation;
                }
                if (!producer_ok && pst->reserved[0] == kDynCsrFailNone) {
                    pst->reserved[0] = kDynCsrFailMissing;
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                DynDcci(reinterpret_cast<__gm__ uint8_t *>(pst),
                        sizeof(*pst));
                if (lane == 0u) {
                    DynPersistentServiceEnd(
                        sym, ctrl,
                        pst->done != ctrl->generation
                            ? kDynCsrFailMissing
                            : pst->reserved[0]);
                }
                return;
            }

            if (ctrl->ready_mode == 6u) {
                // Source/INC-major streaming: pack the complete contribution
                // train from one source to one INC, split it into large
                // globally indexed chunks, and shard chunks across all
                // producer lanes.  This mirrors reduce-scatter chunk trains
                // in mature collective libraries and avoids an RMA boundary
                // at every owner.
                bool producer_ok = true;
                uint64_t local_reduce_cycles = 0u;
                uint64_t local_transport_cycles = 0u;
                // Long packets use a private MTE slot.  Keep two independent
                // UB/event slots in flight, then close the batch before
                // publishing either ready record.  This is a real two-deep
                // transport pipeline while retaining the strict invariant
                // payload remote-complete -> release signal.  Publishing on
                // the event alone is unsafe because the remote write may not
                // yet be globally visible.
                const uint32_t stream_ub_tile = 8u * 1024u;
                __ubuf__ uint8_t *stream_ub[2] = {
                    reinterpret_cast<__ubuf__ uint8_t *>(0),
                    reinterpret_cast<__ubuf__ uint8_t *>(
                        static_cast<uint64_t>(stream_ub_tile))};
                bool stream_busy[2] = {false, false};
                uint32_t stream_slot[2] = {0u, 0u};
                uint32_t stream_packet_bytes[2] = {0u, 0u};
                uint32_t stream_tile_count[2] = {0u, 0u};
                int32_t stream_home[2] = {0, 0};
                uint32_t stream_pending = 0u;
                const bool mn_rank_dedup_pipeline =
                    (ctrl->optimization_flags &
                     kDynCsrOptAsyncRankPrereducePush) != 0u &&
                    ctrl->local_rank_prereduce != 0u &&
                    !staged_rank_dedup && ctrl->producer_lane_count >= 4u &&
                    ctrl->coalesced_chunk_bytes >
                        kIncDcPrivateMtePacketBytes;
                if (mn_rank_dedup_pipeline) {
                    // M:N bounded producer/consumer pipeline.  Most worker
                    // AIVs keep vector reduction throughput; a small cohort
                    // owns transport.  Each reducer exposes one generation-
                    // tagged GM chunk at a time and waits for its TX consumer
                    // before the next cache publication.  This permits local
                    // reduce and push to overlap without allowing a DCCI to
                    // invalidate an in-flight MTE source.
                    const uint32_t aggregate_tx_target =
                        (ctrl->owner_count + 2u) / 3u;
                    uint32_t transport_lanes =
                        (aggregate_tx_target + ctrl->worker_count - 1u) /
                        ctrl->worker_count;
                    if (transport_lanes == 0u) transport_lanes = 1u;
                    const uint32_t max_transport =
                        ctrl->producer_lane_count / 3u;
                    if (transport_lanes > max_transport) {
                        transport_lanes = max_transport;
                    }
                    if (transport_lanes == 0u) transport_lanes = 1u;
                    const uint32_t reducer_lanes =
                        ctrl->producer_lane_count - transport_lanes;
                    const bool transport_lane = lane >= reducer_lanes;
                    const uint32_t tx_index =
                        transport_lane ? lane - reducer_lanes : 0u;

                    uint32_t chunk_tiles =
                        ctrl->tile_bytes == 0u
                            ? 1u
                            : ctrl->coalesced_chunk_bytes / ctrl->tile_bytes;
                    if (chunk_tiles == 0u) chunk_tiles = 1u;
                    do {
                        const uint32_t group = ctrl->this_worker_rank;
                        const uint32_t begin =
                            source_contribution_offsets[group];
                        const uint32_t end =
                            source_contribution_offsets[group + 1u];
                        if (end <= begin || end > ctrl->contribution_count) {
                            continue;
                        }
                        const uint32_t home_pe = ctrl->inc_pe;
                        const uint32_t first_i =
                            source_contribution_entries[begin];
                        if (first_i >= ctrl->contribution_count ||
                            sources[first_i] != ctrl->this_worker_rank ||
                            slots[first_i] >= ctrl->max_ingress_slots) {
                            producer_ok = false;
                            break;
                        }
                        const uint32_t first_slot = slots[first_i];
                        const uint32_t group_size = end - begin;
                        const uint32_t chunk_count =
                            (group_size + chunk_tiles - 1u) / chunk_tiles;
                        const uint32_t sequence_count =
                            (chunk_count + reducer_lanes - 1u) /
                            reducer_lanes;
                    const uint32_t sequence_span = sequence_count + 1u;
                    const uint32_t token_base =
                        ctrl->generation * sequence_span;
                    __gm__ DynCsrProducerStats *cohort_lead =
                        reinterpret_cast<__gm__ DynCsrProducerStats *>(
                            sym + ctrl->owner_stats_off +
                            static_cast<uint64_t>(
                                ctrl->producer_lane_count) *
                                sizeof(DynCsrProducerStats));

                    if (!transport_lane) {
                            for (uint32_t chunk = lane; chunk < chunk_count;
                                 chunk += reducer_lanes) {
                                const uint32_t sequence =
                                    chunk / reducer_lanes + 1u;
                                const uint32_t token = token_base + sequence;
                                const uint32_t local_offset =
                                    chunk * chunk_tiles;
                                const uint32_t first = begin + local_offset;
                                const uint32_t count =
                                    end - first < chunk_tiles
                                        ? end - first : chunk_tiles;
                                const uint64_t reduce_t0 = GetSystemCycle();
                                bool valid = true;
                                for (uint32_t p = first; p < first + count;
                                     ++p) {
                                    const uint32_t i =
                                        source_contribution_entries[p];
                                    if (i >= ctrl->contribution_count ||
                                        sources[i] !=
                                            ctrl->this_worker_rank ||
                                        slots[i] !=
                                            first_slot + (p - begin) ||
                                        slots[i] >=
                                            ctrl->max_ingress_slots ||
                                        DynPrepareRankDedupContribution(
                                            sym, ctrl, slots, i, true) !=
                                            kDynCsrFailNone) {
                                        valid = false;
                                        break;
                                    }
                                }
                                local_reduce_cycles +=
                                    GetSystemCycle() - reduce_t0;
                                if (!valid) {
                                    producer_ok = false;
                                    break;
                                }
                                // Reduction of sequence N overlaps transport
                                // of N-1.  Delay publication until this
                                // reducer's previous private-MTE consumer has
                                // drained; lane 0 below then verifies the same
                                // condition cohort-wide before the one DCCI.
                                if (sequence > 1u) {
                                    const uint32_t previous = token - 1u;
                                    uint32_t spins = 0u;
                                    while (pst->reserved[3] != previous &&
                                           spins < ctrl->ready_spin_cap) {
                                        DynDcci(
                                            reinterpret_cast<__gm__ uint8_t *>(
                                                pst), sizeof(*pst));
                                        ++spins;
                                    }
                                    if (pst->reserved[3] != previous) {
                                        producer_ok = false;
                                        break;
                                    }
                                }
                                pst->reserved[1] = token;
                                AscendC::PipeBarrier<PIPE_ALL>();
                                DynDcci(
                                    reinterpret_cast<__gm__ uint8_t *>(pst),
                                    sizeof(*pst));
                                if (lane == 0u) {
                                    const uint32_t sequence_base =
                                        (sequence - 1u) * reducer_lanes;
                                    const uint32_t active =
                                        chunk_count - sequence_base <
                                                reducer_lanes
                                            ? chunk_count - sequence_base
                                            : reducer_lanes;
                                    for (uint32_t reducer = 0u;
                                         reducer < active; ++reducer) {
                                        __gm__ DynCsrProducerStats *peer =
                                            reinterpret_cast<
                                                __gm__ DynCsrProducerStats *>(
                                                sym + ctrl->owner_stats_off +
                                                static_cast<uint64_t>(reducer) *
                                                    sizeof(DynCsrProducerStats));
                                        uint32_t spins = 0u;
                                        while (peer->reserved[1] != token &&
                                               spins < ctrl->ready_spin_cap) {
                                            DynDcci(
                                                reinterpret_cast<__gm__ uint8_t *>(
                                                    peer), sizeof(*peer));
                                            ++spins;
                                        }
                                        if (peer->reserved[1] != token) {
                                            producer_ok = false;
                                            break;
                                        }
                                    }
                                    if (!producer_ok) break;
                                    if (sequence > 1u) {
                                        const uint32_t previous = token - 1u;
                                        const uint32_t previous_base =
                                            (sequence - 2u) * reducer_lanes;
                                        const uint32_t previous_active =
                                            chunk_count - previous_base <
                                                    reducer_lanes
                                                ? chunk_count - previous_base
                                                : reducer_lanes;
                                        for (uint32_t reducer = 0u;
                                             reducer < previous_active;
                                             ++reducer) {
                                            __gm__ DynCsrProducerStats *peer =
                                                reinterpret_cast<
                                                    __gm__ DynCsrProducerStats *>(
                                                    sym + ctrl->owner_stats_off +
                                                    static_cast<uint64_t>(reducer) *
                                                        sizeof(DynCsrProducerStats));
                                            uint32_t spins = 0u;
                                            while (peer->reserved[3] != previous &&
                                                   spins < ctrl->ready_spin_cap) {
                                                DynDcci(
                                                    reinterpret_cast<__gm__ uint8_t *>(
                                                        peer), sizeof(*peer));
                                                ++spins;
                                            }
                                            if (peer->reserved[3] != previous) {
                                                producer_ok = false;
                                                break;
                                            }
                                        }
                                    }
                                    if (!producer_ok) break;
                                    AscendC::PipeBarrier<PIPE_ALL>();
                                    dcci_entire_cache();
                                    cohort_lead->reserved[2] = token;
                                    AscendC::PipeBarrier<PIPE_ALL>();
                                    DynDcci(
                                        reinterpret_cast<__gm__ uint8_t *>(
                                            cohort_lead),
                                        sizeof(*cohort_lead));
                                } else {
                                    uint32_t spins = 0u;
                                    while (cohort_lead->reserved[2] != token &&
                                           spins < ctrl->ready_spin_cap) {
                                        DynDcci(
                                            reinterpret_cast<__gm__ uint8_t *>(
                                                cohort_lead),
                                            sizeof(*cohort_lead));
                                        ++spins;
                                    }
                                    if (cohort_lead->reserved[2] != token) {
                                        producer_ok = false;
                                        break;
                                    }
                                }
                            }
                        } else {
                            for (uint32_t sequence = 1u;
                                 sequence <= sequence_count && producer_ok;
                                 ++sequence) {
                                for (uint32_t reducer = tx_index;
                                     reducer < reducer_lanes;
                                     reducer += transport_lanes) {
                                    const uint32_t chunk =
                                        reducer +
                                        (sequence - 1u) * reducer_lanes;
                                    if (chunk >= chunk_count) continue;
                                    __gm__ DynCsrProducerStats *peer =
                                        reinterpret_cast<
                                            __gm__ DynCsrProducerStats *>(
                                            sym + ctrl->owner_stats_off +
                                            static_cast<uint64_t>(reducer) *
                                                sizeof(DynCsrProducerStats));
                                    const uint32_t token =
                                        token_base + sequence;
                                    uint32_t spins = 0u;
                                    while (cohort_lead->reserved[2] != token &&
                                           spins < ctrl->ready_spin_cap) {
                                        DynDcci(
                                            reinterpret_cast<__gm__ uint8_t *>(
                                                cohort_lead),
                                            sizeof(*cohort_lead));
                                        if ((spins & 1023u) == 0u &&
                                            DynAbortRequested(
                                                ctrl, ctrl->generation)) {
                                            break;
                                        }
                                        ++spins;
                                    }
                                    if (cohort_lead->reserved[2] != token) {
                                        producer_ok = false;
                                        break;
                                    }
                                    const uint32_t local_offset =
                                        chunk * chunk_tiles;
                                    const uint32_t count =
                                        group_size - local_offset < chunk_tiles
                                            ? group_size - local_offset
                                            : chunk_tiles;
                                    const uint32_t slot =
                                        first_slot + local_offset;
                                    __gm__ uint8_t *tile =
                                        sym + ctrl->ingress_off +
                                        static_cast<uint64_t>(slot) *
                                            ctrl->tile_bytes;
                                    const uint32_t packet_bytes =
                                        count * ctrl->tile_bytes;
                                    const uint64_t transport_t0 =
                                        GetSystemCycle();
                                    if (pst->issued == 0u) {
                                        pst->first_issue_cycle =
                                            transport_t0;
                                    }
                                    aclshmemx_mte_put_nbi(
                                        tile, tile, stream_ub[0],
                                        stream_ub_tile, packet_bytes,
                                        static_cast<int32_t>(home_pe), 0u);
                                    AscendC::SetFlag<
                                        AscendC::HardEvent::MTE3_MTE2>(0u);
                                    AscendC::WaitFlag<
                                        AscendC::HardEvent::MTE3_MTE2>(0u);
                                    aclshmemx_mte_quiet();
                                    pst->last_quiet_cycle = GetSystemCycle();
                                    pst->issued += count;
                                    DynPublishStreamReady(
                                        sym, ctrl, slot,
                                        static_cast<int32_t>(home_pe), pst);
                                    local_transport_cycles +=
                                        GetSystemCycle() - transport_t0;
                                    peer->reserved[3] = token;
                                    AscendC::PipeBarrier<PIPE_ALL>();
                                    DynDcci(
                                        reinterpret_cast<__gm__ uint8_t *>(
                                            peer),
                                        sizeof(*peer));
                                }
                            }
                        }
                    } while (false);
                } else {
                // All lanes execute the same reduce->push state machine, but
                // starting them in lockstep phase-locks the whole worker:
                // every AIV consumes vector/HBM bandwidth, then every AIV
                // switches to MTE/HCCS together.  Release the second runtime-
                // sized wave after lane 0 has prepared its first transport
                // batch.  The task mapping and AIV count are unchanged; this
                // only overlaps the two resource classes across lanes.
                const bool rank_dedup_wavefront =
                    ctrl->local_rank_prereduce != 0u &&
                    !staged_rank_dedup && ctrl->producer_lane_count >= 4u;
                const bool async_rank_dedup_push =
                    (ctrl->optimization_flags &
                     kDynCsrOptAsyncRankPrereducePush) != 0u &&
                    ctrl->local_rank_prereduce != 0u &&
                    !staged_rank_dedup;
                const bool ub_direct_rank_push =
                    (ctrl->optimization_flags &
                     kDynCsrOptUbDirectRankPush) != 0u &&
                    ctrl->local_rank_prereduce != 0u &&
                    !staged_rank_dedup;
                constexpr uint32_t kAsyncMteUbBytes = 2u * 1024u;
                const uint32_t first_wave_lanes =
                    (ctrl->producer_lane_count + 1u) / 2u;
                __gm__ DynCsrProducerStats *wave_lead =
                    reinterpret_cast<__gm__ DynCsrProducerStats *>(
                        sym + ctrl->owner_stats_off);
                if (rank_dedup_wavefront && lane >= first_wave_lanes) {
                    uint32_t spins = 0u;
                    while (wave_lead->reserved[1] != ctrl->generation &&
                           spins < ctrl->ready_spin_cap) {
                        DynDcci(reinterpret_cast<__gm__ uint8_t *>(wave_lead),
                                sizeof(*wave_lead));
                        if ((spins & 1023u) == 0u &&
                            DynAbortRequested(ctrl, ctrl->generation)) {
                            break;
                        }
                        ++spins;
                    }
                    if (wave_lead->reserved[1] != ctrl->generation) {
                        producer_ok = false;
                    }
                }
                uint32_t chunk_tiles =
                    ctrl->tile_bytes == 0u
                        ? 1u
                        : ctrl->coalesced_chunk_bytes / ctrl->tile_bytes;
                if (chunk_tiles == 0u) {
                    chunk_tiles = 1u;
                }
                do {
                    const uint32_t group = ctrl->this_worker_rank;
                    const uint32_t begin =
                        source_contribution_offsets[group];
                    const uint32_t end =
                        source_contribution_offsets[group + 1u];
                    if (end <= begin || end > ctrl->contribution_count) {
                        continue;
                    }
                    const uint32_t home_pe = ctrl->inc_pe;
                    const uint32_t first_i =
                        source_contribution_entries[begin];
                    if (first_i >= ctrl->contribution_count ||
                        sources[first_i] != ctrl->this_worker_rank ||
                        slots[first_i] >= ctrl->max_ingress_slots) {
                        producer_ok = false;
                        continue;
                    }
                    const uint32_t first_slot = slots[first_i];
                    const uint32_t group_size = end - begin;
                    const uint32_t chunk_count =
                        static_cast<uint32_t>(
                            (static_cast<uint64_t>(group_size) +
                             chunk_tiles - 1u) /
                            chunk_tiles);
                    // CANN 9.1 can corrupt a vector block when adjacent K1
                    // rows share one long-packet credit.  Keep rows as
                    // independent 16-KiB puts, but amortize remote completion
                    // across a bounded train of lane-strided (non-adjacent)
                    // rows.  Ready is published only after the train quiet,
                    // so reducers cannot observe a partially completed row.
                    // The credit and ready ownership are lane-local and the
                    // group is already scoped to one home PE.  They therefore
                    // do not depend on worker count: restricting this protocol
                    // to W2 leaves W4+ issuing one quiet/ready per row even
                    // though the same two private MTE slots are available.
                    const bool batched_k1_rows =
                        ctrl->contribution_count == ctrl->result_count &&
                        ctrl->tile_bytes == kIncDcPrivateMtePacketBytes &&
                        chunk_tiles == 1u &&
                        ctrl->local_rank_prereduce == 0u;
                    if (batched_k1_rows) {
                        constexpr uint32_t kK1RowWindow = 8u;
                        const bool private_mte_push =
                            (ctrl->optimization_flags &
                             kDynCsrOptK1PrivateMtePush) != 0u;
                        const bool pair_ready =
                            private_mte_push &&
                            (ctrl->optimization_flags &
                             kDynCsrOptK1PairReady) != 0u;
                        uint32_t ready_head_slot = 0u;
                        uint32_t pending_rows = 0u;
                        for (uint32_t local_offset = lane;
                             local_offset < group_size;
                             local_offset += ctrl->producer_lane_count) {
                            const uint32_t p = begin + local_offset;
                            const uint32_t i =
                                source_contribution_entries[p];
                            if (i >= ctrl->contribution_count ||
                                sources[i] != ctrl->this_worker_rank ||
                                slots[i] != first_slot + local_offset ||
                                slots[i] >= ctrl->max_ingress_slots) {
                                producer_ok = false;
                                break;
                            }
                            __gm__ uint8_t *tile =
                                sym + ctrl->ingress_off +
                                static_cast<uint64_t>(slots[i]) *
                                    ctrl->tile_bytes;
                            if (pst->issued == 0u) {
                                pst->first_issue_cycle = GetSystemCycle();
                            }
                            if (private_mte_push) {
                                const uint32_t bi = pending_rows;
                                aclshmemx_mte_put_nbi(
                                    tile, tile, stream_ub[bi],
                                    stream_ub_tile, ctrl->tile_bytes,
                                    static_cast<int32_t>(home_pe), bi);
                                if (bi == 0u) {
                                    AscendC::SetFlag<
                                        AscendC::HardEvent::MTE3_MTE2>(0u);
                                } else {
                                    AscendC::SetFlag<
                                        AscendC::HardEvent::MTE3_MTE2>(1u);
                                }
                            } else {
                                aclshmem_putmem_nbi(
                                    reinterpret_cast<__gm__ void *>(tile),
                                    reinterpret_cast<__gm__ void *>(tile),
                                    ctrl->tile_bytes,
                                    static_cast<int32_t>(home_pe));
                            }
                            if (pending_rows == 0u) {
                                ready_head_slot = slots[i];
                            }
                            ++pending_rows;
                            ++pst->issued;
                            const uint32_t drain_window =
                                private_mte_push ? 2u : kK1RowWindow;
                            if (pending_rows == drain_window) {
                                if (private_mte_push) {
                                    AscendC::WaitFlag<
                                        AscendC::HardEvent::MTE3_MTE2>(0u);
                                    AscendC::WaitFlag<
                                        AscendC::HardEvent::MTE3_MTE2>(1u);
                                    aclshmemx_mte_quiet();
                                } else {
                                    aclshmem_quiet();
                                }
                                pst->last_quiet_cycle = GetSystemCycle();
                                const uint32_t ready_count =
                                    pair_ready ? 1u : pending_rows;
                                for (uint32_t q = 0u;
                                     q < ready_count; ++q) {
                                    DynPublishStreamReady(
                                        sym, ctrl,
                                        ready_head_slot +
                                            q * ctrl->producer_lane_count,
                                        static_cast<int32_t>(home_pe), pst);
                                }
                                pending_rows = 0u;
                            }
                        }
                        if (pending_rows != 0u) {
                            if (private_mte_push) {
                                if (pending_rows >= 1u) {
                                    AscendC::WaitFlag<
                                        AscendC::HardEvent::MTE3_MTE2>(0u);
                                }
                                if (pending_rows >= 2u) {
                                    AscendC::WaitFlag<
                                        AscendC::HardEvent::MTE3_MTE2>(1u);
                                }
                                aclshmemx_mte_quiet();
                            } else {
                                aclshmem_quiet();
                            }
                            pst->last_quiet_cycle = GetSystemCycle();
                            const uint32_t ready_count =
                                pair_ready ? 1u : pending_rows;
                            for (uint32_t q = 0u;
                                 q < ready_count; ++q) {
                                DynPublishStreamReady(
                                    sym, ctrl,
                                    ready_head_slot +
                                        q * ctrl->producer_lane_count,
                                    static_cast<int32_t>(home_pe), pst);
                            }
                        }
                        continue;
                    }
                    for (uint32_t chunk = lane; chunk < chunk_count;
                         chunk += ctrl->producer_lane_count) {
                        const uint32_t local_offset = chunk * chunk_tiles;
                        const uint32_t first = begin + local_offset;
                        const uint32_t count =
                            end - first < chunk_tiles ? end - first
                                                      : chunk_tiles;
                        const uint64_t reduce_begin_cycle = GetSystemCycle();
                        bool valid = true;
                        for (uint32_t p = first; p < first + count; ++p) {
                            const uint32_t i =
                                source_contribution_entries[p];
                            if (i >= ctrl->contribution_count ||
                                sources[i] != ctrl->this_worker_rank ||
                                slots[i] != first_slot + (p - begin) ||
                                slots[i] >= ctrl->max_ingress_slots) {
                                valid = false;
                                break;
                            }
                            if (!staged_rank_dedup &&
                                DynPrepareRankDedupContribution(sym, ctrl, slots,
                                                               i, true,
                                                               async_rank_dedup_push
                                                                   ? kAsyncMteUbBytes
                                                                   : 0u,
                                                               async_rank_dedup_push
                                                                   ? INC_VEC_UB_BUDGET_BYTES -
                                                                         kAsyncMteUbBytes
                                                                   : INC_VEC_UB_BUDGET_BYTES,
                                                               ub_direct_rank_push
                                                                   ? static_cast<int32_t>(home_pe)
                                                                   : -1) !=
                                kDynCsrFailNone) {
                                valid = false;
                                break;
                            }
                        }
                        local_reduce_cycles +=
                            GetSystemCycle() - reduce_begin_cycle;
                        if (!valid) {
                            producer_ok = false;
                            continue;
                        }
                        // Every locally reduced row above owns disjoint stable
                        // GM.  One visibility barrier for the complete chunk
                        // replaces a full-pipeline barrier after each row and
                        // still precedes the first SHMEM source read.
                        const uint32_t slot = first_slot + local_offset;
                        __gm__ uint8_t *tile =
                            sym + ctrl->ingress_off +
                            static_cast<uint64_t>(slot) * ctrl->tile_bytes;
                        const uint32_t packet_bytes = count * ctrl->tile_bytes;
                        if (ub_direct_rank_push) {
                            // Every vector store above targeted the symmetric
                            // INC ingress address directly.  The reducer's
                            // MTE3 event drain is the payload completion
                            // credit; publish the chunk generation only now.
                            if (pst->issued == 0u) {
                                pst->first_issue_cycle = GetSystemCycle();
                            }
                            pst->issued += count;
                            DynPublishStreamReady(
                                sym, ctrl, slot,
                                static_cast<int32_t>(home_pe), pst);
                            continue;
                        }
                        const uint32_t owners_per_source =
                            (ctrl->owner_count + ctrl->worker_count - 1u) /
                            ctrl->worker_count;
                        // A rank-deduplicated long packet can use the private
                        // two-slot transport only if both staging chunks are
                        // completely produced before either MTE source read
                        // begins.  This keeps the next whole-cache visibility
                        // operation outside the in-flight interval that made
                        // the earlier naive ping-pong experiment corrupt data.
                        const bool prepared_rank_dedup_batch =
                            packet_bytes > kIncDcPrivateMtePacketBytes &&
                            ctrl->local_rank_prereduce != 0u &&
                            !staged_rank_dedup;
                        const uint64_t transport_begin_cycle =
                            GetSystemCycle();
                        if (ctrl->local_rank_prereduce != 0u &&
                            !staged_rank_dedup &&
                            !prepared_rank_dedup_batch) {
                            AscendC::PipeBarrier<PIPE_ALL>();
                            // A ranged flush is correct here but walks one
                            // cacheline at a time, which costs far more than
                            // the chunk it publishes.  The whole-cache form is
                            // the only affordable primitive, and it is why the
                            // two-deep pipeline stays disabled below: it would
                            // invalidate an in-flight MTE2 read.
                            dcci_entire_cache();
                        }
                        if (prepared_rank_dedup_batch) {
                            if (async_rank_dedup_push) {
                                // The previous private MTE owns only the
                                // first 8 KiB of UB; the regular-small vector
                                // reducer above uses the disjoint remainder.
                                // Thus current reduce overlaps previous push.
                                // Drain the one credit only before the next
                                // cache publication and source-buffer reuse.
                                if (stream_pending != 0u) {
                                    AscendC::WaitFlag<
                                        AscendC::HardEvent::MTE3_MTE2>(0u);
                                    aclshmemx_mte_quiet();
                                    pst->last_quiet_cycle = GetSystemCycle();
                                    DynPublishStreamReady(
                                        sym, ctrl, stream_slot[0],
                                        stream_home[0], pst);
                                    stream_busy[0] = false;
                                    stream_pending = 0u;
                                }
                                AscendC::PipeBarrier<PIPE_ALL>();
                                dcci_entire_cache();
                                if (rank_dedup_wavefront && lane == 0u &&
                                    pst->reserved[1] != ctrl->generation) {
                                    pst->reserved[1] = ctrl->generation;
                                    AscendC::PipeBarrier<PIPE_ALL>();
                                    DynDcci(
                                        reinterpret_cast<__gm__ uint8_t *>(
                                            pst),
                                        sizeof(*pst));
                                }
                                stream_slot[0] = slot;
                                stream_packet_bytes[0] = packet_bytes;
                                stream_tile_count[0] = count;
                                stream_home[0] =
                                    static_cast<int32_t>(home_pe);
                                aclshmemx_mte_put_nbi(
                                    tile, tile, stream_ub[0],
                                    kAsyncMteUbBytes,
                                    packet_bytes,
                                    static_cast<int32_t>(home_pe), 0u);
                                AscendC::SetFlag<
                                    AscendC::HardEvent::MTE3_MTE2>(0u);
                                stream_busy[0] = true;
                                stream_pending = 1u;
                                pst->issued += count;
                            } else {
                            // Queue two fully reduced, disjoint GM chunks.
                            // Visibility is published once for the pair, then
                            // both private MTE slots run concurrently.  The
                            // pair is drained before another vector reduction
                            // or cache operation can touch either source.
                            const uint32_t bi = stream_pending;
                            stream_slot[bi] = slot;
                            stream_packet_bytes[bi] = packet_bytes;
                            stream_tile_count[bi] = count;
                            stream_home[bi] = static_cast<int32_t>(home_pe);
                            stream_pending += 1u;
                            if (stream_pending == 2u) {
                                AscendC::PipeBarrier<PIPE_ALL>();
                                dcci_entire_cache();
                                if (rank_dedup_wavefront && lane == 0u &&
                                    pst->reserved[1] != ctrl->generation) {
                                    pst->reserved[1] = ctrl->generation;
                                    AscendC::PipeBarrier<PIPE_ALL>();
                                    DynDcci(
                                        reinterpret_cast<__gm__ uint8_t *>(
                                            pst),
                                        sizeof(*pst));
                                }
                                for (uint32_t prepared = 0u; prepared < 2u;
                                     ++prepared) {
                                    __gm__ uint8_t *prepared_tile =
                                        sym + ctrl->ingress_off +
                                        static_cast<uint64_t>(
                                            stream_slot[prepared]) *
                                            ctrl->tile_bytes;
                                    aclshmemx_mte_put_nbi(
                                        prepared_tile, prepared_tile,
                                        stream_ub[prepared], stream_ub_tile,
                                        stream_packet_bytes[prepared],
                                        stream_home[prepared], prepared);
                                    if (prepared == 0u) {
                                        AscendC::SetFlag<
                                            AscendC::HardEvent::MTE3_MTE2>(0u);
                                    } else {
                                        AscendC::SetFlag<
                                            AscendC::HardEvent::MTE3_MTE2>(1u);
                                    }
                                    stream_busy[prepared] = true;
                                    pst->issued +=
                                        stream_tile_count[prepared];
                                }
                                AscendC::WaitFlag<
                                    AscendC::HardEvent::MTE3_MTE2>(0u);
                                AscendC::WaitFlag<
                                    AscendC::HardEvent::MTE3_MTE2>(1u);
                                aclshmemx_mte_quiet();
                                pst->last_quiet_cycle = GetSystemCycle();
                                DynPublishStreamReady(
                                    sym, ctrl, stream_slot[0], stream_home[0],
                                    pst);
                                DynPublishStreamReady(
                                    sym, ctrl, stream_slot[1], stream_home[1],
                                    pst);
                                stream_busy[0] = false;
                                stream_busy[1] = false;
                                stream_pending = 0u;
                            }
                            }
                        } else if (packet_bytes >
                                       kIncDcPrivateMtePacketBytes &&
                                   owners_per_source >= 8u &&
                                   ctrl->local_rank_prereduce == 0u) {
                            const uint32_t bi = stream_pending;
                            aclshmemx_mte_put_nbi(
                                tile, tile, stream_ub[bi], stream_ub_tile,
                                packet_bytes, static_cast<int32_t>(home_pe),
                                bi);
                            if (bi == 0u) {
                                AscendC::SetFlag<
                                    AscendC::HardEvent::MTE3_MTE2>(0u);
                            } else {
                                AscendC::SetFlag<
                                    AscendC::HardEvent::MTE3_MTE2>(1u);
                            }
                            stream_busy[bi] = true;
                            stream_slot[bi] = slot;
                            stream_home[bi] =
                                static_cast<int32_t>(home_pe);
                            pst->issued += count;
                            stream_pending += 1u;
                            if (stream_pending == 2u) {
                                AscendC::WaitFlag<
                                    AscendC::HardEvent::MTE3_MTE2>(0u);
                                AscendC::WaitFlag<
                                    AscendC::HardEvent::MTE3_MTE2>(1u);
                                aclshmemx_mte_quiet();
                                pst->last_quiet_cycle = GetSystemCycle();
                                DynPublishStreamReady(
                                    sym, ctrl, stream_slot[0], stream_home[0],
                                    pst);
                                DynPublishStreamReady(
                                    sym, ctrl, stream_slot[1], stream_home[1],
                                    pst);
                                stream_busy[0] = false;
                                stream_busy[1] = false;
                                stream_pending = 0u;
                            }
                        } else {
                            DynPutContiguousGroup(
                                tile, count, ctrl->tile_bytes, packet_bytes,
                                1u, static_cast<int32_t>(home_pe), pst);
                            // The short public path is complete before ready.
                            DynPublishStreamReady(
                                sym, ctrl, slot,
                                static_cast<int32_t>(home_pe), pst);
                        }
                        local_transport_cycles +=
                            GetSystemCycle() - transport_begin_cycle;
                    }
                } while (false);
                // A tiny/tail-only train may never fill two slots.  It still
                // has to release the second wave before draining its tail.
                if (rank_dedup_wavefront && lane == 0u &&
                    pst->reserved[1] != ctrl->generation) {
                    pst->reserved[1] = ctrl->generation;
                    AscendC::PipeBarrier<PIPE_ALL>();
                    DynDcci(reinterpret_cast<__gm__ uint8_t *>(pst),
                            sizeof(*pst));
                }
                if (stream_pending != 0u) {
                    // A rank-dedup tail is prepared but not issued yet; make
                    // it visible and start the private transfer only after no
                    // earlier slot remains in flight.  Non-dedup batches were
                    // already issued in the loop and skip this block.
                    if (ctrl->local_rank_prereduce != 0u &&
                        !staged_rank_dedup && !async_rank_dedup_push) {
                        AscendC::PipeBarrier<PIPE_ALL>();
                        dcci_entire_cache();
                        for (uint32_t bi = 0u; bi < stream_pending; ++bi) {
                            __gm__ uint8_t *prepared_tile =
                                sym + ctrl->ingress_off +
                                static_cast<uint64_t>(stream_slot[bi]) *
                                    ctrl->tile_bytes;
                            aclshmemx_mte_put_nbi(
                                prepared_tile, prepared_tile, stream_ub[bi],
                                stream_ub_tile, stream_packet_bytes[bi],
                                stream_home[bi], bi);
                            if (bi == 0u) {
                                AscendC::SetFlag<
                                    AscendC::HardEvent::MTE3_MTE2>(0u);
                            } else {
                                AscendC::SetFlag<
                                    AscendC::HardEvent::MTE3_MTE2>(1u);
                            }
                            stream_busy[bi] = true;
                            pst->issued += stream_tile_count[bi];
                        }
                    }
                    if (stream_busy[0]) {
                        AscendC::WaitFlag<
                            AscendC::HardEvent::MTE3_MTE2>(0u);
                    }
                    if (stream_busy[1]) {
                        AscendC::WaitFlag<
                            AscendC::HardEvent::MTE3_MTE2>(1u);
                    }
                    aclshmemx_mte_quiet();
                    pst->last_quiet_cycle = GetSystemCycle();
                    for (uint32_t bi = 0u; bi < stream_pending; ++bi) {
                        DynPublishStreamReady(sym, ctrl, stream_slot[bi],
                                              stream_home[bi], pst);
                    }
                }
                }
                pst->reserved[2] = static_cast<uint32_t>(
                    local_reduce_cycles > 0xffffffffull
                        ? 0xffffffffull : local_reduce_cycles);
                if (!mn_rank_dedup_pipeline) {
                    pst->reserved[3] = static_cast<uint32_t>(
                        local_transport_cycles > 0xffffffffull
                            ? 0xffffffffull : local_transport_cycles);
                }
                pst->kernel_cycles = GetSystemCycle() - producer_t0;
                pst->done = producer_ok ? ctrl->generation : 0xffffffffu;
                if (!producer_ok) {
                    pst->reserved[0] = kDynCsrFailSlot;
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                DynDcci(reinterpret_cast<__gm__ uint8_t *>(pst),
                        sizeof(*pst));
                if (ctrl->device_completion != 0u && lane == 0u) {
                    const uint64_t completion =
                        DynCollectiveCompletionBase(ctrl) +
                        ctrl->this_worker_rank;
                    __gm__ int32_t *done =
                        reinterpret_cast<__gm__ int32_t *>(
                            sym + ctrl->ready_generation_off +
                            completion * ctrl->ready_stride_bytes);
                    bool completion_ok = producer_ok &&
                        DynWaitProducerPeerLanes(
                            sym, ctrl, ctrl->generation) &&
                        DynWaitCollectiveValue(
                            ctrl, done,
                            static_cast<int32_t>(ctrl->generation));
                    if (completion_ok) {
                        completion_ok = DynAckCollectiveCompletion(
                            sym, ctrl, ctrl->generation);
                    }
                    pst->done = completion_ok ? ctrl->generation
                                              : 0xffffffffu;
                    AscendC::PipeBarrier<PIPE_ALL>();
                    DynDcci(reinterpret_cast<__gm__ uint8_t *>(pst),
                            sizeof(*pst));
                }
                if (lane == 0u) {
                    // Formal rank timing must cover every producer lane, not
                    // merely whichever owner groups happened to land on
                    // lane 0.  With device completion disabled the INC rank
                    // still supplies the collective end time, but this local
                    // join keeps each worker's elapsed interval truthful.
                    if (ctrl->device_completion == 0u) {
                        producer_ok =
                            producer_ok && DynWaitProducerPeerLanes(
                                               sym, ctrl, ctrl->generation);
                    }
                    DynPersistentServiceEnd(
                        sym, ctrl,
                        (!producer_ok || pst->done != ctrl->generation)
                            ? kDynCsrFailMissing
                            : pst->reserved[0]);
                }
                return;
            }

            if (ctrl->ready_mode == 5u) {
                // Pack by owner/source, but expose bounded chunks.  One
                // contiguous ready-cacheline RMA releases all result rows in
                // the chunk after its payload is remotely complete.
                bool producer_ok = true;
                uint32_t chunk_tiles =
                    ctrl->tile_bytes == 0u
                        ? 1u
                        : ctrl->coalesced_chunk_bytes / ctrl->tile_bytes;
                if (chunk_tiles == 0u) {
                    chunk_tiles = 1u;
                }
                // Breadth-first chunk scheduling: publish chunk 0 for every
                // owner assigned to this lane before chunk 1, and so on.
                // This is the same wavefront used by mature pipelined
                // collectives; reducers for later owners no longer wait for
                // an unrelated earlier owner to drain its complete group.
                const bool head_tile_stream =
                    (ctrl->optimization_flags &
                     kDynCsrOptHeadTileStream) != 0u;
                uint32_t wave = 0u;
                for (uint64_t chunk_offset = 0u;
                     chunk_offset < ctrl->contribution_count; ++wave,
                              chunk_offset +=
                                  (head_tile_stream && chunk_offset == 0u)
                                      ? 1u
                                      : chunk_tiles) {
                    bool any_chunk = false;
                    // A lane commonly serves several owners (48 reducers / 16
                    // producer lanes on the current platform).  A fixed
                    // owner order makes the last owner in every lane trail on
                    // every wave and turns that deterministic skew into the
                    // collective tail.  Rotate the per-lane order each wave;
                    // all workers use the same rotation, so a result still
                    // receives its complete source set together while no
                    // owner is permanently last.  The arithmetic is generic
                    // for non-power-of-two AIV and owner counts.
                    const uint32_t lane_group_count =
                        lane < ctrl->owner_count
                            ? 1u + (ctrl->owner_count - 1u - lane) /
                                       ctrl->producer_lane_count
                            : 0u;
                    for (uint32_t group_pos = 0u;
                         group_pos < lane_group_count; ++group_pos) {
                        const uint32_t rotated =
                            (group_pos + wave % lane_group_count) %
                            lane_group_count;
                        const uint32_t order =
                            lane + rotated * ctrl->producer_lane_count;
                        const uint32_t owner = order;
                        const uint32_t group =
                            owner * ctrl->worker_count +
                            ctrl->this_worker_rank;
                        const uint32_t begin = group_offsets[group];
                        const uint32_t end = group_offsets[group + 1u];
                        if (end <= begin ||
                            end > ctrl->contribution_count ||
                            chunk_offset >=
                                static_cast<uint64_t>(end - begin)) {
                            continue;
                        }
                        any_chunk = true;
                        const uint32_t home_pe = ctrl->inc_pe;
                        const uint32_t first_i = group_entries[begin];
                        if (first_i >= ctrl->contribution_count ||
                            sources[first_i] != ctrl->this_worker_rank ||
                            slots[first_i] >= ctrl->max_ingress_slots) {
                            producer_ok = false;
                            continue;
                        }
                        const uint32_t first_slot = slots[first_i];
                        const uint32_t local_chunk_offset =
                            static_cast<uint32_t>(chunk_offset);
                        const uint32_t first = begin + local_chunk_offset;
                        const uint32_t this_chunk_tiles =
                            head_tile_stream && chunk_offset == 0u
                                ? 1u
                                : chunk_tiles;
                        const uint32_t count =
                            end - first < this_chunk_tiles ? end - first
                                                           : this_chunk_tiles;
                        bool valid = true;
                        for (uint32_t p = first; p < first + count; ++p) {
                            const uint32_t i = group_entries[p];
                            if (i >= ctrl->contribution_count ||
                                sources[i] != ctrl->this_worker_rank ||
                                slots[i] != first_slot + (p - begin) ||
                                slots[i] >= ctrl->max_ingress_slots) {
                                valid = false;
                                break;
                            }
                            if (!staged_rank_dedup &&
                                DynPrepareRankDedupContribution(sym, ctrl, slots,
                                                               i) !=
                                kDynCsrFailNone) {
                                valid = false;
                                break;
                            }
                        }
                        if (!valid) {
                            producer_ok = false;
                            continue;
                        }
                        if (ctrl->local_rank_prereduce != 0u &&
                            !staged_rank_dedup) {
                            // See the stream-global producer above: the
                            // payload RMA must observe all vector-produced
                            // staging rows in this owner/source chunk.
                            dcci_entire_cache();
                        }
                        const uint32_t slot =
                            first_slot + local_chunk_offset;
                        __gm__ uint8_t *tile =
                            sym + ctrl->ingress_off +
                            static_cast<uint64_t>(slot) * ctrl->tile_bytes;
                        const bool result_counter =
                            (ctrl->optimization_flags &
                             kDynCsrOptResultArrivalCounter) != 0u;
                        const uint32_t contribution = group_entries[first];
                        const uint32_t result =
                            result_counter ? contrib_result[contribution] : 0u;
                        if (result_counter &&
                            (count != 1u || contribution >=
                                                 ctrl->contribution_count ||
                             result >= ctrl->result_count)) {
                            producer_ok = false;
                            continue;
                        }
                        __gm__ uint8_t *ready_bytes =
                            result_counter
                                ? sym + ctrl->result_arrival_counter_off +
                                      static_cast<uint64_t>(result) * 64u
                                : sym + ctrl->ready_generation_off +
                                      static_cast<uint64_t>(slot) *
                                          ctrl->ready_stride_bytes;
                        __gm__ int32_t *ready =
                            reinterpret_cast<__gm__ int32_t *>(ready_bytes);
                        if (pst->issued == 0u) {
                            pst->first_issue_cycle = GetSystemCycle();
                        }
                        // One ordered payload+signal transaction replaces
                        // the former payload quiet followed by a second
                        // cacheline RMA+quiet.  This is the standard
                        // producer/consumer primitive used by mature SHMEM
                        // streaming collectives: the generation becomes
                        // visible only after this chunk's payload completes.
                        aclshmem_putmem_signal(
                            reinterpret_cast<__gm__ void *>(tile),
                            reinterpret_cast<__gm__ void *>(tile),
                            static_cast<size_t>(count) * ctrl->tile_bytes,
                            ready,
                            result_counter
                                ? 1
                                : static_cast<int32_t>(ctrl->generation),
                            result_counter ? ACLSHMEM_SIGNAL_ADD
                                           : ACLSHMEM_SIGNAL_SET,
                            static_cast<int32_t>(home_pe));
                        pst->issued += count;
                        pst->last_quiet_cycle = GetSystemCycle();
                        ++pst->ready_signals;
                        pst->last_ready_cycle = GetSystemCycle();
                    }
                    if (!any_chunk) {
                        break;
                    }
                }
                pst->kernel_cycles = GetSystemCycle() - producer_t0;
                pst->done = producer_ok ? ctrl->generation : 0xffffffffu;
                if (!producer_ok) {
                    pst->reserved[0] = kDynCsrFailSlot;
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                DynDcci(reinterpret_cast<__gm__ uint8_t *>(pst),
                        sizeof(*pst));
                if (ctrl->device_completion != 0u && lane == 0u) {
                    const uint64_t completion =
                        DynCollectiveCompletionBase(ctrl) +
                        ctrl->this_worker_rank;
                    __gm__ int32_t *done =
                        reinterpret_cast<__gm__ int32_t *>(
                            sym + ctrl->ready_generation_off +
                            completion * ctrl->ready_stride_bytes);
                    bool completion_ok = producer_ok &&
                        DynWaitProducerPeerLanes(
                            sym, ctrl, ctrl->generation) &&
                        DynWaitCollectiveValue(
                            ctrl, done,
                            static_cast<int32_t>(ctrl->generation));
                    if (completion_ok) {
                        completion_ok = DynAckCollectiveCompletion(
                            sym, ctrl, ctrl->generation);
                    }
                    pst->done = completion_ok ? ctrl->generation
                                              : 0xffffffffu;
                    AscendC::PipeBarrier<PIPE_ALL>();
                    DynDcci(reinterpret_cast<__gm__ uint8_t *>(pst),
                            sizeof(*pst));
                }
                if (lane == 0u) {
                    if (ctrl->device_completion == 0u) {
                        producer_ok =
                            producer_ok && DynWaitProducerPeerLanes(
                                               sym, ctrl, ctrl->generation);
                    }
                    DynPersistentServiceEnd(
                        sym, ctrl,
                        producer_ok ? kDynCsrFailNone : kDynCsrFailSlot);
                }
                return;
            }

            if (false && (ctrl->optimization_flags &
                 kDynCsrOptSourceGroupWorklist) != 0u) {
                __gm__ uint32_t *source_group_offsets =
                    reinterpret_cast<__gm__ uint32_t *>(
                        sym + ctrl->source_group_offsets_off);
                __gm__ uint32_t *source_group_entries =
                    reinterpret_cast<__gm__ uint32_t *>(
                        sym + ctrl->source_group_entries_off);
                const uint32_t list_begin =
                    source_group_offsets[ctrl->this_worker_rank];
                const uint32_t list_end =
                    source_group_offsets[ctrl->this_worker_rank + 1u];
                bool producer_ok = true;
                for (uint32_t pos = list_begin + lane; pos < list_end;
                     pos += ctrl->producer_lane_count) {
                    const uint32_t flat = source_group_entries[pos];
                    if (flat >= ctrl->owner_count) {
                        continue;
                    }
                    const uint32_t group =
                        flat * ctrl->worker_count + ctrl->this_worker_rank;
                    const uint32_t begin = group_offsets[group];
                    const uint32_t end = group_offsets[group + 1u];
                    if (end <= begin || end > ctrl->contribution_count) {
                        continue;
                    }
                    const uint32_t home_pe = ctrl->inc_pe;
                    uint32_t pending = 0u;
                    bool valid = true;
                    const bool coalesced =
                        (ctrl->optimization_flags &
                         kDynCsrOptCoalescedGroupPut) != 0u;
                    uint32_t first_slot = 0u;
                    for (uint32_t p = begin; p < end; ++p) {
                        const uint32_t i = group_entries[p];
                        if (i >= ctrl->contribution_count ||
                            sources[i] != ctrl->this_worker_rank ||
                            slots[i] >= ctrl->max_ingress_slots) {
                            valid = false;
                            break;
                        }
                        if (!staged_rank_dedup &&
                            DynPrepareRankDedupContribution(sym, ctrl, slots,
                                                           i) !=
                            kDynCsrFailNone) {
                            valid = false;
                            break;
                        }
                        if (p == begin) {
                            first_slot = slots[i];
                        } else if (coalesced &&
                                   slots[i] != first_slot + (p - begin)) {
                            valid = false;
                            break;
                        }
                        if (coalesced) {
                            continue;
                        }
                        __gm__ uint8_t *tile =
                            sym + ctrl->ingress_off +
                            static_cast<uint64_t>(slots[i]) *
                                ctrl->tile_bytes;
                        if (pst->issued == 0u) {
                            pst->first_issue_cycle = GetSystemCycle();
                        }
                        aclshmem_putmem_nbi(
                            reinterpret_cast<__gm__ void *>(tile),
                            reinterpret_cast<__gm__ void *>(tile),
                            ctrl->tile_bytes, static_cast<int32_t>(home_pe));
                        ++pst->issued;
                        if (++pending >= quiet_window) {
                            aclshmem_quiet();
                            pst->last_quiet_cycle = GetSystemCycle();
                            pending = 0u;
                        }
                    }
                    if (!valid) {
                        producer_ok = false;
                        __gm__ int32_t *failed_ready =
                            reinterpret_cast<__gm__ int32_t *>(
                                sym + ctrl->ready_generation_off +
                                static_cast<uint64_t>(group) *
                                    ctrl->ready_stride_bytes);
                        aclshmemx_signal_op(
                            failed_ready,
                            -static_cast<int32_t>(ctrl->generation),
                            ACLSHMEM_SIGNAL_SET,
                            static_cast<int32_t>(home_pe));
                        continue;
                    }
                    if (coalesced) {
                        __gm__ uint8_t *tile =
                            sym + ctrl->ingress_off +
                            static_cast<uint64_t>(first_slot) *
                                ctrl->tile_bytes;
                        DynPutContiguousGroup(
                            tile, end - begin, ctrl->tile_bytes,
                            ctrl->coalesced_chunk_bytes, quiet_window,
                            static_cast<int32_t>(home_pe), pst);
                    } else if (pending != 0u) {
                        aclshmem_quiet();
                        pst->last_quiet_cycle = GetSystemCycle();
                    }
                    __gm__ int32_t *ready =
                        reinterpret_cast<__gm__ int32_t *>(
                            sym + ctrl->ready_generation_off +
                            static_cast<uint64_t>(group) *
                                ctrl->ready_stride_bytes);
                    aclshmemx_signal_op(
                        ready, static_cast<int32_t>(ctrl->generation),
                        ACLSHMEM_SIGNAL_SET, static_cast<int32_t>(home_pe));
                    ++pst->ready_signals;
                    pst->last_ready_cycle = GetSystemCycle();
                }
                pst->kernel_cycles = GetSystemCycle() - producer_t0;
                // Match the default owner-major path's multi-generation
                // credit.  Without this wait, epoch e+1 can overwrite a
                // ready record with generation e+1 before a slow INC has
                // observed generation e, causing an exact-equality deadlock.
                if (ctrl->device_completion != 0u && lane == 0u) {
                    const uint64_t completion =
                        DynCollectiveCompletionBase(ctrl) +
                        ctrl->this_worker_rank;
                    __gm__ int32_t *done =
                        reinterpret_cast<__gm__ int32_t *>(
                            sym + ctrl->ready_generation_off +
                            completion * ctrl->ready_stride_bytes);
                    bool completion_ok = DynWaitProducerPeerLanes(
                        sym, ctrl, ctrl->generation) &&
                        DynWaitCollectiveValue(
                            ctrl, done,
                            static_cast<int32_t>(ctrl->generation));
                    if (completion_ok) {
                        completion_ok = DynAckCollectiveCompletion(
                            sym, ctrl, ctrl->generation);
                    }
                    pst->done = completion_ok ? ctrl->generation
                                              : 0xffffffffu;
                } else {
                    pst->done = ctrl->generation;
                }
                if (!producer_ok) {
                    pst->reserved[0] = kDynCsrFailSlot;
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                DynDcci(reinterpret_cast<__gm__ uint8_t *>(pst),
                        sizeof(*pst));
                if (lane == 0u) {
                    DynPersistentServiceEnd(
                        sym, ctrl,
                        pst->done != ctrl->generation
                            ? kDynCsrFailMissing
                            : pst->reserved[0]);
                }
                return;
            }

            // Each lane owns complete owner-source batches. Payload
            // visibility is completed before the single generation signal.
            bool producer_ok = true;
            for (uint32_t order = lane; order < ctrl->owner_count;
                 order += ctrl->producer_lane_count) {
                const uint32_t owner = order;
                const uint32_t group =
                    owner * ctrl->worker_count + ctrl->this_worker_rank;
                const uint32_t begin = group_offsets[group];
                const uint32_t end = group_offsets[group + 1u];
                if (end <= begin || end > ctrl->contribution_count) {
                    continue;
                }
                const uint32_t home_pe = ctrl->inc_pe;
                bool valid = true;
                uint32_t pending = 0;
                const bool coalesced =
                    (ctrl->optimization_flags &
                     kDynCsrOptCoalescedGroupPut) != 0u;
                uint32_t first_slot = 0u;
                for (uint32_t p = begin; p < end; ++p) {
                    const uint32_t i = group_entries[p];
                    if (i >= ctrl->contribution_count ||
                        sources[i] != ctrl->this_worker_rank ||
                        slots[i] >= ctrl->max_ingress_slots) {
                        valid = false;
                        break;
                    }
                    if (!staged_rank_dedup &&
                        DynPrepareRankDedupContribution(sym, ctrl, slots, i) !=
                        kDynCsrFailNone) {
                        valid = false;
                        break;
                    }
                    if (p == begin) {
                        first_slot = slots[i];
                    } else if (coalesced &&
                               slots[i] != first_slot + (p - begin)) {
                        valid = false;
                        break;
                    }
                    if (coalesced) {
                        continue;
                    }
                    __gm__ uint8_t *tile =
                        sym + ctrl->ingress_off +
                        static_cast<uint64_t>(slots[i]) * ctrl->tile_bytes;
                    if (pst->issued == 0u) {
                        pst->first_issue_cycle = GetSystemCycle();
                    }
                    aclshmem_putmem_nbi(
                        reinterpret_cast<__gm__ void *>(tile),
                        reinterpret_cast<__gm__ void *>(tile), ctrl->tile_bytes,
                        static_cast<int32_t>(home_pe));
                    ++pst->issued;
                    ++pending;
                    if (pending >= quiet_window) {
                        aclshmem_quiet();
                        pst->last_quiet_cycle = GetSystemCycle();
                        pending = 0;
                    }
                }
                if (!valid) {
                    producer_ok = false;
                    __gm__ int32_t *failed_ready =
                        reinterpret_cast<__gm__ int32_t *>(
                            sym + ctrl->ready_generation_off +
                            static_cast<uint64_t>(group) *
                                ctrl->ready_stride_bytes);
                    aclshmemx_signal_op(
                        failed_ready,
                        -static_cast<int32_t>(ctrl->generation),
                        ACLSHMEM_SIGNAL_SET,
                        static_cast<int32_t>(home_pe));
                    continue;
                }
                if (coalesced) {
                    __gm__ uint8_t *tile =
                        sym + ctrl->ingress_off +
                        static_cast<uint64_t>(first_slot) *
                            ctrl->tile_bytes;
                    DynPutContiguousGroup(
                        tile, end - begin, ctrl->tile_bytes,
                        ctrl->coalesced_chunk_bytes, quiet_window,
                        static_cast<int32_t>(home_pe), pst);
                }
                if (!coalesced && pending != 0u) {
                    aclshmem_quiet();
                    pst->last_quiet_cycle = GetSystemCycle();
                }
                __gm__ int32_t *ready =
                    reinterpret_cast<__gm__ int32_t *>(
                        sym + ctrl->ready_generation_off +
                        static_cast<uint64_t>(group) *
                            ctrl->ready_stride_bytes);
                aclshmemx_signal_op(
                    ready, static_cast<int32_t>(ctrl->generation),
                    ACLSHMEM_SIGNAL_SET, static_cast<int32_t>(home_pe));
                ++pst->ready_signals;
                pst->last_ready_cycle = GetSystemCycle();
            }
            pst->kernel_cycles = GetSystemCycle() - producer_t0;
            // Lane 0 turns local producer completion into collective
            // completion by waiting for one generation record from every
            // INC.  Other lanes may return independently; stream completion
            // still waits for the whole block launch.
            if (ctrl->device_completion != 0u && lane == 0u) {
                const uint64_t completion =
                    DynCollectiveCompletionBase(ctrl) +
                    ctrl->this_worker_rank;
                __gm__ int32_t *done = reinterpret_cast<__gm__ int32_t *>(
                    sym + ctrl->ready_generation_off +
                    completion * ctrl->ready_stride_bytes);
                bool completion_ok = DynWaitProducerPeerLanes(
                    sym, ctrl, ctrl->generation) &&
                    DynWaitCollectiveValue(
                        ctrl, done,
                        static_cast<int32_t>(ctrl->generation));
                if (completion_ok) {
                    completion_ok = DynAckCollectiveCompletion(
                        sym, ctrl, ctrl->generation);
                }
                pst->done =
                    completion_ok ? ctrl->generation : 0xffffffffu;
            } else {
                pst->done = ctrl->generation;
            }
            if (!producer_ok) {
                pst->reserved[0] = kDynCsrFailSlot;
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            DynDcci(reinterpret_cast<__gm__ uint8_t *>(pst), sizeof(*pst));
            if (lane == 0u) {
                DynPersistentServiceEnd(
                    sym, ctrl,
                    pst->done != ctrl->generation
                        ? kDynCsrFailMissing
                        : pst->reserved[0]);
            }
            return;
        }
        uint32_t pending = 0;
        for (uint32_t i = lane; i < ctrl->contribution_count;
             i += ctrl->producer_lane_count) {
            if (sources[i] != ctrl->this_worker_rank ||
                slots[i] >= ctrl->max_ingress_slots) {
                continue;
            }
            if (DynPrepareRankDedupContribution(sym, ctrl, slots, i) !=
                kDynCsrFailNone) {
                pst->reserved[0] = kDynCsrFailVector;
                continue;
            }
            __gm__ uint8_t *tile =
                sym + ctrl->ingress_off +
                static_cast<uint64_t>(slots[i]) * ctrl->tile_bytes;
            if (pst->issued == 0u) {
                pst->first_issue_cycle = GetSystemCycle();
            }
            if (ctrl->ready_mode == 1u) {
                __gm__ int32_t *ready = reinterpret_cast<__gm__ int32_t *>(
                    sym + ctrl->ready_generation_off +
                    static_cast<uint64_t>(slots[i]) *
                        ctrl->ready_stride_bytes);
                aclshmem_putmem_signal_nbi(
                    reinterpret_cast<__gm__ void *>(tile),
                    reinterpret_cast<__gm__ void *>(tile), ctrl->tile_bytes,
                    ready, static_cast<int32_t>(ctrl->generation),
                    ACLSHMEM_SIGNAL_SET, static_cast<int32_t>(ctrl->inc_pe));
            } else {
                aclshmem_putmem_nbi(reinterpret_cast<__gm__ void *>(tile),
                                    reinterpret_cast<__gm__ void *>(tile),
                                    ctrl->tile_bytes,
                                    static_cast<int32_t>(ctrl->inc_pe));
            }
            ++pst->issued;
            ++pending;
            if (pending >= quiet_window) {
                aclshmem_quiet();
                pst->last_quiet_cycle = GetSystemCycle();
                pending = 0;
            }
        }
        if (pending != 0u) {
            aclshmem_quiet();
            pst->last_quiet_cycle = GetSystemCycle();
        }
        pst->kernel_cycles = GetSystemCycle() - producer_t0;
        pst->done = ctrl->generation;
        AscendC::PipeBarrier<PIPE_ALL>();
        DynDcci(reinterpret_cast<__gm__ uint8_t *>(pst), sizeof(*pst));
        if (lane == 0u) {
            DynPersistentServiceEnd(sym, ctrl, kDynCsrFailNone);
        }
    }
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
inc_dc_sv2_dyn_csr_cycle_probe_kernel(__gm__ uint8_t *sym,
                                      uint64_t cycle_off)
{
    if ASCEND_IS_AIV {
        *reinterpret_cast<__gm__ uint64_t *>(sym + cycle_off) =
            GetSystemCycle();
        AscendC::PipeBarrier<PIPE_ALL>();
        DynDcci(sym + cycle_off, 64u);
    }
}

// A strict identity-K1 plan is a routed copy.  Sending it through an INC
// would add a redundant ingress write, local copy and egress write.  This
// kernel preserves the dynamic dst_rank/dst_row route and the same bounded
// NBI/quiet discipline while removing that empty reduction hop.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
inc_dc_sv2_dyn_csr_k1_direct_tx_kernel(__gm__ uint8_t *sym,
                                       uint64_t ctrl_off)
{
    if ASCEND_IS_AIV {
        const uint32_t lane = static_cast<uint32_t>(GetBlockIdx());
        __gm__ DynCsrCtrl *ctrl =
            reinterpret_cast<__gm__ DynCsrCtrl *>(sym + ctrl_off);
        DynDcci(reinterpret_cast<__gm__ uint8_t *>(ctrl), kDynCsrCtrlBytes);
        if (ctrl->magic != kDynCsrMagic || ctrl->producer_lane_count == 0u ||
            lane >= ctrl->producer_lane_count || ctrl->worker_count == 0u ||
            ctrl->this_worker_rank >= ctrl->worker_count ||
            (ctrl->optimization_flags &
             (kDynCsrOptRemoteResultTx | kDynCsrOptK1DirectResultTx)) !=
                (kDynCsrOptRemoteResultTx | kDynCsrOptK1DirectResultTx)) {
            return;
        }

        __gm__ DynCsrProducerStats *pst =
            reinterpret_cast<__gm__ DynCsrProducerStats *>(
                sym + ctrl->owner_stats_off +
                static_cast<uint64_t>(lane) * sizeof(DynCsrProducerStats));
        pst->lane = lane;
        pst->done = 0u;
        pst->issued = 0u;
        pst->ready_signals = 0u;
        pst->first_issue_cycle = 0u;
        pst->last_quiet_cycle = 0u;
        pst->last_ready_cycle = 0u;
        if (!DynWaitPersistentLocalTrigger(sym, ctrl)) {
            pst->done = 0xffffffffu;
            AscendC::PipeBarrier<PIPE_ALL>();
            DynDcci(reinterpret_cast<__gm__ uint8_t *>(pst), sizeof(*pst));
            return;
        }
        const uint64_t t0 = GetSystemCycle();
        if (lane == 0u) {
            DynPersistentServiceStart(sym, ctrl, t0);
        }

        __gm__ uint32_t *offsets = reinterpret_cast<__gm__ uint32_t *>(
            sym + ctrl->result_offsets_off);
        __gm__ uint32_t *slots = reinterpret_cast<__gm__ uint32_t *>(
            sym + ctrl->contrib_slot_off);
        __gm__ uint32_t *sources = reinterpret_cast<__gm__ uint32_t *>(
            sym + ctrl->contrib_source_rank_off);
        __gm__ uint32_t *dst_ranks = reinterpret_cast<__gm__ uint32_t *>(
            sym + ctrl->result_dst_rank_off);
        __gm__ uint32_t *dst_rows = reinterpret_cast<__gm__ uint32_t *>(
            sym + ctrl->result_dst_row_off);
        __gm__ uint32_t *worker_pes = reinterpret_cast<__gm__ uint32_t *>(
            sym + ctrl->worker_pe_off);
        const uint32_t window =
            ctrl->tx_quiet_window == 0u ? 1u : ctrl->tx_quiet_window;
        uint32_t pending = 0u;
        bool valid = true;

        for (uint32_t r = lane; r < ctrl->result_count;
             r += ctrl->producer_lane_count) {
            const uint32_t begin = offsets[r];
            const uint32_t end = offsets[r + 1u];
            if (end != begin + 1u || begin >= ctrl->contribution_count) {
                valid = false;
                break;
            }
            if (sources[begin] != ctrl->this_worker_rank) {
                continue;
            }
            if (slots[begin] >= ctrl->max_ingress_slots ||
                dst_ranks[r] >= ctrl->worker_count ||
                dst_rows[r] >= ctrl->result_count) {
                valid = false;
                break;
            }
            __gm__ uint8_t *src =
                sym + ctrl->ingress_off +
                static_cast<uint64_t>(slots[begin]) * ctrl->tile_bytes;
            __gm__ uint8_t *dst =
                sym + ctrl->output_off +
                static_cast<uint64_t>(dst_rows[r]) * ctrl->tile_bytes;
            if (pst->issued == 0u) {
                pst->first_issue_cycle = GetSystemCycle();
            }
            aclshmem_putmem_nbi(
                reinterpret_cast<__gm__ void *>(dst),
                reinterpret_cast<__gm__ void *>(src), ctrl->tile_bytes,
                static_cast<int32_t>(worker_pes[dst_ranks[r]]));
            ++pst->issued;
            if (++pending >= window) {
                aclshmem_quiet();
                pst->last_quiet_cycle = GetSystemCycle();
                pending = 0u;
            }
        }
        if (pending != 0u) {
            aclshmem_quiet();
            pst->last_quiet_cycle = GetSystemCycle();
        }
        pst->kernel_cycles = GetSystemCycle() - t0;
        pst->done = valid ? ctrl->generation : 0xffffffffu;
        AscendC::PipeBarrier<PIPE_ALL>();
        DynDcci(reinterpret_cast<__gm__ uint8_t *>(pst), sizeof(*pst));

        // The direct-K1 kernel has no INC reducer to close the persistent
        // timing/completion record.  Lane 0 must therefore collect its local
        // producer lanes and exchange source/destination credits even when
        // the general reducer path uses host-side completion.  This also
        // makes zero-contribution workers complete a valid empty service
        // instead of leaving service_end_cycle at zero.
        if (lane == 0u) {
            bool collective_ok = valid;
            for (uint32_t peer_lane = 0u;
                 peer_lane < ctrl->producer_lane_count; ++peer_lane) {
                __gm__ DynCsrProducerStats *peer =
                    reinterpret_cast<__gm__ DynCsrProducerStats *>(
                        sym + ctrl->owner_stats_off +
                        static_cast<uint64_t>(peer_lane) *
                            sizeof(DynCsrProducerStats));
                uint32_t spins = 0u;
                while (peer->done != ctrl->generation &&
                       peer->done != 0xffffffffu &&
                       spins < ctrl->ready_spin_cap) {
                    DynDcci(reinterpret_cast<__gm__ uint8_t *>(peer),
                            sizeof(*peer));
                    if ((spins & 1023u) == 0u &&
                        DynAbortRequested(ctrl, ctrl->generation)) {
                        collective_ok = false;
                        break;
                    }
                    ++spins;
                }
                if (peer->done != ctrl->generation) {
                    collective_ok = false;
                }
            }

            const uint64_t direct_base =
                static_cast<uint64_t>(ctrl->group_count) +
                ctrl->worker_count;
            // Publish source completion only after every local direct-TX lane
            // has quieted.  Publish even on failure so peers can unwind.
            for (uint32_t dst_worker = 0u;
                 dst_worker < ctrl->worker_count; ++dst_worker) {
                const uint64_t record =
                    direct_base +
                    static_cast<uint64_t>(ctrl->this_worker_rank) *
                        ctrl->worker_count +
                    dst_worker;
                __gm__ int32_t *done =
                    reinterpret_cast<__gm__ int32_t *>(
                        sym + ctrl->ready_generation_off +
                        record * ctrl->ready_stride_bytes);
                aclshmemx_signal_op(
                    done, static_cast<int32_t>(ctrl->generation),
                    ACLSHMEM_SIGNAL_SET,
                    static_cast<int32_t>(worker_pes[dst_worker]));
            }
            aclshmem_quiet();

            // Destination completion is the credit for output consumption and
            // for reusing the direct-TX source/output buffers next epoch.
            for (uint32_t src_worker = 0u;
                 src_worker < ctrl->worker_count; ++src_worker) {
                const uint64_t record =
                    direct_base +
                    static_cast<uint64_t>(src_worker) *
                        ctrl->worker_count +
                    ctrl->this_worker_rank;
                __gm__ int32_t *done =
                    reinterpret_cast<__gm__ int32_t *>(
                        sym + ctrl->ready_generation_off +
                        record * ctrl->ready_stride_bytes);
                uint32_t spins = 0u;
                while (*done != static_cast<int32_t>(ctrl->generation) &&
                       spins < ctrl->ready_spin_cap) {
                    DynDcci(reinterpret_cast<__gm__ uint8_t *>(done),
                            sizeof(int32_t));
                    if ((spins & 1023u) == 0u &&
                        DynAbortRequested(ctrl, ctrl->generation)) {
                        collective_ok = false;
                        break;
                    }
                    ++spins;
                }
                if (*done != static_cast<int32_t>(ctrl->generation)) {
                    collective_ok = false;
                }
            }
            pst->done =
                collective_ok ? ctrl->generation : 0xffffffffu;
            AscendC::PipeBarrier<PIPE_ALL>();
            DynDcci(reinterpret_cast<__gm__ uint8_t *>(pst), sizeof(*pst));
            DynPersistentServiceEnd(
                sym, ctrl,
                collective_ok ? kDynCsrFailNone : kDynCsrFailMissing);
        }
    }
}

__aicore__ inline void DynCsrSplitTxLane(
    __gm__ uint8_t *sym, __gm__ DynCsrCtrl *ctrl, uint32_t tx_lane,
    uint32_t total_tx_shards = 0u, bool publish_done = true)
{
    if (total_tx_shards == 0u) {
        total_tx_shards =
            (ctrl->optimization_flags & kDynCsrOptRankPackedResultTx) != 0u
                ? ctrl->tx_lane_count
                : ctrl->tx_lane_count + ctrl->owner_count;
    }
    if (ctrl->tx_lane_count == 0u || tx_lane >= total_tx_shards ||
        ctrl->result_tx_ready_off == 0u || ctrl->tx_done_off == 0u) {
        return;
    }
    if (!DynWaitPersistentLocalTrigger(sym, ctrl)) {
        return;
    }
    __gm__ uint32_t *dst_rank = reinterpret_cast<__gm__ uint32_t *>(
        sym + ctrl->result_dst_rank_off);
    __gm__ uint32_t *dst_row = reinterpret_cast<__gm__ uint32_t *>(
        sym + ctrl->result_dst_row_off);
    __gm__ uint32_t *worker_pes = reinterpret_cast<__gm__ uint32_t *>(
        sym + ctrl->worker_pe_off);
    uint32_t fail = kDynCsrFailNone;
    if ((ctrl->optimization_flags & kDynCsrOptRankPackedResultTx) != 0u) {
        __gm__ uint32_t *rank_offsets =
            reinterpret_cast<__gm__ uint32_t *>(
                sym + ctrl->result_tx_rank_offsets_off);
        __gm__ uint32_t *packed_ids =
            reinterpret_cast<__gm__ uint32_t *>(
                sym + ctrl->packed_result_ids_off);
        const uint32_t stripe_begin = static_cast<uint32_t>(
            static_cast<uint64_t>(ctrl->result_count) * tx_lane /
            ctrl->tx_lane_count);
        const uint32_t stripe_end = static_cast<uint32_t>(
            static_cast<uint64_t>(ctrl->result_count) * (tx_lane + 1u) /
            ctrl->tx_lane_count);
        const uint32_t max_chunk_rows =
            ctrl->tile_bytes >= 1024u * 1024u
                ? 1u : (1024u * 1024u / ctrl->tile_bytes);
        __ubuf__ uint8_t *ub = reinterpret_cast<__ubuf__ uint8_t *>(0);
        const uint32_t ub_tile =
            ctrl->tile_bytes < kIncDcPrivateMtePacketBytes
                ? ctrl->tile_bytes : kIncDcPrivateMtePacketBytes;
        uint32_t storage = stripe_begin;
        uint32_t worker = 0u;
        while (worker + 1u < ctrl->worker_count &&
               rank_offsets[worker + 1u] <= storage) {
            ++worker;
        }
        while (storage < stripe_end && fail == kDynCsrFailNone) {
            while (worker + 1u < ctrl->worker_count &&
                   rank_offsets[worker + 1u] <= storage) {
                ++worker;
            }
            if (worker >= ctrl->worker_count ||
                storage < rank_offsets[worker] ||
                storage >= rank_offsets[worker + 1u]) {
                fail = kDynCsrFailHome;
                break;
            }
            uint32_t chunk_end = storage + max_chunk_rows;
            if (chunk_end > stripe_end) chunk_end = stripe_end;
            if (chunk_end > rank_offsets[worker + 1u]) {
                chunk_end = rank_offsets[worker + 1u];
            }
            for (uint32_t s = storage; s < chunk_end; ++s) {
                const uint32_t r = packed_ids[s];
                if (r >= ctrl->result_count) {
                    fail = kDynCsrFailCsr;
                    break;
                }
                __gm__ uint8_t *ready =
                    sym + ctrl->result_tx_ready_off +
                    static_cast<uint64_t>(r) * 64u;
                uint32_t spins = 0u;
                while (*reinterpret_cast<__gm__ volatile uint32_t *>(ready) !=
                           ctrl->generation &&
                       spins < ctrl->ready_spin_cap) {
                    dcci_cacheline(ready);
                    if ((spins & 1023u) == 0u &&
                        DynAbortRequested(ctrl, ctrl->generation)) {
                        fail = kDynCsrFailCancelled;
                        break;
                    }
                    ++spins;
                }
                if (*reinterpret_cast<__gm__ volatile uint32_t *>(ready) !=
                    ctrl->generation) {
                    if (fail == kDynCsrFailNone) {
                        fail = kDynCsrFailMissing;
                    }
                    break;
                }
            }
            if (fail != kDynCsrFailNone) break;
            const uint32_t packet =
                (chunk_end - storage) * ctrl->tile_bytes;
            __gm__ uint8_t *src =
                sym + ctrl->output_off +
                static_cast<uint64_t>(storage) * ctrl->tile_bytes;
            __gm__ uint8_t *dst =
                sym + ctrl->output_off +
                static_cast<uint64_t>(storage - rank_offsets[worker]) *
                    ctrl->tile_bytes;
            aclshmemx_mte_put_nbi(
                dst, src, ub, ub_tile, packet,
                static_cast<int32_t>(worker_pes[worker]), 0u);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
            aclshmemx_mte_quiet();
            storage = chunk_end;
        }
        __gm__ uint8_t *done =
            sym + ctrl->tx_done_off + static_cast<uint64_t>(tx_lane) * 64u;
        reinterpret_cast<__gm__ uint32_t *>(done)[1] = fail;
        AscendC::PipeBarrier<PIPE_ALL>();
        *reinterpret_cast<__gm__ uint32_t *>(done) = ctrl->generation;
        DynDcci(done, 64u);
        return;
    }
    if (false &&
        (ctrl->optimization_flags & kDynCsrOptRankPackedResultTx) != 0u) {
        // Packed Scheme-B egress: wait until every reducer has published its
        // local completion, then give each TX lane one contiguous stripe per
        // destination.  This removes thousands of row-sized RMAs while the
        // runtime rank offsets keep the implementation independent of W.
        __gm__ DynCsrOwnerStats *owner_stats =
            reinterpret_cast<__gm__ DynCsrOwnerStats *>(
                sym + ctrl->owner_stats_off);
        for (uint32_t owner = 0u; owner < ctrl->owner_count; ++owner) {
            uint32_t spins = 0u;
            while (owner_stats[owner].done != ctrl->generation &&
                   spins < ctrl->ready_spin_cap) {
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
                    &owner_stats[owner]));
                ++spins;
            }
            if (owner_stats[owner].done != ctrl->generation ||
                owner_stats[owner].fail_code != kDynCsrFailNone) {
                fail = owner_stats[owner].fail_code != kDynCsrFailNone
                    ? owner_stats[owner].fail_code : kDynCsrFailMissing;
                break;
            }
        }
        __gm__ uint32_t *rank_offsets =
            reinterpret_cast<__gm__ uint32_t *>(
                sym + ctrl->result_tx_rank_offsets_off);
        constexpr uint64_t kMaxMtePacket = 256ull * 1024ull * 1024ull;
        const uint32_t ub_tile =
            ctrl->tile_bytes < kIncDcPrivateMtePacketBytes
                ? ctrl->tile_bytes : kIncDcPrivateMtePacketBytes;
        __ubuf__ uint8_t *ub[2] = {
            reinterpret_cast<__ubuf__ uint8_t *>(0),
            reinterpret_cast<__ubuf__ uint8_t *>(
                static_cast<uint64_t>(ub_tile))};
        bool busy[2] = {false, false};
        uint32_t ping = 0u;
        for (uint32_t worker = 0u;
             worker < ctrl->worker_count && fail == kDynCsrFailNone;
             ++worker) {
            const uint32_t count = rank_offsets[worker + 1u] -
                                   rank_offsets[worker];
            const uint32_t row_begin = static_cast<uint32_t>(
                static_cast<uint64_t>(count) * tx_lane /
                ctrl->tx_lane_count);
            const uint32_t row_end = static_cast<uint32_t>(
                static_cast<uint64_t>(count) * (tx_lane + 1u) /
                ctrl->tx_lane_count);
            uint64_t bytes = static_cast<uint64_t>(row_end - row_begin) *
                             ctrl->tile_bytes;
            uint64_t within = 0u;
            while (within < bytes) {
                const uint32_t packet = static_cast<uint32_t>(
                    bytes - within < kMaxMtePacket
                        ? bytes - within : kMaxMtePacket);
                const uint32_t bi = ping;
                if (busy[bi]) {
                    if (bi == 0u) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
                    } else {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
                    }
                    busy[bi] = false;
                }
                __gm__ uint8_t *src =
                    sym + ctrl->output_off +
                    (static_cast<uint64_t>(rank_offsets[worker] + row_begin) *
                         ctrl->tile_bytes + within);
                __gm__ uint8_t *dst =
                    sym + ctrl->output_off +
                    static_cast<uint64_t>(row_begin) * ctrl->tile_bytes +
                    within;
                aclshmemx_mte_put_nbi(
                    dst, src, ub[bi], ub_tile, packet,
                    static_cast<int32_t>(worker_pes[worker]), bi);
                if (bi == 0u) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
                } else {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
                }
                busy[bi] = true;
                ping = 1u - ping;
                within += packet;
            }
        }
        if (busy[0]) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
        }
        if (busy[1]) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
        }
        aclshmemx_mte_quiet();
        __gm__ uint8_t *done =
            sym + ctrl->tx_done_off + static_cast<uint64_t>(tx_lane) * 64u;
        reinterpret_cast<__gm__ uint32_t *>(done)[1] = fail;
        AscendC::PipeBarrier<PIPE_ALL>();
        *reinterpret_cast<__gm__ uint32_t *>(done) = ctrl->generation;
        DynDcci(done, 64u);
        return;
    }
    constexpr uint32_t kSplitPublicBatch = 2u;
    uint32_t split_batch[kSplitPublicBatch] = {0u, 0u};
    uint32_t split_count = 0u;
    for (uint32_t r = tx_lane; r < ctrl->result_count;
         r += total_tx_shards) {
        __gm__ uint8_t *ready =
            sym + ctrl->result_tx_ready_off + static_cast<uint64_t>(r) * 64u;
        uint32_t spins = 0u;
        while (*reinterpret_cast<__gm__ volatile uint32_t *>(ready) !=
                   ctrl->generation &&
               spins < ctrl->ready_spin_cap) {
            dcci_cacheline(ready);
            if ((spins & 1023u) == 0u &&
                DynAbortRequested(ctrl, ctrl->generation)) {
                fail = kDynCsrFailCancelled;
                break;
            }
            ++spins;
        }
        if (*reinterpret_cast<__gm__ volatile uint32_t *>(ready) !=
            ctrl->generation) {
            if (fail == kDynCsrFailNone) fail = kDynCsrFailMissing;
            break;
        }
        if (dst_rank[r] >= ctrl->worker_count ||
            dst_row[r] >= ctrl->result_count) {
            fail = kDynCsrFailHome;
            break;
        }
        split_batch[split_count++] = r;
        if (split_count == kSplitPublicBatch) {
            // Both reducer-owned rows are complete before one shared
            // visibility publication.  Retain the proven public put/quiet
            // credit per row; only the redundant whole-cache operations are
            // coalesced, so arbitrary destination order remains safe.
            dcci_entire_cache();
            for (uint32_t bi = 0u; bi < split_count; ++bi) {
                const uint32_t br = split_batch[bi];
                __gm__ uint8_t *src =
                    sym + ctrl->output_off +
                    static_cast<uint64_t>(br) * ctrl->tile_bytes;
                __gm__ uint8_t *dst =
                    sym + ctrl->output_off +
                    static_cast<uint64_t>(dst_row[br]) * ctrl->tile_bytes;
                aclshmem_putmem_nbi(
                    dst, src, ctrl->tile_bytes,
                    static_cast<int32_t>(worker_pes[dst_rank[br]]));
                aclshmem_quiet();
            }
            split_count = 0u;
        }
    }
    if (fail == kDynCsrFailNone && split_count != 0u) {
        dcci_entire_cache();
        const uint32_t br = split_batch[0];
        __gm__ uint8_t *src =
            sym + ctrl->output_off +
            static_cast<uint64_t>(br) * ctrl->tile_bytes;
        __gm__ uint8_t *dst =
            sym + ctrl->output_off +
            static_cast<uint64_t>(dst_row[br]) * ctrl->tile_bytes;
        aclshmem_putmem_nbi(
            dst, src, ctrl->tile_bytes,
            static_cast<int32_t>(worker_pes[dst_rank[br]]));
        aclshmem_quiet();
    }
    if (publish_done) {
        __gm__ uint8_t *done =
            sym + ctrl->tx_done_off + static_cast<uint64_t>(tx_lane) * 64u;
        reinterpret_cast<__gm__ uint32_t *>(done)[1] = fail;
        AscendC::PipeBarrier<PIPE_ALL>();
        *reinterpret_cast<__gm__ uint32_t *>(done) = ctrl->generation;
        DynDcci(done, 64u);
    }
}

// Push a small batch of completed result rows with two private UB staging
// buffers.  The generic putmem_nbi backend owns one shared completion slot per
// AIV, so a window greater than one is unsafe for the non-monotonic source and
// destination schedule produced by arbitrary token plans.  True-MTE exposes
// the UB and event id explicitly: alternating event 0/1 gives a real two-deep
// pipeline without consuming another AIV or imposing a quiet per row.
__aicore__ inline uint32_t DynCsrMtePingPongResultBatch(
    __gm__ uint8_t *output, __gm__ DynCsrCtrl *ctrl,
    __gm__ uint32_t *result_dst_rank, __gm__ uint32_t *result_dst_row,
    __gm__ uint32_t *worker_pes, uint32_t *result_ids, uint32_t count)
{
    if (count == 0u || count > 4u || worker_pes == nullptr) {
        return count == 0u ? kDynCsrFailNone : kDynCsrFailHome;
    }
    const uint32_t ub_tile =
        ctrl->tile_bytes < kIncDcPrivateMtePacketBytes
            ? ctrl->tile_bytes : kIncDcPrivateMtePacketBytes;
    __ubuf__ uint8_t *ub[2] = {
        reinterpret_cast<__ubuf__ uint8_t *>(0),
        reinterpret_cast<__ubuf__ uint8_t *>(static_cast<uint64_t>(ub_tile))};
    bool busy[2] = {false, false};
    uint32_t ping = 0u;
    for (uint32_t q = 0u; q < count; ++q) {
        const uint32_t r = result_ids[q];
        const uint32_t dst_rank = result_dst_rank[r];
        const uint32_t dst_row = result_dst_row[r];
        if (dst_rank >= ctrl->worker_count || dst_row >= ctrl->result_count) {
            return kDynCsrFailHome;
        }
        const uint32_t bi = ping;
        if (busy[bi]) {
            if (bi == 0u) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
            }
            busy[bi] = false;
        }
        __gm__ uint8_t *src =
            output + static_cast<uint64_t>(r) * ctrl->tile_bytes;
        __gm__ uint8_t *dst =
            output + static_cast<uint64_t>(dst_row) * ctrl->tile_bytes;
        aclshmemx_mte_put_nbi(dst, src, ub[bi], ub_tile, ctrl->tile_bytes,
                              static_cast<int32_t>(worker_pes[dst_rank]), bi);
        if (bi == 0u) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
        }
        busy[bi] = true;
        ping = 1u - ping;
    }
    if (busy[0]) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
    }
    if (busy[1]) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
    }
    aclshmemx_mte_quiet();
    return kDynCsrFailNone;
}

__aicore__ inline uint32_t DynCsrMtePackedOwnerStripe(
    __gm__ uint8_t *output, __gm__ DynCsrCtrl *ctrl,
    __gm__ uint32_t *rank_offsets, __gm__ uint32_t *worker_pes,
    uint32_t owner, uint32_t owner_count)
{
    constexpr uint64_t kMaxMtePacket = 256ull * 1024ull * 1024ull;
    const uint32_t ub_tile =
        ctrl->tile_bytes < kIncDcPrivateMtePacketBytes
            ? ctrl->tile_bytes : kIncDcPrivateMtePacketBytes;
    __ubuf__ uint8_t *ub[2] = {
        reinterpret_cast<__ubuf__ uint8_t *>(0),
        reinterpret_cast<__ubuf__ uint8_t *>(static_cast<uint64_t>(ub_tile))};
    bool busy[2] = {false, false};
    uint32_t ping = 0u;
    for (uint32_t worker = 0u; worker < ctrl->worker_count; ++worker) {
        const uint32_t rank_begin = rank_offsets[worker];
        const uint32_t rank_end = rank_offsets[worker + 1u];
        // Owner assignment is floor(storage_row * O / R).  Its exact inverse
        // interval uses ceil boundaries; floor boundaries leave one-row gaps
        // whenever R is not divisible by O.
        const uint32_t global_begin = static_cast<uint32_t>(
            (static_cast<uint64_t>(ctrl->result_count) * owner +
             owner_count - 1u) / owner_count);
        const uint32_t global_end = static_cast<uint32_t>(
            (static_cast<uint64_t>(ctrl->result_count) * (owner + 1u) +
             owner_count - 1u) / owner_count);
        const uint32_t begin = global_begin > rank_begin
            ? global_begin : rank_begin;
        const uint32_t end = global_end < rank_end ? global_end : rank_end;
        if (begin >= end) continue;
        const uint32_t remote_row = begin - rank_begin;
        const uint64_t bytes =
            static_cast<uint64_t>(end - begin) * ctrl->tile_bytes;
        uint64_t within = 0u;
        while (within < bytes) {
            const uint32_t packet = static_cast<uint32_t>(
                bytes - within < kMaxMtePacket
                    ? bytes - within : kMaxMtePacket);
            const uint32_t bi = ping;
            if (busy[bi]) {
                if (bi == 0u) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
                }
                busy[bi] = false;
            }
            __gm__ uint8_t *src =
                output + static_cast<uint64_t>(begin) * ctrl->tile_bytes +
                within;
            __gm__ uint8_t *dst =
                output + static_cast<uint64_t>(remote_row) *
                             ctrl->tile_bytes + within;
            aclshmemx_mte_put_nbi(
                dst, src, ub[bi], ub_tile, packet,
                static_cast<int32_t>(worker_pes[worker]), bi);
            if (bi == 0u) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
            }
            busy[bi] = true;
            ping = 1u - ping;
            within += packet;
        }
    }
    if (busy[0]) AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
    if (busy[1]) AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
    aclshmemx_mte_quiet();
    return kDynCsrFailNone;
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
inc_dc_sv2_dyn_csr_combine_kernel(__gm__ uint8_t *sym, uint64_t ctrl_off)
{
    if ASCEND_IS_AIV {
        const int bid = GetBlockIdx();
        __gm__ DynCsrCtrl *ctrl = (__gm__ DynCsrCtrl *)(sym + ctrl_off);
        DynDcci((__gm__ uint8_t *)ctrl, kDynCsrCtrlBytes);
        if (ctrl->magic != kDynCsrMagic) {
            return;
        }
        const uint32_t oc =
            (ctrl->owner_count > 0u && ctrl->owner_count <= kDynCsrMaxOwners)
                ? ctrl->owner_count
                : 1u;
        if (bid < 0 || static_cast<uint32_t>(bid) >=
                           oc + ctrl->tx_lane_count) {
            return;
        }
        if (static_cast<uint32_t>(bid) >= oc) {
            DynCsrSplitTxLane(sym, ctrl, static_cast<uint32_t>(bid) - oc);
            return;
        }
        const uint32_t owner = static_cast<uint32_t>(bid);
        const uint32_t hidden = ctrl->hidden;
        const uint32_t tile_bytes = ctrl->tile_bytes;
        const uint32_t R = ctrl->result_count;
        const uint32_t C = ctrl->contribution_count;
        const uint64_t generation = ctrl->generation;

        __gm__ DynCsrOwnerStats *ost =
            (__gm__ DynCsrOwnerStats *)(sym + ctrl->owner_stats_off +
                                       static_cast<uint64_t>(owner) *
                                           sizeof(DynCsrOwnerStats));
        ost->owner = owner;
        ost->done = 0;
        ost->fail_code = kDynCsrFailNone;
        ost->reduced = 0;
        ost->ready_wait_cycles = 0u;
        ost->first_ready_cycle = 0u;
        ost->first_reduce_cycle = 0u;
        ost->last_reduce_cycle = 0u;
        AscendC::PipeBarrier<PIPE_ALL>();

        if (hidden == 0u ||
            static_cast<uint64_t>(tile_bytes) <
                static_cast<uint64_t>(hidden) * 2ull) {
            ost->fail_code = kDynCsrFailHidden;
            ost->done = static_cast<uint32_t>(generation);
            DynDcci((__gm__ uint8_t *)ost, sizeof(DynCsrOwnerStats));
            return;
        }

        __gm__ uint32_t *result_offsets =
            (__gm__ uint32_t *)(sym + ctrl->result_offsets_off);
        __gm__ uint32_t *home_owner =
            (__gm__ uint32_t *)(sym + ctrl->result_home_owner_off);
        __gm__ uint32_t *result_dst_rank =
            (__gm__ uint32_t *)(sym + ctrl->result_dst_rank_off);
        __gm__ uint32_t *result_dst_row =
            (__gm__ uint32_t *)(sym + ctrl->result_dst_row_off);
        __gm__ uint32_t *result_tx_rank_offsets =
            (__gm__ uint32_t *)(sym + ctrl->result_tx_rank_offsets_off);
        __gm__ uint32_t *worker_pes =
            (__gm__ uint32_t *)(sym + ctrl->worker_pe_off);
        __gm__ uint32_t *slots =
            (__gm__ uint32_t *)(sym + ctrl->contrib_slot_off);
        __gm__ uint32_t *weights =
            (__gm__ uint32_t *)(sym + ctrl->contrib_weight_off);
        __gm__ uint32_t *ord =
            (__gm__ uint32_t *)(sym + ctrl->contrib_ordinal_off);
        __gm__ uint64_t *gens =
            (__gm__ uint64_t *)(sym + ctrl->contrib_gen_off);
        __gm__ uint64_t *arrival =
            (__gm__ uint64_t *)(sym + ctrl->arrival_off);
        __gm__ uint8_t *ingress = sym + ctrl->ingress_off;
        __gm__ uint8_t *output = sym + ctrl->output_off;
        __gm__ uint32_t *sources =
            (__gm__ uint32_t *)(sym + ctrl->contrib_source_rank_off);
        __gm__ uint32_t *group_offsets =
            (__gm__ uint32_t *)(sym + ctrl->group_offsets_off);
        __gm__ uint32_t *group_entries =
            (__gm__ uint32_t *)(sym + ctrl->group_entries_off);
        __gm__ uint32_t *source_contribution_offsets =
            (__gm__ uint32_t *)(
                sym + ctrl->source_contribution_offsets_off);
        __gm__ uint32_t *source_contribution_entries =
            (__gm__ uint32_t *)(
                sym + ctrl->source_contribution_entries_off);

        if (!DynWaitPersistentLocalTrigger(sym, ctrl)) {
            ost->fail_code = kDynCsrFailMissing;
            ost->done = static_cast<uint32_t>(generation);
            AscendC::PipeBarrier<PIPE_ALL>();
            DynDcci(reinterpret_cast<__gm__ uint8_t *>(ost), sizeof(*ost));
            return;
        }

        const uint64_t t0 = GetSystemCycle();
        if (owner == 0u) {
            DynPersistentServiceStart(sym, ctrl, t0);
        }
        uint32_t reduced = 0;
        uint32_t owned = 0;
        uint32_t fail = kDynCsrFailNone;
        uint32_t dup_rej = 0;
        uint32_t stale_rej = 0;
        uint32_t tx_pending = 0u;
        uint32_t packed_tx_worker = 0xffffffffu;
        uint32_t packed_tx_begin = 0u;
        uint32_t packed_tx_rows = 0u;
        bool packed_tx_inflight = false;
        constexpr uint32_t kResultTxBatch = 4u;
        uint32_t result_tx_batch[kResultTxBatch];
        uint32_t result_tx_batch_count = 0u;
        bool direct_ub_result_tx_issued = false;
        bool k1_relay_busy[2] = {false, false};
        uint32_t k1_relay_ping = 0u;
        bool reduce_timing_started = false;
        bool k1_timing_pending = false;
        const bool batch_result_tx =
            (ctrl->optimization_flags & kDynCsrOptBatchResultTx) != 0u &&
            (ctrl->optimization_flags & kDynCsrOptRemoteResultTx) != 0u &&
            (ctrl->optimization_flags & kDynCsrOptRankPackedResultTx) == 0u &&
            ctrl->tx_lane_count == 0u;
        if (DynAbortRequested(ctrl, static_cast<uint32_t>(generation))) {
            fail = kDynCsrFailCancelled;
        }

        // PERF3 waits only for source ranks that actually contribute to this
        // home INC/owner.  One wait per owner-source batch, never per result
        // or per contribution.
        if ((ctrl->ready_mode == 2u || ctrl->ready_mode == 3u) &&
            fail == kDynCsrFailNone) {
            const uint64_t ready_wait_t0 = GetSystemCycle();
            if (ctrl->worker_count == 0u ||
                ctrl->source_bitmap_words == 0u ||
                ctrl->group_count != oc * ctrl->worker_count ||
                ctrl->ready_stride_bytes < sizeof(int32_t)) {
                fail = kDynCsrFailCsr;
            } else if (ctrl->ready_mode == 3u) {
                // INC-scoped publication has one remote generation per
                // (INC, source), not per owner.  Polling the same remote
                // cachelines from every owner creates a DCCI/coherence storm
                // and makes ready latency highly variable.  Owner 0 is the
                // sole remote poller and publishes one local generation gate;
                // all other owners wait only on local GM.
                if (ctrl->waited_source_bitmap_off == 0u ||
                    ctrl->source_contribution_offsets_off == 0u) {
                    fail = kDynCsrFailCsr;
                } else {
                    __gm__ int32_t *inc_ready_gate =
                        reinterpret_cast<__gm__ int32_t *>(
                            sym + ctrl->waited_source_bitmap_off);
                    const int32_t ready_generation =
                        static_cast<int32_t>(generation);
                    const int32_t failed_generation = -ready_generation;
                    if (owner == 0u) {
                        __gm__ uint32_t *source_contribution_offsets =
                            reinterpret_cast<__gm__ uint32_t *>(
                                sym + ctrl->source_contribution_offsets_off);
                        for (uint32_t source = 0u;
                             source < ctrl->worker_count &&
                             fail == kDynCsrFailNone;
                             ++source) {
                            const uint32_t group = source;
                            const uint32_t begin =
                                source_contribution_offsets[group];
                            const uint32_t end =
                                source_contribution_offsets[group + 1u];
                            if (end <= begin) {
                                continue;
                            }
                            if (end > ctrl->contribution_count) {
                                fail = kDynCsrFailCsr;
                                break;
                            }
                            __gm__ int32_t *ready =
                                reinterpret_cast<__gm__ int32_t *>(
                                    sym + ctrl->ready_generation_off +
                                    static_cast<uint64_t>(group) *
                                        ctrl->ready_stride_bytes);
                            uint32_t spins = 0u;
                            while (*ready != ready_generation &&
                                   *ready != failed_generation &&
                                   spins < ctrl->ready_spin_cap) {
                                DynReadyPollDcci(ready, spins);
                                if ((spins & 1023u) == 0u &&
                                    DynAbortRequested(
                                        ctrl,
                                        static_cast<uint32_t>(generation))) {
                                    fail = kDynCsrFailCancelled;
                                    break;
                                }
                                ++spins;
                            }
                            if (fail == kDynCsrFailCancelled) {
                                break;
                            }
                            if (*ready == failed_generation) {
                                fail = kDynCsrFailCsr;
                                break;
                            }
                            if (*ready != ready_generation) {
                                fail = kDynCsrFailMissing;
                                break;
                            }
                            if (ost->first_ready_cycle == 0u) {
                                ost->first_ready_cycle = GetSystemCycle();
                            }
                        }
                        *inc_ready_gate =
                            fail == kDynCsrFailNone ? ready_generation
                                                   : failed_generation;
                        AscendC::PipeBarrier<PIPE_ALL>();
                        DynDcci(
                            reinterpret_cast<__gm__ uint8_t *>(inc_ready_gate),
                            sizeof(int32_t));
                    } else {
                        uint32_t spins = 0u;
                        while (*inc_ready_gate != ready_generation &&
                               *inc_ready_gate != failed_generation &&
                               spins < ctrl->ready_spin_cap) {
                            DynReadyPollDcci(inc_ready_gate, spins);
                            if ((spins & 1023u) == 0u &&
                                DynAbortRequested(
                                    ctrl,
                                    static_cast<uint32_t>(generation))) {
                                fail = kDynCsrFailCancelled;
                                break;
                            }
                            ++spins;
                        }
                        if (fail != kDynCsrFailCancelled &&
                            *inc_ready_gate == failed_generation) {
                            fail = DynAbortRequested(
                                       ctrl,
                                       static_cast<uint32_t>(generation))
                                       ? kDynCsrFailCancelled
                                       : kDynCsrFailMissing;
                        } else if (fail != kDynCsrFailCancelled &&
                                   *inc_ready_gate != ready_generation) {
                            fail = kDynCsrFailMissing;
                        } else if (fail == kDynCsrFailNone) {
                            ost->first_ready_cycle = GetSystemCycle();
                        }
                    }
                }
            } else {
                const uint32_t flat = owner;
                if (flat >= ctrl->owner_count) {
                    fail = kDynCsrFailHome;
                } else {
                    __gm__ uint32_t *source_bitmap =
                        (__gm__ uint32_t *)(
                            sym + ctrl->owner_source_bitmap_off +
                            static_cast<uint64_t>(flat) *
                                ctrl->source_bitmap_words * sizeof(uint32_t));
                    for (uint32_t source = 0;
                         source < ctrl->worker_count &&
                         fail == kDynCsrFailNone;
                         ++source) {
                        const uint32_t word = source >> 5u;
                        const uint32_t bit = source & 31u;
                        if ((source_bitmap[word] & (1u << bit)) == 0u) {
                            continue;
                        }
                        const uint32_t group =
                            flat * ctrl->worker_count + source;
                        __gm__ int32_t *ready =
                            reinterpret_cast<__gm__ int32_t *>(
                                sym + ctrl->ready_generation_off +
                                static_cast<uint64_t>(group) *
                                    ctrl->ready_stride_bytes);
                        const int32_t ready_generation =
                            static_cast<int32_t>(generation);
                        const int32_t failed_generation =
                            -ready_generation;
                        uint32_t spins = 0;
                        while (*ready != ready_generation &&
                               *ready != failed_generation &&
                               spins < ctrl->ready_spin_cap) {
                            DynDcci(reinterpret_cast<__gm__ uint8_t *>(ready),
                                    sizeof(int32_t));
                            if ((spins & 1023u) == 0u &&
                                DynAbortRequested(
                                    ctrl,
                                    static_cast<uint32_t>(generation))) {
                                fail = kDynCsrFailCancelled;
                                break;
                            }
                            ++spins;
                        }
                        if (fail == kDynCsrFailCancelled) {
                            break;
                        } else if (*ready == failed_generation) {
                            fail = kDynCsrFailCsr;
                        } else if (*ready != ready_generation) {
                            fail = kDynCsrFailMissing;
                        } else if (ost->first_ready_cycle == 0u) {
                            ost->first_ready_cycle = GetSystemCycle();
                        }
                    }
                }
            }
            ost->ready_wait_cycles = GetSystemCycle() - ready_wait_t0;
        }

        uint32_t lazy_cached_word = 0xffffffffu;
        uint32_t lazy_cached_bits = 0u;
        __gm__ uint32_t *lazy_waited = nullptr;
        uint32_t lazy_flat = 0u;
        if (ctrl->ready_mode == 4u) {
            if (ctrl->worker_count == 0u ||
                ctrl->source_bitmap_words == 0u ||
                ctrl->group_count != ctrl->owner_count * ctrl->worker_count ||
                ctrl->waited_source_bitmap_off == 0u) {
                fail = kDynCsrFailCsr;
            } else {
                lazy_flat = owner;
                if (lazy_flat >= ctrl->owner_count) {
                    fail = kDynCsrFailHome;
                } else {
                    lazy_waited = reinterpret_cast<__gm__ uint32_t *>(
                        sym + ctrl->waited_source_bitmap_off +
                        static_cast<uint64_t>(lazy_flat) *
                            ctrl->source_bitmap_words * sizeof(uint32_t));
                    for (uint32_t word = 0u;
                         word < ctrl->source_bitmap_words; ++word) {
                        lazy_waited[word] = 0u;
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            }
        }

        // Result CSR is result-major, so the same packed owner/source chunk
        // recurs across adjacent rows.  This small direct-mapped register
        // cache avoids DCCI on an already-observed ready cacheline.  It is
        // only a hint: collisions fall back to the normal poll and therefore
        // do not impose a worker-count protocol limit.
        constexpr uint32_t kStreamReadyCacheEntries = 8u;
        uint32_t stream_ready_source[kStreamReadyCacheEntries];
        uint32_t stream_ready_slot[kStreamReadyCacheEntries];
        for (uint32_t cache = 0u; cache < kStreamReadyCacheEntries; ++cache) {
            stream_ready_source[cache] = 0xffffffffu;
            stream_ready_slot[cache] = 0xffffffffu;
        }
        const bool cyclic_owner_results =
            (ctrl->optimization_flags & kDynCsrOptCyclicOwnerResults) != 0u;
        const uint32_t result_begin = cyclic_owner_results ? owner : 0u;
        const uint32_t result_step = cyclic_owner_results ? oc : 1u;
        for (uint64_t rr = result_begin;
             rr < R && fail == kDynCsrFailNone; rr += result_step) {
            const uint32_t r = static_cast<uint32_t>(rr);
            if (!cyclic_owner_results && home_owner[r] != owner) {
                continue;
            }
            ++owned;
            const uint32_t b = result_offsets[r];
            const uint32_t e = result_offsets[r + 1];
            if (e < b || e > C) {
                fail = kDynCsrFailCsr;
                break;
            }
            const uint32_t expected = e - b;
            constexpr uint32_t kFp32OneBits = 0x3f800000u;
            // Select the relay protocol from this result descriptor, never
            // from the operator's declared/global top-k.  A persistent INC
            // can therefore process K1 and K>1 results in the same launch
            // with one fixed AIV layout.
            const bool result_k1_identity =
                expected == 1u && weights[b] == kFp32OneBits &&
                (ctrl->optimization_flags & kDynCsrOptK1IdentityCopy) != 0u;
            const bool result_counter_ready =
                (ctrl->optimization_flags &
                 kDynCsrOptResultArrivalCounter) != 0u;
            if (result_counter_ready) {
                const uint64_t target64 = generation * expected;
                if (expected == 0u || target64 > 0x7fffffffull ||
                    ctrl->result_arrival_counter_off == 0u) {
                    fail = kDynCsrFailCsr;
                    break;
                }
                __gm__ int32_t *counter =
                    reinterpret_cast<__gm__ int32_t *>(
                        sym + ctrl->result_arrival_counter_off +
                        static_cast<uint64_t>(r) * 64u);
                const int32_t target = static_cast<int32_t>(target64);
                const uint64_t wait_t0 = GetSystemCycle();
                uint32_t spins = 0u;
                while (*counter != target && spins < ctrl->ready_spin_cap) {
                    DynDcci(reinterpret_cast<__gm__ uint8_t *>(counter),
                            sizeof(int32_t));
                    if ((spins & 1023u) == 0u &&
                        DynAbortRequested(ctrl,
                                          static_cast<uint32_t>(generation))) {
                        fail = kDynCsrFailCancelled;
                        break;
                    }
                    ++spins;
                }
                ost->ready_wait_cycles += GetSystemCycle() - wait_t0;
                if (fail != kDynCsrFailNone || *counter != target) {
                    if (fail == kDynCsrFailNone) {
                        fail = kDynCsrFailMissing;
                    }
                    break;
                }
                if (ost->first_ready_cycle == 0u) {
                    ost->first_ready_cycle = GetSystemCycle();
                }
            }
            if (ctrl->ready_mode == 4u) {
                for (uint32_t i = b; i < e; ++i) {
                    const uint32_t source = sources[i];
                    if (source >= ctrl->worker_count) {
                        fail = kDynCsrFailCsr;
                        break;
                    }
                    const uint32_t word = source >> 5u;
                    const uint32_t bit = 1u << (source & 31u);
                    if (word != lazy_cached_word) {
                        if (lazy_cached_word != 0xffffffffu) {
                            lazy_waited[lazy_cached_word] = lazy_cached_bits;
                        }
                        lazy_cached_word = word;
                        lazy_cached_bits = lazy_waited[word];
                    }
                    if ((lazy_cached_bits & bit) != 0u) {
                        continue;
                    }
                    const uint32_t group =
                        lazy_flat * ctrl->worker_count + source;
                    __gm__ int32_t *ready =
                        reinterpret_cast<__gm__ int32_t *>(
                            sym + ctrl->ready_generation_off +
                            static_cast<uint64_t>(group) *
                                ctrl->ready_stride_bytes);
                    const uint64_t wait_t0 = GetSystemCycle();
                    uint32_t spins = 0u;
                    while (*ready != static_cast<int32_t>(generation) &&
                           spins < ctrl->ready_spin_cap) {
                        DynDcci(reinterpret_cast<__gm__ uint8_t *>(ready),
                                sizeof(int32_t));
                        if ((spins & 1023u) == 0u &&
                            DynAbortRequested(
                                ctrl, static_cast<uint32_t>(generation))) {
                            fail = kDynCsrFailCancelled;
                            break;
                        }
                        ++spins;
                    }
                    ost->ready_wait_cycles += GetSystemCycle() - wait_t0;
                    if (fail == kDynCsrFailCancelled) {
                        break;
                    } else if (*ready != static_cast<int32_t>(generation)) {
                        fail = kDynCsrFailMissing;
                        break;
                    }
                    if (ost->first_ready_cycle == 0u) {
                        ost->first_ready_cycle = GetSystemCycle();
                    }
                    lazy_cached_bits |= bit;
                }
                if (fail != kDynCsrFailNone) {
                    break;
                }
            }
            const bool compact_bitmap = expected <= 64u;
            const bool local_bitmap = compact_bitmap &&
                (ctrl->optimization_flags &
                 kDynCsrOptLocalOrdinalBitmap) != 0u;
            // This result is owner-private.  The optimized path validates in
            // a register and publishes the final bitmap once, avoiding a GM
            // RMW plus PIPE_ALL for every contribution/result.
            uint64_t seen = 0ull;
            if (compact_bitmap && !local_bitmap) {
                arrival[r] = 0ull;
                AscendC::PipeBarrier<PIPE_ALL>();
            } else if (!compact_bitmap) {
                for (uint32_t marker = b; marker < e; ++marker) {
                    arrival[static_cast<uint64_t>(R) + marker] = 0ull;
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            // First pass: validate arrival / generation / unique ordinal.
            uint32_t arrived = 0;
            for (uint32_t i = b; i < e; ++i) {
                if (!result_counter_ready &&
                    (ctrl->ready_mode == 1u || ctrl->ready_mode == 5u ||
                     ctrl->ready_mode == 6u)) {
                    if (slots[i] >= ctrl->max_ingress_slots ||
                        ctrl->ready_stride_bytes < sizeof(int32_t)) {
                        fail = kDynCsrFailSlot;
                        break;
                    }
                    uint32_t ready_slot = slots[i];
                    uint32_t ready_source = 0xffffffffu;
                    if (ctrl->ready_mode == 5u) {
                        const uint32_t source = sources[i];
                        ready_source = source;
                        const uint32_t flat = owner;
                        if (source >= ctrl->worker_count ||
                            flat >= ctrl->owner_count) {
                            fail = kDynCsrFailCsr;
                            break;
                        }
                        const uint32_t group =
                            flat * ctrl->worker_count + source;
                        const uint32_t group_begin = group_offsets[group];
                        const uint32_t group_end = group_offsets[group + 1u];
                        if (group_end <= group_begin ||
                            group_end > ctrl->contribution_count) {
                            fail = kDynCsrFailCsr;
                            break;
                        }
                        const uint32_t first_i =
                            group_entries[group_begin];
                        if (first_i >= ctrl->contribution_count ||
                            slots[i] < slots[first_i]) {
                            fail = kDynCsrFailCsr;
                            break;
                        }
                        uint32_t ready_chunk_tiles =
                            ctrl->tile_bytes == 0u
                                ? 1u
                                : ctrl->coalesced_chunk_bytes /
                                      ctrl->tile_bytes;
                        if (ready_chunk_tiles == 0u) {
                            ready_chunk_tiles = 1u;
                        }
                        const uint32_t relative = slots[i] - slots[first_i];
                        const bool head_tile_stream =
                            (ctrl->optimization_flags &
                             kDynCsrOptHeadTileStream) != 0u;
                        ready_slot =
                            slots[first_i] +
                            (head_tile_stream && relative != 0u
                                 ? 1u + ((relative - 1u) /
                                         ready_chunk_tiles) *
                                            ready_chunk_tiles
                                 : (relative / ready_chunk_tiles) *
                                       ready_chunk_tiles);
                    } else if (ctrl->ready_mode == 6u) {
                        const uint32_t source = sources[i];
                        ready_source = source;
                        if (source >= ctrl->worker_count) {
                            fail = kDynCsrFailCsr;
                            break;
                        }
                        const uint32_t group = source;
                        const uint32_t group_begin =
                            source_contribution_offsets[group];
                        const uint32_t group_end =
                            source_contribution_offsets[group + 1u];
                        if (group_end <= group_begin ||
                            group_end > ctrl->contribution_count) {
                            fail = kDynCsrFailCsr;
                            break;
                        }
                        const uint32_t first_i =
                            source_contribution_entries[group_begin];
                        if (first_i >= ctrl->contribution_count ||
                            slots[i] < slots[first_i]) {
                            fail = kDynCsrFailCsr;
                            break;
                        }
                        uint32_t ready_chunk_tiles =
                            ctrl->tile_bytes == 0u
                                ? 1u
                                : ctrl->coalesced_chunk_bytes /
                                      ctrl->tile_bytes;
                        if (ready_chunk_tiles == 0u) {
                            ready_chunk_tiles = 1u;
                        }
                        const uint32_t relative =
                            slots[i] - slots[first_i];
                        const bool pair_ready =
                            (ctrl->optimization_flags &
                             kDynCsrOptK1PrivateMtePush) != 0u &&
                            (ctrl->optimization_flags &
                             kDynCsrOptK1PairReady) != 0u &&
                            ctrl->contribution_count == ctrl->result_count &&
                            ctrl->tile_bytes == kIncDcPrivateMtePacketBytes &&
                            ctrl->producer_lane_count != 0u &&
                            ctrl->local_rank_prereduce == 0u;
                        if (pair_ready) {
                            const uint32_t lane =
                                relative % ctrl->producer_lane_count;
                            const uint32_t lane_sequence =
                                relative / ctrl->producer_lane_count;
                            ready_slot =
                                slots[first_i] + lane +
                                (lane_sequence / 2u) * 2u *
                                    ctrl->producer_lane_count;
                        } else {
                            ready_slot =
                                slots[first_i] +
                                (relative / ready_chunk_tiles) *
                                    ready_chunk_tiles;
                        }
                    }
                    bool ready_cached = false;
                    uint32_t cache_index = 0u;
                    if (ctrl->ready_mode == 5u || ctrl->ready_mode == 6u) {
                        cache_index =
                            ready_source & (kStreamReadyCacheEntries - 1u);
                        ready_cached =
                            stream_ready_source[cache_index] == ready_source &&
                            stream_ready_slot[cache_index] == ready_slot;
                    }
                    if (!ready_cached) {
                        __gm__ int32_t *ready =
                            reinterpret_cast<__gm__ int32_t *>(
                                sym + ctrl->ready_generation_off +
                                static_cast<uint64_t>(ready_slot) *
                                    ctrl->ready_stride_bytes);
                        const uint64_t wait_t0 = GetSystemCycle();
                        uint32_t spins = 0;
                        while (*ready != static_cast<int32_t>(generation) &&
                               spins < ctrl->ready_spin_cap) {
                            DynDcci(
                                reinterpret_cast<__gm__ uint8_t *>(ready),
                                sizeof(int32_t));
                            if ((spins & 1023u) == 0u &&
                                DynAbortRequested(
                                    ctrl,
                                    static_cast<uint32_t>(generation))) {
                                fail = kDynCsrFailCancelled;
                                break;
                            }
                            ++spins;
                        }
                        ost->ready_wait_cycles += GetSystemCycle() - wait_t0;
                        if (fail == kDynCsrFailCancelled) {
                            break;
                        } else if (*ready !=
                                   static_cast<int32_t>(generation)) {
                            fail = kDynCsrFailMissing;
                            break;
                        }
                        if (ctrl->ready_mode == 5u ||
                            ctrl->ready_mode == 6u) {
                            stream_ready_source[cache_index] = ready_source;
                            stream_ready_slot[cache_index] = ready_slot;
                        }
                    }
                }
                // Generation zero is a route-template entry bound to the
                // enclosing operation generation.  Any explicit generation
                // must match exactly, so stale/future fault injection remains
                // fail-closed.
                if (gens[i] != 0u && gens[i] != generation) {
                    ++stale_rej;
                    fail = kDynCsrFailStale;
                    break;
                }
                const uint32_t o = ord[i];
                if (o >= expected) {
                    fail = kDynCsrFailCsr;
                    break;
                }
                const uint64_t bit =
                    compact_bitmap ? (1ull << o) : 0ull;
                const uint64_t observed =
                    compact_bitmap
                        ? (local_bitmap ? seen : arrival[r])
                        : arrival[static_cast<uint64_t>(R) + b + o];
                const bool duplicate = compact_bitmap
                                           ? ((observed & bit) != 0ull)
                                           : (observed == generation);
                if (duplicate) {
                    ++dup_rej;
                    if (ctrl->fail_closed_on_dup != 0u) {
                        fail = kDynCsrFailDup;
                        break;
                    }
                    continue; // reject dup without counting
                }
                if (local_bitmap) {
                    seen |= bit;
                } else if (!compact_bitmap) {
                    arrival[static_cast<uint64_t>(R) + b + o] = generation;
                } else {
                    arrival[r] |= bit;
                }
                ++arrived;
                if (slots[i] >= ctrl->max_ingress_slots) {
                    fail = kDynCsrFailSlot;
                    break;
                }
            }
            if (fail != kDynCsrFailNone) {
                break;
            }
            if (arrived != expected) {
                fail = kDynCsrFailMissing;
                break;
            }
            if (local_bitmap && !result_k1_identity) {
                arrival[r] = seen;
            } else if (!compact_bitmap) {
                arrival[r] = arrived;
            }

            // CSR vector accumulate — sparse over contribution entries only.
            uint32_t storage_row = r;
            if ((ctrl->optimization_flags &
                 kDynCsrOptRankPackedResultTx) != 0u) {
                const uint32_t dst_rank = result_dst_rank[r];
                if (dst_rank >= ctrl->worker_count) {
                    fail = kDynCsrFailHome;
                    break;
                }
                storage_row = result_tx_rank_offsets[dst_rank] +
                              result_dst_row[r];
                if (storage_row >= R) {
                    fail = kDynCsrFailHome;
                    break;
                }
            }
            __gm__ uint16_t *out_row =
                (__gm__ uint16_t *)(output +
                                   static_cast<uint64_t>(storage_row) *
                                       tile_bytes);
            if (!reduce_timing_started) {
                ost->first_reduce_cycle = GetSystemCycle();
                reduce_timing_started = true;
            }
            const bool k1_identity = result_k1_identity;
            const bool k2_identity =
                expected == 2u && weights[b] == kFp32OneBits &&
                weights[b + 1u] == kFp32OneBits;
            const bool direct_ub_result_tx =
                ctrl->tx_lane_count == 0u &&
                (ctrl->optimization_flags & kDynCsrOptRemoteResultTx) != 0u &&
                (ctrl->optimization_flags & kDynCsrOptRankPackedResultTx) == 0u &&
                (ctrl->optimization_flags & kDynCsrOptBatchResultTx) == 0u;
            int32_t direct_ub_result_pe = -1;
            __gm__ uint16_t *reduce_out_row = out_row;
            if (direct_ub_result_tx) {
                const uint32_t dst_rank = result_dst_rank[r];
                const uint32_t dst_row = result_dst_row[r];
                if (dst_rank >= ctrl->worker_count ||
                    dst_row >= ctrl->result_count || worker_pes == nullptr) {
                    fail = kDynCsrFailHome;
                    break;
                }
                direct_ub_result_pe =
                    static_cast<int32_t>(worker_pes[dst_rank]);
                reduce_out_row = reinterpret_cast<__gm__ uint16_t *>(
                    output + static_cast<uint64_t>(dst_row) * tile_bytes);
                direct_ub_result_tx_issued = true;
            }
            // A mixed plan may place a K>1 result after identity-K1 relay
            // packets on the same owner.  Drain the two relay-private slots
            // before a vector reducer reuses base-zero UB/events.  Pure K>1
            // plans never enter this branch and retain their existing path.
            if (!k1_identity &&
                (k1_relay_busy[0] || k1_relay_busy[1])) {
                if (k1_relay_busy[0]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
                    k1_relay_busy[0] = false;
                }
                if (k1_relay_busy[1]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
                    k1_relay_busy[1] = false;
                }
                aclshmemx_mte_quiet();
            }
            const uint32_t vec_rc =
                k1_identity
                    ? (direct_ub_result_tx
                           ? DynCsrK1StrictIncRelayIssue(
                                 ingress, reduce_out_row, slots, b,
                                 tile_bytes, direct_ub_result_pe,
                                 k1_relay_busy, &k1_relay_ping)
                           : DynCsrK1IdentityCopy(
                                 ingress, out_row, slots, b, hidden,
                                 tile_bytes))
                    : k2_identity
                    ? DynCsrK2IdentityAdd(ingress, reduce_out_row, slots, b,
                                          hidden, tile_bytes,
                                          direct_ub_result_pe)
                    : DynCsrVectorReduce(ingress, reduce_out_row, slots, weights, b,
                                         e, hidden, tile_bytes,
                                         (ctrl->optimization_flags &
                                          kDynCsrOptFirstContributionInit) !=
                                             0u,
                                         (ctrl->optimization_flags &
                                          kDynCsrOptWideVectorTile) != 0u,
                                         batch_result_tx,
                                         direct_ub_result_pe);
            if (vec_rc != kDynCsrFailNone) {
                fail = vec_rc;
                break;
            }
            if (k1_identity) {
                // Per-result cycle sampling and a GM telemetry write are
                // visible overhead at 16-KiB relay granularity.  Record the
                // final K1 issue once at owner close instead.
                k1_timing_pending = true;
            } else {
                ost->last_reduce_cycle = GetSystemCycle();
            }
            if (!direct_ub_result_tx &&
                (ctrl->output_dcci_small_only == 0u || tile_bytes <= 512u)) {
                DynDcci((__gm__ uint8_t *)out_row, tile_bytes);
            }
            if ((ctrl->optimization_flags &
                 kDynCsrOptRankPackedResultTx) != 0u &&
                ctrl->tx_lane_count == 0u &&
                (ctrl->optimization_flags & kDynCsrOptRemoteResultTx) != 0u) {
                const uint32_t dst_rank = result_dst_rank[r];
                const uint32_t dst_row = result_dst_row[r];
                const bool contiguous =
                    packed_tx_rows != 0u &&
                    packed_tx_worker == dst_rank &&
                    packed_tx_begin + packed_tx_rows == dst_row;
                if (packed_tx_rows != 0u && !contiguous) {
                    if (packed_tx_inflight) {
                        aclshmem_quiet();
                        packed_tx_inflight = false;
                    }
                    __gm__ uint8_t *src =
                        output +
                        static_cast<uint64_t>(
                            result_tx_rank_offsets[packed_tx_worker] +
                            packed_tx_begin) * tile_bytes;
                    __gm__ uint8_t *dst =
                        output + static_cast<uint64_t>(packed_tx_begin) *
                                     tile_bytes;
                    aclshmem_putmem_nbi(
                        dst, src, packed_tx_rows * tile_bytes,
                        static_cast<int32_t>(worker_pes[packed_tx_worker]));
                    packed_tx_inflight = true;
                    packed_tx_rows = 0u;
                }
                if (packed_tx_rows == 0u) {
                    packed_tx_worker = dst_rank;
                    packed_tx_begin = dst_row;
                }
                ++packed_tx_rows;
                const uint32_t packed_chunk_bytes =
                    ctrl->coalesced_chunk_bytes < 256u * 1024u
                        ? 256u * 1024u
                        : ctrl->coalesced_chunk_bytes;
                if (static_cast<uint64_t>(packed_tx_rows) * tile_bytes >=
                    packed_chunk_bytes) {
                    if (packed_tx_inflight) {
                        aclshmem_quiet();
                        packed_tx_inflight = false;
                    }
                    __gm__ uint8_t *src =
                        output +
                        static_cast<uint64_t>(
                            result_tx_rank_offsets[packed_tx_worker] +
                            packed_tx_begin) * tile_bytes;
                    __gm__ uint8_t *dst =
                        output + static_cast<uint64_t>(packed_tx_begin) *
                                     tile_bytes;
                    aclshmem_putmem_nbi(
                        dst, src, packed_tx_rows * tile_bytes,
                        static_cast<int32_t>(worker_pes[packed_tx_worker]));
                    packed_tx_inflight = true;
                    packed_tx_rows = 0u;
                }
            } else if (batch_result_tx) {
                result_tx_batch[result_tx_batch_count++] = r;
                if (result_tx_batch_count == kResultTxBatch) {
                    AscendC::PipeBarrier<PIPE_ALL>();
                    if (ctrl->tile_bytes > kIncDcPrivateMtePacketBytes ||
                        (ctrl->optimization_flags &
                         kDynCsrOptLocalRankPrereduce) != 0u) {
                        if ((ctrl->optimization_flags &
                             kDynCsrOptLocalRankPrereduce) != 0u) {
                            dcci_entire_cache();
                        }
                        fail = DynCsrMtePingPongResultBatch(
                            output, ctrl, result_dst_rank, result_dst_row,
                            worker_pes, result_tx_batch,
                            result_tx_batch_count);
                    } else for (uint32_t q = 0u;
                                q < result_tx_batch_count; ++q) {
                        const uint32_t qr = result_tx_batch[q];
                        const uint32_t dst_rank = result_dst_rank[qr];
                        if (dst_rank >= ctrl->worker_count) {
                            fail = kDynCsrFailHome;
                            break;
                        }
                        __gm__ uint8_t *src =
                            output + static_cast<uint64_t>(qr) * tile_bytes;
                        __gm__ uint8_t *dst =
                            output + static_cast<uint64_t>(result_dst_row[qr]) *
                                         tile_bytes;
                        aclshmem_putmem_nbi(dst, src, tile_bytes,
                            static_cast<int32_t>(worker_pes[dst_rank]));
                        ++tx_pending;
                        const uint32_t tx_window =
                            ctrl->tx_quiet_window == 0u
                                ? 1u
                                : ctrl->tx_quiet_window;
                        if (tx_pending >= tx_window) {
                            aclshmem_quiet();
                            tx_pending = 0u;
                        }
                    }
                    result_tx_batch_count = 0u;
                }
            } else if (!direct_ub_result_tx &&
                       (ctrl->optimization_flags &
                 kDynCsrOptRemoteResultTx) != 0u &&
                (ctrl->optimization_flags &
                 kDynCsrOptRankPackedResultTx) == 0u &&
                ctrl->tx_lane_count == 0u) {
                const uint32_t dst_rank = result_dst_rank[r];
                const uint32_t dst_row = result_dst_row[r];
                if (dst_rank >= ctrl->worker_count ||
                    worker_pes == nullptr) {
                    fail = kDynCsrFailHome;
                    break;
                }
                __gm__ uint8_t *remote_row =
                    output + static_cast<uint64_t>(dst_row) * tile_bytes;
                aclshmem_putmem_nbi(
                    reinterpret_cast<__gm__ void *>(remote_row),
                    reinterpret_cast<__gm__ void *>(out_row), tile_bytes,
                    static_cast<int32_t>(worker_pes[dst_rank]));
                ++tx_pending;
                const uint32_t tx_window =
                    ctrl->tx_quiet_window == 0u ? 1u
                                                : ctrl->tx_quiet_window;
                if (tx_pending >= tx_window) {
                    aclshmem_quiet();
                    tx_pending = 0u;
                }
            }
            if (ctrl->tx_lane_count != 0u) {
                __gm__ uint8_t *ready =
                    sym + ctrl->result_tx_ready_off +
                    static_cast<uint64_t>(r) * 64u;
                AscendC::PipeBarrier<PIPE_ALL>();
                *reinterpret_cast<__gm__ uint32_t *>(ready) =
                    static_cast<uint32_t>(generation);
                DynDcci(ready, 64u);
            }
            ++reduced;
        }
        // Per-tile MTE3_MTE2 events above protect UB reuse.  One owner-level
        // completion fence is sufficient to publish every disjoint remote
        // result before the generation completion signal; fencing each row
        // made 16-KiB results latency-bound.
        if (k1_timing_pending) {
            ost->last_reduce_cycle = GetSystemCycle();
        }
        if (direct_ub_result_tx_issued) {
            if (k1_relay_busy[0]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
            }
            if (k1_relay_busy[1]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
            }
            aclshmemx_mte_quiet();
        }
        if (fail == kDynCsrFailNone && result_tx_batch_count != 0u) {
            AscendC::PipeBarrier<PIPE_ALL>();
            if (ctrl->tile_bytes > kIncDcPrivateMtePacketBytes ||
                (ctrl->optimization_flags &
                 kDynCsrOptLocalRankPrereduce) != 0u) {
                if ((ctrl->optimization_flags &
                     kDynCsrOptLocalRankPrereduce) != 0u) {
                    dcci_entire_cache();
                }
                fail = DynCsrMtePingPongResultBatch(
                    output, ctrl, result_dst_rank, result_dst_row, worker_pes,
                    result_tx_batch, result_tx_batch_count);
            } else for (uint32_t q = 0u;
                        q < result_tx_batch_count; ++q) {
                const uint32_t qr = result_tx_batch[q];
                const uint32_t dst_rank = result_dst_rank[qr];
                if (dst_rank >= ctrl->worker_count) {
                    fail = kDynCsrFailHome;
                    break;
                }
                __gm__ uint8_t *src =
                    output + static_cast<uint64_t>(qr) * tile_bytes;
                __gm__ uint8_t *dst =
                    output + static_cast<uint64_t>(result_dst_row[qr]) *
                                 tile_bytes;
                aclshmem_putmem_nbi(dst, src, tile_bytes,
                    static_cast<int32_t>(worker_pes[dst_rank]));
                ++tx_pending;
                const uint32_t tx_window =
                    ctrl->tx_quiet_window == 0u
                        ? 1u
                        : ctrl->tx_quiet_window;
                if (tx_pending >= tx_window) {
                    aclshmem_quiet();
                    tx_pending = 0u;
                }
            }
        }
        if (fail == kDynCsrFailNone && packed_tx_rows != 0u) {
            if (packed_tx_inflight) {
                aclshmem_quiet();
                packed_tx_inflight = false;
            }
            __gm__ uint8_t *src =
                output +
                static_cast<uint64_t>(
                    result_tx_rank_offsets[packed_tx_worker] +
                    packed_tx_begin) * tile_bytes;
            __gm__ uint8_t *dst =
                output + static_cast<uint64_t>(packed_tx_begin) * tile_bytes;
            aclshmem_putmem_nbi(
                dst, src, packed_tx_rows * tile_bytes,
                static_cast<int32_t>(worker_pes[packed_tx_worker]));
            packed_tx_inflight = true;
            packed_tx_rows = 0u;
        }
        if (packed_tx_inflight) {
            aclshmem_quiet();
            packed_tx_inflight = false;
        }
        if (tx_pending != 0u) {
            aclshmem_quiet();
        }
        if (ctrl->ready_mode == 4u &&
            lazy_cached_word != 0xffffffffu && lazy_waited != nullptr) {
            lazy_waited[lazy_cached_word] = lazy_cached_bits;
        }

        if (fail == kDynCsrFailNone &&
            (ctrl->optimization_flags & kDynCsrOptRankPackedResultTx) != 0u &&
            ctrl->tx_lane_count == 0u &&
            (ctrl->optimization_flags & kDynCsrOptRemoteResultTx) != 0u) {
            AscendC::PipeBarrier<PIPE_ALL>();
            fail = DynCsrMtePackedOwnerStripe(
                output, ctrl, result_tx_rank_offsets, worker_pes, owner, oc);
        }
        if (fail == kDynCsrFailNone && ctrl->tx_lane_count != 0u &&
            (ctrl->optimization_flags &
             kDynCsrOptRankPackedResultTx) == 0u) {
            // Dedicated TX shards start immediately.  After an owner has
            // produced all of its rows, the same multifunction AIV joins the
            // egress pool and drains one additional disjoint shard.  The INC
            // role allocation is fixed; only idle lanes steal ready work.
            DynCsrSplitTxLane(
                sym, ctrl, ctrl->tx_lane_count + owner,
                ctrl->tx_lane_count + oc, false);
        }

        const uint64_t t1 = GetSystemCycle();
        ost->fail_code = fail;
        ost->reduced = reduced;
        ost->reduce_cycles = t1 - t0;
        ost->done = static_cast<uint32_t>(generation);
        AscendC::PipeBarrier<PIPE_ALL>();
        DynDcci((__gm__ uint8_t *)ost, sizeof(DynCsrOwnerStats));

        if ((ctrl->optimization_flags & kDynCsrOptRankPackedResultTx) != 0u &&
            ctrl->tx_lane_count == 0xffffffffu &&
            (ctrl->optimization_flags & kDynCsrOptRemoteResultTx) != 0u) {
            // Phase 2 reuses every reducer as a bulk TX lane.  Waiting here
            // is local to the INC and does not consume additional AIVs; once
            // all packed rows are complete, each owner pushes a disjoint
            // contiguous stripe to every destination.
            for (uint32_t o = 0u; o < oc; ++o) {
                __gm__ DynCsrOwnerStats *peer =
                    reinterpret_cast<__gm__ DynCsrOwnerStats *>(
                        sym + ctrl->owner_stats_off +
                        static_cast<uint64_t>(o) *
                            sizeof(DynCsrOwnerStats));
                uint32_t spins = 0u;
                while (peer->done != static_cast<uint32_t>(generation) &&
                       spins < ctrl->ready_spin_cap) {
                    DynDcci(reinterpret_cast<__gm__ uint8_t *>(peer),
                            sizeof(DynCsrOwnerStats));
                    ++spins;
                }
                if (peer->done != static_cast<uint32_t>(generation)) {
                    fail = kDynCsrFailMissing;
                    break;
                }
                if (peer->fail_code != kDynCsrFailNone) {
                    fail = peer->fail_code;
                    break;
                }
            }
            __gm__ uint32_t *rank_offsets =
                reinterpret_cast<__gm__ uint32_t *>(
                    sym + ctrl->result_tx_rank_offsets_off);
            constexpr uint64_t kMaxMtePacket =
                256ull * 1024ull * 1024ull;
            const uint32_t ub_tile =
                tile_bytes < kIncDcPrivateMtePacketBytes
                    ? tile_bytes : kIncDcPrivateMtePacketBytes;
            __ubuf__ uint8_t *tx_ub[2] = {
                reinterpret_cast<__ubuf__ uint8_t *>(0),
                reinterpret_cast<__ubuf__ uint8_t *>(
                    static_cast<uint64_t>(ub_tile))};
            bool tx_busy[2] = {false, false};
            uint32_t tx_ping = 0u;
            for (uint32_t worker = 0u;
                 worker < ctrl->worker_count && fail == kDynCsrFailNone;
                 ++worker) {
                const uint32_t count = rank_offsets[worker + 1u] -
                                       rank_offsets[worker];
                const uint32_t row_begin = static_cast<uint32_t>(
                    static_cast<uint64_t>(count) * owner / oc);
                const uint32_t row_end = static_cast<uint32_t>(
                    static_cast<uint64_t>(count) * (owner + 1u) / oc);
                const uint64_t bytes =
                    static_cast<uint64_t>(row_end - row_begin) * tile_bytes;
                uint64_t within = 0u;
                while (within < bytes) {
                    const uint32_t packet = static_cast<uint32_t>(
                        bytes - within < kMaxMtePacket
                            ? bytes - within : kMaxMtePacket);
                    const uint32_t bi = tx_ping;
                    if (tx_busy[bi]) {
                        if (bi == 0u) {
                            AscendC::WaitFlag<
                                AscendC::HardEvent::MTE3_MTE2>(0u);
                        } else {
                            AscendC::WaitFlag<
                                AscendC::HardEvent::MTE3_MTE2>(1u);
                        }
                        tx_busy[bi] = false;
                    }
                    __gm__ uint8_t *src =
                        output +
                        static_cast<uint64_t>(rank_offsets[worker] +
                                              row_begin) * tile_bytes +
                        within;
                    __gm__ uint8_t *dst =
                        output + static_cast<uint64_t>(row_begin) *
                                     tile_bytes + within;
                    aclshmemx_mte_put_nbi(
                        dst, src, tx_ub[bi], ub_tile, packet,
                        static_cast<int32_t>(worker_pes[worker]), bi);
                    if (bi == 0u) {
                        AscendC::SetFlag<
                            AscendC::HardEvent::MTE3_MTE2>(0u);
                    } else {
                        AscendC::SetFlag<
                            AscendC::HardEvent::MTE3_MTE2>(1u);
                    }
                    tx_busy[bi] = true;
                    tx_ping = 1u - tx_ping;
                    within += packet;
                }
            }
            if (tx_busy[0]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
            }
            if (tx_busy[1]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
            }
            aclshmemx_mte_quiet();
            ost->fail_code = fail;
            ost->reserved[0] = static_cast<uint32_t>(generation);
            AscendC::PipeBarrier<PIPE_ALL>();
            DynDcci(reinterpret_cast<__gm__ uint8_t *>(ost),
                    sizeof(DynCsrOwnerStats));
            if (owner == 0u) {
                for (uint32_t o = 0u; o < oc; ++o) {
                    __gm__ DynCsrOwnerStats *peer =
                        reinterpret_cast<__gm__ DynCsrOwnerStats *>(
                            sym + ctrl->owner_stats_off +
                            static_cast<uint64_t>(o) *
                                sizeof(DynCsrOwnerStats));
                    uint32_t spins = 0u;
                    while (peer->reserved[0] !=
                               static_cast<uint32_t>(generation) &&
                           spins < ctrl->ready_spin_cap) {
                        DynDcci(reinterpret_cast<__gm__ uint8_t *>(peer),
                                sizeof(DynCsrOwnerStats));
                        ++spins;
                    }
                    if (peer->reserved[0] !=
                        static_cast<uint32_t>(generation)) {
                        fail = kDynCsrFailMissing;
                    } else if (peer->fail_code != kDynCsrFailNone) {
                        fail = peer->fail_code;
                    }
                }
            }
        }

        // Owner 0 aggregates global stats.
        if (owner == 0u) {
            const uint64_t completion_aggregate_t0 = GetSystemCycle();
            __gm__ DynCsrStats *st = (__gm__ DynCsrStats *)(sym + ctrl->stats_off);
            uint32_t total_reduced = 0;
            uint32_t total_owned = 0;
            uint32_t any_fail = fail;
            uint64_t cycles = t1 - t0;
            for (uint32_t o = 0; o < oc; ++o) {
                __gm__ DynCsrOwnerStats *os =
                    (__gm__ DynCsrOwnerStats *)(sym + ctrl->owner_stats_off +
                                               static_cast<uint64_t>(o) *
                                                   sizeof(DynCsrOwnerStats));
                // Wait peers (bounded spin).
                uint32_t spins = 0;
                while (os->done != static_cast<uint32_t>(generation) &&
                       spins < 2000000u) {
                    DynDcci((__gm__ uint8_t *)os, sizeof(DynCsrOwnerStats));
                    if ((spins & 1023u) == 0u &&
                        DynAbortRequested(
                            ctrl, static_cast<uint32_t>(generation))) {
                        any_fail = kDynCsrFailCancelled;
                        break;
                    }
                    ++spins;
                }
                if (os->done != static_cast<uint32_t>(generation)) {
                    if (any_fail != kDynCsrFailCancelled) {
                        any_fail = kDynCsrFailMissing;
                    }
                    continue;
                }
                if (os->fail_code != kDynCsrFailNone) {
                    any_fail = os->fail_code;
                }
                total_reduced += os->reduced;
                if (os->reduce_cycles > cycles) {
                    cycles = os->reduce_cycles;
                }
                (void)total_owned;
            }
            if (ctrl->tx_lane_count != 0u) {
                for (uint32_t lane = 0u; lane < ctrl->tx_lane_count; ++lane) {
                    __gm__ uint8_t *done =
                        sym + ctrl->tx_done_off +
                        static_cast<uint64_t>(lane) * 64u;
                    uint32_t spins = 0u;
                    while (*reinterpret_cast<__gm__ volatile uint32_t *>(done) !=
                               static_cast<uint32_t>(generation) &&
                           spins < ctrl->ready_spin_cap) {
                        dcci_cacheline(done);
                        ++spins;
                    }
                    if (*reinterpret_cast<__gm__ volatile uint32_t *>(done) !=
                        static_cast<uint32_t>(generation)) {
                        any_fail = kDynCsrFailMissing;
                        continue;
                    }
                    const uint32_t tx_fail =
                        reinterpret_cast<__gm__ uint32_t *>(done)[1];
                    if (tx_fail != kDynCsrFailNone) any_fail = tx_fail;
                }
            }
            const uint64_t completion_aggregate_t1 = GetSystemCycle();
            st->magic = kDynCsrMagic;
            st->done = 1;
            st->fail_code = any_fail;
            st->reduced = total_reduced;
            st->dup_rejected = dup_rej;
            st->stale_rejected = stale_rej;
            st->owned_results = owned;
            st->reduce_cycles = cycles;
            AscendC::PipeBarrier<PIPE_ALL>();
            DynDcci((__gm__ uint8_t *)st, sizeof(DynCsrStats));

            if (ctrl->device_completion != 0u &&
                ctrl->worker_count != 0u &&
                ctrl->worker_pe_off != 0u) {
                const uint64_t completion_fanout_t0 = GetSystemCycle();
                __gm__ uint32_t *worker_pes =
                    reinterpret_cast<__gm__ uint32_t *>(
                        sym + ctrl->worker_pe_off);
                // Do not source an outgoing completion from the same local
                // cacheline that receives the worker ACK.  A fast worker can
                // otherwise write -generation while the INC MTE is still
                // reading +generation, producing a rare final-epoch miss.
                // Stats is owner-0 private here and is republished below
                // before the host can observe it, so its first cacheline is a
                // safe fixed staging line requiring no shape-dependent heap.
                __gm__ int32_t *completion_stage =
                    reinterpret_cast<__gm__ int32_t *>(st);
                *completion_stage = static_cast<int32_t>(generation);
                AscendC::PipeBarrier<PIPE_ALL>();
                DynDcci(reinterpret_cast<__gm__ uint8_t *>(completion_stage),
                        ctrl->ready_stride_bytes);
                if ((ctrl->optimization_flags &
                     kDynCsrOptCompletionMteFanout) != 0u) {
                    // Result relay events/UB are fully drained before owner
                    // aggregation.  Reuse two small private slots to pipeline
                    // completion fanout without relying on the public NBI
                    // backend's shared packet slot.  A slot is never reused
                    // until its MTE3 completion event fires, and the final
                    // quiet precedes every worker ACK wait.
                    constexpr uint32_t kCompletionBytes = 64u;
                    __ubuf__ uint8_t *completion_ub[2] = {
                        reinterpret_cast<__ubuf__ uint8_t *>(0),
                        reinterpret_cast<__ubuf__ uint8_t *>(
                            kCompletionBytes)};
                    bool completion_busy[2] = {false, false};
                    uint32_t completion_ping = 0u;
                    for (uint32_t worker = 0u;
                         worker < ctrl->worker_count; ++worker) {
                        const uint32_t bi = completion_ping;
                        if (completion_busy[bi]) {
                            if (bi == 0u) {
                                AscendC::WaitFlag<
                                    AscendC::HardEvent::MTE3_MTE2>(0u);
                            } else {
                                AscendC::WaitFlag<
                                    AscendC::HardEvent::MTE3_MTE2>(1u);
                            }
                            completion_busy[bi] = false;
                        }
                        const uint64_t completion =
                            DynCollectiveCompletionBase(ctrl) +
                            worker;
                        __gm__ int32_t *done =
                            reinterpret_cast<__gm__ int32_t *>(
                                sym + ctrl->ready_generation_off +
                                completion * ctrl->ready_stride_bytes);
                        aclshmemx_mte_put_nbi(
                            reinterpret_cast<__gm__ uint8_t *>(done),
                            reinterpret_cast<__gm__ uint8_t *>(
                                completion_stage),
                            completion_ub[bi], kCompletionBytes,
                            ctrl->ready_stride_bytes,
                            static_cast<int32_t>(worker_pes[worker]), bi);
                        if (bi == 0u) {
                            AscendC::SetFlag<
                                AscendC::HardEvent::MTE3_MTE2>(0u);
                        } else {
                            AscendC::SetFlag<
                                AscendC::HardEvent::MTE3_MTE2>(1u);
                        }
                        completion_busy[bi] = true;
                        completion_ping = 1u - bi;
                    }
                    if (completion_busy[0]) {
                        AscendC::WaitFlag<
                            AscendC::HardEvent::MTE3_MTE2>(0u);
                    }
                    if (completion_busy[1]) {
                        AscendC::WaitFlag<
                            AscendC::HardEvent::MTE3_MTE2>(1u);
                    }
                    aclshmemx_mte_quiet();
                } else {
                    for (uint32_t worker = 0u;
                         worker < ctrl->worker_count; ++worker) {
                        const uint64_t completion =
                            DynCollectiveCompletionBase(ctrl) +
                            worker;
                        __gm__ int32_t *done =
                            reinterpret_cast<__gm__ int32_t *>(
                                sym + ctrl->ready_generation_off +
                                completion * ctrl->ready_stride_bytes);
                        // Use an explicitly published cacheline rather than
                        // signal_op.  The latter has no sufficiently strong
                        // mapped-cache visibility guarantee for a persistent
                        // multi-generation service.  A per-worker quiet also
                        // avoids the public backend's shared per-AIV NBI slot.
                        aclshmem_putmem_nbi(
                            reinterpret_cast<__gm__ void *>(done),
                            reinterpret_cast<__gm__ void *>(completion_stage),
                            ctrl->ready_stride_bytes,
                            static_cast<int32_t>(worker_pes[worker]));
                        aclshmem_quiet();
                    }
                }
                const uint64_t completion_fanout_t1 = GetSystemCycle();
                st->magic = kDynCsrMagic;
                AscendC::PipeBarrier<PIPE_ALL>();
                DynDcci(reinterpret_cast<__gm__ uint8_t *>(st), sizeof(*st));
                for (uint32_t worker = 0u; worker < ctrl->worker_count;
                     ++worker) {
                    const uint64_t completion =
                        DynCollectiveCompletionBase(ctrl) + worker;
                    __gm__ int32_t *done =
                        reinterpret_cast<__gm__ int32_t *>(
                            sym + ctrl->ready_generation_off +
                            completion * ctrl->ready_stride_bytes);
                    const int32_t ack =
                        -static_cast<int32_t>(generation);
                    if (!DynWaitCollectiveValue(ctrl, done, ack)) {
                        any_fail = kDynCsrFailCancelled;
                    }
                }
                const uint64_t completion_ack_t1 = GetSystemCycle();
                st->reserved[0] = static_cast<uint32_t>(
                    completion_aggregate_t1 - completion_aggregate_t0);
                st->reserved[1] = static_cast<uint32_t>(
                    completion_fanout_t1 - completion_fanout_t0);
                st->reserved[2] = static_cast<uint32_t>(
                    completion_ack_t1 - completion_fanout_t1);
                if (any_fail != kDynCsrFailNone) {
                    st->fail_code = any_fail;
                    AscendC::PipeBarrier<PIPE_ALL>();
                    DynDcci(reinterpret_cast<__gm__ uint8_t *>(st),
                            sizeof(DynCsrStats));
                } else {
                    AscendC::PipeBarrier<PIPE_ALL>();
                    DynDcci(reinterpret_cast<__gm__ uint8_t *>(st),
                            sizeof(DynCsrStats));
                }
            }
            DynPersistentServiceEnd(sym, ctrl, any_fail);
        }
    }
}

// Validation/benchmark start alignment is a separate queued kernel.  It
// completes before the measured start event, so rendezvous latency is not
// charged to the operation while all following producer/reducer launches are
// already resident in their device streams.  Production callers may omit it.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
inc_dc_sv2_dyn_csr_start_gate_kernel(__gm__ uint8_t *sym,
                                     uint64_t ctrl_off,
                                     uint32_t participant)
{
    if ASCEND_IS_AIV {
        __gm__ DynCsrCtrl *ctrl =
            reinterpret_cast<__gm__ DynCsrCtrl *>(sym + ctrl_off);
        DynDcci(reinterpret_cast<__gm__ uint8_t *>(ctrl), kDynCsrCtrlBytes);
        if (ctrl->magic != kDynCsrMagic || GetBlockIdx() != 0) {
            return;
        }
        __gm__ uint32_t *worker_pes =
            reinterpret_cast<__gm__ uint32_t *>(
                sym + ctrl->worker_pe_off);
        (void)DynDeviceGenerationStartGate(
            sym, ctrl, participant, true, worker_pes);
    }
}

extern "C" void launch_inc_dc_sv2_dyn_csr_combine_kernel(uint8_t *sym,
                                                         uint64_t ctrl_off,
                                                         int block_dim,
                                                         void *stream)
{
    inc_dc_sv2_dyn_csr_combine_kernel<<<block_dim, nullptr, stream>>>(sym,
                                                                     ctrl_off);
}

extern "C" void launch_inc_dc_sv2_dyn_csr_start_gate_kernel(
    uint8_t *sym, uint64_t ctrl_off, uint32_t participant, void *stream)
{
    inc_dc_sv2_dyn_csr_start_gate_kernel<<<1, nullptr, stream>>>(
        sym, ctrl_off, participant);
}

extern "C" void launch_inc_dc_sv2_dyn_csr_cycle_probe_kernel(
    uint8_t *sym, uint64_t cycle_off, void *stream)
{
    inc_dc_sv2_dyn_csr_cycle_probe_kernel<<<1, nullptr, stream>>>(
        sym, cycle_off);
}

extern "C" void launch_inc_dc_sv2_dyn_csr_producer_kernel(
    uint8_t *sym, uint64_t ctrl_off, int block_dim, void *stream)
{
    inc_dc_sv2_dyn_csr_producer_kernel<<<block_dim, nullptr, stream>>>(sym,
                                                                      ctrl_off);
}

extern "C" void launch_inc_dc_sv2_dyn_csr_k1_direct_tx_kernel(
    uint8_t *sym, uint64_t ctrl_off, int block_dim, void *stream)
{
    inc_dc_sv2_dyn_csr_k1_direct_tx_kernel<<<block_dim, nullptr, stream>>>(
        sym, ctrl_off);
}
