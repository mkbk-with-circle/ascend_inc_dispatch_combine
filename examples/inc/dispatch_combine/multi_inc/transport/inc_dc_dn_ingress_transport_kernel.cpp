/**
 * D1 raw ingress transport kernel：共用 inc_dc_dn_transport_abi.h，修复跨 AIV 汇总与 DCCI。
 */
#include "kernel_operator.h"
#include "shmem.h"
#include "shmemi_device_common.h"
#include "host/shmem_host_def.h"
#include "inc_dc_dn_transport_abi.h"

using inc::dc::dn::D1Ctr;
using inc::dc::dn::D1Desc;
using inc::dc::dn::D1ControlLine;
using inc::dc::dn::D1DoneLine;
using inc::dc::dn::D1LaneCtr;
using inc::dc::dn::D1PublishScratch;
using inc::dc::dn::D1LeaderRendezvous;
using inc::dc::dn::D1LeaderCompletion;
using inc::dc::dn::D1GlobalDoneLine;
using inc::dc::dn::D1GlobalDoneTiming;
using inc::dc::dn::D1PairDoneArrival;
using inc::dc::dn::D1LeaderTiming;
using inc::dc::dn::D1LocalReadyLine;
using inc::dc::dn::D1SessionStopLine;
using inc::dc::dn::D1StartLine;
using inc::dc::dn::D1TransportProbe;
using inc::dc::dn::D1WorkerDoneLine;
using inc::dc::dn::kD1Magic;
using inc::dc::dn::kD1MaxLanes;
using inc::dc::dn::kD1PrimP0PerTokenQuiet;
using inc::dc::dn::kD1PrimP1NbiFinalQuiet;
using inc::dc::dn::kD1PrimP2BlockingPut;
using inc::dc::dn::kD1PrimP3NbiMteUbComplete;
using inc::dc::dn::kD1PrimP4RangeQuiet;
using inc::dc::dn::kD1PrimP5LargeRangeQuiet;
using inc::dc::dn::kD1PrimP6RangeMteWaitFinalQuiet;
using inc::dc::dn::kD1PrimAutoRange;
using inc::dc::dn::kD1QuietBatch;
using inc::dc::dn::kD1LeaderPe;
using inc::dc::dn::kD1LeaderRendezvousOff;
using inc::dc::dn::kD1LeaderTimingOff;
using inc::dc::dn::kD1LocalReadyLineOff;
using inc::dc::dn::kD1PublishScratchOff;
using inc::dc::dn::kD1SessionStopMagic;
using inc::dc::dn::kD1TransportProbeOff;

__aicore__ inline void D1KDcciLine(__gm__ uint8_t *p)
{
    dcci_cacheline(p);
}

__aicore__ inline void D1KDcciLane(__gm__ D1LaneCtr *lc)
{
    D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(lc));
}

// 记录 peer transport / UB 配置（Step1）
__aicore__ inline void D1RecordTransportProbe(int peer, __gm__ D1TransportProbe *probe)
{
    __gm__ aclshmem_device_host_state_t *st = aclshmemi_get_state();
    probe->active_transport = st->topo_list[peer];
    probe->mte_ub = static_cast<uint64_t>(st->mte_config.aclshmem_ub);
    probe->mte_ub_size = st->mte_config.ub_size;
    probe->mte_sync_id = st->mte_config.sync_id;
    probe->sdma_ub = static_cast<uint64_t>(st->sdma_config.aclshmem_ub);
    probe->sdma_ub_size = st->sdma_config.ub_size;
    probe->sdma_sync_id = st->sdma_config.sync_id;
    D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(probe));
}

// 轻量 MTE UB ownership：仅等待最后一次 UB→GM（mte_put_nbi 故意留在外部）
__aicore__ inline void D1MteUbComplete(int peer, uint64_t &mte_waits)
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

// 解析正式/实验 transport 模式（MTE 走 P6，其它走 P4）
__aicore__ inline uint32_t D1ResolveTransportMode(uint32_t transport_mode, int peer)
{
    if (transport_mode != kD1PrimAutoRange) {
        return transport_mode;
    }
    __gm__ aclshmem_device_host_state_t *st = aclshmemi_get_state();
    return (st->topo_list[peer] & ACLSHMEM_TRANSPORT_MTE) ? kD1PrimP6RangeMteWaitFinalQuiet
                                                           : kD1PrimP4RangeQuiet;
}

// P6：range NBI + 每 range 轻量 MTE wait + lane 末一次 full quiet
__aicore__ inline void D1RunRangeP6(__gm__ uint8_t *sym, uint64_t source_off, uint64_t dest_off,
                                      uint32_t token_begin, uint32_t token_end, uint32_t nbytes,
                                      uint32_t batch_tokens, int peer, uint64_t &puts, uint64_t &quiets,
                                      uint64_t &mte_waits)
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
            D1MteUbComplete(peer, mte_waits);
        } else {
            // 非 MTE：退化为 P4 per-range full quiet
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

// range / per-token 共用：正式路径默认 auto（MTE→P6）
__aicore__ inline void D1RunPayloadTransport(__gm__ uint8_t *sym, uint64_t source_off, uint64_t dest_off,
                                             uint32_t token_begin, uint32_t token_end, uint32_t nbytes,
                                             uint32_t transport_mode, uint32_t batch_tokens, int peer,
                                             uint64_t &puts, uint64_t &quiets, uint64_t &mte_waits)
{
    const uint32_t mode = D1ResolveTransportMode(transport_mode, peer);
    const uint32_t batch = batch_tokens > 0 ? batch_tokens : kD1QuietBatch;
    if (mode == kD1PrimP0PerTokenQuiet) {
        for (uint32_t t = token_begin; t < token_end; ++t) {
            const uint64_t off = static_cast<uint64_t>(t) * nbytes;
            aclshmem_putmem_nbi(sym + dest_off + off, sym + source_off + off, nbytes, peer);
            ++puts;
            aclshmem_quiet();
            ++quiets;
        }
        return;
    }
    if (mode == kD1PrimP1NbiFinalQuiet) {
        for (uint32_t t = token_begin; t < token_end; ++t) {
            const uint64_t off = static_cast<uint64_t>(t) * nbytes;
            aclshmem_putmem_nbi(sym + dest_off + off, sym + source_off + off, nbytes, peer);
            ++puts;
        }
        aclshmem_quiet();
        ++quiets;
        return;
    }
    if (mode == kD1PrimP2BlockingPut) {
        for (uint32_t t = token_begin; t < token_end; ++t) {
            const uint64_t off = static_cast<uint64_t>(t) * nbytes;
            aclshmem_putmem(sym + dest_off + off, sym + source_off + off, nbytes, peer);
            ++puts;
            ++quiets;
        }
        return;
    }
    if (mode == kD1PrimP3NbiMteUbComplete) {
        for (uint32_t t = token_begin; t < token_end; ++t) {
            const uint64_t off = static_cast<uint64_t>(t) * nbytes;
            aclshmem_putmem_nbi(sym + dest_off + off, sym + source_off + off, nbytes, peer);
            ++puts;
            D1MteUbComplete(peer, mte_waits);
        }
        return;
    }
    if (mode == kD1PrimP6RangeMteWaitFinalQuiet) {
        D1RunRangeP6(sym, source_off, dest_off, token_begin, token_end, nbytes, batch, peer, puts, quiets,
                     mte_waits);
        return;
    }
    // P4/P5：连续 range put + 每 range full quiet
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

// 等待独立 start cacheline（单调 epoch）
__aicore__ inline bool D1PollSessionStop(__gm__ D1SessionStopLine *stop)
{
    D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(stop));
    return stop->value == kD1SessionStopMagic;
}

__aicore__ inline uint64_t D1WaitStartEpoch(__gm__ D1StartLine *start, __gm__ D1SessionStopLine *stop,
                                            uint64_t last_epoch, __gm__ D1Ctr *ctr, uint32_t err_stall)
{
    uint32_t spins = 0;
    while (true) {
        AscendC::PipeBarrier<PIPE_ALL>();
        if (D1PollSessionStop(stop)) {
            return 0;
        }
        D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(start));
        if (start->value > last_epoch) {
            return start->value;
        }
        // host_post verify 可能跨数十秒；persistent control 不得因 spin 上限永久退出
        if (++spins > 8000000u) {
            ctr->error_code = err_stall;
            spins = 0;
        }
    }
}

// 单 epoch 数据面：lane 分工 transport + lane0 汇总 publish
__aicore__ inline void D1WorkerExecuteEpoch(__gm__ uint8_t *sym, uint64_t source_off, uint64_t dest_off,
                                            uint64_t done_line_off, uint64_t worker_done_off,
                                            uint64_t publish_scratch_off, uint64_t run_epoch, uint32_t lane,
                                            uint32_t lane_count, __gm__ D1Desc *desc, __gm__ D1Ctr *ctr, int peer)
{
    if (lane == 0) {
        const uint64_t cyc = AscendC::GetSystemCycle();
        ctr->transport_start_cycle = cyc;
        desc->pair_start_cycle = cyc;
        desc->start_epoch = run_epoch;
        ctr->active_lanes = lane_count;
        D1RecordTransportProbe(peer, (__gm__ D1TransportProbe *)(sym + kD1TransportProbeOff));
        D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&desc->pair_start_cycle));
        D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&desc->start_epoch));
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    const uint32_t tokens = desc->tokens > 0 ? desc->tokens : 512u;
    const uint32_t nbytes = desc->payload_bytes > 0 ? desc->payload_bytes : 8192u;
    const uint32_t tokens_per_lane = tokens / lane_count;
    const uint32_t token_begin = lane * tokens_per_lane;
    const uint32_t token_end = token_begin + tokens_per_lane;
    __gm__ D1LaneCtr *lc = &ctr->lanes[lane];
    uint64_t puts = 0;
    uint64_t quiets = 0;
    uint64_t mte_waits = 0;
    const uint32_t tmode = desc->transport_mode > 0 ? desc->transport_mode : kD1PrimAutoRange;

    D1RunPayloadTransport(sym, source_off, dest_off, token_begin, token_end, nbytes, tmode, desc->quiet_batch, peer,
                          puts, quiets, mte_waits);
    lc->put_count = puts;
    lc->quiet_count = quiets;
    lc->mte_wait_count = mte_waits;
    lc->bytes_sent = static_cast<uint64_t>(token_end - token_begin) * nbytes;
    lc->lane_end_cycle = AscendC::GetSystemCycle();
    lc->done_epoch = run_epoch;
    D1KDcciLane(lc);

    if (lane == 0) {
        uint64_t agg_waits = 0;
        uint64_t max_end = 0;
        uint32_t all_ready = 0;
        while (!all_ready) {
            all_ready = 1;
            max_end = 0;
            for (uint32_t i = 0; i < lane_count; ++i) {
                D1KDcciLane(&ctr->lanes[i]);
                if (ctr->lanes[i].done_epoch != run_epoch) {
                    all_ready = 0;
                }
                if (ctr->lanes[i].lane_end_cycle > max_end) {
                    max_end = ctr->lanes[i].lane_end_cycle;
                }
            }
            if (!all_ready) {
                ++agg_waits;
                if (agg_waits > 8000000u) {
                    ctr->error_code = 3;
                    return;
                }
            }
        }
        ctr->aggregator_wait_cycles = agg_waits;
        ctr->transport_end_cycle = max_end;

        if (worker_done_off != 0u) {
            __gm__ D1WorkerDoneLine *worker_done = (__gm__ D1WorkerDoneLine *)(sym + worker_done_off);
            worker_done->value = run_epoch;
            D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(worker_done));
        }

        __gm__ D1PublishScratch *scratch = (__gm__ D1PublishScratch *)(sym + publish_scratch_off);
        scratch->epoch = run_epoch;
        D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(scratch));

        __gm__ D1DoneLine *remote_done = (__gm__ D1DoneLine *)(sym + done_line_off);
        aclshmem_putmem_nbi(reinterpret_cast<__gm__ void *>(&remote_done->value),
                            reinterpret_cast<__gm__ void *>(&scratch->epoch), sizeof(uint64_t), peer);
        aclshmem_quiet();
        ctr->done_publish_cycle = AscendC::GetSystemCycle();
        ctr->pair_end_cycle = ctr->done_publish_cycle;
        D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&ctr->done_publish_cycle));
        D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&ctr->pair_end_cycle));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void inc_dc_dn_ingress_transport_kernel(
    GM_ADDR sym, uint64_t desc_off, uint64_t ctr_off, uint64_t source_off, uint64_t dest_off,
    uint64_t done_line_off, uint64_t publish_scratch_off)
{
    if ASCEND_IS_AIV {
        const int bi = AscendC::GetBlockIdx();
        if (bi < 0) {
            return;
        }
        const uint32_t lane = static_cast<uint32_t>(bi);
        __gm__ D1Desc *desc = (__gm__ D1Desc *)(sym + desc_off);
        __gm__ D1Ctr *ctr = (__gm__ D1Ctr *)(sym + ctr_off);
        if (desc->magic != kD1Magic || desc->my_role != 0) {
            if (lane == 0) {
                ctr->error_code = 1;
            }
            return;
        }
        const uint32_t lane_count = desc->lane_count > 0 ? desc->lane_count : 1u;
        if (lane >= lane_count || lane >= kD1MaxLanes) {
            return;
        }
        const uint32_t tokens = desc->tokens > 0 ? desc->tokens : 512u;
        const uint32_t nbytes = desc->payload_bytes > 0 ? desc->payload_bytes : 8192u;
        const int peer = static_cast<int>(desc->peer_pe);

        uint32_t spins = 0;
        while (true) {
            AscendC::PipeBarrier<PIPE_ALL>();
            D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&desc->start_epoch));
            if (desc->start_epoch != 0) {
                break;
            }
            if (++spins > 8000000u) {
                ctr->error_code = 2;
                return;
            }
        }
        const uint64_t run_epoch = desc->start_epoch;

        D1WorkerExecuteEpoch(sym, source_off, dest_off, done_line_off, 0, publish_scratch_off, run_epoch, lane,
                             lane_count, desc, ctr, peer);
    }
}

// INC resident wait：仅轮询独立 done cacheline
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void inc_dc_dn_ingress_transport_inc_wait_kernel(
    GM_ADDR sym, uint64_t desc_off, uint64_t ctr_off, uint64_t done_line_off)
{
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        __gm__ D1Desc *desc = (__gm__ D1Desc *)(sym + desc_off);
        __gm__ D1Ctr *ctr = (__gm__ D1Ctr *)(sym + ctr_off);
        __gm__ D1DoneLine *done = (__gm__ D1DoneLine *)(sym + done_line_off);
        if (desc->magic != kD1Magic || desc->my_role != 1u) {
            ctr->error_code = 10;
            return;
        }
        uint32_t spins = 0;
        while (true) {
            AscendC::PipeBarrier<PIPE_ALL>();
            D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&desc->start_epoch));
            if (desc->start_epoch != 0) {
                break;
            }
            if (++spins > 8000000u) {
                ctr->error_code = 11;
                return;
            }
        }
        const uint64_t expect = desc->start_epoch;
        spins = 0;
        while (true) {
            AscendC::PipeBarrier<PIPE_ALL>();
            D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&done->value));
            if (done->value >= expect) {
                ctr->remote_done_seen_cycle = AscendC::GetSystemCycle();
                D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&ctr->remote_done_seen_cycle));
                break;
            }
            if (++spins > 8000000u) {
                ctr->error_code = 12;
                return;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// Persistent worker：service kernel 只 launch 一次，多 epoch 循环直至 session_stop
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void inc_dc_dn_ingress_transport_persistent_worker_kernel(
    GM_ADDR sym, uint64_t desc_off, uint64_t ctr_off, uint64_t source_off, uint64_t dest_off, uint64_t done_line_off,
    uint64_t worker_done_off, uint64_t start_line_off, uint64_t session_stop_off, uint64_t publish_scratch_off)
{
    if ASCEND_IS_AIV {
        const int bi = AscendC::GetBlockIdx();
        if (bi < 0) {
            return;
        }
        const uint32_t lane = static_cast<uint32_t>(bi);
        __gm__ D1Desc *desc = (__gm__ D1Desc *)(sym + desc_off);
        __gm__ D1Ctr *ctr = (__gm__ D1Ctr *)(sym + ctr_off);
        __gm__ D1StartLine *start = (__gm__ D1StartLine *)(sym + start_line_off);
        __gm__ D1SessionStopLine *stop = (__gm__ D1SessionStopLine *)(sym + session_stop_off);
        if (desc->magic != kD1Magic || desc->my_role != 0) {
            if (lane == 0) {
                ctr->error_code = 1;
            }
            return;
        }
        const uint32_t lane_count = desc->lane_count > 0 ? desc->lane_count : 1u;
        if (lane >= lane_count || lane >= kD1MaxLanes) {
            return;
        }
        const int peer = static_cast<int>(desc->peer_pe);
        uint64_t last_epoch = 0;
        while (true) {
            if (D1PollSessionStop(stop)) {
                break;
            }
            const uint64_t run_epoch = D1WaitStartEpoch(start, stop, last_epoch, ctr, 2);
            if (run_epoch == 0) {
                if (D1PollSessionStop(stop)) {
                    break;
                }
                continue;
            }
            D1WorkerExecuteEpoch(sym, source_off, dest_off, done_line_off, worker_done_off, publish_scratch_off,
                                 run_epoch, lane, lane_count, desc, ctr, peer);
            if (ctr->error_code != 0) {
                return;
            }
            last_epoch = run_epoch;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// Persistent INC：每 epoch 等 start_line 与 peer done_line，并向 PE0 发布 pair_done
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void inc_dc_dn_ingress_transport_persistent_inc_kernel(
    GM_ADDR sym, uint64_t desc_off, uint64_t ctr_off, uint64_t done_line_off, uint64_t start_line_off,
    uint64_t session_stop_off, uint64_t leader_completion_off, uint64_t publish_scratch_off, uint32_t leader_pe)
{
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        __gm__ D1Desc *desc = (__gm__ D1Desc *)(sym + desc_off);
        __gm__ D1Ctr *ctr = (__gm__ D1Ctr *)(sym + ctr_off);
        __gm__ D1DoneLine *done = (__gm__ D1DoneLine *)(sym + done_line_off);
        __gm__ D1StartLine *start = (__gm__ D1StartLine *)(sym + start_line_off);
        __gm__ D1SessionStopLine *stop = (__gm__ D1SessionStopLine *)(sym + session_stop_off);
        __gm__ D1PublishScratch *scratch = (__gm__ D1PublishScratch *)(sym + publish_scratch_off);
        if (desc->magic != kD1Magic || desc->my_role != 1u) {
            ctr->error_code = 10;
            return;
        }
        const uint32_t pair_id = desc->pair_id;
        const uint64_t remote_pair_done_off =
            leader_completion_off + offsetof(D1LeaderCompletion, pair_done_slots) +
            static_cast<uint64_t>(pair_id) * sizeof(D1ControlLine);
        uint64_t last_epoch = 0;
        while (true) {
            if (D1PollSessionStop(stop)) {
                break;
            }
            const uint64_t expect = D1WaitStartEpoch(start, stop, last_epoch, ctr, 11);
            if (expect == 0) {
                if (D1PollSessionStop(stop)) {
                    break;
                }
                continue;
            }
            desc->start_epoch = expect;
            desc->pair_start_cycle = AscendC::GetSystemCycle();
            D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&desc->start_epoch));
            D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&desc->pair_start_cycle));

            uint32_t spins = 0;
            while (true) {
                AscendC::PipeBarrier<PIPE_ALL>();
                if (D1PollSessionStop(stop)) {
                    return;
                }
                D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&done->value));
                if (done->value >= expect) {
                    ctr->remote_done_seen_cycle = AscendC::GetSystemCycle();
                    ctr->pair_end_cycle = ctr->remote_done_seen_cycle;
                    D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&ctr->remote_done_seen_cycle));
                    D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&ctr->pair_end_cycle));
                    break;
                }
                // host_post verify 可能很长；persistent INC 须跨 epoch 存活
                if (++spins > 8000000u) {
                    ctr->error_code = 12;
                    spins = 0;
                }
            }
            // 接收端观察完成 → 向 PE0 发布 pair_done（证明 INC 已看到该 epoch）
            scratch->epoch = expect;
            D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(scratch));
            const int my_pe = aclshmem_my_pe();
            if (my_pe == static_cast<int>(leader_pe)) {
                __gm__ D1ControlLine *slot = (__gm__ D1ControlLine *)(sym + remote_pair_done_off);
                slot->value = expect;
                D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(slot));
            } else {
                aclshmem_putmem_nbi(reinterpret_cast<__gm__ void *>(sym + remote_pair_done_off),
                                    reinterpret_cast<__gm__ void *>(&scratch->epoch), sizeof(uint64_t),
                                    static_cast<int>(leader_pe));
                aclshmem_quiet();
            }
            last_epoch = expect;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// Worker control AIV：读 local_ready，向 PE0 leader 发布 ready_epoch（64B doorbell put）
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void inc_dc_dn_d1_worker_ready_control_kernel(
    GM_ADDR sym, uint64_t local_ready_off, uint64_t leader_rendezvous_off, uint64_t publish_scratch_off,
    uint64_t session_stop_off, uint32_t pair_id, uint32_t leader_pe)
{
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        __gm__ D1LocalReadyLine *local_ready = (__gm__ D1LocalReadyLine *)(sym + local_ready_off);
        __gm__ D1SessionStopLine *stop = (__gm__ D1SessionStopLine *)(sym + session_stop_off);
        __gm__ D1PublishScratch *scratch = (__gm__ D1PublishScratch *)(sym + publish_scratch_off);
        const uint64_t remote_ready_off =
            leader_rendezvous_off + offsetof(D1LeaderRendezvous, ready_slots) +
            static_cast<uint64_t>(pair_id) * sizeof(D1ControlLine);
        uint64_t last_ready = 0;
        while (true) {
            if (D1PollSessionStop(stop)) {
                break;
            }
            uint32_t spins = 0;
            uint64_t ready_epoch = 0;
            while (true) {
                AscendC::PipeBarrier<PIPE_ALL>();
                if (D1PollSessionStop(stop)) {
                    return;
                }
                D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(local_ready));
                if (local_ready->value > last_ready) {
                    ready_epoch = local_ready->value;
                    break;
                }
                // host_post verify 可能很长；worker ready control 须跨 epoch 存活
                if (++spins > 8000000u) {
                    spins = 0;
                }
            }
            scratch->epoch = ready_epoch;
            D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(scratch));
            const int my_pe = aclshmem_my_pe();
            if (my_pe == static_cast<int>(leader_pe)) {
                __gm__ D1ControlLine *slot = (__gm__ D1ControlLine *)(sym + remote_ready_off);
                slot->value = ready_epoch;
                D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(slot));
            } else {
                aclshmem_putmem_nbi(reinterpret_cast<__gm__ void *>(sym + remote_ready_off),
                                    reinterpret_cast<__gm__ void *>(&scratch->epoch), sizeof(uint64_t),
                                    static_cast<int>(leader_pe));
                aclshmem_quiet();
            }
            last_ready = ready_epoch;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// PE0 leader control AIV：rendezvous → fanout go → 等 8 pair_done → 发布 global_done
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void inc_dc_dn_d1_leader_release_control_kernel(
    GM_ADDR sym, uint64_t leader_rendezvous_off, uint64_t leader_timing_off, uint64_t leader_completion_off,
    uint64_t global_done_line_off, uint64_t global_done_timing_off, uint64_t pair_done_arrival_off,
    uint64_t go_line_off, uint64_t publish_scratch_off, uint64_t session_stop_off, uint32_t worker_count,
    uint32_t inc_pe_base)
{
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        __gm__ D1LeaderRendezvous *leader = (__gm__ D1LeaderRendezvous *)(sym + leader_rendezvous_off);
        __gm__ D1LeaderTiming *timing = (__gm__ D1LeaderTiming *)(sym + leader_timing_off);
        __gm__ D1LeaderCompletion *completion = (__gm__ D1LeaderCompletion *)(sym + leader_completion_off);
        __gm__ D1GlobalDoneLine *global_done = (__gm__ D1GlobalDoneLine *)(sym + global_done_line_off);
        __gm__ D1GlobalDoneTiming *global_timing = (__gm__ D1GlobalDoneTiming *)(sym + global_done_timing_off);
        __gm__ D1PairDoneArrival *arrival = (__gm__ D1PairDoneArrival *)(sym + pair_done_arrival_off);
        __gm__ D1SessionStopLine *stop = (__gm__ D1SessionStopLine *)(sym + session_stop_off);
        __gm__ D1PublishScratch *scratch = (__gm__ D1PublishScratch *)(sym + publish_scratch_off);
        timing->worker_count = worker_count;
        global_timing->worker_count = worker_count;
        uint64_t last_release = 0;
        while (true) {
            if (D1PollSessionStop(stop)) {
                break;
            }
            uint32_t spins = 0;
            uint64_t expect = 0;
            while (true) {
                AscendC::PipeBarrier<PIPE_ALL>();
                if (D1PollSessionStop(stop)) {
                    return;
                }
                D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&leader->release_command));
                if (leader->release_command.value > last_release) {
                    expect = leader->release_command.value;
                    break;
                }
                // host_post verify 可能很长；leader release control 须跨 epoch 存活
                if (++spins > 8000000u) {
                    timing->error_code = 40;
                    spins = 0;
                }
            }
            timing->release_seen_cycle = AscendC::GetSystemCycle();
            spins = 0;
            while (true) {
                uint32_t ready = 0;
                for (uint32_t i = 0; i < worker_count; ++i) {
                    D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&leader->ready_slots[i]));
                    if (leader->ready_slots[i].value >= expect) {
                        ++ready;
                    }
                }
                if (ready >= worker_count) {
                    timing->rendezvous_complete_cycle = AscendC::GetSystemCycle();
                    break;
                }
                if (++spins > 8000000u) {
                    timing->error_code = 41;
                    spins = 0;
                }
            }
            scratch->epoch = expect;
            D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(scratch));
            timing->go_fanout_start_cycle = AscendC::GetSystemCycle();
            const int leader_pe = aclshmem_my_pe();
            for (uint32_t w = 0; w < worker_count; ++w) {
                const int worker_pe = static_cast<int>(w);
                const int inc_pe = static_cast<int>(inc_pe_base + w);
                if (worker_pe == leader_pe) {
                    __gm__ D1ControlLine *go = (__gm__ D1ControlLine *)(sym + go_line_off);
                    go->value = expect;
                    D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(go));
                } else {
                    aclshmem_putmem_nbi(reinterpret_cast<__gm__ void *>(sym + go_line_off),
                                        reinterpret_cast<__gm__ void *>(&scratch->epoch), sizeof(uint64_t), worker_pe);
                }
                if (inc_pe == leader_pe) {
                    __gm__ D1ControlLine *inc_go = (__gm__ D1ControlLine *)(sym + go_line_off);
                    inc_go->value = expect;
                    D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(inc_go));
                } else {
                    aclshmem_putmem_nbi(reinterpret_cast<__gm__ void *>(sym + go_line_off),
                                        reinterpret_cast<__gm__ void *>(&scratch->epoch), sizeof(uint64_t), inc_pe);
                }
            }
            aclshmem_quiet();
            timing->go_fanout_end_cycle = AscendC::GetSystemCycle();
            timing->last_go_epoch = expect;
            D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(timing));

            // go fanout 后等待全部 INC pair_done，记录各 slot 首次到达 cycle
            global_timing->pair_done_wait_start_cycle = AscendC::GetSystemCycle();
            for (uint32_t i = 0; i < worker_count; ++i) {
                arrival->lanes[i].pair_done_first_seen_cycle = 0;
            }
            D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(arrival));
            spins = 0;
            while (true) {
                uint32_t done_pairs = 0;
                for (uint32_t i = 0; i < worker_count; ++i) {
                    D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&completion->pair_done_slots[i]));
                    if (completion->pair_done_slots[i].value >= expect) {
                        if (arrival->lanes[i].pair_done_first_seen_cycle == 0) {
                            arrival->lanes[i].pair_done_first_seen_cycle = AscendC::GetSystemCycle();
                            D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&arrival->lanes[i]));
                        }
                        ++done_pairs;
                    }
                }
                if (done_pairs >= worker_count) {
                    global_timing->pair_done_complete_cycle = AscendC::GetSystemCycle();
                    break;
                }
                if (++spins > 8000000u) {
                    global_timing->error_code = 42;
                    spins = 0;
                }
            }
            global_done->value = expect;
            D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(global_done));
            global_timing->global_done_publish_cycle = AscendC::GetSystemCycle();
            global_timing->last_global_epoch = expect;
            D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(global_timing));
            last_release = expect;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// completion stream：短生命周期 wait kernel，轮询 done_line / worker_done_line
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void inc_dc_dn_done_observe_kernel(GM_ADDR sym,
                                                                                                    uint64_t observe_off,
                                                                                                    uint64_t expect_epoch,
                                                                                                    uint64_t ctr_off)
{
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        __gm__ D1ControlLine *line = (__gm__ D1ControlLine *)(sym + observe_off);
        __gm__ D1Ctr *ctr = (__gm__ D1Ctr *)(sym + ctr_off);
        uint32_t spins = 0;
        while (true) {
            AscendC::PipeBarrier<PIPE_ALL>();
            D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(line));
            if (line->value >= expect_epoch) {
                ctr->remote_done_seen_cycle = AscendC::GetSystemCycle();
                D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&ctr->remote_done_seen_cycle));
                break;
            }
            if (++spins > 8000000u) {
                ctr->error_code = 30;
                return;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// device verify：DCCI 全 dest 后逐字节比对
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void inc_dc_dn_ingress_transport_verify_kernel(
    GM_ADDR sym, uint64_t dest_off, uint64_t ctr_off, uint64_t desc_off)
{
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        __gm__ D1Desc *desc = (__gm__ D1Desc *)(sym + desc_off);
        __gm__ D1Ctr *ctr = (__gm__ D1Ctr *)(sym + ctr_off);
        const uint32_t pair = desc->pair_id;
        const uint32_t tokens = desc->tokens > 0 ? desc->tokens : 512u;
        const uint32_t nbytes = desc->payload_bytes > 0 ? desc->payload_bytes : 8192u;
        __gm__ uint8_t *base = (__gm__ uint8_t *)(sym + dest_off);
        for (uint64_t off = 0; off < static_cast<uint64_t>(tokens) * nbytes; off += 64u) {
            D1KDcciLine(base + off);
        }
        uint32_t bad = 0;
        for (uint32_t t = 0; t < tokens && bad == 0u; ++t) {
            for (uint32_t i = 0; i < nbytes; ++i) {
                const uint8_t got = base[static_cast<uint64_t>(t) * nbytes + i];
                const uint8_t exp =
                    static_cast<uint8_t>((pair * 31u + (t + 1u) * 17u + i) & 0xFFu);
                if (got != exp) {
                    bad = 1;
                    ctr->aggregator_wait_cycles = static_cast<uint64_t>(t) * nbytes + i;
                    ctr->pair_end_cycle = got;
                    ctr->transport_end_cycle = exp;
                    break;
                }
            }
        }
        ctr->error_code = bad;
        D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&ctr->error_code));
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// D1.0 done-only：仅测试 doorbell put
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void inc_dc_dn_d1_done_only_worker_kernel(
    GM_ADDR sym, uint64_t desc_off, uint64_t done_line_off, uint64_t publish_scratch_off, uint32_t cross_offset)
{
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        __gm__ D1Desc *desc = (__gm__ D1Desc *)(sym + desc_off);
        const int peer = static_cast<int>(desc->peer_pe);
        __gm__ D1PublishScratch *scratch = (__gm__ D1PublishScratch *)(sym + publish_scratch_off);
        scratch->epoch = 1;
        D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(scratch));
        // same-offset: remote done line；cross-offset: remote publish scratch（不同对称偏移）
        __gm__ uint8_t *remote = cross_offset
                                     ? reinterpret_cast<__gm__ uint8_t *>(scratch)
                                     : reinterpret_cast<__gm__ uint8_t *>(
                                           (__gm__ D1DoneLine *)(sym + done_line_off));
        aclshmem_putmem_nbi(remote, reinterpret_cast<__gm__ void *>(&scratch->epoch), sizeof(uint64_t), peer);
        aclshmem_quiet();
    }
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void inc_dc_dn_d1_done_only_inc_kernel(
    GM_ADDR sym, uint64_t done_line_off, uint64_t publish_scratch_off, uint64_t ctr_off, uint32_t cross_offset)
{
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        __gm__ D1Ctr *ctr = (__gm__ D1Ctr *)(sym + ctr_off);
        __gm__ uint64_t *epoch_ptr = cross_offset
                                         ? &((__gm__ D1PublishScratch *)(sym + publish_scratch_off))->epoch
                                         : &((__gm__ D1DoneLine *)(sym + done_line_off))->value;
        uint32_t spins = 0;
        while (true) {
            AscendC::PipeBarrier<PIPE_ALL>();
            D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(epoch_ptr));
            if (*epoch_ptr == 1u) {
                ctr->remote_done_seen_cycle = AscendC::GetSystemCycle();
                break;
            }
            if (++spins > 8000000u) {
                ctr->error_code = 20;
                return;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// M1：absolute local GM source → remote symmetric dest（source 可为 aclrtMalloc）
__aicore__ inline void D1RunRangeP6Abs(__gm__ uint8_t *src_base, __gm__ uint8_t *dst_base,
                                       uint32_t token_begin, uint32_t token_end, uint32_t nbytes,
                                       uint32_t batch_tokens, int peer, uint64_t &puts, uint64_t &quiets,
                                       uint64_t &mte_waits)
{
    __gm__ aclshmem_device_host_state_t *st = aclshmemi_get_state();
    const bool is_mte = (st->topo_list[peer] & ACLSHMEM_TRANSPORT_MTE) != 0u;
    const uint32_t batch = batch_tokens > 0 ? batch_tokens : kD1QuietBatch;
    for (uint32_t t = token_begin; t < token_end;) {
        const uint32_t ntok = (t + batch > token_end) ? (token_end - t) : batch;
        const uint64_t off = static_cast<uint64_t>(t) * nbytes;
        const uint32_t range_bytes = ntok * nbytes;
        aclshmem_putmem_nbi(dst_base + off, src_base + off, range_bytes, peer);
        ++puts;
        if (is_mte) {
            D1MteUbComplete(peer, mte_waits);
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

__aicore__ inline void D1RunPayloadTransportAbs(__gm__ uint8_t *src_base, __gm__ uint8_t *dst_base,
                                                uint32_t token_begin, uint32_t token_end, uint32_t nbytes,
                                                uint32_t transport_mode, uint32_t batch_tokens, int peer,
                                                uint64_t &puts, uint64_t &quiets, uint64_t &mte_waits)
{
    const uint32_t mode = D1ResolveTransportMode(transport_mode, peer);
    const uint32_t batch = batch_tokens > 0 ? batch_tokens : kD1QuietBatch;
    if (mode == kD1PrimP0PerTokenQuiet) {
        for (uint32_t t = token_begin; t < token_end; ++t) {
            const uint64_t off = static_cast<uint64_t>(t) * nbytes;
            aclshmem_putmem_nbi(dst_base + off, src_base + off, nbytes, peer);
            ++puts;
            aclshmem_quiet();
            ++quiets;
        }
        return;
    }
    if (mode == kD1PrimP6RangeMteWaitFinalQuiet || mode == kD1PrimAutoRange) {
        D1RunRangeP6Abs(src_base, dst_base, token_begin, token_end, nbytes, batch, peer, puts, quiets, mte_waits);
        return;
    }
    // default: range NBI + per-range quiet (P4-like)
    for (uint32_t t = token_begin; t < token_end;) {
        const uint32_t ntok = (t + batch > token_end) ? (token_end - t) : batch;
        const uint64_t off = static_cast<uint64_t>(t) * nbytes;
        const uint32_t range_bytes = ntok * nbytes;
        aclshmem_putmem_nbi(dst_base + off, src_base + off, range_bytes, peer);
        ++puts;
        aclshmem_quiet();
        ++quiets;
        t += ntok;
    }
}

// M1 probe worker：src_abs!=0 用绝对 GM；否则用 sym+source_off。lane_count 逻辑扇出到 dest lanes。
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void inc_dc_dn_m1_local_put_worker_kernel(
    GM_ADDR sym, uint64_t desc_off, uint64_t ctr_off, uint64_t source_off, uint64_t dest_off, uint64_t probe_off,
    uint64_t src_abs, uint64_t done_line_off)
{
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        __gm__ D1Desc *desc = (__gm__ D1Desc *)(sym + desc_off);
        __gm__ D1Ctr *ctr = (__gm__ D1Ctr *)(sym + ctr_off);
        const uint32_t tokens = desc->tokens;
        const uint32_t nbytes = desc->payload_bytes;
        const uint32_t lanes = desc->lane_count > 0 ? desc->lane_count : 1u;
        const int peer = static_cast<int>(desc->peer_pe);
        const uint32_t tmode = desc->transport_mode;

        D1RecordTransportProbe(peer, (__gm__ D1TransportProbe *)(sym + probe_off));
        ctr->transport_start_cycle = AscendC::GetSystemCycle();

        __gm__ uint8_t *src_base =
            (src_abs != 0ull) ? reinterpret_cast<__gm__ uint8_t *>(src_abs) : (sym + source_off);
        const uint64_t lane_stride = static_cast<uint64_t>(tokens) * nbytes;

        uint64_t puts = 0;
        uint64_t quiets = 0;
        uint64_t mte_waits = 0;
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            __gm__ uint8_t *lane_src = src_base + static_cast<uint64_t>(lane) * lane_stride;
            __gm__ uint8_t *lane_dst = sym + dest_off + static_cast<uint64_t>(lane) * lane_stride;
            D1RunPayloadTransportAbs(lane_src, lane_dst, 0, tokens, nbytes, tmode, desc->quiet_batch, peer, puts,
                                     quiets, mte_waits);
        }

        // independent done line on remote PE (symmetric) — match D1WorkerExecuteEpoch
        if (done_line_off != 0ull) {
            __gm__ D1PublishScratch *scratch = (__gm__ D1PublishScratch *)(sym + kD1PublishScratchOff);
            scratch->epoch = 1ull;
            D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(scratch));
            __gm__ D1DoneLine *remote_done = (__gm__ D1DoneLine *)(sym + done_line_off);
            aclshmem_putmem_nbi(reinterpret_cast<__gm__ void *>(&remote_done->value),
                                reinterpret_cast<__gm__ void *>(&scratch->epoch), sizeof(uint64_t), peer);
            aclshmem_quiet();
            ++quiets;
        }

        ctr->transport_end_cycle = AscendC::GetSystemCycle();
        ctr->aggregator_wait_cycles = mte_waits;
        __gm__ D1LaneCtr *lc = &ctr->lanes[0];
        lc->put_count = puts;
        lc->quiet_count = quiets;
        lc->mte_wait_count = mte_waits;
        lc->bytes_sent = static_cast<uint64_t>(tokens) * nbytes * lanes;
        D1KDcciLane(lc);
        D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&ctr->transport_end_cycle));
    }
}

// M1 verify：DCCI dest，支持 multi-lane；pair_id 编码 epoch pattern
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void inc_dc_dn_m1_probe_verify_kernel(
    GM_ADDR sym, uint64_t desc_off, uint64_t ctr_off, uint64_t dest_off, uint64_t done_line_off)
{
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        __gm__ D1Desc *desc = (__gm__ D1Desc *)(sym + desc_off);
        __gm__ D1Ctr *ctr = (__gm__ D1Ctr *)(sym + ctr_off);
        const uint32_t pair = desc->pair_id;
        const uint32_t tokens = desc->tokens;
        const uint32_t nbytes = desc->payload_bytes;
        const uint32_t lanes = desc->lane_count > 0 ? desc->lane_count : 1u;
        const uint64_t lane_stride = static_cast<uint64_t>(tokens) * nbytes;

        if (done_line_off != 0ull) {
            __gm__ D1DoneLine *done = (__gm__ D1DoneLine *)(sym + done_line_off);
            uint32_t spins = 0;
            while (true) {
                AscendC::PipeBarrier<PIPE_ALL>();
                D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(done));
                if (done->value == 1ull) {
                    break;
                }
                if (++spins > 8000000u) {
                    ctr->error_code = 21;
                    return;
                }
            }
        }

        uint32_t bad = 0;
        uint32_t verified = 0;
        for (uint32_t lane = 0; lane < lanes && bad == 0u; ++lane) {
            __gm__ uint8_t *dest = sym + dest_off + static_cast<uint64_t>(lane) * lane_stride;
            for (uint64_t off = 0; off < lane_stride; off += 64u) {
                D1KDcciLine(dest + off);
            }
            for (uint32_t t = 0; t < tokens && bad == 0u; ++t) {
                for (uint32_t i = 0; i < nbytes; ++i) {
                    const uint8_t got = dest[static_cast<uint64_t>(t) * nbytes + i];
                    // pair_id encodes epoch; lane mixes into pattern
                    const uint8_t exp = static_cast<uint8_t>(
                        ((pair + lane) * 31u + (t + 1u) * 17u + i) & 0xFFu);
                    ++verified;
                    if (got != exp) {
                        bad = 1;
                        ctr->aggregator_wait_cycles =
                            static_cast<uint64_t>(lane) * lane_stride + static_cast<uint64_t>(t) * nbytes + i;
                        ctr->pair_end_cycle = got;
                        ctr->transport_end_cycle = exp;
                        break;
                    }
                }
            }
        }
        ctr->error_code = bad;
        ctr->active_lanes = verified; // reuse: verify byte count proxy
        D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&ctr->error_code));
    }
}

// D1.1 payload-only / primitive matrix worker
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void inc_dc_dn_d1_payload_worker_kernel(
    GM_ADDR sym, uint64_t desc_off, uint64_t ctr_off, uint64_t source_off, uint64_t dest_off,
    uint64_t probe_off)
{
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        __gm__ D1Desc *desc = (__gm__ D1Desc *)(sym + desc_off);
        __gm__ D1Ctr *ctr = (__gm__ D1Ctr *)(sym + ctr_off);
        const uint32_t tokens = desc->tokens;
        const uint32_t nbytes = desc->payload_bytes;
        const int peer = static_cast<int>(desc->peer_pe);
        const uint32_t tmode = desc->transport_mode;

        D1RecordTransportProbe(peer, (__gm__ D1TransportProbe *)(sym + probe_off));
        const uint64_t t0 = AscendC::GetSystemCycle();
        ctr->transport_start_cycle = t0;

        uint64_t puts = 0;
        uint64_t quiets = 0;
        uint64_t mte_waits = 0;
        D1RunPayloadTransport(sym, source_off, dest_off, 0, tokens, nbytes, tmode, desc->quiet_batch, peer,
                              puts, quiets, mte_waits);

        ctr->transport_end_cycle = AscendC::GetSystemCycle();
        ctr->aggregator_wait_cycles = mte_waits;
        __gm__ D1LaneCtr *lc = &ctr->lanes[0];
        lc->put_count = puts;
        lc->quiet_count = quiets;
        lc->mte_wait_count = mte_waits;
        lc->bytes_sent = static_cast<uint64_t>(tokens) * nbytes;
        D1KDcciLane(lc);
        D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&ctr->transport_end_cycle));
    }
}

// D1.1/D1.2 device probe + verify
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void inc_dc_dn_d1_probe_verify_kernel(
    GM_ADDR sym, uint64_t desc_off, uint64_t ctr_off, uint64_t source_off, uint64_t dest_off,
    uint64_t probe_off, uint32_t probe_len)
{
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        __gm__ D1Desc *desc = (__gm__ D1Desc *)(sym + desc_off);
        __gm__ D1Ctr *ctr = (__gm__ D1Ctr *)(sym + ctr_off);
        const uint32_t pair = desc->pair_id;
        const uint32_t tokens = desc->tokens;
        const uint32_t nbytes = desc->payload_bytes;
        __gm__ uint8_t *dest = (__gm__ uint8_t *)(sym + dest_off);
        __gm__ uint8_t *src = (__gm__ uint8_t *)(sym + source_off);
        __gm__ uint8_t *probe = (__gm__ uint8_t *)(sym + probe_off);

        for (uint64_t off = 0; off < static_cast<uint64_t>(tokens) * nbytes; off += 64u) {
            D1KDcciLine(dest + off);
        }
        D1KDcciLine(src);
        const uint32_t n = probe_len > 64u ? 64u : probe_len;
        for (uint32_t i = 0; i < n; ++i) {
            probe[i] = dest[i];
        }
        for (uint32_t i = 0; i < n; ++i) {
            probe[64 + i] = src[i];
        }
        D1KDcciLine(probe);

        uint32_t bad = 0;
        for (uint32_t t = 0; t < tokens && bad == 0u; ++t) {
            for (uint32_t i = 0; i < nbytes; ++i) {
                const uint8_t got = dest[static_cast<uint64_t>(t) * nbytes + i];
                const uint8_t exp =
                    static_cast<uint8_t>((pair * 31u + (t + 1u) * 17u + i) & 0xFFu);
                if (got != exp) {
                    bad = 1;
                    ctr->aggregator_wait_cycles =
                        static_cast<uint64_t>(t) * nbytes + i; // 复用字段记 first_bad_offset
                    ctr->pair_end_cycle = got;
                    ctr->transport_end_cycle = exp;
                    break;
                }
            }
        }
        ctr->error_code = bad;
        D1KDcciLine(reinterpret_cast<__gm__ uint8_t *>(&ctr->error_code));
    }
}

extern "C" void launch_inc_dc_dn_ingress_transport_kernel(uint8_t *sym, uint64_t desc_off, uint64_t ctr_off,
                                                          uint64_t source_off, uint64_t dest_off,
                                                          uint64_t done_line_off, uint64_t publish_scratch_off,
                                                          int block_dim, void *stream)
{
    inc_dc_dn_ingress_transport_kernel<<<block_dim, nullptr, stream>>>(
        sym, desc_off, ctr_off, source_off, dest_off, done_line_off, publish_scratch_off);
}

extern "C" void launch_inc_dc_dn_ingress_transport_inc_wait_kernel(uint8_t *sym, uint64_t desc_off, uint64_t ctr_off,
                                                                 uint64_t done_line_off, void *stream)
{
    inc_dc_dn_ingress_transport_inc_wait_kernel<<<1, nullptr, stream>>>(sym, desc_off, ctr_off, done_line_off);
}

extern "C" void launch_inc_dc_dn_ingress_transport_verify_kernel(uint8_t *sym, uint64_t dest_off, uint64_t ctr_off,
                                                                 uint64_t desc_off, void *stream)
{
    inc_dc_dn_ingress_transport_verify_kernel<<<1, nullptr, stream>>>(sym, dest_off, ctr_off, desc_off);
}

extern "C" void launch_inc_dc_dn_ingress_transport_persistent_worker_kernel(
    uint8_t *sym, uint64_t desc_off, uint64_t ctr_off, uint64_t source_off, uint64_t dest_off, uint64_t done_line_off,
    uint64_t worker_done_off, uint64_t start_line_off, uint64_t session_stop_off, uint64_t publish_scratch_off,
    int block_dim, void *stream)
{
    inc_dc_dn_ingress_transport_persistent_worker_kernel<<<block_dim, nullptr, stream>>>(
        sym, desc_off, ctr_off, source_off, dest_off, done_line_off, worker_done_off, start_line_off,
        session_stop_off, publish_scratch_off);
}

extern "C" void launch_inc_dc_dn_ingress_transport_persistent_inc_kernel(uint8_t *sym, uint64_t desc_off,
                                                                         uint64_t ctr_off, uint64_t done_line_off,
                                                                         uint64_t start_line_off,
                                                                         uint64_t session_stop_off,
                                                                         uint64_t leader_completion_off,
                                                                         uint64_t publish_scratch_off,
                                                                         uint32_t leader_pe, void *stream)
{
    inc_dc_dn_ingress_transport_persistent_inc_kernel<<<1, nullptr, stream>>>(
        sym, desc_off, ctr_off, done_line_off, start_line_off, session_stop_off, leader_completion_off,
        publish_scratch_off, leader_pe);
}

extern "C" void launch_inc_dc_dn_done_observe_kernel(uint8_t *sym, uint64_t observe_off, uint64_t expect_epoch,
                                                     uint64_t ctr_off, void *stream)
{
    inc_dc_dn_done_observe_kernel<<<1, nullptr, stream>>>(sym, observe_off, expect_epoch, ctr_off);
}

extern "C" void launch_inc_dc_dn_d1_worker_ready_control_kernel(uint8_t *sym, uint64_t local_ready_off,
                                                              uint64_t leader_rendezvous_off,
                                                              uint64_t publish_scratch_off, uint64_t session_stop_off,
                                                              uint32_t pair_id, uint32_t leader_pe, void *stream)
{
    inc_dc_dn_d1_worker_ready_control_kernel<<<1, nullptr, stream>>>(
        sym, local_ready_off, leader_rendezvous_off, publish_scratch_off, session_stop_off, pair_id, leader_pe);
}

extern "C" void launch_inc_dc_dn_d1_leader_release_control_kernel(uint8_t *sym, uint64_t leader_rendezvous_off,
                                                                uint64_t leader_timing_off,
                                                                uint64_t leader_completion_off,
                                                                uint64_t global_done_line_off,
                                                                uint64_t global_done_timing_off,
                                                                uint64_t pair_done_arrival_off, uint64_t go_line_off,
                                                                uint64_t publish_scratch_off, uint64_t session_stop_off,
                                                                uint32_t worker_count, uint32_t inc_pe_base,
                                                                void *stream)
{
    inc_dc_dn_d1_leader_release_control_kernel<<<1, nullptr, stream>>>(
        sym, leader_rendezvous_off, leader_timing_off, leader_completion_off, global_done_line_off,
        global_done_timing_off, pair_done_arrival_off, go_line_off, publish_scratch_off, session_stop_off,
        worker_count, inc_pe_base);
}

extern "C" void launch_inc_dc_dn_d1_done_only_worker_kernel(uint8_t *sym, uint64_t desc_off, uint64_t done_line_off,
                                                            uint64_t publish_scratch_off, uint32_t cross_offset,
                                                            void *stream)
{
    inc_dc_dn_d1_done_only_worker_kernel<<<1, nullptr, stream>>>(sym, desc_off, done_line_off, publish_scratch_off,
                                                                 cross_offset);
}

extern "C" void launch_inc_dc_dn_d1_done_only_inc_kernel(uint8_t *sym, uint64_t done_line_off,
                                                          uint64_t publish_scratch_off, uint64_t ctr_off,
                                                          uint32_t cross_offset, void *stream)
{
    inc_dc_dn_d1_done_only_inc_kernel<<<1, nullptr, stream>>>(sym, done_line_off, publish_scratch_off, ctr_off,
                                                                 cross_offset);
}

extern "C" void launch_inc_dc_dn_d1_payload_worker_kernel(uint8_t *sym, uint64_t desc_off, uint64_t ctr_off,
                                                          uint64_t source_off, uint64_t dest_off, uint64_t probe_off,
                                                          void *stream)
{
    inc_dc_dn_d1_payload_worker_kernel<<<1, nullptr, stream>>>(sym, desc_off, ctr_off, source_off, dest_off,
                                                                 probe_off);
}

extern "C" void launch_inc_dc_dn_d1_probe_verify_kernel(uint8_t *sym, uint64_t desc_off, uint64_t ctr_off,
                                                          uint64_t source_off, uint64_t dest_off, uint64_t probe_off,
                                                          uint32_t probe_len, void *stream)
{
    inc_dc_dn_d1_probe_verify_kernel<<<1, nullptr, stream>>>(sym, desc_off, ctr_off, source_off, dest_off, probe_off,
                                                             probe_len);
}

extern "C" void launch_inc_dc_dn_m1_local_put_worker_kernel(uint8_t *sym, uint64_t desc_off, uint64_t ctr_off,
                                                            uint64_t source_off, uint64_t dest_off, uint64_t probe_off,
                                                            uint64_t src_abs, uint64_t done_line_off, void *stream)
{
    inc_dc_dn_m1_local_put_worker_kernel<<<1, nullptr, stream>>>(sym, desc_off, ctr_off, source_off, dest_off,
                                                                 probe_off, src_abs, done_line_off);
}

extern "C" void launch_inc_dc_dn_m1_probe_verify_kernel(uint8_t *sym, uint64_t desc_off, uint64_t ctr_off,
                                                        uint64_t dest_off, uint64_t done_line_off, void *stream)
{
    inc_dc_dn_m1_probe_verify_kernel<<<1, nullptr, stream>>>(sym, desc_off, ctr_off, dest_off, done_line_off);
}
