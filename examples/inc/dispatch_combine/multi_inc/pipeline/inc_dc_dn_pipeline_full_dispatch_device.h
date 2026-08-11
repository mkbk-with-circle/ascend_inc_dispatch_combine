/**
 * Full Dispatch device 原语：count/prefix + route gather（C1/C2/C3）
 * C3：Worker→partner INC→destination worker count 交换；canonical prefix；topk 双循环
 */
#pragma once

#include "inc_dc_dn_transport_abi.h"
#include "inc_dc_dn_pipeline_device.h"

namespace inc::dc::dn::pl {

constexpr uint32_t kPlFullDispatchMagic = 0x504C4644u; // PLFD

// Common MoE layout specialization.  Eight local experts is the production
// fast path, while every other valid runtime layout retains exact generic
// division semantics.
__aicore__ inline void PlSplitExpertId(uint32_t eid, uint32_t expert_per_pe,
                                      uint32_t &destination,
                                      uint32_t &local_expert)
{
    if (expert_per_pe == 8u) {
        destination = eid >> 3u;
        local_expert = eid & 7u;
        return;
    }
    destination = eid / expert_per_pe;
    local_expert = eid % expert_per_pe;
}

__aicore__ inline uint64_t PlDevSegmentBaseIndex(uint32_t local_expert, uint32_t source_rank, uint32_t worker_count)
{
    return static_cast<uint64_t>(local_expert) * static_cast<uint64_t>(worker_count) +
           static_cast<uint64_t>(source_rank);
}

__aicore__ inline uint64_t PlDevRouteCountLineOff(uint32_t source_rank, uint32_t dest_rank, uint32_t local_expert)
{
    const uint64_t idx =
        static_cast<uint64_t>(source_rank) * static_cast<uint64_t>(kPlMaxSources) * 8u +
        static_cast<uint64_t>(dest_rank) * 8u + static_cast<uint64_t>(local_expert);
    return kPlRouteCountRegionOff + idx * sizeof(PlRouteCountLine);
}

__aicore__ inline uint64_t PlDevSourceCountReadyLineOff(uint32_t source_rank)
{
    return kPlSourceCountReadyOff + static_cast<uint64_t>(source_rank) * 64u;
}

__aicore__ inline uint64_t PlDevDestCountSliceReadyLineOff(uint32_t source_rank, uint32_t dest_rank)
{
    return kPlDestCountSliceReadyOff +
           (static_cast<uint64_t>(source_rank) * static_cast<uint64_t>(kPlMaxSources) +
            static_cast<uint64_t>(dest_rank)) *
               64u;
}

// 等待 GM doorbell：禁止每 spin 都 Dcci；超时按 GetSystemCycle（疏 Dcci 时空转极快，不能再用固定 spin 冒充墙钟）
__aicore__ inline void PlSpinWaitLineDcci(__gm__ uint8_t *line, uint32_t spins)
{
    constexpr uint32_t kDcciEvery = 256u;
    if ((spins & (kDcciEvery - 1u)) == 0u) {
        PlDcci(line);
    }
}

// putmem_signal doorbell：每 8 spin 刷 64B ready line（signal 与 payload 同线）
__aicore__ inline void PlSpinWaitSignalLineDcci(__gm__ uint8_t *line, uint32_t spins)
{
    constexpr uint32_t kDcciEvery = 8u;
    if ((spins & (kDcciEvery - 1u)) == 0u) {
        PlDcci(line);
    }
}

__aicore__ inline bool PlWaitBudgetExceeded(uint64_t t0_cycles, uint64_t timeout_us)
{
    // 统一用已校准常量；禁止为凑 S3 PASS 改该倍率
    constexpr uint64_t kCyclesPerUs = 50ull; // == kD1CyclesPerUs
    return (AscendC::GetSystemCycle() - t0_cycles) > (timeout_us * kCyclesPerUs);
}

__aicore__ inline uint64_t PlDevWorkerSourceCountTraceOff(uint32_t source_rank)
{
    return kPlWorkerSourceCountTraceOff + static_cast<uint64_t>(source_rank) * 64u;
}

__aicore__ inline uint64_t PlDevIncSourceCountTraceOff(uint32_t source_rank)
{
    return kPlIncSourceCountTraceOff + static_cast<uint64_t>(source_rank) * 64u;
}

__aicore__ inline uint64_t PlDevDestinationCountTraceOff(uint32_t dest_rank)
{
    return kPlDestinationCountTraceOff + static_cast<uint64_t>(dest_rank) * 64u;
}

__aicore__ inline uint64_t PlDevSegBasePublishStagingOff(uint32_t source_rank)
{
    return kPlSegBasePublishStagingOff + static_cast<uint64_t>(source_rank) * 64u;
}

__aicore__ inline uint64_t PlDevUploadLaneStageTimingOff(uint32_t lane_id)
{
    return kPlUploadLaneStageTimingOff + static_cast<uint64_t>(lane_id) * 128u;
}

__aicore__ inline uint64_t PlDevSegmentBaseReadyLineOff(uint32_t source_rank, uint32_t dest_rank)
{
    return kPlSegmentBaseReadyOff +
           (static_cast<uint64_t>(source_rank) * static_cast<uint64_t>(kPlMaxSources) +
            static_cast<uint64_t>(dest_rank)) *
               64u;
}

__aicore__ inline uint64_t PlDevOutboundSegBaseIndex(uint32_t dest_rank, uint32_t local_expert)
{
    return static_cast<uint64_t>(dest_rank) * 16u + static_cast<uint64_t>(local_expert);
}

__aicore__ inline uint64_t PlDevRecvDoneLineOff(uint32_t source_rank)
{
    return kPlRecvDoneOff + static_cast<uint64_t>(source_rank) * 64u;
}

__aicore__ inline uint64_t PlDevTransportDoneLineOff(uint32_t destination_rank)
{
    return kPlTransportDoneOff + static_cast<uint64_t>(destination_rank) * 64u;
}

// device 侧直接算偏移（禁止调用 host-only inline）
__aicore__ inline uint64_t PlDevRankDispatchTimingLineOff(uint32_t rank)
{
    return kPlRankDispatchTimingOff + static_cast<uint64_t>(rank) * 64u;
}

__aicore__ inline __gm__ PlRankDispatchTimingLine *PlRankDispatchTimingLinePtr(__gm__ uint8_t *sym, uint32_t rank)
{
    return reinterpret_cast<__gm__ PlRankDispatchTimingLine *>(sym + PlDevRankDispatchTimingLineOff(rank));
}

// P10 strict max-rank timing helpers（每 PE 写本 rank 槽）
__aicore__ inline void PlRankTimingMarkStart(__gm__ uint8_t *sym, uint32_t rank, uint32_t role_mask, uint64_t epoch,
                                              uint32_t generation)
{
    __gm__ PlRankDispatchTimingLine *line = PlRankDispatchTimingLinePtr(sym, rank);
    line->epoch = epoch;
    line->generation = generation;
    line->rank = static_cast<uint16_t>(rank);
    line->role_mask = static_cast<uint16_t>(role_mask);
    line->local_start_cycle = AscendC::GetSystemCycle();
    line->upload_done_cycle = 0u;
    line->forward_done_cycle = 0u;
    line->destination_done_cycle = 0u;
    line->local_completion_cycle = 0u;
    line->error_code = 0u;
    line->magic = 0u;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(line));
}

__aicore__ inline void PlRankTimingMarkUploadDone(__gm__ uint8_t *sym, uint32_t rank, uint64_t cycle)
{
    __gm__ PlRankDispatchTimingLine *line = PlRankDispatchTimingLinePtr(sym, rank);
    if (line->upload_done_cycle == 0u || cycle > line->upload_done_cycle) {
        line->upload_done_cycle = cycle;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(line));
    }
}

__aicore__ inline void PlRankTimingMarkForwardDone(__gm__ uint8_t *sym, uint32_t rank, uint64_t cycle)
{
    __gm__ PlRankDispatchTimingLine *line = PlRankDispatchTimingLinePtr(sym, rank);
    if (line->forward_done_cycle == 0u || cycle > line->forward_done_cycle) {
        line->forward_done_cycle = cycle;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(line));
    }
}

__aicore__ inline void PlRankTimingMarkDestinationDone(__gm__ uint8_t *sym, uint32_t rank, uint64_t cycle)
{
    __gm__ PlRankDispatchTimingLine *line = PlRankDispatchTimingLinePtr(sym, rank);
    if (line->destination_done_cycle == 0u || cycle > line->destination_done_cycle) {
        line->destination_done_cycle = cycle;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(line));
    }
}

__aicore__ inline void PlRankTimingPublishLocalCompletion(__gm__ uint8_t *sym, uint32_t rank, uint32_t role_mask)
{
    __gm__ PlRankDispatchTimingLine *line = PlRankDispatchTimingLinePtr(sym, rank);
    uint64_t completion = 0u;
    if ((role_mask & kPlRoleMaskInc) != 0u) {
        completion = line->forward_done_cycle;
    } else if ((role_mask & kPlRoleMaskWorker) != 0u) {
        const uint64_t upload = line->upload_done_cycle;
        const uint64_t dest = line->destination_done_cycle;
        completion = (upload > dest) ? upload : dest;
    } else {
        completion = line->destination_done_cycle;
    }
    if (completion > 0u && line->local_start_cycle > 0u && completion >= line->local_start_cycle) {
        line->local_completion_cycle = completion;
        line->magic = kPlRankDispatchTimingMagic;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(line));
        // 推到 PE0 槽，避免 host getmem 竞态漏采
        const int my_pe = aclshmem_my_pe();
        if (my_pe != 0) {
            const uint64_t off = PlDevRankDispatchTimingLineOff(rank);
            aclshmem_putmem_nbi(reinterpret_cast<__gm__ void *>(sym + off),
                                reinterpret_cast<__gm__ void *>(line), sizeof(PlRankDispatchTimingLine), 0);
            aclshmem_quiet();
        }
    }
}

__aicore__ inline void PlDevRankTimingMarkStart(__gm__ uint8_t *sym, uint32_t role_mask, uint64_t epoch,
                                                 uint32_t generation)
{
    const uint32_t rank = static_cast<uint32_t>(aclshmem_my_pe());
    PlRankTimingMarkStart(sym, rank, role_mask, epoch, generation);
}

__aicore__ inline void PlDevRankTimingMarkUploadDone(__gm__ uint8_t *sym, uint64_t cycle)
{
    PlRankTimingMarkUploadDone(sym, static_cast<uint32_t>(aclshmem_my_pe()), cycle);
}

__aicore__ inline void PlDevRankTimingMarkForwardDone(__gm__ uint8_t *sym, uint64_t cycle)
{
    PlRankTimingMarkForwardDone(sym, static_cast<uint32_t>(aclshmem_my_pe()), cycle);
}

__aicore__ inline void PlDevRankTimingMarkDestinationDone(__gm__ uint8_t *sym, uint64_t cycle)
{
    PlRankTimingMarkDestinationDone(sym, static_cast<uint32_t>(aclshmem_my_pe()), cycle);
}

__aicore__ inline void PlDevRankTimingPublishLocalCompletion(__gm__ uint8_t *sym, uint32_t role_mask)
{
    PlRankTimingPublishLocalCompletion(sym, static_cast<uint32_t>(aclshmem_my_pe()), role_mask);
}

__aicore__ inline uint64_t PlDevEgressVisibleDoneSlotOff(uint32_t inc_rank, uint32_t dest_rank)
{
    return kPlEgressVisibleDoneOff +
           (static_cast<uint64_t>(inc_rank) * static_cast<uint64_t>(kPlMaxSources) +
            static_cast<uint64_t>(dest_rank)) *
               64u;
}

// S4 单写者：forward lane 独占本地 (lane,dest) 行；禁止多 lane RMW summary cell
__aicore__ inline uint64_t PlDevEgressLaneLocalVisibleSlotOff(uint32_t lane_id, uint32_t dest_rank)
{
    return kPlEgressLaneLocalVisibleOff +
           (static_cast<uint64_t>(lane_id) * static_cast<uint64_t>(kPlMaxSources) +
            static_cast<uint64_t>(dest_rank)) *
               64u;
}

__aicore__ inline uint64_t PlDevIncSecondHopSummaryOff(uint32_t inc_rank)
{
    return kPlIncSecondHopSummaryOff + static_cast<uint64_t>(inc_rank) * 64u;
}

__aicore__ inline uint32_t PlEgressVisibleChecksum(uint32_t inc_rank, uint32_t dest_rank, uint32_t expected,
                                                    uint32_t forwarded, uint64_t epoch, uint32_t generation,
                                                    uint64_t tail_value)
{
    return inc_rank ^ (dest_rank << 8) ^ expected ^ (forwarded << 16) ^ static_cast<uint32_t>(epoch) ^
           (generation << 24) ^ static_cast<uint32_t>(tail_value) ^ static_cast<uint32_t>(tail_value >> 32) ^
           0xE6E55001u;
}

// 负例延迟（前向声明；定义见本文件后部）
__aicore__ inline void PlApplyTimingNegDelay(uint32_t delay_spins);

// INC egress lane：第二跳 put+quiet+tail 后 push 到 PE0 堆槽 summary cell（仅 block9 调用）
__aicore__ inline void PlPublishEgressVisibleDone(__gm__ uint8_t *sym, uint32_t inc_rank, uint32_t dest_rank,
                                                   uint64_t epoch, uint32_t generation, uint32_t expected_routes,
                                                   uint32_t forwarded_routes, uint64_t tail_value, uint32_t error_code,
                                                   uint64_t visible_done_cycle, int leader_pe)
{
    __gm__ PlTimingNegLine *neg = reinterpret_cast<__gm__ PlTimingNegLine *>(sym + kPlTimingNegOff);
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(neg));
    if (neg->magic == kPlTimingNegMagic && neg->drop_egress_cell != 0u) {
        return;
    }
    // delay_second_hop 已移到 forward 数据 put 前；此处只发遥测 cell
    const uint64_t slot_off = PlDevEgressVisibleDoneSlotOff(inc_rank, dest_rank);
    __gm__ PlEgressLaneVisibleDoneLine *line =
        reinterpret_cast<__gm__ PlEgressLaneVisibleDoneLine *>(sym + slot_off);
    line->epoch = epoch;
    line->generation = generation;
    line->inc_rank = inc_rank;
    line->destination_rank = dest_rank;
    line->expected_routes = expected_routes;
    line->forwarded_routes = forwarded_routes;
    line->tail_value = tail_value;
    // visible_done_cycle 存 lane 相对 delta（非绝对时钟）；允许 0
    line->visible_done_cycle = visible_done_cycle;
    line->error_code = error_code;
    line->checksum = PlEgressVisibleChecksum(inc_rank, dest_rank, expected_routes, forwarded_routes, epoch,
                                              generation, tail_value);
    line->magic = kPlEgressVisibleDoneMagic;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(line));
    const int my_pe = aclshmem_my_pe();
    if (my_pe != leader_pe) {
        aclshmem_putmem_nbi(reinterpret_cast<__gm__ void *>(sym + slot_off), reinterpret_cast<__gm__ void *>(line),
                            sizeof(PlEgressLaneVisibleDoneLine), leader_pe);
        aclshmem_quiet();
    }
}

// INC forward lane：逐 destination 发布 egress visible cell（含 expected=0）
__aicore__ inline void PlIncPublishEgressVisiblePerDest(__gm__ uint8_t *sym, uint32_t inc_rank, uint32_t worker_count,
                                                        uint64_t epoch, uint32_t generation, uint32_t dest_rank,
                                                        uint32_t expected_routes, uint32_t forwarded_routes,
                                                        uint64_t tail_value, uint64_t visible_done_cycle, int leader_pe)
{
    (void)worker_count;
    PlPublishEgressVisibleDone(sym, inc_rank, dest_rank, epoch, generation, expected_routes, forwarded_routes,
                               tail_value, 0u, visible_done_cycle, leader_pe);
}

// 前向声明：leader 域映射（定义在 PlResolvePe0ReleaseSeenCycle 之后）
__aicore__ inline uint64_t PlIncMapVisibleDoneToLeaderDomain(__gm__ uint8_t *sym, uint64_t epoch,
                                                             uint64_t lane_cycle_start, uint64_t local_done_cycle);

// S4 单写者：forward lane 只写本 lane 独占 (lane,dest) 行；禁止 put 到 PE0 summary
__aicore__ inline void PlIncLaneWriteEgressVisibleLocal(__gm__ uint8_t *sym, uint32_t lane_id, uint32_t dest_rank,
                                                         uint32_t inc_rank, uint64_t epoch, uint32_t generation,
                                                         uint32_t expected_routes, uint32_t forwarded_routes,
                                                         uint64_t tail_value, uint64_t visible_done_cycle)
{
    const uint64_t slot_off = PlDevEgressLaneLocalVisibleSlotOff(lane_id, dest_rank);
    __gm__ PlEgressLaneVisibleDoneLine *line =
        reinterpret_cast<__gm__ PlEgressLaneVisibleDoneLine *>(sym + slot_off);
    line->epoch = epoch;
    line->generation = generation;
    line->inc_rank = inc_rank;
    line->destination_rank = dest_rank;
    line->expected_routes = expected_routes;
    line->forwarded_routes = forwarded_routes;
    line->tail_value = tail_value;
    line->visible_done_cycle = visible_done_cycle; // lane 相对 delta
    line->error_code = 0u;
    line->checksum = PlEgressVisibleChecksum(inc_rank, dest_rank, expected_routes, forwarded_routes, epoch,
                                              generation, tail_value);
    line->magic = kPlEgressVisibleDoneMagic;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(line));
}

// block9 专用：从全部 lane 行聚合单 dest，校验后写 summary cell（唯一 writer）
__aicore__ inline bool PlIncBlock9AggregateDestFromLanes(__gm__ uint8_t *sym, uint32_t inc_rank, uint32_t dest_rank,
                                                          uint32_t worker_count, uint64_t epoch, uint32_t generation,
                                                          uint32_t *out_sum_fwd, uint64_t *out_max_cyc,
                                                          uint64_t *out_tail, uint32_t *out_exp)
{
    uint32_t sum_fwd = 0u;
    uint32_t exp = 0u;
    uint64_t max_cyc = 0u;
    uint64_t tail = 0u;
    bool saw_lane = false;
    for (uint32_t lane = 0; lane < worker_count; ++lane) {
        __gm__ PlEgressLaneVisibleDoneLine *lv = reinterpret_cast<__gm__ PlEgressLaneVisibleDoneLine *>(
            sym + PlDevEgressLaneLocalVisibleSlotOff(lane, dest_rank));
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(lv));
        if (lv->magic != kPlEgressVisibleDoneMagic || lv->epoch != epoch || lv->generation != generation ||
            lv->inc_rank != inc_rank || lv->destination_rank != dest_rank || lv->error_code != 0u) {
            continue;
        }
        if (lv->checksum != PlEgressVisibleChecksum(inc_rank, dest_rank, lv->expected_routes, lv->forwarded_routes,
                                                       epoch, generation, lv->tail_value)) {
            return false;
        }
        sum_fwd += lv->forwarded_routes;
        if (!saw_lane) {
            exp = lv->expected_routes;
            saw_lane = true;
        } else if (lv->expected_routes != exp) {
            return false;
        }
        if (lv->visible_done_cycle > max_cyc) {
            max_cyc = lv->visible_done_cycle;
        }
        if (lv->tail_value > tail) {
            tail = lv->tail_value;
        }
    }
    // expected==0：owned lane 在入口写 fwd=0；无流量 dest 允许无 lane 行
    if (!saw_lane && exp == 0u && sum_fwd == 0u) {
        *out_sum_fwd = 0u;
        *out_exp = 0u;
        *out_max_cyc = 0u;
        *out_tail = 0u;
        return true;
    }
    if (!saw_lane || sum_fwd != exp) {
        return false;
    }
    *out_sum_fwd = sum_fwd;
    *out_exp = exp;
    *out_max_cyc = max_cyc;
    *out_tail = tail;
    return true;
}

// 已废弃：多 lane RMW summary；保留供 ABI 探针，forward hot path 禁止调用
__aicore__ inline void PlIncMergePublishEgressVisibleDest(
    __gm__ uint8_t *sym, uint32_t inc_rank, uint32_t dest_rank, uint64_t epoch, uint32_t generation,
    uint32_t expected_routes, uint32_t forwarded_add, uint64_t tail_value,
    uint64_t lane_cycle_start, uint64_t visible_done_cycle, int leader_pe)
{
    const uint64_t slot_off = PlDevEgressVisibleDoneSlotOff(inc_rank, dest_rank);
    __gm__ PlEgressLaneVisibleDoneLine *line =
        reinterpret_cast<__gm__ PlEgressLaneVisibleDoneLine *>(sym + slot_off);
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(line));
    uint32_t fwd = forwarded_add;
    uint64_t cyc = PlIncMapVisibleDoneToLeaderDomain(sym, epoch, lane_cycle_start, visible_done_cycle);
    uint64_t tail = tail_value;
    if (line->magic == kPlEgressVisibleDoneMagic && line->epoch == epoch && line->generation == generation &&
        line->inc_rank == inc_rank && line->destination_rank == dest_rank) {
        fwd += line->forwarded_routes;
        if (line->visible_done_cycle > cyc) {
            cyc = line->visible_done_cycle;
        }
        if (tail == 0u) {
            tail = line->tail_value;
        }
    }
    PlPublishEgressVisibleDone(sym, inc_rank, dest_rank, epoch, generation, expected_routes, fwd, tail, 0u, cyc,
                               leader_pe);
}

// Forward lane：只写本 lane 独占行；block9 聚合 summary
__aicore__ inline void PlIncForwardLanePublishEgressVisible(
    __gm__ uint8_t *sym, uint32_t inc_rank, uint32_t lane_id, uint32_t worker_count, uint64_t epoch,
    uint32_t generation, const uint32_t *expected_per_dest, const uint32_t *forwarded_per_dest,
    const uint64_t *tail_per_dest, const uint64_t *visible_done_per_dest,
    uint64_t lane_cycle_start, int leader_pe)
{
    (void)leader_pe;
    for (uint32_t dest = 0; dest < worker_count; ++dest) {
        const uint32_t exp = expected_per_dest[dest];
        const uint32_t fwd_lane = forwarded_per_dest[dest];
        if (fwd_lane > 0u) {
            const uint64_t tail = tail_per_dest != nullptr ? tail_per_dest[dest] : 0u;
            uint64_t cyc = visible_done_per_dest != nullptr ? visible_done_per_dest[dest] : 0u;
            cyc = PlIncMapVisibleDoneToLeaderDomain(sym, epoch, lane_cycle_start, cyc);
            PlIncLaneWriteEgressVisibleLocal(sym, lane_id, dest, inc_rank, epoch, generation, exp, fwd_lane, tail,
                                             cyc);
        }
    }
}

// expected==0 的 owned dest：lane 入口写本 lane 行（禁止 RMW summary）
__aicore__ inline void PlIncForwardLanePublishOwnedZeroDestAtEntry(__gm__ uint8_t *sym, uint32_t inc_rank,
                                                                    uint32_t lane_id, uint32_t worker_count,
                                                                    uint64_t epoch, uint32_t generation,
                                                                    const uint32_t *expected_per_dest,
                                                                    uint64_t lane_cycle_start, uint64_t entry_cycle, int leader_pe)
{
    (void)leader_pe;
    if (lane_id >= worker_count) {
        return;
    }
    if (expected_per_dest[lane_id] != 0u) {
        return;
    }
    const uint64_t leader_cyc =
        PlIncMapVisibleDoneToLeaderDomain(sym, epoch, lane_cycle_start, entry_cycle);
    PlIncLaneWriteEgressVisibleLocal(sym, lane_id, lane_id, inc_rank, epoch, generation, 0u, 0u, 0u, leader_cyc);
}

// 历史/ABI 探针：禁止在 forward hot path 调用（会整表覆写）
__aicore__ inline void PlIncPublishEgressVisibleMatrix(__gm__ uint8_t *sym, uint32_t inc_rank, uint32_t worker_count,
                                                      uint64_t epoch, uint32_t generation,
                                                      const uint32_t *expected_per_dest,
                                                      const uint32_t *forwarded_per_dest,
                                                      const uint64_t *tail_per_dest, uint64_t visible_done_cycle,
                                                      int leader_pe)
{
    for (uint32_t dest = 0; dest < worker_count; ++dest) {
        const uint64_t tail = tail_per_dest != nullptr ? tail_per_dest[dest] : 0u;
        PlIncPublishEgressVisiblePerDest(sym, inc_rank, worker_count, epoch, generation, dest,
                                         expected_per_dest[dest], forwarded_per_dest[dest], tail, visible_done_cycle,
                                         leader_pe);
    }
}

// 从 route count matrix 汇总各 dest 的 expected_routes（source = inc_rank）
__aicore__ inline void PlIncComputeEgressExpectedPerDest(__gm__ uint8_t *sym, uint32_t inc_rank, uint32_t worker_count,
                                                          uint32_t expert_per_pe, uint64_t epoch,
                                                          uint32_t *expected_per_dest)
{
    const uint32_t epp = expert_per_pe == 0u ? 8u : expert_per_pe;
    for (uint32_t d = 0; d < kPlMaxSources; ++d) {
        expected_per_dest[d] = 0u;
    }
    for (uint32_t dest = 0; dest < worker_count; ++dest) {
        uint32_t sum = 0u;
        for (uint32_t le = 0; le < epp; ++le) {
            __gm__ PlRouteCountLine *line =
                reinterpret_cast<__gm__ PlRouteCountLine *>(sym + PlDevRouteCountLineOff(inc_rank, dest, le));
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(line));
            if (line->epoch == epoch && line->source_rank == inc_rank && line->destination_rank == dest) {
                sum += line->count;
            }
        }
        expected_per_dest[dest] = sum;
    }
}

// PE0 公共 release_seen：D1LeaderTiming 优先，0 时回退 PlTransportTimingLine（与 transport timing 同源）
__aicore__ inline uint64_t PlResolvePe0ReleaseSeenCycle(__gm__ uint8_t *sym, uint64_t epoch)
{
    __gm__ D1LeaderTiming *lt = reinterpret_cast<__gm__ D1LeaderTiming *>(sym + kPlD1RegionOff + kD1LeaderTimingOff);
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(lt));
    if (lt->release_seen_cycle > 0u) {
        return lt->release_seen_cycle;
    }
    __gm__ PlTransportTimingLine *tt =
        reinterpret_cast<__gm__ PlTransportTimingLine *>(sym + kPlTransportTimingOff);
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(tt));
    if (tt->magic == kPlTransportTimingMagic && tt->timing_epoch == epoch && tt->release_seen_cycle > 0u) {
        return tt->release_seen_cycle;
    }
    return 0u;
}

// INC 本地 → PE0 域：只存相对 lane_start 的 delta；PE0 再加 release（避免跨 NPU 绝对时钟混算）
__aicore__ inline uint64_t PlIncMapVisibleDoneToLeaderDomain(__gm__ uint8_t *sym, uint64_t epoch,
                                                             uint64_t lane_cycle_start, uint64_t local_done_cycle)
{
    (void)sym;
    (void)epoch;
    if (local_done_cycle == 0u) {
        local_done_cycle = AscendC::GetSystemCycle();
    }
    if (lane_cycle_start == 0u || local_done_cycle <= lane_cycle_start) {
        return 0u;
    }
    return local_done_cycle - lane_cycle_start;
}

// INC block9：等全部 forward lane 完成 → 聚合 lane 行 → 写 summary cell → push summary 到 PE0
__aicore__ inline bool PlIncSecondHopAggPublishSummary(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                        __gm__ D1SessionStopLine *stop,
                                                        __gm__ PlServiceTraceLine *trace, uint64_t epoch,
                                                        uint32_t generation, uint32_t block_id, int leader_pe)
{
    const uint32_t inc_rank = desc->pair_id;
    const uint32_t wc = desc->worker_count;
    uint32_t spins = 0u;
    while (spins < 8000000u) {
        if (stop != nullptr) {
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(stop));
            if (trace != nullptr) {
                trace->observed_stop_value = stop->value;
                PlTraceDcci(trace);
            }
            if (stop->value == inc::dc::dn::kD1SessionStopMagic) {
                return true;
            }
        }
        // workspace / generation 致命错误：提前退出，避免 8M spins 空等
        if (desc->measurement_mode == kPlMeasurementFullDispatch) {
            __gm__ PlFullDispatchConfig *fdc_err =
                reinterpret_cast<__gm__ PlFullDispatchConfig *>(sym + kPlFullDispatchConfigOff);
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(fdc_err));
            __gm__ PlFullOutputDoneLine *fout_err =
                reinterpret_cast<__gm__ PlFullOutputDoneLine *>(sym + fdc_err->full_output_done_off);
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(fout_err));
            if (fout_err->error_code == kPlErrGenerationMismatch ||
                fout_err->error_code == kPlErrWorkspaceDescMissing ||
                fout_err->error_code == kPlErrWorkspaceGenerationMismatch ||
                fout_err->error_code == kPlErrWorkspacePointerAlignment ||
                fout_err->error_code == kPlErrOutputCapacityInsufficient ||
                fout_err->error_code == kPlErrWorkspaceFlagInvalid) {
                return false;
            }
        }
        bool lanes_done = true;
        for (uint32_t lane = 0; lane < wc; ++lane) {
            __gm__ PlForwardDoneLine *fd =
                reinterpret_cast<__gm__ PlForwardDoneLine *>(sym + DevPlForwardDoneLineOff(lane));
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(fd));
            if (fd->done_epoch < epoch) {
                lanes_done = false;
                break;
            }
        }
        if (!lanes_done) {
            ++spins;
            continue;
        }
        bool all_ok = true;
        uint32_t sum_exp = 0u;
        uint32_t sum_fwd = 0u;
        uint32_t cells_missing = 0u;
        uint32_t agg_fwd[kPlMaxSources];
        uint32_t agg_exp[kPlMaxSources];
        uint64_t dest_cyc[kPlMaxSources];
        uint64_t agg_tail[kPlMaxSources];
        for (uint32_t dest = 0; dest < wc; ++dest) {
            if (!PlIncBlock9AggregateDestFromLanes(sym, inc_rank, dest, wc, epoch, generation, &agg_fwd[dest],
                                                    &dest_cyc[dest], &agg_tail[dest], &agg_exp[dest])) {
                all_ok = false;
                ++cells_missing;
            } else {
                sum_exp += agg_exp[dest];
                sum_fwd += agg_fwd[dest];
            }
        }
        if (!all_ok) {
            ++spins;
            continue;
        }
        uint64_t max_cell_cyc = 0u;
        for (uint32_t dest = 0; dest < wc; ++dest) {
            PlPublishEgressVisibleDone(sym, inc_rank, dest, epoch, generation, agg_exp[dest], agg_fwd[dest],
                                       agg_tail[dest], 0u, dest_cyc[dest], leader_pe);
            if (dest_cyc[dest] > max_cell_cyc) {
                max_cell_cyc = dest_cyc[dest];
            }
        }
        const uint64_t global_agg_cyc = max_cell_cyc > 0u ? max_cell_cyc : AscendC::GetSystemCycle();
        const uint64_t sum_off = PlDevIncSecondHopSummaryOff(inc_rank);
        __gm__ PlIncSecondHopSummaryLine *sum =
            reinterpret_cast<__gm__ PlIncSecondHopSummaryLine *>(sym + sum_off);
        sum->epoch = epoch;
        sum->generation = generation;
        sum->inc_rank = inc_rank;
        sum->total_expected = sum_exp;
        sum->total_forwarded = sum_fwd;
        sum->error_code = 0u;
        sum->magic = kPlIncSecondHopSummaryMagic;
        sum->inc_aggregated_cycle = global_agg_cyc;
        sum->dest_cells_ready = wc;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(sum));
        const int my_pe = aclshmem_my_pe();
        if (my_pe != leader_pe) {
            aclshmem_putmem_nbi(reinterpret_cast<__gm__ void *>(sym + sum_off),
                                reinterpret_cast<__gm__ void *>(sum), sizeof(PlIncSecondHopSummaryLine), leader_pe);
            aclshmem_quiet();
        }
        return true;
    }
    // 专用遥测：禁止静默丢失 second_hop 聚合
    __gm__ PlSecondHopAggTelemetryLine *tel =
        reinterpret_cast<__gm__ PlSecondHopAggTelemetryLine *>(sym + kPlSecondHopAggTelemetryOff +
                                                                static_cast<uint64_t>(inc_rank) * 64u);
    tel->epoch = epoch;
    tel->inc_rank = inc_rank;
    tel->block_id = block_id;
    tel->spin_count = spins;
    tel->error_code = 71u; // INC second_hop agg timeout
    tel->cells_missing = wc;
    tel->magic = kPlSecondHopAggTelemetryMagic;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(tel));
    const int my_pe = aclshmem_my_pe();
    if (my_pe != leader_pe) {
        aclshmem_putmem_nbi(reinterpret_cast<__gm__ void *>(sym + kPlSecondHopAggTelemetryOff +
                                                             static_cast<uint64_t>(inc_rank) * 64u),
                            reinterpret_cast<__gm__ void *>(tel), sizeof(PlSecondHopAggTelemetryLine), leader_pe);
        aclshmem_quiet();
    }
    return false;
}

// PE0 单次尝试：全部 INC×dest egress cell 就绪则发布 timing（供 worker control 非阻塞轮询）
// 正式 done = release + max(cell delta)；禁止用 PE0 晚观察时刻，也不阻塞 transport 热路径
__aicore__ inline bool PlPe0TryPublishGlobalSecondHopVisibleTiming(__gm__ uint8_t *sym, uint32_t worker_count,
                                                                    uint64_t epoch, uint32_t generation,
                                                                    uint32_t owner_block_id)
{
    __gm__ PlSecondHopVisibleTimingLine *existing =
        reinterpret_cast<__gm__ PlSecondHopVisibleTimingLine *>(sym + kPlSecondHopVisibleTimingOff);
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(existing));
    if (existing->magic == kPlSecondHopVisibleTimingMagic && existing->timing_epoch == epoch &&
        existing->release_seen_cycle > 0u &&
        existing->inc_second_hop_visible_done_cycle > existing->release_seen_cycle) {
        return true;
    }
    uint32_t sum_fwd = 0u;
    uint32_t sum_exp = 0u;
    uint32_t slow_inc = 0u;
    uint32_t slow_dest = 0u;
    uint64_t remote_max_delta = 0u;
    uint64_t per_dest_max[kPlMaxSources];
    for (uint32_t d = 0; d < kPlMaxSources; ++d) {
        per_dest_max[d] = 0u;
    }
    for (uint32_t inc = 0; inc < worker_count; ++inc) {
        for (uint32_t dest = 0; dest < worker_count; ++dest) {
            __gm__ PlEgressLaneVisibleDoneLine *ev = reinterpret_cast<__gm__ PlEgressLaneVisibleDoneLine *>(
                sym + PlDevEgressVisibleDoneSlotOff(inc, dest));
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(ev));
            if (ev->magic != kPlEgressVisibleDoneMagic || ev->epoch != epoch || ev->generation != generation ||
                ev->inc_rank != inc || ev->destination_rank != dest || ev->error_code != 0u) {
                return false;
            }
            if (ev->checksum != PlEgressVisibleChecksum(inc, dest, ev->expected_routes, ev->forwarded_routes, epoch,
                                                         generation, ev->tail_value) ||
                ev->forwarded_routes != ev->expected_routes) {
                return false;
            }
            sum_fwd += ev->forwarded_routes;
            sum_exp += ev->expected_routes;
            // cell 内存的是 lane 相对 delta
            if (ev->visible_done_cycle > per_dest_max[dest]) {
                per_dest_max[dest] = ev->visible_done_cycle;
            }
            if (ev->visible_done_cycle >= remote_max_delta) {
                remote_max_delta = ev->visible_done_cycle;
                slow_inc = inc;
                slow_dest = dest;
            }
        }
    }
    if (sum_fwd != sum_exp) {
        return false;
    }
    const uint64_t release_cyc = PlResolvePe0ReleaseSeenCycle(sym, epoch);
    if (release_cyc == 0u) {
        return false;
    }
    // 有流量时 delta 必须 >0；全零路由允许 release+1 作占位
    uint64_t done_cyc = release_cyc + remote_max_delta;
    if (sum_exp > 0u && remote_max_delta == 0u) {
        return false;
    }
    if (sum_exp == 0u && done_cyc <= release_cyc) {
        done_cyc = release_cyc + 1u;
    }
    if (done_cyc <= release_cyc) {
        return false;
    }
    existing->timing_epoch = epoch;
    existing->release_seen_cycle = release_cyc;
    existing->inc_second_hop_visible_done_cycle = done_cyc;
    existing->slowest_cell_inc = slow_inc;
    existing->slowest_cell_dest = slow_dest;
    existing->owner_block_id = owner_block_id;
    existing->magic = kPlSecondHopVisibleTimingMagic;
    existing->worker_count = worker_count;
    for (uint32_t d = 0; d < kPlMaxSources; ++d) {
        // 诊断：存 PE0 域绝对 cycle = release + delta
        existing->per_dest_visible_done_cycle[d] =
            (d < worker_count) ? (release_cyc + per_dest_max[d]) : 0u;
    }
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(existing));
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(existing) + 64u);

    __gm__ PlGlobalSecondHopVisibleDoneLine *gs =
        reinterpret_cast<__gm__ PlGlobalSecondHopVisibleDoneLine *>(sym + kPlGlobalSecondHopVisibleDoneOff);
    gs->epoch = epoch;
    gs->global_second_hop_visible_epoch = epoch;
    gs->global_second_hop_visible_cycle = done_cyc;
    gs->slowest_inc = slow_inc;
    gs->slowest_dest = slow_dest;
    gs->magic = kPlGlobalSecondHopVisibleMagic;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(gs));
    return true;
}

// PE0 worker control：校验 count_matrix 与全部 egress visible cell 后发布正式 second_hop timing
__aicore__ inline bool PlPe0PublishGlobalSecondHopVisibleTiming(__gm__ uint8_t *sym, uint32_t worker_count,
                                                                 uint64_t epoch, uint32_t generation,
                                                                 uint32_t owner_block_id)
{
    uint32_t spins = 0u;
    while (spins < 8000000u) {
        if (PlPe0TryPublishGlobalSecondHopVisibleTiming(sym, worker_count, epoch, generation, owner_block_id)) {
            return true;
        }
        ++spins;
    }
    return false;
}

// 负例延迟：spin 次 PipeBarrier（host 用 µs×kD1CyclesPerUs 写入；非 GetSystemCycle 墙钟）
__aicore__ inline void PlApplyTimingNegDelay(uint32_t delay_spins)
{
    for (uint32_t i = 0; i < delay_spins; ++i) {
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// destination control：本 dest RecvDone 齐后发布；非 PE0 put 到 leader 堆槽
__aicore__ inline void PlPublishTransportDone(__gm__ uint8_t *sym, uint32_t dest_rank, uint64_t epoch,
                                               uint32_t expected_routes, uint32_t received_routes,
                                               uint32_t error_code, int leader_pe)
{
    // delay_recv 在 PE0 PublishGlobalTransportTiming：TransportDone 齐后、打点前注入
    __gm__ PlTransportDoneLine *line =
        reinterpret_cast<__gm__ PlTransportDoneLine *>(sym + PlDevTransportDoneLineOff(dest_rank));
    line->epoch = epoch;
    line->destination_rank = dest_rank;
    line->expected_routes = expected_routes;
    line->received_routes = received_routes;
    line->error_code = error_code;
    line->transport_done_cycle = AscendC::GetSystemCycle();
    line->magic = kPlTransportDoneMagic;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(line));
    PlDevRankTimingMarkDestinationDone(sym, line->transport_done_cycle);
    const int my_pe = aclshmem_my_pe();
    if (my_pe != leader_pe) {
        aclshmem_putmem_nbi(reinterpret_cast<__gm__ void *>(sym + PlDevTransportDoneLineOff(dest_rank)),
                            reinterpret_cast<__gm__ void *>(line), sizeof(PlTransportDoneLine), leader_pe);
        aclshmem_quiet();
    }
}

enum PlGlobalTransportResult : uint32_t {
    kPlGlobalTransportOk = 0u,
    kPlGlobalTransportMissing = 1u,
    kPlGlobalTransportPeerError = 2u,
    kPlGlobalTransportReleaseMissing = 3u,
};

__aicore__ inline void PlWriteGlobalTransportFailure(__gm__ uint8_t *sym, uint64_t epoch, uint32_t worker_count,
                                                     uint32_t owner_block_id, uint32_t stage, uint32_t error_rank,
                                                     uint32_t error_destination, uint32_t error_code,
                                                     uint32_t missing_mask, uint32_t spins)
{
    __gm__ PlTransportTimingLine *tt =
        reinterpret_cast<__gm__ PlTransportTimingLine *>(sym + kPlTransportTimingOff);
    tt->timing_epoch = epoch;
    tt->release_seen_cycle = 0u;
    tt->global_transport_done_cycle = 0u;
    tt->slowest_destination = error_destination;
    tt->owner_block_id = owner_block_id;
    tt->magic = kPlTransportTimingMagic;
    tt->worker_count = worker_count;
    for (uint32_t d = 0; d < kPlMaxSources; ++d) {
        tt->per_dest_transport_done_cycle[d] = 0u;
    }
    tt->failure_stage = stage;
    tt->error_rank = error_rank;
    tt->error_destination = error_destination;
    tt->first_error_code = error_code;
    tt->missing_destination_mask = missing_mask;
    tt->global_wait_spins = spins;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(tt));
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(tt) + 64u);
}

// PE0 worker control 单 writer：观察齐全部 dest transport_done → 写 global + timing
__aicore__ inline PlGlobalTransportResult PlPe0PublishGlobalTransportTiming(__gm__ uint8_t *sym,
                                                                             uint32_t worker_count, uint64_t epoch,
                                                                             uint32_t owner_block_id)
{
    uint64_t first_seen[kPlMaxSources];
    for (uint32_t d = 0; d < kPlMaxSources; ++d) {
        first_seen[d] = 0u;
    }
    uint32_t spins = 0u;
    bool saw_all_transport_done = false;
    while (spins < 8000000u) {
        bool all = true;
        uint32_t missing_mask = 0u;
        for (uint32_t d = 0; d < worker_count; ++d) {
            __gm__ PlTransportDoneLine *td =
                reinterpret_cast<__gm__ PlTransportDoneLine *>(sym + PlDevTransportDoneLineOff(d));
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(td));
            if (td->magic == kPlTransportDoneMagic && td->epoch == epoch && td->destination_rank == d &&
                td->error_code != 0u) {
                PlWriteGlobalTransportFailure(sym, epoch, worker_count, owner_block_id,
                                              kPlGlobalTransportPeerError, d, d, td->error_code, 0u, spins);
                return kPlGlobalTransportPeerError;
            }
            if (td->magic != kPlTransportDoneMagic || td->epoch != epoch || td->destination_rank != d) {
                all = false;
                missing_mask |= (1u << d);
                continue;
            }
            if (first_seen[d] == 0u) {
                first_seen[d] = AscendC::GetSystemCycle();
            }
        }
        if (all) {
            saw_all_transport_done = true;
            uint32_t slowest = 0u;
            uint64_t max_seen = 0u;
            for (uint32_t d = 0; d < worker_count; ++d) {
                if (first_seen[d] >= max_seen) {
                    max_seen = first_seen[d];
                    slowest = d;
                }
            }
            const uint64_t release_cyc = PlResolvePe0ReleaseSeenCycle(sym, epoch);
            if (release_cyc == 0u) {
                ++spins;
                continue;
            }
            // H8.0：formal transport 只等全部 Dest TransportDone；禁止在此扫/等/发 second-hop
            // FAULT_DELAY_RECV：齐后、打点前自旋 — 拉长 recv/full，不改 second_hop 热路径
            {
                __gm__ PlTimingNegLine *neg =
                    reinterpret_cast<__gm__ PlTimingNegLine *>(sym + kPlTimingNegOff);
                PlDcci(reinterpret_cast<__gm__ uint8_t *>(neg));
                if (neg->magic == kPlTimingNegMagic && neg->delay_recv_cycles > 0u) {
                    PlApplyTimingNegDelay(neg->delay_recv_cycles);
                }
            }
            const uint64_t done_cyc = AscendC::GetSystemCycle();
            if (done_cyc <= release_cyc) {
                ++spins;
                continue;
            }
            __gm__ PlTransportTimingLine *tt =
                reinterpret_cast<__gm__ PlTransportTimingLine *>(sym + kPlTransportTimingOff);
            tt->timing_epoch = epoch;
            tt->release_seen_cycle = release_cyc;
            tt->global_transport_done_cycle = done_cyc;
            tt->slowest_destination = slowest;
            tt->owner_block_id = owner_block_id;
            tt->magic = kPlTransportTimingMagic;
            tt->worker_count = worker_count;
            for (uint32_t d = 0; d < kPlMaxSources; ++d) {
                tt->per_dest_transport_done_cycle[d] = (d < worker_count) ? first_seen[d] : 0u;
            }
            tt->failure_stage = 0u;
            tt->error_rank = 0u;
            tt->error_destination = 0u;
            tt->first_error_code = 0u;
            tt->missing_destination_mask = 0u;
            tt->global_wait_spins = spins;
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(tt));
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(tt) + 64u);

            __gm__ PlGlobalTransportDoneLine *gtd =
                reinterpret_cast<__gm__ PlGlobalTransportDoneLine *>(sym + kPlGlobalTransportDoneOff);
            gtd->epoch = epoch;
            gtd->global_transport_done_epoch = epoch;
            gtd->global_transport_done_cycle = done_cyc;
            gtd->slowest_destination = slowest;
            gtd->magic = kPlGlobalTransportDoneMagic;
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(gtd));
            return kPlGlobalTransportOk;
        }
        ++spins;
    }
    uint32_t missing_mask = 0u;
    for (uint32_t d = 0; d < worker_count; ++d) {
        __gm__ PlTransportDoneLine *td =
            reinterpret_cast<__gm__ PlTransportDoneLine *>(sym + PlDevTransportDoneLineOff(d));
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(td));
        if (td->magic != kPlTransportDoneMagic || td->epoch != epoch || td->destination_rank != d) {
            missing_mask |= (1u << d);
        }
    }
    const PlGlobalTransportResult result =
        saw_all_transport_done ? kPlGlobalTransportReleaseMissing : kPlGlobalTransportMissing;
    PlWriteGlobalTransportFailure(sym, epoch, worker_count, owner_block_id, result, 0u, 0u,
                                  kPlErrGlobalTransportMissing, missing_mask, spins);
    return result;
}

__aicore__ inline void PlCopyGmBytes(__gm__ uint8_t *dst, __gm__ uint8_t *src, uint32_t nbytes)
{
    for (uint32_t i = 0; i < nbytes; ++i) {
        dst[i] = src[i];
    }
}

__aicore__ inline uint32_t PlFullDispatchEpochIndex(__gm__ PlFullDispatchConfig *cfg, uint64_t epoch)
{
    const uint32_t n = cfg->input_epoch_count == 0u ? 1u : cfg->input_epoch_count;
    return static_cast<uint32_t>((epoch - 1u) % static_cast<uint64_t>(n));
}

// shape 由 slot index 选定；template 或 epoch 精确匹配均可；禁止静默 fallback 到错误 shape
__aicore__ inline bool PlFullDispatchInvocationShapeActive(__gm__ PlInvocationDesc *inv, uint64_t epoch)
{
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(inv));
    if ((inv->flags & kPlInvocationFlagsTemplateSlot) != 0u) {
        return true;
    }
    if (inv->epoch == epoch) {
        return true;
    }
    // 多 shape 循环：slot 已按 (go_epoch-1)%N 选定，shape 字段有效即可
    return inv->worker_count > 0u && inv->local_source_capacity > 0u;
}

__aicore__ inline __gm__ PlInvocationDesc *PlFullDispatchInvocation(__gm__ uint8_t *sym,
                                                                     __gm__ PlFullDispatchConfig *cfg, uint64_t epoch)
{
    const uint32_t idx = PlFullDispatchEpochIndex(cfg, epoch);
    // 设备侧直接算偏移（abi.h 的 host inline 不可从 aicore 调用）
    return reinterpret_cast<__gm__ PlInvocationDesc *>(
        sym + kPlInvocationDescOff + static_cast<uint64_t>(idx) * sizeof(PlInvocationDesc));
}

__aicore__ inline __gm__ PlInvocationWorkspaceDesc *PlFullDispatchInvocationWorkspace(
    __gm__ uint8_t *sym, __gm__ PlFullDispatchConfig *cfg, uint64_t epoch)
{
    const uint32_t idx = PlFullDispatchEpochIndex(cfg, epoch);
    return reinterpret_cast<__gm__ PlInvocationWorkspaceDesc *>(
        sym + kPlInvocationWorkspaceOff + static_cast<uint64_t>(idx) * sizeof(PlInvocationWorkspaceDesc));
}

// M2：读 workspace desc；Legacy 下仍走 symmetric DestFinal。禁止 magic/gen/flag 失败后 silent fallback。
__aicore__ inline bool PlDevValidateInvocationWorkspace(__gm__ PlInvocationWorkspaceDesc *ws,
                                                          __gm__ PlInvocationDesc *inv, uint32_t *out_err)
{
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(ws));
    if (ws == nullptr || ws->magic != kPlInvocationWorkspaceMagic) {
        if (out_err != nullptr) {
            *out_err = kPlErrWorkspaceDescMissing;
        }
        return false;
    }
    if (inv != nullptr && inv->workspace_generation != 0u && ws->generation != inv->workspace_generation) {
        if (out_err != nullptr) {
            *out_err = kPlErrWorkspaceGenerationMismatch;
        }
        return false;
    }
    if (ws->generation == 0u) {
        if (out_err != nullptr) {
            *out_err = kPlErrWorkspaceGenerationMismatch;
        }
        return false;
    }
    const bool legacy = (ws->flags & kPlInvocationWorkspaceFlagLegacySymmetricFinal) != 0u;
    const bool owned =
        (ws->flags & (kPlInvocationWorkspaceFlagCallerOwned | kPlInvocationWorkspaceFlagPooled)) != 0u;
    const bool local_final = (ws->flags & kPlInvocationWorkspaceFlagWorkerLocalFinal) != 0u;
    if (!legacy && !owned && !local_final) {
        if (out_err != nullptr) {
            *out_err = kPlErrWorkspaceFlagInvalid;
        }
        return false;
    }
    if (legacy && (owned || local_final)) {
        if (out_err != nullptr) {
            *out_err = kPlErrWorkspaceFlagInvalid;
        }
        return false;
    }
    if (!legacy) {
        if (local_final && ws->expand_x_ptr == 0u) {
            // INC PE stub：仅声明 LocalFinal 语义；Worker 输出 ptr 在 destination Worker 上绑定
            return true;
        }
        if (ws->expand_x_ptr == 0u || (ws->expand_x_ptr % 64u) != 0u || ws->assist_info_ptr == 0u ||
            (ws->assist_info_ptr % 64u) != 0u) {
            if (out_err != nullptr) {
                *out_err = kPlErrWorkspacePointerAlignment;
            }
            return false;
        }
        if (ws->output_route_capacity == 0u ||
            (inv != nullptr && inv->final_slot_capacity != 0u &&
             ws->output_route_capacity < inv->final_slot_capacity)) {
            if (out_err != nullptr) {
                *out_err = kPlErrOutputCapacityInsufficient;
            }
            return false;
        }
    }
    if (out_err != nullptr) {
        *out_err = kPlErrNone;
    }
    return true;
}

// Full Dispatch epoch 入口共用：失败写 full_output_done.error_code
__aicore__ inline bool PlDevRequireInvocationWorkspace(__gm__ uint8_t *sym, __gm__ PlFullDispatchConfig *cfg,
                                                         uint64_t epoch)
{
    __gm__ PlInvocationDesc *inv = PlFullDispatchInvocation(sym, cfg, epoch);
    __gm__ PlInvocationWorkspaceDesc *ws = PlFullDispatchInvocationWorkspace(sym, cfg, epoch);
    uint32_t err = kPlErrNone;
    if (PlDevValidateInvocationWorkspace(ws, inv, &err)) {
        return true;
    }
    __gm__ PlFullOutputDoneLine *done =
        reinterpret_cast<__gm__ PlFullOutputDoneLine *>(sym + cfg->full_output_done_off);
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(done));
    done->error_code = err;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(done));
    return false;
}

// M3：WorkspaceDesc 是否要求 INC→DestChannel + Worker local final
__aicore__ inline bool PlDevWorkspaceIsWorkerLocalFinal(__gm__ PlInvocationWorkspaceDesc *ws)
{
    if (ws == nullptr) {
        return false;
    }
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(ws));
    return (ws->flags & kPlInvocationWorkspaceFlagWorkerLocalFinal) != 0u;
}

// 有效 token：优先 invocation 模板/shape slot；缺省回退 tokens_per_epoch（对称旧路径）
__aicore__ inline uint32_t PlFullDispatchLocalTokenCount(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                          __gm__ PlFullDispatchConfig *cfg, uint64_t epoch)
{
    __gm__ PlInvocationDesc *inv = PlFullDispatchInvocation(sym, cfg, epoch);
    if (inv->worker_count == desc->worker_count && PlFullDispatchInvocationShapeActive(inv, epoch)) {
        const uint32_t src = desc->pair_id;
        if (src < kPlMaxSources) {
            return inv->source_token_count[src];
        }
        return 0u;
    }
    return desc->tokens_per_epoch;
}

__aicore__ inline uint64_t PlFullDispatchRawInputBase(__gm__ PlFullDispatchConfig *cfg, __gm__ PlPipelineDesc *desc,
                                                       __gm__ uint8_t *sym, uint64_t epoch)
{
    __gm__ PlInvocationWorkspaceDesc *ws = PlFullDispatchInvocationWorkspace(sym, cfg, epoch);
    if (ws != nullptr) {
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(ws));
        if (ws->input_ptr != 0u) {
            const uint32_t idx = PlFullDispatchEpochIndex(cfg, epoch);
            __gm__ PlInvocationDesc *inv = PlFullDispatchInvocation(sym, cfg, epoch);
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(inv));
            uint64_t stride = inv->raw_input_epoch_stride_bytes;
            if (stride == 0u || !PlFullDispatchInvocationShapeActive(inv, epoch)) {
                const uint32_t cap =
                    (inv->local_source_capacity != 0u) ? inv->local_source_capacity : desc->tokens_per_epoch;
                stride = static_cast<uint64_t>(cap) * static_cast<uint64_t>(desc->payload_bytes);
            }
            return ws->input_ptr + static_cast<uint64_t>(idx) * stride;
        }
    }
    // M4：input_ptr==0 → symmetric upload staging；host 每 epoch 刷新当前片，device 不再加 epoch stride
    (void)desc;
    (void)epoch;
    return cfg->raw_input_off;
}

__aicore__ inline uint64_t PlFullDispatchRawExpertBase(__gm__ PlFullDispatchConfig *cfg, __gm__ PlPipelineDesc *desc,
                                                        __gm__ uint8_t *sym, uint64_t epoch)
{
    __gm__ PlInvocationWorkspaceDesc *ws = PlFullDispatchInvocationWorkspace(sym, cfg, epoch);
    if (ws != nullptr) {
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(ws));
        if (ws->expert_ids_ptr != 0u) {
            const uint32_t idx = PlFullDispatchEpochIndex(cfg, epoch);
            __gm__ PlInvocationDesc *inv = PlFullDispatchInvocation(sym, cfg, epoch);
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(inv));
            uint64_t stride = inv->raw_expert_ids_epoch_stride_bytes;
            if (stride == 0u || !PlFullDispatchInvocationShapeActive(inv, epoch)) {
                const uint32_t topk = desc->route_topk == 0u ? 1u : desc->route_topk;
                const uint32_t cap =
                    (inv->local_source_capacity != 0u) ? inv->local_source_capacity : desc->tokens_per_epoch;
                stride = static_cast<uint64_t>(cap) * static_cast<uint64_t>(topk) * sizeof(int32_t);
            }
            return ws->expert_ids_ptr + static_cast<uint64_t>(idx) * stride;
        }
    }
    // M4：expert_ids_ptr==0 → symmetric upload staging（host 每 epoch 刷新）
    (void)desc;
    (void)epoch;
    return cfg->raw_expert_ids_off;
}

// FNV-1a 64-bit（与 host ArtifactChecksum64 / scripts 对齐）
__aicore__ inline uint64_t PlFnv1a64Bytes(__gm__ const uint8_t *data, uint32_t len)
{
    constexpr uint64_t kOffset = 0xCBF29CE484222325ull;
    constexpr uint64_t kPrime = 0x100000001B3ull;
    uint64_t h = kOffset;
    for (uint32_t i = 0u; i < len; ++i) {
        h ^= static_cast<uint64_t>(data[i]);
        h *= kPrime;
    }
    return h;
}

__aicore__ inline uint64_t PlFnv1a64Int32Range(__gm__ const int32_t *data, uint32_t nelem)
{
    return PlFnv1a64Bytes(reinterpret_cast<__gm__ const uint8_t *>(data),
                          nelem * static_cast<uint32_t>(sizeof(int32_t)));
}

// 统一 epoch route 视图：count / meta publish / upload mask / gather 必须同源。
__aicore__ inline PlEpochRouteView PlResolveEpochRouteView(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                           __gm__ PlFullDispatchConfig *cfg, uint64_t epoch)
{
    PlEpochRouteView v{};
    v.source_rank = desc->pair_id;
    v.topk = desc->route_topk == 0u ? 1u : desc->route_topk;
    v.expert_per_pe = desc->expert_per_pe == 0u ? 8u : desc->expert_per_pe;
    v.moe_expert_num =
        desc->moe_expert_num == 0u ? (desc->worker_count * v.expert_per_pe) : desc->moe_expert_num;
    v.token_count = PlFullDispatchLocalTokenCount(sym, desc, cfg, epoch);
    v.eid_base = PlFullDispatchRawExpertBase(cfg, desc, sym, epoch);
    v.metadata_bytes = v.token_count * v.topk * static_cast<uint32_t>(sizeof(int32_t));
    v.ok = 1u;

    __gm__ PlInvocationDesc *inv = PlFullDispatchInvocation(sym, cfg, epoch);
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(inv));
    if (PlFullDispatchInvocationShapeActive(inv, epoch) && inv->worker_count == desc->worker_count) {
        const uint32_t src = desc->pair_id;
        if (src < kPlMaxSources && inv->source_token_count[src] != v.token_count) {
            v.ok = 0u;
        }
    }
    if (v.metadata_bytes > kPlIncRawRouteMetaBytes) {
        v.ok = 0u;
    }
    if (v.eid_base == 0u && v.token_count > 0u) {
        v.ok = 0u;
    }
    return v;
}

// Worker control 单写者：整 epoch expert_ids contiguous put → quiet → RawMetaReady doorbell。
// 禁止 upload lane 分片写 packed metadata（false sharing）。
__aicore__ inline void PlFullDispatchSourcePublishRouteMeta(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                            __gm__ PlFullDispatchConfig *cfg, uint64_t epoch)
{
    const PlEpochRouteView view = PlResolveEpochRouteView(sym, desc, cfg, epoch);
    const int inc_peer = static_cast<int>(desc->peer_pe);
    __gm__ PlRawMetaReadyLine *rdy =
        reinterpret_cast<__gm__ PlRawMetaReadyLine *>(sym + kPlRawMetaReadyOff);
    if (view.ok == 0u) {
        __gm__ PlFullOutputDoneLine *err =
            reinterpret_cast<__gm__ PlFullOutputDoneLine *>(sym + cfg->full_output_done_off);
        err->error_code = 16u;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(err));
        rdy->epoch = epoch;
        rdy->source_rank = view.source_rank;
        rdy->metadata_bytes = 0u;
        rdy->metadata_checksum = 0u;
        rdy->magic = kPlRawMetaReadyMagic;
        rdy->ready_signal = 0;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(rdy));
        return;
    }

    __gm__ int32_t *eids = reinterpret_cast<__gm__ int32_t *>(view.eid_base);
    if (view.metadata_bytes > 0u) {
        const bool packed_route_wire =
            (desc->pipeline_mode & kPlIngressLayoutSourceMajorRaw) != 0u;
        if (!packed_route_wire) {
            Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(eids),
                         view.metadata_bytes);
            aclshmem_putmem_nbi(sym + kPlIncRawRouteMetaOff,
                                reinterpret_cast<__gm__ uint8_t *>(eids),
                                view.metadata_bytes, inc_peer);
            aclshmem_quiet();
        }
    } else {
        aclshmem_quiet();
    }

    rdy->epoch = epoch;
    rdy->source_rank = view.source_rank;
    rdy->metadata_bytes = view.metadata_bytes;
    // The ready line proves ordering only.  Every raw chunk carries an
    // independently verified metadata checksum, so a second serial FNV pass
    // over the complete epoch is redundant and needlessly delays K1 startup.
    rdy->metadata_checksum = 0u;
    rdy->magic = kPlRawMetaReadyMagic;
    // drop_meta_ready：写本地线但不发布 ready_signal / 不 put 到 INC → error 70
    {
        __gm__ PlP53InjectLine *inj = reinterpret_cast<__gm__ PlP53InjectLine *>(sym + kPlP53InjectOff);
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(inj));
        if (inj->magic == kPlP53InjectMagic && (inj->flags & kPlP53InjectDropMetaReady) != 0u) {
            rdy->ready_signal = 0;
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(rdy));
            return;
        }
    }
    rdy->ready_signal = static_cast<int32_t>(epoch);
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(rdy));
    aclshmem_putmem_nbi(sym + kPlRawMetaReadyOff, reinterpret_cast<__gm__ void *>(rdy),
                        sizeof(PlRawMetaReadyLine), inc_peer);
    aclshmem_quiet();
}

__aicore__ inline __gm__ int32_t *PlFullDispatchExpertTokenNumsPtr(__gm__ uint8_t *sym, __gm__ PlFullDispatchConfig *cfg,
                                                                    uint64_t epoch)
{
    __gm__ PlInvocationWorkspaceDesc *ws = PlFullDispatchInvocationWorkspace(sym, cfg, epoch);
    if (ws != nullptr) {
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(ws));
        if (ws->expert_token_nums_ptr != 0u) {
            return reinterpret_cast<__gm__ int32_t *>(ws->expert_token_nums_ptr);
        }
    }
    return reinterpret_cast<__gm__ int32_t *>(cfg->expert_token_nums_off);
}

__aicore__ inline __gm__ int32_t *PlFullDispatchEpRecvCountPtr(__gm__ uint8_t *sym, __gm__ PlFullDispatchConfig *cfg,
                                                                  uint64_t epoch)
{
    __gm__ PlInvocationWorkspaceDesc *ws = PlFullDispatchInvocationWorkspace(sym, cfg, epoch);
    if (ws != nullptr) {
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(ws));
        if (ws->ep_recv_count_ptr != 0u) {
            return reinterpret_cast<__gm__ int32_t *>(ws->ep_recv_count_ptr);
        }
    }
    return reinterpret_cast<__gm__ int32_t *>(cfg->ep_recv_count_off);
}

__aicore__ inline void PlFullDispatchResetTiming(__gm__ PlRouteTimingLine *rt, uint64_t epoch)
{
    rt->timing_epoch = epoch;
    rt->route_count_start_cycle = AscendC::GetSystemCycle();
    rt->route_count_done_cycle = 0u;
    rt->prefix_done_cycle = 0u;
    rt->gather_first_cycle = 0u;
    rt->gather_done_cycle = 0u;
    rt->stage1_done_cycle = 0u;
    rt->finalize_done_cycle = 0u;
    rt->full_done_cycle = 0u;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(rt));
}

__aicore__ inline uint32_t PlCountLineChecksum(uint32_t source_rank, uint32_t dest_rank, uint32_t local_expert,
                                                 uint32_t count, uint64_t epoch)
{
    return source_rank ^ (dest_rank << 8) ^ (local_expert << 16) ^ count ^ static_cast<uint32_t>(epoch) ^
           0xC0FFEE01u;
}

__aicore__ inline void PlFillRouteCountLine(__gm__ uint8_t *sym, uint64_t line_off, uint32_t dest_rank,
                                              uint32_t local_expert, uint32_t source_rank, uint32_t count,
                                              uint64_t epoch)
{
    __gm__ PlRouteCountLine *line = reinterpret_cast<__gm__ PlRouteCountLine *>(sym + line_off);
    line->epoch = epoch;
    line->destination_rank = dest_rank;
    line->local_expert_id = local_expert;
    line->source_rank = source_rank;
    line->count = count;
    line->ready_epoch = epoch; // 数据字段；正式放行靠独立 doorbell
    line->checksum = PlCountLineChecksum(source_rank, dest_rank, local_expert, count, epoch);
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(line));
}

__aicore__ inline bool PlValidateRouteCountLine(__gm__ PlRouteCountLine *line, uint64_t epoch, uint32_t source_rank,
                                                  uint32_t dest_rank, uint32_t local_expert)
{
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(line));
    if (line->epoch != epoch || line->ready_epoch != epoch) {
        return false;
    }
    if (line->source_rank != source_rank || line->destination_rank != dest_rank ||
        line->local_expert_id != local_expert) {
        return false;
    }
    return line->checksum ==
           PlCountLineChecksum(source_rank, dest_rank, local_expert, line->count, epoch);
}

// worker_count=1：本地 prefix 完成后、counts_done 之前发布 SegmentBaseReady（gather lane 依赖）
__aicore__ inline void PlFullDispatchPublishLocalSegmentBaseReady(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                                   uint64_t epoch)
{
    const uint32_t source = desc->pair_id;
    const uint32_t dest = source;
    __gm__ PlSegmentBaseReadyLine *brdy =
        reinterpret_cast<__gm__ PlSegmentBaseReadyLine *>(sym + PlDevSegmentBaseReadyLineOff(source, dest));
    brdy->epoch = epoch;
    brdy->source_rank = source;
    brdy->destination_rank = dest;
    brdy->checksum = static_cast<uint32_t>(epoch) ^ (dest << 16) ^ (source << 8);
    brdy->magic = kPlSegBaseReadyMagic;
    brdy->ready_epoch = epoch;
    brdy->visible_signal = static_cast<int32_t>(epoch);
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(brdy));
}

// C1 快路径：单 worker 本地 count/prefix（兼容 topk 双循环）
__aicore__ inline void PlFullDispatchCountPrefixLocal(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                       __gm__ PlFullDispatchConfig *cfg, uint64_t epoch)
{
    __gm__ PlRouteTimingLine *rt =
        reinterpret_cast<__gm__ PlRouteTimingLine *>(sym + cfg->route_timing_off);
    __gm__ PlFullOutputDoneLine *done =
        reinterpret_cast<__gm__ PlFullOutputDoneLine *>(sym + cfg->full_output_done_off);
    const uint32_t tokens = PlFullDispatchLocalTokenCount(sym, desc, cfg, epoch);
    const uint32_t epp = desc->expert_per_pe == 0u ? 8u : desc->expert_per_pe;
    const uint32_t moe = desc->moe_expert_num == 0u ? (desc->worker_count * epp) : desc->moe_expert_num;
    const uint32_t topk = desc->route_topk == 0u ? 1u : desc->route_topk;
    const uint32_t wc = desc->worker_count;

    PlFullDispatchResetTiming(rt, epoch);

    const uint64_t eid_base = PlFullDispatchRawExpertBase(cfg, desc, sym, epoch);
    __gm__ int32_t *eids = reinterpret_cast<__gm__ int32_t *>(eid_base);
    if (tokens > 0u) {
        Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(eids), tokens * topk * sizeof(int32_t));
    }
    __gm__ int32_t *expert_nums = PlFullDispatchExpertTokenNumsPtr(sym, cfg, epoch);
    __gm__ int32_t *ep_recv = PlFullDispatchEpRecvCountPtr(sym, cfg, epoch);
    __gm__ int32_t *seg_base = reinterpret_cast<__gm__ int32_t *>(sym + cfg->segment_base_off);
    __gm__ int32_t *expected = reinterpret_cast<__gm__ int32_t *>(sym + kPlExpectedCountOff);

    uint32_t running = 0u;
    uint32_t total_routes = 0u;
    for (uint32_t e = 0; e < epp; ++e) {
        uint32_t c = 0u;
        for (uint32_t t = 0; t < tokens; ++t) {
            for (uint32_t slot = 0; slot < topk; ++slot) {
                const int32_t eid = eids[static_cast<uint64_t>(t) * topk + slot];
                if (eid < 0 || static_cast<uint32_t>(eid) >= moe) {
                    done->error_code = kPlErrBadExpertId;
                    PlDcci(reinterpret_cast<__gm__ uint8_t *>(done));
                    return;
                }
                const uint32_t dest = static_cast<uint32_t>(eid) / epp;
                if (dest != desc->pair_id) {
                    done->error_code = kPlErrDestMismatch;
                    PlDcci(reinterpret_cast<__gm__ uint8_t *>(done));
                    return;
                }
                if (static_cast<uint32_t>(eid) % epp == e) {
                    c += 1u;
                }
            }
        }
        expert_nums[e] = static_cast<int32_t>(c);
        running += c;
        ep_recv[e] = static_cast<int32_t>(running);
        seg_base[PlDevSegmentBaseIndex(e, 0u, wc)] = static_cast<int32_t>(running - c);
        total_routes += c;
    }
    expected[0] = static_cast<int32_t>(total_routes);
    Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(expert_nums), epp * sizeof(int32_t));
    Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(ep_recv), epp * sizeof(int32_t));
    Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(seg_base), epp * wc * sizeof(int32_t));
    Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(expected), wc * sizeof(int32_t));

    // gather 读 kPlOutboundSegBaseOff（与多 worker DestPrefix 对齐）；禁止只写 segment_base_off
    __gm__ int32_t *out_local = reinterpret_cast<__gm__ int32_t *>(sym + kPlOutboundSegBaseOff);
    const uint32_t dest = desc->pair_id;
    for (uint32_t le = 0; le < epp && le < 16u; ++le) {
        out_local[PlDevOutboundSegBaseIndex(dest, le)] =
            seg_base[PlDevSegmentBaseIndex(le, 0u, wc)];
    }
    for (uint32_t pad_i = epp; pad_i < 16u; ++pad_i) {
        out_local[PlDevOutboundSegBaseIndex(dest, pad_i)] = 0;
    }
    Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(out_local + PlDevOutboundSegBaseIndex(dest, 0u)), 64u);

    __gm__ PlLaneWorkLine *lane0 =
        reinterpret_cast<__gm__ PlLaneWorkLine *>(sym + cfg->lane_work_off);
    lane0->epoch = epoch;
    lane0->lane_id = 0u;
    lane0->token_count = total_routes;
    lane0->gather_count = 0u;
    lane0->upload_count = 0u;
    lane0->done_epoch = 0u;
    lane0->error_code = 0u;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(lane0));

    for (uint32_t s = 1; s < wc; ++s) {
        __gm__ PlLaneWorkLine *lw =
            reinterpret_cast<__gm__ PlLaneWorkLine *>(sym + cfg->lane_work_off + static_cast<uint64_t>(s) * 64u);
        lw->token_count = 0u;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(lw));
    }

    rt->route_count_done_cycle = AscendC::GetSystemCycle();
    rt->prefix_done_cycle = rt->route_count_done_cycle;
    if (desc->worker_count <= 1u) {
        PlFullDispatchPublishLocalSegmentBaseReady(sym, desc, epoch);
    }
    done->counts_done_epoch = epoch;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(rt));
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(done));
}

// 源 worker：只 push 到 partner INC（禁止直达 destination；连续 NBI 改为 contiguous range）
__aicore__ inline void PlFullDispatchSourcePublishCounts(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                          __gm__ PlFullDispatchConfig *cfg, uint64_t epoch)
{
    // 与 route-meta publish 共用同一 PlEpochRouteView；禁止各自重算 eid 地址。
    const PlEpochRouteView view = PlResolveEpochRouteView(sym, desc, cfg, epoch);
    const uint32_t tokens = view.token_count;
    const uint32_t epp = view.expert_per_pe;
    const uint32_t moe = view.moe_expert_num;
    const uint32_t topk = view.topk;
    const uint32_t source = view.source_rank;
    const uint32_t wc = desc->worker_count;
    const int inc_peer = static_cast<int>(desc->peer_pe);

    __gm__ PlWorkerSourceCountTrace *wtr =
        reinterpret_cast<__gm__ PlWorkerSourceCountTrace *>(sym + PlDevWorkerSourceCountTraceOff(source));
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(wtr));
    if (wtr->epoch == epoch && wtr->stage >= kPlCountStageWorkerSourceReady) {
        __gm__ PlFullOutputDoneLine *dup_err =
            reinterpret_cast<__gm__ PlFullOutputDoneLine *>(sym + cfg->full_output_done_off);
        dup_err->error_code = 19u; // dual SourcePublish on same epoch
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(dup_err));
        wtr->error_code = 19u;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(wtr));
        return;
    }
    wtr->epoch = epoch;
    wtr->generation = static_cast<uint32_t>(epoch);
    wtr->stage = kPlCountStageWorkerEnter;
    wtr->source_rank = source;
    wtr->total_routes = 0u;
    wtr->error_code = 0u;
    wtr->magic = kPlCountTraceMagic;
    wtr->enter_cycle = AscendC::GetSystemCycle();
    wtr->exit_cycle = 0u;
    wtr->observed_epoch = 0u;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(wtr));

    if (view.ok == 0u) {
        __gm__ PlFullOutputDoneLine *err =
            reinterpret_cast<__gm__ PlFullOutputDoneLine *>(sym + cfg->full_output_done_off);
        err->error_code = 16u;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(err));
        wtr->error_code = 16u;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(wtr));
    }

    __gm__ int32_t *eids = reinterpret_cast<__gm__ int32_t *>(view.eid_base);
    if (tokens > 0u && view.ok != 0u) {
        Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(eids), view.metadata_bytes);
    }

    uint32_t local_count[kPlMaxSources * 8];
    for (uint32_t i = 0; i < kPlMaxSources * 8u; ++i) {
        local_count[i] = 0u;
    }
    uint32_t total_routes = 0u;
    uint32_t matrix_checksum = 0u;
    const bool source_major_raw =
        (desc->pipeline_mode & kPlIngressLayoutSourceMajorRaw) != 0u;
    const bool destination_csr =
        source_major_raw &&
        (desc->pipeline_mode & kPlModeDestinationCsrRoutePlan) != 0u;
    if (destination_csr &&
        !PlDevDestinationCsrFits(tokens, topk, wc)) {
        __gm__ PlFullOutputDoneLine *err =
            reinterpret_cast<__gm__ PlFullOutputDoneLine *>(
                sym + cfg->full_output_done_off);
        err->error_code = kPlErrOutputCapacityInsufficient;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(err));
        wtr->error_code = kPlErrOutputCapacityInsufficient;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(wtr));
        return;
    }
    // Source-major consumes expert id and source-segment ordinal as one
    // atomic routing fact.  Keep them in one 32-bit wire word for every K:
    // low 8 bits = expert id, high 24 bits = ordinal.  The previous K>1
    // split-wire representation could pair a fresh expert-id word with a
    // stale/corrupted ordinal and silently collide final slots.
    const bool packed_route_wire = source_major_raw;
    __gm__ uint32_t *route_packed =
        reinterpret_cast<__gm__ uint32_t *>(sym + kPlIncRawRouteMetaOff);
    bool eid_invalid = false;
    const uint32_t route_slots = tokens * topk;
    for (uint32_t route_idx = 0u; route_idx < route_slots; ++route_idx) {
        const int32_t eid = eids[route_idx];
        uint32_t packed = 0xffffffffu;
        if (eid < 0 || static_cast<uint32_t>(eid) >= moe) {
            eid_invalid = true;
        } else {
            uint32_t dest = 0u;
            uint32_t le = 0u;
            PlSplitExpertId(static_cast<uint32_t>(eid), epp, dest, le);
            if (dest < wc) {
                const uint32_t ordinal = local_count[dest * 8u + le];
                packed = (ordinal << 8u) | static_cast<uint32_t>(eid);
                local_count[dest * 8u + le] += 1u;
                ++total_routes;
            }
        }
        if (packed_route_wire) {
            route_packed[route_idx] = packed;
        }
        if (eid_invalid) {
            break;
        }
    }
    if (eid_invalid) {
        __gm__ PlFullOutputDoneLine *err =
            reinterpret_cast<__gm__ PlFullOutputDoneLine *>(sym + cfg->full_output_done_off);
        err->error_code = 16u; // invalid expert id during source count
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(err));
        wtr->error_code = 16u;
    }

    // P5.4 destination CSR compiler.  The first pass above already produced
    // exact destination/expert counts and canonical source ordinals.  A
    // second linear pass writes each route once into its destination slice;
    // this replaces W repeated full scans on the paired INC.
    uint32_t csr_bytes = 0u;
    uint32_t csr_offsets_bytes = 0u;
    uint64_t csr_offsets_off = 0u;
    if (destination_csr && !eid_invalid) {
        csr_offsets_off =
            PlDevDestinationCsrOffsetsOff(view.metadata_bytes);
        __gm__ uint32_t *csr_offsets =
            reinterpret_cast<__gm__ uint32_t *>(sym + csr_offsets_off);
        __gm__ PlDestinationCsrEntry *csr_entries =
            reinterpret_cast<__gm__ PlDestinationCsrEntry *>(
                sym + kPlIncRawRouteOrdinalOff);
        uint32_t cursor[kPlMaxSources * 8u];
        uint32_t prefix = 0u;
        for (uint32_t dest = 0u; dest < wc; ++dest) {
            for (uint32_t le = 0u; le < epp; ++le) {
                const uint32_t cell = dest * 8u + le;
                csr_offsets[cell] = prefix;
                cursor[cell] = prefix;
                prefix += local_count[cell];
            }
        }
        csr_offsets[wc * 8u] = prefix;

        for (uint32_t route_idx = 0u; route_idx < route_slots;
             ++route_idx) {
            const uint32_t packed = route_packed[route_idx];
            if (packed == 0xffffffffu) {
                continue;
            }
            const uint32_t eid = packed & 0xffu;
            uint32_t dest = 0u;
            uint32_t local_expert = 0u;
            PlSplitExpertId(eid, epp, dest, local_expert);
            if (dest >= wc) {
                continue;
            }
            __gm__ PlDestinationCsrEntry *entry =
                csr_entries + cursor[dest * 8u + local_expert]++;
            entry->source_token_id =
                static_cast<uint16_t>(route_idx / topk);
            entry->topk_slot =
                static_cast<uint8_t>(route_idx % topk);
            entry->expert_id = static_cast<uint8_t>(eid);
            entry->source_segment_offset = packed >> 8u;
        }
        csr_offsets_bytes = (wc * 8u + 1u) * sizeof(uint32_t);
        csr_bytes = prefix * sizeof(PlDestinationCsrEntry);
        Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(csr_offsets),
                     csr_offsets_bytes);
        if (csr_bytes > 0u) {
            Qv2DcciRange(
                reinterpret_cast<__gm__ uint8_t *>(csr_entries),
                csr_bytes);
        }
    }

    // tokens=0：仍写全零 PlRouteCountLine（禁止 stale matrix 冒充本 epoch）
    for (uint32_t dest = 0; dest < wc; ++dest) {
        uint32_t lane_total = 0u;
        for (uint32_t le = 0; le < epp; ++le) {
            const uint32_t c = local_count[dest * 8u + le];
            lane_total += c;
            matrix_checksum ^= PlCountLineChecksum(source, dest, le, c, epoch);
            PlFillRouteCountLine(sym, PlDevRouteCountLineOff(source, dest, le), dest, le, source, c, epoch);
        }
        __gm__ PlLaneWorkLine *lw =
            reinterpret_cast<__gm__ PlLaneWorkLine *>(sym + cfg->lane_work_off + static_cast<uint64_t>(dest) * 64u);
        lw->token_count = lane_total;
        lw->epoch = epoch;
        lw->lane_id = dest;
        lw->gather_count = 0u;
        lw->upload_count = 0u;
        lw->done_epoch = 0u;
        lw->error_code = 0u;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(lw));
    }
    wtr->stage = kPlCountStageWorkerMatrixReady;
    wtr->total_routes = total_routes;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(wtr));

    uint64_t mte_waits = 0u;
    const uint64_t blk_off = PlDevRouteCountLineOff(source, 0u, 0u);
    const uint32_t blk_bytes = wc * epp * static_cast<uint32_t>(sizeof(PlRouteCountLine));
    Qv2IssuePayloadRangeP6NoDrain(sym, blk_off, blk_off, blk_bytes, inc_peer, mte_waits, mte_waits);
    if (source_major_raw && view.metadata_bytes > 0u) {
        __gm__ uint32_t *wire = route_packed;
        const uint64_t wire_off = kPlIncRawRouteMetaOff;
        // The MTE transport has a 16-KiB contiguous-window boundary.  A
        // single larger ordinal put can alias the first cache lines of the
        // next window (K2/T4096 reproduced token 2048..2079 overwriting
        // token 0..31 ordinals).  Publish independently completed windows;
        // K1 remains one 16-KiB packed transfer.
        constexpr uint32_t kRouteWirePutWindow = 16u * 1024u;
        uint32_t wire_done = 0u;
        while (wire_done < view.metadata_bytes) {
            uint32_t wire_chunk = view.metadata_bytes - wire_done;
            if (wire_chunk > kRouteWirePutWindow) {
                wire_chunk = kRouteWirePutWindow;
            }
            __gm__ uint8_t *wire_src =
                reinterpret_cast<__gm__ uint8_t *>(wire) + wire_done;
            Qv2DcciRange(wire_src, wire_chunk);
            aclshmem_putmem_nbi(sym + wire_off + wire_done, wire_src,
                                wire_chunk, inc_peer);
            wire_done += wire_chunk;
            // Complete only between transport windows.  The final window is
            // intentionally batched with lane_work and drained by the common
            // quiet below; this preserves the K1 startup pipeline.
            if (wire_done < view.metadata_bytes) {
                aclshmem_quiet();
            }
        }
    }
    if (destination_csr) {
        // Publish offsets and entries before RawMetaReady.  Keep transfers
        // within the proven 16-KiB transport window.
        aclshmem_putmem_nbi(
            sym + csr_offsets_off,
            reinterpret_cast<__gm__ uint8_t *>(sym + csr_offsets_off),
            csr_offsets_bytes, inc_peer);
        constexpr uint32_t kCsrPutWindow = 16u * 1024u;
        uint32_t csr_done = 0u;
        while (csr_done < csr_bytes) {
            uint32_t csr_chunk = csr_bytes - csr_done;
            if (csr_chunk > kCsrPutWindow) {
                csr_chunk = kCsrPutWindow;
            }
            aclshmem_putmem_nbi(
                sym + kPlIncRawRouteOrdinalOff + csr_done,
                reinterpret_cast<__gm__ uint8_t *>(
                    sym + kPlIncRawRouteOrdinalOff + csr_done),
                csr_chunk, inc_peer);
            csr_done += csr_chunk;
            if (csr_done < csr_bytes) {
                aclshmem_quiet();
            }
        }
    }
    // lane_work 整块
    Qv2IssuePayloadRangeP6NoDrain(sym, cfg->lane_work_off, cfg->lane_work_off, wc * 64u, inc_peer, mte_waits,
                                  mte_waits);
    aclshmem_quiet();
    wtr->stage = kPlCountStageWorkerMatrixQuiet;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(wtr));

    // 数据 quiet 后单独 push doorbell
    __gm__ PlSourceCountReadyLine *rdy =
        reinterpret_cast<__gm__ PlSourceCountReadyLine *>(sym + PlDevSourceCountReadyLineOff(source));
    rdy->epoch = epoch;
    rdy->source_rank = source;
    rdy->total_routes = total_routes;
    rdy->checksum = matrix_checksum;
    rdy->magic = kPlCountReadyMagic;
    rdy->ready_epoch = epoch;
    rdy->visible_signal = 0;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(rdy));
    // H8.1：负例故意不发 SourceCountReady doorbell（本地线可写，远端 signal 省略）
    __gm__ PlTimingNegLine *neg = reinterpret_cast<__gm__ PlTimingNegLine *>(sym + kPlTimingNegOff);
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(neg));
    const bool drop_ready = (neg->magic == kPlTimingNegMagic && neg->drop_source_count_ready != 0u &&
                             neg->drop_source_rank == source &&
                             static_cast<uint64_t>(neg->drop_source_epoch) == epoch);
    if (!drop_ready) {
        const uint64_t rdy_off = PlDevSourceCountReadyLineOff(source);
        __gm__ uint8_t *rdy_dst = sym + rdy_off;
        __gm__ int32_t *sig_dst =
            reinterpret_cast<__gm__ int32_t *>(rdy_dst + kPlSourceCountReadySignalOff);
        aclshmem_putmem_nbi(rdy_dst, reinterpret_cast<__gm__ void *>(rdy), sizeof(PlSourceCountReadyLine), inc_peer);
        aclshmem_putmem_signal_nbi(sig_dst, &rdy->visible_signal, 0, sig_dst, static_cast<int32_t>(epoch),
                                   ACLSHMEM_SIGNAL_SET, inc_peer);
        aclshmem_quiet();
    }
    wtr->stage = kPlCountStageWorkerSourceReady;
    wtr->exit_cycle = AscendC::GetSystemCycle();
    wtr->observed_epoch = static_cast<uint32_t>(epoch);
    wtr->source_publish_owner_block = AscendC::GetBlockIdx();
    wtr->source_publish_seq = (wtr->source_publish_seq == 0xffffffffu) ? 1u : (wtr->source_publish_seq + 1u);
    if (drop_ready) {
        wtr->error_code = 21u; // 负例注入标记（非 INC timeout）；INC 侧仍应报 error=11
    }
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(wtr));
    (void)cfg;
}

// INC forward：只等 Worker push 的 lane_work（勿与 8 lane 同刷 SourceCountReady，否则 quiet 被 Dcci 风暴拖到秒级）
__aicore__ inline uint32_t PlIncEpochTokensExpected(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                     uint32_t dest_lane, uint64_t epoch)
{
    __gm__ PlFullDispatchConfig *cfg =
        reinterpret_cast<__gm__ PlFullDispatchConfig *>(sym + kPlFullDispatchConfigOff);
    __gm__ PlLaneWorkLine *lw = reinterpret_cast<__gm__ PlLaneWorkLine *>(
        sym + cfg->lane_work_off + static_cast<uint64_t>(dest_lane) * 64u);
    // 等 Worker lane_work；疏 Dcci + 2s 墙钟，避免假空 lane / 假超时
    constexpr uint64_t kLaneWorkWaitUs = 2000000ull;
    const uint64_t t0 = AscendC::GetSystemCycle();
    uint32_t spins = 0u;
    while (!PlWaitBudgetExceeded(t0, kLaneWorkWaitUs)) {
        PlSpinWaitLineDcci(reinterpret_cast<__gm__ uint8_t *>(lw), spins);
        if (lw->epoch == epoch) {
            return lw->token_count;
        }
        ++spins;
    }
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(lw));
    return (lw->epoch == epoch) ? lw->token_count : 0u;
}

// INC：校验 SourceCountReady 后，按 destination 做 range put + 独立 doorbell（唯一 remote writer）
// 返回 PlCountForwardStatus；调用方必须检查，禁止失败后 last_epoch=run_epoch 静默续跑
__aicore__ inline uint32_t PlFullDispatchIncForwardCounts(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                           __gm__ PlFullDispatchConfig *cfg, uint64_t epoch)
{
    const uint32_t epp = desc->expert_per_pe == 0u ? 8u : desc->expert_per_pe;
    const uint32_t source = desc->pair_id;
    const uint32_t wc = desc->worker_count;

    __gm__ PlIncSourceCountTrace *itr =
        reinterpret_cast<__gm__ PlIncSourceCountTrace *>(sym + PlDevIncSourceCountTraceOff(source));
    itr->epoch = epoch;
    itr->generation = static_cast<uint32_t>(epoch);
    itr->stage = kPlCountStageIncEnter;
    itr->source_rank = source;
    itr->destination_publish_mask = 0u;
    itr->source_ready_seen = 0u;
    itr->error_code = 0u;
    itr->enter_cycle = AscendC::GetSystemCycle();
    itr->exit_cycle = 0u;
    itr->observed_epoch = 0u;
    itr->observed_magic = 0u;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(itr));

    __gm__ PlSourceCountReadyLine *rdy =
        reinterpret_cast<__gm__ PlSourceCountReadyLine *>(sym + PlDevSourceCountReadyLineOff(source));
    __gm__ int32_t *src_sig =
        reinterpret_cast<__gm__ int32_t *>(reinterpret_cast<__gm__ uint8_t *>(rdy) + kPlSourceCountReadySignalOff);
    constexpr uint64_t kSrcReadyWaitUs = 800000ull; // 0.8s：NEG fail-closed 须在数秒内出 error=11
    const uint64_t t0 = AscendC::GetSystemCycle();
    uint32_t spins = 0u;
    bool ready_ok = false;
    while (!PlWaitBudgetExceeded(t0, kSrcReadyWaitUs)) {
        if (aclshmem_int32_test(src_sig, ACLSHMEM_CMP_EQ, static_cast<int32_t>(epoch)) != 0) {
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(rdy));
            if (rdy->magic == kPlCountReadyMagic && rdy->epoch == epoch && rdy->ready_epoch == epoch &&
                rdy->source_rank == source) {
                ready_ok = true;
                break;
            }
        }
        ++spins;
    }
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(rdy));
    itr->observed_epoch = static_cast<uint32_t>(rdy->ready_epoch);
    itr->observed_magic = rdy->magic;
    if (!ready_ok || rdy->ready_epoch != epoch || rdy->magic != kPlCountReadyMagic || rdy->epoch != epoch ||
        rdy->source_rank != source) {
        __gm__ PlFullOutputDoneLine *err =
            reinterpret_cast<__gm__ PlFullOutputDoneLine *>(sym + cfg->full_output_done_off);
        err->error_code = kPlCountFwdSourceReadyTimeout;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(err));
        itr->error_code = kPlCountFwdSourceReadyTimeout;
        itr->exit_cycle = AscendC::GetSystemCycle();
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(itr));
        return kPlCountFwdSourceReadyTimeout;
    }
    itr->source_ready_seen = 1u;
    itr->stage = kPlCountStageIncSourceReadySeen;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(itr));

    uint32_t got_cs = 0u;
    uint32_t total = 0u;
    for (uint32_t dest = 0; dest < wc; ++dest) {
        for (uint32_t le = 0; le < epp; ++le) {
            __gm__ PlRouteCountLine *line =
                reinterpret_cast<__gm__ PlRouteCountLine *>(sym + PlDevRouteCountLineOff(source, dest, le));
            if (!PlValidateRouteCountLine(line, epoch, source, dest, le)) {
                __gm__ PlFullOutputDoneLine *err =
                    reinterpret_cast<__gm__ PlFullOutputDoneLine *>(sym + cfg->full_output_done_off);
                err->error_code = kPlCountFwdSourceIdentityError;
                PlDcci(reinterpret_cast<__gm__ uint8_t *>(err));
                itr->error_code = kPlCountFwdSourceIdentityError;
                itr->exit_cycle = AscendC::GetSystemCycle();
                PlDcci(reinterpret_cast<__gm__ uint8_t *>(itr));
                return kPlCountFwdSourceIdentityError;
            }
            got_cs ^= line->checksum;
            total += line->count;
        }
    }
    if (got_cs != rdy->checksum || total != rdy->total_routes) {
        __gm__ PlFullOutputDoneLine *err =
            reinterpret_cast<__gm__ PlFullOutputDoneLine *>(sym + cfg->full_output_done_off);
        err->error_code = kPlCountFwdCountMatrixError;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(err));
        itr->error_code = kPlCountFwdCountMatrixError;
        itr->exit_cycle = AscendC::GetSystemCycle();
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(itr));
        return kPlCountFwdCountMatrixError;
    }
    itr->stage = kPlCountStageIncMatrixOk;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(itr));

    if ((desc->pipeline_mode & kPlModeDirectCountExchange) != 0u) {
        itr->destination_publish_mask = (wc >= 32u) ? 0xffffffffu : ((1u << wc) - 1u);
        itr->stage = kPlCountStageIncDestPublished;
        itr->exit_cycle = AscendC::GetSystemCycle();
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(itr));
        return kPlCountFwdOk;
    }

    // 含 zero-token：仍向 8 dest 发布 slice_total=0 的 DestCountSliceReady
    uint32_t publish_mask = 0u;
    // 阶段 1：全部 dest 只发 slice range put（nbi），不 per-dest quiet
    for (uint32_t dest = 0; dest < wc; ++dest) {
        const int dest_worker_pe = static_cast<int>(dest);
        const uint64_t slice_off = PlDevRouteCountLineOff(source, dest, 0u);
        const uint32_t slice_bytes = epp * static_cast<uint32_t>(sizeof(PlRouteCountLine));
        uint64_t mte_waits = 0u;
        Qv2IssuePayloadRangeP6NoDrain(sym, slice_off, slice_off, slice_bytes, dest_worker_pe, mte_waits, mte_waits);
    }
    aclshmem_quiet();
    // 阶段 2：全部 dest 写 local DestCountSliceReady + doorbell put，再一次 quiet
    for (uint32_t dest = 0; dest < wc; ++dest) {
        const int dest_worker_pe = static_cast<int>(dest);
        uint32_t slice_total = 0u;
        uint32_t slice_cs = 0u;
        for (uint32_t le = 0; le < epp; ++le) {
            __gm__ PlRouteCountLine *line =
                reinterpret_cast<__gm__ PlRouteCountLine *>(sym + PlDevRouteCountLineOff(source, dest, le));
            slice_total += line->count;
            slice_cs ^= line->checksum;
        }
        __gm__ PlDestCountSliceReadyLine *srdy = reinterpret_cast<__gm__ PlDestCountSliceReadyLine *>(
            sym + PlDevDestCountSliceReadyLineOff(source, dest));
        srdy->epoch = epoch;
        srdy->source_rank = source;
        srdy->destination_rank = dest;
        srdy->total_routes = slice_total;
        srdy->checksum = slice_cs;
        for (uint32_t le = 0u; le < 8u; ++le) {
            srdy->expert_counts[le] =
                (le < epp) ? static_cast<uint16_t>(
                    reinterpret_cast<__gm__ PlRouteCountLine *>(
                        sym + PlDevRouteCountLineOff(source, dest, le))->count) : 0u;
        }
        srdy->magic = kPlCountReadyMagic;
        srdy->ready_epoch = epoch;
        srdy->visible_signal = 0;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(srdy));
        const uint64_t srdy_off = PlDevDestCountSliceReadyLineOff(source, dest);
        __gm__ uint8_t *srdy_dst = sym + srdy_off;
        __gm__ int32_t *sig_dst =
            reinterpret_cast<__gm__ int32_t *>(srdy_dst + kPlDestCountSliceReadySignalOff);
        aclshmem_putmem_nbi(srdy_dst, reinterpret_cast<__gm__ void *>(srdy), sizeof(PlDestCountSliceReadyLine),
                            dest_worker_pe);
        aclshmem_putmem_signal_nbi(sig_dst, &srdy->visible_signal, 0, sig_dst, static_cast<int32_t>(epoch),
                                   ACLSHMEM_SIGNAL_SET, dest_worker_pe);
        publish_mask |= (1u << dest);
        itr->destination_publish_mask = publish_mask;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(itr));
    }
    aclshmem_quiet();
    itr->stage = kPlCountStageIncDestPublished;
    itr->exit_cycle = AscendC::GetSystemCycle();
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(itr));
    return kPlCountFwdOk;
}

// destination：round-robin 等全部 source（单全局 deadline，消 HOL）；失败写 missing_source_mask
__aicore__ inline void PlFullDispatchDestCountPrefix(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                      __gm__ PlFullDispatchConfig *cfg, uint64_t epoch)
{
    __gm__ PlRouteTimingLine *rt =
        reinterpret_cast<__gm__ PlRouteTimingLine *>(sym + cfg->route_timing_off);
    __gm__ PlFullOutputDoneLine *done =
        reinterpret_cast<__gm__ PlFullOutputDoneLine *>(sym + cfg->full_output_done_off);
    const uint32_t dest = desc->pair_id;
    const uint32_t epp = desc->expert_per_pe == 0u ? 8u : desc->expert_per_pe;
    const uint32_t wc = desc->worker_count;
    const uint32_t expect_mask = (wc >= 32u) ? 0xffffffffu : ((1u << wc) - 1u);

    __gm__ PlDestinationCountTrace *dtr =
        reinterpret_cast<__gm__ PlDestinationCountTrace *>(sym + PlDevDestinationCountTraceOff(dest));
    dtr->epoch = epoch;
    dtr->generation = static_cast<uint32_t>(epoch);
    dtr->stage = kPlCountStageDestWaitEnter;
    dtr->destination_rank = dest;
    dtr->source_ready_mask = 0u;
    dtr->missing_source_mask = expect_mask;
    dtr->error_code = 0u;
    dtr->enter_cycle = AscendC::GetSystemCycle();
    dtr->exit_cycle = 0u;
    dtr->observed_epoch = 0u;
    dtr->wait_done_cycle = 0u;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(dtr));

    PlFullDispatchResetTiming(rt, epoch);

    constexpr uint64_t kDestAllSourcesWaitUs = 2000000ull;
    const uint64_t t0 = AscendC::GetSystemCycle();
    uint32_t ready_mask = 0u;
    uint32_t spins = 0u;
    while (ready_mask != expect_mask && !PlWaitBudgetExceeded(t0, kDestAllSourcesWaitUs)) {
        for (uint32_t s = 0; s < wc; ++s) {
            if ((ready_mask & (1u << s)) != 0u) {
                continue;
            }
            __gm__ PlDestCountSliceReadyLine *srdy =
                reinterpret_cast<__gm__ PlDestCountSliceReadyLine *>(sym + PlDevDestCountSliceReadyLineOff(s, dest));
            __gm__ int32_t *slice_sig = reinterpret_cast<__gm__ int32_t *>(
                reinterpret_cast<__gm__ uint8_t *>(srdy) + kPlDestCountSliceReadySignalOff);
            if (aclshmem_int32_test(slice_sig, ACLSHMEM_CMP_EQ, static_cast<int32_t>(epoch)) != 0) {
                PlDcci(reinterpret_cast<__gm__ uint8_t *>(srdy));
                if (srdy->magic == kPlCountReadyMagic && srdy->epoch == epoch && srdy->ready_epoch == epoch &&
                    srdy->source_rank == s && srdy->destination_rank == dest) {
                    ready_mask |= (1u << s);
                }
            }
        }
        dtr->source_ready_mask = ready_mask;
        dtr->missing_source_mask = expect_mask & ~ready_mask;
        if ((spins & 255u) == 0u) {
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(dtr));
        }
        ++spins;
    }
    dtr->source_ready_mask = ready_mask;
    dtr->missing_source_mask = expect_mask & ~ready_mask;
    dtr->wait_done_cycle = AscendC::GetSystemCycle();
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(dtr));

    if (ready_mask != expect_mask) {
        done->error_code = 10u;
        done->missing_source_mask = expect_mask & ~ready_mask;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(done));
        dtr->error_code = 10u;
        dtr->exit_cycle = dtr->wait_done_cycle;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(dtr));
        return;
    }
    dtr->stage = kPlCountStageDestAllSlices;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(dtr));

    for (uint32_t s = 0; s < wc; ++s) {
        __gm__ PlDestCountSliceReadyLine *srdy =
            reinterpret_cast<__gm__ PlDestCountSliceReadyLine *>(sym + PlDevDestCountSliceReadyLineOff(s, dest));
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(srdy));
        uint32_t slice_cs = 0u;
        uint32_t slice_total = 0u;
        const bool direct_count = (desc->pipeline_mode & kPlModeDirectCountExchange) != 0u;
        for (uint32_t le = 0; le < epp; ++le) {
            uint32_t count = 0u;
            if (direct_count) {
                count = static_cast<uint32_t>(srdy->expert_counts[le]);
                slice_cs ^= PlCountLineChecksum(s, dest, le, count, epoch);
            } else {
                __gm__ PlRouteCountLine *line =
                    reinterpret_cast<__gm__ PlRouteCountLine *>(sym + PlDevRouteCountLineOff(s, dest, le));
                if (!PlValidateRouteCountLine(line, epoch, s, dest, le)) {
                    done->error_code = 14u;
                    done->missing_source_mask = (1u << s);
                    PlDcci(reinterpret_cast<__gm__ uint8_t *>(done));
                    dtr->error_code = 14u;
                    dtr->exit_cycle = AscendC::GetSystemCycle();
                    PlDcci(reinterpret_cast<__gm__ uint8_t *>(dtr));
                    return;
                }
                count = line->count;
                slice_cs ^= line->checksum;
            }
            slice_total += count;
        }
        if (slice_cs != srdy->checksum || slice_total != srdy->total_routes) {
            done->error_code = 15u;
            done->missing_source_mask = (1u << s);
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(done));
            dtr->error_code = 15u;
            dtr->exit_cycle = AscendC::GetSystemCycle();
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(dtr));
            return;
        }
    }

    __gm__ int32_t *expert_nums = PlFullDispatchExpertTokenNumsPtr(sym, cfg, epoch);
    __gm__ int32_t *ep_recv = PlFullDispatchEpRecvCountPtr(sym, cfg, epoch);
    __gm__ int32_t *seg_base = reinterpret_cast<__gm__ int32_t *>(sym + cfg->segment_base_off);
    __gm__ int32_t *expected = reinterpret_cast<__gm__ int32_t *>(sym + kPlExpectedCountOff);

    uint32_t global_running = 0u;
    const bool direct_count = (desc->pipeline_mode & kPlModeDirectCountExchange) != 0u;
    for (uint32_t le = 0; le < epp; ++le) {
        uint32_t expert_total = 0u;
        for (uint32_t s = 0; s < wc; ++s) {
            uint32_t c = 0u;
            if (direct_count) {
                __gm__ PlDestCountSliceReadyLine *srdy =
                    reinterpret_cast<__gm__ PlDestCountSliceReadyLine *>(
                        sym + PlDevDestCountSliceReadyLineOff(s, dest));
                c = static_cast<uint32_t>(srdy->expert_counts[le]);
            } else {
                __gm__ PlRouteCountLine *line =
                    reinterpret_cast<__gm__ PlRouteCountLine *>(sym + PlDevRouteCountLineOff(s, dest, le));
                c = line->count;
            }
            seg_base[PlDevSegmentBaseIndex(le, s, wc)] = static_cast<int32_t>(global_running);
            global_running += c;
            expert_total += c;
            ep_recv[le * wc + s] = static_cast<int32_t>(global_running);
        }
        expert_nums[le] = static_cast<int32_t>(expert_total);
    }
    for (uint32_t s = 0; s < wc; ++s) {
        uint32_t src_total = 0u;
        for (uint32_t le = 0; le < epp; ++le) {
            if (direct_count) {
                __gm__ PlDestCountSliceReadyLine *srdy =
                    reinterpret_cast<__gm__ PlDestCountSliceReadyLine *>(
                        sym + PlDevDestCountSliceReadyLineOff(s, dest));
                src_total += static_cast<uint32_t>(srdy->expert_counts[le]);
            } else {
                __gm__ PlRouteCountLine *line =
                    reinterpret_cast<__gm__ PlRouteCountLine *>(sym + PlDevRouteCountLineOff(s, dest, le));
                src_total += line->count;
            }
        }
        expected[s] = static_cast<int32_t>(src_total);
    }
    Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(expert_nums), epp * sizeof(int32_t));
    Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(ep_recv), epp * wc * sizeof(int32_t));
    Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(seg_base), epp * wc * sizeof(int32_t));
    Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(expected), wc * sizeof(int32_t));
    dtr->stage = kPlCountStageDestPrefixDone;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(dtr));

    const bool receiver_final_slot = (desc->pipeline_mode & kPlModeReceiverFinalSlot) != 0u;
    __gm__ int32_t *out_local = reinterpret_cast<__gm__ int32_t *>(sym + kPlOutboundSegBaseOff);
    uint32_t seg_checksum[kPlMaxSources];

    // 两阶段 quiet：先 batch 全部 remote segment_base put，再一次 quiet；再 per-source doorbell+signal
    uint64_t seg_mte_waits = 0u;
    for (uint32_t s = 0; s < wc && !receiver_final_slot; ++s) {
        uint32_t seg_cs = 0u;
        __gm__ int32_t *stage =
            reinterpret_cast<__gm__ int32_t *>(sym + PlDevSegBasePublishStagingOff(s));
        for (uint32_t le = 0; le < epp; ++le) {
            const int32_t base = seg_base[PlDevSegmentBaseIndex(le, s, wc)];
            stage[le] = base;
            seg_cs ^= static_cast<uint32_t>(base) ^ (le << 8) ^ (dest << 16) ^ static_cast<uint32_t>(epoch);
        }
        for (uint32_t pad_i = epp; pad_i < 16u; ++pad_i) {
            stage[pad_i] = 0;
        }
        Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(stage), 64u);
        seg_checksum[s] = seg_cs;
        if (s == dest) {
            for (uint32_t i = 0; i < 16u; ++i) {
                out_local[PlDevOutboundSegBaseIndex(dest, i)] = stage[i];
            }
            Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(out_local + PlDevOutboundSegBaseIndex(dest, 0u)), 64u);
        } else {
            const uint64_t remote_out_off =
                kPlOutboundSegBaseOff + PlDevOutboundSegBaseIndex(dest, 0u) * sizeof(int32_t);
            aclshmem_putmem_nbi(reinterpret_cast<__gm__ void *>(sym + remote_out_off),
                                reinterpret_cast<__gm__ void *>(stage), 64u, static_cast<int>(s));
            Qv2MteUbComplete(static_cast<int>(s), seg_mte_waits);
        }
    }
    if (seg_mte_waits > 0u) {
        aclshmem_quiet();
    }

    uint64_t doorbell_mte_waits = 0u;
    for (uint32_t s = 0; s < wc && !receiver_final_slot; ++s) {
        __gm__ PlSegmentBaseReadyLine *brdy =
            reinterpret_cast<__gm__ PlSegmentBaseReadyLine *>(sym + PlDevSegmentBaseReadyLineOff(s, dest));
        brdy->epoch = epoch;
        brdy->source_rank = s;
        brdy->destination_rank = dest;
        brdy->checksum = seg_checksum[s];
        brdy->magic = kPlSegBaseReadyMagic;
        brdy->ready_epoch = epoch;
        brdy->visible_signal = static_cast<int32_t>(epoch);
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(brdy));
        const uint64_t brdy_off = PlDevSegmentBaseReadyLineOff(s, dest);
        __gm__ uint8_t *brdy_dst = sym + brdy_off;
        __gm__ int32_t *sig_dst =
            reinterpret_cast<__gm__ int32_t *>(brdy_dst + kPlSegBaseReadySignalOff);
        if (s != dest) {
            aclshmem_putmem_signal_nbi(brdy_dst, reinterpret_cast<__gm__ void *>(brdy), sizeof(PlSegmentBaseReadyLine),
                                       sig_dst, static_cast<int32_t>(epoch), ACLSHMEM_SIGNAL_SET,
                                       static_cast<int>(s));
            Qv2MteUbComplete(static_cast<int>(s), doorbell_mte_waits);
        } else {
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(brdy));
        }
    }
    if (doorbell_mte_waits > 0u) {
        aclshmem_quiet();
    }

    rt->route_count_done_cycle = AscendC::GetSystemCycle();
    rt->prefix_done_cycle = rt->route_count_done_cycle;
    done->counts_done_epoch = epoch;
    done->missing_source_mask = 0u;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(rt));
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(done));
    dtr->stage = kPlCountStageDestCountsDone;
    dtr->exit_cycle = AscendC::GetSystemCycle();
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(dtr));
}

__aicore__ inline void PlFullDispatchCountPrefix(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                   __gm__ PlFullDispatchConfig *cfg, uint64_t epoch)
{
    if (desc->worker_count <= 1u) {
        PlFullDispatchCountPrefixLocal(sym, desc, cfg, epoch);
        PlFullDispatchSourcePublishCounts(sym, desc, cfg, epoch);
        return;
    }
    PlFullDispatchDestCountPrefix(sym, desc, cfg, epoch);
}

// direct-count lane sharding：upload lane d 独占 source→destination d 的 count slice 与 doorbell。
// control AIV 只生成矩阵并通知 partner INC；8 个 destination slice 不再在 control 上串行。
__aicore__ inline bool PlWorkerPublishDirectCountSlice(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                        uint32_t dest_lane, uint64_t epoch)
{
    if ((desc->pipeline_mode & kPlModeDirectCountExchange) == 0u) {
        return true;
    }
    const uint32_t wc = desc->worker_count;
    const uint32_t source = desc->pair_id;
    const uint32_t epp = desc->expert_per_pe == 0u ? 8u : desc->expert_per_pe;
    if (dest_lane >= wc) {
        return true;
    }

    __gm__ PlSourceCountReadyLine *rdy =
        reinterpret_cast<__gm__ PlSourceCountReadyLine *>(sym + PlDevSourceCountReadyLineOff(source));
    constexpr uint64_t kLocalMatrixWaitUs = 800000ull;
    const uint64_t wait_start = AscendC::GetSystemCycle();
    uint32_t spins = 0u;
    while (!PlWaitBudgetExceeded(wait_start, kLocalMatrixWaitUs)) {
        PlSpinWaitLineDcci(reinterpret_cast<__gm__ uint8_t *>(rdy), spins);
        if (rdy->magic == kPlCountReadyMagic && rdy->epoch == epoch && rdy->ready_epoch == epoch &&
            rdy->source_rank == source) {
            break;
        }
        ++spins;
    }
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(rdy));
    if (rdy->magic != kPlCountReadyMagic || rdy->epoch != epoch || rdy->ready_epoch != epoch ||
        rdy->source_rank != source) {
        return false;
    }

    uint32_t slice_total = 0u;
    uint32_t slice_cs = 0u;
    for (uint32_t le = 0u; le < epp; ++le) {
        __gm__ PlRouteCountLine *line = reinterpret_cast<__gm__ PlRouteCountLine *>(
            sym + PlDevRouteCountLineOff(source, dest_lane, le));
        slice_total += line->count;
        slice_cs ^= line->checksum;
    }

    __gm__ PlDestCountSliceReadyLine *srdy =
        reinterpret_cast<__gm__ PlDestCountSliceReadyLine *>(
            sym + PlDevDestCountSliceReadyLineOff(source, dest_lane));
    srdy->epoch = epoch;
    srdy->source_rank = source;
    srdy->destination_rank = dest_lane;
    srdy->total_routes = slice_total;
    srdy->checksum = slice_cs;
    for (uint32_t le = 0u; le < 8u; ++le) {
        srdy->expert_counts[le] =
            (le < epp) ? static_cast<uint16_t>(
                reinterpret_cast<__gm__ PlRouteCountLine *>(
                    sym + PlDevRouteCountLineOff(source, dest_lane, le))->count) : 0u;
    }
    srdy->magic = kPlCountReadyMagic;
    srdy->ready_epoch = epoch;
    srdy->visible_signal = static_cast<int32_t>(epoch);
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(srdy));
    if (dest_lane != source) {
        const uint64_t srdy_off = PlDevDestCountSliceReadyLineOff(source, dest_lane);
        __gm__ uint8_t *srdy_dst = sym + srdy_off;
        __gm__ int32_t *slice_sig = reinterpret_cast<__gm__ int32_t *>(
            srdy_dst + kPlDestCountSliceReadySignalOff);
        aclshmem_putmem_signal_nbi(srdy_dst, reinterpret_cast<__gm__ void *>(srdy),
                                   sizeof(PlDestCountSliceReadyLine), slice_sig,
                                   static_cast<int32_t>(epoch), ACLSHMEM_SIGNAL_SET,
                                   static_cast<int>(dest_lane));
        aclshmem_quiet();
    }
    return true;
}

// 仅读本地已被 destination push 的 outbound segment_base（调用方须先 DCCI 整行一次）
__aicore__ inline uint32_t PlReadCachedOutboundSegmentBase(__gm__ int32_t *out_base, uint32_t dest_rank,
                                                            uint32_t local_expert)
{
    return static_cast<uint32_t>(out_base[PlDevOutboundSegBaseIndex(dest_rank, local_expert)]);
}

__aicore__ inline void PlWorkerRouteGatherUploadLaneEpoch(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                          __gm__ PlFullDispatchConfig *cfg, __gm__ PlUploadCounters *uc,
                                                          __gm__ PlPipelineTiming *timing, __gm__ PlServiceTraceLine *trace,
                                                          uint32_t lane_id, uint64_t epoch)
{
    (void)timing;
    const uint32_t nbytes = desc->payload_bytes;
    const uint32_t depth = desc->ring_depth;
    const uint32_t tokens = PlFullDispatchLocalTokenCount(sym, desc, cfg, epoch);
    const uint32_t epp = desc->expert_per_pe == 0u ? 8u : desc->expert_per_pe;
    const uint32_t topk = desc->route_topk == 0u ? 1u : desc->route_topk;
    const uint32_t source = desc->pair_id;
    const uint32_t dest_lane = lane_id;

    trace->epoch_enter_cycle = AscendC::GetSystemCycle();
    PlTraceDcci(trace);

    __gm__ PlUploadLaneStageTiming *stage_tm =
        reinterpret_cast<__gm__ PlUploadLaneStageTiming *>(sym + PlDevUploadLaneStageTimingOff(lane_id));
    stage_tm->epoch = epoch;
    stage_tm->magic = kPlUploadLaneStageMagic;
    stage_tm->owner_block = AscendC::GetBlockIdx();
    stage_tm->counts_wait_start = AscendC::GetSystemCycle();
    stage_tm->counts_wait_end = 0u;
    stage_tm->segment_ready_wait_start = 0u;
    stage_tm->segment_ready_wait_end = 0u;
    stage_tm->route_scan_start = 0u;
    stage_tm->route_scan_end = 0u;
    stage_tm->descriptor_build_start = 0u;
    stage_tm->descriptor_build_end = 0u;
    stage_tm->credit_wait_cycles = 0u;
    stage_tm->accepted_routes = 0u;
    stage_tm->segment_base_dcci_count = 0u;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(stage_tm));

    __gm__ PlFullOutputDoneLine *done =
        reinterpret_cast<__gm__ PlFullOutputDoneLine *>(sym + cfg->full_output_done_off);
    __gm__ PlRouteTimingLine *rt =
        reinterpret_cast<__gm__ PlRouteTimingLine *>(sym + cfg->route_timing_off);

    // M2：epoch 入口校验 workspace desc（Legacy 仍走 DestFinal；禁止 silent fallback）
    if (!PlDevRequireInvocationWorkspace(sym, cfg, epoch)) {
        return;
    }

    if (!PlWorkerPublishDirectCountSlice(sym, desc, dest_lane, epoch)) {
        done->error_code = kPlCountFwdSourceReadyTimeout;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(done));
        return;
    }

    const bool receiver_final_slot = (desc->pipeline_mode & kPlModeReceiverFinalSlot) != 0u;
    if (!receiver_final_slot) {
        uint32_t counts_spins = 0u;
        while (done->counts_done_epoch < epoch) {
            PlSpinWaitLineDcci(reinterpret_cast<__gm__ uint8_t *>(done), counts_spins);
            ++counts_spins;
        }
    }
    stage_tm->counts_wait_end = AscendC::GetSystemCycle();
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(stage_tm));

    uint32_t segment_base_cache[8];
    for (uint32_t i = 0; i < 8u; ++i) {
        segment_base_cache[i] = 0u;
    }
    __gm__ int32_t *outbound_seg =
        reinterpret_cast<__gm__ int32_t *>(sym + kPlOutboundSegBaseOff);

    if (dest_lane < desc->worker_count && !receiver_final_slot) {
        __gm__ PlSegmentBaseReadyLine *brdy = reinterpret_cast<__gm__ PlSegmentBaseReadyLine *>(
            sym + PlDevSegmentBaseReadyLineOff(source, dest_lane));
        __gm__ int32_t *seg_sig =
            reinterpret_cast<__gm__ int32_t *>(reinterpret_cast<__gm__ uint8_t *>(brdy) + kPlSegBaseReadySignalOff);
        stage_tm->segment_ready_wait_start = AscendC::GetSystemCycle();
        constexpr uint64_t kSegReadyWaitUs = 2000000ull;
        const uint64_t seg_t0 = AscendC::GetSystemCycle();
        uint32_t seg_spins = 0u;
        bool seg_ready_ok = false;
        while (!PlWaitBudgetExceeded(seg_t0, kSegReadyWaitUs)) {
            if (aclshmem_int32_test(seg_sig, ACLSHMEM_CMP_EQ, static_cast<int32_t>(epoch)) != 0) {
                PlDcci(reinterpret_cast<__gm__ uint8_t *>(brdy));
                if (brdy->magic == kPlSegBaseReadyMagic && brdy->epoch == epoch && brdy->ready_epoch == epoch &&
                    brdy->source_rank == source && brdy->destination_rank == dest_lane) {
                    seg_ready_ok = true;
                    break;
                }
            }
            PlSpinWaitSignalLineDcci(reinterpret_cast<__gm__ uint8_t *>(brdy), seg_spins);
            ++seg_spins;
        }
        stage_tm->segment_ready_wait_end = AscendC::GetSystemCycle();
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(stage_tm));
        if (!seg_ready_ok) {
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(brdy));
            done->error_code = kPlErrSegmentBaseReadyTimeout;
            done->missing_segment_dest_mask |= (1u << dest_lane);
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(done));
            return;
        }
        Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(outbound_seg + PlDevOutboundSegBaseIndex(dest_lane, 0u)), 64u);
        stage_tm->segment_base_dcci_count = 1u;
        for (uint32_t le = 0; le < epp; ++le) {
            segment_base_cache[le] =
                PlReadCachedOutboundSegmentBase(outbound_seg, dest_lane, le);
        }
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(stage_tm));
    }
    (void)rt;
    if (cfg->probe_mode == kPlFullDispatchProbeCountOnly) {
        return;
    }

    const bool gather_only = (cfg->probe_mode == kPlFullDispatchProbeGatherOnly);
    (void)gather_only;

    const uint64_t eid_base = PlFullDispatchRawExpertBase(cfg, desc, sym, epoch);
    const uint64_t raw_base = PlFullDispatchRawInputBase(cfg, desc, sym, epoch);
    // M4：Raw*Base 已是绝对 GM 指针（aclrtMalloc / workspace），禁止再加 sym
    __gm__ int32_t *eids = reinterpret_cast<__gm__ int32_t *>(eid_base);
    if (tokens > 0u) {
        Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(eids), tokens * topk * sizeof(int32_t));
    }

    const uint64_t transport_seq_base = uc->local_tail;
    uint64_t gather_seq = 0u;
    const bool force_stale_gen = (cfg->version == 2u);

    // per-lane staging：lane d = destination d
    // gather_payload 仍按 ring_slot（gather_only）；upload 有序 desc 写到 sym staging（lane×kPlMaxTokens）
    const uint64_t lane_gather_payload =
        cfg->gather_payload_off + static_cast<uint64_t>(lane_id) * depth * nbytes;
    // ring desc：Worker local（gather_only / host verify）；有序 desc：
    // expanded worker-local per-lane staging（upload put source）。
    const uint64_t lane_gather_desc_ring =
        cfg->gather_desc_off + static_cast<uint64_t>(lane_id) * depth * kPlDescriptorBytes;
    const uint64_t lane_upload_desc_ord =
        cfg->final_payload_off + DevPlWorkerDescLaneRelOff(lane_id);

    // H9-P8 单因素：lane 内 descriptor 按 local_expert 连续布局（非 token-major）
    // 使 destination_final_slot + lane_sequence 同向连续，第二跳 range put 才能聚成 batch≈8
    uint32_t ordinal[8];
    uint32_t lane_local_count[8];
    uint32_t lane_local_base[8];
    for (uint32_t i = 0; i < 8u; ++i) {
        ordinal[i] = 0u;
        lane_local_count[i] = 0u;
        lane_local_base[i] = 0u;
    }

    if (dest_lane >= desc->worker_count) {
        return;
    }

    // 本 lane 无路由则跳过 gather/upload
    __gm__ PlLaneWorkLine *lane_wk =
        reinterpret_cast<__gm__ PlLaneWorkLine *>(sym + cfg->lane_work_off + static_cast<uint64_t>(dest_lane) * 64u);
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(lane_wk));
    if (lane_wk->token_count == 0u) {
        stage_tm->accepted_routes = 0u;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(stage_tm));
        return;
    }

    stage_tm->route_scan_start = AscendC::GetSystemCycle();
    stage_tm->descriptor_build_start = stage_tm->route_scan_start;

    // Count control has already scanned the route tensor and published a
    // source×destination×local-expert matrix before this lane is released.
    // Reuse that immutable matrix instead of rescanning T×K once per
    // destination merely to reconstruct the same eight counters.
    for (uint32_t le = 0u; le < epp && le < 8u; ++le) {
        __gm__ PlRouteCountLine *count_line =
            reinterpret_cast<__gm__ PlRouteCountLine *>(
                sym + PlDevRouteCountLineOff(source, dest_lane, le));
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(count_line));
        lane_local_count[le] = count_line->count;
    }
    for (uint32_t le = 1; le < epp && le < 8u; ++le) {
        lane_local_base[le] = lane_local_base[le - 1u] + lane_local_count[le - 1u];
    }
    uint32_t total_lane_routes = 0u;
    for (uint32_t le = 0; le < epp && le < 8u; ++le) {
        total_lane_routes += lane_local_count[le];
    }
    (void)total_lane_routes; // 与 gather_seq 应对齐；验收靠 four-output / avg_batch

    // token-major 扫描，但按 local_expert 前缀写入 ordered_rel（descriptor-order；无 tile fused upload）
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t slot = 0; slot < topk; ++slot) {
            const int32_t eid = eids[static_cast<uint64_t>(t) * topk + slot];
            const uint32_t dest = static_cast<uint32_t>(eid) / epp;
            if (dest != dest_lane) {
                continue;
            }
            const uint32_t le = static_cast<uint32_t>(eid) % epp;
            if (rt->gather_first_cycle == 0u) {
                rt->gather_first_cycle = AscendC::GetSystemCycle();
            }
            const uint32_t seg_off = ordinal[le];
            const uint32_t ordered_rel = lane_local_base[le] + seg_off;
            ++ordinal[le];
            if (static_cast<uint64_t>(ordered_rel) >= kPlWorkerDescLaneStride) {
                done->error_code = kPlErrOutputCapacityInsufficient;
                PlDcci(reinterpret_cast<__gm__ uint8_t *>(done));
                return;
            }
            const uint32_t base = receiver_final_slot ? 0u : segment_base_cache[le];
            const uint32_t final_slot = base + seg_off;
            // lane_sequence 与 final_slot 同向连续 → INC range coalesce 可跨多个 token
            const uint64_t lane_seq = transport_seq_base + static_cast<uint64_t>(ordered_rel);
            const uint32_t ring_slot = static_cast<uint32_t>(lane_seq % depth);

            if (gather_only) {
                const uint64_t gather_payload = lane_gather_payload + static_cast<uint64_t>(ring_slot) * nbytes;
                const uint64_t raw_src = raw_base + static_cast<uint64_t>(t) * nbytes;
                PlCopyGmBytes(reinterpret_cast<__gm__ uint8_t *>(gather_payload),
                              reinterpret_cast<__gm__ uint8_t *>(raw_src), nbytes);
                Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(gather_payload), nbytes);
            }

            __gm__ PlDescriptor *gd = reinterpret_cast<__gm__ PlDescriptor *>(lane_gather_desc_ring +
                                                                               static_cast<uint64_t>(ring_slot) *
                                                                                   kPlDescriptorBytes);
            gd->epoch = epoch;
            gd->generation = (force_stale_gen && epoch >= 2u) ? 1u : static_cast<uint32_t>(epoch);
            gd->lane_sequence = lane_seq;
            gd->source_rank = source;
            gd->source_token_id = t;
            gd->topk_slot = slot;
            gd->expert_id = static_cast<uint32_t>(eid);
            gd->destination_rank = dest;
            gd->local_expert_id = le;
            gd->source_segment_offset = seg_off;
            gd->destination_final_slot = final_slot;
            gd->destination_slot = final_slot;
            gd->payload_bytes = nbytes;
            gd->source_lane = lane_id;
            gd->ring_slot = ring_slot;
            gd->token_sequence = source * tokens + t + 1u;
            gd->checksum_seed = 0u;
            Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(gd), kPlDescriptorBytes);

            // 有序镜像：Worker-local per-lane staging，同时供 verify 与 upload put。
            __gm__ PlDescriptor *wd = reinterpret_cast<__gm__ PlDescriptor *>(
                lane_upload_desc_ord + static_cast<uint64_t>(ordered_rel) * kPlDescriptorBytes);
            PlCopyGmBytes(reinterpret_cast<__gm__ uint8_t *>(wd), reinterpret_cast<__gm__ uint8_t *>(gd),
                          kPlDescriptorBytes);
            Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(wd), kPlDescriptorBytes);

            ++gather_seq;
        }
    }

    stage_tm->route_scan_end = AscendC::GetSystemCycle();
    stage_tm->descriptor_build_end = stage_tm->route_scan_end;
    stage_tm->accepted_routes = static_cast<uint32_t>(gather_seq);
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(stage_tm));

    PlDcciUploadCtr(uc);

    // gather_done 仅本 lane 诊断；stage1 非正式字段保持 0
    rt->gather_done_cycle = AscendC::GetSystemCycle();
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(rt));

    __gm__ PlLaneWorkLine *lw =
        reinterpret_cast<__gm__ PlLaneWorkLine *>(sym + cfg->lane_work_off + static_cast<uint64_t>(lane_id) * 64u);
    lw->gather_count = static_cast<uint32_t>(gather_seq);
    // descriptor-order：upload 由后续 PlWorkerUploadLaneEpoch 完成
    lw->upload_count = 0u;
    lw->done_epoch = epoch;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(lw));

    const uint64_t done_cyc = AscendC::GetSystemCycle();
    const uint64_t first_cyc = trace->first_issue_cycle > 0 ? trace->first_issue_cycle : trace->epoch_enter_cycle;
    PlPublishAivEpochTiming(sym, DevPlUploadAivTimingOff(lane_id), first_cyc, done_cyc, epoch, lane_id);
}

} // namespace inc::dc::dn::pl
