#include "kernel_operator.h"
#include "shmem.h"

#include "inc_dc_single_inc_stream_abi.h"
#include "inc_dc_gather_mte_aicore.h"

using namespace inc::dc::single_stream;

__aicore__ inline uint64_t StreamMinU64(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}

__aicore__ inline bool StreamWaitGeneration(__gm__ uint8_t *line, uint32_t generation,
                                             uint32_t spin_cap)
{
    __gm__ volatile uint32_t *value =
        reinterpret_cast<__gm__ volatile uint32_t *>(line);
    uint32_t spins = 0u;
    while (*value != generation) {
        dcci_cacheline(line);
        if (++spins >= spin_cap) {
            return false;
        }
    }
    return true;
}

__aicore__ inline __gm__ StreamLaneStat *StreamStat(
    __gm__ uint8_t *sym, __gm__ StreamDispatchDesc *desc, uint32_t lane)
{
    const uint64_t index =
        static_cast<uint64_t>(desc->pe) * kStreamMaxLanes + lane;
    return reinterpret_cast<__gm__ StreamLaneStat *>(
        sym + desc->stats_off + index * sizeof(StreamLaneStat));
}

__aicore__ inline void StreamPublishLocalDone(__gm__ uint8_t *line,
                                               uint32_t generation)
{
    *reinterpret_cast<__gm__ uint32_t *>(line) = generation;
    dcci_cacheline(line);
}

__aicore__ inline void StreamRemoteSignal(__gm__ uint8_t *line,
                                           uint32_t generation, int peer)
{
    *reinterpret_cast<__gm__ uint64_t *>(line + 8u) = generation;
    aclshmem_putmem_signal(
        line + 8u, line + 8u, sizeof(uint64_t),
        reinterpret_cast<__gm__ int32_t *>(line),
        static_cast<int32_t>(generation), ACLSHMEM_SIGNAL_SET, peer);
}

__aicore__ inline void StreamClearStat(__gm__ StreamLaneStat *stat)
{
    stat->start_cycle = 0u;
    stat->end_cycle = 0u;
    stat->input_bytes = 0u;
    stat->output_bytes = 0u;
    stat->gather_cycles = 0u;
    stat->transport_cycles = 0u;
    stat->tasks = 0u;
    stat->error = 0u;
}

__aicore__ inline bool StreamPackGenericTask(
    __gm__ uint8_t *sym, __gm__ StreamDispatchDesc *desc,
    __gm__ StreamDispatchTask *task, __gm__ StreamRouteEntry *routes,
    __ubuf__ uint8_t *ub)
{
    __gm__ uint8_t *packet =
        sym + desc->staging_off + (task->reserved1[1] - 1u);
    uint32_t i = 0u;
    while (i < task->route_count) {
        __gm__ StreamRouteEntry *first = &routes[task->route_begin + i];
        if (first->source_rank != desc->pe ||
            first->source_row >= desc->tokens_per_worker) {
            return false;
        }
        // A worker AIV does not share UB with the INC TX cohort.  Use one
        // descriptor for up to 64 KiB / eight regular rows, shrinking from
        // the runtime hidden size.  The previous 24-KiB cap split a W8/K1
        // packet into three serial gather/store pairs.
        const uint32_t max_rows = desc->hidden_bytes == 0u
            ? 0u
            : static_cast<uint32_t>(StreamMinU64(
                  8u, (64u * 1024u) / desc->hidden_bytes));
        uint32_t rows = static_cast<uint32_t>(StreamMinU64(
            max_rows, task->route_count - i));
        bool regular = rows >= 2u;
        uint32_t step = 0u;
        if (regular) {
            __gm__ StreamRouteEntry *second =
                &routes[task->route_begin + i + 1u];
            regular = second->source_rank == desc->pe &&
                      second->source_row > first->source_row;
            if (regular) step = second->source_row - first->source_row;
        }
        for (uint32_t j = 0u; regular && j < rows; ++j) {
            __gm__ StreamRouteEntry *candidate =
                &routes[task->route_begin + i + j];
            regular = candidate->source_rank == desc->pe &&
                      candidate->source_row == first->source_row + j * step;
        }
        const uint64_t gap64 = regular
            ? static_cast<uint64_t>(step - 1u) * desc->hidden_bytes
            : 0u;
        GM_ADDR src = reinterpret_cast<GM_ADDR>(
            sym + desc->input_off +
            static_cast<uint64_t>(desc->pe) * desc->input_stride +
            static_cast<uint64_t>(first->source_row) * desc->hidden_bytes);
        GM_ADDR dst = reinterpret_cast<GM_ADDR>(
            packet + static_cast<uint64_t>(i) * desc->hidden_bytes);
        if (regular && gap64 <= 0xffffffffull) {
            DcLlGatherCopyRegularRows(
                dst, src, desc->hidden_bytes, rows,
                static_cast<uint32_t>(gap64), ub, kDcLlGatherEvent);
            i += rows;
        } else {
            DcLlGatherCopyGm2GmChunked(
                dst, src, desc->hidden_bytes, ub, kDcLlGatherEvent);
            ++i;
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    return true;
}

__aicore__ inline bool StreamPublishUploadedTile(
    __gm__ uint8_t *sym, __gm__ StreamDispatchDesc *desc, uint32_t worker,
    uint32_t lane, uint32_t tile)
{
    StreamPublishLocalDone(
        sym + desc->upload_chunk_done_off +
            ((static_cast<uint64_t>(worker) * desc->tiles_per_worker + tile) *
                 kStreamMaxLanes +
             lane) *
                64u,
        desc->generation);
    if (lane != 0u) return true;
    for (uint32_t other = 0u; other < desc->upload_lane_count; ++other) {
        if (!StreamWaitGeneration(
                sym + desc->upload_chunk_done_off +
                    ((static_cast<uint64_t>(worker) *
                          desc->tiles_per_worker +
                      tile) *
                         kStreamMaxLanes +
                     other) *
                        64u,
                desc->generation, desc->spin_cap)) {
            return false;
        }
    }
    StreamRemoteSignal(
        sym + desc->tile_ready_off +
            (static_cast<uint64_t>(worker) * desc->tiles_per_worker + tile) *
                64u,
        desc->generation, static_cast<int>(desc->workers));
    return true;
}

__aicore__ inline void StreamWorker(__gm__ uint8_t *sym,
                                     __gm__ StreamDispatchDesc *desc,
                                     uint32_t lane)
{
    if (lane >= desc->upload_lane_count) {
        return;
    }
    __gm__ StreamLaneStat *stat = StreamStat(sym, desc, lane);
    StreamClearStat(stat);
    const uint32_t worker = desc->pe;
    const uint32_t inc_pe = desc->workers;
    __gm__ uint8_t *arrival =
        sym + desc->start_gate_off + static_cast<uint64_t>(worker) * 64u;
    __gm__ uint8_t *arm =
        sym + desc->start_gate_off +
        static_cast<uint64_t>(desc->workers + worker) * 64u;
    __gm__ uint8_t *ack =
        sym + desc->start_gate_off +
        static_cast<uint64_t>(desc->workers * 2u + worker) * 64u;
    __gm__ uint8_t *go =
        sym + desc->start_gate_off +
        static_cast<uint64_t>(desc->workers * 3u + worker) * 64u;
    if (lane == 0u) {
        StreamRemoteSignal(arrival, desc->generation,
                           static_cast<int>(inc_pe));
        if (!StreamWaitGeneration(arm, desc->generation, desc->spin_cap)) {
            stat->error = 8u;
        } else {
            StreamRemoteSignal(ack, desc->generation,
                               static_cast<int>(inc_pe));
        }
    }
    if (!StreamWaitGeneration(go, desc->generation, desc->spin_cap)) {
        stat->error = 9u;
    }
    if (stat->error != 0u) {
        stat->start_cycle = AscendC::GetSystemCycle();
        stat->end_cycle = stat->start_cycle + 1u;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(stat));
        return;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    stat->start_cycle = AscendC::GetSystemCycle();
    // All upload lanes collaborate on each tile.  Publishing wave N after its
    // chunks (rather than assigning a whole tile to each lane) lets the INC
    // consume wave N while workers are already uploading wave N+1.
    if ((desc->reserved32 & kStreamFlagHasDirect) != 0u &&
        (desc->reserved32 & kStreamFlagWorkerDirect) == 0u) {
        if (desc->upload_pingpong != 0u) {
            // Two stable private-MTE slots form one upload wave.  A wave is
            // drained remotely before either tile-ready record is released,
            // so the INC never observes partial payload while one quiet is
            // amortized over two tiles.  Input tiles are immutable and have
            // disjoint addresses, hence no staging-ring lifetime hazard.
            // Two credits are the qualified portable depth: they use the
            // same private-MTE/event footprint as the transport roofline and
            // amortize both payload drain and ready publication.
            constexpr uint32_t kUploadSlots = 2u;
            constexpr uint32_t kUploadUbTile =
                inc::dc::kIncDcPrivateMtePacketBytes;
            for (uint32_t wave = 0u; wave < desc->tiles_per_worker;
                 wave += kUploadSlots) {
                bool busy[kUploadSlots] = {false, false};
                uint64_t wave_bytes[kUploadSlots] = {0u, 0u};
                for (uint32_t slot = 0u; slot < kUploadSlots; ++slot) {
                    const uint32_t tile = wave + slot;
                    if (tile >= desc->tiles_per_worker) continue;
                    const uint32_t tile_row_begin = tile * desc->tile_rows;
                    uint32_t tile_row_count =
                        desc->tokens_per_worker - tile_row_begin;
                    if (tile_row_count > desc->tile_rows) {
                        tile_row_count = desc->tile_rows;
                    }
                    const uint32_t rows_per_lane =
                        (tile_row_count + desc->upload_lane_count - 1u) /
                        desc->upload_lane_count;
                    const uint32_t lane_row_begin = lane * rows_per_lane;
                    if (lane_row_begin >= tile_row_count) continue;
                    uint32_t lane_rows = tile_row_count - lane_row_begin;
                    if (lane_rows > rows_per_lane) lane_rows = rows_per_lane;
                    const uint64_t byte_begin =
                        static_cast<uint64_t>(tile_row_begin +
                                              lane_row_begin) *
                        desc->hidden_bytes;
                    const uint64_t bytes =
                        static_cast<uint64_t>(lane_rows) * desc->hidden_bytes;
                    __gm__ uint8_t *payload =
                        sym + desc->input_off +
                        static_cast<uint64_t>(worker) * desc->input_stride +
                        byte_begin;
                    aclshmemx_mte_put_nbi(
                        payload, payload,
                        reinterpret_cast<__ubuf__ uint8_t *>(
                            static_cast<uint64_t>(slot) * kUploadUbTile),
                        kUploadUbTile,
                        static_cast<uint32_t>(bytes),
                        static_cast<int>(inc_pe), slot);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
                    busy[slot] = true;
                    wave_bytes[slot] = bytes;
                }
                const uint64_t t0 = AscendC::GetSystemCycle();
                bool any_busy = false;
                for (uint32_t slot = 0u; slot < kUploadSlots; ++slot) {
                    if (busy[slot]) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
                        any_busy = true;
                    }
                }
                if (any_busy) aclshmemx_mte_quiet();
                stat->transport_cycles += AscendC::GetSystemCycle() - t0;
                uint32_t wave_tiles = 0u;
                for (uint32_t slot = 0u; slot < kUploadSlots; ++slot) {
                    const uint32_t tile = wave + slot;
                    if (tile >= desc->tiles_per_worker) continue;
                    ++wave_tiles;
                    stat->input_bytes += wave_bytes[slot];
                    if (wave_bytes[slot] != 0u) ++stat->tasks;
                    StreamPublishLocalDone(
                        sym + desc->upload_chunk_done_off +
                            ((static_cast<uint64_t>(worker) *
                                  desc->tiles_per_worker +
                              tile) *
                                 kStreamMaxLanes +
                             lane) *
                                64u,
                        desc->generation);
                }
                if (lane == 0u) {
                    for (uint32_t slot = 0u; slot < wave_tiles; ++slot) {
                        const uint32_t tile = wave + slot;
                        for (uint32_t other = 0u;
                             other < desc->upload_lane_count; ++other) {
                            if (!StreamWaitGeneration(
                                    sym + desc->upload_chunk_done_off +
                                        ((static_cast<uint64_t>(worker) *
                                              desc->tiles_per_worker +
                                          tile) *
                                             kStreamMaxLanes +
                                         other) *
                                            64u,
                                    desc->generation, desc->spin_cap)) {
                                stat->error = 11u;
                                break;
                            }
                        }
                        if (stat->error != 0u) break;
                        __gm__ uint8_t *ready =
                            sym + desc->tile_ready_off +
                            (static_cast<uint64_t>(worker) *
                                 desc->tiles_per_worker +
                             tile) *
                                64u;
                        *reinterpret_cast<__gm__ uint32_t *>(ready) =
                            desc->generation;
                        dcci_cacheline(ready);
                    }
                    if (stat->error == 0u && wave_tiles != 0u) {
                        __gm__ uint8_t *first_ready =
                            sym + desc->tile_ready_off +
                            (static_cast<uint64_t>(worker) *
                                 desc->tiles_per_worker +
                             wave) *
                                64u;
                        aclshmem_putmem_nbi(
                            first_ready, first_ready,
                            static_cast<size_t>(wave_tiles) * 64u,
                            static_cast<int>(inc_pe));
                        aclshmem_quiet();
                    }
                }
                if (stat->error != 0u) break;
            }
        } else {
        for (uint32_t tile = 0u; tile < desc->tiles_per_worker; ++tile) {
        const uint32_t tile_row_begin = tile * desc->tile_rows;
        uint32_t tile_row_count = desc->tokens_per_worker - tile_row_begin;
        if (tile_row_count > desc->tile_rows) {
            tile_row_count = desc->tile_rows;
        }
        const uint32_t rows_per_lane =
            (tile_row_count + desc->upload_lane_count - 1u) /
            desc->upload_lane_count;
        const uint32_t lane_row_begin = lane * rows_per_lane;
        if (lane_row_begin < tile_row_count) {
            uint32_t lane_rows = tile_row_count - lane_row_begin;
            if (lane_rows > rows_per_lane) lane_rows = rows_per_lane;
            const uint64_t byte_begin =
                static_cast<uint64_t>(tile_row_begin + lane_row_begin) *
                desc->hidden_bytes;
            const uint64_t bytes =
                static_cast<uint64_t>(lane_rows) * desc->hidden_bytes;
            __gm__ uint8_t *payload =
                sym + desc->input_off +
                static_cast<uint64_t>(worker) * desc->input_stride +
                byte_begin;
            const uint64_t t0 = AscendC::GetSystemCycle();
            aclshmem_putmem_nbi(payload, payload, bytes,
                                static_cast<int>(inc_pe));
            aclshmem_quiet();
            stat->transport_cycles += AscendC::GetSystemCycle() - t0;
            stat->input_bytes += bytes;
            ++stat->tasks;
        }
        if (!StreamPublishUploadedTile(sym, desc, worker, lane, tile)) {
            stat->error = 11u;
            break;
        }
        }
        }
    }

    // Scheme B for arbitrary plans: each source rank locally packs only its
    // own generic tasks, then pushes one contiguous packet plus a generation
    // signal either to INC staging or, for a low-expansion plan, directly to
    // the final worker while the INC remains completion coordinator.  The
    // task ordinal, not W, shards work across the runtime-sized cohort.
    if ((desc->reserved32 & kStreamFlagWorkerPack) != 0u &&
        stat->error == 0u) {
        __gm__ StreamDispatchTask *tasks =
            reinterpret_cast<__gm__ StreamDispatchTask *>(
                sym + desc->task_off);
        __gm__ StreamRouteEntry *routes =
            reinterpret_cast<__gm__ StreamRouteEntry *>(
                sym + desc->route_off);
        __ubuf__ uint8_t *ub = reinterpret_cast<__ubuf__ uint8_t *>(0);
        __gm__ uint32_t *worker_task_offsets =
            reinterpret_cast<__gm__ uint32_t *>(
                sym + desc->worker_task_offsets_off);
        __gm__ uint32_t *worker_task_indices =
            reinterpret_cast<__gm__ uint32_t *>(
                sym + desc->worker_task_indices_off);
        const uint32_t source_task_begin = worker_task_offsets[worker];
        const uint32_t source_task_end = worker_task_offsets[worker + 1u];
        for (uint32_t task_pos = source_task_begin + lane;
             task_pos < source_task_end;
             task_pos += desc->upload_lane_count) {
            const uint32_t task_index = worker_task_indices[task_pos];
            __gm__ StreamDispatchTask *task = &tasks[task_index];
            const uint64_t t0 = AscendC::GetSystemCycle();
            if (!StreamPackGenericTask(sym, desc, task, routes, ub)) {
                stat->error = 17u;
                break;
            }
            __gm__ uint8_t *packet =
                sym + desc->staging_off + (task->reserved1[1] - 1u);
            __gm__ uint8_t *remote_packet = packet;
            int packet_peer = static_cast<int>(inc_pe);
            if ((desc->reserved32 & kStreamFlagWorkerDirect) != 0u) {
                remote_packet = sym + desc->output_off +
                                task->output_byte_offset;
                packet_peer = static_cast<int>(task->destination_rank);
            }
            __gm__ uint8_t *completion =
                sym + desc->completion_off +
                (static_cast<uint64_t>(desc->workers) + task->reserved0) *
                    64u;
            *reinterpret_cast<__gm__ uint64_t *>(completion + 8u) =
                desc->generation;
            if ((desc->reserved32 & kStreamFlagWorkerDirect) != 0u) {
                aclshmem_putmem_nbi(remote_packet, packet,
                                    task->packet_bytes, packet_peer);
                aclshmem_quiet();
                StreamRemoteSignal(completion, desc->generation,
                                   static_cast<int>(inc_pe));
            } else {
                aclshmem_putmem_signal(
                    remote_packet, packet, task->packet_bytes,
                    reinterpret_cast<__gm__ int32_t *>(completion),
                    static_cast<int32_t>(desc->generation),
                    ACLSHMEM_SIGNAL_SET, packet_peer);
            }
            stat->transport_cycles += AscendC::GetSystemCycle() - t0;
            stat->input_bytes += task->packet_bytes;
            ++stat->tasks;
        }
    }
    StreamPublishLocalDone(
        sym + desc->lane_done_off +
            (static_cast<uint64_t>(worker) * kStreamMaxLanes + lane) * 64u,
        desc->generation);

    if (lane == 0u) {
        for (uint32_t other = 1u; other < desc->upload_lane_count; ++other) {
            if (!StreamWaitGeneration(
                    sym + desc->lane_done_off +
                        (static_cast<uint64_t>(worker) * kStreamMaxLanes + other) * 64u,
                    desc->generation, desc->spin_cap)) {
                stat->error = 1u;
                break;
            }
        }
        if (stat->error == 0u &&
            !StreamWaitGeneration(sym + desc->completion_off +
                                      static_cast<uint64_t>(worker) * 64u,
                                  desc->generation, desc->spin_cap)) {
            stat->error = 2u;
        }
        stat->end_cycle = AscendC::GetSystemCycle();
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(stat));
    } else {
        stat->end_cycle = AscendC::GetSystemCycle();
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(stat));
    }
}

__aicore__ inline void StreamInc(__gm__ uint8_t *sym,
                                  __gm__ StreamDispatchDesc *desc,
                                  uint32_t lane)
{
    if (lane >= desc->lane_count) {
        return;
    }
    __gm__ StreamLaneStat *stat = StreamStat(sym, desc, lane);
    StreamClearStat(stat);
    const uint32_t start_coordinator = desc->gather_lane_count;
    const uint32_t finish_coordinator =
        desc->gather_lane_count == 0u && desc->lane_count > desc->workers
            ? desc->workers
            : start_coordinator;
    __gm__ uint8_t *local_go =
        sym + desc->start_gate_off +
        static_cast<uint64_t>(desc->workers * 4u) * 64u;
    if (lane == start_coordinator) {
        for (uint32_t worker = 0u; worker < desc->workers; ++worker) {
            if (!StreamWaitGeneration(
                    sym + desc->start_gate_off +
                        static_cast<uint64_t>(worker) * 64u,
                    desc->generation, desc->spin_cap)) {
                stat->error = 8u;
                break;
            }
        }
        if (stat->error == 0u) {
            for (uint32_t worker = 0u; worker < desc->workers; ++worker) {
                StreamRemoteSignal(
                    sym + desc->start_gate_off +
                        static_cast<uint64_t>(desc->workers + worker) * 64u,
                    desc->generation, static_cast<int>(worker));
            }
            for (uint32_t worker = 0u; worker < desc->workers; ++worker) {
                if (!StreamWaitGeneration(
                        sym + desc->start_gate_off +
                            static_cast<uint64_t>(desc->workers * 2u +
                                                  worker) * 64u,
                        desc->generation, desc->spin_cap)) {
                    stat->error = 9u;
                    break;
                }
            }
        }
        if (stat->error == 0u) {
            StreamPublishLocalDone(local_go, desc->generation);
        }
    }
    if (!StreamWaitGeneration(local_go, desc->generation, desc->spin_cap)) {
        stat->error = 10u;
    }
    if (stat->error != 0u) {
        stat->start_cycle = AscendC::GetSystemCycle();
        stat->end_cycle = stat->start_cycle + 1u;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(stat));
        return;
    }
    // One INC AIV releases one worker.  The final start wave is therefore
    // parallel instead of charging a coordinator's O(W) notification skew to
    // the earliest worker's operator time.
    for (uint32_t worker = lane; worker < desc->workers;
         worker += desc->lane_count) {
        StreamRemoteSignal(
            sym + desc->start_gate_off +
                static_cast<uint64_t>(desc->workers * 3u + worker) * 64u,
            desc->generation, static_cast<int>(worker));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    stat->start_cycle = AscendC::GetSystemCycle();
    __gm__ StreamDispatchTask *tasks =
        reinterpret_cast<__gm__ StreamDispatchTask *>(sym + desc->task_off);
    __gm__ StreamRouteEntry *routes =
        reinterpret_cast<__gm__ StreamRouteEntry *>(sym + desc->route_off);
    __ubuf__ uint8_t *ub = reinterpret_cast<__ubuf__ uint8_t *>(0);

    if (lane < desc->gather_lane_count) {
        // Stage G: many AIVs fill disjoint packet slices.  Completion is one
        // local cacheline per slice; TX lanes consume it without remote ACKs.
        if (desc->direct_task_count != 0u) {
            const uint32_t tile_total =
                desc->workers * desc->tiles_per_worker;
            for (uint32_t flat = lane; flat < tile_total;
                 flat += desc->gather_lane_count) {
                if (!StreamWaitGeneration(
                        sym + desc->tile_ready_off +
                            static_cast<uint64_t>(flat) * 64u,
                        desc->generation, desc->spin_cap)) {
                    stat->error = 13u;
                    break;
                }
                const uint32_t source = flat / desc->tiles_per_worker;
                const uint32_t tile = flat % desc->tiles_per_worker;
                const uint64_t byte_begin =
                    static_cast<uint64_t>(tile) * desc->tile_bytes;
                const uint64_t bytes = StreamMinU64(
                    desc->tile_bytes, desc->input_stride - byte_begin);
                // Visibility is AIV-local on this platform.  This cohort only
                // publishes ordering; the peer-owned TX AIV invalidates the
                // exact source range it will issue through RMA below.
                StreamPublishLocalDone(
                    sym + desc->direct_ready_off +
                        static_cast<uint64_t>(flat) * 64u,
                    desc->generation);
            }
        }
        if (stat->error != 0u) {
            StreamPublishLocalDone(
                sym + desc->lane_done_off +
                    (static_cast<uint64_t>(desc->workers) *
                         kStreamMaxLanes +
                     lane) * 64u,
                0xffffffffu);
            stat->end_cycle = AscendC::GetSystemCycle();
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(stat));
            return;
        }
        for (uint32_t task_index = 0u;
             (desc->reserved32 & kStreamFlagWorkerPack) == 0u &&
             task_index < desc->task_count;
             ++task_index) {
            __gm__ StreamDispatchTask *task = &tasks[task_index];
            const uint32_t chunk_count =
                static_cast<uint32_t>(task->reserved1[0]);
            for (uint32_t chunk = 0u; chunk < chunk_count; ++chunk) {
                const uint32_t global_chunk = task->reserved0 + chunk;
                if (global_chunk % desc->gather_lane_count != lane) {
                    continue;
                }
                const uint32_t route_local_begin =
                    chunk * desc->gather_chunk_routes;
                uint32_t route_n = task->route_count - route_local_begin;
                if (route_n > desc->gather_chunk_routes) {
                    route_n = desc->gather_chunk_routes;
                }
                const uint32_t ready_begin =
                    task->source_rank == desc->workers ? 0u : task->source_rank;
                const uint32_t ready_end =
                    task->source_rank == desc->workers
                        ? desc->workers
                        : task->source_rank + 1u;
                for (uint32_t source = ready_begin; source < ready_end;
                     ++source) {
                    __gm__ uint8_t *tile_ready =
                        sym + desc->tile_ready_off +
                        (static_cast<uint64_t>(source) *
                             desc->tiles_per_worker +
                         task->source_tile) * 64u;
                    if (!StreamWaitGeneration(tile_ready, desc->generation,
                                              desc->spin_cap)) {
                        stat->error = 4u;
                        break;
                    }
                }
                if (stat->error != 0u) break;
                const uint64_t gather_t0 = AscendC::GetSystemCycle();
                __gm__ uint8_t *packet =
                    sym + desc->staging_off +
                    (task->reserved1[1] - 1u);
                uint32_t i = 0u;
                while (i < route_n) {
                    const uint32_t max_regular_rows =
                        desc->hidden_bytes == 0u
                            ? 0u
                            : StreamMinU64(
                                  3u, inc::dc::kIncDcAivUbBudgetBytes /
                                          desc->hidden_bytes);
                    uint32_t regular_rows =
                        StreamMinU64(max_regular_rows, route_n - i);
                    bool regular = regular_rows >= 2u;
                    uint32_t regular_step = 0u;
                    __gm__ StreamRouteEntry *first_route =
                        &routes[task->route_begin + route_local_begin + i];
                    if (regular) {
                        __gm__ StreamRouteEntry *second_route =
                            &routes[task->route_begin + route_local_begin +
                                    i + 1u];
                        regular = first_route->source_rank < desc->workers &&
                                  first_route->source_row <
                                      desc->tokens_per_worker &&
                                  second_route->source_rank ==
                                      first_route->source_rank &&
                                  second_route->source_row >
                                      first_route->source_row;
                        if (regular) {
                            regular_step = second_route->source_row -
                                           first_route->source_row;
                        }
                    }
                    for (uint32_t j = 0u; regular && j < regular_rows; ++j) {
                        __gm__ StreamRouteEntry *candidate =
                            &routes[task->route_begin + route_local_begin +
                                    i + j];
                        regular =
                            candidate->source_rank ==
                                first_route->source_rank &&
                            candidate->source_row ==
                                first_route->source_row + j * regular_step &&
                            candidate->source_row / desc->tile_rows ==
                                task->source_tile;
                    }
                    const uint64_t gap64 = regular
                        ? static_cast<uint64_t>(regular_step - 1u) *
                              desc->hidden_bytes
                        : 0u;
                    if (regular && gap64 <= 0xffffffffull) {
                        GM_ADDR src = reinterpret_cast<GM_ADDR>(
                            sym + desc->input_off +
                            static_cast<uint64_t>(first_route->source_rank) *
                                desc->input_stride +
                            static_cast<uint64_t>(first_route->source_row) *
                                desc->hidden_bytes);
                        GM_ADDR dst = reinterpret_cast<GM_ADDR>(
                            packet +
                            static_cast<uint64_t>(route_local_begin + i) *
                                desc->hidden_bytes);
                        DcLlGatherCopyRegularRows(
                            dst, src, desc->hidden_bytes, regular_rows,
                            static_cast<uint32_t>(gap64), ub,
                            kDcLlGatherEvent);
                        i += regular_rows;
                        continue;
                    }
                    const uint32_t route_local = route_local_begin + i;
                    __gm__ StreamRouteEntry *route =
                        &routes[task->route_begin + route_local];
                    if (route->source_rank >= desc->workers ||
                        route->source_row >= desc->tokens_per_worker ||
                        route->source_row / desc->tile_rows !=
                            task->source_tile) {
                        stat->error = 5u;
                        break;
                    }
                    GM_ADDR src = reinterpret_cast<GM_ADDR>(
                        sym + desc->input_off +
                        static_cast<uint64_t>(route->source_rank) *
                            desc->input_stride +
                        static_cast<uint64_t>(route->source_row) *
                            desc->hidden_bytes);
                    GM_ADDR dst = reinterpret_cast<GM_ADDR>(
                        packet + static_cast<uint64_t>(route_local) *
                                     desc->hidden_bytes);
                    DcLlGatherCopyGm2GmChunked(dst, src, desc->hidden_bytes,
                                               ub, kDcLlGatherEvent);
                    ++i;
                }
                if (stat->error != 0u) break;
                AscendC::PipeBarrier<PIPE_ALL>();
                // The packet is produced by this gather AIV and consumed by
                // a different TX AIV.  Publish payload visibility before the
                // completion generation; otherwise large generic packets can
                // expose stale cachelines even though every control record is
                // correct.  Whole-cache DCCI is one hardware operation here,
                // far cheaper than walking thousands of 64-byte lines.
                dcci_entire_cache();
                stat->gather_cycles +=
                    AscendC::GetSystemCycle() - gather_t0;
                StreamPublishLocalDone(
                    sym + desc->completion_off +
                        (static_cast<uint64_t>(desc->workers) +
                         global_chunk) * 64u,
                    desc->generation);
                ++stat->tasks;
            }
            if (stat->error != 0u) break;
        }
    } else if (lane < desc->gather_lane_count + desc->tx_lane_count) {
        // Stage T: a smaller TX cohort waits for complete local packets and
        // sends large NBI operations.  Task-private staging permits a bounded
        // multi-destination quiet window without source-buffer reuse.
        const uint32_t tx_lane = lane - desc->gather_lane_count;
        const bool peer_sharded = desc->tx_lane_count >= desc->workers;
        const uint32_t lane_destination =
            peer_sharded ? tx_lane % desc->workers : 0u;
        const uint32_t destination_lane =
            peer_sharded ? tx_lane / desc->workers : 0u;
        const uint32_t destination_lane_count =
            peer_sharded
                ? (desc->tx_lane_count + desc->workers - 1u -
                   lane_destination) /
                      desc->workers
                : 0u;
        uint32_t destination_task_seq = 0u;
        uint32_t tx_pending = 0u;
        // Mirrors the runtime MTE configuration the public putmem path uses,
        // so a packet larger than one buffer is split the same way.
        constexpr uint32_t kStreamTxSlots = 2u;
        constexpr uint32_t kStreamTxUbTile =
            inc::dc::kIncDcPrivateMtePacketBytes;
        __ubuf__ uint8_t *tx_ub[kStreamTxSlots] = {
            reinterpret_cast<__ubuf__ uint8_t *>(0),
            reinterpret_cast<__ubuf__ uint8_t *>(
                static_cast<uint64_t>(kStreamTxUbTile))};
        bool tx_ub_busy[kStreamTxSlots] = {false, false};
        uint32_t tx_ping = 0u;
        uint32_t visible_direct_tile = 0xffffffffu;
        // Splitting a destination slice at every source discontinuity makes
        // task counts scale with rows rather than tiles, so re-polling worker
        // readiness per task would cost more than the transport it guards.
        // A generation, once observed for a (tile, source) pair, stays
        // observed for this epoch.
        uint32_t observed_ready_tile = 0xffffffffu;
        uint32_t observed_ready_source = 0xffffffffu;
        // HCCS MTE direct packets use a non-L2 source read below.  They do not
        // require the cache visibility preflight used by the generic path.
        const bool direct_preflight = false;
        if (direct_preflight) {
            // Cache visibility is shared across AIVs: no lane may invalidate
            // while another lane has an NBI in flight.  Establish one global
            // visibility epoch after all worker uploads, then release TX as a
            // cohort.  The per-lane lines live in INC-local symmetric memory.
            for (uint32_t source = 0u; source < desc->workers; ++source) {
                for (uint32_t tile = 0u; tile < desc->tiles_per_worker;
                     ++tile) {
                    if (!StreamWaitGeneration(
                            sym + desc->tile_ready_off +
                                (static_cast<uint64_t>(source) *
                                     desc->tiles_per_worker +
                                 tile) * 64u,
                            desc->generation, desc->spin_cap)) {
                        stat->error = 14u;
                        break;
                    }
                }
                if (stat->error != 0u) break;
            }
            if (stat->error == 0u) {
                dcci_entire_cache();
                StreamPublishLocalDone(
                    sym + desc->upload_chunk_done_off +
                        static_cast<uint64_t>(lane) * 64u,
                    desc->generation);
                for (uint32_t other = 0u; other < desc->lane_count; ++other) {
                    if (!StreamWaitGeneration(
                            sym + desc->upload_chunk_done_off +
                                static_cast<uint64_t>(other) * 64u,
                            desc->generation, desc->spin_cap)) {
                        stat->error = 15u;
                        break;
                    }
                }
            }
        }
        uint32_t task_begin = 0u;
        uint32_t task_end = desc->task_count;
        if (desc->tx_lane_tasks_contiguous != 0u) {
            __gm__ const uint32_t *lane_offsets =
                reinterpret_cast<__gm__ const uint32_t *>(
                    sym + desc->tx_lane_task_offsets_off);
            task_begin = lane_offsets[tx_lane];
            task_end = lane_offsets[tx_lane + 1u];
        }
        __gm__ const uint32_t *lane_task_indices =
            desc->tx_lane_tasks_contiguous == 2u
                ? reinterpret_cast<__gm__ const uint32_t *>(
                      sym + desc->tx_lane_task_indices_off)
                : nullptr;
        for (uint32_t task_pos = task_begin;
             (desc->reserved32 & kStreamFlagWorkerDirect) == 0u &&
             task_pos < task_end;
             ++task_pos) {
            const uint32_t task_index = lane_task_indices != nullptr
                ? lane_task_indices[task_pos]
                : task_pos;
            if (task_index >= desc->task_count) {
                stat->error = 18u;
                break;
            }
            __gm__ StreamDispatchTask *task = &tasks[task_index];
            bool owns_task = desc->tx_lane_tasks_contiguous != 0u;
            if (!owns_task) {
                if (peer_sharded) {
                    if (task->destination_rank == lane_destination) {
                        owns_task =
                            destination_task_seq % destination_lane_count ==
                            destination_lane;
                        ++destination_task_seq;
                    }
                } else {
                    owns_task = task_index % desc->tx_lane_count == tx_lane;
                }
            }
            if (!owns_task) continue;
            const uint32_t chunk_count =
                static_cast<uint32_t>(task->reserved1[0]);
            const bool direct = chunk_count == 0u;
            if (direct && !direct_preflight) {
                // A whole-cache invalidate must never run while an older NBI
                // can still be reading its source.  Close the previous tile
                // epoch first; all packets in one tile then share one
                // visibility operation and may be submitted as a window.
                if (task->source_tile != visible_direct_tile &&
                    (desc->reserved32 & kStreamFlagDirectDcci) != 0u) {
                    if (tx_pending != 0u) {
                        aclshmem_quiet();
                        tx_pending = 0u;
                    }
                    for (uint32_t slot = 0u; slot < kStreamTxSlots; ++slot) {
                        if (tx_ub_busy[slot]) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                                slot);
                            tx_ub_busy[slot] = false;
                        }
                    }
                    aclshmemx_mte_quiet();
                }
                const uint32_t ready_begin =
                    task->source_rank == desc->workers ? 0u : task->source_rank;
                const uint32_t ready_end =
                    task->source_rank == desc->workers
                        ? desc->workers
                        : task->source_rank + 1u;
                const bool already_observed =
                    task->source_tile == observed_ready_tile &&
                    task->source_rank == observed_ready_source;
                if (!already_observed) {
                    for (uint32_t source = ready_begin; source < ready_end;
                         ++source) {
                        if (!StreamWaitGeneration(
                                sym + (desc->gather_lane_count == 0u
                                           ? desc->tile_ready_off
                                           : desc->direct_ready_off) +
                                    (static_cast<uint64_t>(source) *
                                         desc->tiles_per_worker +
                                     task->source_tile) * 64u,
                                desc->generation, desc->spin_cap)) {
                            stat->error = 12u;
                            break;
                        }
                    }
                }
                if (stat->error == 0u) {
                    visible_direct_tile = task->source_tile;
                    observed_ready_tile = task->source_tile;
                    observed_ready_source = task->source_rank;
                }
            }
            const uint32_t ready_chunks =
                (desc->reserved32 & kStreamFlagWorkerPack) != 0u &&
                        chunk_count != 0u
                    ? 1u
                    : chunk_count;
            for (uint32_t chunk = 0u; chunk < ready_chunks; ++chunk) {
                const uint32_t global_chunk = task->reserved0 + chunk;
                if (!StreamWaitGeneration(
                        sym + desc->completion_off +
                            (static_cast<uint64_t>(desc->workers) +
                             global_chunk) * 64u,
                        desc->generation, desc->spin_cap)) {
                    stat->error = 7u;
                    break;
                }
            }
            if (stat->error != 0u) break;
            __gm__ uint8_t *packet =
                direct
                    ? sym + desc->input_off + (task->reserved1[1] - 1u)
                    : sym + desc->staging_off +
                          (task->reserved1[1] - 1u);
            __gm__ uint8_t *remote_output =
                sym + desc->output_off + task->output_byte_offset;
            const uint64_t tx_t0 = AscendC::GetSystemCycle();
            // The generic public put path owns a different/shared MTE sync
            // slot from the direct ping-pong path.  A lane-major task range
            // may legitimately alternate between both task kinds, so close
            // all private direct slots before entering the generic backend.
            // Without this hand-off, mixed direct/gather shapes can publish
            // completion for a generic row while an older direct MTE still
            // owns the lane engine, producing silent stale bytes.
            if (direct &&
                (desc->reserved32 & kStreamFlagDirectDcci) != 0u) {
                dcci_entire_cache();
            }
            if (desc->tx_pingpong != 0u) {
                const uint32_t slot = tx_ping;
                if (tx_ub_busy[slot]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
                    tx_ub_busy[slot] = false;
                }
                aclshmemx_mte_put_nbi(
                    remote_output, packet, tx_ub[slot], kStreamTxUbTile,
                    static_cast<uint32_t>(task->packet_bytes),
                    static_cast<int>(task->destination_rank), slot);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
                tx_ub_busy[slot] = true;
                tx_ping = (tx_ping + 1u) % kStreamTxSlots;
            } else {
                aclshmem_putmem_nbi(remote_output, packet,
                                    task->packet_bytes,
                                    static_cast<int>(task->destination_rank));
                ++tx_pending;
                if (tx_pending >= desc->tx_window) {
                    aclshmem_quiet();
                    tx_pending = 0u;
                }
            }
            stat->transport_cycles += AscendC::GetSystemCycle() - tx_t0;
            stat->output_bytes += task->packet_bytes;
            ++stat->tasks;
        }
        if (tx_pending != 0u) {
            const uint64_t quiet_t0 = AscendC::GetSystemCycle();
            aclshmem_quiet();
            stat->transport_cycles += AscendC::GetSystemCycle() - quiet_t0;
        }
        bool any_tx_ub_busy = false;
        for (uint32_t slot = 0u; slot < kStreamTxSlots; ++slot) {
            any_tx_ub_busy = any_tx_ub_busy || tx_ub_busy[slot];
        }
        if (any_tx_ub_busy) {
            const uint64_t quiet_t0 = AscendC::GetSystemCycle();
            for (uint32_t slot = 0u; slot < kStreamTxSlots; ++slot) {
                if (tx_ub_busy[slot]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
                    tx_ub_busy[slot] = false;
                }
            }
            aclshmemx_mte_quiet();
            stat->transport_cycles += AscendC::GetSystemCycle() - quiet_t0;
        }
    }

    StreamPublishLocalDone(
        sym + desc->lane_done_off +
            (static_cast<uint64_t>(desc->workers) * kStreamMaxLanes + lane) * 64u,
        stat->error == 0u ? desc->generation : 0xffffffffu);
    if (lane == finish_coordinator && stat->error == 0u) {
        for (uint32_t other = desc->gather_lane_count;
             other < desc->lane_count; ++other) {
            if (other == finish_coordinator) continue;
            if (!StreamWaitGeneration(
                    sym + desc->lane_done_off +
                        (static_cast<uint64_t>(desc->workers) *
                             kStreamMaxLanes +
                         other) * 64u,
                    desc->generation, desc->spin_cap)) {
                stat->error = 6u;
                break;
            }
        }
        if (stat->error == 0u &&
            (desc->reserved32 & kStreamFlagWorkerDirect) != 0u) {
            for (uint32_t task_index = 0u;
                 task_index < desc->task_count; ++task_index) {
                __gm__ StreamDispatchTask *task = &tasks[task_index];
                if (!StreamWaitGeneration(
                        sym + desc->completion_off +
                            (static_cast<uint64_t>(desc->workers) +
                             task->reserved0) * 64u,
                        desc->generation, desc->spin_cap)) {
                    stat->error = 18u;
                    break;
                }
            }
        }
        if (stat->error == 0u && desc->gather_lane_count != 0u) {
            for (uint32_t worker = 0u; worker < desc->workers; ++worker) {
                __gm__ uint8_t *line =
                    sym + desc->completion_off +
                    static_cast<uint64_t>(worker) * 64u;
                *reinterpret_cast<__gm__ uint64_t *>(line + 8u) =
                    desc->generation;
                aclshmem_putmem_signal(
                    line + 8u, line + 8u, sizeof(uint64_t),
                    reinterpret_cast<__gm__ int32_t *>(line),
                    static_cast<int32_t>(desc->generation),
                    ACLSHMEM_SIGNAL_SET, static_cast<int>(worker));
            }
        }
    }
    if (desc->gather_lane_count == 0u &&
        desc->workers >= desc->tx_lane_count && lane == 0u &&
        stat->error == 0u) {
        for (uint32_t tx_lane = 0u; tx_lane < desc->tx_lane_count;
             ++tx_lane) {
            if (!StreamWaitGeneration(
                    sym + desc->lane_done_off +
                        (static_cast<uint64_t>(desc->workers) *
                             kStreamMaxLanes +
                         tx_lane) * 64u,
                    desc->generation, desc->spin_cap)) {
                stat->error = 16u;
                break;
            }
        }
        if (stat->error == 0u) {
            for (uint32_t worker = 0u; worker < desc->workers; ++worker) {
                __gm__ uint8_t *line =
                    sym + desc->completion_off +
                    static_cast<uint64_t>(worker) * 64u;
                *reinterpret_cast<__gm__ uint64_t *>(line + 8u) =
                    desc->generation;
                aclshmem_putmem_signal(
                    line + 8u, line + 8u, sizeof(uint64_t),
                    reinterpret_cast<__gm__ int32_t *>(line),
                    static_cast<int32_t>(desc->generation),
                    ACLSHMEM_SIGNAL_SET, static_cast<int>(worker));
            }
            aclshmem_quiet();
        }
    } else if (desc->gather_lane_count == 0u && lane < desc->workers &&
               stat->error == 0u) {
        for (uint32_t tx_lane = lane; tx_lane < desc->tx_lane_count;
             tx_lane += desc->workers) {
            if (!StreamWaitGeneration(
                    sym + desc->lane_done_off +
                        (static_cast<uint64_t>(desc->workers) *
                             kStreamMaxLanes +
                         tx_lane) * 64u,
                    desc->generation, desc->spin_cap)) {
                stat->error = 16u;
                break;
            }
        }
        if (stat->error == 0u) {
            __gm__ uint8_t *line =
                sym + desc->completion_off + static_cast<uint64_t>(lane) * 64u;
            *reinterpret_cast<__gm__ uint64_t *>(line + 8u) = desc->generation;
            aclshmem_putmem_signal(
                line + 8u, line + 8u, sizeof(uint64_t),
                reinterpret_cast<__gm__ int32_t *>(line),
                static_cast<int32_t>(desc->generation), ACLSHMEM_SIGNAL_SET,
                static_cast<int>(lane));
            aclshmem_quiet();
        }
    }
    stat->end_cycle = AscendC::GetSystemCycle();
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(stat));
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
inc_dc_single_inc_stream_cycle_probe_kernel(GM_ADDR base, uint64_t cycle_off)
{
    if ASCEND_IS_AIV {
        *reinterpret_cast<__gm__ uint64_t *>(
            reinterpret_cast<__gm__ uint8_t *>(base) + cycle_off) =
            AscendC::GetSystemCycle();
    }
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
inc_dc_single_inc_stream_dispatch_kernel(GM_ADDR base)
{
    if ASCEND_IS_AIV {
        __gm__ uint8_t *sym = reinterpret_cast<__gm__ uint8_t *>(base);
        __gm__ StreamDispatchDesc *desc =
            reinterpret_cast<__gm__ StreamDispatchDesc *>(sym + kStreamDescOff);
        if (desc->magic != kStreamMagic || desc->version != kStreamVersion ||
            desc->workers == 0u ||
            desc->lane_count == 0u || desc->lane_count > kStreamMaxLanes ||
            (desc->gather_lane_count == 0u &&
             (desc->reserved32 & kStreamFlagWorkerPack) == 0u &&
             (desc->gather_chunk_count != 0u ||
              desc->direct_task_count != desc->task_count)) ||
            desc->tx_lane_count == 0u ||
            desc->gather_lane_count + desc->tx_lane_count >
                desc->lane_count ||
            desc->upload_lane_count == 0u ||
            desc->upload_lane_count > kStreamMaxLanes ||
            desc->hidden_bytes == 0u || desc->tile_rows == 0u ||
            desc->tiles_per_worker == 0u || desc->tx_window == 0u ||
            desc->tx_window > 64u || desc->gather_chunk_routes == 0u) {
            return;
        }
        while (desc->start_target_cycle != 0u &&
               AscendC::GetSystemCycle() < desc->start_target_cycle) {
        }
        const uint32_t lane = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (desc->pe < desc->workers) {
            StreamWorker(sym, desc, lane);
        } else if (desc->pe == desc->workers) {
            StreamInc(sym, desc, lane);
        }
    }
}

extern "C" void launch_inc_dc_single_inc_stream_cycle_probe_kernel(
    uint8_t *sym, uint64_t cycle_off, void *stream)
{
    inc_dc_single_inc_stream_cycle_probe_kernel<<<1, nullptr, stream>>>(
        sym, cycle_off);
}

extern "C" void launch_inc_dc_single_inc_stream_dispatch_kernel(
    uint8_t *sym, int block_dim, void *stream)
{
    inc_dc_single_inc_stream_dispatch_kernel<<<block_dim, nullptr, stream>>>(sym);
}
