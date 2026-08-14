#include "inc_fusion_compute_device.h"
#include "inc_fusion_protocol_device.h"
#include "inc_fusion_abi.h"

#if defined(INC_FUSION_LEGACY_OP_SYSTEM_RUN_CFG)
// CANN 8.x CATLASS leaves this as an extern for standalone AscendC kernels;
// newer bisheng versions synthesize it and therefore use the other branch.
__gm__ struct OpSystemRunCfg g_opSystemRunCfg{Catlass::L2_OFFSET};
#endif

using namespace inc::fusion;
using namespace inc::fusion::compute;
using namespace inc::fusion::device;

namespace {

constexpr uint32_t kReadyDispatch = 0u;
constexpr uint32_t kReadyGmm1 = 1u;
constexpr uint32_t kReadyActivation = 2u;
constexpr uint32_t kReadyGmm2 = 3u;
constexpr uint32_t kReadyCombineResult = 4u;
// The grouped Cube path publishes one readiness edge per phase, so the
// per-expert dimension of the GMM1 ready phase is free for the two Cube/Vector
// cache-handoff tickets.  grouped_cube implies local_expert_count > live_aic,
// hence expert_count is far above these two indices.
constexpr uint32_t kGroupedGateAcquiredFlat = 1u;
constexpr uint32_t kGroupedActivationAcquiredFlat = 2u;
// A single AIV issuing putmem tops out near 7 GB/s, a quarter of one HCCS
// link, so the dispatch upload is striped over the whole dispatch cohort and
// the doorbell waits on this join line.  Flat index 0 of the dispatch phase
// carries the receive tickets and index 1 the raw-Combine metadata edge.
constexpr uint32_t kDispatchTxJoinFlat = 2u;
constexpr uint32_t kCopyTileBytes = 16u * 1024u;
constexpr uint32_t kReduceTileElements = 1024u;
// The dense Combine reduction is latency bound, not bandwidth bound: each
// contribution costs one exposed GM round trip regardless of tile size.  A
// tile that spans a whole hidden row halves the number of round trips per row
// at typical hidden sizes and collapses the column loop entirely.
constexpr uint32_t kDenseReduceTileElements = 2048u;
// Bulk Combine keeps one BF16 input, one BF16 output and two FP32 vectors in
// UB.  2048 elements use 24 KiB in total and stay below the portable per-AIV
// UB budget on both supported 910B toolchains.  The previous 8192-element
// value requested 96 KiB and could alias buffers at runtime even though the
// kernel compiler accepted the allocation.
constexpr uint32_t kBulkReduceTileElements = 2048u;
constexpr uint32_t kBulkDenseBf16Protocol = 0x44424632u; // "DBF2"
// Hierarchical (dense) Combine: workers reduce their TopK contributions
// locally in FP32 and upload one BF16 partial per worker; INC only sums W
// partials.  Qualified against per-layer golden within 1-2 BF16 ULP and is
// the formal configuration.  Raw per-assignment Combine stays available for
// numerical cross-checks by flipping this switch.
constexpr bool kEnableBulkDenseCombine = true;
constexpr bool kSerialActivationDiagnostic = false;
constexpr bool kSerialDenseIncDiagnostic = false;

__aicore__ inline uint32_t ActiveTokenCount(
    __gm__ const FusionKernelArgs *args, uint32_t source)
{
    if (args->active_token_counts == 0u) return args->token_count;
    __gm__ const uint32_t *counts =
        reinterpret_cast<__gm__ const uint32_t *>(
            args->active_token_counts);
    return counts[source];
}

__aicore__ inline uint32_t SourceWaveTokenCount(
    __gm__ const FusionKernelArgs *args,
    __gm__ const FusionWaveDesc *wave, uint32_t source)
{
    const uint32_t total = ActiveTokenCount(args, source);
    if (total <= wave->token_begin) return 0u;
    const uint32_t remaining = total - wave->token_begin;
    return remaining < args->tokens_per_wave
        ? remaining : args->tokens_per_wave;
}

__aicore__ inline uint32_t GlobalSourceTokenOffset(
    __gm__ const FusionKernelArgs *args, uint32_t source)
{
    uint32_t offset = 0u;
    for (uint32_t other = 0u; other < source; ++other)
        offset += ActiveTokenCount(args, other);
    return offset;
}

__aicore__ inline uint32_t DenseWaveTokenStride(
    __gm__ const FusionKernelArgs *args,
    __gm__ const FusionWaveDesc *wave)
{
    uint32_t stride = 0u;
    for (uint32_t source = 0u; source < args->worker_count; ++source) {
        const uint32_t tokens = SourceWaveTokenCount(args, wave, source);
        if (tokens > stride) stride = tokens;
    }
    return stride;
}

__aicore__ inline uint64_t WaveGeneration(
    __gm__ const FusionKernelArgs *args, uint32_t wave)
{
    return OperationGeneration(args) + static_cast<uint64_t>(wave) + 1u;
}

__aicore__ inline __gm__ FusionSlotState *SlotState(
    __gm__ const FusionKernelArgs *args, uint32_t slot)
{
    return reinterpret_cast<__gm__ FusionSlotState *>(
        args->workspace + args->layout.slot_state_off) + slot;
}

__aicore__ inline __gm__ uint8_t *ReadyLine(
    __gm__ const FusionKernelArgs *args, uint32_t slot, uint32_t phase,
    uint32_t expert, uint32_t producer)
{
    const uint64_t phase_stride =
        static_cast<uint64_t>(args->activation_waves) * args->expert_count;
    const uint64_t line =
        ((static_cast<uint64_t>(slot) * 5u * phase_stride +
          static_cast<uint64_t>(phase) * phase_stride + expert) *
             args->ready_producers +
         producer);
    return reinterpret_cast<__gm__ uint8_t *>(args->workspace) +
           args->layout.tile_ready_off + line * kFusionCacheLineBytes;
}

__aicore__ inline __gm__ int64_t *GroupList(
    __gm__ const FusionKernelArgs *args, uint32_t wave, uint32_t slot)
{
    if (args->group_lists != 0u)
        return reinterpret_cast<__gm__ int64_t *>(args->group_lists) +
               static_cast<uint64_t>(wave) * args->local_expert_count;
    return reinterpret_cast<__gm__ int64_t *>(
        args->workspace + args->layout.expert_ready_off) +
        static_cast<uint64_t>(slot) * args->expert_count;
}

__aicore__ inline __gm__ uint8_t *IncProgressLine(
    __gm__ const FusionKernelArgs *args, uint32_t lane)
{
    return reinterpret_cast<__gm__ uint8_t *>(args->workspace) +
           args->layout.trace_off +
           static_cast<uint64_t>(lane) * kFusionCacheLineBytes;
}

__aicore__ inline __gm__ FusionLaneTrace *IncTraceLine(
    __gm__ const FusionKernelArgs *args, uint32_t lane)
{
    return reinterpret_cast<__gm__ FusionLaneTrace *>(
               reinterpret_cast<__gm__ uint8_t *>(args->workspace) +
               args->layout.trace_off) + lane;
}

__aicore__ inline void WriteWorkerTrace(
    __gm__ const FusionKernelArgs *args, uint32_t line, uint32_t role,
    uint64_t start)
{
    __gm__ FusionLaneTrace *trace = IncTraceLine(args, line);
    trace->start_cycle = start;
    trace->end_cycle = AscendC::GetSystemCycle();
    trace->role = role;
    trace->lane = line;
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(trace));
}

__aicore__ inline void WriteWorkerCheckpoint(
    __gm__ const FusionKernelArgs *args, uint32_t line, uint32_t checkpoint)
{
    if (checkpoint >= 5u) return;
    IncTraceLine(args, line)->reserved64[checkpoint] =
        AscendC::GetSystemCycle();
}

__aicore__ inline void WriteIncDispatchCheckpoint(
    __gm__ const FusionKernelArgs *args, uint32_t lane, uint32_t checkpoint)
{
    if (checkpoint >= 5u) return;
    IncTraceLine(args, lane)->reserved64[checkpoint] =
        AscendC::GetSystemCycle();
}

__aicore__ inline void WriteIncCombineCheckpoint(
    __gm__ const FusionKernelArgs *args, uint32_t owner,
    uint32_t checkpoint)
{
    if (checkpoint >= 5u) return;
    IncTraceLine(args, args->resources.inc_dispatch_aiv + owner)
        ->reserved64[checkpoint] = AscendC::GetSystemCycle();
}

__aicore__ inline void SetError(__gm__ const FusionKernelArgs *args,
                                uint32_t slot, uint32_t error,
                                uint32_t site = 0u,
                                uint64_t detail = 0u)
{
    __gm__ FusionSlotState *state = SlotState(args, slot);
    Dcci(reinterpret_cast<__gm__ uint8_t *>(state),
         sizeof(FusionSlotState));
    if (state->error != kFusionSlotOk) return;
    state->error = error;
    state->owner = site;
    state->reserved = detail;
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(state));
}

__aicore__ inline bool WaitSlotReuse(__gm__ const FusionKernelArgs *args,
                                     uint32_t wave, uint32_t slot)
{
    if (wave < args->slot_count) return true;
    const uint64_t previous = WaveGeneration(args, wave - args->slot_count);
    __gm__ uint8_t *line = reinterpret_cast<__gm__ uint8_t *>(
        &SlotState(args, slot)->release_generation);
    return WaitGeneration(line, previous, args->spin_cap);
}

__aicore__ inline uint32_t FloatBits(float value)
{
    union { float f; uint32_t u; } bits;
    bits.f = value;
    return bits.u;
}

// Dense Combine stamps the wave generation across the two expert id fields of
// each slot, so a slot belongs to this wave only when the pair matches.
__aicore__ inline uint64_t AssignmentGeneration(
    __gm__ const FusionExpertAssignment *assignment)
{
    return (static_cast<uint64_t>(assignment->local_expert) << 32u) |
           assignment->expert_id;
}

__aicore__ inline float BitsFloat(uint32_t value)
{
    union { float f; uint32_t u; } bits;
    bits.u = value;
    return bits.f;
}

__aicore__ inline void CopyGmToGm(
    AscendC::LocalTensor<uint8_t> &tile,
    __gm__ uint8_t *destination, __gm__ const uint8_t *source,
    uint64_t bytes)
{
    AscendC::GlobalTensor<uint8_t> src;
    AscendC::GlobalTensor<uint8_t> dst;
    src.SetGlobalBuffer(const_cast<__gm__ uint8_t *>(source));
    dst.SetGlobalBuffer(destination);
    uint64_t offset = 0u;
    while (offset < bytes) {
        const uint32_t count = static_cast<uint32_t>(
            bytes - offset < kCopyTileBytes ? bytes - offset :
                                              kCopyTileBytes);
        AscendC::DataCopyExtParams copy(1u, count, 0u, 0u, 0u);
        AscendC::DataCopyPadExtParams<uint8_t> pad;
        AscendC::DataCopyPad(tile, src[offset], copy, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(dst[offset], tile, copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        offset += count;
    }
}

__aicore__ inline bool PublishPacket(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint64_t header_off, uint64_t payload_off, uint32_t slot,
    uint32_t worker_index, uint32_t lane, uint32_t sequence,
    __gm__ const uint8_t *payload, uint32_t bytes,
    FusionPacketHeader metadata, int32_t remote_pe)
{
    metadata.payload_bytes = bytes;
    const uint64_t index = QueueIndex(args, slot, worker_index, lane, sequence);
    return Publish(sym, args, header_off, payload_off, index,
                   payload, bytes, metadata,
                   WaveGeneration(args, metadata.wave), sequence, remote_pe);
}

__aicore__ inline __gm__ FusionPacketHeader *WaitPacket(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint64_t header_off, uint32_t slot, uint32_t worker_index,
    uint32_t lane, uint32_t sequence, uint64_t generation)
{
    const uint64_t index = QueueIndex(args, slot, worker_index, lane, sequence);
    __gm__ FusionPacketHeader *header = Header(sym, header_off, index);
    return WaitReady(header, generation, sequence, args->spin_cap)
        ? header : nullptr;
}

__aicore__ inline __gm__ FusionPacketHeader *TryPacket(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint64_t header_off, uint32_t slot, uint32_t worker_index,
    uint32_t lane, uint32_t sequence, uint64_t generation)
{
    const uint64_t index = QueueIndex(args, slot, worker_index, lane, sequence);
    __gm__ FusionPacketHeader *header = Header(sym, header_off, index);
    Dcci(reinterpret_cast<__gm__ uint8_t *>(header), 64u);
    return header->ready == PacketCommit(generation, sequence)
        ? header : nullptr;
}

__aicore__ inline __gm__ uint8_t *PacketPayload(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint64_t payload_off, uint32_t slot, uint32_t worker_index,
    uint32_t lane, uint32_t sequence)
{
    return Payload(sym, args, payload_off,
                   QueueIndex(args, slot, worker_index, lane, sequence));
}

// Every lane of the Dispatch cohort writes into the remote wave arena, so all
// of them have to clear the same slot-reuse fence before the first putmem.
__aicore__ inline bool WaitDispatchUploadGate(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    __gm__ FusionWaveDesc *waves, uint32_t wave, uint32_t slot)
{
    if ((RequestFlags(args) & kFusionStrictSerialPipeline) != 0u &&
        wave != 0u) {
        const uint32_t previous_slot = waves[wave - 1u].slot;
        if (!WaitGeneration(reinterpret_cast<__gm__ uint8_t *>(
                &SlotState(args, previous_slot)->release_generation),
                WaveGeneration(args, wave - 1u), args->spin_cap)) {
            SetError(args, slot, kFusionSlotTimeout, 501u);
            return false;
        }
    }
    if (!WaitSlotReuse(args, wave, slot)) {
        SetError(args, slot, kFusionSlotTimeout, 502u);
        return false;
    }
    if (wave >= args->slot_count) {
        __gm__ FusionBulkControl *reuse_go = BulkControl(
            sym, args, slot,
            BulkReleaseBase(args) + args->worker_count + args->rank);
        if (!WaitBulkGeneration(reuse_go,
                WaveGeneration(args, wave - args->slot_count),
                args->spin_cap)) {
            SetError(args, slot, kFusionSlotTimeout, 509u);
            return false;
        }
    }
    return true;
}

// One AIV issuing putmem reaches roughly a quarter of an HCCS link, so the
// wave payload is cut into one contiguous token stripe per Dispatch lane.
// Lane zero owns stripe zero plus the route arrays and the doorbell; the
// receive lanes push the remaining stripes and report on the join line.
__aicore__ inline bool PushDispatchStripe(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    __gm__ FusionWaveDesc *waves, uint32_t wave, uint32_t lane,
    uint32_t lanes, uint64_t input_stride, uint32_t row_bytes)
{
    const uint32_t slot = waves[wave].slot;
    if (!WaitDispatchUploadGate(sym, args, waves, wave, slot)) return false;
    const uint32_t wave_tokens =
        SourceWaveTokenCount(args, &waves[wave], args->rank);
    const uint32_t stripe = wave_tokens / lanes;
    const uint32_t begin = stripe * lane;
    const uint32_t end = lane + 1u == lanes ? wave_tokens : stripe * (lane + 1u);
    if (end > begin) {
        aclshmem_putmem(
            BulkArena(sym, args, args->symmetric_layout.reserved64[0],
                      input_stride, slot, args->rank) +
                static_cast<uint64_t>(begin) * row_bytes,
            reinterpret_cast<__gm__ uint8_t *>(args->input) +
                static_cast<uint64_t>(waves[wave].token_begin + begin) *
                    row_bytes,
            static_cast<uint64_t>(end - begin) * row_bytes,
            static_cast<int32_t>(args->inc_pe));
        aclshmem_quiet();
    }
    PublishGeneration(
        ReadyLine(args, slot, kReadyDispatch, kDispatchTxJoinFlat, lane),
        WaveGeneration(args, wave));
    return true;
}

// Bulk single-INC Dispatch.  Rank zero of the worker Dispatch cohort uploads
// one contiguous input prefix plus two contiguous route arrays per wave.  INC
// lanes fan those immutable regions to every worker; receiver lanes filter
// by destination while packing expert-major rows.  For dense top-k MoE this
// replaces thousands of 4 KiB packet/credit round trips with O(W^2) puts.
__aicore__ inline void WorkerBulkDispatch(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint32_t logical_aiv)
{
    const uint32_t lanes = args->resources.worker_dispatch_aiv;
    if (logical_aiv >= lanes || lanes < 2u) return;
    const uint64_t trace_start = AscendC::GetSystemCycle();
    __gm__ FusionWaveDesc *waves =
        reinterpret_cast<__gm__ FusionWaveDesc *>(args->waves);
    __gm__ FusionDispatchRow *all_rows =
        reinterpret_cast<__gm__ FusionDispatchRow *>(args->dispatch_rows);
    __gm__ FusionExpertAssignment *all_assignments =
        reinterpret_cast<__gm__ FusionExpertAssignment *>(args->assignments);
    __gm__ uint32_t *expert_owner =
        reinterpret_cast<__gm__ uint32_t *>(args->expert_owner);
    const uint64_t input_stride = BulkInputStride(args);
    const uint64_t route_stride = BulkRouteStride(args);
    const uint64_t assignment_off =
        BulkSourceRows(args) * sizeof(FusionDispatchRow);
    const uint32_t row_bytes = args->hidden * sizeof(bfloat16_t);

    if (logical_aiv == 0u) {
        for (uint32_t wave = 0u; wave < args->wave_count; ++wave) {
            const uint32_t slot = waves[wave].slot;
            const uint64_t generation = WaveGeneration(args, wave);
            if (!WaitDispatchUploadGate(sym, args, waves, wave, slot)) return;
            WriteWorkerCheckpoint(args, 0u, 0u);
            const uint32_t wave_tokens =
                SourceWaveTokenCount(args, &waves[wave], args->rank);
            const uint64_t input_bytes =
                static_cast<uint64_t>(wave_tokens) * row_bytes;
            const uint32_t row_count = waves[wave].dispatch_row_count;
            const uint32_t assignment_count = waves[wave].assignment_count;
            __gm__ uint8_t *remote_input = BulkArena(
                sym, args, args->symmetric_layout.reserved64[0],
                input_stride, slot, args->rank);
            __gm__ uint8_t *remote_route = BulkArena(
                sym, args, args->symmetric_layout.reserved64[1],
                route_stride, slot, args->rank);
            // This lane owns the first token stripe; the receive lanes push
            // the remaining stripes from their own wave iteration, where they
            // would otherwise be spinning on the fanout doorbell.
            const uint64_t tx_bytes =
                static_cast<uint64_t>(wave_tokens / lanes) * row_bytes;
            if (tx_bytes != 0u)
                aclshmem_putmem(remote_input,
                    reinterpret_cast<__gm__ uint8_t *>(args->input) +
                        static_cast<uint64_t>(waves[wave].token_begin) *
                            row_bytes,
                    tx_bytes, static_cast<int32_t>(args->inc_pe));
            WriteWorkerCheckpoint(args, 0u, 1u);
            if (row_count != 0u)
                aclshmem_putmem(remote_route,
                    reinterpret_cast<__gm__ uint8_t *>(
                        all_rows + waves[wave].dispatch_row_begin),
                    static_cast<uint64_t>(row_count) *
                        sizeof(FusionDispatchRow),
                    static_cast<int32_t>(args->inc_pe));
            WriteWorkerCheckpoint(args, 0u, 2u);
            if (assignment_count != 0u)
                aclshmem_putmem(remote_route + assignment_off,
                    reinterpret_cast<__gm__ uint8_t *>(
                        all_assignments + waves[wave].assignment_begin),
                    static_cast<uint64_t>(assignment_count) *
                        sizeof(FusionExpertAssignment),
                    static_cast<int32_t>(args->inc_pe));
            WriteWorkerCheckpoint(args, 0u, 3u);
            // The control line is the release publication for all three
            // logically distinct payload regions.
            aclshmem_quiet();
            WriteWorkerCheckpoint(args, 0u, 4u);
            for (uint32_t other = 1u; other < lanes; ++other) {
                if (!WaitGeneration(ReadyLine(args, slot, kReadyDispatch,
                                              kDispatchTxJoinFlat, other),
                                    generation, args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout, 503u);
                    return;
                }
            }
            PublishBulkRemote(BulkControl(sym, args, slot, args->rank),
                generation, row_count, assignment_count, input_bytes,
                static_cast<uint64_t>(row_count) * sizeof(FusionDispatchRow) +
                    static_cast<uint64_t>(assignment_count) *
                        sizeof(FusionExpertAssignment),
                static_cast<int32_t>(args->inc_pe));
        }
        WriteWorkerTrace(args, 0u, 3u, trace_start);
        return;
    }

    const uint32_t rx = logical_aiv - 1u;
    const uint32_t rx_lanes = lanes - 1u;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> copy_buf;
    pipe.InitBuffer(copy_buf, kCopyTileBytes);
    auto copy_tile = copy_buf.Get<uint8_t>();
    const uint64_t assignment_capacity =
        args->layout.assignment_slot_bytes / row_bytes;
    // Waves that fit inside the slot ring carry no reuse fence, so their
    // stripes can all leave before this lane starts receiving.  Deferring them
    // to their own iteration would hold each doorbell behind the previous
    // wave's gather, which is exactly the latency the fanout has to hide.
    // It also lands the fanout inside the Combine window, where the whole-cache
    // sweeps race the arriving payload; see report/inc_dispatch_prefetch_race.
    const uint32_t prefetch_waves = args->wave_count < args->slot_count
        ? args->wave_count : args->slot_count;
    for (uint32_t wave = 0u; wave < prefetch_waves; ++wave) {
        if (!PushDispatchStripe(sym, args, waves, wave, rx + 1u, lanes,
                                input_stride, row_bytes))
            return;
    }
    for (uint32_t wave = 0u; wave < args->wave_count; ++wave) {
        const uint32_t slot = waves[wave].slot;
        const uint64_t generation = WaveGeneration(args, wave);
        // Pair-indexed receive controls are distinct from local worker upload
        // controls.  This lets RX pack each returned source while INC is
        // still forwarding the remaining pairs.
        __gm__ uint8_t *grouped =
            reinterpret_cast<__gm__ uint8_t *>(args->workspace) +
            args->layout.grouped_input_off +
            static_cast<uint64_t>(slot) * args->layout.assignment_slot_bytes;
        __gm__ FusionReceivedAssignment *received =
            reinterpret_cast<__gm__ FusionReceivedAssignment *>(
                args->workspace + args->layout.assignment_meta_off) +
            static_cast<uint64_t>(slot) * assignment_capacity;
        if (wave >= prefetch_waves &&
            !PushDispatchStripe(sym, args, waves, wave, rx + 1u, lanes,
                                input_stride, row_bytes))
            return;
        if (rx == 0u) WriteWorkerCheckpoint(args, 1u, 0u);
        for (uint32_t source = 0u; source < args->worker_count; ++source) {
            const uint32_t flat =
                source * args->worker_count + args->rank;
            __gm__ FusionBulkControl *uploaded = BulkControl(
                sym, args, slot,
                static_cast<uint64_t>(4u) * args->worker_count + flat);
            if (!WaitBulkGeneration(uploaded, generation, args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, 504u);
                return;
            }
            if (rx == 0u && source == 0u)
                WriteWorkerCheckpoint(args, 1u, 1u);
            // One whole-cache acquire per source doorbell covers that source's
            // payload, route array and assignment records at once.  Keeping it
            // here (rather than after a combined wait on every source) retains
            // the pipelining that lets this lane pack an arrived source while
            // the INC still forwards the remaining pairs, while removing the
            // per-record line acquires and the per-record pipeline drain that
            // used to sit in the metadata loop below.
            DcciAll(sym);
            const uint32_t row_count = uploaded->count0;
            const uint32_t assignment_count = uploaded->count1;
            if (row_count > BulkSourceRows(args) ||
                assignment_count > BulkSourceAssignments(args)) {
                SetError(args, slot, kFusionSlotBadRoute, 505u);
                return;
            }
            __gm__ uint8_t *input = BulkArena(sym, args,
                args->symmetric_layout.reserved64[0], input_stride,
                slot, source);
            __gm__ uint8_t *route = BulkArena(sym, args,
                args->symmetric_layout.reserved64[1], route_stride,
                slot, source);
            __gm__ FusionDispatchRow *rows =
                reinterpret_cast<__gm__ FusionDispatchRow *>(route);
            __gm__ FusionExpertAssignment *assignments =
                reinterpret_cast<__gm__ FusionExpertAssignment *>(
                    route + assignment_off);
            const uint32_t assignment_base = row_count == 0u
                ? 0u : rows[0].assignment_begin;
            // Rebuild control metadata from the complete assignment array.
            // Assignment indices are partitioned across RX AIVs and each
            // destination row is unique, so this remains deterministic with
            // one writer per cache line while avoiding an RX0 serial pass.
            for (uint32_t item = rx; item < assignment_count;
                 item += rx_lanes) {
                __gm__ FusionExpertAssignment *assignment =
                    &assignments[item];
                if (assignment->expert_id >= args->expert_count) {
                    SetError(args, slot, kFusionSlotBadRoute, 5061u);
                    return;
                }
                if (expert_owner[assignment->expert_id] != args->rank)
                    continue;
                if (assignment->wave != wave ||
                    assignment->destination_row >= assignment_capacity ||
                    assignment->route_ordinal >= args->topk) {
                    SetError(args, slot, kFusionSlotBadRoute, 5071u);
                    return;
                }
                __gm__ FusionReceivedAssignment *meta =
                    &received[assignment->destination_row];
                meta->source_rank = source;
                meta->source_token = assignment->destination_token;
                meta->expert_id = assignment->expert_id;
                meta->local_expert = assignment->local_expert;
                meta->route_ordinal = assignment->route_ordinal;
                meta->weight_bits = assignment->weight_bits;
                meta->destination_row = assignment->destination_row;
                meta->wave = wave;
            }
            for (uint32_t row_index = rx; row_index < row_count;
                 row_index += rx_lanes) {
                __gm__ FusionDispatchRow *row = &rows[row_index];
                if (row->destination_rank != args->rank) continue;
                if (row->source_rank != source ||
                    row->source_token < waves[wave].token_begin ||
                    row->source_token >= waves[wave].token_begin +
                        SourceWaveTokenCount(args, &waves[wave], source) ||
                    row->assignment_begin < assignment_base ||
                    row->assignment_count > assignment_count ||
                    row->assignment_begin - assignment_base >
                        assignment_count - row->assignment_count) {
                    SetError(args, slot, kFusionSlotBadRoute, 506u);
                    return;
                }
                const uint32_t local_assignment =
                    row->assignment_begin - assignment_base;
                const uint64_t input_row =
                    row->source_token - waves[wave].token_begin;
                // The payload row itself is identical for every expert
                // selected on this destination, so it is loaded into UB only
                // once per tile and fanned out through the ordered MTE3 queue.
                for (uint32_t item = 0u; item < row->assignment_count;
                     ++item) {
                    __gm__ FusionExpertAssignment *assignment =
                        &assignments[local_assignment + item];
                    if (assignment->destination_row >= assignment_capacity) {
                        SetError(args, slot, kFusionSlotBadRoute, 507u);
                        return;
                    }
                }
                AscendC::GlobalTensor<uint8_t> gm_input;
                gm_input.SetGlobalBuffer(
                    input + input_row * row_bytes);
                uint64_t offset = 0u;
                while (offset < row_bytes) {
                    const uint32_t count = static_cast<uint32_t>(
                        row_bytes - offset < kCopyTileBytes
                            ? row_bytes - offset : kCopyTileBytes);
                    AscendC::DataCopyExtParams copy(
                        1u, count, 0u, 0u, 0u);
                    AscendC::DataCopyPadExtParams<uint8_t> pad;
                    AscendC::DataCopyPad(
                        copy_tile, gm_input[offset], copy, pad);
                    AscendC::SetFlag<
                        AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<
                        AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                    for (uint32_t item = 0u;
                         item < row->assignment_count; ++item) {
                        __gm__ FusionExpertAssignment *assignment =
                            &assignments[local_assignment + item];
                        AscendC::GlobalTensor<uint8_t> gm_output;
                        gm_output.SetGlobalBuffer(grouped +
                            static_cast<uint64_t>(
                                assignment->destination_row) * row_bytes);
                        AscendC::DataCopyPad(
                            gm_output[offset], copy_tile, copy);
                    }
                    AscendC::SetFlag<
                        AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    AscendC::WaitFlag<
                        AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    offset += count;
                }
            }
        }
        // The readiness ticket below is the only ordering edge the Cube cohort
        // observes, so the gathered rows only have to be released once per
        // lane.  Doing it per assignment cost a 64-line sweep for every one of
        // them, which dwarfed the actual gather traffic.
        if (rx == 0u) WriteWorkerCheckpoint(args, 1u, 2u);
        AscendC::PipeBarrier<PIPE_ALL>();
        DcciAll(grouped);
        PublishGeneration(ReadyLine(args, slot, kReadyDispatch, 0u,
                                    rx + 1u), generation);
        if (rx == 0u) {
            WriteWorkerCheckpoint(args, 1u, 3u);
            for (uint32_t other = 0u; other < rx_lanes; ++other) {
                if (!WaitGeneration(ReadyLine(args, slot, kReadyDispatch,
                                              0u, other + 1u),
                                    generation, args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout, 508u);
                    return;
                }
            }
            // RX completion is wave-wide: all receiver lanes have drained
            // every source before lane zero reaches this point.  A single
            // release ticket therefore carries exactly the same ordering as
            // one ticket per local expert, without O(E * AIC) cache-line
            // publications and polls.
            PublishGeneration(ReadyLine(args, slot, kReadyDispatch,
                                        0u, 0u), generation);
            WriteWorkerCheckpoint(args, 1u, 4u);
            __gm__ FusionSlotState *state = SlotState(args, slot);
            state->dispatch_generation = generation;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(state));
        }
    }
    if (rx == 0u) WriteWorkerTrace(args, 1u, 4u, trace_start);
}

__aicore__ inline void IncBulkDispatch(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint32_t inc_lane)
{
    if (inc_lane >= args->resources.inc_dispatch_aiv) return;
    const uint64_t trace_start = AscendC::GetSystemCycle();
    __gm__ FusionWaveDesc *waves =
        reinterpret_cast<__gm__ FusionWaveDesc *>(args->waves);
    const uint64_t row_bytes =
        static_cast<uint64_t>(args->hidden) * sizeof(bfloat16_t);
    const uint64_t input_stride = BulkInputStride(args);
    const uint64_t route_stride = BulkRouteStride(args);
    for (uint32_t wave = 0u; wave < args->wave_count; ++wave) {
        const uint32_t slot = waves[wave].slot;
        const uint64_t generation = WaveGeneration(args, wave);
        if ((RequestFlags(args) & kFusionSerializeIncDc) != 0u &&
            wave != 0u) {
            const uint32_t previous_slot = waves[wave - 1u].slot;
            if (!WaitGeneration(reinterpret_cast<__gm__ uint8_t *>(
                    &SlotState(args, previous_slot)->combine_generation),
                    WaveGeneration(args, wave - 1u), args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, 511u);
                return;
            }
        }
        if (wave >= args->slot_count) {
            const uint64_t previous =
                WaveGeneration(args, wave - args->slot_count);
            if (inc_lane == 0u) {
                for (uint32_t worker = 0u;
                     worker < args->worker_count; ++worker) {
                    if (!WaitBulkGeneration(BulkControl(sym, args, slot,
                            BulkReleaseBase(args) + worker),
                            previous, args->spin_cap)) {
                        SetError(args, slot, kFusionSlotTimeout, 514u);
                        return;
                    }
                }
                for (uint32_t worker = 0u;
                     worker < args->worker_count; ++worker)
                    PublishBulkRemote(BulkControl(sym, args, slot,
                            BulkReleaseBase(args) + args->worker_count +
                                worker),
                        previous, 0u, 0u, 0u, 0u, WorkerPe(args, worker));
                PublishBulkLocal(BulkControl(sym, args, slot,
                        BulkReleaseBase(args) + args->worker_count),
                    previous, 0u, 0u, 0u, 0u);
            } else if (!WaitBulkGeneration(BulkControl(sym, args, slot,
                    BulkReleaseBase(args) + args->worker_count),
                    previous, args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, 515u);
                return;
            }
        }
        for (uint32_t source = inc_lane; source < args->worker_count;
             source += args->resources.inc_dispatch_aiv) {
            __gm__ FusionBulkControl *uploaded =
                BulkControl(sym, args, slot, source);
            if (inc_lane == 0u) WriteIncDispatchCheckpoint(args, 0u, 0u);
            if (!WaitBulkGeneration(uploaded, generation, args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, 512u);
                return;
            }
            if (inc_lane == 0u) WriteIncDispatchCheckpoint(args, 0u, 1u);
            const uint64_t expected_input_bytes =
                static_cast<uint64_t>(SourceWaveTokenCount(
                    args, &waves[wave], source)) * row_bytes;
            const uint64_t expected_route_bytes =
                static_cast<uint64_t>(uploaded->count0) *
                    sizeof(FusionDispatchRow) +
                static_cast<uint64_t>(uploaded->count1) *
                    sizeof(FusionExpertAssignment);
            if (uploaded->count0 > BulkSourceRows(args) ||
                uploaded->count1 > BulkSourceAssignments(args) ||
                uploaded->bytes0 != expected_input_bytes ||
                uploaded->bytes1 != expected_route_bytes) {
                SetError(args, slot, kFusionSlotBadRoute, 518u,
                         uploaded->bytes0);
                return;
            }
            __gm__ uint8_t *input = BulkArena(sym, args,
                args->symmetric_layout.reserved64[0], input_stride,
                slot, source);
            __gm__ uint8_t *route = BulkArena(sym, args,
                args->symmetric_layout.reserved64[1], route_stride,
                slot, source);
            // Only worker_count lanes reach this loop, so a ranged sweep of a
            // quarter-MiB payload lands on one lane at the head of the INC
            // forward, which is exactly the layer's pipeline fill.
            DcciAll(input);
            Dcci(route, uploaded->count0 * sizeof(FusionDispatchRow));
            Dcci(route + BulkSourceRows(args) * sizeof(FusionDispatchRow),
                 uploaded->count1 * sizeof(FusionExpertAssignment));
            if (uploaded->count0 != 0u) {
                __gm__ FusionDispatchRow *first =
                    reinterpret_cast<__gm__ FusionDispatchRow *>(route);
                if (first->source_rank != source || first->wave != wave) {
                    SetError(args, slot, kFusionSlotBadRoute, 519u);
                    return;
                }
            }
            if (uploaded->count1 != 0u) {
                __gm__ FusionExpertAssignment *first =
                    reinterpret_cast<__gm__ FusionExpertAssignment *>(
                        route + BulkSourceRows(args) *
                            sizeof(FusionDispatchRow));
                if (first->wave != wave) {
                    SetError(args, slot, kFusionSlotBadRoute, 5192u);
                    return;
                }
            }
            PublishBulkLocal(BulkControl(sym, args, slot,
                    BulkDispatchSourceBase(args) + source),
                generation, uploaded->count0, uploaded->count1,
                uploaded->bytes0, uploaded->bytes1);
        }

        // A single AIV reaches only a fraction of an HCCS link, so one lane per
        // (source, destination) pair leaves the fanout - the layer's pipeline
        // fill - running well under link rate.  Spread the spare lanes over the
        // pairs as byte stripes.  The stripe count is derived from the lane
        // budget at run time, so a device with fewer AIVs simply falls back to
        // one lane per pair and a device with more drives the link harder.
        const uint32_t pairs = args->worker_count * args->worker_count;
        const uint32_t lanes = args->resources.inc_dispatch_aiv;
        const uint32_t stripes = lanes / pairs < 1u ? 1u : lanes / pairs;
        const uint32_t units = pairs * stripes;
        for (uint32_t unit = inc_lane; unit < units; unit += lanes) {
            const uint32_t flat = unit / stripes;
            const uint32_t stripe = unit % stripes;
            const uint32_t source = flat / args->worker_count;
            const uint32_t destination = flat % args->worker_count;
            __gm__ FusionBulkControl *source_ready = BulkControl(
                sym, args, slot, BulkDispatchSourceBase(args) + source);
            if (!WaitBulkGeneration(source_ready, generation,
                                    args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, 5121u);
                return;
            }
            if (inc_lane == 0u) WriteIncDispatchCheckpoint(args, 0u, 2u);
            __gm__ uint8_t *input = BulkArena(sym, args,
                args->symmetric_layout.reserved64[0], input_stride,
                slot, source);
            __gm__ uint8_t *route = BulkArena(sym, args,
                args->symmetric_layout.reserved64[1], route_stride,
                slot, source);
            const uint64_t payload_rows = source_ready->bytes0 / row_bytes;
            const uint64_t begin_row = payload_rows * stripe / stripes;
            const uint64_t end_row = stripe + 1u == stripes
                ? payload_rows : payload_rows * (stripe + 1u) / stripes;
            if (end_row > begin_row)
                aclshmem_putmem(input + begin_row * row_bytes,
                                input + begin_row * row_bytes,
                                (end_row - begin_row) * row_bytes,
                                WorkerPe(args, destination));
            // The route arrays are two orders of magnitude smaller than the
            // payload, so splitting them would cost more in fixed per-call
            // overhead than it saves.
            if (stripe == 0u) {
                if (source_ready->count0 != 0u)
                    aclshmem_putmem(route, route,
                        static_cast<uint64_t>(source_ready->count0) *
                            sizeof(FusionDispatchRow),
                        WorkerPe(args, destination));
                if (source_ready->count1 != 0u)
                    aclshmem_putmem(
                        route + BulkSourceRows(args) *
                            sizeof(FusionDispatchRow),
                        route + BulkSourceRows(args) *
                            sizeof(FusionDispatchRow),
                        static_cast<uint64_t>(source_ready->count1) *
                            sizeof(FusionExpertAssignment),
                        WorkerPe(args, destination));
            }
            aclshmem_quiet();
            // Each stripe retires its own quiet before reporting, so the pair
            // doorbell below is ordered behind completed remote writes rather
            // than merely issued ones.
            __gm__ FusionBulkControl *stripe_done = BulkControl(
                sym, args, slot, BulkIncPrepBase(args) + unit);
            if (stripes != 1u) {
                PublishBulkLocal(stripe_done, generation, 0u, 0u, 0u, 0u);
                if (stripe != 0u) continue;
                for (uint32_t other = 1u; other < stripes; ++other) {
                    if (!WaitBulkGeneration(BulkControl(sym, args, slot,
                            BulkIncPrepBase(args) + flat * stripes + other),
                            generation, args->spin_cap)) {
                        SetError(args, slot, kFusionSlotTimeout, 5122u);
                        return;
                    }
                }
            }
            __gm__ FusionBulkControl *pair_done = BulkControl(
                sym, args, slot,
                static_cast<uint64_t>(4u) * args->worker_count + flat);
            aclshmem_putmem(pair_done, source_ready,
                            sizeof(FusionBulkControl),
                            WorkerPe(args, destination));
            PublishBulkLocal(pair_done,
                generation, 0u, 0u, 0u, 0u);
            if (inc_lane == 0u) WriteIncDispatchCheckpoint(args, 0u, 3u);
        }
        if (inc_lane == 0u) {
            for (uint32_t flat = 0u; flat < pairs; ++flat) {
                if (!WaitBulkGeneration(BulkControl(sym, args, slot,
                        static_cast<uint64_t>(4u) * args->worker_count + flat),
                        generation, args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout, 513u);
                    return;
                }
            }
            for (uint32_t destination = 0u;
                 destination < args->worker_count; ++destination)
                PublishBulkRemote(BulkControl(sym, args, slot,
                        args->worker_count + destination),
                    generation, 0u, 0u, 0u, 0u,
                    WorkerPe(args, destination));
            if ((RequestFlags(args) & kFusionSerializeIncDc) != 0u) {
                __gm__ FusionSlotState *state = SlotState(args, slot);
                state->dispatch_generation = generation;
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(state));
            }
            WriteIncDispatchCheckpoint(args, 0u, 4u);
        }
    }
    __gm__ FusionLaneTrace *trace = IncTraceLine(args, inc_lane);
    trace->start_cycle = trace_start;
    trace->end_cycle = AscendC::GetSystemCycle();
    trace->role = 1u;
    trace->lane = inc_lane;
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(trace));
}

__aicore__ inline void WorkerDispatchTx(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint32_t logical_aiv)
{
    const uint32_t tx_lanes =
        args->resources.worker_dispatch_aiv > 1u
            ? args->resources.worker_dispatch_aiv / 2u : 1u;
    if (logical_aiv >= tx_lanes) return;
    __gm__ FusionWaveDesc *waves =
        reinterpret_cast<__gm__ FusionWaveDesc *>(args->waves);
    __gm__ FusionDispatchRow *rows =
        reinterpret_cast<__gm__ FusionDispatchRow *>(args->dispatch_rows);
    __gm__ FusionExpertAssignment *assignments =
        reinterpret_cast<__gm__ FusionExpertAssignment *>(args->assignments);
    __gm__ uint8_t *input = reinterpret_cast<__gm__ uint8_t *>(args->input);
    const uint32_t row_bytes = args->hidden * sizeof(bfloat16_t);
    for (uint32_t wave = 0u; wave < args->wave_count; ++wave) {
        const uint32_t slot = waves[wave].slot;
        if ((RequestFlags(args) & kFusionStrictSerialPipeline) != 0u &&
            wave != 0u) {
            const uint32_t previous_slot = waves[wave - 1u].slot;
            if (!WaitGeneration(
                    reinterpret_cast<__gm__ uint8_t *>(
                        &SlotState(args, previous_slot)->release_generation),
                    WaveGeneration(args, wave - 1u), args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, 103u);
                return;
            }
        }
        if (!WaitSlotReuse(args, wave, slot)) {
            SetError(args, slot, kFusionSlotTimeout, 101u);
            return;
        }
        uint32_t sequence = 0u;
        for (uint32_t local = logical_aiv;
             local < waves[wave].dispatch_row_count; local += tx_lanes) {
            __gm__ FusionDispatchRow *row =
                &rows[waves[wave].dispatch_row_begin + local];
            uint32_t offset = 0u;
            while (offset < row_bytes) {
                const uint32_t bytes = row_bytes - offset <
                        args->transport_tile_bytes
                    ? row_bytes - offset : args->transport_tile_bytes;
                FusionPacketHeader packet{};
                packet.wave = wave;
                packet.source_rank = args->rank;
                packet.destination_rank = row->destination_rank;
                packet.source_token = row->source_token;
                packet.payload_offset = offset;
                packet.kind = kFusionDispatchPayload;
                if (!PublishPacket(sym, args,
                        args->symmetric_layout.dispatch_header_off,
                        args->symmetric_layout.dispatch_payload_off,
                        slot, args->rank, logical_aiv, sequence++,
                        input + row->payload_byte_offset + offset, bytes,
                        packet, static_cast<int32_t>(args->inc_pe))) {
                    SetError(args, slot, kFusionSlotTimeout);
                    return;
                }
                offset += bytes;
            }
            for (uint32_t item = 0u; item < row->assignment_count; ++item) {
                __gm__ FusionExpertAssignment *assignment =
                    &assignments[row->assignment_begin + item];
                FusionPacketHeader packet{};
                packet.wave = wave;
                packet.source_rank = args->rank;
                packet.destination_rank = row->destination_rank;
                packet.source_token = row->source_token;
                packet.destination_token = assignment->destination_token;
                packet.expert_id = assignment->expert_id;
                packet.route_ordinal = assignment->route_ordinal;
                packet.weight_bits = assignment->weight_bits;
                packet.payload_offset = assignment->destination_row;
                packet.kind = kFusionDispatchAssignment;
                if (!PublishPacket(sym, args,
                        args->symmetric_layout.dispatch_header_off,
                        args->symmetric_layout.dispatch_payload_off,
                        slot, args->rank, logical_aiv, sequence++,
                        nullptr, 0u, packet,
                        static_cast<int32_t>(args->inc_pe))) {
                    SetError(args, slot, kFusionSlotTimeout);
                    return;
                }
            }
        }
        FusionPacketHeader end{};
        end.wave = wave;
        end.source_rank = args->rank;
        end.kind = kFusionDispatchLaneEnd;
        if (!PublishPacket(sym, args,
                args->symmetric_layout.dispatch_header_off,
                args->symmetric_layout.dispatch_payload_off,
                slot, args->rank, logical_aiv, sequence,
                nullptr, 0u, end, static_cast<int32_t>(args->inc_pe))) {
            SetError(args, slot, kFusionSlotTimeout, 102u);
            return;
        }
    }
}

// Worker-direct SHMEM dispatch uses one SPSC queue per
// (destination, source, source-lane). The destination is encoded in the
// queue's worker axis and (source, source-lane) in its lane axis. Besides
// making every remote receive stream unique, this also gives the producer a
// distinct local credit shadow for every destination; reusing only
// (source, lane) would let concurrent destinations overwrite one another.
__aicore__ inline void WorkerDirectDispatchTx(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint32_t logical_aiv)
{
    const uint32_t tx_lanes =
        args->resources.worker_dispatch_aiv > 1u
            ? args->resources.worker_dispatch_aiv / 2u : 1u;
    if (logical_aiv >= tx_lanes) return;
    __gm__ FusionWaveDesc *waves =
        reinterpret_cast<__gm__ FusionWaveDesc *>(args->waves);
    __gm__ FusionDispatchRow *rows =
        reinterpret_cast<__gm__ FusionDispatchRow *>(args->dispatch_rows);
    __gm__ FusionExpertAssignment *assignments =
        reinterpret_cast<__gm__ FusionExpertAssignment *>(args->assignments);
    __gm__ uint8_t *input = reinterpret_cast<__gm__ uint8_t *>(args->input);
    const uint32_t row_bytes = args->hidden * sizeof(bfloat16_t);
    for (uint32_t wave = 0u; wave < args->wave_count; ++wave) {
        const uint32_t slot = waves[wave].slot;
        if ((RequestFlags(args) & kFusionStrictSerialPipeline) != 0u &&
            wave != 0u) {
            const uint32_t previous_slot = waves[wave - 1u].slot;
            if (!WaitGeneration(
                    reinterpret_cast<__gm__ uint8_t *>(
                        &SlotState(args, previous_slot)->release_generation),
                    WaveGeneration(args, wave - 1u), args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, 403u);
                return;
            }
        }
        if (!WaitSlotReuse(args, wave, slot)) {
            SetError(args, slot, kFusionSlotTimeout, 401u);
            return;
        }
        uint32_t sequence[kFusionMaxWorkers] = {};
        for (uint32_t local = logical_aiv;
             local < waves[wave].dispatch_row_count; local += tx_lanes) {
            __gm__ FusionDispatchRow *row =
                &rows[waves[wave].dispatch_row_begin + local];
            const uint32_t destination = row->destination_rank;
            if (destination >= args->worker_count) {
                SetError(args, slot, kFusionSlotBadRoute, 404u);
                return;
            }
            uint32_t offset = 0u;
            while (offset < row_bytes) {
                const uint32_t bytes = row_bytes - offset <
                        args->transport_tile_bytes
                    ? row_bytes - offset : args->transport_tile_bytes;
                FusionPacketHeader packet{};
                packet.wave = wave;
                packet.source_rank = args->rank;
                packet.destination_rank = destination;
                packet.source_token = row->source_token;
                packet.payload_offset = offset;
                packet.kind = kFusionDispatchPayload;
                if (!PublishPacket(sym, args,
                        args->symmetric_layout.dispatch_result_header_off,
                        args->symmetric_layout.dispatch_result_payload_off,
                        slot, destination,
                        args->rank * tx_lanes + logical_aiv,
                        sequence[destination]++,
                        input + row->payload_byte_offset + offset, bytes,
                        packet, WorkerPe(args, destination))) {
                    SetError(args, slot, kFusionSlotTimeout, 405u);
                    return;
                }
                offset += bytes;
            }
            for (uint32_t item = 0u; item < row->assignment_count; ++item) {
                __gm__ FusionExpertAssignment *assignment =
                    &assignments[row->assignment_begin + item];
                FusionPacketHeader packet{};
                packet.wave = wave;
                packet.source_rank = args->rank;
                packet.destination_rank = destination;
                packet.source_token = row->source_token;
                packet.destination_token = assignment->destination_token;
                packet.expert_id = assignment->expert_id;
                packet.route_ordinal = assignment->route_ordinal;
                packet.weight_bits = assignment->weight_bits;
                packet.payload_offset = assignment->destination_row;
                packet.kind = kFusionDispatchAssignment;
                if (!PublishPacket(sym, args,
                        args->symmetric_layout.dispatch_result_header_off,
                        args->symmetric_layout.dispatch_result_payload_off,
                        slot, destination,
                        args->rank * tx_lanes + logical_aiv,
                        sequence[destination]++, nullptr, 0u, packet,
                        WorkerPe(args, destination))) {
                    SetError(args, slot, kFusionSlotTimeout, 406u);
                    return;
                }
            }
        }
        // Every destination consumes every source/lane queue.  Empty streams
        // therefore still need an explicit end record.
        for (uint32_t destination = 0u;
             destination < args->worker_count; ++destination) {
            FusionPacketHeader end{};
            end.wave = wave;
            end.source_rank = args->rank;
            end.destination_rank = destination;
            end.kind = kFusionDispatchLaneEnd;
            if (!PublishPacket(sym, args,
                    args->symmetric_layout.dispatch_result_header_off,
                    args->symmetric_layout.dispatch_result_payload_off,
                    slot, destination,
                    args->rank * tx_lanes + logical_aiv,
                    sequence[destination],
                    nullptr, 0u, end, WorkerPe(args, destination))) {
                SetError(args, slot, kFusionSlotTimeout, 402u);
                return;
            }
        }
    }
}

__aicore__ inline void WorkerDirectDispatchRx(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint32_t logical_aiv)
{
    const uint32_t tx_lanes =
        args->resources.worker_dispatch_aiv > 1u
            ? args->resources.worker_dispatch_aiv / 2u : 1u;
    const uint32_t rx_lanes = args->resources.worker_dispatch_aiv - tx_lanes;
    if (logical_aiv < tx_lanes || logical_aiv >= tx_lanes + rx_lanes) return;
    const uint32_t rx = logical_aiv - tx_lanes;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> copy_buf;
    pipe.InitBuffer(copy_buf, kCopyTileBytes);
    auto copy_tile = copy_buf.Get<uint8_t>();
    __gm__ FusionWaveDesc *waves =
        reinterpret_cast<__gm__ FusionWaveDesc *>(args->waves);
    const uint32_t row_bytes = args->hidden * sizeof(bfloat16_t);
    for (uint32_t wave = 0u; wave < args->wave_count; ++wave) {
        const uint32_t slot = waves[wave].slot;
        const uint64_t generation = WaveGeneration(args, wave);
        __gm__ uint8_t *dispatch_ring =
            reinterpret_cast<__gm__ uint8_t *>(args->workspace) +
            args->layout.dispatch_ring_off +
            static_cast<uint64_t>(slot) * args->layout.dispatch_slot_bytes;
        __gm__ uint8_t *grouped =
            reinterpret_cast<__gm__ uint8_t *>(args->workspace) +
            args->layout.grouped_input_off +
            static_cast<uint64_t>(slot) * args->layout.assignment_slot_bytes;
        __gm__ FusionReceivedAssignment *received =
            reinterpret_cast<__gm__ FusionReceivedAssignment *>(
                args->workspace + args->layout.assignment_meta_off) +
            static_cast<uint64_t>(slot) *
                (args->layout.assignment_slot_bytes / row_bytes);
        const uint32_t streams = args->worker_count * tx_lanes;
        for (uint32_t flat = rx; flat < streams; flat += rx_lanes) {
            const uint32_t source = flat / tx_lanes;
            const uint32_t source_lane = flat % tx_lanes;
            uint32_t sequence = 0u;
            while (true) {
                __gm__ FusionPacketHeader *header = WaitPacket(
                    sym, args,
                    args->symmetric_layout.dispatch_result_header_off,
                    slot, args->rank,
                    source * tx_lanes + source_lane,
                    sequence, generation);
                if (header == nullptr) {
                    SetError(args, slot, kFusionSlotTimeout, 410u);
                    return;
                }
                const uint32_t kind = header->kind;
                if (kind == kFusionDispatchPayload) {
                    uint32_t bad_site = 0u;
                    if (header->destination_rank != args->rank)
                        bad_site = 4111u;
                    else if (header->source_rank != source)
                        bad_site = 4112u;
                    else if (header->source_token < waves[wave].token_begin)
                        bad_site = 4113u;
                    else if (header->source_token >= waves[wave].token_begin +
                                 SourceWaveTokenCount(
                                     args, &waves[wave], source))
                        bad_site = 4114u;
                    else if (header->payload_offset + header->payload_bytes >
                             row_bytes)
                        bad_site = 4115u;
                    if (bad_site != 0u) {
                        const uint64_t detail =
                            (static_cast<uint64_t>(header->destination_rank) << 48u) |
                            (static_cast<uint64_t>(header->source_rank) << 32u) |
                            header->source_token;
                        SetError(args, slot, kFusionSlotBadRoute, bad_site,
                                 detail);
                        return;
                    }
                    const uint64_t row = static_cast<uint64_t>(source) *
                            waves[wave].token_count +
                        header->source_token - waves[wave].token_begin;
                    __gm__ uint8_t *payload = PacketPayload(
                        sym, args,
                        args->symmetric_layout.dispatch_result_payload_off,
                        slot, args->rank,
                        source * tx_lanes + source_lane,
                        sequence);
                    Dcci(payload, header->payload_bytes);
                    CopyGmToGm(copy_tile,
                        dispatch_ring + row * row_bytes +
                            header->payload_offset,
                        payload, header->payload_bytes);
                } else if (kind == kFusionDispatchAssignment) {
                    const uint32_t destination_row = header->payload_offset;
                    const uint64_t capacity =
                        args->layout.assignment_slot_bytes / row_bytes;
                    if (header->destination_rank != args->rank ||
                        header->source_rank != source ||
                        destination_row >= capacity) {
                        SetError(args, slot, kFusionSlotBadRoute, 412u);
                        return;
                    }
                    const uint64_t row = static_cast<uint64_t>(source) *
                            waves[wave].token_count +
                        header->source_token - waves[wave].token_begin;
                    CopyGmToGm(copy_tile,
                        grouped + static_cast<uint64_t>(destination_row) *
                            row_bytes,
                        dispatch_ring + row * row_bytes, row_bytes);
                    received[destination_row].source_rank = source;
                    received[destination_row].source_token =
                        header->destination_token;
                    received[destination_row].expert_id = header->expert_id;
                    received[destination_row].route_ordinal =
                        header->route_ordinal;
                    received[destination_row].weight_bits = header->weight_bits;
                    received[destination_row].destination_row = destination_row;
                    received[destination_row].wave = wave;
                    AscendC::PipeBarrier<PIPE_ALL>();
                    Dcci(reinterpret_cast<__gm__ uint8_t *>(
                             &received[destination_row]),
                         sizeof(FusionReceivedAssignment));
                }
                Release(header, sequence, WorkerPe(args, source));
                ++sequence;
                if (kind == kFusionDispatchLaneEnd) break;
            }
        }
        PublishGeneration(
            ReadyLine(args, slot, kReadyDispatch, 0u, rx + 1u), generation);
        if (rx == 0u) {
            for (uint32_t other = 0u; other < rx_lanes; ++other) {
                if (!WaitGeneration(ReadyLine(args, slot, kReadyDispatch,
                                              0u, other + 1u),
                                    generation, args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout, 413u);
                    return;
                }
            }
            PublishGeneration(ReadyLine(args, slot, kReadyDispatch,
                                        0u, 0u), generation);
            __gm__ FusionSlotState *state = SlotState(args, slot);
            state->dispatch_generation = generation;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(state));
        }
    }
}

__aicore__ inline void WorkerDispatchRx(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint32_t logical_aiv)
{
    const uint32_t tx_lanes =
        args->resources.worker_dispatch_aiv > 1u
            ? args->resources.worker_dispatch_aiv / 2u : 1u;
    const uint32_t rx_lanes = args->resources.worker_dispatch_aiv - tx_lanes;
    if (logical_aiv < tx_lanes || logical_aiv >= tx_lanes + rx_lanes) return;
    const uint32_t rx = logical_aiv - tx_lanes;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> copy_buf;
    pipe.InitBuffer(copy_buf, kCopyTileBytes);
    auto copy_tile = copy_buf.Get<uint8_t>();
    __gm__ FusionWaveDesc *waves =
        reinterpret_cast<__gm__ FusionWaveDesc *>(args->waves);
    const uint32_t row_bytes = args->hidden * sizeof(bfloat16_t);
    for (uint32_t wave = 0u; wave < args->wave_count; ++wave) {
        const uint32_t slot = waves[wave].slot;
        const uint64_t generation = WaveGeneration(args, wave);
        __gm__ uint8_t *dispatch_ring =
            reinterpret_cast<__gm__ uint8_t *>(args->workspace) +
            args->layout.dispatch_ring_off +
            static_cast<uint64_t>(slot) * args->layout.dispatch_slot_bytes;
        __gm__ uint8_t *grouped =
            reinterpret_cast<__gm__ uint8_t *>(args->workspace) +
            args->layout.grouped_input_off +
            static_cast<uint64_t>(slot) * args->layout.assignment_slot_bytes;
        __gm__ FusionReceivedAssignment *received =
            reinterpret_cast<__gm__ FusionReceivedAssignment *>(
                args->workspace + args->layout.assignment_meta_off) +
            static_cast<uint64_t>(slot) *
                (args->layout.assignment_slot_bytes / row_bytes);
        for (uint32_t inc_lane = rx; inc_lane <
                 args->resources.inc_dispatch_aiv; inc_lane += rx_lanes) {
            uint32_t sequence = 0u;
            while (true) {
                __gm__ FusionPacketHeader *header = WaitPacket(
                    sym, args,
                    args->symmetric_layout.dispatch_result_header_off,
                    slot, args->rank, inc_lane, sequence, generation);
                if (header == nullptr) {
                    SetError(args, slot, kFusionSlotTimeout, 300u);
                    return;
                }
                const uint32_t kind = header->kind;
                if (kind == kFusionDispatchPayload) {
                    if (header->source_token < waves[wave].token_begin ||
                        header->source_token >= waves[wave].token_begin +
                            waves[wave].token_count ||
                        header->source_rank >= args->worker_count ||
                        header->payload_offset + header->payload_bytes >
                            row_bytes) {
                        SetError(args, slot, kFusionSlotBadRoute);
                        return;
                    }
                    const uint64_t row =
                        static_cast<uint64_t>(header->source_rank) *
                            waves[wave].token_count +
                        header->source_token - waves[wave].token_begin;
                    __gm__ uint8_t *payload = PacketPayload(
                        sym, args,
                        args->symmetric_layout.dispatch_result_payload_off,
                        slot, args->rank, inc_lane, sequence);
                    Dcci(payload, header->payload_bytes);
                    CopyGmToGm(copy_tile,
                               dispatch_ring + row * row_bytes +
                                   header->payload_offset,
                               payload, header->payload_bytes);
                } else if (kind == kFusionDispatchAssignment) {
                    const uint32_t destination_row = header->payload_offset;
                    const uint64_t capacity =
                        args->layout.assignment_slot_bytes / row_bytes;
                    if (destination_row >= capacity ||
                        header->source_rank >= args->worker_count ||
                        header->source_token < waves[wave].token_begin ||
                        header->source_token >= waves[wave].token_begin +
                            waves[wave].token_count) {
                        SetError(args, slot, kFusionSlotBadRoute);
                        return;
                    }
                    const uint64_t row =
                        static_cast<uint64_t>(header->source_rank) *
                            waves[wave].token_count +
                        header->source_token - waves[wave].token_begin;
                    CopyGmToGm(copy_tile,
                               grouped +
                                   static_cast<uint64_t>(destination_row) *
                                       row_bytes,
                               dispatch_ring + row * row_bytes, row_bytes);
                    received[destination_row].source_rank =
                        header->source_rank;
                    received[destination_row].source_token =
                        header->destination_token;
                    received[destination_row].expert_id = header->expert_id;
                    received[destination_row].route_ordinal =
                        header->route_ordinal;
                    received[destination_row].weight_bits =
                        header->weight_bits;
                    received[destination_row].destination_row =
                        destination_row;
                    received[destination_row].wave = wave;
                    AscendC::PipeBarrier<PIPE_ALL>();
                    Dcci(reinterpret_cast<__gm__ uint8_t *>(
                             &received[destination_row]),
                         sizeof(FusionReceivedAssignment));
                }
                Release(header, sequence, static_cast<int32_t>(args->inc_pe));
                ++sequence;
                if (kind == kFusionDispatchLaneEnd) break;
            }
        }
        __gm__ uint8_t *done = ReadyLine(
            args, slot, kReadyDispatch, 0u, rx + 1u);
        PublishGeneration(done, generation);
        if (rx == 0u) {
            for (uint32_t other = 0u; other < rx_lanes; ++other) {
                if (!WaitGeneration(ReadyLine(args, slot, kReadyDispatch,
                                              0u, other + 1u),
                                    generation, args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout);
                    return;
                }
            }
            PublishGeneration(ReadyLine(args, slot, kReadyDispatch,
                                        0u, 0u), generation);
            __gm__ FusionSlotState *state = SlotState(args, slot);
            state->dispatch_generation = generation;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(state));
        }
    }
}

__aicore__ inline void IncDispatch(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint32_t inc_lane)
{
    const uint32_t worker_tx_lanes =
        args->resources.worker_dispatch_aiv > 1u
            ? args->resources.worker_dispatch_aiv / 2u : 1u;
    if (inc_lane >= args->resources.inc_dispatch_aiv) return;
    const uint64_t trace_start = AscendC::GetSystemCycle();
    __gm__ FusionWaveDesc *waves =
        reinterpret_cast<__gm__ FusionWaveDesc *>(args->waves);
    for (uint32_t wave = 0u; wave < args->wave_count; ++wave) {
        const uint32_t slot = waves[wave].slot;
        const uint64_t generation = WaveGeneration(args, wave);
        if ((RequestFlags(args) & kFusionSerializeIncDc) != 0u &&
            wave != 0u) {
            const uint32_t previous_slot = waves[wave - 1u].slot;
            __gm__ uint8_t *done = reinterpret_cast<__gm__ uint8_t *>(
                &SlotState(args, previous_slot)->combine_generation);
            if (!WaitGeneration(done, WaveGeneration(args, wave - 1u),
                                args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, 305u);
                return;
            }
        }
        uint32_t output_sequence[kFusionMaxWorkers] = {};
        const uint32_t streams = args->worker_count * worker_tx_lanes;
        for (uint32_t flat = inc_lane; flat < streams;
             flat += args->resources.inc_dispatch_aiv) {
            const uint32_t source = flat / worker_tx_lanes;
            const uint32_t source_lane = flat % worker_tx_lanes;
            uint32_t sequence = 0u;
            while (true) {
                __gm__ FusionPacketHeader *header = WaitPacket(
                    sym, args, args->symmetric_layout.dispatch_header_off,
                    slot, source, source_lane, sequence, generation);
                if (header == nullptr) {
                    SetError(args, slot, kFusionSlotTimeout, 304u);
                    return;
                }
                const uint32_t kind = header->kind;
                if (kind != kFusionDispatchLaneEnd) {
                    const uint32_t destination = header->destination_rank;
                    if (destination >= args->worker_count) {
                        SetError(args, slot, kFusionSlotBadRoute, 301u);
                        return;
                    }
                    FusionPacketHeader metadata{};
                    metadata.wave = header->wave;
                    metadata.source_rank = header->source_rank;
                    metadata.destination_rank = header->destination_rank;
                    metadata.source_token = header->source_token;
                    metadata.destination_token = header->destination_token;
                    metadata.expert_id = header->expert_id;
                    metadata.route_ordinal = header->route_ordinal;
                    metadata.weight_bits = header->weight_bits;
                    metadata.payload_offset = header->payload_offset;
                    metadata.kind = header->kind;
                    __gm__ uint8_t *payload = PacketPayload(
                        sym, args,
                        args->symmetric_layout.dispatch_payload_off,
                        slot, source, source_lane, sequence);
                    if (header->payload_bytes != 0u)
                        Dcci(payload, header->payload_bytes);
                    if (!PublishPacket(sym, args,
                            args->symmetric_layout.dispatch_result_header_off,
                            args->symmetric_layout.dispatch_result_payload_off,
                            slot, destination, inc_lane,
                            output_sequence[destination]++, payload,
                            header->payload_bytes, metadata,
                            WorkerPe(args, destination))) {
                        SetError(args, slot, kFusionSlotTimeout, 302u);
                        return;
                    }
                }
                Release(header, sequence, WorkerPe(args, source));
                ++sequence;
                if (kind == kFusionDispatchLaneEnd) break;
            }
        }
        for (uint32_t destination = 0u;
             destination < args->worker_count; ++destination) {
            FusionPacketHeader end{};
            end.wave = wave;
            end.destination_rank = destination;
            end.kind = kFusionDispatchLaneEnd;
            if (!PublishPacket(sym, args,
                    args->symmetric_layout.dispatch_result_header_off,
                    args->symmetric_layout.dispatch_result_payload_off,
                    slot, destination, inc_lane,
                    output_sequence[destination], nullptr, 0u, end,
                    WorkerPe(args, destination))) {
                SetError(args, slot, kFusionSlotTimeout, 303u);
                return;
            }
        }
        if ((RequestFlags(args) & kFusionSerializeIncDc) != 0u) {
            PublishGeneration(IncProgressLine(args, inc_lane), generation);
            if (inc_lane == 0u) {
                if (!WaitAllGenerations(
                        IncProgressLine(args, 0u),
                        args->resources.inc_dispatch_aiv,
                        generation, args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout, 306u);
                    return;
                }
                __gm__ FusionSlotState *state = SlotState(args, slot);
                state->dispatch_generation = generation;
                AscendC::PipeBarrier<PIPE_ALL>();
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(state));
            }
        }
    }
    __gm__ FusionLaneTrace *trace = IncTraceLine(args, inc_lane);
    trace->start_cycle = trace_start;
    trace->end_cycle = AscendC::GetSystemCycle();
    trace->role = 1u;
    trace->lane = inc_lane;
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(trace));
}

__aicore__ inline void WorkerComputeAic(
    __gm__ const FusionKernelArgs *args)
{
    const uint64_t trace_start = AscendC::GetSystemCycle();
    __gm__ FusionWaveDesc *waves =
        reinterpret_cast<__gm__ FusionWaveDesc *>(args->waves);
    __gm__ uint8_t *workspace =
        reinterpret_cast<__gm__ uint8_t *>(args->workspace);
    for (uint32_t wave = 0u; wave < args->wave_count; ++wave) {
        const uint32_t slot = waves[wave].slot;
        const uint64_t generation = WaveGeneration(args, wave);
        __gm__ bfloat16_t *grouped = reinterpret_cast<__gm__ bfloat16_t *>(
            workspace + args->layout.grouped_input_off +
            static_cast<uint64_t>(slot) * args->layout.assignment_slot_bytes);
        __gm__ bfloat16_t *gate_up = reinterpret_cast<__gm__ bfloat16_t *>(
            workspace + args->layout.gate_up_off +
            static_cast<uint64_t>(slot) *
                (args->layout.assignment_slot_bytes /
                 (args->hidden * sizeof(bfloat16_t))) *
                args->intermediate * 2u * sizeof(bfloat16_t));
        __gm__ bfloat16_t *activation =
            reinterpret_cast<__gm__ bfloat16_t *>(
                workspace + args->layout.activation_off +
                static_cast<uint64_t>(slot) *
                    (args->layout.assignment_slot_bytes /
                     (args->hidden * sizeof(bfloat16_t))) *
                    args->intermediate * sizeof(bfloat16_t));
        __gm__ bfloat16_t *expert_output =
            reinterpret_cast<__gm__ bfloat16_t *>(
                workspace + args->layout.expert_output_off +
                static_cast<uint64_t>(slot) *
                    args->layout.assignment_slot_bytes);
        __gm__ int64_t *groups = GroupList(args, wave, slot);
        const uint32_t core =
            static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t cores =
            static_cast<uint32_t>(AscendC::GetBlockNum());
        const bool grouped_cube =
            args->local_expert_count > args->resources.live_aic;
        if (grouped_cube) {
            const bool row_major_b =
                (args->flags & kFusionWeightBRowMajor) != 0u;
            if (!WaitGeneration(
                    ReadyLine(args, slot, kReadyDispatch, 0u, 0u),
                    generation, args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, 115u);
                return;
            }
            if (core == 0u) WriteWorkerCheckpoint(args, 2u, 0u);
            const int64_t total_rows_i64 =
                groups[args->local_expert_count - 1u];
            if (total_rows_i64 < 0) {
                SetError(args, slot, kFusionSlotBadRoute, 116u);
                return;
            }
            const uint32_t total_rows =
                static_cast<uint32_t>(total_rows_i64);
            if (total_rows != 0u) {
                RunGroupedBf16AllReady(
                    grouped,
                    reinterpret_cast<__gm__ bfloat16_t *>(args->w13),
                    gate_up, groups, total_rows, args->local_expert_count,
                    2u * args->intermediate, args->hidden, row_major_b);
            }
            PublishGeneration(
                ReadyLine(args, slot, kReadyGmm1, 0u, core), generation);
            // Every Cube core stops writing gate_up before any line is
            // acquired, then each core releases its own writes.  SwiGLU
            // consumes the joined tickets, so the cohort still observes a
            // fully acquired buffer without any core walking the buffer.
            if (!WaitAllGenerations(
                    ReadyLine(args, slot, kReadyGmm1, 0u, 0u),
                    args->resources.live_aic, generation,
                    args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, 1161u);
                return;
            }
            if (core == 0u) WriteWorkerCheckpoint(args, 2u, 4u);
            // dcci publishes to the device point of coherency, which the whole
            // cluster shares, so one sweep after the cohort has stopped
            // writing covers every line of the handoff buffer.  Repeating it
            // on all live_aic cores queued that many whole-cache sweeps on the
            // cache controller and added no ordering the join above lacks.
            if (core == 0u) {
                DcciAll(reinterpret_cast<__gm__ uint8_t *>(gate_up));
                PublishGeneration(
                    ReadyLine(args, slot, kReadyGmm1,
                              kGroupedGateAcquiredFlat, 0u),
                    generation);
                WriteWorkerCheckpoint(args, 2u, 1u);
            }
            if (!WaitAllGenerations(
                    ReadyLine(args, slot, kReadyActivation, 0u, 0u),
                    args->resources.worker_compute_aiv,
                    generation, args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, 117u);
                return;
            }
            if (core == 0u) WriteWorkerCheckpoint(args, 2u, 2u);
            // AIV MTE3 stores are complete when all activation tickets are
            // visible, but Cube-side cache acquisition must happen after the
            // entire cohort stops writing.  A dedicated GMM1 ticket line
            // carries that edge to every Cube core without a host sync.
            __gm__ uint8_t *activation_acquired = ReadyLine(
                args, slot, kReadyGmm1, kGroupedActivationAcquiredFlat, 0u);
            if (core == 0u) {
                PublishGeneration(activation_acquired, generation);
            }
            if (!WaitGeneration(activation_acquired, generation,
                                args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, 1171u);
                return;
            }
            if (total_rows != 0u) {
                RunGroupedBf16AllReady(
                    activation,
                    reinterpret_cast<__gm__ bfloat16_t *>(args->w2),
                    expert_output, groups, total_rows,
                    args->local_expert_count, args->hidden,
                    args->intermediate, row_major_b);
            }
            PublishGeneration(
                ReadyLine(args, slot, kReadyGmm2, 0u, core), generation);
            if (core == 0u) WriteWorkerCheckpoint(args, 2u, 3u);
            continue;
        }
        int64_t previous = 0;
        uint32_t start_core = 0u;
        if (!WaitGeneration(
                ReadyLine(args, slot, kReadyDispatch, 0u, 0u),
                generation, args->spin_cap)) {
            SetError(args, slot, kFusionSlotTimeout, 110u);
            return;
        }
        if (core == 0u) WriteWorkerCheckpoint(args, 2u, 0u);
        // Stage all GMM1 expert slices first.  Vector cores consume each slice
        // as soon as its per-core readiness lines are complete, while Cube
        // cores continue issuing later experts.  The former expert-local
        // GMM1 -> SwiGLU -> GMM2 loop forced every Cube core to wait for the
        // activation cohort once per expert/slice; that turns a sparse MoE
        // with dozens of local experts into thousands of fine-grained global
        // handshakes.  Keeping token-wave as the outer pipeline while
        // separating the two grouped GEMM phases removes those handshakes and
        // still overlaps GMM1 with SwiGLU.
        for (uint32_t expert = 0u;
             expert < args->local_expert_count; ++expert) {
            const int64_t end = groups[expert];
            if (end < previous) {
                SetError(args, slot, kFusionSlotBadRoute, 113u);
                return;
            }
            const uint32_t rows = static_cast<uint32_t>(end - previous);
            for (uint32_t activation_wave = 0u;
                 activation_wave < args->activation_waves;
                 ++activation_wave) {
                const uint32_t flat =
                    expert * args->activation_waves + activation_wave;
                const uint32_t begin = static_cast<uint32_t>(
                    static_cast<uint64_t>(rows) * activation_wave /
                    args->activation_waves);
                const uint32_t finish = static_cast<uint32_t>(
                    static_cast<uint64_t>(rows) * (activation_wave + 1u) /
                    args->activation_waves);
                const uint32_t wave_rows = finish - begin;
                if (wave_rows != 0u) {
                    const uint64_t packed_row =
                        static_cast<uint64_t>(previous) + begin;
                    RunBf16Slice(
                        grouped + packed_row * args->hidden,
                        reinterpret_cast<__gm__ bfloat16_t *>(args->w13) +
                            static_cast<uint64_t>(expert) *
                                2u * args->intermediate * args->hidden,
                        gate_up + packed_row * 2u * args->intermediate,
                        wave_rows, 2u * args->intermediate,
                        args->hidden, start_core,
                        (args->flags & kFusionWeightBRowMajor) != 0u);
                }
                PublishGeneration(
                    ReadyLine(args, slot, kReadyGmm1, flat, core),
                    generation);
                start_core = (start_core + 1u) % cores;
            }
            previous = end;
        }
        if (core == 0u) WriteWorkerCheckpoint(args, 2u, 1u);

        // By the time GMM1 has traversed the local expert list, most early
        // activations are already ready.  GMM2 retains expert/slice readiness
        // and publishes the same fine-grained edges consumed by Combine, so
        // communication can still start before the complete token wave exits
        // the FFN.
        previous = 0;
        for (uint32_t expert = 0u;
             expert < args->local_expert_count; ++expert) {
            const int64_t end = groups[expert];
            if (end < previous) {
                SetError(args, slot, kFusionSlotBadRoute, 114u);
                return;
            }
            const uint32_t rows = static_cast<uint32_t>(end - previous);
            for (uint32_t activation_wave = 0u;
                 activation_wave < args->activation_waves;
                 ++activation_wave) {
                const uint32_t flat =
                    expert * args->activation_waves + activation_wave;
                if (!WaitAllGenerations(
                        ReadyLine(args, slot, kReadyActivation, flat, 0u),
                        args->resources.worker_compute_aiv,
                        generation, args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout, 111u);
                    return;
                }
                if (core == 0u && expert == 0u && activation_wave == 0u)
                    WriteWorkerCheckpoint(args, 2u, 2u);
                const uint32_t begin = static_cast<uint32_t>(
                    static_cast<uint64_t>(rows) * activation_wave /
                    args->activation_waves);
                const uint32_t finish = static_cast<uint32_t>(
                    static_cast<uint64_t>(rows) * (activation_wave + 1u) /
                    args->activation_waves);
                const uint32_t wave_rows = finish - begin;
                if (wave_rows != 0u) {
                    const uint64_t packed_row =
                        static_cast<uint64_t>(previous) + begin;
                    RunBf16Slice(
                        activation + packed_row * args->intermediate,
                        reinterpret_cast<__gm__ bfloat16_t *>(args->w2) +
                            static_cast<uint64_t>(expert) *
                                args->hidden * args->intermediate,
                        expert_output + packed_row * args->hidden,
                        wave_rows, args->hidden, args->intermediate,
                        start_core,
                        (args->flags & kFusionWeightBRowMajor) != 0u);
                }
                PublishGeneration(
                    ReadyLine(args, slot, kReadyGmm2, flat, core),
                    generation);
                start_core = (start_core + 1u) % cores;
            }
            previous = end;
        }
        if (core == 0u) WriteWorkerCheckpoint(args, 2u, 3u);
    }
    if (static_cast<uint32_t>(AscendC::GetBlockIdx()) == 0u)
        WriteWorkerTrace(args, 2u, 5u, trace_start);
}

__aicore__ inline void WorkerActivation(
    __gm__ const FusionKernelArgs *args, uint32_t logical_aiv)
{
    const uint32_t compute_begin = args->resources.worker_dispatch_aiv +
                                   args->resources.worker_combine_aiv;
    if (logical_aiv < compute_begin ||
        logical_aiv >= compute_begin + args->resources.worker_compute_aiv)
        return;
    const uint32_t compute_lane = logical_aiv - compute_begin;
    __gm__ FusionWaveDesc *waves =
        reinterpret_cast<__gm__ FusionWaveDesc *>(args->waves);
    __gm__ uint8_t *workspace =
        reinterpret_cast<__gm__ uint8_t *>(args->workspace);
    const uint64_t assignment_capacity =
        args->layout.assignment_slot_bytes /
        (args->hidden * sizeof(bfloat16_t));
    for (uint32_t wave = 0u; wave < args->wave_count; ++wave) {
        const uint32_t slot = waves[wave].slot;
        const uint64_t generation = WaveGeneration(args, wave);
        __gm__ bfloat16_t *gate_up = reinterpret_cast<__gm__ bfloat16_t *>(
            workspace + args->layout.gate_up_off +
            static_cast<uint64_t>(slot) * assignment_capacity *
                args->intermediate * 2u * sizeof(bfloat16_t));
        __gm__ bfloat16_t *activation =
            reinterpret_cast<__gm__ bfloat16_t *>(
                workspace + args->layout.activation_off +
                static_cast<uint64_t>(slot) * assignment_capacity *
                args->intermediate * sizeof(bfloat16_t));
        if (args->local_expert_count > args->resources.live_aic) {
            // One Cube core acquires gate_up once the cohort has stopped
            // writing, so this single ticket orders the SwiGLU cohort behind
            // all of GMM1 and guarantees the whole buffer is visible before
            // the first load.
            if (!WaitGeneration(
                    ReadyLine(args, slot, kReadyGmm1,
                              kGroupedGateAcquiredFlat, 0u),
                    generation, args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, 1121u);
                return;
            }
            const int64_t total_rows_i64 =
                args->local_expert_count == 0u ? 0 :
                GroupList(args, wave, slot)[args->local_expert_count - 1u];
            if (total_rows_i64 < 0) {
                SetError(args, slot, kFusionSlotBadRoute, 118u);
                return;
            }
            if (!kSerialActivationDiagnostic || compute_lane == 0u)
                RunBf16SwiGluAllReady(
                    gate_up, activation,
                    static_cast<uint64_t>(total_rows_i64),
                    args->intermediate,
                    kSerialActivationDiagnostic ? 0u : compute_lane,
                    kSerialActivationDiagnostic
                        ? 1u : args->resources.worker_compute_aiv);
            PublishGeneration(
                ReadyLine(args, slot, kReadyActivation, 0u, compute_lane),
                generation);
            continue;
        }
        if (!RunBf16SwiGlu(
                gate_up, activation, GroupList(args, wave, slot),
                ReadyLine(args, slot, kReadyGmm1, 0u, 0u),
                ReadyLine(args, slot, kReadyActivation, 0u, 0u),
                args->ready_producers, args->resources.live_aic,
                args->ready_producers, compute_lane,
                args->resources.worker_compute_aiv,
                args->local_expert_count, args->intermediate,
                args->activation_waves, generation, args->spin_cap)) {
            SetError(args, slot, kFusionSlotTimeout, 112u);
            return;
        }
    }
}

__aicore__ inline void WeightedStore(
    AscendC::LocalTensor<bfloat16_t> &input_bf16,
    AscendC::LocalTensor<bfloat16_t> &output_bf16,
    AscendC::LocalTensor<float> &input_fp32,
    __gm__ uint8_t *accumulator, __gm__ uint8_t *payload,
    uint32_t bytes, float weight)
{
    AscendC::GlobalTensor<bfloat16_t> gm_input;
    AscendC::GlobalTensor<bfloat16_t> gm_output;
    gm_input.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(payload));
    gm_output.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t *>(accumulator));
    const uint32_t elements = bytes / sizeof(bfloat16_t);
    uint32_t offset = 0u;
    while (offset < elements) {
        const uint32_t count = elements - offset < kReduceTileElements
            ? elements - offset : kReduceTileElements;
        AscendC::DataCopyExtParams copy(
            1u, count * sizeof(bfloat16_t), 0u, 0u, 0u);
        AscendC::DataCopyPadExtParams<bfloat16_t> pad;
        AscendC::DataCopyPad(input_bf16, gm_input[offset], copy, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(input_fp32, input_bf16,
                      AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(input_fp32, input_fp32, weight, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(output_bf16, input_fp32,
                      AscendC::RoundMode::CAST_RINT, count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(gm_output[offset], output_bf16, copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        offset += count;
    }
}

__aicore__ inline void WeightedAdd(
    AscendC::LocalTensor<bfloat16_t> &input_bf16_0,
    AscendC::LocalTensor<bfloat16_t> &input_bf16_1,
    AscendC::LocalTensor<bfloat16_t> &acc_bf16_0,
    AscendC::LocalTensor<bfloat16_t> &acc_bf16_1,
    AscendC::LocalTensor<float> &input_fp32_0,
    AscendC::LocalTensor<float> &input_fp32_1,
    AscendC::LocalTensor<float> &acc_fp32_0,
    AscendC::LocalTensor<float> &acc_fp32_1,
    __gm__ uint8_t *accumulator, __gm__ uint8_t *payload,
    uint32_t bytes, float weight);

// For top-k >= 2, reduce all expert rows owned by this worker into one BF16
// partial per source token before crossing the INC link.  The old protocol
// sent K BF16 expert rows plus dense route metadata and made the INC's small
// AIV cohort perform every weighted reduction.  This hierarchical form keeps
// the same stateless star semantics, but distributes the weighted work over
// all workers and lets the INC sum only W dense partials.  The partial uses
// the model dtype to keep the INC uplink bounded; the INC accumulates the W
// inputs in FP32.  The existing top-k=1 path remains the lean fast path.
__aicore__ inline void WorkerBulkDenseCombine(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint32_t logical_aiv)
{
    const uint32_t begin = args->resources.worker_dispatch_aiv;
    const uint32_t lanes = args->resources.worker_combine_aiv;
    if (logical_aiv < begin || logical_aiv >= begin + lanes) return;
    const uint32_t lane = logical_aiv - begin;
    const uint64_t trace_start = AscendC::GetSystemCycle();
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_bf16_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_fp32_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_fp32_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_bf16_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> copy_buf;
    pipe.InitBuffer(input_bf16_buf,
                    kDenseReduceTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(input_fp32_buf,
                    kDenseReduceTileElements * sizeof(float));
    pipe.InitBuffer(acc_fp32_buf,
                    kDenseReduceTileElements * sizeof(float));
    pipe.InitBuffer(acc_bf16_buf,
                    kDenseReduceTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(copy_buf, kCopyTileBytes);
    auto input_bf16 = input_bf16_buf.Get<bfloat16_t>();
    auto input_fp32 = input_fp32_buf.Get<float>();
    auto acc_fp32 = acc_fp32_buf.Get<float>();
    auto acc_bf16 = acc_bf16_buf.Get<bfloat16_t>();
    auto copy_tile = copy_buf.Get<uint8_t>();
    __gm__ FusionWaveDesc *waves =
        reinterpret_cast<__gm__ FusionWaveDesc *>(args->waves);
    const uint32_t row_bytes = args->hidden * sizeof(bfloat16_t);
    const uint64_t input_stride = BulkInputStride(args);
    const uint64_t expert_stride = BulkExpertStride(args);
    const uint64_t assignment_capacity =
        args->layout.assignment_slot_bytes / row_bytes;
    for (uint32_t wave = 0u; wave < args->wave_count; ++wave) {
        const uint32_t slot = waves[wave].slot;
        const uint64_t generation = WaveGeneration(args, wave);
        const int64_t assignment_count_i64 =
            args->local_expert_count == 0u ? 0 :
            GroupList(args, wave, slot)[args->local_expert_count - 1u];
        if (assignment_count_i64 < 0 ||
            static_cast<uint64_t>(assignment_count_i64) >
                assignment_capacity) {
            SetError(args, slot, kFusionSlotBadRoute, 551u);
            return;
        }
        if (!WaitGeneration(reinterpret_cast<__gm__ uint8_t *>(
                &SlotState(args, slot)->dispatch_generation),
                generation, args->spin_cap)) {
            SetError(args, slot, kFusionSlotTimeout, 552u);
            return;
        }
        WriteWorkerCheckpoint(args, 3u, 0u);
        const uint32_t assignment_count =
            static_cast<uint32_t>(assignment_count_i64);
        const uint32_t dense_token_stride =
            DenseWaveTokenStride(args, &waves[wave]);
        const uint32_t dense_rows =
            args->worker_count * dense_token_stride;
        const uint64_t dense_count =
            static_cast<uint64_t>(dense_rows) * args->topk;
        const uint64_t partial_bytes =
            static_cast<uint64_t>(dense_rows) * row_bytes;
        if (dense_count > assignment_capacity ||
            partial_bytes > args->layout.assignment_slot_bytes ||
            partial_bytes > expert_stride) {
            SetError(args, slot, kFusionSlotBadRoute, 553u,
                     partial_bytes);
            return;
        }
        __gm__ uint8_t *bulk_result = BulkArena(sym, args,
            args->symmetric_layout.reserved64[0], input_stride,
            slot, args->rank);
        const uint32_t source_tokens =
            SourceWaveTokenCount(args, &waves[wave], args->rank);
        const uint64_t result_bytes =
            static_cast<uint64_t>(source_tokens) * row_bytes;
        // INC writes the reduced result into bulk_result, so the buffer must be
        // acquired before the payload upload releases INC.  Every combine lane
        // reaches the upload join below, so acquiring here is ordered exactly
        // like a single-lane sweep while costing one instruction.
        DcciAll(bulk_result);
        __gm__ uint8_t *expert_output =
            reinterpret_cast<__gm__ uint8_t *>(args->workspace) +
                args->layout.expert_output_off +
            static_cast<uint64_t>(slot) *
                args->layout.assignment_slot_bytes;
        __gm__ uint8_t *remote = BulkArena(sym, args,
            args->symmetric_layout.reserved64[2], expert_stride,
            slot, args->rank);
        __gm__ FusionExpertAssignment *combine_meta =
            reinterpret_cast<__gm__ FusionExpertAssignment *>(remote);
        __gm__ uint8_t *metadata_ready =
            ReadyLine(args, slot, kReadyDispatch, 1u, 0u);
        if (lane == 0u) {
            __gm__ FusionReceivedAssignment *received =
                reinterpret_cast<__gm__ FusionReceivedAssignment *>(
                    args->workspace + args->layout.assignment_meta_off) +
                static_cast<uint64_t>(slot) * assignment_capacity;
            DcciAll(reinterpret_cast<__gm__ uint8_t *>(received));
            for (uint32_t row = 0u; row < assignment_count; ++row) {
                __gm__ FusionReceivedAssignment *meta = &received[row];
                if (meta->wave != wave || meta->destination_row != row ||
                    meta->source_rank >= args->worker_count ||
                    meta->source_token < waves[wave].token_begin ||
                    meta->source_token - waves[wave].token_begin >=
                        SourceWaveTokenCount(
                            args, &waves[wave], meta->source_rank) ||
                    meta->route_ordinal >= args->topk) {
                    uint64_t reason = 0u;
                    if (meta->wave != wave) reason |= 1u;
                    if (meta->destination_row != row) reason |= 2u;
                    if (meta->source_rank >= args->worker_count) reason |= 4u;
                    if (meta->source_token < waves[wave].token_begin)
                        reason |= 8u;
                    if (meta->source_rank < args->worker_count &&
                        meta->source_token - waves[wave].token_begin >=
                            SourceWaveTokenCount(
                                args, &waves[wave], meta->source_rank))
                        reason |= 16u;
                    if (meta->route_ordinal >= args->topk) reason |= 32u;
                    SetError(args, slot, kFusionSlotBadRoute, 554u,
                        (reason << 56u) |
                        (static_cast<uint64_t>(meta->source_rank & 0xffu)
                            << 48u) |
                        (static_cast<uint64_t>(meta->wave & 0xffu) << 40u) |
                        (static_cast<uint64_t>(meta->route_ordinal & 0xffu)
                            << 32u) |
                        (static_cast<uint64_t>(meta->source_token & 0xffffu)
                            << 16u) |
                        (meta->destination_row & 0xffffu));
                    return;
                }
                const uint64_t dense_index =
                    (static_cast<uint64_t>(meta->source_rank) *
                         dense_token_stride +
                     meta->source_token - waves[wave].token_begin) *
                        args->topk + meta->route_ordinal;
                if (dense_index >= dense_count) {
                    SetError(args, slot, kFusionSlotBadRoute, 555u,
                             dense_index);
                    return;
                }
                __gm__ FusionExpertAssignment *compact =
                    &combine_meta[dense_index];
                compact->dispatch_row = meta->source_rank;
                compact->expert_id = static_cast<uint32_t>(generation);
                compact->local_expert =
                    static_cast<uint32_t>(generation >> 32u);
                compact->route_ordinal = meta->route_ordinal;
                compact->destination_token = meta->source_token;
                compact->weight_bits = meta->weight_bits;
                compact->wave = meta->wave;
                compact->destination_row = row;
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            DcciAll(reinterpret_cast<__gm__ uint8_t *>(combine_meta));
            PublishGeneration(metadata_ready, generation);
        } else if (!WaitGeneration(metadata_ready, generation,
                                   args->spin_cap)) {
            SetError(args, slot, kFusionSlotTimeout, 556u);
            return;
        }

        const bool grouped_cube =
            args->local_expert_count > args->resources.live_aic;
        if (grouped_cube) {
            if (!WaitAllGenerations(
                    ReadyLine(args, slot, kReadyGmm2, 0u, 0u),
                    args->resources.live_aic, generation,
                    args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, 557u);
                return;
            }
        } else {
            for (uint32_t expert = 0u;
                 expert < args->local_expert_count; ++expert) {
                for (uint32_t activation_wave = 0u;
                     activation_wave < args->activation_waves;
                     ++activation_wave) {
                    const uint32_t flat =
                        expert * args->activation_waves + activation_wave;
                    if (!WaitAllGenerations(
                            ReadyLine(args, slot, kReadyGmm2, flat, 0u),
                            args->resources.live_aic, generation,
                            args->spin_cap)) {
                        SetError(args, slot, kFusionSlotTimeout, 558u);
                        return;
                    }
                }
            }
        }

        // GMM2 completion is a release publication from the AIC cohort, and
        // the tickets waited on above cover every expert row this lane will
        // read.  One whole-cache acquire therefore replaces the per-row sweep
        // that used to run once per (row, ordinal) inside the reduction.
        DcciAll(expert_output);
        if (lane == 0u) WriteWorkerCheckpoint(args, 3u, 1u);
        __gm__ uint8_t *partial =
            reinterpret_cast<__gm__ uint8_t *>(args->workspace) +
            args->layout.combine_ring_off +
            static_cast<uint64_t>(slot) *
                args->layout.assignment_slot_bytes;
        AscendC::GlobalTensor<bfloat16_t> gm_partial;
        gm_partial.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t *>(partial));
        const uint32_t row_begin = static_cast<uint32_t>(
            static_cast<uint64_t>(dense_rows) * lane / lanes);
        const uint32_t row_end = static_cast<uint32_t>(
            static_cast<uint64_t>(dense_rows) * (lane + 1u) / lanes);
        for (uint32_t row = row_begin; row < row_end; ++row) {
            Dcci(reinterpret_cast<__gm__ uint8_t *>(
                     combine_meta + static_cast<uint64_t>(row) * args->topk),
                 args->topk * sizeof(FusionExpertAssignment));
            for (uint32_t column = 0u; column < args->hidden;
                 column += kDenseReduceTileElements) {
                const uint32_t count =
                    args->hidden - column < kDenseReduceTileElements
                        ? args->hidden - column : kDenseReduceTileElements;
                AscendC::Duplicate(acc_fp32, 0.0f, count);
                // The zero-contribution case (common when top-k < workers)
                // reaches the final Cast without an intervening Axpy.  Make
                // the accumulator clear visible unconditionally; otherwise
                // Cast can consume the previous row's UB contents.
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t ordinal = 0u;
                     ordinal < args->topk; ++ordinal) {
                    __gm__ FusionExpertAssignment *assignment =
                        &combine_meta[
                            static_cast<uint64_t>(row) * args->topk +
                            ordinal];
                    if (AssignmentGeneration(assignment) != generation)
                        continue;
                    if (assignment->destination_row >= assignment_count) {
                        SetError(args, slot, kFusionSlotBadRoute, 559u,
                                 assignment->destination_row);
                        return;
                    }
                    AscendC::GlobalTensor<bfloat16_t> gm_input;
                    gm_input.SetGlobalBuffer(
                        reinterpret_cast<__gm__ bfloat16_t *>(
                            expert_output +
                            static_cast<uint64_t>(
                                assignment->destination_row) * row_bytes));
                    AscendC::DataCopyExtParams copy_in(
                        1u, count * sizeof(bfloat16_t), 0u, 0u, 0u);
                    AscendC::DataCopyPadExtParams<bfloat16_t> pad(
                        false, 0, 0, 0);
                    AscendC::DataCopyPad(
                        input_bf16, gm_input[column], copy_in, pad);
                    AscendC::SetFlag<
                        AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<
                        AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::Cast(input_fp32, input_bf16,
                        AscendC::RoundMode::CAST_NONE, count);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Axpy(acc_fp32, input_fp32,
                        BitsFloat(assignment->weight_bits), count);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<
                        AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                    AscendC::WaitFlag<
                        AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                }
                AscendC::Cast(acc_bf16, acc_fp32,
                              AscendC::RoundMode::CAST_RINT, count);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopyExtParams copy_out(
                    1u, count * sizeof(bfloat16_t), 0u, 0u, 0u);
                const uint64_t element =
                    static_cast<uint64_t>(row) * args->hidden + column;
                AscendC::DataCopyPad(gm_partial[element],
                                    acc_bf16, copy_out);
                AscendC::SetFlag<
                    AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::WaitFlag<
                    AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
        const uint64_t lane_bytes =
            static_cast<uint64_t>(row_end - row_begin) * row_bytes;
        if (lane_bytes != 0u) {
            AscendC::PipeBarrier<PIPE_ALL>();
            DcciAll(partial);
        }
        // Every lane ships the slice it just reduced and retires it with its
        // own quiet, so the upload uses the whole AIV cohort rather than
        // draining a megabyte through lane 0.  A lane only publishes its
        // ticket once its quiet has returned, so the join below observes
        // completed remote writes and not merely issued ones; that is what
        // keeps a single doorbell from lane 0 correctly ordered behind the
        // whole payload.
        //
        // The payload needs no further acquire: every lane released its own
        // slice above, covering exactly [0, partial_bytes).
        const uint64_t lane_offset =
            static_cast<uint64_t>(row_begin) * row_bytes;
        if (lane_bytes != 0u) {
            aclshmem_putmem(remote + lane_offset, partial + lane_offset,
                            lane_bytes, static_cast<int32_t>(args->inc_pe));
            aclshmem_quiet();
        }
        const uint32_t upload_ready_producer =
            args->ready_producers - lanes + lane;
        PublishGeneration(ReadyLine(args, slot, kReadyDispatch, 0u,
                                    upload_ready_producer), generation);
        const bool global_output =
            (RequestFlags(args) & kFusionGlobalOutputFanout) != 0u;
        if (lane == 0u) {
            WriteWorkerCheckpoint(args, 3u, 3u);
            for (uint32_t other = 0u; other < lanes; ++other) {
                if (!WaitGeneration(
                        ReadyLine(args, slot, kReadyDispatch, 0u,
                            args->ready_producers - lanes + other),
                        generation, args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout, 560u);
                    return;
                }
            }
            WriteWorkerCheckpoint(args, 3u, 2u);
            PublishBulkRemote(BulkControl(sym, args, slot,
                    static_cast<uint64_t>(2u) * args->worker_count +
                        args->rank),
                generation, dense_rows, kBulkDenseBf16Protocol,
                partial_bytes, 0u, static_cast<int32_t>(args->inc_pe));
            if (global_output)
                PublishGeneration(ReadyLine(args, slot,
                    kReadyCombineResult, 0u, 0u), generation);
        } else if (!global_output) {
            continue;
        } else if (!WaitGeneration(ReadyLine(args, slot,
                       kReadyCombineResult, 0u, 0u),
                   generation, args->spin_cap)) {
            SetError(args, slot, kFusionSlotTimeout, 5601u);
            return;
        }

        const bool source_partition =
            args->resources.inc_combine_aiv >= args->worker_count &&
            args->resources.inc_combine_aiv % args->worker_count == 0u;
        const uint64_t combine_barrier_base =
            static_cast<uint64_t>(4u) * args->worker_count +
            static_cast<uint64_t>(args->worker_count) * args->worker_count;
        if (global_output) {
            // Each lane owns independent source ranks.  It can consume and
            // copy a source as soon as that source's INC owners publish it;
            // no all-source barrier is introduced on the worker.
            for (uint32_t source = lane; source < args->worker_count;
                 source += lanes) {
                __gm__ FusionBulkControl *source_result = BulkControl(
                    sym, args, slot,
                    static_cast<uint64_t>(3u) * args->worker_count + source);
                if (!WaitBulkGeneration(
                        source_result, generation, args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout, 5612u, source);
                    return;
                }
                // The per-source result ticket is published only after every
                // destination lane has completed its put.  Waiting the old
                // per-slice controls as well would serialize an already
                // established release edge and prevent destination sharding.
                const uint32_t source_tokens =
                    SourceWaveTokenCount(args, &waves[wave], source);
                const uint64_t source_bytes =
                    static_cast<uint64_t>(source_tokens) * row_bytes;
                if (source_result->bytes0 != source_bytes) {
                    SetError(args, slot, kFusionSlotBadRoute, 5621u,
                             source_result->bytes0);
                    return;
                }
                __gm__ uint8_t *source_result_payload = BulkArena(
                    sym, args, args->symmetric_layout.reserved64[0],
                    input_stride, slot, source);
                DcciAll(source_result_payload);
                CopyGmToGm(copy_tile,
                    reinterpret_cast<__gm__ uint8_t *>(args->output) +
                        (static_cast<uint64_t>(
                             GlobalSourceTokenOffset(args, source)) +
                         waves[wave].token_begin) * row_bytes,
                    source_result_payload, source_bytes);
            }
            PublishGeneration(ReadyLine(args, slot, kReadyCombineResult,
                                        1u, lane), generation);
            if (lane != 0u) continue;
            for (uint32_t other = 0u; other < lanes; ++other) {
                if (!WaitGeneration(ReadyLine(args, slot,
                        kReadyCombineResult, 1u, other),
                        generation, args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout, 5622u, other);
                    return;
                }
            }
            WriteWorkerCheckpoint(args, 3u, 4u);
            __gm__ FusionSlotState *state = SlotState(args, slot);
            state->combine_generation = generation;
            state->output_generation = generation;
            state->release_generation = generation;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(state));
            PublishBulkRemote(BulkControl(sym, args, slot,
                    BulkReleaseBase(args) + args->rank),
                generation, 0u, 0u, 0u, 0u,
                static_cast<int32_t>(args->inc_pe));
            continue;
        }

        __gm__ FusionBulkControl *result = BulkControl(sym, args, slot,
            static_cast<uint64_t>(3u) * args->worker_count + args->rank);
        if (!WaitBulkGeneration(result, generation, args->spin_cap)) {
            SetError(args, slot, kFusionSlotTimeout, 561u);
            return;
        }
        if (source_partition) {
            const uint32_t owners_per_source =
                args->resources.inc_combine_aiv / args->worker_count;
            const uint32_t owner_begin = args->rank * owners_per_source;
            for (uint32_t owner = owner_begin;
                 owner < owner_begin + owners_per_source; ++owner) {
                if (!WaitBulkGeneration(BulkControl(sym, args, slot,
                        combine_barrier_base + owner),
                        generation, args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout, 5611u);
                    return;
                }
            }
        }
        WriteWorkerCheckpoint(args, 3u, 4u);
        if (result->bytes0 != result_bytes) {
            SetError(args, slot, kFusionSlotBadRoute, 562u,
                     result->bytes0);
            return;
        }
        DcciAll(bulk_result);
        CopyGmToGm(copy_tile,
            reinterpret_cast<__gm__ uint8_t *>(args->output) +
                static_cast<uint64_t>(waves[wave].token_begin) * row_bytes,
            bulk_result, result_bytes);
        __gm__ FusionSlotState *state = SlotState(args, slot);
        state->combine_generation = generation;
        state->output_generation = generation;
        state->release_generation = generation;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(state));
        PublishBulkRemote(BulkControl(sym, args, slot,
                BulkReleaseBase(args) + args->rank),
            generation, 0u, 0u, 0u, 0u,
            static_cast<int32_t>(args->inc_pe));
    }
    if (lane == 0u) WriteWorkerTrace(args, 3u, 6u, trace_start);
}

__aicore__ inline void WorkerBulkCombine(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint32_t logical_aiv)
{
    if (kEnableBulkDenseCombine && args->topk >= 2u) {
        WorkerBulkDenseCombine(sym, args, logical_aiv);
        return;
    }
    const uint32_t begin = args->resources.worker_dispatch_aiv;
    const uint32_t lanes = args->resources.worker_combine_aiv;
    if (logical_aiv < begin || logical_aiv >= begin + lanes) return;
    const uint32_t lane = logical_aiv - begin;
    const uint64_t trace_start = AscendC::GetSystemCycle();
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> copy_buf;
    pipe.InitBuffer(copy_buf, kCopyTileBytes);
    auto copy_tile = copy_buf.Get<uint8_t>();
    __gm__ FusionWaveDesc *waves =
        reinterpret_cast<__gm__ FusionWaveDesc *>(args->waves);
    const uint32_t row_bytes = args->hidden * sizeof(bfloat16_t);
    const uint64_t input_stride = BulkInputStride(args);
    const uint64_t expert_stride = BulkExpertStride(args);
    const uint64_t assignment_capacity =
        args->layout.assignment_slot_bytes / row_bytes;
    for (uint32_t wave = 0u; wave < args->wave_count; ++wave) {
        const uint32_t slot = waves[wave].slot;
        const uint64_t generation = WaveGeneration(args, wave);
        const int64_t assignment_count_i64 =
            args->local_expert_count == 0u ? 0 :
            GroupList(args, wave, slot)[args->local_expert_count - 1u];
        if (assignment_count_i64 < 0 ||
            static_cast<uint64_t>(assignment_count_i64) >
                assignment_capacity) {
            SetError(args, slot, kFusionSlotBadRoute, 521u);
            return;
        }
        if (!WaitGeneration(reinterpret_cast<__gm__ uint8_t *>(
                &SlotState(args, slot)->dispatch_generation),
                generation, args->spin_cap)) {
            SetError(args, slot, kFusionSlotTimeout, 522u);
            return;
        }
        WriteWorkerCheckpoint(args, 3u, 0u);
        const uint32_t assignment_count =
            static_cast<uint32_t>(assignment_count_i64);
        const uint32_t dense_token_stride =
            DenseWaveTokenStride(args, &waves[wave]);
        const uint64_t dense_count =
            static_cast<uint64_t>(args->worker_count) *
                dense_token_stride * args->topk;
        if (dense_count > assignment_capacity) {
            SetError(args, slot, kFusionSlotBadRoute, 526u, dense_count);
            return;
        }
        const uint64_t bytes =
            static_cast<uint64_t>(assignment_count) * row_bytes;
        __gm__ uint8_t *bulk_result = BulkArena(sym, args,
            args->symmetric_layout.reserved64[0], input_stride,
            slot, args->rank);
        const uint32_t source_tokens =
            SourceWaveTokenCount(args, &waves[wave], args->rank);
        const uint64_t result_bytes =
            static_cast<uint64_t>(source_tokens) * row_bytes;
        // INC writes the reduced result into bulk_result, so the buffer must be
        // acquired before the payload upload releases INC.  Every combine lane
        // reaches the upload join below, so acquiring here is ordered exactly
        // like a single-lane sweep while costing one instruction.
        DcciAll(bulk_result);
        __gm__ uint8_t *expert_output =
            reinterpret_cast<__gm__ uint8_t *>(args->workspace) +
                args->layout.expert_output_off +
            static_cast<uint64_t>(slot) *
                args->layout.assignment_slot_bytes;
        __gm__ uint8_t *remote = BulkArena(sym, args,
            args->symmetric_layout.reserved64[2], expert_stride,
            slot, args->rank);
        const uint64_t metadata_bytes =
            dense_count * sizeof(FusionExpertAssignment);
        const bool grouped_cube =
            args->local_expert_count > args->resources.live_aic;
        const uint32_t upload_ready_producer =
            args->ready_producers - lanes + lane;
        // Two-phase receive ownership.  Publish lengths only; INC clears its
        // exact destination ranges and returns GO before any lane puts data.
        if (lane == 0u)
            PublishBulkRemote(BulkControl(sym, args, slot,
                    BulkHandshakeBase(args) + args->rank),
                generation, assignment_count,
                static_cast<uint32_t>(dense_count), bytes, metadata_bytes,
                static_cast<int32_t>(args->inc_pe));
        if (!WaitBulkGeneration(BulkControl(sym, args, slot,
                BulkHandshakeBase(args) + args->worker_count + args->rank),
                generation, args->spin_cap)) {
            SetError(args, slot, kFusionSlotTimeout, 5201u);
            return;
        }
        if (lane != 0u) {
            if (!grouped_cube) continue;
            if (!WaitAllGenerations(
                    ReadyLine(args, slot, kReadyGmm2, 0u, 0u),
                    args->resources.live_aic, generation,
                    args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, 528u);
                return;
            }
            const uint32_t row_begin = static_cast<uint32_t>(
                static_cast<uint64_t>(assignment_count) * lane / lanes);
            const uint32_t row_end = static_cast<uint32_t>(
                static_cast<uint64_t>(assignment_count) * (lane + 1u) /
                    lanes);
            const uint64_t begin_bytes =
                static_cast<uint64_t>(row_begin) * row_bytes;
            const uint64_t lane_bytes =
                static_cast<uint64_t>(row_end - row_begin) * row_bytes;
            if (lane_bytes != 0u) {
                aclshmem_putmem(remote + begin_bytes,
                    expert_output + begin_bytes, lane_bytes,
                    static_cast<int32_t>(args->inc_pe));
                aclshmem_quiet();
            }
            PublishGeneration(ReadyLine(args, slot, kReadyDispatch, 0u,
                                        upload_ready_producer),
                              generation);
            continue;
        }
        __gm__ FusionReceivedAssignment *received =
            reinterpret_cast<__gm__ FusionReceivedAssignment *>(
                args->workspace + args->layout.assignment_meta_off) +
            static_cast<uint64_t>(slot) * assignment_capacity;
        // Dispatch RX lanes publish metadata from different AIV caches.
        // Acquire it in the Combine uploader before compacting the records,
        // especially when a request reuses a workspace slot.  One whole-cache
        // acquire replaces a sweep whose length grows with the wave's row
        // count; the records are only read after this point.
        DcciAll(reinterpret_cast<__gm__ uint8_t *>(received));
        __gm__ FusionExpertAssignment *combine_meta =
            reinterpret_cast<__gm__ FusionExpertAssignment *>(
                BulkArena(sym, args,
                    args->symmetric_layout.reserved64[2], expert_stride,
                    slot, args->rank));
        for (uint32_t row = 0u; row < assignment_count; ++row) {
            __gm__ FusionReceivedAssignment *meta = &received[row];
            if (meta->wave != wave || meta->destination_row != row ||
                meta->source_rank >= args->worker_count ||
                meta->source_token < waves[wave].token_begin ||
                meta->source_token - waves[wave].token_begin >=
                    SourceWaveTokenCount(
                        args, &waves[wave], meta->source_rank) ||
                meta->route_ordinal >= args->topk) {
                SetError(args, slot, kFusionSlotBadRoute, 525u,
                    (static_cast<uint64_t>(row) << 32u) |
                        meta->destination_row);
                return;
            }
            const uint64_t dense_index =
                (static_cast<uint64_t>(meta->source_rank) *
                     dense_token_stride +
                 meta->source_token - waves[wave].token_begin) *
                    args->topk + meta->route_ordinal;
            if (dense_index >= assignment_capacity) {
                SetError(args, slot, kFusionSlotBadRoute, 527u,
                         dense_index);
                return;
            }
            __gm__ FusionExpertAssignment *compact =
                &combine_meta[dense_index];
            // Dense [source][token-in-wave][ordinal] indexing lets every INC
            // owner find exactly K descriptors without scanning all expert
            // rows. expert/local_expert carry the full generation ticket.
            compact->dispatch_row = meta->source_rank;
            compact->expert_id = static_cast<uint32_t>(generation);
            compact->local_expert =
                static_cast<uint32_t>(generation >> 32u);
            compact->route_ordinal = meta->route_ordinal;
            compact->destination_token = meta->source_token;
            compact->weight_bits = meta->weight_bits;
            compact->wave = meta->wave;
            compact->destination_row = row;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        DcciAll(reinterpret_cast<__gm__ uint8_t *>(combine_meta));
        WriteWorkerCheckpoint(args, 3u, 1u);
        if (grouped_cube) {
            // Grouped GMM publishes every expert together, so slicing this
            // already-contiguous payload only adds RMA setup overhead and
            // cannot overlap any remaining compute.
            if (!WaitAllGenerations(
                    ReadyLine(args, slot, kReadyGmm2, 0u, 0u),
                    args->resources.live_aic, generation,
                    args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, 528u);
                return;
            }
            const uint32_t row_end = assignment_count / lanes;
            const uint64_t lane_bytes =
                static_cast<uint64_t>(row_end) * row_bytes;
            if (lane_bytes != 0u)
                aclshmem_putmem(remote, expert_output, lane_bytes,
                    static_cast<int32_t>(args->inc_pe));
            if (lane_bytes != 0u) aclshmem_quiet();
            PublishGeneration(ReadyLine(args, slot, kReadyDispatch, 0u,
                                        upload_ready_producer),
                              generation);
            for (uint32_t other = 0u; other < lanes; ++other) {
                if (!WaitGeneration(
                        ReadyLine(args, slot, kReadyDispatch, 0u,
                            args->ready_producers - lanes + other),
                        generation, args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout, 5281u);
                    return;
                }
            }
        } else {
            // Non-grouped GMM exposes expert-granular completion. Stream each
            // completed slice while later experts are still computing; the
            // destination remains the same dense expert-major prefix.
            int64_t previous_rows = 0;
            for (uint32_t expert = 0u;
                 expert < args->local_expert_count; ++expert) {
                for (uint32_t activation_wave = 0u;
                     activation_wave < args->activation_waves;
                     ++activation_wave) {
                    const uint32_t flat =
                        expert * args->activation_waves + activation_wave;
                    if (!WaitAllGenerations(
                            ReadyLine(args, slot, kReadyGmm2, flat, 0u),
                            args->resources.live_aic, generation,
                            args->spin_cap)) {
                        SetError(args, slot, kFusionSlotTimeout, 528u);
                        return;
                    }
                }
                const int64_t end_rows = GroupList(args, wave, slot)[expert];
                if (end_rows < previous_rows ||
                    end_rows > assignment_count_i64) {
                    SetError(args, slot, kFusionSlotBadRoute, 529u);
                    return;
                }
                const uint64_t begin_bytes =
                    static_cast<uint64_t>(previous_rows) * row_bytes;
                const uint64_t slice_bytes =
                    static_cast<uint64_t>(end_rows - previous_rows) *
                        row_bytes;
                if (slice_bytes != 0u)
                    aclshmem_putmem(remote + begin_bytes,
                        expert_output + begin_bytes, slice_bytes,
                        static_cast<int32_t>(args->inc_pe));
                previous_rows = end_rows;
            }
        }
        if (!grouped_cube && bytes != 0u) aclshmem_quiet();
        WriteWorkerCheckpoint(args, 3u, 2u);
        if (metadata_bytes != 0u) {
            aclshmem_putmem(remote + bytes,
                reinterpret_cast<__gm__ uint8_t *>(combine_meta),
                metadata_bytes, static_cast<int32_t>(args->inc_pe));
            aclshmem_quiet();
        }
        WriteWorkerCheckpoint(args, 3u, 3u);
        PublishBulkRemote(BulkControl(sym, args, slot,
                static_cast<uint64_t>(2u) * args->worker_count +
                    args->rank),
            generation, assignment_count,
            static_cast<uint32_t>(dense_count), bytes,
            metadata_bytes,
            static_cast<int32_t>(args->inc_pe));

        __gm__ FusionBulkControl *result = BulkControl(sym, args, slot,
            static_cast<uint64_t>(3u) * args->worker_count + args->rank);
        if (!WaitBulkGeneration(result, generation, args->spin_cap)) {
            SetError(args, slot, kFusionSlotTimeout, 523u);
            return;
        }
        WriteWorkerCheckpoint(args, 3u, 4u);
        if (result->bytes0 != result_bytes) {
            SetError(args, slot, kFusionSlotBadRoute, 524u);
            return;
        }
        // The pre-request release removed dirty old-generation lines.  This
        // post-doorbell acquisition removes any clean speculative lines
        // filled while waiting for INC put -> quiet -> doorbell.
        DcciAll(bulk_result);
        CopyGmToGm(copy_tile,
            reinterpret_cast<__gm__ uint8_t *>(args->output) +
                static_cast<uint64_t>(waves[wave].token_begin) * row_bytes,
            bulk_result, result_bytes);
        __gm__ FusionSlotState *state = SlotState(args, slot);
        state->combine_generation = generation;
        state->output_generation = generation;
        state->release_generation = generation;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(state));
        PublishBulkRemote(BulkControl(sym, args, slot,
                BulkReleaseBase(args) + args->rank),
            generation, 0u, 0u, 0u, 0u,
            static_cast<int32_t>(args->inc_pe));
    }
    if (lane == 0u) WriteWorkerTrace(args, 3u, 6u, trace_start);
}

__aicore__ inline bool IncPrepareBulkCombineReceive(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint32_t slot, uint64_t generation, uint32_t owner,
    uint64_t expert_stride, uint64_t barrier_base, uint32_t error_site)
{
    const uint32_t owners = args->resources.inc_combine_aiv;
    const uint64_t prep_generation = generation ^ (1ull << 63u);
    for (uint32_t producer = owner; producer < args->worker_count;
         producer += owners) {
        __gm__ FusionBulkControl *request = BulkControl(
            sym, args, slot, BulkHandshakeBase(args) + producer);
        if (!WaitBulkGeneration(request, generation, args->spin_cap)) {
            SetError(args, slot, kFusionSlotTimeout, error_site);
            return false;
        }
        if (request->bytes0 > expert_stride ||
            request->bytes1 > expert_stride - request->bytes0) {
            SetError(args, slot, kFusionSlotBadRoute, error_site + 1u,
                     request->bytes0 + request->bytes1);
            return false;
        }
        Dcci(BulkArena(sym, args, args->symmetric_layout.reserved64[2],
                      expert_stride, slot, producer),
             request->bytes0 + request->bytes1);
    }
    PublishBulkLocal(BulkControl(sym, args, slot, barrier_base + owner),
                     prep_generation, 0u, 0u, 0u, 0u);
    if (owner == 0u) {
        for (uint32_t other = 0u; other < owners; ++other) {
            if (!WaitBulkGeneration(
                    BulkControl(sym, args, slot, barrier_base + other),
                    prep_generation, args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, error_site + 2u);
                return false;
            }
        }
        for (uint32_t worker = 0u; worker < args->worker_count; ++worker)
            PublishBulkRemote(BulkControl(sym, args, slot,
                    BulkHandshakeBase(args) + args->worker_count + worker),
                generation, 0u, 0u, 0u, 0u, WorkerPe(args, worker));
    }
    return true;
}

// Starts the load of one producer's contribution to a dense Combine row.
// Factored out so the reduction loop can issue the next producer's load
// before consuming the current one.
__aicore__ inline void IssuePartialLoad(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args, uint32_t slot,
    uint32_t producer, uint64_t expert_stride, uint64_t element,
    const AscendC::LocalTensor<bfloat16_t> &dest,
    const AscendC::DataCopyExtParams &copy,
    const AscendC::DataCopyPadExtParams<bfloat16_t> &pad)
{
    AscendC::GlobalTensor<bfloat16_t> gm_input;
    gm_input.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
        BulkArena(sym, args, args->symmetric_layout.reserved64[2],
                  expert_stride, slot, producer)));
    AscendC::DataCopyPad(dest, gm_input[element], copy, pad);
}

__aicore__ inline void IncBulkDenseCombine(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint32_t owner)
{
    if (owner >= args->resources.inc_combine_aiv) return;
    const uint64_t trace_start = AscendC::GetSystemCycle();
    AscendC::TPipe pipe;
    // Two input banks let the load for the next producer overlap the vector
    // work of the current one; a single bank exposes a full GM round trip per
    // producer, which is the whole cost of a reduction this small.
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_bf16_buf[2];
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_fp32_buf[2];
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_fp32_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_bf16_buf;
    for (uint32_t bank = 0u; bank < 2u; ++bank) {
        pipe.InitBuffer(input_bf16_buf[bank],
                        kBulkReduceTileElements * sizeof(bfloat16_t));
        pipe.InitBuffer(input_fp32_buf[bank],
                        kBulkReduceTileElements * sizeof(float));
    }
    pipe.InitBuffer(acc_fp32_buf,
                    kBulkReduceTileElements * sizeof(float));
    pipe.InitBuffer(acc_bf16_buf,
                    kBulkReduceTileElements * sizeof(bfloat16_t));
    AscendC::LocalTensor<bfloat16_t> input_bf16[2] = {
        input_bf16_buf[0].Get<bfloat16_t>(),
        input_bf16_buf[1].Get<bfloat16_t>()};
    AscendC::LocalTensor<float> input_fp32[2] = {
        input_fp32_buf[0].Get<float>(), input_fp32_buf[1].Get<float>()};
    auto acc_fp32 = acc_fp32_buf.Get<float>();
    auto acc_bf16 = acc_bf16_buf.Get<bfloat16_t>();
    __gm__ FusionWaveDesc *waves =
        reinterpret_cast<__gm__ FusionWaveDesc *>(args->waves);
    const uint32_t row_bytes = args->hidden * sizeof(bfloat16_t);
    const uint64_t input_stride = BulkInputStride(args);
    const uint64_t expert_stride = BulkExpertStride(args);
    if (args->active_token_counts != 0u)
        Dcci(reinterpret_cast<__gm__ uint8_t *>(args->active_token_counts),
             args->worker_count * sizeof(uint32_t));
    for (uint32_t wave = 0u; wave < args->wave_count; ++wave) {
        const uint32_t slot = waves[wave].slot;
        const uint64_t generation = WaveGeneration(args, wave);
        const uint32_t dense_token_stride =
            DenseWaveTokenStride(args, &waves[wave]);
        const uint32_t dense_rows =
            args->worker_count * dense_token_stride;
        const uint64_t partial_bytes =
            static_cast<uint64_t>(dense_rows) * row_bytes;
        const uint64_t combine_barrier_base =
            static_cast<uint64_t>(4u) * args->worker_count +
            static_cast<uint64_t>(args->worker_count) * args->worker_count;
        const bool source_partition = !kSerialDenseIncDiagnostic &&
            args->resources.inc_combine_aiv >= args->worker_count &&
            args->resources.inc_combine_aiv % args->worker_count == 0u;
        const uint32_t owners_per_source = source_partition
            ? args->resources.inc_combine_aiv / args->worker_count : 0u;
        if ((RequestFlags(args) & kFusionSerializeIncDc) != 0u &&
            !WaitGeneration(reinterpret_cast<__gm__ uint8_t *>(
                    &SlotState(args, slot)->dispatch_generation),
                    generation, args->spin_cap)) {
            SetError(args, slot, kFusionSlotTimeout, 571u);
            return;
        }
        // Dense partial arenas are written only by remote workers and read
        // only by the INC.  The upload generations joined below are the
        // release edge for every producer's arena.
        for (uint32_t producer = 0u;
             producer < args->worker_count; ++producer) {
            __gm__ FusionBulkControl *ready =
                BulkControl(sym, args, slot,
                    static_cast<uint64_t>(2u) * args->worker_count +
                        producer);
            if (!WaitBulkGeneration(ready, generation, args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, 572u);
                return;
            }
            if (ready->count0 != dense_rows ||
                ready->count1 != kBulkDenseBf16Protocol ||
                ready->bytes0 != partial_bytes || ready->bytes1 != 0u ||
                partial_bytes > expert_stride) {
                SetError(args, slot, kFusionSlotBadRoute, 573u,
                         ready->bytes0);
                return;
            }
        }
        WriteIncCombineCheckpoint(args, owner, 0u);
        // All producers have published, so one whole-cache acquire covers
        // every arena this owner reduces.  Acquiring per row and producer put
        // four 64-line sweeps on the critical path for each output row.
        DcciAll(BulkArena(sym, args, args->symmetric_layout.reserved64[2],
                          expert_stride, slot, 0u));
        __gm__ uint8_t *accumulator =
            reinterpret_cast<__gm__ uint8_t *>(args->workspace) +
            args->layout.combine_ring_off +
            static_cast<uint64_t>(slot) * args->layout.output_slot_bytes;
        for (uint32_t source = 0u;
             source < args->worker_count; ++source) {
            const uint32_t tokens =
                SourceWaveTokenCount(args, &waves[wave], source);
            for (uint32_t local_token = 0u;
                 local_token < tokens; ++local_token) {
                const uint64_t output_row =
                    static_cast<uint64_t>(source) *
                        args->tokens_per_wave + local_token;
                if (kSerialDenseIncDiagnostic) {
                    if (owner != 0u) continue;
                } else if (source_partition) {
                    const uint32_t source_owner_begin =
                        source * owners_per_source;
                    if (owner < source_owner_begin ||
                        owner >= source_owner_begin + owners_per_source)
                        continue;
                    const uint32_t local_owner =
                        owner - source_owner_begin;
                    const uint32_t owner_begin = static_cast<uint32_t>(
                        static_cast<uint64_t>(tokens) * local_owner /
                            owners_per_source);
                    const uint32_t owner_end = static_cast<uint32_t>(
                        static_cast<uint64_t>(tokens) *
                            (local_owner + 1u) / owners_per_source);
                    if (local_token < owner_begin ||
                        local_token >= owner_end)
                        continue;
                } else if (output_row %
                               args->resources.inc_combine_aiv != owner) {
                    continue;
                }
                const uint64_t dense_row =
                    static_cast<uint64_t>(source) * dense_token_stride +
                    local_token;
                for (uint32_t column = 0u; column < args->hidden;
                     column += kBulkReduceTileElements) {
                    const uint32_t count =
                        args->hidden - column < kBulkReduceTileElements
                            ? args->hidden - column
                            : kBulkReduceTileElements;
                    AscendC::GlobalTensor<bfloat16_t> gm_output;
                    gm_output.SetGlobalBuffer(
                        reinterpret_cast<__gm__ bfloat16_t *>(
                            accumulator + output_row * row_bytes));
                    AscendC::DataCopyExtParams copy(
                        1u, count * sizeof(bfloat16_t), 0u, 0u, 0u);
                    AscendC::DataCopyPadExtParams<bfloat16_t> pad(
                        false, 0, 0, 0);
                    AscendC::Duplicate(acc_fp32, 0.0f, count);
                    AscendC::PipeBarrier<PIPE_V>();
                    const uint32_t producers = args->worker_count;
                    const uint64_t element =
                        dense_row * args->hidden + column;
                    IssuePartialLoad(sym, args, slot, 0u, expert_stride,
                                     element, input_bf16[0], copy, pad);
                    AscendC::SetFlag<
                        AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    for (uint32_t producer = 0u; producer < producers;
                         ++producer) {
                        const uint32_t slot_bank = producer & 1u;
                        const event_t bank = static_cast<event_t>(slot_bank);
                        const event_t next =
                            static_cast<event_t>(1u - slot_bank);
                        if (producer + 1u < producers) {
                            if (producer != 0u)
                                AscendC::WaitFlag<
                                    AscendC::HardEvent::V_MTE2>(next);
                            IssuePartialLoad(sym, args, slot, producer + 1u,
                                             expert_stride, element,
                                             input_bf16[1u - slot_bank],
                                             copy, pad);
                            AscendC::SetFlag<
                                AscendC::HardEvent::MTE2_V>(next);
                        }
                        AscendC::WaitFlag<
                            AscendC::HardEvent::MTE2_V>(bank);
                        AscendC::Cast(input_fp32[slot_bank],
                            input_bf16[slot_bank],
                            AscendC::RoundMode::CAST_NONE, count);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(acc_fp32, acc_fp32,
                                     input_fp32[slot_bank], count);
                        // Frees this bank for the load two producers ahead.
                        AscendC::SetFlag<
                            AscendC::HardEvent::V_MTE2>(bank);
                    }
                    // Retire the releases the steady-state loop never waited
                    // on.  Leaving a flag set would deadlock the next tile.
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                    if (producers >= 2u)
                        AscendC::WaitFlag<
                            AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                    AscendC::Cast(acc_bf16, acc_fp32,
                        AscendC::RoundMode::CAST_RINT, count);
                    AscendC::SetFlag<
                        AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<
                        AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::DataCopyPad(
                        gm_output[column], acc_bf16, copy);
                    AscendC::SetFlag<
                        AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    AscendC::WaitFlag<
                        AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                }
            }
        }
        // Release every row this owner reduced in one operation.  The return
        // paths below only need the accumulator visible before their put, not
        // a 64-line sweep after each row.
        AscendC::PipeBarrier<PIPE_ALL>();
        DcciAll(accumulator);
        WriteIncCombineCheckpoint(args, owner, 2u);
        if (source_partition) {
            const uint32_t source = owner / owners_per_source;
            const uint32_t local_owner = owner % owners_per_source;
            const uint32_t tokens =
                SourceWaveTokenCount(args, &waves[wave], source);
            const uint32_t token_begin = static_cast<uint32_t>(
                static_cast<uint64_t>(tokens) * local_owner /
                    owners_per_source);
            const uint32_t token_end = static_cast<uint32_t>(
                static_cast<uint64_t>(tokens) * (local_owner + 1u) /
                    owners_per_source);
            const uint64_t begin_bytes =
                static_cast<uint64_t>(token_begin) * row_bytes;
            const uint64_t bytes =
                static_cast<uint64_t>(token_end - token_begin) * row_bytes;
            __gm__ uint8_t *source_output_base = accumulator +
                static_cast<uint64_t>(source) * args->tokens_per_wave *
                    row_bytes;
            const bool global_output =
                (RequestFlags(args) & kFusionGlobalOutputFanout) != 0u;
            if (global_output) {
                // Finish only this source, not the whole wave.  Once its
                // reduction owners have published, use those same owners as
                // destination lanes: on W4 each of the four lanes sends the
                // complete source tensor to one worker.  This preserves the
                // stateless star path and starts fanout as soon as a source is
                // ready, while eliminating four serial remote-PE switches per
                // owner.
                PublishBulkLocal(BulkControl(sym, args, slot,
                        combine_barrier_base + owner),
                    generation, 0u, 0u, 0u, 0u);
                for (uint32_t other = 0u;
                     other < owners_per_source; ++other) {
                    if (!WaitBulkGeneration(BulkControl(sym, args, slot,
                            combine_barrier_base +
                                source * owners_per_source + other),
                            generation, args->spin_cap)) {
                        SetError(args, slot, kFusionSlotTimeout, 574u);
                        return;
                    }
                }
                const uint64_t result_bytes =
                    static_cast<uint64_t>(tokens) * row_bytes;
                DcciAll(source_output_base);
                for (uint32_t destination = local_owner;
                     destination < args->worker_count;
                     destination += owners_per_source) {
                    __gm__ uint8_t *remote = BulkArena(sym, args,
                        args->symmetric_layout.reserved64[0], input_stride,
                        slot, source);
                    if (result_bytes != 0u) {
                        aclshmem_putmem_nbi(remote, source_output_base,
                            result_bytes, WorkerPe(args, destination));
                        // One outstanding destination per owner.  The quiet
                        // is the release edge consumed by the result ticket.
                        aclshmem_quiet();
                    }
                }
                const uint64_t sent_barrier_base = combine_barrier_base +
                    args->resources.inc_combine_aiv;
                PublishBulkLocal(BulkControl(sym, args, slot,
                        sent_barrier_base + owner),
                    generation, 0u, 0u, 0u, 0u);
                if (local_owner == 0u) {
                    for (uint32_t other = 0u;
                         other < owners_per_source; ++other) {
                        if (!WaitBulkGeneration(BulkControl(sym, args, slot,
                                sent_barrier_base +
                                    source * owners_per_source + other),
                                generation, args->spin_cap)) {
                            SetError(args, slot, kFusionSlotTimeout, 5741u);
                            return;
                        }
                    }
                    for (uint32_t destination = 0u;
                         destination < args->worker_count; ++destination)
                        PublishBulkRemote(BulkControl(sym, args, slot,
                                static_cast<uint64_t>(3u) *
                                    args->worker_count + source),
                            generation, tokens, 0u, result_bytes, 0u,
                            WorkerPe(args, destination));
                }
                // This owner can immediately start its next wave.  The
                // source leader publishes completion after the source-local
                // sent join; persistent-lane termination is the request-wide
                // join, so an all-source barrier is intentionally avoided.
                continue;
            }

            // Local-output mode ships exactly the slice this owner reduced;
            // its whole-cache release above already covers that slice.
            __gm__ uint8_t *source_output =
                source_output_base + begin_bytes;
            {
                __gm__ uint8_t *remote = BulkArena(sym, args,
                    args->symmetric_layout.reserved64[0], input_stride,
                    slot, source) + begin_bytes;
                if (bytes != 0u) {
                    aclshmem_putmem_nbi(remote, source_output, bytes,
                        WorkerPe(args, source));
                }
            }
            aclshmem_quiet();
            PublishBulkRemote(BulkControl(sym, args, slot,
                    combine_barrier_base + owner),
                generation, 0u, 0u, bytes, 0u, WorkerPe(args, source));
            PublishBulkLocal(BulkControl(sym, args, slot,
                    combine_barrier_base + owner),
                generation, 0u, 0u, 0u, 0u);
            if (local_owner == 0u) {
                for (uint32_t other = 0u;
                     other < owners_per_source; ++other) {
                    if (!WaitBulkGeneration(BulkControl(sym, args, slot,
                            combine_barrier_base +
                                source * owners_per_source + other),
                            generation, args->spin_cap)) {
                        SetError(args, slot, kFusionSlotTimeout, 574u);
                        return;
                    }
                }
                const uint64_t result_bytes =
                    static_cast<uint64_t>(tokens) * row_bytes;
                PublishBulkRemote(BulkControl(sym, args, slot,
                        static_cast<uint64_t>(3u) *
                            args->worker_count + source),
                    generation, tokens, 0u, result_bytes, 0u,
                    WorkerPe(args, source));
            }
        } else {
            PublishBulkLocal(BulkControl(sym, args, slot,
                    combine_barrier_base + owner),
                generation, 0u, 0u, 0u, 0u);
            for (uint32_t other = 0u;
                 other < args->resources.inc_combine_aiv; ++other) {
                if (!WaitBulkGeneration(BulkControl(sym, args, slot,
                        combine_barrier_base + other),
                        generation, args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout, 575u);
                    return;
                }
            }
            for (uint32_t source = owner;
                 source < args->worker_count;
                 source += args->resources.inc_combine_aiv) {
                const uint32_t tokens =
                    SourceWaveTokenCount(args, &waves[wave], source);
                const uint64_t bytes =
                    static_cast<uint64_t>(tokens) * row_bytes;
                __gm__ uint8_t *source_output = accumulator +
                    static_cast<uint64_t>(source) * args->tokens_per_wave *
                        row_bytes;
                Dcci(source_output, bytes);
                const bool global_output =
                    (RequestFlags(args) & kFusionGlobalOutputFanout) != 0u;
                const uint32_t destination_begin =
                    global_output ? 0u : source;
                const uint32_t destination_end =
                    global_output ? args->worker_count : source + 1u;
                for (uint32_t destination = destination_begin;
                     destination < destination_end; ++destination) {
                    __gm__ uint8_t *remote = BulkArena(sym, args,
                        args->symmetric_layout.reserved64[0], input_stride,
                        slot, source);
                    if (bytes != 0u) {
                        if (global_output) {
                            aclshmem_putmem(remote, source_output, bytes,
                                WorkerPe(args, destination));
                            aclshmem_quiet();
                        } else {
                            aclshmem_putmem_nbi(remote, source_output, bytes,
                                WorkerPe(args, destination));
                        }
                    }
                }
                if (!global_output) aclshmem_quiet();
                for (uint32_t destination = destination_begin;
                     destination < destination_end; ++destination)
                    PublishBulkRemote(BulkControl(sym, args, slot,
                            static_cast<uint64_t>(3u) *
                                args->worker_count + source),
                        generation, tokens, 0u, bytes, 0u,
                        WorkerPe(args, destination));
            }
        }
        WriteIncCombineCheckpoint(args, owner, 3u);
        const uint64_t sent_barrier_base = combine_barrier_base +
            args->resources.inc_combine_aiv;
        PublishBulkLocal(BulkControl(sym, args, slot,
                sent_barrier_base + owner),
            generation, 0u, 0u, 0u, 0u);
        if (owner == 0u) {
            for (uint32_t other = 0u;
                 other < args->resources.inc_combine_aiv; ++other) {
                if (!WaitBulkGeneration(BulkControl(sym, args, slot,
                        sent_barrier_base + other),
                        generation, args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout, 576u);
                    return;
                }
            }
            if ((RequestFlags(args) & kFusionSerializeIncDc) != 0u) {
                __gm__ FusionSlotState *state = SlotState(args, slot);
                state->combine_generation = generation;
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(state));
            }
        }
        WriteIncCombineCheckpoint(args, owner, 4u);
    }
    const uint32_t trace_lane = args->resources.inc_dispatch_aiv + owner;
    __gm__ FusionLaneTrace *trace = IncTraceLine(args, trace_lane);
    trace->start_cycle = trace_start;
    trace->end_cycle = AscendC::GetSystemCycle();
    trace->role = 2u;
    trace->lane = owner;
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(trace));
}

__aicore__ inline void IncBulkCombine(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint32_t owner)
{
    if (kEnableBulkDenseCombine && args->topk >= 2u) {
        IncBulkDenseCombine(sym, args, owner);
        return;
    }
    if (owner >= args->resources.inc_combine_aiv) return;
    const uint64_t trace_start = AscendC::GetSystemCycle();
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_bf16_buf_0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_bf16_buf_0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_fp32_buf_0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_fp32_buf_0;
    pipe.InitBuffer(input_bf16_buf_0,
                    kBulkReduceTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(acc_bf16_buf_0,
                    kBulkReduceTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(input_fp32_buf_0,
                    kBulkReduceTileElements * sizeof(float));
    pipe.InitBuffer(acc_fp32_buf_0,
                    kBulkReduceTileElements * sizeof(float));
    auto input_bf16_0 = input_bf16_buf_0.Get<bfloat16_t>();
    auto acc_bf16_0 = acc_bf16_buf_0.Get<bfloat16_t>();
    auto input_fp32_0 = input_fp32_buf_0.Get<float>();
    auto acc_fp32_0 = acc_fp32_buf_0.Get<float>();
    __gm__ FusionWaveDesc *waves =
        reinterpret_cast<__gm__ FusionWaveDesc *>(args->waves);
    const uint32_t row_bytes = args->hidden * sizeof(bfloat16_t);
    const uint64_t input_stride = BulkInputStride(args);
    const uint64_t route_stride = BulkRouteStride(args);
    const uint64_t expert_stride = BulkExpertStride(args);
    if (args->active_token_counts != 0u)
        Dcci(reinterpret_cast<__gm__ uint8_t *>(args->active_token_counts),
             args->worker_count * sizeof(uint32_t));
    for (uint32_t wave = 0u; wave < args->wave_count; ++wave) {
        const uint32_t slot = waves[wave].slot;
        const uint64_t generation = WaveGeneration(args, wave);
        const uint32_t dense_token_stride =
            DenseWaveTokenStride(args, &waves[wave]);
        const uint64_t combine_barrier_base =
            static_cast<uint64_t>(4u) * args->worker_count +
            static_cast<uint64_t>(args->worker_count) * args->worker_count;
        const bool source_partition =
            args->resources.inc_combine_aiv >= args->worker_count &&
            args->resources.inc_combine_aiv % args->worker_count == 0u;
        const uint32_t owners_per_source = source_partition
            ? args->resources.inc_combine_aiv / args->worker_count : 0u;
        // Combine has the same release/acquire contract as Dispatch.  Each
        // owner waits directly on the producer doorbell below; a global INC
        // arrival barrier would only serialize the two halves of the fused
        // kernel and is not required for arena safety.
        if ((RequestFlags(args) & kFusionSerializeIncDc) != 0u &&
            !WaitGeneration(reinterpret_cast<__gm__ uint8_t *>(
                    &SlotState(args, slot)->dispatch_generation),
                    generation, args->spin_cap)) {
            SetError(args, slot, kFusionSlotTimeout, 531u);
            return;
        }
        if (!IncPrepareBulkCombineReceive(
                sym, args, slot, generation, owner, expert_stride,
                combine_barrier_base, 5301u))
            return;
        for (uint32_t producer = 0u;
             producer < args->worker_count; ++producer) {
            if (!WaitBulkGeneration(BulkControl(sym, args, slot,
                    static_cast<uint64_t>(2u) * args->worker_count +
                        producer),
                    generation, args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, 532u);
                return;
            }
        }
        WriteIncCombineCheckpoint(args, owner, 0u);
        __gm__ uint8_t *accumulator =
            reinterpret_cast<__gm__ uint8_t *>(args->workspace) +
            args->layout.combine_ring_off +
            static_cast<uint64_t>(slot) * args->layout.output_slot_bytes;
        for (uint32_t producer = 0u;
             producer < args->worker_count; ++producer) {
            __gm__ FusionBulkControl *ready =
                BulkControl(sym, args, slot,
                    static_cast<uint64_t>(2u) * args->worker_count +
                        producer);
            if (ready->bytes0 > expert_stride ||
                ready->bytes1 > expert_stride - ready->bytes0) {
                SetError(args, slot, kFusionSlotBadRoute, 541u,
                         ready->bytes0 + ready->bytes1);
                return;
            }
            Dcci(BulkArena(sym, args,
                     args->symmetric_layout.reserved64[2], expert_stride,
                     slot, producer) + ready->bytes0,
                 ready->bytes1);
        }
        WriteIncCombineCheckpoint(args, owner, 1u);
        for (uint32_t source = 0u; source < args->worker_count; ++source) {
            const uint32_t tokens =
                SourceWaveTokenCount(args, &waves[wave], source);
            for (uint32_t local_token = 0u; local_token < tokens;
                 ++local_token) {
                const uint32_t token =
                    waves[wave].token_begin + local_token;
                const uint64_t output_row =
                    static_cast<uint64_t>(source) *
                        args->tokens_per_wave + local_token;
                if (source_partition) {
                    const uint32_t source_owner_begin =
                        source * owners_per_source;
                    if (owner < source_owner_begin ||
                        owner >= source_owner_begin + owners_per_source)
                        continue;
                    const uint32_t local_owner =
                        owner - source_owner_begin;
                    const uint32_t owner_begin = static_cast<uint32_t>(
                        static_cast<uint64_t>(tokens) * local_owner /
                            owners_per_source);
                    const uint32_t owner_end = static_cast<uint32_t>(
                        static_cast<uint64_t>(tokens) *
                            (local_owner + 1u) / owners_per_source);
                    if (local_token < owner_begin || local_token >= owner_end)
                        continue;
                } else if (output_row %
                               args->resources.inc_combine_aiv != owner) {
                    continue;
                }
                Dcci(accumulator + output_row * row_bytes, row_bytes);
                for (uint32_t column = 0u; column < args->hidden;
                     column += kBulkReduceTileElements) {
                    const uint32_t count =
                        args->hidden - column < kBulkReduceTileElements
                            ? args->hidden - column
                            : kBulkReduceTileElements;
                    AscendC::Duplicate(acc_fp32_0, 0.0f, count);
                    uint32_t contributions = 0u;
                    // Producer-major iteration hoists the per-producer
                    // control, arena and dense-count validation out of the
                    // TopK loop.  Ordinal-major nesting repeats that scalar
                    // work args->topk times for the same FP32 accumulation.
                    for (uint32_t producer = 0u;
                         producer < args->worker_count; ++producer) {
                        __gm__ FusionBulkControl *ready =
                            BulkControl(sym, args, slot,
                                static_cast<uint64_t>(2u) *
                                    args->worker_count + producer);
                        if (ready->bytes0 > expert_stride ||
                            ready->bytes1 >
                                expert_stride - ready->bytes0) {
                            SetError(args, slot, kFusionSlotBadRoute, 541u,
                                     ready->bytes0 + ready->bytes1);
                            return;
                        }
                        __gm__ uint8_t *expert_output = BulkArena(
                            sym, args,
                            args->symmetric_layout.reserved64[2],
                            expert_stride, slot, producer);
                        __gm__ FusionExpertAssignment *assignments =
                            reinterpret_cast<__gm__
                                FusionExpertAssignment *>(
                                expert_output + ready->bytes0);
                        const uint64_t dense_count =
                            static_cast<uint64_t>(args->worker_count) *
                                dense_token_stride * args->topk;
                        if (ready->count1 != dense_count ||
                            ready->bytes1 != dense_count *
                                sizeof(FusionExpertAssignment)) {
                            SetError(args, slot, kFusionSlotBadRoute, 544u,
                                     ready->bytes1);
                            return;
                        }
                        const uint64_t dense_begin =
                            (static_cast<uint64_t>(source) *
                                 dense_token_stride + local_token) *
                                args->topk;
                        for (uint32_t ordinal = 0u;
                             ordinal < args->topk; ++ordinal) {
                            __gm__ FusionExpertAssignment *assignment =
                                &assignments[dense_begin + ordinal];
                            const uint64_t assignment_generation =
                                (static_cast<uint64_t>(
                                     assignment->local_expert) << 32u) |
                                assignment->expert_id;
                            if (assignment_generation != generation)
                                continue;
                            const uint32_t assignment_source =
                                assignment->dispatch_row;
                            const uint32_t assignment_token =
                                assignment->destination_token;
                            if (assignment_source >= args->worker_count ||
                                assignment->wave != wave ||
                                assignment_source != source ||
                                assignment_token != token ||
                                assignment->route_ordinal != ordinal ||
                                assignment->destination_row >=
                                    ready->count0) {
                                const uint64_t detail =
                                    (static_cast<uint64_t>(
                                         producer & 0xffu) << 56u) |
                                    (static_cast<uint64_t>(
                                         assignment_source & 0xffu) << 48u) |
                                    (static_cast<uint64_t>(
                                         ordinal & 0xffffu) << 32u) |
                                    assignment->destination_row;
                                SetError(args, slot,
                                         kFusionSlotBadRoute, 5353u,
                                         detail);
                                return;
                            }
                            if (column == 0u) {
                                Dcci(expert_output +
                                         static_cast<uint64_t>(
                                             assignment->destination_row) *
                                             row_bytes,
                                     row_bytes);
                            }
                            AscendC::GlobalTensor<bfloat16_t> gm_input;
                            gm_input.SetGlobalBuffer(
                                reinterpret_cast<__gm__ bfloat16_t *>(
                                    expert_output +
                                    static_cast<uint64_t>(
                                        assignment->destination_row) *
                                        row_bytes));
                            AscendC::DataCopyExtParams copy(
                                1u, count * sizeof(bfloat16_t),
                                0u, 0u, 0u);
                            AscendC::DataCopyPadExtParams<bfloat16_t> pad;
                            AscendC::DataCopyPad(
                                input_bf16_0, gm_input[column], copy, pad);
                            AscendC::SetFlag<
                                AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                            AscendC::WaitFlag<
                                AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                            AscendC::Cast(input_fp32_0, input_bf16_0,
                                AscendC::RoundMode::CAST_NONE, count);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Axpy(acc_fp32_0, input_fp32_0,
                                BitsFloat(assignment->weight_bits), count);
                            AscendC::PipeBarrier<PIPE_V>();
                            // input_bf16_0/input_fp32_0 are reused by the
                            // next contribution.  Fence Vector consumption
                            // before the following MTE2 overwrites the bank.
                            AscendC::SetFlag<
                                AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                            AscendC::WaitFlag<
                                AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                            ++contributions;
                        }
                    }
                    if (contributions != args->topk) {
                        SetError(args, slot, kFusionSlotBadRoute, 543u,
                            (static_cast<uint64_t>(source) << 48u) |
                            (static_cast<uint64_t>(token) << 16u) |
                            contributions);
                        return;
                    }
                    AscendC::Cast(acc_bf16_0, acc_fp32_0,
                                  AscendC::RoundMode::CAST_RINT, count);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::GlobalTensor<bfloat16_t> gm_output;
                    gm_output.SetGlobalBuffer(
                        reinterpret_cast<__gm__ bfloat16_t *>(
                            accumulator + output_row * row_bytes));
                    AscendC::DataCopyExtParams copy(
                        1u, count * sizeof(bfloat16_t), 0u, 0u, 0u);
                    AscendC::DataCopyPad(gm_output[column],
                                        acc_bf16_0, copy);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                }
            }
        }
        // Release every row this owner reduced in one operation.  The return
        // paths below only need the accumulator visible before their put, not
        // a 64-line sweep after each row.
        AscendC::PipeBarrier<PIPE_ALL>();
        DcciAll(accumulator);
        WriteIncCombineCheckpoint(args, owner, 2u);
        // When INC owners divide evenly across sources, each owner has one
        // contiguous token interval and can fan it back directly.  This
        // overlaps reduction and downlink and avoids transferring cache
        // ownership to a separate fanback AIV.  Other world sizes retain the
        // generic global-barrier path below.
        if (source_partition) {
            const uint32_t source = owner / owners_per_source;
            const uint32_t local_owner = owner % owners_per_source;
            const uint32_t tokens =
                SourceWaveTokenCount(args, &waves[wave], source);
            const uint32_t token_begin = static_cast<uint32_t>(
                static_cast<uint64_t>(tokens) * local_owner /
                    owners_per_source);
            const uint32_t token_end = static_cast<uint32_t>(
                static_cast<uint64_t>(tokens) * (local_owner + 1u) /
                    owners_per_source);
            const uint64_t begin_bytes =
                static_cast<uint64_t>(token_begin) * row_bytes;
            const uint64_t bytes =
                static_cast<uint64_t>(token_end - token_begin) * row_bytes;
            __gm__ uint8_t *source_output = accumulator +
                (static_cast<uint64_t>(source) * args->tokens_per_wave) *
                    row_bytes + begin_bytes;
            __gm__ uint8_t *remote = BulkArena(sym, args,
                args->symmetric_layout.reserved64[0], input_stride,
                slot, source) + begin_bytes;
            if (bytes != 0u)
                aclshmem_putmem_nbi(remote, source_output, bytes,
                                    WorkerPe(args, source));
            aclshmem_quiet();
            PublishBulkLocal(BulkControl(sym, args, slot,
                    combine_barrier_base + owner),
                generation, 0u, 0u, 0u, 0u);
            if (local_owner == 0u) {
                for (uint32_t other = 0u; other < owners_per_source;
                     ++other) {
                    if (!WaitBulkGeneration(BulkControl(sym, args, slot,
                            combine_barrier_base +
                                source * owners_per_source + other),
                            generation, args->spin_cap)) {
                        SetError(args, slot, kFusionSlotTimeout, 536u);
                        return;
                    }
                }
                const uint64_t result_bytes =
                    static_cast<uint64_t>(tokens) * row_bytes;
                PublishBulkRemote(BulkControl(sym, args, slot,
                        static_cast<uint64_t>(3u) * args->worker_count +
                            source),
                    generation, tokens, 0u, result_bytes, 0u,
                    WorkerPe(args, source));
            }
        } else {
            PublishBulkLocal(BulkControl(sym, args, slot,
                    combine_barrier_base + owner),
                generation, 0u, 0u, 0u, 0u);
            for (uint32_t other = 0u;
                 other < args->resources.inc_combine_aiv; ++other) {
                if (!WaitBulkGeneration(BulkControl(sym, args, slot,
                        combine_barrier_base + other),
                        generation, args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout, 536u);
                    return;
                }
            }
            for (uint32_t source = owner; source < args->worker_count;
                 source += args->resources.inc_combine_aiv) {
                const uint32_t tokens =
                    SourceWaveTokenCount(args, &waves[wave], source);
                const uint64_t bytes =
                    static_cast<uint64_t>(tokens) * row_bytes;
                __gm__ uint8_t *source_output = accumulator +
                    static_cast<uint64_t>(source) * args->tokens_per_wave *
                        row_bytes;
                Dcci(source_output, bytes);
                __gm__ uint8_t *remote = BulkArena(sym, args,
                    args->symmetric_layout.reserved64[0], input_stride,
                    slot, source);
                if (bytes != 0u)
                    aclshmem_putmem_nbi(remote, source_output, bytes,
                                        WorkerPe(args, source));
                aclshmem_quiet();
                PublishBulkRemote(BulkControl(sym, args, slot,
                        static_cast<uint64_t>(3u) * args->worker_count +
                            source),
                    generation, tokens, 0u, bytes, 0u,
                    WorkerPe(args, source));
            }
        }
        WriteIncCombineCheckpoint(args, owner, 3u);
        const uint64_t sent_barrier_base = combine_barrier_base +
            args->resources.inc_combine_aiv;
        PublishBulkLocal(BulkControl(sym, args, slot,
                sent_barrier_base + owner),
            generation, 0u, 0u, 0u, 0u);
        if (owner == 0u) {
            for (uint32_t other = 0u;
                 other < args->resources.inc_combine_aiv; ++other) {
                if (!WaitBulkGeneration(BulkControl(sym, args, slot,
                        sent_barrier_base + other),
                        generation, args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout, 537u);
                    return;
                }
            }
            if ((RequestFlags(args) & kFusionSerializeIncDc) != 0u) {
                __gm__ FusionSlotState *state = SlotState(args, slot);
                state->combine_generation = generation;
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(state));
            }
        }
        WriteIncCombineCheckpoint(args, owner, 4u);
    }
    const uint32_t trace_lane = args->resources.inc_dispatch_aiv + owner;
    __gm__ FusionLaneTrace *trace = IncTraceLine(args, trace_lane);
    trace->start_cycle = trace_start;
    trace->end_cycle = AscendC::GetSystemCycle();
    trace->role = 2u;
    trace->lane = owner;
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(trace));
}

__aicore__ inline void WorkerCombine(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint32_t logical_aiv)
{
    const uint32_t begin = args->resources.worker_dispatch_aiv;
    const uint32_t lanes = args->resources.worker_combine_aiv;
    if (logical_aiv < begin || logical_aiv >= begin + lanes) return;
    const uint32_t lane = logical_aiv - begin;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> copy_buf;
    pipe.InitBuffer(copy_buf, kCopyTileBytes);
    auto copy_tile = copy_buf.Get<uint8_t>();
    __gm__ FusionWaveDesc *waves =
        reinterpret_cast<__gm__ FusionWaveDesc *>(args->waves);
    const uint32_t row_bytes = args->hidden * sizeof(bfloat16_t);
    const uint64_t assignment_capacity =
        args->layout.assignment_slot_bytes / row_bytes;
    for (uint32_t wave = 0u; wave < args->wave_count; ++wave) {
        const uint32_t slot = waves[wave].slot;
        const uint64_t generation = WaveGeneration(args, wave);
        __gm__ FusionReceivedAssignment *received =
            reinterpret_cast<__gm__ FusionReceivedAssignment *>(
                args->workspace + args->layout.assignment_meta_off) +
            static_cast<uint64_t>(slot) * assignment_capacity;
        __gm__ uint8_t *expert_output =
            reinterpret_cast<__gm__ uint8_t *>(args->workspace) +
            args->layout.expert_output_off +
            static_cast<uint64_t>(slot) * args->layout.assignment_slot_bytes;
        uint32_t sequence[kFusionMaxAiv] = {};
        const int64_t assignment_count =
            args->local_expert_count == 0u
                ? 0 : GroupList(args, wave, slot)[
                          args->local_expert_count - 1u];
        if (assignment_count < 0 ||
            static_cast<uint64_t>(assignment_count) > assignment_capacity) {
            SetError(args, slot, kFusionSlotBadRoute, 121u);
            return;
        }

        // The production path streams each completed expert/slice directly
        // into Combine.  The serial_inc attribution baseline instead waits
        // for this worker's complete FFN wave.  Rank workloads in the formal
        // benchmark use the same balanced route, so this conservative local
        // gate also removes meaningful cross-rank D/FFN/C overlap without
        // introducing different kernels, layouts or setup costs.
        if ((RequestFlags(args) & kFusionStrictSerialPipeline) != 0u) {
            for (uint32_t expert = 0u;
                 expert < args->local_expert_count; ++expert) {
                for (uint32_t activation_wave = 0u;
                     activation_wave < args->activation_waves;
                     ++activation_wave) {
                    const uint32_t flat =
                        expert * args->activation_waves + activation_wave;
                    if (!WaitAllGenerations(
                            ReadyLine(args, slot, kReadyGmm2, flat, 0u),
                            args->resources.live_aic, generation,
                            args->spin_cap)) {
                        SetError(args, slot, kFusionSlotTimeout, 126u);
                        return;
                    }
                }
            }
        }

        // Consume GMM2 in exactly the same expert/slice order in which AIC
        // publishes it.  This removes the old whole-wave compute->Combine
        // barrier: as soon as one slice is complete its rows enter the INC
        // reduction stream while later slices are still running.  The row
        // order remains monotonically increasing, so every owner stream keeps
        // the same deterministic packet order and credit requirements.
        //
        // The receive metadata was published by the Dispatch RX cohort before
        // the FFN even started, so one acquire ahead of the scan is enough;
        // re-acquiring each record inside the row loop cost one cache
        // operation per row per lane.
        DcciAll(reinterpret_cast<__gm__ uint8_t *>(received));
        int64_t previous = 0;
        for (uint32_t expert = 0u; expert < args->local_expert_count;
             ++expert) {
            const int64_t end = GroupList(args, wave, slot)[expert];
            if (end < previous || end > assignment_count) {
                SetError(args, slot, kFusionSlotBadRoute, 125u);
                return;
            }
            const uint32_t rows = static_cast<uint32_t>(end - previous);
            for (uint32_t activation_wave = 0u;
                 activation_wave < args->activation_waves;
                 ++activation_wave) {
                const uint32_t flat =
                    expert * args->activation_waves + activation_wave;
                if (!WaitAllGenerations(
                        ReadyLine(args, slot, kReadyGmm2, flat, 0u),
                        args->resources.live_aic, generation,
                        args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout, 120u);
                    return;
                }
                const uint32_t begin = static_cast<uint32_t>(
                    static_cast<uint64_t>(rows) * activation_wave /
                    args->activation_waves);
                const uint32_t finish = static_cast<uint32_t>(
                    static_cast<uint64_t>(rows) * (activation_wave + 1u) /
                    args->activation_waves);
                for (uint32_t row = static_cast<uint32_t>(previous) + begin;
                     row < static_cast<uint32_t>(previous) + finish; ++row) {
                    __gm__ FusionReceivedAssignment *assignment =
                        &received[row];
                    const uint32_t owner = assignment->source_token %
                                           args->resources.inc_combine_aiv;
                    if (owner % lanes != lane) continue;
                    uint32_t offset = 0u;
                    while (offset < row_bytes) {
                        const uint32_t bytes = row_bytes - offset <
                                args->transport_tile_bytes
                            ? row_bytes - offset :
                                args->transport_tile_bytes;
                        FusionPacketHeader packet{};
                        packet.wave = wave;
                        packet.source_rank = assignment->source_rank;
                        packet.source_token = assignment->source_token;
                        packet.expert_id = assignment->expert_id;
                        packet.route_ordinal = assignment->route_ordinal;
                        packet.weight_bits = assignment->weight_bits;
                        packet.payload_offset = offset;
                        packet.kind = kFusionCombinePayload;
                        if (!PublishPacket(sym, args,
                                args->symmetric_layout.combine_header_off,
                                args->symmetric_layout.combine_payload_off,
                                slot, args->rank, owner, sequence[owner]++,
                                expert_output + static_cast<uint64_t>(row) *
                                    row_bytes + offset,
                                bytes, packet,
                                static_cast<int32_t>(args->inc_pe))) {
                            SetError(args, slot, kFusionSlotTimeout, 122u);
                            return;
                        }
                        offset += bytes;
                    }
                }
            }
            previous = end;
        }
        for (uint32_t owner = lane;
             owner < args->resources.inc_combine_aiv; owner += lanes) {
            FusionPacketHeader end{};
            end.wave = wave;
            end.kind = kFusionCombineLaneEnd;
            if (!PublishPacket(sym, args,
                    args->symmetric_layout.combine_header_off,
                    args->symmetric_layout.combine_payload_off,
                    slot, args->rank, owner, sequence[owner],
                    nullptr, 0u, end, static_cast<int32_t>(args->inc_pe))) {
                SetError(args, slot, kFusionSlotTimeout);
                return;
            }
        }
        // The same cohort receives owner-sharded INC results after completing
        // its producer streams. INC never sends a result before every worker
        // has closed that owner stream, so this ordering is deadlock-free.
        for (uint32_t owner = lane;
             owner < args->resources.inc_combine_aiv; owner += lanes) {
            uint32_t receive_sequence = 0u;
            while (true) {
                __gm__ FusionPacketHeader *header = WaitPacket(
                    sym, args,
                    args->symmetric_layout.combine_result_header_off,
                    slot, args->rank, owner, receive_sequence, generation);
                if (header == nullptr) {
                    SetError(args, slot, kFusionSlotTimeout);
                    return;
                }
                const uint32_t kind = header->kind;
                if (kind == kFusionResultPayload) {
                    if (header->source_token < waves[wave].token_begin ||
                        header->source_token >= waves[wave].token_begin +
                            waves[wave].token_count ||
                        header->payload_offset + header->payload_bytes >
                            row_bytes) {
                        SetError(args, slot, kFusionSlotBadRoute, 123u);
                        return;
                    }
                    __gm__ uint8_t *payload = PacketPayload(
                        sym, args,
                        args->symmetric_layout.combine_result_payload_off,
                        slot, args->rank, owner, receive_sequence);
                    Dcci(payload, header->payload_bytes);
                    CopyGmToGm(copy_tile,
                        reinterpret_cast<__gm__ uint8_t *>(args->output) +
                            static_cast<uint64_t>(header->source_token) *
                                row_bytes + header->payload_offset,
                        payload, header->payload_bytes);
                }
                Release(header, receive_sequence,
                        static_cast<int32_t>(args->inc_pe));
                ++receive_sequence;
                if (kind == kFusionCombineLaneEnd) break;
            }
        }
        PublishGeneration(ReadyLine(args, slot, kReadyGmm2, 0u,
                                    args->resources.live_aic + lane),
                          generation);
        if (lane == 0u) {
            for (uint32_t other = 0u; other < lanes; ++other) {
                if (!WaitGeneration(
                        ReadyLine(args, slot, kReadyGmm2, 0u,
                                  args->resources.live_aic + other),
                        generation, args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout, 124u);
                    return;
                }
            }
            __gm__ FusionSlotState *state = SlotState(args, slot);
            state->combine_generation = generation;
            state->output_generation = generation;
            state->release_generation = generation;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(state));
        }
    }
}

__aicore__ inline void WeightedAdd(
    AscendC::LocalTensor<bfloat16_t> &input_bf16_0,
    AscendC::LocalTensor<bfloat16_t> &input_bf16_1,
    AscendC::LocalTensor<bfloat16_t> &acc_bf16_0,
    AscendC::LocalTensor<bfloat16_t> &acc_bf16_1,
    AscendC::LocalTensor<float> &input_fp32_0,
    AscendC::LocalTensor<float> &input_fp32_1,
    AscendC::LocalTensor<float> &acc_fp32_0,
    AscendC::LocalTensor<float> &acc_fp32_1,
    __gm__ uint8_t *accumulator, __gm__ uint8_t *payload,
    uint32_t bytes, float weight)
{
    AscendC::GlobalTensor<bfloat16_t> gm_input;
    AscendC::GlobalTensor<bfloat16_t> gm_acc;
    gm_input.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(payload));
    gm_acc.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(accumulator));
    const uint32_t elements = bytes / sizeof(bfloat16_t);
    // Preserve the minimal one-tile path used by small/medium rows.  Starting
    // two ping-pong lifetime chains has a measurable fixed cost when there is
    // nothing to overlap; the double-buffer path below is only useful once a
    // packet spans multiple vector tiles.
    if (elements <= kReduceTileElements) {
        AscendC::DataCopyExtParams copy(
            1u, elements * sizeof(bfloat16_t), 0u, 0u, 0u);
        AscendC::DataCopyPadExtParams<bfloat16_t> pad;
        AscendC::DataCopyPad(input_bf16_0, gm_input, copy, pad);
        AscendC::DataCopyPad(acc_bf16_0, gm_acc, copy, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(input_fp32_0, input_bf16_0,
                      AscendC::RoundMode::CAST_NONE, elements);
        AscendC::Cast(acc_fp32_0, acc_bf16_0,
                      AscendC::RoundMode::CAST_NONE, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(input_fp32_0, input_fp32_0, weight, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(acc_fp32_0, acc_fp32_0, input_fp32_0, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(acc_bf16_0, acc_fp32_0,
                      AscendC::RoundMode::CAST_RINT, elements);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(gm_acc, acc_bf16_0, copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        return;
    }
    // MTE3_MTE2 is the lifetime fence for each ping-pong UB bank.  Unlike the
    // former PIPE_ALL after every tile, this permits MTE3(tile n),
    // MTE2(tile n+1) and vector work to occupy independent pipelines while
    // still forbidding a bank from being overwritten before its GM store.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    uint32_t offset = 0u;
    uint32_t ping = 0u;
    while (offset < elements) {
        const uint32_t count = elements - offset < kReduceTileElements
            ? elements - offset : kReduceTileElements;
        const AscendC::TEventID event = ping == 0u ? EVENT_ID0 : EVENT_ID1;
        auto input_bf16 = ping == 0u ? input_bf16_0 : input_bf16_1;
        auto acc_bf16 = ping == 0u ? acc_bf16_0 : acc_bf16_1;
        auto input_fp32 = ping == 0u ? input_fp32_0 : input_fp32_1;
        auto acc_fp32 = ping == 0u ? acc_fp32_0 : acc_fp32_1;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        AscendC::DataCopyExtParams copy(
            1u, count * sizeof(bfloat16_t), 0u, 0u, 0u);
        AscendC::DataCopyPadExtParams<bfloat16_t> pad;
        AscendC::DataCopyPad(input_bf16, gm_input[offset], copy, pad);
        AscendC::DataCopyPad(acc_bf16, gm_acc[offset], copy, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(input_fp32, input_bf16,
                      AscendC::RoundMode::CAST_NONE, count);
        AscendC::Cast(acc_fp32, acc_bf16,
                      AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(input_fp32, input_fp32, weight, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(acc_fp32, acc_fp32, input_fp32, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(acc_bf16, acc_fp32,
                      AscendC::RoundMode::CAST_RINT, count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        AscendC::DataCopyPad(gm_acc[offset], acc_bf16, copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        offset += count;
        ping = 1u - ping;
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// Direct combine splits the worker-combine cohort in half. Producers PUT
// expert rows to the original token owner's symmetric queue; reducer lanes on
// that owner consume every producer queue and accumulate directly into output.
__aicore__ inline void WorkerDirectCombine(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint32_t logical_aiv)
{
    const uint32_t begin = args->resources.worker_dispatch_aiv;
    const uint32_t total = args->resources.worker_combine_aiv;
    if (logical_aiv < begin || logical_aiv >= begin + total) return;
    const uint32_t local_lane = logical_aiv - begin;
    const uint32_t tx_lanes = total > 1u ? total / 2u : 1u;
    const uint32_t reduce_lanes = total - tx_lanes;
    if (reduce_lanes == 0u) {
        SetError(args, 0u, kFusionSlotBadRoute, 420u);
        return;
    }
    __gm__ FusionWaveDesc *waves =
        reinterpret_cast<__gm__ FusionWaveDesc *>(args->waves);
    const uint32_t row_bytes = args->hidden * sizeof(bfloat16_t);
    const uint64_t assignment_capacity =
        args->layout.assignment_slot_bytes / row_bytes;

    if (local_lane < tx_lanes) {
        const uint32_t tx = local_lane;
        for (uint32_t wave = 0u; wave < args->wave_count; ++wave) {
            const uint32_t slot = waves[wave].slot;
            const uint64_t generation = WaveGeneration(args, wave);
            __gm__ FusionReceivedAssignment *received =
                reinterpret_cast<__gm__ FusionReceivedAssignment *>(
                    args->workspace + args->layout.assignment_meta_off) +
                static_cast<uint64_t>(slot) * assignment_capacity;
            __gm__ uint8_t *expert_output =
                reinterpret_cast<__gm__ uint8_t *>(args->workspace) +
                args->layout.expert_output_off +
                static_cast<uint64_t>(slot) *
                    args->layout.assignment_slot_bytes;
            const int64_t assignment_count =
                args->local_expert_count == 0u ? 0 :
                GroupList(args, wave, slot)[args->local_expert_count - 1u];
            if (assignment_count < 0 ||
                static_cast<uint64_t>(assignment_count) >
                    assignment_capacity) {
                SetError(args, slot, kFusionSlotBadRoute, 421u);
                return;
            }
            if ((RequestFlags(args) & kFusionStrictSerialPipeline) != 0u) {
                for (uint32_t expert = 0u;
                     expert < args->local_expert_count; ++expert) {
                    for (uint32_t aw = 0u; aw < args->activation_waves; ++aw) {
                        const uint32_t flat = expert * args->activation_waves + aw;
                        if (!WaitAllGenerations(
                                ReadyLine(args, slot, kReadyGmm2, flat, 0u),
                                args->resources.live_aic, generation,
                                args->spin_cap)) {
                            SetError(args, slot, kFusionSlotTimeout, 422u);
                            return;
                        }
                    }
                }
            }
            // One tx lane owns each reduction owner, so sequence counters are
            // only per destination worker and never contend with another AIV.
            // The per-owner scans below re-read the same records, so acquire
            // them once outside the owner loop instead of once per row per
            // owner.
            DcciAll(reinterpret_cast<__gm__ uint8_t *>(received));
            for (uint32_t owner = tx; owner < reduce_lanes;
                 owner += tx_lanes) {
                uint32_t sequence[kFusionMaxWorkers] = {};
                int64_t previous = 0;
                for (uint32_t expert = 0u;
                     expert < args->local_expert_count; ++expert) {
                    const int64_t end = GroupList(args, wave, slot)[expert];
                    if (end < previous || end > assignment_count) {
                        SetError(args, slot, kFusionSlotBadRoute, 423u);
                        return;
                    }
                    const uint32_t rows = static_cast<uint32_t>(end - previous);
                    for (uint32_t aw = 0u; aw < args->activation_waves; ++aw) {
                        const uint32_t flat = expert * args->activation_waves + aw;
                        if (!WaitAllGenerations(
                                ReadyLine(args, slot, kReadyGmm2, flat, 0u),
                                args->resources.live_aic, generation,
                                args->spin_cap)) {
                            SetError(args, slot, kFusionSlotTimeout, 424u);
                            return;
                        }
                        const uint32_t first = static_cast<uint32_t>(
                            static_cast<uint64_t>(rows) * aw /
                            args->activation_waves);
                        const uint32_t last = static_cast<uint32_t>(
                            static_cast<uint64_t>(rows) * (aw + 1u) /
                            args->activation_waves);
                        for (uint32_t row =
                                 static_cast<uint32_t>(previous) + first;
                             row < static_cast<uint32_t>(previous) + last;
                             ++row) {
                            __gm__ FusionReceivedAssignment *assignment =
                                &received[row];
                            if (assignment->source_rank >= args->worker_count ||
                                assignment->source_token % reduce_lanes != owner)
                                continue;
                            const uint32_t destination = assignment->source_rank;
                            uint32_t offset = 0u;
                            while (offset < row_bytes) {
                                const uint32_t bytes = row_bytes - offset <
                                        args->transport_tile_bytes
                                    ? row_bytes - offset :
                                      args->transport_tile_bytes;
                                FusionPacketHeader packet{};
                                packet.wave = wave;
                                packet.source_rank = destination;
                                packet.destination_rank = args->rank;
                                packet.source_token = assignment->source_token;
                                packet.expert_id = assignment->expert_id;
                                packet.route_ordinal =
                                    assignment->route_ordinal;
                                packet.weight_bits = assignment->weight_bits;
                                packet.payload_offset = offset;
                                packet.kind = kFusionCombinePayload;
                                if (!PublishPacket(sym, args,
                                        args->symmetric_layout.combine_header_off,
                                        args->symmetric_layout.combine_payload_off,
                                        slot, destination,
                                        args->rank * reduce_lanes + owner,
                                        sequence[destination]++,
                                        expert_output +
                                            static_cast<uint64_t>(row) *
                                                row_bytes + offset,
                                        bytes, packet,
                                        WorkerPe(args, destination))) {
                                    SetError(args, slot,
                                             kFusionSlotTimeout, 425u);
                                    return;
                                }
                                offset += bytes;
                            }
                        }
                    }
                    previous = end;
                }
                for (uint32_t destination = 0u;
                     destination < args->worker_count; ++destination) {
                    FusionPacketHeader end{};
                    end.wave = wave;
                    end.source_rank = destination;
                    end.destination_rank = args->rank;
                    end.kind = kFusionCombineLaneEnd;
                    if (!PublishPacket(sym, args,
                            args->symmetric_layout.combine_header_off,
                            args->symmetric_layout.combine_payload_off,
                            slot, destination,
                            args->rank * reduce_lanes + owner,
                            sequence[destination], nullptr, 0u, end,
                            WorkerPe(args, destination))) {
                        SetError(args, slot, kFusionSlotTimeout, 426u);
                        return;
                    }
                }
            }
        }
        return;
    }

    const uint32_t reducer = local_lane - tx_lanes;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_bf16_buf_0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_bf16_buf_1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_bf16_buf_0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_bf16_buf_1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_fp32_buf_0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_fp32_buf_1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_fp32_buf_0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_fp32_buf_1;
    pipe.InitBuffer(input_bf16_buf_0,
                    kReduceTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(input_bf16_buf_1,
                    kReduceTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(acc_bf16_buf_0,
                    kReduceTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(acc_bf16_buf_1,
                    kReduceTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(input_fp32_buf_0, kReduceTileElements * sizeof(float));
    pipe.InitBuffer(input_fp32_buf_1, kReduceTileElements * sizeof(float));
    pipe.InitBuffer(acc_fp32_buf_0, kReduceTileElements * sizeof(float));
    pipe.InitBuffer(acc_fp32_buf_1, kReduceTileElements * sizeof(float));
    auto input_bf16_0 = input_bf16_buf_0.Get<bfloat16_t>();
    auto input_bf16_1 = input_bf16_buf_1.Get<bfloat16_t>();
    auto acc_bf16_0 = acc_bf16_buf_0.Get<bfloat16_t>();
    auto acc_bf16_1 = acc_bf16_buf_1.Get<bfloat16_t>();
    auto input_fp32_0 = input_fp32_buf_0.Get<float>();
    auto input_fp32_1 = input_fp32_buf_1.Get<float>();
    auto acc_fp32_0 = acc_fp32_buf_0.Get<float>();
    auto acc_fp32_1 = acc_fp32_buf_1.Get<float>();
    AscendC::GlobalTensor<bfloat16_t> output_gm;
    output_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t *>(args->output));

    for (uint32_t wave = 0u; wave < args->wave_count; ++wave) {
        const uint32_t slot = waves[wave].slot;
        const uint64_t generation = WaveGeneration(args, wave);
        const uint32_t own_tokens =
            SourceWaveTokenCount(args, &waves[wave], args->rank);
        for (uint32_t owner = reducer; owner < reduce_lanes;
             owner += reduce_lanes) {
            for (uint32_t local_token = 0u; local_token < own_tokens;
                 ++local_token) {
                const uint32_t token = waves[wave].token_begin + local_token;
                if (token % reduce_lanes != owner) continue;
                uint64_t element = static_cast<uint64_t>(token) * args->hidden;
                uint32_t remaining = args->hidden;
                while (remaining != 0u) {
                    const uint32_t count = remaining < kReduceTileElements
                        ? remaining : kReduceTileElements;
                    AscendC::Duplicate(acc_bf16_0,
                        static_cast<bfloat16_t>(0.0f), count);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::DataCopyExtParams copy(
                        1u, count * sizeof(bfloat16_t), 0u, 0u, 0u);
                    AscendC::DataCopyPad(output_gm[element], acc_bf16_0, copy);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    element += count;
                    remaining -= count;
                }
            }
            uint32_t producer_sequence[kFusionMaxWorkers] = {};
            bool producer_ended[kFusionMaxWorkers] = {};
            uint32_t remaining_producers = args->worker_count;
            uint32_t idle_spins = 0u;
            while (remaining_producers != 0u) {
                bool progressed = false;
                for (uint32_t producer = 0u;
                     producer < args->worker_count; ++producer) {
                    if (producer_ended[producer]) continue;
                    const uint32_t sequence = producer_sequence[producer];
                    __gm__ FusionPacketHeader *header = TryPacket(
                        sym, args,
                        args->symmetric_layout.combine_header_off,
                        slot, args->rank,
                        producer * reduce_lanes + owner,
                        sequence, generation);
                    if (header == nullptr) continue;
                    progressed = true;
                    const uint32_t kind = header->kind;
                    if (kind == kFusionCombinePayload) {
                        uint32_t bad_site = 0u;
                        if (header->source_rank != args->rank)
                            bad_site = 4271u;
                        else if (header->source_token <
                                 waves[wave].token_begin)
                            bad_site = 4272u;
                        else if (header->source_token >=
                                 waves[wave].token_begin + own_tokens)
                            bad_site = 4273u;
                        else if (header->source_token % reduce_lanes != owner)
                            bad_site = 4274u;
                        else if (header->payload_offset +
                                     header->payload_bytes > row_bytes)
                            bad_site = 4275u;
                        if (bad_site != 0u) {
                            const uint64_t detail =
                                (static_cast<uint64_t>(header->source_rank) << 48u) |
                                (static_cast<uint64_t>(header->source_token) << 16u) |
                                owner;
                            SetError(args, slot, kFusionSlotBadRoute, bad_site,
                                     detail);
                            return;
                        }
                        __gm__ uint8_t *payload = PacketPayload(
                            sym, args,
                            args->symmetric_layout.combine_payload_off,
                            slot, args->rank,
                            producer * reduce_lanes + owner,
                            sequence);
                        Dcci(payload, header->payload_bytes);
                        WeightedAdd(input_bf16_0, input_bf16_1,
                                    acc_bf16_0, acc_bf16_1,
                                    input_fp32_0, input_fp32_1,
                                    acc_fp32_0, acc_fp32_1,
                                    reinterpret_cast<__gm__ uint8_t *>(
                                        args->output) +
                                        static_cast<uint64_t>(
                                            header->source_token) * row_bytes +
                                        header->payload_offset,
                                    payload, header->payload_bytes,
                                    BitsFloat(header->weight_bits));
                    }
                    Release(header, sequence, WorkerPe(args, producer));
                    producer_sequence[producer] = sequence + 1u;
                    if (kind == kFusionCombineLaneEnd) {
                        producer_ended[producer] = true;
                        --remaining_producers;
                    }
                }
                if (progressed) idle_spins = 0u;
                else if (args->spin_cap != 0u &&
                         ++idle_spins >= args->spin_cap) {
                    SetError(args, slot, kFusionSlotTimeout, 428u);
                    return;
                }
            }
        }
        PublishGeneration(
            ReadyLine(args, slot, kReadyGmm2, 0u,
                      args->resources.live_aic + reducer),
            generation);
        if (reducer == 0u) {
            for (uint32_t other = 0u; other < reduce_lanes; ++other) {
                if (!WaitGeneration(
                        ReadyLine(args, slot, kReadyGmm2, 0u,
                                  args->resources.live_aic + other),
                        generation, args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout, 429u);
                    return;
                }
            }
            __gm__ FusionSlotState *state = SlotState(args, slot);
            state->combine_generation = generation;
            state->output_generation = generation;
            state->release_generation = generation;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(state));
        }
    }
}

__aicore__ inline void IncCombine(
    __gm__ uint8_t *sym, __gm__ const FusionKernelArgs *args,
    uint32_t owner)
{
    if (owner >= args->resources.inc_combine_aiv) return;
    const uint64_t trace_start = AscendC::GetSystemCycle();
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_bf16_buf_0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_bf16_buf_1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_bf16_buf_0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_bf16_buf_1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_fp32_buf_0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_fp32_buf_1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_fp32_buf_0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_fp32_buf_1;
    pipe.InitBuffer(input_bf16_buf_0,
                    kReduceTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(input_bf16_buf_1,
                    kReduceTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(acc_bf16_buf_0,
                    kReduceTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(acc_bf16_buf_1,
                    kReduceTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(input_fp32_buf_0, kReduceTileElements * sizeof(float));
    pipe.InitBuffer(input_fp32_buf_1, kReduceTileElements * sizeof(float));
    pipe.InitBuffer(acc_fp32_buf_0, kReduceTileElements * sizeof(float));
    pipe.InitBuffer(acc_fp32_buf_1, kReduceTileElements * sizeof(float));
    auto input_bf16_0 = input_bf16_buf_0.Get<bfloat16_t>();
    auto input_bf16_1 = input_bf16_buf_1.Get<bfloat16_t>();
    auto acc_bf16_0 = acc_bf16_buf_0.Get<bfloat16_t>();
    auto acc_bf16_1 = acc_bf16_buf_1.Get<bfloat16_t>();
    auto input_fp32_0 = input_fp32_buf_0.Get<float>();
    auto input_fp32_1 = input_fp32_buf_1.Get<float>();
    auto acc_fp32_0 = acc_fp32_buf_0.Get<float>();
    auto acc_fp32_1 = acc_fp32_buf_1.Get<float>();
    __gm__ FusionWaveDesc *waves =
        reinterpret_cast<__gm__ FusionWaveDesc *>(args->waves);
    if (args->active_token_counts != 0u)
        Dcci(reinterpret_cast<__gm__ uint8_t *>(
                 args->active_token_counts),
             args->worker_count * sizeof(uint32_t));
    const uint32_t row_bytes = args->hidden * sizeof(bfloat16_t);
    for (uint32_t wave = 0u; wave < args->wave_count; ++wave) {
        const uint32_t slot = waves[wave].slot;
        const uint64_t generation = WaveGeneration(args, wave);
        if ((RequestFlags(args) & kFusionSerializeIncDc) != 0u) {
            __gm__ uint8_t *done = reinterpret_cast<__gm__ uint8_t *>(
                &SlotState(args, slot)->dispatch_generation);
            if (!WaitGeneration(done, generation, args->spin_cap)) {
                SetError(args, slot, kFusionSlotTimeout, 205u);
                return;
            }
        }
        __gm__ uint8_t *accumulator =
            reinterpret_cast<__gm__ uint8_t *>(args->workspace) +
            args->layout.combine_ring_off +
            static_cast<uint64_t>(slot) * args->layout.output_slot_bytes;
        // Each owner AIV exclusively owns token%inc_combine_aiv, so zero and
        // reductions need no inter-AIV atomic or barrier.
        AscendC::GlobalTensor<bfloat16_t> zero_gm;
        zero_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t *>(accumulator));
        for (uint32_t source = 0u; source < args->worker_count; ++source) {
            const uint32_t source_wave_tokens =
                SourceWaveTokenCount(args, &waves[wave], source);
            for (uint32_t local_token = 0u;
                 local_token < source_wave_tokens; ++local_token) {
                if ((waves[wave].token_begin + local_token) %
                        args->resources.inc_combine_aiv != owner)
                    continue;
                uint64_t row = static_cast<uint64_t>(source) *
                                   args->tokens_per_wave + local_token;
                uint64_t element = row * args->hidden;
                uint32_t remaining = args->hidden;
                while (remaining != 0u) {
                    const uint32_t count = remaining < kReduceTileElements
                        ? remaining : kReduceTileElements;
                    AscendC::Duplicate(acc_bf16_0,
                        static_cast<bfloat16_t>(0.0f), count);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::DataCopyExtParams copy(
                        1u, count * sizeof(bfloat16_t), 0u, 0u, 0u);
                    AscendC::DataCopyPad(zero_gm[element], acc_bf16_0, copy);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    element += count;
                    remaining -= count;
                }
            }
        }
        uint32_t producer_sequence[kFusionMaxWorkers] = {};
        bool producer_ended[kFusionMaxWorkers] = {};
        uint32_t producers_remaining = args->worker_count;
        uint32_t idle_spins = 0u;
        // Do not block on producer rank order. Worker AIVs multiplex several
        // owner streams, so a rank-major blocking receive can form a cycle
        // across fixed-depth queues. Round-robin consumes whichever producer
        // has made progress and is independent of route/arrival order.
        while (producers_remaining != 0u) {
            bool progressed = false;
            for (uint32_t producer = 0u;
                 producer < args->worker_count; ++producer) {
                if (producer_ended[producer]) continue;
                const uint32_t sequence = producer_sequence[producer];
                __gm__ FusionPacketHeader *header = TryPacket(
                    sym, args, args->symmetric_layout.combine_header_off,
                    slot, producer, owner, sequence, generation);
                if (header == nullptr) continue;
                progressed = true;
                const uint32_t kind = header->kind;
                if (kind == kFusionCombinePayload) {
                    uint32_t bad_site = 0u;
                    if (header->source_rank >= args->worker_count)
                        bad_site = 2021u;
                    else if (header->source_token < waves[wave].token_begin)
                        bad_site = 2022u;
                    else if (header->source_token >=
                             waves[wave].token_begin +
                                 SourceWaveTokenCount(
                                     args, &waves[wave],
                                     header->source_rank))
                        bad_site = 2023u;
                    else if (header->payload_offset > row_bytes ||
                             header->payload_bytes >
                                 row_bytes - header->payload_offset)
                        bad_site = 2024u;
                    else if (header->source_token %
                                 args->resources.inc_combine_aiv != owner)
                        bad_site = 2025u;
                    if (bad_site != 0u) {
                        const uint64_t detail =
                            (static_cast<uint64_t>(wave) << 56u) |
                            (static_cast<uint64_t>(producer) << 48u) |
                            (static_cast<uint64_t>(owner) << 40u) |
                            (static_cast<uint64_t>(sequence & 0xffu) << 32u) |
                            header->source_token;
                        SetError(args, slot, kFusionSlotBadRoute, bad_site,
                                 detail);
                        return;
                    }
                    const uint64_t row =
                        static_cast<uint64_t>(header->source_rank) *
                            args->tokens_per_wave +
                        header->source_token - waves[wave].token_begin;
                    __gm__ uint8_t *payload = PacketPayload(
                        sym, args,
                        args->symmetric_layout.combine_payload_off,
                        slot, producer, owner, sequence);
                    Dcci(payload, header->payload_bytes);
                    WeightedAdd(input_bf16_0, input_bf16_1,
                                acc_bf16_0, acc_bf16_1,
                                input_fp32_0, input_fp32_1,
                                acc_fp32_0, acc_fp32_1,
                                accumulator + row * row_bytes +
                                    header->payload_offset,
                                payload, header->payload_bytes,
                                BitsFloat(header->weight_bits));
                }
                Release(header, sequence, WorkerPe(args, producer));
                producer_sequence[producer] = sequence + 1u;
                if (kind == kFusionCombineLaneEnd) {
                    producer_ended[producer] = true;
                    --producers_remaining;
                }
            }
            if (progressed) {
                idle_spins = 0u;
            } else {
                ++idle_spins;
                if (args->spin_cap != 0u &&
                    idle_spins >= args->spin_cap) {
                    SetError(args, slot, kFusionSlotTimeout, 201u);
                    return;
                }
            }
        }
        for (uint32_t source = 0u; source < args->worker_count; ++source) {
            const uint32_t source_wave_tokens =
                SourceWaveTokenCount(args, &waves[wave], source);
            uint32_t sequence = 0u;
            for (uint32_t local_token = 0u;
                 local_token < source_wave_tokens; ++local_token) {
                if ((waves[wave].token_begin + local_token) %
                        args->resources.inc_combine_aiv != owner)
                    continue;
                const uint32_t source_token =
                    waves[wave].token_begin + local_token;
                const uint64_t row = static_cast<uint64_t>(source) *
                                         args->tokens_per_wave + local_token;
                // WeightedAdd stores through MTE3. Make this owner-private
                // row visible to the SHMEM RMA engine without invalidating
                // caches used by another owner's in-flight transfer.
                Dcci(accumulator + row * row_bytes, row_bytes);
                uint32_t offset = 0u;
                while (offset < row_bytes) {
                    const uint32_t bytes = row_bytes - offset <
                            args->transport_tile_bytes
                        ? row_bytes - offset : args->transport_tile_bytes;
                    FusionPacketHeader packet{};
                    packet.wave = wave;
                    packet.source_rank = source;
                    packet.source_token = source_token;
                    packet.payload_offset = offset;
                    packet.kind = kFusionResultPayload;
                    if (!PublishPacket(sym, args,
                            args->symmetric_layout.combine_result_header_off,
                            args->symmetric_layout.combine_result_payload_off,
                            slot, source, owner, sequence++,
                            accumulator + row * row_bytes + offset,
                            bytes, packet, WorkerPe(args, source))) {
                        SetError(args, slot, kFusionSlotTimeout, 203u);
                        return;
                    }
                    offset += bytes;
                }
            }
            FusionPacketHeader end{};
            end.wave = wave;
            end.source_rank = source;
            end.kind = kFusionCombineLaneEnd;
            if (!PublishPacket(sym, args,
                    args->symmetric_layout.combine_result_header_off,
                    args->symmetric_layout.combine_result_payload_off,
                    slot, source, owner, sequence,
                    nullptr, 0u, end, WorkerPe(args, source))) {
                SetError(args, slot, kFusionSlotTimeout, 204u);
                return;
            }
        }
        if ((RequestFlags(args) & kFusionSerializeIncDc) != 0u) {
            const uint32_t progress_lane =
                args->resources.inc_dispatch_aiv + owner;
            PublishGeneration(IncProgressLine(args, progress_lane),
                              generation);
            if (owner == 0u) {
                if (!WaitAllGenerations(
                        IncProgressLine(args,
                            args->resources.inc_dispatch_aiv),
                        args->resources.inc_combine_aiv,
                        generation, args->spin_cap)) {
                    SetError(args, slot, kFusionSlotTimeout, 206u);
                    return;
                }
                __gm__ FusionSlotState *state = SlotState(args, slot);
                state->combine_generation = generation;
                AscendC::PipeBarrier<PIPE_ALL>();
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(state));
            }
        }
    }
    const uint32_t trace_lane = args->resources.inc_dispatch_aiv + owner;
    __gm__ FusionLaneTrace *trace = IncTraceLine(args, trace_lane);
    trace->start_cycle = trace_start;
    trace->end_cycle = AscendC::GetSystemCycle();
    trace->role = 2u;
    trace->lane = owner;
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(trace));
}

__aicore__ inline bool ValidArgs(__gm__ const FusionKernelArgs *args)
{
    if (args == nullptr || args->magic != kFusionMagic ||
        args->abi_version != kFusionAbiVersion)
        return false;
    const bool direct = (args->flags & kFusionWorkerDirectShmem) != 0u;
    const uint32_t direct_dispatch_lanes =
        args->worker_count *
        (args->resources.worker_dispatch_aiv > 1u
             ? args->resources.worker_dispatch_aiv / 2u : 1u);
    const uint32_t direct_combine_lanes =
        args->worker_count *
        (args->resources.worker_combine_aiv > 1u
             ? args->resources.worker_combine_aiv -
                   args->resources.worker_combine_aiv / 2u
             : 0u);
    return args->worker_count >= 2u &&
           args->worker_count <= kFusionMaxWorkers &&
           args->rank < args->worker_count &&
           args->slot_count >= kFusionMinSlots &&
           OperationGeneration(args) + args->wave_count <= 0xffffffffull &&
           args->resources.live_aiv != 0u &&
           args->resources.live_aiv <= kFusionMaxAiv &&
           args->resources.live_aic != 0u &&
           args->symmetric_layout.queue_lanes >=
               args->resources.inc_dispatch_aiv &&
           args->symmetric_layout.queue_lanes >=
               args->resources.inc_combine_aiv &&
           (!direct ||
            (args->symmetric_layout.queue_lanes >= direct_dispatch_lanes &&
             args->symmetric_layout.queue_lanes >= direct_combine_lanes));
}

__aicore__ inline bool ValidServiceControl(
    __gm__ const FusionServiceControl *control)
{
    return control != nullptr && control->magic == kFusionServiceMagic &&
           control->abi_version == kFusionServiceAbiVersion &&
           control->ring_size >= kFusionMinServiceRing &&
           control->ring_size <= kFusionMaxServiceRing &&
           control->live_aiv != 0u &&
           control->live_aiv <= kFusionMaxAiv &&
           control->descriptors != 0u && control->lane_progress != 0u;
}

__aicore__ inline __gm__ FusionServiceDescriptor *WaitServiceDescriptor(
    __gm__ FusionServiceControl *control, uint64_t ticket)
{
    __gm__ FusionServiceDescriptor *descriptor =
        reinterpret_cast<__gm__ FusionServiceDescriptor *>(
            control->descriptors) +
        (ticket - 1u) % control->ring_size;
    uint32_t polls = 0u;
    while (true) {
        if ((polls++ & 31u) == 0u) {
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(descriptor));
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(control));
        }
        if (descriptor->ready == ticket) {
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(descriptor));
            return descriptor->ready == ticket ? descriptor : nullptr;
        }
        if (control->stop != 0u) return nullptr;
    }
}

__aicore__ inline void PublishServiceLane(
    __gm__ FusionServiceControl *control, uint32_t lane, uint64_t ticket)
{
    __gm__ uint8_t *line = reinterpret_cast<__gm__ uint8_t *>(
        control->lane_progress) +
        static_cast<uint64_t>(lane) * kFusionCacheLineBytes;
    *reinterpret_cast<__gm__ uint64_t *>(line) = ticket;
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(line);
}

__aicore__ inline void WaitServiceCompletion(
    __gm__ FusionServiceControl *control, uint64_t ticket)
{
    uint32_t polls = 0u;
    while (control->completed_sequence < ticket) {
        if ((polls++ & 31u) == 0u)
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(control));
    }
}

__aicore__ inline bool PublishRemoteServiceRequest(
    __gm__ uint8_t *sym, __gm__ FusionKernelArgs *args,
    uint32_t logical_aiv)
{
    if ((args->flags & kFusionRemoteIncService) == 0u) return true;
    __gm__ const FusionRemoteServiceLayout *layout = &args->remote_service;
    // One publisher per worker. Rank zero additionally owns the command
    // descriptor and dynamic INC arguments; every rank owns one readiness
    // cache line so arbitrary host/stream timing cannot start a mixed ticket.
    if (logical_aiv != 0u) return true;
    __gm__ FusionServiceControl *local_control =
        reinterpret_cast<__gm__ FusionServiceControl *>(
            sym + layout->control_off);
    uint32_t ready_spins = 0u;
    while (true) {
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(local_control));
        if (local_control->magic == kFusionServiceMagic &&
            local_control->abi_version == kFusionServiceAbiVersion)
            break;
        if (args->spin_cap != 0u && ++ready_spins >= args->spin_cap)
            return false;
    }
    const uint64_t ticket = args->service_ticket;
    if (ticket == 0u || layout->ring_size < kFusionMinServiceRing ||
        layout->ring_size > kFusionMaxServiceRing ||
        args->waves == 0u || args->active_token_counts == 0u)
        return false;
    const uint32_t slot = static_cast<uint32_t>(
        (ticket - 1u) % layout->ring_size);
    __gm__ FusionServiceDescriptor *descriptor =
        reinterpret_cast<__gm__ FusionServiceDescriptor *>(
            sym + layout->descriptors_off) + slot;
    const uint64_t previous = ticket > layout->ring_size
        ? ticket - layout->ring_size : 0u;
    uint32_t spins = 0u;
    while (true) {
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(descriptor));
        if (descriptor->complete == previous) break;
        if (args->spin_cap != 0u && ++spins >= args->spin_cap)
            return false;
    }
    if (args->rank != 0u) return true;
    const int32_t inc = static_cast<int32_t>(args->inc_pe);

    if (layout->request_stride < layout->request_bytes ||
        layout->request_bytes < sizeof(FusionRemoteRequestHeader) ||
        layout->waves_off < layout->request_off ||
        layout->active_token_counts_off < layout->waves_off)
        return false;
    __gm__ uint8_t *request_bytes = sym + layout->request_off +
        static_cast<uint64_t>(slot) * layout->request_stride;
    __gm__ FusionRemoteRequestHeader *request =
        reinterpret_cast<__gm__ FusionRemoteRequestHeader *>(request_bytes);
    __gm__ uint8_t *request_waves = request_bytes +
        (layout->waves_off - layout->request_off);
    __gm__ uint8_t *request_active = request_bytes +
        (layout->active_token_counts_off - layout->request_off);
    Dcci(reinterpret_cast<__gm__ uint8_t *>(args->waves),
         args->wave_count * sizeof(FusionWaveDesc));
    Dcci(reinterpret_cast<__gm__ uint8_t *>(args->active_token_counts),
         args->worker_count * sizeof(uint32_t));
    request->operation_generation = args->operation_generation;
    request->service_ticket = ticket;
    request->request_id = args->request_id;
    request->flags = args->flags;
    request->wave_count = args->wave_count;
    request->worker_count = args->worker_count;
    request->bytes = layout->request_bytes;
    __gm__ const uint64_t *source_waves =
        reinterpret_cast<__gm__ const uint64_t *>(args->waves);
    __gm__ uint64_t *destination_waves =
        reinterpret_cast<__gm__ uint64_t *>(request_waves);
    const uint32_t wave_words = args->wave_count *
        sizeof(FusionWaveDesc) / sizeof(uint64_t);
    for (uint32_t word = 0u; word < wave_words; ++word)
        destination_waves[word] = source_waves[word];
    __gm__ const uint32_t *source_active =
        reinterpret_cast<__gm__ const uint32_t *>(
            args->active_token_counts);
    __gm__ uint32_t *destination_active =
        reinterpret_cast<__gm__ uint32_t *>(request_active);
    for (uint32_t worker = 0u; worker < args->worker_count; ++worker)
        destination_active[worker] = source_active[worker];
    AscendC::PipeBarrier<PIPE_ALL>();
    Dcci(request_bytes, layout->request_bytes);
    // One dynamic record and one ready word are the complete command commit.
    // This is both faster and safer than several independent tiny SDMA SQEs.
    aclshmem_putmem(request_bytes, request_bytes,
                    layout->request_bytes, inc);
    descriptor->ready = ticket;
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(descriptor));
    aclshmem_putmem(
        reinterpret_cast<__gm__ uint8_t *>(&descriptor->ready),
        reinterpret_cast<__gm__ uint8_t *>(&descriptor->ready),
        sizeof(descriptor->ready), inc);
    return true;
}

// Optional control-plane phase executed by lane zero of the persistent INC
// before it accepts the matching data-plane request.  Each worker has already
// put one disjoint count block and published its source doorbell.  The INC
// commits the complete table to every worker and only then publishes one
// completion doorbell.  This is deliberately star-shaped and put-only.
__aicore__ inline bool RelayRouteCounts(
    __gm__ uint8_t *sym, __gm__ FusionServiceControl *control,
    uint64_t generation)
{
    const uint64_t count_words64 = control->reserved64[2];
    const uint32_t worker_count = static_cast<uint32_t>(control->reserved64[3]);
    if (count_words64 == 0u) return true;
    if (count_words64 > 0xffffffffull || worker_count == 0u ||
        worker_count > kFusionMaxWorkers || control->reserved64[4] == 0u)
        return false;
    const uint32_t count_words = static_cast<uint32_t>(count_words64);
    const uint64_t global_bytes = static_cast<uint64_t>(worker_count) *
        count_words * sizeof(uint32_t);
    __gm__ uint8_t *counts = sym + control->reserved64[0];
    __gm__ uint8_t *ready = sym + control->reserved64[1];
    __gm__ const uint32_t *worker_pes =
        reinterpret_cast<__gm__ const uint32_t *>(control->reserved64[4]);

    uint32_t polls = 0u;
    for (uint32_t source = 0u; source < worker_count; ++source) {
        __gm__ uint8_t *source_ready =
            ready + static_cast<uint64_t>(source) * kFusionCacheLineBytes;
        while (true) {
            dcci_cacheline(source_ready);
            if (*reinterpret_cast<__gm__ uint64_t *>(source_ready) ==
                generation)
                break;
            if ((polls++ & 31u) == 0u) {
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(control));
                if (control->stop != 0u) return false;
            }
        }
    }
    Dcci(counts, global_bytes);
    Dcci(reinterpret_cast<__gm__ uint8_t *>(
             const_cast<__gm__ uint32_t *>(worker_pes)),
         static_cast<uint64_t>(worker_count) * sizeof(uint32_t));
    for (uint32_t worker = 0u; worker < worker_count; ++worker)
        aclshmem_putmem(counts, counts, global_bytes,
                        static_cast<int32_t>(worker_pes[worker]));

    __gm__ uint8_t *complete = ready +
        static_cast<uint64_t>(worker_count) * kFusionCacheLineBytes;
    *reinterpret_cast<__gm__ uint64_t *>(complete) = generation;
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(complete);
    for (uint32_t worker = 0u; worker < worker_count; ++worker)
        aclshmem_putmem(complete, complete, sizeof(uint64_t),
                        static_cast<int32_t>(worker_pes[worker]));
    return true;
}

} // namespace

// Latency-oriented put-only exchange for the tiny per-wave route histogram.
// Every source owns one cache-disjoint block and one doorbell on every
// worker.  The storage aliases bulk route/control arenas only before the
// corresponding fusion request starts; stream order and the all-source
// completion below make that reuse safe without growing the symmetric heap.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
inc_fusion_route_exchange_kernel(
    __gm__ uint8_t *sym, __gm__ const uint32_t *local_counts,
    __gm__ uint32_t *global_counts, __gm__ const uint32_t *worker_pes,
    uint64_t ffts_addr, uint64_t counts_off, uint64_t doorbells_off,
    uint32_t count_words, uint32_t worker_count, uint32_t rank,
    int32_t relay_pe, uint64_t generation)
{
    if (static_cast<uint32_t>(AscendC::GetBlockIdx()) != 0u ||
        sym == nullptr || local_counts == nullptr ||
        global_counts == nullptr || worker_pes == nullptr ||
        count_words == 0u || worker_count == 0u || rank >= worker_count)
        return;
    shmemx_set_ffts_config(ffts_addr);
    const uint64_t count_bytes =
        static_cast<uint64_t>(count_words) * sizeof(uint32_t);
    __gm__ uint32_t *counts = reinterpret_cast<__gm__ uint32_t *>(
        sym + counts_off);
    __gm__ uint32_t *owned = counts +
        static_cast<uint64_t>(rank) * count_words;
    Dcci(reinterpret_cast<__gm__ uint8_t *>(
             const_cast<__gm__ uint32_t *>(local_counts)), count_bytes);
    Dcci(reinterpret_cast<__gm__ uint8_t *>(
             const_cast<__gm__ uint32_t *>(worker_pes)),
         static_cast<uint64_t>(worker_count) * sizeof(uint32_t));
    for (uint32_t word = 0u; word < count_words; ++word)
        owned[word] = local_counts[word];
    AscendC::PipeBarrier<PIPE_ALL>();
    Dcci(reinterpret_cast<__gm__ uint8_t *>(owned), count_bytes);
    __gm__ uint8_t *ready = sym + doorbells_off +
        static_cast<uint64_t>(rank) * kFusionCacheLineBytes;
    *reinterpret_cast<__gm__ uint64_t *>(ready) = generation;
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cacheline(ready);
    if (relay_pe >= 0) {
        // Counts are ready before the source doorbell; the INC performs the
        // only fan-out and publishes ready[worker_count] last.
        aclshmem_putmem(
            reinterpret_cast<__gm__ uint8_t *>(owned),
            reinterpret_cast<__gm__ uint8_t *>(owned), count_bytes, relay_pe);
        aclshmem_putmem(ready, ready, sizeof(uint64_t), relay_pe);
        if (!WaitGeneration(
                sym + doorbells_off +
                    static_cast<uint64_t>(worker_count) *
                        kFusionCacheLineBytes,
                generation, 0u))
            return;
    } else {
        for (uint32_t destination = 0u; destination < worker_count;
             ++destination) {
            if (destination == rank) continue;
            aclshmem_putmem(
                reinterpret_cast<__gm__ uint8_t *>(owned),
                reinterpret_cast<__gm__ uint8_t *>(owned), count_bytes,
                static_cast<int32_t>(worker_pes[destination]));
        }
        for (uint32_t destination = 0u; destination < worker_count;
             ++destination) {
            if (destination == rank) continue;
            aclshmem_putmem(
                ready, ready, sizeof(uint64_t),
                static_cast<int32_t>(worker_pes[destination]));
        }
        for (uint32_t source = 0u; source < worker_count; ++source) {
            if (!WaitGeneration(
                    sym + doorbells_off +
                        static_cast<uint64_t>(source) * kFusionCacheLineBytes,
                    generation, 0u))
                return;
        }
    }
    const uint64_t global_words =
        static_cast<uint64_t>(worker_count) * count_words;
    Dcci(reinterpret_cast<__gm__ uint8_t *>(counts),
         global_words * sizeof(uint32_t));
    for (uint64_t word = 0u; word < global_words; ++word)
        global_counts[word] = counts[word];
    AscendC::PipeBarrier<PIPE_ALL>();
    Dcci(reinterpret_cast<__gm__ uint8_t *>(global_counts),
         global_words * sizeof(uint32_t));
}

extern "C" __global__ __aicore__ __mix__(1, 2) void
inc_fusion_worker_kernel(__gm__ uint8_t *sym,
                         __gm__ FusionKernelArgs *args,
                         uint64_t expected_ticket)
{
    // Args are produced by asynchronous H2D into a reusable ring slot. Every
    // MIX sub-core must acquire the record before validation; otherwise an
    // AIC can observe the new generation while an AIV retains the zeroed
    // setup-time cache line and silently skips communication.
    if (args != nullptr) {
        do {
            Dcci(reinterpret_cast<__gm__ uint8_t *>(args),
                 sizeof(FusionKernelArgs));
        } while (expected_ticket != 0u &&
                 args->service_ticket != expected_ticket);
    }
    if (!ValidArgs(args) || args->role != kFusionWorker) return;
    if ASCEND_IS_AIC {
        WorkerComputeAic(args);
    }
    if ASCEND_IS_AIV {
        shmemx_set_ffts_config(args->ffts_addr);
        const uint32_t logical_aiv =
            static_cast<uint32_t>(AscendC::GetBlockIdx());
        if ((args->flags & kFusionRemoteIncService) != 0u) {
            const uint32_t service_slot = static_cast<uint32_t>(
                (args->service_ticket - 1u) %
                    args->remote_service.ring_size);
            __gm__ uint8_t *worker_ready = sym +
                args->remote_service.worker_ready_off +
                (static_cast<uint64_t>(service_slot) * args->worker_count +
                 args->rank) * kFusionCacheLineBytes;
            if (logical_aiv == 0u) {
                *reinterpret_cast<__gm__ uint64_t *>(worker_ready) =
                    args->service_ticket;
                AscendC::PipeBarrier<PIPE_ALL>();
                dcci_cacheline(worker_ready);
                aclshmem_putmem_nbi(
                    worker_ready, worker_ready, sizeof(uint64_t),
                    static_cast<int32_t>(args->inc_pe));
                aclshmem_quiet();
            }
            __gm__ FusionServiceControl *local_control =
                reinterpret_cast<__gm__ FusionServiceControl *>(
                    sym + args->remote_service.control_off);
            if (!WaitGeneration(
                    reinterpret_cast<__gm__ uint8_t *>(
                        &local_control->reserved0),
                    args->service_ticket, args->spin_cap)) {
                SetError(args, 0u, kFusionSlotTimeout, 302u,
                         args->service_ticket);
                return;
            }
        }
        if (logical_aiv < args->resources.worker_dispatch_aiv) {
            if ((args->flags & kFusionWorkerDirectShmem) != 0u) {
                WorkerDirectDispatchTx(sym, args, logical_aiv);
                WorkerDirectDispatchRx(sym, args, logical_aiv);
            } else if ((args->flags & kFusionBulkWaveTransport) != 0u) {
                WorkerBulkDispatch(sym, args, logical_aiv);
            } else {
                WorkerDispatchTx(sym, args, logical_aiv);
                WorkerDispatchRx(sym, args, logical_aiv);
            }
        } else if (logical_aiv <
                   args->resources.worker_dispatch_aiv +
                       args->resources.worker_combine_aiv) {
            if ((args->flags & kFusionWorkerDirectShmem) != 0u)
                WorkerDirectCombine(sym, args, logical_aiv);
            else if ((args->flags & kFusionBulkWaveTransport) != 0u)
                WorkerBulkCombine(sym, args, logical_aiv);
            else
                WorkerCombine(sym, args, logical_aiv);
        } else {
            WorkerActivation(args, logical_aiv);
        }
    }
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
inc_fusion_remote_publish_kernel(__gm__ uint8_t *sym,
                                 __gm__ FusionKernelArgs *args,
                                 uint64_t expected_ticket)
{
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() != 0u || args == nullptr) return;
        do {
            Dcci(reinterpret_cast<__gm__ uint8_t *>(args),
                 sizeof(FusionKernelArgs));
        } while (args->service_ticket != expected_ticket);
        shmemx_set_ffts_config(args->ffts_addr);
        if (!ValidArgs(args) || args->role != kFusionWorker ||
            (args->flags & kFusionRemoteIncService) == 0u)
            return;
        if (!PublishRemoteServiceRequest(sym, args, 0u))
            SetError(args, 0u, kFusionSlotTimeout, 301u,
                     args->service_ticket);
    }
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
inc_fusion_inc_service_kernel(__gm__ uint8_t *sym,
                              __gm__ FusionKernelArgs *args)
{
    if ASCEND_IS_AIV {
        if (args != nullptr)
            Dcci(reinterpret_cast<__gm__ uint8_t *>(args),
                 sizeof(FusionKernelArgs));
        if (!ValidArgs(args) || args->role != kFusionInc) return;
        shmemx_set_ffts_config(args->ffts_addr);
        const uint32_t logical_aiv =
            static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (logical_aiv < args->resources.inc_dispatch_aiv) {
            if ((args->flags & kFusionBulkWaveTransport) != 0u)
                IncBulkDispatch(sym, args, logical_aiv);
            else
                IncDispatch(sym, args, logical_aiv);
        } else {
            if ((args->flags & kFusionBulkWaveTransport) != 0u)
                IncBulkCombine(sym, args,
                    logical_aiv - args->resources.inc_dispatch_aiv);
            else
                IncCombine(sym, args,
                    logical_aiv - args->resources.inc_dispatch_aiv);
        }
    }
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
inc_fusion_inc_persistent_service_kernel(
    __gm__ uint8_t *sym, __gm__ FusionServiceControl *control)
{
    if ASCEND_IS_AIV {
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(control));
        if (!ValidServiceControl(control)) return;
        if (control->reserved64[2] != 0u)
            shmemx_set_ffts_config(control->reserved64[5]);
        const uint32_t logical_aiv =
            static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (logical_aiv >= control->live_aiv) return;
        uint64_t ticket = 1u;
        while (true) {
            if (logical_aiv == 0u && control->reserved64[2] != 0u &&
                !RelayRouteCounts(sym, control, ticket))
                return;
            __gm__ FusionServiceDescriptor *descriptor =
                WaitServiceDescriptor(control, ticket);
            if (descriptor == nullptr) return;
            __gm__ FusionKernelArgs *args =
                reinterpret_cast<__gm__ FusionKernelArgs *>(
                    descriptor->device_args);
            if (args != nullptr)
                Dcci(reinterpret_cast<__gm__ uint8_t *>(args),
                     sizeof(FusionKernelArgs));
            if (args != nullptr && args->remote_request != 0u)
                Dcci(reinterpret_cast<__gm__ uint8_t *>(
                         args->remote_request),
                     args->remote_service.request_bytes);
            bool valid = ValidArgs(args) && args->role == kFusionInc &&
                args->resources.live_aiv == control->live_aiv;
            __gm__ const FusionRemoteRequestHeader *request =
                valid ? RemoteRequest(args) : nullptr;
            if (valid &&
                (args->flags & kFusionRemoteIncService) != 0u) {
                valid = request != nullptr &&
                    request->service_ticket == ticket &&
                    request->wave_count == args->wave_count &&
                    request->worker_count == args->worker_count &&
                    request->bytes == args->remote_service.request_bytes;
            }
            if (valid)
                shmemx_set_ffts_config(args->ffts_addr);
            if (valid &&
                (args->flags & kFusionRemoteIncService) != 0u) {
                const uint32_t slot = static_cast<uint32_t>(
                    (ticket - 1u) % args->remote_service.ring_size);
                __gm__ uint8_t *worker_ready = sym +
                    args->remote_service.worker_ready_off +
                    static_cast<uint64_t>(slot) * args->worker_count *
                        kFusionCacheLineBytes;
                valid = WaitAllGenerations(
                    worker_ready, args->worker_count, ticket,
                    args->spin_cap);
                if (valid && logical_aiv == 0u) {
                    __gm__ uint8_t *start_done =
                        reinterpret_cast<__gm__ uint8_t *>(
                            &control->reserved0);
                    PublishGeneration(start_done, ticket);
                    for (uint32_t worker = 0u;
                         worker < args->worker_count; ++worker) {
                        aclshmem_putmem(
                            start_done, start_done, sizeof(uint64_t),
                            WorkerPe(args, worker));
                    }
                }
            }
            if (valid) {
                if (logical_aiv < args->resources.inc_dispatch_aiv) {
                    if ((args->flags & kFusionBulkWaveTransport) != 0u)
                        IncBulkDispatch(sym, args, logical_aiv);
                    else
                        IncDispatch(sym, args, logical_aiv);
                } else {
                    if ((args->flags & kFusionBulkWaveTransport) != 0u)
                        IncBulkCombine(sym, args,
                            logical_aiv -
                                args->resources.inc_dispatch_aiv);
                    else
                        IncCombine(sym, args,
                            logical_aiv -
                                args->resources.inc_dispatch_aiv);
                }
            }

            PublishServiceLane(control, logical_aiv, ticket);
            if (logical_aiv == 0u) {
                if (!WaitAllGenerations(
                        reinterpret_cast<__gm__ uint8_t *>(
                            control->lane_progress),
                        control->live_aiv, ticket, 0u))
                    return;
                uint32_t status = valid
                    ? kFusionServiceRequestSuccess
                    : kFusionServiceRequestBadArgs;
                uint32_t operator_error = kFusionSlotOk;
                if (valid) {
                    for (uint32_t slot = 0u; slot < args->slot_count; ++slot) {
                        __gm__ FusionSlotState *state = SlotState(args, slot);
                        dcci_cacheline(
                            reinterpret_cast<__gm__ uint8_t *>(state));
                        if (state->error != kFusionSlotOk) {
                            status = kFusionServiceRequestOperatorError;
                            operator_error = state->error;
                            break;
                        }
                    }
                }
                descriptor->status = status;
                descriptor->operator_error = operator_error;
                descriptor->request_id = valid ? RequestId(args) : 0u;
                AscendC::PipeBarrier<PIPE_ALL>();
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(descriptor));
                descriptor->complete = ticket;
                AscendC::PipeBarrier<PIPE_ALL>();
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(descriptor));
                if (valid &&
                    (args->flags & kFusionRemoteIncService) != 0u) {
                    // Return completion to worker-rank-zero's symmetric
                    // mirror. The next wrapped ticket cannot reuse the slot
                    // before this exact complete value is observed.
                    for (uint32_t worker = 0u;
                         worker < args->worker_count; ++worker) {
                        aclshmem_putmem(descriptor, descriptor,
                            sizeof(FusionServiceDescriptor),
                            WorkerPe(args, worker));
                    }
                }
                control->completed_sequence = ticket;
                AscendC::PipeBarrier<PIPE_ALL>();
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(control));
            } else {
                WaitServiceCompletion(control, ticket);
            }
            ++ticket;
        }
    }
}

extern "C" void launch_inc_fusion_worker_kernel(
    uint8_t *sym, FusionKernelArgs *args, uint64_t expected_ticket,
    int block_dim, void *stream)
{
    inc_fusion_worker_kernel<<<block_dim, nullptr, stream>>>(
        sym, args, expected_ticket);
}

extern "C" void launch_inc_fusion_route_exchange_kernel(
    uint8_t *sym, const uint32_t *local_counts, uint32_t *global_counts,
    const uint32_t *worker_pes, uint64_t ffts_addr, uint64_t counts_off,
    uint64_t doorbells_off, uint32_t count_words, uint32_t worker_count,
    uint32_t rank, int32_t relay_pe, uint64_t generation, void *stream)
{
    inc_fusion_route_exchange_kernel<<<1, nullptr, stream>>>(
        sym, local_counts, global_counts, worker_pes, ffts_addr, counts_off,
        doorbells_off, count_words, worker_count, rank, relay_pe, generation);
}

extern "C" void launch_inc_fusion_remote_publish_kernel(
    uint8_t *sym, FusionKernelArgs *args, uint64_t expected_ticket,
    void *stream)
{
    inc_fusion_remote_publish_kernel<<<1, nullptr, stream>>>(
        sym, args, expected_ticket);
}

extern "C" void launch_inc_fusion_inc_service_kernel(
    uint8_t *sym, FusionKernelArgs *args, int block_dim, void *stream)
{
    inc_fusion_inc_service_kernel<<<block_dim, nullptr, stream>>>(sym, args);
}

extern "C" void launch_inc_fusion_inc_persistent_service_kernel(
    uint8_t *sym, FusionServiceControl *control,
    int block_dim, void *stream)
{
    inc_fusion_inc_persistent_service_kernel<<<block_dim, nullptr, stream>>>(
        sym, control);
}
