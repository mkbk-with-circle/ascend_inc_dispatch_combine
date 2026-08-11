#include "kernel_operator.h"

#include "inc_fusion_route_pack.h"

using namespace inc::fusion;

namespace {

__aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return (value - 1u) / divisor + 1u;
}

__aicore__ inline uint32_t LoadExpert(__gm__ const void *ids,
                                      uint32_t id_type, uint64_t index)
{
    if (id_type == kFusionRouteInt64)
        return static_cast<uint32_t>(
            reinterpret_cast<__gm__ const int64_t *>(ids)[index]);
    return reinterpret_cast<__gm__ const uint32_t *>(ids)[index];
}

__aicore__ inline uint32_t FloatBits(float value)
{
    union { float f; uint32_t u; } bits;
    bits.f = value;
    return bits.u;
}

__aicore__ inline bool IsFinite(float value)
{
    return (FloatBits(value) & 0x7f800000u) != 0x7f800000u;
}

__aicore__ inline void CleanInvalidateCacheLine(__gm__ uint8_t *line)
{
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(line);
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<
        uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

__aicore__ inline void FlushRange(__gm__ void *pointer, uint64_t bytes)
{
    __gm__ uint8_t *line = reinterpret_cast<__gm__ uint8_t *>(pointer);
    for (uint64_t offset = 0u; offset < bytes;
         offset += kFusionCacheLineBytes)
        CleanInvalidateCacheLine(line + offset);
}

__aicore__ inline uint64_t ScratchBytes(uint32_t expert_count)
{
    return static_cast<uint64_t>(expert_count) * 2u * sizeof(uint32_t);
}

constexpr uint32_t kParallelLineWords =
    kFusionCacheLineBytes / sizeof(uint32_t);
constexpr uint32_t kParallelRecordWords = 8u;
constexpr uint32_t kParallelBufferedRecords = 64u;

__aicore__ inline uint64_t ParallelExpertStride(uint32_t expert_count)
{
    return (static_cast<uint64_t>(expert_count) + kParallelLineWords - 1u) /
        kParallelLineWords * kParallelLineWords;
}

__aicore__ inline uint64_t ParallelLaneStateOffset(
    uint32_t wave_capacity, uint32_t lane_count, uint64_t expert_stride)
{
    return static_cast<uint64_t>(wave_capacity) * lane_count *
        expert_stride;
}

__aicore__ inline uint64_t ParallelExpertBaseOffset(
    uint32_t wave_capacity, uint32_t lane_count, uint64_t expert_stride)
{
    return ParallelLaneStateOffset(
               wave_capacity, lane_count, expert_stride) +
        static_cast<uint64_t>(wave_capacity) * lane_count *
            kParallelLineWords;
}

__aicore__ inline uint64_t ParallelErrorOffset(
    uint32_t wave_capacity, uint32_t lane_count, uint64_t expert_stride)
{
    return ParallelExpertBaseOffset(
               wave_capacity, lane_count, expert_stride) +
        static_cast<uint64_t>(wave_capacity) * expert_stride;
}

__aicore__ inline uint64_t ParallelScratchWords(
    uint32_t wave_capacity, uint32_t lane_count, uint64_t expert_stride)
{
    return ParallelErrorOffset(
               wave_capacity, lane_count, expert_stride) +
        static_cast<uint64_t>(lane_count) * kParallelLineWords;
}

__aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint32_t PopcountDestinations(uint64_t bits)
{
    uint32_t count = 0u;
    while (bits != 0u) {
        bits &= bits - 1u;
        ++count;
    }
    return count;
}

__aicore__ inline void AcquireRange(__gm__ const void *pointer,
                                    uint64_t bytes)
{
    // The Torch bridge first drains its software task queue into the current
    // ACL stream.  DCCI can then safely refresh scalar reads after all MTE/
    // vector producers that precede this kernel on that stream.
    FlushRange(const_cast<__gm__ void *>(pointer), bytes);
}

__aicore__ inline void SetFailure(
    __gm__ FusionRoutePackStatus *status, uint32_t error,
    uint32_t token = 0u, uint32_t ordinal = 0u, uint32_t expert = 0u)
{
    status->error = error;
    status->error_token = token;
    status->error_ordinal = ordinal;
    status->error_expert = expert;
    AscendC::PipeBarrier<PIPE_ALL>();
    CleanInvalidateCacheLine(reinterpret_cast<__gm__ uint8_t *>(status));
}

__aicore__ inline void SetFailureWithScratch(
    __gm__ FusionRoutePackStatus *status, __gm__ uint32_t *scratch,
    uint32_t expert_count, uint32_t error, uint32_t token = 0u,
    uint32_t ordinal = 0u, uint32_t expert = 0u)
{
    (void)expert_count;
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::GlobalTensor<uint32_t> global;
    global.SetGlobalBuffer(scratch);
    // Error-only path: release scratch and every partially written protocol
    // output so a caching allocator may immediately reuse those addresses.
    AscendC::DataCacheCleanAndInvalid<
        uint32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
        AscendC::DcciDst::CACHELINE_OUT>(global);
    SetFailure(status, error, token, ordinal, expert);
}

__aicore__ inline void ResetStatus(__gm__ FusionRoutePackStatus *status)
{
    status->error = kFusionRoutePackOk;
    status->error_token = 0u;
    status->error_ordinal = 0u;
    status->error_expert = 0u;
    status->dispatch_row_count = 0u;
    status->assignment_count = 0u;
    status->wave_count = 0u;
    status->reserved32 = 0u;
    for (uint32_t i = 0u; i < 4u; ++i) status->reserved64[i] = 0u;
}

__aicore__ inline void StoreDispatchRow(
    __gm__ FusionDispatchRow *output, const FusionDispatchRow &value)
{
    output->source_rank = value.source_rank;
    output->source_token = value.source_token;
    output->destination_rank = value.destination_rank;
    output->assignment_begin = value.assignment_begin;
    output->assignment_count = value.assignment_count;
    output->wave = value.wave;
    output->payload_byte_offset = value.payload_byte_offset;
}

__aicore__ inline void StoreAssignment(
    __gm__ FusionExpertAssignment *output,
    const FusionExpertAssignment &value)
{
    output->dispatch_row = value.dispatch_row;
    output->expert_id = value.expert_id;
    output->local_expert = value.local_expert;
    output->route_ordinal = value.route_ordinal;
    output->destination_token = value.destination_token;
    output->weight_bits = value.weight_bits;
    output->wave = value.wave;
    output->destination_row = value.destination_row;
}

__aicore__ inline void StoreWave(__gm__ FusionWaveDesc *output,
                                  const FusionWaveDesc &value)
{
    output->generation = value.generation;
    output->token_begin = value.token_begin;
    output->token_count = value.token_count;
    output->dispatch_row_begin = value.dispatch_row_begin;
    output->dispatch_row_count = value.dispatch_row_count;
    output->assignment_begin = value.assignment_begin;
    output->assignment_count = value.assignment_count;
    output->slot = value.slot;
    output->activation_waves = value.activation_waves;
    for (uint32_t i = 0u; i < 4u; ++i)
        output->reserved32[i] = value.reserved32[i];
    output->reserved64 = value.reserved64;
}

__aicore__ inline uint32_t ExpertTotal(
    __gm__ const uint32_t *global_counts, uint32_t worker_count,
    uint32_t wave_count, uint32_t expert_count, uint32_t wave,
    uint32_t expert)
{
    const uint32_t count_stride = expert_count + 1u;
    uint32_t total = 0u;
    for (uint32_t source = 0u; source < worker_count; ++source) {
        const uint64_t index =
            (static_cast<uint64_t>(source) * wave_count + wave) *
                count_stride + expert;
        total += global_counts[index];
    }
    return total;
}

__aicore__ inline void FlushRecordBuffer(
    AscendC::LocalTensor<uint32_t> local, uint32_t records,
    __gm__ uint32_t *output, uint32_t output_record)
{
    if (records == 0u) return;
    AscendC::GlobalTensor<uint32_t> global;
    global.SetGlobalBuffer(output +
        static_cast<uint64_t>(output_record) * kParallelRecordWords);
    AscendC::DataCopyExtParams params{
        static_cast<uint16_t>(1u),
        static_cast<uint32_t>(records * kParallelRecordWords *
                              sizeof(uint32_t)),
        0u, 0u, 0u};
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0u);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0u);
    AscendC::DataCopyPad(global, local, params);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0u);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0u);
}

__aicore__ inline void BufferDispatchRow(
    AscendC::LocalTensor<uint32_t> output, uint32_t record,
    uint32_t source_rank, uint32_t source_token,
    uint32_t destination_rank, uint32_t assignment_begin,
    uint32_t assignment_count, uint32_t wave,
    uint64_t payload_byte_offset)
{
    const uint32_t base = record * kParallelRecordWords;
    output.SetValue(base + 0u, source_rank);
    output.SetValue(base + 1u, source_token);
    output.SetValue(base + 2u, destination_rank);
    output.SetValue(base + 3u, assignment_begin);
    output.SetValue(base + 4u, assignment_count);
    output.SetValue(base + 5u, wave);
    output.SetValue(base + 6u,
                    static_cast<uint32_t>(payload_byte_offset));
    output.SetValue(base + 7u,
                    static_cast<uint32_t>(payload_byte_offset >> 32u));
}

__aicore__ inline void BufferAssignment(
    AscendC::LocalTensor<uint32_t> output, uint32_t record,
    uint32_t dispatch_row, uint32_t expert, uint32_t local_expert,
    uint32_t ordinal, uint32_t token, uint32_t weight_bits,
    uint32_t wave, uint32_t destination_row)
{
    const uint32_t base = record * kParallelRecordWords;
    output.SetValue(base + 0u, dispatch_row);
    output.SetValue(base + 1u, expert);
    output.SetValue(base + 2u, local_expert);
    output.SetValue(base + 3u, ordinal);
    output.SetValue(base + 4u, token);
    output.SetValue(base + 5u, weight_bits);
    output.SetValue(base + 6u, wave);
    output.SetValue(base + 7u, destination_row);
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
inc_fusion_route_count_kernel(
    __gm__ const void *topk_ids, uint32_t id_type,
    uint32_t token_count, uint32_t topk, uint32_t expert_count,
    uint32_t tokens_per_wave, uint32_t wave_capacity,
    __gm__ uint32_t *local_counts,
    __gm__ FusionRoutePackStatus *status)
{
    if (AscendC::GetBlockIdx() != 0) return;
    if (status == nullptr) return;
    ResetStatus(status);
    if ((token_count != 0u && topk_ids == nullptr) ||
        local_counts == nullptr || topk == 0u || expert_count == 0u ||
        tokens_per_wave == 0u ||
        wave_capacity == 0u ||
        (id_type != kFusionRouteInt32 && id_type != kFusionRouteInt64)) {
        SetFailure(status, kFusionRoutePackBadArgs);
        return;
    }
    const uint32_t active_wave_count = token_count == 0u
        ? 0u : CeilDiv(token_count, tokens_per_wave);
    if (active_wave_count > wave_capacity) {
        SetFailure(status, kFusionRoutePackCapacity);
        return;
    }
    const uint32_t count_stride = expert_count + 1u;
    const uint64_t count_elements =
        static_cast<uint64_t>(wave_capacity) * count_stride;
    for (uint64_t i = 0u; i < count_elements; ++i)
        local_counts[i] = 0u;
    for (uint32_t token = 0u; token < token_count; ++token) {
        const uint32_t wave = token / tokens_per_wave;
        for (uint32_t ordinal = 0u; ordinal < topk; ++ordinal) {
            const uint64_t index =
                static_cast<uint64_t>(token) * topk + ordinal;
            const uint32_t expert = LoadExpert(topk_ids, id_type, index);
            if (expert >= expert_count) {
                AscendC::PipeBarrier<PIPE_ALL>();
                FlushRange(local_counts,
                           count_elements * sizeof(uint32_t));
                SetFailure(status, kFusionRoutePackBadExpert,
                           token, ordinal, expert);
                return;
            }
            ++local_counts[static_cast<uint64_t>(wave) * count_stride +
                           expert];
        }
    }
    local_counts[expert_count] = token_count;
    status->wave_count = wave_capacity;
    AscendC::PipeBarrier<PIPE_ALL>();
    FlushRange(local_counts, count_elements * sizeof(uint32_t));
    CleanInvalidateCacheLine(reinterpret_cast<__gm__ uint8_t *>(status));
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
inc_fusion_route_pack_kernel(
    __gm__ const void *topk_ids, uint32_t id_type,
    __gm__ const float *topk_weights,
    __gm__ const uint32_t *global_counts,
    __gm__ const uint32_t *expert_owner,
    __gm__ const uint32_t *expert_local_index,
    uint32_t worker_count, uint32_t rank, uint32_t token_count,
    uint32_t topk, uint32_t hidden, uint32_t element_bytes,
    uint32_t expert_count, uint32_t local_expert_count,
    uint32_t tokens_per_wave, uint32_t slot_count,
    uint32_t activation_waves,
    __gm__ FusionDispatchRow *dispatch_rows,
    uint32_t dispatch_row_capacity,
    __gm__ FusionExpertAssignment *assignments,
    uint32_t assignment_capacity, __gm__ int64_t *group_lists,
    __gm__ FusionWaveDesc *waves, uint32_t wave_capacity,
    __gm__ uint32_t *scratch, __gm__ FusionRoutePackStatus *status)
{
    if (AscendC::GetBlockIdx() != 0) return;
    if (status == nullptr) return;
    ResetStatus(status);
    if ((token_count != 0u &&
         (topk_ids == nullptr || topk_weights == nullptr)) ||
        global_counts == nullptr || expert_owner == nullptr ||
        expert_local_index == nullptr || dispatch_rows == nullptr ||
        assignments == nullptr || group_lists == nullptr || waves == nullptr ||
        scratch == nullptr || worker_count == 0u || rank >= worker_count ||
        worker_count > kFusionMaxWorkers || topk == 0u ||
        hidden == 0u || element_bytes == 0u || expert_count == 0u ||
        tokens_per_wave == 0u || slot_count < kFusionMinSlots ||
        activation_waves == 0u ||
        (id_type != kFusionRouteInt32 && id_type != kFusionRouteInt64)) {
        SetFailure(status, kFusionRoutePackBadArgs);
        return;
    }
    const uint32_t count_stride = expert_count + 1u;
    const uint64_t global_count_elements =
        static_cast<uint64_t>(worker_count) * wave_capacity * count_stride;
    AcquireRange(global_counts,
                 global_count_elements * sizeof(uint32_t));
    AcquireRange(topk_ids,
        static_cast<uint64_t>(token_count) * topk *
        (id_type == kFusionRouteInt64 ? sizeof(int64_t) : sizeof(uint32_t)));
    AcquireRange(topk_weights,
        static_cast<uint64_t>(token_count) * topk * sizeof(float));
    AcquireRange(expert_owner,
                 static_cast<uint64_t>(expert_count) * sizeof(uint32_t));
    AcquireRange(expert_local_index,
                 static_cast<uint64_t>(expert_count) * sizeof(uint32_t));
    uint32_t collective_token_count = 0u;
    for (uint32_t source = 0u; source < worker_count; ++source) {
        const uint32_t active = global_counts[
            static_cast<uint64_t>(source) * wave_capacity * count_stride +
            expert_count];
        if (active > collective_token_count)
            collective_token_count = active;
    }
    if (global_counts[
            static_cast<uint64_t>(rank) * wave_capacity * count_stride +
            expert_count] != token_count) {
        SetFailure(status, kFusionRoutePackBadArgs);
        return;
    }
    const uint32_t local_wave_count = token_count == 0u
        ? 0u : CeilDiv(token_count, tokens_per_wave);
    const uint32_t active_wave_count = collective_token_count == 0u
        ? 0u : CeilDiv(collective_token_count, tokens_per_wave);
    if (local_wave_count > wave_capacity ||
        active_wave_count > wave_capacity) {
        SetFailure(status, kFusionRoutePackCapacity);
        return;
    }
    const uint32_t wave_count = wave_capacity;
    uint32_t observed_local_experts = 0u;
    for (uint32_t expert = 0u; expert < expert_count; ++expert) {
        const uint32_t owner = expert_owner[expert];
        const uint32_t local = expert_local_index[expert];
        if (owner >= worker_count) {
            SetFailure(status, kFusionRoutePackBadPlacement, 0u, 0u,
                       expert);
            return;
        }
        uint32_t lower = 0u;
        for (uint32_t other = 0u; other < expert_count; ++other) {
            if (expert_owner[other] != owner) continue;
            const uint32_t other_local = expert_local_index[other];
            if (other_local < local) ++lower;
            if (other != expert && other_local == local) {
                SetFailure(status, kFusionRoutePackBadPlacement, 0u, 0u,
                           expert);
                return;
            }
        }
        if (lower != local) {
            SetFailure(status, kFusionRoutePackBadPlacement, 0u, 0u,
                       expert);
            return;
        }
        if (owner == rank) ++observed_local_experts;
    }
    if (observed_local_experts != local_expert_count) {
        SetFailure(status, kFusionRoutePackBadPlacement);
        return;
    }

    __gm__ uint32_t *expert_starts = scratch;
    __gm__ uint32_t *seen = scratch + expert_count;
    uint32_t owner_counts[kFusionMaxWorkers] = {};
    uint32_t owner_base[kFusionMaxWorkers + 1u] = {};
    for (uint32_t expert = 0u; expert < expert_count; ++expert)
        ++owner_counts[expert_owner[expert]];
    for (uint32_t owner = 0u; owner < worker_count; ++owner)
        owner_base[owner + 1u] = owner_base[owner] + owner_counts[owner];
    uint32_t matches_by_destination[kFusionMaxWorkers] = {};
    uint32_t row_cursor = 0u;
    uint32_t assignment_cursor = 0u;
    for (uint32_t wave = 0u; wave < wave_count; ++wave) {
        const uint32_t token_begin = wave * tokens_per_wave;
        const uint32_t wave_tokens = wave < active_wave_count
            ? (collective_token_count - token_begin < tokens_per_wave
                   ? collective_token_count - token_begin : tokens_per_wave)
            : 0u;
        const uint32_t local_wave_tokens = wave < local_wave_count
            ? (token_count - token_begin < tokens_per_wave
                   ? token_count - token_begin : tokens_per_wave)
            : 0u;
        // Flatten (owner, local_expert) once, then compute both the expert
        // destination-row base and local group-list in O(E*W), not O(E^2).
        for (uint32_t expert = 0u; expert < expert_count; ++expert)
            seen[owner_base[expert_owner[expert]] +
                 expert_local_index[expert]] = expert;
        for (uint32_t destination = 0u;
             destination < worker_count; ++destination) {
            uint32_t cumulative = 0u;
            for (uint32_t local = 0u;
                 local < owner_counts[destination]; ++local) {
                const uint32_t expert =
                    seen[owner_base[destination] + local];
                uint32_t source_prefix = 0u;
                for (uint32_t source = 0u; source < rank; ++source) {
                    const uint64_t count_index =
                        (static_cast<uint64_t>(source) * wave_count + wave) *
                            count_stride + expert;
                    source_prefix += global_counts[count_index];
                }
                expert_starts[expert] = cumulative + source_prefix;
                cumulative += ExpertTotal(
                    global_counts, worker_count, wave_count,
                    expert_count, wave, expert);
                if (destination == rank)
                    group_lists[static_cast<uint64_t>(wave) *
                                    local_expert_count + local] = cumulative;
            }
        }
        for (uint32_t expert = 0u; expert < expert_count; ++expert)
            seen[expert] = 0u;

        FusionWaveDesc descriptor{};
        descriptor.generation = static_cast<uint64_t>(wave) + 1u;
        descriptor.token_begin = token_begin;
        descriptor.token_count = wave_tokens;
        descriptor.dispatch_row_begin = row_cursor;
        descriptor.assignment_begin = assignment_cursor;
        descriptor.slot = wave % slot_count;
        descriptor.activation_waves = activation_waves;

        for (uint32_t local_token = 0u; local_token < local_wave_tokens;
             ++local_token) {
            const uint32_t token = token_begin + local_token;
            for (uint32_t destination = 0u;
                 destination < worker_count; ++destination)
                matches_by_destination[destination] = 0u;
            for (uint32_t ordinal = 0u; ordinal < topk; ++ordinal) {
                const uint64_t index =
                    static_cast<uint64_t>(token) * topk + ordinal;
                const uint32_t expert = LoadExpert(topk_ids, id_type, index);
                if (expert >= expert_count) {
                    SetFailureWithScratch(
                        status, scratch, expert_count,
                        kFusionRoutePackBadExpert, token, ordinal, expert);
                    return;
                }
                ++matches_by_destination[expert_owner[expert]];
            }
            for (uint32_t destination = 0u;
                 destination < worker_count; ++destination) {
                const uint32_t matches =
                    matches_by_destination[destination];
                if (matches == 0u) continue;
                if (row_cursor >= dispatch_row_capacity ||
                    assignment_cursor > assignment_capacity ||
                    matches > assignment_capacity - assignment_cursor) {
                    SetFailureWithScratch(
                        status, scratch, expert_count,
                        kFusionRoutePackCapacity, token, 0u, destination);
                    return;
                }
                FusionDispatchRow row{};
                row.source_rank = rank;
                row.source_token = token;
                row.destination_rank = destination;
                row.assignment_begin = assignment_cursor;
                row.assignment_count = matches;
                row.wave = wave;
                row.payload_byte_offset =
                    static_cast<uint64_t>(token) * hidden * element_bytes;
                StoreDispatchRow(&dispatch_rows[row_cursor], row);
                for (uint32_t ordinal = 0u; ordinal < topk; ++ordinal) {
                    const uint64_t index =
                        static_cast<uint64_t>(token) * topk + ordinal;
                    const uint32_t expert =
                        LoadExpert(topk_ids, id_type, index);
                    if (expert_owner[expert] != destination) continue;
                    FusionExpertAssignment assignment{};
                    assignment.dispatch_row = row_cursor;
                    assignment.expert_id = expert;
                    assignment.local_expert = expert_local_index[expert];
                    assignment.route_ordinal = ordinal;
                    assignment.destination_token = token;
                    const float weight = topk_weights[index];
                    if (!IsFinite(weight)) {
                        SetFailureWithScratch(
                            status, scratch, expert_count,
                            kFusionRoutePackBadWeight, token, ordinal, expert);
                        return;
                    }
                    assignment.weight_bits = FloatBits(weight);
                    assignment.wave = wave;
                    assignment.destination_row =
                        expert_starts[expert] + seen[expert]++;
                    StoreAssignment(&assignments[assignment_cursor],
                                    assignment);
                    ++assignment_cursor;
                }
                ++row_cursor;
            }
        }
        descriptor.dispatch_row_count =
            row_cursor - descriptor.dispatch_row_begin;
        descriptor.assignment_count =
            assignment_cursor - descriptor.assignment_begin;
        StoreWave(&waves[wave], descriptor);
    }
    status->dispatch_row_count = row_cursor;
    status->assignment_count = assignment_cursor;
    status->wave_count = wave_count;
    AscendC::PipeBarrier<PIPE_ALL>();
    FlushRange(dispatch_rows,
               static_cast<uint64_t>(row_cursor) * sizeof(FusionDispatchRow));
    FlushRange(assignments,
               static_cast<uint64_t>(assignment_cursor) *
                   sizeof(FusionExpertAssignment));
    FlushRange(group_lists,
               static_cast<uint64_t>(wave_count) * local_expert_count *
                   sizeof(int64_t));
    FlushRange(waves,
               static_cast<uint64_t>(wave_count) * sizeof(FusionWaveDesc));
    FlushRange(scratch, ScratchBytes(expert_count));
    CleanInvalidateCacheLine(reinterpret_cast<__gm__ uint8_t *>(status));
}

// Phase 1: every AIV owns a contiguous token interval in every wave.  Private
// cache-line-aligned histograms avoid atomics and make the following prefix
// deterministic.  No protocol output is touched in this phase.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
inc_fusion_route_pack_analyze_kernel(
    __gm__ const void *topk_ids, uint32_t id_type,
    __gm__ const float *topk_weights,
    __gm__ const uint32_t *expert_owner, uint32_t worker_count,
    uint32_t token_count, uint32_t topk, uint32_t expert_count,
    uint32_t tokens_per_wave, uint32_t wave_capacity,
    uint32_t lane_count, __gm__ uint32_t *scratch)
{
    const uint32_t lane = static_cast<uint32_t>(AscendC::GetBlockIdx());
    if (lane >= lane_count || lane_count == 0u || scratch == nullptr) return;
    const uint64_t expert_stride = ParallelExpertStride(expert_count);
    const uint64_t state_offset = ParallelLaneStateOffset(
        wave_capacity, lane_count, expert_stride);
    const uint64_t error_offset = ParallelErrorOffset(
        wave_capacity, lane_count, expert_stride);
    __gm__ uint32_t *error = scratch + error_offset +
        static_cast<uint64_t>(lane) * kParallelLineWords;
    uint32_t first_error = kFusionRoutePackOk;
    uint32_t first_token = 0u;
    uint32_t first_ordinal = 0u;
    uint32_t first_expert = 0u;

    if ((token_count != 0u &&
         (topk_ids == nullptr || topk_weights == nullptr)) ||
        expert_owner == nullptr || worker_count == 0u ||
        worker_count > kFusionMaxWorkers || topk == 0u ||
        expert_count == 0u || tokens_per_wave == 0u ||
        wave_capacity == 0u ||
        (id_type != kFusionRouteInt32 && id_type != kFusionRouteInt64)) {
        first_error = kFusionRoutePackBadArgs;
    } else {
        AcquireRange(expert_owner,
                     static_cast<uint64_t>(expert_count) * sizeof(uint32_t));
    }

    for (uint32_t wave = 0u; wave < wave_capacity; ++wave) {
        __gm__ uint32_t *histogram = scratch +
            (static_cast<uint64_t>(wave) * lane_count + lane) *
                expert_stride;
        __gm__ uint32_t *state = scratch + state_offset +
            (static_cast<uint64_t>(wave) * lane_count + lane) *
                kParallelLineWords;
        for (uint64_t expert = 0u; expert < expert_stride; ++expert)
            histogram[expert] = 0u;
        for (uint32_t word = 0u; word < kParallelLineWords; ++word)
            state[word] = 0u;
        if (first_error != kFusionRoutePackOk) {
            AscendC::PipeBarrier<PIPE_ALL>();
            FlushRange(histogram, expert_stride * sizeof(uint32_t));
            CleanInvalidateCacheLine(
                reinterpret_cast<__gm__ uint8_t *>(state));
            continue;
        }
        const uint32_t wave_begin = wave * tokens_per_wave;
        const uint32_t wave_tokens = token_count > wave_begin
            ? MinU32(tokens_per_wave, token_count - wave_begin) : 0u;
        const uint32_t lane_chunk = wave_tokens == 0u
            ? 0u : CeilDiv(wave_tokens, lane_count);
        const uint32_t lane_local_begin = MinU32(
            wave_tokens, lane * lane_chunk);
        const uint32_t lane_local_end = MinU32(
            wave_tokens, lane_local_begin + lane_chunk);
        const uint32_t token_begin = wave_begin + lane_local_begin;
        const uint32_t lane_tokens = lane_local_end - lane_local_begin;
        state[2] = token_begin;
        state[3] = lane_tokens;
        if (lane_tokens != 0u) {
            const uint64_t first_index =
                static_cast<uint64_t>(token_begin) * topk;
            AcquireRange(
                reinterpret_cast<__gm__ const uint8_t *>(topk_ids) +
                    first_index *
                        (id_type == kFusionRouteInt64
                             ? sizeof(int64_t) : sizeof(uint32_t)),
                static_cast<uint64_t>(lane_tokens) * topk *
                    (id_type == kFusionRouteInt64
                         ? sizeof(int64_t) : sizeof(uint32_t)));
            AcquireRange(topk_weights + first_index,
                         static_cast<uint64_t>(lane_tokens) * topk *
                             sizeof(float));
        }
        uint32_t rows = 0u;
        for (uint32_t token = token_begin;
             token < token_begin + lane_tokens; ++token) {
            uint64_t destination_bits = 0u;
            for (uint32_t ordinal = 0u; ordinal < topk; ++ordinal) {
                const uint64_t index =
                    static_cast<uint64_t>(token) * topk + ordinal;
                const uint32_t expert = LoadExpert(topk_ids, id_type, index);
                if (expert >= expert_count) {
                    if (first_error == kFusionRoutePackOk) {
                        first_error = kFusionRoutePackBadExpert;
                        first_token = token;
                        first_ordinal = ordinal;
                        first_expert = expert;
                    }
                    continue;
                }
                const uint32_t owner = expert_owner[expert];
                if (owner >= worker_count) {
                    if (first_error == kFusionRoutePackOk) {
                        first_error = kFusionRoutePackBadPlacement;
                        first_token = token;
                        first_ordinal = ordinal;
                        first_expert = expert;
                    }
                    continue;
                }
                const float weight = topk_weights[index];
                if (!IsFinite(weight) && first_error == kFusionRoutePackOk) {
                    first_error = kFusionRoutePackBadWeight;
                    first_token = token;
                    first_ordinal = ordinal;
                    first_expert = expert;
                }
                ++histogram[expert];
                destination_bits |= uint64_t{1u} << owner;
            }
            rows += PopcountDestinations(destination_bits);
        }
        state[0] = rows;
        state[1] = lane_tokens * topk;
        AscendC::PipeBarrier<PIPE_ALL>();
        FlushRange(histogram, expert_stride * sizeof(uint32_t));
        CleanInvalidateCacheLine(
            reinterpret_cast<__gm__ uint8_t *>(state));
    }
    error[0] = first_error;
    error[1] = first_token;
    error[2] = first_ordinal;
    error[3] = first_expert;
    for (uint32_t word = 4u; word < kParallelLineWords; ++word)
        error[word] = 0u;
    AscendC::PipeBarrier<PIPE_ALL>();
    CleanInvalidateCacheLine(reinterpret_cast<__gm__ uint8_t *>(error));
}

// Phase 2: one AIV validates placement/metadata, turns private counts into
// exclusive prefixes and publishes wave/group metadata.  Separate kernel
// launches provide the only global barriers; no cross-AIV spin protocol is
// added to the route hot path.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
inc_fusion_route_pack_prefix_kernel(
    __gm__ const void *topk_ids, uint32_t id_type,
    __gm__ const float *topk_weights,
    __gm__ const uint32_t *global_counts,
    __gm__ const uint32_t *expert_owner,
    __gm__ const uint32_t *expert_local_index,
    uint32_t worker_count, uint32_t rank, uint32_t token_count,
    uint32_t topk, uint32_t hidden, uint32_t element_bytes,
    uint32_t expert_count, uint32_t local_expert_count,
    uint32_t tokens_per_wave, uint32_t slot_count,
    uint32_t activation_waves,
    __gm__ FusionDispatchRow *dispatch_rows,
    uint32_t dispatch_row_capacity,
    __gm__ FusionExpertAssignment *assignments,
    uint32_t assignment_capacity, __gm__ int64_t *group_lists,
    __gm__ FusionWaveDesc *waves, uint32_t wave_capacity,
    uint32_t lane_count, __gm__ uint32_t *scratch,
    __gm__ FusionRoutePackStatus *status)
{
    if (AscendC::GetBlockIdx() != 0u || status == nullptr) return;
    ResetStatus(status);
    if ((token_count != 0u &&
         (topk_ids == nullptr || topk_weights == nullptr)) ||
        global_counts == nullptr || expert_owner == nullptr ||
        expert_local_index == nullptr || dispatch_rows == nullptr ||
        assignments == nullptr || group_lists == nullptr || waves == nullptr ||
        scratch == nullptr || worker_count == 0u || rank >= worker_count ||
        worker_count > kFusionMaxWorkers || topk == 0u || hidden == 0u ||
        element_bytes == 0u || expert_count == 0u ||
        tokens_per_wave == 0u || slot_count < kFusionMinSlots ||
        activation_waves == 0u || wave_capacity == 0u || lane_count == 0u ||
        lane_count > kFusionMaxAiv ||
        (id_type != kFusionRouteInt32 && id_type != kFusionRouteInt64)) {
        SetFailure(status, kFusionRoutePackBadArgs);
        return;
    }
    const uint64_t expert_stride = ParallelExpertStride(expert_count);
    const uint64_t state_offset = ParallelLaneStateOffset(
        wave_capacity, lane_count, expert_stride);
    const uint64_t expert_base_offset = ParallelExpertBaseOffset(
        wave_capacity, lane_count, expert_stride);
    const uint64_t error_offset = ParallelErrorOffset(
        wave_capacity, lane_count, expert_stride);
    const uint64_t scratch_words = ParallelScratchWords(
        wave_capacity, lane_count, expert_stride);
    const uint32_t count_stride = expert_count + 1u;
    AcquireRange(scratch, scratch_words * sizeof(uint32_t));
    AcquireRange(global_counts,
        static_cast<uint64_t>(worker_count) * wave_capacity * count_stride *
            sizeof(uint32_t));
    AcquireRange(expert_owner,
                 static_cast<uint64_t>(expert_count) * sizeof(uint32_t));
    AcquireRange(expert_local_index,
                 static_cast<uint64_t>(expert_count) * sizeof(uint32_t));

    uint32_t collective_token_count = 0u;
    for (uint32_t source = 0u; source < worker_count; ++source) {
        const uint32_t active = global_counts[
            static_cast<uint64_t>(source) * wave_capacity * count_stride +
            expert_count];
        if (active > collective_token_count) collective_token_count = active;
    }
    if (global_counts[
            static_cast<uint64_t>(rank) * wave_capacity * count_stride +
            expert_count] != token_count) {
        SetFailure(status, kFusionRoutePackBadArgs);
        return;
    }
    const uint32_t local_wave_count = token_count == 0u
        ? 0u : CeilDiv(token_count, tokens_per_wave);
    const uint32_t active_wave_count = collective_token_count == 0u
        ? 0u : CeilDiv(collective_token_count, tokens_per_wave);
    if (local_wave_count > wave_capacity ||
        active_wave_count > wave_capacity) {
        SetFailure(status, kFusionRoutePackCapacity);
        return;
    }

    uint32_t owner_counts[kFusionMaxWorkers] = {};
    uint32_t owner_base[kFusionMaxWorkers + 1u] = {};
    uint32_t observed_local_experts = 0u;
    bool placement_owner_local_ordered = true;
    uint32_t previous_owner = 0u;
    for (uint32_t expert = 0u; expert < expert_count; ++expert) {
        const uint32_t owner = expert_owner[expert];
        const uint32_t local = expert_local_index[expert];
        if (owner >= worker_count) {
            SetFailure(status, kFusionRoutePackBadPlacement, 0u, 0u, expert);
            return;
        }
        // The prepared vLLM/Megatron placement is normally stored in
        // [owner][local-expert] order.  Detect that common, portable layout
        // in O(E); it lets the prefix phase avoid two separate O(E^2)
        // placement scans on every MoE layer.  Arbitrary expert permutations
        // remain supported and take the exhaustive validation fallback.
        if ((expert != 0u && owner < previous_owner) ||
            local != owner_counts[owner])
            placement_owner_local_ordered = false;
        previous_owner = owner;
        ++owner_counts[owner];
        if (owner == rank) ++observed_local_experts;
    }
    if (!placement_owner_local_ordered) {
      for (uint32_t expert = 0u; expert < expert_count; ++expert) {
        const uint32_t owner = expert_owner[expert];
        const uint32_t local = expert_local_index[expert];
        uint32_t lower = 0u;
        for (uint32_t other = 0u; other < expert_count; ++other) {
            if (expert_owner[other] != owner) continue;
            const uint32_t other_local = expert_local_index[other];
            if (other_local < local) ++lower;
            if (other != expert && other_local == local) {
                SetFailure(status, kFusionRoutePackBadPlacement,
                           0u, 0u, expert);
                return;
            }
        }
        if (lower != local) {
            SetFailure(status, kFusionRoutePackBadPlacement, 0u, 0u, expert);
            return;
        }
      }
    }
    if (observed_local_experts != local_expert_count) {
        SetFailure(status, kFusionRoutePackBadPlacement);
        return;
    }
    for (uint32_t owner = 0u; owner < worker_count; ++owner)
        owner_base[owner + 1u] = owner_base[owner] + owner_counts[owner];

    // Placement errors have scalar-path priority.  Route errors are selected
    // lexicographically so the reported token/ordinal is independent of how
    // many AIVs were used.
    uint32_t route_error = kFusionRoutePackOk;
    uint32_t route_token = UINT32_MAX;
    uint32_t route_ordinal = UINT32_MAX;
    uint32_t route_expert = 0u;
    for (uint32_t lane = 0u; lane < lane_count; ++lane) {
        __gm__ uint32_t *error = scratch + error_offset +
            static_cast<uint64_t>(lane) * kParallelLineWords;
        if (error[0] == kFusionRoutePackOk) continue;
        if (error[0] == kFusionRoutePackBadArgs) {
            SetFailure(status, kFusionRoutePackBadArgs);
            return;
        }
        if (route_error == kFusionRoutePackOk ||
            error[1] < route_token ||
            (error[1] == route_token && error[2] < route_ordinal)) {
            route_error = error[0];
            route_token = error[1];
            route_ordinal = error[2];
            route_expert = error[3];
        }
    }
    if (route_error != kFusionRoutePackOk) {
        SetFailure(status, route_error, route_token,
                   route_ordinal, route_expert);
        return;
    }

    __gm__ uint32_t *expert_bases = scratch + expert_base_offset;
    uint32_t row_cursor = 0u;
    uint32_t assignment_cursor = 0u;
    for (uint32_t wave = 0u; wave < wave_capacity; ++wave) {
        uint32_t wave_rows = 0u;
        uint32_t wave_assignments = 0u;
        for (uint32_t lane = 0u; lane < lane_count; ++lane) {
            __gm__ uint32_t *histogram = scratch +
                (static_cast<uint64_t>(wave) * lane_count + lane) *
                    expert_stride;
            __gm__ uint32_t *state = scratch + state_offset +
                (static_cast<uint64_t>(wave) * lane_count + lane) *
                    kParallelLineWords;
            const uint32_t rows = state[0];
            const uint32_t count = state[1];
            state[0] = row_cursor + wave_rows;
            state[1] = assignment_cursor + wave_assignments;
            wave_rows += rows;
            wave_assignments += count;
        }
        for (uint32_t expert = 0u; expert < expert_count; ++expert) {
            uint32_t running = 0u;
            for (uint32_t lane = 0u; lane < lane_count; ++lane) {
                __gm__ uint32_t *histogram = scratch +
                    (static_cast<uint64_t>(wave) * lane_count + lane) *
                        expert_stride;
                const uint32_t one_lane = histogram[expert];
                histogram[expert] = running;
                running += one_lane;
            }
            const uint32_t expected = global_counts[
                (static_cast<uint64_t>(rank) * wave_capacity + wave) *
                    count_stride + expert];
            if (running != expected) {
                SetFailure(status, kFusionRoutePackBadArgs,
                           wave, 0u, expert);
                return;
            }
        }
        if (row_cursor > dispatch_row_capacity ||
            wave_rows > dispatch_row_capacity - row_cursor ||
            assignment_cursor > assignment_capacity ||
            wave_assignments > assignment_capacity - assignment_cursor) {
            SetFailure(status, kFusionRoutePackCapacity, wave);
            return;
        }

        for (uint32_t destination = 0u;
             destination < worker_count; ++destination) {
            uint32_t cumulative = 0u;
            for (uint32_t local = 0u;
                 local < owner_counts[destination]; ++local) {
                uint32_t expert = placement_owner_local_ordered
                    ? owner_base[destination] + local : expert_count;
                if (!placement_owner_local_ordered) {
                    for (uint32_t candidate = 0u;
                         candidate < expert_count; ++candidate) {
                        if (expert_owner[candidate] == destination &&
                            expert_local_index[candidate] == local) {
                            expert = candidate;
                            break;
                        }
                    }
                }
                if (expert == expert_count) {
                    SetFailure(status, kFusionRoutePackBadPlacement,
                               0u, 0u, local);
                    return;
                }
                uint32_t source_prefix = 0u;
                uint32_t total = 0u;
                for (uint32_t source = 0u; source < worker_count; ++source) {
                    const uint32_t count = global_counts[
                        (static_cast<uint64_t>(source) * wave_capacity + wave) *
                            count_stride + expert];
                    if (source < rank) source_prefix += count;
                    total += count;
                }
                expert_bases[static_cast<uint64_t>(wave) * expert_stride +
                             expert] = cumulative + source_prefix;
                cumulative += total;
                if (destination == rank)
                    group_lists[static_cast<uint64_t>(wave) *
                                    local_expert_count + local] = cumulative;
            }
        }

        FusionWaveDesc descriptor{};
        descriptor.generation = static_cast<uint64_t>(wave) + 1u;
        descriptor.token_begin = wave * tokens_per_wave;
        descriptor.token_count = wave < active_wave_count
            ? MinU32(tokens_per_wave,
                     collective_token_count - descriptor.token_begin) : 0u;
        descriptor.dispatch_row_begin = row_cursor;
        descriptor.dispatch_row_count = wave_rows;
        descriptor.assignment_begin = assignment_cursor;
        descriptor.assignment_count = wave_assignments;
        descriptor.slot = wave % slot_count;
        descriptor.activation_waves = activation_waves;
        StoreWave(&waves[wave], descriptor);
        row_cursor += wave_rows;
        assignment_cursor += wave_assignments;
    }
    status->dispatch_row_count = row_cursor;
    status->assignment_count = assignment_cursor;
    status->wave_count = wave_capacity;
    AscendC::PipeBarrier<PIPE_ALL>();
    FlushRange(scratch, scratch_words * sizeof(uint32_t));
    FlushRange(group_lists,
               static_cast<uint64_t>(wave_capacity) * local_expert_count *
                   sizeof(int64_t));
    FlushRange(waves,
               static_cast<uint64_t>(wave_capacity) * sizeof(FusionWaveDesc));
    CleanInvalidateCacheLine(reinterpret_cast<__gm__ uint8_t *>(status));
}

// Phase 3: lanes emit their already-prefixed contiguous ranges.  Records are
// buffered in UB and copied with MTE3, avoiding 32-byte scalar-cache false
// sharing at lane boundaries while preserving the compact public ABI.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
inc_fusion_route_pack_emit_kernel(
    __gm__ const void *topk_ids, uint32_t id_type,
    __gm__ const float *topk_weights,
    __gm__ const uint32_t *expert_owner,
    __gm__ const uint32_t *expert_local_index,
    uint32_t worker_count, uint32_t rank, uint32_t topk, uint32_t hidden,
    uint32_t element_bytes, uint32_t expert_count,
    uint32_t wave_capacity, uint32_t lane_count,
    __gm__ FusionDispatchRow *dispatch_rows,
    __gm__ FusionExpertAssignment *assignments,
    __gm__ uint32_t *scratch, __gm__ FusionRoutePackStatus *status)
{
    const uint32_t lane = static_cast<uint32_t>(AscendC::GetBlockIdx());
    if (lane >= lane_count || scratch == nullptr || status == nullptr) return;
    AcquireRange(status, sizeof(FusionRoutePackStatus));
    if (status->error != kFusionRoutePackOk) return;
    const uint64_t expert_stride = ParallelExpertStride(expert_count);
    const uint64_t state_offset = ParallelLaneStateOffset(
        wave_capacity, lane_count, expert_stride);
    const uint64_t expert_base_offset = ParallelExpertBaseOffset(
        wave_capacity, lane_count, expert_stride);
    AcquireRange(expert_owner,
                 static_cast<uint64_t>(expert_count) * sizeof(uint32_t));
    AcquireRange(expert_local_index,
                 static_cast<uint64_t>(expert_count) * sizeof(uint32_t));

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> row_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> assignment_buffer;
    pipe.InitBuffer(row_buffer,
        kParallelBufferedRecords * sizeof(FusionDispatchRow));
    pipe.InitBuffer(assignment_buffer,
        kParallelBufferedRecords * sizeof(FusionExpertAssignment));
    AscendC::LocalTensor<uint32_t> local_rows = row_buffer.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> local_assignments =
        assignment_buffer.Get<uint32_t>();

    for (uint32_t wave = 0u; wave < wave_capacity; ++wave) {
        __gm__ uint32_t *histogram = scratch +
            (static_cast<uint64_t>(wave) * lane_count + lane) *
                expert_stride;
        __gm__ uint32_t *state = scratch + state_offset +
            (static_cast<uint64_t>(wave) * lane_count + lane) *
                kParallelLineWords;
        __gm__ uint32_t *expert_bases = scratch + expert_base_offset +
            static_cast<uint64_t>(wave) * expert_stride;
        AcquireRange(histogram, expert_stride * sizeof(uint32_t));
        AcquireRange(state, kFusionCacheLineBytes);
        AcquireRange(expert_bases, expert_stride * sizeof(uint32_t));
        const uint32_t token_begin = state[2];
        const uint32_t lane_tokens = state[3];
        uint32_t row_cursor = state[0];
        uint32_t assignment_cursor = state[1];
        const uint32_t row_output_begin = row_cursor;
        const uint32_t assignment_output_begin = assignment_cursor;
        uint32_t row_buffer_begin = row_cursor;
        uint32_t assignment_buffer_begin = assignment_cursor;
        uint32_t buffered_rows = 0u;
        uint32_t buffered_assignments = 0u;
        if (lane_tokens != 0u) {
            const uint64_t first_index =
                static_cast<uint64_t>(token_begin) * topk;
            AcquireRange(
                reinterpret_cast<__gm__ const uint8_t *>(topk_ids) +
                    first_index *
                        (id_type == kFusionRouteInt64
                             ? sizeof(int64_t) : sizeof(uint32_t)),
                static_cast<uint64_t>(lane_tokens) * topk *
                    (id_type == kFusionRouteInt64
                         ? sizeof(int64_t) : sizeof(uint32_t)));
            AcquireRange(topk_weights + first_index,
                         static_cast<uint64_t>(lane_tokens) * topk *
                             sizeof(float));
        }

        for (uint32_t token = token_begin;
             token < token_begin + lane_tokens; ++token) {
            uint32_t matches_by_destination[kFusionMaxWorkers] = {};
            for (uint32_t ordinal = 0u; ordinal < topk; ++ordinal) {
                const uint64_t index =
                    static_cast<uint64_t>(token) * topk + ordinal;
                const uint32_t expert = LoadExpert(topk_ids, id_type, index);
                ++matches_by_destination[expert_owner[expert]];
            }
            for (uint32_t destination = 0u;
                 destination < worker_count; ++destination) {
                const uint32_t matches = matches_by_destination[destination];
                if (matches == 0u) continue;
                if (buffered_rows == kParallelBufferedRecords) {
                    FlushRecordBuffer(local_rows, buffered_rows,
                        reinterpret_cast<__gm__ uint32_t *>(dispatch_rows),
                        row_buffer_begin);
                    row_buffer_begin += buffered_rows;
                    buffered_rows = 0u;
                }
                BufferDispatchRow(local_rows, buffered_rows++, rank, token,
                    destination, assignment_cursor, matches, wave,
                    static_cast<uint64_t>(token) * hidden * element_bytes);
                for (uint32_t ordinal = 0u; ordinal < topk; ++ordinal) {
                    const uint64_t index =
                        static_cast<uint64_t>(token) * topk + ordinal;
                    const uint32_t expert =
                        LoadExpert(topk_ids, id_type, index);
                    if (expert_owner[expert] != destination) continue;
                    if (buffered_assignments == kParallelBufferedRecords) {
                        FlushRecordBuffer(local_assignments,
                            buffered_assignments,
                            reinterpret_cast<__gm__ uint32_t *>(assignments),
                            assignment_buffer_begin);
                        assignment_buffer_begin += buffered_assignments;
                        buffered_assignments = 0u;
                    }
                    const uint32_t destination_row =
                        expert_bases[expert] + histogram[expert]++;
                    BufferAssignment(local_assignments,
                        buffered_assignments++, row_cursor, expert,
                        expert_local_index[expert], ordinal, token,
                        FloatBits(topk_weights[index]), wave, destination_row);
                    ++assignment_cursor;
                }
                ++row_cursor;
            }
        }
        FlushRecordBuffer(local_rows, buffered_rows,
            reinterpret_cast<__gm__ uint32_t *>(dispatch_rows),
            row_buffer_begin);
        FlushRecordBuffer(local_assignments, buffered_assignments,
            reinterpret_cast<__gm__ uint32_t *>(assignments),
            assignment_buffer_begin);
        AscendC::PipeBarrier<PIPE_ALL>();
        FlushRange(dispatch_rows + row_output_begin,
            static_cast<uint64_t>(row_cursor - row_output_begin) *
                sizeof(FusionDispatchRow));
        FlushRange(assignments + assignment_output_begin,
            static_cast<uint64_t>(assignment_cursor -
                                  assignment_output_begin) *
                sizeof(FusionExpertAssignment));
        FlushRange(histogram, expert_stride * sizeof(uint32_t));
    }
}

extern "C" void launch_inc_fusion_route_count_kernel(
    const void *topk_ids, uint32_t id_type,
    uint32_t token_count, uint32_t topk, uint32_t expert_count,
    uint32_t tokens_per_wave, uint32_t wave_capacity,
    uint32_t *local_counts,
    FusionRoutePackStatus *status, void *stream)
{
    inc_fusion_route_count_kernel<<<1, nullptr, stream>>>(
        topk_ids, id_type, token_count, topk, expert_count,
        tokens_per_wave, wave_capacity, local_counts, status);
}

extern "C" void launch_inc_fusion_route_pack_kernel(
    const void *topk_ids, uint32_t id_type, const float *topk_weights,
    const uint32_t *global_counts, const uint32_t *expert_owner,
    const uint32_t *expert_local_index, uint32_t worker_count,
    uint32_t rank, uint32_t token_count, uint32_t topk,
    uint32_t hidden, uint32_t element_bytes, uint32_t expert_count,
    uint32_t local_expert_count, uint32_t tokens_per_wave,
    uint32_t slot_count, uint32_t activation_waves,
    FusionDispatchRow *dispatch_rows, uint32_t dispatch_row_capacity,
    FusionExpertAssignment *assignments, uint32_t assignment_capacity,
    int64_t *group_lists, FusionWaveDesc *waves, uint32_t wave_capacity,
    uint32_t *scratch, FusionRoutePackStatus *status, void *stream)
{
    inc_fusion_route_pack_kernel<<<1, nullptr, stream>>>(
        topk_ids, id_type, topk_weights, global_counts, expert_owner,
        expert_local_index, worker_count, rank, token_count,
        topk, hidden, element_bytes, expert_count,
        local_expert_count,
        tokens_per_wave, slot_count, activation_waves, dispatch_rows,
        dispatch_row_capacity, assignments, assignment_capacity,
        group_lists, waves, wave_capacity, scratch, status);
}

extern "C" void launch_inc_fusion_route_pack_parallel_kernel(
    const void *topk_ids, uint32_t id_type, const float *topk_weights,
    const uint32_t *global_counts, const uint32_t *expert_owner,
    const uint32_t *expert_local_index, uint32_t worker_count,
    uint32_t rank, uint32_t token_count, uint32_t topk,
    uint32_t hidden, uint32_t element_bytes, uint32_t expert_count,
    uint32_t local_expert_count, uint32_t tokens_per_wave,
    uint32_t slot_count, uint32_t activation_waves,
    FusionDispatchRow *dispatch_rows, uint32_t dispatch_row_capacity,
    FusionExpertAssignment *assignments, uint32_t assignment_capacity,
    int64_t *group_lists, FusionWaveDesc *waves, uint32_t wave_capacity,
    uint32_t lane_count, uint32_t *parallel_scratch,
    FusionRoutePackStatus *status, void *stream)
{
    if (lane_count == 0u || lane_count > kFusionMaxAiv) {
        // Preserve asynchronous API semantics for valid prepared calls.  Bad
        // lane counts are routed through the prefix kernel, which publishes a
        // structured BadArgs status instead of launching an invalid grid.
        inc_fusion_route_pack_prefix_kernel<<<1, nullptr, stream>>>(
            topk_ids, id_type, topk_weights, global_counts, expert_owner,
            expert_local_index, worker_count, rank, token_count, topk,
            hidden, element_bytes, expert_count, local_expert_count,
            tokens_per_wave, slot_count, activation_waves, dispatch_rows,
            dispatch_row_capacity, assignments, assignment_capacity,
            group_lists, waves, wave_capacity, lane_count, parallel_scratch,
            status);
        return;
    }
    inc_fusion_route_pack_analyze_kernel<<<lane_count, nullptr, stream>>>(
        topk_ids, id_type, topk_weights, expert_owner, worker_count,
        token_count, topk, expert_count, tokens_per_wave, wave_capacity,
        lane_count, parallel_scratch);
    inc_fusion_route_pack_prefix_kernel<<<1, nullptr, stream>>>(
        topk_ids, id_type, topk_weights, global_counts, expert_owner,
        expert_local_index, worker_count, rank, token_count, topk,
        hidden, element_bytes, expert_count, local_expert_count,
        tokens_per_wave, slot_count, activation_waves, dispatch_rows,
        dispatch_row_capacity, assignments, assignment_capacity,
        group_lists, waves, wave_capacity, lane_count, parallel_scratch,
        status);
    inc_fusion_route_pack_emit_kernel<<<lane_count, nullptr, stream>>>(
        topk_ids, id_type, topk_weights, expert_owner, expert_local_index,
        worker_count, rank, topk, hidden, element_bytes, expert_count,
        wave_capacity, lane_count, dispatch_rows, assignments,
        parallel_scratch, status);
}
