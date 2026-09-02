#include "kernel_operator.h"
#include "shmem.h"

#include "inc_dc_pull_combine_abi.h"
#include "inc_dc_pull_combine_device_e2e_abi.h"
#include "inc_dc_vector_reduce_aicore.h"

using namespace inc::dc::pull_combine;

namespace {

constexpr uint32_t kRmaAlignment = 64u;
constexpr uint32_t kPullLanesPerSource = 2u;
constexpr uint64_t kReadySpinLimit = 1000000000ull;
constexpr uint32_t kVecPingMte2V = 0u;
constexpr uint32_t kVecPingVMte2 = 1u;
constexpr uint32_t kVecPongMte2V = 2u;
constexpr uint32_t kVecPongVMte2 = 3u;
constexpr uint32_t kVecMte3V = 4u;
constexpr uint32_t kVecVMte3 = 5u;

__aicore__ inline void GetExact(__gm__ uint8_t *destination,
                                __gm__ uint8_t *source, uint32_t bytes,
                                int32_t pe)
{
    const uint32_t bulk = bytes / kRmaAlignment * kRmaAlignment;
    if (bulk != 0u) aclshmem_getmem(destination, source, bulk, pe);
    if (bulk != bytes)
        aclshmem_getmem(destination + bulk, source + bulk, bytes - bulk, pe);
}

__aicore__ inline void CopyFp32GmToUb(
    __gm__ float *source, __ubuf__ uint8_t *destination, uint32_t elements,
    uint32_t ready_for_mte2, uint32_t ready_for_vector)
{
    const uint32_t bytes = elements * sizeof(float);
    AscendC::LocalTensor<uint8_t> ub;
    AscendC::GlobalTensor<uint8_t> gm;
    AscendC::DataCopyExtParams params(1u, bytes, 0u, 0u, 0u);
    AscendC::DataCopyPadExtParams<uint8_t> padding;
    ub.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub.address_.bufferAddr = reinterpret_cast<uint64_t>(destination);
    ub.address_.dataLen = static_cast<uint32_t>(IncVecUbAlignUp(bytes, 32));
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(source));
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ready_for_mte2);
    AscendC::DataCopyPad(ub, gm, params, padding);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ready_for_vector);
}

__aicore__ inline void PutFp32UbRemote(
    __ubuf__ uint8_t *source, __gm__ float *destination, uint32_t elements,
    int32_t destination_pe)
{
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kVecVMte3);
    aclshmemx_mte_put_nbi(
        destination, reinterpret_cast<__ubuf__ float *>(source), elements,
        destination_pe, 0u);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kVecMte3V);
}

__aicore__ inline bool WaitReadyChunk(
    __gm__ uint8_t *ready_flags, uint32_t worker_count,
    uint32_t chunk_count, uint32_t chunk, uint64_t generation,
    __gm__ uint32_t *status)
{
    for (uint32_t worker = 0u; worker < worker_count; ++worker) {
        __gm__ uint64_t *ready = reinterpret_cast<__gm__ uint64_t *>(
            ready_flags +
            (static_cast<uint64_t>(worker) * chunk_count + chunk) *
                kDevicePipelineReadyStride);
        bool observed = false;
        for (uint64_t spin = 0u; spin < kReadySpinLimit; ++spin) {
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(ready));
            if (*ready == generation) {
                observed = true;
                break;
            }
        }
        if (!observed) {
            *status = kDeviceE2eStatusReadyTimeout;
            AscendC::PipeBarrier<PIPE_ALL>();
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(status));
            return false;
        }
    }
    // Producer GET completion precedes ready publication.  Acquire the MTE
    // view before consuming the just-published staging chunk.
    AscendC::PipeBarrier<PIPE_ALL>();
    return true;
}

__aicore__ inline uint32_t PipelineChunkFromSequence(
    uint32_t sequence, uint32_t chunk_count, uint32_t worker_count)
{
    // Visit the first chunk of every output-owner region before visiting the
    // second chunk of any region.  This removes the otherwise large startup
    // bubble for later owners while retaining exactly one producer per chunk.
    const uint32_t chunks_per_owner =
        (chunk_count + worker_count - 1u) / worker_count;
    return sequence % worker_count * chunks_per_owner +
           sequence / worker_count;
}

__aicore__ inline bool ReduceOwnerFp32Vector(
    __gm__ float *staging, __gm__ float *output, uint64_t padded_elements,
    uint32_t worker_count, uint64_t logical_begin,
    uint64_t local_begin, uint64_t local_end, int32_t owner_pe,
    bool pipeline, __gm__ uint8_t *ready_flags, uint32_t chunk_count,
    uint64_t generation, __gm__ uint32_t *status)
{
    if (local_begin >= local_end) return true;
    // The 910B wide-vector path is qualified at 1536 elements.  Ping, pong
    // and accumulator consume 18 KiB of the 24-KiB AIV UB.
    constexpr uint32_t kTileElements = 1536u;
    constexpr uint32_t kTileBytes = kTileElements * sizeof(float);
    static_assert(kTileBytes * 3u <= INC_VEC_UB_BUDGET_BYTES,
                  "ping/pong FP32 tile exceeds AIV UB budget");
    __ubuf__ uint8_t *ping_ub = reinterpret_cast<__ubuf__ uint8_t *>(0);
    __ubuf__ uint8_t *pong_ub = ping_ub + kTileBytes;
    __ubuf__ uint8_t *acc_ub = pong_ub + kTileBytes;

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kVecPingVMte2);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kVecPongVMte2);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kVecMte3V);
    uint32_t next_chunk_to_wait = static_cast<uint32_t>(
        (logical_begin + local_begin) /
        (kDevicePipelineChunkBytes / sizeof(float)));
    for (uint64_t local = local_begin; local < local_end;
         local += kTileElements) {
        const uint32_t count = static_cast<uint32_t>(
            local_end - local < kTileElements
                ? local_end - local
                : kTileElements);
        const uint64_t logical = logical_begin + local;
        if (pipeline) {
            const uint32_t last_chunk = static_cast<uint32_t>(
                (logical + count - 1u) /
                (kDevicePipelineChunkBytes / sizeof(float)));
            while (next_chunk_to_wait <= last_chunk) {
                if (!WaitReadyChunk(ready_flags, worker_count, chunk_count,
                                    next_chunk_to_wait, generation, status))
                    return false;
                ++next_chunk_to_wait;
            }
        }
        // The accumulator UB cannot be reused until the preceding MTE3 store
        // has completed.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kVecMte3V);
        AscendC::LocalTensor<float> acc =
            IncVecBindFloatUb(acc_ub, count * sizeof(float));
        AscendC::Duplicate(acc, 0.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        CopyFp32GmToUb(staging + logical, ping_ub, count,
                       kVecPingVMte2, kVecPingMte2V);
        for (uint32_t worker = 0u; worker < worker_count; ++worker) {
            const bool ping = (worker & 1u) == 0u;
            __ubuf__ uint8_t *current = ping ? ping_ub : pong_ub;
            const uint32_t current_mte2v =
                ping ? kVecPingMte2V : kVecPongMte2V;
            const uint32_t current_vmte2 =
                ping ? kVecPingVMte2 : kVecPongVMte2;
            if (worker + 1u < worker_count) {
                const uint32_t next = worker + 1u;
                const bool next_ping = (next & 1u) == 0u;
                CopyFp32GmToUb(
                    staging + static_cast<uint64_t>(next) *
                                  padded_elements + logical,
                    next_ping ? ping_ub : pong_ub, count,
                    next_ping ? kVecPingVMte2 : kVecPongVMte2,
                    next_ping ? kVecPingMte2V : kVecPongMte2V);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(current_mte2v);
            AscendC::LocalTensor<float> temp =
                IncVecBindFloatUb(current, count * sizeof(float));
            AscendC::Add(acc, acc, temp, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(current_vmte2);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kVecVMte3);
        PutFp32UbRemote(acc_ub, output + local, count, owner_pe);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kVecMte3V);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kVecPingVMte2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kVecPongVMte2);
    return true;
}

__aicore__ inline bool DescriptorValid(
    __gm__ const CombineReadyDescriptor *descriptor, uint32_t worker,
    uint32_t worker_count, uint32_t elements, uint64_t generation,
    uint32_t wave, uint64_t digest)
{
    if (descriptor->magic != kPullCombineMagic ||
        descriptor->abi_version != kPullCombineAbiVersion ||
        descriptor->struct_bytes != sizeof(CombineReadyDescriptor) ||
        descriptor->generation != generation ||
        descriptor->sequence != 1u || descriptor->wave != wave ||
        descriptor->source_rank != worker ||
        descriptor->combine_row_begin !=
            static_cast<uint64_t>(worker) * elements ||
        descriptor->row_count != elements ||
        descriptor->partial_dtype !=
            static_cast<uint32_t>(PartialDType::FP32) ||
        descriptor->source_region_id == 0u ||
        descriptor->source_offset != 0u ||
        descriptor->payload_bytes !=
            static_cast<uint64_t>(elements) * sizeof(float) ||
        descriptor->semantic_digest != digest || descriptor->flags != 0u ||
        descriptor->reserved0 != 0u) {
        return false;
    }
    for (uint32_t i = 0u; i < 5u; ++i) {
        if (descriptor->reserved[i] != 0u) return false;
    }
    return worker < worker_count;
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void inc_dc_pull_combine_device_e2e_kernel(
    GM_ADDR symmetric_partials, GM_ADDR inc_staging, GM_ADDR reduced_output,
    GM_ADDR descriptor_mailbox, GM_ADDR ack_mailbox, GM_ADDR status_line,
    GM_ADDR ready_flags, uint64_t ffts_addr, uint32_t elements,
    uint32_t worker_count,
    uint32_t lanes_per_worker, int32_t inc_pe, uint64_t generation,
    uint32_t wave, uint64_t digest)
{
    // SyncAll requires the per-process FFTS workspace registered by SHMEM.
    // Passing the address explicitly keeps this qualification kernel aligned
    // with the production fusion and MegaMoE device entry points.
    shmemx_set_ffts_config(ffts_addr);
    const uint32_t block = AscendC::GetBlockIdx();
    const int32_t pe = aclshmem_my_pe();
    constexpr uint64_t kFloatsPerCacheLine = 64u / sizeof(float);
    const uint64_t padded_elements =
        (static_cast<uint64_t>(elements) + kFloatsPerCacheLine - 1u) /
        kFloatsPerCacheLine * kFloatsPerCacheLine;
    constexpr uint64_t kPipelineChunkElements =
        kDevicePipelineChunkBytes / sizeof(float);
    const uint32_t chunk_count = static_cast<uint32_t>(
        (padded_elements + kPipelineChunkElements - 1u) /
        kPipelineChunkElements);
    __gm__ DeviceE2eTimeline *timeline =
        reinterpret_cast<__gm__ DeviceE2eTimeline *>(status_line);
    __gm__ uint32_t *status = &timeline->status;

    if (pe == inc_pe && block == 0u) {
        timeline->cycle[kTimelineStart] = AscendC::GetSystemCycle();
        *status = kDeviceE2eStatusOk;
        for (uint32_t worker = 0u; worker < worker_count; ++worker) {
            __gm__ CombineReadyDescriptor *descriptor =
                reinterpret_cast<__gm__ CombineReadyDescriptor *>(
                    descriptor_mailbox +
                    static_cast<uint64_t>(worker) *
                        sizeof(CombineReadyDescriptor));
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(descriptor));
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(descriptor) +
                           64u);
            if (!DescriptorValid(descriptor, worker, worker_count, elements,
                                 generation, wave, digest)) {
                *status = kDeviceE2eStatusDescriptor;
                break;
            }
        }
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(status));
    }
    AscendC::SyncAll<true>();
    if (pe == inc_pe && block == 0u)
        timeline->cycle[kTimelineDescriptorDone] = AscendC::GetSystemCycle();

    if (pe == inc_pe && *status == kDeviceE2eStatusOk) {
        const bool pipeline = lanes_per_worker > kPullLanesPerSource &&
                              chunk_count >= 2u;
        if (pipeline) {
            const uint32_t pull_blocks =
                worker_count * kPullLanesPerSource;
            if (block < pull_blocks) {
                const uint32_t source = block / kPullLanesPerSource;
                const uint32_t pull_lane = block % kPullLanesPerSource;
                __gm__ float *staging =
                    reinterpret_cast<__gm__ float *>(inc_staging);
                __gm__ float *remote =
                    reinterpret_cast<__gm__ float *>(symmetric_partials);
                const uint32_t chunks_per_owner =
                    (chunk_count + worker_count - 1u) / worker_count;
                const uint32_t sequence_count =
                    chunks_per_owner * worker_count;
                for (uint32_t sequence = pull_lane;
                     sequence < sequence_count;
                     sequence += kPullLanesPerSource) {
                    const uint32_t chunk = PipelineChunkFromSequence(
                        sequence, chunk_count, worker_count);
                    if (chunk >= chunk_count) continue;
                    const uint64_t begin =
                        static_cast<uint64_t>(chunk) * kPipelineChunkElements;
                    const uint64_t remaining = padded_elements - begin;
                    const uint64_t count =
                        remaining < kPipelineChunkElements
                            ? remaining
                            : kPipelineChunkElements;
                    GetExact(
                        reinterpret_cast<__gm__ uint8_t *>(
                            staging + static_cast<uint64_t>(source) *
                                          padded_elements + begin),
                        reinterpret_cast<__gm__ uint8_t *>(remote + begin),
                        static_cast<uint32_t>(count * sizeof(float)),
                        static_cast<int32_t>(source));
                    AscendC::PipeBarrier<PIPE_ALL>();
                    __gm__ uint64_t *ready =
                        reinterpret_cast<__gm__ uint64_t *>(
                            ready_flags +
                            (static_cast<uint64_t>(source) * chunk_count +
                             chunk) * kDevicePipelineReadyStride);
                    *ready = generation;
                    AscendC::PipeBarrier<PIPE_ALL>();
                    dcci_cacheline(
                        reinterpret_cast<__gm__ uint8_t *>(ready));
                }
            } else {
                const uint32_t reducer_lanes =
                    lanes_per_worker - kPullLanesPerSource;
                const uint32_t reducer = block - pull_blocks;
                const uint32_t owner = reducer / reducer_lanes;
                const uint32_t lane = reducer % reducer_lanes;
                const uint64_t owner_begin =
                    static_cast<uint64_t>(elements) * owner / worker_count;
                const uint64_t owner_end =
                    static_cast<uint64_t>(elements) * (owner + 1u) /
                    worker_count;
                const uint64_t owner_elements = owner_end - owner_begin;
                const uint64_t owner_lines =
                    (owner_elements + kFloatsPerCacheLine - 1u) /
                    kFloatsPerCacheLine;
                const uint64_t first_line =
                    owner_lines * lane / reducer_lanes;
                const uint64_t last_line =
                    owner_lines * (lane + 1u) / reducer_lanes;
                const uint64_t first_local_element =
                    first_line * kFloatsPerCacheLine;
                const uint64_t last_local_unclamped =
                    last_line * kFloatsPerCacheLine;
                const uint64_t last_local_element =
                    last_local_unclamped < owner_elements
                        ? last_local_unclamped
                        : owner_elements;
                ReduceOwnerFp32Vector(
                    reinterpret_cast<__gm__ float *>(inc_staging),
                    reinterpret_cast<__gm__ float *>(reduced_output),
                    padded_elements, worker_count, owner_begin,
                    first_local_element, last_local_element,
                    static_cast<int32_t>(owner), true, ready_flags,
                    chunk_count, generation, status);
            }
        } else {
            // One AIV owns one cache-line partition of one output owner.
            // Small messages retain this zero-handshake path.
            const uint32_t owner = block / lanes_per_worker;
            const uint32_t lane = block % lanes_per_worker;
            const uint64_t owner_begin =
                static_cast<uint64_t>(elements) * owner / worker_count;
            const uint64_t owner_end = static_cast<uint64_t>(elements) *
                                       (owner + 1u) / worker_count;
            const uint64_t owner_elements = owner_end - owner_begin;
            const uint64_t owner_lines =
                (owner_elements + kFloatsPerCacheLine - 1u) /
                kFloatsPerCacheLine;
            const uint64_t first_line =
                owner_lines * lane / lanes_per_worker;
            const uint64_t last_line =
                owner_lines * (lane + 1u) / lanes_per_worker;
            const uint64_t first_local_element =
                first_line * kFloatsPerCacheLine;
            const uint64_t last_local_unclamped =
                last_line * kFloatsPerCacheLine;
            const uint64_t last_local_element =
                last_local_unclamped < owner_elements
                    ? last_local_unclamped
                    : owner_elements;
            if (first_local_element < last_local_element) {
                const uint64_t logical_begin =
                    owner_begin + first_local_element;
                const uint64_t logical_end =
                    owner_begin + last_local_element;
                const uint64_t pull_begin = logical_begin /
                                            kFloatsPerCacheLine *
                                            kFloatsPerCacheLine;
                const uint64_t pull_end_unclamped =
                    (logical_end + kFloatsPerCacheLine - 1u) /
                    kFloatsPerCacheLine * kFloatsPerCacheLine;
                const uint64_t pull_end =
                    pull_end_unclamped < padded_elements
                        ? pull_end_unclamped
                        : padded_elements;
                const uint32_t pull_bytes = static_cast<uint32_t>(
                    (pull_end - pull_begin) * sizeof(float));
                __gm__ float *staging =
                    reinterpret_cast<__gm__ float *>(inc_staging);
                __gm__ float *remote =
                    reinterpret_cast<__gm__ float *>(symmetric_partials);
                // Rotate first source to avoid an instantaneous hot peer.
                const uint32_t first_source =
                    (owner + lane) % worker_count;
                for (uint32_t step = 0u; step < worker_count; ++step) {
                    const uint32_t source =
                        (first_source + step) % worker_count;
                    GetExact(
                        reinterpret_cast<__gm__ uint8_t *>(
                            staging + static_cast<uint64_t>(source) *
                                          padded_elements + pull_begin),
                        reinterpret_cast<__gm__ uint8_t *>(remote + pull_begin),
                        pull_bytes, static_cast<int32_t>(source));
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                ReduceOwnerFp32Vector(
                    staging,
                    reinterpret_cast<__gm__ float *>(reduced_output),
                    padded_elements, worker_count, owner_begin,
                    first_local_element, last_local_element,
                    static_cast<int32_t>(owner), false, ready_flags,
                    chunk_count, generation, status);
            }
        }
    }
    AscendC::SyncAll<true>();
    if (pe == inc_pe && block == 0u)
        timeline->cycle[kTimelinePullDone] = AscendC::GetSystemCycle();

    if (pe == inc_pe && *status == kDeviceE2eStatusOk) {
        // GET completion and the following consumer both use the MTE path.
        // No scalar cache observes staging, so a full DCCI sweep is neither
        // required nor desirable for large messages.
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    AscendC::SyncAll<true>();
    if (pe == inc_pe && block == 0u)
        timeline->cycle[kTimelineAcquireDone] = AscendC::GetSystemCycle();

    if (pe == inc_pe) {
        // Every slice has completed all source GETs before this global point.
        // Success ACK releases consumed source storage.  A negative ACK is
        // equally important: it releases a failed wave's slot without ever
        // claiming that source rows were consumed.
        if (block < worker_count) {
            __gm__ CombineAck *local_ack =
                reinterpret_cast<__gm__ CombineAck *>(ack_mailbox) + block;
            local_ack->magic = kPullCombineMagic;
            local_ack->abi_version = kPullCombineAbiVersion;
            local_ack->struct_bytes = sizeof(CombineAck);
            local_ack->generation = generation;
            local_ack->sequence = 1u;
            local_ack->source_rank = block;
            local_ack->status = *status;
            local_ack->rows_consumed =
                *status == kDeviceE2eStatusOk ? elements : 0u;
            for (uint32_t i = 0u; i < 3u; ++i)
                local_ack->reserved[i] = 0u;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(local_ack));
            aclshmem_putmem(local_ack, local_ack, sizeof(CombineAck),
                            static_cast<int32_t>(block));
        }

        if (*status == kDeviceE2eStatusOk)
            AscendC::PipeBarrier<PIPE_ALL>();
    }
    AscendC::SyncAll<true>();
    if (pe == inc_pe && block == 0u)
        timeline->cycle[kTimelineReduceDone] = AscendC::GetSystemCycle();

    if (pe == inc_pe && *status == kDeviceE2eStatusOk) {
        // Reduce output is produced by MTE3 and consumed by MTE egress.  The
        // vector helper has already drained MTE3 before this global phase.
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    AscendC::SyncAll<true>();
    if (pe == inc_pe && block == 0u)
        timeline->cycle[kTimelineReleaseDone] = AscendC::GetSystemCycle();

    // Egress was issued tile-by-tile directly from UB during reduction.
    AscendC::SyncAll<true>();
    if (pe == inc_pe && block == 0u) {
        timeline->cycle[kTimelineEgressDone] = AscendC::GetSystemCycle();
        AscendC::PipeBarrier<PIPE_ALL>();
        dcci_cacheline(status_line);
    }
}

extern "C" void launch_inc_dc_pull_combine_device_e2e(
    uint32_t block_dim, void *stream, uint8_t *symmetric_partials,
    uint8_t *inc_staging, uint8_t *reduced_output,
    uint8_t *descriptor_mailbox, uint8_t *ack_mailbox,
    uint8_t *status_line, uint8_t *ready_flags, uint64_t ffts_addr,
    uint32_t elements,
    uint32_t worker_count, uint32_t lanes_per_worker, int32_t inc_pe,
    uint64_t generation, uint32_t wave, uint64_t digest)
{
    inc_dc_pull_combine_device_e2e_kernel<<<block_dim, nullptr, stream>>>(
        symmetric_partials, inc_staging, reduced_output, descriptor_mailbox,
        ack_mailbox, status_line, ready_flags, ffts_addr, elements, worker_count,
        lanes_per_worker, inc_pe, generation, wave, digest);
}
