#include "kernel_operator.h"
#include "shmem.h"

#include "inc_dc_pull_combine_abi.h"

using namespace inc::dc::pull_combine;

namespace {

constexpr uint32_t kDeviceE2eStatusOk = 0u;
constexpr uint32_t kDeviceE2eStatusDescriptor = 1u;
constexpr uint32_t kRmaAlignment = 64u;

__aicore__ inline void GetExact(__gm__ uint8_t *destination,
                                __gm__ uint8_t *source, uint32_t bytes,
                                int32_t pe)
{
    const uint32_t bulk = bytes / kRmaAlignment * kRmaAlignment;
    if (bulk != 0u) aclshmem_getmem(destination, source, bulk, pe);
    if (bulk != bytes)
        aclshmem_getmem(destination + bulk, source + bulk, bytes - bulk, pe);
}

__aicore__ inline void PutExact(__gm__ uint8_t *destination,
                                __gm__ uint8_t *source, uint32_t bytes,
                                int32_t pe)
{
    const uint32_t bulk = bytes / kRmaAlignment * kRmaAlignment;
    if (bulk != 0u) aclshmem_putmem(destination, source, bulk, pe);
    if (bulk != bytes)
        aclshmem_putmem(destination + bulk, source + bulk, bytes - bulk, pe);
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
    uint64_t ffts_addr, uint32_t elements, uint32_t worker_count,
    uint32_t lanes_per_worker, int32_t inc_pe, uint64_t generation,
    uint32_t wave, uint64_t digest)
{
    // SyncAll requires the per-process FFTS workspace registered by SHMEM.
    // Passing the address explicitly keeps this qualification kernel aligned
    // with the production fusion and MegaMoE device entry points.
    shmemx_set_ffts_config(ffts_addr);
    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t blocks = AscendC::GetBlockNum();
    const int32_t pe = aclshmem_my_pe();
    constexpr uint64_t kFloatsPerCacheLine = 64u / sizeof(float);
    const uint64_t padded_elements =
        (static_cast<uint64_t>(elements) + kFloatsPerCacheLine - 1u) /
        kFloatsPerCacheLine * kFloatsPerCacheLine;
    __gm__ uint32_t *status =
        reinterpret_cast<__gm__ uint32_t *>(status_line);

    if (pe == inc_pe && block == 0u) {
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

    if (pe == inc_pe && *status == kDeviceE2eStatusOk) {
        const uint32_t worker = block / lanes_per_worker;
        const uint32_t stripe = block % lanes_per_worker;
        if (worker < worker_count) {
            const uint64_t source_lines = padded_elements /
                                          kFloatsPerCacheLine;
            const uint64_t first_line = source_lines * stripe /
                                        lanes_per_worker;
            const uint64_t last_line = source_lines * (stripe + 1u) /
                                       lanes_per_worker;
            const uint64_t stripe_begin =
                first_line * kFloatsPerCacheLine;
            const uint64_t stripe_end = last_line * kFloatsPerCacheLine;
            const uint32_t stripe_elements = static_cast<uint32_t>(
                stripe_end - stripe_begin);
            __gm__ float *local = reinterpret_cast<__gm__ float *>(
                inc_staging) + static_cast<uint64_t>(worker) *
                padded_elements +
                stripe_begin;
            __gm__ float *remote = reinterpret_cast<__gm__ float *>(
                symmetric_partials) + stripe_begin;
            GetExact(reinterpret_cast<__gm__ uint8_t *>(local),
                     reinterpret_cast<__gm__ uint8_t *>(remote),
                     stripe_elements * sizeof(float),
                     static_cast<int32_t>(worker));
        }
    }
    AscendC::SyncAll<true>();

    if (pe == inc_pe && *status == kDeviceE2eStatusOk) {
        // GET completes through MTE/SDMA, outside the scalar cache hierarchy.
        // Acquire every padded staging cache line before AIV code consumes it.
        const uint64_t staging_bytes = static_cast<uint64_t>(worker_count) *
                                       padded_elements * sizeof(float);
        for (uint64_t offset = static_cast<uint64_t>(block) * 64u;
             offset < staging_bytes;
             offset += static_cast<uint64_t>(blocks) * 64u) {
            dcci_cacheline(inc_staging + offset);
        }
    }
    AscendC::SyncAll<true>();

    if (pe == inc_pe && *status == kDeviceE2eStatusOk) {
        // GET completion is now global across the INC kernel.  Publish one
        // ACK per worker before reduction; source partial buffers are no
        // longer read after this point.
        if (block < worker_count) {
            __gm__ CombineAck *local_ack =
                reinterpret_cast<__gm__ CombineAck *>(ack_mailbox) + block;
            local_ack->magic = kPullCombineMagic;
            local_ack->abi_version = kPullCombineAbiVersion;
            local_ack->struct_bytes = sizeof(CombineAck);
            local_ack->generation = generation;
            local_ack->sequence = 1u;
            local_ack->source_rank = block;
            local_ack->status = kDeviceE2eStatusOk;
            local_ack->rows_consumed = elements;
            for (uint32_t i = 0u; i < 3u; ++i)
                local_ack->reserved[i] = 0u;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(local_ack));
            aclshmem_putmem(local_ack, local_ack, sizeof(CombineAck),
                            static_cast<int32_t>(block));
        }

        // Correctness-first strict FP32 reduction.  The next milestone
        // replaces this scalar loop with tiled UB vector reduction while
        // keeping the same staging and ACK boundaries.
        __gm__ float *staging = reinterpret_cast<__gm__ float *>(inc_staging);
        __gm__ float *output = reinterpret_cast<__gm__ float *>(
            reduced_output);
        const uint32_t owner = block / lanes_per_worker;
        const uint32_t lane = block % lanes_per_worker;
        const uint64_t owner_begin = static_cast<uint64_t>(elements) * owner /
                                     worker_count;
        const uint64_t owner_end = static_cast<uint64_t>(elements) *
                                   (owner + 1u) / worker_count;
        const uint64_t owner_elements = owner_end - owner_begin;
        const uint64_t owner_lines =
            (owner_elements + kFloatsPerCacheLine - 1u) /
            kFloatsPerCacheLine;
        const uint64_t first_line = owner_lines * lane / lanes_per_worker;
        const uint64_t last_line = owner_lines * (lane + 1u) /
                                   lanes_per_worker;
        const uint64_t first_local_element =
            first_line * kFloatsPerCacheLine;
        const uint64_t last_local_unclamped =
            last_line * kFloatsPerCacheLine;
        const uint64_t last_local_element =
            last_local_unclamped < owner_elements
                ? last_local_unclamped
                : owner_elements;
        // A cache line has exactly one producer AIV.  Interleaving elements
        // across blocks would create cross-AIV false sharing and allow one
        // cache writeback to discard another AIV's scalar stores.
        for (uint64_t local_element = first_local_element;
             local_element < last_local_element; ++local_element) {
            const uint64_t element = owner_begin + local_element;
            float sum = staging[element];
            for (uint32_t worker = 1u; worker < worker_count; ++worker)
                sum += staging[static_cast<uint64_t>(worker) *
                               padded_elements + element];
            output[static_cast<uint64_t>(owner) * padded_elements +
                   local_element] = sum;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    AscendC::SyncAll<true>();

    if (pe == inc_pe && *status == kDeviceE2eStatusOk) {
        // Scalar stores may still reside in an AIV cache.  The MTE/SDMA
        // transport reads the source through a different engine, so every
        // produced cache line must be made visible before egress starts.
        const uint32_t owner = block / lanes_per_worker;
        const uint32_t lane = block % lanes_per_worker;
        const uint64_t owner_begin = static_cast<uint64_t>(elements) * owner /
                                     worker_count;
        const uint64_t owner_end = static_cast<uint64_t>(elements) *
                                   (owner + 1u) / worker_count;
        const uint64_t owner_elements = owner_end - owner_begin;
        const uint64_t owner_lines =
            (owner_elements + kFloatsPerCacheLine - 1u) /
            kFloatsPerCacheLine;
        const uint64_t first_line = owner_lines * lane / lanes_per_worker;
        const uint64_t last_line = owner_lines * (lane + 1u) /
                                   lanes_per_worker;
        for (uint64_t line = first_line; line < last_line; ++line) {
            dcci_cacheline(reduced_output +
                           static_cast<uint64_t>(owner) *
                               padded_elements * sizeof(float) +
                           line * 64u);
        }
    }
    AscendC::SyncAll<true>();

    const uint32_t egress_owner = block / lanes_per_worker;
    const uint32_t egress_lane = block % lanes_per_worker;
    if (pe == inc_pe && *status == kDeviceE2eStatusOk &&
        egress_owner < worker_count && egress_lane == 0u) {
        const uint64_t begin = static_cast<uint64_t>(elements) *
                               egress_owner / worker_count;
        const uint64_t end = static_cast<uint64_t>(elements) *
                             (egress_owner + 1u) / worker_count;
        __gm__ float *segment = reinterpret_cast<__gm__ float *>(
            reduced_output) + static_cast<uint64_t>(egress_owner) *
            padded_elements;
        const uint64_t owner_elements = end - begin;
        const uint64_t owner_padded_elements =
            (owner_elements + kFloatsPerCacheLine - 1u) /
            kFloatsPerCacheLine * kFloatsPerCacheLine;
        PutExact(reduced_output,
                 reinterpret_cast<__gm__ uint8_t *>(segment),
                 static_cast<uint32_t>(owner_padded_elements * sizeof(float)),
                 static_cast<int32_t>(egress_owner));
    }
    AscendC::SyncAll<true>();
}

extern "C" void launch_inc_dc_pull_combine_device_e2e(
    uint32_t block_dim, void *stream, uint8_t *symmetric_partials,
    uint8_t *inc_staging, uint8_t *reduced_output,
    uint8_t *descriptor_mailbox, uint8_t *ack_mailbox,
    uint8_t *status_line, uint64_t ffts_addr, uint32_t elements,
    uint32_t worker_count, uint32_t lanes_per_worker, int32_t inc_pe,
    uint64_t generation, uint32_t wave, uint64_t digest)
{
    inc_dc_pull_combine_device_e2e_kernel<<<block_dim, nullptr, stream>>>(
        symmetric_partials, inc_staging, reduced_output, descriptor_mailbox,
        ack_mailbox, status_line, ffts_addr, elements, worker_count,
        lanes_per_worker, inc_pe, generation, wave, digest);
}
