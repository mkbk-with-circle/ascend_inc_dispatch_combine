#include "kernel_operator.h"
#include "shmem.h"

// One correctness anchor for the Combine pull direction.  The ready
// notification is represented by a host-side SHMEM barrier before launch:
// the remote source is initialized before the INC is allowed to issue GET.
// aclshmem_getmem is synchronous at this boundary and selects the supported
// transport for the current topology instead of hard-coding UDMA.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void inc_dc_pull_combine_transport_probe_kernel(
    GM_ADDR local_destination, GM_ADDR symmetric_source,
    uint32_t bytes, int32_t inc_pe, int32_t worker_count,
    uint32_t lanes_per_worker, uint32_t warmup, uint32_t iterations)
{
    if (aclshmem_my_pe() == inc_pe) {
        const uint32_t block = AscendC::GetBlockIdx();
        const uint32_t worker = block / lanes_per_worker;
        const uint32_t stripe = block % lanes_per_worker;
        if (worker < static_cast<uint32_t>(worker_count)) {
            const int32_t source_pe = worker >= static_cast<uint32_t>(inc_pe)
                ? static_cast<int32_t>(worker + 1u)
                : static_cast<int32_t>(worker);
            const uint64_t stripe_begin =
                static_cast<uint64_t>(bytes) * stripe / lanes_per_worker;
            const uint64_t stripe_end =
                static_cast<uint64_t>(bytes) * (stripe + 1u) /
                lanes_per_worker;
            const uint32_t stripe_bytes = static_cast<uint32_t>(
                stripe_end - stripe_begin);
            __gm__ uint8_t *destination = local_destination +
                static_cast<uint64_t>(worker) * bytes + stripe_begin;
            __gm__ uint8_t *source = symmetric_source + stripe_begin;
            for (uint32_t i = 0u; i < warmup; ++i)
                aclshmem_getmem(destination, source, stripe_bytes, source_pe);
            for (uint32_t i = 0u; i < iterations; ++i)
                aclshmem_getmem(destination, source, stripe_bytes, source_pe);
        }
    }
}

extern "C" void launch_inc_dc_pull_combine_transport_probe(
    uint32_t block_dim, void *stream, uint8_t *local_destination,
    uint8_t *symmetric_source, uint32_t bytes, int32_t inc_pe,
    int32_t worker_count, uint32_t lanes_per_worker, uint32_t warmup,
    uint32_t iterations)
{
    inc_dc_pull_combine_transport_probe_kernel<<<block_dim, nullptr, stream>>>(
        local_destination, symmetric_source, bytes, inc_pe, worker_count,
        lanes_per_worker, warmup, iterations);
}
