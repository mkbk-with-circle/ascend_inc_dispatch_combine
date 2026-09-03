#include "kernel_operator.h"
#include "shmem.h"

namespace {

constexpr uint32_t kUbSlots = 2u;
constexpr uint32_t kUbBytes = 12u * 1024u;

__aicore__ inline void SetGetReady(uint32_t slot)
{
    if (slot == 0u)
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0u);
    else
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(1u);
}

__aicore__ inline void WaitGetReady(uint32_t slot)
{
    if (slot == 0u)
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0u);
    else
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(1u);
}

__aicore__ inline void SetPutDone(uint32_t slot)
{
    if (slot == 0u)
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
    else
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
}

__aicore__ inline void WaitPutDone(uint32_t slot)
{
    if (slot == 0u)
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0u);
    else
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1u);
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void inc_dc_pull_dispatch_v2_relay_probe_kernel(
    GM_ADDR symmetric_source, GM_ADDR symmetric_destination,
    GM_ADDR inc_staging, uint64_t ffts_addr, uint64_t bytes_per_source,
    uint32_t worker_count, int32_t inc_pe, uint32_t fanout,
    uint32_t tile_bytes, uint32_t channels_per_source)
{
    shmemx_set_ffts_config(ffts_addr);
    if (aclshmem_my_pe() != inc_pe) return;
    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t blocks = AscendC::GetBlockNum();
    const uint32_t active_channels = worker_count * channels_per_source;
    if (worker_count < 2u || fanout == 0u || fanout > worker_count ||
        tile_bytes == 0u || tile_bytes > kUbBytes ||
        channels_per_source == 0u || active_channels > blocks)
        return;
    if (block >= active_channels) return;
    (void)inc_staging;

    const uint32_t source = block % worker_count;
    const uint32_t lane = block / worker_count;
    const uint64_t tiles =
        (bytes_per_source + tile_bytes - 1u) / tile_bytes;
    if (lane >= tiles) return;

    __ubuf__ uint8_t *ub[kUbSlots] = {
        reinterpret_cast<__ubuf__ uint8_t *>(0u),
        reinterpret_cast<__ubuf__ uint8_t *>(
            static_cast<uint64_t>(kUbBytes))};
    bool put_busy[kUbSlots]{false, false};

    uint64_t tile = lane;
    uint32_t slot = 0u;
    uint64_t offset = tile * tile_bytes;
    uint32_t bytes = static_cast<uint32_t>(
        bytes_per_source - offset < tile_bytes
            ? bytes_per_source - offset : tile_bytes);
    aclshmemx_mte_get_nbi(
        ub[slot], symmetric_source + offset, bytes,
        static_cast<int32_t>(source), 0u);
    SetGetReady(slot);

    while (tile < tiles) {
        WaitGetReady(slot);

        const uint64_t next_tile = tile + channels_per_source;
        const uint32_t next_slot = 1u - slot;
        uint64_t next_offset = 0u;
        uint32_t next_bytes = 0u;
        if (next_tile < tiles) {
            if (put_busy[next_slot]) {
                WaitPutDone(next_slot);
                put_busy[next_slot] = false;
            }
            next_offset = next_tile * tile_bytes;
            next_bytes = static_cast<uint32_t>(
                bytes_per_source - next_offset < tile_bytes
                    ? bytes_per_source - next_offset : tile_bytes);
            aclshmemx_mte_get_nbi(
                ub[next_slot], symmetric_source + next_offset, next_bytes,
                static_cast<int32_t>(source), 0u);
            SetGetReady(next_slot);
        }

        for (uint32_t destination_ordinal = 0u;
             destination_ordinal < fanout; ++destination_ordinal) {
            const uint32_t destination =
                (source + destination_ordinal) % worker_count;
            aclshmemx_mte_put_nbi(
                symmetric_destination +
                    static_cast<uint64_t>(source) * bytes_per_source +
                    offset,
                ub[slot], bytes, static_cast<int32_t>(destination), 0u);
        }
        SetPutDone(slot);
        put_busy[slot] = true;

        if (next_tile >= tiles) break;
        tile = next_tile;
        slot = next_slot;
        offset = next_offset;
        bytes = next_bytes;
    }

    for (uint32_t i = 0u; i < kUbSlots; ++i)
        if (put_busy[i]) WaitPutDone(i);
    aclshmemx_mte_quiet();
}

extern "C" void launch_inc_dc_pull_dispatch_v2_relay_probe(
    uint32_t block_dim, void *stream, uint8_t *symmetric_source,
    uint8_t *symmetric_destination, uint8_t *inc_staging,
    uint64_t ffts_addr, uint64_t bytes_per_source, uint32_t worker_count,
    int32_t inc_pe, uint32_t fanout, uint32_t tile_bytes,
    uint32_t channels_per_source)
{
    inc_dc_pull_dispatch_v2_relay_probe_kernel<<<
        block_dim, nullptr, stream>>>(
        symmetric_source, symmetric_destination, inc_staging,
        ffts_addr, bytes_per_source, worker_count, inc_pe, fanout,
        tile_bytes, channels_per_source);
}
