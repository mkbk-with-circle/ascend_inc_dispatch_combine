#include "kernel_operator.h"
#include "shmem.h"

#include "inc_dc_device_journal_abi.h"
#include "inc_dc_endpoint_dispatch_abi.h"

using namespace inc::dc::pull_combine;

namespace {

constexpr uint32_t kIndexInvalidHeader = 10u;
constexpr uint32_t kIndexCapacity = 11u;
constexpr uint32_t kIndexDuplicateToken = 12u;

__aicore__ inline uint64_t HashToken(uint64_t value)
{
    value ^= value >> 33u;
    value *= 0xff51afd7ed558ccdull;
    value ^= value >> 33u;
    value *= 0xc4ceb9fe1a85ec53ull;
    return value ^ (value >> 33u);
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void inc_dc_device_journal_index_kernel(
    GM_ADDR inc_packets, GM_ADDR journal_header, GM_ADDR journal_entries,
    GM_ADDR journal_hash, GM_ADDR journal_row_map,
    GM_ADDR destination_rows, uint64_t ffts_addr, uint64_t slot_bytes,
    uint64_t entry_capacity, uint64_t hash_capacity, uint32_t worker_count,
    int32_t inc_pe, uint64_t generation, uint32_t wave)
{
    shmemx_set_ffts_config(ffts_addr);
    if (aclshmem_my_pe() != inc_pe || AscendC::GetBlockIdx() != 0u) return;

    __gm__ DeviceJournalHeader *header =
        reinterpret_cast<__gm__ DeviceJournalHeader *>(journal_header);
    dcci_cacheline(journal_header);
    if (header->magic != kDeviceJournalMagic ||
        header->abi_version != kDeviceJournalAbiVersion ||
        header->struct_bytes != sizeof(DeviceJournalHeader) ||
        header->generation != generation || header->wave != wave ||
        header->worker_count != worker_count || header->status != 0u ||
        header->flags != 0u || worker_count < 2u ||
        worker_count > kEndpointDispatchMaxWorkers || hash_capacity == 0u ||
        (hash_capacity & (hash_capacity - 1u)) != 0u) {
        header->status = kIndexInvalidHeader;
        dcci_cacheline(journal_header);
        return;
    }

    __gm__ DeviceJournalEntry *entries =
        reinterpret_cast<__gm__ DeviceJournalEntry *>(journal_entries);
    __gm__ uint32_t *hash = reinterpret_cast<__gm__ uint32_t *>(journal_hash);
    __gm__ uint32_t *row_map =
        reinterpret_cast<__gm__ uint32_t *>(journal_row_map);
    __gm__ uint32_t *destination_count =
        reinterpret_cast<__gm__ uint32_t *>(destination_rows);
    uint32_t destination_cursor[128]{};
    uint64_t index = 0u;
    for (uint32_t source = 0u; source < worker_count; ++source) {
        __gm__ uint8_t *packet = inc_packets +
            static_cast<uint64_t>(source) * slot_bytes;
        __gm__ EndpointDispatchPacketHeader *packet_header =
            reinterpret_cast<__gm__ EndpointDispatchPacketHeader *>(packet);
        dcci_cacheline(packet);
        for (uint32_t token = 0u; token < packet_header->token_count;
             ++token, ++index) {
            if (index >= entry_capacity || index >= kDeviceJournalEmpty) {
                header->status = kIndexCapacity;
                dcci_cacheline(journal_header);
                return;
            }
            __gm__ EndpointDispatchTokenRecord *record =
                reinterpret_cast<__gm__ EndpointDispatchTokenRecord *>(
                    packet + packet_header->tokens_offset) + token;
            uint64_t expected[2]{0u, 0u};
            for (uint32_t local = 0u; local < record->assignment_count;
                 ++local) {
                __gm__ EndpointDispatchAssignmentRecord *assignment =
                    reinterpret_cast<
                        __gm__ EndpointDispatchAssignmentRecord *>(
                            packet + packet_header->assignments_offset) +
                    record->assignment_begin + local;
                const uint32_t destination = assignment->destination_rank;
                expected[destination >> 6u] |=
                    1ull << (destination & 63u);
            }
            uint64_t bucket = HashToken(record->token_id) &
                (hash_capacity - 1u);
            bool inserted = false;
            for (uint64_t probe = 0u; probe < hash_capacity; ++probe) {
                const uint32_t prior = hash[bucket];
                if (prior == kDeviceJournalEmpty) {
                    hash[bucket] = static_cast<uint32_t>(index);
                    inserted = true;
                    break;
                }
                if (entries[prior].token_id == record->token_id) break;
                bucket = (bucket + 1u) & (hash_capacity - 1u);
            }
            if (!inserted) {
                header->status = kIndexDuplicateToken;
                dcci_cacheline(journal_header);
                return;
            }
            __gm__ DeviceJournalEntry *entry = entries + index;
            entry->token_id = record->token_id;
            entry->owner_rank = source;
            entry->owner_row = token;
            entry->expected[0] = expected[0];
            entry->expected[1] = expected[1];
            entry->received[0] = 0u;
            entry->received[1] = 0u;
            entry->accumulator_index = static_cast<uint32_t>(index);
            entry->flags = 0u;
            entry->reserved = 0u;
            for (uint32_t destination = 0u; destination < worker_count;
                 ++destination) {
                if ((expected[destination >> 6u] &
                     (1ull << (destination & 63u))) == 0u)
                    continue;
                row_map[static_cast<uint64_t>(destination) * entry_capacity +
                        index] = destination_cursor[destination]++;
            }
        }
    }
    if (index != header->token_count) {
        header->status = kIndexInvalidHeader;
        dcci_cacheline(journal_header);
        return;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    for (uint32_t destination = 0u; destination < worker_count;
         ++destination)
        destination_count[destination] = destination_cursor[destination];
    // Header::flags is the publication word for entries, hash, canonical row
    // map and destination counts.  Flush every preceding scalar GM write
    // before making INDEX_READY observable to a later kernel/AIV.
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_entire_cache();
    header->flags = kDeviceJournalIndexReady;
    dcci_cacheline(journal_header);
}

extern "C" void launch_inc_dc_device_journal_index(
    uint32_t block_dim, void *stream, uint8_t *inc_packets,
    uint8_t *journal_header, uint8_t *journal_entries,
    uint8_t *journal_hash, uint8_t *journal_row_map,
    uint8_t *destination_rows, uint64_t ffts_addr, uint64_t slot_bytes,
    uint64_t entry_capacity, uint64_t hash_capacity, uint32_t worker_count,
    int32_t inc_pe, uint64_t generation, uint32_t wave)
{
    inc_dc_device_journal_index_kernel<<<block_dim, nullptr, stream>>>(
        inc_packets, journal_header, journal_entries, journal_hash,
        journal_row_map, destination_rows, ffts_addr, slot_bytes,
        entry_capacity,
        hash_capacity, worker_count, inc_pe,
        generation, wave);
}
