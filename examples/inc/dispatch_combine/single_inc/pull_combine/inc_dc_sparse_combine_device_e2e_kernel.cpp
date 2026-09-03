#include "kernel_operator.h"
#include "shmem.h"

#include "inc_dc_device_journal_abi.h"
#include "inc_dc_sparse_combine_abi.h"
#include "inc_dc_vector_reduce_aicore.h"

using namespace inc::dc::pull_combine;

namespace {

constexpr uint32_t kStatusOk = 0u;
constexpr uint32_t kStatusDescriptorTimeout = 1u;
constexpr uint32_t kStatusInvalidDescriptor = 2u;
constexpr uint32_t kStatusInvalidJournal = 3u;
constexpr uint32_t kStatusInvalidRowMap = 4u;
constexpr uint32_t kStatusTokenMismatch = 5u;
constexpr uint64_t kSpinLimit = 1000000000ull;
// Keep one vector operation within the platform-qualified repeat range.
// Larger rows are tiled; this avoids silent tail corruption on 910B.
// Ping/pong inputs and ping/pong accumulators fill the live 24-KiB AIV UB.
// This overlaps the next B->INC GET and the preceding INC->A PUT with vector
// reduction of the current tile.
constexpr uint32_t kTileElements = 1536u;
constexpr uint32_t kTileBytes = kTileElements * sizeof(float);
constexpr uint32_t kMte2V[2]{0u, 2u};
constexpr uint32_t kVMte2[2]{1u, 3u};
constexpr uint32_t kMte3V[2]{4u, 6u};
constexpr uint32_t kVMte3[2]{5u, 7u};
static_assert(kTileBytes * 4u <= INC_VEC_UB_BUDGET_BYTES,
              "sparse Combine tile exceeds AIV UB budget");

__aicore__ inline uint64_t HashToken(uint64_t value)
{
    value ^= value >> 33u;
    value *= 0xff51afd7ed558ccdull;
    value ^= value >> 33u;
    value *= 0xc4ceb9fe1a85ec53ull;
    return value ^ (value >> 33u);
}

__aicore__ inline bool WaitControlValue(__gm__ uint32_t *value,
                                        uint32_t expected)
{
    for (uint64_t spin = 0u; spin < kSpinLimit; ++spin) {
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(value));
        if (*value == expected) return true;
    }
    return false;
}

__aicore__ inline bool WaitSourceReady(__gm__ uint32_t *ready,
                                       __gm__ uint32_t *status)
{
    for (uint64_t spin = 0u; spin < kSpinLimit; ++spin) {
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(ready));
        if (*ready != 0u) return true;
        if ((spin & 1023u) == 0u) {
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(status));
            if (*status != kStatusOk) return false;
        }
    }
    return false;
}

__aicore__ inline bool DescriptorValid(
    __gm__ SparseCombineReadyDescriptor *descriptor, uint32_t source,
    uint32_t expected_rows, uint32_t hidden, uint64_t generation,
    uint32_t wave)
{
    const uint64_t payload_elements =
        static_cast<uint64_t>(expected_rows) * hidden;
    if (payload_elements > ~0ull / sizeof(float))
        return false;
    const uint64_t payload_bytes = payload_elements * sizeof(float);
    if (descriptor->magic != kSparseCombineMagic ||
        descriptor->abi_version != kSparseCombineAbiVersion ||
        descriptor->struct_bytes != sizeof(SparseCombineReadyDescriptor) ||
        descriptor->generation != generation || descriptor->sequence != 1u ||
        descriptor->wave != wave || descriptor->source_rank != source ||
        descriptor->row_count != expected_rows || descriptor->hidden != hidden ||
        descriptor->partial_dtype !=
            static_cast<uint32_t>(SparseCombineDType::FP32) ||
        descriptor->slot != 0u || descriptor->source_region_id == 0u ||
        descriptor->token_ids_offset != 0u || descriptor->payload_offset != 0u ||
        descriptor->payload_bytes != payload_bytes ||
        descriptor->metadata_digest == 0u ||
        (descriptor->flags & ~kSparseCombineFlagCanonicalRows) != 0u ||
        descriptor->reserved0 != 0u)
        return false;
    for (uint32_t i = 0u; i < 4u; ++i)
        if (descriptor->reserved[i] != 0u) return false;
    return true;
}

__aicore__ inline void GetExact(__gm__ uint8_t *destination,
                                 __gm__ uint8_t *source, uint32_t bytes,
                                 int32_t pe)
{
    const uint32_t bulk = bytes / 64u * 64u;
    if (bulk != 0u) aclshmem_getmem(destination, source, bulk, pe);
    if (bulk != bytes)
        aclshmem_getmem(destination + bulk, source + bulk, bytes - bulk, pe);
}

__aicore__ inline void PutFp32UbRemote(__ubuf__ uint8_t *source,
                                       __gm__ float *destination,
                                       uint32_t elements, int32_t pe,
                                       uint32_t v_mte3,
                                       uint32_t mte3_v)
{
    if (v_mte3 == kVMte3[0])
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kVMte3[0]);
    else
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kVMte3[1]);
    aclshmemx_mte_put_nbi(
        destination, reinterpret_cast<__ubuf__ float *>(source), elements,
        pe, 0u);
    if (mte3_v == kMte3V[0])
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kMte3V[0]);
    else
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kMte3V[1]);
}

__aicore__ inline void GetFp32RemoteToUb(__ubuf__ uint8_t *destination,
                                         __gm__ float *source,
                                         uint32_t elements, int32_t pe,
                                         uint32_t ping)
{
    if (ping == 0u)
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kVMte2[0]);
    else
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kVMte2[1]);
    aclshmemx_mte_get_nbi(
        reinterpret_cast<__ubuf__ float *>(destination), source, elements,
        pe, 0u);
    if (ping == 0u)
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kMte2V[0]);
    else
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kMte2V[1]);
}

__aicore__ inline void PublishAck(
    __gm__ SparseCombineDeviceAck *ack, uint32_t source, uint32_t status,
    uint64_t rows, uint64_t generation)
{
    const int32_t pe = static_cast<int32_t>(source);
    aclshmem_uint32_p(&ack->magic, kSparseCombineMagic, pe);
    aclshmem_uint16_p(&ack->abi_version, kSparseCombineAbiVersion, pe);
    aclshmem_uint16_p(&ack->struct_bytes, sizeof(SparseCombineDeviceAck), pe);
    aclshmem_uint64_p(&ack->sequence, 1u, pe);
    aclshmem_uint32_p(&ack->source_rank, source, pe);
    aclshmem_uint32_p(&ack->status, status, pe);
    aclshmem_uint64_p(&ack->rows_consumed, status == 0u ? rows : 0u, pe);
    for (uint32_t i = 0u; i < 3u; ++i)
        aclshmem_uint64_p(&ack->reserved[i], 0u, pe);
    aclshmem_quiet();
    aclshmem_uint64_p(&ack->generation, generation, pe);
    aclshmem_quiet();
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void inc_dc_sparse_combine_device_e2e_kernel(
    GM_ADDR symmetric_token_ids, GM_ADDR symmetric_partials,
    GM_ADDR reduced_output, GM_ADDR descriptor_mailbox, GM_ADDR ack_mailbox,
    GM_ADDR completion_mailbox, GM_ADDR journal_header,
    GM_ADDR journal_entries, GM_ADDR journal_hash,
    GM_ADDR journal_row_map, GM_ADDR combine_row_map,
    GM_ADDR destination_rows,
    GM_ADDR inc_token_ids, GM_ADDR status_line, uint64_t ffts_addr,
    uint64_t journal_capacity, uint64_t journal_hash_capacity,
    uint64_t row_capacity, uint32_t hidden, uint32_t worker_count,
    int32_t inc_pe, uint64_t generation, uint32_t wave, int32_t delay_rank,
    uint64_t delay_cycles, uint32_t combine_flags)
{
    shmemx_set_ffts_config(ffts_addr);
    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t blocks = AscendC::GetBlockNum();
    const int32_t pe = aclshmem_my_pe();
    __gm__ uint32_t *control =
        reinterpret_cast<__gm__ uint32_t *>(status_line);
    __gm__ uint32_t *status = control;

    if (worker_count < 2u || worker_count > kSparseCombineMaxWorkers) {
        if (pe == inc_pe && block == 0u) {
            *status = kStatusInvalidJournal;
            dcci_cacheline(status_line);
        }
        return;
    }

    if (pe != inc_pe) {
        if (block == 0u) {
            if (pe == delay_rank && delay_cycles != 0u) {
                const uint64_t delay_begin = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - delay_begin <
                       delay_cycles) {
                }
            }
            __gm__ SparseCombineReadyDescriptor *descriptor =
                reinterpret_cast<__gm__ SparseCombineReadyDescriptor *>(
                    descriptor_mailbox) + pe;
            const uint64_t publish_generation = descriptor->generation;
            // Two-phase publication: generation is the commit word.  A
            // single 128-byte PUT may expose its first cache line before the
            // second one, so INC must never accept generation until every
            // descriptor field is remotely visible.
            descriptor->generation = 0u;
            AscendC::PipeBarrier<PIPE_ALL>();
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(descriptor));
            aclshmem_putmem(descriptor, descriptor,
                            sizeof(SparseCombineReadyDescriptor), inc_pe);
            aclshmem_quiet();
            aclshmem_uint64_p(&descriptor->generation, publish_generation,
                              inc_pe);
            aclshmem_quiet();
            descriptor->generation = publish_generation;
            AscendC::PipeBarrier<PIPE_ALL>();
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(descriptor));
        }
        return;
    }

    __gm__ uint32_t *initialized = control + 1u;
    __gm__ uint32_t *source_ready = control + 2u;
    __gm__ uint32_t *expected_rows =
        reinterpret_cast<__gm__ uint32_t *>(destination_rows);
    __gm__ DeviceJournalHeader *header =
        reinterpret_cast<__gm__ DeviceJournalHeader *>(journal_header);
    __gm__ DeviceJournalEntry *entries =
        reinterpret_cast<__gm__ DeviceJournalEntry *>(journal_entries);
    __gm__ uint32_t *hash = reinterpret_cast<__gm__ uint32_t *>(journal_hash);
    __gm__ uint32_t *row_map =
        reinterpret_cast<__gm__ uint32_t *>(combine_row_map);
    __gm__ uint32_t *canonical_row_map =
        reinterpret_cast<__gm__ uint32_t *>(journal_row_map);
    __gm__ uint64_t *pulled_ids =
        reinterpret_cast<__gm__ uint64_t *>(inc_token_ids);
    if (block == 0u) {
        *status = kStatusOk;
        *initialized = 0u;
        for (uint32_t source = 0u; source < worker_count; ++source)
            source_ready[source] = 0u;
        AscendC::PipeBarrier<PIPE_ALL>();
        dcci_cacheline(status_line);
        dcci_cacheline(journal_header);
        if (header->magic != kDeviceJournalMagic ||
            header->generation != generation || header->wave != wave ||
            header->worker_count != worker_count || header->status != 0u ||
            header->flags != kDeviceJournalIndexReady ||
            header->token_count > journal_capacity ||
            journal_hash_capacity == 0u ||
            (journal_hash_capacity & (journal_hash_capacity - 1u)) != 0u) {
            *status = kStatusInvalidJournal;
        }
        *initialized = wave + 1u;
        AscendC::PipeBarrier<PIPE_ALL>();
        dcci_cacheline(status_line);

        uint32_t remaining = worker_count;
        for (uint64_t spin = 0u;
             spin < kSpinLimit && remaining != 0u && *status == kStatusOk;
             ++spin) {
            for (uint32_t source = 0u; source < worker_count; ++source) {
                if (source_ready[source] != 0u) continue;
                __gm__ SparseCombineReadyDescriptor *descriptor =
                    reinterpret_cast<__gm__ SparseCombineReadyDescriptor *>(
                        descriptor_mailbox) + source;
                dcci_cacheline(
                    reinterpret_cast<__gm__ uint8_t *>(descriptor));
                dcci_cacheline(
                    reinterpret_cast<__gm__ uint8_t *>(descriptor) + 64u);
                if (descriptor->generation != generation ||
                    descriptor->source_rank != source)
                    continue;
                if (expected_rows[source] > row_capacity ||
                    !DescriptorValid(descriptor, source,
                                     expected_rows[source], hidden,
                                     generation, wave) ||
                    descriptor->flags != combine_flags) {
                    *status = kStatusInvalidDescriptor;
                    break;
                }
                GetExact(
                    reinterpret_cast<__gm__ uint8_t *>(inc_token_ids) +
                        static_cast<uint64_t>(source) * row_capacity *
                            sizeof(uint64_t),
                    reinterpret_cast<__gm__ uint8_t *>(
                        symmetric_token_ids),
                    expected_rows[source] * sizeof(uint64_t), source);
                const bool canonical_rows = (combine_flags &
                    kSparseCombineFlagCanonicalRows) != 0u;
                if (!canonical_rows) {
                    // General fallback: B may return locally reduced rows in
                    // any order. Resolve carried IDs through the Dispatch hash
                    // and publish a source-private runtime map.
                    __gm__ uint32_t *source_map = row_map +
                        static_cast<uint64_t>(source) * journal_capacity;
                    for (uint64_t index = 0u; index < header->token_count;
                         ++index)
                        source_map[index] = kDeviceJournalEmpty;
                    for (uint32_t row = 0u;
                         row < expected_rows[source]; ++row) {
                        const uint64_t token_id = pulled_ids[
                            static_cast<uint64_t>(source) * row_capacity +
                            row];
                        uint64_t bucket = HashToken(token_id) &
                            (journal_hash_capacity - 1u);
                        uint32_t entry_index = kDeviceJournalEmpty;
                        for (uint64_t probe = 0u;
                             probe < journal_hash_capacity; ++probe) {
                            const uint32_t candidate = hash[bucket];
                            if (candidate == kDeviceJournalEmpty) break;
                            if (candidate < header->token_count &&
                                entries[candidate].token_id == token_id) {
                                entry_index = candidate;
                                break;
                            }
                            bucket = (bucket + 1u) &
                                (journal_hash_capacity - 1u);
                        }
                        if (entry_index == kDeviceJournalEmpty ||
                            (entries[entry_index].expected[source >> 6u] &
                             (1ull << (source & 63u))) == 0u ||
                            source_map[entry_index] != kDeviceJournalEmpty) {
                            *status = kStatusTokenMismatch;
                            break;
                        }
                        source_map[entry_index] = row;
                    }
                }
                if (*status != kStatusOk) break;
                AscendC::PipeBarrier<PIPE_ALL>();
                if (!canonical_rows) dcci_entire_cache();
                source_ready[source] = 1u;
                AscendC::PipeBarrier<PIPE_ALL>();
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(
                    source_ready + source));
                --remaining;
            }
        }
        if (remaining != 0u && *status == kStatusOk)
            *status = kStatusDescriptorTimeout;
        AscendC::PipeBarrier<PIPE_ALL>();
        dcci_cacheline(status_line);
    } else if (!WaitControlValue(initialized, wave + 1u)) {
        *status = kStatusDescriptorTimeout;
        AscendC::PipeBarrier<PIPE_ALL>();
        dcci_cacheline(status_line);
    }
    dcci_cacheline(status_line);

    __gm__ float *partials =
        reinterpret_cast<__gm__ float *>(symmetric_partials);
    __gm__ float *output = reinterpret_cast<__gm__ float *>(reduced_output);
    __ubuf__ uint8_t *input_ub[2]{
        reinterpret_cast<__ubuf__ uint8_t *>(0u),
        reinterpret_cast<__ubuf__ uint8_t *>(kTileBytes)};
    __ubuf__ uint8_t *acc_ub[2]{
        reinterpret_cast<__ubuf__ uint8_t *>(kTileBytes * 2u),
        reinterpret_cast<__ubuf__ uint8_t *>(kTileBytes * 3u)};

    // Block 0 joins the same static token partition after it finishes the
    // controller scan.  Other reducers have already started on early-ready
    // sources, so this recovers the lane without reintroducing a global wait.
    const bool reducer = true;
    const uint32_t reducer_index = block;
    const uint32_t reducer_count = blocks;
    if (reducer && *status == kStatusOk) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kVMte2[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kVMte2[1]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kMte3V[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kMte3V[1]);
        const uint64_t entries_per_reducer =
            (header->token_count + reducer_count - 1u) / reducer_count;
        const uint64_t index_begin =
            static_cast<uint64_t>(reducer_index) * entries_per_reducer;
        const uint64_t index_end = index_begin + entries_per_reducer <
                header->token_count
            ? index_begin + entries_per_reducer
            : header->token_count;
        bool source_acquired[128]{};
        const bool canonical = (combine_flags &
            kSparseCombineFlagCanonicalRows) != 0u;
        for (uint64_t index = index_begin; index < index_end; ++index) {
            __gm__ DeviceJournalEntry *entry = entries + index;
            uint32_t token_rows[128]{};
            for (uint32_t source = 0u; source < worker_count; ++source) {
                const bool expected =
                    (entry->expected[source >> 6u] &
                     (1ull << (source & 63u))) != 0u;
                if (!expected) continue;
                if (!source_acquired[source]) {
                    if (!WaitSourceReady(source_ready + source, status)) {
                        if (*status == kStatusOk)
                            *status = kStatusDescriptorTimeout;
                        break;
                    }
                    source_acquired[source] = true;
                }
                __gm__ uint32_t *selected_map = canonical
                    ? canonical_row_map : row_map;
                __gm__ uint32_t *row_pointer = selected_map +
                    static_cast<uint64_t>(source) * journal_capacity + index;
                if (!canonical)
                    dcci_cacheline(
                        reinterpret_cast<__gm__ uint8_t *>(row_pointer));
                const uint32_t row = *row_pointer;
                if (row >= expected_rows[source]) {
                    *status = kStatusInvalidRowMap;
                    break;
                }
                if (canonical && pulled_ids[
                        static_cast<uint64_t>(source) * row_capacity + row] !=
                        entry->token_id) {
                    *status = kStatusTokenMismatch;
                    break;
                }
                token_rows[source] = row;
            }
            if (*status != kStatusOk) break;

            for (uint64_t begin = 0u; begin < hidden;
                 begin += kTileElements) {
                const uint32_t count = static_cast<uint32_t>(
                    static_cast<uint64_t>(hidden) - begin < kTileElements
                        ? static_cast<uint64_t>(hidden) - begin
                        : kTileElements);
                const uint32_t ping = static_cast<uint32_t>(
                    (begin / kTileElements) & 1u);
                // The preceding remote MTE3 still owns acc_ub until this
                // event completes.  Fence before Duplicate overwrites it.
                if (ping == 0u)
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                        kMte3V[0]);
                else
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                        kMte3V[1]);
                AscendC::LocalTensor<float> acc =
                    IncVecBindFloatUb(acc_ub[ping], count * sizeof(float));
                AscendC::Duplicate(acc, 0.0f, count);
                AscendC::PipeBarrier<PIPE_V>();
                uint32_t source = 0u;
                while (source < worker_count &&
                       (entry->expected[source >> 6u] &
                        (1ull << (source & 63u))) == 0u)
                    ++source;
                uint32_t input_ping = 0u;
                if (source < worker_count) {
                    const uint32_t row = token_rows[source];
                    GetFp32RemoteToUb(
                        input_ub[input_ping],
                        partials + static_cast<uint64_t>(row) * hidden +
                            begin,
                        count, static_cast<int32_t>(source), 0u);
                }
                while (source < worker_count) {
                    uint32_t next = source + 1u;
                    while (next < worker_count &&
                           (entry->expected[next >> 6u] &
                            (1ull << (next & 63u))) == 0u)
                        ++next;
                    if (next < worker_count) {
                        const uint32_t next_ping = 1u - input_ping;
                        const uint32_t next_row = token_rows[next];
                        GetFp32RemoteToUb(
                            input_ub[next_ping],
                            partials + static_cast<uint64_t>(next_row) *
                                hidden + begin,
                            count, static_cast<int32_t>(next), next_ping);
                    }
                    if (input_ping == 0u)
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                            kMte2V[0]);
                    else
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                            kMte2V[1]);
                    AscendC::LocalTensor<float> temp =
                        IncVecBindFloatUb(input_ub[input_ping],
                                          count * sizeof(float));
                    AscendC::Add(acc, acc, temp, count);
                    AscendC::PipeBarrier<PIPE_V>();
                    if (input_ping == 0u)
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                            kVMte2[0]);
                    else
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                            kVMte2[1]);
                    source = next;
                    input_ping = 1u - input_ping;
                }
                if (ping == 0u)
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                        kVMte3[0]);
                else
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                        kVMte3[1]);
                PutFp32UbRemote(
                    acc_ub[ping],
                    output + static_cast<uint64_t>(entry->owner_row) * hidden +
                        begin,
                    count, static_cast<int32_t>(entry->owner_rank),
                    kVMte3[ping], kMte3V[ping]);
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kMte3V[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kMte3V[1]);
        // MTE3 completion makes the local UB reusable; quiet additionally
        // closes remote visibility before the cross-AIV join and completion
        // publication.  Without this fence, long streams can expose the
        // completion before early owner rows are globally observable.
        aclshmemx_mte_quiet();
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kVMte2[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kVMte2[1]);
    }
    AscendC::SyncAll<true>();
    dcci_cacheline(status_line);

    if (block == 0u) {
        for (uint32_t source = 0u; source < worker_count; ++source) {
            __gm__ SparseCombineDeviceAck *ack =
                reinterpret_cast<__gm__ SparseCombineDeviceAck *>(
                    ack_mailbox) + source;
            PublishAck(ack, source, *status, expected_rows[source], generation);
        }
        for (uint32_t owner = 0u; owner < worker_count; ++owner) {
            uint32_t rows = 0u;
            if (*status == kStatusOk) {
                for (uint64_t index = 0u; index < header->token_count; ++index)
                    rows += entries[index].owner_rank == owner;
            }
            __gm__ SparseCombineEgressCompletion *completion =
                reinterpret_cast<__gm__ SparseCombineEgressCompletion *>(
                    completion_mailbox) + owner;
            const int32_t owner_pe = static_cast<int32_t>(owner);
            aclshmem_uint32_p(&completion->magic, kSparseCombineMagic,
                              owner_pe);
            aclshmem_uint16_p(&completion->abi_version,
                              kSparseCombineAbiVersion, owner_pe);
            aclshmem_uint16_p(&completion->struct_bytes,
                              sizeof(SparseCombineEgressCompletion), owner_pe);
            aclshmem_uint32_p(&completion->wave, wave, owner_pe);
            aclshmem_uint32_p(&completion->owner_rank, owner, owner_pe);
            aclshmem_uint32_p(&completion->status, *status, owner_pe);
            aclshmem_uint32_p(&completion->row_count,
                              *status == 0u ? rows : 0u, owner_pe);
            for (uint32_t i = 0u; i < 4u; ++i)
                aclshmem_uint64_p(&completion->reserved[i], 0u, owner_pe);
            aclshmem_quiet();
            aclshmem_uint64_p(&completion->generation, generation, owner_pe);
            aclshmem_quiet();
        }
    }
}

extern "C" void launch_inc_dc_sparse_combine_device_e2e(
    uint32_t block_dim, void *stream, uint8_t *symmetric_token_ids,
    uint8_t *symmetric_partials, uint8_t *reduced_output,
    uint8_t *descriptor_mailbox, uint8_t *ack_mailbox,
    uint8_t *completion_mailbox, uint8_t *journal_header,
    uint8_t *journal_entries, uint8_t *journal_hash,
    uint8_t *journal_row_map, uint8_t *combine_row_map,
    uint8_t *destination_rows,
    uint8_t *inc_token_ids,
    uint8_t *status_line, uint64_t ffts_addr,
    uint64_t journal_capacity, uint64_t journal_hash_capacity,
    uint64_t row_capacity, uint32_t hidden, uint32_t worker_count,
    int32_t inc_pe, uint64_t generation, uint32_t wave, int32_t delay_rank,
    uint64_t delay_cycles, uint32_t combine_flags)
{
    inc_dc_sparse_combine_device_e2e_kernel<<<block_dim, nullptr, stream>>>(
        symmetric_token_ids, symmetric_partials, reduced_output,
        descriptor_mailbox, ack_mailbox, completion_mailbox, journal_header,
        journal_entries, journal_hash, journal_row_map, combine_row_map,
        destination_rows, inc_token_ids, status_line, ffts_addr,
        journal_capacity, journal_hash_capacity, row_capacity, hidden,
        worker_count, inc_pe, generation, wave, delay_rank, delay_cycles,
        combine_flags);
}
