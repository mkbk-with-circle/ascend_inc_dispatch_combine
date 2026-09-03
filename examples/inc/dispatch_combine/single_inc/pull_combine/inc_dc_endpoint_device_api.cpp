#include "inc_dc_endpoint_device_api.h"

#include "inc_dc_endpoint_dispatch_abi.h"
#include "inc_dc_sparse_combine_abi.h"

extern "C" void launch_inc_dc_endpoint_dispatch_device_e2e(
    uint32_t block_dim, void *stream, uint8_t *source_packet,
    uint8_t *inc_packets, uint8_t *commit_mailbox, uint8_t *recv_hidden,
    uint8_t *recv_rows, uint8_t *recv_assignments, uint8_t *recv_counts,
    uint8_t *ack_mailbox, uint8_t *completion_mailbox, uint8_t *cursors,
    uint8_t *hidden_staging, uint8_t *journal_header, uint8_t *status_line,
    uint64_t ffts_addr, uint64_t slot_bytes,
    uint64_t staging_bytes_per_destination, uint64_t row_capacity,
    uint64_t assignment_capacity, uint32_t hidden, uint32_t dtype,
    uint32_t expert_count, uint32_t worker_count, int32_t inc_pe,
    uint64_t generation, uint64_t sequence, uint32_t wave,
    uint32_t ring_slot);

extern "C" void launch_inc_dc_device_journal_index(
    uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,
    uint8_t *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, int32_t,
    uint64_t, uint32_t);

extern "C" void launch_inc_dc_sparse_combine_device_e2e(
    uint32_t block_dim, void *stream, uint8_t *symmetric_token_ids,
    uint8_t *symmetric_partials, uint8_t *reduced_output,
    uint8_t *descriptor_mailbox, uint8_t *ack_mailbox,
    uint8_t *completion_mailbox, uint8_t *journal_header,
    uint8_t *journal_entries, uint8_t *journal_hash,
    uint8_t *journal_row_map, uint8_t *combine_row_map,
    uint8_t *destination_rows, uint8_t *inc_token_ids,
    uint8_t *status_line, uint64_t ffts_addr, uint64_t journal_capacity,
    uint64_t journal_hash_capacity, uint64_t row_capacity,
    uint64_t token_ids_region_bytes, uint64_t partials_region_bytes,
    uint32_t hidden, uint32_t worker_count, int32_t inc_pe,
    uint64_t generation, uint64_t sequence, uint32_t wave,
    uint32_t ring_slot, int32_t delay_rank, uint64_t delay_cycles,
    uint32_t combine_flags);

namespace inc::dc::pull_combine {

namespace {

bool PowerOfTwo(uint64_t value)
{
    return value != 0u && (value & (value - 1u)) == 0u;
}

bool ValidWorkers(uint32_t workers, int32_t inc_pe)
{
    return workers >= 2u && workers <= kEndpointDispatchMaxWorkers &&
        workers <= kSparseCombineMaxWorkers &&
        inc_pe == static_cast<int32_t>(workers);
}

} // namespace

uint64_t SparseCombineControlBytes(uint32_t worker_count)
{
    const uint64_t bytes =
        static_cast<uint64_t>(worker_count + 2u) * sizeof(uint32_t);
    return (bytes + 63u) / 64u * 64u;
}

EndpointLaunchStatus LaunchEndpointDispatch(
    uint32_t aiv_count, void *stream, const EndpointDispatchDeviceArgs &a)
{
    if (aiv_count == 0u || stream == nullptr || a.source_packet == nullptr ||
        a.inc_packets == nullptr || a.commit_mailbox == nullptr ||
        a.recv_hidden == nullptr || a.recv_rows == nullptr ||
        a.recv_assignments == nullptr || a.recv_counts == nullptr ||
        a.ack_mailbox == nullptr || a.completion_mailbox == nullptr ||
        a.cursors == nullptr ||
        (a.staging_bytes_per_aiv != 0u && a.hidden_staging == nullptr) ||
        a.journal_header == nullptr || a.status_line == nullptr ||
        a.ffts_addr == 0u ||
        a.slot_bytes < sizeof(EndpointDispatchPacketHeader) ||
        a.slot_bytes % kEndpointDispatchAlignment != 0u ||
        a.row_capacity == 0u ||
        a.assignment_capacity == 0u || a.hidden == 0u ||
        (a.dtype != static_cast<uint32_t>(EndpointDataType::FP16) &&
         a.dtype != static_cast<uint32_t>(EndpointDataType::BF16) &&
         a.dtype != static_cast<uint32_t>(EndpointDataType::FP32)) ||
        a.expert_count == 0u || !ValidWorkers(a.worker_count, a.inc_pe) ||
        a.generation == 0u || a.sequence == 0u)
        return EndpointLaunchStatus::INVALID_ARGUMENT;
    launch_inc_dc_endpoint_dispatch_device_e2e(
        aiv_count, stream, a.source_packet, a.inc_packets, a.commit_mailbox,
        a.recv_hidden, a.recv_rows, a.recv_assignments, a.recv_counts,
        a.ack_mailbox, a.completion_mailbox, a.cursors, a.hidden_staging,
        a.journal_header, a.status_line, a.ffts_addr, a.slot_bytes,
        a.staging_bytes_per_aiv, a.row_capacity, a.assignment_capacity,
        a.hidden, a.dtype, a.expert_count, a.worker_count, a.inc_pe,
        a.generation, a.sequence, a.wave, a.ring_slot);
    return EndpointLaunchStatus::OK;
}

EndpointLaunchStatus LaunchDeviceJournalIndex(
    void *stream, const DeviceJournalIndexArgs &a)
{
    if (stream == nullptr || a.inc_packets == nullptr ||
        a.journal_header == nullptr || a.journal_entries == nullptr ||
        a.journal_hash == nullptr || a.journal_row_map == nullptr ||
        a.destination_rows == nullptr || a.ffts_addr == 0u ||
        a.slot_bytes == 0u || a.entry_capacity == 0u ||
        !PowerOfTwo(a.hash_capacity) ||
        a.hash_capacity < a.entry_capacity ||
        !ValidWorkers(a.worker_count, a.inc_pe) || a.generation == 0u)
        return EndpointLaunchStatus::INVALID_ARGUMENT;
    launch_inc_dc_device_journal_index(
        1u, stream, a.inc_packets, a.journal_header, a.journal_entries,
        a.journal_hash, a.journal_row_map, a.destination_rows, a.ffts_addr,
        a.slot_bytes, a.entry_capacity, a.hash_capacity, a.worker_count,
        a.inc_pe, a.generation, a.wave);
    return EndpointLaunchStatus::OK;
}

EndpointLaunchStatus LaunchSparseCombine(
    uint32_t aiv_count, void *stream, const SparseCombineDeviceArgs &a)
{
    if (aiv_count == 0u || stream == nullptr ||
        a.symmetric_token_ids == nullptr || a.symmetric_partials == nullptr ||
        a.reduced_output == nullptr || a.descriptor_mailbox == nullptr ||
        a.ack_mailbox == nullptr || a.completion_mailbox == nullptr ||
        a.journal_header == nullptr || a.journal_entries == nullptr ||
        a.journal_hash == nullptr || a.journal_row_map == nullptr ||
        a.combine_row_map == nullptr || a.destination_rows == nullptr ||
        a.inc_token_ids == nullptr || a.status_line == nullptr ||
        a.ffts_addr == 0u || a.journal_capacity == 0u ||
        !PowerOfTwo(a.journal_hash_capacity) ||
        a.journal_hash_capacity < a.journal_capacity ||
        a.row_capacity == 0u ||
        a.token_ids_region_bytes == 0u || a.partials_region_bytes == 0u ||
        a.hidden == 0u || !ValidWorkers(a.worker_count, a.inc_pe) ||
        a.generation == 0u || a.sequence == 0u ||
        a.delay_rank < -1 || a.delay_rank >=
            static_cast<int32_t>(a.worker_count) ||
        (a.delay_rank == -1 && a.delay_cycles != 0u) ||
        (a.flags & ~kSparseCombineFlagCanonicalRows) != 0u)
        return EndpointLaunchStatus::INVALID_ARGUMENT;
    launch_inc_dc_sparse_combine_device_e2e(
        aiv_count, stream, a.symmetric_token_ids, a.symmetric_partials,
        a.reduced_output, a.descriptor_mailbox, a.ack_mailbox,
        a.completion_mailbox, a.journal_header, a.journal_entries,
        a.journal_hash, a.journal_row_map, a.combine_row_map,
        a.destination_rows, a.inc_token_ids, a.status_line, a.ffts_addr,
        a.journal_capacity, a.journal_hash_capacity, a.row_capacity,
        a.token_ids_region_bytes, a.partials_region_bytes, a.hidden,
        a.worker_count, a.inc_pe, a.generation, a.sequence, a.wave,
        a.ring_slot, a.delay_rank, a.delay_cycles, a.flags);
    return EndpointLaunchStatus::OK;
}

} // namespace inc::dc::pull_combine
