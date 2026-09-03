#ifndef INC_DC_ENDPOINT_DEVICE_API_H
#define INC_DC_ENDPOINT_DEVICE_API_H

#include <cstdint>

namespace inc::dc::pull_combine {

enum class EndpointLaunchStatus : uint32_t {
    OK = 0u,
    INVALID_ARGUMENT,
};

// These argument objects only borrow caller-owned symmetric HBM.  The three
// launch functions are asynchronous with respect to the host and never
// allocate, synchronize, or initialize SHMEM.  A runtime can therefore keep
// one SHMEM session/workspace and submit waves directly on its own stream.
struct EndpointDispatchDeviceArgs {
    uint8_t *source_packet = nullptr;
    uint8_t *inc_packets = nullptr;
    uint8_t *commit_mailbox = nullptr;
    uint8_t *recv_hidden = nullptr;
    uint8_t *recv_rows = nullptr;
    uint8_t *recv_assignments = nullptr;
    uint8_t *recv_counts = nullptr;
    uint8_t *ack_mailbox = nullptr;
    uint8_t *completion_mailbox = nullptr;
    uint8_t *cursors = nullptr;
    uint8_t *hidden_staging = nullptr;
    uint8_t *upload_ready = nullptr;
    uint8_t *journal_header = nullptr;
    uint8_t *status_line = nullptr;
    uint64_t ffts_addr = 0u;
    uint64_t slot_bytes = 0u;
    uint64_t staging_bytes_per_aiv = 0u;
    uint64_t upload_chunk_bytes = 0u;
    uint32_t upload_chunks_per_source = 0u;
    uint64_t row_capacity = 0u;
    uint64_t assignment_capacity = 0u;
    uint32_t hidden = 0u;
    uint32_t dtype = 0u;
    uint32_t expert_count = 0u;
    uint32_t worker_count = 0u;
    int32_t inc_pe = -1;
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint32_t wave = 0u;
    uint32_t ring_slot = 0u;
};

struct DeviceJournalIndexArgs {
    uint8_t *inc_packets = nullptr;
    uint8_t *journal_header = nullptr;
    uint8_t *journal_entries = nullptr;
    uint8_t *journal_hash = nullptr;
    uint8_t *journal_row_map = nullptr;
    uint8_t *destination_rows = nullptr;
    uint64_t ffts_addr = 0u;
    uint64_t slot_bytes = 0u;
    uint64_t entry_capacity = 0u;
    uint64_t hash_capacity = 0u;
    uint32_t worker_count = 0u;
    int32_t inc_pe = -1;
    uint64_t generation = 0u;
    uint32_t wave = 0u;
};

struct SparseCombineDeviceArgs {
    uint8_t *symmetric_token_ids = nullptr;
    uint8_t *symmetric_partials = nullptr;
    uint8_t *reduced_output = nullptr;
    uint8_t *descriptor_mailbox = nullptr;
    uint8_t *ack_mailbox = nullptr;
    uint8_t *completion_mailbox = nullptr;
    uint8_t *journal_header = nullptr;
    uint8_t *journal_entries = nullptr;
    uint8_t *journal_hash = nullptr;
    uint8_t *journal_row_map = nullptr;
    uint8_t *combine_row_map = nullptr;
    uint8_t *destination_rows = nullptr;
    uint8_t *inc_token_ids = nullptr;
    uint8_t *status_line = nullptr;
    uint64_t ffts_addr = 0u;
    uint64_t journal_capacity = 0u;
    uint64_t journal_hash_capacity = 0u;
    uint64_t row_capacity = 0u;
    uint64_t token_ids_region_bytes = 0u;
    uint64_t partials_region_bytes = 0u;
    uint32_t hidden = 0u;
    uint32_t worker_count = 0u;
    int32_t inc_pe = -1;
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint32_t wave = 0u;
    uint32_t ring_slot = 0u;
    int32_t delay_rank = -1;
    uint64_t delay_cycles = 0u;
    uint32_t flags = 0u;
};

EndpointLaunchStatus LaunchEndpointDispatch(
    uint32_t aiv_count, void *stream, const EndpointDispatchDeviceArgs &args);
EndpointLaunchStatus LaunchDeviceJournalIndex(
    void *stream, const DeviceJournalIndexArgs &args);
EndpointLaunchStatus LaunchSparseCombine(
    uint32_t aiv_count, void *stream, const SparseCombineDeviceArgs &args);

// Required bytes for status + initialization + one ready word per source.
uint64_t SparseCombineControlBytes(uint32_t worker_count);

} // namespace inc::dc::pull_combine

#endif
