#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include "acl/acl.h"
#include "shmem.h"
#include "utils.h"

#include "inc_dc_endpoint_dispatch_packet.h"
#include "inc_dc_device_journal_abi.h"
#include "inc_dc_sparse_combine_abi.h"

using namespace inc::dc::pull_combine;

extern "C" void launch_inc_dc_endpoint_dispatch_device_e2e(
    uint32_t block_dim, void *stream, uint8_t *source_packet,
    uint8_t *inc_packets, uint8_t *commit_mailbox, uint8_t *recv_hidden,
    uint8_t *recv_rows, uint8_t *recv_assignments, uint8_t *recv_counts,
    uint8_t *ack_mailbox, uint8_t *completion_mailbox, uint8_t *cursors,
    uint8_t *hidden_staging, uint8_t *journal_header,
    uint8_t *status_line,
    uint64_t ffts_addr,
    uint64_t slot_bytes, uint64_t staging_bytes_per_destination,
    uint64_t row_capacity, uint64_t assignment_capacity,
    uint32_t hidden,
    uint32_t dtype, uint32_t expert_count, uint32_t worker_count,
    int32_t inc_pe, uint64_t generation, uint64_t sequence, uint32_t wave,
    uint32_t ring_slot);

extern "C" void launch_inc_dc_device_journal_index(
    uint32_t block_dim, void *stream, uint8_t *inc_packets,
    uint8_t *journal_header, uint8_t *journal_entries,
    uint8_t *journal_hash, uint8_t *journal_row_map,
    uint8_t *destination_rows, uint64_t ffts_addr, uint64_t slot_bytes,
    uint64_t entry_capacity, uint64_t hash_capacity, uint32_t worker_count,
    int32_t inc_pe, uint64_t generation, uint32_t wave);

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
    int32_t inc_pe, uint64_t generation, uint64_t sequence, uint32_t wave,
    uint32_t ring_slot, int32_t delay_rank, uint64_t delay_cycles,
    uint32_t combine_flags);

int g_npus = 5;
const char *ipport = "tcp://127.0.0.1:28780";
int f_pe = 0;
int f_npu = 0;
aclshmemx_uniqueid_t default_flag_uid;

namespace {

constexpr uint64_t kGeneration = 17u;
constexpr uint32_t kWave = 5u;
constexpr uint32_t kExpertCount = 64u;

void HashBytes(uint64_t *hash, const void *data, uint64_t bytes)
{
    constexpr uint64_t kPrime = 1099511628211ull;
    const uint8_t *p = static_cast<const uint8_t *>(data);
    for (uint64_t i = 0u; i < bytes; ++i) {
        *hash ^= p[i];
        *hash *= kPrime;
    }
}

uint64_t RecomputeMetadataDigest(std::vector<uint8_t> *packet)
{
    constexpr uint64_t kOffset = 1469598103934665603ull;
    EndpointDispatchPacketHeader *header =
        reinterpret_cast<EndpointDispatchPacketHeader *>(packet->data());
    EndpointDispatchPacketHeader canonical = *header;
    canonical.metadata_digest = 0u;
    uint64_t hash = kOffset;
    HashBytes(&hash, &canonical, sizeof(canonical));
    HashBytes(&hash, packet->data() + header->counts_offset,
              static_cast<uint64_t>(header->worker_count) *
                  sizeof(uint32_t) * 2u);
    HashBytes(&hash, packet->data() + header->tokens_offset,
              static_cast<uint64_t>(header->token_count) *
                  sizeof(EndpointDispatchTokenRecord));
    HashBytes(&hash, packet->data() + header->assignments_offset,
              static_cast<uint64_t>(header->assignment_count) *
                  sizeof(EndpointDispatchAssignmentRecord));
    header->metadata_digest = hash;
    return hash;
}

uint64_t NextPowerOfTwo(uint64_t value)
{
    uint64_t result = 1u;
    while (result < value) result <<= 1u;
    return result;
}

uint64_t HashToken(uint64_t value)
{
    value ^= value >> 33u;
    value *= 0xff51afd7ed558ccdull;
    value ^= value >> 33u;
    value *= 0xc4ceb9fe1a85ec53ull;
    return value ^ (value >> 33u);
}

uint8_t HiddenByte(uint32_t source, uint32_t token, uint64_t byte)
{
    return static_cast<uint8_t>(
        (static_cast<uint64_t>(source) * 131u +
         static_cast<uint64_t>(token) * 17u + byte) % 251u);
}

float PartialValue(uint32_t expert_worker, uint64_t token_id,
                   uint32_t element)
{
    return static_cast<float>(expert_worker + 1u) +
        static_cast<float>(token_id % 17u) * 0.03125f +
        static_cast<float>(element % 11u) * 0.001953125f;
}

EndpointDispatchInput MakeInput(uint32_t source, uint32_t workers,
                                uint32_t tokens, uint32_t hidden,
                                uint32_t topk)
{
    EndpointDispatchInput input{};
    input.config.worker_count = workers;
    input.config.expert_count = kExpertCount;
    input.config.hidden = hidden;
    input.config.dtype = EndpointDataType::BF16;
    input.config.wave = kWave;
    input.config.source_rank = source;
    input.config.generation = kGeneration;
    input.config.sequence = 1u;
    input.assignment_offsets.push_back(0u);
    for (uint32_t token = 0u; token < tokens; ++token) {
        input.token_ids.push_back(
            static_cast<uint64_t>(source) * 1000000u + token + 1u);
        for (uint32_t k = 0u; k < topk; ++k) {
            EndpointDispatchAssignmentRecord assignment{};
            // Adjacent assignments intentionally share a destination.  This
            // verifies that INC sends one hidden row, not one per expert.
            assignment.destination_rank =
                (source + token + k / 2u) % workers;
            assignment.expert_id =
                (source * 11u + token * 7u + k) % kExpertCount;
            assignment.ordinal = k;
            assignment.weight = static_cast<float>(k + 1u) /
                static_cast<float>(topk + 1u);
            input.assignments.push_back(assignment);
        }
        input.assignment_offsets.push_back(
            static_cast<uint32_t>(input.assignments.size()));
    }
    const uint64_t row_bytes = static_cast<uint64_t>(hidden) * 2u;
    input.hidden_payload.resize(static_cast<size_t>(tokens * row_bytes));
    for (uint32_t token = 0u; token < tokens; ++token) {
        for (uint64_t byte = 0u; byte < row_bytes; ++byte) {
            input.hidden_payload[static_cast<size_t>(token * row_bytes + byte)]
                = HiddenByte(source, token, byte);
        }
    }
    return input;
}

int Fail(const char *step, int status)
{
    std::cerr << "[FAIL] " << step << " status=" << status << '\n';
    return status == 0 ? 1 : status;
}

template <typename T>
bool CopyFromDevice(std::vector<T> *host, uint8_t *device, uint64_t count)
{
    host->resize(static_cast<size_t>(count));
    if (count == 0u) return true;
    return aclrtMemcpy(host->data(), count * sizeof(T), device,
                       count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST) == 0;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 8 || argc > 13) {
        std::cerr << "usage: " << argv[0]
                  << " <workers> <pe> <ipport> <first_npu> <tokens>"
                     " <hidden> <topk> [aiv; 0=half of live AIV]"
                     " [fault; 0=none, 1=digest, 2=duplicate ordinal,"
                     " 3=nonfinite weight, 4=count mismatch,"
                     " 5=combine descriptor, 6=combine token ID]"
                     " [delay_rank; -1=none] [device_delay_cycles]"
                     " [reorder_combine_rows; 0|1]\n";
        return 2;
    }
    const uint32_t workers = static_cast<uint32_t>(std::strtoul(
        argv[1], nullptr, 10));
    const int pe = std::atoi(argv[2]);
    ipport = argv[3];
    f_npu = std::atoi(argv[4]);
    const uint32_t tokens = static_cast<uint32_t>(std::strtoul(
        argv[5], nullptr, 10));
    const uint32_t hidden = static_cast<uint32_t>(std::strtoul(
        argv[6], nullptr, 10));
    const uint32_t topk = static_cast<uint32_t>(std::strtoul(
        argv[7], nullptr, 10));
    const uint32_t requested_aiv = argc >= 9
        ? static_cast<uint32_t>(std::strtoul(argv[8], nullptr, 10)) : 0u;
    const uint32_t fault = argc >= 10
        ? static_cast<uint32_t>(std::strtoul(argv[9], nullptr, 10)) : 0u;
    const int32_t delay_rank = argc >= 11 ? std::atoi(argv[10]) : -1;
    const uint64_t delay_cycles = argc >= 12
        ? std::strtoull(argv[11], nullptr, 10) : 0u;
    const uint32_t reorder_combine_rows = argc >= 13
        ? static_cast<uint32_t>(std::strtoul(argv[12], nullptr, 10)) : 0u;
    const uint32_t pes = workers + 1u;
    const int inc_pe = static_cast<int>(workers);
    g_npus = static_cast<int>(pes);
    if (workers < 2u || workers > 8u || pe < 0 ||
        pe >= static_cast<int>(pes) || hidden == 0u ||
        (tokens != 0u && topk == 0u) || topk > 32u || fault > 6u ||
        (tokens == 0u && fault != 0u) || delay_rank < -1 ||
        delay_rank >= static_cast<int32_t>(workers) ||
        (delay_rank == -1 && delay_cycles != 0u) ||
        reorder_combine_rows > 1u) {
        return Fail("arguments", 2);
    }

    const uint32_t packet_source = pe < inc_pe
        ? static_cast<uint32_t>(pe) : 0u;
    const EndpointDispatchInput local_input = MakeInput(
        packet_source, workers, tokens, hidden, topk);
    std::vector<uint8_t> local_packet;
    EndpointDispatchCommit local_commit{};
    std::string packet_error;
    if (BuildEndpointDispatchPacket(local_input, &local_packet,
                                    &local_commit, &packet_error) !=
        EndpointDispatchStatus::OK) {
        std::cerr << packet_error << '\n';
        return Fail("build packet", 2);
    }
    const bool expect_reject = fault >= 1u && fault <= 4u;
    const bool expect_combine_reject = fault >= 5u;
    const uint32_t expected_combine_status = fault == 5u ? 2u :
        (fault == 6u ? 5u : 0u);
    if (expect_reject && pe == 0) {
        EndpointDispatchPacketHeader *header =
            reinterpret_cast<EndpointDispatchPacketHeader *>(
                local_packet.data());
        if (fault == 1u) {
            local_packet[header->tokens_offset] ^= 0x1u;
        } else if (fault == 2u) {
            EndpointDispatchAssignmentRecord *assignment =
                reinterpret_cast<EndpointDispatchAssignmentRecord *>(
                    local_packet.data() + header->assignments_offset);
            assignment[1].ordinal = assignment[0].ordinal;
        } else if (fault == 3u) {
            EndpointDispatchAssignmentRecord *assignment =
                reinterpret_cast<EndpointDispatchAssignmentRecord *>(
                    local_packet.data() + header->assignments_offset);
            assignment[0].weight =
                std::numeric_limits<float>::infinity();
        } else {
            uint32_t *counts = reinterpret_cast<uint32_t *>(
                local_packet.data() + header->counts_offset);
            ++counts[0];
        }
        if (fault != 1u)
            local_commit.metadata_digest =
                RecomputeMetadataDigest(&local_packet);
    }
    const uint64_t slot_bytes = local_packet.size();
    const uint64_t row_bytes = static_cast<uint64_t>(hidden) * 2u;
    const uint64_t row_capacity = std::max<uint64_t>(
        1u, static_cast<uint64_t>(workers) * tokens);
    const uint64_t assignment_capacity = std::max<uint64_t>(
        1u, static_cast<uint64_t>(workers) * tokens * topk);
    const uint64_t journal_capacity = row_capacity;
    const uint64_t journal_hash_capacity = NextPowerOfTwo(
        journal_capacity * 2u);
    std::vector<uint32_t> combine_rows_per_worker(workers, 0u);
    std::vector<std::vector<uint64_t>> combine_ids_by_worker(workers);
    for (uint32_t source = 0u; source < workers; ++source) {
        const EndpointDispatchInput input = MakeInput(
            source, workers, tokens, hidden, topk);
        for (uint32_t token = 0u; token < tokens; ++token) {
            std::vector<uint8_t> seen(workers, 0u);
            for (uint32_t assignment = input.assignment_offsets[token];
                 assignment < input.assignment_offsets[token + 1u];
                 ++assignment)
                seen[input.assignments[assignment].destination_rank] = 1u;
            for (uint32_t destination = 0u; destination < workers;
                 ++destination) {
                if (seen[destination] == 0u) continue;
                combine_ids_by_worker[destination].push_back(
                    input.token_ids[token]);
            }
        }
    }
    uint64_t combine_row_capacity = 1u;
    for (uint32_t worker = 0u; worker < workers; ++worker) {
        combine_rows_per_worker[worker] =
            static_cast<uint32_t>(combine_ids_by_worker[worker].size());
        combine_row_capacity = std::max<uint64_t>(
            combine_row_capacity, combine_rows_per_worker[worker]);
    }
    const uint64_t counts_bytes =
        static_cast<uint64_t>(workers) * sizeof(uint32_t) * 4u;
    const uint64_t combine_control_bytes =
        (static_cast<uint64_t>(workers + 2u) * sizeof(uint32_t) + 63u) /
        64u * 64u;
    constexpr uint64_t kMaxStagingBytesPerDestination = 4ull << 20;
    const uint64_t staging_bytes_per_destination =
        kMaxStagingBytesPerDestination;

    const int32_t device = pe + f_npu;
    aclrtStream stream = nullptr;
    bool shmem_initialized = false;
    uint8_t *source_packet = nullptr;
    uint8_t *inc_packets = nullptr;
    uint8_t *commits = nullptr;
    uint8_t *recv_hidden = nullptr;
    uint8_t *recv_rows = nullptr;
    uint8_t *recv_assignments = nullptr;
    uint8_t *recv_counts = nullptr;
    uint8_t *acks = nullptr;
    uint8_t *completions = nullptr;
    uint8_t *cursors = nullptr;
    uint8_t *hidden_staging = nullptr;
    uint8_t *journal_header = nullptr;
    uint8_t *journal_entries = nullptr;
    uint8_t *journal_hash = nullptr;
    uint8_t *journal_row_map = nullptr;
    uint8_t *combine_row_map = nullptr;
    uint8_t *destination_rows = nullptr;
    uint8_t *combine_token_ids = nullptr;
    uint8_t *combine_partials = nullptr;
    uint8_t *combine_output = nullptr;
    uint8_t *combine_descriptors = nullptr;
    uint8_t *combine_acks = nullptr;
    uint8_t *combine_completions = nullptr;
    uint8_t *inc_combine_token_ids = nullptr;
    uint8_t *combine_status_line = nullptr;
    uint8_t *status_line = nullptr;
    double e2e_us = 0.0;
    double index_us = 0.0;
    double combine_us = 0.0;
    uint32_t dispatch_aiv = 0u;

    int status = aclInit(nullptr);
    if (status == 0) status = aclrtSetDevice(device);
    int64_t live_aiv = 0;
    if (status == 0) {
        status = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                    &live_aiv);
    }
    if (status == 0) {
        const uint32_t half_aiv = static_cast<uint32_t>(live_aiv / 2);
        dispatch_aiv = requested_aiv == 0u ? half_aiv : requested_aiv;
        if (dispatch_aiv == 0u || dispatch_aiv > live_aiv)
            status = 2;
    }
    if (status == 0) status = aclrtCreateStream(&stream);
    if (status == 0) {
        aclshmemx_init_attr_t attr;
        test_set_attr(pe, pes, 2ull * 1024ull * 1024ull * 1024ull, ipport,
                      default_flag_uid, &attr);
        status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
        shmem_initialized = status == 0;
    }
    if (status == 0) {
        source_packet = static_cast<uint8_t *>(aclshmem_malloc(slot_bytes));
        inc_packets = static_cast<uint8_t *>(aclshmem_malloc(
            slot_bytes * workers));
        commits = static_cast<uint8_t *>(aclshmem_malloc(
            sizeof(EndpointDispatchCommit) * workers));
        recv_hidden = static_cast<uint8_t *>(aclshmem_malloc(
            row_capacity * row_bytes));
        recv_rows = static_cast<uint8_t *>(aclshmem_malloc(
            row_capacity * sizeof(EndpointDispatchFanoutRecord)));
        recv_assignments = static_cast<uint8_t *>(aclshmem_malloc(
            assignment_capacity *
                sizeof(EndpointDispatchAssignmentRecord)));
        recv_counts = static_cast<uint8_t *>(aclshmem_malloc(counts_bytes));
        acks = static_cast<uint8_t *>(aclshmem_malloc(
            sizeof(EndpointDispatchAck) * workers));
        completions = static_cast<uint8_t *>(aclshmem_malloc(
            sizeof(EndpointDispatchReceiveCompletion) * workers));
        cursors = static_cast<uint8_t *>(aclshmem_malloc(
            static_cast<uint64_t>(workers) * workers * 64u));
        hidden_staging = staging_bytes_per_destination == 0u
            ? nullptr
            : static_cast<uint8_t *>(aclshmem_malloc(
                  staging_bytes_per_destination * dispatch_aiv));
        journal_header = static_cast<uint8_t *>(aclshmem_malloc(
            sizeof(DeviceJournalHeader)));
        journal_entries = static_cast<uint8_t *>(aclshmem_malloc(
            journal_capacity * sizeof(DeviceJournalEntry)));
        journal_hash = static_cast<uint8_t *>(aclshmem_malloc(
            journal_hash_capacity * sizeof(uint32_t)));
        journal_row_map = static_cast<uint8_t *>(aclshmem_malloc(
            static_cast<uint64_t>(workers) * journal_capacity *
            sizeof(uint32_t)));
        combine_row_map = static_cast<uint8_t *>(aclshmem_malloc(
            static_cast<uint64_t>(workers) * journal_capacity *
            sizeof(uint32_t)));
        destination_rows = static_cast<uint8_t *>(aclshmem_malloc(
            static_cast<uint64_t>(workers) * sizeof(uint32_t)));
        combine_token_ids = static_cast<uint8_t *>(aclshmem_malloc(
            combine_row_capacity * sizeof(uint64_t)));
        combine_partials = static_cast<uint8_t *>(aclshmem_malloc(
            combine_row_capacity * hidden * sizeof(float)));
        combine_output = static_cast<uint8_t *>(aclshmem_malloc(
            std::max<uint64_t>(1u, static_cast<uint64_t>(tokens) * hidden) *
            sizeof(float)));
        combine_descriptors = static_cast<uint8_t *>(aclshmem_malloc(
            static_cast<uint64_t>(workers) *
            sizeof(SparseCombineReadyDescriptor)));
        combine_acks = static_cast<uint8_t *>(aclshmem_malloc(
            static_cast<uint64_t>(workers) *
            sizeof(SparseCombineDeviceAck)));
        combine_completions = static_cast<uint8_t *>(aclshmem_malloc(
            static_cast<uint64_t>(workers) *
            sizeof(SparseCombineEgressCompletion)));
        inc_combine_token_ids = static_cast<uint8_t *>(aclshmem_malloc(
            static_cast<uint64_t>(workers) * combine_row_capacity *
            sizeof(uint64_t)));
        combine_status_line = static_cast<uint8_t *>(
            aclshmem_malloc(combine_control_bytes));
        status_line = static_cast<uint8_t *>(aclshmem_malloc(64u));
        if (source_packet == nullptr || inc_packets == nullptr ||
            commits == nullptr || recv_hidden == nullptr ||
            recv_rows == nullptr || recv_assignments == nullptr ||
            recv_counts == nullptr || acks == nullptr ||
            completions == nullptr || cursors == nullptr ||
            (staging_bytes_per_destination != 0u &&
             hidden_staging == nullptr) ||
            journal_header == nullptr || journal_entries == nullptr ||
            journal_hash == nullptr || journal_row_map == nullptr ||
            combine_row_map == nullptr ||
            destination_rows == nullptr ||
            combine_token_ids == nullptr || combine_partials == nullptr ||
            combine_output == nullptr || combine_descriptors == nullptr ||
            combine_acks == nullptr || combine_completions == nullptr ||
            inc_combine_token_ids == nullptr ||
            combine_status_line == nullptr ||
            status_line == nullptr)
            status = 1;
    }

    std::vector<uint8_t> zero;
    auto ZeroDevice = [&](uint8_t *pointer, uint64_t bytes) {
        if (status != 0) return;
        zero.assign(static_cast<size_t>(bytes), 0u);
        status = aclrtMemcpy(pointer, bytes, zero.data(), bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE);
    };
    if (status == 0) {
        ZeroDevice(source_packet, slot_bytes);
        ZeroDevice(inc_packets, slot_bytes * workers);
        ZeroDevice(commits, sizeof(EndpointDispatchCommit) * workers);
        ZeroDevice(recv_hidden, row_capacity * row_bytes);
        ZeroDevice(recv_rows,
                   row_capacity * sizeof(EndpointDispatchFanoutRecord));
        ZeroDevice(recv_assignments,
                   assignment_capacity *
                       sizeof(EndpointDispatchAssignmentRecord));
        ZeroDevice(recv_counts, counts_bytes);
        ZeroDevice(acks, sizeof(EndpointDispatchAck) * workers);
        ZeroDevice(completions,
                   sizeof(EndpointDispatchReceiveCompletion) * workers);
        ZeroDevice(cursors,
                   static_cast<uint64_t>(workers) * workers * 64u);
        if (hidden_staging != nullptr)
            ZeroDevice(hidden_staging,
                       staging_bytes_per_destination * dispatch_aiv);
        ZeroDevice(journal_header, sizeof(DeviceJournalHeader));
        ZeroDevice(journal_entries,
                   journal_capacity * sizeof(DeviceJournalEntry));
        std::vector<uint32_t> empty_hash(
            static_cast<size_t>(journal_hash_capacity),
            kDeviceJournalEmpty);
        if (status == 0)
            status = aclrtMemcpy(
                journal_hash, journal_hash_capacity * sizeof(uint32_t),
                empty_hash.data(), journal_hash_capacity * sizeof(uint32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);
        std::vector<uint32_t> empty_rows(
            static_cast<size_t>(workers * journal_capacity),
            kDeviceJournalEmpty);
        if (status == 0)
            status = aclrtMemcpy(
                journal_row_map,
                static_cast<uint64_t>(workers) * journal_capacity *
                    sizeof(uint32_t),
                empty_rows.data(),
                static_cast<uint64_t>(workers) * journal_capacity *
                    sizeof(uint32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);
        if (status == 0)
            status = aclrtMemcpy(
                combine_row_map,
                static_cast<uint64_t>(workers) * journal_capacity *
                    sizeof(uint32_t),
                empty_rows.data(),
                static_cast<uint64_t>(workers) * journal_capacity *
                    sizeof(uint32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);
        ZeroDevice(destination_rows,
                   static_cast<uint64_t>(workers) * sizeof(uint32_t));
        ZeroDevice(combine_token_ids,
                   combine_row_capacity * sizeof(uint64_t));
        ZeroDevice(combine_partials,
                   combine_row_capacity * hidden * sizeof(float));
        ZeroDevice(combine_output,
                   std::max<uint64_t>(
                       1u, static_cast<uint64_t>(tokens) * hidden) *
                       sizeof(float));
        ZeroDevice(combine_descriptors,
                   static_cast<uint64_t>(workers) *
                       sizeof(SparseCombineReadyDescriptor));
        ZeroDevice(combine_acks,
                   static_cast<uint64_t>(workers) *
                       sizeof(SparseCombineDeviceAck));
        ZeroDevice(combine_completions,
                   static_cast<uint64_t>(workers) *
                       sizeof(SparseCombineEgressCompletion));
        ZeroDevice(inc_combine_token_ids,
                   static_cast<uint64_t>(workers) * combine_row_capacity *
                       sizeof(uint64_t));
        ZeroDevice(combine_status_line, combine_control_bytes);
        ZeroDevice(status_line, 64u);
    }
    if (status == 0 && pe < inc_pe && !expect_reject) {
        std::vector<uint64_t> ids = combine_ids_by_worker[pe];
        if (reorder_combine_rows != 0u && ids.size() > 1u && fault != 6u) {
            const size_t rotate = (static_cast<size_t>(pe) + 1u) % ids.size();
            std::rotate(ids.begin(), ids.begin() + rotate, ids.end());
        }
        if (fault == 6u && pe == 0 && !ids.empty()) ids[0] ^= 1u;
        std::vector<float> partials(
            static_cast<size_t>(ids.size()) * hidden);
        for (uint32_t row = 0u; row < ids.size(); ++row)
            for (uint32_t element = 0u; element < hidden; ++element)
                partials[static_cast<uint64_t>(row) * hidden + element] =
                    PartialValue(pe, ids[row], element);
        if (!ids.empty())
            status = aclrtMemcpy(combine_token_ids,
                                 ids.size() * sizeof(uint64_t), ids.data(),
                                 ids.size() * sizeof(uint64_t),
                                 ACL_MEMCPY_HOST_TO_DEVICE);
        if (status == 0 && !partials.empty())
            status = aclrtMemcpy(
                combine_partials, partials.size() * sizeof(float),
                partials.data(), partials.size() * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);
        SparseCombineReadyDescriptor descriptor{};
        descriptor.generation = kGeneration;
        descriptor.sequence = 1u;
        descriptor.wave = kWave;
        descriptor.source_rank = static_cast<uint32_t>(pe);
        descriptor.row_count = ids.size();
        descriptor.hidden = hidden;
        descriptor.source_region_id = 1u;
        descriptor.payload_bytes =
            static_cast<uint64_t>(ids.size()) * hidden * sizeof(float);
        descriptor.metadata_digest = 1u + static_cast<uint32_t>(pe);
        descriptor.flags = reorder_combine_rows == 0u
            ? kSparseCombineFlagCanonicalRows : 0u;
        if (fault == 5u && pe == 0) ++descriptor.row_count;
        if (status == 0)
            status = aclrtMemcpy(
                combine_descriptors + static_cast<uint64_t>(pe) *
                    sizeof(descriptor),
                sizeof(descriptor), &descriptor, sizeof(descriptor),
                ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (status == 0 && pe < inc_pe) {
        status = aclrtMemcpy(source_packet, slot_bytes, local_packet.data(),
                             slot_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (status == 0) {
            status = aclrtMemcpy(
                commits + static_cast<uint64_t>(pe) * sizeof(local_commit),
                sizeof(local_commit), &local_commit, sizeof(local_commit),
                ACL_MEMCPY_HOST_TO_DEVICE);
        }
    }
    if (status == 0) aclshmem_barrier_all();

    if (status == 0) {
        const auto begin = std::chrono::steady_clock::now();
        launch_inc_dc_endpoint_dispatch_device_e2e(
            dispatch_aiv, stream, source_packet, inc_packets, commits,
            recv_hidden,
            recv_rows, recv_assignments, recv_counts, acks, completions,
            cursors, hidden_staging, journal_header, status_line,
            shmemx_get_ffts_config(),
            slot_bytes, staging_bytes_per_destination,
            row_capacity,
            assignment_capacity, hidden,
            static_cast<uint32_t>(EndpointDataType::BF16), kExpertCount,
            workers, inc_pe, kGeneration, 1u, kWave, 0u);
        status = aclrtSynchronizeStream(stream);
        const auto end = std::chrono::steady_clock::now();
        e2e_us = std::chrono::duration<double, std::micro>(end - begin).count();
    }
    if (status == 0) aclshmem_barrier_all();

    if (status == 0 && !expect_reject) {
        const auto begin = std::chrono::steady_clock::now();
        launch_inc_dc_device_journal_index(
            1u, stream, inc_packets, journal_header, journal_entries,
            journal_hash, journal_row_map, destination_rows,
            shmemx_get_ffts_config(), slot_bytes,
            journal_capacity, journal_hash_capacity, workers, inc_pe,
            kGeneration, kWave);
        status = aclrtSynchronizeStream(stream);
        const auto end = std::chrono::steady_clock::now();
        index_us = std::chrono::duration<double, std::micro>(end - begin)
            .count();
    }
    if (status == 0) aclshmem_barrier_all();

    if (status == 0 && !expect_reject) {
        const auto begin = std::chrono::steady_clock::now();
        launch_inc_dc_sparse_combine_device_e2e(
            dispatch_aiv, stream, combine_token_ids, combine_partials,
            combine_output, combine_descriptors, combine_acks,
            combine_completions, journal_header, journal_entries,
            journal_hash, journal_row_map, combine_row_map,
            destination_rows, inc_combine_token_ids, combine_status_line,
            shmemx_get_ffts_config(), journal_capacity,
            journal_hash_capacity, combine_row_capacity, hidden, workers,
            inc_pe, kGeneration, 1u, kWave, 0u, delay_rank, delay_cycles,
            reorder_combine_rows == 0u
                ? kSparseCombineFlagCanonicalRows : 0u);
        status = aclrtSynchronizeStream(stream);
        const auto end = std::chrono::steady_clock::now();
        combine_us = std::chrono::duration<double, std::micro>(end - begin)
            .count();
    }
    if (status == 0) aclshmem_barrier_all();

    bool correct = status == 0;
    if (correct && pe == inc_pe) {
        uint32_t device_status = std::numeric_limits<uint32_t>::max();
        status = aclrtMemcpy(&device_status, sizeof(device_status), status_line,
                             sizeof(device_status),
                             ACL_MEMCPY_DEVICE_TO_HOST);
        correct = status == 0 && device_status ==
            (expect_reject ? 3u : 0u);
        if (!correct)
            std::cerr << "[FAIL] INC device_status=" << device_status
                      << " expected=" << (expect_reject ? 3u : 0u) << '\n';
        if (expect_reject) {
            // The packet copy itself is intentionally corrupt; the gate is
            // that INC rejects it and publishes negative completions.
        }
        for (uint32_t source = 0u; correct && source < workers; ++source) {
            if (expect_reject) break;
            std::vector<uint8_t> expected_packet;
            EndpointDispatchCommit expected_commit{};
            std::string error;
            const EndpointDispatchInput expected_input = MakeInput(
                source, workers, tokens, hidden, topk);
            correct = BuildEndpointDispatchPacket(
                          expected_input, &expected_packet, &expected_commit,
                          &error) == EndpointDispatchStatus::OK;
            std::vector<uint8_t> actual_packet(expected_packet.size());
            if (correct) {
                status = aclrtMemcpy(
                    actual_packet.data(), actual_packet.size(),
                    inc_packets + static_cast<uint64_t>(source) * slot_bytes,
                    actual_packet.size(), ACL_MEMCPY_DEVICE_TO_HOST);
                correct = status == 0;
            }
            if (correct && actual_packet != expected_packet) {
                size_t mismatch = 0u;
                while (mismatch < actual_packet.size() &&
                       actual_packet[mismatch] == expected_packet[mismatch])
                    ++mismatch;
                std::cerr << "[FAIL] INC packet source=" << source
                          << " mismatch byte=" << mismatch
                          << " actual="
                          << static_cast<uint32_t>(actual_packet[mismatch])
                          << " expected="
                          << static_cast<uint32_t>(expected_packet[mismatch])
                          << '\n';
                correct = false;
            }
        }
        if (correct && !expect_reject) {
            DeviceJournalHeader actual_header{};
            std::vector<DeviceJournalEntry> actual_entries;
            std::vector<uint32_t> actual_hash;
            std::vector<uint32_t> actual_row_map;
            std::vector<uint32_t> actual_destination_rows;
            status = aclrtMemcpy(&actual_header,
                                 sizeof(actual_header), journal_header,
                                 sizeof(actual_header),
                                 ACL_MEMCPY_DEVICE_TO_HOST);
            const uint64_t actual_journal_size = actual_header.token_count;
            correct = status == 0 &&
                actual_header.magic == kDeviceJournalMagic &&
                actual_header.abi_version == kDeviceJournalAbiVersion &&
                actual_header.struct_bytes == sizeof(DeviceJournalHeader) &&
                actual_header.generation == kGeneration &&
                actual_header.wave == kWave &&
                actual_header.worker_count == workers &&
                actual_header.status == 0u &&
                actual_header.flags == kDeviceJournalIndexReady &&
                actual_journal_size ==
                    static_cast<uint64_t>(workers) * tokens &&
                CopyFromDevice(&actual_entries, journal_entries,
                               actual_journal_size) &&
                CopyFromDevice(&actual_hash, journal_hash,
                               journal_hash_capacity) &&
                CopyFromDevice(&actual_row_map, journal_row_map,
                               static_cast<uint64_t>(workers) *
                                   actual_journal_size) &&
                CopyFromDevice(&actual_destination_rows, destination_rows,
                               workers);
            if (!correct)
                std::cerr << "[FAIL] journal size=" << actual_journal_size
                          << " expected="
                          << static_cast<uint64_t>(workers) * tokens
                          << " copy_status=" << status << '\n';
            uint64_t entry_index = 0u;
            std::vector<uint32_t> expected_destination_rows(workers, 0u);
            for (uint32_t source = 0u; correct && source < workers;
                 ++source) {
                const EndpointDispatchInput input = MakeInput(
                    source, workers, tokens, hidden, topk);
                for (uint32_t token = 0u; correct && token < tokens;
                     ++token, ++entry_index) {
                    uint64_t expected[2]{0u, 0u};
                    for (uint32_t assignment =
                             input.assignment_offsets[token];
                         assignment < input.assignment_offsets[token + 1u];
                         ++assignment) {
                        const uint32_t destination =
                            input.assignments[assignment].destination_rank;
                        expected[destination >> 6u] |=
                            1ull << (destination & 63u);
                    }
                    const DeviceJournalEntry &entry =
                        actual_entries[entry_index];
                    correct = entry.token_id == input.token_ids[token] &&
                        entry.owner_rank == source &&
                        entry.owner_row == token &&
                        entry.expected[0] == expected[0] &&
                        entry.expected[1] == expected[1] &&
                        entry.received[0] == 0u &&
                        entry.received[1] == 0u &&
                        entry.accumulator_index == entry_index &&
                        entry.flags == 0u && entry.reserved == 0u;
                    for (uint32_t destination = 0u;
                         correct && destination < workers; ++destination) {
                        const bool selected =
                            (expected[destination >> 6u] &
                             (1ull << (destination & 63u))) != 0u;
                        const uint32_t actual = actual_row_map[
                            static_cast<uint64_t>(destination) *
                                actual_journal_size + entry_index];
                        if (selected) {
                            correct = actual ==
                                expected_destination_rows[destination]++;
                        } else {
                            correct = actual == kDeviceJournalEmpty;
                        }
                    }
                    uint64_t bucket = HashToken(entry.token_id) &
                        (journal_hash_capacity - 1u);
                    bool found = false;
                    for (uint64_t probe = 0u;
                         correct && probe < journal_hash_capacity; ++probe) {
                        const uint32_t candidate = actual_hash[bucket];
                        if (candidate == kDeviceJournalEmpty) break;
                        if (candidate == entry_index) {
                            found = true;
                            break;
                        }
                        bucket = (bucket + 1u) &
                            (journal_hash_capacity - 1u);
                    }
                    correct = correct && found;
                }
            }
            correct = correct &&
                actual_destination_rows == expected_destination_rows;
            if (!correct)
                std::cerr << "[FAIL] INC dynamic journal mismatch\n";
        }
    }
    if (correct && pe < inc_pe && expect_reject) {
        EndpointDispatchAck ack{};
        EndpointDispatchReceiveCompletion completion{};
        status = aclrtMemcpy(
            &ack, sizeof(ack),
            acks + static_cast<uint64_t>(pe) * sizeof(ack), sizeof(ack),
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (status == 0)
            status = aclrtMemcpy(
                &completion, sizeof(completion),
                completions + static_cast<uint64_t>(pe) * sizeof(completion),
                sizeof(completion), ACL_MEMCPY_DEVICE_TO_HOST);
        correct = status == 0 && ack.magic == kEndpointDispatchMagic &&
            ack.generation == kGeneration && ack.source_rank ==
                static_cast<uint32_t>(pe) && ack.status == 3u &&
            ack.tokens_consumed == 0u &&
            completion.magic == kEndpointDispatchMagic &&
            completion.generation == kGeneration &&
            completion.destination_rank == static_cast<uint32_t>(pe) &&
            completion.status == 3u && completion.row_count == 0u &&
            completion.assignment_count == 0u;
    }
    if (correct && pe < inc_pe && !expect_reject) {
        EndpointDispatchAck ack{};
        EndpointDispatchReceiveCompletion completion{};
        status = aclrtMemcpy(
            &ack, sizeof(ack),
            acks + static_cast<uint64_t>(pe) * sizeof(ack), sizeof(ack),
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (status == 0)
            status = aclrtMemcpy(
                &completion, sizeof(completion),
                completions + static_cast<uint64_t>(pe) * sizeof(completion),
                sizeof(completion), ACL_MEMCPY_DEVICE_TO_HOST);
        correct = status == 0 && ack.magic == kEndpointDispatchMagic &&
            ack.generation == kGeneration && ack.sequence == 1u &&
            ack.source_rank == static_cast<uint32_t>(pe) &&
            ack.status == 0u && ack.tokens_consumed == tokens &&
            completion.magic == kEndpointDispatchMagic &&
            completion.generation == kGeneration &&
            completion.destination_rank == static_cast<uint32_t>(pe) &&
            completion.status == 0u;
        if (!correct) {
            std::cerr << "[FAIL] pe=" << pe << " ack={magic=" << ack.magic
                      << ",gen=" << ack.generation << ",seq=" << ack.sequence
                      << ",src=" << ack.source_rank << ",status="
                      << ack.status << ",tokens=" << ack.tokens_consumed
                      << "} completion={magic=" << completion.magic
                      << ",gen=" << completion.generation << ",dst="
                      << completion.destination_rank << ",status="
                      << completion.status << ",rows="
                      << completion.row_count << ",assignments="
                      << completion.assignment_count << "}\n";
        }

        std::vector<EndpointDispatchInput> all_inputs;
        for (uint32_t source = 0u; source < workers; ++source)
            all_inputs.push_back(MakeInput(source, workers, tokens, hidden,
                                           topk));
        std::vector<EndpointDispatchFanoutRecord> expected_rows;
        std::vector<EndpointDispatchAssignmentRecord> expected_assignments;
        std::vector<EndpointDispatchTokenRecord> expected_wire_tokens;
        std::vector<EndpointDispatchAssignmentRecord>
            expected_wire_assignments;
        std::vector<uint8_t> expected_hidden;
        std::vector<uint32_t> expected_counts(workers * 4u, 0u);
        for (uint32_t source = 0u; source < workers; ++source) {
            expected_counts[workers * 2u + source] =
                static_cast<uint32_t>(expected_rows.size());
            expected_counts[workers * 3u + source] =
                static_cast<uint32_t>(expected_assignments.size());
            const EndpointDispatchInput &input = all_inputs[source];
            for (uint32_t token = 0u; token < tokens; ++token) {
                EndpointDispatchTokenRecord record{};
                record.token_id = input.token_ids[token];
                record.source_token = token;
                record.assignment_begin = input.assignment_offsets[token];
                record.assignment_count = input.assignment_offsets[token + 1u]
                    - input.assignment_offsets[token];
                expected_wire_tokens.push_back(record);
            }
            expected_wire_assignments.insert(
                expected_wire_assignments.end(), input.assignments.begin(),
                input.assignments.end());
            uint32_t source_rows = 0u;
            uint32_t source_assignments = 0u;
            for (uint32_t token = 0u; token < tokens; ++token) {
                std::vector<EndpointDispatchAssignmentRecord> grouped;
                for (uint32_t index = input.assignment_offsets[token];
                     index < input.assignment_offsets[token + 1u]; ++index) {
                    if (input.assignments[index].destination_rank ==
                        static_cast<uint32_t>(pe))
                        grouped.push_back(input.assignments[index]);
                }
                if (grouped.empty()) continue;
                EndpointDispatchFanoutRecord row{};
                row.token_id = input.token_ids[token];
                row.source_rank = source;
                row.source_token = token;
                row.assignment_begin = static_cast<uint32_t>(
                    expected_assignments.size());
                row.assignment_count = static_cast<uint32_t>(grouped.size());
                expected_rows.push_back(row);
                expected_assignments.insert(expected_assignments.end(),
                                            grouped.begin(), grouped.end());
                for (uint64_t byte = 0u; byte < row_bytes; ++byte)
                    expected_hidden.push_back(HiddenByte(source, token, byte));
                ++source_rows;
                source_assignments += grouped.size();
            }
            expected_counts[source] = source_rows;
            expected_counts[workers + source] = source_assignments;
        }
        correct = correct && completion.row_count == expected_rows.size() &&
            completion.assignment_count == expected_assignments.size();

        std::vector<uint32_t> actual_counts;
        std::vector<EndpointDispatchTokenRecord> actual_wire_tokens;
        std::vector<EndpointDispatchAssignmentRecord>
            actual_wire_assignments;
        std::vector<uint8_t> actual_hidden;
        const bool copied = CopyFromDevice(&actual_counts, recv_counts,
                                           workers * 4u) &&
            CopyFromDevice(&actual_wire_tokens, recv_rows,
                           expected_wire_tokens.size()) &&
            CopyFromDevice(&actual_wire_assignments, recv_assignments,
                           expected_wire_assignments.size()) &&
            CopyFromDevice(&actual_hidden, recv_hidden,
                           expected_hidden.size());
        if (copied && actual_counts != expected_counts)
            std::cerr << "[FAIL] pe=" << pe << " count reply mismatch\n";
        if (copied && actual_hidden != expected_hidden) {
            size_t mismatch = 0u;
            while (mismatch < actual_hidden.size() &&
                   mismatch < expected_hidden.size() &&
                   actual_hidden[mismatch] == expected_hidden[mismatch])
                ++mismatch;
            std::cerr << "[FAIL] pe=" << pe
                      << " hidden fanout mismatch byte=" << mismatch;
            if (mismatch < actual_hidden.size() &&
                mismatch < expected_hidden.size())
                std::cerr << " actual="
                          << static_cast<uint32_t>(actual_hidden[mismatch])
                          << " expected="
                          << static_cast<uint32_t>(expected_hidden[mismatch]);
            std::cerr << '\n';
        }
        correct = correct && copied && actual_counts == expected_counts &&
            actual_hidden == expected_hidden &&
            actual_wire_tokens.size() == expected_wire_tokens.size() &&
            actual_wire_assignments.size() ==
                expected_wire_assignments.size();
        for (size_t i = 0u; correct &&
             i < expected_wire_tokens.size(); ++i) {
            if (std::memcmp(&actual_wire_tokens[i],
                            &expected_wire_tokens[i],
                            sizeof(expected_wire_tokens[i])) != 0) {
                std::cerr << "[FAIL] pe=" << pe << " wire_token=" << i
                          << " actual={token="
                          << actual_wire_tokens[i].token_id
                          << ",local="
                          << actual_wire_tokens[i].source_token
                          << ",begin="
                          << actual_wire_tokens[i].assignment_begin
                          << ",count="
                          << actual_wire_tokens[i].assignment_count
                          << "} expected={token="
                          << expected_wire_tokens[i].token_id
                          << ",local="
                          << expected_wire_tokens[i].source_token
                          << ",begin="
                          << expected_wire_tokens[i].assignment_begin
                          << ",count="
                          << expected_wire_tokens[i].assignment_count
                          << "}\n";
                correct = false;
            }
        }
        for (size_t i = 0u; correct &&
             i < expected_wire_assignments.size(); ++i) {
            const bool assignment_ok =
                actual_wire_assignments[i].destination_rank ==
                    expected_wire_assignments[i].destination_rank &&
                actual_wire_assignments[i].expert_id ==
                    expected_wire_assignments[i].expert_id &&
                actual_wire_assignments[i].ordinal ==
                    expected_wire_assignments[i].ordinal &&
                actual_wire_assignments[i].weight ==
                    expected_wire_assignments[i].weight;
            if (!assignment_ok) {
                std::cerr << "[FAIL] pe=" << pe << " assignment=" << i
                          << " actual={dst="
                          << actual_wire_assignments[i].destination_rank
                          << ",expert="
                          << actual_wire_assignments[i].expert_id
                          << ",ordinal="
                          << actual_wire_assignments[i].ordinal
                          << ",weight="
                          << actual_wire_assignments[i].weight
                          << "} expected={dst="
                          << expected_wire_assignments[i].destination_rank
                          << ",expert="
                          << expected_wire_assignments[i].expert_id
                          << ",ordinal="
                          << expected_wire_assignments[i].ordinal
                          << ",weight="
                          << expected_wire_assignments[i].weight
                          << "}\n";
                correct = false;
            }
        }

        SparseCombineDeviceAck combine_ack{};
        SparseCombineEgressCompletion combine_completion{};
        if (correct) {
            status = aclrtMemcpy(
                &combine_ack, sizeof(combine_ack),
                combine_acks + static_cast<uint64_t>(pe) *
                    sizeof(combine_ack),
                sizeof(combine_ack), ACL_MEMCPY_DEVICE_TO_HOST);
        }
        if (status == 0 && correct) {
            status = aclrtMemcpy(
                &combine_completion, sizeof(combine_completion),
                combine_completions + static_cast<uint64_t>(pe) *
                    sizeof(combine_completion),
                sizeof(combine_completion), ACL_MEMCPY_DEVICE_TO_HOST);
        }
        correct = correct && status == 0 &&
            combine_ack.magic == kSparseCombineMagic &&
            combine_ack.abi_version == kSparseCombineAbiVersion &&
            combine_ack.struct_bytes == sizeof(SparseCombineDeviceAck) &&
            combine_ack.generation == kGeneration &&
            combine_ack.sequence == 1u &&
            combine_ack.source_rank == static_cast<uint32_t>(pe) &&
            combine_ack.status == expected_combine_status &&
            combine_ack.rows_consumed ==
                (expect_combine_reject ? 0u : combine_rows_per_worker[pe]) &&
            combine_completion.magic == kSparseCombineMagic &&
            combine_completion.abi_version == kSparseCombineAbiVersion &&
            combine_completion.struct_bytes ==
                sizeof(SparseCombineEgressCompletion) &&
            combine_completion.generation == kGeneration &&
            combine_completion.wave == kWave &&
            combine_completion.owner_rank == static_cast<uint32_t>(pe) &&
            combine_completion.status == expected_combine_status &&
            combine_completion.row_count ==
                (expect_combine_reject ? 0u : tokens);
        if (!correct) {
            std::cerr << "[FAIL] pe=" << pe << " combine ack={status="
                      << combine_ack.status << ",rows="
                      << combine_ack.rows_consumed << "} completion={status="
                      << combine_completion.status << ",rows="
                      << combine_completion.row_count << "}\n";
        }
        std::vector<float> actual_output;
        if (correct && !expect_combine_reject)
            correct = CopyFromDevice(
                &actual_output, combine_output,
                static_cast<uint64_t>(tokens) * hidden);
        const EndpointDispatchInput owner_input = MakeInput(
            static_cast<uint32_t>(pe), workers, tokens, hidden, topk);
        for (uint32_t token = 0u;
             correct && !expect_combine_reject && token < tokens; ++token) {
            std::vector<uint8_t> selected(workers, 0u);
            for (uint32_t assignment = owner_input.assignment_offsets[token];
                 assignment < owner_input.assignment_offsets[token + 1u];
                 ++assignment) {
                selected[owner_input.assignments[assignment]
                             .destination_rank] = 1u;
            }
            for (uint32_t element = 0u; correct && element < hidden;
                 ++element) {
                float expected = 0.0f;
                for (uint32_t expert = 0u; expert < workers; ++expert)
                    if (selected[expert] != 0u)
                        expected += PartialValue(
                            expert, owner_input.token_ids[token], element);
                const float actual = actual_output[
                    static_cast<uint64_t>(token) * hidden + element];
                if (actual != expected) {
                    std::cerr << "[FAIL] pe=" << pe
                              << " combine output token=" << token
                              << " element=" << element << " actual="
                              << actual << " expected=" << expected << '\n';
                    correct = false;
                }
            }
        }
    }

    if (correct && pe == inc_pe && !expect_reject) {
        uint32_t combine_device_status =
            std::numeric_limits<uint32_t>::max();
        status = aclrtMemcpy(
            &combine_device_status, sizeof(combine_device_status),
            combine_status_line, sizeof(combine_device_status),
            ACL_MEMCPY_DEVICE_TO_HOST);
        correct = status == 0 &&
            combine_device_status == expected_combine_status;
        if (!correct)
            std::cerr << "[FAIL] INC sparse Combine device_status="
                      << combine_device_status << '\n';
        if (!correct && combine_device_status == 2u) {
            std::vector<SparseCombineReadyDescriptor> observed;
            if (CopyFromDevice(&observed, combine_descriptors, workers)) {
                for (uint32_t source = 0u; source < workers; ++source) {
                    const auto &d = observed[source];
                    std::cerr << "[FAIL] descriptor source=" << source
                              << " magic=" << d.magic << " abi="
                              << d.abi_version << " bytes="
                              << d.struct_bytes << " gen=" << d.generation
                              << " seq=" << d.sequence << " wave="
                              << d.wave << " src=" << d.source_rank
                              << " rows=" << d.row_count << " hidden="
                              << d.hidden << " dtype=" << d.partial_dtype
                              << " slot=" << d.slot << " region="
                              << d.source_region_id << " payload="
                              << d.payload_bytes << " digest="
                              << d.metadata_digest << " flags=" << d.flags
                              << '\n';
                }
            }
        }
        if (!correct && combine_device_status == 4u) {
            std::vector<DeviceJournalEntry> debug_entries;
            std::vector<uint32_t> debug_map;
            std::vector<uint32_t> debug_rows;
            if (CopyFromDevice(&debug_entries, journal_entries,
                               static_cast<uint64_t>(workers) * tokens) &&
                CopyFromDevice(&debug_map, combine_row_map,
                               static_cast<uint64_t>(workers) *
                                   journal_capacity) &&
                CopyFromDevice(&debug_rows, destination_rows, workers)) {
                for (uint32_t source = 0u; source < workers; ++source) {
                    for (uint64_t index = 0u;
                         index < debug_entries.size(); ++index) {
                        const bool expected =
                            (debug_entries[index].expected[source >> 6u] &
                             (1ull << (source & 63u))) != 0u;
                        const uint32_t row = debug_map[
                            static_cast<uint64_t>(source) *
                                journal_capacity + index];
                        if ((!expected && row != kDeviceJournalEmpty) ||
                            (expected && row >= debug_rows[source])) {
                            std::cerr << "[FAIL] runtime row map source="
                                      << source << " index=" << index
                                      << " token="
                                      << debug_entries[index].token_id
                                      << " expected=" << expected
                                      << " row=" << row << " rows="
                                      << debug_rows[source] << '\n';
                            source = workers;
                            break;
                        }
                    }
                }
            }
        }
    }

    if (status == 0) aclshmem_barrier_all();
    if (status_line != nullptr) aclshmem_free(status_line);
    if (combine_status_line != nullptr) aclshmem_free(combine_status_line);
    if (inc_combine_token_ids != nullptr)
        aclshmem_free(inc_combine_token_ids);
    if (combine_completions != nullptr) aclshmem_free(combine_completions);
    if (combine_acks != nullptr) aclshmem_free(combine_acks);
    if (combine_descriptors != nullptr) aclshmem_free(combine_descriptors);
    if (combine_output != nullptr) aclshmem_free(combine_output);
    if (combine_partials != nullptr) aclshmem_free(combine_partials);
    if (combine_token_ids != nullptr) aclshmem_free(combine_token_ids);
    if (destination_rows != nullptr) aclshmem_free(destination_rows);
    if (combine_row_map != nullptr) aclshmem_free(combine_row_map);
    if (journal_row_map != nullptr) aclshmem_free(journal_row_map);
    if (journal_hash != nullptr) aclshmem_free(journal_hash);
    if (journal_entries != nullptr) aclshmem_free(journal_entries);
    if (journal_header != nullptr) aclshmem_free(journal_header);
    if (hidden_staging != nullptr) aclshmem_free(hidden_staging);
    if (cursors != nullptr) aclshmem_free(cursors);
    if (completions != nullptr) aclshmem_free(completions);
    if (acks != nullptr) aclshmem_free(acks);
    if (recv_counts != nullptr) aclshmem_free(recv_counts);
    if (recv_assignments != nullptr) aclshmem_free(recv_assignments);
    if (recv_rows != nullptr) aclshmem_free(recv_rows);
    if (recv_hidden != nullptr) aclshmem_free(recv_hidden);
    if (commits != nullptr) aclshmem_free(commits);
    if (inc_packets != nullptr) aclshmem_free(inc_packets);
    if (source_packet != nullptr) aclshmem_free(source_packet);
    if (shmem_initialized) aclshmem_finalize();
    if (stream != nullptr) aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();

    if (!correct) return Fail("device endpoint Dispatch", status);
    std::cout << "[PASS] pe=" << pe << " workers=" << workers
              << " tokens=" << tokens << " hidden=" << hidden
              << " topk=" << topk << " aiv=" << dispatch_aiv;
    if (expect_reject || expect_combine_reject)
        std::cout << " expected_reject=" << fault;
    if (!expect_reject && !expect_combine_reject && pe == inc_pe &&
        e2e_us > 0.0) {
        uint64_t fanout_rows = 0u;
        for (uint32_t source = 0u; source < workers; ++source) {
            const EndpointDispatchInput input = MakeInput(
                source, workers, tokens, hidden, topk);
            for (uint32_t token = 0u; token < tokens; ++token) {
                std::vector<uint8_t> seen(workers, 0u);
                for (uint32_t index = input.assignment_offsets[token];
                     index < input.assignment_offsets[token + 1u]; ++index)
                    seen[input.assignments[index].destination_rank] = 1u;
                for (uint8_t value : seen) fanout_rows += value;
            }
        }
        const double dispatch_ingress_bytes =
            static_cast<double>(workers) * tokens * row_bytes;
        const double dispatch_egress_bytes =
            static_cast<double>(fanout_rows) * row_bytes;
        const double logical_hidden_bytes = dispatch_ingress_bytes +
            dispatch_egress_bytes;
        std::cout << " e2e_us=" << e2e_us
                  << " logical_hidden_gb_s="
                  << logical_hidden_bytes / e2e_us / 1.0e3
                  << " dispatch_ingress_gb_s="
                  << dispatch_ingress_bytes / e2e_us / 1.0e3
                  << " dispatch_egress_gb_s="
                  << dispatch_egress_bytes / e2e_us / 1.0e3
                  << " journal_index_us=" << index_us;
        uint64_t combine_ingress_rows = 0u;
        for (uint32_t rows : combine_rows_per_worker)
            combine_ingress_rows += rows;
        const double combine_ingress_bytes =
            static_cast<double>(combine_ingress_rows) * hidden *
            sizeof(float);
        const double combine_egress_bytes =
            static_cast<double>(workers) * tokens * hidden * sizeof(float);
        const double logical_combine_bytes = combine_ingress_bytes +
            combine_egress_bytes;
        std::cout << " combine_us=" << combine_us
                  << " logical_combine_gb_s="
                  << logical_combine_bytes / combine_us / 1.0e3
                  << " combine_ingress_gb_s="
                  << combine_ingress_bytes / combine_us / 1.0e3
                  << " combine_egress_gb_s="
                  << combine_egress_bytes / combine_us / 1.0e3;
        const double serial_dc_us = e2e_us + index_us + combine_us;
        std::cout << " serial_dc_us=" << serial_dc_us
                  << " serial_dc_logical_gb_s="
                  << (logical_hidden_bytes + logical_combine_bytes) /
                         serial_dc_us / 1.0e3;
    }
    std::cout << '\n';
    return 0;
}
