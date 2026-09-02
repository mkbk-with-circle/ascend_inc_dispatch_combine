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

using namespace inc::dc::pull_combine;

extern "C" void launch_inc_dc_endpoint_dispatch_device_e2e(
    uint32_t block_dim, void *stream, uint8_t *source_packet,
    uint8_t *inc_packets, uint8_t *commit_mailbox, uint8_t *recv_hidden,
    uint8_t *recv_rows, uint8_t *recv_assignments, uint8_t *recv_counts,
    uint8_t *ack_mailbox, uint8_t *completion_mailbox, uint8_t *cursors,
    uint8_t *hidden_staging, uint8_t *status_line, uint64_t ffts_addr,
    uint64_t slot_bytes, uint64_t staging_bytes_per_destination,
    uint64_t row_capacity, uint64_t assignment_capacity, uint32_t hidden,
    uint32_t dtype, uint32_t expert_count, uint32_t worker_count,
    int32_t inc_pe, uint64_t generation, uint32_t wave);

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

uint8_t HiddenByte(uint32_t source, uint32_t token, uint64_t byte)
{
    return static_cast<uint8_t>(
        (static_cast<uint64_t>(source) * 131u +
         static_cast<uint64_t>(token) * 17u + byte) % 251u);
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
    if (argc < 8 || argc > 10) {
        std::cerr << "usage: " << argv[0]
                  << " <workers> <pe> <ipport> <first_npu> <tokens>"
                     " <hidden> <topk> [aiv; 0=half of live AIV]"
                     " [fault; 0=none, 1=digest, 2=duplicate ordinal,"
                     " 3=nonfinite weight, 4=count mismatch]\n";
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
    const uint32_t fault = argc == 10
        ? static_cast<uint32_t>(std::strtoul(argv[9], nullptr, 10)) : 0u;
    const uint32_t pes = workers + 1u;
    const int inc_pe = static_cast<int>(workers);
    g_npus = static_cast<int>(pes);
    if (workers < 2u || workers > 8u || pe < 0 ||
        pe >= static_cast<int>(pes) || tokens == 0u || hidden == 0u ||
        topk == 0u || topk > 32u || fault > 4u) {
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
    const bool expect_reject = fault != 0u;
    if (fault != 0u && pe == 0) {
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
    const uint64_t row_capacity = static_cast<uint64_t>(workers) * tokens;
    const uint64_t assignment_capacity =
        static_cast<uint64_t>(workers) * tokens * topk;
    const uint64_t counts_bytes =
        static_cast<uint64_t>(workers) * sizeof(uint32_t) * 4u;
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
    uint8_t *status_line = nullptr;
    double e2e_us = 0.0;
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
        test_set_attr(pe, pes, 1024ull * 1024ull * 1024ull, ipport,
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
                  staging_bytes_per_destination * workers));
        status_line = static_cast<uint8_t *>(aclshmem_malloc(64u));
        if (source_packet == nullptr || inc_packets == nullptr ||
            commits == nullptr || recv_hidden == nullptr ||
            recv_rows == nullptr || recv_assignments == nullptr ||
            recv_counts == nullptr || acks == nullptr ||
            completions == nullptr || cursors == nullptr ||
            (staging_bytes_per_destination != 0u &&
             hidden_staging == nullptr) ||
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
                       staging_bytes_per_destination * workers);
        ZeroDevice(status_line, 64u);
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
            cursors, hidden_staging, status_line, shmemx_get_ffts_config(),
            slot_bytes, staging_bytes_per_destination,
            row_capacity,
            assignment_capacity, hidden,
            static_cast<uint32_t>(EndpointDataType::BF16), kExpertCount,
            workers, inc_pe, kGeneration, kWave);
        status = aclrtSynchronizeStream(stream);
        const auto end = std::chrono::steady_clock::now();
        e2e_us = std::chrono::duration<double, std::micro>(end - begin).count();
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
        std::vector<uint8_t> expected_hidden;
        std::vector<uint32_t> expected_counts(workers * 4u, 0u);
        for (uint32_t source = 0u; source < workers; ++source) {
            expected_counts[workers * 2u + source] =
                static_cast<uint32_t>(expected_rows.size());
            expected_counts[workers * 3u + source] =
                static_cast<uint32_t>(expected_assignments.size());
            const EndpointDispatchInput &input = all_inputs[source];
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
        std::vector<EndpointDispatchFanoutRecord> actual_rows;
        std::vector<EndpointDispatchAssignmentRecord> actual_assignments;
        std::vector<uint8_t> actual_hidden;
        const bool copied = CopyFromDevice(&actual_counts, recv_counts,
                                           workers * 4u) &&
            CopyFromDevice(&actual_rows, recv_rows, expected_rows.size()) &&
            CopyFromDevice(&actual_assignments, recv_assignments,
                           expected_assignments.size()) &&
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
            actual_rows.size() == expected_rows.size() &&
            actual_assignments.size() == expected_assignments.size();
        for (size_t i = 0u; correct && i < expected_rows.size(); ++i) {
            if (std::memcmp(&actual_rows[i], &expected_rows[i],
                            sizeof(expected_rows[i])) != 0) {
                std::cerr << "[FAIL] pe=" << pe << " row=" << i
                          << " actual={token=" << actual_rows[i].token_id
                          << ",src=" << actual_rows[i].source_rank
                          << ",local=" << actual_rows[i].source_token
                          << ",begin=" << actual_rows[i].assignment_begin
                          << ",count=" << actual_rows[i].assignment_count
                          << "} expected={token=" << expected_rows[i].token_id
                          << ",src=" << expected_rows[i].source_rank
                          << ",local=" << expected_rows[i].source_token
                          << ",begin=" << expected_rows[i].assignment_begin
                          << ",count=" << expected_rows[i].assignment_count
                          << "}\n";
                correct = false;
            }
        }
        for (size_t i = 0u; correct && i < expected_assignments.size(); ++i) {
            const bool assignment_ok =
                actual_assignments[i].destination_rank ==
                    expected_assignments[i].destination_rank &&
                actual_assignments[i].expert_id ==
                    expected_assignments[i].expert_id &&
                actual_assignments[i].ordinal ==
                    expected_assignments[i].ordinal &&
                actual_assignments[i].weight ==
                    expected_assignments[i].weight;
            if (!assignment_ok) {
                std::cerr << "[FAIL] pe=" << pe << " assignment=" << i
                          << " actual={dst="
                          << actual_assignments[i].destination_rank
                          << ",expert=" << actual_assignments[i].expert_id
                          << ",ordinal=" << actual_assignments[i].ordinal
                          << ",weight=" << actual_assignments[i].weight
                          << "} expected={dst="
                          << expected_assignments[i].destination_rank
                          << ",expert=" << expected_assignments[i].expert_id
                          << ",ordinal=" << expected_assignments[i].ordinal
                          << ",weight=" << expected_assignments[i].weight
                          << "}\n";
                correct = false;
            }
        }
    }

    if (status == 0) aclshmem_barrier_all();
    if (status_line != nullptr) aclshmem_free(status_line);
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
    if (expect_reject) std::cout << " expected_reject=" << fault;
    if (!expect_reject && pe == inc_pe && e2e_us > 0.0) {
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
        const double logical_hidden_bytes =
            static_cast<double>(workers) * tokens * row_bytes +
            static_cast<double>(fanout_rows) * row_bytes;
        std::cout << " e2e_us=" << e2e_us
                  << " logical_hidden_gb_s="
                  << logical_hidden_bytes / e2e_us / 1.0e3;
    }
    std::cout << '\n';
    return 0;
}
