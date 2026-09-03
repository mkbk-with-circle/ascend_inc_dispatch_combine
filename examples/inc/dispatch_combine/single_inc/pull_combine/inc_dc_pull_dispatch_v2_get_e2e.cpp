#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "shmem.h"
#include "utils.h"

#include "inc_dc_pull_dispatch_v2.h"

using namespace inc::dc::pull_v2;

extern "C" void launch_inc_dc_pull_dispatch_v2_get(
    uint32_t block_dim, void *stream, uint8_t *source_region,
    uint8_t *ready_mailbox, uint8_t *inc_slots, uint8_t *source_acks,
    uint8_t *status_line, uint64_t ffts_addr, uint64_t session_id,
    uint64_t placement_epoch, uint64_t generation, uint64_t sequence,
    uint64_t slot_stride, uint32_t worker_count, int32_t inc_pe,
    uint32_t region_id, uint32_t wave, uint32_t ring_slot,
    uint32_t slot_count, uint32_t pull_lanes_per_source, uint64_t spin_cap);

int g_npus = 5;
const char *ipport = "tcp://127.0.0.1:28795";
int f_pe = 0;
int f_npu = 0;
aclshmemx_uniqueid_t default_flag_uid;

namespace {

constexpr uint64_t kSessionId = 0x50444c4c56320001ull;
constexpr uint64_t kPlacementEpoch = 1u;
constexpr uint64_t kGeneration = 17u;
constexpr uint64_t kSequence = 3u;
constexpr uint32_t kWave = 5u;
constexpr uint32_t kRegionId = 1u;
constexpr uint32_t kRingSlots = 2u;

int Fail(const char *step, int status)
{
    std::cerr << "[FAIL] " << step << " status=" << status << '\n';
    return status == 0 ? 1 : status;
}

uint64_t AckPublication(uint64_t generation, uint64_t sequence,
                        uint32_t source)
{
    uint64_t value = generation ^ (sequence << 1u) ^
        (static_cast<uint64_t>(source) << 48u) ^ 0xa55aa55aa55aa55aull;
    return value == 0u ? 1u : value;
}

uint8_t HiddenByte(uint32_t source, uint32_t token, uint64_t byte)
{
    return static_cast<uint8_t>((static_cast<uint64_t>(source) * 131u +
        static_cast<uint64_t>(token) * 17u + byte) % 251u);
}

SourceInput MakeInput(uint32_t source, uint32_t workers, uint32_t tokens,
                      uint32_t hidden, uint32_t topk, uint32_t ring_slot)
{
    SourceInput input{};
    input.session.session_id = kSessionId;
    input.session.placement_epoch = kPlacementEpoch;
    input.session.worker_count = workers;
    input.session.expert_count = 64u;
    input.session.hidden = hidden;
    input.session.dtype = DataType::BF16;
    input.session.ring_slots = kRingSlots;
    input.generation = kGeneration;
    input.sequence = kSequence;
    input.wave = kWave;
    input.source_rank = source;
    input.source_region_id = kRegionId;
    input.ring_slot = static_cast<uint16_t>(ring_slot);
    input.assignment_offsets.push_back(0u);
    for (uint32_t token = 0u; token < tokens; ++token) {
        input.token_ids.push_back(
            static_cast<uint64_t>(source) * 1000000u + token + 1u);
        for (uint32_t k = 0u; k < topk; ++k) {
            input.assignments.push_back(AssignmentRecord{
                (source + token + k / 2u) % workers,
                (source * 11u + token * 7u + k) % 64u, k,
                static_cast<float>(k + 1u) /
                    static_cast<float>(topk + 1u)});
        }
        input.assignment_offsets.push_back(input.assignments.size());
    }
    const uint64_t row_bytes = static_cast<uint64_t>(hidden) * 2u;
    input.hidden_payload.resize(static_cast<size_t>(tokens) * row_bytes);
    for (uint32_t token = 0u; token < tokens; ++token)
        for (uint64_t byte = 0u; byte < row_bytes; ++byte)
            input.hidden_payload[static_cast<uint64_t>(token) * row_bytes +
                                 byte] = HiddenByte(source, token, byte);
    return input;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 8 || argc > 12) {
        std::cerr << "usage: " << argv[0]
                  << " <workers> <pe> <ipport> <first_npu> <tokens>"
                     " <hidden> <topk> [aiv=24] [pull_lanes=2]"
                     " [ring_slot=0] [fault=0]\n";
        return 2;
    }
    const uint32_t workers = std::strtoul(argv[1], nullptr, 10);
    const int pe = std::atoi(argv[2]);
    ipport = argv[3];
    f_npu = std::atoi(argv[4]);
    const uint32_t tokens = std::strtoul(argv[5], nullptr, 10);
    const uint32_t hidden = std::strtoul(argv[6], nullptr, 10);
    const uint32_t topk = std::strtoul(argv[7], nullptr, 10);
    const uint32_t aiv = argc >= 9
        ? std::strtoul(argv[8], nullptr, 10) : 24u;
    const uint32_t pull_lanes = argc >= 10
        ? std::strtoul(argv[9], nullptr, 10) : 2u;
    const uint32_t ring_slot = argc >= 11
        ? std::strtoul(argv[10], nullptr, 10) : 0u;
    const uint32_t fault = argc >= 12
        ? std::strtoul(argv[11], nullptr, 10) : 0u;
    const uint32_t pes = workers + 1u;
    const int inc_pe = workers;
    g_npus = pes;
    if (workers < 2u || workers > 8u || pe < 0 ||
        pe >= static_cast<int>(pes) || hidden == 0u ||
        (tokens != 0u && topk == 0u) || topk > 32u || aiv == 0u ||
        pull_lanes == 0u || workers * pull_lanes > aiv ||
        ring_slot >= kRingSlots || fault > 3u)
        return Fail("arguments", 2);

    std::vector<std::vector<uint8_t>> expected_slots(workers);
    std::vector<Ready> expected_ready(workers);
    uint64_t slot_stride = 0u;
    for (uint32_t source = 0u; source < workers; ++source) {
        SlotHeader header{};
        const SourceInput input = MakeInput(
            source, workers, tokens, hidden, topk, ring_slot);
        std::string error;
        if (BuildSlot(input, &expected_slots[source], &header,
                      &expected_ready[source], &error) != Status::OK) {
            std::cerr << error << '\n';
            return Fail("build source slot", 2);
        }
        if (slot_stride == 0u) slot_stride = expected_slots[source].size();
        if (expected_slots[source].size() != slot_stride)
            return Fail("nonuniform qualification slot", 2);
    }
    if (slot_stride > std::numeric_limits<uint32_t>::max())
        return Fail("slot too large", 2);

    const int device = pe + f_npu;
    aclrtStream stream = nullptr;
    bool shmem_initialized = false;
    uint8_t *source_region = nullptr;
    uint8_t *ready_mailbox = nullptr;
    uint8_t *inc_slots = nullptr;
    uint8_t *source_acks = nullptr;
    uint8_t *status_line = nullptr;
    int status = aclInit(nullptr);
    if (status == 0) status = aclrtSetDevice(device);
    int64_t live_aiv = 0;
    if (status == 0)
        status = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                    &live_aiv);
    if (status == 0 && (live_aiv <= 0 || aiv > live_aiv / 2)) status = 2;
    if (status == 0) status = aclrtCreateStream(&stream);
    if (status == 0) {
        aclshmemx_init_attr_t attr;
        test_set_attr(pe, pes, 2ull * 1024ull * 1024ull * 1024ull, ipport,
                      default_flag_uid, &attr);
        status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
        shmem_initialized = status == 0;
    }
    if (status == 0) {
        source_region = static_cast<uint8_t *>(aclshmem_malloc(
            slot_stride * kRingSlots));
        ready_mailbox = static_cast<uint8_t *>(aclshmem_malloc(
            static_cast<uint64_t>(workers) * sizeof(Ready)));
        inc_slots = static_cast<uint8_t *>(aclshmem_malloc(
            static_cast<uint64_t>(workers) * slot_stride));
        source_acks = static_cast<uint8_t *>(aclshmem_malloc(
            static_cast<uint64_t>(workers) * sizeof(SourceConsumed)));
        status_line = static_cast<uint8_t *>(aclshmem_malloc(
            sizeof(PullTimeline)));
        if (source_region == nullptr || ready_mailbox == nullptr ||
            inc_slots == nullptr || source_acks == nullptr ||
            status_line == nullptr)
            status = 1;
    }
    std::vector<uint8_t> zero;
    auto Zero = [&](uint8_t *pointer, uint64_t bytes) {
        if (status != 0) return;
        zero.assign(static_cast<size_t>(bytes), 0u);
        status = aclrtMemcpy(pointer, bytes, zero.data(), bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE);
    };
    if (status == 0) {
        Zero(source_region, slot_stride * kRingSlots);
        Zero(ready_mailbox,
             static_cast<uint64_t>(workers) * sizeof(Ready));
        Zero(inc_slots, static_cast<uint64_t>(workers) * slot_stride);
        Zero(source_acks,
             static_cast<uint64_t>(workers) * sizeof(SourceConsumed));
        Zero(status_line, sizeof(PullTimeline));
    }
    if (status == 0 && pe < inc_pe) {
        std::vector<uint8_t> local_slot = expected_slots[pe];
        Ready local_ready = expected_ready[pe];
        if (fault == 1u && pe == 0) ++local_ready.sequence;
        if (fault == 2u && pe == 0) {
            SlotHeader *header =
                reinterpret_cast<SlotHeader *>(local_slot.data());
            ++header->generation;
        }
        if (fault == 3u && pe == 0) local_ready.publication = 0u;
        status = aclrtMemcpy(
            source_region + static_cast<uint64_t>(ring_slot) * slot_stride,
            slot_stride, local_slot.data(), slot_stride,
            ACL_MEMCPY_HOST_TO_DEVICE);
        if (status == 0)
            status = aclrtMemcpy(
                ready_mailbox + static_cast<uint64_t>(pe) * sizeof(Ready),
                sizeof(Ready), &local_ready, sizeof(Ready),
                ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (status == 0) aclshmem_barrier_all();

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    if (status == 0) {
        begin = std::chrono::steady_clock::now();
        launch_inc_dc_pull_dispatch_v2_get(
            aiv, stream, source_region, ready_mailbox, inc_slots,
            source_acks, status_line, shmemx_get_ffts_config(), kSessionId,
            kPlacementEpoch, kGeneration, kSequence, slot_stride, workers,
            inc_pe, kRegionId, kWave, ring_slot, kRingSlots, pull_lanes,
            fault == 3u ? 1000000ull : 1000000000ull);
        status = aclrtSynchronizeStream(stream);
        end = std::chrono::steady_clock::now();
    }
    if (status == 0) aclshmem_barrier_all();

    const bool expect_reject = fault != 0u;
    bool correct = status == 0;
    if (correct && pe == inc_pe) {
        PullTimeline timeline{};
        status = aclrtMemcpy(&timeline, sizeof(timeline), status_line,
                             sizeof(timeline), ACL_MEMCPY_DEVICE_TO_HOST);
        const uint32_t expected_status = fault == 1u ? 2u :
            (fault == 2u ? 3u : (fault == 3u ? 1u : 0u));
        correct = status == 0 && timeline.status == expected_status;
        if (correct && !expect_reject) {
            std::vector<uint8_t> actual(slot_stride);
            for (uint32_t source = 0u; source < workers && correct;
                 ++source) {
                status = aclrtMemcpy(
                    actual.data(), actual.size(),
                    inc_slots + static_cast<uint64_t>(source) * slot_stride,
                    actual.size(), ACL_MEMCPY_DEVICE_TO_HOST);
                correct = status == 0 && actual == expected_slots[source];
                if (!correct)
                    std::cerr << "[FAIL] pulled slot source=" << source
                              << '\n';
            }
        }
        const double seconds = std::chrono::duration<double>(end - begin)
            .count();
        const double hidden_bytes = static_cast<double>(workers) * tokens *
            hidden * 2u;
        const double wire_bytes = static_cast<double>(workers) * slot_stride;
        std::cout << "PULL_DISPATCH_V2_GET workers=" << workers
                  << " tokens=" << tokens << " hidden=" << hidden
                  << " topk=" << topk << " aiv=" << aiv
                  << " pull_lanes=" << pull_lanes
                  << " status=" << timeline.status
                  << " ready_sources=" << timeline.ready_sources
                  << " us=" << seconds * 1e6
                  << " hidden_get_gb_s=" << hidden_bytes / seconds / 1e9
                  << " wire_get_gb_s=" << wire_bytes / seconds / 1e9
                  << '\n';
    }
    if (correct && pe < inc_pe) {
        SourceConsumed ack{};
        status = aclrtMemcpy(
            &ack, sizeof(ack),
            source_acks + static_cast<uint64_t>(pe) * sizeof(ack),
            sizeof(ack), ACL_MEMCPY_DEVICE_TO_HOST);
        const uint32_t expected_status = expect_reject
            ? (fault == 1u ? 2u : (fault == 2u ? 3u : 1u)) : 0u;
        correct = status == 0 && ack.magic == kPullDispatchMagic &&
            ack.generation == kGeneration && ack.sequence == kSequence &&
            ack.source_rank == static_cast<uint32_t>(pe) &&
            ack.ring_slot == ring_slot && ack.status == expected_status &&
            ack.bytes_consumed ==
                (expect_reject ? 0u : expected_slots[pe].size()) &&
            ack.publication == AckPublication(
                kGeneration, kSequence, static_cast<uint32_t>(pe));
    }

    if (status == 0) aclshmem_barrier_all();
    if (status_line != nullptr) aclshmem_free(status_line);
    if (source_acks != nullptr) aclshmem_free(source_acks);
    if (inc_slots != nullptr) aclshmem_free(inc_slots);
    if (ready_mailbox != nullptr) aclshmem_free(ready_mailbox);
    if (source_region != nullptr) aclshmem_free(source_region);
    if (shmem_initialized) aclshmem_finalize();
    if (stream != nullptr) aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();
    if (!correct) return Fail("pull Dispatch V2 GET qualification", status);
    std::cout << "[PASS] pe=" << pe << " workers=" << workers
              << " pull_dispatch_v2_get\n";
    return 0;
}
