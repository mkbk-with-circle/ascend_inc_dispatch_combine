#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include "acl/acl.h"
#include "shmem.h"
#include "utils.h"

#include "inc_dc_pull_combine_abi.h"
#include "inc_dc_pull_combine_device_e2e_abi.h"

using namespace inc::dc::pull_combine;

extern "C" void launch_inc_dc_pull_combine_device_e2e(
    uint32_t block_dim, void *stream, uint8_t *symmetric_partials,
    uint8_t *inc_staging, uint8_t *reduced_output,
    uint8_t *descriptor_mailbox, uint8_t *ack_mailbox,
    uint8_t *status_line, uint64_t ffts_addr, uint32_t elements,
    uint32_t worker_count, uint32_t lanes_per_worker, int32_t inc_pe,
    uint64_t generation, uint32_t wave, uint64_t digest);

int g_npus = 5;
const char *ipport = "tcp://127.0.0.1:28780";
int f_pe = 0;
int f_npu = 0;
aclshmemx_uniqueid_t default_flag_uid;

namespace {

constexpr uint64_t kGeneration = 7u;
constexpr uint32_t kWave = 3u;
constexpr uint64_t kDigest = 0x9c3d2a71b5e4806full;

float PartialValue(uint32_t worker, uint64_t element)
{
    return static_cast<float>(worker + 1u) * 0.25f +
           static_cast<float>(element % 17u) * 0.03125f;
}

int Fail(const char *step, int status)
{
    std::cerr << "[FAIL] " << step << " status=" << status << '\n';
    return status == 0 ? 1 : status;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 7) {
        std::cerr << "usage: " << argv[0]
                  << " <workers> <pe> <ipport> <first_npu> <bytes>"
                     " <lanes_per_worker; 0=half-AIV auto>\n";
        return 2;
    }
    const uint32_t workers = static_cast<uint32_t>(std::strtoul(
        argv[1], nullptr, 10));
    const int pe = std::atoi(argv[2]);
    ipport = argv[3];
    f_npu = std::atoi(argv[4]);
    const uint64_t bytes = std::strtoull(argv[5], nullptr, 10);
    uint32_t lanes = static_cast<uint32_t>(std::strtoul(
        argv[6], nullptr, 10));
    const uint32_t pes = workers + 1u;
    const int inc_pe = static_cast<int>(workers);
    g_npus = static_cast<int>(pes);
    if (workers < 2u || workers > kPullCombineMaxWorkers || pe < 0 ||
        pe >= static_cast<int>(pes) || bytes == 0u ||
        bytes % sizeof(float) != 0u ||
        bytes > std::numeric_limits<uint32_t>::max() ||
        bytes / sizeof(float) > std::numeric_limits<uint32_t>::max()) {
        return Fail("arguments", 2);
    }
    const uint32_t elements = static_cast<uint32_t>(bytes / sizeof(float));
    constexpr uint64_t kFloatsPerCacheLine = 64u / sizeof(float);
    const uint64_t padded_elements =
        (static_cast<uint64_t>(elements) + kFloatsPerCacheLine - 1u) /
        kFloatsPerCacheLine * kFloatsPerCacheLine;
    const uint64_t padded_bytes = padded_elements * sizeof(float);
    const int32_t device = pe + f_npu;
    aclrtStream stream = nullptr;
    bool shmem_initialized = false;
    uint8_t *partials = nullptr;
    uint8_t *staging = nullptr;
    uint8_t *output = nullptr;
    uint8_t *descriptors = nullptr;
    uint8_t *acks = nullptr;
    uint8_t *status_line = nullptr;
    double device_e2e_us = 0.0;
    DeviceE2eTimeline timeline{};

    int status = aclInit(nullptr);
    if (status == 0) status = aclrtSetDevice(device);
    int64_t vector_cores = 0;
    if (status == 0)
        status = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                    &vector_cores);
    if (status == 0 && lanes == 0u && vector_cores > 0) {
        // Dispatch and Combine each own half of the live AIV budget.  Derive
        // the per-worker share from the actual device instead of encoding
        // 910B's 40 AIVs or a particular W2/W4 shape in the protocol.
        const uint64_t combine_aiv = static_cast<uint64_t>(vector_cores) / 2u;
        lanes = static_cast<uint32_t>(combine_aiv / workers);
    }
    if (status == 0 && (vector_cores <= 0 ||
        lanes == 0u || static_cast<uint64_t>(workers) * lanes >
            static_cast<uint64_t>(vector_cores)))
        status = 2;
    if (status == 0) status = aclrtCreateStream(&stream);
    if (status == 0) {
        aclshmemx_init_attr_t attr;
        test_set_attr(pe, pes, 1024ull * 1024ull * 1024ull, ipport,
                      default_flag_uid, &attr);
        status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
        shmem_initialized = status == 0;
    }
    if (status == 0) {
        partials = static_cast<uint8_t *>(aclshmem_malloc(padded_bytes));
        staging = static_cast<uint8_t *>(aclshmem_malloc(
            padded_bytes * workers));
        output = static_cast<uint8_t *>(aclshmem_malloc(
            padded_bytes * workers));
        descriptors = static_cast<uint8_t *>(aclshmem_malloc(
            sizeof(CombineReadyDescriptor) * workers));
        acks = static_cast<uint8_t *>(aclshmem_malloc(
            sizeof(CombineAck) * workers));
        status_line = static_cast<uint8_t *>(aclshmem_malloc(64u));
        if (partials == nullptr || staging == nullptr || output == nullptr ||
            descriptors == nullptr || acks == nullptr ||
            status_line == nullptr)
            status = 1;
    }

    std::vector<float> partial_host(padded_elements, 0.0f);
    std::vector<uint8_t> zero_bytes;
    if (status == 0) {
        for (uint32_t i = 0u; i < elements; ++i)
            partial_host[i] = PartialValue(static_cast<uint32_t>(pe), i);
        status = aclrtMemcpy(partials, padded_bytes, partial_host.data(),
                             padded_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        zero_bytes.assign(static_cast<size_t>(padded_bytes * workers), 0u);
        if (status == 0)
            status = aclrtMemcpy(staging, zero_bytes.size(),
                                 zero_bytes.data(), zero_bytes.size(),
                                 ACL_MEMCPY_HOST_TO_DEVICE);
        zero_bytes.assign(static_cast<size_t>(padded_bytes * workers), 0u);
        if (status == 0)
            status = aclrtMemcpy(output, zero_bytes.size(), zero_bytes.data(),
                                 zero_bytes.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        zero_bytes.assign(sizeof(CombineReadyDescriptor) * workers, 0u);
        if (status == 0)
            status = aclrtMemcpy(descriptors, zero_bytes.size(),
                                 zero_bytes.data(), zero_bytes.size(),
                                 ACL_MEMCPY_HOST_TO_DEVICE);
        zero_bytes.assign(sizeof(CombineAck) * workers, 0u);
        if (status == 0)
            status = aclrtMemcpy(acks, zero_bytes.size(), zero_bytes.data(),
                                 zero_bytes.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        uint32_t zero_status[16]{};
        if (status == 0)
            status = aclrtMemcpy(status_line, sizeof(zero_status),
                                 zero_status, sizeof(zero_status),
                                 ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (status == 0) aclshmem_barrier_all();

    if (status == 0 && pe < inc_pe) {
        CombineReadyDescriptor descriptor{};
        descriptor.generation = kGeneration;
        descriptor.sequence = 1u;
        descriptor.wave = kWave;
        descriptor.source_rank = static_cast<uint32_t>(pe);
        descriptor.combine_row_begin = static_cast<uint64_t>(pe) * elements;
        descriptor.row_count = elements;
        descriptor.partial_dtype = static_cast<uint32_t>(PartialDType::FP32);
        descriptor.source_region_id = 1u + static_cast<uint64_t>(pe);
        descriptor.source_offset = 0u;
        descriptor.payload_bytes = bytes;
        descriptor.semantic_digest = kDigest;
        uint8_t *slot = descriptors + static_cast<uint64_t>(pe) *
            sizeof(CombineReadyDescriptor);
        status = aclrtMemcpy(slot, sizeof(descriptor), &descriptor,
                             sizeof(descriptor), ACL_MEMCPY_HOST_TO_DEVICE);
        if (status == 0)
            aclshmem_putmem(slot, slot, sizeof(descriptor), inc_pe);
    }
    if (status == 0) aclshmem_barrier_all();

    if (status == 0) {
        const auto begin = std::chrono::steady_clock::now();
        launch_inc_dc_pull_combine_device_e2e(
            workers * lanes, stream, partials, staging, output, descriptors,
            acks, status_line, shmemx_get_ffts_config(), elements, workers,
            lanes, inc_pe, kGeneration, kWave, kDigest);
        status = aclrtSynchronizeStream(stream);
        const auto end = std::chrono::steady_clock::now();
        device_e2e_us = std::chrono::duration<double, std::micro>(
                            end - begin).count();
    }
    if (status == 0) aclshmem_barrier_all();

    bool correct = status == 0;
    if (correct && pe == inc_pe) {
        status = aclrtMemcpy(&timeline, sizeof(timeline), status_line,
                             sizeof(timeline), ACL_MEMCPY_DEVICE_TO_HOST);
        correct = status == 0 && timeline.status == 0u &&
                  timeline.reserved == 0u;
    }
    if (correct && pe < inc_pe) {
        CombineAck ack{};
        status = aclrtMemcpy(&ack, sizeof(ack),
                             acks + static_cast<uint64_t>(pe) * sizeof(ack),
                             sizeof(ack), ACL_MEMCPY_DEVICE_TO_HOST);
        correct = status == 0 && ack.magic == kPullCombineMagic &&
            ack.abi_version == kPullCombineAbiVersion &&
            ack.struct_bytes == sizeof(CombineAck) &&
            ack.generation == kGeneration && ack.sequence == 1u &&
            ack.source_rank == static_cast<uint32_t>(pe) &&
            ack.status == 0u && ack.rows_consumed == elements;

        const uint64_t begin = static_cast<uint64_t>(elements) * pe / workers;
        const uint64_t end = static_cast<uint64_t>(elements) * (pe + 1u) /
                             workers;
        std::vector<float> actual(static_cast<size_t>(end - begin));
        if (correct && !actual.empty())
            status = aclrtMemcpy(actual.data(), actual.size() * sizeof(float),
                                 output,
                                 actual.size() * sizeof(float),
                                 ACL_MEMCPY_DEVICE_TO_HOST);
        correct = correct && status == 0;
        uint32_t mismatches = 0u;
        for (uint64_t element = begin; correct && element < end; ++element) {
            float expected = 0.0f;
            for (uint32_t worker = 0u; worker < workers; ++worker)
                expected += PartialValue(worker, element);
            if (std::fabs(actual[static_cast<size_t>(element - begin)] -
                          expected) > 1e-6f) {
                if (mismatches < 8u)
                    std::cerr << "[FAIL] pe=" << pe
                              << " element=" << element << " actual="
                              << actual[static_cast<size_t>(element - begin)]
                              << " expected=" << expected << '\n';
                ++mismatches;
            }
        }
        correct = correct && mismatches == 0u;
    }

    if (status == 0) aclshmem_barrier_all();
    if (status_line != nullptr) aclshmem_free(status_line);
    if (acks != nullptr) aclshmem_free(acks);
    if (descriptors != nullptr) aclshmem_free(descriptors);
    if (output != nullptr) aclshmem_free(output);
    if (staging != nullptr) aclshmem_free(staging);
    if (partials != nullptr) aclshmem_free(partials);
    if (shmem_initialized) aclshmem_finalize();
    if (stream != nullptr) aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();

    if (!correct) return Fail("device e2e", status);
    std::cout << "[PASS] pe=" << pe << " workers=" << workers
              << " bytes=" << bytes << " lanes=" << lanes;
    if (pe == inc_pe && device_e2e_us > 0.0) {
        const double logical_rma_bytes =
            static_cast<double>(bytes) * (workers + 1u);
        std::cout << " e2e_us=" << device_e2e_us
                  << " logical_rma_gb_s="
                  << logical_rma_bytes / device_e2e_us / 1.0e3;
        const uint64_t total_cycles =
            timeline.cycle[kTimelineEgressDone] -
            timeline.cycle[kTimelineStart];
        if (total_cycles != 0u) {
            std::cout << " phase_pct=";
            for (uint32_t point = 1u; point < kTimelinePointCount; ++point) {
                if (point != 1u) std::cout << ',';
                const uint64_t phase_cycles = timeline.cycle[point] -
                    timeline.cycle[point - 1u];
                std::cout << 100.0 * static_cast<double>(phase_cycles) /
                    static_cast<double>(total_cycles);
            }
        }
    }
    std::cout << '\n';
    return 0;
}
