#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "acl/acl.h"
#include "shmem.h"
#include "utils.h"

extern "C" void launch_inc_dc_pull_combine_transport_probe(
    uint32_t block_dim, void *stream, uint8_t *local_destination,
    uint8_t *symmetric_source, uint32_t bytes, int32_t inc_pe,
    int32_t worker_count, uint32_t lanes_per_worker, uint32_t warmup,
    uint32_t iterations);

int g_npus = 2;
const char *ipport = "tcp://127.0.0.1:28770";
int f_pe = 0;
int f_npu = 0;
aclshmemx_uniqueid_t default_flag_uid;

namespace {

int Fail(const char *step, int status)
{
    std::cerr << "[FAIL] " << step << " status=" << status << '\n';
    return status == 0 ? 1 : status;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 11) {
        std::cerr << "usage: " << argv[0]
                  << " <pes> <pe> <ipport> <gpus> <first_npu> <bytes>"
                     " <inc_pe> <lanes_per_worker> <warmup> <iterations>\n";
        return 2;
    }
    const int pes = std::atoi(argv[1]);
    const int pe = std::atoi(argv[2]);
    ipport = argv[3];
    g_npus = std::atoi(argv[4]);
    f_npu = std::atoi(argv[5]);
    const uint64_t bytes64 = std::strtoull(argv[6], nullptr, 10);
    const int inc_pe = std::atoi(argv[7]);
    const uint32_t lanes_per_worker = static_cast<uint32_t>(std::strtoul(
        argv[8], nullptr, 10));
    const uint32_t warmup = static_cast<uint32_t>(std::strtoul(
        argv[9], nullptr, 10));
    const uint32_t iterations = static_cast<uint32_t>(std::strtoul(
        argv[10], nullptr, 10));
    const int worker_count = pes - 1;
    if (pes < 2 || pe < 0 || pe >= pes || g_npus < pes ||
        inc_pe < 0 || inc_pe >= pes || bytes64 == 0u ||
        bytes64 > std::numeric_limits<uint32_t>::max() || iterations == 0u ||
        lanes_per_worker == 0u ||
        bytes64 > std::numeric_limits<size_t>::max() /
            static_cast<uint64_t>(worker_count)) {
        return Fail("arguments", 2);
    }

    const int32_t device = pe % g_npus + f_npu;
    aclrtStream stream = nullptr;
    uint8_t *source = nullptr;
    uint8_t *destination = nullptr;
    bool shmem_initialized = false;
    int status = aclInit(nullptr);
    if (status == 0) status = aclrtSetDevice(device);
    int64_t vector_cores = 0;
    if (status == 0)
        status = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                    &vector_cores);
    if (status == 0 && (vector_cores <= 0 ||
        static_cast<uint64_t>(worker_count) * lanes_per_worker >
            static_cast<uint64_t>(vector_cores))) {
        status = 2;
    }
    if (status == 0) status = aclrtCreateStream(&stream);
    if (status == 0) {
        aclshmemx_init_attr_t attr;
        const uint64_t heap_bytes = 1024ull * 1024ull * 1024ull;
        test_set_attr(pe, pes, heap_bytes, ipport, default_flag_uid, &attr);
        status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
        shmem_initialized = status == 0;
    }
    if (status == 0) {
        source = static_cast<uint8_t *>(aclshmem_malloc(bytes64));
        destination = static_cast<uint8_t *>(aclshmem_malloc(
            bytes64 * static_cast<uint64_t>(worker_count)));
        if (source == nullptr || destination == nullptr) status = 1;
    }

    std::vector<uint8_t> source_host;
    std::vector<uint8_t> zero_host;
    if (status == 0) {
        source_host.resize(static_cast<size_t>(bytes64));
        zero_host.assign(static_cast<size_t>(bytes64) * worker_count, 0u);
        for (uint64_t i = 0u; i < bytes64; ++i)
            source_host[static_cast<size_t>(i)] = static_cast<uint8_t>(
                (pe * 73u + i * 29u + 17u) & 0xffu);
        status = aclrtMemcpy(source, bytes64, source_host.data(), bytes64,
                             ACL_MEMCPY_HOST_TO_DEVICE);
        if (status == 0)
            status = aclrtMemcpy(destination, zero_host.size(),
                                 zero_host.data(), zero_host.size(),
                                 ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (status == 0) {
        std::cerr << "[STAGE] pe=" << pe << " initialized\n" << std::flush;
        aclshmem_barrier_all();
        std::cerr << "[STAGE] pe=" << pe << " ready\n" << std::flush;
    }
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    if (status == 0) {
        start = std::chrono::steady_clock::now();
        launch_inc_dc_pull_combine_transport_probe(
            static_cast<uint32_t>(worker_count) * lanes_per_worker, stream,
            destination, source, static_cast<uint32_t>(bytes64), inc_pe,
            worker_count, lanes_per_worker, warmup, iterations);
        status = aclrtSynchronizeStream(stream);
        end = std::chrono::steady_clock::now();
        std::cerr << "[STAGE] pe=" << pe << " kernel_status=" << status
                  << '\n' << std::flush;
    }

    bool correct = status == 0;
    if (correct && pe == inc_pe) {
        std::vector<uint8_t> actual(
            static_cast<size_t>(bytes64) * worker_count);
        status = aclrtMemcpy(actual.data(), actual.size(), destination,
                             actual.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        correct = status == 0;
        for (uint32_t lane = 0u; correct && lane <
             static_cast<uint32_t>(worker_count); ++lane) {
            const int source_pe = lane >= static_cast<uint32_t>(inc_pe)
                ? static_cast<int>(lane + 1u) : static_cast<int>(lane);
            for (uint64_t i = 0u; correct && i < bytes64; ++i) {
                const uint8_t expected = static_cast<uint8_t>(
                    (source_pe * 73u + i * 29u + 17u) & 0xffu);
                const size_t index = static_cast<size_t>(lane) * bytes64 + i;
                if (actual[index] != expected) {
                    std::cerr << "[FAIL] source=" << source_pe
                              << " byte=" << i << " actual="
                              << static_cast<uint32_t>(actual[index])
                              << " expected="
                              << static_cast<uint32_t>(expected) << '\n';
                    correct = false;
                }
            }
        }
        const double seconds = std::chrono::duration<double>(end - start).count();
        const double measured_bytes = static_cast<double>(bytes64) *
            worker_count * (warmup + iterations);
        const double steady_bytes = static_cast<double>(bytes64) *
            worker_count * iterations;
        // Both are printed: total is the directly measured host interval;
        // steady is a conservative estimate that charges warmup time but not
        // warmup bytes to the numerator.
        std::cout << "[BW] workers=" << worker_count
                  << " bytes_per_worker=" << bytes64
                  << " lanes_per_worker=" << lanes_per_worker
                  << " warmup=" << warmup
                  << " iterations=" << iterations
                  << " seconds=" << seconds
                  << " aggregate_total_GBps="
                  << measured_bytes / seconds / 1e9
                  << " aggregate_conservative_GBps="
                  << steady_bytes / seconds / 1e9 << '\n';
    }

    if (status == 0) {
        aclshmem_barrier_all();
        std::cerr << "[STAGE] pe=" << pe << " verified=" << correct
                  << '\n' << std::flush;
    }

    if (destination != nullptr) aclshmem_free(destination);
    if (source != nullptr) aclshmem_free(source);
    if (shmem_initialized) aclshmem_finalize();
    if (stream != nullptr) aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();

    if (!correct) return Fail("transport visibility", status);
    std::cout << "[PASS] pe=" << pe << " bytes=" << bytes64
              << " transport-selected-by-shmem\n";
    return 0;
}
