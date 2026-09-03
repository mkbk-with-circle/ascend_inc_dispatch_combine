#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "acl/acl.h"
#include "shmem.h"
#include "utils.h"

extern "C" void launch_inc_dc_pull_dispatch_v2_relay_probe(
    uint32_t block_dim, void *stream, uint8_t *symmetric_source,
    uint8_t *symmetric_destination, uint8_t *inc_staging,
    uint64_t ffts_addr, uint64_t bytes_per_source, uint32_t worker_count,
    int32_t inc_pe, uint32_t fanout, uint32_t tile_bytes,
    uint32_t channels_per_source);

int g_npus = 5;
const char *ipport = "tcp://127.0.0.1:28796";
int f_pe = 0;
int f_npu = 0;
aclshmemx_uniqueid_t default_flag_uid;

namespace {

int Fail(const char *step, int status)
{
    std::cerr << "[FAIL] " << step << " status=" << status << '\n';
    return status == 0 ? 1 : status;
}

uint8_t SourceByte(uint32_t source, uint64_t offset)
{
    return static_cast<uint8_t>((source * 73u + offset * 29u + 17u) &
                                0xffu);
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 13) {
        std::cerr << "usage: " << argv[0]
                  << " <workers> <pe> <ipport> <first_npu>"
                     " <bytes_per_source> <fanout> <tile_bytes> <aiv>"
                     " <warmup> <measure> <inner_iterations>"
                     " <channels_per_source>\n";
        return 2;
    }
    const uint32_t workers = std::strtoul(argv[1], nullptr, 10);
    const int pe = std::atoi(argv[2]);
    ipport = argv[3];
    f_npu = std::atoi(argv[4]);
    const uint64_t bytes = std::strtoull(argv[5], nullptr, 10);
    const uint32_t fanout = std::strtoul(argv[6], nullptr, 10);
    const uint32_t tile_bytes = std::strtoul(argv[7], nullptr, 10);
    const uint32_t aiv = std::strtoul(argv[8], nullptr, 10);
    const uint32_t warmup = std::strtoul(argv[9], nullptr, 10);
    const uint32_t measure = std::strtoul(argv[10], nullptr, 10);
    const uint32_t inner = std::strtoul(argv[11], nullptr, 10);
    const uint32_t channels = std::strtoul(argv[12], nullptr, 10);
    const uint32_t pes = workers + 1u;
    const int inc_pe = workers;
    g_npus = pes;
    if (workers < 2u || workers > 8u || pe < 0 ||
        pe >= static_cast<int>(pes) || bytes == 0u ||
        bytes > std::numeric_limits<uint32_t>::max() || fanout == 0u ||
        fanout > workers || tile_bytes == 0u || tile_bytes > bytes ||
        aiv == 0u || measure == 0u || inner == 0u || channels == 0u ||
        workers * channels > aiv ||
        bytes > std::numeric_limits<size_t>::max() / workers)
        return Fail("arguments", 2);

    const int device = pe + f_npu;
    aclrtStream stream = nullptr;
    bool shmem_initialized = false;
    uint8_t *source = nullptr;
    uint8_t *destination = nullptr;
    uint8_t *staging = nullptr;
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
        source = static_cast<uint8_t *>(aclshmem_malloc(bytes));
        destination = static_cast<uint8_t *>(aclshmem_malloc(
            bytes * workers));
        staging = static_cast<uint8_t *>(aclshmem_malloc(
            static_cast<uint64_t>(aiv) * 2u * tile_bytes));
        if (source == nullptr || destination == nullptr || staging == nullptr)
            status = 1;
    }
    std::vector<uint8_t> source_host;
    std::vector<uint8_t> zero;
    if (status == 0) {
        source_host.resize(static_cast<size_t>(bytes));
        for (uint64_t offset = 0u; offset < bytes; ++offset)
            source_host[offset] = SourceByte(pe, offset);
        zero.assign(static_cast<size_t>(bytes) * workers, 0u);
        status = aclrtMemcpy(source, bytes, source_host.data(), bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE);
        if (status == 0)
            status = aclrtMemcpy(destination, zero.size(), zero.data(),
                                 zero.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        if (status == 0) {
            zero.assign(static_cast<size_t>(aiv) * 2u * tile_bytes, 0u);
            status = aclrtMemcpy(staging, zero.size(), zero.data(),
                                 zero.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        }
    }
    if (status == 0) aclshmem_barrier_all();

    for (uint32_t i = 0u; status == 0 && i < warmup; ++i) {
        launch_inc_dc_pull_dispatch_v2_relay_probe(
            aiv, stream, source, destination, staging,
            shmemx_get_ffts_config(), bytes, workers, inc_pe, fanout,
            tile_bytes, channels);
        status = aclrtSynchronizeStream(stream);
    }
    std::vector<double> samples;
    for (uint32_t sample = 0u; status == 0 && sample < measure; ++sample) {
        const auto begin = std::chrono::steady_clock::now();
        for (uint32_t i = 0u; i < inner; ++i)
            launch_inc_dc_pull_dispatch_v2_relay_probe(
                aiv, stream, source, destination, staging,
                shmemx_get_ffts_config(), bytes, workers, inc_pe, fanout,
                tile_bytes, channels);
        status = aclrtSynchronizeStream(stream);
        const auto end = std::chrono::steady_clock::now();
        if (pe == inc_pe && status == 0) {
            const double seconds = std::chrono::duration<double>(end - begin)
                .count();
            const double logical_bytes = static_cast<double>(inner) *
                workers * bytes * (1u + fanout);
            samples.push_back(logical_bytes / seconds / 1e9);
        }
    }
    if (status == 0) aclshmem_barrier_all();

    bool correct = status == 0;
    if (correct && pe < inc_pe) {
        std::vector<uint8_t> actual(static_cast<size_t>(bytes) * workers);
        status = aclrtMemcpy(actual.data(), actual.size(), destination,
                             actual.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        correct = status == 0;
        for (uint32_t source_rank = 0u;
             correct && source_rank < workers; ++source_rank) {
            const uint32_t distance =
                (static_cast<uint32_t>(pe) + workers - source_rank) %
                workers;
            const bool selected = distance < fanout;
            for (uint64_t offset = 0u; correct && offset < bytes; ++offset) {
                const uint8_t expected = selected
                    ? SourceByte(source_rank, offset) : 0u;
                const uint8_t value = actual[
                    static_cast<uint64_t>(source_rank) * bytes + offset];
                if (value != expected) {
                    std::cerr << "[FAIL] destination=" << pe
                              << " source=" << source_rank
                              << " offset=" << offset
                              << " actual=" << static_cast<uint32_t>(value)
                              << " expected="
                              << static_cast<uint32_t>(expected) << '\n';
                    correct = false;
                }
            }
        }
    }
    if (correct && pe == inc_pe) {
        double mean = 0.0;
        double minimum = std::numeric_limits<double>::max();
        for (double value : samples) {
            mean += value;
            if (value < minimum) minimum = value;
        }
        mean /= samples.size();
        double variance = 0.0;
        for (double value : samples)
            variance += (value - mean) * (value - mean);
        variance /= samples.size();
        const double cv = std::sqrt(variance) / mean;
        std::cout << "PULL_DISPATCH_V2_RELAY workers=" << workers
                  << " bytes_per_source=" << bytes
                  << " fanout=" << fanout
                  << " tile_bytes=" << tile_bytes
                  << " aiv=" << aiv
                  << " channels_per_source=" << channels
                  << " warmup=" << warmup
                  << " measure=" << measure << " inner=" << inner
                  << " min_gb_s=" << minimum << " mean_gb_s=" << mean
                  << " cv=" << cv << '\n';
    }
    if (status == 0) aclshmem_barrier_all();

    if (staging != nullptr) aclshmem_free(staging);
    if (destination != nullptr) aclshmem_free(destination);
    if (source != nullptr) aclshmem_free(source);
    if (shmem_initialized) aclshmem_finalize();
    if (stream != nullptr) aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();
    if (!correct) return Fail("pull relay probe", status);
    std::cout << "[PASS] pe=" << pe << " workers=" << workers
              << " pull_dispatch_v2_relay\n";
    return 0;
}
