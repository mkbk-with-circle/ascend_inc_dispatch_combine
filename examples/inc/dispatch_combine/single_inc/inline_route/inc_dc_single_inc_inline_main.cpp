#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "param.h"
#include "shmem.h"
#include "utils.h"

#include "inc_dc_inline_route_protocol.h"
#include "inc_dc_physical_map.h"
#include "inc_dc_resource_policy.h"
#include "inc_dc_single_inc_inline_bench_abi.h"

using namespace inc::dc;
using namespace inc::dc::single_inline;

extern "C" void launch_inc_dc_single_inc_inline_dispatch_kernel(
    uint8_t *sym, int block_dim, void *stream);

int g_npus = 16;
const char *ipport = "tcp://127.0.0.1:8969";
int f_npu = 0;
aclshmemx_uniqueid_t default_flag_uid;

namespace {

uint8_t PayloadByte(uint32_t source, uint64_t token, uint32_t byte)
{
    return static_cast<uint8_t>((source * 29u + token * 17u + byte * 13u +
                                 7u) & 0xffu);
}

uint32_t FloatBits(float value)
{
    uint32_t bits = 0u;
    static_assert(sizeof(bits) == sizeof(value), "float32 required");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

uint32_t LogicalExpert(uint32_t workers, uint32_t source, uint64_t token,
                       uint32_t ordinal)
{
    const uint32_t destination =
        (source + static_cast<uint32_t>(token % workers) + ordinal) % workers;
    return ordinal * workers + destination;
}

int ResolvePhysicalNpuForPe(int pe)
{
    return inc::dc::ResolvePhysicalNpuForPe(pe, g_npus, f_npu);
}

int InitShmem(int pe, int npe, uint64_t heap_bytes, int32_t *device,
              aclrtStream *stream)
{
    *device = ResolvePhysicalNpuForPe(pe);
    int status = aclInit(nullptr);
    if (status == 0) status = aclrtSetDevice(*device);
    if (status == 0) status = aclrtCreateStream(stream);
    if (status != 0) return status;
    aclshmemx_init_attr_t attr;
    test_set_attr(pe, npe, heap_bytes, ipport, default_flag_uid, &attr);
    return aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
}

void Cleanup(int32_t device, aclrtStream stream, uint8_t *symmetric)
{
    if (symmetric != nullptr) aclshmem_free(symmetric);
    aclshmem_finalize();
    if (stream != nullptr) aclrtDestroyStream(stream);
    if (device >= 0) aclrtResetDevice(device);
    aclFinalize();
}

bool QueryLiveAivForPlanning(int pe, uint32_t *aiv)
{
    if (aiv == nullptr) return false;
    const int32_t device = ResolvePhysicalNpuForPe(pe);
    int status = aclInit(nullptr);
    if (status == 0) status = aclrtSetDevice(device);
    int64_t value = 0;
    if (status == 0)
        status = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                    &value);
    if (status == 0) status = aclrtResetDevice(device);
    const int finalize_status = aclFinalize();
    if (status == 0 && finalize_status != 0) status = finalize_status;
    if (status != 0 || value <= 0 || value > kInlineDispatchMaxLanes)
        return false;
    *aiv = static_cast<uint32_t>(value);
    return true;
}

bool AddRegion(uint64_t bytes, uint64_t *cursor, uint64_t *offset)
{
    uint64_t aligned = 0u;
    uint64_t end = 0u;
    if (cursor == nullptr || offset == nullptr ||
        !InlineCheckedAlignUp(*cursor, 256u, &aligned) ||
        !InlineCheckedAdd(aligned, bytes, &end)) {
        return false;
    }
    *offset = aligned;
    *cursor = end;
    return true;
}

bool Mul(uint64_t a, uint64_t b, uint64_t *out)
{
    return InlineCheckedMul(a, b, out);
}

int CopyH2D(uint8_t *dst, const uint8_t *src, uint64_t bytes)
{
    constexpr uint64_t chunk = 64ull * 1024ull * 1024ull;
    for (uint64_t offset = 0u; offset < bytes; offset += chunk) {
        const uint64_t count = std::min(chunk, bytes - offset);
        const int status = aclrtMemcpy(dst + offset, count, src + offset,
                                       count, ACL_MEMCPY_HOST_TO_DEVICE);
        if (status != 0) return status;
    }
    return 0;
}

int CopyD2H(uint8_t *dst, const uint8_t *src, uint64_t bytes)
{
    constexpr uint64_t chunk = 64ull * 1024ull * 1024ull;
    for (uint64_t offset = 0u; offset < bytes; offset += chunk) {
        const uint64_t count = std::min(chunk, bytes - offset);
        const int status = aclrtMemcpy(dst + offset, count, src + offset,
                                       count, ACL_MEMCPY_DEVICE_TO_HOST);
        if (status != 0) return status;
    }
    return 0;
}

bool VerifyFanout(uint8_t *symmetric,
                  const InlineDispatchBenchDescV2 &desc,
                  uint32_t generation, uint64_t *verified_records,
                  uint64_t *verified_assignments,
                  uint64_t *verified_payload_bytes)
{
    if (desc.pe >= desc.worker_count || verified_records == nullptr ||
        verified_assignments == nullptr ||
        verified_payload_bytes == nullptr) {
        return desc.pe == desc.inc_pe;
    }
    *verified_records = 0u;
    *verified_assignments = 0u;
    *verified_payload_bytes = 0u;
    const uint32_t slot = generation % desc.slot_count;
    const bool dense_segmented =
        desc.max_topk == desc.worker_count;
    std::vector<uint8_t> record(static_cast<size_t>(
        dense_segmented ? desc.dense_fanout_metadata_stride
                        : desc.fanout_record_stride));
    std::vector<uint8_t> payload_bytes(desc.hidden_bytes);
    for (uint32_t source = 0u; source < desc.worker_count; ++source) {
        for (uint32_t token = 0u; token < desc.tokens_per_worker; ++token) {
            std::vector<uint32_t> expected_ordinals;
            for (uint32_t ordinal = 0u; ordinal < desc.max_topk; ++ordinal) {
                const uint32_t expert = LogicalExpert(
                    desc.worker_count, source, token, ordinal);
                if (expert % desc.worker_count == desc.pe)
                    expected_ordinals.push_back(ordinal);
            }
            if (expected_ordinals.empty()) continue;
            const uint64_t global_token =
                static_cast<uint64_t>(source) * desc.tokens_per_worker +
                token;
            const uint64_t record_index =
                (static_cast<uint64_t>(slot) * desc.worker_count + desc.pe) *
                    desc.worker_count * desc.tokens_per_worker +
                global_token;
            const uint64_t metadata_stride = dense_segmented
                ? desc.dense_fanout_metadata_stride
                : desc.fanout_record_stride;
            const uint64_t metadata_off = dense_segmented
                ? desc.dense_fanout_metadata_off
                : desc.fanout_record_off;
            if (CopyD2H(record.data(),
                        symmetric + metadata_off +
                            record_index * metadata_stride,
                        metadata_stride) != 0) {
                return false;
            }
            InlineFanoutRecordHeaderV2 header{};
            std::memcpy(&header, record.data(), sizeof(header));
            if (header.magic != kInlineRouteFanoutMagic ||
                header.version != kInlineRouteVersion ||
                header.header_bytes != kInlineRouteHeaderBytes ||
                header.resolved_destination_rank != desc.pe ||
                header.session_id != desc.session_id ||
                header.placement_epoch != desc.placement_epoch ||
                header.generation != generation ||
                header.request_id != desc.request_id ||
                header.source_rank != source ||
                header.source_token != token ||
                header.hidden_bytes != desc.hidden_bytes ||
                header.assignment_count != expected_ordinals.size() ||
                header.commit != generation ||
                (!dense_segmented &&
                 header.record_bytes > desc.fanout_record_stride) ||
                (dense_segmented &&
                 header.payload_offset >
                     desc.dense_fanout_metadata_stride) ||
                header.payload_offset + header.hidden_bytes !=
                    header.record_bytes) {
                return false;
            }
            if (dense_segmented) {
                if (CopyD2H(payload_bytes.data(),
                            symmetric + desc.dense_fanout_payload_off +
                                record_index * desc.hidden_bytes,
                            desc.hidden_bytes) != 0)
                    return false;
            } else {
                std::memcpy(payload_bytes.data(),
                            record.data() + header.payload_offset,
                            desc.hidden_bytes);
            }
            const auto *assignments =
                reinterpret_cast<const InlineFanoutAssignmentV2 *>(
                    record.data() + sizeof(InlineFanoutRecordHeaderV2));
            for (size_t index = 0u; index < expected_ordinals.size(); ++index) {
                const uint32_t ordinal = expected_ordinals[index];
                const uint32_t expert = LogicalExpert(
                    desc.worker_count, source, token, ordinal);
                if (assignments[index].expert_id != expert ||
                    assignments[index].local_expert !=
                        expert / desc.worker_count ||
                    assignments[index].route_ordinal != ordinal ||
                    assignments[index].route_key.journal_locator == 0u ||
                    assignments[index].route_key.authenticator == 0u) {
                    return false;
                }
            }
            for (uint32_t byte = 0u; byte < desc.hidden_bytes; ++byte) {
                if (payload_bytes[byte] != PayloadByte(source, token, byte))
                    return false;
            }
            ++*verified_records;
            *verified_assignments += header.assignment_count;
            *verified_payload_bytes += header.hidden_bytes;
        }
    }
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 10) {
        std::cerr << "usage: inc_dc_single_inc_inline <npe> <pe> <ipport> "
                     "<g_npus> <f_npu> <tokens_per_worker> <hidden_bytes> "
                     "<topk> <mode:dispatch|combine|overlap> "
                     "[batch_tokens] [warmup] [measure] [inc_lanes] "
                     "[worker_lanes]\n";
        return 2;
    }
    const int npe = std::atoi(argv[1]);
    const int pe = std::atoi(argv[2]);
    ipport = argv[3];
    g_npus = std::atoi(argv[4]);
    f_npu = std::atoi(argv[5]);
    const uint32_t workers = npe > 0 ? static_cast<uint32_t>(npe - 1) : 0u;
    const uint32_t tokens =
        static_cast<uint32_t>(std::strtoul(argv[6], nullptr, 10));
    const uint32_t hidden_bytes =
        static_cast<uint32_t>(std::strtoul(argv[7], nullptr, 10));
    const uint32_t topk =
        static_cast<uint32_t>(std::strtoul(argv[8], nullptr, 10));
    const std::string mode = argv[9];
    if (mode != "dispatch") {
        if (mode == "combine" || mode == "overlap") {
            std::cerr << "INLINE_V2_MODE_NOT_IMPLEMENTED mode=" << mode
                      << " timing_emitted=0\n";
            return 2;
        }
        std::cerr << "INLINE_V2_BAD_MODE mode=" << mode << '\n';
        return 2;
    }
    const uint64_t transport_window_bytes =
        static_cast<uint64_t>(inc::dc::kIncDcPrivateMtePacketBytes) * 2u;
    const uint32_t derived_batch_tokens = static_cast<uint32_t>(
        std::max<uint64_t>(
            1u, std::min<uint64_t>(
                    tokens, hidden_bytes == 0u
                                ? 1u
                                : transport_window_bytes / hidden_bytes)));
    const uint32_t batch_tokens = argc > 10
        ? static_cast<uint32_t>(std::strtoul(argv[10], nullptr, 10))
        : derived_batch_tokens;
    const uint32_t warmup = argc > 11
        ? static_cast<uint32_t>(std::strtoul(argv[11], nullptr, 10))
        : 2u;
    const uint32_t measure = argc > 12
        ? static_cast<uint32_t>(std::strtoul(argv[12], nullptr, 10))
        : 5u;
    uint32_t inc_lanes = argc > 13
        ? static_cast<uint32_t>(std::strtoul(argv[13], nullptr, 10))
        : 0u;
    uint32_t worker_lanes = argc > 14
        ? static_cast<uint32_t>(std::strtoul(argv[14], nullptr, 10))
        : 0u;
    if ((workers != 2u && workers != 4u) || pe < 0 || pe >= npe ||
        tokens == 0u || hidden_bytes == 0u || topk == 0u || topk > workers ||
        batch_tokens == 0u || batch_tokens > tokens || measure == 0u) {
        return 3;
    }

    inc::dc::IncDcPhysicalMapValidation physical_map{};
    std::string physical_map_error;
    if (!inc::dc::ValidatePhysicalNpuMap(
            npe, g_npus, f_npu, true, &physical_map,
            &physical_map_error)) {
        std::cerr << "INLINE_V2_PHYSICAL_MAP_FAIL pe=" << pe
                  << " reason=" << physical_map_error << '\n';
        return 3;
    }

    InlineDispatchBenchDescV2 desc{};
    desc.pe = static_cast<uint32_t>(pe);
    desc.inc_pe = workers;
    desc.worker_count = workers;
    desc.expert_count = workers * topk;
    desc.hidden_bytes = hidden_bytes;
    desc.max_topk = topk;
    desc.tokens_per_worker = tokens;
    desc.batch_tokens = batch_tokens;
    // Bound every protocol wait so a malformed generation fails closed and
    // returns diagnostics instead of leaving a persistent AIV kernel behind.
    desc.spin_cap = 10000000u;
    desc.slot_count = 1u;
    desc.session_id = 0x763273657373696full;
    desc.placement_epoch = 1u;
    desc.request_id = 1u;

    const uint64_t token_payload_offset =
        (sizeof(InlineTokenRecordHeaderV2) +
             static_cast<uint64_t>(topk) * sizeof(InlineRouteEntryV2) + 63u) &
        ~63ull;
    const uint64_t stored_token_stride =
        (token_payload_offset + hidden_bytes + 63u) & ~63ull;
    const uint64_t batch_records_offset =
        (sizeof(InlineTokenBatchHeaderV2) +
             static_cast<uint64_t>(batch_tokens + 1u) * sizeof(uint64_t) +
         63u) &
        ~63ull;
    desc.ingress_frame_stride =
        (batch_records_offset +
             static_cast<uint64_t>(batch_tokens) * stored_token_stride +
         63u) &
        ~63ull;
    const uint64_t fanout_payload_offset =
        (sizeof(InlineFanoutRecordHeaderV2) +
             static_cast<uint64_t>(topk) *
                 sizeof(InlineFanoutAssignmentV2) +
         63u) &
        ~63ull;
    desc.fanout_record_stride =
        (fanout_payload_offset + hidden_bytes + 63u) & ~63ull;
    desc.dense_fanout_metadata_stride =
        (sizeof(InlineFanoutRecordHeaderV2) +
             sizeof(InlineFanoutAssignmentV2) + 63u) &
        ~63ull;
    const uint32_t batches =
        tokens / batch_tokens + (tokens % batch_tokens != 0u ? 1u : 0u);

    uint64_t cursor = 4096u;
    uint64_t bytes = 0u;
    bool layout_ok =
        Mul(tokens, hidden_bytes, &bytes) &&
        AddRegion(bytes, &cursor, &desc.source_hidden_off) &&
        Mul(tokens, sizeof(uint32_t), &bytes) &&
        AddRegion(bytes, &cursor, &desc.source_route_count_off) &&
        Mul(tokens, topk, &bytes) &&
        Mul(bytes, sizeof(InlineRouteEntryV2), &bytes) &&
        AddRegion(bytes, &cursor, &desc.source_route_entry_off) &&
        Mul(desc.slot_count, workers, &bytes) &&
        Mul(bytes, batches, &bytes) &&
        Mul(bytes, desc.ingress_frame_stride, &bytes) &&
        AddRegion(bytes, &cursor, &desc.ingress_frame_off) &&
        Mul(desc.slot_count, workers, &bytes) &&
        Mul(bytes, tokens, &bytes) &&
        Mul(bytes, workers, &bytes) &&
        Mul(bytes, desc.fanout_record_stride, &bytes) &&
        AddRegion(bytes, &cursor, &desc.fanout_record_off) &&
        Mul(desc.slot_count, workers, &bytes) &&
        Mul(bytes, tokens, &bytes) &&
        Mul(bytes, workers, &bytes) &&
        Mul(bytes, desc.dense_fanout_metadata_stride, &bytes) &&
        AddRegion(bytes, &cursor, &desc.dense_fanout_metadata_off) &&
        Mul(desc.slot_count, workers, &bytes) &&
        Mul(bytes, tokens, &bytes) &&
        Mul(bytes, workers, &bytes) &&
        Mul(bytes, hidden_bytes, &bytes) &&
        AddRegion(bytes, &cursor, &desc.dense_fanout_payload_off) &&
        Mul(desc.expert_count, sizeof(uint32_t), &bytes) &&
        AddRegion(bytes, &cursor, &desc.expert_owner_off) &&
        AddRegion(bytes, &cursor, &desc.expert_local_index_off) &&
        Mul(desc.slot_count, workers, &bytes) &&
        Mul(bytes, tokens, &bytes) &&
        Mul(bytes, topk, &bytes) &&
        Mul(bytes, sizeof(InlineDispatchJournalEntryV2), &bytes) &&
        AddRegion(bytes, &cursor, &desc.journal_off) &&
        Mul(desc.slot_count, workers, &bytes) &&
        Mul(bytes, batches, &bytes) &&
        Mul(bytes, workers, &bytes) &&
        Mul(bytes, 64u, &bytes) &&
        AddRegion(bytes, &cursor, &desc.fanout_ready_off) &&
        AddRegion(static_cast<uint64_t>(workers * 2u +
                                        kInlineDispatchMaxLanes) * 64u,
                  &cursor, &desc.worker_done_off) &&
        AddRegion(static_cast<uint64_t>(workers + 1u) *
                      kInlineDispatchMaxLanes *
                      sizeof(InlineDispatchBenchStatV2),
                  &cursor, &desc.stats_off);
    if (!layout_ok) return 4;
    desc.total_bytes = (cursor + 4095u) & ~4095ull;
    const uint64_t heap_bytes = std::max<uint64_t>(
        2ull * 1024ull * 1024ull * 1024ull,
        (desc.total_bytes + 64ull * 1024ull * 1024ull +
         2ull * 1024ull * 1024ull - 1u) &
            ~(2ull * 1024ull * 1024ull - 1u));

    aclrtStream stream = nullptr;
    aclrtEvent begin_event = nullptr;
    aclrtEvent end_event = nullptr;
    int32_t device = -1;
    uint8_t *symmetric = nullptr;
    int status = InitShmem(pe, npe, heap_bytes, &device, &stream);
    if (status != 0) {
        std::cerr << "INLINE_V2_SHMEM_INIT_FAIL pe=" << pe
                  << " status=" << status << '\n';
        return 10;
    }
    int64_t live_aiv_raw = 0;
    status = aclrtGetDeviceInfo(
        device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &live_aiv_raw);
    if (status != 0 || live_aiv_raw <= 0) {
        Cleanup(device, stream, symmetric);
        return 10;
    }
    const uint32_t half_aiv = std::max<uint32_t>(
        1u, static_cast<uint32_t>(live_aiv_raw) / 2u);
    if (inc_lanes == 0u)
        inc_lanes = std::min(half_aiv, kInlineDispatchMaxLanes);
    if (worker_lanes == 0u)
        worker_lanes = std::min(half_aiv, kInlineDispatchMaxLanes);
    if (inc_lanes == 0u || worker_lanes == 0u ||
        inc_lanes > kInlineDispatchMaxLanes ||
        worker_lanes > kInlineDispatchMaxLanes ||
        inc_lanes > static_cast<uint32_t>(live_aiv_raw) ||
        worker_lanes > static_cast<uint32_t>(live_aiv_raw)) {
        Cleanup(device, stream, symmetric);
        return 3;
    }
    desc.inc_lane_count = inc_lanes;
    desc.worker_lane_count = worker_lanes;

    symmetric = static_cast<uint8_t *>(aclshmem_malloc(desc.total_bytes));
    if (symmetric == nullptr) {
        Cleanup(device, stream, symmetric);
        return 11;
    }
    status = aclrtMemset(symmetric, desc.total_bytes, 0, desc.total_bytes);

    std::vector<uint8_t> hidden;
    std::vector<uint32_t> route_counts;
    std::vector<InlineRouteEntryV2> routes;
    if (status == 0 && static_cast<uint32_t>(pe) < workers) {
        hidden.resize(static_cast<size_t>(tokens) * hidden_bytes);
        route_counts.assign(tokens, topk);
        routes.resize(static_cast<size_t>(tokens) * topk);
        for (uint32_t token = 0u; token < tokens; ++token) {
            for (uint32_t byte = 0u; byte < hidden_bytes; ++byte)
                hidden[static_cast<size_t>(token) * hidden_bytes + byte] =
                    PayloadByte(static_cast<uint32_t>(pe), token, byte);
            for (uint32_t ordinal = 0u; ordinal < topk; ++ordinal) {
                auto &route = routes[static_cast<size_t>(token) * topk +
                                     ordinal];
                route.expert_id = LogicalExpert(
                    workers, static_cast<uint32_t>(pe), token, ordinal);
                route.route_ordinal = ordinal;
                route.weight_bits =
                    FloatBits(1.0f / static_cast<float>(topk));
            }
        }
        status = CopyH2D(symmetric + desc.source_hidden_off, hidden.data(),
                         hidden.size());
        if (status == 0)
            status = CopyH2D(symmetric + desc.source_route_count_off,
                             reinterpret_cast<const uint8_t *>(
                                 route_counts.data()),
                             route_counts.size() * sizeof(uint32_t));
        if (status == 0)
            status = CopyH2D(symmetric + desc.source_route_entry_off,
                             reinterpret_cast<const uint8_t *>(routes.data()),
                             routes.size() * sizeof(InlineRouteEntryV2));
    }
    if (status == 0 && static_cast<uint32_t>(pe) == workers) {
        std::vector<uint32_t> owners(desc.expert_count);
        std::vector<uint32_t> local(desc.expert_count);
        for (uint32_t expert = 0u; expert < desc.expert_count; ++expert) {
            owners[expert] = expert % workers;
            local[expert] = expert / workers;
        }
        status = CopyH2D(symmetric + desc.expert_owner_off,
                         reinterpret_cast<const uint8_t *>(owners.data()),
                         owners.size() * sizeof(uint32_t));
        if (status == 0)
            status = CopyH2D(
                symmetric + desc.expert_local_index_off,
                reinterpret_cast<const uint8_t *>(local.data()),
                local.size() * sizeof(uint32_t));
    }
    if (status != 0 || aclrtSynchronizeStream(stream) != 0) {
        Cleanup(device, stream, symmetric);
        return 12;
    }

    std::cout << "INLINE_V2_DISPATCH_CONFIG pe=" << pe
              << " physical_npu=" << device
              << " role=" << (static_cast<uint32_t>(pe) == workers
                                   ? "inc" : "worker")
              << " workers=" << workers << " tokens=" << tokens
              << " hidden_bytes=" << hidden_bytes << " topk=" << topk
              << " batch_tokens=" << batch_tokens
              << " live_aiv=" << live_aiv_raw
              << " inc_lanes=" << inc_lanes
              << " worker_lanes=" << worker_lanes
              << " total_bytes=" << desc.total_bytes
              << " path=complete_token_record_to_inc_online_parse_fanout\n";

    aclrtCreateEvent(&begin_event);
    aclrtCreateEvent(&end_event);
    bool all_pass = true;
    const uint64_t control_bytes =
        static_cast<uint64_t>(workers * 2u + kInlineDispatchMaxLanes) * 64u;
    std::vector<InlineDispatchBenchStatV2> stats(
        static_cast<size_t>(workers + 1u) * kInlineDispatchMaxLanes);
    for (uint32_t epoch = 0u; epoch < warmup + measure; ++epoch) {
        desc.generation = epoch + 1u;
        status = aclrtMemset(symmetric + desc.worker_done_off, control_bytes,
                             0, control_bytes);
        if (status == 0)
            status = aclrtMemset(
                symmetric + desc.stats_off,
                stats.size() * sizeof(InlineDispatchBenchStatV2), 0,
                stats.size() * sizeof(InlineDispatchBenchStatV2));
        if (status == 0)
            status = aclrtMemcpy(symmetric, sizeof(desc), &desc, sizeof(desc),
                                 ACL_MEMCPY_HOST_TO_DEVICE);
        if (status != 0 || aclrtSynchronizeStream(stream) != 0) {
            all_pass = false;
            break;
        }
        aclshmem_barrier_all();
        aclrtRecordEvent(begin_event, stream);
        const int blocks = static_cast<uint32_t>(pe) == workers
            ? static_cast<int>(inc_lanes)
            : static_cast<int>(worker_lanes);
        launch_inc_dc_single_inc_inline_dispatch_kernel(
            symmetric, blocks, stream);
        aclrtRecordEvent(end_event, stream);
        status = aclrtSynchronizeStream(stream);
        float elapsed_ms = 0.0f;
        if (status == 0)
            status = aclrtEventElapsedTime(
                &elapsed_ms, begin_event, end_event);
        if (status == 0)
            status = CopyD2H(
                reinterpret_cast<uint8_t *>(stats.data()),
                symmetric + desc.stats_off,
                stats.size() * sizeof(InlineDispatchBenchStatV2));
        bool epoch_pass = status == 0;
        const uint32_t active = static_cast<uint32_t>(pe) == workers
            ? inc_lanes : worker_lanes;
        const size_t stat_base =
            static_cast<size_t>(pe) * kInlineDispatchMaxLanes;
        for (uint32_t lane = 0u; lane < active; ++lane) {
            epoch_pass = epoch_pass &&
                stats[stat_base + lane].error == kInlineDispatchOk &&
                stats[stat_base + lane].done_generation == desc.generation;
            if (stats[stat_base + lane].error != kInlineDispatchOk ||
                stats[stat_base + lane].done_generation != desc.generation) {
                std::cerr << "INLINE_V2_LANE_ERROR pe=" << pe
                          << " lane=" << lane
                          << " error=" << stats[stat_base + lane].error
                          << " done_generation="
                          << stats[stat_base + lane].done_generation
                          << " expected_generation=" << desc.generation
                          << " batches=" << stats[stat_base + lane].batches
                          << " tokens="
                          << stats[stat_base + lane].token_records
                          << " assignments="
                          << stats[stat_base + lane].assignments
                          << " detail="
                          << stats[stat_base + lane].reserved32 << '\n';
            }
        }
        aclshmem_barrier_all();
        uint64_t verified_records = 0u;
        uint64_t verified_assignments = 0u;
        uint64_t verified_payload_bytes = 0u;
        if (epoch_pass && epoch + 1u == warmup + measure) {
            epoch_pass = VerifyFanout(
                symmetric, desc, desc.generation, &verified_records,
                &verified_assignments, &verified_payload_bytes);
        }
        all_pass = all_pass && epoch_pass;
        if (epoch >= warmup) {
            const uint64_t fanout_payload_bytes =
                static_cast<uint64_t>(workers) * tokens * topk * hidden_bytes;
            const uint64_t upload_payload_bytes =
                static_cast<uint64_t>(workers) * tokens * hidden_bytes;
            std::cout << "INLINE_V2_DISPATCH_TIMING pe=" << pe
                      << " sample=" << (epoch - warmup)
                      << " rank_us=" << elapsed_ms * 1000.0f
                      << " upload_payload_bytes=" << upload_payload_bytes
                      << " fanout_payload_bytes=" << fanout_payload_bytes
                      << " verified_records=" << verified_records
                      << " pass=" << (epoch_pass ? 1 : 0) << '\n';
        }
        if (!epoch_pass) break;
    }

    std::cout << "INLINE_V2_DISPATCH_RESULT pe=" << pe
              << " pass=" << (all_pass ? 1 : 0)
              << " generations=" << warmup + measure << '\n';
    if (begin_event != nullptr) aclrtDestroyEvent(begin_event);
    if (end_event != nullptr) aclrtDestroyEvent(end_event);
    Cleanup(device, stream, symmetric);
    return all_pass ? 0 : 20;
}
