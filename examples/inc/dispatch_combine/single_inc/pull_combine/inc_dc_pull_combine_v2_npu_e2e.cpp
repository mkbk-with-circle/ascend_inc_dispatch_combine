#include "inc_dc_pull_combine_v2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "shmem.h"
#include "utils.h"

#include "inc_dc_pull_v2_test_start_gate.h"

using namespace inc::dc::pull_v2;

extern "C" void launch_inc_dc_pull_combine_v2_device(
    uint32_t block_dim, void *stream, uint8_t *symmetric_partials,
    uint8_t *ready_records, uint8_t *ready_notices,
    uint8_t *ready_staging, uint8_t *registrations,
    uint8_t *source_acks, uint8_t *owner_output,
    uint8_t *owner_completions, uint8_t *source_offsets, uint8_t *pulls,
    uint8_t *owner_offsets, uint8_t *results, uint8_t *journal_header,
    uint8_t *pull_next, uint8_t *accumulator_heads,
    uint8_t *accumulator_contributor_counts,
    uint8_t *accumulator_result_index, uint8_t *source_ready_state,
    uint8_t *source_payload_offsets, uint8_t *status_line,
    uint64_t ffts_addr, uint64_t session_id, uint64_t placement_epoch,
    uint64_t generation, uint64_t sequence, uint64_t dispatch_cookie,
    uint64_t owner_output_slot_stride, uint64_t pull_count,
    uint64_t result_count, uint64_t pull_index_capacity,
    uint64_t accumulator_index_capacity,
    uint64_t source_scratch_capacity, uint32_t accumulator_count,
    uint32_t worker_count, uint32_t hidden, uint32_t partial_dtype,
    int32_t inc_pe, uint32_t wave, uint32_t ring_slot,
    uint32_t slot_count, uint64_t spin_cap);

int g_npus = 5;
const char *ipport = "tcp://127.0.0.1:28842";
int f_pe = 0;
int f_npu = 0;
aclshmemx_uniqueid_t default_flag_uid;

namespace {

constexpr uint64_t kSessionId = 0x434f4d4232563201ull;
constexpr uint64_t kPlacementEpoch = 19u;
constexpr uint64_t kFirstGeneration = 5001u;
constexpr uint64_t kFirstSequence = 9001u;
constexpr uint32_t kFirstWave = 31u;
constexpr uint32_t kRingSlots = 2u;
constexpr uint64_t kSpinCap = 50000000ull; // 1 s at the 50 MHz counter.
constexpr uint64_t kGuardBytes = 64u;
constexpr uint64_t kSourceScratchStride = 64u;
constexpr uint8_t kHeadGuard = 0xa5u;
constexpr uint8_t kTailGuard = 0x5au;
constexpr uint8_t kPoison = 0xc7u;
constexpr uint64_t kHashOffset = 1469598103934665603ull;

// Mirrored from the device TU until these wire records move into public ABI.
struct alignas(64) CombineSourceAckV2 {
    uint32_t magic = kPullCombineV2Magic;
    uint16_t abi_version = kPullCombineV2AbiVersion;
    uint16_t struct_bytes = sizeof(CombineSourceAckV2);
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint64_t dispatch_cookie = 0u;
    uint32_t wave = 0u;
    uint32_t source_rank = 0u;
    uint32_t source_region_id = 0u;
    uint32_t status = 0u;
    uint16_t ring_slot = 0u;
    uint16_t flags = 0u;
    uint32_t row_count = 0u;
    uint64_t bytes_consumed = 0u;
    uint64_t reserved[5]{};
    uint64_t publication = 0u;
};
static_assert(sizeof(CombineSourceAckV2) == 128u);
static_assert(offsetof(CombineSourceAckV2, publication) + sizeof(uint64_t) ==
              sizeof(CombineSourceAckV2));

struct alignas(64) CombineOwnerCompletionV2 {
    uint32_t magic = kPullCombineV2Magic;
    uint16_t abi_version = kPullCombineV2AbiVersion;
    uint16_t struct_bytes = sizeof(CombineOwnerCompletionV2);
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint64_t dispatch_cookie = 0u;
    uint32_t wave = 0u;
    uint32_t owner_rank = 0u;
    uint32_t status = 0u;
    uint32_t row_count = 0u;
    uint16_t ring_slot = 0u;
    uint16_t flags = 0u;
    uint32_t reserved0 = 0u;
    uint64_t bytes_produced = 0u;
    uint64_t reserved[5]{};
    uint64_t publication = 0u;
};
static_assert(sizeof(CombineOwnerCompletionV2) == 128u);
static_assert(offsetof(CombineOwnerCompletionV2, publication) +
                  sizeof(uint64_t) == sizeof(CombineOwnerCompletionV2));

struct alignas(64) CombineDeviceTimelineV2 {
    uint32_t status = 0u;
    uint32_t ready_sources = 0u;
    uint64_t kernel_start = 0u;
    uint64_t plan_index_done = 0u;
    uint64_t first_ready = 0u;
    uint64_t all_ready = 0u;
    uint64_t first_get = 0u;
    uint64_t last_get = 0u;
    uint64_t last_reduce = 0u;
    uint64_t last_owner_put = 0u;
    uint64_t source_acks_done = 0u;
    uint64_t owner_completions_done = 0u;
    uint64_t kernel_done = 0u;
    uint64_t reserved[4]{};
};
static_assert(sizeof(CombineDeviceTimelineV2) == 128u);

enum class Workload {
    SYM_TOPK_ALL,
    SYM_K2_BALANCED,
    ASYMMETRIC,
    READY_SKEW
};

struct Options {
    uint32_t workers = 0u;
    int pe = -1;
    int first_npu = 0;
    uint32_t hidden = 0u;
    uint32_t rows = 0u;
    Workload workload = Workload::SYM_K2_BALANCED;
    const char *workload_name = nullptr;
    uint32_t warmup = 0u;
    uint32_t measure = 0u;
};

struct GuardedBuffer {
    uint8_t *allocation = nullptr;
    uint8_t *data = nullptr;
    uint64_t requested_bytes = 0u;
    uint64_t payload_bytes = 0u;
    bool symmetric = false;
    const char *name = nullptr;
};

struct Wave {
    CompiledLayout layout;
    CombinePullPlan plan;
    std::vector<CombineRegionRegistration> registrations;
    std::vector<CombineReadyV2> ready;
    std::vector<CombineReadyNoticeV2> notices;
    uint64_t partial_slot_stride = 0u;
    uint64_t output_slot_stride = 0u;
    uint64_t ingress_bytes = 0u;
    uint64_t egress_bytes = 0u;
};

uint64_t Align64(uint64_t value)
{
    if (value > std::numeric_limits<uint64_t>::max() - 63u) return 0u;
    return (value + 63u) & ~63ull;
}

bool Add(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr || b > std::numeric_limits<uint64_t>::max() - a)
        return false;
    *out = a + b;
    return true;
}

bool Mul(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr || (a != 0u &&
        b > std::numeric_limits<uint64_t>::max() / a)) return false;
    *out = a * b;
    return true;
}

uint64_t Mix(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

uint64_t Publication(const void *record, size_t bytes)
{
    const auto *data = static_cast<const uint8_t *>(record);
    uint64_t hash = kHashOffset;
    for (size_t i = 0u; i < bytes; ++i) {
        const uint64_t x = hash ^ data[i];
        hash = (x << 40u) + (x << 8u) + (x << 7u) + (x << 5u) +
            (x << 4u) + (x << 1u) + x;
    }
    return hash == 0u ? 1u : hash;
}

float PartialValue(uint32_t source, uint32_t row, uint32_t element)
{
    const int32_t value = static_cast<int32_t>(
        (source * 131u + row * 17u + element * 7u) % 1021u) - 510;
    return static_cast<float>(value) / 64.0f;
}

bool ParseWorkload(const char *text, Workload *workload)
{
    if (std::strcmp(text, "sym_topk_all") == 0 ||
        std::strcmp(text, "symmetric") == 0)
        *workload = Workload::SYM_TOPK_ALL;
    else if (std::strcmp(text, "sym_k2_balanced") == 0)
        *workload = Workload::SYM_K2_BALANCED;
    else if (std::strcmp(text, "asymmetric") == 0)
        *workload = Workload::ASYMMETRIC;
    else if (std::strcmp(text, "ready_skew") == 0)
        *workload = Workload::READY_SKEW;
    else
        return false;
    return true;
}

int Fail(const char *step, int status)
{
    std::cerr << "[FAIL] " << step << " status=" << status << '\n';
    return status == 0 ? 1 : status;
}

bool Allocate(GuardedBuffer *buffer, uint64_t bytes, bool symmetric,
              const char *name)
{
    if (buffer == nullptr) return false;
    const uint64_t payload = std::max<uint64_t>(bytes, 64u);
    uint64_t total = 0u;
    if (!Add(payload, 2u * kGuardBytes, &total)) return false;
    uint8_t *base = nullptr;
    if (symmetric) {
        base = static_cast<uint8_t *>(aclshmem_malloc(total));
    } else {
        void *memory = nullptr;
        if (aclrtMalloc(&memory, total, ACL_MEM_MALLOC_HUGE_FIRST) ==
            ACL_SUCCESS) base = static_cast<uint8_t *>(memory);
    }
    if (base == nullptr) return false;
    buffer->allocation = base;
    buffer->data = base + kGuardBytes;
    buffer->requested_bytes = bytes;
    buffer->payload_bytes = payload;
    buffer->symmetric = symmetric;
    buffer->name = name;
    std::vector<uint8_t> head(kGuardBytes, kHeadGuard);
    std::vector<uint8_t> tail(kGuardBytes, kTailGuard);
    return aclrtMemcpy(base, kGuardBytes, head.data(), kGuardBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
        aclrtMemset(buffer->data, payload, kPoison, payload) == ACL_SUCCESS &&
        aclrtMemcpy(buffer->data + payload, kGuardBytes, tail.data(),
                    kGuardBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

void Release(GuardedBuffer *buffer)
{
    if (buffer == nullptr || buffer->allocation == nullptr) return;
    if (buffer->symmetric) aclshmem_free(buffer->allocation);
    else aclrtFree(buffer->allocation);
    *buffer = GuardedBuffer{};
}

bool Fill(GuardedBuffer *buffer, uint8_t value)
{
    return buffer != nullptr && aclrtMemset(
        buffer->data, buffer->payload_bytes, value,
        buffer->payload_bytes) == ACL_SUCCESS;
}

bool GuardsValid(const GuardedBuffer &buffer)
{
    std::vector<uint8_t> head(kGuardBytes), tail(kGuardBytes);
    if (aclrtMemcpy(head.data(), head.size(), buffer.allocation, head.size(),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(tail.data(), tail.size(),
                    buffer.data + buffer.payload_bytes, tail.size(),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
        return false;
    const bool valid = std::all_of(head.begin(), head.end(),
            [](uint8_t x) { return x == kHeadGuard; }) &&
        std::all_of(tail.begin(), tail.end(),
            [](uint8_t x) { return x == kTailGuard; });
    if (!valid) std::cerr << "[FAIL] guard changed: " << buffer.name << '\n';
    return valid;
}

template <typename T>
bool CopyToDevice(uint8_t *destination, const std::vector<T> &source)
{
    return source.empty() || aclrtMemcpy(
        destination, source.size() * sizeof(T), source.data(),
        source.size() * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

ParsedSource MakeSource(const Options &o, uint32_t owner,
                        uint64_t generation, uint64_t sequence,
                        uint32_t wave, uint16_t ring_slot)
{
    ParsedSource source{};
    source.header.session_id = kSessionId;
    source.header.placement_epoch = kPlacementEpoch;
    source.header.generation = generation;
    source.header.sequence = sequence;
    source.header.wave = wave;
    source.header.source_rank = owner;
    source.header.source_region_id = 100u + owner;
    source.header.ring_slot = ring_slot;
    source.header.worker_count = o.workers;
    source.header.hidden = o.hidden;
    source.header.dtype = static_cast<uint32_t>(DataType::BF16);
    source.header.metadata_digest = Mix(generation ^ (sequence << 1u) ^
        (static_cast<uint64_t>(owner) << 48u) ^ o.rows);

    const uint32_t owner_rows = o.rows / o.workers +
        (owner < o.rows % o.workers ? 1u : 0u);
    source.tokens.reserve(owner_rows);
    for (uint32_t row = 0u; row < owner_rows; ++row) {
        TokenRecord token{};
        token.token_id = (static_cast<uint64_t>(wave) << 32u) | row;
        token.source_token = row;
        token.assignment_begin = source.assignments.size();
        const uint64_t global_row = static_cast<uint64_t>(owner) +
            static_cast<uint64_t>(row) * o.workers;
        for (uint32_t destination = 0u; destination < o.workers;
             ++destination) {
            bool selected = true;
            if (o.workload == Workload::SYM_K2_BALANCED) {
                selected = destination == owner ||
                    destination == (owner + 1u) % o.workers;
            } else if (o.workload == Workload::ASYMMETRIC &&
                       destination != 0u) {
                const uint32_t shift = std::min<uint32_t>(destination, 30u);
                selected = (global_row & ((1u << shift) - 1u)) == 0u;
            }
            if (!selected) continue;
            for (uint32_t local = 0u; local < 2u; ++local) {
                source.assignments.push_back(AssignmentRecord{
                    destination, (destination * 2u + local) % 16u,
                    destination * 2u + local,
                    local == 0u ? 0.75f : 0.25f});
            }
        }
        token.assignment_count = source.assignments.size() -
            token.assignment_begin;
        source.tokens.push_back(token);
    }
    source.header.token_count = source.tokens.size();
    source.header.assignment_count = source.assignments.size();
    return source;
}

bool BuildWave(const Options &o, uint64_t generation, uint64_t sequence,
               uint32_t wave, uint16_t ring_slot, Wave *wave_data)
{
    if (wave_data == nullptr) return false;
    std::vector<ParsedSource> sources;
    uint64_t assignments = 0u;
    for (uint32_t source = 0u; source < o.workers; ++source) {
        sources.push_back(MakeSource(o, source, generation, sequence, wave,
                                     ring_slot));
        assignments += sources.back().assignments.size();
    }
    std::vector<uint32_t> arrival(o.workers);
    for (uint32_t rank = 0u; rank < o.workers; ++rank)
        arrival[rank] = o.workers - rank - 1u;
    LayoutConfig layout_config{};
    layout_config.worker_count = o.workers;
    layout_config.expert_count = 16u;
    layout_config.destination_row_capacity = std::max<uint64_t>(o.rows, 1u);
    layout_config.destination_assignment_capacity =
        std::max<uint64_t>(assignments, 1u);
    Wave built{};
    std::string error;
    if (CompileLayout(sources, arrival, layout_config, &built.layout,
                      &error) != Status::OK) {
        std::cerr << "CompileLayout: " << error << '\n';
        return false;
    }
    CombineV2Config config{};
    config.session_id = kSessionId;
    config.placement_epoch = kPlacementEpoch;
    config.worker_count = o.workers;
    config.hidden = o.hidden;
    config.partial_dtype = PartialDataType::FP32;
    config.ring_slots = kRingSlots;
    if (CompileCombinePullPlan(built.layout, config, &built.plan, &error) !=
        CombineV2Status::OK) {
        std::cerr << "CompileCombinePullPlan: " << error << '\n';
        return false;
    }
    if (built.plan.generation != generation ||
        built.plan.sequence != sequence || built.plan.wave != wave ||
        built.plan.ring_slot != ring_slot ||
        built.plan.dispatch_cookie != built.layout.journal_header.dispatch_cookie ||
        built.plan.accumulator_count != built.layout.journal_tokens.size() ||
        built.plan.pull_next.size() != built.plan.pulls.size() ||
        built.plan.accumulator_heads.size() !=
            built.plan.accumulator_count ||
        built.plan.accumulator_contributor_counts.size() !=
            built.plan.accumulator_count ||
        built.plan.accumulator_result_index.size() !=
            built.plan.accumulator_count) {
        std::cerr << "strong wave identity mismatch\n";
        return false;
    }

    const uint64_t row_bytes = static_cast<uint64_t>(o.hidden) * sizeof(float);
    uint32_t max_source_rows = 0u;
    for (uint32_t rows : built.plan.source_row_counts)
        max_source_rows = std::max(max_source_rows, rows);
    uint64_t max_owner_rows = 0u;
    for (uint32_t owner = 0u; owner < o.workers; ++owner)
        max_owner_rows = std::max(max_owner_rows,
            built.plan.owner_offsets[owner + 1u] -
                built.plan.owner_offsets[owner]);
    built.partial_slot_stride = Align64(std::max<uint64_t>(
        64u, static_cast<uint64_t>(max_source_rows) * row_bytes));
    built.output_slot_stride = Align64(std::max<uint64_t>(
        64u, max_owner_rows * row_bytes));
    if (built.partial_slot_stride == 0u || built.output_slot_stride == 0u)
        return false;

    built.registrations.resize(o.workers);
    built.ready.resize(o.workers);
    built.notices.resize(o.workers);
    for (uint32_t source = 0u; source < o.workers; ++source) {
        auto &registration = built.registrations[source];
        registration.session_id = kSessionId;
        registration.placement_epoch = kPlacementEpoch;
        registration.source_rank = source;
        registration.region_id = 700u + source;
        registration.slot_count = kRingSlots;
        registration.slot_stride = built.partial_slot_stride;
        registration.region_bytes = built.partial_slot_stride * kRingSlots;

        auto &ready = built.ready[source];
        ready.session_id = kSessionId;
        ready.placement_epoch = kPlacementEpoch;
        ready.generation = generation;
        ready.sequence = sequence;
        ready.dispatch_cookie = built.plan.dispatch_cookie;
        ready.wave = wave;
        ready.source_rank = source;
        ready.source_region_id = registration.region_id;
        ready.ring_slot = ring_slot;
        ready.row_count = built.plan.source_row_counts[source];
        ready.hidden = o.hidden;
        ready.partial_dtype = static_cast<uint32_t>(PartialDataType::FP32);
        ready.source_offset = static_cast<uint64_t>(ring_slot) *
            built.partial_slot_stride;
        ready.payload_bytes = static_cast<uint64_t>(ready.row_count) *
            row_bytes;
        ready.publication = CombineReadyPublication(ready);
        if (ValidateCombineReady(ready, registration, config, built.plan,
                                 &error) != CombineV2Status::OK) {
            std::cerr << "ValidateCombineReady: " << error << '\n';
            return false;
        }
        auto &notice = built.notices[source];
        notice.session_id = kSessionId;
        notice.placement_epoch = kPlacementEpoch;
        notice.generation = generation;
        notice.sequence = sequence;
        notice.wave = wave;
        notice.source_rank = source;
        notice.ring_slot = ring_slot;
        notice.publication = CombineReadyNoticePublication(notice);
        built.ingress_bytes += ready.payload_bytes;
    }
    built.egress_bytes = static_cast<uint64_t>(built.plan.accumulator_count) *
        row_bytes;
    *wave_data = std::move(built);
    return true;
}

bool ValidateWorker(const Options &o, const Wave &wave_data,
                    uint16_t ring_slot, const GuardedBuffer &acks,
                    const GuardedBuffer &output,
                    const GuardedBuffer &completions)
{
    const uint32_t rank = static_cast<uint32_t>(o.pe);
    CombineSourceAckV2 ack{};
    CombineOwnerCompletionV2 completion{};
    const uint64_t index = static_cast<uint64_t>(ring_slot) * o.workers + rank;
    if (aclrtMemcpy(&ack, sizeof(ack),
                    acks.data + index * sizeof(ack), sizeof(ack),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(&completion, sizeof(completion),
                    completions.data + index * sizeof(completion),
                    sizeof(completion), ACL_MEMCPY_DEVICE_TO_HOST) !=
            ACL_SUCCESS)
        return false;
    const uint32_t source_rows = wave_data.plan.source_row_counts[rank];
    const uint32_t owner_rows = static_cast<uint32_t>(
        wave_data.plan.owner_offsets[rank + 1u] -
        wave_data.plan.owner_offsets[rank]);
    const uint64_t source_bytes = static_cast<uint64_t>(source_rows) *
        o.hidden * sizeof(float);
    const uint64_t owner_bytes = static_cast<uint64_t>(owner_rows) *
        o.hidden * sizeof(float);
    const bool ack_ok = ack.magic == kPullCombineV2Magic &&
        ack.abi_version == kPullCombineV2AbiVersion &&
        ack.struct_bytes == sizeof(ack) && ack.session_id == kSessionId &&
        ack.placement_epoch == kPlacementEpoch &&
        ack.generation == wave_data.plan.generation &&
        ack.sequence == wave_data.plan.sequence &&
        ack.dispatch_cookie == wave_data.plan.dispatch_cookie &&
        ack.wave == wave_data.plan.wave && ack.source_rank == rank &&
        ack.source_region_id == wave_data.registrations[rank].region_id &&
        ack.status == 0u && ack.ring_slot == ring_slot && ack.flags == 0u &&
        ack.row_count == source_rows && ack.bytes_consumed == source_bytes &&
        std::all_of(std::begin(ack.reserved), std::end(ack.reserved),
                    [](uint64_t x) { return x == 0u; }) &&
        ack.publication == Publication(&ack,
            offsetof(CombineSourceAckV2, publication));
    const bool completion_ok = completion.magic == kPullCombineV2Magic &&
        completion.abi_version == kPullCombineV2AbiVersion &&
        completion.struct_bytes == sizeof(completion) &&
        completion.session_id == kSessionId &&
        completion.placement_epoch == kPlacementEpoch &&
        completion.generation == wave_data.plan.generation &&
        completion.sequence == wave_data.plan.sequence &&
        completion.dispatch_cookie == wave_data.plan.dispatch_cookie &&
        completion.wave == wave_data.plan.wave &&
        completion.owner_rank == rank && completion.status == 0u &&
        completion.row_count == owner_rows &&
        completion.ring_slot == ring_slot && completion.flags == 0u &&
        completion.reserved0 == 0u &&
        completion.bytes_produced == owner_bytes &&
        std::all_of(std::begin(completion.reserved),
                    std::end(completion.reserved),
                    [](uint64_t x) { return x == 0u; }) &&
        completion.publication == Publication(&completion,
            offsetof(CombineOwnerCompletionV2, publication));
    if (!ack_ok || !completion_ok) {
        std::cerr << "[FAIL] PE" << rank << " ACK/completion identity\n";
        return false;
    }

    std::vector<float> actual(static_cast<size_t>(owner_rows) * o.hidden);
    if (!actual.empty() && aclrtMemcpy(actual.data(), owner_bytes,
            output.data + static_cast<uint64_t>(ring_slot) *
                wave_data.output_slot_stride,
            owner_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
        return false;
    uint64_t mismatches = 0u;
    for (uint32_t owner_row = 0u; owner_row < owner_rows; ++owner_row) {
        const auto &result = wave_data.plan.results[
            wave_data.plan.owner_offsets[rank] + owner_row];
        const auto &token = wave_data.layout.journal_tokens[
            result.journal_token];
        if (token.owner_rank != rank || token.owner_row != owner_row ||
            token.route_key != RouteKey(rank, owner_row)) return false;
        for (uint32_t element = 0u; element < o.hidden; ++element) {
            float expected = 0.0f;
            for (uint32_t local = 0u; local < token.contributors_count;
                 ++local) {
                const auto &contributor = wave_data.layout.contributors[
                    token.contributors_begin + local];
                expected += PartialValue(contributor.worker_rank,
                                         contributor.destination_row,
                                         element);
            }
            const float value = actual[
                static_cast<size_t>(owner_row) * o.hidden + element];
            if (value != expected) {
                if (mismatches < 8u)
                    std::cerr << "[FAIL] PE" << rank << " row=" << owner_row
                              << " element=" << element << " actual="
                              << value << " expected=" << expected << '\n';
                ++mismatches;
            }
        }
    }
    return mismatches == 0u;
}

bool ValidateInc(const Wave &wave_data, const GuardedBuffer &journal,
                 const GuardedBuffer &ready_staging,
                 const GuardedBuffer &ready_notices,
                 const GuardedBuffer &ready_state,
                 const GuardedBuffer &timeline_buffer,
                 CombineDeviceTimelineV2 *timeline)
{
    JournalSlotHeader header{};
    if (aclrtMemcpy(&header, sizeof(header), journal.data, sizeof(header),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(timeline, sizeof(*timeline), timeline_buffer.data,
                    sizeof(*timeline), ACL_MEMCPY_DEVICE_TO_HOST) !=
            ACL_SUCCESS)
        return false;
    const bool identity = header.magic == kPullDispatchMagic &&
        header.abi_version == kPullDispatchAbiVersion &&
        header.struct_bytes == sizeof(header) &&
        header.generation == wave_data.plan.generation &&
        header.sequence == wave_data.plan.sequence &&
        header.dispatch_cookie == wave_data.plan.dispatch_cookie &&
        header.wave == wave_data.plan.wave &&
        header.ring_slot == wave_data.plan.ring_slot &&
        header.state == static_cast<uint16_t>(JournalSlotState::COMPLETE) &&
        header.token_count == wave_data.plan.accumulator_count &&
        header.contributor_count == wave_data.plan.pulls.size() &&
        header.status == 0u && timeline->status == 0u &&
        timeline->ready_sources == wave_data.plan.worker_count &&
        timeline->kernel_start != 0u &&
        timeline->plan_index_done >= timeline->kernel_start &&
        timeline->first_ready >= timeline->plan_index_done &&
        timeline->all_ready >= timeline->first_ready &&
        timeline->kernel_done >= timeline->plan_index_done &&
        timeline->source_acks_done <= timeline->kernel_done &&
        timeline->owner_completions_done == timeline->kernel_done &&
        std::all_of(std::begin(timeline->reserved),
                    std::end(timeline->reserved),
                    [](uint64_t x) { return x == 0u; });
    if (!identity) {
        std::cerr << "[FAIL] INC journal/timeline identity status="
                  << timeline->status << " ready_sources="
                  << timeline->ready_sources << " journal_state="
                  << header.state << " journal_status=" << header.status
                  << '\n';
        for (uint32_t source = 0u; source < wave_data.plan.worker_count;
             ++source) {
            uint32_t state = 0xffffffffu;
            (void)aclrtMemcpy(&state, sizeof(state),
                              ready_state.data +
                                  static_cast<uint64_t>(source) *
                                      kSourceScratchStride,
                              sizeof(state), ACL_MEMCPY_DEVICE_TO_HOST);
            CombineReadyV2 descriptor{};
            const uint64_t notice_index =
                static_cast<uint64_t>(wave_data.plan.ring_slot) *
                    wave_data.plan.worker_count + source;
            if (aclrtMemcpy(&descriptor, sizeof(descriptor),
                            ready_staging.data +
                                static_cast<uint64_t>(source) *
                                    sizeof(descriptor),
                            sizeof(descriptor),
                            ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) {
                CombineReadyNoticeV2 notice{};
                (void)aclrtMemcpy(
                    &notice, sizeof(notice),
                    ready_notices.data +
                        notice_index * sizeof(CombineReadyNoticeV2),
                    sizeof(notice), ACL_MEMCPY_DEVICE_TO_HOST);
                std::cerr << "[DEBUG] ready source=" << source
                          << " state=" << state
                          << " notice_publication=" << notice.publication
                          << " publication=" << descriptor.publication
                          << " generation=" << descriptor.generation
                          << " sequence=" << descriptor.sequence
                          << " wave=" << descriptor.wave
                          << " ring=" << descriptor.ring_slot
                          << " session=" << descriptor.session_id
                          << " placement=" << descriptor.placement_epoch
                          << " cookie=" << descriptor.dispatch_cookie
                          << " rows=" << descriptor.row_count << '\n';
            }
        }
    }
    return identity;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 10) {
        std::cerr << "usage: " << argv[0]
                  << " <workers:2|4> <pe> <ipport> <first_npu> <hidden>"
                     " <rows> <sym_k2_balanced|sym_topk_all|asymmetric|"
                     "ready_skew>"
                     " <warmup> <measure>\n"
                  << "rows is the total A-token count. sym_k2_balanced is "
                     "the W2/W4 gate workload; sym_topk_all (legacy alias "
                     "symmetric) is the topk=W expansion stress case. "
                     "rows=0 is valid; hidden=2049 tests the 2048 tail.\n";
        return 2;
    }
    Options o{};
    o.workers = static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10));
    o.pe = std::atoi(argv[2]);
    ipport = argv[3];
    o.first_npu = std::atoi(argv[4]);
    f_npu = o.first_npu;
    o.hidden = static_cast<uint32_t>(std::strtoul(argv[5], nullptr, 10));
    o.rows = static_cast<uint32_t>(std::strtoul(argv[6], nullptr, 10));
    o.workload_name = argv[7];
    o.warmup = static_cast<uint32_t>(std::strtoul(argv[8], nullptr, 10));
    o.measure = static_cast<uint32_t>(std::strtoul(argv[9], nullptr, 10));
    const uint32_t pes = o.workers + 1u;
    const int inc_pe = static_cast<int>(o.workers);
    g_npus = static_cast<int>(pes);
    if ((o.workers != 2u && o.workers != 4u) || o.pe < 0 ||
        o.pe >= static_cast<int>(pes) || o.first_npu < 0 ||
        o.hidden == 0u || o.measure == 0u ||
        !ParseWorkload(o.workload_name, &o.workload))
        return Fail("arguments", 2);
    uint64_t row_bytes = 0u;
    if (!Mul(o.hidden, sizeof(float), &row_bytes) ||
        static_cast<uint64_t>(o.rows) * row_bytes >
            std::numeric_limits<size_t>::max())
        return Fail("shape overflow", 2);

    Wave sizing{};
    if (!BuildWave(o, kFirstGeneration, kFirstSequence, kFirstWave, 0u,
                   &sizing))
        return Fail("sizing wave", 2);

    const int device = o.pe + o.first_npu;
    aclrtStream stream = nullptr;
    bool shmem_initialized = false;
    int status = aclInit(nullptr);
    if (status == 0) status = aclrtSetDevice(device);
    int64_t live_aiv = 0;
    uint32_t combine_aiv = 0u;
    if (status == 0)
        status = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                    &live_aiv);
    if (status == 0) {
        combine_aiv = live_aiv <= 0 ? 0u :
            static_cast<uint32_t>(live_aiv) / 2u;
        if (combine_aiv == 0u) status = 2;
    }
    if (status == 0) status = aclrtCreateStream(&stream);
    if (status == 0) {
        aclshmemx_init_attr_t attr;
        test_set_attr(o.pe, pes, 2ull * 1024ull * 1024ull * 1024ull,
                      ipport, default_flag_uid, &attr);
        status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
        shmem_initialized = status == 0;
    }
    if (status == 0)
        std::cerr << "[STAGE] pe=" << o.pe << " initialized\n" << std::flush;

    GuardedBuffer partials, ready, notices, ready_staging, registrations;
    GuardedBuffer acks, output, completions;
    GuardedBuffer source_offsets, pulls, owner_offsets, results, journal;
    GuardedBuffer pull_next, heads, counts, result_index, ready_state;
    GuardedBuffer payload_offsets, timeline_buffer;
    std::vector<GuardedBuffer *> buffers{&partials, &ready, &notices,
        &ready_staging, &registrations, &acks, &output, &completions,
        &source_offsets, &pulls, &owner_offsets, &results, &journal,
        &pull_next, &heads, &counts, &result_index, &ready_state,
        &payload_offsets, &timeline_buffer};
    auto Alloc = [&](GuardedBuffer *buffer, uint64_t bytes, bool symmetric,
                     const char *name) {
        if (status == 0 && !Allocate(buffer, bytes, symmetric, name))
            status = 1;
    };
    const uint64_t pull_capacity = std::max<uint64_t>(
        sizing.plan.pulls.size(), 1u);
    const uint64_t accumulator_capacity = std::max<uint64_t>(
        sizing.plan.accumulator_count, 1u);
    const uint64_t source_capacity = std::max<uint64_t>(o.workers, 1u);
    if (status == 0) {
        Alloc(&partials, sizing.partial_slot_stride * kRingSlots, true,
              "partials");
        Alloc(&ready, static_cast<uint64_t>(kRingSlots) * o.workers *
            sizeof(CombineReadyV2), true, "ready");
        Alloc(&notices, static_cast<uint64_t>(kRingSlots) * o.workers *
            sizeof(CombineReadyNoticeV2), true, "ready_notices");
        Alloc(&ready_staging, static_cast<uint64_t>(o.workers) *
            sizeof(CombineReadyV2), false, "ready_staging");
        Alloc(&registrations, static_cast<uint64_t>(o.workers) *
            sizeof(CombineRegionRegistration), true, "registrations");
        Alloc(&acks, static_cast<uint64_t>(kRingSlots) * o.workers *
            sizeof(CombineSourceAckV2), true, "acks");
        Alloc(&output, sizing.output_slot_stride * kRingSlots, true,
              "output");
        Alloc(&completions, static_cast<uint64_t>(kRingSlots) * o.workers *
            sizeof(CombineOwnerCompletionV2), true, "completions");
        Alloc(&source_offsets, (o.workers + 1u) * sizeof(uint64_t), false,
              "source_offsets");
        Alloc(&pulls, pull_capacity * sizeof(CombinePullOp), false, "pulls");
        Alloc(&owner_offsets, (o.workers + 1u) * sizeof(uint64_t), false,
              "owner_offsets");
        Alloc(&results, accumulator_capacity * sizeof(CombineResultOp),
              false, "results");
        Alloc(&journal, sizeof(JournalSlotHeader), false, "journal");
        Alloc(&pull_next, pull_capacity * sizeof(uint32_t), false,
              "pull_next");
        Alloc(&heads, accumulator_capacity * sizeof(uint32_t), false,
              "heads");
        Alloc(&counts, accumulator_capacity * sizeof(uint32_t), false,
              "counts");
        Alloc(&result_index, accumulator_capacity * sizeof(uint32_t), false,
              "result_index");
        Alloc(&ready_state, source_capacity * kSourceScratchStride, false,
              "ready_state");
        Alloc(&payload_offsets, source_capacity * kSourceScratchStride, false,
              "payload_offsets");
        Alloc(&timeline_buffer, sizeof(CombineDeviceTimelineV2), false,
              "timeline");
    }

    bool correct = status == 0;
    std::vector<double> measured_us, measured_gbps;
    const uint32_t iterations = o.warmup + o.measure;
    for (uint32_t iteration = 0u; iteration < iterations && correct;
         ++iteration) {
        const uint64_t generation = kFirstGeneration + iteration;
        const uint64_t sequence = kFirstSequence + iteration;
        const uint32_t wave_id = kFirstWave + iteration;
        const uint16_t ring_slot = iteration % kRingSlots;
        Wave wave_data{};
        correct = BuildWave(o, generation, sequence, wave_id, ring_slot,
                            &wave_data) &&
            wave_data.partial_slot_stride == sizing.partial_slot_stride &&
            wave_data.output_slot_stride == sizing.output_slot_stride &&
            wave_data.plan.pulls.size() <= pull_capacity &&
            wave_data.plan.accumulator_count <= accumulator_capacity;
        if (!correct) break;

        if (o.pe < inc_pe) {
            const uint32_t source = static_cast<uint32_t>(o.pe);
            const uint32_t rows = wave_data.plan.source_row_counts[source];
            std::vector<float> host(static_cast<size_t>(rows) * o.hidden);
            for (uint32_t row = 0u; row < rows; ++row)
                for (uint32_t element = 0u; element < o.hidden; ++element)
                    host[static_cast<size_t>(row) * o.hidden + element] =
                        PartialValue(source, row, element);
            const uint64_t offset = static_cast<uint64_t>(ring_slot) *
                sizing.partial_slot_stride;
            if (!host.empty())
                status = aclrtMemcpy(partials.data + offset,
                    host.size() * sizeof(float), host.data(),
                    host.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
            if (status == 0)
                status = aclrtMemcpy(ready.data +
                        (static_cast<uint64_t>(ring_slot) * o.workers +
                         source) * sizeof(CombineReadyV2),
                    sizeof(CombineReadyV2), &wave_data.ready[source],
                    sizeof(CombineReadyV2), ACL_MEMCPY_HOST_TO_DEVICE);
            if (status == 0)
                status = aclrtMemcpy(notices.data +
                        (static_cast<uint64_t>(ring_slot) * o.workers +
                         source) * sizeof(CombineReadyNoticeV2),
                    sizeof(CombineReadyNoticeV2),
                    &wave_data.notices[source],
                    sizeof(CombineReadyNoticeV2),
                    ACL_MEMCPY_HOST_TO_DEVICE);
        }
        if (status == 0)
            status = CopyToDevice(registrations.data,
                                  wave_data.registrations) ? 0 : 1;
        if (status == 0)
            status = Fill(&acks, 0u) && Fill(&output, kPoison) &&
                Fill(&completions, 0u) ? 0 : 1;
        if (status == 0 && o.pe == inc_pe) {
            status = CopyToDevice(source_offsets.data,
                                  wave_data.plan.source_offsets) &&
                CopyToDevice(pulls.data, wave_data.plan.pulls) &&
                CopyToDevice(owner_offsets.data,
                             wave_data.plan.owner_offsets) &&
                CopyToDevice(results.data, wave_data.plan.results) &&
                CopyToDevice(pull_next.data,
                             wave_data.plan.pull_next) &&
                CopyToDevice(heads.data,
                             wave_data.plan.accumulator_heads) &&
                CopyToDevice(
                    counts.data,
                    wave_data.plan.accumulator_contributor_counts) &&
                CopyToDevice(
                    result_index.data,
                    wave_data.plan.accumulator_result_index) &&
                Fill(&ready_staging, kPoison) &&
                Fill(&ready_state, 0u) && Fill(&payload_offsets, 0u) &&
                Fill(&timeline_buffer, 0u) ? 0 : 1;
            if (status == 0)
                status = aclrtMemcpy(journal.data,
                    sizeof(JournalSlotHeader),
                    &wave_data.layout.journal_header,
                    sizeof(JournalSlotHeader), ACL_MEMCPY_HOST_TO_DEVICE);
        }
        if (status == 0) status = aclrtSynchronizeStream(stream);
        if (status == 0)
            std::cerr << "[STAGE] pe=" << o.pe << " prepared iteration="
                      << iteration << '\n' << std::flush;
        if (status == 0) aclshmem_barrier_all();
        if (status == 0 && !test::WaitForExternalStartGate(
                               "combine", o.pe, iteration)) {
            std::cerr << "[FAIL] combine external start gate\n";
            status = 2;
        }
        if (status != 0) { correct = false; break; }

        if (o.workload == Workload::READY_SKEW && o.pe < inc_pe) {
            const uint32_t reverse = o.workers - 1u -
                static_cast<uint32_t>(o.pe);
            std::this_thread::sleep_for(std::chrono::microseconds(
                static_cast<uint64_t>(reverse) * 750u));
        }
        // The standalone processes do not share a launch coordinator.  Give
        // worker READY kernels a bounded head start; a production resident
        // INC server is already polling before workers notify it.
        if (o.pe == inc_pe && !test::ExternalStartGateEnabled())
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const auto begin = std::chrono::steady_clock::now();
        std::cerr << "[STAGE] pe=" << o.pe << " launch iteration="
                  << iteration << '\n' << std::flush;
        launch_inc_dc_pull_combine_v2_device(
            combine_aiv, stream, partials.data, ready.data,
            notices.data, ready_staging.data, registrations.data,
            acks.data, output.data, completions.data, source_offsets.data,
            pulls.data, owner_offsets.data, results.data, journal.data,
            pull_next.data, heads.data, counts.data, result_index.data,
            ready_state.data, payload_offsets.data, timeline_buffer.data,
            shmemx_get_ffts_config(), kSessionId, kPlacementEpoch,
            generation, sequence, wave_data.plan.dispatch_cookie,
            sizing.output_slot_stride, wave_data.plan.pulls.size(),
            wave_data.plan.results.size(), pull_capacity,
            accumulator_capacity, source_capacity,
            wave_data.plan.accumulator_count, o.workers, o.hidden,
            static_cast<uint32_t>(PartialDataType::FP32), inc_pe, wave_id,
            ring_slot, kRingSlots, kSpinCap);
        status = aclrtSynchronizeStream(stream);
        std::cerr << "[STAGE] pe=" << o.pe << " kernel_status=" << status
                  << " iteration=" << iteration << '\n' << std::flush;
        const auto end = std::chrono::steady_clock::now();
        if (status == 0) aclshmem_barrier_all();
        if (status != 0) { correct = false; break; }

        CombineDeviceTimelineV2 timeline{};
        if (o.pe < inc_pe)
            correct = ValidateWorker(o, wave_data, ring_slot, acks, output,
                                     completions);
        else
            correct = ValidateInc(wave_data, journal, ready_staging,
                                  notices, ready_state, timeline_buffer,
                                  &timeline);
        for (GuardedBuffer *buffer : buffers)
            correct = GuardsValid(*buffer) && correct;
        if (status == 0) aclshmem_barrier_all();

        if (o.pe == inc_pe) {
            const double us = std::chrono::duration<double, std::micro>(
                end - begin).count();
            const double logical_bytes = static_cast<double>(
                wave_data.ingress_bytes + wave_data.egress_bytes);
            const double gbps = logical_bytes / us / 1.0e3;
            const bool warmup = iteration < o.warmup;
            std::cout << std::setprecision(12)
                      << "{\"test\":\"pull_combine_v2_npu_e2e\""
                      << ",\"iteration\":" << iteration
                      << ",\"warmup\":" << (warmup ? "true" : "false")
                      << ",\"workers\":" << o.workers
                      << ",\"workload\":\"" << o.workload_name << "\""
                      << ",\"hidden\":" << o.hidden
                      << ",\"rows\":" << o.rows
                      << ",\"ingress_bytes\":" << wave_data.ingress_bytes
                      << ",\"egress_bytes\":" << wave_data.egress_bytes
                      << ",\"e2e_us\":" << us
                      << ",\"logical_gb_s\":" << gbps
                      << ",\"status\":" << timeline.status
                      << ",\"cycle_kernel_start\":"
                      << timeline.kernel_start
                      << ",\"cycle_plan_index_done\":"
                      << timeline.plan_index_done
                      << ",\"cycle_kernel_done\":" << timeline.kernel_done
                      << ",\"correct\":" << (correct ? "true" : "false")
                      << "}\n";
            if (!warmup) {
                measured_us.push_back(us);
                measured_gbps.push_back(gbps);
            }
        }
    }

    if (correct && o.pe == inc_pe && !measured_gbps.empty()) {
        const double mean = std::accumulate(measured_gbps.begin(),
            measured_gbps.end(), 0.0) / measured_gbps.size();
        const double minimum = *std::min_element(measured_gbps.begin(),
                                                 measured_gbps.end());
        double variance = 0.0;
        for (double value : measured_gbps)
            variance += (value - mean) * (value - mean);
        variance /= measured_gbps.size();
        const double mean_us = std::accumulate(measured_us.begin(),
            measured_us.end(), 0.0) / measured_us.size();
        std::cout << std::setprecision(12)
                  << "{\"test\":\"pull_combine_v2_npu_e2e_summary\""
                  << ",\"workers\":" << o.workers
                  << ",\"workload\":\"" << o.workload_name << "\""
                  << ",\"hidden\":" << o.hidden
                  << ",\"rows\":" << o.rows
                  << ",\"measure\":" << measured_gbps.size()
                  << ",\"mean_us\":" << mean_us
                  << ",\"min_logical_gb_s\":" << minimum
                  << ",\"mean_logical_gb_s\":" << mean
                  << ",\"cv_pct\":"
                  << (mean == 0.0 ? 0.0 : 100.0 * std::sqrt(variance) / mean)
                  << ",\"correct\":true}\n";
    }

    for (auto it = buffers.rbegin(); it != buffers.rend(); ++it)
        Release(*it);
    if (shmem_initialized) aclshmem_finalize();
    if (stream != nullptr) aclrtDestroyStream(stream);
    if (device >= 0) aclrtResetDevice(device);
    aclFinalize();
    return correct ? 0 : Fail("NPU E2E", status);
}
