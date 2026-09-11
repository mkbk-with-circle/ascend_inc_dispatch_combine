#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "shmem.h"
#include "utils.h"

#include "inc_dc_partitioned_combine.h"
#include "inc_dc_partitioned_dispatch.h"
#include "inc_dc_pull_dispatch_v2.h"
#include "inc_dc_source_partition_layout.h"

using namespace inc::dc::pull_v2;

int g_npus = 5;
const char *ipport = "tcp://127.0.0.1:28798";
int f_pe = 0;
int f_npu = 0;
aclshmemx_uniqueid_t default_flag_uid;

namespace {

constexpr uint64_t kSessionId = 0x5350434456320001ull;
constexpr uint64_t kPlacementEpoch = 11u;
constexpr uint64_t kFirstGeneration = 5001u;
constexpr uint64_t kFirstSequence = 9001u;
constexpr uint32_t kFirstWave = 31u;
constexpr uint32_t kDispatchRegionId = 107u;
constexpr uint32_t kCombineRegionBase = 700u;
constexpr uint32_t kRingCount = 2u;
constexpr uint64_t kSpinCap = 5000000000ull;
constexpr uint64_t kGuardBytes = 64u;
constexpr uint8_t kHeadGuard = 0xa5u;
constexpr uint8_t kTailGuard = 0x5au;
constexpr uint8_t kPoison = 0xc7u;

enum class Mode {
    CHAIN,
    DISPATCH,
    COMBINE,
};

enum class RouteStyle {
    RANDOM_GPU,
    RANDOM_EXPERT,
    SYM_K2_BALANCED,
    SYM_K4_GPU4,
};

struct Options {
    uint32_t workers = 0u;
    int pe = -1;
    int first_npu = 0;
    std::vector<uint32_t> rows;
    uint32_t hidden = 0u;
    uint32_t expert_count = 0u;
    uint32_t topk = 0u;
    uint64_t route_seed = 0u;
    uint32_t waves = 0u;
    uint32_t delay_rank0_ms = 100u;
    uint32_t warmup = 0u;
    Mode mode = Mode::CHAIN;
    RouteStyle route_style = RouteStyle::RANDOM_GPU;
    const char *mode_name = "chain";
    const char *route_name = "random_gpu";
};

struct GuardedBuffer {
    uint8_t *allocation = nullptr;
    uint8_t *data = nullptr;
    uint64_t bytes = 0u;
    uint64_t allocated_bytes = 0u;
    bool symmetric = false;
    const char *name = nullptr;
};

struct RouteAssignment {
    uint32_t destination = 0u;
    uint32_t expert = 0u;
    uint32_t ordinal = 0u;
    float weight = 0.0f;
};

bool Add(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr || b > std::numeric_limits<uint64_t>::max() - a)
        return false;
    *out = a + b;
    return true;
}

bool Mul(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr ||
        (a != 0u && b > std::numeric_limits<uint64_t>::max() / a))
        return false;
    *out = a * b;
    return true;
}

bool AlignUp(uint64_t value, uint64_t alignment, uint64_t *out)
{
    if (out == nullptr || alignment == 0u ||
        (alignment & (alignment - 1u)) != 0u ||
        value > std::numeric_limits<uint64_t>::max() - (alignment - 1u))
        return false;
    *out = (value + alignment - 1u) & ~(alignment - 1u);
    return true;
}

int Fail(const char *step, int status = 1)
{
    std::cerr << "[FAIL] " << step << " status=" << status << '\n';
    return status == 0 ? 1 : status;
}

uint64_t Mix(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

bool ParseRows(const char *text, uint32_t workers,
               std::vector<uint32_t> *rows)
{
    if (text == nullptr || rows == nullptr) return false;
    std::stringstream stream(text);
    std::string item;
    std::vector<uint32_t> parsed;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) return false;
        char *end = nullptr;
        const unsigned long value = std::strtoul(item.c_str(), &end, 10);
        if (end == item.c_str() || *end != '\0' ||
            value > std::numeric_limits<uint32_t>::max())
            return false;
        parsed.push_back(static_cast<uint32_t>(value));
    }
    if (parsed.size() != workers) return false;
    *rows = std::move(parsed);
    return true;
}

bool ParseMode(const char *text, Mode *mode)
{
    if (text == nullptr || mode == nullptr) return false;
    if (std::strcmp(text, "chain") == 0)
        *mode = Mode::CHAIN;
    else if (std::strcmp(text, "dispatch") == 0)
        *mode = Mode::DISPATCH;
    else if (std::strcmp(text, "combine") == 0)
        *mode = Mode::COMBINE;
    else
        return false;
    return true;
}

bool ParseRouteStyle(const char *text, RouteStyle *style)
{
    if (text == nullptr || style == nullptr) return false;
    if (std::strcmp(text, "random_gpu") == 0)
        *style = RouteStyle::RANDOM_GPU;
    else if (std::strcmp(text, "random_expert") == 0)
        *style = RouteStyle::RANDOM_EXPERT;
    else if (std::strcmp(text, "sym_k2_balanced") == 0)
        *style = RouteStyle::SYM_K2_BALANCED;
    else if (std::strcmp(text, "sym_k4_gpu4") == 0)
        *style = RouteStyle::SYM_K4_GPU4;
    else
        return false;
    return true;
}

uint32_t RowsFor(const Options &options, uint32_t origin,
                 uint32_t iteration)
{
    return options.rows[(origin + iteration) % options.workers];
}

std::vector<RouteAssignment> RouteFor(const Options &options,
                                      uint32_t origin, uint32_t token,
                                      uint32_t iteration)
{
    uint64_t random = Mix(options.route_seed ^
        (static_cast<uint64_t>(iteration) << 48u) ^
        (static_cast<uint64_t>(origin) << 32u) ^ token);
    const uint32_t local_experts = options.expert_count / options.workers;
    std::vector<RouteAssignment> route;
    route.reserve(options.topk);
    if (options.route_style == RouteStyle::RANDOM_EXPERT) {
        std::vector<uint32_t> experts(options.expert_count);
        std::iota(experts.begin(), experts.end(), 0u);
        for (uint32_t ordinal = 0u; ordinal < options.topk; ++ordinal) {
            const uint32_t selected = ordinal + static_cast<uint32_t>(
                Mix(random ^ ordinal) %
                (options.expert_count - ordinal));
            std::swap(experts[ordinal], experts[selected]);
            const uint32_t expert = experts[ordinal];
            const uint32_t destination = expert / local_experts;
            route.push_back({destination, expert, ordinal,
                             1.0f / static_cast<float>(options.topk)});
        }
        return route;
    }

    std::vector<uint32_t> destinations(options.workers);
    std::iota(destinations.begin(), destinations.end(), 0u);
    if (options.route_style == RouteStyle::RANDOM_GPU) {
        for (uint32_t i = options.workers - 1u; i != 0u; --i) {
            const uint32_t selected = static_cast<uint32_t>(
                Mix(random ^ i) % (i + 1u));
            std::swap(destinations[i], destinations[selected]);
        }
    } else if (options.route_style == RouteStyle::SYM_K2_BALANCED) {
        destinations[0] = origin;
        destinations[1] = (origin + 1u) % options.workers;
    }
    for (uint32_t ordinal = 0u; ordinal < options.topk; ++ordinal) {
        const uint32_t destination = options.route_style ==
                RouteStyle::SYM_K4_GPU4
            ? ordinal : destinations[ordinal % options.workers];
        const uint32_t cycle = ordinal / options.workers;
        const uint32_t local = static_cast<uint32_t>(
            options.route_style == RouteStyle::SYM_K4_GPU4
                ? origin % local_experts
                : (Mix(random ^
                       (static_cast<uint64_t>(destination) << 24u)) +
                   cycle) % local_experts);
        route.push_back({destination,
                         destination * local_experts + local,
                         ordinal,
                         1.0f / static_cast<float>(options.topk)});
    }
    return route;
}

uint64_t TokenId(uint32_t origin, uint32_t token, uint32_t iteration)
{
    return (static_cast<uint64_t>(origin) << 48u) |
        (static_cast<uint64_t>(iteration) << 32u) | token;
}

uint8_t HiddenByte(const Options &options, uint32_t origin, uint32_t token,
                   uint32_t iteration, uint64_t byte)
{
    const uint64_t block = byte >> 8u;
    const uint64_t lane = byte & 255u;
    return static_cast<uint8_t>((Mix(
        options.route_seed ^ (static_cast<uint64_t>(iteration) << 52u) ^
        (static_cast<uint64_t>(origin) << 40u) ^
        (static_cast<uint64_t>(token) << 12u) ^ block) + lane) % 251u);
}

float PartialValue(uint32_t worker_b, uint32_t origin, uint32_t token,
                   uint32_t element, uint32_t iteration)
{
    // All terms are binary fractions, so the CPU oracle and NPU FP32 Add use
    // the same exactly representable local-FFN placeholder values.
    return static_cast<float>(worker_b + 1u) +
        static_cast<float>(origin) / 8.0f +
        static_cast<float>(token) / 1024.0f +
        static_cast<float>(element & 31u) / 32768.0f +
        static_cast<float>(iteration & 7u) / 262144.0f;
}

SourceInput MakeSourceInput(const Options &options, uint32_t origin,
                            uint32_t iteration, uint64_t generation,
                            uint64_t sequence, uint32_t wave,
                            uint32_t ring)
{
    SourceInput input{};
    input.session.session_id = kSessionId;
    input.session.placement_epoch = kPlacementEpoch;
    input.session.worker_count = options.workers;
    input.session.expert_count = options.expert_count;
    input.session.hidden = options.hidden;
    input.session.dtype = DataType::BF16;
    input.session.ring_slots = kRingCount;
    input.generation = generation;
    input.sequence = sequence;
    input.wave = wave;
    input.source_rank = origin;
    input.source_region_id = kDispatchRegionId;
    input.ring_slot = static_cast<uint16_t>(ring);
    const uint32_t rows = RowsFor(options, origin, iteration);
    input.token_ids.reserve(rows);
    input.assignment_offsets.reserve(static_cast<size_t>(rows) + 1u);
    input.assignment_offsets.push_back(0u);
    for (uint32_t token = 0u; token < rows; ++token) {
        input.token_ids.push_back(TokenId(origin, token, iteration));
        for (const RouteAssignment &route :
             RouteFor(options, origin, token, iteration)) {
            AssignmentRecord assignment{};
            assignment.destination_rank = route.destination;
            assignment.expert_id = route.expert;
            assignment.ordinal = route.ordinal;
            assignment.weight = route.weight;
            input.assignments.push_back(assignment);
        }
        input.assignment_offsets.push_back(
            static_cast<uint32_t>(input.assignments.size()));
    }
    uint64_t hidden_bytes = 0u;
    const uint64_t row_bytes = static_cast<uint64_t>(options.hidden) * 2u;
    if (!Mul(rows, row_bytes, &hidden_bytes) ||
        hidden_bytes > std::numeric_limits<size_t>::max())
        return SourceInput{};
    input.hidden_payload.resize(static_cast<size_t>(hidden_bytes));
    for (uint32_t token = 0u; token < rows; ++token)
        for (uint64_t byte = 0u; byte < row_bytes; ++byte)
            input.hidden_payload[static_cast<uint64_t>(token) * row_bytes +
                                 byte] =
                HiddenByte(options, origin, token, iteration, byte);
    return input;
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
            ACL_SUCCESS)
            base = static_cast<uint8_t *>(memory);
    }
    if (base == nullptr) return false;
    buffer->allocation = base;
    buffer->data = base + kGuardBytes;
    buffer->bytes = bytes;
    buffer->allocated_bytes = payload;
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
    if (buffer->symmetric)
        aclshmem_free(buffer->allocation);
    else
        aclrtFree(buffer->allocation);
    *buffer = GuardedBuffer{};
}

bool Fill(GuardedBuffer *buffer, uint8_t value)
{
    return buffer != nullptr && aclrtMemset(
        buffer->data, buffer->allocated_bytes, value,
        buffer->allocated_bytes) == ACL_SUCCESS;
}

bool GuardsValid(const GuardedBuffer &buffer)
{
    std::vector<uint8_t> head(kGuardBytes);
    std::vector<uint8_t> tail(kGuardBytes);
    if (buffer.allocation == nullptr ||
        aclrtMemcpy(head.data(), kGuardBytes, buffer.allocation, kGuardBytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(tail.data(), kGuardBytes,
                    buffer.data + buffer.allocated_bytes, kGuardBytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
        return false;
    const bool valid =
        std::all_of(head.begin(), head.end(),
                    [](uint8_t byte) { return byte == kHeadGuard; }) &&
        std::all_of(tail.begin(), tail.end(),
                    [](uint8_t byte) { return byte == kTailGuard; });
    if (!valid) std::cerr << "[FAIL] guard changed: " << buffer.name << '\n';
    return valid;
}

bool SymmetricHeapBytes(const Options &options,
                        const SourcePartitionLayout &layout,
                        uint64_t *heap_bytes)
{
    uint64_t ready_bytes = 0u;
    uint64_t dispatch_ack_bytes = 0u;
    if (heap_bytes == nullptr ||
        !Mul(options.workers, sizeof(Ready), &ready_bytes) ||
        !Mul(options.workers, sizeof(SourceConsumed), &dispatch_ack_bytes))
        return false;
    const uint64_t allocations[]{
        layout.source.worker_stride,
        ready_bytes,
        dispatch_ack_bytes,
        layout.destination.hidden_arena_bytes,
        layout.destination.rows_arena_bytes,
        layout.destination.assignments_arena_bytes,
        layout.destination.expert_counts_arena_bytes,
        layout.origin_ring_control128_bytes, // D completion
        layout.combine.partial_arena_bytes,
        layout.origin_ring_control128_bytes, // C READY
        layout.origin_ring_control64_bytes,  // C notice
        layout.origin_ring_control128_bytes, // C ACK
        layout.combine.output_arena_bytes,
        layout.origin_ring_control128_bytes, // C owner completion
    };
    uint64_t total = 0u;
    for (uint64_t requested : allocations) {
        uint64_t guarded = 0u;
        uint64_t allocation = 0u;
        if (!Add(std::max<uint64_t>(requested, 64u), 2u * kGuardBytes,
                 &guarded) ||
            !AlignUp(guarded, kSourcePartitionAlignment, &allocation) ||
            !Add(total, allocation, &total))
            return false;
    }
    constexpr uint64_t kTransportSlack = 16ull * 1024ull * 1024ull;
    constexpr uint64_t kHeapAlignment = 2ull * 1024ull * 1024ull;
    return Add(total, kTransportSlack, &total) &&
        AlignUp(total, kHeapAlignment, heap_bytes);
}

template <typename T>
bool CopyFromDevice(T *host, const uint8_t *device)
{
    return host != nullptr && device != nullptr &&
        aclrtMemcpy(host, sizeof(T), device, sizeof(T),
                    ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

template <typename T>
bool CopyVectorFromDevice(std::vector<T> *host, const uint8_t *device,
                          uint64_t count)
{
    if (host == nullptr || device == nullptr ||
        count > std::numeric_limits<size_t>::max() / sizeof(T))
        return false;
    host->resize(static_cast<size_t>(count));
    return count == 0u ||
        aclrtMemcpy(host->data(), count * sizeof(T), device,
                    count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST) ==
            ACL_SUCCESS;
}

template <typename T>
bool CopyToDevice(uint8_t *device, const T &host)
{
    return device != nullptr &&
        aclrtMemcpy(device, sizeof(T), &host, sizeof(T),
                    ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

bool OriginControlOffset(const SourcePartitionLayout &layout,
                         uint32_t origin, uint32_t ring,
                         uint32_t worker_b, uint32_t record_bytes,
                         uint64_t *offset)
{
    return SourcePartitionOffsets::OriginRingControl(
        layout, origin, ring, worker_b, record_bytes, offset);
}

uint64_t IncDispatchBase(const SourcePartitionLayout &layout,
                         uint32_t origin, uint32_t ring)
{
    uint64_t offset = 0u;
    return SourcePartitionOffsets::IncDispatchWorkspace(
               layout, origin, ring, &offset)
        ? offset : std::numeric_limits<uint64_t>::max();
}

uint64_t JournalBase(const SourcePartitionLayout &layout, uint32_t origin,
                     uint32_t ring)
{
    uint64_t offset = 0u;
    return SourcePartitionOffsets::IncJournal(layout, origin, ring, &offset)
        ? offset : std::numeric_limits<uint64_t>::max();
}

uint64_t IncCombineBase(const SourcePartitionLayout &layout,
                        uint32_t origin, uint32_t ring)
{
    uint64_t offset = 0u;
    return SourcePartitionOffsets::IncCombineWorkspace(
               layout, origin, ring, &offset)
        ? offset : std::numeric_limits<uint64_t>::max();
}

uint64_t OriginDataBase(uint32_t origin, uint64_t partition_stride)
{
    return static_cast<uint64_t>(origin) * partition_stride;
}

uint64_t OriginControlBase(const SourcePartitionLayout &layout,
                           uint32_t origin, uint32_t record_bytes)
{
    uint64_t offset = 0u;
    return OriginControlOffset(layout, origin, 0u, 0u, record_bytes,
                               &offset)
        ? offset : std::numeric_limits<uint64_t>::max();
}

uint64_t DispatchAckPublication(uint64_t generation, uint64_t sequence,
                                uint32_t source, uint32_t status)
{
    uint64_t value = generation ^ (sequence << 1u) ^
        (static_cast<uint64_t>(source) << 48u) ^
        (static_cast<uint64_t>(status) << 24u) ^
        0xa55aa55aa55aa55aull;
    return value == 0u ? 1u : value;
}

uint64_t DispatchCompletionPublication(uint64_t generation,
                                       uint64_t sequence,
                                       uint32_t destination,
                                       uint32_t status)
{
    uint64_t value = generation ^ (sequence << 3u) ^
        (static_cast<uint64_t>(destination) << 44u) ^
        (static_cast<uint64_t>(status) << 20u) ^
        0x5aa55aa55aa55aa5ull;
    return value == 0u ? 1u : value;
}

bool CompletionMatches(const DestinationCompletion &completion,
                       uint32_t destination, uint32_t origin,
                       uint32_t ring, uint64_t generation,
                       uint64_t sequence, uint32_t wave)
{
    return completion.magic == kPullDispatchMagic &&
        completion.abi_version == kPullDispatchAbiVersion &&
        completion.struct_bytes == sizeof(DestinationCompletion) &&
        completion.generation == generation &&
        completion.sequence == sequence && completion.wave == wave &&
        completion.destination_rank == destination &&
        completion.ring_slot == ring && completion.status == 0u &&
        completion.dispatch_cookie != 0u &&
        completion.publication == DispatchCompletionPublication(
            generation, sequence, destination, 0u) &&
        origin < kPullDispatchMaxWorkers;
}

bool ValidateDispatchAck(uint32_t source, uint32_t ring,
                         uint64_t generation, uint64_t sequence,
                         uint32_t wave, uint64_t expected_packet_bytes,
                         uint64_t expected_cookie,
                         GuardedBuffer *dispatch_source_acks)
{
    SourceConsumed ack{};
    if (!CopyFromDevice(
            &ack, dispatch_source_acks->data +
                      static_cast<uint64_t>(source) * sizeof(ack)))
        return false;
    return ack.magic == kPullDispatchMagic &&
        ack.abi_version == kPullDispatchAbiVersion &&
        ack.struct_bytes == sizeof(ack) && ack.session_id == kSessionId &&
        ack.placement_epoch == kPlacementEpoch &&
        ack.generation == generation && ack.sequence == sequence &&
        ack.dispatch_cookie == expected_cookie && ack.wave == wave &&
        ack.source_rank == source &&
        ack.source_region_id == kDispatchRegionId && ack.status == 0u &&
        ack.ring_slot == ring && ack.bytes_consumed == expected_packet_bytes &&
        ack.publication ==
            DispatchAckPublication(generation, sequence, source, 0u);
}

bool ActualCommunicationBytes(const Options &options,
                              const SourcePartitionLayout &layout,
                              uint32_t ring, GuardedBuffer *inc_dispatch,
                              uint64_t *dispatch_fanout_bytes,
                              uint64_t *combine_ingress_bytes)
{
    uint64_t rows = 0u;
    for (uint32_t origin = 0u; origin < options.workers; ++origin) {
        const uint64_t workspace = IncDispatchBase(layout, origin, ring);
        std::vector<uint32_t> counts;
        if (!CopyVectorFromDevice(
                &counts,
                inc_dispatch->data + workspace +
                    layout.inc_dispatch.destination_row_counts_offset,
                options.workers))
            return false;
        for (uint32_t count : counts)
            if (!Add(rows, count, &rows)) return false;
    }
    return Mul(rows, layout.dispatch_row_bytes, dispatch_fanout_bytes) &&
        Mul(rows, layout.combine_row_bytes, combine_ingress_bytes);
}

std::vector<uint32_t> UniqueDestinations(const Options &options,
                                         uint32_t origin, uint32_t token,
                                         uint32_t iteration)
{
    std::vector<uint32_t> destinations;
    for (const RouteAssignment &assignment :
         RouteFor(options, origin, token, iteration))
        destinations.push_back(assignment.destination);
    std::sort(destinations.begin(), destinations.end());
    destinations.erase(std::unique(destinations.begin(), destinations.end()),
                       destinations.end());
    return destinations;
}

bool PrepareCombinePartial(
    const Options &options, const SourcePartitionLayout &layout,
    uint32_t local_worker, uint32_t origin, uint32_t iteration,
    uint32_t ring, uint64_t generation, uint64_t sequence, uint32_t wave,
    const DestinationCompletion &completion, GuardedBuffer *destination_rows,
    GuardedBuffer *destination_assignments,
    GuardedBuffer *destination_hidden, GuardedBuffer *partials,
    GuardedBuffer *combine_ready, GuardedBuffer *combine_notices)
{
    const uint32_t source_rows = RowsFor(options, origin, iteration);
    std::vector<uint32_t> expected_tokens;
    std::vector<uint32_t> expected_assignment_counts;
    std::vector<RouteAssignment> expected_assignments;
    std::vector<uint32_t> expected_assignment_rows;
    for (uint32_t token = 0u; token < source_rows; ++token) {
        uint32_t assignments = 0u;
        const uint32_t destination_row =
            static_cast<uint32_t>(expected_tokens.size());
        for (const RouteAssignment &assignment :
             RouteFor(options, origin, token, iteration))
            if (assignment.destination == local_worker) {
                ++assignments;
                expected_assignments.push_back(assignment);
                expected_assignment_rows.push_back(destination_row);
            }
        if (assignments != 0u) {
            expected_tokens.push_back(token);
            expected_assignment_counts.push_back(assignments);
        }
    }
    const uint32_t expected_assignment_total = std::accumulate(
        expected_assignment_counts.begin(), expected_assignment_counts.end(),
        uint32_t{0});
    if (completion.row_count != expected_tokens.size() ||
        completion.assignment_count != expected_assignment_total) {
        std::cerr << "[FAIL] D counts origin=" << origin
                  << " B=" << local_worker << " rows="
                  << completion.row_count << " expected="
                  << expected_tokens.size() << " assignments="
                  << completion.assignment_count << " expected="
                  << expected_assignment_total << '\n';
        return false;
    }

    uint64_t rows_offset = 0u;
    uint64_t assignments_offset = 0u;
    uint64_t hidden_offset = 0u;
    uint64_t partial_offset = 0u;
    if (completion.row_count != 0u &&
        (!SourcePartitionOffsets::DestinationRow(
             layout, ring, origin, 0u, &rows_offset) ||
         !SourcePartitionOffsets::DestinationHidden(
             layout, ring, origin, 0u, 0u, &hidden_offset) ||
         !SourcePartitionOffsets::DestinationAssignment(
             layout, ring, origin, 0u, &assignments_offset) ||
         !SourcePartitionOffsets::CombinePartial(
             layout, ring, origin, 0u, 0u, &partial_offset)))
        return false;
    std::vector<DestinationRow> rows;
    if (completion.row_count != 0u &&
        !CopyVectorFromDevice(&rows, destination_rows->data + rows_offset,
                              completion.row_count))
        return false;
    std::vector<ExpertAssignment> actual_assignments;
    if (completion.assignment_count != 0u &&
        !CopyVectorFromDevice(
            &actual_assignments,
            destination_assignments->data + assignments_offset,
            completion.assignment_count))
        return false;

    const uint64_t dispatch_row_bytes = layout.dispatch_row_bytes;
    std::vector<uint8_t> actual_hidden(
        static_cast<size_t>(dispatch_row_bytes));
    uint32_t assignment_cursor = 0u;
    for (uint32_t row = 0u; row < completion.row_count; ++row) {
        const uint32_t token = expected_tokens[row];
        const DestinationRow &metadata = rows[row];
        if (metadata.route_key != RouteKey(origin, token) ||
            metadata.token_id != TokenId(origin, token, iteration) ||
            metadata.source_rank != origin ||
            metadata.source_token != token ||
            metadata.destination_row != row ||
            metadata.assignments_begin != assignment_cursor ||
            metadata.assignments_count != expected_assignment_counts[row]) {
            std::cerr << "[FAIL] D row metadata origin=" << origin
                      << " B=" << local_worker << " row=" << row << '\n';
            return false;
        }
        assignment_cursor += metadata.assignments_count;
        if (aclrtMemcpy(actual_hidden.data(), dispatch_row_bytes,
                        destination_hidden->data + hidden_offset +
                            static_cast<uint64_t>(row) * dispatch_row_bytes,
                        dispatch_row_bytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
            ACL_SUCCESS)
            return false;
        for (uint64_t byte = 0u; byte < dispatch_row_bytes; ++byte)
            if (actual_hidden[byte] != HiddenByte(
                    options, origin, token, iteration, byte)) {
                std::cerr << "[FAIL] D hidden origin=" << origin
                          << " B=" << local_worker << " row=" << row
                          << " byte=" << byte << '\n';
                return false;
            }
    }
    if (actual_assignments.size() != expected_assignments.size())
        return false;
    for (uint32_t index = 0u; index < actual_assignments.size(); ++index) {
        const RouteAssignment &expected = expected_assignments[index];
        const ExpertAssignment &actual = actual_assignments[index];
        if (actual.destination_row != expected_assignment_rows[index] ||
            actual.expert_id != expected.expert ||
            actual.ordinal != expected.ordinal ||
            actual.weight != expected.weight) {
            std::cerr << "[FAIL] D assignment origin=" << origin
                      << " B=" << local_worker << " index=" << index
                      << '\n';
            return false;
        }
    }

    // This deterministic local-FFN stand-in is intentionally prepared after
    // D completion and outside any communication timing interval.
    std::vector<float> partial(
        static_cast<size_t>(completion.row_count) * options.hidden);
    for (uint32_t row = 0u; row < completion.row_count; ++row)
        for (uint32_t element = 0u; element < options.hidden; ++element)
            partial[static_cast<uint64_t>(row) * options.hidden + element] =
                PartialValue(local_worker, origin, expected_tokens[row],
                             element, iteration);
    const uint64_t partial_bytes = partial.size() * sizeof(float);
    if (partial_bytes != 0u &&
        aclrtMemcpy(partials->data + partial_offset, partial_bytes,
                    partial.data(), partial_bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
        return false;

    CombineReadyV2 ready{};
    ready.session_id = kSessionId;
    ready.placement_epoch = kPlacementEpoch;
    ready.generation = generation;
    ready.sequence = sequence;
    ready.dispatch_cookie = completion.dispatch_cookie;
    ready.wave = wave;
    ready.source_rank = local_worker;
    ready.source_region_id = kCombineRegionBase + local_worker;
    ready.ring_slot = static_cast<uint16_t>(ring);
    ready.row_count = completion.row_count;
    ready.hidden = options.hidden;
    ready.partial_dtype = static_cast<uint32_t>(PartialDataType::FP32);
    if (!SourcePartitionOffsets::CombinePartial(
            layout, ring, origin, 0u, 0u, &ready.source_offset)) {
        // Zero-capacity layouts are rejected by the CLI, so row zero always
        // has a fixed address even when this B's live row count is zero.
        return false;
    }
    ready.payload_bytes = partial_bytes;
    ready.publication = CombineReadyPublication(ready);
    CombineReadyNoticeV2 notice{};
    notice.session_id = kSessionId;
    notice.placement_epoch = kPlacementEpoch;
    notice.generation = generation;
    notice.sequence = sequence;
    notice.wave = wave;
    notice.source_rank = local_worker;
    notice.ring_slot = static_cast<uint16_t>(ring);
    notice.publication = CombineReadyNoticePublication(notice);
    uint64_t ready_offset = 0u;
    uint64_t notice_offset = 0u;
    return OriginControlOffset(layout, origin, ring, local_worker,
                               sizeof(CombineReadyV2), &ready_offset) &&
        OriginControlOffset(layout, origin, ring, local_worker,
                            sizeof(CombineReadyNoticeV2), &notice_offset) &&
        CopyToDevice(combine_ready->data + ready_offset, ready) &&
        CopyToDevice(combine_notices->data + notice_offset, notice);
}

PartitionedDispatchLaunchArgs MakeDispatchArgs(
    const Options &options, const SourcePartitionLayout &layout,
    uint32_t origin, uint32_t ring, uint64_t generation,
    uint64_t sequence, uint32_t wave, int inc_pe,
    GuardedBuffer *source_region, GuardedBuffer *ready_mailbox,
    GuardedBuffer *source_acks, GuardedBuffer *destination_hidden,
    GuardedBuffer *destination_rows,
    GuardedBuffer *destination_assignments,
    GuardedBuffer *destination_expert_counts,
    GuardedBuffer *destination_completions,
    GuardedBuffer *inc_dispatch, GuardedBuffer *inc_journal)
{
    const uint64_t workspace = IncDispatchBase(layout, origin, ring);
    const uint64_t journal = JournalBase(layout, origin, ring);
    PartitionedDispatchLaunchArgs args{};
    args.origin_rank = origin;
    args.source_region = source_region->data;
    args.ready_mailbox = ready_mailbox->data;
    args.inc_slots = inc_dispatch->data + workspace +
        layout.inc_dispatch.source_packet_offset;
    args.source_acks = source_acks->data;
    args.destination_hidden = destination_hidden->data +
        OriginDataBase(origin,
                       layout.destination.hidden_partition_stride);
    args.destination_rows = destination_rows->data +
        OriginDataBase(origin, layout.destination.rows_partition_stride);
    args.destination_assignments = destination_assignments->data +
        OriginDataBase(
            origin, layout.destination.assignments_partition_stride);
    args.destination_expert_counts =
        destination_expert_counts->data + OriginDataBase(
            origin, layout.destination.expert_counts_partition_stride);
    args.destination_completions = destination_completions->data +
        OriginControlBase(layout, origin, sizeof(DestinationCompletion));
    args.inc_destination_rows = inc_dispatch->data + workspace +
        layout.inc_dispatch.destination_rows_offset;
    args.inc_destination_assignments = inc_dispatch->data + workspace +
        layout.inc_dispatch.destination_assignments_offset;
    args.journal_header = inc_journal->data + journal +
        layout.inc_journal.header_offset;
    args.journal_tokens = inc_journal->data + journal +
        layout.inc_journal.tokens_offset;
    args.journal_contributors = inc_journal->data + journal +
        layout.inc_journal.contributors_offset;
    args.journal_assignments = inc_journal->data + journal +
        layout.inc_journal.assignments_offset;
    args.row_map = inc_dispatch->data + workspace +
        layout.inc_dispatch.row_map_offset;
    args.source_token_prefix = inc_dispatch->data + workspace +
        layout.inc_dispatch.source_token_prefix_offset;
    args.source_destination_prefix = inc_dispatch->data + workspace +
        layout.inc_dispatch.source_destination_prefix_offset;
    args.destination_row_counts = inc_dispatch->data + workspace +
        layout.inc_dispatch.destination_row_counts_offset;
    args.destination_assignment_counts = inc_dispatch->data + workspace +
        layout.inc_dispatch.destination_assignment_counts_offset;
    args.expert_counts = inc_dispatch->data + workspace +
        layout.inc_dispatch.expert_counts_offset;
    args.parser_scratch = inc_dispatch->data + workspace +
        layout.inc_dispatch.parser_scratch_offset;
    args.status_line = inc_dispatch->data + workspace +
        layout.inc_dispatch.timeline_offset;
    args.ffts_addr = shmemx_get_ffts_config();
    args.session_id = kSessionId;
    args.placement_epoch = kPlacementEpoch;
    args.generation = generation;
    args.sequence = sequence;
    args.source_slot_stride = layout.source.packet_stride;
    args.destination_hidden_slot_stride =
        layout.config.worker_count *
        layout.destination.hidden_partition_stride;
    args.destination_rows_slot_stride =
        layout.config.worker_count *
        layout.destination.rows_partition_stride;
    args.destination_assignments_slot_stride =
        layout.config.worker_count *
        layout.destination.assignments_partition_stride;
    args.destination_expert_counts_slot_stride =
        layout.config.worker_count *
        layout.destination.expert_counts_partition_stride;
    args.journal_token_capacity = layout.row_capacity;
    args.journal_contributor_capacity = layout.contributor_capacity;
    // Partitioned Combine consumes token/contributor ranges directly. The
    // duplicated full assignment journal is optional diagnostics storage.
    args.journal_assignment_capacity = 0u;
    args.destination_row_capacity = layout.row_capacity;
    args.destination_assignment_capacity = layout.assignment_capacity;
    args.inc_destination_rows_stride_bytes =
        layout.destination.rows_partition_stride;
    args.inc_destination_assignments_stride_bytes =
        layout.destination.assignments_partition_stride;
    args.row_map_capacity_entries =
        layout.inc_dispatch.row_map_capacity_entries;
    args.source_destination_prefix_capacity_entries =
        layout.inc_dispatch.source_destination_prefix_capacity_entries;
    args.expert_counts_capacity_entries =
        layout.inc_dispatch.expert_counts_capacity_entries;
    args.parser_scratch_capacity_entries =
        layout.inc_dispatch.parser_scratch_capacity_entries;
    args.worker_count = options.workers;
    args.expert_count = options.expert_count;
    args.hidden = options.hidden;
    args.dtype = static_cast<uint32_t>(DataType::BF16);
    args.inc_pe = inc_pe;
    args.region_id = kDispatchRegionId;
    args.wave = wave;
    args.ring_slot = ring;
    args.slot_count = kRingCount;
    // Let the partitioned wrapper select the balanced cohort default:
    // W4/AIV6 -> 3 relay lanes; W2/AIV12 -> 5 relay lanes.
    args.channels_per_source = 0u;
    args.spin_cap = kSpinCap;
    return args;
}

PartitionedCombineLaunchArgs MakeCombineArgs(
    const Options &options, const SourcePartitionLayout &layout,
    uint32_t origin, uint32_t ring, uint64_t generation,
    uint64_t sequence, uint32_t wave, int inc_pe, GuardedBuffer *partials,
    GuardedBuffer *combine_ready, GuardedBuffer *combine_notices,
    GuardedBuffer *combine_registrations, GuardedBuffer *combine_acks,
    GuardedBuffer *owner_output, GuardedBuffer *owner_completions,
    GuardedBuffer *inc_dispatch, GuardedBuffer *inc_journal,
    GuardedBuffer *inc_combine)
{
    uint64_t ready_base = 0u;
    uint64_t notice_base = 0u;
    uint64_t ack_base = 0u;
    uint64_t completion_base = 0u;
    OriginControlOffset(layout, origin, ring, 0u, sizeof(CombineReadyV2),
                        &ready_base);
    OriginControlOffset(layout, origin, ring, 0u,
                        sizeof(CombineReadyNoticeV2), &notice_base);
    OriginControlOffset(layout, origin, ring, 0u,
                        sizeof(PartitionedCombineSourceAck), &ack_base);
    OriginControlOffset(
        layout, origin, ring, 0u,
        sizeof(PartitionedCombineOwnerCompletion), &completion_base);
    const uint64_t workspace = IncDispatchBase(layout, origin, ring);
    const uint64_t journal = JournalBase(layout, origin, ring);
    const uint64_t combine_workspace = IncCombineBase(layout, origin, ring);
    PartitionedCombineLaunchArgs args{};
    args.symmetric_partials = partials->data;
    args.ready_records = combine_ready->data + ready_base;
    args.ready_notices = combine_notices->data + notice_base;
    args.ready_staging = inc_combine->data + combine_workspace +
        layout.inc_combine.ready_staging_offset;
    args.registrations = combine_registrations->data;
    args.source_acks = combine_acks->data + ack_base;
    args.owner_output = owner_output->data;
    args.owner_completions = owner_completions->data + completion_base;
    args.journal_header = inc_journal->data + journal +
        layout.inc_journal.header_offset;
    args.journal_tokens = inc_journal->data + journal +
        layout.inc_journal.tokens_offset;
    args.journal_contributors = inc_journal->data + journal +
        layout.inc_journal.contributors_offset;
    args.destination_row_counts = inc_dispatch->data + workspace +
        layout.inc_dispatch.destination_row_counts_offset;
    args.source_state = inc_combine->data + combine_workspace +
        layout.inc_combine.source_state_offset;
    args.source_payload_offsets = inc_combine->data + combine_workspace +
        layout.inc_combine.source_payload_offsets_offset;
    args.status_line = inc_combine->data + combine_workspace +
        layout.inc_combine.timeline_offset;
    args.ffts_addr = shmemx_get_ffts_config();
    args.session_id = kSessionId;
    args.placement_epoch = kPlacementEpoch;
    args.generation = generation;
    args.sequence = sequence;
    args.dispatch_cookie = 0u;
    args.owner_output_slot_stride = layout.combine.output_ring_stride;
    args.journal_token_capacity = layout.row_capacity;
    args.journal_contributor_capacity = layout.contributor_capacity;
    args.source_scratch_capacity = options.workers;
    args.validation_scratch = inc_combine->data + combine_workspace +
        layout.inc_combine.validation_scratch_offset;
    args.validation_scratch_capacity_bytes = layout.inc_combine.validation_scratch_bytes;
    args.spin_cap = kSpinCap;
    args.worker_count = options.workers;
    args.hidden = options.hidden;
    args.origin_rank = origin;
    args.origin_row_capacity = layout.row_capacity;
    args.inc_pe = inc_pe;
    args.wave = wave;
    args.ring_slot = ring;
    args.slot_count = kRingCount;
    return args;
}

bool ValidateOwnerOutput(const Options &options,
                         const SourcePartitionLayout &layout,
                         uint32_t owner, uint32_t iteration, uint32_t ring,
                         uint64_t generation, uint64_t sequence,
                         uint32_t wave, GuardedBuffer *owner_output,
                         GuardedBuffer *owner_completions)
{
    uint64_t completion_offset = 0u;
    if (!OriginControlOffset(
            layout, owner, ring, owner,
            sizeof(PartitionedCombineOwnerCompletion),
            &completion_offset))
        return false;
    PartitionedCombineOwnerCompletion completion{};
    if (!CopyFromDevice(&completion,
                        owner_completions->data + completion_offset) ||
        completion.magic != kPullCombineV2Magic ||
        completion.abi_version != kPullCombineV2AbiVersion ||
        completion.struct_bytes != sizeof(completion) ||
        completion.session_id != kSessionId ||
        completion.placement_epoch != kPlacementEpoch ||
        completion.generation != generation ||
        completion.sequence != sequence || completion.wave != wave ||
        completion.owner_rank != owner || completion.status != 0u ||
        completion.row_count != RowsFor(options, owner, iteration) ||
        completion.ring_slot != ring || completion.publication == 0u) {
        std::cerr << "[FAIL] C owner completion owner=" << owner << '\n';
        return false;
    }
    const uint32_t rows = completion.row_count;
    std::vector<float> output(static_cast<size_t>(rows) * options.hidden);
    const uint64_t output_offset =
        static_cast<uint64_t>(ring) * layout.combine.output_ring_stride;
    const uint64_t output_bytes = output.size() * sizeof(float);
    if (output_bytes != 0u &&
        aclrtMemcpy(output.data(), output_bytes,
                    owner_output->data + output_offset, output_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
        return false;
    for (uint32_t token = 0u; token < rows; ++token) {
        const std::vector<uint32_t> contributors = UniqueDestinations(
            options, owner, token, iteration);
        for (uint32_t element = 0u; element < options.hidden; ++element) {
            float expected = 0.0f;
            for (uint32_t worker_b : contributors)
                expected += PartialValue(worker_b, owner, token, element,
                                         iteration);
            const float actual = output[
                static_cast<uint64_t>(token) * options.hidden + element];
            if (std::fabs(actual - expected) > 1e-5f) {
                std::cerr << "[FAIL] C output owner=" << owner
                          << " token=" << token << " element=" << element
                          << " actual=" << actual
                          << " expected=" << expected << '\n';
                return false;
            }
        }
    }
    return true;
}

bool ValidateJournal(const Options &options,
                     const SourcePartitionLayout &layout, uint32_t origin,
                     uint32_t iteration, uint32_t ring, uint64_t generation,
                     uint64_t sequence, uint32_t wave,
                     JournalSlotState expected_state,
                     GuardedBuffer *inc_journal)
{
    const uint64_t journal = JournalBase(layout, origin, ring);
    JournalSlotHeader header{};
    if (!CopyFromDevice(
            &header, inc_journal->data + journal +
                         layout.inc_journal.header_offset) ||
        header.magic != kPullDispatchMagic ||
        header.abi_version != kPullDispatchAbiVersion ||
        header.generation != generation || header.sequence != sequence ||
        header.wave != wave || header.ring_slot != ring ||
        header.state != static_cast<uint16_t>(expected_state) ||
        header.status != 0u ||
        header.token_count != RowsFor(options, origin, iteration)) {
        std::cerr << "[FAIL] final journal origin=" << origin << '\n';
        return false;
    }
    std::vector<JournalTokenEntry> tokens;
    std::vector<JournalContributor> contributors;
    if (!CopyVectorFromDevice(
            &tokens, inc_journal->data + journal +
                         layout.inc_journal.tokens_offset,
            header.token_count) ||
        !CopyVectorFromDevice(
            &contributors, inc_journal->data + journal +
                              layout.inc_journal.contributors_offset,
            header.contributor_count))
        return false;
    std::vector<uint32_t> next_row(options.workers, 0u);
    uint64_t contributor_cursor = 0u;
    for (uint32_t token = 0u; token < tokens.size(); ++token) {
        const JournalTokenEntry &entry = tokens[token];
        const std::vector<uint32_t> expected = UniqueDestinations(
            options, origin, token, iteration);
        if (entry.route_key != RouteKey(origin, token) ||
            entry.token_id != TokenId(origin, token, iteration) ||
            entry.owner_rank != origin || entry.owner_row != token ||
            entry.accumulator_index != token ||
            entry.contributors_begin != contributor_cursor ||
            entry.contributors_count != expected.size())
            return false;
        std::vector<uint32_t> actual;
        for (uint32_t i = 0u; i < entry.contributors_count; ++i) {
            const JournalContributor &contributor =
                contributors[entry.contributors_begin + i];
            if (contributor.worker_rank >= options.workers ||
                contributor.destination_row !=
                    next_row[contributor.worker_rank]++)
                return false;
            actual.push_back(contributor.worker_rank);
        }
        std::sort(actual.begin(), actual.end());
        if (actual != expected) return false;
        contributor_cursor += entry.contributors_count;
    }
    return contributor_cursor == contributors.size();
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 11 || argc > 15) {
        std::cerr
            << "usage: " << argv[0]
            << " <workers:2|4> <pe> <ipport> <first_npu>"
               " <rows_csv_per_origin> <hidden> <expert_count> <topk>"
               " <route_seed> <measure>"
               " [delay_rank0_ready_ms=100]"
               " [chain|dispatch|combine] [warmup=0]"
               " [random_gpu|random_expert|sym_k2_balanced|sym_k4_gpu4]\n"
               "example rows_csv: 31,0,7,19\n";
        return 2;
    }
    Options options{};
    options.workers = static_cast<uint32_t>(
        std::strtoul(argv[1], nullptr, 10));
    options.pe = std::atoi(argv[2]);
    ipport = argv[3];
    options.first_npu = std::atoi(argv[4]);
    options.hidden = static_cast<uint32_t>(
        std::strtoul(argv[6], nullptr, 10));
    options.expert_count = static_cast<uint32_t>(
        std::strtoul(argv[7], nullptr, 10));
    options.topk = static_cast<uint32_t>(
        std::strtoul(argv[8], nullptr, 10));
    options.route_seed = std::strtoull(argv[9], nullptr, 10);
    options.waves = static_cast<uint32_t>(
        std::strtoul(argv[10], nullptr, 10));
    if (argc >= 12)
        options.delay_rank0_ms = static_cast<uint32_t>(
            std::strtoul(argv[11], nullptr, 10));
    if (argc >= 13) {
        options.mode_name = argv[12];
        if (!ParseMode(options.mode_name, &options.mode))
            return Fail("mode", 2);
    }
    if (argc >= 14)
        options.warmup = static_cast<uint32_t>(
            std::strtoul(argv[13], nullptr, 10));
    if (argc >= 15) {
        options.route_name = argv[14];
        if (!ParseRouteStyle(options.route_name, &options.route_style))
            return Fail("route style", 2);
    }
    const uint32_t pes = options.workers + 1u;
    const int inc_pe = static_cast<int>(options.workers);
    g_npus = static_cast<int>(pes);
    f_npu = options.first_npu;
    if ((options.workers != 2u && options.workers != 4u) ||
        options.pe < 0 || options.pe >= static_cast<int>(pes) ||
        !ParseRows(argv[5], options.workers, &options.rows) ||
        options.hidden == 0u || options.expert_count == 0u ||
        options.expert_count % options.workers != 0u ||
        options.topk == 0u || options.topk > options.expert_count ||
        options.waves == 0u || options.warmup >
            std::numeric_limits<uint32_t>::max() - options.waves)
        return Fail("arguments", 2);
    const uint32_t max_rows =
        *std::max_element(options.rows.begin(), options.rows.end());
    const uint32_t local_experts =
        options.expert_count / options.workers;
    const uint32_t expert_cycles =
        (options.topk + options.workers - 1u) / options.workers;
    uint64_t max_assignments = 0u;
    const bool route_shape_valid =
        (options.route_style != RouteStyle::SYM_K2_BALANCED ||
         options.topk == 2u) &&
        (options.route_style != RouteStyle::SYM_K4_GPU4 ||
         (options.workers == 4u && options.topk == 4u));
    if (!route_shape_valid || expert_cycles > local_experts ||
        !Mul(max_rows, options.topk, &max_assignments) ||
        max_assignments > std::numeric_limits<uint32_t>::max())
        return Fail("shape/capacity", 2);

    const int device = options.pe + options.first_npu;
    int status = aclInit(nullptr);
    if (status == ACL_SUCCESS) status = aclrtSetDevice(device);
    int64_t live_aiv = 0;
    uint32_t dispatch_aiv_per_origin = 0u;
    if (status == ACL_SUCCESS)
        status = aclrtGetDeviceInfo(
            device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &live_aiv);
    if (status == ACL_SUCCESS && live_aiv > 0 &&
        static_cast<uint64_t>(live_aiv) <=
            std::numeric_limits<uint32_t>::max()) {
        const uint32_t dispatch_half =
            static_cast<uint32_t>(live_aiv) / 2u;
        dispatch_aiv_per_origin = dispatch_half / options.workers;
        if (dispatch_aiv_per_origin < 2u) status = 2;
    } else if (status == ACL_SUCCESS) {
        status = 2;
    }

    SourcePartitionConfig layout_config{};
    layout_config.worker_count = options.workers;
    layout_config.ring_count = kRingCount;
    layout_config.max_source_tokens = std::max<uint32_t>(max_rows, 1u);
    layout_config.max_source_assignments =
        std::max<uint32_t>(static_cast<uint32_t>(max_assignments), 1u);
    layout_config.hidden = options.hidden;
    layout_config.expert_count = options.expert_count;
    layout_config.dispatch_aiv_per_origin = dispatch_aiv_per_origin;
    layout_config.dtype = DataType::BF16;
    SourcePartitionLayout layout{};
    std::string layout_error;
    if (status == ACL_SUCCESS &&
        BuildSourcePartitionLayout(layout_config, &layout, &layout_error) !=
            SourcePartitionLayoutStatus::OK) {
        std::cerr << "layout: " << layout_error << '\n';
        status = 2;
    }
    uint64_t symmetric_heap_bytes = 0u;
    if (status == ACL_SUCCESS &&
        !SymmetricHeapBytes(options, layout, &symmetric_heap_bytes))
        status = 2;

    bool shmem_initialized = false;
    std::vector<aclrtStream> streams(options.workers, nullptr);
    for (uint32_t origin = 0u;
         status == ACL_SUCCESS && origin < options.workers; ++origin)
        status = aclrtCreateStream(&streams[origin]);
    if (status == ACL_SUCCESS) {
        aclshmemx_init_attr_t attr;
        test_set_attr(options.pe, static_cast<int32_t>(pes),
                      symmetric_heap_bytes,
                      ipport, default_flag_uid, &attr);
        status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
        shmem_initialized = status == ACL_SUCCESS;
    }
    if (status == ACL_SUCCESS && options.pe == inc_pe)
        std::cout << "{\"test\":\"source_partition_manifest\""
                  << ",\"workers\":" << options.workers
                  << ",\"hidden\":" << options.hidden
                  << ",\"row_capacity\":" << layout.row_capacity
                  << ",\"assignment_capacity\":"
                  << layout.assignment_capacity
                  << ",\"dispatch_aiv_per_origin\":"
                  << layout.config.dispatch_aiv_per_origin
                  << ",\"symmetric_heap_bytes\":"
                  << symmetric_heap_bytes << "}\n";

    GuardedBuffer source_region, ready_mailbox, dispatch_source_acks;
    GuardedBuffer destination_hidden, destination_rows;
    GuardedBuffer destination_assignments, destination_expert_counts;
    GuardedBuffer destination_completions;
    GuardedBuffer combine_partials, combine_ready, combine_notices;
    GuardedBuffer combine_registrations, combine_acks;
    GuardedBuffer owner_output, owner_completions;
    GuardedBuffer inc_dispatch, inc_journal, inc_combine;
    std::vector<GuardedBuffer *> buffers{
        &source_region, &ready_mailbox, &dispatch_source_acks,
        &destination_hidden, &destination_rows, &destination_assignments,
        &destination_expert_counts, &destination_completions,
        &combine_partials, &combine_ready, &combine_notices,
        &combine_registrations, &combine_acks, &owner_output,
        &owner_completions, &inc_dispatch, &inc_journal, &inc_combine};
    auto Alloc = [&](GuardedBuffer *buffer, uint64_t bytes, bool symmetric,
                     const char *name) {
        if (status == ACL_SUCCESS &&
            !Allocate(buffer, bytes, symmetric, name))
            status = 1;
    };
    uint64_t ready_mailbox_bytes = 0u;
    uint64_t dispatch_ack_bytes = 0u;
    if (!Mul(options.workers, sizeof(Ready), &ready_mailbox_bytes) ||
        !Mul(options.workers, sizeof(SourceConsumed), &dispatch_ack_bytes))
        status = 1;
    if (status == ACL_SUCCESS) {
        // Remote D/C data and publication records are symmetric. INC-private
        // per-origin workspaces and Journals use ordinary HBM.
        Alloc(&source_region, layout.source.worker_stride, true,
              "source_region");
        Alloc(&ready_mailbox, ready_mailbox_bytes, true, "ready_mailbox");
        Alloc(&dispatch_source_acks, dispatch_ack_bytes, true,
              "dispatch_source_acks");
        Alloc(&destination_hidden, layout.destination.hidden_arena_bytes,
              true, "destination_hidden");
        Alloc(&destination_rows, layout.destination.rows_arena_bytes, true,
              "destination_rows");
        Alloc(&destination_assignments,
              layout.destination.assignments_arena_bytes, true,
              "destination_assignments");
        Alloc(&destination_expert_counts,
              layout.destination.expert_counts_arena_bytes, true,
              "destination_expert_counts");
        Alloc(&destination_completions,
              layout.origin_ring_control128_bytes, true,
              "destination_completions");
        Alloc(&combine_partials, layout.combine.partial_arena_bytes, true,
              "combine_partials");
        Alloc(&combine_ready, layout.origin_ring_control128_bytes, true,
              "combine_ready");
        Alloc(&combine_notices, layout.origin_ring_control64_bytes, true,
              "combine_notices");
        Alloc(&combine_registrations,
              static_cast<uint64_t>(options.workers) *
                  sizeof(CombineRegionRegistration),
              false, "combine_registrations");
        Alloc(&combine_acks, layout.origin_ring_control128_bytes, true,
              "combine_acks");
        Alloc(&owner_output, layout.combine.output_arena_bytes, true,
              "owner_output");
        Alloc(&owner_completions, layout.origin_ring_control128_bytes, true,
              "owner_completions");
        Alloc(&inc_dispatch, layout.inc_dispatch.arena_bytes, false,
              "inc_dispatch");
        Alloc(&inc_journal, layout.inc_journal.arena_bytes, false,
              "inc_journal");
        Alloc(&inc_combine, layout.inc_combine.arena_bytes, false,
              "inc_combine");
    }

    std::vector<CombineRegionRegistration> registrations(options.workers);
    const uint64_t combine_slot_stride =
        static_cast<uint64_t>(options.workers) *
        layout.combine.partial_partition_stride;
    for (uint32_t worker = 0u; worker < options.workers; ++worker) {
        CombineRegionRegistration &registration = registrations[worker];
        registration.session_id = kSessionId;
        registration.placement_epoch = kPlacementEpoch;
        registration.source_rank = worker;
        registration.region_id = kCombineRegionBase + worker;
        registration.slot_count = kRingCount;
        registration.region_bytes = layout.combine.partial_arena_bytes;
        registration.slot_stride = combine_slot_stride;
    }
    if (status == ACL_SUCCESS &&
        aclrtMemcpy(combine_registrations.data,
                    registrations.size() * sizeof(registrations[0]),
                    registrations.data(),
                    registrations.size() * sizeof(registrations[0]),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
        status = 1;
    for (GuardedBuffer *buffer : buffers)
        if (status == ACL_SUCCESS && !Fill(buffer, kPoison)) status = 1;
    if (status == ACL_SUCCESS &&
        aclrtMemcpy(combine_registrations.data,
                    registrations.size() * sizeof(registrations[0]),
                    registrations.data(),
                    registrations.size() * sizeof(registrations[0]),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
        status = 1;

    bool correct = status == ACL_SUCCESS;
    std::vector<double> measured_us;
    std::vector<double> measured_directional_gbps;
    uint64_t measured_dispatch_bytes = 0u;
    uint64_t measured_combine_bytes = 0u;
    const uint32_t total_iterations = options.warmup + options.waves;
    for (uint32_t iteration = 0u;
         iteration < total_iterations && status == ACL_SUCCESS; ++iteration) {
        const uint64_t generation = kFirstGeneration + iteration;
        const uint64_t sequence = kFirstSequence + iteration;
        const uint32_t wave = kFirstWave + iteration;
        const uint32_t ring = iteration % kRingCount;

        // Clearing is explicit ring-slot reuse protocol, not address sizing;
        // every allocation and partition base remains session-fixed.
        if (!Fill(&ready_mailbox, 0u) ||
            !Fill(&dispatch_source_acks, 0u) ||
            !Fill(&destination_hidden, kPoison) ||
            !Fill(&destination_rows, kPoison) ||
            !Fill(&destination_assignments, kPoison) ||
            !Fill(&destination_expert_counts, kPoison) ||
            !Fill(&destination_completions, 0u) ||
            !Fill(&combine_partials, kPoison) ||
            !Fill(&combine_ready, 0u) || !Fill(&combine_notices, 0u) ||
            !Fill(&combine_acks, 0u) || !Fill(&owner_output, kPoison) ||
            !Fill(&owner_completions, 0u) || !Fill(&inc_dispatch, 0u) ||
            !Fill(&inc_journal, 0u) || !Fill(&inc_combine, 0u)) {
            status = 1;
            break;
        }

        uint64_t local_packet_bytes = 0u;
        if (options.pe < inc_pe) {
            const uint32_t origin = static_cast<uint32_t>(options.pe);
            SourceInput input = MakeSourceInput(
                options, origin, iteration, generation, sequence, wave,
                ring);
            std::vector<uint8_t> slot;
            SlotHeader header{};
            Ready ready{};
            std::string error;
            if (input.session.session_id == 0u ||
                BuildSlot(input, &slot, &header, &ready, &error) !=
                    Status::OK ||
                slot.size() > layout.source.packet_stride) {
                std::cerr << "BuildSlot origin=" << origin << ": "
                          << error << '\n';
                status = 1;
            } else {
                local_packet_bytes = header.packet_bytes;
                slot.resize(static_cast<size_t>(layout.source.packet_stride),
                            0u);
                const uint64_t slot_offset =
                    static_cast<uint64_t>(ring) *
                    layout.source.packet_stride;
                if (aclrtMemcpy(source_region.data + slot_offset,
                                layout.source.packet_stride, slot.data(),
                                layout.source.packet_stride,
                                ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
                    aclrtMemcpy(ready_mailbox.data +
                                    static_cast<uint64_t>(origin) *
                                        sizeof(Ready),
                                sizeof(Ready), &ready, sizeof(Ready),
                                ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
                    status = 1;
            }
        }
        if (status == ACL_SUCCESS) aclshmem_barrier_all();
        if (status != ACL_SUCCESS) break;

        // Submission order is deliberately independent of READY timing.
        // Rank 0 withholds only its own publication after the INC has queued
        // D0..D(W-1); every other source launches only its own READY kernel.
        auto operation_begin = std::chrono::steady_clock::now();
        auto operation_end = operation_begin;
        std::vector<uint32_t> launch_order(options.workers);
        std::iota(launch_order.begin(), launch_order.end(), 0u);
        for (uint32_t origin : launch_order) {
            if (iteration == 0u && options.pe == 0 && origin == 0u &&
                options.delay_rank0_ms != 0u)
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(options.delay_rank0_ms));
            if (options.pe != inc_pe &&
                static_cast<uint32_t>(options.pe) != origin)
                continue;
            const PartitionedDispatchLaunchArgs dispatch_args =
                MakeDispatchArgs(
                    options, layout, origin, ring, generation, sequence,
                    wave, inc_pe, &source_region, &ready_mailbox,
                    &dispatch_source_acks, &destination_hidden,
                    &destination_rows, &destination_assignments,
                    &destination_expert_counts, &destination_completions,
                    &inc_dispatch, &inc_journal);
            launch_inc_dc_partitioned_dispatch(
                layout.config.dispatch_aiv_per_origin, streams[origin],
                &dispatch_args);
        }

        // Submit all independent D kernels before any C waiter. This avoids
        // runtime launch-slot head-of-line blocking when origin 0 is delayed.
        if (options.mode == Mode::CHAIN && options.pe == inc_pe) {
            for (uint32_t origin : launch_order) {
                const PartitionedCombineLaunchArgs combine_args =
                    MakeCombineArgs(
                        options, layout, origin, ring, generation, sequence,
                        wave, inc_pe, &combine_partials, &combine_ready,
                        &combine_notices, &combine_registrations,
                        &combine_acks, &owner_output, &owner_completions,
                        &inc_dispatch, &inc_journal, &inc_combine);
                launch_inc_dc_partitioned_combine(
                    layout.config.dispatch_aiv_per_origin,
                    streams[origin], &combine_args);
            }
        }

        if (options.mode != Mode::CHAIN) {
            for (uint32_t origin = 0u;
                 status == ACL_SUCCESS && origin < options.workers; ++origin)
                status = aclrtSynchronizeStream(streams[origin]);
            if (options.mode == Mode::DISPATCH)
                operation_end = std::chrono::steady_clock::now();
        }

        std::vector<DestinationCompletion> observed(options.workers);
        if (options.pe < inc_pe) {
            const uint32_t local_worker =
                static_cast<uint32_t>(options.pe);
            std::vector<uint8_t> prepared(options.workers, 0u);
            uint32_t remaining = options.workers;
            const auto deadline = std::chrono::steady_clock::now() +
                std::chrono::seconds(120);
            while (remaining != 0u &&
                   std::chrono::steady_clock::now() < deadline) {
                bool progressed = false;
                for (uint32_t origin = 0u; origin < options.workers;
                     ++origin) {
                    if (prepared[origin] != 0u) continue;
                    uint64_t completion_offset = 0u;
                    if (!OriginControlOffset(
                            layout, origin, ring, local_worker,
                            sizeof(DestinationCompletion),
                            &completion_offset)) {
                        status = 1;
                        break;
                    }
                    DestinationCompletion completion{};
                    if (!CopyFromDevice(
                            &completion,
                            destination_completions.data +
                                completion_offset)) {
                        status = 1;
                        break;
                    }
                    if (completion.publication == 0u ||
                        completion.generation != generation ||
                        completion.sequence != sequence)
                        continue;
                    if (!CompletionMatches(
                            completion, local_worker, origin, ring,
                            generation, sequence, wave)) {
                        status = 1;
                        break;
                    }
                    observed[origin] = completion;
                    if (options.mode == Mode::CHAIN) {
                        if (!PrepareCombinePartial(
                                options, layout, local_worker, origin,
                                iteration, ring, generation, sequence, wave,
                                completion, &destination_rows,
                                &destination_assignments,
                                &destination_hidden, &combine_partials,
                                &combine_ready, &combine_notices)) {
                            status = 1;
                            break;
                        }
                        const PartitionedCombineLaunchArgs combine_args =
                            MakeCombineArgs(
                                options, layout, origin, ring, generation,
                                sequence, wave, inc_pe, &combine_partials,
                                &combine_ready, &combine_notices,
                                &combine_registrations, &combine_acks,
                                &owner_output, &owner_completions,
                                &inc_dispatch, &inc_journal, &inc_combine);
                        launch_inc_dc_partitioned_combine(
                            layout.config.dispatch_aiv_per_origin,
                            streams[origin], &combine_args);
                    }
                    prepared[origin] = 1u;
                    --remaining;
                    progressed = true;
                }
                if (status != ACL_SUCCESS) break;
                if (!progressed)
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(1));
            }
            if (remaining != 0u) status = 1;
            if (status == ACL_SUCCESS && options.mode != Mode::CHAIN)
                for (uint32_t origin = 0u;
                     origin < options.workers; ++origin)
                    if (!PrepareCombinePartial(
                            options, layout, local_worker, origin, iteration,
                            ring, generation, sequence, wave,
                            observed[origin], &destination_rows,
                            &destination_assignments, &destination_hidden,
                            &combine_partials, &combine_ready,
                            &combine_notices)) {
                        status = 1;
                        break;
                    }
            if (status == ACL_SUCCESS &&
                !ValidateDispatchAck(
                    local_worker, ring, generation, sequence, wave,
                    local_packet_bytes,
                    observed[local_worker].dispatch_cookie,
                    &dispatch_source_acks))
                status = 1;
        }

        if (options.mode == Mode::COMBINE) {
            // D, actual metadata validation, and the local-FFN placeholder
            // are complete before the C-only communication timer starts.
            // Opt-in negative qualification only; never used by performance suites.
            const char *fault = std::getenv("INC_DC_PARTITION_TEST_BAD_OWNER");
            if (status == ACL_SUCCESS && options.pe == inc_pe &&
                iteration == 0u && fault != nullptr && std::strcmp(fault, "1") == 0 &&
                RowsFor(options, 0u, iteration) != 0u) {
                const auto probe_args = MakeCombineArgs(
                    options, layout, 0u, ring, generation, sequence, wave, inc_pe,
                    &combine_partials, &combine_ready, &combine_notices,
                    &combine_registrations, &combine_acks, &owner_output,
                    &owner_completions, &inc_dispatch, &inc_journal, &inc_combine);
                auto *entry = probe_args.journal_tokens +
                    (RowsFor(options, 0u, iteration) / 2u) * sizeof(JournalTokenEntry);
                JournalTokenEntry token{};
                if (!CopyFromDevice(&token, entry)) status = 1;
                token.owner_rank = options.workers;
                if (!CopyToDevice(entry, token)) status = 1;
                std::cerr << "[INJECTED] invalid journal owner at middle token\n";
            }
            if (status == ACL_SUCCESS) aclshmem_barrier_all();
            operation_begin = std::chrono::steady_clock::now();
            for (uint32_t origin : launch_order) {
                const PartitionedCombineLaunchArgs combine_args =
                    MakeCombineArgs(
                        options, layout, origin, ring, generation, sequence,
                        wave, inc_pe, &combine_partials, &combine_ready,
                        &combine_notices, &combine_registrations,
                        &combine_acks, &owner_output, &owner_completions,
                        &inc_dispatch, &inc_journal, &inc_combine);
                launch_inc_dc_partitioned_combine(
                    layout.config.dispatch_aiv_per_origin, streams[origin],
                    &combine_args);
            }
        }

        if (options.mode != Mode::DISPATCH)
            for (uint32_t origin = 0u;
                 status == ACL_SUCCESS && origin < options.workers; ++origin)
                status = aclrtSynchronizeStream(streams[origin]);
        if (options.mode != Mode::DISPATCH)
            operation_end = std::chrono::steady_clock::now();
        if (options.pe == inc_pe && options.mode == Mode::COMBINE &&
            std::getenv("INC_DC_PARTITION_TEST_BAD_OWNER") != nullptr) {
            PartitionedCombineTimeline fault_timeline{};
            if (CopyFromDevice(&fault_timeline, inc_combine.data +
                    IncCombineBase(layout, 0u, ring) + layout.inc_combine.timeline_offset))
                std::cerr << "[FAULT_STATUS] " << fault_timeline.status << '\n';
        }
        if (status == ACL_SUCCESS) aclshmem_barrier_all();
        bool wave_correct = status == ACL_SUCCESS;
        std::vector<uint64_t> dispatch_ready_cycles(options.workers, 0u);
        std::vector<uint64_t> dispatch_completion_cycles(
            options.workers, 0u);
        std::vector<PartitionedCombineTimeline> combine_timelines(options.workers);
        std::vector<PullTimeline> dispatch_timelines(options.workers);
        bool early_independence_checked = false;
        bool early_independence_passed = false;
        if (wave_correct && options.pe < inc_pe &&
            options.mode != Mode::DISPATCH)
            wave_correct = ValidateOwnerOutput(
                options, layout, static_cast<uint32_t>(options.pe),
                iteration, ring, generation, sequence, wave, &owner_output,
                &owner_completions);
        if (wave_correct && options.pe == inc_pe) {
            const JournalSlotState expected_state =
                options.mode == Mode::DISPATCH
                ? JournalSlotState::DISPATCH_SEALED
                : JournalSlotState::COMPLETE;
            for (uint32_t origin = 0u;
                 origin < options.workers && wave_correct; ++origin)
                wave_correct = ValidateJournal(
                    options, layout, origin, iteration, ring, generation,
                    sequence, wave, expected_state, &inc_journal);
            for (uint32_t origin = 0u;
                 origin < options.workers && wave_correct; ++origin) {
                PullTimeline timeline{};
                const uint64_t base =
                    IncDispatchBase(layout, origin, ring);
                wave_correct = CopyFromDevice(
                    &timeline,
                    inc_dispatch.data + base +
                        layout.inc_dispatch.timeline_offset);
                dispatch_ready_cycles[origin] = timeline.all_ready;
                dispatch_completion_cycles[origin] =
                    timeline.destination_completions_done;
                dispatch_timelines[origin] = timeline;
                if (wave_correct && options.mode != Mode::DISPATCH)
                    wave_correct = CopyFromDevice(
                        &combine_timelines[origin],
                        inc_combine.data + IncCombineBase(layout, origin, ring) +
                            layout.inc_combine.timeline_offset);
            }
            if (iteration == 0u && options.delay_rank0_ms != 0u &&
                options.workers > 1u) {
                bool compared_nonempty_origin = false;
                early_independence_passed =
                    dispatch_ready_cycles[0] != 0u;
                for (uint32_t origin = 1u;
                     origin < options.workers; ++origin) {
                    if (RowsFor(options, origin, iteration) == 0u) continue;
                    compared_nonempty_origin = true;
                    const bool independent =
                        dispatch_completion_cycles[origin] != 0u &&
                        dispatch_completion_cycles[origin] <
                            dispatch_ready_cycles[0];
                    early_independence_passed =
                        early_independence_passed && independent;
                    if (!independent)
                        std::cerr
                            << "[FAIL] delayed origin 0 blocked origin "
                            << origin << " completion="
                            << dispatch_completion_cycles[origin]
                            << " origin0_ready="
                            << dispatch_ready_cycles[0] << '\n';
                }
                early_independence_checked = compared_nonempty_origin;
                if (early_independence_checked)
                    wave_correct =
                        wave_correct && early_independence_passed;
                else
                    early_independence_passed = false;
            }
        }
        uint64_t dispatch_fanout_bytes = 0u;
        uint64_t combine_ingress_bytes = 0u;
        if (wave_correct && options.pe == inc_pe)
            wave_correct = ActualCommunicationBytes(
                options, layout, ring, &inc_dispatch,
                &dispatch_fanout_bytes, &combine_ingress_bytes);
        for (GuardedBuffer *buffer : buffers)
            wave_correct = GuardsValid(*buffer) && wave_correct;
        correct = correct && wave_correct;
        const bool warmup = iteration < options.warmup;
        const double host_us =
            std::chrono::duration<double, std::micro>(
                operation_end - operation_begin).count();
        const uint64_t directional_bytes = options.mode == Mode::DISPATCH
            ? dispatch_fanout_bytes
            : options.mode == Mode::COMBINE ? combine_ingress_bytes : 0u;
        const double directional_gbps =
            host_us == 0.0 || options.mode == Mode::CHAIN ? 0.0 :
            static_cast<double>(directional_bytes) /
                (host_us * 1e-6) / 1e9;
        if (options.pe == inc_pe) {
            std::cout << "{\"test\":\"source_partition_device_e2e\""
                      << ",\"iteration\":" << iteration
                      << ",\"ring\":" << ring
                      << ",\"mode\":\"" << options.mode_name << "\""
                      << ",\"route\":\"" << options.route_name << "\""
                      << ",\"warmup\":"
                      << (warmup ? "true" : "false")
                      << ",\"host_us\":" << host_us
                      << ",\"dispatch_fanout_bytes\":"
                      << dispatch_fanout_bytes
                      << ",\"combine_ingress_bytes\":"
                      << combine_ingress_bytes
                      << ",\"route_seed\":" << options.route_seed
                      << ",\"route_iteration\":" << iteration
                      << ",\"dispatch_ready_cycles\":[";
            for (uint32_t origin = 0u; origin < options.workers; ++origin) {
                if (origin != 0u) std::cout << ',';
                std::cout << dispatch_ready_cycles[origin];
            }
            std::cout << "]"
                      << ",\"dispatch_completion_cycles\":[";
            for (uint32_t origin = 0u; origin < options.workers; ++origin) {
                if (origin != 0u) std::cout << ',';
                std::cout << dispatch_completion_cycles[origin];
            }
            std::cout << "]" << ",\"dispatch_phase_cycles\":[";
            for (uint32_t origin = 0u; origin < options.workers; ++origin) {
                if (origin != 0u) std::cout << ',';
                const auto &t = dispatch_timelines[origin];
                std::cout << "{\"start\":" << t.kernel_start
                          << ",\"headers\":" << t.headers_pulled
                          << ",\"metadata_done\":" << t.metadata_parse_done
                          << ",\"relay_begin\":" << t.hidden_get_begin
                          << ",\"relay_done\":" << t.hidden_get_done
                          << ",\"done\":" << t.kernel_done << '}';
            }
            std::cout << "]" << ",\"combine_phase_cycles\":[";
            for (uint32_t origin = 0u; origin < options.workers; ++origin) {
                if (origin != 0u) std::cout << ',';
                const auto &t = combine_timelines[origin];
                // Absolute device cycles; joined_payload is a grid completion
                // timestamp, not the exact final GET or vector instruction.
                std::cout << "{\"start\":" << t.kernel_start
                          << ",\"validated\":" << t.journal_validated
                          << ",\"ready\":" << t.all_required_ready
                          << ",\"joined_payload\":" << t.last_owner_put
                          << ",\"done\":" << t.kernel_done << '}';
            }
            std::cout << "]"
                      << ",\"early_independence_checked\":"
                      << (early_independence_checked ? "true" : "false")
                      << ",\"early_independence_passed\":"
                      << (early_independence_passed ? "true" : "false");
            if (options.mode != Mode::CHAIN)
                std::cout << ",\"directional_gb_s\":"
                          << directional_gbps;
            std::cout
                      << ",\"correct\":"
                      << (wave_correct ? "true" : "false") << "}\n";
            if (!warmup && wave_correct) {
                measured_us.push_back(host_us);
                if (options.mode != Mode::CHAIN)
                    measured_directional_gbps.push_back(directional_gbps);
                if (!Add(measured_dispatch_bytes, dispatch_fanout_bytes,
                         &measured_dispatch_bytes) ||
                    !Add(measured_combine_bytes, combine_ingress_bytes,
                         &measured_combine_bytes))
                    correct = false;
            }
        }
        if (status == ACL_SUCCESS) aclshmem_barrier_all();
    }

    if (correct && options.pe == inc_pe && !measured_us.empty()) {
        const double total_us = std::accumulate(
            measured_us.begin(), measured_us.end(), 0.0);
        const double mean_us = total_us / measured_us.size();
        double latency_variance = 0.0;
        for (double sample : measured_us)
            latency_variance += (sample - mean_us) * (sample - mean_us);
        latency_variance /= measured_us.size();
        std::cout << "{\"test\":\"source_partition_device_e2e_summary\""
                  << ",\"mode\":\"" << options.mode_name << "\""
                  << ",\"route\":\"" << options.route_name << "\""
                  << ",\"warmup\":" << options.warmup
                  << ",\"measure\":" << measured_us.size()
                  << ",\"min_us\":"
                  << *std::min_element(measured_us.begin(),
                                       measured_us.end())
                  << ",\"mean_us\":" << mean_us
                  << ",\"max_us\":"
                  << *std::max_element(measured_us.begin(),
                                       measured_us.end())
                  << ",\"latency_cv_percent\":"
                  << (mean_us == 0.0 ? 0.0 :
                      std::sqrt(latency_variance) / mean_us * 100.0)
                  << ",\"dispatch_fanout_bytes\":"
                  << measured_dispatch_bytes
                  << ",\"combine_ingress_bytes\":"
                  << measured_combine_bytes;
        if (options.mode != Mode::CHAIN &&
            !measured_directional_gbps.empty()) {
            const double mean_gbps = std::accumulate(
                measured_directional_gbps.begin(),
                measured_directional_gbps.end(), 0.0) /
                measured_directional_gbps.size();
            double variance = 0.0;
            for (double sample : measured_directional_gbps)
                variance += (sample - mean_gbps) * (sample - mean_gbps);
            variance /= measured_directional_gbps.size();
            std::cout << ",\"directional_min_gb_s\":"
                      << *std::min_element(
                             measured_directional_gbps.begin(),
                             measured_directional_gbps.end())
                      << ",\"directional_mean_gb_s\":" << mean_gbps
                      << ",\"directional_max_gb_s\":"
                      << *std::max_element(
                             measured_directional_gbps.begin(),
                             measured_directional_gbps.end())
                      << ",\"directional_cv_percent\":"
                      << (mean_gbps == 0.0 ? 0.0 :
                          std::sqrt(variance) / mean_gbps * 100.0);
        }
        std::cout << ",\"correct\":true}\n";
    }

    for (auto it = buffers.rbegin(); it != buffers.rend(); ++it)
        Release(*it);
    if (shmem_initialized) aclshmem_finalize();
    for (aclrtStream stream : streams)
        if (stream != nullptr) aclrtDestroyStream(stream);
    if (device >= 0) aclrtResetDevice(device);
    aclFinalize();
    if (!correct || status != ACL_SUCCESS)
        return Fail("source-partition D->C device E2E", status);
    std::cout << "[PASS] pe=" << options.pe
              << " workers=" << options.workers
              << " waves=" << options.waves << '\n';
    return 0;
}
