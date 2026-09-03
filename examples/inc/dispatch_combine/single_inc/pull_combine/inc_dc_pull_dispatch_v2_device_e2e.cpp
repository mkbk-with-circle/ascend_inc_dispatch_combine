#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "shmem.h"
#include "utils.h"

#include "inc_dc_pull_dispatch_v2.h"

using namespace inc::dc::pull_v2;

extern "C" void launch_inc_dc_pull_dispatch_v2_device(
    uint32_t block_dim, void *stream, uint8_t *source_region,
    uint8_t *ready_mailbox, uint8_t *inc_slots, uint8_t *source_acks,
    uint8_t *destination_hidden, uint8_t *destination_rows,
    uint8_t *destination_assignments, uint8_t *destination_expert_counts,
    uint8_t *destination_completions, uint8_t *inc_destination_rows,
    uint8_t *inc_destination_assignments, uint8_t *journal_header,
    uint8_t *journal_tokens, uint8_t *journal_contributors,
    uint8_t *journal_assignments, uint8_t *row_map,
    uint8_t *source_token_prefix, uint8_t *source_destination_prefix,
    uint8_t *destination_row_counts,
    uint8_t *destination_assignment_counts, uint8_t *expert_counts,
    uint8_t *parser_scratch, uint8_t *status_line, uint64_t ffts_addr,
    uint64_t session_id,
    uint64_t placement_epoch, uint64_t generation, uint64_t sequence,
    uint64_t source_slot_stride, uint64_t destination_hidden_slot_stride,
    uint64_t destination_rows_slot_stride,
    uint64_t destination_assignments_slot_stride,
    uint64_t destination_expert_counts_slot_stride,
    uint64_t journal_token_capacity, uint64_t journal_contributor_capacity,
    uint64_t journal_assignment_capacity,
    uint64_t destination_row_capacity,
    uint64_t destination_assignment_capacity,
    uint64_t inc_destination_rows_stride_bytes,
    uint64_t inc_destination_assignments_stride_bytes,
    uint64_t row_map_capacity_entries,
    uint64_t source_destination_prefix_capacity_entries,
    uint64_t expert_counts_capacity_entries,
    uint64_t parser_scratch_capacity_entries, uint32_t worker_count,
    uint32_t expert_count, uint32_t hidden, uint32_t dtype, int32_t inc_pe,
    uint32_t region_id, uint32_t wave, uint32_t ring_slot,
    uint32_t slot_count, uint32_t channels_per_source, uint64_t spin_cap);

int g_npus = 5;
const char *ipport = "tcp://127.0.0.1:28798";
int f_pe = 0;
int f_npu = 0;
aclshmemx_uniqueid_t default_flag_uid;

namespace {

constexpr uint64_t kSessionId = 0x5044324554450001ull;
constexpr uint64_t kPlacementEpoch = 3u;
constexpr uint64_t kFirstGeneration = 1001u;
constexpr uint64_t kFirstSequence = 7001u;
constexpr uint32_t kFirstWave = 11u;
constexpr uint32_t kRegionId = 7u;
constexpr uint32_t kRingSlots = 2u;
constexpr uint64_t kSpinCap = 1000000000ull;
constexpr uint64_t kFaultSpinCap = 2000000ull;
constexpr uint64_t kGuardBytes = 64u;
constexpr uint8_t kHeadGuard = 0xa5u;
constexpr uint8_t kTailGuard = 0x5au;
constexpr uint8_t kPoison = 0xc7u;
constexpr uint64_t kValidationChunk = 4ull << 20;
constexpr double kSystemCycleUs = 0.02; // GetSystemCycle is 50 MHz on 910B.

enum class Workload {
    SYM_DENSE,
    SYM_K2_BALANCED,
    SYM_K1_RR,
    HOTSPOT,
    RAGGED,
};

struct Options {
    uint32_t workers = 0u;
    int pe = -1;
    int first_npu = 0;
    uint64_t payload_bytes = 0u;
    Workload workload = Workload::SYM_K2_BALANCED;
    const char *workload_name = nullptr;
    uint32_t hidden = 0u;
    uint32_t expert_count = 0u;
    uint32_t channels = 0u;
    uint32_t warmup = 0u;
    uint32_t measure = 0u;
    uint64_t seed = 0u;
    uint32_t fault = 0u;
};

struct WaveOracle {
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint32_t wave = 0u;
    uint32_t ring_slot = 0u;
    std::vector<ParsedSource> sources;
    std::vector<uint64_t> source_packet_bytes;
    std::vector<uint32_t> source_tokens;
    CompiledLayout layout;
    std::vector<uint32_t> source_token_prefix;
    std::vector<uint32_t> source_destination_prefix;
    std::vector<uint32_t> row_map;
    std::vector<JournalTokenEntry> fixed_journal_tokens;
    std::vector<JournalContributor> fixed_contributors;
    uint64_t ingress_hidden_bytes = 0u;
    uint64_t egress_hidden_bytes = 0u;
    uint64_t logical_bytes = 0u;
};

struct GuardedBuffer {
    uint8_t *allocation = nullptr;
    uint8_t *data = nullptr;
    uint64_t bytes = 0u;
    uint64_t allocated_bytes = 0u;
    bool symmetric = false;
    const char *name = nullptr;
};

uint64_t Align64(uint64_t value)
{
    if (value > std::numeric_limits<uint64_t>::max() - 63u) return 0u;
    return (value + 63u) / 64u * 64u;
}

bool Mul(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr || (a != 0u &&
        b > std::numeric_limits<uint64_t>::max() / a)) return false;
    *out = a * b;
    return true;
}

bool Add(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr || b > std::numeric_limits<uint64_t>::max() - a)
        return false;
    *out = a + b;
    return true;
}

bool AlignEntries16(uint64_t value, uint64_t *out)
{
    uint64_t expanded = 0u;
    if (out == nullptr || !Add(value, 15u, &expanded)) return false;
    *out = expanded & ~15ull;
    return true;
}

bool ParserScratchEntries(uint32_t blocks, uint32_t workers,
                          uint32_t experts, uint64_t *out)
{
    if (out == nullptr || workers == 0u || experts == 0u) return false;
    const uint32_t active = (blocks / workers) * workers;
    if (active == 0u) return false;
    uint64_t source_extent = 0u;
    uint64_t block_extent = 0u;
    uint64_t row_stride = 0u;
    uint64_t row_extent = 0u;
    uint64_t expert_values = 0u;
    uint64_t expert_stride = 0u;
    uint64_t expert_extent = 0u;
    uint64_t entries = 0u;
    return Mul(workers, 16u, &source_extent) &&
        Mul(active, 16u, &block_extent) &&
        AlignEntries16(workers, &row_stride) &&
        Mul(active, row_stride, &row_extent) &&
        Mul(workers, experts, &expert_values) &&
        AlignEntries16(expert_values, &expert_stride) &&
        Mul(active, expert_stride, &expert_extent) &&
        Add(entries, source_extent, &entries) &&
        Add(entries, source_extent, &entries) &&
        Add(entries, static_cast<uint64_t>(workers) + 1u, &entries) &&
        AlignEntries16(entries, &entries) &&
        Add(entries, block_extent, &entries) &&
        Add(entries, block_extent, &entries) &&
        Add(entries, row_extent, &entries) &&
        Add(entries, row_extent, &entries) &&
        Add(entries, expert_extent, &entries) &&
        Add(entries, source_extent, out);
}

uint64_t Mix(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

uint8_t HiddenByte(uint64_t seed, uint32_t source, uint32_t token,
                   uint64_t byte)
{
    const uint64_t lane = byte & 255u;
    const uint64_t block = byte >> 8u;
    return static_cast<uint8_t>((Mix(seed ^
        (static_cast<uint64_t>(source) << 48u) ^
        (static_cast<uint64_t>(token) << 16u) ^ block) + lane) % 251u);
}

uint64_t AckPublication(uint64_t generation, uint64_t sequence,
                        uint32_t source, uint32_t status)
{
    uint64_t value = generation ^ (sequence << 1u) ^
        (static_cast<uint64_t>(source) << 48u) ^
        (static_cast<uint64_t>(status) << 24u) ^ 0xa55aa55aa55aa55aull;
    return value == 0u ? 1u : value;
}

uint64_t CompletionPublication(uint64_t generation, uint64_t sequence,
                               uint32_t destination, uint32_t status)
{
    uint64_t value = generation ^ (sequence << 3u) ^
        (static_cast<uint64_t>(destination) << 44u) ^
        (static_cast<uint64_t>(status) << 20u) ^ 0x5aa55aa55aa55aa5ull;
    return value == 0u ? 1u : value;
}

int Fail(const char *step, int status)
{
    std::cerr << "[FAIL] " << step << " status=" << status << '\n';
    return status == 0 ? 1 : status;
}

bool ParseWorkload(const char *text, Workload *workload)
{
    if (std::strcmp(text, "sym_dense") == 0)
        *workload = Workload::SYM_DENSE;
    else if (std::strcmp(text, "sym_k2_balanced") == 0)
        *workload = Workload::SYM_K2_BALANCED;
    else if (std::strcmp(text, "sym_k1_rr") == 0)
        *workload = Workload::SYM_K1_RR;
    else if (std::strcmp(text, "hotspot") == 0)
        *workload = Workload::HOTSPOT;
    else if (std::strcmp(text, "ragged") == 0)
        *workload = Workload::RAGGED;
    else
        return false;
    return true;
}

uint32_t TokensForSource(const Options &o, uint32_t source,
                         uint64_t row_bytes)
{
    if (o.payload_bytes == 0u) return 0u;
    uint64_t base = (o.payload_bytes + row_bytes - 1u) / row_bytes;
    if (o.workload == Workload::RAGGED) {
        // 100%, 75%, 50%, 25%, ...; never silently turn nonzero input into
        // an empty source.  This deliberately exercises unequal READY sizes.
        static constexpr uint32_t numerator[4]{4u, 3u, 2u, 1u};
        base = (base * numerator[source & 3u] + 3u) / 4u;
        base = std::max<uint64_t>(base, 1u);
    }
    return base > std::numeric_limits<uint32_t>::max()
        ? 0u : static_cast<uint32_t>(base);
}

SourceInput MakeInput(const Options &o, uint32_t source,
                      uint64_t generation, uint64_t sequence,
                      uint32_t wave, uint32_t ring_slot)
{
    SourceInput input{};
    input.session.session_id = kSessionId;
    input.session.placement_epoch = kPlacementEpoch;
    input.session.worker_count = o.workers;
    input.session.expert_count = o.expert_count;
    input.session.hidden = o.hidden;
    input.session.dtype = DataType::BF16;
    input.session.ring_slots = kRingSlots;
    input.generation = generation;
    input.sequence = sequence;
    input.wave = wave;
    input.source_rank = source;
    input.source_region_id = kRegionId;
    input.ring_slot = static_cast<uint16_t>(ring_slot);
    const uint64_t row_bytes = static_cast<uint64_t>(o.hidden) * 2u;
    const uint32_t tokens = TokensForSource(o, source, row_bytes);
    input.assignment_offsets.reserve(static_cast<size_t>(tokens) + 1u);
    input.assignment_offsets.push_back(0u);
    input.token_ids.reserve(tokens);

    for (uint32_t token = 0u; token < tokens; ++token) {
        input.token_ids.push_back((static_cast<uint64_t>(source) << 48u) |
                                  (static_cast<uint64_t>(wave) << 32u) |
                                  token);
        const uint64_t random = Mix(o.seed ^
            (static_cast<uint64_t>(source) << 32u) ^ token);
        std::vector<uint32_t> destinations;
        switch (o.workload) {
            case Workload::SYM_DENSE:
                for (uint32_t destination = 0u;
                     destination < o.workers; ++destination)
                    destinations.push_back(destination);
                break;
            case Workload::SYM_K2_BALANCED:
                destinations.push_back(source % o.workers);
                destinations.push_back((source + 1u) % o.workers);
                break;
            case Workload::SYM_K1_RR:
                destinations.push_back(
                    (source + token + static_cast<uint32_t>(o.seed)) %
                    o.workers);
                break;
            case Workload::HOTSPOT:
                destinations.push_back(static_cast<uint32_t>(o.seed %
                                                             o.workers));
                destinations.push_back(static_cast<uint32_t>(o.seed %
                                                             o.workers));
                break;
            case Workload::RAGGED: {
                const uint32_t topk = 1u + static_cast<uint32_t>(
                    random % std::min<uint32_t>(8u, o.workers * 2u));
                for (uint32_t k = 0u; k < topk; ++k)
                    destinations.push_back(static_cast<uint32_t>(
                        Mix(random + k) % o.workers));
                break;
            }
        }
        for (uint32_t ordinal = 0u; ordinal < destinations.size();
             ++ordinal) {
            AssignmentRecord assignment{};
            assignment.destination_rank = destinations[ordinal];
            assignment.expert_id = static_cast<uint32_t>(Mix(
                random + ordinal * 17u) % o.expert_count);
            assignment.ordinal = ordinal;
            assignment.weight = static_cast<float>(ordinal + 1u) /
                static_cast<float>(destinations.size() + 1u);
            input.assignments.push_back(assignment);
        }
        input.assignment_offsets.push_back(
            static_cast<uint32_t>(input.assignments.size()));
    }

    uint64_t hidden_bytes = 0u;
    if (!Mul(tokens, row_bytes, &hidden_bytes) ||
        hidden_bytes > std::numeric_limits<size_t>::max())
        return SourceInput{};
    input.hidden_payload.resize(static_cast<size_t>(hidden_bytes));
    for (uint32_t token = 0u; token < tokens; ++token) {
        uint8_t *row = input.hidden_payload.data() +
            static_cast<uint64_t>(token) * row_bytes;
        for (uint64_t byte = 0u; byte < row_bytes; ++byte)
            row[byte] = HiddenByte(o.seed, source, token, byte);
    }
    return input;
}

bool EqualBytes(const void *lhs, const void *rhs, uint64_t bytes)
{
    return bytes == 0u || std::memcmp(lhs, rhs,
                                      static_cast<size_t>(bytes)) == 0;
}

template <typename T>
bool CopyVector(std::vector<T> *host, const uint8_t *device, uint64_t count)
{
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) return false;
    host->resize(static_cast<size_t>(count));
    return count == 0u || aclrtMemcpy(
        host->data(), count * sizeof(T), device, count * sizeof(T),
        ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

template <typename T>
bool SameVector(const std::vector<T> &actual,
                const std::vector<T> &expected)
{
    return actual.size() == expected.size() &&
        EqualBytes(actual.data(), expected.data(),
                   actual.size() * sizeof(T));
}

bool Allocate(GuardedBuffer *buffer, uint64_t bytes, bool symmetric,
              const char *name)
{
    if (buffer == nullptr) return false;
    const uint64_t payload = std::max<uint64_t>(bytes, 64u);
    uint64_t total = 0u;
    if (!Add(payload, kGuardBytes * 2u, &total)) return false;
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
    if (buffer->symmetric) aclshmem_free(buffer->allocation);
    else aclrtFree(buffer->allocation);
    *buffer = GuardedBuffer{};
}

bool GuardsValid(const GuardedBuffer &buffer)
{
    std::vector<uint8_t> head(kGuardBytes);
    std::vector<uint8_t> tail(kGuardBytes);
    if (aclrtMemcpy(head.data(), kGuardBytes, buffer.allocation, kGuardBytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(tail.data(), kGuardBytes,
                    buffer.data + buffer.allocated_bytes, kGuardBytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
        return false;
    const bool good = std::all_of(head.begin(), head.end(),
        [](uint8_t value) { return value == kHeadGuard; }) &&
        std::all_of(tail.begin(), tail.end(),
        [](uint8_t value) { return value == kTailGuard; });
    if (!good) std::cerr << "[FAIL] guard changed: " << buffer.name << '\n';
    return good;
}

bool Fill(GuardedBuffer *buffer, uint8_t value)
{
    return buffer != nullptr && aclrtMemset(
        buffer->data, buffer->allocated_bytes, value,
        buffer->allocated_bytes) == ACL_SUCCESS;
}

bool BuildOracle(const Options &o, uint64_t generation, uint64_t sequence,
                 uint32_t wave, uint32_t ring_slot, int local_source,
                 WaveOracle *oracle, std::vector<uint8_t> *local_slot,
                 Ready *local_ready, uint64_t *source_stride)
{
    if (oracle == nullptr || local_slot == nullptr || local_ready == nullptr ||
        source_stride == nullptr) return false;
    WaveOracle built{};
    built.generation = generation;
    built.sequence = sequence;
    built.wave = wave;
    built.ring_slot = ring_slot;
    built.sources.resize(o.workers);
    built.source_packet_bytes.resize(o.workers);
    built.source_tokens.resize(o.workers);
    uint64_t max_slot = 64u;
    std::string error;
    for (uint32_t source = 0u; source < o.workers; ++source) {
        SourceInput input = MakeInput(o, source, generation, sequence,
                                      wave, ring_slot);
        if (input.session.session_id == 0u) return false;
        std::vector<uint8_t> slot;
        SlotHeader header{};
        Ready ready{};
        if (BuildSlot(input, &slot, &header, &ready, &error) != Status::OK) {
            std::cerr << "BuildSlot source=" << source << ": " << error
                      << '\n';
            return false;
        }
        const uint64_t metadata_end = header.assignments_offset +
            static_cast<uint64_t>(header.assignment_count) *
                sizeof(AssignmentRecord);
        if (header.hidden_offset % kPullDispatchAlignment != 0u ||
            metadata_end > header.hidden_offset ||
            !std::all_of(slot.begin() + static_cast<size_t>(metadata_end),
                         slot.begin() +
                             static_cast<size_t>(header.hidden_offset),
                         [](uint8_t byte) { return byte == 0u; })) {
            std::cerr << "BuildSlot source=" << source
                      << ": noncanonical hidden padding\n";
            return false;
        }
        RegionRegistration registration{};
        registration.session_id = kSessionId;
        registration.placement_epoch = kPlacementEpoch;
        registration.source_rank = source;
        registration.region_id = kRegionId;
        registration.slot_count = kRingSlots;
        registration.region_bytes = header.packet_bytes * kRingSlots;
        registration.slot_stride = header.packet_bytes;
        if (ParseReadyAndSlot(registration, input.session, ready, slot.data(),
                              slot.size(), &built.sources[source], &error) !=
            Status::OK) {
            std::cerr << "ParseReadyAndSlot source=" << source << ": "
                      << error << '\n';
            return false;
        }
        // CompileLayout does not consume hidden bytes.  Drop this copy so a
        // 128-MiB/source oracle remains bounded in host memory.
        built.sources[source].hidden_payload.clear();
        built.sources[source].hidden_payload.shrink_to_fit();
        built.source_packet_bytes[source] = header.packet_bytes;
        built.source_tokens[source] = header.token_count;
        built.ingress_hidden_bytes += input.hidden_payload.size();
        max_slot = std::max<uint64_t>(max_slot, header.packet_bytes);
        if (local_source >= 0 && source == static_cast<uint32_t>(local_source)) {
            *local_slot = std::move(slot);
            *local_ready = ready;
        }
    }
    *source_stride = Align64(max_slot);
    if (local_source >= 0) local_slot->resize(*source_stride, 0u);

    std::vector<uint32_t> arrival(o.workers);
    std::iota(arrival.begin(), arrival.end(), 0u);
    uint64_t total_tokens = std::accumulate(
        built.source_tokens.begin(), built.source_tokens.end(), uint64_t{0});
    uint64_t total_assignments = 0u;
    for (const ParsedSource &source : built.sources)
        total_assignments += source.assignments.size();
    LayoutConfig config{};
    config.worker_count = o.workers;
    config.expert_count = o.expert_count;
    config.destination_row_capacity = std::max<uint64_t>(total_tokens, 1u);
    config.destination_assignment_capacity =
        std::max<uint64_t>(total_assignments, 1u);
    if (CompileLayout(built.sources, arrival, config, &built.layout,
                      &error) != Status::OK) {
        std::cerr << "CompileLayout: " << error << '\n';
        return false;
    }

    built.source_token_prefix.assign(o.workers + 1u, 0u);
    for (uint32_t source = 0u; source < o.workers; ++source)
        built.source_token_prefix[source + 1u] =
            built.source_token_prefix[source] + built.source_tokens[source];
    built.source_destination_prefix.assign(
        static_cast<size_t>(o.workers + 1u) * o.workers, 0u);
    built.row_map.assign(static_cast<size_t>(total_tokens) * o.workers,
                         std::numeric_limits<uint32_t>::max());
    std::vector<uint32_t> running(o.workers, 0u);
    for (uint32_t source = 0u; source < o.workers; ++source) {
        for (uint32_t destination = 0u; destination < o.workers;
             ++destination)
            built.source_destination_prefix[
                static_cast<size_t>(source) * o.workers + destination] =
                running[destination];
        const uint32_t begin = built.source_token_prefix[source];
        const uint32_t end = built.source_token_prefix[source + 1u];
        for (uint32_t token = begin; token < end; ++token) {
            const JournalTokenEntry &entry = built.layout.journal_tokens[token];
            for (uint32_t i = 0u; i < entry.contributors_count; ++i) {
                const JournalContributor &contributor =
                    built.layout.contributors[entry.contributors_begin + i];
                built.row_map[static_cast<size_t>(token) * o.workers +
                              contributor.worker_rank] =
                    contributor.destination_row;
                ++running[contributor.worker_rank];
            }
        }
    }
    for (uint32_t destination = 0u; destination < o.workers; ++destination)
        built.source_destination_prefix[
            static_cast<size_t>(o.workers) * o.workers + destination] =
            running[destination];

    built.fixed_journal_tokens = built.layout.journal_tokens;
    built.fixed_contributors = built.layout.contributors;
    const uint64_t row_bytes = static_cast<uint64_t>(o.hidden) * 2u;
    for (const auto &rows : built.layout.destination_rows)
        built.egress_hidden_bytes += rows.size() * row_bytes;
    built.logical_bytes = built.ingress_hidden_bytes +
                          built.egress_hidden_bytes;
    *oracle = std::move(built);
    return true;
}

bool ValidateHidden(const Options &o, const WaveOracle &oracle,
                    const uint8_t *device, uint64_t slot_offset,
                    uint32_t destination)
{
    const uint64_t row_bytes = static_cast<uint64_t>(o.hidden) * 2u;
    const std::vector<DestinationRow> &rows =
        oracle.layout.destination_rows[destination];
    const uint64_t bytes = rows.size() * row_bytes;
    std::vector<uint8_t> actual(static_cast<size_t>(
        std::min<uint64_t>(kValidationChunk, std::max<uint64_t>(bytes, 1u))));
    for (uint64_t offset = 0u; offset < bytes;) {
        const uint64_t chunk = std::min<uint64_t>(actual.size(), bytes - offset);
        if (aclrtMemcpy(actual.data(), chunk, device + slot_offset + offset,
                        chunk, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
            return false;
        for (uint64_t i = 0u; i < chunk; ++i) {
            const uint64_t position = offset + i;
            const uint32_t row = static_cast<uint32_t>(position / row_bytes);
            const uint64_t byte = position % row_bytes;
            const DestinationRow &metadata = rows[row];
            const uint8_t expected = HiddenByte(
                o.seed, metadata.source_rank, metadata.source_token, byte);
            if (actual[i] != expected) {
                std::cerr << "[FAIL] hidden dst=" << destination
                          << " row=" << row << " byte=" << byte
                          << " actual=" << static_cast<uint32_t>(actual[i])
                          << " expected=" << static_cast<uint32_t>(expected)
                          << '\n';
                return false;
            }
        }
        offset += chunk;
    }
    return true;
}

template <typename T>
bool ValidateDeviceVector(const char *name, const uint8_t *device,
                          const std::vector<T> &expected)
{
    std::vector<T> actual;
    if (!CopyVector(&actual, device, expected.size()) ||
        !SameVector(actual, expected)) {
        std::cerr << "[FAIL] " << name << " mismatch count="
                  << expected.size() << '\n';
        return false;
    }
    return true;
}

bool ValidateDestinationRows(const uint8_t *device,
                             const std::vector<DestinationRow> &expected)
{
    std::vector<DestinationRow> actual;
    if (!CopyVector(&actual, device, expected.size())) return false;
    for (size_t i = 0u; i < expected.size(); ++i) {
        if (EqualBytes(&actual[i], &expected[i], sizeof(DestinationRow)))
            continue;
        std::cerr << "[FAIL] destination_row index=" << i
                  << " route=" << actual[i].route_key << "/"
                  << expected[i].route_key
                  << " token=" << actual[i].token_id << "/"
                  << expected[i].token_id
                  << " source=" << actual[i].source_rank << "/"
                  << expected[i].source_rank
                  << " source_token=" << actual[i].source_token << "/"
                  << expected[i].source_token
                  << " row=" << actual[i].destination_row << "/"
                  << expected[i].destination_row << '\n';
        return false;
    }
    return true;
}

bool ValidateWorker(const Options &o, const WaveOracle &oracle,
                    uint32_t expected_status, uint64_t hidden_slot_stride,
                    uint64_t rows_slot_stride,
                    uint64_t assignments_slot_stride,
                    uint64_t expert_slot_stride,
                    const GuardedBuffer &destination_hidden,
                    const GuardedBuffer &destination_rows,
                    const GuardedBuffer &destination_assignments,
                    const GuardedBuffer &destination_expert_counts,
                    const GuardedBuffer &destination_completions,
                    const GuardedBuffer &source_acks)
{
    const uint32_t rank = static_cast<uint32_t>(o.pe);
    const uint64_t cookie = expected_status == 0u
        ? oracle.layout.journal_header.dispatch_cookie : 0u;
    SourceConsumed ack{};
    DestinationCompletion completion{};
    const uint8_t *ack_device = source_acks.data +
        static_cast<uint64_t>(rank) * sizeof(SourceConsumed);
    const uint8_t *completion_device = destination_completions.data +
        (static_cast<uint64_t>(oracle.ring_slot) * o.workers + rank) *
            sizeof(DestinationCompletion);
    if (aclrtMemcpy(&ack, sizeof(ack), ack_device, sizeof(ack),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(&completion, sizeof(completion), completion_device,
                    sizeof(completion), ACL_MEMCPY_DEVICE_TO_HOST) !=
            ACL_SUCCESS)
        return false;
    const uint32_t rows = expected_status == 0u
        ? oracle.layout.destination_rows[rank].size() : 0u;
    const uint32_t assignments = expected_status == 0u
        ? oracle.layout.expert_assignments[rank].size() : 0u;
    const bool ack_ok = ack.magic == kPullDispatchMagic &&
        ack.abi_version == kPullDispatchAbiVersion &&
        ack.struct_bytes == sizeof(SourceConsumed) &&
        ack.session_id == kSessionId &&
        ack.placement_epoch == kPlacementEpoch &&
        ack.generation == oracle.generation &&
        ack.sequence == oracle.sequence && ack.dispatch_cookie == cookie &&
        ack.wave == oracle.wave && ack.source_rank == rank &&
        ack.source_region_id == kRegionId && ack.status == expected_status &&
        ack.ring_slot == oracle.ring_slot && ack.flags == 0u &&
        ack.bytes_consumed == (expected_status == 0u
            ? oracle.source_packet_bytes[rank] : 0u) &&
        ack.publication == AckPublication(oracle.generation, oracle.sequence,
                                         rank, expected_status);
    const bool completion_ok = completion.magic == kPullDispatchMagic &&
        completion.abi_version == kPullDispatchAbiVersion &&
        completion.struct_bytes == sizeof(DestinationCompletion) &&
        completion.session_id == kSessionId &&
        completion.placement_epoch == kPlacementEpoch &&
        completion.generation == oracle.generation &&
        completion.sequence == oracle.sequence &&
        completion.dispatch_cookie == cookie &&
        completion.wave == oracle.wave &&
        completion.destination_rank == rank &&
        completion.status == expected_status && completion.row_count == rows &&
        completion.assignment_count == assignments &&
        completion.ring_slot == oracle.ring_slot && completion.flags == 0u &&
        completion.publication == CompletionPublication(
            oracle.generation, oracle.sequence, rank, expected_status);
    if (!ack_ok || !completion_ok) {
        std::cerr << "[FAIL] ACK/completion identity rank=" << rank
                  << " ack_status=" << ack.status
                  << " ack_cookie=" << ack.dispatch_cookie
                  << " expect_cookie=" << cookie
                  << " ack_pub=" << ack.publication
                  << " expect_ack_pub=" << AckPublication(
                         oracle.generation, oracle.sequence, rank,
                         expected_status)
                  << " completion_status=" << completion.status
                  << " completion_cookie=" << completion.dispatch_cookie
                  << " completion_rows=" << completion.row_count
                  << " expect_rows=" << rows
                  << " completion_assignments="
                  << completion.assignment_count
                  << " expect_assignments=" << assignments
                  << " completion_pub=" << completion.publication
                  << " expect_completion_pub=" << CompletionPublication(
                         oracle.generation, oracle.sequence, rank,
                         expected_status)
                  << '\n';
        return false;
    }
    if (expected_status != 0u) return true;
    const uint64_t hidden_offset = static_cast<uint64_t>(oracle.ring_slot) *
        hidden_slot_stride;
    const uint64_t rows_offset = static_cast<uint64_t>(oracle.ring_slot) *
        rows_slot_stride;
    const uint64_t assignments_offset =
        static_cast<uint64_t>(oracle.ring_slot) * assignments_slot_stride;
    const uint64_t expert_offset = static_cast<uint64_t>(oracle.ring_slot) *
        expert_slot_stride;
    if (!ValidateHidden(o, oracle, destination_hidden.data, hidden_offset,
                        rank) ||
        !ValidateDestinationRows(destination_rows.data + rows_offset,
                                 oracle.layout.destination_rows[rank]) ||
        !ValidateDeviceVector("destination_assignments",
            destination_assignments.data + assignments_offset,
            oracle.layout.expert_assignments[rank]))
        return false;
    std::vector<uint32_t> expected_experts(o.expert_count);
    std::copy_n(oracle.layout.expert_counts.begin() +
                    static_cast<size_t>(rank) * o.expert_count,
                o.expert_count, expected_experts.begin());
    return ValidateDeviceVector("destination_expert_counts",
        destination_expert_counts.data + expert_offset, expected_experts);
}

bool ValidateInc(const Options &o, const WaveOracle &oracle,
                 uint32_t expected_status,
                 const GuardedBuffer &inc_destination_rows,
                 const GuardedBuffer &inc_destination_assignments,
                 const GuardedBuffer &journal_header,
                 const GuardedBuffer &journal_tokens,
                 const GuardedBuffer &journal_contributors,
                 const GuardedBuffer &journal_assignments,
                 const GuardedBuffer &row_map,
                 const GuardedBuffer &source_token_prefix,
                 const GuardedBuffer &source_destination_prefix,
                 const GuardedBuffer &destination_row_counts,
                 const GuardedBuffer &destination_assignment_counts,
                 const GuardedBuffer &expert_counts,
                 const GuardedBuffer &status_line,
                 PullTimeline *timeline_out,
                 uint64_t inc_rows_stride_bytes,
                 uint64_t inc_assignments_stride_bytes)
{
    PullTimeline timeline{};
    if (aclrtMemcpy(&timeline, sizeof(timeline), status_line.data,
                    sizeof(timeline), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
        return false;
    if (timeline.status != expected_status) {
        std::cerr << "[FAIL] status actual=" << timeline.status
                  << " expected=" << expected_status << '\n';
        return false;
    }
    *timeline_out = timeline;
    if (expected_status != 0u) return true;
    JournalSlotHeader header{};
    const uint8_t *header_device = journal_header.data +
        static_cast<uint64_t>(oracle.ring_slot) * sizeof(JournalSlotHeader);
    if (aclrtMemcpy(&header, sizeof(header), header_device, sizeof(header),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
        return false;
    if (!EqualBytes(&header, &oracle.layout.journal_header, sizeof(header))) {
        const JournalSlotHeader &expected = oracle.layout.journal_header;
        std::cerr << "[FAIL] journal_header mismatch"
                  << " magic=" << header.magic << "/" << expected.magic
                  << " abi=" << header.abi_version << "/"
                  << expected.abi_version
                  << " bytes=" << header.struct_bytes << "/"
                  << expected.struct_bytes
                  << " generation=" << header.generation << "/"
                  << expected.generation
                  << " sequence=" << header.sequence << "/"
                  << expected.sequence
                  << " cookie=" << header.dispatch_cookie << "/"
                  << expected.dispatch_cookie
                  << " wave=" << header.wave << "/" << expected.wave
                  << " slot=" << header.ring_slot << "/"
                  << expected.ring_slot
                  << " state=" << header.state << "/" << expected.state
                  << " tokens=" << header.token_count << "/"
                  << expected.token_count
                  << " contributors=" << header.contributor_count << "/"
                  << expected.contributor_count
                  << " status=" << header.status << "/" << expected.status
                  << '\n';
        for (uint32_t source = 0u; source < oracle.sources.size(); ++source)
            std::cerr << "[DEBUG] source=" << source
                      << " metadata_digest="
                      << oracle.sources[source].header.metadata_digest
                      << '\n';
        return false;
    }
    std::vector<JournalContributor> actual_contributors;
    if (!CopyVector(&actual_contributors, journal_contributors.data,
                    oracle.fixed_contributors.size()))
        return false;
    if (!SameVector(actual_contributors, oracle.fixed_contributors)) {
        for (size_t i = 0u; i < actual_contributors.size(); ++i) {
            if (EqualBytes(&actual_contributors[i],
                           &oracle.fixed_contributors[i],
                           sizeof(JournalContributor)))
                continue;
            const JournalContributor &actual = actual_contributors[i];
            const JournalContributor &expected = oracle.fixed_contributors[i];
            std::cerr << "[FAIL] journal_contributor index=" << i
                      << " worker=" << actual.worker_rank << "/"
                      << expected.worker_rank
                      << " row=" << actual.destination_row << "/"
                      << expected.destination_row
                      << " assignment_begin=" << actual.assignment_begin
                      << "/" << expected.assignment_begin
                      << " assignment_count=" << actual.assignment_count
                      << "/" << expected.assignment_count << '\n';
            const size_t begin = i > 4u ? i - 4u : 0u;
            const size_t end = std::min(actual_contributors.size(), i + 8u);
            for (size_t j = begin; j < end; ++j) {
                const JournalContributor &a = actual_contributors[j];
                const JournalContributor &e = oracle.fixed_contributors[j];
                std::cerr << "[DEBUG] contributor[" << j << "]="
                          << a.worker_rank << "," << a.destination_row
                          << "," << a.assignment_begin << ","
                          << a.assignment_count << " expected="
                          << e.worker_rank << "," << e.destination_row
                          << "," << e.assignment_begin << ","
                          << e.assignment_count << '\n';
            }
            std::vector<JournalTokenEntry> actual_tokens;
            if (CopyVector(&actual_tokens, journal_tokens.data,
                           oracle.fixed_journal_tokens.size())) {
                const size_t token_index = i / std::max<uint32_t>(
                    1u, oracle.fixed_journal_tokens[0].contributors_count);
                const size_t token_begin = token_index > 3u
                    ? token_index - 3u : 0u;
                const size_t token_end = std::min(actual_tokens.size(),
                                                   token_index + 5u);
                for (size_t token = token_begin; token < token_end; ++token)
                    std::cerr << "[DEBUG] token[" << token << "] begin/count="
                              << actual_tokens[token].contributors_begin
                              << "/"
                              << actual_tokens[token].contributors_count
                              << " expected="
                              << oracle.fixed_journal_tokens[token]
                                     .contributors_begin
                              << "/"
                              << oracle.fixed_journal_tokens[token]
                                     .contributors_count << '\n';
            }
            break;
        }
        return false;
    }
    if (!ValidateDeviceVector("journal_tokens", journal_tokens.data,
                              oracle.fixed_journal_tokens) ||
        (journal_assignments.bytes != 0u &&
         !ValidateDeviceVector("journal_assignments",
                               journal_assignments.data,
                               oracle.layout.journal_assignments)) ||
        (row_map.bytes != 0u &&
         !ValidateDeviceVector("row_map", row_map.data, oracle.row_map)) ||
        !ValidateDeviceVector("source_token_prefix", source_token_prefix.data,
                              oracle.source_token_prefix) ||
        !ValidateDeviceVector("source_destination_prefix",
                              source_destination_prefix.data,
                              oracle.source_destination_prefix) ||
        !ValidateDeviceVector("expert_counts", expert_counts.data,
                              oracle.layout.expert_counts))
        return false;

    std::vector<uint32_t> expected_rows(o.workers);
    std::vector<uint32_t> expected_assignments(o.workers);
    for (uint32_t destination = 0u; destination < o.workers; ++destination) {
        expected_rows[destination] =
            oracle.layout.destination_rows[destination].size();
        expected_assignments[destination] =
            oracle.layout.expert_assignments[destination].size();
        if (!ValidateDeviceVector("inc_destination_rows",
                inc_destination_rows.data +
                    static_cast<uint64_t>(destination) *
                        inc_rows_stride_bytes,
                oracle.layout.destination_rows[destination]) ||
            !ValidateDeviceVector("inc_destination_assignments",
                inc_destination_assignments.data +
                    static_cast<uint64_t>(destination) *
                        inc_assignments_stride_bytes,
                oracle.layout.expert_assignments[destination]))
            return false;
    }
    return ValidateDeviceVector("destination_row_counts",
               destination_row_counts.data, expected_rows) &&
        ValidateDeviceVector("destination_assignment_counts",
               destination_assignment_counts.data, expected_assignments);
}

uint32_t ExpectedStatus(uint32_t fault)
{
    switch (fault) {
        case 0u: return 0u;
        case 1u: return 8u;   // digest mismatch
        case 2u: return 10u;  // invalid assignment
        case 3u: return 13u;  // missing READY timeout
        case 4u: return 3u;   // strong READY identity mismatch
        case 5u: return 12u;  // journal slot is busy
    }
    return 1u;
}

void ApplyFault(uint32_t fault, uint32_t rank, uint32_t workers,
                std::vector<uint8_t> *slot, Ready *ready)
{
    if (rank != 0u || fault == 0u) return;
    SlotHeader *header = reinterpret_cast<SlotHeader *>(slot->data());
    if (fault == 1u && header->assignment_count != 0u) {
        (*slot)[header->assignments_offset] ^= 1u;
    } else if (fault == 2u && header->assignment_count != 0u) {
        AssignmentRecord *assignments = reinterpret_cast<AssignmentRecord *>(
            slot->data() + header->assignments_offset);
        assignments[0].destination_rank = workers;
        header->metadata_digest = 0u;
        header->metadata_digest = MetadataDigest(*header, slot->data());
    } else if (fault == 3u) {
        ready->publication = 0u;
    } else if (fault == 4u) {
        ++ready->session_id;
        ready->publication = ReadyPublication(*ready);
    }
}

void PrintJson(const Options &o, const WaveOracle &oracle, uint32_t iteration,
               bool warmup, double us, const PullTimeline &timeline,
               bool correct)
{
    const double seconds = us * 1e-6;
    const double gbps = seconds == 0.0 ? 0.0 :
        static_cast<double>(oracle.logical_bytes) / seconds / 1e9;
    const double protocol_us = timeline.kernel_done > timeline.all_ready
        ? static_cast<double>(timeline.kernel_done - timeline.all_ready) *
            kSystemCycleUs
        : 0.0;
    const double protocol_gbps = protocol_us == 0.0 ? 0.0 :
        static_cast<double>(oracle.logical_bytes) /
            (protocol_us * 1e-6) / 1e9;
    std::cout << std::setprecision(12)
              << "{\"test\":\"pull_dispatch_v2_device_e2e\""
              << ",\"workers\":" << o.workers
              << ",\"workload\":\"" << o.workload_name << "\""
              << ",\"iteration\":" << iteration
              << ",\"phase\":\"" << (warmup ? "warmup" : "measure")
              << "\",\"generation\":" << oracle.generation
              << ",\"sequence\":" << oracle.sequence
              << ",\"wave\":" << oracle.wave
              << ",\"ring_slot\":" << oracle.ring_slot
              << ",\"requested_payload_bytes_per_worker\":"
              << o.payload_bytes
              << ",\"ingress_hidden_bytes\":"
              << oracle.ingress_hidden_bytes
              << ",\"egress_hidden_bytes\":"
              << oracle.egress_hidden_bytes
              << ",\"logical_bytes\":" << oracle.logical_bytes
              << ",\"makespan_us\":" << us
              << ",\"logical_gb_s\":" << gbps
              << ",\"protocol_makespan_us\":" << protocol_us
              << ",\"protocol_logical_gb_s\":" << protocol_gbps
              << ",\"status\":" << timeline.status
              << ",\"ready_sources\":" << timeline.ready_sources
              << ",\"cycle_kernel_start\":" << timeline.kernel_start
              << ",\"cycle_all_ready\":" << timeline.all_ready
              << ",\"cycle_headers_pulled\":" << timeline.headers_pulled
              << ",\"cycle_metadata_parse_begin\":"
              << timeline.metadata_parse_begin
              << ",\"cycle_metadata_parse_done\":"
              << timeline.metadata_parse_done
              << ",\"cycle_journal_reserved\":"
              << timeline.journal_reserved
              << ",\"cycle_hidden_get_begin\":"
              << timeline.hidden_get_begin
              << ",\"cycle_hidden_get_done\":" << timeline.hidden_get_done
              << ",\"cycle_fanout_put_begin\":"
              << timeline.fanout_put_begin
              << ",\"cycle_fanout_put_done\":"
              << timeline.fanout_put_done
              << ",\"cycle_reorg_done\":" << timeline.reorg_done
              << ",\"cycle_destination_completions_done\":"
              << timeline.destination_completions_done
              << ",\"cycle_source_acks_done\":"
              << timeline.source_acks_done
              << ",\"cycle_kernel_done\":" << timeline.kernel_done
              << ",\"correct\":" << (correct ? "true" : "false")
              << "}\n";
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 14) {
        std::cerr
            << "usage: " << argv[0]
            << " <workers> <pe> <ipport> <first_npu>"
               " <per_worker_payload_bytes>"
               " <sym_k2_balanced|sym_dense|sym_k1_rr|hotspot|ragged>"
               " <hidden> <expert_count> <channels_per_source>"
               " <warmup> <measure> <seed>"
               " <fault:0=none,1=digest,2=assignment,3=missing_ready,"
               "4=ready_identity,5=busy_journal>\n";
        return 2;
    }
    Options o{};
    o.workers = static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10));
    o.pe = std::atoi(argv[2]);
    ipport = argv[3];
    o.first_npu = std::atoi(argv[4]);
    f_npu = o.first_npu;
    o.payload_bytes = std::strtoull(argv[5], nullptr, 10);
    o.workload_name = argv[6];
    o.hidden = static_cast<uint32_t>(std::strtoul(argv[7], nullptr, 10));
    o.expert_count = static_cast<uint32_t>(std::strtoul(
        argv[8], nullptr, 10));
    o.channels = static_cast<uint32_t>(std::strtoul(argv[9], nullptr, 10));
    o.warmup = static_cast<uint32_t>(std::strtoul(argv[10], nullptr, 10));
    o.measure = static_cast<uint32_t>(std::strtoul(argv[11], nullptr, 10));
    o.seed = std::strtoull(argv[12], nullptr, 10);
    o.fault = static_cast<uint32_t>(std::strtoul(argv[13], nullptr, 10));
    const uint32_t pes = o.workers + 1u;
    const int inc_pe = static_cast<int>(o.workers);
    g_npus = static_cast<int>(pes);
    if (!ParseWorkload(o.workload_name, &o.workload) ||
        o.workers < 2u || o.workers > 8u || o.pe < 0 ||
        o.pe >= static_cast<int>(pes) || o.hidden == 0u ||
        o.expert_count == 0u || o.channels == 0u || o.measure == 0u ||
        o.fault > 5u || (o.fault != 0u && o.payload_bytes == 0u &&
                         (o.fault == 1u || o.fault == 2u)))
        return Fail("arguments", 2);
    const uint64_t row_bytes = static_cast<uint64_t>(o.hidden) * 2u;
    if (row_bytes / 2u != o.hidden ||
        o.payload_bytes > static_cast<uint64_t>(
            std::numeric_limits<uint32_t>::max()) * row_bytes)
        return Fail("payload range", 2);

    // Capacity is the exact maximum required by this workload.  It is
    // independent of wave identity, so one bounded allocation serves every
    // ring slot without hiding reallocations in the timed region.
    WaveOracle sizing{};
    std::vector<uint8_t> unused_slot;
    Ready unused_ready{};
    uint64_t source_stride = 0u;
    if (!BuildOracle(o, kFirstGeneration, kFirstSequence, kFirstWave, 0u,
                     -1, &sizing, &unused_slot, &unused_ready,
                     &source_stride))
        return Fail("sizing oracle", 2);
    const uint64_t total_tokens = sizing.layout.journal_tokens.size();
    uint64_t row_capacity = 1u;
    uint64_t assignment_capacity = 1u;
    for (uint32_t destination = 0u; destination < o.workers; ++destination) {
        row_capacity = std::max<uint64_t>(
            row_capacity, sizing.layout.destination_rows[destination].size());
        assignment_capacity = std::max<uint64_t>(assignment_capacity,
            sizing.layout.expert_assignments[destination].size());
    }
    const uint64_t journal_token_capacity = std::max<uint64_t>(total_tokens, 1u);
    const uint64_t journal_contributor_capacity =
        std::max<uint64_t>(sizing.layout.contributors.size(), 1u);
    // Combine V2 consumes token/contributor ranges; the duplicated full
    // assignment journal is an optional diagnostics workspace.
    const uint64_t journal_assignment_capacity = 0u;
    // Canonical V2 relay and Combine consume compact contributors directly.
    // A zero capacity disables the legacy dense W x token row map.
    const uint64_t row_map_entries = 0u;
    const uint64_t prefix_entries =
        static_cast<uint64_t>(o.workers + 1u) * o.workers;
    const uint64_t expert_entries =
        static_cast<uint64_t>(o.workers) * o.expert_count;
    const uint64_t hidden_slot_stride = Align64(row_capacity * row_bytes);
    const uint64_t rows_slot_stride = Align64(
        row_capacity * sizeof(DestinationRow));
    const uint64_t assignments_slot_stride = Align64(
        assignment_capacity * sizeof(ExpertAssignment));
    const uint64_t expert_slot_stride = Align64(
        static_cast<uint64_t>(o.expert_count) * sizeof(uint32_t));

    const int device = o.pe + o.first_npu;
    aclrtStream stream = nullptr;
    bool shmem_initialized = false;
    std::vector<GuardedBuffer *> buffers;
    GuardedBuffer source_region, ready_mailbox, source_acks;
    GuardedBuffer destination_hidden, destination_rows;
    GuardedBuffer destination_assignments, destination_expert_counts;
    GuardedBuffer destination_completions;
    GuardedBuffer inc_slots, inc_destination_rows;
    GuardedBuffer inc_destination_assignments, journal_header;
    GuardedBuffer journal_tokens, journal_contributors, journal_assignments;
    GuardedBuffer row_map, source_token_prefix, source_destination_prefix;
    GuardedBuffer destination_row_counts, destination_assignment_counts;
    GuardedBuffer expert_counts, parser_scratch, status_line;
    buffers = {&source_region, &ready_mailbox, &source_acks,
        &destination_hidden, &destination_rows, &destination_assignments,
        &destination_expert_counts, &destination_completions, &inc_slots,
        &inc_destination_rows, &inc_destination_assignments, &journal_header,
        &journal_tokens, &journal_contributors, &journal_assignments, &row_map,
        &source_token_prefix, &source_destination_prefix,
        &destination_row_counts, &destination_assignment_counts,
        &expert_counts, &parser_scratch, &status_line};

    int status = aclInit(nullptr);
    if (status == 0) status = aclrtSetDevice(device);
    int64_t live_aiv = 0;
    uint32_t dispatch_aiv_budget = 0u;
    uint64_t parser_scratch_entries = 0u;
    if (status == 0)
        status = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                    &live_aiv);
    if (status == 0) {
        dispatch_aiv_budget = live_aiv <= 0
            ? 0u : static_cast<uint32_t>(live_aiv) / 2u;
        if (dispatch_aiv_budget == 0u ||
            static_cast<uint64_t>(o.workers) * (o.channels + 1u) >
                dispatch_aiv_budget ||
            !ParserScratchEntries(dispatch_aiv_budget, o.workers,
                                  o.expert_count,
                                  &parser_scratch_entries))
            status = 2;
    }
    if (status == 0) status = aclrtCreateStream(&stream);
    if (status == 0) {
        aclshmemx_init_attr_t attr;
        test_set_attr(o.pe, pes, 2ull * 1024ull * 1024ull * 1024ull,
                      ipport, default_flag_uid, &attr);
        status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
        shmem_initialized = status == 0;
    }
    auto Alloc = [&](GuardedBuffer *buffer, uint64_t bytes, bool symmetric,
                     const char *name) {
        if (status == 0 && !Allocate(buffer, bytes, symmetric, name)) status = 1;
    };
    if (status == 0) {
        // Only objects addressed by remote SHMEM operations are symmetric.
        // INC-private staging uses ordinary HBM, which keeps 128-MiB/source
        // dense cases comfortably below the 2-GiB symmetric heap limit.
        Alloc(&source_region, source_stride * kRingSlots, true,
              "source_region");
        Alloc(&ready_mailbox, o.workers * sizeof(Ready), true,
              "ready_mailbox");
        Alloc(&source_acks, o.workers * sizeof(SourceConsumed), true,
              "source_acks");
        Alloc(&destination_hidden, hidden_slot_stride * kRingSlots, true,
              "destination_hidden");
        Alloc(&destination_rows, rows_slot_stride * kRingSlots, true,
              "destination_rows");
        Alloc(&destination_assignments,
              assignments_slot_stride * kRingSlots, true,
              "destination_assignments");
        Alloc(&destination_expert_counts, expert_slot_stride * kRingSlots,
              true, "destination_expert_counts");
        Alloc(&destination_completions,
              static_cast<uint64_t>(kRingSlots) * o.workers *
                  sizeof(DestinationCompletion), true,
              "destination_completions");
        Alloc(&inc_slots, static_cast<uint64_t>(o.workers) * source_stride,
              false, "inc_slots");
        Alloc(&inc_destination_rows,
              static_cast<uint64_t>(o.workers) * rows_slot_stride,
              false, "inc_destination_rows");
        Alloc(&inc_destination_assignments,
              static_cast<uint64_t>(o.workers) * assignments_slot_stride,
              false,
              "inc_destination_assignments");
        Alloc(&journal_header, kRingSlots * sizeof(JournalSlotHeader), false,
              "journal_header");
        Alloc(&journal_tokens,
              journal_token_capacity * sizeof(JournalTokenEntry), false,
              "journal_tokens");
        Alloc(&journal_contributors,
              journal_contributor_capacity * sizeof(JournalContributor),
              false, "journal_contributors");
        Alloc(&journal_assignments,
              journal_assignment_capacity * sizeof(AssignmentRecord), false,
              "journal_assignments");
        Alloc(&row_map, row_map_entries * sizeof(uint32_t), false, "row_map");
        Alloc(&source_token_prefix,
              static_cast<uint64_t>(o.workers + 1u) * sizeof(uint32_t), false,
              "source_token_prefix");
        Alloc(&source_destination_prefix,
              prefix_entries * sizeof(uint32_t), false,
              "source_destination_prefix");
        Alloc(&destination_row_counts, o.workers * sizeof(uint32_t), false,
              "destination_row_counts");
        Alloc(&destination_assignment_counts,
              o.workers * sizeof(uint32_t), false,
              "destination_assignment_counts");
        Alloc(&expert_counts, expert_entries * sizeof(uint32_t), false,
              "expert_counts");
        Alloc(&parser_scratch,
              parser_scratch_entries * sizeof(uint32_t), false,
              "parser_scratch");
        Alloc(&status_line, sizeof(PullTimeline), false, "status_line");
    }

    bool correct = status == 0;
    const uint32_t iterations = o.warmup + o.measure;
    std::vector<double> measured;
    std::vector<double> measured_protocol_gbps;
    for (uint32_t iteration = 0u; iteration < iterations; ++iteration) {
        bool iteration_correct = true;
        const uint64_t generation = kFirstGeneration + iteration;
        const uint64_t sequence = kFirstSequence + iteration;
        const uint32_t wave = kFirstWave + iteration;
        const uint32_t ring_slot = iteration % kRingSlots;
        WaveOracle oracle{};
        std::vector<uint8_t> local_slot;
        Ready local_ready{};
        uint64_t wave_stride = 0u;
        const int local_source = o.pe < inc_pe ? o.pe : -1;
        iteration_correct = BuildOracle(
            o, generation, sequence, wave, ring_slot, local_source, &oracle,
            &local_slot, &local_ready, &wave_stride) &&
            wave_stride == source_stride;
        if (!iteration_correct) {
            correct = false;
            break;
        }
        if (o.pe < inc_pe)
            ApplyFault(o.fault, static_cast<uint32_t>(o.pe), o.workers,
                       &local_slot, &local_ready);

        // All preparation and slot reuse happen before the timing barrier.
        // The previous SourceConsumed/completion was already validated, so
        // clearing and overwriting this ring slot is lease-safe.
        if (o.pe < inc_pe) {
            status = aclrtMemcpy(source_region.data +
                    static_cast<uint64_t>(ring_slot) * source_stride,
                source_stride, local_slot.data(), source_stride,
                ACL_MEMCPY_HOST_TO_DEVICE);
            if (status == 0)
                status = aclrtMemcpy(ready_mailbox.data +
                        static_cast<uint64_t>(o.pe) * sizeof(Ready),
                    sizeof(Ready), &local_ready, sizeof(Ready),
                    ACL_MEMCPY_HOST_TO_DEVICE);
        }
        if (status == 0) {
            status = Fill(&source_acks, 0u) &&
                Fill(&destination_hidden, kPoison) &&
                Fill(&destination_rows, kPoison) &&
                Fill(&destination_assignments, kPoison) &&
                Fill(&destination_expert_counts, kPoison) &&
                Fill(&destination_completions, 0u) ? 0 : 1;
        }
        if (status == 0 && o.pe == inc_pe) {
            status = Fill(&inc_slots, 0u) &&
                Fill(&inc_destination_rows, kPoison) &&
                Fill(&inc_destination_assignments, kPoison) &&
                Fill(&journal_tokens, kPoison) &&
                Fill(&journal_contributors, 0u) &&
                Fill(&journal_assignments, kPoison) &&
                Fill(&row_map, kPoison) && Fill(&source_token_prefix, 0u) &&
                Fill(&source_destination_prefix, kPoison) &&
                Fill(&destination_row_counts, 0u) &&
                Fill(&destination_assignment_counts, 0u) &&
                Fill(&expert_counts, 0u) && Fill(&parser_scratch, 0u) &&
                Fill(&status_line, 0u) ? 0 : 1;
            JournalSlotHeader initial{};
            if (o.fault == 5u) {
                initial.magic = kPullDispatchMagic;
                initial.abi_version = kPullDispatchAbiVersion;
                initial.struct_bytes = sizeof(JournalSlotHeader);
                initial.state = static_cast<uint16_t>(
                    JournalSlotState::DISPATCH_OPEN);
            }
            if (status == 0)
                status = aclrtMemcpy(journal_header.data +
                        static_cast<uint64_t>(ring_slot) * sizeof(initial),
                    sizeof(initial), &initial, sizeof(initial),
                    ACL_MEMCPY_HOST_TO_DEVICE);
        }
        if (status == 0) status = aclrtSynchronizeStream(stream);
        if (status == 0) aclshmem_barrier_all();
        if (status != 0) {
            correct = false;
            break;
        }

        const auto begin = std::chrono::steady_clock::now();
        launch_inc_dc_pull_dispatch_v2_device(
            dispatch_aiv_budget, stream, source_region.data,
            ready_mailbox.data, inc_slots.data, source_acks.data,
            destination_hidden.data, destination_rows.data,
            destination_assignments.data, destination_expert_counts.data,
            destination_completions.data, inc_destination_rows.data,
            inc_destination_assignments.data,
            journal_header.data + static_cast<uint64_t>(ring_slot) *
                sizeof(JournalSlotHeader),
            journal_tokens.data, journal_contributors.data,
            journal_assignments.data, row_map.data, source_token_prefix.data,
            source_destination_prefix.data, destination_row_counts.data,
            destination_assignment_counts.data, expert_counts.data,
            parser_scratch.data, status_line.data,
            shmemx_get_ffts_config(), kSessionId,
            kPlacementEpoch, generation, sequence, source_stride,
            hidden_slot_stride, rows_slot_stride, assignments_slot_stride,
            expert_slot_stride, journal_token_capacity,
            journal_contributor_capacity, journal_assignment_capacity,
            row_capacity, assignment_capacity, rows_slot_stride,
            assignments_slot_stride, row_map_entries,
            prefix_entries, expert_entries, parser_scratch_entries,
            o.workers, o.expert_count, o.hidden,
            static_cast<uint32_t>(DataType::BF16), inc_pe,
            kRegionId, wave, ring_slot, kRingSlots, o.channels,
            o.fault == 3u || o.fault == 4u ? kFaultSpinCap : kSpinCap);
        status = aclrtSynchronizeStream(stream);
        const auto end = std::chrono::steady_clock::now();
        if (status == 0) aclshmem_barrier_all();
        if (status != 0) {
            correct = false;
            break;
        }

        const uint32_t expected_status = ExpectedStatus(o.fault);
        PullTimeline timeline{};
        if (o.pe < inc_pe)
            iteration_correct = ValidateWorker(o, oracle, expected_status,
                hidden_slot_stride, rows_slot_stride,
                assignments_slot_stride, expert_slot_stride,
                destination_hidden, destination_rows,
                destination_assignments, destination_expert_counts,
                destination_completions, source_acks);
        else
            iteration_correct = ValidateInc(o, oracle, expected_status,
                inc_destination_rows, inc_destination_assignments,
                journal_header, journal_tokens, journal_contributors,
                journal_assignments, row_map, source_token_prefix,
                source_destination_prefix, destination_row_counts,
                destination_assignment_counts, expert_counts, status_line,
                &timeline, rows_slot_stride, assignments_slot_stride);
        for (GuardedBuffer *buffer : buffers)
            iteration_correct = GuardsValid(*buffer) && iteration_correct;
        if (status == 0) aclshmem_barrier_all();
        // Every rank executes the same fixed wave count even after a local
        // validation failure.  This converges the control flow at the barrier
        // and prevents one rank from exiting while peers enter the next wave.
        correct = iteration_correct && correct;

        if (o.pe == inc_pe) {
            const double us = std::chrono::duration<double, std::micro>(
                end - begin).count();
            const bool warmup = iteration < o.warmup;
            PrintJson(o, oracle, iteration, warmup, us, timeline,
                      iteration_correct);
            if (!warmup) {
                measured.push_back(us);
                const double protocol_us =
                    timeline.kernel_done > timeline.all_ready
                    ? static_cast<double>(timeline.kernel_done -
                                          timeline.all_ready) *
                        kSystemCycleUs
                    : 0.0;
                measured_protocol_gbps.push_back(protocol_us == 0.0 ? 0.0 :
                    static_cast<double>(oracle.logical_bytes) /
                        (protocol_us * 1e-6) / 1e9);
            }
        }
    }

    if (correct && o.pe == inc_pe && !measured.empty()) {
        const double mean = std::accumulate(measured.begin(), measured.end(),
                                            0.0) / measured.size();
        double variance = 0.0;
        for (double value : measured)
            variance += (value - mean) * (value - mean);
        variance /= measured.size();
        const double protocol_mean = std::accumulate(
            measured_protocol_gbps.begin(),
            measured_protocol_gbps.end(), 0.0) /
            measured_protocol_gbps.size();
        double protocol_variance = 0.0;
        for (double value : measured_protocol_gbps)
            protocol_variance += (value - protocol_mean) *
                                 (value - protocol_mean);
        protocol_variance /= measured_protocol_gbps.size();
        std::cout << std::setprecision(12)
                  << "{\"test\":\"pull_dispatch_v2_device_e2e_summary\""
                  << ",\"workers\":" << o.workers
                  << ",\"workload\":\"" << o.workload_name << "\""
                  << ",\"measure\":" << measured.size()
                  << ",\"min_us\":"
                  << *std::min_element(measured.begin(), measured.end())
                  << ",\"mean_us\":" << mean
                  << ",\"max_us\":"
                  << *std::max_element(measured.begin(), measured.end())
                  << ",\"cv_percent\":"
                  << (mean == 0.0 ? 0.0 : std::sqrt(variance) / mean * 100.0)
                  << ",\"protocol_min_gb_s\":"
                  << *std::min_element(measured_protocol_gbps.begin(),
                                       measured_protocol_gbps.end())
                  << ",\"protocol_mean_gb_s\":" << protocol_mean
                  << ",\"protocol_cv_percent\":"
                  << (protocol_mean == 0.0 ? 0.0 :
                      std::sqrt(protocol_variance) / protocol_mean * 100.0)
                  << ",\"correct\":true}\n";
    }

    // All ranks release symmetric objects in exactly the same reverse order.
    for (auto it = buffers.rbegin(); it != buffers.rend(); ++it)
        Release(*it);
    if (shmem_initialized) aclshmem_finalize();
    if (stream != nullptr) aclrtDestroyStream(stream);
    if (status == 0) status = aclrtResetDevice(device);
    aclFinalize();
    if (!correct || status != 0)
        return Fail("pull Dispatch V2 device E2E qualification", status);
    std::cout << "[PASS] pe=" << o.pe << " workers=" << o.workers
              << " workload=" << o.workload_name << '\n';
    return 0;
}
