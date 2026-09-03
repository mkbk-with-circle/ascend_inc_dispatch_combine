#include "inc_dc_pull_dispatch_v2.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace inc::dc::pull_v2 {
namespace {

constexpr uint64_t kHashOffset = 1469598103934665603ull;
constexpr uint64_t kHashPrime = 1099511628211ull;

Status Fail(Status status, const char *message, std::string *error)
{
    if (error != nullptr) *error = message;
    return status;
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
        b > std::numeric_limits<uint64_t>::max() / a))
        return false;
    *out = a * b;
    return true;
}

bool Align(uint64_t value, uint64_t alignment, uint64_t *out)
{
    uint64_t expanded = 0u;
    return alignment != 0u && (alignment & (alignment - 1u)) == 0u &&
        Add(value, alignment - 1u, &expanded) &&
        ((*out = expanded / alignment * alignment), true);
}

uint32_t DTypeBytes(DataType dtype)
{
    switch (dtype) {
        case DataType::FP16:
        case DataType::BF16: return 2u;
        case DataType::FP32: return 4u;
    }
    return 0u;
}

bool SessionValid(const SessionConfig &session)
{
    return session.session_id != 0u && session.placement_epoch != 0u &&
        session.worker_count >= 2u &&
        session.worker_count <= kPullDispatchMaxWorkers &&
        session.expert_count != 0u && session.hidden != 0u &&
        DTypeBytes(session.dtype) != 0u && session.ring_slots != 0u &&
        session.ring_slots <= std::numeric_limits<uint16_t>::max();
}

void HashBytes(uint64_t *hash, const void *data, uint64_t bytes)
{
    const uint8_t *p = static_cast<const uint8_t *>(data);
    for (uint64_t i = 0u; i < bytes; ++i) {
        *hash ^= p[i];
        *hash *= kHashPrime;
    }
}

template <typename T>
void HashValue(uint64_t *hash, const T &value)
{
    HashBytes(hash, &value, sizeof(value));
}

bool AllZero(const uint64_t *values, uint32_t count)
{
    for (uint32_t i = 0u; i < count; ++i)
        if (values[i] != 0u) return false;
    return true;
}

} // namespace

uint64_t ReadyPublication(const Ready &ready)
{
    Ready canonical = ready;
    canonical.publication = 0u;
    uint64_t hash = kHashOffset;
    HashBytes(&hash, &canonical, sizeof(canonical));
    return hash == 0u ? 1u : hash;
}

uint64_t MetadataDigest(const SlotHeader &header, const uint8_t *slot)
{
    if (slot == nullptr) return 0u;
    SlotHeader canonical = header;
    canonical.metadata_digest = 0u;
    uint64_t hash = kHashOffset;
    HashBytes(&hash, &canonical, sizeof(canonical));
    HashBytes(&hash, slot + header.tokens_offset,
              static_cast<uint64_t>(header.token_count) *
                  sizeof(TokenRecord));
    HashBytes(&hash, slot + header.assignments_offset,
              static_cast<uint64_t>(header.assignment_count) *
                  sizeof(AssignmentRecord));
    return hash;
}

Status ValidateRegistration(const RegionRegistration &registration,
                            const SessionConfig &session,
                            std::string *error)
{
    uint64_t required = 0u;
    if (!SessionValid(session) ||
        registration.session_id != session.session_id ||
        registration.placement_epoch != session.placement_epoch ||
        registration.source_rank >= session.worker_count ||
        registration.region_id == 0u || registration.slot_count == 0u ||
        registration.slot_count < session.ring_slots ||
        registration.alignment < kPullDispatchAlignment ||
        (registration.alignment & (registration.alignment - 1u)) != 0u ||
        registration.slot_stride < sizeof(SlotHeader) ||
        registration.slot_stride % registration.alignment != 0u ||
        !Mul(registration.slot_count, registration.slot_stride, &required) ||
        required > registration.region_bytes ||
        !AllZero(registration.reserved, 2u)) {
        return Fail(Status::INVALID_REGISTRATION,
                    "invalid registered pull region", error);
    }
    if (error != nullptr) error->clear();
    return Status::OK;
}

Status BuildSlot(const SourceInput &input, std::vector<uint8_t> *slot,
                 SlotHeader *header, Ready *ready, std::string *error)
{
    if (slot == nullptr || header == nullptr || ready == nullptr ||
        !SessionValid(input.session) || input.generation == 0u ||
        input.sequence == 0u ||
        input.source_rank >= input.session.worker_count ||
        input.source_region_id == 0u ||
        input.ring_slot >= input.session.ring_slots ||
        input.token_ids.size() > std::numeric_limits<uint32_t>::max() ||
        input.assignments.size() > std::numeric_limits<uint32_t>::max() ||
        input.assignment_offsets.size() != input.token_ids.size() + 1u ||
        input.assignment_offsets.empty() ||
        input.assignment_offsets.front() != 0u ||
        input.assignment_offsets.back() != input.assignments.size() ||
        !std::is_sorted(input.assignment_offsets.begin(),
                        input.assignment_offsets.end())) {
        return Fail(Status::INVALID_ARGUMENT, "invalid source input", error);
    }

    uint64_t hidden_elements = 0u;
    uint64_t hidden_bytes = 0u;
    if (!Mul(input.token_ids.size(), input.session.hidden,
             &hidden_elements) ||
        !Mul(hidden_elements, DTypeBytes(input.session.dtype),
             &hidden_bytes)) {
        return Fail(Status::SIZE_OVERFLOW, "hidden size overflow", error);
    }
    if (hidden_bytes != input.hidden_payload.size())
        return Fail(Status::INVALID_ARGUMENT,
                    "hidden payload size mismatch", error);

    std::vector<TokenRecord> tokens(input.token_ids.size());
    for (uint32_t token = 0u; token < tokens.size(); ++token) {
        const uint32_t begin = input.assignment_offsets[token];
        const uint32_t end = input.assignment_offsets[token + 1u];
        std::vector<uint8_t> ordinals(end - begin, 0u);
        for (uint32_t i = begin; i < end; ++i) {
            const AssignmentRecord &assignment = input.assignments[i];
            if (assignment.destination_rank >= input.session.worker_count ||
                assignment.expert_id >= input.session.expert_count ||
                !std::isfinite(assignment.weight) ||
                assignment.ordinal >= end - begin ||
                ordinals[assignment.ordinal] != 0u) {
                return Fail(Status::INVALID_ASSIGNMENT,
                            "invalid token assignment", error);
            }
            ordinals[assignment.ordinal] = 1u;
        }
        tokens[token].token_id = input.token_ids[token];
        tokens[token].source_token = token;
        tokens[token].assignment_begin = begin;
        tokens[token].assignment_count = end - begin;
    }

    uint64_t token_bytes = 0u;
    uint64_t assignment_bytes = 0u;
    uint64_t end = 0u;
    SlotHeader built{};
    built.session_id = input.session.session_id;
    built.placement_epoch = input.session.placement_epoch;
    built.generation = input.generation;
    built.sequence = input.sequence;
    built.wave = input.wave;
    built.source_rank = input.source_rank;
    built.source_region_id = input.source_region_id;
    built.ring_slot = input.ring_slot;
    built.worker_count = input.session.worker_count;
    built.token_count = static_cast<uint32_t>(tokens.size());
    built.assignment_count = static_cast<uint32_t>(input.assignments.size());
    built.hidden = input.session.hidden;
    built.dtype = static_cast<uint32_t>(input.session.dtype);
    built.token_record_bytes = sizeof(TokenRecord);
    built.assignment_record_bytes = sizeof(AssignmentRecord);
    if (!Mul(tokens.size(), sizeof(TokenRecord), &token_bytes) ||
        !Mul(input.assignments.size(), sizeof(AssignmentRecord),
             &assignment_bytes) ||
        !Align(sizeof(SlotHeader), kPullDispatchAlignment,
               &built.tokens_offset) ||
        !Add(built.tokens_offset, token_bytes, &end) ||
        !Align(end, kPullDispatchAlignment, &built.assignments_offset) ||
        !Add(built.assignments_offset, assignment_bytes, &end) ||
        !Align(end, kPullDispatchAlignment, &built.hidden_offset) ||
        !Add(built.hidden_offset, hidden_bytes, &end) ||
        !Align(end, kPullDispatchAlignment, &built.packet_bytes) ||
        built.packet_bytes > std::numeric_limits<size_t>::max()) {
        return Fail(Status::SIZE_OVERFLOW, "slot size overflow", error);
    }

    slot->assign(static_cast<size_t>(built.packet_bytes), 0u);
    std::memcpy(slot->data(), &built, sizeof(built));
    if (!tokens.empty())
        std::memcpy(slot->data() + built.tokens_offset, tokens.data(),
                    static_cast<size_t>(token_bytes));
    if (!input.assignments.empty())
        std::memcpy(slot->data() + built.assignments_offset,
                    input.assignments.data(),
                    static_cast<size_t>(assignment_bytes));
    if (!input.hidden_payload.empty())
        std::memcpy(slot->data() + built.hidden_offset,
                    input.hidden_payload.data(),
                    input.hidden_payload.size());
    built.metadata_digest = MetadataDigest(built, slot->data());
    std::memcpy(slot->data(), &built, sizeof(built));

    Ready built_ready{};
    built_ready.session_id = built.session_id;
    built_ready.placement_epoch = built.placement_epoch;
    built_ready.generation = built.generation;
    built_ready.sequence = built.sequence;
    built_ready.wave = built.wave;
    built_ready.source_rank = built.source_rank;
    built_ready.source_region_id = built.source_region_id;
    built_ready.ring_slot = built.ring_slot;
    built_ready.publication = ReadyPublication(built_ready);
    *header = built;
    *ready = built_ready;
    if (error != nullptr) error->clear();
    return Status::OK;
}

Status ParseReadyAndSlot(const RegionRegistration &registration,
                         const SessionConfig &session, const Ready &ready,
                         const uint8_t *slot, uint64_t slot_capacity,
                         ParsedSource *parsed, std::string *error)
{
    if (parsed == nullptr || slot == nullptr ||
        ValidateRegistration(registration, session, error) != Status::OK)
        return parsed == nullptr || slot == nullptr
            ? Fail(Status::INVALID_ARGUMENT, "invalid parser output", error)
            : Status::INVALID_REGISTRATION;
    if (ready.magic != kPullDispatchMagic ||
        ready.abi_version != kPullDispatchAbiVersion ||
        ready.struct_bytes != sizeof(Ready) || ready.flags != 0u ||
        ready.session_id != session.session_id ||
        ready.placement_epoch != session.placement_epoch ||
        ready.source_rank != registration.source_rank ||
        ready.source_region_id != registration.region_id ||
        ready.ring_slot >= registration.slot_count ||
        ready.ring_slot >= session.ring_slots || ready.generation == 0u ||
        ready.sequence == 0u ||
        ready.publication != ReadyPublication(ready)) {
        return Fail(Status::INVALID_READY, "invalid READY descriptor", error);
    }
    if (slot_capacity < sizeof(SlotHeader) ||
        slot_capacity > registration.slot_stride)
        return Fail(Status::CAPACITY_EXCEEDED,
                    "invalid source slot capacity", error);

    SlotHeader header{};
    std::memcpy(&header, slot, sizeof(header));
    if (header.magic != kPullDispatchMagic ||
        header.abi_version != kPullDispatchAbiVersion ||
        header.header_bytes != sizeof(SlotHeader) || header.flags != 0u ||
        header.session_id != ready.session_id ||
        header.placement_epoch != ready.placement_epoch ||
        header.generation != ready.generation ||
        header.sequence != ready.sequence || header.wave != ready.wave ||
        header.source_rank != ready.source_rank ||
        header.source_region_id != ready.source_region_id ||
        header.ring_slot != ready.ring_slot ||
        header.worker_count != session.worker_count ||
        header.hidden != session.hidden ||
        header.dtype != static_cast<uint32_t>(session.dtype) ||
        header.token_record_bytes != sizeof(TokenRecord) ||
        header.assignment_record_bytes != sizeof(AssignmentRecord)) {
        return Fail(Status::INVALID_HEADER, "slot header mismatch", error);
    }

    uint64_t token_bytes = 0u;
    uint64_t assignment_bytes = 0u;
    uint64_t hidden_elements = 0u;
    uint64_t hidden_bytes = 0u;
    uint64_t canonical_tokens = 0u;
    uint64_t canonical_assignments = 0u;
    uint64_t canonical_hidden = 0u;
    uint64_t canonical_packet = 0u;
    uint64_t end = 0u;
    if (!Mul(header.token_count, sizeof(TokenRecord), &token_bytes) ||
        !Mul(header.assignment_count, sizeof(AssignmentRecord),
             &assignment_bytes) ||
        !Mul(header.token_count, header.hidden, &hidden_elements) ||
        !Mul(hidden_elements, DTypeBytes(session.dtype), &hidden_bytes) ||
        !Align(sizeof(SlotHeader), kPullDispatchAlignment,
               &canonical_tokens) ||
        !Add(canonical_tokens, token_bytes, &end) ||
        !Align(end, kPullDispatchAlignment, &canonical_assignments) ||
        !Add(canonical_assignments, assignment_bytes, &end) ||
        !Align(end, kPullDispatchAlignment, &canonical_hidden) ||
        !Add(canonical_hidden, hidden_bytes, &end) ||
        !Align(end, kPullDispatchAlignment, &canonical_packet)) {
        return Fail(Status::SIZE_OVERFLOW, "slot header overflow", error);
    }
    if (header.tokens_offset != canonical_tokens ||
        header.assignments_offset != canonical_assignments ||
        header.hidden_offset != canonical_hidden ||
        header.packet_bytes != canonical_packet ||
        header.packet_bytes > slot_capacity) {
        return Fail(Status::INVALID_HEADER,
                    "noncanonical slot offsets", error);
    }
    if (MetadataDigest(header, slot) != header.metadata_digest)
        return Fail(Status::DIGEST_MISMATCH,
                    "slot metadata digest mismatch", error);

    ParsedSource built{};
    built.header = header;
    built.tokens.resize(header.token_count);
    built.assignments.resize(header.assignment_count);
    built.hidden_payload.resize(static_cast<size_t>(hidden_bytes));
    if (token_bytes != 0u)
        std::memcpy(built.tokens.data(), slot + header.tokens_offset,
                    static_cast<size_t>(token_bytes));
    if (assignment_bytes != 0u)
        std::memcpy(built.assignments.data(),
                    slot + header.assignments_offset,
                    static_cast<size_t>(assignment_bytes));
    if (hidden_bytes != 0u)
        std::memcpy(built.hidden_payload.data(), slot + header.hidden_offset,
                    static_cast<size_t>(hidden_bytes));

    uint32_t expected_assignment = 0u;
    for (uint32_t token = 0u; token < header.token_count; ++token) {
        const TokenRecord &record = built.tokens[token];
        if (record.source_token != token ||
            record.assignment_begin != expected_assignment ||
            record.assignment_count >
                header.assignment_count - expected_assignment ||
            record.reserved0 != 0u || record.reserved1 != 0u) {
            return Fail(Status::INVALID_TOKEN, "invalid token CSR", error);
        }
        std::vector<uint8_t> ordinals(record.assignment_count, 0u);
        for (uint32_t local = 0u; local < record.assignment_count; ++local) {
            const AssignmentRecord &assignment =
                built.assignments[record.assignment_begin + local];
            if (assignment.destination_rank >= session.worker_count ||
                assignment.expert_id >= session.expert_count ||
                !std::isfinite(assignment.weight) ||
                assignment.ordinal >= record.assignment_count ||
                ordinals[assignment.ordinal] != 0u) {
                return Fail(Status::INVALID_ASSIGNMENT,
                            "invalid assignment metadata", error);
            }
            ordinals[assignment.ordinal] = 1u;
        }
        expected_assignment += record.assignment_count;
    }
    if (expected_assignment != header.assignment_count)
        return Fail(Status::INVALID_TOKEN,
                    "token CSR does not cover assignments", error);
    *parsed = std::move(built);
    if (error != nullptr) error->clear();
    return Status::OK;
}

Status CompileLayout(const std::vector<ParsedSource> &sources,
                     const std::vector<uint32_t> &arrival_order,
                     const LayoutConfig &config, CompiledLayout *layout,
                     std::string *error)
{
    if (layout == nullptr || config.worker_count < 2u ||
        config.worker_count > kPullDispatchMaxWorkers ||
        config.expert_count == 0u ||
        config.destination_row_capacity == 0u ||
        config.destination_assignment_capacity == 0u ||
        sources.size() != config.worker_count ||
        arrival_order.size() != config.worker_count) {
        return Fail(Status::INVALID_ARGUMENT, "invalid layout input", error);
    }

    std::vector<const ParsedSource *> by_rank(config.worker_count, nullptr);
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint32_t wave = 0u;
    uint16_t ring_slot = 0u;
    for (const ParsedSource &source : sources) {
        const uint32_t rank = source.header.source_rank;
        if (rank >= config.worker_count || by_rank[rank] != nullptr)
            return Fail(Status::DUPLICATE_SOURCE,
                        "duplicate or invalid source", error);
        by_rank[rank] = &source;
        if (generation == 0u) {
            generation = source.header.generation;
            sequence = source.header.sequence;
            wave = source.header.wave;
            ring_slot = source.header.ring_slot;
        } else if (source.header.generation != generation ||
                   source.header.sequence != sequence ||
                   source.header.wave != wave ||
                   source.header.ring_slot != ring_slot ||
                   source.header.worker_count != config.worker_count) {
            return Fail(Status::STALE_EPOCH,
                        "mixed source epochs", error);
        }
    }
    std::vector<uint8_t> seen(config.worker_count, 0u);
    for (uint32_t rank : arrival_order) {
        if (rank >= config.worker_count || seen[rank] != 0u)
            return Fail(Status::DUPLICATE_SOURCE,
                        "arrival order is not a permutation", error);
        seen[rank] = 1u;
    }

    CompiledLayout built{};
    built.destination_rows.resize(config.worker_count);
    built.expert_assignments.resize(config.worker_count);
    built.expert_counts.assign(
        static_cast<size_t>(config.worker_count) * config.expert_count, 0u);
    std::vector<std::vector<std::vector<JournalContributor>>>
        contributor_map(config.worker_count);
    for (uint32_t rank = 0u; rank < config.worker_count; ++rank)
        contributor_map[rank].resize(by_rank[rank]->tokens.size());

    for (uint32_t rank : arrival_order) {
        const ParsedSource &source = *by_rank[rank];
        for (uint32_t token = 0u; token < source.tokens.size(); ++token) {
            const TokenRecord &record = source.tokens[token];
            std::vector<std::vector<AssignmentRecord>> grouped(
                config.worker_count);
            for (uint32_t local = 0u; local < record.assignment_count;
                 ++local) {
                const AssignmentRecord &assignment =
                    source.assignments[record.assignment_begin + local];
                grouped[assignment.destination_rank].push_back(assignment);
            }
            for (uint32_t destination = 0u;
                 destination < config.worker_count; ++destination) {
                if (grouped[destination].empty()) continue;
                if (built.destination_rows[destination].size() >=
                        config.destination_row_capacity ||
                    built.expert_assignments[destination].size() +
                            grouped[destination].size() >
                        config.destination_assignment_capacity) {
                    return Fail(Status::CAPACITY_EXCEEDED,
                                "destination layout capacity exceeded",
                                error);
                }
                const uint32_t destination_row = static_cast<uint32_t>(
                    built.destination_rows[destination].size());
                const uint32_t assignment_begin = static_cast<uint32_t>(
                    built.expert_assignments[destination].size());
                for (const AssignmentRecord &assignment :
                     grouped[destination]) {
                    const uint64_t count_index =
                        static_cast<uint64_t>(destination) *
                            config.expert_count +
                        assignment.expert_id;
                    ExpertAssignment expert{};
                    expert.destination_row = destination_row;
                    expert.expert_id = assignment.expert_id;
                    expert.expert_row = built.expert_counts[count_index]++;
                    expert.ordinal = assignment.ordinal;
                    expert.weight = assignment.weight;
                    built.expert_assignments[destination].push_back(expert);
                }
                DestinationRow row{};
                row.route_key = RouteKey(rank, record.source_token);
                row.token_id = record.token_id;
                row.source_rank = rank;
                row.source_token = record.source_token;
                row.destination_row = destination_row;
                row.assignments_begin = assignment_begin;
                row.assignments_count = grouped[destination].size();
                built.destination_rows[destination].push_back(row);
                contributor_map[rank][token].push_back(
                    JournalContributor{destination, destination_row,
                                       assignment_begin,
                                       static_cast<uint32_t>(
                                           grouped[destination].size())});
            }
        }
    }

    uint64_t cookie = kHashOffset;
    HashValue(&cookie, generation);
    HashValue(&cookie, sequence);
    HashValue(&cookie, wave);
    for (uint32_t rank = 0u; rank < config.worker_count; ++rank) {
        const ParsedSource &source = *by_rank[rank];
        HashValue(&cookie, source.header.metadata_digest);
        for (uint32_t token = 0u; token < source.tokens.size(); ++token) {
            const TokenRecord &record = source.tokens[token];
            if (built.journal_tokens.size() >=
                    std::numeric_limits<uint32_t>::max() ||
                built.contributors.size() +
                        contributor_map[rank][token].size() >
                    std::numeric_limits<uint32_t>::max() ||
                built.journal_assignments.size() + record.assignment_count >
                    std::numeric_limits<uint32_t>::max()) {
                return Fail(Status::SIZE_OVERFLOW,
                            "journal size overflow", error);
            }
            JournalTokenEntry entry{};
            entry.route_key = RouteKey(rank, record.source_token);
            entry.token_id = record.token_id;
            entry.owner_rank = rank;
            entry.owner_row = record.source_token;
            entry.contributors_begin = static_cast<uint32_t>(
                built.contributors.size());
            entry.contributors_count = contributor_map[rank][token].size();
            entry.assignments_begin = static_cast<uint32_t>(
                built.journal_assignments.size());
            entry.assignments_count = record.assignment_count;
            entry.accumulator_index = static_cast<uint32_t>(
                built.journal_tokens.size());
            built.contributors.insert(built.contributors.end(),
                contributor_map[rank][token].begin(),
                contributor_map[rank][token].end());
            built.journal_assignments.insert(
                built.journal_assignments.end(),
                source.assignments.begin() + record.assignment_begin,
                source.assignments.begin() + record.assignment_begin +
                    record.assignment_count);
            built.journal_tokens.push_back(entry);
        }
    }

    built.journal_header.generation = generation;
    built.journal_header.sequence = sequence;
    built.journal_header.dispatch_cookie = cookie == 0u ? 1u : cookie;
    built.journal_header.wave = wave;
    built.journal_header.ring_slot = ring_slot;
    built.journal_header.state = static_cast<uint16_t>(
        JournalSlotState::DISPATCH_SEALED);
    built.journal_header.token_count = built.journal_tokens.size();
    built.journal_header.contributor_count = built.contributors.size();
    *layout = std::move(built);
    if (error != nullptr) error->clear();
    return Status::OK;
}

bool ValidJournalTransition(JournalSlotState from, JournalSlotState to)
{
    switch (from) {
        case JournalSlotState::FREE:
            return to == JournalSlotState::DISPATCH_OPEN;
        case JournalSlotState::DISPATCH_OPEN:
            return to == JournalSlotState::DISPATCH_SEALED ||
                to == JournalSlotState::ABORTED;
        case JournalSlotState::DISPATCH_SEALED:
            return to == JournalSlotState::COMBINE_ACTIVE ||
                to == JournalSlotState::ABORTED;
        case JournalSlotState::COMBINE_ACTIVE:
            return to == JournalSlotState::COMPLETE ||
                to == JournalSlotState::ABORTED;
        case JournalSlotState::COMPLETE:
        case JournalSlotState::ABORTED:
            return to == JournalSlotState::FREE;
    }
    return false;
}

Status TransitionJournal(JournalSlotHeader *header, JournalSlotState to,
                         std::string *error)
{
    if (header == nullptr || header->magic != kPullDispatchMagic ||
        header->abi_version != kPullDispatchAbiVersion ||
        header->struct_bytes != sizeof(JournalSlotHeader)) {
        return Fail(Status::INVALID_ARGUMENT,
                    "invalid journal header", error);
    }
    const JournalSlotState from = static_cast<JournalSlotState>(
        header->state);
    if (!ValidJournalTransition(from, to))
        return Fail(Status::INVALID_STATE_TRANSITION,
                    "invalid journal state transition", error);
    header->state = static_cast<uint16_t>(to);
    if (error != nullptr) error->clear();
    return Status::OK;
}

const char *StatusString(Status status)
{
    switch (status) {
        case Status::OK: return "ok";
        case Status::INVALID_ARGUMENT: return "invalid_argument";
        case Status::INVALID_REGISTRATION: return "invalid_registration";
        case Status::INVALID_READY: return "invalid_ready";
        case Status::INVALID_HEADER: return "invalid_header";
        case Status::STALE_EPOCH: return "stale_epoch";
        case Status::SIZE_OVERFLOW: return "size_overflow";
        case Status::CAPACITY_EXCEEDED: return "capacity_exceeded";
        case Status::DIGEST_MISMATCH: return "digest_mismatch";
        case Status::INVALID_TOKEN: return "invalid_token";
        case Status::INVALID_ASSIGNMENT: return "invalid_assignment";
        case Status::DUPLICATE_SOURCE: return "duplicate_source";
        case Status::INVALID_STATE_TRANSITION:
            return "invalid_state_transition";
    }
    return "unknown";
}

} // namespace inc::dc::pull_v2
