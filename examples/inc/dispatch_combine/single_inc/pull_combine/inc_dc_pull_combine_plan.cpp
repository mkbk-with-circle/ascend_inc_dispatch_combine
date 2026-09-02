#include "inc_dc_pull_combine_plan.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>

namespace inc::dc::pull_combine {
namespace {

constexpr uint64_t kFnvOffset = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

bool Add(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr || a > std::numeric_limits<uint64_t>::max() - b)
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

uint64_t MatrixIndex(uint32_t row, uint32_t column, uint32_t width)
{
    return static_cast<uint64_t>(row) * width + column;
}

void DigestBytes(uint64_t *digest, const void *data, size_t bytes)
{
    const auto *p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        *digest ^= p[i];
        *digest *= kFnvPrime;
    }
}

template <typename T>
void DigestValue(uint64_t *digest, const T &value)
{
    DigestBytes(digest, &value, sizeof(value));
}

PlanStatus Fail(PlanStatus status, const char *message, std::string *error)
{
    if (error != nullptr) *error = message;
    return status;
}

} // namespace

PlanStatus BuildWavePlan(const WavePlanDesc &desc, WavePlan *plan,
                         std::string *error)
{
    if (plan == nullptr || desc.worker_count == 0u ||
        desc.worker_count > 128u || desc.expert_count == 0u ||
        desc.hidden == 0u ||
        (desc.element_bytes != 2u && desc.element_bytes != 4u) ||
        desc.generation == 0u ||
        desc.tokens_per_rank.size() != desc.worker_count) {
        return Fail(PlanStatus::INVALID_ARGUMENT,
                    "invalid wave shape or generation", error);
    }

    uint64_t total_tokens = 0u;
    for (uint32_t tokens : desc.tokens_per_rank) {
        if (!Add(total_tokens, tokens, &total_tokens) ||
            total_tokens > std::numeric_limits<uint32_t>::max()) {
            return Fail(PlanStatus::CAPACITY_EXCEEDED,
                        "result id space exceeded", error);
        }
    }
    if (desc.routes.size() != total_tokens) {
        return Fail(PlanStatus::INVALID_ROUTE,
                    "routes must contain exactly one record per token", error);
    }

    WavePlan built{};
    built.worker_count = desc.worker_count;
    built.expert_count = desc.expert_count;
    built.hidden = desc.hidden;
    built.element_bytes = desc.element_bytes;
    built.wave = desc.wave;
    built.generation = desc.generation;
    built.tokens_per_rank = desc.tokens_per_rank;
    built.result_rank_offsets.resize(desc.worker_count + 1u, 0u);
    for (uint32_t rank = 0u; rank < desc.worker_count; ++rank) {
        built.result_rank_offsets[rank + 1u] =
            built.result_rank_offsets[rank] + desc.tokens_per_rank[rank];
    }
    built.expected_contributors.assign(total_tokens, 0u);

    uint64_t matrix_elements = 0u;
    if (!Mul(desc.worker_count, desc.worker_count, &matrix_elements) ||
        matrix_elements > std::numeric_limits<size_t>::max()) {
        return Fail(PlanStatus::CAPACITY_EXCEEDED,
                    "worker matrix capacity exceeded", error);
    }
    built.token_row_counts.assign(matrix_elements, 0u);
    built.assignment_counts.assign(matrix_elements, 0u);
    built.dispatch_receive_offsets.assign(matrix_elements, 0u);
    built.combine_segment_offsets.assign(matrix_elements, 0u);

    std::vector<const TokenRoute *> ordered;
    ordered.reserve(desc.routes.size());
    std::vector<uint8_t> token_seen(total_tokens, 0u);
    for (const TokenRoute &route : desc.routes) {
        if (route.source_rank >= desc.worker_count ||
            route.source_token >= desc.tokens_per_rank[route.source_rank]) {
            return Fail(PlanStatus::INVALID_ROUTE,
                        "token owner or local row is out of range", error);
        }
        const uint64_t result_id =
            built.result_rank_offsets[route.source_rank] +
            route.source_token;
        if (token_seen[result_id] != 0u) {
            return Fail(PlanStatus::INVALID_ROUTE,
                        "duplicate token route record", error);
        }
        token_seen[result_id] = 1u;
        ordered.push_back(&route);
    }
    if (std::find(token_seen.begin(), token_seen.end(), 0u) !=
        token_seen.end()) {
        return Fail(PlanStatus::INVALID_ROUTE,
                    "missing token route record", error);
    }
    std::sort(ordered.begin(), ordered.end(), [](const TokenRoute *a,
                                                  const TokenRoute *b) {
        return std::pair<uint32_t, uint32_t>(a->source_rank,
                                             a->source_token) <
               std::pair<uint32_t, uint32_t>(b->source_rank,
                                             b->source_token);
    });

    // Pass one validates semantic identity and computes the two independent
    // matrices: unique hidden rows and expert assignments.
    for (const TokenRoute *route : ordered) {
        std::vector<uint32_t> ordinals;
        std::vector<uint8_t> destination_seen(desc.worker_count, 0u);
        ordinals.reserve(route->assignments.size());
        for (const RouteAssignment &assignment : route->assignments) {
            if (assignment.destination_rank >= desc.worker_count ||
                assignment.expert_id >= desc.expert_count) {
                return Fail(PlanStatus::INVALID_ROUTE,
                            "assignment rank or expert is out of range", error);
            }
            if (!std::isfinite(assignment.weight)) {
                return Fail(PlanStatus::NONFINITE_WEIGHT,
                            "route weight must be finite", error);
            }
            ordinals.push_back(assignment.ordinal);
            const uint64_t pair = MatrixIndex(
                route->source_rank, assignment.destination_rank,
                desc.worker_count);
            ++built.assignment_counts[pair];
            if (destination_seen[assignment.destination_rank] == 0u) {
                destination_seen[assignment.destination_rank] = 1u;
                ++built.token_row_counts[pair];
            }
        }
        std::sort(ordinals.begin(), ordinals.end());
        if (std::adjacent_find(ordinals.begin(), ordinals.end()) !=
            ordinals.end()) {
            return Fail(PlanStatus::DUPLICATE_ORDINAL,
                        "top-k ordinal is duplicated for one token", error);
        }
        const uint64_t result_id =
            built.result_rank_offsets[route->source_rank] +
            route->source_token;
        built.expected_contributors[result_id] = static_cast<uint32_t>(
            std::count(destination_seen.begin(), destination_seen.end(), 1u));
    }

    // Destination B receives source ranks in deterministic A order.
    for (uint32_t b = 0u; b < desc.worker_count; ++b) {
        uint64_t prefix = 0u;
        for (uint32_t a = 0u; a < desc.worker_count; ++a) {
            const uint64_t receive_index = MatrixIndex(b, a,
                                                       desc.worker_count);
            const uint64_t count_index = MatrixIndex(a, b,
                                                      desc.worker_count);
            built.dispatch_receive_offsets[receive_index] = prefix;
            if (!Add(prefix, built.token_row_counts[count_index], &prefix)) {
                return Fail(PlanStatus::CAPACITY_EXCEEDED,
                            "dispatch receive rows overflow", error);
            }
        }
    }

    std::vector<uint64_t> next_receive = built.dispatch_receive_offsets;
    std::vector<std::vector<uint32_t>> pair_slots(matrix_elements);
    uint64_t route_slot_capacity = std::accumulate(
        built.token_row_counts.begin(), built.token_row_counts.end(),
        uint64_t{0});
    if (route_slot_capacity > std::numeric_limits<uint32_t>::max()) {
        return Fail(PlanStatus::CAPACITY_EXCEEDED,
                    "route slot id space exceeded", error);
    }
    built.route_slots.reserve(static_cast<size_t>(route_slot_capacity));
    const uint64_t assignment_capacity = std::accumulate(
        built.assignment_counts.begin(), built.assignment_counts.end(),
        uint64_t{0});
    if (assignment_capacity > std::numeric_limits<uint32_t>::max()) {
        return Fail(PlanStatus::CAPACITY_EXCEEDED,
                    "assignment id space exceeded", error);
    }
    built.assignments.reserve(static_cast<size_t>(assignment_capacity));

    // Pass two emits one route slot per unique destination rank.  Assignment
    // order is destination then ordinal, independent of input ordering.
    for (const TokenRoute *route : ordered) {
        std::vector<RouteAssignment> assignments = route->assignments;
        std::sort(assignments.begin(), assignments.end(),
                  [](const RouteAssignment &a, const RouteAssignment &b) {
            if (a.destination_rank != b.destination_rank)
                return a.destination_rank < b.destination_rank;
            return a.ordinal < b.ordinal;
        });
        size_t begin = 0u;
        while (begin < assignments.size()) {
            size_t end = begin + 1u;
            while (end < assignments.size() &&
                   assignments[end].destination_rank ==
                       assignments[begin].destination_rank)
                ++end;
            const uint32_t b = assignments[begin].destination_rank;
            const uint64_t receive_index = MatrixIndex(b, route->source_rank,
                                                       desc.worker_count);
            const uint64_t pair = MatrixIndex(route->source_rank, b,
                                              desc.worker_count);
            if (next_receive[receive_index] >
                std::numeric_limits<uint32_t>::max()) {
                return Fail(PlanStatus::CAPACITY_EXCEEDED,
                            "destination row id space exceeded", error);
            }
            RouteSlot slot{};
            slot.token_key = (static_cast<uint64_t>(route->source_rank) << 32u) |
                             route->source_token;
            slot.result_id = static_cast<uint32_t>(
                built.result_rank_offsets[route->source_rank] +
                route->source_token);
            slot.source_rank = route->source_rank;
            slot.source_token = route->source_token;
            slot.destination_rank = b;
            slot.destination_row = static_cast<uint32_t>(
                next_receive[receive_index]++);
            slot.assignment_begin = static_cast<uint32_t>(
                built.assignments.size());
            slot.assignment_count = static_cast<uint32_t>(end - begin);
            const uint32_t slot_id = static_cast<uint32_t>(
                built.route_slots.size());
            built.route_slots.push_back(slot);
            pair_slots[pair].push_back(slot_id);
            for (size_t i = begin; i < end; ++i) {
                built.assignments.push_back(PackedAssignment{
                    slot_id, assignments[i].expert_id,
                    assignments[i].ordinal, assignments[i].weight});
            }
            begin = end;
        }
    }

    built.combine_source_offsets.resize(desc.worker_count + 1u, 0u);
    uint64_t combine_cursor = 0u;
    for (uint32_t b = 0u; b < desc.worker_count; ++b) {
        built.combine_source_offsets[b] = combine_cursor;
        for (uint32_t a = 0u; a < desc.worker_count; ++a) {
            const uint64_t segment = MatrixIndex(b, a, desc.worker_count);
            const uint64_t pair = MatrixIndex(a, b, desc.worker_count);
            built.combine_segment_offsets[segment] = combine_cursor;
            for (uint32_t slot_id : pair_slots[pair]) {
                const RouteSlot &slot = built.route_slots[slot_id];
                built.combine_rows.push_back(CombineRow{
                    slot_id, slot.result_id, b, a, slot.source_token});
                ++combine_cursor;
            }
        }
    }
    built.combine_source_offsets[desc.worker_count] = combine_cursor;
    if (combine_cursor != built.route_slots.size()) {
        return Fail(PlanStatus::INVALID_ROUTE,
                    "combine inverse does not cover every route slot", error);
    }

    uint64_t hidden_bytes = 0u;
    if (!Mul(desc.hidden, desc.element_bytes, &hidden_bytes) ||
        !Mul(total_tokens, hidden_bytes, &built.dispatch_payload_bytes) ||
        !Mul(built.route_slots.size(), desc.hidden,
             &built.strict_partial_bytes) ||
        !Mul(built.strict_partial_bytes, sizeof(float),
             &built.strict_partial_bytes)) {
        return Fail(PlanStatus::CAPACITY_EXCEEDED,
                    "payload byte count overflow", error);
    }

    uint64_t digest = kFnvOffset;
    DigestValue(&digest, built.worker_count);
    DigestValue(&digest, built.expert_count);
    DigestValue(&digest, built.hidden);
    DigestValue(&digest, built.element_bytes);
    DigestValue(&digest, built.wave);
    DigestValue(&digest, built.generation);
    for (uint32_t value : built.tokens_per_rank) DigestValue(&digest, value);
    // Hash fields individually.  Hashing whole structs would include
    // indeterminate padding and make the semantic digest compiler-dependent.
    for (const RouteSlot &slot : built.route_slots) {
        DigestValue(&digest, slot.token_key);
        DigestValue(&digest, slot.result_id);
        DigestValue(&digest, slot.source_rank);
        DigestValue(&digest, slot.source_token);
        DigestValue(&digest, slot.destination_rank);
        DigestValue(&digest, slot.destination_row);
        DigestValue(&digest, slot.assignment_begin);
        DigestValue(&digest, slot.assignment_count);
    }
    for (const PackedAssignment &assignment : built.assignments) {
        DigestValue(&digest, assignment.route_slot);
        DigestValue(&digest, assignment.expert_id);
        DigestValue(&digest, assignment.ordinal);
        DigestValue(&digest, assignment.weight);
    }
    built.semantic_digest = digest;
    if (built.semantic_digest == 0u) built.semantic_digest = 1u;

    *plan = std::move(built);
    if (error != nullptr) error->clear();
    return PlanStatus::OK;
}

const char *PlanStatusString(PlanStatus status)
{
    switch (status) {
        case PlanStatus::OK: return "OK";
        case PlanStatus::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case PlanStatus::INVALID_ROUTE: return "INVALID_ROUTE";
        case PlanStatus::DUPLICATE_ORDINAL: return "DUPLICATE_ORDINAL";
        case PlanStatus::NONFINITE_WEIGHT: return "NONFINITE_WEIGHT";
        case PlanStatus::CAPACITY_EXCEEDED: return "CAPACITY_EXCEEDED";
    }
    return "UNKNOWN";
}

} // namespace inc::dc::pull_combine
