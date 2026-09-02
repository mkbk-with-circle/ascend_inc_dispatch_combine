#ifndef INC_DC_PULL_COMBINE_PLAN_H
#define INC_DC_PULL_COMBINE_PLAN_H

#include <cstdint>
#include <string>
#include <vector>

namespace inc::dc::pull_combine {

enum class PlanStatus : uint32_t {
    OK = 0u,
    INVALID_ARGUMENT,
    INVALID_ROUTE,
    DUPLICATE_ORDINAL,
    NONFINITE_WEIGHT,
    CAPACITY_EXCEEDED,
};

struct RouteAssignment {
    uint32_t destination_rank = 0u;
    uint32_t expert_id = 0u;
    uint32_t ordinal = 0u;
    float weight = 1.0f;
};

struct TokenRoute {
    uint32_t source_rank = 0u;
    uint32_t source_token = 0u;
    std::vector<RouteAssignment> assignments;
};

struct WavePlanDesc {
    uint32_t worker_count = 0u;
    uint32_t expert_count = 0u;
    uint32_t hidden = 0u;
    uint32_t element_bytes = 2u;
    uint32_t wave = 0u;
    uint64_t generation = 0u;
    std::vector<uint32_t> tokens_per_rank;
    std::vector<TokenRoute> routes;
};

struct RouteSlot {
    uint64_t token_key = 0u;
    uint32_t result_id = 0u;
    uint32_t source_rank = 0u;
    uint32_t source_token = 0u;
    uint32_t destination_rank = 0u;
    uint32_t destination_row = 0u;
    uint32_t assignment_begin = 0u;
    uint32_t assignment_count = 0u;
};

struct PackedAssignment {
    uint32_t route_slot = 0u;
    uint32_t expert_id = 0u;
    uint32_t ordinal = 0u;
    float weight = 1.0f;
};

// Combine rows are packed source-B first, destination-A second, then in the
// deterministic Dispatch route-slot order.  A descriptor from B therefore
// names a contiguous range without repeating TokenKey/token IDs on the wire.
struct CombineRow {
    uint32_t route_slot = 0u;
    uint32_t result_id = 0u;
    uint32_t source_rank = 0u; // expert worker B
    uint32_t destination_rank = 0u; // original token owner A
    uint32_t destination_row = 0u;
};

struct WavePlan {
    uint32_t worker_count = 0u;
    uint32_t expert_count = 0u;
    uint32_t hidden = 0u;
    uint32_t element_bytes = 0u;
    uint32_t wave = 0u;
    uint64_t generation = 0u;
    uint64_t semantic_digest = 0u;
    uint64_t dispatch_payload_bytes = 0u;
    uint64_t strict_partial_bytes = 0u;

    std::vector<uint32_t> tokens_per_rank;
    std::vector<uint64_t> result_rank_offsets;
    std::vector<uint32_t> expected_contributors;
    std::vector<uint64_t> token_row_counts; // [A][B]
    std::vector<uint64_t> assignment_counts; // [A][B]
    std::vector<uint64_t> dispatch_receive_offsets; // [B][A], rows
    std::vector<uint64_t> combine_segment_offsets; // [B][A], packed rows
    std::vector<uint64_t> combine_source_offsets; // [B+1]
    std::vector<RouteSlot> route_slots;
    std::vector<PackedAssignment> assignments;
    std::vector<CombineRow> combine_rows;
};

PlanStatus BuildWavePlan(const WavePlanDesc &desc, WavePlan *plan,
                         std::string *error = nullptr);
const char *PlanStatusString(PlanStatus status);

} // namespace inc::dc::pull_combine

#endif
