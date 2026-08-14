#include "inc_fusion_route.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <tuple>

namespace inc::fusion {
namespace {

bool Fail(std::string *error, const char *message)
{
    if (error != nullptr) *error = message;
    return false;
}

uint32_t FloatBits(float value)
{
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

struct Pending {
    uint32_t source = 0u;
    uint32_t token = 0u;
    uint32_t ordinal = 0u;
    uint32_t expert = 0u;
    uint32_t destination = 0u;
    uint32_t local_expert = 0u;
    uint32_t destination_row = 0u;
    uint32_t weight_bits = 0u;
};

} // namespace

bool CompileFusionRouteActive(const FusionPlan &plan,
                              const uint32_t *expert_ids,
                              const float *weights,
                              const uint32_t *active_token_counts,
                              FusionRouteBundle *bundle,
                              std::string *error)
{
    if (expert_ids == nullptr || weights == nullptr ||
        active_token_counts == nullptr || bundle == nullptr)
        return Fail(error, "null route input or output");
    const FusionPlanConfig &config = plan.config;
    if (plan.waves.empty() || plan.expert_owner.size() != config.expert_count ||
        plan.expert_local_index.size() != config.expert_count)
        return Fail(error, "incomplete fusion plan");

    FusionRouteBundle built{};
    built.ranks.resize(config.worker_count);
    built.group_lists.resize(config.worker_count);
    for (uint32_t worker = 0u; worker < config.worker_count; ++worker)
        built.group_lists[worker].resize(plan.waves.size());

    const uint64_t rank_stride =
        static_cast<uint64_t>(config.token_count) * config.topk;
    uint32_t collective_token_count = 0u;
    for (uint32_t source = 0u; source < config.worker_count; ++source) {
        if (active_token_counts[source] > config.token_count)
            return Fail(error, "active token count exceeds plan capacity");
        collective_token_count = std::max(
            collective_token_count, active_token_counts[source]);
    }
    for (uint32_t wave = 0u; wave < plan.waves.size(); ++wave) {
        FusionWaveDesc base_wave = plan.waves[wave];
        base_wave.token_count = collective_token_count > base_wave.token_begin
            ? std::min(config.tokens_per_wave,
                       collective_token_count - base_wave.token_begin)
            : 0u;
        std::vector<Pending> pending;
        pending.reserve(static_cast<size_t>(config.worker_count) *
                        base_wave.token_count * config.topk);
        for (uint32_t source = 0u; source < config.worker_count; ++source) {
            const uint32_t source_wave_tokens =
                active_token_counts[source] > base_wave.token_begin
                    ? std::min(config.tokens_per_wave,
                        active_token_counts[source] - base_wave.token_begin)
                    : 0u;
            for (uint32_t local_token = 0u;
                 local_token < source_wave_tokens; ++local_token) {
                const uint32_t token = base_wave.token_begin + local_token;
                for (uint32_t ordinal = 0u; ordinal < config.topk; ++ordinal) {
                    const uint64_t index =
                        static_cast<uint64_t>(source) * rank_stride +
                        static_cast<uint64_t>(token) * config.topk + ordinal;
                    const uint32_t expert = expert_ids[index];
                    if (expert >= config.expert_count)
                        return Fail(error, "expert id is outside plan");
                    const float weight = weights[index];
                    if (!(weight == weight) ||
                        weight > std::numeric_limits<float>::max() ||
                        weight < -std::numeric_limits<float>::max())
                        return Fail(error, "route weight is not finite");
                    Pending item{};
                    item.source = source;
                    item.token = token;
                    item.ordinal = ordinal;
                    item.expert = expert;
                    item.destination = plan.expert_owner[expert];
                    item.local_expert = plan.expert_local_index[expert];
                    item.weight_bits = FloatBits(weight);
                    pending.push_back(item);
                }
            }
        }

        // Destination rows are expert-major, so the device can feed grouped
        // GEMM directly without a post-Dispatch sort or a shape-specific path.
        for (uint32_t destination = 0u;
             destination < config.worker_count; ++destination) {
            const uint32_t local_count = plan.local_expert_counts[destination];
            std::vector<int64_t> groups(local_count, 0);
            uint32_t next_row = 0u;
            for (uint32_t local_expert = 0u;
                 local_expert < local_count; ++local_expert) {
                for (Pending &item : pending) {
                    if (item.destination != destination ||
                        item.local_expert != local_expert)
                        continue;
                    item.destination_row = next_row++;
                }
                groups[local_expert] = next_row;
            }
            if (next_row > plan.max_assignments_per_wave)
                return Fail(error, "compiled destination exceeds workspace");
            built.group_lists[destination][wave] = std::move(groups);
        }

        for (uint32_t source = 0u; source < config.worker_count; ++source) {
            FusionRankRoute &rank = built.ranks[source];
            FusionWaveDesc desc = base_wave;
            desc.dispatch_row_begin =
                static_cast<uint32_t>(rank.dispatch_rows.size());
            desc.assignment_begin =
                static_cast<uint32_t>(rank.assignments.size());
            const uint32_t source_wave_tokens =
                active_token_counts[source] > base_wave.token_begin
                    ? std::min(config.tokens_per_wave,
                        active_token_counts[source] - base_wave.token_begin)
                    : 0u;
            for (uint32_t local_token = 0u;
                 local_token < source_wave_tokens; ++local_token) {
                const uint32_t token = base_wave.token_begin + local_token;
                // std::map provides deterministic destination-major order and
                // naturally implements physical row deduplication.
                std::map<uint32_t, std::vector<Pending *>> destinations;
                for (Pending &item : pending) {
                    if (item.source == source && item.token == token)
                        destinations[item.destination].push_back(&item);
                }
                for (auto &entry : destinations) {
                    FusionDispatchRow row{};
                    row.source_rank = source;
                    row.source_token = token;
                    row.destination_rank = entry.first;
                    row.assignment_begin =
                        static_cast<uint32_t>(rank.assignments.size());
                    row.assignment_count =
                        static_cast<uint32_t>(entry.second.size());
                    row.wave = wave;
                    row.payload_byte_offset =
                        static_cast<uint64_t>(token) * config.hidden *
                        config.element_bytes;
                    const uint32_t row_index =
                        static_cast<uint32_t>(rank.dispatch_rows.size());
                    rank.dispatch_rows.push_back(row);
                    for (Pending *item : entry.second) {
                        FusionExpertAssignment assignment{};
                        assignment.dispatch_row = row_index;
                        assignment.expert_id = item->expert;
                        assignment.local_expert = item->local_expert;
                        assignment.route_ordinal = item->ordinal;
                        assignment.destination_token = item->token;
                        assignment.weight_bits = item->weight_bits;
                        assignment.wave = wave;
                        assignment.destination_row = item->destination_row;
                        rank.assignments.push_back(assignment);
                    }
                }
            }
            desc.dispatch_row_count =
                static_cast<uint32_t>(rank.dispatch_rows.size()) -
                desc.dispatch_row_begin;
            desc.assignment_count =
                static_cast<uint32_t>(rank.assignments.size()) -
                desc.assignment_begin;
            if (desc.dispatch_row_count >
                    plan.max_source_dispatch_rows_per_wave ||
                desc.assignment_count >
                    plan.max_source_assignments_per_wave)
                return Fail(error, "compiled source route exceeds plan");
            rank.waves.push_back(desc);
        }
    }
    *bundle = std::move(built);
    if (error != nullptr) error->clear();
    return true;
}

bool CompileFusionRoute(const FusionPlan &plan,
                        const uint32_t *expert_ids,
                        const float *weights,
                        FusionRouteBundle *bundle,
                        std::string *error)
{
    std::vector<uint32_t> active(
        plan.config.worker_count, plan.config.token_count);
    return CompileFusionRouteActive(plan, expert_ids, weights, active.data(),
                                    bundle, error);
}

} // namespace inc::fusion
