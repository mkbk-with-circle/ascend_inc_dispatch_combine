#include "inc_dc_combine_logical_plan.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace inc {
namespace dc {
namespace {

uint64_t Mix(uint64_t h, uint64_t v)
{
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}

void FinalizeUniform(IncDcCombineLogicalPlanV2 *plan)
{
    if (plan->results.empty()) {
        plan->uniform_topk_valid = true;
        plan->uniform_topk = 0;
        return;
    }
    const uint32_t first = plan->results[0].contribution_count;
    plan->uniform_topk_valid = true;
    for (const auto &r : plan->results) {
        if (r.contribution_count != first) {
            plan->uniform_topk_valid = false;
            plan->uniform_topk = 0;
            return;
        }
    }
    plan->uniform_topk = first;
}

} // namespace

uint64_t ComputeLogicalPlanSemanticDigest(const IncDcCombineLogicalPlanV2 &plan)
{
    uint64_t h = 0x44494e4c50563201ull; // 'DINLPV2\x01'
    h = Mix(h, plan.abi_version);
    h = Mix(h, plan.worker_world_size);
    h = Mix(h, plan.result_count);
    h = Mix(h, plan.contribution_count);
    h = Mix(h, plan.declared_max_topk);
    for (const auto &r : plan.results) {
        h = Mix(h, r.dst_rank);
        h = Mix(h, r.dst_local_row);
        h = Mix(h, r.contribution_begin);
        h = Mix(h, r.contribution_count);
    }
    for (const auto &c : plan.contributions) {
        h = Mix(h, c.contribution_uid);
        h = Mix(h, c.result_id);
        h = Mix(h, c.ordinal);
        h = Mix(h, c.contributor_rank);
        h = Mix(h, c.contributor_local_row);
        uint32_t wb = 0;
        static_assert(sizeof(float) == 4, "float");
        std::memcpy(&wb, &c.weight, sizeof(wb));
        h = Mix(h, wb);
    }
    return h == 0 ? 1 : h;
}

IncDcStatus BuildSyntheticLogicalPlanV2(uint32_t worker_world_size,
                                        uint32_t declared_max_topk,
                                        uint32_t result_count, uint32_t mode,
                                        IncDcCombineLogicalPlanV2 *out)
{
    if (out == nullptr || worker_world_size == 0 ||
        declared_max_topk == 0 || result_count == 0) {
        return IncDcStatus::INVALID_ARGUMENT;
    }
    out->abi_version = kIncDcCombineLogicalPlanAbiV2;
    out->worker_world_size = worker_world_size;
    out->declared_max_topk = declared_max_topk;
    out->results.clear();
    out->contributions.clear();
    out->results.reserve(result_count);
    const uint64_t maximum_contributions =
        static_cast<uint64_t>(result_count) * declared_max_topk;
    if (maximum_contributions > std::numeric_limits<uint32_t>::max()) {
        return IncDcStatus::CAPACITY_EXCEEDED;
    }
    out->contributions.reserve(
        static_cast<size_t>(maximum_contributions));

    uint64_t uid = 1;
    for (uint32_t r = 0; r < result_count; ++r) {
        uint32_t k = declared_max_topk;
        if (mode == 2u) {
            k = 1u + (r % declared_max_topk);
        }
        IncDcLogicalResultV2 res{};
        res.dst_rank = r % worker_world_size;
        res.dst_local_row = r / worker_world_size;
        res.contribution_begin = static_cast<uint32_t>(out->contributions.size());
        res.contribution_count = k;

        for (uint32_t o = 0; o < k; ++o) {
            IncDcLogicalContributionV2 c{};
            c.contribution_uid = uid++;
            c.result_id = r;
            c.ordinal = o;
            if (mode == 1u && worker_world_size >= 2u) {
                // Concentrate many ordinals on rank0; leave last rank unused
                // when possible (zero-contribution worker).
                if (o == 0u) {
                    c.contributor_rank = 0;
                } else if (o == 1u && worker_world_size > 2u) {
                    c.contributor_rank = 1;
                } else {
                    c.contributor_rank = 0;
                }
                // Never use worker_world_size-1 in mode1 when W>=2
                if (c.contributor_rank == worker_world_size - 1u) {
                    c.contributor_rank = 0;
                }
            } else {
                // Per-result top-k is independent of the number of workers
                // participating across the batch.  Rotate the starting rank
                // by result so W8/K1, K2, K4, K6 all exercise all eight
                // workers instead of incorrectly treating K as active ranks.
                // K>W remains supported via wrap and is intentionally allowed
                // to place multiple ordinals from one result on one worker.
                c.contributor_rank = (r + o) % worker_world_size;
            }
            c.contributor_local_row = r + o * 17u;
            c.weight = 1.0f;
            out->contributions.push_back(c);
        }
        out->results.push_back(res);
    }
    out->result_count = static_cast<uint32_t>(out->results.size());
    out->contribution_count =
        static_cast<uint32_t>(out->contributions.size());
    FinalizeUniform(out);
    out->semantic_digest = ComputeLogicalPlanSemanticDigest(*out);
    return IncDcStatus::OK;
}

IncDcStatus ValidateLogicalPlanV2(const IncDcCombineLogicalPlanV2 &plan,
                                  IncDcLogicalPlanValidateReport *report)
{
    IncDcLogicalPlanValidateReport local{};
    auto *rep = report ? report : &local;
    rep->ok = false;
    rep->first_error.clear();
    rep->contributions_per_worker.assign(plan.worker_world_size, 0);
    rep->zero_contribution_workers.clear();

    auto fail = [&](const std::string &msg) -> IncDcStatus {
        rep->first_error = msg;
        return IncDcStatus::INVALID_ARGUMENT;
    };

    if (plan.abi_version != kIncDcCombineLogicalPlanAbiV2) {
        return fail("bad_abi_version");
    }
    if (plan.worker_world_size == 0) return fail("worker_world_size_zero");
    if (plan.results.size() != plan.result_count) return fail("result_count_mismatch");
    if (plan.contributions.size() != plan.contribution_count) {
        return fail("contribution_count_mismatch");
    }
    if (plan.semantic_digest != ComputeLogicalPlanSemanticDigest(plan)) {
        return fail("semantic_digest_mismatch");
    }

    uint32_t emin = std::numeric_limits<uint32_t>::max(), emax = 0;
    // contribution_uid is a global operation identity (not per-result).
    std::unordered_set<uint64_t> global_uids;
    for (uint32_t ri = 0; ri < plan.result_count; ++ri) {
        const auto &r = plan.results[ri];
        if (r.dst_rank >= plan.worker_world_size) return fail("dst_rank_oob");
        if (r.contribution_count > plan.declared_max_topk) {
            return fail("expected_count_gt_declared_max_topk");
        }
        if (static_cast<uint64_t>(r.contribution_begin) + r.contribution_count >
            plan.contribution_count) {
            return fail("contrib_range_oob");
        }
        emin = std::min(emin, r.contribution_count);
        emax = std::max(emax, r.contribution_count);

        std::unordered_set<uint32_t> ords;
        for (uint32_t i = 0; i < r.contribution_count; ++i) {
            const auto &c = plan.contributions[r.contribution_begin + i];
            if (c.result_id != ri) return fail("contrib_result_id_mismatch");
            if (c.ordinal != i && c.ordinal >= r.contribution_count) {
                return fail("ordinal_oob");
            }
            if (!ords.insert(c.ordinal).second) return fail("duplicate_ordinal");
            if (c.ordinal >= r.contribution_count) return fail("ordinal_oob");
            if (c.contribution_uid == 0ull) return fail("uid_zero");
            if (!global_uids.insert(c.contribution_uid).second) {
                return fail("duplicate_uid_global");
            }
            if (c.contributor_rank >= plan.worker_world_size) {
                return fail("contributor_rank_oob");
            }
            rep->contributions_per_worker[c.contributor_rank]++;
        }
        // Require dense ordinal cover [0, expected)
        if (ords.size() != r.contribution_count) return fail("ordinal_not_dense");
        for (uint32_t o = 0; o < r.contribution_count; ++o) {
            if (!ords.count(o)) return fail("missing_ordinal");
        }
    }
    if (plan.result_count == 0) {
        emin = 0;
        emax = 0;
    }
    rep->expected_count_min =
        emin == std::numeric_limits<uint32_t>::max() ? 0 : emin;
    rep->expected_count_max = emax;
    for (uint32_t w = 0; w < plan.worker_world_size; ++w) {
        if (rep->contributions_per_worker[w] == 0) {
            rep->zero_contribution_workers.push_back(w);
        }
    }
    rep->ok = true;
    return IncDcStatus::OK;
}

uint64_t LogicalUsefulBytes(const IncDcCombineLogicalPlanV2 &plan,
                            uint32_t hidden, uint32_t input_elem_bytes,
                            uint32_t output_elem_bytes)
{
    const uint64_t in_b =
        static_cast<uint64_t>(plan.contribution_count) * hidden * input_elem_bytes;
    const uint64_t out_b =
        static_cast<uint64_t>(plan.result_count) * hidden * output_elem_bytes;
    return in_b + out_b;
}

} // namespace dc
} // namespace inc
