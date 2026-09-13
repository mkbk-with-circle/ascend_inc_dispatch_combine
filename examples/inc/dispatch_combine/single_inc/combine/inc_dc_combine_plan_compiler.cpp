#include "inc_dc_combine_plan_compiler.h"

#include "inc_dc_checked_arith.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace inc {
namespace dc {
namespace {

uint64_t Mix(uint64_t h, uint64_t v)
{
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}

} // namespace

uint64_t ComputeExecutionDigest(const IncDcCompiledExecutionPlan &plan)
{
    uint64_t h = 0x45584543504c4e02ull; // EXEPLN\x02
    h = Mix(h, plan.semantic_digest);
    h = Mix(h, plan.topology_digest);
    h = Mix(h, plan.hidden);
    h = Mix(h, plan.element_bytes);
    h = Mix(h, plan.row_bytes);
    h = Mix(h, static_cast<uint64_t>(plan.schedule.size()));
    for (const auto &c : plan.schedule) {
        h = Mix(h, c.logical_contribution_index);
        h = Mix(h, c.owner_index);
        h = Mix(h, c.ingress_channel);
        h = Mix(h, c.ingress_slot);
        h = Mix(h, c.payload_offset);
        h = Mix(h, c.buffer_id);
        h = Mix(h, c.element_bytes);
    }
    for (uint32_t v : plan.result_home_owner) h = Mix(h, v);
    for (uint32_t v : plan.owner_worklist_offsets) h = Mix(h, v);
    for (uint32_t v : plan.owner_worklist_entries) h = Mix(h, v);
    for (uint32_t v : plan.result_offsets) h = Mix(h, v);
    for (uint32_t v : plan.contribution_entry_indices) h = Mix(h, v);
    for (uint32_t v : plan.owner_result_offsets) h = Mix(h, v);
    for (uint32_t v : plan.owner_result_ids) h = Mix(h, v);
    return h == 0 ? 1 : h;
}

IncDcStatus CompileLogicalPlanToExecution(
    const IncDcCombineLogicalPlanV2 &logical,
    const IncDcTopologyDescriptor &topology, uint32_t hidden,
    uint32_t element_bytes, IncDcCompiledExecutionPlan *out,
    IncDcPlanCompileReport *report)
{
    IncDcPlanCompileReport local{};
    auto *rep = report ? report : &local;
    rep->ok = false;
    rep->first_error.clear();
    if (out == nullptr) {
        rep->first_error = "null_out";
        return IncDcStatus::INVALID_ARGUMENT;
    }
    IncDcLogicalPlanValidateReport lv{};
    if (ValidateLogicalPlanV2(logical, &lv) != IncDcStatus::OK) {
        rep->first_error = "logical_invalid:" + lv.first_error;
        return IncDcStatus::INVALID_ARGUMENT;
    }
    IncDcTopologyValidateReport tv{};
    if (ValidateTopologyDescriptor(topology, &tv) != IncDcStatus::OK) {
        rep->first_error = "topology_invalid:" + tv.first_error;
        return IncDcStatus::INVALID_ARGUMENT;
    }
    if (logical.worker_world_size != topology.worker_count) {
        rep->first_error = "worker_count_mismatch";
        return IncDcStatus::INVALID_ARGUMENT;
    }
    uint64_t row_bytes = 0;
    if (CheckedRowBytes(hidden, element_bytes, &row_bytes) != IncDcStatus::OK) {
        rep->first_error = "row_bytes_overflow";
        return IncDcStatus::CAPACITY_EXCEEDED;
    }

    *out = IncDcCompiledExecutionPlan{};
    out->topology = topology;
    out->semantic_digest = logical.semantic_digest;
    out->topology_digest = topology.topology_digest;
    out->hidden = hidden;
    out->element_bytes = element_bytes;
    out->row_bytes = row_bytes;
    out->schedule.resize(logical.contribution_count);
    out->result_offsets.assign(logical.result_count + 1u, 0);
    out->contribution_entry_indices.assign(logical.contribution_count, UINT32_MAX);
    out->result_home_owner.assign(logical.result_count, 0);

    const uint32_t owners_total = topology.owner_count;
    std::vector<std::vector<uint32_t>> per_owner(owners_total);
    std::vector<std::vector<uint32_t>> per_owner_results(owners_total);
    uint32_t next_owner = 0u;

    uint32_t sched_i = 0;
    for (uint32_t ri = 0; ri < logical.result_count; ++ri) {
        out->result_offsets[ri] = sched_i;
        const auto &r = logical.results[ri];
        const uint32_t home_owner = next_owner % topology.owner_count;
        ++next_owner;
        out->result_home_owner[ri] = home_owner;

        for (uint32_t j = 0; j < r.contribution_count; ++j) {
            const uint32_t li = r.contribution_begin + j;
            const auto &lc = logical.contributions[li];
            uint32_t channel = 0;
            if (!LookupIngressChannel(topology, lc.contributor_rank,
                                      &channel)) {
                rep->first_error = "missing_ingress_channel_edge";
                return IncDcStatus::INVALID_ARGUMENT;
            }
            uint32_t slot = 0;
            if (!CheckedMulU32(ri, logical.declared_max_topk, &slot)) {
                rep->first_error = "ingress_slot_overflow";
                return IncDcStatus::CAPACITY_EXCEEDED;
            }
            if (slot > UINT32_MAX - lc.ordinal) {
                rep->first_error = "ingress_slot_overflow";
                return IncDcStatus::CAPACITY_EXCEEDED;
            }
            slot += lc.ordinal;
            uint64_t payload_off = 0;
            if (!CheckedMulU64(slot, row_bytes, &payload_off)) {
                rep->first_error = "payload_offset_overflow";
                return IncDcStatus::CAPACITY_EXCEEDED;
            }

            IncDcCompiledContribution cc{};
            cc.logical_contribution_index = li;
            cc.owner_index = home_owner;
            cc.ingress_channel = channel;
            cc.ingress_slot = slot;
            cc.payload_offset = payload_off;
            cc.buffer_id = 0;
            cc.element_bytes = element_bytes;
            out->schedule[sched_i] = cc;
            out->contribution_entry_indices[li] = sched_i;

            per_owner[home_owner].push_back(sched_i);
            if (per_owner_results[home_owner].empty() ||
                per_owner_results[home_owner].back() != ri) {
                per_owner_results[home_owner].push_back(ri);
            }
            ++sched_i;
        }
    }
    out->result_offsets[logical.result_count] = sched_i;

    out->owner_worklist_offsets.assign(owners_total + 1u, 0);
    out->owner_worklist_entries.clear();
    out->owner_result_offsets.assign(owners_total + 1u, 0);
    out->owner_result_ids.clear();
    for (uint32_t o = 0; o < owners_total; ++o) {
        out->owner_worklist_offsets[o] =
            static_cast<uint32_t>(out->owner_worklist_entries.size());
        out->owner_worklist_entries.insert(out->owner_worklist_entries.end(),
                                           per_owner[o].begin(),
                                           per_owner[o].end());
        out->owner_result_offsets[o] =
            static_cast<uint32_t>(out->owner_result_ids.size());
        out->owner_result_ids.insert(out->owner_result_ids.end(),
                                     per_owner_results[o].begin(),
                                     per_owner_results[o].end());
    }
    out->owner_worklist_offsets[owners_total] =
        static_cast<uint32_t>(out->owner_worklist_entries.size());
    out->owner_result_offsets[owners_total] =
        static_cast<uint32_t>(out->owner_result_ids.size());

    out->execution_digest = ComputeExecutionDigest(*out);
    IncDcPlanCompileReport vr{};
    if (ValidateCompiledExecutionPlan(logical, *out, &vr) != IncDcStatus::OK) {
        rep->first_error = "post_compile_validate:" + vr.first_error;
        return IncDcStatus::INVALID_ARGUMENT;
    }
    rep->ok = true;
    return IncDcStatus::OK;
}

IncDcStatus ValidateCompiledExecutionPlan(
    const IncDcCombineLogicalPlanV2 &logical,
    const IncDcCompiledExecutionPlan &plan, IncDcPlanCompileReport *report)
{
    IncDcPlanCompileReport local{};
    auto *rep = report ? report : &local;
    rep->ok = false;
    rep->first_error.clear();
    auto fail = [&](const char *msg) -> IncDcStatus {
        rep->first_error = msg;
        return IncDcStatus::INVALID_ARGUMENT;
    };

    IncDcLogicalPlanValidateReport lv{};
    if (ValidateLogicalPlanV2(logical, &lv) != IncDcStatus::OK) {
        return fail("logical_invalid");
    }
    IncDcTopologyValidateReport tv{};
    if (ValidateTopologyDescriptor(plan.topology, &tv) != IncDcStatus::OK) {
        return fail("topology_invalid");
    }
    if (plan.topology_digest != plan.topology.topology_digest ||
        plan.topology_digest != ComputeTopologyDigest(plan.topology)) {
        return fail("topology_digest_mismatch");
    }
    if (plan.semantic_digest != logical.semantic_digest) {
        return fail("semantic_digest_mismatch");
    }
    if (plan.execution_digest != ComputeExecutionDigest(plan)) {
        return fail("execution_digest_mismatch");
    }
    if (plan.hidden == 0u || plan.element_bytes == 0u || plan.row_bytes == 0u) {
        return fail("layout_zero");
    }
    uint64_t expect_row = 0;
    if (CheckedRowBytes(plan.hidden, plan.element_bytes, &expect_row) !=
            IncDcStatus::OK ||
        expect_row != plan.row_bytes) {
        return fail("row_bytes_mismatch");
    }
    if (plan.schedule.size() != logical.contribution_count) {
        return fail("schedule_size_mismatch");
    }
    if (plan.result_home_owner.size() != logical.result_count) {
        return fail("result_home_size");
    }
    if (plan.result_offsets.size() != logical.result_count + 1u) {
        return fail("result_offsets_size");
    }
    if (plan.result_offsets[0] != 0u) return fail("result_offsets_not_zero");
    if (plan.result_offsets.back() != plan.schedule.size()) {
        return fail("result_offsets_end");
    }
    if (plan.contribution_entry_indices.size() != logical.contribution_count) {
        return fail("contrib_entry_size");
    }

    // logical_contribution_index must be a permutation of [0,C)
    std::vector<uint8_t> seen_li(logical.contribution_count, 0);
    for (size_t i = 0; i < plan.schedule.size(); ++i) {
        const auto &c = plan.schedule[i];
        if (c.logical_contribution_index >= logical.contribution_count) {
            return fail("logical_contribution_index_oob");
        }
        if (seen_li[c.logical_contribution_index]) {
            return fail("logical_contribution_index_dup");
        }
        seen_li[c.logical_contribution_index] = 1;
        if (plan.contribution_entry_indices[c.logical_contribution_index] !=
            static_cast<uint32_t>(i)) {
            return fail("contribution_entry_indices_mismatch");
        }
    }
    for (uint8_t s : seen_li) {
        if (!s) return fail("logical_contribution_index_missing");
    }

    for (uint32_t ri = 0; ri < logical.result_count; ++ri) {
        const uint32_t b = plan.result_offsets[ri];
        const uint32_t e = plan.result_offsets[ri + 1];
        if (e < b) return fail("result_offsets_non_monotonic");
        const auto &lr = logical.results[ri];
        if (e - b != lr.contribution_count) return fail("result_csr_count");
        const uint32_t home_owner = plan.result_home_owner[ri];
        if (home_owner >= plan.topology.owner_count) {
            return fail("home_owner_oob");
        }
        for (uint32_t i = b; i < e; ++i) {
            const auto &cc = plan.schedule[i];
            const auto &lc =
                logical.contributions[cc.logical_contribution_index];
            if (lc.result_id != ri) return fail("schedule_result_mismatch");
            if (cc.owner_index != home_owner) {
                return fail("contribution_not_on_unique_home");
            }
            uint32_t expect_ch = 0;
            if (!LookupIngressChannel(plan.topology, lc.contributor_rank,
                                      &expect_ch) ||
                expect_ch != cc.ingress_channel) {
                return fail("ingress_channel_not_from_edge_map");
            }
            if (cc.owner_index >= plan.topology.owner_count) {
                return fail("owner_oob");
            }
            uint64_t expect_off = 0;
            if (!CheckedMulU64(cc.ingress_slot, plan.row_bytes, &expect_off) ||
                expect_off != cc.payload_offset) {
                return fail("payload_offset_mismatch");
            }
            if (cc.element_bytes != plan.element_bytes) {
                return fail("element_bytes_mismatch");
            }
        }
    }

    const uint32_t owners_total = plan.topology.owner_count;
    if (plan.owner_worklist_offsets.size() != owners_total + 1u) {
        return fail("owner_worklist_offsets_size");
    }
    if (plan.owner_result_offsets.size() != owners_total + 1u) {
        return fail("owner_result_offsets_size");
    }
    if (plan.owner_worklist_offsets[0] != 0u ||
        plan.owner_result_offsets[0] != 0u) {
        return fail("owner_csr_not_zero");
    }
    if (plan.owner_worklist_offsets.back() !=
        plan.owner_worklist_entries.size()) {
        return fail("owner_worklist_end");
    }
    if (plan.owner_result_offsets.back() != plan.owner_result_ids.size()) {
        return fail("owner_result_end");
    }

    std::vector<uint8_t> sched_seen(plan.schedule.size(), 0);
    std::vector<uint8_t> result_owner_seen(logical.result_count, 0);
    for (uint32_t o = 0; o < owners_total; ++o) {
        const uint32_t wb = plan.owner_worklist_offsets[o];
        const uint32_t we = plan.owner_worklist_offsets[o + 1];
        if (we < wb) return fail("owner_worklist_non_monotonic");
        for (uint32_t i = wb; i < we; ++i) {
            const uint32_t si = plan.owner_worklist_entries[i];
            if (si >= plan.schedule.size()) return fail("owner_worklist_oob");
            if (sched_seen[si]) return fail("owner_worklist_dup");
            sched_seen[si] = 1;
            const auto &cc = plan.schedule[si];
            if (cc.owner_index != o) {
                return fail("owner_worklist_owner_mismatch");
            }
        }
        const uint32_t rb = plan.owner_result_offsets[o];
        const uint32_t re = plan.owner_result_offsets[o + 1];
        if (re < rb) return fail("owner_result_non_monotonic");
        std::unordered_set<uint32_t> local_results;
        for (uint32_t i = rb; i < re; ++i) {
            const uint32_t rid = plan.owner_result_ids[i];
            if (rid >= logical.result_count) return fail("owner_result_id_oob");
            if (!local_results.insert(rid).second) {
                return fail("owner_result_dup");
            }
            if (plan.result_home_owner[rid] != o) {
                return fail("owner_result_home_mismatch");
            }
            if (result_owner_seen[rid]) return fail("result_home_owner_dup");
            result_owner_seen[rid] = 1;
        }
    }
    for (uint8_t s : sched_seen) {
        if (!s) return fail("owner_worklist_missing");
    }
    for (uint32_t ri = 0; ri < logical.result_count; ++ri) {
        if (logical.results[ri].contribution_count == 0u) continue;
        if (!result_owner_seen[ri]) return fail("result_missing_from_owner_csr");
    }

    rep->ok = true;
    return IncDcStatus::OK;
}

} // namespace dc
} // namespace inc
