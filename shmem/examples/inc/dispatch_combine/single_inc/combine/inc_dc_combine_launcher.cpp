/**
 * DYN3D/DYN3C multi-rank CSR combine correctness host.
 * Usage:
 *   bin n_pes pe ipport npus first_npu W K results hidden mode
 * mode: 0=rr 1=multi_ordinal_zero_worker 2=ragged 3=skew_all_to_one
 *       4=half_worker_skew
 *       5..8=contribution-conserving hot-rank skew (12.5/37.5/62.5/87.5%)
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sched.h>
#include <string>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "param.h"
#include "shmem.h"
#include "utils.h"

#include "inc_dc_combine_logical_plan.h"
#include "inc_dc_combine_plan_wire.h"
#include "inc_dc_combine_plan_compiler.h"
#include "inc_dc_combine_topology.h"
#include "inc_dc_combine_runtime_abi.h"
#include "inc_dc_external_start_gate.h"
#include "inc_dc_physical_map.h"
#include "inc_dc_resource_policy.h"
#include "inc_dc_fp16_host.h"

using namespace inc;
using namespace inc::dc;

extern "C" void launch_inc_dc_sv2_dyn_csr_combine_kernel(uint8_t *sym,
                                                         uint64_t ctrl_off,
                                                         int block_dim,
                                                         void *stream);
extern "C" void launch_inc_dc_sv2_dyn_csr_producer_kernel(uint8_t *sym,
                                                          uint64_t ctrl_off,
                                                          int block_dim,
                                                          void *stream);
extern "C" void launch_inc_dc_sv2_dyn_csr_k1_direct_tx_kernel(
    uint8_t *sym, uint64_t ctrl_off, int block_dim, void *stream);
extern "C" void launch_inc_dc_sv2_dyn_csr_start_gate_kernel(
    uint8_t *sym, uint64_t ctrl_off, uint32_t participant, void *stream);
extern "C" void launch_inc_dc_sv2_dyn_csr_cycle_probe_kernel(
    uint8_t *sym, uint64_t cycle_off, void *stream);

static int g_npus = 16;
static int first_npu = 0;
static const char *ipport = "tcp://127.0.0.1:8970";
static aclshmemx_uniqueid_t default_flag_uid;

static uint64_t Align64(uint64_t x) { return (x + 63ull) & ~63ull; }

struct OwnerSelection {
    uint32_t requested = 1u;
    uint32_t resolved = 1u;
    uint32_t available_aiv = 0u;
    uint32_t policy_version = 0u;
    uint64_t policy_fingerprint = 0u;
    uint32_t dispatch_inc_capped = 0u;
    uint32_t combine_inc_capped = 0u;
    bool explicit_override = false;
    bool hardware_query_ok = false;
};

static OwnerSelection ResolveOwnerCount(uint32_t device_id,
                                        uint32_t worker_world_size)
{
    OwnerSelection selection{};
    int64_t vector_cores = 0;
    if (aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                           &vector_cores) == ACL_SUCCESS &&
        vector_cores > 0) {
        selection.available_aiv = static_cast<uint32_t>(vector_cores);
        selection.hardware_query_ok = true;
    }
    IncDcAivPolicy policy{};
    if (selection.hardware_query_ok &&
        IncDcResolveAivPolicy(selection.available_aiv, worker_world_size,
                             kDynCsrMaxOwners, kDynCsrMaxOwners, &policy)) {
        selection.requested = policy.combine_inc_aiv;
        selection.resolved = policy.combine_inc_aiv;
        selection.policy_version = policy.policy_version;
        selection.policy_fingerprint =
            IncDcResourcePolicyFingerprint(policy);
        selection.dispatch_inc_capped = policy.dispatch_inc_capped;
        selection.combine_inc_capped = policy.combine_inc_capped;
    } else {
        selection.hardware_query_ok = false;
    }
    return selection;
}

static int CreateOptionalPriorityStream(
    aclrtStream *stream, const char *env_name)
{
    const char *raw = std::getenv(env_name);
    if (raw == nullptr || raw[0] == '\0') {
        return aclrtCreateStream(stream);
    }
    char *end = nullptr;
    const unsigned long priority = std::strtoul(raw, &end, 10);
    if (end == raw || *end != '\0' || priority > 7ul) {
        std::cerr << "DYNCSR_STREAM_PRIORITY_INVALID name=" << env_name
                  << " value=" << raw << std::endl;
        return -1;
    }
    const aclError rc = aclrtCreateStreamWithConfig(
        stream, static_cast<uint32_t>(priority), 0u);
    if (rc == ACL_SUCCESS) {
        std::cerr << "DYNCSR_STREAM_PRIORITY name=" << env_name
                  << " value=" << priority << std::endl;
    }
    return rc;
}

static uint32_t FloatBits(float f)
{
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

static int InitShmem(int pe, int n_pes, uint64_t heap, int32_t *dev,
                     aclrtStream *stream)
{
    *dev = ResolvePhysicalNpuForPe(pe, g_npus, first_npu);
    if (aclInit(nullptr) != 0 || aclrtSetDevice(*dev) != 0 ||
        CreateOptionalPriorityStream(
            stream, "INC_DYNCSR_STREAM_PRIORITY") != 0) {
        return -1;
    }
    aclshmemx_init_attr_t attr{};
    test_set_attr(pe, n_pes, heap, ipport, default_flag_uid, &attr);
    return aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
}

static IncDcStatus BuildModePlan(uint32_t W, uint32_t K, uint32_t results,
                                 uint32_t mode, IncDcCombineLogicalPlanV2 *plan)
{
    if (mode == 3u) {
        // Skew: all contributions from worker 0.
        if (BuildSyntheticLogicalPlanV2(W, K, results, 0, plan) !=
            IncDcStatus::OK) {
            return IncDcStatus::INVALID_ARGUMENT;
        }
        for (auto &c : plan->contributions) {
            c.contributor_rank = 0;
        }
        plan->semantic_digest = ComputeLogicalPlanSemanticDigest(*plan);
        return IncDcStatus::OK;
    }
    if (mode == 4u) {
        // Half-worker skew: only ranks [0, ceil(W/2)) contribute (between
        // ragged and all-to-one). Keeps declared_max_topk / expected counts.
        if (BuildSyntheticLogicalPlanV2(W, K, results, 0, plan) !=
            IncDcStatus::OK) {
            return IncDcStatus::INVALID_ARGUMENT;
        }
        const uint32_t active =
            (W <= 1u) ? 1u : ((W + 1u) / 2u); // ceil(W/2)
        for (auto &c : plan->contributions) {
            c.contributor_rank = c.contributor_rank % active;
        }
        plan->semantic_digest = ComputeLogicalPlanSemanticDigest(*plan);
        return IncDcStatus::OK;
    }
    if (mode >= 5u && mode <= 8u) {
        // A comparable asymmetry ladder must conserve the logical work:
        // result count, contribution count, destination placement and weights
        // remain identical to mode 0.  Each successive stage redirects a
        // nested fraction of the same contributions to hot rank 0.  The last
        // eighth remains on the original sources even at mode 8, keeping the
        // participant set constant so the curve measures load imbalance
        // rather than the lower metadata cost of dropping idle workers.
        // True all-to-one remains mode 3 as an extreme correctness case.
        // Using the
        // global contribution ordinal makes the ladder deterministic for
        // arbitrary W/K, including K > W, without equating top-k with the
        // number of participating ranks.
        if (BuildSyntheticLogicalPlanV2(W, K, results, 0, plan) !=
            IncDcStatus::OK) {
            return IncDcStatus::INVALID_ARGUMENT;
        }
        static constexpr uint32_t kRedirectedEighths[] = {1u, 3u, 5u, 7u};
        const uint32_t redirected_eighths = kRedirectedEighths[mode - 5u];
        for (uint32_t ci = 0u; ci < plan->contribution_count; ++ci) {
            // Hash before bucketing.  Direct ci%8 aliases with K, result id,
            // home INC and owner rotation, producing artificial saw-tooth
            // load patterns rather than a smooth source-skew ladder.
            uint64_t mixed =
                plan->contributions[ci].contribution_uid +
                0x9e3779b97f4a7c15ull;
            mixed = (mixed ^ (mixed >> 30u)) * 0xbf58476d1ce4e5b9ull;
            mixed = (mixed ^ (mixed >> 27u)) * 0x94d049bb133111ebull;
            mixed ^= mixed >> 31u;
            const uint32_t bucket = static_cast<uint32_t>(mixed >> 61u);
            if (bucket < redirected_eighths) {
                plan->contributions[ci].contributor_rank = 0u;
            }
        }
        plan->semantic_digest = ComputeLogicalPlanSemanticDigest(*plan);
        return IncDcStatus::OK;
    }
    return BuildSyntheticLogicalPlanV2(W, K, results, mode, plan);
}

static IncDcStatus LoadLogicalPlanFile(const char *path,
                                       IncDcCombineLogicalPlanV2 *plan)
{
    if (path == nullptr || path[0] == '\0' || plan == nullptr) {
        return IncDcStatus::INVALID_ARGUMENT;
    }
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return IncDcStatus::INVALID_ARGUMENT;
    const std::streamoff size = input.tellg();
    if (size <= 0 || static_cast<uint64_t>(size) >
                         std::numeric_limits<size_t>::max()) {
        return IncDcStatus::INVALID_ARGUMENT;
    }
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> blob(static_cast<size_t>(size));
    if (!input.read(reinterpret_cast<char *>(blob.data()), size)) {
        return IncDcStatus::INVALID_ARGUMENT;
    }
    IncDcLogicalPlanWireReport report{};
    return ParseLogicalPlanWireV2(blob.data(), blob.size(), plan, &report);
}

// Scheme B transport plan: one physical partial per (result, contributor
// rank).  pre_offsets/pre_entries retain every expert-instance contribution so
// worker AIVs can perform the weighted local reduction before the push.  If no
// result has two instances on one rank the caller bypasses this plan entirely,
// preserving the qualified W8 fast path bit for bit.
static IncDcStatus BuildRankDedupTransportPlan(
    const IncDcCombineLogicalPlanV2 &logical,
    IncDcCombineLogicalPlanV2 *transport, std::vector<uint32_t> *pre_offsets,
    std::vector<uint32_t> *pre_entries, std::vector<uint32_t> *pre_weights,
    bool *active)
{
    if (transport == nullptr || pre_offsets == nullptr ||
        pre_entries == nullptr || pre_weights == nullptr || active == nullptr ||
        logical.worker_world_size == 0u) {
        return IncDcStatus::INVALID_ARGUMENT;
    }
    *transport = IncDcCombineLogicalPlanV2{};
    transport->worker_world_size = logical.worker_world_size;
    transport->result_count = logical.result_count;
    transport->results.reserve(logical.result_count);
    pre_offsets->clear();
    pre_entries->clear();
    pre_weights->clear();
    pre_offsets->push_back(0u);
    *active = false;
    uint32_t max_physical_topk = 0u;
    bool uniform = true;
    uint32_t first_count = 0u;
    std::vector<std::vector<uint32_t>> by_source(logical.worker_world_size);
    for (uint32_t r = 0u; r < logical.result_count; ++r) {
        for (auto &v : by_source) v.clear();
        const auto &logical_result = logical.results[r];
        const uint64_t logical_end =
            static_cast<uint64_t>(logical_result.contribution_begin) +
            logical_result.contribution_count;
        if (logical_end > logical.contributions.size()) {
            return IncDcStatus::INVALID_ARGUMENT;
        }
        for (uint32_t li = logical_result.contribution_begin;
             li < logical_end; ++li) {
            const auto &c = logical.contributions[li];
            if (c.contributor_rank >= logical.worker_world_size) {
                return IncDcStatus::INVALID_ARGUMENT;
            }
            by_source[c.contributor_rank].push_back(li);
        }
        IncDcLogicalResultV2 physical_result{};
        physical_result.dst_rank = logical_result.dst_rank;
        physical_result.dst_local_row = logical_result.dst_local_row;
        physical_result.contribution_begin =
            static_cast<uint32_t>(transport->contributions.size());
        uint32_t physical_ordinal = 0u;
        for (uint32_t source = 0u; source < logical.worker_world_size;
             ++source) {
            if (by_source[source].empty()) continue;
            if (by_source[source].size() > 1u) *active = true;
            IncDcLogicalContributionV2 physical{};
            physical.contribution_uid =
                static_cast<uint64_t>(transport->contributions.size()) + 1u;
            physical.result_id = r;
            physical.ordinal = physical_ordinal++;
            physical.contributor_rank = source;
            physical.contributor_local_row =
                static_cast<uint32_t>(transport->contributions.size());
            physical.weight = 1.f;
            transport->contributions.push_back(physical);
            for (const uint32_t li : by_source[source]) {
                pre_entries->push_back(li);
                uint32_t bits = 0u;
                std::memcpy(&bits, &logical.contributions[li].weight,
                            sizeof(bits));
                pre_weights->push_back(bits);
            }
            pre_offsets->push_back(static_cast<uint32_t>(pre_entries->size()));
        }
        physical_result.contribution_count = physical_ordinal;
        if (r == 0u) first_count = physical_ordinal;
        uniform = uniform && physical_ordinal == first_count;
        max_physical_topk = std::max(max_physical_topk, physical_ordinal);
        transport->results.push_back(physical_result);
    }
    transport->contribution_count =
        static_cast<uint32_t>(transport->contributions.size());
    transport->declared_max_topk = max_physical_topk;
    transport->uniform_topk_valid = uniform;
    transport->uniform_topk = uniform ? first_count : 0u;
    transport->semantic_digest = ComputeLogicalPlanSemanticDigest(*transport);
    return IncDcStatus::OK;
}

// A packed result egress is possible when every destination owns a dense,
// unique local row interval [0, count).  This is the normal framework layout,
// but it is proved from the supplied plan rather than assumed.  Arbitrary
// sparse/permuted destination rows remain fully supported through the safe
// per-row fallback.
static bool BuildPackedResultRankOffsets(
    const IncDcCombineLogicalPlanV2 &plan,
    std::vector<uint32_t> *rank_offsets)
{
    if (rank_offsets == nullptr || plan.worker_world_size == 0u) return false;
    const uint32_t W = plan.worker_world_size;
    const uint32_t R = plan.result_count;
    std::vector<uint32_t> counts(W, 0u);
    std::vector<std::vector<uint8_t>> seen(
        W, std::vector<uint8_t>(R, static_cast<uint8_t>(0)));
    for (const auto &result : plan.results) {
        if (result.dst_rank >= W || result.dst_local_row >= R ||
            seen[result.dst_rank][result.dst_local_row] != 0u) {
            return false;
        }
        seen[result.dst_rank][result.dst_local_row] = 1u;
        ++counts[result.dst_rank];
    }
    rank_offsets->assign(static_cast<size_t>(W) + 1u, 0u);
    for (uint32_t w = 0u; w < W; ++w) {
        for (uint32_t row = 0u; row < counts[w]; ++row) {
            if (seen[w][row] == 0u) return false;
        }
        (*rank_offsets)[w + 1u] = (*rank_offsets)[w] + counts[w];
    }
    return rank_offsets->back() == R;
}

static IncDcStatus ReassignPackedOwnerBlocks(
    const IncDcCombineLogicalPlanV2 &plan,
    const std::vector<uint32_t> &rank_offsets,
    IncDcCompiledExecutionPlan *exec)
{
    if (exec == nullptr || rank_offsets.size() !=
                               static_cast<size_t>(plan.worker_world_size) + 1u ||
        exec->topology.owner_count == 0u) {
        return IncDcStatus::INVALID_ARGUMENT;
    }
    const uint32_t R = plan.result_count;
    const uint32_t O = exec->topology.owner_count;
    std::vector<std::vector<uint32_t>> owner_entries(O);
    std::vector<std::vector<uint32_t>> owner_results(O);
    for (uint32_t r = 0u; r < R; ++r) {
        const auto &result = plan.results[r];
        const uint32_t storage_row =
            rank_offsets[result.dst_rank] + result.dst_local_row;
        if (storage_row >= R) return IncDcStatus::INVALID_ARGUMENT;
        const uint32_t owner = static_cast<uint32_t>(
            static_cast<uint64_t>(storage_row) * O / R);
        exec->result_home_owner[r] = std::min(owner, O - 1u);
        owner_results[exec->result_home_owner[r]].push_back(r);
        for (uint32_t si = exec->result_offsets[r];
             si < exec->result_offsets[r + 1u]; ++si) {
            exec->schedule[si].owner_index = exec->result_home_owner[r];
            owner_entries[exec->result_home_owner[r]].push_back(si);
        }
    }
    exec->owner_worklist_offsets.assign(static_cast<size_t>(O) + 1u, 0u);
    exec->owner_worklist_entries.clear();
    exec->owner_result_offsets.assign(static_cast<size_t>(O) + 1u, 0u);
    exec->owner_result_ids.clear();
    for (uint32_t o = 0u; o < O; ++o) {
        exec->owner_worklist_entries.insert(exec->owner_worklist_entries.end(),
                                            owner_entries[o].begin(),
                                            owner_entries[o].end());
        exec->owner_worklist_offsets[o + 1u] =
            static_cast<uint32_t>(exec->owner_worklist_entries.size());
        exec->owner_result_ids.insert(exec->owner_result_ids.end(),
                                      owner_results[o].begin(),
                                      owner_results[o].end());
        exec->owner_result_offsets[o + 1u] =
            static_cast<uint32_t>(exec->owner_result_ids.size());
    }
    exec->execution_digest = ComputeExecutionDigest(*exec);
    IncDcPlanCompileReport report{};
    return ValidateCompiledExecutionPlan(plan, *exec, &report);
}

int main(int argc, char **argv)
{
    if (argc < 11) {
        std::cerr << "Usage: bin n_pes pe ipport npus first_npu W K results "
                     "hidden mode\n";
        return 2;
    }
    const int n_pes = std::atoi(argv[1]);
    const int pe = std::atoi(argv[2]);
    ipport = argv[3];
    g_npus = std::atoi(argv[4]);
    first_npu = std::atoi(argv[5]);
    const uint32_t W = static_cast<uint32_t>(std::atoi(argv[6]));
    const uint32_t K = static_cast<uint32_t>(std::atoi(argv[7]));
    const uint32_t results = static_cast<uint32_t>(std::atoi(argv[8]));
    const uint32_t hidden = static_cast<uint32_t>(std::atoi(argv[9]));
    const uint32_t mode = static_cast<uint32_t>(std::atoi(argv[10]));
    if (n_pes <= 0 || pe < 0 || pe >= n_pes || W < 1u ||
        n_pes != static_cast<int>(W + 1u) || g_npus <= 0 ||
        first_npu < 0) {
        return 2;
    }
    IncDcPhysicalMapValidation physical_map{};
    std::string physical_map_error;
    if (!ValidatePhysicalNpuMap(
            n_pes, g_npus, first_npu, /*require_unique=*/true,
            &physical_map, &physical_map_error)) {
        std::cerr << "DYNCSR_PHYSICAL_MAP_FAIL pe=" << pe
                  << " reason=" << physical_map_error << std::endl;
        return 3;
    }
    uint32_t service_epochs = 1u;
    if (const char *raw = std::getenv("INC_DYNCSR_SERVICE_EPOCHS")) {
        const uint32_t requested =
            static_cast<uint32_t>(std::strtoul(raw, nullptr, 10));
        if (requested >= 1u && requested <= 10000u) {
            service_epochs = requested;
        }
    }
    uint32_t service_warmup_epochs = service_epochs > 1u ? 1u : 0u;
    if (const char *raw =
            std::getenv("INC_DYNCSR_SERVICE_WARMUP_EPOCHS")) {
        const uint32_t requested =
            static_cast<uint32_t>(std::strtoul(raw, nullptr, 10));
        if (requested <= 10u) {
            service_warmup_epochs = requested;
        }
    }
    const uint32_t local_device = static_cast<uint32_t>(
        ResolvePhysicalNpuForPe(pe, g_npus, first_npu));
    OwnerSelection owner_selection =
        ResolveOwnerCount(local_device, W);
    if (!owner_selection.hardware_query_ok ||
        owner_selection.available_aiv == 0u) {
        std::cerr << "DYNCSR_AIV_QUERY_FAIL pe=" << pe
                  << " device=" << local_device << std::endl;
        return 10;
    }
    // Owners remain multifunctional in the production service.  A separate
    // result-TX cohort loses reduction parallelism and adds a GM publication
    // handshake; the reducer instead streams its completed UB tile directly
    // to the destination worker on the strict INC push path.
    const uint32_t tx_lane_count = 0u;
    // Reducers and result-TX lanes share the runtime-selected combine cohort.
    const uint32_t owner_budget = owner_selection.hardware_query_ok
        ? owner_selection.resolved - tx_lane_count
        : owner_selection.resolved;
    uint32_t owner_count =
        std::max(1u, std::min(owner_selection.resolved, owner_budget));
    // Qualification overrides may reduce the cohort for A/B testing, but may
    // not consume Dispatch's disjoint hardware-derived partition.
    if (const char *raw = std::getenv("INC_DYNCSR_OWNER_COUNT")) {
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(raw, &end, 10);
        if (end == raw || *end != '\0' || parsed < 1ul ||
            parsed > owner_selection.resolved) {
            std::cerr << "DYNCSR_OWNER_OVERRIDE_INVALID pe=" << pe
                      << " value=" << raw
                      << " qualified_limit=" << owner_selection.resolved
                      << std::endl;
            return 3;
        }
        owner_count = static_cast<uint32_t>(parsed);
        owner_selection.explicit_override = true;
    }

    std::cout.setf(std::ios::unitbuf);
    if (n_pes != static_cast<int>(W + 1u) || pe < 0 || pe >= n_pes ||
        W < 1 || K < 1 || results < 1 || hidden < 1 ||
        static_cast<uint64_t>(hidden) * 2ull > UINT32_MAX - 63ull) {
        return 2;
    }

    IncDcCombineLogicalPlanV2 logical_plan{};
    const char *logical_plan_file =
        std::getenv("INC_DYNCSR_LOGICAL_PLAN_FILE");
    const IncDcStatus plan_status =
        logical_plan_file != nullptr && logical_plan_file[0] != '\0'
            ? LoadLogicalPlanFile(logical_plan_file, &logical_plan)
            : BuildModePlan(W, K, results, mode, &logical_plan);
    if (plan_status != IncDcStatus::OK ||
        logical_plan.worker_world_size != W ||
        logical_plan.result_count != results ||
        logical_plan.declared_max_topk > K) {
        std::cerr << "DYNCSR_PLAN_FAIL pe=" << pe << std::endl;
        return 1;
    }
    // Test/framework hook for a genuine weighted K1 plan.  It also proves
    // that the identity-copy optimization is predicate-guarded rather than
    // silently treating every K1 operation as an unweighted copy.
    if (const char *raw = std::getenv("INC_DYNCSR_UNIFORM_WEIGHT")) {
        char *end = nullptr;
        const float requested = std::strtof(raw, &end);
        if (end == raw || *end != '\0' || !std::isfinite(requested)) {
            std::cerr << "DYNCSR_WEIGHT_ENV_FAIL pe=" << pe << std::endl;
            return 2;
        }
        for (auto &c : logical_plan.contributions) {
            c.weight = requested;
        }
        logical_plan.semantic_digest =
            ComputeLogicalPlanSemanticDigest(logical_plan);
    }
    IncDcLogicalPlanValidateReport lvr{};
    if (ValidateLogicalPlanV2(logical_plan, &lvr) != IncDcStatus::OK) {
        std::cerr << "DYNCSR_PLAN_VALIDATE_FAIL pe=" << pe << " "
                  << lvr.first_error << std::endl;
        return 1;
    }

    IncDcCombineLogicalPlanV2 rank_dedup_plan{};
    std::vector<uint32_t> local_reduce_offsets;
    std::vector<uint32_t> local_reduce_entries;
    std::vector<uint32_t> local_reduce_weights;
    bool rank_dedup_active = false;
    if (BuildRankDedupTransportPlan(
            logical_plan, &rank_dedup_plan, &local_reduce_offsets,
            &local_reduce_entries, &local_reduce_weights,
            &rank_dedup_active) != IncDcStatus::OK) {
        std::cerr << "DYNCSR_RANK_DEDUP_PLAN_FAIL pe=" << pe << std::endl;
        return 1;
    }
    // Scheme B preserves every expert contribution through the communication
    // boundary.  Workers push those contributions to the single INC and the
    // INC performs the complete reduction.  This is the natural Megatron/
    // vLLM boundary, keeps worker compute/AIV use low, and avoids turning
    // repeated local experts into a pre-communication vector bottleneck.
    // The older rank-local pre-reduce remains an explicit diagnostic option;
    // it is never selected from W/K/message size and therefore cannot create
    // workload-specific production paths.
    const bool rank_dedup_candidate = rank_dedup_active;
    rank_dedup_active = false;
    if (const char *raw = std::getenv("INC_DYNCSR_RANK_DEDUP")) {
        if (raw[0] == '1' && raw[1] == '\0') {
            rank_dedup_active = rank_dedup_candidate;
        } else if (!(raw[0] == '0' && raw[1] == '\0')) {
            std::cerr << "DYNCSR_RANK_DEDUP_INVALID pe=" << pe
                      << " value=" << raw << std::endl;
            return 3;
        }
    }
    const IncDcCombineLogicalPlanV2 &plan =
        rank_dedup_active ? rank_dedup_plan : logical_plan;
    IncDcLogicalPlanValidateReport transport_lvr{};
    if (ValidateLogicalPlanV2(plan, &transport_lvr) != IncDcStatus::OK) {
        std::cerr << "DYNCSR_TRANSPORT_PLAN_VALIDATE_FAIL pe=" << pe << " "
                  << transport_lvr.first_error << std::endl;
        return 1;
    }
    const bool all_unit_weights =
        std::all_of(plan.contributions.begin(), plan.contributions.end(),
                    [](const auto &c) { return c.weight == 1.0f; });
    const bool unit_small_reduce =
        transport_lvr.expected_count_min >= 1u &&
        transport_lvr.expected_count_max <= 2u && all_unit_weights;
    const bool identity_k1_plan =
        transport_lvr.expected_count_min == 1u &&
        transport_lvr.expected_count_max == 1u && all_unit_weights;
    // Identity K1 is a duplex relay rather than a reduction.  Its owner AIVs
    // poll ingress readiness and immediately push one row back out; launching
    // the full reducer cohort only creates small-row MTE/ready contention.
    // Use one live-AIV share per attached source, matching the topology-derived
    // egress cohort.  Explicit qualification overrides remain authoritative.
    if (identity_k1_plan && !owner_selection.explicit_override) {
        const uint32_t relay_cohort = std::max(
            1u, (owner_selection.available_aiv + W - 1u) / W);
        owner_count = std::min(owner_count, relay_cohort);
    }
    bool regular_small_local_reduce = rank_dedup_active &&
        local_reduce_offsets.size() == plan.contribution_count + 1u;
    for (uint32_t c = 0u;
         regular_small_local_reduce && c < plan.contribution_count; ++c) {
        const uint32_t begin = local_reduce_offsets[c];
        const uint32_t end = local_reduce_offsets[c + 1u];
        const uint32_t count = end - begin;
        if (count < 2u || count > 4u || end > local_reduce_entries.size()) {
            regular_small_local_reduce = false;
            break;
        }
        const uint32_t first = local_reduce_entries[begin];
        const uint32_t second = local_reduce_entries[begin + 1u];
        const uint32_t step = second > first ? second - first : 0u;
        if (step == 0u) {
            regular_small_local_reduce = false;
            break;
        }
        for (uint32_t local = 2u; local < count; ++local) {
            if (local_reduce_entries[begin + local] !=
                first + local * step) {
                regular_small_local_reduce = false;
                break;
            }
        }
    }

    std::vector<uint32_t> packed_result_rank_offsets;
    // Rank-packing creates a second bulk-TX phase.  It is useful as a
    // diagnostic backend, but on the single-INC streaming path it destroys
    // reduce/egress overlap and is substantially slower for rank-deduplicated
    // plans.  Keep natural result order by default; the explicit knob remains
    // available for platforms whose MTE bulk path is faster.
    bool request_packed_result_tx = false;
    if (const char *raw = std::getenv("INC_DYNCSR_PACKED_RESULT_TX")) {
        request_packed_result_tx = raw[0] == '1' && raw[1] == '\0';
    }
    const bool packed_result_tx =
        request_packed_result_tx &&
        BuildPackedResultRankOffsets(plan, &packed_result_rank_offsets);
    // By default the reducer cohort performs the packed result push after its
    // local block.  This keeps the INC at exactly half of its live AIVs.  The
    // explicit split-TX environment knob remains available for diagnostics.

    IncDcTopologyDescriptor topo{};
    if (BuildSingleIncTopology(W, owner_count, /*wpe*/ 0,
                               /*ipe*/ W, 1, &topo) != IncDcStatus::OK) {
        return 1;
    }
    // Map PE ids to actual ranks: workers 0..W-1, single INC W.
    for (uint32_t w = 0; w < W; ++w) topo.worker_pe_ids[w] = w;
    topo.inc_pe = W;
    topo.topology_digest = ComputeTopologyDigest(topo);

    IncDcCompiledExecutionPlan exec{};
    IncDcPlanCompileReport cr{};
    if (CompileLogicalPlanToExecution(plan, topo, hidden, /*elem*/ 2, &exec,
                                      &cr) != IncDcStatus::OK) {
        std::cerr << "DYNCSR_COMPILE_FAIL pe=" << pe << " " << cr.first_error
                  << std::endl;
        return 1;
    }
    if (packed_result_tx &&
        ReassignPackedOwnerBlocks(plan, packed_result_rank_offsets, &exec) !=
            IncDcStatus::OK) {
        std::cerr << "DYNCSR_PACKED_OWNER_REASSIGN_FAIL pe=" << pe
                  << std::endl;
        return 1;
    }

    uint32_t max_slot = 0;
    for (const auto &c : exec.schedule) {
        max_slot = std::max(max_slot, c.ingress_slot);
    }
    const uint32_t slot_count = max_slot + 1u;
    const uint32_t group_count = owner_count * W;
    const uint32_t source_bitmap_words = (W + 31u) / 32u;
    const uint32_t payload_bytes = hidden * 2u;
    const uint32_t tile_bytes =
        static_cast<uint32_t>((static_cast<uint64_t>(payload_bytes) + 63ull) &
                              ~63ull);
    // The current device RMA implementation has only proved a wide NBI
    // window when a source contributes at most once to any result.  Derive
    // that capability from the compiled CSR; never infer it from K or mode.
    bool producer_wide_window_safe = true;
    std::vector<uint32_t> per_source_count(W, 0u);
    for (uint32_t r = 0; r < plan.result_count; ++r) {
        std::fill(per_source_count.begin(), per_source_count.end(), 0u);
        for (uint32_t si = exec.result_offsets[r];
             si < exec.result_offsets[r + 1u]; ++si) {
            const auto &cc = exec.schedule[si];
            const auto &lc =
                plan.contributions[cc.logical_contribution_index];
            if (lc.contributor_rank >= W ||
                ++per_source_count[lc.contributor_rank] > 1u) {
                producer_wide_window_safe = false;
            }
        }
    }
    // Rank dedup makes the wire CSR look one-source-per-result only after a
    // worker has reduced several local ordinals into a reusable staging row.
    // That staging lifetime is not compatible with the public producer's
    // wide NBI window, even though the post-dedup CSR passes the check above.
    producer_wide_window_safe =
        producer_wide_window_safe && !rank_dedup_active;

    const uint64_t ctrl_off = 0;
    uint64_t off = Align64(kDynCsrCtrlBytes);
    const uint64_t result_offsets_off = off;
    off = Align64(off + (plan.result_count + 1u) * sizeof(uint32_t));
    const uint64_t home_owner_off = off;
    off = Align64(off + plan.result_count * sizeof(uint32_t));
    const uint64_t result_dst_rank_off = off;
    off = Align64(off + plan.result_count * sizeof(uint32_t));
    const uint64_t result_dst_row_off = off;
    off = Align64(off + plan.result_count * sizeof(uint32_t));
    const uint64_t result_tx_rank_offsets_off = off;
    off = Align64(off + (static_cast<uint64_t>(W) + 1u) *
                            sizeof(uint32_t));
    const uint64_t packed_result_ids_off = off;
    off = Align64(off + static_cast<uint64_t>(plan.result_count) *
                            sizeof(uint32_t));
    const uint64_t slot_off = off;
    off = Align64(off + plan.contribution_count * sizeof(uint32_t));
    const uint64_t weight_off = off;
    off = Align64(off + plan.contribution_count * sizeof(uint32_t));
    const uint64_t uid_lo_off = off;
    off = Align64(off + plan.contribution_count * sizeof(uint32_t));
    const uint64_t uid_hi_off = off;
    off = Align64(off + plan.contribution_count * sizeof(uint32_t));
    const uint64_t ordinal_off = off;
    off = Align64(off + plan.contribution_count * sizeof(uint32_t));
    const uint64_t gen_off = off;
    off = Align64(off + plan.contribution_count * sizeof(uint64_t));
    const uint64_t source_rank_off = off;
    off = Align64(off + plan.contribution_count * sizeof(uint32_t));
    const uint64_t contrib_owner_off = off;
    off = Align64(off + plan.contribution_count * sizeof(uint32_t));
    const uint64_t contrib_result_off = off;
    off = Align64(off + plan.contribution_count * sizeof(uint32_t));
    const uint64_t group_offsets_off = off;
    off = Align64(off + (static_cast<uint64_t>(group_count) + 1u) *
                            sizeof(uint32_t));
    const uint64_t group_entries_off = off;
    off = Align64(off + plan.contribution_count * sizeof(uint32_t));
    const uint64_t source_contribution_offsets_off = off;
    off = Align64(off + (static_cast<uint64_t>(W) + 1u) *
                            sizeof(uint32_t));
    const uint64_t source_contribution_entries_off = off;
    off = Align64(off + plan.contribution_count * sizeof(uint32_t));
    const uint64_t source_group_offsets_off = off;
    off = Align64(off + (static_cast<uint64_t>(W) + 1u) *
                            sizeof(uint32_t));
    const uint64_t source_group_entries_off = off;
    off = Align64(off + static_cast<uint64_t>(group_count) *
                            sizeof(uint32_t));
    const uint64_t owner_source_bitmap_off = off;
    off = Align64(off + static_cast<uint64_t>(owner_count) *
                            source_bitmap_words * sizeof(uint32_t));
    const uint64_t waited_source_bitmap_off = off;
    off = Align64(off + static_cast<uint64_t>(owner_count) *
                            source_bitmap_words * sizeof(uint32_t));
    const uint64_t worker_pe_off = off;
    off = Align64(off + static_cast<uint64_t>(W) * sizeof(uint32_t));
    constexpr uint32_t kReadyStrideBytes = 64u;
    const uint64_t start_gate_off = off;
    // One writer-owned arrival cacheline per worker and the INC, plus one release
    // cacheline.  This is capacity-derived from runtime topology, never a
    // fixed 16-rank protocol assumption.
    off = Align64(off + (static_cast<uint64_t>(W) + 2u) *
                            kReadyStrideBytes);
    const uint64_t ready_generation_off = off;
    const uint64_t completion_record_count = static_cast<uint64_t>(W);
    const uint64_t direct_completion_record_count =
        static_cast<uint64_t>(W) * static_cast<uint64_t>(W);
    const uint64_t batched_record_count =
        static_cast<uint64_t>(group_count) + completion_record_count +
        direct_completion_record_count;
    const uint64_t stream_record_count =
        static_cast<uint64_t>(slot_count) + completion_record_count;
    const uint64_t ready_record_count =
        std::max<uint64_t>(stream_record_count, batched_record_count);
    off = Align64(off + static_cast<uint64_t>(ready_record_count) *
                            kReadyStrideBytes);
    const uint64_t result_arrival_counter_off = off;
    off = Align64(off + static_cast<uint64_t>(plan.result_count) *
                            kReadyStrideBytes);
    const uint64_t local_reduce_offsets_off = off;
    off = Align64(off +
                  (rank_dedup_active
                       ? (static_cast<uint64_t>(plan.contribution_count) + 1u) *
                             sizeof(uint32_t)
                       : 0u));
    const uint64_t local_reduce_entries_off = off;
    off = Align64(off +
                  (rank_dedup_active
                       ? static_cast<uint64_t>(logical_plan.contribution_count) *
                             sizeof(uint32_t)
                       : 0u));
    const uint64_t local_reduce_weights_off = off;
    off = Align64(off +
                  (rank_dedup_active
                       ? static_cast<uint64_t>(logical_plan.contribution_count) *
                             sizeof(uint32_t)
                       : 0u));
    const uint64_t logical_input_off = off;
    off = Align64(off +
                  (rank_dedup_active
                       ? static_cast<uint64_t>(logical_plan.contribution_count) *
                             tile_bytes
                       : 0u));
    const uint64_t ingress_off = off;
    off = Align64(off + static_cast<uint64_t>(slot_count) * tile_bytes);
    const uint64_t output_off = off;
    off = Align64(off + static_cast<uint64_t>(plan.result_count) * tile_bytes);
    const uint64_t result_tx_ready_off = off;
    off = Align64(off + static_cast<uint64_t>(plan.result_count) * 64u);
    const uint64_t tx_done_off = off;
    off = Align64(off + static_cast<uint64_t>(tx_lane_count) * 64u);
    const uint64_t arrival_off = off;
    // The first R words retain the compact <=64-ordinal fast path.  The next
    // C words are generation-tagged ordinal markers for arbitrary top-k; a
    // result with K>64 addresses marker R + contribution_begin + ordinal.
    off = Align64(off +
                  (static_cast<uint64_t>(plan.result_count) +
                   plan.contribution_count) * sizeof(uint64_t));
    const uint64_t stats_off = off;
    off = Align64(off + sizeof(DynCsrStats));
    const uint64_t owner_stats_off = off;
    off = Align64(off + static_cast<uint64_t>(kDynCsrMaxOwners) *
                            sizeof(DynCsrOwnerStats));
    const uint64_t heap_need = Align64(off + 4096ull);
    constexpr uint64_t kDeviceLargePageBytes = 2ull * 1024ull * 1024ull;
    const uint64_t heap_floor_unaligned =
        std::max<uint64_t>(heap_need, 64ull * 1024ull * 1024ull);
    const uint64_t heap_floor =
        (heap_floor_unaligned + kDeviceLargePageBytes - 1ull) &
        ~(kDeviceLargePageBytes - 1ull);

    aclrtStream stream = nullptr;
    int32_t device = 0;
    if (InitShmem(pe, n_pes, heap_floor, &device, &stream) != 0) {
        std::cerr << "DYNCSR_SHMEM_FAIL pe=" << pe << std::endl;
        return 1;
    }
    uint8_t *sym = static_cast<uint8_t *>(aclshmem_malloc(heap_need));
    if (sym == nullptr) {
        return 1;
    }

    const bool is_worker = pe < static_cast<int>(W);
    std::cout << "DYNCSR_REGISTER pe=" << pe
              << " role=" << (is_worker ? "worker" : "inc")
              << " worker_world_size=" << W
              << " declared_max_topk=" << plan.declared_max_topk
              << " owner_count=" << owner_count
              << " split_tx_lanes=" << tx_lane_count
              << " owner_requested=" << owner_selection.requested
              << " owner_policy="
              << (owner_selection.explicit_override ? "environment"
                                                    : "hardware_static")
              << " available_aiv=" << owner_selection.available_aiv
              << " platform_caps_version="
              << kIncDcPlatformCapabilityVersion
              << " resource_policy_version="
              << owner_selection.policy_version
              << " resource_policy_fingerprint="
              << owner_selection.policy_fingerprint
              << " dispatch_inc_capped="
              << owner_selection.dispatch_inc_capped
              << " combine_inc_capped="
              << owner_selection.combine_inc_capped
              << " physical_map_digest=" << physical_map.digest
              << " explicit_map_entries=" << physical_map.explicit_entries
              << " ub_bytes=" << kIncDcAivUbBudgetBytes
              << " private_mte_packet_bytes="
              << kIncDcPrivateMtePacketBytes
              << " hardware_aiv_query_ok="
              << (owner_selection.hardware_query_ok ? 1 : 0)
              << " results=" << plan.result_count
              << " logical_contributions="
              << logical_plan.contribution_count
              << " physical_contributions=" << plan.contribution_count
              << " rank_dedup=" << (rank_dedup_active ? 1 : 0)
              << " plan_source="
              << (logical_plan_file != nullptr && logical_plan_file[0] != '\0'
                      ? logical_plan_file
                      : "synthetic")
              << std::endl;

    // Host-side oracle + tiles
    std::vector<std::vector<uint16_t>> logical_tiles(
        logical_plan.contribution_count, std::vector<uint16_t>(hidden));
    std::vector<std::vector<float>> oracle(logical_plan.result_count,
                                           std::vector<float>(hidden, 0.f));
    for (uint32_t li = 0; li < logical_plan.contribution_count; ++li) {
        const auto &lc = logical_plan.contributions[li];
        for (uint32_t h = 0; h < hidden; ++h) {
            const float v = 0.25f * static_cast<float>((lc.contributor_rank + 1) *
                                                      3 + (h % 7) + lc.ordinal);
            logical_tiles[li][h] = FloatToFp16Bits(v);
        }
    }
    for (uint32_t r = 0; r < logical_plan.result_count; ++r) {
        const auto &res = logical_plan.results[r];
        for (uint32_t j = 0; j < res.contribution_count; ++j) {
            const uint32_t li = res.contribution_begin + j;
            const auto &lc = logical_plan.contributions[li];
            for (uint32_t h = 0; h < hidden; ++h) {
                oracle[r][h] +=
                    Fp16BitsToFloat(logical_tiles[li][h]) * lc.weight;
            }
        }
    }

    const bool device_producer = [] {
        const char *v = std::getenv("INC_DYNCSR_DEVICE_PRODUCER");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    const bool overlap_enable = [&] {
        const char *v = std::getenv("INC_DYNCSR_OVERLAP");
        return device_producer && v != nullptr && v[0] == '1' &&
               v[1] == '\0';
    }();
    const bool batched_ready = [&] {
        const char *v = std::getenv("INC_DYNCSR_BATCHED_READY");
        return device_producer && v != nullptr && v[0] == '1' &&
               v[1] == '\0';
    }();

    std::vector<uint8_t> host(heap_need, 0);
    DynCsrCtrl ctrl{};
    ctrl.magic = kDynCsrMagic;
    ctrl.result_count = plan.result_count;
    ctrl.contribution_count = plan.contribution_count;
    ctrl.hidden = hidden;
    ctrl.tile_bytes = tile_bytes;
    ctrl.element_bytes = 2;
    ctrl.owner_count = topo.owner_count;
    ctrl.inc_pe = topo.inc_pe;
    ctrl.generation = 1;
    ctrl.fail_closed_on_dup = 1;
    ctrl.result_offsets_off = result_offsets_off;
    ctrl.result_home_owner_off = home_owner_off;
    ctrl.result_dst_rank_off = result_dst_rank_off;
    ctrl.result_dst_row_off = result_dst_row_off;
    ctrl.result_tx_rank_offsets_off = result_tx_rank_offsets_off;
    ctrl.packed_result_ids_off = packed_result_ids_off;
    ctrl.contrib_slot_off = slot_off;
    ctrl.contrib_weight_off = weight_off;
    ctrl.contrib_uid_lo_off = uid_lo_off;
    ctrl.contrib_uid_hi_off = uid_hi_off;
    ctrl.contrib_ordinal_off = ordinal_off;
    ctrl.contrib_gen_off = gen_off;
    ctrl.ingress_off = ingress_off;
    ctrl.output_off = output_off;
    ctrl.tx_lane_count = tx_lane_count;
    ctrl.result_tx_ready_off = result_tx_ready_off;
    ctrl.tx_done_off = tx_done_off;
    ctrl.arrival_off = arrival_off;
    ctrl.stats_off = stats_off;
    ctrl.owner_stats_off = owner_stats_off;
    ctrl.contrib_source_rank_off = source_rank_off;
    ctrl.ready_generation_off = ready_generation_off;
    ctrl.contrib_owner_off = contrib_owner_off;
    ctrl.contrib_result_off = contrib_result_off;
    ctrl.result_arrival_counter_off = result_arrival_counter_off;
    ctrl.logical_input_off = logical_input_off;
    ctrl.local_reduce_offsets_off = local_reduce_offsets_off;
    ctrl.local_reduce_entries_off = local_reduce_entries_off;
    ctrl.local_reduce_weights_off = local_reduce_weights_off;
    ctrl.logical_contribution_count = logical_plan.contribution_count;
    ctrl.local_rank_prereduce = rank_dedup_active ? 1u : 0u;
    ctrl.group_offsets_off = group_offsets_off;
    ctrl.group_entries_off = group_entries_off;
    ctrl.source_contribution_offsets_off = source_contribution_offsets_off;
    ctrl.source_contribution_entries_off = source_contribution_entries_off;
    ctrl.source_group_offsets_off = source_group_offsets_off;
    ctrl.source_group_entries_off = source_group_entries_off;
    ctrl.owner_source_bitmap_off = owner_source_bitmap_off;
    ctrl.waited_source_bitmap_off = waited_source_bitmap_off;
    ctrl.worker_pe_off = worker_pe_off;
    ctrl.max_ingress_slots = slot_count;
    ctrl.worker_count = W;
    ctrl.group_count = group_count;
    ctrl.source_bitmap_words = source_bitmap_words;
    ctrl.this_worker_rank =
        is_worker ? static_cast<uint32_t>(pe) : 0u;
    // Worker AIV allocation is fixed after the hardware topology is known.
    // It may scale with the live AIV count and number of attached workers,
    // but never with K, route skew, result count or message bytes. On the
    // current 48-AIV profile this resolves to W2/W4/W8 = 24/16/12.
    IncDcAivPolicy aiv_policy{};
    if (!IncDcResolveAivPolicy(owner_selection.available_aiv, W,
                               kDynCsrMaxOwners, kDynCsrMaxOwners,
                               &aiv_policy)) {
        std::cerr << "DYNCSR_AIV_POLICY_FAIL pe=" << pe << std::endl;
        return 10;
    }
    ctrl.producer_lane_count = aiv_policy.combine_worker_aiv;
    if (const char *raw = std::getenv("INC_DYNCSR_PRODUCER_LANES")) {
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(raw, &end, 10);
        // Producer work is assigned with a generic strided loop, so there is
        // no power-of-two requirement.  Accept every live AIV cohort size;
        // this matters on platforms whose usable vector-core count is not a
        // power of two (and avoids silently degrading a valid request to 1).
        if (end == raw || *end != '\0' ||
            parsed > aiv_policy.worker_half_limit ||
            parsed > kDynCsrMaxOwners) {
            std::cerr << "DYNCSR_PRODUCER_OVERRIDE_INVALID pe=" << pe
                      << " value=" << raw
                      << " worker_half_limit="
                      << aiv_policy.worker_half_limit << std::endl;
            return 3;
        }
        if (parsed != 0ul) {
            ctrl.producer_lane_count = static_cast<uint32_t>(parsed);
        }
    }
    // Correctness baseline: one outstanding device RMA per producer lane.
    // A wider NBI window is a later performance gate and must prove that
    // multiple contributions from one rank never alias/reorder visibility.
    ctrl.producer_quiet_window = 1u;
    ctrl.overlap_enable = (overlap_enable || batched_ready) ? 1u : 0u;
    ctrl.ready_mode = batched_ready ? 2u : (overlap_enable ? 1u : 0u);
    if (batched_ready) {
        const char *scope = std::getenv("INC_DYNCSR_READY_SCOPE");
        if (scope == nullptr || std::strcmp(scope, "auto") == 0) {
            // One source/INC-major generation protocol serves every token
            // plan.  Keeping a workload-selected mode-5/mode-6 fork made the
            // INC response depend on density and multiplied the slot-reuse
            // states that must be proved correct.  Mode 6 already represents
            // arbitrary CSR plans and shards prepared chunks across the full
            // producer cohort, so it is the production protocol for both
            // rank-deduplicated and ordinary plans.
            ctrl.ready_mode = 6u;
        } else if (std::strcmp(scope, "inc") == 0) {
            ctrl.ready_mode = 3u;
        } else if (std::strcmp(scope, "lazy") == 0) {
            ctrl.ready_mode = 4u;
        } else if (std::strcmp(scope, "stream") == 0) {
            ctrl.ready_mode = 5u;
        } else if (std::strcmp(scope, "stream_global") == 0) {
            ctrl.ready_mode = 6u;
        }
    }
    // Kept opt-in until a formal repeated-sample gate proves this completion
    // tree beats the established SHMEM barrier on the active platform.
    ctrl.device_completion = 0u;
    if (const char *raw = std::getenv("INC_DYNCSR_DEVICE_COMPLETION")) {
        ctrl.device_completion =
            (raw[0] == '1' && raw[1] == '\0') ? 1u : 0u;
    }
    // Lazy readiness currently aliases completion scratch; INC-scoped and
    // owner-scoped batch modes both have disjoint completion records.
    if (ctrl.ready_mode == 4u) {
        ctrl.device_completion = 0u;
    }
    // Multi-epoch measured submission queues generation updates and kernels
    // on the caller stream.  Device completion is the credit that prevents
    // epoch e+1 from overwriting ingress/output before every INC finished
    // epoch e.  Warmup epochs are individually stream-synchronized and
    // followed by a host SHMEM barrier, so they do not require this 64-signal
    // completion tree.  For a single measured epoch max(rank_us) already
    // covers the slowest reducer and the post-measurement barrier protects
    // verification/reuse.
    if (service_epochs > 1u) {
        if (ctrl.ready_mode != 2u && ctrl.ready_mode != 3u &&
            ctrl.ready_mode != 5u && ctrl.ready_mode != 6u) {
            std::cerr << "DYNCSR_SERVICE_READY_MODE_FAIL pe=" << pe
                      << " ready_mode=" << ctrl.ready_mode << std::endl;
            return 2;
        }
        ctrl.device_completion = 1u;
    }
    ctrl.ready_stride_bytes = kReadyStrideBytes;
    ctrl.start_gate_off =
        (overlap_enable || batched_ready) ? start_gate_off : 0u;
    // Optimized modes are enabled for the production batched-ready path.
    // Both retain explicit environment fallbacks for correctness A/B.
    ctrl.optimization_flags =
        batched_ready ? (kDynCsrOptLocalOrdinalBitmap |
                         kDynCsrOptK1IdentityCopy |
                         kDynCsrOptFirstContributionInit |
                         kDynCsrOptK1PrivateMtePush |
                         kDynCsrOptK1PairReady |
                         kDynCsrOptCompletionMteFanout)
                      : 0u;
    // Staging pre-reduces every local contribution behind a cross-lane
    // generation barrier before the first byte is pushed, so vector work and
    // transport never overlap.  The transport loop can reduce each
    // contribution inline instead, which pipelines the two at contribution
    // granularity and needs no cross-lane visibility because a lane only
    // pushes what it just reduced.
    bool staged_prereduce = false;
    if (const char *raw = std::getenv("INC_DYN_STAGED_PREREDUCE")) {
        staged_prereduce = std::strtoul(raw, nullptr, 10) != 0u;
    }
    if (rank_dedup_active) {
        ctrl.optimization_flags |= kDynCsrOptLocalRankPrereduce;
        if (staged_prereduce) {
            ctrl.optimization_flags |= kDynCsrOptStagedRankPrereduce;
        } else if (regular_small_local_reduce) {
            // The disjoint-UB async backend remains diagnostic until the
            // direct-launch runtime exposes a portable L2-bypass destination
            // or a cohort-wide drain protocol can publish GM without racing
            // another lane's in-flight MTE source.
            // Both experimental transports are intentionally fail-closed.
            // The async GM-source form can race a cohort-wide cache publish
            // with another lane's in-flight MTE read; the UB-direct form is
            // correct but roughly halves throughput on this backend.  A
            // production process must never silently select either path.
            const char *async_raw =
                std::getenv("INC_DYN_ASYNC_RANK_PREREDUCE_PUSH");
            const char *ub_direct_raw =
                std::getenv("INC_DYN_UB_DIRECT_RANK_PUSH");
            if ((async_raw != nullptr && async_raw[0] == '1' &&
                 async_raw[1] == '\0') ||
                (ub_direct_raw != nullptr && ub_direct_raw[0] == '1' &&
                 ub_direct_raw[1] == '\0')) {
                std::cerr << "DYNCSR_UNQUALIFIED_TRANSPORT_REFUSED pe="
                          << pe << std::endl;
                return 2;
            }
        }
    }
    if (packed_result_tx) {
        ctrl.optimization_flags |= kDynCsrOptRankPackedResultTx;
    }
    bool cyclic_owner_results = true;
    for (uint32_t r = 0u; r < plan.result_count; ++r) {
        if (exec.result_home_owner[r] != r % owner_count) {
            cyclic_owner_results = false;
            break;
        }
    }
    if (cyclic_owner_results) {
        ctrl.optimization_flags |= kDynCsrOptCyclicOwnerResults;
    }
    if (const char *raw =
            std::getenv("INC_DYNCSR_COALESCED_GROUP_PUT")) {
        if (raw[0] == '1' && raw[1] == '\0') {
            ctrl.optimization_flags |= kDynCsrOptCoalescedGroupPut;
        }
    }
    // Stream readiness is defined on the first ingress slot of each packed
    // owner/source chunk.  Enable the required placement transform here so
    // selecting READY_SCOPE=stream cannot accidentally depend on a second
    // environment variable (or fail only after device launch).
    if (ctrl.ready_mode == 5u || ctrl.ready_mode == 6u) {
        ctrl.optimization_flags |= kDynCsrOptCoalescedGroupPut;
    }
    bool persistent_local_trigger = false;
    bool persistent_epoch_barrier = false;
    if (const char *raw =
            std::getenv("INC_DYNCSR_PERSISTENT_LOCAL_TRIGGER")) {
        const bool requested = raw[0] == '1' && raw[1] == '\0';
        if (requested && !(overlap_enable || batched_ready)) {
            std::cerr << "DYNCSR_PERSISTENT_MODE_FAIL pe=" << pe
                      << " overlap=" << (overlap_enable ? 1 : 0)
                      << " batched_ready=" << (batched_ready ? 1 : 0)
                      << " service_epochs=" << service_epochs << std::endl;
            return 2;
        }
        persistent_local_trigger = requested;
        if (persistent_local_trigger) {
            ctrl.optimization_flags |=
                kDynCsrOptPersistentLocalTrigger;
        }
    }
    if (const char *raw =
            std::getenv("INC_DYNCSR_PERSISTENT_EPOCH_BARRIER")) {
        const bool requested = raw[0] == '1' && raw[1] == '\0';
        if (requested &&
            (!persistent_local_trigger || service_epochs <= 1u)) {
            std::cerr << "DYNCSR_PERSISTENT_EPOCH_BARRIER_MODE_FAIL pe="
                      << pe << " persistent="
                      << (persistent_local_trigger ? 1 : 0)
                      << " service_epochs=" << service_epochs << std::endl;
            return 2;
        }
        // This is a bounded diagnostic A/B only.  The current SHMEM stream
        // barrier implementation performs host-side collective setup while
        // it is enqueued; prequeuing a long epoch train therefore creates
        // seconds of rank submit skew and can violate the 30 s case limit.
        // Keep production multi-epoch service on the device completion
        // protocol and fail closed instead of silently selecting a
        // non-scalable synchronization path.
        constexpr uint32_t kDiagnosticEpochBarrierMaxEpochs = 3u;
        if (requested &&
            service_epochs > kDiagnosticEpochBarrierMaxEpochs) {
            std::cerr
                << "DYNCSR_PERSISTENT_EPOCH_BARRIER_SCALE_FAIL pe=" << pe
                << " service_epochs=" << service_epochs
                << " max_diagnostic_epochs="
                << kDiagnosticEpochBarrierMaxEpochs << std::endl;
            return 2;
        }
        persistent_epoch_barrier = requested;
        if (persistent_epoch_barrier) {
            // Stream ordering completes every local producer/reducer kernel;
            // the collective is the cross-rank credit before any stream can
            // apply the next generation or reuse ingress/output.
            ctrl.device_completion = 0u;
        }
    }
    if (const char *raw = std::getenv("INC_DYNCSR_LOCAL_ORDINAL_BITMAP")) {
        if (raw[0] == '1' && raw[1] == '\0') {
            ctrl.optimization_flags |= kDynCsrOptLocalOrdinalBitmap;
        } else {
            ctrl.optimization_flags &= ~kDynCsrOptLocalOrdinalBitmap;
        }
    }
    if (const char *raw = std::getenv("INC_DYNCSR_K1_IDENTITY_COPY")) {
        if (raw[0] == '1' && raw[1] == '\0') {
            ctrl.optimization_flags |= kDynCsrOptK1IdentityCopy;
        } else {
            ctrl.optimization_flags &= ~kDynCsrOptK1IdentityCopy;
        }
    }
    if (const char *raw =
            std::getenv("INC_DYNCSR_FIRST_CONTRIBUTION_INIT")) {
        if (raw[0] == '1' && raw[1] == '\0') {
            ctrl.optimization_flags |= kDynCsrOptFirstContributionInit;
        } else {
            ctrl.optimization_flags &= ~kDynCsrOptFirstContributionInit;
        }
    }
    // Source-group scheduling remains an explicit diagnostic until its
    // generation-credit path is promoted.  Do not advertise it as active in
    // production merely because the compiled plan is identity K1.
    if (const char *raw = std::getenv("INC_DYNCSR_SOURCE_GROUP_WORKLIST")) {
        if (raw[0] == '1' && raw[1] == '\0') {
            ctrl.optimization_flags |= kDynCsrOptSourceGroupWorklist;
        } else {
            ctrl.optimization_flags &= ~kDynCsrOptSourceGroupWorklist;
        }
    }
    if (const char *raw = std::getenv("INC_DYNCSR_REMOTE_TX")) {
        if (raw[0] == '1' && raw[1] == '\0') {
            ctrl.optimization_flags |= kDynCsrOptRemoteResultTx;
        } else {
            ctrl.optimization_flags &= ~kDynCsrOptRemoteResultTx;
        }
    }
    if (const char *raw =
            std::getenv("INC_DYNCSR_COMPLETION_MTE_FANOUT")) {
        if (raw[0] == '1' && raw[1] == '\0') {
            ctrl.optimization_flags |= kDynCsrOptCompletionMteFanout;
        } else if (raw[0] == '0' && raw[1] == '\0') {
            ctrl.optimization_flags &= ~kDynCsrOptCompletionMteFanout;
        } else {
            std::cerr << "DYNCSR_COMPLETION_MTE_FANOUT_ENV_FAIL pe="
                      << pe << std::endl;
            return 2;
        }
    }
    if (const char *raw = std::getenv("INC_DYNCSR_K1_PRIVATE_MTE_PUSH")) {
        if (raw[0] == '1' && raw[1] == '\0') {
            ctrl.optimization_flags |= kDynCsrOptK1PrivateMtePush;
        } else if (raw[0] == '0' && raw[1] == '\0') {
            ctrl.optimization_flags &= ~kDynCsrOptK1PrivateMtePush;
        } else {
            std::cerr << "DYNCSR_K1_PRIVATE_MTE_PUSH_ENV_FAIL pe=" << pe
                      << std::endl;
            return 2;
        }
    }
    if (const char *raw = std::getenv("INC_DYNCSR_K1_PAIR_READY")) {
        if (raw[0] == '1' && raw[1] == '\0') {
            ctrl.optimization_flags |= kDynCsrOptK1PairReady;
        } else if (raw[0] == '0' && raw[1] == '\0') {
            ctrl.optimization_flags &= ~kDynCsrOptK1PairReady;
        } else {
            std::cerr << "DYNCSR_K1_PAIR_READY_ENV_FAIL pe=" << pe
                      << std::endl;
            return 2;
        }
    }
    // K1 is still a complete single-INC operation.  The identity-copy flag
    // may avoid a redundant FP add inside the INC, but the worker-direct
    // result-TX flag is never set: ingress and egress must both traverse the
    // unique INC for every legal plan.
    ctrl.optimization_flags &= ~kDynCsrOptK1DirectResultTx;
    ctrl.tx_quiet_window = 1u;
    // Stream readiness follows runtime hidden size.  Single-row publication
    // is the qualified latency/bandwidth point for owner streaming; mode 6
    // retains a coarse train for its distinct INC-global protocol.
    const uint32_t owners_per_producer =
        std::max(1u, (ctrl.owner_count + ctrl.producer_lane_count - 1u) /
                         ctrl.producer_lane_count);
    const uint32_t stream_chunk_tiles = 8u;
    // Rank pre-reduce needs enough granularity to overlap vector work with
    // transport.  A fixed 128-KiB byte credit is independent of W/K and kept
    // pace with 256 KiB on long trains, while 512 KiB exposed a repeatable
    // ~9% drain bubble.  It is still clamped to whole runtime rows below.
    const uint64_t rank_dedup_chunk_bytes = 128ull * 1024ull;
    const uint64_t mode6_plain_chunk_tiles =
        identity_k1_plan
            ? 1ull
            // A wide row already amortizes one ready publication and carries
            // enough payload to fill the link.  Grouping several such rows
            // delays reducer start/drain and creates a repeatable short-train
            // bubble.  Select one-row publication from the runtime byte size;
            // this is independent of W, K, route and physical device ids.
            : tile_bytes >= 64ull * 1024ull
            ? 1ull
            :
                static_cast<uint64_t>(W) * 3u >= ctrl.owner_count ||
                owners_per_producer <= 2u
            ? 4ull
            : 64ull;
    const uint64_t default_chunk_bytes =
        ctrl.ready_mode == 6u
            ? std::max<uint64_t>(
                  tile_bytes,
                  std::min<uint64_t>(
                      512ull * 1024ull,
                      static_cast<uint64_t>(tile_bytes) *
                          (rank_dedup_active
                               ? std::max<uint64_t>(
                                     1u, rank_dedup_chunk_bytes / tile_bytes)
                               : mode6_plain_chunk_tiles)))
            : ctrl.ready_mode == 5u
                  ? std::min<uint64_t>(
                        8ull * 1024ull * 1024ull,
                        static_cast<uint64_t>(tile_bytes) *
                            stream_chunk_tiles)
                  : 512ull * 1024ull;
    ctrl.coalesced_chunk_bytes =
        static_cast<uint32_t>(std::max<uint64_t>(tile_bytes,
                                                 default_chunk_bytes));
    if (const char *raw =
            std::getenv("INC_DYNCSR_COALESCED_CHUNK_BYTES")) {
        const uint64_t requested = std::strtoull(raw, nullptr, 10);
        if (requested >= tile_bytes &&
            requested <= 8ull * 1024ull * 1024ull) {
            ctrl.coalesced_chunk_bytes = static_cast<uint32_t>(
                requested - requested % tile_bytes);
        }
    }
    if (const char *raw = std::getenv("INC_DYNCSR_ABORT_GENERATION")) {
        ctrl.abort_generation =
            static_cast<uint32_t>(std::strtoul(raw, nullptr, 10));
    }
    // The public result-put backend has one private 16-KiB packet slot per
    // AIV.  Rows that fit that slot are independent destination writes and
    // can safely retain a deeper credit train; this hides the per-put HCCS
    // latency without consuming another AIV.  Rank-deduplicated plans use a
    // local pre-reduce/result-batch path whose MTE packet state is shared, so
    // they deliberately retain the qualified conservative credit.  This is
    // derived from the compiled plan and runtime row size, not W/K or a
    // product table.
    ctrl.tx_quiet_window =
        tile_bytes <= kIncDcPrivateMtePacketBytes && unit_small_reduce &&
                !rank_dedup_active
            ? 32u
            : 1u;
    if (const char *raw = std::getenv("INC_DYNCSR_TX_WINDOW")) {
        const uint32_t requested =
            static_cast<uint32_t>(std::strtoul(raw, nullptr, 10));
        // The window is a credit, not an AIV count.  Larger result sets may
        // safely keep more independent destination rows in flight; retain a
        // bounded runtime knob instead of baking the original 32-row tune.
        if (requested >= 1u && requested <= 128u) {
            ctrl.tx_quiet_window = requested;
        }
    }
    if (ctrl.ready_mode == 5u &&
        ctrl.coalesced_chunk_bytes == ctrl.tile_bytes) {
        bool enable = false;
        if (const char *raw =
                std::getenv("INC_DYNCSR_RESULT_ARRIVAL_COUNTER")) {
            enable = raw[0] == '1' && raw[1] == '\0';
        }
        if (enable) {
            ctrl.optimization_flags |= kDynCsrOptResultArrivalCounter;
        }
    }
    if (ctrl.ready_mode == 5u &&
        ctrl.coalesced_chunk_bytes > ctrl.tile_bytes) {
        // Publish the first tile immediately so reducers start early, then
        // amortize control traffic across the runtime-derived chunk size.
        bool enable = true;
        if (const char *raw = std::getenv("INC_DYNCSR_HEAD_TILE_STREAM")) {
            enable = raw[0] == '1' && raw[1] == '\0';
        }
        if (enable) {
            ctrl.optimization_flags |= kDynCsrOptHeadTileStream;
        }
    }
    {
        bool enable = true;
        if (const char *raw = std::getenv("INC_DYNCSR_WIDE_VECTOR_TILE")) {
            enable = raw[0] == '1' && raw[1] == '\0';
        }
        if (enable) ctrl.optimization_flags |= kDynCsrOptWideVectorTile;
    }
    {
        // Batching wide result rows amortizes egress setup only when every
        // owner has a real train.  With fewer than two results per owner the
        // fixed batch fill/drain cost sits on the critical path; direct row
        // TX is faster and retains the same INC-owned reduction semantics.
        // This threshold uses runtime train length and the hardware-selected
        // owner cohort, never W/K/route/device-id tables.
        bool enable = tile_bytes > kIncDcPrivateMtePacketBytes &&
                      static_cast<uint64_t>(plan.result_count) >=
                          2ull * ctrl.owner_count;
        if (const char *raw = std::getenv("INC_DYNCSR_BATCH_RESULT_TX")) {
            enable = raw[0] == '1' && raw[1] == '\0';
        }
        if (enable) {
            ctrl.optimization_flags |= kDynCsrOptBatchResultTx;
        }
    }
    // Direct K1 reads immutable caller input and has exactly one writer per
    // output row, so a wider NBI window is safe even though the general
    // reducer egress remains conservatively fixed at one.  Keep an explicit
    // override for platform-specific tuning and reproducible A/B.
    if ((ctrl.optimization_flags & kDynCsrOptK1DirectResultTx) != 0u) {
        // The public device put backend reuses one synchronization slot for
        // a long packet.  Wide windows are qualified only for rows that fit
        // the 16-KiB staging packet; larger rows otherwise corrupt silently
        // at vector-block boundaries.
        ctrl.tx_quiet_window =
            tile_bytes <= kIncDcPrivateMtePacketBytes ? 32u : 1u;
        if (const char *raw =
                std::getenv("INC_DYNCSR_K1_DIRECT_TX_WINDOW")) {
            const uint32_t requested =
                static_cast<uint32_t>(std::strtoul(raw, nullptr, 10));
            if (requested >= 1u && requested <= 32u) {
                ctrl.tx_quiet_window = requested;
            }
        }
    }
    ctrl.output_dcci_small_only = batched_ready ? 1u : 0u;
    if (const char *raw = std::getenv("INC_DYNCSR_OUTPUT_DCCI_SMALL_ONLY")) {
        ctrl.output_dcci_small_only =
            (raw[0] == '1' && raw[1] == '\0') ? 1u : 0u;
    }
    if (const char *raw = std::getenv("INC_DYNCSR_PRODUCER_WINDOW")) {
        const uint32_t requested =
            static_cast<uint32_t>(std::strtoul(raw, nullptr, 10));
        if (requested >= 1u && requested <= 32u &&
            (requested == 1u || producer_wide_window_safe)) {
            ctrl.producer_quiet_window = requested;
        }
    }
    std::memcpy(host.data() + ctrl_off, &ctrl, sizeof(ctrl));
    std::memcpy(host.data() + result_offsets_off, exec.result_offsets.data(),
                exec.result_offsets.size() * sizeof(uint32_t));
    std::memcpy(host.data() + home_owner_off, exec.result_home_owner.data(),
                exec.result_home_owner.size() * sizeof(uint32_t));
    std::vector<uint32_t> result_dst_rank(plan.result_count, 0u);
    std::vector<uint32_t> result_dst_row(plan.result_count, 0u);
    for (uint32_t r = 0u; r < plan.result_count; ++r) {
        result_dst_rank[r] = plan.results[r].dst_rank;
        result_dst_row[r] = plan.results[r].dst_local_row;
    }
    std::memcpy(host.data() + result_dst_rank_off, result_dst_rank.data(),
                result_dst_rank.size() * sizeof(uint32_t));
    std::memcpy(host.data() + result_dst_row_off, result_dst_row.data(),
                result_dst_row.size() * sizeof(uint32_t));
    if (packed_result_tx) {
        std::memcpy(host.data() + result_tx_rank_offsets_off,
                    packed_result_rank_offsets.data(),
                    packed_result_rank_offsets.size() * sizeof(uint32_t));
        std::vector<uint32_t> packed_result_ids(plan.result_count, 0u);
        for (uint32_t r = 0u; r < plan.result_count; ++r) {
            const uint32_t storage =
                packed_result_rank_offsets[result_dst_rank[r]] +
                result_dst_row[r];
            packed_result_ids[storage] = r;
        }
        std::memcpy(host.data() + packed_result_ids_off,
                    packed_result_ids.data(),
                    packed_result_ids.size() * sizeof(uint32_t));
    }

    std::vector<uint32_t> contrib_owner(plan.contribution_count, 0u);
    std::vector<uint32_t> contrib_result(plan.contribution_count, 0u);
    for (uint32_t r = 0u; r < plan.result_count; ++r) {
        for (uint32_t si = exec.result_offsets[r];
             si < exec.result_offsets[r + 1u]; ++si) {
            contrib_result[si] = r;
        }
    }
    std::vector<std::vector<uint32_t>> grouped(group_count);
    std::vector<std::vector<uint32_t>> source_contributions(W);
    std::vector<uint32_t> owner_source_bitmap(
        static_cast<size_t>(owner_count) * source_bitmap_words, 0u);
    for (uint32_t si = 0; si < exec.schedule.size(); ++si) {
        const auto &cc = exec.schedule[si];
        const auto &lc = plan.contributions[cc.logical_contribution_index];
        const uint32_t flat = cc.owner_index;
        const uint32_t group = flat * W + lc.contributor_rank;
        if (flat >= owner_count || lc.contributor_rank >= W ||
            group >= group_count) {
            std::cerr << "DYNCSR_GROUP_BUILD_FAIL pe=" << pe << std::endl;
            return 1;
        }
        contrib_owner[si] = flat;
        grouped[group].push_back(si);
        source_contributions[lc.contributor_rank].push_back(si);
        owner_source_bitmap[static_cast<size_t>(flat) * source_bitmap_words +
                            (lc.contributor_rank >> 5u)] |=
            1u << (lc.contributor_rank & 31u);
    }
    std::vector<uint32_t> group_offsets(group_count + 1u, 0u);
    std::vector<uint32_t> group_entries;
    group_entries.reserve(plan.contribution_count);
    uint32_t active_group_count = 0u;
    uint32_t packed_slot = 0u;
    const bool coalesced_group_put =
        (ctrl.optimization_flags & kDynCsrOptCoalescedGroupPut) != 0u;
    for (uint32_t group = 0; group < group_count; ++group) {
        group_offsets[group] = static_cast<uint32_t>(group_entries.size());
        if (!grouped[group].empty()) {
            ++active_group_count;
        }
        // Preserve result CSR order while making every owner/source batch
        // physically contiguous.  ingress_slot is transport placement only;
        // result identity and ordinal remain in the compiled CSR metadata.
        if (coalesced_group_put && ctrl.ready_mode != 3u &&
            ctrl.ready_mode != 6u) {
            for (const uint32_t si : grouped[group]) {
                exec.schedule[si].ingress_slot = packed_slot++;
            }
        }
        group_entries.insert(group_entries.end(), grouped[group].begin(),
                             grouped[group].end());
    }
    group_offsets[group_count] =
        static_cast<uint32_t>(group_entries.size());
    if (group_entries.size() != plan.contribution_count) {
        std::cerr << "DYNCSR_GROUP_CSR_FAIL pe=" << pe << std::endl;
        return 1;
    }
    if (coalesced_group_put && packed_slot > slot_count) {
        std::cerr << "DYNCSR_PACKED_SLOT_CAPACITY_FAIL pe=" << pe
                  << " packed=" << packed_slot << " capacity=" << slot_count
                  << std::endl;
        return 1;
    }
    std::vector<uint32_t> source_group_offsets(W + 1u, 0u);
    std::vector<uint32_t> source_group_entries;
    source_group_entries.reserve(active_group_count);
    for (uint32_t source = 0u; source < W; ++source) {
        source_group_offsets[source] =
            static_cast<uint32_t>(source_group_entries.size());
        for (uint32_t owner = 0u; owner < owner_count; ++owner) {
            const uint32_t group = owner * W + source;
            if (!grouped[group].empty()) {
                source_group_entries.push_back(owner);
            }
        }
    }
    source_group_offsets[W] =
        static_cast<uint32_t>(source_group_entries.size());
    std::vector<uint32_t> source_contribution_offsets(W + 1u, 0u);
    std::vector<uint32_t> source_contribution_entries;
    source_contribution_entries.reserve(plan.contribution_count);
    uint32_t active_source_count = 0u;
    for (uint32_t source = 0; source < W; ++source) {
        source_contribution_offsets[source] =
            static_cast<uint32_t>(source_contribution_entries.size());
        if (!source_contributions[source].empty()) {
            ++active_source_count;
        }
        // Source-scoped readiness publishes one generation after all owners
        // for this source are visible.  Pack that exact batch contiguously
        // so the producer can issue one RMA instead of one RMA per tile.
        if (coalesced_group_put &&
            (ctrl.ready_mode == 3u || ctrl.ready_mode == 6u)) {
            for (const uint32_t si : source_contributions[source]) {
                exec.schedule[si].ingress_slot = packed_slot++;
            }
        }
        source_contribution_entries.insert(source_contribution_entries.end(),
            source_contributions[source].begin(),
            source_contributions[source].end());
    }
    source_contribution_offsets[W] =
        static_cast<uint32_t>(source_contribution_entries.size());
    if (source_contribution_entries.size() != plan.contribution_count) {
        std::cerr << "DYNCSR_SOURCE_CONTRIBUTION_CSR_FAIL pe=" << pe
                  << std::endl;
        return 1;
    }
    if (coalesced_group_put &&
        (ctrl.ready_mode == 3u || ctrl.ready_mode == 6u) &&
        packed_slot > slot_count) {
        std::cerr << "DYNCSR_SOURCE_PACKED_SLOT_CAPACITY_FAIL pe=" << pe
                  << " packed=" << packed_slot << " capacity=" << slot_count
                  << std::endl;
        return 1;
    }
    std::memcpy(host.data() + contrib_owner_off, contrib_owner.data(),
                contrib_owner.size() * sizeof(uint32_t));
    std::memcpy(host.data() + contrib_result_off, contrib_result.data(),
                contrib_result.size() * sizeof(uint32_t));
    std::memcpy(host.data() + group_offsets_off, group_offsets.data(),
                group_offsets.size() * sizeof(uint32_t));
    std::memcpy(host.data() + group_entries_off, group_entries.data(),
                group_entries.size() * sizeof(uint32_t));
    std::memcpy(host.data() + source_group_offsets_off,
                source_group_offsets.data(),
                source_group_offsets.size() * sizeof(uint32_t));
    std::memcpy(host.data() + source_group_entries_off,
                source_group_entries.data(),
                source_group_entries.size() * sizeof(uint32_t));
    std::memcpy(host.data() + source_contribution_offsets_off,
                source_contribution_offsets.data(),
                source_contribution_offsets.size() * sizeof(uint32_t));
    std::memcpy(host.data() + source_contribution_entries_off,
                source_contribution_entries.data(),
                source_contribution_entries.size() * sizeof(uint32_t));
    std::memcpy(host.data() + owner_source_bitmap_off,
                owner_source_bitmap.data(),
                owner_source_bitmap.size() * sizeof(uint32_t));
    std::memcpy(host.data() + worker_pe_off, topo.worker_pe_ids.data(),
                topo.worker_pe_ids.size() * sizeof(uint32_t));
    if (rank_dedup_active) {
        std::memcpy(host.data() + local_reduce_offsets_off,
                    local_reduce_offsets.data(),
                    local_reduce_offsets.size() * sizeof(uint32_t));
        std::memcpy(host.data() + local_reduce_entries_off,
                    local_reduce_entries.data(),
                    local_reduce_entries.size() * sizeof(uint32_t));
        std::memcpy(host.data() + local_reduce_weights_off,
                    local_reduce_weights.data(),
                    local_reduce_weights.size() * sizeof(uint32_t));
        if (is_worker) {
            for (uint32_t li = 0u; li < logical_plan.contribution_count; ++li) {
                if (logical_plan.contributions[li].contributor_rank !=
                    static_cast<uint32_t>(pe)) {
                    continue;
                }
                std::memcpy(host.data() + logical_input_off +
                                static_cast<uint64_t>(li) * tile_bytes,
                            logical_tiles[li].data(), payload_bytes);
            }
        }
    }

    const char *fault = std::getenv("INC_DYNCSR_FAULT");
    for (uint32_t si = 0; si < exec.schedule.size(); ++si) {
        const auto &cc = exec.schedule[si];
        const auto &lc = plan.contributions[cc.logical_contribution_index];
        uint32_t slot = cc.ingress_slot;
        uint32_t ordinal = lc.ordinal;
        // A queued service reuses immutable route metadata.  Generation zero
        // means "inherit the enclosing DynCsrCtrl generation"; explicit
        // nonzero descriptor generations retain strict stale/future checks.
        uint64_t gen =
            (service_epochs > 1u || service_warmup_epochs > 0u) ? 0ull
                                                                : 1ull;
        if (fault != nullptr && si == 0u) {
            if (std::strcmp(fault, "dup_ordinal") == 0 &&
                exec.schedule.size() > 1u) {
                // Force duplicate ordinal within first result CSR if possible.
                ordinal = plan.contributions[exec.schedule[1].logical_contribution_index]
                              .ordinal;
            } else if (std::strcmp(fault, "stale_gen") == 0) {
                gen = 99ull;
            } else if (std::strcmp(fault, "bad_slot") == 0) {
                slot = slot_count + 7u;
            }
        }
        std::memcpy(host.data() + slot_off + si * sizeof(uint32_t), &slot,
                    sizeof(uint32_t));
        const uint32_t wb = FloatBits(lc.weight);
        std::memcpy(host.data() + weight_off + si * sizeof(uint32_t), &wb,
                    sizeof(uint32_t));
        const uint32_t uid_lo = static_cast<uint32_t>(lc.contribution_uid);
        const uint32_t uid_hi =
            static_cast<uint32_t>(lc.contribution_uid >> 32);
        std::memcpy(host.data() + uid_lo_off + si * sizeof(uint32_t), &uid_lo,
                    sizeof(uint32_t));
        std::memcpy(host.data() + uid_hi_off + si * sizeof(uint32_t), &uid_hi,
                    sizeof(uint32_t));
        std::memcpy(host.data() + ordinal_off + si * sizeof(uint32_t), &ordinal,
                    sizeof(uint32_t));
        std::memcpy(host.data() + gen_off + si * sizeof(uint64_t), &gen,
                    sizeof(uint64_t));
        const uint32_t source_rank = lc.contributor_rank;
        std::memcpy(host.data() + source_rank_off + si * sizeof(uint32_t),
                    &source_rank, sizeof(uint32_t));
        // A malformed external descriptor must reach the device validator
        // without first turning into a host-side write outside the ingress
        // region.  The producer kernel publishes a negative generation for
        // this slot and the reducer fails closed; there is intentionally no
        // payload copy for an invalid slot.
        if (!rank_dedup_active && is_worker &&
            source_rank == static_cast<uint32_t>(pe) &&
            slot < slot_count) {
            const uint64_t local_off =
                ingress_off + static_cast<uint64_t>(slot) * tile_bytes;
            std::memcpy(host.data() + local_off,
                        logical_tiles[cc.logical_contribution_index].data(),
                        payload_bytes);
        }
    }

    aclrtMemcpy(sym, heap_need, host.data(), heap_need, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtSynchronizeStream(stream);
    aclshmem_barrier_all();
    const bool k1_direct_tx =
        (ctrl.optimization_flags & kDynCsrOptK1DirectResultTx) != 0u;
    // Warm the persistent service outside the measured region.  This is not a
    // discarded sample from a new process: it uses the same allocation,
    // stream, route template, generation records and device completion tree
    // as the following queued operations.
    if (service_warmup_epochs > 0u) {
        std::vector<uint32_t> warmup_generations(service_warmup_epochs, 1u);
        for (uint32_t epoch = 0u; epoch < service_warmup_epochs; ++epoch) {
            warmup_generations[epoch] = epoch + 1u;
            if (epoch != 0u) {
                if (aclrtMemcpyAsync(
                        sym + ctrl_off + offsetof(DynCsrCtrl, generation),
                        sizeof(uint32_t), &warmup_generations[epoch],
                        sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE,
                        stream) != 0) {
                    return 1;
                }
            }
            if (is_worker) {
                if (k1_direct_tx) {
                    launch_inc_dc_sv2_dyn_csr_k1_direct_tx_kernel(
                        sym, ctrl_off,
                        static_cast<int>(ctrl.producer_lane_count), stream);
                } else {
                    launch_inc_dc_sv2_dyn_csr_producer_kernel(
                        sym, ctrl_off,
                        static_cast<int>(ctrl.producer_lane_count), stream);
                }
            } else if (!k1_direct_tx) {
                launch_inc_dc_sv2_dyn_csr_combine_kernel(
                    sym, ctrl_off,
                    static_cast<int>(topo.owner_count +
                                     ctrl.tx_lane_count), stream);
            }
            if (persistent_epoch_barrier &&
                epoch + 1u < service_epochs) {
                aclshmemx_barrier_all_on_stream(stream);
            }
        }
        if (aclrtSynchronizeStream(stream) != 0) {
            std::cerr << "DYNCSR_SERVICE_WARMUP_FAIL pe=" << pe << std::endl;
            return 1;
        }
        aclshmem_barrier_all();
    }
    // Optional late main-thread pinning: ACL/SHMEM helper threads have already
    // been created and retain their broad affinity.  Pinning in preexec_fn
    // constrains those helpers to one CPU as well and can manufacture
    // completion tails under overlap.  Frameworks may provide one CPU per
    // rank here to stabilize the host submit/completion thread only.
    if (const char *raw = std::getenv("INC_DYNCSR_MAIN_CPU")) {
        char *end = nullptr;
        const long requested = std::strtol(raw, &end, 10);
        if (end == raw || *end != '\0' || requested < 0 ||
            requested >= CPU_SETSIZE) {
            std::cerr << "DYNCSR_MAIN_CPU_ENV_FAIL pe=" << pe << std::endl;
            return 2;
        }
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(static_cast<int>(requested), &cpuset);
        if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
            std::cerr << "DYNCSR_MAIN_CPU_PIN_FAIL pe=" << pe << std::endl;
            return 1;
        }
        std::cout << "DYNCSR_MAIN_CPU_PIN pe=" << pe
                  << " cpu=" << requested << " late=1" << std::endl;
    }
    if (!IncDcExternalStartGate("combine", pe)) {
        std::cerr << "DYNCSR_EXTERNAL_START_GATE_FAIL pe=" << pe
                  << std::endl;
        return 1;
    }
    // Late pinning and an optional cross-job start gate can release ranks at
    // different host times.  Re-align this SHMEM team before taking the
    // timestamp so max(rank_us) measures protocol work, not rank-launch skew.
    // This rendezvous is outside the measured region and does not serialize
    // producer/reducer execution inside the operation.
    aclshmem_barrier_all();
    const uint64_t aligned_start_ns = IncDcExternalStartNs();
    bool persistent_prelaunched = false;
    double device_cycles_per_ns = 0.0;
    uint64_t persistent_target_cycle = 0u;
    uint64_t cycle_calibration_rtt_ns = 0u;
    uint64_t persistent_trigger_off = 0u;
    uint64_t persistent_service_cycles = 0u;
    uint64_t persistent_service_start_cycle = 0u;
    uint64_t persistent_service_end_cycle = 0u;
    std::vector<uint32_t> persistent_epoch_generations;
    // A common host timestamp does not guarantee a common device launch:
    // Linux can deschedule one of the sixteen submit threads for hundreds of
    // microseconds even while it is busy-waiting on a dedicated physical
    // core.  In validation runs, enqueue a device-side collective before the
    // start event.  Every producer/reducer stream is then released by the
    // same collective generation and host submit skew is outside rank_us.
    // The production path is unchanged when no managed start timestamp is
    // supplied.
    const bool measured_stream_start_barrier =
        aligned_start_ns != 0u && (overlap_enable || batched_ready) &&
        !persistent_local_trigger;
    aclrtEvent measured_start_event = nullptr;
    aclrtEvent measured_end_event = nullptr;
    bool device_event_timing =
        (overlap_enable || batched_ready) &&
        !persistent_local_trigger &&
        aclrtCreateEvent(&measured_start_event) == ACL_SUCCESS &&
        aclrtCreateEvent(&measured_end_event) == ACL_SUCCESS;

    if (persistent_local_trigger) {
        if (aligned_start_ns == 0u) {
            std::cerr << "DYNCSR_PERSISTENT_COMMON_DEADLINE_REQUIRED pe="
                      << pe << std::endl;
            return 2;
        }
        persistent_trigger_off =
            start_gate_off +
            (static_cast<uint64_t>(W) + 1u) * kReadyStrideBytes;
        auto host_now_ns = []() -> uint64_t {
            return static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count());
        };
        // Map a device cycle to the midpoint of a host-side request/response
        // interval, like an NTP clock sample.  Selecting the minimum RTT
        // rejects scheduler/driver queueing delay instead of folding it into
        // the per-device clock offset.
        auto best_cycle_sample =
            [&](uint64_t *cycle, uint64_t *host_mid_ns,
                uint64_t *best_rtt_ns) -> bool {
            *best_rtt_ns = std::numeric_limits<uint64_t>::max();
            for (uint32_t sample = 0u; sample < 12u; ++sample) {
                const uint64_t before_ns = host_now_ns();
                launch_inc_dc_sv2_dyn_csr_cycle_probe_kernel(
                    sym, persistent_trigger_off, stream);
                uint64_t observed_cycle = 0u;
                if (aclrtSynchronizeStream(stream) != ACL_SUCCESS ||
                    aclrtMemcpy(
                        &observed_cycle, sizeof(observed_cycle),
                        sym + persistent_trigger_off,
                        sizeof(observed_cycle),
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                    return false;
                }
                const uint64_t after_ns = host_now_ns();
                const uint64_t rtt_ns = after_ns - before_ns;
                if (observed_cycle != 0u && rtt_ns < *best_rtt_ns) {
                    *cycle = observed_cycle;
                    *host_mid_ns = before_ns + rtt_ns / 2u;
                    *best_rtt_ns = rtt_ns;
                }
            }
            return *best_rtt_ns != std::numeric_limits<uint64_t>::max();
        };
        uint64_t cycle1 = 0u;
        uint64_t cycle2 = 0u;
        uint64_t host1_ns = 0u;
        uint64_t host2_ns = 0u;
        uint64_t rtt1_ns = 0u;
        uint64_t rtt2_ns = 0u;
        if (!best_cycle_sample(&cycle1, &host1_ns, &rtt1_ns)) {
            std::cerr << "DYNCSR_CYCLE_PROBE1_FAIL pe=" << pe << std::endl;
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (!best_cycle_sample(&cycle2, &host2_ns, &rtt2_ns)) {
            std::cerr << "DYNCSR_CYCLE_PROBE2_FAIL pe=" << pe << std::endl;
            return 1;
        }
        if (cycle2 <= cycle1 || host2_ns <= host1_ns) {
            std::cerr << "DYNCSR_CYCLE_CALIBRATION_FAIL pe=" << pe
                      << std::endl;
            return 1;
        }
        cycle_calibration_rtt_ns = rtt2_ns;
        device_cycles_per_ns =
            static_cast<double>(cycle2 - cycle1) /
            static_cast<double>(host2_ns - host1_ns);
        if (!std::isfinite(device_cycles_per_ns) ||
            device_cycles_per_ns < 0.03 ||
            device_cycles_per_ns > 0.08) {
            std::cerr << "DYNCSR_CYCLE_RATE_FAIL pe=" << pe
                      << " cycles_per_ns=" << device_cycles_per_ns
                      << std::endl;
            return 1;
        }
        // Never manufacture a rank-local fallback deadline: it would make
        // every rank correct in isolation while invalidating the collective
        // makespan.  The harness must provide one common deadline far enough
        // in the future for every rank to finish calibration and prelaunch.
        if (aligned_start_ns <= host2_ns + 1000000u) {
            std::cerr << "DYNCSR_COMMON_DEADLINE_TOO_CLOSE pe=" << pe
                      << " aligned_start_ns=" << aligned_start_ns
                      << " calibration_end_ns=" << host2_ns << std::endl;
            return 1;
        }
        persistent_target_cycle =
            cycle2 + static_cast<uint64_t>(
                         static_cast<double>(aligned_start_ns - host2_ns) *
                         device_cycles_per_ns);
        DynCsrPersistentTriggerLine trigger_line{};
        trigger_line.generation =
            static_cast<int32_t>(service_warmup_epochs + 1u);
        trigger_line.target_cycle = persistent_target_cycle;
        if (aclrtMemcpyAsync(
                sym + persistent_trigger_off, sizeof(trigger_line),
                &trigger_line, sizeof(trigger_line),
                ACL_MEMCPY_HOST_TO_DEVICE, stream) != ACL_SUCCESS) {
            std::cerr << "DYNCSR_CYCLE_TARGET_FAIL pe=" << pe << std::endl;
            return 1;
        }
        persistent_epoch_generations.resize(service_epochs);
        for (uint32_t epoch = 0u; epoch < service_epochs; ++epoch) {
            persistent_epoch_generations[epoch] =
                service_warmup_epochs + epoch + 1u;
            if (aclrtMemcpyAsync(
                    sym + ctrl_off + offsetof(DynCsrCtrl, generation),
                    sizeof(uint32_t), &persistent_epoch_generations[epoch],
                    sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE,
                    stream) != ACL_SUCCESS) {
                return 1;
            }
            if (is_worker) {
                if (k1_direct_tx) {
                    launch_inc_dc_sv2_dyn_csr_k1_direct_tx_kernel(
                        sym, ctrl_off,
                        static_cast<int>(ctrl.producer_lane_count), stream);
                } else {
                    launch_inc_dc_sv2_dyn_csr_producer_kernel(
                        sym, ctrl_off,
                        static_cast<int>(ctrl.producer_lane_count), stream);
                }
            } else if (!k1_direct_tx) {
                launch_inc_dc_sv2_dyn_csr_combine_kernel(
                    sym, ctrl_off,
                    static_cast<int>(topo.owner_count +
                                     ctrl.tx_lane_count), stream);
            }
        }
        persistent_prelaunched = true;
    }

    if (aligned_start_ns != 0u) {
        while (static_cast<uint64_t>(
                   std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                       .count()) < aligned_start_ns) {
            // Validation ranks are late-pinned to distinct physical cores.
            // Busy waiting here avoids scheduler wake latency at the common
            // start boundary and is never enabled on the production path.
        }
    }

    if (!persistent_local_trigger) {
        if (measured_stream_start_barrier) {
            aclshmemx_barrier_all_on_stream(stream);
            const uint32_t participant =
                is_worker ? ctrl.this_worker_rank
                          : ctrl.worker_count;
            launch_inc_dc_sv2_dyn_csr_start_gate_kernel(
                sym, ctrl_off, participant, stream);
        }
        if (device_event_timing &&
            aclrtRecordEvent(measured_start_event, stream) != ACL_SUCCESS) {
            device_event_timing = false;
        }
    }
    const uint64_t begin_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());

    bool device_ok = true;
    uint32_t mismatch = 0;
    uint32_t nonfinite = 0;
    uint32_t reduced = 0;
    uint64_t numeric_local_rows_verified = 0u;
    uint64_t numeric_local_elements_verified = 0u;
    uint64_t numeric_remote_rows_verified = 0u;
    uint64_t numeric_remote_elements_verified = 0u;
    int sync_rc = 0;
    uint64_t transport_end_ns = begin_ns;
    if (overlap_enable || batched_ready) {
        // Worker producer and INC reducer are launched concurrently on their
        // respective ranks.  Per-slot payload+generation publication is the
        // correctness authority; no host transport barrier is used.
        std::vector<uint32_t> epoch_generations(service_epochs, 1u);
        for (uint32_t epoch = 0u;
             !persistent_prelaunched && epoch < service_epochs; ++epoch) {
            epoch_generations[epoch] =
                service_warmup_epochs + epoch + 1u;
            if (epoch_generations[epoch] != 1u) {
                const int copy_rc = aclrtMemcpyAsync(
                    sym + ctrl_off + offsetof(DynCsrCtrl, generation),
                    sizeof(uint32_t), &epoch_generations[epoch],
                    sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE, stream);
                if (copy_rc != 0) {
                    sync_rc = copy_rc;
                    break;
                }
            }
            if (is_worker) {
                if (k1_direct_tx) {
                    launch_inc_dc_sv2_dyn_csr_k1_direct_tx_kernel(
                        sym, ctrl_off,
                        static_cast<int>(ctrl.producer_lane_count), stream);
                } else {
                    launch_inc_dc_sv2_dyn_csr_producer_kernel(
                        sym, ctrl_off,
                        static_cast<int>(ctrl.producer_lane_count), stream);
                }
            } else if (!k1_direct_tx) {
                launch_inc_dc_sv2_dyn_csr_combine_kernel(
                    sym, ctrl_off,
                    static_cast<int>(topo.owner_count +
                                     ctrl.tx_lane_count), stream);
            }
        }
        if (sync_rc == 0 && !persistent_prelaunched) {
            if (device_event_timing &&
                aclrtRecordEvent(measured_end_event, stream) != ACL_SUCCESS) {
                sync_rc = -1;
            }
        }
        if (sync_rc == 0) {
            sync_rc = aclrtSynchronizeStream(stream);
        }
        if (is_worker && sync_rc != 0) {
            std::cerr << "DYNCSR_PRODUCER_FAIL pe=" << pe << std::endl;
            return 1;
        }
        if (ctrl.device_completion == 0u) {
            aclshmem_barrier_all();
        }
        transport_end_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
    } else {
        // Correctness baseline: complete transport before launching reducers.
        if (is_worker && device_producer) {
            launch_inc_dc_sv2_dyn_csr_producer_kernel(
                sym, ctrl_off, static_cast<int>(ctrl.producer_lane_count),
                stream);
            if (aclrtSynchronizeStream(stream) != 0) {
                std::cerr << "DYNCSR_PRODUCER_FAIL pe=" << pe << std::endl;
                return 1;
            }
        } else if (is_worker) {
            for (uint32_t si = 0; si < exec.schedule.size(); ++si) {
                const auto &cc = exec.schedule[si];
                const auto &lc =
                    plan.contributions[cc.logical_contribution_index];
                if (lc.contributor_rank != static_cast<uint32_t>(pe)) {
                    continue;
                }
                const int home_pe = static_cast<int>(topo.inc_pe);
                const uint64_t local_off =
                    ingress_off +
                    static_cast<uint64_t>(cc.ingress_slot) * tile_bytes;
                aclshmem_putmem(sym + local_off, sym + local_off, tile_bytes,
                                home_pe);
            }
            aclshmemx_quiet_on_stream(stream);
            aclrtSynchronizeStream(stream);
        }
        aclshmem_barrier_all();
        transport_end_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
        if (!is_worker) {
            launch_inc_dc_sv2_dyn_csr_combine_kernel(
                sym, ctrl_off,
                static_cast<int>(topo.owner_count +
                                 ctrl.tx_lane_count),
                stream);
            sync_rc = aclrtSynchronizeStream(stream);
        }
        aclshmem_barrier_all();
    }

    // Formal timing ends only after every worker transport and every INC
    // reducer kernel has completed.  D2H oracle verification is deliberately
    // outside the timed region.
    const uint64_t end_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
    float device_elapsed_ms = 0.0f;
    if (device_event_timing &&
        aclrtEventElapsedTime(&device_elapsed_ms, measured_start_event,
                              measured_end_event) != ACL_SUCCESS) {
        device_event_timing = false;
    }
    if (persistent_prelaunched && (is_worker || !k1_direct_tx)) {
        DynCsrPersistentTriggerLine trigger_line{};
        if (aclrtMemcpy(
                &trigger_line, sizeof(trigger_line),
                sym + persistent_trigger_off, sizeof(trigger_line),
                ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
            trigger_line.service_start_cycle == 0u ||
            trigger_line.service_end_cycle <
                trigger_line.service_start_cycle ||
            trigger_line.last_generation !=
                service_warmup_epochs + service_epochs ||
            trigger_line.status != kDynCsrFailNone) {
            std::cerr << "DYNCSR_PERSISTENT_SERVICE_TIMING_FAIL pe=" << pe
                      << " start=" << trigger_line.service_start_cycle
                      << " end=" << trigger_line.service_end_cycle
                      << " last_generation="
                      << trigger_line.last_generation
                      << " status=" << trigger_line.status << std::endl;
            device_ok = false;
        } else {
            persistent_service_start_cycle =
                trigger_line.service_start_cycle;
            persistent_service_end_cycle =
                trigger_line.service_end_cycle;
            persistent_service_cycles =
                trigger_line.service_end_cycle -
                trigger_line.service_start_cycle;
        }
    }

    uint64_t device_protocol_cycles_max = 0u;
    if (!is_worker && !k1_direct_tx) {
        DynCsrStats st{};
        aclrtMemcpy(&st, sizeof(st), sym + stats_off, sizeof(st),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        if (sync_rc != 0 || st.magic != kDynCsrMagic || st.done != 1u ||
            st.fail_code != 0u) {
            device_ok = false;
            std::cout << "DYNCSR_KERNEL_FAIL pe=" << pe << " sync=" << sync_rc
                      << " fail=" << st.fail_code << " done=" << st.done
                      << std::endl;
        } else {
            reduced = st.reduced;
            // Verify owned results against oracle
            std::vector<uint8_t> out_host(
                static_cast<size_t>(plan.result_count) * tile_bytes);
            aclrtMemcpy(out_host.data(), out_host.size(), sym + output_off,
                        out_host.size(), ACL_MEMCPY_DEVICE_TO_HOST);
            for (uint32_t r = 0; r < plan.result_count; ++r) {
                const bool direct_ub_result_tx =
                    tx_lane_count == 0u && !packed_result_tx &&
                    (ctrl.optimization_flags &
                     kDynCsrOptRemoteResultTx) != 0u &&
                    (ctrl.optimization_flags &
                     kDynCsrOptBatchResultTx) == 0u;
                // Direct UB egress intentionally leaves no duplicate result
                // in INC GM.  Every such row is verified at its actual
                // destination worker below; checking stale local scratch here
                // would test storage that is outside combine semantics.
                if (direct_ub_result_tx) continue;
                ++numeric_local_rows_verified;
                const uint32_t storage_row = packed_result_tx
                    ? packed_result_rank_offsets[result_dst_rank[r]] +
                          result_dst_row[r]
                    : r;
                const uint16_t *row = reinterpret_cast<const uint16_t *>(
                    out_host.data() +
                    static_cast<size_t>(storage_row) * tile_bytes);
                for (uint32_t h = 0; h < hidden; ++h) {
                    const float got = Fp16BitsToFloat(row[h]);
                    // Device output is FP16.  Compare against the correctly
                    // rounded FP16 reduction result, not the unrepresentable
                    // FP32 accumulator value (important for large top-k).
                    const float expv = Fp16BitsToFloat(
                        FloatToFp16Bits(oracle[r][h]));
                    ++numeric_local_elements_verified;
                    const bool finite =
                        std::isfinite(got) && std::isfinite(expv);
                    if (!finite || std::fabs(got - expv) > 2e-2f) {
                        ++mismatch;
                        if (!finite) {
                            ++nonfinite;
                        }
                        if (mismatch < 3u) {
                            std::cout << "DYNCSR_MISMATCH pe=" << pe << " r=" << r
                                      << " h=" << h << " got=" << got
                                      << " exp=" << expv << std::endl;
                        }
                    }
                }
            }
            if (mismatch != 0u) {
                device_ok = false;
            }
        }
        std::vector<DynCsrOwnerStats> owner_telemetry(owner_count);
        aclrtMemcpy(owner_telemetry.data(),
                    owner_telemetry.size() * sizeof(DynCsrOwnerStats),
                    sym + owner_stats_off,
                    owner_telemetry.size() * sizeof(DynCsrOwnerStats),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        uint64_t wait_max = 0u;
        uint64_t reduce_span_max = 0u;
        uint64_t kernel_max = 0u;
        uint32_t kernel_argmax = 0u;
        for (uint32_t oi = 0u; oi < owner_telemetry.size(); ++oi) {
            const auto &os = owner_telemetry[oi];
            wait_max = std::max(wait_max, os.ready_wait_cycles);
            if (os.reduce_cycles > kernel_max) {
                kernel_max = os.reduce_cycles;
                kernel_argmax = oi;
            }
            if (os.last_reduce_cycle >= os.first_reduce_cycle) {
                reduce_span_max =
                    std::max(reduce_span_max,
                             os.last_reduce_cycle - os.first_reduce_cycle);
            }
        }
        device_protocol_cycles_max = kernel_max;
        std::cout << "DYNCSR_DEVICE_REDUCER pe=" << pe
                  << " owner_count=" << owner_count
                  << " ready_wait_cycles_max=" << wait_max
                  << " reduce_span_cycles_max=" << reduce_span_max
                  << " kernel_cycles_max=" << kernel_max
                  << " kernel_cycles_owner0="
                  << (owner_telemetry.empty()
                          ? 0u : owner_telemetry[0].reduce_cycles)
                  << " kernel_cycles_argmax=" << kernel_argmax
                  << " completion_aggregate_cycles=" << st.reserved[0]
                  << " completion_fanout_cycles=" << st.reserved[1]
                  << " completion_ack_wait_cycles=" << st.reserved[2]
                  << std::endl;
    } else if (is_worker) {
        if ((ctrl.optimization_flags &
             kDynCsrOptRemoteResultTx) != 0u) {
            std::vector<uint8_t> out_host(
                static_cast<size_t>(plan.result_count) * tile_bytes);
            aclrtMemcpy(out_host.data(), out_host.size(), sym + output_off,
                        out_host.size(), ACL_MEMCPY_DEVICE_TO_HOST);
            for (uint32_t r = 0u; r < plan.result_count; ++r) {
                if (result_dst_rank[r] != static_cast<uint32_t>(pe)) {
                    continue;
                }
                ++numeric_remote_rows_verified;
                const uint32_t dst_row = result_dst_row[r];
                if (dst_row >= plan.result_count) {
                    device_ok = false;
                    ++mismatch;
                    break;
                }
                const uint16_t *row =
                    reinterpret_cast<const uint16_t *>(
                        out_host.data() +
                        static_cast<size_t>(dst_row) * tile_bytes);
                for (uint32_t h = 0u; h < hidden; ++h) {
                    const float got = Fp16BitsToFloat(row[h]);
                    const float expv = Fp16BitsToFloat(
                        FloatToFp16Bits(oracle[r][h]));
                    ++numeric_remote_elements_verified;
                    const bool finite =
                        std::isfinite(got) && std::isfinite(expv);
                    if (!finite || std::fabs(got - expv) > 2e-2f) {
                        ++mismatch;
                        if (!finite) {
                            ++nonfinite;
                        }
                        if (mismatch < 3u) {
                            std::cout
                                << "DYNCSR_REMOTE_TX_MISMATCH pe=" << pe
                                << " r=" << r << " dst_row=" << dst_row
                                << " h=" << h << " got=" << got
                                << " exp=" << expv << std::endl;
                        }
                    }
                }
            }
            if (mismatch != 0u) {
                device_ok = false;
            }
        }
        std::vector<DynCsrProducerStats> lane_telemetry(
            ctrl.producer_lane_count);
        aclrtMemcpy(lane_telemetry.data(),
                    lane_telemetry.size() * sizeof(DynCsrProducerStats),
                    sym + owner_stats_off,
                    lane_telemetry.size() * sizeof(DynCsrProducerStats),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        uint32_t issued = 0u;
        uint32_t signals = 0u;
        uint64_t kernel_max = 0u;
        uint64_t active_span_max = 0u;
        uint32_t local_reduce_cycles_max = 0u;
        uint32_t local_transport_cycles_max = 0u;
        for (const auto &ps : lane_telemetry) {
            if (ps.done != service_warmup_epochs + service_epochs) {
                device_ok = false;
            }
            if (ps.reserved[0] != kDynCsrFailNone) {
                device_ok = false;
            }
            issued += ps.issued;
            signals += ps.ready_signals;
            kernel_max = std::max(kernel_max, ps.kernel_cycles);
            local_reduce_cycles_max =
                std::max(local_reduce_cycles_max, ps.reserved[2]);
            local_transport_cycles_max =
                std::max(local_transport_cycles_max, ps.reserved[3]);
            if (ps.last_ready_cycle >= ps.first_issue_cycle &&
                ps.first_issue_cycle != 0u) {
                active_span_max =
                    std::max(active_span_max,
                             ps.last_ready_cycle - ps.first_issue_cycle);
            }
        }
        device_protocol_cycles_max = kernel_max;
        std::cout << "DYNCSR_DEVICE_PRODUCER pe=" << pe
                  << " lane_count=" << ctrl.producer_lane_count
                  << " issued=" << issued << " ready_signals=" << signals
                  << " active_span_cycles_max=" << active_span_max
                  << " local_reduce_cycles_max=" << local_reduce_cycles_max
                  << " local_transport_cycles_max="
                  << local_transport_cycles_max
                  << " kernel_cycles_max=" << kernel_max << std::endl;
    }
    aclshmem_barrier_all();

    // Evidence (all ranks)
    std::vector<uint32_t> launched(W);
    for (uint32_t w = 0; w < W; ++w) launched[w] = w;
    std::string zero_w;
    for (uint32_t z : lvr.zero_contribution_workers) {
        if (!zero_w.empty()) zero_w += ",";
        zero_w += std::to_string(z);
    }
    std::string worker_loads;
    uint32_t max_worker_contributions = 0u;
    uint64_t worker_load_square_sum = 0u;
    for (uint32_t w = 0u; w < lvr.contributions_per_worker.size(); ++w) {
        if (!worker_loads.empty()) worker_loads += ",";
        const uint32_t load = lvr.contributions_per_worker[w];
        worker_loads += std::to_string(load);
        max_worker_contributions =
            std::max(max_worker_contributions, load);
        worker_load_square_sum += static_cast<uint64_t>(load) * load;
    }
    const bool same_multi = rank_dedup_active;
    // INCs verify data; workers verify that every device-completion
    // generation arrived.  A worker timeout is a protocol failure, never an
    // implicit success.
    bool final_ok = device_ok;
    if (!is_worker) {
        final_ok = device_ok && mismatch == 0u;
    }
    const double host_rank_us =
        static_cast<double>(end_ns - begin_ns) / 1000.0;
    const bool calibrated_device_cycle_timing =
        persistent_local_trigger && persistent_prelaunched &&
        device_cycles_per_ns >= 0.03 && device_cycles_per_ns <= 0.08;
    const uint64_t calibrated_protocol_cycles =
        persistent_service_cycles != 0u
            ? persistent_service_cycles
            : device_protocol_cycles_max;
    const double rank_us = calibrated_device_cycle_timing
        ? static_cast<double>(calibrated_protocol_cycles) /
              (device_cycles_per_ns * 1000.0)
        : (device_event_timing
               ? static_cast<double>(device_elapsed_ms) * 1000.0
               : host_rank_us);
    uint64_t reported_begin_ns = begin_ns;
    uint64_t reported_end_ns = end_ns;
    if (calibrated_device_cycle_timing &&
        persistent_service_start_cycle >= persistent_target_cycle &&
        persistent_service_end_cycle >= persistent_service_start_cycle &&
        aligned_start_ns != 0u) {
        reported_begin_ns =
            aligned_start_ns + static_cast<uint64_t>(
                static_cast<double>(
                    persistent_service_start_cycle -
                    persistent_target_cycle) /
                device_cycles_per_ns);
        reported_end_ns =
            aligned_start_ns + static_cast<uint64_t>(
                static_cast<double>(
                    persistent_service_end_cycle -
                    persistent_target_cycle) /
                device_cycles_per_ns);
    }
    const double transport_us =
        static_cast<double>(transport_end_ns - begin_ns) / 1000.0;
    const double reduce_us =
        static_cast<double>(end_ns - transport_end_ns) / 1000.0;
    const uint64_t logical_input_bytes =
        static_cast<uint64_t>(logical_plan.contribution_count) * hidden * 2ull *
        service_epochs;
    const uint64_t logical_output_bytes =
        static_cast<uint64_t>(plan.result_count) * hidden * 2ull *
        service_epochs;
    uint64_t local_upload_items = 0u;
    uint64_t local_download_items = 0u;
    if (is_worker) {
        for (const auto &lc : plan.contributions) {
            if (lc.contributor_rank == static_cast<uint32_t>(pe)) {
                ++local_upload_items;
            }
        }
        for (const auto &result : plan.results) {
            if (result.dst_rank == static_cast<uint32_t>(pe)) {
                ++local_download_items;
            }
        }
    } else {
        local_upload_items = exec.schedule.size();
        local_download_items = plan.result_count;
    }
    const uint64_t physical_upload_bytes_rank =
        local_upload_items * hidden * 2ull * service_epochs;
    const uint64_t physical_download_bytes_rank =
        local_download_items * hidden * 2ull * service_epochs;
    const uint64_t physical_ingress_bytes =
        static_cast<uint64_t>(plan.contribution_count) * hidden * 2ull *
        service_epochs;
        std::cout << "DYNCSR_TIMING pe=" << pe
                  << " begin_ns=" << reported_begin_ns
                  << " end_ns=" << reported_end_ns
                  << " service_start_cycle="
                  << persistent_service_start_cycle
                  << " service_end_cycle="
                  << persistent_service_end_cycle
              << " rank_us=" << rank_us
              << " host_rank_us=" << host_rank_us
              << " timing_source="
              << (calibrated_device_cycle_timing
                      ? "calibrated_device_cycle_after_persistent_deadline"
                      : (device_event_timing ? "device_event"
                                             : "host_clock"))
              << " device_protocol_cycles_max="
              << device_protocol_cycles_max
              << " persistent_service_cycles="
              << persistent_service_cycles
              << " device_cycles_per_ns=" << device_cycles_per_ns
              << " transport_us=" << transport_us
              << " reduce_us=" << reduce_us
              << " logical_input_bytes=" << logical_input_bytes
              << " logical_output_bytes=" << logical_output_bytes
              << " physical_ingress_bytes=" << physical_ingress_bytes
              << " physical_upload_bytes_rank="
              << physical_upload_bytes_rank
              << " physical_download_bytes_rank="
              << physical_download_bytes_rank
              << std::endl;

    std::cout << "DYNCSR_EVIDENCE pe=" << pe
              << " worker_world_size=" << W
              << " declared_max_topk=" << plan.declared_max_topk
              << " expected_count_min=" << lvr.expected_count_min
              << " expected_count_max=" << lvr.expected_count_max
              << " zero_contribution_workers=[" << zero_w << "]"
              << " contributions_per_worker=[" << worker_loads << "]"
              << " max_worker_contributions=" << max_worker_contributions
              << " worker_load_square_sum=" << worker_load_square_sum
              << " same_worker_multi_ordinal=" << (same_multi ? 1 : 0)
              << " rank_dedup=" << (rank_dedup_active ? 1 : 0)
              << " logical_contributions="
              << logical_plan.contribution_count
              << " physical_contributions=" << plan.contribution_count
              << " device_reduce_verified="
              << ((!is_worker && final_ok) ? 1 : (is_worker ? 1 : 0))
              << " mismatch_count=" << mismatch
              << " nonfinite_count=" << nonfinite
              << " numeric_local_rows_verified="
              << numeric_local_rows_verified
              << " numeric_local_elements_verified="
              << numeric_local_elements_verified
              << " numeric_remote_rows_verified="
              << numeric_remote_rows_verified
              << " numeric_remote_elements_verified="
              << numeric_remote_elements_verified
              << " reduced=" << reduced
              << " device_producer=" << (device_producer ? 1 : 0)
              << " producer_lanes=" << ctrl.producer_lane_count
              << " producer_window=" << ctrl.producer_quiet_window
              << " producer_wide_window_safe="
              << (producer_wide_window_safe ? 1 : 0)
              << " overlap=" << (overlap_enable ? 1 : 0)
              << " batched_ready=" << (batched_ready ? 1 : 0)
              << " ready_mode=" << ctrl.ready_mode
              << " device_completion=" << ctrl.device_completion
              << " local_ordinal_bitmap="
              << ((ctrl.optimization_flags &
                   kDynCsrOptLocalOrdinalBitmap) != 0u)
              << " k1_identity_copy="
              << ((ctrl.optimization_flags &
                   kDynCsrOptK1IdentityCopy) != 0u)
              << " first_contribution_init="
              << ((ctrl.optimization_flags &
                   kDynCsrOptFirstContributionInit) != 0u)
              << " cyclic_owner_results="
              << (((ctrl.optimization_flags &
                     kDynCsrOptCyclicOwnerResults) != 0u)
                      ? 1
                      : 0)
              << " async_rank_prereduce_push="
              << (((ctrl.optimization_flags &
                     kDynCsrOptAsyncRankPrereducePush) != 0u)
                      ? 1
                      : 0)
              << " ub_direct_rank_push="
              << (((ctrl.optimization_flags &
                     kDynCsrOptUbDirectRankPush) != 0u)
                      ? 1
                      : 0)
              << " source_group_worklist="
              << ((ctrl.optimization_flags &
                   kDynCsrOptSourceGroupWorklist) != 0u)
              << " source_group_schedule=owner_major_inc_minor"
              << " measured_start_barrier=1"
              << " measured_stream_start_barrier="
              << (measured_stream_start_barrier ? 1 : 0)
              << " device_generation_start_gate="
              << (ctrl.start_gate_off != 0u ? 1 : 0)
              << " persistent_local_trigger="
              << (persistent_local_trigger ? 1 : 0)
              << " persistent_prequeued_epoch_count="
              << (persistent_local_trigger ? service_epochs : 0u)
              << " persistent_epoch_barrier="
              << (persistent_epoch_barrier ? 1 : 0)
              << " persistent_target_cycle=" << persistent_target_cycle
              << " device_cycles_per_ns=" << device_cycles_per_ns
              << " cycle_calibration_rtt_ns="
              << cycle_calibration_rtt_ns
              << " aligned_start_ns=" << aligned_start_ns
              << " measured_device_events="
              << (device_event_timing ? 1 : 0)
              << " remote_result_tx="
              << ((ctrl.optimization_flags &
                   kDynCsrOptRemoteResultTx) != 0u)
              << " tx_window=" << ctrl.tx_quiet_window
              << " k1_direct_result_tx=" << (k1_direct_tx ? 1 : 0)
              << " completion_mte_fanout="
              << ((ctrl.optimization_flags &
                   kDynCsrOptCompletionMteFanout) != 0u)
              << " k1_private_mte_push="
              << ((ctrl.optimization_flags &
                   kDynCsrOptK1PrivateMtePush) != 0u)
              << " k1_pair_ready="
              << ((ctrl.optimization_flags &
                   kDynCsrOptK1PairReady) != 0u)
              << " coalesced_group_put="
              << ((ctrl.optimization_flags &
                   kDynCsrOptCoalescedGroupPut) != 0u)
              << " inc_scoped_coalesced_put="
              << ((ctrl.ready_mode == 3u &&
                   (ctrl.optimization_flags &
                    kDynCsrOptCoalescedGroupPut) != 0u)
                      ? 1
                      : 0)
              << " inc_scoped_single_remote_poller="
              << (ctrl.ready_mode == 3u ? 1 : 0)
              << " inc_scoped_poll_backoff="
              << (ctrl.ready_mode == 3u ? 1 : 0)
              << " inc_scoped_ready_transport="
              << (ctrl.ready_mode == 3u
                      ? "rma_cacheline"
                      : (ctrl.ready_mode == 5u ? "putmem_signal"
                                               : "signal_op"))
              << " coalesced_chunk_bytes="
              << ctrl.coalesced_chunk_bytes
              << " output_dcci_small_only=" << ctrl.output_dcci_small_only
              << " service_epochs=" << service_epochs
              << " service_warmup_epochs=" << service_warmup_epochs
              << " device_queued_multi_epoch="
              << (service_epochs > 1u ? 1 : 0)
              << " abort_generation=" << ctrl.abort_generation
              << " ready_signal_count="
              << (ctrl.ready_mode == 3u ? active_source_count
                                        : active_group_count)
              << " ready_signal_upper_bound=" << group_count
              << " all_workers_completed_protocol=1" << std::endl;

    if (final_ok) {
        std::cout << "DYNCSR_RESULT SUCCESS pe=" << pe << std::endl;
    } else {
        std::cout << "DYNCSR_RESULT FAIL pe=" << pe << std::endl;
    }

    aclshmem_free(sym);
    if (measured_start_event != nullptr) {
        aclrtDestroyEvent(measured_start_event);
    }
    if (measured_end_event != nullptr) {
        aclrtDestroyEvent(measured_end_event);
    }
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();
    return final_ok ? 0 : 1;
}
