#include "inc_dc_combine_topology.h"

#include <algorithm>
#include <unordered_set>

namespace inc {
namespace dc {
namespace {

uint64_t Mix(uint64_t h, uint64_t v)
{
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}

void AssignSequentialChannels(IncDcTopologyDescriptor *topo)
{
    topo->worker_inc_channels.clear();
    topo->worker_inc_channels.reserve(topo->worker_inc_indices.size());
    for (uint32_t ch = 0; ch < topo->worker_inc_indices.size(); ++ch) {
        topo->worker_inc_channels.push_back(ch);
    }
}

} // namespace

uint64_t ComputeTopologyDigest(const IncDcTopologyDescriptor &topo)
{
    uint64_t h = 0x544f504f44595302ull; // TOPODYS\x02 (channel map included)
    h = Mix(h, topo.worker_count);
    h = Mix(h, topo.inc_count);
    h = Mix(h, topo.owner_count_per_inc);
    h = Mix(h, topo.topology_generation);
    for (uint32_t v : topo.worker_pe_ids) h = Mix(h, v);
    for (uint32_t v : topo.inc_pe_ids) h = Mix(h, v);
    for (uint32_t v : topo.worker_inc_offsets) h = Mix(h, v);
    for (uint32_t v : topo.worker_inc_indices) h = Mix(h, v);
    for (uint32_t v : topo.worker_inc_channels) h = Mix(h, v);
    return h == 0 ? 1 : h;
}

bool LookupIngressChannel(const IncDcTopologyDescriptor &topo, uint32_t worker,
                          uint32_t inc_index, uint32_t *channel_out)
{
    if (channel_out == nullptr || worker >= topo.worker_count) {
        return false;
    }
    if (topo.worker_inc_offsets.size() != topo.worker_count + 1u ||
        topo.worker_inc_channels.size() != topo.worker_inc_indices.size()) {
        return false;
    }
    const uint32_t b = topo.worker_inc_offsets[worker];
    const uint32_t e = topo.worker_inc_offsets[worker + 1];
    if (e < b || e > topo.worker_inc_indices.size()) {
        return false;
    }
    for (uint32_t i = b; i < e; ++i) {
        if (topo.worker_inc_indices[i] == inc_index) {
            *channel_out = topo.worker_inc_channels[i];
            return true;
        }
    }
    return false;
}

IncDcStatus BuildSingleIncTopology(uint32_t worker_count,
                                   uint32_t owner_count,
                                   uint32_t worker_pe_base,
                                   uint32_t inc_pe,
                                   uint64_t generation,
                                   IncDcTopologyDescriptor *out)
{
    if (out == nullptr || worker_count == 0u || owner_count == 0u) {
        return IncDcStatus::INVALID_ARGUMENT;
    }
    *out = IncDcTopologyDescriptor{};
    out->worker_count = worker_count;
    out->inc_count = 1u;
    out->owner_count_per_inc = owner_count;
    out->topology_generation = generation;
    out->worker_pe_ids.resize(worker_count);
    for (uint32_t worker = 0u; worker < worker_count; ++worker) {
        out->worker_pe_ids[worker] = worker_pe_base + worker;
    }
    out->inc_pe_ids = {inc_pe};
    out->worker_inc_offsets.resize(worker_count + 1u);
    out->worker_inc_indices.assign(worker_count, 0u);
    for (uint32_t worker = 0u; worker <= worker_count; ++worker) {
        out->worker_inc_offsets[worker] = worker;
    }
    AssignSequentialChannels(out);
    out->topology_digest = ComputeTopologyDigest(*out);
    return IncDcStatus::OK;
}

IncDcStatus ValidateTopologyDescriptor(const IncDcTopologyDescriptor &topo,
                                       IncDcTopologyValidateReport *report)
{
    IncDcTopologyValidateReport local{};
    auto *rep = report ? report : &local;
    rep->ok = false;
    auto fail = [&](const char *msg) -> IncDcStatus {
        rep->first_error = msg;
        return IncDcStatus::INVALID_ARGUMENT;
    };
    if (topo.worker_count == 0) return fail("empty_worker_count");
    if (topo.inc_count != 1u) return fail("single_inc_required");
    if (topo.owner_count_per_inc == 0) return fail("owner_count_zero");
    if (topo.worker_pe_ids.size() != topo.worker_count) return fail("worker_pe_size");
    if (topo.inc_pe_ids.size() != topo.inc_count) return fail("inc_pe_size");
    if (topo.worker_inc_offsets.size() != topo.worker_count + 1u) {
        return fail("worker_inc_offsets_size");
    }
    if (topo.worker_inc_channels.size() != topo.worker_inc_indices.size()) {
        return fail("channel_map_size_mismatch");
    }
    if (topo.topology_digest != ComputeTopologyDigest(topo)) {
        return fail("topology_digest_mismatch");
    }
    std::unordered_set<uint32_t> pes;
    for (uint32_t pe : topo.worker_pe_ids) {
        if (!pes.insert(pe).second) return fail("duplicate_worker_pe");
    }
    for (uint32_t pe : topo.inc_pe_ids) {
        if (!pes.insert(pe).second) return fail("worker_inc_pe_overlap_or_dup");
    }
    std::unordered_set<uint32_t> channels;
    for (uint32_t w = 0; w < topo.worker_count; ++w) {
        const uint32_t b = topo.worker_inc_offsets[w];
        const uint32_t e = topo.worker_inc_offsets[w + 1];
        if (e < b || e > topo.worker_inc_indices.size()) return fail("csr_oob");
        if (b == e) return fail("worker_has_no_reachable_inc");
        std::unordered_set<uint32_t> seen_inc;
        for (uint32_t i = b; i < e; ++i) {
            if (topo.worker_inc_indices[i] >= topo.inc_count) {
                return fail("inc_index_oob");
            }
            if (!seen_inc.insert(topo.worker_inc_indices[i]).second) {
                return fail("duplicate_reachable_inc");
            }
            if (!channels.insert(topo.worker_inc_channels[i]).second) {
                return fail("duplicate_ingress_channel");
            }
        }
    }
    rep->ok = true;
    return IncDcStatus::OK;
}

} // namespace dc
} // namespace inc
