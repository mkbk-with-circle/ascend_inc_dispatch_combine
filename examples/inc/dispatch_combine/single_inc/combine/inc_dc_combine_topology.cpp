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

IncDcStatus BuildAllToAllIncReachability(IncDcTopologyDescriptor *topo)
{
    if (topo == nullptr || topo->worker_count == 0 || topo->inc_count == 0) {
        return IncDcStatus::INVALID_ARGUMENT;
    }
    topo->worker_inc_offsets.assign(topo->worker_count + 1u, 0);
    topo->worker_inc_indices.clear();
    topo->worker_inc_indices.reserve(
        static_cast<size_t>(topo->worker_count) * topo->inc_count);
    for (uint32_t w = 0; w < topo->worker_count; ++w) {
        topo->worker_inc_offsets[w] =
            static_cast<uint32_t>(topo->worker_inc_indices.size());
        for (uint32_t i = 0; i < topo->inc_count; ++i) {
            topo->worker_inc_indices.push_back(i);
        }
    }
    topo->worker_inc_offsets[topo->worker_count] =
        static_cast<uint32_t>(topo->worker_inc_indices.size());
    AssignSequentialChannels(topo);
    topo->topology_digest = ComputeTopologyDigest(*topo);
    return IncDcStatus::OK;
}

IncDcStatus BuildExplicitPairedTopology(uint32_t worker_count, uint32_t inc_count,
                                        uint32_t owner_count_per_inc,
                                        uint32_t worker_pe_base,
                                        uint32_t inc_pe_base,
                                        uint64_t generation,
                                        IncDcTopologyDescriptor *out)
{
    if (out == nullptr || worker_count == 0 || inc_count == 0 ||
        owner_count_per_inc == 0) {
        return IncDcStatus::INVALID_ARGUMENT;
    }
    *out = IncDcTopologyDescriptor{};
    out->worker_count = worker_count;
    out->inc_count = inc_count;
    out->owner_count_per_inc = owner_count_per_inc;
    out->topology_generation = generation;
    out->worker_pe_ids.resize(worker_count);
    out->inc_pe_ids.resize(inc_count);
    for (uint32_t w = 0; w < worker_count; ++w) {
        out->worker_pe_ids[w] = worker_pe_base + w;
    }
    for (uint32_t i = 0; i < inc_count; ++i) {
        out->inc_pe_ids[i] = inc_pe_base + i;
    }
    out->worker_inc_offsets.assign(worker_count + 1u, 0);
    out->worker_inc_indices.clear();
    for (uint32_t w = 0; w < worker_count; ++w) {
        out->worker_inc_offsets[w] =
            static_cast<uint32_t>(out->worker_inc_indices.size());
        out->worker_inc_indices.push_back(w % inc_count);
    }
    out->worker_inc_offsets[worker_count] =
        static_cast<uint32_t>(out->worker_inc_indices.size());
    AssignSequentialChannels(out);
    out->topology_digest = ComputeTopologyDigest(*out);
    return IncDcStatus::OK;
}

IncDcStatus BuildExplicitAllToAllTopology(uint32_t worker_count,
                                          uint32_t inc_count,
                                          uint32_t owner_count_per_inc,
                                          uint32_t worker_pe_base,
                                          uint32_t inc_pe_base,
                                          uint64_t generation,
                                          IncDcTopologyDescriptor *out)
{
    if (BuildExplicitPairedTopology(worker_count, inc_count, owner_count_per_inc,
                                    worker_pe_base, inc_pe_base, generation,
                                    out) != IncDcStatus::OK) {
        return IncDcStatus::INVALID_ARGUMENT;
    }
    return BuildAllToAllIncReachability(out);
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
    if (topo.worker_count == 0 || topo.inc_count == 0) return fail("empty_counts");
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
