#include "inc_dc_combine_topology.h"

#include <unordered_set>

namespace inc {
namespace dc {
namespace {

uint64_t Mix(uint64_t h, uint64_t v)
{
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}

} // namespace

uint64_t ComputeTopologyDigest(const IncDcTopologyDescriptor &topo)
{
    uint64_t h = 0x53494e474c455403ull; // SINGLET\x03
    h = Mix(h, topo.worker_count);
    h = Mix(h, topo.owner_count);
    h = Mix(h, topo.inc_pe);
    h = Mix(h, topo.topology_generation);
    for (uint32_t v : topo.worker_pe_ids) h = Mix(h, v);
    for (uint32_t v : topo.worker_ingress_channels) h = Mix(h, v);
    return h == 0 ? 1 : h;
}

bool LookupIngressChannel(const IncDcTopologyDescriptor &topo, uint32_t worker,
                          uint32_t *channel_out)
{
    if (channel_out == nullptr || worker >= topo.worker_count ||
        topo.worker_ingress_channels.size() != topo.worker_count) {
        return false;
    }
    *channel_out = topo.worker_ingress_channels[worker];
    return true;
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
    out->owner_count = owner_count;
    out->inc_pe = inc_pe;
    out->topology_generation = generation;
    out->worker_pe_ids.resize(worker_count);
    for (uint32_t worker = 0u; worker < worker_count; ++worker) {
        out->worker_pe_ids[worker] = worker_pe_base + worker;
    }
    out->worker_ingress_channels.resize(worker_count);
    for (uint32_t worker = 0u; worker < worker_count; ++worker) {
        out->worker_ingress_channels[worker] = worker;
    }
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
    if (topo.owner_count == 0) return fail("owner_count_zero");
    if (topo.worker_pe_ids.size() != topo.worker_count) return fail("worker_pe_size");
    if (topo.worker_ingress_channels.size() != topo.worker_count) {
        return fail("channel_map_size_mismatch");
    }
    if (topo.topology_digest != ComputeTopologyDigest(topo)) {
        return fail("topology_digest_mismatch");
    }
    std::unordered_set<uint32_t> pes;
    for (uint32_t pe : topo.worker_pe_ids) {
        if (!pes.insert(pe).second) return fail("duplicate_worker_pe");
    }
    if (!pes.insert(topo.inc_pe).second) return fail("worker_inc_pe_overlap");
    std::unordered_set<uint32_t> channels;
    for (uint32_t w = 0; w < topo.worker_count; ++w) {
        if (!channels.insert(topo.worker_ingress_channels[w]).second) {
            return fail("duplicate_ingress_channel");
        }
    }
    rep->ok = true;
    return IncDcStatus::OK;
}

} // namespace dc
} // namespace inc
