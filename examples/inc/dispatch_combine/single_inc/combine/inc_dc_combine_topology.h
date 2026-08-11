#ifndef INC_DC_COMBINE_TOPOLOGY_H
#define INC_DC_COMBINE_TOPOLOGY_H

#include <cstdint>
#include <string>
#include <vector>

#include "inc_dc_types.h"

namespace inc {
namespace dc {

struct IncDcTopologyDescriptor {
    uint32_t worker_count = 0;
    uint32_t inc_count = 0;
    uint32_t owner_count_per_inc = 0;
    std::vector<uint32_t> worker_pe_ids;
    std::vector<uint32_t> inc_pe_ids;
    // CSR: worker -> reachable INC indices
    std::vector<uint32_t> worker_inc_offsets; // size worker_count+1
    std::vector<uint32_t> worker_inc_indices;
    // Explicit edge resource map: parallel to worker_inc_indices.
    // ingress_channel for (worker, inc) = worker_inc_channels[edge_pos].
    std::vector<uint32_t> worker_inc_channels;
    uint64_t topology_generation = 0;
    uint64_t topology_digest = 0;
};

struct IncDcTopologyValidateReport {
    bool ok = false;
    std::string first_error;
};

// Build the only supported execution topology: every worker reaches one INC.
// The descriptor remains explicit so a caller can map logical worker/INC PEs
// to any physical devices without changing the plan compiler.
IncDcStatus BuildSingleIncTopology(uint32_t worker_count,
                                   uint32_t owner_count,
                                   uint32_t worker_pe_base,
                                   uint32_t inc_pe,
                                   uint64_t generation,
                                   IncDcTopologyDescriptor *out);

// Lookup explicit channel for (worker, inc). Returns false if no edge.
bool LookupIngressChannel(const IncDcTopologyDescriptor &topo, uint32_t worker,
                          uint32_t inc_index, uint32_t *channel_out);

uint64_t ComputeTopologyDigest(const IncDcTopologyDescriptor &topo);

IncDcStatus ValidateTopologyDescriptor(const IncDcTopologyDescriptor &topo,
                                       IncDcTopologyValidateReport *report);

} // namespace dc
} // namespace inc

#endif
