#ifndef INC_DC_RESOURCE_POLICY_H
#define INC_DC_RESOURCE_POLICY_H

#include <algorithm>
#include <cstdint>

#include "inc_dc_platform_capabilities.h"

constexpr uint32_t kIncDcResourcePolicyVersion = 1u;

// Hardware/service policy only: no W/K/shape lookup table is permitted here.
// Version 1 preserves the qualified 1:2 Dispatch/Combine split exactly.
struct IncDcResourcePolicyConfig {
    uint32_t version = kIncDcResourcePolicyVersion;
    uint32_t dispatch_weight = 1u;
    uint32_t combine_weight = 2u;
    uint32_t reserved_inc_aiv = 0u;
};

/* Hardware/topology-only policy.  Workload fields are intentionally absent. */
struct IncDcAivPolicy {
    uint32_t policy_version = 0u;
    uint32_t live_aiv = 0u;
    uint32_t worker_world_size = 0u;
    uint32_t worker_half_limit = 0u;
    uint32_t dispatch_inc_aiv = 0u;
    uint32_t combine_inc_aiv = 0u;
    uint32_t dispatch_worker_aiv = 0u;
    uint32_t combine_worker_aiv = 0u;
    uint32_t reserved_inc_aiv = 0u;
    uint32_t dispatch_weight = 0u;
    uint32_t combine_weight = 0u;
    uint32_t dispatch_inc_capped = 0u;
    uint32_t combine_inc_capped = 0u;
};

inline bool IncDcResolveAivPolicyWithConfig(
    uint32_t live_aiv, uint32_t worker_world_size,
    uint32_t dispatch_inc_limit, uint32_t combine_inc_limit,
    const IncDcResourcePolicyConfig &config, IncDcAivPolicy *policy)
{
    if (policy == nullptr || live_aiv == 0u || worker_world_size == 0u ||
        dispatch_inc_limit == 0u || combine_inc_limit == 0u ||
        config.version != kIncDcResourcePolicyVersion ||
        config.dispatch_weight == 0u || config.combine_weight == 0u ||
        config.reserved_inc_aiv >= live_aiv) {
        return false;
    }
    const uint32_t usable_inc_aiv = live_aiv - config.reserved_inc_aiv;
    if (usable_inc_aiv < 2u) return false;
    const uint64_t total_weight =
        static_cast<uint64_t>(config.dispatch_weight) +
        config.combine_weight;
    IncDcAivPolicy out{};
    out.policy_version = config.version;
    out.live_aiv = live_aiv;
    out.worker_world_size = worker_world_size;
    out.reserved_inc_aiv = config.reserved_inc_aiv;
    out.dispatch_weight = config.dispatch_weight;
    out.combine_weight = config.combine_weight;
    out.worker_half_limit = std::max(1u, live_aiv / 2u);
    const uint32_t desired_dispatch = std::max(
        1u, static_cast<uint32_t>(
                static_cast<uint64_t>(usable_inc_aiv) *
                config.dispatch_weight / total_weight));
    out.dispatch_inc_aiv = std::min(desired_dispatch, dispatch_inc_limit);
    out.dispatch_inc_capped = desired_dispatch > dispatch_inc_limit ? 1u : 0u;
    const uint32_t combine_available =
        usable_inc_aiv > out.dispatch_inc_aiv
            ? usable_inc_aiv - out.dispatch_inc_aiv
            : 0u;
    if (combine_available == 0u) return false;
    out.combine_inc_aiv =
        std::max(1u, std::min(combine_available, combine_inc_limit));
    out.combine_inc_capped =
        combine_available > combine_inc_limit ? 1u : 0u;
    out.dispatch_worker_aiv = std::min(
        out.worker_half_limit,
        std::max(1u, (out.dispatch_inc_aiv + worker_world_size - 1u) /
                         worker_world_size));
    const uint32_t topology_share =
        (live_aiv + worker_world_size - 1u) / worker_world_size +
        std::max(1u, live_aiv / 12u);
    out.combine_worker_aiv = std::min(
        out.worker_half_limit,
        std::max(std::max(1u, live_aiv / 4u), topology_share));
    if (out.dispatch_inc_aiv + out.combine_inc_aiv +
                out.reserved_inc_aiv > live_aiv ||
        out.dispatch_worker_aiv > out.worker_half_limit ||
        out.combine_worker_aiv > out.worker_half_limit) {
        return false;
    }
    *policy = out;
    return true;
}

inline bool IncDcResolveAivPolicy(uint32_t live_aiv,
                                 uint32_t worker_world_size,
                                 uint32_t dispatch_inc_limit,
                                 uint32_t combine_inc_limit,
                                 IncDcAivPolicy *policy)
{
    return IncDcResolveAivPolicyWithConfig(
        live_aiv, worker_world_size, dispatch_inc_limit, combine_inc_limit,
        IncDcResourcePolicyConfig{}, policy);
}

inline uint64_t IncDcResourcePolicyFingerprint(const IncDcAivPolicy &policy)
{
    uint64_t hash = 1469598103934665603ull;
    const uint32_t words[] = {
        inc::dc::kIncDcPlatformCapabilityVersion,
        inc::dc::kIncDcAivUbBudgetBytes,
        inc::dc::kIncDcPrivateMtePacketBytes,
        inc::dc::kIncDcMaxDispatchLanes,
        inc::dc::kIncDcMaxCombineOwners,
        policy.policy_version,
        policy.live_aiv,
        policy.worker_world_size,
        policy.worker_half_limit,
        policy.dispatch_inc_aiv,
        policy.combine_inc_aiv,
        policy.dispatch_worker_aiv,
        policy.combine_worker_aiv,
        policy.reserved_inc_aiv,
        policy.dispatch_weight,
        policy.combine_weight,
        policy.dispatch_inc_capped,
        policy.combine_inc_capped,
    };
    for (uint32_t word : words) {
        hash ^= word;
        hash *= 1099511628211ull;
    }
    return hash;
}

#endif
