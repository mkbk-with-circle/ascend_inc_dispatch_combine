#ifndef INC_DC_COMBINE_PLAN_WIRE_H
#define INC_DC_COMBINE_PLAN_WIRE_H

#include <cstdint>
#include <string>
#include <vector>

#include "inc_dc_combine_logical_plan.h"
#include "inc_dc_types.h"

namespace inc {
namespace dc {

constexpr uint32_t kIncDcLogicalPlanWireMagicV2 = 0x324C5044u; // 'DPL2'
constexpr uint16_t kIncDcLogicalPlanWireAbiMajor = 2u;
constexpr uint16_t kIncDcLogicalPlanWireAbiMinor = 0u;

#pragma pack(push, 1)
struct IncDcLogicalPlanWireHeaderV2 {
    uint32_t magic = kIncDcLogicalPlanWireMagicV2;
    uint16_t abi_major = kIncDcLogicalPlanWireAbiMajor;
    uint16_t abi_minor = kIncDcLogicalPlanWireAbiMinor;
    uint64_t total_bytes = 0;
    uint32_t worker_world_size = 0;
    uint32_t result_count = 0;
    uint32_t contribution_count = 0;
    uint32_t declared_max_topk = 0;
    uint32_t uniform_topk = 0;
    uint32_t flags = 0; // bit0 = uniform_topk_valid
    uint64_t results_offset = 0;
    uint64_t contributions_offset = 0;
    uint64_t semantic_digest = 0;
    uint64_t integrity_digest = 0; // covers full canonical blob excl. this field
};

struct IncDcLogicalResultWireV2 {
    uint32_t dst_rank = 0;
    uint32_t dst_local_row = 0;
    uint32_t contribution_begin = 0;
    uint32_t contribution_count = 0;
};

struct IncDcLogicalContributionWireV2 {
    uint64_t contribution_uid = 0;
    uint32_t result_id = 0;
    uint32_t ordinal = 0;
    uint32_t contributor_rank = 0;
    uint32_t contributor_local_row = 0;
    uint32_t weight_bits = 0;
};
#pragma pack(pop)

struct IncDcLogicalPlanWireReport {
    bool ok = false;
    std::string first_error;
};

IncDcStatus SerializeLogicalPlanV2(const IncDcCombineLogicalPlanV2 &plan,
                                   std::vector<uint8_t> *blob,
                                   IncDcLogicalPlanWireReport *report);

IncDcStatus ValidateLogicalPlanWireV2(const void *blob, uint64_t bytes,
                                      IncDcLogicalPlanWireReport *report);

IncDcStatus ParseLogicalPlanWireV2(const void *blob, uint64_t bytes,
                                   IncDcCombineLogicalPlanV2 *out,
                                   IncDcLogicalPlanWireReport *report);

uint64_t ComputeLogicalPlanWireIntegrityDigest(const uint8_t *blob,
                                               uint64_t bytes);

} // namespace dc
} // namespace inc

#endif
