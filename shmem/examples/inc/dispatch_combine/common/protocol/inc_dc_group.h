#ifndef INC_DC_GROUP_H
#define INC_DC_GROUP_H

#include <cstdint>

#include "inc_dc_types.h"

namespace inc {
namespace dc {

struct IncDcGroupConfig {
    uint32_t abi_version = kAbiVersion;
    uint32_t group_size = 0; // runtime 2..16
    int32_t local_rank = -1;
    int32_t switch_rank = -1; // SHMEM PE of the single INC
    int32_t inc_logical_pe = -1;
    const int32_t *worker_pes = nullptr; // length=group_size
    uint32_t num_global_experts = 0;
    const int32_t *expert_to_rank = nullptr; // length=num_global_experts
    uint32_t max_topk = 0;
    uint32_t flags = 0;
    uint32_t group_id = 0;
    uint32_t topology = 0; // IncDcTopology as uint32_t
};

// 校验 group 配置；禁止写死 N=8 或 rank 8
IncDcStatus ValidateGroupConfig(const IncDcGroupConfig *group, IncDcErrorInfo *err = nullptr);

// 将 logical rank 映射到 SHMEM PE
int32_t GroupWorkerPe(const IncDcGroupConfig *group, uint32_t rank);

// expert id -> destination rank；-1 表示未映射
int32_t ExpertDestinationRank(const IncDcGroupConfig *group, int32_t expert_id);

} // namespace dc
} // namespace inc

#endif
