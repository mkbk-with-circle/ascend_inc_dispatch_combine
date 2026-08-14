#include "inc_fusion_api.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

const char *StatusName(inc_fusion_status_t status)
{
    switch (status) {
        case INC_FUSION_OK: return "OK";
        case INC_FUSION_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case INC_FUSION_INVALID_PLAN: return "INVALID_PLAN";
        case INC_FUSION_BUFFER_TOO_SMALL: return "BUFFER_TOO_SMALL";
        case INC_FUSION_BUSY: return "BUSY";
        case INC_FUSION_RUNTIME_ERROR: return "RUNTIME_ERROR";
    }
    return "UNKNOWN";
}

double MiB(uint64_t bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

} // namespace

int main()
{
    // 这是一个纯 Host smoke：只编译执行计划，不初始化 NPU，也不会启动
    // Dispatch/FFN/Combine。它适合用来确认公开 C API、shape 和资源预算。
    inc_fusion_plan_desc_t desc{};
    inc_fusion_plan_desc_init(&desc);  // 先取得 slot/ring 等稳定默认值。

    // 示例拓扑：4 个 Worker（PE 0..3）+ 1 个 INC（PE 4）。
    desc.live_aiv = 48;
    desc.live_aic = 24;
    desc.worker_count = 4;
    desc.rank = 0;                     // 当前进程对应 Worker rank 0。
    desc.inc_pe = 4;

    // 示例 MoE shape。token_count 不需要整除 tokens_per_wave；末 wave 会缩短。
    desc.hidden = 1024;
    desc.intermediate = 4096;
    desc.expert_count = 8;
    desc.topk = 2;
    desc.token_count = 513;
    desc.tokens_per_wave = 128;
    desc.slot_count = 3;               // 至少容纳流水中的在途 wave。
    desc.service_ring_size = 4;        // 同时在途的推理请求上限。
    desc.activation_waves = 2;
    desc.spin_cap = 0;                 // 0：任意时序下等待 generation，而非超时失败。

    // expert_owner[e] 给出全局 expert e 所属的 Worker rank；
    // expert_local_index[e] 给出它在该 Worker 本地权重张量中的连续下标。
    std::vector<uint32_t> expert_owner(desc.expert_count);
    std::vector<uint32_t> expert_local_index(desc.expert_count);
    std::vector<uint32_t> next_local(desc.worker_count, 0);
    for (uint32_t expert = 0; expert < desc.expert_count; ++expert) {
        expert_owner[expert] = expert % desc.worker_count;
        expert_local_index[expert] = next_local[expert_owner[expert]]++;
    }

    inc_fusion_prepared_plan_t *plan = nullptr;
    const inc_fusion_status_t create_status = inc_fusion_prepared_plan_create(
        &desc, expert_owner.data(), expert_local_index.data(), &plan);
    if (create_status != INC_FUSION_OK) {
        std::fprintf(stderr, "plan create failed: %s (%d)\n",
                     StatusName(create_status), static_cast<int>(create_status));
        return 1;
    }

    inc_fusion_plan_info_t info{};
    const inc_fusion_status_t info_status =
        inc_fusion_prepared_plan_info(plan, &info);
    if (info_status != INC_FUSION_OK) {
        std::fprintf(stderr, "plan info failed: %s (%d)\n",
                     StatusName(info_status), static_cast<int>(info_status));
        inc_fusion_prepared_plan_destroy(plan);
        return 1;
    }

    // 513 / 128 向上取整为 5 个 token wave。输出同时展示对称堆、两种
    // workspace 与 AIV 分组；集成方应使用这些值分配内存，不能写死大小。
    std::printf("INC Fusion plan ready\n");
    std::printf("  ABI version       : %u\n", info.fusion_abi_version);
    std::printf("  token waves       : %u\n", info.wave_count);
    std::printf("  local experts     : %u\n", info.local_expert_count);
    std::printf("  symmetric heap    : %.2f MiB (%" PRIu64 " bytes)\n",
                MiB(info.symmetric_bytes), info.symmetric_bytes);
    std::printf("  worker workspace  : %.2f MiB (%" PRIu64 " bytes)\n",
                MiB(info.worker_workspace_bytes), info.worker_workspace_bytes);
    std::printf("  INC workspace     : %.2f MiB (%" PRIu64 " bytes)\n",
                MiB(info.inc_workspace_bytes), info.inc_workspace_bytes);
    std::printf("  worker AIV D/C/FFN: %u/%u/%u\n",
                info.worker_dispatch_aiv, info.worker_combine_aiv,
                info.worker_compute_aiv);
    std::printf("  INC AIV D/C       : %u/%u\n",
                info.inc_dispatch_aiv, info.inc_combine_aiv);
    std::printf("  service ring      : %u\n", info.remote_service_ring_size);
    std::printf("  resource hash     : 0x%016" PRIx64 "\n",
                info.resource_fingerprint);

    const bool sane = info.wave_count == 5 && info.symmetric_bytes != 0 &&
        info.worker_workspace_bytes != 0 && info.inc_workspace_bytes != 0 &&
        info.worker_dispatch_aiv != 0 && info.worker_combine_aiv != 0 &&
        info.worker_compute_aiv != 0 && info.inc_dispatch_aiv != 0 &&
        info.inc_combine_aiv != 0;
    inc_fusion_prepared_plan_destroy(plan);

    std::printf("  smoke result      : %s\n", sane ? "PASS" : "FAIL");
    return sane ? 0 : 1;
}
