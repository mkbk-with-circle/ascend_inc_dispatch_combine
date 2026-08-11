#include "inc_fusion_api.h"
#include "inc_fusion_abi.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

int main()
{
    inc_fusion_plan_desc_t desc{};
    inc_fusion_plan_desc_init(&desc);
    desc.live_aiv = 48u;
    desc.live_aic = 24u;
    desc.worker_count = 4u;
    desc.rank = 1u;
    desc.inc_pe = 4u;
    desc.hidden = 2048u;
    desc.intermediate = 8192u;
    desc.expert_count = 10u;
    desc.topk = 4u;
    desc.token_count = 1025u;
    desc.tokens_per_wave = 256u;
    std::vector<uint32_t> owner(desc.expert_count);
    std::vector<uint32_t> local(desc.expert_count);
    uint32_t next[4] = {};
    for (uint32_t e = 0u; e < desc.expert_count; ++e) {
        owner[e] = e % 4u;
        local[e] = next[owner[e]]++;
    }
    inc_fusion_prepared_plan_t *plan = nullptr;
    assert(inc_fusion_prepared_plan_create(
               &desc, owner.data(), local.data(), &plan) == INC_FUSION_OK);
    inc_fusion_plan_info_t info{};
    assert(inc_fusion_prepared_plan_info(plan, &info) == INC_FUSION_OK);
    assert(info.wave_count == 5u);
    assert(info.worker_dispatch_aiv == 8u);
    assert(info.worker_combine_aiv == 16u);
    assert(info.worker_compute_aiv == 24u);
    assert(info.inc_dispatch_aiv == 32u);
    assert(info.inc_combine_aiv == 16u);
    assert(info.symmetric_bytes > 0u);
    assert(info.worker_workspace_bytes > 0u);
    assert(info.inc_workspace_bytes > 0u);
    assert(info.remote_service_bytes > 0u);
    // Every PE allocates the same symmetric layout even when ranks own a
    // different number of experts and therefore have different local
    // workspace requirements.
    for (uint32_t rank = 0u; rank < desc.worker_count; ++rank) {
        inc_fusion_plan_desc_t peer_desc = desc;
        peer_desc.rank = rank;
        inc_fusion_prepared_plan_t *peer = nullptr;
        assert(inc_fusion_prepared_plan_create(
                   &peer_desc, owner.data(), local.data(), &peer) ==
               INC_FUSION_OK);
        inc_fusion_plan_info_t peer_info{};
        assert(inc_fusion_prepared_plan_info(peer, &peer_info) ==
               INC_FUSION_OK);
        assert(peer_info.symmetric_bytes == info.symmetric_bytes);
        assert(peer_info.wave_count == info.wave_count);
        assert(peer_info.fusion_abi_version == info.fusion_abi_version);
        assert(peer_info.remote_service_bytes == info.remote_service_bytes);
        inc_fusion_prepared_plan_destroy(peer);
    }
    std::vector<uint8_t> args(info.kernel_args_bytes);
    inc_fusion_device_bindings_t bindings{};
    bindings.symmetric_base = reinterpret_cast<void *>(1u);
    bindings.input = reinterpret_cast<void *>(2u);
    bindings.output = reinterpret_cast<void *>(3u);
    bindings.w13 = reinterpret_cast<void *>(4u);
    bindings.w2 = reinterpret_cast<void *>(5u);
    bindings.dispatch_rows = reinterpret_cast<void *>(6u);
    bindings.assignments = reinterpret_cast<void *>(7u);
    bindings.waves = reinterpret_cast<void *>(8u);
    bindings.worker_pes = reinterpret_cast<void *>(9u);
    bindings.active_token_counts = reinterpret_cast<void *>(12u);
    bindings.group_lists = reinterpret_cast<void *>(10u);
    bindings.workspace = reinterpret_cast<void *>(11u);
    bindings.execution_flags = INC_FUSION_EXEC_SERIALIZE_INC_DC |
        INC_FUSION_EXEC_STRICT_SERIAL_PIPELINE |
        INC_FUSION_EXEC_WORKER_DIRECT_SHMEM |
        INC_FUSION_EXEC_WEIGHT_B_ROW_MAJOR |
        INC_FUSION_EXEC_GLOBAL_OUTPUT_FANOUT;
    assert(inc_fusion_prepared_build_args(
               plan, INC_FUSION_ROLE_WORKER, 17u, &bindings,
               args.data(), args.size()) == INC_FUSION_OK);
    const auto *kernel_args = reinterpret_cast<const inc::fusion::FusionKernelArgs *>(
        args.data());
    assert((kernel_args->flags & inc::fusion::kFusionSerializeIncDc) != 0u);
    assert((kernel_args->flags &
            inc::fusion::kFusionStrictSerialPipeline) != 0u);
    assert((kernel_args->flags &
            inc::fusion::kFusionWorkerDirectShmem) != 0u);
    assert((kernel_args->flags &
            inc::fusion::kFusionWeightBRowMajor) != 0u);
    assert((kernel_args->flags &
            inc::fusion::kFusionGlobalOutputFanout) != 0u);
    assert(kernel_args->active_token_counts == 12u);
    assert(inc_fusion_prepared_build_args(
               plan, INC_FUSION_ROLE_WORKER, UINT64_MAX, &bindings,
               args.data(), args.size()) == INC_FUSION_INVALID_ARGUMENT);
    assert(inc_fusion_prepared_build_args(
               plan, INC_FUSION_ROLE_WORKER, 17u, &bindings,
               args.data(), args.size() - 1u) ==
           INC_FUSION_BUFFER_TOO_SMALL);
    inc_fusion_worker_executor_t *executor = nullptr;
    assert(inc_fusion_worker_executor_create(
               plan, &bindings, 1u, &executor) ==
           INC_FUSION_INVALID_ARGUMENT);
    assert(executor == nullptr);
    assert(inc_fusion_worker_executor_enqueue(
               nullptr, 17u, &bindings,
               reinterpret_cast<void *>(1u)) ==
           INC_FUSION_INVALID_ARGUMENT);
    inc_fusion_worker_executor_info_t executor_info{};
    assert(inc_fusion_worker_executor_info(nullptr, &executor_info) ==
           INC_FUSION_INVALID_ARGUMENT);
    inc_fusion_worker_executor_destroy(nullptr);
    inc_fusion_persistent_service_t *service = nullptr;
    assert(inc_fusion_persistent_service_create(
               plan, reinterpret_cast<void *>(1u), 1u,
               reinterpret_cast<void *>(2u), &service) ==
           INC_FUSION_INVALID_ARGUMENT);
    assert(service == nullptr);
    assert(inc_fusion_persistent_service_submit(
               nullptr, reinterpret_cast<void *>(1u), 1u,
               reinterpret_cast<void *>(2u), nullptr) ==
           INC_FUSION_INVALID_ARGUMENT);
    inc_fusion_service_result_t result{};
    assert(inc_fusion_persistent_service_query(nullptr, 1u, &result) ==
           INC_FUSION_INVALID_ARGUMENT);
    assert(inc_fusion_persistent_service_device_control(nullptr) == nullptr);
    inc_fusion_prepared_plan_destroy(plan);
    std::cout << "inc fusion prepared API tests passed\n";
    return 0;
}
