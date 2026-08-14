#include "inc_fusion_api.h"

#include <acl/acl.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "inc_fusion_plan.h"

using inc::fusion::FusionKernelArgs;
using inc::fusion::FusionPlan;
using inc::fusion::FusionPlanConfig;
using inc::fusion::FusionServiceControl;
using inc::fusion::FusionServiceDescriptor;
using inc::fusion::MakeFusionKernelArgs;
using inc::fusion::kFusionInc;
using inc::fusion::kFusionMaxServiceRing;
using inc::fusion::kFusionMinSlots;
using inc::fusion::kFusionMinServiceRing;
using inc::fusion::kFusionServiceRequestPending;
using inc::fusion::kFusionWorker;

extern "C" void launch_inc_fusion_worker_kernel(
    uint8_t *sym, FusionKernelArgs *args, uint64_t expected_ticket,
    int block_dim, void *stream);
extern "C" void launch_inc_fusion_remote_publish_kernel(
    uint8_t *sym, FusionKernelArgs *args, uint64_t expected_ticket,
    void *stream);
extern "C" void launch_inc_fusion_inc_service_kernel(
    uint8_t *sym, FusionKernelArgs *args, int block_dim, void *stream);
extern "C" void launch_inc_fusion_inc_persistent_service_kernel(
    uint8_t *sym, FusionServiceControl *control,
    int block_dim, void *stream);

extern "C" void launch_inc_fusion_route_exchange_kernel(
    uint8_t *sym, const uint32_t *local_counts, uint32_t *global_counts,
    const uint32_t *worker_pes, uint64_t ffts_addr, uint64_t counts_off,
    uint64_t doorbells_off, uint32_t count_words, uint32_t worker_count,
    uint32_t rank, int32_t relay_pe, uint64_t generation, void *stream);

struct inc_fusion_prepared_plan {
    FusionPlan value;
};

struct inc_fusion_persistent_service {
    const inc_fusion_prepared_plan_t *plan = nullptr;
    uint8_t *symmetric_base = nullptr;
    FusionServiceControl *device_control = nullptr;
    FusionServiceDescriptor *device_descriptors = nullptr;
    uint8_t *device_lane_progress = nullptr;
    FusionServiceDescriptor *host_staging = nullptr;
    uint64_t *host_stop = nullptr;
    void *service_stream = nullptr;
    void *control_stream = nullptr;
    uint32_t ring_size = 0u;
    uint64_t next_ticket = 1u;
    uint64_t completed_cached = 0u;
    bool running = false;
    bool owns_device_storage = true;
    bool remote = false;
};

struct inc_fusion_worker_executor {
    const inc_fusion_prepared_plan_t *plan = nullptr;
    inc_fusion_device_bindings_t static_bindings{};
    void *workspace = nullptr;
    void *device_args = nullptr;
    void *host_args = nullptr;
    uint64_t workspace_bytes = 0u;
    uint32_t ring_size = 0u;
    uint64_t submitted = 0u;
    bool owns_workspace = false;
    std::vector<aclrtEvent> events;
    std::vector<uint8_t> in_use;
};

namespace {

void FreeWorkerExecutorStorage(inc_fusion_worker_executor_t *executor,
                               bool wait)
{
    if (executor == nullptr) return;
    for (size_t slot = 0u; slot < executor->events.size(); ++slot) {
        if (executor->events[slot] == nullptr) continue;
        if (wait && slot < executor->in_use.size() &&
            executor->in_use[slot] != 0u)
            (void)aclrtSynchronizeEvent(executor->events[slot]);
        (void)aclrtDestroyEvent(executor->events[slot]);
    }
    executor->events.clear();
    executor->in_use.clear();
    if (executor->host_args != nullptr)
        (void)aclrtFreeHost(executor->host_args);
    if (executor->device_args != nullptr)
        (void)aclrtFree(executor->device_args);
    if (executor->owns_workspace && executor->workspace != nullptr)
        (void)aclrtFree(executor->workspace);
    executor->host_args = nullptr;
    executor->device_args = nullptr;
    executor->workspace = nullptr;
}

void FreeServiceStorage(inc_fusion_persistent_service_t *service)
{
    if (service == nullptr) return;
    if (service->owns_device_storage && service->device_control != nullptr)
        (void)aclrtFree(service->device_control);
    if (service->owns_device_storage &&
        service->device_descriptors != nullptr)
        (void)aclrtFree(service->device_descriptors);
    if (service->owns_device_storage &&
        service->device_lane_progress != nullptr)
        (void)aclrtFree(service->device_lane_progress);
    if (service->host_staging != nullptr)
        (void)aclrtFreeHost(service->host_staging);
    if (service->host_stop != nullptr)
        (void)aclrtFreeHost(service->host_stop);
    if (service->control_stream != nullptr)
        (void)aclrtDestroyStream(
            static_cast<aclrtStream>(service->control_stream));
}

inc_fusion_status_t RefreshServiceCompleted(
    inc_fusion_persistent_service_t *service)
{
    uint64_t completed = 0u;
    auto *source = reinterpret_cast<uint8_t *>(service->device_control) +
        offsetof(FusionServiceControl, completed_sequence);
    if (aclrtMemcpy(&completed, sizeof(completed), source,
                    sizeof(completed), ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS)
        return INC_FUSION_RUNTIME_ERROR;
    if (completed > service->completed_cached)
        service->completed_cached = completed;
    return INC_FUSION_OK;
}

} // namespace

extern "C" void inc_fusion_plan_desc_init(inc_fusion_plan_desc_t *desc)
{
    if (desc == nullptr) return;
    std::memset(desc, 0, sizeof(*desc));
    desc->slot_count = kFusionMinSlots;
    desc->service_ring_size = 4u;
    desc->activation_waves = 2u;
}

extern "C" inc_fusion_status_t inc_fusion_prepared_plan_create(
    const inc_fusion_plan_desc_t *desc,
    const uint32_t *expert_owner,
    const uint32_t *expert_local_index,
    inc_fusion_prepared_plan_t **out)
{
    if (desc == nullptr || expert_owner == nullptr ||
        expert_local_index == nullptr || out == nullptr)
        return INC_FUSION_INVALID_ARGUMENT;
    *out = nullptr;
    FusionPlanConfig config{};
    config.live_aiv = desc->live_aiv;
    config.live_aic = desc->live_aic;
    config.worker_count = desc->worker_count;
    config.rank = desc->rank;
    config.inc_pe = desc->inc_pe;
    config.hidden = desc->hidden;
    config.intermediate = desc->intermediate;
    config.expert_count = desc->expert_count;
    config.topk = desc->topk;
    config.token_count = desc->token_count;
    config.tokens_per_wave = desc->tokens_per_wave;
    config.slot_count = desc->slot_count;
    config.service_ring_size = desc->service_ring_size;
    config.activation_waves = desc->activation_waves;
    config.spin_cap = desc->spin_cap;
    inc_fusion_prepared_plan_t *prepared =
        new (std::nothrow) inc_fusion_prepared_plan_t;
    if (prepared == nullptr) return INC_FUSION_INVALID_PLAN;
    std::string error;
    if (!inc::fusion::BuildFusionPlan(
            config, expert_owner, expert_local_index,
            &prepared->value, &error)) {
        delete prepared;
        return INC_FUSION_INVALID_PLAN;
    }
    *out = prepared;
    return INC_FUSION_OK;
}

extern "C" void inc_fusion_prepared_plan_destroy(
    inc_fusion_prepared_plan_t *plan)
{
    delete plan;
}

extern "C" inc_fusion_status_t inc_fusion_prepared_plan_info(
    const inc_fusion_prepared_plan_t *plan,
    inc_fusion_plan_info_t *info)
{
    if (plan == nullptr || info == nullptr)
        return INC_FUSION_INVALID_ARGUMENT;
    std::memset(info, 0, sizeof(*info));
    const FusionPlan &value = plan->value;
    info->symmetric_bytes = value.symmetric.total_bytes;
    info->worker_workspace_bytes = value.worker_workspace.total_bytes;
    info->inc_workspace_bytes = value.inc_workspace.total_bytes;
    info->resource_fingerprint = value.resources.fingerprint;
    info->wave_count = static_cast<uint32_t>(value.waves.size());
    info->local_expert_count = value.local_expert_counts[value.config.rank];
    info->worker_dispatch_aiv = value.resources.worker_dispatch_aiv;
    info->worker_combine_aiv = value.resources.worker_combine_aiv;
    info->worker_compute_aiv = value.resources.worker_compute_aiv;
    info->inc_dispatch_aiv = value.resources.inc_dispatch_aiv;
    info->inc_combine_aiv = value.resources.inc_combine_aiv;
    info->kernel_args_bytes = sizeof(FusionKernelArgs);
    info->remote_service_ring_size = value.remote_service.ring_size;
    info->fusion_abi_version = inc::fusion::kFusionAbiVersion;
    info->remote_service_bytes = value.remote_service.total_bytes;
    return INC_FUSION_OK;
}

extern "C" inc_fusion_status_t inc_fusion_prepared_route_exchange_info(
    const inc_fusion_prepared_plan_t *plan,
    inc_fusion_route_exchange_info_t *info)
{
    if (plan == nullptr || info == nullptr)
        return INC_FUSION_INVALID_ARGUMENT;
    info->counts_off = plan->value.symmetric.reserved64[1];
    info->doorbells_off = plan->value.symmetric.reserved64[3];
    return INC_FUSION_OK;
}

extern "C" inc_fusion_status_t inc_fusion_prepared_build_args(
    const inc_fusion_prepared_plan_t *plan,
    inc_fusion_role_t role,
    uint64_t generation,
    const inc_fusion_device_bindings_t *bindings,
    void *args_buffer,
    size_t args_buffer_bytes)
{
    if (plan == nullptr || bindings == nullptr || args_buffer == nullptr ||
        (role != INC_FUSION_ROLE_WORKER && role != INC_FUSION_ROLE_INC))
        return INC_FUSION_INVALID_ARGUMENT;
    if (args_buffer_bytes < sizeof(FusionKernelArgs))
        return INC_FUSION_BUFFER_TOO_SMALL;
    // ABI v2 commits encode an exact 32-bit wave generation in the high half
    // of the 64-bit queue ticket. Reject wrap instead of aliasing an old op.
    if (plan->value.waves.size() > 0xffffffffull ||
        generation > 0xffffffffull - plan->value.waves.size())
        return INC_FUSION_INVALID_ARGUMENT;
    if (bindings->symmetric_base == nullptr || bindings->waves == nullptr ||
        bindings->worker_pes == nullptr || bindings->workspace == nullptr)
        return INC_FUSION_INVALID_ARGUMENT;
    if (role == INC_FUSION_ROLE_WORKER &&
        (bindings->input == nullptr || bindings->output == nullptr ||
         bindings->w13 == nullptr || bindings->w2 == nullptr ||
         bindings->dispatch_rows == nullptr || bindings->assignments == nullptr ||
         bindings->group_lists == nullptr))
        return INC_FUSION_INVALID_ARGUMENT;
    FusionKernelArgs args = MakeFusionKernelArgs(
        plan->value, role == INC_FUSION_ROLE_WORKER
                         ? kFusionWorker : kFusionInc,
        generation);
    args.symmetric_base = reinterpret_cast<uint64_t>(bindings->symmetric_base);
    args.input = reinterpret_cast<uint64_t>(bindings->input);
    args.output = reinterpret_cast<uint64_t>(bindings->output);
    args.w13 = reinterpret_cast<uint64_t>(bindings->w13);
    args.w2 = reinterpret_cast<uint64_t>(bindings->w2);
    args.dispatch_rows = reinterpret_cast<uint64_t>(bindings->dispatch_rows);
    args.assignments = reinterpret_cast<uint64_t>(bindings->assignments);
    args.waves = reinterpret_cast<uint64_t>(bindings->waves);
    args.expert_owner = reinterpret_cast<uint64_t>(bindings->expert_owner);
    args.expert_local_index =
        reinterpret_cast<uint64_t>(bindings->expert_local_index);
    args.worker_pes = reinterpret_cast<uint64_t>(bindings->worker_pes);
    args.active_token_counts =
        reinterpret_cast<uint64_t>(bindings->active_token_counts);
    args.group_lists = reinterpret_cast<uint64_t>(bindings->group_lists);
    args.workspace = reinterpret_cast<uint64_t>(bindings->workspace);
    args.system_workspace =
        reinterpret_cast<uint64_t>(bindings->system_workspace);
    args.ffts_addr = bindings->ffts_addr;
    if ((bindings->execution_flags &
         INC_FUSION_EXEC_SERIALIZE_INC_DC) != 0u)
        args.flags |= inc::fusion::kFusionSerializeIncDc;
    if ((bindings->execution_flags &
         INC_FUSION_EXEC_REMOTE_INC_SERVICE) != 0u)
        args.flags |= inc::fusion::kFusionRemoteIncService;
    if ((bindings->execution_flags &
         INC_FUSION_EXEC_STRICT_SERIAL_PIPELINE) != 0u)
        args.flags |= inc::fusion::kFusionStrictSerialPipeline;
    if ((bindings->execution_flags &
         INC_FUSION_EXEC_WORKER_DIRECT_SHMEM) != 0u)
        args.flags |= inc::fusion::kFusionWorkerDirectShmem;
    if ((bindings->execution_flags &
         INC_FUSION_EXEC_WEIGHT_B_ROW_MAJOR) != 0u)
        args.flags |= inc::fusion::kFusionWeightBRowMajor;
    if ((bindings->execution_flags &
         INC_FUSION_EXEC_GLOBAL_OUTPUT_FANOUT) != 0u)
        args.flags |= inc::fusion::kFusionGlobalOutputFanout;
    std::memcpy(args_buffer, &args, sizeof(args));
    return INC_FUSION_OK;
}

extern "C" inc_fusion_status_t inc_fusion_prepared_enqueue(
    const inc_fusion_prepared_plan_t *plan,
    inc_fusion_role_t role,
    void *symmetric_base,
    void *device_args,
    void *stream)
{
    if (plan == nullptr || symmetric_base == nullptr ||
        device_args == nullptr || stream == nullptr ||
        (role != INC_FUSION_ROLE_WORKER && role != INC_FUSION_ROLE_INC))
        return INC_FUSION_INVALID_ARGUMENT;
    if (role == INC_FUSION_ROLE_WORKER)
        launch_inc_fusion_worker_kernel(
            static_cast<uint8_t *>(symmetric_base),
            static_cast<FusionKernelArgs *>(device_args),
            0u, static_cast<int>(plan->value.resources.live_aic), stream);
    else
        launch_inc_fusion_inc_service_kernel(
            static_cast<uint8_t *>(symmetric_base),
            static_cast<FusionKernelArgs *>(device_args),
            static_cast<int>(plan->value.resources.live_aiv), stream);
    return INC_FUSION_OK;
}

extern "C" inc_fusion_status_t inc_fusion_worker_executor_create(
    const inc_fusion_prepared_plan_t *plan,
    const inc_fusion_device_bindings_t *static_bindings,
    uint32_t ring_size,
    inc_fusion_worker_executor_t **out)
{
    if (plan == nullptr || static_bindings == nullptr || out == nullptr ||
        ring_size < kFusionMinServiceRing ||
        ring_size > kFusionMaxServiceRing ||
        static_bindings->symmetric_base == nullptr ||
        static_bindings->expert_owner == nullptr ||
        static_bindings->expert_local_index == nullptr ||
        static_bindings->worker_pes == nullptr ||
        (((static_bindings->execution_flags &
           INC_FUSION_EXEC_REMOTE_INC_SERVICE) != 0u) &&
         static_bindings->active_token_counts == nullptr))
        return INC_FUSION_INVALID_ARGUMENT;
    *out = nullptr;
    auto *executor = new (std::nothrow) inc_fusion_worker_executor_t;
    if (executor == nullptr) return INC_FUSION_RUNTIME_ERROR;
    executor->plan = plan;
    executor->static_bindings = *static_bindings;
    executor->ring_size = ring_size;
    executor->workspace_bytes = plan->value.worker_workspace.total_bytes;
    executor->workspace = static_bindings->workspace;
    executor->owns_workspace = executor->workspace == nullptr;
    const size_t args_bytes =
        static_cast<size_t>(ring_size) * sizeof(FusionKernelArgs);
    bool ok = true;
    if (executor->owns_workspace)
        ok = aclrtMalloc(&executor->workspace, executor->workspace_bytes,
                         ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    if (ok)
        ok = aclrtMalloc(&executor->device_args, args_bytes,
                         ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    if (ok)
        ok = aclrtMallocHost(&executor->host_args, args_bytes) == ACL_SUCCESS;
    if (ok && executor->owns_workspace)
        ok = aclrtMemset(executor->workspace, executor->workspace_bytes,
                         0, executor->workspace_bytes) == ACL_SUCCESS;
    if (ok)
        ok = aclrtMemset(executor->device_args, args_bytes,
                         0, args_bytes) == ACL_SUCCESS;
    if (ok && (static_bindings->execution_flags &
               INC_FUSION_EXEC_REMOTE_INC_SERVICE) != 0u) {
        uint8_t *descriptors = static_cast<uint8_t *>(
            static_bindings->symmetric_base) +
            plan->value.remote_service.descriptors_off;
        const uint64_t bytes = static_cast<uint64_t>(
            plan->value.remote_service.ring_size) *
            sizeof(FusionServiceDescriptor);
        ok = aclrtMemset(descriptors, bytes, 0, bytes) == ACL_SUCCESS;
        const uint64_t ready =
            static_cast<uint64_t>(inc::fusion::kFusionServiceAbiVersion)
                << 32u |
            inc::fusion::kFusionServiceMagic;
        uint8_t *control = static_cast<uint8_t *>(
            static_bindings->symmetric_base) +
            plan->value.remote_service.control_off;
        if (ok)
            ok = aclrtMemcpy(control, sizeof(ready), &ready, sizeof(ready),
                             ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    }
    if (ok) {
        std::memset(executor->host_args, 0, args_bytes);
        executor->events.assign(ring_size, nullptr);
        executor->in_use.assign(ring_size, 0u);
        for (uint32_t slot = 0u; slot < ring_size; ++slot) {
            if (aclrtCreateEvent(&executor->events[slot]) != ACL_SUCCESS) {
                ok = false;
                break;
            }
        }
    }
    if (!ok) {
        FreeWorkerExecutorStorage(executor, false);
        delete executor;
        return INC_FUSION_RUNTIME_ERROR;
    }
    executor->static_bindings.workspace = executor->workspace;
    *out = executor;
    return INC_FUSION_OK;
}

extern "C" inc_fusion_status_t inc_fusion_worker_executor_enqueue(
    inc_fusion_worker_executor_t *executor,
    uint64_t generation,
    const inc_fusion_device_bindings_t *dynamic_bindings,
    void *stream)
{
    if (executor == nullptr || dynamic_bindings == nullptr ||
        stream == nullptr)
        return INC_FUSION_INVALID_ARGUMENT;
    const uint32_t slot = static_cast<uint32_t>(
        executor->submitted % executor->ring_size);
    if (executor->in_use[slot] != 0u) {
        aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
        if (aclrtQueryEventStatus(executor->events[slot], &status) !=
            ACL_SUCCESS)
            return INC_FUSION_RUNTIME_ERROR;
        if (status != ACL_EVENT_RECORDED_STATUS_COMPLETE)
            return INC_FUSION_BUSY;
        executor->in_use[slot] = 0u;
    }

    inc_fusion_device_bindings_t bindings = *dynamic_bindings;
    bindings.symmetric_base = executor->static_bindings.symmetric_base;
    bindings.expert_owner = executor->static_bindings.expert_owner;
    bindings.expert_local_index =
        executor->static_bindings.expert_local_index;
    bindings.worker_pes = executor->static_bindings.worker_pes;
    bindings.workspace = executor->workspace;
    bindings.system_workspace =
        executor->static_bindings.system_workspace;
    bindings.ffts_addr = executor->static_bindings.ffts_addr;
    if (bindings.active_token_counts == nullptr)
        bindings.active_token_counts =
            executor->static_bindings.active_token_counts;
    bindings.execution_flags |=
        executor->static_bindings.execution_flags;

    auto *host = reinterpret_cast<uint8_t *>(executor->host_args) +
        static_cast<size_t>(slot) * sizeof(FusionKernelArgs);
    auto *device = reinterpret_cast<uint8_t *>(executor->device_args) +
        static_cast<size_t>(slot) * sizeof(FusionKernelArgs);
    const inc_fusion_status_t built = inc_fusion_prepared_build_args(
        executor->plan, INC_FUSION_ROLE_WORKER, generation,
        &bindings, host, sizeof(FusionKernelArgs));
    if (built != INC_FUSION_OK) return built;
    auto *built_args = reinterpret_cast<FusionKernelArgs *>(host);
    if ((built_args->flags & inc::fusion::kFusionRemoteIncService) != 0u) {
        built_args->service_ticket = executor->submitted + 1u;
        built_args->request_id = executor->submitted + 1u;
    }
    if (aclrtMemcpyAsync(device, sizeof(FusionKernelArgs), host,
                         sizeof(FusionKernelArgs),
                         ACL_MEMCPY_HOST_TO_DEVICE,
                         static_cast<aclrtStream>(stream)) != ACL_SUCCESS)
        return INC_FUSION_RUNTIME_ERROR;
    if ((built_args->flags & inc::fusion::kFusionRemoteIncService) != 0u &&
        executor->plan->value.config.rank <
            executor->plan->value.config.worker_count) {
        launch_inc_fusion_remote_publish_kernel(
            static_cast<uint8_t *>(executor->static_bindings.symmetric_base),
            reinterpret_cast<FusionKernelArgs *>(device),
            built_args->service_ticket, stream);
    }
    launch_inc_fusion_worker_kernel(
        static_cast<uint8_t *>(executor->static_bindings.symmetric_base),
        reinterpret_cast<FusionKernelArgs *>(device),
        built_args->service_ticket,
        static_cast<int>(executor->plan->value.resources.live_aic), stream);
    const inc_fusion_status_t launched = INC_FUSION_OK;
    if (launched != INC_FUSION_OK) {
        (void)aclrtSynchronizeStream(static_cast<aclrtStream>(stream));
        return launched;
    }
    if (aclrtRecordEvent(executor->events[slot],
                         static_cast<aclrtStream>(stream)) != ACL_SUCCESS) {
        (void)aclrtSynchronizeStream(static_cast<aclrtStream>(stream));
        return INC_FUSION_RUNTIME_ERROR;
    }
    executor->in_use[slot] = 1u;
    ++executor->submitted;
    return INC_FUSION_OK;
}

extern "C" inc_fusion_status_t inc_fusion_worker_executor_info(
    const inc_fusion_worker_executor_t *executor,
    inc_fusion_worker_executor_info_t *info)
{
    if (executor == nullptr || info == nullptr)
        return INC_FUSION_INVALID_ARGUMENT;
    std::memset(info, 0, sizeof(*info));
    info->workspace = executor->workspace;
    info->device_args = executor->device_args;
    info->host_args = executor->host_args;
    info->workspace_bytes = executor->workspace_bytes;
    info->ring_size = executor->ring_size;
    info->owns_workspace = executor->owns_workspace ? 1u : 0u;
    info->submitted = executor->submitted;
    info->kernel_args_bytes = sizeof(FusionKernelArgs);
    return INC_FUSION_OK;
}

extern "C" void inc_fusion_worker_executor_destroy(
    inc_fusion_worker_executor_t *executor)
{
    if (executor == nullptr) return;
    FreeWorkerExecutorStorage(executor, true);
    delete executor;
}

extern "C" inc_fusion_status_t inc_fusion_persistent_service_create(
    const inc_fusion_prepared_plan_t *plan,
    void *symmetric_base,
    uint32_t ring_size,
    void *service_stream,
    inc_fusion_persistent_service_t **out)
{
    if (plan == nullptr || symmetric_base == nullptr ||
        service_stream == nullptr || out == nullptr ||
        ring_size < kFusionMinServiceRing ||
        ring_size > kFusionMaxServiceRing)
        return INC_FUSION_INVALID_ARGUMENT;
    *out = nullptr;
    auto *service = new (std::nothrow) inc_fusion_persistent_service_t;
    if (service == nullptr) return INC_FUSION_RUNTIME_ERROR;
    service->plan = plan;
    service->symmetric_base = static_cast<uint8_t *>(symmetric_base);
    service->service_stream = service_stream;
    service->ring_size = ring_size;
    const size_t descriptor_bytes =
        static_cast<size_t>(ring_size) * sizeof(FusionServiceDescriptor);
    const size_t lane_bytes =
        static_cast<size_t>(plan->value.resources.live_aiv) *
        inc::fusion::kFusionCacheLineBytes;
    aclrtStream control_stream = nullptr;
    bool ok =
        aclrtMalloc(reinterpret_cast<void **>(&service->device_control),
                    sizeof(FusionServiceControl),
                    ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
        aclrtMalloc(reinterpret_cast<void **>(&service->device_descriptors),
                    descriptor_bytes,
                    ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
        aclrtMalloc(reinterpret_cast<void **>(&service->device_lane_progress),
                    lane_bytes,
                    ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
        aclrtMallocHost(reinterpret_cast<void **>(&service->host_staging),
                        descriptor_bytes) == ACL_SUCCESS &&
        aclrtMallocHost(reinterpret_cast<void **>(&service->host_stop),
                        sizeof(uint64_t)) == ACL_SUCCESS &&
        aclrtCreateStream(&control_stream) == ACL_SUCCESS;
    service->control_stream = control_stream;
    if (!ok) {
        FreeServiceStorage(service);
        delete service;
        return INC_FUSION_RUNTIME_ERROR;
    }
    std::memset(service->host_staging, 0, descriptor_bytes);
    *service->host_stop = 0u;
    FusionServiceControl control{};
    control.ring_size = ring_size;
    control.live_aiv = plan->value.resources.live_aiv;
    control.descriptors = reinterpret_cast<uint64_t>(
        service->device_descriptors);
    control.lane_progress = reinterpret_cast<uint64_t>(
        service->device_lane_progress);
    if (aclrtMemset(service->device_descriptors, descriptor_bytes,
                    0, descriptor_bytes) != ACL_SUCCESS ||
        aclrtMemset(service->device_lane_progress, lane_bytes,
                    0, lane_bytes) != ACL_SUCCESS ||
        aclrtMemcpy(service->device_control, sizeof(control),
                    &control, sizeof(control),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        FreeServiceStorage(service);
        delete service;
        return INC_FUSION_RUNTIME_ERROR;
    }
    launch_inc_fusion_inc_persistent_service_kernel(
        service->symmetric_base, service->device_control,
        static_cast<int>(plan->value.resources.live_aiv), service_stream);
    service->running = true;
    *out = service;
    return INC_FUSION_OK;
}

extern "C" inc_fusion_status_t inc_fusion_remote_service_create(
    const inc_fusion_prepared_plan_t *plan,
    const inc_fusion_device_bindings_t *static_bindings,
    void *service_stream,
    inc_fusion_persistent_service_t **out)
{
    if (plan == nullptr || static_bindings == nullptr ||
        static_bindings->symmetric_base == nullptr ||
        static_bindings->workspace == nullptr ||
        static_bindings->worker_pes == nullptr ||
        service_stream == nullptr || out == nullptr)
        return INC_FUSION_INVALID_ARGUMENT;
    *out = nullptr;
    const auto &layout = plan->value.remote_service;
    if (layout.ring_size < kFusionMinServiceRing ||
        layout.ring_size > kFusionMaxServiceRing)
        return INC_FUSION_INVALID_PLAN;
    auto *service = new (std::nothrow) inc_fusion_persistent_service_t;
    if (service == nullptr) return INC_FUSION_RUNTIME_ERROR;
    service->plan = plan;
    service->symmetric_base = static_cast<uint8_t *>(
        static_bindings->symmetric_base);
    service->service_stream = service_stream;
    service->ring_size = layout.ring_size;
    service->owns_device_storage = false;
    service->remote = true;
    service->device_control = reinterpret_cast<FusionServiceControl *>(
        service->symmetric_base + layout.control_off);
    service->device_descriptors =
        reinterpret_cast<FusionServiceDescriptor *>(
            service->symmetric_base + layout.descriptors_off);
    service->device_lane_progress =
        service->symmetric_base + layout.lane_progress_off;

    aclrtStream control_stream = nullptr;
    bool ok =
        aclrtMallocHost(reinterpret_cast<void **>(&service->host_stop),
                        sizeof(uint64_t)) == ACL_SUCCESS &&
        aclrtCreateStream(&control_stream) == ACL_SUCCESS;
    service->control_stream = control_stream;
    if (!ok) {
        FreeServiceStorage(service);
        delete service;
        return INC_FUSION_RUNTIME_ERROR;
    }
    *service->host_stop = 0u;
    const uint32_t ring = layout.ring_size;
    const size_t descriptor_bytes =
        static_cast<size_t>(ring) * sizeof(FusionServiceDescriptor);
    const size_t args_bytes =
        static_cast<size_t>(ring) * sizeof(FusionKernelArgs);
    const size_t request_bytes = static_cast<size_t>(ring) *
        layout.request_stride;
    const size_t lane_bytes =
        static_cast<size_t>(plan->value.resources.live_aiv) *
        inc::fusion::kFusionCacheLineBytes;
    const size_t worker_ready_bytes = static_cast<size_t>(ring) *
        plan->value.config.worker_count * inc::fusion::kFusionCacheLineBytes;
    auto *device_args = reinterpret_cast<FusionKernelArgs *>(
        service->symmetric_base + layout.args_off);
    auto *device_requests = service->symmetric_base + layout.request_off;
    auto *device_worker_pes =
        service->symmetric_base + layout.worker_pes_off;

    std::vector<FusionKernelArgs> host_args(ring);
    std::vector<FusionServiceDescriptor> descriptors(ring);
    for (uint32_t slot = 0u; slot < ring; ++slot) {
        auto *request = device_requests +
            static_cast<size_t>(slot) * layout.request_stride;
        inc_fusion_device_bindings_t bindings = *static_bindings;
        bindings.symmetric_base = service->symmetric_base;
        bindings.waves = request +
            (layout.waves_off - layout.request_off);
        bindings.active_token_counts = request +
            (layout.active_token_counts_off - layout.request_off);
        bindings.worker_pes = device_worker_pes;
        bindings.execution_flags |= INC_FUSION_EXEC_REMOTE_INC_SERVICE;
        if (inc_fusion_prepared_build_args(
                plan, INC_FUSION_ROLE_INC, 0u, &bindings,
                &host_args[slot], sizeof(FusionKernelArgs)) !=
            INC_FUSION_OK) {
            FreeServiceStorage(service);
            delete service;
            return INC_FUSION_INVALID_PLAN;
        }
        host_args[slot].remote_request = reinterpret_cast<uint64_t>(request);
        descriptors[slot].device_args = reinterpret_cast<uint64_t>(
            device_args + slot);
    }
    FusionServiceControl control{};
    control.ring_size = ring;
    control.live_aiv = plan->value.resources.live_aiv;
    control.descriptors = reinterpret_cast<uint64_t>(
        service->device_descriptors);
    control.lane_progress = reinterpret_cast<uint64_t>(
        service->device_lane_progress);
    if ((static_bindings->execution_flags &
         INC_FUSION_EXEC_INC_ROUTE_RELAY) != 0u) {
        inc_fusion_route_exchange_info_t exchange{};
        if (inc_fusion_prepared_route_exchange_info(plan, &exchange) !=
            INC_FUSION_OK) {
            FreeServiceStorage(service);
            delete service;
            return INC_FUSION_INVALID_PLAN;
        }
        const uint64_t count_words =
            static_cast<uint64_t>(plan->value.waves.size()) *
            (static_cast<uint64_t>(plan->value.config.expert_count) + 1u);
        if (count_words == 0u || count_words > UINT32_MAX) {
            FreeServiceStorage(service);
            delete service;
            return INC_FUSION_INVALID_PLAN;
        }
        // Reserved control fields are an optional service extension. They are
        // offsets except worker_pes, which is an immutable device pointer.
        control.reserved64[0] = exchange.counts_off;
        control.reserved64[1] = exchange.doorbells_off;
        control.reserved64[2] = count_words;
        control.reserved64[3] = plan->value.config.worker_count;
        control.reserved64[4] = reinterpret_cast<uint64_t>(device_worker_pes);
        control.reserved64[5] = static_bindings->ffts_addr;
    }
    if (aclrtMemset(service->device_control, sizeof(control),
                    0, sizeof(control)) != ACL_SUCCESS ||
        aclrtMemset(service->device_descriptors, descriptor_bytes,
                    0, descriptor_bytes) != ACL_SUCCESS ||
        aclrtMemset(device_args, args_bytes, 0, args_bytes) != ACL_SUCCESS ||
        aclrtMemset(device_requests, request_bytes, 0, request_bytes) !=
            ACL_SUCCESS ||
        aclrtMemset(service->symmetric_base + layout.worker_ready_off,
                    worker_ready_bytes, 0, worker_ready_bytes) !=
            ACL_SUCCESS ||
        aclrtMemset(service->device_lane_progress, lane_bytes,
                    0, lane_bytes) != ACL_SUCCESS ||
        aclrtMemcpy(device_worker_pes,
                    static_cast<size_t>(plan->value.config.worker_count) *
                        sizeof(uint32_t),
                    static_bindings->worker_pes,
                    static_cast<size_t>(plan->value.config.worker_count) *
                        sizeof(uint32_t),
                    ACL_MEMCPY_DEVICE_TO_DEVICE) != ACL_SUCCESS ||
        aclrtMemcpy(device_args, args_bytes, host_args.data(), args_bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        aclrtMemcpy(service->device_descriptors, descriptor_bytes,
                    descriptors.data(), descriptor_bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        aclrtMemcpy(service->device_control, sizeof(control),
                    &control, sizeof(control),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        FreeServiceStorage(service);
        delete service;
        return INC_FUSION_RUNTIME_ERROR;
    }
    *out = service;
    return INC_FUSION_OK;
}

extern "C" inc_fusion_status_t inc_fusion_remote_service_start(
    inc_fusion_persistent_service_t *service)
{
    if (service == nullptr || !service->remote ||
        service->service_stream == nullptr)
        return INC_FUSION_INVALID_ARGUMENT;
    if (service->running) return INC_FUSION_OK;
    launch_inc_fusion_inc_persistent_service_kernel(
        service->symmetric_base, service->device_control,
        static_cast<int>(service->plan->value.resources.live_aiv),
        service->service_stream);
    service->running = true;
    return INC_FUSION_OK;
}

extern "C" inc_fusion_status_t inc_fusion_persistent_service_submit(
    inc_fusion_persistent_service_t *service,
    void *device_args,
    uint64_t request_id,
    void *submit_stream,
    uint64_t *ticket)
{
    if (service == nullptr || device_args == nullptr ||
        submit_stream == nullptr || ticket == nullptr || !service->running)
        return INC_FUSION_INVALID_ARGUMENT;
    if (service->remote) return INC_FUSION_INVALID_ARGUMENT;
    if (service->next_ticket - service->completed_cached >
        service->ring_size) {
        const inc_fusion_status_t refresh = RefreshServiceCompleted(service);
        if (refresh != INC_FUSION_OK) return refresh;
        if (service->next_ticket - service->completed_cached >
            service->ring_size)
            return INC_FUSION_BUSY;
    }
    const uint64_t next = service->next_ticket;
    const uint32_t slot = static_cast<uint32_t>(
        (next - 1u) % service->ring_size);
    FusionServiceDescriptor *host = &service->host_staging[slot];
    std::memset(host, 0, sizeof(*host));
    host->ready = next;
    host->device_args = reinterpret_cast<uint64_t>(device_args);
    host->request_id = request_id;
    host->status = kFusionServiceRequestPending;
    uint8_t *device = reinterpret_cast<uint8_t *>(
        service->device_descriptors + slot);
    const size_t metadata_offset = offsetof(FusionServiceDescriptor, complete);
    if (aclrtMemcpyAsync(device + metadata_offset,
                         sizeof(FusionServiceDescriptor) - metadata_offset,
                         reinterpret_cast<uint8_t *>(host) + metadata_offset,
                         sizeof(FusionServiceDescriptor) - metadata_offset,
                         ACL_MEMCPY_HOST_TO_DEVICE,
                         static_cast<aclrtStream>(submit_stream)) !=
            ACL_SUCCESS ||
        aclrtMemcpyAsync(device, sizeof(uint64_t), host, sizeof(uint64_t),
                         ACL_MEMCPY_HOST_TO_DEVICE,
                         static_cast<aclrtStream>(submit_stream)) !=
            ACL_SUCCESS)
        return INC_FUSION_RUNTIME_ERROR;
    service->next_ticket = next + 1u;
    *ticket = next;
    return INC_FUSION_OK;
}

extern "C" inc_fusion_status_t inc_fusion_persistent_service_query(
    inc_fusion_persistent_service_t *service,
    uint64_t ticket,
    inc_fusion_service_result_t *result)
{
    if (service == nullptr || result == nullptr || ticket == 0u ||
        (!service->remote && ticket >= service->next_ticket))
        return INC_FUSION_INVALID_ARGUMENT;
    if (service->remote && ticket >= service->next_ticket)
        service->next_ticket = ticket + 1u;
    std::memset(result, 0, sizeof(*result));
    result->ticket = ticket;
    const inc_fusion_status_t refresh = RefreshServiceCompleted(service);
    if (refresh != INC_FUSION_OK) return refresh;
    if (service->completed_cached < ticket) return INC_FUSION_OK;
    const uint32_t slot = static_cast<uint32_t>(
        (ticket - 1u) % service->ring_size);
    FusionServiceDescriptor descriptor{};
    if (aclrtMemcpy(&descriptor, sizeof(descriptor),
                    service->device_descriptors + slot,
                    sizeof(descriptor), ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS)
        return INC_FUSION_RUNTIME_ERROR;
    // A newer wrapped ticket may already own this slot. Completed requests are
    // in order, so callers must query before submitting >ring_size successors.
    if (descriptor.complete != ticket)
        return INC_FUSION_BUSY;
    result->request_id = descriptor.request_id;
    result->complete = 1u;
    result->status = descriptor.status;
    result->operator_error = descriptor.operator_error;
    return INC_FUSION_OK;
}

extern "C" inc_fusion_status_t inc_fusion_persistent_service_stop(
    inc_fusion_persistent_service_t *service)
{
    if (service == nullptr) return INC_FUSION_INVALID_ARGUMENT;
    if (!service->running) return INC_FUSION_OK;
    *service->host_stop = 1u;
    auto *destination = reinterpret_cast<uint8_t *>(
        service->device_control) + offsetof(FusionServiceControl, stop);
    if (aclrtMemcpyAsync(destination, sizeof(uint64_t), service->host_stop,
                         sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE,
                         static_cast<aclrtStream>(service->control_stream)) !=
            ACL_SUCCESS ||
        aclrtSynchronizeStream(
            static_cast<aclrtStream>(service->control_stream)) != ACL_SUCCESS ||
        aclrtSynchronizeStream(
            static_cast<aclrtStream>(service->service_stream)) != ACL_SUCCESS)
        return INC_FUSION_RUNTIME_ERROR;
    service->running = false;
    return INC_FUSION_OK;
}

extern "C" void inc_fusion_persistent_service_destroy(
    inc_fusion_persistent_service_t *service)
{
    if (service == nullptr) return;
    if (service->running &&
        inc_fusion_persistent_service_stop(service) != INC_FUSION_OK)
        return;
    FreeServiceStorage(service);
    delete service;
}

extern "C" void *inc_fusion_persistent_service_device_control(
    const inc_fusion_persistent_service_t *service)
{
    return service == nullptr ? nullptr : service->device_control;
}
