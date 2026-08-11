#include <torch/library.h>

#include <acl/acl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"

#include "inc_fusion_api.h"
#include "inc_fusion_plan.h"

extern "C" void launch_inc_fusion_route_exchange_kernel(
    uint8_t *sym, const uint32_t *local_counts, uint32_t *global_counts,
    const uint32_t *worker_pes, uint64_t ffts_addr, uint64_t counts_off,
    uint64_t doorbells_off, uint32_t count_words, uint32_t worker_count,
    uint32_t rank, int32_t relay_pe, uint64_t generation, void *stream);

namespace {

constexpr int64_t kSerialShmemMode = 1;
constexpr int64_t kSerialIncMode = 2;
constexpr int64_t kFusedShmemMode = 3;
constexpr int64_t kFusedIncMode = 4;

bool IsIncMode(int64_t mode)
{
    return mode == kSerialIncMode || mode == kFusedIncMode;
}

bool IsShmemMode(int64_t mode)
{
    return mode == kSerialShmemMode || mode == kFusedShmemMode;
}

uint32_t CheckedU32(int64_t value, const char *name,
                    bool allow_zero = false)
{
    TORCH_CHECK((allow_zero ? value >= 0 : value > 0) &&
                    static_cast<uint64_t>(value) <= UINT32_MAX,
                name, " is outside uint32 range: ", value);
    return static_cast<uint32_t>(value);
}

void CheckNpuContiguous(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                name, " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckSameDevice(const at::Tensor &tensor, c10::DeviceIndex device,
                     const char *name)
{
    CheckNpuContiguous(tensor, name);
    TORCH_CHECK(tensor.device().index() == device,
                name, " is on a different NPU");
}

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

struct WorkerState {
    inc_fusion_prepared_plan_t *plan = nullptr;
    inc_fusion_worker_executor_t *executor = nullptr;
    at::Tensor expert_owner;
    at::Tensor expert_local_index;
    at::Tensor worker_pes;
    at::Tensor active_token_counts;
    c10::DeviceIndex device = -1;
    uint32_t worker_count = 0;
    uint32_t rank = 0;
    uint32_t inc_pe = 0;
    uint32_t hidden = 0;
    uint32_t intermediate = 0;
    uint32_t expert_count = 0;
    uint32_t local_expert_count = 0;
    uint32_t topk = 0;
    uint32_t token_capacity = 0;
    uint32_t wave_count = 0;
    int64_t backend_mode = 0;
    uint64_t next_generation = 1;
    uint64_t next_route_generation = 1;
    uint8_t *symmetric_base = nullptr;
    uint64_t ffts_addr = 0;
    inc_fusion_route_exchange_info_t route_exchange{};
    aclrtStream bound_stream = nullptr;
    std::mutex enqueue_mutex;

    ~WorkerState()
    {
        if (executor != nullptr)
            inc_fusion_worker_executor_destroy(executor);
        if (plan != nullptr)
            inc_fusion_prepared_plan_destroy(plan);
    }
};

// vLLM's compile/warm-up path can execute the same out-of-graph custom op on
// a profiling stream and later on its steady-state stream.  A prepared
// executor may move between those streams, but only after every command on
// the previous stream has completed.  This keeps the host-owned generation
// and event ring single-producer without imposing a steady-state sync when
// the stream does not change.
void BindIdleWorkerStream(WorkerState *state, aclrtStream stream)
{
    if (state->bound_stream == nullptr) {
        state->bound_stream = stream;
        return;
    }
    if (state->bound_stream == stream) return;
    TORCH_CHECK(aclrtSynchronizeStream(state->bound_stream) == ACL_SUCCESS,
                "failed to drain the previous single-INC worker stream "
                "before a serial stream handoff");
    state->bound_stream = stream;
}

struct ServiceState {
    inc_fusion_prepared_plan_t *plan = nullptr;
    inc_fusion_persistent_service_t *service = nullptr;
    void *workspace = nullptr;
    aclrtStream stream = nullptr;
    at::Tensor worker_pes;
    c10::DeviceIndex device = -1;
    bool started = false;
    std::mutex mutex;

    ~ServiceState()
    {
        if (service != nullptr)
            inc_fusion_persistent_service_destroy(service);
        if (stream != nullptr)
            (void)aclrtDestroyStream(stream);
        if (workspace != nullptr)
            (void)aclrtFree(workspace);
        if (plan != nullptr)
            inc_fusion_prepared_plan_destroy(plan);
    }
};

std::mutex g_registry_mutex;
std::unordered_map<int64_t, std::shared_ptr<WorkerState>> g_registry;
int64_t g_next_handle = 1;
std::mutex g_service_registry_mutex;
std::unordered_map<int64_t, std::shared_ptr<ServiceState>> g_service_registry;
int64_t g_next_service_handle = 1;

std::shared_ptr<WorkerState> Lookup(int64_t handle)
{
    std::lock_guard<std::mutex> guard(g_registry_mutex);
    const auto found = g_registry.find(handle);
    TORCH_CHECK(found != g_registry.end(),
                "single-INC worker handle is closed or invalid: ", handle);
    return found->second;
}

std::shared_ptr<ServiceState> LookupService(int64_t handle)
{
    std::lock_guard<std::mutex> guard(g_service_registry_mutex);
    const auto found = g_service_registry.find(handle);
    TORCH_CHECK(found != g_service_registry.end(),
                "single-INC service handle is closed or invalid: ", handle);
    return found->second;
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> CopyPlacementToHost(
    const at::Tensor &owner, const at::Tensor &local)
{
    std::vector<uint32_t> host_owner(static_cast<size_t>(owner.numel()));
    std::vector<uint32_t> host_local(static_cast<size_t>(local.numel()));
    const size_t bytes = host_owner.size() * sizeof(uint32_t);
    aclrtStream stream = c10_npu::getCurrentNPUStream(
        owner.device().index()).stream();
    TORCH_CHECK(aclrtMemcpyAsync(
                    host_owner.data(), bytes, owner.const_data_ptr<int32_t>(),
                    bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS,
                "failed to copy expert_owner during setup");
    TORCH_CHECK(aclrtMemcpyAsync(
                    host_local.data(), bytes, local.const_data_ptr<int32_t>(),
                    bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS,
                "failed to copy expert_local_index during setup");
    TORCH_CHECK(aclrtSynchronizeStream(stream) == ACL_SUCCESS,
                "failed to synchronize expert placement setup copy");
    return {std::move(host_owner), std::move(host_local)};
}

std::vector<int64_t> PlanInfo(
    const at::Tensor &expert_owner, const at::Tensor &expert_local_index,
    int64_t live_aiv, int64_t live_aic, int64_t worker_count, int64_t rank,
    int64_t inc_pe, int64_t hidden, int64_t intermediate,
    int64_t expert_count, int64_t topk, int64_t token_capacity,
    int64_t tokens_per_wave, int64_t slot_count,
    int64_t service_ring_size, int64_t activation_waves, int64_t spin_cap)
{
    TORCH_CHECK(expert_owner.device().is_cpu() &&
                    expert_local_index.device().is_cpu() &&
                    expert_owner.is_contiguous() &&
                    expert_local_index.is_contiguous() &&
                    expert_owner.scalar_type() == at::ScalarType::Int &&
                    expert_local_index.scalar_type() == at::ScalarType::Int &&
                    expert_owner.dim() == 1 &&
                    expert_local_index.sizes() == expert_owner.sizes() &&
                    expert_owner.numel() == expert_count,
                "plan_info placement must be matching contiguous CPU int32");
    inc_fusion_plan_desc_t desc{};
    inc_fusion_plan_desc_init(&desc);
    desc.live_aiv = CheckedU32(live_aiv, "live_aiv");
    desc.live_aic = CheckedU32(live_aic, "live_aic");
    desc.worker_count = CheckedU32(worker_count, "worker_count");
    desc.rank = CheckedU32(rank, "rank", true);
    desc.inc_pe = CheckedU32(inc_pe, "inc_pe", true);
    desc.hidden = CheckedU32(hidden, "hidden");
    desc.intermediate = CheckedU32(intermediate, "intermediate");
    desc.expert_count = CheckedU32(expert_count, "expert_count");
    desc.topk = CheckedU32(topk, "topk");
    desc.token_count = CheckedU32(token_capacity, "token_capacity");
    desc.tokens_per_wave = CheckedU32(tokens_per_wave, "tokens_per_wave");
    desc.slot_count = CheckedU32(slot_count, "slot_count");
    desc.service_ring_size = CheckedU32(service_ring_size,
                                        "service_ring_size");
    desc.activation_waves = CheckedU32(activation_waves,
                                       "activation_waves");
    desc.spin_cap = CheckedU32(spin_cap, "spin_cap", true);
    inc_fusion_prepared_plan_t *plan = nullptr;
    const inc_fusion_status_t created = inc_fusion_prepared_plan_create(
        &desc,
        reinterpret_cast<const uint32_t *>(
            expert_owner.const_data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t *>(
            expert_local_index.const_data_ptr<int32_t>()), &plan);
    TORCH_CHECK(created == INC_FUSION_OK,
                "plan_info creation failed: ", StatusName(created));
    inc_fusion_plan_info_t info{};
    const inc_fusion_status_t queried = inc_fusion_prepared_plan_info(
        plan, &info);
    inc_fusion_prepared_plan_destroy(plan);
    TORCH_CHECK(queried == INC_FUSION_OK,
                "plan_info query failed: ", StatusName(queried));
    const uint64_t values[] = {
        info.symmetric_bytes, info.worker_workspace_bytes,
        info.inc_workspace_bytes, info.wave_count, info.local_expert_count,
        info.fusion_abi_version, info.remote_service_bytes,
    };
    std::vector<int64_t> result;
    result.reserve(sizeof(values) / sizeof(values[0]));
    for (uint64_t value : values) {
        TORCH_CHECK(value <= static_cast<uint64_t>(INT64_MAX),
                    "plan_info value exceeds int64");
        result.push_back(static_cast<int64_t>(value));
    }
    return result;
}

int64_t WorkerPrepare(
    const at::Tensor &expert_owner, const at::Tensor &expert_local_index,
    const at::Tensor &worker_pes, const at::Tensor &active_token_counts,
    int64_t symmetric_base, int64_t symmetric_bytes, int64_t ffts_addr,
    int64_t live_aiv, int64_t live_aic, int64_t worker_count, int64_t rank,
    int64_t inc_pe, int64_t hidden, int64_t intermediate,
    int64_t expert_count, int64_t topk, int64_t token_capacity,
    int64_t tokens_per_wave, int64_t slot_count,
    int64_t service_ring_size, int64_t activation_waves, int64_t spin_cap,
    int64_t executor_ring_size, int64_t backend_mode)
{
    CheckNpuContiguous(expert_owner, "expert_owner");
    const c10::DeviceIndex device = expert_owner.device().index();
    CheckSameDevice(expert_local_index, device, "expert_local_index");
    CheckSameDevice(worker_pes, device, "worker_pes");
    CheckSameDevice(active_token_counts, device, "active_token_counts");
    TORCH_CHECK(expert_owner.scalar_type() == at::ScalarType::Int &&
                    expert_owner.dim() == 1,
                "expert_owner must be a contiguous int32 vector");
    TORCH_CHECK(expert_local_index.scalar_type() == at::ScalarType::Int &&
                    expert_local_index.sizes() == expert_owner.sizes(),
                "expert_local_index must match expert_owner");
    TORCH_CHECK(worker_pes.scalar_type() == at::ScalarType::Int &&
                    worker_pes.dim() == 1 &&
                    worker_pes.numel() == worker_count,
                "worker_pes must be int32 [worker_count]");
    TORCH_CHECK(active_token_counts.scalar_type() == at::ScalarType::Int &&
                    active_token_counts.dim() == 1 &&
                    active_token_counts.numel() == worker_count,
                "active_token_counts must be int32 [worker_count]");
    TORCH_CHECK(expert_owner.numel() == expert_count,
                "expert placement length disagrees with expert_count");
    TORCH_CHECK(symmetric_base > 0 && symmetric_bytes > 0,
                "symmetric allocation address/size must be positive");
    TORCH_CHECK(ffts_addr >= 0, "ffts_addr must be non-negative");
    TORCH_CHECK(IsIncMode(backend_mode) || IsShmemMode(backend_mode),
                "backend_mode must be one of serial/fused SHMEM/INC (1..4)");

    auto placement = CopyPlacementToHost(expert_owner, expert_local_index);
    const uint32_t workers = CheckedU32(worker_count, "worker_count");
    const uint32_t worker_rank = CheckedU32(rank, "rank", true);
    TORCH_CHECK(worker_rank < workers, "rank is outside worker group");
    uint32_t local_experts = 0;
    for (size_t expert = 0; expert < placement.first.size(); ++expert) {
        TORCH_CHECK(placement.first[expert] < workers,
                    "expert_owner contains an invalid worker at expert ",
                    expert);
        if (placement.first[expert] == worker_rank)
            ++local_experts;
    }
    TORCH_CHECK(local_experts > 0,
                "every compute worker must own at least one expert");

    inc_fusion_plan_desc_t desc{};
    inc_fusion_plan_desc_init(&desc);
    desc.live_aiv = CheckedU32(live_aiv, "live_aiv");
    desc.live_aic = CheckedU32(live_aic, "live_aic");
    desc.worker_count = workers;
    desc.rank = worker_rank;
    desc.inc_pe = CheckedU32(inc_pe, "inc_pe", true);
    desc.hidden = CheckedU32(hidden, "hidden");
    desc.intermediate = CheckedU32(intermediate, "intermediate");
    desc.expert_count = CheckedU32(expert_count, "expert_count");
    desc.topk = CheckedU32(topk, "topk");
    desc.token_count = CheckedU32(token_capacity, "token_capacity");
    desc.tokens_per_wave = CheckedU32(tokens_per_wave, "tokens_per_wave");
    desc.slot_count = CheckedU32(slot_count, "slot_count");
    desc.service_ring_size = CheckedU32(service_ring_size,
                                        "service_ring_size");
    desc.activation_waves = CheckedU32(activation_waves,
                                       "activation_waves");
    desc.spin_cap = CheckedU32(spin_cap, "spin_cap", true);

    auto state = std::make_shared<WorkerState>();
    const inc_fusion_status_t created = inc_fusion_prepared_plan_create(
        &desc, placement.first.data(), placement.second.data(), &state->plan);
    TORCH_CHECK(created == INC_FUSION_OK,
                "inc_fusion_prepared_plan_create failed: ",
                StatusName(created));
    inc_fusion_plan_info_t info{};
    const inc_fusion_status_t queried = inc_fusion_prepared_plan_info(
        state->plan, &info);
    TORCH_CHECK(queried == INC_FUSION_OK,
                "inc_fusion_prepared_plan_info failed: ", StatusName(queried));
    TORCH_CHECK(static_cast<uint64_t>(symmetric_bytes) >= info.symmetric_bytes,
                "symmetric allocation is too small: need ",
                info.symmetric_bytes, ", got ", symmetric_bytes);
    const inc_fusion_status_t exchange_queried =
        inc_fusion_prepared_route_exchange_info(
            state->plan, &state->route_exchange);
    TORCH_CHECK(exchange_queried == INC_FUSION_OK,
                "route exchange layout query failed: ",
                StatusName(exchange_queried));

    inc_fusion_device_bindings_t bindings{};
    bindings.symmetric_base = reinterpret_cast<void *>(
        static_cast<uintptr_t>(symmetric_base));
    bindings.expert_owner = const_cast<void *>(expert_owner.const_data_ptr());
    bindings.expert_local_index = const_cast<void *>(
        expert_local_index.const_data_ptr());
    bindings.worker_pes = const_cast<void *>(worker_pes.const_data_ptr());
    bindings.active_token_counts = const_cast<void *>(
        active_token_counts.const_data_ptr());
    bindings.ffts_addr = static_cast<uint64_t>(ffts_addr);
    bindings.execution_flags = IsIncMode(backend_mode)
        ? INC_FUSION_EXEC_REMOTE_INC_SERVICE
        : INC_FUSION_EXEC_WORKER_DIRECT_SHMEM;
    if (IsIncMode(backend_mode) && desc.topk >= 2u)
        bindings.execution_flags |= INC_FUSION_EXEC_GLOBAL_OUTPUT_FANOUT;
    const inc_fusion_status_t executor_created =
        inc_fusion_worker_executor_create(
            state->plan, &bindings,
            CheckedU32(executor_ring_size, "executor_ring_size"),
            &state->executor);
    TORCH_CHECK(executor_created == INC_FUSION_OK,
                "inc_fusion_worker_executor_create failed: ",
                StatusName(executor_created));

    state->expert_owner = expert_owner;
    state->expert_local_index = expert_local_index;
    state->worker_pes = worker_pes;
    state->active_token_counts = active_token_counts;
    state->device = device;
    state->worker_count = workers;
    state->rank = worker_rank;
    state->inc_pe = desc.inc_pe;
    state->hidden = desc.hidden;
    state->intermediate = desc.intermediate;
    state->expert_count = desc.expert_count;
    state->local_expert_count = local_experts;
    state->topk = desc.topk;
    state->token_capacity = desc.token_count;
    state->wave_count = info.wave_count;
    state->backend_mode = backend_mode;
    state->symmetric_base = reinterpret_cast<uint8_t *>(
        static_cast<uintptr_t>(symmetric_base));
    state->ffts_addr = static_cast<uint64_t>(ffts_addr);

    std::lock_guard<std::mutex> guard(g_registry_mutex);
    TORCH_CHECK(g_next_handle < std::numeric_limits<int64_t>::max(),
                "single-INC worker handle space exhausted");
    const int64_t handle = g_next_handle++;
    g_registry.emplace(handle, std::move(state));
    return handle;
}

void WorkerDestroy(const at::Tensor &device_anchor, int64_t handle)
{
    CheckNpuContiguous(device_anchor, "device_anchor");
    std::shared_ptr<WorkerState> removed;
    {
        std::lock_guard<std::mutex> guard(g_registry_mutex);
        const auto found = g_registry.find(handle);
        if (found == g_registry.end()) return;
        TORCH_CHECK(device_anchor.device().index() == found->second->device,
                    "device_anchor is on a different NPU");
        removed = std::move(found->second);
        g_registry.erase(found);
    }
    // Destruction is intentionally outside the registry lock and waits only
    // for this executor's event ring.
    removed.reset();
}

std::vector<int64_t> WorkerDebugSlots(
    const at::Tensor &device_anchor, int64_t handle)
{
    const std::shared_ptr<WorkerState> state = Lookup(handle);
    CheckSameDevice(device_anchor, state->device, "device_anchor");
    std::lock_guard<std::mutex> guard(state->enqueue_mutex);
    TORCH_CHECK(state->bound_stream != nullptr,
                "worker executor has not submitted a request");
    TORCH_CHECK(aclrtSynchronizeStream(state->bound_stream) == ACL_SUCCESS,
                "failed to synchronize worker stream for debug state");
    inc_fusion_worker_executor_info_t info{};
    TORCH_CHECK(inc_fusion_worker_executor_info(state->executor, &info) ==
                    INC_FUSION_OK,
                "failed to query worker executor debug state");
    TORCH_CHECK(info.submitted != 0u && info.host_args != nullptr,
                "worker executor has no submitted argument record");
    const uint32_t record = static_cast<uint32_t>(
        (info.submitted - 1u) % info.ring_size);
    const auto *args = reinterpret_cast<const inc::fusion::FusionKernelArgs *>(
        info.host_args) + record;
    std::vector<inc::fusion::FusionSlotState> slots(args->slot_count);
    const auto *source = reinterpret_cast<const uint8_t *>(info.workspace) +
        args->layout.slot_state_off;
    TORCH_CHECK(aclrtMemcpy(slots.data(),
                    slots.size() * sizeof(inc::fusion::FusionSlotState), source,
                    slots.size() * sizeof(inc::fusion::FusionSlotState),
                    ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
                "failed to copy worker slot debug state");
    std::vector<int64_t> result;
    result.reserve(slots.size() * 7u);
    for (const auto &slot : slots) {
        result.push_back(slot.error);
        result.push_back(slot.owner);
        result.push_back(static_cast<int64_t>(slot.dispatch_generation));
        result.push_back(static_cast<int64_t>(slot.combine_generation));
        result.push_back(static_cast<int64_t>(slot.output_generation));
        result.push_back(static_cast<int64_t>(slot.release_generation));
        result.push_back(static_cast<int64_t>(slot.reserved));
    }
    return result;
}

std::vector<int64_t> WorkerDebugTraces(
    const at::Tensor &device_anchor, int64_t handle)
{
    const std::shared_ptr<WorkerState> state = Lookup(handle);
    CheckSameDevice(device_anchor, state->device, "device_anchor");
    std::lock_guard<std::mutex> guard(state->enqueue_mutex);
    TORCH_CHECK(state->bound_stream != nullptr,
                "worker executor has not submitted a request");
    TORCH_CHECK(aclrtSynchronizeStream(state->bound_stream) == ACL_SUCCESS,
                "failed to synchronize worker stream for debug trace");
    inc_fusion_worker_executor_info_t info{};
    TORCH_CHECK(inc_fusion_worker_executor_info(state->executor, &info) ==
                    INC_FUSION_OK,
                "failed to query worker executor debug trace");
    TORCH_CHECK(info.submitted != 0u && info.host_args != nullptr,
                "worker executor has no submitted argument record");
    const uint32_t record = static_cast<uint32_t>(
        (info.submitted - 1u) % info.ring_size);
    const auto *args = reinterpret_cast<const inc::fusion::FusionKernelArgs *>(
        info.host_args) + record;
    constexpr uint32_t kWorkerTraceLines = 4u;
    std::vector<inc::fusion::FusionLaneTrace> traces(kWorkerTraceLines);
    const auto *source = reinterpret_cast<const uint8_t *>(info.workspace) +
        args->layout.trace_off;
    TORCH_CHECK(aclrtMemcpy(traces.data(),
                    traces.size() * sizeof(inc::fusion::FusionLaneTrace),
                    source,
                    traces.size() * sizeof(inc::fusion::FusionLaneTrace),
                    ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
                "failed to copy worker debug trace");
    std::vector<int64_t> result;
    result.reserve(traces.size() * 9u);
    for (const auto &trace : traces) {
        result.push_back(static_cast<int64_t>(trace.role));
        result.push_back(static_cast<int64_t>(trace.lane));
        result.push_back(static_cast<int64_t>(trace.start_cycle));
        result.push_back(static_cast<int64_t>(trace.end_cycle));
        for (uint32_t checkpoint = 0u; checkpoint < 5u; ++checkpoint)
            result.push_back(static_cast<int64_t>(
                trace.reserved64[checkpoint]));
    }
    return result;
}

void WorkerRouteExchangeOut(
    int64_t handle, const at::Tensor &local_counts,
    at::Tensor &global_counts)
{
    const std::shared_ptr<WorkerState> state = Lookup(handle);
    CheckSameDevice(local_counts, state->device, "local_counts");
    CheckSameDevice(global_counts, state->device, "global_counts");
    TORCH_CHECK(local_counts.scalar_type() == at::ScalarType::Int &&
                    local_counts.dim() == 2 &&
                    local_counts.size(1) == state->expert_count + 1u,
                "local_counts must be int32 [waves,expert_count+1]");
    TORCH_CHECK(global_counts.scalar_type() == at::ScalarType::Int &&
                    global_counts.dim() == 3 &&
                    global_counts.size(0) == state->worker_count &&
                    global_counts.size(1) == local_counts.size(0) &&
                    global_counts.size(2) == local_counts.size(1),
                "global_counts must be int32 [workers,waves,experts+1]");
    TORCH_CHECK(local_counts.numel() > 0 &&
                    static_cast<uint64_t>(local_counts.numel()) <= UINT32_MAX,
                "route count exchange is outside uint32 word range");
    std::lock_guard<std::mutex> guard(state->enqueue_mutex);
    aclrtStream stream = c10_npu::getCurrentNPUStream(state->device).stream();
    BindIdleWorkerStream(state.get(), stream);
    TORCH_CHECK(state->next_route_generation < UINT64_MAX,
                "route exchange generation exhausted; restart the engine");
    launch_inc_fusion_route_exchange_kernel(
        state->symmetric_base,
        reinterpret_cast<const uint32_t *>(
            local_counts.const_data_ptr<int32_t>()),
        reinterpret_cast<uint32_t *>(
            global_counts.mutable_data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t *>(
            state->worker_pes.const_data_ptr<int32_t>()), state->ffts_addr,
        state->route_exchange.counts_off,
        state->route_exchange.doorbells_off,
        static_cast<uint32_t>(local_counts.numel()), state->worker_count,
        state->rank,
        IsIncMode(state->backend_mode)
            ? static_cast<int32_t>(state->inc_pe) : -1,
        state->next_route_generation++, stream);
}

void WorkerMoeOut(
    int64_t handle, const at::Tensor &input, const at::Tensor &w13,
    const at::Tensor &w2, const at::Tensor &dispatch_rows,
    const at::Tensor &assignments, const at::Tensor &waves,
    const at::Tensor &active_token_counts, const at::Tensor &group_lists,
    at::Tensor &output, int64_t mode)
{
    const std::shared_ptr<WorkerState> state = Lookup(handle);
    // Capturing the host-built generation/ticket would replay an old command
    // record.  torch.compile eager custom-op execution is supported; NPU graph
    // capture/replay is deliberately rejected until generation becomes a
    // graph-replay-aware device counter.  Use the exported query directly:
    // torch-npu 2.9 declares IsContextInitialized() in its public header but
    // does not export that symbol for third-party extensions, which makes the
    // otherwise convenient inline assertNotCapturing() unloadable.
    const auto capture_status =
        c10_npu::currentStreamCaptureStatusMayInitCtx();
    TORCH_CHECK(capture_status == c10_npu::CaptureStatus::None,
                "single-INC worker_moe_out is not supported during NPU "
                "graph capture");
    const std::pair<const at::Tensor *, const char *> tensors[] = {
        {&input, "input"}, {&w13, "w13"}, {&w2, "w2"},
        {&dispatch_rows, "dispatch_rows"}, {&assignments, "assignments"},
        {&waves, "waves"}, {&active_token_counts, "active_token_counts"},
        {&group_lists, "group_lists"}, {&output, "output"},
    };
    for (const auto &entry : tensors)
        CheckSameDevice(*entry.first, state->device, entry.second);
    TORCH_CHECK(mode == state->backend_mode,
                "worker_moe_out mode disagrees with prepared executor: ",
                mode, " != ", state->backend_mode);
    TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16 &&
                    input.dim() == 2 && input.size(1) == state->hidden &&
                    input.size(0) <= state->token_capacity,
                "input must be BF16 [tokens<=capacity, hidden]");
    const bool global_output = IsIncMode(state->backend_mode) &&
        state->topk >= 2u;
    const int64_t minimum_output_rows = input.size(0);
    const int64_t maximum_output_rows = static_cast<int64_t>(
        state->token_capacity) * (global_output ? state->worker_count : 1u);
    TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16 &&
                    output.dim() == 2 &&
                    output.size(0) >= minimum_output_rows &&
                    output.size(0) <= maximum_output_rows &&
                    (!global_output ||
                     output.size(0) >= input.size(0)) &&
                    output.size(1) == state->hidden,
                "output must be BF16 [rows,", state->hidden,
                "] with rows in [", minimum_output_rows, ",",
                maximum_output_rows, "]");
    const bool fusion_layout =
        w13.dim() == 3 && w2.dim() == 3 &&
        w13.size(0) == state->local_expert_count &&
        w2.size(0) == state->local_expert_count &&
        w13.size(1) == 2 * state->intermediate &&
        w13.size(2) == state->hidden &&
        w2.size(1) == state->hidden &&
        w2.size(2) == state->intermediate;
    const bool row_major_b_layout =
        w13.dim() == 3 && w2.dim() == 3 &&
        w13.size(0) == state->local_expert_count &&
        w2.size(0) == state->local_expert_count &&
        w13.size(1) == state->hidden &&
        w13.size(2) == 2 * state->intermediate &&
        w2.size(1) == state->intermediate &&
        w2.size(2) == state->hidden;
    TORCH_CHECK(w13.scalar_type() == at::ScalarType::BFloat16 &&
                    w2.scalar_type() == at::ScalarType::BFloat16 &&
                    w13.is_contiguous() && w2.is_contiguous() &&
                    (fusion_layout || row_major_b_layout),
                "weights must be contiguous BF16 [E,2I,H]/[E,H,I] or "
                "[E,H,2I]/[E,I,H]");
    TORCH_CHECK(dispatch_rows.scalar_type() == at::ScalarType::Byte &&
                    static_cast<uint64_t>(dispatch_rows.numel()) >=
                        static_cast<uint64_t>(state->token_capacity) *
                        std::min(state->topk, state->worker_count) * 32u,
                "dispatch_rows protocol buffer is too small");
    TORCH_CHECK(assignments.scalar_type() == at::ScalarType::Byte &&
                    static_cast<uint64_t>(assignments.numel()) >=
                        static_cast<uint64_t>(state->token_capacity) *
                        state->topk * 32u,
                "assignments protocol buffer is too small");
    TORCH_CHECK(waves.scalar_type() == at::ScalarType::Byte &&
                    static_cast<uint64_t>(waves.numel()) >=
                        static_cast<uint64_t>(state->wave_count) * 64u,
                "waves protocol buffer is too small");
    TORCH_CHECK(active_token_counts.scalar_type() == at::ScalarType::Int &&
                    active_token_counts.numel() == state->worker_count,
                "active_token_counts must be int32 [worker_count]");
    TORCH_CHECK(group_lists.scalar_type() == at::ScalarType::Long &&
                    group_lists.dim() == 2 &&
                    group_lists.size(0) >= state->wave_count &&
                    group_lists.size(1) == state->local_expert_count,
                "group_lists must be int64 [waves,local_experts]");

    std::lock_guard<std::mutex> guard(state->enqueue_mutex);
    aclrtStream stream = c10_npu::getCurrentNPUStream(state->device).stream();
    BindIdleWorkerStream(state.get(), stream);
    const uint64_t max_generation =
        static_cast<uint64_t>(UINT32_MAX) - state->wave_count;
    TORCH_CHECK(state->next_generation <= max_generation,
                "fusion generation space exhausted; perform an orderly "
                "engine/service restart before generation wrap");
    inc_fusion_device_bindings_t dynamic{};
    // Empty EP ranks are legal.  Their persistent output allocation supplies
    // a non-null dummy address while active_token_counts prevents any access.
    dynamic.input = input.numel() == 0
        ? output.mutable_data_ptr() : const_cast<void *>(input.const_data_ptr());
    dynamic.output = output.mutable_data_ptr();
    dynamic.w13 = const_cast<void *>(w13.const_data_ptr());
    dynamic.w2 = const_cast<void *>(w2.const_data_ptr());
    dynamic.dispatch_rows = const_cast<void *>(dispatch_rows.const_data_ptr());
    dynamic.assignments = const_cast<void *>(assignments.const_data_ptr());
    dynamic.waves = const_cast<void *>(waves.const_data_ptr());
    dynamic.active_token_counts = const_cast<void *>(
        active_token_counts.const_data_ptr());
    dynamic.group_lists = const_cast<void *>(group_lists.const_data_ptr());
    if (mode == kSerialShmemMode || mode == kSerialIncMode)
        dynamic.execution_flags = INC_FUSION_EXEC_STRICT_SERIAL_PIPELINE;
    if (row_major_b_layout)
        dynamic.execution_flags |= INC_FUSION_EXEC_WEIGHT_B_ROW_MAJOR;
    const inc_fusion_status_t status = inc_fusion_worker_executor_enqueue(
        state->executor, state->next_generation, &dynamic, stream);
    TORCH_CHECK(status == INC_FUSION_OK,
                "inc_fusion_worker_executor_enqueue failed: ",
                StatusName(status),
                status == INC_FUSION_BUSY
                    ? "; executor ring is full—raise executor_ring_size or "
                      "bound asynchronous in-flight forwards"
                    : "");
    state->next_generation += static_cast<uint64_t>(state->wave_count) + 1u;
}

int64_t ServicePrepare(
    const at::Tensor &expert_owner, const at::Tensor &expert_local_index,
    const at::Tensor &worker_pes, int64_t symmetric_base,
    int64_t symmetric_bytes, int64_t ffts_addr, int64_t live_aiv,
    int64_t live_aic, int64_t worker_count, int64_t inc_pe, int64_t hidden,
    int64_t intermediate, int64_t expert_count, int64_t topk,
    int64_t token_capacity, int64_t tokens_per_wave, int64_t slot_count,
    int64_t service_ring_size, int64_t activation_waves, int64_t spin_cap,
    int64_t inc_route_relay)
{
    CheckNpuContiguous(expert_owner, "expert_owner");
    const c10::DeviceIndex device = expert_owner.device().index();
    CheckSameDevice(expert_local_index, device, "expert_local_index");
    CheckSameDevice(worker_pes, device, "worker_pes");
    TORCH_CHECK(expert_owner.scalar_type() == at::ScalarType::Int &&
                    expert_owner.dim() == 1 &&
                    expert_owner.numel() == expert_count,
                "expert_owner must be int32 [expert_count]");
    TORCH_CHECK(expert_local_index.scalar_type() == at::ScalarType::Int &&
                    expert_local_index.sizes() == expert_owner.sizes(),
                "expert_local_index must match expert_owner");
    TORCH_CHECK(worker_pes.scalar_type() == at::ScalarType::Int &&
                    worker_pes.dim() == 1 &&
                    worker_pes.numel() == worker_count,
                "worker_pes must be int32 [worker_count]");
    TORCH_CHECK(symmetric_base > 0 && symmetric_bytes > 0 && ffts_addr >= 0,
                "invalid symmetric allocation or FFTS address");
    auto placement = CopyPlacementToHost(expert_owner, expert_local_index);

    inc_fusion_plan_desc_t desc{};
    inc_fusion_plan_desc_init(&desc);
    desc.live_aiv = CheckedU32(live_aiv, "live_aiv");
    desc.live_aic = CheckedU32(live_aic, "live_aic");
    desc.worker_count = CheckedU32(worker_count, "worker_count");
    desc.rank = 0u; // INC owns no experts; rank is plan-validation metadata.
    desc.inc_pe = CheckedU32(inc_pe, "inc_pe", true);
    desc.hidden = CheckedU32(hidden, "hidden");
    desc.intermediate = CheckedU32(intermediate, "intermediate");
    desc.expert_count = CheckedU32(expert_count, "expert_count");
    desc.topk = CheckedU32(topk, "topk");
    desc.token_count = CheckedU32(token_capacity, "token_capacity");
    desc.tokens_per_wave = CheckedU32(tokens_per_wave, "tokens_per_wave");
    desc.slot_count = CheckedU32(slot_count, "slot_count");
    desc.service_ring_size = CheckedU32(service_ring_size,
                                        "service_ring_size");
    desc.activation_waves = CheckedU32(activation_waves,
                                       "activation_waves");
    desc.spin_cap = CheckedU32(spin_cap, "spin_cap", true);

    auto state = std::make_shared<ServiceState>();
    const inc_fusion_status_t created = inc_fusion_prepared_plan_create(
        &desc, placement.first.data(), placement.second.data(), &state->plan);
    TORCH_CHECK(created == INC_FUSION_OK,
                "INC service plan creation failed: ", StatusName(created));
    inc_fusion_plan_info_t info{};
    const inc_fusion_status_t queried = inc_fusion_prepared_plan_info(
        state->plan, &info);
    TORCH_CHECK(queried == INC_FUSION_OK,
                "INC service plan query failed: ", StatusName(queried));
    TORCH_CHECK(static_cast<uint64_t>(symmetric_bytes) >= info.symmetric_bytes,
                "symmetric allocation is too small: need ",
                info.symmetric_bytes, ", got ", symmetric_bytes);
    TORCH_CHECK(aclrtMalloc(&state->workspace, info.inc_workspace_bytes,
                            ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
                    aclrtMemset(state->workspace, info.inc_workspace_bytes, 0,
                                info.inc_workspace_bytes) == ACL_SUCCESS &&
                    aclrtCreateStream(&state->stream) == ACL_SUCCESS,
                "failed to allocate INC service workspace/stream");
    inc_fusion_device_bindings_t bindings{};
    bindings.symmetric_base = reinterpret_cast<void *>(
        static_cast<uintptr_t>(symmetric_base));
    bindings.workspace = state->workspace;
    bindings.worker_pes = const_cast<void *>(worker_pes.const_data_ptr());
    bindings.ffts_addr = static_cast<uint64_t>(ffts_addr);
    if (inc_route_relay != 0)
        bindings.execution_flags |= INC_FUSION_EXEC_INC_ROUTE_RELAY;
    const inc_fusion_status_t service_created =
        inc_fusion_remote_service_create(
            state->plan, &bindings, state->stream, &state->service);
    TORCH_CHECK(service_created == INC_FUSION_OK,
                "inc_fusion_remote_service_create failed: ",
                StatusName(service_created));
    state->worker_pes = worker_pes;
    state->device = device;

    std::lock_guard<std::mutex> guard(g_service_registry_mutex);
    TORCH_CHECK(g_next_service_handle < std::numeric_limits<int64_t>::max(),
                "single-INC service handle space exhausted");
    const int64_t handle = g_next_service_handle++;
    g_service_registry.emplace(handle, std::move(state));
    return handle;
}

void ServiceStart(const at::Tensor &device_anchor, int64_t handle)
{
    const std::shared_ptr<ServiceState> state = LookupService(handle);
    CheckSameDevice(device_anchor, state->device, "device_anchor");
    std::lock_guard<std::mutex> guard(state->mutex);
    if (state->started) return;
    const inc_fusion_status_t status = inc_fusion_remote_service_start(
        state->service);
    TORCH_CHECK(status == INC_FUSION_OK,
                "inc_fusion_remote_service_start failed: ",
                StatusName(status));
    state->started = true;
}

void ServiceDestroy(const at::Tensor &device_anchor, int64_t handle)
{
    CheckNpuContiguous(device_anchor, "device_anchor");
    std::shared_ptr<ServiceState> removed;
    {
        std::lock_guard<std::mutex> guard(g_service_registry_mutex);
        const auto found = g_service_registry.find(handle);
        if (found == g_service_registry.end()) return;
        TORCH_CHECK(device_anchor.device().index() == found->second->device,
                    "device_anchor is on a different NPU");
        removed = std::move(found->second);
        g_service_registry.erase(found);
    }
    removed.reset();
}

} // namespace

TORCH_LIBRARY_FRAGMENT(inc_fusion_native, library)
{
    library.def(
        "plan_info(Tensor expert_owner, Tensor expert_local_index, "
        "int live_aiv, int live_aic, int worker_count, int rank, int inc_pe, "
        "int hidden, int intermediate, int expert_count, int topk, "
        "int token_capacity, int tokens_per_wave, int slot_count, "
        "int service_ring_size, int activation_waves, int spin_cap) -> int[]");
    library.def(
        "worker_prepare(Tensor expert_owner, Tensor expert_local_index, "
        "Tensor worker_pes, Tensor active_token_counts, int symmetric_base, "
        "int symmetric_bytes, int ffts_addr, int live_aiv, int live_aic, "
        "int worker_count, int rank, int inc_pe, int hidden, int intermediate, "
        "int expert_count, int topk, int token_capacity, int tokens_per_wave, "
        "int slot_count, int service_ring_size, int activation_waves, "
        "int spin_cap, int executor_ring_size, int backend_mode) -> int");
    library.def(
        "worker_destroy(Tensor device_anchor, int handle) -> ()");
    library.def(
        "worker_debug_slots(Tensor device_anchor, int handle) -> int[]");
    library.def(
        "worker_debug_traces(Tensor device_anchor, int handle) -> int[]");
    library.def(
        "worker_route_exchange_out(int handle, Tensor local_counts, "
        "Tensor(a!) global_counts) -> ()");
    library.def(
        "worker_moe_out(int handle, Tensor input, Tensor w13, Tensor w2, "
        "Tensor dispatch_rows, Tensor assignments, Tensor waves, "
        "Tensor active_token_counts, Tensor group_lists, "
        "Tensor(a!) output, int mode) -> ()");
    library.def(
        "service_prepare(Tensor expert_owner, Tensor expert_local_index, "
        "Tensor worker_pes, int symmetric_base, int symmetric_bytes, "
        "int ffts_addr, int live_aiv, int live_aic, int worker_count, "
        "int inc_pe, int hidden, int intermediate, int expert_count, int topk, "
        "int token_capacity, int tokens_per_wave, int slot_count, "
        "int service_ring_size, int activation_waves, int spin_cap, "
        "int inc_route_relay) -> int");
    library.def("service_start(Tensor device_anchor, int handle) -> ()");
    library.def("service_destroy(Tensor device_anchor, int handle) -> ()");
}

TORCH_LIBRARY_IMPL(inc_fusion_native, CPU, library)
{
    library.impl("plan_info", TORCH_FN(PlanInfo));
}

TORCH_LIBRARY_IMPL(inc_fusion_native, PrivateUse1, library)
{
    library.impl("worker_prepare", TORCH_FN(WorkerPrepare));
    library.impl("worker_destroy", TORCH_FN(WorkerDestroy));
    library.impl("worker_debug_slots", TORCH_FN(WorkerDebugSlots));
    library.impl("worker_debug_traces", TORCH_FN(WorkerDebugTraces));
    library.impl("worker_route_exchange_out",
                 TORCH_FN(WorkerRouteExchangeOut));
    library.impl("worker_moe_out", TORCH_FN(WorkerMoeOut));
    library.impl("service_prepare", TORCH_FN(ServicePrepare));
    library.impl("service_start", TORCH_FN(ServiceStart));
    library.impl("service_destroy", TORCH_FN(ServiceDestroy));
}
