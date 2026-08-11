#include <torch/library.h>

#include <cstdint>

#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "inc_fusion_route_pack.h"

namespace {

using inc::fusion::FusionDispatchRow;
using inc::fusion::FusionExpertAssignment;
using inc::fusion::FusionRoutePackStatus;
using inc::fusion::FusionWaveDesc;
using inc::fusion::kFusionRouteInt32;
using inc::fusion::kFusionRouteInt64;

void CheckNpuContiguous(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                name, " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

uint32_t IdType(const at::Tensor &ids)
{
    if (ids.scalar_type() == at::ScalarType::Int)
        return kFusionRouteInt32;
    TORCH_CHECK(ids.scalar_type() == at::ScalarType::Long,
                "topk_ids must be int32 or int64");
    return kFusionRouteInt64;
}

uint32_t U32(int64_t value, const char *name, bool allow_zero = false)
{
    TORCH_CHECK((allow_zero ? value >= 0 : value > 0) &&
                    static_cast<uint64_t>(value) <= UINT32_MAX,
                name, " is outside uint32 range: ", value);
    return static_cast<uint32_t>(value);
}

void RouteCountOut(const at::Tensor &topk_ids, at::Tensor &local_counts,
                   at::Tensor &status, int64_t tokens_per_wave)
{
    CheckNpuContiguous(topk_ids, "topk_ids");
    CheckNpuContiguous(local_counts, "local_counts");
    CheckNpuContiguous(status, "status");
    TORCH_CHECK(topk_ids.dim() == 2,
                "topk_ids must have shape [active_tokens, topk]");
    TORCH_CHECK(local_counts.dim() == 2 && local_counts.size(1) > 1 &&
                    local_counts.scalar_type() == at::ScalarType::Int,
                "local_counts must be int32 [wave_capacity, experts+1]");
    TORCH_CHECK(status.scalar_type() == at::ScalarType::Byte &&
                    status.numel() >= sizeof(FusionRoutePackStatus),
                "status must provide at least 64 uint8 bytes");
    const uint32_t token_count = U32(topk_ids.size(0), "active_tokens", true);
    const uint32_t topk = U32(topk_ids.size(1), "topk");
    const uint32_t wave_capacity = U32(local_counts.size(0),
                                       "wave_capacity");
    const uint32_t expert_count =
        U32(local_counts.size(1) - 1, "expert_count");
    aclrtStream stream = c10_npu::getCurrentNPUStream(
        topk_ids.device().index()).stream();
    launch_inc_fusion_route_count_kernel(
        topk_ids.numel() == 0 ? nullptr : topk_ids.const_data_ptr(),
        IdType(topk_ids), token_count, topk, expert_count,
        U32(tokens_per_wave, "tokens_per_wave"), wave_capacity,
        reinterpret_cast<uint32_t *>(
            local_counts.mutable_data_ptr<int32_t>()),
        reinterpret_cast<FusionRoutePackStatus *>(
            status.mutable_data_ptr<uint8_t>()),
        stream);
}

void RoutePackOutImpl(
    const at::Tensor &topk_ids, const at::Tensor &topk_weights,
    const at::Tensor &global_counts, const at::Tensor &expert_owner,
    const at::Tensor &expert_local_index, at::Tensor &dispatch_rows,
    at::Tensor &assignments, at::Tensor &group_lists, at::Tensor &waves,
    at::Tensor &scratch, at::Tensor &status, int64_t rank,
    int64_t hidden, int64_t element_bytes,
    int64_t tokens_per_wave, int64_t slot_count,
    int64_t activation_waves, int64_t route_lanes)
{
    const at::Tensor *inputs[] = {&topk_ids, &topk_weights, &global_counts,
        &expert_owner, &expert_local_index, &dispatch_rows, &assignments,
        &group_lists, &waves, &scratch, &status};
    const char *names[] = {"topk_ids", "topk_weights", "global_counts",
        "expert_owner", "expert_local_index", "dispatch_rows",
        "assignments", "group_lists", "waves", "scratch", "status"};
    for (size_t i = 0u; i < sizeof(inputs) / sizeof(inputs[0]); ++i)
        CheckNpuContiguous(*inputs[i], names[i]);
    TORCH_CHECK(topk_ids.dim() == 2 && topk_weights.sizes() == topk_ids.sizes(),
                "topk ids/weights must share [active_tokens, topk]");
    TORCH_CHECK(topk_weights.scalar_type() == at::ScalarType::Float,
                "topk_weights must be float32");
    TORCH_CHECK(global_counts.dim() == 3 && global_counts.size(2) > 1 &&
                    global_counts.scalar_type() == at::ScalarType::Int,
                "global_counts must be int32 [workers,waves,experts+1]");
    TORCH_CHECK(expert_owner.dim() == 1 &&
                    expert_owner.scalar_type() == at::ScalarType::Int &&
                    expert_local_index.sizes() == expert_owner.sizes() &&
                    expert_local_index.scalar_type() == at::ScalarType::Int,
                "expert placement must be matching int32 vectors");
    TORCH_CHECK(global_counts.size(2) - 1 == expert_owner.size(0),
                "global_counts expert dimension disagrees with placement");
    TORCH_CHECK(dispatch_rows.scalar_type() == at::ScalarType::Byte &&
                    dispatch_rows.numel() % sizeof(FusionDispatchRow) == 0,
                "dispatch_rows must be a uint8 protocol buffer");
    TORCH_CHECK(assignments.scalar_type() == at::ScalarType::Byte &&
                    assignments.numel() % sizeof(FusionExpertAssignment) == 0,
                "assignments must be a uint8 protocol buffer");
    TORCH_CHECK(group_lists.dim() == 2 &&
                    group_lists.scalar_type() == at::ScalarType::Long &&
                    group_lists.size(0) == global_counts.size(1),
                "group_lists must be int64 [wave_capacity,local_experts]");
    TORCH_CHECK(waves.scalar_type() == at::ScalarType::Byte &&
                    waves.numel() >= global_counts.size(1) *
                        static_cast<int64_t>(sizeof(FusionWaveDesc)),
                "waves protocol buffer is too small");
    TORCH_CHECK(scratch.scalar_type() == at::ScalarType::Int,
                "scratch must be int32");
    TORCH_CHECK(status.scalar_type() == at::ScalarType::Byte &&
                    status.numel() >= sizeof(FusionRoutePackStatus),
                "status must provide at least 64 uint8 bytes");

    const uint32_t token_count = U32(topk_ids.size(0), "active_tokens", true);
    const uint32_t worker_count = U32(global_counts.size(0), "worker_count");
    const uint32_t wave_capacity = U32(global_counts.size(1),
                                       "wave_capacity");
    const uint32_t expert_count =
        U32(global_counts.size(2) - 1, "expert_count");
    const uint32_t local_expert_count = U32(group_lists.size(1),
                                             "local_expert_count", true);
    TORCH_CHECK(rank >= 0 && rank < worker_count,
                "rank is outside worker group");
    const uint32_t lanes = U32(route_lanes, "route_lanes", true);
    if (lanes == 0u) {
        TORCH_CHECK(scratch.numel() >= 2 * expert_owner.numel(),
                    "scalar scratch must hold two int32 vectors per expert");
    } else {
        TORCH_CHECK(lanes <= inc::fusion::kFusionMaxAiv,
                    "route_lanes exceeds the fusion ABI maximum");
        const uint64_t required =
            inc::fusion::FusionRoutePackParallelScratchWords(
                wave_capacity, expert_count, lanes);
        TORCH_CHECK(static_cast<uint64_t>(scratch.numel()) >= required,
                    "parallel route scratch is too small: need ", required,
                    " int32, got ", scratch.numel());
    }
    aclrtStream stream = c10_npu::getCurrentNPUStream(
        topk_ids.device().index()).stream();
    if (lanes == 0u) {
        launch_inc_fusion_route_pack_kernel(
        topk_ids.numel() == 0 ? nullptr : topk_ids.const_data_ptr(),
        IdType(topk_ids),
        topk_weights.numel() == 0 ? nullptr
                                  : topk_weights.const_data_ptr<float>(),
        reinterpret_cast<const uint32_t *>(
            global_counts.const_data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t *>(
            expert_owner.const_data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t *>(
            expert_local_index.const_data_ptr<int32_t>()), worker_count,
        static_cast<uint32_t>(rank), token_count,
        U32(topk_ids.size(1), "topk"), U32(hidden, "hidden"),
        U32(element_bytes, "element_bytes"), expert_count,
        local_expert_count, U32(tokens_per_wave, "tokens_per_wave"),
        U32(slot_count, "slot_count"),
        U32(activation_waves, "activation_waves"),
        reinterpret_cast<FusionDispatchRow *>(
            dispatch_rows.mutable_data_ptr<uint8_t>()),
        static_cast<uint32_t>(dispatch_rows.numel() /
                              sizeof(FusionDispatchRow)),
        reinterpret_cast<FusionExpertAssignment *>(
            assignments.mutable_data_ptr<uint8_t>()),
        static_cast<uint32_t>(assignments.numel() /
                              sizeof(FusionExpertAssignment)),
        group_lists.mutable_data_ptr<int64_t>(),
        reinterpret_cast<FusionWaveDesc *>(waves.mutable_data_ptr<uint8_t>()),
        wave_capacity, reinterpret_cast<uint32_t *>(
            scratch.mutable_data_ptr<int32_t>()),
        reinterpret_cast<FusionRoutePackStatus *>(
            status.mutable_data_ptr<uint8_t>()),
        stream);
    } else {
        launch_inc_fusion_route_pack_parallel_kernel(
        topk_ids.numel() == 0 ? nullptr : topk_ids.const_data_ptr(),
        IdType(topk_ids),
        topk_weights.numel() == 0 ? nullptr
                                  : topk_weights.const_data_ptr<float>(),
        reinterpret_cast<const uint32_t *>(
            global_counts.const_data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t *>(
            expert_owner.const_data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t *>(
            expert_local_index.const_data_ptr<int32_t>()), worker_count,
        static_cast<uint32_t>(rank), token_count,
        U32(topk_ids.size(1), "topk"), U32(hidden, "hidden"),
        U32(element_bytes, "element_bytes"), expert_count,
        local_expert_count, U32(tokens_per_wave, "tokens_per_wave"),
        U32(slot_count, "slot_count"),
        U32(activation_waves, "activation_waves"),
        reinterpret_cast<FusionDispatchRow *>(
            dispatch_rows.mutable_data_ptr<uint8_t>()),
        static_cast<uint32_t>(dispatch_rows.numel() /
                              sizeof(FusionDispatchRow)),
        reinterpret_cast<FusionExpertAssignment *>(
            assignments.mutable_data_ptr<uint8_t>()),
        static_cast<uint32_t>(assignments.numel() /
                              sizeof(FusionExpertAssignment)),
        group_lists.mutable_data_ptr<int64_t>(),
        reinterpret_cast<FusionWaveDesc *>(waves.mutable_data_ptr<uint8_t>()),
        wave_capacity, lanes, reinterpret_cast<uint32_t *>(
            scratch.mutable_data_ptr<int32_t>()),
        reinterpret_cast<FusionRoutePackStatus *>(
            status.mutable_data_ptr<uint8_t>()),
        stream);
    }
}

void RoutePackOut(
    const at::Tensor &topk_ids, const at::Tensor &topk_weights,
    const at::Tensor &global_counts, const at::Tensor &expert_owner,
    const at::Tensor &expert_local_index, at::Tensor &dispatch_rows,
    at::Tensor &assignments, at::Tensor &group_lists, at::Tensor &waves,
    at::Tensor &scratch, at::Tensor &status, int64_t rank,
    int64_t hidden, int64_t element_bytes, int64_t tokens_per_wave,
    int64_t slot_count, int64_t activation_waves)
{
    RoutePackOutImpl(topk_ids, topk_weights, global_counts, expert_owner,
        expert_local_index, dispatch_rows, assignments, group_lists, waves,
        scratch, status, rank, hidden, element_bytes, tokens_per_wave,
        slot_count, activation_waves, 0);
}

void RoutePackParallelOut(
    const at::Tensor &topk_ids, const at::Tensor &topk_weights,
    const at::Tensor &global_counts, const at::Tensor &expert_owner,
    const at::Tensor &expert_local_index, at::Tensor &dispatch_rows,
    at::Tensor &assignments, at::Tensor &group_lists, at::Tensor &waves,
    at::Tensor &scratch, at::Tensor &status, int64_t rank,
    int64_t hidden, int64_t element_bytes, int64_t tokens_per_wave,
    int64_t slot_count, int64_t activation_waves, int64_t route_lanes)
{
    RoutePackOutImpl(topk_ids, topk_weights, global_counts, expert_owner,
        expert_local_index, dispatch_rows, assignments, group_lists, waves,
        scratch, status, rank, hidden, element_bytes, tokens_per_wave,
        slot_count, activation_waves, route_lanes);
}

} // namespace

TORCH_LIBRARY(inc_fusion_native, library)
{
    library.def(
        "route_count_out(Tensor topk_ids, Tensor(a!) local_counts, "
        "Tensor(b!) status, int tokens_per_wave) -> ()");
    library.def(
        "route_pack_out(Tensor topk_ids, Tensor topk_weights, "
        "Tensor global_counts, Tensor expert_owner, "
        "Tensor expert_local_index, Tensor(a!) dispatch_rows, "
        "Tensor(b!) assignments, Tensor(c!) group_lists, Tensor(d!) waves, "
        "Tensor(e!) scratch, Tensor(f!) status, int rank, "
        "int hidden, int element_bytes, "
        "int tokens_per_wave, int slot_count, int activation_waves) -> ()");
    library.def(
        "route_pack_parallel_out(Tensor topk_ids, Tensor topk_weights, "
        "Tensor global_counts, Tensor expert_owner, "
        "Tensor expert_local_index, Tensor(a!) dispatch_rows, "
        "Tensor(b!) assignments, Tensor(c!) group_lists, Tensor(d!) waves, "
        "Tensor(e!) scratch, Tensor(f!) status, int rank, "
        "int hidden, int element_bytes, int tokens_per_wave, "
        "int slot_count, int activation_waves, int route_lanes) -> ()");
}

TORCH_LIBRARY_IMPL(inc_fusion_native, PrivateUse1, library)
{
    library.impl("route_count_out", TORCH_FN(RouteCountOut));
    library.impl("route_pack_out", TORCH_FN(RoutePackOut));
    library.impl("route_pack_parallel_out", TORCH_FN(RoutePackParallelOut));
}
