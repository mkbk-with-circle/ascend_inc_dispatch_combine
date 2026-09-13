#include "inc_fusion_benchmark.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {

bool Fail(char *error, size_t bytes, const char *message)
{
    if (error != nullptr && bytes != 0u)
        std::snprintf(error, bytes, "%s", message);
    return false;
}

bool SameU32(uint32_t lhs, uint32_t rhs, char *error, size_t bytes,
             const char *field)
{
    if (lhs == rhs) return true;
    char message[160] = {};
    std::snprintf(message, sizeof(message),
                  "factorial baseline mismatch: %s", field);
    return Fail(error, bytes, message);
}

bool SameU64(uint64_t lhs, uint64_t rhs, char *error, size_t bytes,
             const char *field)
{
    if (lhs == rhs) return true;
    char message[160] = {};
    std::snprintf(message, sizeof(message),
                  "factorial baseline mismatch: %s", field);
    return Fail(error, bytes, message);
}

} // namespace

extern "C" void inc_fusion_benchmark_signature_init(
    inc_fusion_benchmark_signature_t *signature)
{
    if (signature == nullptr) return;
    std::memset(signature, 0, sizeof(*signature));
    signature->struct_size = sizeof(*signature);
    signature->abi_version = INC_FUSION_BENCHMARK_ABI_VERSION;
    signature->timing_flags = INC_FUSION_BENCH_REQUIRED_TIMING_FLAGS;
}

extern "C" void inc_fusion_benchmark_stage_model_init(
    inc_fusion_benchmark_stage_model_t *model)
{
    if (model == nullptr) return;
    std::memset(model, 0, sizeof(*model));
    model->struct_size = sizeof(*model);
    model->abi_version = INC_FUSION_BENCHMARK_ABI_VERSION;
}

extern "C" int inc_fusion_benchmark_mode_info(
    inc_fusion_benchmark_mode_t mode,
    inc_fusion_benchmark_mode_info_t *info)
{
    if (info == nullptr) return 0;
    std::memset(info, 0, sizeof(*info));
    info->struct_size = sizeof(*info);
    info->abi_version = INC_FUSION_BENCHMARK_ABI_VERSION;
    info->mode = static_cast<uint32_t>(mode);
    switch (mode) {
    case INC_FUSION_BENCH_SERIAL_SHMEM:
        info->transport = INC_FUSION_BENCH_TRANSPORT_SHMEM;
        info->schedule = INC_FUSION_BENCH_SCHEDULE_SERIAL;
        info->is_factorial_baseline = 1u;
        return 1;
    case INC_FUSION_BENCH_SERIAL_INC:
        info->transport = INC_FUSION_BENCH_TRANSPORT_INC;
        info->schedule = INC_FUSION_BENCH_SCHEDULE_SERIAL;
        info->uses_dedicated_inc = 1u;
        info->is_factorial_baseline = 1u;
        return 1;
    case INC_FUSION_BENCH_FUSED_SHMEM:
        info->transport = INC_FUSION_BENCH_TRANSPORT_SHMEM;
        info->schedule = INC_FUSION_BENCH_SCHEDULE_TOKEN_WAVE;
        info->is_factorial_baseline = 1u;
        return 1;
    case INC_FUSION_BENCH_FUSED_INC:
        info->transport = INC_FUSION_BENCH_TRANSPORT_INC;
        info->schedule = INC_FUSION_BENCH_SCHEDULE_TOKEN_WAVE;
        info->uses_dedicated_inc = 1u;
        info->is_factorial_baseline = 1u;
        return 1;
    case INC_FUSION_BENCH_NATIVE_VLLM:
        info->transport = INC_FUSION_BENCH_TRANSPORT_NATIVE;
        info->schedule = INC_FUSION_BENCH_SCHEDULE_NATIVE;
        return 1;
    default:
        std::memset(info, 0, sizeof(*info));
        return 0;
    }
}

extern "C" int inc_fusion_benchmark_validate_signature(
    const inc_fusion_benchmark_signature_t *signature,
    char *error, size_t error_bytes)
{
    if (signature == nullptr)
        return Fail(error, error_bytes, "null benchmark signature");
    if (signature->struct_size != sizeof(*signature) ||
        signature->abi_version != INC_FUSION_BENCHMARK_ABI_VERSION)
        return Fail(error, error_bytes, "benchmark signature ABI mismatch");
    inc_fusion_benchmark_mode_info_t mode{};
    if (!inc_fusion_benchmark_mode_info(
            static_cast<inc_fusion_benchmark_mode_t>(signature->mode), &mode))
        return Fail(error, error_bytes, "invalid benchmark mode");
    if (signature->worker_count == 0u || signature->token_count == 0u ||
        signature->hidden == 0u || signature->intermediate == 0u ||
        signature->expert_count == 0u || signature->topk == 0u ||
        signature->topk > signature->expert_count ||
        signature->token_wave_capacity == 0u)
        return Fail(error, error_bytes, "invalid benchmark workload shape");
    if (signature->uses_dedicated_inc != mode.uses_dedicated_inc)
        return Fail(error, error_bytes,
                    "dedicated INC usage does not match benchmark mode");
    if (signature->route_digest == 0u ||
        signature->expert_placement_digest == 0u ||
        signature->compute_implementation_digest == 0u ||
        signature->weight_layout_digest == 0u)
        return Fail(error, error_bytes,
                    "all comparability digests must be non-zero");
    if ((signature->timing_flags &
         INC_FUSION_BENCH_REQUIRED_TIMING_FLAGS) !=
        INC_FUSION_BENCH_REQUIRED_TIMING_FLAGS)
        return Fail(error, error_bytes,
                    "incomplete production timing boundary");
    if (error != nullptr && error_bytes != 0u) error[0] = '\0';
    return 1;
}

extern "C" int inc_fusion_benchmark_validate_factorial_pair(
    const inc_fusion_benchmark_signature_t *reference,
    const inc_fusion_benchmark_signature_t *candidate,
    char *error, size_t error_bytes)
{
    if (!inc_fusion_benchmark_validate_signature(
            reference, error, error_bytes) ||
        !inc_fusion_benchmark_validate_signature(
            candidate, error, error_bytes))
        return 0;
    inc_fusion_benchmark_mode_info_t ref_mode{};
    inc_fusion_benchmark_mode_info_t candidate_mode{};
    (void)inc_fusion_benchmark_mode_info(
        static_cast<inc_fusion_benchmark_mode_t>(reference->mode), &ref_mode);
    (void)inc_fusion_benchmark_mode_info(
        static_cast<inc_fusion_benchmark_mode_t>(candidate->mode),
        &candidate_mode);
    if (ref_mode.is_factorial_baseline == 0u ||
        candidate_mode.is_factorial_baseline == 0u)
        return Fail(error, error_bytes,
                    "native vLLM is an external, not factorial, baseline");
    if (!SameU32(reference->worker_count, candidate->worker_count,
                 error, error_bytes, "worker_count") ||
        !SameU64(reference->token_count, candidate->token_count,
                 error, error_bytes, "token_count") ||
        !SameU32(reference->dtype, candidate->dtype,
                 error, error_bytes, "dtype") ||
        !SameU32(reference->hidden, candidate->hidden,
                 error, error_bytes, "hidden") ||
        !SameU32(reference->intermediate, candidate->intermediate,
                 error, error_bytes, "intermediate") ||
        !SameU32(reference->expert_count, candidate->expert_count,
                 error, error_bytes, "expert_count") ||
        !SameU32(reference->topk, candidate->topk,
                 error, error_bytes, "topk") ||
        !SameU32(reference->token_wave_capacity,
                 candidate->token_wave_capacity,
                 error, error_bytes, "token_wave_capacity") ||
        !SameU64(reference->route_digest, candidate->route_digest,
                 error, error_bytes, "route_digest") ||
        !SameU64(reference->expert_placement_digest,
                 candidate->expert_placement_digest,
                 error, error_bytes, "expert_placement_digest") ||
        !SameU64(reference->compute_implementation_digest,
                 candidate->compute_implementation_digest,
                 error, error_bytes, "compute_implementation_digest") ||
        !SameU64(reference->weight_layout_digest,
                 candidate->weight_layout_digest,
                 error, error_bytes, "weight_layout_digest") ||
        !SameU64(reference->timing_flags, candidate->timing_flags,
                 error, error_bytes, "timing_flags"))
        return 0;
    if (error != nullptr && error_bytes != 0u) error[0] = '\0';
    return 1;
}

extern "C" int inc_fusion_benchmark_compute_theory(
    const inc_fusion_benchmark_stage_model_t *model,
    inc_fusion_benchmark_theory_t *theory)
{
    if (model == nullptr || theory == nullptr ||
        model->struct_size != sizeof(*model) ||
        model->abi_version != INC_FUSION_BENCHMARK_ABI_VERSION ||
        model->wave_count == 0u || !std::isfinite(model->dispatch_us) ||
        !std::isfinite(model->ffn_us) ||
        !std::isfinite(model->combine_us) ||
        model->dispatch_us <= 0.0 || model->ffn_us <= 0.0 ||
        model->combine_us <= 0.0)
        return 0;
    std::memset(theory, 0, sizeof(*theory));
    theory->struct_size = sizeof(*theory);
    theory->abi_version = INC_FUSION_BENCHMARK_ABI_VERSION;
    const double dc_max = std::max(model->dispatch_us, model->combine_us);
    const double stage_max = std::max(
        model->dispatch_us, std::max(model->ffn_us, model->combine_us));
    const double first_wave =
        model->dispatch_us + model->ffn_us + model->combine_us;
    theory->communication_window_speedup_limit =
        (model->dispatch_us + model->combine_us) / dc_max;
    theory->serial_pipeline_us =
        static_cast<double>(model->wave_count) * first_wave;
    theory->ideal_fused_pipeline_us =
        first_wave + static_cast<double>(model->wave_count - 1u) * stage_max;
    theory->end_to_end_speedup_limit =
        theory->serial_pipeline_us / theory->ideal_fused_pipeline_us;
    return 1;
}
