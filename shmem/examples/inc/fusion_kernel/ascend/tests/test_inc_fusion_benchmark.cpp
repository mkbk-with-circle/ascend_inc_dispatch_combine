#include "inc_fusion_benchmark.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

namespace {

inc_fusion_benchmark_signature_t Signature(
    inc_fusion_benchmark_mode_t mode)
{
    inc_fusion_benchmark_signature_t result{};
    inc_fusion_benchmark_signature_init(&result);
    result.mode = mode;
    result.worker_count = 4u;
    result.uses_dedicated_inc =
        mode == INC_FUSION_BENCH_SERIAL_INC ||
                mode == INC_FUSION_BENCH_FUSED_INC
            ? 1u : 0u;
    result.dtype = 1u;
    result.token_count = 4097u;
    result.hidden = 2048u;
    result.intermediate = 8192u;
    result.expert_count = 64u;
    result.topk = 8u;
    result.token_wave_capacity = 512u;
    result.route_digest = 11u;
    result.expert_placement_digest = 12u;
    result.compute_implementation_digest = 13u;
    result.weight_layout_digest = 14u;
    return result;
}

void TestModes()
{
    inc_fusion_benchmark_mode_info_t info{};
    assert(inc_fusion_benchmark_mode_info(
        INC_FUSION_BENCH_SERIAL_SHMEM, &info));
    assert(info.transport == INC_FUSION_BENCH_TRANSPORT_SHMEM);
    assert(info.schedule == INC_FUSION_BENCH_SCHEDULE_SERIAL);
    assert(info.uses_dedicated_inc == 0u);
    assert(info.is_factorial_baseline == 1u);
    assert(inc_fusion_benchmark_mode_info(
        INC_FUSION_BENCH_FUSED_INC, &info));
    assert(info.transport == INC_FUSION_BENCH_TRANSPORT_INC);
    assert(info.schedule == INC_FUSION_BENCH_SCHEDULE_TOKEN_WAVE);
    assert(info.uses_dedicated_inc == 1u);
    assert(inc_fusion_benchmark_mode_info(
        INC_FUSION_BENCH_NATIVE_VLLM, &info));
    assert(info.is_factorial_baseline == 0u);
    assert(!inc_fusion_benchmark_mode_info(
        static_cast<inc_fusion_benchmark_mode_t>(99), &info));
}

void TestComparabilityGate()
{
    char error[256] = {};
    const auto serial = Signature(INC_FUSION_BENCH_SERIAL_SHMEM);
    auto fused = Signature(INC_FUSION_BENCH_FUSED_INC);
    assert(inc_fusion_benchmark_validate_factorial_pair(
        &serial, &fused, error, sizeof(error)));
    fused.compute_implementation_digest++;
    assert(!inc_fusion_benchmark_validate_factorial_pair(
        &serial, &fused, error, sizeof(error)));
    assert(std::strstr(error, "compute_implementation_digest") != nullptr);
    fused = Signature(INC_FUSION_BENCH_FUSED_INC);
    fused.timing_flags &= ~INC_FUSION_BENCH_TIME_ROUTE_PACK;
    assert(!inc_fusion_benchmark_validate_factorial_pair(
        &serial, &fused, error, sizeof(error)));
    fused = Signature(INC_FUSION_BENCH_NATIVE_VLLM);
    assert(!inc_fusion_benchmark_validate_factorial_pair(
        &serial, &fused, error, sizeof(error)));
    assert(std::strstr(error, "external") != nullptr);
}

void TestFiniteWaveTheory()
{
    inc_fusion_benchmark_stage_model_t model{};
    inc_fusion_benchmark_stage_model_init(&model);
    model.wave_count = 4u;
    model.dispatch_us = 2.0;
    model.ffn_us = 5.0;
    model.combine_us = 3.0;
    inc_fusion_benchmark_theory_t theory{};
    assert(inc_fusion_benchmark_compute_theory(&model, &theory));
    assert(std::abs(theory.communication_window_speedup_limit - 5.0 / 3.0) < 1e-12);
    assert(std::abs(theory.serial_pipeline_us - 40.0) < 1e-12);
    assert(std::abs(theory.ideal_fused_pipeline_us - 25.0) < 1e-12);
    assert(std::abs(theory.end_to_end_speedup_limit - 1.6) < 1e-12);
    model.wave_count = 1u;
    assert(inc_fusion_benchmark_compute_theory(&model, &theory));
    assert(std::abs(theory.end_to_end_speedup_limit - 1.0) < 1e-12);
}

} // namespace

int main()
{
    TestModes();
    TestComparabilityGate();
    TestFiniteWaveTheory();
    std::cout << "inc fusion benchmark contract tests passed\n";
    return 0;
}
