#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

extern "C" void launch_inc_fusion_bf16_gmm_probe_kernel(
    void *a, void *b, void *c, int64_t *group_list,
    uint8_t *input_ready, uint8_t *output_ready, uint32_t experts,
    uint32_t n, uint32_t k, uint64_t generation, uint32_t spin_cap,
    uint8_t *system_workspace, int block_dim, void *stream);

extern "C" void launch_inc_fusion_bf16_ffn_probe_kernel(
    void *a, void *w13, void *gate_up, void *activation, void *w2, void *out,
    int64_t *group_list, uint8_t *dispatch_ready, uint8_t *gmm1_ready,
    uint8_t *activation_ready, uint8_t *gmm2_ready, uint32_t experts,
    uint32_t hidden, uint32_t intermediate, uint64_t generation,
    uint32_t spin_cap, uint8_t *system_workspace, int block_dim, void *stream);

extern "C" void launch_inc_fusion_bf16_activation_probe_kernel(
    void *gate_up, void *activation, uint64_t rows, uint32_t intermediate,
    uint8_t *system_workspace, int block_dim, void *stream);

namespace {

#define ACL_OK(expr) do { \
    const aclError status = (expr); \
    if (status != ACL_SUCCESS) { \
        std::cerr << #expr << " failed: " << status << "\n"; \
        return 1; \
    } \
} while (0)

uint16_t ToBf16(float value)
{
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<uint16_t>((bits + rounding) >> 16u);
}

float FromBf16(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16u;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

} // namespace

int main()
{
    constexpr uint32_t experts = 32u;
    constexpr uint32_t k = 2048u;
    constexpr uint32_t n = 768u;
    constexpr uint32_t rows_per_active_expert = 8u;
    constexpr uint32_t first_active_expert = experts / 2u;
    constexpr uint32_t rows =
        (experts - first_active_expert) * rows_per_active_expert;
    constexpr uint64_t generation = 7u;
    constexpr size_t system_workspace_bytes = 16u * 1024u * 1024u;

    std::vector<uint16_t> a(rows * k);
    std::vector<uint16_t> b(experts * k * n);
    for (uint32_t row = 0u; row < rows; ++row)
        for (uint32_t col = 0u; col < k; ++col)
        {
            uint32_t value = row * 0x85ebca6bu ^ col * 0xc2b2ae35u ^
                             0x243f6a88u;
            value ^= value >> 16u;
            value *= 0x7feb352du;
            value ^= value >> 15u;
            a[row * k + col] = ToBf16(static_cast<float>(
                static_cast<int32_t>(value % 257u) - 128) / 256.0f);
        }
    for (uint32_t expert = 0u; expert < experts; ++expert)
        for (uint32_t col = 0u; col < n; ++col)
            for (uint32_t inner = 0u; inner < k; ++inner)
            {
                uint32_t value = expert * 0x27d4eb2du ^
                                 col * 0x165667b1u ^
                                 inner * 0xd3a2646cu ^ 0x13198a2eu;
                value ^= value >> 15u;
                value *= 0x846ca68bu;
                value ^= value >> 16u;
                b[(static_cast<size_t>(expert) * n + col) * k + inner] =
                    ToBf16(static_cast<float>(
                        static_cast<int32_t>(value % 129u) - 64) / 1024.0f);
            }
    std::vector<float> golden(rows * n, 0.0f);
    for (uint32_t row = 0u; row < rows; ++row) {
        const uint32_t expert = first_active_expert +
            row / rows_per_active_expert;
        for (uint32_t col = 0u; col < n; ++col)
            for (uint32_t inner = 0u; inner < k; ++inner)
                golden[row * n + col] += FromBf16(a[row * k + inner]) *
                    FromBf16(b[(static_cast<size_t>(expert) * n + col) * k + inner]);
    }
    int64_t groups[experts] = {};
    for (uint32_t expert = 0u; expert < experts; ++expert)
        groups[expert] = expert < first_active_expert ? 0 :
            static_cast<int64_t>((expert - first_active_expert + 1u) *
                                 rows_per_active_expert);
    int64_t cube_cores = 0;
    std::vector<uint8_t> ready;
    const aclError init_status = aclInit(nullptr);
    if (init_status != ACL_SUCCESS &&
        init_status != ACL_ERROR_REPEAT_INITIALIZE) {
        std::cerr << "aclInit failed: " << init_status << "\n";
        return 1;
    }
    ACL_OK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_OK(aclrtCreateStream(&stream));
    ACL_OK(aclrtGetDeviceInfo(0, ACL_DEV_ATTR_CUBE_CORE_NUM, &cube_cores));
    if (cube_cores <= 0) return 1;
    ready.assign(experts * static_cast<size_t>(cube_cores) * 64u, 0u);
    for (uint32_t expert = 0u; expert < experts; ++expert)
        std::memcpy(ready.data() + static_cast<size_t>(expert) * 64u,
                    &generation, sizeof(generation));

    void *da = nullptr, *db = nullptr, *dc = nullptr;
    void *dg = nullptr, *din = nullptr, *dout = nullptr, *dws = nullptr;
    ACL_OK(aclrtMalloc(&da, a.size() * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&db, b.size() * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&dc, golden.size() * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&dg, sizeof(groups), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&din, ready.size(), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&dout, ready.size(), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&dws, system_workspace_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMemcpy(da, a.size() * sizeof(uint16_t), a.data(),
                      a.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(db, b.size() * sizeof(uint16_t), b.data(),
                      b.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(dg, sizeof(groups), groups, sizeof(groups),
                      ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(din, ready.size(), ready.data(), ready.size(),
                      ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemset(dout, ready.size(), 0, ready.size()));

    launch_inc_fusion_bf16_gmm_probe_kernel(
        da, db, dc, static_cast<int64_t *>(dg), static_cast<uint8_t *>(din),
        static_cast<uint8_t *>(dout), experts, n, k, generation, 40000000u,
        static_cast<uint8_t *>(dws), static_cast<int>(cube_cores), stream);
    ACL_OK(aclrtSynchronizeStream(stream));

    std::vector<uint16_t> output(golden.size());
    ACL_OK(aclrtMemcpy(output.data(), output.size() * sizeof(uint16_t), dc,
                      output.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<uint8_t> output_ready(ready.size());
    ACL_OK(aclrtMemcpy(output_ready.data(), output_ready.size(), dout,
                      output_ready.size(), ACL_MEMCPY_DEVICE_TO_HOST));
    float max_error = 0.0f;
    for (size_t i = 0; i < output.size(); ++i)
        max_error = std::max(max_error,
            std::abs(FromBf16(output[i]) - golden[i]));
    for (uint32_t expert = 0u; expert < experts; ++expert)
        for (int64_t core = 0; core < cube_cores; ++core) {
            uint64_t observed = 0u;
            std::memcpy(&observed, output_ready.data() +
                            (static_cast<size_t>(expert) * cube_cores + core) * 64u,
                        sizeof(observed));
            if (observed != generation) {
                std::cerr << "expert " << expert << ", core " << core
                          << " did not publish ready\n";
                return 1;
            }
        }
    std::cout << "BF16 grouped GEMM max_abs_error=" << max_error << "\n";
    if (!std::isfinite(max_error) || max_error > 0.5f) return 1;

    // End-to-end compute half of the fused worker: GMM1 and GMM2 execute on
    // AIC while all AIVs consume per-expert readiness and run BF16 SwiGLU.
    constexpr uint32_t hidden = k;
    constexpr uint32_t intermediate = n;
    const uint32_t aiv_cores = static_cast<uint32_t>(cube_cores) * 2u;
    std::vector<uint16_t> w13(
        static_cast<size_t>(experts) * 2u * intermediate * hidden);
    std::vector<uint16_t> w2(
        static_cast<size_t>(experts) * hidden * intermediate);
    for (size_t i = 0; i < w13.size(); ++i) {
        uint32_t value = static_cast<uint32_t>(i) * 0x27d4eb2du ^
                         0x13198a2eu;
        value ^= value >> 15u;
        value *= 0x846ca68bu;
        value ^= value >> 16u;
        w13[i] = ToBf16(static_cast<float>(
            static_cast<int32_t>(value % 129u) - 64) / 1024.0f);
    }
    for (size_t i = 0; i < w2.size(); ++i) {
        uint32_t value = static_cast<uint32_t>(i) * 0x94d049bbu ^
                         0xa4093822u;
        value ^= value >> 16u;
        value *= 0x45d9f3bu;
        value ^= value >> 16u;
        w2[i] = ToBf16(static_cast<float>(
            static_cast<int32_t>(value % 129u) - 64) / 1024.0f);
    }
    std::vector<uint16_t> golden_gate_up(
        static_cast<size_t>(rows) * 2u * intermediate);
    std::vector<uint16_t> golden_activation(
        static_cast<size_t>(rows) * intermediate);
    std::vector<float> golden_ffn(static_cast<size_t>(rows) * hidden, 0.0f);
    for (uint32_t row = 0u; row < rows; ++row) {
        const uint32_t expert = first_active_expert +
            row / rows_per_active_expert;
        for (uint32_t col = 0u; col < 2u * intermediate; ++col) {
            float sum = 0.0f;
            for (uint32_t inner = 0u; inner < hidden; ++inner)
                sum += FromBf16(a[row * hidden + inner]) *
                    FromBf16(w13[(static_cast<size_t>(expert) *
                        2u * intermediate + col) * hidden + inner]);
            golden_gate_up[static_cast<size_t>(row) * 2u * intermediate + col] =
                ToBf16(sum);
        }
        for (uint32_t col = 0u; col < intermediate; ++col) {
            const float gate = FromBf16(golden_gate_up[
                static_cast<size_t>(row) * 2u * intermediate + col]);
            const float up = FromBf16(golden_gate_up[
                static_cast<size_t>(row) * 2u * intermediate +
                intermediate + col]);
            golden_activation[static_cast<size_t>(row) * intermediate + col] =
                ToBf16((gate / (1.0f + std::exp(-gate))) * up);
        }
        for (uint32_t col = 0u; col < hidden; ++col)
            for (uint32_t inner = 0u; inner < intermediate; ++inner)
                golden_ffn[static_cast<size_t>(row) * hidden + col] +=
                    FromBf16(golden_activation[
                        static_cast<size_t>(row) * intermediate + inner]) *
                    FromBf16(w2[(static_cast<size_t>(expert) * hidden + col) *
                        intermediate + inner]);
    }
    std::vector<uint8_t> dispatch_ready(experts * 64u, 0u);
    for (uint32_t expert = 0u; expert < experts; ++expert)
        std::memcpy(dispatch_ready.data() + expert * 64u,
                    &generation, sizeof(generation));
    std::vector<uint8_t> gmm1_ready(
        static_cast<size_t>(experts) * cube_cores * 64u, 0u);
    std::vector<uint8_t> activation_ready(
        static_cast<size_t>(experts) * aiv_cores * 64u, 0u);
    std::vector<uint8_t> gmm2_ready(
        static_cast<size_t>(experts) * cube_cores * 64u, 0u);
    void *dw13 = nullptr, *dw2 = nullptr, *dgate = nullptr;
    void *dactivation = nullptr, *dffn = nullptr;
    void *ddispatch_ready = nullptr, *dgmm1_ready = nullptr;
    void *dactivation_ready = nullptr, *dgmm2_ready = nullptr;
#define ACL_ALLOC_COPY(ptr, host) do { \
    ACL_OK(aclrtMalloc(&(ptr), (host).size() * sizeof((host)[0]), \
                       ACL_MEM_MALLOC_HUGE_FIRST)); \
    ACL_OK(aclrtMemcpy((ptr), (host).size() * sizeof((host)[0]), (host).data(), \
                      (host).size() * sizeof((host)[0]), \
                      ACL_MEMCPY_HOST_TO_DEVICE)); \
} while (0)
    ACL_ALLOC_COPY(dw13, w13);
    ACL_ALLOC_COPY(dw2, w2);
    ACL_ALLOC_COPY(ddispatch_ready, dispatch_ready);
    ACL_OK(aclrtMalloc(&dgate, golden_gate_up.size() * sizeof(uint16_t),
                       ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&dactivation,
                       golden_activation.size() * sizeof(uint16_t),
                       ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&dffn, golden_ffn.size() * sizeof(uint16_t),
                       ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&dgmm1_ready, gmm1_ready.size(),
                       ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&dactivation_ready, activation_ready.size(),
                       ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&dgmm2_ready, gmm2_ready.size(),
                       ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMemset(dgmm1_ready, gmm1_ready.size(), 0, gmm1_ready.size()));
    ACL_OK(aclrtMemset(dactivation_ready, activation_ready.size(), 0,
                       activation_ready.size()));
    ACL_OK(aclrtMemset(dgmm2_ready, gmm2_ready.size(), 0, gmm2_ready.size()));
    launch_inc_fusion_bf16_ffn_probe_kernel(
        da, dw13, dgate, dactivation, dw2, dffn,
        static_cast<int64_t *>(dg), static_cast<uint8_t *>(ddispatch_ready),
        static_cast<uint8_t *>(dgmm1_ready),
        static_cast<uint8_t *>(dactivation_ready),
        static_cast<uint8_t *>(dgmm2_ready), experts, hidden, intermediate,
        generation, 40000000u, static_cast<uint8_t *>(dws),
        static_cast<int>(cube_cores), stream);
    ACL_OK(aclrtSynchronizeStream(stream));
    std::vector<uint16_t> observed_gate(golden_gate_up.size());
    std::vector<uint16_t> observed_activation(golden_activation.size());
    ACL_OK(aclrtMemcpy(observed_gate.data(), observed_gate.size() *
                      sizeof(uint16_t), dgate,
                      observed_gate.size() * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_OK(aclrtMemcpy(observed_activation.data(),
                      observed_activation.size() * sizeof(uint16_t),
                      dactivation,
                      observed_activation.size() * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<uint16_t> ffn_output(golden_ffn.size());
    ACL_OK(aclrtMemcpy(ffn_output.data(), ffn_output.size() * sizeof(uint16_t),
                      dffn, ffn_output.size() * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST));
    float ffn_max_error = 0.0f;
    float gate_max_error = 0.0f;
    float activation_max_error = 0.0f;
    for (size_t i = 0; i < observed_gate.size(); ++i)
        gate_max_error = std::max(gate_max_error,
            std::abs(FromBf16(observed_gate[i]) -
                     FromBf16(golden_gate_up[i])));
    for (size_t i = 0; i < observed_activation.size(); ++i)
        activation_max_error = std::max(activation_max_error,
            std::abs(FromBf16(observed_activation[i]) -
                     FromBf16(golden_activation[i])));
    for (size_t i = 0; i < ffn_output.size(); ++i)
        ffn_max_error = std::max(ffn_max_error,
            std::abs(FromBf16(ffn_output[i]) - golden_ffn[i]));
    std::cout << "BF16 fused FFN gate_max_abs_error=" << gate_max_error
              << " activation_max_abs_error=" << activation_max_error
              << " output_max_abs_error=" << ffn_max_error << "\n";
    ACL_OK(aclrtMemcpy(dgate,
                      golden_gate_up.size() * sizeof(uint16_t),
                      golden_gate_up.data(),
                      golden_gate_up.size() * sizeof(uint16_t),
                      ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemset(dactivation,
                      golden_activation.size() * sizeof(uint16_t), 0,
                      golden_activation.size() * sizeof(uint16_t)));
    launch_inc_fusion_bf16_activation_probe_kernel(
        dgate, dactivation, rows, intermediate,
        static_cast<uint8_t *>(dws), static_cast<int>(cube_cores), stream);
    ACL_OK(aclrtSynchronizeStream(stream));
    ACL_OK(aclrtMemcpy(observed_activation.data(),
                      observed_activation.size() * sizeof(uint16_t),
                      dactivation,
                      observed_activation.size() * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST));
    float isolated_activation_max_error = 0.0f;
    float isolated_swapped_max_error = 0.0f;
    size_t isolated_max_index = 0u;
    size_t isolated_bad_count = 0u;
    size_t isolated_zero_count = 0u;
    for (size_t i = 0; i < observed_activation.size(); ++i) {
        const float observed = FromBf16(observed_activation[i]);
        const size_t row = i / intermediate;
        const size_t col = i - row * intermediate;
        const float expected = FromBf16(golden_activation[i]);
        const float gate = FromBf16(golden_gate_up[
            row * 2u * intermediate + col]);
        const float up = FromBf16(golden_gate_up[
            row * 2u * intermediate + intermediate + col]);
        const float swapped = FromBf16(ToBf16(
            (up / (1.0f + std::exp(-up))) * gate));
        const float error = std::abs(observed - expected);
        isolated_swapped_max_error = std::max(
            isolated_swapped_max_error, std::abs(observed - swapped));
        if (error > isolated_activation_max_error) {
            isolated_activation_max_error = error;
            isolated_max_index = i;
        }
        if (error > 0.01f) ++isolated_bad_count;
        if (observed_activation[i] == 0u) ++isolated_zero_count;
    }
    std::cout << "BF16 isolated activation max_abs_error="
              << isolated_activation_max_error
              << " max_index=" << isolated_max_index
              << " observed="
              << FromBf16(observed_activation[isolated_max_index])
              << " expected="
              << FromBf16(golden_activation[isolated_max_index])
              << " swapped_max_error=" << isolated_swapped_max_error
              << " bad_count=" << isolated_bad_count
              << " zero_count=" << isolated_zero_count
              << " elements=" << observed_activation.size() << "\n";
    for (size_t i = 0u; i < std::min<size_t>(16u, observed_activation.size());
         ++i)
        std::cout << "ACT_SAMPLE i=" << i
                  << " observed=" << FromBf16(observed_activation[i])
                  << " expected=" << FromBf16(golden_activation[i])
                  << "\n";
    if (!std::isfinite(isolated_activation_max_error) ||
        isolated_activation_max_error > 0.01f)
        return 1;
#undef ACL_ALLOC_COPY
    ACL_OK(aclrtFree(dw13)); ACL_OK(aclrtFree(dw2));
    ACL_OK(aclrtFree(dgate)); ACL_OK(aclrtFree(dactivation));
    ACL_OK(aclrtFree(dffn)); ACL_OK(aclrtFree(ddispatch_ready));
    ACL_OK(aclrtFree(dgmm1_ready)); ACL_OK(aclrtFree(dactivation_ready));
    ACL_OK(aclrtFree(dgmm2_ready));

    ACL_OK(aclrtFree(da)); ACL_OK(aclrtFree(db)); ACL_OK(aclrtFree(dc));
    ACL_OK(aclrtFree(dg)); ACL_OK(aclrtFree(din)); ACL_OK(aclrtFree(dout));
    ACL_OK(aclrtFree(dws));
    ACL_OK(aclrtDestroyStream(stream));
    ACL_OK(aclrtResetDevice(0));
    ACL_OK(aclFinalize());
    return 0;
}
