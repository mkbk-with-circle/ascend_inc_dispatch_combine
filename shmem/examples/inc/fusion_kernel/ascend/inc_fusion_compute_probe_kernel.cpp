#include "inc_fusion_compute_device.h"

#if defined(INC_FUSION_LEGACY_OP_SYSTEM_RUN_CFG)
// CANN 8.x declares this CATLASS runtime record but does not synthesize its
// definition for a standalone AscendC translation unit.  Newer bisheng
// toolchains provide it automatically, so keep the definition on the legacy
// branch selected by the top-level compiler-version probe.
__gm__ struct OpSystemRunCfg g_opSystemRunCfg{Catlass::L2_OFFSET};
#endif

using namespace inc::fusion::compute;

__aicore__ inline void RunBf16GateCopyProbe(
    __gm__ bfloat16_t *gate_up, __gm__ bfloat16_t *activation,
    uint64_t rows, uint32_t intermediate)
{
    constexpr uint32_t kTile = 1024u;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> up_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> output_buf;
    pipe.InitBuffer(input_buf, kTile * sizeof(bfloat16_t));
    pipe.InitBuffer(up_buf, kTile * sizeof(bfloat16_t));
    pipe.InitBuffer(fp32_buf, kTile * sizeof(float));
    pipe.InitBuffer(output_buf, kTile * sizeof(bfloat16_t));
    auto input = input_buf.Get<bfloat16_t>();
    auto up = up_buf.Get<bfloat16_t>();
    auto fp32 = fp32_buf.Get<float>();
    auto output = output_buf.Get<bfloat16_t>();
    AscendC::GlobalTensor<bfloat16_t> gm_in;
    AscendC::GlobalTensor<bfloat16_t> gm_out;
    gm_in.SetGlobalBuffer(gate_up);
    gm_out.SetGlobalBuffer(activation);
    for (uint64_t row = 0u; row < rows; ++row) {
        for (uint32_t column = 0u; column < intermediate;
             column += kTile) {
            const uint32_t count = intermediate - column < kTile
                ? intermediate - column : kTile;
            AscendC::DataCopyExtParams copy(
                1u, count * sizeof(bfloat16_t), 0u, 0u, 0u);
            AscendC::DataCopyPadExtParams<bfloat16_t> pad(false, 0, 0, 0);
            AscendC::DataCopyPad(
                input, gm_in[row * 2u * intermediate + column], copy, pad);
            AscendC::DataCopyPad(
                up, gm_in[row * 2u * intermediate + intermediate + column],
                copy, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast(fp32, up, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(output, fp32, AscendC::RoundMode::CAST_RINT, count);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(
                gm_out[row * intermediate + column], output, copy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
}

extern "C" __global__ __aicore__ __mix__(1, 2) void
inc_fusion_bf16_gmm_probe_kernel(
    __gm__ bfloat16_t *a, __gm__ bfloat16_t *b, __gm__ bfloat16_t *c,
    __gm__ int64_t *group_list, __gm__ uint8_t *input_ready,
    __gm__ uint8_t *output_ready, uint32_t experts, uint32_t n, uint32_t k,
    uint64_t generation, uint32_t spin_cap, __gm__ uint8_t *system_workspace)
{
    (void)system_workspace;
    if ASCEND_IS_AIC {
        (void)RunGroupedBf16(a, b, c, group_list, input_ready,
                             output_ready,
                             1u, 1u,
                             static_cast<uint32_t>(AscendC::GetBlockNum()),
                             experts, n, k,
                             generation, spin_cap);
    }
}

extern "C" void launch_inc_fusion_bf16_gmm_probe_kernel(
    bfloat16_t *a, bfloat16_t *b, bfloat16_t *c, int64_t *group_list,
    uint8_t *input_ready, uint8_t *output_ready, uint32_t experts,
    uint32_t n, uint32_t k, uint64_t generation, uint32_t spin_cap,
    uint8_t *system_workspace, int block_dim, void *stream)
{
    inc_fusion_bf16_gmm_probe_kernel<<<block_dim, nullptr, stream>>>(
        a, b, c, group_list, input_ready, output_ready, experts, n, k,
        generation, spin_cap, system_workspace);
}

extern "C" __global__ __aicore__ __mix__(1, 2) void
inc_fusion_bf16_ffn_probe_kernel(
    __gm__ bfloat16_t *a, __gm__ bfloat16_t *w13,
    __gm__ bfloat16_t *gate_up, __gm__ bfloat16_t *activation,
    __gm__ bfloat16_t *w2, __gm__ bfloat16_t *out,
    __gm__ int64_t *group_list, __gm__ uint8_t *dispatch_ready,
    __gm__ uint8_t *gmm1_ready, __gm__ uint8_t *activation_ready,
    __gm__ uint8_t *gmm2_ready, uint32_t experts, uint32_t hidden,
    uint32_t intermediate, uint64_t generation, uint32_t spin_cap,
    __gm__ uint8_t *system_workspace)
{
    (void)system_workspace;
    const uint32_t aic_count = static_cast<uint32_t>(AscendC::GetBlockNum());
    const uint32_t aiv_count = aic_count *
        static_cast<uint32_t>(AscendC::GetSubBlockNum());
    const bool all_ready_grouped = experts > aic_count;
    if ASCEND_IS_AIC {
        if (all_ready_grouped) {
            if (!WaitGeneration(dispatch_ready, generation, spin_cap)) return;
            const uint32_t rows = static_cast<uint32_t>(group_list[experts - 1u]);
            RunGroupedBf16AllReady(a, w13, gate_up, group_list, rows,
                                   experts, 2u * intermediate, hidden, false);
            PublishGeneration(gmm1_ready +
                static_cast<uint64_t>(AscendC::GetBlockIdx()) * 64u,
                generation);
            if (!WaitAllGenerations(activation_ready, aiv_count,
                                    generation, spin_cap)) return;
            RunGroupedBf16AllReady(activation, w2, out, group_list, rows,
                                   experts, hidden, intermediate, false);
            PublishGeneration(gmm2_ready +
                static_cast<uint64_t>(AscendC::GetBlockIdx()) * 64u,
                generation);
        } else {
            if (!RunGroupedBf16(a, w13, gate_up, group_list,
                                dispatch_ready, gmm1_ready,
                                1u, 1u, aic_count, experts,
                                2u * intermediate, hidden,
                                generation, spin_cap))
                return;
            (void)RunGroupedBf16(activation, w2, out, group_list,
                                 activation_ready, gmm2_ready,
                                 aiv_count, aiv_count, aic_count, experts,
                                 hidden, intermediate, generation, spin_cap);
        }
    }
    if ASCEND_IS_AIV {
        const uint32_t aiv = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (all_ready_grouped) {
            if (!WaitAllGenerations(gmm1_ready, aic_count,
                                    generation, spin_cap)) return;
            const uint32_t rows = static_cast<uint32_t>(group_list[experts - 1u]);
            RunBf16SwiGluAllReady(gate_up, activation, rows, intermediate,
                                  aiv, aiv_count);
            PublishGeneration(activation_ready +
                static_cast<uint64_t>(aiv) * 64u, generation);
        } else {
            (void)RunBf16SwiGlu(gate_up, activation, group_list,
                                gmm1_ready, activation_ready,
                                aic_count, aic_count, aiv_count,
                                aiv, aiv_count, experts, intermediate,
                                1u, generation, spin_cap);
        }
    }
}

extern "C" void launch_inc_fusion_bf16_ffn_probe_kernel(
    bfloat16_t *a, bfloat16_t *w13, bfloat16_t *gate_up,
    bfloat16_t *activation, bfloat16_t *w2, bfloat16_t *out,
    int64_t *group_list, uint8_t *dispatch_ready, uint8_t *gmm1_ready,
    uint8_t *activation_ready, uint8_t *gmm2_ready, uint32_t experts,
    uint32_t hidden, uint32_t intermediate, uint64_t generation,
    uint32_t spin_cap, uint8_t *system_workspace, int block_dim, void *stream)
{
    inc_fusion_bf16_ffn_probe_kernel<<<block_dim, nullptr, stream>>>(
        a, w13, gate_up, activation, w2, out, group_list, dispatch_ready,
        gmm1_ready, activation_ready, gmm2_ready, experts, hidden,
        intermediate, generation, spin_cap, system_workspace);
}

// Diagnostic-only launch that removes the in-kernel Cube-to-Vector handoff
// from the equation.  The host supplies a complete gate/up tensor and this
// kernel executes only the AIV SwiGLU stage.
extern "C" __global__ __aicore__ __mix__(1, 2) void
inc_fusion_bf16_activation_probe_kernel(
    __gm__ bfloat16_t *gate_up, __gm__ bfloat16_t *activation,
    uint64_t rows, uint32_t intermediate, __gm__ uint8_t *system_workspace)
{
    (void)system_workspace;
    if ASCEND_IS_AIV {
        const uint32_t aiv = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (aiv == 0u)
            RunBf16SwiGluAllReady(gate_up, activation, rows, intermediate,
                                  0u, 1u);
    }
}

extern "C" void launch_inc_fusion_bf16_activation_probe_kernel(
    bfloat16_t *gate_up, bfloat16_t *activation, uint64_t rows,
    uint32_t intermediate, uint8_t *system_workspace, int block_dim,
    void *stream)
{
    inc_fusion_bf16_activation_probe_kernel<<<block_dim, nullptr, stream>>>(
        gate_up, activation, rows, intermediate, system_workspace);
}
