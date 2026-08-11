#ifndef INC_DC_COMBINE_VECTOR_REDUCE_AICORE_H
#define INC_DC_COMBINE_VECTOR_REDUCE_AICORE_H

/**
 * H7-C: C0 slab weighted reduce via AscendC MTE2/Cast/Muls/Add/MTE3.
 * Fail-closed: never silent-fallback to scalar. Caller chooses engine.
 */
#include "inc_dc_vector_reduce_aicore.h"
#include "inc_dc_fp16_aicore.h"

constexpr uint32_t kC0VecFailNone = 0u;
constexpr uint32_t kC0VecFailIntegrity = 14u;
constexpr uint32_t kC0VecFailOrdMissing = 16u;
constexpr uint32_t kC0VecFailVector = 19u; // UB/tile/event/tail precondition failed
constexpr uint32_t kC0VecWeightMagic = 0x57474B54u; // 'WGKT'
constexpr uint32_t kC0VecAgenStrideBytes = 64u;
constexpr uint32_t kC0VecWeightStrideBytes = 64u;
constexpr int kC0VecTileElemsRequested = 512;

struct alignas(64) C0VecWeightRec {
    uint64_t generation = 0;
    uint32_t weight_bits = 0;
    uint32_t payload_checksum = 0;
    uint32_t valid_cookie = 0;
    uint32_t reserved[11]{};
};
static_assert(sizeof(C0VecWeightRec) == 64, "C0VecWeightRec 64B");

__aicore__ inline float C0VecBitsToFloat(uint32_t bits)
{
    union {
        uint32_t u;
        float f;
    } x;
    x.u = bits;
    return x.f;
}

__aicore__ inline uint32_t C0VecWeightValidCookie(uint64_t generation, uint32_t weight_bits, uint32_t checksum,
                                                   uint32_t idx)
{
    return kC0VecWeightMagic ^ static_cast<uint32_t>(generation & 0xffffffffu) ^ weight_bits ^ checksum ^ idx;
}

__aicore__ inline uint32_t C0VecPayloadChecksum(__gm__ uint8_t *pay, uint32_t payload_bytes, uint32_t weight_bits)
{
    uint32_t chk = 0xA5A5u ^ weight_bits;
    const uint32_t n = payload_bytes / 4u;
    __gm__ uint32_t *pw = (__gm__ uint32_t *)pay;
    if (n <= 128u) {
        for (uint32_t i = 0; i < n; ++i) {
            chk ^= pw[i] + (i * 131u);
        }
    } else {
        for (uint32_t i = 0; i < 64u; ++i) {
            chk ^= pw[i] + (i * 131u);
        }
        for (uint32_t i = 64u; i + 64u < n; i += 16u) {
            chk ^= pw[i] + (i * 131u);
        }
        for (uint32_t i = n - 64u; i < n; ++i) {
            chk ^= pw[i] + (i * 131u);
        }
    }
    return chk;
}

__aicore__ inline void C0VecDcci(__gm__ uint8_t *p, uint32_t n)
{
    const uint32_t lines = (n + 63u) / 64u;
    for (uint32_t i = 0; i < lines; ++i) {
        dcci_cacheline(p + static_cast<uint64_t>(i) * 64u);
    }
}

// PERF-cut3：ready 已核对 agen；热路径只轻量确认 weight generation（免 slab/checksum 重扫）
__aicore__ inline uint32_t C0VecValidateReadyOrds(__gm__ uint8_t *base, uint64_t slab_off, uint64_t weight_off,
                                                   uint64_t arrival_gen_off, uint32_t agen_base, uint32_t expected,
                                                   uint32_t tile_bytes, uint64_t generation)
{
    (void)slab_off;
    (void)tile_bytes;
    for (uint32_t ord = 0; ord < expected; ++ord) {
        const uint32_t idx = agen_base + ord;
        __gm__ uint64_t *agen =
            (__gm__ uint64_t *)(base + arrival_gen_off + static_cast<uint64_t>(idx) * kC0VecAgenStrideBytes);
        __gm__ C0VecWeightRec *wrec =
            (__gm__ C0VecWeightRec *)(base + weight_off + static_cast<uint64_t>(idx) * kC0VecWeightStrideBytes);
        C0VecDcci((__gm__ uint8_t *)agen, 8u);
        C0VecDcci((__gm__ uint8_t *)wrec, 64u);
        if (*agen != generation) {
            return kC0VecFailOrdMissing;
        }
        if (wrec->generation != generation) {
            return kC0VecFailIntegrity;
        }
        const uint32_t expect_cookie =
            C0VecWeightValidCookie(generation, wrec->weight_bits, wrec->payload_checksum, idx);
        if (wrec->valid_cookie != expect_cookie) {
            return kC0VecFailIntegrity;
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    return kC0VecFailNone;
}

// RT0：单次 vector reduce 的 MTE2/V/MTE3 分项（cycle 累加；非 makespan）
struct C0VecPipeTel {
    uint64_t mte2_wait_cyc = 0;
    uint64_t v_active_cyc = 0;
    uint64_t v_wait_cyc = 0;
    uint64_t mte3_wait_cyc = 0;
    uint32_t mte2_submit = 0;
    uint32_t mte2_wait = 0;
    uint32_t v_active = 0;
    uint32_t v_wait = 0;
    uint32_t mte3_submit = 0;
    uint32_t mte3_wait = 0;
    uint32_t event_init = 0;
    uint32_t event_drain = 0;
    uint32_t pipe_all = 0;
};

// 带可选分项计时的 contributor tile（共用同一组 MTE2_V/V_MTE2 → 仍串行）
__aicore__ inline void C0VecWeightedAccumHalfGmTileTel(GM_ADDR half_gm, float weight, int n_elems,
                                                        __ubuf__ uint8_t *half_ub, __ubuf__ uint8_t *fp32_temp_ub,
                                                        __ubuf__ uint8_t *fp32_acc_ub, int fp16_nbytes,
                                                        int fp32_nbytes, C0VecPipeTel *tel)
{
    // MTE2 submit + wait V_MTE2
    {
        AscendC::LocalTensor<uint8_t> ub_tensor;
        AscendC::GlobalTensor<uint8_t> gm_tensor;
        AscendC::DataCopyExtParams params(1, static_cast<uint32_t>(fp16_nbytes), 0, 0, 0);
        AscendC::DataCopyPadExtParams<uint8_t> pad_in;
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(half_ub);
        ub_tensor.address_.dataLen = static_cast<uint32_t>(IncVecUbAlignUp(fp16_nbytes, 32));
        gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(half_gm));
        if (tel != nullptr) {
            const uint64_t t0 = AscendC::GetSystemCycle();
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(INC_VEC_EVT_VMTE2);
            tel->mte2_wait_cyc += (AscendC::GetSystemCycle() - t0);
            tel->mte2_wait += 1u;
            tel->mte2_submit += 1u;
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(INC_VEC_EVT_VMTE2);
        }
        AscendC::DataCopyPad(ub_tensor, gm_tensor, params, pad_in);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(INC_VEC_EVT_MTE2V);
    }
    // V wait MTE2_V + Cast/Muls/Add
    {
        uint64_t t_wait0 = 0ull;
        if (tel != nullptr) {
            t_wait0 = AscendC::GetSystemCycle();
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(INC_VEC_EVT_MTE2V);
        if (tel != nullptr) {
            tel->v_wait_cyc += (AscendC::GetSystemCycle() - t_wait0);
            tel->v_wait += 1u;
        }
        const uint64_t t_v0 = (tel != nullptr) ? AscendC::GetSystemCycle() : 0ull;
        AscendC::LocalTensor<half> half_t =
            IncVecBindHalfUb(half_ub, n_elems * static_cast<int>(sizeof(uint16_t)));
        AscendC::LocalTensor<float> fp32_t = IncVecBindFloatUb(fp32_temp_ub, fp32_nbytes);
        AscendC::Cast(fp32_t, half_t, AscendC::RoundMode::CAST_NONE, n_elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<float> temp = IncVecBindFloatUb(fp32_temp_ub, fp32_nbytes);
        if (weight != 1.f) {
            AscendC::Muls(temp, temp, weight, n_elems);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::LocalTensor<float> acc = IncVecBindFloatUb(fp32_acc_ub, fp32_nbytes);
        AscendC::Add(acc, acc, temp, n_elems);
        AscendC::PipeBarrier<PIPE_V>();
        if (tel != nullptr) {
            tel->v_active_cyc += (AscendC::GetSystemCycle() - t_v0);
            tel->v_active += 1u;
        }
    }
}

// Dual-buffer tiled weighted reduce；tel!=null 时累计 MTE2/V/MTE3 wait（RT0 真相计时）
// true_pingpong=1 → 转 VR1 真双缓冲路径
__aicore__ inline uint32_t C0WeightedReduceVectorTruePingpong(__gm__ uint8_t *base, uint64_t slab_off,
                                                                uint64_t weight_off, uint64_t arrival_gen_off,
                                                                __gm__ uint16_t *out, uint32_t agen_base,
                                                                uint32_t expected, uint32_t hidden, uint32_t tile_bytes,
                                                                uint64_t generation, C0VecPipeTel *tel);

__aicore__ inline uint32_t C0WeightedReduceVectorTel(__gm__ uint8_t *base, uint64_t slab_off, uint64_t weight_off,
                                                      uint64_t arrival_gen_off, __gm__ uint16_t *out,
                                                      uint32_t agen_base, uint32_t expected, uint32_t hidden,
                                                      uint32_t tile_bytes, uint64_t generation, C0VecPipeTel *tel,
                                                      uint32_t true_pingpong)
{
    if (true_pingpong != 0u) {
        return C0WeightedReduceVectorTruePingpong(base, slab_off, weight_off, arrival_gen_off, out, agen_base, expected,
                                                  hidden, tile_bytes, generation, tel);
    }
    if (out == nullptr || expected == 0u || hidden == 0u || tile_bytes < hidden * 2u) {
        return kC0VecFailVector;
    }
    // PERF-cut4：ready 合同已保证 agen/weight；跳过热路径重校验（仍读 weight_bits）
    (void)arrival_gen_off;
    (void)generation;
    AscendC::PipeBarrier<PIPE_ALL>();
    if (tel != nullptr) {
        tel->pipe_all += 1u;
    }

    const int tile_elems = IncVecEffectiveTileElemsAicore(kC0VecTileElemsRequested);
    if (tile_elems < 1 || tile_elems > INC_VEC_MAX_REPEAT) {
        return kC0VecFailVector;
    }
    const int fp16_row = IncVecFp16RowBytes(tile_elems);
    const int fp32_row = IncVecFp32RowBytes(tile_elems);
    if (IncVecVectorUbBytes(tile_elems) > INC_VEC_UB_BUDGET_BYTES) {
        return kC0VecFailVector;
    }

    __ubuf__ uint8_t *ub_base = reinterpret_cast<__ubuf__ uint8_t *>(0);
    __ubuf__ uint8_t *ub_ping = ub_base;
    __ubuf__ uint8_t *ub_pong = ub_base + static_cast<uint64_t>(fp16_row);
    __ubuf__ uint8_t *ub_fp32_temp = ub_pong + static_cast<uint64_t>(fp16_row);
    __ubuf__ uint8_t *ub_fp32_acc = ub_fp32_temp + static_cast<uint64_t>(fp32_row);
    __ubuf__ uint8_t *ub_fp16_out = ub_fp32_acc + static_cast<uint64_t>(fp32_row);

    const int contrib_stride_elems = static_cast<int>(tile_bytes / 2u);
    IncVecInitPipeEvents();
    if (tel != nullptr) {
        tel->event_init += 1u;
    }
    for (uint32_t e0 = 0; e0 < hidden; e0 += static_cast<uint32_t>(tile_elems)) {
        const int n = static_cast<int>(hidden - e0 < static_cast<uint32_t>(tile_elems) ? hidden - e0
                                                                                        : static_cast<uint32_t>(tile_elems));
        if (n <= 0 || n > INC_VEC_MAX_REPEAT) {
            IncVecDrainPipeEvents();
            if (tel != nullptr) {
                tel->event_drain += 1u;
            }
            return kC0VecFailVector;
        }
        const int fp16_nbytes = n * static_cast<int>(sizeof(uint16_t));
        const int fp32_nbytes = n * static_cast<int>(sizeof(float));
        AscendC::LocalTensor<float> acc = IncVecBindFloatUb(ub_fp32_acc, fp32_nbytes);
        AscendC::Duplicate(acc, static_cast<float>(0), n);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t ord = 0; ord < expected; ++ord) {
            const uint32_t idx = agen_base + ord;
            __gm__ C0VecWeightRec *wrec =
                (__gm__ C0VecWeightRec *)(base + weight_off + static_cast<uint64_t>(idx) * kC0VecWeightStrideBytes);
            const float w = C0VecBitsToFloat(wrec->weight_bits);
            GM_ADDR payload = reinterpret_cast<GM_ADDR>(
                base + slab_off + static_cast<uint64_t>(idx) * tile_bytes +
                static_cast<uint64_t>(e0) * sizeof(uint16_t));
            __ubuf__ uint8_t *half_ub = (ord & 1u) ? ub_pong : ub_ping;
            C0VecWeightedAccumHalfGmTileTel(payload, w, n, half_ub, ub_fp32_temp, ub_fp32_acc, fp16_nbytes,
                                             fp32_nbytes, tel);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(INC_VEC_EVT_VMTE2);
        }

        // MTE3：wait → cast → store
        {
            uint64_t t0 = 0ull;
            if (tel != nullptr) {
                t0 = AscendC::GetSystemCycle();
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(INC_VEC_EVT_MTE3V);
            if (tel != nullptr) {
                tel->mte3_wait_cyc += (AscendC::GetSystemCycle() - t0);
                tel->mte3_wait += 1u;
            }
        }
        {
            const uint64_t t_v0 = (tel != nullptr) ? AscendC::GetSystemCycle() : 0ull;
            IncVecCastFp32UbToHalf(ub_fp32_acc, ub_fp16_out, n);
            if (tel != nullptr) {
                tel->v_active_cyc += (AscendC::GetSystemCycle() - t_v0);
                tel->v_active += 1u;
            }
        }
        GM_ADDR dst = reinterpret_cast<GM_ADDR>((__gm__ uint8_t *)out + static_cast<uint64_t>(e0) * sizeof(uint16_t));
        if (tel != nullptr) {
            tel->mte3_submit += 1u;
        }
        IncVecCopyHalfUbToGm(ub_fp16_out, dst, fp16_nbytes);
        (void)contrib_stride_elems;
    }
    IncVecDrainPipeEvents();
    if (tel != nullptr) {
        tel->event_drain += 1u;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    if (tel != nullptr) {
        tel->pipe_all += 1u;
    }
    // PERF-cut4：大包免输出全量 DCCI（后续 TX put 走 MTE，小包仍刷）
    if (tile_bytes <= 512u) {
        C0VecDcci((__gm__ uint8_t *)out, tile_bytes);
    }
    return kC0VecFailNone;
}

__aicore__ inline uint32_t C0WeightedReduceVector(__gm__ uint8_t *base, uint64_t slab_off, uint64_t weight_off,
                                                   uint64_t arrival_gen_off, __gm__ uint16_t *out, uint32_t agen_base,
                                                   uint32_t expected, uint32_t hidden, uint32_t tile_bytes,
                                                   uint64_t generation)
{
    return C0WeightedReduceVectorTel(base, slab_off, weight_off, arrival_gen_off, out, agen_base, expected, hidden,
                                     tile_bytes, generation, nullptr, 0u);
}

// VR1：真 MTE2/V 双缓冲——ping/pong 各一对独立 event；preload + V∥下一 contributor MTE2
constexpr uint32_t kC0VecEvtPingMte2V = 0u;
constexpr uint32_t kC0VecEvtPingVMte2 = 1u;
constexpr uint32_t kC0VecEvtMte3V = 2u;
constexpr uint32_t kC0VecEvtVMte3 = 3u;
constexpr uint32_t kC0VecEvtPongMte2V = 4u;
constexpr uint32_t kC0VecEvtPongVMte2 = 5u;

__aicore__ inline void C0VecInitTruePingpongEvents()
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kC0VecEvtPingVMte2);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kC0VecEvtPongVMte2);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kC0VecEvtMte3V);
}

__aicore__ inline void C0VecDrainTruePingpongEvents()
{
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kC0VecEvtPingVMte2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kC0VecEvtPongVMte2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kC0VecEvtMte3V);
}

__aicore__ inline void C0VecCopyHalfGmToUbEvt(GM_ADDR gm_src, __ubuf__ uint8_t *ub_dst, int nbytes, uint32_t evt_vmte2,
                                               uint32_t evt_mte2v, C0VecPipeTel *tel)
{
    AscendC::LocalTensor<uint8_t> ub_tensor;
    AscendC::GlobalTensor<uint8_t> gm_tensor;
    AscendC::DataCopyExtParams params(1, static_cast<uint32_t>(nbytes), 0, 0, 0);
    AscendC::DataCopyPadExtParams<uint8_t> pad_in;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ub_dst);
    ub_tensor.address_.dataLen = static_cast<uint32_t>(IncVecUbAlignUp(nbytes, 32));
    gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gm_src));
    if (tel != nullptr) {
        const uint64_t t0 = AscendC::GetSystemCycle();
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evt_vmte2);
        tel->mte2_wait_cyc += (AscendC::GetSystemCycle() - t0);
        tel->mte2_wait += 1u;
        tel->mte2_submit += 1u;
    } else {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evt_vmte2);
    }
    AscendC::DataCopyPad(ub_tensor, gm_tensor, params, pad_in);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evt_mte2v);
}

__aicore__ inline void C0VecAccumHalfUb(float weight, int n_elems, __ubuf__ uint8_t *half_ub,
                                          __ubuf__ uint8_t *fp32_temp_ub, __ubuf__ uint8_t *fp32_acc_ub,
                                          int fp32_nbytes, uint32_t evt_mte2v, uint32_t evt_vmte2, C0VecPipeTel *tel)
{
    if (tel != nullptr) {
        const uint64_t t0 = AscendC::GetSystemCycle();
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evt_mte2v);
        tel->v_wait_cyc += (AscendC::GetSystemCycle() - t0);
        tel->v_wait += 1u;
    } else {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evt_mte2v);
    }
    const uint64_t t_v0 = (tel != nullptr) ? AscendC::GetSystemCycle() : 0ull;
    AscendC::LocalTensor<half> half_t =
        IncVecBindHalfUb(half_ub, n_elems * static_cast<int>(sizeof(uint16_t)));
    AscendC::LocalTensor<float> fp32_t = IncVecBindFloatUb(fp32_temp_ub, fp32_nbytes);
    AscendC::Cast(fp32_t, half_t, AscendC::RoundMode::CAST_NONE, n_elems);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::LocalTensor<float> temp = IncVecBindFloatUb(fp32_temp_ub, fp32_nbytes);
    if (weight != 1.f) {
        AscendC::Muls(temp, temp, weight, n_elems);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::LocalTensor<float> acc = IncVecBindFloatUb(fp32_acc_ub, fp32_nbytes);
    AscendC::Add(acc, acc, temp, n_elems);
    AscendC::PipeBarrier<PIPE_V>();
    if (tel != nullptr) {
        tel->v_active_cyc += (AscendC::GetSystemCycle() - t_v0);
        tel->v_active += 1u;
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evt_vmte2);
}

__aicore__ inline uint32_t C0WeightedReduceVectorTruePingpong(__gm__ uint8_t *base, uint64_t slab_off,
                                                                uint64_t weight_off, uint64_t arrival_gen_off,
                                                                __gm__ uint16_t *out, uint32_t agen_base,
                                                                uint32_t expected, uint32_t hidden, uint32_t tile_bytes,
                                                                uint64_t generation, C0VecPipeTel *tel)
{
    if (out == nullptr || expected == 0u || hidden == 0u || tile_bytes < hidden * 2u) {
        return kC0VecFailVector;
    }
    (void)arrival_gen_off;
    (void)generation;
    AscendC::PipeBarrier<PIPE_ALL>();
    if (tel != nullptr) {
        tel->pipe_all += 1u;
    }

    const int tile_elems = IncVecEffectiveTileElemsAicore(kC0VecTileElemsRequested);
    if (tile_elems < 1 || tile_elems > INC_VEC_MAX_REPEAT) {
        return kC0VecFailVector;
    }
    const int fp16_row = IncVecFp16RowBytes(tile_elems);
    const int fp32_row = IncVecFp32RowBytes(tile_elems);
    if (IncVecVectorUbBytes(tile_elems) > INC_VEC_UB_BUDGET_BYTES) {
        return kC0VecFailVector;
    }

    __ubuf__ uint8_t *ub_base = reinterpret_cast<__ubuf__ uint8_t *>(0);
    __ubuf__ uint8_t *ub_ping = ub_base;
    __ubuf__ uint8_t *ub_pong = ub_base + static_cast<uint64_t>(fp16_row);
    __ubuf__ uint8_t *ub_fp32_temp = ub_pong + static_cast<uint64_t>(fp16_row);
    __ubuf__ uint8_t *ub_fp32_acc = ub_fp32_temp + static_cast<uint64_t>(fp32_row);
    __ubuf__ uint8_t *ub_fp16_out = ub_fp32_acc + static_cast<uint64_t>(fp32_row);

    C0VecInitTruePingpongEvents();
    if (tel != nullptr) {
        tel->event_init += 1u;
    }

    // VR2-A：每 result 的 topk weight 只读一次（禁止每 tile×contrib 重读 GM）
    float wcache[16];
    if (expected > 16u) {
        return kC0VecFailVector;
    }
    for (uint32_t ord = 0; ord < expected; ++ord) {
        const uint32_t idx = agen_base + ord;
        __gm__ C0VecWeightRec *wrec =
            (__gm__ C0VecWeightRec *)(base + weight_off + static_cast<uint64_t>(idx) * kC0VecWeightStrideBytes);
        wcache[ord] = C0VecBitsToFloat(wrec->weight_bits);
    }

    for (uint32_t e0 = 0; e0 < hidden; e0 += static_cast<uint32_t>(tile_elems)) {
        const int n = static_cast<int>(hidden - e0 < static_cast<uint32_t>(tile_elems) ? hidden - e0
                                                                                        : static_cast<uint32_t>(tile_elems));
        if (n <= 0 || n > INC_VEC_MAX_REPEAT) {
            C0VecDrainTruePingpongEvents();
            if (tel != nullptr) {
                tel->event_drain += 1u;
            }
            return kC0VecFailVector;
        }
        const int fp16_nbytes = n * static_cast<int>(sizeof(uint16_t));
        const int fp32_nbytes = n * static_cast<int>(sizeof(float));
        AscendC::LocalTensor<float> acc = IncVecBindFloatUb(ub_fp32_acc, fp32_nbytes);
        AscendC::Duplicate(acc, static_cast<float>(0), n);
        AscendC::PipeBarrier<PIPE_V>();

        // preload contrib0 → ping
        {
            const uint32_t idx0 = agen_base;
            GM_ADDR p0 = reinterpret_cast<GM_ADDR>(base + slab_off + static_cast<uint64_t>(idx0) * tile_bytes +
                                                   static_cast<uint64_t>(e0) * sizeof(uint16_t));
            C0VecCopyHalfGmToUbEvt(p0, ub_ping, fp16_nbytes, kC0VecEvtPingVMte2, kC0VecEvtPingMte2V, tel);
        }

        for (uint32_t ord = 0; ord < expected; ++ord) {
            const uint32_t cur_ping = ((ord & 1u) == 0u) ? 1u : 0u;
            __ubuf__ uint8_t *cur_ub = (cur_ping != 0u) ? ub_ping : ub_pong;
            const uint32_t cur_m2v = (cur_ping != 0u) ? kC0VecEvtPingMte2V : kC0VecEvtPongMte2V;
            const uint32_t cur_vm2 = (cur_ping != 0u) ? kC0VecEvtPingVMte2 : kC0VecEvtPongVMte2;
            const float w = wcache[ord];

            // 先 kick 下一 contributor 的 MTE2，再对本 buffer 做 V（真 overlap）
            if (ord + 1u < expected) {
                const uint32_t nxt = ord + 1u;
                const uint32_t nxt_ping = ((nxt & 1u) == 0u) ? 1u : 0u;
                __ubuf__ uint8_t *nxt_ub = (nxt_ping != 0u) ? ub_ping : ub_pong;
                const uint32_t nxt_m2v = (nxt_ping != 0u) ? kC0VecEvtPingMte2V : kC0VecEvtPongMte2V;
                const uint32_t nxt_vm2 = (nxt_ping != 0u) ? kC0VecEvtPingVMte2 : kC0VecEvtPongVMte2;
                const uint32_t nidx = agen_base + nxt;
                GM_ADDR pn =
                    reinterpret_cast<GM_ADDR>(base + slab_off + static_cast<uint64_t>(nidx) * tile_bytes +
                                              static_cast<uint64_t>(e0) * sizeof(uint16_t));
                C0VecCopyHalfGmToUbEvt(pn, nxt_ub, fp16_nbytes, nxt_vm2, nxt_m2v, tel);
            }

            C0VecAccumHalfUb(w, n, cur_ub, ub_fp32_temp, ub_fp32_acc, fp32_nbytes, cur_m2v, cur_vm2, tel);
        }

        {
            uint64_t t0 = 0ull;
            if (tel != nullptr) {
                t0 = AscendC::GetSystemCycle();
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kC0VecEvtMte3V);
            if (tel != nullptr) {
                tel->mte3_wait_cyc += (AscendC::GetSystemCycle() - t0);
                tel->mte3_wait += 1u;
            }
        }
        {
            const uint64_t t_v0 = (tel != nullptr) ? AscendC::GetSystemCycle() : 0ull;
            AscendC::LocalTensor<float> fp32_t = IncVecBindFloatUb(ub_fp32_acc, fp32_nbytes);
            AscendC::LocalTensor<half> half_t = IncVecBindHalfUb(ub_fp16_out, fp16_nbytes);
            AscendC::Cast(half_t, fp32_t, AscendC::RoundMode::CAST_RINT, n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kC0VecEvtVMte3);
            if (tel != nullptr) {
                tel->v_active_cyc += (AscendC::GetSystemCycle() - t_v0);
                tel->v_active += 1u;
            }
        }
        GM_ADDR dst = reinterpret_cast<GM_ADDR>((__gm__ uint8_t *)out + static_cast<uint64_t>(e0) * sizeof(uint16_t));
        if (tel != nullptr) {
            tel->mte3_submit += 1u;
        }
        {
            AscendC::LocalTensor<uint8_t> ub_tensor;
            AscendC::GlobalTensor<uint8_t> gm_tensor;
            AscendC::DataCopyExtParams params(1, static_cast<uint32_t>(fp16_nbytes), 0, 0, 0);
            ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
            ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ub_fp16_out);
            ub_tensor.address_.dataLen = static_cast<uint32_t>(IncVecUbAlignUp(fp16_nbytes, 32));
            gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(dst));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kC0VecEvtVMte3);
            AscendC::DataCopyPad(gm_tensor, ub_tensor, params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kC0VecEvtMte3V);
        }
    }
    C0VecDrainTruePingpongEvents();
    if (tel != nullptr) {
        tel->event_drain += 1u;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    if (tel != nullptr) {
        tel->pipe_all += 1u;
    }
    if (tile_bytes <= 512u) {
        C0VecDcci((__gm__ uint8_t *)out, tile_bytes);
    }
    return kC0VecFailNone;
}

#endif
