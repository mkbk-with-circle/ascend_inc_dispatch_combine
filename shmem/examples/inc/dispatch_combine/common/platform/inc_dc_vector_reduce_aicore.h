#ifndef INC_DC_VECTOR_REDUCE_AICORE_H
#define INC_DC_VECTOR_REDUCE_AICORE_H

#include "kernel_operator.h"
#include "inc_dc_ub_tile_aicore.h"

// Vector reduce / Cast path: MTE2_V -> PIPE_V (Cast/Add) -> V_MTE3.
// Effective tile is computed on host; kernels must not clamp.

constexpr uint32_t INC_VEC_EVT_MTE2V = 0;
constexpr uint32_t INC_VEC_EVT_VMTE2 = 1;
constexpr uint32_t INC_VEC_EVT_MTE3V = 2;
constexpr uint32_t INC_VEC_EVT_VMTE3 = 3;
// Ascend910 AIV UB is 24 KiB; vector reduce uses 3x fp16 rows + 2x fp32 rows per tile.
constexpr int INC_VEC_UB_BUDGET_BYTES = INC_AIV_UB_BUDGET_BYTES_AICORE;
constexpr int INC_VEC_MAX_REPEAT = 512;

__aicore__ inline int IncVecUbAlignUp(int x, int align)
{
    return (x + align - 1) / align * align;
}

__aicore__ inline void IncVecInitPipeEvents()
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(INC_VEC_EVT_VMTE2);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(INC_VEC_EVT_MTE3V);
}

__aicore__ inline void IncVecDrainPipeEvents()
{
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(INC_VEC_EVT_VMTE2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(INC_VEC_EVT_MTE3V);
}

__aicore__ inline int IncVecFp16RowBytes(int tile_elems)
{
    return IncVecUbAlignUp(tile_elems * static_cast<int>(sizeof(uint16_t)), 32);
}

__aicore__ inline int IncVecFp32RowBytes(int tile_elems)
{
    return IncVecUbAlignUp(tile_elems * static_cast<int>(sizeof(float)), 32);
}

__aicore__ inline int IncVecVectorUbBytes(int tile_elems)
{
    const int fp16_row = IncVecFp16RowBytes(tile_elems);
    const int fp32_row = IncVecFp32RowBytes(tile_elems);
    return fp16_row * 3 + fp32_row * 2;
}

// Mirrors IncVectorEffectiveTileElemsHost; kernels must not hardcode hidden/tile.
__aicore__ inline int IncVecEffectiveTileElemsAicore(int requested_tile = 512)
{
    const int budget = INC_VEC_UB_BUDGET_BYTES;
    int tile = requested_tile > 0 ? requested_tile : 512;
    if (tile > INC_VEC_MAX_REPEAT) {
        tile = INC_VEC_MAX_REPEAT;
    }
    while (tile > 1 && IncVecVectorUbBytes(tile) > budget) {
        tile /= 2;
    }
    return tile > 0 ? tile : 1;
}

__aicore__ inline AscendC::LocalTensor<half> IncVecBindHalfUb(__ubuf__ uint8_t *ptr, int nbytes)
{
    AscendC::LocalTensor<half> t;
    t.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    t.address_.bufferAddr = reinterpret_cast<uint64_t>(ptr);
    t.address_.dataLen = static_cast<uint32_t>(IncVecUbAlignUp(nbytes, 32));
    return t;
}

__aicore__ inline AscendC::LocalTensor<float> IncVecBindFloatUb(__ubuf__ uint8_t *ptr, int nbytes)
{
    AscendC::LocalTensor<float> t;
    t.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    t.address_.bufferAddr = reinterpret_cast<uint64_t>(ptr);
    t.address_.dataLen = static_cast<uint32_t>(IncVecUbAlignUp(nbytes, 32));
    return t;
}

__aicore__ inline void IncVecCopyFp32UbToGm(__ubuf__ uint8_t *fp32_ub, GM_ADDR gm_dst, int n_elems)
{
    const int nbytes = n_elems * static_cast<int>(sizeof(float));
    AscendC::LocalTensor<float> fp32_t = IncVecBindFloatUb(fp32_ub, nbytes);
    AscendC::GlobalTensor<float> gm_out;
    gm_out.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_dst));
    AscendC::DataCopyExtParams params(1, static_cast<uint32_t>(nbytes), 0, 0, 0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(INC_VEC_EVT_VMTE3);
    AscendC::DataCopyPad(gm_out, fp32_t, params);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(INC_VEC_EVT_MTE3V);
}

__aicore__ inline void IncVecCopyHalfGmToUb(GM_ADDR gm_src, __ubuf__ uint8_t *ub_dst, int nbytes)
{
    AscendC::LocalTensor<uint8_t> ub_tensor;
    AscendC::GlobalTensor<uint8_t> gm_tensor;
    AscendC::DataCopyExtParams params(1, static_cast<uint32_t>(nbytes), 0, 0, 0);
    AscendC::DataCopyPadExtParams<uint8_t> pad_in;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ub_dst);
    ub_tensor.address_.dataLen = static_cast<uint32_t>(IncVecUbAlignUp(nbytes, 32));
    gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gm_src));
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(INC_VEC_EVT_VMTE2);
    AscendC::DataCopyPad(ub_tensor, gm_tensor, params, pad_in);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(INC_VEC_EVT_MTE2V);
}

__aicore__ inline void IncVecCopyHalfUbToGm(__ubuf__ uint8_t *ub_src, GM_ADDR gm_dst, int nbytes)
{
    AscendC::LocalTensor<uint8_t> ub_tensor;
    AscendC::GlobalTensor<uint8_t> gm_tensor;
    AscendC::DataCopyExtParams params(1, static_cast<uint32_t>(nbytes), 0, 0, 0);
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ub_src);
    ub_tensor.address_.dataLen = static_cast<uint32_t>(IncVecUbAlignUp(nbytes, 32));
    gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gm_dst));
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(INC_VEC_EVT_VMTE3);
    AscendC::DataCopyPad(gm_tensor, ub_tensor, params);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(INC_VEC_EVT_MTE3V);
}

__aicore__ inline void IncVecCastHalfUbToFp32(__ubuf__ uint8_t *half_ub, __ubuf__ uint8_t *fp32_ub, int n_elems)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(INC_VEC_EVT_MTE2V);
    AscendC::LocalTensor<half> half_t = IncVecBindHalfUb(half_ub, n_elems * static_cast<int>(sizeof(uint16_t)));
    AscendC::LocalTensor<float> fp32_t = IncVecBindFloatUb(fp32_ub, n_elems * static_cast<int>(sizeof(float)));
    AscendC::Cast(fp32_t, half_t, AscendC::RoundMode::CAST_NONE, n_elems);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void IncVecCastFp32UbToHalf(__ubuf__ uint8_t *fp32_ub, __ubuf__ uint8_t *half_ub, int n_elems)
{
    AscendC::LocalTensor<float> fp32_t = IncVecBindFloatUb(fp32_ub, n_elems * static_cast<int>(sizeof(float)));
    AscendC::LocalTensor<half> half_t = IncVecBindHalfUb(half_ub, n_elems * static_cast<int>(sizeof(uint16_t)));
    AscendC::Cast(half_t, fp32_t, AscendC::RoundMode::CAST_RINT, n_elems);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(INC_VEC_EVT_VMTE3);
}

// FP16->FP32 cast probe: GM half in -> GM fp32 out (no scalar conversion).
__aicore__ inline void IncVecCastHalfGmToFp32Gm(GM_ADDR half_gm, GM_ADDR fp32_gm, int n_elems, __ubuf__ uint8_t *ub_base,
                                                int fp16_row_bytes, int fp32_row_bytes)
{
    IncVecInitPipeEvents();
    __ubuf__ uint8_t *ub_half = ub_base;
    __ubuf__ uint8_t *ub_fp32 = ub_base + static_cast<uint64_t>(fp16_row_bytes);
    const int half_nbytes = n_elems * static_cast<int>(sizeof(uint16_t));
    IncVecCopyHalfGmToUb(half_gm, ub_half, half_nbytes);
    IncVecCastHalfUbToFp32(ub_half, ub_fp32, n_elems);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(INC_VEC_EVT_VMTE3);
    IncVecCopyFp32UbToGm(ub_fp32, fp32_gm, n_elems);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(INC_VEC_EVT_MTE3V);
    (void)fp32_row_bytes;
}

// True vector UB tile reduce: DataCopy -> Cast -> Add; final Cast -> DataCopy.
// Caller must IncVecInitPipeEvents() once before tiles and IncVecDrainPipeEvents() after.
// n_elems must be <= INC_VEC_MAX_REPEAT (outer tid loop splits larger payloads).
__aicore__ inline void IncReduceUbTileVector(GM_ADDR contrib_base, GM_ADDR out_base, int elem_base, int n_elems,
                                             int n_contrib, int contrib_stride_elems, __ubuf__ uint8_t *ub_base,
                                             int fp16_row_bytes, int fp32_row_bytes)
{
    const int fp16_nbytes = n_elems * static_cast<int>(sizeof(uint16_t));
    const int fp32_nbytes = n_elems * static_cast<int>(sizeof(float));
    __ubuf__ uint8_t *ub_ping = ub_base;
    __ubuf__ uint8_t *ub_pong = ub_base + static_cast<uint64_t>(fp16_row_bytes);
    __ubuf__ uint8_t *ub_fp32_temp = ub_pong + static_cast<uint64_t>(fp16_row_bytes);
    __ubuf__ uint8_t *ub_fp32_acc = ub_fp32_temp + static_cast<uint64_t>(fp32_row_bytes);
    __ubuf__ uint8_t *ub_fp16_out = ub_fp32_acc + static_cast<uint64_t>(fp32_row_bytes);

    AscendC::LocalTensor<float> acc = IncVecBindFloatUb(ub_fp32_acc, fp32_nbytes);
    AscendC::Duplicate(acc, static_cast<float>(0), n_elems);
    AscendC::PipeBarrier<PIPE_V>();

    for (int c = 0; c < n_contrib; ++c) {
        const uint64_t gm_off = (static_cast<uint64_t>(c) * static_cast<uint64_t>(contrib_stride_elems) +
                                 static_cast<uint64_t>(elem_base)) *
                                sizeof(uint16_t);
        __ubuf__ uint8_t *half_ub = (c & 1) ? ub_pong : ub_ping;
        IncVecCopyHalfGmToUb(contrib_base + gm_off, half_ub, fp16_nbytes);
        IncVecCastHalfUbToFp32(half_ub, ub_fp32_temp, n_elems);
        AscendC::LocalTensor<float> temp = IncVecBindFloatUb(ub_fp32_temp, fp32_nbytes);
        AscendC::Add(acc, acc, temp, n_elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(INC_VEC_EVT_VMTE2);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(INC_VEC_EVT_MTE3V);
    IncVecCastFp32UbToHalf(ub_fp32_acc, ub_fp16_out, n_elems);
    GM_ADDR dst = out_base + static_cast<uint64_t>(elem_base) * sizeof(uint16_t);
    IncVecCopyHalfUbToGm(ub_fp16_out, dst, fp16_nbytes);
}

// One weighted contributor tile: GM half -> Cast -> Muls(weight) -> Add into fp32 acc (UB only).
__aicore__ inline void IncVecWeightedAccumHalfGmTile(GM_ADDR half_gm, float weight, int n_elems,
                                                     __ubuf__ uint8_t *half_ub, __ubuf__ uint8_t *fp32_temp_ub,
                                                     __ubuf__ uint8_t *fp32_acc_ub, int fp16_nbytes, int fp32_nbytes)
{
    IncVecCopyHalfGmToUb(half_gm, half_ub, fp16_nbytes);
    IncVecCastHalfUbToFp32(half_ub, fp32_temp_ub, n_elems);
    AscendC::LocalTensor<float> temp = IncVecBindFloatUb(fp32_temp_ub, fp32_nbytes);
    if (weight != 1.f) {
        AscendC::Muls(temp, temp, weight, n_elems);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::LocalTensor<float> acc = IncVecBindFloatUb(fp32_acc_ub, fp32_nbytes);
    AscendC::Add(acc, acc, temp, n_elems);
    AscendC::PipeBarrier<PIPE_V>();
}

// Weighted variant of IncReduceUbTileVector: optional per-contrib weights in GM (nullptr => 1.f).
__aicore__ inline void IncReduceUbTileWeightedVector(GM_ADDR contrib_base, GM_ADDR out_base, int elem_base, int n_elems,
                                                     int n_contrib, int contrib_stride_elems, GM_ADDR weights_gm,
                                                     __ubuf__ uint8_t *ub_base, int fp16_row_bytes, int fp32_row_bytes)
{
    const int fp16_nbytes = n_elems * static_cast<int>(sizeof(uint16_t));
    const int fp32_nbytes = n_elems * static_cast<int>(sizeof(float));
    __ubuf__ uint8_t *ub_ping = ub_base;
    __ubuf__ uint8_t *ub_pong = ub_base + static_cast<uint64_t>(fp16_row_bytes);
    __ubuf__ uint8_t *ub_fp32_temp = ub_pong + static_cast<uint64_t>(fp16_row_bytes);
    __ubuf__ uint8_t *ub_fp32_acc = ub_fp32_temp + static_cast<uint64_t>(fp32_row_bytes);
    __ubuf__ uint8_t *ub_fp16_out = ub_fp32_acc + static_cast<uint64_t>(fp32_row_bytes);

    AscendC::LocalTensor<float> acc = IncVecBindFloatUb(ub_fp32_acc, fp32_nbytes);
    AscendC::Duplicate(acc, static_cast<float>(0), n_elems);
    AscendC::PipeBarrier<PIPE_V>();

    for (int c = 0; c < n_contrib; ++c) {
        float w = 1.f;
        if (weights_gm != nullptr) {
            w = reinterpret_cast<__gm__ float *>(weights_gm)[c];
        }
        const uint64_t gm_off = (static_cast<uint64_t>(c) * static_cast<uint64_t>(contrib_stride_elems) +
                                 static_cast<uint64_t>(elem_base)) *
                                sizeof(uint16_t);
        __ubuf__ uint8_t *half_ub = (c & 1) ? ub_pong : ub_ping;
        IncVecWeightedAccumHalfGmTile(contrib_base + gm_off, w, n_elems, half_ub, ub_fp32_temp, ub_fp32_acc, fp16_nbytes,
                                      fp32_nbytes);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(INC_VEC_EVT_VMTE2);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(INC_VEC_EVT_MTE3V);
    IncVecCastFp32UbToHalf(ub_fp32_acc, ub_fp16_out, n_elems);
    GM_ADDR dst = out_base + static_cast<uint64_t>(elem_base) * sizeof(uint16_t);
    IncVecCopyHalfUbToGm(ub_fp16_out, dst, fp16_nbytes);
}

// 7-contributor staging reduce: rows at w * staging_stride_elems, skip owner_shard row.
// Same vector Cast/Add path as IncReduceUbTileVector; reads non-contiguous GM rows.
__aicore__ inline void IncReduceUbStagingSkipOwnerTileVector(GM_ADDR staging_base, GM_ADDR out_base, int elem_base,
                                                             int n_elems, int owner_shard, int staging_stride_elems,
                                                             int n_workers, __ubuf__ uint8_t *ub_base,
                                                             int fp16_row_bytes, int fp32_row_bytes)
{
    const int fp16_nbytes = n_elems * static_cast<int>(sizeof(uint16_t));
    const int fp32_nbytes = n_elems * static_cast<int>(sizeof(float));
    __ubuf__ uint8_t *ub_ping = ub_base;
    __ubuf__ uint8_t *ub_pong = ub_base + static_cast<uint64_t>(fp16_row_bytes);
    __ubuf__ uint8_t *ub_fp32_temp = ub_pong + static_cast<uint64_t>(fp16_row_bytes);
    __ubuf__ uint8_t *ub_fp32_acc = ub_fp32_temp + static_cast<uint64_t>(fp32_row_bytes);
    __ubuf__ uint8_t *ub_fp16_out = ub_fp32_acc + static_cast<uint64_t>(fp32_row_bytes);

    AscendC::LocalTensor<float> acc = IncVecBindFloatUb(ub_fp32_acc, fp32_nbytes);
    AscendC::Duplicate(acc, static_cast<float>(0), n_elems);
    AscendC::PipeBarrier<PIPE_V>();

    int ci = 0;
    for (int w = 0; w < n_workers; ++w) {
        if (w == owner_shard) {
            continue;
        }
        const uint64_t gm_off = (static_cast<uint64_t>(w) * static_cast<uint64_t>(staging_stride_elems) +
                                 static_cast<uint64_t>(elem_base)) *
                                sizeof(uint16_t);
        __ubuf__ uint8_t *half_ub = (ci & 1) ? ub_pong : ub_ping;
        IncVecCopyHalfGmToUb(staging_base + gm_off, half_ub, fp16_nbytes);
        IncVecCastHalfUbToFp32(half_ub, ub_fp32_temp, n_elems);
        AscendC::LocalTensor<float> temp = IncVecBindFloatUb(ub_fp32_temp, fp32_nbytes);
        AscendC::Add(acc, acc, temp, n_elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(INC_VEC_EVT_VMTE2);
        ++ci;
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(INC_VEC_EVT_MTE3V);
    IncVecCastFp32UbToHalf(ub_fp32_acc, ub_fp16_out, n_elems);
    GM_ADDR dst = out_base + static_cast<uint64_t>(elem_base) * sizeof(uint16_t);
    IncVecCopyHalfUbToGm(ub_fp16_out, dst, fp16_nbytes);
}

#endif
