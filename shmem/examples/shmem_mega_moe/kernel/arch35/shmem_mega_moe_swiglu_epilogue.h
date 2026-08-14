/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file shmem_mega_moe_swiglu_epilogue.h
 * \brief Catlass epilogue for fused SwiGLU activation and MX quantization.
 */

#ifndef SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_SWIGLU_EPILOGUE_H
#define SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_SWIGLU_EPILOGUE_H

#if defined(__DAV_C310__) || defined(SHMEM_MEGA_MOE_EXAMPLE_ARCH35)
#if ASC_DEVKIT_MAJOR >= 9
#if __has_include("basic_api/kernel_basic_intf.h")
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#else
#include "kernel_operator.h"
#endif
#if __has_include("basic_api/reg_compute/kernel_reg_compute_intf.h")
#include "basic_api/reg_compute/kernel_reg_compute_intf.h"
#elif __has_include("reg_compute/kernel_reg_compute_intf.h")
#include "reg_compute/kernel_reg_compute_intf.h"
#else
#error "ShmemMegaMoe arch35 requires the AscendC register-compute headers"
#endif
#if defined(__ASC_NPU_HOST__) && !defined(__NPU_ARCH__)
namespace AscendC {
namespace MicroAPI = Reg;
}
#endif
#include "shmem_mega_moe_types.h"

namespace ShmemMegaMoeQuantization {
enum class QuantMode : uint32_t {
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERBLOCK_MODE = 0x1U << 4,
};

enum class QuantDtype : uint8_t {
    DEFAULT = 0x0U,
    FP8_E4M3FN = 0x1U,
    FP8_E5M2 = 0x1U << 1,
};
} // namespace ShmemMegaMoeQuantization

namespace {
constexpr int64_t PACKED_OUTPUT_ELEMENTS_PER_BLOCK = 64;
constexpr uint32_t OUTPUT_DATA_OFFSET_INDEX = 0;
constexpr uint32_t OUTPUT_SCALE_OFFSET_INDEX = 1;
constexpr uint32_t READY_FLAG_OFFSET_INDEX = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_TILE_ELEMENTS = 256 * 256;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr int16_t BF16_EXPONENT_SHIFT = 7;
constexpr uint16_t CUSTOM_NAN_VALUE = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100;
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;
} // namespace

static constexpr AscendC::MicroAPI::DivSpecificMode SHMEM_MEGA_MOE_SWISH_DIV_MODE = {
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    true,
};
static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_FP16_TO_FP32 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_BF16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
#define SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_DECL                                                    \
    template <                                                                                          \
        typename OutputType_, typename InputType_, typename SecondScaleType_, typename FirstScaleType_, \
        bool UsesTensorList_>
#define SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_ARGS \
    OutputType_, InputType_, SecondScaleType_, FirstScaleType_, UsesTensorList_

SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_DECL
class ShmemMegaMoeSwiGluEpilogue {
public:
    __aicore__ inline ShmemMegaMoeSwiGluEpilogue() {}

    struct Arguments {
        GM_ADDR outputData{nullptr};
        GM_ADDR outputScales{nullptr};
        GM_ADDR readyFlags{nullptr};
        uint32_t tileRows;
        Arguments() = default;
    };

    using Config = Arguments;

    using OutputType = OutputType_;
    using InputType = InputType_;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

public:
    __aicore__ inline void Init(Config const& config);
    __aicore__ inline void operator()(const BlockShape& blockShape, const BlockCoord& blockCoord);
    __aicore__ inline void BindOutputBuffers(const BlockCoord& baseOffset);
    __aicore__ inline void ConfigureTile(const ProblemShape& problemShape);

private:
    __aicore__ inline void ComputeSwiGluActivation(uint16_t rowCount);
    __aicore__ inline void RepackMxScales(uint16_t rowCount, uint16_t scaleCountPerRow);

    template <ShmemMegaMoeQuantization::QuantMode quantizationMode>
    __aicore__ inline void ExecuteSwiGluMxQuantization(
        __ubuf__ int8_t* outputDst, __ubuf__ uint16_t* scaleDst, __ubuf__ InputType* firstSrc,
        __ubuf__ InputType* secondSrc, uint16_t rowCount, uint16_t columnCount);

    __aicore__ inline void ComputeScale(
        __ubuf__ uint16_t* maxExponentData, __ubuf__ uint16_t* encodedScaleData, __ubuf__ uint16_t* inverseScaleData,
        uint32_t remainingScaleCount, uint16_t scaleRepeatCount);

    __aicore__ inline void ComputeMaxExp(
        __ubuf__ bfloat16_t* inputData, __ubuf__ uint16_t* maxExponentData, uint32_t remainingElementCount,
        uint16_t repeatCount);

    __aicore__ inline void ComputeDataForQuantTargetFp8(
        __ubuf__ bfloat16_t* inputData, __ubuf__ uint16_t* inverseScaleData, __ubuf__ int8_t* quantizedData,
        uint32_t remainingElementCount, uint16_t repeatCount);

    __aicore__ inline void ComputeDataForQuantTargetFp4(
        __ubuf__ bfloat16_t* inputData, __ubuf__ uint16_t* inverseScaleData, __ubuf__ int8_t* quantizedData,
        uint32_t remainingElementCount, uint16_t repeatCount);

    __aicore__ inline void StoreQuantizedOutput(
        uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t>& src);

    __aicore__ inline void StoreQuantizedScales(
        uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t>& src);

    AscendC::GlobalTensor<int8_t> quantOutputGlobal_;
    AscendC::GlobalTensor<int8_t> quantScaleGlobal_;
    __gm__ int32_t* readyFlags_;

    AscendC::LocalTensor<InputType> activationInputBuffer0_{AscendC::TPosition::VECIN, 0, MAX_TILE_ELEMENTS};
    AscendC::LocalTensor<InputType> activationInputBuffer1_{
        AscendC::TPosition::VECIN, MAX_TILE_ELEMENTS * sizeof(InputType), MAX_TILE_ELEMENTS};
    AscendC::LocalTensor<int8_t> quantizedActivation_;
    AscendC::LocalTensor<int8_t> activationScales_;
    AscendC::LocalTensor<int8_t> packedActivationScales_;
    AscendC::LocalTensor<bfloat16_t> activatedValues_;
    AscendC::LocalTensor<uint16_t> maximumExponents_;
    AscendC::LocalTensor<uint16_t> inverseScales_;
    Config config_{};

    int64_t outputWidth_;
    int64_t scaleCountPerRow_;
    int64_t alignedScaleCountPerRow_;
    uint32_t tileRowCount_;
    uint32_t tileColumnCount_;
    int64_t ubBlockBytes_ = 0;
    uint32_t vectorHalfElementCount_ = 0;
    uint16_t reducedElementCount_ = 0;
    uint16_t targetMaxExponent_ = 0;
};

SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_DECL
__aicore__ inline void ShmemMegaMoeSwiGluEpilogue<SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_ARGS>::Init(
    Config const& config)
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    config_ = config;
    if constexpr (AscendC::IsSameType<OutputType, fp8_e4m3fn_t>::value) {
        targetMaxExponent_ = FP8_E4M3_MAX_EXP;
    } else if constexpr (AscendC::IsSameType<OutputType, fp8_e5m2_t>::value) {
        targetMaxExponent_ = FP8_E5M2_MAX_EXP; // this
    } else if constexpr (AscendC::IsSameType<OutputType, fp4x2_e2m1_t>::value) {
        targetMaxExponent_ = FP4_E2M1_MAX_EXP;
    } else {
        targetMaxExponent_ = FP4_E1M2_MAX_EXP;
    }

    constexpr uint32_t gluResOffset = 0;
    activatedValues_ = AscendC::LocalTensor<bfloat16_t>(AscendC::TPosition::VECCALC, gluResOffset, MAX_TILE_ELEMENTS);
    constexpr uint32_t quantOutputOffset = gluResOffset + MAX_TILE_ELEMENTS * sizeof(bfloat16_t);
    quantizedActivation_ =
        AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, quantOutputOffset, MAX_TILE_ELEMENTS);
    constexpr uint32_t quantScaleOffset = quantOutputOffset + MAX_TILE_ELEMENTS * sizeof(int8_t);
    activationScales_ = AscendC::LocalTensor<int8_t>(
        AscendC::TPosition::VECOUT, quantScaleOffset, MAX_TILE_ELEMENTS / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t maxExpOffset = quantScaleOffset + MAX_TILE_ELEMENTS / AscendC::ONE_BLK_SIZE * sizeof(int8_t);
    maximumExponents_ = AscendC::LocalTensor<uint16_t>(
        AscendC::TPosition::VECCALC, maxExpOffset, MAX_TILE_ELEMENTS / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t halfScaleOffset = maxExpOffset + MAX_TILE_ELEMENTS / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
    inverseScales_ = AscendC::LocalTensor<uint16_t>(
        AscendC::TPosition::VECCALC, halfScaleOffset, MAX_TILE_ELEMENTS / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t quantScaleBlockOffset =
        halfScaleOffset + MAX_TILE_ELEMENTS / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
    packedActivationScales_ = AscendC::LocalTensor<int8_t>(
        AscendC::TPosition::VECOUT, quantScaleBlockOffset, config_.tileRows * AscendC::ONE_BLK_SIZE);
}

SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_DECL
__aicore__ inline void ShmemMegaMoeSwiGluEpilogue<SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_ARGS>::BindOutputBuffers(
    const BlockCoord& baseOffset)
{
    if constexpr (g_coreType == AscendC::AIV) {
        quantOutputGlobal_.SetGlobalBuffer(
            (__gm__ int8_t*)config_.outputData + Get<OUTPUT_DATA_OFFSET_INDEX>(baseOffset));
        quantScaleGlobal_.SetGlobalBuffer(
            (__gm__ int8_t*)config_.outputScales + Get<OUTPUT_SCALE_OFFSET_INDEX>(baseOffset));
        readyFlags_ = (__gm__ int32_t*)config_.readyFlags + Get<READY_FLAG_OFFSET_INDEX>(baseOffset) * INT_CACHELINE;
    }
}

SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_DECL
__aicore__ inline void ShmemMegaMoeSwiGluEpilogue<SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_ARGS>::ConfigureTile(
    const ProblemShape& problemShape)
{
    outputWidth_ = Get<SHAPE_COLUMN_INDEX>(problemShape); // n/2
    scaleCountPerRow_ =
        ShmemMegaMoeKernel::CeilDiv(static_cast<uint64_t>(outputWidth_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
}

SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_DECL
__aicore__ inline void ShmemMegaMoeSwiGluEpilogue<SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_ARGS>::StoreQuantizedOutput(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t>& src)
{
    AscendC::DataCopyExtParams outputCopyParams{1, 0, 0, 0, 0};
    outputCopyParams.blockCount = blockCount;                      // 128
    outputCopyParams.blockLen = tileColumnCount_ * sizeof(int8_t); // 256
    outputCopyParams.dstStride = (outputWidth_ - tileColumnCount_) * sizeof(int8_t);
    if constexpr (
        AscendC::IsSameType<OutputType, fp4x2_e2m1_t>::value || AscendC::IsSameType<OutputType, fp4x2_e1m2_t>::value) {
        outputCopyParams.blockLen = outputCopyParams.blockLen >> 1;
        outputCopyParams.dstStride = outputCopyParams.dstStride >> 1;
        offset = offset >> 1;
    }
    AscendC::DataCopyPad(quantOutputGlobal_[offset], src, outputCopyParams);
}

SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_DECL
__aicore__ inline void ShmemMegaMoeSwiGluEpilogue<SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_ARGS>::StoreQuantizedScales(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t>& src)
{
    AscendC::DataCopyExtParams outputCopyParams{1, 0, 0, 0, 0};
    auto blockScaleN =
        ShmemMegaMoeKernel::CeilDiv(static_cast<uint64_t>(tileColumnCount_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;                                 // 256 / 32 = 8
    outputCopyParams.blockLen = blockScaleN * sizeof(int8_t); // 8
    outputCopyParams.blockCount = blockCount;                 // 128
    outputCopyParams.dstStride = (scaleCountPerRow_ - blockScaleN) * sizeof(int8_t);
    AscendC::DataCopyPad(quantScaleGlobal_[offset], src, outputCopyParams);
}

SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_DECL
__aicore__ inline void ShmemMegaMoeSwiGluEpilogue<SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_ARGS>::ComputeMaxExp(
    __ubuf__ bfloat16_t* inputData, __ubuf__ uint16_t* maxExponentData, uint32_t remainingElementCount,
    uint16_t repeatCount)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<bfloat16_t> inputRegister0;
        AscendC::MicroAPI::RegTensor<bfloat16_t> inputRegister1;
        AscendC::MicroAPI::RegTensor<bfloat16_t> bf16InputRegister0;
        AscendC::MicroAPI::RegTensor<bfloat16_t> bf16InputRegister1;
        AscendC::MicroAPI::RegTensor<uint16_t> exponentRegister0;
        AscendC::MicroAPI::RegTensor<uint16_t> exponentRegister1;

        AscendC::MicroAPI::RegTensor<uint16_t> bf16ExponentMask;
        AscendC::MicroAPI::Duplicate(bf16ExponentMask, MAX_EXP_FOR_BF16);

        AscendC::MicroAPI::RegTensor<uint16_t> maximumExponentRegister;
        AscendC::MicroAPI::MaskReg scaleMask1;
        AscendC::MicroAPI::MaskReg scaleMask2;
        AscendC::MicroAPI::UnalignReg u1;
        static constexpr AscendC::MicroAPI::CastTrait bf16ReductionCastConfig = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};
        for (uint16_t i = 0; i < repeatCount; i++) {
            scaleMask1 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(remainingElementCount);
            scaleMask2 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(remainingElementCount);
            AscendC::MicroAPI::DataCopy<
                bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                inputRegister0, inputRegister1, inputData,
                vectorHalfElementCount_ * 2); // copy two chunks from inputData to regbase
            AscendC::MicroAPI::And(
                exponentRegister0, (AscendC::MicroAPI::RegTensor<uint16_t>&)inputRegister0, bf16ExponentMask,
                scaleMask1);
            AscendC::MicroAPI::And(
                exponentRegister1, (AscendC::MicroAPI::RegTensor<uint16_t>&)inputRegister1, bf16ExponentMask,
                scaleMask1);
            AscendC::MicroAPI::Max(maximumExponentRegister, exponentRegister0, exponentRegister1, scaleMask1);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(maximumExponentRegister, maximumExponentRegister, scaleMask1);

            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExponentData, maximumExponentRegister, u1, reducedElementCount_);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(maxExponentData, u1, 0);
    }
    return;
}

SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_DECL
__aicore__ inline void ShmemMegaMoeSwiGluEpilogue<SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_ARGS>::ComputeScale(
    __ubuf__ uint16_t* maxExponentData, __ubuf__ uint16_t* encodedScaleData, __ubuf__ uint16_t* inverseScaleData,
    uint32_t remainingScaleCount, uint16_t scaleRepeatCount) // 128*8  8
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> expMask, sharedExp, scaleValue, scaleBias, halfScale, fp8NanRegister;
        AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::RegTensor<uint16_t> maximumExponentRegister;
        AscendC::MicroAPI::RegTensor<bfloat16_t> inputRegister0, inputRegister1;
        AscendC::MicroAPI::MaskReg cmpResult, zeroMask, cmpResultSub, preMaskScale;
        AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue, zeroRegister, nanRegister, specialExponentRegister;
        AscendC::MicroAPI::Duplicate(maxExpValue, targetMaxExponent_);
        AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::MicroAPI::Duplicate(fp8NanRegister, MAX_EXP_FOR_FP8);
        AscendC::MicroAPI::Duplicate(zeroRegister, 0);
        AscendC::MicroAPI::Duplicate(nanRegister, CUSTOM_NAN_VALUE);
        AscendC::MicroAPI::MaskReg invalidDataMask, specialDataMask;
        AscendC::MicroAPI::Duplicate(specialExponentRegister, SPECIAL_EXP_THRESHOLD);
        for (uint16_t i = 0; i < scaleRepeatCount; i++) {                                // 8
            preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(remainingScaleCount); // 128*8
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maximumExponentRegister, maxExponentData, vectorHalfElementCount_);

            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(
                cmpResult, maximumExponentRegister, expMask,
                preMaskScale); // INF/NAN

            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(
                zeroMask, maximumExponentRegister, zeroRegister, preMaskScale);

            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::LE>(
                invalidDataMask, maximumExponentRegister, maxExpValue, preMaskScale);

            AscendC::MicroAPI::Select<uint16_t>(
                maximumExponentRegister, maxExpValue, maximumExponentRegister, invalidDataMask);
            AscendC::MicroAPI::Sub(
                sharedExp, maximumExponentRegister, maxExpValue,
                preMaskScale); // sharedExp = maximumExponentRegister - 0x0780
            AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, BF16_EXPONENT_SHIFT, preMaskScale);

            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegister, cmpResult);

            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegister, zeroMask);

            AscendC::MicroAPI::DataCopy<
                uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                encodedScaleData, scaleValue, vectorHalfElementCount_ >> 1, preMaskScale);

            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::EQ>(
                specialDataMask, sharedExp, scaleBias, preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale); // halfScale = 0x7f00 - sharedExp

            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegister, cmpResult);

            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegister, zeroMask);

            AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExponentRegister, halfScale, specialDataMask);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                inverseScaleData, halfScale, vectorHalfElementCount_, preMaskScale);
        }
    }
    return;
}

SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_DECL
__aicore__ inline void
ShmemMegaMoeSwiGluEpilogue<SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_ARGS>::ComputeDataForQuantTargetFp8(
    __ubuf__ bfloat16_t* inputData, __ubuf__ uint16_t* inverseScaleData, __ubuf__ int8_t* quantizedData,
    uint32_t remainingElementCount, uint16_t repeatCount)
{
    uint32_t expandedElementCount = remainingElementCount * 2; // 128*256*2
    using T = bfloat16_t;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg inputMask0, inputMask1, outputMask0, outputMask1;
        AscendC::MicroAPI::MaskReg maskAll =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<float> floatScaleForMul;
        AscendC::MicroAPI::RegTensor<T> inputRegister0, inputRegister1, convertedInputRegister0,
            convertedInputRegister1;
        AscendC::MicroAPI::RegTensor<bfloat16_t> bf16InputRegister0, bf16InputRegister1;
        AscendC::MicroAPI::RegTensor<float> inputFp32Even0, inputFp32Odd0, inputFp32Even1, inputFp32Odd1;
        AscendC::MicroAPI::RegTensor<OutputType> quantizedFp8Even0, quantizedFp8Odd0, quantizedFp8Even1,
            quantizedFp8Odd1;
        AscendC::MicroAPI::RegTensor<bfloat16_t> fp4PackedBf16Input0, fp4PackedBf16Input1;

        static constexpr AscendC::MicroAPI::CastTrait evenLaneCastConfig = {

            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait oddLaneCastConfig = {

            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait fp32ToFp8CastConfig = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

        for (uint16_t i = 0; i < repeatCount; i++) {                              // 128
            inputMask0 = AscendC::MicroAPI::UpdateMask<T>(remainingElementCount); // 128*256
            inputMask1 = AscendC::MicroAPI::UpdateMask<T>(remainingElementCount); // 128*256
            outputMask0 = AscendC::MicroAPI::UpdateMask<T>(expandedElementCount); // 128*256*2
            outputMask1 = AscendC::MicroAPI::UpdateMask<T>(expandedElementCount); // 128*256*2

            AscendC::MicroAPI::DataCopy<
                T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                inputRegister0, inputRegister1, inputData, vectorHalfElementCount_ * 2);

            AscendC::MicroAPI::DataCopy<
                uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, inverseScaleData, reducedElementCount_);
            // inputRegister0*halfScaleForMul
            AscendC::MicroAPI::Mul(
                inputRegister0, inputRegister0, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, inputMask0);
            // inputRegister1*halfScaleForMul
            AscendC::MicroAPI::Mul(
                inputRegister1, inputRegister1, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, inputMask0);

            AscendC::MicroAPI::Interleave(inputRegister0, inputRegister1, inputRegister0, inputRegister1);
            AscendC::MicroAPI::Cast<float, T, evenLaneCastConfig>(inputFp32Even0, inputRegister0, inputMask0);
            AscendC::MicroAPI::Cast<float, T, oddLaneCastConfig>(inputFp32Odd0, inputRegister0, inputMask0);

            AscendC::MicroAPI::Interleave(inputFp32Even0, inputFp32Odd0, inputFp32Even0, inputFp32Odd0);
            AscendC::MicroAPI::Cast<OutputType, float, fp32ToFp8CastConfig>(
                quantizedFp8Even0, inputFp32Even0, outputMask0);

            AscendC::MicroAPI::Cast<OutputType, float, fp32ToFp8CastConfig>(
                quantizedFp8Odd0, inputFp32Odd0, outputMask0);

            AscendC::MicroAPI::Cast<float, T, evenLaneCastConfig>(inputFp32Even1, inputRegister1, inputMask1);
            AscendC::MicroAPI::Cast<float, T, oddLaneCastConfig>(inputFp32Odd1, inputRegister1, inputMask1);

            AscendC::MicroAPI::Interleave(inputFp32Even1, inputFp32Odd1, inputFp32Even1, inputFp32Odd1);
            AscendC::MicroAPI::Cast<OutputType, float, fp32ToFp8CastConfig>(
                quantizedFp8Even1, inputFp32Even1, outputMask1);

            AscendC::MicroAPI::Cast<OutputType, float, fp32ToFp8CastConfig>(
                quantizedFp8Odd1, inputFp32Odd1, outputMask1);

            AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(

                quantizedData, (AscendC::MicroAPI::RegTensor<int8_t>&)quantizedFp8Even0,
                PACKED_OUTPUT_ELEMENTS_PER_BLOCK, outputMask0);

            AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                quantizedData, (AscendC::MicroAPI::RegTensor<int8_t>&)quantizedFp8Odd0,
                PACKED_OUTPUT_ELEMENTS_PER_BLOCK, outputMask0);
            AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                quantizedData, (AscendC::MicroAPI::RegTensor<int8_t>&)quantizedFp8Even1,
                PACKED_OUTPUT_ELEMENTS_PER_BLOCK, outputMask1);
            AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                quantizedData, (AscendC::MicroAPI::RegTensor<int8_t>&)quantizedFp8Odd1,
                PACKED_OUTPUT_ELEMENTS_PER_BLOCK, outputMask1);
        }
    }
    return;
}

SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_DECL
__aicore__ inline void
ShmemMegaMoeSwiGluEpilogue<SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_ARGS>::ComputeDataForQuantTargetFp4(
    __ubuf__ bfloat16_t* inputData, __ubuf__ uint16_t* inverseScaleData, __ubuf__ int8_t* quantizedData,
    uint32_t remainingElementCount, uint16_t repeatCount)
{
    using T = bfloat16_t;
    using U = OutputType;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg inputMask0;
        AscendC::MicroAPI::MaskReg inputMask1;
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<T> inputRegister0;
        AscendC::MicroAPI::RegTensor<T> inputRegister1;
        AscendC::MicroAPI::RegTensor<T> convertedInputRegister0;
        AscendC::MicroAPI::RegTensor<T> convertedInputRegister1;

        AscendC::MicroAPI::RegTensor<bfloat16_t> bf16InputRegister0;
        AscendC::MicroAPI::RegTensor<bfloat16_t> bf16InputRegister1;

        AscendC::MicroAPI::RegTensor<U> quantizedFp4Input0;
        AscendC::MicroAPI::RegTensor<U> quantizedFp4Input1;

        AscendC::MicroAPI::RegTensor<bfloat16_t> fp4PackedBf16Input0;
        AscendC::MicroAPI::RegTensor<bfloat16_t> fp4PackedBf16Input1;
        static constexpr AscendC::MicroAPI::CastTrait fp4CastConfig = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < repeatCount; i++) {
            inputMask0 = AscendC::MicroAPI::UpdateMask<T>(remainingElementCount);
            inputMask1 = AscendC::MicroAPI::UpdateMask<T>(remainingElementCount);
            AscendC::MicroAPI::DataCopy<
                T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                inputRegister0, inputRegister1, inputData,
                vectorHalfElementCount_ * 2); // copy two chunks from inputData to regbase
            AscendC::MicroAPI::DataCopy<
                uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, inverseScaleData, reducedElementCount_);

            AscendC::MicroAPI::Mul(
                inputRegister0, inputRegister0, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, inputMask0);
            AscendC::MicroAPI::Mul(
                inputRegister1, inputRegister1, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, inputMask0);
            AscendC::MicroAPI::Interleave(inputRegister0, inputRegister1, inputRegister0, inputRegister1);
            AscendC::MicroAPI::Cast<U, T, fp4CastConfig>(quantizedFp4Input0, inputRegister0, inputMask0);
            AscendC::MicroAPI::Cast<U, T, fp4CastConfig>(quantizedFp4Input1, inputRegister1, inputMask1);

            AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                quantizedData, (AscendC::MicroAPI::RegTensor<int8_t>&)quantizedFp4Input0,
                PACKED_OUTPUT_ELEMENTS_PER_BLOCK, inputMask0);
            AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                quantizedData, (AscendC::MicroAPI::RegTensor<int8_t>&)quantizedFp4Input1,
                PACKED_OUTPUT_ELEMENTS_PER_BLOCK, inputMask1);
        }
    }
    return;
}

SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_DECL
template <ShmemMegaMoeQuantization::QuantMode quantizationMode>
__aicore__ inline void
ShmemMegaMoeSwiGluEpilogue<SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_ARGS>::ExecuteSwiGluMxQuantization(
    __ubuf__ int8_t* outputDst, __ubuf__ uint16_t* scaleDst, __ubuf__ InputType* firstSrc,
    __ubuf__ InputType* secondSrc, uint16_t rowCount, uint16_t columnCount) // rowCount=128 columnCount=256
{
    constexpr uint16_t sizePerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(InputType); // 128
    uint16_t rowRepeatCount =
        ShmemMegaMoeKernel::CeilDiv(static_cast<uint64_t>(columnCount), static_cast<uint64_t>(sizePerRepeat));
    uint32_t inputRowStride = ShmemMegaMoeKernel::AlignUp(
        static_cast<uint32_t>(columnCount), static_cast<uint32_t>(AscendC::ONE_BLK_SIZE / sizeof(InputType)));
    uint32_t outputRowStride =
        ShmemMegaMoeKernel::AlignUp(static_cast<uint32_t>(columnCount), static_cast<uint32_t>(AscendC::ONE_BLK_SIZE));
    const float scalarOne = 1.0;
    __ubuf__ bfloat16_t* activationOutput = (__ubuf__ bfloat16_t*)activatedValues_.GetPhyAddr();

    // swiglu
    __VEC_SCOPE__
    {
        for (uint16_t rowIndex = 0; rowIndex < rowCount; rowIndex++) {
            uint32_t remainingColumns = columnCount;
            AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<InputType>(remainingColumns);
            for (uint16_t vectorChunkIndex = 0; vectorChunkIndex < rowRepeatCount; vectorChunkIndex++) {
                AscendC::MicroAPI::RegTensor<InputType> swishInput0, gateInput0, swishInput1, gateInput1;
                AscendC::MicroAPI::RegTensor<float> swishData0, swishData1, gateData0, gateData1;

                uint32_t sourceOffset = rowIndex * inputRowStride + vectorChunkIndex * sizePerRepeat;
                AscendC::MicroAPI::DataCopy(swishInput0, firstSrc + sourceOffset); // swishInput0:x bf16
                AscendC::MicroAPI::DataCopy(gateInput0, secondSrc + sourceOffset); // gateInput0:y  bf16

                AscendC::MicroAPI::Interleave(swishInput0, swishInput1, swishInput0, swishInput1);
                AscendC::MicroAPI::Interleave(gateInput0, gateInput1, gateInput0, gateInput1);

                AscendC::MicroAPI::Cast<float, InputType, CAST_BF16_FP16_TO_FP32>(swishData0, swishInput0, mask);
                AscendC::MicroAPI::Cast<float, InputType, CAST_BF16_FP16_TO_FP32>(swishData1, swishInput1, mask);
                AscendC::MicroAPI::Cast<float, InputType, CAST_BF16_FP16_TO_FP32>(gateData0, gateInput0, mask);
                AscendC::MicroAPI::Cast<float, InputType, CAST_BF16_FP16_TO_FP32>(gateData1, gateInput1, mask);

                uint16_t conversionGroupCount = sizeof(float) / sizeof(InputType);   // 2
                uint16_t conversionGroupSize = sizePerRepeat / conversionGroupCount; // 128/2 = 64
                for (uint16_t i = 0; i < conversionGroupCount; i++) {
                    AscendC::MicroAPI::RegTensor<bfloat16_t> output; // output:swish(x)*y
                    AscendC::MicroAPI::RegTensor<float> swishData = i == 0 ? swishData0 : swishData1;
                    AscendC::MicroAPI::RegTensor<float> gateData = i == 0 ? gateData0 : gateData1;
                    AscendC::MicroAPI::RegTensor<float> negatedInput, exponentialValue, sigmoidDenominator;
                    AscendC::MicroAPI::RegTensor<float> activatedOutput, swishOutput;

                    // swish
                    AscendC::MicroAPI::Muls(negatedInput, swishData, -(scalarOne), mask); // negatedInput:-x
                    AscendC::MicroAPI::Exp(exponentialValue, negatedInput, mask);         // exponentialValue:exp(-x)
                    AscendC::MicroAPI::Adds(
                        sigmoidDenominator, exponentialValue, scalarOne, mask); // sigmoidDenominator:exp(-x)+1
                    AscendC::MicroAPI::Div<float, &SHMEM_MEGA_MOE_SWISH_DIV_MODE>(
                        swishOutput, swishData, sigmoidDenominator,
                        mask); // swishOutput:swish(x)=x/(exp(-x)+1)
                    // swish * gate
                    AscendC::MicroAPI::Mul(activatedOutput, swishOutput, gateData, mask); // activatedOutput:swish(x)*y

                    AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(output, activatedOutput, mask);

                    uint32_t destinationOffset =
                        rowIndex * outputRowStride + vectorChunkIndex * sizePerRepeat + i * conversionGroupSize;
                    AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                        activationOutput + destinationOffset, output, mask);
                }
            }
        }
    }

    // quant
    uint32_t activationElementCount = rowCount * outputRowStride;                   // 128*256
    uint32_t activationScaleCount = activationElementCount / AscendC::ONE_BLK_SIZE; // 128*256 / 32 = 128 * 8
    uint16_t dataRepeatCount =
        (activationElementCount + vectorHalfElementCount_ * 2 - 1) / (vectorHalfElementCount_ * 2);             // 128
    uint16_t scaleRepeatCount = (activationScaleCount + vectorHalfElementCount_ - 1) / vectorHalfElementCount_; // 8
    __ubuf__ uint16_t* maxExponentData = (__ubuf__ uint16_t*)maximumExponents_.GetPhyAddr();
    ComputeMaxExp(activationOutput, maxExponentData, activationElementCount, dataRepeatCount);
    __ubuf__ uint16_t* inverseScaleData = (__ubuf__ uint16_t*)inverseScales_.GetPhyAddr();
    ComputeScale(maxExponentData, scaleDst, inverseScaleData, activationScaleCount, scaleRepeatCount);
    if constexpr (
        AscendC::IsSameType<OutputType, fp8_e4m3fn_t>::value || AscendC::IsSameType<OutputType, fp8_e5m2_t>::value) {
        ComputeDataForQuantTargetFp8(
            activationOutput, inverseScaleData, outputDst, activationElementCount, dataRepeatCount);
    }
    if constexpr (
        AscendC::IsSameType<OutputType, fp4x2_e2m1_t>::value || AscendC::IsSameType<OutputType, fp4x2_e1m2_t>::value) {
        ComputeDataForQuantTargetFp4(
            activationOutput, inverseScaleData, outputDst, activationElementCount, dataRepeatCount);
    }
    return;
}

SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_DECL
__aicore__ inline void
ShmemMegaMoeSwiGluEpilogue<SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_ARGS>::ComputeSwiGluActivation(uint16_t rowCount)
{
    __ubuf__ int8_t* quantizedOutputBuffer = (__ubuf__ int8_t*)quantizedActivation_.GetPhyAddr();
    __ubuf__ uint16_t* scaleOutputBuffer = (__ubuf__ uint16_t*)activationScales_.GetPhyAddr();
    __ubuf__ InputType* firstActivationInput = (__ubuf__ InputType*)activationInputBuffer0_.GetPhyAddr();
    __ubuf__ InputType* secondActivationInput = (__ubuf__ InputType*)activationInputBuffer1_.GetPhyAddr();
    ExecuteSwiGluMxQuantization<ShmemMegaMoeQuantization::QuantMode::MX_PERGROUP_MODE>(
        quantizedOutputBuffer, scaleOutputBuffer, firstActivationInput, secondActivationInput, rowCount,
        tileColumnCount_);
}

SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_DECL
__aicore__ inline void ShmemMegaMoeSwiGluEpilogue<SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_ARGS>::RepackMxScales(
    uint16_t rowCount, uint16_t scaleCountPerRow)
{
    __ubuf__ int8_t* scaleOutputBuffer = (__ubuf__ int8_t*)activationScales_.GetPhyAddr();
    __ubuf__ int8_t* packedScaleBuffer = (__ubuf__ int8_t*)packedActivationScales_.GetPhyAddr();
    // scale layout: (rowCount*8) -> (rowCount,32)
    __VEC_SCOPE__
    {
        for (uint16_t rowIndex = 0; rowIndex < rowCount; ++rowIndex) { // 128
            uint32_t scaleElementCount = scaleCountPerRow;             // 8
            AscendC::MicroAPI::MaskReg maskScaleN = AscendC::MicroAPI::UpdateMask<int8_t>(scaleElementCount);
            AscendC::MicroAPI::RegTensor<int8_t> packedScaleRegister;
            AscendC::MicroAPI::UnalignReg u0, u1;
            auto sourceScale = scaleOutputBuffer + rowIndex * scaleCountPerRow;
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, sourceScale);
            AscendC::MicroAPI::DataCopyUnAlign(packedScaleRegister, u0, sourceScale);           // ?
            auto packedScaleDestination = packedScaleBuffer + rowIndex * AscendC::ONE_BLK_SIZE; // mId * 32
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(
                packedScaleDestination, packedScaleRegister, maskScaleN);
        }
    }
}

SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_DECL
__aicore__ inline void ShmemMegaMoeSwiGluEpilogue<SHMEM_MEGA_MOE_SWIGLU_EPILOGUE_TEMPLATE_ARGS>::operator()(
    const BlockShape& blockShape, const BlockCoord& blockCoord)
{
    tileRowCount_ = Get<SHAPE_ROW_INDEX>(blockShape);       // 128
    tileColumnCount_ = Get<SHAPE_COLUMN_INDEX>(blockShape); // 256
    alignedScaleCountPerRow_ =
        ShmemMegaMoeKernel::CeilDiv(static_cast<uint64_t>(tileColumnCount_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
    if (tileRowCount_ == 0) {
        return;
    }

    vectorHalfElementCount_ = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t); // 256 / 2 = 128
    ubBlockBytes_ = BLOCK_SIZE;                                               // 32
    reducedElementCount_ = AscendC::VECTOR_REG_WIDTH / ubBlockBytes_;         // 256 / 32 = 8

    uint64_t yOffset = Get<OUTPUT_DATA_OFFSET_INDEX>(blockCoord);
    uint64_t yScaleOffset = Get<OUTPUT_SCALE_OFFSET_INDEX>(blockCoord);
    ComputeSwiGluActivation(tileRowCount_);
    RepackMxScales(tileRowCount_, alignedScaleCountPerRow_);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    StoreQuantizedOutput(tileRowCount_, yOffset, quantizedActivation_);
    StoreQuantizedScales(tileRowCount_, yScaleOffset, packedActivationScales_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
    AscendC::AtomicAdd(readyFlags_, 1);
    return;
}

#endif // Arch35 device implementation
#endif // SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_SWIGLU_EPILOGUE_H
