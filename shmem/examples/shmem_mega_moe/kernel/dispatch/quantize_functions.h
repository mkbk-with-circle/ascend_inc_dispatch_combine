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
 * \file quantize_functions.h
 * \brief
 */

#ifndef SHMEM_EXAMPLES_MEGA_MOE_KERNEL_DISPATCH_QUANTIZE_FUNCTIONS_H
#define SHMEM_EXAMPLES_MEGA_MOE_KERNEL_DISPATCH_QUANTIZE_FUNCTIONS_H

#include "dispatch/dispatch_common.h"

namespace quant {

constexpr int DIGIT_TWO = 2;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // elem_emax右移7位(BF16E8M7)
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
constexpr uint16_t FP4_E1M2_BF16_MAX_EXP = 0x0000;
constexpr uint16_t SPECIAL_VALUE_E2M1 = 0x00ff;
constexpr uint16_t SPECIAL_VALUE_E1M2 = 0x007f;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3_MAX_VALUE = 448.0f;
constexpr float HIFP8_MAX_VALUE = 32768.0f;
constexpr float INT8_MAX_VALUE = 127.0f;
constexpr uint16_t INVALID_FLOAT16 = 0x7c00;
constexpr uint16_t NEW_MANTISSA = 0x0008;

__aicore__ inline constexpr uint32_t GetUbBlockSizeDispatch() { return 32U; }

__aicore__ inline constexpr uint32_t GetVRegSizeDispatch()
{
#if __NPU_ARCH__ == 3510
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

template <typename T>
__aicore__ inline void ComputeMaxExp(__ubuf__ T* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint32_t totalCountInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(T); // 每个向量寄存器可以存储的元素个数
    // Number of elements written by one reduction.
    uint16_t elementAfterReduce = GetVRegSizeDispatch() / GetUbBlockSizeDispatch();
    uint16_t loopNum = AscendC::Ceil(totalCountInUB, 2 * vlForHalfNumber);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vdExp0;
        AscendC::MicroAPI::RegTensor<T> vdExp1;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpSelect0;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpSelect1;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract1;

        AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
        AscendC::MicroAPI::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);

        AscendC::MicroAPI::RegTensor<uint16_t> invalidMaskFP16;
        AscendC::MicroAPI::Duplicate(invalidMaskFP16, INVALID_FLOAT16);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::MaskReg scaleMask1;
        AscendC::MicroAPI::MaskReg scaleMask2;
        AscendC::MicroAPI::MaskReg invalidDataMask0;
        AscendC::MicroAPI::MaskReg invalidDataMask1;
        AscendC::MicroAPI::UnalignReg u1;
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            scaleMask2 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            // 双搬，将数据交织搬运到两个向量寄存器上
            AscendC::MicroAPI::DataCopy<
                T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, vlForHalfNumber * DIGIT_TWO);
            if constexpr (AscendC::Std::IsSame<T, half>::value) {
                AscendC::MicroAPI::And(
                    vdExpSelect0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, invalidMaskFP16, scaleMask1);
                AscendC::MicroAPI::And(
                    vdExpSelect1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, invalidMaskFP16, scaleMask1);
                // 将FP16非正常指数位与实际指数位作对比，生成非正常指数位掩码
                AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(
                    invalidDataMask0, vdExpSelect0, invalidMaskFP16, scaleMask1);
                AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(
                    invalidDataMask1, vdExpSelect1, invalidMaskFP16, scaleMask1);
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, scaleMask1);
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, scaleMask1);
                AscendC::MicroAPI::And(
                    vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0BF16, expMaskBF16,
                    scaleMask1); // 与操作保留指数位
                AscendC::MicroAPI::And(
                    vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1BF16, expMaskBF16, scaleMask1);
                // 筛选正常指数位，非正常值则使用expMaskBF16替代
                AscendC::MicroAPI::Select<uint16_t>(vdExpExtract0, vdExpExtract0, expMaskBF16, invalidDataMask0);
                AscendC::MicroAPI::Select<uint16_t>(vdExpExtract1, vdExpExtract1, expMaskBF16, invalidDataMask1);
            } else {
                AscendC::MicroAPI::And(
                    vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, expMaskBF16, scaleMask1);
                AscendC::MicroAPI::And(
                    vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, expMaskBF16, scaleMask1);
            }
            // 两个向量寄存器上的元素一一对比输出最大指数位
            AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            // 按每个block输出一个最大指数位
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, elementAfterReduce);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
}

template <typename T>
__aicore__ inline void ComputeScale(
    __ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
    uint32_t totalScaleInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(uint16_t);
    uint16_t loopNumScale = AscendC::Ceil(totalScaleInUB, vlForHalfNumber);
    uint16_t maxExponent;
    if constexpr (AscendC::Std::IsSame<T, fp8_e4m3fn_t>::value) {
        maxExponent = FP8_E4M3_MAX_EXP;
    } else if constexpr (AscendC::Std::IsSame<T, fp8_e5m2_t>::value) {
        maxExponent = FP8_E5M2_MAX_EXP;
    } else if constexpr (AscendC::Std::IsSame<T, fp4x2_e2m1_t>::value) {
        maxExponent = FP4_E2M1_BF16_MAX_EXP;
    } else {
        maxExponent = FP4_E1M2_BF16_MAX_EXP;
    }

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> expMask, vdMaxExp;
        AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::MaskReg cmpResult, zeroMask, preMaskScale;
        AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue;
        AscendC::MicroAPI::Duplicate(maxExpValue, maxExponent);
        AscendC::MicroAPI::RegTensor<uint16_t> sharedExp, scaleValue, scaleBias;
        AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::MicroAPI::RegTensor<uint16_t> halfScale, fp8NanRegTensor;
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::MicroAPI::RegTensor<uint16_t> zeroRegTensor;
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::RegTensor<uint16_t> nanRegTensor;
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::MicroAPI::MaskReg invalidDataMask, specialDataMask;
        AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalScaleInUB);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vdMaxExp, maxExpAddr, vlForHalfNumber);
            // 检测非正常值
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
            // 检测零值
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            // 检测超出目标格式的最大值
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::LE>(
                invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
            // 限制最大指数不超过目标格式最大指数
            AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
            // 计算相对指数差值
            AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            // 右移得到缩放值
            AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            // 特殊值处理
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

            AscendC::MicroAPI::DataCopy<
                uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, vlForHalfNumber / DIGIT_TWO, preMaskScale);

            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::EQ>(
                specialDataMask, sharedExp, scaleBias, preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, vlForHalfNumber, preMaskScale);
        }
    }
}

template <typename T, typename U, AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void ComputeFp8Data(
    __ubuf__ T* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr, __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(T);
    uint16_t elementAfterReduce = GetVRegSizeDispatch() / GetUbBlockSizeDispatch();
    uint32_t totalCountInUB2 = totalCountInUB * DIGIT_TWO;
    uint16_t loopNum = AscendC::Ceil(totalCountInUB, 2 * vlForHalfNumber);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1;
        AscendC::MicroAPI::MaskReg dataMask2;
        AscendC::MicroAPI::MaskReg dataMask3;
        AscendC::MicroAPI::MaskReg dataMask4;
        AscendC::MicroAPI::MaskReg maskAll =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<float> floatScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0;
        AscendC::MicroAPI::RegTensor<T> vdExp1;

        AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero;
        AscendC::MicroAPI::RegTensor<float> vdExp0FP32One;
        AscendC::MicroAPI::RegTensor<float> vdExp1FP32Zero;
        AscendC::MicroAPI::RegTensor<float> vdExp1FP32One;
        AscendC::MicroAPI::RegTensor<U> vdExp0FP8Zero;
        AscendC::MicroAPI::RegTensor<U> vdExp0FP8One;
        AscendC::MicroAPI::RegTensor<U> vdExp1FP8Zero;
        AscendC::MicroAPI::RegTensor<U> vdExp1FP8One;
        // 放到索引位置0
        static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        // 放到索引位置1
        static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTrait32to8 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            dataMask2 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            dataMask3 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB2);
            dataMask4 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB2);
            AscendC::MicroAPI::DataCopy<
                T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, vlForHalfNumber * DIGIT_TWO);
            // 这里DIST_E2B_B16是将每个16bit元素广播到一个DataBlock(32B)中
            AscendC::MicroAPI::DataCopy<
                uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, halfScaleLocalAddr, elementAfterReduce);
            // 因为前面scale计算用的BF16类型，所以对于fp16需要先cast到float再计算
            if constexpr (AscendC::Std::IsSame<T, half>::value) {
                // 取偶数索引的元素Cast到vdExp0FP32Zero
                AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                // 取奇数索引的元素Cast到vdExp0FP32One
                AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                // 由于前面搬入是一个广播操作，所以直接取偶数索引元素即可
                AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(
                    floatScaleForMul, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, maskAll);
                AscendC::MicroAPI::Mul(vdExp0FP32Zero, vdExp0FP32Zero, floatScaleForMul, dataMask3);
                AscendC::MicroAPI::Mul(vdExp0FP32One, vdExp0FP32One, floatScaleForMul, dataMask4);
                AscendC::MicroAPI::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
                AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask1);
                AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask1);
                AscendC::MicroAPI::Mul(vdExp1FP32Zero, vdExp1FP32Zero, floatScaleForMul, dataMask3);
                AscendC::MicroAPI::Mul(vdExp1FP32One, vdExp1FP32One, floatScaleForMul, dataMask4);
                AscendC::MicroAPI::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
                AscendC::MicroAPI::Interleave(vdExp0FP32Zero, vdExp1FP32Zero, vdExp0FP32Zero, vdExp1FP32Zero);
                AscendC::MicroAPI::Interleave(vdExp0FP32One, vdExp1FP32One, vdExp0FP32One, vdExp1FP32One);
                AscendC::MicroAPI::Cast<U, float, castTrait32to8>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
                AscendC::MicroAPI::Cast<U, float, castTrait32to8>(vdExp0FP8One, vdExp1FP32Zero, dataMask3);
                AscendC::MicroAPI::Cast<U, float, castTrait32to8>(vdExp1FP8Zero, vdExp0FP32One, dataMask4);
                AscendC::MicroAPI::Cast<U, float, castTrait32to8>(vdExp1FP8One, vdExp1FP32One, dataMask4);
            } else {
                AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                AscendC::MicroAPI::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
                AscendC::MicroAPI::Cast<U, float, castTrait32to8>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
                AscendC::MicroAPI::Cast<U, float, castTrait32to8>(vdExp0FP8One, vdExp0FP32One, dataMask3);
                AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
                AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
                AscendC::MicroAPI::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
                AscendC::MicroAPI::Cast<U, float, castTrait32to8>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
                AscendC::MicroAPI::Cast<U, float, castTrait32to8>(vdExp1FP8One, vdExp1FP32One, dataMask4);
            }
            AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask3);
            AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP8One, OUT_ELE_NUM_ONE_BLK, dataMask3);
            AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp1FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask4);
            AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp1FP8One, OUT_ELE_NUM_ONE_BLK, dataMask4);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void FP16Convert(
    AscendC::MicroAPI::RegTensor<half>& output, AscendC::MicroAPI::RegTensor<half>& input,
    AscendC::MicroAPI::MaskReg& mask)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> specialValueTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> newMantissa;
        AscendC::MicroAPI::RegTensor<uint16_t> andResult;
        AscendC::MicroAPI::RegTensor<uint16_t> newValue;
        AscendC::MicroAPI::MaskReg specialMask;
        AscendC::MicroAPI::MaskReg nonzeroMask;
        uint16_t specialValue = SPECIAL_VALUE_E1M2;
        if constexpr (AscendC::Std::IsSame<U, fp4x2_e2m1_t>::value) {
            specialValue = SPECIAL_VALUE_E2M1;
        }
        AscendC::MicroAPI::Duplicate(specialValueTensor, specialValue);
        AscendC::MicroAPI::Duplicate(newMantissa, NEW_MANTISSA);
        AscendC::MicroAPI::And(andResult, (AscendC::MicroAPI::RegTensor<uint16_t>&)input, specialValueTensor, mask);
        AscendC::MicroAPI::CompareScalar<uint16_t, AscendC::CMPMODE::GT>(nonzeroMask, andResult, 0, mask);
        AscendC::MicroAPI::CompareScalar<uint16_t, AscendC::CMPMODE::LT>(specialMask, andResult, NEW_MANTISSA, mask);
        AscendC::MicroAPI::MaskAnd(specialMask, specialMask, nonzeroMask, mask);
        AscendC::MicroAPI::Or(newValue, (AscendC::MicroAPI::RegTensor<uint16_t>&)input, newMantissa, mask);
        AscendC::MicroAPI::Select<uint16_t>(
            (AscendC::MicroAPI::RegTensor<uint16_t>&)output, newValue, (AscendC::MicroAPI::RegTensor<uint16_t>&)input,
            specialMask);
    }
}

template <typename T, typename U, AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void ComputeFp4Data(
    __ubuf__ T* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr, __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(T);
    uint16_t elementAfterReduce = GetVRegSizeDispatch() / GetUbBlockSizeDispatch();
    uint16_t loopNum = AscendC::Ceil(totalCountInUB, 2 * vlForHalfNumber);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1;
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0;
        AscendC::MicroAPI::RegTensor<T> vdExp1;

        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;

        AscendC::MicroAPI::RegTensor<U> vdExp0FP4;
        AscendC::MicroAPI::RegTensor<U> vdExp1FP4;

        static constexpr AscendC::MicroAPI::CastTrait castTrait = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, toBf16RoundMode};
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<
                T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, vlForHalfNumber * DIGIT_TWO);
            AscendC::MicroAPI::DataCopy<
                uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, halfScaleLocalAddr, elementAfterReduce);
            if constexpr (AscendC::Std::IsSame<T, half>::value) {
                if constexpr (roundMode == AscendC::RoundMode::CAST_RINT) {
                    FP16Convert<T, U>(vdExp0, vdExp0, dataMask1);
                    FP16Convert<T, U>(vdExp1, vdExp1, dataMask1);
                }
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, dataMask1);
                AscendC::MicroAPI::Mul(
                    vdExp0BF16, vdExp0BF16, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Mul(
                    vdExp1BF16, vdExp1BF16, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Interleave(vdExp0BF16, vdExp1BF16, vdExp0BF16, vdExp1BF16);
                AscendC::MicroAPI::Cast<U, bfloat16_t, castTrait>(vdExp0FP4, vdExp0BF16, dataMask1);
                AscendC::MicroAPI::Cast<U, bfloat16_t, castTrait>(vdExp1FP4, vdExp1BF16, dataMask1);
            } else {
                AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMask1);
            }
            AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
            AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp1FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
        }
    }
}

template <typename T, typename U, AscendC::RoundMode RMode, bool HasSmooth>
__aicore__ inline void ComputePerTileDynamic(
    __ubuf__ T* srcAddr, __ubuf__ float* smoothLocalAddr, __ubuf__ float* scaleOutLocalAddr,
    __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUB)
{
    uint32_t vlB16 = GetVRegSizeDispatch() / sizeof(T);
    uint32_t vlB32 = GetVRegSizeDispatch() / sizeof(float);
    uint16_t loopNum = AscendC::Ceil(totalCountInUB, vlB16);
    uint32_t totalCntForB32 = totalCountInUB;
    float maxVal = 0.0f;
    if constexpr (AscendC::Std::IsSame<U, fp8_e5m2_t>::value) {
        maxVal = FP8_E5M2_MAX_VALUE;
    } else if constexpr (AscendC::Std::IsSame<U, fp8_e4m3fn_t>::value) {
        maxVal = FP8_E4M3_MAX_VALUE;
    } else if constexpr (AscendC::Std::IsSame<U, hifloat8_t>::value) {
        maxVal = HIFP8_MAX_VALUE;
    } else if constexpr (AscendC::Std::IsSame<U, int8_t>::value) {
        maxVal = INT8_MAX_VALUE;
    }

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1;
        AscendC::MicroAPI::MaskReg dataMask2;
        AscendC::MicroAPI::MaskReg dataMask3;
        AscendC::MicroAPI::MaskReg maskAll =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg maskOne =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();

        AscendC::MicroAPI::RegTensor<T> vInB16;

        AscendC::MicroAPI::RegTensor<float> vInFP32Zero;
        AscendC::MicroAPI::RegTensor<float> vInFP32One;
        AscendC::MicroAPI::RegTensor<float> vSmooth0;
        AscendC::MicroAPI::RegTensor<float> vSmooth1;
        AscendC::MicroAPI::RegTensor<float> vTileMax;
        AscendC::MicroAPI::RegTensor<float> vDynScale;
        AscendC::MicroAPI::RegTensor<float> vMaxVal;
        AscendC::MicroAPI::RegTensor<float> vOneVal;

        AscendC::MicroAPI::RegTensor<U> vOut0;
        AscendC::MicroAPI::RegTensor<U> vOut1;

        AscendC::MicroAPI::Duplicate(vMaxVal, maxVal);
        AscendC::MicroAPI::Duplicate(vOneVal, 1.0f);

        static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

        constexpr static AscendC::MicroAPI::CastTrait castTrait32tof8 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RMode};

        constexpr static AscendC::MicroAPI::CastTrait castTrait32tof16 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

        constexpr static AscendC::MicroAPI::CastTrait castTrait16toi8 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};

        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            dataMask2 = AscendC::MicroAPI::UpdateMask<float>(totalCntForB32);
            dataMask3 = AscendC::MicroAPI::UpdateMask<float>(totalCntForB32);

            AscendC::MicroAPI::DataCopy(vInB16, srcAddr + i * vlB16);
            AscendC::MicroAPI::Cast<float, T, castTraitZero>(vInFP32Zero, vInB16, dataMask1);
            AscendC::MicroAPI::Cast<float, T, castTraitOne>(vInFP32One, vInB16, dataMask1);
            if constexpr (HasSmooth) {
                AscendC::MicroAPI::DataCopy<
                    float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    vSmooth0, vSmooth1, smoothLocalAddr, vlB32 * DIGIT_TWO);
                AscendC::MicroAPI::Mul(vInFP32Zero, vInFP32Zero, vSmooth0, maskAll);
                AscendC::MicroAPI::Mul(vInFP32One, vInFP32One, vSmooth1, maskAll);
            }
            AscendC::MicroAPI::Interleave(vSmooth0, vSmooth1, vInFP32Zero, vInFP32One);
            AscendC::MicroAPI::Abs(vInFP32Zero, vSmooth0, maskAll);
            AscendC::MicroAPI::Abs(vInFP32One, vSmooth1, maskAll);
            AscendC::MicroAPI::Max(vTileMax, vInFP32Zero, vInFP32One, maskAll);
            AscendC::MicroAPI::ReduceMax(vTileMax, vTileMax, dataMask2);
            AscendC::MicroAPI::Duplicate(vTileMax, vTileMax, maskAll);
            AscendC::MicroAPI::Div(vDynScale, vMaxVal, vTileMax, maskAll);
            AscendC::MicroAPI::Mul(vSmooth0, vSmooth0, vDynScale, maskAll);
            AscendC::MicroAPI::Mul(vSmooth1, vSmooth1, vDynScale, maskAll);

            if constexpr (AscendC::Std::IsSame<U, int8_t>::value) {
                AscendC::MicroAPI::RegTensor<half> vHalf0;
                AscendC::MicroAPI::RegTensor<half> vHalf1;
                AscendC::MicroAPI::Cast<half, float, castTrait32tof16>(vHalf0, vSmooth0, maskAll);
                AscendC::MicroAPI::Cast<half, float, castTrait32tof16>(vHalf1, vSmooth1, maskAll);
                AscendC::MicroAPI::Cast<U, half, castTrait16toi8>(vOut0, vHalf0, maskAll);
                AscendC::MicroAPI::Cast<U, half, castTrait16toi8>(vOut1, vHalf1, maskAll);
            } else {
                AscendC::MicroAPI::Cast<U, float, castTrait32tof8>(vOut0, vSmooth0, dataMask2);
                AscendC::MicroAPI::Cast<U, float, castTrait32tof8>(vOut1, vSmooth1, dataMask3);
            }

            AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vOut0, OUT_ELE_NUM_ONE_BLK, dataMask2);
            AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vOut1, OUT_ELE_NUM_ONE_BLK, dataMask3);

            AscendC::MicroAPI::Div(vDynScale, vOneVal, vDynScale, maskAll);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                scaleOutLocalAddr + i, vDynScale, maskOne);
        }
    }
}

} // namespace quant
#endif // SHMEM_EXAMPLES_MEGA_MOE_KERNEL_DISPATCH_QUANTIZE_FUNCTIONS_H
