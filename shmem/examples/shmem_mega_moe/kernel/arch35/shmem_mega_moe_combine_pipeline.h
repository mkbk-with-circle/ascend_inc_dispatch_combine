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
 * \file shmem_mega_moe_combine_pipeline.h
 * \brief
 */

#ifndef SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_COMBINE_PIPELINE_H
#define SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_COMBINE_PIPELINE_H

#include "shmem_ascendc_compat.h"
#include "shmem_mega_moe_types.h"
#include "dispatch/quantize_functions.h"
namespace ShmemMegaMoeKernel {
using namespace AscendC;

template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokens(
    uint32_t rowOffset, uint32_t columnOffset, uint32_t n, LocalTensor<int32_t>& routingMetadata,
    LocalTensor<ElementMMadOut2>& secondProjectionOutput, BlockShape& actualBlockShape,
    const ShmemMegaMoeKernelParams& params)
{
    int32_t tileRowCount = Get<SHAPE_ROW_INDEX>(actualBlockShape);
    AscendC::GlobalTensor<ElementMMadOut2> remoteOutput;
    uint64_t combinedTokenOffset = params.symmetricMemory.combinedTokens - params.symmetricMemory.localBase;
    AscendC::DataCopyExtParams outputCopyParams{1, 0, 0, 0, 0};
    outputCopyParams.blockCount = 1;
    outputCopyParams.blockLen = Get<SHAPE_COLUMN_INDEX>(actualBlockShape) * sizeof(ElementMMadOut2);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    for (int32_t tileRowIndex = 0; tileRowIndex < tileRowCount; ++tileRowIndex) {
        uint32_t routingRecordOffset = tileRowIndex * ROUTING_RECORD_WIDTH;
        uint32_t destinationRank = routingMetadata.GetValue(routingRecordOffset + RANK_ID);
        uint32_t tokenIndex = routingMetadata.GetValue(routingRecordOffset + TOKEN_ID);
        uint32_t routeSlotIndex = routingMetadata.GetValue(routingRecordOffset + ROUTE_SLOT_INDEX);
        remoteOutput.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(
            GetRemoteAddress(params.symmetricMemory.localBase, destinationRank, combinedTokenOffset)));
        uint64_t destinationOffset =
            (static_cast<uint64_t>(tokenIndex) * params.config->expertsPerToken + routeSlotIndex) * n + columnOffset;
        AscendC::DataCopyPad(
            remoteOutput[destinationOffset],
            secondProjectionOutput[tileRowIndex * Get<SHAPE_COLUMN_INDEX>(actualBlockShape)], outputCopyParams);
    }
}

template <uint8_t QuantMode, typename ExpandXType>
__aicore__ inline void QuantizeMxFp8(
    LocalTensor<ExpandXType>& quantizedBuffer, LocalTensor<ExpandXType>& inputBuffer,
    LocalTensor<float>& quantizationScratch, int32_t elementCount)
{
    PipeBarrier<PIPE_V>();
    uint32_t scaleCount = Align2(Ceil32(elementCount));
    using Fp8Type = typename std::conditional<QuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
    LocalTensor<Fp8Type> fp8Buffer = quantizedBuffer.template ReinterpretCast<Fp8Type>();
    __ubuf__ ExpandXType* inputData = (__ubuf__ ExpandXType*)inputBuffer.GetPhyAddr();
    __ubuf__ uint16_t* maxExponentData = (__ubuf__ uint16_t*)quantizationScratch.GetPhyAddr();
    __ubuf__ uint16_t* inverseScaleData = (__ubuf__ uint16_t*)quantizationScratch[Align32(scaleCount)].GetPhyAddr();
    __ubuf__ int8_t* quantizedData = (__ubuf__ int8_t*)fp8Buffer.GetPhyAddr();
    __ubuf__ uint16_t* encodedScaleData = (__ubuf__ uint16_t*)fp8Buffer[elementCount].GetPhyAddr();
    quant::ComputeMaxExp(inputData, maxExponentData, elementCount);

    quant::ComputeScale<Fp8Type>(maxExponentData, encodedScaleData, inverseScaleData, scaleCount);
    quant::ComputeFp8Data<ExpandXType, Fp8Type, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
        inputData, inverseScaleData, quantizedData, elementCount);
}

template <typename T, typename XType>
__aicore__ inline void DequantizeMxFp8(
    LocalTensor<XType>& inputBuffer, LocalTensor<float>& dequantizedValues, LocalTensor<bfloat16_t>& bf16ScaleBuffer,
    LocalTensor<float>& fp32ScaleBuffer, uint32_t scaleLen, uint32_t tokenLen)
{
    LocalTensor<T> fp8Buffer = inputBuffer.template ReinterpretCast<T>();
    LocalTensor<AscendC::fp8_e8m0_t> encodedScales =
        inputBuffer[Align256<uint32_t>(tokenLen) / 2].template ReinterpretCast<AscendC::fp8_e8m0_t>();
    __ubuf__ bfloat16_t* bf16ScaleData = (__ubuf__ bfloat16_t*)bf16ScaleBuffer.GetPhyAddr();
    __ubuf__ float* fp32ScaleData = (__ubuf__ float*)fp32ScaleBuffer.GetPhyAddr();
    __ubuf__ AscendC::fp8_e8m0_t* encodedScaleInput = (__ubuf__ AscendC::fp8_e8m0_t*)encodedScales.GetPhyAddr();
    __ubuf__ T* quantizedTokenInput = (__ubuf__ T*)fp8Buffer.GetPhyAddr();
    __ubuf__ float* dequantizedOutput = (__ubuf__ float*)dequantizedValues.GetPhyAddr();
    uint32_t bf16ElementsPerRepeat = quant::GetVRegSizeDispatch() / sizeof(bfloat16_t);
    uint32_t fp32ElementsPerRepeat = quant::GetVRegSizeDispatch() / sizeof(float);
    uint16_t scaleRepeatCount = Ceil(scaleLen, bf16ElementsPerRepeat);
    uint16_t tokenRepeatCount = Ceil(tokenLen, fp32ElementsPerRepeat);
    uint16_t expandedScaleRepeatCount = Ceil(scaleLen * 2, fp32ElementsPerRepeat);
    uint32_t expandedScaleCount = scaleLen * 2;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<AscendC::fp8_e8m0_t> encodedScaleRegister;
        AscendC::MicroAPI::RegTensor<T> quantizedTokenRegister;
        AscendC::MicroAPI::RegTensor<float> tokenFp32Register;
        AscendC::MicroAPI::RegTensor<bfloat16_t> bf16ScaleRegister;
        AscendC::MicroAPI::RegTensor<bfloat16_t> bf16ScaleLoadRegister;
        AscendC::MicroAPI::RegTensor<float> fp32ScaleRegister;
        AscendC::MicroAPI::RegTensor<float> dequantizedRegister;
        AscendC::MicroAPI::RegTensor<float> scaledTokenRegister;
        static constexpr AscendC::MicroAPI::CastTrait FP82BF16CastTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait FP162FP32CastTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        AscendC::MicroAPI::MaskReg scaleLoadMask;
        AscendC::MicroAPI::MaskReg scaleConversionMask;
        AscendC::MicroAPI::MaskReg tokenConversionMask;
        for (uint16_t i = 0; i < scaleRepeatCount; i++) {
            scaleLoadMask = AscendC::MicroAPI::UpdateMask<bfloat16_t>(scaleLen);
            MicroAPI::DataCopy<AscendC::fp8_e8m0_t, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                encodedScaleRegister, encodedScaleInput + i * bf16ElementsPerRepeat);
            MicroAPI::Cast<bfloat16_t, AscendC::fp8_e8m0_t, FP82BF16CastTraitZero>(
                bf16ScaleRegister, encodedScaleRegister, scaleLoadMask);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
                bf16ScaleData + i * bf16ElementsPerRepeat * 2, bf16ScaleRegister, bf16ScaleRegister, scaleLoadMask);
        }
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < expandedScaleRepeatCount; i++) {
            scaleConversionMask = AscendC::MicroAPI::UpdateMask<float>(expandedScaleCount);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                bf16ScaleLoadRegister, bf16ScaleData + i * fp32ElementsPerRepeat);
            MicroAPI::Cast<float, bfloat16_t, FP162FP32CastTraitZero>(
                fp32ScaleRegister, bf16ScaleLoadRegister, scaleConversionMask);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(
                fp32ScaleData + i * fp32ElementsPerRepeat * 2, fp32ScaleRegister, fp32ScaleRegister,
                scaleConversionMask);
        }
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < tokenRepeatCount; i++) {
            tokenConversionMask = AscendC::MicroAPI::UpdateMask<float>(tokenLen);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(fp32ScaleRegister, fp32ScaleData + i * 8);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                quantizedTokenRegister, quantizedTokenInput + i * fp32ElementsPerRepeat);
            MicroAPI::Cast<float, T, FP82BF16CastTraitZero>(
                tokenFp32Register, quantizedTokenRegister, tokenConversionMask);
            MicroAPI::Mul(scaledTokenRegister, fp32ScaleRegister, tokenFp32Register, tokenConversionMask);
            MicroAPI::DataCopy(dequantizedOutput + i * fp32ElementsPerRepeat, scaledTokenRegister, tokenConversionMask);
        }
    }
}

template <typename QuantOutType>
__aicore__ inline void CombineQuantizedTokens(
    uint32_t batchStart, uint32_t rowCount, uint32_t n, uint32_t scaleCountPerRow, uint32_t expertIndex,
    uint32_t rankId, LocalTensor<int32_t>& routingMetadata, LocalTensor<QuantOutType>& quantizedToken,
    const ShmemMegaMoeKernelParams& params)
{
    uint64_t quantRowSize = static_cast<uint64_t>(n) + scaleCountPerRow;
    uint32_t destinationRank = routingMetadata.GetValue(batchStart * ROUTING_RECORD_WIDTH + RANK_ID);
    uint32_t tokenIndex = routingMetadata.GetValue(batchStart * ROUTING_RECORD_WIDTH + TOKEN_ID);
    uint32_t routeSlotIndex = routingMetadata.GetValue(batchStart * ROUTING_RECORD_WIDTH + ROUTE_SLOT_INDEX);

    AscendC::GlobalTensor<QuantOutType> remoteOutput;
    uint64_t combinedTokenOffset = params.symmetricMemory.combinedTokens - params.symmetricMemory.localBase;
    __gm__ void* remoteAddress =
        GetRemoteAddress(params.symmetricMemory.localBase, destinationRank, combinedTokenOffset);
    remoteOutput.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(remoteAddress));

    uint64_t destinationOffset =
        (static_cast<uint64_t>(tokenIndex) * params.config->expertsPerToken + routeSlotIndex) * quantRowSize;

    AscendC::DataCopyExtParams rowCopyConfig{1, static_cast<uint32_t>(quantRowSize), 0, 0, 0};
    AscendC::DataCopyPad(remoteOutput[destinationOffset], quantizedToken, rowCopyConfig);
}

template <uint8_t QuantMode, typename T>
__aicore__ inline void CombineRowGroup(
    uint32_t rowStart, uint32_t rowCount, uint32_t n, uint32_t expertIndex, uint32_t rankId,
    GM_ADDR secondProjectionOutputAddress, const ShmemMegaMoeKernelParams& params,
    LocalTensor<int32_t>& routingMetadata, int64_t bufferElementCount, int64_t offset, uint32_t quantRowSizeBytes)
{
    LocalTensor<T> combineBuffer(TPosition::VECIN, offset, bufferElementCount);
    offset += bufferElementCount * sizeof(T);

    uint32_t scaleCountPerRow = ShmemMegaMoeKernel::CeilDiv(n, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t alignedColumnCount = ShmemMegaMoeKernel::AlignUp(n, static_cast<uint32_t>(ALIGN_32));
    uint32_t alignedScaleCount = AscendC::Align32(scaleCountPerRow);
    uint32_t quantizationScratchBytes = alignedScaleCount * sizeof(float) + scaleCountPerRow * sizeof(uint16_t);
    uint32_t quantizationScratchElements =
        ShmemMegaMoeKernel::CeilDiv(quantizationScratchBytes, static_cast<uint32_t>(sizeof(float)));
    LocalTensor<float> quantizationScratch = LocalTensor<float>(TPosition::VECIN, offset, quantizationScratchElements);

    GlobalTensor<T> secondProjectionOutputGlobal;
    secondProjectionOutputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(secondProjectionOutputAddress));

    using Fp8Type = typename std::conditional<QuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;

    uint32_t elementsPerBuffer = (alignedColumnCount * sizeof(T) + quantRowSizeBytes) / sizeof(T);
    DataCopyPadExtParams<T> paddingConfig{false, 0U, 0U, 0U};
    AscendC::DataCopyExtParams inputCopyParams{1U, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};

    for (uint32_t i = 0; i < rowCount; i++) {
        uint32_t bufferIndex = i % 2;
        LocalTensor<T> tokenBuffer = combineBuffer[bufferIndex * elementsPerBuffer];
        LocalTensor<T> quantizedTokenBuffer = tokenBuffer[alignedColumnCount];

        // MTE2: read from GM
        SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID3>();
        AscendC::DataCopyPad(
            tokenBuffer, secondProjectionOutputGlobal[(rowStart + i) * n], inputCopyParams, paddingConfig);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID4>();

        // V: quantize
        QuantizeMxFp8<QuantMode, T>(quantizedTokenBuffer, tokenBuffer, quantizationScratch, n);
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID5>();

        // MTE3: send to GM
        LocalTensor<Fp8Type> quantizedTokenFp8 = quantizedTokenBuffer.template ReinterpretCast<Fp8Type>();
        CombineQuantizedTokens<Fp8Type>(
            i, 1, n, scaleCountPerRow, expertIndex, rankId, routingMetadata, quantizedTokenFp8, params);
    }

    // Wait for all MTE3 operations to complete
    SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
}

} // namespace ShmemMegaMoeKernel

#endif // SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_COMBINE_PIPELINE_H
