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
 * \file shmem_mega_moe_kernel.h
 * \brief
 */

#ifndef SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_KERNEL_H
#define SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_KERNEL_H

#include "shmem_ascendc_compat.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "shmem_mega_moe_types.h"
#include "shmem_mega_moe_memory_layout.h"
#include "shmem_mega_moe_swiglu_epilogue.h"
#include "shmem_mega_moe_catlass_pipeline.h"
#include "dispatch/quantize_functions.h"

namespace ShmemMegaMoeKernel {
using namespace AscendC;

using ShmemMegaMoeProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
using ShmemMegaMoeBufferOffsets = AscendC::Shape<
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;

#define SHMEM_MEGA_MOE_TEMPLATE_DECL \
    typename XType, typename OutputType, typename RoutingWeightType, int32_t QuantMode, int32_t CombineQuantMode
#define SHMEM_MEGA_MOE_TEMPLATE_ARGS XType, OutputType, RoutingWeightType, QuantMode, CombineQuantMode

template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
class ShmemMegaMoePipeline {
public:
    template <int32_t QM>
    struct QuantTraits {
        using OutType = fp8_e4m3fn_t;
    };
    template <>
    struct QuantTraits<E5M2_QUANT> {
        using OutType = fp8_e5m2_t;
    };
    template <>
    struct QuantTraits<E2M1_QUANT> {
        using OutType = fp4x2_e2m1_t;
    };
    using QuantOutType = typename QuantTraits<QuantMode>::OutType;
    using ActivationType =
        typename std::conditional<Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value, uint8_t, QuantOutType>::type;
    using QuantScaleOutType = typename std::conditional<(QuantMode >= E5M2_QUANT), AscendC::fp8_e8m0_t, float>::type;
    struct ExpertPipelineState {
        ShmemMegaMoeProblemShape problemShape;
        ShmemMegaMoeBufferOffsets baseOffset;
    };
    __aicore__ inline ShmemMegaMoePipeline(){};
    __aicore__ inline void Init(
        GM_ADDR symmetricMemoryBase, GM_ADDR x, GM_ADDR expertIds, GM_ADDR routingWeights, GM_ADDR weight1,
        GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales,
        GM_ADDR yOut, GM_ADDR expertTokenCountsOut, GM_ADDR workspaceGM, ShmemMegaMoeTilingData* config);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitDispatchBuffers();
    __aicore__ inline void InitSendQuantBuffers();
    __aicore__ inline void InitOutputBuffers();
    __aicore__ inline void ResetPipelineFlags();
    __aicore__ inline void ResetRowGroupFlags();
    __aicore__ inline void BuildSendMasks();
    __aicore__ inline void CountReceivedTokens(int32_t localExpertId);
    __aicore__ inline void BuildRoutingMetadataAndDispatch(
        ShmemMegaMoeMatmulBuffers& matmulBuffers, int32_t localExpertId);
    template <ShmemMegaMoeMatmulStage Mode>
    __aicore__ inline bool PrepareExpertProblem(ExpertPipelineState& state, uint32_t expertIndex);
    template <ShmemMegaMoeMatmulStage Mode>
    __aicore__ inline void UpdateMatmulBuffers(
        ShmemMegaMoeMatmulBuffers& matmulBuffers, const ExpertPipelineState& state);
    __aicore__ inline void CombineRoutedTokens();
    __aicore__ inline void InitCombineBuffers();
    __aicore__ inline void RunCombine(
        ShmemMegaMoeMatmulBuffers& matmulBuffers, ExpertPipelineState& secondProjectionState, uint32_t expertIndex);
    __aicore__ inline void SynchronizeRanks();
    __aicore__ inline void CopyExpertTokenCounts();
    __aicore__ inline void CopyQuantizedTokens(
        int32_t destinationRowOffset, int32_t sourceRank, int32_t firstRouteIndex, int32_t routeCopyCount);
    __aicore__ inline void QuantizeLocalTokens();

    ShmemMegaMoeKernelParams params_{};

    GlobalTensor<int32_t> pipelineFlags_;
    GlobalTensor<int32_t> expertTokenCountsOut_;
    GlobalTensor<int32_t> routingMetadataGlobal_;
    GlobalTensor<int32_t> receivedTokenCountsGlobal_;

    uint32_t tokenCount_ = 0;
    uint32_t modelDim_ = 0;
    uint32_t cubeCoreCount_ = 0;
    uint32_t expertsPerToken_ = 0;
    uint32_t rankId_ = 0;
    uint32_t rankCount_ = 0;
    uint32_t localExpertCount_ = 0;
    int64_t ffnDim_ = 0;
    uint64_t maxReceivedTokens_ = 0;
    int32_t vectorSyncState_ = 0;
    uint32_t schedulerStartBlock_ = 0;
    uint32_t cubeBlocksPerRank_ = 2;
    int32_t dispatchFlagSlotsPerExpert_ = 0;
    int32_t maxWavesPerExpert_ = 0;
    uint32_t cubeBlockCount_ = GetBlockNum();
    uint32_t vectorCoreCount_ = GetBlockNum() * 2;
    uint32_t cubeBlockIndex_ = GetBlockIdx() / GetTaskRation();
    uint32_t vectorCoreIndex_ = GetBlockIdx();
    uint32_t vectorSubBlockIndex_ = GetSubBlockIdx();
    uint32_t scaleCountPerToken_ = 0;
    uint32_t alignedQuantizedTokenBytes_ = 0;
    uint32_t alignedScaleBytes_ = 0;
    uint32_t quantizedRowBytes_ = 0;
    uint32_t processedTokenCount_ = 0;
    uint32_t nextUbOffset_ = 0;
    uint16_t secondProjectionBufferIndex_ = 0;
    uint64_t routeCount_ = 0;
    uint32_t alignedMaskBytes_ = 0;
    uint32_t maskSlotBytes_ = 0;
    uint64_t inboundMaskOffset_ = 0;
    uint64_t quantizedTokenOffset_ = 0;
    uint64_t receivedTokenCount_ = 0;
    uint64_t receivedTokenPrefix_ = 0;
    int32_t paddedRouteCount_ = 0;
    int64_t expertCounterStride_ = 0;
    int64_t combineBufferElementCount_ = 0;

    static constexpr uint32_t ELEMS_PER_BYTE = (QuantMode == E2M1_QUANT) ? 2 : 1;
    static constexpr int32_t DISPATCH_BUFFER_NUM = 6;
    LocalTensor<int32_t> routeIndices_;
    LocalTensor<uint8_t> receivedMaskBytes_;
    LocalTensor<uint32_t> receivedMaskWords_;
    LocalTensor<int32_t> currentExpertTokenCountBuffer_;
    LocalTensor<int32_t> selectedRoutes_;
    LocalTensor<int32_t> receivedTokenPrefixBuffer_;
    LocalTensor<ActivationType> dispatchCopyBuffers_[DISPATCH_BUFFER_NUM];
    LocalTensor<int32_t> routingMetadata_;
    LocalTensor<bfloat16_t> quantInputBuffer0_;
    LocalTensor<bfloat16_t> quantInputBuffer1_;
    LocalTensor<ActivationType> quantOutputBuffer0_;
    LocalTensor<ActivationType> quantOutputBuffer1_;
    LocalTensor<uint16_t> quantizationScratch_;
    LocalTensor<int32_t> resetBuffer_;
    LocalTensor<int32_t> expertIdsBuffer_;
    LocalTensor<uint8_t> outboundMaskBuffers_[DOUBLE_BUFFER];
    LocalTensor<int32_t> maskCountBuffer_;
    LocalTensor<int32_t> expertTokenCountBuffer_;
    LocalTensor<bfloat16_t> combinedOutputBf16_;
    LocalTensor<float> combinedOutputFp32_;
    LocalTensor<float> routingWeightsBuffer_;
    LocalTensor<float> fp32ScaleBuffer_;
    LocalTensor<bfloat16_t> bf16ScaleBuffer_;

    using BlockEpilogue =
        ShmemMegaMoeSwiGluEpilogue<QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true>;
    BlockEpilogue epilogueOp_;
};

template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::Init(
    GM_ADDR symmetricMemoryBase, GM_ADDR x, GM_ADDR expertIds, GM_ADDR routingWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR yOut,
    GM_ADDR expertTokenCountsOut, GM_ADDR workspaceGM, ShmemMegaMoeTilingData* config)
{
    tokenCount_ = config->tokenCount;
    modelDim_ = config->modelDim;
    cubeCoreCount_ = config->cubeCoreCount;
    expertsPerToken_ = config->expertsPerToken;
    routeCount_ = tokenCount_ * expertsPerToken_;
    localExpertCount_ = config->localExpertCount;
    cubeBlocksPerRank_ = config->cubeBlocksPerRank;
    maxReceivedTokens_ = config->maxReceivedTokens;

    maxWavesPerExpert_ = static_cast<int32_t>(
        ShmemMegaMoeKernel::CeilDiv(static_cast<int64_t>(maxReceivedTokens_), DISPATCH_WAVE_TILE_M));
    dispatchFlagSlotsPerExpert_ = static_cast<int32_t>(
        ShmemMegaMoeKernel::AlignUp(static_cast<int64_t>(maxWavesPerExpert_), static_cast<int64_t>(INT_CACHELINE)));
    ffnDim_ = config->ffnDim;
    rankId_ = static_cast<uint32_t>(aclshmem_my_pe());
    rankCount_ = static_cast<uint32_t>(aclshmem_n_pes());
    params_.input = x;
    params_.expertIds = expertIds;
    params_.weight1 = weight1;
    params_.weight2 = weight2;
    params_.weight1Scales = weightScales1;
    params_.weight2Scales = weightScales2;

    params_.output = yOut;
    params_.expertTokenCountsOut = expertTokenCountsOut;
    params_.routingWeights = routingWeights;
    params_.workspace = ShmemMegaMoeWorkspaceLayout(workspaceGM, config);
    params_.symmetricMemory = ShmemMegaMoeSymmetricMemoryLayout(symmetricMemoryBase, config, ELEMS_PER_BYTE);
    params_.config = config;
    expertTokenCountsOut_.SetGlobalBuffer((__gm__ int32_t*)params_.expertTokenCountsOut);
    receivedTokenCountsGlobal_.SetGlobalBuffer((__gm__ int32_t*)params_.workspace.receivedTokenCounts);
    routingMetadataGlobal_.SetGlobalBuffer((__gm__ int32_t*)params_.workspace.routingMetadata);
    epilogueOp_.Init(
        {params_.workspace.activatedTokenData, params_.workspace.activatedTokenScales,
         params_.workspace.activationReadyFlags, ALIGN_256});
    // All ranks use the same symmetric-memory layout, so remote accesses reuse these local offsets.
    inboundMaskOffset_ = static_cast<uint64_t>(params_.symmetricMemory.inboundMask - params_.symmetricMemory.localBase);
    quantizedTokenOffset_ =
        static_cast<uint64_t>(params_.symmetricMemory.quantizedTokens - params_.symmetricMemory.localBase);
    // alignedMaskBytes_ must match the formula in ShmemMegaMoeSymmetricMemoryLayout.
    paddedRouteCount_ = ShmemMegaMoeKernel::AlignUp(
                            static_cast<int64_t>(routeCount_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_256)) /
                        sizeof(int32_t);
    alignedMaskBytes_ =
        ShmemMegaMoeKernel::AlignUp(static_cast<int64_t>(paddedRouteCount_) / 8, static_cast<int64_t>(ALIGN_32));
    // Each rank slot appends a 32-byte count and must match the layout calculated by ShmemMegaMoeSymmetricMemoryLayout.
    maskSlotBytes_ = alignedMaskBytes_ + static_cast<uint32_t>(ALIGN_32);
    scaleCountPerToken_ = ShmemMegaMoeKernel::CeilDiv(modelDim_, static_cast<uint32_t>(ALIGN_32));
    alignedQuantizedTokenBytes_ =
        ShmemMegaMoeKernel::AlignUp(
            static_cast<uint32_t>(modelDim_ / ELEMS_PER_BYTE), static_cast<uint32_t>(ALIGN_256)) *
        sizeof(ActivationType);
    alignedScaleBytes_ = scaleCountPerToken_ * sizeof(uint8_t);
    quantizedRowBytes_ =
        ShmemMegaMoeKernel::AlignUp(alignedQuantizedTokenBytes_ + alignedScaleBytes_, static_cast<uint32_t>(ALIGN_32));
}

template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::InitDispatchBuffers()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    uint32_t currentExpertCountOffset = 0;
    uint32_t currentExpertTokenCountBytes = ALIGN_32;
    currentExpertTokenCountBuffer_ = LocalTensor<int32_t>(
        TPosition::VECCALC, currentExpertCountOffset, currentExpertTokenCountBytes / sizeof(int32_t));

    uint32_t tokenPrefixOffset = currentExpertCountOffset + currentExpertTokenCountBytes;
    uint32_t receivedTokenPrefixBytes = ShmemMegaMoeKernel::AlignUp(
        static_cast<int64_t>(rankCount_ * localExpertCount_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    receivedTokenPrefixBuffer_ =
        LocalTensor<int32_t>(TPosition::VECCALC, tokenPrefixOffset, receivedTokenPrefixBytes / sizeof(int32_t));

    uint32_t receivedMaskOffset = tokenPrefixOffset + receivedTokenPrefixBytes;
    uint32_t receivedMaskBufferBytes = maskSlotBytes_ * rankCount_;
    receivedMaskBytes_ =
        LocalTensor<uint8_t>(TPosition::VECCALC, receivedMaskOffset, receivedMaskBufferBytes / sizeof(uint8_t));
    receivedMaskWords_ =
        LocalTensor<uint32_t>(TPosition::VECCALC, receivedMaskOffset, receivedMaskBufferBytes / sizeof(uint32_t));

    uint32_t selectedRouteBufferOffset = receivedMaskOffset + receivedMaskBufferBytes;
    uint32_t selectedRouteIndexBytes = ShmemMegaMoeKernel::AlignUp(
        static_cast<int64_t>(routeCount_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    selectedRoutes_ =
        LocalTensor<int32_t>(TPosition::VECCALC, selectedRouteBufferOffset, selectedRouteIndexBytes / sizeof(int32_t));

    uint32_t routeIndexBufferOffset = selectedRouteBufferOffset + selectedRouteIndexBytes;
    uint32_t routeIndexBytes = ShmemMegaMoeKernel::AlignUp(
        static_cast<int64_t>(routeCount_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    routeIndices_ = LocalTensor<int32_t>(TPosition::VECCALC, routeIndexBufferOffset, routeIndexBytes / sizeof(int32_t));

    constexpr uint32_t COPY_TMP_BUFFER_SIZE = TOKEN_SCALE_SIZE;
    uint32_t dispatchBufferOffset = routeIndexBufferOffset + routeIndexBytes;
    for (int32_t index = 0; index < DISPATCH_BUFFER_NUM; ++index) {
        dispatchCopyBuffers_[index] = LocalTensor<ActivationType>(
            TPosition::VECCALC, dispatchBufferOffset + static_cast<uint32_t>(index) * COPY_TMP_BUFFER_SIZE,
            COPY_TMP_BUFFER_SIZE / sizeof(ActivationType));
    }

    uint32_t expertCountBufferOffset =
        dispatchBufferOffset + static_cast<uint32_t>(DISPATCH_BUFFER_NUM) * COPY_TMP_BUFFER_SIZE;
    uint32_t expertTokenNumsOutTensorSize = ShmemMegaMoeKernel::AlignUp(
        static_cast<int64_t>(localExpertCount_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    expertTokenCountBuffer_ = LocalTensor<int32_t>(
        TPosition::VECCALC, expertCountBufferOffset, expertTokenNumsOutTensorSize / sizeof(int32_t));

    nextUbOffset_ = expertCountBufferOffset + expertTokenNumsOutTensorSize;
    CreateVecIndex(routeIndices_, 0, (routeIndexBytes / sizeof(int32_t)));
    Duplicate<int32_t>(receivedTokenPrefixBuffer_, 0, (receivedTokenPrefixBytes / sizeof(int32_t)));
    PipeBarrier<PIPE_ALL>();
}

template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::InitSendQuantBuffers()
{
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        expertCounterStride_ = ShmemMegaMoeKernel::CalculateRowGroupStride(tokenCount_, rankCount_, vectorCoreCount_);
    }
    if constexpr (g_coreType == AIC) {
        return;
    }

    uint32_t expertIdBufferOffset = 0;
    uint32_t expertIdBufferBytes = ShmemMegaMoeKernel::AlignUp(
        static_cast<int64_t>(routeCount_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_256));
    expertIdsBuffer_ =
        LocalTensor<int32_t>(TPosition::VECCALC, expertIdBufferOffset, expertIdBufferBytes / sizeof(int32_t));

    uint32_t resetBufferOffset = expertIdBufferOffset + expertIdBufferBytes;
    uint64_t totalFlagInt32 =
        static_cast<uint64_t>(localExpertCount_) *
        (static_cast<uint64_t>(INT_CACHELINE) + static_cast<uint64_t>(dispatchFlagSlotsPerExpert_));
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        int64_t rowGroupResetSize = static_cast<int64_t>(localExpertCount_) * expertCounterStride_;
        totalFlagInt32 = (static_cast<int64_t>(totalFlagInt32) > rowGroupResetSize) ?
                             static_cast<int64_t>(totalFlagInt32) :
                             rowGroupResetSize;
    }
    uint32_t resetNumPerCore = ShmemMegaMoeKernel::CeilDiv(totalFlagInt32, static_cast<uint64_t>(vectorCoreCount_));
    uint32_t resetBufferBytes =
        ShmemMegaMoeKernel::AlignUp(static_cast<uint64_t>(resetNumPerCore), static_cast<uint64_t>(INT32_PER_256B)) *
        sizeof(int32_t);
    resetBuffer_ = LocalTensor<int32_t>(TPosition::VECCALC, resetBufferOffset, resetBufferBytes / sizeof(int32_t));
    Duplicate<int32_t>(resetBuffer_, 0, (resetBufferBytes / sizeof(int32_t)));

    uint32_t quantizationScratchOffset = resetBufferOffset + resetBufferBytes;
    uint32_t quantizationScratchBytes = 2 * 1024;
    quantizationScratch_ = LocalTensor<uint16_t>(
        TPosition::VECCALC, quantizationScratchOffset, quantizationScratchBytes / sizeof(uint16_t));

    uint32_t quantizedOutputOffset0 = quantizationScratchOffset + quantizationScratchBytes;
    uint32_t xOutTokenBytes = ShmemMegaMoeKernel::AlignUp(
        static_cast<uint32_t>(modelDim_ / ELEMS_PER_BYTE), static_cast<uint32_t>(ALIGN_256));
    uint32_t quantizedOutputElementCount0 =
        xOutTokenBytes + ShmemMegaMoeKernel::CeilDiv(modelDim_, static_cast<uint32_t>(ALIGN_32));
    quantOutputBuffer0_ = LocalTensor<ActivationType>(
        TPosition::VECCALC, quantizedOutputOffset0, quantizedOutputElementCount0 / sizeof(ActivationType));
    uint32_t quantizedOutputOffset1 = quantizedOutputOffset0 + quantizedOutputElementCount0;
    uint32_t quantizedOutputElementCount1 = quantizedOutputElementCount0;
    quantOutputBuffer1_ = LocalTensor<ActivationType>(
        TPosition::VECCALC, quantizedOutputOffset1, quantizedOutputElementCount1 / sizeof(ActivationType));

    uint32_t quantizedInputOffset0 = quantizedOutputOffset1 + quantizedOutputElementCount1;
    uint32_t quantizedInputElementCount0 =
        ShmemMegaMoeKernel::AlignUp(modelDim_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    quantInputBuffer0_ = LocalTensor<bfloat16_t>(
        TPosition::VECCALC, quantizedInputOffset0, quantizedInputElementCount0 / sizeof(bfloat16_t));
    uint32_t quantizedInputOffset1 = quantizedInputOffset0 + quantizedInputElementCount0;
    uint32_t quantizedInputElementCount1 = quantizedInputElementCount0;
    quantInputBuffer1_ = LocalTensor<bfloat16_t>(
        TPosition::VECCALC, quantizedInputOffset1, quantizedInputElementCount1 / sizeof(bfloat16_t));

    uint32_t outboundMaskOffset = quantizedInputOffset1 + quantizedInputElementCount1;
    for (int32_t index = 0; index < DOUBLE_BUFFER; ++index) {
        outboundMaskBuffers_[index] = LocalTensor<uint8_t>(
            TPosition::VECCALC, outboundMaskOffset + static_cast<uint32_t>(index) * maskSlotBytes_, maskSlotBytes_);
    }

    uint32_t maskCountBufferOffset = outboundMaskOffset + static_cast<uint32_t>(DOUBLE_BUFFER) * maskSlotBytes_;
    uint32_t maskCountScratchBytes = ShmemMegaMoeKernel::AlignUp(
        static_cast<int64_t>(paddedRouteCount_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_256));
    maskCountBuffer_ =
        LocalTensor<int32_t>(TPosition::VECCALC, maskCountBufferOffset, maskCountScratchBytes / sizeof(int32_t));
}

template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::ResetPipelineFlags()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    // Clear the activation-ready and input-ready regions for every local expert.
    pipelineFlags_.SetGlobalBuffer((__gm__ int32_t*)params_.workspace.activationReadyFlags);
    int32_t flagCount =
        static_cast<int32_t>(localExpertCount_) * (static_cast<int32_t>(INT_CACHELINE) + dispatchFlagSlotsPerExpert_);
    int32_t coreLen, coreOffset;
    SplitWorkAcrossVectorCores(flagCount, coreLen, coreOffset, 1);
    DataCopyExtParams resetCopyConfig{1U, static_cast<uint32_t>(coreLen * sizeof(int32_t)), 0U, 0U, 0U};
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    if (coreLen != 0) {
        DataCopyPad(pipelineFlags_[coreOffset], resetBuffer_, resetCopyConfig);
    }
}

template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::CopyExpertTokenCounts()
{
    int32_t finalRankIndex = static_cast<int32_t>(rankCount_ - 1);
    expertTokenCountBuffer_.SetValue(0, receivedTokenPrefixBuffer_.GetValue(finalRankIndex));
    for (int32_t expertIndex = 1; expertIndex < localExpertCount_; expertIndex++) {
        int32_t cur =
            receivedTokenPrefixBuffer_.GetValue(expertIndex * static_cast<int32_t>(rankCount_) + finalRankIndex);
        int32_t prev =
            receivedTokenPrefixBuffer_.GetValue((expertIndex - 1) * static_cast<int32_t>(rankCount_) + finalRankIndex);
        expertTokenCountBuffer_.SetValue(expertIndex, cur - prev);
    }
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopyExtParams copyConfig{1U, static_cast<uint32_t>(localExpertCount_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(expertTokenCountsOut_, expertTokenCountBuffer_, copyConfig);
}

template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::BuildSendMasks()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    GlobalTensor<int32_t> inputGlobal;
    inputGlobal.SetGlobalBuffer((__gm__ int32_t*)params_.expertIds);
    Duplicate<int32_t>(expertIdsBuffer_, 0, paddedRouteCount_);
    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
    DataCopyExtParams loadConfig{1U, static_cast<uint32_t>(routeCount_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsBuffer_, inputGlobal, loadConfig, loadPad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

    constexpr TEventID PIPELINE_EVENTS[DOUBLE_BUFFER] = {EVENT_ID0, EVENT_ID1};
    for (int32_t index = 0; index < static_cast<int32_t>(DOUBLE_BUFFER); ++index) {
        SetFlag<AscendC::HardEvent::MTE3_V>(PIPELINE_EVENTS[index]);
    }
    int32_t totalExperts = static_cast<int32_t>(rankCount_ * localExpertCount_);
    uint32_t countWordIndex = static_cast<uint32_t>(alignedMaskBytes_) / sizeof(int32_t);
    DataCopyExtParams maskCopyConfig{1U, static_cast<uint32_t>(maskSlotBytes_), 0U, 0U, 0U};
    int32_t iter = 0;
    GlobalTensor<uint8_t> outputGlobal;
    for (int32_t curExpertId = vectorCoreIndex_; curExpertId < totalExperts; curExpertId += vectorCoreCount_, ++iter) {
        int32_t dstRank = curExpertId / static_cast<int32_t>(localExpertCount_);
        int32_t expertNumPer = curExpertId % static_cast<int32_t>(localExpertCount_);
        TEventID eventId = PIPELINE_EVENTS[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
        LocalTensor<uint8_t> maskBuffer = outboundMaskBuffers_[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
        LocalTensor<uint32_t> maskWords = maskBuffer.template ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> maskCountWords = maskBuffer.template ReinterpretCast<int32_t>();

        WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
        CompareScalar(maskBuffer, expertIdsBuffer_, curExpertId, AscendC::CMPMODE::EQ, paddedRouteCount_);

        uint64_t routedTokenCount = 0;
        GatherMask(
            maskCountBuffer_, expertIdsBuffer_, maskWords, true, static_cast<uint32_t>(routeCount_), {1, 1, 0, 0},
            routedTokenCount);
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
        maskCountWords.SetValue(countWordIndex, static_cast<int32_t>(routedTokenCount));
        SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();

        uint64_t dstOffset =
            inboundMaskOffset_ +
            static_cast<uint64_t>(expertNumPer * static_cast<int32_t>(rankCount_) + static_cast<int32_t>(rankId_)) *
                static_cast<uint64_t>(maskSlotBytes_);
        outputGlobal.SetGlobalBuffer(
            (__gm__ uint8_t*)GetRemoteAddress(params_.symmetricMemory.localBase, dstRank, dstOffset));
        DataCopyPad(outputGlobal, maskBuffer, maskCopyConfig);
        SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
    }
    for (int32_t index = 0; index < static_cast<int32_t>(DOUBLE_BUFFER); ++index) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(PIPELINE_EVENTS[index]);
    }
}

template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::QuantizeLocalTokens()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    // Split local tokens evenly across vector cores.
    int32_t currentTokenCount;
    int32_t currentTokenOffset;
    SplitWorkAcrossVectorCores(tokenCount_, currentTokenCount, currentTokenOffset, 1);
    uint32_t H = modelDim_;
    GlobalTensor<bfloat16_t> inputGlobal;
    GlobalTensor<uint8_t> outputGlobal;
    DataCopyExtParams inputCopyParams = {1U, static_cast<uint32_t>(H * sizeof(bfloat16_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<bfloat16_t> inputPaddingParams{true, 0U, 0U, 0U};
    DataCopyExtParams outputCopyParams = {
        1U, static_cast<uint32_t>(alignedQuantizedTokenBytes_ + alignedScaleBytes_), 0U, 0U, 0U};
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t index = 0; index < currentTokenCount; index++) {
        inputGlobal.SetGlobalBuffer(
            (__gm__ bfloat16_t*)(params_.input + (currentTokenOffset + index) * H * sizeof(bfloat16_t)));
        outputGlobal.SetGlobalBuffer((__gm__ uint8_t*)(params_.symmetricMemory.quantizedTokens +
                                                       (currentTokenOffset + index) * quantizedRowBytes_));
        auto event = (index % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto quantInputBuffer = (index % DOUBLE_BUFFER == 0) ? quantInputBuffer0_ : quantInputBuffer1_;
        auto quantOutputBuffer = (index % DOUBLE_BUFFER == 0) ? quantOutputBuffer0_ : quantOutputBuffer1_;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(quantInputBuffer, inputGlobal, inputCopyParams, inputPaddingParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        __ubuf__ bfloat16_t* inputData = (__ubuf__ bfloat16_t*)quantInputBuffer.GetPhyAddr();
        __ubuf__ uint16_t* maxExponentData = (__ubuf__ uint16_t*)quantizationScratch_.GetPhyAddr();
        __ubuf__ uint16_t* inverseScaleData =
            (__ubuf__ uint16_t*)
                quantizationScratch_[ShmemMegaMoeKernel::AlignUp(scaleCountPerToken_, static_cast<uint32_t>(ALIGN_32))]
                    .GetPhyAddr();
        __ubuf__ int8_t* quantizedData = (__ubuf__ int8_t*)quantOutputBuffer.GetPhyAddr();
        __ubuf__ uint16_t* encodedScaleData =
            (__ubuf__ uint16_t*)quantOutputBuffer[alignedQuantizedTokenBytes_].GetPhyAddr();

        quant::ComputeMaxExp(inputData, maxExponentData, H);
        quant::ComputeScale<QuantOutType>(maxExponentData, encodedScaleData, inverseScaleData, scaleCountPerToken_);
        if constexpr (QuantMode == E2M1_QUANT) {
            quant::ComputeFp4Data<
                bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
                inputData, inverseScaleData, quantizedData, H);
        } else {
            quant::ComputeFp8Data<
                bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
                inputData, inverseScaleData, quantizedData, H);
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        auto quantizedRowBytes = quantOutputBuffer.template ReinterpretCast<uint8_t>();
        DataCopyPad(outputGlobal, quantizedRowBytes, outputCopyParams);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::CountReceivedTokens(int32_t localExpertId)
{
    receivedTokenCount_ = 0;
    uint32_t slotStrideWords = static_cast<uint32_t>(maskSlotBytes_) / sizeof(uint32_t);
    uint32_t countWordIndex = static_cast<uint32_t>(alignedMaskBytes_) / sizeof(uint32_t);

    GlobalTensor<uint8_t> inboundMaskGlobal;
    inboundMaskGlobal.SetGlobalBuffer(
        (__gm__ uint8_t*)(params_.symmetricMemory.inboundMask +
                          static_cast<uint64_t>(localExpertId) * rankCount_ * maskSlotBytes_));
    DataCopy(receivedMaskBytes_, inboundMaskGlobal, rankCount_ * maskSlotBytes_);
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

    for (int32_t sourceRank = 0; sourceRank < static_cast<int32_t>(rankCount_); ++sourceRank) {
        int32_t routedTokenCount = receivedMaskWords_.GetValue(sourceRank * slotStrideWords + countWordIndex);
        uint64_t remainingCapacity =
            receivedTokenPrefix_ < maxReceivedTokens_ ? maxReceivedTokens_ - receivedTokenPrefix_ : 0;
        uint64_t acceptedTokenCount = static_cast<uint64_t>(routedTokenCount) < remainingCapacity ?
                                          static_cast<uint64_t>(routedTokenCount) :
                                          remainingCapacity;
        receivedTokenCount_ += acceptedTokenCount;
        receivedTokenPrefix_ += acceptedTokenCount;
        receivedTokenPrefixBuffer_.SetValue(
            localExpertId * rankCount_ + sourceRank, static_cast<int32_t>(receivedTokenPrefix_));
    }

    currentExpertTokenCountBuffer_.SetValue(0, receivedTokenCount_);
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopy<int32_t>(
        receivedTokenCountsGlobal_[localExpertId * INT32_PER_256B * cubeCoreCount_ + INT32_PER_256B * cubeBlockIndex_],
        currentExpertTokenCountBuffer_, INT32_PER_256B);
    PipeBarrier<PIPE_ALL>();
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_MTE3>(2);
}

template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::CopyQuantizedTokens(
    int32_t destinationRowOffset, int32_t sourceRank, int32_t firstRouteIndex, int32_t routeCopyCount)
{
    if (routeCopyCount <= 0) {
        return;
    }
    constexpr int32_t PIPELINE_BUFFER_COUNT = DISPATCH_BUFFER_NUM;
    constexpr TEventID PIPELINE_EVENTS[PIPELINE_BUFFER_COUNT] = {EVENT_ID0, EVENT_ID1, EVENT_ID2,
                                                                 EVENT_ID3, EVENT_ID4, EVENT_ID5};
    int64_t widthA = modelDim_ / ELEMS_PER_BYTE;
    int64_t widthAScale =
        ShmemMegaMoeKernel::CeilDiv(static_cast<int64_t>(modelDim_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
    uint32_t quantizedRowElements = ShmemMegaMoeKernel::AlignUp(
        static_cast<int64_t>(alignedQuantizedTokenBytes_ + alignedScaleBytes_), static_cast<int64_t>(ALIGN_32));
    GlobalTensor<ActivationType> remoteQuantizedRows;
    GlobalTensor<ActivationType> receivedTokenData;
    GlobalTensor<QuantScaleOutType> receivedTokenScales;
    receivedTokenData.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType*>(params_.workspace.receivedTokenData + destinationRowOffset * widthA));
    receivedTokenScales.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType*>(
        params_.workspace.receivedTokenScales + destinationRowOffset * widthAScale));
    remoteQuantizedRows.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(
        GetRemoteAddress(params_.symmetricMemory.localBase, sourceRank, quantizedTokenOffset_)));

    for (int32_t bufferIndex = 0; bufferIndex < PIPELINE_BUFFER_COUNT; ++bufferIndex) {
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(PIPELINE_EVENTS[bufferIndex]);
    }
    PipeBarrier<PIPE_ALL>();

    for (int32_t i = 0; i < routeCopyCount; ++i) {
        int32_t routeIndexValue = selectedRoutes_.GetValue(firstRouteIndex + i);
        int32_t tokenIndex = routeIndexValue / expertsPerToken_;
        routingMetadata_[i * INT32_PER_256B].SetValue(RANK_ID, sourceRank);
        routingMetadata_[i * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
        routingMetadata_[i * INT32_PER_256B].SetValue(ROUTE_SLOT_INDEX, routeIndexValue % expertsPerToken_);
    }

    int32_t primeCount = (routeCopyCount < PIPELINE_BUFFER_COUNT) ? routeCopyCount : PIPELINE_BUFFER_COUNT;
    for (int32_t initialRouteIndex = 0; initialRouteIndex < primeCount; ++initialRouteIndex) {
        int32_t tokenIndex = routingMetadata_[initialRouteIndex * INT32_PER_256B].GetValue(TOKEN_ID);
        uint64_t remoteCopyOffset = static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(quantizedRowElements);
        TEventID eventId = PIPELINE_EVENTS[initialRouteIndex];
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        DataCopy(dispatchCopyBuffers_[initialRouteIndex], remoteQuantizedRows[remoteCopyOffset], quantizedRowElements);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    }
    // Phase 2 steady: MTE3[routeIndex] + issueMTE2[routeIndex + PIPELINE_BUFFER_COUNT]
    for (int32_t routeIndex = 0; routeIndex < routeCopyCount; ++routeIndex) {
        int32_t bufferIndex = routeIndex % PIPELINE_BUFFER_COUNT;
        TEventID eventId = PIPELINE_EVENTS[bufferIndex];
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);

        LocalTensor<ActivationType> quantizedRowBuffer = dispatchCopyBuffers_[bufferIndex];
        LocalTensor<QuantScaleOutType> scaleBuffer =
            quantizedRowBuffer[alignedQuantizedTokenBytes_].template ReinterpretCast<QuantScaleOutType>();
        DataCopyExtParams tokenCopyParams = {1U, static_cast<uint32_t>(widthA * sizeof(ActivationType)), 0U, 0U, 0U};
        DataCopyExtParams scaleCopyParams = {
            1U, static_cast<uint32_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U};
        DataCopyPad(receivedTokenData[routeIndex * widthA], quantizedRowBuffer, tokenCopyParams);
        DataCopyPad(receivedTokenScales[routeIndex * widthAScale], scaleBuffer, scaleCopyParams);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);

        int32_t nextRouteIndex = routeIndex + PIPELINE_BUFFER_COUNT;
        if (nextRouteIndex < routeCopyCount) {
            int32_t tokenIndex = routingMetadata_[nextRouteIndex * INT32_PER_256B].GetValue(TOKEN_ID);
            uint64_t remoteCopyOffset = static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(quantizedRowElements);

            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            DataCopy(dispatchCopyBuffers_[bufferIndex], remoteQuantizedRows[remoteCopyOffset], quantizedRowElements);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        }
    }

    for (int32_t bufferIndex = 0; bufferIndex < PIPELINE_BUFFER_COUNT; ++bufferIndex) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(PIPELINE_EVENTS[bufferIndex]);
    }

    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
    DataCopy(
        routingMetadataGlobal_[destinationRowOffset * INT32_PER_256B], routingMetadata_,
        routeCopyCount * INT32_PER_256B);
}

template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::BuildRoutingMetadataAndDispatch(
    ShmemMegaMoeMatmulBuffers& matmulBuffers, int32_t localExpertId)
{
    constexpr int32_t L1_TILE_M_I32 = static_cast<int32_t>(ShmemMegaMoeKernel::L1_TILE_M_256);
    int32_t priorExpertTokenCount =
        (localExpertId == 0) ? 0 : receivedTokenPrefixBuffer_.GetValue(localExpertId * rankCount_ - 1);
    for (uint32_t idx = cubeBlockIndex_; idx < rankCount_ * cubeBlocksPerRank_; idx += cubeBlockCount_) {
        uint32_t sourceRank = idx / cubeBlocksPerRank_;
        uint64_t gatheredCount = 0;
        GatherMask(
            selectedRoutes_, routeIndices_, receivedMaskWords_[sourceRank * (maskSlotBytes_ / sizeof(uint32_t))], true,
            routeCount_, {1, 1, 0, 0}, gatheredCount);
        uint32_t rankCoreIndex = idx % cubeBlocksPerRank_;
        uint32_t destinationRowStart =
            ((sourceRank == 0 && localExpertId == 0) ?
                 0 :
                 receivedTokenPrefixBuffer_.GetValue(localExpertId * rankCount_ + sourceRank - 1));
        uint32_t destinationRowEnd = receivedTokenPrefixBuffer_.GetValue(localExpertId * rankCount_ + sourceRank);
        uint32_t expertRouteCount = destinationRowEnd - destinationRowStart;
        int32_t rowsThisCore = 0;
        int32_t destinationRowOffset = 0;
        if (destinationRowStart < maxReceivedTokens_) {
            int32_t boundedRouteCount = expertRouteCount;
            if (destinationRowStart + boundedRouteCount > maxReceivedTokens_) {
                boundedRouteCount = maxReceivedTokens_ - destinationRowStart;
            }
            int32_t coreRouteCount =
                ShmemMegaMoeKernel::CeilDiv(boundedRouteCount, static_cast<int32_t>(cubeBlocksPerRank_));
            int32_t coreRouteOffset = rankCoreIndex * coreRouteCount;
            destinationRowOffset = destinationRowStart + coreRouteOffset;
            if (destinationRowOffset + coreRouteCount > destinationRowStart + boundedRouteCount) {
                coreRouteCount = destinationRowStart + boundedRouteCount - destinationRowOffset;
            }
            if (coreRouteCount > 0) {
                uint32_t routingMetadataOffset = nextUbOffset_;
                uint32_t routingMetadataSize = coreRouteCount * ALIGN_32;
                routingMetadata_ = LocalTensor<int32_t>(
                    TPosition::VECCALC, routingMetadataOffset, routingMetadataSize / sizeof(int32_t));
                SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();
                CopyQuantizedTokens(destinationRowOffset, sourceRank, coreRouteOffset, coreRouteCount);
                rowsThisCore = coreRouteCount;
            }
        }

        if (rowsThisCore > 0) {
            SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();

            int32_t localRowStart = destinationRowOffset - priorExpertTokenCount;
            int32_t localRowEnd = localRowStart + rowsThisCore;
            int32_t firstWaveIndex = localRowStart / L1_TILE_M_I32;
            int32_t finalWaveIndex = (localRowEnd - 1) / L1_TILE_M_I32;
            __gm__ int32_t* waveReadyFlags = matmulBuffers.waveReadyFlags;
            for (int32_t w = firstWaveIndex; w <= finalWaveIndex; ++w) {
                int32_t waveRowStart = w * L1_TILE_M_I32;
                int32_t waveRowEnd = waveRowStart + L1_TILE_M_I32;
                int32_t lo = localRowStart > waveRowStart ? localRowStart : waveRowStart;
                int32_t hi = localRowEnd < waveRowEnd ? localRowEnd : waveRowEnd;
                int32_t dispatchedRows = hi - lo;
                AtomicAdd(waveReadyFlags + w, dispatchedRows);
            }
        }
    }
}

//   Phase 1:

template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
template <ShmemMegaMoeMatmulStage Mode>
__aicore__ inline bool ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::PrepareExpertProblem(
    ExpertPipelineState& state, uint32_t expertIndex)
{
    if (expertIndex != 0) {
        uint64_t m = Get<SHAPE_ROW_INDEX>(state.problemShape);
        uint64_t n = Get<SHAPE_COLUMN_INDEX>(state.problemShape);
        uint64_t k = Get<SHAPE_REDUCTION_INDEX>(state.problemShape);
        processedTokenCount_ += m;
        Get<FIRST_INPUT_OFFSET_INDEX>(state.baseOffset) += m * k / ELEMS_PER_BYTE;
        Get<FIRST_WEIGHT_OFFSET_INDEX>(state.baseOffset) += n * k / ELEMS_PER_BYTE;
        // only splitM
        auto scaleK = ShmemMegaMoeKernel::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<FIRST_INPUT_SCALE_OFFSET_INDEX>(state.baseOffset) += m * scaleK;
        Get<FIRST_WEIGHT_SCALE_OFFSET_INDEX>(state.baseOffset) += n * scaleK;
        Get<ACTIVATION_OUTPUT_OFFSET_INDEX>(state.baseOffset) += m * n / SWIGLU_N_HALF / ELEMS_PER_BYTE;
        Get<ACTIVATION_SCALE_OFFSET_INDEX>(state.baseOffset) +=
            m * ShmemMegaMoeKernel::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
        Get<EXPERT_FLAG_OFFSET_INDEX>(state.baseOffset) += 1;
        Get<SECOND_WEIGHT_OFFSET_INDEX>(state.baseOffset) += k * n / SWIGLU_N_HALF / ELEMS_PER_BYTE;
        Get<SECOND_WEIGHT_SCALE_OFFSET_INDEX>(state.baseOffset) +=
            k * ShmemMegaMoeKernel::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
        Get<SECOND_OUTPUT_OFFSET_INDEX>(state.baseOffset) += m * k;
        Get<TOKEN_PREFIX_OFFSET_INDEX>(state.baseOffset) += m;
    }

    if constexpr (Mode == ShmemMegaMoeMatmulStage::kFirstProjection) {
        if constexpr (g_coreType == AscendC::AIC) {
            CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_S>(2 + 16);
            uint64_t expertCountOffset =
                static_cast<uint64_t>(expertIndex) * 8U * cubeCoreCount_ + 8U * cubeBlockIndex_;
            DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                receivedTokenCountsGlobal_[expertCountOffset]);
            Get<SHAPE_ROW_INDEX>(state.problemShape) = receivedTokenCountsGlobal_.GetValue(expertCountOffset);
            CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_S>(3);
        }
        if constexpr (g_coreType == AscendC::AIV) {
            if (vectorSubBlockIndex_ == 0) {
                CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_S>(3);
                uint64_t expertCountOffset =
                    static_cast<uint64_t>(expertIndex) * 8U * cubeCoreCount_ + 8U * cubeBlockIndex_;
                DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                    receivedTokenCountsGlobal_[expertCountOffset]);
                Get<SHAPE_ROW_INDEX>(state.problemShape) = receivedTokenCountsGlobal_.GetValue(expertCountOffset);
            } else {
                Get<SHAPE_ROW_INDEX>(state.problemShape) = receivedTokenCount_;
            }
        }
    } else if constexpr (Mode == ShmemMegaMoeMatmulStage::kSecondProjection) {
        uint64_t expertCountOffset = static_cast<uint64_t>(expertIndex) * 8U * cubeCoreCount_ + 8U * cubeBlockIndex_;
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            receivedTokenCountsGlobal_[expertCountOffset]);
        Get<SHAPE_ROW_INDEX>(state.problemShape) = receivedTokenCountsGlobal_.GetValue(expertCountOffset);
    }

    if (Get<SHAPE_ROW_INDEX>(state.problemShape) == 0) {
        return false;
    }
    return true;
}

template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
template <ShmemMegaMoeMatmulStage Mode>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::UpdateMatmulBuffers(
    ShmemMegaMoeMatmulBuffers& matmulBuffers, const ExpertPipelineState& state)
{
    if constexpr (Mode == ShmemMegaMoeMatmulStage::kFirstProjection) {
        matmulBuffers.lhs = params_.workspace.receivedTokenData +
                            Get<FIRST_INPUT_OFFSET_INDEX>(state.baseOffset) * sizeof(ActivationType);
        matmulBuffers.lhsScale = params_.workspace.receivedTokenScales +
                                 Get<FIRST_INPUT_SCALE_OFFSET_INDEX>(state.baseOffset) * sizeof(QuantScaleOutType);
        if constexpr (g_coreType == AIC) {
            matmulBuffers.rhs =
                params_.weight1 + Get<FIRST_WEIGHT_OFFSET_INDEX>(state.baseOffset) * sizeof(ActivationType);
            matmulBuffers.rhsScale = params_.weight1Scales +
                                     Get<FIRST_WEIGHT_SCALE_OFFSET_INDEX>(state.baseOffset) * sizeof(QuantScaleOutType);
        } else {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<ACTIVATION_OUTPUT_OFFSET_INDEX>(state.baseOffset),
                Get<ACTIVATION_SCALE_OFFSET_INDEX>(state.baseOffset),
                Get<EXPERT_FLAG_OFFSET_INDEX>(state.baseOffset),
                0L,
                0L,
                0L};
            epilogueOp_.BindOutputBuffers(vecBaseOffset);
        }
    } else if constexpr (Mode == ShmemMegaMoeMatmulStage::kSecondProjection && g_coreType == AIC) {
        matmulBuffers.lhs = params_.workspace.activatedTokenData +
                            Get<ACTIVATION_OUTPUT_OFFSET_INDEX>(state.baseOffset) * sizeof(ActivationType);
        matmulBuffers.lhsScale = params_.workspace.activatedTokenScales +
                                 Get<ACTIVATION_SCALE_OFFSET_INDEX>(state.baseOffset) * sizeof(QuantScaleOutType);
        matmulBuffers.rhs =
            params_.weight2 + Get<SECOND_WEIGHT_OFFSET_INDEX>(state.baseOffset) * sizeof(ActivationType);
        matmulBuffers.rhsScale =
            params_.weight2Scales + Get<SECOND_WEIGHT_SCALE_OFFSET_INDEX>(state.baseOffset) * sizeof(QuantScaleOutType);
    }
    if constexpr (Mode == ShmemMegaMoeMatmulStage::kSecondProjection && CombineQuantMode != COMBINE_NO_QUANT) {
        matmulBuffers.output =
            params_.workspace.matmulOutput + Get<SECOND_OUTPUT_OFFSET_INDEX>(state.baseOffset) * sizeof(bfloat16_t);
    }
    matmulBuffers.readyFlags = (__gm__ int32_t*)params_.workspace.activationReadyFlags +
                               Get<EXPERT_FLAG_OFFSET_INDEX>(state.baseOffset) * INT_CACHELINE;

    matmulBuffers.waveReadyFlags = (__gm__ int32_t*)params_.workspace.inputReadyFlags +
                                   Get<EXPERT_FLAG_OFFSET_INDEX>(state.baseOffset) * dispatchFlagSlotsPerExpert_;
    pipelineFlags_.SetGlobalBuffer(matmulBuffers.readyFlags);
}

template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::ResetRowGroupFlags()
{
    if constexpr (g_coreType == AIV) {
        int32_t totalCounters = static_cast<int32_t>(static_cast<int64_t>(localExpertCount_) * expertCounterStride_);
        int32_t coreLen, coreOffset;
        SplitWorkAcrossVectorCores(totalCounters, coreLen, coreOffset);
        GlobalTensor<int32_t> rowGroupCountersGlobal;
        rowGroupCountersGlobal.SetGlobalBuffer((__gm__ int32_t*)params_.workspace.rowGroupReadyFlags);
        if (coreLen > 0) {
            Duplicate(resetBuffer_, 0, coreLen);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
            DataCopy(rowGroupCountersGlobal[coreOffset], resetBuffer_, coreLen);
        }
    }
}

template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::InitCombineBuffers()
{
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        uint32_t nAlign32 = ShmemMegaMoeKernel::AlignUp(modelDim_, static_cast<uint32_t>(ALIGN_32));
        uint32_t nScale = ShmemMegaMoeKernel::CeilDiv(modelDim_, uint32_t(MXFP_SCALE_GROUP_NUM));
        uint32_t quantRowSizeBytes = ShmemMegaMoeKernel::AlignUp(modelDim_ + nScale, static_cast<uint32_t>(ALIGN_32));
        uint32_t singleTokenBytes = nAlign32 * sizeof(bfloat16_t) + quantRowSizeBytes;
        combineBufferElementCount_ = (singleTokenBytes * 2) / sizeof(bfloat16_t);
    }
}

template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::RunCombine(
    ShmemMegaMoeMatmulBuffers& matmulBuffers, ExpertPipelineState& secondProjectionState, uint32_t expertIndex)
{
    uint32_t requiredTileCount = ShmemMegaMoeKernel::CeilDiv(modelDim_, L1_TILE_N);

    GlobalTensor<int32_t> rowGroupCounters;
    rowGroupCounters.SetGlobalBuffer((__gm__ int32_t*)params_.workspace.rowGroupReadyFlags);

    uint32_t nScale = ShmemMegaMoeKernel::CeilDiv(modelDim_, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t quantRowSizeBytes = ShmemMegaMoeKernel::AlignUp(modelDim_ + nScale, static_cast<uint32_t>(ALIGN_32));

    uint32_t expertTokenCount = Get<SHAPE_ROW_INDEX>(secondProjectionState.problemShape);
    uint32_t rowGroupsThisExpert = ShmemMegaMoeKernel::CeilDiv(expertTokenCount, L1_TILE_M_256);

    uint32_t assignedRowGroup, coreIndexInGroup, coresInGroup;
    ShmemMegaMoeKernel::ResolveCoreGroup(
        vectorCoreIndex_, rowGroupsThisExpert, vectorCoreCount_, assignedRowGroup, coreIndexInGroup, coresInGroup);

    if (assignedRowGroup >= rowGroupsThisExpert) {
        return;
    }

    __gm__ int32_t* completionCounter =
        (__gm__ int32_t*)rowGroupCounters.GetPhyAddr() + expertIndex * expertCounterStride_ +
        assignedRowGroup * vectorCoreCount_ * INT_CACHELINE + vectorCoreIndex_ * INT_CACHELINE;
    while (AscendC::ReadGmByPassDCache(completionCounter) != requiredTileCount) {
        AscendC::Nop<200>();
    }
    uint32_t rowStart = assignedRowGroup * L1_TILE_M_256;
    uint32_t rowCount = (L1_TILE_M_256 < expertTokenCount - rowStart) ? L1_TILE_M_256 : expertTokenCount - rowStart;
    uint32_t rowsPerCore = ShmemMegaMoeKernel::CeilDiv(rowCount, coresInGroup);
    int32_t coreRowOffset = coreIndexInGroup * rowsPerCore;
    int32_t coreRowCount = 0;
    if (coreRowOffset < (int32_t)rowCount) {
        coreRowCount = (rowsPerCore < rowCount - coreRowOffset) ? rowsPerCore : rowCount - coreRowOffset;
    }
    if (coreRowCount > 0) {
        AscendC::SetCtrlSpr<60, 60>(0);
        int64_t offset = 0;
        LocalTensor<int32_t> routingMetadata =
            LocalTensor<int32_t>(TPosition::VECIN, offset, coreRowCount * ROUTING_RECORD_WIDTH);
        offset += coreRowCount * ROUTING_RECORD_WIDTH * sizeof(int32_t);
        AscendC::GlobalTensor<int32_t> routingMetadataGlobal;
        routingMetadataGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
            params_.workspace.routingMetadata +
            (processedTokenCount_ + rowStart + coreRowOffset) * ROUTING_RECORD_WIDTH * sizeof(int32_t)));
        AscendC::DataCopy(routingMetadata, routingMetadataGlobal, coreRowCount * ROUTING_RECORD_WIDTH);
        PipeBarrier<PIPE_MTE2>();
        ShmemMegaMoeKernel::CombineRowGroup<CombineQuantMode, bfloat16_t>(
            rowStart + coreRowOffset, coreRowCount, modelDim_, expertIndex, rankId_, matmulBuffers.output, params_,
            routingMetadata, combineBufferElementCount_, offset, quantRowSizeBytes);
    }
}

template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::InitOutputBuffers()
{
    uint32_t dataResBufAlign = ShmemMegaMoeKernel::AlignUp(
        static_cast<uint32_t>(COMBINE_INPUT_LIST_COUNT * modelDim_ * sizeof(bfloat16_t)),
        static_cast<uint32_t>(ALIGN_32));
    int32_t num = rankCount_ *
                  ShmemMegaMoeKernel::AlignUp(
                      static_cast<uint32_t>(rankCount_ * localExpertCount_), static_cast<uint32_t>(ALIGN_128)) *
                  sizeof(int32_t);
    uint32_t dataResFp32BufAlign = dataResBufAlign * HALF_TO_FP32;
    uint32_t routingWeightBufferBytes = ShmemMegaMoeKernel::AlignUp(
        static_cast<uint32_t>(tokenCount_ * expertsPerToken_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    uint32_t tempBufAlign = ShmemMegaMoeKernel::AlignUp(
        static_cast<uint32_t>(tokenCount_ * expertsPerToken_ * sizeof(bfloat16_t)), uint32_t(ALIGN_32));

    uint32_t bf16OutputOffset = 0;
    uint32_t dataResSize = dataResBufAlign / sizeof(bfloat16_t);
    combinedOutputBf16_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, bf16OutputOffset, dataResSize);

    uint32_t fp32OutputOffset = bf16OutputOffset + dataResBufAlign;
    uint32_t dataResFp32Size = dataResFp32BufAlign / sizeof(float);
    combinedOutputFp32_ = LocalTensor<float>(TPosition::VECCALC, fp32OutputOffset, dataResFp32Size);

    uint32_t routingWeightOffset = fp32OutputOffset + dataResFp32BufAlign;
    uint32_t routingWeightElementCount = routingWeightBufferBytes / sizeof(float);
    routingWeightsBuffer_ = LocalTensor<float>(TPosition::VECCALC, routingWeightOffset, routingWeightElementCount);
    uint32_t nextBufferOffset = routingWeightOffset + routingWeightBufferBytes;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t alignedScaleCount =
            ShmemMegaMoeKernel::AlignUp(static_cast<uint32_t>(modelDim_), static_cast<uint32_t>(ALIGN_32));

        uint32_t bf16ScaleBufAlign = ShmemMegaMoeKernel::AlignUp(
            static_cast<uint32_t>(alignedScaleCount * sizeof(bfloat16_t) * DOUBLE_BUFFER * HALF_TO_FP32),
            static_cast<uint32_t>(ALIGN_32));
        bf16ScaleBuffer_ =
            LocalTensor<bfloat16_t>(TPosition::VECCALC, nextBufferOffset, bf16ScaleBufAlign / sizeof(bfloat16_t));
        nextBufferOffset += bf16ScaleBufAlign;

        uint32_t fp32ScaleBufAlign = ShmemMegaMoeKernel::AlignUp(
            static_cast<uint32_t>(alignedScaleCount * sizeof(float) * DOUBLE_BUFFER * HALF_TO_FP32),
            static_cast<uint32_t>(ALIGN_32));
        fp32ScaleBuffer_ = LocalTensor<float>(TPosition::VECCALC, nextBufferOffset, fp32ScaleBufAlign / sizeof(float));
        nextBufferOffset += fp32ScaleBufAlign;
    }
    if constexpr (Std::IsSame<RoutingWeightType, float>::value) {
        GlobalTensor<float> routingWeightsGlobal;
        routingWeightsGlobal.SetGlobalBuffer((__gm__ float*)params_.routingWeights);
        DataCopyExtParams copyConfig = {
            1U, static_cast<uint32_t>(tokenCount_ * expertsPerToken_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> paddingConfig{false, 0U, 0U, 0U};
        DataCopyPad(routingWeightsBuffer_, routingWeightsGlobal, copyConfig, paddingConfig);
    }
    if constexpr (Std::IsSame<RoutingWeightType, bfloat16_t>::value) {
        uint32_t tempSize = tempBufAlign / sizeof(bfloat16_t);
        LocalTensor<bfloat16_t> tempLocal(TPosition::VECCALC, nextBufferOffset, tempSize);
        GlobalTensor<bfloat16_t> routingWeightsGlobal;
        routingWeightsGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)params_.routingWeights);
        DataCopyExtParams copyConfig = {
            1U, static_cast<uint32_t>(tokenCount_ * expertsPerToken_ * sizeof(bfloat16_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<bfloat16_t> paddingConfig{false, 0U, 0U, 0U};
        DataCopyPad(tempLocal, routingWeightsGlobal, copyConfig, paddingConfig);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID2>();
        Cast(routingWeightsBuffer_, tempLocal, AscendC::RoundMode::CAST_NONE, tokenCount_ * expertsPerToken_);
    }
}

template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::CombineRoutedTokens()
{
    int32_t coreLen, coreOffset;
    SplitWorkAcrossVectorCores(tokenCount_, coreLen, coreOffset, 1);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t*)params_.symmetricMemory.combinedTokens);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t*)params_.output);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    for (int32_t tokenIndex = coreOffset; tokenIndex < coreLen + coreOffset; tokenIndex++) {
        SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
        LocalTensor<bfloat16_t> dataIn0Bf16 = combinedOutputBf16_[modelDim_];
        LocalTensor<bfloat16_t> dataIn1Bf16 = combinedOutputBf16_[modelDim_ * 2];
        LocalTensor<float> dataIn0Fp32 = combinedOutputFp32_[modelDim_];
        LocalTensor<float> dataIn1Fp32 = combinedOutputFp32_[modelDim_ * 2];
        for (int32_t expId = 0; expId < expertsPerToken_; ++expId) {
            float expScale = routingWeightsBuffer_.GetValue(tokenIndex * expertsPerToken_ + expId);
            auto event = (expId % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
            auto dataInBf16 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Bf16 : dataIn1Bf16;
            auto dataInFp32 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Fp32 : dataIn1Fp32;
            if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
                WaitFlag<AscendC::HardEvent::V_MTE2>(event);
                DataCopy(dataInBf16, expandedX[(tokenIndex * expertsPerToken_ + expId) * modelDim_], modelDim_);
                SetFlag<AscendC::HardEvent::MTE2_V>(event);
                WaitFlag<AscendC::HardEvent::MTE2_V>(event);
                SetFlag<AscendC::HardEvent::S_V>(event);
                WaitFlag<AscendC::HardEvent::S_V>(event);
                Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, modelDim_);
            } else {
                uint32_t nScale = ShmemMegaMoeKernel::CeilDiv(modelDim_, uint32_t(MXFP_SCALE_GROUP_NUM));
                uint32_t quantTokenSize = modelDim_ + nScale;
                uint32_t quantizedElementCount = quantTokenSize / sizeof(bfloat16_t);
                WaitFlag<AscendC::HardEvent::V_MTE2>(event);
                DataCopy(
                    dataInBf16, expandedX[(tokenIndex * expertsPerToken_ + expId) * quantizedElementCount],
                    quantizedElementCount);
                SetFlag<AscendC::HardEvent::MTE2_V>(event);
                WaitFlag<AscendC::HardEvent::MTE2_V>(event);
                using Fp8Type = typename std::conditional<
                    CombineQuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
                ShmemMegaMoeKernel::DequantizeMxFp8<Fp8Type, bfloat16_t>(
                    dataInBf16, dataInFp32, bf16ScaleBuffer_, fp32ScaleBuffer_, nScale, modelDim_);
            }
            PipeBarrier<PIPE_V>();
            if (expId == 0) {
                Muls(combinedOutputFp32_, dataInFp32, expScale, modelDim_);
            } else {
                Muls(dataInFp32, dataInFp32, expScale, modelDim_);
                PipeBarrier<PIPE_V>();
                Add(combinedOutputFp32_, combinedOutputFp32_, dataInFp32, modelDim_);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(event);
        }
        // fp32 -> bf16
        Cast(combinedOutputBf16_, combinedOutputFp32_, AscendC::RoundMode::CAST_RINT, modelDim_);
        SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID3>();
        DataCopy(output[tokenIndex * modelDim_], combinedOutputBf16_, modelDim_);
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
}

// SynchronizeRanks uses the first 48 KiB of symmetric memory for cross-rank synchronization.
template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::SynchronizeRanks()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    __gm__ int32_t* syncRank = (__gm__ int32_t*)(params_.symmetricMemory.localBase);
    __gm__ int32_t* syncCount =
        (__gm__ int32_t*)(params_.symmetricMemory.localBase + 48 * 1024 + vectorCoreIndex_ * 64);
    int count = ReadGmByPassDCache(syncCount) + 1;
    for (int i = vectorCoreIndex_; i < rankCount_; i += vectorCoreCount_) {
        __gm__ int32_t* remoteSynchronizationFlag =
            reinterpret_cast<__gm__ int32_t*>(GetRemoteAddress(params_.symmetricMemory.localBase, i, 0)) + rankId_ * 16;
        WriteGmByPassDCache(remoteSynchronizationFlag, count);
        auto syncCheck = syncRank + i * 16;
        WaitForRankGeneration(syncCheck, count);
    }
    WriteGmByPassDCache(syncCount, count);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::Process()
{
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    InitSendQuantBuffers();
    BuildSendMasks();
    ResetPipelineFlags();
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        ResetRowGroupFlags();
    }
    QuantizeLocalTokens();
    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }
    SynchronizeRanks();

    InitDispatchBuffers();
    ShmemMegaMoeMatmulBuffers matmulBuffers;
    ShmemMegaMoeProblemShape initShape;
    Get<SHAPE_COLUMN_INDEX>(initShape) = ffnDim_;
    Get<SHAPE_REDUCTION_INDEX>(initShape) = modelDim_;
    ShmemMegaMoeBufferOffsets initOffset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ExpertPipelineState firstProjectionState{initShape, initOffset};
    for (int localExpertId = 0; localExpertId < localExpertCount_; localExpertId++) {
        if constexpr (g_coreType == AIV) {
            if (vectorSubBlockIndex_ == 1) {
                CountReceivedTokens(localExpertId);
            }
        }
        if (!PrepareExpertProblem<ShmemMegaMoeMatmulStage::kFirstProjection>(firstProjectionState, localExpertId)) {
            continue;
        }
        UpdateMatmulBuffers<ShmemMegaMoeMatmulStage::kFirstProjection>(matmulBuffers, firstProjectionState);
        if constexpr (g_coreType == AIV) {
            if (vectorSubBlockIndex_ == 1) {
                BuildRoutingMetadataAndDispatch(matmulBuffers, localExpertId);
            }
        }
        if (vectorSubBlockIndex_ == 0) {
            ShmemMegaMoeKernel::RunFirstProjection<
                QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
                epilogueOp_, firstProjectionState.problemShape, matmulBuffers, schedulerStartBlock_, vectorSyncState_);
        }
    }
    FinishFirstProjectionSync(vectorSyncState_);
    if constexpr (g_coreType == AIV) {
        if (vectorSubBlockIndex_ == 1) {
            CopyExpertTokenCounts();
        }
    }

    vectorSyncState_ = 0;
    processedTokenCount_ = 0;
    ExpertPipelineState secondProjectionState{initShape, initOffset};
    InitCombineBuffers();

    for (uint32_t expertIndex = 0; expertIndex < localExpertCount_; expertIndex++) {
        if (!PrepareExpertProblem<ShmemMegaMoeMatmulStage::kSecondProjection>(secondProjectionState, expertIndex)) {
            continue;
        }
        UpdateMatmulBuffers<ShmemMegaMoeMatmulStage::kSecondProjection>(matmulBuffers, secondProjectionState);

        ShmemMegaMoeKernel::RunSecondProjection<
            CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
            params_, secondProjectionState.problemShape, matmulBuffers, schedulerStartBlock_, vectorSyncState_,
            processedTokenCount_, secondProjectionBufferIndex_, expertIndex, expertCounterStride_);

        if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
            RunCombine(matmulBuffers, secondProjectionState, expertIndex);
        }
    }
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        FinishSecondProjectionSync(vectorSyncState_, secondProjectionBufferIndex_);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    if constexpr (g_coreType == AIV) {
        InitOutputBuffers();
        SynchronizeRanks();
        CombineRoutedTokens();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

} // namespace ShmemMegaMoeKernel
#endif // SHMEM_EXAMPLES_MEGA_MOE_KERNEL_ARCH35_KERNEL_H
