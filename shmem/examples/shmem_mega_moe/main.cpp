/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "host/init/shmem_host_init.h"
#include "host/mem/shmem_host_heap.h"
#include "host/shmem_host_def.h"
#include "shmem.h"
#include "utils.h"

#include "shmem_mega_moe_host.h"
#include "shmem_mega_moe_main.h"
#include "shmem_mega_moe_types.h"

namespace {

aclshmemx_uniqueid_t g_defaultFlagUid;

bool ParseUint32(const char* text, const char* name, bool allowZero, uint32_t& value)
{
    if (text == nullptr || *text == '\0' || *text == '-') {
        std::cerr << "[shmem_mega_moe] invalid " << name << ": " << (text == nullptr ? "<null>" : text) << std::endl;
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max() ||
        (!allowZero && parsed == 0)) {
        std::cerr << "[shmem_mega_moe] invalid " << name << ": " << text << std::endl;
        return false;
    }
    value = static_cast<uint32_t>(parsed);
    return true;
}

bool ReadPositiveEnv(const char* name, uint32_t& value)
{
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') {
        return false;
    }
    return ParseUint32(text, name, false, value);
}

bool ConfigureDeviceCores(int deviceId, ShmemMegaMoeExample::RunConfig& cfg)
{
    int64_t cubeCoreNum = 0;
    int64_t vectorCoreNum = 0;
    bool hasCubeCoreNum =
        aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_CUBE_CORE_NUM, &cubeCoreNum) == ACL_SUCCESS && cubeCoreNum > 0;
    bool hasVectorCoreNum =
        aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vectorCoreNum) == ACL_SUCCESS && vectorCoreNum > 0;
    if (!hasCubeCoreNum) {
        hasCubeCoreNum = aclGetDeviceCapability(deviceId, ACL_DEVICE_INFO_AI_CORE_NUM, &cubeCoreNum) == ACL_SUCCESS &&
                         cubeCoreNum > 0;
    }
    if (!hasVectorCoreNum) {
        hasVectorCoreNum =
            aclGetDeviceCapability(deviceId, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &vectorCoreNum) == ACL_SUCCESS &&
            vectorCoreNum > 0;
    }
    const int64_t maxCoreCount = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    if ((hasCubeCoreNum && cubeCoreNum > maxCoreCount) || (hasVectorCoreNum && vectorCoreNum > maxCoreCount)) {
        std::cerr << "[shmem_mega_moe] detected device core count exceeds the supported range" << std::endl;
        return false;
    }
    if (hasCubeCoreNum) {
        cfg.cubeCoreCount = static_cast<uint32_t>(cubeCoreNum);
    }
    if (hasVectorCoreNum) {
        cfg.vectorCoreCount = static_cast<uint32_t>(vectorCoreNum);
    }
    if (!hasCubeCoreNum && hasVectorCoreNum) {
        cfg.cubeCoreCount = std::max(1U, cfg.vectorCoreCount / 2U);
    } else if (!hasVectorCoreNum) {
        if (cfg.cubeCoreCount > std::numeric_limits<uint32_t>::max() / 2U) {
            std::cerr << "[shmem_mega_moe] fallback device core count exceeds the supported range" << std::endl;
            return false;
        }
        cfg.vectorCoreCount = cfg.cubeCoreCount * 2U;
    }

    const uint32_t detectedCubeCoreCount = cfg.cubeCoreCount;
    const uint32_t detectedVectorCoreCount = cfg.vectorCoreCount;
    uint32_t requestedCubeCoreCount = detectedCubeCoreCount;
    uint32_t requestedVectorCoreCount = detectedVectorCoreCount;
    const bool hasCubeOverride = std::getenv("SHMEM_MEGA_MOE_AIC_NUM") != nullptr;
    const bool hasVectorOverride = std::getenv("SHMEM_MEGA_MOE_AIV_NUM") != nullptr;
    if ((hasCubeOverride && !ReadPositiveEnv("SHMEM_MEGA_MOE_AIC_NUM", requestedCubeCoreCount)) ||
        (hasVectorOverride && !ReadPositiveEnv("SHMEM_MEGA_MOE_AIV_NUM", requestedVectorCoreCount))) {
        return false;
    }
    if (requestedCubeCoreCount > detectedCubeCoreCount || requestedVectorCoreCount > detectedVectorCoreCount) {
        std::cerr << "[shmem_mega_moe] requested AIC/AIV count exceeds the detected hardware limit" << std::endl;
        return false;
    }
    if ((hasCubeOverride || hasVectorOverride) && (requestedCubeCoreCount > std::numeric_limits<uint32_t>::max() / 2U ||
                                                   requestedVectorCoreCount != requestedCubeCoreCount * 2U)) {
        std::cerr << "[shmem_mega_moe] AIC/AIV override must keep the 1:2 MIX-kernel ratio" << std::endl;
        return false;
    }
    cfg.cubeCoreCount = requestedCubeCoreCount;
    cfg.vectorCoreCount = requestedVectorCoreCount;
    return true;
}

uint64_t ByteChecksum(const void* ptr, size_t size)
{
    const auto* bytes = reinterpret_cast<const uint8_t*>(ptr);
    uint64_t checksum = 0;
    for (size_t i = 0; i < size; ++i) {
        checksum = checksum * 131U + bytes[i];
    }
    return checksum;
}

int64_t Int32Sum(const int32_t* ptr, size_t count) { return std::accumulate(ptr, ptr + count, int64_t{0}); }

void BuildExpectedExpertTokens(std::vector<int32_t>& expected, const ShmemMegaMoeExample::RunConfig& cfg)
{
    expected.assign(cfg.localExpertCount, 0);
    const uint32_t expertNum = cfg.rankCount * cfg.localExpertCount;
    const uint32_t localExpertBegin = cfg.rankId * cfg.localExpertCount;
    const uint32_t localExpertEnd = localExpertBegin + cfg.localExpertCount;
    std::vector<std::vector<uint32_t>> receivedByRank(cfg.localExpertCount, std::vector<uint32_t>(cfg.rankCount, 0));
    for (uint32_t srcRank = 0; srcRank < cfg.rankCount; ++srcRank) {
        for (uint32_t i = 0; i < cfg.tokenCount; ++i) {
            for (uint32_t k = 0; k < cfg.expertsPerToken; ++k) {
                const uint32_t globalExpert = (i + k + srcRank) % expertNum;
                if (globalExpert >= localExpertBegin && globalExpert < localExpertEnd) {
                    ++receivedByRank[globalExpert - localExpertBegin][srcRank];
                }
            }
        }
    }

    uint32_t remainingCapacity = ShmemMegaMoeExample::DefaultMaxReceivedTokens(cfg);
    for (uint32_t localExpert = 0; localExpert < cfg.localExpertCount; ++localExpert) {
        for (uint32_t srcRank = 0; srcRank < cfg.rankCount; ++srcRank) {
            const uint32_t accepted = std::min(receivedByRank[localExpert][srcRank], remainingCapacity);
            expected[localExpert] += static_cast<int32_t>(accepted);
            remainingCapacity -= accepted;
        }
    }
}

bool CheckExpertTokens(const int32_t* actual, const ShmemMegaMoeExample::RunConfig& cfg)
{
    std::vector<int32_t> expected;
    BuildExpectedExpertTokens(expected, cfg);
    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << "[shmem_mega_moe] rank " << cfg.rankId << " expert_token mismatch at local_expert " << i
                      << ": actual=" << actual[i] << ", expected=" << expected[i] << std::endl;
            return false;
        }
    }
    return true;
}

float Bf16ToFloat(uint16_t value)
{
    const bool negative = (value & 0x8000U) != 0;
    const uint32_t exponent = (value >> 7U) & 0xffU;
    const uint32_t mantissa = value & 0x7fU;
    float result = 0.0F;
    if (exponent == 0) {
        result = std::ldexp(static_cast<float>(mantissa), -133);
    } else if (exponent == 0xffU) {
        result = mantissa == 0 ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
    } else {
        result = std::ldexp(1.0F + static_cast<float>(mantissa) / 128.0F, static_cast<int>(exponent) - 127);
    }
    return negative ? -result : result;
}

float ExpectedOutputValue(const ShmemMegaMoeExample::RunConfig& cfg)
{
    const float firstProjection = static_cast<float>(cfg.modelDim);
    const float silu = firstProjection / (1.0F + std::exp(-firstProjection));
    return silu * firstProjection * static_cast<float>(cfg.ffnDim / 2U);
}

bool CheckNumericalOutput(const void* output, const ShmemMegaMoeExample::RunConfig& cfg)
{
    const float expected = ExpectedOutputValue(cfg);
    const float relativeTolerance = cfg.mode == ShmemMegaMoeExample::Mode::Arch35E4M3 ? 0.05F : 0.15F;
    const float absoluteTolerance = 1.0F;
    const auto* values = reinterpret_cast<const uint16_t*>(output);
    const size_t elementCount = static_cast<size_t>(cfg.tokenCount) * cfg.modelDim;
    for (size_t i = 0; i < elementCount; ++i) {
        const float actual = Bf16ToFloat(values[i]);
        const float difference = std::abs(actual - expected);
        if (!std::isfinite(actual) || difference > absoluteTolerance + relativeTolerance * std::abs(expected)) {
            std::cerr << "[shmem_mega_moe] rank " << cfg.rankId << " output mismatch at element " << i
                      << ": actual=" << actual << ", expected=" << expected << ", rtol=" << relativeTolerance
                      << ", atol=" << absoluteTolerance << std::endl;
            return false;
        }
    }
    return true;
}

bool ValidateOutputs(const ShmemMegaMoeExample::RunConfig& cfg, const void* outputHost, const void* expertTokenHost)
{
    return CheckExpertTokens(reinterpret_cast<const int32_t*>(expertTokenHost), cfg) &&
           CheckNumericalOutput(outputHost, cfg);
}

uint8_t* ToKernelPtr(void* ptr) { return static_cast<uint8_t*>(ptr); }

void FillRoutingWeights(float* ptr, const ShmemMegaMoeExample::RunConfig& cfg)
{
    const float value = 1.0f / static_cast<float>(cfg.expertsPerToken);
    const size_t routeCount = static_cast<size_t>(cfg.tokenCount) * cfg.expertsPerToken;
    for (size_t i = 0; i < routeCount; ++i) {
        ptr[i] = value;
    }
}

void FillBf16Ones(void* ptr, size_t elemCount)
{
    auto* values = reinterpret_cast<uint16_t*>(ptr);
    constexpr uint16_t bf16One = 0x3f80U;
    for (size_t i = 0; i < elemCount; ++i) {
        values[i] = bf16One;
    }
}

void FillQuantizedWeights(
    void* weight1, size_t weight1Bytes, void* weight2, size_t weight2Bytes, ShmemMegaMoeExample::Mode mode)
{
    constexpr uint8_t fp8E4M3One = 0x38U;
    constexpr uint8_t fp8E5M2One = 0x3cU;
    const uint8_t value = mode == ShmemMegaMoeExample::Mode::Arch35E4M3 ? fp8E4M3One : fp8E5M2One;
    std::fill_n(static_cast<uint8_t*>(weight1), weight1Bytes, value);
    std::fill_n(static_cast<uint8_t*>(weight2), weight2Bytes, value);
}

void FillMxScales(void* scales, size_t size)
{
    constexpr uint8_t e8m0One = 0x7fU;
    std::fill_n(static_cast<uint8_t*>(scales), size, e8m0One);
}

void FillExpertIds(int32_t* ptr, const ShmemMegaMoeExample::RunConfig& cfg)
{
    const uint32_t expertNum = cfg.rankCount * cfg.localExpertCount;
    for (uint32_t i = 0; i < cfg.tokenCount; ++i) {
        for (uint32_t k = 0; k < cfg.expertsPerToken; ++k) {
            ptr[i * cfg.expertsPerToken + k] = static_cast<int32_t>((i + k + cfg.rankId) % expertNum);
        }
    }
}

bool CheckedProduct(std::initializer_list<size_t> factors, size_t& result)
{
    result = 1;
    for (size_t factor : factors) {
        if (factor != 0 && result > std::numeric_limits<size_t>::max() / factor) {
            return false;
        }
        result *= factor;
    }
    return true;
}

bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool CheckedMultiply(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool CheckedAlignUp(uint64_t value, uint64_t alignment, uint64_t& result)
{
    uint64_t adjusted = 0;
    if (alignment == 0 || !CheckedAdd(value, alignment - 1U, adjusted)) {
        return false;
    }
    result = adjusted / alignment * alignment;
    return true;
}

bool CalculateSymmetricMemorySize(const ShmemMegaMoeExample::RunConfig& cfg, uint64_t& result)
{
    uint64_t routeCount = 0;
    uint64_t compareBytes = 0;
    uint64_t compareCount = 0;
    uint64_t maskBytes = 0;
    uint64_t tokenBytes = 0;
    uint64_t quantizedTokensBytes = 0;
    uint64_t combineBytes = 0;
    uint64_t offset = static_cast<uint64_t>(SHMEM_CONTROL_BYTES);
    const uint64_t scaleCount = (static_cast<uint64_t>(cfg.modelDim) + ALIGN_32 - 1U) / ALIGN_32;
    uint64_t dataBytes = 0;

    if (!CheckedMultiply(cfg.tokenCount, cfg.expertsPerToken, routeCount) ||
        !CheckedMultiply(routeCount, sizeof(int32_t), compareBytes) ||
        !CheckedAlignUp(compareBytes, ALIGN_256, compareBytes)) {
        return false;
    }
    compareCount = compareBytes / sizeof(int32_t);
    uint64_t maskSlotBytes = 0;
    if (!CheckedAlignUp(compareCount / 8U, ALIGN_32, maskSlotBytes) ||
        !CheckedAdd(maskSlotBytes, ALIGN_32, maskSlotBytes) ||
        !CheckedMultiply(cfg.localExpertCount, cfg.rankCount, maskBytes) ||
        !CheckedMultiply(maskBytes, maskSlotBytes, maskBytes) || !CheckedAlignUp(maskBytes, ALIGN_512, maskBytes) ||
        !CheckedAdd(offset, maskBytes, offset) || !CheckedAlignUp(cfg.modelDim, ALIGN_256, dataBytes) ||
        !CheckedAdd(dataBytes, scaleCount, tokenBytes) || !CheckedAlignUp(tokenBytes, ALIGN_32, tokenBytes) ||
        !CheckedMultiply(cfg.tokenCount, tokenBytes, quantizedTokensBytes) ||
        !CheckedAlignUp(quantizedTokensBytes, ALIGN_512, quantizedTokensBytes) ||
        !CheckedAdd(offset, quantizedTokensBytes, offset) || !CheckedMultiply(routeCount, cfg.modelDim, combineBytes) ||
        !CheckedMultiply(combineBytes, sizeof(uint16_t), combineBytes) || !CheckedAdd(offset, combineBytes, offset) ||
        !CheckedAdd(offset, 64ULL * ShmemMegaMoeExample::MiB, offset) || !CheckedAlignUp(offset, ALIGN_512, result)) {
        return false;
    }
    return true;
}

bool ResolveAndValidateShape(ShmemMegaMoeExample::RunConfig& cfg)
{
    const uint64_t expertCount = static_cast<uint64_t>(cfg.rankCount) * cfg.localExpertCount;
    const uint64_t routeCount = static_cast<uint64_t>(cfg.tokenCount) * cfg.expertsPerToken;
    const uint64_t requiredCapacity =
        static_cast<uint64_t>(cfg.tokenCount) * cfg.rankCount * std::min(cfg.expertsPerToken, cfg.localExpertCount);
    const uint64_t quantizedRowBytes =
        static_cast<uint64_t>(cfg.modelDim) + (static_cast<uint64_t>(cfg.modelDim) + 31U) / 32U;
    if ((cfg.ffnDim % 2U) != 0U || expertCount > std::numeric_limits<uint32_t>::max() ||
        cfg.expertsPerToken > expertCount || routeCount > std::numeric_limits<uint32_t>::max() ||
        requiredCapacity > std::numeric_limits<uint32_t>::max() ||
        routeCount > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / sizeof(int32_t) ||
        static_cast<uint64_t>(cfg.modelDim) * sizeof(bfloat16_t) > std::numeric_limits<uint32_t>::max() ||
        quantizedRowBytes > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "[shmem_mega_moe] shape exceeds the supported indexing or copy-length range" << std::endl;
        return false;
    }
    if (cfg.maxReceivedTokens == 0) {
        cfg.maxReceivedTokens = static_cast<uint32_t>(requiredCapacity);
    } else if (cfg.maxReceivedTokens < requiredCapacity) {
        std::cerr << "[shmem_mega_moe] max_received_tokens must be at least " << requiredCapacity
                  << " for numeric verification" << std::endl;
        return false;
    }

    size_t workspaceTerm = 0;
    if (!CheckedProduct({cfg.maxReceivedTokens, cfg.modelDim, sizeof(int32_t)}, workspaceTerm) ||
        workspaceTerm > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        !CheckedProduct({cfg.maxReceivedTokens, cfg.ffnDim, sizeof(int32_t)}, workspaceTerm) ||
        workspaceTerm > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        !std::isfinite(ExpectedOutputValue(cfg))) {
        std::cerr << "[shmem_mega_moe] shape overflows workspace or reference-value calculations" << std::endl;
        return false;
    }
    return true;
}

struct BufferSizes {
    size_t x = 0;
    size_t expertIds = 0;
    size_t routingWeights = 0;
    size_t weight1 = 0;
    size_t weight2 = 0;
    size_t scale1 = 0;
    size_t scale2 = 0;
    size_t mask = 0;
    size_t extraScales = 0;
    size_t output = 0;
    size_t expertTokens = 0;
};

bool CalculateBufferSizes(const ShmemMegaMoeExample::RunConfig& cfg, BufferSizes& sizes)
{
    const size_t scale1K = ShmemMegaMoeExample::AlignUp(static_cast<size_t>(cfg.modelDim), 64) / 64 * 2;
    const size_t scale2K = ShmemMegaMoeExample::AlignUp(static_cast<size_t>(cfg.ffnDim) / 2, 64) / 64 * 2;
    const bool valid = CheckedProduct({cfg.tokenCount, cfg.modelDim, sizeof(bfloat16_t)}, sizes.x) &&
                       CheckedProduct({cfg.tokenCount, cfg.expertsPerToken, sizeof(int32_t)}, sizes.expertIds) &&
                       CheckedProduct({cfg.tokenCount, cfg.expertsPerToken, sizeof(float)}, sizes.routingWeights) &&
                       CheckedProduct({cfg.localExpertCount, cfg.ffnDim, cfg.modelDim}, sizes.weight1) &&
                       CheckedProduct({cfg.localExpertCount, cfg.modelDim, cfg.ffnDim / 2U}, sizes.weight2) &&
                       CheckedProduct({cfg.localExpertCount, cfg.ffnDim, scale1K}, sizes.scale1) &&
                       CheckedProduct({cfg.localExpertCount, cfg.modelDim, scale2K}, sizes.scale2) &&
                       CheckedProduct({cfg.tokenCount, sizeof(bool)}, sizes.mask) &&
                       CheckedProduct({cfg.maxReceivedTokens, sizeof(float)}, sizes.extraScales) &&
                       CheckedProduct({cfg.tokenCount, cfg.modelDim, sizeof(bfloat16_t)}, sizes.output) &&
                       CheckedProduct({cfg.localExpertCount, sizeof(int32_t)}, sizes.expertTokens);
    if (!valid) {
        std::cerr << "[shmem_mega_moe] shape overflows host allocation sizes" << std::endl;
        return false;
    }
    return true;
}

bool CheckRuntimeStatus(int status, const char* operation)
{
    if (status == 0) {
        return true;
    }
    std::cerr << "[shmem_mega_moe] " << operation << " failed, status=" << status << std::endl;
    return false;
}

class RuntimeResources {
public:
    ~RuntimeResources() { Cleanup(); }

    bool AllocateHost(void*& ptr, size_t size)
    {
        if (!CheckRuntimeStatus(aclrtMallocHost(&ptr, size), "aclrtMallocHost")) {
            return false;
        }
        hostAllocations_.push_back(ptr);
        return true;
    }

    bool AllocateDevice(void*& ptr, size_t size)
    {
        if (!CheckRuntimeStatus(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc")) {
            return false;
        }
        deviceAllocations_.push_back(ptr);
        return true;
    }

    bool DestroyEvent(aclrtEvent& event)
    {
        if (event == nullptr) {
            return true;
        }
        const bool ok = CheckRuntimeStatus(aclrtDestroyEvent(event), "aclrtDestroyEvent");
        event = nullptr;
        return ok;
    }

    bool Cleanup()
    {
        if (cleaned_) {
            return cleanupSucceeded_;
        }
        cleaned_ = true;
        cleanupSucceeded_ = DestroyEvent(endEvent) && cleanupSucceeded_;
        cleanupSucceeded_ = DestroyEvent(startEvent) && cleanupSucceeded_;
        if (symmetricMemory != nullptr) {
            aclshmem_free(symmetricMemory);
            symmetricMemory = nullptr;
        }
        for (auto it = deviceAllocations_.rbegin(); it != deviceAllocations_.rend(); ++it) {
            cleanupSucceeded_ = CheckRuntimeStatus(aclrtFree(*it), "aclrtFree") && cleanupSucceeded_;
        }
        deviceAllocations_.clear();
        for (auto it = hostAllocations_.rbegin(); it != hostAllocations_.rend(); ++it) {
            cleanupSucceeded_ = CheckRuntimeStatus(aclrtFreeHost(*it), "aclrtFreeHost") && cleanupSucceeded_;
        }
        hostAllocations_.clear();
        if (shmemInitialized) {
            cleanupSucceeded_ = CheckRuntimeStatus(aclshmem_finalize(), "aclshmem_finalize") && cleanupSucceeded_;
            shmemInitialized = false;
        }
        if (stream != nullptr) {
            cleanupSucceeded_ =
                CheckRuntimeStatus(aclrtDestroyStream(stream), "aclrtDestroyStream") && cleanupSucceeded_;
            stream = nullptr;
        }
        if (deviceSet) {
            cleanupSucceeded_ = CheckRuntimeStatus(aclrtResetDevice(deviceId), "aclrtResetDevice") && cleanupSucceeded_;
            deviceSet = false;
        }
        if (aclInitialized) {
            cleanupSucceeded_ = CheckRuntimeStatus(aclFinalize(), "aclFinalize") && cleanupSucceeded_;
            aclInitialized = false;
        }
        return cleanupSucceeded_;
    }

    bool aclInitialized = false;
    bool deviceSet = false;
    bool shmemInitialized = false;
    int deviceId = 0;
    aclrtStream stream = nullptr;
    aclrtEvent startEvent = nullptr;
    aclrtEvent endEvent = nullptr;
    void* symmetricMemory = nullptr;

private:
    bool cleaned_ = false;
    bool cleanupSucceeded_ = true;
    std::vector<void*> hostAllocations_;
    std::vector<void*> deviceAllocations_;
};

void PrintUsage()
{
    std::cerr << "Usage: shmem_mega_moe <rank_size> <rank> <ip:port> <npu_num> <first_npu> "
                 "<mode> <token_count> <model_dim> <ffn_dim> <experts_per_token> <local_expert_count> "
                 "[max_received_tokens] [warmup] [loop]\n"
              << "  mode: arch35_e4m3 | arch35_e5m2\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 12 || argc > 15) {
        PrintUsage();
        return 1;
    }

    ShmemMegaMoeExample::RunConfig cfg;
    uint32_t rankCountValue = 0;
    uint32_t rankIdValue = 0;
    uint32_t npuNumValue = 0;
    uint32_t firstNpuValue = 0;
    uint32_t warmupValue = 0;
    uint32_t loopValue = 1;
    if (!ParseUint32(argv[1], "rank_size", false, rankCountValue) || !ParseUint32(argv[2], "rank", true, rankIdValue) ||
        !ParseUint32(argv[4], "npu_num", false, npuNumValue) ||
        !ParseUint32(argv[5], "first_npu", true, firstNpuValue) ||
        !ParseUint32(argv[7], "token_count", false, cfg.tokenCount) ||
        !ParseUint32(argv[8], "model_dim", false, cfg.modelDim) ||
        !ParseUint32(argv[9], "ffn_dim", false, cfg.ffnDim) ||
        !ParseUint32(argv[10], "experts_per_token", false, cfg.expertsPerToken) ||
        !ParseUint32(argv[11], "local_expert_count", false, cfg.localExpertCount) ||
        (argc > 12 && !ParseUint32(argv[12], "max_received_tokens", true, cfg.maxReceivedTokens)) ||
        (argc > 13 && !ParseUint32(argv[13], "warmup", true, warmupValue)) ||
        (argc > 14 && !ParseUint32(argv[14], "loop", false, loopValue))) {
        PrintUsage();
        return 1;
    }
    if (rankCountValue > static_cast<uint32_t>(std::numeric_limits<int>::max()) || rankIdValue >= rankCountValue ||
        npuNumValue > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        firstNpuValue > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        firstNpuValue > static_cast<uint32_t>(std::numeric_limits<int>::max()) - npuNumValue ||
        warmupValue > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        loopValue > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        warmupValue > static_cast<uint32_t>(std::numeric_limits<int>::max()) - loopValue) {
        std::cerr << "[shmem_mega_moe] rank, device, or loop argument exceeds the supported range" << std::endl;
        return 1;
    }

    const int rankCount = static_cast<int>(rankCountValue);
    const int rankId = static_cast<int>(rankIdValue);
    const char* ipPort = argv[3];
    const int npuNum = static_cast<int>(npuNumValue);
    const int firstNpu = static_cast<int>(firstNpuValue);
    if (ipPort == nullptr || *ipPort == '\0') {
        std::cerr << "[shmem_mega_moe] ip:port must not be empty" << std::endl;
        return 1;
    }
    if (!ShmemMegaMoeExample::ParseMode(argv[6], cfg.mode)) {
        std::cerr << "[shmem_mega_moe] unsupported mode: " << argv[6] << std::endl;
        PrintUsage();
        return 1;
    }
    cfg.rankCount = rankCountValue;
    cfg.rankId = rankIdValue;
    cfg.warmupCount = static_cast<int>(warmupValue);
    cfg.loopCount = static_cast<int>(loopValue);
    if (!ResolveAndValidateShape(cfg)) {
        return 1;
    }
    BufferSizes sizes;
    if (!CalculateBufferSizes(cfg, sizes)) {
        return 1;
    }
    uint64_t requiredSymmetricMemory = 0;
    if (!CalculateSymmetricMemorySize(cfg, requiredSymmetricMemory) ||
        requiredSymmetricMemory > ShmemMegaMoeExample::DefaultLocalMemSize) {
        std::cerr << "[shmem_mega_moe] shape exceeds the configured 1 GiB symmetric heap" << std::endl;
        return 1;
    }

    const int deviceId = rankId % npuNum + firstNpu;
    RuntimeResources resources;
    if (!CheckRuntimeStatus(aclInit(nullptr), "aclInit")) {
        return 1;
    }
    resources.aclInitialized = true;
    if (!CheckRuntimeStatus(aclrtSetDevice(deviceId), "aclrtSetDevice")) {
        return 1;
    }
    resources.deviceSet = true;
    resources.deviceId = deviceId;
    if (!ConfigureDeviceCores(deviceId, cfg)) {
        return 1;
    }
    if (cfg.cubeCoreCount == 0 || cfg.vectorCoreCount == 0 || cfg.localExpertCount > cfg.cubeCoreCount) {
        std::cerr << "[shmem_mega_moe] local_expert_count must not exceed the available AIC count" << std::endl;
        return 1;
    }
    if (rankId == 0) {
        std::cout << "[shmem_mega_moe] device cores: aic=" << cfg.cubeCoreCount << ", aiv=" << cfg.vectorCoreCount
                  << std::endl;
    }
    if (!CheckRuntimeStatus(aclrtCreateStream(&resources.stream), "aclrtCreateStream")) {
        return 1;
    }

    aclshmemx_init_attr_t attributes{};
    test_set_attr(rankId, rankCount, ShmemMegaMoeExample::DefaultLocalMemSize, ipPort, g_defaultFlagUid, &attributes);
    attributes.comm_args = reinterpret_cast<void*>(&g_defaultFlagUid);
    if (!CheckRuntimeStatus(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes), "aclshmemx_init_attr")) {
        return 1;
    }
    resources.shmemInitialized = true;

    const auto arch35Tiling = ShmemMegaMoeExample::BuildArch35Tiling(cfg);
    const size_t workspaceBytes = ShmemMegaMoeExample::CalcArch35WorkspaceSize(arch35Tiling);
    const size_t symmetricMemoryBytes = ShmemMegaMoeExample::CalcArch35SymmetricMemorySize(arch35Tiling, 1);
    if (workspaceBytes == 0 || symmetricMemoryBytes == 0 || symmetricMemoryBytes != requiredSymmetricMemory ||
        symmetricMemoryBytes > ShmemMegaMoeExample::DefaultLocalMemSize) {
        std::cerr << "[shmem_mega_moe] calculated workspace or symmetric-memory size is invalid: workspace="
                  << workspaceBytes << ", symmetric=" << symmetricMemoryBytes << std::endl;
        return 1;
    }

    void* xHost = nullptr;
    void* expertIdsHost = nullptr;
    void* routingWeightsHost = nullptr;
    void* weight1Host = nullptr;
    void* weight2Host = nullptr;
    void* scale1Host = nullptr;
    void* scale2Host = nullptr;
    void* maskHost = nullptr;
    void* extraScalesHost = nullptr;
    void* yHost = nullptr;
    void* expertTokenHost = nullptr;
    if (!resources.AllocateHost(xHost, sizes.x) || !resources.AllocateHost(expertIdsHost, sizes.expertIds) ||
        !resources.AllocateHost(routingWeightsHost, sizes.routingWeights) ||
        !resources.AllocateHost(weight1Host, sizes.weight1) || !resources.AllocateHost(weight2Host, sizes.weight2) ||
        !resources.AllocateHost(scale1Host, sizes.scale1) || !resources.AllocateHost(scale2Host, sizes.scale2) ||
        !resources.AllocateHost(maskHost, sizes.mask) || !resources.AllocateHost(extraScalesHost, sizes.extraScales) ||
        !resources.AllocateHost(yHost, sizes.output) || !resources.AllocateHost(expertTokenHost, sizes.expertTokens)) {
        return 1;
    }

    FillBf16Ones(xHost, static_cast<size_t>(cfg.tokenCount) * cfg.modelDim);
    FillExpertIds(reinterpret_cast<int32_t*>(expertIdsHost), cfg);
    FillRoutingWeights(reinterpret_cast<float*>(routingWeightsHost), cfg);
    FillQuantizedWeights(weight1Host, sizes.weight1, weight2Host, sizes.weight2, cfg.mode);
    FillMxScales(scale1Host, sizes.scale1);
    FillMxScales(scale2Host, sizes.scale2);
    std::fill_n(static_cast<bool*>(maskHost), cfg.tokenCount, true);
    std::fill_n(reinterpret_cast<float*>(extraScalesHost), cfg.maxReceivedTokens, 1.0F);

    void* xDev = nullptr;
    void* expertIdsDevice = nullptr;
    void* routingWeightsDevice = nullptr;
    void* weight1Dev = nullptr;
    void* weight2Dev = nullptr;
    void* scale1Dev = nullptr;
    void* scale2Dev = nullptr;
    void* maskDev = nullptr;
    void* extraScalesDev = nullptr;
    void* yDev = nullptr;
    void* expertTokenDev = nullptr;
    void* workspaceDev = nullptr;
    if (!resources.AllocateDevice(xDev, sizes.x) || !resources.AllocateDevice(expertIdsDevice, sizes.expertIds) ||
        !resources.AllocateDevice(routingWeightsDevice, sizes.routingWeights) ||
        !resources.AllocateDevice(weight1Dev, sizes.weight1) || !resources.AllocateDevice(weight2Dev, sizes.weight2) ||
        !resources.AllocateDevice(scale1Dev, sizes.scale1) || !resources.AllocateDevice(scale2Dev, sizes.scale2) ||
        !resources.AllocateDevice(maskDev, sizes.mask) ||
        !resources.AllocateDevice(extraScalesDev, sizes.extraScales) || !resources.AllocateDevice(yDev, sizes.output) ||
        !resources.AllocateDevice(expertTokenDev, sizes.expertTokens) ||
        !resources.AllocateDevice(workspaceDev, workspaceBytes)) {
        return 1;
    }
    resources.symmetricMemory = aclshmem_malloc(symmetricMemoryBytes);
    if (resources.symmetricMemory == nullptr) {
        std::cerr << "[shmem_mega_moe] aclshmem_malloc failed, size=" << symmetricMemoryBytes << std::endl;
        return 1;
    }
    if (!CheckRuntimeStatus(aclrtMemcpy(xDev, sizes.x, xHost, sizes.x, ACL_MEMCPY_HOST_TO_DEVICE), "copy input") ||
        !CheckRuntimeStatus(
            aclrtMemcpy(expertIdsDevice, sizes.expertIds, expertIdsHost, sizes.expertIds, ACL_MEMCPY_HOST_TO_DEVICE),
            "copy expert ids") ||
        !CheckRuntimeStatus(
            aclrtMemcpy(
                routingWeightsDevice, sizes.routingWeights, routingWeightsHost, sizes.routingWeights,
                ACL_MEMCPY_HOST_TO_DEVICE),
            "copy routing weights") ||
        !CheckRuntimeStatus(
            aclrtMemcpy(weight1Dev, sizes.weight1, weight1Host, sizes.weight1, ACL_MEMCPY_HOST_TO_DEVICE),
            "copy weight1") ||
        !CheckRuntimeStatus(
            aclrtMemcpy(weight2Dev, sizes.weight2, weight2Host, sizes.weight2, ACL_MEMCPY_HOST_TO_DEVICE),
            "copy weight2") ||
        !CheckRuntimeStatus(
            aclrtMemcpy(scale1Dev, sizes.scale1, scale1Host, sizes.scale1, ACL_MEMCPY_HOST_TO_DEVICE), "copy scale1") ||
        !CheckRuntimeStatus(
            aclrtMemcpy(scale2Dev, sizes.scale2, scale2Host, sizes.scale2, ACL_MEMCPY_HOST_TO_DEVICE), "copy scale2") ||
        !CheckRuntimeStatus(
            aclrtMemcpy(maskDev, sizes.mask, maskHost, sizes.mask, ACL_MEMCPY_HOST_TO_DEVICE), "copy mask") ||
        !CheckRuntimeStatus(
            aclrtMemcpy(
                extraScalesDev, sizes.extraScales, extraScalesHost, sizes.extraScales, ACL_MEMCPY_HOST_TO_DEVICE),
            "copy extra scales") ||
        !CheckRuntimeStatus(aclrtMemset(yDev, sizes.output, 0, sizes.output), "clear output") ||
        !CheckRuntimeStatus(
            aclrtMemset(expertTokenDev, sizes.expertTokens, 0, sizes.expertTokens), "clear expert tokens") ||
        !CheckRuntimeStatus(aclrtMemset(workspaceDev, workspaceBytes, 0, workspaceBytes), "clear workspace") ||
        !CheckRuntimeStatus(
            aclrtMemset(resources.symmetricMemory, symmetricMemoryBytes, 0, symmetricMemoryBytes),
            "clear symmetric memory") ||
        !CheckRuntimeStatus(aclrtSynchronizeStream(resources.stream), "aclrtSynchronizeStream")) {
        return 1;
    }
    aclshmem_barrier_all();

    const uint64_t fftsAddr = shmemx_get_ffts_config();
    const uint32_t blockNum = std::max(1U, cfg.cubeCoreCount);
    double e2eTotalMs = 0.0;
    double e2eMinMs = 0.0;
    double e2eMaxMs = 0.0;
    int e2eSamples = 0;
    double kernelEventTotalMs = 0.0;
    double kernelEventMinMs = 0.0;
    double kernelEventMaxMs = 0.0;
    int kernelEventSamples = 0;
    for (int i = 0; i < cfg.warmupCount + cfg.loopCount; ++i) {
        if (!CheckRuntimeStatus(aclrtMemset(yDev, sizes.output, 0, sizes.output), "clear output") ||
            !CheckRuntimeStatus(
                aclrtMemset(expertTokenDev, sizes.expertTokens, 0, sizes.expertTokens), "clear expert tokens") ||
            !CheckRuntimeStatus(aclrtMemset(workspaceDev, workspaceBytes, 0, workspaceBytes), "clear workspace") ||
            !CheckRuntimeStatus(
                aclrtMemset(resources.symmetricMemory, symmetricMemoryBytes, 0, symmetricMemoryBytes),
                "clear symmetric memory") ||
            !CheckRuntimeStatus(aclrtSynchronizeStream(resources.stream), "aclrtSynchronizeStream")) {
            return 1;
        }
        aclshmem_barrier_all();

        const bool measureIteration = i >= cfg.warmupCount;
        if (measureIteration) {
            if (!CheckRuntimeStatus(aclrtCreateEvent(&resources.startEvent), "aclrtCreateEvent(start)") ||
                !CheckRuntimeStatus(aclrtCreateEvent(&resources.endEvent), "aclrtCreateEvent(end)")) {
                return 1;
            }
        }
        const auto e2eStart = std::chrono::steady_clock::now();
        if (measureIteration &&
            !CheckRuntimeStatus(aclrtRecordEvent(resources.startEvent, resources.stream), "aclrtRecordEvent(start)")) {
            return 1;
        }
        if (cfg.mode == ShmemMegaMoeExample::Mode::Arch35E5M2) {
            ShmemMegaMoeExample::shmem_mega_moe_arch35_kernel<
                bfloat16_t, bfloat16_t, float, DISPATCH_QUANT_MODE_MXFP, E5M2_QUANT, COMBINE_NO_QUANT>
                <<<blockNum, nullptr, resources.stream>>>(
                    fftsAddr, ToKernelPtr(resources.symmetricMemory), ToKernelPtr(xDev), ToKernelPtr(expertIdsDevice),
                    ToKernelPtr(routingWeightsDevice), ToKernelPtr(weight1Dev), ToKernelPtr(weight2Dev),
                    ToKernelPtr(scale1Dev), ToKernelPtr(scale2Dev), ToKernelPtr(maskDev), ToKernelPtr(extraScalesDev),
                    ToKernelPtr(yDev), ToKernelPtr(expertTokenDev), ToKernelPtr(workspaceDev), arch35Tiling);
        } else {
            ShmemMegaMoeExample::shmem_mega_moe_arch35_kernel<
                bfloat16_t, bfloat16_t, float, DISPATCH_QUANT_MODE_MXFP, E4M3_QUANT, COMBINE_NO_QUANT>
                <<<blockNum, nullptr, resources.stream>>>(
                    fftsAddr, ToKernelPtr(resources.symmetricMemory), ToKernelPtr(xDev), ToKernelPtr(expertIdsDevice),
                    ToKernelPtr(routingWeightsDevice), ToKernelPtr(weight1Dev), ToKernelPtr(weight2Dev),
                    ToKernelPtr(scale1Dev), ToKernelPtr(scale2Dev), ToKernelPtr(maskDev), ToKernelPtr(extraScalesDev),
                    ToKernelPtr(yDev), ToKernelPtr(expertTokenDev), ToKernelPtr(workspaceDev), arch35Tiling);
        }
        if (measureIteration &&
            !CheckRuntimeStatus(aclrtRecordEvent(resources.endEvent, resources.stream), "aclrtRecordEvent(end)")) {
            return 1;
        }
        if (!CheckRuntimeStatus(aclrtSynchronizeStream(resources.stream), "aclrtSynchronizeStream")) {
            return 1;
        }
        aclshmem_barrier_all();
        if (measureIteration) {
            const auto e2eEnd = std::chrono::steady_clock::now();
            const double e2eMs = std::chrono::duration<double, std::milli>(e2eEnd - e2eStart).count();
            float kernelEventMs = 0.0F;
            if (!CheckRuntimeStatus(
                    aclrtEventElapsedTime(&kernelEventMs, resources.startEvent, resources.endEvent),
                    "aclrtEventElapsedTime")) {
                return 1;
            }
            e2eTotalMs += e2eMs;
            e2eMinMs = (e2eSamples == 0) ? e2eMs : std::min(e2eMinMs, e2eMs);
            e2eMaxMs = (e2eSamples == 0) ? e2eMs : std::max(e2eMaxMs, e2eMs);
            ++e2eSamples;
            kernelEventTotalMs += kernelEventMs;
            kernelEventMinMs = (kernelEventSamples == 0) ?
                                   kernelEventMs :
                                   std::min(kernelEventMinMs, static_cast<double>(kernelEventMs));
            kernelEventMaxMs = (kernelEventSamples == 0) ?
                                   kernelEventMs :
                                   std::max(kernelEventMaxMs, static_cast<double>(kernelEventMs));
            ++kernelEventSamples;
            if (!resources.DestroyEvent(resources.startEvent) || !resources.DestroyEvent(resources.endEvent)) {
                return 1;
            }
        }
    }

    if (!CheckRuntimeStatus(
            aclrtMemcpy(yHost, sizes.output, yDev, sizes.output, ACL_MEMCPY_DEVICE_TO_HOST), "copy output") ||
        !CheckRuntimeStatus(
            aclrtMemcpy(
                expertTokenHost, sizes.expertTokens, expertTokenDev, sizes.expertTokens, ACL_MEMCPY_DEVICE_TO_HOST),
            "copy expert tokens") ||
        !ValidateOutputs(cfg, yHost, expertTokenHost)) {
        return 1;
    }

    std::cout << "[shmem_mega_moe] completed rank " << rankId << ", mode=" << argv[6]
              << ", workspace=" << workspaceBytes << ", symmetric_memory=" << symmetricMemoryBytes
              << ", y_checksum=" << ByteChecksum(yHost, sizes.output) << ", expected_y=" << ExpectedOutputValue(cfg)
              << ", expert_token_sum=" << Int32Sum(reinterpret_cast<int32_t*>(expertTokenHost), cfg.localExpertCount);
    if (e2eSamples > 0) {
        std::cout << ", e2e_avg_ms=" << (e2eTotalMs / e2eSamples) << ", e2e_min_ms=" << e2eMinMs
                  << ", e2e_max_ms=" << e2eMaxMs << ", e2e_samples=" << e2eSamples;
    } else {
        std::cout << ", e2e_samples=0";
    }
    if (kernelEventSamples > 0) {
        std::cout << ", kernel_event_avg_ms=" << (kernelEventTotalMs / kernelEventSamples)
                  << ", kernel_event_min_ms=" << kernelEventMinMs << ", kernel_event_max_ms=" << kernelEventMaxMs
                  << ", kernel_event_samples=" << kernelEventSamples;
    } else {
        std::cout << ", kernel_event_samples=0";
    }
    std::cout << ", verify=pass(expert_token,numeric_golden)" << std::endl;

    return resources.Cleanup() ? 0 : 1;
}
