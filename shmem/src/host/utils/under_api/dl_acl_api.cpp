/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <dlfcn.h>
#include "dl_acl_api.h"
#include "dl_hal_api.h"
#include "shmemi_file_util.h"

namespace shm {
bool DlAclApi::gLoaded = false;
std::mutex DlAclApi::gMutex;
void* DlAclApi::rtHandle;
void* DlAclApi::runtimeHandle;
const char* DlAclApi::gAscendAclLibName = "libascendcl.so";
const char* DlAclApi::gAscendRuntimeLibName = "libruntime.so";

aclrtGetDeviceFunc DlAclApi::pAclrtGetDevice = nullptr;
aclrtSetDeviceFunc DlAclApi::pAclrtSetDevice = nullptr;
aclrtDeviceEnablePeerAccessFunc DlAclApi::pAclrtDeviceEnablePeerAccess = nullptr;
aclrtCreateStreamFunc DlAclApi::pAclrtCreateStream = nullptr;
aclrtDestroyStreamFunc DlAclApi::pAclrtDestroyStream = nullptr;
aclrtSynchronizeStreamFunc DlAclApi::pAclrtSynchronizeStream = nullptr;
aclrtMallocFunc DlAclApi::pAclrtMalloc = nullptr;
aclrtFreeFunc DlAclApi::pAclrtFree = nullptr;
aclrtMemcpyFunc DlAclApi::pAclrtMemcpy = nullptr;
aclrtMemcpyAsyncFunc DlAclApi::pAclrtMemcpyAsync = nullptr;
aclrtMemsetFunc DlAclApi::pAclrtMemset = nullptr;
aclrtSetExceptionInfoCallbackFunc DlAclApi::pAclrtSetExceptionInfoCallback = nullptr;
aclrtGetExceptionInfoFieldFunc DlAclApi::pAclrtGetTaskIdFromExceptionInfo = nullptr;
aclrtGetExceptionInfoFieldFunc DlAclApi::pAclrtGetStreamIdFromExceptionInfo = nullptr;
aclrtGetExceptionInfoFieldFunc DlAclApi::pAclrtGetThreadIdFromExceptionInfo = nullptr;
aclrtGetExceptionInfoFieldFunc DlAclApi::pAclrtGetDeviceIdFromExceptionInfo = nullptr;
aclrtGetExceptionInfoFieldFunc DlAclApi::pAclrtGetErrorCodeFromExceptionInfo = nullptr;
rtDeviceGetBareTgidFunc DlAclApi::pRtDeviceGetBareTgid = nullptr;
rtGetDeviceInfoFunc DlAclApi::pRtGetDeviceInfo = nullptr;
rtSetIpcMemorySuperPodPidFunc DlAclApi::pRtSetIpcMemorySuperPodPid = nullptr;
rtIpcDestroyMemoryNameFunc DlAclApi::pRtIpcDestroyMemoryName = nullptr;
rtIpcSetMemoryNameFunc DlAclApi::pRtIpcSetMemoryName = nullptr;
rtIpcOpenMemoryFunc DlAclApi::pRtIpcOpenMemory = nullptr;
rtIpcCloseMemoryFunc DlAclApi::pRtIpcCloseMemory = nullptr;
aclrtGetSocNameFunc DlAclApi::pAclrtGetSocName = nullptr;
rtGetLogicDevIdByUserDevIdFunc DlAclApi::pRtGetLogicDevIdByUserDevId = nullptr;
aclrtGetUserDevIdByPhyDevIdFunc DlAclApi::pAclrtGetUserDevIdByPhyDevId = nullptr;
aclrtGetPhyDevIdByUserDevIdFunc DlAclApi::pAclrtGetPhyDevIdByUserDevId = nullptr;
aclrtGetPhyDevIdByLogicDevIdFunc DlAclApi::pAclrtGetPhyDevIdByLogicDevId = nullptr;
rtGetDevicePhyIdByIndexFunc DlAclApi::pRtGetDevicePhyIdByIndex = nullptr;
rtEnableP2PFunc DlAclApi::pRtEnableP2P = nullptr;
aclrtReserveMemAddressFunc DlAclApi::pAclrtReserveMemAddress = nullptr;
aclrtReleaseMemAddressFunc DlAclApi::pAclrtReleaseMemAddress = nullptr;

Result DlAclApi::LoadLibrary(const std::string& libDirPath)
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (gLoaded) {
        return ACLSHMEM_SUCCESS;
    }

    std::string realPath;
    if (!shm::utils::FileUtil::LibraryRealPath(libDirPath, std::string(gAscendAclLibName), realPath)) {
        SHM_LOG_ERROR(libDirPath << " get lib path failed.");
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    /* dlopen library */
    rtHandle = dlopen(realPath.c_str(), RTLD_NOW);
    if (rtHandle == nullptr) {
        SHM_LOG_ERROR("Failed to open library error: " << dlerror());
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    /* load sym */
    DL_LOAD_SYM(pAclrtGetDevice, aclrtGetDeviceFunc, rtHandle, "aclrtGetDevice");
    DL_LOAD_SYM(pAclrtSetDevice, aclrtSetDeviceFunc, rtHandle, "aclrtSetDevice");
    DL_LOAD_SYM(pAclrtDeviceEnablePeerAccess, aclrtDeviceEnablePeerAccessFunc, rtHandle, "aclrtDeviceEnablePeerAccess");
    DL_LOAD_SYM(pAclrtCreateStream, aclrtCreateStreamFunc, rtHandle, "aclrtCreateStream");
    DL_LOAD_SYM(pAclrtDestroyStream, aclrtDestroyStreamFunc, rtHandle, "aclrtDestroyStream");
    DL_LOAD_SYM(pAclrtSynchronizeStream, aclrtSynchronizeStreamFunc, rtHandle, "aclrtSynchronizeStream");
    DL_LOAD_SYM(pAclrtMalloc, aclrtMallocFunc, rtHandle, "aclrtMalloc");
    DL_LOAD_SYM(pAclrtFree, aclrtFreeFunc, rtHandle, "aclrtFree");
    DL_LOAD_SYM(pAclrtMemcpy, aclrtMemcpyFunc, rtHandle, "aclrtMemcpy");
    DL_LOAD_SYM(pAclrtMemcpyAsync, aclrtMemcpyAsyncFunc, rtHandle, "aclrtMemcpyAsync");
    DL_LOAD_SYM(pAclrtMemset, aclrtMemsetFunc, rtHandle, "aclrtMemset");
    pAclrtSetExceptionInfoCallback =
        reinterpret_cast<aclrtSetExceptionInfoCallbackFunc>(dlsym(rtHandle, "aclrtSetExceptionInfoCallback"));
    pAclrtGetTaskIdFromExceptionInfo =
        reinterpret_cast<aclrtGetExceptionInfoFieldFunc>(dlsym(rtHandle, "aclrtGetTaskIdFromExceptionInfo"));
    pAclrtGetStreamIdFromExceptionInfo =
        reinterpret_cast<aclrtGetExceptionInfoFieldFunc>(dlsym(rtHandle, "aclrtGetStreamIdFromExceptionInfo"));
    pAclrtGetThreadIdFromExceptionInfo =
        reinterpret_cast<aclrtGetExceptionInfoFieldFunc>(dlsym(rtHandle, "aclrtGetThreadIdFromExceptionInfo"));
    pAclrtGetDeviceIdFromExceptionInfo =
        reinterpret_cast<aclrtGetExceptionInfoFieldFunc>(dlsym(rtHandle, "aclrtGetDeviceIdFromExceptionInfo"));
    pAclrtGetErrorCodeFromExceptionInfo =
        reinterpret_cast<aclrtGetExceptionInfoFieldFunc>(dlsym(rtHandle, "aclrtGetErrorCodeFromExceptionInfo"));
    if (!AclrtExceptionInfoApisAvailable()) {
        SHM_LOG_DEBUG("Optional runtime exception report symbols are not fully loaded.");
    }
    DL_LOAD_SYM(pRtDeviceGetBareTgid, rtDeviceGetBareTgidFunc, rtHandle, "rtDeviceGetBareTgid");
    DL_LOAD_SYM(pRtGetDeviceInfo, rtGetDeviceInfoFunc, rtHandle, "rtGetDeviceInfo");
    DL_LOAD_SYM(pRtSetIpcMemorySuperPodPid, rtSetIpcMemorySuperPodPidFunc, rtHandle, "rtSetIpcMemorySuperPodPid");
    DL_LOAD_SYM(pRtIpcSetMemoryName, rtIpcSetMemoryNameFunc, rtHandle, "rtIpcSetMemoryName");
    DL_LOAD_SYM(pRtIpcDestroyMemoryName, rtIpcDestroyMemoryNameFunc, rtHandle, "rtIpcDestroyMemoryName");
    DL_LOAD_SYM(pRtIpcOpenMemory, rtIpcOpenMemoryFunc, rtHandle, "rtIpcOpenMemory");
    DL_LOAD_SYM(pRtIpcCloseMemory, rtIpcCloseMemoryFunc, rtHandle, "rtIpcCloseMemory");
    DL_LOAD_SYM(pAclrtGetSocName, aclrtGetSocNameFunc, rtHandle, "aclrtGetSocName");
    DL_LOAD_SYM(pRtGetLogicDevIdByUserDevId, rtGetLogicDevIdByUserDevIdFunc, rtHandle, "rtGetLogicDevIdByUserDevId");

    pAclrtGetUserDevIdByPhyDevId =
        reinterpret_cast<aclrtGetUserDevIdByPhyDevIdFunc>(dlsym(rtHandle, "aclrtGetUserDevIdByPhyDevId"));
    if (pAclrtGetUserDevIdByPhyDevId == nullptr) {
        pAclrtGetUserDevIdByPhyDevId =
            reinterpret_cast<aclrtGetUserDevIdByPhyDevIdFunc>(dlsym(rtHandle, "aclrtGetLogicDevIdByPhyDevId"));
    }
    if (pAclrtGetUserDevIdByPhyDevId == nullptr) {
        SHM_LOG_WARN("Optional symbol aclrtGetUserDevIdByPhyDevId is not loaded.");
    }

    pAclrtGetPhyDevIdByUserDevId =
        reinterpret_cast<aclrtGetPhyDevIdByUserDevIdFunc>(dlsym(rtHandle, "aclrtGetPhyDevIdByUserDevId"));
    if (pAclrtGetPhyDevIdByUserDevId == nullptr) {
        SHM_LOG_WARN("Optional symbol aclrtGetPhyDevIdByUserDevId is not loaded.");
    }

    pAclrtGetPhyDevIdByLogicDevId =
        reinterpret_cast<aclrtGetPhyDevIdByLogicDevIdFunc>(dlsym(rtHandle, "aclrtGetPhyDevIdByLogicDevId"));
    if (pAclrtGetPhyDevIdByLogicDevId == nullptr) {
        SHM_LOG_WARN("Optional symbol aclrtGetPhyDevIdByLogicDevId is not loaded.");
    }

    pAclrtReserveMemAddress = reinterpret_cast<aclrtReserveMemAddressFunc>(dlsym(rtHandle, "aclrtReserveMemAddress"));
    pAclrtReleaseMemAddress = reinterpret_cast<aclrtReleaseMemAddressFunc>(dlsym(rtHandle, "aclrtReleaseMemAddress"));
    if (pAclrtReserveMemAddress == nullptr || pAclrtReleaseMemAddress == nullptr) {
        pAclrtReserveMemAddress = nullptr;
        pAclrtReleaseMemAddress = nullptr;
        SHM_LOG_WARN("Optional symbol aclrtReserveMemAddress/aclrtReleaseMemAddress is not loaded, "
                     "nullptr reserve requires aclrt; specified-address reserve will use Hal.");
    }

    std::string runtimePath;
    if (shm::utils::FileUtil::LibraryRealPath(libDirPath, std::string(gAscendRuntimeLibName), runtimePath)) {
        runtimeHandle = dlopen(runtimePath.c_str(), RTLD_NOW | RTLD_LOCAL);
    } else {
        runtimeHandle = dlopen(gAscendRuntimeLibName, RTLD_NOW | RTLD_LOCAL);
    }
    if (runtimeHandle == nullptr) {
        SHM_LOG_ERROR("Failed to open libruntime.so error: " << dlerror());
        dlclose(rtHandle);
        rtHandle = nullptr;
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    pRtGetDevicePhyIdByIndex =
        reinterpret_cast<rtGetDevicePhyIdByIndexFunc>(dlsym(runtimeHandle, "rtGetDevicePhyIdByIndex"));
    pRtEnableP2P = reinterpret_cast<rtEnableP2PFunc>(dlsym(runtimeHandle, "rtEnableP2P"));
    if (pRtEnableP2P == nullptr) {
        SHM_LOG_WARN("Optional symbol rtEnableP2P is not loaded, grouped visible P2P falls back to "
                     "aclrtDeviceEnablePeerAccess.");
    }
    if (pRtGetDevicePhyIdByIndex == nullptr && pAclrtGetPhyDevIdByUserDevId == nullptr &&
        pAclrtGetPhyDevIdByLogicDevId == nullptr) {
        SHM_LOG_ERROR("No phy id mapping API available (aclrtGetPhyDevIdByUserDevId, "
                      "aclrtGetPhyDevIdByLogicDevId, rtGetDevicePhyIdByIndex).");
        dlclose(runtimeHandle);
        runtimeHandle = nullptr;
        dlclose(rtHandle);
        rtHandle = nullptr;
        return ACLSHMEM_DL_FUNC_FAILED;
    }
    if (pAclrtGetPhyDevIdByUserDevId != nullptr) {
        SHM_LOG_INFO("Use aclrtGetPhyDevIdByUserDevId for phy id mapping.");
    } else if (pAclrtGetPhyDevIdByLogicDevId == nullptr) {
        SHM_LOG_INFO("aclrtGetPhyDevIdByLogicDevId not found, use rtGetDevicePhyIdByIndex for phy id mapping.");
    }

    gLoaded = true;
    SHM_LOG_INFO("Load " << realPath << " success.");
    return ACLSHMEM_SUCCESS;
}

Result DlAclApi::AclrtReserveMemAddress(void** virPtr, size_t size, size_t alignment, void* expectPtr, uint64_t flags)
{
    if (expectPtr == nullptr) {
        if (pAclrtReserveMemAddress == nullptr) {
            SHM_LOG_ERROR("AclrtReserveMemAddress is not available, nullptr reserve requires aclrt.");
            return ACLSHMEM_DL_FUNC_FAILED;
        }
        return pAclrtReserveMemAddress(virPtr, size, alignment, nullptr, flags);
    }

    if (pAclrtReserveMemAddress != nullptr) {
        auto ret = pAclrtReserveMemAddress(virPtr, size, alignment, expectPtr, flags);
        if (ret == 0) {
            return ret;
        }
        SHM_LOG_WARN(
            "AclrtReserveMemAddress specified failed ret=" << ret << ", fallback to HalMemAddressReserve expectAddr="
                                                           << expectPtr);
    }
    return DlHalApi::HalMemAddressReserve(virPtr, size, alignment, expectPtr, flags);
}

Result DlAclApi::AclrtReleaseMemAddress(void* virPtr)
{
    if (pAclrtReleaseMemAddress != nullptr) {
        auto ret = pAclrtReleaseMemAddress(virPtr);
        if (ret == 0) {
            return ret;
        }
    }
    return DlHalApi::HalMemAddressFree(virPtr);
}

void DlAclApi::CleanupLibrary()
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (!gLoaded) {
        return;
    }

    pAclrtGetDevice = nullptr;
    pAclrtSetDevice = nullptr;
    pAclrtDeviceEnablePeerAccess = nullptr;
    pAclrtCreateStream = nullptr;
    pAclrtDestroyStream = nullptr;
    pAclrtSynchronizeStream = nullptr;
    pAclrtMalloc = nullptr;
    pAclrtFree = nullptr;
    pAclrtMemcpy = nullptr;
    pAclrtMemcpyAsync = nullptr;
    pAclrtMemset = nullptr;
    pAclrtSetExceptionInfoCallback = nullptr;
    pAclrtGetTaskIdFromExceptionInfo = nullptr;
    pAclrtGetStreamIdFromExceptionInfo = nullptr;
    pAclrtGetThreadIdFromExceptionInfo = nullptr;
    pAclrtGetDeviceIdFromExceptionInfo = nullptr;
    pAclrtGetErrorCodeFromExceptionInfo = nullptr;
    pRtDeviceGetBareTgid = nullptr;
    pRtGetDeviceInfo = nullptr;
    pRtSetIpcMemorySuperPodPid = nullptr;
    pRtIpcDestroyMemoryName = nullptr;
    pRtIpcSetMemoryName = nullptr;
    pAclrtGetSocName = nullptr;
    pRtGetLogicDevIdByUserDevId = nullptr;
    pAclrtGetUserDevIdByPhyDevId = nullptr;
    pAclrtGetPhyDevIdByUserDevId = nullptr;
    pAclrtGetPhyDevIdByLogicDevId = nullptr;
    pRtGetDevicePhyIdByIndex = nullptr;
    pRtEnableP2P = nullptr;
    pAclrtReserveMemAddress = nullptr;
    pAclrtReleaseMemAddress = nullptr;

    if (runtimeHandle != nullptr) {
        dlclose(runtimeHandle);
        runtimeHandle = nullptr;
    }

    if (rtHandle != nullptr) {
        dlclose(rtHandle);
        rtHandle = nullptr;
    }

    gLoaded = false;
}
} // namespace shm
