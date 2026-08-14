/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dl_hcomm_api.h"

#include <dlfcn.h>

#include "shmemi_functions.h"
#include "utils/shmemi_logger.h"

namespace shm {

std::mutex DlHcommApi::gMutex;
bool DlHcommApi::gLoaded = false;
void* DlHcommApi::gHcommHandle = nullptr;
const char* DlHcommApi::gHcommLibName = "libhcomm.so";

HcommEndpointCreateFunc DlHcommApi::gHcommEndpointCreate = nullptr;
HcommEndpointDestroyFunc DlHcommApi::gHcommEndpointDestroy = nullptr;
HcommEndpointGetListenPortFunc DlHcommApi::gHcommEndpointGetListenPort = nullptr;
HcommMemRegFunc DlHcommApi::gHcommMemReg = nullptr;
HcommMemUnregFunc DlHcommApi::gHcommMemUnreg = nullptr;
HcommChannelCreateFunc DlHcommApi::gHcommChannelCreate = nullptr;
HcommChannelDestroyFunc DlHcommApi::gHcommChannelDestroy = nullptr;
HcommChannelGetStatusFunc DlHcommApi::gHcommChannelGetStatus = nullptr;

Result DlHcommApi::LoadLibrary()
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (gLoaded) {
        return ACLSHMEM_SUCCESS;
    }

    gHcommHandle = dlopen(gHcommLibName, RTLD_NOW);
    if (gHcommHandle == nullptr) {
        SHM_LOG_ERROR(
            "Failed to open library ["
            << gHcommLibName
            << "], please source ascend-toolkit set_env.sh, or add ascend lib path into LD_LIBRARY_PATH, error: "
            << dlerror());
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    DL_LOAD_SYM(gHcommEndpointCreate, HcommEndpointCreateFunc, gHcommHandle, "HcommEndpointCreate");
    DL_LOAD_SYM(gHcommEndpointDestroy, HcommEndpointDestroyFunc, gHcommHandle, "HcommEndpointDestroy");
    DL_LOAD_SYM(
        gHcommEndpointGetListenPort, HcommEndpointGetListenPortFunc, gHcommHandle, "HcommEndpointGetListenPort");
    DL_LOAD_SYM(gHcommMemReg, HcommMemRegFunc, gHcommHandle, "HcommMemReg");
    DL_LOAD_SYM(gHcommMemUnreg, HcommMemUnregFunc, gHcommHandle, "HcommMemUnreg");
    DL_LOAD_SYM(gHcommChannelCreate, HcommChannelCreateFunc, gHcommHandle, "HcommChannelCreate");
    DL_LOAD_SYM(gHcommChannelDestroy, HcommChannelDestroyFunc, gHcommHandle, "HcommChannelDestroy");
    DL_LOAD_SYM(gHcommChannelGetStatus, HcommChannelGetStatusFunc, gHcommHandle, "HcommChannelGetStatus");

    Dl_info hcomm_lib_info{};
    if (dladdr(reinterpret_cast<void*>(gHcommChannelCreate), &hcomm_lib_info) != 0 &&
        hcomm_lib_info.dli_fname != nullptr) {
        SHM_LOG_INFO("HcommChannelCreate loaded from " << hcomm_lib_info.dli_fname);
    }
    SHM_LOG_INFO("LoadLibrary for DlHcommApi success");
    gLoaded = true;
    return ACLSHMEM_SUCCESS;
}

void DlHcommApi::CleanupLibrary()
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (!gLoaded) {
        return;
    }

    gHcommEndpointCreate = nullptr;
    gHcommEndpointDestroy = nullptr;
    gHcommEndpointGetListenPort = nullptr;
    gHcommMemReg = nullptr;
    gHcommMemUnreg = nullptr;
    gHcommChannelCreate = nullptr;
    gHcommChannelDestroy = nullptr;
    gHcommChannelGetStatus = nullptr;

    if (gHcommHandle != nullptr) {
        dlclose(gHcommHandle);
        gHcommHandle = nullptr;
    }
    gLoaded = false;
}

} // namespace shm
