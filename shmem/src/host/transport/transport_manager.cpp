/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "transport_manager.h"

#include "shmemi_logger.h"
#include "composite_transport_manager.h"
#include "transport_def.h"
#include "device_rdma_transport_manager.h"
#if defined(ACLSHMEM_RDMA_V2_SUPPORT)
#include "device_rdma_transport_manager_v2.h"
#endif
#include "device_sdma_transport_manager.h"
#if defined(ACLSHMEM_SOC_950)
#include "device_udma_transport_manager.h"
#endif

namespace shm {
namespace transport {

std::shared_ptr<TransportManager> TransportManager::Create(TransportType type)
{
    switch (type) {
        case TT_HCCP:
#if defined(ACLSHMEM_RDMA_V2_SUPPORT)
            return std::make_shared<device::RdmaTransportManagerV2>();
#else
            return std::make_shared<device::RdmaTransportManager>();
#endif
        case TT_SDMA:
            return std::make_shared<device::SdmaTransportManager>();
#if defined(ACLSHMEM_SOC_950)
        case TT_UDMA:
            return std::make_shared<device::UdmaTransportManager>();
#endif
        default:
            SHM_LOG_ERROR("Invalid trans type: " << type);
            return nullptr;
    }
}

std::shared_ptr<TransportManager> TransportManager::CreateForDataOpType(uint32_t dataOpType)
{
    std::vector<TransportType> order;
    if ((dataOpType & HYBM_DOP_TYPE_DEVICE_RDMA) != 0) {
        order.push_back(TT_HCCP);
    }
    if ((dataOpType & HYBM_DOP_TYPE_DEVICE_SDMA) != 0) {
        order.push_back(TT_SDMA);
    }
    if ((dataOpType & HYBM_DOP_TYPE_DEVICE_UDMA) != 0) {
        order.push_back(TT_UDMA);
    }

    if (order.empty()) {
        return nullptr;
    }
    if (order.size() == 1) {
        return Create(order.front());
    }
    return std::make_shared<CompositeTransportManager>(std::move(order));
}

const void* TransportManager::GetQpInfo() const
{
    SHM_LOG_DEBUG("Not Implement GetQpInfo()");
    return nullptr;
}

TransportDeviceInfo TransportManager::GetDeviceInfo() const
{
    TransportDeviceInfo info;
    info.rdmaInfoAddress = reinterpret_cast<uint64_t>(GetQpInfo());
    return info;
}

Result TransportManager::ConnectWithOptions(const HybmTransPrepareOptions& options)
{
    SHM_LOG_DEBUG("ConnectWithOptions now connected=" << connected_);
    if (!connected_) {
        auto ret = Prepare(options);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("prepare connection failed: " << ret);
            return ret;
        }

        ret = Connect();
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("connect failed: " << ret);
            return ret;
        }

        connected_ = true;
        return ACLSHMEM_SUCCESS;
    }

    return UpdateRankOptions(options);
}

} // namespace transport
} // namespace shm
