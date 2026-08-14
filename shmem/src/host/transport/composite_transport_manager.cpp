/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "composite_transport_manager.h"

#include <algorithm>
#include <utility>
#include "shmemi_logger.h"

namespace shm {
namespace transport {

CompositeTransportManager::CompositeTransportManager(std::vector<TransportType> order)
    : CompositeTransportManager(std::move(order), TransportManager::Create)
{}

CompositeTransportManager::CompositeTransportManager(std::vector<TransportType> order, ManagerFactory managerFactory)
    : order_(std::move(order)), managerFactory_(std::move(managerFactory))
{}

TransManagerPtr CompositeTransportManager::GetManager(TransportType type) const
{
    if (type < TT_HCCP || type >= TT_BUTT) {
        return nullptr;
    }
    return managers_[static_cast<size_t>(type)];
}

void CompositeTransportManager::CloseManagersReverse()
{
    for (auto it = order_.rbegin(); it != order_.rend(); ++it) {
        auto index = static_cast<size_t>(*it);
        if (managers_[index] != nullptr) {
            (void)managers_[index]->CloseDevice();
            managers_[index] = nullptr;
        }
    }
    connected_ = false;
}

Result CompositeTransportManager::OpenDevice(const TransportOptions& options)
{
    for (auto type : order_) {
        auto manager = managerFactory_(type);
        if (manager == nullptr) {
            SHM_LOG_ERROR("create transport manager failed, type: " << type);
            CloseManagersReverse();
            return ACLSHMEM_NOT_SUPPORTED;
        }

        auto ret = manager->OpenDevice(options);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("open transport manager failed, type: " << type << ", ret: " << ret);
            (void)manager->CloseDevice();
            CloseManagersReverse();
            return ret;
        }
        managers_[static_cast<size_t>(type)] = std::move(manager);
    }
    return ACLSHMEM_SUCCESS;
}

Result CompositeTransportManager::CloseDevice()
{
    Result result = ACLSHMEM_SUCCESS;
    for (auto it = order_.rbegin(); it != order_.rend(); ++it) {
        auto index = static_cast<size_t>(*it);
        if (managers_[index] == nullptr) {
            continue;
        }
        auto ret = managers_[index]->CloseDevice();
        if (ret != ACLSHMEM_SUCCESS && result == ACLSHMEM_SUCCESS) {
            result = ret;
        }
        managers_[index] = nullptr;
    }
    connected_ = false;
    return result;
}

Result CompositeTransportManager::RegisterMemoryRegion(const TransportMemoryRegion& mr)
{
    std::vector<TransportType> registered;
    for (auto type : order_) {
        auto manager = GetManager(type);
        if (manager == nullptr) {
            continue;
        }
        auto ret = manager->RegisterMemoryRegion(mr);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("register memory region failed, type: " << type << ", ret: " << ret);
            for (auto it = registered.rbegin(); it != registered.rend(); ++it) {
                (void)GetManager(*it)->UnregisterMemoryRegion(mr.addr);
            }
            return ret;
        }
        registered.push_back(type);
    }
    return ACLSHMEM_SUCCESS;
}

Result CompositeTransportManager::UnregisterMemoryRegion(uint64_t addr)
{
    Result result = ACLSHMEM_SUCCESS;
    for (auto it = order_.rbegin(); it != order_.rend(); ++it) {
        auto manager = GetManager(*it);
        if (manager == nullptr) {
            continue;
        }
        auto ret = manager->UnregisterMemoryRegion(addr);
        if (ret != ACLSHMEM_SUCCESS && result == ACLSHMEM_SUCCESS) {
            result = ret;
        }
    }
    return result;
}

Result CompositeTransportManager::QueryMemoryKey(uint64_t addr, TransportMemoryKey& key)
{
    auto rdma = GetManager(TT_HCCP);
    if (rdma == nullptr) {
        return ACLSHMEM_SUCCESS;
    }
    return rdma->QueryMemoryKey(addr, key);
}

Result CompositeTransportManager::ParseMemoryKey(const TransportMemoryKey& key, uint64_t& addr, uint64_t& size)
{
    auto rdma = GetManager(TT_HCCP);
    if (rdma == nullptr) {
        return ACLSHMEM_SUCCESS;
    }
    return rdma->ParseMemoryKey(key, addr, size);
}

Result CompositeTransportManager::Prepare(const HybmTransPrepareOptions& options)
{
    for (auto type : order_) {
        auto manager = GetManager(type);
        if (manager == nullptr) {
            continue;
        }
        auto ret = manager->Prepare(options);
        if (ret != ACLSHMEM_SUCCESS) {
            return ret;
        }
    }
    return ACLSHMEM_SUCCESS;
}

Result CompositeTransportManager::Connect()
{
    for (auto type : order_) {
        auto manager = GetManager(type);
        if (manager == nullptr) {
            continue;
        }
        auto ret = manager->Connect();
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("connect transport manager failed, type: " << type << ", ret: " << ret);
            (void)CloseDevice();
            return ret;
        }
    }
    return ACLSHMEM_SUCCESS;
}

Result CompositeTransportManager::ConnectWithOptions(const HybmTransPrepareOptions& options)
{
    for (auto type : order_) {
        auto manager = GetManager(type);
        if (manager == nullptr) {
            continue;
        }
        auto ret = manager->ConnectWithOptions(options);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("connect transport manager failed, type: " << type << ", ret: " << ret);
            (void)CloseDevice();
            return ret;
        }
    }
    connected_ = true;
    return ACLSHMEM_SUCCESS;
}

Result CompositeTransportManager::AsyncConnect()
{
    for (auto type : order_) {
        auto manager = GetManager(type);
        if (manager == nullptr) {
            continue;
        }
        auto ret = manager->AsyncConnect();
        if (ret != ACLSHMEM_SUCCESS) {
            return ret;
        }
    }
    return ACLSHMEM_SUCCESS;
}

Result CompositeTransportManager::WaitForConnected(int64_t timeoutNs)
{
    for (auto type : order_) {
        auto manager = GetManager(type);
        if (manager == nullptr) {
            continue;
        }
        auto ret = manager->WaitForConnected(timeoutNs);
        if (ret != ACLSHMEM_SUCCESS) {
            return ret;
        }
    }
    return ACLSHMEM_SUCCESS;
}

Result CompositeTransportManager::UpdateRankOptions(const HybmTransPrepareOptions& options)
{
    for (auto type : order_) {
        auto manager = GetManager(type);
        if (manager == nullptr) {
            continue;
        }
        auto ret = manager->UpdateRankOptions(options);
        if (ret != ACLSHMEM_SUCCESS) {
            return ret;
        }
    }
    return ACLSHMEM_SUCCESS;
}

const std::string& CompositeTransportManager::GetNic() const
{
    auto rdma = GetManager(TT_HCCP);
    if (rdma != nullptr) {
        return rdma->GetNic();
    }
    static const std::string empty_nic;
    return empty_nic;
}

const void* CompositeTransportManager::GetQpInfo() const
{
    auto rdma = GetManager(TT_HCCP);
    if (rdma == nullptr) {
        return nullptr;
    }
    return rdma->GetQpInfo();
}

TransportDeviceInfo CompositeTransportManager::GetDeviceInfo() const
{
    TransportDeviceInfo info;
    auto rdma = GetManager(TT_HCCP);
    if (rdma != nullptr) {
        info.rdmaInfoAddress = reinterpret_cast<uint64_t>(rdma->GetQpInfo());
    }
    auto udma = GetManager(TT_UDMA);
    if (udma != nullptr) {
        info.udmaInfoAddress = reinterpret_cast<uint64_t>(udma->GetQpInfo());
    }
    return info;
}

} // namespace transport
} // namespace shm
