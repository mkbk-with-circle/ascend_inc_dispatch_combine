/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBRID_HYBM_COMPOSITE_TRANSPORT_MANAGER_H
#define MF_HYBRID_HYBM_COMPOSITE_TRANSPORT_MANAGER_H

#include <array>
#include <functional>
#include <vector>
#include "transport_manager.h"

namespace shm {
namespace transport {

class CompositeTransportManager : public TransportManager {
public:
    using ManagerFactory = std::function<TransManagerPtr(TransportType)>;

    explicit CompositeTransportManager(std::vector<TransportType> order);
    CompositeTransportManager(std::vector<TransportType> order, ManagerFactory managerFactory);
    ~CompositeTransportManager() override = default;

    Result OpenDevice(const TransportOptions& options) override;
    Result CloseDevice() override;
    Result RegisterMemoryRegion(const TransportMemoryRegion& mr) override;
    Result UnregisterMemoryRegion(uint64_t addr) override;
    Result QueryMemoryKey(uint64_t addr, TransportMemoryKey& key) override;
    Result ParseMemoryKey(const TransportMemoryKey& key, uint64_t& addr, uint64_t& size) override;
    Result Prepare(const HybmTransPrepareOptions& options) override;
    Result Connect() override;
    Result ConnectWithOptions(const HybmTransPrepareOptions& options) override;
    Result AsyncConnect() override;
    Result WaitForConnected(int64_t timeoutNs) override;
    Result UpdateRankOptions(const HybmTransPrepareOptions& options) override;
    const std::string& GetNic() const override;
    const void* GetQpInfo() const override;
    TransportDeviceInfo GetDeviceInfo() const override;

private:
    TransManagerPtr GetManager(TransportType type) const;
    void CloseManagersReverse();

private:
    std::array<TransManagerPtr, TT_BUTT> managers_{};
    std::vector<TransportType> order_;
    ManagerFactory managerFactory_;
};

} // namespace transport
} // namespace shm

#endif // MF_HYBRID_HYBM_COMPOSITE_TRANSPORT_MANAGER_H
