/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBRID_DEVICE_RDMA_TRANSPORT_MANAGER_H
#define MF_HYBRID_DEVICE_RDMA_TRANSPORT_MANAGER_H

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <map>
#include <mutex>
#include <memory>

#include "mem_entity_def.h"
#include "transport_manager.h"
#include "device_chip_info.h"
#include "device_rdma_common.h"
#include "device_qp_manager.h"

namespace shm {
namespace transport {
namespace device {
class RdmaTransportManager : public TransportManager {
public:
    ~RdmaTransportManager() override;
    Result OpenDevice(const TransportOptions& options) override;
    Result CloseDevice() override;
    Result RegisterMemoryRegion(const TransportMemoryRegion& mr) override;
    Result UnregisterMemoryRegion(uint64_t addr) override;
    Result QueryMemoryKey(uint64_t addr, TransportMemoryKey& key) override;
    Result ParseMemoryKey(const TransportMemoryKey& key, uint64_t& addr, uint64_t& size) override;
    Result Prepare(const HybmTransPrepareOptions& options) override;
    Result Connect() override;
    Result AsyncConnect() override;
    Result WaitForConnected(int64_t timeoutNs) override;
    Result UpdateRankOptions(const HybmTransPrepareOptions& options) override;
    const std::string& GetNic() const override;
    const void* GetQpInfo() const override;

private:
    struct RaResourceState {
        uint32_t refCount{0};
        bool selfOwned{false};
        bool deviceIpRetired{false};
        net_addr_t retiredIp{};
        void* rdmaHandle{nullptr};
    };

    bool PrepareOpenDevice(
        uint32_t userId, uint32_t phyId, uint32_t rankCount, net_addr_t& deviceIp, void*& rdmaHandle);
    static bool OpenTsd(uint32_t userId, uint32_t rankCount);
    static bool TryGetRdmaHandleAndReferenceManagedRa(uint32_t phyId, void*& rdmaHandle, bool& referenced);
    static bool RaInit(uint32_t phyId);
    static void RaDeinit(uint32_t phyId);
    static void ResetOpenDeviceState(uint32_t phyId);
    static bool HandleRetiredDeviceIp(uint32_t phyId, net_addr_t& deviceIp);
    static bool RetireDeviceIp(uint32_t phyId, net_addr_t& deviceIp);
    static bool RaRdevInit(uint32_t phyId, net_addr_t deviceIp, void*& rdmaHandle);
    void ClearAllRegisterMRs();
    int CheckPrepareOptions(const HybmTransPrepareOptions& options);
    void InitializeDeviceAddress(mf_sockaddr& deviceAddr);

private:
    bool started_{false};
    bool raReferenced_{false};
    uint32_t rankId_{0};
    uint32_t rankCount_{1};
    uint32_t deviceId_{0};
    uint32_t phyId_{0};
    hybm_role_type role_{HYBM_ROLE_PEER};
    net_addr_t deviceIp_{};
    uint16_t devicePort_{0};
    void* rdmaHandle_{nullptr};
    static bool tsdOpened_;
    static std::mutex raMutex_;
    static std::map<uint32_t, RaResourceState> raInstances_;
    std::string nicInfo_;
    MemoryRegionMap registerMRS_;
    std::shared_ptr<DeviceQpManager> qpManager_;
    std::shared_ptr<DeviceChipInfo> deviceChipInfo_;
};
} // namespace device
} // namespace transport
} // namespace shm

#endif // MF_HYBRID_DEVICE_RDMA_TRANSPORT_MANAGER_H
