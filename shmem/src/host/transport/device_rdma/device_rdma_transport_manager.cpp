/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mem_entity_def.h"
#include "shmemi_logger.h"
#include "dl_acl_api.h"
#include "dl_hccp_api.h"
#include "device_rdma_common.h"
#include "device_rdma_helper.h"
#include "fixed_ranks_qp_manager.h"
#include "dynamic_ranks_qp_manager.h"
#include "device_rdma_transport_manager.h"

namespace shm {
namespace transport {
namespace device {
bool RdmaTransportManager::tsdOpened_ = false;
std::mutex RdmaTransportManager::raMutex_;
std::map<uint32_t, RdmaTransportManager::RaResourceState> RdmaTransportManager::raInstances_;

RdmaTransportManager::~RdmaTransportManager()
{
    ClearAllRegisterMRs();
    if (raReferenced_) {
        RaDeinit(phyId_);
        raReferenced_ = false;
    } else {
        ResetOpenDeviceState(phyId_);
    }
}

void RdmaTransportManager::InitializeDeviceAddress(mf_sockaddr& deviceAddr)
{
    if (deviceIp_.type == IpV4) {
        deviceAddr.ip.ipv4.sin_family = AF_INET;
        deviceAddr.ip.ipv4.sin_addr = deviceIp_.ip.ipv4;
        deviceAddr.ip.ipv4.sin_port = devicePort_;
        deviceAddr.type = IpV4;
    } else if (deviceIp_.type == IpV6) {
        deviceAddr.ip.ipv6.sin6_family = AF_INET6;
        deviceAddr.ip.ipv6.sin6_addr = deviceIp_.ip.ipv6;
        deviceAddr.ip.ipv6.sin6_port = devicePort_;
        deviceAddr.type = IpV6;
    }
}

Result RdmaTransportManager::OpenDevice(const TransportOptions& options)
{
    int32_t userId = -1;
    int32_t logicId = -1;

    SHM_LOG_DEBUG(rankId_ << " begin to open device with " << options);
    auto ret = DlAclApi::AclrtGetDevice(&userId);
    SHM_ASSERT_LOG_AND_RETURN(
        ret == 0 && userId >= 0, "AclrtGetDevice() return=" << ret << ", output deviceId=" << userId,
        ACLSHMEM_DL_FUNC_FAILED);

    ret = DlAclApi::RtGetLogicDevIdByUserDevId(userId, &logicId);
    SHM_ASSERT_LOG_AND_RETURN(
        ret == 0 && logicId >= 0, "RtGetLogicDevIdByUserDevId() return=" << ret << ", output deviceId=" << logicId,
        ACLSHMEM_DL_FUNC_FAILED);

    int32_t phyId = -1;
    // HCCP/topo use global phyId; pass userId to deprecated API (MR !407), not logicId.
    ret = DlAclApi::AclrtGetPhyDevIdByLogicDevId(userId, &phyId);
    SHM_ASSERT_LOG_AND_RETURN(
        ret == 0 && phyId >= 0,
        "AclrtGetPhyDevIdByLogicDevId() return=" << ret << ", userId=" << userId << ", logicDeviceId=" << logicId
                                                 << ", output phyId=" << phyId,
        ACLSHMEM_DL_FUNC_FAILED);

    deviceId_ = static_cast<uint32_t>(logicId);
    phyId_ = static_cast<uint32_t>(phyId);
    SHM_LOG_INFO(
        rankId_ << " resolved device mapping: userId=" << userId << ", logicDeviceId=" << deviceId_
                << ", phyId=" << phyId_);
    rankId_ = options.rankId;
    rankCount_ = options.rankCount;
    role_ = options.role;
    ret = ParseDeviceNic(options.nic, devicePort_);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " parse input nic(" << options.nic << ") failed!");
        return ACLSHMEM_INVALID_PARAM;
    }

    if (options.type == IpV4) {
        deviceIp_.type = IpV4;
    } else if (options.type == IpV6) {
        deviceIp_.type = IpV6;
    }

    if (!PrepareOpenDevice(userId, phyId_, rankCount_, deviceIp_, rdmaHandle_)) {
        SHM_LOG_ERROR("phyId=" << phyId_ << " PrepareOpenDevice failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    nicInfo_ = GenerateDeviceNic(deviceIp_, devicePort_);

    mf_sockaddr deviceAddr;
    InitializeDeviceAddress(deviceAddr);
    if (role_ == HYBM_ROLE_PEER) {
        qpManager_ = std::make_shared<FixedRanksQpManager>(userId, phyId_, rankId_, rankCount_, deviceAddr);
    } else {
        qpManager_ = std::make_shared<DynamicRanksQpManager>(
            userId, phyId_, rankId_, rankCount_, deviceAddr, role_ == HYBM_ROLE_RECEIVER);
    }

    deviceChipInfo_ = std::make_shared<DeviceChipInfo>(userId);
    ret = deviceChipInfo_->Init();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " device info init failed: " << ret);
        return ret;
    }
    SHM_LOG_INFO(rankId_ << " open device with " << options << " success.");
    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManager::CloseDevice()
{
    if (qpManager_ != nullptr) {
        qpManager_->Shutdown();
        qpManager_ = nullptr;
    }
    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManager::RegisterMemoryRegion(const TransportMemoryRegion& mr)
{
    void* mrHandle = nullptr;
    HccpMrInfo info{};
    info.addr = (void*)(ptrdiff_t)mr.addr;
    info.size = mr.size;
    info.access = mr.access;
    auto ret = DlHccpApi::RaRegisterMR(rdmaHandle_, &info, mrHandle);
    if (ret != 0) {
        SHM_LOG_ERROR(rankId_ << " register MR=" << mr << " failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    RegMemResult result{mr.addr, mr.size, mrHandle, info.lkey, info.rkey};
    SHM_LOG_DEBUG(rankId_ << " register MR result=" << result);

    registerMRS_.emplace(mr.addr, result);
    ret = qpManager_->SetLocalMemories(registerMRS_);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " qp manager set mr failed: " << ret);
        return ret;
    }

    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManager::UnregisterMemoryRegion(uint64_t addr)
{
    auto pos = registerMRS_.find(addr);
    if (pos == registerMRS_.end()) {
        SHM_LOG_ERROR(rankId_ << " input address not register!");
        return ACLSHMEM_INVALID_PARAM;
    }

    auto ret = DlHccpApi::RaDeregisterMR(rdmaHandle_, pos->second.mrHandle);
    if (ret != 0) {
        SHM_LOG_ERROR(rankId_ << " Unregister MR addr failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    registerMRS_.erase(pos);
    ret = qpManager_->SetLocalMemories(registerMRS_);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " qp manager set mr failed: " << ret);
        return ret;
    }
    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManager::QueryMemoryKey(uint64_t addr, TransportMemoryKey& key)
{
    RegMemKeyUnion keyUnion{};
    auto pos = registerMRS_.lower_bound(addr);
    if (pos == registerMRS_.end() || pos->first + pos->second.size <= addr) {
        SHM_LOG_ERROR(rankId_ << " input address not register");
        return ACLSHMEM_INVALID_PARAM;
    }

    keyUnion.deviceKey = pos->second;

    key = keyUnion.commonKey;
    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManager::ParseMemoryKey(const TransportMemoryKey& key, uint64_t& addr, uint64_t& size)
{
    RegMemKeyUnion keyUnion{};
    keyUnion.commonKey = key;
    if (keyUnion.deviceKey.type != TT_HCCP) {
        SHM_LOG_ERROR(rankId_ << " parse memory key type invalid: " << keyUnion.deviceKey.type);
        return ACLSHMEM_INNER_ERROR;
    }

    addr = keyUnion.deviceKey.address;
    size = keyUnion.deviceKey.size;
    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManager::Prepare(const HybmTransPrepareOptions& options)
{
    SHM_LOG_DEBUG(rankId_ << " RdmaTransportManager Prepare with : " << options);
    int ret;
    if ((ret = CheckPrepareOptions(options)) != 0) {
        return ret;
    }

    mf_sockaddr deviceNetwork;
    std::unordered_map<uint32_t, ConnectRankInfo> rankInfo;
    for (auto it = options.options.begin(); it != options.options.end(); ++it) {
        ret = ParseDeviceNic(it->second.nic, deviceNetwork);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR(rankId_ << " parse networks[" << it->first << "]=" << it->second.nic << " failed: " << ret);
            return ACLSHMEM_INVALID_PARAM;
        }

        rankInfo.emplace(it->first, ConnectRankInfo{it->second.role, deviceNetwork, it->second.memKeys});
    }
    SHM_LOG_DEBUG(rankId_ << " SetRemoteRankInfo rankInfo.size=" << rankInfo.size());

    ret = qpManager_->SetRemoteRankInfo(rankInfo);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " qp manager set remote rank info failed: " << ret);
        return ret;
    }

    ret = qpManager_->Startup(rdmaHandle_);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " qp manager startup failed: " << ret);
        return ret;
    }

    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManager::Connect()
{
    auto ret = AsyncConnect();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " AsyncConnect() failed: " << ret);
        return ret;
    }

    ret = WaitForConnected(-1L);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " WaitForConnected(-1) failed: " << ret);
        return ret;
    }

    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManager::AsyncConnect() { return ACLSHMEM_SUCCESS; }

Result RdmaTransportManager::WaitForConnected(int64_t timeoutNs)
{
    if (qpManager_ == nullptr) {
        SHM_LOG_ERROR(rankId_ << " server side not listen!");
        return ACLSHMEM_INNER_ERROR;
    }

    auto ret = qpManager_->WaitingConnectionReady();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " wait for server side connected on device failed: " << ret);
        return ret;
    }

    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManager::UpdateRankOptions(const HybmTransPrepareOptions& options)
{
    SHM_LOG_DEBUG(rankId_ << " RdmaTransportManager Prepare with : " << options);
    if (qpManager_ == nullptr) {
        SHM_LOG_ERROR(rankId_ << " qp manager not created");
        return ACLSHMEM_INNER_ERROR;
    }

    mf_sockaddr deviceNetwork;
    std::unordered_map<uint32_t, ConnectRankInfo> ranksInfo;
    for (auto it = options.options.begin(); it != options.options.end(); ++it) {
        auto ret = ParseDeviceNic(it->second.nic, deviceNetwork);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR(rankId_ << " update rank network(" << it->second.nic << ") invalid.");
            return ACLSHMEM_INVALID_PARAM;
        }
        SHM_LOG_INFO(rankId_ << " UpdateRankOptions update rank: " << it->first);
        ranksInfo.emplace(it->first, ConnectRankInfo{it->second.role, deviceNetwork, it->second.memKeys});
    }
    SHM_LOG_DEBUG(rankId_ << " UpdateRankOptions ranksInfo.size=" << ranksInfo.size());

    auto ret = qpManager_->SetRemoteRankInfo(ranksInfo);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " update rank options failed: " << ret);
        return ret;
    }

    return ACLSHMEM_SUCCESS;
}

const std::string& RdmaTransportManager::GetNic() const { return nicInfo_; }

const void* RdmaTransportManager::GetQpInfo() const
{
    if (qpManager_ == nullptr) {
        SHM_LOG_ERROR(rankId_ << " GetQpInfo(): connection manager not created.");
        return nullptr;
    }
    return qpManager_->GetQpInfoAddress();
}

bool RdmaTransportManager::PrepareOpenDevice(
    uint32_t userId, uint32_t phyId, uint32_t rankCount, net_addr_t& deviceIp, void*& rdmaHandle)
{
    // If can get rdmaHandle, maybe the device has been opened, can try get rdmaHandle directly.
    bool referenced = false;
    if (TryGetRdmaHandleAndReferenceManagedRa(phyId, rdmaHandle, referenced)) {
        if (rdmaHandle != nullptr) {
            if (!RetireDeviceIp(phyId, deviceIp)) {
                SHM_LOG_ERROR("phyId=" << phyId << " RetireDeviceIp failed.");
                if (referenced) {
                    RaDeinit(phyId);
                }
                return false;
            }
            raReferenced_ = referenced;
            SHM_LOG_DEBUG("phyId=" << phyId << " Had prepared device and get rdmaHandle success.");
            return true;
        }
        SHM_LOG_INFO("phyId=" << phyId << " Had prepared device, but rdmaHandle is null, need init again.");
    }
    if (!OpenTsd(userId, rankCount)) {
        SHM_LOG_ERROR("phyId=" << phyId << " open tsd failed.");
        return false;
    }

    if (!RaInit(phyId)) {
        SHM_LOG_ERROR("phyId=" << phyId << " RaInit failed.");
        return false;
    }
    raReferenced_ = true;

    if (!RetireDeviceIp(phyId, deviceIp)) {
        SHM_LOG_ERROR("phyId=" << phyId << " RetireDeviceIp failed.");
        return false;
    }

    if (!RaRdevInit(phyId, deviceIp, rdmaHandle)) {
        SHM_LOG_ERROR("phyId=" << phyId << " RaRdevInit failed.");
        return false;
    }
    return true;
}

bool RdmaTransportManager::TryGetRdmaHandleAndReferenceManagedRa(uint32_t phyId, void*& rdmaHandle, bool& referenced)
{
    referenced = false;
    std::lock_guard<std::mutex> lock(raMutex_);
    if (DlHccpApi::RaRdevGetHandle(phyId, rdmaHandle) != 0) {
        return false;
    }

    auto it = raInstances_.find(phyId);
    if (rdmaHandle != nullptr && it != raInstances_.end() && it->second.refCount != 0) {
        ++it->second.refCount;
        it->second.rdmaHandle = rdmaHandle;
        referenced = true;
        SHM_LOG_INFO("phyId=" << phyId << " reference managed ra, ref=" << it->second.refCount << ".");
    }
    return true;
}

bool RdmaTransportManager::OpenTsd(uint32_t userId, uint32_t rankCount)
{
    std::lock_guard<std::mutex> lock(raMutex_);
    if (tsdOpened_) {
        SHM_LOG_INFO("userId=" << userId << " tsd already opened.");
        return true;
    }

    auto res = DlHccpApi::TsdOpen(userId, rankCount);
    if (res != 0) {
        SHM_LOG_ERROR("TsdOpen for (userId=" << userId << ", rankCount=" << rankCount << ") failed: " << res);
        return false;
    }

    SHM_LOG_DEBUG("open tsd for user id: " << userId << ", rank count: " << rankCount << " success.");
    tsdOpened_ = true;
    return true;
}

bool RdmaTransportManager::RaInit(uint32_t phyId)
{
    std::lock_guard<std::mutex> lock(raMutex_);
    auto& state = raInstances_[phyId];
    if (state.refCount != 0) {
        ++state.refCount;
        SHM_LOG_INFO("phyId=" << phyId << " ra already initialized, ref=" << state.refCount << ".");
        return true;
    }

    const std::chrono::seconds WAIT_TIME(3);
    HccpRaInitConfig initConfig{};
    initConfig.phyId = phyId;
    initConfig.nicPosition = NETWORK_OFFLINE;
    initConfig.hdcType = 6; // HDC_SERVICE_TYPE_RDMA = 6
    SHM_LOG_DEBUG("phyId=" << phyId << " RaInit=" << initConfig);
    std::this_thread::sleep_for(WAIT_TIME); // avoid hccl init conflict
    auto ret = DlHccpApi::RaInit(initConfig);
    if (ret != 0) {
        SHM_LOG_WARN("phyId=" << phyId << " Hccp Init RA failed: " << ret);
        // maybe hccl have already initialized ra, wait 3s then return true.
        std::this_thread::sleep_for(WAIT_TIME);
        state.refCount = 1;
        state.selfOwned = false;
        return true;
    }

    SHM_LOG_DEBUG("phyId=" << phyId << " ra init success.");
    state.refCount = 1;
    state.selfOwned = true;
    return true;
}

void RdmaTransportManager::RaDeinit(uint32_t phyId)
{
    std::lock_guard<std::mutex> lock(raMutex_);
    auto it = raInstances_.find(phyId);
    if (it == raInstances_.end() || it->second.refCount == 0) {
        SHM_LOG_WARN("phyId=" << phyId << " not ra init.");
        return;
    }

    --it->second.refCount;
    SHM_LOG_INFO("phyId=" << phyId << " release ra, ref=" << it->second.refCount << ".");
    if (it->second.refCount != 0) {
        return;
    }

    if (!it->second.selfOwned) {
        SHM_LOG_INFO("phyId=" << phyId << " skip RaDeinit in shmem.");
        raInstances_.erase(it);
        if (raInstances_.empty()) {
            tsdOpened_ = false;
        }
        return;
    }

    HccpRaInitConfig deinitConfig{};
    deinitConfig.phyId = phyId;
    deinitConfig.nicPosition = NETWORK_OFFLINE;
    deinitConfig.hdcType = 6; // HDC_SERVICE_TYPE_RDMA = 6
    auto ret = DlHccpApi::RaDeinit(deinitConfig);
    if (ret != 0) {
        SHM_LOG_WARN("phyId=" << phyId << " Hccp Deinit RA failed: " << ret);
    } else {
        SHM_LOG_DEBUG("phyId=" << phyId << " ra deinit success.");
    }

    raInstances_.erase(it);
    if (raInstances_.empty()) {
        tsdOpened_ = false;
    }
}

void RdmaTransportManager::ResetOpenDeviceState(uint32_t phyId)
{
    std::lock_guard<std::mutex> lock(raMutex_);
    auto it = raInstances_.find(phyId);
    if (it != raInstances_.end() && it->second.refCount == 0) {
        raInstances_.erase(it);
    }
    if (raInstances_.empty()) {
        tsdOpened_ = false;
    }
}

bool RdmaTransportManager::HandleRetiredDeviceIp(uint32_t phyId, net_addr_t& deviceIp)
{
    std::lock_guard<std::mutex> lock(raMutex_);
    auto it = raInstances_.find(phyId);
    if (it == raInstances_.end() || !it->second.deviceIpRetired || it->second.retiredIp.type != deviceIp.type) {
        return false;
    }

    deviceIp = it->second.retiredIp;
    if (deviceIp.type == IpV4) {
        SHM_LOG_INFO("phyId=" << phyId << " device ip already retired : " << inet_ntoa(deviceIp.ip.ipv4));
        return true;
    } else if (deviceIp.type == IpV6) {
        char ipv6Str[INET6_ADDRSTRLEN];
        inet_ntop(AF_INET6, &deviceIp.ip.ipv6, ipv6Str, INET6_ADDRSTRLEN);
        SHM_LOG_INFO("phyId=" << phyId << " device ip already retired : " << ipv6Str);
        return true;
    }
    return false;
}

bool RdmaTransportManager::RetireDeviceIp(uint32_t phyId, net_addr_t& deviceIp)
{
    auto isRetire = HandleRetiredDeviceIp(phyId, deviceIp);
    if (isRetire) {
        return true;
    }

    uint32_t count = 0;
    std::vector<HccpInterfaceInfo> infos;

    HccpRaGetIfAttr config;
    config.phyId = phyId;
    config.nicPosition = NETWORK_OFFLINE;
    config.isAll = false;

    auto ret = DlHccpApi::RaGetIfNum(config, count);
    if (ret != 0 || count == 0) {
        SHM_LOG_ERROR("phyId=" << phyId << " get interface count failed: " << ret << ", count: " << count);
        return false;
    }

    infos.resize(count);
    ret = DlHccpApi::RaGetIfAddrs(config, infos.data(), count);
    if (ret != 0) {
        SHM_LOG_ERROR("phyId=" << phyId << " get interface information failed: " << ret);
        return false;
    }

    for (auto& info : infos) {
        SHM_LOG_DEBUG(
            "phyId=" << phyId << " found interface: ifname=" << info.ifname << ", scopeId=" << info.scopeId
                     << ", family=" << info.family);
        if (info.family == AF_INET && deviceIp.type == IpV4) {
            deviceIp.ip.ipv4 = info.ifaddr.ip.addr;
            deviceIp.type = IpV4;
            {
                std::lock_guard<std::mutex> lock(raMutex_);
                auto& state = raInstances_[phyId];
                state.retiredIp = deviceIp;
                state.deviceIpRetired = true;
            }
            SHM_LOG_DEBUG("phyId=" << phyId << " retire device ip success : " << inet_ntoa(deviceIp.ip.ipv4));
            return true;
        }
        if (info.family == AF_INET6 && deviceIp.type == IpV6) {
            deviceIp.ip.ipv6 = info.ifaddr.ip.addr6;
            deviceIp.type = IpV6;
            {
                std::lock_guard<std::mutex> lock(raMutex_);
                auto& state = raInstances_[phyId];
                state.retiredIp = deviceIp;
                state.deviceIpRetired = true;
            }
            char ipv6Str[INET6_ADDRSTRLEN];
            inet_ntop(AF_INET6, &deviceIp.ip.ipv6, ipv6Str, INET6_ADDRSTRLEN);
            SHM_LOG_DEBUG("phyId=" << phyId << " retire device ip success : " << ipv6Str);
            return true;
        }
    }

    SHM_LOG_ERROR("phyId=" << phyId << " not found requested network device type " << deviceIp.type << " on NPU.");
    return false;
}

bool RdmaTransportManager::RaRdevInit(uint32_t phyId, net_addr_t deviceIp, void*& rdmaHandle)
{
    std::lock_guard<std::mutex> lock(raMutex_);
    auto& state = raInstances_[phyId];
    if (state.rdmaHandle != nullptr) {
        SHM_LOG_INFO("phyId=" << phyId << " ra rdev already initialized.");
        rdmaHandle = state.rdmaHandle;
        return true;
    }

    HccpRdevInitInfo info{};
    HccpRdev rdev{};

    info.mode = NETWORK_OFFLINE;
    info.notifyType = NOTIFY;
    info.enabled2mbLite = true;
    rdev.phyId = phyId;
    rdev.family = (deviceIp.type == IpV4) ? AF_INET : AF_INET6;
    if (deviceIp.type == IpV4) {
        rdev.localIp.addr = deviceIp.ip.ipv4;
    } else if (deviceIp.type == IpV6) {
        rdev.localIp.addr6 = deviceIp.ip.ipv6;
    }
    SHM_LOG_DEBUG("phyId=" << phyId << " RaRdevInitV2, info=" << info << "rdev=" << rdev);
    auto ret = DlHccpApi::RaRdevInitV2(info, rdev, rdmaHandle);
    if (ret != 0) {
        SHM_LOG_ERROR("phyId=" << phyId << " Hccp Init RDev failed: " << ret);
        return false;
    }

    state.rdmaHandle = rdmaHandle;
    SHM_LOG_INFO("phyId=" << phyId << " initialize RDev success.");
    return true;
}

void RdmaTransportManager::ClearAllRegisterMRs()
{
    for (auto it = registerMRS_.begin(); it != registerMRS_.end(); ++it) {
        auto ret = DlHccpApi::RaDeregisterMR(rdmaHandle_, it->second.mrHandle);
        if (ret != 0) {
            SHM_LOG_WARN(
                rankId_ << " Unregister:" << (void*)(ptrdiff_t)it->first << " : " << it->second << " failed: " << ret);
        }
    }
    registerMRS_.clear();
}

int RdmaTransportManager::CheckPrepareOptions(const shm::transport::HybmTransPrepareOptions& options)
{
    if (role_ != HYBM_ROLE_PEER) {
        SHM_LOG_INFO(rankId_ << " transport role: " << role_ << " check options passed.");
        return ACLSHMEM_SUCCESS;
    }

    if (options.options.size() > rankCount_) {
        SHM_LOG_ERROR(
            rankId_ << " options size():" << options.options.size() << " larger than rank count: " << rankCount_);
        return ACLSHMEM_INVALID_PARAM;
    }

    if (options.options.find(rankId_) == options.options.end()) {
        SHM_LOG_ERROR(rankId_ << " options not contains self rankId: " << rankId_);
        return ACLSHMEM_INVALID_PARAM;
    }

    for (auto it = options.options.begin(); it != options.options.end(); ++it) {
        if (it->first >= rankCount_) {
            SHM_LOG_ERROR(
                rankId_ << " input options of nics contains rankId:" << it->first << ", rank count: " << rankCount_);
            return ACLSHMEM_INVALID_PARAM;
        }
    }

    return ACLSHMEM_SUCCESS;
}
} // namespace device
} // namespace transport
} // namespace shm
