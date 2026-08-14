/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "securec.h"
#include "dl_acl_api.h"
#include "dl_hcomm_api.h"
#include "dl_hcomm_def.h"
#include "device_udma_def.h"
#include "hcomm_entity_compat.h"
#include "hybm_mem_segment.h"
#include "shmemi_host_common.h"
#include "../host_device/shmemi_host_device_constant.h"
#include "device_udma_transport_manager.h"

namespace shm {
namespace transport {
namespace device {

namespace {

constexpr int32_t INVALID_EID_INDEX = -1;
constexpr const char ACLSHMEM_HCOMM_MEM_TAG[] = "aclshmem_udma";
constexpr uint32_t ACLSHMEM_HCOMM_SINGLE_CHANNEL_NUM = 1;
constexpr uint32_t ACLSHMEM_HCOMM_DEFAULT_QOS = 4;
constexpr uint32_t CHANNEL_STATUS_POLL_INTERVAL_MS = 10;
constexpr uint32_t CHANNEL_STATUS_POLL_TIMEOUT_MS = 120000;

template <typename T>
bool CheckedMultiply(T lhs, T rhs, T& result)
{
    static_assert(std::numeric_limits<T>::is_integer && !std::numeric_limits<T>::is_signed);
    if (lhs != 0 && rhs > std::numeric_limits<T>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

template <typename T>
bool CheckedAdd(T lhs, T rhs, T& result)
{
    static_assert(std::numeric_limits<T>::is_integer && !std::numeric_limits<T>::is_signed);
    if (rhs > std::numeric_limits<T>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

std::string MakeUdmaChannelName(uint32_t local_rank, uint32_t peer, uint32_t qp_idx)
{
    const uint32_t low_rank = std::min(local_rank, peer);
    const uint32_t high_rank = std::max(local_rank, peer);
    return "aclshmem_udma_" + std::to_string(low_rank) + "_" + std::to_string(high_rank) + "_qp_" +
           std::to_string(qp_idx);
}

Result WaitHcommChannelReady(const std::vector<ChannelHandle>& channels)
{
    if (channels.empty()) {
        return ACLSHMEM_SUCCESS;
    }

    const uint32_t channel_num = static_cast<uint32_t>(channels.size());
    std::vector<int32_t> statuses(channel_num, HCOMM_CHANNEL_STATUS_CONNECTING);
    uint32_t elapsed_ms = 0;
    while (elapsed_ms <= CHANNEL_STATUS_POLL_TIMEOUT_MS) {
        auto hcomm_ret = DlHcommApi::HcommChannelGetStatus(channels.data(), channel_num, statuses.data());
        if (hcomm_ret != 0) {
            SHM_LOG_ERROR("HcommChannelGetStatus failed, ret = " << hcomm_ret << ", channelNum = " << channel_num);
            return ACLSHMEM_INNER_ERROR;
        }

        bool all_ready = true;
        for (uint32_t idx = 0; idx < channel_num; ++idx) {
            const int32_t status = statuses[idx];
            if (status == HCOMM_CHANNEL_STATUS_READY) {
                continue;
            }

            all_ready = false;
            if (status == HCOMM_CHANNEL_STATUS_CONNECTING) {
                continue;
            }

            if (status == HCOMM_CHANNEL_STATUS_FAILED) {
                SHM_LOG_ERROR("Hcomm UDMA channel connect failed, index = " << idx << ", status = " << status);
            } else if (status == HCOMM_CHANNEL_STATUS_TIMEOUT) {
                SHM_LOG_ERROR("Hcomm UDMA channel connect timeout, index = " << idx << ", status = " << status);
            } else {
                SHM_LOG_ERROR("Hcomm UDMA channel unknown status, index = " << idx << ", status = " << status);
            }
            return ACLSHMEM_INNER_ERROR;
        }
        if (all_ready) {
            SHM_LOG_DEBUG(
                "All hcomm UDMA channels connected, channelNum = " << channel_num << ", elapsedMs = " << elapsed_ms);
            return ACLSHMEM_SUCCESS;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(CHANNEL_STATUS_POLL_INTERVAL_MS));
        elapsed_ms += CHANNEL_STATUS_POLL_INTERVAL_MS;
    }

    SHM_LOG_ERROR(
        "Wait hcomm UDMA channel ready timeout, channelNum = "
        << channel_num << ", pollIntervalMs = " << CHANNEL_STATUS_POLL_INTERVAL_MS
        << ", pollTimeoutMs = " << CHANNEL_STATUS_POLL_TIMEOUT_MS << ", elapsedMs = " << elapsed_ms);
    return ACLSHMEM_TIMEOUT_ERROR;
}

Result AllocateAndClearDeviceBuffer(void*& ptr, uint64_t size, const char* name)
{
    if (ptr != nullptr) {
        return ACLSHMEM_SUCCESS;
    }
    auto ret = DlAclApi::AclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACLSHMEM_SUCCESS || ptr == nullptr) {
        SHM_LOG_ERROR("Allocate " << name << " failed, ret = " << ret << ", size = " << size);
        ptr = nullptr;
        return ACLSHMEM_INNER_ERROR;
    }
    ret = DlAclApi::AclrtMemset(ptr, size, 0, size);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Clear " << name << " failed, ret = " << ret << ", size = " << size);
        (void)DlAclApi::AclrtFree(ptr);
        ptr = nullptr;
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

Result CopyHcommChannelEntityFromDevice(void* dst, uint64_t size, uint64_t devAddr, const char* name, uint32_t peer)
{
    if (dst == nullptr || size == 0 || devAddr == 0) {
        SHM_LOG_ERROR(
            "Invalid Hcomm " << name << " input for peer " << peer << ", size = " << size << ", devAddr = " << devAddr);
        return ACLSHMEM_INNER_ERROR;
    }

    auto ret = DlAclApi::AclrtMemcpy(
        dst, size, reinterpret_cast<const void*>(static_cast<uintptr_t>(devAddr)), size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Copy Hcomm " << name << " from device failed for peer " << peer << ", ret = " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

// Holds the per-peer HCOMM channel contexts read back from device memory. These
// are translated into the legacy udmaInfo layout (WQCtx / CqCtx / UBmemInfo) that
// the unmodified data plane consumes.
struct PeerChannelContexts {
    SqContext sq_context{};
    CqContext cq_context{};
    RegedBufferEntity remote_buffer{};
};

Result ReadPeerChannelContexts(uint64_t channel_ptr, uint32_t peer, bool read_remote_buffer, PeerChannelContexts& out)
{
    ChannelEntity channel{};
    auto ret = CopyHcommChannelEntityFromDevice(&channel, sizeof(channel), channel_ptr, "channel entity", peer);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }

    if (channel.sqNum == 0 || channel.cqNum == 0 || channel.sqContextAddr == nullptr ||
        channel.cqContextAddr == nullptr ||
        (read_remote_buffer && (channel.remoteBufferNum == 0 || channel.remoteBufferAddr == nullptr))) {
        SHM_LOG_ERROR(
            "Invalid Hcomm channel entity for peer " << peer << ", sqNum = " << channel.sqNum
                                                     << ", cqNum = " << channel.cqNum
                                                     << ", remoteBufferNum = " << channel.remoteBufferNum);
        return ACLSHMEM_INNER_ERROR;
    }

    uint64_t sqContextAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(channel.sqContextAddr));
    uint64_t cqContextAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(channel.cqContextAddr));
    ret = CopyHcommChannelEntityFromDevice(&out.sq_context, sizeof(out.sq_context), sqContextAddr, "SQ context", peer);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }
    if (out.sq_context.contextInfo.ubJfs.sqVa == 0 ||
        out.sq_context.contextInfo.ubJfs.sqDepth != shm::UDMA_SQ_BASKBLK_CNT) {
        SHM_LOG_ERROR(
            "Invalid Hcomm SQ context for peer " << peer << ", sqVa = " << out.sq_context.contextInfo.ubJfs.sqVa
                                                 << ", sqDepth = " << out.sq_context.contextInfo.ubJfs.sqDepth
                                                 << ", expectedDepth = " << shm::UDMA_SQ_BASKBLK_CNT);
        return ACLSHMEM_INNER_ERROR;
    }

    ret = CopyHcommChannelEntityFromDevice(&out.cq_context, sizeof(out.cq_context), cqContextAddr, "CQ context", peer);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }
    if (out.cq_context.contextInfo.ubJfc.scqVa == 0 ||
        out.cq_context.contextInfo.ubJfc.cqDepth != shm::UDMA_CQ_DEPTH_DEFAULT) {
        SHM_LOG_ERROR(
            "Invalid Hcomm CQ context for peer " << peer << ", scqVa = " << out.cq_context.contextInfo.ubJfc.scqVa
                                                 << ", cqDepth = " << out.cq_context.contextInfo.ubJfc.cqDepth
                                                 << ", expectedDepth = " << shm::UDMA_CQ_DEPTH_DEFAULT);
        return ACLSHMEM_INNER_ERROR;
    }

    if (read_remote_buffer) {
        uint64_t remoteBufferAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(channel.remoteBufferAddr));
        ret = CopyHcommChannelEntityFromDevice(
            &out.remote_buffer, sizeof(out.remote_buffer), remoteBufferAddr, "remote buffer", peer);
        if (ret != ACLSHMEM_SUCCESS) {
            return ret;
        }
    }
    return ACLSHMEM_SUCCESS;
}

// Active queue/memory table for this build. The union members share storage, so this resolves to
// the same bytes as the other member; the constexpr just documents which path owns them. Templated
// on InfoT (const or non-const aclshmemi_aiv_udma_info_t) so the returned reference's constness
// follows the argument -- one body serves both the fill (write) and print (read-only) call sites.
template <typename InfoT>
auto& ActiveUdmaTable(InfoT& info)
{
    if constexpr (ACLSHMEM_UDMA_RELAY_ENABLED) {
        return info.relay;
    } else {
        return info.direct;
    }
}

void SetUdmaInfoSectionPtrs(aclshmemi_aiv_udma_info_t& info, void* infoBase, uint64_t slot_count, uint32_t qp_num)
{
    aclshmemi_udma_qp_table_t& tbl = ActiveUdmaTable(info);
    tbl.sq_ptr = reinterpret_cast<uint64_t>(static_cast<aclshmemi_aiv_udma_info_t*>(infoBase) + 1);
    tbl.rq_ptr = reinterpret_cast<uint64_t>(
        reinterpret_cast<aclshmemi_udma_wq_ctx_t*>(tbl.sq_ptr) + static_cast<uint64_t>(slot_count) * qp_num);
    tbl.scq_ptr = reinterpret_cast<uint64_t>(
        reinterpret_cast<aclshmemi_udma_wq_ctx_t*>(tbl.rq_ptr) + static_cast<uint64_t>(slot_count) * qp_num);
    tbl.rcq_ptr = reinterpret_cast<uint64_t>(
        reinterpret_cast<aclshmemi_udma_cq_ctx_t*>(tbl.scq_ptr) + static_cast<uint64_t>(slot_count) * qp_num);
    tbl.mem_ptr = reinterpret_cast<uint64_t>(
        reinterpret_cast<aclshmemi_udma_cq_ctx_t*>(tbl.rcq_ptr) + static_cast<uint64_t>(slot_count) * qp_num);
}

} // namespace

UdmaTransportManager::UdmaTransportManager() noexcept {}

UdmaTransportManager::~UdmaTransportManager() noexcept { CleanupResources(); }

Result UdmaTransportManager::OpenDevice(const TransportOptions& options)
{
    const uint32_t qp_num = options.udmaQpConfig.qpNum;
    if (qp_num == 0 || qp_num > ACLSHMEM_MAX_QP_NUM) {
        SHM_LOG_ERROR("Invalid UDMA QP count: " << qp_num);
        return ACLSHMEM_INVALID_VALUE;
    }
    if constexpr (ACLSHMEM_UDMA_RELAY_ENABLED) {
        if (qp_num != 1) {
            SHM_LOG_ERROR("UDMA multi-QP is not supported when relay mode is enabled, qpNum = " << qp_num);
            return ACLSHMEM_NOT_SUPPORTED;
        }
    }

    int32_t user_id = -1;
    int32_t logic_id = -1;

    auto ret = DlAclApi::AclrtGetDevice(&user_id);
    SHM_ASSERT_LOG_AND_RETURN(
        ret == 0 && user_id >= 0, "AclrtGetDevice() return=" << ret << ", output device_id=" << user_id,
        ACLSHMEM_DL_FUNC_FAILED);

    ret = DlAclApi::RtGetLogicDevIdByUserDevId(user_id, &logic_id);
    SHM_ASSERT_LOG_AND_RETURN(
        ret == 0 && logic_id >= 0, "RtGetLogicDevIdByUserDevId() return=" << ret << ", output device_id=" << logic_id,
        ACLSHMEM_DL_FUNC_FAILED);

    const uint32_t device_id = static_cast<uint32_t>(logic_id);
    user_id_ = static_cast<uint32_t>(user_id);
    rank_id_ = options.rankId;
    rank_count_ = options.rankCount;
    role_ = options.role;
    qp_num_ = qp_num;

    if (!PrepareOpenDevice(device_id, rank_count_)) {
        SHM_LOG_ERROR("PrepareOpenDevice failed.");
        return ACLSHMEM_INNER_ERROR;
    }

    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::CloseDevice()
{
    CleanupResources();
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::RegisterMemoryRegion(const TransportMemoryRegion& mr)
{
    auto addrRecordIt = mem_record_map_.find(mr.addr);
    if (addrRecordIt != mem_record_map_.end()) {
        SHM_LOG_INFO("Hcomm MR address has been registered, address = " << mr.addr);
        return ACLSHMEM_SUCCESS;
    }
    if (connected_) {
        SHM_LOG_ERROR("Register Hcomm MR after connected is not supported, address = " << mr.addr);
        return ACLSHMEM_INVALID_PARAM;
    }

    bool is_hbm = (mr.flags & REG_MR_FLAG_HBM) != 0;
    bool is_dram = (mr.flags & REG_MR_FLAG_DRAM) != 0;
    if (is_hbm == is_dram) {
        SHM_LOG_ERROR("Invalid memory region flags for Hcomm MR, flags = " << mr.flags);
        return ACLSHMEM_INVALID_PARAM;
    }
    CommMemType mem_type = is_hbm ? CommMemType::DEVICE : CommMemType::HOST;

    std::map<uint32_t, HcommMemHandle> hcomm_handles;
    auto rollback_registered_handles = [this, &hcomm_handles]() {
        for (const auto& registered : hcomm_handles) {
            auto registered_endpoint_it = endpoint_handle_map_.find(registered.first);
            if (registered_endpoint_it == endpoint_handle_map_.end() || registered_endpoint_it->second == nullptr ||
                registered.second == nullptr) {
                continue;
            }
            auto hcomm_ret = DlHcommApi::HcommMemUnreg(registered_endpoint_it->second, registered.second);
            if (hcomm_ret != 0) {
                SHM_LOG_WARN(
                    "Rollback Hcomm memory registration failed for EID index " << registered.first
                                                                               << ", ret = " << hcomm_ret);
            }
        }
    };
    for (const auto& endpoint_entry : endpoint_handle_map_) {
        const uint32_t eid_index = endpoint_entry.first;
        EndpointHandle endpoint_handle = endpoint_entry.second;
        if (endpoint_handle == nullptr) {
            SHM_LOG_ERROR("Invalid hcomm endpoint for EID index " << eid_index);
            rollback_registered_handles();
            return ACLSHMEM_INNER_ERROR;
        }

        CommMem mem{};
        mem.type = mem_type;
        mem.addr = reinterpret_cast<void*>(mr.addr);
        mem.size = mr.size;

        HcommMemHandle hcomm_mem_handle = nullptr;
        auto hcomm_ret = DlHcommApi::HcommMemReg(endpoint_handle, ACLSHMEM_HCOMM_MEM_TAG, &mem, &hcomm_mem_handle);
        if (hcomm_ret != 0 || hcomm_mem_handle == nullptr) {
            SHM_LOG_ERROR("Failed to register hcomm memory for EID index " << eid_index << ", ret = " << hcomm_ret);
            rollback_registered_handles();
            return ACLSHMEM_INNER_ERROR;
        }
        hcomm_handles[eid_index] = hcomm_mem_handle;
    }

    mem_record_map_[mr.addr] = hcomm_handles;

    SHM_LOG_INFO("Register MR success.");
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::UnregisterMemoryRegion(uint64_t addr)
{
    auto hcomm_pos = mem_record_map_.find(addr);
    if (hcomm_pos == mem_record_map_.end()) {
        SHM_LOG_ERROR("Input address is not registered, address = " << addr);
        return ACLSHMEM_INVALID_PARAM;
    }

    auto& hcomm_handles = hcomm_pos->second;
    if (!channel_handles_.empty() || udma_info_dev_ != nullptr) {
        DestroyChannels();
        FreeDeviceInfo();
        connected_ = false;
    }
    Result result = ACLSHMEM_SUCCESS;
    for (const auto& hcomm_mem_entry : hcomm_handles) {
        auto endpoint_it = endpoint_handle_map_.find(hcomm_mem_entry.first);
        if (endpoint_it == endpoint_handle_map_.end() || endpoint_it->second == nullptr ||
            hcomm_mem_entry.second == nullptr) {
            continue;
        }
        auto hcomm_ret = DlHcommApi::HcommMemUnreg(endpoint_it->second, hcomm_mem_entry.second);
        if (hcomm_ret != 0) {
            SHM_LOG_WARN(
                "Failed to unregister hcomm memory for EID index " << hcomm_mem_entry.first << ", ret = " << hcomm_ret);
            result = ACLSHMEM_INNER_ERROR;
        }
    }
    mem_record_map_.erase(hcomm_pos);
    return result;
}

Result UdmaTransportManager::Prepare(const HybmTransPrepareOptions& options)
{
    SHM_LOG_DEBUG("UdmaTransportManager Prepare with options: " << options);
    return CheckPrepareOptions(options);
}

Result UdmaTransportManager::Connect()
{
    if (connected_) {
        return ACLSHMEM_SUCCESS;
    }
    auto ret = AsyncConnect();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Async connect failed, ret = " << ret);
        return ret;
    }
    SHM_LOG_INFO("Async connect success");
    return ACLSHMEM_SUCCESS;
}

const void* UdmaTransportManager::GetQpInfo() const { return udma_info_dev_; }

TransportDeviceInfo UdmaTransportManager::GetDeviceInfo() const
{
    TransportDeviceInfo info;
    info.udmaInfoAddress = reinterpret_cast<uint64_t>(udma_info_dev_);
    return info;
}

uint64_t UdmaTransportManager::SlotCount() const
{
    if constexpr (ACLSHMEM_UDMA_RELAY_ENABLED) {
        // Each (actual_pe, relay_pe) pair gets one slot. Compute in uint64_t so the N*N product
        // (and the byte-size offsets derived from it) cannot silently wrap in uint32_t (TOP-03:
        // memory sizes/offsets must use a 64-bit type).
        return static_cast<uint64_t>(rank_count_) * rank_count_;
    } else {
        // Direct path: one slot per target pe.
        return static_cast<uint64_t>(rank_count_);
    }
}

Result UdmaTransportManager::BuildUdmaInfo(
    const std::vector<uint64_t>& channel_ptrs, const std::vector<uint32_t>& channel_slots,
    const std::vector<uint32_t>& channel_dst_pes, const std::vector<uint32_t>& channel_qp_indices)
{
    if (channel_ptrs.empty() || channel_slots.size() != channel_ptrs.size() ||
        channel_dst_pes.size() != channel_ptrs.size() || channel_qp_indices.size() != channel_ptrs.size()) {
        SHM_LOG_ERROR(
            "Invalid hcomm channel ptr input, channel_ptrs = "
            << channel_ptrs.size() << ", channel_slots = " << channel_slots.size() << ", channel_dst_pes = "
            << channel_dst_pes.size() << ", channel_qp_indices = " << channel_qp_indices.size());
        return ACLSHMEM_INVALID_PARAM;
    }

    const uint64_t slot_count = SlotCount();
    if (slot_count > std::numeric_limits<size_t>::max()) {
        SHM_LOG_ERROR("UDMA slot count exceeds size_t, slotCount = " << slot_count);
        return ACLSHMEM_INVALID_VALUE;
    }
    const size_t slot_count_size = static_cast<size_t>(slot_count);
    size_t queue_entry_count = 0;
    if (!CheckedMultiply(slot_count_size, static_cast<size_t>(qp_num_), queue_entry_count)) {
        SHM_LOG_ERROR("UDMA queue entry count overflow, slotCount = " << slot_count << ", qpNum = " << qp_num_);
        return ACLSHMEM_INVALID_VALUE;
    }
    std::vector<SqContext> sq_contexts_by_qp(queue_entry_count);
    std::vector<CqContext> cq_contexts_by_qp(queue_entry_count);
    std::vector<RegedBufferEntity> remote_buffers_by_slot(slot_count_size);
    std::vector<bool> slot_valid(slot_count_size, false);
    auto ret = ReadChannelContexts(
        channel_ptrs, channel_slots, channel_dst_pes, channel_qp_indices, sq_contexts_by_qp, cq_contexts_by_qp,
        remote_buffers_by_slot, slot_valid);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }

    std::vector<uint8_t> eid_table_host;
    ret = PrepareUdmaInfoBuffers(eid_table_host);
    if (ret != ACLSHMEM_SUCCESS) {
        FreeDeviceInfo();
        return ret;
    }

    std::vector<uint8_t> udma_info_buffer;
    aclshmemi_aiv_udma_info_t* copy_info = nullptr;
    ret = InitHostUdmaInfo(udma_info_buffer, copy_info);
    if (ret != ACLSHMEM_SUCCESS) {
        FreeDeviceInfo();
        return ret;
    }
    ret = FillHostUdmaInfo(
        sq_contexts_by_qp, cq_contexts_by_qp, remote_buffers_by_slot, slot_valid, eid_table_host, *copy_info);
    if (ret != ACLSHMEM_SUCCESS) {
        FreeDeviceInfo();
        return ret;
    }

    ret = CopyEidTableToDevice(eid_table_host);
    if (ret != ACLSHMEM_SUCCESS) {
        FreeDeviceInfo();
        return ret;
    }

    // Print the filled host-side udmaInfo (section pointers still reference the local
    // buffer here) for maintenance/debugging before rebasing to device addresses.
    PrintHostUdmaInfo(*copy_info);

#if defined(ACLSHMEM_MSSANITIZER_BUILD)
    // Report the URMA-allocated SQ/CQ rings to mssanitizer while the section pointers still
    // reference the host buffer (CopyUdmaInfoToDevice rebases them to device addresses next).
    RegisterUdmaQpSanitizerRegions(*copy_info);
#endif

    ret = CopyUdmaInfoToDevice(udma_info_buffer, *copy_info);
    if (ret != ACLSHMEM_SUCCESS) {
        FreeDeviceInfo();
        return ret;
    }

    SHM_LOG_INFO(
        "Build UDMA info success, rank_count = " << rank_count_ << ", totalChannelNum = " << channel_handles_.size()
                                                 << ", qp_num = " << qp_num_);
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::ReadChannelContexts(
    const std::vector<uint64_t>& channel_ptrs, const std::vector<uint32_t>& channel_slots,
    const std::vector<uint32_t>& channel_dst_pes, const std::vector<uint32_t>& channel_qp_indices,
    std::vector<SqContext>& sq_contexts_by_qp, std::vector<CqContext>& cq_contexts_by_qp,
    std::vector<RegedBufferEntity>& remote_buffers_by_slot, std::vector<bool>& slot_valid) const
{
    const uint64_t slot_count = SlotCount();
    for (size_t idx = 0; idx < channel_ptrs.size(); ++idx) {
        const uint32_t slot = channel_slots[idx];
        const uint32_t dst_pe = channel_dst_pes[idx];
        const uint32_t qp_idx = channel_qp_indices[idx];
        if (slot >= slot_count || dst_pe >= rank_count_ || dst_pe == rank_id_ || qp_idx >= qp_num_ ||
            channel_ptrs[idx] == 0) {
            SHM_LOG_ERROR(
                "Invalid hcomm channel entity for slot " << slot << " qp_idx " << qp_idx << " dst_pe " << dst_pe
                                                         << ", index = " << idx);
            return ACLSHMEM_INVALID_PARAM;
        }

        const bool read_remote_buffer = !slot_valid[slot];
        PeerChannelContexts peer_contexts{};
        auto read_ret = ReadPeerChannelContexts(channel_ptrs[idx], dst_pe, read_remote_buffer, peer_contexts);
        if (read_ret != ACLSHMEM_SUCCESS) {
            return read_ret;
        }

        const uint64_t queue_offset = static_cast<uint64_t>(slot) * qp_num_ + qp_idx;
        sq_contexts_by_qp[queue_offset] = peer_contexts.sq_context;
        cq_contexts_by_qp[queue_offset] = peer_contexts.cq_context;

        if (read_remote_buffer) {
            remote_buffers_by_slot[slot] = peer_contexts.remote_buffer;
            slot_valid[slot] = true;
        }
    }
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::PrepareUdmaInfoBuffers(std::vector<uint8_t>& eid_table_host)
{
    // Reserve the per-peer AMO scratch buffers the same way the legacy
    // DeviceJettyManager did: one malloc per peer, referenced from WQCtx.
    auto ret = ReserveScratchBuffers();
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }

    // Allocate the device-side remote EID table: uint8_t[slot_count][URMA_EID_RAW_SIZE].
    // Direct build: slot_count == rank_count_, indexed by pe (unchanged). Relay build: one entry
    // per (actual_pe, relay_pe) slot, since each slot's remote target EID differs.
    const uint64_t eid_table_size = static_cast<uint64_t>(SlotCount()) * URMA_EID_RAW_SIZE;
    ret = AllocateAndClearDeviceBuffer(eid_dev_, eid_table_size, "udma remote eid table");
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }
    eid_table_host.assign(eid_table_size, 0);
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::InitHostUdmaInfo(
    std::vector<uint8_t>& udma_info_buffer, aclshmemi_aiv_udma_info_t*& copy_info)
{
    // Build the contiguous udmaInfo blob in host memory using the legacy layout:
    //   [info][WQCtx * slot_count * qp_num][WQCtx(rq) * slot_count * qp_num]
    //   [CqCtx(scq) * slot_count * qp_num][CqCtx(rcq) * slot_count * qp_num][UBmemInfo * slot_count]
    const uint64_t slot_count = SlotCount();
    if (slot_count > std::numeric_limits<size_t>::max()) {
        SHM_LOG_ERROR("UDMA slot count exceeds size_t, slotCount = " << slot_count);
        return ACLSHMEM_INVALID_VALUE;
    }

    const size_t slot_count_size = static_cast<size_t>(slot_count);
    constexpr size_t bytes_per_queue = 2U * sizeof(aclshmemi_udma_wq_ctx_t) + 2U * sizeof(aclshmemi_udma_cq_ctx_t);
    constexpr size_t header_bytes = sizeof(aclshmemi_aiv_udma_info_t);
    size_t queue_entry_count = 0;
    size_t queue_bytes = 0;
    size_t mem_bytes = 0;
    size_t info_and_queue_bytes = 0;
    size_t total_bytes = 0;
    if (!CheckedMultiply(slot_count_size, static_cast<size_t>(qp_num_), queue_entry_count) ||
        !CheckedMultiply(queue_entry_count, bytes_per_queue, queue_bytes) ||
        !CheckedMultiply(slot_count_size, sizeof(aclshmemi_ubmem_info_t), mem_bytes) ||
        !CheckedAdd(header_bytes, queue_bytes, info_and_queue_bytes) ||
        !CheckedAdd(info_and_queue_bytes, mem_bytes, total_bytes)) {
        SHM_LOG_ERROR("UDMA info size overflow, slotCount = " << slot_count << ", qpNum = " << qp_num_);
        return ACLSHMEM_INVALID_VALUE;
    }
    udma_info_size_ = static_cast<uint64_t>(total_bytes);

    udma_info_buffer.assign(total_bytes, 0);
    copy_info = reinterpret_cast<aclshmemi_aiv_udma_info_t*>(udma_info_buffer.data());
    copy_info->qp_num = qp_num_;
    // Host-side section pointers used only while filling the local buffer.
    SetUdmaInfoSectionPtrs(*copy_info, copy_info, slot_count, qp_num_);
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::FillHostUdmaInfo(
    const std::vector<SqContext>& sq_contexts_by_qp, const std::vector<CqContext>& cq_contexts_by_qp,
    const std::vector<RegedBufferEntity>& remote_buffers_by_slot, const std::vector<bool>& slot_valid,
    std::vector<uint8_t>& eid_table_host, aclshmemi_aiv_udma_info_t& copy_info)
{
    const aclshmemi_udma_qp_table_t& tbl = ActiveUdmaTable(copy_info);
    auto* wq_array = reinterpret_cast<aclshmemi_udma_wq_ctx_t*>(tbl.sq_ptr);
    auto* rq_array = reinterpret_cast<aclshmemi_udma_wq_ctx_t*>(tbl.rq_ptr);
    auto* scq_array = reinterpret_cast<aclshmemi_udma_cq_ctx_t*>(tbl.scq_ptr);
    auto* rcq_array = reinterpret_cast<aclshmemi_udma_cq_ctx_t*>(tbl.rcq_ptr);
    auto* mem_array = reinterpret_cast<aclshmemi_ubmem_info_t*>(tbl.mem_ptr);

    const uint64_t slot_count = SlotCount();
    for (uint64_t slot = 0; slot < slot_count; ++slot) {
        if (!slot_valid[slot]) {
            // Self entry, unconnected peer, and skipped relay-diagonal slots stay zero-initialized;
            // the data plane never issues a self-send and asserts against self pe.
            continue;
        }
        // The amo scratch buffer is indexed by the actual destination pe. Direct build: slot == pe.
        // Relay build: slot == actual_pe * rank_count_ + relay_pe, so recover actual_pe from slot.
        const uint32_t dst_pe =
            ACLSHMEM_UDMA_RELAY_ENABLED ? static_cast<uint32_t>(slot / rank_count_) : static_cast<uint32_t>(slot);
        // Guard index range and null amo buffer before FillWqCtx dereferences amo_dev_list_[dst_pe]
        // (self entry is null; RED-02 / IMPL-16). Inconsistent slot mapping fails the build.
        if (dst_pe >= rank_count_ || dst_pe == rank_id_ || amo_dev_list_[dst_pe] == nullptr) {
            SHM_LOG_ERROR(
                "Invalid udma slot mapping: slot "
                << slot << " -> dst_pe " << dst_pe << " (rank_count = " << rank_count_ << ", rank_id = " << rank_id_
                << ", amo buffer " << (dst_pe < rank_count_ && amo_dev_list_[dst_pe] != nullptr ? "set" : "null")
                << ")");
            return ACLSHMEM_INNER_ERROR;
        }
        for (uint32_t qp_idx = 0; qp_idx < qp_num_; ++qp_idx) {
            const uint64_t queue_offset = slot * qp_num_ + qp_idx;
            FillWqCtx(sq_contexts_by_qp[queue_offset], dst_pe, wq_array[queue_offset]);
            rq_array[queue_offset] = wq_array[queue_offset];
            FillCqCtx(cq_contexts_by_qp[queue_offset], scq_array[queue_offset]);
            rcq_array[queue_offset] = scq_array[queue_offset];
        }
        const uint64_t first_qp_offset = slot * qp_num_;
        FillMemInfo(sq_contexts_by_qp[first_qp_offset], remote_buffers_by_slot[slot], mem_array[slot]);

        // Stage the remote EID raw bytes (from the SQ context) into the device EID table
        // and point mem_array[slot].eid_addr at the device-resident copy, matching the
        // legacy behavior where the data plane reads rmtEid[0..1] from eid_addr.
        uint8_t* eid_slot = eid_table_host.data() + static_cast<uint64_t>(slot) * URMA_EID_RAW_SIZE;
        int copy_ret = memcpy_s(
            eid_slot, URMA_EID_RAW_SIZE, sq_contexts_by_qp[first_qp_offset].contextInfo.ubJfs.remoteEID,
            URMA_EID_RAW_SIZE);
        if (copy_ret != EOK) {
            SHM_LOG_ERROR("Copy Hcomm UDMA remote EID failed, slot = " << slot << ", ret = " << copy_ret);
            return ACLSHMEM_INNER_ERROR;
        }
        mem_array[slot].eid_addr = reinterpret_cast<uint64_t>(
            static_cast<uint8_t*>(eid_dev_) + static_cast<uint64_t>(slot) * URMA_EID_RAW_SIZE);
    }
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::CopyEidTableToDevice(const std::vector<uint8_t>& eid_table_host)
{
    auto ret = DlAclApi::AclrtMemcpy(
        eid_dev_, eid_table_host.size(), eid_table_host.data(), eid_table_host.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Copy udma remote eid table to device failed, ret = " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::CopyUdmaInfoToDevice(
    std::vector<uint8_t>& udma_info_buffer, aclshmemi_aiv_udma_info_t& copy_info)
{
    // Allocate the device blob and rebase the section pointers to device addresses.
    auto alloc_ret = AllocateAndClearDeviceBuffer(udma_info_dev_, udma_info_size_, "udma info");
    if (alloc_ret != ACLSHMEM_SUCCESS) {
        return alloc_ret;
    }
    SetUdmaInfoSectionPtrs(copy_info, udma_info_dev_, SlotCount(), qp_num_);

    auto ret = DlAclApi::AclrtMemcpy(
        udma_info_dev_, udma_info_size_, udma_info_buffer.data(), udma_info_size_, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Copy udma info to device failed, ret = " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::ReserveScratchBuffers()
{
    amo_dev_list_.assign(rank_count_, nullptr);
    for (uint32_t peer = 0; peer < rank_count_; ++peer) {
        if (peer == rank_id_) {
            continue;
        }
        auto ret = AllocateAndClearDeviceBuffer(amo_dev_list_[peer], sizeof(uint64_t), "udma amo scratch");
        if (ret != ACLSHMEM_SUCCESS) {
            return ret;
        }
    }
    return ACLSHMEM_SUCCESS;
}

void UdmaTransportManager::FillWqCtx(
    const SqContext& sq_context, uint32_t dst_pe, aclshmemi_udma_wq_ctx_t& dst_wq) const
{
    const auto& ubJfs = sq_context.contextInfo.ubJfs;
    dst_wq.wqn = ubJfs.jfsID;
    dst_wq.buf_addr = ubJfs.sqVa;
    // HCOMM exposes the raw WQE byte size; the data plane consumes it directly.
    dst_wq.wqe_size = ubJfs.wqeSize;
    dst_wq.depth = shm::UDMA_SQ_BASKBLK_CNT;
    dst_wq.head = 0;
    dst_wq.tail = 0;
    dst_wq.db_mode = aclshmemi_udma_db_mode_t::SW_DB;
    dst_wq.db_addr = ubJfs.dbVa;
    dst_wq.sl = 0;
    dst_wq.wqe_cnt = 0;
    // amo scratch is per actual destination pe (the data plane reads it from the direct slot).
    dst_wq.amo_addr = reinterpret_cast<uint64_t>(amo_dev_list_[dst_pe]);
}

void UdmaTransportManager::FillCqCtx(const CqContext& cq_context, aclshmemi_udma_cq_ctx_t& dst_cq) const
{
    const auto& ubJfc = cq_context.contextInfo.ubJfc;
    dst_cq.cqn = ubJfc.jfcID;
    dst_cq.buf_addr = ubJfc.scqVa;
    dst_cq.cqe_size = ubJfc.cqeSize;
    dst_cq.depth = shm::UDMA_CQ_DEPTH_DEFAULT;
    dst_cq.head = 0;
    dst_cq.tail = 0;
    dst_cq.db_mode = aclshmemi_udma_db_mode_t::SW_DB;
    dst_cq.db_addr = ubJfc.dbVa;
}

void UdmaTransportManager::FillMemInfo(
    const SqContext& sq_context, const RegedBufferEntity& remote_buffer, aclshmemi_ubmem_info_t& dst_mem) const
{
    const auto& ub = remote_buffer.bufferInfo.rma.protectionInfo.memInfo.ub;
    dst_mem.token_value_valid = true; // token-based access control enabled (data plane sets tokenEn = 1)
    dst_mem.rmt_jetty_type = 1;       // remote jetty type: 1 = jetty (peer-to-peer)
    dst_mem.target_hint = 0;          // no target selection preference
    dst_mem.tpn = sq_context.contextInfo.ubJfs.tpID;
    dst_mem.tid = ub.tokenId;
    dst_mem.rmt_token_value = ub.tokenValue;
    dst_mem.len = static_cast<uint32_t>(remote_buffer.bufferInfo.rma.size);
    dst_mem.addr = remote_buffer.bufferInfo.rma.addr;
    dst_mem.eid_addr = 0; // filled by the caller after the device EID table is staged
}

void UdmaTransportManager::PrintHostUdmaInfo(const aclshmemi_aiv_udma_info_t& host_info) const
{
    SHM_LOG_DEBUG("=======================rank [" << rank_id_ << "] udma host info====================");
    SHM_LOG_DEBUG("rank[" << rank_id_ << "] udmaInfo.qp_num: " << host_info.qp_num);

    const aclshmemi_udma_qp_table_t& tbl = ActiveUdmaTable(host_info);
    const auto* wq_array = reinterpret_cast<const aclshmemi_udma_wq_ctx_t*>(tbl.sq_ptr);
    const auto* cq_array = reinterpret_cast<const aclshmemi_udma_cq_ctx_t*>(tbl.scq_ptr);
    const auto* mem_array = reinterpret_cast<const aclshmemi_ubmem_info_t*>(tbl.mem_ptr);
    if (wq_array == nullptr || cq_array == nullptr || mem_array == nullptr) {
        SHM_LOG_WARN("rank[" << rank_id_ << "] udma host info section pointer is null, skip printing.");
        return;
    }

    // Direct build: slot_count == rank_count_ (index is the target pe). Relay build:
    // slot_count == rank_count_^2 (index is actual_pe * rank_count_ + relay_pe).
    const uint64_t slot_count = SlotCount();
    for (uint64_t slot = 0; slot < slot_count; ++slot) {
        for (uint32_t qp_idx = 0; qp_idx < host_info.qp_num; ++qp_idx) {
            const uint64_t queue_offset = slot * host_info.qp_num + qp_idx;
            const auto& wq = wq_array[queue_offset];
            const auto& cq = cq_array[queue_offset];
            SHM_LOG_DEBUG("rank[" << rank_id_ << "] slot[" << slot << "] qp[" << qp_idx << "] WQCtx.wqn: " << wq.wqn);
            SHM_LOG_DEBUG(
                "rank[" << rank_id_ << "] slot[" << slot << "] qp[" << qp_idx << "] WQCtx.buf_addr: " << wq.buf_addr
                        << ", wqe_size: " << wq.wqe_size << ", depth: " << wq.depth << ", head: " << wq.head
                        << ", tail: " << wq.tail << ", db_mode: " << static_cast<int>(wq.db_mode)
                        << ", db_addr: " << wq.db_addr << ", sl: " << wq.sl << ", wqe_cnt: " << wq.wqe_cnt
                        << ", amo_addr: " << wq.amo_addr);
            SHM_LOG_DEBUG(
                "rank[" << rank_id_ << "] slot[" << slot << "] qp[" << qp_idx << "] CQCtx.cqn: " << cq.cqn
                        << ", buf_addr: " << cq.buf_addr << ", cqe_size: " << cq.cqe_size << ", depth: " << cq.depth
                        << ", head: " << cq.head << ", tail: " << cq.tail
                        << ", db_mode: " << static_cast<int>(cq.db_mode) << ", db_addr: " << cq.db_addr);
        }

        const auto& mem = mem_array[slot];
        SHM_LOG_DEBUG(
            "rank[" << rank_id_ << "] slot[" << slot << "] MemInfo.token_value_valid: " << mem.token_value_valid);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] slot[" << slot << "] MemInfo.rmt_jetty_type: " << mem.rmt_jetty_type);
        SHM_LOG_DEBUG(
            "rank[" << rank_id_ << "] slot[" << slot << "] MemInfo.target_hint: " << static_cast<int>(mem.target_hint));
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] slot[" << slot << "] MemInfo.tpn: " << mem.tpn);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] slot[" << slot << "] MemInfo.tid: " << mem.tid);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] slot[" << slot << "] MemInfo.rmt_token_value: " << mem.rmt_token_value);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] slot[" << slot << "] MemInfo.len: " << mem.len);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] slot[" << slot << "] MemInfo.addr: " << mem.addr);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] slot[" << slot << "] MemInfo.eid_addr: " << mem.eid_addr);
    }
}

std::vector<HcommMemHandle> UdmaTransportManager::CollectChannelMemHandles(uint32_t eid_index) const
{
    std::vector<HcommMemHandle> mem_handles;
    for (const auto& addrEntry : mem_record_map_) {
        const auto& hcomm_handles = addrEntry.second;
        auto handleIt = hcomm_handles.find(eid_index);
        if (handleIt != hcomm_handles.end() && handleIt->second != nullptr) {
            mem_handles.push_back(handleIt->second);
        }
    }
    return mem_handles;
}

Result UdmaTransportManager::ResolveRelaySlotRoute(
    uint32_t actual_pe, uint32_t relay_pe, bool& skip, uint32_t& local_eid, uint32_t& remote_eid) const
{
    skip = false;
    local_eid = 0;
    remote_eid = 0;

    // Diagonal (actual == relay) where actual is a real peer means "peer relays to itself" --
    // physically meaningless. The slot is left unused; callers skip it.
    if (actual_pe == relay_pe && actual_pe != rank_id_) {
        skip = true;
        return ACLSHMEM_SUCCESS;
    }

    // Source EID bucket (my local egress port). For the (actual, relay == me) direct slot we must
    // egress straight toward actual_pe (matching the OFF build); for a relay slot we egress toward
    // relay_pe.
    if (relay_pe == rank_id_) {
        auto local_route_it = peer_eid_index_map_.find(actual_pe);
        if (local_route_it == peer_eid_index_map_.end()) {
            SHM_LOG_ERROR("Missing local route for direct peer rank " << actual_pe);
            return ACLSHMEM_INNER_ERROR;
        }
        local_eid = local_route_it->second;
    } else {
        auto local_route_it = peer_eid_index_map_.find(relay_pe);
        if (local_route_it == peer_eid_index_map_.end()) {
            SHM_LOG_ERROR("Missing local route for relay rank " << relay_pe);
            return ACLSHMEM_INNER_ERROR;
        }
        local_eid = local_route_it->second;
    }

    // Target EID: actual_pe's port toward relay_pe; the fabric forwards by this EID via relay.
    if (relay_pe == rank_id_) {
        // (actual, relay == me) is the direct path: actual_pe's port toward me.
        auto remote_route_it = peer_remote_eid_index_map_.find(actual_pe);
        if (remote_route_it == peer_remote_eid_index_map_.end()) {
            SHM_LOG_ERROR("Missing remote route for peer rank " << actual_pe);
            return ACLSHMEM_INNER_ERROR;
        }
        remote_eid = remote_route_it->second;
    } else {
        // Look up actual_pe's local EID toward relay_pe in the global routing matrix.
        if (all_local_routes_.size() != static_cast<size_t>(rank_count_) * rank_count_) {
            SHM_LOG_ERROR(
                "all_local_routes_ size " << all_local_routes_.size() << " != rankCount^2 "
                                          << rank_count_ * rank_count_);
            return ACLSHMEM_INNER_ERROR;
        }
        int32_t r = all_local_routes_[actual_pe * rank_count_ + relay_pe];
        if (r < 0) {
            SHM_LOG_ERROR("Invalid global route for (actual=" << actual_pe << ", relay=" << relay_pe << "): " << r);
            return ACLSHMEM_INNER_ERROR;
        }
        remote_eid = static_cast<uint32_t>(r);
    }
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::ExchangeEndpointDescriptors(EndpointExchange& exchange) const
{
    const uint32_t local_endpoint_count = static_cast<uint32_t>(endpoint_desc_map_.size());
    exchange.counts.assign(rank_count_, 0);
    g_boot_handle.allgather(&local_endpoint_count, exchange.counts.data(), sizeof(uint32_t), &g_boot_handle);

    const auto max_it = std::max_element(exchange.counts.begin(), exchange.counts.end());
    exchange.max_count = (max_it == exchange.counts.end()) ? 0 : *max_it;
    if (exchange.max_count == 0) {
        SHM_LOG_ERROR("No local hcomm endpoint descriptor was exchanged.");
        return ACLSHMEM_INNER_ERROR;
    }

    // Pack this rank's descriptors into the first slots; the rest stay zero (valid == 0).
    std::vector<ExchangedEndpointDesc> local_endpoints(exchange.max_count);
    uint32_t packed_index = 0;
    for (const auto& endpoint_entry : endpoint_desc_map_) {
        ExchangedEndpointDesc& packed = local_endpoints[packed_index];
        packed.eid_index = endpoint_entry.first;
        packed.valid = 1;
        packed.desc = endpoint_entry.second;
        ++packed_index;
    }

    exchange.descs.assign(static_cast<size_t>(rank_count_) * exchange.max_count, ExchangedEndpointDesc{});
    g_boot_handle.allgather(
        local_endpoints.data(), exchange.descs.data(),
        static_cast<uint64_t>(sizeof(ExchangedEndpointDesc) * exchange.max_count), &g_boot_handle);
    return ACLSHMEM_SUCCESS;
}

const UdmaTransportManager::ExchangedEndpointDesc* UdmaTransportManager::FindRemoteEndpointDesc(
    const EndpointExchange& exchange, uint32_t dst_pe, uint32_t remote_eid_index) const
{
    const uint32_t count = exchange.counts[dst_pe];
    for (uint32_t idx = 0; idx < count; ++idx) {
        const ExchangedEndpointDesc& candidate = exchange.descs[dst_pe * exchange.max_count + idx];
        if (candidate.valid != 0 && candidate.eid_index == remote_eid_index) {
            return &candidate;
        }
    }
    return nullptr;
}

Result UdmaTransportManager::CreateChannelsForSlot(
    const EndpointExchange& exchange, uint32_t local_eid_index, uint32_t remote_eid_index, uint32_t dst_pe,
    uint32_t slot, uint32_t channel_num, ChannelBuildState& state)
{
    auto endpoint_it = endpoint_handle_map_.find(local_eid_index);
    if (endpoint_it == endpoint_handle_map_.end() || endpoint_it->second == nullptr) {
        SHM_LOG_ERROR(
            "Failed to find hcomm endpoint for dst_pe " << dst_pe << ", local_eid_index = " << local_eid_index);
        return ACLSHMEM_INNER_ERROR;
    }

    const ExchangedEndpointDesc* remote_endpoint_info = FindRemoteEndpointDesc(exchange, dst_pe, remote_eid_index);
    if (remote_endpoint_info == nullptr) {
        SHM_LOG_ERROR(
            "No remote hcomm endpoint descriptor was exchanged for dst_pe " << dst_pe << " on remote EID index "
                                                                            << remote_eid_index);
        return ACLSHMEM_INNER_ERROR;
    }

    std::vector<HcommMemHandle> mem_handles = CollectChannelMemHandles(local_eid_index);
    if (mem_handles.empty()) {
        SHM_LOG_ERROR(
            "No active Hcomm mem handle for local EID index " << local_eid_index << " when creating channel for dst_pe "
                                                              << dst_pe);
        return ACLSHMEM_INNER_ERROR;
    }

    std::vector<HcommChannelDesc> channel_descs(channel_num);
    auto desc_init_ret = ShmemHcommChannelDescInit(channel_descs.data(), channel_num);
    if (desc_init_ret != 0) {
        SHM_LOG_ERROR(
            "HcommChannelDescInit failed for dst_pe " << dst_pe << ", channelNum = " << channel_num
                                                      << ", ret = " << desc_init_ret);
        return ACLSHMEM_INNER_ERROR;
    }

    const bool is_direct_multi_qp = !ACLSHMEM_UDMA_RELAY_ENABLED && channel_num > 1;
    const bool use_named_channel = is_direct_multi_qp;
    const size_t channel_names_start = state.channel_names.size();
    std::vector<const char*> channel_names(channel_num, nullptr);
    for (uint32_t qp_idx = 0; qp_idx < channel_num; ++qp_idx) {
        HcommChannelDesc& channel_desc = channel_descs[qp_idx];
        channel_desc.remoteEndpoint = remote_endpoint_info->desc;
        channel_desc.notifyNum = 0;
        channel_desc.exchangeAllMems = false;
        channel_desc.memHandles = mem_handles.data();
        channel_desc.memHandleNum = static_cast<uint32_t>(mem_handles.size());
        if (use_named_channel) {
            state.channel_names.push_back(MakeUdmaChannelName(rank_id_, dst_pe, qp_idx));
            channel_names[qp_idx] = state.channel_names.back().c_str();
            channel_desc.channelName = channel_names[qp_idx];
        }
#if defined(HCOMM_CHANNEL_DESC_ABI_V1_SIZE)
        // CANN HCOM ABI v2 moves the communication-domain QoS out of the
        // protocol-specific union.
        channel_desc.qos = ACLSHMEM_HCOMM_DEFAULT_QOS;
#elif defined(SHMEM_HCOMM_ENTITY_FROM_CANN) || defined(SHMEM_HCOMM_RES_FROM_CANN)
        channel_desc.hccsAttr.qos = ACLSHMEM_HCOMM_DEFAULT_QOS;
#else
        channel_desc.hccsAttr.qos = ACLSHMEM_HCOMM_DEFAULT_QOS;
#endif
        SHM_LOG_INFO(
            "Prepare HCOMM channel descriptor: rank="
            << rank_id_ << ", peer=" << dst_pe << ", slot=" << slot << ", qp=" << qp_idx
            << ", channelName=" << (channel_names[qp_idx] == nullptr ? "<anonymous>" : channel_names[qp_idx]));
    }

#if defined(HCOMM_CHANNEL_DESC_ABI_V1_SIZE)
    SHM_LOG_INFO(
        "HCOMM channel descriptor batch extended ABI, header={version="
        << channel_descs[0].header.version << ", magic=" << channel_descs[0].header.magicWord
        << ", size=" << channel_descs[0].header.size << "}, qos=" << channel_descs[0].qos
        << ", memHandleNum=" << channel_descs[0].memHandleNum << ", channelNum=" << channel_num);
#elif defined(SHMEM_HCOMM_ENTITY_FROM_CANN) || defined(SHMEM_HCOMM_RES_FROM_CANN)
    SHM_LOG_INFO(
        "HCOMM channel descriptor batch ABI v1, header={version="
        << channel_descs[0].header.version << ", magic=" << channel_descs[0].header.magicWord
        << ", size=" << channel_descs[0].header.size << "}, qos=" << channel_descs[0].hccsAttr.qos
        << ", memHandleNum=" << channel_descs[0].memHandleNum << ", channelNum=" << channel_num);
#else
    SHM_LOG_INFO(
        "HCOMM channel descriptor batch fallback ABI, qos=" << channel_descs[0].hccsAttr.qos
                                                            << ", memHandleNum=" << channel_descs[0].memHandleNum
                                                            << ", channelNum=" << channel_num);
#endif

    std::vector<ChannelHandle> channel_handles(channel_num, 0);
    auto hcomm_ret = DlHcommApi::HcommChannelCreate(
        endpoint_it->second, COMM_ENGINE_AIV, channel_descs.data(), channel_num, channel_handles.data());
    const bool has_invalid_handle =
        std::any_of(channel_handles.begin(), channel_handles.end(), [](ChannelHandle handle) { return handle == 0; });
    if (hcomm_ret != 0 || has_invalid_handle) {
        SHM_LOG_ERROR(
            "HcommChannelCreate batch failed: rank=" << rank_id_ << ", peer=" << dst_pe << ", slot=" << slot
                                                     << ", channelNum=" << channel_num << ", ret=" << hcomm_ret
                                                     << ", hasInvalidHandle=" << has_invalid_handle);
        for (uint32_t qp_idx = 0; qp_idx < channel_num; ++qp_idx) {
            SHM_LOG_ERROR(
                "HcommChannelCreate batch output: rank=" << rank_id_ << ", peer=" << dst_pe << ", slot=" << slot
                                                         << ", qp=" << qp_idx
                                                         << ", channelHandle=" << channel_handles[qp_idx]);
        }
        std::vector<ChannelHandle> created_handles;
        created_handles.reserve(channel_num);
        for (ChannelHandle handle : channel_handles) {
            if (handle != 0) {
                created_handles.push_back(handle);
            }
        }
        if (!created_handles.empty()) {
            const auto destroy_ret =
                DlHcommApi::HcommChannelDestroy(created_handles.data(), static_cast<uint32_t>(created_handles.size()));
            if (destroy_ret != 0) {
                SHM_LOG_WARN(
                    "HcommChannelDestroy failed while rolling back batch: rank="
                    << rank_id_ << ", peer=" << dst_pe << ", channelNum=" << created_handles.size()
                    << ", ret=" << destroy_ret);
            }
        }
        state.channel_names.resize(channel_names_start);
        return ACLSHMEM_INNER_ERROR;
    }

    for (uint32_t qp_idx = 0; qp_idx < channel_num; ++qp_idx) {
        state.handles.push_back(channel_handles[qp_idx]);
        state.slots.push_back(slot);
        state.dst_pes.push_back(dst_pe);
        state.qp_indices.push_back(qp_idx);
        SHM_LOG_INFO(
            "Create HCOMM channel success: rank="
            << rank_id_ << ", peer=" << dst_pe << ", slot=" << slot << ", qp=" << qp_idx
            << ", channelHandle=" << channel_handles[qp_idx]
            << ", channelName=" << (channel_names[qp_idx] == nullptr ? "<anonymous>" : channel_names[qp_idx]));
    }
    SHM_LOG_INFO(
        "Create HCOMM channel batch success: rank=" << rank_id_ << ", peer=" << dst_pe << ", slot=" << slot
                                                    << ", channelNum=" << channel_num);
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::BuildRelayChannels(const EndpointExchange& exchange, ChannelBuildState& state)
{
    // One channel per meaningful (actual_pe, relay_pe) slot. HcommChannelCreate is non-blocking
    // (it only starts the async connect; readiness is polled later via WaitHcommChannelReady), so
    // channels can be created in a single straight pass -- no per-relay_pe wave synchronization is
    // needed to avoid the old blocking-create cross-card deadlock.
    for (uint32_t relay_pe = 0; relay_pe < rank_count_; ++relay_pe) {
        for (uint32_t actual_pe = 0; actual_pe < rank_count_; ++actual_pe) {
            if (actual_pe == rank_id_) {
                continue; // self target is local; no channel
            }
            bool skip = false;
            uint32_t local_eid_index = 0;
            uint32_t remote_eid_index = 0;
            if (ResolveRelaySlotRoute(actual_pe, relay_pe, skip, local_eid_index, remote_eid_index) !=
                ACLSHMEM_SUCCESS) {
                return ACLSHMEM_INNER_ERROR;
            }
            if (skip) {
                continue; // diagonal (actual == relay) slot is physically meaningless
            }
            const uint32_t slot = actual_pe * rank_count_ + relay_pe;
            if (CreateChannelsForSlot(
                    exchange, local_eid_index, remote_eid_index, actual_pe, slot, ACLSHMEM_HCOMM_SINGLE_CHANNEL_NUM,
                    state) != ACLSHMEM_SUCCESS) {
                return ACLSHMEM_INNER_ERROR;
            }
        }
    }
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::BuildDirectChannels(const EndpointExchange& exchange, ChannelBuildState& state)
{
    for (uint32_t peer = 0; peer < rank_count_; ++peer) {
        if (peer == rank_id_) {
            continue;
        }
        auto local_eid_it = peer_eid_index_map_.find(peer);
        auto remote_eid_it = peer_remote_eid_index_map_.find(peer);
        if (local_eid_it == peer_eid_index_map_.end() || remote_eid_it == peer_remote_eid_index_map_.end()) {
            SHM_LOG_ERROR("Failed to find EID route for peer " << peer);
            return ACLSHMEM_INNER_ERROR;
        }
        // Direct path: slot == peer. Create all QP channels for this peer in one HCOMM batch.
        const Result create_ret =
            CreateChannelsForSlot(exchange, local_eid_it->second, remote_eid_it->second, peer, peer, qp_num_, state);
        if (create_ret != ACLSHMEM_SUCCESS) {
            return create_ret;
        }
    }
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::BuildChannels(const EndpointExchange& exchange, ChannelBuildState& state)
{
    const size_t peer_count = static_cast<size_t>(rank_count_ - 1U);
    const size_t channel_multiplier = ACLSHMEM_UDMA_RELAY_ENABLED ? peer_count : static_cast<size_t>(qp_num_);
    size_t channel_count = 0;
    if (!CheckedMultiply(peer_count, channel_multiplier, channel_count)) {
        SHM_LOG_ERROR(
            "UDMA channel count overflow, peerCount = " << peer_count << ", multiplier = " << channel_multiplier);
        return ACLSHMEM_INVALID_VALUE;
    }
    state.handles.reserve(channel_count);
    state.slots.reserve(channel_count);
    state.dst_pes.reserve(channel_count);
    state.qp_indices.reserve(channel_count);
    if constexpr (!ACLSHMEM_UDMA_RELAY_ENABLED) {
        if (qp_num_ > 1U) {
            state.channel_names.reserve(channel_count);
        }
    }

    // Single relay vs. direct decision for the whole class (see ACLSHMEM_UDMA_RELAY_ENABLED).
    if constexpr (ACLSHMEM_UDMA_RELAY_ENABLED) {
        return BuildRelayChannels(exchange, state);
    } else {
        return BuildDirectChannels(exchange, state);
    }
}

Result UdmaTransportManager::AsyncConnect()
{
    EndpointExchange exchange;
    auto ret = ExchangeEndpointDescriptors(exchange);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }

    ChannelBuildState state;
    const Result channel_build_ret = BuildChannels(exchange, state);
    if (channel_build_ret != ACLSHMEM_SUCCESS) {
        // Destroy any channels created before the failure.
        if (!state.handles.empty()) {
            (void)DlHcommApi::HcommChannelDestroy(state.handles.data(), static_cast<uint32_t>(state.handles.size()));
        }
        return channel_build_ret;
    }

    channel_handles_ = std::move(state.handles);
    // HcommChannelCreate is non-blocking; wait for every channel to finish connecting before the
    // data plane consumes them.
    auto status_ret = WaitHcommChannelReady(channel_handles_);
    if (status_ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Check hcomm UDMA channel status failed, ret = " << status_ret);
        DestroyChannels();
        return status_ret;
    }

    auto build_ret = BuildUdmaInfo(channel_handles_, state.slots, state.dst_pes, state.qp_indices);
    if (build_ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("BuildUdmaInfo failed, ret = " << build_ret);
        DestroyChannels();
        FreeDeviceInfo();
        return build_ret;
    }

    SHM_LOG_INFO(
        "Create hcomm channels success, totalChannelNum = " << channel_handles_.size()
                                                            << ", qpNumPerPeer = " << qp_num_);
    connected_ = true;
    return ACLSHMEM_SUCCESS;
}

bool UdmaTransportManager::BuildSyncEndpoints(
    const RootInfo& root_info, uint32_t rank_count, std::vector<std::vector<SyncEndpoint>>& out)
{
    std::vector<SyncEndpoint> local_endpoints;
    for (const auto& item : root_info.eidIndexToRankAddr) {
        const auto& rank_addr = item.second;
        if (rank_addr == nullptr || rank_addr->levelInfo == nullptr) {
            continue;
        }
        SyncEndpoint endpoint;
        endpoint.netLayer = rank_addr->levelInfo->netLayer;
        endpoint.netInstanceId = rank_addr->levelInfo->netInstanceId;
        endpoint.planeId = rank_addr->planeId;
        endpoint.ports = rank_addr->ports;
        endpoint.eidIndex = item.first;
        local_endpoints.push_back(std::move(endpoint));
    }

    const std::vector<uint8_t> local_blob = nlohmann::json::to_msgpack(nlohmann::json(local_endpoints));

    uint32_t local_len = static_cast<uint32_t>(local_blob.size());
    std::vector<uint32_t> all_len(rank_count, 0);
    g_boot_handle.allgather(&local_len, all_len.data(), sizeof(uint32_t), &g_boot_handle);

    uint32_t max_len = 0;
    for (const uint32_t len : all_len) {
        max_len = std::max(max_len, len);
    }
    if (max_len == 0) {
        SHM_LOG_ERROR("No sync endpoints found on any rank.");
        return false;
    }

    std::vector<uint8_t> send_buf(max_len, 0);
    std::copy(local_blob.begin(), local_blob.end(), send_buf.begin());

    std::vector<uint8_t> recv_buf(static_cast<size_t>(max_len) * rank_count, 0);
    g_boot_handle.allgather(send_buf.data(), recv_buf.data(), static_cast<int>(max_len), &g_boot_handle);

    out.assign(rank_count, std::vector<SyncEndpoint>{});
    for (uint32_t rank = 0; rank < rank_count; ++rank) {
        const uint32_t len = all_len[rank];
        if (len == 0) {
            continue;
        }
        const uint8_t* begin = recv_buf.data() + static_cast<size_t>(rank) * max_len;
        try {
            out[rank] = nlohmann::json::from_msgpack(begin, begin + len).get<std::vector<SyncEndpoint>>();
        } catch (const std::exception& ex) {
            SHM_LOG_ERROR("Failed to decode sync endpoints for rank " << rank << ": " << ex.what());
            return false;
        }
    }
    return true;
}

bool UdmaTransportManager::PrepareOpenDevice(uint32_t device_id, uint32_t rank_count)
{
    RootInfo root_info;
    TopoInfo topo_info;
    uint32_t local_id = 0;
    uint32_t eid_count = 0;
    int32_t phy_id = -1;

    auto ret = DlAclApi::AclrtGetPhyDevIdByLogicDevId(static_cast<int32_t>(user_id_), &phy_id);
    SHM_ASSERT_LOG_AND_RETURN(
        ret == 0 && phy_id >= 0,
        "AclrtGetPhyDevIdByLogicDevId() return=" << ret << ", user_id=" << user_id_ << ", logic_device_id=" << device_id
                                                 << ", output phy_id=" << phy_id,
        false);

    if (!TopoReader::ParseRootInfo(phy_id, root_info)) {
        SHM_LOG_ERROR("Failed to parse the rootinfo file.");
        return false;
    }
    phy_id_ = static_cast<uint32_t>(phy_id);
    SHM_LOG_INFO(
        "Resolved phy id from current device mapping, user_id=" << user_id_ << ", logic_device_id=" << device_id
                                                                << ", phy_id=" << phy_id_);
    if (!TopoReader::ParseTopoInfo(root_info.topoFilePath, topo_info)) {
        SHM_LOG_ERROR("Failed to parse the topology file at path " << root_info.topoFilePath);
        return false;
    }

    if (!TopoReader::GetLocalId(root_info, phy_id_, local_id)) {
        SHM_LOG_ERROR("Failed to find local_id for phy_id: " << phy_id_);
        return false;
    }
    if (!TopoReader::GetEidCount(root_info, eid_count)) {
        SHM_LOG_ERROR("Failed to find eid count from rootinfo.");
        return false;
    }

    std::vector<uint32_t> eid_count_list(rank_count);
    g_boot_handle.allgather(&eid_count, eid_count_list.data(), sizeof(uint32_t), &g_boot_handle);
    const auto max_eid_count_it = std::max_element(eid_count_list.begin(), eid_count_list.end());
    const uint32_t max_eid_count = (max_eid_count_it == eid_count_list.end()) ? 0 : *max_eid_count_it;
    if (max_eid_count == 0) {
        SHM_LOG_ERROR("Invalid eidSlotCount resolved from rootinfo rank_addr_list.");
        return false;
    }

    std::vector<uint32_t> local_id_list(rank_count);
    g_boot_handle.allgather(&local_id, local_id_list.data(), sizeof(uint32_t), &g_boot_handle);

    std::vector<std::vector<SyncEndpoint>> rank_idx_to_sync_endpoint;
    if (!BuildSyncEndpoints(root_info, rank_count, rank_idx_to_sync_endpoint)) {
        SHM_LOG_ERROR("Failed to gather sync endpoints across ranks.");
        return false;
    }

    TopoQuerier topo_querier(root_info, topo_info, rank_id_, local_id_list, rank_idx_to_sync_endpoint);

    // Local outbound EID index toward each peer, later allgathered into the full N x N routing
    // matrix the relay path consumes. INVALID_EID_INDEX marks self / unroutable slots.
    std::vector<int32_t> local_route_by_peer(rank_count, INVALID_EID_INDEX);

    for (uint32_t peer = 0; peer < rank_count; ++peer) {
        if (peer == rank_id_) {
            continue;
        }
        uint32_t eid_index = 0;
        uint32_t remote_eid_index = 0;
        EidData eid_raw{};
        if (!topo_querier.GetEidRoute(peer, eid_index, eid_raw, remote_eid_index)) {
            SHM_LOG_ERROR(
                "Failed to resolve the local EID route for peer rank "
                << peer << ". The local_id was " << local_id << " and the peer local_id was " << local_id_list[peer]);
            return false;
        }
        if (remote_eid_index >= max_eid_count) {
            SHM_LOG_ERROR(
                "Invalid remote EID route for peer rank "
                << peer << ", remote_eid_index = " << remote_eid_index << ", local rank = " << rank_id_
                << ", local_id = " << local_id << ", peer_local_id = " << local_id_list[peer]
                << ", eidSlotCount = " << max_eid_count);
            return false;
        }
        // The Clos-aware router yields both the local outbound and the peer's inbound EID on the
        // same plane, so the remote index is taken directly (no reverse-lookup allgather needed).
        peer_eid_index_map_[peer] = eid_index;
        peer_remote_eid_index_map_[peer] = remote_eid_index;
        local_route_by_peer[peer] = static_cast<int32_t>(eid_index);

        if (!CreateEndpoint(eid_index, eid_raw)) {
            SHM_LOG_ERROR("CreateEndpoint failed for peer " << peer << " with EID index " << eid_index);
            return false;
        }
    }

    // Relay path only: allgather the per-peer local routes into the full N x N matrix so
    // ResolveRelaySlotRoute can look up any (actual_pe, relay_pe) forwarding EID. The direct
    // path relies solely on the peer maps filled above, so this is gated on the relay switch.
    if constexpr (ACLSHMEM_UDMA_RELAY_ENABLED) {
        std::vector<int32_t> all_route_by_peer(rank_count * rank_count, INVALID_EID_INDEX);
        g_boot_handle.allgather(
            local_route_by_peer.data(), all_route_by_peer.data(), sizeof(int32_t) * rank_count, &g_boot_handle);
        all_local_routes_ = std::move(all_route_by_peer);
    }

    return true;
}

bool UdmaTransportManager::CreateEndpoint(
    uint32_t eid_index, const std::array<uint8_t, URMA_EID_RAW_SIZE>& target_eid_raw)
{
    auto endpoint_it = endpoint_handle_map_.find(eid_index);
    if (endpoint_it != endpoint_handle_map_.end() && endpoint_it->second != nullptr) {
        return true;
    }

    EndpointDesc endpoint_desc{};
    auto desc_init_ret = EndpointDescInit(&endpoint_desc, 1);
    if (desc_init_ret != 0) {
        SHM_LOG_ERROR("EndpointDescInit failed for EID index " << eid_index << ", ret = " << desc_init_ret);
        return false;
    }

    uint32_t sd_id = 0;
    uint32_t server_id = 0;
    uint32_t super_pod_id = 0;
    auto device_info_ret = shm::MemSegment::GetDeviceInfo(sd_id, server_id, super_pod_id);
    if (device_info_ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Get local device info for HCOMM endpoint failed, ret = " << device_info_ret);
        return false;
    }

    endpoint_desc.protocol = COMM_PROTOCOL_UBC_CTP;
    endpoint_desc.commAddr.type = COMM_ADDR_TYPE_EID;
    int copyRet = memcpy_s(
        endpoint_desc.commAddr.eid, sizeof(endpoint_desc.commAddr.eid), target_eid_raw.data(), target_eid_raw.size());
    if (copyRet != EOK) {
        SHM_LOG_ERROR("Copy target EID to HCOMM endpoint desc failed, ret = " << copyRet);
        return false;
    }
    endpoint_desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    endpoint_desc.loc.device.devPhyId = phy_id_;
    endpoint_desc.loc.device.superDevId = sd_id;
    endpoint_desc.loc.device.serverIdx = server_id;
    endpoint_desc.loc.device.superPodIdx = super_pod_id;

    EndpointHandle endpoint_handle = nullptr;
    auto ret = DlHcommApi::HcommEndpointCreate(&endpoint_desc, &endpoint_handle);
    if (ret != 0 || endpoint_handle == nullptr) {
        SHM_LOG_ERROR("HcommEndpointCreate failed for EID index " << eid_index << ", ret = " << ret);
        return false;
    }
    endpoint_desc_map_[eid_index] = endpoint_desc;
    endpoint_handle_map_[eid_index] = endpoint_handle;
    return true;
}

Result UdmaTransportManager::CheckPrepareOptions(const HybmTransPrepareOptions& options)
{
    if (role_ != HYBM_ROLE_PEER) {
        SHM_LOG_INFO("Transport role: " << role_ << " check options passed.");
        return ACLSHMEM_SUCCESS;
    }

    if (options.options.size() > rank_count_) {
        SHM_LOG_ERROR("Options size() is " << options.options.size() << " larger than rank count: " << rank_count_);
        return ACLSHMEM_INVALID_PARAM;
    }

    if (options.options.find(rank_id_) == options.options.end()) {
        SHM_LOG_ERROR("Options do not contain self rankId: " << rank_id_);
        return ACLSHMEM_INVALID_PARAM;
    }

    for (auto it = options.options.begin(); it != options.options.end(); ++it) {
        if (it->first >= rank_count_) {
            SHM_LOG_ERROR("RankId: " << it->first << " is out of range [0, " << rank_count_ << ")");
            return ACLSHMEM_INVALID_PARAM;
        }
    }

    return ACLSHMEM_SUCCESS;
}

#if defined(ACLSHMEM_MSSANITIZER_BUILD)
void UdmaTransportManager::RegisterUdmaQpSanitizerRegions(const aclshmemi_aiv_udma_info_t& copy_info)
{
    // Drop any regions reported by a previous BuildUdmaInfo (e.g. reconnect) so the sanitizer's
    // region list does not accumulate stale URMA ring addresses.
    if (udma_qp_regions_registered_) {
        UnregisterUdmaQpSanitizerRegions();
    }
    if (mstx_reg_ptr_ == nullptr) {
        mstx_reg_ptr_.reset(shm::create_mstx_mem_register_instance());
    }
    if (mstx_reg_ptr_ == nullptr) {
        SHM_LOG_WARN("create mstx_mem_register instance failed, skip UDMA QP region report");
        return;
    }

    // At the call site the section pointers still reference the host-side blob, and buf_addr
    // holds the URMA/HCOMM device VA of each ring (see FillWqCtx / FillCqCtx).
    const aclshmemi_udma_qp_table_t& tbl = ActiveUdmaTable(copy_info);
    const auto* wq_array = reinterpret_cast<const aclshmemi_udma_wq_ctx_t*>(tbl.sq_ptr);
    const auto* cq_array = reinterpret_cast<const aclshmemi_udma_cq_ctx_t*>(tbl.scq_ptr);

    bool has_region = false;
    const uint64_t qp_count = static_cast<uint64_t>(copy_info.qp_num);
    const uint64_t context_count = SlotCount() * qp_count;
    for (uint64_t context_idx = 0; context_idx < context_count; ++context_idx) {
        // The data plane writes a 32-bit index into the SQ/CQ doorbells (db_addr = dbVa),
        // which are separate URMA MMIO addresses (see post_send_update_info / poll_cq_update_info).
        // They must be reported too, otherwise the doorbell ring is flagged as an illegal write.
        constexpr uint64_t doorbell_report_size = sizeof(uint32_t);
        const auto& wq = wq_array[context_idx];
        if (wq.buf_addr != 0 && wq.wqe_size != 0 && wq.depth != 0) {
            mstx_reg_ptr_->add_mem_regions(
                reinterpret_cast<void*>(wq.buf_addr), static_cast<uint64_t>(wq.depth) * wq.wqe_size,
                MSTX_GROUP_FOR_UDMA_QP);
            has_region = true;
        }
        if (wq.db_addr != 0) {
            mstx_reg_ptr_->add_mem_regions(
                reinterpret_cast<void*>(wq.db_addr), doorbell_report_size, MSTX_GROUP_FOR_UDMA_QP);
            has_region = true;
        }
        const auto& cq = cq_array[context_idx];
        if (cq.buf_addr != 0 && cq.cqe_size != 0 && cq.depth != 0) {
            mstx_reg_ptr_->add_mem_regions(
                reinterpret_cast<void*>(cq.buf_addr), static_cast<uint64_t>(cq.depth) * cq.cqe_size,
                MSTX_GROUP_FOR_UDMA_QP);
            has_region = true;
        }
        if (cq.db_addr != 0) {
            mstx_reg_ptr_->add_mem_regions(
                reinterpret_cast<void*>(cq.db_addr), doorbell_report_size, MSTX_GROUP_FOR_UDMA_QP);
            has_region = true;
        }
    }

    if (has_region) {
        mstx_reg_ptr_->mstx_mem_regions_register(MSTX_GROUP_FOR_UDMA_QP);
        udma_qp_regions_registered_ = true;
        SHM_LOG_INFO(
            "Reported UDMA SQ/CQ rings to mssanitizer, slot_count = " << SlotCount()
                                                                      << ", qp_num = " << copy_info.qp_num);
    }
}

void UdmaTransportManager::UnregisterUdmaQpSanitizerRegions()
{
    if (mstx_reg_ptr_ == nullptr) {
        return;
    }
    mstx_reg_ptr_->mstx_mem_regions_unregister(MSTX_GROUP_FOR_UDMA_QP);
    // The registrar keeps the range descriptors cached after unregister and exposes no
    // per-group clear; drop the instance so the next report starts from a clean list.
    // RegisterUdmaQpSanitizerRegions recreates it on demand, which keeps teardown paths
    // from building a domain only to destroy it right away.
    mstx_reg_ptr_.reset();
    udma_qp_regions_registered_ = false;
}
#endif

void UdmaTransportManager::FreeDeviceInfo()
{
    if (udma_info_dev_ != nullptr) {
        (void)DlAclApi::AclrtFree(udma_info_dev_);
        udma_info_dev_ = nullptr;
    }
    if (eid_dev_ != nullptr) {
        (void)DlAclApi::AclrtFree(eid_dev_);
        eid_dev_ = nullptr;
    }
    for (auto& amo_dev : amo_dev_list_) {
        if (amo_dev != nullptr) {
            (void)DlAclApi::AclrtFree(amo_dev);
            amo_dev = nullptr;
        }
    }
    amo_dev_list_.clear();
    udma_info_size_ = 0;
}

void UdmaTransportManager::DestroyChannels()
{
#if defined(ACLSHMEM_MSSANITIZER_BUILD)
    // mssanitizer: withdraw the UDMA SQ/CQ ring/doorbell regions BEFORE the HCOMM channels
    // (and the URMA-owned sqVa/scqVa/dbVa they hold) are torn down. Converging the unregister
    // here makes every channel-teardown path do it in the right order: CleanupResources
    // (destructor), UnregisterMemoryRegion, AsyncConnect rollback and the BuildUdmaInfo failure that
    // falls back to AsyncConnect. This guarantees each register has exactly one matching
    // unregister that precedes the channel release, so no stale region survives a VA reuse.
    if (udma_qp_regions_registered_) {
        UnregisterUdmaQpSanitizerRegions();
    }
#endif
    if (!channel_handles_.empty()) {
        auto hcomm_ret =
            DlHcommApi::HcommChannelDestroy(channel_handles_.data(), static_cast<uint32_t>(channel_handles_.size()));
        if (hcomm_ret != 0) {
            SHM_LOG_WARN("HcommChannelDestroy failed, ret = " << hcomm_ret);
        }
    }
    channel_handles_.clear();
    SHM_LOG_INFO("Destroy hcomm channels success.");
}

void UdmaTransportManager::CleanupResources()
{
    DestroyChannels();
    FreeDeviceInfo();

#if defined(ACLSHMEM_MSSANITIZER_BUILD)
    // DestroyChannels() already withdrew the UDMA SQ/CQ regions and dropped the registrar, so
    // this only covers the path where regions were never reported.
    mstx_reg_ptr_.reset();
    udma_qp_regions_registered_ = false;
#endif

    for (const auto& mem_by_addr : mem_record_map_) {
        for (const auto& mem_entry : mem_by_addr.second) {
            auto endpoint_it = endpoint_handle_map_.find(mem_entry.first);
            if (endpoint_it == endpoint_handle_map_.end() || endpoint_it->second == nullptr ||
                mem_entry.second == nullptr) {
                continue;
            }
            auto hcomm_ret = DlHcommApi::HcommMemUnreg(endpoint_it->second, mem_entry.second);
            if (hcomm_ret != 0) {
                SHM_LOG_WARN("HcommMemUnreg failed for EID index " << mem_entry.first << ", ret = " << hcomm_ret);
            }
        }
    }
    mem_record_map_.clear();
    SHM_LOG_INFO("Unregister memory success.");

    for (const auto& endpoint_entry : endpoint_handle_map_) {
        if (endpoint_entry.second == nullptr) {
            continue;
        }
        auto ret = DlHcommApi::HcommEndpointDestroy(endpoint_entry.second);
        if (ret != 0) {
            SHM_LOG_WARN("HcommEndpointDestroy failed for EID index " << endpoint_entry.first << ", ret = " << ret);
        }
    }
    endpoint_handle_map_.clear();
    peer_eid_index_map_.clear();
    peer_remote_eid_index_map_.clear();
    endpoint_desc_map_.clear();
    channel_handles_.clear();
    connected_ = false;
}

} // namespace device
} // namespace transport
} // namespace shm
