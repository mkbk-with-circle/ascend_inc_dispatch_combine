/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "utils/exception/shmem_exception_udma_dump.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <securec.h>

#include "utils/exception/shmem_exception_report.h"
#include "utils/exception/shmemi_device_udma_exception_report_kernel.h"
#include "dl_acl_api.h"
#include "shmemi_host_common.h"
#include "transport/device_udma/device_udma_def.h"

namespace {
using shm::transport::device::aclshmemi_aiv_udma_info_t;
using shm::transport::device::aclshmemi_ubmem_info_t;
using shm::transport::device::aclshmemi_udma_cq_ctx_t;
using shm::transport::device::aclshmemi_udma_db_mode_t;
using shm::transport::device::aclshmemi_udma_qp_table_t;
using shm::transport::device::aclshmemi_udma_wq_ctx_t;

constexpr uint32_t ACLSHMEMI_EXCEPTION_REPORT_MAX_QP_NUM = 64;
constexpr uint32_t ACLSHMEMI_EXCEPTION_REPORT_CQE_WINDOW = 4;
constexpr uint32_t ACLSHMEMI_EXCEPTION_REPORT_WQE_WINDOW = 2;
constexpr uint32_t ACLSHMEMI_EXCEPTION_REPORT_WQE_RAW_BYTES_PER_LINE = 16;
constexpr uint32_t ACLSHMEMI_EXCEPTION_REPORT_MAX_WQE_SIZE = 256;
constexpr uint32_t ACLSHMEMI_EXCEPTION_REPORT_MAX_WQ_DEPTH = 1U << 20U;

// Only the common SQE header is decoded here. Opcode-specific payload remains raw diagnostic data.
struct aclshmemi_exception_report_sqe_t {
    uint32_t sqe_bb_idx : 16;
    uint32_t flag : 8;
    uint32_t rsv0 : 3;
    uint32_t nf : 1;
    uint32_t token_en : 1;
    uint32_t rmt_jetty_type : 2;
    uint32_t owner : 1;
    uint32_t target_hint : 8;
    uint32_t opcode : 8;
    uint32_t rsv1 : 6;
    uint32_t inline_msg_len : 10;
    uint32_t tp_id : 24;
    uint32_t sge_num : 8;
    uint32_t rmt_jetty_or_seg_id : 20;
    uint32_t rsv2 : 12;
    uint64_t rmt_eid_l;
    uint64_t rmt_eid_h;
    uint32_t rmt_token_value;
    uint32_t udf_type : 8;
    uint32_t reduce_data_type : 4;
    uint32_t reduce_opcode : 4;
    uint32_t rsv3 : 16;
    uint32_t rmt_addr_l_or_token_id;
    uint32_t rmt_addr_h_or_token_value;
};

static_assert(sizeof(aclshmemi_exception_report_sqe_t) == 48U, "UDMA SQE layout changed.");

struct aclshmemi_exception_report_wqe_block_t {
    uint32_t base_bb{0};
    uint32_t index{0};
    uint64_t captured_size{0};
    uint64_t requested_size{0};
    uint64_t addr{0};
    bool header_valid{false};
    aclshmemi_exception_report_sqe_t header{};
    std::vector<uint8_t> raw{};
};

struct aclshmemi_exception_report_slot_t {
    uint64_t slot{0};
    uint32_t peer{0};
    uint32_t relay_peer{0};
    bool relay_enabled{false};
};

const aclshmemi_udma_qp_table_t& aclshmemi_exception_report_active_udma_table(const aclshmemi_aiv_udma_info_t& info)
{
#if defined(ACLSHMEM_RELAY_SUPPORT)
    return info.relay;
#else
    return info.direct;
#endif
}

uint64_t aclshmemi_exception_report_udma_slot_count()
{
    const uint64_t npes = static_cast<uint64_t>(g_state.npes);
#if defined(ACLSHMEM_RELAY_SUPPORT)
    return npes * npes;
#else
    return npes;
#endif
}

aclshmemi_exception_report_slot_t aclshmemi_exception_report_make_slot(uint64_t slot)
{
    aclshmemi_exception_report_slot_t label{};
    label.slot = slot;
#if defined(ACLSHMEM_RELAY_SUPPORT)
    const uint64_t npes = static_cast<uint64_t>(g_state.npes);
    label.peer = static_cast<uint32_t>(slot / npes);
    label.relay_peer = static_cast<uint32_t>(slot % npes);
    label.relay_enabled = true;
#else
    label.peer = static_cast<uint32_t>(slot);
    label.relay_peer = 0U;
    label.relay_enabled = false;
#endif
    return label;
}

std::string aclshmemi_exception_report_slot_text(const aclshmemi_exception_report_slot_t& label)
{
    std::ostringstream stream;
    stream << "slot=" << label.slot << " peer=" << label.peer;
    if (label.relay_enabled) {
        stream << " relayPeer=" << label.relay_peer;
    }
    return stream.str();
}

template <typename T>
int aclshmemi_exception_report_copy_from_device(T& dst, uint64_t src_addr, const char* name)
{
    if (src_addr == 0) {
        SHM_LOG_ERROR("[EXCEPTION][UDMA] " << name << " device address is null, skip dump.");
        return ACLSHMEM_INNER_ERROR;
    }
    auto ret = shm::DlAclApi::AclrtMemcpy(
        &dst, sizeof(T), reinterpret_cast<const void*>(static_cast<uintptr_t>(src_addr)), sizeof(T),
        ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("[EXCEPTION][UDMA] copy " << name << " failed, ret = " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

template <typename T>
int aclshmemi_exception_report_copy_array(std::vector<T>& dst, uint64_t src_addr, const char* name)
{
    if (dst.empty()) {
        return ACLSHMEM_SUCCESS;
    }
    if (src_addr == 0) {
        SHM_LOG_ERROR("[EXCEPTION][UDMA] " << name << " section address is null, skip dump.");
        return ACLSHMEM_INNER_ERROR;
    }
    const size_t copy_size = dst.size() * sizeof(T);
    auto ret = shm::DlAclApi::AclrtMemcpy(
        dst.data(), copy_size, reinterpret_cast<const void*>(static_cast<uintptr_t>(src_addr)), copy_size,
        ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("[EXCEPTION][UDMA] copy " << name << " failed, ret = " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

template <typename T>
bool aclshmemi_exception_report_read_raw(const std::vector<uint8_t>& raw, size_t offset, T& value)
{
    if (offset > raw.size() || sizeof(T) > raw.size() - offset) {
        return false;
    }
    int ret = memcpy_s(&value, sizeof(T), raw.data() + offset, sizeof(T));
    if (ret != EOK) {
        SHM_LOG_ERROR(
            "[EXCEPTION][UDMA] copy raw WQE field failed, offset = " << offset << ", size = " << sizeof(T)
                                                                     << ", ret = " << ret);
        return false;
    }
    return true;
}

std::string aclshmemi_exception_report_bytes_to_hex(const uint8_t* data, size_t size)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (size_t i = 0; i < size; ++i) {
        if (i != 0) {
            stream << ' ';
        }
        stream << std::setw(2) << static_cast<uint32_t>(data[i]);
    }
    return stream.str();
}

int aclshmemi_udma_exception_report_read_entry(
    uint32_t entry_type, uint64_t entry_addr, uint64_t entry_size, aclshmemi_udma_exception_report_entry_t& entry,
    aclshmemi_udma_exception_report_entry_t* device_entry, const char* name)
{
    if (entry_addr == 0 || entry_size == 0 || device_entry == nullptr) {
        return ACLSHMEM_INNER_ERROR;
    }

    int ret = shm::DlAclApi::AclrtMemset(
        device_entry, sizeof(aclshmemi_udma_exception_report_entry_t), 0,
        sizeof(aclshmemi_udma_exception_report_entry_t));
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("[EXCEPTION][UDMA] clear " << name << " report buffer failed, ret = " << ret);
        return ACLSHMEM_INNER_ERROR;
    }

    ret = aclshmemi_udma_exception_report_read_entry_on_stream(
        entry_type, entry_addr, entry_size, static_cast<aclshmemi_udma_exception_report_entry_t*>(device_entry),
        static_cast<aclrtStream>(g_state_host.default_stream));
    if (ret == ACLSHMEM_SUCCESS) {
        ret = shm::DlAclApi::AclrtSynchronizeStream(g_state_host.default_stream);
    }
    if (ret == ACLSHMEM_SUCCESS) {
        ret = shm::DlAclApi::AclrtMemcpy(&entry, sizeof(entry), device_entry, sizeof(entry), ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (ret != ACLSHMEM_SUCCESS || entry.ret != ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_SUCCESS) {
        SHM_LOG_ERROR(
            "[EXCEPTION][UDMA] read " << name << " by kernel failed, ret = " << ret << ", entryRet = " << entry.ret);
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_udma_exception_report_read_cqe(
    uint64_t entry_addr, uint64_t entry_size, aclshmemi_udma_exception_report_entry_t& entry,
    aclshmemi_udma_exception_report_entry_t* device_entry)
{
    return aclshmemi_udma_exception_report_read_entry(
        ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_CQE, entry_addr, entry_size, entry, device_entry, "CQE");
}

int aclshmemi_udma_exception_report_read_wq_ctx(
    uint64_t entry_addr, aclshmemi_udma_wq_ctx_t& wq, aclshmemi_udma_exception_report_entry_t* device_entry)
{
    aclshmemi_udma_exception_report_entry_t entry{};
    int ret = aclshmemi_udma_exception_report_read_entry(
        ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_WQ, entry_addr, sizeof(aclshmemi_udma_wq_ctx_t), entry, device_entry,
        "UDMA WQ context");
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }
    wq.wqn = entry.wq.wqn;
    wq.buf_addr = entry.wq.buf_addr;
    wq.wqe_size = entry.wq.wqe_size;
    wq.depth = entry.wq.depth;
    wq.head = entry.wq.head;
    wq.tail = entry.wq.tail;
    wq.db_mode = static_cast<aclshmemi_udma_db_mode_t>(entry.wq.db_mode);
    wq.db_addr = entry.wq.db_addr;
    wq.sl = entry.wq.sl;
    wq.wqe_cnt = entry.wq.wqe_cnt;
    wq.amo_addr = entry.wq.amo_addr;
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_udma_exception_report_read_cq_ctx(
    uint64_t entry_addr, aclshmemi_udma_cq_ctx_t& cq, aclshmemi_udma_exception_report_entry_t* device_entry)
{
    aclshmemi_udma_exception_report_entry_t entry{};
    int ret = aclshmemi_udma_exception_report_read_entry(
        ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_CQ, entry_addr, sizeof(aclshmemi_udma_cq_ctx_t), entry, device_entry,
        "UDMA CQ context");
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }
    cq.cqn = entry.cq.cqn;
    cq.buf_addr = entry.cq.buf_addr;
    cq.cqe_size = entry.cq.cqe_size;
    cq.depth = entry.cq.depth;
    cq.head = entry.cq.head;
    cq.tail = entry.cq.tail;
    cq.db_mode = static_cast<aclshmemi_udma_db_mode_t>(entry.cq.db_mode);
    cq.db_addr = entry.cq.db_addr;
    return ACLSHMEM_SUCCESS;
}

void aclshmemi_exception_report_log_mem(
    const aclshmemi_exception_report_slot_t& label, const aclshmemi_ubmem_info_t& mem)
{
    SHM_LOG_ERROR(
        "[EXCEPTION][UDMA MEM] " << aclshmemi_exception_report_slot_text(label)
                                 << " tokenValid=" << mem.token_value_valid << " rmtJettyType=" << mem.rmt_jetty_type
                                 << " targetHint=" << static_cast<uint32_t>(mem.target_hint) << " tpn=" << mem.tpn
                                 << " tid=" << mem.tid << " token=" << mem.rmt_token_value << " len=" << mem.len
                                 << " addr=0x" << std::hex << mem.addr << " eidAddr=0x" << mem.eid_addr << std::dec);
}

uint32_t aclshmemi_exception_report_cqe_expected_owner(uint32_t tail, uint32_t depth)
{
    if (depth == 0) {
        return 0;
    }
    return ((tail / depth) & 1U) ^ 1U;
}

uint32_t aclshmemi_exception_report_pending_depth(uint32_t head, uint32_t tail) { return head - tail; }

void aclshmemi_exception_report_log_cqe(
    const aclshmemi_exception_report_slot_t& label, uint32_t qp_idx, uint32_t tail, uint32_t depth, uint32_t head,
    uint32_t index, uint64_t cqe_addr, const aclshmemi_udma_exception_report_cqe_t& cqe)
{
    const uint64_t user_data = (static_cast<uint64_t>(cqe.user_data_h) << 32U) | cqe.user_data_l;
    const uint32_t expected_owner = aclshmemi_exception_report_cqe_expected_owner(tail, depth);
    const uint32_t ready = cqe.owner == expected_owner ? 1U : 0U;
    SHM_LOG_ERROR(
        "[EXCEPTION][UDMA CQE] " << aclshmemi_exception_report_slot_text(label) << " qp=" << qp_idx << " tail=" << tail
                                 << " index=" << index << " addr=0x" << std::hex << cqe_addr << " owner=" << std::dec
                                 << cqe.owner << " expectedOwner=" << expected_owner << " ready=" << ready
                                 << " pendingDepth=" << aclshmemi_exception_report_pending_depth(head, tail)
                                 << " opcode=" << cqe.opcode << " status=0x" << std::hex << cqe.status
                                 << " substatus=0x" << cqe.substatus << std::dec);
    SHM_LOG_ERROR(
        "[EXCEPTION][UDMA CQE DETAIL] " << aclshmemi_exception_report_slot_text(label) << " qp=" << qp_idx << " index="
                                        << index << " entryIdx=" << cqe.entry_idx << " byteCnt=" << cqe.byte_cnt
                                        << " rmtIdx=" << cqe.rmt_idx << " tpn=" << cqe.tpn << " userData=0x" << std::hex
                                        << user_data << " rmtEid={0x" << cqe.rmt_eid[0] << ",0x" << cqe.rmt_eid[1]
                                        << ",0x" << cqe.rmt_eid[2] << ",0x" << cqe.rmt_eid[3] << "}" << std::dec);
}

void aclshmemi_exception_report_log_wqe_block(
    const aclshmemi_exception_report_slot_t& label, uint32_t qp_idx, const char* queue_name,
    const aclshmemi_exception_report_wqe_block_t& block)
{
    const uint32_t partial = block.captured_size < block.requested_size ? 1U : 0U;
    if (block.header_valid) {
        const auto& sqe = block.header;
        const uint64_t remote_addr =
            (static_cast<uint64_t>(sqe.rmt_addr_h_or_token_value) << 32U) | sqe.rmt_addr_l_or_token_id;
        SHM_LOG_ERROR(
            "[EXCEPTION][UDMA " << queue_name << " WQE HEADER CANDIDATE] "
                                << aclshmemi_exception_report_slot_text(label) << " qp=" << qp_idx
                                << " baseBb=" << block.base_bb << " index=" << block.index << " addr=0x" << std::hex
                                << block.addr << " owner=" << std::dec << sqe.owner << " opcode=" << sqe.opcode
                                << " sqeBbIdx=" << sqe.sqe_bb_idx << " flag=" << sqe.flag << " nf=" << sqe.nf
                                << " tokenEn=" << sqe.token_en << " rmtJettyType=" << sqe.rmt_jetty_type
                                << " targetHint=" << sqe.target_hint << " inlineMsgLen=" << sqe.inline_msg_len
                                << " tpId=" << sqe.tp_id << " sgeNum=" << sqe.sge_num
                                << " rmtJettyOrSegId=" << sqe.rmt_jetty_or_seg_id << " rmtToken=" << sqe.rmt_token_value
                                << " udfType=" << sqe.udf_type << " reduceDataType=" << sqe.reduce_data_type
                                << " reduceOpcode=" << sqe.reduce_opcode << " remoteAddrOrToken=0x" << std::hex
                                << remote_addr << " rmtEid={0x" << sqe.rmt_eid_l << ",0x" << sqe.rmt_eid_h << "}"
                                << std::dec);
    } else {
        SHM_LOG_ERROR(
            "[EXCEPTION][UDMA " << queue_name << " WQE HEADER CANDIDATE] "
                                << aclshmemi_exception_report_slot_text(label) << " qp=" << qp_idx
                                << " baseBb=" << block.base_bb << " index=" << block.index << " headerValid=0"
                                << " addr=0x" << std::hex << block.addr << std::dec);
    }
    SHM_LOG_ERROR(
        "[EXCEPTION][UDMA " << queue_name << " WQE RAW] " << aclshmemi_exception_report_slot_text(label)
                            << " qp=" << qp_idx << " baseBb=" << block.base_bb << " index=" << block.index
                            << " capturedBytes=" << block.captured_size << " requestedBytes=" << block.requested_size
                            << " partial=" << partial << " addr=0x" << std::hex << block.addr << std::dec);
    for (size_t offset = 0; offset < block.raw.size(); offset += ACLSHMEMI_EXCEPTION_REPORT_WQE_RAW_BYTES_PER_LINE) {
        const size_t line_size =
            std::min(static_cast<size_t>(ACLSHMEMI_EXCEPTION_REPORT_WQE_RAW_BYTES_PER_LINE), block.raw.size() - offset);
        SHM_LOG_ERROR(
            "[EXCEPTION][UDMA " << queue_name << " WQE RAW] " << aclshmemi_exception_report_slot_text(label) << " qp="
                                << qp_idx << " baseBb=" << block.base_bb << " rawOffset=" << offset << " data="
                                << aclshmemi_exception_report_bytes_to_hex(block.raw.data() + offset, line_size));
    }
}

void aclshmemi_exception_report_log_wqe_unavailable(
    const aclshmemi_exception_report_slot_t& label, uint32_t qp_idx, const char* queue_name,
    const aclshmemi_udma_wq_ctx_t& wq, const char* reason)
{
    SHM_LOG_ERROR(
        "[EXCEPTION][UDMA " << queue_name << " WQE DETAIL] unavailable " << aclshmemi_exception_report_slot_text(label)
                            << " qp=" << qp_idx << " reason=" << reason << " head=" << wq.head << " tail=" << wq.tail
                            << " depth=" << wq.depth << " wqeSize=" << wq.wqe_size << " wqeCnt=" << wq.wqe_cnt
                            << " bufAddr=0x" << std::hex << wq.buf_addr << std::dec);
}

bool aclshmemi_exception_report_is_wq_valid(const aclshmemi_udma_wq_ctx_t& wq)
{
    return wq.buf_addr != 0 && wq.depth != 0 && wq.depth <= ACLSHMEMI_EXCEPTION_REPORT_MAX_WQ_DEPTH &&
           wq.wqe_size != 0 && wq.wqe_size <= ACLSHMEMI_EXCEPTION_REPORT_MAX_WQE_SIZE;
}

void aclshmemi_exception_report_dump_cqes(
    const aclshmemi_exception_report_slot_t& label, uint32_t qp_idx, const aclshmemi_udma_cq_ctx_t& cq,
    aclshmemi_udma_exception_report_entry_t* device_entry)
{
    if (cq.buf_addr == 0 || cq.depth == 0 || cq.cqe_size == 0) {
        SHM_LOG_ERROR(
            "[EXCEPTION][UDMA CQE] " << aclshmemi_exception_report_slot_text(label) << " qp=" << qp_idx
                                     << " has invalid CQ buffer/depth/cqeSize.");
        return;
    }
    const uint32_t first_tail =
        cq.tail >= ACLSHMEMI_EXCEPTION_REPORT_CQE_WINDOW ? cq.tail - ACLSHMEMI_EXCEPTION_REPORT_CQE_WINDOW + 1U : 0U;
    for (uint32_t tail = first_tail; tail <= cq.tail; ++tail) {
        const uint32_t index = tail % cq.depth;
        const uint64_t cqe_addr = cq.buf_addr + static_cast<uint64_t>(index) * cq.cqe_size;
        aclshmemi_udma_exception_report_entry_t entry{};
        if (aclshmemi_udma_exception_report_read_cqe(cqe_addr, cq.cqe_size, entry, device_entry) != ACLSHMEM_SUCCESS) {
            return;
        }
        aclshmemi_exception_report_log_cqe(label, qp_idx, tail, cq.depth, cq.head, index, cqe_addr, entry.cqe);
        if (tail == UINT32_MAX) {
            break;
        }
    }
}

bool aclshmemi_udma_exception_report_read_wqe_block(
    const aclshmemi_udma_wq_ctx_t& wq, uint32_t base_bb, aclshmemi_exception_report_wqe_block_t& block,
    aclshmemi_udma_exception_report_entry_t* device_entry)
{
    if (!aclshmemi_exception_report_is_wq_valid(wq)) {
        return false;
    }
    const uint32_t index = base_bb % wq.depth;
    const uint64_t requested_size = wq.wqe_size;
    const uint64_t first_addr = wq.buf_addr + static_cast<uint64_t>(index) * wq.wqe_size;
    aclshmemi_udma_exception_report_entry_t entry{};
    if (aclshmemi_udma_exception_report_read_entry(
            ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_WQE_RAW, first_addr, requested_size, entry, device_entry,
            "WQE raw") != ACLSHMEM_SUCCESS) {
        return false;
    }
    block.base_bb = base_bb;
    block.index = index;
    block.captured_size = entry.wqe_raw.size;
    block.requested_size = requested_size;
    block.addr = first_addr;
    std::vector<uint8_t> raw(entry.wqe_raw.data, entry.wqe_raw.data + entry.wqe_raw.size);
    block.header_valid = aclshmemi_exception_report_read_raw(raw, 0, block.header);
    block.raw = std::move(raw);
    return true;
}

bool aclshmemi_exception_report_dump_wqe_block(
    const aclshmemi_exception_report_slot_t& label, uint32_t qp_idx, const char* queue_name,
    const aclshmemi_udma_wq_ctx_t& wq, uint32_t base_bb, aclshmemi_udma_exception_report_entry_t* device_entry)
{
    aclshmemi_exception_report_wqe_block_t block{};
    if (!aclshmemi_udma_exception_report_read_wqe_block(wq, base_bb, block, device_entry)) {
        aclshmemi_exception_report_log_wqe_unavailable(label, qp_idx, queue_name, wq, "readFailed");
        return false;
    }
    aclshmemi_exception_report_log_wqe_block(label, qp_idx, queue_name, block);
    return true;
}

void aclshmemi_exception_report_dump_wqe_window(
    const aclshmemi_exception_report_slot_t& label, uint32_t qp_idx, const char* queue_name,
    const aclshmemi_udma_wq_ctx_t& wq, uint32_t start_bb, uint32_t block_count,
    aclshmemi_udma_exception_report_entry_t* device_entry)
{
    for (uint32_t i = 0; i < block_count; ++i) {
        (void)aclshmemi_exception_report_dump_wqe_block(label, qp_idx, queue_name, wq, start_bb + i, device_entry);
    }
}

void aclshmemi_exception_report_dump_wqe(
    const aclshmemi_exception_report_slot_t& label, uint32_t qp_idx, const char* queue_name,
    const aclshmemi_udma_wq_ctx_t& wq, aclshmemi_udma_exception_report_entry_t* device_entry)
{
    if (!aclshmemi_exception_report_is_wq_valid(wq)) {
        aclshmemi_exception_report_log_wqe_unavailable(label, qp_idx, queue_name, wq, "invalidWqState");
        return;
    }
    if (wq.head == 0) {
        aclshmemi_exception_report_dump_wqe_window(label, qp_idx, queue_name, wq, 0U, 1U, device_entry);
        return;
    }
    const uint32_t block_count = std::min(wq.head, ACLSHMEMI_EXCEPTION_REPORT_WQE_WINDOW);
    aclshmemi_exception_report_dump_wqe_window(
        label, qp_idx, queue_name, wq, wq.head - block_count, block_count, device_entry);
}

int aclshmemi_exception_report_load_udma_info(aclshmemi_aiv_udma_info_t& info)
{
    const uint64_t udma_info_address = aclshmemi_exception_report_udma_info_address();
    if (udma_info_address == 0) {
        SHM_LOG_ERROR("[EXCEPTION][UDMA] udmaInfo is null, skip UDMA dump.");
        return ACLSHMEM_NOT_SUPPORTED;
    }
    if (g_state.npes <= 0 || g_state.npes > ACLSHMEM_MAX_PES) {
        SHM_LOG_ERROR("[EXCEPTION][UDMA] invalid npes=" << g_state.npes << ", skip UDMA dump.");
        return ACLSHMEM_INNER_ERROR;
    }

    ACLSHMEM_CHECK_RET(aclshmemi_exception_report_copy_from_device(info, udma_info_address, "UDMA info"));
    if (info.qp_num == 0 || info.qp_num > ACLSHMEMI_EXCEPTION_REPORT_MAX_QP_NUM) {
        SHM_LOG_ERROR("[EXCEPTION][UDMA] invalid qp_num=" << info.qp_num << ", skip UDMA dump.");
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}
} // namespace

int aclshmemi_exception_report_dump_udma(bool detail_enabled)
{
    aclshmemi_aiv_udma_info_t info{};
    int ret = aclshmemi_exception_report_load_udma_info(info);
    if (ret == ACLSHMEM_NOT_SUPPORTED) {
        return ACLSHMEM_SUCCESS;
    }
    ACLSHMEM_CHECK_RET(ret);

    const auto& table = aclshmemi_exception_report_active_udma_table(info);
    const uint64_t slot_count = aclshmemi_exception_report_udma_slot_count();
    if (slot_count > std::numeric_limits<size_t>::max() / info.qp_num) {
        SHM_LOG_ERROR(
            "[EXCEPTION][UDMA] slot count overflow, npes=" << g_state.npes << ", qpNum=" << info.qp_num
                                                           << ", slotCount=" << slot_count);
        return ACLSHMEM_INNER_ERROR;
    }
    const size_t entry_count = static_cast<size_t>(slot_count) * info.qp_num;
    std::vector<aclshmemi_udma_cq_ctx_t> scq(entry_count);
    ACLSHMEM_CHECK_RET(aclshmemi_exception_report_copy_array(scq, table.scq_ptr, "SCQ context"));

    std::vector<aclshmemi_udma_wq_ctx_t> sq(entry_count);
    ACLSHMEM_CHECK_RET(aclshmemi_exception_report_copy_array(sq, table.sq_ptr, "SQ context"));

    std::vector<aclshmemi_udma_wq_ctx_t> rq;
    std::vector<aclshmemi_ubmem_info_t> mem;
    if (detail_enabled) {
        rq.resize(entry_count);
        mem.resize(static_cast<size_t>(slot_count));
        ACLSHMEM_CHECK_RET(aclshmemi_exception_report_copy_array(rq, table.rq_ptr, "RQ context"));
        ACLSHMEM_CHECK_RET(aclshmemi_exception_report_copy_array(mem, table.mem_ptr, "MemInfo"));
    }

    SHM_LOG_ERROR(
        "[EXCEPTION][UDMA INFO] qpNum=" << info.qp_num << " slotCount=" << slot_count << " sqPtr=0x" << std::hex
                                        << table.sq_ptr << " rqPtr=0x" << table.rq_ptr << " scqPtr=0x" << table.scq_ptr
                                        << " rcqPtr=0x" << table.rcq_ptr << " memPtr=0x" << table.mem_ptr << std::dec);
    void* entry_report_device = nullptr;
    if (detail_enabled) {
        ret = shm::DlAclApi::AclrtMalloc(
            &entry_report_device, sizeof(aclshmemi_udma_exception_report_entry_t), ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("[EXCEPTION][UDMA] alloc CQE report buffer failed, ret = " << ret);
            return ACLSHMEM_INNER_ERROR;
        }
    }
    for (uint64_t slot = 0; slot < slot_count; ++slot) {
        const auto label = aclshmemi_exception_report_make_slot(slot);
        for (uint32_t qp_idx = 0; qp_idx < info.qp_num; ++qp_idx) {
            const size_t offset = static_cast<size_t>(slot) * info.qp_num + qp_idx;
            auto cq = scq[offset];
            auto wq = sq[offset];
            auto rq_wq = detail_enabled ? rq[offset] : aclshmemi_udma_wq_ctx_t{};
            bool cq_realtime_read = false;
            bool sq_realtime_read = false;
            bool rq_realtime_read = false;
            if (detail_enabled) {
                const uint64_t scq_ctx_addr =
                    table.scq_ptr + static_cast<uint64_t>(offset) * sizeof(aclshmemi_udma_cq_ctx_t);
                const uint64_t sq_ctx_addr =
                    table.sq_ptr + static_cast<uint64_t>(offset) * sizeof(aclshmemi_udma_wq_ctx_t);
                const uint64_t rq_ctx_addr =
                    table.rq_ptr + static_cast<uint64_t>(offset) * sizeof(aclshmemi_udma_wq_ctx_t);
                cq_realtime_read =
                    aclshmemi_udma_exception_report_read_cq_ctx(
                        scq_ctx_addr, cq, static_cast<aclshmemi_udma_exception_report_entry_t*>(entry_report_device)) ==
                    ACLSHMEM_SUCCESS;
                sq_realtime_read =
                    aclshmemi_udma_exception_report_read_wq_ctx(
                        sq_ctx_addr, wq, static_cast<aclshmemi_udma_exception_report_entry_t*>(entry_report_device)) ==
                    ACLSHMEM_SUCCESS;
                rq_realtime_read =
                    aclshmemi_udma_exception_report_read_wq_ctx(
                        rq_ctx_addr, rq_wq,
                        static_cast<aclshmemi_udma_exception_report_entry_t*>(entry_report_device)) == ACLSHMEM_SUCCESS;
            }
            if (cq.cqn == 0 && cq.buf_addr == 0) {
                continue;
            }
            std::ostringstream qp_log;
            qp_log << "[EXCEPTION][UDMA QP] " << aclshmemi_exception_report_slot_text(label) << " qp=" << qp_idx
                   << " cqPi=" << cq.head << " cqCi=" << cq.tail << " wqPi=" << wq.head << " wqCi=" << wq.tail
                   << " wqeCnt=" << wq.wqe_cnt;
            if (detail_enabled) {
                qp_log << " cqRealtimeRead=" << cq_realtime_read << " sqRealtimeRead=" << sq_realtime_read
                       << " rqRealtimeRead=" << rq_realtime_read;
            }
            SHM_LOG_ERROR(qp_log.str());
            if (detail_enabled) {
                SHM_LOG_ERROR(
                    "[EXCEPTION][UDMA QP DETAIL] " << aclshmemi_exception_report_slot_text(label) << " qp=" << qp_idx
                                                   << " cqn=" << cq.cqn << " wqn=" << wq.wqn << " depth=" << cq.depth
                                                   << " cqeSize=" << cq.cqe_size);
                aclshmemi_exception_report_dump_cqes(
                    label, qp_idx, cq, static_cast<aclshmemi_udma_exception_report_entry_t*>(entry_report_device));
                aclshmemi_exception_report_dump_wqe(
                    label, qp_idx, "SQ", wq,
                    static_cast<aclshmemi_udma_exception_report_entry_t*>(entry_report_device));
                aclshmemi_exception_report_dump_wqe(
                    label, qp_idx, "RQ", rq_wq,
                    static_cast<aclshmemi_udma_exception_report_entry_t*>(entry_report_device));
            }
        }
        if (detail_enabled) {
            aclshmemi_exception_report_log_mem(label, mem[static_cast<size_t>(slot)]);
        }
    }
    if (entry_report_device != nullptr) {
        (void)shm::DlAclApi::AclrtFree(entry_report_device);
    }
    return ACLSHMEM_SUCCESS;
}
