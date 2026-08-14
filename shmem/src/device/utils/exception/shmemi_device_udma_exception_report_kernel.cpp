/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "utils/exception/shmemi_device_udma_exception_report_kernel.h"

#include "kernel_operator.h"
#include "host_device/shmem_common_types.h"
#include "device/gm2gm/engine/shmemi_device_udma.h"

constexpr uint64_t ACLSHMEMI_UDMA_EXCEPTION_REPORT_DATA_CACHE_LINE_SIZE = 64U;

ACLSHMEM_DEVICE bool aclshmemi_udma_exception_report_read_cqe(
    __gm__ aclshmemi_jfc_cqe_ctx_t* cqe, uint64_t entry_size, __gm__ aclshmemi_udma_exception_report_entry_t* out);
ACLSHMEM_DEVICE bool aclshmemi_udma_exception_report_read_wq(
    __gm__ aclshmemi_udma_wq_ctx_t* wq, uint64_t entry_size, __gm__ aclshmemi_udma_exception_report_entry_t* out);
ACLSHMEM_DEVICE bool aclshmemi_udma_exception_report_read_cq(
    __gm__ aclshmemi_udma_cq_ctx_t* cq, uint64_t entry_size, __gm__ aclshmemi_udma_exception_report_entry_t* out);
ACLSHMEM_DEVICE bool aclshmemi_udma_exception_report_read_wqe_raw(
    __gm__ uint8_t* wqe, uint64_t entry_size, __gm__ aclshmemi_udma_exception_report_entry_t* out);

ACLSHMEM_DEVICE void aclshmemi_udma_exception_report_dcci_cachelines(__gm__ uint8_t* addr, uint64_t length)
{
    if (addr == nullptr || length == 0) {
        return;
    }
    __gm__ uint8_t* start = (__gm__ uint8_t*)((uint64_t)addr / ACLSHMEMI_UDMA_EXCEPTION_REPORT_DATA_CACHE_LINE_SIZE *
                                              ACLSHMEMI_UDMA_EXCEPTION_REPORT_DATA_CACHE_LINE_SIZE);
    __gm__ uint8_t* end =
        (__gm__ uint8_t*)(((uint64_t)addr + length - 1U) / ACLSHMEMI_UDMA_EXCEPTION_REPORT_DATA_CACHE_LINE_SIZE *
                          ACLSHMEMI_UDMA_EXCEPTION_REPORT_DATA_CACHE_LINE_SIZE);
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(start);
    for (uint64_t i = 0; i <= end - start; i += ACLSHMEMI_UDMA_EXCEPTION_REPORT_DATA_CACHE_LINE_SIZE) {
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<
            uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global[i]);
        __asm__ __volatile__("");
    }
}

ACLSHMEM_DEVICE bool aclshmemi_udma_exception_report_read_cqe(
    __gm__ aclshmemi_jfc_cqe_ctx_t* cqe, uint64_t entry_size, __gm__ aclshmemi_udma_exception_report_entry_t* out)
{
    if (entry_size < sizeof(aclshmemi_jfc_cqe_ctx_t)) {
        return false;
    }
    aclshmemi_udma_exception_report_dcci_cachelines(
        reinterpret_cast<__gm__ uint8_t*>(cqe), sizeof(aclshmemi_jfc_cqe_ctx_t));

    out->cqe.owner = cqe->owner;
    out->cqe.opcode = cqe->opcode;
    out->cqe.status = cqe->status;
    out->cqe.substatus = cqe->substatus;
    out->cqe.entry_idx = cqe->entry_idx;
    out->cqe.byte_cnt = cqe->byte_cnt;
    out->cqe.rmt_idx = cqe->rmt_idx;
    out->cqe.tpn = cqe->tpn;
    out->cqe.user_data_l = cqe->user_data_l;
    out->cqe.user_data_h = cqe->user_data_h;
    out->cqe.rmt_eid[0] = cqe->rmt_eid[0];
    out->cqe.rmt_eid[1] = cqe->rmt_eid[1];
    out->cqe.rmt_eid[2] = cqe->rmt_eid[2];
    out->cqe.rmt_eid[3] = cqe->rmt_eid[3];
    return true;
}

ACLSHMEM_DEVICE bool aclshmemi_udma_exception_report_read_wq(
    __gm__ aclshmemi_udma_wq_ctx_t* wq, uint64_t entry_size, __gm__ aclshmemi_udma_exception_report_entry_t* out)
{
    if (entry_size < sizeof(aclshmemi_udma_wq_ctx_t)) {
        return false;
    }
    aclshmemi_udma_exception_report_dcci_cachelines(
        reinterpret_cast<__gm__ uint8_t*>(wq), sizeof(aclshmemi_udma_wq_ctx_t));

    out->wq.wqn = wq->wqn;
    out->wq.buf_addr = wq->buf_addr;
    out->wq.wqe_size = wq->wqe_size;
    out->wq.depth = wq->depth;
    out->wq.head = wq->head;
    out->wq.tail = wq->tail;
    out->wq.db_mode = static_cast<int32_t>(wq->db_mode);
    out->wq.db_addr = wq->db_addr;
    out->wq.sl = wq->sl;
    out->wq.wqe_cnt = wq->wqe_cnt;
    out->wq.amo_addr = wq->amo_addr;
    return true;
}

ACLSHMEM_DEVICE bool aclshmemi_udma_exception_report_read_cq(
    __gm__ aclshmemi_udma_cq_ctx_t* cq, uint64_t entry_size, __gm__ aclshmemi_udma_exception_report_entry_t* out)
{
    if (entry_size < sizeof(aclshmemi_udma_cq_ctx_t)) {
        return false;
    }
    aclshmemi_udma_exception_report_dcci_cachelines(
        reinterpret_cast<__gm__ uint8_t*>(cq), sizeof(aclshmemi_udma_cq_ctx_t));

    out->cq.cqn = cq->cqn;
    out->cq.buf_addr = cq->buf_addr;
    out->cq.cqe_size = cq->cqe_size;
    out->cq.depth = cq->depth;
    out->cq.head = cq->head;
    out->cq.tail = cq->tail;
    out->cq.db_mode = static_cast<int32_t>(cq->db_mode);
    out->cq.db_addr = cq->db_addr;
    return true;
}

ACLSHMEM_DEVICE bool aclshmemi_udma_exception_report_read_wqe_raw(
    __gm__ uint8_t* wqe, uint64_t entry_size, __gm__ aclshmemi_udma_exception_report_entry_t* out)
{
    if (entry_size == 0 || entry_size > ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_MAX_RAW_SIZE) {
        return false;
    }
    aclshmemi_udma_exception_report_dcci_cachelines(wqe, entry_size);

    out->wqe_raw.addr = reinterpret_cast<uint64_t>(wqe);
    out->wqe_raw.size = static_cast<uint32_t>(entry_size);
    for (uint32_t i = 0; i < entry_size; ++i) {
        out->wqe_raw.data[i] = wqe[i];
    }
    return true;
}

ACLSHMEM_GLOBAL void aclshmemi_udma_exception_report_read_entry_kernel(
    uint32_t entry_type, GM_ADDR entry_addr, uint64_t entry_size, GM_ADDR out_addr)
{
    auto* entry_out = reinterpret_cast<__gm__ aclshmemi_udma_exception_report_entry_t*>(out_addr);
    if (entry_out == nullptr) {
        return;
    }
    entry_out->ret = ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_INVALID_PARAM;
    entry_out->entry_type = entry_type;
    if (entry_addr == 0 || entry_size == 0) {
        return;
    }

    bool success = false;
    if (entry_type == ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_CQE) {
        success = aclshmemi_udma_exception_report_read_cqe(
            reinterpret_cast<__gm__ aclshmemi_jfc_cqe_ctx_t*>(entry_addr), entry_size, entry_out);
    } else if (entry_type == ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_WQ) {
        success = aclshmemi_udma_exception_report_read_wq(
            reinterpret_cast<__gm__ aclshmemi_udma_wq_ctx_t*>(entry_addr), entry_size, entry_out);
    } else if (entry_type == ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_CQ) {
        success = aclshmemi_udma_exception_report_read_cq(
            reinterpret_cast<__gm__ aclshmemi_udma_cq_ctx_t*>(entry_addr), entry_size, entry_out);
    } else if (entry_type == ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_WQE_RAW) {
        success = aclshmemi_udma_exception_report_read_wqe_raw(
            reinterpret_cast<__gm__ uint8_t*>(entry_addr), entry_size, entry_out);
    } else {
        return;
    }
    if (success) {
        entry_out->ret = ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_SUCCESS;
    }
}

int32_t aclshmemi_udma_exception_report_read_entry_on_stream(
    uint32_t entry_type, uint64_t entry_addr, uint64_t entry_size, aclshmemi_udma_exception_report_entry_t* out,
    aclrtStream stream)
{
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    aclshmemi_udma_exception_report_read_entry_kernel<<<1, nullptr, stream>>>(
        entry_type, (GM_ADDR)entry_addr, entry_size, (GM_ADDR)out);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
