/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEMI_DEVICE_UDMA_EXCEPTION_REPORT_KERNEL_H
#define ACLSHMEMI_DEVICE_UDMA_EXCEPTION_REPORT_KERNEL_H

#include <cstdint>

#include "acl/acl_rt.h"

constexpr uint32_t ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_CQE = 0U;
constexpr uint32_t ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_WQ = 1U;
constexpr uint32_t ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_CQ = 2U;
constexpr uint32_t ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_WQE_RAW = 3U;
constexpr uint32_t ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_SUCCESS = 0xAC00U;
constexpr uint32_t ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_INVALID_PARAM = 0xAC01U;
constexpr uint32_t ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_MAX_RAW_SIZE = 256U;

struct aclshmemi_udma_exception_report_cqe_t {
    uint32_t owner;
    uint32_t opcode;
    uint32_t status;
    uint32_t substatus;
    uint32_t entry_idx;
    uint32_t byte_cnt;
    uint32_t rmt_idx;
    uint32_t tpn;
    uint32_t user_data_l;
    uint32_t user_data_h;
    uint32_t rmt_eid[4];
};

struct aclshmemi_udma_exception_report_wq_t {
    uint32_t wqn;
    uint64_t buf_addr;
    uint32_t wqe_size;
    uint32_t depth;
    uint32_t head;
    uint32_t tail;
    int32_t db_mode;
    uint64_t db_addr;
    uint32_t sl;
    uint32_t wqe_cnt;
    uint64_t amo_addr;
};

struct aclshmemi_udma_exception_report_cq_t {
    uint32_t cqn;
    uint64_t buf_addr;
    uint32_t cqe_size;
    uint32_t depth;
    uint32_t head;
    uint32_t tail;
    int32_t db_mode;
    uint64_t db_addr;
};

struct aclshmemi_udma_exception_report_wqe_raw_t {
    uint64_t addr;
    uint32_t size;
    uint8_t data[ACLSHMEMI_UDMA_EXCEPTION_REPORT_ENTRY_MAX_RAW_SIZE];
};

struct aclshmemi_udma_exception_report_entry_t {
    uint32_t ret;
    uint32_t entry_type;
    aclshmemi_udma_exception_report_cqe_t cqe;
    aclshmemi_udma_exception_report_wq_t wq;
    aclshmemi_udma_exception_report_cq_t cq;
    aclshmemi_udma_exception_report_wqe_raw_t wqe_raw;
};

int32_t aclshmemi_udma_exception_report_read_entry_on_stream(
    uint32_t entry_type, uint64_t entry_addr, uint64_t entry_size, aclshmemi_udma_exception_report_entry_t* out,
    aclrtStream stream = nullptr);

#endif // ACLSHMEMI_DEVICE_UDMA_EXCEPTION_REPORT_KERNEL_H
