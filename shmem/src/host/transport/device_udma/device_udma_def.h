/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBRID_DEVICE_UDMA_DEF_H
#define MF_HYBRID_DEVICE_UDMA_DEF_H

#include <cstdint>

namespace shm {
namespace transport {
namespace device {

// NOTE: The structs below mirror the data-plane layout defined in
// src/device/gm2gm/engine/shmemi_device_udma.h. The control plane fills these
// host-side structs and copies them to device memory, so the field order and
// types MUST stay byte-for-byte identical to the data-plane definitions.
struct aclshmemi_ubmem_info_t {
    bool token_value_valid;      // token_en 表示是否使能token
    uint32_t rmt_jetty_type : 2; // 表示远端jetty的类型
    uint8_t target_hint;         // jettygrp场景使用
    uint32_t tpn;                // 对应着tp_id 区分传输层是简易传输层还是完整传输层
    uint32_t tid;                // 对应着SQE的rmt_jetty_or_seg_id，来源是udma_seg->tid;
    uint32_t rmt_token_value;    // 对应着SQE的rmt_token_value，来源是udma_seg->token_value.token;
    uint32_t len;
    uint64_t addr; // 来源urma_sge的addr，对应SQE的rmt_addr_l_or_token_id，rmt_addr_h_or_token_value
    uint64_t eid_addr;
};

enum class aclshmemi_udma_db_mode_t : int32_t { INVALID_DB = -1, HW_DB = 0, SW_DB };

struct aclshmemi_udma_wq_ctx_t {
    uint32_t wqn;      // work queue number
    uint64_t buf_addr; // start address of ring buffer
    uint32_t wqe_size; // size in bytes of each WQE
    uint32_t depth;    // depth of ring buffer
    uint32_t head;     // work queue head (Producer Index) address
    uint32_t tail;     // work queue tail (Consumer Index) address
    aclshmemi_udma_db_mode_t db_mode;
    uint64_t db_addr;  // doorbell address
    uint32_t sl;       // service level
    uint32_t wqe_cnt;  // wqe count
    uint64_t amo_addr; // amo address to store fetch data
};

struct aclshmemi_udma_cq_ctx_t {
    uint32_t cqn;      // completion queue number
    uint64_t buf_addr; // start address of ring buffer
    uint32_t cqe_size; // size in bytes of each CQE
    uint32_t depth;    // depth of ring buffer
    uint32_t head;     // completion queue head (Producer Index)
    uint32_t tail;     // completion queue tail (Consumer Index)
    aclshmemi_udma_db_mode_t db_mode;
    uint64_t db_addr; // doorbell address
};

// Queue / memory-region base-address table. All arrays are indexed by "slot"; the slot count
// differs per path (see aclshmemi_aiv_udma_info_t below).
struct aclshmemi_udma_qp_table_t {
    uint64_t sq_ptr;  // send queue address array,              [slot_count][qp_num]
    uint64_t rq_ptr;  // receive queue address array,           [slot_count][qp_num]
    uint64_t scq_ptr; // send completion queue address array,   [slot_count][qp_num]
    uint64_t rcq_ptr; // receive completion queue address array,[slot_count][qp_num]
    uint64_t mem_ptr; // memory region array,                   [slot_count]
};

struct aclshmemi_aiv_udma_info_t {
    uint32_t qp_num; // number of QP per connection
    // Direct and relay paths are mutually exclusive per build (ACLSHMEM_RELAY_SUPPORT), so they
    // share storage.
    //   * direct table: slot_count == PE_NUM,           slot == pe
    //   * relay table:  slot_count == PE_NUM*rank_count, slot == pe * rank_count + relay_pe
    union {
        aclshmemi_udma_qp_table_t direct;
        aclshmemi_udma_qp_table_t relay;
    };
};

} // namespace device
} // namespace transport
} // namespace shm
#endif // MF_HYBRID_DEVICE_UDMA_DEF_H
