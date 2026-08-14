/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_EXCEPTION_REPORT_H
#define ACLSHMEM_EXCEPTION_REPORT_H

#include <stdbool.h>
#include <cstdint>
#include "host/utils/shmem_host_exception.h"
#include "host_device/shmem_common_types.h"

enum aclshmemi_exception_report_mode_t : uint32_t {
    ACLSHMEMI_EXCEPTION_REPORT_MODE_OFF = 0,
    ACLSHMEMI_EXCEPTION_REPORT_MODE_INFO,
    ACLSHMEMI_EXCEPTION_REPORT_MODE_DEBUG,
};

struct aclshmemi_exception_report_context_t {
    bool explicit_configured{false};
    uint32_t enabled_engines{0};
    uint32_t mode{ACLSHMEMI_EXCEPTION_REPORT_MODE_OFF};
    aclshmemx_exception_info_callback_t user_callback{nullptr};
    uint32_t registration_state{0};
    uint64_t reported_seq{0};
    uint64_t reported_lost_snapshot_count{0};
    uint64_t udma_info_address{0};
    uint64_t seq{0};
    uint32_t task_id{0};
    uint32_t stream_id{0};
    uint32_t thread_id{0};
    uint32_t device_id{0};
    uint32_t error_code{0};
    uint64_t lost_snapshot_count{0};
};

int aclshmemi_exception_report_apply_deferred_config(data_op_engine_type_t enabled_engines);
void aclshmemi_exception_report_finalize(void);
bool aclshmemi_exception_report_pending(void);
int aclshmemi_exception_report_dump(void);
void aclshmemi_exception_report_set_udma_info_address(uint64_t udma_info_address);
uint64_t aclshmemi_exception_report_udma_info_address(void);
bool aclshmemi_exception_report_record_snapshot(
    uint32_t task_id, uint32_t stream_id, uint32_t thread_id, uint32_t device_id, uint32_t error_code);
void aclshmemi_exception_report_save_context(aclshmemi_exception_report_context_t& context);
void aclshmemi_exception_report_restore_context(const aclshmemi_exception_report_context_t& context);

#endif // ACLSHMEM_EXCEPTION_REPORT_H
