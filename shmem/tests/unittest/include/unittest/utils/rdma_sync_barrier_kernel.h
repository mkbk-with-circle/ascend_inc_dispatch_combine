/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ROCE_SYNC_BARRIER_KERNEL_H
#define ROCE_SYNC_BARRIER_KERNEL_H

void roce_sync_all_data_do(void *stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size, int message_length);
void roce_sync_all_with_buf_data_do(void *stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size, int message_length);
void roce_barrier_all_data_do(void *stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size, int message_length);
void roce_barrier_all_with_buf_data_do(void *stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size, int message_length);
void roce_sync_team_data_do(void *stream, uint64_t config, uint8_t *addr, int rank_id,
    int rank_size, int message_length, aclshmem_team_t team_id);
void roce_sync_team_with_buf_data_do(void *stream, uint64_t config, uint8_t *addr, int rank_id,
    int rank_size, int message_length, aclshmem_team_t team_id);
void roce_barrier_team_data_do(void *stream, uint64_t config, uint8_t *addr, int rank_id,
    int rank_size, int message_length, aclshmem_team_t team_id);
void roce_barrier_team_with_buf_data_do(void *stream, uint64_t config, uint8_t *addr, int rank_id,
    int rank_size, int message_length, aclshmem_team_t team_id);
void roce_quiet_data_test_do(void *stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size, int message_length);

#endif
