/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PERF_COMMON_TYPES_H
#define PERF_COMMON_TYPES_H

namespace perftest {

constexpr int PERFTEST_WARMUP_ITERS = 100;

typedef enum {
    DATA_TYPE_FLOAT,
    DATA_TYPE_INT8,
    DATA_TYPE_INT16,
    DATA_TYPE_INT32,
    DATA_TYPE_INT64,
    DATA_TYPE_UINT8,
    DATA_TYPE_UINT16,
    DATA_TYPE_UINT32,
    DATA_TYPE_UINT64,
    DATA_TYPE_CHAR
} perf_data_type_t;

typedef enum {
    TEST_MODE_ASCENDC_PUT,
    TEST_MODE_ASCENDC_GET,
    TEST_MODE_UB2GM_LOCAL,
    TEST_MODE_UB2GM_REMOTE,
    TEST_MODE_GM2UB_LOCAL,
    TEST_MODE_GM2UB_REMOTE
} ascendc_mode_t;

typedef enum { TEST_MODE_MTE_PUT, TEST_MODE_BI_PUT, TEST_MODE_MTE_GET, TEST_MODE_BI_GET } mte_mode_t;

typedef enum {
    TEST_MODE_UDMA_PUT,
    TEST_MODE_UDMA_BI_PUT,
    TEST_MODE_UDMA_GET,
    TEST_MODE_UDMA_BI_GET,
    TEST_MODE_UDMA_PUT_SIGNAL
} udma_mode_t;

typedef enum {
    TEST_MODE_RDMA_PUT,
    TEST_MODE_RDMA_BI_PUT,
    TEST_MODE_RDMA_GET,
    TEST_MODE_RDMA_BI_GET,
    TEST_MODE_RDMA_INVALID
} rdma_mode_t;

typedef enum { PERF_METRIC_BW, PERF_METRIC_LAT, PERF_METRIC_INVALID } perf_metric_t;

} // namespace perftest

#endif
