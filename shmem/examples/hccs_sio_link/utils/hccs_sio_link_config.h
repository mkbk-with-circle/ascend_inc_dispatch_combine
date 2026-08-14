/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCCS_SIO_LINK_CONFIG_H
#define HCCS_SIO_LINK_CONFIG_H

#include <cstdint>
#include <cstddef>

constexpr uint32_t HCCS_SIO_BLOCK_DIM = 32;   // total kernel block count
constexpr int HCCS_SIO_UB_SIZE_KB = 16;       // unified buffer size per block (KB)

constexpr int32_t HCCS_SIO_RATIO_NUM = 3;     // SIO ratio numerator (SIO elements = total * NUM / DEN)
constexpr int32_t HCCS_SIO_RATIO_DEN = 5;     // SIO ratio denominator

inline int32_t calc_sio_block_count(int32_t total_blocks)
{
    int32_t count = total_blocks * HCCS_SIO_RATIO_NUM / HCCS_SIO_RATIO_DEN;
    if (count == 0)
        count = 1;
    if (count > total_blocks)
        count = total_blocks;
    return count;
}

inline size_t calc_sio_elements(size_t total_elements)
{
    return total_elements * HCCS_SIO_RATIO_NUM / HCCS_SIO_RATIO_DEN;
}

// perf test: data size range in log2(bytes), step in log2
// e.g. MIN=12 MAX=28 STEP=1 -> 4KB,8KB,16KB,...,256MB
constexpr int32_t HCCS_SIO_PERF_MIN_LOG2_BYTES = 4;
constexpr int32_t HCCS_SIO_PERF_MAX_LOG2_BYTES = 20;
constexpr int32_t HCCS_SIO_PERF_STEP_LOG2 = 1;    // log2 step: 1=doubles each iteration
constexpr int HCCS_SIO_PERF_WARMUP = 100;          // warmup iterations before measurement
constexpr int HCCS_SIO_PERF_LOOP_COUNT = 1000;     // measurement loop count
constexpr bool HCCS_SIO_PERF_IS_UNILATERAL = true; // true: only prof_pe runs, false: all PEs run (bidirectional)

#endif
