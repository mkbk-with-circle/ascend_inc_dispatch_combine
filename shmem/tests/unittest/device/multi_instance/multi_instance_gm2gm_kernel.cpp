/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"

#define MULTI_INSTANCE

#include "shmem.h"
#include "unittest/utils/func_type.h"

const int length  = 256;
constexpr int32_t UNIT_COPY_SIZE = 64 * 1024;

template<typename T>
__attribute__((always_inline)) inline __aicore__ void gm2gm(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> ub_buff, size_t data_size)
{
    int32_t step = 0;

    int elem_size = data_size / sizeof(T);
    int elem_unit = UNIT_COPY_SIZE / sizeof(T);
    while (elem_size >= elem_unit) {
        AscendC::DataCopy(ub_buff, src[elem_unit * step], elem_unit);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        AscendC::DataCopy(dst[elem_unit * step], ub_buff, elem_unit);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        step += 1;
        elem_size -= elem_unit;
    }
    if (elem_size <= 0) {
        return;
    }

    // 尾块处理
    AscendC::DataCopy(ub_buff, src[elem_unit * step], elem_size);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    AscendC::DataCopy(dst[elem_unit * step], ub_buff, elem_size);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}


#define KERNEL_MULTI_INSTANCE_PUT(NAME, TYPE)                                                                           \
    class kernel_multi_instance_##NAME##_put {                                                                          \
    public:                                                                                                             \
        __aicore__ inline kernel_multi_instance_##NAME##_put()                                                          \
        {                                                                                                               \
        }                                                                                                               \
        __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst, GM_ADDR gva_list, int count)                              \
        {                                                                                                               \
            src_gm = (__gm__ TYPE *)src;                                                                                \
            dst_gm = (__gm__ TYPE *)dst;                                                                                \
                                                                                                                        \
            /* Set GM Buffer */                                                                                         \
            src_tensor.SetGlobalBuffer(src_gm);                                                                         \
            dst_tensor.SetGlobalBuffer(dst_gm);                                                                         \
                                                                                                                        \
            /* Get Shmem addrs */                                                                                       \
            shmem_addr_list_ = (__gm__ uint64_t *)gva_list;                                                             \
                                                                                                                        \
            /* 1x4096 Bytes Buffer */                                                                                   \
            pipe.InitBuffer(buf_queue, 1, UNIT_COPY_SIZE);                                                              \
                                                                                                                        \
            /* Get shmem instance count */                                                                              \
            instance_count_ = count;                                                                                    \
        }                                                                                                               \
        __aicore__ inline void Process(uint64_t config)                                                                 \
        {                                                                                                               \
            util_set_ffts_config(config);                                                                               \
            AscendC::LocalTensor<TYPE> buf_tensor = buf_queue.AllocTensor<TYPE>();                                      \
            uintptr_t addr                        = static_cast<uintptr_t>(buf_tensor.address_.bufferAddr);             \
            __ubuf__ TYPE *buf                    = (__ubuf__ TYPE *)addr;                                              \
            if ASCEND_IS_AIV {                                                                                          \
                for (int i = 0; i < instance_count_; i++) {                                                             \
                    /* Set multi-instance context. */                                                                   \
                    uint64_t shmem_instance_offset = 1;                                                                 \
                    aclshmemx_instance_ctx_set(shmem_instance_offset + i);                                              \
                                                                                                                        \
                    __gm__ TYPE *gva_gm = (__gm__ TYPE *)(*(shmem_addr_list_ + i));                                     \
                    AscendC::GlobalTensor<TYPE> gva_tensor;                                                             \
                    gva_tensor.SetGlobalBuffer(gva_gm);                                                                 \
                    /* Copy from GM TO SHMEM GM Through DataCopy API */                                                 \
                    gm2gm(gva_tensor, src_tensor[i * length], buf_tensor, length * sizeof(TYPE));                       \
                    pipe_barrier(PIPE_ALL);                                                                             \
                    /* Copy from GM TO GM Through DataCopy API */                                                       \
                    gm2gm(dst_tensor[i * length], src_tensor[i * length], buf_tensor, length * sizeof(TYPE));           \
                    pipe_barrier(PIPE_ALL);                                                                             \
                                                                                                                        \
                    aclshmemx_barrier_all_vec();                                                                        \
                }                                                                                                       \
                buf_queue.FreeTensor(buf_tensor);                                                                       \
            }                                                                                                           \
        }                                                                                                               \
                                                                                                                        \
    private:                                                                                                            \
        AscendC::TPipe pipe;                                                                                            \
        AscendC::TQue<AscendC::TPosition::VECIN, 2> buf_queue;                                                          \
                                                                                                                        \
        __gm__ TYPE *gva_gm;                                                                                            \
        __gm__ TYPE *src_gm;                                                                                            \
        __gm__ TYPE *dst_gm;                                                                                            \
        AscendC::GlobalTensor<TYPE> src_tensor, dst_tensor;                                                             \
                                                                                                                        \
        int64_t instance_count_ = 0;                                                                                    \
        __gm__ uint64_t *shmem_addr_list_;                                                                              \
    }

ACLSHMEM_FUNC_TYPE_KERNEL(KERNEL_MULTI_INSTANCE_PUT);

#define MULTI_INSTANCE_PUT_TEST(NAME, TYPE)                                                                                 \
    extern "C" __global__ __aicore__ void multi_instance_put_##NAME##_test(                                                 \
                        GM_ADDR src, GM_ADDR dst, GM_ADDR shmem_addrs_device, int inst_count, uint64_t config)              \
    {                                                                                                                       \
        kernel_multi_instance_##NAME##_put op;                                                                              \
        op.Init(src, dst, shmem_addrs_device, inst_count);                                                                  \
        op.Process(config);                                                                                                 \
    }

ACLSHMEM_FUNC_TYPE_KERNEL(MULTI_INSTANCE_PUT_TEST);

#define TEST_PUT(NAME, TYPE)                                                                                                \
    void multi_instance_##NAME##_test_put(uint32_t block_dim, void *stream,                                                 \
                                uint64_t config, uint8_t *src, uint8_t *dst, uint8_t *shmem_addrs_device, int inst_count)   \
    {                                                                                                                       \
        multi_instance_put_##NAME##_test<<<block_dim, nullptr, stream>>>(src, dst, shmem_addrs_device, inst_count, config); \
    }

ACLSHMEM_FUNC_TYPE_KERNEL(TEST_PUT);

#undef MULTI_INSTANCE