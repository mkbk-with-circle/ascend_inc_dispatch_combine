/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_SIMT_UB2GM_MTE_HPP
#define SHMEM_DEVICE_SIMT_UB2GM_MTE_HPP

#include "shmemi_device_simt_common.h"
#include "device_simt/gm2gm/engine/shmem_device_simt_mte.h"

namespace simt {

template <typename T, thread_group_t scope>
__simt_callee__ inline void aclshmemi_mte_get_nbi(__ubuf__ T *dst, __gm__ T *src, size_t elem_size, int pe)
{
    aclshmemi_threadgroup_sync<scope>();
    size_t myIndex = aclshmemi_thread_id_in_threadgroup<scope>();
    size_t groupSize = aclshmemi_threadgroup_size<scope>();

    __gm__ T* srcSym = (__gm__ T*)aclshmem_ptr(src, pe);

    for (size_t i = myIndex; i < elem_size; i += groupSize) {
        dst[i] = srcSym[i];
    }
    aclshmemi_threadgroup_sync<scope>();
}

template <typename T, thread_group_t scope>
__simt_callee__ inline void aclshmemi_mte_put_nbi(__gm__ T *dst, __ubuf__ T *src, size_t elem_size, int pe)
{
    aclshmemi_threadgroup_sync<scope>();

    size_t myIndex = aclshmemi_thread_id_in_threadgroup<scope>();
    size_t groupSize = aclshmemi_threadgroup_size<scope>();

    __gm__ T* dstSym = (__gm__ T*)aclshmem_ptr(dst, pe);

    for (size_t i = myIndex; i < elem_size; i += groupSize) {
        dstSym[i] = src[i];
    }
    aclshmemi_threadgroup_sync<scope>();
}

template <typename T>
__simt_callee__ inline void aclshmemx_mte_get_nbi(__ubuf__ T *dst, __gm__ T *src, size_t elem_size, int pe)
{
    aclshmemi_mte_get_nbi<T, ACLSHMEMI_THREADGROUP_THREAD>(dst, src, elem_size, pe);
}

template <typename T>
__simt_callee__ inline void aclshmemx_mte_get_nbi_block(__ubuf__ T *dst, __gm__ T *src, size_t elem_size, int pe)
{
    aclshmemi_mte_get_nbi<T, ACLSHMEMI_THREADGROUP_BLOCK>(dst, src, elem_size, pe);
}

template <typename T>
__simt_callee__ inline void aclshmemx_mte_get_nbi_warp(__ubuf__ T *dst, __gm__ T *src, size_t elem_size, int pe)
{
    aclshmemi_mte_get_nbi<T, ACLSHMEMI_THREADGROUP_WARP>(dst, src, elem_size, pe);
}

template <typename T>
__simt_callee__ inline void aclshmemx_mte_put_nbi(__gm__ T *dst, __ubuf__ T *src, size_t elem_size, int pe)
{
    aclshmemi_mte_put_nbi<T, ACLSHMEMI_THREADGROUP_THREAD>(dst, src, elem_size, pe);
}

template <typename T>
__simt_callee__ inline void aclshmemx_mte_put_nbi_block(__gm__ T *dst, __ubuf__ T *src, size_t elem_size, int pe)
{
    aclshmemi_mte_put_nbi<T, ACLSHMEMI_THREADGROUP_BLOCK>(dst, src, elem_size, pe);
}

template <typename T>
__simt_callee__ inline void aclshmemx_mte_put_nbi_warp(__gm__ T *dst, __ubuf__ T *src, size_t elem_size, int pe)
{
    aclshmemi_mte_put_nbi<T, ACLSHMEMI_THREADGROUP_WARP>(dst, src, elem_size, pe);
}

} // namespace simt

#endif // !SHMEM_DEVICE_SIMT_MTE_HPP
