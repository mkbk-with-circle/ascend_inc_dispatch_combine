/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEM_DEVICE_UDMA_H
#define SHMEM_DEVICE_UDMA_H

#include <stdint.h>

#include "kernel_operator.h"
#include "device/shmem_def.h"
#include "host_device/shmem_common_types.h"

/**
 * @anchor udma_submit_action_contract
 * @par UDMA submit action contract
 * UDMA defer/submit overloads add operations to a caller-managed batch through
 * @ref aclshmemx_defer_t and @ref aclshmemx_submit_t.
 * - Action overloads support PIPE_MTE3 only.
 * - Use one initialized @ref aclshmemx_submit_state_t for one active batch.
 * - Every call in a batch must use the same operation kind, state, PE arguments, buf base, and sync_id.
 * - n is the total number of operations in the batch, including the final submit call. The batch must be
 *   smaller than the SQ ring depth.
 * - @p buf is caller-provided UB scratch. Its capacity must be at least 64 * n bytes and must not exceed the
 *   caller's available UB capacity; split larger batches into multiple submit batches.
 * - The submit call contributes one operation to n, submits all pending operations, and resets pending_count
 *   after a successful submit.
 * - On a submit failure, the calling device kernel is aborted and the submit state is not reset.
 * - After submit, call @ref aclshmemx_udma_quiet for the target PE before consuming Get destinations or reusing
 *   Put sources whose contents must remain stable.
 */

/**
 * @name aclshmemx_udma_get_nbi
 * @brief Asynchronously copy contiguous data from symmetric address on the specified PE to the local PE.
 *
 * @warning When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same PE are not
 *          supported.
 *
 * @details Common semantics:
 * - @p dst is a local destination address and @p src is a symmetric source address.
 * - @p elem_size is the number of T elements; one UDMA get request transfers at most 256 MB.
 * - After submitting asynchronous requests, call @ref aclshmemx_udma_quiet before reading @p dst.
 *
 * @{
 */
/**
 * @brief Pointer overload with immediate submission.
 *
 * @tparam T                  Element type of the transfer.
 * @tparam WQE_PIPE           Pipe used to publish the operation. PIPE_MTE3 stages one operation in @p buf;
 *                            PIPE_S writes directly and ignores @p buf / @p sync_id.
 * @param dst               [in] Local destination address.
 * @param src               [in] Symmetric source address on the target PE specified by pe.
 * @param buf               [in] Local UB scratch used when WQE_PIPE == PIPE_MTE3; ignored when WQE_PIPE == PIPE_S.
 * @param elem_size         [in] Number of elements transferred by this Get operation.
 * @param pe                [in] Target PE that owns src.
 * @param sync_id           [in] Hardware event ID used by the PIPE_MTE3 path. Ignored when WQE_PIPE == PIPE_S.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id = 0);

/**
 * @brief Adds the current nonblocking UDMA Get operation to a batch and keeps the batch pending.
 *
 * @details See @ref udma_submit_action_contract. Finish the batch with an aclshmemx_submit_t action.
 *
 * @tparam WQE_PIPE           Must be PIPE_MTE3 for this action overload.
 * @param dst               [in] Local destination address.
 * @param src               [in] Symmetric source address on the target PE specified by pe.
 * @param buf               [in] Local UB workspace shared by all calls in this batch.
 * @param elem_size         [in] Number of elements transferred by this Get operation.
 * @param pe                [in] Target PE that owns src.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_defer_t action referencing the batch's initialized
 *                          aclshmemx_submit_state_t.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_defer_t action);

/**
 * @brief Adds the current nonblocking UDMA Get operation and submits all operations in the batch.
 *
 * @details See @ref udma_submit_action_contract. After submit, call aclshmemx_udma_quiet(pe) before reading
 *          or reusing dst.
 *
 * @tparam WQE_PIPE           Must be PIPE_MTE3 for this action overload.
 * @param dst               [in] Local destination address.
 * @param src               [in] Symmetric source address on the target PE specified by pe.
 * @param buf               [in] Local UB workspace shared by all calls in this batch.
 * @param elem_size         [in] Number of elements transferred by this Get operation.
 * @param pe                [in] Target PE that owns src.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_submit_t action referencing the same aclshmemx_submit_state_t.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_submit_t action);

/**
 * @brief GlobalTensor overload with immediate submission.
 *
 * @tparam T                  Element type of the transfer.
 * @tparam WQE_PIPE           Pipe used to publish the operation. PIPE_MTE3 stages one operation in @p buf;
 *                            PIPE_S writes directly and ignores @p buf / @p sync_id.
 * @param dst               [in] Local GlobalTensor for the destination data.
 * @param src               [in] Remote symmetric GlobalTensor for the source data on the target PE specified by pe.
 * @param buf               [in] Local UB LocalTensor used when WQE_PIPE == PIPE_MTE3; ignored when WQE_PIPE == PIPE_S.
 * @param elem_size         [in] Number of elements transferred by this Get operation.
 * @param pe                [in] Target PE that owns src.
 * @param sync_id           [in] Hardware event ID used by the PIPE_MTE3 path. Ignored when WQE_PIPE == PIPE_S.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t sync_id = 0);

/**
 * @brief Adds the current nonblocking UDMA Get operation to a batch and keeps the batch pending.
 *
 * @details See @ref udma_submit_action_contract. Finish the batch with an aclshmemx_submit_t action.
 *
 * @tparam WQE_PIPE           Must be PIPE_MTE3 for this action overload.
 * @param dst               [in] Local GlobalTensor for the destination data.
 * @param src               [in] Remote symmetric GlobalTensor for the source data on the target PE specified by pe.
 * @param buf               [in] Local UB LocalTensor workspace shared by all calls in this batch.
 * @param elem_size         [in] Number of elements transferred by this Get operation.
 * @param pe                [in] Target PE that owns src.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_defer_t action referencing the batch's initialized
 *                          aclshmemx_submit_state_t.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t sync_id, aclshmemx_defer_t action);

/**
 * @brief Adds the current nonblocking UDMA Get operation and submits all operations in the batch.
 *
 * @details See @ref udma_submit_action_contract. After submit, call aclshmemx_udma_quiet(pe) before reading
 *          or reusing dst.
 *
 * @tparam WQE_PIPE           Must be PIPE_MTE3 for this action overload.
 * @param dst               [in] Local GlobalTensor for the destination data.
 * @param src               [in] Remote symmetric GlobalTensor for the source data on the target PE specified by pe.
 * @param buf               [in] Local UB LocalTensor workspace shared by all calls in this batch.
 * @param elem_size         [in] Number of elements transferred by this Get operation.
 * @param pe                [in] Target PE that owns src.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_submit_t action referencing the same aclshmemx_submit_state_t.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t sync_id, aclshmemx_submit_t action);
/** @} */

/**
 * @name aclshmemx_udma_put_nbi
 * @brief Asynchronously copy contiguous data on the local PE to symmetric address on the specified PE.
 *
 * @warning When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same PE are not
 *          supported.
 *
 * @details Common semantics:
 * - @p dst is a symmetric destination address and @p src is a local source address.
 * - @p elem_size is the number of T elements; one UDMA put request transfers at most 256 MB.
 * - After submitting asynchronous requests, call @ref aclshmemx_udma_quiet before reading the destination data or
 *   reusing buffers whose contents must remain stable.
 *
 * @{
 */
/**
 * @brief Pointer overload with immediate submission.
 *
 * @tparam T                  Element type of the transfer.
 * @tparam WQE_PIPE           Pipe used to publish the operation. PIPE_MTE3 stages one operation in @p buf;
 *                            PIPE_S writes directly and ignores @p buf / @p sync_id.
 * @param dst               [in] Symmetric destination address on the target PE specified by pe.
 * @param src               [in] Local source address.
 * @param buf               [in] Local UB scratch used when WQE_PIPE == PIPE_MTE3; ignored when WQE_PIPE == PIPE_S.
 * @param elem_size         [in] Number of elements transferred by this Put operation.
 * @param pe                [in] Target PE that owns dst.
 * @param sync_id           [in] Hardware event ID used by the PIPE_MTE3 path. Ignored when WQE_PIPE == PIPE_S.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id = 0);

/**
 * @brief Adds the current nonblocking UDMA Put operation to a batch and keeps the batch pending.
 *
 * @details See @ref udma_submit_action_contract. Finish the batch with an aclshmemx_submit_t action;
 *          src must remain valid and unchanged until quiet returns.
 *
 * @tparam WQE_PIPE           Must be PIPE_MTE3 for this action overload.
 * @param dst               [in] Symmetric destination address on the target PE specified by pe.
 * @param src               [in] Local source address.
 * @param buf               [in] Local UB workspace shared by all calls in this batch.
 * @param elem_size         [in] Number of elements transferred by this Put operation.
 * @param pe                [in] Target PE that owns dst.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_defer_t action referencing the batch's initialized
 *                          aclshmemx_submit_state_t.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_defer_t action);

/**
 * @brief Adds the current nonblocking UDMA Put operation and submits all operations in the batch.
 *
 * @details See @ref udma_submit_action_contract. After submit, call aclshmemx_udma_quiet(pe);
 *          src must remain valid and unchanged until it returns.
 *
 * @tparam WQE_PIPE           Must be PIPE_MTE3 for this action overload.
 * @param dst               [in] Symmetric destination address on the target PE specified by pe.
 * @param src               [in] Local source address.
 * @param buf               [in] Local UB workspace shared by all calls in this batch.
 * @param elem_size         [in] Number of elements transferred by this Put operation.
 * @param pe                [in] Target PE that owns dst.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_submit_t action referencing the same aclshmemx_submit_state_t.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id,
    aclshmemx_submit_t action);

/**
 * @brief GlobalTensor overload with immediate submission.
 *
 * @tparam T                  Element type of the transfer.
 * @tparam WQE_PIPE           Pipe used to publish the operation. PIPE_MTE3 stages one operation in @p buf;
 *                            PIPE_S writes directly and ignores @p buf / @p sync_id.
 * @param dst               [in] Remote symmetric GlobalTensor for the destination data on the target PE specified by
 * pe.
 * @param src               [in] Local GlobalTensor for the source data.
 * @param buf               [in] Local UB LocalTensor used when WQE_PIPE == PIPE_MTE3; ignored when WQE_PIPE == PIPE_S.
 * @param elem_size         [in] Number of elements transferred by this Put operation.
 * @param pe                [in] Target PE that owns dst.
 * @param sync_id           [in] Hardware event ID used by the PIPE_MTE3 path. Ignored when WQE_PIPE == PIPE_S.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief Adds the current nonblocking UDMA Put operation to a batch and keeps the batch pending.
 *
 * @details See @ref udma_submit_action_contract. Finish the batch with an aclshmemx_submit_t action;
 *          src must remain valid and unchanged until quiet returns.
 *
 * @tparam WQE_PIPE           Must be PIPE_MTE3 for this action overload.
 * @param dst               [in] Remote symmetric GlobalTensor for the destination data on the target PE specified by
 * pe.
 * @param src               [in] Local GlobalTensor for the source data.
 * @param buf               [in] Local UB LocalTensor workspace shared by all calls in this batch.
 * @param elem_size         [in] Number of elements transferred by this Put operation.
 * @param pe                [in] Target PE that owns dst.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_defer_t action referencing the batch's initialized
 *                          aclshmemx_submit_state_t.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t sync_id, aclshmemx_defer_t action);

/**
 * @brief Adds the current nonblocking UDMA Put operation and submits all operations in the batch.
 *
 * @details See @ref udma_submit_action_contract. After submit, call aclshmemx_udma_quiet(pe);
 *          src must remain valid and unchanged until it returns.
 *
 * @tparam WQE_PIPE           Must be PIPE_MTE3 for this action overload.
 * @param dst               [in] Remote symmetric GlobalTensor for the destination data on the target PE specified by
 * pe.
 * @param src               [in] Local GlobalTensor for the source data.
 * @param buf               [in] Local UB LocalTensor workspace shared by all calls in this batch.
 * @param elem_size         [in] Number of elements transferred by this Put operation.
 * @param pe                [in] Target PE that owns dst.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_submit_t action referencing the same aclshmemx_submit_state_t.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t sync_id, aclshmemx_submit_t action);
/** @} */

/**
 * @brief Direct-mode asynchronous UDMA Get on an explicitly selected QP.
 * @note qp_idx must be smaller than the UDMA QP count configured at initialization. An invalid index prints an
 *       error and aborts the calling kernel.
 * @note A single request transfers at most 256 MB. A normal return only means the WQE was submitted; call
 *       aclshmemx_udma_qp_quiet(pe, qp_idx) before reading @p dst.
 *
 * @tparam T                Element type of the transfer.
 * @tparam WQE_PIPE         Pipe used to publish the operation. PIPE_MTE3 stages one operation in @p buf;
 *                          PIPE_S writes directly and ignores @p buf and @p sync_id.
 * @param dst               [in] Local destination address.
 * @param src               [in] Symmetric source address on the target PE specified by pe.
 * @param buf               [in] Local UB scratch used when WQE_PIPE == PIPE_MTE3; it must provide at least
 *                          ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE bytes. Ignored when WQE_PIPE == PIPE_S.
 * @param elem_size         [in] Number of elements transferred by this Get operation.
 * @param pe                [in] Target PE that owns src.
 * @param qp_idx            [in] QP index selected for this operation. Must be in the configured QP range.
 * @param sync_id           [in] Hardware event ID used by the PIPE_MTE3 path. Ignored when WQE_PIPE == PIPE_S.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id = 0);

/**
 * @brief GlobalTensor/LocalTensor overload of @ref aclshmemx_udma_qp_get_nbi.
 * @note The QP range, 256 MB request limit, scratch capacity, and completion requirements are the same as the
 *       pointer overload. Call aclshmemx_udma_qp_quiet(pe, qp_idx) before reading @p dst.
 *
 * @tparam T                Element type of the transfer.
 * @tparam WQE_PIPE         Pipe used to publish the operation. PIPE_MTE3 stages one operation in @p buf;
 *                          PIPE_S writes directly and ignores @p buf and @p sync_id.
 * @param dst               [in] Local GlobalTensor for the destination data.
 * @param src               [in] Remote symmetric GlobalTensor for the source data on the target PE specified by pe.
 * @param buf               [in] Local UB LocalTensor used when WQE_PIPE == PIPE_MTE3; its backing buffer must provide
 *                          at least ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE bytes. Ignored when WQE_PIPE == PIPE_S.
 * @param elem_size         [in] Number of elements transferred by this Get operation.
 * @param pe                [in] Target PE that owns src.
 * @param qp_idx            [in] QP index selected for this operation. Must be in the configured QP range.
 * @param sync_id           [in] Hardware event ID used by the PIPE_MTE3 path. Ignored when WQE_PIPE == PIPE_S.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id = 0);

/**
 * @brief Direct-mode asynchronous UDMA Put on an explicitly selected QP.
 * @note qp_idx must be smaller than the UDMA QP count configured at initialization. An invalid index prints an
 *       error and aborts the calling kernel.
 * @note A single request transfers at most 256 MB. A normal return only means the WQE was submitted; call
 *       aclshmemx_udma_qp_quiet(pe, qp_idx) before reusing @p src or relying on remote visibility.
 *
 * @tparam T                Element type of the transfer.
 * @tparam WQE_PIPE         Pipe used to publish the operation. PIPE_MTE3 stages one operation in @p buf;
 *                          PIPE_S writes directly and ignores @p buf and @p sync_id.
 * @param dst               [in] Symmetric destination address on the target PE specified by pe.
 * @param src               [in] Local source address.
 * @param buf               [in] Local UB scratch used when WQE_PIPE == PIPE_MTE3; it must provide at least
 *                          ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE bytes. Ignored when WQE_PIPE == PIPE_S.
 * @param elem_size         [in] Number of elements transferred by this Put operation.
 * @param pe                [in] Target PE that owns dst.
 * @param qp_idx            [in] QP index selected for this operation. Must be in the configured QP range.
 * @param sync_id           [in] Hardware event ID used by the PIPE_MTE3 path. Ignored when WQE_PIPE == PIPE_S.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id = 0);

/**
 * @brief GlobalTensor/LocalTensor overload of @ref aclshmemx_udma_qp_put_nbi.
 * @note The QP range, 256 MB request limit, scratch capacity, and completion requirements are the same as the
 *       pointer overload. Call aclshmemx_udma_qp_quiet(pe, qp_idx) before reusing @p src or relying on remote
 *       visibility.
 *
 * @tparam T                Element type of the transfer.
 * @tparam WQE_PIPE         Pipe used to publish the operation. PIPE_MTE3 stages one operation in @p buf;
 *                          PIPE_S writes directly and ignores @p buf and @p sync_id.
 * @param dst               [in] Remote symmetric GlobalTensor for the destination data on the target PE specified by
 * pe.
 * @param src               [in] Local GlobalTensor for the source data.
 * @param buf               [in] Local UB LocalTensor used when WQE_PIPE == PIPE_MTE3; its backing buffer must provide
 *                          at least ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE bytes. Ignored when WQE_PIPE == PIPE_S.
 * @param elem_size         [in] Number of elements transferred by this Put operation.
 * @param pe                [in] Target PE that owns dst.
 * @param qp_idx            [in] QP index selected for this operation. Must be in the configured QP range.
 * @param sync_id           [in] Hardware event ID used by the PIPE_MTE3 path. Ignored when WQE_PIPE == PIPE_S.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t qp_idx, uint32_t sync_id = 0);

/**
 * @brief Asynchronous relay-mode UDMA Put. Copy contiguous data from local PE to symmetric address on
 *        pe, while egressing on the local source EID that reaches relay_pe. The fabric forwards
 *        the packet via relay_pe to pe. Use this to spread traffic across multiple physical
 *        links between the same pair of nodes for higher aggregate bandwidth.
 *        Requires building with ACLSHMEM_RELAY_SUPPORT=ON.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 *        pe are not supported.
 *
 * @note  Preconditions on the PE arguments (all must hold, where myPe is the calling PE and
 *        rankCount is the total number of PEs):
 *          - 0 <= pe < rankCount and 0 <= relay_pe < rankCount
 *          - pe != relay_pe
 *          - pe != myPe and relay_pe != myPe (self is not a valid actual/relay target)
 *        If any precondition is violated the call submits nothing and returns immediately; in a
 *        debug build it aborts the kernel instead. Because the return type is void, the caller
 *        cannot detect a skipped submission at runtime -- validate the arguments before calling.
 * @note  Completion semantics: this is a non-blocking (_nbi) call. A normal return only means the
 *        WQE was published to the send queue; it does NOT mean the transfer finished or that the
 *        data is visible on pe. Call aclshmemx_udma_quiet(pe) (or a higher-level barrier) before
 *        reading the result or reusing @p src to guarantee completion and remote visibility.
 * @note  A single UDMA put request transfers at most 256 MB (256 * 1024 * 1024 bytes). @p elem_size
 *        is the number of T elements, so the transfer size is @p elem_size * sizeof(T) bytes. Split
 *        larger transfers into multiple put requests whose byte size does not exceed 256 MB.
 *
 * @tparam T                  Element type of the transfer.
 * @tparam WQE_PIPE           Pipe used to publish the WQE to the SQ ring. See @ref aclshmemx_udma_put_nbi
 *                            for semantics. PIPE_MTE3 (default) stages the WQE in the caller-provided
 *                            UB scratch (see @p buf) and DataCopyPads it to HBM in one shot; PIPE_S
 *                            scalar-writes the SQE/SGE block directly to HBM and ignores @p buf /
 *                            @p sync_id. Other pipe values are not supported.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data (on pe).
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB. Used as WQE staging scratch when
 *                               WQE_PIPE == PIPE_MTE3 (must hold one full WQE block; 256 B is safe
 *                               for all current opcodes); ignored when WQE_PIPE == PIPE_S.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the actual destination. Must satisfy the preconditions above.
 * @param relay_pe          [in] PE whose port path is used to forward the packet. Must satisfy the
 *                               preconditions above.
 * @param sync_id           [in] Hardware event ID used by the MTE3->S sync after DataCopyPad in the
 *                               PIPE_MTE3 path. Ignored when WQE_PIPE == PIPE_S. Defaults to 0 for
 *                               backward compatibility with existing callers.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id = 0);

/**
 * @brief Adds the current nonblocking relay UDMA Put operation to a batch and keeps the batch pending.
 *
 * @details See @ref udma_submit_action_contract. Same PE-argument preconditions as the immediate relay Put
 *          overload. Finish the batch with an aclshmemx_submit_t action; src must remain valid and unchanged until
 *          quiet returns.
 *
 * @tparam WQE_PIPE           Must be PIPE_MTE3 for this action overload.
 * @param dst               [in] Symmetric destination address on pe.
 * @param src               [in] Local source address.
 * @param buf               [in] Local UB workspace shared by all calls in this batch.
 * @param elem_size         [in] Number of elements transferred by this Put operation.
 * @param pe                [in] Actual destination PE.
 * @param relay_pe          [in] PE whose port path is used to forward the packet.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_defer_t action referencing the batch's initialized
 *                          aclshmemx_submit_state_t.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id,
    aclshmemx_defer_t action);

/**
 * @brief Adds the current nonblocking relay UDMA Put operation and submits all operations in the batch.
 *
 * @details See @ref udma_submit_action_contract. Same PE-argument preconditions as the immediate relay Put
 *          overload. After submit, call aclshmemx_udma_quiet(pe); src must remain valid and unchanged until it
 *          returns.
 *
 * @tparam WQE_PIPE           Must be PIPE_MTE3 for this action overload.
 * @param dst               [in] Symmetric destination address on pe.
 * @param src               [in] Local source address.
 * @param buf               [in] Local UB workspace shared by all calls in this batch.
 * @param elem_size         [in] Number of elements transferred by this Put operation.
 * @param pe                [in] Actual destination PE.
 * @param relay_pe          [in] PE whose port path is used to forward the packet.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_submit_t action referencing the same aclshmemx_submit_state_t.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id,
    aclshmemx_submit_t action);

/**
 * @brief GlobalTensor/LocalTensor overload of @ref aclshmemx_udma_relay_put_nbi.
 *
 * @note  Same PE-argument preconditions and non-blocking completion semantics as the bare-pointer
 *        overload: invalid (pe, relay_pe) combinations submit nothing (abort in debug) and cannot be
 *        detected via the void return; a normal return only means the WQE was published, so call
 *        aclshmemx_udma_quiet(pe) before consuming the result or reusing @p src.
 * @note  A single UDMA put request transfers at most 256 MB (256 * 1024 * 1024 bytes); split larger
 *        transfers into multiple put requests whose byte size (@p elem_size * sizeof(T)) does not
 *        exceed 256 MB.
 *
 * @tparam T                  Element type of the transfer.
 * @tparam WQE_PIPE           Pipe used to publish the WQE. See the bare-pointer overload for semantics.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data (on pe).
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param buf               [in] LocalTensor on local UB. WQE staging scratch when
 *                               WQE_PIPE == PIPE_MTE3; ignored when WQE_PIPE == PIPE_S.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the actual destination. Must satisfy the preconditions above.
 * @param relay_pe          [in] PE whose port path is used to forward the packet. Must satisfy the
 *                               preconditions above.
 * @param sync_id           [in] Hardware event ID used by the MTE3->S sync in the PIPE_MTE3 path.
 *                               Ignored when WQE_PIPE == PIPE_S. Defaults to 0.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id = 0);

/**
 * @brief Adds the current nonblocking relay UDMA Put operation to a batch and keeps the batch pending.
 *
 * @details See @ref udma_submit_action_contract. Same PE-argument preconditions as the immediate relay Put
 *          overload. Finish the batch with an aclshmemx_submit_t action; src must remain valid and unchanged until
 *          quiet returns.
 *
 * @tparam WQE_PIPE           Must be PIPE_MTE3 for this action overload.
 * @param dst               [in] Remote symmetric GlobalTensor for the destination data on pe.
 * @param src               [in] Local GlobalTensor for the source data.
 * @param buf               [in] Local UB LocalTensor workspace shared by all calls in this batch.
 * @param elem_size         [in] Number of elements transferred by this Put operation.
 * @param pe                [in] Actual destination PE.
 * @param relay_pe          [in] PE whose port path is used to forward the packet.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_defer_t action referencing the batch's initialized
 *                          aclshmemx_submit_state_t.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id, aclshmemx_defer_t action);

/**
 * @brief Adds the current nonblocking relay UDMA Put operation and submits all operations in the batch.
 *
 * @details See @ref udma_submit_action_contract. Same PE-argument preconditions as the immediate relay Put
 *          overload. After submit, call aclshmemx_udma_quiet(pe); src must remain valid and unchanged until it
 *          returns.
 *
 * @tparam WQE_PIPE           Must be PIPE_MTE3 for this action overload.
 * @param dst               [in] Remote symmetric GlobalTensor for the destination data on pe.
 * @param src               [in] Local GlobalTensor for the source data.
 * @param buf               [in] Local UB LocalTensor workspace shared by all calls in this batch.
 * @param elem_size         [in] Number of elements transferred by this Put operation.
 * @param pe                [in] Actual destination PE.
 * @param relay_pe          [in] PE whose port path is used to forward the packet.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_submit_t action referencing the same aclshmemx_submit_state_t.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id, aclshmemx_submit_t action);

/**
 * @brief Asynchronous relay-mode UDMA Get. Symmetric to @ref aclshmemx_udma_relay_put_nbi.
 *        Requires building with ACLSHMEM_RELAY_SUPPORT=ON.
 *
 * @note  Preconditions on the PE arguments (all must hold, where myPe is the calling PE and
 *        rankCount is the total number of PEs):
 *          - 0 <= pe < rankCount and 0 <= relay_pe < rankCount
 *          - pe != relay_pe
 *          - pe != myPe and relay_pe != myPe (self is not a valid actual/relay target)
 *        If any precondition is violated the call submits nothing and returns immediately; in a
 *        debug build it aborts the kernel instead. Because the return type is void, the caller
 *        cannot detect a skipped submission at runtime -- validate the arguments before calling.
 * @note  Completion semantics: this is a non-blocking (_nbi) call. A normal return only means the
 *        WQE was published to the send queue; it does NOT mean the transfer finished or that @p dst
 *        holds the fetched data. Call aclshmemx_udma_quiet(pe) (or a higher-level barrier) before
 *        reading @p dst to guarantee completion.
 * @note  A single UDMA get request transfers at most 256 MB (256 * 1024 * 1024 bytes). @p elem_size
 *        is the number of T elements, so the transfer size is @p elem_size * sizeof(T) bytes. Split
 *        larger transfers into multiple get requests whose byte size does not exceed 256 MB.
 *
 * @tparam T                  Element type of the transfer.
 * @tparam WQE_PIPE           Pipe used to publish the WQE. See @ref aclshmemx_udma_relay_put_nbi.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data (on pe).
 * @param buf               [in] Pointer on local UB. WQE staging scratch when
 *                               WQE_PIPE == PIPE_MTE3; ignored when WQE_PIPE == PIPE_S.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the actual source. Must satisfy the preconditions above.
 * @param relay_pe          [in] PE whose port path is used to forward the packet. Must satisfy the
 *                               preconditions above.
 * @param sync_id           [in] Hardware event ID used by the MTE3->S sync in the PIPE_MTE3 path.
 *                               Ignored when WQE_PIPE == PIPE_S. Defaults to 0.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id = 0);

/**
 * @brief Adds the current nonblocking relay UDMA Get operation to a batch and keeps the batch pending.
 *
 * @details See @ref udma_submit_action_contract. Same PE-argument preconditions as the immediate relay Get
 *          overload. Finish the batch with an aclshmemx_submit_t action.
 *
 * @tparam WQE_PIPE           Must be PIPE_MTE3 for this action overload.
 * @param dst               [in] Local destination address.
 * @param src               [in] Symmetric source address on pe.
 * @param buf               [in] Local UB workspace shared by all calls in this batch.
 * @param elem_size         [in] Number of elements transferred by this Get operation.
 * @param pe                [in] Actual source PE.
 * @param relay_pe          [in] PE whose port path is used to forward the packet.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_defer_t action referencing the batch's initialized
 *                          aclshmemx_submit_state_t.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id,
    aclshmemx_defer_t action);

/**
 * @brief Adds the current nonblocking relay UDMA Get operation and submits all operations in the batch.
 *
 * @details See @ref udma_submit_action_contract. Same PE-argument preconditions as the immediate relay Get
 *          overload. After submit, call aclshmemx_udma_quiet(pe) before reading or reusing dst.
 *
 * @tparam WQE_PIPE           Must be PIPE_MTE3 for this action overload.
 * @param dst               [in] Local destination address.
 * @param src               [in] Symmetric source address on pe.
 * @param buf               [in] Local UB workspace shared by all calls in this batch.
 * @param elem_size         [in] Number of elements transferred by this Get operation.
 * @param pe                [in] Actual source PE.
 * @param relay_pe          [in] PE whose port path is used to forward the packet.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_submit_t action referencing the same aclshmemx_submit_state_t.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id,
    aclshmemx_submit_t action);

/**
 * @brief GlobalTensor/LocalTensor overload of @ref aclshmemx_udma_relay_get_nbi.
 *
 * @note  Same PE-argument preconditions and non-blocking completion semantics as the bare-pointer
 *        overload: invalid (pe, relay_pe) combinations submit nothing (abort in debug) and cannot be
 *        detected via the void return; a normal return only means the WQE was published, so call
 *        aclshmemx_udma_quiet(pe) before reading @p dst.
 * @note  A single UDMA get request transfers at most 256 MB (256 * 1024 * 1024 bytes); split larger
 *        transfers into multiple get requests whose byte size (@p elem_size * sizeof(T)) does not
 *        exceed 256 MB.
 *
 * @tparam T                  Element type of the transfer.
 * @tparam WQE_PIPE           Pipe used to publish the WQE. See the bare-pointer overload for semantics.
 *
 * @param dst               [in] GlobalTensor on local device of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data (on pe).
 * @param buf               [in] LocalTensor on local UB. WQE staging scratch when
 *                               WQE_PIPE == PIPE_MTE3; ignored when WQE_PIPE == PIPE_S.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the actual source. Must satisfy the preconditions above.
 * @param relay_pe          [in] PE whose port path is used to forward the packet. Must satisfy the
 *                               preconditions above.
 * @param sync_id           [in] Hardware event ID used by the MTE3->S sync in the PIPE_MTE3 path.
 *                               Ignored when WQE_PIPE == PIPE_S. Defaults to 0.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id = 0);

/**
 * @brief Adds the current nonblocking relay UDMA Get operation to a batch and keeps the batch pending.
 *
 * @details See @ref udma_submit_action_contract. Same PE-argument preconditions as the immediate relay Get
 *          overload. Finish the batch with an aclshmemx_submit_t action.
 *
 * @tparam WQE_PIPE           Must be PIPE_MTE3 for this action overload.
 * @param dst               [in] Local GlobalTensor for the destination data.
 * @param src               [in] Remote symmetric GlobalTensor for the source data on pe.
 * @param buf               [in] Local UB LocalTensor workspace shared by all calls in this batch.
 * @param elem_size         [in] Number of elements transferred by this Get operation.
 * @param pe                [in] Actual source PE.
 * @param relay_pe          [in] PE whose port path is used to forward the packet.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_defer_t action referencing the batch's initialized
 *                          aclshmemx_submit_state_t.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id, aclshmemx_defer_t action);

/**
 * @brief Adds the current nonblocking relay UDMA Get operation and submits all operations in the batch.
 *
 * @details See @ref udma_submit_action_contract. Same PE-argument preconditions as the immediate relay Get
 *          overload. After submit, call aclshmemx_udma_quiet(pe) before reading or reusing dst.
 *
 * @tparam WQE_PIPE           Must be PIPE_MTE3 for this action overload.
 * @param dst               [in] Local GlobalTensor for the destination data.
 * @param src               [in] Remote symmetric GlobalTensor for the source data on pe.
 * @param buf               [in] Local UB LocalTensor workspace shared by all calls in this batch.
 * @param elem_size         [in] Number of elements transferred by this Get operation.
 * @param pe                [in] Actual source PE.
 * @param relay_pe          [in] PE whose port path is used to forward the packet.
 * @param sync_id           [in] Hardware event ID used for MTE3 pipeline synchronization.
 * @param action            [in] aclshmemx_submit_t action referencing the same aclshmemx_submit_state_t.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id, aclshmemx_submit_t action);

/**
 * @brief Asynchronous interface. Copy a contiguous data from local to symmetric address on the specified PE and
 *        updating a remote signal flag on completion using UDMA.
 *        Template function for different data types.
 *        No-buf overload: not templated on WQE_PIPE and takes no UB scratch. Submits the WQE
 *        through the direct (non-staged) path, behaviorally equivalent to the buf-taking
 *        overload with WQE_PIPE = PIPE_S. Provided to match the original signature for
 *        backward compatibility with existing callers; new callers should use the buf-taking
 *        overload (which defaults to WQE_PIPE = PIPE_MTE3).
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 *        PE are not supported.
 *
 * @tparam T                  Element type of the transfer.
 *
 * @note A single UDMA put request transfers at most 256 MB
 *       (256 * 1024 * 1024 bytes). @p elem_size is the number of T
 *       elements, so the transfer size is @p elem_size * sizeof(T) bytes.
 *       Split larger transfers into multiple put requests whose byte size
 *       does not exceed 256 MB. After submitting one or more asynchronous
 *       requests, call the matching completion/synchronization interface,
 *       such as @ref aclshmemx_udma_quiet or the corresponding signal wait
 *       protocol, before reading the destination data or reusing buffers
 *       whose contents must remain stable.
 *
 * @param dst                 [in] Pointer on Symmetric memory of the destination data.
 * @param src                 [in] Pointer on local device of the source data.
 * @param elem_size           [in] Number of elements in the dest and source arrays.
 * @param sig_addr            [in] Symmetric address of the signal word to be updated.
 * @param signal              [in] The value used to update sig_addr.
 * @param pe                  [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_put_signal_nbi(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, __gm__ uint64_t* sig_addr, uint64_t signal, int pe);

/**
 * @brief Buf-taking overload of @ref aclshmemx_udma_put_signal_nbi. PIPE_MTE3 (default)
 *        stages the WRITE_WITH_NOTIFY WQE in the caller-provided UB scratch and
 *        DataCopyPads it to the SQ ring. PIPE_S falls through to the no-buf path and
 *        ignores @p buf and @p sync_id; provided so a single call site can be
 *        templated on WQE_PIPE.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 *        PE are not supported.
 *
 * @tparam T                  Element type of the transfer.
 * @tparam WQE_PIPE           PIPE_MTE3 (default) or PIPE_S; other pipes are not supported.
 *
 * @note A single UDMA put request transfers at most 256 MB
 *       (256 * 1024 * 1024 bytes). @p elem_size is the number of T
 *       elements, so the transfer size is @p elem_size * sizeof(T) bytes.
 *       Split larger transfers into multiple put requests whose byte size
 *       does not exceed 256 MB. After submitting one or more asynchronous
 *       requests, call the matching completion/synchronization interface,
 *       such as @ref aclshmemx_udma_quiet or the corresponding signal wait
 *       protocol, before reading the destination data or reusing buffers
 *       whose contents must remain stable.
 *
 * @param dst                 [in] Pointer on Symmetric memory of the destination data.
 * @param src                 [in] Pointer on local device of the source data.
 * @param elem_size           [in] Number of elements in the dest and source arrays.
 * @param sig_addr            [in] Symmetric address of the signal word to be updated.
 * @param signal              [in] The value used to update sig_addr.
 * @param pe                  [in] PE number of the remote PE.
 * @param buf                 [in] Pointer on local UB used as WQE staging scratch
 *                                 (must hold one full WRITE_WITH_NOTIFY WQE block;
 *                                 128 B is safe). Ignored when WQE_PIPE == PIPE_S.
 * @param sync_id             [in] Hardware event ID used by the MTE3->S sync after
 *                                 DataCopyPad in the PIPE_MTE3 path. Ignored when
 *                                 WQE_PIPE == PIPE_S. Defaults to 0.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_put_signal_nbi(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, __gm__ uint64_t* sig_addr, uint64_t signal, int pe,
    __ubuf__ uint8_t* buf, uint32_t sync_id = 0);

/**
 * @brief Direct-mode asynchronous UDMA Put with remote signal update on an explicitly selected QP.
 * @note The data write and signal update are encoded in one WRITE_WITH_NOTIFY WQE on @p qp_idx. The caller must
 *       ensure qp_idx is smaller than the UDMA QP count configured at initialization. An invalid index prints an
 *       error and aborts the calling kernel.
 * @note A single request transfers at most 256 MB. A normal return only means the WQE was submitted; call
 *       aclshmemx_udma_qp_quiet(pe, qp_idx) or use the corresponding signal-wait protocol before reusing @p src.
 *
 * @tparam T                Element type of the transfer.
 * @param dst               [in] Symmetric destination address on the target PE specified by pe.
 * @param src               [in] Local source address.
 * @param elem_size         [in] Number of elements transferred by this Put operation.
 * @param sig_addr          [in] Symmetric address of the signal word to update on the target PE.
 * @param signal            [in] Value written to sig_addr after the data transfer.
 * @param pe                [in] Target PE that owns dst and sig_addr.
 * @param qp_idx            [in] QP index selected for this operation. Must be in the configured QP range.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_put_signal_nbi(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, __gm__ uint64_t* sig_addr, uint64_t signal, int pe,
    uint32_t qp_idx);

/**
 * @brief Buf-taking overload of @ref aclshmemx_udma_qp_put_signal_nbi.
 * @note PIPE_MTE3 stages the selected-QP WRITE_WITH_NOTIFY WQE in @p buf. PIPE_S ignores @p buf and @p sync_id and
 *       uses the direct scalar-store path. Only direct-mode UDMA is supported.
 * @note A single request transfers at most 256 MB. A normal return only means the WQE was submitted; call
 *       aclshmemx_udma_qp_quiet(pe, qp_idx) or use the corresponding signal-wait protocol before reusing @p src.
 *
 * @tparam T                Element type of the transfer.
 * @tparam WQE_PIPE         Pipe used to publish the operation. Must be PIPE_MTE3 or PIPE_S.
 * @param dst               [in] Symmetric destination address on the target PE specified by pe.
 * @param src               [in] Local source address.
 * @param elem_size         [in] Number of elements transferred by this Put operation.
 * @param sig_addr          [in] Symmetric address of the signal word to update on the target PE.
 * @param signal            [in] Value written to sig_addr after the data transfer.
 * @param pe                [in] Target PE that owns dst and sig_addr.
 * @param qp_idx            [in] QP index selected for this operation. Must be in the configured QP range.
 * @param buf               [in] Local UB scratch used when WQE_PIPE == PIPE_MTE3; it must provide at least
 *                          ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE bytes. Ignored when WQE_PIPE == PIPE_S.
 * @param sync_id           [in] Hardware event ID used by the PIPE_MTE3 path. Ignored when WQE_PIPE == PIPE_S.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_qp_put_signal_nbi(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, __gm__ uint64_t* sig_addr, uint64_t signal, int pe,
    uint32_t qp_idx, __ubuf__ uint8_t* buf, uint32_t sync_id = 0);

/**
 * @brief UDMA Quiet function. This synchronous function ensures all previous UDMA WQEs are completed
 * (data has arrived at the destination PE).
 *
 * @param pe                [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmemx_udma_quiet(int pe);

/**
 * @brief Wait for all UDMA requests submitted to one direct-mode QP of a remote PE.
 * @note qp_idx must be smaller than the UDMA QP count configured at initialization. An invalid index prints an
 *       error and aborts the calling kernel.
 *
 * @param pe                [in] PE number of the remote PE.
 * @param qp_idx            [in] QP index whose submitted requests are to be completed.
 */
#if defined(ACLSHMEM_RELAY_SUPPORT)
ACLSHMEM_DEVICE void aclshmemx_udma_qp_quiet(int pe, uint32_t qp_idx) = delete;
#else
ACLSHMEM_DEVICE void aclshmemx_udma_qp_quiet(int pe, uint32_t qp_idx);
#endif

/**
 * @brief Asynchronous interface. Add value to dst (remote symmetric address) on the specified PE pe,
 * and atomically update the dst without returning the value. Supported types: int32, uint32, int64, uint64, float.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of atomic add
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_add(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Add value to dst (remote symmetric address) on the specified PE pe,
 * and return the previous content of dst. Supported types: int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of atomic add
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous content of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_add(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Conditionally update dst (remote symmetric address) on the specified PE pe
 * and return the previous content of dst. If cond and the remote dst value are equal,
 * then value is swapped into the remote dst; otherwise, the remote dst is unchanged. In either case, the old
 * value of the remote dest is returned. Supported types: int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param cond              [in] condition for swap
 * @param value             [in] Operand of atomic add
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous content of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_compare_swap(__gm__ T* dst, T cond, T value, int32_t pe);

/**
 * @brief Synchronous interface. Fetch the contents of dst (remote symmetric address) on the specified PE pe
 * and return the contents. Supported types: int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch(__gm__ T* dst, int32_t pe);

/**
 * @brief Synchronous interface. Set value to dst (remote symmetric address) on the specified PE pe
 * without returning a value. Supported types: int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Value to be atomically written to the remote PE.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_set(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Swap value to dst (remote symmetric address) on the specified PE pe
 * and return the previous contents of dst. Supported types: int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Value to be atomically written to the remote PE.
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_swap(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Increment dst (remote symmetric address) on the specified PE pe by one
 * and return the previous contents of dst. Supported types: int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_inc(__gm__ T* dst, int32_t pe);

/**
 * @brief Synchronous interface. Increment dst (remote symmetric address) on the specified PE pe by one
 * without returning a value. Supported types: int32, uint32, int64, uint64, float.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_inc(__gm__ T* dst, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise AND operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, and return the previous contents of dst. Supported types:
 * int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of bitwise AND operation.
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_and(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise AND operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, without returning a value. Supported types:
 * int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of bitwise AND operation.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_and(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise OR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, and return the previous contents of dst. Supported types:
 * int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of bitwise OR operation.
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_or(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise OR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, without returning a value. Supported types:
 * int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of bitwise OR operation.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_or(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise XOR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, and return the previous contents of dst. Supported types:
 * int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of bitwise XOR operation.
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_xor(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise XOR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, without returning a value. Supported types:
 * int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of bitwise XOR operation.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_xor(__gm__ T* dst, T value, int32_t pe);

#include "gm2gm/engine/shmem_device_udma.hpp" // IWYU pragma: keep

#endif
