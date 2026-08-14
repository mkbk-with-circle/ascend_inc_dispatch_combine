#ifndef INC_DC_GATHER_MTE_AICORE_H
#define INC_DC_GATHER_MTE_AICORE_H

#include "kernel_operator.h"
#include "shmem.h"

// MP11R/MP11P：MTE2→UB→MTE3 本地 GM gather，禁止逐字节 AIV 循环。
constexpr uint32_t kDcLlGatherEvent = 6u;
constexpr uint32_t kDcLlGatherEventAlt = 7u;
constexpr uint32_t kDcLlGatherUbTileBytes = 8192u;

__aicore__ inline uint32_t DcLlGatherAlign32(uint32_t n)
{
    return (n + 31u) / 32u * 32u;
}

// 仅 issue MTE2（GM→UB），不 wait
__aicore__ inline void DcLlGatherIssueMte2(GM_ADDR src, uint32_t nbytes, __ubuf__ uint8_t *ub, uint32_t evt)
{
    AscendC::LocalTensor<uint8_t> ub_tensor;
    AscendC::GlobalTensor<uint8_t> gm_src;
    AscendC::DataCopyExtParams params(1, nbytes, 0, 0, 0);
    AscendC::DataCopyPadExtParams<uint8_t> pad;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ub);
    ub_tensor.address_.dataLen = DcLlGatherAlign32(nbytes);
    gm_src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(src));
    AscendC::DataCopyPad(ub_tensor, gm_src, params, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evt);
}

// wait MTE2 完成后 issue MTE3（UB→GM）
__aicore__ inline void DcLlGatherWaitMte2IssueMte3(GM_ADDR dst, uint32_t nbytes, __ubuf__ uint8_t *ub, uint32_t evt)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evt);
    AscendC::LocalTensor<uint8_t> ub_tensor;
    AscendC::GlobalTensor<uint8_t> gm_dst;
    AscendC::DataCopyExtParams params(1, nbytes, 0, 0, 0);
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ub);
    ub_tensor.address_.dataLen = DcLlGatherAlign32(nbytes);
    gm_dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(dst));
    AscendC::DataCopyPad(gm_dst, ub_tensor, params);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evt);
}

__aicore__ inline void DcLlGatherWaitMte3(uint32_t evt)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evt);
}

// 单块同步 GM→UB→GM（兼容路径；nbytes 不得超过 UB tile）
__aicore__ inline void DcLlGatherCopyGm2Gm(GM_ADDR dst, GM_ADDR src, uint32_t nbytes, __ubuf__ uint8_t *ub,
                                            uint32_t evt)
{
    DcLlGatherIssueMte2(src, nbytes, ub, evt);
    DcLlGatherWaitMte2IssueMte3(dst, nbytes, ub, evt);
    DcLlGatherWaitMte3(evt);
}

// Runtime Payload V2：按 8KiB UB tile 分块；支持 12–24KiB token，禁止单次 >8192 调 DcLlGatherCopyGm2Gm
__aicore__ inline void DcLlGatherCopyGm2GmChunked(GM_ADDR dst, GM_ADDR src, uint32_t nbytes, __ubuf__ uint8_t *ub,
                                                   uint32_t evt)
{
    uint32_t done = 0;
    while (done < nbytes) {
        uint32_t n = nbytes - done;
        if (n > kDcLlGatherUbTileBytes) {
            n = kDcLlGatherUbTileBytes;
        }
        DcLlGatherCopyGm2Gm(dst + done, src + done, n, ub, evt);
        done += n;
    }
}

// Copy a small regular-stride group into contiguous staging with one MTE2
// descriptor and one MTE3 descriptor.  total bytes must fit the caller-owned
// UB window.  This is particularly useful for top-k=1 round-robin routing,
// where adjacent output rows are every W-th source row.
__aicore__ inline void DcLlGatherCopyRegularRows(
    GM_ADDR dst, GM_ADDR src, uint32_t row_bytes, uint32_t rows,
    uint32_t src_gap_bytes, __ubuf__ uint8_t *ub, uint32_t evt)
{
    AscendC::LocalTensor<uint8_t> ub_tensor;
    AscendC::GlobalTensor<uint8_t> gm_src;
    AscendC::GlobalTensor<uint8_t> gm_dst;
    AscendC::DataCopyPadExtParams<uint8_t> pad;
    AscendC::DataCopyExtParams gather_params(
        static_cast<uint16_t>(rows), row_bytes, src_gap_bytes, 0u, 0u);
    ub_tensor.address_.logicPos =
        static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ub);
    ub_tensor.address_.dataLen = DcLlGatherAlign32(row_bytes * rows);
    gm_src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(src));
    AscendC::DataCopyPad(ub_tensor, gm_src, gather_params, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evt);
    AscendC::DataCopyExtParams store_params(
        1u, row_bytes * rows, 0u, 0u, 0u);
    gm_dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(dst));
    AscendC::DataCopyPad(gm_dst, ub_tensor, store_params);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evt);
}

__aicore__ inline void DcLlGatherAppend(GM_ADDR staging, uint32_t staging_off, GM_ADDR src, uint32_t nbytes,
                                       __ubuf__ uint8_t *ub)
{
    // 大块按 8KiB tile 切分，避免单次 DataCopy 超 UB。
    uint32_t done = 0;
    while (done < nbytes) {
        uint32_t n = nbytes - done;
        if (n > kDcLlGatherUbTileBytes) {
            n = kDcLlGatherUbTileBytes;
        }
        DcLlGatherCopyGm2Gm(staging + staging_off + done, src + done, n, ub, kDcLlGatherEvent);
        done += n;
    }
}

// MP11P：4×非连续 token → 连续 staging；真双缓冲 MTE2(N+1)|MTE3(N)
__aicore__ inline void DcLlGatherFourChunks(GM_ADDR staging, GM_ADDR src0, GM_ADDR src1, GM_ADDR src2, GM_ADDR src3,
                                            uint32_t chunk_bytes, __ubuf__ uint8_t *ub0, __ubuf__ uint8_t *ub1)
{
    GM_ADDR srcs[4];
    srcs[0] = src0;
    srcs[1] = src1;
    srcs[2] = src2;
    srcs[3] = src3;
    DcLlGatherIssueMte2(srcs[0], chunk_bytes, ub0, kDcLlGatherEvent);
    for (uint32_t c = 0; c < 4u; ++c) {
        __ubuf__ uint8_t *ub_cur = (c & 1u) ? ub1 : ub0;
        const uint32_t evt_cur = (c & 1u) ? kDcLlGatherEventAlt : kDcLlGatherEvent;
        DcLlGatherWaitMte2IssueMte3(staging + c * chunk_bytes, chunk_bytes, ub_cur, evt_cur);
        if (c + 1u < 4u) {
            __ubuf__ uint8_t *ub_next = ((c + 1u) & 1u) ? ub1 : ub0;
            const uint32_t evt_next = ((c + 1u) & 1u) ? kDcLlGatherEventAlt : kDcLlGatherEvent;
            // 复用同 UB 前先等上一轮 MTE3（c>=1 时 ub_next 曾用于 MTE3(c-1)）
            if (c >= 1u) {
                DcLlGatherWaitMte3(evt_next);
            }
            DcLlGatherIssueMte2(srcs[c + 1u], chunk_bytes, ub_next, evt_next);
        }
    }
    // 收尾：两路 MTE3 都完成
    DcLlGatherWaitMte3(kDcLlGatherEvent);
    DcLlGatherWaitMte3(kDcLlGatherEventAlt);
}

#endif
