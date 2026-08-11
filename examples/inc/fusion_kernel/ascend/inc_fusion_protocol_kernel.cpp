#include "inc_fusion_protocol_device.h"

using namespace inc::fusion;
using namespace inc::fusion::device;

// Compile/runtime smoke for the queue ABI. The production MIX worker and INC
// service kernels use the same helpers; this small entry remains useful for
// validating a new CANN/SHMEM port before GEMM is involved.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
inc_fusion_protocol_probe_kernel(__gm__ uint8_t *sym,
                                 __gm__ FusionKernelArgs *args_gm)
{
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() != 0u || args_gm == nullptr) return;
        Dcci(reinterpret_cast<__gm__ uint8_t *>(args_gm),
             sizeof(FusionKernelArgs));
        if (args_gm->magic != kFusionMagic ||
            args_gm->abi_version != kFusionAbiVersion ||
            args_gm->slot_count < kFusionMinSlots ||
            args_gm->worker_count < 2u ||
            args_gm->symmetric_layout.queue_lanes == 0u ||
            args_gm->symmetric_layout.queue_depth == 0u ||
            args_gm->symmetric_layout.packet_bytes == 0u) return;
        const uint64_t index = QueueIndex(
            args_gm, 0u, 0u, 0u, 0u);
        __gm__ FusionPacketHeader *header = Header(
            sym, args_gm->symmetric_layout.dispatch_header_off, index);
        header->kind = kFusionDispatchLaneEnd;
        header->ready = PacketCommit(args_gm->operation_generation, 0u);
        AscendC::PipeBarrier<PIPE_ALL>();
        Dcci(reinterpret_cast<__gm__ uint8_t *>(header),
             sizeof(FusionPacketHeader));
    }
}

extern "C" void launch_inc_fusion_protocol_probe_kernel(
    uint8_t *sym, FusionKernelArgs *args, void *stream)
{
    inc_fusion_protocol_probe_kernel<<<1, nullptr, stream>>>(sym, args);
}
