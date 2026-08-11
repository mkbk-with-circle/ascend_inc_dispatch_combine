/**
 * D1 transport ABI：host/device 共用，禁止在 kernel 内重复定义结构体。
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace inc::dc::dn {

constexpr uint32_t kD1Magic = 0x44315431u;
constexpr uint32_t kD1FormalTokens = 512;
constexpr uint32_t kD1TokenBytes = 8192;
constexpr uint64_t kD1BytesPerSource =
    static_cast<uint64_t>(kD1FormalTokens) * static_cast<uint64_t>(kD1TokenBytes);
constexpr uint32_t kD1MaxLanes = 8;
constexpr uint32_t kD1QuietBatch = 8;
constexpr uint8_t kD1CanarySource = 0xA5u;
constexpr uint8_t kD1CanaryDest = 0x5Au;

// Primitive matrix：P0..P5（pad0/transport_mode）
constexpr uint32_t kD1PrimP0PerTokenQuiet = 0u;
constexpr uint32_t kD1PrimP1NbiFinalQuiet = 1u;
constexpr uint32_t kD1PrimP2BlockingPut = 2u;
constexpr uint32_t kD1PrimP3NbiMteUbComplete = 3u;
constexpr uint32_t kD1PrimP4RangeQuiet = 4u;
constexpr uint32_t kD1PrimP5LargeRangeQuiet = 5u;
constexpr uint32_t kD1PrimP6RangeMteWaitFinalQuiet = 6u;
// 正式路径：MTE→P6，其它 transport→P4
constexpr uint32_t kD1PrimAutoRange = 7u;
// device cycle → µs（与样机 AICore 频率一致，gate 脚本可校准）
constexpr double kD1CyclesPerUs = 50.0;
constexpr uint8_t kD1TopoMte = 1u << 0;
constexpr uint8_t kD1TopoRoce = 1u << 1;
constexpr uint8_t kD1TopoSdma = 1u << 2;
constexpr uint8_t kD1TopoUdma = 1u << 3;

// 独立 cacheline doorbell，禁止把 done 塞在 Ctr 尾部
struct D1ControlLine {
    uint64_t value;
    uint8_t pad[56];
};
static_assert(sizeof(D1ControlLine) == 64u);

using D1DoneLine = D1ControlLine;
using D1GoLine = D1ControlLine; // worker 等待 device leader fanout
using D1StartLine = D1GoLine;   // INC 等待 paired worker/leader 通知
using D1WorkerDoneLine = D1ControlLine;
using D1LocalReadyLine = D1ControlLine;
using D1LeaderReleaseLine = D1ControlLine;
using D1SessionStopLine = D1ControlLine;

constexpr uint64_t kD1SessionStopMagic = UINT64_MAX;
constexpr uint32_t kD1LeaderPe = 0u;

struct D1Desc {
    uint32_t magic;
    uint32_t tokens;
    uint32_t payload_bytes;
    uint32_t peer_pe;
    uint32_t my_role; // 0=worker 1=inc
    uint32_t lane_count;
    uint32_t pair_id;
    uint32_t quiet_batch;     // range 模式：每批 token 数
    uint32_t transport_mode;  // kD1PrimP0..P6；正式默认 kD1PrimAutoRange
    uint64_t start_epoch;
    uint64_t pair_start_cycle;
};
static_assert(sizeof(D1Desc) <= 64u);

// 每 lane 独占 64B cacheline，避免 false sharing
struct D1LaneCtr {
    uint64_t put_count;
    uint64_t quiet_count;
    uint64_t lane_end_cycle;
    uint64_t bytes_sent;
    uint64_t done_epoch;
    uint64_t mte_wait_count;
    uint8_t pad[16];
};
static_assert(sizeof(D1LaneCtr) == 64u);

struct D1Ctr {
    D1LaneCtr lanes[kD1MaxLanes];
    uint64_t transport_start_cycle;
    uint64_t transport_end_cycle;
    uint64_t pair_end_cycle;
    uint64_t aggregator_wait_cycles;
    uint64_t done_publish_cycle;
    uint64_t remote_done_seen_cycle;
    uint32_t error_code;
    uint32_t active_lanes;
    uint8_t pad[40];
};
static_assert(offsetof(D1Ctr, lanes) % 64u == 0u);

// Worker 稳定 publish scratch（put 源地址必须在 GM 且已 DCCI）
struct D1PublishScratch {
    uint64_t epoch;
    uint8_t pad[56];
};
static_assert(sizeof(D1PublishScratch) == 64u);

// Step1：记录 peer 实际 transport 与 MTE/SDMA UB 配置
struct D1TransportProbe {
    uint8_t active_transport; // topo_list[peer] 原始位图
    uint8_t pad0[7];
    uint64_t mte_ub;
    uint32_t mte_ub_size;
    uint32_t mte_sync_id;
    uint64_t sdma_ub;
    uint32_t sdma_ub_size;
    uint32_t sdma_sync_id;
    uint8_t pad1[24];
};
static_assert(sizeof(D1TransportProbe) == 64u);

// PE0 设备侧 rendezvous：release_command + 每 worker 独占 ready slot（各 64B）
struct D1LeaderRendezvous {
    D1LeaderReleaseLine release_command;
    D1ControlLine ready_slots[kD1MaxLanes];
};
static_assert(sizeof(D1LeaderRendezvous) == 576u);

// INC 向 PE0 报告 pair 完成（各 64B，由接收端观察 done_line 后发布）
struct D1LeaderCompletion {
    D1ControlLine pair_done_slots[kD1MaxLanes];
};
static_assert(sizeof(D1LeaderCompletion) == 512u);

using D1GlobalDoneLine = D1ControlLine;

// 全局完成时序：pair_done 聚合 → global_done 发布（PE0 独占）
struct D1GlobalDoneTiming {
    uint64_t pair_done_wait_start_cycle;
    uint64_t pair_done_complete_cycle;
    uint64_t global_done_publish_cycle;
    uint64_t last_global_epoch;
    uint32_t error_code;
    uint32_t worker_count;
    uint8_t pad[24];
};
static_assert(sizeof(D1GlobalDoneTiming) == 64u);

// leader 首次观察到各 pair_done slot 的 device cycle（每 lane 64B，PE0 同一时钟）
struct D1PairDoneArrivalLane {
    uint64_t pair_done_first_seen_cycle;
    uint8_t pad[56];
};
static_assert(sizeof(D1PairDoneArrivalLane) == 64u);

struct D1PairDoneArrival {
    D1PairDoneArrivalLane lanes[kD1MaxLanes];
};
static_assert(sizeof(D1PairDoneArrival) == 512u);

// leader 时序探针（PE0 独占）：rendezvous + go fanout
struct D1LeaderTiming {
    uint64_t release_seen_cycle;
    uint64_t rendezvous_complete_cycle;
    uint64_t go_fanout_start_cycle;
    uint64_t go_fanout_end_cycle;
    uint64_t last_go_epoch;
    uint32_t error_code;
    uint32_t worker_count;
    uint8_t pad[16];
};
static_assert(sizeof(D1LeaderTiming) == 64u);

constexpr uint64_t kD1DescOff = 0;
constexpr uint64_t kD1CtrOff = 256;
constexpr uint64_t kD1CtrBytes = 768; // lanes512 + ctr tail，64B 对齐区
constexpr uint64_t kD1DoneLineOff = kD1CtrOff + kD1CtrBytes;
constexpr uint64_t kD1WorkerDoneLineOff = kD1DoneLineOff + 64;
constexpr uint64_t kD1StartLineOff = kD1WorkerDoneLineOff + 64;
constexpr uint64_t kD1SessionStopLineOff = kD1StartLineOff + 64;
constexpr uint64_t kD1LocalReadyLineOff = kD1SessionStopLineOff + 64;
constexpr uint64_t kD1LeaderRendezvousOff = kD1LocalReadyLineOff + 64;
constexpr uint64_t kD1LeaderCompletionOff = kD1LeaderRendezvousOff + sizeof(D1LeaderRendezvous);
constexpr uint64_t kD1GlobalDoneLineOff = kD1LeaderCompletionOff + sizeof(D1LeaderCompletion);
constexpr uint64_t kD1GlobalDoneTimingOff = kD1GlobalDoneLineOff + 64;
constexpr uint64_t kD1PairDoneArrivalOff = kD1GlobalDoneTimingOff + 64;
constexpr uint64_t kD1LeaderTimingOff = kD1PairDoneArrivalOff + sizeof(D1PairDoneArrival);
constexpr uint64_t kD1PublishScratchOff = kD1LeaderTimingOff + 64;
constexpr uint64_t kD1TransportProbeOff = kD1PublishScratchOff + 64;
constexpr uint64_t kD1ProbeBufOff = kD1TransportProbeOff + 64; // verify probe 128B
constexpr uint64_t kD1SourceOff = 4096;
constexpr uint64_t kD1DestOff = kD1SourceOff + kD1BytesPerSource;
constexpr uint64_t kD1HeapNeed = kD1DestOff + kD1BytesPerSource + 4096;

inline uint8_t D1PatternByte(uint32_t pair_id, uint32_t token_seq, uint32_t byte_off)
{
    return static_cast<uint8_t>((pair_id * 31u + token_seq * 17u + byte_off) & 0xFFu);
}


} // namespace inc::dc::dn
