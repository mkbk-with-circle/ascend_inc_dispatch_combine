/**
 * Queue-v2 正式 ABI：host/device 共用布局，禁止 alignas 造成偏移差异。
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace inc::dc::dn::qv2 {

constexpr uint32_t kQv2Magic = 0x51563244u; // QV2D
constexpr uint32_t kQv2FormalTokens = 512;
// Default / legacy compile-time width (fp16 hidden=4096). Runtime Payload V2 may use
// any allowed width up to kQv2MaxTokenBytes; device must read per-invocation payload_bytes.
constexpr uint32_t kQv2TokenBytes = 8192;
constexpr uint32_t kQv2MaxTokenBytes = 24576; // fp16 hidden=12288
constexpr uint32_t kQv2RangeBytes = 64u * 1024u; // transport range budget (byte-aware batching later)
constexpr uint32_t kQv2PayloadBatchTokens = 8;
constexpr uint32_t kQv2PublishTileTokens = 32;
constexpr uint32_t kQv2HeadPublishTileTokens = 32;
constexpr uint32_t kQv2RingDepth = 64;
constexpr uint32_t kQv2LaneCount = 8;
constexpr uint32_t kQv2PairCount = 8;
constexpr uint32_t kQv2DescriptorBytes = 64;

// D2.0 ordering probe 场景
constexpr uint32_t kQv2ScenarioBatch8 = 0u;
constexpr uint32_t kQv2ScenarioTile32x4 = 1u;
constexpr uint32_t kQv2ScenarioTile64x8 = 2u;
constexpr uint32_t kQv2ScenarioWrapSplit = 3u; // tail 起点 60 + 64 token
constexpr uint32_t kQv2ScenarioDelayedConsumer = 4u;
constexpr uint32_t kQv2ScenarioDelayedCredit = 5u;

struct Qv2Descriptor {
    uint64_t epoch;
    uint64_t lane_sequence;
    uint32_t token_sequence;
    uint32_t source_rank;
    uint32_t destination_rank;
    uint32_t payload_bytes;
    uint32_t source_lane;
    uint32_t ring_slot;
    uint32_t flags;
    uint32_t checksum_seed;
    uint8_t pad[16];
};
static_assert(sizeof(Qv2Descriptor) == 64u, "Qv2Descriptor must be 64B");

struct Qv2TailLine {
    uint64_t tail;
    uint64_t epoch_tag;
    uint32_t magic;
    uint32_t lane_id;
    // 单调递增，与 cumulative tail 对齐；epoch 内多次 publish 时 signal 不能只用 epoch
    int32_t notify_seq;
    uint8_t pad[64u - 28u];
};
static_assert(sizeof(Qv2TailLine) == 64u, "Qv2TailLine must be 64B");
constexpr uint32_t kQv2TailNotifySeqOff = offsetof(Qv2TailLine, notify_seq);
static_assert(kQv2TailNotifySeqOff == 24u);

struct Qv2HeadLine {
    uint64_t head;
    uint64_t epoch_tag;
    uint32_t magic;
    uint32_t lane_id;
    // 单调递增，与 cumulative head 对齐；destination→INC credit 门铃
    int32_t notify_seq;
    uint8_t pad[64u - 28u];
};
static_assert(sizeof(Qv2HeadLine) == 64u, "Qv2HeadLine must be 64B");
constexpr uint32_t kQv2HeadNotifySeqOff = offsetof(Qv2HeadLine, notify_seq);
static_assert(kQv2HeadNotifySeqOff == 24u);

struct Qv2ProbeDesc {
    uint32_t magic;
    uint32_t scenario_id;
    uint32_t tokens_per_epoch;
    uint32_t payload_bytes;
    uint32_t peer_pe;
    uint32_t my_role; // 0=worker 1=inc
    uint32_t ring_depth;
    uint32_t payload_batch_tokens;
    uint32_t publish_tile_tokens;
    uint32_t head_publish_tile_tokens;
    uint32_t tail_start_offset; // wrap probe：单调 tail 起点
    uint32_t transport_mode;    // kD1PrimP6 或 kD1PrimP4 对照
    uint32_t delayed_consumer_spins;
    uint32_t delayed_credit_spins;
    uint32_t source_lane;
    uint32_t pair_id;
    uint64_t epoch_id;
};
static_assert(sizeof(Qv2ProbeDesc) <= 128u);

struct Qv2Telemetry {
    uint64_t payload_put_count;
    uint64_t descriptor_put_count;
    uint64_t payload_mte_wait_count;
    uint64_t descriptor_mte_wait_count;
    uint64_t tail_publish_count;
    uint64_t tail_completion_count;
    uint64_t head_publish_count;
    uint64_t credit_wait_cycles;
    uint64_t early_tail_count;
    uint64_t stale_desc_count;
    uint64_t slot_overwrite_count;
    uint64_t max_occupancy;
    uint64_t tokens_published;
    uint64_t tokens_consumed;
    uint64_t transport_start_cycle;
    uint64_t transport_end_cycle;
    uint32_t error_code;
    uint32_t verify_ok;
    uint32_t scenario_id;
    uint32_t epoch_id;
    uint8_t pad[48];
};
static_assert(sizeof(Qv2Telemetry) == 192u);

constexpr uint64_t kQv2DescOff = 0;
constexpr uint64_t kQv2TelemetryOff = 256;
constexpr uint64_t kQv2TailLineOff = 512;
constexpr uint64_t kQv2HeadLineOff = 576;
constexpr uint64_t kQv2WorkerSourceOff = 4096;
constexpr uint64_t kQv2WorkerDescStagingOff =
    kQv2WorkerSourceOff + static_cast<uint64_t>(kQv2RingDepth) * kQv2TokenBytes;
constexpr uint64_t kQv2IncPayloadOff =
    kQv2WorkerDescStagingOff + static_cast<uint64_t>(kQv2RingDepth) * kQv2DescriptorBytes;
constexpr uint64_t kQv2IncDescOff =
    kQv2IncPayloadOff + static_cast<uint64_t>(kQv2RingDepth) * kQv2TokenBytes;
constexpr uint64_t kQv2HeapNeed = kQv2IncDescOff + static_cast<uint64_t>(kQv2RingDepth) * kQv2DescriptorBytes + 4096;

inline uint8_t Qv2PatternByte(uint32_t pair_id, uint32_t token_seq, uint32_t byte_off)
{
    return static_cast<uint8_t>((pair_id * 31u + token_seq * 17u + byte_off) & 0xFFu);
}

inline uint32_t Qv2ChecksumSeed(uint32_t pair_id, uint32_t token_seq, uint32_t epoch)
{
    return pair_id * 1315423911u + token_seq * 2654435761u + epoch;
}

} // namespace inc::dc::dn::qv2

// ---------------------------------------------------------------------------
// D2.1 campaign 布局（与 D2.0 probe 固定 offset 分离）
// ---------------------------------------------------------------------------
namespace inc::dc::dn::qv2::camp {

constexpr uint32_t kQv2CampMagic = 0x51563243u; // QV2C
constexpr uint32_t kQv2RouteCount = 2u;
constexpr uint32_t kCampMaxTokens = 4096u; // stress 上限，布局按此规划

// start 路径探针：A/B/C/D（见 T2）
constexpr uint32_t kCampStartHostDirectD1B = 0u;
constexpr uint32_t kCampStartHostDirectLow = 1u;
constexpr uint32_t kCampStartLeaderFanoutLow = 2u;
constexpr uint32_t kCampStartLeaderFanoutD1B = 3u;

constexpr uint32_t kCampKernelExitNone = 0u;
constexpr uint32_t kCampKernelExitStop = 1u;
constexpr uint32_t kCampKernelExitLaneError = 2u;
constexpr uint32_t kCampKernelExitRouteError = 3u;
constexpr uint32_t kCampKernelExitDescInvalid = 4u;
constexpr uint32_t kCampKernelExitUploadAggFail = 5u;

// payload 预填模式：默认 static=0 不影响正式路径
constexpr uint32_t kQv2PayloadPatternStatic = 0u;
constexpr uint32_t kQv2PayloadPatternEpochUnique = 1u;

struct Qv2CampaignDesc {
    uint32_t magic;
    uint32_t pair_id;
    uint32_t peer_pe;
    uint32_t my_role; // 0=worker upload 1=inc route
    uint32_t tokens_per_epoch;
    uint32_t tokens_per_lane;
    uint32_t payload_bytes;
    uint32_t ring_depth;
    uint32_t batch_tokens;
    uint32_t tile_tokens;
    uint32_t head_tile_tokens;
    uint32_t lane_count;
    uint32_t route_id;
    uint32_t transport_mode;
    uint32_t start_mode;
    uint64_t go_epoch;
    uint64_t d1_start_off;
    uint64_t d1_stop_off;
    uint32_t payload_pattern_mode;       // kQv2PayloadPattern*
    uint32_t source_epoch_stride_tokens; // 每 epoch token 数（=tokens_per_epoch）
    uint32_t source_epoch_count;         // 预填 epoch 数
    uint32_t stale_source_inject_epoch;  // probe 负例：该 epoch 故意读旧 source
};
static_assert(sizeof(Qv2CampaignDesc) <= 128u);

// 每 AIV 独占 128B trace/cacheline，禁止跨 block 非原子写
struct Qv2ServiceTraceLine {
    uint64_t kernel_enter_cycle;
    uint32_t desc_valid;
    uint32_t block_id;
    uint64_t wait_start_enter_cycle;
    uint64_t observed_start_value;
    uint64_t observed_stop_value;
    uint64_t wait_start_exit_cycle;
    uint64_t epoch_enter_cycle;
    uint64_t first_payload_issue_cycle;
    uint64_t first_tail_seen_cycle;
    uint32_t kernel_exit_reason;
    uint32_t wait_start_spin_count;
    uint64_t service_exit_epoch;
    uint64_t start_seen_epoch;
    uint8_t pad[32];
};
static_assert(sizeof(Qv2ServiceTraceLine) == 128u);

struct Qv2LaneCounters {
    uint64_t local_tail;
    uint64_t cached_remote_head;
    uint64_t payload_put_count;
    uint64_t payload_mte_wait_count;
    uint64_t descriptor_tile_put_count;
    uint64_t descriptor_mte_wait_count;
    uint64_t data_descriptor_drain_quiet_count;
    uint64_t tail_publish_count;
    uint64_t tail_completion_quiet_count;
    uint64_t credit_wait_cycles;
    uint32_t wrap_split_put_count;
    uint32_t error_code;
    uint8_t pad[40];
};
static_assert(sizeof(Qv2LaneCounters) == 128u);

struct Qv2RouteCounters {
    uint64_t tokens_consumed;
    uint64_t payload_verified_count;
    uint64_t descriptor_verified_count;
    uint64_t verify_error_count;
    uint64_t max_occupancy;
    uint64_t slot_overwrite_count;
    uint64_t stale_epoch_count;
    uint64_t duplicate_count;
    uint64_t lost_count;
    uint64_t underflow_count;
    uint64_t head_range_publish_count;
    uint64_t head_completion_quiet_count;
    uint64_t tail_poll_cycles;
    uint64_t descriptor_consume_cycles;
    uint64_t descriptor_dcci_retry_count;
    uint64_t persistent_stale_descriptor_count;
    uint64_t consumed_lane_head[4];
    uint32_t error_code;
    uint32_t pad0;
    uint8_t pad1[88];
};
static_assert(sizeof(Qv2RouteCounters) == 256u);

struct Qv2RouteDoneLine {
    uint64_t done_epoch;
    uint32_t route_id;
    uint32_t magic;
    uint8_t pad[48];
};
static_assert(sizeof(Qv2RouteDoneLine) == 64u);

struct Qv2CampTiming {
    uint64_t first_payload_issue_cycle;
    uint64_t first_tail_publish_cycle;
    uint64_t first_inc_consume_cycle;
    uint64_t last_payload_issue_cycle;
    uint64_t last_inc_consume_cycle;
    uint64_t first_head_publish_cycle;
    uint64_t epoch_start_cycle;
    uint64_t epoch_end_cycle;
    uint64_t upload_done_epoch;
    uint64_t route_done_epoch[2];
    uint32_t upload_done_bitmap;
    uint32_t route_done_bitmap;
    uint32_t error_code;
    uint32_t pad0;
    uint8_t pad1[24];
};
static_assert(sizeof(Qv2CampTiming) == 128u);

struct Qv2ResidentLine {
    uint64_t value; // 1 = block kernel resident
    uint32_t block_id;
    uint32_t magic;
    uint8_t pad[48];
};
static_assert(sizeof(Qv2ResidentLine) == 64u);

// persistent session：每 epoch 设备侧写 history，session stop 后 host 批量 D2H
constexpr uint32_t kQv2CampMaxEpochHist = 16u;

struct Qv2WorkerEpochHistEntry {
    uint64_t epoch;
    uint64_t payload_put_count;
    uint64_t payload_mte_wait_count;
    uint64_t descriptor_tile_put_count;
    uint64_t descriptor_mte_wait_count;
    uint64_t data_descriptor_drain_quiet_count;
    uint64_t tail_publish_count;
    uint64_t tail_completion_quiet_count;
    uint64_t credit_wait_cycles;
    uint32_t wrap_split_put_count;
    uint32_t upload_done_bitmap;
    uint64_t final_local_tail;
    uint32_t error_code;
    uint32_t pad0;
    uint8_t pad1[32];
};
static_assert(sizeof(Qv2WorkerEpochHistEntry) == 128u);

struct Qv2IncEpochHistEntry {
    uint64_t epoch;
    uint64_t descriptor_verified_count;
    uint64_t payload_verified_count;
    uint64_t tokens_consumed;
    uint64_t head_range_publish_count;
    uint64_t head_completion_quiet_count;
    uint32_t route_done_bitmap;
    uint32_t verify_error_count;
    uint64_t final_route_head;
    uint64_t max_occupancy;
    uint64_t stale_epoch_count;
    uint64_t duplicate_count;
    uint64_t lost_count;
    uint64_t slot_overwrite_count;
    uint64_t underflow_count;
    uint32_t error_code;
    uint32_t pad0;
    uint8_t pad1[8];
};
static_assert(sizeof(Qv2IncEpochHistEntry) == 128u);

struct Qv2WorkerEpochHistRing {
    uint32_t magic;
    uint32_t count;
    uint8_t pad[56];
    Qv2WorkerEpochHistEntry entries[kQv2CampMaxEpochHist];
};

struct Qv2IncEpochHistRing {
    uint32_t magic;
    uint32_t count;
    uint8_t pad[56];
    Qv2IncEpochHistEntry entries[kQv2CampMaxEpochHist];
};

constexpr uint64_t kCampDescOff = 0;
constexpr uint64_t kCampResidentOff = 256;
constexpr uint64_t kCampWorkerResidentOff = kCampResidentOff;
// 8×64B resident 后紧接 route resident，禁止再硬编码 512（会与 lane4+ resident 重叠）
constexpr uint64_t kCampRouteResidentOff = kCampWorkerResidentOff + static_cast<uint64_t>(kQv2LaneCount) * 64u;
constexpr uint64_t kCampWorkerTraceOff = kCampRouteResidentOff + static_cast<uint64_t>(kQv2RouteCount) * 64u;
constexpr uint64_t kCampRouteTraceOff = kCampWorkerTraceOff + static_cast<uint64_t>(kQv2LaneCount) * 128u;
constexpr uint64_t kCampRouteDoneLineOff = kCampRouteTraceOff + static_cast<uint64_t>(kQv2RouteCount) * 128u;
constexpr uint64_t kCampLaneCtrOff = kCampRouteDoneLineOff + static_cast<uint64_t>(kQv2RouteCount) * 64u;
constexpr uint64_t kCampRouteCtrOff = kCampLaneCtrOff + static_cast<uint64_t>(kQv2LaneCount) * 128u;
constexpr uint64_t kCampTimingOff = kCampRouteCtrOff + static_cast<uint64_t>(kQv2RouteCount) * 256u;
constexpr uint64_t kCampTailLineOff = 4096;
// 控制面必须落在 tail 线之前
static_assert(kCampTimingOff + sizeof(Qv2CampTiming) <= kCampTailLineOff, "camp control overlaps tail");
static_assert(kCampWorkerTraceOff >= kCampRouteResidentOff + static_cast<uint64_t>(kQv2RouteCount) * 64u,
              "worker trace overlaps route resident");
constexpr uint64_t kCampHeadLineOff = kCampTailLineOff + 512;
constexpr uint64_t kCampLowControlOff = 8192;
constexpr uint64_t kCampLowStartLineOff = kCampLowControlOff;
constexpr uint64_t kCampLowStopLineOff = kCampLowControlOff + 64u;
constexpr uint64_t kCampDataBaseOff = 16384;
constexpr uint64_t kCampWorkerSourceOff = kCampDataBaseOff;
constexpr uint64_t kCampWorkerDescOff =
    kCampWorkerSourceOff + static_cast<uint64_t>(kCampMaxTokens) * kQv2TokenBytes;
constexpr uint64_t kCampIncPayloadOff =
    kCampWorkerDescOff + static_cast<uint64_t>(kCampMaxTokens) * kQv2DescriptorBytes;
constexpr uint64_t kCampIncDescOff =
    kCampIncPayloadOff + static_cast<uint64_t>(kQv2LaneCount) * kQv2RingDepth * kQv2TokenBytes;
constexpr uint64_t kCampD1RegionOff = (kCampIncDescOff + 1048575u) & ~1048575u; // 1MiB 对齐
// epoch history 落在 D1 区 scratch，避免与 tail/head/data 重叠
constexpr uint64_t kCampHistBaseOff = kCampD1RegionOff + 4096u;
constexpr uint64_t kCampWorkerHistStride = sizeof(Qv2WorkerEpochHistRing);
constexpr uint64_t kCampIncHistStride = sizeof(Qv2IncEpochHistRing);

inline uint64_t CampWorkerHistRingOff(uint32_t pair_id)
{
    return kCampHistBaseOff + static_cast<uint64_t>(pair_id) * kCampWorkerHistStride;
}

inline uint64_t CampIncHistRingOff(uint32_t pair_id)
{
    return kCampHistBaseOff + static_cast<uint64_t>(kQv2PairCount) * kCampWorkerHistStride +
           static_cast<uint64_t>(pair_id) * kCampIncHistStride;
}

constexpr uint64_t kCampHeapNeed = kCampD1RegionOff + 65536u;

constexpr uint64_t kCampHistEndOff =
    kCampHistBaseOff + static_cast<uint64_t>(kQv2PairCount) * kCampWorkerHistStride +
    static_cast<uint64_t>(kQv2PairCount) * kCampIncHistStride;
static_assert(kCampHistEndOff <= kCampD1RegionOff + 65536u, "epoch hist exceeds D1 scratch");

inline uint64_t CampLaneTokenBase(uint32_t lane_id, uint32_t tokens_per_lane)
{
    return static_cast<uint64_t>(lane_id) * tokens_per_lane;
}

inline uint64_t CampIncPayloadLaneOff(uint32_t lane_id)
{
    return kCampIncPayloadOff + static_cast<uint64_t>(lane_id) * kQv2RingDepth * kQv2TokenBytes;
}

inline uint64_t CampIncDescLaneOff(uint32_t lane_id)
{
    return kCampIncDescOff + static_cast<uint64_t>(lane_id) * kQv2RingDepth * kQv2DescriptorBytes;
}

inline uint64_t CampTailLineOff(uint32_t lane_id)
{
    return kCampTailLineOff + static_cast<uint64_t>(lane_id) * 64u;
}

inline uint64_t CampHeadLineOff(uint32_t lane_id)
{
    return kCampHeadLineOff + static_cast<uint64_t>(lane_id) * 64u;
}

inline uint64_t CampWorkerTraceOff(uint32_t lane_id)
{
    return kCampWorkerTraceOff + static_cast<uint64_t>(lane_id) * 128u;
}

inline uint64_t CampRouteTraceOff(uint32_t route_id)
{
    return kCampRouteTraceOff + static_cast<uint64_t>(route_id) * 128u;
}

inline uint64_t CampRouteDoneLineOff(uint32_t route_id)
{
    return kCampRouteDoneLineOff + static_cast<uint64_t>(route_id) * 64u;
}

// host/device 边界校验
inline bool CampValidateHeap(uint32_t tokens, uint32_t lanes)
{
    if (tokens == 0 || lanes == 0 || tokens > kCampMaxTokens || (tokens % lanes) != 0) {
        return false;
    }
    const uint64_t need = kCampIncDescOff + static_cast<uint64_t>(lanes) * kQv2RingDepth * kQv2DescriptorBytes;
    return need <= kCampD1RegionOff;
}

// epoch-unique 预填：source_epoch_count × tokens_per_epoch 不得越界 descriptor 区
inline bool CampValidateEpochUniqueSource(uint32_t tokens_per_epoch, uint32_t source_epoch_count)
{
    if (tokens_per_epoch == 0 || source_epoch_count == 0) {
        return false;
    }
    const uint64_t total = static_cast<uint64_t>(source_epoch_count) * tokens_per_epoch;
    return total <= kCampMaxTokens;
}

} // namespace inc::dc::dn::qv2::camp
