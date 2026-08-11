/**
 * INC Dispatch 完整两跳 persistent pipeline ABI（destination-major channel）。
 * D2.1 standalone 继续作 correctness reference，本 ABI 独立堆布局。
 */
#pragma once

#include <cstddef>
#include <cstdint>

#include "inc_dc_dn_queue_v2_abi.h"
#include "inc_dc_dn_transport_abi.h"

namespace inc::dc::dn::pl {

using inc::dc::dn::qv2::Qv2Descriptor;
using inc::dc::dn::qv2::Qv2HeadLine;
using inc::dc::dn::qv2::Qv2TailLine;
using inc::dc::dn::qv2::kQv2DescriptorBytes;
using inc::dc::dn::qv2::kQv2LaneCount;
using inc::dc::dn::qv2::kQv2PayloadBatchTokens;
using inc::dc::dn::qv2::kQv2PublishTileTokens;
using inc::dc::dn::qv2::kQv2RingDepth;
using inc::dc::dn::qv2::kQv2TokenBytes;
using inc::dc::dn::qv2::kQv2MaxTokenBytes;
using inc::dc::dn::qv2::Qv2PatternByte;
using inc::dc::dn::qv2::Qv2ChecksumSeed;

// Runtime Payload V2：对称环按最大 token 预留；lane/source 内 slot 按本次 payload_bytes 紧密排列
constexpr uint32_t kPlMaxTokenBytes = kQv2MaxTokenBytes;
constexpr uint32_t kPlDefaultTokenBytes = kQv2TokenBytes; // h4096 fp16 默认

inline bool PlCheckedMulU32(uint32_t a, uint32_t b, uint64_t *out)
{
    // u32×u32 总是落入 u64
    if (out != nullptr) {
        *out = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    }
    return true;
}

inline bool PlCheckedMulU64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (a != 0u && b > (~0ull / a)) {
        return false;
    }
    if (out != nullptr) {
        *out = a * b;
    }
    return true;
}

constexpr uint32_t kPlMagic = 0x504C5044u; // PLPD
constexpr uint32_t kPlMaxTokens = 4096u;
constexpr uint32_t kPlMaxSources = 8u;
// Candidate1：logical recv channel compile bound（物理 Worker 仍 ≤ kPlMaxSources）
constexpr uint32_t kPlMaxLogicalRecvChannels = 32u;
constexpr uint32_t kPlMaxRecvLanes = 8u; // 与 Worker recv AIV 块数一致（blocks 8–15）
// 机内最大：topk 可到 16（与 inc_dc_types.h kMaxTopk 对齐）；正式常用 {1,2,4,6,8,16}
constexpr uint32_t kPlMaxTopk = 16u;
constexpr uint32_t kPlFormalTokensPerSource = 512u;
// topk 路由元数据与 final 槽位上界（heap 可控；运行时 PlValidateHeap 校验）
constexpr uint32_t kPlMaxExpertIdsSlots = kPlMaxTokens * kPlMaxTopk;
// 默认 runtime final 槽位（未设 INC_DN_PL_FINAL_SLOT_CAPACITY 时兼容旧 4096 行为）
constexpr uint32_t kPlMaxFinalSlots = 4096u;
// compile-time heap 预留上界：机内最坏 8×4096×topk16 = 524288
// （显存：h4096≈4GiB/PE final 区，h12k≈12GiB/PE，相对 910 64GiB/die 仍有余量）
constexpr uint32_t kPlMaxFinalSlotsCompileBound = 524288u;
static_assert(kPlMaxFinalSlots <= kPlMaxFinalSlotsCompileBound,
              "default runtime capacity must fit compile bound");
static_assert(kPlMaxSources * kPlMaxTokens * kPlMaxTopk <= kPlMaxFinalSlotsCompileBound,
              "compile bound must cover max workers×tokens×topk");
// invocation 表：shape slot 容量（warmup+measure≤16 或单 slot 模板复用）
constexpr uint32_t kPlMaxInvocationEpochs = 16u;
constexpr uint32_t kPlInvocationFlagsHasRaggedSources = 1u;
// 单 shape 模板：slot 循环复用，go_epoch/generation 单调，禁止每 epoch H2D patch
constexpr uint32_t kPlInvocationFlagsTemplateSlot = 2u;
// P6 分段时序：PlRouteTimingLine 非正式；正式 makespan / transport 见下方开关
constexpr uint32_t kPlTimingFormalMakespan = 1u; // release_seen→global_done_publish（PE0）
constexpr uint32_t kPlTimingFormalTransport = 1u; // release_seen→global_transport_done（PE0，RecvDone 诊断）
constexpr uint32_t kPlTimingFormalSecondHopVisible = 1u; // release_seen→second_hop_visible（PE0，对齐 Native comm）
constexpr uint32_t kPlTimingFormalSegmentP6 = 0u; // PlRouteTimingLine 禁止 gate 采信

// descriptor / completion 错误码（device→host 可观测）
constexpr uint32_t kPlErrNone = 0u;
constexpr uint32_t kPlErrBadExpertId = 1u;
constexpr uint32_t kPlErrDestMismatch = 2u;
constexpr uint32_t kPlErrFutureDone = 3u;
constexpr uint32_t kPlErrStaleDone = 4u;
constexpr uint32_t kPlErrGenerationMismatch = 52u; // kGenerationMismatch
constexpr uint32_t kPlErrSegmentBaseReadyTimeout = 18u;
// M2 workspace descriptor fail-closed codes（device→host 可观测）
constexpr uint32_t kPlErrWorkspaceDescMissing = 53u;
constexpr uint32_t kPlErrWorkspaceGenerationMismatch = 54u;
constexpr uint32_t kPlErrWorkspacePointerAlignment = 55u;
constexpr uint32_t kPlErrOutputCapacityInsufficient = 56u;
constexpr uint32_t kPlErrWorkspaceFlagInvalid = 57u;
// Source-major epoch / transport protocol errors. Keep these distinct: host
// gates must be able to separate a recv data-plane regression from PE0's
// global completion observation.
constexpr uint32_t kPlErrEpochLocalHeadGtTail = 58u;
constexpr uint32_t kPlErrEpochLocalStagingOcc = 59u;
constexpr uint32_t kPlErrRecvIncomplete = 60u;
constexpr uint32_t kPlErrRecvTailEpochMissing = 61u;
constexpr uint32_t kPlErrRecvTailRegression = 62u;
constexpr uint32_t kPlErrEpochLocalCreditGtSeq = 63u;
constexpr uint32_t kPlErrEpochLocalEgressWait = 64u;
constexpr uint32_t kPlErrGlobalTransportMissing = 65u;
constexpr uint32_t kPlErrGlobalTransportPeerError = 66u;
constexpr uint32_t kPlErrRecvTailIncomplete = 67u;
constexpr uint32_t kPlErrRecvDescriptorIncomplete = 68u;
constexpr uint32_t kPlErrRecvPayloadVisibility = 69u;
constexpr uint32_t kPlErrRawMetaReadyWait = 70u;
constexpr uint32_t kPlErrRawMetaChecksum = 71u;
constexpr uint32_t kPlErrRawMetaDestMask = 72u;

// AIV 资源合同：Dispatch 与 Combine 无重叠
// Legacy U8/R8 固定布局常量（heap 上限 / 兼容诊断）；运行时用 PlWorkerServiceLayout
constexpr uint32_t kPlMaxUploadLanes = 8u;
constexpr uint32_t kPlWorkerUploadBlockBegin = 0u;
constexpr uint32_t kPlWorkerUploadBlockEnd = 7u;
constexpr uint32_t kPlWorkerRecvBlockBegin = 8u;
constexpr uint32_t kPlWorkerRecvBlockEnd = 15u;
constexpr uint32_t kPlWorkerControlBlock = 16u;
constexpr uint32_t kPlWorkerServiceBlockDim = 17u; // MAX = U8+R8+1；真 launch 可为更小
// Candidate2：recv_lane_count 打包进 pipeline_mode[31:24]（0=默认 R=8）
constexpr uint32_t kPlRecvLaneCountShift = 24u;
constexpr uint32_t kPlRecvLaneCountMask = 0xFF000000u;
constexpr uint32_t kPlCombineReserveBegin = 20u;
constexpr uint32_t kPlCombineReserveEnd = 39u;
constexpr uint32_t kPlDispatchBlockBegin = 0u;
constexpr uint32_t kPlDispatchBlockEnd = 19u;

// 测量语义：prepacked 诊断 / Stage1 / Full Dispatch 正式路径
constexpr uint32_t kPlMeasurementPrepackedTransport = 0u;
constexpr uint32_t kPlMeasurementStage1Diagnostic = 1u;
constexpr uint32_t kPlMeasurementFullDispatch = 2u;

constexpr uint32_t kPlIncForwardBlockBegin = 0u;
constexpr uint32_t kPlIncForwardBlockEnd = 7u;
constexpr uint32_t kPlIncControlBlock = 8u;
constexpr uint32_t kPlIncSecondHopAggBlock = 9u; // 第二跳完成聚合 AIV（与 full_done 解耦）
constexpr uint32_t kPlIncServiceBlockDim = 10u;  // legacy dest-major：blocks 0–9
// P5 source-major：20 AIV（Combine 仍从 20 起保留）
constexpr uint32_t kPlIncP5GatherBlockBegin = 0u;
constexpr uint32_t kPlIncP5GatherBlockEnd = 7u;
constexpr uint32_t kPlIncP5EgressBlockBegin = 8u;
constexpr uint32_t kPlIncP5EgressBlockEnd = 15u;
constexpr uint32_t kPlIncP5IngressReclaimBlock = 16u;
constexpr uint32_t kPlIncP5CountPrefixBlock = 17u;
constexpr uint32_t kPlIncP5CompletionSpareBegin = 18u;
constexpr uint32_t kPlIncP5SecondHopAggBlock = 19u;
constexpr uint32_t kPlIncP5ServiceBlockDim = 20u;

constexpr uint32_t kPlVerifyDeviceFull = 0u;
constexpr uint32_t kPlVerifyHostPost = 1u;

constexpr uint32_t kPlModeFullPipeline = 0u;
constexpr uint32_t kPlModeEgressOnly = 1u;
constexpr uint32_t kPlModeControlOnly = 2u; // 仅验证 mode3 控制面，不跑数据面
constexpr uint32_t kPlModeBaseMask = 0xFFu;
// P3：upload 前把 tile 收成 destination-major 连续 staging，再 range put（bit flag，可与 base mode OR）
constexpr uint32_t kPlModeUploadCompactGather = 0x100u;
// Count metadata 直接 Worker→destination Worker；完整本地矩阵仍发 partner INC 做 expected/checksum 校验。
// 只缩短控制面路径，不改变 payload 的 Worker→INC→destination 两跳契约。
constexpr uint32_t kPlModeDirectCountExchange = 0x200u;
// destination recv 用本地 canonical prefix + descriptor identity 重建 final_slot；
// source 不再等待 destination 回传 segment_base。
constexpr uint32_t kPlModeReceiverFinalSlot = 0x400u;
// P5：Worker source-major raw ingress（唯一 token 上传一次；INC 侧 route/gather）
constexpr uint32_t kPlIngressLayoutSourceMajorRaw = 0x1000u;
// P5.4：Worker control 将 route 编译为 destination CSR。INC gather AIV
// 只遍历自己的 slice，不再各自扫描完整 T×K metadata。
constexpr uint32_t kPlModeDestinationCsrRoutePlan = 0x2000u;

inline bool PlIngressLayoutIsSourceMajorRaw(uint32_t pipeline_mode)
{
    return (pipeline_mode & kPlIngressLayoutSourceMajorRaw) != 0u;
}

inline bool PlDestinationCsrRoutePlanEnabled(uint32_t pipeline_mode)
{
    return (pipeline_mode & kPlModeDestinationCsrRoutePlan) != 0u;
}

inline bool PlDirectCountExchangeEnabled(uint32_t pipeline_mode)
{
    return (pipeline_mode & kPlModeDirectCountExchange) != 0u;
}

constexpr uint32_t kPlCompletionModeCounterScan = 0u;
constexpr uint32_t kPlCompletionModeDoneLines = 1u;

constexpr uint32_t kPlHostCompletionSyncPoll = 0u;
constexpr uint32_t kPlHostCompletionAsyncProduction = 1u;

constexpr uint32_t kPlPayloadSourceStatic = 0u;      // 多 epoch 复用同一份 source payload
constexpr uint32_t kPlPayloadSourceEpochUnique = 1u; // 每 epoch 独立 payload 区（须预填校验）

constexpr uint32_t kPlKernelExitNone = 0u;
constexpr uint32_t kPlKernelExitStop = 1u;
constexpr uint32_t kPlKernelExitResource = 2u;
constexpr uint32_t kPlKernelExitUpload = 3u;
constexpr uint32_t kPlKernelExitForward = 4u;
constexpr uint32_t kPlKernelExitRecv = 5u;
constexpr uint32_t kPlKernelExitDestAgg = 6u;
constexpr uint32_t kPlKernelExitSecondHopAgg = 7u;

constexpr uint32_t kPlDescriptorBytes = 128;

// 两跳 pipeline 专用 128B descriptor（完整 Dispatch identity，cacheline 对齐）
struct PlDescriptor {
    uint64_t epoch;
    uint64_t generation;
    uint64_t lane_sequence;
    uint32_t source_rank;
    uint32_t source_token_id;
    uint32_t topk_slot;
    uint32_t expert_id;
    uint32_t destination_rank;
    uint32_t local_expert_id;
    uint32_t source_segment_offset;
    uint32_t destination_final_slot;
    uint32_t destination_slot;
    uint32_t payload_bytes;
    uint32_t source_lane;
    uint32_t ring_slot;
    uint32_t token_sequence; // legacy 全局序号
    uint32_t checksum_seed;
    uint8_t pad[48];
};
static_assert(sizeof(PlDescriptor) == kPlDescriptorBytes);

// P5.3：INC raw ingress chunk descriptor（128B）；每 upload-lane 独立 SPSC ring
constexpr uint32_t kPlRawChunkDescMagic = 0x50524348u; // PRCH
// One maximum-sized epoch (4096 tokens, 8 upload lanes, 32-token tiles)
// produces 16 generations per lane.  Keeping the whole epoch resident avoids
// making correctness depend on an in-epoch descriptor-wrap/reclaim race.  The
// runtime validator still rejects shapes whose required generation count is
// larger than this capacity.
constexpr uint32_t kPlRawChunkRingDepth = 16u;
constexpr uint32_t kPlRawUploadLaneCount = 8u;
constexpr uint32_t kPlRawChunkTargetBytes = 131072u;
struct PlRawChunkDesc {
    uint64_t epoch;
    uint64_t generation; // 1-based cumulative lane tile index（= published raw_tail）
    uint32_t source_rank;
    uint32_t token_begin;
    uint32_t token_count;
    uint32_t payload_bytes;
    uint64_t route_metadata_offset;
    uint32_t destination_present_mask;
    uint32_t deprecated_pending_route_mask; // P5.3：禁止作 reclaim 合同；保留布局
    uint32_t magic;
    uint32_t upload_lane;
    // P5.2 Truth：Worker 本地期望 checksum64（FNV-1a），INC 计时外对照
    uint64_t expect_payload_checksum64;
    uint64_t expect_meta_checksum64;
    uint8_t pad[56];
};
static_assert(sizeof(PlRawChunkDesc) == 128u);

// P5.3：per-lane / per-dest 独占 64B 控制线
struct PlRawLaneCtrlLine {
    uint64_t value; // tail 或 head（cumulative generation）
    uint64_t epoch;
    uint32_t magic;
    uint32_t lane_id;
    uint8_t pad[40];
};
static_assert(sizeof(PlRawLaneCtrlLine) == 64u);

// Worker control 单写者：整 epoch packed expert_ids 推到 partner INC 后发布。
// 独占 64B；禁止与 raw tail/head/desc/scratch 共 cacheline。
constexpr uint32_t kPlRawMetaReadyMagic = 0x504D5244u; // PMRD
struct PlRawMetaReadyLine {
    uint64_t epoch;
    uint32_t source_rank;
    uint32_t metadata_bytes;
    uint64_t metadata_checksum;
    uint32_t magic;
    int32_t ready_signal;
    uint8_t pad[32];
};
static_assert(sizeof(PlRawMetaReadyLine) == 64u);

// Worker control 唯一写者：epoch N semantic 完成后、返回 WaitStart 前发布。
// Host 发布 N+1 local_ready 前只等 retire_signal==N（禁止 cycle 推断 quiescent）。
constexpr uint32_t kPlControlEpochRetireMagic = 0x50434552u; // PCER
struct ControlEpochRetireLine {
    uint64_t epoch;
    uint64_t generation;
    uint64_t semantic_done_epoch;
    uint32_t post_epoch_cleanup_done;
    uint32_t error_code;
    uint32_t magic;
    uint32_t telemetry_valid; // 0=second-hop telemetry missing/incomplete; must not block retire
    uint64_t retire_signal;   // == completed epoch N
    uint8_t pad[16];
};
static_assert(sizeof(ControlEpochRetireLine) == 64u);

// Device/host 统一 route 视图（禁止各路径各自重算 eid 地址）
struct PlEpochRouteView {
    uint64_t eid_base;
    uint32_t token_count;
    uint32_t topk;
    uint32_t expert_per_pe;
    uint32_t moe_expert_num;
    uint32_t metadata_bytes;
    uint32_t source_rank;
    uint32_t ok; // 1=assertions held
};

struct PlRouteDoneGenLine {
    uint64_t done_generation; // 已完成的最高 raw tile generation（含 0-match）
    uint64_t epoch;
    uint32_t magic;
    uint32_t upload_lane;
    uint32_t destination;
    uint32_t pad0;
    uint8_t pad[32];
};
static_assert(sizeof(PlRouteDoneGenLine) == 64u);

struct PlStagingCtrlLine {
    uint64_t value; // tail（gather）或 head（egress）cumulative token seq
    uint64_t epoch;
    uint32_t magic;
    uint32_t destination;
    // gather→egress：仅 final flush 置 1；coalesce 中间发布必须为 0（防 raw_done 早退）
    uint32_t gather_complete;
    uint8_t pad[36];
};
static_assert(sizeof(PlStagingCtrlLine) == 64u);

constexpr uint32_t kPlRawLaneCtrlMagic = 0x50524C43u;   // PRLC
constexpr uint32_t kPlRouteDoneGenMagic = 0x50524447u;  // PRDG
constexpr uint32_t kPlStagingCtrlMagic = 0x50535443u;   // PSTC
// P5.3 故障注入（host→device 64B）；0=off
constexpr uint32_t kPlP53InjectMagic = 0x50353349u; // P53I
constexpr uint32_t kPlP53InjectEarlyTail = 1u;
constexpr uint32_t kPlP53InjectStaleGen = 2u;
constexpr uint32_t kPlP53InjectWithholdCredit = 4u;
constexpr uint32_t kPlP53InjectDropStagingTail = 8u;
// E6：预填旧 epoch 控制线（正例应 ignore + 完成）；同 epoch head>tail 协议错误另注
constexpr uint32_t kPlP53InjectPrefillStaleEpoch = 16u;
constexpr uint32_t kPlP53InjectSameEpochHeadGtTail = 32u;
// 仅诊断：终态 tail 重发不是正式协议路径；默认必须关闭。
constexpr uint32_t kPlP53InjectTerminalTailRepublish = 64u;
// 负例：Worker control 故意不发布 RawMetaReady → gather fail-closed error 70
constexpr uint32_t kPlP53InjectDropMetaReady = 128u;

// E5b：每 dest egress AIV 独占完成线；block19 等全部 active done_epoch==epoch
struct PlEgressDoneLine {
    uint64_t done_epoch;
    uint32_t dest_lane;
    uint32_t magic;
    uint64_t egress_seq_end;
    uint8_t pad[40];
};
static_assert(sizeof(PlEgressDoneLine) == 64u);
constexpr uint32_t kPlEgressDoneMagic = 0x50454744u; // PEGD
struct PlP53InjectLine {
    uint32_t flags;
    uint32_t magic;
    uint64_t epoch;
    uint8_t pad[48];
};
static_assert(sizeof(PlP53InjectLine) == 64u);

// P5.2：first_hop 计时外真值线（INC → Worker）；不参与 makespan
constexpr uint32_t kPlFirstHopVerifyMagic = 0x50464856u; // PFHV
struct PlFirstHopVerifyLine {
    uint64_t epoch;
    uint32_t source_rank;
    uint32_t tokens_checked;
    uint64_t payload_bytes_checked;
    uint64_t meta_bytes_checked;
    uint64_t payload_checksum64_obs;
    uint32_t payload_exact; // 1=match expect
    uint32_t meta_exact;
    uint32_t error_code;
    uint32_t magic;
    uint8_t pad[8];
};
static_assert(sizeof(PlFirstHopVerifyLine) == 64u);

struct PlPipelineDesc {
    uint32_t magic;
    uint32_t pair_id;
    uint32_t peer_pe;
    uint32_t my_role; // 0=worker 1=inc
    uint32_t worker_count;
    uint32_t lane_count;
    uint32_t tokens_per_epoch;
    uint32_t tokens_per_lane;
    uint32_t payload_bytes;
    uint32_t ring_depth;
    uint32_t batch_tokens;
    uint32_t tile_tokens;
    uint32_t head_tile_tokens;
    uint32_t transport_mode;
    uint32_t verify_mode;
    uint32_t pipeline_mode;
    uint64_t go_epoch;
    uint64_t d1_start_off;
    uint64_t d1_stop_off;
    uint32_t expected_tokens_per_dest;
    uint32_t payload_source_mode; // kPlPayloadSourceStatic / kPlPayloadSourceEpochUnique
    uint32_t completion_mode;     // kPlCompletionModeCounterScan / kPlCompletionModeDoneLines
    uint32_t host_completion_mode; // kPlHostCompletionSyncPoll / kPlHostCompletionAsyncProduction
    uint32_t measurement_mode;    // kPlMeasurement*
    uint32_t route_topk;
    uint32_t expert_per_pe;
    uint32_t moe_expert_num;
    // P1：byte-sized tile 目标（0=沿用固定 tile_tokens / head_tile_tokens）
    uint32_t upload_tile_target_bytes;
    uint32_t egress_publish_target_bytes;
};
static_assert(sizeof(PlPipelineDesc) == 128u);

// Full Dispatch 堆偏移与元数据（host 初始化，device 只读）
struct PlFullDispatchConfig {
    uint32_t magic;
    uint32_t version;
    uint32_t worker_count;
    uint32_t lane_count;
    uint64_t raw_input_off;
    uint64_t raw_expert_ids_off;
    uint64_t gather_payload_off;
    uint64_t gather_desc_off;
    uint64_t segment_base_off;
    uint64_t final_payload_off;
    uint64_t final_assist_off;
    uint64_t ep_recv_count_off;
    uint64_t expert_token_nums_off;
    uint64_t lane_work_off;
    uint64_t full_output_done_off;
    uint64_t route_timing_off;
    uint64_t first_hop_done_off;
    uint32_t probe_mode; // 0=full 1=count 2=gather 3=first_hop
    uint32_t input_epoch_count; // 预载 epoch 数；1=单 epoch（C1），>=2 用 stride 选片
};
static_assert(sizeof(PlFullDispatchConfig) == 128u);

// Host/device 同一映射（纯算术 inline，与 PlIngressLayoutIsSourceMajorRaw 同风格）
inline uint32_t PlRecvLaneOwner(uint32_t logical_channel, uint32_t recv_lane_count)
{
    return (recv_lane_count == 0u) ? 0u : (logical_channel % recv_lane_count);
}
inline uint32_t PlActiveRecvLaneCount(uint32_t recv_lane_count, uint32_t logical_recv_channel_count)
{
    if (recv_lane_count == 0u) {
        return 0u;
    }
    return (recv_lane_count < logical_recv_channel_count) ? recv_lane_count : logical_recv_channel_count;
}
inline uint32_t PlResolveRecvLaneCount(uint32_t configured, uint32_t fallback_lanes)
{
    if (configured == 0u) {
        return fallback_lanes > 0u ? fallback_lanes : kPlMaxRecvLanes;
    }
    return (configured > kPlMaxRecvLanes) ? kPlMaxRecvLanes : configured;
}
inline uint32_t PlResolveLogicalRecvChannelCount(uint32_t configured, uint32_t worker_count)
{
    if (configured == 0u) {
        return worker_count;
    }
    return (configured > kPlMaxLogicalRecvChannels) ? kPlMaxLogicalRecvChannels : configured;
}
// recv_map 运行时默认：R=kPlMaxRecvLanes，logical=worker_count（U8/R8）。
// 结构体已满 128B，不扩 PlFullDispatchConfig；host 单测覆盖 32-channel 映射。
// Candidate2：upload/recv/control 动态 block 布局（host+device 同口径）
struct PlWorkerServiceLayout {
    uint32_t upload_lane_count;
    uint32_t recv_lane_count;
    uint32_t upload_begin;  // 恒 0
    uint32_t recv_begin;    // U
    uint32_t control_block; // U+R
    uint32_t block_dim;     // U+R+1（真减 launch；禁止空转占 AIV）
};
inline uint32_t PlClampUploadLaneCount(uint32_t upload_lanes)
{
    if (upload_lanes == 0u) {
        return kPlMaxUploadLanes;
    }
    return (upload_lanes > kPlMaxUploadLanes) ? kPlMaxUploadLanes : upload_lanes;
}
inline uint32_t PlClampRecvLaneCountCfg(uint32_t recv_lanes)
{
    if (recv_lanes == 0u) {
        return kPlMaxRecvLanes;
    }
    return (recv_lanes > kPlMaxRecvLanes) ? kPlMaxRecvLanes : recv_lanes;
}
inline PlWorkerServiceLayout PlMakeWorkerServiceLayout(uint32_t upload_lanes, uint32_t recv_lanes)
{
    PlWorkerServiceLayout L{};
    L.upload_lane_count = PlClampUploadLaneCount(upload_lanes);
    L.recv_lane_count = PlClampRecvLaneCountCfg(recv_lanes);
    L.upload_begin = 0u;
    L.recv_begin = L.upload_lane_count;
    L.control_block = L.upload_lane_count + L.recv_lane_count;
    L.block_dim = L.control_block + 1u;
    return L;
}
inline uint32_t PlPackRecvLaneCountIntoPipelineMode(uint32_t pipeline_mode, uint32_t recv_lanes)
{
    const uint32_t r = PlClampRecvLaneCountCfg(recv_lanes);
    return (pipeline_mode & ~kPlRecvLaneCountMask) | (r << kPlRecvLaneCountShift);
}
inline uint32_t PlUnpackRecvLaneCountFromPipelineMode(uint32_t pipeline_mode)
{
    const uint32_t r = (pipeline_mode & kPlRecvLaneCountMask) >> kPlRecvLaneCountShift;
    return (r == 0u) ? kPlMaxRecvLanes : PlClampRecvLaneCountCfg(r);
}
inline PlWorkerServiceLayout PlWorkerServiceLayoutFromDescFields(uint32_t lane_count, uint32_t pipeline_mode)
{
    return PlMakeWorkerServiceLayout(lane_count, PlUnpackRecvLaneCountFromPipelineMode(pipeline_mode));
}
inline uint32_t PlWorkerResidentExpectMask(uint32_t block_dim)
{
    if (block_dim == 0u || block_dim > 31u) {
        return 0u;
    }
    return (block_dim >= 32u) ? 0xffffffffu : ((1u << block_dim) - 1u);
}

// Runtime Payload ABI V2 layout_version（与 V1 h4096 字节布局不兼容混用）
constexpr uint32_t kPlLayoutVersionV1Fixed8192 = 1u;
constexpr uint32_t kPlLayoutVersionRuntimePayloadV2 = 2u;
constexpr uint32_t kPlDtypeBytesFp16 = 2u;
constexpr uint32_t kPlAllowedPayloadBytes[] = {4096u, 8192u, 12288u, 14336u, 16384u, 24576u};
constexpr uint32_t kPlAllowedHiddenFp16[] = {2048u, 4096u, 6144u, 7168u, 8192u, 12288u};
constexpr uint32_t kPlAllowedPayloadCount =
    static_cast<uint32_t>(sizeof(kPlAllowedPayloadBytes) / sizeof(kPlAllowedPayloadBytes[0]));

inline bool PlPayloadBytesAllowed(uint32_t payload_bytes)
{
    for (uint32_t i = 0; i < kPlAllowedPayloadCount; ++i) {
        if (kPlAllowedPayloadBytes[i] == payload_bytes) {
            return true;
        }
    }
    return false;
}

inline bool PlHiddenFp16Allowed(uint32_t hidden_size)
{
    for (uint32_t i = 0; i < kPlAllowedPayloadCount; ++i) {
        if (kPlAllowedHiddenFp16[i] == hidden_size) {
            return true;
        }
    }
    return false;
}

inline bool PlRuntimePayloadCrossCheck(uint32_t hidden_size, uint32_t dtype_bytes, uint32_t payload_bytes)
{
    if (dtype_bytes != kPlDtypeBytesFp16) {
        return false;
    }
    if (!PlHiddenFp16Allowed(hidden_size) || !PlPayloadBytesAllowed(payload_bytes)) {
        return false;
    }
    if ((payload_bytes % 32u) != 0u) {
        return false;
    }
    return payload_bytes == hidden_size * dtype_bytes;
}

// 每 epoch 动态 shape（160B V2）；所有 PE 同 epoch 视图必须一致
struct PlInvocationDesc {
    uint64_t epoch;
    uint64_t generation;
    uint32_t worker_count;
    uint32_t route_topk;
    uint32_t source_token_count[kPlMaxSources]; // 有效 token；可为 0
    uint32_t total_input_tokens;
    uint32_t total_route_instances;
    uint32_t final_slot_capacity;   // runtime contract；与 source×topk 解耦
    uint32_t local_source_capacity; // 每 source 文件/堆 stride（≥ max valid）
    uint64_t raw_input_epoch_stride_bytes;
    uint64_t raw_expert_ids_epoch_stride_bytes;
    uint32_t workspace_generation;
    uint32_t flags;
    uint64_t final_payload_offset; // runtime offset（≤ compile-time 安全边界）
    uint64_t final_assist_offset;
    uint32_t descriptor_capacity;
    uint32_t expert_ids_capacity;
    uint64_t workspace_bytes;
    // Runtime Payload V2（必须进入 descriptor，禁止只存在于 host 局部变量）
    uint32_t layout_version; // kPlLayoutVersionRuntimePayloadV2
    uint32_t hidden_size;
    uint32_t dtype_bytes;  // 仅允许 2 (fp16)
    uint32_t payload_bytes; // == hidden_size * dtype_bytes，且 ∈ allowed set
    uint8_t pad_v2[16];
};
static_assert(sizeof(PlInvocationDesc) == 160u);

constexpr uint32_t kPlWorkspaceContractMagic = 0x504C5743u; // PLWC
constexpr uint32_t kPlInvocationWorkspaceMagic = 0x504C4957u; // PLIW

// M2：Worker 本地 invocation workspace（caller-owned 或 pooled）。
// 指针仅所属 Worker 的 AIV / host 可解引用；remote PE 禁止把这些 ptr 当 SHMEM put/get target。
// 远端通信仍只使用固定 symmetric transport offsets（Inc*/DestChannel*/control）。
constexpr uint32_t kPlInvocationWorkspaceFlagCallerOwned = 1u;
constexpr uint32_t kPlInvocationWorkspaceFlagPooled = 2u;
constexpr uint32_t kPlInvocationWorkspaceFlagLegacySymmetricFinal = 4u; // 过渡：仍写 symmetric DestFinal*
// M3：INC put → DestChannel ring；Worker 本地 MTE → expand_x_ptr/assist_info_ptr（可暂指向 DestFinal 区）
constexpr uint32_t kPlInvocationWorkspaceFlagWorkerLocalFinal = 8u;

struct PlInvocationWorkspaceDesc {
    uint32_t magic; // kPlInvocationWorkspaceMagic
    uint32_t flags;
    uint32_t source_token_capacity;
    uint32_t output_route_capacity;
    uint32_t topk;
    uint32_t generation;
    uint64_t input_ptr;             // Worker-local / caller-owned x
    uint64_t expert_ids_ptr;        // Worker-local / caller-owned
    uint64_t expand_x_ptr;          // Worker-local final payload (expand_x)
    uint64_t assist_info_ptr;       // Worker-local assist
    uint64_t ep_recv_count_ptr;     // Worker-local
    uint64_t expert_token_nums_ptr; // Worker-local
    uint64_t input_bytes;
    uint64_t output_bytes;
    uint64_t assist_bytes;
    uint32_t layout_version;
    uint32_t hidden_size;
    uint32_t dtype_bytes;
    uint32_t payload_bytes;
    uint8_t pad[16];
};
static_assert(sizeof(PlInvocationWorkspaceDesc) == 128u);

// Host session 级 workspace/capacity ABI（与 PlInvocationDesc 字段对齐，便于预检与负例）
struct PlWorkspaceContract {
    uint32_t magic;
    uint32_t workspace_generation;
    uint32_t source_capacity;
    uint32_t route_topk;
    uint32_t final_slot_capacity;
    uint32_t descriptor_capacity;
    uint32_t expert_ids_capacity;
    uint32_t worker_count;
    uint64_t final_payload_offset;
    uint64_t final_assist_offset;
    uint64_t workspace_bytes;
    uint64_t ep_recv_count_offset;
    uint32_t flags;
    uint32_t layout_version;
    uint32_t hidden_size;
    uint32_t dtype_bytes;
    uint32_t payload_bytes;
    uint8_t pad[44];
};
static_assert(sizeof(PlWorkspaceContract) == 128u);

constexpr uint32_t kPlFullDispatchProbeFull = 0u;
constexpr uint32_t kPlFullDispatchProbeCountOnly = 1u;
constexpr uint32_t kPlFullDispatchProbeGatherOnly = 2u;
constexpr uint32_t kPlFullDispatchProbeFirstHop = 3u;

constexpr uint64_t kPlFullDispatchConfigOff = 128u;

struct PlRouteCountLine {
    uint64_t epoch;
    uint32_t destination_rank;
    uint32_t local_expert_id;
    uint32_t source_rank;
    uint32_t count;
    uint64_t ready_epoch; // 必须 == epoch；仅数据 quiet 后单独 doorbell 置位
    uint32_t checksum;
    uint8_t pad[28];
};
static_assert(sizeof(PlRouteCountLine) == 64u);

// Worker→partner INC：整片 count matrix 传完后的 doorbell
struct PlSourceCountReadyLine {
    uint64_t epoch;
    uint32_t source_rank;
    uint32_t total_routes;
    uint32_t checksum;
    uint32_t magic;
    uint64_t ready_epoch; // == epoch
    int32_t visible_signal; // putmem_signal doorbell：data 可见后 SET 为 (int32_t)epoch
    uint8_t pad[28];
};
static_assert(sizeof(PlSourceCountReadyLine) == 64u);
constexpr uint32_t kPlSourceCountReadySignalOff = offsetof(PlSourceCountReadyLine, visible_signal);
static_assert(kPlSourceCountReadySignalOff == 32u);

// partner INC→destination：某 source 的 dest slice 传完后的 doorbell
struct PlDestCountSliceReadyLine {
    uint64_t epoch;
    uint64_t ready_epoch; // == epoch
    uint32_t source_rank;
    uint32_t destination_rank;
    uint32_t total_routes;
    uint32_t checksum;
    // 单 source→destination 的 8 expert counts；每项 <= source tokens (compile bound 4096)。
    // direct-count 模式由此 64B line 自包含 count slice，旧模式仍校验 PlRouteCountLine。
    uint16_t expert_counts[8];
    uint32_t magic;
    int32_t visible_signal; // putmem_signal doorbell：data 可见后 SET 为 (int32_t)epoch
    uint8_t pad[8];
};
static_assert(sizeof(PlDestCountSliceReadyLine) == 64u);
constexpr uint32_t kPlDestCountSliceReadySignalOff = offsetof(PlDestCountSliceReadyLine, visible_signal);
static_assert(kPlDestCountSliceReadySignalOff == 52u);

struct PlSegmentBaseReadyLine {
    uint64_t epoch;
    uint32_t source_rank;
    uint32_t destination_rank;
    uint32_t checksum;
    uint32_t magic;
    uint64_t ready_epoch; // == epoch
    int32_t visible_signal; // putmem_signal doorbell：data 可见后 SET 为 (int32_t)epoch
    uint8_t pad[28];
};
static_assert(sizeof(PlSegmentBaseReadyLine) == 64u);
constexpr uint32_t kPlSegBaseReadySignalOff = offsetof(PlSegmentBaseReadyLine, visible_signal);
static_assert(kPlSegBaseReadySignalOff == 32u);

// 每 source recv 独占完成线；full_done 仅 control 写
struct PlRecvDoneLine {
    uint64_t epoch;
    uint32_t source_rank;
    uint32_t expected;
    uint32_t received;
    uint32_t final_head;
    uint32_t error_code;
    uint32_t magic;
    uint32_t last_remote_tail;
    uint32_t last_descriptor_seq;
    uint8_t pad[24];
};
static_assert(sizeof(PlRecvDoneLine) == 64u);

// P5 cell accounting is split by ownership: destination route/recv and INC
// gather/egress never perform shared RMW on one cache line.
struct PlP5CellRouteRecvLedgerLine {
    uint64_t epoch;
    uint32_t source;
    uint32_t destination;
    uint32_t route_expected;
    uint32_t remote_tail_observed;
    uint32_t recv_seen;
    uint32_t recv_expected;
    uint32_t last_descriptor_seq;
    uint32_t error_code;
    uint32_t magic;
    uint8_t pad[20];
};
static_assert(sizeof(PlP5CellRouteRecvLedgerLine) == 64u);

// Gather 独占：staging_final_gen。禁止与 egress 共 cacheline（RMW/DCCI 会覆盖）。
struct PlP5CellGatherLedgerLine {
    uint64_t epoch;
    uint32_t source;
    uint32_t destination;
    uint32_t staging_final_gen;
    uint32_t error_code;
    uint32_t magic;
    uint8_t pad[36];
};
static_assert(sizeof(PlP5CellGatherLedgerLine) == 64u);

struct PlP5CellEgressLedgerLine {
    uint64_t epoch;
    uint32_t source;
    uint32_t destination;
    uint32_t reserved_do_not_use_staging; // staging 已迁至 GatherLedgerLine
    uint32_t egress_seq_end;
    uint32_t normal_tail_publish_count;
    uint32_t terminal_republish_count;
    uint32_t last_published_tail;
    uint32_t error_code;
    uint64_t first_publish_cycle;
    uint64_t last_publish_cycle;
    uint32_t magic;
    uint8_t pad[4];
};
static_assert(sizeof(PlP5CellEgressLedgerLine) == 64u);

constexpr uint32_t kPlP5CellLedgerMagic = 0x504C434Cu; // PLCL

constexpr uint32_t kPlCountReadyMagic = 0x504C4352u; // PLCR
constexpr uint32_t kPlSegBaseReadyMagic = 0x504C5342u; // PLSB
constexpr uint32_t kPlRecvDoneMagic = 0x504C5244u;     // PLRD
constexpr uint32_t kPlTransportDoneMagic = 0x504C5444u;       // PLTD
constexpr uint32_t kPlGlobalTransportDoneMagic = 0x504C4754u; // PLGT
constexpr uint32_t kPlTransportTimingMagic = 0x504C5454u;     // PLTT

// 正式 transport 完成：每 destination 独占；不等待 four-output compact / full_done
struct PlTransportDoneLine {
    uint64_t epoch;
    uint32_t destination_rank;
    uint32_t expected_routes;
    uint32_t received_routes;
    uint32_t error_code;
    uint64_t transport_done_cycle; // destination control 本地 cycle（诊断）
    uint32_t magic;
    uint8_t pad[28];
};
static_assert(sizeof(PlTransportDoneLine) == 64u);

// PE0 control 观察齐全部 destination transport_done 后发布
struct PlGlobalTransportDoneLine {
    uint64_t epoch;
    uint64_t global_transport_done_epoch;
    uint64_t global_transport_done_cycle;
    uint32_t slowest_destination;
    uint32_t magic;
    uint8_t pad[32];
};
static_assert(sizeof(PlGlobalTransportDoneLine) == 64u);

// 正式 transport timing：PE0 control AIV 单 writer（禁止复用 PlRouteTimingLine）
struct PlTransportTimingLine {
    uint64_t timing_epoch;
    uint64_t release_seen_cycle;              // 拷自 PE0 D1LeaderTiming
    uint64_t global_transport_done_cycle;     // PE0 观察齐后打点
    uint32_t slowest_destination;             // PE0 最晚观察到的 dest
    uint32_t owner_block_id;
    uint32_t magic;
    uint32_t worker_count;
    uint64_t per_dest_transport_done_cycle[kPlMaxSources]; // PE0 首次观察到各 dest 的 cycle
    // 失败时 PE0 单 writer 的有界控制面快照；成功时为 0。
    uint32_t failure_stage;
    uint32_t error_rank;
    uint32_t error_destination;
    uint32_t first_error_code;
    uint32_t missing_destination_mask;
    uint32_t global_wait_spins;
};
static_assert(sizeof(PlTransportTimingLine) == 128u);

// P10 strict max-rank timing：每 physical PE 独占 64B；本地 clock 起止，host getmem 聚合
constexpr uint32_t kPlRankDispatchTimingMagic = 0x504C5254u; // PLRT
constexpr uint32_t kPlMaxPhysicalPe = 16u;
constexpr uint32_t kPlRoleMaskWorker = 1u;
constexpr uint32_t kPlRoleMaskInc = 2u;
constexpr uint32_t kPlRoleMaskPe0Observer = 4u;

// 字段重排避免 role_mask 后自然对齐膨胀；rank/role_mask 用 u16（PE≤15、role 位掩码）
struct PlRankDispatchTimingLine {
    uint64_t epoch;
    uint64_t local_start_cycle;
    uint64_t upload_done_cycle;
    uint64_t forward_done_cycle;
    uint64_t destination_done_cycle;
    uint64_t local_completion_cycle;
    uint32_t generation;
    uint16_t rank;
    uint16_t role_mask;
    uint32_t error_code;
    uint32_t magic;
};
static_assert(sizeof(PlRankDispatchTimingLine) == 64u);

// 第二跳 egress 可见：INC×dest 独占 64B（最大 8×8）；payload/descriptor/ready 已发布，不要求 destination recv
struct PlEgressVisibleDoneLine {
    uint64_t epoch;
    uint32_t generation;
    uint32_t inc_rank;
    uint32_t destination_rank;
    uint32_t expected_routes;
    uint32_t forwarded_routes;
    uint64_t tail_value; // 该 dest 第二跳 egress tail（quiet 后）
    uint64_t visible_done_cycle; // lane 相对 delta（PE0：release + delta）；非跨 PE 绝对时钟
    uint32_t error_code;
    uint32_t checksum;
    uint32_t magic;
    uint8_t pad[4];
};
static_assert(sizeof(PlEgressVisibleDoneLine) == 64u);
using PlEgressLaneVisibleDoneLine = PlEgressVisibleDoneLine;

constexpr uint32_t kPlEgressVisibleDoneMagic = 0x504C4556u;       // PLEV
constexpr uint32_t kPlSecondHopVisibleTimingMagic = 0x504C5356u;   // PLSV
constexpr uint32_t kPlGlobalSecondHopVisibleMagic = 0x504C4753u;   // PLGS
constexpr uint32_t kPlTimingNegMagic = 0x504C544Eu;               // PLTN

// 正式 second_hop_visible timing：PE0 control 单 writer
struct PlSecondHopVisibleTimingLine {
    uint64_t timing_epoch;
    uint64_t release_seen_cycle;
    uint64_t inc_second_hop_visible_done_cycle;
    uint32_t slowest_cell_inc;
    uint32_t slowest_cell_dest;
    uint32_t owner_block_id;
    uint32_t magic;
    uint32_t worker_count;
    uint32_t pad_align; // 对齐至 64B 边界
    uint64_t per_dest_visible_done_cycle[kPlMaxSources]; // 各 dest 最晚 cell cycle（诊断）
    uint8_t pad[16];
};
static_assert(sizeof(PlSecondHopVisibleTimingLine) == 128u);

struct PlGlobalSecondHopVisibleDoneLine {
    uint64_t epoch;
    uint64_t global_second_hop_visible_epoch;
    uint64_t global_second_hop_visible_cycle;
    uint32_t slowest_inc;
    uint32_t slowest_dest;
    uint32_t magic;
    uint8_t pad[28];
};
static_assert(sizeof(PlGlobalSecondHopVisibleDoneLine) == 64u);

constexpr uint32_t kPlIncSecondHopSummaryMagic = 0x504C4953u; // PLIS

// INC block9 聚合后 push 到 PE0：每 INC 独占 64B
struct PlIncSecondHopSummaryLine {
    uint64_t epoch;
    uint32_t generation;
    uint32_t inc_rank;
    uint32_t total_expected;
    uint32_t total_forwarded;
    uint32_t error_code;
    uint32_t magic;
    uint64_t inc_aggregated_cycle;
    uint32_t dest_cells_ready;
    uint8_t pad[20];
};
static_assert(sizeof(PlIncSecondHopSummaryLine) == 64u);

constexpr uint32_t kPlSecondHopAggTelemetryMagic = 0x504C5341u; // PLSA

// block9 专用遥测：超时/校验失败禁止静默丢弃
struct PlSecondHopAggTelemetryLine {
    uint64_t epoch;
    uint32_t inc_rank;
    uint32_t block_id;
    uint32_t spin_count;
    uint32_t error_code;
    uint32_t cells_missing;
    uint32_t magic;
    uint8_t pad[32];
};
static_assert(sizeof(PlSecondHopAggTelemetryLine) == 64u);

// 负例 ladder 钩子：host 从 env 写入，device 只读
struct PlTimingNegLine {
    uint32_t delay_second_hop_cycles;
    uint32_t delay_recv_cycles;
    uint32_t delay_finalize_cycles;
    uint32_t drop_egress_cell;
    uint32_t magic;
    // H7-E：PublishGlobalTransport 内 second_hop 阻塞自旋上限（H8.0 后 formal 路径已不再使用）
    uint32_t transport_second_hop_wait_spins;
    // H8.1：故意不发 SourceCountReady doorbell（fail-closed 负例）
    uint32_t drop_source_count_ready; // 1=启用
    uint32_t drop_source_rank;
    uint32_t drop_source_epoch; // 目标 go_epoch
    uint8_t pad[28];
};
static_assert(sizeof(PlTimingNegLine) == 64u);

struct PlSegmentBaseLine {
    uint64_t epoch;
    uint32_t destination_rank;
    uint32_t local_expert_id;
    uint32_t source_rank;
    uint32_t final_base;
    uint32_t count;
    uint64_t ready_epoch;
    uint8_t pad[24];
};
static_assert(sizeof(PlSegmentBaseLine) == 64u);

struct PlLaneWorkLine {
    uint64_t epoch;
    uint32_t lane_id;
    uint32_t token_count;
    uint32_t gather_count;
    uint32_t upload_count;
    uint64_t done_epoch;
    uint32_t error_code;
    uint8_t pad[28];
};
static_assert(sizeof(PlLaneWorkLine) == 64u);

struct PlFullOutputDoneLine {
    uint64_t stage1_done_epoch;
    uint64_t counts_done_epoch;
    uint64_t assist_done_epoch;
    uint64_t payload_done_epoch;
    uint64_t full_done_epoch;
    uint32_t error_code;
    uint32_t stale_done_observation_count;
    uint32_t future_done_observation_count;
    uint32_t generation_mismatch_count;
    // DestCountPrefix 超时：仍缺的 source 位图（bit s = source s）
    uint32_t missing_source_mask;
    // SegmentBaseReady 超时：仍缺的 dest 位图（bit d = destination d）
    uint32_t missing_segment_dest_mask;
};
static_assert(sizeof(PlFullOutputDoneLine) == 64u);

// count-path 单写者 trace stage（Worker / INC / Dest 各独占 64B×N）
constexpr uint32_t kPlCountTraceMagic = 0x504C4354u; // PLCT
constexpr uint32_t kPlCountStageWorkerEnter = 1u;
constexpr uint32_t kPlCountStageWorkerMatrixReady = 2u;
constexpr uint32_t kPlCountStageWorkerMatrixQuiet = 3u;
constexpr uint32_t kPlCountStageWorkerSourceReady = 4u;
constexpr uint32_t kPlCountStageIncEnter = 10u;
constexpr uint32_t kPlCountStageIncSourceReadySeen = 11u;
constexpr uint32_t kPlCountStageIncMatrixOk = 12u;
constexpr uint32_t kPlCountStageIncDestPublished = 13u;
constexpr uint32_t kPlCountStageDestWaitEnter = 20u;
constexpr uint32_t kPlCountStageDestAllSlices = 21u;
constexpr uint32_t kPlCountStageDestPrefixDone = 22u;
constexpr uint32_t kPlCountStageDestCountsDone = 23u;

// 每 source 一条：Worker control 独占写
struct PlWorkerSourceCountTrace {
    uint64_t epoch;
    uint32_t generation;
    uint32_t stage;
    uint32_t source_rank;
    uint32_t total_routes;
    uint32_t error_code;
    uint32_t magic;
    uint64_t enter_cycle;
    uint64_t exit_cycle;
    uint32_t observed_epoch;
    uint32_t source_publish_owner_block; // lane0=0 on success publish
    uint32_t source_publish_seq;         // +1 per successful publish
    uint32_t pad1;
};
static_assert(sizeof(PlWorkerSourceCountTrace) == 64u);

// 每 source（pair）一条：INC control 独占写；destination_publish_mask 记录已 push 的 dest
struct PlIncSourceCountTrace {
    uint64_t epoch;
    uint32_t generation;
    uint32_t stage;
    uint32_t source_rank;
    uint32_t destination_publish_mask;
    uint32_t source_ready_seen;
    uint32_t error_code;
    uint64_t enter_cycle;
    uint64_t exit_cycle;
    uint32_t observed_epoch;
    uint32_t observed_magic;
    uint64_t pad1;
};
static_assert(sizeof(PlIncSourceCountTrace) == 64u);

// 每 destination 一条：Dest control 独占写；source_ready_mask 记录已见 slice
struct PlDestinationCountTrace {
    uint64_t epoch;
    uint32_t generation;
    uint32_t stage;
    uint32_t destination_rank;
    uint32_t source_ready_mask;
    uint32_t missing_source_mask;
    uint32_t error_code;
    uint64_t enter_cycle;
    uint64_t exit_cycle;
    uint32_t observed_epoch;
    uint32_t pad_reserved;
    uint64_t wait_done_cycle; // ready_mask 齐或 timeout 时；host 算 dest_wait_us vs dest_segment_us
};
static_assert(sizeof(PlDestinationCountTrace) == 64u);

constexpr uint32_t kPlUploadLaneStageMagic = 0x504C554Cu; // PLUL

// 每 upload lane 独占 128B：正式 stage breakdown（禁止复用 PlRouteTimingLine / gather 多写者）
struct PlUploadLaneStageTiming {
    uint64_t epoch;
    uint32_t magic;
    uint32_t owner_block;
    uint64_t counts_wait_start;
    uint64_t counts_wait_end;
    uint64_t segment_ready_wait_start;
    uint64_t segment_ready_wait_end;
    uint64_t route_scan_start;
    uint64_t route_scan_end;
    uint64_t descriptor_build_start;
    uint64_t descriptor_build_end;
    uint64_t credit_wait_cycles;
    uint32_t accepted_routes;
    uint32_t segment_base_dcci_count;
    uint8_t pad[32];
};
static_assert(sizeof(PlUploadLaneStageTiming) == 128u);

constexpr uint32_t kPlDestCompletionTraceMagic = 0x504C4443u; // PLDC
constexpr uint32_t kPlRecvChanCompletionTraceMagic = 0x504C5243u; // PLRC
constexpr uint32_t kPlIncFwdLaneStageMagic = 0x504C4946u; // PLIF

// 每 destination Worker control 独占 128B（RecvDone 聚合 → pair_done publish）
struct PlDestinationCompletionTrace {
    uint64_t epoch;
    uint32_t generation;
    uint32_t destination_rank;
    uint32_t expected_total;
    uint32_t received_total;
    uint32_t recv_done_mask;
    uint32_t error_code;
    uint64_t counts_done_cycle;
    uint64_t first_recv_cycle;
    uint64_t last_recv_done_cycle;
    uint64_t all_recv_done_cycle;
    uint64_t full_done_publish_cycle;
    uint64_t pair_done_publish_start_cycle;
    uint64_t pair_done_publish_end_cycle;
    // H5：control 阶段锚点（同 PE delta；PE0 才有 global_transport_wait）
    uint64_t transport_done_local_cycle;
    uint64_t global_transport_wait_start;
    uint64_t global_transport_wait_end;
    uint32_t magic;
    uint8_t pad[12];
};
static_assert(sizeof(PlDestinationCompletionTrace) == 128u);

// 每 (dest, source) recv AIV 独占 256B（H5 阶段总量 accumulator）
struct PlRecvChannelCompletionTrace {
    uint64_t epoch;
    uint32_t source_rank;
    uint32_t expected;
    uint64_t first_tail_seen_cycle;
    uint64_t last_tail_seen_cycle;
    uint64_t first_descriptor_cycle;
    uint64_t last_descriptor_cycle;
    uint64_t assist_done_cycle;
    uint64_t last_credit_publish_cycle;
    uint64_t recv_done_cycle;
    uint64_t tail_poll_cycles;
    uint32_t descriptor_retry_count;
    uint32_t head_publish_count;
    uint32_t head_quiet_count;
    uint32_t tail_signal_wake_count;
    // H5：本 epoch 累计 cycle（禁止跨 PE 相减）
    uint64_t counts_done_wait_cycles;
    uint64_t wait_first_tail_cycles;
    uint64_t tail_wait_total_cycles;
    uint64_t descriptor_read_verify_cycles;
    uint64_t payload_assist_ready_cycles;
    uint64_t head_credit_publish_cycles;
    uint64_t recv_done_publish_cycles;
    uint32_t token_count;
    uint32_t stale_count;
    uint32_t duplicate_count;
    uint32_t lost_count;
    uint32_t overwrite_count;
    uint32_t magic;
    // D1：LocalFinal 真实 device counters（禁止 host 按 token 推导）
    uint64_t local_copy_issue_count;
    uint64_t local_copy_complete_count;
    uint64_t local_copy_bytes;
    uint64_t local_copy_cycles;
    uint64_t ring_payload_dcci_count;
    uint64_t ring_payload_dcci_bytes;
    uint64_t ring_payload_dcci_cycles;
    uint64_t output_dcci_count;
    uint64_t output_dcci_bytes;
    uint64_t output_dcci_cycles;
    uint64_t assist_write_count;
    uint64_t assist_write_cycles;
    uint64_t ready_publish_count;
    uint64_t ready_publish_cycles;
    uint64_t credit_publish_count;
    uint64_t credit_publish_cycles;
    // overlap / stage markers（D4）
    uint64_t first_local_copy_cycle;
    uint64_t last_local_copy_cycle;
    uint64_t first_credit_publish_cycle_lf;
    uint32_t local_copy_before_final_forward_count;
    uint32_t pad_lf0;
    uint8_t pad[48];
};
static_assert(sizeof(PlRecvChannelCompletionTrace) == 384u);

// 每 INC forward lane AIV 独占 256B（H5 阶段总量 accumulator）
struct PlIncForwardLaneStageTiming {
    uint64_t epoch;
    uint64_t first_ingress_tail_cycle;
    uint64_t last_ingress_cycle;
    uint64_t credit_wait_start;
    uint64_t credit_wait_end;
    uint64_t payload_issue_start;
    uint64_t payload_issue_end;
    uint64_t tail_publish_start;
    uint64_t tail_publish_end;
    uint64_t ingress_head_publish_end;
    uint64_t forward_done_cycle;
    uint64_t egress_credit_wait_cycles;
    uint32_t egress_tail_publish_count;
    uint32_t batch_count;
    uint64_t batch_tokens;
    uint32_t lane_id;
    uint32_t pad_align;
    uint64_t egress_tail_publish_cycles_sum;
    // H5：本 epoch 累计 cycle / counts
    uint64_t wait_ingress_tail_cycles;
    uint64_t descriptor_validation_cycles;
    uint64_t payload_descriptor_issue_cycles;
    uint64_t mte_completion_cycles;
    uint64_t data_drain_quiet_cycles;
    uint64_t ingress_head_publish_cycles;
    uint32_t payload_put_count;
    uint32_t descriptor_put_count;
    uint32_t drain_quiet_count;
    uint32_t ingress_head_publish_count;
    uint32_t max_batch;
    uint32_t magic;
    // P8.2 publish amortization 遥测（占用原 pad）
    uint32_t publish_group_count;
    uint32_t publish_group_tokens;
    uint32_t publish_group_max_tokens;
    uint32_t partial_publish_group_count;
    uint32_t forced_wrap_flush_count;
    uint32_t forced_credit_flush_count;
    uint32_t upstream_empty_flush_count;
    uint32_t final_flush_count;
    uint32_t pending_group_high_watermark;
    uint32_t first_tail_early_flush_count; // T1-A：首批 <tile_pub 提前 publish 次数
    uint8_t pad[12];
};
static_assert(sizeof(PlIncForwardLaneStageTiming) == 256u);

// INC IncForwardCounts 显式状态（禁止失败后静默进下一 epoch）
enum PlCountForwardStatus : uint32_t {
    kPlCountFwdOk = 0u,
    kPlCountFwdSourceReadyTimeout = 11u,
    kPlCountFwdSourceIdentityError = 12u,
    kPlCountFwdCountMatrixError = 13u,
    kPlCountFwdDestPublishError = 17u,
};

// P6 非正式：多 AIV 并发写 gather_*；gather_done 非全局 max，禁止作 Stage1 终点
struct PlRouteTimingLine {
    // timing_epoch 必须落在首个 64B（PlDcci 按 line 刷），供 gather lane 可见
    uint64_t timing_epoch;
    uint64_t route_count_start_cycle;
    uint64_t route_count_done_cycle;
    uint64_t prefix_done_cycle;
    uint64_t gather_first_cycle;
    uint64_t gather_done_cycle; // 诊断：末写 lane 局部值，非 max
    uint64_t stage1_done_cycle; // 非正式：保持 0；正式 Stage1 用 D1 leader + AIV aggregate
    uint64_t finalize_done_cycle;
    uint64_t full_done_cycle;
    uint8_t pad[56];
};
static_assert(sizeof(PlRouteTimingLine) == 128u);

// single-owner 正式分段时序骨架（堆未挂载；control AIV 独占 finalize/global）
struct PlRouteTimingFormalOwner {
    uint64_t timing_epoch;
    uint64_t route_count_done_cycle;
    uint64_t prefix_done_cycle;
    uint64_t gather_done_max_cycle;
    uint64_t finalize_done_cycle;
    uint64_t full_done_cycle;
    uint32_t magic;
    uint32_t owner_block_id;
    uint8_t pad[72];
};
static_assert(sizeof(PlRouteTimingFormalOwner) == 128u);

// 每 AIV 独占 64B epoch 时序（禁止多 writer 写共享 PlPipelineTiming）
struct PlAivEpochTimingLine {
    uint64_t first_issue_cycle;
    uint64_t done_cycle;
    uint64_t done_epoch;
    uint32_t block_id;
    uint32_t magic;
    uint8_t pad[32];
};
static_assert(sizeof(PlAivEpochTimingLine) == 64u);

struct PlForwardDoneLine {
    uint64_t done_epoch;
    uint32_t lane_id;
    uint32_t magic;
    uint8_t pad[48];
};
static_assert(sizeof(PlForwardDoneLine) == 64u);

// 每 lane 独占：INC first-hop 完成后 remote 写回 Worker
struct PlFirstHopDoneLine {
    uint64_t epoch;
    uint32_t lane_id;
    uint32_t tokens_received;
    uint64_t payload_bytes_received;
    uint64_t descriptor_bytes_received;
    uint64_t ingress_head;
    uint32_t error_code;
    uint32_t magic;
    uint8_t pad[16];
};
static_assert(sizeof(PlFirstHopDoneLine) == 64u);

constexpr uint32_t kPlDispatchAssistFields = 3u;
// 每 final_slot 独占 64B，避免多 source recv AIV 写邻接 12B 记录时 false-sharing 丢更新
constexpr uint32_t kPlDispatchAssistStrideBytes = 64u;
static_assert(kPlDispatchAssistFields * sizeof(int32_t) <= kPlDispatchAssistStrideBytes);

struct PlUploadCounters {
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
    // P0 telemetry：upload batch 形态 + 首个 tail publish 锚点
    uint64_t upload_batch_count;
    uint64_t upload_batch_tokens_sum;
    uint64_t upload_scalar_put_count;
    uint64_t upload_range_put_count;
    uint64_t first_tail_publish_cycle;
};
static_assert(sizeof(PlUploadCounters) == 128u);

struct PlForwardCounters {
    uint64_t ingress_seen;
    uint64_t ingress_tail_observed;
    uint64_t egress_forwarded;
    uint64_t payload_put_count;
    uint64_t payload_mte_wait_count;
    uint64_t descriptor_put_count;
    uint64_t descriptor_mte_wait_count;
    uint64_t drain_quiet_count;
    uint64_t egress_tail_publish_count;
    uint64_t ingress_head_publish_count;
    uint64_t egress_credit_wait_cycles;
    uint64_t max_ingress_occupancy;
    uint64_t max_egress_occupancy;
    uint64_t egress_payload_range_put_count;
    uint64_t egress_payload_scalar_put_count;
    uint64_t egress_descriptor_range_put_count;
    uint64_t egress_descriptor_scalar_put_count;
    uint64_t egress_batch_count;
    uint64_t egress_batch_tokens;
    uint64_t egress_batch_max_tokens;
    uint64_t forward_before_final_ingress_count;
    uint64_t first_ingress_tail;
    uint32_t error_code;
    uint32_t gather_token_dcci_count; // P5.P3：每 token gather 后 payload DCCI 次数
    // D1：device 写 0；Host 禁止再硬编码，只读此字段
    uint64_t direct_destfinal_remote_put_count;
};
static_assert(sizeof(PlForwardCounters) == 192u);

struct PlRecvCounters {
    uint64_t channel_seen;
    uint64_t tokens_received;
    uint64_t payload_verified_count;
    uint64_t descriptor_verified_count;
    uint64_t verify_error_count;
    uint64_t stale_epoch_count;
    uint64_t duplicate_count;
    uint64_t lost_count;
    uint64_t slot_overwrite_count;
    uint64_t underflow_count;
    uint64_t head_publish_count;
    uint64_t head_completion_quiet_count;
    uint64_t token_ready_publish_count;
    uint64_t last_credit_head_published;
    uint64_t tail_poll_cycles;
    uint64_t destination_head_observed_before_final_forward_count;
    uint64_t first_destination_consume;
    uint64_t max_occupancy;
    uint32_t error_code;
    uint32_t pad0;
    uint8_t pad1[40];
};
static_assert(sizeof(PlRecvCounters) == 192u);

struct PlOverlapTelemetry {
    uint64_t forward_before_final_ingress_total;
    uint64_t destination_early_consume_total;
    uint32_t inc_early_forward_mask;
    uint32_t dest_early_consume_mask;
    uint32_t real_pipeline_overlap; // cycle-domain: worker_inc_overlap && inc_dest_overlap
    uint32_t overlap_fail_reason;
    uint32_t worker_inc_overlap; // same GetSystemCycle domain: upload∩forward
    uint32_t inc_dest_overlap;   // forward∩destination consume
    uint8_t pad[24];
};
static_assert(sizeof(PlOverlapTelemetry) == 64u);

struct PlPipelineTiming {
    uint64_t epoch_start_cycle;
    uint64_t epoch_end_cycle;
    uint64_t first_upload_issue;
    uint64_t last_upload_issue;
    uint64_t first_forward_issue;
    uint64_t last_forward_issue;
    uint64_t first_destination_consume;
    uint64_t destination_done_cycle;
    uint64_t tokens_sent;
    uint64_t tokens_forwarded;
    uint64_t tokens_received;
    uint32_t upload_done_bitmap;
    uint32_t forward_done_bitmap;
    uint32_t recv_done_bitmap;
    uint32_t destination_done_epoch;
    uint32_t error_code;
    uint32_t pad0;
    uint8_t pad1[16];
};
static_assert(sizeof(PlPipelineTiming) == 128u);

struct PlServiceTraceLine {
    uint64_t kernel_enter_cycle;
    uint32_t desc_valid;
    uint32_t block_id;
    uint64_t wait_start_enter_cycle;
    uint64_t observed_start_value;
    uint64_t observed_stop_value;
    uint64_t wait_start_exit_cycle;
    uint64_t epoch_enter_cycle;
    uint64_t first_issue_cycle;
    uint32_t kernel_exit_reason;
    uint32_t wait_start_spin_count;
    uint64_t service_exit_epoch;
    uint64_t start_seen_epoch;
    uint8_t pad[40];
};
static_assert(sizeof(PlServiceTraceLine) == 128u);

struct PlResidentLine {
    uint64_t value;
    uint32_t block_id;
    uint32_t magic;
    uint8_t pad[48];
};
static_assert(sizeof(PlResidentLine) == 64u);

struct PlTokenReadyLine {
    uint64_t ready_epoch;
    uint32_t token_index;
    uint32_t magic;
    uint8_t pad[48];
};
static_assert(sizeof(PlTokenReadyLine) == 64u);

struct PlSourceDoneLine {
    uint64_t done_epoch;
    uint32_t source_rank;
    uint32_t magic;
    uint8_t pad[48];
};
static_assert(sizeof(PlSourceDoneLine) == 64u);

struct PlDestDoneLine {
    uint64_t done_epoch;
    uint32_t destination_rank;
    uint32_t magic;
    uint64_t tokens_received;
    uint8_t pad[40];
};
static_assert(sizeof(PlDestDoneLine) == 64u);

// ----- 堆布局（host/device 一致）-----
constexpr uint64_t kPlDescOff = 0;
constexpr uint64_t kPlResidentOff = 256;
constexpr uint64_t kPlWorkerResidentOff = kPlResidentOff;
constexpr uint64_t kPlIncResidentOff = kPlResidentOff;
constexpr uint64_t kPlWorkerTraceOff = kPlWorkerResidentOff + static_cast<uint64_t>(kPlWorkerServiceBlockDim) * 64u;
constexpr uint64_t kPlIncTraceOff = kPlIncResidentOff + static_cast<uint64_t>(kPlIncServiceBlockDim) * 64u;
constexpr uint64_t kPlUploadCtrOff = kPlWorkerTraceOff + static_cast<uint64_t>(kPlWorkerServiceBlockDim) * 128u;
constexpr uint64_t kPlForwardCtrOff = kPlUploadCtrOff + static_cast<uint64_t>(kQv2LaneCount) * 128u;
constexpr uint64_t kPlRecvCtrOff = kPlForwardCtrOff + static_cast<uint64_t>(kQv2LaneCount) * 192u;
// 对称堆仍按物理 kPlMaxSources 定界（扩 32 会挤破 DataBase）；logical>8 仅 host 映射证明
constexpr uint64_t kPlOverlapOff = kPlRecvCtrOff + static_cast<uint64_t>(kPlMaxSources) * 192u;
constexpr uint64_t kPlTimingOff = kPlOverlapOff + 64u;
constexpr uint64_t kPlSourceDoneOff = kPlTimingOff + 128u;
constexpr uint64_t kPlDestDoneOff = kPlSourceDoneOff + static_cast<uint64_t>(kPlMaxSources) * 64u;
// AIV 独占 timing / forward_done（各 8×64B，control 聚合）
constexpr uint64_t kPlUploadAivTimingOff = kPlDestDoneOff + 64u;
constexpr uint64_t kPlRecvAivTimingOff = kPlUploadAivTimingOff + static_cast<uint64_t>(kQv2LaneCount) * 64u;
constexpr uint64_t kPlForwardDoneOff = kPlRecvAivTimingOff + static_cast<uint64_t>(kPlMaxSources) * 64u;
constexpr uint64_t kPlForwardAivTimingOff = kPlForwardDoneOff + static_cast<uint64_t>(kQv2LaneCount) * 64u;
constexpr uint64_t kPlFirstHopDoneOff = kPlForwardAivTimingOff + static_cast<uint64_t>(kQv2LaneCount) * 64u;

// 队列线（tail/head/credit/scratch）必须位于全部 counter 之后，避免与 PlForwardCounters 重叠
constexpr uint64_t kPlMetaEndOff = kPlFirstHopDoneOff + static_cast<uint64_t>(kQv2LaneCount) * 64u;
constexpr uint64_t kPlIngressTailOff = (kPlMetaEndOff + 511ull) & ~511ull;
static_assert(kPlIngressTailOff >= kPlMetaEndOff, "queue lines must follow counter region");
constexpr uint64_t kPlIngressHeadOff = kPlIngressTailOff + 512;
constexpr uint64_t kPlEgressTailOff = kPlIngressHeadOff + 512;
constexpr uint64_t kPlEgressHeadOff = kPlEgressTailOff + 512;
// INC 侧：destination 回传 egress credit（每 destination 一条线）
constexpr uint64_t kPlIncEgressCreditOff = kPlEgressHeadOff + 512;
// 多 AIV 独占 publish scratch（每 lane/source 64B，禁止与正式 tail/head 重叠）
constexpr uint64_t kPlIncEgressPublishScratchOff = kPlIncEgressCreditOff + 512;
constexpr uint64_t kPlDestCreditPublishScratchOff = kPlIncEgressPublishScratchOff + 512;
static_assert((kPlIncEgressPublishScratchOff & 63u) == 0u, "egress publish scratch must be 64B aligned");
static_assert(kPlIncEgressPublishScratchOff >= kPlIncEgressCreditOff + 512u,
              "egress publish scratch must not overlap INC credit lines");
static_assert(kPlIncEgressPublishScratchOff + static_cast<uint64_t>(kQv2LaneCount) * 64u <=
                  kPlDestCreditPublishScratchOff,
              "per-lane egress publish scratch must not overlap destination credit scratch");

// count-path 单写者 trace：meta 尾隙（DestCreditScratch+512 → DataBase）
constexpr uint64_t kPlWorkerSourceCountTraceOff = kPlDestCreditPublishScratchOff + 512u;
constexpr uint64_t kPlWorkerSourceCountTraceBytes = static_cast<uint64_t>(kPlMaxSources) * 64u;
constexpr uint64_t kPlIncSourceCountTraceOff = kPlWorkerSourceCountTraceOff + kPlWorkerSourceCountTraceBytes;
constexpr uint64_t kPlIncSourceCountTraceBytes = static_cast<uint64_t>(kPlMaxSources) * 64u;
constexpr uint64_t kPlDestinationCountTraceOff = kPlIncSourceCountTraceOff + kPlIncSourceCountTraceBytes;
constexpr uint64_t kPlDestinationCountTraceBytes = static_cast<uint64_t>(kPlMaxSources) * 64u;

constexpr uint64_t kPlDataBaseOff = 16384;
static_assert(kPlDestinationCountTraceOff + kPlDestinationCountTraceBytes <= kPlDataBaseOff,
              "count-path trace region must fit before data base");
static_assert(kPlDestCreditPublishScratchOff + 512u <= kPlWorkerSourceCountTraceOff,
              "scratch region must not overlap count-path traces");

// ---------------------------------------------------------------------------
// M4 symmetric transport heap：所有 PE 同大小 aclshmem_malloc(kPlSymmetricTransportHeapNeed)。
//   control + IncPayload/IncDesc + RouteCount…InvocationWorkspace + DestChannel + TokenReady
// WorkerSource/ExpertIds/Gather/WorkerDesc/DestFinal* → Worker aclrtMalloc（PlInvocationWorkspaceDesc）。
// INC 可有独立 local workspace；禁止 remote put/get 到 Worker raw ptr。
// kPlMaxFinalSlotsCompileBound 仅 capability 上界，不 sizing symmetric heap。
// ---------------------------------------------------------------------------
constexpr uint32_t kPlMaxStackedInputEpochs = 3u;
static_assert(kPlMaxStackedInputEpochs >= 1u && kPlMaxStackedInputEpochs <= kPlMaxInvocationEpochs,
              "stacked input epochs must fit invocation table");
// Worker-local 容量（session sizing；非 symmetric offset）。compile 上界按 max token；runtime 按 payload_bytes。
constexpr uint64_t kPlWorkerSourceBytesMax =
    static_cast<uint64_t>(kPlMaxTokens) * kPlMaxStackedInputEpochs * kPlMaxTokenBytes;
constexpr uint64_t kPlWorkerSourceBytes =
    static_cast<uint64_t>(kPlMaxTokens) * kPlMaxStackedInputEpochs * kQv2TokenBytes; // legacy h4096 名
constexpr uint64_t kPlWorkerExpertIdsBytes =
    static_cast<uint64_t>(kPlMaxExpertIdsSlots) * sizeof(int32_t);
constexpr uint64_t kPlWorkerGatherPayloadBytesMax =
    static_cast<uint64_t>(kQv2LaneCount) * kQv2RingDepth * kPlMaxTokenBytes;
constexpr uint64_t kPlWorkerGatherPayloadBytes =
    static_cast<uint64_t>(kQv2LaneCount) * kQv2RingDepth * kQv2TokenBytes; // legacy h4096 名
constexpr uint64_t kPlWorkerGatherDescBytes =
    static_cast<uint64_t>(kQv2LaneCount) * kQv2RingDepth * kPlDescriptorBytes;
// A destination lane can receive more than one expert route from the same
// token.  The old max_tokens stride silently assumed topk1/per-destination
// uniqueness and let hot/asymmetric top-k overwrite the next lane and then
// protocol control lines.  Size each lane for the full legal source route
// vector; runtime code still writes only its actual destination subset.
constexpr uint64_t kPlWorkerDescLaneStride = static_cast<uint64_t>(kPlMaxExpertIdsSlots);
static_assert(kPlWorkerDescLaneStride >=
                  static_cast<uint64_t>(kPlMaxTokens) * static_cast<uint64_t>(kPlMaxTopk),
              "worker desc lane stride must hold worst-case multi-route hot destination");
// Deprecated symmetric staging retains its old footprint for ABI/layout
// compatibility. Full-dispatch upload reads the expanded worker-local
// descriptor lanes directly.
constexpr uint64_t kPlWorkerSymDescLaneStride = static_cast<uint64_t>(kPlMaxTokens);
constexpr uint64_t kPlWorkerDescBytes =
    kPlWorkerDescLaneStride * static_cast<uint64_t>(kQv2LaneCount) * kPlDescriptorBytes;
// M4 已迁出 symmetric heap（deprecated；禁止 remote target）
constexpr uint64_t kPlWorkerSourceOff = 0u;
constexpr uint64_t kPlWorkerExpertIdsOff = 0u;
constexpr uint64_t kPlWorkerGatherPayloadOff = 0u;
constexpr uint64_t kPlWorkerGatherDescOff = 0u;
constexpr uint64_t kPlWorkerDescOff = 0u;
constexpr uint64_t kPlDestFinalPayloadOff = 0u;
constexpr uint64_t kPlDestFinalAssistOff = 0u;
constexpr uint64_t kPlDestEpRecvCountOff = 0u;
constexpr uint64_t kPlDestExpertTokenNumsOff = 0u;
// Symmetric transport data：Inc ring 按 max token 预留（lane 内 slot 按 runtime payload 紧密排）
constexpr uint64_t kPlIncPayloadOff = kPlDataBaseOff;
constexpr uint64_t kPlIncPayloadRegionBytes =
    static_cast<uint64_t>(kQv2LaneCount) * kQv2RingDepth * kPlMaxTokenBytes;
constexpr uint64_t kPlIncDescOff = kPlIncPayloadOff + kPlIncPayloadRegionBytes;
constexpr uint64_t kPlIncDescRegionBytes =
    static_cast<uint64_t>(kQv2LaneCount) * kQv2RingDepth * kPlDescriptorBytes;
// P5.3：source-major raw slab + per-lane SPSC chunk ring（插在 IncDesc 与 RouteCount 之间）
// 34MiB slab：覆盖 landmark F4/F7（h12288×t683/t1365 → ~16/32MiB）连续 source-major 上传；
// 扩容后 transport heap 仍须 ≤64MiB（见 kPlSymmetricTransportHeapNeed static_assert）。
constexpr uint64_t kPlIncRawSlabOff = kPlIncDescOff + kPlIncDescRegionBytes;
constexpr uint64_t kPlIncRawSlabBytes = 34ull << 20;
constexpr uint64_t kPlRawChunkDescOff = kPlIncRawSlabOff + kPlIncRawSlabBytes;
// 8 upload-lane × depth=8 × 128B
constexpr uint64_t kPlRawChunkDescBytes = static_cast<uint64_t>(kPlRawUploadLaneCount) *
                                          static_cast<uint64_t>(kPlRawChunkRingDepth) * sizeof(PlRawChunkDesc);
constexpr uint64_t kPlIncRawRouteMetaOff = kPlRawChunkDescOff + kPlRawChunkDescBytes;
constexpr uint64_t kPlIncRawRouteMetaBytes = 128ull << 10;
constexpr uint64_t kPlIncRawRouteOrdinalOff =
    kPlIncRawRouteMetaOff + kPlIncRawRouteMetaBytes;
constexpr uint64_t kPlIncRawRouteOrdinalBytes = 128ull << 10;

/*
 * One device epoch uses a bounded CSR workspace.  This is deliberately not a
 * public message-size limit: the framework chunk planner reduces the epoch
 * row count until both metadata and entries fit, then reuses this workspace.
 *
 * offsets[worker_count + 1] live immediately after the aligned packed route
 * metadata in kPlIncRawRouteMetaBytes.  Entries live in the ordinal region.
 */
struct PlDestinationCsrEntry {
    uint16_t source_token_id;
    uint8_t topk_slot;
    uint8_t expert_id;
    uint32_t source_segment_offset;
};
static_assert(sizeof(PlDestinationCsrEntry) == 8u);

inline uint64_t PlAlign64(uint64_t value)
{
    return (value + 63u) & ~uint64_t{63u};
}

inline uint64_t PlDestinationCsrOffsetsOff(uint32_t metadata_bytes)
{
    return kPlIncRawRouteMetaOff + PlAlign64(metadata_bytes);
}

inline bool PlDestinationCsrFits(uint32_t token_count, uint32_t topk,
                                 uint32_t worker_count)
{
    const uint64_t route_count =
        static_cast<uint64_t>(token_count) * static_cast<uint64_t>(topk);
    const uint64_t metadata_bytes = route_count * sizeof(uint32_t);
    const uint64_t offsets_end =
        PlAlign64(metadata_bytes) +
        (static_cast<uint64_t>(worker_count) * 8u + 1u) *
            sizeof(uint32_t);
    const uint64_t entries_bytes =
        route_count * sizeof(PlDestinationCsrEntry);
    return token_count <= 0xffffu &&
           offsets_end <= kPlIncRawRouteMetaBytes &&
           entries_bytes <= kPlIncRawRouteOrdinalBytes;
}
// per-lane raw_tail[L] / raw_head[L]（各独占 64B；禁止共享 RMW）
constexpr uint64_t kPlRawIngressTailOff =
    kPlIncRawRouteOrdinalOff + kPlIncRawRouteOrdinalBytes;
constexpr uint64_t kPlRawIngressTailBytes = static_cast<uint64_t>(kPlRawUploadLaneCount) * 64u;
constexpr uint64_t kPlRawIngressHeadOff = kPlRawIngressTailOff + kPlRawIngressTailBytes;
constexpr uint64_t kPlRawIngressHeadBytes = static_cast<uint64_t>(kPlRawUploadLaneCount) * 64u;
// route_done_generation[upload_lane][destination] — gather 唯一写本 dest 行；reclaim 只读
constexpr uint64_t kPlRouteDoneGenOff = kPlRawIngressHeadOff + kPlRawIngressHeadBytes;
constexpr uint64_t kPlRouteDoneGenBytes =
    static_cast<uint64_t>(kPlRawUploadLaneCount) * static_cast<uint64_t>(kPlMaxSources) * 64u;
constexpr uint64_t kPlFirstHopVerifyOff = kPlRouteDoneGenOff + kPlRouteDoneGenBytes; // INC→Worker 计时外真值
constexpr uint64_t kPlP53InjectOff = kPlFirstHopVerifyOff + 64u;
// ExtResident[0] = PlRawMetaReadyLine（单写者 route-meta doorbell）
// ExtResident[1] = ControlEpochRetireLine（Worker control epoch retire）
constexpr uint64_t kPlIncP5ExtResidentOff = kPlP53InjectOff + 64u;
constexpr uint64_t kPlRawMetaReadyOff = kPlIncP5ExtResidentOff;
constexpr uint64_t kPlControlEpochRetireOff = kPlIncP5ExtResidentOff + 64u;
constexpr uint64_t kPlIncP5ExtResidentBytes = 10ull * 64u; // blocks 10–19；[0]=RawMetaReady [1]=Retire
constexpr uint64_t kPlIncP5ExtTraceOff = kPlIncP5ExtResidentOff + kPlIncP5ExtResidentBytes;
constexpr uint64_t kPlIncP5ExtTraceBytes = 10ull * 128u;
// per-dest staging SPSC：tail=gather 写；head=egress 写（替换 ping/pong ready bool）
constexpr uint64_t kPlIncP5StagingTailOff = kPlIncP5ExtTraceOff + kPlIncP5ExtTraceBytes;
constexpr uint64_t kPlIncP5StagingTailBytes = static_cast<uint64_t>(kQv2LaneCount) * 64u;
constexpr uint64_t kPlIncP5StagingHeadOff = kPlIncP5StagingTailOff + kPlIncP5StagingTailBytes;
constexpr uint64_t kPlIncP5StagingHeadBytes = static_cast<uint64_t>(kQv2LaneCount) * 64u;
// E5b：per-dest egress done（egress AIV 写；block19 聚合读）
constexpr uint64_t kPlIncP5EgressDoneOff = kPlIncP5StagingHeadOff + kPlIncP5StagingHeadBytes;
constexpr uint64_t kPlIncP5EgressDoneBytes = static_cast<uint64_t>(kQv2LaneCount) * 64u;
// 兼容旧名：指向 staging tail 区（禁止再当 ready ping/pong 用）
constexpr uint64_t kPlIncP5StagingReadyOff = kPlIncP5StagingTailOff;
constexpr uint64_t kPlIncP5StagingReadyBytes = kPlIncP5StagingTailBytes + kPlIncP5StagingHeadBytes;
// C3 控制面：route count 交换 + 每 dest 的 expected_count[source]
constexpr uint64_t kPlRouteCountRegionOff = kPlIncP5EgressDoneOff + kPlIncP5EgressDoneBytes;
constexpr uint64_t kPlRouteCountRegionBytes =
    static_cast<uint64_t>(kPlMaxSources) * static_cast<uint64_t>(kPlMaxSources) * 8u * sizeof(PlRouteCountLine);
constexpr uint64_t kPlExpectedCountOff = kPlRouteCountRegionOff + kPlRouteCountRegionBytes;
constexpr uint64_t kPlExpectedCountBytes = static_cast<uint64_t>(kPlMaxSources) * 64u;
// push doorbell：SourceCountReady / DestCountSliceReady / SegmentBaseReady
constexpr uint64_t kPlSourceCountReadyOff = kPlExpectedCountOff + kPlExpectedCountBytes;
constexpr uint64_t kPlSourceCountReadyBytes = static_cast<uint64_t>(kPlMaxSources) * 64u;
constexpr uint64_t kPlDestCountSliceReadyOff = kPlSourceCountReadyOff + kPlSourceCountReadyBytes;
constexpr uint64_t kPlDestCountSliceReadyBytes =
    static_cast<uint64_t>(kPlMaxSources) * static_cast<uint64_t>(kPlMaxSources) * 64u;
constexpr uint64_t kPlSegmentBaseReadyOff = kPlDestCountSliceReadyOff + kPlDestCountSliceReadyBytes;
constexpr uint64_t kPlSegmentBaseReadyBytes =
    static_cast<uint64_t>(kPlMaxSources) * static_cast<uint64_t>(kPlMaxSources) * 64u;
// DestPrefix segment_base publish 专用 staging（每 source 64B；禁止复用 kPlWorkerResidentOff）
constexpr uint64_t kPlSegBasePublishStagingOff = kPlSegmentBaseReadyOff + kPlSegmentBaseReadyBytes;
constexpr uint64_t kPlSegBasePublishStagingBytes = static_cast<uint64_t>(kPlMaxSources) * 64u;
// source 本地：各 destination 推回的 segment_base；每 dest 独占 64B cacheline（前 8×int32 有效）
constexpr uint64_t kPlOutboundSegBaseOff = kPlSegBasePublishStagingOff + kPlSegBasePublishStagingBytes;
constexpr uint64_t kPlOutboundSegBaseBytes = static_cast<uint64_t>(kPlMaxSources) * 64u;
// segment_base[local_expert * worker_count + source]（destination 本地 canonical）
constexpr uint64_t kPlSegmentBaseOff = kPlOutboundSegBaseOff + kPlOutboundSegBaseBytes;
constexpr uint64_t kPlSegmentBaseBytes =
    static_cast<uint64_t>(kPlMaxSources) * sizeof(int32_t) * 8u;
constexpr uint64_t kPlLaneWorkOff = kPlSegmentBaseOff + kPlSegmentBaseBytes;
constexpr uint64_t kPlFullOutputDoneOff = kPlLaneWorkOff + static_cast<uint64_t>(kQv2LaneCount) * 64u;
constexpr uint64_t kPlRecvDoneOff = kPlFullOutputDoneOff + 64u;
constexpr uint64_t kPlRecvDoneBytes = static_cast<uint64_t>(kPlMaxSources) * 64u;
constexpr uint64_t kPlRouteTimingOff = kPlRecvDoneOff + kPlRecvDoneBytes;
// 正式 transport timing（独占 cacheline；禁止复用 kPlRouteTimingOff）
constexpr uint64_t kPlTransportDoneOff = kPlRouteTimingOff + 128u;
constexpr uint64_t kPlTransportDoneBytes = static_cast<uint64_t>(kPlMaxSources) * 64u;
constexpr uint64_t kPlGlobalTransportDoneOff = kPlTransportDoneOff + kPlTransportDoneBytes;
constexpr uint64_t kPlTransportTimingOff = kPlGlobalTransportDoneOff + 64u;
// S4 单写者：forward lane 独占 (lane,dest) 64B；block9 聚合后写 summary cell
constexpr uint64_t kPlEgressLaneLocalVisibleOff = kPlTransportTimingOff + 128u;
constexpr uint64_t kPlEgressLaneLocalVisibleBytes =
    static_cast<uint64_t>(kQv2LaneCount) * static_cast<uint64_t>(kPlMaxSources) * 64u;
// R3：8 INC × 8 dest egress visible summary（PE0 堆槽，仅 block9 putmem）
constexpr uint64_t kPlEgressVisibleDoneOff = kPlEgressLaneLocalVisibleOff + kPlEgressLaneLocalVisibleBytes;
constexpr uint64_t kPlEgressVisibleDoneBytes =
    static_cast<uint64_t>(kPlMaxSources) * static_cast<uint64_t>(kPlMaxSources) * 64u;
constexpr uint64_t kPlSecondHopVisibleTimingOff = kPlEgressVisibleDoneOff + kPlEgressVisibleDoneBytes;
constexpr uint64_t kPlGlobalSecondHopVisibleDoneOff = kPlSecondHopVisibleTimingOff + 128u;
constexpr uint64_t kPlIncSecondHopSummaryOff = kPlGlobalSecondHopVisibleDoneOff + 64u;
constexpr uint64_t kPlIncSecondHopSummaryBytes = static_cast<uint64_t>(kPlMaxSources) * 64u;
constexpr uint64_t kPlSecondHopAggTelemetryOff = kPlIncSecondHopSummaryOff + kPlIncSecondHopSummaryBytes;
constexpr uint64_t kPlUploadLaneStageTimingOff = kPlSecondHopAggTelemetryOff + 64u;
constexpr uint64_t kPlUploadLaneStageTimingBytes = static_cast<uint64_t>(kQv2LaneCount) * 128u;
constexpr uint64_t kPlDestinationCompletionTraceOff = kPlUploadLaneStageTimingOff + kPlUploadLaneStageTimingBytes;
constexpr uint64_t kPlDestinationCompletionTraceBytes = 128u;
constexpr uint64_t kPlRecvChannelCompletionTraceOff =
    kPlDestinationCompletionTraceOff + kPlDestinationCompletionTraceBytes;
constexpr uint64_t kPlRecvChannelCompletionTraceBytes = static_cast<uint64_t>(kPlMaxSources) * 384u;
constexpr uint64_t kPlIncForwardLaneStageTimingOff =
    kPlRecvChannelCompletionTraceOff + kPlRecvChannelCompletionTraceBytes;
constexpr uint64_t kPlIncForwardLaneStageTimingBytes = static_cast<uint64_t>(kQv2LaneCount) * 256u;
constexpr uint64_t kPlTimingNegOff = kPlIncForwardLaneStageTimingOff + kPlIncForwardLaneStageTimingBytes;
// P5 source×destination accounting: route/recv lives on destination Worker;
// gather/egress lives on source INC. Separate regions enforce single-writer
// ownership without cross-role RMW.
constexpr uint64_t kPlP5CellRouteRecvLedgerOff = kPlTimingNegOff + 64u;
constexpr uint64_t kPlP5CellRouteRecvLedgerBytes =
    static_cast<uint64_t>(kPlMaxSources) * static_cast<uint64_t>(kPlMaxSources) * 64u;
constexpr uint64_t kPlP5CellGatherLedgerOff = kPlP5CellRouteRecvLedgerOff + kPlP5CellRouteRecvLedgerBytes;
constexpr uint64_t kPlP5CellGatherLedgerBytes =
    static_cast<uint64_t>(kPlMaxSources) * static_cast<uint64_t>(kPlMaxSources) * 64u;
constexpr uint64_t kPlP5CellEgressLedgerOff = kPlP5CellGatherLedgerOff + kPlP5CellGatherLedgerBytes;
constexpr uint64_t kPlP5CellEgressLedgerBytes =
    static_cast<uint64_t>(kPlMaxSources) * static_cast<uint64_t>(kPlMaxSources) * 64u;
constexpr uint64_t kPlRankDispatchTimingOff = kPlP5CellEgressLedgerOff + kPlP5CellEgressLedgerBytes;
constexpr uint64_t kPlRankDispatchTimingBytes = static_cast<uint64_t>(kPlMaxPhysicalPe) * 64u;
// Host 集中读 remote 时的独立 scratch（禁止 getmem 覆盖 live telemetry）
constexpr uint64_t kPlHostProbeScratchOff = kPlRankDispatchTimingOff + kPlRankDispatchTimingBytes;
constexpr uint64_t kPlHostProbeScratchBytes = 4096u;
// P8：epoch 级 invocation 表（在 DestChannel 之前，避免与 counter 区重叠）
constexpr uint64_t kPlInvocationDescOff = kPlHostProbeScratchOff + kPlHostProbeScratchBytes;
constexpr uint64_t kPlInvocationDescBytes =
    static_cast<uint64_t>(kPlMaxInvocationEpochs) * sizeof(PlInvocationDesc);
// M2：与 PlInvocationDesc 同 epoch 邻接的 workspace 旁路表（仍 Legacy 时 ptr 可为空）
constexpr uint64_t kPlInvocationWorkspaceOff = kPlInvocationDescOff + kPlInvocationDescBytes;
constexpr uint64_t kPlInvocationWorkspaceBytes =
    static_cast<uint64_t>(kPlMaxInvocationEpochs) * sizeof(PlInvocationWorkspaceDesc);
// Destination-major channel ring：按 max token 预留（source 内 slot 按 runtime payload 紧密排）
constexpr uint64_t kPlDestChannelPayloadOff = kPlInvocationWorkspaceOff + kPlInvocationWorkspaceBytes;
constexpr uint64_t kPlDestChannelPayloadRegionBytes =
    static_cast<uint64_t>(kPlMaxSources) * kQv2RingDepth * kPlMaxTokenBytes;
constexpr uint64_t kPlDestChannelDescOff = kPlDestChannelPayloadOff + kPlDestChannelPayloadRegionBytes;
constexpr uint64_t kPlDestChannelDescRegionBytes =
    static_cast<uint64_t>(kPlMaxSources) * kQv2RingDepth * kPlDescriptorBytes;
// V2：删除 symmetric Worker upload payload/expert_ids staging（第一跳从 Worker local input_ptr）
// 保留 gather descriptor staging 作为有序 desc put 源
constexpr uint64_t kPlWorkerUploadSrcOff = 0u; // removed; fail-closed if used as remote put source sizing
constexpr uint64_t kPlWorkerUploadSrcBytes = 0u;
constexpr uint64_t kPlWorkerUploadExpertIdsOff = 0u;
constexpr uint64_t kPlWorkerUploadExpertIdsBytes = 0u;
constexpr uint64_t kPlWorkerGatherDescStagingOff = kPlDestChannelDescOff + kPlDestChannelDescRegionBytes;
constexpr uint64_t kPlWorkerGatherDescStagingBytes =
    static_cast<uint64_t>(kQv2LaneCount) * kPlWorkerSymDescLaneStride * kPlDescriptorBytes;
constexpr uint64_t kPlTokenReadyOff = kPlWorkerGatherDescStagingOff + kPlWorkerGatherDescStagingBytes;
constexpr uint64_t kPlD1RegionOff = (kPlTokenReadyOff + 1048575ull) & ~1048575ull;
constexpr uint64_t kPlSymmetricTransportHeapNeed = kPlD1RegionOff + 65536ull;
constexpr uint64_t kPlHeapNeed = kPlSymmetricTransportHeapNeed;
static_assert(kPlSymmetricTransportHeapNeed <= (64ull << 20),
              "M4/V2 symmetric transport heap must fit 64MiB goal");
static_assert(kPlIncPayloadRegionBytes == 12ull * 1024ull * 1024ull, "INC payload ring = 8*64*24576");
static_assert(kPlDestChannelPayloadRegionBytes == 12ull * 1024ull * 1024ull,
              "DestChannel payload ring = 8*64*24576");

inline uint64_t PlIngressTailLineOff(uint32_t lane_id)
{
    return kPlIngressTailOff + static_cast<uint64_t>(lane_id) * 64u;
}

inline uint64_t PlIngressHeadLineOff(uint32_t lane_id)
{
    return kPlIngressHeadOff + static_cast<uint64_t>(lane_id) * 64u;
}

// Worker-local upload desc mirror：base 为 cfg.final_payload_off（绝对 device ptr）
inline uint64_t PlWorkerDescLaneRelOff(uint32_t lane_id)
{
    return static_cast<uint64_t>(lane_id) * kPlWorkerDescLaneStride * kPlDescriptorBytes;
}

inline uint64_t PlEgressTailLineOff(uint32_t source_id)
{
    return kPlEgressTailOff + static_cast<uint64_t>(source_id) * 64u;
}

inline uint64_t PlEgressHeadLineOff(uint32_t source_id)
{
    return kPlEgressHeadOff + static_cast<uint64_t>(source_id) * 64u;
}

inline uint64_t PlIncEgressCreditLineOff(uint32_t dest_rank)
{
    return kPlIncEgressCreditOff + static_cast<uint64_t>(dest_rank) * 64u;
}

inline uint64_t PlIncEgressPublishScratchOff(uint32_t lane_id)
{
    return kPlIncEgressPublishScratchOff + static_cast<uint64_t>(lane_id) * 64u;
}

inline uint64_t PlDestCreditPublishScratchOff(uint32_t source_id)
{
    return kPlDestCreditPublishScratchOff + static_cast<uint64_t>(source_id) * 64u;
}

inline uint64_t PlIncPayloadLaneOff(uint32_t lane_id)
{
    // 区域按 max token 预留；lane 内 slot 用 runtime payload_bytes 紧密排
    return kPlIncPayloadOff + static_cast<uint64_t>(lane_id) * kQv2RingDepth * kPlMaxTokenBytes;
}

inline uint64_t PlIncDescLaneOff(uint32_t lane_id)
{
    return kPlIncDescOff + static_cast<uint64_t>(lane_id) * kQv2RingDepth * kPlDescriptorBytes;
}

inline uint64_t PlDestChannelPayloadOff(uint32_t source_id)
{
    return kPlDestChannelPayloadOff +
           static_cast<uint64_t>(source_id) * kQv2RingDepth * kPlMaxTokenBytes;
}

inline uint64_t PlDestChannelDescOff(uint32_t source_id)
{
    return kPlDestChannelDescOff +
           static_cast<uint64_t>(source_id) * kQv2RingDepth * kPlDescriptorBytes;
}

inline uint64_t PlTokenReadyLineOff(uint32_t token_index)
{
    return kPlTokenReadyOff + static_cast<uint64_t>(token_index) * 64u;
}

inline uint64_t PlSourceDoneLineOff(uint32_t source_id)
{
    return kPlSourceDoneOff + static_cast<uint64_t>(source_id) * 64u;
}

inline uint64_t PlUploadAivTimingOff(uint32_t lane_id)
{
    return kPlUploadAivTimingOff + static_cast<uint64_t>(lane_id) * 64u;
}

inline uint64_t PlRecvAivTimingOff(uint32_t source_id)
{
    return kPlRecvAivTimingOff + static_cast<uint64_t>(source_id) * 64u;
}

inline uint64_t PlForwardDoneLineOff(uint32_t lane_id)
{
    return kPlForwardDoneOff + static_cast<uint64_t>(lane_id) * 64u;
}

inline uint64_t PlFirstHopDoneLineOff(uint32_t lane_id)
{
    return kPlFirstHopDoneOff + static_cast<uint64_t>(lane_id) * 64u;
}

inline uint64_t PlForwardAivTimingOff(uint32_t lane_id)
{
    return kPlForwardAivTimingOff + static_cast<uint64_t>(lane_id) * 64u;
}

inline bool PlValidateRingDepth(uint32_t ring_depth)
{
    return ring_depth == kQv2RingDepth;
}

inline bool PlValidateBatchTile(uint32_t batch_tokens, uint32_t tile_tokens)
{
    if (batch_tokens == 0 || tile_tokens == 0 || batch_tokens > tile_tokens || tile_tokens > kQv2RingDepth) {
        return false;
    }
    return true;
}

inline uint64_t PlLaneTokenBase(uint32_t lane_id, uint32_t tokens_per_lane)
{
    return static_cast<uint64_t>(lane_id) * tokens_per_lane;
}

inline bool PlValidateResourceMap()
{
    return kPlWorkerControlBlock < kPlCombineReserveBegin && kPlIncControlBlock < kPlCombineReserveBegin &&
           kPlIncSecondHopAggBlock < kPlCombineReserveBegin &&
           kPlIncP5ServiceBlockDim <= kPlCombineReserveBegin &&
           kPlWorkerServiceBlockDim <= kPlCombineReserveBegin && kPlIncServiceBlockDim <= kPlCombineReserveBegin;
}

inline uint64_t PlRawLaneTailOff(uint32_t lane)
{
    return kPlRawIngressTailOff + static_cast<uint64_t>(lane % kPlRawUploadLaneCount) * 64u;
}

inline uint64_t PlRawLaneHeadOff(uint32_t lane)
{
    return kPlRawIngressHeadOff + static_cast<uint64_t>(lane % kPlRawUploadLaneCount) * 64u;
}

inline uint64_t PlRawChunkDescLaneSlotOff(uint32_t lane, uint32_t slot)
{
    const uint32_t L = lane % kPlRawUploadLaneCount;
    const uint32_t s = slot % kPlRawChunkRingDepth;
    return kPlRawChunkDescOff +
           (static_cast<uint64_t>(L) * kPlRawChunkRingDepth + static_cast<uint64_t>(s)) * sizeof(PlRawChunkDesc);
}

// 兼容旧单槽 API：按全局 slot 映射到 lane0 ring（仅诊断；正式路径用 PlRawChunkDescLaneSlotOff）
inline uint64_t PlRawChunkDescSlotOff(uint32_t slot)
{
    return PlRawChunkDescLaneSlotOff(0u, slot);
}

inline uint64_t PlIncRawSlabChunkOff(uint32_t slot)
{
    return kPlIncRawSlabOff + static_cast<uint64_t>(slot % kPlRawChunkRingDepth) * kPlRawChunkTargetBytes;
}

inline uint64_t PlIncRawRouteMetaChunkOff(uint32_t slot)
{
    return kPlIncRawRouteMetaOff + static_cast<uint64_t>(slot % kPlRawChunkRingDepth) * 4096ull;
}

inline uint64_t PlRouteDoneGenOff(uint32_t upload_lane, uint32_t destination)
{
    const uint32_t L = upload_lane % kPlRawUploadLaneCount;
    const uint32_t d = destination % kPlMaxSources;
    return kPlRouteDoneGenOff + (static_cast<uint64_t>(L) * kPlMaxSources + static_cast<uint64_t>(d)) * 64u;
}

inline uint64_t PlIncP5ExtResidentLineOff(uint32_t block_id)
{
    // blocks 10–19 → index 0–9
    const uint32_t idx = (block_id >= 10u) ? (block_id - 10u) : 0u;
    return kPlIncP5ExtResidentOff + static_cast<uint64_t>(idx) * 64u;
}

inline uint64_t PlIncP5ExtTraceLineOff(uint32_t block_id)
{
    const uint32_t idx = (block_id >= 10u) ? (block_id - 10u) : 0u;
    return kPlIncP5ExtTraceOff + static_cast<uint64_t>(idx) * 128u;
}

inline uint64_t PlIncP5StagingTailOff(uint32_t dest_lane)
{
    return kPlIncP5StagingTailOff + static_cast<uint64_t>(dest_lane % kQv2LaneCount) * 64u;
}

inline uint64_t PlIncP5StagingHeadOff(uint32_t dest_lane)
{
    return kPlIncP5StagingHeadOff + static_cast<uint64_t>(dest_lane % kQv2LaneCount) * 64u;
}

inline uint64_t PlIncP5EgressDoneOff(uint32_t dest_lane)
{
    return kPlIncP5EgressDoneOff + static_cast<uint64_t>(dest_lane % kQv2LaneCount) * 64u;
}

// deprecated：旧 ping/pong ready；映射到 staging tail 线
inline uint64_t PlIncP5StagingReadyOff(uint32_t dest_lane, uint32_t /*slot*/)
{
    return PlIncP5StagingTailOff(dest_lane);
}

inline uint64_t PlRouteCountLineOff(uint32_t source_rank, uint32_t dest_rank, uint32_t local_expert)
{
    const uint64_t idx =
        static_cast<uint64_t>(source_rank) * static_cast<uint64_t>(kPlMaxSources) * 8u +
        static_cast<uint64_t>(dest_rank) * 8u + static_cast<uint64_t>(local_expert);
    return kPlRouteCountRegionOff + idx * sizeof(PlRouteCountLine);
}

inline uint64_t PlExpectedCountLineOff(uint32_t source_rank)
{
    return kPlExpectedCountOff + static_cast<uint64_t>(source_rank) * 64u;
}

inline uint64_t PlSourceCountReadyLineOff(uint32_t source_rank)
{
    return kPlSourceCountReadyOff + static_cast<uint64_t>(source_rank) * 64u;
}

inline uint64_t PlWorkerSourceCountTraceOff(uint32_t source_rank)
{
    return kPlWorkerSourceCountTraceOff + static_cast<uint64_t>(source_rank) * 64u;
}

inline uint64_t PlIncSourceCountTraceOff(uint32_t source_rank)
{
    return kPlIncSourceCountTraceOff + static_cast<uint64_t>(source_rank) * 64u;
}

inline uint64_t PlDestinationCountTraceOff(uint32_t dest_rank)
{
    return kPlDestinationCountTraceOff + static_cast<uint64_t>(dest_rank) * 64u;
}

inline uint64_t PlDestCountSliceReadyLineOff(uint32_t source_rank, uint32_t dest_rank)
{
    return kPlDestCountSliceReadyOff +
           (static_cast<uint64_t>(source_rank) * static_cast<uint64_t>(kPlMaxSources) +
            static_cast<uint64_t>(dest_rank)) *
               64u;
}

inline uint64_t PlSegmentBaseReadyLineOff(uint32_t source_rank, uint32_t dest_rank)
{
    return kPlSegmentBaseReadyOff +
           (static_cast<uint64_t>(source_rank) * static_cast<uint64_t>(kPlMaxSources) +
            static_cast<uint64_t>(dest_rank)) *
               64u;
}

inline uint64_t PlP5CellRouteRecvLedgerLineOff(uint32_t source_rank, uint32_t dest_rank)
{
    return kPlP5CellRouteRecvLedgerOff +
           (static_cast<uint64_t>(source_rank) * static_cast<uint64_t>(kPlMaxSources) +
            static_cast<uint64_t>(dest_rank)) *
               64u;
}

inline uint64_t PlP5CellGatherLedgerLineOff(uint32_t source_rank, uint32_t dest_rank)
{
    return kPlP5CellGatherLedgerOff +
           (static_cast<uint64_t>(source_rank) * static_cast<uint64_t>(kPlMaxSources) +
            static_cast<uint64_t>(dest_rank)) *
               64u;
}

inline uint64_t PlP5CellEgressLedgerLineOff(uint32_t source_rank, uint32_t dest_rank)
{
    return kPlP5CellEgressLedgerOff +
           (static_cast<uint64_t>(source_rank) * static_cast<uint64_t>(kPlMaxSources) +
            static_cast<uint64_t>(dest_rank)) *
               64u;
}

inline uint64_t PlSegBasePublishStagingSlotOff(uint32_t source_rank)
{
    return kPlSegBasePublishStagingOff + static_cast<uint64_t>(source_rank) * 64u;
}

inline uint64_t PlUploadLaneStageTimingOff(uint32_t lane_id)
{
    return kPlUploadLaneStageTimingOff + static_cast<uint64_t>(lane_id) * 128u;
}

inline uint64_t PlDestinationCompletionTraceOff()
{
    return kPlDestinationCompletionTraceOff;
}

inline uint64_t PlRecvChannelCompletionTraceOff(uint32_t source_rank)
{
    return kPlRecvChannelCompletionTraceOff + static_cast<uint64_t>(source_rank) * 384u;
}

inline uint64_t PlIncForwardLaneStageTimingOff(uint32_t lane_id)
{
    return kPlIncForwardLaneStageTimingOff + static_cast<uint64_t>(lane_id) * 256u;
}

inline uint64_t PlRankDispatchTimingLineOff(uint32_t rank)
{
    return kPlRankDispatchTimingOff + static_cast<uint64_t>(rank) * sizeof(PlRankDispatchTimingLine);
}

inline uint64_t PlOutboundSegBaseIndex(uint32_t dest_rank, uint32_t local_expert)
{
    // 每 dest 16×int32 = 64B，避免邻 dest 共享 cacheline / range put 互踩
    return static_cast<uint64_t>(dest_rank) * 16u + static_cast<uint64_t>(local_expert);
}

inline uint64_t PlRecvDoneLineOff(uint32_t source_rank)
{
    return kPlRecvDoneOff + static_cast<uint64_t>(source_rank) * 64u;
}

inline uint64_t PlTransportDoneLineOff(uint32_t destination_rank)
{
    return kPlTransportDoneOff + static_cast<uint64_t>(destination_rank) * 64u;
}

inline uint64_t PlEgressLaneLocalVisibleSlotOff(uint32_t lane_id, uint32_t dest_rank)
{
    return kPlEgressLaneLocalVisibleOff +
           (static_cast<uint64_t>(lane_id) * static_cast<uint64_t>(kPlMaxSources) +
            static_cast<uint64_t>(dest_rank)) *
               64u;
}

inline uint64_t PlEgressVisibleDoneSlotOff(uint32_t inc_rank, uint32_t dest_rank)
{
    return kPlEgressVisibleDoneOff +
           (static_cast<uint64_t>(inc_rank) * static_cast<uint64_t>(kPlMaxSources) +
            static_cast<uint64_t>(dest_rank)) *
               64u;
}

inline uint64_t PlIncSecondHopSummaryOff(uint32_t inc_rank)
{
    return kPlIncSecondHopSummaryOff + static_cast<uint64_t>(inc_rank) * 64u;
}

inline uint64_t PlSegmentBaseIndex(uint32_t local_expert, uint32_t source_rank, uint32_t worker_count)
{
    return static_cast<uint64_t>(local_expert) * static_cast<uint64_t>(worker_count) +
           static_cast<uint64_t>(source_rank);
}

inline bool PlValidateRouteTopk(uint32_t route_topk)
{
    return route_topk >= 1u && route_topk <= kPlMaxTopk;
}

inline bool PlValidateHeapLayoutMonotonic()
{
    const uint64_t need = kPlTokenReadyOff;
    if (need > kPlD1RegionOff) {
        return false;
    }
    return kPlIncPayloadOff < kPlIncDescOff && kPlIncDescOff < kPlRouteCountRegionOff &&
           kPlRouteCountRegionOff < kPlExpectedCountOff && kPlExpectedCountOff < kPlSourceCountReadyOff &&
           kPlSourceCountReadyOff < kPlDestCountSliceReadyOff &&
           kPlDestCountSliceReadyOff < kPlSegmentBaseReadyOff &&
           kPlSegmentBaseReadyOff < kPlSegBasePublishStagingOff &&
           kPlSegBasePublishStagingOff < kPlOutboundSegBaseOff && kPlOutboundSegBaseOff < kPlSegmentBaseOff &&
           kPlSegmentBaseOff < kPlLaneWorkOff && kPlLaneWorkOff < kPlFullOutputDoneOff &&
           kPlFullOutputDoneOff < kPlRecvDoneOff && kPlRecvDoneOff < kPlRouteTimingOff &&
           kPlRouteTimingOff < kPlTransportDoneOff && kPlTransportDoneOff < kPlGlobalTransportDoneOff &&
           kPlGlobalTransportDoneOff < kPlTransportTimingOff &&
           kPlTransportTimingOff < kPlEgressLaneLocalVisibleOff &&
           kPlEgressLaneLocalVisibleOff < kPlEgressVisibleDoneOff &&
           kPlEgressVisibleDoneOff < kPlSecondHopVisibleTimingOff &&
           kPlSecondHopVisibleTimingOff < kPlGlobalSecondHopVisibleDoneOff &&
           kPlGlobalSecondHopVisibleDoneOff < kPlIncSecondHopSummaryOff &&
           kPlIncSecondHopSummaryOff < kPlSecondHopAggTelemetryOff &&
           kPlSecondHopAggTelemetryOff < kPlUploadLaneStageTimingOff &&
           kPlUploadLaneStageTimingOff < kPlDestinationCompletionTraceOff &&
           kPlDestinationCompletionTraceOff < kPlRecvChannelCompletionTraceOff &&
           kPlRecvChannelCompletionTraceOff < kPlIncForwardLaneStageTimingOff &&
           kPlIncForwardLaneStageTimingOff < kPlTimingNegOff && kPlTimingNegOff < kPlRankDispatchTimingOff &&
           kPlRankDispatchTimingOff < kPlHostProbeScratchOff &&
           kPlHostProbeScratchOff < kPlInvocationDescOff &&
           kPlInvocationDescOff + kPlInvocationDescBytes <= kPlInvocationWorkspaceOff &&
           kPlInvocationWorkspaceOff + kPlInvocationWorkspaceBytes <= kPlDestChannelPayloadOff &&
           kPlDestChannelPayloadOff < kPlDestChannelDescOff &&
           kPlDestChannelDescOff + kPlDestChannelDescRegionBytes <= kPlWorkerGatherDescStagingOff &&
           kPlWorkerGatherDescStagingOff + kPlWorkerGatherDescStagingBytes <= kPlTokenReadyOff &&
           kPlWorkerUploadSrcBytes == 0u;
}

// Full Dispatch：tokens=local_source_capacity；允许 0 有效 token、不要求整除 lane
// final_slot_capacity：0 => kPlMaxFinalSlots（旧 ABI 兼容）；非 0 => runtime contract
// out_reason / out_required_slots 供 CAP_NEG 等 fail-closed 真值（非 generic heap fail）
inline uint32_t PlResolveFinalSlotCapacity(uint32_t final_slot_capacity)
{
    return (final_slot_capacity == 0u) ? kPlMaxFinalSlots : final_slot_capacity;
}

inline uint32_t PlRequiredFinalSlots(uint32_t worker_count, uint32_t local_source_capacity, uint32_t route_topk)
{
    return worker_count * local_source_capacity * route_topk;
}

inline uint64_t PlWorkspaceFinalRegionBytes(uint32_t final_slot_capacity,
                                            uint32_t payload_bytes = kQv2TokenBytes)
{
    const uint64_t slots = static_cast<uint64_t>(final_slot_capacity);
    const uint32_t pb = (payload_bytes == 0u) ? kQv2TokenBytes : payload_bytes;
    return slots * static_cast<uint64_t>(pb) + slots * static_cast<uint64_t>(kPlDispatchAssistStrideBytes);
}

inline uint64_t PlWorkerGatherPayloadBytesFor(uint32_t payload_bytes)
{
    const uint32_t pb = (payload_bytes == 0u) ? kQv2TokenBytes : payload_bytes;
    return static_cast<uint64_t>(kQv2LaneCount) * kQv2RingDepth * static_cast<uint64_t>(pb);
}

// M4：Worker session local workspace 总字节（aclrtMalloc；按 runtime capacity + payload_bytes）
inline uint64_t PlWorkerLocalWorkspaceBytes(uint32_t source_capacity, uint32_t route_topk,
                                            uint32_t final_slot_capacity, uint32_t input_epoch_count,
                                            uint32_t expert_per_pe, uint32_t worker_count,
                                            uint32_t payload_bytes = kQv2TokenBytes)
{
    const uint32_t epochs = input_epoch_count == 0u ? 1u : input_epoch_count;
    const uint32_t pb = (payload_bytes == 0u) ? kQv2TokenBytes : payload_bytes;
    uint64_t input_bytes = 0u;
    uint64_t expand_part = 0u;
    if (!PlCheckedMulU64(static_cast<uint64_t>(source_capacity) * static_cast<uint64_t>(epochs), pb, &input_bytes)) {
        return 0u;
    }
    const uint64_t expert_ids_bytes =
        static_cast<uint64_t>(source_capacity) * static_cast<uint64_t>(route_topk) *
        static_cast<uint64_t>(epochs) * sizeof(int32_t);
    const uint64_t gather_bytes = PlWorkerGatherPayloadBytesFor(pb) + kPlWorkerGatherDescBytes;
    const uint64_t desc_bytes = kPlWorkerDescBytes;
    const uint64_t final_bytes = PlWorkspaceFinalRegionBytes(final_slot_capacity, pb);
    const uint64_t ep_recv_bytes =
        static_cast<uint64_t>(expert_per_pe) * static_cast<uint64_t>(worker_count) * sizeof(int32_t);
    const uint64_t expert_nums_bytes = static_cast<uint64_t>(expert_per_pe) * sizeof(int32_t);
    (void)expand_part;
    return input_bytes + expert_ids_bytes + gather_bytes + desc_bytes + final_bytes + ep_recv_bytes +
           expert_nums_bytes;
}

inline bool PlValidateWorkspaceContract(const PlWorkspaceContract *contract, const char **out_reason = nullptr)
{
    const auto fail = [&](const char *reason) -> bool {
        if (out_reason != nullptr) {
            *out_reason = reason;
        }
        return false;
    };
    if (contract == nullptr) {
        return fail("null_contract");
    }
    if (contract->magic != kPlWorkspaceContractMagic) {
        return fail("bad_magic");
    }
    if (contract->worker_count == 0u || contract->worker_count > kPlMaxSources ||
        contract->source_capacity == 0u || contract->source_capacity > kPlMaxTokens ||
        !PlValidateRouteTopk(contract->route_topk)) {
        return fail("bad_args");
    }
    if (contract->final_slot_capacity == 0u ||
        contract->final_slot_capacity > kPlMaxFinalSlotsCompileBound) {
        return fail("final_slot_capacity_overflow");
    }
    const uint32_t required_slots =
        PlRequiredFinalSlots(contract->worker_count, contract->source_capacity, contract->route_topk);
    if (required_slots > contract->final_slot_capacity) {
        return fail("final_slot_capacity");
    }
    if (contract->expert_ids_capacity < contract->source_capacity * contract->route_topk) {
        return fail("expert_ids_capacity");
    }
    // M4：final 区在 Worker local；symmetric contract 只校验容量，offset 必须为 0
    if (contract->final_payload_offset != 0u || contract->final_assist_offset != 0u) {
        return fail("final_offset_not_worker_local");
    }
    if (contract->ep_recv_count_offset != 0u) {
        return fail("ep_recv_count_offset");
    }
    if (contract->layout_version == 0u) {
        return fail("layout_version_unset");
    }
    if (contract->layout_version != kPlLayoutVersionRuntimePayloadV2 &&
        contract->layout_version != kPlLayoutVersionV1Fixed8192) {
        return fail("layout_version_unsupported");
    }
    if (!PlRuntimePayloadCrossCheck(contract->hidden_size, contract->dtype_bytes, contract->payload_bytes)) {
        return fail("runtime_payload_mismatch");
    }
    const uint64_t need_region = PlWorkspaceFinalRegionBytes(contract->final_slot_capacity, contract->payload_bytes);
    if (contract->workspace_bytes < need_region) {
        return fail("workspace_bytes");
    }
    if (out_reason != nullptr) {
        *out_reason = "ok";
    }
    return true;
}

// M2：校验 Worker 本地 workspace；legacy 模式下允许 ptr==0 并回退 symmetric DestFinal*。
inline bool PlValidateInvocationWorkspace(const PlInvocationWorkspaceDesc *ws, const char **out_reason = nullptr)
{
    const auto fail = [&](const char *reason) -> bool {
        if (out_reason != nullptr) {
            *out_reason = reason;
        }
        return false;
    };
    if (ws == nullptr) {
        return fail("null_workspace");
    }
    if (ws->magic != kPlInvocationWorkspaceMagic) {
        return fail("bad_magic");
    }
    if (ws->source_token_capacity == 0u || ws->source_token_capacity > kPlMaxTokens) {
        return fail("source_token_capacity");
    }
    if (ws->output_route_capacity == 0u || ws->output_route_capacity > kPlMaxFinalSlotsCompileBound) {
        return fail("output_route_capacity");
    }
    if (!PlValidateRouteTopk(ws->topk)) {
        return fail("topk");
    }
    if (ws->generation == 0u) {
        return fail("generation");
    }
    if (ws->layout_version != 0u) {
        if (ws->layout_version != kPlLayoutVersionRuntimePayloadV2 &&
            ws->layout_version != kPlLayoutVersionV1Fixed8192) {
            return fail("layout_version_unsupported");
        }
        if (!PlRuntimePayloadCrossCheck(ws->hidden_size, ws->dtype_bytes, ws->payload_bytes)) {
            return fail("runtime_payload_mismatch");
        }
    }
    const uint32_t pb = (ws->payload_bytes != 0u) ? ws->payload_bytes : kQv2TokenBytes;
    const bool legacy = (ws->flags & kPlInvocationWorkspaceFlagLegacySymmetricFinal) != 0u;
    const bool owned =
        (ws->flags & (kPlInvocationWorkspaceFlagCallerOwned | kPlInvocationWorkspaceFlagPooled)) != 0u;
    const bool local_final = (ws->flags & kPlInvocationWorkspaceFlagWorkerLocalFinal) != 0u;
    if (!legacy && !owned && !local_final) {
        return fail("flag_invalid");
    }
    if (legacy && (owned || local_final)) {
        return fail("flag_conflict");
    }
    if (legacy) {
        // M4：symmetric DestFinal 已移除；legacy 必须 fail-closed
        return fail("legacy_destfinal_removed_m4");
    }
    // LocalFinal / CallerOwned / Pooled：输出指针与容量必须齐备；alignment 最低 64B。
    const auto aligned64 = [](uint64_t p) -> bool { return p != 0u && (p % 64u) == 0u; };
    if (!aligned64(ws->expand_x_ptr)) {
        return fail("expand_x_ptr");
    }
    if (!aligned64(ws->assist_info_ptr)) {
        return fail("assist_info_ptr");
    }
    if (ws->ep_recv_count_ptr == 0u || (ws->ep_recv_count_ptr % 4u) != 0u) {
        return fail("ep_recv_count_ptr");
    }
    if (ws->expert_token_nums_ptr == 0u || (ws->expert_token_nums_ptr % 4u) != 0u) {
        return fail("expert_token_nums_ptr");
    }
    const uint64_t need_out =
        static_cast<uint64_t>(ws->output_route_capacity) * static_cast<uint64_t>(pb);
    const uint64_t need_assist =
        static_cast<uint64_t>(ws->output_route_capacity) * static_cast<uint64_t>(kPlDispatchAssistStrideBytes);
    if (ws->output_bytes < need_out) {
        return fail("output_bytes");
    }
    if (ws->assist_bytes < need_assist) {
        return fail("assist_bytes");
    }
    // input / expert_ids：允许 caller 稍后绑定，但若提供则必须对齐且容量足够。
    if (ws->input_ptr != 0u) {
        if ((ws->input_ptr % 64u) != 0u) {
            return fail("input_ptr_align");
        }
        const uint64_t need_in =
            static_cast<uint64_t>(ws->source_token_capacity) * static_cast<uint64_t>(pb);
        if (ws->input_bytes < need_in) {
            return fail("input_bytes");
        }
    }
    if (ws->expert_ids_ptr != 0u) {
        if ((ws->expert_ids_ptr % 4u) != 0u) {
            return fail("expert_ids_ptr_align");
        }
    }
    if (out_reason != nullptr) {
        *out_reason = "ok";
    }
    return true;
}

inline bool PlBuildLegacySymmetricInvocationWorkspace(PlInvocationWorkspaceDesc *out, uint32_t source_token_capacity,
                                                      uint32_t output_route_capacity, uint32_t topk,
                                                      uint32_t generation)
{
    (void)out;
    (void)source_token_capacity;
    (void)output_route_capacity;
    (void)topk;
    (void)generation;
    // M4：DestFinal 物理区已迁出 symmetric heap；禁止 silent legacy
    return false;
}

// M3：WorkerLocalFinal —— INC 只 put DestChannel；Worker 本地写 expand/assist。
// 过渡期 ptr 可指向 DestFinal 区（Worker 独占写，INC 禁止 remote put 到该区）。
inline bool PlBuildWorkerLocalFinalInvocationWorkspace(PlInvocationWorkspaceDesc *out, uint64_t expand_x_ptr,
                                                       uint64_t assist_info_ptr, uint64_t ep_recv_count_ptr,
                                                       uint64_t expert_token_nums_ptr,
                                                       uint32_t source_token_capacity,
                                                       uint32_t output_route_capacity, uint32_t topk,
                                                       uint32_t generation,
                                                       uint32_t payload_bytes = kQv2TokenBytes,
                                                       uint32_t hidden_size = 4096u,
                                                       uint32_t dtype_bytes = kPlDtypeBytesFp16,
                                                       uint32_t layout_version = kPlLayoutVersionRuntimePayloadV2)
{
    if (out == nullptr) {
        return false;
    }
    const uint32_t pb = (payload_bytes == 0u) ? kQv2TokenBytes : payload_bytes;
    *out = PlInvocationWorkspaceDesc{};
    out->magic = kPlInvocationWorkspaceMagic;
    out->flags = kPlInvocationWorkspaceFlagWorkerLocalFinal;
    out->source_token_capacity = source_token_capacity;
    out->output_route_capacity = output_route_capacity;
    out->topk = topk;
    out->generation = generation;
    out->expand_x_ptr = expand_x_ptr;
    out->assist_info_ptr = assist_info_ptr;
    out->ep_recv_count_ptr = ep_recv_count_ptr;
    out->expert_token_nums_ptr = expert_token_nums_ptr;
    out->input_bytes =
        static_cast<uint64_t>(source_token_capacity) * static_cast<uint64_t>(pb);
    out->output_bytes =
        static_cast<uint64_t>(output_route_capacity) * static_cast<uint64_t>(pb);
    out->assist_bytes =
        static_cast<uint64_t>(output_route_capacity) * static_cast<uint64_t>(kPlDispatchAssistStrideBytes);
    out->layout_version = layout_version;
    out->hidden_size = hidden_size;
    out->dtype_bytes = dtype_bytes;
    out->payload_bytes = pb;
    const char *reason = nullptr;
    return PlValidateInvocationWorkspace(out, &reason);
}

inline bool PlBuildWorkspaceContract(PlWorkspaceContract *out, uint32_t worker_count, uint32_t source_capacity,
                                     uint32_t route_topk, uint32_t final_slot_capacity, uint32_t workspace_generation,
                                     uint32_t payload_bytes = kQv2TokenBytes, uint32_t hidden_size = 4096u,
                                     uint32_t dtype_bytes = kPlDtypeBytesFp16,
                                     uint32_t layout_version = kPlLayoutVersionRuntimePayloadV2)
{
    if (out == nullptr) {
        return false;
    }
    const uint32_t slot_cap = PlResolveFinalSlotCapacity(final_slot_capacity);
    const uint32_t descriptor_capacity = static_cast<uint32_t>(kQv2LaneCount) * kQv2RingDepth;
    const uint32_t expert_ids_capacity = source_capacity * route_topk;
    const uint64_t workspace_bytes = PlWorkspaceFinalRegionBytes(slot_cap, payload_bytes);
    *out = PlWorkspaceContract{};
    out->magic = kPlWorkspaceContractMagic;
    out->workspace_generation = workspace_generation;
    out->source_capacity = source_capacity;
    out->route_topk = route_topk;
    out->final_slot_capacity = slot_cap;
    out->descriptor_capacity = descriptor_capacity;
    out->expert_ids_capacity = expert_ids_capacity;
    out->worker_count = worker_count;
    out->final_payload_offset = 0u;
    out->final_assist_offset = 0u;
    out->workspace_bytes = workspace_bytes;
    out->ep_recv_count_offset = 0u;
    out->layout_version = layout_version;
    out->hidden_size = hidden_size;
    out->dtype_bytes = dtype_bytes;
    out->payload_bytes = payload_bytes;
    const char *reason = nullptr;
    return PlValidateWorkspaceContract(out, &reason);
}

inline bool PlValidateFullDispatchShape(uint32_t local_source_capacity, uint32_t lanes, uint32_t worker_count,
                                        uint32_t route_topk, uint32_t input_epoch_count,
                                        const uint32_t *source_token_count /*nullable wc*/,
                                        uint32_t *out_total_routes = nullptr,
                                        const char **out_reason = nullptr,
                                        uint32_t *out_required_slots = nullptr,
                                        uint32_t final_slot_capacity = 0u)
{
    const auto fail = [&](const char *reason, uint32_t required = 0u) -> bool {
        if (out_reason != nullptr) {
            *out_reason = reason;
        }
        if (out_required_slots != nullptr) {
            *out_required_slots = required;
        }
        return false;
    };
    if (local_source_capacity == 0u || lanes == 0u || worker_count == 0u ||
        local_source_capacity > kPlMaxTokens || lanes > kQv2LaneCount || worker_count > kPlMaxSources ||
        !PlValidateRouteTopk(route_topk) || input_epoch_count == 0u ||
        input_epoch_count > kPlMaxInvocationEpochs) {
        return fail("bad_args");
    }
    const uint32_t slot_cap = PlResolveFinalSlotCapacity(final_slot_capacity);
    if (slot_cap > kPlMaxFinalSlotsCompileBound) {
        return fail("final_slot_capacity_overflow");
    }
    if (local_source_capacity * route_topk * input_epoch_count > kPlMaxExpertIdsSlots) {
        return fail("expert_ids_slots");
    }
    // dump/final 容量按 capacity 预留（非本 epoch 有效 token）；与 source capacity 解耦
    const uint32_t required_slots = PlRequiredFinalSlots(worker_count, local_source_capacity, route_topk);
    if (required_slots > slot_cap) {
        return fail("final_slot_capacity", required_slots);
    }
    uint32_t total_input = 0u;
    for (uint32_t s = 0; s < worker_count; ++s) {
        const uint32_t n = (source_token_count != nullptr) ? source_token_count[s] : local_source_capacity;
        if (n > local_source_capacity) {
            return fail("source_token_gt_capacity");
        }
        total_input += n;
    }
    const uint32_t total_routes = total_input * route_topk;
    // worst-case：全部 route 落同一 destination
    if (total_routes > slot_cap) {
        return fail("total_routes_capacity", total_routes);
    }
    if (out_total_routes != nullptr) {
        *out_total_routes = total_routes;
    }
    if (!PlValidateHeapLayoutMonotonic() || !PlValidateResourceMap()) {
        return fail("heap_layout");
    }
    if (out_reason != nullptr) {
        *out_reason = "ok";
    }
    if (out_required_slots != nullptr) {
        *out_required_slots = required_slots;
    }
    return true;
}

inline bool PlValidateHeap(uint32_t tokens, uint32_t lanes, uint32_t worker_count, uint32_t ring_depth = kQv2RingDepth,
                           uint32_t batch_tokens = kQv2PayloadBatchTokens, uint32_t tile_tokens = kQv2PublishTileTokens,
                           uint32_t route_topk = 1u, uint32_t input_epoch_count = 1u,
                           uint32_t measurement_mode = kPlMeasurementPrepackedTransport,
                           const uint32_t *source_token_count = nullptr, uint32_t final_slot_capacity = 0u)
{
    if (!PlValidateRingDepth(ring_depth) || !PlValidateBatchTile(batch_tokens, tile_tokens)) {
        return false;
    }
    if (measurement_mode == kPlMeasurementFullDispatch) {
        // shape/capacity 由 host 先做 PlValidateFullDispatchShape（带 failure_reason）；此处只校 layout
        (void)tokens;
        (void)lanes;
        (void)worker_count;
        (void)route_topk;
        (void)input_epoch_count;
        (void)source_token_count;
        (void)final_slot_capacity;
        return PlValidateHeapLayoutMonotonic() && PlValidateResourceMap();
    }
    if (tokens == 0 || lanes == 0 || worker_count == 0 || tokens > kPlMaxTokens || lanes > kQv2LaneCount ||
        worker_count > kPlMaxSources || !PlValidateRouteTopk(route_topk) || input_epoch_count == 0u ||
        (tokens % lanes) != 0) {
        return false;
    }
    if (tokens * route_topk * input_epoch_count > kPlMaxExpertIdsSlots) {
        return false;
    }
    const uint32_t slot_cap = PlResolveFinalSlotCapacity(final_slot_capacity);
    if (slot_cap > kPlMaxFinalSlotsCompileBound) {
        return false;
    }
    if (worker_count * tokens * route_topk > slot_cap) {
        return false;
    }
    return PlValidateHeapLayoutMonotonic() && PlValidateResourceMap();
}

inline uint32_t PlInvocationSlotIndex(uint32_t input_epoch_count, uint64_t go_epoch)
{
    const uint32_t n = input_epoch_count == 0u ? 1u : input_epoch_count;
    return static_cast<uint32_t>((go_epoch - 1u) % static_cast<uint64_t>(n));
}

inline bool PlInvocationSlotIsTemplate(const PlInvocationDesc *inv)
{
    return (inv->flags & kPlInvocationFlagsTemplateSlot) != 0u;
}

inline uint64_t PlInvocationDescLineOff(uint32_t epoch_index)
{
    return kPlInvocationDescOff + static_cast<uint64_t>(epoch_index) * sizeof(PlInvocationDesc);
}

inline uint64_t PlInvocationWorkspaceLineOff(uint32_t epoch_index)
{
    return kPlInvocationWorkspaceOff + static_cast<uint64_t>(epoch_index) * sizeof(PlInvocationWorkspaceDesc);
}

inline void PlFillInvocationDesc(PlInvocationDesc *inv, uint64_t epoch, uint64_t generation, uint32_t worker_count,
                                 uint32_t route_topk, const uint32_t *source_token_count,
                                 uint32_t local_source_capacity, uint32_t payload_bytes,
                                 const PlWorkspaceContract *workspace = nullptr,
                                 uint32_t hidden_size = 0u, uint32_t dtype_bytes = kPlDtypeBytesFp16,
                                 uint32_t layout_version = kPlLayoutVersionRuntimePayloadV2)
{
    inv->epoch = epoch;
    inv->generation = generation;
    inv->worker_count = worker_count;
    inv->route_topk = route_topk;
    uint32_t total_input = 0u;
    for (uint32_t s = 0; s < kPlMaxSources; ++s) {
        const uint32_t n = (s < worker_count) ? source_token_count[s] : 0u;
        inv->source_token_count[s] = n;
        total_input += n;
    }
    inv->total_input_tokens = total_input;
    inv->total_route_instances = total_input * route_topk;
    if (workspace != nullptr) {
        inv->final_slot_capacity = workspace->final_slot_capacity;
        inv->workspace_generation = workspace->workspace_generation;
        inv->final_payload_offset = workspace->final_payload_offset;
        inv->final_assist_offset = workspace->final_assist_offset;
        inv->descriptor_capacity = workspace->descriptor_capacity;
        inv->expert_ids_capacity = workspace->expert_ids_capacity;
        inv->workspace_bytes = workspace->workspace_bytes;
        if (workspace->payload_bytes != 0u) {
            payload_bytes = workspace->payload_bytes;
        }
        if (workspace->hidden_size != 0u) {
            hidden_size = workspace->hidden_size;
        }
        if (workspace->dtype_bytes != 0u) {
            dtype_bytes = workspace->dtype_bytes;
        }
        if (workspace->layout_version != 0u) {
            layout_version = workspace->layout_version;
        }
    } else {
        inv->final_slot_capacity = kPlMaxFinalSlots;
        inv->workspace_generation = 0u;
        inv->final_payload_offset = 0u;
        inv->final_assist_offset = 0u;
        inv->descriptor_capacity = static_cast<uint32_t>(kQv2LaneCount) * kQv2RingDepth;
        inv->expert_ids_capacity = local_source_capacity * route_topk;
        inv->workspace_bytes = PlWorkspaceFinalRegionBytes(kPlMaxFinalSlots, payload_bytes);
    }
    if (hidden_size == 0u && dtype_bytes != 0u && payload_bytes != 0u) {
        hidden_size = payload_bytes / dtype_bytes;
    }
    if (payload_bytes == 0u) {
        payload_bytes = kQv2TokenBytes;
        hidden_size = 4096u;
        dtype_bytes = kPlDtypeBytesFp16;
        layout_version = kPlLayoutVersionV1Fixed8192;
    }
    inv->local_source_capacity = local_source_capacity;
    inv->raw_input_epoch_stride_bytes =
        static_cast<uint64_t>(local_source_capacity) * static_cast<uint64_t>(payload_bytes);
    inv->raw_expert_ids_epoch_stride_bytes =
        static_cast<uint64_t>(local_source_capacity) * static_cast<uint64_t>(route_topk) * sizeof(int32_t);
    inv->flags = kPlInvocationFlagsHasRaggedSources;
    inv->layout_version = layout_version;
    inv->hidden_size = hidden_size;
    inv->dtype_bytes = dtype_bytes;
    inv->payload_bytes = payload_bytes;
    for (uint32_t i = 0; i < sizeof(inv->pad_v2); ++i) {
        inv->pad_v2[i] = 0u;
    }
}

inline bool PlInvocationPayloadValid(const PlInvocationDesc *inv)
{
    if (inv == nullptr) {
        return false;
    }
    if (inv->layout_version != kPlLayoutVersionRuntimePayloadV2 &&
        inv->layout_version != kPlLayoutVersionV1Fixed8192) {
        return false;
    }
    return PlRuntimePayloadCrossCheck(inv->hidden_size, inv->dtype_bytes, inv->payload_bytes);
}

inline PlFullDispatchConfig PlBuildFullDispatchConfig(uint32_t worker_count, uint32_t lane_count)
{
    PlFullDispatchConfig cfg{};
    cfg.magic = kPlMagic;
    cfg.version = 1u;
    cfg.worker_count = worker_count;
    cfg.lane_count = lane_count;
    cfg.raw_input_off = 0u;
    cfg.raw_expert_ids_off = 0u;
    cfg.gather_payload_off = 0u;
    cfg.gather_desc_off = 0u;
    cfg.segment_base_off = kPlSegmentBaseOff;
    cfg.final_payload_off = 0u; // M4：repurpose at runtime → worker upload desc base (device ptr)
    cfg.final_assist_off = 0u;
    cfg.ep_recv_count_off = 0u;
    cfg.expert_token_nums_off = 0u;
    cfg.lane_work_off = kPlLaneWorkOff;
    cfg.full_output_done_off = kPlFullOutputDoneOff;
    cfg.route_timing_off = kPlRouteTimingOff;
    cfg.first_hop_done_off = kPlFirstHopDoneOff;
    cfg.probe_mode = kPlFullDispatchProbeFull;
    cfg.input_epoch_count = 1u;
    return cfg;
}

// 8×8 balanced：lane d 对应 destination d
inline uint32_t PlBalancedDestinationRank(uint32_t lane_id, uint32_t worker_count)
{
    return lane_id < worker_count ? lane_id : lane_id;
}

} // namespace inc::dc::dn::pl
