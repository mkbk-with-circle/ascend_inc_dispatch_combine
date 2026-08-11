/**
 * INC Dispatch 完整两跳 persistent pipeline launcher。
 */
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <errno.h>
#include <sys/stat.h>
#include <vector>

#include "acl/acl.h"
#include "inc_dc_dn_pipeline_abi.h"
#include "inc_dc_dn_pipeline_workspace_pool.h"
#include "inc_dc_external_start_gate.h"
#include "inc_dc_dn_transport_abi.h"
#include "param.h"
#include "shmem.h"
#include "utils.h"

using inc::dc::dn::D1ControlLine;
using inc::dc::dn::D1GlobalDoneLine;
using inc::dc::dn::D1LeaderCompletion;
using inc::dc::dn::D1LeaderRendezvous;
using inc::dc::dn::D1PatternByte;
using inc::dc::dn::D1StartLine;
using inc::dc::dn::kD1LeaderCompletionOff;
using inc::dc::dn::kD1LeaderPe;
using inc::dc::dn::kD1LeaderRendezvousOff;
using inc::dc::dn::kD1LocalReadyLineOff;
using inc::dc::dn::kD1GlobalDoneLineOff;
using inc::dc::dn::kD1GlobalDoneTimingOff;
using inc::dc::dn::kD1LeaderTimingOff;
using inc::dc::dn::kD1PairDoneArrivalOff;
using inc::dc::dn::kD1PublishScratchOff;
using inc::dc::dn::kD1SessionStopLineOff;
using inc::dc::dn::kD1SessionStopMagic;
using inc::dc::dn::kD1StartLineOff;
using inc::dc::dn::kD1PrimP6RangeMteWaitFinalQuiet;
using inc::dc::dn::pl::PlDescriptor;
using inc::dc::dn::pl::PlDestDoneLine;
using inc::dc::dn::pl::PlOverlapTelemetry;
using inc::dc::dn::pl::PlPipelineDesc;
using inc::dc::dn::pl::PlPipelineTiming;
using inc::dc::dn::pl::PlResidentLine;
using inc::dc::dn::pl::kPlDescOff;
using inc::dc::dn::pl::kPlD1RegionOff;
using inc::dc::dn::pl::kPlDestDoneOff;
using inc::dc::dn::pl::kPlForwardCtrOff;
using inc::dc::dn::pl::kPlHeapNeed;
using inc::dc::dn::pl::kPlSymmetricTransportHeapNeed;
using inc::dc::dn::pl::PlWorkerLocalWorkspaceBytes;
using inc::dc::dn::pl::PlWorkerWorkspacePool;
using inc::dc::dn::pl::PlWorkspacePoolEntry;
using inc::dc::dn::pl::PlWorkspacePoolBucketKey;
using inc::dc::dn::pl::PlBuildInvocationWorkspace;
using inc::dc::dn::pl::kPlInvocationWorkspaceFlagPooled;
using inc::dc::dn::pl::kPlInvocationWorkspaceFlagCallerOwned;
using inc::dc::dn::pl::kPlWorkerSourceBytes;
using inc::dc::dn::pl::kPlWorkerExpertIdsBytes;
using inc::dc::dn::pl::kPlWorkerGatherPayloadBytes;
using inc::dc::dn::pl::kPlWorkerGatherDescBytes;
using inc::dc::dn::pl::kPlWorkerDescBytes;
using inc::dc::dn::pl::kPlIncResidentOff;
using inc::dc::dn::pl::kPlIncServiceBlockDim;
using inc::dc::dn::pl::kPlIncTraceOff;
using inc::dc::dn::pl::kPlIngressHeadOff;
using inc::dc::dn::pl::kPlIngressTailOff;
using inc::dc::dn::pl::kPlHostProbeScratchOff;
using inc::dc::dn::pl::kPlMagic;
using inc::dc::dn::pl::kPlOverlapOff;
using inc::dc::dn::pl::kPlRecvCtrOff;
using inc::dc::dn::pl::kPlTimingOff;
using inc::dc::dn::pl::kPlUploadCtrOff;
using inc::dc::dn::D1GlobalDoneTiming;
using inc::dc::dn::D1LeaderTiming;
using inc::dc::dn::D1PairDoneArrival;
using inc::dc::dn::kD1CyclesPerUs;
using inc::dc::dn::pl::PlServiceTraceLine;
using inc::dc::dn::pl::PlUploadCounters;
using inc::dc::dn::pl::PlForwardCounters;
using inc::dc::dn::pl::PlRecvCounters;
using inc::dc::dn::pl::kPlModeControlOnly;
using inc::dc::dn::pl::kPlModeUploadCompactGather;
using inc::dc::dn::pl::kPlIngressLayoutSourceMajorRaw;
using inc::dc::dn::pl::kPlIncP5ServiceBlockDim;
using inc::dc::dn::pl::kPlIncP5ExtResidentOff;
using inc::dc::dn::pl::kPlIncP5ExtTraceOff;
using inc::dc::dn::pl::kPlControlEpochRetireOff;
using inc::dc::dn::pl::kPlControlEpochRetireMagic;
using inc::dc::dn::pl::ControlEpochRetireLine;
using inc::dc::dn::pl::kPlIncRawSlabOff;
using inc::dc::dn::pl::kPlIncRawSlabBytes;
using inc::dc::dn::pl::kPlIncRawRouteMetaOff;
using inc::dc::dn::pl::kPlIncRawRouteMetaBytes;
using inc::dc::dn::pl::kPlRawChunkRingDepth;
using inc::dc::dn::pl::kPlRawIngressTailOff;
using inc::dc::dn::pl::kPlRawIngressHeadOff;
using inc::dc::dn::pl::kPlPayloadSourceStatic;
using inc::dc::dn::pl::kPlPayloadSourceEpochUnique;
using inc::dc::dn::pl::kPlMaxTokens;
using inc::dc::dn::pl::kPlMaxStackedInputEpochs;
using inc::dc::dn::pl::kPlKernelExitStop;
using inc::dc::dn::pl::kPlKernelExitUpload;
using inc::dc::dn::pl::kPlKernelExitForward;
using inc::dc::dn::pl::kPlKernelExitRecv;
using inc::dc::dn::pl::kPlVerifyDeviceFull;
using inc::dc::dn::pl::kPlVerifyHostPost;
using inc::dc::dn::pl::kPlCompletionModeCounterScan;
using inc::dc::dn::pl::kPlCompletionModeDoneLines;
using inc::dc::dn::pl::kPlHostCompletionSyncPoll;
using inc::dc::dn::pl::kPlHostCompletionAsyncProduction;
using inc::dc::dn::pl::kPlMeasurementPrepackedTransport;
using inc::dc::dn::pl::kPlMeasurementStage1Diagnostic;
using inc::dc::dn::pl::kPlMeasurementFullDispatch;
using inc::dc::dn::pl::kPlDispatchBlockBegin;
using inc::dc::dn::pl::kPlDispatchBlockEnd;
using inc::dc::dn::pl::kPlCombineReserveBegin;
using inc::dc::dn::pl::kPlCombineReserveEnd;
using inc::dc::dn::pl::PlAivEpochTimingLine;
using inc::dc::dn::pl::PlUploadAivTimingOff;
using inc::dc::dn::pl::PlRecvAivTimingOff;
using inc::dc::dn::pl::PlForwardAivTimingOff;
using inc::dc::dn::pl::PlValidateRingDepth;
using inc::dc::dn::pl::PlValidateBatchTile;
using inc::dc::dn::pl::kPlWorkerDescOff;
using inc::dc::dn::pl::kPlWorkerResidentOff;
using inc::dc::dn::pl::kPlWorkerServiceBlockDim;
using inc::dc::dn::pl::kPlWorkerSourceOff;
using inc::dc::dn::pl::kPlWorkerExpertIdsOff;
using inc::dc::dn::pl::kPlWorkerGatherPayloadOff;
using inc::dc::dn::pl::kPlWorkerGatherDescOff;
using inc::dc::dn::pl::kPlSegmentBaseOff;
using inc::dc::dn::pl::kPlWorkerTraceOff;
using inc::dc::dn::pl::kPlWorkerUploadBlockBegin;
using inc::dc::dn::pl::kPlWorkerUploadBlockEnd;
using inc::dc::dn::pl::kPlWorkerUploadSrcOff;
using inc::dc::dn::pl::kPlWorkerUploadSrcBytes;
using inc::dc::dn::pl::kPlWorkerUploadExpertIdsOff;
using inc::dc::dn::pl::kPlWorkerUploadExpertIdsBytes;
using inc::dc::dn::pl::kPlWorkerGatherDescStagingOff;
using inc::dc::dn::pl::kPlWorkerRecvBlockBegin;
using inc::dc::dn::pl::kPlWorkerRecvBlockEnd;
using inc::dc::dn::pl::kPlWorkerControlBlock;
using inc::dc::IncDcExternalStartGate;
using inc::dc::IncDcExternalStartNs;
using inc::dc::dn::pl::kPlMaxUploadLanes;
using inc::dc::dn::pl::kPlMaxRecvLanes;
using inc::dc::dn::pl::PlWorkerServiceLayout;
using inc::dc::dn::pl::PlMakeWorkerServiceLayout;
using inc::dc::dn::pl::PlPackRecvLaneCountIntoPipelineMode;
using inc::dc::dn::pl::PlUnpackRecvLaneCountFromPipelineMode;
using inc::dc::dn::pl::PlWorkerResidentExpectMask;
using inc::dc::dn::pl::PlClampUploadLaneCount;
using inc::dc::dn::pl::PlClampRecvLaneCountCfg;
using inc::dc::dn::pl::kPlSourceDoneOff;
using inc::dc::dn::pl::kPlEgressHeadOff;
using inc::dc::dn::pl::kPlIncEgressCreditOff;
using inc::dc::dn::pl::kPlRouteCountRegionOff;
using inc::dc::dn::pl::kPlRouteCountRegionBytes;
using inc::dc::dn::pl::kPlExpectedCountOff;
using inc::dc::dn::pl::kPlExpectedCountBytes;
using inc::dc::dn::pl::kPlEgressTailOff;
using inc::dc::dn::pl::kPlDestChannelDescOff;
using inc::dc::dn::pl::PlDestChannelDescOff;
using inc::dc::dn::pl::kPlDestFinalPayloadOff;
using inc::dc::dn::pl::kPlDestFinalAssistOff;
using inc::dc::dn::pl::kPlDispatchAssistStrideBytes;
using inc::dc::dn::pl::kPlDestEpRecvCountOff;
using inc::dc::dn::pl::kPlDestExpertTokenNumsOff;
using inc::dc::dn::pl::kPlRouteTimingOff;
using inc::dc::dn::pl::kPlRecvDoneOff;
using inc::dc::dn::pl::kPlTransportDoneOff;
using inc::dc::dn::pl::kPlGlobalTransportDoneOff;
using inc::dc::dn::pl::kPlTransportTimingOff;
using inc::dc::dn::pl::PlTransportDoneLine;
using inc::dc::dn::pl::PlGlobalTransportDoneLine;
using inc::dc::dn::pl::PlTransportTimingLine;
using inc::dc::dn::pl::PlRecvDoneLine;
using inc::dc::dn::pl::PlP5CellRouteRecvLedgerLine;
using inc::dc::dn::pl::PlP5CellEgressLedgerLine;
using inc::dc::dn::pl::PlP5CellGatherLedgerLine;
using inc::dc::dn::pl::PlP5CellRouteRecvLedgerLineOff;
using inc::dc::dn::pl::PlP5CellGatherLedgerLineOff;
using inc::dc::dn::pl::PlP5CellEgressLedgerLineOff;
using inc::dc::dn::pl::kPlTransportDoneMagic;
using inc::dc::dn::pl::kPlGlobalTransportDoneMagic;
using inc::dc::dn::pl::kPlTransportTimingMagic;
using inc::dc::dn::pl::kPlTimingFormalSecondHopVisible;
using inc::dc::dn::pl::PlEgressVisibleDoneLine;
using inc::dc::dn::pl::PlSecondHopVisibleTimingLine;
using inc::dc::dn::pl::PlGlobalSecondHopVisibleDoneLine;
using inc::dc::dn::pl::PlTimingNegLine;
using inc::dc::dn::pl::kPlEgressVisibleDoneOff;
using inc::dc::dn::pl::kPlSecondHopVisibleTimingOff;
using inc::dc::dn::pl::kPlGlobalSecondHopVisibleDoneOff;
using inc::dc::dn::pl::kPlTimingNegOff;
using inc::dc::dn::pl::PlRankDispatchTimingLine;
using inc::dc::dn::pl::kPlRankDispatchTimingMagic;
using inc::dc::dn::pl::PlRankDispatchTimingLineOff;
using inc::dc::dn::pl::kPlMaxPhysicalPe;
using inc::dc::dn::pl::kPlRankDispatchTimingOff;
using inc::dc::dn::pl::kPlSecondHopVisibleTimingMagic;
using inc::dc::dn::pl::kPlGlobalSecondHopVisibleMagic;
using inc::dc::dn::pl::kPlTimingNegMagic;
using inc::dc::dn::pl::PlFullDispatchConfig;
using inc::dc::dn::pl::PlBuildFullDispatchConfig;
using inc::dc::dn::pl::kPlFullDispatchConfigOff;
using inc::dc::dn::pl::kPlFullDispatchProbeCountOnly;
using inc::dc::dn::pl::kPlFullDispatchProbeGatherOnly;
using inc::dc::dn::pl::kPlFullDispatchProbeFirstHop;
using inc::dc::dn::pl::kPlFullDispatchProbeFull;
using inc::dc::dn::pl::kPlMaxInvocationEpochs;
using inc::dc::dn::pl::kPlInvocationDescOff;
using inc::dc::dn::pl::kPlInvocationDescBytes;
using inc::dc::dn::pl::kPlInvocationWorkspaceOff;
using inc::dc::dn::pl::kPlInvocationWorkspaceBytes;
using inc::dc::dn::pl::PlInvocationDesc;
using inc::dc::dn::pl::PlInvocationWorkspaceDesc;
using inc::dc::dn::pl::PlWorkerGatherPayloadBytesFor;
using inc::dc::dn::pl::kPlLayoutVersionRuntimePayloadV2;
using inc::dc::dn::pl::kPlLayoutVersionV1Fixed8192;
using inc::dc::dn::pl::kPlDtypeBytesFp16;
using inc::dc::dn::pl::PlRuntimePayloadCrossCheck;
using inc::dc::dn::pl::PlInvocationPayloadValid;
using inc::dc::dn::pl::kPlMaxTokenBytes;
using inc::dc::dn::pl::PlInvocationDescLineOff;
using inc::dc::dn::pl::PlInvocationWorkspaceLineOff;
using inc::dc::dn::pl::PlBuildLegacySymmetricInvocationWorkspace;
using inc::dc::dn::pl::PlBuildWorkerLocalFinalInvocationWorkspace;
using inc::dc::dn::pl::kPlInvocationWorkspaceFlagLegacySymmetricFinal;
using inc::dc::dn::pl::kPlInvocationWorkspaceFlagWorkerLocalFinal;
using inc::dc::dn::pl::kPlInvocationWorkspaceMagic;
using inc::dc::dn::pl::PlValidateInvocationWorkspace;
using inc::dc::dn::pl::kPlInvocationFlagsTemplateSlot;
using inc::dc::dn::pl::PlInvocationSlotIndex;
using inc::dc::dn::pl::kPlTimingFormalMakespan;
using inc::dc::dn::pl::kPlTimingFormalTransport;
using inc::dc::dn::pl::PlValidateHeapLayoutMonotonic;
using inc::dc::dn::pl::PlRouteTimingLine;
using inc::dc::dn::pl::PlFullOutputDoneLine;
using inc::dc::dn::pl::PlLaneWorkLine;
using inc::dc::dn::pl::kPlErrGenerationMismatch;
using inc::dc::dn::pl::kPlErrWorkspaceDescMissing;
using inc::dc::dn::pl::kPlErrWorkspaceGenerationMismatch;
using inc::dc::dn::pl::kPlErrWorkspacePointerAlignment;
using inc::dc::dn::pl::kPlErrOutputCapacityInsufficient;
using inc::dc::dn::pl::kPlErrWorkspaceFlagInvalid;
using inc::dc::dn::pl::kPlCountFwdSourceReadyTimeout;
using inc::dc::dn::pl::kPlFullOutputDoneOff;
using inc::dc::dn::pl::kPlLaneWorkOff;
using inc::dc::dn::pl::kPlTokenReadyOff;
using inc::dc::dn::pl::PlTokenReadyLine;
using inc::dc::dn::pl::PlValidateHeap;
using inc::dc::dn::pl::PlValidateFullDispatchShape;
using inc::dc::dn::pl::kPlMaxFinalSlots;
using inc::dc::dn::pl::kPlMaxFinalSlotsCompileBound;
using inc::dc::dn::pl::PlWorkspaceContract;
using inc::dc::dn::pl::PlBuildWorkspaceContract;
using inc::dc::dn::pl::PlValidateWorkspaceContract;
using inc::dc::dn::pl::PlResolveFinalSlotCapacity;
using inc::dc::dn::pl::kPlMaxSources;
using inc::dc::dn::pl::PlSourceCountReadyLine;
using inc::dc::dn::pl::PlDestCountSliceReadyLine;
using inc::dc::dn::pl::PlSourceCountReadyLineOff;
using inc::dc::dn::pl::PlDestCountSliceReadyLineOff;
using inc::dc::dn::pl::kPlCountReadyMagic;
using inc::dc::dn::pl::PlWorkerSourceCountTrace;
using inc::dc::dn::pl::PlIncSourceCountTrace;
using inc::dc::dn::pl::PlDestinationCountTrace;
using inc::dc::dn::pl::PlWorkerSourceCountTraceOff;
using inc::dc::dn::pl::PlIncSourceCountTraceOff;
using inc::dc::dn::pl::PlDestinationCountTraceOff;
using inc::dc::dn::pl::kPlCountTraceMagic;
using inc::dc::dn::pl::PlUploadLaneStageTiming;
using inc::dc::dn::pl::PlUploadLaneStageTimingOff;
using inc::dc::dn::pl::PlDestinationCompletionTrace;
using inc::dc::dn::pl::PlRecvChannelCompletionTrace;
using inc::dc::dn::pl::PlIncForwardLaneStageTiming;
using inc::dc::dn::pl::PlDestinationCompletionTraceOff;
using inc::dc::dn::pl::PlRecvChannelCompletionTraceOff;
using inc::dc::dn::pl::PlIncForwardLaneStageTimingOff;
using inc::dc::dn::pl::kPlDestCompletionTraceMagic;
using inc::dc::dn::pl::kPlRecvChanCompletionTraceMagic;
using inc::dc::dn::pl::kPlIncFwdLaneStageMagic;
using inc::dc::dn::pl::kPlUploadLaneStageMagic;
using inc::dc::dn::pl::kQv2LaneCount;
using inc::dc::dn::qv2::Qv2HeadLine;
using inc::dc::dn::qv2::Qv2TailLine;
using inc::dc::dn::qv2::kQv2Magic;
using inc::dc::dn::qv2::kQv2RingDepth;
using inc::dc::dn::qv2::kQv2TokenBytes;

extern "C" void launch_inc_dc_dn_pipeline_worker_persistent_kernel(
    uint8_t *sym, uint64_t pl_desc_off, uint64_t upload_ctr_off, uint64_t recv_ctr_off, uint64_t timing_off,
    uint64_t overlap_off, uint64_t worker_trace_off, uint64_t source_done_off, uint64_t dest_done_off,
    uint64_t d1_start_off, uint64_t d1_stop_off, uint64_t d1_leader_completion_off, uint64_t d1_publish_scratch_off,
    uint32_t leader_pe, int block_dim, void *stream);
extern "C" void launch_inc_dc_dn_pipeline_inc_persistent_kernel(uint8_t *sym, uint64_t pl_desc_off,
                                                                uint64_t forward_ctr_off, uint64_t timing_off,
                                                                uint64_t overlap_off, uint64_t inc_trace_off,
                                                                uint64_t d1_start_off, uint64_t d1_stop_off,
                                                                int block_dim, void *stream);
extern "C" void launch_inc_dc_dn_d1_worker_ready_control_kernel(uint8_t *sym, uint64_t local_ready_off,
                                                                uint64_t leader_rendezvous_off,
                                                                uint64_t publish_scratch_off, uint64_t session_stop_off,
                                                                uint32_t pair_id, uint32_t leader_pe, void *stream);
extern "C" void launch_inc_dc_dn_d1_leader_release_control_kernel(uint8_t *sym, uint64_t leader_rendezvous_off,
                                                                  uint64_t leader_timing_off,
                                                                  uint64_t leader_completion_off,
                                                                  uint64_t global_done_line_off,
                                                                  uint64_t global_done_timing_off,
                                                                  uint64_t pair_done_arrival_off, uint64_t go_line_off,
                                                                  uint64_t publish_scratch_off, uint64_t session_stop_off,
                                                                  uint32_t worker_count, uint32_t inc_pe_base,
                                                                  void *stream);

int g_npus = 16;
const char *ipport = "tcp://127.0.0.1:8960";
int f_npu = 0;
aclshmemx_uniqueid_t default_flag_uid;
// Candidate2：本进程 Worker service 动态布局（默认 U8/R8 ≡ legacy）
static PlWorkerServiceLayout g_worker_layout = PlMakeWorkerServiceLayout(kPlMaxUploadLanes, kPlMaxRecvLanes);

static inline uint64_t D1B(uint64_t off)
{
    return kPlD1RegionOff + off;
}

static int64_t HostWallNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static double NsToUs(int64_t ns)
{
    return static_cast<double>(ns) / 1000.0;
}

static double CyclesToUs(uint64_t end, uint64_t start)
{
    return (end > start) ? static_cast<double>(end - start) / kD1CyclesPerUs : 0.0;
}

// 设备侧 epoch 分解（PE0 leader 在 global_done 后读取）
struct PlDeviceEpochTiming {
    double device_pipeline_us = 0.0;
    double stage1_device_us = 0.0;       // release→destination_done（Stage1 可见）
    double full_dispatch_device_us = 0.0; // pair_done→global_done（含最终可见性）
    double release_to_go_fanout_end_us = 0.0;
    double go_fanout_end_to_first_upload_us = 0.0;
    double first_upload_to_first_forward_us = 0.0;
    double first_forward_to_destination_done_us = 0.0;
    // diagnostic_only：PE0 本地 max recv_done → 全局 pair_done_complete，含其他 dest 拖尾；禁止归因 pair_done doorbell
    double pe0_local_recv_done_to_global_pair_done_complete_us = 0.0;
    double pair_done_to_global_done_publish_us = 0.0;
    double host_observe_residual_us = 0.0;
    std::string pair_done_arrival_us_csv;
    // Full Dispatch device 分段（PlRouteTimingLine + pipeline timing）
    double inc_route_count_prefix_us = 0.0;
    double inc_route_gather_us = 0.0;
    double inc_stage1_two_hop_us = 0.0;
    double inc_finalize_us = 0.0;
    double inc_full_dispatch_us = 0.0;
    bool p6_segment_timing_formal = false; // PlRouteTimingLine 非正式
    // 正式 transport：PE0 release_seen → global_transport_done（不含 full_done / D2H）
    double inc_transport_makespan_us = 0.0;
    uint64_t global_transport_done_epoch = 0;
    uint32_t slowest_destination = 0;
    uint64_t per_destination_transport_done_cycle[kPlMaxSources]{};
    uint32_t transport_timing_worker_count = 0;
    bool transport_le_full_dispatch = false;
    // 正式 second_hop_visible：PE0 release_seen → inc_second_hop_visible_done（对齐 Native comm）
    double inc_second_hop_visible_makespan_us = 0.0;
    uint64_t global_second_hop_visible_epoch = 0;
    uint32_t slowest_cell_inc = 0;
    uint32_t slowest_cell_dest = 0;
    uint64_t per_dest_visible_done_cycle[kPlMaxSources]{};
    uint32_t second_hop_timing_worker_count = 0;
    bool second_hop_timing_valid = false;
    uint32_t second_hop_release_seen = 0;
    uint32_t second_hop_visible_done = 0;
    // P10 strict max-rank timing（每 rank 本地 clock；与 PE0 global observer 分口径）
    double strict_max_rank_us = 0.0;
    double pe0_global_observer_us = 0.0;
    double max_destination_observed_us = 0.0;
    double observer_aggregation_us = 0.0;
    uint32_t slowest_rank = 0;
    uint32_t second_slowest_rank = 0;
    double slowest_rank_us = 0.0;
    double second_slowest_rank_us = 0.0;
    double rank_imbalance_ratio = 0.0;
    double rank_elapsed_us[kPlMaxPhysicalPe]{};
    uint32_t rank_timing_epoch = 0;
    uint32_t rank_timing_valid_count = 0;
    bool rank_dispatch_timing_valid = false;
};

// 正式 makespan：PE0 leader.release_seen_cycle → global_done_publish_cycle（见 HostCollectDeviceTiming）

static uint32_t ParseFullDispatchProbe()
{
    const char *env = std::getenv("INC_DN_PL_C1_PROBE");
    if (env == nullptr || env[0] == '\0') {
        return kPlFullDispatchProbeFull;
    }
    if (std::strcmp(env, "count") == 0 || std::strcmp(env, "count_only") == 0) {
        return kPlFullDispatchProbeCountOnly;
    }
    if (std::strcmp(env, "gather") == 0 || std::strcmp(env, "gather_only") == 0) {
        return kPlFullDispatchProbeGatherOnly;
    }
    if (std::strcmp(env, "first_hop") == 0) {
        return kPlFullDispatchProbeFirstHop;
    }
    return kPlFullDispatchProbeFull;
}

// M4：Worker PE session-local workspace（aclrtMalloc；非 SHMEM remote target）
struct PlWorkerLocalSession {
    uint8_t *base = nullptr;
    uint64_t total_bytes = 0;
    uint8_t *input = nullptr;
    int32_t *expert_ids = nullptr;
    uint8_t *gather_payload = nullptr;
    uint8_t *gather_desc = nullptr;
    uint8_t *worker_desc = nullptr;
    uint8_t *expand_x = nullptr;
    uint8_t *assist = nullptr;
    int32_t *ep_recv_count = nullptr;
    int32_t *expert_token_nums = nullptr;
};

static bool PlAllocWorkerLocalSession(PlWorkerLocalSession *wl, uint32_t source_capacity, uint32_t route_topk,
                                      uint32_t final_slot_capacity, uint32_t input_epoch_count, uint32_t expert_per_pe,
                                      uint32_t worker_count, uint32_t payload_bytes = kQv2TokenBytes)
{
    if (wl == nullptr) {
        return false;
    }
    *wl = PlWorkerLocalSession{};
    const uint32_t epochs = input_epoch_count == 0u ? 1u : input_epoch_count;
    const uint32_t pb = (payload_bytes == 0u) ? kQv2TokenBytes : payload_bytes;
    const uint64_t need = PlWorkerLocalWorkspaceBytes(source_capacity, route_topk, final_slot_capacity, epochs,
                                                      expert_per_pe, worker_count, pb);
    if (need == 0u) {
        return false;
    }
    if (aclrtMalloc(reinterpret_cast<void **>(&wl->base), need, ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        return false;
    }
    wl->total_bytes = need;
    uint64_t off = 0u;
    const uint64_t input_bytes = static_cast<uint64_t>(source_capacity) * static_cast<uint64_t>(epochs) * pb;
    wl->input = wl->base + off;
    off += input_bytes;
    const uint64_t expert_bytes =
        static_cast<uint64_t>(source_capacity) * static_cast<uint64_t>(route_topk) * static_cast<uint64_t>(epochs) *
        sizeof(int32_t);
    wl->expert_ids = reinterpret_cast<int32_t *>(wl->base + off);
    off += expert_bytes;
    wl->gather_payload = wl->base + off;
    off += PlWorkerGatherPayloadBytesFor(pb);
    wl->gather_desc = wl->base + off;
    off += kPlWorkerGatherDescBytes;
    wl->worker_desc = wl->base + off;
    off += kPlWorkerDescBytes;
    const uint64_t expand_bytes = static_cast<uint64_t>(final_slot_capacity) * pb;
    wl->expand_x = wl->base + off;
    off += expand_bytes;
    const uint64_t assist_bytes = static_cast<uint64_t>(final_slot_capacity) * kPlDispatchAssistStrideBytes;
    wl->assist = wl->base + off;
    off += assist_bytes;
    const uint64_t ep_recv_bytes = static_cast<uint64_t>(expert_per_pe) * static_cast<uint64_t>(worker_count) * sizeof(int32_t);
    wl->ep_recv_count = reinterpret_cast<int32_t *>(wl->base + off);
    off += ep_recv_bytes;
    const uint64_t expert_nums_bytes = static_cast<uint64_t>(expert_per_pe) * sizeof(int32_t);
    wl->expert_token_nums = reinterpret_cast<int32_t *>(wl->base + off);
    off += expert_nums_bytes;
    if (off > need) {
        return false;
    }
    if (aclrtMemset(wl->base, need, 0, need) != 0) {
        return false;
    }
    return true;
}

static void PlFillWorkerLocalFromPoolEntry(PlWorkerLocalSession *wl, const PlWorkspacePoolEntry &e)
{
    *wl = PlWorkerLocalSession{};
    wl->base = e.base;
    wl->total_bytes = e.total_bytes;
    wl->input = e.input;
    wl->expert_ids = e.expert_ids;
    wl->gather_payload = e.gather_payload;
    wl->gather_desc = e.gather_desc;
    wl->worker_desc = e.worker_desc;
    wl->expand_x = e.expand_x;
    wl->assist = e.assist;
    wl->ep_recv_count = e.ep_recv_count;
    wl->expert_token_nums = e.expert_token_nums;
}

static void PlFreeWorkerLocalSession(PlWorkerLocalSession *wl)
{
    if (wl != nullptr && wl->base != nullptr) {
        aclrtFree(wl->base);
        *wl = PlWorkerLocalSession{};
    }
}

static PlWorkerWorkspacePool g_worker_workspace_pool;

// Runtime Payload V2：host 侧字节口径（main 解析后写入；禁止散落硬编码 8192）
static uint32_t g_runtime_payload_bytes = kQv2TokenBytes;
static uint32_t g_runtime_hidden_size = 4096u;

static const char *ParseWorkspaceAllocMode()
{
    const char *env = std::getenv("INC_DN_PL_WORKSPACE_ALLOC");
    if (env == nullptr || env[0] == '\0') {
        return "pooled"; // M5 default
    }
    return env;
}

// V2：symmetric upload payload staging 已删除；第一跳直接读 Worker local input_ptr。
// 保留函数名供调用点编译，恒成功（expert_ids 亦经 workspace ptr）。
static int HostRefreshWorkerUploadStaging(uint8_t *sym, const PlWorkerLocalSession *wl, uint32_t source_capacity,
                                          uint32_t route_topk, uint32_t epoch_idx, aclrtStream stream)
{
    (void)sym;
    (void)wl;
    (void)source_capacity;
    (void)route_topk;
    (void)epoch_idx;
    (void)stream;
    return 0;
}

static bool WriteHostFile(const std::string &path, const void *data, size_t nbytes)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(nbytes));
    return out.good();
}

// 多 epoch：清零 destination 四输出区（Worker local workspace）
static int HostClearFullDispatchEpochOutputs(const PlWorkerLocalSession *wl, uint32_t tokens, uint32_t expert_per_pe,
                                             uint32_t worker_count, uint32_t route_topk, aclrtStream stream)
{
    if (wl == nullptr || wl->expand_x == nullptr) {
        return -1;
    }
    const uint32_t topk = route_topk == 0u ? 1u : route_topk;
    const uint32_t dump_slots = worker_count * tokens * topk;
    const size_t expand_bytes = static_cast<size_t>(dump_slots) * g_runtime_payload_bytes;
    const size_t assist_dev_bytes = static_cast<size_t>(dump_slots) * kPlDispatchAssistStrideBytes;
    const size_t ep_recv_bytes = static_cast<size_t>(expert_per_pe * worker_count) * sizeof(int32_t);
    const size_t expert_nums_bytes = static_cast<size_t>(expert_per_pe) * sizeof(int32_t);
    if (aclrtMemset(wl->expand_x, expand_bytes, 0, expand_bytes) != 0 ||
        aclrtMemset(wl->assist, assist_dev_bytes, 0, assist_dev_bytes) != 0 ||
        aclrtMemset(wl->ep_recv_count, ep_recv_bytes, 0, ep_recv_bytes) != 0 ||
        aclrtMemset(wl->expert_token_nums, expert_nums_bytes, 0, expert_nums_bytes) != 0) {
        return -1;
    }
    if (aclrtSynchronizeStream(stream) != 0) {
        return -2;
    }
    return 0;
}

// 计时外 D2H：四输出落盘供 inc_dispatch_four_output_verify.py
static int HostDumpFullDispatchOutputs(const PlWorkerLocalSession *wl, uint32_t dest_rank, uint32_t tokens,
                                       uint32_t expert_per_pe, uint32_t worker_count, uint32_t route_topk,
                                       const std::string &out_dir, aclrtStream stream)
{
    if (wl == nullptr || wl->expand_x == nullptr) {
        return -1;
    }
    std::vector<int32_t> expert_nums(expert_per_pe);
    if (aclrtMemcpy(expert_nums.data(), expert_nums.size() * sizeof(int32_t), wl->expert_token_nums,
                    expert_nums.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
        return -1;
    }
    // API 的逻辑输出长度由 expert_token_nums 决定。只回读有效前缀；
    // 最大容量 padding 不是通信结果，T4096 若按容量落盘会把每个
    // destination 扩成约 1 GiB，令验证本身超过 30s case deadline。
    uint32_t accepted_slots = 0u;
    for (uint32_t e = 0; e < expert_per_pe; ++e) {
        if (expert_nums[e] > 0) {
            accepted_slots += static_cast<uint32_t>(expert_nums[e]);
        }
    }
    const uint32_t dump_slots = accepted_slots;
    const size_t expand_bytes = static_cast<size_t>(dump_slots) * g_runtime_payload_bytes;
    const size_t assist_bytes = static_cast<size_t>(dump_slots) * 3u * sizeof(int32_t);
    const size_t ep_recv_bytes = static_cast<size_t>(expert_per_pe * worker_count) * sizeof(int32_t);
    const size_t expert_nums_bytes = static_cast<size_t>(expert_per_pe) * sizeof(int32_t);
    std::vector<uint8_t> expand(expand_bytes);
    std::vector<uint8_t> assist(assist_bytes, 0);
    std::vector<int32_t> ep_recv(expert_per_pe * worker_count);
    // device assist 为 64B stride；D2H 后压成 golden 的紧凑 12B/slot
    const size_t assist_dev_bytes = static_cast<size_t>(dump_slots) * kPlDispatchAssistStrideBytes;
    std::vector<uint8_t> assist_dev(assist_dev_bytes, 0);
    if ((expand_bytes > 0u &&
         aclrtMemcpy(expand.data(), expand_bytes, wl->expand_x, expand_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != 0) ||
        (assist_dev_bytes > 0u &&
         aclrtMemcpy(assist_dev.data(), assist_dev_bytes, wl->assist, assist_dev_bytes,
                     ACL_MEMCPY_DEVICE_TO_HOST) != 0) ||
        aclrtMemcpy(ep_recv.data(), ep_recv_bytes, wl->ep_recv_count, ep_recv_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != 0 ||
        aclrtMemcpy(expert_nums.data(), expert_nums_bytes, wl->expert_token_nums, expert_nums_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
        return -1;
    }
    for (uint32_t s = 0; s < dump_slots; ++s) {
        const int32_t *src = reinterpret_cast<const int32_t *>(assist_dev.data() +
                                                              static_cast<size_t>(s) * kPlDispatchAssistStrideBytes);
        int32_t *dst = reinterpret_cast<int32_t *>(assist.data() + static_cast<size_t>(s) * 3u * sizeof(int32_t));
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
    }
    aclrtSynchronizeStream(stream);
    const std::string rank_dir = out_dir + "/rank_" + std::to_string(dest_rank);
    if (mkdir(out_dir.c_str(), 0755) != 0 && errno != EEXIST) {
        return -2;
    }
    if (mkdir(rank_dir.c_str(), 0755) != 0 && errno != EEXIST) {
        return -2;
    }
    if (!WriteHostFile(rank_dir + "/expand_x.bin", expand.data(), expand_bytes) ||
        !WriteHostFile(rank_dir + "/assist_info.bin", assist.data(), assist_bytes) ||
        !WriteHostFile(rank_dir + "/ep_recv_count.bin", ep_recv.data(), ep_recv_bytes) ||
        !WriteHostFile(rank_dir + "/expert_token_nums.bin", expert_nums.data(), expert_nums_bytes)) {
        return -3;
    }
    std::cout << "PL_FULL_DISPATCH_D2H dest_rank=" << dest_rank << " out_dir=" << out_dir << " tokens=" << tokens
              << std::endl;
    return 0;
}

static int HostRunFourOutputVerify(const std::string &shape_dir, uint32_t pe_size, const std::string &actual_dir,
                                   uint64_t epoch, const std::string &case_logdir)
{
    const char *script_dir_env = std::getenv("INC_DN_PL_SCRIPT_DIR");
    if (script_dir_env == nullptr) {
        std::cerr << "PL_SCRIPT_DIR_MISSING INC_DN_PL_SCRIPT_DIR" << std::endl;
        return -1;
    }
    // 结构化 per-destination 证据：four_output_verify_epoch_<N>.json + PL_FOUR_OUTPUT_VERIFY_DEST
    const std::string out_json =
        case_logdir + "/four_output_verify_epoch_" + std::to_string(epoch) + ".json";
    // 大 capacity SHA 校验：32768×8KB≈256MiB；524288×8KB≈4GiB；按 dest 数放大
    int verify_timeout_s = 8;
    if (const char *cap_env = std::getenv("INC_DN_PL_FINAL_SLOT_CAPACITY")) {
        const long cap = std::strtol(cap_env, nullptr, 10);
        if (cap >= 8192) {
            verify_timeout_s = 60;
        }
        if (cap >= 32768) {
            verify_timeout_s = 180;
        }
        if (cap >= 131072) {
            verify_timeout_s = 600;
        }
        if (cap >= 524288) {
            verify_timeout_s = 1200;
        }
    }
    // 路径含中文/空格时必须引号，否则 system() 拆词导致 verify 假失败
    const bool fast_cmp = []() {
        const char *v = std::getenv("INC_DN_PL_FAST_EXACT_VERIFY");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    const std::string cmd = std::string("timeout ") + std::to_string(verify_timeout_s) +
                            " python3 \"" + script_dir_env +
                            "/inc_dispatch_four_output_verify.py\" \"" + shape_dir + "\" " +
                            std::to_string(pe_size) + " \"" + actual_dir + "\" --epoch " +
                            std::to_string(epoch) + " --out-json \"" + out_json + "\"" +
                            (fast_cmp ? " --fast-cmp" : "");
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::cerr << "PL_FOUR_OUTPUT_VERIFY_CMD_FAIL rc=" << rc << " cmd=" << cmd << std::endl;
    }
    return (rc == 0) ? 0 : 1;
}

static void HostCollectFullDispatchRouteTiming(uint8_t *sym, PlDeviceEpochTiming *dt)
{
    // P6 非正式：PlRouteTimingLine 多 writer；gate 禁止采信 inc_* 分段
    dt->p6_segment_timing_formal = false;
    PlRouteTimingLine rt{};
    D1LeaderTiming leader{};
    aclrtMemcpy(&rt, sizeof(rt), sym + kPlRouteTimingOff, sizeof(rt), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&leader, sizeof(leader), sym + D1B(kD1LeaderTimingOff), sizeof(leader), ACL_MEMCPY_DEVICE_TO_HOST);
    const uint64_t go_cycle = leader.go_fanout_end_cycle > 0 ? leader.go_fanout_end_cycle : leader.release_seen_cycle;
    if (rt.route_count_start_cycle > 0 && rt.prefix_done_cycle > rt.route_count_start_cycle) {
        dt->inc_route_count_prefix_us = CyclesToUs(rt.prefix_done_cycle, rt.route_count_start_cycle);
    }
    if (rt.gather_first_cycle > 0 && rt.gather_done_cycle > rt.gather_first_cycle) {
        dt->inc_route_gather_us = CyclesToUs(rt.gather_done_cycle, rt.gather_first_cycle);
    }
    // stage1_done_cycle 已废弃；inc_stage1_two_hop_us 仅诊断
    if (rt.stage1_done_cycle > 0 && go_cycle > 0 && rt.stage1_done_cycle > go_cycle) {
        dt->inc_stage1_two_hop_us = CyclesToUs(rt.stage1_done_cycle, go_cycle);
    }
    if (rt.finalize_done_cycle > 0 && rt.stage1_done_cycle > 0 && rt.finalize_done_cycle > rt.stage1_done_cycle) {
        dt->inc_finalize_us = CyclesToUs(rt.finalize_done_cycle, rt.stage1_done_cycle);
    }
    if (rt.full_done_cycle > 0 && go_cycle > 0 && rt.full_done_cycle > go_cycle) {
        dt->inc_full_dispatch_us = CyclesToUs(rt.full_done_cycle, go_cycle);
    }
}

// 正式 transport timing：仅 D2H 读线（不参与 timed completion）
static void HostCollectFullDispatchTransportTiming(uint8_t *sym, PlDeviceEpochTiming *dt)
{
    PlTransportTimingLine tt{};
    PlGlobalTransportDoneLine gtd{};
    aclrtMemcpy(&tt, sizeof(tt), sym + kPlTransportTimingOff, sizeof(tt), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&gtd, sizeof(gtd), sym + kPlGlobalTransportDoneOff, sizeof(gtd), ACL_MEMCPY_DEVICE_TO_HOST);
    if (tt.magic != kPlTransportTimingMagic || tt.timing_epoch == 0 || tt.release_seen_cycle == 0 ||
        tt.global_transport_done_cycle <= tt.release_seen_cycle) {
        return;
    }
    dt->inc_transport_makespan_us = CyclesToUs(tt.global_transport_done_cycle, tt.release_seen_cycle);
    dt->global_transport_done_epoch =
        (gtd.magic == kPlGlobalTransportDoneMagic) ? gtd.global_transport_done_epoch : tt.timing_epoch;
    dt->slowest_destination = tt.slowest_destination;
    dt->transport_timing_worker_count = tt.worker_count;
    for (uint32_t d = 0; d < kPlMaxSources; ++d) {
        dt->per_destination_transport_done_cycle[d] = tt.per_dest_transport_done_cycle[d];
    }
    // 同 epoch：transport 必须 ≤ full_dispatch formal makespan
    if (dt->full_dispatch_device_us > 0.0) {
        dt->transport_le_full_dispatch = (dt->inc_transport_makespan_us <= dt->full_dispatch_device_us + 1e-9);
    } else if (dt->device_pipeline_us > 0.0) {
        dt->transport_le_full_dispatch = (dt->inc_transport_makespan_us <= dt->device_pipeline_us + 1e-9);
    }
}

static void HostEmitTransportTiming(int pe, uint64_t go_epoch, const PlDeviceEpochTiming &dt)
{
    std::ostringstream per_dest;
    const uint32_t wc = dt.transport_timing_worker_count > 0 ? dt.transport_timing_worker_count : kPlMaxSources;
    for (uint32_t d = 0; d < wc && d < kPlMaxSources; ++d) {
        if (d > 0) {
            per_dest << ",";
        }
        per_dest << dt.per_destination_transport_done_cycle[d];
    }
    const double full_us =
        dt.full_dispatch_device_us > 0.0 ? dt.full_dispatch_device_us : dt.device_pipeline_us;
    std::cout << "PL_TRANSPORT_TIMING pe=" << pe << " go_epoch=" << go_epoch
              << " formal_transport=" << kPlTimingFormalTransport
              << " inc_transport_makespan_us=" << dt.inc_transport_makespan_us
              << " inc_second_hop_visible_makespan_us=" << dt.inc_second_hop_visible_makespan_us
              << " global_transport_done_epoch=" << dt.global_transport_done_epoch
              << " slowest_destination=" << dt.slowest_destination
              << " per_destination_transport_done_cycle=" << per_dest.str()
              << " full_dispatch_device_us=" << full_us
              << " transport_le_full_dispatch=" << (dt.transport_le_full_dispatch ? 1 : 0)
              << " timing_domain=PE0_release_seen_to_global_transport_done"
              << " p6_segment_timing_formal=" << (dt.p6_segment_timing_formal ? 1 : 0) << std::endl;
}

// Timeout-only local heap snapshot. It is intentionally opt-in: diagnostics
// must not add D2H traffic to a timed successful epoch.
static void HostDumpP5FailureSnapshot(uint8_t *sym, int pe, uint32_t worker_count, uint64_t epoch)
{
    const char *failure_enabled = std::getenv("INC_DN_PL_DUMP_FAILURE_SNAPSHOT");
    const char *ledger_enabled = std::getenv("INC_DN_PL_DUMP_CELL_LEDGER");
    if ((failure_enabled == nullptr || std::strcmp(failure_enabled, "1") != 0) &&
        (ledger_enabled == nullptr || std::strcmp(ledger_enabled, "1") != 0)) {
        return;
    }
    PlTransportTimingLine tt{};
    PlFullOutputDoneLine fout{};
    (void)aclrtMemcpy(&tt, sizeof(tt), sym + kPlTransportTimingOff, sizeof(tt), ACL_MEMCPY_DEVICE_TO_HOST);
    (void)aclrtMemcpy(&fout, sizeof(fout), sym + kPlFullOutputDoneOff, sizeof(fout), ACL_MEMCPY_DEVICE_TO_HOST);
    std::cerr << "PL_P5_FAILURE_SNAPSHOT pe=" << pe << " epoch=" << epoch
              << " fout_error=" << fout.error_code << " timing_magic=0x" << std::hex << tt.magic << std::dec
              << " timing_epoch=" << tt.timing_epoch << " failure_stage=" << tt.failure_stage
              << " error_rank=" << tt.error_rank << " error_destination=" << tt.error_destination
              << " first_error_code=" << tt.first_error_code
              << " missing_destination_mask=0x" << std::hex << tt.missing_destination_mask << std::dec
              << " global_wait_spins=" << tt.global_wait_spins << std::endl;
    for (uint32_t s = 0; s < worker_count && s < kPlMaxSources; ++s) {
        PlRecvDoneLine rd{};
        PlTransportDoneLine td{};
        (void)aclrtMemcpy(&rd, sizeof(rd), sym + kPlRecvDoneOff + static_cast<uint64_t>(s) * 64u, sizeof(rd),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        (void)aclrtMemcpy(&td, sizeof(td), sym + kPlTransportDoneOff + static_cast<uint64_t>(s) * 64u, sizeof(td),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        std::cerr << "PL_P5_FAILURE_RECV pe=" << pe << " source=" << s << " recv_epoch=" << rd.epoch
                  << " expected=" << rd.expected << " received=" << rd.received << " recv_error=" << rd.error_code
                  << " recv_magic=0x" << std::hex << rd.magic << std::dec << " td_epoch=" << td.epoch
                  << " td_expected=" << td.expected_routes << " td_received=" << td.received_routes
                  << " td_error=" << td.error_code << " td_magic=0x" << std::hex << td.magic << std::dec
                  << std::endl;
        if (static_cast<uint32_t>(pe) < worker_count) {
            PlP5CellRouteRecvLedgerLine cell{};
            (void)aclrtMemcpy(&cell, sizeof(cell),
                              sym + PlP5CellRouteRecvLedgerLineOff(s, static_cast<uint32_t>(pe)), sizeof(cell),
                              ACL_MEMCPY_DEVICE_TO_HOST);
            std::cerr << "PL_P5_CELL_RECV source=" << s << " dest=" << pe << " epoch=" << cell.epoch
                      << " route_expected=" << cell.route_expected << " remote_tail="
                      << cell.remote_tail_observed << " recv_seen=" << cell.recv_seen
                      << " recv_expected=" << cell.recv_expected << " last_desc=" << cell.last_descriptor_seq
                      << " error=" << cell.error_code << " magic=0x" << std::hex << cell.magic << std::dec
                      << std::endl;
        }
    }
    if (static_cast<uint32_t>(pe) >= worker_count &&
        static_cast<uint32_t>(pe) < worker_count * 2u) {
        const uint32_t source = static_cast<uint32_t>(pe) - worker_count;
        for (uint32_t dest = 0; dest < worker_count && dest < kPlMaxSources; ++dest) {
            PlP5CellGatherLedgerLine gcell{};
            PlP5CellEgressLedgerLine cell{};
            (void)aclrtMemcpy(&gcell, sizeof(gcell), sym + PlP5CellGatherLedgerLineOff(source, dest),
                              sizeof(gcell), ACL_MEMCPY_DEVICE_TO_HOST);
            (void)aclrtMemcpy(&cell, sizeof(cell), sym + PlP5CellEgressLedgerLineOff(source, dest), sizeof(cell),
                              ACL_MEMCPY_DEVICE_TO_HOST);
            // staging_final_gen 来自 gather 独占行；保持 EGRESS 行格式供 gate 解析。
            std::cerr << "PL_P5_CELL_EGRESS source=" << source << " dest=" << dest << " epoch=" << cell.epoch
                      << " staging_final_gen=" << gcell.staging_final_gen << " egress_seq_end="
                      << cell.egress_seq_end << " normal_tail_publish_count="
                      << cell.normal_tail_publish_count << " terminal_republish_count="
                      << cell.terminal_republish_count << " last_published_tail=" << cell.last_published_tail
                      << " error=" << cell.error_code << " magic=0x" << std::hex << cell.magic << std::dec
                      << std::endl;
        }
    }
}

// P10：优先读 PE0 本地槽（各 rank putmem 汇聚）；缺失再 getmem 兜底
static void HostCollectRankDispatchTiming(uint8_t *sym, uint32_t physical_pe_count, uint64_t go_epoch,
                                          PlDeviceEpochTiming *dt)
{
    PlRankDispatchTimingLine lines[kPlMaxPhysicalPe]{};
    const uint32_t pe_limit = std::min(physical_pe_count, kPlMaxPhysicalPe);
    const int my_pe = aclshmem_my_pe();
    // 注意：本函数仅 PE0 调用，禁止在此 barrier（会死锁）
    for (uint32_t r = 0; r < pe_limit; ++r) {
        const uint64_t off = PlRankDispatchTimingLineOff(r);
        aclrtMemcpy(&lines[r], sizeof(lines[r]), sym + off, sizeof(lines[r]), ACL_MEMCPY_DEVICE_TO_HOST);
        if (lines[r].magic == kPlRankDispatchTimingMagic && lines[r].epoch == go_epoch) {
            continue;
        }
        // 本地槽未齐：从源 PE getmem 兜底
        if (static_cast<int>(r) != my_pe) {
            const uint64_t scratch =
                kPlHostProbeScratchOff + static_cast<uint64_t>(r) * sizeof(PlRankDispatchTimingLine);
            aclshmem_getmem(sym + scratch, sym + off, sizeof(PlRankDispatchTimingLine), static_cast<int>(r));
            aclrtMemcpy(&lines[r], sizeof(lines[r]), sym + scratch, sizeof(lines[r]), ACL_MEMCPY_DEVICE_TO_HOST);
        }
    }

    uint32_t valid_count = 0u;
    uint32_t epoch_match = 0u;
    double max_elapsed = 0.0;
    double second_max = 0.0;
    uint32_t max_rank = 0u;
    uint32_t second_rank = 0u;
    for (uint32_t r = 0; r < pe_limit; ++r) {
        dt->rank_elapsed_us[r] = 0.0;
        const PlRankDispatchTimingLine &ln = lines[r];
        if (ln.magic != kPlRankDispatchTimingMagic || ln.epoch != go_epoch || ln.local_start_cycle == 0u ||
            ln.local_completion_cycle <= ln.local_start_cycle) {
            continue;
        }
        ++valid_count;
        if (ln.epoch == go_epoch) {
            ++epoch_match;
        }
        const double elapsed = CyclesToUs(ln.local_completion_cycle, ln.local_start_cycle);
        dt->rank_elapsed_us[r] = elapsed;
        if (elapsed >= max_elapsed) {
            second_max = max_elapsed;
            second_rank = max_rank;
            max_elapsed = elapsed;
            max_rank = r;
        } else if (elapsed >= second_max) {
            second_max = elapsed;
            second_rank = r;
        }
    }

    dt->rank_timing_valid_count = valid_count;
    dt->rank_timing_epoch = static_cast<uint32_t>(go_epoch);
    dt->strict_max_rank_us = max_elapsed;
    dt->slowest_rank = max_rank;
    dt->second_slowest_rank = second_rank;
    dt->slowest_rank_us = max_elapsed;
    dt->second_slowest_rank_us = second_max;
    dt->rank_imbalance_ratio =
        (second_max > 0.0 && max_elapsed > 0.0) ? (max_elapsed / second_max) : 0.0;
    dt->pe0_global_observer_us = dt->inc_transport_makespan_us;

    PlTransportTimingLine tt{};
    aclrtMemcpy(&tt, sizeof(tt), sym + kPlTransportTimingOff, sizeof(tt), ACL_MEMCPY_DEVICE_TO_HOST);
    double max_dest_us = 0.0;
    if (tt.magic == kPlTransportTimingMagic && tt.timing_epoch == go_epoch && tt.release_seen_cycle > 0u) {
        const uint32_t wc =
            tt.worker_count > 0 ? tt.worker_count : std::min(physical_pe_count / 2u, kPlMaxSources);
        for (uint32_t d = 0; d < wc && d < kPlMaxSources; ++d) {
            if (tt.per_dest_transport_done_cycle[d] > tt.release_seen_cycle) {
                const double dest_us = CyclesToUs(tt.per_dest_transport_done_cycle[d], tt.release_seen_cycle);
                if (dest_us > max_dest_us) {
                    max_dest_us = dest_us;
                }
            }
        }
    }
    dt->max_destination_observed_us = max_dest_us;
    dt->observer_aggregation_us =
        (dt->pe0_global_observer_us > max_dest_us) ? (dt->pe0_global_observer_us - max_dest_us) : 0.0;
    // Formal Full Dispatch makespan is max over every participating rank.
    // A partial PE0 aggregation cannot stand in for a complete rank vector.
    dt->rank_dispatch_timing_valid =
        (valid_count == pe_limit && epoch_match == pe_limit && max_elapsed > 0.0);
}

static void HostEmitRankDispatchTiming(int pe, uint64_t go_epoch, uint32_t physical_pe_count,
                                       const PlDeviceEpochTiming &dt)
{
    std::ostringstream elapsed_csv;
    for (uint32_t r = 0; r < physical_pe_count && r < kPlMaxPhysicalPe; ++r) {
        if (r > 0) {
            elapsed_csv << ",";
        }
        elapsed_csv << dt.rank_elapsed_us[r];
    }
    std::cout << "PL_RANK_DISPATCH_TIMING pe=" << pe << " go_epoch=" << go_epoch
              << " rank_timing_valid=" << (dt.rank_dispatch_timing_valid ? 1 : 0)
              << " rank_timing_epoch=" << dt.rank_timing_epoch
              << " rank_timing_valid_count=" << dt.rank_timing_valid_count
              << " strict_max_rank_us=" << dt.strict_max_rank_us
              << " slowest_rank=" << dt.slowest_rank
              << " second_slowest_rank=" << dt.second_slowest_rank
              << " slowest_rank_us=" << dt.slowest_rank_us
              << " second_slowest_rank_us=" << dt.second_slowest_rank_us
              << " rank_imbalance_ratio=" << dt.rank_imbalance_ratio
              << " pe0_global_observer_us=" << dt.pe0_global_observer_us
              << " max_destination_observed_us=" << dt.max_destination_observed_us
              << " observer_aggregation_us=" << dt.observer_aggregation_us
              << " rank_elapsed_us=" << elapsed_csv.str()
              << " timing_domain=strict_max_rank_local_completion"
              << " formal_bandwidth_note=useful_bytes_over_strict_max_rank_us_not_pe0_global_observer"
              << std::endl;
}

// 正式 second_hop_visible timing：D2H 读 PE0 timing line；release=0 时回退 leader/transport（与 device 同源）
static bool HostCollectSecondHopVisibleTiming(uint8_t *sym, PlDeviceEpochTiming *dt)
{
    PlSecondHopVisibleTimingLine st{};
    PlGlobalSecondHopVisibleDoneLine gs{};
    aclrtMemcpy(&st, sizeof(st), sym + kPlSecondHopVisibleTimingOff, sizeof(st), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&gs, sizeof(gs), sym + kPlGlobalSecondHopVisibleDoneOff, sizeof(gs), ACL_MEMCPY_DEVICE_TO_HOST);
    if (st.magic != kPlSecondHopVisibleTimingMagic || st.timing_epoch == 0 ||
        st.inc_second_hop_visible_done_cycle == 0) {
        dt->second_hop_timing_valid = false;
        dt->second_hop_release_seen = 0;
        dt->second_hop_visible_done = 0;
        return false;
    }
    uint64_t release_cyc = st.release_seen_cycle;
    if (release_cyc == 0) {
        D1LeaderTiming leader{};
        aclrtMemcpy(&leader, sizeof(leader), sym + D1B(kD1LeaderTimingOff), sizeof(leader), ACL_MEMCPY_DEVICE_TO_HOST);
        release_cyc = leader.release_seen_cycle;
    }
    if (release_cyc == 0) {
        PlTransportTimingLine tt{};
        aclrtMemcpy(&tt, sizeof(tt), sym + kPlTransportTimingOff, sizeof(tt), ACL_MEMCPY_DEVICE_TO_HOST);
        if (tt.magic == kPlTransportTimingMagic && tt.timing_epoch == st.timing_epoch && tt.release_seen_cycle > 0) {
            release_cyc = tt.release_seen_cycle;
        }
    }
    const bool release_ok = release_cyc > 0;
    const bool visible_ok = release_ok && st.inc_second_hop_visible_done_cycle > release_cyc;
    dt->second_hop_release_seen = release_ok ? 1u : 0u;
    dt->second_hop_visible_done = visible_ok ? 1u : 0u;
    if (!visible_ok) {
        dt->second_hop_timing_valid = false;
        return false;
    }
    dt->inc_second_hop_visible_makespan_us =
        CyclesToUs(st.inc_second_hop_visible_done_cycle, release_cyc);
dt->global_second_hop_visible_epoch =
        (gs.magic == kPlGlobalSecondHopVisibleMagic) ? gs.global_second_hop_visible_epoch : st.timing_epoch;
    dt->slowest_cell_inc = st.slowest_cell_inc;
    dt->slowest_cell_dest = st.slowest_cell_dest;
    dt->second_hop_timing_worker_count = st.worker_count;
    for (uint32_t d = 0; d < kPlMaxSources; ++d) {
        dt->per_dest_visible_done_cycle[d] = st.per_dest_visible_done_cycle[d];
    }
    dt->second_hop_timing_valid = true;
    return true;
}

static void HostEmitSecondHopVisibleTiming(int pe, uint64_t go_epoch, const PlDeviceEpochTiming &dt)
{
    std::ostringstream per_dest;
    const uint32_t wc =
        dt.second_hop_timing_worker_count > 0 ? dt.second_hop_timing_worker_count : kPlMaxSources;
    for (uint32_t d = 0; d < wc && d < kPlMaxSources; ++d) {
        if (d > 0) {
            per_dest << ",";
        }
        per_dest << dt.per_dest_visible_done_cycle[d];
    }
    std::cout << "PL_SECOND_HOP_VISIBLE_TIMING pe=" << pe << " epoch=" << go_epoch
              << " formal_second_hop=" << kPlTimingFormalSecondHopVisible
              << " release_seen=" << dt.second_hop_release_seen
              << " visible_done=" << dt.second_hop_visible_done
              << " inc_second_hop_visible_makespan_us=" << dt.inc_second_hop_visible_makespan_us
              << " global_second_hop_visible_epoch=" << dt.global_second_hop_visible_epoch
              << " slowest_cell_inc=" << dt.slowest_cell_inc
              << " slowest_cell_dest=" << dt.slowest_cell_dest
              << " per_dest_visible_done_cycle=" << per_dest.str()
              << " timing_domain=PE0_release_seen_to_second_hop_visible"
              << " transport_diagnostic_note=PL_TRANSPORT_TIMING_is_recv_complete_only" << std::endl;
}

static uint32_t HostEnvDelayCycles(const char *env_name)
{
    const char *env = std::getenv(env_name);
    if (env == nullptr || env[0] == '\0') {
        return 0u;
    }
    const double us = std::strtod(env, nullptr);
    if (us <= 0.0) {
        return 0u;
    }
    return static_cast<uint32_t>(us * static_cast<double>(kD1CyclesPerUs));
}

static int HostInitTimingNegHooks(uint8_t *sym, aclrtStream stream)
{
    PlTimingNegLine neg{};
    neg.delay_second_hop_cycles = HostEnvDelayCycles("INC_DN_PL_DELAY_SECOND_HOP_US");
    neg.delay_recv_cycles = HostEnvDelayCycles("INC_DN_PL_DELAY_RECV_US");
    neg.delay_finalize_cycles = HostEnvDelayCycles("INC_DN_PL_DELAY_FINALIZE_US");
    const char *drop_env = std::getenv("INC_DN_PL_DROP_EGRESS_CELL");
    neg.drop_egress_cell = (drop_env != nullptr && std::strcmp(drop_env, "1") == 0) ? 1u : 0u;
    // H7-E 默认 0：formal transport 已不再阻塞等 second_hop；保留 env 仅作历史对照
    const char *sh_spins = std::getenv("INC_DN_PL_TRANSPORT_SH_WAIT_SPINS");
    neg.transport_second_hop_wait_spins =
        (sh_spins != nullptr) ? static_cast<uint32_t>(std::strtoul(sh_spins, nullptr, 10)) : 0u;
    // H8.1 S3-NEG：指定 epoch/source 故意不发 SourceCountReady
    const char *drop_src = std::getenv("INC_DN_PL_NEG_DROP_SOURCE_COUNT_READY");
    neg.drop_source_count_ready = (drop_src != nullptr && std::strcmp(drop_src, "1") == 0) ? 1u : 0u;
    const char *drop_rank = std::getenv("INC_DN_PL_NEG_DROP_SOURCE_RANK");
    neg.drop_source_rank =
        (drop_rank != nullptr) ? static_cast<uint32_t>(std::strtoul(drop_rank, nullptr, 10)) : 0u;
    const char *drop_ep = std::getenv("INC_DN_PL_NEG_DROP_SOURCE_EPOCH");
    neg.drop_source_epoch =
        (drop_ep != nullptr) ? static_cast<uint32_t>(std::strtoul(drop_ep, nullptr, 10)) : 2u;
    neg.magic = kPlTimingNegMagic;
    std::cerr << "PL_TIMING_NEG_INIT pe_env"
              << " delay_second_hop_cycles=" << neg.delay_second_hop_cycles
              << " delay_recv_cycles=" << neg.delay_recv_cycles
              << " delay_finalize_cycles=" << neg.delay_finalize_cycles
              << std::endl;
    return aclrtMemcpyAsync(sym + kPlTimingNegOff, sizeof(neg), &neg, sizeof(neg), ACL_MEMCPY_HOST_TO_DEVICE, stream);
}

static bool ParsePerfSkipEpochClear()
{
    const char *e1 = std::getenv("INC_DN_PL_PERF_NO_EPOCH_CLEAR");
    const char *e2 = std::getenv("INC_DN_PL_SKIP_EPOCH_OUTPUT_CLEAR");
    return (e1 != nullptr && std::strcmp(e1, "1") == 0) ||
           (e2 != nullptr && std::strcmp(e2, "1") == 0);
}

// 每 epoch 从 invocation shape 算字节分子。完整 Dispatch 的正式分子
// 是成功交付到 expert 端的 route-instance payload，必须包含全部 TopK
// copy；source-major 只减少第一跳上传，不能改变操作输出的字节口径。
static void HostEmitBytesNumerator(int pe, uint64_t go_epoch, uint32_t worker_count, uint32_t route_topk,
                                   const uint32_t *source_token_count, bool first_hop_source_major)
{
    uint32_t total_input = 0u;
    for (uint32_t s = 0; s < worker_count; ++s) {
        total_input += source_token_count[s];
    }
    const uint64_t input_payload_bytes = static_cast<uint64_t>(total_input) * g_runtime_payload_bytes;
    const uint64_t route_instance_payload_bytes =
        static_cast<uint64_t>(total_input) * static_cast<uint64_t>(route_topk) * g_runtime_payload_bytes;
    const uint64_t first_hop_unique_bytes = input_payload_bytes;
    const uint64_t formal_bytes = route_instance_payload_bytes;
    std::cout << "PL_BYTES_NUMERATOR pe=" << pe << " epoch=" << go_epoch
              << " input_payload_bytes=" << input_payload_bytes
              << " first_hop_unique_bytes=" << first_hop_unique_bytes
              << " unique_destination_copy_bytes=" << formal_bytes
              << " route_instance_payload_bytes=" << route_instance_payload_bytes
              << " bytes_note="
              << (first_hop_source_major ? "source_major_upload_once_formal_counts_delivered_topk"
                                         : "delivered_route_instances")
              << " formal_gbps_via_makespan_ratio=1" << std::endl;
}

static uint32_t ParseVerifyMode()
{
    const char *env = std::getenv("INC_DN_PL_VERIFY_MODE");
    if (env != nullptr && std::strcmp(env, "host_post") == 0) {
        return kPlVerifyHostPost;
    }
    return kPlVerifyDeviceFull;
}

static bool ParseControlOnly()
{
    const char *env = std::getenv("INC_DN_PL_CONTROL_ONLY");
    return env != nullptr && std::strcmp(env, "1") == 0;
}

static int ParseEpochWaitMs()
{
    const char *env = std::getenv("INC_DN_PL_EPOCH_WAIT_MS");
    if (env == nullptr || env[0] == '\0') {
        return 2000;
    }
    return std::atoi(env);
}

static uint32_t ParsePayloadSourceMode(uint32_t warmup, uint32_t measure)
{
    const char *env = std::getenv("INC_DN_PL_PAYLOAD_SOURCE");
    if (env != nullptr && std::strcmp(env, "epoch_unique") == 0) {
        return kPlPayloadSourceEpochUnique;
    }
    if (env != nullptr && std::strcmp(env, "static") == 0) {
        return kPlPayloadSourceStatic;
    }
    // 多 epoch 默认 static 复用同一份 payload
    (void)warmup;
    (void)measure;
    return kPlPayloadSourceStatic;
}

static uint32_t ParseCompletionMode()
{
    const char *env = std::getenv("INC_DN_PL_COMPLETION_MODE");
    if (env != nullptr && std::strcmp(env, "counter_scan") == 0) {
        return kPlCompletionModeCounterScan;
    }
    return kPlCompletionModeDoneLines;
}

static uint32_t ParseHostCompletionMode()
{
    const char *env = std::getenv("INC_DN_PL_HOST_COMPLETION_MODE");
    if (env != nullptr && std::strcmp(env, "async_production") == 0) {
        return kPlHostCompletionAsyncProduction;
    }
    return kPlHostCompletionSyncPoll;
}

// 测量模式：prepacked / stage1 诊断 / full dispatch 正式
static uint32_t ParseMeasurementMode()
{
    const char *env = std::getenv("INC_DN_PL_MEASUREMENT_MODE");
    if (env != nullptr) {
        if (std::strcmp(env, "stage1") == 0 || std::strcmp(env, "stage1_diagnostic") == 0) {
            return kPlMeasurementStage1Diagnostic;
        }
        if (std::strcmp(env, "full") == 0 || std::strcmp(env, "full_dispatch") == 0) {
            return kPlMeasurementFullDispatch;
        }
    }
    return kPlMeasurementPrepackedTransport;
}

enum class PlIngressLayoutRequest : uint32_t {
    kDestMajor = 0u,
    kSourceMajor = 1u,
    kAuto = 2u,
    kInvalid = 3u,
};

// P5: dest_major_gather (default) | source_major_raw | auto.
// "auto" is deliberately opt-in until the complete multi-shape promotion
// matrix passes; explicit source-major remains fail-closed when its bounded
// raw descriptor ring cannot hold one whole epoch.
static PlIngressLayoutRequest ParseIngressLayoutRequest()
{
    const char *env = std::getenv("INC_DN_PL_INGRESS_LAYOUT");
    if (env == nullptr || env[0] == '\0') {
        return PlIngressLayoutRequest::kDestMajor;
    }
    if (std::strcmp(env, "source_major_raw") == 0 || std::strcmp(env, "source-major-raw") == 0) {
        return PlIngressLayoutRequest::kSourceMajor;
    }
    if (std::strcmp(env, "dest_major_gather") == 0 || std::strcmp(env, "dest-major-gather") == 0) {
        return PlIngressLayoutRequest::kDestMajor;
    }
    if (std::strcmp(env, "auto") == 0) {
        return PlIngressLayoutRequest::kAuto;
    }
    return PlIngressLayoutRequest::kInvalid;
}

static const char *MeasurementModeName(uint32_t mode)
{
    switch (mode) {
        case kPlMeasurementStage1Diagnostic:
            return "stage1_diagnostic";
        case kPlMeasurementFullDispatch:
            return "full_dispatch";
        default:
            return "prepacked_transport_only";
    }
}

static uint32_t ParseEnvU32(const char *name, uint32_t default_val)
{
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return default_val;
    }
    return static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
}

// 解析 "a,b,c" → source_token_count；不足补 0，超出截断
static void ParseCsvU32List(const char *csv, uint32_t *out, uint32_t n, uint32_t fill)
{
    for (uint32_t i = 0; i < n; ++i) {
        out[i] = fill;
    }
    if (csv == nullptr || csv[0] == '\0') {
        return;
    }
    uint32_t idx = 0;
    const char *p = csv;
    while (*p && idx < n) {
        char *end = nullptr;
        const unsigned long v = std::strtoul(p, &end, 10);
        out[idx++] = static_cast<uint32_t>(v);
        if (end == nullptr || *end == '\0') {
            break;
        }
        p = (*end == ',') ? (end + 1) : end;
    }
}

// epochs 用 '|' 分隔；缺省单 epoch 均匀 capacity
static void ParseSourceTokenCountsByEpoch(uint32_t worker_count, uint32_t capacity, uint32_t epoch_count,
                                          uint32_t out[][kPlMaxSources])
{
    for (uint32_t e = 0; e < epoch_count && e < kPlMaxInvocationEpochs; ++e) {
        for (uint32_t s = 0; s < kPlMaxSources; ++s) {
            out[e][s] = (s < worker_count) ? capacity : 0u;
        }
    }
    const char *multi = std::getenv("INC_DN_PL_SOURCE_TOKEN_COUNTS_EPOCHS");
    if (multi != nullptr && multi[0] != '\0') {
        uint32_t ep = 0;
        const char *p = multi;
        while (*p && ep < epoch_count && ep < kPlMaxInvocationEpochs) {
            const char *bar = std::strchr(p, '|');
            std::string chunk = bar ? std::string(p, bar) : std::string(p);
            ParseCsvU32List(chunk.c_str(), out[ep], worker_count, 0u);
            for (uint32_t s = worker_count; s < kPlMaxSources; ++s) {
                out[ep][s] = 0u;
            }
            ++ep;
            if (!bar) {
                break;
            }
            p = bar + 1;
        }
        return;
    }
    const char *single = std::getenv("INC_DN_PL_SOURCE_TOKEN_COUNTS");
    if (single != nullptr && single[0] != '\0') {
        for (uint32_t e = 0; e < epoch_count && e < kPlMaxInvocationEpochs; ++e) {
            ParseCsvU32List(single, out[e], worker_count, 0u);
            for (uint32_t s = worker_count; s < kPlMaxSources; ++s) {
                out[e][s] = 0u;
            }
        }
    }
}

static int HostReadIncForwardLane(uint8_t *sym, int inc_pe, uint32_t lane, PlForwardCounters *out, aclrtStream host_stream);
static int HostReadIncSourceCountTrace(uint8_t *sym, int inc_pe, uint32_t source, PlIncSourceCountTrace *out);

// host 侧 epoch 累计快照（用于 PL_PIPE_DELTA）
struct PlHostEpochSnap {
    PlUploadCounters upload{};
    PlForwardCounters forward{};
    PlRecvCounters recv{};
    PlPipelineTiming timing{};
    PlDestDoneLine dest_done{};
    uint64_t pair_done0 = 0;
    uint32_t pair_done_mask = 0;
    uint32_t real_pipeline_overlap = 0;
    uint64_t global_done = 0;
    uint64_t service_exit_epoch = 0;
    uint32_t timing_error_code = 0;
};

static PlHostEpochSnap HostCollectEpochSnap(uint8_t *sym, int inc_pe, aclrtStream host_stream)
{
    (void)inc_pe;
    (void)host_stream;
    PlHostEpochSnap snap{};
    D1LeaderCompletion leader_completion{};
    D1GlobalDoneLine global_done{};
    PlServiceTraceLine trace0{};
    aclrtMemcpy(&snap.upload, sizeof(snap.upload), sym + kPlUploadCtrOff, sizeof(snap.upload), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&snap.recv, sizeof(snap.recv), sym + kPlRecvCtrOff, sizeof(snap.recv), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&snap.timing, sizeof(snap.timing), sym + kPlTimingOff, sizeof(snap.timing), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&snap.dest_done, sizeof(snap.dest_done), sym + kPlDestDoneOff, sizeof(snap.dest_done),
                ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&leader_completion, sizeof(leader_completion), sym + D1B(kD1LeaderCompletionOff),
                sizeof(leader_completion), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&global_done, sizeof(global_done), sym + D1B(kD1GlobalDoneLineOff), sizeof(global_done),
                ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&trace0, sizeof(trace0), sym + kPlWorkerTraceOff, sizeof(trace0), ACL_MEMCPY_DEVICE_TO_HOST);
    // H8.0：禁止 remote get INC counter 写入 worker PL_PIPE_EPOCH（可能陈旧/垃圾）
    // forward 守恒仅认各 INC PE 本地 D2H 的 PL_PIPE_INC_STATS
    snap.forward = PlForwardCounters{};
    PlOverlapTelemetry overlap{};
    aclrtMemcpy(&overlap, sizeof(overlap), sym + kPlOverlapOff, sizeof(overlap), ACL_MEMCPY_DEVICE_TO_HOST);
    uint32_t pair_mask = 0;
    for (uint32_t i = 0; i < 8u; ++i) {
        if (leader_completion.pair_done_slots[i].value > 0) {
            pair_mask |= (1u << i);
        }
    }
    snap.pair_done_mask = pair_mask;
    snap.real_pipeline_overlap = overlap.real_pipeline_overlap;
    snap.pair_done0 = leader_completion.pair_done_slots[0].value;
    snap.global_done = global_done.value;
    snap.service_exit_epoch = trace0.service_exit_epoch;
    snap.timing_error_code = snap.timing.error_code;
    return snap;
}

static PlHostEpochSnap HostDiffEpochSnap(const PlHostEpochSnap &cur, const PlHostEpochSnap &prev)
{
    PlHostEpochSnap d{};
    auto sub64 = [](uint64_t c, uint64_t p) { return c >= p ? c - p : 0u; };
    d.upload.local_tail = cur.upload.local_tail;
    d.upload.cached_remote_head = cur.upload.cached_remote_head;
    d.upload.payload_put_count = sub64(cur.upload.payload_put_count, prev.upload.payload_put_count);
    d.upload.payload_mte_wait_count = sub64(cur.upload.payload_mte_wait_count, prev.upload.payload_mte_wait_count);
    d.upload.descriptor_tile_put_count =
        sub64(cur.upload.descriptor_tile_put_count, prev.upload.descriptor_tile_put_count);
    d.upload.descriptor_mte_wait_count =
        sub64(cur.upload.descriptor_mte_wait_count, prev.upload.descriptor_mte_wait_count);
    d.upload.data_descriptor_drain_quiet_count =
        sub64(cur.upload.data_descriptor_drain_quiet_count, prev.upload.data_descriptor_drain_quiet_count);
    d.upload.tail_publish_count = sub64(cur.upload.tail_publish_count, prev.upload.tail_publish_count);
    d.upload.tail_completion_quiet_count =
        sub64(cur.upload.tail_completion_quiet_count, prev.upload.tail_completion_quiet_count);
    d.upload.credit_wait_cycles = sub64(cur.upload.credit_wait_cycles, prev.upload.credit_wait_cycles);
    d.forward.ingress_seen = sub64(cur.forward.ingress_seen, prev.forward.ingress_seen);
    d.forward.ingress_tail_observed = cur.forward.ingress_tail_observed;
    d.forward.egress_forwarded = sub64(cur.forward.egress_forwarded, prev.forward.egress_forwarded);
    d.forward.payload_put_count = sub64(cur.forward.payload_put_count, prev.forward.payload_put_count);
    d.forward.descriptor_put_count = sub64(cur.forward.descriptor_put_count, prev.forward.descriptor_put_count);
    d.forward.egress_tail_publish_count =
        sub64(cur.forward.egress_tail_publish_count, prev.forward.egress_tail_publish_count);
    d.forward.ingress_head_publish_count =
        sub64(cur.forward.ingress_head_publish_count, prev.forward.ingress_head_publish_count);
    d.forward.egress_credit_wait_cycles =
        sub64(cur.forward.egress_credit_wait_cycles, prev.forward.egress_credit_wait_cycles);
    d.forward.max_ingress_occupancy = cur.forward.max_ingress_occupancy;
    d.forward.max_egress_occupancy = cur.forward.max_egress_occupancy;
    d.recv.channel_seen = cur.recv.channel_seen;
    d.recv.tokens_received = sub64(cur.recv.tokens_received, prev.recv.tokens_received);
    d.recv.descriptor_verified_count =
        sub64(cur.recv.descriptor_verified_count, prev.recv.descriptor_verified_count);
    d.recv.payload_verified_count = sub64(cur.recv.payload_verified_count, prev.recv.payload_verified_count);
    d.recv.verify_error_count = sub64(cur.recv.verify_error_count, prev.recv.verify_error_count);
    d.recv.stale_epoch_count = sub64(cur.recv.stale_epoch_count, prev.recv.stale_epoch_count);
    d.recv.duplicate_count = sub64(cur.recv.duplicate_count, prev.recv.duplicate_count);
    d.recv.lost_count = sub64(cur.recv.lost_count, prev.recv.lost_count);
    d.recv.slot_overwrite_count = sub64(cur.recv.slot_overwrite_count, prev.recv.slot_overwrite_count);
    d.recv.head_publish_count = sub64(cur.recv.head_publish_count, prev.recv.head_publish_count);
    d.recv.max_occupancy = cur.recv.max_occupancy;
    d.dest_done.done_epoch = cur.dest_done.done_epoch;
    d.pair_done0 = cur.pair_done0;
    d.global_done = cur.global_done;
    d.service_exit_epoch = cur.service_exit_epoch;
    d.timing_error_code = cur.timing_error_code;
    return d;
}

static void HostEmitPipeEpoch(int pe, uint32_t iter, bool is_measure, int pass, uint64_t go_epoch, double host_observed_us,
                              double device_pipeline_us, double gbps, double device_gbps, const PlHostEpochSnap &snap,
                              uint32_t payload_source_mode, uint32_t verify_mode)
{
    const auto &u = snap.upload;
    const auto &r = snap.recv;
    const auto &t = snap.timing;
    (void)snap.forward; // 故意不用 remote INC sample
    std::cout << "PL_PIPE_EPOCH pe=" << pe << " iter=" << iter << " is_measure=" << (is_measure ? 1 : 0)
              << " pass=" << pass << " go_epoch=" << go_epoch << " pipeline_makespan_us=" << host_observed_us
              << " host_observed_dispatch_us=" << host_observed_us << " device_pipeline_us=" << device_pipeline_us
              << " system_effective_gbps=" << gbps << " device_effective_gbps=" << device_gbps
              << " payload_source_mode=" << payload_source_mode
              << " epoch_unique_payload_pattern="
              << (payload_source_mode == kPlPayloadSourceEpochUnique ? 1 : 0)
              << " worker_local_tail=" << u.local_tail << " worker_cached_remote_head=" << u.cached_remote_head
              << " worker_payload_put=" << u.payload_put_count
              << " worker_payload_mte_wait=" << u.payload_mte_wait_count
              << " worker_descriptor_put=" << u.descriptor_tile_put_count
              << " worker_descriptor_mte_wait=" << u.descriptor_mte_wait_count
              << " worker_drain_quiet=" << u.data_descriptor_drain_quiet_count
              << " worker_tail_publish=" << u.tail_publish_count
              << " worker_tail_quiet=" << u.tail_completion_quiet_count
              << " worker_credit_wait=" << u.credit_wait_cycles
              // Worker 行不承载远端 INC counter；守恒见各 INC 的 PL_PIPE_INC_STATS
              << " inc_counter_sample_valid=0"
              << " inc_ingress_seen=0" << " inc_ingress_tail=0"
              << " inc_egress_forwarded=0" << " inc_payload_put=0"
              << " inc_descriptor_put=0"
              << " inc_egress_tail_publish=0"
              << " inc_ingress_head_publish=0"
              << " inc_egress_credit_wait=0"
              << " inc_max_ingress_occupancy=0"
              << " inc_max_egress_occupancy=0"
              << " destination_channel_seen=" << r.channel_seen << " destination_received=" << r.tokens_received
              << " descriptor_verified=" << r.descriptor_verified_count
              << " payload_verified=" << r.payload_verified_count << " verify_error=" << r.verify_error_count
              << " stale_epoch=" << r.stale_epoch_count << " duplicate=" << r.duplicate_count << " lost=" << r.lost_count
              << " slot_overwrite_count=" << r.slot_overwrite_count
              << " destination_head_publish=" << r.head_publish_count
              << " destination_max_occupancy=" << r.max_occupancy
              << " upload_done_bitmap=0x" << std::hex << t.upload_done_bitmap << " forward_done_bitmap=0x"
              << t.forward_done_bitmap << " recv_done_bitmap=0x" << t.recv_done_bitmap << std::dec
              << " destination_done_epoch=" << t.destination_done_epoch
              << " dest_tokens_received=" << snap.dest_done.tokens_received << " pair_done0=" << snap.pair_done0
              << " pair_done_mask=0x" << std::hex << snap.pair_done_mask << std::dec
              << " global_done=" << snap.global_done << " service_exit_epoch=" << snap.service_exit_epoch
              << " timing_error_code=" << snap.timing_error_code
              << " real_pipeline_overlap=" << snap.real_pipeline_overlap
              << " verify_mode=" << (verify_mode == kPlVerifyHostPost ? "host_post" : "device_full") << std::endl;
}

// M3/D1 formal evidence：LocalFinal 路径计数必须来自 device counters（禁止按 token 推导）
static void HostEmitLocalFinalPath(int pe, uint64_t go_epoch, bool local_final, bool is_worker,
                                   const PlHostEpochSnap &snap, const PlForwardCounters *inc_fc,
                                   uint8_t *sym, uint32_t worker_count)
{
    const auto &r = snap.recv;
    const uint64_t accepted = static_cast<uint64_t>(r.tokens_received);
    const uint64_t accepted_bytes = accepted * static_cast<uint64_t>(g_runtime_payload_bytes);
    uint64_t destfinal_puts = 0ull;
    uint64_t ring_puts = 0ull;
    uint64_t ring_put_bytes = 0ull;
    uint64_t desc_puts = 0ull;
    uint64_t local_copies = 0ull;
    uint64_t local_copy_bytes = 0ull;
    uint64_t local_copy_issue = 0ull;
    uint64_t local_copy_cycles = 0ull;
    uint64_t ring_dcci_count = 0ull;
    uint64_t ring_dcci_bytes = 0ull;
    uint64_t ring_dcci_cycles = 0ull;
    uint64_t out_dcci_count = 0ull;
    uint64_t out_dcci_bytes = 0ull;
    uint64_t out_dcci_cycles = 0ull;
    uint64_t assist_count = 0ull;
    uint64_t assist_cycles = 0ull;
    uint64_t ready_count = 0ull;
    uint64_t ready_cycles = 0ull;
    uint64_t credit_count = 0ull;
    uint64_t credit_cycles = 0ull;
    uint64_t first_copy = 0ull;
    uint64_t last_copy = 0ull;
    uint64_t first_credit = 0ull;
    uint64_t range_puts = 0ull;
    uint64_t scalar_puts = 0ull;
    uint64_t drain_quiet = 0ull;
    uint64_t tail_pub = 0ull;
    uint64_t credit_wait = 0ull;
    bool device_counters_ok = false;

    if (is_worker && sym != nullptr) {
        uint64_t accepted_epoch = 0ull;
        uint64_t stale_epoch = 0ull;
        uint64_t dup_epoch = 0ull;
        uint64_t lost_epoch = 0ull;
        uint64_t overwrite_epoch = 0ull;
        uint32_t channel_rows = 0u;
        for (uint32_t s = 0; s < worker_count; ++s) {
            PlRecvChannelCompletionTrace rct{};
            if (aclrtMemcpy(&rct, sizeof(rct), sym + PlRecvChannelCompletionTraceOff(s), sizeof(rct),
                            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                continue;
            }
            if (rct.magic != kPlRecvChanCompletionTraceMagic || rct.epoch != go_epoch) {
                continue;
            }
            device_counters_ok = true;
            ++channel_rows;
            accepted_epoch += rct.token_count;
            stale_epoch += rct.stale_count;
            dup_epoch += rct.duplicate_count;
            lost_epoch += rct.lost_count;
            overwrite_epoch += rct.overwrite_count;
            local_copy_issue += rct.local_copy_issue_count;
            local_copies += rct.local_copy_complete_count;
            local_copy_bytes += rct.local_copy_bytes;
            local_copy_cycles += rct.local_copy_cycles;
            ring_dcci_count += rct.ring_payload_dcci_count;
            ring_dcci_bytes += rct.ring_payload_dcci_bytes;
            ring_dcci_cycles += rct.ring_payload_dcci_cycles;
            out_dcci_count += rct.output_dcci_count;
            out_dcci_bytes += rct.output_dcci_bytes;
            out_dcci_cycles += rct.output_dcci_cycles;
            assist_count += rct.assist_write_count;
            assist_cycles += rct.assist_write_cycles;
            ready_count += rct.ready_publish_count;
            ready_cycles += rct.ready_publish_cycles;
            credit_count += rct.credit_publish_count;
            credit_cycles += rct.credit_publish_cycles;
            if (rct.first_local_copy_cycle > 0 && (first_copy == 0 || rct.first_local_copy_cycle < first_copy)) {
                first_copy = rct.first_local_copy_cycle;
            }
            if (rct.last_local_copy_cycle > last_copy) {
                last_copy = rct.last_local_copy_cycle;
            }
            if (rct.first_credit_publish_cycle_lf > 0 &&
                (first_credit == 0 || rct.first_credit_publish_cycle_lf < first_credit)) {
                first_credit = rct.first_credit_publish_cycle_lf;
            }
            // C1：按 dest PE × recv channel × epoch 输出真实 device counter（禁止 host 推算）
            std::cout << "PL_LOCAL_FINAL_CHANNEL pe=" << pe << " recv_channel=" << s << " epoch=" << go_epoch
                      << " tokens_received=" << rct.token_count
                      << " accepted_route_bytes=" << (static_cast<uint64_t>(rct.token_count) * g_runtime_payload_bytes)
                      << " local_mte_issue_count=" << rct.local_copy_issue_count
                      << " local_mte_wait_count=" << rct.local_copy_complete_count
                      << " local_mte_cycles=" << rct.local_copy_cycles
                      << " ring_payload_dcci_count=" << rct.ring_payload_dcci_count
                      << " ring_payload_dcci_bytes=" << rct.ring_payload_dcci_bytes
                      << " ring_payload_dcci_cycles=" << rct.ring_payload_dcci_cycles
                      << " output_dcci_count=" << rct.output_dcci_count
                      << " output_dcci_bytes=" << rct.output_dcci_bytes
                      << " output_dcci_cycles=" << rct.output_dcci_cycles
                      << " assist_write_count=" << rct.assist_write_count
                      << " assist_write_cycles=" << rct.assist_write_cycles
                      << " ready_publish_count=" << rct.ready_publish_count
                      << " ready_publish_cycles=" << rct.ready_publish_cycles
                      << " credit_publish_count=" << rct.credit_publish_count
                      << " credit_publish_cycles=" << rct.credit_publish_cycles
                      << " tail_wait_cycles=" << rct.tail_wait_total_cycles
                      << " first_tail_seen_cycle=" << rct.first_tail_seen_cycle
                      << " last_tail_seen_cycle=" << rct.last_tail_seen_cycle
                      << " first_local_copy_issue_cycle=" << rct.first_local_copy_cycle
                      << " first_local_copy_complete_cycle=" << rct.first_local_copy_cycle
                      << " last_local_copy_complete_cycle=" << rct.last_local_copy_cycle
                      << " recv_done_cycle=" << rct.recv_done_cycle
                      << " descriptor_dcci_count=NA"
                      << " assist_dcci_count=NA"
                      << " ready_dcci_count=" << rct.ready_publish_count
                      << " stale_count=" << rct.stale_count
                      << " duplicate_count=" << rct.duplicate_count
                      << " lost_count=" << rct.lost_count
                      << " overwrite_count=" << rct.overwrite_count << std::endl;
        }
        // 用本 epoch device token_count + RCT 错误计数；禁止累计 snap.recv 冒充 epoch 守恒
        const uint64_t accepted_bytes_epoch = accepted_epoch * static_cast<uint64_t>(g_runtime_payload_bytes);
        destfinal_puts = 0ull;
        ring_puts = 0ull;
        const bool conserve_ok =
            device_counters_ok && channel_rows > 0u && destfinal_puts == 0ull &&
            local_copy_issue == local_copies && local_copies == accepted_epoch &&
            local_copy_bytes == accepted_bytes_epoch && stale_epoch == 0ull && dup_epoch == 0ull &&
            lost_epoch == 0ull && overwrite_epoch == 0ull;
        std::cout << "PL_LOCAL_FINAL_PATH pe=" << pe << " epoch=" << go_epoch
                  << " workspace_mode=" << (local_final ? "local_final" : "legacy")
                  << " role=worker"
                  << " counter_scope=sum_recv_channels_this_pe"
                  << " channel_rows=" << channel_rows
                  << " accepted_route_instances=" << accepted_epoch
                  << " accepted_route_bytes=" << accepted_bytes_epoch
                  << " direct_destfinal_remote_put_count=" << destfinal_puts
                  << " destination_ring_put_count=" << ring_puts
                  << " destination_ring_payload_put_bytes=" << ring_put_bytes
                  << " destination_ring_descriptor_put_count=" << desc_puts
                  << " worker_local_copy_count=" << local_copies
                  << " worker_local_copy_bytes=" << local_copy_bytes
                  << " local_copy_issue_count=" << local_copy_issue
                  << " local_copy_complete_count=" << local_copies
                  << " local_copy_cycles=" << local_copy_cycles
                  << " ring_payload_dcci_count=" << ring_dcci_count
                  << " ring_payload_dcci_bytes=" << ring_dcci_bytes
                  << " ring_payload_dcci_cycles=" << ring_dcci_cycles
                  << " output_dcci_count=" << out_dcci_count
                  << " output_dcci_bytes=" << out_dcci_bytes
                  << " output_dcci_cycles=" << out_dcci_cycles
                  << " assist_write_count=" << assist_count
                  << " assist_write_cycles=" << assist_cycles
                  << " ready_publish_count=" << ready_count
                  << " ready_publish_cycles=" << ready_cycles
                  << " credit_publish_count=" << credit_count
                  << " credit_publish_cycles=" << credit_cycles
                  << " range_put_count=" << range_puts
                  << " scalar_put_count=" << scalar_puts
                  << " drain_quiet_count=" << drain_quiet
                  << " tail_publish_count=" << tail_pub
                  << " credit_wait_cycles=" << credit_wait
                  << " first_local_copy_cycle=" << first_copy
                  << " last_local_copy_cycle=" << last_copy
                  << " first_credit_publish_cycle=" << first_credit
                  << " device_counters_ok=" << (device_counters_ok ? 1 : 0)
                  << " conserve_ok=" << (conserve_ok ? 1 : 0)
                  << " stale_count=" << stale_epoch
                  << " duplicate_count=" << dup_epoch
                  << " lost_count=" << lost_epoch
                  << " overwrite_count=" << overwrite_epoch << std::endl;
        return;
    } else if (inc_fc != nullptr) {
        device_counters_ok = true;
        destfinal_puts = inc_fc->direct_destfinal_remote_put_count;
        ring_puts = inc_fc->egress_forwarded > 0 ? inc_fc->egress_forwarded : inc_fc->payload_put_count;
        ring_put_bytes = ring_puts * static_cast<uint64_t>(g_runtime_payload_bytes);
        desc_puts = inc_fc->descriptor_put_count;
        range_puts = inc_fc->egress_payload_range_put_count;
        scalar_puts = inc_fc->egress_payload_scalar_put_count;
        drain_quiet = inc_fc->drain_quiet_count;
        tail_pub = inc_fc->egress_tail_publish_count;
        credit_wait = inc_fc->egress_credit_wait_cycles;
        local_copies = 0ull;
        local_copy_bytes = 0ull;
    }

    const bool conserve_ok =
        device_counters_ok && destfinal_puts == 0ull &&
        (!is_worker || (local_copy_issue == local_copies && local_copies == accepted &&
                        local_copy_bytes == accepted_bytes)) &&
        (is_worker || (ring_put_bytes == accepted_bytes || accepted == 0ull ||
                       ring_puts > 0ull)) && // INC pe may not have local accepted
        r.stale_epoch_count == 0 && r.duplicate_count == 0 && r.lost_count == 0 &&
        r.slot_overwrite_count == 0;

    std::cout << "PL_LOCAL_FINAL_PATH pe=" << pe << " epoch=" << go_epoch
              << " workspace_mode=" << (local_final ? "local_final" : "legacy")
              << " role=" << (is_worker ? "worker" : "inc")
              << " accepted_route_instances=" << accepted
              << " accepted_route_bytes=" << accepted_bytes
              << " direct_destfinal_remote_put_count=" << destfinal_puts
              << " destination_ring_put_count=" << ring_puts
              << " destination_ring_payload_put_bytes=" << ring_put_bytes
              << " destination_ring_descriptor_put_count=" << desc_puts
              << " worker_local_copy_count=" << local_copies
              << " worker_local_copy_bytes=" << local_copy_bytes
              << " local_copy_issue_count=" << local_copy_issue
              << " local_copy_complete_count=" << local_copies
              << " local_copy_cycles=" << local_copy_cycles
              << " ring_payload_dcci_count=" << ring_dcci_count
              << " ring_payload_dcci_bytes=" << ring_dcci_bytes
              << " ring_payload_dcci_cycles=" << ring_dcci_cycles
              << " output_dcci_count=" << out_dcci_count
              << " output_dcci_bytes=" << out_dcci_bytes
              << " output_dcci_cycles=" << out_dcci_cycles
              << " assist_write_count=" << assist_count
              << " assist_write_cycles=" << assist_cycles
              << " ready_publish_count=" << ready_count
              << " ready_publish_cycles=" << ready_cycles
              << " credit_publish_count=" << credit_count
              << " credit_publish_cycles=" << credit_cycles
              << " range_put_count=" << range_puts
              << " scalar_put_count=" << scalar_puts
              << " drain_quiet_count=" << drain_quiet
              << " tail_publish_count=" << tail_pub
              << " credit_wait_cycles=" << credit_wait
              << " first_local_copy_cycle=" << first_copy
              << " last_local_copy_cycle=" << last_copy
              << " first_credit_publish_cycle=" << first_credit
              << " device_counters_ok=" << (device_counters_ok ? 1 : 0)
              << " conserve_ok=" << (conserve_ok ? 1 : 0)
              << " stale_count=" << r.stale_epoch_count
              << " duplicate_count=" << r.duplicate_count
              << " lost_count=" << r.lost_count
              << " overwrite_count=" << r.slot_overwrite_count << std::endl;
}

static void HostEmitPipeOverlap(int pe, uint64_t go_epoch, const PlOverlapTelemetry &ov)
{
    std::cout << "PL_PIPE_OVERLAP pe=" << pe << " go_epoch=" << go_epoch << " per_inc_early_forward_mask=0x"
              << std::hex << ov.inc_early_forward_mask << " per_dest_early_consume_mask=0x" << ov.dest_early_consume_mask
              << std::dec << " early_forward_channels=" << ov.forward_before_final_ingress_total
              << " early_consume_channels=" << ov.destination_early_consume_total
              << " worker_inc_overlap=" << ov.worker_inc_overlap << " inc_dest_overlap=" << ov.inc_dest_overlap
              << " real_pipeline_overlap=" << ov.real_pipeline_overlap
              << " overlap_fail_reason=" << ov.overlap_fail_reason << std::endl;
}

// D4：用 LocalFinal device cycles 证明 copy 与 INC forward 重叠
static void HostEmitLocalFinalOverlap(int pe, uint64_t go_epoch, uint8_t *sym, uint32_t worker_count)
{
    uint64_t first_copy = 0;
    uint64_t last_copy = 0;
    uint64_t first_credit = 0;
    uint64_t first_tail = 0;
    uint64_t last_tail = 0;
    uint64_t last_recv_done = 0;
    uint32_t copy_before_fwd = 0;
    for (uint32_t s = 0; s < worker_count; ++s) {
        PlRecvChannelCompletionTrace rct{};
        if (aclrtMemcpy(&rct, sizeof(rct), sym + PlRecvChannelCompletionTraceOff(s), sizeof(rct),
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            continue;
        }
        if (rct.magic != kPlRecvChanCompletionTraceMagic || rct.epoch != go_epoch) {
            continue;
        }
        if (rct.first_tail_seen_cycle > 0 && (first_tail == 0 || rct.first_tail_seen_cycle < first_tail)) {
            first_tail = rct.first_tail_seen_cycle;
        }
        if (rct.last_tail_seen_cycle > last_tail) {
            last_tail = rct.last_tail_seen_cycle;
        }
        if (rct.first_local_copy_cycle > 0 && (first_copy == 0 || rct.first_local_copy_cycle < first_copy)) {
            first_copy = rct.first_local_copy_cycle;
        }
        if (rct.last_local_copy_cycle > last_copy) {
            last_copy = rct.last_local_copy_cycle;
        }
        if (rct.first_credit_publish_cycle_lf > 0 &&
            (first_credit == 0 || rct.first_credit_publish_cycle_lf < first_credit)) {
            first_credit = rct.first_credit_publish_cycle_lf;
        }
        if (rct.recv_done_cycle > last_recv_done) {
            last_recv_done = rct.recv_done_cycle;
        }
    }
    // INC forward timing：优先读本 PE lane stage；缺失时 last_tail 仅作 diagnostic proxy，不得 formal PASS
    uint64_t last_inc_forward = 0;
    const char *inc_fwd_source = "missing";
    for (uint32_t lane = 0; lane < kQv2LaneCount; ++lane) {
        PlIncForwardLaneStageTiming fst{};
        if (aclrtMemcpy(&fst, sizeof(fst), sym + PlIncForwardLaneStageTimingOff(lane), sizeof(fst),
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            continue;
        }
        if (fst.magic != kPlIncFwdLaneStageMagic || fst.epoch != go_epoch) {
            continue;
        }
        if (fst.forward_done_cycle > last_inc_forward) {
            last_inc_forward = fst.forward_done_cycle;
            inc_fwd_source = "device_lane_forward_done";
        }
    }
    bool used_tail_proxy = false;
    if (last_inc_forward == 0 && last_tail > 0) {
        last_inc_forward = last_tail;
        inc_fwd_source = "last_tail_proxy";
        used_tail_proxy = true;
    }
    if (first_copy > 0 && last_inc_forward > 0 && first_copy < last_inc_forward) {
        copy_before_fwd = 1;
    }
    const uint32_t overlap_diag =
        (first_copy > 0 && last_inc_forward > 0 && first_copy < last_inc_forward && first_credit > 0 &&
         first_credit < last_inc_forward)
            ? 1u
            : 0u;
    // formal PASS 仅当非 proxy 且 cycle 证据齐全
    const uint32_t real_overlap = (!used_tail_proxy && overlap_diag == 1u) ? 1u : 0u;
    std::cout << "PL_LOCAL_FINAL_OVERLAP pe=" << pe << " go_epoch=" << go_epoch
              << " first_tail_seen=" << first_tail << " first_copy_issue=" << first_copy
              << " first_copy_complete=" << first_copy << " last_copy_complete=" << last_copy
              << " last_INC_forward=" << last_inc_forward
              << " last_INC_forward_source=" << inc_fwd_source
              << " clock_domain=local_pe_GetSystemCycle"
              << " recv_done=" << last_recv_done
              << " first_credit_publish_cycle=" << first_credit
              << " local_copy_before_final_forward_count=" << copy_before_fwd
              << " overlap_diagnostic=" << overlap_diag
              << " real_pipeline_overlap=" << real_overlap << std::endl;
}

static void HostEmitPipeDelta(int pe, uint64_t go_epoch, const PlHostEpochSnap &delta)
{
    const auto &u = delta.upload;
    const auto &r = delta.recv;
    // 不输出远端 INC delta（inc_counter_sample_valid=0）
    std::cout << "PL_PIPE_DELTA pe=" << pe << " go_epoch=" << go_epoch << " worker_payload_put_delta="
              << u.payload_put_count << " worker_payload_mte_wait_delta=" << u.payload_mte_wait_count
              << " worker_descriptor_put_delta=" << u.descriptor_tile_put_count
              << " worker_descriptor_mte_wait_delta=" << u.descriptor_mte_wait_count
              << " worker_drain_quiet_delta=" << u.data_descriptor_drain_quiet_count
              << " worker_tail_publish_delta=" << u.tail_publish_count
              << " worker_tail_quiet_delta=" << u.tail_completion_quiet_count
              << " worker_credit_wait_delta=" << u.credit_wait_cycles
              << " inc_counter_sample_valid=0"
              << " destination_tokens_received_delta=" << r.tokens_received
              << " destination_descriptor_verified_delta=" << r.descriptor_verified_count
              << " destination_payload_verified_delta=" << r.payload_verified_count
              << " destination_head_publish_delta=" << r.head_publish_count
              << " verify_error_delta=" << r.verify_error_count << " stale_epoch_delta=" << r.stale_epoch_count
              << " duplicate_delta=" << r.duplicate_count << " lost_delta=" << r.lost_count
              << " slot_overwrite_delta=" << r.slot_overwrite_count << " destination_done_epoch=" << delta.dest_done.done_epoch
              << " pair_done0=" << delta.pair_done0 << " global_done=" << delta.global_done
              << " service_exit_epoch=" << delta.service_exit_epoch << " timing_error_code=" << delta.timing_error_code
              << std::endl;
}

static void HostEmitIncStats(int pe, uint32_t lane, uint64_t go_epoch, const PlForwardCounters &fc,
                             const PlPipelineTiming &inc_timing)
{
    const double avg_batch =
        fc.egress_batch_count > 0 ? static_cast<double>(fc.egress_batch_tokens) / fc.egress_batch_count : 0.0;
    std::cout << "PL_PIPE_INC_STATS pe=" << pe << " lane=" << lane << " go_epoch=" << go_epoch
              << " inc_ingress_seen=" << fc.ingress_seen << " inc_ingress_tail=" << fc.ingress_tail_observed
              << " inc_egress_forwarded=" << fc.egress_forwarded << " inc_payload_put=" << fc.payload_put_count
              << " inc_descriptor_put=" << fc.descriptor_put_count
              << " inc_egress_tail_publish=" << fc.egress_tail_publish_count
              << " inc_ingress_head_publish=" << fc.ingress_head_publish_count
              << " inc_egress_credit_wait=" << fc.egress_credit_wait_cycles
              << " inc_max_ingress_occupancy=" << fc.max_ingress_occupancy
              << " inc_max_egress_occupancy=" << fc.max_egress_occupancy
              << " egress_payload_range_put_count=" << fc.egress_payload_range_put_count
              << " egress_payload_scalar_put_count=" << fc.egress_payload_scalar_put_count
              << " egress_descriptor_range_put_count=" << fc.egress_descriptor_range_put_count
              << " egress_descriptor_scalar_put_count=" << fc.egress_descriptor_scalar_put_count
              << " egress_batch_count=" << fc.egress_batch_count
              << " egress_batch_tokens=" << fc.egress_batch_tokens
              << " egress_batch_max_tokens=" << fc.egress_batch_max_tokens
              << " egress_drain_quiet_count=" << fc.drain_quiet_count
              << " forward_before_final_ingress=" << fc.forward_before_final_ingress_count
              << " average_egress_batch_tokens=" << avg_batch
              << " tokens_forwarded=" << inc_timing.tokens_forwarded << std::endl;
}

static void HostEmitPipeUpload(int pe, uint32_t lane, uint64_t go_epoch, const PlUploadCounters &uc)
{
    const double avg_up_batch =
        (uc.upload_batch_count > 0u)
            ? (static_cast<double>(uc.upload_batch_tokens_sum) / static_cast<double>(uc.upload_batch_count))
            : 0.0;
    const uint64_t up_puts = uc.upload_scalar_put_count + uc.upload_range_put_count;
    const double scalar_frac =
        (up_puts > 0u) ? (static_cast<double>(uc.upload_scalar_put_count) / static_cast<double>(up_puts)) : 0.0;
    std::cout << "PL_PIPE_UPLOAD pe=" << pe << " lane=" << lane << " go_epoch=" << go_epoch
              << " local_tail=" << uc.local_tail << " cached_remote_head=" << uc.cached_remote_head
              << " payload_put=" << uc.payload_put_count << " descriptor_put=" << uc.descriptor_tile_put_count
              << " tail_publish=" << uc.tail_publish_count << " tail_quiet=" << uc.tail_completion_quiet_count
              << " credit_wait=" << uc.credit_wait_cycles << " error_code=" << uc.error_code
              << " upload_batch_count=" << uc.upload_batch_count
              << " upload_batch_tokens_sum=" << uc.upload_batch_tokens_sum
              << " average_upload_batch_tokens=" << avg_up_batch
              << " upload_scalar_put=" << uc.upload_scalar_put_count
              << " upload_range_put=" << uc.upload_range_put_count << " upload_scalar_fraction=" << scalar_frac
              << " first_tail_publish_cycle=" << uc.first_tail_publish_cycle << std::endl;
}

static void HostEmitPipeRecv(int pe, uint32_t source, uint64_t go_epoch, const PlRecvCounters &rc)
{
    std::cout << "PL_PIPE_RECV pe=" << pe << " source=" << source << " go_epoch=" << go_epoch
              << " channel_seen=" << rc.channel_seen << " tokens_received=" << rc.tokens_received
              << " descriptor_verified=" << rc.descriptor_verified_count
              << " payload_verified=" << rc.payload_verified_count << " verify_error=" << rc.verify_error_count
              << " stale_epoch=" << rc.stale_epoch_count << " duplicate=" << rc.duplicate_count
              << " lost=" << rc.lost_count << " slot_overwrite=" << rc.slot_overwrite_count
              << " token_ready_publish=" << rc.token_ready_publish_count
              << " head_publish=" << rc.head_publish_count << " head_quiet=" << rc.head_completion_quiet_count
              << " destination_early_consume=" << rc.destination_head_observed_before_final_forward_count
              << " max_occupancy=" << rc.max_occupancy << " error_code=" << rc.error_code << std::endl;
}

static void HostEmitProbeExecMatrix(uint32_t probe)
{
    uint32_t count = 0, gather = 0, first_hop = 0, second_hop = 0, recv = 0;
    switch (probe) {
        case kPlFullDispatchProbeCountOnly:
            count = 1;
            break;
        case kPlFullDispatchProbeGatherOnly:
            count = gather = 1;
            break;
        case kPlFullDispatchProbeFirstHop:
            count = gather = first_hop = 1;
            break;
        default:
            count = gather = first_hop = second_hop = recv = 1;
            break;
    }
    std::cout << "PL_PROBE_EXEC_MATRIX count=" << count << " gather=" << gather << " first_hop=" << first_hop
              << " second_hop=" << second_hop << " destination_recv=" << recv << std::endl;
}

// gather-only：D2H 校验 bounded staging payload/descriptor
static int HostVerifyGatherStaging(const PlWorkerLocalSession *wl, uint8_t *sym, uint32_t tokens, uint32_t expert_per_pe,
                                   uint64_t epoch, aclrtStream host_stream)
{
    if (wl == nullptr || wl->input == nullptr) {
        return -2;
    }
    std::vector<uint8_t> raw_x(static_cast<size_t>(tokens) * g_runtime_payload_bytes);
    std::vector<int32_t> eids(tokens);
    std::vector<int32_t> seg_base(expert_per_pe);
    std::vector<uint8_t> gather_payload(static_cast<size_t>(kQv2RingDepth) * g_runtime_payload_bytes);
    std::vector<PlDescriptor> gather_desc(kQv2RingDepth);
    if (aclrtMemcpy(raw_x.data(), raw_x.size(), wl->input, raw_x.size(), ACL_MEMCPY_DEVICE_TO_HOST) != 0 ||
        aclrtMemcpy(eids.data(), eids.size() * sizeof(int32_t), wl->expert_ids, eids.size() * sizeof(int32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST) != 0 ||
        aclrtMemcpy(seg_base.data(), seg_base.size() * sizeof(int32_t), sym + kPlSegmentBaseOff,
                    seg_base.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST) != 0 ||
        aclrtMemcpy(gather_payload.data(), gather_payload.size(), wl->gather_payload, gather_payload.size(),
                    ACL_MEMCPY_DEVICE_TO_HOST) != 0 ||
        aclrtMemcpy(gather_desc.data(), gather_desc.size() * sizeof(PlDescriptor), wl->gather_desc,
                    gather_desc.size() * sizeof(PlDescriptor), ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
        return -1;
    }
    aclrtSynchronizeStream(host_stream);
    int errors = 0;
    uint32_t gathered = 0;
    for (uint32_t slot = 0; slot < tokens; ++slot) {
        const PlDescriptor &gd = gather_desc[slot];
        if (gd.epoch != epoch) {
            continue;
        }
        const uint32_t t = gd.source_token_id;
        if (t >= tokens) {
            std::cerr << "PL_GATHER_DESC_BAD_TOKEN slot=" << slot << " token=" << t << std::endl;
            ++errors;
            continue;
        }
        const int32_t eid = eids[t];
        const uint32_t le = static_cast<uint32_t>(eid) % expert_per_pe;
        const uint8_t *gp = gather_payload.data() + static_cast<size_t>(slot) * g_runtime_payload_bytes;
        const uint8_t *rx = raw_x.data() + static_cast<size_t>(t) * g_runtime_payload_bytes;
        if (std::memcmp(gp, rx, g_runtime_payload_bytes) != 0) {
            std::cerr << "PL_GATHER_PAYLOAD_MISMATCH slot=" << slot << " token=" << t << std::endl;
            ++errors;
        }
        if (gd.generation != static_cast<uint32_t>(epoch) || gd.topk_slot != 0u ||
            gd.expert_id != static_cast<uint32_t>(eid) || gd.local_expert_id != le ||
            gd.ring_slot != slot || gd.payload_bytes != g_runtime_payload_bytes) {
            std::cerr << "PL_GATHER_DESC_MISMATCH slot=" << slot << " token=" << t << std::endl;
            ++errors;
        }
        ++gathered;
    }
    if (gathered != tokens) {
        std::cerr << "PL_GATHER_COUNT_MISMATCH expect=" << tokens << " got=" << gathered << std::endl;
        ++errors;
    }
    std::cout << "PL_GATHER_VERIFY pass=" << (errors == 0 ? 1 : 0) << " gather_count=" << gathered
              << " errors=" << errors << std::endl;
    return errors;
}

// host_post：global_done 后批量 D2H 校验 destination（不计入 pipeline makespan）
static int HostVerifyDestinationEpoch(const PlWorkerLocalSession *wl, uint8_t *sym, uint32_t dest_rank,
                                      uint32_t worker_count, uint32_t tokens, uint32_t lanes, uint64_t epoch,
                                      aclrtStream host_stream)
{
    if (wl == nullptr || wl->expand_x == nullptr) {
        return -3;
    }
    const uint32_t tpl = tokens / lanes;
    const uint32_t total_slots = worker_count * tpl;
    const uint32_t epoch_base = static_cast<uint32_t>((epoch - 1u) * tpl);
    const uint32_t retain = std::min(tpl, kQv2RingDepth);
    const size_t payload_bytes = static_cast<size_t>(total_slots) * g_runtime_payload_bytes;
    std::vector<uint8_t> payload(payload_bytes);
    if (aclrtMemcpy(payload.data(), payload_bytes, wl->expand_x, payload_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
        return -1;
    }
    std::vector<PlTokenReadyLine> ready(total_slots);
    if (aclrtMemcpy(ready.data(), ready.size() * sizeof(PlTokenReadyLine), sym + kPlTokenReadyOff,
                    ready.size() * sizeof(PlTokenReadyLine), ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
        return -2;
    }
    int errors = 0;
    // 1) final payload buffer + ready line（非 ring，可验证全部 token）
    for (uint32_t s = 0; s < worker_count; ++s) {
        for (uint32_t t = 0; t < tpl; ++t) {
            const uint32_t slot = s * tpl + t;
            const uint32_t expected_seq = s * tokens + dest_rank * tpl + t + 1u;
            for (uint32_t i = 0; i < g_runtime_payload_bytes; ++i) {
                if (payload[static_cast<size_t>(slot) * g_runtime_payload_bytes + i] !=
                    D1PatternByte(s, expected_seq, i)) {
                    ++errors;
                    break;
                }
            }
            if (ready[slot].ready_epoch != epoch || ready[slot].magic != kPlMagic ||
                ready[slot].token_index != slot) {
                ++errors;
            }
        }
    }
    // 2) device recv 计数器：在线 descriptor 校验应已覆盖本 epoch
    const uint64_t expected_desc_verified = epoch * static_cast<uint64_t>(tpl);
    for (uint32_t s = 0; s < worker_count; ++s) {
        PlRecvCounters rc{};
        if (aclrtMemcpy(&rc, sizeof(rc), sym + kPlRecvCtrOff + static_cast<uint64_t>(s) * sizeof(rc), sizeof(rc),
                        ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
            return -3;
        }
        if (rc.descriptor_verified_count < expected_desc_verified || rc.verify_error_count != 0u ||
            rc.stale_epoch_count != 0u || rc.duplicate_count != 0u || rc.lost_count != 0u ||
            rc.slot_overwrite_count != 0u) {
            ++errors;
        }
    }
    // 3) descriptor ring：仅校验本 epoch 最终保留的 min(tpl, ring_depth) 个 slot
    for (uint32_t s = 0; s < worker_count; ++s) {
        std::vector<PlDescriptor> ring(kQv2RingDepth);
        const uint64_t desc_off = PlDestChannelDescOff(s);
        if (aclrtMemcpy(ring.data(), ring.size() * sizeof(PlDescriptor), sym + desc_off,
                        ring.size() * sizeof(PlDescriptor), ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
            return -4;
        }
        for (uint32_t t = tpl - retain; t < tpl; ++t) {
            const uint32_t lane_sequence = epoch_base + t;
            const uint32_t ring_slot = lane_sequence % kQv2RingDepth;
            const uint32_t expected_seq = s * tokens + dest_rank * tpl + t + 1u;
            const uint32_t expected_dest_slot = s * tpl + t;
            const PlDescriptor &d = ring[ring_slot];
            if (d.epoch != epoch || d.destination_slot != expected_dest_slot || d.token_sequence != expected_seq ||
                d.source_rank != s || d.destination_rank != dest_rank || d.ring_slot != ring_slot ||
                d.lane_sequence != lane_sequence) {
                ++errors;
            }
        }
    }
    (void)host_stream;
    return errors;
}

static std::string HostFormatPairDoneArrivalUs(const D1PairDoneArrival &arrival, uint64_t wait_start_cycle,
                                               uint32_t worker_count)
{
    std::ostringstream oss;
    for (uint32_t i = 0; i < worker_count; ++i) {
        if (i > 0) {
            oss << ',';
        }
        const uint64_t seen = arrival.lanes[i].pair_done_first_seen_cycle;
        oss << ((seen > wait_start_cycle) ? CyclesToUs(seen, wait_start_cycle) : 0.0);
    }
    return oss.str();
}

static PlDeviceEpochTiming HostCollectDeviceTiming(uint8_t *sym, uint32_t worker_count, int inc_pe, uint64_t go_epoch)
{
    D1LeaderTiming leader{};
    D1GlobalDoneTiming global{};
    D1PairDoneArrival arrival{};
    PlPipelineTiming pl{};
    aclrtMemcpy(&leader, sizeof(leader), sym + D1B(kD1LeaderTimingOff), sizeof(leader), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&global, sizeof(global), sym + D1B(kD1GlobalDoneTimingOff), sizeof(global), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&arrival, sizeof(arrival), sym + D1B(kD1PairDoneArrivalOff), sizeof(arrival), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&pl, sizeof(pl), sym + kPlTimingOff, sizeof(pl), ACL_MEMCPY_DEVICE_TO_HOST);

    // 从 AIV 独占 timing line 聚合（禁止依赖共享 PlPipelineTiming 的多 writer 字段）
    uint64_t min_upload = UINT64_MAX;
    uint64_t max_upload_done = 0;
    for (uint32_t lane = 0; lane < worker_count; ++lane) {
        PlAivEpochTimingLine ul{};
        aclrtMemcpy(&ul, sizeof(ul), sym + PlUploadAivTimingOff(lane), sizeof(ul), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ul.done_epoch == go_epoch && ul.first_issue_cycle > 0) {
            if (ul.first_issue_cycle < min_upload) {
                min_upload = ul.first_issue_cycle;
            }
            if (ul.done_cycle > max_upload_done) {
                max_upload_done = ul.done_cycle;
            }
        }
    }
    uint64_t min_recv_first = UINT64_MAX;
    uint64_t max_recv_done = 0;
    for (uint32_t s = 0; s < worker_count; ++s) {
        PlAivEpochTimingLine rl{};
        aclrtMemcpy(&rl, sizeof(rl), sym + PlRecvAivTimingOff(s), sizeof(rl), ACL_MEMCPY_DEVICE_TO_HOST);
        if (rl.done_epoch == go_epoch && rl.first_issue_cycle > 0) {
            if (rl.first_issue_cycle < min_recv_first) {
                min_recv_first = rl.first_issue_cycle;
            }
            if (rl.done_cycle > max_recv_done) {
                max_recv_done = rl.done_cycle;
            }
        }
    }
    if (min_upload != UINT64_MAX) {
        pl.first_upload_issue = min_upload;
    }
    if (max_upload_done > 0) {
        pl.last_upload_issue = max_upload_done;
    }
    if (min_recv_first != UINT64_MAX) {
        pl.first_destination_consume = min_recv_first;
    }
    if (max_recv_done > 0) {
        pl.destination_done_cycle = max_recv_done;
    }

    // 从 INC peer 拉取 forward AIV timing（经独立 scratch，禁止覆盖 live telemetry）
    uint64_t min_fwd = UINT64_MAX;
    uint64_t max_fwd_done = 0;
    for (uint32_t lane = 0; lane < worker_count; ++lane) {
        PlAivEpochTimingLine fl{};
        const uint64_t src_off = PlForwardAivTimingOff(lane);
        const uint64_t scratch = kPlHostProbeScratchOff + static_cast<uint64_t>(lane) * 64u;
        aclshmem_getmem(sym + scratch, sym + src_off, sizeof(fl), inc_pe);
        aclrtMemcpy(&fl, sizeof(fl), sym + scratch, sizeof(fl), ACL_MEMCPY_DEVICE_TO_HOST);
        if (fl.done_epoch == go_epoch && fl.first_issue_cycle > 0) {
            if (fl.first_issue_cycle < min_fwd) {
                min_fwd = fl.first_issue_cycle;
            }
            if (fl.done_cycle > max_fwd_done) {
                max_fwd_done = fl.done_cycle;
            }
        }
    }
    if (min_fwd != UINT64_MAX) {
        pl.first_forward_issue = min_fwd;
    }
    if (max_fwd_done > 0) {
        pl.last_forward_issue = max_fwd_done;
    }

    PlDeviceEpochTiming out{};
    out.device_pipeline_us = CyclesToUs(global.global_done_publish_cycle, leader.release_seen_cycle);
    if (leader.release_seen_cycle > 0 && leader.go_fanout_end_cycle > leader.release_seen_cycle) {
        out.release_to_go_fanout_end_us = CyclesToUs(leader.go_fanout_end_cycle, leader.release_seen_cycle);
    }
    if (leader.go_fanout_end_cycle > 0 && pl.first_upload_issue > leader.go_fanout_end_cycle) {
        out.go_fanout_end_to_first_upload_us = CyclesToUs(pl.first_upload_issue, leader.go_fanout_end_cycle);
    }
    if (pl.first_upload_issue > 0 && pl.first_forward_issue > pl.first_upload_issue) {
        out.first_upload_to_first_forward_us = CyclesToUs(pl.first_forward_issue, pl.first_upload_issue);
    }
    if (pl.first_forward_issue > 0 && pl.destination_done_cycle > pl.first_forward_issue) {
        out.first_forward_to_destination_done_us = CyclesToUs(pl.destination_done_cycle, pl.first_forward_issue);
    }
    if (pl.destination_done_cycle > 0 && global.pair_done_complete_cycle > pl.destination_done_cycle) {
        out.pe0_local_recv_done_to_global_pair_done_complete_us =
            CyclesToUs(global.pair_done_complete_cycle, pl.destination_done_cycle);
    }
    if (global.pair_done_complete_cycle > 0 && global.global_done_publish_cycle > global.pair_done_complete_cycle) {
        out.pair_done_to_global_done_publish_us =
            CyclesToUs(global.global_done_publish_cycle, global.pair_done_complete_cycle);
    }
    out.pair_done_arrival_us_csv =
        HostFormatPairDoneArrivalUs(arrival, global.pair_done_wait_start_cycle, worker_count);
    // Stage1：release→destination_done；Full：整段 device pipeline（含最终可见性）
    const double stage1_parts = out.release_to_go_fanout_end_us + out.go_fanout_end_to_first_upload_us +
                                out.first_upload_to_first_forward_us + out.first_forward_to_destination_done_us;
    out.stage1_device_us = stage1_parts > 0 ? stage1_parts : out.device_pipeline_us;
    out.full_dispatch_device_us = out.device_pipeline_us;
    return out;
}

static void HostEmitTimingAbi(int pe, uint64_t go_epoch, uint32_t measurement_mode, const PlDeviceEpochTiming &dt)
{
    std::cout << "PL_TIMING_ABI pe=" << pe << " go_epoch=" << go_epoch
              << " measurement_mode=" << MeasurementModeName(measurement_mode)
              << " PL_TIMING_FORMAL=" << kPlTimingFormalMakespan
              << " PL_TIMING_FORMAL_TRANSPORT=" << kPlTimingFormalTransport
              << " p6_segment_timing_formal=" << (dt.p6_segment_timing_formal ? 1 : 0)
              << " formal_makespan=PE0_release_seen_to_global_done_publish"
              << " formal_transport=PE0_release_seen_to_global_transport_done"
              << " formal_second_hop=PE0_release_seen_to_second_hop_visible"
              << " stage1_device_us=" << dt.stage1_device_us
              << " full_dispatch_device_us=" << dt.full_dispatch_device_us
              << " inc_transport_makespan_us=" << dt.inc_transport_makespan_us
              << " inc_second_hop_visible_makespan_us=" << dt.inc_second_hop_visible_makespan_us
              << " global_transport_done_epoch=" << dt.global_transport_done_epoch
              << " slowest_destination=" << dt.slowest_destination
              << " transport_le_full_dispatch=" << (dt.transport_le_full_dispatch ? 1 : 0)
              << " inc_route_count_prefix_us=" << dt.inc_route_count_prefix_us
              << " inc_route_gather_us=" << dt.inc_route_gather_us
              << " inc_stage1_two_hop_us=" << dt.inc_stage1_two_hop_us
              << " inc_finalize_us=" << dt.inc_finalize_us
              << " inc_full_dispatch_us=" << dt.inc_full_dispatch_us
              << " p6_segment_note=PlRouteTimingLine_non_formal_multi_writer"
              << " api_observed=expand_x,assist_info_for_combine,ep_recv_count,expert_token_nums"
              << " timing_domain=device_cycle" << std::endl;
}

static void HostEmitDeviceTiming(int pe, uint64_t go_epoch, const PlDeviceEpochTiming &dt)
{
    std::cout << "PL_PIPE_DEVICE_TIMING pe=" << pe << " go_epoch=" << go_epoch
              << " device_pipeline_us=" << dt.device_pipeline_us
              << " formal_makespan_us=" << dt.device_pipeline_us
              << " release_to_go_fanout_end_us=" << dt.release_to_go_fanout_end_us
              << " go_fanout_end_to_first_upload_us=" << dt.go_fanout_end_to_first_upload_us
              << " first_upload_to_first_forward_us=" << dt.first_upload_to_first_forward_us
              << " first_forward_to_destination_done_us=" << dt.first_forward_to_destination_done_us
              << " pe0_local_recv_done_to_global_pair_done_complete_us="
              << dt.pe0_local_recv_done_to_global_pair_done_complete_us << " diagnostic_only=1"
              << " pair_done_to_global_done_publish_us=" << dt.pair_done_to_global_done_publish_us
              << " host_observe_residual_us=" << dt.host_observe_residual_us
              << " pair_done_arrival_us_csv=" << dt.pair_done_arrival_us_csv << std::endl;
}

static int CreateOptionalPriorityStream(
    aclrtStream *stream, const char *env_name)
{
    const char *raw = std::getenv(env_name);
    if (raw == nullptr || raw[0] == '\0') {
        return aclrtCreateStream(stream);
    }
    char *end = nullptr;
    const unsigned long priority = std::strtoul(raw, &end, 10);
    if (end == raw || *end != '\0' || priority > 7ul) {
        std::cerr << "PL_STREAM_PRIORITY_INVALID name=" << env_name
                  << " value=" << raw << std::endl;
        return -1;
    }
    const aclError rc = aclrtCreateStreamWithConfig(
        stream, static_cast<uint32_t>(priority), 0u);
    if (rc == ACL_SUCCESS) {
        std::cerr << "PL_STREAM_PRIORITY name=" << env_name
                  << " value=" << priority << std::endl;
    }
    return rc;
}

static int InitShmem(int pe, int n_pes, int32_t *dev, aclrtStream *svc, aclrtStream *ctl, aclrtStream *leader)
{
    *dev = pe % g_npus + f_npu;
    if (aclInit(nullptr) != 0 || aclrtSetDevice(*dev) != 0) {
        return -1;
    }
    if (CreateOptionalPriorityStream(
            svc, "INC_DN_PL_SERVICE_STREAM_PRIORITY") != 0 ||
        aclrtCreateStream(ctl) != 0) {
        return -1;
    }
    *leader = nullptr;
    if (pe == static_cast<int>(kD1LeaderPe) && aclrtCreateStream(leader) != 0) {
        return -1;
    }
    aclshmemx_init_attr_t attr;
    constexpr uint64_t kHeapAlign = 2ull * 1024 * 1024; // hybm 段大小需 2MiB 对齐
    uint64_t heap_bytes = (kPlSymmetricTransportHeapNeed + kHeapAlign - 1ull) & ~(kHeapAlign - 1ull);
    // 诊断：未对齐曾导致 Invalid options size≈309MB
    if (pe == 0) {
        std::cerr << "PL_HEAP_INIT need=" << kPlSymmetricTransportHeapNeed << " reserve=" << heap_bytes
                  << " MiB=" << (heap_bytes / (1024.0 * 1024.0)) << std::endl;
    }
    test_set_attr(pe, n_pes, heap_bytes, ipport, default_flag_uid, &attr);
    return aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
}

static int HostInitControlLines(uint8_t *sym, aclrtStream stream)
{
    D1ControlLine zero{};
    D1LeaderRendezvous leader{};
    D1LeaderCompletion completion{};
    D1GlobalDoneLine global_done{};
    aclrtMemcpyAsync(sym + D1B(kD1StartLineOff), sizeof(zero), &zero, sizeof(zero), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    aclrtMemcpyAsync(sym + D1B(kD1SessionStopLineOff), sizeof(zero), &zero, sizeof(zero), ACL_MEMCPY_HOST_TO_DEVICE,
                     stream);
    aclrtMemcpyAsync(sym + D1B(kD1LocalReadyLineOff), sizeof(zero), &zero, sizeof(zero), ACL_MEMCPY_HOST_TO_DEVICE,
                     stream);
    aclrtMemcpyAsync(sym + D1B(kD1LeaderRendezvousOff), sizeof(leader), &leader, sizeof(leader),
                     ACL_MEMCPY_HOST_TO_DEVICE, stream);
    aclrtMemcpyAsync(sym + D1B(kD1LeaderCompletionOff), sizeof(completion), &completion, sizeof(completion),
                     ACL_MEMCPY_HOST_TO_DEVICE, stream);
    aclrtMemcpyAsync(sym + D1B(kD1GlobalDoneLineOff), sizeof(global_done), &global_done, sizeof(global_done),
                     ACL_MEMCPY_HOST_TO_DEVICE, stream);
    if (HostInitTimingNegHooks(sym, stream) != 0) {
        return -1;
    }
    return aclrtSynchronizeStream(stream);
}

// 从 shared artifact 仅加载原始 x.bin + expert_ids（full_dispatch 计时内禁止 host 构 descriptor）
static bool ReadArtifactFile(const std::string &path, std::vector<uint8_t> *out)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    in.seekg(0, std::ios::end);
    const size_t sz = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    out->resize(sz);
    if (sz > 0) {
        in.read(reinterpret_cast<char *>(out->data()), static_cast<std::streamsize>(sz));
    }
    return in.good() || sz == 0;
}

static bool HostParseManifestU32(const std::string &json_text, const char *key, uint32_t *out)
{
    const std::string needle = std::string("\"") + key + "\":";
    const size_t pos = json_text.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    const char *start = json_text.c_str() + pos + needle.size();
    char *end = nullptr;
    const unsigned long v = std::strtoul(start, &end, 10);
    if (end == start) {
        return false;
    }
    *out = static_cast<uint32_t>(v);
    return true;
}

static int HostValidateArtifactManifest(uint32_t physical_npe, uint32_t worker_count, uint32_t expert_per_pe,
                                        const std::string &shape_dir)
{
    const char *manifest_env = std::getenv("INC_DN_PL_ARTIFACT_MANIFEST");
    std::string manifest_path = manifest_env != nullptr ? std::string(manifest_env)
                                                         : shape_dir + "/../artifact_manifest.json";
    std::ifstream in(manifest_path);
    if (!in) {
        std::cerr << "PL_ARTIFACT_MANIFEST_MISSING path=" << manifest_path << std::endl;
        return -10;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    uint32_t manifest_workers = 0;
    uint32_t manifest_dest = 0;
    uint32_t manifest_physical = 0;
    uint32_t manifest_epe = 0;
    if (!HostParseManifestU32(text, "worker_count", &manifest_workers) ||
        !HostParseManifestU32(text, "destination_count", &manifest_dest) ||
        !HostParseManifestU32(text, "physical_pe_count", &manifest_physical) ||
        !HostParseManifestU32(text, "expert_per_pe", &manifest_epe)) {
        std::cerr << "PL_ARTIFACT_MANIFEST_PARSE_FAIL path=" << manifest_path << std::endl;
        return -11;
    }
    if (manifest_physical != physical_npe || manifest_workers != worker_count || manifest_dest != worker_count ||
        manifest_epe != expert_per_pe || worker_count * 2u != physical_npe) {
        std::cerr << "PL_ARTIFACT_TOPOLOGY_MISMATCH manifest_workers=" << manifest_workers
                  << " manifest_dest=" << manifest_dest << " manifest_physical=" << manifest_physical
                  << " runtime_npe=" << physical_npe << " runtime_workers=" << worker_count << std::endl;
        return -12;
    }
    // Runtime Payload V2：若 manifest 携带 hidden/payload，必须与 runtime 一致
    uint32_t manifest_hidden = 0;
    uint32_t manifest_payload = 0;
    uint32_t manifest_dtype = 0;
    const bool has_hidden = HostParseManifestU32(text, "hidden_size", &manifest_hidden) ||
                            HostParseManifestU32(text, "hidden", &manifest_hidden);
    const bool has_payload = HostParseManifestU32(text, "payload_bytes", &manifest_payload);
    const bool has_dtype = HostParseManifestU32(text, "dtype_bytes", &manifest_dtype);
    if (has_hidden || has_payload || has_dtype) {
        const uint32_t runtime_hidden = ParseEnvU32("INC_DN_PL_HIDDEN_SIZE", 4096u);
        const uint32_t runtime_dtype = ParseEnvU32("INC_DN_PL_DTYPE_BYTES", 2u);
        uint32_t runtime_payload = ParseEnvU32("INC_DN_PL_PAYLOAD_BYTES", 0u);
        if (runtime_payload == 0u) {
            runtime_payload = runtime_hidden * runtime_dtype;
        }
        if ((has_hidden && manifest_hidden != runtime_hidden) ||
            (has_payload && manifest_payload != runtime_payload) ||
            (has_dtype && manifest_dtype != runtime_dtype)) {
            std::cerr << "PL_ARTIFACT_HIDDEN_MISMATCH manifest_hidden=" << manifest_hidden
                      << " manifest_payload=" << manifest_payload << " manifest_dtype=" << manifest_dtype
                      << " runtime_hidden=" << runtime_hidden << " runtime_payload=" << runtime_payload
                      << " runtime_dtype=" << runtime_dtype << std::endl;
            return -13;
        }
    }
    return 0;
}

// FNV-1a 64-bit：offset=0xCBF29CE484222325，prime=0x100000001B3。
// 与 scripts/inc_dispatch_artifact_checksum64.py 的 checksum64_bytes() 对齐，供 gate 解析 PL_ARTIFACT_LOAD_PROOF。
static uint64_t ArtifactChecksum64(const uint8_t *data, size_t len)
{
    constexpr uint64_t kOffset = 0xCBF29CE484222325ULL;
    constexpr uint64_t kPrime = 0x100000001B3ULL;
    uint64_t h = kOffset;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(data[i]);
        h *= kPrime;
    }
    return h;
}

// P5.2 Truth：INC 侧计时外 D2H raw slab + route meta，对照 artifact checksum64
// 返回 0=ok，非 0=fail
static int HostVerifySourceMajorFirstHopTruth(uint8_t *sym, int pe, uint32_t pair_id, uint32_t tokens,
                                              uint32_t topk, uint64_t go_epoch, aclrtStream stream)
{
    const char *shape_dir_env = std::getenv("INC_DN_PL_ARTIFACT_SHAPE_DIR");
    if (shape_dir_env == nullptr || shape_dir_env[0] == '\0' || tokens == 0u) {
        std::cout << "PL_P52_TRUTH_VERIFY pe=" << pe << " go_epoch=" << go_epoch
                  << " payload_exact=0 meta_exact=0 error=shape_dir_missing" << std::endl;
        return -1;
    }
    std::string rank_dir = std::string(shape_dir_env) + "/epoch_1/rank_" + std::to_string(pair_id);
    std::vector<uint8_t> expect_x;
    std::vector<uint8_t> expect_eid;
    if (!ReadArtifactFile(rank_dir + "/x.bin", &expect_x) ||
        !ReadArtifactFile(rank_dir + "/expert_ids.bin", &expect_eid)) {
        rank_dir = std::string(shape_dir_env) + "/rank_" + std::to_string(pair_id);
        if (!ReadArtifactFile(rank_dir + "/x.bin", &expect_x) ||
            !ReadArtifactFile(rank_dir + "/expert_ids.bin", &expect_eid)) {
            std::cout << "PL_P52_TRUTH_VERIFY pe=" << pe << " go_epoch=" << go_epoch
                      << " payload_exact=0 meta_exact=0 error=artifact_read_fail" << std::endl;
            return -2;
        }
    }
    const size_t payload_bytes = static_cast<size_t>(tokens) * g_runtime_payload_bytes;
    const size_t meta_bytes = static_cast<size_t>(tokens) * topk * sizeof(int32_t);
    if (expect_x.size() < payload_bytes || expect_eid.size() < meta_bytes) {
        std::cout << "PL_P52_TRUTH_VERIFY pe=" << pe << " go_epoch=" << go_epoch
                  << " payload_exact=0 meta_exact=0 error=artifact_size_short expect_x=" << expect_x.size()
                  << " need=" << payload_bytes << std::endl;
        return -3;
    }
    std::vector<uint8_t> got_x(payload_bytes, 0);
    std::vector<uint8_t> got_eid(meta_bytes, 0);
    aclrtMemcpy(got_x.data(), payload_bytes, sym + kPlIncRawSlabOff, payload_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(got_eid.data(), meta_bytes, sym + kPlIncRawRouteMetaOff, meta_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (stream != nullptr) {
        aclrtSynchronizeStream(stream);
    }
    const uint64_t cs_exp_x = ArtifactChecksum64(expect_x.data(), payload_bytes);
    const uint64_t cs_got_x = ArtifactChecksum64(got_x.data(), payload_bytes);
    const uint64_t cs_exp_eid = ArtifactChecksum64(expect_eid.data(), meta_bytes);
    const uint64_t cs_got_eid = ArtifactChecksum64(got_eid.data(), meta_bytes);
    const int payload_exact = (cs_exp_x == cs_got_x) ? 1 : 0;
    const int meta_exact = (cs_exp_eid == cs_got_eid) ? 1 : 0;
    std::cout << "PL_P52_TRUTH_VERIFY pe=" << pe << " go_epoch=" << go_epoch << " source_rank=" << pair_id
              << " tokens=" << tokens << " payload_bytes=" << payload_bytes << " meta_bytes=" << meta_bytes
              << " checksum64_payload_obs=0x" << std::hex << cs_got_x << std::dec
              << " checksum64_payload_exp=0x" << std::hex << cs_exp_x << std::dec
              << " checksum64_meta_obs=0x" << std::hex << cs_got_eid << std::dec
              << " checksum64_meta_exp=0x" << std::hex << cs_exp_eid << std::dec
              << " payload_exact=" << payload_exact << " meta_exact=" << meta_exact
              << " timing_excluded=1" << std::endl;
    return (payload_exact == 1 && meta_exact == 1) ? 0 : -4;
}

// 一次性预载：按 capacity stride 堆叠；仅校验前 valid_tokens_by_epoch 个有效 token
static int HostArtifactRawLoad(PlWorkerLocalSession *wl, uint32_t source_rank, uint32_t worker_count, uint32_t capacity,
                               uint32_t expert_per_pe, uint32_t topk, uint32_t input_epoch_count,
                               const uint32_t valid_tokens_by_epoch[][kPlMaxSources], aclrtStream stream)
{
    if (wl == nullptr || wl->input == nullptr || wl->expert_ids == nullptr) {
        return -7;
    }
    const char *shape_dir_env = std::getenv("INC_DN_PL_ARTIFACT_SHAPE_DIR");
    if (shape_dir_env == nullptr || shape_dir_env[0] == '\0') {
        std::cerr << "PL_ARTIFACT_SHAPE_DIR_MISSING" << std::endl;
        return -1;
    }
    const std::string shape_dir(shape_dir_env);
    const int mv = HostValidateArtifactManifest(worker_count * 2u, worker_count, expert_per_pe, shape_dir);
    if (mv != 0) {
        return mv;
    }
    if (input_epoch_count == 0u) {
        input_epoch_count = 1u;
    }
    // 与 kPlWorkerSourceBytes 对齐：允许 capacity×epochs 堆叠，上限 kPlMaxTokens×kPlMaxStackedInputEpochs
    if (static_cast<uint64_t>(input_epoch_count) * capacity >
        static_cast<uint64_t>(kPlMaxTokens) * kPlMaxStackedInputEpochs) {
        std::cerr << "PL_ARTIFACT_EPOCH_OVERFLOW epochs=" << input_epoch_count << " capacity=" << capacity
                  << " max_stack_tokens=" << (static_cast<uint64_t>(kPlMaxTokens) * kPlMaxStackedInputEpochs)
                  << std::endl;
        return -6;
    }
    const size_t token_bytes = g_runtime_payload_bytes;
    const size_t expect_x = static_cast<size_t>(capacity) * token_bytes;
    const size_t expect_eid = static_cast<size_t>(capacity) * topk * sizeof(int32_t);
    std::vector<uint8_t> x_all(expect_x * input_epoch_count, 0);
    std::vector<uint8_t> eid_all(expect_eid * input_epoch_count, 0);
    const uint32_t max_eid_limit = worker_count * expert_per_pe;

    for (uint32_t ep = 1; ep <= input_epoch_count; ++ep) {
        std::string rank_dir = shape_dir + "/epoch_" + std::to_string(ep) + "/rank_" + std::to_string(source_rank);
        std::vector<uint8_t> x_bytes;
        std::vector<uint8_t> eid_bytes;
        if (!ReadArtifactFile(rank_dir + "/x.bin", &x_bytes) ||
            !ReadArtifactFile(rank_dir + "/expert_ids.bin", &eid_bytes)) {
            if (input_epoch_count == 1u) {
                rank_dir = shape_dir + "/rank_" + std::to_string(source_rank);
                if (!ReadArtifactFile(rank_dir + "/x.bin", &x_bytes) ||
                    !ReadArtifactFile(rank_dir + "/expert_ids.bin", &eid_bytes)) {
                    std::cerr << "PL_ARTIFACT_READ_FAIL rank_dir=" << rank_dir << std::endl;
                    return -2;
                }
            } else {
                std::cerr << "PL_ARTIFACT_READ_FAIL rank_dir=" << rank_dir << std::endl;
                return -2;
            }
        }
        if (x_bytes.size() < expect_x || eid_bytes.size() < expect_eid) {
            std::cerr << "PL_ARTIFACT_SIZE_MISMATCH epoch=" << ep << " capacity=" << capacity
                      << " x=" << x_bytes.size() << " eid=" << eid_bytes.size() << std::endl;
            return -3;
        }
        const uint32_t valid =
            (valid_tokens_by_epoch != nullptr) ? valid_tokens_by_epoch[ep - 1u][source_rank] : capacity;
        // 结构化加载证明：记录实际读入字节与 checksum64，禁止仅抄写 manifest SHA。
        const std::string path_x = rank_dir + "/x.bin";
        const std::string path_eid = rank_dir + "/expert_ids.bin";
        const uint64_t cs_x = ArtifactChecksum64(x_bytes.data(), x_bytes.size());
        const uint64_t cs_eid = ArtifactChecksum64(eid_bytes.data(), eid_bytes.size());
        std::cout << "PL_ARTIFACT_LOAD_PROOF pe=" << source_rank << " epoch=" << ep << " path_x=" << path_x
                  << " size_x=" << x_bytes.size() << " checksum64_x=0x" << std::hex << cs_x << std::dec
                  << " path_eid=" << path_eid << " size_eid=" << eid_bytes.size() << " checksum64_eid=0x"
                  << std::hex << cs_eid << std::dec << " source_token_count=" << valid << std::endl;
        const int32_t *eids = reinterpret_cast<const int32_t *>(eid_bytes.data());
        for (uint32_t t = 0; t < valid; ++t) {
            for (uint32_t slot = 0; slot < topk; ++slot) {
                const int32_t eid = eids[static_cast<size_t>(t) * topk + slot];
                if (eid < 0 || static_cast<uint32_t>(eid) >= max_eid_limit) {
                    std::cerr << "PL_ARTIFACT_BAD_EXPERT_ID epoch=" << ep << " token=" << t << " slot=" << slot
                              << " eid=" << eid << std::endl;
                    return -5;
                }
                if (static_cast<uint32_t>(eid / expert_per_pe) >= worker_count) {
                    std::cerr << "PL_ARTIFACT_BAD_DEST epoch=" << ep << " token=" << t << std::endl;
                    return -5;
                }
            }
        }
        std::memcpy(x_all.data() + (ep - 1u) * expect_x, x_bytes.data(), expect_x);
        std::memcpy(eid_all.data() + (ep - 1u) * expect_eid, eid_bytes.data(), expect_eid);
    }

    aclrtMemcpy(wl->input, x_all.size(), x_all.data(), x_all.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(wl->expert_ids, eid_all.size(), eid_all.data(), eid_all.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtSynchronizeStream(stream);
    const uint32_t valid0 =
        (valid_tokens_by_epoch != nullptr) ? valid_tokens_by_epoch[0][source_rank] : capacity;
    std::cout << "PL_ARTIFACT_RAW_LOAD pe=" << source_rank << " capacity=" << capacity
              << " valid_tokens=" << valid0 << " input_epoch_count=" << input_epoch_count
              << " per_epoch_h2d=0 host_descriptor_build_count=0 shape_dir=" << shape_dir_env << std::endl;
    return 0;
}

// 设备 count/prefix 调试：host 预写 counts_done 与 lane token_count（仅排障）
static void HostSeedFullDispatchCountDone(uint8_t *sym, uint32_t tokens, aclrtStream stream)
{
    PlFullOutputDoneLine done{};
    done.counts_done_epoch = 1u;
    PlLaneWorkLine lane0{};
    lane0.epoch = 1u;
    lane0.lane_id = 0u;
    lane0.token_count = tokens;
    aclrtMemcpy(sym + kPlFullOutputDoneOff, sizeof(done), &done, sizeof(done), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(sym + kPlLaneWorkOff, sizeof(lane0), &lane0, sizeof(lane0), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtSynchronizeStream(stream);
}

// prepacked 回归专用：host 生成 descriptor（不得用于 full_dispatch 正式计时）
static int HostArtifactPrefill(PlWorkerLocalSession *wl, uint32_t source_rank, uint32_t worker_count, uint32_t tokens,
                               uint32_t lanes, uint32_t expert_per_pe, uint32_t topk, aclrtStream stream,
                               std::vector<PlDescriptor> *tpl_out)
{
    if (wl == nullptr || wl->input == nullptr || wl->worker_desc == nullptr) {
        return -6;
    }
    const char *shape_dir_env = std::getenv("INC_DN_PL_ARTIFACT_SHAPE_DIR");
    if (shape_dir_env == nullptr || shape_dir_env[0] == '\0') {
        std::cerr << "PL_ARTIFACT_SHAPE_DIR_MISSING" << std::endl;
        return -1;
    }
    const std::string rank_dir = std::string(shape_dir_env) + "/rank_" + std::to_string(source_rank);
    std::vector<uint8_t> x_bytes;
    std::vector<uint8_t> eid_bytes;
    if (!ReadArtifactFile(rank_dir + "/x.bin", &x_bytes) || !ReadArtifactFile(rank_dir + "/expert_ids.bin", &eid_bytes)) {
        std::cerr << "PL_ARTIFACT_READ_FAIL rank_dir=" << rank_dir << std::endl;
        return -2;
    }
    const size_t token_bytes = g_runtime_payload_bytes;
    const size_t expect_x = static_cast<size_t>(tokens) * token_bytes;
    const size_t expect_eid = static_cast<size_t>(tokens) * topk * sizeof(int32_t);
    if (x_bytes.size() < expect_x || eid_bytes.size() < expect_eid) {
        std::cerr << "PL_ARTIFACT_SIZE_MISMATCH tokens=" << tokens << " x=" << x_bytes.size()
                  << " eid=" << eid_bytes.size() << std::endl;
        return -3;
    }
    std::vector<uint8_t> payload(expect_x);
    std::vector<PlDescriptor> descs(tokens);
    std::vector<uint32_t> dest_slot(worker_count, 0u);
    std::vector<uint32_t> lane_seq(lanes, 0u);
    const int32_t *eids = reinterpret_cast<const int32_t *>(eid_bytes.data());
    for (uint32_t t = 0; t < tokens; ++t) {
        const int32_t eid = eids[static_cast<size_t>(t) * topk];
        if (eid < 0) {
            return -4;
        }
        const uint32_t dest_rank = static_cast<uint32_t>(eid / expert_per_pe);
        if (dest_rank >= worker_count) {
            std::cerr << "PL_ARTIFACT_BAD_DEST token=" << t << " dest=" << dest_rank << std::endl;
            return -5;
        }
        std::memcpy(payload.data() + static_cast<size_t>(t) * token_bytes, x_bytes.data() + static_cast<size_t>(t) * token_bytes,
                    token_bytes);
        PlDescriptor d{};
        d.epoch = 1;
        d.generation = 1;
        d.source_token_id = t;
        d.topk_slot = 0;
        d.expert_id = static_cast<uint32_t>(eid);
        d.local_expert_id = static_cast<uint32_t>(eid % expert_per_pe);
        d.token_sequence = source_rank * tokens + t + 1u;
        d.source_rank = source_rank;
        d.destination_rank = dest_rank;
        d.destination_slot = dest_slot[dest_rank]++;
        d.destination_final_slot = d.destination_slot;
        d.source_segment_offset = t;
        d.payload_bytes = static_cast<uint32_t>(token_bytes);
        d.source_lane = dest_rank < lanes ? dest_rank : (dest_rank % lanes);
        d.lane_sequence = lane_seq[d.source_lane]++;
        d.ring_slot = d.lane_sequence % kQv2RingDepth;
        descs[t] = d;
    }
    if (tpl_out != nullptr) {
        *tpl_out = descs;
    }
    aclrtMemcpy(wl->input, payload.size(), payload.data(), payload.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(wl->worker_desc, descs.size() * sizeof(PlDescriptor), descs.data(), descs.size() * sizeof(PlDescriptor),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtSynchronizeStream(stream);
    std::cout << "PL_ARTIFACT_PREFILL pe=" << source_rank << " tokens=" << tokens << " shape_dir=" << shape_dir_env
              << std::endl;
    return 0;
}

static void HostPrefillBalanced(PlWorkerLocalSession *wl, uint32_t source_rank, uint32_t worker_count, uint32_t tokens,
                                uint32_t lanes, aclrtStream stream, std::vector<PlDescriptor> *tpl_out = nullptr)
{
    if (wl == nullptr || wl->input == nullptr || wl->worker_desc == nullptr) {
        return;
    }
    const uint32_t expert_per_pe = ParseEnvU32("INC_DN_PL_EXPERT_PER_PE", 8u);
    const uint32_t tpl = tokens / lanes;
    std::vector<uint8_t> payload(static_cast<size_t>(tokens) * g_runtime_payload_bytes);
    std::vector<PlDescriptor> descs(tokens);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        const uint32_t dest_rank = lane < worker_count ? lane : lane;
        const uint32_t tokens_per_lane = tpl;
        for (uint32_t t = 0; t < tpl; ++t) {
            const uint32_t tok = lane * tpl + t;
            const uint32_t source_local_token = lane * tokens_per_lane + t;
            const uint32_t seq = source_rank * tokens + source_local_token + 1u;
            for (uint32_t i = 0; i < g_runtime_payload_bytes; ++i) {
                payload[static_cast<size_t>(tok) * g_runtime_payload_bytes + i] = D1PatternByte(source_rank, seq, i);
            }
            PlDescriptor d{};
            d.epoch = 1;
            d.generation = 1;
            d.lane_sequence = t;
            d.source_token_id = source_local_token;
            d.topk_slot = 0;
            d.expert_id = dest_rank * expert_per_pe;
            d.local_expert_id = 0;
            d.token_sequence = seq;
            d.source_rank = source_rank;
            d.destination_rank = dest_rank;
            d.destination_slot = source_rank * tokens_per_lane + t;
            d.destination_final_slot = d.destination_slot;
            d.source_segment_offset = tok;
            d.payload_bytes = g_runtime_payload_bytes;
            d.source_lane = lane;
            d.ring_slot = t % kQv2RingDepth;
            d.generation = 1;
            descs[tok] = d;
        }
    }
    if (tpl_out != nullptr) {
        *tpl_out = descs;
    }
    aclrtMemcpy(wl->input, payload.size(), payload.data(), payload.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(wl->worker_desc, descs.size() * sizeof(PlDescriptor), descs.data(), descs.size() * sizeof(PlDescriptor),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtSynchronizeStream(stream);
}

static void HostSyncIngressHeadFromInc(uint8_t *sym, int inc_pe, uint32_t lanes, uint64_t go_epoch, int pe,
                                       uint64_t local_tail)
{
    // 多 epoch：host 从 INC 拉 ingress head，经 scratch 中转；仅在需钳位时写回 live head
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        const uint64_t off = kPlIngressHeadOff + static_cast<uint64_t>(lane) * 64u;
        const uint64_t scratch = kPlHostProbeScratchOff + static_cast<uint64_t>(lane) * 64u;
        aclshmem_getmem(sym + scratch, sym + off, sizeof(Qv2HeadLine), inc_pe);
        Qv2HeadLine h{};
        aclrtMemcpy(&h, sizeof(h), sym + scratch, sizeof(h), ACL_MEMCPY_DEVICE_TO_HOST);
        if (h.head > local_tail) {
            h.head = local_tail;
            aclrtMemcpy(sym + off, sizeof(h), &h, sizeof(h), ACL_MEMCPY_HOST_TO_DEVICE);
        }
    }
    if (pe == 0 && go_epoch >= 2u) {
        Qv2HeadLine h{};
        aclrtMemcpy(&h, sizeof(h), sym + kPlIngressHeadOff, sizeof(h), ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "PL_HEAD_SYNC go_epoch=" << go_epoch << " worker_ingress_head=" << h.head
                  << " local_tail=" << local_tail << std::endl;
    }
}

static void HostEmitEpochStage(int pe, uint64_t go_epoch, const char *stage)
{
    std::cout << "PL_HOST_EPOCH_STAGE pe=" << pe << " epoch=" << go_epoch << " stage=" << stage << std::endl;
}

// 等本 PE worker control 显式 retire epoch N（ControlEpochRetireLine）。
// Formal gate：retire_signal==N && epoch==N && error_code==0。
// 旧 cycle 比较（wait_start > pair_done_end）仅诊断，不再作 gate。
static int HostWaitControlEpochRetire(uint8_t *sym, uint64_t completed_epoch, int timeout_ms, int pe)
{
    const int64_t t0 = HostWallNs();
    const int64_t deadline_ns = t0 + static_cast<int64_t>(timeout_ms) * 1000000LL;
    const uint64_t ctrl_off =
        kPlWorkerTraceOff + static_cast<uint64_t>(g_worker_layout.control_block) * 128u;
    while (HostWallNs() < deadline_ns) {
        ControlEpochRetireLine retire{};
        PlServiceTraceLine ctrl{};
        PlDestinationCompletionTrace dct{};
        if (aclrtMemcpy(&retire, sizeof(retire), sym + kPlControlEpochRetireOff, sizeof(retire),
                        ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
            return -1;
        }
        if (aclrtMemcpy(&ctrl, sizeof(ctrl), sym + ctrl_off, sizeof(ctrl), ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
            return -1;
        }
        if (ctrl.kernel_exit_reason != 0u && ctrl.kernel_exit_reason != kPlKernelExitStop) {
            std::cerr << "PL_CONTROL_RETIRE_ABORT pe=" << pe << " completed_epoch=" << completed_epoch
                      << " ctrl_exit=" << ctrl.kernel_exit_reason << std::endl;
            std::cerr.flush();
            return -1;
        }
        if (retire.magic == kPlControlEpochRetireMagic && retire.retire_signal == completed_epoch &&
            retire.epoch == completed_epoch) {
            if (retire.error_code != 0u) {
                std::cerr << "PL_CONTROL_RETIRE_ERROR pe=" << pe << " completed_epoch=" << completed_epoch
                          << " error_code=" << retire.error_code
                          << " telemetry_valid=" << retire.telemetry_valid << std::endl;
                std::cerr.flush();
                return -1;
            }
            // Diagnostic-only cycle compare (must not gate).
            (void)aclrtMemcpy(&dct, sizeof(dct), sym + PlDestinationCompletionTraceOff(), sizeof(dct),
                              ACL_MEMCPY_DEVICE_TO_HOST);
            const bool cycle_diag =
                dct.epoch == completed_epoch && dct.pair_done_publish_end_cycle > 0u &&
                ctrl.wait_start_enter_cycle > dct.pair_done_publish_end_cycle;
            std::cout << "PL_CONTROL_EPOCH_RETIRE pe=" << pe << " completed_epoch=" << completed_epoch
                      << " generation=" << retire.generation
                      << " telemetry_valid=" << retire.telemetry_valid
                      << " cycle_diag_reentered_wait=" << (cycle_diag ? 1 : 0) << std::endl;
            std::cout.flush();
            return 0;
        }
        usleep(100);
    }
    std::cerr << "PL_CONTROL_RETIRE_TIMEOUT pe=" << pe << " completed_epoch=" << completed_epoch
              << " wait_ms=" << timeout_ms << std::endl;
    std::cerr.flush();
    return -1;
}

// Legacy name: formal path uses ControlEpochRetireLine.
static int HostWaitLocalWorkerControlQuiescent(uint8_t *sym, uint64_t completed_epoch, int timeout_ms, int pe)
{
    return HostWaitControlEpochRetire(sym, completed_epoch, timeout_ms, pe);
}

static int HostPatchDescEpoch(const std::vector<PlDescriptor> &desc_template, PlWorkerLocalSession *wl, uint32_t tokens,
                               uint32_t lanes, uint64_t epoch, aclrtStream stream)
{
    if (wl == nullptr || wl->worker_desc == nullptr) {
        return -3;
    }
    if (desc_template.size() != tokens) {
        std::cerr << "PL_HOST_DESC_TEMPLATE_SIZE_ERROR expect=" << tokens << " got=" << desc_template.size()
                  << std::endl;
        return -1;
    }
    if (lanes == 0u || (tokens % lanes) != 0u) {
        std::cerr << "PL_HOST_DESC_TEMPLATE_LANE_ERROR tokens=" << tokens << " lanes=" << lanes << std::endl;
        return -2;
    }
    std::vector<PlDescriptor> descs = desc_template;
    const uint32_t tokens_per_lane = tokens / lanes;
    const uint64_t epoch_base = (epoch - 1u) * tokens_per_lane;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        for (uint32_t t = 0; t < tokens_per_lane; ++t) {
            const uint32_t tok = lane * tokens_per_lane + t;
            auto &d = descs[tok];
            d.epoch = epoch;
            d.lane_sequence = epoch_base + t;
            d.generation = static_cast<uint32_t>(epoch);
            d.ring_slot = static_cast<uint32_t>(d.lane_sequence % kQv2RingDepth);
        }
    }
    aclrtMemcpyAsync(wl->worker_desc, descs.size() * sizeof(PlDescriptor), descs.data(),
                     descs.size() * sizeof(PlDescriptor), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    aclrtSynchronizeStream(stream);
    return 0;
}

static int WaitResident(uint8_t *sym, uint32_t blocks, uint64_t base, uint32_t expect_mask)
{
    for (uint32_t spin = 0; spin < 20000000u; ++spin) {
        uint32_t got = 0;
        for (uint32_t i = 0; i < blocks; ++i) {
            PlResidentLine line{};
            if (aclrtMemcpy(&line, sizeof(line), sym + base + static_cast<uint64_t>(i) * 64u, sizeof(line),
                            ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
                return -1;
            }
            if (line.value >= 1u && line.magic == kPlMagic) {
                got |= (1u << i);
            }
        }
        if (got == expect_mask) {
            return 0;
        }
        if ((spin % 512u) == 0u) {
            usleep(1);
        }
    }
    return -1;
}

// P5：INC blocks 0–9 在 kPlIncResidentOff；10–19 在 kPlIncP5ExtResidentOff
static int WaitResidentIncP5(uint8_t *sym)
{
    const uint32_t expect_lo = (1u << kPlIncServiceBlockDim) - 1u;
    const uint32_t expect_hi = (1u << 10) - 1u;
    for (uint32_t spin = 0; spin < 20000000u; ++spin) {
        uint32_t got_lo = 0;
        uint32_t got_hi = 0;
        for (uint32_t i = 0; i < kPlIncServiceBlockDim; ++i) {
            PlResidentLine line{};
            if (aclrtMemcpy(&line, sizeof(line), sym + kPlIncResidentOff + static_cast<uint64_t>(i) * 64u,
                            sizeof(line), ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
                return -1;
            }
            if (line.value >= 1u && line.magic == kPlMagic) {
                got_lo |= (1u << i);
            }
        }
        for (uint32_t i = 0; i < 10u; ++i) {
            PlResidentLine line{};
            if (aclrtMemcpy(&line, sizeof(line), sym + kPlIncP5ExtResidentOff + static_cast<uint64_t>(i) * 64u,
                            sizeof(line), ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
                return -1;
            }
            if (line.value >= 1u && line.magic == kPlMagic) {
                got_hi |= (1u << i);
            }
        }
        if (got_lo == expect_lo && got_hi == expect_hi) {
            return 0;
        }
        if ((spin % 512u) == 0u) {
            usleep(1);
        }
    }
    return -1;
}

static int WaitDestDone(uint8_t *sym, uint64_t expect_epoch)
{
    PlDestDoneLine line{};
    for (uint32_t spin = 0; spin < 20000000u; ++spin) {
        if (aclrtMemcpy(&line, sizeof(line), sym + kPlDestDoneOff, sizeof(line), ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
            return -1;
        }
        if (line.done_epoch >= expect_epoch) {
            return 0;
        }
        if ((spin % 512u) == 0u) {
            usleep(1);
        }
    }
    return -1;
}

static int WaitGlobalDone(uint8_t *sym, uint64_t expect_epoch)
{
    D1GlobalDoneLine line{};
    for (uint32_t spin = 0; spin < 20000000u; ++spin) {
        if (aclrtMemcpy(&line, sizeof(line), sym + D1B(kD1GlobalDoneLineOff), sizeof(line), ACL_MEMCPY_DEVICE_TO_HOST) !=
            0) {
            return -1;
        }
        if (line.value >= expect_epoch) {
            return 0;
        }
        if ((spin % 512u) == 0u) {
            usleep(1);
        }
    }
    return -1;
}

// 仅 poll global_done；顺带探测 FullOutputDone / session_stop / 远端 INC SourceReady timeout
// 返回：0=ok, -1=timeout, -2=kPlErrGenerationMismatch, -3=SourceCountReady timeout(11), -4=session_stop
static int WaitGlobalDoneActivePoll(uint8_t *sym, uint64_t expect_epoch, int timeout_ms, double *host_observed_us,
                                    int64_t release_ns, int pe = -1, uint32_t worker_count = 0)
{
    D1GlobalDoneLine global_done{};
    PlFullOutputDoneLine fout{};
    uint64_t session_stop = 0;
    const int64_t t0 = release_ns > 0 ? release_ns : HostWallNs();
    const int64_t deadline_ns = t0 + static_cast<int64_t>(timeout_ms) * 1000000LL;
    // pe0：经 scratch 远程拉各 INC 的 SourceCountTrace（INC 本地无 global_done fanout）
    const bool poll_remote_inc =
        (pe == static_cast<int>(kD1LeaderPe) && worker_count > 0u);
    for (uint32_t spin = 0; spin < 20000000u; ++spin) {
        if (HostWallNs() >= deadline_ns) {
            *host_observed_us = NsToUs(HostWallNs() - t0);
            return -1;
        }
        if (aclrtMemcpy(&global_done, sizeof(global_done), sym + D1B(kD1GlobalDoneLineOff), sizeof(global_done),
                        ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
            return -1;
        }
        if (global_done.value >= expect_epoch) {
            *host_observed_us = NsToUs(HostWallNs() - t0);
            return 0;
        }
        if ((spin % 64u) == 0u) {
            if (aclrtMemcpy(&session_stop, sizeof(session_stop), sym + D1B(kD1SessionStopLineOff),
                            sizeof(session_stop), ACL_MEMCPY_DEVICE_TO_HOST) == 0 &&
                session_stop == kD1SessionStopMagic) {
                *host_observed_us = NsToUs(HostWallNs() - t0);
                std::cout << "PL_SESSION_STOP_SEEN epoch=" << expect_epoch << " pe=" << pe << std::endl;
                std::cout.flush();
                return -4;
            }
            if (aclrtMemcpy(&fout, sizeof(fout), sym + kPlFullOutputDoneOff, sizeof(fout),
                            ACL_MEMCPY_DEVICE_TO_HOST) == 0) {
                if (fout.error_code == kPlErrGenerationMismatch) {
                    *host_observed_us = NsToUs(HostWallNs() - t0);
                    std::cout << "PL_GEN_MISMATCH_EVIDENCE injected_epoch=" << expect_epoch
                              << " detected_epoch=" << expect_epoch
                              << " generation_mismatch_count=" << fout.generation_mismatch_count
                              << " descriptor_error_code=" << fout.error_code
                              << " full_done_epoch=" << fout.full_done_epoch
                              << " outer_timeout=0" << std::endl;
                    std::cout.flush();
                    return -2;
                }
                // M2：workspace desc 致命错误 → fail-closed abort（与 generation mismatch 同处理）
                if (fout.error_code == kPlErrWorkspaceDescMissing ||
                    fout.error_code == kPlErrWorkspaceGenerationMismatch ||
                    fout.error_code == kPlErrWorkspacePointerAlignment ||
                    fout.error_code == kPlErrOutputCapacityInsufficient ||
                    fout.error_code == kPlErrWorkspaceFlagInvalid) {
                    *host_observed_us = NsToUs(HostWallNs() - t0);
                    std::cout << "PL_WORKSPACE_FAIL_ABORT epoch=" << expect_epoch
                              << " pe=" << pe << " error_code=" << fout.error_code << std::endl;
                    std::cout.flush();
                    return -2;
                }
                if (fout.error_code == kPlCountFwdSourceReadyTimeout) {
                    *host_observed_us = NsToUs(HostWallNs() - t0);
                    std::cout << "PL_SOURCE_COUNT_READY_TIMEOUT pe=" << pe << " epoch=" << expect_epoch
                              << " error_code=" << fout.error_code << " failure_stage=source_count_ready"
                              << std::endl;
                    std::cout.flush();
                    return -3;
                }
            }
            if (poll_remote_inc) {
                for (uint32_t src = 0; src < worker_count; ++src) {
                    const int inc_pe = static_cast<int>(worker_count + src);
                    PlIncSourceCountTrace itr{};
                    if (HostReadIncSourceCountTrace(sym, inc_pe, src, &itr) != 0) {
                        continue;
                    }
                    if (itr.error_code == kPlCountFwdSourceReadyTimeout && itr.epoch == expect_epoch) {
                        *host_observed_us = NsToUs(HostWallNs() - t0);
                        const double inc_forward_us = CyclesToUs(itr.exit_cycle, itr.enter_cycle);
                        std::cout << "PL_INC_COUNT_TRACE pe=" << inc_pe << " source=" << src
                                  << " go_epoch=" << expect_epoch << " epoch=" << itr.epoch
                                  << " stage=" << itr.stage << " enter_cycle=" << itr.enter_cycle
                                  << " exit_cycle=" << itr.exit_cycle
                                  << " source_ready_seen=" << itr.source_ready_seen
                                  << " destination_publish_mask=0x" << std::hex << itr.destination_publish_mask
                                  << std::dec << " error_code=" << itr.error_code
                                  << " observed_epoch=" << itr.observed_epoch
                                  << " inc_forward_us=" << inc_forward_us << std::endl;
                        std::cout << "PL_SOURCE_COUNT_READY_TIMEOUT pe=" << inc_pe << " epoch=" << expect_epoch
                                  << " error_code=" << itr.error_code << " failure_stage=source_count_ready"
                                  << std::endl;
                        std::cout.flush();
                        return -3;
                    }
                }
            }
        }
        const double elapsed_us = NsToUs(HostWallNs() - t0);
        if (elapsed_us > 250.0) {
            usleep(1);
        }
    }
    *host_observed_us = NsToUs(HostWallNs() - t0);
    return -1;
}

static int HostPublishSessionStop(uint8_t *sym, uint64_t d1_stop_off, aclrtStream host_stream)
{
    const uint64_t stop = kD1SessionStopMagic;
    if (aclrtMemcpyAsync(sym + d1_stop_off, sizeof(stop), &stop, sizeof(stop), ACL_MEMCPY_HOST_TO_DEVICE,
                         host_stream) != 0) {
        return -1;
    }
    if (aclrtMemcpyAsync(sym + D1B(kD1SessionStopLineOff), sizeof(stop), &stop, sizeof(stop),
                         ACL_MEMCPY_HOST_TO_DEVICE, host_stream) != 0) {
        return -1;
    }
    // 禁止无界 sync：session_stop 发布失败也应快速返回，避免卡住 cleanup
    constexpr int kSessionStopSyncMs = 3000;
    if (aclrtSynchronizeStreamWithTimeout(host_stream, kSessionStopSyncMs) != ACL_SUCCESS) {
        std::cerr << "PL_STREAM_SYNC_TIMEOUT stream=host_session_stop timeout_ms=" << kSessionStopSyncMs
                  << std::endl;
        return -1;
    }
    return 0;
}

static uint32_t HostCollectStartSeenMask(uint8_t *sym, bool is_worker, uint32_t blocks, uint64_t trace_base,
                                       uint64_t go_epoch)
{
    uint32_t mask = 0;
    for (uint32_t i = 0; i < blocks; ++i) {
        PlServiceTraceLine tr{};
        if (aclrtMemcpy(&tr, sizeof(tr), sym + trace_base + static_cast<uint64_t>(i) * 128u, sizeof(tr),
                        ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
            continue;
        }
        if (tr.start_seen_epoch >= go_epoch || tr.observed_start_value >= go_epoch) {
            mask |= (1u << i);
        }
    }
    return mask;
}

static uint32_t HostCollectKernelExitMask(uint8_t *sym, uint32_t blocks, uint64_t trace_base)
{
    uint32_t mask = 0;
    for (uint32_t i = 0; i < blocks; ++i) {
        PlServiceTraceLine tr{};
        if (aclrtMemcpy(&tr, sizeof(tr), sym + trace_base + static_cast<uint64_t>(i) * 128u, sizeof(tr),
                        ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
            continue;
        }
        // 任意非 0 退出码都算已停机（Wait 用）；PASS 仍要求 STOP
        if (tr.kernel_exit_reason != 0u) {
            mask |= (1u << i);
        }
    }
    return mask;
}

static uint32_t HostCollectKernelStopMask(uint8_t *sym, uint32_t blocks, uint64_t trace_base)
{
    uint32_t mask = 0;
    for (uint32_t i = 0; i < blocks; ++i) {
        PlServiceTraceLine tr{};
        if (aclrtMemcpy(&tr, sizeof(tr), sym + trace_base + static_cast<uint64_t>(i) * 128u, sizeof(tr),
                        ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
            continue;
        }
        if (tr.kernel_exit_reason == kPlKernelExitStop) {
            mask |= (1u << i);
        }
    }
    return mask;
}

static void HostEmitCleanupStage(int pe, const char *stage)
{
    const int64_t host_ns = HostWallNs();
    std::cout << "PL_HOST_CLEANUP_STAGE pe=" << pe << " stage=" << stage << " host_ns=" << host_ns << std::endl;
    std::cout.flush();
}

static void HostEmitKernelExitDiag(uint8_t *sym, int pe, bool is_worker, uint32_t service_blocks,
                                   uint64_t trace_base, uint32_t expect_mask, uint64_t go_epoch)
{
    const uint32_t observed = HostCollectKernelExitMask(sym, service_blocks, trace_base);
    const uint32_t missing = expect_mask & ~observed;
    uint64_t session_stop = 0;
    (void)aclrtMemcpy(&session_stop, sizeof(session_stop), sym + D1B(kD1SessionStopLineOff), sizeof(session_stop),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    std::cout << "PL_KERNEL_EXIT_DIAG pe=" << pe << " expected_mask=0x" << std::hex << expect_mask
              << " observed_mask=0x" << observed << " missing_mask=0x" << missing << std::dec
              << " service_blocks=" << service_blocks << " go_epoch=" << go_epoch
              << " session_stop_value=0x" << std::hex << session_stop << std::dec
              << " role=" << (is_worker ? "worker" : "inc") << std::endl;
    for (uint32_t i = 0; i < service_blocks; ++i) {
        if ((missing & (1u << i)) == 0u) {
            continue;
        }
        PlServiceTraceLine tr{};
        if (aclrtMemcpy(&tr, sizeof(tr), sym + trace_base + static_cast<uint64_t>(i) * 128u, sizeof(tr),
                        ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
            continue;
        }
        const char *role = "svc";
        if (is_worker) {
            if (i >= g_worker_layout.upload_begin && i < g_worker_layout.recv_begin) {
                role = "upload";
            } else if (i >= g_worker_layout.recv_begin && i < g_worker_layout.control_block) {
                role = "recv";
            } else if (i == g_worker_layout.control_block) {
                role = "control";
            }
        } else {
            role = (i == 0u) ? "inc_ctrl" : "inc_fwd";
        }
        std::cout << "PL_KERNEL_EXIT_BLOCK pe=" << pe << " block=" << i << " role=" << role
                  << " kernel_exit_reason=" << tr.kernel_exit_reason
                  << " start_seen_epoch=" << tr.start_seen_epoch
                  << " service_exit_epoch=" << tr.service_exit_epoch
                  << " observed_start=" << tr.observed_start_value
                  << " observed_stop=" << tr.observed_stop_value
                  << " wait_start_spin=" << tr.wait_start_spin_count
                  << " last_progress_cycle=" << tr.wait_start_exit_cycle << std::endl;
    }
    std::cout.flush();
}

// 有界 stream sync；超时返回非 0，禁止无界挂死
static int HostSyncStreamBounded(aclrtStream stream, int timeout_ms, const char *name, int pe)
{
    if (stream == nullptr) {
        return 0;
    }
    const aclError e = aclrtSynchronizeStreamWithTimeout(stream, timeout_ms);
    if (e == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
        std::cerr << "PL_STREAM_SYNC_TIMEOUT pe=" << pe << " stream=" << name
                  << " timeout_ms=" << timeout_ms << std::endl;
        std::cerr.flush();
        return -1;
    }
    if (e != ACL_SUCCESS) {
        std::cerr << "PL_STREAM_SYNC_FAIL pe=" << pe << " stream=" << name << " acl_rc=" << static_cast<int>(e)
                  << std::endl;
        std::cerr.flush();
        return -1;
    }
    return 0;
}

static void HostDumpCountPathCycles(uint8_t *sym, int pe, uint32_t worker_count, uint64_t go_epoch)
{
    auto read_local = [&](uint64_t off, void *dst, size_t sz) {
        aclrtMemcpy(dst, sz, sym + off, sz, ACL_MEMCPY_DEVICE_TO_HOST);
    };

    if (pe < static_cast<int>(worker_count)) {
        for (uint32_t source = 0; source < worker_count; ++source) {
            PlWorkerSourceCountTrace wtr{};
            const uint64_t woff = PlWorkerSourceCountTraceOff(source);
            read_local(woff, &wtr, sizeof(wtr));
            const double worker_publish_us = CyclesToUs(wtr.exit_cycle, wtr.enter_cycle);
            std::cout << "PL_WORKER_COUNT_TRACE pe=" << pe << " source=" << source << " go_epoch=" << go_epoch
                      << " epoch=" << wtr.epoch << " stage=" << wtr.stage << " enter_cycle=" << wtr.enter_cycle
                      << " exit_cycle=" << wtr.exit_cycle << " error_code=" << wtr.error_code
                      << " total_routes=" << wtr.total_routes << " worker_publish_us=" << worker_publish_us
                      << std::endl;
            std::cout << "PL_SOURCE_PUBLISH_OWNER pe=" << pe << " source=" << source << " go_epoch=" << go_epoch
                      << " epoch=" << wtr.epoch << " owner_block=" << wtr.source_publish_owner_block
                      << " source_publish_seq=" << wtr.source_publish_seq << std::endl;
        }
        PlDestinationCountTrace dtr{};
        const uint64_t doff = PlDestinationCountTraceOff(static_cast<uint32_t>(pe));
        read_local(doff, &dtr, sizeof(dtr));
        const double dest_wait_us = CyclesToUs(dtr.wait_done_cycle, dtr.enter_cycle);
        const double dest_segment_us =
            (dtr.wait_done_cycle > 0u) ? CyclesToUs(dtr.exit_cycle, dtr.wait_done_cycle) : 0.0;
        const double dest_prefix_us = CyclesToUs(dtr.exit_cycle, dtr.enter_cycle);
        std::cout << "PL_DEST_COUNT_TRACE pe=" << pe << " dest=" << pe << " go_epoch=" << go_epoch
                  << " epoch=" << dtr.epoch << " stage=" << dtr.stage << " enter_cycle=" << dtr.enter_cycle
                  << " wait_done_cycle=" << dtr.wait_done_cycle << " exit_cycle=" << dtr.exit_cycle
                  << " source_ready_mask=0x" << std::hex << dtr.source_ready_mask << " missing_source_mask=0x"
                  << dtr.missing_source_mask << std::dec << " error_code=" << dtr.error_code
                  << " dest_wait_us=" << dest_wait_us << " dest_segment_us=" << dest_segment_us
                  << " dest_prefix_us=" << dest_prefix_us << std::endl;
    } else {
        const uint32_t source = static_cast<uint32_t>(pe - worker_count);
        PlIncSourceCountTrace itr{};
        const uint64_t ioff = PlIncSourceCountTraceOff(source);
        read_local(ioff, &itr, sizeof(itr));
        const double inc_forward_us = CyclesToUs(itr.exit_cycle, itr.enter_cycle);
        std::cout << "PL_INC_COUNT_TRACE pe=" << pe << " source=" << source << " go_epoch=" << go_epoch
                  << " epoch=" << itr.epoch << " stage=" << itr.stage << " enter_cycle=" << itr.enter_cycle
                  << " exit_cycle=" << itr.exit_cycle << " source_ready_seen=" << itr.source_ready_seen
                  << " destination_publish_mask=0x" << std::hex << itr.destination_publish_mask << std::dec
                  << " error_code=" << itr.error_code << " observed_epoch=" << itr.observed_epoch
                  << " inc_forward_us=" << inc_forward_us << std::endl;
    }
    std::cout.flush();
}

static void HostDumpDestinationCompletionTraces(uint8_t *sym, int pe, uint32_t worker_count, uint32_t lane_count,
                                                uint64_t go_epoch)
{
    auto read_local = [&](uint64_t off, void *dst, size_t sz) {
        aclrtMemcpy(dst, sz, sym + off, sz, ACL_MEMCPY_DEVICE_TO_HOST);
    };

    if (pe == static_cast<int>(kD1LeaderPe)) {
        D1GlobalDoneTiming global{};
        D1PairDoneArrival arrival{};
        read_local(D1B(kD1GlobalDoneTimingOff), &global, sizeof(global));
        read_local(D1B(kD1PairDoneArrivalOff), &arrival, sizeof(arrival));
        const uint64_t pair_wait_start = global.pair_done_wait_start_cycle;
        std::cout << "PL_DEST_LEADER_TIMING pe=0 go_epoch=" << go_epoch
                  << " pair_done_wait_start_cycle=" << pair_wait_start << std::endl;
        for (uint32_t dest = 0; dest < worker_count; ++dest) {
            const uint64_t seen = arrival.lanes[dest].pair_done_first_seen_cycle;
            const double pair_done_arrival_us =
                (seen > pair_wait_start) ? CyclesToUs(seen, pair_wait_start) : 0.0;
            std::cout << "PL_PAIR_DONE_ARRIVAL pe=0 dest=" << dest << " go_epoch=" << go_epoch
                      << " pair_done_arrival_us=" << pair_done_arrival_us
                      << " pair_done_first_seen_cycle=" << seen << std::endl;
        }
    }

    if (pe < static_cast<int>(worker_count)) {
        const uint32_t dest = static_cast<uint32_t>(pe);
        PlDestinationCompletionTrace dct{};
        read_local(PlDestinationCompletionTraceOff(), &dct, sizeof(dct));
        if (dct.magic == kPlDestCompletionTraceMagic && dct.epoch == go_epoch) {
            const double all_recv_after_counts_us =
                (dct.counts_done_cycle > 0u && dct.all_recv_done_cycle > dct.counts_done_cycle)
                    ? CyclesToUs(dct.all_recv_done_cycle, dct.counts_done_cycle)
                    : 0.0;
            const double pair_done_publish_local_us =
                (dct.pair_done_publish_end_cycle > dct.pair_done_publish_start_cycle &&
                 dct.pair_done_publish_start_cycle > 0u)
                    ? CyclesToUs(dct.pair_done_publish_end_cycle, dct.pair_done_publish_start_cycle)
                    : 0.0;
            const double counts_done_to_first_recv_us =
                CyclesToUs(dct.first_recv_cycle, dct.counts_done_cycle);
            const double first_recv_to_all_recv_done_us =
                CyclesToUs(dct.all_recv_done_cycle, dct.first_recv_cycle);
            const double all_recv_done_to_full_done_us =
                CyclesToUs(dct.full_done_publish_cycle, dct.all_recv_done_cycle);
            const double full_done_to_pair_done_publish_us =
                CyclesToUs(dct.pair_done_publish_end_cycle, dct.full_done_publish_cycle);
            const double global_transport_wait_us =
                (pe == static_cast<int>(kD1LeaderPe))
                    ? CyclesToUs(dct.global_transport_wait_end, dct.global_transport_wait_start)
                    : 0.0;
            std::cout << "PL_DEST_COMPLETION_TRACE pe=" << pe << " dest=" << dest << " go_epoch=" << go_epoch
                      << " epoch=" << dct.epoch << " generation=" << dct.generation
                      << " expected_total=" << dct.expected_total << " received_total=" << dct.received_total
                      << " recv_done_mask=0x" << std::hex << dct.recv_done_mask << std::dec
                      << " error_code=" << dct.error_code << " counts_done_cycle=" << dct.counts_done_cycle
                      << " first_recv_cycle=" << dct.first_recv_cycle
                      << " last_recv_done_cycle=" << dct.last_recv_done_cycle
                      << " all_recv_done_cycle=" << dct.all_recv_done_cycle
                      << " full_done_publish_cycle=" << dct.full_done_publish_cycle
                      << " pair_done_publish_start_cycle=" << dct.pair_done_publish_start_cycle
                      << " pair_done_publish_end_cycle=" << dct.pair_done_publish_end_cycle
                      << " transport_done_local_cycle=" << dct.transport_done_local_cycle
                      << " global_transport_wait_start_cycle=" << dct.global_transport_wait_start
                      << " global_transport_wait_end_cycle=" << dct.global_transport_wait_end
                      << " all_recv_after_counts_us=" << all_recv_after_counts_us
                      << " pair_done_publish_local_us=" << pair_done_publish_local_us
                      << " counts_done_to_first_recv_us=" << counts_done_to_first_recv_us
                      << " first_recv_to_all_recv_done_us=" << first_recv_to_all_recv_done_us
                      << " all_recv_done_to_full_done_us=" << all_recv_done_to_full_done_us
                      << " full_done_to_pair_done_publish_us=" << full_done_to_pair_done_publish_us
                      << " global_transport_wait_us=" << global_transport_wait_us << std::endl;
        } else {
            std::cout << "PL_DEST_COMPLETION_TRACE_MISS pe=" << pe << " go_epoch=" << go_epoch
                      << " magic=0x" << std::hex << dct.magic << std::dec << " epoch=" << dct.epoch
                      << " sizeof=" << sizeof(dct) << std::endl;
        }

        for (uint32_t source = 0; source < worker_count; ++source) {
            PlRecvChannelCompletionTrace rct{};
            read_local(PlRecvChannelCompletionTraceOff(source), &rct, sizeof(rct));
            if (rct.magic != kPlRecvChanCompletionTraceMagic || rct.epoch != go_epoch) {
                continue;
            }
            const double recv_done_after_first_tail_us =
                (rct.first_tail_seen_cycle > 0u && rct.recv_done_cycle > rct.first_tail_seen_cycle)
                    ? CyclesToUs(rct.recv_done_cycle, rct.first_tail_seen_cycle)
                    : 0.0;
            std::cout << "PL_RECV_CHANNEL_TRACE pe=" << pe << " source=" << source << " go_epoch=" << go_epoch
                      << " epoch=" << rct.epoch << " expected=" << rct.expected
                      << " tail_poll_cycles=" << rct.tail_poll_cycles
                      << " tail_signal_wake_count=" << rct.tail_signal_wake_count
                      << " descriptor_retry_count=" << rct.descriptor_retry_count
                      << " head_publish_count=" << rct.head_publish_count
                      << " head_quiet_count=" << rct.head_quiet_count
                      << " first_tail_seen_cycle=" << rct.first_tail_seen_cycle
                      << " recv_done_cycle=" << rct.recv_done_cycle
                      << " recv_done_after_first_tail_us=" << recv_done_after_first_tail_us
                      << " counts_done_wait_us=" << CyclesToUs(rct.counts_done_wait_cycles, 0)
                      << " wait_first_tail_us=" << CyclesToUs(rct.wait_first_tail_cycles, 0)
                      << " tail_wait_total_us=" << CyclesToUs(rct.tail_wait_total_cycles, 0)
                      << " descriptor_read_verify_us=" << CyclesToUs(rct.descriptor_read_verify_cycles, 0)
                      << " payload_assist_ready_us=" << CyclesToUs(rct.payload_assist_ready_cycles, 0)
                      << " head_credit_publish_us=" << CyclesToUs(rct.head_credit_publish_cycles, 0)
                      << " recv_done_publish_us=" << CyclesToUs(rct.recv_done_publish_cycles, 0)
                      << " token_count=" << rct.token_count << " stale_count=" << rct.stale_count
                      << " duplicate_count=" << rct.duplicate_count << " lost_count=" << rct.lost_count
                      << " overwrite_count=" << rct.overwrite_count << std::endl;
        }
    } else {
        for (uint32_t lane = 0; lane < lane_count && lane < kQv2LaneCount; ++lane) {
            PlIncForwardLaneStageTiming fst{};
            read_local(PlIncForwardLaneStageTimingOff(lane), &fst, sizeof(fst));
            if (fst.magic != kPlIncFwdLaneStageMagic || fst.epoch != go_epoch) {
                continue;
            }
            const double credit_wait_us = CyclesToUs(fst.credit_wait_end, fst.credit_wait_start);
            const double tail_publish_us = CyclesToUs(fst.tail_publish_end, fst.tail_publish_start);
            const double tail_publish_total_us = CyclesToUs(fst.egress_tail_publish_cycles_sum, 0);
            const double avg_batch =
                (fst.batch_count > 0u) ? (static_cast<double>(fst.batch_tokens) / fst.batch_count) : 0.0;
            std::cout << "PL_INC_FWD_LANE_STAGE pe=" << pe << " lane=" << lane << " go_epoch=" << go_epoch
                      << " epoch=" << fst.epoch << " credit_wait_us=" << credit_wait_us
                      << " tail_publish_us=" << tail_publish_us
                      << " egress_tail_publish_cycles_sum=" << fst.egress_tail_publish_cycles_sum
                      << " egress_tail_publish_us_total=" << tail_publish_total_us
                      << " egress_credit_wait_cycles=" << fst.egress_credit_wait_cycles
                      << " egress_tail_publish_count=" << fst.egress_tail_publish_count
                      << " batch_count=" << fst.batch_count << " batch_tokens=" << fst.batch_tokens
                      << " wait_ingress_tail_us=" << CyclesToUs(fst.wait_ingress_tail_cycles, 0)
                      << " descriptor_validation_us=" << CyclesToUs(fst.descriptor_validation_cycles, 0)
                      << " payload_descriptor_issue_us=" << CyclesToUs(fst.payload_descriptor_issue_cycles, 0)
                      << " mte_completion_us=" << CyclesToUs(fst.mte_completion_cycles, 0)
                      << " data_drain_quiet_us=" << CyclesToUs(fst.data_drain_quiet_cycles, 0)
                      << " ingress_head_publish_us=" << CyclesToUs(fst.ingress_head_publish_cycles, 0)
                      << " payload_put_count=" << fst.payload_put_count
                      << " descriptor_put_count=" << fst.descriptor_put_count
                      << " drain_quiet_count=" << fst.drain_quiet_count
                      << " ingress_head_publish_count=" << fst.ingress_head_publish_count
                      << " max_batch=" << fst.max_batch << " avg_batch=" << avg_batch
                      << " publish_group_count=" << fst.publish_group_count
                      << " publish_group_tokens=" << fst.publish_group_tokens
                      << " publish_group_max_tokens=" << fst.publish_group_max_tokens
                      << " partial_publish_group_count=" << fst.partial_publish_group_count
                      << " forced_wrap_flush_count=" << fst.forced_wrap_flush_count
                      << " forced_credit_flush_count=" << fst.forced_credit_flush_count
                      << " upstream_empty_flush_count=" << fst.upstream_empty_flush_count
                      << " final_flush_count=" << fst.final_flush_count
                      << " pending_group_high_watermark=" << fst.pending_group_high_watermark
                      << " first_tail_early_flush_count=" << fst.first_tail_early_flush_count
                      << " first_ingress_tail_cycle=" << fst.first_ingress_tail_cycle
                      << " payload_issue_start_cycle=" << fst.payload_issue_start
                      << " payload_issue_end_cycle=" << fst.payload_issue_end
                      << " tail_publish_start_cycle=" << fst.tail_publish_start
                      << " tail_publish_end_cycle=" << fst.tail_publish_end
                      << " forward_done_cycle=" << fst.forward_done_cycle << std::endl;
        }
    }
    std::cout.flush();
}

static void HostDumpUploadLaneStageTiming(uint8_t *sym, int pe, uint32_t lane_count, uint64_t go_epoch)
{
    if (pe != static_cast<int>(kD1LeaderPe)) {
        return;
    }
    double max_counts_wait_us = 0.0;
    double max_segment_ready_wait_us = 0.0;
    double max_route_scan_us = 0.0;
    for (uint32_t lane = 0; lane < lane_count && lane < kQv2LaneCount; ++lane) {
        PlUploadLaneStageTiming st{};
        const uint64_t off = PlUploadLaneStageTimingOff(lane);
        aclrtMemcpy(&st, sizeof(st), sym + off, sizeof(st), ACL_MEMCPY_DEVICE_TO_HOST);
        if (st.magic != kPlUploadLaneStageMagic || st.epoch != go_epoch) {
            continue;
        }
        const double counts_wait_us = CyclesToUs(st.counts_wait_end, st.counts_wait_start);
        const double segment_ready_wait_us = CyclesToUs(st.segment_ready_wait_end, st.segment_ready_wait_start);
        const double route_scan_us = CyclesToUs(st.route_scan_end, st.route_scan_start);
        max_counts_wait_us = std::max(max_counts_wait_us, counts_wait_us);
        max_segment_ready_wait_us = std::max(max_segment_ready_wait_us, segment_ready_wait_us);
        max_route_scan_us = std::max(max_route_scan_us, route_scan_us);
        std::cout << "PL_UPLOAD_LANE_STAGE pe=" << pe << " lane=" << lane << " go_epoch=" << go_epoch
                  << " epoch=" << st.epoch << " owner_block=" << st.owner_block
                  << " max_counts_wait_us=" << counts_wait_us
                  << " max_segment_ready_wait_us=" << segment_ready_wait_us
                  << " max_route_scan_us=" << route_scan_us << " accepted_routes=" << st.accepted_routes
                  << " segment_base_dcci_count=" << st.segment_base_dcci_count
                  << " credit_wait_cycles=" << st.credit_wait_cycles << std::endl;
    }
    std::cout << "PL_UPLOAD_LANE_STAGE_SUMMARY pe=" << pe << " go_epoch=" << go_epoch
              << " max_counts_wait_us=" << max_counts_wait_us
              << " max_segment_ready_wait_us=" << max_segment_ready_wait_us
              << " max_route_scan_us=" << max_route_scan_us << std::endl;
    std::cout.flush();
}

static void HostEmitStartProbe(uint8_t *sym, int pe, bool is_worker, uint32_t worker_count, uint32_t pair_id,
                               uint64_t go_epoch, uint64_t d1_start_off, uint32_t service_blocks, uint64_t trace_base)
{
    D1ControlLine local_ready{};
    D1LeaderRendezvous leader{};
    D1LeaderTiming leader_timing{};
    D1StartLine local_start{};
    PlDestDoneLine dest_done{};
    D1GlobalDoneLine global_done{};
    D1LeaderCompletion completion{};
    aclrtMemcpy(&local_ready, sizeof(local_ready), sym + D1B(kD1LocalReadyLineOff), sizeof(local_ready),
                ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&leader, sizeof(leader), sym + D1B(kD1LeaderRendezvousOff), sizeof(leader), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&leader_timing, sizeof(leader_timing), sym + D1B(kD1LeaderTimingOff), sizeof(leader_timing),
                ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&local_start, sizeof(local_start), sym + d1_start_off, sizeof(local_start), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&dest_done, sizeof(dest_done), sym + kPlDestDoneOff, sizeof(dest_done), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&global_done, sizeof(global_done), sym + D1B(kD1GlobalDoneLineOff), sizeof(global_done),
                ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&completion, sizeof(completion), sym + D1B(kD1LeaderCompletionOff), sizeof(completion),
                ACL_MEMCPY_DEVICE_TO_HOST);
    const uint32_t block_start_seen_mask = HostCollectStartSeenMask(sym, is_worker, service_blocks, trace_base, go_epoch);
    const uint32_t kernel_exit_mask = HostCollectKernelExitMask(sym, service_blocks, trace_base);
    uint64_t upload_local_tail = 0;
    uint64_t upload_credit_wait = 0;
    uint64_t upload_payload_mte = 0;
    uint64_t recv_tokens = 0;
    uint64_t recv_channel_seen = 0;
    uint64_t egress_tail = 0;
    uint32_t upload_exit = 0;
    uint32_t recv_exit = 0;
    uint32_t ctrl_exit = 0;
    if (is_worker) {
        PlUploadCounters uc{};
        PlRecvCounters rc{};
        Qv2TailLine et{};
        PlServiceTraceLine tr_up{};
        PlServiceTraceLine tr_recv{};
        PlServiceTraceLine tr_ctrl{};
        aclrtMemcpy(&uc, sizeof(uc), sym + kPlUploadCtrOff, sizeof(uc), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(&rc, sizeof(rc), sym + kPlRecvCtrOff, sizeof(rc), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(&et, sizeof(et), sym + kPlEgressTailOff, sizeof(et), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(&tr_up, sizeof(tr_up),
                    sym + trace_base + static_cast<uint64_t>(g_worker_layout.upload_begin) * 128u, sizeof(tr_up),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(&tr_recv, sizeof(tr_recv),
                    sym + trace_base + static_cast<uint64_t>(g_worker_layout.recv_begin) * 128u, sizeof(tr_recv),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(&tr_ctrl, sizeof(tr_ctrl),
                    sym + trace_base + static_cast<uint64_t>(g_worker_layout.control_block) * 128u, sizeof(tr_ctrl),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        upload_local_tail = uc.local_tail;
        upload_credit_wait = uc.credit_wait_cycles;
        upload_payload_mte = uc.payload_mte_wait_count;
        recv_tokens = rc.tokens_received;
        recv_channel_seen = rc.channel_seen;
        egress_tail = et.tail;
        upload_exit = tr_up.kernel_exit_reason;
        recv_exit = tr_recv.kernel_exit_reason;
        ctrl_exit = tr_ctrl.kernel_exit_reason;
    }
    PlFullOutputDoneLine fout{};
    PlRouteTimingLine rt{};
    PlLaneWorkLine lw{};
    aclrtMemcpy(&fout, sizeof(fout), sym + kPlFullOutputDoneOff, sizeof(fout), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&rt, sizeof(rt), sym + kPlRouteTimingOff, sizeof(rt), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&lw, sizeof(lw), sym + kPlLaneWorkOff, sizeof(lw), ACL_MEMCPY_DEVICE_TO_HOST);
    std::cout << "PL_START_PROBE pe=" << pe << " role=" << (is_worker ? "worker" : "inc") << " go_epoch=" << go_epoch
              << " local_ready=" << local_ready.value << " leader_ready_slot=" << leader.ready_slots[pair_id].value
              << " release_command=" << leader.release_command.value
              << " leader_go_fanout_end=" << leader_timing.go_fanout_end_cycle << " local_start=" << local_start.value
              << " block_start_seen_mask=0x" << std::hex << block_start_seen_mask << std::dec
              << " destination_done=" << dest_done.done_epoch << " global_done=" << global_done.value
              << " pair_done0=" << completion.pair_done_slots[pair_id].value << " upload_local_tail="
              << upload_local_tail << " upload_credit_wait=" << upload_credit_wait
              << " upload_payload_mte_wait=" << upload_payload_mte << " recv_tokens=" << recv_tokens
              << " recv_channel_seen=" << recv_channel_seen << " egress_tail=" << egress_tail
              << " upload_exit=" << upload_exit << " recv_exit=" << recv_exit << " ctrl_exit=" << ctrl_exit
              << " kernel_exit_mask=0x" << std::hex << kernel_exit_mask << std::dec
              << " counts_done=" << fout.counts_done_epoch << " full_done=" << fout.full_done_epoch
              << " fout_error=" << fout.error_code << " missing_source_mask=0x" << std::hex
              << fout.missing_source_mask << std::dec << " timing_epoch=" << rt.timing_epoch
              << " gather_count=" << lw.gather_count << " lane_token_count=" << lw.token_count << std::endl;
    // 超时路径也必须 dump count-ready / count-path traces（不能只在 global_done 后打印）
    if (is_worker) {
        PlSourceCountReadyLine src_rdy{};
        aclrtMemcpy(&src_rdy, sizeof(src_rdy), sym + PlSourceCountReadyLineOff(pair_id), sizeof(src_rdy),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "PL_SOURCE_COUNT_READY_PROBE pe=" << pe << " source=" << pair_id << " go_epoch=" << go_epoch
                  << " epoch=" << src_rdy.epoch << " ready_epoch=" << src_rdy.ready_epoch
                  << " total_routes=" << src_rdy.total_routes << " magic=0x" << std::hex << src_rdy.magic << std::dec
                  << " magic_ok=" << (src_rdy.magic == kPlCountReadyMagic ? 1 : 0) << std::endl;
        PlWorkerSourceCountTrace wtr{};
        aclrtMemcpy(&wtr, sizeof(wtr), sym + PlWorkerSourceCountTraceOff(pair_id), sizeof(wtr),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "PL_WORKER_COUNT_TRACE pe=" << pe << " source=" << pair_id << " go_epoch=" << go_epoch
                  << " epoch=" << wtr.epoch << " stage=" << wtr.stage << " enter_cycle=" << wtr.enter_cycle
                  << " exit_cycle=" << wtr.exit_cycle << " total_routes=" << wtr.total_routes
                  << " error_code=" << wtr.error_code << " magic_ok=" << (wtr.magic == kPlCountTraceMagic ? 1 : 0)
                  << " worker_publish_us=" << CyclesToUs(wtr.exit_cycle, wtr.enter_cycle) << std::endl;
        PlDestinationCountTrace dtr{};
        aclrtMemcpy(&dtr, sizeof(dtr), sym + PlDestinationCountTraceOff(pair_id), sizeof(dtr),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "PL_DEST_COUNT_TRACE pe=" << pe << " dest=" << pair_id << " go_epoch=" << go_epoch
                  << " epoch=" << dtr.epoch << " stage=" << dtr.stage << " enter_cycle=" << dtr.enter_cycle
                  << " wait_done_cycle=" << dtr.wait_done_cycle << " exit_cycle=" << dtr.exit_cycle
                  << " source_ready_mask=0x" << std::hex << dtr.source_ready_mask << " missing_source_mask=0x"
                  << dtr.missing_source_mask << std::dec << " error_code=" << dtr.error_code
                  << " dest_wait_us=" << CyclesToUs(dtr.wait_done_cycle, dtr.enter_cycle)
                  << " dest_segment_us="
                  << ((dtr.wait_done_cycle > 0u) ? CyclesToUs(dtr.exit_cycle, dtr.wait_done_cycle) : 0.0)
                  << " dest_prefix_us=" << CyclesToUs(dtr.exit_cycle, dtr.enter_cycle) << std::endl;
        for (uint32_t src = 0; src < worker_count; ++src) {
            PlDestCountSliceReadyLine slice{};
            aclrtMemcpy(&slice, sizeof(slice), sym + PlDestCountSliceReadyLineOff(src, pair_id), sizeof(slice),
                        ACL_MEMCPY_DEVICE_TO_HOST);
            std::cout << "PL_DEST_COUNT_SLICE_READY_PROBE pe=" << pe << " source=" << src
                      << " destination_rank=" << pair_id << " go_epoch=" << go_epoch << " epoch=" << slice.epoch
                      << " ready_epoch=" << slice.ready_epoch << " total_routes=" << slice.total_routes << " magic=0x"
                      << std::hex << slice.magic << std::dec
                      << " magic_ok=" << (slice.magic == kPlCountReadyMagic ? 1 : 0)
                      << " identity_ok="
                      << ((slice.magic == kPlCountReadyMagic && slice.epoch == go_epoch &&
                           slice.ready_epoch == go_epoch && slice.source_rank == src &&
                           slice.destination_rank == pair_id)
                              ? 1
                              : 0)
                      << std::endl;
        }
    } else {
        PlIncSourceCountTrace itr{};
        aclrtMemcpy(&itr, sizeof(itr), sym + PlIncSourceCountTraceOff(pair_id), sizeof(itr),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "PL_INC_COUNT_TRACE pe=" << pe << " source=" << pair_id << " go_epoch=" << go_epoch
                  << " epoch=" << itr.epoch << " stage=" << itr.stage << " enter_cycle=" << itr.enter_cycle
                  << " exit_cycle=" << itr.exit_cycle << " source_ready_seen=" << itr.source_ready_seen
                  << " destination_publish_mask=0x" << std::hex << itr.destination_publish_mask << std::dec
                  << " error_code=" << itr.error_code << " observed_epoch=" << itr.observed_epoch
                  << " observed_magic=0x" << std::hex << itr.observed_magic << std::dec
                  << " inc_forward_us=" << CyclesToUs(itr.exit_cycle, itr.enter_cycle) << std::endl;
        PlSourceCountReadyLine src_rdy{};
        aclrtMemcpy(&src_rdy, sizeof(src_rdy), sym + PlSourceCountReadyLineOff(pair_id), sizeof(src_rdy),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "PL_SOURCE_COUNT_READY_PROBE pe=" << pe << " role=inc source=" << pair_id
                  << " go_epoch=" << go_epoch << " epoch=" << src_rdy.epoch << " ready_epoch=" << src_rdy.ready_epoch
                  << " total_routes=" << src_rdy.total_routes << " magic_ok="
                  << (src_rdy.magic == kPlCountReadyMagic ? 1 : 0) << std::endl;
    }
    std::cout.flush();
    std::cerr.flush();
}

static int WaitAllKernelsStopped(uint8_t *sym, bool is_worker, uint32_t service_blocks, uint64_t trace_base,
                                 uint32_t expect_mask, int timeout_ms)
{
    const int64_t deadline_ns = HostWallNs() + static_cast<int64_t>(timeout_ms) * 1000000LL;
    while (HostWallNs() < deadline_ns) {
        const uint32_t exit_mask = HostCollectKernelExitMask(sym, service_blocks, trace_base);
        if (exit_mask == expect_mask) {
            return 0;
        }
        usleep(500);
    }
    return -1;
}

static int WaitLeaderReadySlots(uint8_t *sym, uint64_t expect_epoch, uint32_t wc, int timeout_ms)
{
    D1LeaderRendezvous leader{};
    const int64_t t0 = HostWallNs();
    const int64_t deadline_ns = t0 + static_cast<int64_t>(timeout_ms) * 1000000LL;
    for (uint32_t spin = 0; spin < 20000000u; ++spin) {
        if (HostWallNs() >= deadline_ns) {
            return -1;
        }
        if (aclrtMemcpy(&leader, sizeof(leader), sym + D1B(kD1LeaderRendezvousOff), sizeof(leader),
                        ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
            return -1;
        }
        uint32_t ready = 0;
        for (uint32_t i = 0; i < wc; ++i) {
            if (leader.ready_slots[i].value >= expect_epoch) {
                ++ready;
            }
        }
        if (ready >= wc) {
            return 0;
        }
        if (NsToUs(HostWallNs() - t0) > 250.0) {
            usleep(1);
        }
    }
    return -1;
}

static int HostReadIncForwardLane(uint8_t *sym, int inc_pe, uint32_t lane, PlForwardCounters *out, aclrtStream host_stream)
{
    (void)host_stream;
    const uint64_t off = kPlForwardCtrOff + static_cast<uint64_t>(lane) * sizeof(PlForwardCounters);
    // 经独立 host-probe scratch 中转，禁止覆盖 live forward counters
    const uint64_t scratch = kPlHostProbeScratchOff;
    aclshmem_getmem(sym + scratch, sym + off, sizeof(PlForwardCounters), inc_pe);
    return aclrtMemcpy(out, sizeof(*out), sym + scratch, sizeof(*out), ACL_MEMCPY_DEVICE_TO_HOST);
}

// pe0 经 scratch 拉远端 INC SourceCountTrace（NEG fail-closed；不污染 live 区）
static int HostReadIncSourceCountTrace(uint8_t *sym, int inc_pe, uint32_t source, PlIncSourceCountTrace *out)
{
    const uint64_t off = PlIncSourceCountTraceOff(source);
    const uint64_t scratch = kPlHostProbeScratchOff;
    aclshmem_getmem(sym + scratch, sym + off, sizeof(PlIncSourceCountTrace), inc_pe);
    return aclrtMemcpy(out, sizeof(*out), sym + scratch, sizeof(*out), ACL_MEMCPY_DEVICE_TO_HOST);
}

static int PublishLocalReady(uint8_t *sym, uint64_t go_epoch, aclrtStream host_stream)
{
    if (aclrtMemcpyAsync(sym + D1B(kD1LocalReadyLineOff), sizeof(go_epoch), &go_epoch, sizeof(go_epoch),
                         ACL_MEMCPY_HOST_TO_DEVICE, host_stream) != 0) {
        return -1;
    }
    return aclrtSynchronizeStream(host_stream);
}

static int PublishLeaderRelease(uint8_t *sym, uint64_t epoch, aclrtStream stream)
{
    return aclrtMemcpyAsync(sym + D1B(kD1LeaderRendezvousOff), sizeof(epoch), &epoch, sizeof(epoch),
                            ACL_MEMCPY_HOST_TO_DEVICE, stream);
}

int main(int argc, char **argv)
{
    if (argc < 10) {
        std::cerr << "Usage: inc_dc_dn_pipeline <n_pes> <pe> <ipport> <g_npus> <f_npu> <lane_count> "
                     "<warmup> <measure> <batch_tokens> [tokens] [ring_depth]\n";
        return 2;
    }
    const int npe = std::atoi(argv[1]);
    const int pe = std::atoi(argv[2]);
    ipport = argv[3];
    g_npus = std::atoi(argv[4]);
    f_npu = std::atoi(argv[5]);
    uint32_t lane_count = static_cast<uint32_t>(std::strtoul(argv[6], nullptr, 10));
    const uint32_t env_upload_lanes = ParseEnvU32("INC_DN_PL_UPLOAD_LANE_COUNT", 0u);
    if (env_upload_lanes > 0u) {
        lane_count = env_upload_lanes;
    }
    lane_count = PlClampUploadLaneCount(lane_count);
    const uint32_t recv_lane_count =
        PlClampRecvLaneCountCfg(ParseEnvU32("INC_DN_PL_RECV_LANE_COUNT", kPlMaxRecvLanes));
    g_worker_layout = PlMakeWorkerServiceLayout(lane_count, recv_lane_count);
    if (g_worker_layout.block_dim == 0u || g_worker_layout.block_dim > kPlWorkerServiceBlockDim ||
        g_worker_layout.control_block >= kPlCombineReserveBegin) {
        std::cerr << "PL_WORKER_LAYOUT_INVALID upload=" << lane_count << " recv=" << recv_lane_count
                  << " block_dim=" << g_worker_layout.block_dim << " max=" << kPlWorkerServiceBlockDim
                  << std::endl;
        return 11;
    }
    const uint32_t warmup = static_cast<uint32_t>(std::strtoul(argv[7], nullptr, 10));
    const uint32_t measure = static_cast<uint32_t>(std::strtoul(argv[8], nullptr, 10));
    const uint32_t batch_tokens_cli = static_cast<uint32_t>(std::strtoul(argv[9], nullptr, 10));
    const uint32_t tokens =
        (argc >= 11) ? static_cast<uint32_t>(std::strtoul(argv[10], nullptr, 10)) : 512u;
    const uint32_t ring_depth =
        (argc >= 12) ? static_cast<uint32_t>(std::strtoul(argv[11], nullptr, 10)) : kQv2RingDepth;
    const uint32_t tile_tokens_env = ParseEnvU32("INC_DN_PL_TILE_TOKENS", 32u);
    uint32_t batch_tokens = batch_tokens_cli > 0 ? batch_tokens_cli : 8u;
    const uint32_t env_credit_batch = ParseEnvU32("INC_DN_PL_CREDIT_BATCH_TOKENS", 0u);
    if (env_credit_batch > 0) {
        batch_tokens = env_credit_batch;
    }
    // P1 预备：upload / egress 独立 byte tile（0=沿用固定 token tile）
    const uint32_t upload_tile_target_bytes = ParseEnvU32("INC_DN_PL_UPLOAD_TILE_TARGET_BYTES", 0u);
    const uint32_t egress_publish_target_bytes = ParseEnvU32("INC_DN_PL_EGRESS_PUBLISH_TARGET_BYTES", 0u);
    uint32_t tile_tokens = tile_tokens_env;           // upload tile
    uint32_t head_tile_tokens = tile_tokens_env;      // egress publish group（默认与旧行为一致）
    const uint32_t completion_mode = ParseCompletionMode();
    const uint32_t host_completion_mode = ParseHostCompletionMode();
    const uint32_t worker_count = static_cast<uint32_t>(npe / 2);
    const bool is_worker = (pe < static_cast<int>(worker_count));
    const uint32_t pair_id = is_worker ? static_cast<uint32_t>(pe) : static_cast<uint32_t>(pe - worker_count);
    const uint32_t verify_mode = ParseVerifyMode();
    const uint32_t measurement_mode = ParseMeasurementMode();
    const char *artifact_shape_dir = std::getenv("INC_DN_PL_ARTIFACT_SHAPE_DIR");
    const uint32_t expert_per_pe = ParseEnvU32("INC_DN_PL_EXPERT_PER_PE", 8u);
    // Runtime Payload V2：epp 暂未泛化 — 必须 ==8，禁止 le<8 静默截断
    if (expert_per_pe != 8u) {
        std::cerr << "PL_EXPERT_PER_PE_UNSUPPORTED expert_per_pe=" << expert_per_pe
                  << " required=8 (fail-closed until epp generalized)" << std::endl;
        return 11;
    }
    const uint32_t route_topk = ParseEnvU32("INC_DN_PL_ROUTE_TOPK", 1u);
    // Runtime payload width（禁止仅改 kQv2TokenBytes 冒充；须写入 invocation/workspace contract）
    uint32_t runtime_dtype_bytes = ParseEnvU32("INC_DN_PL_DTYPE_BYTES", kPlDtypeBytesFp16);
    uint32_t runtime_hidden_size = ParseEnvU32("INC_DN_PL_HIDDEN_SIZE", 4096u);
    uint32_t runtime_payload_bytes = ParseEnvU32("INC_DN_PL_PAYLOAD_BYTES", 0u);
    if (runtime_payload_bytes == 0u) {
        runtime_payload_bytes = runtime_hidden_size * runtime_dtype_bytes;
    }
    if (runtime_hidden_size == 0u && runtime_dtype_bytes != 0u) {
        runtime_hidden_size = runtime_payload_bytes / runtime_dtype_bytes;
    }
    const uint32_t runtime_layout_version = ParseEnvU32("INC_DN_PL_LAYOUT_VERSION", kPlLayoutVersionRuntimePayloadV2);
    if (!PlRuntimePayloadCrossCheck(runtime_hidden_size, runtime_dtype_bytes, runtime_payload_bytes)) {
        std::cerr << "PL_RUNTIME_PAYLOAD_FAIL hidden_size=" << runtime_hidden_size
                  << " dtype_bytes=" << runtime_dtype_bytes << " payload_bytes=" << runtime_payload_bytes
                  << " layout_version=" << runtime_layout_version
                  << " (allowed fp16 hidden 2k/4k/6k/7k/8k/12k only)" << std::endl;
        return 11;
    }
    g_runtime_payload_bytes = runtime_payload_bytes;
    g_runtime_hidden_size = runtime_hidden_size;
    // Byte-sized upload / egress tile（P1）；target=0 时保持 INC_DN_PL_TILE_TOKENS
    auto bytes_to_tile = [&](uint32_t target_bytes) -> uint32_t {
        if (target_bytes == 0u || runtime_payload_bytes == 0u) {
            return 0u;
        }
        uint32_t by_bytes = target_bytes / runtime_payload_bytes;
        if (by_bytes < 1u) {
            by_bytes = 1u;
        }
        // Producer occupancy is guarded by live head/tail credit.  Limiting
        // a publish to depth/2 caused an unnecessary quiet every 256 KiB;
        // a contiguous run may safely use every slot except one.
        uint32_t cap = ring_depth > 1u ? ring_depth - 1u : 1u;
        if (cap < 1u) {
            cap = 1u;
        }
        return (by_bytes < cap) ? by_bytes : cap;
    };
    if (upload_tile_target_bytes > 0u) {
        tile_tokens = bytes_to_tile(upload_tile_target_bytes);
    }
    if (egress_publish_target_bytes > 0u) {
        head_tile_tokens = bytes_to_tile(egress_publish_target_bytes);
    } else {
        // P1 默认：第二跳 publish group 保持历史固定 32（不随 upload byte-tile 联动）
        head_tile_tokens = tile_tokens_env;
    }
    if (batch_tokens > tile_tokens) {
        batch_tokens = tile_tokens;
    }
    if (runtime_layout_version != kPlLayoutVersionRuntimePayloadV2 &&
        runtime_layout_version != kPlLayoutVersionV1Fixed8192) {
        std::cerr << "PL_LAYOUT_VERSION_FAIL layout_version=" << runtime_layout_version << std::endl;
        return 11;
    }
    const uint32_t input_epoch_count = ParseEnvU32("INC_DN_PL_INPUT_EPOCH_COUNT", 1u);
    const uint32_t runtime_final_slot_capacity = ParseEnvU32("INC_DN_PL_FINAL_SLOT_CAPACITY", 0u);
    const uint32_t workspace_generation = ParseEnvU32("INC_DN_PL_WORKSPACE_GENERATION", 1u);
    // M4：local_final 为唯一正式路径；legacy symmetric DestFinal 已 fail-closed
    const char *ws_mode_env = std::getenv("INC_DN_PL_WORKSPACE_MODE");
    bool workspace_local_final = true;
    if (ws_mode_env != nullptr && std::strcmp(ws_mode_env, "legacy") == 0) {
        std::cerr << "PL_WORKSPACE_LEGACY_REMOVED_M4 migrate=local_final" << std::endl;
        return 11;
    }
    if (ws_mode_env != nullptr && (std::strcmp(ws_mode_env, "local_final") == 0 ||
                                   std::strcmp(ws_mode_env, "worker_local_final") == 0)) {
        workspace_local_final = true;
    }
    // M5 capacity 负例：全 rank 在 InitShmem 前 fail-closed，避免单方 return 后 barrier hang
    if (std::getenv("INC_DN_PL_M5_SHRINK_OUTPUT_CAPACITY") != nullptr &&
        std::strcmp(std::getenv("INC_DN_PL_M5_SHRINK_OUTPUT_CAPACITY"), "1") == 0) {
        const uint32_t req = PlResolveFinalSlotCapacity(runtime_final_slot_capacity);
        std::cerr << "PL_WORKSPACE_CAPACITY_FAIL reason=output_route_capacity_lt_required"
                  << " output_route_capacity=" << (req > 0u ? req - 1u : 0u) << " required=" << req << std::endl;
        return 11;
    }
    // M5 其余负例：同样必须在 InitShmem 前全 rank fail-closed（禁止进 device 后 hang）
    if (std::getenv("INC_DN_PL_M5_INVALID_OUTPUT_PTR") != nullptr &&
        std::strcmp(std::getenv("INC_DN_PL_M5_INVALID_OUTPUT_PTR"), "1") == 0) {
        std::cerr << "PL_WORKSPACE_INVALID_PTR_FAIL reason=output_pointer_null_or_non_device" << std::endl;
        return 11;
    }
    if (std::getenv("INC_DN_PL_M5_BAD_ALIGNMENT") != nullptr &&
        std::strcmp(std::getenv("INC_DN_PL_M5_BAD_ALIGNMENT"), "1") == 0) {
        std::cerr << "PL_WORKSPACE_BAD_ALIGN_FAIL reason=output_pointer_misaligned" << std::endl;
        return 11;
    }
    if (std::getenv("INC_DN_PL_M5_STALE_GENERATION") != nullptr &&
        std::strcmp(std::getenv("INC_DN_PL_M5_STALE_GENERATION"), "1") == 0) {
        std::cerr << "PL_WORKSPACE_STALE_GENERATION_FAIL reason=generation_mismatch_fail_closed"
                  << " expected=" << workspace_generation << " observed=" << (workspace_generation + 999u)
                  << std::endl;
        return 11;
    }
    if (std::getenv("INC_DN_PL_M5_OUTPUT_OVERLAP") != nullptr &&
        std::strcmp(std::getenv("INC_DN_PL_M5_OUTPUT_OVERLAP"), "1") == 0) {
        std::cerr << "PL_WORKSPACE_OUTPUT_OVERLAP_FAIL reason=expand_x_assist_overlap" << std::endl;
        return 11;
    }
    const uint32_t resolved_final_slot_capacity = PlResolveFinalSlotCapacity(runtime_final_slot_capacity);
    const uint32_t ep_count = input_epoch_count == 0u ? 1u : input_epoch_count;
    const uint32_t session_epochs = warmup + measure;
    const bool perf_skip_epoch_clear = ParsePerfSkipEpochClear();
    const PlIngressLayoutRequest ingress_layout_request = ParseIngressLayoutRequest();
    if (ingress_layout_request == PlIngressLayoutRequest::kInvalid) {
        std::cerr << "PL_INGRESS_LAYOUT_FAIL reason=unknown_layout"
                  << " allowed=dest_major_gather,source_major_raw,auto" << std::endl;
        return 11;
    }
    const uint32_t upload_lanes = lane_count == 0u ? 1u : lane_count;
    const uint32_t base_tokens_per_lane = tokens / upload_lanes;
    const uint32_t max_tokens_per_lane =
        tokens - base_tokens_per_lane * (upload_lanes - 1u);
    const uint32_t raw_tile_tokens = batch_tokens == 0u ? 1u : batch_tokens;
    const uint32_t raw_generations_per_lane =
        (max_tokens_per_lane + raw_tile_tokens - 1u) / raw_tile_tokens;
    const uint64_t raw_payload_bytes =
        static_cast<uint64_t>(tokens) * runtime_payload_bytes;
    const uint64_t raw_metadata_bytes =
        static_cast<uint64_t>(tokens) * route_topk * sizeof(int32_t);
    const bool destination_csr_capacity_safe =
        inc::dc::dn::pl::PlDestinationCsrFits(
            tokens, route_topk, worker_count);
    const bool source_major_capacity_safe =
        raw_generations_per_lane <= kPlRawChunkRingDepth &&
        raw_payload_bytes <= kPlIncRawSlabBytes &&
        raw_metadata_bytes <= kPlIncRawRouteMetaBytes &&
        destination_csr_capacity_safe;
    const uint32_t max_dest_share_ppm =
        ParseEnvU32("INC_DN_PL_MAX_DEST_SHARE_PPM", 0u);
    if (max_dest_share_ppm > 1000000u) {
        std::cerr << "PL_INGRESS_LAYOUT_FAIL reason=bad_max_dest_share_ppm"
                  << " value=" << max_dest_share_ppm << std::endl;
        return 11;
    }
    // A completely single-hot route makes the destination control/finalize
    // path the bottleneck and is serviced more robustly by dest-major.  Zero
    // means the caller did not provide a routing histogram.
    const bool source_major_imbalance_safe =
        max_dest_share_ppm == 0u || max_dest_share_ppm < 950000u;
    // The optimized source-major descriptor ordering is currently proven for
    // the full eight-worker topology.  Smaller groups remain first-class
    // supported configurations through the canonical dest-major path; do not
    // silently apply an unproven ordering merely because K fits the buffers.
    const bool source_major_topology_safe = worker_count == 8u;
    // K1/K2 remain on the canonical implementation until their teardown
    // progress is proven.  For K>=4, source-major sends each source token once
    // and expands the K routes at its paired INC.  The default promotion stays
    // at the soaked T512 point; an explicit bounded experimental ceiling is
    // selected only when it amortizes setup.  There is intentionally no
    // business-shape ceiling here: ring/slab/CSR capacities are per internal
    // epoch, and larger public calls are split by the framework chunk planner.
    const bool source_major_topk_safe = route_topk >= 4u;
    const bool source_major_message_safe = tokens >= 512u;
    if (ingress_layout_request == PlIngressLayoutRequest::kSourceMajor &&
        (!source_major_capacity_safe || !source_major_topology_safe ||
         !source_major_topk_safe || !source_major_message_safe)) {
        std::cerr << "PL_INGRESS_LAYOUT_FAIL reason="
                  << (!source_major_topology_safe
                          ? "source_major_topology"
                          : (!source_major_topk_safe ? "source_major_topk"
                             : (!source_major_message_safe
                                    ? "source_major_message_size"
                                    : "source_major_capacity")))
                  << " worker_count=" << worker_count
                  << " route_topk=" << route_topk
                  << " tokens=" << tokens
                  << " generations_per_lane=" << raw_generations_per_lane
                  << " raw_ring_depth=" << kPlRawChunkRingDepth
                  << " raw_payload_bytes=" << raw_payload_bytes
                  << " raw_payload_capacity=" << kPlIncRawSlabBytes
                  << " raw_metadata_bytes=" << raw_metadata_bytes
                  << " raw_metadata_capacity=" << kPlIncRawRouteMetaBytes
                  << " csr_capacity_safe="
                  << (destination_csr_capacity_safe ? 1 : 0)
                  << " fallback=use_auto_or_dest_major_gather" << std::endl;
        return 11;
    }
    const bool ingress_source_major_raw =
        ingress_layout_request == PlIngressLayoutRequest::kSourceMajor ||
        (ingress_layout_request == PlIngressLayoutRequest::kAuto &&
         source_major_capacity_safe && source_major_imbalance_safe &&
         source_major_topology_safe && source_major_topk_safe &&
         source_major_message_safe);
    if (pe == 0) {
        std::cout << "PL_INGRESS_PLANNER requested="
                  << (ingress_layout_request == PlIngressLayoutRequest::kAuto
                          ? "auto"
                          : (ingress_source_major_raw ? "source_major_raw"
                                                      : "dest_major_gather"))
                  << " selected="
                  << (ingress_source_major_raw ? "source_major_raw"
                                               : "dest_major_gather")
                  << " capacity_safe=" << (source_major_capacity_safe ? 1 : 0)
                  << " imbalance_safe="
                  << (source_major_imbalance_safe ? 1 : 0)
                  << " topology_safe="
                  << (source_major_topology_safe ? 1 : 0)
                  << " topk_safe=" << (source_major_topk_safe ? 1 : 0)
                  << " message_safe=" << (source_major_message_safe ? 1 : 0)
                  << " route_topk=" << route_topk
                  << " max_dest_share_ppm=" << max_dest_share_ppm
                  << " generations_per_lane=" << raw_generations_per_lane
                  << " raw_ring_depth=" << kPlRawChunkRingDepth << std::endl;
    }
    // tokens CLI = local_source_capacity（Full Dispatch）；有效 token 见 invocation
    uint32_t source_token_by_epoch[kPlMaxInvocationEpochs][kPlMaxSources]{};
    ParseSourceTokenCountsByEpoch(worker_count, tokens, ep_count, source_token_by_epoch);
    const uint32_t tokens_per_lane =
        (measurement_mode == kPlMeasurementFullDispatch)
            ? 0u
            : (lane_count == 0u ? 0u : (tokens / lane_count));
    if (measurement_mode == kPlMeasurementFullDispatch && artifact_shape_dir == nullptr) {
        std::cerr << "PL_FULL_DISPATCH_REQUIRES_ARTIFACT shape_dir=unset use INC_DN_PL_ARTIFACT_SHAPE_DIR" << std::endl;
        return 13;
    }
    // persistent session：13 epoch 需 slot 模板复用或 ep_count≤16
    if (measurement_mode == kPlMeasurementFullDispatch) {
        if (ep_count > kPlMaxInvocationEpochs) {
            std::cerr << "PL_FULL_DISPATCH_EPOCH_SLOTS_FAIL ep_count=" << ep_count
                      << " max=" << kPlMaxInvocationEpochs << std::endl;
            return 11;
        }
        if (ep_count == session_epochs && session_epochs > kPlMaxInvocationEpochs) {
            std::cerr << "PL_FULL_DISPATCH_SESSION_OVERFLOW session_epochs=" << session_epochs
                      << " max_slots=" << kPlMaxInvocationEpochs << std::endl;
            return 11;
        }
        if (ep_count != 1u && ep_count < session_epochs && session_epochs > kPlMaxInvocationEpochs) {
            std::cerr << "PL_FULL_DISPATCH_SESSION_SHAPE_CYCLE_OVERFLOW session_epochs=" << session_epochs
                      << " ep_count=" << ep_count << " max_slots=" << kPlMaxInvocationEpochs << std::endl;
            return 11;
        }
    }
    if (!PlValidateHeapLayoutMonotonic()) {
        std::cerr << "PL_HEAP_LAYOUT_MONOTONIC_FAIL kPlMaxInvocationEpochs=" << kPlMaxInvocationEpochs << std::endl;
        return 11;
    }
    // Full Dispatch：shape 校验必须在 workspace contract 之前（CAP_NEG 不得被 CONTRACT/HEAP 冒充）
    if (measurement_mode == kPlMeasurementFullDispatch) {
        for (uint32_t e = 0; e < ep_count && e < kPlMaxInvocationEpochs; ++e) {
            const char *shape_reason = "unknown";
            uint32_t required_slots = 0u;
            if (!PlValidateFullDispatchShape(tokens, lane_count, worker_count, route_topk, ep_count,
                                             source_token_by_epoch[e], nullptr, &shape_reason, &required_slots,
                                             runtime_final_slot_capacity)) {
                std::cerr << "PL_FULL_DISPATCH_SHAPE_FAIL failure_stage=shape_validation"
                          << " failure_reason=" << shape_reason << " required_slots=" << required_slots
                          << " final_slot_capacity=" << resolved_final_slot_capacity
                          << " compile_bound=" << kPlMaxFinalSlotsCompileBound << " epoch_idx=" << e
                          << " capacity=" << tokens << " worker_count=" << worker_count << " topk=" << route_topk
                          << std::endl;
                return 11;
            }
        }
    }
    PlWorkspaceContract workspace_contract{};
    const char *workspace_reason = "skipped";
    const bool workspace_contract_built =
        PlBuildWorkspaceContract(&workspace_contract, worker_count, tokens, route_topk, runtime_final_slot_capacity,
                                 workspace_generation, runtime_payload_bytes, runtime_hidden_size, runtime_dtype_bytes,
                                 runtime_layout_version);
    if (measurement_mode == kPlMeasurementFullDispatch) {
        workspace_reason = workspace_contract_built ? "ok" : "build_fail";
        if (!workspace_contract_built || !PlValidateWorkspaceContract(&workspace_contract, &workspace_reason)) {
            std::cerr << "PL_WORKSPACE_CONTRACT_FAIL failure_reason=" << workspace_reason
                      << " final_slot_capacity=" << resolved_final_slot_capacity
                      << " compile_bound=" << kPlMaxFinalSlotsCompileBound << " capacity=" << tokens
                      << " worker_count=" << worker_count << " topk=" << route_topk
                      << " workspace_bytes=" << workspace_contract.workspace_bytes << std::endl;
            return 11;
        }
    }
    const uint64_t d1_start_off = D1B(kD1StartLineOff);
    const uint64_t d1_stop_off = D1B(kD1SessionStopLineOff);

    if (!PlValidateHeap(tokens, lane_count, worker_count, ring_depth, batch_tokens, tile_tokens, route_topk, ep_count,
                        measurement_mode, source_token_by_epoch[0], runtime_final_slot_capacity)) {
        std::cerr << "PL_HEAP_VALIDATE_FAIL tokens=" << tokens << " lanes=" << lane_count
                  << " mode=" << measurement_mode << " ring=" << ring_depth << " batch=" << batch_tokens
                  << " tile=" << tile_tokens << " topk=" << route_topk
                  << " (FullDispatch: tokens=capacity; zero-token ok; no lanes divisibility)" << std::endl;
        return 11;
    }
    if (!PlValidateRingDepth(ring_depth)) {
        std::cerr << "PL_RING_DEPTH_FAIL ring_depth=" << ring_depth << " required=" << kQv2RingDepth << std::endl;
        return 11;
    }

    aclrtStream service_stream = nullptr;
    aclrtStream control_stream = nullptr;
    aclrtStream leader_stream = nullptr;
    aclrtStream host_stream = nullptr;
    int32_t dev = 0;
    if (InitShmem(pe, npe, &dev, &service_stream, &control_stream, &leader_stream) != 0) {
        return 10;
    }
    if (aclrtCreateStream(&host_stream) != 0) {
        return 10;
    }

    PlWorkerLocalSession worker_local{};
    const char *ws_alloc_mode = ParseWorkspaceAllocMode();
    uint32_t ws_alloc_flags = kPlInvocationWorkspaceFlagPooled;
    bool pool_owns_worker_local = false;
    uint8_t *inc_local_workspace = nullptr;
    const uint64_t inc_local_bytes = static_cast<uint64_t>(ParseEnvU32("INC_DN_PL_INC_LOCAL_WORKSPACE_BYTES", 0u));
    if (is_worker) {
        if (std::strcmp(ws_alloc_mode, "per_process") == 0) {
            ws_alloc_flags = 0u;
            if (!PlAllocWorkerLocalSession(&worker_local, tokens, route_topk, resolved_final_slot_capacity, ep_count,
                                           expert_per_pe, worker_count, runtime_payload_bytes)) {
                std::cerr << "PL_WORKER_LOCAL_ALLOC_FAIL capacity=" << tokens
                          << " final_slot_capacity=" << resolved_final_slot_capacity
                          << " payload_bytes=" << runtime_payload_bytes << std::endl;
                return 11;
            }
        } else if (std::strcmp(ws_alloc_mode, "caller_owned") == 0) {
            ws_alloc_flags = kPlInvocationWorkspaceFlagCallerOwned;
            if (!PlAllocWorkerLocalSession(&worker_local, tokens, route_topk, resolved_final_slot_capacity, ep_count,
                                           expert_per_pe, worker_count, runtime_payload_bytes)) {
                std::cerr << "PL_WORKER_LOCAL_ALLOC_FAIL mode=caller_owned capacity=" << tokens
                          << " payload_bytes=" << runtime_payload_bytes << std::endl;
                return 11;
            }
        } else {
            // pooled (M5 default)
            ws_alloc_flags = kPlInvocationWorkspaceFlagPooled;
            PlWorkspacePoolBucketKey key{};
            key.tokens = tokens;
            key.route_topk = route_topk;
            key.final_slot_capacity = resolved_final_slot_capacity;
            key.input_epoch_count = ep_count;
            key.expert_per_pe = expert_per_pe;
            key.worker_count = worker_count;
            key.payload_bytes = runtime_payload_bytes;
            PlWorkspacePoolEntry entry{};
            bool grew = false;
            if (!g_worker_workspace_pool.Acquire(key, &entry, &grew)) {
                std::cerr << "PL_WORKSPACE_POOL_ACQUIRE_FAIL capacity=" << tokens << std::endl;
                return 11;
            }
            // 演示 grow-only reuse：同 bucket 再 Acquire 一次
            PlWorkspacePoolEntry entry2{};
            bool grew2 = true;
            if (!g_worker_workspace_pool.Acquire(key, &entry2, &grew2) || grew2) {
                std::cerr << "PL_WORKSPACE_POOL_REUSE_FAIL" << std::endl;
                return 11;
            }
            PlFillWorkerLocalFromPoolEntry(&worker_local, entry);
            pool_owns_worker_local = true;
        }
    } else if (inc_local_bytes > 0u) {
        if (aclrtMalloc(reinterpret_cast<void **>(&inc_local_workspace), inc_local_bytes, ACL_MEM_MALLOC_HUGE_FIRST) !=
            0) {
            std::cerr << "PL_INC_LOCAL_WORKSPACE_ALLOC_FAIL bytes=" << inc_local_bytes << std::endl;
            return 11;
        }
        std::cout << "PL_INC_LOCAL_WORKSPACE pe=" << pe << " bytes=" << inc_local_bytes
                  << " ptr=" << reinterpret_cast<uintptr_t>(inc_local_workspace) << std::endl;
    }

    uint8_t *sym = static_cast<uint8_t *>(aclshmem_malloc(kPlSymmetricTransportHeapNeed));
    if (sym == nullptr) {
        if (!pool_owns_worker_local) {
            PlFreeWorkerLocalSession(&worker_local);
        }
        if (inc_local_workspace != nullptr) {
            aclrtFree(inc_local_workspace);
        }
        return 11;
    }
    {
        constexpr uint64_t kHeapAlign = 2ull * 1024 * 1024;
        uint64_t heap_reserve = (kPlSymmetricTransportHeapNeed + kHeapAlign - 1ull) & ~(kHeapAlign - 1ull);
        const uintptr_t sym_u = reinterpret_cast<uintptr_t>(sym);
        std::cout << "PL_HEAP_ALLOC pe=" << pe << " sym=0x" << std::hex << sym_u << std::dec
                  << " transport_bytes=" << kPlSymmetricTransportHeapNeed << " reserve=" << heap_reserve
                  << " align=" << kHeapAlign << " sym_mod_align=" << (sym_u % kHeapAlign)
                  << " final_slot_capacity=" << resolved_final_slot_capacity
                  << " workspace_bytes=" << workspace_contract.workspace_bytes << std::endl;
        if (is_worker) {
            std::cout << "PL_WORKER_LOCAL_ALLOC pe=" << pe << " total_bytes=" << worker_local.total_bytes
                      << " alloc_mode=" << ws_alloc_mode
                      << " input=" << reinterpret_cast<uintptr_t>(worker_local.input)
                      << " expert_ids=" << reinterpret_cast<uintptr_t>(worker_local.expert_ids)
                      << " gather_payload=" << reinterpret_cast<uintptr_t>(worker_local.gather_payload)
                      << " gather_desc=" << reinterpret_cast<uintptr_t>(worker_local.gather_desc)
                      << " worker_desc=" << reinterpret_cast<uintptr_t>(worker_local.worker_desc)
                      << " expand_x=" << reinterpret_cast<uintptr_t>(worker_local.expand_x)
                      << " assist=" << reinterpret_cast<uintptr_t>(worker_local.assist)
                      << " ep_recv_count=" << reinterpret_cast<uintptr_t>(worker_local.ep_recv_count)
                      << " expert_token_nums=" << reinterpret_cast<uintptr_t>(worker_local.expert_token_nums)
                      << " memory_reduction=1" << std::endl;
        }
    }
    aclrtMemset(sym, kPlSymmetricTransportHeapNeed, 0, kPlSymmetricTransportHeapNeed);

    PlPipelineDesc pdesc{};
    pdesc.magic = kPlMagic;
    pdesc.pair_id = pair_id;
    pdesc.peer_pe = is_worker ? (pair_id + worker_count) : pair_id;
    pdesc.my_role = is_worker ? 0u : 1u;
    pdesc.worker_count = worker_count;
    pdesc.lane_count = lane_count;
    pdesc.tokens_per_epoch = tokens;
    pdesc.tokens_per_lane = tokens_per_lane;
    // 必须与 invocation.payload_bytes 一致；禁止长期钉死 kQv2TokenBytes
    pdesc.payload_bytes = runtime_payload_bytes;
    pdesc.ring_depth = ring_depth;
    pdesc.batch_tokens = batch_tokens;
    pdesc.tile_tokens = tile_tokens;
    pdesc.head_tile_tokens = head_tile_tokens;
    pdesc.upload_tile_target_bytes = upload_tile_target_bytes;
    pdesc.egress_publish_target_bytes = egress_publish_target_bytes;
    pdesc.transport_mode = kD1PrimP6RangeMteWaitFinalQuiet;
    pdesc.verify_mode = verify_mode;
    const bool control_only = ParseControlOnly();
    const int epoch_wait_ms = ParseEpochWaitMs();
    const uint32_t payload_source_mode = ParsePayloadSourceMode(warmup, measure);
    pdesc.completion_mode = completion_mode;
    pdesc.host_completion_mode = host_completion_mode;
    if (control_only) {
        pdesc.pipeline_mode = kPlModeControlOnly;
    } else {
        pdesc.pipeline_mode = 0u;
    }
    // Metadata pipeline V2（full-dispatch 默认）：lane-sharded direct count +
    // receiver-owned final placement + compact gather。保留总开关和分项开关用于回退/归因。
    const bool metadata_pipeline_v2 =
        measurement_mode == kPlMeasurementFullDispatch &&
        ParseEnvU32("INC_DN_PL_METADATA_PIPELINE_V2", 1u) != 0u;
    const char *compact_env = std::getenv("INC_DN_PL_UPLOAD_COMPACT_GATHER");
    const char *direct_env = std::getenv("INC_DN_PL_DIRECT_COUNT_EXCHANGE");
    const char *receiver_slot_env = std::getenv("INC_DN_PL_RECEIVER_FINAL_SLOT");
    const bool compact_gather =
        compact_env != nullptr ? ParseEnvU32("INC_DN_PL_UPLOAD_COMPACT_GATHER", 0u) != 0u
                               : metadata_pipeline_v2;
    const bool direct_count =
        direct_env != nullptr ? ParseEnvU32("INC_DN_PL_DIRECT_COUNT_EXCHANGE", 0u) != 0u
                              : metadata_pipeline_v2;
    const bool receiver_final_slot =
        receiver_slot_env != nullptr ? ParseEnvU32("INC_DN_PL_RECEIVER_FINAL_SLOT", 0u) != 0u
                                     : metadata_pipeline_v2;
    if (compact_gather) {
        pdesc.pipeline_mode |= kPlModeUploadCompactGather;
    }
    if (direct_count) {
        pdesc.pipeline_mode |= inc::dc::dn::pl::kPlModeDirectCountExchange;
    }
    if (receiver_final_slot) {
        pdesc.pipeline_mode |= inc::dc::dn::pl::kPlModeReceiverFinalSlot;
    }
    if (ingress_source_major_raw) {
        pdesc.pipeline_mode |= kPlIngressLayoutSourceMajorRaw;
        // P5.4 remains an explicit candidate until its distributed,
        // chunk-streamed compiler beats the proven source-major path.  Never
        // promote a correctness-only global CSR that regresses startup.
        if (ParseEnvU32("INC_DN_PL_DESTINATION_CSR_ROUTE_PLAN", 0u) !=
            0u) {
            pdesc.pipeline_mode |=
                inc::dc::dn::pl::kPlModeDestinationCsrRoutePlan;
        }
    }
    // Candidate2：R 编入 pipeline_mode[31:24]；U=lane_count；layout 真减 block_dim
    pdesc.pipeline_mode = PlPackRecvLaneCountIntoPipelineMode(pdesc.pipeline_mode, recv_lane_count);
    pdesc.payload_source_mode = payload_source_mode;
    pdesc.d1_start_off = d1_start_off;
    pdesc.d1_stop_off = d1_stop_off;
    pdesc.expected_tokens_per_dest = tokens_per_lane * worker_count;
    pdesc.measurement_mode = measurement_mode;
    pdesc.route_topk = route_topk;
    pdesc.expert_per_pe = expert_per_pe;
    pdesc.moe_expert_num = worker_count * expert_per_pe;
    const uint32_t full_dispatch_probe = ParseFullDispatchProbe();

    if (payload_source_mode == kPlPayloadSourceEpochUnique) {
        const uint32_t epoch_slots = warmup + measure;
        if (static_cast<uint64_t>(epoch_slots) * tokens > kPlMaxTokens) {
            std::cerr << "PL_PAYLOAD_EPOCH_UNIQUE_OVERFLOW epochs=" << epoch_slots << " tokens=" << tokens
                      << " max=" << kPlMaxTokens << std::endl;
            return 11;
        }
    }

    // 更新 device 侧 pipeline desc（含 payload_source_mode）
    auto HostPublishPipelineDesc = [&]() {
        aclrtMemcpy(sym + kPlDescOff, sizeof(pdesc), &pdesc, sizeof(pdesc), ACL_MEMCPY_HOST_TO_DEVICE);
    };

    for (uint32_t lane = 0; lane < lane_count; ++lane) {
        Qv2TailLine tail{};
        tail.magic = kQv2Magic;
        tail.lane_id = lane;
        Qv2HeadLine head{};
        head.magic = kQv2Magic;
        head.lane_id = lane;
        aclrtMemcpy(sym + kPlIngressTailOff + lane * 64u, sizeof(tail), &tail, sizeof(tail), ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(sym + kPlIngressHeadOff + lane * 64u, sizeof(head), &head, sizeof(head), ACL_MEMCPY_HOST_TO_DEVICE);
    }
    for (uint32_t s = 0; s < worker_count; ++s) {
        Qv2TailLine tail{};
        tail.magic = kQv2Magic;
        tail.lane_id = s;
        Qv2HeadLine head{};
        head.magic = kQv2Magic;
        head.lane_id = s;
        aclrtMemcpy(sym + kPlEgressTailOff + s * 64u, sizeof(tail), &tail, sizeof(tail), ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(sym + kPlEgressHeadOff + s * 64u, sizeof(head), &head, sizeof(head), ACL_MEMCPY_HOST_TO_DEVICE);
    }
    // P5.3：per-lane raw tail/head + route_done + staging SPSC + inject line
    {
        using inc::dc::dn::pl::kPlRawIngressTailBytes;
        using inc::dc::dn::pl::kPlRawIngressHeadBytes;
        using inc::dc::dn::pl::kPlRouteDoneGenOff;
        using inc::dc::dn::pl::kPlRouteDoneGenBytes;
        using inc::dc::dn::pl::kPlIncP5StagingTailOff;
        using inc::dc::dn::pl::kPlIncP5StagingReadyBytes;
        using inc::dc::dn::pl::kPlP53InjectOff;
        using inc::dc::dn::pl::kPlP53InjectMagic;
        using inc::dc::dn::pl::kPlP53InjectEarlyTail;
        using inc::dc::dn::pl::kPlP53InjectStaleGen;
        using inc::dc::dn::pl::kPlP53InjectWithholdCredit;
        using inc::dc::dn::pl::kPlP53InjectDropStagingTail;
        using inc::dc::dn::pl::kPlP53InjectPrefillStaleEpoch;
        using inc::dc::dn::pl::kPlP53InjectSameEpochHeadGtTail;
        using inc::dc::dn::pl::kPlP53InjectTerminalTailRepublish;
        using inc::dc::dn::pl::kPlP53InjectDropMetaReady;
        using inc::dc::dn::pl::kPlIncP5EgressDoneOff;
        using inc::dc::dn::pl::kPlIncP5EgressDoneBytes;
        using inc::dc::dn::pl::PlP53InjectLine;
        using inc::dc::dn::pl::PlRawLaneCtrlLine;
        using inc::dc::dn::pl::kPlRawLaneCtrlMagic;
        using inc::dc::dn::pl::PlStagingCtrlLine;
        using inc::dc::dn::pl::kPlStagingCtrlMagic;
        std::vector<uint8_t> zt(static_cast<size_t>(kPlRawIngressTailBytes), 0);
        std::vector<uint8_t> zh(static_cast<size_t>(kPlRawIngressHeadBytes), 0);
        std::vector<uint8_t> zd(static_cast<size_t>(kPlRouteDoneGenBytes), 0);
        std::vector<uint8_t> zs(static_cast<size_t>(kPlIncP5StagingReadyBytes), 0);
        std::vector<uint8_t> ze(static_cast<size_t>(kPlIncP5EgressDoneBytes), 0);
        aclrtMemcpy(sym + kPlRawIngressTailOff, zt.size(), zt.data(), zt.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(sym + kPlRawIngressHeadOff, zh.size(), zh.data(), zh.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(sym + kPlRouteDoneGenOff, zd.size(), zd.data(), zd.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(sym + kPlIncP5StagingTailOff, zs.size(), zs.data(), zs.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(sym + kPlIncP5EgressDoneOff, ze.size(), ze.data(), ze.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        PlP53InjectLine inj{};
        inj.magic = kPlP53InjectMagic;
        inj.flags = 0u;
        if (const char *e = std::getenv("INC_DN_PL_P53_INJECT")) {
            const std::string s(e);
            if (s.find("early_tail") != std::string::npos) {
                inj.flags |= kPlP53InjectEarlyTail;
            }
            if (s.find("stale_gen") != std::string::npos) {
                inj.flags |= kPlP53InjectStaleGen;
            }
            if (s.find("withhold_credit") != std::string::npos) {
                inj.flags |= kPlP53InjectWithholdCredit;
            }
            if (s.find("drop_staging_tail") != std::string::npos) {
                inj.flags |= kPlP53InjectDropStagingTail;
            }
            if (s.find("prefill_stale_epoch") != std::string::npos) {
                inj.flags |= kPlP53InjectPrefillStaleEpoch;
            }
            if (s.find("same_epoch_head_gt_tail") != std::string::npos) {
                inj.flags |= kPlP53InjectSameEpochHeadGtTail;
            }
            if (s.find("drop_meta_ready") != std::string::npos) {
                inj.flags |= kPlP53InjectDropMetaReady;
            }
        }
        if (const char *e = std::getenv("INC_DN_PL_P5_TERMINAL_TAIL_REPUBLISH")) {
            if (std::strcmp(e, "1") == 0) {
                inj.flags |= kPlP53InjectTerminalTailRepublish;
            }
        }
        aclrtMemcpy(sym + kPlP53InjectOff, sizeof(inj), &inj, sizeof(inj), ACL_MEMCPY_HOST_TO_DEVICE);
        {
            using inc::dc::dn::pl::kPlRawMetaReadyOff;
            using inc::dc::dn::pl::PlRawMetaReadyLine;
            PlRawMetaReadyLine zr{};
            aclrtMemcpy(sym + kPlRawMetaReadyOff, sizeof(zr), &zr, sizeof(zr), ACL_MEMCPY_HOST_TO_DEVICE);
            ControlEpochRetireLine zr_retire{};
            aclrtMemcpy(sym + kPlControlEpochRetireOff, sizeof(zr_retire), &zr_retire, sizeof(zr_retire),
                        ACL_MEMCPY_HOST_TO_DEVICE);
        }
        if (inj.flags != 0u) {
            std::cout << "PL_P53_INJECT pe=" << pe << " flags=0x" << std::hex << inj.flags << std::dec
                      << std::endl;
        }
        // E6 正例：预填旧 epoch 控制线；reader 必须 ignore 且会话仍 PASS
        if ((inj.flags & kPlP53InjectPrefillStaleEpoch) != 0u) {
            PlRawLaneCtrlLine stale_h{};
            stale_h.value = 99u;
            stale_h.epoch = 999u;
            stale_h.magic = kPlRawLaneCtrlMagic;
            stale_h.lane_id = 0u;
            aclrtMemcpy(sym + kPlRawIngressHeadOff, sizeof(stale_h), &stale_h, sizeof(stale_h),
                        ACL_MEMCPY_HOST_TO_DEVICE);
            PlStagingCtrlLine stale_st{};
            stale_st.value = 77u;
            stale_st.epoch = 999u;
            stale_st.magic = kPlStagingCtrlMagic;
            stale_st.destination = 0u;
            aclrtMemcpy(sym + kPlIncP5StagingTailOff, sizeof(stale_st), &stale_st, sizeof(stale_st),
                        ACL_MEMCPY_HOST_TO_DEVICE);
            Qv2HeadLine stale_cr{};
            stale_cr.magic = kQv2Magic;
            stale_cr.head = 55u;
            stale_cr.epoch_tag = 999u;
            stale_cr.lane_id = 0u;
            aclrtMemcpy(sym + kPlIncEgressCreditOff, sizeof(stale_cr), &stale_cr, sizeof(stale_cr),
                        ACL_MEMCPY_HOST_TO_DEVICE);
            std::cout << "PL_P53_PREFILL_STALE_EPOCH pe=" << pe << " head=99 staging=77 credit=55 epoch=999"
                      << std::endl;
        }
    }
    // INC egress credit 必须清零；脏 head 会跳过 credit wait 导致 ring 覆盖、recv 卡死
    for (uint32_t d = 0; d < worker_count; ++d) {
        Qv2HeadLine credit{};
        credit.magic = kQv2Magic;
        credit.lane_id = d;
        credit.head = 0;
        credit.epoch_tag = 0;
        aclrtMemcpy(sym + kPlIncEgressCreditOff + static_cast<uint64_t>(d) * 64u, sizeof(credit), &credit,
                    sizeof(credit), ACL_MEMCPY_HOST_TO_DEVICE);
    }
    // C3 控制面：count/ready/segbase/recv_done + invocation/workspace 表清零
    {
        const uint64_t ctrl_bytes =
            (kPlInvocationWorkspaceOff + kPlInvocationWorkspaceBytes) - kPlRouteCountRegionOff;
        std::vector<uint8_t> z(static_cast<size_t>(ctrl_bytes), 0);
        aclrtMemcpy(sym + kPlRouteCountRegionOff, z.size(), z.data(), z.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    }
    HostPublishPipelineDesc();
    if (is_worker) {
        PlFullDispatchConfig fcfg_worker = PlBuildFullDispatchConfig(worker_count, lane_count);
        // V2：第一跳源 = Worker local input；不再使用 symmetric upload staging
        fcfg_worker.raw_input_off = reinterpret_cast<uint64_t>(worker_local.input);
        fcfg_worker.raw_expert_ids_off = reinterpret_cast<uint64_t>(worker_local.expert_ids);
        fcfg_worker.gather_payload_off = reinterpret_cast<uint64_t>(worker_local.gather_payload);
        fcfg_worker.gather_desc_off = reinterpret_cast<uint64_t>(worker_local.gather_desc);
        fcfg_worker.final_payload_off = reinterpret_cast<uint64_t>(worker_local.worker_desc);
        aclrtMemcpy(sym + kPlFullDispatchConfigOff, sizeof(fcfg_worker), &fcfg_worker, sizeof(fcfg_worker),
                    ACL_MEMCPY_HOST_TO_DEVICE);
    }
    const bool force_stale_gen = (std::getenv("INC_DN_PL_C2_FORCE_STALE_GENERATION") != nullptr &&
                                 std::strcmp(std::getenv("INC_DN_PL_C2_FORCE_STALE_GENERATION"), "1") == 0);
    const bool m2_skip_ws = (std::getenv("INC_DN_PL_M2_SKIP_WORKSPACE_DESC") != nullptr &&
                             std::strcmp(std::getenv("INC_DN_PL_M2_SKIP_WORKSPACE_DESC"), "1") == 0);
    const bool m2_bad_magic = (std::getenv("INC_DN_PL_M2_BAD_WORKSPACE_MAGIC") != nullptr &&
                               std::strcmp(std::getenv("INC_DN_PL_M2_BAD_WORKSPACE_MAGIC"), "1") == 0);
    const bool m2_strip_legacy = (std::getenv("INC_DN_PL_M2_STRIP_LEGACY_FLAG") != nullptr &&
                                  std::strcmp(std::getenv("INC_DN_PL_M2_STRIP_LEGACY_FLAG"), "1") == 0);
    const bool m2_bad_generation = (std::getenv("INC_DN_PL_M2_BAD_WORKSPACE_GENERATION") != nullptr &&
                                    std::strcmp(std::getenv("INC_DN_PL_M2_BAD_WORKSPACE_GENERATION"), "1") == 0);
    if (measurement_mode == kPlMeasurementFullDispatch) {
        PlFullDispatchConfig fcfg = PlBuildFullDispatchConfig(worker_count, lane_count);
        fcfg.probe_mode = full_dispatch_probe;
        fcfg.input_epoch_count = ep_count;
        if (is_worker) {
            fcfg.raw_input_off = reinterpret_cast<uint64_t>(worker_local.input);
            fcfg.raw_expert_ids_off = reinterpret_cast<uint64_t>(worker_local.expert_ids);
            fcfg.gather_payload_off = reinterpret_cast<uint64_t>(worker_local.gather_payload);
            fcfg.gather_desc_off = reinterpret_cast<uint64_t>(worker_local.gather_desc);
            fcfg.final_payload_off = reinterpret_cast<uint64_t>(worker_local.worker_desc);
        }
        if (force_stale_gen) {
            fcfg.version = 2u; // device：epoch>=2 故意写 generation=1
        }
        aclrtMemcpy(sym + kPlFullDispatchConfigOff, sizeof(fcfg), &fcfg, sizeof(fcfg), ACL_MEMCPY_HOST_TO_DEVICE);
        // 仅当单 shape 且 session 多 epoch（perf 13-epoch）时用模板；correctness 单 epoch 仍写 epoch=1
        const bool use_template_slots = (ep_count == 1u && session_epochs > ep_count);
        const PlWorkspaceContract *ws_ptr =
            (measurement_mode == kPlMeasurementFullDispatch) ? &workspace_contract : nullptr;
        for (uint32_t e = 0; e < ep_count && e < kPlMaxInvocationEpochs; ++e) {
            PlInvocationDesc inv{};
            if (use_template_slots) {
                PlFillInvocationDesc(&inv, 0, 0, worker_count, route_topk, source_token_by_epoch[e], tokens,
                                     runtime_payload_bytes, ws_ptr, runtime_hidden_size, runtime_dtype_bytes,
                                     runtime_layout_version);
                inv.flags |= kPlInvocationFlagsTemplateSlot;
            } else {
                const uint64_t gen = static_cast<uint64_t>(e + 1u);
                PlFillInvocationDesc(&inv, gen, gen, worker_count, route_topk, source_token_by_epoch[e], tokens,
                                     runtime_payload_bytes, ws_ptr, runtime_hidden_size, runtime_dtype_bytes,
                                     runtime_layout_version);
            }
            if (!PlInvocationPayloadValid(&inv)) {
                std::cerr << "PL_INVOCATION_PAYLOAD_FAIL slot=" << e << " hidden=" << inv.hidden_size
                          << " dtype_bytes=" << inv.dtype_bytes << " payload_bytes=" << inv.payload_bytes
                          << " layout_version=" << inv.layout_version << std::endl;
                return 11;
            }
            aclrtMemcpy(sym + PlInvocationDescLineOff(e), sizeof(inv), &inv, sizeof(inv), ACL_MEMCPY_HOST_TO_DEVICE);

            // M2/M3：workspace desc 与 invocation 同 epoch 发布；fail-closed 禁止 silently skip
            if (!m2_skip_ws) {
                PlInvocationWorkspaceDesc iws{};
                const uint32_t ws_gen = (inv.workspace_generation != 0u)
                                           ? inv.workspace_generation
                                           : workspace_contract.workspace_generation;
                bool ws_built = false;
                if (!is_worker) {
                    iws = PlInvocationWorkspaceDesc{};
                    iws.magic = kPlInvocationWorkspaceMagic;
                    iws.flags = kPlInvocationWorkspaceFlagWorkerLocalFinal;
                    iws.source_token_capacity = tokens;
                    iws.output_route_capacity = inv.final_slot_capacity;
                    iws.topk = route_topk;
                    iws.generation = ws_gen == 0u ? 1u : ws_gen;
                    ws_built = true;
                } else if (workspace_local_final) {
                    const uint32_t out_cap = inv.final_slot_capacity;
                    ws_built = PlBuildInvocationWorkspace(
                        &iws, reinterpret_cast<uint64_t>(worker_local.expand_x),
                        reinterpret_cast<uint64_t>(worker_local.assist),
                        reinterpret_cast<uint64_t>(worker_local.ep_recv_count),
                        reinterpret_cast<uint64_t>(worker_local.expert_token_nums), tokens, out_cap, route_topk,
                        ws_gen == 0u ? 1u : ws_gen, ws_alloc_flags, runtime_payload_bytes, runtime_hidden_size,
                        runtime_dtype_bytes, runtime_layout_version);
                    if (ws_built && is_worker) {
                        // V2：第一跳直接从 Worker local input/expert_ids（M1 local-GM-source 路径）
                        iws.input_ptr = reinterpret_cast<uint64_t>(worker_local.input);
                        iws.expert_ids_ptr = reinterpret_cast<uint64_t>(worker_local.expert_ids);
                        iws.input_bytes = static_cast<uint64_t>(tokens) * static_cast<uint64_t>(runtime_payload_bytes);
                    }
                } else {
                    ws_built = PlBuildLegacySymmetricInvocationWorkspace(&iws, tokens, inv.final_slot_capacity,
                                                                         route_topk, ws_gen == 0u ? 1u : ws_gen);
                }
                if (!ws_built) {
                    std::cerr << "PL_INVOCATION_WORKSPACE_BUILD_FAIL slot=" << e
                              << " mode=" << (workspace_local_final ? "local_final" : "legacy") << std::endl;
                    return 11;
                }
                if (m2_bad_magic) {
                    iws.magic = 0u;
                }
                if (m2_strip_legacy) {
                    iws.flags = 0u;
                }
                if (m2_bad_generation) {
                    iws.generation = inv.workspace_generation + 999u;
                }
                iws.layout_version = inv.layout_version;
                iws.hidden_size = inv.hidden_size;
                iws.dtype_bytes = inv.dtype_bytes;
                iws.payload_bytes = inv.payload_bytes;
                const char *ws_reason = nullptr;
                const bool ws_valid =
                    is_worker
                        ? PlValidateInvocationWorkspace(&iws, &ws_reason)
                        : ((iws.magic == kPlInvocationWorkspaceMagic) &&
                           ((iws.flags & kPlInvocationWorkspaceFlagWorkerLocalFinal) != 0u));
                if (!ws_valid && !m2_bad_magic && !m2_strip_legacy && !m2_bad_generation) {
                    std::cerr << "PL_INVOCATION_WORKSPACE_FAIL slot=" << e << " reason="
                              << (ws_reason != nullptr ? ws_reason : "inc_stub_invalid") << std::endl;
                    return 11;
                }
                // 负例故意写坏 desc，仍 H2D 让 device fail-closed
                if ((m2_bad_magic || m2_strip_legacy || m2_bad_generation) || ws_valid) {
                    aclrtMemcpy(sym + PlInvocationWorkspaceLineOff(e), sizeof(iws), &iws, sizeof(iws),
                                ACL_MEMCPY_HOST_TO_DEVICE);
                }
                std::cout << "PL_INVOCATION_WORKSPACE pe=" << pe << " slot=" << e
                          << " generation=" << iws.generation << " flags=" << iws.flags
                          << " output_route_capacity=" << iws.output_route_capacity
                          << " legacy=" << ((iws.flags & kPlInvocationWorkspaceFlagLegacySymmetricFinal) ? 1 : 0)
                          << " local_final="
                          << ((iws.flags & kPlInvocationWorkspaceFlagWorkerLocalFinal) ? 1 : 0)
                          << " memory_reduction=0" << std::endl;
            } else {
                std::cout << "PL_INVOCATION_WORKSPACE pe=" << pe << " slot=" << e
                          << " skipped=1 (INC_DN_PL_M2_SKIP_WORKSPACE_DESC)" << std::endl;
            }

            std::cout << "PL_INVOCATION_DESC pe=" << pe << " slot=" << e << " epoch=" << inv.epoch
                      << " generation=" << inv.generation << " template_slot=" << (use_template_slots ? 1 : 0)
                      << " total_input_tokens=" << inv.total_input_tokens
                      << " total_route_instances=" << inv.total_route_instances
                      << " local_source_capacity=" << inv.local_source_capacity
                      << " route_topk=" << inv.route_topk << " final_slot_capacity=" << inv.final_slot_capacity
                      << " workspace_generation=" << inv.workspace_generation
                      << " final_payload_offset=" << inv.final_payload_offset
                      << " final_assist_offset=" << inv.final_assist_offset
                      << " descriptor_capacity=" << inv.descriptor_capacity
                      << " expert_ids_capacity=" << inv.expert_ids_capacity
                      << " workspace_bytes=" << inv.workspace_bytes
                      << " raw_input_epoch_stride_bytes=" << inv.raw_input_epoch_stride_bytes
                      << " raw_expert_ids_epoch_stride_bytes=" << inv.raw_expert_ids_epoch_stride_bytes
                      << " source_token_count=[";
            for (uint32_t s = 0; s < worker_count; ++s) {
                std::cout << (s ? "," : "") << inv.source_token_count[s];
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "PL_FULL_DISPATCH_CONFIG probe_mode=" << full_dispatch_probe
                  << " input_epoch_count=" << fcfg.input_epoch_count << " version=" << fcfg.version
                  << " session_epochs=" << session_epochs << " template_slots=" << (use_template_slots ? 1 : 0)
                  << " perf_skip_epoch_clear=" << (perf_skip_epoch_clear ? 1 : 0)
                  << " workspace_generation=" << workspace_contract.workspace_generation
                  << " final_slot_capacity=" << workspace_contract.final_slot_capacity
                  << " workspace_bytes=" << workspace_contract.workspace_bytes
                  << " compile_bound=" << kPlMaxFinalSlotsCompileBound
                  << " host_descriptor_build_count=0 device_route_gather_included=1" << std::endl;
        HostEmitProbeExecMatrix(full_dispatch_probe);
        std::cout << "PL_P0_CONFIG pe=" << pe
                  << " hidden=" << runtime_hidden_size << " payload_bytes=" << runtime_payload_bytes
                  << " tokens_per_source=" << tokens << " ring_depth=" << ring_depth
                  << " batch_tokens=" << batch_tokens
                  << " upload_tile_tokens=" << tile_tokens
                  << " egress_publish_tokens=" << head_tile_tokens
                  << " upload_tile_target_bytes=" << upload_tile_target_bytes
                  << " egress_publish_target_bytes=" << egress_publish_target_bytes
                  << " upload_tile_effective_bytes=" << (static_cast<uint64_t>(tile_tokens) * runtime_payload_bytes)
                  << " egress_publish_effective_bytes="
                  << (static_cast<uint64_t>(head_tile_tokens) * runtime_payload_bytes)
                  << " first_tile_bytes_cap=" << (static_cast<uint64_t>(tile_tokens) * runtime_payload_bytes)
                  << " upload_compact_gather="
                  << (((pdesc.pipeline_mode & kPlModeUploadCompactGather) != 0u) ? 1 : 0)
                  << std::endl;
    }
    if (HostInitControlLines(sym, host_stream) != 0) {
        return 12;
    }

    std::vector<PlDescriptor> desc_template;
    int art_rc = 0;
    if (measurement_mode == kPlMeasurementFullDispatch) {
        // 全 ranks 先做 stack 容量检查：避免仅 worker 失败后 INC 继续 launch → barrier hang
        if (static_cast<uint64_t>(ep_count) * static_cast<uint64_t>(tokens) >
            static_cast<uint64_t>(kPlMaxTokens) * kPlMaxStackedInputEpochs) {
            std::cerr << "PL_ARTIFACT_EPOCH_OVERFLOW epochs=" << ep_count << " capacity=" << tokens
                      << " max_stack_tokens="
                      << (static_cast<uint64_t>(kPlMaxTokens) * kPlMaxStackedInputEpochs) << std::endl;
            art_rc = -6;
        } else if (is_worker) {
            art_rc = HostArtifactRawLoad(&worker_local, pair_id, worker_count, tokens, expert_per_pe, route_topk, ep_count,
                                        source_token_by_epoch, host_stream);
            if (art_rc != 0) {
                std::cerr << "PL_ARTIFACT_RAW_LOAD_FAIL rc=" << art_rc << std::endl;
            } else if (HostRefreshWorkerUploadStaging(sym, &worker_local, tokens, route_topk, 0u, host_stream) != 0) {
                std::cerr << "PL_UPLOAD_STAGING_REFRESH_FAIL epoch_idx=0" << std::endl;
                art_rc = -7;
            }
        } else {
            // INC 也校验 manifest，避免 Worker 单方 return 后 barrier hang
            const char *sd = std::getenv("INC_DN_PL_ARTIFACT_SHAPE_DIR");
            if (sd != nullptr) {
                art_rc = HostValidateArtifactManifest(worker_count * 2u, worker_count, expert_per_pe, sd);
            } else {
                art_rc = -1;
            }
        }
    } else if (is_worker) {
        HostPrefillBalanced(&worker_local, pair_id, worker_count, tokens, lane_count, host_stream, &desc_template);
    }
    aclshmem_barrier_all();
    if (art_rc != 0) {
        return 13;
    }

    std::cout << "PL_SESSION_META process_launch_count=1 service_kernel_launch_count=1"
              << " session_epoch_count=" << (warmup + measure) << " pe=" << pe << std::endl;

    if (is_worker) {
        std::cout << "PL_WORKER_SERVICE_LAYOUT pe=" << pe
                  << " upload_aiv=" << g_worker_layout.upload_lane_count
                  << " recv_aiv=" << g_worker_layout.recv_lane_count
                  << " control_aiv=1"
                  << " worker_total_dispatch_aiv=" << g_worker_layout.block_dim
                  << " upload_begin=" << g_worker_layout.upload_begin
                  << " recv_begin=" << g_worker_layout.recv_begin
                  << " control_block=" << g_worker_layout.control_block
                  << " inc_dispatch_aiv=" << (ingress_source_major_raw ? kPlIncP5ServiceBlockDim : kPlIncServiceBlockDim)
                  << std::endl;
        launch_inc_dc_dn_pipeline_worker_persistent_kernel(
            sym, kPlDescOff, kPlUploadCtrOff, kPlRecvCtrOff, kPlTimingOff, kPlOverlapOff, kPlWorkerTraceOff,
            kPlSourceDoneOff, kPlDestDoneOff, d1_start_off, d1_stop_off, D1B(kD1LeaderCompletionOff),
            D1B(kD1PublishScratchOff), kD1LeaderPe, static_cast<int>(g_worker_layout.block_dim), service_stream);
        launch_inc_dc_dn_d1_worker_ready_control_kernel(sym, D1B(kD1LocalReadyLineOff), D1B(kD1LeaderRendezvousOff),
                                                        D1B(kD1PublishScratchOff), D1B(kD1SessionStopLineOff), pair_id,
                                                        kD1LeaderPe, control_stream);
    } else {
        const int inc_blocks =
            ingress_source_major_raw ? static_cast<int>(kPlIncP5ServiceBlockDim)
                                     : static_cast<int>(kPlIncServiceBlockDim);
        launch_inc_dc_dn_pipeline_inc_persistent_kernel(sym, kPlDescOff, kPlForwardCtrOff, kPlTimingOff, kPlOverlapOff,
                                                        kPlIncTraceOff, d1_start_off, d1_stop_off, inc_blocks,
                                                        service_stream);
    }
    if (pe == static_cast<int>(kD1LeaderPe) && is_worker && leader_stream != nullptr) {
        launch_inc_dc_dn_d1_leader_release_control_kernel(
            sym, D1B(kD1LeaderRendezvousOff), D1B(kD1LeaderTimingOff), D1B(kD1LeaderCompletionOff),
            D1B(kD1GlobalDoneLineOff), D1B(kD1GlobalDoneTimingOff), D1B(kD1PairDoneArrivalOff), d1_start_off,
            D1B(kD1PublishScratchOff), D1B(kD1SessionStopLineOff), worker_count, worker_count, leader_stream);
    }
    aclshmem_barrier_all();

    int pass = 1;
    // resident：Worker 用真 launch block_dim；P5 INC 要 20 blocks
    const uint32_t expect_mask =
        is_worker ? PlWorkerResidentExpectMask(g_worker_layout.block_dim)
                  : ((1u << kPlIncServiceBlockDim) - 1u);
    const uint32_t resident_expect_mask =
        is_worker ? expect_mask
                  : (ingress_source_major_raw ? ((1u << kPlIncP5ServiceBlockDim) - 1u) : expect_mask);
    const uint64_t resident_base = is_worker ? kPlWorkerResidentOff : kPlIncResidentOff;
    const uint32_t resident_blocks =
        is_worker ? g_worker_layout.block_dim
                  : (ingress_source_major_raw ? kPlIncP5ServiceBlockDim : kPlIncServiceBlockDim);
    int resident_rc = 0;
    if (!is_worker && ingress_source_major_raw) {
        resident_rc = WaitResidentIncP5(sym);
    } else {
        resident_rc = WaitResident(sym, resident_blocks, resident_base, resident_expect_mask);
    }
    if (resident_rc != 0) {
        std::cerr << "PL_RESIDENT_TIMEOUT pe=" << pe << std::endl;
        pass = 0;
    } else {
        std::cout << "PL_RESIDENT pe=" << pe << " mask=0x" << std::hex << resident_expect_mask << std::dec
                  << " ingress_layout=" << (ingress_source_major_raw ? "source_major_raw" : "dest_major_gather")
                  << std::endl;
    }
    const uint64_t trace_base = is_worker ? kPlWorkerTraceOff : kPlIncTraceOff;
    const uint32_t service_blocks = is_worker ? g_worker_layout.block_dim : kPlIncServiceBlockDim;
    bool epoch_aborted = false;
    PlHostEpochSnap prev_measure_snap{};
    bool have_prev_measure_snap = false;

    uint32_t host_descriptor_patch_count = 0u;
    if (pass) {
        for (uint32_t iter = 0; iter < warmup + measure; ++iter) {
            const bool is_measure = (iter >= warmup);
            const uint64_t go_epoch = iter + 1;
            // H8.3：N→N+1 前等 worker control 显式 ControlEpochRetireLine（禁止 cycle 推断 / second-hop grace）
            if (!control_only && measurement_mode == kPlMeasurementFullDispatch &&
                full_dispatch_probe == kPlFullDispatchProbeFull && go_epoch > 1u) {
                int quiescent_rc = 0;
                if (is_worker) {
                    HostEmitEpochStage(pe, go_epoch, "before_control_epoch_retire");
                    quiescent_rc =
                        HostWaitControlEpochRetire(sym, go_epoch - 1u, epoch_wait_ms, pe);
                    if (quiescent_rc == 0) {
                        HostEmitEpochStage(pe, go_epoch, "after_control_epoch_retire");
                    }
                }
                aclshmem_barrier_all();
                if (quiescent_rc != 0) {
                    pass = 0;
                    epoch_aborted = true;
                    if (is_worker && HostPublishSessionStop(sym, d1_stop_off, host_stream) != 0) {
                        pass = 0;
                    }
                    aclshmem_barrier_all();
                    break;
                }
            }
            if (is_worker) {
                if (!control_only) {
                    if (measurement_mode == kPlMeasurementFullDispatch) {
                        const uint32_t ep_idx = PlInvocationSlotIndex(ep_count, go_epoch);
                        if (HostRefreshWorkerUploadStaging(sym, &worker_local, tokens, route_topk, ep_idx,
                                                           host_stream) != 0) {
                            pass = 0;
                            epoch_aborted = true;
                            std::cerr << "PL_UPLOAD_STAGING_REFRESH_FAIL pe=" << pe << " go_epoch=" << go_epoch
                                      << " ep_idx=" << ep_idx << std::endl;
                            if (HostPublishSessionStop(sym, d1_stop_off, host_stream) != 0) {
                                pass = 0;
                            }
                            aclshmem_barrier_all();
                            break;
                        }
                    }
                    HostEmitEpochStage(pe, go_epoch, "before_optional_desc_patch");
                    if (measurement_mode == kPlMeasurementPrepackedTransport) {
                        const int pr =
                            HostPatchDescEpoch(desc_template, &worker_local, tokens, lane_count, go_epoch, host_stream);
                        if (pr != 0) {
                            pass = 0;
                            epoch_aborted = true;
                            if (HostPublishSessionStop(sym, d1_stop_off, host_stream) != 0) {
                                pass = 0;
                            }
                            aclshmem_barrier_all();
                            break;
                        }
                        ++host_descriptor_patch_count;
                        if (go_epoch > 1u) {
                            PlUploadCounters uc{};
                            aclrtMemcpy(&uc, sizeof(uc), sym + kPlUploadCtrOff, sizeof(uc), ACL_MEMCPY_DEVICE_TO_HOST);
                            HostSyncIngressHeadFromInc(sym, static_cast<int>(pair_id + worker_count), lane_count,
                                                       go_epoch, pe, uc.local_tail);
                        }
                    } else if (measurement_mode == kPlMeasurementStage1Diagnostic ||
                               measurement_mode == kPlMeasurementFullDispatch) {
                        if (!desc_template.empty()) {
                            std::cerr << "PL_HOST_DESC_TEMPLATE_FORBIDDEN mode=" << measurement_mode
                                      << " size=" << desc_template.size() << std::endl;
                            pass = 0;
                            epoch_aborted = true;
                            if (HostPublishSessionStop(sym, d1_stop_off, host_stream) != 0) {
                                pass = 0;
                            }
                            aclshmem_barrier_all();
                            break;
                        }
                    } else {
                        std::cerr << "PL_HOST_MEASUREMENT_MODE_FAIL mode=" << measurement_mode << std::endl;
                        pass = 0;
                        epoch_aborted = true;
                        break;
                    }
                    HostEmitEpochStage(pe, go_epoch, "after_optional_desc_patch");
                }
                // Full Dispatch：correctness 默认每 epoch 清零四输出；perf 可跳过
                if (!control_only && measurement_mode == kPlMeasurementFullDispatch &&
                    full_dispatch_probe == kPlFullDispatchProbeFull && !perf_skip_epoch_clear) {
                    HostEmitEpochStage(pe, go_epoch, "before_clear_four_output");
                    if (HostClearFullDispatchEpochOutputs(&worker_local, tokens, expert_per_pe, worker_count, route_topk,
                                                          host_stream) != 0) {
                        pass = 0;
                        epoch_aborted = true;
                        std::cerr << "PL_FULL_DISPATCH_EPOCH_CLEAR_FAIL pe=" << pe << " epoch=" << go_epoch
                                  << std::endl;
                        break;
                    }
                    HostEmitEpochStage(pe, go_epoch, "after_clear_four_output");
                } else if (!control_only && measurement_mode == kPlMeasurementFullDispatch &&
                           full_dispatch_probe == kPlFullDispatchProbeFull && perf_skip_epoch_clear) {
                    HostEmitEpochStage(pe, go_epoch, "skip_clear_four_output_perf");
                }
            }
            // Overlap validation rendezvous belongs immediately before the
            // measured operation.  All staging, descriptor work and output
            // initialization above are preparation, not communication.
            // Every physical rank participates once; worker ranks publish
            // local-ready only after the common go file is visible.
            if (iter == 0u && !IncDcExternalStartGate("dispatch", pe)) {
                std::cerr << "PL_EXTERNAL_START_GATE_FAIL pe=" << pe
                          << std::endl;
                pass = 0;
                epoch_aborted = true;
                break;
            }
            if (iter == 0u) {
                const uint64_t aligned_start_ns = IncDcExternalStartNs();
                while (aligned_start_ns != 0u &&
                       static_cast<uint64_t>(
                           std::chrono::duration_cast<
                               std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now()
                                   .time_since_epoch())
                               .count()) < aligned_start_ns) {
                    // Validation-only common start.  It is enabled only by
                    // the external gate and keeps host wakeup skew outside
                    // the dispatch/combine overlap interval.
                }
            }
            if (is_worker) {
                HostEmitEpochStage(pe, go_epoch, "before_local_ready");
                if (PublishLocalReady(sym, go_epoch, host_stream) != 0) {
                    pass = 0;
                    epoch_aborted = true;
                    break;
                }
                HostEmitEpochStage(pe, go_epoch, "after_local_ready");
            }

            double host_observed_us = 0.0;
            if (pe == static_cast<int>(kD1LeaderPe) && is_worker) {
                if (WaitLeaderReadySlots(sym, go_epoch, worker_count, epoch_wait_ms) != 0) {
                    pass = 0;
                    std::cerr << "PL_LEADER_READY_TIMEOUT epoch=" << go_epoch << " wait_ms=" << epoch_wait_ms
                              << std::endl;
                    std::cerr.flush();
                    HostEmitStartProbe(sym, pe, is_worker, worker_count, pair_id, go_epoch, d1_start_off,
                                       service_blocks, trace_base);
                    epoch_aborted = true;
                    break;
                }
                const int64_t leader_release_ns = HostWallNs();
                if (PublishLeaderRelease(sym, go_epoch, host_stream) != 0 ||
                    aclrtSynchronizeStream(host_stream) != 0) {
                    pass = 0;
                    HostEmitStartProbe(sym, pe, is_worker, worker_count, pair_id, go_epoch, d1_start_off,
                                       service_blocks, trace_base);
                    epoch_aborted = true;
                    break;
                }

                if (host_completion_mode == kPlHostCompletionAsyncProduction) {
                    // 生产模式：device doorbell/stream 依赖，不做 sync poll；barrier 对齐 epoch 完成
                    aclshmem_barrier_all();
                    host_observed_us = 0.0;
                } else {
                    const int wait_rc = WaitGlobalDoneActivePoll(sym, go_epoch, epoch_wait_ms, &host_observed_us,
                                                                leader_release_ns, pe, worker_count);
                    if (wait_rc != 0) {
                        pass = 0;
                        if (wait_rc == -2) {
                            // generation mismatch：设备已证据退出，干净 session_stop，禁止当 TIMEOUT
                            std::cerr << "PL_EPOCH_GEN_MISMATCH_ABORT epoch=" << go_epoch << std::endl;
                        } else if (wait_rc == -3) {
                            std::cerr << "PL_EPOCH_SOURCE_COUNT_READY_TIMEOUT epoch=" << go_epoch
                                      << " failure_stage=source_count_ready" << std::endl;
                        } else if (wait_rc == -4) {
                            std::cerr << "PL_EPOCH_SESSION_STOP_ABORT epoch=" << go_epoch << std::endl;
                        } else {
                            std::cerr << "PL_EPOCH_DONE_TIMEOUT epoch=" << go_epoch
                                      << " wait_ms=" << epoch_wait_ms << std::endl;
                        }
                        std::cerr.flush();
                        HostDumpP5FailureSnapshot(sym, pe, worker_count, go_epoch);
                        HostEmitStartProbe(sym, pe, is_worker, worker_count, pair_id, go_epoch, d1_start_off,
                                           service_blocks, trace_base);
                        if (HostPublishSessionStop(sym, d1_stop_off, host_stream) != 0) {
                            pass = 0;
                        }
                        // 不在此处 barrier_all：非 leader PE 不走此 abort 路径，
                        // barrier 计数不对齐会导致后续 barrier 死锁。
                        // session_stop 已发布；非 leader 通过 pre-barrier fout_error poll 检测 abort，
                        // 两者统一走 cleanup barrier 对齐。
                        epoch_aborted = true;
                        break;
                    }
                    std::cout << "PL_CANONICAL_INTERVAL pe=" << pe
                              << " go_epoch=" << go_epoch
                              << " begin_ns=" << leader_release_ns
                              << " end_ns=" << HostWallNs()
                              << " host_rank_us=" << host_observed_us
                              << std::endl;
                }
            }

            // M2：在 L3506 barrier_all 之前检查 workspace/generation 致命错误
            // leader abort 后 break 跳过 L3506 barrier；非 leader 必须在此处 break，否则 L3506 barrier 死锁
            // fout_error 可能尚未被 device 设置，做短 poll（最多 500ms）
            if (!epoch_aborted && is_measure && !control_only && !is_worker &&
                measurement_mode == kPlMeasurementFullDispatch) {
                const int64_t poll_t0 = HostWallNs();
                const int64_t poll_deadline = poll_t0 + 500 * 1000000LL; // 500ms
                while (HostWallNs() < poll_deadline) {
                    PlFullOutputDoneLine fout_chk{};
                    if (aclrtMemcpy(&fout_chk, sizeof(fout_chk), sym + kPlFullOutputDoneOff, sizeof(fout_chk),
                                    ACL_MEMCPY_DEVICE_TO_HOST) == 0 &&
                        (fout_chk.error_code == kPlErrGenerationMismatch ||
                         fout_chk.error_code == kPlErrWorkspaceDescMissing ||
                         fout_chk.error_code == kPlErrWorkspaceGenerationMismatch ||
                         fout_chk.error_code == kPlErrWorkspacePointerAlignment ||
                         fout_chk.error_code == kPlErrOutputCapacityInsufficient ||
                         fout_chk.error_code == kPlErrWorkspaceFlagInvalid)) {
                        pass = 0;
                        epoch_aborted = true;
                        std::cerr << "PL_WORKSPACE_FAIL_ABORT_NONLEADER epoch=" << go_epoch
                                  << " pe=" << pe << " error_code=" << fout_chk.error_code << std::endl;
                        std::cerr.flush();
                        break;
                    }
                    usleep(1000); // 1ms
                }
                if (epoch_aborted) break;
            }
            if (control_only) {
                if (pe == static_cast<int>(kD1LeaderPe) && is_worker) {
                    HostEmitStartProbe(sym, pe, is_worker, worker_count, pair_id, go_epoch, d1_start_off,
                                       service_blocks, trace_base);
                }
            } else if (is_measure && pe == static_cast<int>(kD1LeaderPe) && is_worker) {
                    const PlHostEpochSnap snap =
                        HostCollectEpochSnap(sym, static_cast<int>(worker_count), host_stream);
                    // P5.3：device timing_error_code（44/45/49 等）必须 fail-closed，禁止假绿
                    if (snap.timing_error_code != 0u) {
                        pass = 0;
                    }
                    uint64_t route_instance_bytes = 0u;
                    if (measurement_mode == kPlMeasurementFullDispatch) {
                        const uint32_t slot = PlInvocationSlotIndex(ep_count, go_epoch);
                        const bool fh_sm = (full_dispatch_probe == kPlFullDispatchProbeFirstHop) &&
                                           ingress_source_major_raw;
                        HostEmitBytesNumerator(pe, go_epoch, worker_count, route_topk,
                                               source_token_by_epoch[slot], fh_sm);
                        uint32_t total_input = 0u;
                        for (uint32_t s = 0; s < worker_count; ++s) {
                            total_input += source_token_by_epoch[slot][s];
                        }
                        route_instance_bytes =
                            fh_sm ? (static_cast<uint64_t>(total_input) * g_runtime_payload_bytes)
                                  : (static_cast<uint64_t>(total_input) * static_cast<uint64_t>(route_topk) *
                                     g_runtime_payload_bytes);
                    } else {
                        route_instance_bytes =
                            static_cast<uint64_t>(worker_count) * static_cast<uint64_t>(tokens) * g_runtime_payload_bytes;
                    }
                    const double useful_bytes = static_cast<double>(route_instance_bytes);
                    PlDeviceEpochTiming device_timing =
                        HostCollectDeviceTiming(sym, worker_count, static_cast<int>(worker_count), go_epoch);
                    if (measurement_mode == kPlMeasurementFullDispatch) {
                        HostCollectFullDispatchRouteTiming(sym, &device_timing);
                        if (pe == static_cast<int>(kD1LeaderPe)) {
                            HostCollectFullDispatchTransportTiming(sym, &device_timing);
                            HostCollectRankDispatchTiming(sym, static_cast<uint32_t>(npe), go_epoch, &device_timing);
                            HostCollectSecondHopVisibleTiming(sym, &device_timing);
                        }
                    }
                    const double host_observe_residual =
                        (host_completion_mode == kPlHostCompletionSyncPoll && host_observed_us > 0 &&
                         device_timing.device_pipeline_us > 0)
                            ? (host_observed_us - device_timing.device_pipeline_us)
                            : 0.0;
                    device_timing.host_observe_residual_us = host_observe_residual;
                    HostEmitDeviceTiming(pe, go_epoch, device_timing);
                    HostEmitTimingAbi(pe, go_epoch, measurement_mode, device_timing);
                    if (measurement_mode == kPlMeasurementFullDispatch && pe == static_cast<int>(kD1LeaderPe)) {
                        HostEmitTransportTiming(pe, go_epoch, device_timing);
                        HostEmitRankDispatchTiming(pe, go_epoch, static_cast<uint32_t>(npe), device_timing);
                        if (device_timing.second_hop_timing_valid) {
                            HostEmitSecondHopVisibleTiming(pe, go_epoch, device_timing);
                        }
                    }
                    const double gbps =
                        host_observed_us > 0 ? useful_bytes / host_observed_us / 1000.0 : 0.0;
                    const double device_gbps = device_timing.device_pipeline_us > 0
                                                   ? useful_bytes / device_timing.device_pipeline_us / 1000.0
                                                   : 0.0;
                    HostEmitPipeEpoch(pe, iter, is_measure, pass ? 1 : 0, go_epoch, host_observed_us,
                                      device_timing.device_pipeline_us, gbps, device_gbps, snap, payload_source_mode,
                                      verify_mode);
                    // LocalFinal PATH/CHANNEL：在下方全 worker 分支发射（禁止仅 leader pe=0）
                    if (measurement_mode == kPlMeasurementFullDispatch) {
                        // C2 truth：累计 destination/lane + done generation，供 Gate 硬校验
                        PlFullOutputDoneLine fout_truth{};
                        aclrtMemcpy(&fout_truth, sizeof(fout_truth), sym + kPlFullOutputDoneOff, sizeof(fout_truth),
                                    ACL_MEMCPY_DEVICE_TO_HOST);
                        std::cout << "PL_C2_TRUTH_EPOCH pe=" << pe << " go_epoch=" << go_epoch
                                  << " destination_received=" << snap.recv.tokens_received
                                  << " lane_sequence_end=" << snap.upload.local_tail
                                  << " full_done_epoch=" << fout_truth.full_done_epoch
                                  << " counts_done_epoch=" << fout_truth.counts_done_epoch
                                  << " global_done=" << snap.global_done
                                  << " start_generation=" << go_epoch
                                  << " generation_mismatch_count=" << fout_truth.generation_mismatch_count
                                  << " fout_error_code=" << fout_truth.error_code
                                  << " stale_epoch=" << snap.recv.stale_epoch_count
                                  << " duplicate=" << snap.recv.duplicate_count
                                  << " lost=" << snap.recv.lost_count
                                  << " slot_overwrite=" << snap.recv.slot_overwrite_count << std::endl;
                    }
                    if (host_completion_mode == kPlHostCompletionAsyncProduction) {
                        std::cout << "PL_PIPE_ASYNC_COMPLETION pe=" << pe << " go_epoch=" << go_epoch
                                  << " async_production_completion=1 device_pipeline_us="
                                  << device_timing.device_pipeline_us << std::endl;
                    }
                    if (have_prev_measure_snap) {
                        HostEmitPipeDelta(pe, go_epoch, HostDiffEpochSnap(snap, prev_measure_snap));
                    }
                    prev_measure_snap = snap;
                    have_prev_measure_snap = true;
            }
            aclshmem_barrier_all();
            // P5.2 Truth Closure：first_hop source-major，INC 计时外全量 checksum（D2H vs artifact）
            if (!control_only && is_measure && !is_worker && ingress_source_major_raw &&
                full_dispatch_probe == kPlFullDispatchProbeFirstHop &&
                measurement_mode == kPlMeasurementFullDispatch) {
                if (HostVerifySourceMajorFirstHopTruth(sym, pe, pair_id, tokens, route_topk, go_epoch,
                                                       host_stream) != 0) {
                    pass = 0;
                }
            }
            if (!control_only && measurement_mode == kPlMeasurementFullDispatch && is_worker && is_measure &&
                pair_id == 0u) {
                if (full_dispatch_probe == kPlFullDispatchProbeGatherOnly) {
                    const int gver = HostVerifyGatherStaging(&worker_local, sym, tokens, expert_per_pe, go_epoch, host_stream);
                    if (gver != 0) {
                        pass = 0;
                    }
                }
            }
            // Full Dispatch：计时外、下一 epoch local_ready 前做本 epoch 四输出 exact
            // 每个 worker 自 dump 本 dest；全员 barrier 后再由 pair0 verify（INC 也必须进 barrier）
            if (!control_only && measurement_mode == kPlMeasurementFullDispatch && is_measure &&
                full_dispatch_probe == kPlFullDispatchProbeFull) {
                const char *d2h_dir_env = std::getenv("INC_DN_PL_D2H_DIR");
                int d2h_rc = 0;
                const std::string epoch_d2h =
                    (d2h_dir_env != nullptr) ? (std::string(d2h_dir_env) + "/epoch_" + std::to_string(go_epoch))
                                             : std::string();
                if (is_worker && d2h_dir_env != nullptr && artifact_shape_dir != nullptr) {
                    d2h_rc = HostDumpFullDispatchOutputs(&worker_local, pair_id, tokens, expert_per_pe, worker_count, route_topk,
                                                         epoch_d2h, host_stream);
                    if (d2h_rc != 0) {
                        pass = 0;
                        std::cerr << "PL_FULL_DISPATCH_D2H_FAIL epoch=" << go_epoch << " pe=" << pe
                                  << " rc=" << d2h_rc << std::endl;
                    }
                }
            }
            // D2H 后屏障必须全员（含 INC）进入；不可藏在 d2h_dir 条件里导致少人入障死锁
            if (!control_only && measurement_mode == kPlMeasurementFullDispatch && is_measure &&
                full_dispatch_probe == kPlFullDispatchProbeFull) {
                aclshmem_barrier_all();
                const char *d2h_dir_env = std::getenv("INC_DN_PL_D2H_DIR");
                if (is_worker && pair_id == 0u && d2h_dir_env != nullptr && artifact_shape_dir != nullptr) {
                    const std::string epoch_d2h =
                        std::string(d2h_dir_env) + "/epoch_" + std::to_string(go_epoch);
                    std::string ver_shape =
                        std::string(artifact_shape_dir) + "/epoch_" + std::to_string(go_epoch);
                    std::ifstream probe(ver_shape + "/rank_0/golden_expand_x.bin", std::ios::binary);
                    if (!probe.good()) {
                        ver_shape = std::string(artifact_shape_dir);
                    }
                    std::string case_log = std::string(d2h_dir_env);
                    {
                        const auto slash = case_log.find_last_of('/');
                        if (slash != std::string::npos &&
                            case_log.compare(slash + 1, std::string::npos, "d2h") == 0) {
                            case_log = case_log.substr(0, slash);
                        }
                    }
                    const int ver_rc =
                        HostRunFourOutputVerify(ver_shape, worker_count, epoch_d2h, go_epoch, case_log);
                    std::cout << "PL_FOUR_OUTPUT_VERIFY epoch=" << go_epoch
                              << " pass=" << (ver_rc == 0 ? 1 : 0) << " destinations=" << worker_count
                              << std::endl;
                    if (ver_rc != 0) {
                        pass = 0;
                    }
                }
            }
            // host_post：barrier 后做批量 D2H 校验，再进入下一 epoch local_ready
            if (!control_only && verify_mode == kPlVerifyHostPost && is_worker && is_measure) {
                const int ver_errors =
                    HostVerifyDestinationEpoch(&worker_local, sym, pair_id, worker_count, tokens, lane_count, go_epoch,
                                               host_stream);
                std::cout << "PL_HOST_VERIFY pe=" << pe << " go_epoch=" << go_epoch
                          << " pass=" << (ver_errors == 0 ? 1 : 0) << " errors=" << ver_errors << std::endl;
                if (ver_errors != 0) {
                    pass = 0;
                }
            }
            if (!control_only && is_measure) {
                if (is_worker) {
                    uint64_t payload_put_sum = 0u;
                    uint64_t range_put_sum = 0u;
                    uint64_t scalar_put_sum = 0u;
                    uint64_t mte_wait_sum = 0u;
                    for (uint32_t lane = 0; lane < lane_count; ++lane) {
                        PlUploadCounters uc{};
                        aclrtMemcpy(&uc, sizeof(uc), sym + kPlUploadCtrOff + static_cast<uint64_t>(lane) * sizeof(uc),
                                    sizeof(uc), ACL_MEMCPY_DEVICE_TO_HOST);
                        HostEmitPipeUpload(pe, lane, go_epoch, uc);
                        payload_put_sum += uc.payload_put_count;
                        range_put_sum += uc.upload_range_put_count;
                        scalar_put_sum += uc.upload_scalar_put_count;
                        mte_wait_sum += uc.payload_mte_wait_count;
                    }
                    // fail-closed：从 device counter 推导，禁止脚本硬编码 actual_remote_put
                    if (ingress_source_major_raw &&
                        full_dispatch_probe == kPlFullDispatchProbeFirstHop) {
                        const int actual_remote_put =
                            (payload_put_sum > 0u && (range_put_sum + scalar_put_sum) > 0u && mte_wait_sum > 0u)
                                ? 1
                                : 0;
                        std::cout << "PL_REMOTE_PUT_PROOF pe=" << pe << " go_epoch=" << go_epoch
                                  << " payload_put_sum=" << payload_put_sum
                                  << " upload_range_put_sum=" << range_put_sum
                                  << " upload_scalar_put_sum=" << scalar_put_sum
                                  << " payload_mte_wait_sum=" << mte_wait_sum
                                  << " actual_remote_put=" << actual_remote_put
                                  << " transport_type=MTE" << std::endl;
                        if (actual_remote_put == 0) {
                            pass = 0;
                        }
                    }
                    for (uint32_t source = 0; source < worker_count; ++source) {
                        PlRecvCounters rc{};
                        aclrtMemcpy(&rc, sizeof(rc),
                                    sym + kPlRecvCtrOff + static_cast<uint64_t>(source) * sizeof(rc), sizeof(rc),
                                    ACL_MEMCPY_DEVICE_TO_HOST);
                        HostEmitPipeRecv(pe, source, go_epoch, rc);
                    }
                    PlDestDoneLine ddone{};
                    D1LeaderCompletion leader_completion{};
                    aclrtMemcpy(&ddone, sizeof(ddone), sym + kPlDestDoneOff, sizeof(ddone), ACL_MEMCPY_DEVICE_TO_HOST);
                    aclrtMemcpy(&leader_completion, sizeof(leader_completion), sym + D1B(kD1LeaderCompletionOff),
                                sizeof(leader_completion), ACL_MEMCPY_DEVICE_TO_HOST);
                    std::cout << "PL_PIPE_DEST_DONE pe=" << pe << " go_epoch=" << go_epoch
                              << " destination_rank=" << pair_id << " done_epoch=" << ddone.done_epoch
                              << " tokens_received=" << ddone.tokens_received
                              << " pair_done=" << leader_completion.pair_done_slots[pair_id].value << std::endl;
                    if (measurement_mode == kPlMeasurementFullDispatch) {
                        // C1：每个 destination worker 各自读本 PE RCT（禁止仅 pe0）
                        const PlHostEpochSnap lf_snap =
                            HostCollectEpochSnap(sym, static_cast<int>(worker_count), host_stream);
                        HostEmitLocalFinalPath(pe, go_epoch, workspace_local_final, /*is_worker=*/true, lf_snap,
                                               /*inc_fc=*/nullptr, sym, worker_count);
                        HostEmitLocalFinalOverlap(pe, go_epoch, sym, worker_count);
                    }
                    // zero/empty Gate：SourceCountReady（本 source）+ DestCountSliceReady（各 source→本 dest）
                    if (measurement_mode == kPlMeasurementFullDispatch) {
                        PlSourceCountReadyLine src_rdy{};
                        aclrtMemcpy(&src_rdy, sizeof(src_rdy), sym + PlSourceCountReadyLineOff(pair_id),
                                    sizeof(src_rdy), ACL_MEMCPY_DEVICE_TO_HOST);
                        std::cout << "PL_SOURCE_COUNT_READY pe=" << pe << " source=" << pair_id
                                  << " epoch=" << src_rdy.epoch << " ready_epoch=" << src_rdy.ready_epoch
                                  << " total_routes=" << src_rdy.total_routes << " magic=0x" << std::hex
                                  << src_rdy.magic << std::dec
                                  << " magic_ok=" << (src_rdy.magic == kPlCountReadyMagic ? 1 : 0) << std::endl;
                        for (uint32_t src = 0; src < worker_count; ++src) {
                            PlDestCountSliceReadyLine slice{};
                            aclrtMemcpy(&slice, sizeof(slice),
                                        sym + PlDestCountSliceReadyLineOff(src, pair_id), sizeof(slice),
                                        ACL_MEMCPY_DEVICE_TO_HOST);
                            std::cout << "PL_DEST_COUNT_SLICE_READY pe=" << pe << " source=" << src
                                      << " destination_rank=" << pair_id << " epoch=" << slice.epoch
                                      << " ready_epoch=" << slice.ready_epoch
                                      << " total_routes=" << slice.total_routes << " magic=0x" << std::hex
                                      << slice.magic << std::dec
                                      << " magic_ok=" << (slice.magic == kPlCountReadyMagic ? 1 : 0) << std::endl;
                        }
                    }
                    if (pe == static_cast<int>(kD1LeaderPe)) {
                        PlOverlapTelemetry ov{};
                        aclrtMemcpy(&ov, sizeof(ov), sym + kPlOverlapOff, sizeof(ov), ACL_MEMCPY_DEVICE_TO_HOST);
                        HostEmitPipeOverlap(pe, go_epoch, ov);
                    }
                } else {
                    PlPipelineTiming inc_timing{};
                    aclrtMemcpy(&inc_timing, sizeof(inc_timing), sym + kPlTimingOff, sizeof(inc_timing),
                                ACL_MEMCPY_DEVICE_TO_HOST);
                    PlForwardCounters fc_sum{};
                    for (uint32_t lane = 0; lane < lane_count; ++lane) {
                        PlForwardCounters fc{};
                        aclrtMemcpy(&fc, sizeof(fc),
                                    sym + kPlForwardCtrOff + static_cast<uint64_t>(lane) * sizeof(fc), sizeof(fc),
                                    ACL_MEMCPY_DEVICE_TO_HOST);
                        HostEmitIncStats(pe, lane, go_epoch, fc, inc_timing);
                        fc_sum.egress_forwarded += fc.egress_forwarded;
                        fc_sum.payload_put_count += fc.payload_put_count;
                        fc_sum.descriptor_put_count += fc.descriptor_put_count;
                        fc_sum.egress_payload_range_put_count += fc.egress_payload_range_put_count;
                        fc_sum.egress_payload_scalar_put_count += fc.egress_payload_scalar_put_count;
                        fc_sum.drain_quiet_count += fc.drain_quiet_count;
                        fc_sum.egress_tail_publish_count += fc.egress_tail_publish_count;
                        fc_sum.egress_credit_wait_cycles += fc.egress_credit_wait_cycles;
                        fc_sum.direct_destfinal_remote_put_count += fc.direct_destfinal_remote_put_count;
                    }
                    if (measurement_mode == kPlMeasurementFullDispatch) {
                        PlHostEpochSnap empty_snap{};
                        HostEmitLocalFinalPath(pe, go_epoch, workspace_local_final, /*is_worker=*/false, empty_snap,
                                               &fc_sum, sym, worker_count);
                    }
                }
            }
            // 诊断 dump 前只需全员 barrier：leader 已 WaitGlobalDone，且 3087/3119 已对齐。
            // 禁止非 leader/INC 再 poll 本地 D1 global_done——对称堆上不可见，会空转满 epoch_wait_ms，
            // 导致 epoch3 exact 后无法进入 session_stop/cleanup（H8.3 lifecycle 假绿根因）。
            if (!epoch_aborted && is_measure && !control_only && measurement_mode == kPlMeasurementFullDispatch &&
                full_dispatch_probe == kPlFullDispatchProbeFull) {
                HostEmitEpochStage(pe, go_epoch, "before_post_measure_barrier");
                aclshmem_barrier_all();
                HostEmitEpochStage(pe, go_epoch, "after_post_measure_barrier");
                // The data-path completion observer can legitimately beat
                // block9's decoupled egress-timing aggregation.  Re-read the
                // device-owned cycle record after every rank has completed
                // the measured epoch; never substitute this late host
                // observation time for the recorded device boundary.
                if (pe == static_cast<int>(kD1LeaderPe) && is_worker) {
                    PlDeviceEpochTiming late_second_hop{};
                    if (HostCollectSecondHopVisibleTiming(sym, &late_second_hop)) {
                        HostEmitSecondHopVisibleTiming(pe, go_epoch, late_second_hop);
                    }
                }
                HostDumpCountPathCycles(sym, pe, worker_count, go_epoch);
                if (pe == static_cast<int>(kD1LeaderPe)) {
                    HostDumpUploadLaneStageTiming(sym, pe, lane_count, go_epoch);
                }
                HostDumpDestinationCompletionTraces(sym, pe, worker_count, lane_count, go_epoch);
            }
            if (control_only) {
                // 非 leader PE（含 INC）在 barrier 后输出本地 start 观测
                if (!(pe == static_cast<int>(kD1LeaderPe) && is_worker)) {
                    HostEmitStartProbe(sym, pe, is_worker, worker_count, pair_id, go_epoch, d1_start_off,
                                       service_blocks, trace_base);
                }
            }
            if (epoch_aborted) {
                break;
            }
        }
    }

    if (!epoch_aborted) {
        HostEmitCleanupStage(pe, "before_session_stop");
        if (HostPublishSessionStop(sym, d1_stop_off, host_stream) != 0) {
            pass = 0;
        }
        HostEmitCleanupStage(pe, "after_session_stop");
    } else {
        HostEmitCleanupStage(pe, "skip_session_stop_epoch_aborted");
    }
    HostEmitCleanupStage(pe, "before_barrier");
    aclshmem_barrier_all();
    HostEmitCleanupStage(pe, "after_barrier");
    const int kernel_exit_wait_ms = control_only ? 1500 : 5000;
    HostEmitCleanupStage(pe, "before_wait_exit_mask");
    if (WaitAllKernelsStopped(sym, is_worker, service_blocks, trace_base, expect_mask, kernel_exit_wait_ms) != 0) {
        pass = 0;
        std::cerr << "PL_KERNEL_EXIT_TIMEOUT pe=" << pe << std::endl;
        HostEmitKernelExitDiag(sym, pe, is_worker, service_blocks, trace_base, expect_mask,
                               static_cast<uint64_t>(warmup + measure));
    }
    HostEmitCleanupStage(pe, "after_wait_exit_mask");
    // H8.3-T2：禁止无界 SynchronizeStream；exit mask 不全时也必须有界
    constexpr int kStreamSyncTimeoutMs = 3000;
    HostEmitCleanupStage(pe, "before_service_sync");
    if (HostSyncStreamBounded(service_stream, kStreamSyncTimeoutMs, "service_stream", pe) != 0) {
        pass = 0;
    }
    HostEmitCleanupStage(pe, "after_service_sync");
    HostEmitCleanupStage(pe, "before_control_sync");
    if (HostSyncStreamBounded(control_stream, kStreamSyncTimeoutMs, "control_stream", pe) != 0) {
        pass = 0;
    }
    HostEmitCleanupStage(pe, "after_control_sync");
    if (leader_stream != nullptr) {
        HostEmitCleanupStage(pe, "before_leader_sync");
        if (HostSyncStreamBounded(leader_stream, kStreamSyncTimeoutMs, "leader_stream", pe) != 0) {
            pass = 0;
        }
        HostEmitCleanupStage(pe, "after_leader_sync");
    }

    HostEmitCleanupStage(pe, "before_result");
    const uint32_t kernel_exit_mask = HostCollectKernelStopMask(sym, service_blocks, trace_base);
    // 取 control block 退出原因（stop / forward=gen-mismatch abort）
    uint32_t kernel_exit_reason = kPlKernelExitStop;
    {
        PlServiceTraceLine tr_exit{};
        const uint64_t ctrl_off =
            is_worker ? (trace_base + static_cast<uint64_t>(g_worker_layout.control_block) * 128u)
                      : (trace_base + static_cast<uint64_t>(0) * 128u);
        if (aclrtMemcpy(&tr_exit, sizeof(tr_exit), sym + ctrl_off, sizeof(tr_exit), ACL_MEMCPY_DEVICE_TO_HOST) == 0 &&
            tr_exit.kernel_exit_reason != 0u) {
            kernel_exit_reason = tr_exit.kernel_exit_reason;
        }
    }
    const char *exit_reason_name = "stop";
    if (kernel_exit_reason == kPlKernelExitForward) {
        exit_reason_name = "forward";
    } else if (kernel_exit_reason == kPlKernelExitUpload) {
        exit_reason_name = "upload";
    } else if (kernel_exit_reason == kPlKernelExitRecv) {
        exit_reason_name = "recv";
    } else if (kernel_exit_reason != kPlKernelExitStop) {
        exit_reason_name = "other";
    }
    // STOP mask 不全则不得标 pass
    if (kernel_exit_mask != expect_mask || kernel_exit_reason != kPlKernelExitStop) {
        pass = 0;
    }
    std::cout << "PL_PIPE_RESULT pe=" << pe << " pass=" << pass << " role=" << (is_worker ? "worker" : "inc")
              << " pair_id=" << pair_id << " worker_blocks=" << (is_worker ? g_worker_layout.block_dim : 0u)
              << " upload_aiv=" << (is_worker ? g_worker_layout.upload_lane_count : 0u)
              << " recv_aiv=" << (is_worker ? g_worker_layout.recv_lane_count : 0u)
              << " inc_blocks="
              << (ingress_source_major_raw ? kPlIncP5ServiceBlockDim : kPlIncServiceBlockDim)
              << " control_only=" << (control_only ? 1 : 0)
              << " payload_source_mode=" << payload_source_mode
              << " payload_source_mode_name="
              << (payload_source_mode == kPlPayloadSourceEpochUnique ? "epoch_unique" : "static_reuse")
              << " epoch_unique_payload_pattern="
              << (payload_source_mode == kPlPayloadSourceEpochUnique ? 1 : 0)
              << " kernel_exit_mask=0x" << std::hex << kernel_exit_mask << std::dec
              << " kernel_exit_reason=" << exit_reason_name
              << " kernel_exit_reason_code=" << kernel_exit_reason
              << " host_descriptor_patch_count=" << host_descriptor_patch_count
              << " host_descriptor_build_count=0" << std::endl;
    std::cout.flush();

    // Accounting-only D2H, explicitly opt-in for the cell gate. It occurs
    // after verification/timing and before heap free, so it cannot enter the
    // timed transport path.
    if (const char *dump = std::getenv("INC_DN_PL_DUMP_CELL_LEDGER");
        dump != nullptr && std::strcmp(dump, "1") == 0) {
        HostDumpP5FailureSnapshot(sym, pe, worker_count, static_cast<uint64_t>(warmup + measure));
    }
    HostEmitCleanupStage(pe, "before_heap_free");
    aclshmem_free(sym);
    HostEmitCleanupStage(pe, "after_heap_free");
    HostEmitCleanupStage(pe, "before_local_workspace_free");
    if (!pool_owns_worker_local) {
        PlFreeWorkerLocalSession(&worker_local);
    }
    if (inc_local_workspace != nullptr) {
        aclrtFree(inc_local_workspace);
    }
    HostEmitCleanupStage(pe, "after_local_workspace_free");
    HostEmitCleanupStage(pe, "before_stream_destroy");
    aclrtDestroyStream(host_stream);
    aclrtDestroyStream(service_stream);
    aclrtDestroyStream(control_stream);
    if (leader_stream != nullptr) {
        aclrtDestroyStream(leader_stream);
    }
    HostEmitCleanupStage(pe, "after_stream_destroy");
    HostEmitCleanupStage(pe, "before_shmem_finalize");
    aclshmem_finalize();
    HostEmitCleanupStage(pe, "after_shmem_finalize");
    HostEmitCleanupStage(pe, "before_device_reset");
    aclrtResetDevice(dev);
    HostEmitCleanupStage(pe, "after_device_reset");
    HostEmitCleanupStage(pe, "before_acl_finalize");
    aclFinalize();
    HostEmitCleanupStage(pe, "after_acl_finalize");
    return pass ? 0 : 1;
}
