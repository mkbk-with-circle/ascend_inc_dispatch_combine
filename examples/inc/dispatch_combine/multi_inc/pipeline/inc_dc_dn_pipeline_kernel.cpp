/**
 * INC Dispatch 完整两跳 persistent pipeline kernel（destination-major channel）。
 */
#include "inc_dc_dn_pipeline_device.h"
#include "inc_dc_dn_pipeline_full_dispatch_device.h"
#include "inc_dc_dn_pipeline_source_major_device.h"
#include "inc_dc_dn_transport_abi.h"
#include "inc_dc_gather_mte_aicore.h"

using inc::dc::dn::D1StartLine;
using inc::dc::dn::D1SessionStopLine;
using inc::dc::dn::D1ControlLine;
using inc::dc::dn::D1LeaderCompletion;
using inc::dc::dn::D1PublishScratch;
using inc::dc::dn::kD1SessionStopMagic;
using inc::dc::dn::kD1LeaderPe;
using inc::dc::dn::pl::PlIsSourceMajorRaw;
using inc::dc::dn::pl::PlWorkerSourceMajorRawUploadEpoch;
using inc::dc::dn::pl::PlIncSourceMajorReclaimControlEpoch;
using inc::dc::dn::pl::PlIncSourceMajorGatherProducerEpoch;
using inc::dc::dn::pl::PlIncSourceMajorEgressSenderEpoch;
using inc::dc::dn::pl::ReadEgressTailEpochValue;
using inc::dc::dn::pl::DevPlIncP5EgressDoneOff;
using inc::dc::dn::pl::PlEgressDoneLine;
using inc::dc::dn::pl::kPlEgressDoneMagic;
using inc::dc::dn::pl::kPlErrEpochLocalEgressWait;
using inc::dc::dn::pl::kPlIncP5GatherBlockBegin;
using inc::dc::dn::pl::kPlIncP5GatherBlockEnd;
using inc::dc::dn::pl::kPlIncP5EgressBlockBegin;
using inc::dc::dn::pl::kPlIncP5EgressBlockEnd;
using inc::dc::dn::pl::kPlIncP5IngressReclaimBlock;
using inc::dc::dn::pl::kPlIncP5CountPrefixBlock;
using inc::dc::dn::pl::kPlIncP5SecondHopAggBlock;
using inc::dc::dn::pl::kPlIncP5ServiceBlockDim;
using inc::dc::dn::pl::kPlRawUploadLaneCount;
using inc::dc::dn::pl::kQv2LaneCount;
using inc::dc::dn::pl::kPlIncP5ExtResidentOff;
using inc::dc::dn::pl::kPlIncP5ExtTraceOff;
using inc::dc::dn::pl::kPlControlEpochRetireOff;
using inc::dc::dn::pl::kPlControlEpochRetireMagic;
using inc::dc::dn::pl::ControlEpochRetireLine;
using inc::dc::dn::pl::DevPlIncP5ExtResidentLineOff;
using inc::dc::dn::pl::DevPlIncP5ExtTraceLineOff;
using inc::dc::dn::pl::PlDestDoneLine;
using inc::dc::dn::pl::PlForwardCounters;
using inc::dc::dn::pl::PlOverlapTelemetry;
using inc::dc::dn::pl::PlPipelineDesc;
using inc::dc::dn::pl::PlPipelineTiming;
using inc::dc::dn::pl::PlRecvCounters;
using inc::dc::dn::pl::PlResidentLine;
using inc::dc::dn::pl::PlServiceTraceLine;
using inc::dc::dn::pl::PlSourceDoneLine;
using inc::dc::dn::pl::PlForwardDoneLine;
using inc::dc::dn::pl::kPlTimingNegMagic;
using inc::dc::dn::pl::kPlTimingNegOff;
using inc::dc::dn::pl::PlTimingNegLine;
using inc::dc::dn::pl::PlApplyTimingNegDelay;
using inc::dc::dn::pl::PlFirstHopDoneLine;
using inc::dc::dn::pl::kPlDispatchAssistFields;
using inc::dc::dn::pl::kPlDispatchAssistStrideBytes;
using inc::dc::dn::pl::DevPlFirstHopDoneLineOff;
using inc::dc::dn::pl::PlAivEpochTimingLine;
using inc::dc::dn::pl::kPlCompletionModeCounterScan;
using inc::dc::dn::pl::kPlCompletionModeDoneLines;
using inc::dc::dn::pl::DevPlUploadAivTimingOff;
using inc::dc::dn::pl::DevPlRecvAivTimingOff;
using inc::dc::dn::pl::DevPlForwardDoneLineOff;
using inc::dc::dn::pl::DevPlWorkerDescLaneRelOff;
using inc::dc::dn::pl::PlClearTrace128;
using inc::dc::dn::pl::PlClearTrace256;
using inc::dc::dn::pl::PlClearTrace384;
using inc::dc::dn::pl::PlDcciTrace128;
using inc::dc::dn::pl::PlDcciTrace256;
using inc::dc::dn::pl::PlDcciTrace384;
using inc::dc::dn::pl::DevPlForwardAivTimingOff;
using inc::dc::dn::pl::DevPlDestinationCompletionTraceOff;
using inc::dc::dn::pl::DevPlRecvChannelCompletionTraceOff;
using inc::dc::dn::pl::DevPlIncForwardLaneStageTimingOff;
using inc::dc::dn::pl::PlDestinationCompletionTrace;
using inc::dc::dn::pl::PlRecvChannelCompletionTrace;
using inc::dc::dn::pl::PlIncForwardLaneStageTiming;
using inc::dc::dn::pl::kPlDestCompletionTraceMagic;
using inc::dc::dn::pl::kPlRecvChanCompletionTraceMagic;
using inc::dc::dn::pl::kPlIncFwdLaneStageMagic;
using inc::dc::dn::pl::PlTokenReadyLine;
using inc::dc::dn::pl::PlUploadCounters;
using inc::dc::dn::pl::kPlDestDoneOff;
using inc::dc::dn::pl::kPlDestFinalPayloadOff;
using inc::dc::dn::pl::kPlWorkerUploadSrcOff;
using inc::dc::dn::pl::kPlWorkerGatherDescStagingOff;
using inc::dc::dn::pl::kPlIncResidentOff;
using inc::dc::dn::pl::kPlIncServiceBlockDim;
using inc::dc::dn::pl::kPlErrGenerationMismatch;
using inc::dc::dn::pl::kPlErrWorkspaceDescMissing;
using inc::dc::dn::pl::kPlErrWorkspaceGenerationMismatch;
using inc::dc::dn::pl::kPlErrWorkspacePointerAlignment;
using inc::dc::dn::pl::kPlErrOutputCapacityInsufficient;
using inc::dc::dn::pl::kPlErrWorkspaceFlagInvalid;
using inc::dc::dn::pl::kPlKernelExitDestAgg;
using inc::dc::dn::pl::kPlKernelExitForward;
using inc::dc::dn::pl::kPlKernelExitRecv;
using inc::dc::dn::pl::kPlKernelExitResource;
using inc::dc::dn::pl::kPlKernelExitStop;
using inc::dc::dn::pl::kPlKernelExitUpload;
using inc::dc::dn::pl::kPlMagic;
using inc::dc::dn::pl::kPlSourceDoneOff;
using inc::dc::dn::pl::kPlVerifyDeviceFull;
using inc::dc::dn::pl::kPlMeasurementFullDispatch;
using inc::dc::dn::pl::PlFullDispatchConfig;
using inc::dc::dn::pl::PlFullOutputDoneLine;
using inc::dc::dn::pl::PlLaneWorkLine;
using inc::dc::dn::pl::PlRecvDoneLine;
using inc::dc::dn::pl::PlP5CellRouteRecvLedgerLine;
using inc::dc::dn::pl::kPlP5CellLedgerMagic;
using inc::dc::dn::pl::DevPlP5CellRouteRecvLedgerLineOff;
using inc::dc::dn::pl::PlRouteTimingLine;
using inc::dc::dn::pl::kPlFullDispatchConfigOff;
using inc::dc::dn::pl::kPlFullDispatchProbeCountOnly;
using inc::dc::dn::pl::kPlExpectedCountOff;
using inc::dc::dn::pl::kPlRecvDoneMagic;
using inc::dc::dn::pl::PlDevRecvDoneLineOff;
using inc::dc::dn::pl::PlPublishTransportDone;
using inc::dc::dn::pl::PlPe0PublishGlobalTransportTiming;
using inc::dc::dn::pl::PlGlobalTransportResult;
using inc::dc::dn::pl::kPlGlobalTransportOk;
using inc::dc::dn::pl::kPlGlobalTransportMissing;
using inc::dc::dn::pl::kPlGlobalTransportPeerError;
using inc::dc::dn::pl::kPlGlobalTransportReleaseMissing;
using inc::dc::dn::pl::kPlErrRecvIncomplete;
using inc::dc::dn::pl::kPlErrRecvTailEpochMissing;
using inc::dc::dn::pl::kPlErrRecvTailRegression;
using inc::dc::dn::pl::kPlErrRecvTailIncomplete;
using inc::dc::dn::pl::kPlErrRecvDescriptorIncomplete;
using inc::dc::dn::pl::kPlErrGlobalTransportMissing;
using inc::dc::dn::pl::kPlErrGlobalTransportPeerError;
using inc::dc::dn::pl::PlDevRankTimingMarkStart;
using inc::dc::dn::pl::PlDevRankTimingMarkUploadDone;
using inc::dc::dn::pl::PlDevRankTimingMarkForwardDone;
using inc::dc::dn::pl::PlDevRankTimingMarkDestinationDone;
using inc::dc::dn::pl::PlDevRankTimingPublishLocalCompletion;
using inc::dc::dn::pl::kPlRoleMaskWorker;
using inc::dc::dn::pl::kPlRoleMaskInc;
using inc::dc::dn::pl::kPlRoleMaskPe0Observer;
using inc::dc::dn::pl::kPlMaxSources;
using inc::dc::dn::pl::kPlMaxLogicalRecvChannels;
using inc::dc::dn::pl::kPlMaxRecvLanes;
using inc::dc::dn::pl::PlIncComputeEgressExpectedPerDest;
using inc::dc::dn::pl::PlResolvePe0ReleaseSeenCycle;
using inc::dc::dn::pl::PlIncForwardLanePublishEgressVisible;
using inc::dc::dn::pl::PlIncForwardLanePublishOwnedZeroDestAtEntry;
using inc::dc::dn::pl::PlIncMergePublishEgressVisibleDest;
using inc::dc::dn::pl::PlIncComputeEgressExpectedPerDest;
using inc::dc::dn::pl::PlResolvePe0ReleaseSeenCycle;
using inc::dc::dn::pl::PlIncSecondHopAggPublishSummary;
using inc::dc::dn::pl::PlPe0TryPublishGlobalSecondHopVisibleTiming;
using inc::dc::dn::pl::PlFullDispatchInvocation;
using inc::dc::dn::pl::PlFullDispatchInvocationWorkspace;
using inc::dc::dn::pl::PlInvocationWorkspaceDesc;
using inc::dc::dn::pl::PlInvocationDesc;
using inc::dc::dn::pl::PlFullDispatchIncForwardCounts;
using inc::dc::dn::pl::kPlCountFwdOk;
using inc::dc::dn::pl::PlFullDispatchCountPrefix;
using inc::dc::dn::pl::PlFullDispatchDestCountPrefix;
using inc::dc::dn::pl::PlFullDispatchSourcePublishCounts;
using inc::dc::dn::pl::PlFullDispatchSourcePublishRouteMeta;
using inc::dc::dn::pl::PlIncEpochTokensExpected;
using inc::dc::dn::pl::kPlFullDispatchProbeFull;
using inc::dc::dn::pl::kPlFullDispatchProbeGatherOnly;
using inc::dc::dn::pl::kPlFullDispatchProbeFirstHop;
using inc::dc::dn::pl::kPlModeControlOnly;
using inc::dc::dn::pl::kPlModeBaseMask;
using inc::dc::dn::pl::kPlModeUploadCompactGather;
using inc::dc::dn::pl::kPlPayloadSourceStatic;
using inc::dc::dn::pl::kPlPayloadSourceEpochUnique;
using inc::dc::dn::pl::kPlWorkerControlBlock;
using inc::dc::dn::pl::kPlWorkerDescOff;
using inc::dc::dn::pl::kPlWorkerRecvBlockBegin;
using inc::dc::dn::pl::kPlWorkerResidentOff;
using inc::dc::dn::pl::kPlWorkerServiceBlockDim;
using inc::dc::dn::pl::kPlWorkerSourceOff;
using inc::dc::dn::pl::kPlWorkerUploadBlockBegin;
using inc::dc::dn::pl::kPlWorkerUploadBlockEnd;
using inc::dc::dn::pl::kPlMaxUploadLanes;
using inc::dc::dn::pl::kPlMaxRecvLanes;
using inc::dc::dn::pl::kPlMaxLogicalRecvChannels;
using inc::dc::dn::pl::PlWorkerServiceLayout;
using inc::dc::dn::pl::PlDevWorkerServiceLayout;
using inc::dc::dn::pl::kPlIncControlBlock;
using inc::dc::dn::pl::kPlIncSecondHopAggBlock;
using inc::dc::dn::pl::kPlKernelExitSecondHopAgg;
using inc::dc::dn::pl::kPlIncForwardBlockBegin;
using inc::dc::dn::pl::kPlIncForwardBlockEnd;
using inc::dc::dn::pl::DevPlDestChannelDescOff;
using inc::dc::dn::pl::DevPlDestChannelPayloadOff;
using inc::dc::dn::pl::PlDevWorkspaceIsWorkerLocalFinal;
using inc::dc::dn::pl::PlCopyGmBytes;
using inc::dc::dn::pl::kPlInvocationWorkspaceFlagWorkerLocalFinal;
using inc::dc::dn::pl::DevPlEgressTailLineOff;
using inc::dc::dn::pl::DevPlIncDescLaneOff;
using inc::dc::dn::pl::DevPlIncEgressPublishScratchOff;
using inc::dc::dn::pl::DevPlDestCreditPublishScratchOff;
using inc::dc::dn::pl::DevPlIncEgressCreditLineOff;
using inc::dc::dn::pl::DevPlIncPayloadLaneOff;
using inc::dc::dn::pl::DevPlIngressHeadLineOff;
using inc::dc::dn::pl::DevPlIngressTailLineOff;
using inc::dc::dn::pl::DevPlLaneTokenBase;
using inc::dc::dn::pl::DevPlTokenReadyLineOff;
using inc::dc::dn::pl::kPlWorkerRecvBlockEnd;
using inc::dc::dn::pl::PlDcci;
using inc::dc::dn::pl::PlDcciDoneLine;
using inc::dc::dn::pl::PlPublishAivEpochTiming;
using inc::dc::dn::pl::PlDcciForwardCtr;
using inc::dc::dn::pl::PlDcciRecvCtr;
using inc::dc::dn::pl::kPlRecvCtrOff;
using inc::dc::dn::pl::PlDcciUploadCtr;
using inc::dc::dn::pl::PlPublishEgressHead;
using inc::dc::dn::pl::PlReadDescriptorWithRetry;
using inc::dc::dn::pl::PlTraceDcci;
using inc::dc::dn::pl::PlDescriptor;
using inc::dc::dn::qv2::Qv2HeadLine;
using inc::dc::dn::qv2::Qv2TailLine;
using inc::dc::dn::qv2::kQv2TailNotifySeqOff;
using inc::dc::dn::qv2::kQv2HeadNotifySeqOff;
using inc::dc::dn::pl::PlPublishEgressHeadSignaled;
using inc::dc::dn::pl::PlDescriptor;
using inc::dc::dn::pl::kPlDescriptorBytes;
using inc::dc::dn::pl::PlIssueDescriptorTileP6NoDrain;
using inc::dc::dn::pl::PlIssueDescriptorTileP6NoDrainOff;
using inc::dc::dn::pl::PlIssuePayloadBatchGmSrc;
using inc::dc::dn::qv2::kQv2TokenBytes;
using inc::dc::dn::qv2::Qv2CheckPayload;
using inc::dc::dn::qv2::Qv2DcciRange;
using inc::dc::dn::qv2::Qv2IssuePayloadRangeP6NoDrain;
using inc::dc::dn::qv2::Qv2IssueDescriptorRangeP6NoDrain;
using inc::dc::dn::pl::PlPublishEgressTail;
using inc::dc::dn::pl::PlWaitBudgetExceeded;
using inc::dc::dn::pl::PlSpinWaitSignalLineDcci;
using inc::dc::dn::qv2::Qv2DrainTileDataAndDescriptor;
using inc::dc::dn::qv2::Qv2IssueDescriptorTileP6NoDrain;
using inc::dc::dn::qv2::Qv2IssuePayloadBatchP6NoDrain;
using inc::dc::dn::qv2::Qv2MteUbComplete;
using inc::dc::dn::qv2::Qv2PublishTailOrdered;
using inc::dc::dn::qv2::Qv2DcciLine;

__aicore__ inline bool PlPollStop(__gm__ D1SessionStopLine *stop, __gm__ PlServiceTraceLine *trace)
{
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(stop));
    trace->observed_stop_value = stop->value;
    PlTraceDcci(trace);
    return stop->value == kD1SessionStopMagic;
}

__aicore__ inline uint64_t PlWaitStartPersistent(__gm__ D1StartLine *start, __gm__ D1SessionStopLine *stop,
                                                 uint64_t last_epoch, __gm__ PlServiceTraceLine *trace)
{
    trace->wait_start_enter_cycle = AscendC::GetSystemCycle();
    PlTraceDcci(trace);
    uint32_t spins = 0;
    while (true) {
        AscendC::PipeBarrier<PIPE_ALL>();
        if (PlPollStop(stop, trace)) {
            return 0;
        }
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(start));
        trace->observed_start_value = start->value;
        PlTraceDcci(trace);
        if (start->value > last_epoch) {
            trace->wait_start_exit_cycle = AscendC::GetSystemCycle();
            trace->wait_start_spin_count = spins;
            trace->start_seen_epoch = start->value;
            PlTraceDcci(trace);
            return start->value;
        }
        ++spins;
    }
}

__aicore__ inline void PlPublishPairDone(__gm__ uint8_t *sym, uint64_t leader_completion_off, uint32_t pair_id,
                                         uint64_t epoch, uint64_t publish_scratch_off, int leader_pe)
{
    __gm__ D1PublishScratch *scratch = reinterpret_cast<__gm__ D1PublishScratch *>(sym + publish_scratch_off);
    scratch->epoch = epoch;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(scratch));
    const uint64_t remote_pair_done_off =
        leader_completion_off + offsetof(D1LeaderCompletion, pair_done_slots) +
        static_cast<uint64_t>(pair_id) * sizeof(D1ControlLine);
    const int my_pe = aclshmem_my_pe();
    if (my_pe == leader_pe) {
        __gm__ D1ControlLine *slot = reinterpret_cast<__gm__ D1ControlLine *>(sym + remote_pair_done_off);
        slot->value = epoch;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(slot));
    } else {
        aclshmem_putmem_nbi(reinterpret_cast<__gm__ void *>(sym + remote_pair_done_off),
                            reinterpret_cast<__gm__ void *>(&scratch->epoch), sizeof(uint64_t), leader_pe);
        aclshmem_quiet();
    }
}

__aicore__ inline uint32_t PlDescriptorPayloadSlot(__gm__ PlDescriptor *d, __gm__ PlPipelineDesc *desc)
{
    return desc->measurement_mode == kPlMeasurementFullDispatch ? d->destination_final_slot : d->destination_slot;
}

__aicore__ inline uint32_t PlEpochTokensExpected(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc, uint32_t lane_or_source)
{
    if (desc->measurement_mode != kPlMeasurementFullDispatch) {
        return desc->tokens_per_lane;
    }
    __gm__ PlFullDispatchConfig *cfg =
        reinterpret_cast<__gm__ PlFullDispatchConfig *>(sym + kPlFullDispatchConfigOff);
    const uint64_t lw_off = cfg->lane_work_off + static_cast<uint64_t>(lane_or_source) * 64u;
    __gm__ PlLaneWorkLine *lw = reinterpret_cast<__gm__ PlLaneWorkLine *>(sym + lw_off);
    // INC：须由调用方传入 epoch 调 PlIncEpochTokensExpected
    if (desc->my_role == 1u) {
        return 0u;
    }
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(lw));
    if (lw->token_count > 0u) {
        return lw->token_count;
    }
    // Full dispatch：lane_work 已初始化后 0 表示该 lane 无 token，禁止回退 tokens_per_epoch
    if (desc->measurement_mode == kPlMeasurementFullDispatch) {
        return 0u;
    }
    return desc->tokens_per_epoch;
}

// destination recv：读 expected_count[source]，与 upload outbound lane_work 分离
__aicore__ inline uint32_t PlEpochRecvTokensExpected(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc, uint32_t source_id)
{
    if (desc->measurement_mode != kPlMeasurementFullDispatch || desc->worker_count <= 1u) {
        return PlEpochTokensExpected(sym, desc, source_id);
    }
    __gm__ int32_t *expected = reinterpret_cast<__gm__ int32_t *>(sym + kPlExpectedCountOff);
    Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(expected), desc->worker_count * sizeof(int32_t));
    if (expected[source_id] > 0) {
        return static_cast<uint32_t>(expected[source_id]);
    }
    return 0u;
}

// Worker upload lane：第一跳 Worker→INC
__aicore__ inline void PlWorkerUploadLaneEpoch(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                               __gm__ PlUploadCounters *uc, __gm__ PlPipelineTiming *timing,
                                               __gm__ PlServiceTraceLine *trace, uint32_t lane_id, uint64_t epoch)
{
    const int peer = static_cast<int>(desc->peer_pe);
    const uint32_t nbytes = desc->payload_bytes;
    const uint32_t depth = desc->ring_depth;
    const uint32_t batch = desc->batch_tokens;
    const uint32_t tile = desc->tile_tokens;
    uint32_t tokens_lane = 0u;
    if (desc->measurement_mode == kPlMeasurementFullDispatch) {
        __gm__ PlFullDispatchConfig *fdc =
            reinterpret_cast<__gm__ PlFullDispatchConfig *>(sym + kPlFullDispatchConfigOff);
        __gm__ PlLaneWorkLine *lw = reinterpret_cast<__gm__ PlLaneWorkLine *>(
            sym + fdc->lane_work_off + static_cast<uint64_t>(lane_id) * 64u);
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(lw));
        // gather 完成后优先用实际 gather_count，避免与 lane_work 竞态
        if (lw->done_epoch >= epoch) {
            tokens_lane = lw->gather_count;
        } else {
            tokens_lane = PlEpochTokensExpected(sym, desc, lane_id);
        }
    } else {
        tokens_lane = PlEpochTokensExpected(sym, desc, lane_id);
    }

    trace->epoch_enter_cycle = AscendC::GetSystemCycle();
    PlTraceDcci(trace);

    if (tokens_lane == 0u) {
        PlDcciUploadCtr(uc);
        return;
    }

    __gm__ Qv2HeadLine *head_line =
        reinterpret_cast<__gm__ Qv2HeadLine *>(sym + DevPlIngressHeadLineOff(lane_id));
    __gm__ Qv2TailLine *tail_line =
        reinterpret_cast<__gm__ Qv2TailLine *>(sym + DevPlIngressTailLineOff(lane_id));

    AscendC::PipeBarrier<PIPE_ALL>();
    PlRefreshIngressHeadFromInc(sym, desc, head_line, lane_id, uc);
    if (uc->local_tail == 0) {
        uc->local_tail = uc->cached_remote_head;
    }
    // reclaim 读数不应领先 producer；钳位避免 unsigned credit 下溢死锁
    if (uc->cached_remote_head > uc->local_tail) {
        uc->cached_remote_head = uc->local_tail;
    }

    const uint64_t lane_token_base = DevPlLaneTokenBase(lane_id, tokens_lane);
    const uint64_t inc_payload_base = DevPlIncPayloadLaneOff(lane_id);
    const uint64_t inc_desc_base = DevPlIncDescLaneOff(lane_id);
    const uint64_t remote_tail_off = DevPlIngressTailLineOff(lane_id);
    // static：多 epoch 复用 epoch1 预填 payload；epoch_unique 才按 epoch 偏移 source
    const uint64_t source_epoch_base =
        (desc->payload_source_mode == kPlPayloadSourceStatic)
            ? 0u
            : (epoch - 1u) * static_cast<uint64_t>(desc->tokens_per_epoch);
    __gm__ PlFullDispatchConfig *fdc =
        reinterpret_cast<__gm__ PlFullDispatchConfig *>(sym + kPlFullDispatchConfigOff);
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(fdc));
    const bool upload_from_gather = (desc->measurement_mode == kPlMeasurementFullDispatch);
    const bool upload_compact_gather =
        upload_from_gather && ((desc->pipeline_mode & kPlModeUploadCompactGather) != 0u) &&
        (fdc->gather_payload_off != 0u);

    const uint64_t lane_desc_base =
        fdc->final_payload_off + DevPlWorkerDescLaneRelOff(lane_id);
    // V2：默认 payload 从 Worker local input_ptr；P3 compact 时先收成 gather_payload tile staging
    uint64_t seq = uc->local_tail;
    const uint64_t seq_end = seq + tokens_lane;

    uint64_t payload_puts = 0;
    uint64_t payload_mte = 0;
    uint64_t desc_puts = 0;
    uint64_t desc_mte = 0;
    uint64_t drain_quiet = 0;
    uint64_t tail_pub = 0;
    uint64_t tail_quiet = 0;
    uint32_t wrap_split = 0;
    uint64_t up_batch_count = 0;
    uint64_t up_batch_tokens = 0;
    uint64_t up_scalar = 0;
    uint64_t up_range = 0;
    uint64_t first_tail_pub_cyc = 0;
    uint64_t gather_cycles = 0;

    while (seq < seq_end) {
        const uint32_t tile_tokens = static_cast<uint32_t>((seq + tile > seq_end) ? (seq_end - seq) : tile);
        while (seq + tile_tokens > uc->cached_remote_head + static_cast<uint64_t>(depth)) {
            ++uc->credit_wait_cycles;
            PlRefreshIngressHeadFromInc(sym, desc, head_line, lane_id, uc);
            if (uc->cached_remote_head > seq) {
                uc->cached_remote_head = seq;
            }
        }

        const uint64_t cyc0 = AscendC::GetSystemCycle();
        if (trace->first_issue_cycle == 0) {
            trace->first_issue_cycle = cyc0;
            PlTraceDcci(trace);
        }

        const uint64_t rel_base = seq - (seq_end - tokens_lane);
        uint32_t issued = 0;
        if (upload_compact_gather) {
            // P3：tile 内按 descriptor 序 MTE gather → 连续 staging，再 range put。
            // staging 复用 Worker local gather_payload[lane]（已按 ring_depth 分配；tile≤ring/2）。
            // 禁止 fused 进 gather 环写；与 descriptor ordered staging 解耦。
            if (tile_tokens > depth) {
                uc->error_code = 61; // compact staging overflow
                PlDcciUploadCtr(uc);
                return;
            }
            const uint64_t staging_base =
                fdc->gather_payload_off + static_cast<uint64_t>(lane_id) * static_cast<uint64_t>(depth) * nbytes;
            __gm__ uint8_t *staging_gm = reinterpret_cast<__gm__ uint8_t *>(staging_base);
            const uint64_t raw_base = PlFullDispatchRawInputBase(fdc, desc, sym, epoch);
            __ubuf__ uint8_t *ub = reinterpret_cast<__ubuf__ uint8_t *>(0);
            const uint64_t g0 = AscendC::GetSystemCycle();
            for (uint32_t i = 0; i < tile_tokens; ++i) {
                __gm__ PlDescriptor *wd = reinterpret_cast<__gm__ PlDescriptor *>(
                    reinterpret_cast<__gm__ uint8_t *>(lane_desc_base) + (rel_base + i) * kPlDescriptorBytes);
                AscendC::PipeBarrier<PIPE_ALL>();
                Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(wd), kPlDescriptorBytes);
                __gm__ uint8_t *src =
                    reinterpret_cast<__gm__ uint8_t *>(raw_base + static_cast<uint64_t>(wd->source_token_id) * nbytes);
                DcLlGatherCopyGm2GmChunked(reinterpret_cast<GM_ADDR>(staging_gm + static_cast<uint64_t>(i) * nbytes),
                                          reinterpret_cast<GM_ADDR>(src), nbytes, ub, kDcLlGatherEvent);
            }
            gather_cycles += (AscendC::GetSystemCycle() - g0);
            // MTE GM→UB→GM 后 PipeBarrier 即可；禁止对整 tile 做大块 DCCI（会吞掉 range-put 收益）
            AscendC::PipeBarrier<PIPE_ALL>();
            while (issued < tile_tokens) {
                const uint32_t bt = (issued + batch > tile_tokens) ? (tile_tokens - issued) : batch;
                PlIssuePayloadBatchGmSrc(sym, staging_gm + static_cast<uint64_t>(issued) * nbytes, inc_payload_base,
                                         seq + issued, bt, nbytes, depth, peer, payload_puts, payload_mte, wrap_split);
                if (bt <= 1u) {
                    ++up_scalar;
                } else {
                    ++up_range;
                }
                ++up_batch_count;
                up_batch_tokens += bt;
                issued += bt;
            }
        } else {
            while (issued < tile_tokens) {
                if (upload_from_gather) {
                    // V2：合并 source_token_id 连续的 run，减少 per-token put/MTE wait
                    const uint64_t rel0 = rel_base + issued;
                    __gm__ PlDescriptor *wd0 = reinterpret_cast<__gm__ PlDescriptor *>(
                        reinterpret_cast<__gm__ uint8_t *>(lane_desc_base) + rel0 * kPlDescriptorBytes);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(wd0), kPlDescriptorBytes);
                    uint32_t run = 1u;
                    const uint32_t run_cap = (issued + batch > tile_tokens) ? (tile_tokens - issued) : batch;
                    while (run < run_cap) {
                        __gm__ PlDescriptor *wdn = reinterpret_cast<__gm__ PlDescriptor *>(
                            reinterpret_cast<__gm__ uint8_t *>(lane_desc_base) +
                            (rel0 + run) * kPlDescriptorBytes);
                        Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(wdn), kPlDescriptorBytes);
                        if (wdn->source_token_id != wd0->source_token_id + run) {
                            break;
                        }
                        ++run;
                    }
                    const uint64_t raw_base = PlFullDispatchRawInputBase(fdc, desc, sym, epoch);
                    __gm__ uint8_t *worker_src_gm = reinterpret_cast<__gm__ uint8_t *>(
                        raw_base + static_cast<uint64_t>(wd0->source_token_id) * nbytes);
                    PlIssuePayloadBatchGmSrc(sym, worker_src_gm, inc_payload_base, seq + issued, run, nbytes, depth, peer,
                                             payload_puts, payload_mte, wrap_split);
                    if (run <= 1u) {
                        ++up_scalar;
                    } else {
                        ++up_range;
                    }
                    ++up_batch_count;
                    up_batch_tokens += run;
                    issued += run;
                } else {
                    const uint32_t bt = (issued + batch > tile_tokens) ? (tile_tokens - issued) : batch;
                    const uint64_t worker_src =
                        PlFullDispatchRawInputBase(fdc, desc, sym, epoch) +
                        (source_epoch_base + lane_token_base + rel_base + issued) * nbytes;
                    __gm__ uint8_t *worker_src_gm = reinterpret_cast<__gm__ uint8_t *>(worker_src);
                    PlIssuePayloadBatchGmSrc(sym, worker_src_gm, inc_payload_base, seq + issued, bt, nbytes, depth, peer,
                                             payload_puts, payload_mte, wrap_split);
                    if (bt <= 1u) {
                        ++up_scalar;
                    } else {
                        ++up_range;
                    }
                    ++up_batch_count;
                    up_batch_tokens += bt;
                    issued += bt;
                }
            }
        }

        if (upload_from_gather) {
            const uint64_t worker_desc =
                lane_desc_base + rel_base * kPlDescriptorBytes;
            PlIssueDescriptorTileP6NoDrain(sym, worker_desc, inc_desc_base, seq, tile_tokens, depth, peer, desc_puts,
                                            desc_mte, wrap_split);
        } else {
            const uint64_t worker_desc =
                fdc->final_payload_off + (lane_token_base + rel_base) * kPlDescriptorBytes;
            PlIssueDescriptorTileP6NoDrain(sym, worker_desc, inc_desc_base, seq, tile_tokens, depth, peer, desc_puts,
                                            desc_mte, wrap_split);
        }
        Qv2DrainTileDataAndDescriptor(drain_quiet);

        seq += tile_tokens;
        Qv2PublishTailOrdered(tail_line, sym, remote_tail_off, peer, seq, epoch, tail_pub, tail_quiet);
        if (first_tail_pub_cyc == 0u) {
            first_tail_pub_cyc = AscendC::GetSystemCycle();
        }
    }

    uc->local_tail = seq;
    uc->payload_put_count += payload_puts;
    uc->payload_mte_wait_count += payload_mte;
    uc->descriptor_tile_put_count += desc_puts;
    uc->descriptor_mte_wait_count += desc_mte;
    uc->data_descriptor_drain_quiet_count += drain_quiet;
    uc->tail_publish_count += tail_pub;
    uc->tail_completion_quiet_count += tail_quiet;
    uc->wrap_split_put_count += wrap_split;
    uc->upload_batch_count += up_batch_count;
    uc->upload_batch_tokens_sum += up_batch_tokens;
    uc->upload_scalar_put_count += up_scalar;
    uc->upload_range_put_count += up_range;
    // per-epoch 锚点：覆盖写（host 按 go_epoch 读）
    uc->first_tail_publish_cycle = first_tail_pub_cyc;
    // gather_cycles 暂无独立 ABI 槽；host 用 average_upload_batch / scalar_fraction 作 P3 gate
    (void)gather_cycles;
    PlDcciUploadCtr(uc);
    const uint64_t done_cyc = AscendC::GetSystemCycle();
    const uint64_t first_cyc = trace->first_issue_cycle > 0 ? trace->first_issue_cycle : trace->epoch_enter_cycle;
    PlPublishAivEpochTiming(sym, DevPlUploadAivTimingOff(lane_id), first_cyc, done_cyc, epoch, lane_id);
}

// INC→Worker：只写 FullOutputDone 的 error 字段（offset 40..55），不覆盖 counts/full_done
__aicore__ inline void PlPublishWorkerGenerationMismatch(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                          uint32_t lane_id, uint64_t epoch, uint64_t gen_mm)
{
    __gm__ PlFullDispatchConfig *cfg =
        reinterpret_cast<__gm__ PlFullDispatchConfig *>(sym + kPlFullDispatchConfigOff);
    __gm__ uint32_t *scratch = reinterpret_cast<__gm__ uint32_t *>(sym + DevPlIncEgressPublishScratchOff(lane_id));
    scratch[0] = kPlErrGenerationMismatch;
    scratch[1] = 0u;
    scratch[2] = 0u;
    scratch[3] = static_cast<uint32_t>(gen_mm);
    AscendC::PipeBarrier<PIPE_ALL>();
    Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(scratch), 16u);
    // PlFullOutputDoneLine: error_code@40, generation_mismatch_count@52
    aclshmem_putmem_nbi(sym + cfg->full_output_done_off + 40u, reinterpret_cast<__gm__ uint8_t *>(scratch), 4u,
                        static_cast<int>(desc->peer_pe));
    aclshmem_putmem_nbi(sym + cfg->full_output_done_off + 52u, reinterpret_cast<__gm__ uint8_t *>(scratch + 3), 4u,
                        static_cast<int>(desc->peer_pe));
    aclshmem_quiet();
    (void)epoch;
}

// FIRST_HOP：仅消费 INC ingress（verify + reclaim），禁止第二跳 egress
__aicore__ inline void PlIncIngressOnlyLaneEpoch(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                 __gm__ PlForwardCounters *fc, __gm__ PlServiceTraceLine *trace,
                                                 uint32_t lane_id, uint64_t epoch)
{
    const uint32_t nbytes = desc->payload_bytes;
    const uint32_t depth = desc->ring_depth;
    const uint32_t tokens_lane =
        (desc->measurement_mode == kPlMeasurementFullDispatch)
            ? PlIncEpochTokensExpected(sym, desc, lane_id, epoch)
            : PlEpochTokensExpected(sym, desc, lane_id);
    __gm__ PlFullDispatchConfig *cfg =
        reinterpret_cast<__gm__ PlFullDispatchConfig *>(sym + kPlFullDispatchConfigOff);
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(cfg));
    __gm__ Qv2TailLine *ingress_tail =
        reinterpret_cast<__gm__ Qv2TailLine *>(sym + DevPlIngressTailLineOff(lane_id));
    __gm__ Qv2HeadLine *ingress_head =
        reinterpret_cast<__gm__ Qv2HeadLine *>(sym + DevPlIngressHeadLineOff(lane_id));
    uint64_t ingress_seen = fc->ingress_seen;
    if (desc->measurement_mode == kPlMeasurementFullDispatch) {
        // epoch1 强制从 0 开始；后续 epoch 沿用累计 ingress_seen（C3-B persistent）
        if (epoch == 1u) {
            ingress_seen = 0u;
            fc->ingress_seen = 0u;
        } else {
            ingress_seen = fc->ingress_seen;
        }
    } else if (ingress_seen == 0u) {
        ingress_seen = ingress_head->head;
    }
    const uint64_t ingress_end = ingress_seen + tokens_lane;
    while (ingress_seen < ingress_end) {
        AscendC::PipeBarrier<PIPE_ALL>();
        Qv2DcciLine(reinterpret_cast<__gm__ uint8_t *>(ingress_tail));
        const uint64_t remote_ingress_tail = ingress_tail->tail;
        while (ingress_seen < remote_ingress_tail && ingress_seen < ingress_end) {
            const uint32_t slot0 = static_cast<uint32_t>(ingress_seen % depth);
            __gm__ PlDescriptor *d0 = reinterpret_cast<__gm__ PlDescriptor *>(
                sym + DevPlIncDescLaneOff(lane_id) + static_cast<uint64_t>(slot0) * kPlDescriptorBytes);
            uint64_t stale0 = 0;
            uint64_t dup0 = 0;
            uint64_t lost0 = 0;
            uint64_t gen_mm0 = 0;
            if (!PlReadDescriptorWithRetry(d0, epoch, ingress_seen, slot0, stale0, dup0, lost0, gen_mm0)) {
                fc->error_code = (gen_mm0 > 0) ? kPlErrGenerationMismatch : 41u;
                (void)stale0;
                (void)dup0;
                (void)lost0;
                PlDcciForwardCtr(fc);
                if (gen_mm0 > 0) {
                    PlPublishWorkerGenerationMismatch(sym, desc, lane_id, epoch, gen_mm0);
                }
                return;
            }
            ++ingress_seen;
            ++fc->ingress_seen;
        }
    }
    ingress_head->head = ingress_seen;
    const uint64_t remote_ingress_head_off = DevPlIngressHeadLineOff(lane_id);
    uint64_t head_pub = 0;
    uint64_t head_quiet = 0;
    PlPublishEgressHead(ingress_head, sym, remote_ingress_head_off, static_cast<int>(desc->peer_pe), ingress_seen,
                        epoch, head_pub, head_quiet);
    aclshmem_quiet();
    fc->ingress_head_publish_count += head_pub;
    PlDcciForwardCtr(fc);
    const uint64_t done_cyc = AscendC::GetSystemCycle();
    PlPublishAivEpochTiming(sym, DevPlForwardAivTimingOff(lane_id), done_cyc, done_cyc, epoch, lane_id);
    __gm__ PlForwardDoneLine *fdone =
        reinterpret_cast<__gm__ PlForwardDoneLine *>(sym + DevPlForwardDoneLineOff(lane_id));
    fdone->done_epoch = epoch;
    fdone->lane_id = lane_id;
    fdone->magic = kPlMagic;
    PlDcciDoneLine(reinterpret_cast<__gm__ uint8_t *>(fdone));
    // remote 写回 Worker：用 GM scratch 避免 stack→GM cast
    __gm__ PlFirstHopDoneLine *scratch = reinterpret_cast<__gm__ PlFirstHopDoneLine *>(
        sym + DevPlIncEgressPublishScratchOff(lane_id));
    scratch->epoch = epoch;
    scratch->lane_id = lane_id;
    scratch->tokens_received = tokens_lane;
    scratch->payload_bytes_received = static_cast<uint64_t>(tokens_lane) * nbytes;
    scratch->descriptor_bytes_received = static_cast<uint64_t>(tokens_lane) * kPlDescriptorBytes;
    scratch->ingress_head = ingress_seen;
    scratch->error_code = 0u;
    scratch->magic = kPlMagic;
    PlDcciDoneLine(reinterpret_cast<__gm__ uint8_t *>(scratch));
    const uint64_t worker_dst = cfg->first_hop_done_off + static_cast<uint64_t>(lane_id) * 64u;
    aclshmem_putmem_nbi(sym + worker_dst, reinterpret_cast<__gm__ uint8_t *>(scratch), sizeof(PlFirstHopDoneLine),
                        static_cast<int>(desc->peer_pe));
    aclshmem_quiet();
}

// INC forward lane：第二跳 ingress→destination channel（batch8 fast path）
__aicore__ inline void PlIncForwardLaneEpoch(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                             __gm__ PlForwardCounters *fc, __gm__ PlPipelineTiming *timing,
                                             __gm__ PlOverlapTelemetry *overlap, __gm__ PlServiceTraceLine *trace,
                                             uint32_t lane_id, uint64_t epoch)
{
    (void)timing;
    (void)overlap;
    const uint32_t nbytes = desc->payload_bytes;
    const uint32_t depth = desc->ring_depth;
    // Full dispatch INC 须读 Worker push 的 lane_work；PlEpochTokensExpected 对 my_role==1 恒返回 0
    const uint32_t tokens_lane = (desc->measurement_mode == kPlMeasurementFullDispatch)
                                       ? PlIncEpochTokensExpected(sym, desc, lane_id, epoch)
                                       : PlEpochTokensExpected(sym, desc, lane_id);
    const uint32_t batch_max = desc->batch_tokens > 0u ? desc->batch_tokens : 8u;
    __gm__ PlIncForwardLaneStageTiming *fst = reinterpret_cast<__gm__ PlIncForwardLaneStageTiming *>(
        sym + DevPlIncForwardLaneStageTimingOff(lane_id));
    PlClearTrace256(reinterpret_cast<__gm__ uint8_t *>(fst));
    fst->epoch = epoch;
    fst->lane_id = lane_id;

    // 空 lane：仍发布 egress visible（全 dest 0 route）供 PE0 校验
    if (tokens_lane == 0u) {
        if (desc->measurement_mode == kPlMeasurementFullDispatch) {
            __gm__ PlFullDispatchConfig *zcfg =
                reinterpret_cast<__gm__ PlFullDispatchConfig *>(sym + kPlFullDispatchConfigOff);
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(zcfg));
            if (zcfg->probe_mode == kPlFullDispatchProbeFull) {
                __gm__ PlInvocationDesc *zinv = PlFullDispatchInvocation(sym, zcfg, epoch);
                uint32_t exp_zero[kPlMaxSources];
                uint32_t fwd_zero[kPlMaxSources];
                uint64_t tail_zero[kPlMaxSources];
                for (uint32_t d = 0; d < kPlMaxSources; ++d) {
                    exp_zero[d] = 0u;
                    fwd_zero[d] = 0u;
                    tail_zero[d] = 0u;
                }
                uint32_t exp_dest[kPlMaxSources];
                PlIncComputeEgressExpectedPerDest(sym, desc->pair_id, desc->worker_count, desc->expert_per_pe, epoch,
                                                  exp_dest);
                uint64_t vis_cyc[kPlMaxSources];
                for (uint32_t d = 0; d < kPlMaxSources; ++d) {
                    vis_cyc[d] = 0u;
                }
                const uint64_t lane_cycle_start = AscendC::GetSystemCycle();
                    PlIncForwardLanePublishOwnedZeroDestAtEntry(sym, desc->pair_id, lane_id, desc->worker_count, epoch,
                                                            zinv->generation, exp_dest, lane_cycle_start, lane_cycle_start,
                                                            static_cast<int>(kD1LeaderPe));
            }
        }
        __gm__ PlForwardDoneLine *fdone =
            reinterpret_cast<__gm__ PlForwardDoneLine *>(sym + DevPlForwardDoneLineOff(lane_id));
        fdone->done_epoch = epoch;
        fdone->lane_id = lane_id;
        fdone->magic = kPlMagic;
        PlDcciDoneLine(reinterpret_cast<__gm__ uint8_t *>(fdone));
        fst->forward_done_cycle = AscendC::GetSystemCycle();
        fst->magic = kPlIncFwdLaneStageMagic;
        PlDcciTrace256(reinterpret_cast<__gm__ uint8_t *>(fst));
        return;
    }

    __gm__ Qv2TailLine *ingress_tail =
        reinterpret_cast<__gm__ Qv2TailLine *>(sym + DevPlIngressTailLineOff(lane_id));
    __gm__ Qv2HeadLine *ingress_head =
        reinterpret_cast<__gm__ Qv2HeadLine *>(sym + DevPlIngressHeadLineOff(lane_id));
    __gm__ Qv2TailLine *egress_publish_scratch =
        reinterpret_cast<__gm__ Qv2TailLine *>(sym + DevPlIncEgressPublishScratchOff(lane_id));

    uint64_t ingress_seen = fc->ingress_seen;
    if (desc->measurement_mode == kPlMeasurementFullDispatch) {
        // epoch1 强制从 0 开始；后续 epoch 沿用累计 ingress_seen（C3-B persistent）
        if (epoch == 1u) {
            ingress_seen = 0u;
            fc->ingress_seen = 0u;
        } else {
            ingress_seen = fc->ingress_seen;
        }
    } else if (ingress_seen == 0u) {
        ingress_seen = ingress_head->head;
    }
    const uint64_t ingress_end = ingress_seen + tokens_lane;
    uint64_t egress_seq = ingress_seen;

    uint64_t payload_puts = 0;
    uint64_t payload_mte = 0;
    uint64_t desc_puts = 0;
    uint64_t desc_mte = 0;
    uint64_t drain_quiet = 0;
    uint64_t egress_tail_pub = 0;
    uint64_t egress_tail_quiet = 0;
    uint64_t egress_credit_wait_cycles_local = 0;
    uint64_t tail_publish_cycles_sum = 0;
    uint64_t ingress_head_pub = 0;
    uint64_t ingress_head_quiet = 0;
    uint64_t payload_range_puts = 0;
    uint64_t payload_scalar_puts = 0;
    uint64_t desc_range_puts = 0;
    uint64_t desc_scalar_puts = 0;
    uint64_t batch_count = 0;
    uint64_t batch_tokens = 0;
    uint64_t batch_max_tokens = 0;
    // H5：本 epoch 阶段 cycle accumulator（epoch 末一次性写 fst）
    uint64_t wait_ingress_tail_cycles = 0;
    uint64_t descriptor_validation_cycles = 0;
    uint64_t payload_descriptor_issue_cycles = 0;
    uint64_t mte_completion_cycles = 0;
    uint64_t data_drain_quiet_cycles = 0;
    uint64_t ingress_head_publish_cycles = 0;
    // P8.2 publish amortization 状态（group 上限 = tile=32 < ring=64）
    const uint32_t tile_pub = desc->head_tile_tokens > 0u ? desc->head_tile_tokens : 32u;
    uint32_t pending_publish_tokens = 0u;
    int pending_dest_pe = -1;
    uint32_t pending_source_ch = 0u;
    uint64_t last_visible_egress_seq = egress_seq;
    uint64_t last_reclaimed_ingress_seq = ingress_seen;
    uint32_t publish_group_count = 0u;
    uint32_t publish_group_tokens = 0u;
    uint32_t publish_group_max_tokens = 0u;
    uint32_t partial_publish_group_count = 0u;
    uint32_t forced_wrap_flush_count = 0u;
    uint32_t forced_credit_flush_count = 0u;
    uint32_t upstream_empty_flush_count = 0u;
    uint32_t final_flush_count = 0u;
    uint32_t pending_group_high_watermark = 0u;
    uint32_t first_tail_early_flush_count = 0u; // T1-A REJECT_PERF：字段保留、恒 0
    uint32_t fwd_per_dest[kPlMaxSources];
    uint64_t tail_per_dest[kPlMaxSources];
    uint64_t visible_done_per_dest[kPlMaxSources];
    for (uint32_t d = 0; d < kPlMaxSources; ++d) {
        fwd_per_dest[d] = 0u;
        tail_per_dest[d] = 0u;
        visible_done_per_dest[d] = 0u;
    }

    if (desc->measurement_mode == kPlMeasurementFullDispatch) {
        __gm__ PlFullDispatchConfig *fdc_entry =
            reinterpret_cast<__gm__ PlFullDispatchConfig *>(sym + kPlFullDispatchConfigOff);
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(fdc_entry));
        if (fdc_entry->probe_mode == kPlFullDispatchProbeFull) {
            if (!PlDevRequireInvocationWorkspace(sym, fdc_entry, epoch)) {
                __gm__ PlFullOutputDoneLine *fout_ws =
                    reinterpret_cast<__gm__ PlFullOutputDoneLine *>(sym + fdc_entry->full_output_done_off);
                PlDcci(reinterpret_cast<__gm__ uint8_t *>(fout_ws));
                fc->error_code = fout_ws->error_code;
                PlDcciForwardCtr(fc);
                return;
            }
            __gm__ PlInvocationDesc *inv_entry = PlFullDispatchInvocation(sym, fdc_entry, epoch);
            uint32_t exp_entry[kPlMaxSources];
            PlIncComputeEgressExpectedPerDest(sym, desc->pair_id, desc->worker_count, desc->expert_per_pe, epoch,
                                              exp_entry);
            const uint64_t lane_cycle_start = AscendC::GetSystemCycle();
            PlIncForwardLanePublishOwnedZeroDestAtEntry(sym, desc->pair_id, lane_id, desc->worker_count, epoch,
                                                        inv_entry->generation, exp_entry, lane_cycle_start, lane_cycle_start,
                                                        static_cast<int>(kD1LeaderPe));
        }
    }

    const uint64_t lane_cycle_start = AscendC::GetSystemCycle();
    bool second_hop_delay_applied = false;
    // M3：LocalFinal → INC put DestChannel ring（与 desc 同 slot）；Legacy → DestFinal
    bool use_dest_channel = false;
    if (desc->measurement_mode == kPlMeasurementFullDispatch) {
        __gm__ PlFullDispatchConfig *fdc_ws =
            reinterpret_cast<__gm__ PlFullDispatchConfig *>(sym + kPlFullDispatchConfigOff);
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(fdc_ws));
        __gm__ PlInvocationWorkspaceDesc *ws_fwd = PlFullDispatchInvocationWorkspace(sym, fdc_ws, epoch);
        use_dest_channel = PlDevWorkspaceIsWorkerLocalFinal(ws_fwd);
        if (!use_dest_channel) {
            fc->error_code = kPlErrWorkspaceFlagInvalid;
            PlDcciForwardCtr(fc);
            return;
        }
    }

    while (ingress_seen < ingress_end) {
        AscendC::PipeBarrier<PIPE_ALL>();
        const uint64_t wait_t0 = AscendC::GetSystemCycle();
        Qv2DcciLine(reinterpret_cast<__gm__ uint8_t *>(ingress_tail));
        const uint64_t remote_ingress_tail = ingress_tail->tail;
        if (remote_ingress_tail <= ingress_seen) {
            wait_ingress_tail_cycles += (AscendC::GetSystemCycle() - wait_t0);
        }
        if (fc->first_ingress_tail == 0 && remote_ingress_tail > ingress_seen) {
            fc->first_ingress_tail = remote_ingress_tail;
            if (fst->first_ingress_tail_cycle == 0u) {
                fst->first_ingress_tail_cycle = AscendC::GetSystemCycle();
            }
        }
        fc->ingress_tail_observed = remote_ingress_tail;
        fst->last_ingress_cycle = AscendC::GetSystemCycle();
        const uint64_t occ = remote_ingress_tail - ingress_seen;
        if (occ > fc->max_ingress_occupancy) {
            fc->max_ingress_occupancy = occ;
        }

        while (ingress_seen < remote_ingress_tail && ingress_seen < ingress_end) {
            if (trace->first_issue_cycle == 0) {
                trace->first_issue_cycle = AscendC::GetSystemCycle();
                PlTraceDcci(trace);
            }
            const uint32_t slot0 = static_cast<uint32_t>(ingress_seen % depth);
            __gm__ PlDescriptor *d0 = reinterpret_cast<__gm__ PlDescriptor *>(
                sym + DevPlIncDescLaneOff(lane_id) + static_cast<uint64_t>(slot0) * kPlDescriptorBytes);

            uint64_t stale0 = 0;
            uint64_t dup0 = 0;
            uint64_t lost0 = 0;
            uint64_t gen_mm0 = 0;
            const uint64_t desc_t0 = AscendC::GetSystemCycle();
            if (!PlReadDescriptorWithRetry(d0, epoch, ingress_seen, slot0, stale0, dup0, lost0, gen_mm0)) {
                descriptor_validation_cycles += (AscendC::GetSystemCycle() - desc_t0);
                fc->error_code = (gen_mm0 > 0) ? kPlErrGenerationMismatch : 40u;
                (void)stale0;
                (void)dup0;
                (void)lost0;
                PlDcciForwardCtr(fc);
                if (gen_mm0 > 0) {
                    PlPublishWorkerGenerationMismatch(sym, desc, lane_id, epoch, gen_mm0);
                }
                return;
            }
            descriptor_validation_cycles += (AscendC::GetSystemCycle() - desc_t0);

            const int dest_pe = static_cast<int>(d0->destination_rank);
            const uint32_t source_ch = d0->source_rank;
            __gm__ Qv2HeadLine *credit_line =
                reinterpret_cast<__gm__ Qv2HeadLine *>(sym + DevPlIncEgressCreditLineOff(d0->destination_rank));

            uint64_t cached_dest_credit = credit_line->head;
            AscendC::PipeBarrier<PIPE_ALL>();
            Qv2DcciLine(reinterpret_cast<__gm__ uint8_t *>(credit_line));
            cached_dest_credit = credit_line->head;
            __gm__ int32_t *credit_notify_sig = reinterpret_cast<__gm__ int32_t *>(
                reinterpret_cast<__gm__ uint8_t *>(credit_line) + kQv2HeadNotifySeqOff);
            while (egress_seq - cached_dest_credit >= depth) {
                if (fst->credit_wait_start == 0u) {
                    fst->credit_wait_start = AscendC::GetSystemCycle();
                }
                ++egress_credit_wait_cycles_local;
                ++fc->egress_credit_wait_cycles;
                const int32_t need_credit =
                    static_cast<int32_t>(egress_seq - static_cast<uint64_t>(depth) + 1u);
                if (aclshmem_int32_test(credit_notify_sig, ACLSHMEM_CMP_GE, need_credit) != 0) {
                    AscendC::PipeBarrier<PIPE_ALL>();
                    Qv2DcciLine(reinterpret_cast<__gm__ uint8_t *>(credit_line));
                    if (credit_line->head > cached_dest_credit) {
                        cached_dest_credit = credit_line->head;
                    }
                } else {
                    PlSpinWaitSignalLineDcci(reinterpret_cast<__gm__ uint8_t *>(credit_line),
                                             static_cast<uint32_t>(fc->egress_credit_wait_cycles));
                }
            }
            if (fst->credit_wait_start > 0u && fst->credit_wait_end == 0u) {
                fst->credit_wait_end = AscendC::GetSystemCycle();
            }

            // FAULT_DELAY_SECOND_HOP：在首次第二跳 put 前注入（避免与 worker upload 重叠被吞掉）
            if (!second_hop_delay_applied) {
                __gm__ PlTimingNegLine *neg =
                    reinterpret_cast<__gm__ PlTimingNegLine *>(sym + kPlTimingNegOff);
                PlDcci(reinterpret_cast<__gm__ uint8_t *>(neg));
                if (neg->magic == kPlTimingNegMagic && neg->delay_second_hop_cycles > 0u) {
                    PlApplyTimingNegDelay(neg->delay_second_hop_cycles);
                }
                second_hop_delay_applied = true;
            }

            const uint64_t cyc_fwd = AscendC::GetSystemCycle();
            if (fst->payload_issue_start == 0u) {
                fst->payload_issue_start = cyc_fwd;
            }
            if (remote_ingress_tail < ingress_end && fc->forward_before_final_ingress_count == 0) {
                ++fc->forward_before_final_ingress_count;
            }

            const uint64_t ingress_avail = remote_ingress_tail - ingress_seen;
            const uint64_t egress_credit_avail = depth - (egress_seq - cached_dest_credit);
            const uint32_t ingress_until_wrap = depth - slot0;
            const uint32_t egress_slot0 = static_cast<uint32_t>(egress_seq % depth);
            const uint32_t egress_until_wrap = depth - egress_slot0;
            uint32_t max_run = batch_max;
            if (static_cast<uint64_t>(max_run) > ingress_avail) {
                max_run = static_cast<uint32_t>(ingress_avail);
            }
            if (static_cast<uint64_t>(max_run) > egress_credit_avail) {
                max_run = static_cast<uint32_t>(egress_credit_avail);
            }
            if (max_run > ingress_until_wrap) {
                max_run = ingress_until_wrap;
            }
            if (max_run > egress_until_wrap) {
                max_run = egress_until_wrap;
            }
            if (max_run == 0u) {
                max_run = 1u;
            }

            uint32_t run = max_run;
            const uint64_t coalesce_t0 = AscendC::GetSystemCycle();
            while (run > 1u) {
                bool ok = true;
                const uint32_t dest_rank0 = d0->destination_rank;
                const uint32_t slot_base0 = PlDescriptorPayloadSlot(d0, desc);
                for (uint32_t i = 1; i < run; ++i) {
                    const uint32_t slot_i = static_cast<uint32_t>((ingress_seen + i) % depth);
                    __gm__ PlDescriptor *di = reinterpret_cast<__gm__ PlDescriptor *>(
                        sym + DevPlIncDescLaneOff(lane_id) + static_cast<uint64_t>(slot_i) * kPlDescriptorBytes);
                    uint64_t stale = 0;
                    uint64_t dup = 0;
                    uint64_t lost = 0;
                    uint64_t gen_mm = 0;
                    if (!PlReadDescriptorWithRetry(di, epoch, ingress_seen + i, slot_i, stale, dup, lost, gen_mm)) {
                        if (gen_mm > 0) {
                            descriptor_validation_cycles += (AscendC::GetSystemCycle() - coalesce_t0);
                            fc->error_code = kPlErrGenerationMismatch;
                            PlPublishWorkerGenerationMismatch(sym, desc, lane_id, epoch, gen_mm);
                            PlDcciForwardCtr(fc);
                            return;
                        }
                        ok = false;
                        break;
                    }
                    // DestChannel：按 ring 连续 put，不要求 final_slot 连续；Legacy DestFinal 仍要求
                    const bool slot_ok =
                        use_dest_channel || (PlDescriptorPayloadSlot(di, desc) == slot_base0 + i);
                    if (di->destination_rank != dest_rank0 || di->source_rank != source_ch || !slot_ok ||
                        di->lane_sequence != d0->lane_sequence + i || di->payload_bytes != nbytes) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    break;
                }
                --run;
            }
            descriptor_validation_cycles += (AscendC::GetSystemCycle() - coalesce_t0);

            if (run >= 2u) {
                const uint64_t payload_range_bytes = static_cast<uint64_t>(run) * nbytes;
                const uint64_t desc_range_bytes = static_cast<uint64_t>(run) * kPlDescriptorBytes;
                const uint64_t ingress_payload_src = DevPlIncPayloadLaneOff(lane_id) + static_cast<uint64_t>(slot0) * nbytes;
                // M3 LocalFinal：DestChannelPayload[egress_slot]；Legacy：DestFinal[final_slot]
                const uint64_t dest_payload_dst =
                    DevPlDestChannelPayloadOff(source_ch) + static_cast<uint64_t>(egress_slot0) * nbytes;
                const uint64_t issue_t0 = AscendC::GetSystemCycle();
                if (run <= ingress_until_wrap) {
                    Qv2IssuePayloadRangeP6NoDrain(sym, ingress_payload_src, dest_payload_dst,
                                                  static_cast<uint32_t>(payload_range_bytes), dest_pe, payload_puts,
                                                  payload_mte);
                } else {
                    const uint32_t bytes1 = ingress_until_wrap * nbytes;
                    const uint32_t bytes2 = static_cast<uint32_t>(payload_range_bytes) - bytes1;
                    Qv2IssuePayloadRangeP6NoDrain(sym, ingress_payload_src, dest_payload_dst, bytes1, dest_pe,
                                                  payload_puts, payload_mte);
                    Qv2IssuePayloadRangeP6NoDrain(
                        sym, DevPlIncPayloadLaneOff(lane_id),
                        dest_payload_dst + static_cast<uint64_t>(ingress_until_wrap) * nbytes, bytes2, dest_pe,
                        payload_puts, payload_mte);
                }
                ++payload_range_puts;

                const uint64_t ingress_desc_src = DevPlIncDescLaneOff(lane_id) + static_cast<uint64_t>(slot0) * kPlDescriptorBytes;
                const uint64_t dest_desc_dst =
                    DevPlDestChannelDescOff(source_ch) + static_cast<uint64_t>(egress_slot0) * kPlDescriptorBytes;
                if (run <= egress_until_wrap) {
                    Qv2IssueDescriptorRangeP6NoDrain(sym, ingress_desc_src, dest_desc_dst,
                                                     static_cast<uint32_t>(desc_range_bytes), dest_pe, desc_puts,
                                                     desc_mte);
                } else {
                    const uint32_t bytes1 = egress_until_wrap * kPlDescriptorBytes;
                    const uint32_t bytes2 = static_cast<uint32_t>(desc_range_bytes) - bytes1;
                    Qv2IssueDescriptorRangeP6NoDrain(sym, ingress_desc_src, dest_desc_dst, bytes1, dest_pe, desc_puts,
                                                     desc_mte);
                    Qv2IssueDescriptorRangeP6NoDrain(
                        sym, DevPlIncDescLaneOff(lane_id),
                        DevPlDestChannelDescOff(source_ch), bytes2, dest_pe, desc_puts, desc_mte);
                }
                ++desc_range_puts;
                // range 路径 MTE 完成含在 Issue* 内，整段计入 issue+mte 合集后对半粗分
                const uint64_t issue_span = AscendC::GetSystemCycle() - issue_t0;
                payload_descriptor_issue_cycles += issue_span / 2u;
                mte_completion_cycles += issue_span - issue_span / 2u;
            } else {
                run = 1u;
                __gm__ uint8_t *ingress_payload =
                    sym + DevPlIncPayloadLaneOff(lane_id) + static_cast<uint64_t>(slot0) * nbytes;
                const uint64_t dest_final_off =
                    DevPlDestChannelPayloadOff(source_ch) + static_cast<uint64_t>(egress_slot0) * nbytes;
                const uint64_t issue_t0 = AscendC::GetSystemCycle();
                aclshmem_putmem_nbi(sym + dest_final_off, ingress_payload, nbytes, dest_pe);
                ++payload_puts;
                ++payload_scalar_puts;
                payload_descriptor_issue_cycles += (AscendC::GetSystemCycle() - issue_t0);
                const uint64_t mte_t0 = AscendC::GetSystemCycle();
                Qv2MteUbComplete(dest_pe, payload_mte);
                mte_completion_cycles += (AscendC::GetSystemCycle() - mte_t0);

                const uint64_t dest_desc_off =
                    DevPlDestChannelDescOff(source_ch) + static_cast<uint64_t>(egress_slot0) * kPlDescriptorBytes;
                const uint64_t issue_t1 = AscendC::GetSystemCycle();
                aclshmem_putmem_nbi(sym + dest_desc_off, reinterpret_cast<__gm__ uint8_t *>(d0), kPlDescriptorBytes,
                                    dest_pe);
                ++desc_puts;
                ++desc_scalar_puts;
                payload_descriptor_issue_cycles += (AscendC::GetSystemCycle() - issue_t1);
                const uint64_t mte_t1 = AscendC::GetSystemCycle();
                Qv2MteUbComplete(dest_pe, desc_mte);
                mte_completion_cycles += (AscendC::GetSystemCycle() - mte_t1);
            }

            // P8.2：同 dest 累积 issue，到安全条件再一次 drain+egress_tail+ingress_head
            // 换 dest / 即将 wrap：先 flush 旧 group
            if (pending_publish_tokens > 0u &&
                (pending_dest_pe != dest_pe || pending_source_ch != source_ch || slot0 == 0u || egress_slot0 == 0u)) {
                if (slot0 == 0u || egress_slot0 == 0u) {
                    ++forced_wrap_flush_count;
                }
                {
                    const uint64_t drain_t0 = AscendC::GetSystemCycle();
                    Qv2DrainTileDataAndDescriptor(drain_quiet);
                    data_drain_quiet_cycles += (AscendC::GetSystemCycle() - drain_t0);
                    const uint64_t remote_egress_tail_off = DevPlEgressTailLineOff(pending_source_ch);
                    const uint64_t tp_start = AscendC::GetSystemCycle();
                    if (fst->tail_publish_start == 0u) {
                        fst->tail_publish_start = tp_start;
                    }
                    PlPublishEgressTail(egress_publish_scratch, sym, remote_egress_tail_off, pending_dest_pe,
                                        last_visible_egress_seq, epoch, egress_tail_pub, egress_tail_quiet);
                    const uint64_t tp_end = AscendC::GetSystemCycle();
                    fst->tail_publish_end = tp_end;
                    if (tp_end > tp_start) {
                        tail_publish_cycles_sum += (tp_end - tp_start);
                    }
                    if (pending_dest_pe >= 0 && static_cast<uint32_t>(pending_dest_pe) < kPlMaxSources) {
                        visible_done_per_dest[static_cast<uint32_t>(pending_dest_pe)] = AscendC::GetSystemCycle();
                    }
                    ingress_head->head = last_reclaimed_ingress_seq;
                    const uint64_t remote_ingress_head_off = DevPlIngressHeadLineOff(lane_id);
                    const uint64_t head_t0 = AscendC::GetSystemCycle();
                    PlPublishEgressHead(ingress_head, sym, remote_ingress_head_off, static_cast<int>(desc->peer_pe),
                                        last_reclaimed_ingress_seq, epoch, ingress_head_pub, ingress_head_quiet);
                    ingress_head_publish_cycles += (AscendC::GetSystemCycle() - head_t0);
                    ++publish_group_count;
                    publish_group_tokens += pending_publish_tokens;
                    if (pending_publish_tokens > publish_group_max_tokens) {
                        publish_group_max_tokens = pending_publish_tokens;
                    }
                    if (pending_publish_tokens < tile_pub) {
                        ++partial_publish_group_count;
                    }
                    pending_publish_tokens = 0u;
                    pending_dest_pe = -1;
                }
            }

            egress_seq += run;
            ingress_seen += run;
            fc->egress_forwarded += run;
            const uint32_t fwd_dest = d0->destination_rank;
            if (fwd_dest < kPlMaxSources) {
                fwd_per_dest[fwd_dest] += run;
                tail_per_dest[fwd_dest] = egress_seq;
            }
            ++batch_count;
            batch_tokens += run;
            if (run > batch_max_tokens) {
                batch_max_tokens = run;
            }

            if (pending_publish_tokens == 0u) {
                pending_dest_pe = dest_pe;
                pending_source_ch = source_ch;
            }
            pending_publish_tokens += run;
            last_visible_egress_seq = egress_seq;
            last_reclaimed_ingress_seq = ingress_seen;
            if (pending_publish_tokens > pending_group_high_watermark) {
                pending_group_high_watermark = pending_publish_tokens;
            }

            // A full ring must never contain unpublished data.  Under a hot
            // destination the old tile-amortized policy could issue 64
            // entries, then block on credit before making their tail visible:
            // producer waited for head while consumer waited for tail.
            // Each run is already a coalesced range (up to batch_max), so
            // publish once per run to preserve batching without a hidden-full
            // ring window.
            bool need_flush = pending_publish_tokens > 0u;
            if (!need_flush && pending_publish_tokens > 0u) {
                // 不 flush 将耗尽 credit：剩余 credit 不足以再发一批
                const uint64_t credit_left = depth - (egress_seq - cached_dest_credit);
                if (credit_left <= static_cast<uint64_t>(batch_max)) {
                    need_flush = true;
                    ++forced_credit_flush_count;
                }
            }
            if (!need_flush && pending_publish_tokens > 0u) {
                // upstream 暂时空且本 lane 未结束 → flush 避免 HOL
                if (ingress_seen >= remote_ingress_tail && ingress_seen < ingress_end) {
                    need_flush = true;
                    ++upstream_empty_flush_count;
                }
            }
            if (need_flush && pending_publish_tokens > 0u) {
                const uint64_t drain_t0 = AscendC::GetSystemCycle();
                Qv2DrainTileDataAndDescriptor(drain_quiet);
                data_drain_quiet_cycles += (AscendC::GetSystemCycle() - drain_t0);
                const uint64_t remote_egress_tail_off = DevPlEgressTailLineOff(pending_source_ch);
                const uint64_t tp_start = AscendC::GetSystemCycle();
                if (fst->tail_publish_start == 0u) {
                    fst->tail_publish_start = tp_start;
                }
                PlPublishEgressTail(egress_publish_scratch, sym, remote_egress_tail_off, pending_dest_pe,
                                    last_visible_egress_seq, epoch, egress_tail_pub, egress_tail_quiet);
                const uint64_t tp_end = AscendC::GetSystemCycle();
                fst->tail_publish_end = tp_end;
                if (tp_end > tp_start) {
                    tail_publish_cycles_sum += (tp_end - tp_start);
                }
                if (pending_dest_pe >= 0 && static_cast<uint32_t>(pending_dest_pe) < kPlMaxSources) {
                    visible_done_per_dest[static_cast<uint32_t>(pending_dest_pe)] = AscendC::GetSystemCycle();
                }
                ingress_head->head = last_reclaimed_ingress_seq;
                const uint64_t remote_ingress_head_off = DevPlIngressHeadLineOff(lane_id);
                const uint64_t head_t0 = AscendC::GetSystemCycle();
                PlPublishEgressHead(ingress_head, sym, remote_ingress_head_off, static_cast<int>(desc->peer_pe),
                                    last_reclaimed_ingress_seq, epoch, ingress_head_pub, ingress_head_quiet);
                ingress_head_publish_cycles += (AscendC::GetSystemCycle() - head_t0);
                ++publish_group_count;
                publish_group_tokens += pending_publish_tokens;
                if (pending_publish_tokens > publish_group_max_tokens) {
                    publish_group_max_tokens = pending_publish_tokens;
                }
                if (pending_publish_tokens < tile_pub) {
                    ++partial_publish_group_count;
                }
                pending_publish_tokens = 0u;
                pending_dest_pe = -1;
            }
        }
        fc->ingress_seen = ingress_seen;
    }

    // epoch/lane 结束：flush 残余 pending；禁止再发相同 head/tail
    if (pending_publish_tokens > 0u) {
        ++final_flush_count;
        const uint64_t drain_t0 = AscendC::GetSystemCycle();
        Qv2DrainTileDataAndDescriptor(drain_quiet);
        data_drain_quiet_cycles += (AscendC::GetSystemCycle() - drain_t0);
        const uint64_t remote_egress_tail_off = DevPlEgressTailLineOff(pending_source_ch);
        const uint64_t tp_start = AscendC::GetSystemCycle();
        if (fst->tail_publish_start == 0u) {
            fst->tail_publish_start = tp_start;
        }
        PlPublishEgressTail(egress_publish_scratch, sym, remote_egress_tail_off, pending_dest_pe,
                            last_visible_egress_seq, epoch, egress_tail_pub, egress_tail_quiet);
        const uint64_t tp_end = AscendC::GetSystemCycle();
        fst->tail_publish_end = tp_end;
        if (tp_end > tp_start) {
            tail_publish_cycles_sum += (tp_end - tp_start);
        }
        if (pending_dest_pe >= 0 && static_cast<uint32_t>(pending_dest_pe) < kPlMaxSources) {
            visible_done_per_dest[static_cast<uint32_t>(pending_dest_pe)] = AscendC::GetSystemCycle();
        }
        ingress_head->head = last_reclaimed_ingress_seq;
        const uint64_t remote_ingress_head_off = DevPlIngressHeadLineOff(lane_id);
        const uint64_t head_t0 = AscendC::GetSystemCycle();
        PlPublishEgressHead(ingress_head, sym, remote_ingress_head_off, static_cast<int>(desc->peer_pe),
                            last_reclaimed_ingress_seq, epoch, ingress_head_pub, ingress_head_quiet);
        ingress_head_publish_cycles += (AscendC::GetSystemCycle() - head_t0);
        fst->ingress_head_publish_end = AscendC::GetSystemCycle();
        ++publish_group_count;
        publish_group_tokens += pending_publish_tokens;
        if (pending_publish_tokens > publish_group_max_tokens) {
            publish_group_max_tokens = pending_publish_tokens;
        }
        if (pending_publish_tokens < tile_pub) {
            ++partial_publish_group_count;
        }
        pending_publish_tokens = 0u;
    } else if (ingress_seen > 0u) {
        fst->ingress_head_publish_end = AscendC::GetSystemCycle();
    }

    fc->payload_put_count += payload_puts;
    fc->payload_mte_wait_count += payload_mte;
    fc->descriptor_put_count += desc_puts;
    fc->descriptor_mte_wait_count += desc_mte;
    fc->drain_quiet_count += drain_quiet;
    fc->egress_tail_publish_count += egress_tail_pub;
    fc->ingress_head_publish_count += ingress_head_pub;
    fc->egress_payload_range_put_count += payload_range_puts;
    fc->egress_payload_scalar_put_count += payload_scalar_puts;
    fc->egress_descriptor_range_put_count += desc_range_puts;
    fc->egress_descriptor_scalar_put_count += desc_scalar_puts;
    fc->egress_batch_count += batch_count;
    fc->egress_batch_tokens += batch_tokens;
    fc->egress_batch_max_tokens = batch_max_tokens;
    // LocalFinal：DestFinal remote put 已迁出；device 显式保持 0（禁止 host 硬编码）
    fc->direct_destfinal_remote_put_count = 0u;
    fst->payload_issue_end = AscendC::GetSystemCycle();
    fst->egress_credit_wait_cycles = egress_credit_wait_cycles_local;
    fst->egress_tail_publish_count = static_cast<uint32_t>(egress_tail_pub);
    fst->batch_count = static_cast<uint32_t>(batch_count);
    fst->batch_tokens = batch_tokens;
    fst->egress_tail_publish_cycles_sum = tail_publish_cycles_sum;
    fst->wait_ingress_tail_cycles = wait_ingress_tail_cycles;
    fst->descriptor_validation_cycles = descriptor_validation_cycles;
    fst->payload_descriptor_issue_cycles = payload_descriptor_issue_cycles;
    fst->mte_completion_cycles = mte_completion_cycles;
    fst->data_drain_quiet_cycles = data_drain_quiet_cycles;
    fst->ingress_head_publish_cycles = ingress_head_publish_cycles;
    fst->payload_put_count = static_cast<uint32_t>(payload_puts);
    fst->descriptor_put_count = static_cast<uint32_t>(desc_puts);
    fst->drain_quiet_count = static_cast<uint32_t>(drain_quiet);
    fst->ingress_head_publish_count = static_cast<uint32_t>(ingress_head_pub);
    fst->max_batch = static_cast<uint32_t>(batch_max_tokens);
    fst->publish_group_count = publish_group_count;
    fst->publish_group_tokens = publish_group_tokens;
    fst->publish_group_max_tokens = publish_group_max_tokens;
    fst->partial_publish_group_count = partial_publish_group_count;
    fst->forced_wrap_flush_count = forced_wrap_flush_count;
    fst->forced_credit_flush_count = forced_credit_flush_count;
    fst->upstream_empty_flush_count = upstream_empty_flush_count;
    fst->final_flush_count = final_flush_count;
    fst->pending_group_high_watermark = pending_group_high_watermark;
    fst->first_tail_early_flush_count = first_tail_early_flush_count;
    PlDcciForwardCtr(fc);
    const uint64_t done_cyc = AscendC::GetSystemCycle();
    fst->forward_done_cycle = done_cyc;
    fst->magic = kPlIncFwdLaneStageMagic;
    PlDcciTrace256(reinterpret_cast<__gm__ uint8_t *>(fst));
    const uint64_t first_cyc = trace->first_issue_cycle > 0 ? trace->first_issue_cycle : done_cyc;
    PlPublishAivEpochTiming(sym, DevPlForwardAivTimingOff(lane_id), first_cyc, done_cyc, epoch, lane_id);
    // 第二跳 egress 可见：lane 结束后 push（有流量 dest + owned zero dest）
    if (desc->measurement_mode == kPlMeasurementFullDispatch) {
        __gm__ PlFullDispatchConfig *fdc =
            reinterpret_cast<__gm__ PlFullDispatchConfig *>(sym + kPlFullDispatchConfigOff);
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(fdc));
        if (fdc->probe_mode == kPlFullDispatchProbeFull) {
            __gm__ PlInvocationDesc *inv = PlFullDispatchInvocation(sym, fdc, epoch);
            uint32_t exp_dest[kPlMaxSources];
            PlIncComputeEgressExpectedPerDest(sym, desc->pair_id, desc->worker_count, desc->expert_per_pe, epoch,
                                              exp_dest);
            PlIncForwardLanePublishEgressVisible(sym, desc->pair_id, lane_id, desc->worker_count, epoch,
                                                    static_cast<uint32_t>(inv->generation), exp_dest, fwd_per_dest,
                                                    tail_per_dest, visible_done_per_dest, lane_cycle_start,
                                                    static_cast<int>(kD1LeaderPe));
        }
    }
    __gm__ PlForwardDoneLine *fdone =
        reinterpret_cast<__gm__ PlForwardDoneLine *>(sym + DevPlForwardDoneLineOff(lane_id));
    fdone->done_epoch = epoch;
    fdone->lane_id = lane_id;
    fdone->magic = kPlMagic;
    PlDcciDoneLine(reinterpret_cast<__gm__ uint8_t *>(fdone));
}

// Destination receive：每 source 独占 SPSC channel；head credit 按 8 token 聚合
// Full Dispatch recv AIV 的唯一 completion/error 出口。失败也必须发布本
// source 的 RecvDoneLine，避免 destination control / PE0 只能靠超时猜测。
__aicore__ inline void PlPublishRecvDone(__gm__ uint8_t *sym, uint32_t source_id, uint64_t epoch,
                                         uint32_t expected, uint32_t received, uint32_t final_head,
                                         uint32_t error_code, uint32_t last_remote_tail = 0u,
                                         uint32_t last_descriptor_seq = 0u)
{
    __gm__ PlRecvDoneLine *rdone =
        reinterpret_cast<__gm__ PlRecvDoneLine *>(sym + PlDevRecvDoneLineOff(source_id));
    rdone->epoch = epoch;
    rdone->source_rank = source_id;
    rdone->expected = expected;
    rdone->received = received;
    rdone->final_head = final_head;
    rdone->error_code = error_code;
    rdone->magic = kPlRecvDoneMagic;
    rdone->last_remote_tail = last_remote_tail;
    rdone->last_descriptor_seq = last_descriptor_seq;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(rdone));
}

__aicore__ inline void PlWorkerRecvChannelEpoch(__gm__ uint8_t *sym, __gm__ PlPipelineDesc *desc,
                                                __gm__ PlRecvCounters *rc, __gm__ PlPipelineTiming *timing,
                                                __gm__ PlOverlapTelemetry *overlap, __gm__ PlServiceTraceLine *trace,
                                                uint32_t source_id, uint64_t epoch, bool device_full_verify)
{
    (void)timing;
    (void)overlap;
    trace->epoch_enter_cycle = AscendC::GetSystemCycle();
    PlTraceDcci(trace);
    __gm__ PlRecvChannelCompletionTrace *rct = reinterpret_cast<__gm__ PlRecvChannelCompletionTrace *>(
        sym + DevPlRecvChannelCompletionTraceOff(source_id));
    PlClearTrace384(reinterpret_cast<__gm__ uint8_t *>(rct));
    rct->epoch = epoch;
    rct->source_rank = source_id;
    uint64_t tail_poll_local = 0;
    uint32_t descriptor_retry_local = 0;
    uint32_t tail_signal_wake_local = 0;
    uint64_t counts_done_wait_cycles = 0;
    uint64_t wait_first_tail_cycles = 0;
    uint64_t tail_wait_total_cycles = 0;
    uint64_t descriptor_read_verify_cycles = 0;
    uint64_t payload_assist_ready_cycles = 0;
    uint64_t head_credit_publish_cycles = 0;
    uint64_t recv_done_publish_cycles = 0;
    uint64_t lf_copy_issue = 0;
    uint64_t lf_copy_complete = 0;
    uint64_t lf_copy_bytes = 0;
    uint64_t lf_copy_cycles = 0;
    uint64_t lf_ring_dcci_count = 0;
    uint64_t lf_ring_dcci_bytes = 0;
    uint64_t lf_ring_dcci_cycles = 0;
    uint64_t lf_out_dcci_count = 0;
    uint64_t lf_out_dcci_bytes = 0;
    uint64_t lf_out_dcci_cycles = 0;
    uint64_t lf_assist_count = 0;
    uint64_t lf_assist_cycles = 0;
    uint64_t lf_ready_count = 0;
    uint64_t lf_ready_cycles = 0;
    uint64_t lf_credit_count = 0;
    uint64_t lf_credit_cycles = 0;
    uint64_t lf_first_copy_cycle = 0;
    uint64_t lf_last_copy_cycle = 0;
    uint64_t lf_first_credit_cycle = 0;
    uint32_t stale_local = 0;
    uint32_t dup_local = 0;
    uint32_t lost_local = 0;
    const uint32_t nbytes = desc->payload_bytes;
    const uint32_t depth = desc->ring_depth;
    uint32_t tokens_expected = 0u;
    const int inc_peer = static_cast<int>(source_id + desc->worker_count);
    uint32_t credit_batch_raw =
        desc->batch_tokens > 0u ? desc->batch_tokens : 8u;
    if (PlIsSourceMajorRaw(desc) &&
        desc->measurement_mode == kPlMeasurementFullDispatch &&
        desc->egress_publish_target_bytes > 0u &&
        nbytes > 0u) {
        const uint32_t by_bytes =
            desc->egress_publish_target_bytes / nbytes;
        if (by_bytes > 0u) {
            credit_batch_raw = by_bytes;
        }
    }
    // Credit is occupancy-checked by the producer; use the same near-full
    // window as egress so a 63-token transfer is not fragmented into a
    // permanent 32-token cadence after the first ring turn.
    const uint32_t credit_batch_cap = depth > 1u ? depth - 1u : 1u;
    const uint32_t credit_batch =
        credit_batch_raw > credit_batch_cap ? credit_batch_cap
                                            : credit_batch_raw;

    // C3：读 expected；expected=0 仍必须发布 RecvDoneLine（禁止无完成证据直接 return）
    if (desc->measurement_mode == kPlMeasurementFullDispatch) {
        __gm__ PlFullDispatchConfig *fcfg =
            reinterpret_cast<__gm__ PlFullDispatchConfig *>(sym + kPlFullDispatchConfigOff);
        if (!PlDevRequireInvocationWorkspace(sym, fcfg, epoch)) {
            __gm__ PlFullOutputDoneLine *fout_ws =
                reinterpret_cast<__gm__ PlFullOutputDoneLine *>(sym + fcfg->full_output_done_off);
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(fout_ws));
            rc->error_code = fout_ws->error_code;
            PlDcciRecvCtr(rc);
            return;
        }
        __gm__ PlFullOutputDoneLine *fout_done =
            reinterpret_cast<__gm__ PlFullOutputDoneLine *>(sym + fcfg->full_output_done_off);
        const uint64_t counts_t0 = AscendC::GetSystemCycle();
        while (fout_done->counts_done_epoch < epoch) {
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(fout_done));
        }
        counts_done_wait_cycles = AscendC::GetSystemCycle() - counts_t0;
        __gm__ int32_t *expected = reinterpret_cast<__gm__ int32_t *>(sym + kPlExpectedCountOff);
        Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(expected), desc->worker_count * sizeof(int32_t));
        tokens_expected = expected[source_id] > 0 ? static_cast<uint32_t>(expected[source_id]) : 0u;
        rct->expected = tokens_expected;
        __gm__ PlP5CellRouteRecvLedgerLine *cell_ledger =
            reinterpret_cast<__gm__ PlP5CellRouteRecvLedgerLine *>(
                sym + DevPlP5CellRouteRecvLedgerLineOff(source_id, desc->pair_id));
        cell_ledger->epoch = epoch;
        cell_ledger->source = source_id;
        cell_ledger->destination = desc->pair_id;
        cell_ledger->route_expected = tokens_expected;
        cell_ledger->recv_expected = tokens_expected;
        cell_ledger->magic = kPlP5CellLedgerMagic;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(cell_ledger));
        if (tokens_expected == 0u) {
            const uint64_t rd_t0 = AscendC::GetSystemCycle();
            PlPublishRecvDone(sym, source_id, epoch, 0u, 0u, 0u, 0u);
            recv_done_publish_cycles = AscendC::GetSystemCycle() - rd_t0;
            rct->recv_done_cycle = AscendC::GetSystemCycle();
            rct->counts_done_wait_cycles = counts_done_wait_cycles;
            rct->recv_done_publish_cycles = recv_done_publish_cycles;
            rct->magic = kPlRecvChanCompletionTraceMagic;
            PlDcciTrace384(reinterpret_cast<__gm__ uint8_t *>(rct));
            PlDcciRecvCtr(rc);
            return;
        }
        (void)fcfg;
    } else {
        tokens_expected = PlEpochRecvTokensExpected(sym, desc, source_id);
        rct->expected = tokens_expected;
    }

    rc->slot_overwrite_count = 0;
    __gm__ Qv2TailLine *egress_tail =
        reinterpret_cast<__gm__ Qv2TailLine *>(sym + DevPlEgressTailLineOff(source_id));
    __gm__ Qv2HeadLine *credit_scratch =
        reinterpret_cast<__gm__ Qv2HeadLine *>(sym + DevPlDestCreditPublishScratchOff(source_id));

    // E4/E5：source-major 每 epoch 本地协议 seq 从 0；统计 tokens_received 可累计
    const bool sm_full =
        PlIsSourceMajorRaw(desc) && (desc->measurement_mode == kPlMeasurementFullDispatch);
    uint64_t seen = 0u;
    if (sm_full) {
        seen = 0u;
        rc->channel_seen = 0u;
        rc->last_credit_head_published = 0u;
    } else {
        seen = rc->channel_seen;
        if (desc->measurement_mode == kPlMeasurementFullDispatch && epoch == 1u) {
            seen = 0u;
            rc->channel_seen = 0u;
        }
    }
    const uint64_t seq_end = seen + tokens_expected;
    uint64_t head_pub = 0;
    uint64_t head_quiet = 0;
    uint64_t last_published_head = sm_full ? 0u : rc->last_credit_head_published;
    uint32_t last_remote_tail = 0u;
    uint32_t last_descriptor_seq = 0u;
    const uint64_t final_forward_target = seq_end;
    const uint64_t tail_wait_t0 = AscendC::GetSystemCycle();
    uint64_t last_progress_cycle = tail_wait_t0;
    constexpr uint64_t kRecvTailWaitUs = 8000000ull;
    __gm__ int32_t *notify_sig = reinterpret_cast<__gm__ int32_t *>(
        reinterpret_cast<__gm__ uint8_t *>(egress_tail) + kQv2TailNotifySeqOff);

    // E4：两级等待 —— 先 DCCI 等到 egress_tail.epoch_tag==epoch，再 notify 前进
    if (sm_full) {
        bool epoch_ready = false;
        while (!PlWaitBudgetExceeded(tail_wait_t0, kRecvTailWaitUs)) {
            Qv2DcciLine(reinterpret_cast<__gm__ uint8_t *>(egress_tail));
            uint64_t tv = 0u;
            if (ReadEgressTailEpochValue(egress_tail, epoch, tv)) {
                epoch_ready = true;
                break;
            }
            // 仅当 magic 正确但 epoch 错才计 stale（未初始化不计）
            if (egress_tail->magic == inc::dc::dn::qv2::kQv2Magic && egress_tail->epoch_tag != epoch &&
                egress_tail->epoch_tag != 0u) {
                ++rc->stale_epoch_count;
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        if (!epoch_ready) {
            rc->error_code = kPlErrRecvTailEpochMissing;
            PlPublishRecvDone(sym, source_id, epoch, tokens_expected, 0u, 0u, rc->error_code);
            PlDcciRecvCtr(rc);
            return;
        }
    }

    while (seen < seq_end) {
        ++tail_poll_local;
        ++rc->tail_poll_cycles;
        // Stall is progress-based: a valid epoch may run long, but a tail,
        // descriptor, or receive counter that stops advancing is a protocol
        // failure rather than a reason to extend a global timeout.
        if (PlWaitBudgetExceeded(last_progress_cycle, kRecvTailWaitUs) ||
            PlWaitBudgetExceeded(tail_wait_t0, kRecvTailWaitUs)) {
            rc->error_code = kPlErrRecvTailIncomplete;
            break;
        }

        const uint64_t poll_t0 = AscendC::GetSystemCycle();
        bool wake = false;
        if (sm_full) {
            Qv2DcciLine(reinterpret_cast<__gm__ uint8_t *>(egress_tail));
            uint64_t tv = 0u;
            if (ReadEgressTailEpochValue(egress_tail, epoch, tv)) {
                if (aclshmem_int32_test(notify_sig, ACLSHMEM_CMP_GE, static_cast<int32_t>(seen + 1u)) != 0) {
                    wake = true;
                } else if (tv > seen) {
                    wake = true;
                }
            } else if (egress_tail->magic == inc::dc::dn::qv2::kQv2Magic && egress_tail->epoch_tag != epoch &&
                       egress_tail->epoch_tag != 0u) {
                ++rc->stale_epoch_count;
            }
        } else {
            wake = (aclshmem_int32_test(notify_sig, ACLSHMEM_CMP_GE, static_cast<int32_t>(seen + 1u)) != 0);
        }
        if (wake) {
            ++tail_signal_wake_local;
            AscendC::PipeBarrier<PIPE_ALL>();
            Qv2DcciLine(reinterpret_cast<__gm__ uint8_t *>(egress_tail));
            uint64_t remote_tail = 0u;
            if (sm_full) {
                if (!ReadEgressTailEpochValue(egress_tail, epoch, remote_tail)) {
                    if (egress_tail->magic == inc::dc::dn::qv2::kQv2Magic && egress_tail->epoch_tag != epoch &&
                        egress_tail->epoch_tag != 0u) {
                        ++rc->stale_epoch_count;
                    }
                    continue;
                }
            } else {
                remote_tail = egress_tail->tail;
            }
            if (remote_tail < seen) {
                rc->error_code = kPlErrRecvTailRegression; // same-epoch tail < seen
                break;
            }
            last_remote_tail = static_cast<uint32_t>(remote_tail);
            if (sm_full) {
                __gm__ PlP5CellRouteRecvLedgerLine *cell_ledger =
                    reinterpret_cast<__gm__ PlP5CellRouteRecvLedgerLine *>(
                        sym + DevPlP5CellRouteRecvLedgerLineOff(source_id, desc->pair_id));
                cell_ledger->remote_tail_observed = static_cast<uint32_t>(remote_tail);
                cell_ledger->magic = kPlP5CellLedgerMagic;
                PlDcci(reinterpret_cast<__gm__ uint8_t *>(cell_ledger));
            }
            if (remote_tail > seen) {
                last_progress_cycle = AscendC::GetSystemCycle();
                const uint64_t tail_cyc = AscendC::GetSystemCycle();
                if (rct->first_tail_seen_cycle == 0u) {
                    rct->first_tail_seen_cycle = tail_cyc;
                    wait_first_tail_cycles = tail_cyc - tail_wait_t0;
                }
                rct->last_tail_seen_cycle = tail_cyc;
            }
            const uint64_t occ = remote_tail - seen;
            if (occ > rc->max_occupancy) {
                rc->max_occupancy = occ;
            }

            if (seen < remote_tail && remote_tail < final_forward_target &&
                rc->destination_head_observed_before_final_forward_count == 0) {
                ++rc->destination_head_observed_before_final_forward_count;
            }

            while (seen < remote_tail && seen < seq_end) {
            const uint32_t slot = static_cast<uint32_t>(seen % depth);
            __gm__ PlDescriptor *d = reinterpret_cast<__gm__ PlDescriptor *>(
                sym + DevPlDestChannelDescOff(source_id) + static_cast<uint64_t>(slot) * kPlDescriptorBytes);

            uint64_t stale = 0;
            uint64_t dup = 0;
            uint64_t lost = 0;
            uint64_t gen_mm = 0;
            uint32_t desc_retries = 0u;
            bool desc_ok = false;
            const uint64_t desc_t0 = AscendC::GetSystemCycle();
            for (uint32_t di = 0; di < 64u; ++di) {
                AscendC::PipeBarrier<PIPE_ALL>();
                PlDcciDescriptor(d);
                if (d->epoch != epoch) {
                    ++stale;
                    ++desc_retries;
                    continue;
                }
                if (d->generation != static_cast<uint32_t>(epoch)) {
                    ++gen_mm;
                    ++desc_retries;
                    desc_ok = false;
                    break;
                }
                if (d->lane_sequence < seen) {
                    ++dup;
                    ++desc_retries;
                    continue;
                }
                if (d->lane_sequence > seen) {
                    ++lost;
                    ++desc_retries;
                    desc_ok = false;
                    break;
                }
                if (d->ring_slot != slot) {
                    ++desc_retries;
                    continue;
                }
                desc_ok = true;
                break;
            }
            descriptor_read_verify_cycles += (AscendC::GetSystemCycle() - desc_t0);
            descriptor_retry_local += desc_retries;
            stale_local += static_cast<uint32_t>(stale);
            dup_local += static_cast<uint32_t>(dup);
            lost_local += static_cast<uint32_t>(lost);
            if (!desc_ok) {
                rc->error_code = (gen_mm > 0) ? kPlErrGenerationMismatch : kPlErrRecvDescriptorIncomplete;
                rc->verify_error_count += 1;
                rc->stale_epoch_count += stale;
                if (gen_mm > 0) {
                    __gm__ PlFullDispatchConfig *fcfg =
                        reinterpret_cast<__gm__ PlFullDispatchConfig *>(sym + kPlFullDispatchConfigOff);
                    __gm__ PlFullOutputDoneLine *fout =
                        reinterpret_cast<__gm__ PlFullOutputDoneLine *>(sym + fcfg->full_output_done_off);
                    fout->error_code = kPlErrGenerationMismatch;
                    fout->generation_mismatch_count += static_cast<uint32_t>(gen_mm);
                    PlDcci(reinterpret_cast<__gm__ uint8_t *>(fout));
                }
                if (desc->measurement_mode == kPlMeasurementFullDispatch) {
                    PlPublishRecvDone(sym, source_id, epoch, tokens_expected,
                                      static_cast<uint32_t>(seen - (seq_end - tokens_expected)),
                                      static_cast<uint32_t>(last_published_head), rc->error_code,
                                      last_remote_tail, last_descriptor_seq);
                }
                PlDcciRecvCtr(rc);
                return;
            }
            rc->stale_epoch_count += stale;
            rc->duplicate_count += dup;
            ++rc->descriptor_verified_count;
            last_progress_cycle = AscendC::GetSystemCycle();

            const uint64_t cyc = AscendC::GetSystemCycle();
            if (rct->first_descriptor_cycle == 0u) {
                rct->first_descriptor_cycle = cyc;
            }
            rct->last_descriptor_cycle = cyc;
            if (trace->first_issue_cycle == 0) {
                trace->first_issue_cycle = cyc;
                PlTraceDcci(trace);
            }
            if (rc->first_destination_consume == 0) {
                rc->first_destination_consume = cyc;
            }

            if (device_full_verify && desc->measurement_mode != kPlMeasurementFullDispatch) {
                __gm__ PlFullDispatchConfig *cfg_verify =
                    reinterpret_cast<__gm__ PlFullDispatchConfig *>(sym + kPlFullDispatchConfigOff);
                const uint32_t pslot = PlDescriptorPayloadSlot(d, desc);
                if (cfg_verify->final_payload_off != 0u) {
                    __gm__ uint8_t *payload = reinterpret_cast<__gm__ uint8_t *>(cfg_verify->final_payload_off) +
                                              static_cast<uint64_t>(pslot) * nbytes;
                    AscendC::PipeBarrier<PIPE_ALL>();
                    Qv2DcciRange(payload, nbytes);
                    if (!Qv2CheckPayload(payload, nbytes, d->source_rank, d->token_sequence)) {
                        ++rc->verify_error_count;
                        rc->error_code = 51;
                        PlDcciRecvCtr(rc);
                        return;
                    }
                    ++rc->payload_verified_count;
                }
            }

            if (desc->measurement_mode == kPlMeasurementFullDispatch) {
                __gm__ PlFullDispatchConfig *cfg =
                    reinterpret_cast<__gm__ PlFullDispatchConfig *>(sym + kPlFullDispatchConfigOff);
                uint32_t final_slot = PlDescriptorPayloadSlot(d, desc);
                if ((desc->pipeline_mode & inc::dc::dn::pl::kPlModeReceiverFinalSlot) != 0u) {
                    __gm__ PlFullDispatchConfig *cfg_slot =
                        reinterpret_cast<__gm__ PlFullDispatchConfig *>(sym + kPlFullDispatchConfigOff);
                    __gm__ int32_t *seg_base =
                        reinterpret_cast<__gm__ int32_t *>(sym + cfg_slot->segment_base_off);
                    const uint32_t epp = desc->expert_per_pe == 0u ? 8u : desc->expert_per_pe;
                    if (d->source_rank >= desc->worker_count || d->local_expert_id >= epp) {
                        ++rc->verify_error_count;
                        rc->error_code = 56u;
                        PlDcciRecvCtr(rc);
                        return;
                    }
                    const uint64_t seg_idx =
                        static_cast<uint64_t>(d->local_expert_id) * desc->worker_count + d->source_rank;
                    const int32_t base = seg_base[seg_idx];
                    if (base < 0) {
                        ++rc->verify_error_count;
                        rc->error_code = 57u;
                        PlDcciRecvCtr(rc);
                        return;
                    }
                    final_slot = static_cast<uint32_t>(base) + d->source_segment_offset;
                }
                __gm__ PlInvocationWorkspaceDesc *ws_recv = PlFullDispatchInvocationWorkspace(sym, cfg, epoch);
                if (ws_recv == nullptr || final_slot >= ws_recv->output_route_capacity) {
                    ++rc->verify_error_count;
                    rc->error_code = kPlErrOutputCapacityInsufficient;
                    PlDcciRecvCtr(rc);
                    return;
                }
                const bool local_final = PlDevWorkspaceIsWorkerLocalFinal(ws_recv);
                if (local_final) {
                    // R5 V1：observe tail/generation already done above; PipeBarrier then MTE-only.
                    // 禁止 per-token ring payload DCCI（R4 persistent PASS；保持 O1 output）。
                    __gm__ uint8_t *ring_payload =
                        sym + DevPlDestChannelPayloadOff(source_id) + static_cast<uint64_t>(slot) * nbytes;
                    AscendC::PipeBarrier<PIPE_ALL>();
                    // ring_payload_dcci_* 保持 0（不调用 Qv2DcciRange on ring）
                    __gm__ uint8_t *expand_dst =
                        reinterpret_cast<__gm__ uint8_t *>(ws_recv->expand_x_ptr) +
                        static_cast<uint64_t>(final_slot) * nbytes;
                    const uint64_t copy_t0 = AscendC::GetSystemCycle();
                    ++lf_copy_issue;
                    // per-token MTE GM→UB→GM；每 recv AIV 独占 UB@0/event
                    {
                        __ubuf__ uint8_t *ub = reinterpret_cast<__ubuf__ uint8_t *>(0);
                        DcLlGatherCopyGm2GmChunked(reinterpret_cast<GM_ADDR>(expand_dst),
                                                  reinterpret_cast<GM_ADDR>(ring_payload), nbytes, ub,
                                                  kDcLlGatherEvent);
                    }
                    ++lf_copy_complete;
                    lf_copy_bytes += nbytes;
                    lf_copy_cycles += (AscendC::GetSystemCycle() - copy_t0);
                    if (lf_first_copy_cycle == 0u) {
                        lf_first_copy_cycle = copy_t0;
                    }
                    lf_last_copy_cycle = AscendC::GetSystemCycle();
                    payload_assist_ready_cycles += (AscendC::GetSystemCycle() - copy_t0);
                    __gm__ uint8_t *assist_line =
                        reinterpret_cast<__gm__ uint8_t *>(ws_recv->assist_info_ptr) +
                        static_cast<uint64_t>(final_slot) * kPlDispatchAssistStrideBytes;
                    __gm__ int32_t *assist = reinterpret_cast<__gm__ int32_t *>(assist_line);
                    const uint64_t assist_t0 = AscendC::GetSystemCycle();
                    assist[0] = static_cast<int32_t>(d->source_rank);
                    assist[1] = static_cast<int32_t>(d->source_token_id);
                    assist[2] = static_cast<int32_t>(d->topk_slot);
                    lf_assist_cycles += (AscendC::GetSystemCycle() - assist_t0);
                    ++lf_assist_count;
                    payload_assist_ready_cycles += (AscendC::GetSystemCycle() - assist_t0);
                    rct->assist_done_cycle = AscendC::GetSystemCycle();
                    // C-output O1：MTE3 wait 已保证本 AIV GM 写完成；PipeBarrier 后发布 ready。
                    // 禁止无意义 per-token output DCCI（microbench：aiv1 1000-epoch exact + host D2H）。
                    AscendC::PipeBarrier<PIPE_ALL>();
                    // output_dcci counters：本路径为 0 bytes（显式记录语义）
                    lf_out_dcci_cycles += 0u;
                    // keep counts at 0; bytes stay 0
                } else {
                    rc->error_code = kPlErrWorkspaceFlagInvalid;
                    PlDcciRecvCtr(rc);
                    return;
                }
            }

            // Full-dispatch local-final completion is RecvDone after the MTE3
            // wait above.  The legacy per-final-slot 64B ready line has no
            // consumer here and its fixed symmetric region cannot represent
            // runtime capacities up to 524288 slots; writing it corrupted D1
            // control state on hot destinations. Keep it only for the legacy
            // non-full-dispatch contract.
            if (desc->measurement_mode != kPlMeasurementFullDispatch) {
                __gm__ PlTokenReadyLine *ready = reinterpret_cast<__gm__ PlTokenReadyLine *>(
                    sym + DevPlTokenReadyLineOff(PlDescriptorPayloadSlot(d, desc)));
                const uint64_t ready_t0 = AscendC::GetSystemCycle();
                ready->ready_epoch = epoch;
                ready->token_index = PlDescriptorPayloadSlot(d, desc);
                ready->magic = kPlMagic;
                PlDcci(reinterpret_cast<__gm__ uint8_t *>(ready));
                lf_ready_cycles += (AscendC::GetSystemCycle() - ready_t0);
                ++lf_ready_count;
            }

            ++seen;
            last_progress_cycle = AscendC::GetSystemCycle();
            last_descriptor_seq = static_cast<uint32_t>(seen - 1u);
            ++rc->tokens_received;
            ++rc->token_ready_publish_count;
            if (sm_full) {
                __gm__ PlP5CellRouteRecvLedgerLine *cell_ledger =
                    reinterpret_cast<__gm__ PlP5CellRouteRecvLedgerLine *>(
                        sym + DevPlP5CellRouteRecvLedgerLineOff(source_id, desc->pair_id));
                cell_ledger->recv_seen = static_cast<uint32_t>(seen);
                cell_ledger->last_descriptor_seq = static_cast<uint32_t>(seen - 1u);
                cell_ledger->magic = kPlP5CellLedgerMagic;
                // Diagnostic ledger is not a progress primitive.  Publish it
                // with the credit batch below instead of flushing one
                // cacheline per token.
            }

            const uint64_t pending = seen - last_published_head;
            if (pending >= credit_batch || seen >= seq_end) {
                const uint64_t remote_credit_off = DevPlIncEgressCreditLineOff(desc->pair_id);
                const uint64_t head_t0 = AscendC::GetSystemCycle();
                credit_scratch->magic = inc::dc::dn::qv2::kQv2Magic;
                PlPublishEgressHeadSignaled(credit_scratch, sym, remote_credit_off, inc_peer, seen, epoch, head_pub,
                                            true, head_quiet);
                const uint64_t credit_dt = AscendC::GetSystemCycle() - head_t0;
                head_credit_publish_cycles += credit_dt;
                lf_credit_cycles += credit_dt;
                ++lf_credit_count;
                if (lf_first_credit_cycle == 0u) {
                    lf_first_credit_cycle = head_t0;
                }
                last_published_head = seen;
                rc->last_credit_head_published = seen;
                if (sm_full) {
                    __gm__ PlP5CellRouteRecvLedgerLine *cell_ledger =
                        reinterpret_cast<__gm__ PlP5CellRouteRecvLedgerLine *>(
                            sym + DevPlP5CellRouteRecvLedgerLineOff(
                                      source_id, desc->pair_id));
                    PlDcci(reinterpret_cast<__gm__ uint8_t *>(cell_ledger));
                }
                rct->last_credit_publish_cycle = AscendC::GetSystemCycle();
            }
        }
            rc->channel_seen = seen;
        } else {
            AscendC::PipeBarrier<PIPE_ALL>();
            PlSpinWaitSignalLineDcci(reinterpret_cast<__gm__ uint8_t *>(egress_tail),
                                     static_cast<uint32_t>(rc->tail_poll_cycles));
            tail_wait_total_cycles += (AscendC::GetSystemCycle() - poll_t0);
        }
    }

    rc->head_publish_count += head_pub;
    rc->head_completion_quiet_count += head_quiet;
    rc->slot_overwrite_count = 0;
    PlDcciRecvCtr(rc);
    // Full Dispatch：recv 只写独占 RecvDoneLine；payload/assist/full_done 仅 control 写
    if (desc->measurement_mode == kPlMeasurementFullDispatch) {
        const uint32_t epoch_recv = static_cast<uint32_t>(seen - (seq_end - tokens_expected));
        const uint64_t rd_t0 = AscendC::GetSystemCycle();
        const uint32_t recv_error =
            (rc->error_code != 0u) ? rc->error_code
                                   : ((epoch_recv == tokens_expected) ? 0u : kPlErrRecvIncomplete);
        PlPublishRecvDone(sym, source_id, epoch, tokens_expected, epoch_recv,
                          static_cast<uint32_t>(last_published_head), recv_error, last_remote_tail,
                          last_descriptor_seq);
        recv_done_publish_cycles = AscendC::GetSystemCycle() - rd_t0;
    }
    const uint64_t recv_done_cyc = AscendC::GetSystemCycle();
    rct->recv_done_cycle = recv_done_cyc;
    rct->tail_poll_cycles = tail_poll_local;
    rct->tail_signal_wake_count = tail_signal_wake_local;
    rct->descriptor_retry_count = descriptor_retry_local;
    rct->head_publish_count = static_cast<uint32_t>(head_pub);
    rct->head_quiet_count = static_cast<uint32_t>(head_quiet);
    rct->counts_done_wait_cycles = counts_done_wait_cycles;
    rct->wait_first_tail_cycles = wait_first_tail_cycles;
    rct->tail_wait_total_cycles = tail_wait_total_cycles;
    rct->descriptor_read_verify_cycles = descriptor_read_verify_cycles;
    rct->payload_assist_ready_cycles = payload_assist_ready_cycles;
    rct->head_credit_publish_cycles = head_credit_publish_cycles;
    rct->recv_done_publish_cycles = recv_done_publish_cycles;
    rct->token_count = static_cast<uint32_t>(seen - (seq_end - tokens_expected));
    rct->stale_count = stale_local;
    rct->duplicate_count = dup_local;
    rct->lost_count = lost_local;
    rct->overwrite_count = 0u;
    rct->local_copy_issue_count = lf_copy_issue;
    rct->local_copy_complete_count = lf_copy_complete;
    rct->local_copy_bytes = lf_copy_bytes;
    rct->local_copy_cycles = lf_copy_cycles;
    rct->ring_payload_dcci_count = lf_ring_dcci_count;
    rct->ring_payload_dcci_bytes = lf_ring_dcci_bytes;
    rct->ring_payload_dcci_cycles = lf_ring_dcci_cycles;
    rct->output_dcci_count = lf_out_dcci_count;
    rct->output_dcci_bytes = lf_out_dcci_bytes;
    rct->output_dcci_cycles = lf_out_dcci_cycles;
    rct->assist_write_count = lf_assist_count;
    rct->assist_write_cycles = lf_assist_cycles;
    rct->ready_publish_count = lf_ready_count;
    rct->ready_publish_cycles = lf_ready_cycles;
    rct->credit_publish_count = lf_credit_count;
    rct->credit_publish_cycles = lf_credit_cycles;
    rct->first_local_copy_cycle = lf_first_copy_cycle;
    rct->last_local_copy_cycle = lf_last_copy_cycle;
    rct->first_credit_publish_cycle_lf = lf_first_credit_cycle;
    rct->local_copy_before_final_forward_count = 0u; // filled by host/D4 overlap reconcile
    rct->magic = kPlRecvChanCompletionTraceMagic;
    PlDcciTrace384(reinterpret_cast<__gm__ uint8_t *>(rct));
    const uint64_t recv_first_cyc = trace->first_issue_cycle > 0 ? trace->first_issue_cycle : recv_done_cyc;
    const PlWorkerServiceLayout recv_lay = PlDevWorkerServiceLayout(desc);
    const uint32_t recv_aiv_block =
        recv_lay.recv_begin + (recv_lay.recv_lane_count == 0u ? 0u : (source_id % recv_lay.recv_lane_count));
    PlPublishAivEpochTiming(sym, DevPlRecvAivTimingOff(source_id), recv_first_cyc, recv_done_cyc, epoch,
                            recv_aiv_block);
}

// control 聚合 AIV 独占 timing → 共享 PlPipelineTiming（仅 control block 写）
__aicore__ inline void PlAggregateWorkerAivTiming(__gm__ uint8_t *base, __gm__ PlPipelineDesc *desc,
                                                  __gm__ PlPipelineTiming *timing, uint64_t run_epoch)
{
    uint64_t min_upload = UINT64_MAX;
    uint64_t max_upload_done = 0;
    for (uint32_t i = 0; i < desc->lane_count; ++i) {
        __gm__ PlAivEpochTimingLine *ul =
            reinterpret_cast<__gm__ PlAivEpochTimingLine *>(base + DevPlUploadAivTimingOff(i));
        PlDcciDoneLine(reinterpret_cast<__gm__ uint8_t *>(ul));
        if (ul->done_epoch == run_epoch && ul->first_issue_cycle > 0) {
            if (ul->first_issue_cycle < min_upload) {
                min_upload = ul->first_issue_cycle;
            }
            if (ul->done_cycle > max_upload_done) {
                max_upload_done = ul->done_cycle;
            }
        }
    }
    uint64_t min_recv_first = UINT64_MAX;
    uint64_t max_recv_done = 0;
    for (uint32_t s = 0; s < desc->worker_count; ++s) {
        __gm__ PlAivEpochTimingLine *rl =
            reinterpret_cast<__gm__ PlAivEpochTimingLine *>(base + DevPlRecvAivTimingOff(s));
        PlDcciDoneLine(reinterpret_cast<__gm__ uint8_t *>(rl));
        if (rl->done_epoch == run_epoch && rl->first_issue_cycle > 0) {
            if (rl->first_issue_cycle < min_recv_first) {
                min_recv_first = rl->first_issue_cycle;
            }
            if (rl->done_cycle > max_recv_done) {
                max_recv_done = rl->done_cycle;
            }
        }
    }
    if (min_upload != UINT64_MAX) {
        timing->first_upload_issue = min_upload;
    }
    if (max_upload_done > 0) {
        timing->last_upload_issue = max_upload_done;
        PlDevRankTimingMarkUploadDone(base, max_upload_done);
    }
    if (min_recv_first != UINT64_MAX) {
        timing->first_destination_consume = min_recv_first;
    }
    if (max_recv_done > 0) {
        timing->destination_done_cycle = max_recv_done;
    }
}

__aicore__ inline void PlWorkerPublishDestinationDone(__gm__ uint8_t *base, __gm__ PlPipelineDesc *desc,
                                                      __gm__ PlPipelineTiming *timing, __gm__ PlOverlapTelemetry *overlap,
                                                      uint64_t dest_done_off, uint64_t d1_leader_completion_off,
                                                      uint64_t d1_publish_scratch_off, uint64_t run_epoch,
                                                      int leader_pe, uint64_t total_recv)
{
    timing->destination_done_epoch = static_cast<uint32_t>(run_epoch);
    PlAggregateWorkerAivTiming(base, desc, timing, run_epoch);
    __gm__ PlDestDoneLine *ddone = reinterpret_cast<__gm__ PlDestDoneLine *>(base + dest_done_off);
    ddone->done_epoch = run_epoch;
    ddone->destination_rank = desc->pair_id;
    ddone->tokens_received = total_recv;
    ddone->magic = kPlMagic;
    PlDcciDoneLine(reinterpret_cast<__gm__ uint8_t *>(ddone));
    __gm__ PlDestinationCompletionTrace *dct = reinterpret_cast<__gm__ PlDestinationCompletionTrace *>(
        base + DevPlDestinationCompletionTraceOff());
    if (dct != nullptr && dct->epoch == run_epoch) {
        dct->received_total = static_cast<uint32_t>(total_recv);
        dct->pair_done_publish_start_cycle = AscendC::GetSystemCycle();
    }
    PlPublishPairDone(base, d1_leader_completion_off, desc->pair_id, run_epoch, d1_publish_scratch_off, leader_pe);
    if (dct != nullptr && dct->epoch == run_epoch) {
        dct->pair_done_publish_end_cycle = AscendC::GetSystemCycle();
        dct->magic = kPlDestCompletionTraceMagic;
        PlDcciTrace128(reinterpret_cast<__gm__ uint8_t *>(dct));
    }
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(timing));
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(timing) + 64u);
    (void)overlap;
}

// Control AIV only: semantic complete → cleanup → retire line → return WaitStart.
// telemetry_valid=0 is allowed; must not block host epoch N+1.
__aicore__ inline void PlWorkerPublishControlEpochRetire(__gm__ uint8_t *base, uint64_t run_epoch,
                                                         uint64_t generation, uint32_t error_code,
                                                         uint32_t telemetry_valid)
{
    __gm__ ControlEpochRetireLine *line =
        reinterpret_cast<__gm__ ControlEpochRetireLine *>(base + kPlControlEpochRetireOff);
    line->epoch = run_epoch;
    line->generation = generation;
    line->semantic_done_epoch = run_epoch;
    line->post_epoch_cleanup_done = 1u;
    line->error_code = error_code;
    line->magic = kPlControlEpochRetireMagic;
    line->telemetry_valid = telemetry_valid;
    // Publish payload fields before retire_signal so host never observes signal without epoch.
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(line));
    line->retire_signal = run_epoch;
    PlDcci(reinterpret_cast<__gm__ uint8_t *>(line));
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void inc_dc_dn_pipeline_worker_persistent_kernel(
    GM_ADDR sym, uint64_t pl_desc_off, uint64_t upload_ctr_off, uint64_t recv_ctr_off, uint64_t timing_off,
    uint64_t overlap_off, uint64_t worker_trace_off, uint64_t source_done_off, uint64_t dest_done_off,
    uint64_t d1_start_off, uint64_t d1_stop_off, uint64_t d1_leader_completion_off, uint64_t d1_publish_scratch_off,
    uint32_t leader_pe)
{
    if ASCEND_IS_AIV {
        const int bi = AscendC::GetBlockIdx();
        if (bi < 0 || bi >= static_cast<int>(kPlWorkerServiceBlockDim)) {
            return;
        }
        __gm__ uint8_t *base = reinterpret_cast<__gm__ uint8_t *>(sym);
        __gm__ PlPipelineDesc *desc = reinterpret_cast<__gm__ PlPipelineDesc *>(base + pl_desc_off);
        const PlWorkerServiceLayout wlay = PlDevWorkerServiceLayout(desc);
        if (bi >= static_cast<int>(wlay.block_dim)) {
            return;
        }
        const uint32_t control_bi = wlay.control_block;
        __gm__ PlPipelineTiming *timing = reinterpret_cast<__gm__ PlPipelineTiming *>(base + timing_off);
        __gm__ PlOverlapTelemetry *overlap = reinterpret_cast<__gm__ PlOverlapTelemetry *>(base + overlap_off);
        __gm__ D1StartLine *start = reinterpret_cast<__gm__ D1StartLine *>(base + d1_start_off);
        __gm__ D1SessionStopLine *stop = reinterpret_cast<__gm__ D1SessionStopLine *>(base + d1_stop_off);
        __gm__ PlServiceTraceLine *trace =
            reinterpret_cast<__gm__ PlServiceTraceLine *>(base + worker_trace_off) + bi;

        trace->block_id = static_cast<uint32_t>(bi);
        trace->kernel_enter_cycle = AscendC::GetSystemCycle();
        trace->desc_valid = (desc->magic == kPlMagic && desc->my_role == 0u) ? 1u : 0u;
        PlTraceDcci(trace);

        __gm__ PlResidentLine *resident =
            reinterpret_cast<__gm__ PlResidentLine *>(base + kPlWorkerResidentOff + static_cast<uint64_t>(bi) * 64u);
        resident->value = 1;
        resident->block_id = static_cast<uint32_t>(bi);
        resident->magic = kPlMagic;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(resident));

        if (trace->desc_valid == 0u) {
            trace->kernel_exit_reason = kPlKernelExitResource;
            PlTraceDcci(trace);
            return;
        }

        const bool device_full_verify = (desc->verify_mode == kPlVerifyDeviceFull);
        __gm__ PlFullDispatchConfig *fdc =
            reinterpret_cast<__gm__ PlFullDispatchConfig *>(base + kPlFullDispatchConfigOff);
        const bool full_dispatch = (desc->measurement_mode == kPlMeasurementFullDispatch);
        uint64_t last_epoch = 0;

        while (true) {
            if (PlPollStop(stop, trace)) {
                break;
            }
            const uint64_t run_epoch = PlWaitStartPersistent(start, stop, last_epoch, trace);
            if (run_epoch == 0) {
                if (PlPollStop(stop, trace)) {
                    break;
                }
                continue;
            }
            if (timing->epoch_start_cycle == 0) {
                timing->epoch_start_cycle = AscendC::GetSystemCycle();
            }

            // control-only：仅验证 start 可见 + control 聚合，不跑 upload/recv 数据面
            if ((desc->pipeline_mode & kPlModeBaseMask) == kPlModeControlOnly) {
                if (bi == static_cast<int>(control_bi)) {
                    __gm__ PlDestDoneLine *ddone =
                        reinterpret_cast<__gm__ PlDestDoneLine *>(base + dest_done_off);
                    ddone->done_epoch = run_epoch;
                    ddone->destination_rank = desc->pair_id;
                    ddone->tokens_received = 0;
                    ddone->magic = kPlMagic;
                    PlDcci(reinterpret_cast<__gm__ uint8_t *>(ddone));
                    timing->destination_done_epoch = static_cast<uint32_t>(run_epoch);
                    timing->destination_done_cycle = AscendC::GetSystemCycle();
                    PlPublishPairDone(base, d1_leader_completion_off, desc->pair_id, run_epoch,
                                      d1_publish_scratch_off, static_cast<int>(leader_pe));
                }
                last_epoch = run_epoch;
                continue;
            }

            // P5 count 环解除：Worker control 独占 SourcePublish + DestPrefix（与 upload 并行）
            // upload AIV 不得再做 count/prefix，也不得写 lane_work（属 count-forward 合同）
            if (full_dispatch && bi == static_cast<int>(control_bi)) {
                PlDcci(reinterpret_cast<__gm__ uint8_t *>(fdc));
                if (PlIsSourceMajorRaw(desc)) {
                    // Count scan also builds canonical route ordinals.  Make
                    // both visible before the metadata-ready doorbell.
                    if (desc->worker_count <= 1u) {
                        PlFullDispatchSourcePublishCounts(base, desc, fdc, run_epoch);
                        PlFullDispatchSourcePublishRouteMeta(base, desc, fdc, run_epoch);
                        PlFullDispatchCountPrefixLocal(base, desc, fdc, run_epoch);
                    } else {
                        PlFullDispatchSourcePublishCounts(base, desc, fdc, run_epoch);
                        PlFullDispatchSourcePublishRouteMeta(base, desc, fdc, run_epoch);
                        PlFullDispatchDestCountPrefix(base, desc, fdc, run_epoch);
                    }
                } else if (desc->worker_count <= 1u) {
                    PlFullDispatchCountPrefix(base, desc, fdc, run_epoch);
                } else {
                    // dest-major multi-worker：control 独占 source count；direct 模式的 destination
                    // prefix 由 recv lane0 并行执行，避免 SourcePublish→DestPrefix 串行固定开销。
                    PlFullDispatchSourcePublishCounts(base, desc, fdc, run_epoch);
                    if ((desc->pipeline_mode & inc::dc::dn::pl::kPlModeDirectCountExchange) == 0u) {
                        PlFullDispatchDestCountPrefix(base, desc, fdc, run_epoch);
                    }
                }
            }

            if (bi >= static_cast<int>(wlay.upload_begin) &&
                bi < static_cast<int>(wlay.recv_begin)) {
                const uint32_t lane_id = static_cast<uint32_t>(bi - static_cast<int>(wlay.upload_begin));
                if (lane_id < desc->lane_count && lane_id < wlay.upload_lane_count) {
                    // count/prefix 已迁至 Worker control；upload 仅数据面
                    __gm__ PlUploadCounters *uc =
                        reinterpret_cast<__gm__ PlUploadCounters *>(base + upload_ctr_off) + lane_id;
                    if (full_dispatch) {
                        PlDcci(reinterpret_cast<__gm__ uint8_t *>(fdc));
                        if (PlIsSourceMajorRaw(desc)) {
                            // Direct-count has exactly one writer per
                            // source->destination slice: upload lane d.  The
                            // partner INC intentionally skips publishing
                            // these slices when the flag is set, so the raw
                            // ingress path must publish before entering its
                            // payload loop as well (the old dest-major path
                            // already did this inside RouteGather).
                            const bool count_ok =
                                inc::dc::dn::pl::PlWorkerPublishDirectCountSlice(
                                    base, desc, lane_id, run_epoch);
                            if (!count_ok) {
                                uc->error_code =
                                    inc::dc::dn::pl::kPlCountFwdSourceReadyTimeout;
                                PlDcci(reinterpret_cast<__gm__ uint8_t *>(uc));
                            } else if (fdc->probe_mode !=
                                       kPlFullDispatchProbeCountOnly) {
                                // P5：source-major raw upload；禁止写 lane_work_off
                                PlWorkerSourceMajorRawUploadEpoch(base, desc, fdc, uc, trace, lane_id,
                                                                  run_epoch);
                            }
                        } else {
                            if (fdc->probe_mode == kPlFullDispatchProbeCountOnly) {
                                if (!inc::dc::dn::pl::PlWorkerPublishDirectCountSlice(
                                        base, desc, lane_id, run_epoch)) {
                                    uc->error_code = inc::dc::dn::pl::kPlCountFwdSourceReadyTimeout;
                                    PlDcci(reinterpret_cast<__gm__ uint8_t *>(uc));
                                }
                            } else {
                                PlWorkerRouteGatherUploadLaneEpoch(base, desc, fdc, uc, timing, trace, lane_id,
                                                                 run_epoch);
                                if (fdc->probe_mode != kPlFullDispatchProbeGatherOnly) {
                                    const uint64_t tail_before = uc->local_tail;
                                    PlWorkerUploadLaneEpoch(base, desc, uc, timing, trace, lane_id, run_epoch);
                                    __gm__ PlLaneWorkLine *lw = reinterpret_cast<__gm__ PlLaneWorkLine *>(
                                        base + fdc->lane_work_off + static_cast<uint64_t>(lane_id) * 64u);
                                    lw->upload_count = static_cast<uint32_t>(uc->local_tail - tail_before);
                                    PlDcci(reinterpret_cast<__gm__ uint8_t *>(lw));
                                }
                            }
                        }
                    } else {
                        PlWorkerUploadLaneEpoch(base, desc, uc, timing, trace, lane_id, run_epoch);
                    }
                    if (uc->error_code != 0) {
                        timing->error_code = uc->error_code;
                        trace->kernel_exit_reason = kPlKernelExitUpload;
                        PlTraceDcci(trace);
                        return;
                    }
                }
            } else if (bi >= static_cast<int>(wlay.recv_begin) && bi < static_cast<int>(control_bi)) {
                // Candidate1+2：recv_lane_id 拥有 logical_ch = lane, lane+R, ...；block = [U, U+R)
                const uint32_t recv_lane_id = static_cast<uint32_t>(bi - static_cast<int>(wlay.recv_begin));
                const uint32_t recv_lanes = wlay.recv_lane_count;
                uint32_t logical_n = desc->worker_count;
                bool skip_recv = false;
                if (full_dispatch) {
                    PlDcci(reinterpret_cast<__gm__ uint8_t *>(fdc));
                    skip_recv = (fdc->probe_mode != kPlFullDispatchProbeFull);
                    if (!PlIsSourceMajorRaw(desc) && desc->worker_count > 1u && recv_lane_id == 0u &&
                        (desc->pipeline_mode & inc::dc::dn::pl::kPlModeDirectCountExchange) != 0u) {
                        PlFullDispatchDestCountPrefix(base, desc, fdc, run_epoch);
                    }
                }
                if (logical_n > kPlMaxLogicalRecvChannels) {
                    logical_n = kPlMaxLogicalRecvChannels;
                }
                if (!skip_recv && recv_lane_id < recv_lanes) {
                    // Round-robin 顺序遍历本 lane 的 channel；U8/R8 时与旧 1:1 等价
                    for (uint32_t ch = recv_lane_id; ch < logical_n; ch += recv_lanes) {
                        if (ch >= kPlMaxLogicalRecvChannels || ch >= desc->worker_count) {
                            continue;
                        }
                        __gm__ PlRecvCounters *rc =
                            reinterpret_cast<__gm__ PlRecvCounters *>(base + recv_ctr_off) + ch;
                        PlWorkerRecvChannelEpoch(base, desc, rc, timing, overlap, trace, ch, run_epoch,
                                                 device_full_verify);
                        if (rc->error_code != 0) {
                            timing->error_code = rc->error_code;
                            trace->kernel_exit_reason = kPlKernelExitRecv;
                            PlTraceDcci(trace);
                            return;
                        }
                        __gm__ PlSourceDoneLine *sdone =
                            reinterpret_cast<__gm__ PlSourceDoneLine *>(base + source_done_off + ch * 64u);
                        sdone->done_epoch = run_epoch;
                        sdone->source_rank = ch;
                        sdone->magic = kPlMagic;
                        PlDcci(reinterpret_cast<__gm__ uint8_t *>(sdone));
                    }
                }
            } else if (bi == static_cast<int>(control_bi)) {
                {
                    uint32_t rank_role = kPlRoleMaskWorker;
                    if (aclshmem_my_pe() == static_cast<int>(leader_pe)) {
                        rank_role |= kPlRoleMaskPe0Observer;
                    }
                    uint32_t gen = 0u;
                    if (full_dispatch) {
                        __gm__ PlInvocationDesc *inv_start = PlFullDispatchInvocation(base, fdc, run_epoch);
                        if (inv_start != nullptr) {
                            gen = static_cast<uint32_t>(inv_start->generation);
                        }
                    }
                    PlDevRankTimingMarkStart(base, rank_role, run_epoch, gen);
                }
                const uint32_t expect_recv = (1u << desc->worker_count) - 1u;
                const bool use_done_lines = (desc->completion_mode == kPlCompletionModeDoneLines);
                const bool full_dispatch = (desc->measurement_mode == kPlMeasurementFullDispatch);
                uint32_t probe_mode = kPlFullDispatchProbeFull;
                if (full_dispatch) {
                    PlDcci(reinterpret_cast<__gm__ uint8_t *>(fdc));
                    probe_mode = fdc->probe_mode;
                }
                uint32_t spins = 0;
                bool completed = false;
                bool second_hop_timing_ok = false;
                bool recv_path_done = false;
                // S3 等多 epoch 大 token：8e6 会在 payload/recv 未齐时误报 DestAgg
                const uint32_t ctrl_spins_max =
                    (full_dispatch && desc->worker_count > 1u) ? 80000000u : 8000000u;
                __gm__ PlDestinationCompletionTrace *dct = nullptr;
                if (full_dispatch && probe_mode == kPlFullDispatchProbeFull) {
                    dct = reinterpret_cast<__gm__ PlDestinationCompletionTrace *>(base + DevPlDestinationCompletionTraceOff());
                    PlClearTrace128(reinterpret_cast<__gm__ uint8_t *>(dct));
                    if (!PlDevRequireInvocationWorkspace(base, fdc, run_epoch)) {
                        __gm__ PlFullOutputDoneLine *fout_ws =
                            reinterpret_cast<__gm__ PlFullOutputDoneLine *>(base + fdc->full_output_done_off);
                        PlDcci(reinterpret_cast<__gm__ uint8_t *>(fout_ws));
                        timing->error_code = fout_ws->error_code;
                        trace->kernel_exit_reason = kPlKernelExitForward;
                        PlTraceDcci(trace);
                        return;
                    }
                    __gm__ PlInvocationDesc *inv_init = PlFullDispatchInvocation(base, fdc, run_epoch);
                    dct->epoch = run_epoch;
                    dct->generation = static_cast<uint32_t>(inv_init->generation);
                    dct->destination_rank = desc->pair_id;
                    dct->recv_done_mask = 0u;
                    dct->error_code = 0u;
                }
                while (spins < ctrl_spins_max) {
                    // PE0：非阻塞轮询 INC block9 summary → 正式 second_hop timing（禁止阻塞 full_done）
                    if (full_dispatch && probe_mode == kPlFullDispatchProbeFull &&
                        aclshmem_my_pe() == static_cast<int>(leader_pe) && !second_hop_timing_ok) {
                        __gm__ PlInvocationDesc *inv = PlFullDispatchInvocation(base, fdc, run_epoch);
                        if (PlPe0TryPublishGlobalSecondHopVisibleTiming(base, desc->worker_count, run_epoch,
                                                                        inv->generation, control_bi)) {
                            second_hop_timing_ok = true;
                        }
                    }
                    // generation mismatch 等致命错误：禁止继续等 full_done
                    if (full_dispatch) {
                        __gm__ PlFullOutputDoneLine *fout_err =
                            reinterpret_cast<__gm__ PlFullOutputDoneLine *>(base + fdc->full_output_done_off);
                        PlDcci(reinterpret_cast<__gm__ uint8_t *>(fout_err));
                        if (fout_err->error_code == kPlErrGenerationMismatch ||
                            fout_err->error_code == kPlErrWorkspaceDescMissing ||
                            fout_err->error_code == kPlErrWorkspaceGenerationMismatch ||
                            fout_err->error_code == kPlErrWorkspacePointerAlignment ||
                            fout_err->error_code == kPlErrOutputCapacityInsufficient ||
                            fout_err->error_code == kPlErrWorkspaceFlagInvalid) {
                            timing->error_code = fout_err->error_code;
                            trace->kernel_exit_reason = kPlKernelExitForward;
                            PlTraceDcci(trace);
                            return;
                        }
                    }
                    if (full_dispatch && probe_mode == kPlFullDispatchProbeCountOnly) {
                        __gm__ PlFullOutputDoneLine *fout =
                            reinterpret_cast<__gm__ PlFullOutputDoneLine *>(base + fdc->full_output_done_off);
                        PlDcci(reinterpret_cast<__gm__ uint8_t *>(fout));
                        if (fout->counts_done_epoch >= run_epoch) {
                            timing->destination_done_epoch = static_cast<uint32_t>(run_epoch);
                            timing->destination_done_cycle = AscendC::GetSystemCycle();
                            PlWorkerPublishDestinationDone(base, desc, timing, overlap, dest_done_off,
                                                           d1_leader_completion_off, d1_publish_scratch_off,
                                                           run_epoch, static_cast<int>(leader_pe), 0);
                            completed = true;
                            break;
                        }
                    } else if (full_dispatch && probe_mode == kPlFullDispatchProbeGatherOnly) {
                        uint32_t gather_mask = 0;
                        for (uint32_t i = 0; i < desc->lane_count; ++i) {
                            __gm__ PlLaneWorkLine *lw = reinterpret_cast<__gm__ PlLaneWorkLine *>(
                                base + fdc->lane_work_off + static_cast<uint64_t>(i) * 64u);
                            PlDcci(reinterpret_cast<__gm__ uint8_t *>(lw));
                            if (lw->done_epoch >= run_epoch && lw->gather_count > 0u && lw->upload_count == 0u) {
                                gather_mask |= (1u << i);
                            }
                        }
                        if (gather_mask == ((1u << desc->lane_count) - 1u)) {
                            timing->destination_done_epoch = static_cast<uint32_t>(run_epoch);
                            timing->destination_done_cycle = AscendC::GetSystemCycle();
                            PlWorkerPublishDestinationDone(base, desc, timing, overlap, dest_done_off,
                                                           d1_leader_completion_off, d1_publish_scratch_off,
                                                           run_epoch, static_cast<int>(leader_pe), 0);
                            completed = true;
                            break;
                        }
                    } else if (full_dispatch && probe_mode == kPlFullDispatchProbeFirstHop) {
                        uint32_t hop_mask = 0;
                        if (PlIsSourceMajorRaw(desc)) {
                            // P5：8-lane 并行 unique upload；INC 只写 first_hop_done[0]
                            __gm__ PlFirstHopDoneLine *fhd = reinterpret_cast<__gm__ PlFirstHopDoneLine *>(
                                base + fdc->first_hop_done_off);
                            PlDcciDoneLine(reinterpret_cast<__gm__ uint8_t *>(fhd));
                            const uint64_t expect_tokens =
                                static_cast<uint64_t>(desc->tokens_per_epoch) * run_epoch;
                            // error_code!=0：fail-closed 完成，避免 Worker 永等
                            if (fhd->epoch >= run_epoch && fhd->magic == kPlMagic &&
                                (fhd->error_code != 0u ||
                                 (fhd->error_code == 0u && fhd->tokens_received >= expect_tokens))) {
                                hop_mask = (1u << desc->lane_count) - 1u;
                                if (fhd->error_code != 0u) {
                                    timing->error_code = fhd->error_code;
                                }
                            }
                        } else {
                        for (uint32_t i = 0; i < desc->lane_count; ++i) {
                            const uint64_t expect_lane =
                                static_cast<uint64_t>(PlEpochTokensExpected(base, desc, i)) * run_epoch;
                            __gm__ PlFirstHopDoneLine *fhd = reinterpret_cast<__gm__ PlFirstHopDoneLine *>(
                                base + fdc->first_hop_done_off + static_cast<uint64_t>(i) * 64u);
                            PlDcciDoneLine(reinterpret_cast<__gm__ uint8_t *>(fhd));
                            __gm__ PlUploadCounters *uc =
                                reinterpret_cast<__gm__ PlUploadCounters *>(base + upload_ctr_off) + i;
                            PlDcciUploadCtr(uc);
                            if (fhd->epoch >= run_epoch && fhd->magic == kPlMagic && fhd->error_code == 0u &&
                                fhd->tokens_received >= expect_lane && fhd->ingress_head == uc->local_tail &&
                                uc->local_tail >= expect_lane) {
                                hop_mask |= (1u << i);
                            }
                        }
                        }
                        if (hop_mask == ((1u << desc->lane_count) - 1u)) {
                            timing->destination_done_epoch = static_cast<uint32_t>(run_epoch);
                            timing->destination_done_cycle = AscendC::GetSystemCycle();
                            PlDevRankTimingMarkDestinationDone(base, timing->destination_done_cycle);
                            PlDevRankTimingPublishLocalCompletion(base, kPlRoleMaskWorker);
                            PlWorkerPublishDestinationDone(base, desc, timing, overlap, dest_done_off,
                                                           d1_leader_completion_off, d1_publish_scratch_off,
                                                           run_epoch, static_cast<int>(leader_pe), 0);
                            // first_hop_done.error_code!=0：结构化失败，禁止假绿 pass
                            if (timing->error_code != 0u) {
                                trace->kernel_exit_reason = kPlKernelExitForward;
                                PlTraceDcci(trace);
                                return;
                            }
                            completed = true;
                            break;
                        }
                    } else if (full_dispatch && probe_mode == kPlFullDispatchProbeFull) {
                        if (!recv_path_done) {
                        // control 唯一 owner：聚合 RecvDoneLine → 写 payload/assist/full_done → dest_done
                        __gm__ PlFullOutputDoneLine *fout =
                            reinterpret_cast<__gm__ PlFullOutputDoneLine *>(base + fdc->full_output_done_off);
                        __gm__ PlRouteTimingLine *rt =
                            reinterpret_cast<__gm__ PlRouteTimingLine *>(base + fdc->route_timing_off);
                        __gm__ int32_t *expected =
                            reinterpret_cast<__gm__ int32_t *>(base + kPlExpectedCountOff);
                        Qv2DcciRange(reinterpret_cast<__gm__ uint8_t *>(expected),
                                     desc->worker_count * sizeof(int32_t));
                        if (dct != nullptr) {
                            PlDcci(reinterpret_cast<__gm__ uint8_t *>(fout));
                            if (dct->counts_done_cycle == 0u && fout->counts_done_epoch >= run_epoch) {
                                dct->counts_done_cycle = AscendC::GetSystemCycle();
                            }
                            uint32_t exp_total = 0u;
                            for (uint32_t si = 0; si < desc->worker_count; ++si) {
                                exp_total += expected[si] > 0 ? static_cast<uint32_t>(expected[si]) : 0u;
                            }
                            dct->expected_total = exp_total;
                        }
                        bool all_rd = true;
                        uint32_t sum_recv = 0u;
                        uint32_t sum_exp = 0u;
                        uint32_t expected_total = 0u;
                        for (uint32_t s = 0; s < desc->worker_count; ++s) {
                            expected_total += expected[s] > 0 ? static_cast<uint32_t>(expected[s]) : 0u;
                        }
                        uint32_t recv_mask = 0u;
                        uint32_t rd_err = 0u;
                        for (uint32_t s = 0; s < desc->worker_count; ++s) {
                            __gm__ PlRecvDoneLine *rdone =
                                reinterpret_cast<__gm__ PlRecvDoneLine *>(base + PlDevRecvDoneLineOff(s));
                            PlDcci(reinterpret_cast<__gm__ uint8_t *>(rdone));
                            if (rdone->magic != kPlRecvDoneMagic || rdone->epoch != run_epoch ||
                                rdone->source_rank != s) {
                                all_rd = false;
                                break;
                            }
                            const uint32_t exp_s =
                                expected[s] > 0 ? static_cast<uint32_t>(expected[s]) : 0u;
                            // Error is authoritative even when a failed recv happened to observe
                            // expected tokens; do not hide a tail regression behind count equality.
                            if (rdone->error_code != 0u) {
                                rd_err = rdone->error_code;
                                sum_recv += rdone->received;
                                all_rd = false;
                                break;
                            }
                            if (rdone->expected != exp_s || rdone->received != rdone->expected) {
                                rd_err = (rdone->error_code != 0u) ? rdone->error_code : 61u;
                                sum_recv += rdone->received;
                                all_rd = false;
                                break;
                            }
                            sum_recv += rdone->received;
                            sum_exp += rdone->expected;
                            recv_mask |= (1u << s);
                            if (dct != nullptr) {
                                const uint64_t rd_cyc = AscendC::GetSystemCycle();
                                if (dct->first_recv_cycle == 0u) {
                                    dct->first_recv_cycle = rd_cyc;
                                }
                                dct->last_recv_done_cycle = rd_cyc;
                                dct->recv_done_mask = recv_mask;
                            }
                        }
                        if (all_rd && sum_recv == sum_exp && recv_mask == expect_recv) {
                            if (dct != nullptr) {
                                dct->all_recv_done_cycle = AscendC::GetSystemCycle();
                                dct->recv_done_mask = recv_mask;
                            }
                            AscendC::PipeBarrier<PIPE_ALL>();
                            // 正式 transport 完成点：RecvDone 齐；先于 full_done / four-output compact
                            PlPublishTransportDone(base, desc->pair_id, run_epoch, sum_exp, sum_recv, 0u,
                                                   static_cast<int>(leader_pe));
                            PlDevRankTimingPublishLocalCompletion(base, kPlRoleMaskWorker);
                            if (dct != nullptr) {
                                dct->transport_done_local_cycle = AscendC::GetSystemCycle();
                            }
                            if (aclshmem_my_pe() == static_cast<int>(leader_pe)) {
                                // PE0 control 单 writer：观察齐全部 dest → global_transport_done
                                if (dct != nullptr) {
                                    dct->global_transport_wait_start = AscendC::GetSystemCycle();
                                }
                                const PlGlobalTransportResult global_result =
                                    PlPe0PublishGlobalTransportTiming(base, desc->worker_count, run_epoch,
                                                                      control_bi);
                                if (global_result != kPlGlobalTransportOk) {
                                    const uint32_t global_error =
                                        (global_result == kPlGlobalTransportPeerError)
                                            ? kPlErrGlobalTransportPeerError
                                            : kPlErrGlobalTransportMissing;
                                    timing->error_code = global_error;
                                    fout->error_code = global_error;
                                    PlDcci(reinterpret_cast<__gm__ uint8_t *>(fout));
                                    trace->kernel_exit_reason = kPlKernelExitDestAgg;
                                    PlTraceDcci(trace);
                                    return;
                                }
                                if (dct != nullptr) {
                                    dct->global_transport_wait_end = AscendC::GetSystemCycle();
                                }
                            }
                            // full_done 与 second_hop_visible 解耦：禁止在此等待 global_second_hop
                            fout->payload_done_epoch = run_epoch;
                            fout->assist_done_epoch = run_epoch;
                            fout->full_done_epoch = run_epoch;
                            // delay_finalize：只拉长 full_done，second_hop/recv 应基本不变
                            {
                                __gm__ PlTimingNegLine *neg =
                                    reinterpret_cast<__gm__ PlTimingNegLine *>(base + kPlTimingNegOff);
                                PlDcci(reinterpret_cast<__gm__ uint8_t *>(neg));
                                if (neg->magic == kPlTimingNegMagic && neg->delay_finalize_cycles > 0u) {
                                    for (uint32_t di = 0; di < neg->delay_finalize_cycles; ++di) {
                                        AscendC::PipeBarrier<PIPE_ALL>();
                                    }
                                }
                            }
                            rt->finalize_done_cycle = AscendC::GetSystemCycle();
                            rt->full_done_cycle = rt->finalize_done_cycle;
                            if (dct != nullptr) {
                                dct->full_done_publish_cycle = rt->full_done_cycle;
                            }
                            PlDcci(reinterpret_cast<__gm__ uint8_t *>(fout));
                            PlDcci(reinterpret_cast<__gm__ uint8_t *>(rt));
                            timing->recv_done_bitmap = recv_mask;
                            timing->tokens_received = sum_recv;
                            PlWorkerPublishDestinationDone(base, desc, timing, overlap, dest_done_off,
                                                           d1_leader_completion_off, d1_publish_scratch_off,
                                                           run_epoch, static_cast<int>(leader_pe), sum_recv);
                            recv_path_done = true;
                        }
                        if (rd_err != 0u) {
                            // Publish failure evidence to PE0 before exiting. Without this,
                            // global transport sees a missing destination and turns a precise
                            // recv error into a timeout.
                            PlPublishTransportDone(base, desc->pair_id, run_epoch, expected_total, sum_recv, rd_err,
                                                   static_cast<int>(leader_pe));
                            fout->error_code = rd_err;
                            timing->error_code = rd_err;
                            PlDcci(reinterpret_cast<__gm__ uint8_t *>(fout));
                            trace->kernel_exit_reason = kPlKernelExitRecv;
                            PlTraceDcci(trace);
                            return;
                        }
                        }
                    } else if (use_done_lines) {
                        // 非 Full Dispatch：轮询 source_done
                        uint32_t recv_mask = 0;
                        for (uint32_t s = 0; s < desc->worker_count; ++s) {
                            __gm__ PlSourceDoneLine *sdone = reinterpret_cast<__gm__ PlSourceDoneLine *>(
                                base + source_done_off + static_cast<uint64_t>(s) * 64u);
                            PlDcciDoneLine(reinterpret_cast<__gm__ uint8_t *>(sdone));
                            if (sdone->done_epoch >= run_epoch && sdone->magic == kPlMagic) {
                                recv_mask |= (1u << s);
                            }
                        }
                        if (recv_mask == expect_recv) {
                            timing->recv_done_bitmap = recv_mask;
                            uint64_t total_recv = 0u;
                            for (uint32_t s = 0; s < desc->worker_count; ++s) {
                                total_recv +=
                                    static_cast<uint64_t>(PlEpochRecvTokensExpected(base, desc, s)) * run_epoch;
                            }
                            timing->tokens_received = total_recv;
                            PlWorkerPublishDestinationDone(base, desc, timing, overlap, dest_done_off,
                                                           d1_leader_completion_off, d1_publish_scratch_off,
                                                           run_epoch, static_cast<int>(leader_pe), total_recv);
                            completed = true;
                            break;
                        }
                    } else {
                        const uint64_t expect_tail =
                            static_cast<uint64_t>(PlEpochTokensExpected(base, desc, 0u)) * run_epoch;
                        const uint32_t expect_upload = (1u << desc->lane_count) - 1u;
                        uint32_t upload_mask = 0;
                        uint32_t recv_mask = 0;
                        uint64_t total_recv = 0;
                        for (uint32_t i = 0; i < desc->lane_count; ++i) {
                            __gm__ PlUploadCounters *uc =
                                reinterpret_cast<__gm__ PlUploadCounters *>(base + upload_ctr_off) + i;
                            PlDcciUploadCtr(uc);
                            const uint64_t lane_expect =
                                static_cast<uint64_t>(PlEpochTokensExpected(base, desc, i)) * run_epoch;
                            if (uc->local_tail >= lane_expect) {
                                upload_mask |= (1u << i);
                            }
                        }
                        for (uint32_t s = 0; s < desc->worker_count; ++s) {
                            __gm__ PlRecvCounters *rc =
                                reinterpret_cast<__gm__ PlRecvCounters *>(base + recv_ctr_off) + s;
                            PlDcciRecvCtr(rc);
                            const uint64_t expect_recv_n =
                                static_cast<uint64_t>(PlEpochRecvTokensExpected(base, desc, s)) * run_epoch;
                            if (rc->tokens_received >= expect_recv_n) {
                                recv_mask |= (1u << s);
                            }
                            total_recv += rc->tokens_received;
                        }
                        if (upload_mask == expect_upload && recv_mask == expect_recv) {
                            // Full Dispatch 完成面已由 RecvDoneLine 分支独占；此处仅非 FD
                            timing->upload_done_bitmap = upload_mask;
                            timing->recv_done_bitmap = recv_mask;
                            timing->tokens_received = total_recv;
                            uint32_t early_dest_mask = 0;
                            uint64_t early_dest_total = 0;
                            for (uint32_t s = 0; s < desc->worker_count; ++s) {
                                __gm__ PlRecvCounters *rc_early =
                                    reinterpret_cast<__gm__ PlRecvCounters *>(base + recv_ctr_off) + s;
                                PlDcciRecvCtr(rc_early);
                                if (rc_early->destination_head_observed_before_final_forward_count > 0) {
                                    early_dest_mask |= (1u << s);
                                    early_dest_total +=
                                        rc_early->destination_head_observed_before_final_forward_count;
                                }
                            }
                            overlap->dest_early_consume_mask = early_dest_mask;
                            overlap->destination_early_consume_total = early_dest_total;
                            PlDcci(reinterpret_cast<__gm__ uint8_t *>(overlap));
                            PlWorkerPublishDestinationDone(base, desc, timing, overlap, dest_done_off,
                                                           d1_leader_completion_off, d1_publish_scratch_off,
                                                           run_epoch, static_cast<int>(leader_pe), total_recv);
                            completed = true;
                            break;
                        }
                    }
                    // Semantic completion must never wait for optional
                    // second-hop telemetry.  In particular, session stop or
                    // epoch N+1 cannot be held hostage by a late/missing
                    // timing summary after all RecvDone lines are complete.
                    // telemetry_valid below records whether the non-blocking
                    // sample happened to converge.
                    if (recv_path_done) {
                        completed = true;
                        break;
                    }
                    ++spins;
                }
                if (!completed) {
                    timing->error_code = kPlKernelExitDestAgg;
                    trace->kernel_exit_reason = kPlKernelExitDestAgg;
                    PlTraceDcci(trace);
                    return;
                }
                /*
                 * RecvDone may become complete in the same control-loop
                 * iteration in which the final INC visible-cell telemetry
                 * arrives.  The loop then exits before its next non-blocking
                 * telemetry poll, producing an intermittent missing formal
                 * timestamp despite a fully correct operation.  Perform one
                 * final non-blocking sample after semantic completion.  This
                 * never delays the next epoch and never turns optional
                 * telemetry into a correctness dependency.
                 */
                if (full_dispatch &&
                    probe_mode == kPlFullDispatchProbeFull &&
                    aclshmem_my_pe() == static_cast<int>(leader_pe) &&
                    !second_hop_timing_ok) {
                    __gm__ PlInvocationDesc *inv =
                        PlFullDispatchInvocation(base, fdc, run_epoch);
                    if (inv != nullptr) {
                        second_hop_timing_ok =
                            PlPe0TryPublishGlobalSecondHopVisibleTiming(
                                base, desc->worker_count, run_epoch,
                                inv->generation, control_bi);
                    }
                }
                {
                    uint64_t retire_gen = 0u;
                    if (full_dispatch) {
                        __gm__ PlInvocationDesc *inv_ret = PlFullDispatchInvocation(base, fdc, run_epoch);
                        if (inv_ret != nullptr) {
                            retire_gen = inv_ret->generation;
                        }
                    }
                    const uint32_t telem_valid = second_hop_timing_ok ? 1u : 0u;
                    PlWorkerPublishControlEpochRetire(base, run_epoch, retire_gen, timing->error_code,
                                                     telem_valid);
                }
            }

            last_epoch = run_epoch;
            timing->epoch_end_cycle = AscendC::GetSystemCycle();
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(timing));
            PlDcci(reinterpret_cast<__gm__ uint8_t *>(timing) + 64u);
        }
        trace->kernel_exit_reason = kPlKernelExitStop;
        trace->service_exit_epoch = last_epoch;
        PlTraceDcci(trace);
    }
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void inc_dc_dn_pipeline_inc_persistent_kernel(
    GM_ADDR sym, uint64_t pl_desc_off, uint64_t forward_ctr_off, uint64_t timing_off, uint64_t overlap_off,
    uint64_t inc_trace_off, uint64_t d1_start_off, uint64_t d1_stop_off)
{
    if ASCEND_IS_AIV {
        const int bi = AscendC::GetBlockIdx();
        __gm__ uint8_t *base = reinterpret_cast<__gm__ uint8_t *>(sym);
        __gm__ PlPipelineDesc *desc_probe = reinterpret_cast<__gm__ PlPipelineDesc *>(base + pl_desc_off);
        const bool p5_sm = (desc_probe->magic == kPlMagic) && PlIsSourceMajorRaw(desc_probe);
        const int block_limit = p5_sm ? static_cast<int>(kPlIncP5ServiceBlockDim)
                                     : static_cast<int>(kPlIncServiceBlockDim);
        if (bi < 0 || bi >= block_limit) {
            return;
        }
        __gm__ PlPipelineDesc *desc = desc_probe;
        __gm__ PlPipelineTiming *timing = reinterpret_cast<__gm__ PlPipelineTiming *>(base + timing_off);
        __gm__ PlOverlapTelemetry *overlap = reinterpret_cast<__gm__ PlOverlapTelemetry *>(base + overlap_off);
        __gm__ D1StartLine *start = reinterpret_cast<__gm__ D1StartLine *>(base + d1_start_off);
        __gm__ D1SessionStopLine *stop = reinterpret_cast<__gm__ D1SessionStopLine *>(base + d1_stop_off);
        __gm__ PlServiceTraceLine *trace = nullptr;
        if (p5_sm && bi >= 10) {
            trace = reinterpret_cast<__gm__ PlServiceTraceLine *>(base + DevPlIncP5ExtTraceLineOff(static_cast<uint32_t>(bi)));
        } else {
            trace = reinterpret_cast<__gm__ PlServiceTraceLine *>(base + inc_trace_off) + bi;
        }

        trace->block_id = static_cast<uint32_t>(bi);
        trace->kernel_enter_cycle = AscendC::GetSystemCycle();
        trace->desc_valid = (desc->magic == kPlMagic && desc->my_role == 1u) ? 1u : 0u;
        PlTraceDcci(trace);

        __gm__ PlResidentLine *resident = nullptr;
        if (p5_sm && bi >= 10) {
            resident = reinterpret_cast<__gm__ PlResidentLine *>(base + DevPlIncP5ExtResidentLineOff(static_cast<uint32_t>(bi)));
        } else {
            resident =
                reinterpret_cast<__gm__ PlResidentLine *>(base + kPlIncResidentOff + static_cast<uint64_t>(bi) * 64u);
        }
        resident->value = 1;
        resident->block_id = static_cast<uint32_t>(bi);
        resident->magic = kPlMagic;
        PlDcci(reinterpret_cast<__gm__ uint8_t *>(resident));

        if (trace->desc_valid == 0u) {
            trace->kernel_exit_reason = kPlKernelExitResource;
            PlTraceDcci(trace);
            return;
        }

        uint64_t last_epoch = 0;
        while (true) {
            if (PlPollStop(stop, trace)) {
                break;
            }
            const uint64_t run_epoch = PlWaitStartPersistent(start, stop, last_epoch, trace);
            if (run_epoch == 0) {
                if (PlPollStop(stop, trace)) {
                    break;
                }
                continue;
            }

            if ((desc->pipeline_mode & kPlModeBaseMask) == kPlModeControlOnly) {
                last_epoch = run_epoch;
                continue;
            }

            if (desc->measurement_mode == kPlMeasurementFullDispatch) {
                __gm__ PlFullDispatchConfig *fdc_err =
                    reinterpret_cast<__gm__ PlFullDispatchConfig *>(base + kPlFullDispatchConfigOff);
                PlDcci(reinterpret_cast<__gm__ uint8_t *>(fdc_err));
                __gm__ PlFullOutputDoneLine *fout_err =
                    reinterpret_cast<__gm__ PlFullOutputDoneLine *>(base + fdc_err->full_output_done_off);
                PlDcci(reinterpret_cast<__gm__ uint8_t *>(fout_err));
                if (fout_err->error_code == kPlErrGenerationMismatch ||
                    fout_err->error_code == kPlErrWorkspaceDescMissing ||
                    fout_err->error_code == kPlErrWorkspaceGenerationMismatch ||
                    fout_err->error_code == kPlErrWorkspacePointerAlignment ||
                    fout_err->error_code == kPlErrOutputCapacityInsufficient ||
                    fout_err->error_code == kPlErrWorkspaceFlagInvalid) {
                    timing->error_code = fout_err->error_code;
                    trace->kernel_exit_reason = kPlKernelExitForward;
                    PlTraceDcci(trace);
                    last_epoch = run_epoch;
                    continue;
                }
            }

            // ---------- P5 source-major AIV map ----------
            if (p5_sm) {
                __gm__ PlFullDispatchConfig *fdc =
                    reinterpret_cast<__gm__ PlFullDispatchConfig *>(base + kPlFullDispatchConfigOff);
                PlDcci(reinterpret_cast<__gm__ uint8_t *>(fdc));
                const uint32_t probe = fdc->probe_mode;
                if (bi >= static_cast<int>(kPlIncP5GatherBlockBegin) &&
                    bi <= static_cast<int>(kPlIncP5GatherBlockEnd)) {
                    const uint32_t dest_lane = static_cast<uint32_t>(bi);
                    __gm__ PlForwardCounters *fc =
                        reinterpret_cast<__gm__ PlForwardCounters *>(base + forward_ctr_off) + dest_lane;
                    if (probe == kPlFullDispatchProbeFirstHop) {
                        PlIncSourceMajorIngressVerifyEpoch(base, desc, fdc, fc, dest_lane, run_epoch);
                    } else if (probe == kPlFullDispatchProbeFull) {
                        PlIncSourceMajorGatherProducerEpoch(base, desc, fdc, fc, dest_lane, run_epoch);
                    }
                    if (fc->error_code != 0) {
                        timing->error_code = fc->error_code;
                        trace->kernel_exit_reason = kPlKernelExitForward;
                        PlTraceDcci(trace);
                        return;
                    }
                } else if (bi >= static_cast<int>(kPlIncP5EgressBlockBegin) &&
                           bi <= static_cast<int>(kPlIncP5EgressBlockEnd)) {
                    if (probe == kPlFullDispatchProbeFull) {
                        const uint32_t dest_lane = static_cast<uint32_t>(bi - kPlIncP5EgressBlockBegin);
                        __gm__ PlForwardCounters *fc =
                            reinterpret_cast<__gm__ PlForwardCounters *>(base + forward_ctr_off) + dest_lane;
                        PlIncSourceMajorEgressSenderEpoch(base, desc, fc, dest_lane, run_epoch);
                    }
                } else if (bi == static_cast<int>(kPlIncP5IngressReclaimBlock)) {
                    // block16：只做 raw reclaim；不得串 IncForwardCounts（否则与 gather/upload 成环）
                    if (probe == kPlFullDispatchProbeFull || probe == kPlFullDispatchProbeGatherOnly) {
                        PlIncSourceMajorReclaimControlEpoch(base, desc, run_epoch);
                    }
                } else if (bi == static_cast<int>(kPlIncP5CountPrefixBlock)) {
                    // block17：IncForwardCounts 独立于 reclaim；另记 INC timing start
                    uint32_t gen = 0u;
                    __gm__ PlInvocationDesc *inv_start = PlFullDispatchInvocation(base, fdc, run_epoch);
                    if (inv_start != nullptr) {
                        gen = static_cast<uint32_t>(inv_start->generation);
                    }
                    PlDevRankTimingMarkStart(base, kPlRoleMaskInc, run_epoch, gen);
                    if (desc->worker_count > 1u) {
                        const uint32_t fwd_st = PlFullDispatchIncForwardCounts(base, desc, fdc, run_epoch);
                        if (fwd_st != kPlCountFwdOk) {
                            timing->error_code = fwd_st;
                            trace->kernel_exit_reason = kPlKernelExitForward;
                            PlTraceDcci(trace);
                            return;
                        }
                    }
                } else if (bi == static_cast<int>(kPlIncP5SecondHopAggBlock)) {
                    // first_hop：completion 在 IngressVerify 末尾发布，避免与 gather 竞态
                    if (probe != kPlFullDispatchProbeFirstHop) {
                        // E5b：等所有 active egress done_epoch==epoch 再发 INC rank completion
                        const uint32_t active_n =
                            (desc->worker_count == 0u) ? 1u : desc->worker_count;
                        constexpr uint64_t kEgressDoneWaitUs = 8000000ull;
                        const uint64_t ed0 = AscendC::GetSystemCycle();
                        bool all_egress = false;
                        while (!PlWaitBudgetExceeded(ed0, kEgressDoneWaitUs)) {
                            all_egress = true;
                            for (uint32_t d = 0u; d < active_n && d < kQv2LaneCount; ++d) {
                                __gm__ PlEgressDoneLine *dl = reinterpret_cast<__gm__ PlEgressDoneLine *>(
                                    base + DevPlIncP5EgressDoneOff(d));
                                PlDcci(reinterpret_cast<__gm__ uint8_t *>(dl));
                                if (dl->magic != kPlEgressDoneMagic || dl->done_epoch != run_epoch) {
                                    all_egress = false;
                                    break;
                                }
                            }
                            if (all_egress) {
                                break;
                            }
                        }
                        if (!all_egress) {
                            timing->error_code = kPlErrEpochLocalEgressWait;
                            trace->kernel_exit_reason = kPlKernelExitForward;
                            PlTraceDcci(trace);
                            return;
                        }
                        PlDevRankTimingMarkForwardDone(base, AscendC::GetSystemCycle());
                        PlDevRankTimingPublishLocalCompletion(base, kPlRoleMaskInc);
                    }
                }
                last_epoch = run_epoch;
                timing->epoch_end_cycle = AscendC::GetSystemCycle();
                PlDcci(reinterpret_cast<__gm__ uint8_t *>(timing));
                continue;
            }

            // ---------- legacy dest-major path (blocks 0–9) ----------
            if (bi >= static_cast<int>(kPlIncForwardBlockBegin) && bi <= static_cast<int>(kPlIncForwardBlockEnd)) {
                const uint32_t lane_id = static_cast<uint32_t>(bi);
                if (lane_id < desc->lane_count) {
                    __gm__ PlForwardCounters *fc =
                        reinterpret_cast<__gm__ PlForwardCounters *>(base + forward_ctr_off) + lane_id;
                    const bool full_dispatch = (desc->measurement_mode == kPlMeasurementFullDispatch);
                    if (full_dispatch) {
                        __gm__ PlFullDispatchConfig *fdc =
                            reinterpret_cast<__gm__ PlFullDispatchConfig *>(base + kPlFullDispatchConfigOff);
                        PlDcci(reinterpret_cast<__gm__ uint8_t *>(fdc));
                        if (fdc->probe_mode == kPlFullDispatchProbeFirstHop) {
                            PlIncIngressOnlyLaneEpoch(base, desc, fc, trace, lane_id, run_epoch);
                        } else if (fdc->probe_mode == kPlFullDispatchProbeFull) {
                            PlIncForwardLaneEpoch(base, desc, fc, timing, overlap, trace, lane_id, run_epoch);
                        }
                    } else {
                        PlIncForwardLaneEpoch(base, desc, fc, timing, overlap, trace, lane_id, run_epoch);
                    }
                    if (fc->error_code != 0) {
                        timing->error_code = fc->error_code;
                        trace->kernel_exit_reason = kPlKernelExitForward;
                        PlTraceDcci(trace);
                        return;
                    }
                }
            } else if (bi == static_cast<int>(kPlIncControlBlock)) {
                {
                    uint32_t gen = 0u;
                    const bool full_dispatch_entry = (desc->measurement_mode == kPlMeasurementFullDispatch);
                    if (full_dispatch_entry) {
                        __gm__ PlFullDispatchConfig *fdc_entry =
                            reinterpret_cast<__gm__ PlFullDispatchConfig *>(base + kPlFullDispatchConfigOff);
                        PlDcci(reinterpret_cast<__gm__ uint8_t *>(fdc_entry));
                        __gm__ PlInvocationDesc *inv_start = PlFullDispatchInvocation(base, fdc_entry, run_epoch);
                        if (inv_start != nullptr) {
                            gen = static_cast<uint32_t>(inv_start->generation);
                        }
                    }
                    PlDevRankTimingMarkStart(base, kPlRoleMaskInc, run_epoch, gen);
                }
                const bool full_dispatch = (desc->measurement_mode == kPlMeasurementFullDispatch);
                uint32_t probe_mode = kPlFullDispatchProbeFull;
                if (full_dispatch) {
                    __gm__ PlFullDispatchConfig *fdc =
                        reinterpret_cast<__gm__ PlFullDispatchConfig *>(base + kPlFullDispatchConfigOff);
                    PlDcci(reinterpret_cast<__gm__ uint8_t *>(fdc));
                    probe_mode = fdc->probe_mode;
                    // 多 worker：任何 probe 都先严格 push 转发 count（CountOnly/GatherOnly 也依赖）
                    if (desc->worker_count > 1u) {
                        const uint32_t fwd_st = PlFullDispatchIncForwardCounts(base, desc, fdc, run_epoch);
                        if (fwd_st != kPlCountFwdOk) {
                            // 禁止失败后静默 last_epoch=run_epoch 进入下一 epoch
                            timing->error_code = fwd_st;
                            trace->kernel_exit_reason = kPlKernelExitForward;
                            PlTraceDcci(trace);
                            return;
                        }
                    }
                }
                // count/gather probe：无数据面 forward，count 转发后即可结束本 epoch
                if (full_dispatch &&
                    (probe_mode == kPlFullDispatchProbeCountOnly ||
                     probe_mode == kPlFullDispatchProbeGatherOnly)) {
                    last_epoch = run_epoch;
                    continue;
                }
                // full_dispatch 多 worker：count 已由 IncForwardCounts 完成；payload forward 由 lane AIV
                // 独立推进。control 再等 forward_done 会在大 token 下误超时并拖死后续 epoch（S3）。
                if (full_dispatch && desc->worker_count > 1u &&
                    probe_mode == kPlFullDispatchProbeFull) {
                    // multi-worker：lane AIV 独立推进；control 以本 epoch 退出点为 forward 上界
                    // （严格 lane 齐套可后续改为扫 PlForwardDoneLine）
                    PlDevRankTimingMarkForwardDone(base, AscendC::GetSystemCycle());
                    PlDevRankTimingPublishLocalCompletion(base, kPlRoleMaskInc);
                    last_epoch = run_epoch;
                    continue;
                }
                const uint32_t expect_fwd_mask = (1u << desc->lane_count) - 1u;
                const bool use_done_lines = (desc->completion_mode == kPlCompletionModeDoneLines);
                // full_dispatch 大 token：放宽 forward_done 等待，优先避免误杀 control
                const uint32_t fwd_spins_max =
                    (full_dispatch && desc->worker_count > 1u) ? 80000000u : 8000000u;
                uint32_t spins = 0;
                bool completed = false;
                while (spins < fwd_spins_max) {
                    if (PlPollStop(stop, trace)) {
                        completed = true;
                        break;
                    }
                    // workspace / generation 等致命错误：fail-closed 退出（与 worker control 对齐）
                    if (full_dispatch) {
                        __gm__ PlFullDispatchConfig *fdc_err =
                            reinterpret_cast<__gm__ PlFullDispatchConfig *>(base + kPlFullDispatchConfigOff);
                        PlDcci(reinterpret_cast<__gm__ uint8_t *>(fdc_err));
                        __gm__ PlFullOutputDoneLine *fout_err =
                            reinterpret_cast<__gm__ PlFullOutputDoneLine *>(base + fdc_err->full_output_done_off);
                        PlDcci(reinterpret_cast<__gm__ uint8_t *>(fout_err));
                        if (fout_err->error_code == kPlErrGenerationMismatch ||
                            fout_err->error_code == kPlErrWorkspaceDescMissing ||
                            fout_err->error_code == kPlErrWorkspaceGenerationMismatch ||
                            fout_err->error_code == kPlErrWorkspacePointerAlignment ||
                            fout_err->error_code == kPlErrOutputCapacityInsufficient ||
                            fout_err->error_code == kPlErrWorkspaceFlagInvalid) {
                            timing->error_code = fout_err->error_code;
                            trace->kernel_exit_reason = kPlKernelExitForward;
                            PlTraceDcci(trace);
                            return;
                        }
                    }
                    if (use_done_lines) {
                        uint32_t fwd_mask = 0;
                        for (uint32_t i = 0; i < desc->lane_count; ++i) {
                            __gm__ PlForwardDoneLine *fdone = reinterpret_cast<__gm__ PlForwardDoneLine *>(
                                base + DevPlForwardDoneLineOff(i));
                            PlDcciDoneLine(reinterpret_cast<__gm__ uint8_t *>(fdone));
                            if (fdone->done_epoch >= run_epoch && fdone->magic == kPlMagic) {
                                fwd_mask |= (1u << i);
                            }
                        }
                        if (fwd_mask == expect_fwd_mask) {
                            timing->forward_done_bitmap = fwd_mask;
                            const uint64_t total_fwd =
                                static_cast<uint64_t>(desc->tokens_per_lane) * run_epoch * desc->lane_count;
                            timing->tokens_forwarded = total_fwd;
                            timing->tokens_sent = total_fwd;
                            uint64_t min_fwd = UINT64_MAX;
                            uint64_t max_fwd_done = 0;
                            for (uint32_t i = 0; i < desc->lane_count; ++i) {
                                __gm__ PlAivEpochTimingLine *fl =
                                    reinterpret_cast<__gm__ PlAivEpochTimingLine *>(base + DevPlForwardAivTimingOff(i));
                                PlDcciDoneLine(reinterpret_cast<__gm__ uint8_t *>(fl));
                                if (fl->done_epoch >= run_epoch && fl->first_issue_cycle > 0) {
                                    if (fl->first_issue_cycle < min_fwd) {
                                        min_fwd = fl->first_issue_cycle;
                                    }
                                    if (fl->done_cycle > max_fwd_done) {
                                        max_fwd_done = fl->done_cycle;
                                    }
                                }
                            }
                            if (min_fwd != UINT64_MAX) {
                                timing->first_forward_issue = min_fwd;
                            }
                            if (max_fwd_done > 0) {
                                timing->last_forward_issue = max_fwd_done;
                                PlDevRankTimingMarkForwardDone(base, max_fwd_done);
                            }
                            PlDevRankTimingPublishLocalCompletion(base, kPlRoleMaskInc);
                            uint32_t early_mask = 0;
                            uint64_t early_total = 0;
                            for (uint32_t i = 0; i < desc->lane_count; ++i) {
                                __gm__ PlForwardCounters *fc =
                                    reinterpret_cast<__gm__ PlForwardCounters *>(base + forward_ctr_off) + i;
                                PlDcciForwardCtr(fc);
                                if (fc->forward_before_final_ingress_count > 0) {
                                    early_mask |= (1u << i);
                                }
                                early_total += fc->forward_before_final_ingress_count;
                            }
                            overlap->inc_early_forward_mask = early_mask;
                            overlap->forward_before_final_ingress_total = early_total;
                            // P5.4：同 domain cycle overlap（禁止 counter 代理）
                            {
                                const uint64_t u0 = timing->first_upload_issue;
                                const uint64_t u1 = timing->last_upload_issue;
                                const uint64_t f0 = timing->first_forward_issue;
                                const uint64_t f1 = timing->last_forward_issue;
                                const uint64_t d0 = timing->first_destination_consume;
                                const uint64_t d1 = timing->destination_done_cycle;
                                const uint32_t wi =
                                    (u0 > 0u && f0 > 0u && u1 > 0u && f1 > 0u && u0 < f1 && f0 < u1) ? 1u : 0u;
                                const uint32_t id =
                                    (f0 > 0u && d0 > 0u && f1 > 0u && d1 > 0u && f0 < d1 && d0 < f1) ? 1u : 0u;
                                overlap->worker_inc_overlap = wi;
                                overlap->inc_dest_overlap = id;
                                overlap->real_pipeline_overlap = (wi != 0u && id != 0u) ? 1u : 0u;
                                overlap->overlap_fail_reason =
                                    (wi == 0u ? 1u : 0u) | (id == 0u ? 2u : 0u);
                            }
                            PlDcci(reinterpret_cast<__gm__ uint8_t *>(overlap));
                            completed = true;
                            break;
                        }
                    } else {
                        const uint64_t expect_fwd = static_cast<uint64_t>(desc->tokens_per_lane) * run_epoch;
                        uint32_t fwd_mask = 0;
                        uint64_t total_fwd = 0;
                        uint32_t early_mask = 0;
                        for (uint32_t i = 0; i < desc->lane_count; ++i) {
                            __gm__ PlForwardCounters *fc =
                                reinterpret_cast<__gm__ PlForwardCounters *>(base + forward_ctr_off) + i;
                            PlDcciForwardCtr(fc);
                            if (fc->egress_forwarded >= expect_fwd) {
                                fwd_mask |= (1u << i);
                            }
                            total_fwd += fc->egress_forwarded;
                            if (fc->forward_before_final_ingress_count > 0) {
                                early_mask |= (1u << i);
                            }
                        }
                        if (fwd_mask == expect_fwd_mask) {
                            timing->forward_done_bitmap = fwd_mask;
                            timing->tokens_forwarded = total_fwd;
                            timing->tokens_sent = total_fwd;
                            overlap->inc_early_forward_mask = early_mask;
                            uint64_t early_total = 0;
                            for (uint32_t i = 0; i < desc->lane_count; ++i) {
                                __gm__ PlForwardCounters *fc =
                                    reinterpret_cast<__gm__ PlForwardCounters *>(base + forward_ctr_off) + i;
                                early_total += fc->forward_before_final_ingress_count;
                            }
                            overlap->forward_before_final_ingress_total = early_total;
                            {
                                const uint64_t u0 = timing->first_upload_issue;
                                const uint64_t u1 = timing->last_upload_issue;
                                const uint64_t f0 = timing->first_forward_issue;
                                const uint64_t f1 = timing->last_forward_issue;
                                const uint64_t d0 = timing->first_destination_consume;
                                const uint64_t d1 = timing->destination_done_cycle;
                                const uint32_t wi =
                                    (u0 > 0u && f0 > 0u && u1 > 0u && f1 > 0u && u0 < f1 && f0 < u1) ? 1u : 0u;
                                const uint32_t id =
                                    (f0 > 0u && d0 > 0u && f1 > 0u && d1 > 0u && f0 < d1 && d0 < f1) ? 1u : 0u;
                                overlap->worker_inc_overlap = wi;
                                overlap->inc_dest_overlap = id;
                                overlap->real_pipeline_overlap = (wi != 0u && id != 0u) ? 1u : 0u;
                                overlap->overlap_fail_reason =
                                    (wi == 0u ? 1u : 0u) | (id == 0u ? 2u : 0u);
                            }
                            PlDcci(reinterpret_cast<__gm__ uint8_t *>(overlap));
                            completed = true;
                            break;
                        }
                    }
                    ++spins;
                }
                if (!completed) {
                    // full_dispatch 多 epoch：禁止因 payload forward_done 超时杀掉 control AIV，
                    // 否则下一 epoch 不再执行 IncForwardCounts（S3：仅 pe8 存活 → fout_error=10）
                    timing->error_code = kPlKernelExitForward;
                    trace->kernel_exit_reason = kPlKernelExitForward;
                    PlTraceDcci(trace);
                    if (!(full_dispatch && desc->worker_count > 1u)) {
                        return;
                    }
                }
            } else if (bi == static_cast<int>(kPlIncSecondHopAggBlock)) {
                // block9：专用 second_hop 聚合；不等待 destination recv / full_done
                const bool full_dispatch = (desc->measurement_mode == kPlMeasurementFullDispatch);
                if (full_dispatch) {
                    __gm__ PlFullDispatchConfig *fdc =
                        reinterpret_cast<__gm__ PlFullDispatchConfig *>(base + kPlFullDispatchConfigOff);
                    PlDcci(reinterpret_cast<__gm__ uint8_t *>(fdc));
                    if (fdc->probe_mode == kPlFullDispatchProbeFull) {
                        __gm__ PlInvocationDesc *inv = PlFullDispatchInvocation(base, fdc, run_epoch);
                        if (!PlIncSecondHopAggPublishSummary(base, desc, stop, trace, run_epoch, inv->generation,
                                                             static_cast<uint32_t>(kPlIncSecondHopAggBlock),
                                                             static_cast<int>(kD1LeaderPe))) {
                            timing->error_code = 71u;
                            trace->kernel_exit_reason = kPlKernelExitSecondHopAgg;
                            PlTraceDcci(trace);
                            return;
                        }
                    }
                }
            }

            last_epoch = run_epoch;
        }
        trace->kernel_exit_reason = kPlKernelExitStop;
        trace->service_exit_epoch = last_epoch;
        PlTraceDcci(trace);
    }
}

extern "C" void launch_inc_dc_dn_pipeline_worker_persistent_kernel(
    uint8_t *sym, uint64_t pl_desc_off, uint64_t upload_ctr_off, uint64_t recv_ctr_off, uint64_t timing_off,
    uint64_t overlap_off, uint64_t worker_trace_off, uint64_t source_done_off, uint64_t dest_done_off,
    uint64_t d1_start_off, uint64_t d1_stop_off, uint64_t d1_leader_completion_off, uint64_t d1_publish_scratch_off,
    uint32_t leader_pe, int block_dim, void *stream)
{
    inc_dc_dn_pipeline_worker_persistent_kernel<<<block_dim, nullptr, stream>>>(
        sym, pl_desc_off, upload_ctr_off, recv_ctr_off, timing_off, overlap_off, worker_trace_off, source_done_off,
        dest_done_off, d1_start_off, d1_stop_off, d1_leader_completion_off, d1_publish_scratch_off, leader_pe);
}

extern "C" void launch_inc_dc_dn_pipeline_inc_persistent_kernel(uint8_t *sym, uint64_t pl_desc_off,
                                                               uint64_t forward_ctr_off, uint64_t timing_off,
                                                               uint64_t overlap_off, uint64_t inc_trace_off,
                                                               uint64_t d1_start_off, uint64_t d1_stop_off,
                                                               int block_dim, void *stream)
{
    inc_dc_dn_pipeline_inc_persistent_kernel<<<block_dim, nullptr, stream>>>(
        sym, pl_desc_off, forward_ctr_off, timing_off, overlap_off, inc_trace_off, d1_start_off, d1_stop_off);
}
