#include "inc_dc_single_inc_inline_plan.h"

#include <limits>

namespace inc::dc::single_inline {
namespace {

bool Add(uint64_t a, uint64_t b, uint64_t *out)
{
    return InlineCheckedAdd(a, b, out);
}

bool Mul(uint64_t a, uint64_t b, uint64_t *out)
{
    return InlineCheckedMul(a, b, out);
}

bool Align(uint64_t value, uint64_t *out)
{
    return InlineCheckedAlignUp(value, kSingleInlineWorkspaceAlignment, out);
}

bool Allocate(uint64_t bytes, uint64_t *cursor, uint64_t *offset)
{
    uint64_t aligned = 0u;
    uint64_t end = 0u;
    if (cursor == nullptr || offset == nullptr || !Align(*cursor, &aligned) ||
        !Add(aligned, bytes, &end))
        return false;
    *offset = aligned;
    *cursor = end;
    return true;
}

bool CountBytes(uint64_t a, uint64_t b, uint64_t element_bytes,
                uint64_t *bytes)
{
    uint64_t count = 0u;
    return Mul(a, b, &count) && Mul(count, element_bytes, bytes);
}

uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return value / divisor + (value % divisor != 0u ? 1u : 0u);
}

} // namespace

SingleInlinePlanStatus BuildSingleInlinePlan(
    const SingleInlinePlanDesc &desc, SingleInlinePlan *plan)
{
    if (plan == nullptr || desc.worker_count == 0u ||
        desc.expert_count == 0u || desc.hidden_bytes == 0u ||
        desc.max_topk == 0u || desc.max_topk > desc.expert_count ||
        desc.tokens_per_wave == 0u || desc.batch_tokens == 0u ||
        desc.batch_tokens > desc.tokens_per_wave ||
        desc.payload_tile_rows == 0u ||
        desc.payload_tile_rows > desc.tokens_per_wave ||
        desc.slot_count < 2u || desc.dispatch_lane_count == 0u ||
        desc.combine_lane_count == 0u ||
        desc.dispatch_lane_count > kIncDcMaxDispatchLanes ||
        desc.combine_lane_count > kIncDcMaxCombineOwners ||
        desc.session_id == 0u || desc.placement_epoch == 0u) {
        return SingleInlinePlanStatus::INVALID_ARGUMENT;
    }

    SingleInlinePlan built{};
    auto &runtime = built.runtime;
    runtime.worker_count = desc.worker_count;
    runtime.inc_pe = desc.worker_count;
    runtime.expert_count = desc.expert_count;
    runtime.hidden_bytes = desc.hidden_bytes;
    runtime.max_topk = desc.max_topk;
    runtime.tokens_per_wave = desc.tokens_per_wave;
    runtime.batch_tokens = desc.batch_tokens;
    runtime.payload_tile_rows = desc.payload_tile_rows;
    runtime.slot_count = desc.slot_count;
    runtime.batches_per_wave =
        CeilDiv(desc.tokens_per_wave, desc.batch_tokens);
    runtime.payload_tiles_per_wave =
        CeilDiv(desc.tokens_per_wave, desc.payload_tile_rows);
    runtime.dispatch_lane_count = desc.dispatch_lane_count;
    runtime.combine_lane_count = desc.combine_lane_count;
    runtime.session_id = desc.session_id;
    runtime.placement_epoch = desc.placement_epoch;

    uint64_t token_payload_offset = 0u;
    if (!InlineTokenRecordLayout(
            desc.max_topk, desc.hidden_bytes, &token_payload_offset,
            &built.max_token_record_bytes) ||
        !InlineCheckedAlignUp(
            built.max_token_record_bytes, kInlineRouteWireAlignment,
            &built.stored_token_record_stride) ||
        token_payload_offset % kInlineRouteWireAlignment != 0u) {
        return SingleInlinePlanStatus::CAPACITY_EXCEEDED;
    }
    uint64_t offset_table_bytes = 0u;
    uint64_t records_offset = 0u;
    uint64_t stored_records_bytes = 0u;
    if (!InlineBatchPreambleLayout(
            desc.batch_tokens, &offset_table_bytes, &records_offset) ||
        !Mul(desc.batch_tokens, built.stored_token_record_stride,
             &stored_records_bytes) ||
        !Add(records_offset, stored_records_bytes,
             &built.max_batch_frame_bytes) ||
        !Align(built.max_batch_frame_bytes,
               &runtime.ingress_frame_stride) ||
        offset_table_bytes !=
            (static_cast<uint64_t>(desc.batch_tokens) + 1u) *
                sizeof(uint64_t)) {
        return SingleInlinePlanStatus::CAPACITY_EXCEEDED;
    }

    if (!Mul(desc.worker_count, desc.tokens_per_wave,
             &built.results_per_slot) ||
        !Mul(built.results_per_slot, desc.max_topk,
             &built.contributions_per_slot)) {
        return SingleInlinePlanStatus::CAPACITY_EXCEEDED;
    }
    built.fanout_tokens_per_worker_slot = built.results_per_slot;
    built.fanout_assignments_per_worker_slot =
        built.contributions_per_slot;

    uint64_t fanout_payload_offset = 0u;
    uint64_t fanout_record_bytes = 0u;
    uint64_t payload_record_bytes = 0u;
    if (!InlineFanoutRecordLayout(
            desc.max_topk, desc.hidden_bytes, &fanout_payload_offset,
            &fanout_record_bytes) ||
        !InlineCheckedAlignUp(
            fanout_record_bytes, kInlineRouteWireAlignment,
            &runtime.fanout_record_stride) ||
        !InlinePayloadRecordLayout(desc.hidden_bytes, &payload_record_bytes) ||
        !InlineCheckedAlignUp(
            payload_record_bytes, kInlineRouteWireAlignment,
            &runtime.combine_record_stride) ||
        fanout_payload_offset % kInlineRouteWireAlignment != 0u) {
        return SingleInlinePlanStatus::CAPACITY_EXCEEDED;
    }
    runtime.result_record_stride = runtime.combine_record_stride;

    uint64_t cursor = sizeof(SingleInlineRuntimeDesc);
    uint64_t bytes = 0u;
    if (!CountBytes(desc.slot_count, 1u,
                    sizeof(SingleInlineGenerationControl), &bytes) ||
        !Allocate(bytes, &cursor, &runtime.generation_control_off) ||
        !CountBytes(desc.expert_count, 1u, sizeof(uint32_t), &bytes) ||
        !Allocate(bytes, &cursor, &runtime.expert_owner_off) ||
        !Allocate(bytes, &cursor, &runtime.expert_local_index_off) ||
        !CountBytes(desc.worker_count, 1u, sizeof(uint32_t), &bytes) ||
        !Allocate(bytes, &cursor, &runtime.worker_endpoint_off)) {
        return SingleInlinePlanStatus::CAPACITY_EXCEEDED;
    }

    uint64_t slot_worker_batches = 0u;
    if (!Mul(desc.slot_count, desc.worker_count, &slot_worker_batches) ||
        !Mul(slot_worker_batches, runtime.batches_per_wave,
             &slot_worker_batches) ||
        !Mul(slot_worker_batches, runtime.ingress_frame_stride, &bytes) ||
        !Allocate(bytes, &cursor, &runtime.ingress_frame_off)) {
        return SingleInlinePlanStatus::CAPACITY_EXCEEDED;
    }

    if (!Mul(desc.slot_count, built.results_per_slot, &bytes) ||
        !Mul(bytes, sizeof(SingleInlineResultState), &bytes) ||
        !Allocate(bytes, &cursor, &runtime.journal_result_off) ||
        !Mul(desc.slot_count, built.contributions_per_slot, &bytes) ||
        !Mul(bytes, sizeof(SingleInlineJournalEntry), &bytes) ||
        !Allocate(bytes, &cursor, &runtime.journal_entry_off)) {
        return SingleInlinePlanStatus::CAPACITY_EXCEEDED;
    }

    if (!CountBytes(desc.slot_count, desc.expert_count, sizeof(uint64_t),
                    &bytes) ||
        !Allocate(bytes, &cursor, &runtime.expert_row_counter_off) ||
        !Mul(desc.slot_count, built.contributions_per_slot, &bytes) ||
        !Mul(bytes, runtime.fanout_record_stride, &bytes) ||
        !Allocate(bytes, &cursor, &runtime.fanout_record_off) ||
        !CountBytes(desc.slot_count, desc.worker_count,
                    kSingleInlineCacheLine, &bytes) ||
        !Allocate(bytes, &cursor, &runtime.fanout_count_off)) {
        return SingleInlinePlanStatus::CAPACITY_EXCEEDED;
    }

    if (!Mul(desc.slot_count, built.contributions_per_slot, &bytes) ||
        !Mul(bytes, runtime.combine_record_stride, &bytes) ||
        !Allocate(bytes, &cursor, &runtime.combine_record_off)) {
        return SingleInlinePlanStatus::CAPACITY_EXCEEDED;
    }

    if (!Mul(desc.slot_count, built.results_per_slot, &bytes) ||
        !Mul(bytes, desc.hidden_bytes, &bytes) ||
        !Allocate(bytes, &cursor, &runtime.accumulator_off) ||
        !Mul(desc.slot_count, built.results_per_slot, &bytes) ||
        !Mul(bytes, runtime.result_record_stride, &bytes) ||
        !Allocate(bytes, &cursor, &runtime.result_record_off)) {
        return SingleInlinePlanStatus::CAPACITY_EXCEEDED;
    }

    if (!CountBytes(desc.slot_count,
                    desc.dispatch_lane_count + desc.combine_lane_count,
                    kSingleInlineCacheLine, &bytes) ||
        !Allocate(bytes, &cursor, &runtime.stats_off) ||
        !Align(cursor, &runtime.total_bytes)) {
        return SingleInlinePlanStatus::CAPACITY_EXCEEDED;
    }

    *plan = built;
    return SingleInlinePlanStatus::OK;
}

const char *SingleInlinePlanStatusString(SingleInlinePlanStatus status)
{
    switch (status) {
        case SingleInlinePlanStatus::OK: return "OK";
        case SingleInlinePlanStatus::INVALID_ARGUMENT:
            return "INVALID_ARGUMENT";
        case SingleInlinePlanStatus::CAPACITY_EXCEEDED:
            return "CAPACITY_EXCEEDED";
    }
    return "UNKNOWN";
}

} // namespace inc::dc::single_inline
