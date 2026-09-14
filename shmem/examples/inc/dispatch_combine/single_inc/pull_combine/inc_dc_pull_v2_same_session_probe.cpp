// Qualification-only same-session D+C probe.
//
// The production kernels remain defined in their original translation units.
// For the host smoke path, include the existing E2E harness implementations
// under separate namespaces so their private oracle builders are reused
// without copying or weakening either correctness model.  The NPU execution
// context will be added on this composition boundary: one aclshmem init per
// PE, two streams, and the existing device launch functions.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "shmem.h"
#include "utils.h"

#include "inc_dc_pull_combine_v2.h"
#include "inc_dc_pull_dispatch_v2.h"
#include "inc_dc_pull_v2_test_start_gate.h"

namespace pull_v2_dispatch_embed {
#define main inc_dc_pull_dispatch_v2_embedded_main_unused
#include "inc_dc_pull_dispatch_v2_device_e2e.cpp"
#undef main
} // namespace pull_v2_dispatch_embed

namespace pull_v2_combine_embed {
#define main inc_dc_pull_combine_v2_embedded_main_unused
#include "inc_dc_pull_combine_v2_npu_e2e.cpp"
#undef main
} // namespace pull_v2_combine_embed

namespace {

constexpr uint32_t kWorkers = 4u;
constexpr uint64_t kPayloadBytesPerWorker = 128ull << 20u;
constexpr uint32_t kHidden = 8192u;
constexpr uint32_t kExperts = 64u;
constexpr uint32_t kDispatchChannels = 3u;

namespace D = pull_v2_dispatch_embed;
namespace C = pull_v2_combine_embed;
namespace P = inc::dc::pull_v2;

D::Options DispatchOptions(int pe, int first_npu)
{
    D::Options o{};
    o.workers = kWorkers;
    o.pe = pe;
    o.first_npu = first_npu;
    o.payload_bytes = kPayloadBytesPerWorker;
    o.workload = D::Workload::SYM_K2_BALANCED;
    o.workload_name = "sym_k2_balanced";
    o.hidden = kHidden;
    o.expert_count = kExperts;
    o.channels = kDispatchChannels;
    o.measure = 1u;
    o.seed = 20260904u;
    return o;
}

C::Options CombineOptions(int pe, int first_npu)
{
    C::Options o{};
    o.workers = kWorkers;
    o.pe = pe;
    o.first_npu = first_npu;
    o.hidden = kHidden;
    const uint64_t row_bytes = static_cast<uint64_t>(kHidden) * sizeof(float);
    const uint64_t rows_per_b = kPayloadBytesPerWorker / row_bytes;
    o.rows = static_cast<uint32_t>(rows_per_b * kWorkers / 2u);
    o.workload = C::Workload::SYM_K2_BALANCED;
    o.workload_name = "sym_k2_balanced";
    o.measure = 1u;
    return o;
}

struct DispatchContext {
    D::Options o{};
    D::WaveOracle sizing{};
    D::WaveOracle wave{};
    std::vector<uint8_t> local_slot;
    P::Ready local_ready{};
    uint64_t source_stride = 0u;
    uint64_t metadata_inbox_stride = sizeof(P::SlotHeader);
    uint64_t row_capacity = 1u;
    uint64_t assignment_capacity = 1u;
    uint64_t journal_token_capacity = 1u;
    uint64_t journal_contributor_capacity = 1u;
    uint64_t journal_assignment_capacity = 0u;
    uint64_t row_map_entries = 0u;
    uint64_t prefix_entries = 0u;
    uint64_t expert_entries = 0u;
    uint64_t hidden_slot_stride = 0u;
    uint64_t rows_slot_stride = 0u;
    uint64_t assignments_slot_stride = 0u;
    uint64_t expert_slot_stride = 0u;
    uint64_t parser_scratch_entries = 0u;
    uint32_t dispatch_blocks = 0u;
    D::GuardedBuffer source_region, ready_mailbox, metadata_inbox, source_acks;
    D::GuardedBuffer destination_hidden, destination_rows;
    D::GuardedBuffer destination_assignments, destination_expert_counts;
    D::GuardedBuffer destination_completions;
    D::GuardedBuffer inc_slots, inc_destination_rows;
    D::GuardedBuffer inc_destination_assignments, journal_header;
    D::GuardedBuffer journal_tokens, journal_contributors;
    D::GuardedBuffer journal_assignments, row_map, source_token_prefix;
    D::GuardedBuffer source_destination_prefix, destination_row_counts;
    D::GuardedBuffer destination_assignment_counts, expert_counts;
    D::GuardedBuffer parser_scratch, status_line;
    std::vector<D::GuardedBuffer *> buffers;

    bool Plan(int pe, int first_npu, uint32_t half_aiv)
    {
        dispatch_blocks = half_aiv;
        o = DispatchOptions(pe, first_npu);
        std::vector<uint8_t> unused_slot;
        P::Ready unused_ready{};
        if (!D::BuildOracle(o, D::kFirstGeneration, D::kFirstSequence,
                            D::kFirstWave, 0u, -1, &sizing, &unused_slot,
                            &unused_ready, &source_stride))
            return false;
        for (uint32_t destination = 0u; destination < o.workers;
             ++destination) {
            row_capacity = std::max<uint64_t>(row_capacity,
                sizing.layout.destination_rows[destination].size());
            assignment_capacity = std::max<uint64_t>(assignment_capacity,
                sizing.layout.expert_assignments[destination].size());
        }
        journal_token_capacity = std::max<uint64_t>(
            sizing.layout.journal_tokens.size(), 1u);
        journal_contributor_capacity = std::max<uint64_t>(
            sizing.layout.contributors.size(), 1u);
        prefix_entries = static_cast<uint64_t>(o.workers + 1u) * o.workers;
        expert_entries = static_cast<uint64_t>(o.workers) * o.expert_count;
        for (const P::ParsedSource &source : sizing.sources)
            metadata_inbox_stride = std::max<uint64_t>(
                metadata_inbox_stride, source.header.hidden_offset);
        metadata_inbox_stride = D::Align64(metadata_inbox_stride);
        const uint64_t row_bytes = static_cast<uint64_t>(o.hidden) * 2u;
        hidden_slot_stride = D::Align64(row_capacity * row_bytes);
        rows_slot_stride = D::Align64(
            row_capacity * sizeof(P::DestinationRow));
        assignments_slot_stride = D::Align64(
            assignment_capacity * sizeof(P::ExpertAssignment));
        expert_slot_stride = D::Align64(
            static_cast<uint64_t>(o.expert_count) * sizeof(uint32_t));
        buffers = {&source_region, &ready_mailbox, &metadata_inbox,
            &source_acks,
            &destination_hidden, &destination_rows, &destination_assignments,
            &destination_expert_counts, &destination_completions, &inc_slots,
            &inc_destination_rows, &inc_destination_assignments,
            &journal_header, &journal_tokens, &journal_contributors,
            &journal_assignments, &row_map, &source_token_prefix,
            &source_destination_prefix, &destination_row_counts,
            &destination_assignment_counts, &expert_counts, &parser_scratch,
            &status_line};
        return half_aiv != 0u &&
            static_cast<uint64_t>(o.workers) * (o.channels + 1u) <= half_aiv &&
            D::ParserScratchEntries(half_aiv, o.workers, o.expert_count,
                                    &parser_scratch_entries);
    }

    bool AllocateAll()
    {
        auto alloc = [](D::GuardedBuffer *buffer, uint64_t bytes,
                        bool symmetric, const char *name) {
            return D::Allocate(buffer, bytes, symmetric, name);
        };
        return
            alloc(&source_region, source_stride * D::kRingSlots, true,
                  "ss_d_source_region") &&
            alloc(&ready_mailbox, o.workers * sizeof(P::Ready), true,
                  "ss_d_ready") &&
            alloc(&metadata_inbox,
                  static_cast<uint64_t>(o.workers) * metadata_inbox_stride,
                  true, "ss_d_metadata_inbox") &&
            alloc(&source_acks, o.workers * sizeof(P::SourceConsumed), true,
                  "ss_d_acks") &&
            alloc(&destination_hidden,
                  hidden_slot_stride * D::kRingSlots, true,
                  "ss_d_hidden") &&
            alloc(&destination_rows, rows_slot_stride * D::kRingSlots, true,
                  "ss_d_rows") &&
            alloc(&destination_assignments,
                  assignments_slot_stride * D::kRingSlots, true,
                  "ss_d_assignments") &&
            alloc(&destination_expert_counts,
                  expert_slot_stride * D::kRingSlots, true,
                  "ss_d_expert_counts") &&
            alloc(&destination_completions,
                  static_cast<uint64_t>(D::kRingSlots) * o.workers *
                      sizeof(P::DestinationCompletion), true,
                  "ss_d_completions") &&
            alloc(&inc_slots,
                  static_cast<uint64_t>(o.workers) * source_stride, false,
                  "ss_d_inc_slots") &&
            alloc(&inc_destination_rows,
                  static_cast<uint64_t>(o.workers) * rows_slot_stride, false,
                  "ss_d_inc_rows") &&
            alloc(&inc_destination_assignments,
                  static_cast<uint64_t>(o.workers) *
                      assignments_slot_stride, false, "ss_d_inc_assignments") &&
            alloc(&journal_header,
                  D::kRingSlots * sizeof(P::JournalSlotHeader), false,
                  "ss_d_journal_header") &&
            alloc(&journal_tokens,
                  journal_token_capacity * sizeof(P::JournalTokenEntry),
                  false, "ss_d_journal_tokens") &&
            alloc(&journal_contributors,
                  journal_contributor_capacity *
                      sizeof(P::JournalContributor), false,
                  "ss_d_journal_contributors") &&
            alloc(&journal_assignments,
                  journal_assignment_capacity * sizeof(P::AssignmentRecord),
                  false, "ss_d_journal_assignments") &&
            alloc(&row_map, row_map_entries * sizeof(uint32_t), false,
                  "ss_d_row_map") &&
            alloc(&source_token_prefix,
                  static_cast<uint64_t>(o.workers + 1u) * sizeof(uint32_t),
                  false, "ss_d_token_prefix") &&
            alloc(&source_destination_prefix,
                  prefix_entries * sizeof(uint32_t), false,
                  "ss_d_destination_prefix") &&
            alloc(&destination_row_counts,
                  o.workers * sizeof(uint32_t), false, "ss_d_row_counts") &&
            alloc(&destination_assignment_counts,
                  o.workers * sizeof(uint32_t), false,
                  "ss_d_assignment_counts") &&
            alloc(&expert_counts, expert_entries * sizeof(uint32_t), false,
                  "ss_d_expert_counts_private") &&
            alloc(&parser_scratch,
                  parser_scratch_entries * sizeof(uint32_t), false,
                  "ss_d_parser_scratch") &&
            alloc(&status_line, sizeof(D::PullTimeline), false,
                  "ss_d_timeline");
    }

    bool Prepare(int pe, int inc_pe, aclrtStream stream)
    {
        uint64_t wave_stride = 0u;
        if (!D::BuildOracle(o, D::kFirstGeneration, D::kFirstSequence,
                            D::kFirstWave, 0u, pe < inc_pe ? pe : -1,
                            &wave, &local_slot, &local_ready, &wave_stride) ||
            wave_stride != source_stride)
            return false;
        int status = 0;
        if (pe < inc_pe) {
            status = aclrtMemcpy(source_region.data, source_stride,
                local_slot.data(), source_stride, ACL_MEMCPY_HOST_TO_DEVICE);
            if (status == 0)
                status = aclrtMemcpy(ready_mailbox.data +
                        static_cast<uint64_t>(pe) * sizeof(P::Ready),
                    sizeof(P::Ready), &local_ready, sizeof(P::Ready),
                    ACL_MEMCPY_HOST_TO_DEVICE);
        }
        if (status == 0)
            status = D::Fill(&source_acks, 0u) &&
                D::Fill(&destination_hidden, D::kPoison) &&
                D::Fill(&destination_rows, D::kPoison) &&
                D::Fill(&destination_assignments, D::kPoison) &&
                D::Fill(&destination_expert_counts, D::kPoison) &&
                D::Fill(&destination_completions, 0u) ? 0 : 1;
        if (status == 0 && pe == inc_pe) {
            status = D::Fill(&metadata_inbox, 0u) &&
                D::Fill(&inc_slots, 0u) &&
                D::Fill(&inc_destination_rows, D::kPoison) &&
                D::Fill(&inc_destination_assignments, D::kPoison) &&
                D::Fill(&journal_tokens, D::kPoison) &&
                D::Fill(&journal_contributors, 0u) &&
                D::Fill(&journal_assignments, D::kPoison) &&
                D::Fill(&row_map, D::kPoison) &&
                D::Fill(&source_token_prefix, 0u) &&
                D::Fill(&source_destination_prefix, D::kPoison) &&
                D::Fill(&destination_row_counts, 0u) &&
                D::Fill(&destination_assignment_counts, 0u) &&
                D::Fill(&expert_counts, 0u) &&
                D::Fill(&parser_scratch, 0u) &&
                D::Fill(&status_line, 0u) ? 0 : 1;
            P::JournalSlotHeader initial{};
            if (status == 0)
                status = aclrtMemcpy(journal_header.data, sizeof(initial),
                    &initial, sizeof(initial), ACL_MEMCPY_HOST_TO_DEVICE);
        }
        return status == 0 && aclrtSynchronizeStream(stream) == ACL_SUCCESS;
    }

    void Launch(uint32_t aiv, aclrtStream stream, int inc_pe)
    {
        D::launch_inc_dc_pull_dispatch_v2_device(
            aiv, stream, source_region.data, ready_mailbox.data,
            metadata_inbox.data, inc_slots.data, source_acks.data,
            destination_hidden.data,
            destination_rows.data, destination_assignments.data,
            destination_expert_counts.data, destination_completions.data,
            inc_destination_rows.data, inc_destination_assignments.data,
            journal_header.data, journal_tokens.data,
            journal_contributors.data, journal_assignments.data, row_map.data,
            source_token_prefix.data, source_destination_prefix.data,
            destination_row_counts.data, destination_assignment_counts.data,
            expert_counts.data, parser_scratch.data, status_line.data,
            shmemx_get_ffts_config(), D::kSessionId, D::kPlacementEpoch,
            D::kFirstGeneration, D::kFirstSequence, source_stride,
            metadata_inbox_stride,
            hidden_slot_stride, rows_slot_stride, assignments_slot_stride,
            expert_slot_stride, journal_token_capacity,
            journal_contributor_capacity, journal_assignment_capacity,
            row_capacity, assignment_capacity, rows_slot_stride,
            assignments_slot_stride, row_map_entries, prefix_entries,
            expert_entries, parser_scratch_entries, o.workers,
            o.expert_count, o.hidden, static_cast<uint32_t>(P::DataType::BF16),
            inc_pe, D::kRegionId, D::kFirstWave, 0u, D::kRingSlots,
            o.channels, D::kSpinCap);
    }

    bool Validate(int pe, int inc_pe, D::PullTimeline *timeline)
    {
        bool ok = pe < inc_pe
            ? D::ValidateWorker(o, wave, 0u, hidden_slot_stride,
                rows_slot_stride, assignments_slot_stride,
                expert_slot_stride, destination_hidden, destination_rows,
                destination_assignments, destination_expert_counts,
                destination_completions, source_acks)
            : D::ValidateInc(o, wave, 0u, inc_destination_rows,
                inc_destination_assignments, journal_header, journal_tokens,
                journal_contributors, journal_assignments, row_map,
                source_token_prefix, source_destination_prefix,
                destination_row_counts, destination_assignment_counts,
                expert_counts, status_line, parser_scratch, dispatch_blocks,
                timeline, rows_slot_stride, assignments_slot_stride);
        for (D::GuardedBuffer *buffer : buffers)
            ok = D::GuardsValid(*buffer) && ok;
        return ok;
    }

    void ReleaseAll()
    {
        for (auto it = buffers.rbegin(); it != buffers.rend(); ++it)
            D::Release(*it);
    }
};

struct CombineContext {
    C::Options o{};
    C::Wave sizing{};
    C::Wave wave{};
    uint64_t pull_capacity = 1u;
    uint64_t accumulator_capacity = 1u;
    uint64_t source_capacity = 1u;
    C::GuardedBuffer partials, ready, notices, registrations;
    C::GuardedBuffer acks, output, completions, source_offsets, pulls;
    C::GuardedBuffer owner_offsets, results, journal, pull_next, heads;
    C::GuardedBuffer counts, result_index, ready_state, payload_offsets;
    C::GuardedBuffer timeline_buffer;
    std::vector<C::GuardedBuffer *> buffers;

    bool Plan(int pe, int first_npu)
    {
        o = CombineOptions(pe, first_npu);
        if (!C::BuildWave(o, C::kFirstGeneration, C::kFirstSequence,
                          C::kFirstWave, 0u, &sizing))
            return false;
        pull_capacity = std::max<uint64_t>(sizing.plan.pulls.size(), 1u);
        accumulator_capacity = std::max<uint64_t>(
            sizing.plan.accumulator_count, 1u);
        source_capacity = std::max<uint64_t>(o.workers, 1u);
        buffers = {&partials, &ready, &notices, &registrations, &acks,
            &output, &completions, &source_offsets,
            &pulls, &owner_offsets, &results, &journal, &pull_next, &heads,
            &counts, &result_index, &ready_state, &payload_offsets,
            &timeline_buffer};
        return true;
    }

    bool AllocateAll()
    {
        auto alloc = [](C::GuardedBuffer *buffer, uint64_t bytes,
                        bool symmetric, const char *name) {
            return C::Allocate(buffer, bytes, symmetric, name);
        };
        return
            alloc(&partials, sizing.partial_slot_stride * C::kRingSlots,
                  true, "ss_c_partials") &&
            alloc(&ready, static_cast<uint64_t>(C::kRingSlots) * o.workers *
                  sizeof(P::CombineReadyV2), true, "ss_c_ready") &&
            alloc(&notices,
                  static_cast<uint64_t>(C::kRingSlots) * o.workers *
                      sizeof(P::CombineReadyNoticeV2), true,
                  "ss_c_notices") &&
            alloc(&registrations, static_cast<uint64_t>(o.workers) *
                  sizeof(P::CombineRegionRegistration), true,
                  "ss_c_registrations") &&
            alloc(&acks, static_cast<uint64_t>(C::kRingSlots) * o.workers *
                  sizeof(C::CombineSourceAckV2), true, "ss_c_acks") &&
            alloc(&output, sizing.output_slot_stride * C::kRingSlots, true,
                  "ss_c_output") &&
            alloc(&completions,
                  static_cast<uint64_t>(C::kRingSlots) * o.workers *
                      sizeof(C::CombineOwnerCompletionV2), true,
                  "ss_c_completions") &&
            alloc(&source_offsets,
                  static_cast<uint64_t>(o.workers + 1u) * sizeof(uint64_t),
                  false, "ss_c_source_offsets") &&
            alloc(&pulls, pull_capacity * sizeof(P::CombinePullOp), false,
                  "ss_c_pulls") &&
            alloc(&owner_offsets,
                  static_cast<uint64_t>(o.workers + 1u) * sizeof(uint64_t),
                  false, "ss_c_owner_offsets") &&
            alloc(&results, accumulator_capacity * sizeof(P::CombineResultOp),
                  false, "ss_c_results") &&
            alloc(&journal, sizeof(P::JournalSlotHeader), false,
                  "ss_c_journal") &&
            alloc(&pull_next, pull_capacity * sizeof(uint32_t), false,
                  "ss_c_pull_next") &&
            alloc(&heads, accumulator_capacity * sizeof(uint32_t), false,
                  "ss_c_heads") &&
            alloc(&counts, accumulator_capacity * sizeof(uint32_t), false,
                  "ss_c_counts") &&
            alloc(&result_index, accumulator_capacity * sizeof(uint32_t),
                  false, "ss_c_result_index") &&
            alloc(&ready_state,
                  source_capacity * C::kSourceScratchStride, false,
                  "ss_c_ready_state") &&
            alloc(&payload_offsets,
                  source_capacity * C::kSourceScratchStride, false,
                  "ss_c_payload_offsets") &&
            alloc(&timeline_buffer, sizeof(C::CombineDeviceTimelineV2), false,
                  "ss_c_timeline");
    }

    bool Prepare(int pe, int inc_pe, aclrtStream stream)
    {
        if (!C::BuildWave(o, C::kFirstGeneration, C::kFirstSequence,
                          C::kFirstWave, 0u, &wave) ||
            wave.partial_slot_stride != sizing.partial_slot_stride ||
            wave.output_slot_stride != sizing.output_slot_stride)
            return false;
        int status = 0;
        if (pe < inc_pe) {
            const uint32_t source = static_cast<uint32_t>(pe);
            const uint32_t rows = wave.plan.source_row_counts[source];
            std::vector<float> host(static_cast<size_t>(rows) * o.hidden);
            for (uint32_t row = 0u; row < rows; ++row)
                for (uint32_t element = 0u; element < o.hidden; ++element)
                    host[static_cast<size_t>(row) * o.hidden + element] =
                        C::PartialValue(source, row, element);
            if (!host.empty())
                status = aclrtMemcpy(partials.data,
                    host.size() * sizeof(float), host.data(),
                    host.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
            if (status == 0)
                status = aclrtMemcpy(ready.data +
                        static_cast<uint64_t>(source) *
                            sizeof(P::CombineReadyV2),
                    sizeof(P::CombineReadyV2), &wave.ready[source],
                    sizeof(P::CombineReadyV2), ACL_MEMCPY_HOST_TO_DEVICE);
            if (status == 0)
                status = aclrtMemcpy(notices.data +
                        static_cast<uint64_t>(source) *
                            sizeof(P::CombineReadyNoticeV2),
                    sizeof(P::CombineReadyNoticeV2), &wave.notices[source],
                    sizeof(P::CombineReadyNoticeV2),
                    ACL_MEMCPY_HOST_TO_DEVICE);
        }
        if (status == 0)
            status = C::CopyToDevice(registrations.data, wave.registrations)
                ? 0 : 1;
        if (status == 0)
            status = C::Fill(&acks, 0u) && C::Fill(&output, C::kPoison) &&
                C::Fill(&completions, 0u) ? 0 : 1;
        if (status == 0 && pe == inc_pe) {
            status = C::CopyToDevice(source_offsets.data,
                         wave.plan.source_offsets) &&
                C::CopyToDevice(pulls.data, wave.plan.pulls) &&
                C::CopyToDevice(owner_offsets.data,
                                wave.plan.owner_offsets) &&
                C::CopyToDevice(results.data, wave.plan.results) &&
                C::CopyToDevice(pull_next.data, wave.plan.pull_next) &&
                C::CopyToDevice(heads.data, wave.plan.accumulator_heads) &&
                C::CopyToDevice(counts.data,
                    wave.plan.accumulator_contributor_counts) &&
                C::CopyToDevice(result_index.data,
                    wave.plan.accumulator_result_index) &&
                C::Fill(&ready_state, 0u) &&
                C::Fill(&payload_offsets, 0u) &&
                C::Fill(&timeline_buffer, 0u) ? 0 : 1;
            if (status == 0)
                status = aclrtMemcpy(journal.data,
                    sizeof(P::JournalSlotHeader), &wave.layout.journal_header,
                    sizeof(P::JournalSlotHeader), ACL_MEMCPY_HOST_TO_DEVICE);
        }
        return status == 0 && aclrtSynchronizeStream(stream) == ACL_SUCCESS;
    }

    void Launch(uint32_t aiv, aclrtStream stream, int inc_pe)
    {
        C::launch_inc_dc_pull_combine_v2_device(
            aiv, stream, partials.data, ready.data, notices.data,
            registrations.data, acks.data, output.data,
            completions.data, source_offsets.data, pulls.data,
            owner_offsets.data, results.data, journal.data, pull_next.data,
            heads.data, counts.data, result_index.data, ready_state.data,
            payload_offsets.data, timeline_buffer.data,
            shmemx_get_ffts_config(), C::kSessionId, C::kPlacementEpoch,
            C::kFirstGeneration, C::kFirstSequence,
            wave.plan.dispatch_cookie, sizing.output_slot_stride,
            wave.plan.pulls.size(), wave.plan.results.size(), pull_capacity,
            accumulator_capacity, source_capacity,
            wave.plan.accumulator_count, o.workers, o.hidden,
            static_cast<uint32_t>(P::PartialDataType::FP32), inc_pe,
            C::kFirstWave, 0u, C::kRingSlots, C::kSpinCap);
    }

    bool Validate(int pe, int inc_pe, C::CombineDeviceTimelineV2 *timeline)
    {
        bool ok = pe < inc_pe
            ? C::ValidateWorker(o, wave, 0u, acks, output, completions)
            : C::ValidateInc(wave, journal, ready, notices,
                ready_state, timeline_buffer, timeline);
        for (C::GuardedBuffer *buffer : buffers)
            ok = C::GuardsValid(*buffer) && ok;
        return ok;
    }

    void ReleaseAll()
    {
        for (auto it = buffers.rbegin(); it != buffers.rend(); ++it)
            C::Release(*it);
    }
};

bool BuildMatchedPlans(
    pull_v2_dispatch_embed::WaveOracle *dispatch,
    pull_v2_combine_embed::Wave *combine)
{
    pull_v2_dispatch_embed::Options d{};
    d.workers = kWorkers;
    d.payload_bytes = kPayloadBytesPerWorker;
    d.workload = pull_v2_dispatch_embed::Workload::SYM_K2_BALANCED;
    d.workload_name = "sym_k2_balanced";
    d.hidden = kHidden;
    d.expert_count = kExperts;
    d.channels = kDispatchChannels;
    d.measure = 1u;
    d.seed = 20260904u;
    std::vector<uint8_t> unused_slot;
    inc::dc::pull_v2::Ready unused_ready{};
    uint64_t source_stride = 0u;
    if (!pull_v2_dispatch_embed::BuildOracle(
            d, pull_v2_dispatch_embed::kFirstGeneration,
            pull_v2_dispatch_embed::kFirstSequence,
            pull_v2_dispatch_embed::kFirstWave, 0u, -1, dispatch,
            &unused_slot, &unused_ready, &source_stride))
        return false;

    pull_v2_combine_embed::Options c{};
    c.workers = kWorkers;
    c.hidden = kHidden;
    const uint64_t row_bytes = static_cast<uint64_t>(kHidden) * sizeof(float);
    const uint64_t rows_per_b = kPayloadBytesPerWorker / row_bytes;
    c.rows = static_cast<uint32_t>(rows_per_b * kWorkers / 2u);
    c.workload = pull_v2_combine_embed::Workload::SYM_K2_BALANCED;
    c.workload_name = "sym_k2_balanced";
    c.measure = 1u;
    if (!pull_v2_combine_embed::BuildWave(
            c, pull_v2_combine_embed::kFirstGeneration,
            pull_v2_combine_embed::kFirstSequence,
            pull_v2_combine_embed::kFirstWave, 0u, combine))
        return false;

    return dispatch->ingress_hidden_bytes == 512ull << 20u &&
        dispatch->egress_hidden_bytes == 1ull << 30u &&
        dispatch->logical_bytes == (1536ull << 20u) &&
        combine->ingress_bytes == 512ull << 20u &&
        combine->egress_bytes == 256ull << 20u &&
        pull_v2_dispatch_embed::kSessionId !=
            pull_v2_combine_embed::kSessionId;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc == 2 && std::strcmp(argv[1], "--host-smoke") == 0) {
        D::WaveOracle dispatch{};
        C::Wave combine{};
        const bool ok = BuildMatchedPlans(&dispatch, &combine);
        std::cout << "{\"test\":\"pull_v2_same_session_host_smoke\""
                  << ",\"workers\":" << kWorkers
                  << ",\"dispatch_logical_bytes\":"
                  << dispatch.logical_bytes
                  << ",\"combine_ingress_bytes\":"
                  << combine.ingress_bytes
                  << ",\"combine_egress_bytes\":" << combine.egress_bytes
                  << ",\"strong_identities_disjoint\":"
                  << (D::kSessionId != C::kSessionId ? "true" : "false")
                  << ",\"correct\":" << (ok ? "true" : "false")
                  << "}\n";
        return ok ? 0 : 1;
    }
    if (argc != 6 || std::strcmp(argv[1], "--npu") != 0 ||
        (std::strcmp(argv[2], "both") != 0 &&
         std::strcmp(argv[2], "dispatch") != 0 &&
         std::strcmp(argv[2], "combine") != 0)) {
        std::cerr << "usage: " << argv[0] << " --host-smoke\n"
                  << "   or: " << argv[0]
                  << " --npu <both|dispatch|combine> <pe> <ipport>"
                     " <first_npu>\n";
        return 2;
    }

    const bool run_dispatch = std::strcmp(argv[2], "combine") != 0;
    const bool run_combine = std::strcmp(argv[2], "dispatch") != 0;
    const int pe = std::atoi(argv[3]);
    const char *endpoint = argv[4];
    const int first_npu = std::atoi(argv[5]);
    const int inc_pe = static_cast<int>(kWorkers);
    const int pes = static_cast<int>(kWorkers + 1u);
    if (pe < 0 || pe >= pes || first_npu < 0) return 2;

    int status = aclInit(nullptr);
    const int device = first_npu + pe;
    if (status == 0) status = aclrtSetDevice(device);
    int64_t live_aiv = 0;
    if (status == 0)
        status = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                    &live_aiv);
    const uint32_t half_aiv = live_aiv <= 0
        ? 0u : static_cast<uint32_t>(live_aiv) / 2u;
    DispatchContext dispatch;
    CombineContext combine;
    if (status == 0 &&
        (!dispatch.Plan(pe, first_npu, half_aiv) ||
         !combine.Plan(pe, first_npu)))
        status = 2;

    aclrtStream dispatch_stream = nullptr;
    aclrtStream combine_stream = nullptr;
    if (status == 0) status = aclrtCreateStream(&dispatch_stream);
    if (status == 0) status = aclrtCreateStream(&combine_stream);
    bool shmem_initialized = false;
    if (status == 0) {
        aclshmemx_uniqueid_t uid{};
        aclshmemx_init_attr_t attr;
        test_set_attr(pe, pes, 2ull * 1024ull * 1024ull * 1024ull,
                      endpoint, uid, &attr);
        status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
        shmem_initialized = status == 0;
    }
    // Every PE allocates the same complete D-then-C symmetric layout once.
    if (status == 0 && (!dispatch.AllocateAll() || !combine.AllocateAll()))
        status = 1;
    if (status == 0 &&
        (!dispatch.Prepare(pe, inc_pe, dispatch_stream) ||
         !combine.Prepare(pe, inc_pe, combine_stream)))
        status = 1;
    if (status == 0) aclshmem_barrier_all();

    const auto host_begin = std::chrono::steady_clock::now();
    if (status == 0 && run_dispatch)
        dispatch.Launch(half_aiv, dispatch_stream, inc_pe);
    if (status == 0 && run_combine)
        combine.Launch(half_aiv, combine_stream, inc_pe);
    if (status == 0 && run_dispatch)
        status = aclrtSynchronizeStream(dispatch_stream);
    if (status == 0 && run_combine)
        status = aclrtSynchronizeStream(combine_stream);
    const auto host_end = std::chrono::steady_clock::now();
    if (status == 0) aclshmem_barrier_all();

    D::PullTimeline dispatch_timeline{};
    C::CombineDeviceTimelineV2 combine_timeline{};
    const bool dispatch_correct = !run_dispatch ||
        (status == 0 && dispatch.Validate(pe, inc_pe, &dispatch_timeline));
    const bool combine_correct = !run_combine ||
        (status == 0 && combine.Validate(pe, inc_pe, &combine_timeline));
    const bool correct = status == 0 && dispatch_correct && combine_correct;
    if (status == 0) aclshmem_barrier_all();

    if (pe == inc_pe) {
        const double host_us =
            std::chrono::duration<double, std::micro>(
                host_end - host_begin).count();
        std::cout << std::setprecision(12)
                  << "{\"test\":\"pull_v2_same_session_probe\""
                  << ",\"mode\":\"" << argv[2] << "\""
                  << ",\"workers\":" << kWorkers
                  << ",\"active_aiv_per_operator\":" << half_aiv
                  << ",\"host_makespan_us\":" << host_us
                  << ",\"dispatch_logical_bytes\":"
                  << dispatch.wave.logical_bytes
                  << ",\"combine_logical_bytes\":"
                  << combine.wave.ingress_bytes + combine.wave.egress_bytes
                  << ",\"dispatch_cycle_start\":"
                  << dispatch_timeline.kernel_start
                  << ",\"dispatch_cycle_metadata_done\":"
                  << dispatch_timeline.metadata_parse_done
                  << ",\"dispatch_cycle_fanout_begin\":"
                  << dispatch_timeline.fanout_put_begin
                  << ",\"dispatch_cycle_fanout_done\":"
                  << dispatch_timeline.fanout_put_done
                  << ",\"dispatch_cycle_done\":"
                  << dispatch_timeline.kernel_done
                  << ",\"combine_cycle_start\":"
                  << combine_timeline.kernel_start
                  << ",\"combine_cycle_plan_done\":"
                  << combine_timeline.plan_index_done
                  << ",\"combine_cycle_done\":"
                  << combine_timeline.kernel_done
                  << ",\"correct\":" << (correct ? "true" : "false")
                  << "}\n";
    }

    // Free in exact reverse global allocation order: C then D.
    if (shmem_initialized) {
        combine.ReleaseAll();
        dispatch.ReleaseAll();
        aclshmem_finalize();
    }
    if (combine_stream != nullptr) aclrtDestroyStream(combine_stream);
    if (dispatch_stream != nullptr) aclrtDestroyStream(dispatch_stream);
    if (device >= 0) aclrtResetDevice(device);
    aclFinalize();
    return correct ? 0 : 1;
}
