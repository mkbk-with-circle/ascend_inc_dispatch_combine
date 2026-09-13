#include "inc_dc_inline_route_journal.h"

#include <cstddef>
#include <limits>
#include <mutex>
#include <new>

namespace inc::dc {
namespace {

uint64_t Mix64(uint64_t value)
{
    value ^= value >> 30u;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27u;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31u;
    return value;
}

bool AddBytes(uint64_t count, uint64_t element_bytes, uint64_t *total)
{
    if (total == nullptr ||
        (count != 0u &&
         element_bytes > std::numeric_limits<uint64_t>::max() / count)) {
        return false;
    }
    const uint64_t bytes = count * element_bytes;
    if (bytes > std::numeric_limits<uint64_t>::max() - *total) return false;
    *total += bytes;
    return true;
}

} // namespace

IncDcInlineRouteJournal::IncDcInlineRouteJournal(
    const IncDcInlineRouteJournalConfig &config)
    : config_(config)
{
}

IncDcInlineJournalStatus IncDcInlineRouteJournal::Create(
    const IncDcInlineRouteJournalConfig &config,
    std::unique_ptr<IncDcInlineRouteJournal> *journal)
{
    if (journal == nullptr || *journal != nullptr) {
        return IncDcInlineJournalStatus::INVALID_ARGUMENT;
    }
    if (config.worker_count == 0u || config.expert_count == 0u ||
        config.generation_slot_count == 0u ||
        config.max_results_per_slot == 0u ||
        config.max_contributions_per_slot == 0u ||
        config.session_id == 0u || config.placement_epoch == 0u ||
        config.session_cookie == 0u || config.lookup_placement == nullptr) {
        return IncDcInlineJournalStatus::INVALID_CONFIG;
    }
    uint64_t bytes = 0u;
    if (!AddBytes(config.generation_slot_count, sizeof(GenerationSlot),
                  &bytes) ||
        !AddBytes(config.generation_slot_count,
                  static_cast<uint64_t>(config.max_results_per_slot) *
                      sizeof(ResultState),
                  &bytes) ||
        !AddBytes(config.generation_slot_count,
                  static_cast<uint64_t>(config.max_contributions_per_slot) *
                      sizeof(ContributionState),
                  &bytes) ||
        bytes > std::numeric_limits<size_t>::max()) {
        return IncDcInlineJournalStatus::INVALID_CONFIG;
    }
    try {
        auto created = std::unique_ptr<IncDcInlineRouteJournal>(
            new IncDcInlineRouteJournal(config));
        created->slots_.resize(config.generation_slot_count);
        *journal = std::move(created);
    } catch (const std::bad_alloc &) {
        return IncDcInlineJournalStatus::NO_MEMORY;
    } catch (...) {
        return IncDcInlineJournalStatus::INVALID_CONFIG;
    }
    return IncDcInlineJournalStatus::OK;
}

uint64_t IncDcInlineRouteJournal::HandleAuthenticator(
    uint32_t slot_index, uint64_t generation, uint64_t request_id,
    uint64_t epoch) const
{
    uint64_t value = config_.session_cookie ^ 0x48414e444c453256ull;
    value = Mix64(value ^ config_.session_id);
    value = Mix64(value ^ config_.placement_epoch);
    value = Mix64(value ^ generation);
    value = Mix64(value ^ request_id);
    value = Mix64(value ^ epoch);
    return Mix64(value ^ slot_index);
}

uint64_t IncDcInlineRouteJournal::RouteAuthenticator(
    uint32_t slot_index, uint32_t contribution_index,
    uint64_t generation, uint64_t request_id, uint64_t epoch) const
{
    uint64_t value = config_.session_cookie ^ 0x524f5554454b3256ull;
    value = Mix64(value ^ config_.session_id);
    value = Mix64(value ^ config_.placement_epoch);
    value = Mix64(value ^ generation);
    value = Mix64(value ^ request_id);
    value = Mix64(value ^ epoch);
    value = Mix64(value ^ slot_index);
    return Mix64(value ^ contribution_index);
}

InlineRouteKeyV2 IncDcInlineRouteJournal::MakeRouteKey(
    uint32_t slot_index, uint32_t contribution_index,
    const GenerationSlot &slot) const
{
    InlineRouteKeyV2 key{};
    key.journal_locator = (static_cast<uint64_t>(slot_index) << 32u) |
                          contribution_index;
    key.authenticator = RouteAuthenticator(
        slot_index, contribution_index, slot.generation, slot.request_id,
        slot.epoch);
    return key;
}

IncDcInlineRouteJournal::GenerationSlot *
IncDcInlineRouteJournal::FindSlot(
    const IncDcInlineGenerationHandleV2 &handle)
{
    if (handle.reserved != 0u || handle.slot_index >= slots_.size()) {
        return nullptr;
    }
    GenerationSlot &slot = slots_[handle.slot_index];
    if (slot.state == IncDcInlineGenerationState::FREE ||
        slot.generation != handle.generation ||
        slot.request_id != handle.request_id ||
        slot.epoch != handle.slot_epoch ||
        handle.authenticator != HandleAuthenticator(
            handle.slot_index, handle.generation, handle.request_id,
            handle.slot_epoch)) {
        return nullptr;
    }
    return &slot;
}

const IncDcInlineRouteJournal::GenerationSlot *
IncDcInlineRouteJournal::FindSlot(
    const IncDcInlineGenerationHandleV2 &handle) const
{
    return const_cast<IncDcInlineRouteJournal *>(this)->FindSlot(handle);
}

IncDcInlineJournalStatus IncDcInlineRouteJournal::OpenGeneration(
    const IncDcInlineGenerationOpenV2 &open,
    IncDcInlineGenerationHandleV2 *handle)
{
    if (handle == nullptr || open.generation == 0u || open.request_id == 0u ||
        open.result_capacity > config_.max_results_per_slot ||
        open.contribution_capacity > config_.max_contributions_per_slot) {
        return IncDcInlineJournalStatus::INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (const GenerationSlot &slot : slots_) {
        if (slot.state != IncDcInlineGenerationState::FREE &&
            slot.generation == open.generation &&
            slot.request_id == open.request_id) {
            return IncDcInlineJournalStatus::GENERATION_ALREADY_OPEN;
        }
    }
    uint32_t slot_index = 0u;
    while (slot_index < slots_.size() &&
           slots_[slot_index].state != IncDcInlineGenerationState::FREE) {
        ++slot_index;
    }
    if (slot_index == slots_.size()) {
        return IncDcInlineJournalStatus::NO_FREE_GENERATION_SLOT;
    }
    uint64_t bytes = 0u;
    if (!AddBytes(open.result_capacity, sizeof(ResultState), &bytes) ||
        !AddBytes(open.contribution_capacity, sizeof(ContributionState),
                  &bytes) ||
        bytes > std::numeric_limits<size_t>::max()) {
        return IncDcInlineJournalStatus::CAPACITY_EXCEEDED;
    }

    GenerationSlot &slot = slots_[slot_index];
    try {
        slot.results.clear();
        slot.contributions.clear();
        slot.results.reserve(open.result_capacity);
        slot.contributions.reserve(open.contribution_capacity);
    } catch (const std::bad_alloc &) {
        slot.results.clear();
        slot.contributions.clear();
        return IncDcInlineJournalStatus::NO_MEMORY;
    } catch (...) {
        slot.results.clear();
        slot.contributions.clear();
        return IncDcInlineJournalStatus::CAPACITY_EXCEEDED;
    }
    ++slot.epoch;
    if (slot.epoch == 0u) ++slot.epoch;
    slot.state = IncDcInlineGenerationState::DISPATCH_OPEN;
    slot.generation = open.generation;
    slot.request_id = open.request_id;
    slot.result_capacity = open.result_capacity;
    slot.contribution_capacity = open.contribution_capacity;
    slot.completed_results = 0u;
    slot.arrived_contributions = 0u;

    *handle = {};
    handle->generation = slot.generation;
    handle->request_id = slot.request_id;
    handle->slot_epoch = slot.epoch;
    handle->slot_index = slot_index;
    handle->authenticator = HandleAuthenticator(
        slot_index, slot.generation, slot.request_id, slot.epoch);
    return IncDcInlineJournalStatus::OK;
}

IncDcInlineJournalStatus IncDcInlineRouteJournal::RecordDispatchToken(
    const IncDcInlineGenerationHandleV2 &handle,
    const IncDcInlineTokenRecordV2 &token,
    std::vector<IncDcInlineResolvedRouteV2> *routes)
{
    std::lock_guard<std::mutex> lock(mutex_);
    GenerationSlot *slot = FindSlot(handle);
    if (slot == nullptr) return IncDcInlineJournalStatus::STALE_HANDLE;
    if (slot->state == IncDcInlineGenerationState::ABORTED) {
        return IncDcInlineJournalStatus::GENERATION_ABORTED;
    }
    if (slot->state != IncDcInlineGenerationState::DISPATCH_OPEN) {
        return IncDcInlineJournalStatus::DISPATCH_SEALED;
    }
    if (routes == nullptr || token.origin_rank >= config_.worker_count ||
        token.hidden_bytes == 0u ||
        (token.endpoint_count != 0u && token.endpoints == nullptr) ||
        token.endpoint_count > std::numeric_limits<uint32_t>::max()) {
        return IncDcInlineJournalStatus::INVALID_ARGUMENT;
    }
    if (slot->results.size() >= slot->result_capacity ||
        token.endpoint_count >
            slot->contribution_capacity - slot->contributions.size()) {
        return IncDcInlineJournalStatus::CAPACITY_EXCEEDED;
    }
    for (const ResultState &result : slot->results) {
        if (result.origin_rank == token.origin_rank &&
            result.wave == token.wave &&
            result.origin_token == token.origin_token) {
            return IncDcInlineJournalStatus::DUPLICATE_TOKEN;
        }
    }
    for (uint64_t i = 0u; i < token.endpoint_count; ++i) {
        if (token.endpoints[i].reserved32 != 0u ||
            token.endpoints[i].expert_id >= config_.expert_count ||
            token.endpoints[i].route_ordinal >= token.endpoint_count) {
            return IncDcInlineJournalStatus::INVALID_ARGUMENT;
        }
        for (uint64_t j = 0u; j < i; ++j) {
            if (token.endpoints[i].route_ordinal ==
                token.endpoints[j].route_ordinal) {
                return IncDcInlineJournalStatus::DUPLICATE_ORDINAL;
            }
        }
    }

    std::vector<uint32_t> contributors;
    std::vector<uint32_t> local_experts;
    std::vector<uint64_t> destination_rows;
    try {
        contributors.reserve(static_cast<size_t>(token.endpoint_count));
        local_experts.reserve(static_cast<size_t>(token.endpoint_count));
        destination_rows.reserve(static_cast<size_t>(token.endpoint_count));
        routes->clear();
        routes->reserve(static_cast<size_t>(token.endpoint_count));
        for (uint64_t i = 0u; i < token.endpoint_count; ++i) {
            uint32_t contributor = 0u;
            uint32_t local_expert = 0u;
            bool found = false;
            try {
                found = config_.lookup_placement(
                    config_.placement_context, token.endpoints[i].expert_id,
                    &contributor, &local_expert);
            } catch (...) {
                found = false;
            }
            if (!found || contributor >= config_.worker_count ||
                local_expert >= config_.expert_count) {
                routes->clear();
                return IncDcInlineJournalStatus::PLACEMENT_NOT_FOUND;
            }
            uint64_t destination_row = 0u;
            for (const ContributionState &existing : slot->contributions) {
                if (existing.wave == token.wave &&
                    existing.contributor_rank == contributor &&
                    existing.local_expert == local_expert) {
                    if (existing.destination_row ==
                        std::numeric_limits<uint64_t>::max()) {
                        routes->clear();
                        return IncDcInlineJournalStatus::CAPACITY_EXCEEDED;
                    }
                    destination_row = existing.destination_row + 1u;
                }
            }
            for (size_t previous = 0u; previous < contributors.size();
                 ++previous) {
                if (contributors[previous] == contributor &&
                    local_experts[previous] == local_expert) {
                    if (destination_rows[previous] ==
                        std::numeric_limits<uint64_t>::max()) {
                        routes->clear();
                        return IncDcInlineJournalStatus::CAPACITY_EXCEEDED;
                    }
                    destination_row = destination_rows[previous] + 1u;
                }
            }
            contributors.push_back(contributor);
            local_experts.push_back(local_expert);
            destination_rows.push_back(destination_row);
        }
    } catch (const std::bad_alloc &) {
        routes->clear();
        return IncDcInlineJournalStatus::NO_MEMORY;
    } catch (...) {
        routes->clear();
        return IncDcInlineJournalStatus::CAPACITY_EXCEEDED;
    }

    const uint32_t result_index =
        static_cast<uint32_t>(slot->results.size());
    const uint32_t first_contribution =
        static_cast<uint32_t>(slot->contributions.size());
    ResultState result{};
    result.origin_rank = token.origin_rank;
    result.origin_token = token.origin_token;
    result.first_contribution = first_contribution;
    result.expected = static_cast<uint32_t>(token.endpoint_count);
    result.wave = token.wave;
    result.hidden_bytes = token.hidden_bytes;
    result.complete = result.expected == 0u;
    slot->results.push_back(result);
    if (result.complete) ++slot->completed_results;

    for (uint32_t i = 0u; i < result.expected; ++i) {
        const uint32_t contribution_index =
            static_cast<uint32_t>(slot->contributions.size());
        const InlineRouteEntryV2 &endpoint = token.endpoints[i];
        ContributionState contribution{};
        contribution.result_index = result_index;
        contribution.contributor_rank = contributors[i];
        contribution.expert_id = endpoint.expert_id;
        contribution.local_expert = local_experts[i];
        contribution.ordinal = endpoint.route_ordinal;
        contribution.weight_bits = endpoint.weight_bits;
        contribution.wave = token.wave;
        contribution.destination_row = destination_rows[i];
        slot->contributions.push_back(contribution);

        IncDcInlineResolvedRouteV2 resolved{};
        resolved.route_key = MakeRouteKey(
            handle.slot_index, contribution_index, *slot);
        resolved.contributor_rank = contribution.contributor_rank;
        resolved.expert_id = contribution.expert_id;
        resolved.local_expert = contribution.local_expert;
        resolved.ordinal = contribution.ordinal;
        resolved.weight_bits = contribution.weight_bits;
        resolved.destination_row = contribution.destination_row;
        routes->push_back(resolved);
    }
    return IncDcInlineJournalStatus::OK;
}

IncDcInlineJournalStatus
IncDcInlineRouteJournal::RecordCombineContribution(
    const InlineRouteKeyV2 &route_key, uint32_t contributor_rank,
    IncDcInlineCombineReceiptV2 *receipt)
{
    if (receipt == nullptr || contributor_rank >= config_.worker_count) {
        return IncDcInlineJournalStatus::INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const uint32_t slot_index =
        static_cast<uint32_t>(route_key.journal_locator >> 32u);
    const uint32_t contribution_index =
        static_cast<uint32_t>(route_key.journal_locator);
    if (slot_index >= slots_.size()) {
        return IncDcInlineJournalStatus::INVALID_ROUTE_KEY;
    }
    GenerationSlot &slot = slots_[slot_index];
    if (slot.state == IncDcInlineGenerationState::FREE) {
        return IncDcInlineJournalStatus::STALE_ROUTE_KEY;
    }
    if (contribution_index >= slot.contributions.size()) {
        return IncDcInlineJournalStatus::INVALID_ROUTE_KEY;
    }
    if (route_key.authenticator != RouteAuthenticator(
            slot_index, contribution_index, slot.generation, slot.request_id,
            slot.epoch)) {
        return IncDcInlineJournalStatus::STALE_ROUTE_KEY;
    }
    if (slot.state == IncDcInlineGenerationState::ABORTED) {
        return IncDcInlineJournalStatus::GENERATION_ABORTED;
    }
    if (slot.state != IncDcInlineGenerationState::DISPATCH_OPEN &&
        slot.state != IncDcInlineGenerationState::DISPATCH_SEALED) {
        return IncDcInlineJournalStatus::INVALID_STATE;
    }
    ContributionState &contribution = slot.contributions[contribution_index];
    if (contribution.contributor_rank != contributor_rank) {
        return IncDcInlineJournalStatus::WRONG_CONTRIBUTOR;
    }
    if (contribution.arrived) {
        return IncDcInlineJournalStatus::DUPLICATE_CONTRIBUTION;
    }
    if (contribution.result_index >= slot.results.size()) {
        return IncDcInlineJournalStatus::INVALID_ROUTE_KEY;
    }
    ResultState &result = slot.results[contribution.result_index];
    if (result.arrived >= result.expected) {
        return IncDcInlineJournalStatus::DUPLICATE_CONTRIBUTION;
    }

    contribution.arrived = true;
    ++result.arrived;
    ++slot.arrived_contributions;
    if (result.arrived == result.expected) {
        result.complete = true;
        ++slot.completed_results;
    }
    *receipt = {};
    receipt->origin_rank = result.origin_rank;
    receipt->origin_token = result.origin_token;
    receipt->ordinal = contribution.ordinal;
    receipt->weight_bits = contribution.weight_bits;
    receipt->expected = result.expected;
    receipt->arrived = result.arrived;
    receipt->wave = result.wave;
    receipt->hidden_bytes = result.hidden_bytes;
    receipt->result_complete = result.complete;
    return IncDcInlineJournalStatus::OK;
}

IncDcInlineJournalStatus IncDcInlineRouteJournal::SealDispatch(
    const IncDcInlineGenerationHandleV2 &handle)
{
    std::lock_guard<std::mutex> lock(mutex_);
    GenerationSlot *slot = FindSlot(handle);
    if (slot == nullptr) return IncDcInlineJournalStatus::STALE_HANDLE;
    if (slot->state == IncDcInlineGenerationState::ABORTED) {
        return IncDcInlineJournalStatus::GENERATION_ABORTED;
    }
    if (slot->state != IncDcInlineGenerationState::DISPATCH_OPEN) {
        return IncDcInlineJournalStatus::INVALID_STATE;
    }
    slot->state = IncDcInlineGenerationState::DISPATCH_SEALED;
    return IncDcInlineJournalStatus::OK;
}

IncDcInlineJournalStatus IncDcInlineRouteJournal::AbortGeneration(
    const IncDcInlineGenerationHandleV2 &handle)
{
    std::lock_guard<std::mutex> lock(mutex_);
    GenerationSlot *slot = FindSlot(handle);
    if (slot == nullptr) return IncDcInlineJournalStatus::STALE_HANDLE;
    slot->state = IncDcInlineGenerationState::ABORTED;
    return IncDcInlineJournalStatus::OK;
}

IncDcInlineJournalStatus IncDcInlineRouteJournal::ReleaseGeneration(
    const IncDcInlineGenerationHandleV2 &handle)
{
    std::lock_guard<std::mutex> lock(mutex_);
    GenerationSlot *slot = FindSlot(handle);
    if (slot == nullptr) return IncDcInlineJournalStatus::STALE_HANDLE;
    if (slot->state == IncDcInlineGenerationState::DISPATCH_OPEN) {
        return IncDcInlineJournalStatus::INVALID_STATE;
    }
    if (slot->state == IncDcInlineGenerationState::DISPATCH_SEALED &&
        (slot->completed_results != slot->results.size() ||
         slot->arrived_contributions != slot->contributions.size())) {
        return IncDcInlineJournalStatus::INCOMPLETE;
    }
    slot->state = IncDcInlineGenerationState::FREE;
    slot->generation = 0u;
    slot->request_id = 0u;
    slot->result_capacity = 0u;
    slot->contribution_capacity = 0u;
    slot->completed_results = 0u;
    slot->arrived_contributions = 0u;
    slot->results.clear();
    slot->contributions.clear();
    return IncDcInlineJournalStatus::OK;
}

IncDcInlineJournalStatus IncDcInlineRouteJournal::QueryGeneration(
    const IncDcInlineGenerationHandleV2 &handle,
    IncDcInlineGenerationSummaryV2 *summary) const
{
    if (summary == nullptr) return IncDcInlineJournalStatus::INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(mutex_);
    const GenerationSlot *slot = FindSlot(handle);
    if (slot == nullptr) return IncDcInlineJournalStatus::STALE_HANDLE;
    *summary = {};
    summary->state = slot->state;
    summary->generation = slot->generation;
    summary->request_id = slot->request_id;
    summary->slot_epoch = slot->epoch;
    summary->result_count = static_cast<uint32_t>(slot->results.size());
    summary->contribution_count =
        static_cast<uint32_t>(slot->contributions.size());
    summary->completed_result_count = slot->completed_results;
    summary->arrived_contribution_count = slot->arrived_contributions;
    return IncDcInlineJournalStatus::OK;
}

const char *IncDcInlineJournalStatusString(IncDcInlineJournalStatus status)
{
    switch (status) {
    case IncDcInlineJournalStatus::OK: return "OK";
    case IncDcInlineJournalStatus::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
    case IncDcInlineJournalStatus::INVALID_CONFIG: return "INVALID_CONFIG";
    case IncDcInlineJournalStatus::NO_MEMORY: return "NO_MEMORY";
    case IncDcInlineJournalStatus::CAPACITY_EXCEEDED: return "CAPACITY_EXCEEDED";
    case IncDcInlineJournalStatus::NO_FREE_GENERATION_SLOT: return "NO_FREE_GENERATION_SLOT";
    case IncDcInlineJournalStatus::GENERATION_ALREADY_OPEN: return "GENERATION_ALREADY_OPEN";
    case IncDcInlineJournalStatus::STALE_HANDLE: return "STALE_HANDLE";
    case IncDcInlineJournalStatus::INVALID_STATE: return "INVALID_STATE";
    case IncDcInlineJournalStatus::DISPATCH_SEALED: return "DISPATCH_SEALED";
    case IncDcInlineJournalStatus::PLACEMENT_NOT_FOUND: return "PLACEMENT_NOT_FOUND";
    case IncDcInlineJournalStatus::DUPLICATE_TOKEN: return "DUPLICATE_TOKEN";
    case IncDcInlineJournalStatus::DUPLICATE_ORDINAL: return "DUPLICATE_ORDINAL";
    case IncDcInlineJournalStatus::INVALID_ROUTE_KEY: return "INVALID_ROUTE_KEY";
    case IncDcInlineJournalStatus::STALE_ROUTE_KEY: return "STALE_ROUTE_KEY";
    case IncDcInlineJournalStatus::WRONG_CONTRIBUTOR: return "WRONG_CONTRIBUTOR";
    case IncDcInlineJournalStatus::DUPLICATE_CONTRIBUTION: return "DUPLICATE_CONTRIBUTION";
    case IncDcInlineJournalStatus::GENERATION_ABORTED: return "GENERATION_ABORTED";
    case IncDcInlineJournalStatus::INCOMPLETE: return "INCOMPLETE";
    }
    return "UNKNOWN";
}

} // namespace inc::dc
