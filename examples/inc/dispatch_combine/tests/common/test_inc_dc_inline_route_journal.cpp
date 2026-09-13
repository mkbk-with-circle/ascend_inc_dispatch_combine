#include "inc_dc_inline_route_journal.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

using namespace inc::dc;

namespace {

bool LookupPlacement(void *, uint32_t expert, uint32_t *destination,
                     uint32_t *local_expert)
{
    if (destination == nullptr || local_expert == nullptr || expert >= 31u)
        return false;
    *destination = expert % 4u;
    *local_expert = expert / 4u;
    return true;
}

IncDcInlineRouteJournalConfig Config()
{
    IncDcInlineRouteJournalConfig config{};
    config.worker_count = 4u;
    config.expert_count = 32u;
    config.generation_slot_count = 2u;
    config.max_results_per_slot = 8u;
    config.max_contributions_per_slot = 32u;
    config.session_id = 0x51u;
    config.placement_epoch = 3u;
    config.session_cookie = 0x123456789abcdef0ull;
    config.lookup_placement = LookupPlacement;
    return config;
}

} // namespace

int main()
{
    {
        IncDcInlineRouteJournalConfig overflow = Config();
        overflow.generation_slot_count = UINT32_MAX;
        overflow.max_results_per_slot = UINT32_MAX;
        overflow.max_contributions_per_slot = UINT32_MAX;
        std::unique_ptr<IncDcInlineRouteJournal> rejected;
        assert(IncDcInlineRouteJournal::Create(overflow, &rejected) ==
               IncDcInlineJournalStatus::INVALID_CONFIG);
    }
    std::unique_ptr<IncDcInlineRouteJournal> journal;
    assert(IncDcInlineRouteJournal::Create(Config(), &journal) ==
           IncDcInlineJournalStatus::OK);

    IncDcInlineGenerationHandleV2 generation7{};
    assert(journal->OpenGeneration({7u, 70u, 4u, 8u}, &generation7) ==
           IncDcInlineJournalStatus::OK);
    assert(journal->OpenGeneration({7u, 70u, 1u, 1u}, &generation7) ==
           IncDcInlineJournalStatus::GENERATION_ALREADY_OPEN);

    const InlineRouteEntryV2 endpoints[] = {
        {10u, 0u, 0x3f800000u},
        {11u, 1u, 0x3f000000u},
    };
    const IncDcInlineTokenRecordV2 token{
        1u, 99u, 3u, 8192u, endpoints, 2u};
    std::vector<IncDcInlineResolvedRouteV2> routes;
    assert(journal->RecordDispatchToken(generation7, token, &routes) ==
           IncDcInlineJournalStatus::OK);
    assert(routes.size() == 2u);
    assert(routes[0].contributor_rank == 2u);
    assert(routes[1].contributor_rank == 3u);
    assert(routes[0].local_expert == 2u &&
           routes[1].local_expert == 2u);
    assert(routes[0].destination_row == 0u &&
           routes[1].destination_row == 0u);
    const InlineRouteKeyV2 first_generation_first_key = routes[0].route_key;

    IncDcInlineGenerationSummaryV2 summary{};
    assert(journal->QueryGeneration(generation7, &summary) ==
           IncDcInlineJournalStatus::OK);
    assert(summary.result_count == 1u && summary.contribution_count == 2u);
    assert(summary.request_id == 70u);

    // A failed Dispatch record is transactional.
    const InlineRouteEntryV2 missing_placement[] = {
        {31u, 0u, 0x3f800000u}};
    assert(journal->RecordDispatchToken(
               generation7,
               {0u, 1u, 3u, 8192u, missing_placement, 1u}, &routes) ==
           IncDcInlineJournalStatus::PLACEMENT_NOT_FOUND);
    const InlineRouteEntryV2 duplicate_ordinals[] = {
        {4u, 1u, 0x3f800000u}, {8u, 1u, 0x3f800000u}};
    assert(journal->RecordDispatchToken(
               generation7,
               {0u, 1u, 3u, 8192u, duplicate_ordinals, 2u}, &routes) ==
           IncDcInlineJournalStatus::DUPLICATE_ORDINAL);
    assert(journal->QueryGeneration(generation7, &summary) ==
           IncDcInlineJournalStatus::OK);
    assert(summary.result_count == 1u && summary.contribution_count == 2u);
    assert(journal->RecordDispatchToken(generation7, token, &routes) ==
           IncDcInlineJournalStatus::DUPLICATE_TOKEN);

    // Recover the valid routes after the transactional-failure checks.
    assert(journal->RecordDispatchToken(
               generation7, {2u, 5u, 3u, 8192u, endpoints, 2u}, &routes) ==
           IncDcInlineJournalStatus::OK);
    assert(routes[0].destination_row == 1u &&
           routes[1].destination_row == 1u);
    const InlineRouteKeyV2 first_key = routes[0].route_key;
    const InlineRouteKeyV2 second_key = routes[1].route_key;

    IncDcInlineCombineReceiptV2 receipt{};
    assert(journal->RecordCombineContribution(first_key, 3u, &receipt) ==
           IncDcInlineJournalStatus::WRONG_CONTRIBUTOR);
    assert(journal->RecordCombineContribution(first_key, 2u, &receipt) ==
           IncDcInlineJournalStatus::OK);
    assert(receipt.origin_rank == 2u && receipt.origin_token == 5u);
    assert(receipt.expected == 2u && receipt.arrived == 1u &&
           !receipt.result_complete);
    assert(receipt.wave == 3u && receipt.hidden_bytes == 8192u);
    assert(journal->RecordCombineContribution(first_key, 2u, &receipt) ==
           IncDcInlineJournalStatus::DUPLICATE_CONTRIBUTION);
    assert(journal->RecordCombineContribution(second_key, 3u, &receipt) ==
           IncDcInlineJournalStatus::OK);
    assert(receipt.result_complete && receipt.arrived == 2u);

    // The first token is still missing; a sealed generation cannot be freed.
    assert(journal->SealDispatch(generation7) ==
           IncDcInlineJournalStatus::OK);
    assert(journal->ReleaseGeneration(generation7) ==
           IncDcInlineJournalStatus::INCOMPLETE);
    assert(journal->AbortGeneration(generation7) ==
           IncDcInlineJournalStatus::OK);
    assert(journal->RecordCombineContribution(
               first_key, 2u, &receipt) ==
           IncDcInlineJournalStatus::GENERATION_ABORTED);
    assert(journal->ReleaseGeneration(generation7) ==
           IncDcInlineJournalStatus::OK);
    assert(journal->QueryGeneration(generation7, &summary) ==
           IncDcInlineJournalStatus::STALE_HANDLE);
    assert(journal->RecordCombineContribution(first_key, 2u, &receipt) ==
           IncDcInlineJournalStatus::STALE_ROUTE_KEY);

    // Slot reuse increments epoch, so old keys cannot alias new state.
    IncDcInlineGenerationHandleV2 generation8{};
    assert(journal->OpenGeneration({8u, 80u, 1u, 2u}, &generation8) ==
           IncDcInlineJournalStatus::OK);
    assert(generation8.slot_index == generation7.slot_index);
    assert(generation8.slot_epoch != generation7.slot_epoch);
    assert(journal->RecordDispatchToken(
               generation8, {0u, 7u, 0u, 4096u, endpoints, 2u}, &routes) ==
           IncDcInlineJournalStatus::OK);
    assert(journal->RecordCombineContribution(
               first_generation_first_key, 2u, &receipt) ==
           IncDcInlineJournalStatus::STALE_ROUTE_KEY);
    InlineRouteKeyV2 corrupted = routes[0].route_key;
    corrupted.authenticator ^= 1u;
    assert(journal->RecordCombineContribution(corrupted, 2u, &receipt) ==
           IncDcInlineJournalStatus::STALE_ROUTE_KEY);
    assert(journal->RecordCombineContribution(
               routes[0].route_key, 2u, &receipt) ==
           IncDcInlineJournalStatus::OK);
    assert(journal->RecordCombineContribution(
               routes[1].route_key, 3u, &receipt) ==
           IncDcInlineJournalStatus::OK);
    assert(journal->SealDispatch(generation8) ==
           IncDcInlineJournalStatus::OK);
    assert(journal->ReleaseGeneration(generation8) ==
           IncDcInlineJournalStatus::OK);

    // Dynamic slot capacity is enforced independently of configured maxima.
    IncDcInlineGenerationHandleV2 small{};
    assert(journal->OpenGeneration({9u, 90u, 1u, 1u}, &small) ==
           IncDcInlineJournalStatus::OK);
    assert(journal->RecordDispatchToken(
               small, {0u, 0u, 0u, 2048u, endpoints, 2u}, &routes) ==
           IncDcInlineJournalStatus::CAPACITY_EXCEEDED);
    assert(journal->AbortGeneration(small) == IncDcInlineJournalStatus::OK);
    assert(journal->ReleaseGeneration(small) ==
           IncDcInlineJournalStatus::OK);

    return 0;
}
