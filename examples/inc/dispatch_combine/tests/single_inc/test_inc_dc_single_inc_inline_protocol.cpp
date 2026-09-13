#include "inc_dc_inline_route_journal.h"
#include "inc_dc_inline_route_parser.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

using namespace inc::dc;

namespace {

constexpr uint64_t kSessionId = 0x53494e474c455632ull;
constexpr uint64_t kPlacementEpoch = 9u;
constexpr uint64_t kGeneration = 17u;
constexpr uint64_t kRequestId = 23u;
constexpr uint32_t kWave = 3u;
constexpr uint32_t kExpertCount = 64u;

void StoreLe16(std::vector<uint8_t> *bytes, size_t offset, uint16_t value)
{
    assert(bytes != nullptr && offset + sizeof(value) <= bytes->size());
    for (uint32_t byte = 0u; byte < sizeof(value); ++byte) {
        (*bytes)[offset + byte] =
            static_cast<uint8_t>(value >> (byte * 8u));
    }
}

void StoreLe32(std::vector<uint8_t> *bytes, size_t offset, uint32_t value)
{
    assert(bytes != nullptr && offset + sizeof(value) <= bytes->size());
    for (uint32_t byte = 0u; byte < sizeof(value); ++byte) {
        (*bytes)[offset + byte] =
            static_cast<uint8_t>(value >> (byte * 8u));
    }
}

void StoreLe64(std::vector<uint8_t> *bytes, size_t offset, uint64_t value)
{
    assert(bytes != nullptr && offset + sizeof(value) <= bytes->size());
    for (uint32_t byte = 0u; byte < sizeof(value); ++byte) {
        (*bytes)[offset + byte] =
            static_cast<uint8_t>(value >> (byte * 8u));
    }
}

uint32_t FloatBits(float value)
{
    uint32_t bits = 0u;
    static_assert(sizeof(bits) == sizeof(value), "float32 is required");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float BitsFloat(uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

struct RouteSpec {
    uint32_t expert_id = 0u;
    float weight = 0.0f;
};

struct TokenSpec {
    uint64_t source_token = 0u;
    std::vector<RouteSpec> routes;
    std::vector<float> hidden;
};

std::vector<uint8_t> BuildTokenRecord(const TokenSpec &token,
                                      uint32_t source_rank,
                                      bool committed)
{
    assert(!token.routes.empty() && !token.hidden.empty());
    const uint64_t hidden_bytes =
        static_cast<uint64_t>(token.hidden.size() * sizeof(float));
    uint64_t payload_offset = 0u;
    uint64_t record_bytes = 0u;
    assert(InlineTokenRecordLayout(
        static_cast<uint32_t>(token.routes.size()), hidden_bytes,
        &payload_offset, &record_bytes));
    std::vector<uint8_t> record(static_cast<size_t>(record_bytes), 0u);

    StoreLe32(&record, 0u, kInlineRouteTokenMagic);
    StoreLe16(&record, 4u, kInlineRouteVersion);
    StoreLe16(&record, 6u, kInlineRouteHeaderBytes);
    StoreLe32(&record, 8u, kInlineRouteRecordNone);
    StoreLe32(&record, 12u, source_rank);
    StoreLe64(&record, 16u, kSessionId);
    StoreLe64(&record, 24u, kPlacementEpoch);
    StoreLe64(&record, 32u, kGeneration);
    StoreLe64(&record, 40u, kRequestId);
    StoreLe64(&record, 48u, token.source_token);
    StoreLe64(&record, 56u, hidden_bytes);
    StoreLe64(&record, 64u, payload_offset);
    StoreLe64(&record, 72u, record_bytes);
    StoreLe32(&record, 80u, static_cast<uint32_t>(token.routes.size()));
    StoreLe32(&record, 84u, sizeof(InlineRouteEntryV2));
    StoreLe32(&record, 88u, kWave);

    for (uint32_t index = 0u; index < token.routes.size(); ++index) {
        const size_t offset = kInlineRouteHeaderBytes +
            static_cast<size_t>(index) * sizeof(InlineRouteEntryV2);
        StoreLe32(&record, offset, token.routes[index].expert_id);
        StoreLe32(&record, offset + 4u, index);
        StoreLe32(&record, offset + 8u,
                  FloatBits(token.routes[index].weight));
    }
    std::memcpy(record.data() + static_cast<size_t>(payload_offset),
                token.hidden.data(), static_cast<size_t>(hidden_bytes));

    // Commit is the last store in the logical producer sequence.
    if (committed) StoreLe64(&record, 96u, kGeneration);
    return record;
}

struct BatchBuffer {
    std::vector<uint8_t> bytes;
    std::vector<uint64_t> record_offsets;
};

BatchBuffer BuildBatch(const std::vector<TokenSpec> &tokens,
                       uint32_t source_rank)
{
    assert(!tokens.empty());
    std::vector<std::vector<uint8_t>> records;
    records.reserve(tokens.size());
    for (size_t index = 0u; index < tokens.size(); ++index) {
        records.push_back(BuildTokenRecord(tokens[index], source_rank,
                                           index == 0u));
    }

    uint64_t table_bytes = 0u;
    uint64_t records_offset = 0u;
    assert(InlineBatchPreambleLayout(static_cast<uint32_t>(records.size()),
                                     &table_bytes, &records_offset));
    BatchBuffer batch{};
    batch.record_offsets.push_back(records_offset);
    uint64_t next = records_offset;
    for (const std::vector<uint8_t> &record : records) {
        assert(InlineStoredRecordEnd(next, record.size(), &next));
        batch.record_offsets.push_back(next);
    }
    batch.bytes.resize(static_cast<size_t>(next), 0u);

    StoreLe32(&batch.bytes, 0u, kInlineRouteBatchMagic);
    StoreLe16(&batch.bytes, 4u, kInlineRouteVersion);
    StoreLe16(&batch.bytes, 6u, kInlineRouteHeaderBytes);
    StoreLe32(&batch.bytes, 8u, kInlineRouteRecordNone);
    StoreLe32(&batch.bytes, 12u, source_rank);
    StoreLe64(&batch.bytes, 16u, kSessionId);
    StoreLe64(&batch.bytes, 24u, kPlacementEpoch);
    StoreLe64(&batch.bytes, 32u, kGeneration);
    StoreLe64(&batch.bytes, 40u, kRequestId);
    StoreLe32(&batch.bytes, 48u, kWave);
    StoreLe32(&batch.bytes, 52u,
              static_cast<uint32_t>(records.size()));
    StoreLe64(&batch.bytes, 56u, table_bytes);
    StoreLe64(&batch.bytes, 64u, records_offset);
    StoreLe64(&batch.bytes, 72u, batch.bytes.size());
    for (size_t index = 0u; index < batch.record_offsets.size(); ++index) {
        StoreLe64(&batch.bytes,
                  kInlineRouteHeaderBytes + index * sizeof(uint64_t),
                  batch.record_offsets[index]);
    }
    for (size_t index = 0u; index < records.size(); ++index) {
        std::memcpy(batch.bytes.data() +
                        static_cast<size_t>(batch.record_offsets[index]),
                    records[index].data(), records[index].size());
    }

    // The directory is complete only after offsets and stored records exist.
    StoreLe64(&batch.bytes, 80u, kGeneration);
    return batch;
}

struct PlacementContext {
    uint32_t worker_count = 0u;
};

bool LookupPlacement(void *opaque, uint32_t expert_id,
                     uint32_t *destination_rank, uint32_t *local_expert)
{
    const auto *context = static_cast<const PlacementContext *>(opaque);
    if (context == nullptr || context->worker_count == 0u ||
        destination_rank == nullptr || local_expert == nullptr ||
        expert_id >= kExpertCount) {
        return false;
    }
    *destination_rank = expert_id % context->worker_count;
    *local_expert = expert_id / context->worker_count;
    return true;
}

struct FanoutPacket {
    InlineFanoutRecordHeaderV2 header{};
    std::vector<InlineFanoutAssignmentV2> assignments;
    // One payload vector per destination packet, regardless of assignment
    // count. This models the INC transmitting a token's hidden state once to
    // a worker that hosts several selected experts.
    std::vector<float> hidden;
};

struct WorkerContribution {
    InlineRouteKeyV2 route_key{};
    uint32_t contributor_rank = 0u;
    std::vector<float> output;
};

struct ResultPacket {
    InlineResultRecordHeaderV2 header{};
    std::vector<float> payload;
};

using ResultId = std::pair<uint32_t, uint64_t>;

struct CombineModel {
    std::map<ResultId, std::vector<float>> accumulators;
    std::map<ResultId, ResultPacket> completed;

    IncDcInlineJournalStatus Accept(
        IncDcInlineRouteJournal *journal,
        const WorkerContribution &contribution)
    {
        assert(journal != nullptr);
        IncDcInlineCombineReceiptV2 receipt{};
        const IncDcInlineJournalStatus status =
            journal->RecordCombineContribution(contribution.route_key,
                                                contribution.contributor_rank,
                                                &receipt);
        if (status != IncDcInlineJournalStatus::OK) return status;

        assert(receipt.hidden_bytes ==
               contribution.output.size() * sizeof(float));
        const ResultId id{receipt.origin_rank, receipt.origin_token};
        std::vector<float> &sum = accumulators[id];
        if (sum.empty()) sum.assign(contribution.output.size(), 0.0f);
        assert(sum.size() == contribution.output.size());
        const float weight = BitsFloat(receipt.weight_bits);
        for (size_t element = 0u; element < sum.size(); ++element) {
            sum[element] += contribution.output[element] * weight;
        }
        if (receipt.result_complete) {
            assert(completed.count(id) == 0u);
            ResultPacket result{};
            result.header.origin_rank = receipt.origin_rank;
            result.header.session_id = kSessionId;
            result.header.placement_epoch = kPlacementEpoch;
            result.header.generation = kGeneration;
            result.header.request_id = kRequestId;
            result.header.source_token = receipt.origin_token;
            result.header.payload_bytes = receipt.hidden_bytes;
            result.header.record_bytes =
                kInlineRouteHeaderBytes + receipt.hidden_bytes;
            result.header.wave = receipt.wave;
            result.payload = sum;
            completed.emplace(id, std::move(result));
        }
        return status;
    }
};

struct ParsedToken {
    std::vector<FanoutPacket> fanouts;
    uint64_t transmitted_hidden_bytes = 0u;
};

ParsedToken ParseJournalAndFanout(
    const InlineRouteBatchViewV2 &batch, uint32_t token_index,
    const IncDcInlineGenerationHandleV2 &handle,
    IncDcInlineRouteJournal *journal)
{
    assert(journal != nullptr);
    InlineRouteTokenViewV2 token{};
    assert(InlineBatchTokenAtV2(batch, token_index, &token) ==
           InlineRouteParseStatusV2::OK);

    std::vector<InlineRouteEntryV2> endpoints(token.route_count);
    for (uint32_t index = 0u; index < token.route_count; ++index) {
        InlineRouteEntryValueV2 value{};
        assert(InlineTokenRouteAtV2(token, index, &value) ==
               InlineRouteParseStatusV2::OK);
        endpoints[index].expert_id = value.expert_id;
        endpoints[index].route_ordinal = value.route_ordinal;
        endpoints[index].weight_bits = value.weight_bits;
    }
    std::vector<float> hidden(static_cast<size_t>(token.hidden_bytes /
                                                   sizeof(float)));
    assert(token.hidden_bytes % sizeof(float) == 0u);
    const uint8_t *payload = InlineTokenPayloadV2(token);
    assert(payload != nullptr);
    std::memcpy(hidden.data(), payload, static_cast<size_t>(token.hidden_bytes));

    std::vector<IncDcInlineResolvedRouteV2> routes;
    assert(journal->RecordDispatchToken(
               handle,
               {token.source_rank, token.source_token, token.wave,
                token.hidden_bytes, endpoints.data(), endpoints.size()},
               &routes) == IncDcInlineJournalStatus::OK);
    assert(routes.size() == endpoints.size());

    std::map<uint32_t, FanoutPacket> by_destination;
    for (const IncDcInlineResolvedRouteV2 &route : routes) {
        FanoutPacket &packet = by_destination[route.contributor_rank];
        if (packet.assignments.empty()) {
            packet.header.resolved_destination_rank = route.contributor_rank;
            packet.header.session_id = token.session_id;
            packet.header.placement_epoch = token.placement_epoch;
            packet.header.generation = token.generation;
            packet.header.request_id = token.request_id;
            packet.header.source_token = token.source_token;
            packet.header.hidden_bytes = token.hidden_bytes;
            packet.header.source_rank = token.source_rank;
            packet.header.wave = token.wave;
            packet.hidden = hidden;
        }
        InlineFanoutAssignmentV2 assignment{};
        assignment.route_key = route.route_key;
        assignment.expert_id = route.expert_id;
        assignment.local_expert = route.local_expert;
        assignment.route_ordinal = route.ordinal;
        assignment.destination_row = route.destination_row;
        packet.assignments.push_back(assignment);
    }

    ParsedToken parsed{};
    parsed.fanouts.reserve(by_destination.size());
    for (auto &entry : by_destination) {
        FanoutPacket &packet = entry.second;
        packet.header.assignment_count =
            static_cast<uint32_t>(packet.assignments.size());
        uint64_t payload_offset = 0u;
        uint64_t record_bytes = 0u;
        assert(InlineFanoutRecordLayout(packet.header.assignment_count,
                                        packet.header.hidden_bytes,
                                        &payload_offset, &record_bytes));
        packet.header.payload_offset = payload_offset;
        packet.header.record_bytes = record_bytes;
        parsed.transmitted_hidden_bytes += packet.header.hidden_bytes;
        parsed.fanouts.push_back(std::move(packet));
    }
    return parsed;
}

std::vector<WorkerContribution> RunIdentityExperts(
    const ParsedToken &token)
{
    std::vector<WorkerContribution> contributions;
    for (const FanoutPacket &packet : token.fanouts) {
        assert(packet.header.assignment_count == packet.assignments.size());
        for (const InlineFanoutAssignmentV2 &assignment :
             packet.assignments) {
            WorkerContribution contribution{};
            contribution.route_key = assignment.route_key;
            contribution.contributor_rank =
                packet.header.resolved_destination_rank;
            contribution.output = packet.hidden;
            contributions.push_back(std::move(contribution));
        }
    }
    // Combine arrival order is independent of Dispatch route order.
    std::reverse(contributions.begin(), contributions.end());
    return contributions;
}

void AssertClose(const std::vector<float> &actual,
                 const std::vector<float> &expected)
{
    assert(actual.size() == expected.size());
    for (size_t index = 0u; index < actual.size(); ++index) {
        assert(std::fabs(actual[index] - expected[index]) <= 1.0e-6f);
    }
}

void AssertMapUnchanged(
    const std::map<ResultId, std::vector<float>> &before,
    const std::map<ResultId, std::vector<float>> &after)
{
    assert(before.size() == after.size());
    auto left = before.begin();
    auto right = after.begin();
    for (; left != before.end(); ++left, ++right) {
        assert(left->first == right->first);
        AssertClose(left->second, right->second);
    }
}

std::vector<RouteSpec> RoutesFor(uint32_t worker_count, uint32_t topk)
{
    assert(topk == 1u || topk == 2u || topk == 4u);
    const uint32_t experts[] = {0u, worker_count, 1u, worker_count + 1u};
    std::vector<RouteSpec> routes;
    for (uint32_t index = 0u; index < topk; ++index) {
        routes.push_back({experts[index], 0.25f * (index + 1u)});
    }
    return routes;
}

void RunCase(uint32_t worker_count, uint32_t topk)
{
    assert(topk <= worker_count);
    const uint32_t source_rank = worker_count - 1u;
    const std::vector<RouteSpec> routes = RoutesFor(worker_count, topk);
    const std::vector<TokenSpec> tokens{
        {10u, routes, {1.0f, -2.0f, 3.5f, 0.25f, 8.0f}},
        {11u, routes, {-4.0f, 5.0f, 0.5f, 7.0f, -1.25f}},
    };
    BatchBuffer batch_storage = BuildBatch(tokens, source_rank);

    InlineRouteBatchParseContextV2 parse_context{};
    parse_context.token.expected_source_rank = source_rank;
    parse_context.token.worker_count = worker_count;
    parse_context.token.expected_session_id = kSessionId;
    parse_context.token.expected_placement_epoch = kPlacementEpoch;
    parse_context.token.expected_generation = kGeneration;
    parse_context.token.expected_request_id = kRequestId;
    parse_context.token.expected_wave = kWave;
    parse_context.token.expert_count = kExpertCount;
    parse_context.token.max_route_count = worker_count;
    parse_context.token.max_source_tokens = 1024u;
    parse_context.token.expected_hidden_bytes =
        tokens[0].hidden.size() * sizeof(float);
    parse_context.token.max_hidden_bytes = 4096u;
    parse_context.token.max_record_bytes = 8192u;
    parse_context.max_record_count = 8u;
    parse_context.max_frame_bytes = 65536u;

    InlineRouteBatchViewV2 batch{};
    assert(ParseInlineTokenBatchV2(batch_storage.bytes.data(),
                                   batch_storage.bytes.size(), parse_context,
                                   &batch) == InlineRouteParseStatusV2::OK);
    InlineRouteTokenViewV2 token_view{};
    assert(InlineBatchTokenAtV2(batch, 0u, &token_view) ==
           InlineRouteParseStatusV2::OK);
    assert(InlineBatchTokenAtV2(batch, 1u, &token_view) ==
           InlineRouteParseStatusV2::NOT_COMMITTED);

    PlacementContext placement{worker_count};
    IncDcInlineRouteJournalConfig journal_config{};
    journal_config.worker_count = worker_count;
    journal_config.expert_count = kExpertCount;
    journal_config.generation_slot_count = 1u;
    journal_config.max_results_per_slot = tokens.size();
    journal_config.max_contributions_per_slot = tokens.size() * topk;
    journal_config.session_id = kSessionId;
    journal_config.placement_epoch = kPlacementEpoch;
    journal_config.session_cookie = 0x9e3779b97f4a7c15ull ^
        (static_cast<uint64_t>(worker_count) << 32u) ^ topk;
    journal_config.lookup_placement = LookupPlacement;
    journal_config.placement_context = &placement;
    std::unique_ptr<IncDcInlineRouteJournal> journal;
    assert(IncDcInlineRouteJournal::Create(journal_config, &journal) ==
           IncDcInlineJournalStatus::OK);
    IncDcInlineGenerationHandleV2 handle{};
    assert(journal->OpenGeneration(
               {kGeneration, kRequestId,
                static_cast<uint32_t>(tokens.size()),
                static_cast<uint32_t>(tokens.size() * topk)},
               &handle) == IncDcInlineJournalStatus::OK);

    CombineModel combine{};
    ParsedToken first = ParseJournalAndFanout(batch, 0u, handle,
                                              journal.get());
    std::set<uint32_t> unique_destinations;
    for (const RouteSpec &route : routes) {
        unique_destinations.insert(route.expert_id % worker_count);
    }
    assert(first.fanouts.size() == unique_destinations.size());
    assert(first.transmitted_hidden_bytes ==
           unique_destinations.size() * tokens[0].hidden.size() *
               sizeof(float));
    if (topk > 1u) {
        assert(first.transmitted_hidden_bytes <
               topk * tokens[0].hidden.size() * sizeof(float));
    }

    std::vector<WorkerContribution> first_outputs =
        RunIdentityExperts(first);
    assert(first_outputs.size() == topk);

    // Fail closed before accepting a contribution: neither a corrupted key
    // nor the wrong physical contributor may alter reduction state.
    const auto empty_accumulators = combine.accumulators;
    WorkerContribution corrupted = first_outputs.front();
    corrupted.route_key.authenticator ^= 1u;
    assert(combine.Accept(journal.get(), corrupted) ==
           IncDcInlineJournalStatus::STALE_ROUTE_KEY);
    AssertMapUnchanged(empty_accumulators, combine.accumulators);
    WorkerContribution wrong_port = first_outputs.front();
    wrong_port.contributor_rank =
        (wrong_port.contributor_rank + 1u) % worker_count;
    assert(combine.Accept(journal.get(), wrong_port) ==
           IncDcInlineJournalStatus::WRONG_CONTRIBUTOR);
    AssertMapUnchanged(empty_accumulators, combine.accumulators);

    for (const WorkerContribution &contribution : first_outputs) {
        assert(combine.Accept(journal.get(), contribution) ==
               IncDcInlineJournalStatus::OK);
    }
    const ResultId first_id{source_rank, tokens[0].source_token};
    assert(combine.completed.count(first_id) == 1u);

    // Duplicate delivery is rejected by the authoritative journal and the
    // already-completed output remains bit-for-bit semantically unchanged.
    const auto completed_first = combine.completed.at(first_id).payload;
    const auto accumulated_first = combine.accumulators;
    assert(combine.Accept(journal.get(), first_outputs.front()) ==
           IncDcInlineJournalStatus::DUPLICATE_CONTRIBUTION);
    AssertMapUnchanged(accumulated_first, combine.accumulators);
    AssertClose(completed_first, combine.completed.at(first_id).payload);

    // This is the key overlap assertion: token 0 has reached the origin while
    // token 1 is still uncommitted and therefore invisible to the INC parser.
    assert(InlineBatchTokenAtV2(batch, 1u, &token_view) ==
           InlineRouteParseStatusV2::NOT_COMMITTED);

    // Publish token 1 independently, then Dispatch and Combine it without a
    // generation-wide Dispatch seal.
    StoreLe64(&batch_storage.bytes,
              static_cast<size_t>(batch_storage.record_offsets[1]) + 96u,
              kGeneration);
    assert(InlineBatchTokenAtV2(batch, 1u, &token_view) ==
           InlineRouteParseStatusV2::OK);
    ParsedToken second = ParseJournalAndFanout(batch, 1u, handle,
                                               journal.get());
    assert(second.fanouts.size() == unique_destinations.size());
    assert(second.transmitted_hidden_bytes ==
           unique_destinations.size() * tokens[1].hidden.size() *
               sizeof(float));
    std::vector<WorkerContribution> second_outputs =
        RunIdentityExperts(second);
    for (const WorkerContribution &contribution : second_outputs) {
        assert(combine.Accept(journal.get(), contribution) ==
               IncDcInlineJournalStatus::OK);
    }

    float weight_sum = 0.0f;
    for (const RouteSpec &route : routes) weight_sum += route.weight;
    for (const TokenSpec &token : tokens) {
        const ResultId id{source_rank, token.source_token};
        assert(combine.completed.count(id) == 1u);
        std::vector<float> expected = token.hidden;
        for (float &element : expected) element *= weight_sum;
        const ResultPacket &result = combine.completed.at(id);
        assert(result.header.origin_rank == source_rank);
        assert(result.header.source_token == token.source_token);
        assert(result.header.wave == kWave);
        AssertClose(result.payload, expected);
    }

    assert(journal->SealDispatch(handle) == IncDcInlineJournalStatus::OK);
    IncDcInlineGenerationSummaryV2 summary{};
    assert(journal->QueryGeneration(handle, &summary) ==
           IncDcInlineJournalStatus::OK);
    assert(summary.result_count == tokens.size());
    assert(summary.contribution_count == tokens.size() * topk);
    assert(summary.completed_result_count == tokens.size());
    assert(summary.arrived_contribution_count == tokens.size() * topk);
    const InlineRouteKeyV2 stale_key = second_outputs.front().route_key;
    const uint32_t stale_rank = second_outputs.front().contributor_rank;
    assert(journal->ReleaseGeneration(handle) ==
           IncDcInlineJournalStatus::OK);

    const auto final_accumulators = combine.accumulators;
    const auto final_first_result = combine.completed.at(first_id).payload;
    const ResultId second_id{source_rank, tokens[1].source_token};
    const auto final_second_result = combine.completed.at(second_id).payload;
    WorkerContribution stale{};
    stale.route_key = stale_key;
    stale.contributor_rank = stale_rank;
    stale.output = second_outputs.front().output;
    assert(combine.Accept(journal.get(), stale) ==
           IncDcInlineJournalStatus::STALE_ROUTE_KEY);
    AssertMapUnchanged(final_accumulators, combine.accumulators);
    AssertClose(final_first_result, combine.completed.at(first_id).payload);
    AssertClose(final_second_result, combine.completed.at(second_id).payload);

    std::cout << "PASS W" << worker_count << " topk" << topk
              << " fanout_payloads/token=" << unique_destinations.size()
              << '\n';
}

} // namespace

int main()
{
    for (uint32_t worker_count : {2u, 4u}) {
        for (uint32_t topk : {1u, 2u, 4u}) {
            if (topk <= worker_count) RunCase(worker_count, topk);
        }
    }
    return 0;
}
