#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

#include "inc_dc_dynamic_journal.h"

using namespace inc::dc::pull_combine;

namespace {

EndpointDispatchInput MakeInput(uint32_t source, uint32_t workers,
                                uint32_t hidden, uint32_t tokens,
                                std::mt19937_64 *rng)
{
    EndpointDispatchInput input{};
    input.config.worker_count = workers;
    input.config.expert_count = 64u;
    input.config.hidden = hidden;
    input.config.dtype = EndpointDataType::BF16;
    input.config.wave = 7u;
    input.config.source_rank = source;
    input.config.generation = 19u;
    input.config.sequence = 1u;
    input.assignment_offsets.push_back(0u);
    for (uint32_t token = 0u; token < tokens; ++token) {
        input.token_ids.push_back(
            (static_cast<uint64_t>(source) << 48u) |
            (static_cast<uint64_t>(token) + 1u));
        const uint32_t topk = (*rng)() % (workers * 2u + 1u);
        for (uint32_t k = 0u; k < topk; ++k) {
            EndpointDispatchAssignmentRecord assignment{};
            assignment.destination_rank = (*rng)() % workers;
            assignment.expert_id = (*rng)() % 64u;
            assignment.ordinal = k;
            assignment.weight = static_cast<float>(k + 1u) /
                static_cast<float>(topk + 1u);
            input.assignments.push_back(assignment);
        }
        input.assignment_offsets.push_back(input.assignments.size());
    }
    input.hidden_payload.assign(
        static_cast<size_t>(tokens) * hidden * 2u, 0x5au);
    return input;
}

ParsedEndpointDispatch Parse(const EndpointDispatchInput &input)
{
    std::vector<uint8_t> packet;
    EndpointDispatchCommit commit{};
    std::string error;
    assert(BuildEndpointDispatchPacket(input, &packet, &commit, &error) ==
           EndpointDispatchStatus::OK);
    ParsedEndpointDispatch parsed;
    assert(ParseEndpointDispatchPacket(packet.data(), packet.size(),
                                       input.config, &parsed, &error) ==
           EndpointDispatchStatus::OK);
    return parsed;
}

void RunRandomWave(uint32_t workers, uint32_t hidden, uint32_t tokens,
                   std::mt19937_64 *rng)
{
    DynamicWaveJournal journal;
    assert(journal.Initialize({workers, hidden, 7u, 19u}) ==
           DynamicJournalStatus::OK);
    std::vector<ParsedEndpointDispatch> dispatches;
    std::unordered_map<uint64_t, std::vector<uint32_t>> contributors;
    std::unordered_map<uint64_t, uint32_t> owners;
    for (uint32_t source = 0u; source < workers; ++source) {
        dispatches.push_back(Parse(MakeInput(source, workers, hidden, tokens,
                                             rng)));
        for (const EndpointDispatchTokenRecord &token :
             dispatches.back().source_tokens)
            owners[token.token_id] = source;
        for (uint32_t destination = 0u; destination < workers;
             ++destination) {
            for (const EndpointFanoutRow &row :
                 dispatches.back().destination_rows[destination]) {
                contributors[row.token_id].push_back(destination);
                owners[row.token_id] = source;
            }
        }
    }
    std::shuffle(dispatches.begin(), dispatches.end(), *rng);
    for (const auto &dispatch : dispatches)
        assert(journal.AddDispatchSource(dispatch) ==
               DynamicJournalStatus::OK);
    assert(journal.SealDispatch() == DynamicJournalStatus::OK);

    std::vector<std::vector<uint64_t>> by_source(workers);
    for (const auto &[token_id, ranks] : contributors) {
        for (uint32_t rank : ranks) by_source[rank].push_back(token_id);
    }
    std::vector<uint64_t> sequence(workers, 1u);
    std::vector<std::pair<uint32_t, std::vector<uint64_t>>> chunks;
    for (uint32_t source = 0u; source < workers; ++source) {
        std::shuffle(by_source[source].begin(), by_source[source].end(),
                     *rng);
        size_t begin = 0u;
        while (begin < by_source[source].size()) {
            const size_t count = std::min<size_t>(
                1u + (*rng)() % 5u, by_source[source].size() - begin);
            chunks.push_back({source, std::vector<uint64_t>(
                by_source[source].begin() + begin,
                by_source[source].begin() + begin + count)});
            begin += count;
        }
    }
    std::shuffle(chunks.begin(), chunks.end(), *rng);
    // Sequence is per source, so arbitrary inter-source readiness is allowed;
    // preserve only each source's local descriptor order.
    std::stable_sort(chunks.begin(), chunks.end(),
                     [](const auto &a, const auto &b) {
                         return a.first < b.first;
                     });
    // Interleave sources again without changing their individual queues.
    std::vector<std::deque<std::vector<uint64_t>>> queues(workers);
    for (auto &chunk : chunks) queues[chunk.first].push_back(std::move(chunk.second));
    uint64_t remaining = 0u;
    for (const auto &queue : queues) remaining += queue.size();
    while (remaining != 0u) {
        uint32_t source = (*rng)() % workers;
        if (queues[source].empty()) continue;
        SparseCombineBatch batch{};
        batch.source_rank = source;
        batch.sequence = sequence[source]++;
        batch.token_ids = std::move(queues[source].front());
        queues[source].pop_front();
        --remaining;
        batch.values.resize(batch.token_ids.size() * hidden);
        for (uint32_t row = 0u; row < batch.token_ids.size(); ++row) {
            const float value = static_cast<float>(source + 1u);
            std::fill_n(batch.values.data() +
                            static_cast<uint64_t>(row) * hidden,
                        hidden, value);
        }
        SparseCombineAck ack{};
        assert(journal.ConsumeCombine(batch, &ack) ==
               DynamicJournalStatus::OK);
        assert(ack.rows_consumed == batch.token_ids.size());
    }

    uint64_t egress_rows = 0u;
    for (uint32_t owner = 0u; owner < workers; ++owner) {
        for (;;) {
            DynamicEgressBatch out;
            const auto status = journal.PopEgress(owner, 3u, &out);
            if (status == DynamicJournalStatus::NOT_READY) break;
            assert(status == DynamicJournalStatus::OK);
            for (uint32_t row = 0u; row < out.token_ids.size(); ++row) {
                assert(owners[out.token_ids[row]] == owner);
                float expected = 0.0f;
                for (uint32_t contributor : contributors[out.token_ids[row]])
                    expected += static_cast<float>(contributor + 1u);
                for (uint32_t element = 0u; element < hidden; ++element)
                    assert(out.values[static_cast<uint64_t>(row) * hidden +
                                      element] == expected);
                ++egress_rows;
            }
        }
    }
    assert(egress_rows == journal.token_count());
    assert(journal.combine_complete());
}

void NegativeCases()
{
    std::mt19937_64 rng(1u);
    DynamicWaveJournal journal;
    assert(journal.Initialize({2u, 4u, 7u, 19u}) ==
           DynamicJournalStatus::OK);
    auto d0 = Parse(MakeInput(0u, 2u, 4u, 3u, &rng));
    auto d1 = Parse(MakeInput(1u, 2u, 4u, 3u, &rng));
    assert(journal.AddDispatchSource(d0) == DynamicJournalStatus::OK);
    assert(journal.AddDispatchSource(d0) ==
           DynamicJournalStatus::SOURCE_ALREADY_PUBLISHED);
    assert(journal.SealDispatch() ==
           DynamicJournalStatus::DISPATCH_NOT_SEALED);
    assert(journal.AddDispatchSource(d1) == DynamicJournalStatus::OK);
    assert(journal.SealDispatch() == DynamicJournalStatus::OK);

    SparseCombineAck ack{};
    SparseCombineBatch unknown{0u, 1u, {999999u},
                               std::vector<float>(4u, 1.0f)};
    assert(journal.ConsumeCombine(unknown, &ack) ==
           DynamicJournalStatus::UNKNOWN_TOKEN_ID);
    SparseCombineBatch bad_size{0u, 1u, {d0.destination_rows[0][0].token_id},
                                std::vector<float>(3u, 1.0f)};
    assert(journal.ConsumeCombine(bad_size, &ack) ==
           DynamicJournalStatus::PAYLOAD_SIZE_MISMATCH);
}

} // namespace

int main()
{
    NegativeCases();
    std::mt19937_64 rng(0x71a39d55u);
    for (uint32_t iteration = 0u; iteration < 500u; ++iteration) {
        const uint32_t workers = 2u + rng() % 7u;
        const uint32_t hidden = 1u + rng() % 17u;
        const uint32_t tokens = 1u + rng() % 13u;
        RunRandomWave(workers, hidden, tokens, &rng);
    }
    std::cout << "[PASS] dynamic endpoint journal: 500 randomized waves\n";
    return 0;
}
