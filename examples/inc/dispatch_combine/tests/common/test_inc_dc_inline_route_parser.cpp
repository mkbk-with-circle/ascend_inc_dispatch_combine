#include "inc_dc_inline_route_parser.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

using namespace inc::dc;

namespace {

void StoreLe16(std::vector<uint8_t> *bytes, size_t offset, uint16_t value)
{
    assert(bytes != nullptr && offset + 2u <= bytes->size());
    (*bytes)[offset] = static_cast<uint8_t>(value);
    (*bytes)[offset + 1u] = static_cast<uint8_t>(value >> 8u);
}

void StoreLe32(std::vector<uint8_t> *bytes, size_t offset, uint32_t value)
{
    assert(bytes != nullptr && offset + 4u <= bytes->size());
    for (uint32_t i = 0u; i < 4u; ++i)
        (*bytes)[offset + i] = static_cast<uint8_t>(value >> (i * 8u));
}

void StoreLe64(std::vector<uint8_t> *bytes, size_t offset, uint64_t value)
{
    assert(bytes != nullptr && offset + 8u <= bytes->size());
    for (uint32_t i = 0u; i < 8u; ++i)
        (*bytes)[offset + i] = static_cast<uint8_t>(value >> (i * 8u));
}

InlineRouteTokenParseContextV2 Context()
{
    InlineRouteTokenParseContextV2 context{};
    context.expected_source_rank = 1u;
    context.worker_count = 4u;
    context.expected_session_id = 11u;
    context.expected_placement_epoch = 13u;
    context.expected_generation = 17u;
    context.expected_request_id = 19u;
    context.expected_wave = 3u;
    context.expert_count = 32u;
    context.max_route_count = 8u;
    context.max_source_tokens = 1024u;
    context.expected_hidden_bytes = 130u;
    context.max_hidden_bytes = 4096u;
    context.max_record_bytes = 8192u;
    return context;
}

std::vector<uint8_t> Token(uint64_t source_token = 7u)
{
    constexpr uint32_t routes = 3u;
    constexpr uint64_t hidden = 130u;
    uint64_t payload = 0u;
    uint64_t record = 0u;
    assert(InlineTokenRecordLayout(routes, hidden, &payload, &record));
    std::vector<uint8_t> bytes(static_cast<size_t>(record), 0u);
    StoreLe32(&bytes, 0u, kInlineRouteTokenMagic);
    StoreLe16(&bytes, 4u, kInlineRouteVersion);
    StoreLe16(&bytes, 6u, kInlineRouteHeaderBytes);
    StoreLe32(&bytes, 8u, kInlineRouteRecordNone);
    StoreLe32(&bytes, 12u, 1u);
    StoreLe64(&bytes, 16u, 11u);
    StoreLe64(&bytes, 24u, 13u);
    StoreLe64(&bytes, 32u, 17u);
    StoreLe64(&bytes, 40u, 19u);
    StoreLe64(&bytes, 48u, source_token);
    StoreLe64(&bytes, 56u, hidden);
    StoreLe64(&bytes, 64u, payload);
    StoreLe64(&bytes, 72u, record);
    StoreLe32(&bytes, 80u, routes);
    StoreLe32(&bytes, 84u, sizeof(InlineRouteEntryV2));
    StoreLe32(&bytes, 88u, 3u);
    StoreLe64(&bytes, 96u, 17u);
    for (uint32_t route = 0u; route < routes; ++route) {
        const size_t offset = kInlineRouteHeaderBytes +
            static_cast<size_t>(route) * sizeof(InlineRouteEntryV2);
        StoreLe32(&bytes, offset, 5u + route);
        StoreLe32(&bytes, offset + 4u, route);
        StoreLe32(&bytes, offset + 8u, 0x3f800000u);
    }
    for (uint64_t i = 0u; i < hidden; ++i)
        bytes[static_cast<size_t>(payload + i)] =
            static_cast<uint8_t>((i * 17u + source_token) & 0xffu);
    return bytes;
}

InlineRouteParseStatusV2 ParseToken(const std::vector<uint8_t> &bytes,
                                    InlineRouteTokenViewV2 *view)
{
    return ParseInlineTokenRecordV2(bytes.data(), bytes.size(), Context(), view);
}

std::vector<uint8_t> Batch()
{
    const std::vector<uint8_t> first = Token(7u);
    const std::vector<uint8_t> second = Token(8u);
    uint64_t table_bytes = 0u;
    uint64_t records_offset = 0u;
    assert(InlineBatchPreambleLayout(2u, &table_bytes, &records_offset));
    uint64_t second_offset = 0u;
    uint64_t frame_bytes = 0u;
    assert(InlineStoredRecordEnd(records_offset, first.size(), &second_offset));
    assert(InlineStoredRecordEnd(second_offset, second.size(), &frame_bytes));
    std::vector<uint8_t> frame(static_cast<size_t>(frame_bytes), 0u);
    StoreLe32(&frame, 0u, kInlineRouteBatchMagic);
    StoreLe16(&frame, 4u, kInlineRouteVersion);
    StoreLe16(&frame, 6u, kInlineRouteHeaderBytes);
    StoreLe32(&frame, 8u, kInlineRouteRecordNone);
    StoreLe32(&frame, 12u, 1u);
    StoreLe64(&frame, 16u, 11u);
    StoreLe64(&frame, 24u, 13u);
    StoreLe64(&frame, 32u, 17u);
    StoreLe64(&frame, 40u, 19u);
    StoreLe32(&frame, 48u, 3u);
    StoreLe32(&frame, 52u, 2u);
    StoreLe64(&frame, 56u, table_bytes);
    StoreLe64(&frame, 64u, records_offset);
    StoreLe64(&frame, 72u, frame_bytes);
    StoreLe64(&frame, 80u, 17u);
    StoreLe64(&frame, 128u, records_offset);
    StoreLe64(&frame, 136u, second_offset);
    StoreLe64(&frame, 144u, frame_bytes);
    for (size_t i = 0u; i < first.size(); ++i)
        frame[static_cast<size_t>(records_offset) + i] = first[i];
    for (size_t i = 0u; i < second.size(); ++i)
        frame[static_cast<size_t>(second_offset) + i] = second[i];
    return frame;
}

InlineRouteBatchParseContextV2 BatchContext()
{
    InlineRouteBatchParseContextV2 context{};
    context.token = Context();
    context.max_record_count = 8u;
    context.max_frame_bytes = 65536u;
    return context;
}

} // namespace

int main()
{
    InlineRouteTokenViewV2 token_view{};
    std::vector<uint8_t> token = Token();
    assert(ParseToken(token, &token_view) == InlineRouteParseStatusV2::OK);
    assert(token_view.route_count == 3u && token_view.hidden_bytes == 130u);
    InlineRouteEntryValueV2 route{};
    assert(InlineTokenRouteAtV2(token_view, 2u, &route) ==
           InlineRouteParseStatusV2::OK);
    assert(route.expert_id == 7u && route.route_ordinal == 2u &&
           route.weight_bits == 0x3f800000u);
    assert(InlineTokenPayloadV2(token_view) != nullptr &&
           InlineTokenPayloadV2(token_view)[0] == 7u);
    // Raw decoding must not rely on the backing address satisfying the wire
    // struct's native alignment.
    std::vector<uint8_t> unaligned(token.size() + 1u, 0u);
    for (size_t i = 0u; i < token.size(); ++i) unaligned[i + 1u] = token[i];
    assert(ParseInlineTokenRecordV2(unaligned.data() + 1u, token.size(),
                                    Context(), &token_view) ==
           InlineRouteParseStatusV2::OK);

    std::vector<uint8_t> bad = token;
    StoreLe32(&bad, 0u, 0u);
    assert(ParseToken(bad, &token_view) == InlineRouteParseStatusV2::BAD_MAGIC);
    bad = token;
    StoreLe16(&bad, 4u, kInlineRouteVersion + 1u);
    assert(ParseToken(bad, &token_view) == InlineRouteParseStatusV2::BAD_VERSION);
    bad = token;
    StoreLe16(&bad, 6u, 64u);
    assert(ParseToken(bad, &token_view) == InlineRouteParseStatusV2::BAD_HEADER);
    bad = token;
    StoreLe32(&bad, 8u, 0x80000000u);
    assert(ParseToken(bad, &token_view) ==
           InlineRouteParseStatusV2::UNKNOWN_FLAGS);
    bad = token;
    StoreLe64(&bad, 96u, 0u);
    assert(ParseToken(bad, &token_view) ==
           InlineRouteParseStatusV2::NOT_COMMITTED);
    bad = token;
    bad[104u] = 1u;
    assert(ParseToken(bad, &token_view) ==
           InlineRouteParseStatusV2::NONZERO_RESERVED);

    for (size_t identity_offset : {size_t{12u}, size_t{16u}, size_t{24u},
                                   size_t{32u}, size_t{40u}, size_t{88u}}) {
        bad = token;
        bad[identity_offset] ^= 1u;
        assert(ParseToken(bad, &token_view) ==
               InlineRouteParseStatusV2::IDENTITY_MISMATCH);
    }
    bad = token;
    StoreLe64(&bad, 48u, Context().max_source_tokens);
    assert(ParseToken(bad, &token_view) ==
           InlineRouteParseStatusV2::CAPACITY_EXCEEDED);
    bad = token;
    StoreLe64(&bad, 56u, 132u);
    assert(ParseToken(bad, &token_view) ==
           InlineRouteParseStatusV2::SHAPE_MISMATCH);
    bad = token;
    StoreLe32(&bad, 80u, Context().max_route_count + 1u);
    assert(ParseToken(bad, &token_view) ==
           InlineRouteParseStatusV2::CAPACITY_EXCEEDED);
    InlineRouteTokenParseContextV2 small_record = Context();
    small_record.max_record_bytes = token.size() - 1u;
    assert(ParseInlineTokenRecordV2(token.data(), token.size(), small_record,
                                    &token_view) ==
           InlineRouteParseStatusV2::CAPACITY_EXCEEDED);
    InlineRouteTokenParseContextV2 invalid_context = Context();
    invalid_context.expected_generation = 0u;
    assert(ParseInlineTokenRecordV2(token.data(), token.size(), invalid_context,
                                    &token_view) ==
           InlineRouteParseStatusV2::INVALID_CONTEXT);
    bad = token;
    StoreLe64(&bad, 64u, 128u);
    assert(ParseToken(bad, &token_view) ==
           InlineRouteParseStatusV2::BAD_LAYOUT);
    bad = token;
    StoreLe64(&bad, 72u, token.size() + 1u);
    assert(ParseToken(bad, &token_view) ==
           InlineRouteParseStatusV2::BAD_LAYOUT);
    assert(ParseInlineTokenRecordV2(token.data(), token.size() - 1u, Context(),
                                    &token_view) ==
           InlineRouteParseStatusV2::TRUNCATED);

    bad = token;
    StoreLe32(&bad, 128u, Context().expert_count);
    assert(ParseToken(bad, &token_view) ==
           InlineRouteParseStatusV2::INVALID_EXPERT);
    bad = token;
    StoreLe32(&bad, 132u, 1u);
    assert(ParseToken(bad, &token_view) ==
           InlineRouteParseStatusV2::INVALID_ORDINAL);
    bad = token;
    StoreLe32(&bad, 136u, 0x7f800000u);
    assert(ParseToken(bad, &token_view) ==
           InlineRouteParseStatusV2::INVALID_WEIGHT);
    bad = token;
    StoreLe32(&bad, 140u, 1u);
    assert(ParseToken(bad, &token_view) ==
           InlineRouteParseStatusV2::NONZERO_RESERVED);
    bad = token;
    bad[180u] = 1u;
    assert(ParseToken(bad, &token_view) ==
           InlineRouteParseStatusV2::NONZERO_PADDING);

    bad = token;
    StoreLe32(&bad, 80u, std::numeric_limits<uint32_t>::max());
    StoreLe64(&bad, 56u, std::numeric_limits<uint64_t>::max());
    StoreLe64(&bad, 64u, 0u);
    StoreLe64(&bad, 72u, 0u);
    InlineRouteTokenParseContextV2 huge = Context();
    huge.max_route_count = std::numeric_limits<uint32_t>::max();
    huge.max_hidden_bytes = std::numeric_limits<uint64_t>::max();
    huge.expected_hidden_bytes = 0u;
    huge.max_record_bytes = std::numeric_limits<uint64_t>::max();
    assert(ParseInlineTokenRecordV2(bad.data(), bad.size(), huge,
                                    &token_view) ==
           InlineRouteParseStatusV2::ARITHMETIC_OVERFLOW);

    std::vector<uint8_t> batch = Batch();
    InlineRouteBatchViewV2 batch_view{};
    assert(ParseInlineTokenBatchV2(batch.data(), batch.size(), BatchContext(),
                                   &batch_view) ==
           InlineRouteParseStatusV2::OK);
    assert(batch_view.record_count == 2u);
    assert(InlineBatchTokenAtV2(batch_view, 1u, &token_view) ==
           InlineRouteParseStatusV2::OK);
    assert(token_view.source_token == 8u && token_view.route_count == 3u);

    bad = batch;
    StoreLe64(&bad, 80u, 0u);
    assert(ParseInlineTokenBatchV2(bad.data(), bad.size(), BatchContext(),
                                   &batch_view) ==
           InlineRouteParseStatusV2::NOT_COMMITTED);
    bad = batch;
    StoreLe32(&bad, 8u, 0x80000000u);
    assert(ParseInlineTokenBatchV2(bad.data(), bad.size(), BatchContext(),
                                   &batch_view) ==
           InlineRouteParseStatusV2::UNKNOWN_FLAGS);
    bad = batch;
    StoreLe64(&bad, 56u, 8u);
    assert(ParseInlineTokenBatchV2(bad.data(), bad.size(), BatchContext(),
                                   &batch_view) ==
           InlineRouteParseStatusV2::BAD_LAYOUT);
    bad = batch;
    StoreLe64(&bad, 128u, 193u);
    assert(ParseInlineTokenBatchV2(bad.data(), bad.size(), BatchContext(),
                                   &batch_view) ==
           InlineRouteParseStatusV2::BAD_BATCH_OFFSETS);
    bad = batch;
    StoreLe64(&bad, 136u, 192u);
    assert(ParseInlineTokenBatchV2(bad.data(), bad.size(), BatchContext(),
                                   &batch_view) ==
           InlineRouteParseStatusV2::BAD_BATCH_OFFSETS);
    bad = batch;
    StoreLe64(&bad, 144u, batch.size() + 64u);
    assert(ParseInlineTokenBatchV2(bad.data(), bad.size(), BatchContext(),
                                   &batch_view) ==
           InlineRouteParseStatusV2::BAD_BATCH_OFFSETS);
    bad = batch;
    bad[152u] = 1u;
    assert(ParseInlineTokenBatchV2(bad.data(), bad.size(), BatchContext(),
                                   &batch_view) ==
           InlineRouteParseStatusV2::NONZERO_PADDING);
    bad = batch;
    uint64_t table_bytes = 0u;
    uint64_t first_offset = 0u;
    assert(InlineBatchPreambleLayout(2u, &table_bytes, &first_offset));
    uint64_t second_offset = 0u;
    for (uint32_t byte = 0u; byte < 8u; ++byte)
        second_offset |= static_cast<uint64_t>(bad[136u + byte]) << (byte * 8u);
    // The directory is usable before later members commit. This lets the INC
    // consume token 0 without waiting for token 1.
    StoreLe64(&bad, static_cast<size_t>(second_offset) + 96u, 0u);
    assert(ParseInlineTokenBatchV2(bad.data(), bad.size(), BatchContext(),
                                   &batch_view) ==
           InlineRouteParseStatusV2::OK);
    assert(InlineBatchTokenAtV2(batch_view, 0u, &token_view) ==
           InlineRouteParseStatusV2::OK);
    assert(InlineBatchTokenAtV2(batch_view, 1u, &token_view) ==
           InlineRouteParseStatusV2::NOT_COMMITTED);
    StoreLe64(&bad, static_cast<size_t>(second_offset) + 96u, 17u);
    assert(InlineBatchTokenAtV2(batch_view, 1u, &token_view) ==
           InlineRouteParseStatusV2::OK);

    bad = batch;
    bad[static_cast<size_t>(first_offset) + 16u] ^= 1u;
    assert(ParseInlineTokenBatchV2(bad.data(), bad.size(), BatchContext(),
                                   &batch_view) ==
           InlineRouteParseStatusV2::OK);
    assert(InlineBatchTokenAtV2(batch_view, 0u, &token_view) ==
           InlineRouteParseStatusV2::IDENTITY_MISMATCH);
    bad = batch;
    uint64_t first_record_bytes = 0u;
    uint64_t unused_payload = 0u;
    assert(InlineTokenRecordLayout(3u, 130u, &unused_payload,
                                   &first_record_bytes));
    bad[static_cast<size_t>(first_offset + first_record_bytes)] = 1u;
    assert(ParseInlineTokenBatchV2(bad.data(), bad.size(), BatchContext(),
                                   &batch_view) ==
           InlineRouteParseStatusV2::OK);
    assert(InlineBatchTokenAtV2(batch_view, 0u, &token_view) ==
           InlineRouteParseStatusV2::NONZERO_PADDING);
    bad = batch;
    StoreLe32(&bad, 52u, BatchContext().max_record_count + 1u);
    assert(ParseInlineTokenBatchV2(bad.data(), bad.size(), BatchContext(),
                                   &batch_view) ==
           InlineRouteParseStatusV2::CAPACITY_EXCEEDED);
    bad = batch;
    bad[16u] ^= 1u;
    assert(ParseInlineTokenBatchV2(bad.data(), bad.size(), BatchContext(),
                                   &batch_view) ==
           InlineRouteParseStatusV2::IDENTITY_MISMATCH);
    bad = batch;
    bad[88u] = 1u;
    assert(ParseInlineTokenBatchV2(bad.data(), bad.size(), BatchContext(),
                                   &batch_view) ==
           InlineRouteParseStatusV2::NONZERO_RESERVED);
    assert(ParseInlineTokenBatchV2(batch.data(), batch.size() - 1u,
                                   BatchContext(), &batch_view) ==
           InlineRouteParseStatusV2::TRUNCATED);
    return 0;
}
