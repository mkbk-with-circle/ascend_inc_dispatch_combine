#include "inc_dc_inline_route_protocol.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

using namespace inc::dc;

int main()
{
    InlineTokenRecordHeaderV2 token{};
    assert(token.magic == kInlineRouteTokenMagic);
    assert(token.version == kInlineRouteVersion);
    assert(token.header_bytes == sizeof(token));
    assert(offsetof(InlineTokenRecordHeaderV2, session_id) == 16u);
    assert(offsetof(InlineTokenRecordHeaderV2, payload_offset) == 64u);

    uint64_t payload = 0u;
    uint64_t bytes = 0u;
    assert(InlineTokenRecordLayout(8u, 16384u, &payload, &bytes));
    assert(payload == 256u);
    assert(bytes == 256u + 16384u);

    // No historical top-k=16, 32-bit destination-mask or 4GiB payload limit.
    assert(InlineTokenRecordLayout(
        257u, 8ull * 1024ull * 1024ull * 1024ull, &payload, &bytes));
    assert(payload % kInlineRouteWireAlignment == 0u);
    assert(bytes > 8ull * 1024ull * 1024ull * 1024ull);
    assert(!InlineTokenRecordLayout(
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint64_t>::max(), &payload, &bytes));

    assert(InlineFanoutRecordLayout(8u, 16384u, &payload, &bytes));
    assert(payload == 512u);
    assert(bytes == 512u + 16384u);

    uint64_t table_bytes = 0u;
    uint64_t records_offset = 0u;
    assert(InlineBatchPreambleLayout(7u, &table_bytes, &records_offset));
    assert(table_bytes == 8u * sizeof(uint64_t));
    assert(records_offset == 192u);
    assert(records_offset % kInlineRouteWireAlignment == 0u);

    uint64_t next = 0u;
    assert(InlineStoredRecordEnd(records_offset, 16517u, &next));
    assert(next % kInlineRouteWireAlignment == 0u);
    assert(next >= records_offset + 16517u);

    assert(InlinePayloadRecordLayout(
        8ull * 1024ull * 1024ull * 1024ull, &bytes));
    assert(bytes == kInlineRouteHeaderBytes +
                        8ull * 1024ull * 1024ull * 1024ull);
    assert(!InlinePayloadRecordLayout(
        std::numeric_limits<uint64_t>::max(), &bytes));

    InlineCombineRecordHeaderV2 contribution{};
    contribution.session_id = 5u;
    contribution.placement_epoch = 7u;
    contribution.generation = 9u;
    contribution.request_id = 17u;
    contribution.route_key.journal_locator = 42u;
    contribution.route_key.authenticator = 0x1234u;
    contribution.payload_bytes = 16384u;
    assert(InlinePayloadRecordLayout(contribution.payload_bytes,
                                     &contribution.record_bytes));
    assert(contribution.route_key.journal_locator == 42u);
    assert(contribution.record_bytes == 128u + 16384u);

    InlineRouteEntryV2 route{};
    route.expert_id = 11u;
    route.route_ordinal = 2u;
    route.weight_bits = 0x3f800000u;
    assert(route.reserved32 == 0u);
    return 0;
}
