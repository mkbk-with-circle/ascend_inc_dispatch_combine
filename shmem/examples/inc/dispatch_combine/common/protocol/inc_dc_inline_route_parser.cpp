#include "inc_dc_inline_route_parser.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace inc::dc {
namespace {

bool RangeFits(size_t bytes, uint64_t offset, uint64_t width)
{
    return offset <= bytes && width <= static_cast<uint64_t>(bytes) - offset;
}

bool LoadLe16(const uint8_t *bytes, size_t size, uint64_t offset,
              uint16_t *value)
{
    if (bytes == nullptr || value == nullptr || !RangeFits(size, offset, 2u))
        return false;
    const size_t at = static_cast<size_t>(offset);
    const uint32_t decoded = static_cast<uint32_t>(bytes[at]) |
        (static_cast<uint32_t>(bytes[at + 1u]) << 8u);
    *value = static_cast<uint16_t>(decoded);
    return true;
}

bool LoadLe32(const uint8_t *bytes, size_t size, uint64_t offset,
              uint32_t *value)
{
    if (bytes == nullptr || value == nullptr || !RangeFits(size, offset, 4u))
        return false;
    const size_t at = static_cast<size_t>(offset);
    *value = static_cast<uint32_t>(bytes[at]) |
             (static_cast<uint32_t>(bytes[at + 1u]) << 8u) |
             (static_cast<uint32_t>(bytes[at + 2u]) << 16u) |
             (static_cast<uint32_t>(bytes[at + 3u]) << 24u);
    return true;
}

bool LoadLe64(const uint8_t *bytes, size_t size, uint64_t offset,
              uint64_t *value)
{
    if (bytes == nullptr || value == nullptr || !RangeFits(size, offset, 8u))
        return false;
    const size_t at = static_cast<size_t>(offset);
    uint64_t result = 0u;
    for (uint32_t i = 0u; i < 8u; ++i)
        result |= static_cast<uint64_t>(bytes[at + i]) << (i * 8u);
    *value = result;
    return true;
}

bool AllZero(const uint8_t *bytes, size_t size, uint64_t begin, uint64_t end)
{
    if (bytes == nullptr || begin > end || !RangeFits(size, begin, end - begin))
        return false;
    for (uint64_t offset = begin; offset < end; ++offset)
        if (bytes[static_cast<size_t>(offset)] != 0u) return false;
    return true;
}

bool TokenContextValid(const InlineRouteTokenParseContextV2 &context)
{
    return context.worker_count != 0u &&
           context.expected_source_rank < context.worker_count &&
           context.expected_session_id != 0u &&
           context.expected_placement_epoch != 0u &&
           context.expected_generation != 0u &&
           context.expected_request_id != 0u && context.expert_count != 0u &&
           context.max_route_count != 0u &&
           context.max_source_tokens != 0u &&
           context.max_hidden_bytes != 0u &&
           context.max_record_bytes >= kInlineRouteHeaderBytes &&
           (context.expected_hidden_bytes == 0u ||
            context.expected_hidden_bytes <= context.max_hidden_bytes);
}

bool IdentityMatches(uint32_t source_rank, uint64_t session_id,
                     uint64_t placement_epoch, uint64_t generation,
                     uint64_t request_id, uint32_t wave,
                     const InlineRouteTokenParseContextV2 &context)
{
    return source_rank == context.expected_source_rank &&
           session_id == context.expected_session_id &&
           placement_epoch == context.expected_placement_epoch &&
           generation == context.expected_generation &&
           request_id == context.expected_request_id &&
           wave == context.expected_wave;
}

InlineRouteParseStatusV2 DecodeTokenHeader(
    const uint8_t *record, size_t available_bytes,
    const InlineRouteTokenParseContextV2 &context,
    InlineRouteTokenViewV2 *decoded)
{
    if (available_bytes < kInlineRouteHeaderBytes)
        return InlineRouteParseStatusV2::TRUNCATED;

    // The transport publishes this field last.  Read it before any mutable
    // metadata, then establish the acquire side of the publication contract.
    // The adapter must provide a stable byte snapshot (or an atomic aligned
    // commit load) before entering this raw parser.
    uint64_t commit = 0u;
    if (!LoadLe64(record, available_bytes, 96u, &commit))
        return InlineRouteParseStatusV2::TRUNCATED;
    if (commit != context.expected_generation)
        return InlineRouteParseStatusV2::NOT_COMMITTED;
    std::atomic_thread_fence(std::memory_order_acquire);

    uint32_t magic = 0u;
    uint16_t version = 0u;
    uint16_t header_bytes = 0u;
    uint32_t route_entry_bytes = 0u;
    uint32_t reserved32 = 0u;
    if (!LoadLe32(record, available_bytes, 0u, &magic) ||
        !LoadLe16(record, available_bytes, 4u, &version) ||
        !LoadLe16(record, available_bytes, 6u, &header_bytes) ||
        !LoadLe32(record, available_bytes, 8u, &decoded->flags) ||
        !LoadLe32(record, available_bytes, 12u, &decoded->source_rank) ||
        !LoadLe64(record, available_bytes, 16u, &decoded->session_id) ||
        !LoadLe64(record, available_bytes, 24u,
                  &decoded->placement_epoch) ||
        !LoadLe64(record, available_bytes, 32u, &decoded->generation) ||
        !LoadLe64(record, available_bytes, 40u, &decoded->request_id) ||
        !LoadLe64(record, available_bytes, 48u, &decoded->source_token) ||
        !LoadLe64(record, available_bytes, 56u, &decoded->hidden_bytes) ||
        !LoadLe64(record, available_bytes, 64u, &decoded->payload_offset) ||
        !LoadLe64(record, available_bytes, 72u, &decoded->record_bytes) ||
        !LoadLe32(record, available_bytes, 80u, &decoded->route_count) ||
        !LoadLe32(record, available_bytes, 84u, &route_entry_bytes) ||
        !LoadLe32(record, available_bytes, 88u, &decoded->wave) ||
        !LoadLe32(record, available_bytes, 92u, &reserved32)) {
        return InlineRouteParseStatusV2::TRUNCATED;
    }
    if (magic != kInlineRouteTokenMagic)
        return InlineRouteParseStatusV2::BAD_MAGIC;
    if (version != kInlineRouteVersion)
        return InlineRouteParseStatusV2::BAD_VERSION;
    if (header_bytes != kInlineRouteHeaderBytes ||
        route_entry_bytes != sizeof(InlineRouteEntryV2))
        return InlineRouteParseStatusV2::BAD_HEADER;
    if ((decoded->flags & ~kInlineRouteRecordAllowedFlags) != 0u)
        return InlineRouteParseStatusV2::UNKNOWN_FLAGS;
    if (reserved32 != 0u ||
        !AllZero(record, available_bytes, 104u, kInlineRouteHeaderBytes))
        return InlineRouteParseStatusV2::NONZERO_RESERVED;
    if (!IdentityMatches(decoded->source_rank, decoded->session_id,
                         decoded->placement_epoch, decoded->generation,
                         decoded->request_id, decoded->wave, context))
        return InlineRouteParseStatusV2::IDENTITY_MISMATCH;
    return InlineRouteParseStatusV2::OK;
}

InlineRouteParseStatusV2 LoadBatchOffset(const InlineRouteBatchViewV2 &view,
                                         uint32_t index, uint64_t *offset)
{
    uint64_t byte_offset = 0u;
    if (!InlineCheckedMul(index, sizeof(uint64_t), &byte_offset) ||
        !InlineCheckedAdd(kInlineRouteHeaderBytes, byte_offset, &byte_offset))
        return InlineRouteParseStatusV2::ARITHMETIC_OVERFLOW;
    return LoadLe64(view.frame, static_cast<size_t>(view.frame_bytes),
                    byte_offset, offset)
               ? InlineRouteParseStatusV2::OK
               : InlineRouteParseStatusV2::TRUNCATED;
}

} // namespace

InlineRouteParseStatusV2 ParseInlineTokenRecordV2(
    const uint8_t *record, size_t available_bytes,
    const InlineRouteTokenParseContextV2 &context,
    InlineRouteTokenViewV2 *view)
{
    if (record == nullptr || view == nullptr)
        return InlineRouteParseStatusV2::INVALID_ARGUMENT;
    *view = {};
    if (!TokenContextValid(context))
        return InlineRouteParseStatusV2::INVALID_CONTEXT;

    InlineRouteTokenViewV2 decoded{};
    InlineRouteParseStatusV2 status =
        DecodeTokenHeader(record, available_bytes, context, &decoded);
    if (status != InlineRouteParseStatusV2::OK) return status;
    if (decoded.source_token >= context.max_source_tokens ||
        decoded.route_count > context.max_route_count ||
        decoded.hidden_bytes > context.max_hidden_bytes ||
        decoded.record_bytes > context.max_record_bytes)
        return InlineRouteParseStatusV2::CAPACITY_EXCEEDED;
    if (context.expected_hidden_bytes != 0u &&
        decoded.hidden_bytes != context.expected_hidden_bytes)
        return InlineRouteParseStatusV2::SHAPE_MISMATCH;

    uint64_t expected_payload = 0u;
    uint64_t expected_record = 0u;
    if (!InlineTokenRecordLayout(decoded.route_count, decoded.hidden_bytes,
                                 &expected_payload, &expected_record))
        return InlineRouteParseStatusV2::ARITHMETIC_OVERFLOW;
    if (decoded.payload_offset != expected_payload ||
        decoded.record_bytes != expected_record)
        return InlineRouteParseStatusV2::BAD_LAYOUT;
    if (decoded.record_bytes > available_bytes)
        return InlineRouteParseStatusV2::TRUNCATED;

    uint64_t route_bytes = 0u;
    uint64_t route_end = 0u;
    if (!InlineCheckedMul(decoded.route_count, sizeof(InlineRouteEntryV2),
                          &route_bytes) ||
        !InlineCheckedAdd(kInlineRouteHeaderBytes, route_bytes, &route_end))
        return InlineRouteParseStatusV2::ARITHMETIC_OVERFLOW;
    if (route_end > decoded.payload_offset ||
        !RangeFits(available_bytes, kInlineRouteHeaderBytes, route_bytes))
        return InlineRouteParseStatusV2::BAD_LAYOUT;

    for (uint32_t route_index = 0u; route_index < decoded.route_count;
         ++route_index) {
        uint64_t route_offset = 0u;
        uint64_t relative = 0u;
        if (!InlineCheckedMul(route_index, sizeof(InlineRouteEntryV2),
                              &relative) ||
            !InlineCheckedAdd(kInlineRouteHeaderBytes, relative,
                              &route_offset))
            return InlineRouteParseStatusV2::ARITHMETIC_OVERFLOW;
        uint32_t expert_id = 0u;
        uint32_t ordinal = 0u;
        uint32_t weight_bits = 0u;
        uint32_t reserved = 0u;
        if (!LoadLe32(record, available_bytes, route_offset, &expert_id) ||
            !LoadLe32(record, available_bytes, route_offset + 4u, &ordinal) ||
            !LoadLe32(record, available_bytes, route_offset + 8u,
                      &weight_bits) ||
            !LoadLe32(record, available_bytes, route_offset + 12u,
                      &reserved))
            return InlineRouteParseStatusV2::TRUNCATED;
        if (reserved != 0u)
            return InlineRouteParseStatusV2::NONZERO_RESERVED;
        if (expert_id >= context.expert_count)
            return InlineRouteParseStatusV2::INVALID_EXPERT;
        if (ordinal != route_index)
            return InlineRouteParseStatusV2::INVALID_ORDINAL;
        if ((weight_bits & 0x7f800000u) == 0x7f800000u)
            return InlineRouteParseStatusV2::INVALID_WEIGHT;
    }
    if (!AllZero(record, available_bytes, route_end,
                 decoded.payload_offset))
        return InlineRouteParseStatusV2::NONZERO_PADDING;

    decoded.record = record;
    *view = decoded;
    return InlineRouteParseStatusV2::OK;
}

InlineRouteParseStatusV2 InlineTokenRouteAtV2(
    const InlineRouteTokenViewV2 &view, uint32_t route_index,
    InlineRouteEntryValueV2 *route)
{
    if (view.record == nullptr || route == nullptr)
        return InlineRouteParseStatusV2::INVALID_ARGUMENT;
    *route = {};
    if (route_index >= view.route_count)
        return InlineRouteParseStatusV2::INVALID_ARGUMENT;
    uint64_t relative = 0u;
    uint64_t route_offset = 0u;
    if (!InlineCheckedMul(route_index, sizeof(InlineRouteEntryV2), &relative) ||
        !InlineCheckedAdd(kInlineRouteHeaderBytes, relative, &route_offset))
        return InlineRouteParseStatusV2::ARITHMETIC_OVERFLOW;
    if (view.record_bytes > std::numeric_limits<size_t>::max())
        return InlineRouteParseStatusV2::CAPACITY_EXCEEDED;
    const size_t record_size = static_cast<size_t>(view.record_bytes);
    if (!LoadLe32(view.record, record_size, route_offset, &route->expert_id) ||
        !LoadLe32(view.record, record_size, route_offset + 4u,
                  &route->route_ordinal) ||
        !LoadLe32(view.record, record_size, route_offset + 8u,
                  &route->weight_bits))
        return InlineRouteParseStatusV2::TRUNCATED;
    return InlineRouteParseStatusV2::OK;
}

const uint8_t *InlineTokenPayloadV2(const InlineRouteTokenViewV2 &view)
{
    if (view.record == nullptr || view.payload_offset > view.record_bytes ||
        view.payload_offset > std::numeric_limits<size_t>::max())
        return nullptr;
    return view.record + static_cast<size_t>(view.payload_offset);
}

InlineRouteParseStatusV2 ParseInlineTokenBatchV2(
    const uint8_t *frame, size_t available_bytes,
    const InlineRouteBatchParseContextV2 &context,
    InlineRouteBatchViewV2 *view)
{
    if (frame == nullptr || view == nullptr)
        return InlineRouteParseStatusV2::INVALID_ARGUMENT;
    *view = {};
    if (!TokenContextValid(context.token) || context.max_record_count == 0u ||
        context.max_frame_bytes < kInlineRouteHeaderBytes)
        return InlineRouteParseStatusV2::INVALID_CONTEXT;
    if (available_bytes < kInlineRouteHeaderBytes)
        return InlineRouteParseStatusV2::TRUNCATED;

    // A batch commit publishes only its immutable directory. Each member token
    // is still independently commit-gated by ParseInlineTokenRecordV2.
    uint64_t directory_commit = 0u;
    if (!LoadLe64(frame, available_bytes, 80u, &directory_commit))
        return InlineRouteParseStatusV2::TRUNCATED;
    if (directory_commit != context.token.expected_generation)
        return InlineRouteParseStatusV2::NOT_COMMITTED;
    std::atomic_thread_fence(std::memory_order_acquire);

    uint32_t magic = 0u;
    uint16_t version = 0u;
    uint16_t header_bytes = 0u;
    uint32_t flags = 0u;
    uint32_t source_rank = 0u;
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t request_id = 0u;
    uint32_t wave = 0u;
    uint32_t record_count = 0u;
    uint64_t offset_table_bytes = 0u;
    uint64_t records_offset = 0u;
    uint64_t frame_bytes = 0u;
    if (!LoadLe32(frame, available_bytes, 0u, &magic) ||
        !LoadLe16(frame, available_bytes, 4u, &version) ||
        !LoadLe16(frame, available_bytes, 6u, &header_bytes) ||
        !LoadLe32(frame, available_bytes, 8u, &flags) ||
        !LoadLe32(frame, available_bytes, 12u, &source_rank) ||
        !LoadLe64(frame, available_bytes, 16u, &session_id) ||
        !LoadLe64(frame, available_bytes, 24u, &placement_epoch) ||
        !LoadLe64(frame, available_bytes, 32u, &generation) ||
        !LoadLe64(frame, available_bytes, 40u, &request_id) ||
        !LoadLe32(frame, available_bytes, 48u, &wave) ||
        !LoadLe32(frame, available_bytes, 52u, &record_count) ||
        !LoadLe64(frame, available_bytes, 56u, &offset_table_bytes) ||
        !LoadLe64(frame, available_bytes, 64u, &records_offset) ||
        !LoadLe64(frame, available_bytes, 72u, &frame_bytes))
        return InlineRouteParseStatusV2::TRUNCATED;
    if (magic != kInlineRouteBatchMagic)
        return InlineRouteParseStatusV2::BAD_MAGIC;
    if (version != kInlineRouteVersion)
        return InlineRouteParseStatusV2::BAD_VERSION;
    if (header_bytes != kInlineRouteHeaderBytes)
        return InlineRouteParseStatusV2::BAD_HEADER;
    if ((flags & ~kInlineRouteRecordAllowedFlags) != 0u)
        return InlineRouteParseStatusV2::UNKNOWN_FLAGS;
    if (!AllZero(frame, available_bytes, 88u, kInlineRouteHeaderBytes))
        return InlineRouteParseStatusV2::NONZERO_RESERVED;
    if (!IdentityMatches(source_rank, session_id, placement_epoch, generation,
                         request_id, wave, context.token))
        return InlineRouteParseStatusV2::IDENTITY_MISMATCH;
    if (record_count > context.max_record_count ||
        frame_bytes > context.max_frame_bytes)
        return InlineRouteParseStatusV2::CAPACITY_EXCEEDED;

    uint64_t expected_table_bytes = 0u;
    uint64_t expected_records_offset = 0u;
    if (!InlineBatchPreambleLayout(record_count, &expected_table_bytes,
                                   &expected_records_offset))
        return InlineRouteParseStatusV2::ARITHMETIC_OVERFLOW;
    if (offset_table_bytes != expected_table_bytes ||
        records_offset != expected_records_offset ||
        records_offset > frame_bytes ||
        (records_offset & (kInlineRouteWireAlignment - 1u)) != 0u)
        return InlineRouteParseStatusV2::BAD_LAYOUT;
    if (frame_bytes > available_bytes)
        return InlineRouteParseStatusV2::TRUNCATED;

    uint64_t table_end = 0u;
    if (!InlineCheckedAdd(kInlineRouteHeaderBytes, offset_table_bytes,
                          &table_end))
        return InlineRouteParseStatusV2::ARITHMETIC_OVERFLOW;
    if (table_end > records_offset ||
        !AllZero(frame, available_bytes, table_end, records_offset))
        return InlineRouteParseStatusV2::NONZERO_PADDING;

    InlineRouteBatchViewV2 decoded{};
    decoded.frame = frame;
    decoded.frame_bytes = frame_bytes;
    decoded.records_offset = records_offset;
    decoded.record_count = record_count;
    decoded.flags = flags;
    decoded.token_context = context.token;

    uint64_t current = 0u;
    InlineRouteParseStatusV2 status = LoadBatchOffset(decoded, 0u, &current);
    if (status != InlineRouteParseStatusV2::OK) return status;
    if (current != records_offset)
        return InlineRouteParseStatusV2::BAD_BATCH_OFFSETS;
    for (uint32_t index = 0u; index < record_count; ++index) {
        uint64_t next = 0u;
        status = LoadBatchOffset(decoded, index + 1u, &next);
        if (status != InlineRouteParseStatusV2::OK) return status;
        if (current >= next || next > frame_bytes ||
            (current & (kInlineRouteWireAlignment - 1u)) != 0u ||
            (next & (kInlineRouteWireAlignment - 1u)) != 0u)
            return InlineRouteParseStatusV2::BAD_BATCH_OFFSETS;
        current = next;
    }
    if (current != frame_bytes)
        return InlineRouteParseStatusV2::BAD_BATCH_OFFSETS;
    *view = decoded;
    return InlineRouteParseStatusV2::OK;
}

InlineRouteParseStatusV2 InlineBatchTokenAtV2(
    const InlineRouteBatchViewV2 &view, uint32_t record_index,
    InlineRouteTokenViewV2 *token)
{
    if (view.frame == nullptr || token == nullptr ||
        record_index >= view.record_count)
        return InlineRouteParseStatusV2::INVALID_ARGUMENT;
    uint64_t begin = 0u;
    uint64_t end = 0u;
    InlineRouteParseStatusV2 status =
        LoadBatchOffset(view, record_index, &begin);
    if (status != InlineRouteParseStatusV2::OK) return status;
    status = LoadBatchOffset(view, record_index + 1u, &end);
    if (status != InlineRouteParseStatusV2::OK) return status;
    if (begin >= end || end > view.frame_bytes ||
        end - begin > std::numeric_limits<size_t>::max())
        return InlineRouteParseStatusV2::BAD_BATCH_OFFSETS;
    InlineRouteParseStatusV2 parse_status = ParseInlineTokenRecordV2(
        view.frame + static_cast<size_t>(begin),
        static_cast<size_t>(end - begin), view.token_context, token);
    if (parse_status != InlineRouteParseStatusV2::OK) return parse_status;
    uint64_t logical_end = 0u;
    if (!InlineCheckedAdd(begin, token->record_bytes, &logical_end))
        return InlineRouteParseStatusV2::ARITHMETIC_OVERFLOW;
    if (logical_end > end)
        return InlineRouteParseStatusV2::BAD_BATCH_OFFSETS;
    return AllZero(view.frame, static_cast<size_t>(view.frame_bytes),
                   logical_end, end)
               ? InlineRouteParseStatusV2::OK
               : InlineRouteParseStatusV2::NONZERO_PADDING;
}

const char *InlineRouteParseStatusStringV2(InlineRouteParseStatusV2 status)
{
    switch (status) {
        case InlineRouteParseStatusV2::OK: return "OK";
        case InlineRouteParseStatusV2::INVALID_ARGUMENT:
            return "INVALID_ARGUMENT";
        case InlineRouteParseStatusV2::INVALID_CONTEXT:
            return "INVALID_CONTEXT";
        case InlineRouteParseStatusV2::TRUNCATED: return "TRUNCATED";
        case InlineRouteParseStatusV2::BAD_MAGIC: return "BAD_MAGIC";
        case InlineRouteParseStatusV2::BAD_VERSION: return "BAD_VERSION";
        case InlineRouteParseStatusV2::BAD_HEADER: return "BAD_HEADER";
        case InlineRouteParseStatusV2::UNKNOWN_FLAGS: return "UNKNOWN_FLAGS";
        case InlineRouteParseStatusV2::NOT_COMMITTED: return "NOT_COMMITTED";
        case InlineRouteParseStatusV2::NONZERO_RESERVED:
            return "NONZERO_RESERVED";
        case InlineRouteParseStatusV2::IDENTITY_MISMATCH:
            return "IDENTITY_MISMATCH";
        case InlineRouteParseStatusV2::SHAPE_MISMATCH:
            return "SHAPE_MISMATCH";
        case InlineRouteParseStatusV2::CAPACITY_EXCEEDED:
            return "CAPACITY_EXCEEDED";
        case InlineRouteParseStatusV2::ARITHMETIC_OVERFLOW:
            return "ARITHMETIC_OVERFLOW";
        case InlineRouteParseStatusV2::BAD_LAYOUT: return "BAD_LAYOUT";
        case InlineRouteParseStatusV2::NONZERO_PADDING:
            return "NONZERO_PADDING";
        case InlineRouteParseStatusV2::INVALID_EXPERT:
            return "INVALID_EXPERT";
        case InlineRouteParseStatusV2::INVALID_ORDINAL:
            return "INVALID_ORDINAL";
        case InlineRouteParseStatusV2::INVALID_WEIGHT:
            return "INVALID_WEIGHT";
        case InlineRouteParseStatusV2::BAD_BATCH_OFFSETS:
            return "BAD_BATCH_OFFSETS";
    }
    return "UNKNOWN";
}

} // namespace inc::dc
