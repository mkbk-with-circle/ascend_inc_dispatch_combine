#ifndef INC_DC_INLINE_ROUTE_PARSER_H
#define INC_DC_INLINE_ROUTE_PARSER_H

#include <cstddef>
#include <cstdint>

#include "inc_dc_inline_route_protocol.h"

namespace inc::dc {

enum class InlineRouteParseStatusV2 : uint32_t {
    OK = 0u,
    INVALID_ARGUMENT,
    INVALID_CONTEXT,
    TRUNCATED,
    BAD_MAGIC,
    BAD_VERSION,
    BAD_HEADER,
    UNKNOWN_FLAGS,
    NOT_COMMITTED,
    NONZERO_RESERVED,
    IDENTITY_MISMATCH,
    SHAPE_MISMATCH,
    CAPACITY_EXCEEDED,
    ARITHMETIC_OVERFLOW,
    BAD_LAYOUT,
    NONZERO_PADDING,
    INVALID_EXPERT,
    INVALID_ORDINAL,
    INVALID_WEIGHT,
    BAD_BATCH_OFFSETS,
};

// Every value is supplied by the already-authenticated ingress/session
// context.  The parser treats the wire record only as a claim and requires an
// exact match before exposing any route or payload bytes to the INC data path.
struct InlineRouteTokenParseContextV2 {
    uint32_t expected_source_rank = 0u;
    uint32_t worker_count = 0u;
    uint64_t expected_session_id = 0u;
    uint64_t expected_placement_epoch = 0u;
    uint64_t expected_generation = 0u;
    uint64_t expected_request_id = 0u;
    uint32_t expected_wave = 0u;
    uint32_t expert_count = 0u;
    uint32_t max_route_count = 0u;
    uint64_t max_source_tokens = 0u;
    uint64_t expected_hidden_bytes = 0u; // zero accepts any value within max
    uint64_t max_hidden_bytes = 0u;
    uint64_t max_record_bytes = 0u;
};

struct InlineRouteEntryValueV2 {
    uint32_t expert_id = 0u;
    uint32_t route_ordinal = 0u;
    uint32_t weight_bits = 0u;
};

// A non-owning view. The caller keeps the backing byte buffer alive and
// immutable.
// All offsets were checked against both the wire record and the supplied
// capacity before this view is returned.
struct InlineRouteTokenViewV2 {
    const uint8_t *record = nullptr;
    uint64_t record_bytes = 0u;
    uint64_t payload_offset = 0u;
    uint64_t hidden_bytes = 0u;
    uint64_t source_token = 0u;
    uint32_t route_count = 0u;
    uint32_t source_rank = 0u;
    uint32_t wave = 0u;
    uint32_t flags = 0u;
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t request_id = 0u;
};

InlineRouteParseStatusV2 ParseInlineTokenRecordV2(
    const uint8_t *record, size_t available_bytes,
    const InlineRouteTokenParseContextV2 &context,
    InlineRouteTokenViewV2 *view);

InlineRouteParseStatusV2 InlineTokenRouteAtV2(
    const InlineRouteTokenViewV2 &view, uint32_t route_index,
    InlineRouteEntryValueV2 *route);

const uint8_t *InlineTokenPayloadV2(const InlineRouteTokenViewV2 &view);

struct InlineRouteBatchParseContextV2 {
    InlineRouteTokenParseContextV2 token{};
    uint32_t max_record_count = 0u;
    uint64_t max_frame_bytes = 0u;
};

struct InlineRouteBatchViewV2 {
    const uint8_t *frame = nullptr;
    uint64_t frame_bytes = 0u;
    uint64_t records_offset = 0u;
    uint32_t record_count = 0u;
    uint32_t flags = 0u;
    InlineRouteTokenParseContextV2 token_context{};
};

// Validates only the committed directory and member byte ranges. It never
// waits for or parses member commits, so an early token can be consumed while
// later records are still arriving.
InlineRouteParseStatusV2 ParseInlineTokenBatchV2(
    const uint8_t *frame, size_t available_bytes,
    const InlineRouteBatchParseContextV2 &context,
    InlineRouteBatchViewV2 *view);

// Applies the member commit gate, full token parser and stored-record padding
// validation for exactly one directory entry.
InlineRouteParseStatusV2 InlineBatchTokenAtV2(
    const InlineRouteBatchViewV2 &view, uint32_t record_index,
    InlineRouteTokenViewV2 *token);

const char *InlineRouteParseStatusStringV2(InlineRouteParseStatusV2 status);

} // namespace inc::dc

#endif
