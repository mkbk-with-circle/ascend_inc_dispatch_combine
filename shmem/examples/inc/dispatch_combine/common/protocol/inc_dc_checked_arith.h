#ifndef INC_DC_CHECKED_ARITH_H
#define INC_DC_CHECKED_ARITH_H

#include <cstdint>
#include <limits>

#include "inc_dc_types.h"

namespace inc {
namespace dc {

inline bool CheckedMulU32(uint32_t a, uint32_t b, uint32_t *out)
{
    if (out == nullptr) {
        return false;
    }
    const uint64_t p = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    if (p > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    *out = static_cast<uint32_t>(p);
    return true;
}

inline bool CheckedMulU64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr) {
        return false;
    }
    if (a != 0ull && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    *out = a * b;
    return true;
}

inline IncDcStatus CheckedRowBytes(uint32_t hidden, uint32_t element_bytes,
                                   uint64_t *row_bytes)
{
    if (row_bytes == nullptr || hidden == 0u || element_bytes == 0u) {
        return IncDcStatus::INVALID_ARGUMENT;
    }
    if (!CheckedMulU64(hidden, element_bytes, row_bytes)) {
        return IncDcStatus::CAPACITY_EXCEEDED;
    }
    return IncDcStatus::OK;
}

} // namespace dc
} // namespace inc

#endif
