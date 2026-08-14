#ifndef INC_DC_PHYSICAL_MAP_H
#define INC_DC_PHYSICAL_MAP_H

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace inc::dc {

// Resolve a logical PE without imposing a compile-time cluster-size limit.
// INC_PE_TO_NPU_MAP uses "logical:physical,..." and may be sparse; an absent
// entry falls back to the launcher's legacy contiguous mapping.
inline int ResolvePhysicalNpuForPe(int pe, int device_count, int first_device)
{
    const char *raw = std::getenv("INC_PE_TO_NPU_MAP");
    if (raw != nullptr && raw[0] != '\0') {
        const std::string map(raw);
        size_t pos = 0;
        while (pos < map.size()) {
            const size_t comma = map.find(',', pos);
            const std::string token = map.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            const size_t colon = token.find(':');
            if (colon != std::string::npos) {
                const std::string logical_text = token.substr(0, colon);
                const std::string physical_text = token.substr(colon + 1);
                char *lend = nullptr;
                char *pend = nullptr;
                errno = 0;
                const long logical =
                    std::strtol(logical_text.c_str(), &lend, 10);
                const long physical =
                    std::strtol(physical_text.c_str(), &pend, 10);
                if (errno == 0 && lend != nullptr && *lend == '\0' && pend != nullptr && *pend == '\0' &&
                    logical == pe && physical >= 0 && physical <= INT_MAX) {
                    return static_cast<int>(physical);
                }
            }
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }
    return device_count > 0 ? pe % device_count + first_device : pe + first_device;
}

struct IncDcPhysicalMapValidation {
    uint64_t digest = 1469598103934665603ull;
    uint32_t explicit_entries = 0u;
};

// Validate the complete effective PE map once during service registration.
// Sparse explicit maps remain supported: unspecified PEs retain the legacy
// contiguous mapping.  The device data path never parses this configuration.
inline bool ValidatePhysicalNpuMap(int pe_count, int device_count,
                                   int first_device, bool require_unique,
                                   IncDcPhysicalMapValidation *validation,
                                   std::string *error)
{
    if (validation == nullptr || pe_count <= 0 || device_count <= 0 ||
        first_device < 0 ||
        first_device > std::numeric_limits<int>::max() - device_count) {
        if (error != nullptr) *error = "invalid_map_extent";
        return false;
    }
    std::vector<int> resolved(static_cast<size_t>(pe_count));
    std::vector<uint8_t> explicit_seen(static_cast<size_t>(pe_count), 0u);
    for (int pe = 0; pe < pe_count; ++pe) {
        resolved[static_cast<size_t>(pe)] = pe % device_count + first_device;
    }
    uint32_t explicit_entries = 0u;
    const char *raw = std::getenv("INC_PE_TO_NPU_MAP");
    if (raw != nullptr && raw[0] != '\0') {
        const std::string map(raw);
        size_t pos = 0u;
        while (pos < map.size()) {
            const size_t comma = map.find(',', pos);
            const std::string token = map.substr(
                pos, comma == std::string::npos ? std::string::npos
                                                : comma - pos);
            const size_t colon = token.find(':');
            if (token.empty() || colon == std::string::npos ||
                token.find(':', colon + 1u) != std::string::npos) {
                if (error != nullptr) *error = "malformed_map_token";
                return false;
            }
            char *logical_end = nullptr;
            char *physical_end = nullptr;
            const std::string logical_text = token.substr(0u, colon);
            const std::string physical_text = token.substr(colon + 1u);
            errno = 0;
            const long logical =
                std::strtol(logical_text.c_str(), &logical_end, 10);
            const int logical_errno = errno;
            errno = 0;
            const long physical =
                std::strtol(physical_text.c_str(), &physical_end, 10);
            const int physical_errno = errno;
            if (logical_errno != 0 || physical_errno != 0 ||
                logical_end == nullptr || *logical_end != '\0' ||
                physical_end == nullptr || *physical_end != '\0' ||
                logical < 0 || logical >= pe_count ||
                physical < first_device ||
                physical >= static_cast<long>(first_device) + device_count) {
                if (error != nullptr) *error = "map_entry_out_of_range";
                return false;
            }
            const size_t logical_index = static_cast<size_t>(logical);
            if (explicit_seen[logical_index] != 0u) {
                if (error != nullptr) *error = "duplicate_logical_pe";
                return false;
            }
            explicit_seen[logical_index] = 1u;
            resolved[logical_index] = static_cast<int>(physical);
            ++explicit_entries;
            if (comma == std::string::npos) break;
            pos = comma + 1u;
            if (pos == map.size()) {
                if (error != nullptr) *error = "trailing_map_separator";
                return false;
            }
        }
    }
    if (require_unique) {
        std::vector<uint8_t> used(static_cast<size_t>(device_count), 0u);
        for (int physical : resolved) {
            const size_t index = static_cast<size_t>(physical - first_device);
            if (used[index] != 0u) {
                if (error != nullptr) *error = "duplicate_physical_npu";
                return false;
            }
            used[index] = 1u;
        }
    }
    uint64_t digest = 1469598103934665603ull;
    for (int physical : resolved) {
        digest ^= static_cast<uint32_t>(physical);
        digest *= 1099511628211ull;
    }
    validation->digest = digest;
    validation->explicit_entries = explicit_entries;
    return true;
}

} // namespace inc::dc

#endif
