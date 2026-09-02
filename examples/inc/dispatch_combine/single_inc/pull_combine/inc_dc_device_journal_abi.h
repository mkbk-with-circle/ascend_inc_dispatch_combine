#ifndef INC_DC_DEVICE_JOURNAL_ABI_H
#define INC_DC_DEVICE_JOURNAL_ABI_H

#include <cstdint>

namespace inc::dc::pull_combine {

constexpr uint32_t kDeviceJournalMagic = 0x4a524e4cu; // 'JRNL'
constexpr uint16_t kDeviceJournalAbiVersion = 1u;
constexpr uint32_t kDeviceJournalEmpty = 0xffffffffu;
constexpr uint32_t kDeviceJournalIndexReady = 1u << 0;

struct alignas(64) DeviceJournalHeader {
    uint32_t magic = kDeviceJournalMagic;
    uint16_t abi_version = kDeviceJournalAbiVersion;
    uint16_t struct_bytes = sizeof(DeviceJournalHeader);
    uint64_t generation = 0u;
    uint32_t wave = 0u;
    uint32_t worker_count = 0u;
    uint64_t token_count = 0u;
    uint32_t status = 0u;
    uint32_t flags = 0u;
    uint64_t reserved[3]{};
};
static_assert(sizeof(DeviceJournalHeader) == 64u,
              "device journal header must own one cache line");

struct alignas(64) DeviceJournalEntry {
    uint64_t token_id = 0u;
    uint32_t owner_rank = 0u;
    uint32_t owner_row = 0u;
    uint64_t expected[2]{0u, 0u};
    uint64_t received[2]{0u, 0u};
    uint32_t accumulator_index = 0u;
    uint32_t flags = 0u;
    uint64_t reserved = 0u;
};
static_assert(sizeof(DeviceJournalEntry) == 64u,
              "device journal entry must own one cache line");

} // namespace inc::dc::pull_combine

#endif
