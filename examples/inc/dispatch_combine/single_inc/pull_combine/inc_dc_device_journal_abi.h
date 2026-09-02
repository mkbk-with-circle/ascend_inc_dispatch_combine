#ifndef INC_DC_DEVICE_JOURNAL_ABI_H
#define INC_DC_DEVICE_JOURNAL_ABI_H

#include <cstdint>

namespace inc::dc::pull_combine {

constexpr uint32_t kDeviceJournalMagic = 0x4a524e4cu; // 'JRNL'
constexpr uint16_t kDeviceJournalAbiVersion = 1u;

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

} // namespace inc::dc::pull_combine

#endif
