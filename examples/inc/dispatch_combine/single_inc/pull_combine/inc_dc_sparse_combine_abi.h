#ifndef INC_DC_SPARSE_COMBINE_ABI_H
#define INC_DC_SPARSE_COMBINE_ABI_H

#include <cstdint>

namespace inc::dc::pull_combine {

constexpr uint32_t kSparseCombineMagic = 0x53434d42u; // 'SCMB'
constexpr uint16_t kSparseCombineAbiVersion = 1u;

enum class SparseCombineDType : uint32_t {
    FP32 = 0u,
    FP16 = 1u,
    BF16 = 2u,
};

// Worker B publishes this cache-line-aligned descriptor only after token IDs
// and locally reduced rows are visible in its registered ring slot.  INC GETs
// both arrays; no payload is pushed with the notification itself.
struct alignas(64) SparseCombineReadyDescriptor {
    uint32_t magic = kSparseCombineMagic;
    uint16_t abi_version = kSparseCombineAbiVersion;
    uint16_t struct_bytes = sizeof(SparseCombineReadyDescriptor);
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint32_t wave = 0u;
    uint32_t source_rank = 0u;
    uint32_t row_count = 0u;
    uint32_t hidden = 0u;
    uint32_t partial_dtype = static_cast<uint32_t>(SparseCombineDType::FP32);
    uint32_t slot = 0u;
    uint64_t source_region_id = 0u;
    uint64_t token_ids_offset = 0u;
    uint64_t payload_offset = 0u;
    uint64_t payload_bytes = 0u;
    uint64_t metadata_digest = 0u;
    uint32_t flags = 0u;
    uint32_t reserved0 = 0u;
    uint64_t reserved[4]{};
};
static_assert(sizeof(SparseCombineReadyDescriptor) == 128u,
              "sparse Combine ready descriptor ABI drift");

// Published after both GETs have completed.  On a negative ACK the worker may
// reclaim the slot, but rows_consumed is zero and no partial accumulator was
// modified.
struct alignas(64) SparseCombineDeviceAck {
    uint32_t magic = kSparseCombineMagic;
    uint16_t abi_version = kSparseCombineAbiVersion;
    uint16_t struct_bytes = sizeof(SparseCombineDeviceAck);
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint32_t source_rank = 0u;
    uint32_t status = 0u;
    uint64_t rows_consumed = 0u;
    uint64_t reserved[3]{};
};
static_assert(sizeof(SparseCombineDeviceAck) == 64u,
              "sparse Combine ACK must own one cache line");

struct alignas(64) SparseCombineEgressCompletion {
    uint32_t magic = kSparseCombineMagic;
    uint16_t abi_version = kSparseCombineAbiVersion;
    uint16_t struct_bytes = sizeof(SparseCombineEgressCompletion);
    uint64_t generation = 0u;
    uint32_t wave = 0u;
    uint32_t owner_rank = 0u;
    uint32_t status = 0u;
    uint32_t row_count = 0u;
    uint64_t reserved[4]{};
};
static_assert(sizeof(SparseCombineEgressCompletion) == 64u,
              "sparse Combine completion must own one cache line");

} // namespace inc::dc::pull_combine

#endif
