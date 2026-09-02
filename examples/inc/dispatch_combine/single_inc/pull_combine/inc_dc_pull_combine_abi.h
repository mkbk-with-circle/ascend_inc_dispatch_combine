#ifndef INC_DC_PULL_COMBINE_ABI_H
#define INC_DC_PULL_COMBINE_ABI_H

#include <cstdint>

namespace inc::dc::pull_combine {

constexpr uint32_t kPullCombineMagic = 0x50434D42u; // 'PCMB'
constexpr uint16_t kPullCombineAbiVersion = 1u;
constexpr uint32_t kPullCombineCacheLine = 64u;
constexpr uint32_t kPullCombineMaxWorkers = 128u;

enum class PartialDType : uint32_t {
    FP32 = 0u,
    FP16 = 1u,
    BF16 = 2u,
};

enum class DescriptorFlags : uint32_t {
    NONE = 0u,
    FINAL_FOR_WAVE = 1u << 0,
    DEBUG_TOKEN_KEYS = 1u << 1,
};

inline constexpr uint32_t ToBits(DescriptorFlags flags)
{
    return static_cast<uint32_t>(flags);
}

// A worker pushes one count commit to the INC for every wave.  token counts
// and assignment counts are two worker_count-element uint64_t vectors in the
// registered source region.  Counts are physically exchanged only through
// the INC; the INC computes the transpose and receive offsets.
struct alignas(64) CountCommitDescriptor {
    uint32_t magic = kPullCombineMagic;
    uint16_t abi_version = kPullCombineAbiVersion;
    uint16_t struct_bytes = sizeof(CountCommitDescriptor);
    uint64_t generation = 0u;
    uint32_t wave = 0u;
    uint32_t source_rank = 0u;
    uint32_t worker_count = 0u;
    uint32_t reserved0 = 0u;
    uint64_t source_region_id = 0u;
    uint64_t token_counts_offset = 0u;
    uint64_t assignment_counts_offset = 0u;
    uint64_t semantic_digest = 0u;
};
static_assert(sizeof(CountCommitDescriptor) == 64u,
              "count commit must own one cache line");

// A has already pushed this contiguous token range into its INC-owned staging
// slot when it publishes the descriptor.  The INC may fan one hidden row out
// to several B ranks, but A transfers exactly one hidden copy per token.
struct alignas(64) DispatchIngressDescriptor {
    uint32_t magic = kPullCombineMagic;
    uint16_t abi_version = kPullCombineAbiVersion;
    uint16_t struct_bytes = sizeof(DispatchIngressDescriptor);
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint32_t wave = 0u;
    uint32_t source_rank = 0u;
    uint32_t source_token_begin = 0u;
    uint32_t token_count = 0u;
    uint32_t element_bytes = 0u;
    uint32_t flags = 0u;
    uint64_t inc_region_id = 0u;
    uint64_t inc_offset = 0u;
    uint64_t payload_bytes = 0u;
    uint64_t semantic_digest = 0u;
    uint64_t reserved[6]{};
};
static_assert(sizeof(DispatchIngressDescriptor) == 128u,
              "dispatch ingress descriptor ABI drift");

// The INC publishes this ACK only after every downstream PUT that reads the
// corresponding staging range has completed.  A may then reuse its INC slot.
struct alignas(64) DispatchAck {
    uint32_t magic = kPullCombineMagic;
    uint16_t abi_version = kPullCombineAbiVersion;
    uint16_t struct_bytes = sizeof(DispatchAck);
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint32_t source_rank = 0u;
    uint32_t status = 0u;
    uint64_t tokens_consumed = 0u;
    uint64_t reserved[3]{};
};
static_assert(sizeof(DispatchAck) == 64u,
              "dispatch ack must own one cache line");

// B publishes one descriptor after a contiguous range of locally reduced
// partial rows is visible in its registered symmetric send region.  A normal
// descriptor publication transfers no payload: it only makes the range
// eligible for an INC-initiated nonblocking GET.
struct alignas(64) CombineReadyDescriptor {
    uint32_t magic = kPullCombineMagic;
    uint16_t abi_version = kPullCombineAbiVersion;
    uint16_t struct_bytes = sizeof(CombineReadyDescriptor);
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint32_t wave = 0u;
    uint32_t source_rank = 0u;
    uint64_t combine_row_begin = 0u;
    uint32_t row_count = 0u;
    uint32_t partial_dtype = static_cast<uint32_t>(PartialDType::FP32);
    uint64_t source_region_id = 0u;
    uint64_t source_offset = 0u;
    uint64_t payload_bytes = 0u;
    uint64_t semantic_digest = 0u;
    uint32_t flags = 0u;
    uint32_t reserved0 = 0u;
    uint64_t reserved[5]{};
};
static_assert(sizeof(CombineReadyDescriptor) == 128u,
              "ready descriptor ABI drift");

// The INC publishes ACK only after GET completion.  Before observing this
// sequence, B must not overwrite or unregister the corresponding source
// range.  Status zero means success; nonzero is fail-closed.
struct alignas(64) CombineAck {
    uint32_t magic = kPullCombineMagic;
    uint16_t abi_version = kPullCombineAbiVersion;
    uint16_t struct_bytes = sizeof(CombineAck);
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint32_t source_rank = 0u;
    uint32_t status = 0u;
    uint64_t rows_consumed = 0u;
    uint64_t reserved[3]{};
};
static_assert(sizeof(CombineAck) == 64u, "ack must own one cache line");

} // namespace inc::dc::pull_combine

#endif
