#ifndef INC_DC_COMBINE_LINEAGE_SIDECAR_H
#define INC_DC_COMBINE_LINEAGE_SIDECAR_H

#include <cstdint>
#include <cstdlib>

namespace inc {
namespace dc {

// BW05-K1 data-lineage diagnostic sidecar (debug-only; outside schema3 ownership).
constexpr uint32_t kBw05K1DiagMagic = 0x4B314447u; // 'K1DG'
constexpr uint32_t kBw05K1DiagSchema = 1u;
constexpr uint32_t kBw05K1StageA = 0u; // local_hidden
constexpr uint32_t kBw05K1StageB = 1u; // lane_staging after gather (+barrier only)
constexpr uint32_t kBw05K1StageC = 2u; // payload ring after desc visible
constexpr uint32_t kBw05K1StageD = 3u; // ingress after consume
constexpr uint32_t kBw05K1StageE = 4u; // egress after topk1 copy
constexpr uint32_t kBw05K1StageF = 5u; // recv_hidden after result_ready
constexpr uint32_t kBw05K1StageCount = 6u;

struct alignas(64) Bw05K1DiagHeader {
    uint32_t magic = kBw05K1DiagMagic;
    uint32_t schema = kBw05K1DiagSchema;
    uint32_t pe = 0;
    uint32_t claim = 0; // 0=free, 1=claimed (first sample wins)
    uint32_t filled_mask = 0; // bit i set => stage i filled
    uint32_t generation = 0;
    uint32_t assignment_id = 0xFFFFFFFFu;
    uint32_t src_worker = 0xFFFFFFFFu;
    uint32_t dst_switch = 0xFFFFFFFFu;
    uint32_t lane = 0xFFFFFFFFu;
    uint32_t desc_slot = 0xFFFFFFFFu;
    uint32_t worklist_offset = 0xFFFFFFFFu;
    uint32_t expert_output_offset = 0xFFFFFFFFu;
    uint32_t ingress_slot = 0xFFFFFFFFu;
    uint32_t payload_bytes = 0;
    uint32_t entry_count = 0;
    uint32_t pad0 = 0;
    uint64_t digest[kBw05K1StageCount]{};
    uint64_t u64_first[kBw05K1StageCount]{};
    uint64_t u64_mid[kBw05K1StageCount]{};
    uint64_t u64_last[kBw05K1StageCount]{};
    uint8_t pad[64]{};
};
static_assert(sizeof(Bw05K1DiagHeader) % 64u == 0u, "K1 diag header cacheline aligned");

struct Bw05K1DiagSidecarLayout {
    uint64_t header_off = 0;
    uint64_t total_bytes = 0;
};

inline Bw05K1DiagSidecarLayout ComputeBw05K1DiagSidecarLayout(uint64_t base_off)
{
    Bw05K1DiagSidecarLayout lo{};
    lo.header_off = base_off;
    lo.total_bytes = sizeof(Bw05K1DiagHeader);
    return lo;
}

inline bool Bw05K1DiagEnabled()
{
    const char *raw = std::getenv("INC_DC_BW05_K1_DIAG");
    return raw != nullptr && raw[0] == '1' && raw[1] == '\0';
}

} // namespace dc
} // namespace inc

#endif
