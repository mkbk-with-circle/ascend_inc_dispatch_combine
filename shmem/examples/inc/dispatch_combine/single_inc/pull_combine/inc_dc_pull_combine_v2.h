#ifndef INC_DC_PULL_COMBINE_V2_H
#define INC_DC_PULL_COMBINE_V2_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "inc_dc_pull_dispatch_v2.h"

namespace inc::dc::pull_v2 {

constexpr uint32_t kPullCombineV2Magic = 0x32434e49u; // 'INC2'
constexpr uint16_t kPullCombineV2AbiVersion = 1u;
constexpr uint32_t kPullCombineV2Alignment = 64u;
constexpr uint32_t kPullCombineV2CanonicalRows = 1u << 0u;

enum class CombineV2Status : uint32_t {
    OK = 0u,
    INVALID_ARGUMENT,
    INVALID_JOURNAL,
    INVALID_REGISTRATION,
    INVALID_READY,
    COOKIE_MISMATCH,
    STALE_EPOCH,
    SIZE_OVERFLOW,
    CAPACITY_EXCEEDED,
    DUPLICATE_SOURCE,
    NOT_READY,
    INVALID_STATE_TRANSITION,
    ABORTED,
};

enum class PartialDataType : uint32_t {
    FP16 = 0u,
    BF16 = 1u,
    FP32 = 2u,
};

// Installed once by the transport. Per-wave READY records contain only a
// registered region id and an offset; they never carry a process pointer.
struct alignas(64) CombineRegionRegistration {
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint32_t source_rank = 0u;
    uint32_t region_id = 0u;
    uint32_t slot_count = 0u;
    uint32_t alignment = kPullCombineV2Alignment;
    uint64_t region_bytes = 0u;
    uint64_t slot_stride = 0u;
    uint64_t reserved[2]{};
};
static_assert(sizeof(CombineRegionRegistration) == 64u,
              "pull Combine V2 registration ABI drift");

// B keeps this full READY in its registered local region after all locally
// reduced rows are visible.  It publishes only one cacheline-sized notice;
// the INC pulls this record after observing that notice.
struct alignas(64) CombineReadyV2 {
    uint32_t magic = kPullCombineV2Magic;
    uint16_t abi_version = kPullCombineV2AbiVersion;
    uint16_t struct_bytes = sizeof(CombineReadyV2);
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint64_t dispatch_cookie = 0u;
    uint32_t wave = 0u;
    uint32_t source_rank = 0u;
    uint32_t source_region_id = 0u;
    uint16_t ring_slot = 0u;
    uint16_t flags = kPullCombineV2CanonicalRows;
    uint32_t row_count = 0u;
    uint32_t hidden = 0u;
    uint32_t partial_dtype = static_cast<uint32_t>(PartialDataType::FP32);
    uint64_t source_offset = 0u;
    uint64_t payload_bytes = 0u;
    uint64_t publication = 0u;
    uint64_t reserved[3]{};
};
static_assert(sizeof(CombineReadyV2) == 128u,
              "pull Combine V2 READY ABI drift");

// The only worker->INC control transfer for one source/wave.  This fits one
// cacheline and keeps publication as the final word.  It identifies where in
// the agreed READY ring the INC should pull; the pulled READY carries the
// cookie, payload offset and full validation digest.
struct alignas(64) CombineReadyNoticeV2 {
    uint32_t magic = kPullCombineV2Magic;
    uint16_t abi_version = kPullCombineV2AbiVersion;
    uint16_t struct_bytes = sizeof(CombineReadyNoticeV2);
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint32_t wave = 0u;
    uint32_t source_rank = 0u;
    uint16_t ring_slot = 0u;
    uint16_t flags = 0u;
    uint32_t reserved0 = 0u;
    uint64_t publication = 0u;
};
static_assert(sizeof(CombineReadyNoticeV2) == 64u,
              "pull Combine V2 READY notice ABI drift");
static_assert(offsetof(CombineReadyNoticeV2, publication) +
                  sizeof(uint64_t) == sizeof(CombineReadyNoticeV2),
              "READY notice publication must be last");

struct CombineV2Config {
    uint64_t session_id = 0u;
    uint64_t placement_epoch = 0u;
    uint32_t worker_count = 0u;
    uint32_t hidden = 0u;
    PartialDataType partial_dtype = PartialDataType::FP32;
    uint32_t ring_slots = 0u;
};

// A canonical B buffer is partial[destination_row][hidden]. The compiler
// turns every Dispatch journal contributor directly into one indexed pull;
// no token-id hash table or W x row map is required by the data plane.
struct CombinePullOp {
    uint32_t source_rank = 0u;
    uint32_t source_row = 0u;
    uint32_t journal_token = 0u;
    uint32_t accumulator_index = 0u;
};
static_assert(sizeof(CombinePullOp) == 16u,
              "pull Combine V2 pull op ABI drift");

struct CombineResultOp {
    uint32_t owner_rank = 0u;
    uint32_t owner_row = 0u;
    uint32_t journal_token = 0u;
    uint32_t accumulator_index = 0u;
    uint32_t expected_contributors = 0u;
    uint32_t reserved[3]{};
};
static_assert(sizeof(CombineResultOp) == 32u,
              "pull Combine V2 result ABI drift");

struct CombinePullPlan {
    uint64_t generation = 0u;
    uint64_t sequence = 0u;
    uint64_t dispatch_cookie = 0u;
    uint32_t wave = 0u;
    uint16_t ring_slot = 0u;
    uint16_t reserved0 = 0u;
    uint32_t worker_count = 0u;
    uint32_t hidden = 0u;
    PartialDataType partial_dtype = PartialDataType::FP32;
    uint32_t accumulator_count = 0u;
    // source_offsets[B]..source_offsets[B+1] is a dense, canonical list of
    // B rows. Thus source_row is also the local array subscript.
    std::vector<uint64_t> source_offsets;
    std::vector<uint32_t> source_row_counts;
    std::vector<CombinePullOp> pulls;
    // Results are grouped by owner rank for contiguous INC->A publication.
    std::vector<uint64_t> owner_offsets;
    std::vector<CombineResultOp> results;
    // Immutable, deterministic reduction index produced together with the
    // sealed Dispatch journal. Combine consumes these arrays read-only.
    std::vector<uint32_t> pull_next;
    std::vector<uint32_t> accumulator_heads;
    std::vector<uint32_t> accumulator_contributor_counts;
    std::vector<uint32_t> accumulator_result_index;
};

CombineV2Status ValidateCombineRegistration(
    const CombineRegionRegistration &registration,
    const CombineV2Config &config, std::string *error = nullptr);

CombineV2Status CompileCombinePullPlan(const CompiledLayout &dispatch,
                                       const CombineV2Config &config,
                                       CombinePullPlan *plan,
                                       std::string *error = nullptr);

uint64_t CombineReadyPublication(const CombineReadyV2 &ready);
uint64_t CombineReadyNoticePublication(const CombineReadyNoticeV2 &notice);

CombineV2Status ValidateCombineReady(
    const CombineReadyV2 &ready,
    const CombineRegionRegistration &registration,
    const CombineV2Config &config, const CombinePullPlan &plan,
    std::string *error = nullptr);

// Host reference state machine for the resident INC controller. Initialize
// consumes a DISPATCH_SEALED journal and moves it to COMBINE_ACTIVE. READYs
// may arrive in any B order. Finish/Abort own the terminal journal transition.
class CombineV2Coordinator {
public:
    CombineV2Status Initialize(
        CompiledLayout *dispatch, const CombineV2Config &config,
        const std::vector<CombineRegionRegistration> &registrations,
        std::string *error = nullptr);
    CombineV2Status NotifyReady(const CombineReadyV2 &ready,
                                std::string *error = nullptr);
    CombineV2Status Finish(std::string *error = nullptr);
    CombineV2Status Abort(std::string *error = nullptr);

    bool all_ready() const;
    const CombinePullPlan &plan() const { return plan_; }
    JournalSlotState state() const;

private:
    CompiledLayout *dispatch_ = nullptr;
    CombineV2Config config_{};
    CombinePullPlan plan_{};
    std::vector<CombineRegionRegistration> registrations_;
    std::vector<uint8_t> ready_seen_;
    uint32_t ready_count_ = 0u;
};

const char *CombineV2StatusString(CombineV2Status status);

} // namespace inc::dc::pull_v2

#endif
