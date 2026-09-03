#include "inc_dc_pull_combine_v2.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace inc::dc::pull_v2 {
namespace {

constexpr uint64_t kHashOffset = 1469598103934665603ull;
constexpr uint64_t kHashPrime = 1099511628211ull;

CombineV2Status Fail(CombineV2Status status, const char *message,
                     std::string *error)
{
    if (error != nullptr) *error = message;
    return status;
}

bool Add(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr || b > std::numeric_limits<uint64_t>::max() - a)
        return false;
    *out = a + b;
    return true;
}

bool Mul(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr || (a != 0u &&
        b > std::numeric_limits<uint64_t>::max() / a))
        return false;
    *out = a * b;
    return true;
}

uint32_t PartialBytes(PartialDataType dtype)
{
    switch (dtype) {
        case PartialDataType::FP16:
        case PartialDataType::BF16: return 2u;
        case PartialDataType::FP32: return 4u;
    }
    return 0u;
}

bool ConfigValid(const CombineV2Config &config)
{
    return config.session_id != 0u && config.placement_epoch != 0u &&
        config.worker_count >= 2u &&
        config.worker_count <= kPullDispatchMaxWorkers &&
        config.hidden != 0u && PartialBytes(config.partial_dtype) != 0u &&
        config.ring_slots != 0u &&
        config.ring_slots <= std::numeric_limits<uint16_t>::max();
}

void HashBytes(uint64_t *hash, const void *data, size_t bytes)
{
    const auto *p = static_cast<const uint8_t *>(data);
    for (size_t i = 0u; i < bytes; ++i) {
        *hash ^= p[i];
        *hash *= kHashPrime;
    }
}

template <typename T>
void HashValue(uint64_t *hash, const T &value)
{
    HashBytes(hash, &value, sizeof(value));
}

bool ReservedZero(const uint64_t *values, uint32_t count)
{
    for (uint32_t i = 0u; i < count; ++i)
        if (values[i] != 0u) return false;
    return true;
}

bool HeaderCanonical(const JournalSlotHeader &header)
{
    return header.magic == kPullDispatchMagic &&
        header.abi_version == kPullDispatchAbiVersion &&
        header.struct_bytes == sizeof(JournalSlotHeader) &&
        header.generation != 0u && header.sequence != 0u &&
        header.dispatch_cookie != 0u && header.status == 0u &&
        header.flags == 0u && ReservedZero(header.reserved, 1u);
}

} // namespace

uint64_t CombineReadyPublication(const CombineReadyV2 &ready)
{
    uint64_t hash = kHashOffset;
    // Hash fields rather than the whole ABI struct: the four bytes before
    // source_offset are alignment padding and must not affect publication.
    HashValue(&hash, ready.magic);
    HashValue(&hash, ready.abi_version);
    HashValue(&hash, ready.struct_bytes);
    HashValue(&hash, ready.session_id);
    HashValue(&hash, ready.placement_epoch);
    HashValue(&hash, ready.generation);
    HashValue(&hash, ready.sequence);
    HashValue(&hash, ready.dispatch_cookie);
    HashValue(&hash, ready.wave);
    HashValue(&hash, ready.source_rank);
    HashValue(&hash, ready.source_region_id);
    HashValue(&hash, ready.ring_slot);
    HashValue(&hash, ready.flags);
    HashValue(&hash, ready.row_count);
    HashValue(&hash, ready.hidden);
    HashValue(&hash, ready.partial_dtype);
    HashValue(&hash, ready.source_offset);
    HashValue(&hash, ready.payload_bytes);
    for (uint64_t value : ready.reserved) HashValue(&hash, value);
    return hash == 0u ? 1u : hash;
}

CombineV2Status ValidateCombineRegistration(
    const CombineRegionRegistration &registration,
    const CombineV2Config &config, std::string *error)
{
    uint64_t required = 0u;
    if (!ConfigValid(config) ||
        registration.session_id != config.session_id ||
        registration.placement_epoch != config.placement_epoch ||
        registration.source_rank >= config.worker_count ||
        registration.region_id == 0u ||
        registration.slot_count < config.ring_slots ||
        registration.alignment < kPullCombineV2Alignment ||
        (registration.alignment & (registration.alignment - 1u)) != 0u ||
        registration.slot_stride == 0u ||
        registration.slot_stride % registration.alignment != 0u ||
        !Mul(registration.slot_count, registration.slot_stride, &required) ||
        required > registration.region_bytes ||
        !ReservedZero(registration.reserved, 2u)) {
        return Fail(CombineV2Status::INVALID_REGISTRATION,
                    "invalid registered Combine region", error);
    }
    if (error != nullptr) error->clear();
    return CombineV2Status::OK;
}

CombineV2Status CompileCombinePullPlan(const CompiledLayout &dispatch,
                                       const CombineV2Config &config,
                                       CombinePullPlan *plan,
                                       std::string *error)
{
    if (plan == nullptr || !ConfigValid(config))
        return Fail(CombineV2Status::INVALID_ARGUMENT,
                    "invalid Combine plan arguments", error);
    const JournalSlotHeader &header = dispatch.journal_header;
    if (!HeaderCanonical(header) ||
        static_cast<JournalSlotState>(header.state) !=
            JournalSlotState::DISPATCH_SEALED ||
        header.ring_slot >= config.ring_slots ||
        dispatch.destination_rows.size() != config.worker_count ||
        dispatch.expert_assignments.size() != config.worker_count ||
        header.token_count != dispatch.journal_tokens.size() ||
        header.contributor_count != dispatch.contributors.size() ||
        dispatch.journal_tokens.size() >
            std::numeric_limits<uint32_t>::max()) {
        return Fail(CombineV2Status::INVALID_JOURNAL,
                    "Dispatch journal is not sealed and canonical", error);
    }

    CombinePullPlan built{};
    built.generation = header.generation;
    built.sequence = header.sequence;
    built.dispatch_cookie = header.dispatch_cookie;
    built.wave = header.wave;
    built.ring_slot = header.ring_slot;
    built.worker_count = config.worker_count;
    built.hidden = config.hidden;
    built.partial_dtype = config.partial_dtype;
    built.accumulator_count = static_cast<uint32_t>(
        dispatch.journal_tokens.size());
    built.source_row_counts.resize(config.worker_count, 0u);
    built.pulls.reserve(dispatch.contributors.size());
    built.results.reserve(dispatch.journal_tokens.size());

    std::vector<uint8_t> accumulator_seen(dispatch.journal_tokens.size(),
                                          0u);
    uint64_t expected_contributor_begin = 0u;
    for (uint32_t token_index = 0u;
         token_index < dispatch.journal_tokens.size(); ++token_index) {
        const JournalTokenEntry &token = dispatch.journal_tokens[token_index];
        uint64_t contributor_end = 0u;
        uint64_t assignment_end = 0u;
        if (token.owner_rank >= config.worker_count ||
            token.route_key != RouteKey(token.owner_rank, token.owner_row) ||
            token.contributors_begin != expected_contributor_begin ||
            !Add(token.contributors_begin, token.contributors_count,
                 &contributor_end) ||
            contributor_end > dispatch.contributors.size() ||
            !Add(token.assignments_begin, token.assignments_count,
                 &assignment_end) ||
            assignment_end > dispatch.journal_assignments.size() ||
            token.accumulator_index >= dispatch.journal_tokens.size() ||
            accumulator_seen[token.accumulator_index] != 0u ||
            token.flags != 0u || !ReservedZero(token.reserved, 2u)) {
            return Fail(CombineV2Status::INVALID_JOURNAL,
                        "invalid token entry in Dispatch journal", error);
        }
        accumulator_seen[token.accumulator_index] = 1u;
        expected_contributor_begin = contributor_end;

        // Dispatch emits one contributor per unique destination B. A nested
        // check is intentional: top-k is small and this avoids any token-id
        // hash or W-wide row map in both compilation and the device plan.
        for (uint32_t local = 0u; local < token.contributors_count; ++local) {
            const JournalContributor &contributor = dispatch.contributors[
                token.contributors_begin + local];
            if (contributor.worker_rank >= config.worker_count ||
                contributor.destination_row >=
                    dispatch.destination_rows[
                        contributor.worker_rank].size() ||
                contributor.assignment_count == 0u) {
                return Fail(CombineV2Status::INVALID_JOURNAL,
                            "invalid contributor in Dispatch journal",
                            error);
            }
            uint64_t local_assignment_end = 0u;
            if (!Add(contributor.assignment_begin,
                     contributor.assignment_count,
                     &local_assignment_end) ||
                local_assignment_end >
                    dispatch.expert_assignments[
                        contributor.worker_rank].size()) {
                return Fail(CombineV2Status::INVALID_JOURNAL,
                            "contributor assignment range is invalid",
                            error);
            }
            const DestinationRow &row = dispatch.destination_rows[
                contributor.worker_rank][contributor.destination_row];
            if (row.route_key != token.route_key ||
                row.destination_row != contributor.destination_row ||
                row.assignments_begin != contributor.assignment_begin ||
                row.assignments_count != contributor.assignment_count) {
                return Fail(CombineV2Status::INVALID_JOURNAL,
                            "contributor does not name its canonical B row",
                            error);
            }
            for (uint32_t previous = 0u; previous < local; ++previous) {
                if (dispatch.contributors[token.contributors_begin + previous]
                        .worker_rank == contributor.worker_rank) {
                    return Fail(CombineV2Status::INVALID_JOURNAL,
                                "token has duplicate contributor B", error);
                }
            }
            built.pulls.push_back(CombinePullOp{
                contributor.worker_rank, contributor.destination_row,
                token_index, token.accumulator_index});
        }
        built.results.push_back(CombineResultOp{
            token.owner_rank, token.owner_row, token_index,
            token.accumulator_index, token.contributors_count, {0u, 0u, 0u}});
    }
    if (expected_contributor_begin != dispatch.contributors.size())
        return Fail(CombineV2Status::INVALID_JOURNAL,
                    "journal contributor ranges do not cover storage",
                    error);

    std::sort(built.pulls.begin(), built.pulls.end(),
              [](const CombinePullOp &a, const CombinePullOp &b) {
        if (a.source_rank != b.source_rank)
            return a.source_rank < b.source_rank;
        return a.source_row < b.source_row;
    });
    built.source_offsets.assign(config.worker_count + 1u, 0u);
    size_t pull = 0u;
    for (uint32_t source = 0u; source < config.worker_count; ++source) {
        built.source_offsets[source] = pull;
        const size_t expected_rows =
            dispatch.destination_rows[source].size();
        if (expected_rows > std::numeric_limits<uint32_t>::max())
            return Fail(CombineV2Status::CAPACITY_EXCEEDED,
                        "source row count exceeds device index width", error);
        for (uint32_t row = 0u; row < expected_rows; ++row, ++pull) {
            if (pull >= built.pulls.size() ||
                built.pulls[pull].source_rank != source ||
                built.pulls[pull].source_row != row) {
                return Fail(CombineV2Status::INVALID_JOURNAL,
                            "B rows are not a dense canonical array", error);
            }
        }
        built.source_row_counts[source] = static_cast<uint32_t>(expected_rows);
    }
    built.source_offsets[config.worker_count] = pull;
    if (pull != built.pulls.size())
        return Fail(CombineV2Status::INVALID_JOURNAL,
                    "pull plan contains an out-of-range source", error);

    std::sort(built.results.begin(), built.results.end(),
              [](const CombineResultOp &a, const CombineResultOp &b) {
        if (a.owner_rank != b.owner_rank)
            return a.owner_rank < b.owner_rank;
        return a.owner_row < b.owner_row;
    });
    built.owner_offsets.assign(config.worker_count + 1u, 0u);
    size_t result = 0u;
    for (uint32_t owner = 0u; owner < config.worker_count; ++owner) {
        built.owner_offsets[owner] = result;
        uint32_t owner_row = 0u;
        while (result < built.results.size() &&
               built.results[result].owner_rank == owner) {
            if (built.results[result].owner_row != owner_row++)
                return Fail(CombineV2Status::INVALID_JOURNAL,
                            "owner rows are not dense and canonical", error);
            ++result;
        }
    }
    built.owner_offsets[config.worker_count] = result;
    if (result != built.results.size())
        return Fail(CombineV2Status::INVALID_JOURNAL,
                    "result plan contains an out-of-range owner", error);

    *plan = std::move(built);
    if (error != nullptr) error->clear();
    return CombineV2Status::OK;
}

CombineV2Status ValidateCombineReady(
    const CombineReadyV2 &ready,
    const CombineRegionRegistration &registration,
    const CombineV2Config &config, const CombinePullPlan &plan,
    std::string *error)
{
    if (ValidateCombineRegistration(registration, config, error) !=
        CombineV2Status::OK)
        return CombineV2Status::INVALID_REGISTRATION;
    if (ready.magic != kPullCombineV2Magic ||
        ready.abi_version != kPullCombineV2AbiVersion ||
        ready.struct_bytes != sizeof(CombineReadyV2) ||
        ready.session_id != config.session_id ||
        ready.placement_epoch != config.placement_epoch ||
        ready.source_rank != registration.source_rank ||
        ready.source_region_id != registration.region_id ||
        ready.source_rank >= plan.worker_count ||
        ready.ring_slot >= config.ring_slots ||
        ready.ring_slot >= registration.slot_count ||
        ready.flags != kPullCombineV2CanonicalRows ||
        ready.hidden != config.hidden ||
        ready.partial_dtype != static_cast<uint32_t>(config.partial_dtype) ||
        !ReservedZero(ready.reserved, 3u) ||
        ready.publication != CombineReadyPublication(ready)) {
        return Fail(CombineV2Status::INVALID_READY,
                    "invalid Combine READY descriptor", error);
    }
    if (ready.dispatch_cookie != plan.dispatch_cookie)
        return Fail(CombineV2Status::COOKIE_MISMATCH,
                    "Combine READY dispatch cookie mismatch", error);
    if (ready.generation != plan.generation ||
        ready.sequence != plan.sequence || ready.wave != plan.wave ||
        ready.ring_slot != plan.ring_slot)
        return Fail(CombineV2Status::STALE_EPOCH,
                    "Combine READY epoch mismatch", error);
    if (ready.row_count != plan.source_row_counts[ready.source_rank])
        return Fail(CombineV2Status::INVALID_READY,
                    "Combine READY row count mismatch", error);

    uint64_t row_bytes = 0u;
    uint64_t expected_payload = 0u;
    uint64_t slot_base = 0u;
    uint64_t slot_end = 0u;
    uint64_t payload_end = 0u;
    if (!Mul(config.hidden, PartialBytes(config.partial_dtype), &row_bytes) ||
        !Mul(ready.row_count, row_bytes, &expected_payload) ||
        !Mul(ready.ring_slot, registration.slot_stride, &slot_base) ||
        !Add(slot_base, registration.slot_stride, &slot_end) ||
        !Add(ready.source_offset, ready.payload_bytes, &payload_end)) {
        return Fail(CombineV2Status::SIZE_OVERFLOW,
                    "Combine READY byte range overflow", error);
    }
    if (ready.payload_bytes != expected_payload ||
        ready.source_offset % registration.alignment != 0u ||
        ready.source_offset < slot_base || payload_end > slot_end ||
        payload_end > registration.region_bytes) {
        return Fail(CombineV2Status::CAPACITY_EXCEEDED,
                    "Combine READY is outside its registered slot", error);
    }
    if (error != nullptr) error->clear();
    return CombineV2Status::OK;
}

CombineV2Status CombineV2Coordinator::Initialize(
    CompiledLayout *dispatch, const CombineV2Config &config,
    const std::vector<CombineRegionRegistration> &registrations,
    std::string *error)
{
    if (dispatch_ != nullptr || dispatch == nullptr || !ConfigValid(config) ||
        registrations.size() != config.worker_count) {
        return Fail(CombineV2Status::INVALID_ARGUMENT,
                    "invalid Combine coordinator initialization", error);
    }
    CombinePullPlan built;
    CombineV2Status status = CompileCombinePullPlan(*dispatch, config,
                                                    &built, error);
    if (status != CombineV2Status::OK) return status;

    std::vector<CombineRegionRegistration> by_rank(config.worker_count);
    std::vector<uint8_t> seen(config.worker_count, 0u);
    for (const CombineRegionRegistration &registration : registrations) {
        status = ValidateCombineRegistration(registration, config, error);
        if (status != CombineV2Status::OK) return status;
        if (seen[registration.source_rank] != 0u)
            return Fail(CombineV2Status::DUPLICATE_SOURCE,
                        "duplicate Combine region registration", error);
        seen[registration.source_rank] = 1u;
        by_rank[registration.source_rank] = registration;
    }
    if (TransitionJournal(&dispatch->journal_header,
                          JournalSlotState::COMBINE_ACTIVE) != Status::OK) {
        return Fail(CombineV2Status::INVALID_STATE_TRANSITION,
                    "journal cannot enter COMBINE_ACTIVE", error);
    }
    dispatch_ = dispatch;
    config_ = config;
    plan_ = std::move(built);
    registrations_ = std::move(by_rank);
    ready_seen_.assign(config.worker_count, 0u);
    ready_count_ = 0u;
    if (error != nullptr) error->clear();
    return CombineV2Status::OK;
}

CombineV2Status CombineV2Coordinator::NotifyReady(
    const CombineReadyV2 &ready, std::string *error)
{
    if (dispatch_ == nullptr || state() != JournalSlotState::COMBINE_ACTIVE)
        return Fail(CombineV2Status::INVALID_STATE_TRANSITION,
                    "Combine wave is not active", error);
    if (ready.source_rank >= config_.worker_count)
        return Fail(CombineV2Status::INVALID_READY,
                    "Combine READY source is out of range", error);
    CombineV2Status status = ValidateCombineReady(
        ready, registrations_[ready.source_rank], config_, plan_, error);
    if (status != CombineV2Status::OK) return status;
    if (ready_seen_[ready.source_rank] != 0u)
        return Fail(CombineV2Status::DUPLICATE_SOURCE,
                    "worker published more than one Combine READY", error);
    ready_seen_[ready.source_rank] = 1u;
    ++ready_count_;
    if (error != nullptr) error->clear();
    return CombineV2Status::OK;
}

CombineV2Status CombineV2Coordinator::Finish(std::string *error)
{
    if (dispatch_ == nullptr || state() != JournalSlotState::COMBINE_ACTIVE)
        return Fail(CombineV2Status::INVALID_STATE_TRANSITION,
                    "Combine wave is not active", error);
    if (!all_ready())
        return Fail(CombineV2Status::NOT_READY,
                    "not every B has published Combine READY", error);
    if (TransitionJournal(&dispatch_->journal_header,
                          JournalSlotState::COMPLETE) != Status::OK) {
        return Fail(CombineV2Status::INVALID_STATE_TRANSITION,
                    "journal cannot enter COMPLETE", error);
    }
    if (error != nullptr) error->clear();
    return CombineV2Status::OK;
}

CombineV2Status CombineV2Coordinator::Abort(std::string *error)
{
    if (dispatch_ == nullptr || state() != JournalSlotState::COMBINE_ACTIVE)
        return Fail(CombineV2Status::INVALID_STATE_TRANSITION,
                    "Combine wave is not active", error);
    if (TransitionJournal(&dispatch_->journal_header,
                          JournalSlotState::ABORTED) != Status::OK) {
        return Fail(CombineV2Status::INVALID_STATE_TRANSITION,
                    "journal cannot enter ABORTED", error);
    }
    if (error != nullptr) error->clear();
    return CombineV2Status::ABORTED;
}

bool CombineV2Coordinator::all_ready() const
{
    return dispatch_ != nullptr && ready_count_ == config_.worker_count;
}

JournalSlotState CombineV2Coordinator::state() const
{
    return dispatch_ == nullptr ? JournalSlotState::FREE :
        static_cast<JournalSlotState>(dispatch_->journal_header.state);
}

const char *CombineV2StatusString(CombineV2Status status)
{
    switch (status) {
        case CombineV2Status::OK: return "ok";
        case CombineV2Status::INVALID_ARGUMENT: return "invalid_argument";
        case CombineV2Status::INVALID_JOURNAL: return "invalid_journal";
        case CombineV2Status::INVALID_REGISTRATION:
            return "invalid_registration";
        case CombineV2Status::INVALID_READY: return "invalid_ready";
        case CombineV2Status::COOKIE_MISMATCH: return "cookie_mismatch";
        case CombineV2Status::STALE_EPOCH: return "stale_epoch";
        case CombineV2Status::SIZE_OVERFLOW: return "size_overflow";
        case CombineV2Status::CAPACITY_EXCEEDED:
            return "capacity_exceeded";
        case CombineV2Status::DUPLICATE_SOURCE: return "duplicate_source";
        case CombineV2Status::NOT_READY: return "not_ready";
        case CombineV2Status::INVALID_STATE_TRANSITION:
            return "invalid_state_transition";
        case CombineV2Status::ABORTED: return "aborted";
    }
    return "unknown";
}

} // namespace inc::dc::pull_v2
