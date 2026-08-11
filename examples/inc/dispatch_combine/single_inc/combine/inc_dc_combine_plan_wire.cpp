#include "inc_dc_combine_plan_wire.h"

#include "inc_dc_checked_arith.h"

#include <cstring>
#include <limits>

namespace inc {
namespace dc {
namespace {

constexpr uint32_t kFlagUniformTopkValid = 1u;

uint64_t Mix(uint64_t h, uint64_t v)
{
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}

uint32_t FloatBits(float v)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

float BitsFloat(uint32_t bits)
{
    float v = 0.f;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

bool AlignOk(uint64_t offset, uint64_t align)
{
    return (offset % align) == 0ull;
}

} // namespace

uint64_t ComputeLogicalPlanWireIntegrityDigest(const uint8_t *blob,
                                               uint64_t bytes)
{
    if (blob == nullptr || bytes < sizeof(IncDcLogicalPlanWireHeaderV2)) {
        return 0;
    }
    // Canonical: entire blob with integrity_digest field zeroed.
    std::vector<uint8_t> tmp(blob, blob + bytes);
    auto *hdr = reinterpret_cast<IncDcLogicalPlanWireHeaderV2 *>(tmp.data());
    hdr->integrity_digest = 0;
    uint64_t h = 0x5749524556320001ull; // WIREV2\x01
    for (uint64_t i = 0; i < bytes; ++i) {
        h = Mix(h, tmp[static_cast<size_t>(i)]);
    }
    return h == 0 ? 1 : h;
}

IncDcStatus SerializeLogicalPlanV2(const IncDcCombineLogicalPlanV2 &plan,
                                   std::vector<uint8_t> *blob,
                                   IncDcLogicalPlanWireReport *report)
{
    IncDcLogicalPlanWireReport local{};
    auto *rep = report ? report : &local;
    rep->ok = false;
    if (blob == nullptr) {
        rep->first_error = "null_blob";
        return IncDcStatus::INVALID_ARGUMENT;
    }
    IncDcLogicalPlanValidateReport lv{};
    if (ValidateLogicalPlanV2(plan, &lv) != IncDcStatus::OK) {
        rep->first_error = "logical_invalid:" + lv.first_error;
        return IncDcStatus::INVALID_ARGUMENT;
    }

    const uint64_t hdr_sz = sizeof(IncDcLogicalPlanWireHeaderV2);
    const uint64_t res_off = hdr_sz;
    uint64_t res_bytes = 0;
    if (!CheckedMulU64(plan.result_count, sizeof(IncDcLogicalResultWireV2),
                       &res_bytes)) {
        rep->first_error = "results_bytes_overflow";
        return IncDcStatus::CAPACITY_EXCEEDED;
    }
    const uint64_t contrib_off = res_off + res_bytes;
    uint64_t contrib_bytes = 0;
    if (!CheckedMulU64(plan.contribution_count,
                       sizeof(IncDcLogicalContributionWireV2), &contrib_bytes)) {
        rep->first_error = "contrib_bytes_overflow";
        return IncDcStatus::CAPACITY_EXCEEDED;
    }
    if (contrib_off > std::numeric_limits<uint64_t>::max() - contrib_bytes) {
        rep->first_error = "total_bytes_overflow";
        return IncDcStatus::CAPACITY_EXCEEDED;
    }
    const uint64_t total = contrib_off + contrib_bytes;

    blob->assign(static_cast<size_t>(total), 0);
    auto *hdr = reinterpret_cast<IncDcLogicalPlanWireHeaderV2 *>(blob->data());
    *hdr = IncDcLogicalPlanWireHeaderV2{};
    hdr->total_bytes = total;
    hdr->worker_world_size = plan.worker_world_size;
    hdr->result_count = plan.result_count;
    hdr->contribution_count = plan.contribution_count;
    hdr->declared_max_topk = plan.declared_max_topk;
    hdr->uniform_topk = plan.uniform_topk;
    hdr->flags = plan.uniform_topk_valid ? kFlagUniformTopkValid : 0u;
    hdr->results_offset = res_off;
    hdr->contributions_offset = contrib_off;
    hdr->semantic_digest = plan.semantic_digest;

    auto *res = reinterpret_cast<IncDcLogicalResultWireV2 *>(blob->data() +
                                                            res_off);
    for (uint32_t i = 0; i < plan.result_count; ++i) {
        res[i].dst_rank = plan.results[i].dst_rank;
        res[i].dst_local_row = plan.results[i].dst_local_row;
        res[i].contribution_begin = plan.results[i].contribution_begin;
        res[i].contribution_count = plan.results[i].contribution_count;
    }
    auto *cc = reinterpret_cast<IncDcLogicalContributionWireV2 *>(
        blob->data() + contrib_off);
    for (uint32_t i = 0; i < plan.contribution_count; ++i) {
        cc[i].contribution_uid = plan.contributions[i].contribution_uid;
        cc[i].result_id = plan.contributions[i].result_id;
        cc[i].ordinal = plan.contributions[i].ordinal;
        cc[i].contributor_rank = plan.contributions[i].contributor_rank;
        cc[i].contributor_local_row = plan.contributions[i].contributor_local_row;
        cc[i].weight_bits = FloatBits(plan.contributions[i].weight);
    }
    hdr->integrity_digest =
        ComputeLogicalPlanWireIntegrityDigest(blob->data(), total);
    IncDcLogicalPlanWireReport vr{};
    if (ValidateLogicalPlanWireV2(blob->data(), total, &vr) != IncDcStatus::OK) {
        rep->first_error = "post_serialize_validate:" + vr.first_error;
        return IncDcStatus::INVALID_ARGUMENT;
    }
    rep->ok = true;
    return IncDcStatus::OK;
}

IncDcStatus ValidateLogicalPlanWireV2(const void *blob, uint64_t bytes,
                                      IncDcLogicalPlanWireReport *report)
{
    IncDcLogicalPlanWireReport local{};
    auto *rep = report ? report : &local;
    rep->ok = false;
    auto fail = [&](const char *msg) -> IncDcStatus {
        rep->first_error = msg;
        return IncDcStatus::INVALID_ARGUMENT;
    };
    if (blob == nullptr) return fail("null_blob");
    if (bytes < sizeof(IncDcLogicalPlanWireHeaderV2)) return fail("short_header");
    const auto *hdr = static_cast<const IncDcLogicalPlanWireHeaderV2 *>(blob);
    if (hdr->magic != kIncDcLogicalPlanWireMagicV2) return fail("bad_magic");
    if (hdr->abi_major != kIncDcLogicalPlanWireAbiMajor) return fail("bad_abi_major");
    if (hdr->total_bytes != bytes) return fail("total_bytes_mismatch");
    if (!AlignOk(hdr->results_offset, alignof(IncDcLogicalResultWireV2))) {
        return fail("results_unaligned");
    }
    if (!AlignOk(hdr->contributions_offset,
                 alignof(IncDcLogicalContributionWireV2))) {
        return fail("contrib_unaligned");
    }
    uint64_t res_bytes = 0;
    if (!CheckedMulU64(hdr->result_count, sizeof(IncDcLogicalResultWireV2),
                       &res_bytes)) {
        return fail("results_bytes_overflow");
    }
    uint64_t contrib_bytes = 0;
    if (!CheckedMulU64(hdr->contribution_count,
                       sizeof(IncDcLogicalContributionWireV2), &contrib_bytes)) {
        return fail("contrib_bytes_overflow");
    }
    if (hdr->results_offset > bytes ||
        hdr->results_offset + res_bytes > bytes) {
        return fail("results_oob");
    }
    if (hdr->contributions_offset > bytes ||
        hdr->contributions_offset + contrib_bytes > bytes) {
        return fail("contrib_oob");
    }
    if (hdr->results_offset != sizeof(IncDcLogicalPlanWireHeaderV2)) {
        return fail("results_offset_unexpected");
    }
    if (hdr->contributions_offset != hdr->results_offset + res_bytes) {
        return fail("contrib_offset_unexpected");
    }
    if (hdr->results_offset + res_bytes + contrib_bytes != bytes) {
        return fail("trailing_bytes");
    }
    const uint64_t expect_integrity = ComputeLogicalPlanWireIntegrityDigest(
        static_cast<const uint8_t *>(blob), bytes);
    if (hdr->integrity_digest == 0ull ||
        hdr->integrity_digest != expect_integrity) {
        return fail("integrity_digest_mismatch");
    }
    // Rebuild host plan and recompute semantic digest.
    IncDcCombineLogicalPlanV2 tmp{};
    IncDcLogicalPlanWireReport pr{};
    if (ParseLogicalPlanWireV2(blob, bytes, &tmp, &pr) != IncDcStatus::OK) {
        return fail("parse_failed");
    }
    if (tmp.semantic_digest != hdr->semantic_digest) {
        return fail("semantic_digest_mismatch");
    }
    IncDcLogicalPlanValidateReport lv{};
    if (ValidateLogicalPlanV2(tmp, &lv) != IncDcStatus::OK) {
        return fail("logical_validate_failed");
    }
    rep->ok = true;
    return IncDcStatus::OK;
}

IncDcStatus ParseLogicalPlanWireV2(const void *blob, uint64_t bytes,
                                   IncDcCombineLogicalPlanV2 *out,
                                   IncDcLogicalPlanWireReport *report)
{
    IncDcLogicalPlanWireReport local{};
    auto *rep = report ? report : &local;
    rep->ok = false;
    if (out == nullptr || blob == nullptr) {
        rep->first_error = "null_arg";
        return IncDcStatus::INVALID_ARGUMENT;
    }
    if (bytes < sizeof(IncDcLogicalPlanWireHeaderV2)) {
        rep->first_error = "short_header";
        return IncDcStatus::INVALID_ARGUMENT;
    }
    const auto *hdr = static_cast<const IncDcLogicalPlanWireHeaderV2 *>(blob);
    if (hdr->magic != kIncDcLogicalPlanWireMagicV2 ||
        hdr->abi_major != kIncDcLogicalPlanWireAbiMajor ||
        hdr->total_bytes != bytes) {
        rep->first_error = "bad_header";
        return IncDcStatus::INVALID_ARGUMENT;
    }
    uint64_t res_bytes = 0;
    uint64_t contrib_bytes = 0;
    if (!CheckedMulU64(hdr->result_count, sizeof(IncDcLogicalResultWireV2),
                       &res_bytes) ||
        !CheckedMulU64(hdr->contribution_count,
                       sizeof(IncDcLogicalContributionWireV2), &contrib_bytes)) {
        rep->first_error = "size_overflow";
        return IncDcStatus::CAPACITY_EXCEEDED;
    }
    if (hdr->results_offset + res_bytes > bytes ||
        hdr->contributions_offset + contrib_bytes > bytes) {
        rep->first_error = "range_oob";
        return IncDcStatus::INVALID_ARGUMENT;
    }

    *out = IncDcCombineLogicalPlanV2{};
    out->abi_version = kIncDcCombineLogicalPlanAbiV2;
    out->worker_world_size = hdr->worker_world_size;
    out->result_count = hdr->result_count;
    out->contribution_count = hdr->contribution_count;
    out->declared_max_topk = hdr->declared_max_topk;
    out->uniform_topk = hdr->uniform_topk;
    out->uniform_topk_valid = (hdr->flags & kFlagUniformTopkValid) != 0u;
    out->results.resize(hdr->result_count);
    out->contributions.resize(hdr->contribution_count);

    const auto *res = reinterpret_cast<const IncDcLogicalResultWireV2 *>(
        static_cast<const uint8_t *>(blob) + hdr->results_offset);
    for (uint32_t i = 0; i < hdr->result_count; ++i) {
        out->results[i].dst_rank = res[i].dst_rank;
        out->results[i].dst_local_row = res[i].dst_local_row;
        out->results[i].contribution_begin = res[i].contribution_begin;
        out->results[i].contribution_count = res[i].contribution_count;
    }
    const auto *cc = reinterpret_cast<const IncDcLogicalContributionWireV2 *>(
        static_cast<const uint8_t *>(blob) + hdr->contributions_offset);
    for (uint32_t i = 0; i < hdr->contribution_count; ++i) {
        out->contributions[i].contribution_uid = cc[i].contribution_uid;
        out->contributions[i].result_id = cc[i].result_id;
        out->contributions[i].ordinal = cc[i].ordinal;
        out->contributions[i].contributor_rank = cc[i].contributor_rank;
        out->contributions[i].contributor_local_row = cc[i].contributor_local_row;
        out->contributions[i].weight = BitsFloat(cc[i].weight_bits);
    }
    out->semantic_digest = ComputeLogicalPlanSemanticDigest(*out);
    if (hdr->semantic_digest != 0ull &&
        out->semantic_digest != hdr->semantic_digest) {
        rep->first_error = "semantic_digest_mismatch";
        return IncDcStatus::INVALID_ARGUMENT;
    }
    rep->ok = true;
    return IncDcStatus::OK;
}

} // namespace dc
} // namespace inc
