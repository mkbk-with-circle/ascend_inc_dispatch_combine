#include "inc_dc_combine_logical_plan.h"
#include "inc_dc_combine_plan_wire.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <sys/random.h>
#include <vector>

namespace {

using inc::dc::ComputeLogicalPlanSemanticDigest;
using inc::dc::IncDcCombineLogicalPlanV2;
using inc::dc::IncDcLogicalContributionV2;
using inc::dc::IncDcLogicalPlanWireReport;
using inc::dc::IncDcLogicalResultV2;
using inc::dc::IncDcStatus;
using inc::dc::SerializeLogicalPlanV2;

bool ParseU32(const char *text, uint32_t *value)
{
    if (text == nullptr || value == nullptr || text[0] == '\0') return false;
    char *end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    *value = static_cast<uint32_t>(parsed);
    return true;
}

bool OsRandomU64(uint64_t *value)
{
    auto *out = reinterpret_cast<unsigned char *>(value);
    size_t done = 0;
    while (done < sizeof(*value)) {
        const ssize_t got =
            getrandom(out + done, sizeof(*value) - done, /*flags=*/0);
        if (got > 0) {
            done += static_cast<size_t>(got);
            continue;
        }
        if (got < 0 && errno == EINTR) continue;
        return false;
    }
    return true;
}

bool UniformBelow(uint32_t bound, uint32_t *value)
{
    if (bound == 0u || value == nullptr) return false;
    const uint64_t limit =
        std::numeric_limits<uint64_t>::max() -
        (std::numeric_limits<uint64_t>::max() % bound);
    uint64_t sample = 0;
    do {
        if (!OsRandomU64(&sample)) return false;
    } while (sample >= limit);
    *value = static_cast<uint32_t>(sample % bound);
    return true;
}

bool RandomExperts(uint32_t expert_count, uint32_t topk,
                   std::vector<uint32_t> *experts)
{
    if (experts == nullptr || topk == 0u || topk > expert_count) return false;
    experts->resize(expert_count);
    for (uint32_t i = 0; i < expert_count; ++i) (*experts)[i] = i;
    // Partial Fisher-Yates. Every selection consumes fresh kernel entropy;
    // there is no seeded PRNG state and experts remain unique within a token.
    for (uint32_t i = 0; i < topk; ++i) {
        uint32_t offset = 0;
        if (!UniformBelow(expert_count - i, &offset)) return false;
        const uint32_t selected = i + offset;
        std::swap((*experts)[i], (*experts)[selected]);
    }
    experts->resize(topk);
    return true;
}

bool WriteDispatch(uint32_t workers, uint32_t topk, uint32_t tokens,
                   const std::string &path)
{
    std::ofstream out(path, std::ios::trunc);
    if (!out) return false;
    out << "INC_TOKEN_PLAN_V1 " << workers << ' ' << tokens << ' ' << topk
        << '\n';
    std::vector<uint32_t> experts;
    for (uint32_t source = 0; source < workers; ++source) {
        for (uint32_t row = 0; row < tokens; ++row) {
            if (!RandomExperts(workers * 8u, topk, &experts)) return false;
            for (uint32_t ordinal = 0; ordinal < topk; ++ordinal) {
                const uint32_t expert = experts[ordinal];
                out << source << ' ' << row << ' ' << ordinal << ' '
                    << expert / 8u << ' ' << expert << " 1\n";
            }
        }
    }
    return static_cast<bool>(out);
}

bool WriteCombine(uint32_t workers, uint32_t topk, uint32_t results,
                  const std::string &path)
{
    IncDcCombineLogicalPlanV2 plan{};
    plan.worker_world_size = workers;
    plan.result_count = results;
    plan.contribution_count = results * topk;
    plan.declared_max_topk = topk;
    plan.uniform_topk_valid = true;
    plan.uniform_topk = topk;
    plan.results.reserve(results);
    plan.contributions.reserve(plan.contribution_count);
    std::vector<uint32_t> destination_rows(workers, 0u);
    std::vector<uint32_t> contributor_rows(workers, 0u);
    std::vector<uint32_t> experts;
    uint64_t uid = 1u;
    for (uint32_t result = 0; result < results; ++result) {
        uint32_t destination = 0;
        if (!UniformBelow(workers, &destination) ||
            !RandomExperts(workers * 8u, topk, &experts)) {
            return false;
        }
        IncDcLogicalResultV2 logical_result{};
        logical_result.dst_rank = destination;
        logical_result.dst_local_row = destination_rows[destination]++;
        logical_result.contribution_begin =
            static_cast<uint32_t>(plan.contributions.size());
        logical_result.contribution_count = topk;
        plan.results.push_back(logical_result);
        for (uint32_t ordinal = 0; ordinal < topk; ++ordinal) {
            const uint32_t contributor = experts[ordinal] / 8u;
            IncDcLogicalContributionV2 contribution{};
            contribution.contribution_uid = uid++;
            contribution.result_id = result;
            contribution.ordinal = ordinal;
            contribution.contributor_rank = contributor;
            contribution.contributor_local_row =
                contributor_rows[contributor]++;
            contribution.weight = 1.0f;
            plan.contributions.push_back(contribution);
        }
    }
    plan.semantic_digest = ComputeLogicalPlanSemanticDigest(plan);
    std::vector<uint8_t> wire;
    IncDcLogicalPlanWireReport report{};
    if (SerializeLogicalPlanV2(plan, &wire, &report) != IncDcStatus::OK) {
        std::cerr << "serialize failed: " << report.first_error << '\n';
        return false;
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(wire.data()),
              static_cast<std::streamsize>(wire.size()));
    if (!out) return false;
    std::cout << "RANDOM_PLAN_WIRE semantic_digest=" << plan.semantic_digest
              << " wire_bytes=" << wire.size() << '\n';
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 6) {
        std::cerr << "usage: " << argv[0]
                  << " dispatch|combine WORKERS TOPK TOKENS_OR_RESULTS OUT\n";
        return 2;
    }
    uint32_t workers = 0, topk = 0, count = 0;
    if (!ParseU32(argv[2], &workers) || !ParseU32(argv[3], &topk) ||
        !ParseU32(argv[4], &count) || workers == 0u || count == 0u ||
        topk == 0u || topk > workers * 8u) {
        std::cerr << "invalid shape; require 1 <= topk <= workers*8\n";
        return 2;
    }
    bool ok = false;
    if (std::string(argv[1]) == "dispatch") {
        ok = WriteDispatch(workers, topk, count, argv[5]);
    } else if (std::string(argv[1]) == "combine") {
        ok = WriteCombine(workers, topk, count, argv[5]);
    } else {
        std::cerr << "operator must be dispatch or combine\n";
        return 2;
    }
    if (!ok) {
        std::cerr << "plan generation failed errno=" << errno << '\n';
        return 1;
    }
    std::cout << "RANDOM_PLAN operator=" << argv[1]
              << " entropy=getrandom_per_draw workers=" << workers
              << " topk=" << topk << " count=" << count
              << " output=" << argv[5] << '\n';
    return 0;
}
