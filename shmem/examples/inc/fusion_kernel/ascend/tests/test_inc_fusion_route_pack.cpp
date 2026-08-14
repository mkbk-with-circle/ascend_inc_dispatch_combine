#include "acl/acl.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "inc_fusion_plan.h"
#include "inc_fusion_route.h"
#include "inc_fusion_route_pack.h"

using namespace inc::fusion;

namespace {

struct DeviceBuffer {
    void *data = nullptr;
    size_t bytes = 0u;

    explicit DeviceBuffer(size_t requested) : bytes(requested)
    {
        if (bytes != 0u &&
            aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_HUGE_FIRST) !=
                ACL_SUCCESS)
            data = nullptr;
    }
    ~DeviceBuffer()
    {
        if (data != nullptr) aclrtFree(data);
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    bool valid() const { return bytes == 0u || data != nullptr; }
    bool FromHost(const void *source, size_t source_bytes)
    {
        return source_bytes <= bytes &&
            aclrtMemcpy(data, bytes, source, source_bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    }
    bool ToHost(void *destination, size_t destination_bytes) const
    {
        return destination_bytes <= bytes &&
            aclrtMemcpy(destination, destination_bytes, data,
                        destination_bytes, ACL_MEMCPY_DEVICE_TO_HOST) ==
                ACL_SUCCESS;
    }
};

template <typename T>
size_t Bytes(const std::vector<T> &values)
{
    return values.size() * sizeof(T);
}

template <typename T>
bool Exact(const char *name, const std::vector<T> &actual,
           const std::vector<T> &expected)
{
    if (actual.size() != expected.size() ||
        (Bytes(actual) != 0u &&
         std::memcmp(actual.data(), expected.data(), Bytes(actual)) != 0)) {
        std::cerr << name << " does not match CPU route compiler\n";
        return false;
    }
    return true;
}

FusionPlan MakePlan(const std::vector<uint32_t> &owner,
                    const std::vector<uint32_t> &local,
                    uint32_t token_count)
{
    FusionPlanConfig config{};
    config.live_aiv = 40u;
    config.live_aic = 20u;
    config.worker_count = 4u;
    config.rank = 0u;
    config.inc_pe = 4u;
    config.hidden = 256u;
    config.intermediate = 512u;
    config.expert_count = static_cast<uint32_t>(owner.size());
    config.topk = 3u;
    config.token_count = token_count;
    config.tokens_per_wave = 3u;
    config.slot_count = 3u;
    config.activation_waves = 2u;
    FusionPlan plan{};
    std::string error;
    if (!BuildFusionPlan(config, owner.data(), local.data(), &plan, &error)) {
        std::cerr << "BuildFusionPlan failed: " << error << "\n";
        return {};
    }
    return plan;
}

template <typename Id>
bool RunCase(uint32_t id_type, uint32_t active_tokens,
             uint32_t collective_tokens, uint32_t token_capacity,
             uint32_t lane_count, aclrtStream stream)
{
    const std::vector<uint32_t> owner = {2u, 0u, 3u, 2u, 0u, 2u, 1u};
    const std::vector<uint32_t> local = {0u, 0u, 0u, 1u, 1u, 2u, 0u};
    const FusionPlan plan = MakePlan(owner, local, token_capacity);
    const FusionPlan active_plan = MakePlan(owner, local, active_tokens);
    const FusionPlan collective_plan =
        MakePlan(owner, local, collective_tokens);
    if (plan.waves.empty()) return false;
    const FusionPlanConfig &config = plan.config;
    const size_t rank_stride =
        static_cast<size_t>(active_tokens) * config.topk;
    std::vector<uint32_t> ids32(
        static_cast<size_t>(config.worker_count) * rank_stride);
    std::vector<float> weights(ids32.size());
    for (uint32_t source = 0u; source < config.worker_count; ++source) {
        for (uint32_t token = 0u; token < active_tokens; ++token) {
            for (uint32_t ordinal = 0u; ordinal < config.topk; ++ordinal) {
                const size_t index = static_cast<size_t>(source) * rank_stride +
                    static_cast<size_t>(token) * config.topk + ordinal;
                ids32[index] =
                    (source * 5u + token * 3u + ordinal * 2u) %
                    config.expert_count;
                weights[index] = static_cast<float>(ordinal + 1u) / 6.0f +
                    static_cast<float>(source + token) / 1024.0f;
            }
        }
    }
    std::vector<Id> ids(ids32.begin(), ids32.end());
    FusionRouteBundle golden{};
    std::string error;
    if (!CompileFusionRoute(active_plan, ids32.data(), weights.data(), &golden,
                            &error)) {
        std::cerr << "CompileFusionRoute failed: " << error << "\n";
        return false;
    }

    const uint32_t wave_count = static_cast<uint32_t>(plan.waves.size());
    const uint32_t count_stride = config.expert_count + 1u;
    std::vector<uint32_t> global_counts(
        static_cast<size_t>(config.worker_count) * wave_count *
            count_stride,
        0u);
    for (uint32_t source = 0u; source < config.worker_count; ++source)
        for (uint32_t token = 0u; token < active_tokens; ++token)
            for (uint32_t ordinal = 0u; ordinal < config.topk; ++ordinal) {
                const uint32_t wave = token / config.tokens_per_wave;
                const uint32_t expert = ids32[
                    static_cast<size_t>(source) * rank_stride +
                    static_cast<size_t>(token) * config.topk + ordinal];
                ++global_counts[(static_cast<size_t>(source) * wave_count +
                                 wave) * count_stride + expert];
            }
    for (uint32_t source = 0u; source < config.worker_count; ++source)
        global_counts[static_cast<size_t>(source) * wave_count *
                          count_stride + config.expert_count] = active_tokens;

    DeviceBuffer dglobal(Bytes(global_counts));
    DeviceBuffer downer(Bytes(owner));
    DeviceBuffer dlocal(Bytes(local));
    if (!dglobal.valid() || !downer.valid() || !dlocal.valid() ||
        !dglobal.FromHost(global_counts.data(), Bytes(global_counts)) ||
        !downer.FromHost(owner.data(), Bytes(owner)) ||
        !dlocal.FromHost(local.data(), Bytes(local)))
        return false;

    for (uint32_t rank = 0u; rank < config.worker_count; ++rank) {
        FusionRankRoute expected = golden.ranks[rank];
        const std::vector<FusionWaveDesc> local_waves = expected.waves;
        expected.waves.clear();
        for (uint32_t wave = 0u; wave < wave_count; ++wave) {
            FusionWaveDesc descriptor = plan.waves[wave];
            descriptor.token_count = wave < collective_plan.waves.size()
                ? collective_plan.waves[wave].token_count : 0u;
            if (wave < local_waves.size()) {
                descriptor.dispatch_row_begin =
                    local_waves[wave].dispatch_row_begin;
                descriptor.dispatch_row_count =
                    local_waves[wave].dispatch_row_count;
                descriptor.assignment_begin =
                    local_waves[wave].assignment_begin;
                descriptor.assignment_count =
                    local_waves[wave].assignment_count;
            } else {
                descriptor.dispatch_row_begin =
                    static_cast<uint32_t>(expected.dispatch_rows.size());
                descriptor.assignment_begin =
                    static_cast<uint32_t>(expected.assignments.size());
            }
            expected.waves.push_back(descriptor);
        }
        std::vector<Id> rank_ids(ids.begin() + rank * rank_stride,
                                 ids.begin() + (rank + 1u) * rank_stride);
        std::vector<float> rank_weights(
            weights.begin() + rank * rank_stride,
            weights.begin() + (rank + 1u) * rank_stride);
        std::vector<uint32_t> local_counts(
            static_cast<size_t>(wave_count) * count_stride);
        std::vector<FusionDispatchRow> rows(expected.dispatch_rows.size());
        std::vector<FusionExpertAssignment> assignments(
            expected.assignments.size());
        std::vector<int64_t> groups(
            static_cast<size_t>(wave_count) *
            plan.local_expert_counts[rank]);
        std::vector<FusionWaveDesc> waves(wave_count);
        FusionRoutePackStatus status{};

        DeviceBuffer dids(Bytes(rank_ids));
        DeviceBuffer dweights(Bytes(rank_weights));
        DeviceBuffer dcounts(Bytes(local_counts));
        DeviceBuffer drows(Bytes(rows));
        DeviceBuffer dassignments(Bytes(assignments));
        DeviceBuffer dgroups(Bytes(groups));
        DeviceBuffer dwaves(Bytes(waves));
        DeviceBuffer dscratch(FusionRoutePackScratchBytes(
            config.expert_count));
        DeviceBuffer dstatus(sizeof(status));
        if (!dids.valid() || !dweights.valid() || !dcounts.valid() ||
            !drows.valid() || !dassignments.valid() || !dgroups.valid() ||
            !dwaves.valid() || !dscratch.valid() || !dstatus.valid() ||
            !dids.FromHost(rank_ids.data(), Bytes(rank_ids)) ||
            !dweights.FromHost(rank_weights.data(), Bytes(rank_weights)))
            return false;

        launch_inc_fusion_route_count_kernel(
            dids.data, id_type, active_tokens, config.topk,
            config.expert_count, config.tokens_per_wave, wave_count,
            static_cast<uint32_t *>(dcounts.data),
            static_cast<FusionRoutePackStatus *>(dstatus.data), stream);
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS ||
            !dstatus.ToHost(&status, sizeof(status)) ||
            status.error != kFusionRoutePackOk ||
            !dcounts.ToHost(local_counts.data(), Bytes(local_counts))) {
            std::cerr << "count kernel failed for rank " << rank
                      << ", error=" << status.error << "\n";
            return false;
        }
        const auto expected_counts_begin = global_counts.begin() +
            static_cast<size_t>(rank) * wave_count * count_stride;
        const std::vector<uint32_t> expected_counts(
            expected_counts_begin,
            expected_counts_begin + local_counts.size());
        if (!Exact("local counts", local_counts, expected_counts))
            return false;

        launch_inc_fusion_route_pack_kernel(
            dids.data, id_type, static_cast<const float *>(dweights.data),
            static_cast<const uint32_t *>(dglobal.data),
            static_cast<const uint32_t *>(downer.data),
            static_cast<const uint32_t *>(dlocal.data), config.worker_count,
            rank, active_tokens, config.topk,
            config.hidden,
            config.element_bytes, config.expert_count,
            plan.local_expert_counts[rank], config.tokens_per_wave,
            config.slot_count, config.activation_waves,
            static_cast<FusionDispatchRow *>(drows.data),
            static_cast<uint32_t>(rows.size()),
            static_cast<FusionExpertAssignment *>(dassignments.data),
            static_cast<uint32_t>(assignments.size()),
            static_cast<int64_t *>(dgroups.data),
            static_cast<FusionWaveDesc *>(dwaves.data), wave_count,
            static_cast<uint32_t *>(dscratch.data),
            static_cast<FusionRoutePackStatus *>(dstatus.data), stream);
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS ||
            !dstatus.ToHost(&status, sizeof(status)) ||
            status.error != kFusionRoutePackOk ||
            status.dispatch_row_count != rows.size() ||
            status.assignment_count != assignments.size() ||
            status.wave_count != wave_count ||
            !drows.ToHost(rows.data(), Bytes(rows)) ||
            !dassignments.ToHost(assignments.data(), Bytes(assignments)) ||
            !dgroups.ToHost(groups.data(), Bytes(groups)) ||
            !dwaves.ToHost(waves.data(), Bytes(waves))) {
            std::cerr << "pack kernel failed for rank " << rank
                      << ", error=" << status.error << "\n";
            return false;
        }
        std::vector<int64_t> expected_groups;
        for (uint32_t wave = 0u; wave < wave_count; ++wave) {
            if (wave < golden.group_lists[rank].size()) {
                const auto &one_wave = golden.group_lists[rank][wave];
                expected_groups.insert(expected_groups.end(),
                                       one_wave.begin(), one_wave.end());
            } else {
                expected_groups.insert(expected_groups.end(),
                    plan.local_expert_counts[rank], int64_t{0});
            }
        }
        if (!Exact("dispatch rows", rows, expected.dispatch_rows) ||
            !Exact("assignments", assignments, expected.assignments) ||
            !Exact("group lists", groups, expected_groups) ||
            !Exact("wave descriptors", waves, expected.waves))
            return false;

        DeviceBuffer dparallel_scratch(
            FusionRoutePackParallelScratchBytes(
                wave_count, config.expert_count, lane_count));
        if (!dparallel_scratch.valid()) return false;
        launch_inc_fusion_route_pack_parallel_kernel(
            dids.data, id_type, static_cast<const float *>(dweights.data),
            static_cast<const uint32_t *>(dglobal.data),
            static_cast<const uint32_t *>(downer.data),
            static_cast<const uint32_t *>(dlocal.data), config.worker_count,
            rank, active_tokens, config.topk, config.hidden,
            config.element_bytes, config.expert_count,
            plan.local_expert_counts[rank], config.tokens_per_wave,
            config.slot_count, config.activation_waves,
            static_cast<FusionDispatchRow *>(drows.data),
            static_cast<uint32_t>(rows.size()),
            static_cast<FusionExpertAssignment *>(dassignments.data),
            static_cast<uint32_t>(assignments.size()),
            static_cast<int64_t *>(dgroups.data),
            static_cast<FusionWaveDesc *>(dwaves.data), wave_count,
            lane_count, static_cast<uint32_t *>(dparallel_scratch.data),
            static_cast<FusionRoutePackStatus *>(dstatus.data), stream);
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS ||
            !dstatus.ToHost(&status, sizeof(status)) ||
            status.error != kFusionRoutePackOk ||
            status.dispatch_row_count != rows.size() ||
            status.assignment_count != assignments.size() ||
            status.wave_count != wave_count ||
            !drows.ToHost(rows.data(), Bytes(rows)) ||
            !dassignments.ToHost(assignments.data(), Bytes(assignments)) ||
            !dgroups.ToHost(groups.data(), Bytes(groups)) ||
            !dwaves.ToHost(waves.data(), Bytes(waves))) {
            std::cerr << "parallel pack kernel failed for rank " << rank
                      << ", error=" << status.error << "\n";
            return false;
        }
        if (!Exact("parallel dispatch rows", rows,
                   expected.dispatch_rows) ||
            !Exact("parallel assignments", assignments,
                   expected.assignments) ||
            !Exact("parallel group lists", groups, expected_groups) ||
            !Exact("parallel wave descriptors", waves, expected.waves))
            return false;
    }
    return true;
}

} // namespace

int main()
{
    const aclError init = aclInit(nullptr);
    if (init != ACL_SUCCESS && init != ACL_ERROR_REPEAT_INITIALIZE) {
        std::cerr << "aclInit failed: " << init << "\n";
        return 1;
    }
    if (aclrtSetDevice(0) != ACL_SUCCESS) return 1;
    int64_t live_aiv = 0;
    if (aclrtGetDeviceInfo(0, ACL_DEV_ATTR_VECTOR_CORE_NUM, &live_aiv) !=
            ACL_SUCCESS ||
        live_aiv <= 0 || live_aiv > kFusionMaxAiv)
        return 1;
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) return 1;
    const bool ok32 =
        RunCase<uint32_t>(kFusionRouteInt32, 7u, 7u, 7u,
                          static_cast<uint32_t>(live_aiv), stream);
    const bool ok64 = ok32 &&
        RunCase<int64_t>(kFusionRouteInt64, 7u, 7u, 7u,
                         static_cast<uint32_t>(live_aiv), stream);
    const bool ok_capacity = ok64 &&
        RunCase<uint32_t>(kFusionRouteInt32, 5u, 5u, 8u,
                          static_cast<uint32_t>(live_aiv), stream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    if (!ok_capacity) return 1;
    std::cout << "scalar and parallel device route-pack match CPU compiler "
                 "for int32/int64 and inactive capacity waves\n";
    return 0;
}
