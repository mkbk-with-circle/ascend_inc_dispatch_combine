#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "param.h"
#include "shmem.h"
#include "utils.h"

#include "inc_fusion_plan.h"
#include "inc_fusion_route.h"
#include "inc_fusion_api.h"

using namespace inc::fusion;

int g_npus = 16;
int f_npu = 0;
const char *ipport = nullptr;
aclshmemx_uniqueid_t default_flag_uid;

extern "C" void launch_inc_fusion_worker_kernel(
    uint8_t *sym, FusionKernelArgs *args, uint64_t expected_ticket,
    int block_dim, void *stream);
extern "C" void launch_inc_fusion_inc_service_kernel(
    uint8_t *sym, FusionKernelArgs *args, int block_dim, void *stream);

namespace {

uint16_t ToBf16(float value)
{
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<uint16_t>((bits + rounding) >> 16u);
}

float FromBf16(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16u;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int ResolveDevice(int pe)
{
    const char *raw = std::getenv("INC_PE_TO_NPU_MAP");
    if (raw != nullptr && raw[0] != '\0') {
        const std::string map(raw);
        size_t begin = 0u;
        while (begin < map.size()) {
            const size_t end = map.find(',', begin);
            const std::string item = map.substr(
                begin, end == std::string::npos ? std::string::npos :
                                                   end - begin);
            const size_t colon = item.find(':');
            if (colon != std::string::npos &&
                std::atoi(item.substr(0u, colon).c_str()) == pe)
                return std::atoi(item.substr(colon + 1u).c_str());
            if (end == std::string::npos) break;
            begin = end + 1u;
        }
    }
    return pe;
}

float InputValue(uint32_t rank, uint32_t token, uint32_t column)
{
    if (std::getenv("FUSION_IDENTICAL_SOURCES") != nullptr) rank = 0u;
    if (std::getenv("FUSION_RANDOM_DATA") != nullptr) {
        uint32_t value = rank * 0x9e3779b9u ^ token * 0x85ebca6bu ^
                         column * 0xc2b2ae35u ^ 0x243f6a88u;
        value ^= value >> 16u;
        value *= 0x7feb352du;
        value ^= value >> 15u;
        return static_cast<float>(static_cast<int32_t>(value % 257u) - 128) /
               256.0f;
    }
    return static_cast<float>(
        (rank * 17u + token * 7u + column * 3u) % 11u + 1u) / 16.0f;
}

float W13Value(uint32_t expert, uint32_t column, uint32_t inner)
{
    if (std::getenv("FUSION_RANDOM_DATA") != nullptr) {
        uint32_t value = expert * 0x27d4eb2du ^ column * 0x165667b1u ^
                         inner * 0xd3a2646cu ^ 0x13198a2eu;
        value ^= value >> 15u;
        value *= 0x846ca68bu;
        value ^= value >> 16u;
        return static_cast<float>(static_cast<int32_t>(value % 129u) - 64) /
               1024.0f;
    }
    return static_cast<float>(
        (expert * 11u + column * 5u + inner) % 7u + 1u) / 128.0f;
}

float W2Value(uint32_t expert, uint32_t column, uint32_t inner)
{
    if (std::getenv("FUSION_RANDOM_DATA") != nullptr) {
        uint32_t value = expert * 0x94d049bbu ^ column * 0x369dea0fu ^
                         inner * 0x7f4a7c15u ^ 0xa4093822u;
        value ^= value >> 16u;
        value *= 0x45d9f3bu;
        value ^= value >> 16u;
        return static_cast<float>(static_cast<int32_t>(value % 129u) - 64) /
               1024.0f;
    }
    return static_cast<float>(
        (expert * 13u + column * 3u + inner * 5u) % 7u + 1u) / 128.0f;
}

template <class T>
bool DeviceCopy(const std::vector<T> &host, void **device)
{
    if (host.empty()) {
        *device = nullptr;
        return true;
    }
    const size_t bytes = host.size() * sizeof(T);
    return aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
           aclrtMemcpy(*device, bytes, host.data(), bytes,
                       ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

void FreeAll(std::vector<void *> &buffers)
{
    for (void *buffer : buffers)
        if (buffer != nullptr) (void)aclrtFree(buffer);
    buffers.clear();
}

bool ParseActiveTokens(uint32_t workers, uint32_t capacity,
                       std::vector<uint32_t> *active)
{
    active->assign(workers, capacity);
    const char *raw = std::getenv("FUSION_ACTIVE_TOKENS");
    if (raw == nullptr || raw[0] == '\0') return true;
    const std::string text(raw);
    size_t begin = 0u;
    for (uint32_t rank = 0u; rank < workers; ++rank) {
        const size_t end = text.find(',', begin);
        if (begin >= text.size() ||
            (end == std::string::npos && rank + 1u != workers))
            return false;
        const std::string item = text.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        char *tail = nullptr;
        const unsigned long value = std::strtoul(item.c_str(), &tail, 10);
        if (tail == item.c_str() || *tail != '\0' || value == 0u ||
            value > capacity)
            return false;
        (*active)[rank] = static_cast<uint32_t>(value);
        begin = end == std::string::npos ? text.size() : end + 1u;
    }
    return begin == text.size();
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 9 && argc != 11) {
        std::cerr << "usage: fusion_e2e NPES PE IPPORT NPUS FIRST_NPU "
                     "TOKENS HIDDEN INTERMEDIATE [TOPK ACT_WAVES]\n";
        return 2;
    }
    const int npes = std::atoi(argv[1]);
    const int pe = std::atoi(argv[2]);
    ipport = argv[3];
    g_npus = std::atoi(argv[4]);
    f_npu = std::atoi(argv[5]);
    const uint32_t tokens = std::strtoul(argv[6], nullptr, 10);
    const uint32_t hidden = std::strtoul(argv[7], nullptr, 10);
    const uint32_t intermediate = std::strtoul(argv[8], nullptr, 10);
    const bool direct_shmem =
        std::getenv("FUSION_DIRECT_SHMEM") != nullptr &&
        std::atoi(std::getenv("FUSION_DIRECT_SHMEM")) != 0;
    const uint32_t workers = npes > 0
        ? static_cast<uint32_t>(npes - (direct_shmem ? 0 : 1)) : 0u;
    const uint32_t topk = argc == 11
        ? std::strtoul(argv[9], nullptr, 10) : 2u;
    const uint32_t activation_waves = argc == 11
        ? std::strtoul(argv[10], nullptr, 10) : 2u;
    const char *expert_count_env = std::getenv("FUSION_EXPERT_COUNT");
    const uint32_t experts = expert_count_env == nullptr
        ? workers * 2u
        : static_cast<uint32_t>(std::strtoul(expert_count_env, nullptr, 10));
    if (workers < 2u || workers > 4u || pe < 0 || pe >= npes ||
        tokens < 2u || hidden == 0u || intermediate == 0u ||
        topk == 0u || topk > experts || experts % workers != 0u ||
        activation_waves == 0u)
        return 2;
    const char *wave_tokens_env = std::getenv("FUSION_TOKENS_PER_WAVE");
    const uint32_t wave_tokens = wave_tokens_env == nullptr
        ? std::max(1u, tokens / 4u)
        : static_cast<uint32_t>(
              std::strtoul(wave_tokens_env, nullptr, 10));
    if (wave_tokens == 0u || wave_tokens > tokens) return 2;

    int rc = 0;
    int device = ResolveDevice(pe);
    aclrtStream stream = nullptr;
    aclrtStream persistent_stream = nullptr;
    inc_fusion_prepared_plan_t *persistent_plan = nullptr;
    inc_fusion_persistent_service_t *persistent_service = nullptr;
    inc_fusion_prepared_plan_t *worker_plan = nullptr;
    inc_fusion_worker_executor_t *worker_executor = nullptr;
    uint8_t *sym = nullptr;
    std::vector<void *> device_buffers;
    const aclError init = aclInit(nullptr);
    if ((init != ACL_SUCCESS && init != ACL_ERROR_REPEAT_INITIALIZE) ||
        aclrtSetDevice(device) != ACL_SUCCESS ||
        aclrtCreateStream(&stream) != ACL_SUCCESS) {
        std::cerr << "ACL initialization failed\n";
        return 3;
    }
    aclshmemx_init_attr_t attr{};
    const char *heap_bytes_env = std::getenv("FUSION_SHMEM_HEAP_BYTES");
    const uint64_t capacity = heap_bytes_env == nullptr
        ? 1024ull * 1024ull * 1024ull
        : std::strtoull(heap_bytes_env, nullptr, 10);
    if (capacity == 0u || capacity > ACLSHMEM_MAX_LOCAL_SIZE) {
        std::cerr << "invalid FUSION_SHMEM_HEAP_BYTES=" << capacity
                  << "\n";
        return 2;
    }
    test_set_attr(pe, npes, capacity, ipport, default_flag_uid, &attr);
    if (aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr) != 0) {
        std::cerr << "SHMEM initialization failed\n";
        rc = 3;
        goto cleanup;
    }

    {
        int64_t live_aiv64 = 0;
        int64_t live_aic64 = 0;
        if (aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                               &live_aiv64) != ACL_SUCCESS ||
            aclrtGetDeviceInfo(device, ACL_DEV_ATTR_CUBE_CORE_NUM,
                               &live_aic64) != ACL_SUCCESS ||
            live_aiv64 <= 0 || live_aic64 <= 0) {
            rc = 4;
            goto shmem_cleanup;
        }
        FusionPlanConfig config{};
        config.live_aiv = static_cast<uint32_t>(live_aiv64);
        config.live_aic = static_cast<uint32_t>(live_aic64);
        config.worker_count = workers;
        config.rank = static_cast<uint32_t>(pe) < workers
            ? static_cast<uint32_t>(pe) : 0u;
        config.inc_pe = workers;
        config.hidden = hidden;
        config.intermediate = intermediate;
        config.expert_count = experts;
        config.topk = topk;
        config.token_count = tokens;
        config.tokens_per_wave = wave_tokens;
        config.slot_count = 3u;
        config.service_ring_size = static_cast<uint32_t>(std::max(
            2, std::min(64,
                std::getenv("FUSION_SERVICE_RING") == nullptr ? 4 :
                std::atoi(std::getenv("FUSION_SERVICE_RING")))));
        config.activation_waves = activation_waves;
        const char *spin_cap = std::getenv("FUSION_SPIN_CAP");
        config.spin_cap = spin_cap == nullptr
            ? 0u : static_cast<uint32_t>(
                std::strtoul(spin_cap, nullptr, 10));
        std::vector<uint32_t> active_tokens;
        if (!ParseActiveTokens(workers, tokens, &active_tokens)) {
            std::cerr << "invalid FUSION_ACTIVE_TOKENS; expected "
                      << workers << " comma-separated values in [1,"
                      << tokens << "]\n";
            rc = 4;
            goto shmem_cleanup;
        }
        std::vector<uint32_t> owner(experts);
        std::vector<uint32_t> local(experts);
        std::vector<uint32_t> next(workers, 0u);
        const bool linear_placement =
            std::getenv("FUSION_LINEAR_PLACEMENT") != nullptr &&
            std::atoi(std::getenv("FUSION_LINEAR_PLACEMENT")) != 0;
        for (uint32_t expert = 0u; expert < experts; ++expert) {
            owner[expert] = linear_placement
                ? std::min(expert / (experts / workers), workers - 1u)
                : expert % workers;
            local[expert] = next[owner[expert]]++;
        }
        FusionPlan plan{};
        std::string error;
        if (!BuildFusionPlan(config, owner.data(), local.data(),
                             &plan, &error)) {
            std::cerr << "plan failed: " << error << "\n";
            rc = 4;
            goto shmem_cleanup;
        }
        const size_t route_count = static_cast<size_t>(workers) *
                                   tokens * topk;
        std::vector<uint32_t> expert_ids(route_count);
        std::vector<float> weights(route_count);
        for (uint32_t source = 0u; source < workers; ++source)
            for (uint32_t token = 0u; token < tokens; ++token)
                for (uint32_t ordinal = 0u; ordinal < topk; ++ordinal) {
                    const size_t index =
                        (static_cast<size_t>(source) * tokens + token) *
                            topk + ordinal;
                    const uint32_t route_source =
                        std::getenv("FUSION_IDENTICAL_SOURCES") == nullptr
                            ? source : 0u;
                    expert_ids[index] =
                        (route_source * 3u + token + ordinal) % experts;
                    const float denominator =
                        static_cast<float>(topk * (topk + 1u) / 2u);
                    weights[index] =
                        static_cast<float>(ordinal + 1u) / denominator;
                }
        FusionRouteBundle routes{};
        if (!CompileFusionRouteActive(
                plan, expert_ids.data(), weights.data(),
                active_tokens.data(), &routes, &error)) {
            std::cerr << "route failed: " << error << "\n";
            rc = 4;
            goto shmem_cleanup;
        }

        sym = static_cast<uint8_t *>(aclshmem_malloc(plan.symmetric.total_bytes));
        if (sym == nullptr ||
            aclrtMemset(sym, plan.symmetric.total_bytes, 0,
                        plan.symmetric.total_bytes) != ACL_SUCCESS) {
            std::cerr << "symmetric allocation failed: required="
                      << plan.symmetric.total_bytes
                      << " heap_capacity=" << capacity << "\n";
            rc = 5;
            goto shmem_cleanup;
        }
        const bool worker = static_cast<uint32_t>(pe) < workers;
        const bool global_output = worker && topk >= 2u &&
            std::getenv("FUSION_GLOBAL_OUTPUT") != nullptr &&
            std::atoi(std::getenv("FUSION_GLOBAL_OUTPUT")) != 0;
        const bool weight_b_row_major =
            std::getenv("FUSION_WEIGHT_B_ROW_MAJOR") != nullptr &&
            std::atoi(std::getenv("FUSION_WEIGHT_B_ROW_MAJOR")) != 0;
        const uint64_t workspace_bytes = worker
            ? plan.worker_workspace.total_bytes : plan.inc_workspace.total_bytes;
        void *workspace = nullptr;
        if (aclrtMalloc(&workspace, workspace_bytes,
                        ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            aclrtMemset(workspace, workspace_bytes, 0,
                        workspace_bytes) != ACL_SUCCESS) {
            rc = 5;
            goto shmem_cleanup;
        }
        device_buffers.push_back(workspace);

        std::vector<uint32_t> worker_pes(workers);
        for (uint32_t rank = 0u; rank < workers; ++rank)
            worker_pes[rank] = rank;
        void *d_worker_pes = nullptr;
        if (!DeviceCopy(worker_pes, &d_worker_pes)) {
            rc = 5; goto shmem_cleanup;
        }
        device_buffers.push_back(d_worker_pes);

        void *d_active_tokens = nullptr;
        if (!DeviceCopy(active_tokens, &d_active_tokens)) {
            rc = 5; goto shmem_cleanup;
        }
        device_buffers.push_back(d_active_tokens);
        void *d_owner = nullptr;
        void *d_local = nullptr;
        if (!DeviceCopy(owner, &d_owner) || !DeviceCopy(local, &d_local)) {
            rc = 5; goto shmem_cleanup;
        }
        device_buffers.push_back(d_owner);
        device_buffers.push_back(d_local);

        const std::vector<FusionWaveDesc> &host_waves = worker
            ? routes.ranks[pe].waves : routes.ranks[0].waves;
        void *d_waves = nullptr;
        if (!DeviceCopy(host_waves, &d_waves)) {
            rc = 5; goto shmem_cleanup;
        }
        device_buffers.push_back(d_waves);
        void *d_rows = nullptr, *d_assignments = nullptr;
        void *d_group_lists = nullptr, *d_input = nullptr, *d_output = nullptr;
        void *d_w13 = nullptr, *d_w2 = nullptr;
        std::vector<uint16_t> host_input;
        std::vector<uint16_t> host_output;
        if (worker) {
            const uint32_t rank_tokens = active_tokens[pe];
            if (!DeviceCopy(routes.ranks[pe].dispatch_rows, &d_rows) ||
                !DeviceCopy(routes.ranks[pe].assignments, &d_assignments)) {
                rc = 5; goto shmem_cleanup;
            }
            device_buffers.push_back(d_rows);
            device_buffers.push_back(d_assignments);
            std::vector<int64_t> group_lists;
            for (uint32_t wave = 0u; wave < plan.waves.size(); ++wave) {
                const auto &groups = routes.group_lists[pe][wave];
                group_lists.insert(group_lists.end(), groups.begin(), groups.end());
            }
            if (!DeviceCopy(group_lists, &d_group_lists)) {
                rc = 5; goto shmem_cleanup;
            }
            device_buffers.push_back(d_group_lists);
            host_input.resize(static_cast<size_t>(rank_tokens) * hidden);
            const uint64_t output_tokens = global_output
                ? std::accumulate(active_tokens.begin(), active_tokens.end(),
                                  uint64_t{0})
                : rank_tokens;
            host_output.resize(
                static_cast<size_t>(output_tokens) * hidden, 0u);
            for (uint32_t token = 0u; token < rank_tokens; ++token)
                for (uint32_t column = 0u; column < hidden; ++column)
                    host_input[static_cast<size_t>(token) * hidden + column] =
                        ToBf16(InputValue(pe, token, column));
            if (!DeviceCopy(host_input, &d_input) ||
                !DeviceCopy(host_output, &d_output)) {
                rc = 5; goto shmem_cleanup;
            }
            device_buffers.push_back(d_input);
            device_buffers.push_back(d_output);
            const uint32_t local_experts = plan.local_expert_counts[pe];
            std::vector<uint16_t> w13(
                static_cast<size_t>(local_experts) * 2u * intermediate * hidden);
            std::vector<uint16_t> w2(
                static_cast<size_t>(local_experts) * hidden * intermediate);
            for (uint32_t expert = 0u; expert < experts; ++expert) {
                if (owner[expert] != static_cast<uint32_t>(pe)) continue;
                const uint32_t le = local[expert];
                for (uint32_t column = 0u; column < 2u * intermediate; ++column)
                    for (uint32_t inner = 0u; inner < hidden; ++inner)
                        w13[weight_b_row_major
                            ? (static_cast<size_t>(le) * hidden + inner) *
                                  (2u * intermediate) + column
                            : (static_cast<size_t>(le) * 2u * intermediate +
                                  column) * hidden + inner] =
                            ToBf16(W13Value(expert, column, inner));
                for (uint32_t column = 0u; column < hidden; ++column)
                    for (uint32_t inner = 0u; inner < intermediate; ++inner)
                        w2[weight_b_row_major
                            ? (static_cast<size_t>(le) * intermediate + inner) *
                                  hidden + column
                            : (static_cast<size_t>(le) * hidden + column) *
                                  intermediate + inner] =
                            ToBf16(W2Value(expert, column, inner));
            }
            if (!DeviceCopy(w13, &d_w13) || !DeviceCopy(w2, &d_w2)) {
                rc = 5; goto shmem_cleanup;
            }
            device_buffers.push_back(d_w13);
            device_buffers.push_back(d_w2);
        }

        FusionKernelArgs args = MakeFusionKernelArgs(
            plan, worker ? kFusionWorker : kFusionInc, 100u);
        const char *disable_bulk = std::getenv("FUSION_DISABLE_BULK");
        if (disable_bulk != nullptr && std::atoi(disable_bulk) != 0)
            args.flags &= ~kFusionBulkWaveTransport;
        const char *serialize_dc = std::getenv("FUSION_SERIALIZE_DC");
        if (serialize_dc != nullptr && std::atoi(serialize_dc) != 0)
            args.flags |= kFusionSerializeIncDc;
        const char *strict_serial = std::getenv("FUSION_STRICT_SERIAL");
        if (strict_serial != nullptr && std::atoi(strict_serial) != 0)
            args.flags |= kFusionStrictSerialPipeline;
        if (direct_shmem)
            args.flags |= kFusionWorkerDirectShmem;
        if (weight_b_row_major)
            args.flags |= kFusionWeightBRowMajor;
        if (global_output)
            args.flags |= kFusionGlobalOutputFanout;
        args.symmetric_base = reinterpret_cast<uint64_t>(sym);
        args.input = reinterpret_cast<uint64_t>(d_input);
        args.output = reinterpret_cast<uint64_t>(d_output);
        args.w13 = reinterpret_cast<uint64_t>(d_w13);
        args.w2 = reinterpret_cast<uint64_t>(d_w2);
        args.dispatch_rows = reinterpret_cast<uint64_t>(d_rows);
        args.assignments = reinterpret_cast<uint64_t>(d_assignments);
        args.waves = reinterpret_cast<uint64_t>(d_waves);
        args.expert_owner = reinterpret_cast<uint64_t>(d_owner);
        args.expert_local_index = reinterpret_cast<uint64_t>(d_local);
        args.worker_pes = reinterpret_cast<uint64_t>(d_worker_pes);
        args.active_token_counts =
            reinterpret_cast<uint64_t>(d_active_tokens);
        args.group_lists = reinterpret_cast<uint64_t>(d_group_lists);
        args.workspace = reinterpret_cast<uint64_t>(workspace);
        args.ffts_addr = shmemx_get_ffts_config();
        void *d_args = nullptr;
        std::vector<FusionKernelArgs> one_arg(1u, args);
        if (!DeviceCopy(one_arg, &d_args)) {
            rc = 5; goto shmem_cleanup;
        }
        device_buffers.push_back(d_args);

        const bool remote_inc_mode =
            std::getenv("FUSION_REMOTE_INC") != nullptr &&
            std::atoi(std::getenv("FUSION_REMOTE_INC")) != 0;
        const bool persistent_mode = remote_inc_mode ||
            std::getenv("FUSION_PERSISTENT_INC") != nullptr &&
            std::atoi(std::getenv("FUSION_PERSISTENT_INC")) != 0;
        const bool use_persistent_inc = !worker && persistent_mode;
        const bool use_prepared_worker = worker && (direct_shmem || remote_inc_mode ||
            (std::getenv("FUSION_PREPARED_WORKER") != nullptr &&
             std::atoi(std::getenv("FUSION_PREPARED_WORKER")) != 0));
        uint32_t service_ring = 4u;
        if (use_prepared_worker) {
            inc_fusion_plan_desc_t worker_desc{};
            inc_fusion_plan_desc_init(&worker_desc);
            worker_desc.live_aiv = config.live_aiv;
            worker_desc.live_aic = config.live_aic;
            worker_desc.worker_count = config.worker_count;
            worker_desc.rank = config.rank;
            worker_desc.inc_pe = config.inc_pe;
            worker_desc.hidden = config.hidden;
            worker_desc.intermediate = config.intermediate;
            worker_desc.expert_count = config.expert_count;
            worker_desc.topk = config.topk;
            worker_desc.token_count = config.token_count;
            worker_desc.tokens_per_wave = config.tokens_per_wave;
            worker_desc.slot_count = config.slot_count;
            worker_desc.service_ring_size = config.service_ring_size;
            worker_desc.activation_waves = config.activation_waves;
            worker_desc.spin_cap = config.spin_cap;
            inc_fusion_device_bindings_t static_bindings{};
            static_bindings.symmetric_base = sym;
            static_bindings.expert_owner = d_owner;
            static_bindings.expert_local_index = d_local;
            static_bindings.worker_pes = d_worker_pes;
            static_bindings.active_token_counts = d_active_tokens;
            static_bindings.workspace = workspace;
            static_bindings.ffts_addr = shmemx_get_ffts_config();
            if (remote_inc_mode)
                static_bindings.execution_flags |=
                    INC_FUSION_EXEC_REMOTE_INC_SERVICE;
            if (global_output)
                static_bindings.execution_flags |=
                    INC_FUSION_EXEC_GLOBAL_OUTPUT_FANOUT;
            if (direct_shmem)
                static_bindings.execution_flags |=
                    INC_FUSION_EXEC_WORKER_DIRECT_SHMEM;
            const uint32_t worker_ring = static_cast<uint32_t>(std::max(
                2, std::min(64,
                    std::getenv("FUSION_WORKER_RING") == nullptr ? 4 :
                    std::atoi(std::getenv("FUSION_WORKER_RING")))));
            if (inc_fusion_prepared_plan_create(
                    &worker_desc, owner.data(), local.data(),
                    &worker_plan) != INC_FUSION_OK ||
                inc_fusion_worker_executor_create(
                    worker_plan, &static_bindings, worker_ring,
                    &worker_executor) != INC_FUSION_OK) {
                std::cerr << "prepared worker initialization failed\n";
                rc = 6;
                goto shmem_cleanup;
            }
        }
        if (use_persistent_inc) {
            inc_fusion_plan_desc_t service_desc{};
            inc_fusion_plan_desc_init(&service_desc);
            service_desc.live_aiv = config.live_aiv;
            service_desc.live_aic = config.live_aic;
            service_desc.worker_count = config.worker_count;
            service_desc.rank = config.rank;
            service_desc.inc_pe = config.inc_pe;
            service_desc.hidden = config.hidden;
            service_desc.intermediate = config.intermediate;
            service_desc.expert_count = config.expert_count;
            service_desc.topk = config.topk;
            service_desc.token_count = config.token_count;
            service_desc.tokens_per_wave = config.tokens_per_wave;
            service_desc.slot_count = config.slot_count;
            service_desc.service_ring_size = config.service_ring_size;
            service_desc.activation_waves = config.activation_waves;
            service_desc.spin_cap = config.spin_cap;
            service_ring = config.service_ring_size;
            if (inc_fusion_prepared_plan_create(
                    &service_desc, owner.data(), local.data(),
                    &persistent_plan) != INC_FUSION_OK ||
                aclrtCreateStream(&persistent_stream) != ACL_SUCCESS) {
                std::cerr << "persistent INC service initialization failed\n";
                rc = 6;
                goto shmem_cleanup;
            }
            // Remote startup is deliberately two phase.  Every PE first
            // initializes its local symmetric mirror, then the final setup
            // barrier below makes the ring visible everywhere, and only
            // afterwards may the INC occupy all of its AIVs persistently.
            // Creating the INC ring after that barrier can erase ticket one
            // when rank zero wins the startup race.
            if (remote_inc_mode) {
                inc_fusion_device_bindings_t service_bindings{};
                service_bindings.symmetric_base = sym;
                service_bindings.workspace = workspace;
                service_bindings.worker_pes = d_worker_pes;
                service_bindings.system_workspace = nullptr;
                service_bindings.ffts_addr = shmemx_get_ffts_config();
                if (inc_fusion_remote_service_create(
                        persistent_plan, &service_bindings,
                        persistent_stream, &persistent_service) !=
                    INC_FUSION_OK) {
                    std::cerr << "remote INC service preparation failed\n";
                    rc = 6;
                    goto shmem_cleanup;
                }
            }
        }

        float elapsed_ms = 0.0f;
        const uint32_t iterations = std::max(
            1, std::min(1000, std::getenv("FUSION_ITERATIONS") == nullptr
                ? 1 : std::atoi(std::getenv("FUSION_ITERATIONS"))));
        const uint32_t warmup = std::min<uint32_t>(
            iterations - 1u,
            std::getenv("FUSION_WARMUP") == nullptr
                ? 0u : static_cast<uint32_t>(std::max(
                    0, std::atoi(std::getenv("FUSION_WARMUP")))));
        std::vector<float> measured_us;
        std::vector<uint16_t> first_iteration_output;
        bool iteration_output_stable = true;
        for (uint32_t iteration = 0u;
             iteration < iterations && rc == 0; ++iteration) {
            args.operation_generation =
                100u + static_cast<uint64_t>(iteration) *
                    (plan.waves.size() + 1u);
            if (!use_prepared_worker &&
                aclrtMemcpy(d_args, sizeof(args), &args, sizeof(args),
                            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                rc = 6;
                break;
            }
            // A persistent kernel occupies every INC AIV, so no device-side
            // collective may be launched on the INC after it starts.  One
            // final setup barrier is sufficient; subsequent generations are
            // ordered by descriptor tickets and exact packet credits.
            if (!persistent_mode || iteration == 0u)
                aclshmem_barrier_all();
            if (use_persistent_inc) {
                inc_fusion_status_t create_status = INC_FUSION_OK;
                if (remote_inc_mode) {
                    create_status = inc_fusion_remote_service_start(
                        persistent_service);
                } else if (persistent_service == nullptr) {
                    create_status = inc_fusion_persistent_service_create(
                        persistent_plan, sym, service_ring,
                        persistent_stream, &persistent_service);
                }
                if (create_status != INC_FUSION_OK) {
                    std::cerr << "persistent INC service launch failed\n";
                    rc = 6;
                    break;
                }
                const auto host_begin = std::chrono::steady_clock::now();
                uint64_t ticket = static_cast<uint64_t>(iteration) + 1u;
                if (!remote_inc_mode) {
                    if (inc_fusion_persistent_service_submit(
                            persistent_service, d_args, ticket,
                            stream, &ticket) != INC_FUSION_OK ||
                        aclrtSynchronizeStream(stream) != ACL_SUCCESS)
                        rc = 6;
                }
                inc_fusion_service_result_t result{};
                uint64_t remote_polls = 0u;
                while (rc == 0 && result.complete == 0u) {
                    const inc_fusion_status_t status =
                        inc_fusion_persistent_service_query(
                            persistent_service, ticket, &result);
                    if (status != INC_FUSION_OK) {
                        rc = 6;
                        break;
                    }
                    if (remote_inc_mode &&
                        std::getenv("FUSION_REMOTE_DEBUG") != nullptr &&
                        (++remote_polls % 100000u) == 0u) {
                        FusionServiceDescriptor debug_descriptor{};
                        FusionKernelArgs debug_args{};
                        FusionServiceControl debug_control{};
                        std::vector<uint64_t> debug_lane_progress(
                            plan.resources.live_aiv, 0u);
                        const uint32_t debug_slot = static_cast<uint32_t>(
                            (ticket - 1u) %
                            plan.remote_service.ring_size);
                        auto *descriptor_ptr = sym +
                            plan.remote_service.descriptors_off +
                            static_cast<uint64_t>(debug_slot) *
                                sizeof(FusionServiceDescriptor);
                        auto *args_ptr = sym + plan.remote_service.args_off +
                            static_cast<uint64_t>(debug_slot) *
                                sizeof(FusionKernelArgs);
                        (void)aclrtMemcpy(&debug_descriptor,
                            sizeof(debug_descriptor), descriptor_ptr,
                            sizeof(debug_descriptor),
                            ACL_MEMCPY_DEVICE_TO_HOST);
                        (void)aclrtMemcpy(&debug_args, sizeof(debug_args),
                            args_ptr, sizeof(debug_args),
                            ACL_MEMCPY_DEVICE_TO_HOST);
                        (void)aclrtMemcpy(&debug_control,
                            sizeof(debug_control),
                            sym + plan.remote_service.control_off,
                            sizeof(debug_control),
                            ACL_MEMCPY_DEVICE_TO_HOST);
                        for (uint32_t lane = 0u;
                             lane < plan.resources.live_aiv; ++lane) {
                            (void)aclrtMemcpy(&debug_lane_progress[lane],
                                sizeof(uint64_t),
                                sym + plan.remote_service.lane_progress_off +
                                    static_cast<uint64_t>(lane) *
                                        kFusionCacheLineBytes,
                                sizeof(uint64_t),
                                ACL_MEMCPY_DEVICE_TO_HOST);
                        }
                        std::cerr << "REMOTE_DEBUG ticket=" << ticket
                                  << " ready=" << debug_descriptor.ready
                                  << " complete="
                                  << debug_descriptor.complete
                                  << " device_args=0x" << std::hex
                                  << debug_descriptor.device_args << std::dec
                                  << " generation="
                                  << debug_args.operation_generation
                                  << " flags=" << debug_args.flags
                                  << " service_completed="
                                  << debug_control.completed_sequence
                                  << " service_error="
                                  << debug_control.service_error
                                  << " lanes=";
                        for (uint32_t lane = 0u;
                             lane < debug_lane_progress.size(); ++lane) {
                            if (lane != 0u) std::cerr << ',';
                            std::cerr << debug_lane_progress[lane];
                        }
                        std::cerr << "\n";
                    }
                }
                if (rc == 0 &&
                    (result.request_id !=
                         static_cast<uint64_t>(iteration) + 1u ||
                     result.status != INC_FUSION_SERVICE_SUCCESS ||
                     result.operator_error != 0u)) {
                    std::cerr << "persistent request failed ticket=" << ticket
                              << " request_id=" << result.request_id
                              << " status=" << result.status
                              << " operator_error=" << result.operator_error
                              << "\n";
                    if (!worker &&
                        std::getenv("FUSION_REMOTE_DEBUG") != nullptr) {
                        std::vector<FusionSlotState> debug_states(
                            config.slot_count);
                        if (aclrtMemcpy(
                                debug_states.data(),
                                debug_states.size() *
                                    sizeof(FusionSlotState),
                                static_cast<uint8_t *>(workspace) +
                                    plan.inc_workspace.slot_state_off,
                                debug_states.size() *
                                    sizeof(FusionSlotState),
                                ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) {
                            for (uint32_t debug_slot = 0u;
                                 debug_slot < debug_states.size();
                                 ++debug_slot) {
                                const auto &state =
                                    debug_states[debug_slot];
                                std::cerr
                                    << "REMOTE_INC_SLOT slot="
                                    << debug_slot
                                    << " error=" << state.error
                                    << " site=" << state.owner
                                    << " D="
                                    << state.dispatch_generation
                                    << " C="
                                    << state.combine_generation
                                    << " O="
                                    << state.output_generation
                                    << " R="
                                    << state.release_generation
                                    << " detail=" << state.reserved
                                    << "\n";
                            }
                        }
                        const uint32_t remote_slot = static_cast<uint32_t>(
                            (ticket - 1u) %
                            plan.remote_service.ring_size);
                        std::vector<uint32_t> debug_active(workers, 0u);
                        std::vector<FusionWaveDesc> debug_waves(
                            plan.waves.size());
                        auto *remote_active = sym +
                            plan.remote_service.request_off +
                            static_cast<uint64_t>(remote_slot) *
                                plan.remote_service.request_stride +
                            (plan.remote_service.active_token_counts_off -
                             plan.remote_service.request_off);
                        auto *remote_waves = sym +
                            plan.remote_service.request_off +
                            static_cast<uint64_t>(remote_slot) *
                                plan.remote_service.request_stride +
                            (plan.remote_service.waves_off -
                             plan.remote_service.request_off);
                        if (aclrtMemcpy(
                                debug_active.data(),
                                debug_active.size() * sizeof(uint32_t),
                                remote_active,
                                debug_active.size() * sizeof(uint32_t),
                                ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS &&
                            aclrtMemcpy(
                                debug_waves.data(),
                                debug_waves.size() *
                                    sizeof(FusionWaveDesc),
                                remote_waves,
                                debug_waves.size() *
                                    sizeof(FusionWaveDesc),
                                ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) {
                            std::cerr << "REMOTE_INC_DYNAMIC active=";
                            for (uint32_t rank = 0u; rank < workers;
                                 ++rank) {
                                if (rank != 0u) std::cerr << ',';
                                std::cerr << debug_active[rank];
                            }
                            std::cerr << " waves=";
                            for (uint32_t wave = 0u;
                                 wave < debug_waves.size(); ++wave) {
                                if (wave != 0u) std::cerr << ';';
                                std::cerr
                                    << debug_waves[wave].token_begin
                                    << '+'
                                    << debug_waves[wave].token_count
                                    << '@' << debug_waves[wave].slot;
                            }
                            std::cerr << "\n";
                        }
                    }
                    rc = 7;
                }
                const auto host_end = std::chrono::steady_clock::now();
                const float iteration_ms =
                    std::chrono::duration<float, std::milli>(
                        host_end - host_begin).count();
                if (iteration >= warmup)
                    measured_us.push_back(iteration_ms * 1000.0f);
                continue;
            }
            aclrtEvent start_event = nullptr;
            aclrtEvent end_event = nullptr;
            float iteration_ms = 0.0f;
            if (aclrtCreateEvent(&start_event) != ACL_SUCCESS ||
                aclrtCreateEvent(&end_event) != ACL_SUCCESS ||
                aclrtRecordEvent(start_event, stream) != ACL_SUCCESS) {
                rc = 6;
            }
            if (rc == 0 && worker && worker_executor != nullptr) {
                inc_fusion_device_bindings_t dynamic_bindings{};
                dynamic_bindings.input = d_input;
                dynamic_bindings.output = d_output;
                dynamic_bindings.w13 = d_w13;
                dynamic_bindings.w2 = d_w2;
                dynamic_bindings.dispatch_rows = d_rows;
                dynamic_bindings.assignments = d_assignments;
                dynamic_bindings.waves = d_waves;
                dynamic_bindings.active_token_counts = d_active_tokens;
                dynamic_bindings.group_lists = d_group_lists;
                if ((args.flags & kFusionSerializeIncDc) != 0u)
                    dynamic_bindings.execution_flags =
                        INC_FUSION_EXEC_SERIALIZE_INC_DC;
                if ((args.flags & kFusionStrictSerialPipeline) != 0u)
                    dynamic_bindings.execution_flags |=
                        INC_FUSION_EXEC_STRICT_SERIAL_PIPELINE;
                if ((args.flags & kFusionWorkerDirectShmem) != 0u)
                    dynamic_bindings.execution_flags |=
                        INC_FUSION_EXEC_WORKER_DIRECT_SHMEM;
                if ((args.flags & kFusionWeightBRowMajor) != 0u)
                    dynamic_bindings.execution_flags |=
                        INC_FUSION_EXEC_WEIGHT_B_ROW_MAJOR;
                if ((args.flags & kFusionGlobalOutputFanout) != 0u)
                    dynamic_bindings.execution_flags |=
                        INC_FUSION_EXEC_GLOBAL_OUTPUT_FANOUT;
                if (inc_fusion_worker_executor_enqueue(
                        worker_executor, args.operation_generation,
                        &dynamic_bindings, stream) != INC_FUSION_OK)
                    rc = 6;
                if (rc == 0 && remote_inc_mode &&
                    std::getenv("FUSION_REMOTE_DEBUG") != nullptr) {
                    inc_fusion_worker_executor_info_t debug_info{};
                    if (inc_fusion_worker_executor_info(
                            worker_executor, &debug_info) == INC_FUSION_OK &&
                        debug_info.host_args != nullptr) {
                        const uint32_t debug_slot = static_cast<uint32_t>(
                            (debug_info.submitted - 1u) %
                            debug_info.ring_size);
                        const auto *debug_args =
                            reinterpret_cast<const FusionKernelArgs *>(
                                debug_info.host_args) + debug_slot;
                        std::cerr << "REMOTE_WORKER_DEBUG pe=" << pe
                                  << " rank=" << debug_args->rank
                                  << " flags=" << debug_args->flags
                                  << " ticket="
                                  << debug_args->service_ticket
                                  << " generation="
                                  << debug_args->operation_generation
                                  << " descriptor_off="
                                  << debug_args->remote_service.descriptors_off
                                  << "\n";
                    }
                }
            } else if (rc == 0 && worker) {
                launch_inc_fusion_worker_kernel(
                    sym, static_cast<FusionKernelArgs *>(d_args),
                    0u, static_cast<int>(live_aic64), stream);
            }
            else if (rc == 0)
                launch_inc_fusion_inc_service_kernel(
                    sym, static_cast<FusionKernelArgs *>(d_args),
                    static_cast<int>(live_aiv64), stream);
            if (rc == 0 &&
                std::getenv("FUSION_LIVE_WATCHDOG") != nullptr) {
                const uint64_t state_off = worker
                    ? plan.worker_workspace.slot_state_off
                    : plan.inc_workspace.slot_state_off;
                const uint32_t debug_slots = config.slot_count;
                std::thread([workspace, state_off, debug_slots, device, pe]() {
                    std::this_thread::sleep_for(std::chrono::seconds(3));
                    if (aclrtSetDevice(device) != ACL_SUCCESS) return;
                    std::vector<FusionSlotState> states(debug_slots);
                    const aclError copy = aclrtMemcpy(
                        states.data(), states.size() * sizeof(FusionSlotState),
                        static_cast<uint8_t *>(workspace) + state_off,
                        states.size() * sizeof(FusionSlotState),
                        ACL_MEMCPY_DEVICE_TO_HOST);
                    std::cerr << "FUSION_LIVE_WATCHDOG pe=" << pe
                              << " copy=" << copy;
                    if (copy == ACL_SUCCESS) {
                        for (uint32_t slot = 0u; slot < debug_slots; ++slot) {
                            const auto &state = states[slot];
                            std::cerr << " slot" << slot
                                      << "[err=" << state.error
                                      << ",site=" << state.owner
                                      << ",claim=" << state.claim_generation
                                      << ",d=" << state.dispatch_generation
                                      << ",c=" << state.compute_generation
                                      << ",combine=" << state.combine_generation
                                      << ",out=" << state.output_generation
                                      << ",release=" << state.release_generation
                                      << ",detail=" << state.reserved << ']';
                        }
                    }
                    std::cerr << "\n";
                }).detach();
            }
            if (rc == 0 &&
                (aclrtRecordEvent(end_event, stream) != ACL_SUCCESS ||
                 aclrtSynchronizeStream(stream) != ACL_SUCCESS ||
                 aclrtEventElapsedTime(&iteration_ms, start_event,
                                       end_event) != ACL_SUCCESS)) {
                std::cerr << "kernel synchronization failed pe=" << pe
                          << " iteration=" << iteration << "\n";
                rc = 6;
            }
            if (rc == 0 && worker &&
                std::getenv("FUSION_VALIDATE_EACH_ITERATION") != nullptr) {
                std::vector<uint16_t> iteration_output(host_output.size());
                if (aclrtMemcpy(iteration_output.data(),
                                iteration_output.size() * sizeof(uint16_t),
                                d_output,
                                iteration_output.size() * sizeof(uint16_t),
                                ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                    rc = 6;
                } else {
                    uint64_t hash = 1469598103934665603ull;
                    double iteration_checksum = 0.0;
                    for (uint16_t value : iteration_output) {
                        hash ^= value;
                        hash *= 1099511628211ull;
                        iteration_checksum += FromBf16(value);
                    }
                    const bool same = first_iteration_output.empty() ||
                        iteration_output == first_iteration_output;
                    if (first_iteration_output.empty())
                        first_iteration_output = iteration_output;
                    std::cout << "FUSION_ITERATION_OUTPUT pe=" << pe
                              << " iteration=" << iteration
                              << " checksum=" << iteration_checksum
                              << " hash=" << hash
                              << " same_as_first=" << (same ? 1 : 0)
                              << "\n";
                    if (!same) iteration_output_stable = false;
                    if (plan.waves.size() == 1u) {
                        const uint32_t wave_slot = routes.ranks[pe].waves[0].slot;
                        const uint64_t row_bytes =
                            static_cast<uint64_t>(hidden) * sizeof(uint16_t);
                        const uint64_t capacity =
                            plan.worker_workspace.assignment_slot_bytes /
                            row_bytes;
                        const uint32_t rows = static_cast<uint32_t>(
                            routes.group_lists[pe][0].back());
                        auto copy_hash = [&](uint64_t offset,
                                             uint64_t bytes) -> uint64_t {
                            std::vector<uint8_t> values(bytes);
                            if (aclrtMemcpy(values.data(), values.size(),
                                    static_cast<uint8_t *>(workspace) + offset,
                                    values.size(),
                                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                                rc = 6;
                                return 0u;
                            }
                            uint64_t value = 1469598103934665603ull;
                            for (uint8_t byte : values) {
                                value ^= byte;
                                value *= 1099511628211ull;
                            }
                            return value;
                        };
                        const uint64_t grouped_hash = copy_hash(
                            plan.worker_workspace.grouped_input_off +
                                static_cast<uint64_t>(wave_slot) *
                                    plan.worker_workspace.assignment_slot_bytes,
                            static_cast<uint64_t>(rows) * row_bytes);
                        const uint64_t gate_hash = copy_hash(
                            plan.worker_workspace.gate_up_off +
                                static_cast<uint64_t>(wave_slot) * capacity *
                                    2u * intermediate * sizeof(uint16_t),
                            static_cast<uint64_t>(rows) * 2u * intermediate *
                                sizeof(uint16_t));
                        const uint64_t activation_hash = copy_hash(
                            plan.worker_workspace.activation_off +
                                static_cast<uint64_t>(wave_slot) * capacity *
                                    intermediate * sizeof(uint16_t),
                            static_cast<uint64_t>(rows) * intermediate *
                                sizeof(uint16_t));
                        const uint64_t expert_hash = copy_hash(
                            plan.worker_workspace.expert_output_off +
                                static_cast<uint64_t>(wave_slot) *
                                    plan.worker_workspace.assignment_slot_bytes,
                            static_cast<uint64_t>(rows) * row_bytes);
                        std::cout << "FUSION_ITERATION_STAGES pe=" << pe
                                  << " iteration=" << iteration
                                  << " grouped_hash=" << grouped_hash
                                  << " gate_hash=" << gate_hash
                                  << " activation_hash=" << activation_hash
                                  << " expert_hash=" << expert_hash << "\n";
                    }
                }
            }
            if (rc == 0 && !worker &&
                std::getenv("FUSION_VALIDATE_EACH_ITERATION") != nullptr &&
                plan.waves.size() == 1u) {
                const uint32_t wave_slot = plan.waves[0].slot;
                const uint64_t expert_stride =
                    static_cast<uint64_t>(config.tokens_per_wave) *
                    config.worker_count * config.topk *
                    (static_cast<uint64_t>(config.hidden) *
                         sizeof(uint16_t) +
                     sizeof(FusionExpertAssignment));
                const uint64_t control_lines =
                    static_cast<uint64_t>(4u) * config.worker_count +
                    static_cast<uint64_t>(config.worker_count) *
                        config.worker_count +
                    static_cast<uint64_t>(2u) *
                        plan.resources.inc_combine_aiv +
                    static_cast<uint64_t>(5u) * config.worker_count +
                    plan.resources.inc_dispatch_aiv;
                std::vector<std::vector<uint16_t>> dense_inputs(
                    config.worker_count);
                for (uint32_t producer = 0u;
                     producer < config.worker_count; ++producer) {
                    FusionBulkControl published{};
                    if (aclrtMemcpy(
                            &published, sizeof(published),
                            reinterpret_cast<FusionBulkControl *>(
                                static_cast<uint8_t *>(sym) +
                                plan.symmetric.reserved64[3]) +
                                static_cast<uint64_t>(wave_slot) *
                                    control_lines +
                                static_cast<uint64_t>(2u) *
                                    config.worker_count + producer,
                            sizeof(published),
                            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                        rc = 6;
                        break;
                    }
                    const uint64_t input_bytes =
                        published.bytes0 + published.bytes1;
                    std::vector<uint8_t> input_values(input_bytes);
                    if (aclrtMemcpy(
                            input_values.data(), input_values.size(),
                            static_cast<uint8_t *>(sym) +
                                plan.symmetric.reserved64[2] +
                                (static_cast<uint64_t>(wave_slot) *
                                     config.worker_count +
                                 producer) * expert_stride,
                            input_values.size(),
                            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                        rc = 6;
                        break;
                    }
                    if (published.bytes1 == 0u &&
                        published.bytes0 % sizeof(uint16_t) == 0u) {
                        dense_inputs[producer].resize(
                            published.bytes0 / sizeof(uint16_t));
                        std::memcpy(dense_inputs[producer].data(),
                                    input_values.data(), published.bytes0);
                    }
                    auto hash_range = [&](uint64_t begin,
                                          uint64_t bytes) {
                        uint64_t hash = 1469598103934665603ull;
                        for (uint64_t i = begin; i < begin + bytes; ++i) {
                            hash ^= input_values[i];
                            hash *= 1099511628211ull;
                        }
                        return hash;
                    };
                    uint64_t normalized_metadata_hash =
                        1469598103934665603ull;
                    const auto *metadata =
                        reinterpret_cast<const FusionExpertAssignment *>(
                            input_values.data() + published.bytes0);
                    const uint64_t metadata_count = published.bytes1 /
                        sizeof(FusionExpertAssignment);
                    uint64_t normalized_metadata_count = 0u;
                    auto hash_word = [&](uint32_t word) {
                        for (uint32_t byte = 0u; byte < 4u; ++byte) {
                            normalized_metadata_hash ^=
                                static_cast<uint8_t>(word >> (byte * 8u));
                            normalized_metadata_hash *= 1099511628211ull;
                        }
                    };
                    for (uint64_t i = 0u; i < metadata_count; ++i) {
                        const uint64_t entry_generation =
                            (static_cast<uint64_t>(
                                 metadata[i].local_expert) << 32u) |
                            metadata[i].expert_id;
                        if (entry_generation != published.generation)
                            continue;
                        ++normalized_metadata_count;
                        hash_word(metadata[i].dispatch_row);
                        hash_word(metadata[i].route_ordinal);
                        hash_word(metadata[i].destination_token);
                        hash_word(metadata[i].weight_bits);
                        hash_word(metadata[i].wave);
                        hash_word(metadata[i].destination_row);
                    }
                    std::cout << "FUSION_ITERATION_INC_INPUT pe=" << pe
                              << " iteration=" << iteration
                              << " producer=" << producer
                              << " payload_hash="
                              << hash_range(0u, published.bytes0)
                              << " metadata_hash="
                              << hash_range(published.bytes0,
                                            published.bytes1)
                              << " normalized_metadata_hash="
                              << normalized_metadata_hash
                              << " normalized_metadata_count="
                              << normalized_metadata_count
                              << " bytes0=" << published.bytes0
                              << " bytes1=" << published.bytes1 << "\n";
                }
                const uint64_t valid_bytes =
                    static_cast<uint64_t>(config.worker_count) *
                    config.tokens_per_wave * config.hidden *
                    sizeof(uint16_t);
                std::vector<uint8_t> values(valid_bytes);
                if (aclrtMemcpy(
                        values.data(), values.size(),
                        static_cast<uint8_t *>(workspace) +
                            plan.inc_workspace.combine_ring_off +
                            static_cast<uint64_t>(wave_slot) *
                                plan.inc_workspace.output_slot_bytes,
                        values.size(), ACL_MEMCPY_DEVICE_TO_HOST) !=
                    ACL_SUCCESS) {
                    rc = 6;
                } else {
                    uint64_t hash = 1469598103934665603ull;
                    for (uint8_t value : values) {
                        hash ^= value;
                        hash *= 1099511628211ull;
                    }
                    std::cout << "FUSION_ITERATION_INC_ACCUMULATOR pe="
                              << pe << " iteration=" << iteration
                              << " hash=" << hash << "\n";
                    const auto *actual = reinterpret_cast<const uint16_t *>(
                        values.data());
                    const size_t elements = values.size() / sizeof(uint16_t);
                    bool dense_shape = true;
                    for (const auto &input : dense_inputs)
                        dense_shape = dense_shape && input.size() == elements;
                    if (dense_shape) {
                        float max_error = 0.0f;
                        size_t max_index = 0u;
                        for (size_t i = 0u; i < elements; ++i) {
                            float sum = 0.0f;
                            for (const auto &input : dense_inputs)
                                sum += FromBf16(input[i]);
                            const float error = std::abs(
                                FromBf16(actual[i]) -
                                FromBf16(ToBf16(sum)));
                            if (error > max_error) {
                                max_error = error;
                                max_index = i;
                            }
                        }
                        std::cout << "FUSION_ITERATION_INC_DENSE_GOLDEN pe="
                                  << pe << " iteration=" << iteration
                                  << " max_abs_error=" << max_error
                                  << " max_error_index=" << max_index
                                  << " actual=" << FromBf16(actual[max_index])
                                  << "\n";
                    }
                }
            }
            if (end_event != nullptr) (void)aclrtDestroyEvent(end_event);
            if (start_event != nullptr) (void)aclrtDestroyEvent(start_event);
            if (!persistent_mode) aclshmem_barrier_all();
            if (iteration >= warmup) measured_us.push_back(iteration_ms * 1000.0f);
        }
        if (!iteration_output_stable && rc == 0) rc = 7;
        if (persistent_service != nullptr) {
            if (inc_fusion_persistent_service_stop(persistent_service) !=
                INC_FUSION_OK)
                rc = 6;
            inc_fusion_persistent_service_destroy(persistent_service);
            persistent_service = nullptr;
        }
        if (persistent_stream != nullptr) {
            (void)aclrtDestroyStream(persistent_stream);
            persistent_stream = nullptr;
        }
        if (persistent_plan != nullptr) {
            inc_fusion_prepared_plan_destroy(persistent_plan);
            persistent_plan = nullptr;
        }
        if (worker_executor != nullptr) {
            inc_fusion_worker_executor_destroy(worker_executor);
            worker_executor = nullptr;
        }
        if (worker_plan != nullptr) {
            inc_fusion_prepared_plan_destroy(worker_plan);
            worker_plan = nullptr;
        }
        if (!measured_us.empty()) {
            std::vector<float> ordered = measured_us;
            std::sort(ordered.begin(), ordered.end());
            elapsed_ms = ordered[ordered.size() / 2u] / 1000.0f;
        }
        if (rc == 0 && worker) {
            std::vector<FusionSlotState> states(config.slot_count);
            if (aclrtMemcpy(states.data(), states.size() * sizeof(states[0]),
                            static_cast<uint8_t *>(workspace) +
                                plan.worker_workspace.slot_state_off,
                            states.size() * sizeof(states[0]),
                            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                aclrtMemcpy(host_output.data(),
                            host_output.size() * sizeof(uint16_t), d_output,
                            host_output.size() * sizeof(uint16_t),
                            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                rc = 6;
            }
            for (uint32_t slot = 0u; slot < states.size(); ++slot) {
                const auto &state = states[slot];
                if (state.error != kFusionSlotOk) {
                    std::cerr << "slot=" << slot
                              << " error=" << state.error
                              << " site=" << state.owner
                              << " D=" << state.dispatch_generation
                              << " C=" << state.combine_generation
                              << " O=" << state.output_generation
                              << " R=" << state.release_generation
                              << " detail=" << state.reserved << "\n";
                    const uint64_t row_bytes =
                        static_cast<uint64_t>(hidden) * sizeof(uint16_t);
                    const uint64_t capacity =
                        plan.worker_workspace.assignment_slot_bytes /
                        row_bytes;
                    std::vector<FusionReceivedAssignment> received(capacity);
                    const auto *received_gm =
                        static_cast<const uint8_t *>(workspace) +
                        plan.worker_workspace.assignment_meta_off +
                        static_cast<uint64_t>(slot) * capacity *
                            sizeof(FusionReceivedAssignment);
                    if (aclrtMemcpy(
                            received.data(), received.size() *
                                sizeof(FusionReceivedAssignment),
                            received_gm, received.size() *
                                sizeof(FusionReceivedAssignment),
                            ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) {
                        for (uint32_t row = 0u; row < received.size(); ++row) {
                            const auto &entry = received[row];
                            if (entry.wave == slot && entry.source_token != 0u)
                                continue;
                            if (row >= 24u) break;
                            std::cerr << "  meta row=" << row
                                      << " wave=" << entry.wave
                                      << " src=" << entry.source_rank
                                      << " token=" << entry.source_token
                                      << " expert=" << entry.expert_id
                                      << " ordinal=" << entry.route_ordinal
                                      << " dst_row=" << entry.destination_row
                                      << "\n";
                        }
                    }
                    rc = 7;
                }
            }
            if (std::getenv("FUSION_VALIDATE_INTERMEDIATES") != nullptr &&
                plan.waves.size() == 1u) {
                const uint32_t slot = routes.ranks[pe].waves[0].slot;
                const uint64_t row_bytes =
                    static_cast<uint64_t>(hidden) * sizeof(uint16_t);
                const uint64_t capacity =
                    plan.worker_workspace.assignment_slot_bytes / row_bytes;
                const int64_t assignment_count_i64 =
                    routes.group_lists[pe][0].empty()
                        ? 0 : routes.group_lists[pe][0].back();
                const uint32_t assignment_count = assignment_count_i64 < 0
                    ? 0u : static_cast<uint32_t>(assignment_count_i64);
                std::vector<FusionReceivedAssignment> received(
                    assignment_count);
                std::vector<uint16_t> grouped(
                    static_cast<size_t>(assignment_count) * hidden);
                std::vector<uint16_t> expert_rows(grouped.size());
                std::vector<uint16_t> observed_gate(
                    static_cast<size_t>(assignment_count) * 2u * intermediate);
                std::vector<uint16_t> observed_activation(
                    static_cast<size_t>(assignment_count) * intermediate);
                const auto *received_gm =
                    static_cast<const uint8_t *>(workspace) +
                    plan.worker_workspace.assignment_meta_off +
                    static_cast<uint64_t>(slot) * capacity *
                        sizeof(FusionReceivedAssignment);
                const auto *grouped_gm =
                    static_cast<const uint8_t *>(workspace) +
                    plan.worker_workspace.grouped_input_off +
                    static_cast<uint64_t>(slot) *
                        plan.worker_workspace.assignment_slot_bytes;
                const auto *expert_gm =
                    static_cast<const uint8_t *>(workspace) +
                    plan.worker_workspace.expert_output_off +
                    static_cast<uint64_t>(slot) *
                        plan.worker_workspace.assignment_slot_bytes;
                const auto *gate_gm =
                    static_cast<const uint8_t *>(workspace) +
                    plan.worker_workspace.gate_up_off +
                    static_cast<uint64_t>(slot) * capacity * 2u *
                        intermediate * sizeof(uint16_t);
                const auto *activation_gm =
                    static_cast<const uint8_t *>(workspace) +
                    plan.worker_workspace.activation_off +
                    static_cast<uint64_t>(slot) * capacity * intermediate *
                        sizeof(uint16_t);
                float dispatch_max_error = 0.0f;
                float gate_max_error = 0.0f;
                float activation_max_error = 0.0f;
                float compute_max_error = 0.0f;
                uint64_t metadata_errors = 0u;
                if (aclrtMemcpy(received.data(), received.size() *
                            sizeof(FusionReceivedAssignment), received_gm,
                            received.size() * sizeof(FusionReceivedAssignment),
                            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                    aclrtMemcpy(grouped.data(), grouped.size() *
                            sizeof(uint16_t), grouped_gm,
                            grouped.size() * sizeof(uint16_t),
                            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                    aclrtMemcpy(expert_rows.data(), expert_rows.size() *
                            sizeof(uint16_t), expert_gm,
                            expert_rows.size() * sizeof(uint16_t),
                            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                    aclrtMemcpy(observed_gate.data(), observed_gate.size() *
                            sizeof(uint16_t), gate_gm,
                            observed_gate.size() * sizeof(uint16_t),
                            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                    aclrtMemcpy(observed_activation.data(),
                            observed_activation.size() * sizeof(uint16_t),
                            activation_gm,
                            observed_activation.size() * sizeof(uint16_t),
                            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                    rc = 6;
                } else {
                    for (uint32_t row = 0u; row < assignment_count; ++row) {
                        const auto &meta = received[row];
                        if (meta.destination_row != row ||
                            meta.source_rank >= workers ||
                            meta.source_token >= active_tokens[meta.source_rank] ||
                            meta.expert_id >= experts ||
                            meta.local_expert != local[meta.expert_id]) {
                            ++metadata_errors;
                            continue;
                        }
                        const int64_t expert_begin = meta.local_expert == 0u
                            ? 0 : routes.group_lists[pe][0][
                                meta.local_expert - 1u];
                        const int64_t expert_end = routes.group_lists[pe][0][
                            meta.local_expert];
                        if (row < static_cast<uint64_t>(expert_begin) ||
                            row >= static_cast<uint64_t>(expert_end))
                            ++metadata_errors;
                        for (uint32_t column = 0u; column < hidden; ++column) {
                            const float actual = FromBf16(grouped[
                                static_cast<size_t>(row) * hidden + column]);
                            const float expected = InputValue(
                                meta.source_rank, meta.source_token, column);
                            dispatch_max_error = std::max(
                                dispatch_max_error,
                                std::abs(actual - FromBf16(ToBf16(expected))));
                        }
                        if (meta.source_token == 0u) {
                            std::vector<uint16_t> debug_gate(
                                2u * intermediate);
                            std::vector<uint16_t> debug_activation(intermediate);
                            for (uint32_t column = 0u;
                                 column < 2u * intermediate; ++column) {
                                float sum = 0.0f;
                                for (uint32_t inner = 0u; inner < hidden;
                                     ++inner)
                                    sum += FromBf16(grouped[
                                               static_cast<size_t>(row) * hidden +
                                               inner]) *
                                        FromBf16(ToBf16(W13Value(
                                            meta.expert_id, column, inner)));
                                debug_gate[column] = ToBf16(sum);
                                gate_max_error = std::max(
                                    gate_max_error,
                                    std::abs(FromBf16(observed_gate[
                                        (static_cast<size_t>(row) * 2u *
                                         intermediate) + column]) -
                                        FromBf16(debug_gate[column])));
                            }
                            for (uint32_t column = 0u; column < intermediate;
                                 ++column) {
                                const float gate = FromBf16(debug_gate[column]);
                                const float up = FromBf16(
                                    debug_gate[intermediate + column]);
                                debug_activation[column] = ToBf16(
                                    gate / (1.0f + std::exp(-gate)) * up);
                                activation_max_error = std::max(
                                    activation_max_error,
                                    std::abs(FromBf16(observed_activation[
                                        static_cast<size_t>(row) * intermediate +
                                        column]) -
                                        FromBf16(debug_activation[column])));
                            }
                            for (uint32_t column = 0u; column < hidden;
                                 ++column) {
                                float sum = 0.0f;
                                for (uint32_t inner = 0u;
                                     inner < intermediate; ++inner)
                                    sum += FromBf16(debug_activation[inner]) *
                                        FromBf16(ToBf16(W2Value(
                                            meta.expert_id, column, inner)));
                                compute_max_error = std::max(
                                    compute_max_error,
                                    std::abs(FromBf16(expert_rows[
                                        static_cast<size_t>(row) * hidden +
                                        column]) - FromBf16(ToBf16(sum))));
                            }
                        }
                    }
                }
                std::cout << "FUSION_INTERMEDIATE_DISPATCH pe=" << pe
                          << " rows=" << assignment_count
                          << " metadata_errors=" << metadata_errors
                          << " max_abs_error=" << dispatch_max_error
                          << " gate_max_abs_error=" << gate_max_error
                          << " activation_max_abs_error="
                          << activation_max_error
                          << " compute_max_abs_error=" << compute_max_error
                          << "\n";
                if (topk >= 2u) {
                    const uint32_t token_begin = plan.waves[0].token_begin;
                    uint32_t dense_token_stride = 0u;
                    for (uint32_t source = 0u; source < workers; ++source) {
                        const uint32_t count = active_tokens[source] <= token_begin
                            ? 0u : std::min(config.tokens_per_wave,
                                active_tokens[source] - token_begin);
                        dense_token_stride = std::max(dense_token_stride, count);
                    }
                    const uint32_t dense_rows = workers * dense_token_stride;
                    std::vector<uint16_t> observed_partial(
                        static_cast<size_t>(dense_rows) * hidden);
                    std::vector<float> expected_partial(
                        observed_partial.size(), 0.0f);
                    const auto *partial_gm =
                        static_cast<const uint8_t *>(workspace) +
                        plan.worker_workspace.combine_ring_off +
                        static_cast<uint64_t>(slot) *
                            plan.worker_workspace.assignment_slot_bytes;
                    if (aclrtMemcpy(observed_partial.data(),
                            observed_partial.size() * sizeof(uint16_t),
                            partial_gm,
                            observed_partial.size() * sizeof(uint16_t),
                            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                        rc = 6;
                    } else {
                        for (uint32_t row = 0u; row < assignment_count; ++row) {
                            const auto &meta = received[row];
                            if (meta.source_rank >= workers ||
                                meta.source_token < token_begin ||
                                meta.source_token - token_begin >=
                                    dense_token_stride ||
                                meta.route_ordinal >= topk)
                                continue;
                            const uint64_t dense_row =
                                static_cast<uint64_t>(meta.source_rank) *
                                    dense_token_stride +
                                meta.source_token - token_begin;
                            const size_t route =
                                (static_cast<size_t>(meta.source_rank) * tokens +
                                 meta.source_token) * topk +
                                meta.route_ordinal;
                            for (uint32_t column = 0u; column < hidden; ++column)
                                expected_partial[dense_row * hidden + column] +=
                                    weights[route] * FromBf16(expert_rows[
                                        static_cast<size_t>(row) * hidden +
                                        column]);
                        }
                        float dense_partial_max_error = 0.0f;
                        size_t dense_partial_max_index = 0u;
                        for (size_t i = 0u; i < observed_partial.size(); ++i) {
                            const float error = std::abs(
                                FromBf16(observed_partial[i]) -
                                FromBf16(ToBf16(expected_partial[i])));
                            if (error > dense_partial_max_error) {
                                dense_partial_max_error = error;
                                dense_partial_max_index = i;
                            }
                        }
                        std::cout << "FUSION_INTERMEDIATE_DENSE_PARTIAL pe="
                                  << pe << " rows=" << dense_rows
                                  << " max_abs_error="
                                  << dense_partial_max_error
                                  << " max_error_index="
                                  << dense_partial_max_index << "\n";
                    }
                }
            }
            double checksum = 0.0;
            for (uint16_t value : host_output) checksum += FromBf16(value);
            std::vector<float> golden(host_output.size(), 0.0f);
            const uint32_t golden_limit =
                std::getenv("FUSION_GOLDEN_TOKENS") == nullptr
                    ? UINT32_MAX : static_cast<uint32_t>(std::max(
                        1, std::atoi(std::getenv("FUSION_GOLDEN_TOKENS"))));
            std::vector<uint16_t> gate_up(2u * intermediate);
            std::vector<uint16_t> activation(intermediate);
            const uint32_t source_begin = global_output ? 0u : pe;
            const uint32_t source_end = global_output ? workers : pe + 1u;
            uint64_t global_token_base = 0u;
            uint64_t verified_tokens = 0u;
            for (uint32_t source = 0u; source < source_end; ++source) {
                const uint32_t source_tokens = active_tokens[source];
                if (source < source_begin) {
                    global_token_base += source_tokens;
                    continue;
                }
                const uint32_t source_golden_tokens =
                    std::min(source_tokens, golden_limit);
                verified_tokens += source_golden_tokens;
                for (uint32_t token = 0u;
                     token < source_golden_tokens; ++token) {
                    for (uint32_t ordinal = 0u; ordinal < topk; ++ordinal) {
                        const size_t route =
                            (static_cast<size_t>(source) * tokens + token) *
                                topk + ordinal;
                        const uint32_t expert = expert_ids[route];
                        for (uint32_t column = 0u;
                             column < 2u * intermediate; ++column) {
                            float sum = 0.0f;
                            for (uint32_t inner = 0u; inner < hidden; ++inner) {
                                const float input = source == pe
                                    ? FromBf16(host_input[
                                          static_cast<size_t>(token) * hidden +
                                          inner])
                                    : FromBf16(ToBf16(InputValue(
                                          source, token, inner)));
                                sum += input * FromBf16(ToBf16(W13Value(
                                    expert, column, inner)));
                            }
                            gate_up[column] = ToBf16(sum);
                        }
                        for (uint32_t column = 0u; column < intermediate;
                             ++column) {
                            const float gate = FromBf16(gate_up[column]);
                            const float up =
                                FromBf16(gate_up[intermediate + column]);
                            activation[column] = ToBf16(
                                gate / (1.0f + std::exp(-gate)) * up);
                        }
                        for (uint32_t column = 0u;
                             column < hidden; ++column) {
                            float sum = 0.0f;
                            for (uint32_t inner = 0u;
                                 inner < intermediate; ++inner)
                                sum += FromBf16(activation[inner]) *
                                       FromBf16(ToBf16(W2Value(
                                           expert, column, inner)));
                            const uint64_t output_token = global_output
                                ? global_token_base + token : token;
                            golden[output_token * hidden + column] +=
                                weights[route] * FromBf16(ToBf16(sum));
                        }
                    }
                }
                global_token_base += source_tokens;
            }
            float max_error = 0.0f;
            float max_reference = 0.0f;
            size_t max_error_index = 0u;
            uint64_t checked_token_base = 0u;
            for (uint32_t source = source_begin; source < source_end;
                 ++source) {
                float source_max_error = 0.0f;
                size_t source_max_index = 0u;
                const uint32_t source_golden_tokens =
                    std::min(active_tokens[source], golden_limit);
                for (uint32_t token = 0u;
                     token < source_golden_tokens; ++token) {
                    const uint64_t output_token = global_output
                        ? checked_token_base + token : token;
                    for (uint32_t column = 0u; column < hidden; ++column) {
                        const size_t i = output_token * hidden + column;
                        const float error =
                            std::abs(FromBf16(host_output[i]) - golden[i]);
                        if (error > max_error) {
                            max_error = error;
                            max_error_index = i;
                        }
                        if (error > source_max_error) {
                            source_max_error = error;
                            source_max_index = i;
                        }
                        max_reference = std::max(max_reference,
                                                 std::abs(golden[i]));
                    }
                }
                if (global_output)
                    std::cout << "FUSION_GLOBAL_SOURCE pe=" << pe
                              << " source=" << source
                              << " rows=" << active_tokens[source]
                              << " verified=" << source_golden_tokens
                              << " max_abs_error=" << source_max_error
                              << " max_error_index=" << source_max_index
                              << " actual="
                              << FromBf16(host_output[source_max_index])
                              << " expected=" << golden[source_max_index]
                              << "\n";
                checked_token_base += active_tokens[source];
            }
            const float relative_error = max_reference == 0.0f
                ? max_error : max_error / max_reference;
            if (!std::isfinite(checksum) || !std::isfinite(max_error) ||
                relative_error > 0.02f)
                rc = 7;
            std::vector<FusionLaneTrace> worker_trace(4u);
            if (aclrtMemcpy(worker_trace.data(),
                            worker_trace.size() * sizeof(worker_trace[0]),
                            static_cast<uint8_t *>(workspace) +
                                plan.worker_workspace.trace_off,
                            worker_trace.size() * sizeof(worker_trace[0]),
                            ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) {
                std::cout << "FUSION_WORKER_TRACE pe=" << pe;
                for (const auto &trace : worker_trace) {
                    const uint64_t span = trace.end_cycle > trace.start_cycle
                        ? trace.end_cycle - trace.start_cycle : 0u;
                    std::cout << " role" << trace.role << "=" << span;
                    if (trace.role >= 3u && trace.role <= 6u) {
                        std::cout << " checkpoints=";
                        for (uint32_t checkpoint = 0u; checkpoint < 5u;
                             ++checkpoint) {
                            if (checkpoint != 0u) std::cout << ',';
                            const uint64_t cycle =
                                trace.reserved64[checkpoint];
                            std::cout << (cycle > trace.start_cycle
                                ? cycle - trace.start_cycle : 0u);
                        }
                    }
                }
                std::cout << "\n";
            }
            std::cout << "FUSION_E2E_RESULT pe=" << pe
                      << " role=worker checksum=" << checksum
                      << " max_abs_error=" << max_error
                      << " max_reference=" << max_reference
                      << " max_relative_error=" << relative_error
                      << " max_error_index=" << max_error_index
                      << " max_error_actual="
                      << FromBf16(host_output[max_error_index])
                      << " max_error_expected=" << golden[max_error_index]
                      << " verified_tokens=" << verified_tokens
                      << " elapsed_us=" << elapsed_ms * 1000.0f
                      << " pass=" << (rc == 0 ? 1 : 0) << "\n";
        } else if (rc == 0) {
            std::vector<FusionSlotState> states(config.slot_count);
            std::vector<FusionLaneTrace> lane_trace(
                plan.resources.inc_dispatch_aiv +
                plan.resources.inc_combine_aiv);
            if (aclrtMemcpy(states.data(), states.size() * sizeof(states[0]),
                            static_cast<uint8_t *>(workspace) +
                                plan.inc_workspace.slot_state_off,
                            states.size() * sizeof(states[0]),
                            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                aclrtMemcpy(lane_trace.data(),
                            lane_trace.size() * sizeof(lane_trace[0]),
                            static_cast<uint8_t *>(workspace) +
                                plan.inc_workspace.trace_off,
                            lane_trace.size() * sizeof(lane_trace[0]),
                            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                rc = 6;
            }
            for (uint32_t slot = 0u; slot < states.size(); ++slot) {
                if (states[slot].error != kFusionSlotOk) {
                    std::cerr << "inc_slot=" << slot
                              << " error=" << states[slot].error
                              << " site=" << states[slot].owner
                              << " detail=" << states[slot].reserved << "\n";
                    rc = 7;
                }
            }
            std::cout << "FUSION_E2E_RESULT pe=" << pe
                      << " role=inc elapsed_us=" << elapsed_ms * 1000.0f
                      << " pass=" << (rc == 0 ? 1 : 0) << "\n";
            uint64_t d_begin = UINT64_MAX, d_end = 0u;
            uint64_t c_begin = UINT64_MAX, c_end = 0u;
            for (const auto &trace : lane_trace) {
                if (trace.end_cycle <= trace.start_cycle) continue;
                if (trace.role == 1u) {
                    d_begin = std::min(d_begin, trace.start_cycle);
                    d_end = std::max(d_end, trace.end_cycle);
                    if (trace.lane == 0u) {
                        std::cout << "FUSION_INC_DISPATCH_CHECKPOINTS";
                        for (uint32_t checkpoint = 0u; checkpoint < 5u;
                             ++checkpoint) {
                            const uint64_t cycle =
                                trace.reserved64[checkpoint];
                            std::cout << (checkpoint == 0u ? " " : ",")
                                      << (cycle > trace.start_cycle
                                          ? cycle - trace.start_cycle : 0u);
                        }
                        std::cout << "\n";
                    }
                } else if (trace.role == 2u) {
                    c_begin = std::min(c_begin, trace.start_cycle);
                    c_end = std::max(c_end, trace.end_cycle);
                    if (trace.lane == 0u) {
                        std::cout << "FUSION_INC_COMBINE_CHECKPOINTS";
                        for (uint32_t checkpoint = 0u; checkpoint < 5u;
                             ++checkpoint) {
                            const uint64_t cycle =
                                trace.reserved64[checkpoint];
                            std::cout << (checkpoint == 0u ? " " : ",")
                                      << (cycle > trace.start_cycle
                                          ? cycle - trace.start_cycle : 0u);
                        }
                        std::cout << "\n";
                    }
                }
            }
            if (d_begin != UINT64_MAX && c_begin != UINT64_MAX) {
                const uint64_t d_span = d_end - d_begin;
                const uint64_t c_span = c_end - c_begin;
                const uint64_t overlap_begin = std::max(d_begin, c_begin);
                const uint64_t overlap_end = std::min(d_end, c_end);
                const uint64_t overlap = overlap_end > overlap_begin
                    ? overlap_end - overlap_begin : 0u;
                const double realized = std::min(d_span, c_span) == 0u
                    ? 0.0 : static_cast<double>(overlap) /
                        std::min(d_span, c_span);
                const double window_speedup =
                    d_span + c_span == overlap
                    ? 1.0 : static_cast<double>(d_span + c_span) /
                        static_cast<double>(d_span + c_span - overlap);
                const double theoretical = std::max(d_span, c_span) == 0u
                    ? 1.0 : static_cast<double>(d_span + c_span) /
                        static_cast<double>(std::max(d_span, c_span));
                std::cout << "FUSION_INC_TRACE d_span_cycles=" << d_span
                          << " c_span_cycles=" << c_span
                          << " overlap_cycles=" << overlap
                          << " overlap_realized=" << realized
                          << " dc_window_speedup=" << window_speedup
                          << " dc_theoretical_max=" << theoretical << "\n";
            }
        }
        std::cout << "FUSION_SAMPLES pe=" << pe << " us=";
        for (size_t sample = 0u; sample < measured_us.size(); ++sample) {
            if (sample != 0u) std::cout << ',';
            std::cout << measured_us[sample];
        }
        std::cout << "\n";
    }

shmem_cleanup:
    if (persistent_service != nullptr) {
        (void)inc_fusion_persistent_service_stop(persistent_service);
        inc_fusion_persistent_service_destroy(persistent_service);
    }
    if (persistent_stream != nullptr)
        (void)aclrtDestroyStream(persistent_stream);
    if (persistent_plan != nullptr)
        inc_fusion_prepared_plan_destroy(persistent_plan);
    if (worker_executor != nullptr)
        inc_fusion_worker_executor_destroy(worker_executor);
    if (worker_plan != nullptr)
        inc_fusion_prepared_plan_destroy(worker_plan);
    aclshmem_barrier_all();
    if (sym != nullptr) aclshmem_free(sym);
    FreeAll(device_buffers);
    aclshmem_finalize();
cleanup:
    if (stream != nullptr) (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(device);
    (void)aclFinalize();
    return rc;
}
