/*
 * 单 INC 最简 API：可在已配置 CANN toolchain 的 host 上运行的完整示例。
 *
 * 这个程序故意使用 CPU mock backend：运行时不占用 NPU，也能观察完整生命周期
 * （配置/链接本仓库仍需 CANN/BiSheng）。除了 backend、allocator 和 stream 是 host mock，
 * session / prepared plan / request / route handle 全部调用正式公共 API。
 *
 * 模拟的数据流：
 *   token [3, 4] --Dispatch(top-k=2)--> expert rows [6, 4]
 *                --模拟 expert compute--> expert output [6, 4]
 *                --Combine(weighted reduce)--> token output [3, 4]
 */

#include "inc_dc_single_inc.hpp"
#include "inc_dc_fp16_host.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr uint64_t kTokens = 3u;
constexpr uint32_t kHidden = 4u;
constexpr uint32_t kTopK = 2u;
constexpr uint64_t kAssignments = kTokens * kTopK;
constexpr uint32_t kWorldSize = 2u;
constexpr uint32_t kExpertsPerWorker = 2u;
constexpr uint64_t kRouteGeneration = 101u;

/* backend ticket 只模拟异步请求的状态；真正的 backend 通常保存 device event。 */
struct MockTicket {
    uint64_t generation = 0u;
    uint32_t state = INC_DC_FW_REQUEST_FREE;
};

struct CpuMockBackend {
    std::array<MockTicket, 8> tickets{};
    uint64_t next_generation = 1u;
    uint64_t allocations = 0u;
    uint64_t frees = 0u;
};

float BitsToFloat(uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

/*
 * 校验并解析由 inc_dc_easy_token_plan_build() 构造的公开 token-plan ABI。
 * 真机 backend 会从 HBM 中读取同样的结构；mock 直接读取 host vector。
 */
inc_dc_fw_status_t ParseRoute(
    const inc_dc_fw_route_desc_t &route,
    const inc_dc_easy_token_plan_header_v1_t **header,
    const inc_dc_easy_token_assignment_v1_t **assignments)
{
    if (route.format != INC_DC_FW_ROUTE_TOPK_DENSE || route.data == nullptr ||
        route.bytes < sizeof(inc_dc_easy_token_plan_header_v1_t)) {
        return INC_DC_FW_LAYOUT_MISMATCH;
    }
    const auto *parsed = static_cast<
        const inc_dc_easy_token_plan_header_v1_t *>(route.data);
    if (parsed->magic != INC_DC_EASY_TOKEN_PLAN_MAGIC ||
        parsed->abi_major != INC_DC_EASY_TOKEN_PLAN_ABI_VERSION ||
        parsed->header_bytes != sizeof(*parsed) ||
        parsed->assignment_bytes !=
            sizeof(inc_dc_easy_token_assignment_v1_t) ||
        parsed->total_bytes > route.bytes ||
        parsed->assignment_count != parsed->tokens * parsed->topk) {
        return INC_DC_FW_LAYOUT_MISMATCH;
    }
    *header = parsed;
    *assignments = reinterpret_cast<
        const inc_dc_easy_token_assignment_v1_t *>(
            static_cast<const uint8_t *>(route.data) + parsed->header_bytes);
    return INC_DC_FW_OK;
}

MockTicket *FindTicket(
    CpuMockBackend *backend, inc_dc_fw_backend_ticket_t ticket)
{
    if (ticket.value == 0u || ticket.value > backend->tickets.size()) {
        return nullptr;
    }
    MockTicket &found = backend->tickets[ticket.value - 1u];
    return found.generation == ticket.generation ? &found : nullptr;
}

inc_dc_fw_status_t MockGetCapabilities(
    void *, inc_dc_fw_capabilities_t *capabilities)
{
    if (capabilities == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    *capabilities = {};
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    /* 只声明本示例真正实现的最小能力，不把 mock 当作并发性能 backend。 */
    capabilities->feature_bits = INC_DC_FW_FEATURE_EXTERNAL_STREAM |
                                 INC_DC_FW_FEATURE_DEVICE_ROUTE;
    capabilities->max_world_size = 16u;
    capabilities->max_topk = 8u;
    capabilities->max_inflight = 8u;
    capabilities->workspace_alignment = 64u;
    capabilities->dtype_bits = 1ull << INC_DC_FW_DTYPE_FP16;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t MockQueryWorkspace(
    void *, const inc_dc_fw_plan_desc_t *, uint32_t operation,
    const inc_dc_fw_shape_t *shape, inc_dc_fw_workspace_t *workspace)
{
    if (shape == nullptr || workspace == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    *workspace = {};
    workspace->struct_size = sizeof(*workspace);
    workspace->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    workspace->alignment = 64u;
    /* 这里只需要非零、对齐的 mock workspace；真机由 native backend 精确 query。 */
    workspace->bytes = 256u + shape->tokens * shape->topk * sizeof(uint32_t) +
        (operation == INC_DC_FW_OP_COMBINE ? 128u : 0u);
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t MockDispatch(const inc_dc_fw_invocation_t *invocation)
{
    const inc_dc_easy_token_plan_header_v1_t *header = nullptr;
    const inc_dc_easy_token_assignment_v1_t *assignments = nullptr;
    inc_dc_fw_status_t status = ParseRoute(
        invocation->route, &header, &assignments);
    if (status != INC_DC_FW_OK) return status;
    if (header->tokens != invocation->shape.tokens ||
        header->topk != invocation->shape.topk ||
        invocation->input.dims[0] < static_cast<int64_t>(header->tokens) ||
        invocation->output.dims[0] <
            static_cast<int64_t>(header->assignment_count)) {
        return INC_DC_FW_LAYOUT_MISMATCH;
    }

    const auto *input = static_cast<const uint16_t *>(invocation->input.data);
    auto *fanout = static_cast<uint16_t *>(invocation->output.data);
    for (uint64_t token = 0u; token < header->tokens; ++token) {
        for (uint32_t slot = 0u; slot < header->topk; ++slot) {
            const uint64_t row = token * header->topk + slot;
            /*
             * 每个 (token, top-k slot) 形成一行 expert 输入；真实单 INC backend
             * 会按 assignments[row].destination_rank 经 INC fan-out。
             */
            for (uint32_t hidden = 0u; hidden < invocation->shape.hidden_size;
                 ++hidden) {
                fanout[row * invocation->shape.hidden_size + hidden] =
                    input[token * invocation->shape.hidden_size + hidden];
            }
        }
    }
    (void)assignments; /* Dispatch 结果的行序由 token-plan 固定。 */
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t MockCombine(const inc_dc_fw_invocation_t *invocation)
{
    const inc_dc_easy_token_plan_header_v1_t *header = nullptr;
    const inc_dc_easy_token_assignment_v1_t *assignments = nullptr;
    inc_dc_fw_status_t status = ParseRoute(
        invocation->route, &header, &assignments);
    if (status != INC_DC_FW_OK) return status;
    if (header->tokens != invocation->shape.tokens ||
        header->topk != invocation->shape.topk ||
        invocation->input.dims[0] <
            static_cast<int64_t>(header->assignment_count) ||
        invocation->output.dims[0] < static_cast<int64_t>(header->tokens)) {
        return INC_DC_FW_LAYOUT_MISMATCH;
    }

    const auto *expert_output =
        static_cast<const uint16_t *>(invocation->input.data);
    auto *token_output = static_cast<uint16_t *>(invocation->output.data);
    std::memset(
        token_output, 0,
        header->tokens * invocation->shape.hidden_size * sizeof(uint16_t));
    for (uint64_t token = 0u; token < header->tokens; ++token) {
        for (uint32_t slot = 0u; slot < header->topk; ++slot) {
            const uint64_t row = token * header->topk + slot;
            const float weight = BitsToFloat(assignments[row].weight_bits);
            for (uint32_t hidden = 0u; hidden < invocation->shape.hidden_size;
                 ++hidden) {
                const uint64_t output_index =
                    token * invocation->shape.hidden_size + hidden;
                const float accumulated = inc::Fp16BitsToFloat(
                    token_output[output_index]);
                const float contribution = weight * inc::Fp16BitsToFloat(
                    expert_output[row * invocation->shape.hidden_size + hidden]);
                token_output[output_index] =
                    inc::FloatToFp16Bits(accumulated + contribution);
            }
        }
    }
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t MockEnqueue(
    void *opaque, const inc_dc_fw_plan_desc_t *, uint32_t operation,
    const inc_dc_fw_invocation_t *invocation,
    inc_dc_fw_backend_ticket_t *ticket)
{
    auto *backend = static_cast<CpuMockBackend *>(opaque);
    if (backend == nullptr || invocation == nullptr || ticket == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }

    MockTicket *free_ticket = nullptr;
    size_t index = 0u;
    for (; index < backend->tickets.size(); ++index) {
        if (backend->tickets[index].state == INC_DC_FW_REQUEST_FREE) {
            free_ticket = &backend->tickets[index];
            break;
        }
    }
    if (free_ticket == nullptr) return INC_DC_FW_CAPACITY_EXCEEDED;

    /* CPU 计算立即完成，但状态先保留 SUBMITTED，以展示 query/wait 语义。 */
    const inc_dc_fw_status_t status =
        operation == INC_DC_FW_OP_DISPATCH ? MockDispatch(invocation)
                                           : MockCombine(invocation);
    if (status != INC_DC_FW_OK) return status;
    free_ticket->generation = backend->next_generation++;
    free_ticket->state = INC_DC_FW_REQUEST_SUBMITTED;
    ticket->value = index + 1u;
    ticket->generation = free_ticket->generation;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t MockQuery(
    void *opaque, inc_dc_fw_backend_ticket_t ticket, uint32_t *state)
{
    MockTicket *found = FindTicket(
        static_cast<CpuMockBackend *>(opaque), ticket);
    if (found == nullptr || state == nullptr) return INC_DC_FW_STALE_HANDLE;
    *state = found->state;
    return found->state == INC_DC_FW_REQUEST_SUBMITTED ? INC_DC_FW_NOT_READY
                                                       : INC_DC_FW_OK;
}

inc_dc_fw_status_t MockWait(
    void *opaque, inc_dc_fw_backend_ticket_t ticket, uint64_t)
{
    MockTicket *found = FindTicket(
        static_cast<CpuMockBackend *>(opaque), ticket);
    if (found == nullptr) return INC_DC_FW_STALE_HANDLE;
    found->state = INC_DC_FW_REQUEST_COMPLETED;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t MockCancel(
    void *opaque, inc_dc_fw_backend_ticket_t ticket)
{
    MockTicket *found = FindTicket(
        static_cast<CpuMockBackend *>(opaque), ticket);
    if (found == nullptr) return INC_DC_FW_STALE_HANDLE;
    found->state = INC_DC_FW_REQUEST_CANCELLED;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t MockRelease(
    void *opaque, inc_dc_fw_backend_ticket_t ticket)
{
    MockTicket *found = FindTicket(
        static_cast<CpuMockBackend *>(opaque), ticket);
    if (found == nullptr) return INC_DC_FW_STALE_HANDLE;
    if (found->state == INC_DC_FW_REQUEST_SUBMITTED) {
        return INC_DC_FW_NOT_READY;
    }
    found->state = INC_DC_FW_REQUEST_FREE;
    return INC_DC_FW_OK;
}

void *MockAllocate(uint64_t bytes, uint32_t alignment, void *opaque)
{
    auto *backend = static_cast<CpuMockBackend *>(opaque);
    void *pointer = nullptr;
    if (backend == nullptr ||
        posix_memalign(&pointer, alignment, static_cast<size_t>(bytes)) != 0) {
        return nullptr;
    }
    ++backend->allocations;
    return pointer;
}

void MockFree(void *pointer, void *opaque)
{
    auto *backend = static_cast<CpuMockBackend *>(opaque);
    if (backend != nullptr) ++backend->frees;
    std::free(pointer);
}

inc_dc_fw_backend_ops_t MakeMockBackend(CpuMockBackend *backend)
{
    inc_dc_fw_backend_ops_t ops{};
    ops.struct_size = sizeof(ops);
    ops.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    ops.backend_context = backend;
    ops.get_capabilities = MockGetCapabilities;
    ops.query_workspace = MockQueryWorkspace;
    ops.enqueue = MockEnqueue;
    ops.query = MockQuery;
    ops.wait = MockWait;
    ops.cancel = MockCancel;
    ops.release = MockRelease;
    return ops;
}

float ExpertScale(int32_t expert_id)
{
    return 1.0f + 0.1f * static_cast<float>(expert_id);
}

void PrintMatrix(const char *name, const uint16_t *matrix, uint64_t rows)
{
    std::printf("%s (%llu x %u):\n", name,
                static_cast<unsigned long long>(rows), kHidden);
    for (uint64_t row = 0u; row < rows; ++row) {
        std::printf("  row %llu:", static_cast<unsigned long long>(row));
        for (uint32_t hidden = 0u; hidden < kHidden; ++hidden) {
            std::printf(
                " %8.4f",
                inc::Fp16BitsToFloat(matrix[row * kHidden + hidden]));
        }
        std::printf("\n");
    }
}

} // namespace

int main()
{
    CpuMockBackend backend{};
    inc_dc_single_inc_config_t config{};
    const std::array<int32_t, kAssignments> expert_ids = {
        0, 2,  /* token 0 -> worker 0 / worker 1 */
        1, 3,  /* token 1 -> worker 0 / worker 1 */
        0, 3   /* token 2 -> worker 0 / worker 1 */
    };
    const std::array<float, kAssignments> route_weights = {
        0.75f, 0.25f,
        0.60f, 0.40f,
        0.50f, 0.50f
    };
    const std::array<float, kTokens * kHidden> token_input_fp32 = {
         1.0f,  2.0f,  3.0f, 4.0f,
         2.0f,  4.0f,  6.0f, 8.0f,
        -1.0f,  1.0f, -2.0f, 2.0f
    };
    std::array<uint16_t, kTokens * kHidden> token_input{};
    std::array<uint16_t, kAssignments * kHidden> dispatched_rows{};
    std::array<uint16_t, kAssignments * kHidden> expert_output{};
    std::array<uint16_t, kTokens * kHidden> combined_output{};
    std::array<float, kTokens * kHidden> expected_output{};

    inc_dc_single_inc_config_init(&config);
    config.model_id = 7u;
    config.process_group_id = 11u;
    config.worker_world_size = kWorldSize;
    config.worker_rank = 0u;
    config.hidden_size = kHidden;
    config.tokens = kTokens;
    config.topk = kTopK;
    config.max_tokens_per_chunk = 128u;
    config.backend = MakeMockBackend(&backend);
    config.allocate_device = MockAllocate;
    config.free_device = MockFree;
    config.allocator_context = &backend;

    for (size_t index = 0u; index < token_input.size(); ++index) {
        token_input[index] = inc::FloatToFp16Bits(token_input_fp32[index]);
    }

    try {
        auto require = [](inc_dc_fw_status_t status, const char *operation) {
            if (status != INC_DC_FW_OK)
                throw inc::dc::SingleIncError(operation, status);
        };

        /* Route planning is setup work; the model hot path starts at create. */
        inc_dc_easy_token_plan_desc_t plan{};
        inc_dc_easy_token_plan_desc_init(&plan);
        plan.tokens = kTokens;
        plan.topk = kTopK;
        plan.worker_world_size = kWorldSize;
        plan.worker_rank = 0u;
        plan.experts_per_worker = kExpertsPerWorker;
        plan.expert_ids = expert_ids.data();
        plan.weights = route_weights.data();
        plan.generation = kRouteGeneration;
        uint64_t route_bytes = 0u;
        require(inc_dc_easy_token_plan_query(&plan, &route_bytes),
                "token plan query");
        std::vector<uint8_t> route_blob(static_cast<size_t>(route_bytes));
        inc_dc_easy_token_plan_info_t route_info{};
        require(inc_dc_easy_token_plan_build(
                    &plan, route_blob.data(), route_blob.size(), &route_info),
                "token plan build");

        inc::dc::SingleIncRoute route{};
        inc_dc_easy_token_plan_device_route_init(
            &route.device, route_blob.data(), &route_info);
        route.dispatch_output_rows = kAssignments;
        route.combine_input_rows = kAssignments;

        std::printf("=== 单 INC 最短 API / CPU mock 完整示例 ===\n");
        auto op = inc::dc::single_inc_create(config);
        std::printf("[1/6] create 完成\n");

        auto batch = op.dispatch(
            token_input.data(), dispatched_rows.data(), route, 1u);
        std::printf("[2/6] Dispatch 完成\n");
        PrintMatrix("Dispatch fan-out", dispatched_rows.data(), kAssignments);

        for (uint64_t row = 0u; row < kAssignments; ++row) {
            const float scale = ExpertScale(expert_ids[row]);
            for (uint32_t hidden = 0u; hidden < kHidden; ++hidden) {
                expert_output[row * kHidden + hidden] = inc::FloatToFp16Bits(
                    inc::Fp16BitsToFloat(
                        dispatched_rows[row * kHidden + hidden]) * scale);
            }
        }
        std::printf("[3/6] expert compute 完成\n");

        op.combine(batch, expert_output.data(), combined_output.data(), 2u);
        std::printf("[4/6] Combine 完成\n");
        PrintMatrix("Combined token output", combined_output.data(), kTokens);

        for (uint64_t token = 0u; token < kTokens; ++token) {
            for (uint32_t slot = 0u; slot < kTopK; ++slot) {
                const uint64_t row = token * kTopK + slot;
                const float coefficient =
                    route_weights[row] * ExpertScale(expert_ids[row]);
                for (uint32_t hidden = 0u; hidden < kHidden; ++hidden) {
                    expected_output[token * kHidden + hidden] +=
                        coefficient * inc::Fp16BitsToFloat(
                            token_input[token * kHidden + hidden]);
                }
            }
        }
        for (size_t index = 0u; index < combined_output.size(); ++index) {
            const float actual = inc::Fp16BitsToFloat(combined_output[index]);
            if (std::fabs(actual - expected_output[index]) > 1.0e-2f) {
                std::fprintf(
                    stderr,
                    "正确性失败: index=%zu actual=%f expected=%f\n",
                    index, actual, expected_output[index]);
                throw std::runtime_error("golden mismatch");
            }
        }
        std::printf("[5/6] 正确性校验 PASS\n");

        inc::dc::single_inc_destroy(op);
    } catch (const inc::dc::SingleIncError &error) {
        std::fprintf(stderr, "单 INC 调用失败: %s\n", error.what());
        return EXIT_FAILURE;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "示例失败: %s\n", error.what());
        return EXIT_FAILURE;
    }

    if (backend.allocations != backend.frees) {
        std::fprintf(
            stderr, "资源泄漏: workspace alloc/free=%llu/%llu\n",
            static_cast<unsigned long long>(backend.allocations),
            static_cast<unsigned long long>(backend.frees));
        return EXIT_FAILURE;
    }
    std::printf("[6/6] destroy 完成，workspace alloc/free=%llu/%llu\n",
                static_cast<unsigned long long>(backend.allocations),
                static_cast<unsigned long long>(backend.frees));
    std::printf("示例执行成功。\n");
    return EXIT_SUCCESS;
}
