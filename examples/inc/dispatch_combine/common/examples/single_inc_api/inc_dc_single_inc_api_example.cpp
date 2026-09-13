/*
 * 单 INC 最短应用 API 完整示例 / Complete single-INC application API example
 * -------------------------------------------------------------------------
 *
 * 中文
 *   业务侧只应走 inc_dc_single_inc.hpp 的这条链：
 *     create → dispatch → (外部 expert compute) → combine → destroy
 *   SingleIncBatch 持有 Dispatch 捕获的 route；Combine 成功后释放，析构也会释放。
 *
 *   本程序用 CPU mock backend，不占用 NPU，也能走完正式 API 生命周期
 *   （配置 / 链接本仓库仍需 CANN / BiSheng）。mock 只替换三件事：
 *     1) backend ops（enqueue / query / wait / release）
 *     2) device allocator（posix_memalign）
 *     3) stream（这里只是一个整数标签）
 *   session、prepared plan、workspace、request/route handle 全部走正式公共 API。
 *
 *   模拟数据流（W=2 workers, top-k=2, hidden=4）：
 *     token [3, 4]
 *       --Dispatch--> expert rows [6, 4]     // 每个 token 复制到 2 个 expert 行
 *       --模拟 expert compute--> [6, 4]      // 按 expert_id 做标量缩放
 *       --Combine weighted reduce--> [3, 4]  // 按路由权重加回源 token
 *
 * English
 *   Application code should only use inc_dc_single_inc.hpp:
 *     create → dispatch → (external expert compute) → combine → destroy
 *   SingleIncBatch owns the captured Dispatch route. Combine consumes it;
 *   the destructor also releases it if combine never runs.
 *
 *   This program uses a CPU mock backend so it can run the full public API
 *   lifecycle without an NPU (building/linking still needs CANN / BiSheng).
 *   The mock replaces only backend ops, the device allocator, and the stream
 *   handle. Session / plan / workspace / route handles still go through the
 *   real public API.
 *
 *   Toy data flow (2 workers, top-k=2, hidden=4):
 *     token [3, 4] --Dispatch--> expert rows [6, 4]
 *                  --fake GEMM--> [6, 4]
 *                  --Combine--> token output [3, 4]
 *
 * 注意 / Note
 *   mock 在 enqueue 内立刻算完，只验证 API、数值和生命周期，
 *   不是 INC 带宽或 Dispatch/Combine 交叠的证据。
 *   The mock finishes inside enqueue; it is not bandwidth or overlap evidence.
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

/* 固定 toy shape，方便对照打印矩阵。 / Fixed toy shape for readable dumps. */
constexpr uint64_t kTokens = 3u;
constexpr uint32_t kHidden = 4u;
constexpr uint32_t kTopK = 2u;
constexpr uint64_t kAssignments = kTokens * kTopK; /* Dispatch 输出行数 / fan-out rows */
constexpr uint32_t kWorldSize = 2u;                /* worker 数，不含 INC / workers only */
constexpr uint32_t kExpertsPerWorker = 2u;         /* expert_id = rank * E + local */
constexpr uint64_t kRouteGeneration = 101u;        /* 与 token-plan / Dispatch 对齐 */

/*
 * MockTicket
 *   中文：模拟异步请求的一代 ticket。真机 backend 通常握着 device event；
 *        这里只用 generation + 状态机演示 query / wait / release。
 *   English: Stand-in for an async request ticket. A real backend would keep a
 *            device event; here generation + state show query/wait/release.
 */
struct MockTicket {
    uint64_t generation = 0u;
    uint32_t state = INC_DC_FW_REQUEST_FREE;
};

struct CpuMockBackend {
    std::array<MockTicket, 8> tickets{};
    uint64_t next_generation = 1u;
    uint64_t allocations = 0u; /* workspace 分配次数 / allocator calls */
    uint64_t frees = 0u;
};

float BitsToFloat(uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

/*
 * ParseRoute
 *   中文：校验 inc_dc_easy_token_plan_build() 写出的公开 token-plan ABI，
 *        并取出 header / assignment 表。真机 backend 从 HBM 读同一布局；
 *        mock 直接读 host 上的 vector。
 *   English: Validate the public token-plan ABI produced by
 *            inc_dc_easy_token_plan_build() and return header/assignments.
 *            A device backend reads the same layout from HBM; the mock reads
 *            the host vector in place.
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
    /* generation 对不上就是 stale handle。 / Mismatched generation => stale. */
    return found.generation == ticket.generation ? &found : nullptr;
}

inc_dc_fw_status_t MockGetCapabilities(
    void *, inc_dc_fw_capabilities_t *capabilities)
{
    if (capabilities == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    *capabilities = {};
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    /*
     * 只声明本示例真正实现的最小能力，避免把 mock 误当成并发性能 backend。
     * Advertise only what this mock implements; it is not a perf backend.
     */
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
    /*
     * 只要非零、对齐的 mock workspace，让 create 能走完 allocator。
     * 真机由 native backend 按 kernel 精确 query。
     * Non-zero aligned bytes are enough for create(); native backends query
     * the real kernel workspace.
     */
    workspace->bytes = 256u + shape->tokens * shape->topk * sizeof(uint32_t) +
        (operation == INC_DC_FW_OP_COMBINE ? 128u : 0u);
    return INC_DC_FW_OK;
}

/*
 * MockDispatch
 *   中文：按 token-plan 把每个 token 复制 top-k 份，写成 expert 输入行。
 *        行序固定为 token-major：row = token * topk + slot。
 *        真机还会按 destination_rank 经 INC fan-out 到专家所在 worker。
 *   English: Replicate each token top-k times into expert-input rows.
 *            Row order is token-major: row = token * topk + slot.
 *            A real backend also ships the row to destination_rank via INC.
 */
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
            for (uint32_t hidden = 0u; hidden < invocation->shape.hidden_size;
                 ++hidden) {
                fanout[row * invocation->shape.hidden_size + hidden] =
                    input[token * invocation->shape.hidden_size + hidden];
            }
        }
    }
    (void)assignments; /* 行序由 token-plan 固定 / row order is plan-defined */
    return INC_DC_FW_OK;
}

/*
 * MockCombine
 *   中文：把 expert 输出按 assignment.weight 加权加回源 token。
 *        这是 Combine 的语义金标准；真机在 INC 上做同样的加权归约再 fan-back。
 *   English: Weighted-reduce expert rows back onto source tokens using
 *            assignment.weight. Same semantics as the device Combine path.
 */
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

/*
 * MockEnqueue
 *   中文：公开 API 把 Dispatch/Combine 提交看成异步 enqueue。mock 立刻算完，
 *        但状态先停在 SUBMITTED，好让后续 query 返回 NOT_READY、wait 再完成。
 *   English: The public API treats Dispatch/Combine as async enqueue. The mock
 *            computes immediately, then leaves the ticket SUBMITTED so query
 *            can return NOT_READY until wait completes it.
 */
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
    /* 仍在飞行中的 ticket 不能回收。 / In-flight tickets cannot be recycled. */
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

/* 假装 grouped-GEMM：按 expert_id 做不同缩放。 / Fake grouped GEMM via scale. */
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

    /*
     * 路由表 / routing table
     *   expert_id = destination_rank * experts_per_worker + local_expert
     *   token 0 → experts 0,2（worker0 / worker1）
     *   token 1 → experts 1,3
     *   token 2 → experts 0,3
     * 权重与 Combine 金标准共用。 / Weights are reused by the Combine golden.
     */
    const std::array<int32_t, kAssignments> expert_ids = {
        0, 2,
        1, 3,
        0, 3
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

    /* 公开 API 吃 FP16 bit pattern。 / Public API consumes FP16 bit patterns. */
    std::array<uint16_t, kTokens * kHidden> token_input{};
    std::array<uint16_t, kAssignments * kHidden> dispatched_rows{};
    std::array<uint16_t, kAssignments * kHidden> expert_output{};
    std::array<uint16_t, kTokens * kHidden> combined_output{};
    std::array<float, kTokens * kHidden> expected_output{};

    /*
     * 部署配置只填一次：world / shape / backend / allocator。
     * Deployment fills world, shape, backend, and allocator once.
     * 真机把 backend 换成 native_composite_backend，allocator 换成 ACL，
     * 再 BindNativeSingleIncWorkerControl。
     * On device: native_composite_backend + ACL allocator + bind INC service.
     */
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

        /*
         * Token-plan 是 setup，不是模型热路径。
         * Token-plan construction is setup, not the model hot path.
         * create 之后热路径只有 dispatch / expert / combine。
         * After create the hot path is only dispatch / expert / combine.
         */
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

        /* [1] 创建 operator：编译 plan、向 mock 要 workspace。
         * [1] Create the operator: compile plans and allocate workspace. */
        auto op = inc::dc::single_inc_create(config);
        std::printf("[1/6] create 完成\n");

        /* [2] Dispatch：token -> expert 输入行；batch 持有捕获的 route。
         * [2] Dispatch tokens into expert rows; batch owns the captured route.
         * 最后一个参数是 stream 标签（真机为 aclrtStream）。
         * The last argument is a stream tag (aclrtStream on device). */
        auto batch = op.dispatch(
            token_input.data(), dispatched_rows.data(), route, 1u);
        std::printf("[2/6] Dispatch 完成\n");
        PrintMatrix("Dispatch fan-out", dispatched_rows.data(), kAssignments);

        /* [3] 框架侧 expert compute，不属于 INC API。
         * [3] Framework expert compute; not part of the INC API. */
        for (uint64_t row = 0u; row < kAssignments; ++row) {
            const float scale = ExpertScale(expert_ids[row]);
            for (uint32_t hidden = 0u; hidden < kHidden; ++hidden) {
                expert_output[row * kHidden + hidden] = inc::FloatToFp16Bits(
                    inc::Fp16BitsToFloat(
                        dispatched_rows[row * kHidden + hidden]) * scale);
            }
        }
        std::printf("[3/6] expert compute 完成\n");

        /* [4] Combine：消费 batch 里的 route，写回 token 布局并释放 route。
         * [4] Combine consumes the batch route, writes tokens, releases route. */
        op.combine(batch, expert_output.data(), combined_output.data(), 2u);
        std::printf("[4/6] Combine 完成\n");
        PrintMatrix("Combined token output", combined_output.data(), kTokens);

        /* [5] 金标准：out[t] = Σ_k weight[t,k] * scale(expert[t,k]) * in[t]
         * [5] Golden: same weighted sum the mock Combine should have produced. */
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

        /* [6] 显式 destroy；若漏调用，SingleInc 析构仍会尝试释放。
         * [6] Explicit destroy; the SingleInc destructor also tries to free. */
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
