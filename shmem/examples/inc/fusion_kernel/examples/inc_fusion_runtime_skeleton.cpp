#include "inc_fusion_api.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

// 本文件刻意只依赖公开 inc_fusion_api.h。它展示生产集成的完整生命周期，
// 但不冒充一份可脱离应用运行的真机程序：SHMEM bootstrap、路由 pack、模型
// 权重与 ACL stream 都属于上层框架。把 PlatformAdapter 对接到 vLLM/Megatron
// 后，RunFusionRole() 中的 API 调用顺序无需改变。
namespace fusion_example {

enum class Role {
    kWorker,
    kInc,
};

struct ModelConfig {
    uint32_t worker_count = 4;
    uint32_t rank = 0;
    uint32_t inc_pe = 4;
    uint32_t hidden = 1024;
    uint32_t intermediate = 4096;
    uint32_t expert_count = 8;
    uint32_t topk = 2;
    uint32_t token_count = 512;
    uint32_t tokens_per_wave = 128;
    uint32_t slot_count = 3;
    uint32_t service_ring_size = 4;
    uint32_t activation_waves = 2;
    uint32_t request_count = 1;
};

// 全部字段都是当前 PE 上的设备地址；Host 指针不能填入这些位置。
// symmetric_base 必须由每个 PE 以相同顺序调用 aclshmem_malloc() 获得。
struct DeviceResources {
    void *symmetric_base = nullptr;
    uint64_t symmetric_capacity = 0;
    void *worker_stream = nullptr;
    void *service_stream = nullptr;
    void *workspace = nullptr;
    void *system_workspace = nullptr;
    uint64_t ffts_addr = 0;

    // 每个 PE 都需要的静态元数据。
    void *expert_owner = nullptr;       // uint32_t[expert_count]
    void *expert_local_index = nullptr; // uint32_t[expert_count]
    void *worker_pes = nullptr;         // uint32_t[worker_count]
    void *active_token_counts = nullptr;// uint32_t[worker_count]

    // 仅 Worker 使用；路由框架须为每次请求填充这些设备 buffer。
    void *input = nullptr;              // BF16[token_count, hidden]
    void *output = nullptr;             // BF16[token_count, hidden]
    void *w13 = nullptr;                // 本地 experts 的 gate/up 权重
    void *w2 = nullptr;                 // 本地 experts 的 down 权重
    void *dispatch_rows = nullptr;      // 已 pack 的 Dispatch 行描述
    void *assignments = nullptr;        // 已 pack 的 top-k assignment
    void *waves = nullptr;              // wave 描述数组
    void *group_lists = nullptr;        // 每 wave 的 expert prefix-sum
    uint64_t output_elements = 0;       // 输出中的 BF16 元素个数
};

// 应用需要实现的薄适配层。它只负责平台资源，不重新实现 Fusion 协议。
class PlatformAdapter {
public:
    virtual ~PlatformAdapter() = default;

    // 初始化 ACL device、SHMEM world 和两个 stream，并返回当前设备可用核数。
    virtual bool Initialize(Role role, const ModelConfig &config,
                            uint32_t *live_aiv, uint32_t *live_aic) = 0;

    // 根据 plan_info 动态分配资源：
    //   symmetric_base >= symmetric_bytes；
    //   Worker workspace >= worker_workspace_bytes（也可留空交给 executor）；
    //   INC workspace >= inc_workspace_bytes。
    // 同时把模型权重、输入、route/wave 元数据上传到设备。
    virtual bool Prepare(const ModelConfig &config,
                         const inc_fusion_plan_info_t &plan_info,
                         const std::vector<uint32_t> &expert_owner,
                         const std::vector<uint32_t> &expert_local_index,
                         DeviceResources *resources) = 0;

    // remote service 是两阶段启动：所有 PE 初始化对称镜像后只做一次 barrier，
    // 然后 INC 才能占用全部 AIV 启动常驻 kernel。
    virtual bool BarrierAll() = 0;
    virtual bool Synchronize(void *stream) = 0;

    // 把 Worker output 从设备拷回 Host，供调用者打印/校验/传给下一层。
    virtual bool CopyOutput(const DeviceResources &resources,
                            std::vector<uint16_t> *host_output) = 0;

    // 释放 ACL/SHMEM/device buffer。实现时应先保证所有 PE 已退出本次请求。
    virtual void Release(DeviceResources *resources) = 0;
};

const char *StatusName(inc_fusion_status_t status)
{
    switch (status) {
        case INC_FUSION_OK: return "OK";
        case INC_FUSION_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case INC_FUSION_INVALID_PLAN: return "INVALID_PLAN";
        case INC_FUSION_BUFFER_TOO_SMALL: return "BUFFER_TOO_SMALL";
        case INC_FUSION_BUSY: return "BUSY";
        case INC_FUSION_RUNTIME_ERROR: return "RUNTIME_ERROR";
    }
    return "UNKNOWN";
}

// 该函数就是集成方应保留的主流程：初始化 -> plan -> 绑定 -> 调用 ->
// 读取/打印 -> 停服/释放。Worker 与 INC 进程必须使用完全一致的 ModelConfig
// 和 expert placement，仅 rank/role 不同。
int RunFusionRole(Role role, const ModelConfig &config, PlatformAdapter *platform)
{
    if (platform == nullptr) return 2;

    uint32_t live_aiv = 0;
    uint32_t live_aic = 0;
    if (!platform->Initialize(role, config, &live_aiv, &live_aic)) {
        std::fprintf(stderr, "platform initialization failed\n");
        return 3;
    }

    // Round-robin expert placement；真实模型可直接传入 checkpoint 的布局。
    std::vector<uint32_t> owner(config.expert_count);
    std::vector<uint32_t> local(config.expert_count);
    std::vector<uint32_t> next(config.worker_count, 0);
    for (uint32_t expert = 0; expert < config.expert_count; ++expert) {
        owner[expert] = expert % config.worker_count;
        local[expert] = next[owner[expert]]++;
    }

    inc_fusion_plan_desc_t desc{};
    inc_fusion_plan_desc_init(&desc);
    desc.live_aiv = live_aiv;
    desc.live_aic = live_aic;
    desc.worker_count = config.worker_count;
    // INC 也必须构造同一个 rank-specific plan；约定使用 rank 0 的布局。
    desc.rank = role == Role::kWorker ? config.rank : 0;
    desc.inc_pe = config.inc_pe;
    desc.hidden = config.hidden;
    desc.intermediate = config.intermediate;
    desc.expert_count = config.expert_count;
    desc.topk = config.topk;
    desc.token_count = config.token_count;
    desc.tokens_per_wave = config.tokens_per_wave;
    desc.slot_count = config.slot_count;
    desc.service_ring_size = config.service_ring_size;
    desc.activation_waves = config.activation_waves;
    desc.spin_cap = 0; // 允许 Dispatch/Combine 任意时序交叠。

    inc_fusion_prepared_plan_t *plan = nullptr;
    inc_fusion_persistent_service_t *service = nullptr;
    inc_fusion_worker_executor_t *executor = nullptr;
    DeviceResources resources{};
    inc_fusion_plan_info_t plan_info{};
    bool setup_barrier_done = false;
    int result = 0;

    inc_fusion_status_t status = inc_fusion_prepared_plan_create(
        &desc, owner.data(), local.data(), &plan);
    if (status != INC_FUSION_OK) {
        std::fprintf(stderr, "plan create failed: %s\n", StatusName(status));
        result = 4;
        goto cleanup;
    }

    status = inc_fusion_prepared_plan_info(plan, &plan_info);
    if (status != INC_FUSION_OK ||
        !platform->Prepare(config, plan_info, owner, local, &resources) ||
        resources.symmetric_base == nullptr ||
        resources.symmetric_capacity < plan_info.symmetric_bytes) {
        std::fprintf(stderr, "buffer preparation failed (symmetric needs %llu bytes)\n",
                     static_cast<unsigned long long>(plan_info.symmetric_bytes));
        result = 5;
        goto cleanup;
    }

    if (role == Role::kWorker) {
        // 静态绑定只在初始化阶段设置一次；executor 可以自主管理 workspace。
        inc_fusion_device_bindings_t fixed{};
        fixed.symmetric_base = resources.symmetric_base;
        fixed.expert_owner = resources.expert_owner;
        fixed.expert_local_index = resources.expert_local_index;
        fixed.worker_pes = resources.worker_pes;
        fixed.active_token_counts = resources.active_token_counts;
        fixed.workspace = resources.workspace; // nullptr 表示由 executor 分配。
        fixed.system_workspace = resources.system_workspace;
        fixed.ffts_addr = resources.ffts_addr;
        fixed.execution_flags = INC_FUSION_EXEC_REMOTE_INC_SERVICE |
            INC_FUSION_EXEC_WEIGHT_B_ROW_MAJOR;

        status = inc_fusion_worker_executor_create(
            plan, &fixed, config.service_ring_size, &executor);
        if (status != INC_FUSION_OK) {
            std::fprintf(stderr, "worker executor create failed: %s\n",
                         StatusName(status));
            result = 6;
            goto cleanup;
        }
    } else {
        // INC 只绑定对称堆、workspace 和 worker PE 表；输入/权重均在 Worker。
        inc_fusion_device_bindings_t fixed{};
        fixed.symmetric_base = resources.symmetric_base;
        fixed.workspace = resources.workspace;
        fixed.system_workspace = resources.system_workspace;
        fixed.worker_pes = resources.worker_pes;
        fixed.ffts_addr = resources.ffts_addr;
        status = inc_fusion_remote_service_create(
            plan, &fixed, resources.service_stream, &service);
        if (status != INC_FUSION_OK) {
            std::fprintf(stderr, "INC service prepare failed: %s\n",
                         StatusName(status));
            result = 6;
            goto cleanup;
        }
    }

    // 所有 PE 的 remote ring 和 symmetric mirror 均初始化完成后再启动服务。
    if (!platform->BarrierAll()) {
        std::fprintf(stderr, "setup barrier failed\n");
        result = 7;
        goto cleanup;
    }
    setup_barrier_done = true;

    if (role == Role::kInc) {
        status = inc_fusion_remote_service_start(service);
        if (status != INC_FUSION_OK) {
            std::fprintf(stderr, "INC service start failed: %s\n",
                         StatusName(status));
            result = 8;
            goto cleanup;
        }

        // 示例的控制面预先知道会有 request_count 个请求，因此可按 remote
        // worker executor 发布的连续 ticket 1..N 查询。线上服务通常不逐请求
        // Host 轮询，而是由 engine shutdown 控制面通知它何时 stop。
        for (uint64_t ticket = 1; ticket <= config.request_count; ++ticket) {
            inc_fusion_service_result_t service_result{};
            do {
                status = inc_fusion_persistent_service_query(
                    service, ticket, &service_result);
                if (status == INC_FUSION_BUSY) continue;
                if (status != INC_FUSION_OK) break;
                if (service_result.complete == 0)
                    std::this_thread::sleep_for(std::chrono::microseconds(20));
            } while (service_result.complete == 0);
            if (status != INC_FUSION_OK || service_result.complete == 0 ||
                service_result.status != INC_FUSION_SERVICE_SUCCESS) {
                std::fprintf(stderr,
                    "INC request %llu failed: api=%s service=%u op=%u\n",
                    static_cast<unsigned long long>(ticket), StatusName(status),
                    service_result.status, service_result.operator_error);
                result = 9;
                goto cleanup;
            }
            std::printf("INC request %llu complete, request_id=%llu\n",
                static_cast<unsigned long long>(ticket),
                static_cast<unsigned long long>(service_result.request_id));
        }
    } else {
        for (uint32_t request = 0; request < config.request_count; ++request) {
            // 动态绑定可逐请求替换，因此能直接承接 vLLM 的输入、路由和权重。
            inc_fusion_device_bindings_t dynamic{};
            dynamic.input = resources.input;
            dynamic.output = resources.output;
            dynamic.w13 = resources.w13;
            dynamic.w2 = resources.w2;
            dynamic.dispatch_rows = resources.dispatch_rows;
            dynamic.assignments = resources.assignments;
            dynamic.waves = resources.waves;
            dynamic.active_token_counts = resources.active_token_counts;
            dynamic.group_lists = resources.group_lists;
            dynamic.execution_flags = INC_FUSION_EXEC_WEIGHT_B_ROW_MAJOR;

            // 一个请求会占用 [generation+1, generation+wave_count]；下一
            // 请求必须跨过整个区间，不能简单 generation++，否则不同请求
            // 的 wave generation 会别名。
            const uint64_t generation = 100ULL +
                static_cast<uint64_t>(request) * (plan_info.wave_count + 1ULL);
            do {
                status = inc_fusion_worker_executor_enqueue(
                    executor, generation, &dynamic, resources.worker_stream);
                // ring 发生背压时等待已提交 stream，再重试同一 generation；
                // 绝不能跳过 generation 或覆盖仍在途的 slot。
                if (status == INC_FUSION_BUSY &&
                    !platform->Synchronize(resources.worker_stream)) {
                    status = INC_FUSION_RUNTIME_ERROR;
                }
            } while (status == INC_FUSION_BUSY);
            if (status != INC_FUSION_OK) {
                std::fprintf(stderr, "worker enqueue %u failed: %s\n",
                             request, StatusName(status));
                result = 8;
                goto cleanup;
            }
        }
        if (!platform->Synchronize(resources.worker_stream)) {
            std::fprintf(stderr, "worker stream synchronization failed\n");
            result = 9;
            goto cleanup;
        }

        std::vector<uint16_t> output;
        if (!platform->CopyOutput(resources, &output)) {
            std::fprintf(stderr, "output copy failed\n");
            result = 10;
            goto cleanup;
        }
        // 示例用轻量 checksum 展示结果已可读；正式正确性 gate 应与 BF16
        // golden/reference 对比，而不能只看 checksum。
        uint64_t checksum = 1469598103934665603ULL;
        for (uint16_t value : output) {
            checksum ^= value;
            checksum *= 1099511628211ULL;
        }
        std::printf("Worker rank %u output: elements=%zu checksum=0x%016llx\n",
                    config.rank, output.size(),
                    static_cast<unsigned long long>(checksum));
    }

cleanup:
    // 先停常驻 kernel，再销毁 service；worker executor 的 destroy 会等待
    // 自己 ring 中仍在途的 event。二者都必须发生在 plan 销毁之前。
    if (service != nullptr) {
        (void)inc_fusion_persistent_service_stop(service);
        inc_fusion_persistent_service_destroy(service);
    }
    inc_fusion_worker_executor_destroy(executor);
    // 正常有限请求示例在释放 symmetric heap 前做 teardown barrier。异常
    // 恢复时生产框架应走自己的 peer-failure/abort 路径，不能无限等失联 PE。
    if (setup_barrier_done) (void)platform->BarrierAll();
    inc_fusion_prepared_plan_destroy(plan);
    platform->Release(&resources);
    return result;
}

} // namespace fusion_example

int main()
{
    // 这里不构造假设备地址，也不声称完成了 W+1 真机运行。实际应用应实现
    // PlatformAdapter，然后调用：
    //   RunFusionRole(Role::kWorker, config, &adapter);  // Worker PE
    //   RunFusionRole(Role::kInc,    config, &adapter);  // INC PE
    // 完整可运行的 SHMEM 资格化驱动见 ascend/tests/inc_fusion_e2e_main.cpp。
    std::puts("INC Fusion runtime skeleton compiled successfully.");
    std::puts("Implement PlatformAdapter and call RunFusionRole() for real NPU execution.");
    return 0;
}
