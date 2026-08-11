#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "param.h"
#include "shmem.h"
#include "utils.h"

#include "inc_dc_easy_api.h"
#include "inc_dc_native_combine_backend.h"
#include "inc_dc_native_composite_backend.h"
#include "inc_dc_native_dispatch_backend.h"
#include "inc_dc_native_expert_layout_adapter.h"
#include "inc_dc_native_inc_service.h"
#include "inc_dc_single_inc_combine_plan_compiler.h"
#include "inc_dc_fp16_host.h"

using namespace inc;
using namespace inc::dc::single_stream;

int g_npus = 16;
const char *ipport = "tcp://127.0.0.1:8969";
int f_npu = 0;
aclshmemx_uniqueid_t default_flag_uid;

namespace {

constexpr uint64_t kTimeoutNs = 30ull * 1000ull * 1000ull * 1000ull;

int ResolveDevice(int pe)
{
    const char *raw = std::getenv("INC_PE_TO_NPU_MAP");
    if (raw != nullptr && raw[0] != '\0') {
        std::string map(raw);
        size_t position = 0u;
        while (position < map.size()) {
            const size_t comma = map.find(',', position);
            const std::string item = map.substr(
                position, comma == std::string::npos ? std::string::npos
                                                      : comma - position);
            const size_t colon = item.find(':');
            if (colon != std::string::npos &&
                std::atoi(item.substr(0u, colon).c_str()) == pe)
                return std::atoi(item.substr(colon + 1u).c_str());
            if (comma == std::string::npos) break;
            position = comma + 1u;
        }
    }
    return pe % g_npus + f_npu;
}

uint16_t InputValue(uint32_t rank, uint32_t row)
{
    return FloatToFp16Bits(static_cast<float>(rank * 16u + row % 13u + 1u));
}

uint32_t EnvU32(const char *name)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') return 0u;
    const unsigned long value = std::strtoul(raw, nullptr, 10);
    return value <= std::numeric_limits<uint32_t>::max()
        ? static_cast<uint32_t>(value) : 0u;
}

bool MakeSourcePlan(uint32_t source, uint32_t world, uint32_t tokens,
                    uint32_t topk, uint32_t hidden_bytes,
                    std::vector<uint8_t> *wire,
                    StreamCompiledSourcePlan *compiled)
{
    const uint64_t count = static_cast<uint64_t>(tokens) * topk;
    std::vector<int32_t> experts(count);
    std::vector<uint32_t> destinations(count);
    std::vector<float> weights(count, 1.0f);
    for (uint32_t token = 0u; token < tokens; ++token) {
        for (uint32_t ordinal = 0u; ordinal < topk; ++ordinal) {
            const uint32_t destination =
                (source * 3u + token + ordinal) % world;
            const uint64_t index =
                static_cast<uint64_t>(token) * topk + ordinal;
            destinations[index] = destination;
            experts[index] =
                static_cast<int32_t>(destination * topk + ordinal);
        }
    }
    inc_dc_easy_token_plan_desc_t description{};
    inc_dc_easy_token_plan_desc_init(&description);
    description.tokens = tokens;
    description.topk = topk;
    description.worker_world_size = world;
    description.worker_rank = source;
    description.experts_per_worker = topk;
    description.expert_ids = experts.data();
    description.destination_ranks = destinations.data();
    description.weights = weights.data();
    description.generation = 1u;
    uint64_t bytes = 0u;
    if (inc_dc_easy_token_plan_query(&description, &bytes) != INC_DC_FW_OK)
        return false;
    wire->resize(bytes);
    inc_dc_easy_token_plan_info_t info{};
    if (inc_dc_easy_token_plan_build(
            &description, wire->data(), wire->size(), &info) != INC_DC_FW_OK)
        return false;
    StreamPlanCompileInput input{};
    input.source_rank = source;
    input.worker_world_size = world;
    input.hidden_bytes = hidden_bytes;
    input.tile_rows = ResolveStreamTileRows(tokens, hidden_bytes);
    input.max_routes_per_packet = ResolveStreamPacketRows(hidden_bytes);
    input.host_token_plan = wire->data();
    input.host_token_plan_bytes = wire->size();
    return CompileStreamSourcePlan(input, compiled);
}

void Cleanup(int device, aclrtStream stream, uint8_t *dispatch_heap,
             uint8_t *combine_heap)
{
    if (combine_heap != nullptr) aclshmem_free(combine_heap);
    if (dispatch_heap != nullptr) aclshmem_free(dispatch_heap);
    aclshmem_finalize();
    if (stream != nullptr) (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(device);
    (void)aclFinalize();
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 9 || argc > 10) {
        std::cerr << "usage: native_full NPES PE IPPORT NPUS FIRST_NPU "
                     "TOKENS HIDDEN_BYTES TOPK [GENERATIONS]\n";
        return 2;
    }
    const int npes = std::atoi(argv[1]);
    const int pe = std::atoi(argv[2]);
    ipport = argv[3];
    g_npus = std::atoi(argv[4]);
    f_npu = std::atoi(argv[5]);
    const uint32_t tokens =
        static_cast<uint32_t>(std::strtoul(argv[6], nullptr, 10));
    const uint32_t hidden_bytes =
        static_cast<uint32_t>(std::strtoul(argv[7], nullptr, 10));
    const uint32_t topk =
        static_cast<uint32_t>(std::strtoul(argv[8], nullptr, 10));
    const uint32_t generations = argc > 9
        ? static_cast<uint32_t>(std::strtoul(argv[9], nullptr, 10)) : 1u;
    const uint32_t world = npes > 0 ? static_cast<uint32_t>(npes - 1) : 0u;
    if (world < 2u || pe < 0 || pe >= npes || tokens == 0u ||
        hidden_bytes == 0u || (hidden_bytes & 1u) != 0u || topk == 0u ||
        generations == 0u) return 2;

    std::vector<std::vector<uint8_t>> wires(world);
    std::vector<StreamCompiledSourcePlan> sources(world);
    for (uint32_t source = 0u; source < world; ++source) {
        if (!MakeSourcePlan(source, world, tokens, topk, hidden_bytes,
                            &wires[source], &sources[source])) return 3;
    }
    StreamCompiledGlobalPlan dispatch_plan{};
    CombineReverseLayout reverse{};
    if (!MergeStreamSourcePlans(sources, &dispatch_plan) ||
        !BuildCombineReverseLayout(dispatch_plan, &reverse)) return 3;
    std::vector<std::vector<uint32_t>> configured_experts(world);
    for (uint32_t rank = 0u; rank < world; ++rank) {
        for (uint32_t ordinal = 0u; ordinal < topk; ++ordinal)
            configured_experts[rank].push_back(rank * topk + ordinal);
    }
    NativeExpertLayout expert_layout{};
    constexpr uint32_t expert_alignment = 8u;
    if (!BuildNativeExpertLayout(
            dispatch_plan, reverse, configured_experts,
            expert_alignment, &expert_layout)) return 3;

    const int device = ResolveDevice(pe);
    aclrtStream stream = nullptr;
    if (aclInit(nullptr) != ACL_SUCCESS ||
        aclrtSetDevice(device) != ACL_SUCCESS ||
        aclrtCreateStream(&stream) != ACL_SUCCESS) return 4;
    constexpr uint64_t capacity = 4ull * 1024ull * 1024ull * 1024ull;
    aclshmemx_init_attr_t attr{};
    test_set_attr(pe, npes, capacity, ipport, default_flag_uid, &attr);
    if (aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr) != 0) {
        Cleanup(device, stream, nullptr, nullptr);
        return 4;
    }
    int64_t live_aiv_raw = 0;
    if (aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                           &live_aiv_raw) != ACL_SUCCESS || live_aiv_raw <= 0) {
        Cleanup(device, stream, nullptr, nullptr);
        return 4;
    }
    const uint32_t live_aiv = static_cast<uint32_t>(live_aiv_raw);
    StreamWorkspaceBuildInput dispatch_input{};
    dispatch_input.live_aiv = live_aiv;
    dispatch_input.tile_rows =
        ResolveStreamTileRows(tokens, hidden_bytes);
    StreamPreparedWorkspace dispatch_workspace{};
    NativeCombinePreparedWorkspace combine_workspace{};
    if (!BuildStreamPreparedWorkspace(
            dispatch_plan, dispatch_input, &dispatch_workspace) ||
        !BuildNativeCombinePreparedWorkspace(
            reverse.logical_plan, reverse.contributor_rows,
            hidden_bytes / 2u, live_aiv, &combine_workspace)) {
        Cleanup(device, stream, nullptr, nullptr);
        return 5;
    }
    uint8_t *dispatch_heap = static_cast<uint8_t *>(
        aclshmem_malloc(dispatch_workspace.descriptor.total_bytes));
    uint8_t *combine_heap = static_cast<uint8_t *>(
        aclshmem_malloc(combine_workspace.heap_bytes));
    if (dispatch_heap == nullptr || combine_heap == nullptr) {
        Cleanup(device, stream, dispatch_heap, combine_heap);
        return 5;
    }
    NativeDispatchSessionConfig dispatch_config{};
    dispatch_config.symmetric_heap = dispatch_heap;
    dispatch_config.symmetric_heap_bytes =
        dispatch_workspace.descriptor.total_bytes;
    dispatch_config.local_pe = static_cast<uint32_t>(pe);
    dispatch_config.prepared = &dispatch_workspace;
    NativeCombineSessionConfig combine_config{};
    combine_config.symmetric_heap = combine_heap;
    combine_config.symmetric_heap_bytes = combine_workspace.heap_bytes;
    combine_config.local_pe = static_cast<uint32_t>(pe);
    combine_config.tokens_per_worker = tokens;
    combine_config.topk = topk;
    combine_config.prepared = &combine_workspace;
    combine_config.source_semantic_digests =
        &dispatch_plan.source_semantic_digests;
    NativeDispatchSession *dispatch_native = nullptr;
    NativeCombineSession *combine_native = nullptr;
    if (CreateNativeDispatchSession(
            dispatch_config, &dispatch_native) != INC_DC_FW_OK ||
        CreateNativeCombineSession(
            combine_config, &combine_native) != INC_DC_FW_OK) {
        if (dispatch_native != nullptr)
            (void)DestroyNativeDispatchSession(dispatch_native);
        Cleanup(device, stream, dispatch_heap, combine_heap);
        return 5;
    }

    const uint32_t service_port_raw = EnvU32("INC_NATIVE_SERVICE_PORT");
    if (service_port_raw == 0u ||
        service_port_raw > std::numeric_limits<uint16_t>::max()) {
        (void)DestroyNativeCombineSession(combine_native);
        (void)DestroyNativeDispatchSession(dispatch_native);
        Cleanup(device, stream, dispatch_heap, combine_heap);
        return 5;
    }
    NativeIncService *inc_service = nullptr;
    NativeIncServiceClient *service_client = nullptr;
    int rc = 0;
    if (static_cast<uint32_t>(pe) == world) {
        NativeIncServiceConfig service_config{};
        service_config.worker_world_size = world;
        service_config.tcp_port = static_cast<uint16_t>(service_port_raw);
        service_config.stream = reinterpret_cast<uint64_t>(stream);
        service_config.timeout_ns = kTimeoutNs;
        service_config.dispatch = dispatch_native;
        service_config.combine = combine_native;
        if (CreateNativeIncService(
                service_config, &inc_service) != INC_DC_FW_OK) rc = 5;
    } else if (CreateNativeIncServiceClient(
                   "127.0.0.1", static_cast<uint16_t>(service_port_raw),
                   static_cast<uint32_t>(pe), kTimeoutNs,
                   &service_client) != INC_DC_FW_OK) {
        rc = 5;
    }

    if (static_cast<uint32_t>(pe) == world) {
        if (rc == 0 && NativeIncServiceRun(
                inc_service, generations) != INC_DC_FW_OK) rc = 6;
    } else {
        const uint32_t rank = static_cast<uint32_t>(pe);
        const uint32_t fault_pe = EnvU32("INC_NATIVE_FULL_FAULT_PE");
        const uint32_t dispatch_fault_generation =
            EnvU32("INC_NATIVE_FULL_DISPATCH_FAULT_GENERATION");
        const uint32_t combine_fault_generation =
            EnvU32("INC_NATIVE_FULL_COMBINE_FAULT_GENERATION");
        void *dispatch_tensor = nullptr;
        void *dispatch_output = nullptr;
        void *combine_input = nullptr;
        void *combine_output = nullptr;
        void *expert_buffer = nullptr;
        uint64_t dispatch_tensor_bytes = 0u;
        uint64_t dispatch_output_bytes = 0u;
        uint64_t combine_input_bytes = 0u;
        uint64_t combine_output_bytes = 0u;
        void *route = nullptr;
        void *dispatch_scratch = nullptr;
        void *combine_scratch = nullptr;
        inc_dc_easy_comm_t *comm = nullptr;
        NativeCompositeBackend *composite = nullptr;
        inc_dc_fw_workspace_t dispatch_ws{};
        inc_dc_fw_workspace_t combine_ws{};
        inc_dc_fw_backend_ops_t dispatch_ops{};
        inc_dc_fw_backend_ops_t combine_ops{};
        inc_dc_fw_backend_ops_t composite_ops{};
        if (NativeDispatchWorkerBuffers(
                dispatch_native, &dispatch_tensor, &dispatch_tensor_bytes,
                &dispatch_output, &dispatch_output_bytes) != INC_DC_FW_OK ||
            NativeCombineWorkerBuffers(
                combine_native, &combine_input, &combine_input_bytes,
                &combine_output, &combine_output_bytes) != INC_DC_FW_OK ||
            NativeDispatchBackendOps(dispatch_native, &dispatch_ops) !=
                INC_DC_FW_OK ||
            NativeCombineBackendOps(combine_native, &combine_ops) !=
                INC_DC_FW_OK ||
            CreateNativeCompositeBackend(
                dispatch_ops, combine_ops, &composite) != INC_DC_FW_OK ||
            NativeCompositeBackendOps(composite, &composite_ops) !=
                INC_DC_FW_OK) rc = 6;

        const uint64_t row_elements = hidden_bytes / 2u;
        std::vector<uint16_t> host_input(dispatch_tensor_bytes / 2u);
        std::vector<uint16_t> host_output(combine_output_bytes / 2u, 0u);
        for (uint32_t row = 0u; row < tokens; ++row) {
            std::fill_n(host_input.data() +
                            static_cast<uint64_t>(row) * row_elements,
                        row_elements, InputValue(rank, row));
        }
        if (rc == 0 &&
            (aclrtMalloc(&route, wires[rank].size(),
                         ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
             aclrtMalloc(
                 &expert_buffer,
                 expert_layout.ranks[rank].padded_row_count * hidden_bytes,
                 ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
             aclrtMemcpy(dispatch_tensor, dispatch_tensor_bytes,
                         host_input.data(), dispatch_tensor_bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
             aclrtMemcpy(route, wires[rank].size(), wires[rank].data(),
                         wires[rank].size(), ACL_MEMCPY_HOST_TO_DEVICE) !=
                 ACL_SUCCESS)) rc = 6;
        if (rc == 0) {
            inc_dc_easy_comm_config_t config{};
            inc_dc_easy_comm_config_init(&config);
            config.session_id = 3u;
            config.model_id = 1u;
            config.process_group_id = 1u;
            config.topology_generation = 1u;
            config.worker_world_size = world;
            config.worker_rank = rank;
            config.max_tokens_per_chunk = tokens;
            config.hidden_size = hidden_bytes / 2u;
            config.max_topk = topk;
            config.dtype = INC_DC_FW_DTYPE_FP16;
            config.max_inflight = 1u;
            config.max_workspace_queries = 2u;
            config.backend = composite_ops;
            inc_dc_easy_token_plan_info_t info{};
            const auto *header = reinterpret_cast<
                const inc_dc_easy_token_plan_header_v1_t *>(
                    wires[rank].data());
            info.struct_size = sizeof(info);
            info.abi_version = INC_DC_EASY_ABI_VERSION;
            info.bytes = wires[rank].size();
            info.semantic_digest = header->semantic_digest;
            info.generation = 1u;
            inc_dc_easy_token_plan_device_route_init(
                &config.static_route, route, &info);
            if (inc_dc_easy_comm_create(&config, &comm) != INC_DC_FW_OK)
                rc = 6;
        }
        if (rc == 0) {
            dispatch_ws.struct_size = sizeof(dispatch_ws);
            combine_ws.struct_size = sizeof(combine_ws);
            if (inc_dc_easy_workspace_query(
                    comm, INC_DC_FW_OP_DISPATCH, tokens, topk,
                    &dispatch_ws) != INC_DC_FW_OK ||
                inc_dc_easy_workspace_query(
                    comm, INC_DC_FW_OP_COMBINE, tokens, topk,
                    &combine_ws) != INC_DC_FW_OK ||
                aclrtMalloc(&dispatch_scratch, dispatch_ws.bytes,
                            ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
                aclrtMalloc(&combine_scratch, combine_ws.bytes,
                            ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) rc = 6;
        }
        for (uint32_t epoch = 0u; epoch < generations && rc == 0; ++epoch) {
            const uint64_t generation = epoch + 1u;
            inc_dc_easy_op_t dispatch_op{};
            inc_dc_easy_op_init(&dispatch_op);
            dispatch_op.tokens = tokens;
            dispatch_op.output_rows =
                dispatch_workspace.destination_physical_rows[rank];
            dispatch_op.topk = topk;
            dispatch_op.input = dispatch_tensor;
            dispatch_op.output = dispatch_output;
            dispatch_op.workspace = dispatch_scratch;
            dispatch_op.workspace_bytes = dispatch_ws.bytes;
            dispatch_op.workspace_query_token = dispatch_ws.query_token;
            dispatch_op.stream = reinterpret_cast<uint64_t>(stream);
            dispatch_op.operation_generation = generation;
            inc_dc_fw_request_t dispatch_request{};
            inc_dc_fw_route_handle_t route_handle{};
            if (NativeDispatchPrepareGeneration(
                    dispatch_native, generation) != INC_DC_FW_OK) rc = 6;
            if (rc == 0 && NativeIncServiceSubmitAndWait(
                    service_client, NativeIncServiceOp::DISPATCH_PREPARE,
                    generation) != INC_DC_FW_OK) rc = 6;
            const bool expect_dispatch_fault = rank == fault_pe &&
                generation == dispatch_fault_generation;
            if (expect_dispatch_fault && NativeDispatchArmLaneError(
                    dispatch_native, 0u, 0xd15a0001u) != INC_DC_FW_OK)
                rc = 6;
            if (rc == 0 &&
                (inc_dc_easy_dispatch_async(
                     comm, &dispatch_op, &dispatch_request) != INC_DC_FW_OK ||
                 inc_dc_easy_route_handle_create(
                     comm, dispatch_request, &dispatch_op,
                     &route_handle) != INC_DC_FW_OK)) rc = 6;
            if (rc == 0 && NativeIncServiceSubmitAndWait(
                    service_client, NativeIncServiceOp::DISPATCH,
                    generation) != INC_DC_FW_OK) rc = 6;
            if (rc == 0) {
                const inc_dc_fw_status_t wait_status =
                    inc_dc_easy_request_wait(
                        comm, dispatch_request, kTimeoutNs);
                if ((!expect_dispatch_fault && wait_status != INC_DC_FW_OK) ||
                    (expect_dispatch_fault &&
                     wait_status != INC_DC_FW_BACKEND_ERROR) ||
                    inc_dc_easy_request_release(
                        comm, dispatch_request) != INC_DC_FW_OK) {
                    rc = 6;
                } else if (expect_dispatch_fault) {
                    std::cout << "NATIVE_FULL_FAULT_OBSERVED pe=" << pe
                              << " op=dispatch generation=" << generation
                              << std::endl;
                }
            }
            const auto &rank_layout = expert_layout.ranks[rank];
            for (uint64_t row = 0u;
                 row < rank_layout.dispatch_rows.size() && rc == 0; ++row) {
                if (aclrtMemcpyAsync(
                        static_cast<uint8_t *>(expert_buffer) +
                            rank_layout.padded_rows[row] * hidden_bytes,
                        hidden_bytes,
                        static_cast<const uint8_t *>(dispatch_output) +
                            static_cast<uint64_t>(
                                rank_layout.dispatch_rows[row]) *
                                hidden_bytes,
                        hidden_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                        stream) != ACL_SUCCESS) rc = 6;
            }
            // Identity expert compute: production grouped GEMM consumes and
            // produces the same padded expert-major rows. The inverse map is
            // then the only path into Combine contributor order.
            for (uint64_t combine_row = 0u;
                 combine_row < rank_layout.combine_row_to_padded_row.size() &&
                 rc == 0; ++combine_row) {
                if (aclrtMemcpyAsync(
                        static_cast<uint8_t *>(combine_input) +
                            combine_row * hidden_bytes,
                        hidden_bytes,
                        static_cast<const uint8_t *>(expert_buffer) +
                            rank_layout.combine_row_to_padded_row[combine_row] *
                                hidden_bytes,
                        hidden_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                        stream) != ACL_SUCCESS) rc = 6;
            }
            inc_dc_easy_op_t combine_op{};
            inc_dc_easy_op_init(&combine_op);
            combine_op.tokens = tokens;
            combine_op.input_rows = reverse.contributor_rows[rank];
            combine_op.output_rows = tokens;
            combine_op.topk = topk;
            combine_op.input = combine_input;
            combine_op.output = combine_output;
            combine_op.workspace = combine_scratch;
            combine_op.workspace_bytes = combine_ws.bytes;
            combine_op.workspace_query_token = combine_ws.query_token;
            combine_op.stream = reinterpret_cast<uint64_t>(stream);
            combine_op.operation_generation = generation;
            inc_dc_fw_request_t combine_request{};
            const bool expect_combine_fault = rank == fault_pe &&
                generation == combine_fault_generation;
            if (expect_combine_fault && NativeCombineArmProducerError(
                    combine_native, 0u, 0xc0ab0001u) != INC_DC_FW_OK)
                rc = 6;
            if (rc == 0 && inc_dc_easy_combine_with_route_async(
                    comm, route_handle, &combine_op,
                    &combine_request) != INC_DC_FW_OK) rc = 6;
            if (rc == 0 && NativeIncServiceSubmitAndWait(
                    service_client, NativeIncServiceOp::COMBINE,
                    generation) != INC_DC_FW_OK) rc = 6;
            if (rc == 0) {
                const inc_dc_fw_status_t wait_status =
                    inc_dc_easy_request_wait(
                        comm, combine_request, kTimeoutNs);
                if ((!expect_combine_fault && wait_status != INC_DC_FW_OK) ||
                    (expect_combine_fault &&
                     wait_status != INC_DC_FW_BACKEND_ERROR) ||
                    inc_dc_easy_request_release(
                        comm, combine_request) != INC_DC_FW_OK) {
                    rc = 6;
                } else if (expect_combine_fault) {
                    std::cout << "NATIVE_FULL_FAULT_OBSERVED pe=" << pe
                              << " op=combine generation=" << generation
                              << std::endl;
                }
            }
            if (route_handle.slot != 0u &&
                inc_dc_easy_route_handle_release(
                    comm, route_handle) != INC_DC_FW_OK && rc == 0) rc = 6;
        }
        if (rc == 0 &&
            aclrtMemcpy(host_output.data(), combine_output_bytes,
                        combine_output, combine_output_bytes,
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) rc = 6;
        if (rc == 0) {
            for (uint32_t row = 0u; row < tokens && rc == 0; ++row) {
                const uint16_t expected = FloatToFp16Bits(
                    static_cast<float>(rank * 16u + row % 13u + 1u) * topk);
                for (uint64_t column = 0u; column < row_elements; ++column) {
                    if (host_output[static_cast<uint64_t>(row) *
                                        row_elements + column] != expected) {
                        rc = 7;
                        break;
                    }
                }
            }
        }
        if (comm != nullptr) {
            if (dispatch_ws.query_token != 0u)
                (void)inc_dc_easy_workspace_release(
                    comm, dispatch_ws.query_token);
            if (combine_ws.query_token != 0u)
                (void)inc_dc_easy_workspace_release(
                    comm, combine_ws.query_token);
            (void)inc_dc_easy_comm_destroy(comm);
        }
        if (combine_scratch != nullptr) (void)aclrtFree(combine_scratch);
        if (dispatch_scratch != nullptr) (void)aclrtFree(dispatch_scratch);
        if (route != nullptr) (void)aclrtFree(route);
        if (expert_buffer != nullptr) (void)aclrtFree(expert_buffer);
        if (composite != nullptr &&
            DestroyNativeCompositeBackend(composite) != INC_DC_FW_OK &&
            rc == 0) rc = 8;
    }

    if (service_client != nullptr &&
        DestroyNativeIncServiceClient(service_client) != INC_DC_FW_OK &&
        rc == 0) rc = 8;
    if (inc_service != nullptr &&
        DestroyNativeIncService(inc_service) != INC_DC_FW_OK && rc == 0)
        rc = 8;

    std::cout << "NATIVE_FULL_RESULT pe=" << pe
              << " role=" << (static_cast<uint32_t>(pe) == world
                                   ? "inc" : "worker")
              << " pass=" << (rc == 0 ? 1 : 0) << " rc=" << rc
              << " generations=" << generations
              << " dispatch_worker_lanes="
              << dispatch_workspace.resources.dispatch_worker_aiv
              << " combine_worker_lanes="
              << combine_workspace.resources.combine_worker_aiv
              << " expert_alignment=" << expert_alignment << std::endl;
    if (DestroyNativeCombineSession(combine_native) != INC_DC_FW_OK && rc == 0)
        rc = 8;
    if (DestroyNativeDispatchSession(dispatch_native) != INC_DC_FW_OK &&
        rc == 0) rc = 8;
    Cleanup(device, stream, dispatch_heap, combine_heap);
    return rc;
}
