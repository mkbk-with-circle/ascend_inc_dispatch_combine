#include "inc_dc_easy_api.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

struct inc_dc_easy_comm {
    inc_dc_context_t *context = nullptr;
    inc_dc_plan_t *plan = nullptr;
    bool owns_context = false;
    uint32_t hidden_size = 0u;
    uint32_t max_topk = 0u;
    uint32_t dtype = INC_DC_FW_DTYPE_FP16;
};

static_assert(sizeof(inc_dc_easy_token_assignment_v1_t) == 16u,
              "token-plan assignment ABI changed");

namespace {

bool HeaderValid(uint32_t size, uint32_t expected, uint32_t version)
{
    return size >= expected && version == INC_DC_EASY_ABI_VERSION;
}

bool FitsRows(uint64_t rows)
{
    return rows <= static_cast<uint64_t>(
                       std::numeric_limits<int64_t>::max());
}

uint64_t Mix(uint64_t digest, uint64_t value)
{
    digest ^= value + 0x9e3779b97f4a7c15ull + (digest << 6u) +
              (digest >> 2u);
    return digest;
}

uint32_t FloatBits(float value)
{
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool TokenPlanDescriptionValid(
    const inc_dc_easy_token_plan_desc_t *description,
    uint64_t *assignment_count, uint64_t *bytes)
{
    if (description == nullptr || assignment_count == nullptr ||
        bytes == nullptr ||
        !HeaderValid(description->struct_size, sizeof(*description),
                     description->abi_version) ||
        description->tokens == 0u || description->topk == 0u ||
        description->worker_world_size == 0u ||
        description->worker_rank >= description->worker_world_size ||
        description->flags > std::numeric_limits<uint32_t>::max() ||
        description->expert_ids == nullptr ||
        (description->destination_ranks == nullptr &&
         description->experts_per_worker == 0u) ||
        description->tokens >
            std::numeric_limits<uint64_t>::max() / description->topk) {
        return false;
    }
    *assignment_count = description->tokens * description->topk;
    if (*assignment_count >
        (std::numeric_limits<uint64_t>::max() -
         sizeof(inc_dc_easy_token_plan_header_v1_t)) /
            sizeof(inc_dc_easy_token_assignment_v1_t)) {
        return false;
    }
    *bytes = sizeof(inc_dc_easy_token_plan_header_v1_t) +
             *assignment_count *
                 sizeof(inc_dc_easy_token_assignment_v1_t);
    return true;
}

inc_dc_fw_tensor_desc_t Tensor2d(
    void *data, uint64_t rows, uint32_t hidden, uint32_t dtype)
{
    inc_dc_fw_tensor_desc_t tensor{};
    tensor.struct_size = sizeof(tensor);
    tensor.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    tensor.data = data;
    tensor.dtype = dtype;
    tensor.memory_location = INC_DC_FW_MEMORY_DEVICE;
    tensor.ndim = 2u;
    tensor.dims[0] = static_cast<int64_t>(rows);
    tensor.dims[1] = static_cast<int64_t>(hidden);
    tensor.strides[0] = static_cast<int64_t>(hidden);
    tensor.strides[1] = 1;
    return tensor;
}

inc_dc_fw_tensor_desc_t Weights(
    void *data, uint64_t elements, uint32_t dtype)
{
    inc_dc_fw_tensor_desc_t tensor{};
    if (data == nullptr) return tensor;
    tensor.struct_size = sizeof(tensor);
    tensor.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    tensor.data = data;
    tensor.dtype = dtype;
    tensor.memory_location = INC_DC_FW_MEMORY_DEVICE;
    tensor.ndim = 1u;
    tensor.dims[0] = static_cast<int64_t>(elements);
    tensor.strides[0] = 1;
    return tensor;
}

bool WeightDTypeValid(uint32_t dtype)
{
    return dtype == INC_DC_FW_DTYPE_FP16 ||
           dtype == INC_DC_FW_DTYPE_BF16 ||
           dtype == INC_DC_FW_DTYPE_FP32;
}

inc_dc_fw_status_t MakeInvocation(
    const inc_dc_easy_comm_t *comm, const inc_dc_easy_op_t *op,
    inc_dc_fw_invocation_t *invocation)
{
    if (comm == nullptr || op == nullptr || invocation == nullptr ||
        !HeaderValid(op->struct_size, sizeof(*op), op->abi_version) ||
        op->topk == 0u || op->topk > comm->max_topk ||
        !FitsRows(op->tokens) || !FitsRows(op->input_rows) ||
        !FitsRows(op->output_rows) || !FitsRows(op->weight_elements)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    const uint64_t input_rows =
        op->input_rows == 0u ? op->tokens : op->input_rows;
    const uint64_t output_rows =
        op->output_rows == 0u ? op->tokens : op->output_rows;
    if (op->tokens != 0u &&
        (op->input == nullptr || op->output == nullptr ||
         input_rows == 0u || output_rows == 0u)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    if (op->weights != nullptr &&
        (op->weight_elements == 0u ||
         !WeightDTypeValid(op->weight_dtype))) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }

    *invocation = {};
    invocation->struct_size = sizeof(*invocation);
    invocation->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    invocation->shape.struct_size = sizeof(invocation->shape);
    invocation->shape.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    invocation->shape.tokens = op->tokens;
    invocation->shape.hidden_size = comm->hidden_size;
    invocation->shape.topk = op->topk;
    invocation->shape.dtype = comm->dtype;
    if (op->tokens != 0u) {
        invocation->input = Tensor2d(
            op->input, input_rows, comm->hidden_size, comm->dtype);
        invocation->output = Tensor2d(
            op->output, output_rows, comm->hidden_size, comm->dtype);
    }
    invocation->weights = Weights(
        op->weights, op->weight_elements, op->weight_dtype);
    invocation->route = op->route;
    invocation->workspace = op->workspace;
    invocation->workspace_bytes = op->workspace_bytes;
    invocation->workspace_query_token = op->workspace_query_token;
    invocation->stream = op->stream;
    invocation->operation_generation = op->operation_generation;
    invocation->deadline_ns = op->deadline_ns;
    invocation->flags = op->flags;
    if (op->metadata != nullptr) {
        invocation->flags |= INC_DC_FW_INVOCATION_HAS_EXTENSIONS;
        invocation->reserved[0] = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(&op->metadata->header));
    }
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Enqueue(
    inc_dc_easy_comm_t *comm, uint32_t operation,
    const inc_dc_easy_op_t *op, inc_dc_fw_request_t *request)
{
    if (comm == nullptr || request == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    inc_dc_fw_invocation_t invocation{};
    const inc_dc_fw_status_t status =
        MakeInvocation(comm, op, &invocation);
    if (status != INC_DC_FW_OK) return status;
    return operation == INC_DC_FW_OP_DISPATCH
               ? inc_dc_fw_dispatch_async(comm->plan, &invocation, request)
               : inc_dc_fw_combine_async(comm->plan, &invocation, request);
}

inc_dc_fw_status_t CreateCommPlan(
    const inc_dc_easy_comm_config_t *config, inc_dc_context_t *context,
    bool owns_context, inc_dc_easy_comm_t **comm)
{
    auto *created = new (std::nothrow) inc_dc_easy_comm_t();
    if (created == nullptr) return INC_DC_FW_INTERNAL;
    created->context = context;
    created->owns_context = owns_context;

    inc_dc_fw_plan_desc_t plan{};
    plan.struct_size = sizeof(plan);
    plan.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    plan.session_id = config->session_id;
    plan.model_id = config->model_id;
    plan.process_group_id = config->process_group_id;
    plan.topology_generation = config->topology_generation;
    plan.worker_world_size = config->worker_world_size;
    plan.worker_rank = config->worker_rank;
    plan.max_tokens = config->max_tokens_per_chunk;
    plan.hidden_size = config->hidden_size;
    plan.max_topk = config->max_topk;
    plan.dtype = config->dtype;
    plan.flags = config->flags;
    plan.static_route = config->static_route;
    const inc_dc_fw_status_t status =
        inc_dc_fw_plan_create(context, &plan, &created->plan);
    if (status != INC_DC_FW_OK) {
        delete created;
        return status;
    }
    created->hidden_size = config->hidden_size;
    created->max_topk = config->max_topk;
    created->dtype = config->dtype;
    *comm = created;
    return INC_DC_FW_OK;
}

} // namespace

extern "C" {

void inc_dc_easy_comm_config_init(inc_dc_easy_comm_config_t *config)
{
    if (config == nullptr) return;
    *config = {};
    config->struct_size = sizeof(*config);
    config->abi_version = INC_DC_EASY_ABI_VERSION;
    config->dtype = INC_DC_FW_DTYPE_FP16;
    config->max_inflight = 64u;
    config->max_workspace_queries = 64u;
    config->backend.struct_size = sizeof(config->backend);
    config->backend.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
}

void inc_dc_easy_op_init(inc_dc_easy_op_t *operation)
{
    if (operation == nullptr) return;
    *operation = {};
    operation->struct_size = sizeof(*operation);
    operation->abi_version = INC_DC_EASY_ABI_VERSION;
    operation->weight_dtype = INC_DC_FW_DTYPE_FP32;
}

void inc_dc_easy_route_device_init(
    inc_dc_fw_route_desc_t *route, uint32_t format, const void *data,
    uint64_t bytes, uint64_t semantic_digest, uint64_t generation)
{
    if (route == nullptr) return;
    *route = {};
    route->struct_size = sizeof(*route);
    route->abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    route->format = format;
    route->memory_location = INC_DC_FW_MEMORY_DEVICE;
    route->data = data;
    route->bytes = bytes;
    route->semantic_digest = semantic_digest;
    route->generation = generation;
}

void inc_dc_easy_token_plan_desc_init(
    inc_dc_easy_token_plan_desc_t *description)
{
    if (description == nullptr) return;
    *description = {};
    description->struct_size = sizeof(*description);
    description->abi_version = INC_DC_EASY_ABI_VERSION;
}

inc_dc_fw_status_t inc_dc_easy_token_plan_query(
    const inc_dc_easy_token_plan_desc_t *description, uint64_t *bytes)
{
    uint64_t assignments = 0u;
    uint64_t required = 0u;
    if (bytes == nullptr ||
        !TokenPlanDescriptionValid(description, &assignments, &required)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    *bytes = required;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_easy_token_plan_build(
    const inc_dc_easy_token_plan_desc_t *description, void *host_plan,
    uint64_t host_plan_capacity, inc_dc_easy_token_plan_info_t *info)
{
    uint64_t assignment_count = 0u;
    uint64_t required = 0u;
    if (host_plan == nullptr || info == nullptr ||
        !TokenPlanDescriptionValid(description, &assignment_count,
                                   &required) ||
        host_plan_capacity < required) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    auto *header =
        static_cast<inc_dc_easy_token_plan_header_v1_t *>(host_plan);
    auto *assignments = reinterpret_cast<inc_dc_easy_token_assignment_v1_t *>(
        static_cast<uint8_t *>(host_plan) +
        sizeof(inc_dc_easy_token_plan_header_v1_t));
    *header = {};
    header->magic = INC_DC_EASY_TOKEN_PLAN_MAGIC;
    header->abi_major = INC_DC_EASY_TOKEN_PLAN_ABI_VERSION;
    header->abi_minor = 0u;
    header->header_bytes = sizeof(*header);
    header->assignment_bytes = sizeof(*assignments);
    header->total_bytes = required;
    header->tokens = description->tokens;
    header->assignment_count = assignment_count;
    header->topk = description->topk;
    header->worker_world_size = description->worker_world_size;
    header->experts_per_worker = description->experts_per_worker;
    header->flags = static_cast<uint32_t>(description->flags);
    header->generation = description->generation;

    uint64_t global_physical_rows = 0u;
    uint64_t local_assignments = 0u;
    uint64_t local_physical_rows = 0u;
    uint64_t digest = 0x4541535954503101ull; /* EASYTP1\x01 */
    digest = Mix(digest, description->tokens);
    digest = Mix(digest, description->topk);
    digest = Mix(digest, description->worker_world_size);
    digest = Mix(digest, description->experts_per_worker);
    digest = Mix(digest, description->generation);
    digest = Mix(digest, description->flags);
    for (uint64_t token = 0u; token < description->tokens; ++token) {
        for (uint32_t slot = 0u; slot < description->topk; ++slot) {
            const uint64_t index = token * description->topk + slot;
            const int32_t expert = description->expert_ids[index];
            if (expert < 0) return INC_DC_FW_INVALID_ARGUMENT;
            uint32_t destination = 0u;
            if (description->destination_ranks != nullptr) {
                destination = description->destination_ranks[index];
            } else {
                const uint64_t expert64 = static_cast<uint32_t>(expert);
                const uint64_t expert_capacity =
                    static_cast<uint64_t>(description->worker_world_size) *
                    description->experts_per_worker;
                if (expert64 >= expert_capacity) {
                    return INC_DC_FW_INVALID_ARGUMENT;
                }
                destination = static_cast<uint32_t>(
                    expert64 / description->experts_per_worker);
            }
            if (destination >= description->worker_world_size) {
                return INC_DC_FW_INVALID_ARGUMENT;
            }
            const float weight = description->weights == nullptr
                                     ? 1.0f
                                     : description->weights[index];
            if (!std::isfinite(weight)) return INC_DC_FW_INVALID_ARGUMENT;
            for (uint32_t previous = 0u; previous < slot; ++previous) {
                const uint64_t previous_index =
                    token * description->topk + previous;
                if (description->expert_ids[previous_index] == expert) {
                    return INC_DC_FW_INVALID_ARGUMENT;
                }
            }
            bool destination_seen = false;
            for (uint32_t previous = 0u; previous < slot; ++previous) {
                if (assignments[token * description->topk + previous]
                        .destination_rank == destination) {
                    destination_seen = true;
                    break;
                }
            }
            if (!destination_seen) {
                ++global_physical_rows;
                if (destination == description->worker_rank) {
                    ++local_physical_rows;
                }
            }
            if (destination == description->worker_rank) {
                ++local_assignments;
            }
            assignments[index] = {expert, destination, FloatBits(weight), 0u};
            digest = Mix(digest, token);
            digest = Mix(digest, slot);
            digest = Mix(digest, static_cast<uint32_t>(expert));
            digest = Mix(digest, destination);
            digest = Mix(digest, assignments[index].weight_bits);
        }
    }
    if (digest == 0u) digest = 1u;
    header->global_physical_rows = global_physical_rows;
    header->semantic_digest = digest;
    *info = {};
    info->struct_size = sizeof(*info);
    info->abi_version = INC_DC_EASY_ABI_VERSION;
    info->bytes = required;
    info->semantic_digest = digest;
    info->generation = description->generation;
    info->logical_assignments = assignment_count;
    info->global_physical_rows = global_physical_rows;
    info->local_assignments = local_assignments;
    info->local_physical_rows = local_physical_rows;
    return INC_DC_FW_OK;
}

void inc_dc_easy_token_plan_device_route_init(
    inc_dc_fw_route_desc_t *route, const void *device_plan,
    const inc_dc_easy_token_plan_info_t *info)
{
    if (route == nullptr) return;
    if (info == nullptr || device_plan == nullptr ||
        !HeaderValid(info->struct_size, sizeof(*info), info->abi_version)) {
        *route = {};
        return;
    }
    inc_dc_easy_route_device_init(
        route, INC_DC_FW_ROUTE_TOPK_DENSE, device_plan, info->bytes,
        info->semantic_digest, info->generation);
}

inc_dc_fw_status_t inc_dc_easy_comm_create(
    const inc_dc_easy_comm_config_t *config, inc_dc_easy_comm_t **comm)
{
    if (config == nullptr || comm == nullptr ||
        !HeaderValid(config->struct_size, sizeof(*config),
                     config->abi_version)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    *comm = nullptr;
    inc_dc_fw_context_config_t context_config{};
    context_config.struct_size = sizeof(context_config);
    context_config.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    context_config.max_inflight = config->max_inflight;
    context_config.max_workspace_queries = config->max_workspace_queries;
    context_config.flags = config->flags;
    context_config.backend = config->backend;
    inc_dc_context_t *context = nullptr;
    inc_dc_fw_status_t status = inc_dc_fw_context_create(
        &context_config, &context);
    if (status != INC_DC_FW_OK) {
        return status;
    }
    status = CreateCommPlan(config, context, true, comm);
    if (status != INC_DC_FW_OK) {
        (void)inc_dc_fw_context_destroy(context);
        return status;
    }
    return status;
}

inc_dc_fw_status_t inc_dc_easy_comm_create_from_context(
    const inc_dc_easy_comm_config_t *config, inc_dc_context_t *context,
    inc_dc_easy_comm_t **comm)
{
    if (config == nullptr || context == nullptr || comm == nullptr ||
        !HeaderValid(config->struct_size, sizeof(*config),
                     config->abi_version)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    *comm = nullptr;
    return CreateCommPlan(config, context, false, comm);
}

inc_dc_fw_status_t inc_dc_easy_comm_get_capabilities(
    const inc_dc_easy_comm_t *comm,
    inc_dc_fw_capabilities_t *capabilities)
{
    return comm == nullptr
               ? INC_DC_FW_INVALID_ARGUMENT
               : inc_dc_fw_context_get_capabilities(
                     comm->context, capabilities);
}

inc_dc_fw_status_t inc_dc_easy_comm_get_stats(
    const inc_dc_easy_comm_t *comm, inc_dc_fw_context_stats_t *stats)
{
    return comm == nullptr
               ? INC_DC_FW_INVALID_ARGUMENT
               : inc_dc_fw_context_get_stats(comm->context, stats);
}

inc_dc_fw_status_t inc_dc_easy_workspace_query(
    inc_dc_easy_comm_t *comm, uint32_t operation, uint64_t tokens,
    uint32_t topk, inc_dc_fw_workspace_t *workspace)
{
    if (comm == nullptr || workspace == nullptr || topk == 0u ||
        topk > comm->max_topk) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    inc_dc_fw_shape_t shape{};
    shape.struct_size = sizeof(shape);
    shape.abi_version = INC_DC_FRAMEWORK_ABI_VERSION;
    shape.tokens = tokens;
    shape.hidden_size = comm->hidden_size;
    shape.topk = topk;
    shape.dtype = comm->dtype;
    workspace->struct_size = sizeof(*workspace);
    return inc_dc_fw_query_workspace(
        comm->plan, operation, &shape, workspace);
}

inc_dc_fw_status_t inc_dc_easy_workspace_release(
    inc_dc_easy_comm_t *comm, uint64_t workspace_query_token)
{
    return comm == nullptr
               ? INC_DC_FW_INVALID_ARGUMENT
               : inc_dc_fw_workspace_release(
                     comm->plan, workspace_query_token);
}

inc_dc_fw_status_t inc_dc_easy_dispatch_async(
    inc_dc_easy_comm_t *comm, const inc_dc_easy_op_t *operation,
    inc_dc_fw_request_t *request)
{
    return Enqueue(comm, INC_DC_FW_OP_DISPATCH, operation, request);
}

inc_dc_fw_status_t inc_dc_easy_combine_async(
    inc_dc_easy_comm_t *comm, const inc_dc_easy_op_t *operation,
    inc_dc_fw_request_t *request)
{
    return Enqueue(comm, INC_DC_FW_OP_COMBINE, operation, request);
}

inc_dc_fw_status_t inc_dc_easy_route_handle_create(
    inc_dc_easy_comm_t *comm, inc_dc_fw_request_t dispatch_request,
    const inc_dc_easy_op_t *dispatch_operation,
    inc_dc_fw_route_handle_t *route_handle)
{
    if (comm == nullptr || dispatch_operation == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    const inc_dc_fw_route_desc_t *route =
        dispatch_operation->route.data == nullptr
            ? nullptr
            : &dispatch_operation->route;
    return inc_dc_fw_route_handle_create(
        comm->plan, dispatch_request, route, route_handle);
}

inc_dc_fw_status_t inc_dc_easy_combine_with_route_async(
    inc_dc_easy_comm_t *comm, inc_dc_fw_route_handle_t route_handle,
    const inc_dc_easy_op_t *operation, inc_dc_fw_request_t *request)
{
    if (comm == nullptr || request == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    inc_dc_fw_invocation_t invocation{};
    const inc_dc_fw_status_t status =
        MakeInvocation(comm, operation, &invocation);
    return status == INC_DC_FW_OK
               ? inc_dc_fw_combine_with_route_async(
                     comm->plan, route_handle, &invocation, request)
               : status;
}

inc_dc_fw_status_t inc_dc_easy_route_handle_release(
    inc_dc_easy_comm_t *comm, inc_dc_fw_route_handle_t route_handle)
{
    return comm == nullptr
               ? INC_DC_FW_INVALID_ARGUMENT
               : inc_dc_fw_route_handle_release(
                     comm->plan, route_handle);
}

inc_dc_fw_status_t inc_dc_easy_request_query(
    inc_dc_easy_comm_t *comm, inc_dc_fw_request_t request,
    uint32_t *state)
{
    return comm == nullptr
               ? INC_DC_FW_INVALID_ARGUMENT
               : inc_dc_fw_request_query(comm->context, request, state);
}

inc_dc_fw_status_t inc_dc_easy_request_wait(
    inc_dc_easy_comm_t *comm, inc_dc_fw_request_t request,
    uint64_t timeout_ns)
{
    return comm == nullptr
               ? INC_DC_FW_INVALID_ARGUMENT
               : inc_dc_fw_request_wait(
                     comm->context, request, timeout_ns);
}

inc_dc_fw_status_t inc_dc_easy_request_cancel(
    inc_dc_easy_comm_t *comm, inc_dc_fw_request_t request)
{
    return comm == nullptr
               ? INC_DC_FW_INVALID_ARGUMENT
               : inc_dc_fw_request_cancel(comm->context, request);
}

inc_dc_fw_status_t inc_dc_easy_request_release(
    inc_dc_easy_comm_t *comm, inc_dc_fw_request_t request)
{
    return comm == nullptr
               ? INC_DC_FW_INVALID_ARGUMENT
               : inc_dc_fw_request_release(comm->context, request);
}

inc_dc_fw_status_t inc_dc_easy_request_wait_and_release(
    inc_dc_easy_comm_t *comm, inc_dc_fw_request_t request,
    uint64_t timeout_ns)
{
    const inc_dc_fw_status_t status =
        inc_dc_easy_request_wait(comm, request, timeout_ns);
    if (status != INC_DC_FW_OK) return status;
    return inc_dc_easy_request_release(comm, request);
}

inc_dc_fw_status_t inc_dc_easy_comm_destroy(inc_dc_easy_comm_t *comm)
{
    if (comm == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    inc_dc_fw_status_t status = inc_dc_fw_plan_release(comm->plan);
    if (status != INC_DC_FW_OK) return status;
    comm->plan = nullptr;
    if (comm->owns_context) {
        status = inc_dc_fw_context_destroy(comm->context);
        if (status != INC_DC_FW_OK) return status;
    }
    comm->context = nullptr;
    delete comm;
    return INC_DC_FW_OK;
}

} // extern "C"
