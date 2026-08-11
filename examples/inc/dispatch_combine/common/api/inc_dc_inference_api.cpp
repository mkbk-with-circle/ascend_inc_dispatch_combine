#include "inc_dc_inference_api.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <new>
#include <unordered_set>

struct LeaseKey {
    uint64_t tokens;
    uint32_t topk;
    uint32_t operation;
    bool operator<(const LeaseKey &other) const
    {
        if (tokens != other.tokens) return tokens < other.tokens;
        if (topk != other.topk) return topk < other.topk;
        return operation < other.operation;
    }
};

struct SharedLease {
    inc_dc_fw_workspace_t workspace{};
    uint64_t references = 0u;
};

struct inc_dc_infer_session {
    inc_dc_easy_comm_t *comm = nullptr;
    inc_dc_infer_allocate_fn allocate = nullptr;
    inc_dc_infer_free_fn release = nullptr;
    void *allocator_context = nullptr;
    uint64_t generation = 0u;
    std::mutex mutex;
    std::unordered_set<inc_dc_infer_plan_t *> plans;
    std::map<LeaseKey, SharedLease> leases;
    uint64_t active_creations = 0u;
    uint64_t live_routes = 0u;
    bool closing = false;
};

struct OperationSlot {
    inc_dc_fw_workspace_t workspace{};
    void *storage = nullptr;
    bool active = false;
    inc_dc_fw_request_t request{};
    bool workspace_referenced = false;
};

struct inc_dc_infer_plan {
    inc_dc_infer_session_t *session = nullptr;
    uint64_t tokens = 0u;
    uint32_t topk = 0u;
    uint64_t generation = 0u;
    OperationSlot dispatch;
    OperationSlot combine;
    std::mutex mutex;
};

namespace {

std::atomic<uint64_t> g_plan_generation{1u};
std::atomic<uint64_t> g_session_generation{1u};

bool HeaderValid(uint32_t size, uint32_t expected, uint32_t version)
{
    return size >= expected && version == INC_DC_INFERENCE_ABI_VERSION;
}

bool SameRequest(inc_dc_fw_request_t left, inc_dc_fw_request_t right)
{
    return left.slot == right.slot && left.generation == right.generation;
}

OperationSlot *Slot(inc_dc_infer_plan_t *plan, uint32_t operation)
{
    if (operation == INC_DC_FW_OP_DISPATCH) return &plan->dispatch;
    if (operation == INC_DC_FW_OP_COMBINE) return &plan->combine;
    return nullptr;
}

inc_dc_fw_status_t AcquireWorkspace(
    inc_dc_infer_session_t *session, uint64_t tokens, uint32_t topk,
    uint32_t operation, OperationSlot *slot)
{
    const LeaseKey key{tokens, topk, operation};
    std::lock_guard<std::mutex> lock(session->mutex);
    auto found = session->leases.find(key);
    if (found == session->leases.end()) {
        inc_dc_fw_workspace_t workspace{};
        const inc_dc_fw_status_t status = inc_dc_easy_workspace_query(
            session->comm, operation, tokens, topk, &workspace);
        if (status != INC_DC_FW_OK) return status;
        found = session->leases.emplace(
            key, SharedLease{workspace, 0u}).first;
    }
    ++found->second.references;
    slot->workspace = found->second.workspace;
    slot->workspace_referenced = true;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t ReleaseWorkspace(
    inc_dc_infer_session_t *session, uint64_t tokens, uint32_t topk,
    uint32_t operation, OperationSlot *slot)
{
    if (!slot->workspace_referenced) return INC_DC_FW_OK;
    const LeaseKey key{tokens, topk, operation};
    std::lock_guard<std::mutex> lock(session->mutex);
    auto found = session->leases.find(key);
    if (found == session->leases.end() || found->second.references == 0u ||
        found->second.workspace.query_token != slot->workspace.query_token) {
        return INC_DC_FW_INTERNAL;
    }
    if (found->second.references == 1u) {
        const inc_dc_fw_status_t status = inc_dc_easy_workspace_release(
            session->comm, found->second.workspace.query_token);
        if (status != INC_DC_FW_OK) return status;
        session->leases.erase(found);
    } else {
        --found->second.references;
    }
    slot->workspace_referenced = false;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t ValidateRequestLocked(
    inc_dc_infer_plan_t *plan, const inc_dc_infer_request_t *request,
    OperationSlot **slot)
{
    if (plan == nullptr || request == nullptr || slot == nullptr ||
        !HeaderValid(request->struct_size, sizeof(*request),
                     request->abi_version) ||
        request->plan_generation != plan->generation) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    *slot = Slot(plan, request->operation);
    if (*slot == nullptr || !(*slot)->active ||
        !SameRequest((*slot)->request, request->framework_request)) {
        return INC_DC_FW_STALE_HANDLE;
    }
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t MakeEasyOp(
    const inc_dc_infer_plan_t *plan, const inc_dc_infer_io_t *io,
    uint32_t operation, inc_dc_easy_op_t *op)
{
    if (plan == nullptr || io == nullptr || op == nullptr ||
        !HeaderValid(io->struct_size, sizeof(*io), io->abi_version)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    const OperationSlot *slot = operation == INC_DC_FW_OP_DISPATCH
                                    ? &plan->dispatch
                                    : &plan->combine;
    inc_dc_easy_op_init(op);
    op->tokens = plan->tokens;
    op->input_rows = io->input_rows;
    op->output_rows = io->output_rows;
    op->topk = plan->topk;
    op->weight_dtype = io->weight_dtype;
    op->input = io->input;
    op->output = io->output;
    op->weights = io->weights;
    op->weight_elements = io->weight_elements;
    op->route = io->route;
    op->workspace = slot->storage;
    op->workspace_bytes = slot->workspace.bytes;
    op->workspace_query_token = slot->workspace.query_token;
    op->stream = io->stream;
    op->operation_generation = io->operation_generation;
    op->deadline_ns = io->deadline_ns;
    op->metadata = io->metadata;
    op->flags = io->flags;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t Enqueue(
    inc_dc_infer_plan_t *plan, const inc_dc_infer_io_t *io,
    uint32_t operation, const inc_dc_infer_route_t *route,
    inc_dc_infer_request_t *request)
{
    if (plan == nullptr || request == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    if (route != nullptr &&
        !HeaderValid(route->struct_size, sizeof(*route), route->abi_version)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    inc_dc_easy_op_t op{};
    inc_dc_fw_status_t status = MakeEasyOp(plan, io, operation, &op);
    if (status != INC_DC_FW_OK) return status;

    OperationSlot *slot = Slot(plan, operation);
    {
        std::lock_guard<std::mutex> lock(plan->mutex);
        if (slot->active) return INC_DC_FW_BUSY;
        slot->active = true; /* reserve before entering a possibly concurrent backend */
    }
    inc_dc_fw_request_t framework_request{};
    if (operation == INC_DC_FW_OP_DISPATCH) {
        status = inc_dc_easy_dispatch_async(
            plan->session->comm, &op, &framework_request);
    } else if (route == nullptr) {
        status = inc_dc_easy_combine_async(
            plan->session->comm, &op, &framework_request);
    } else {
        status = inc_dc_easy_combine_with_route_async(
            plan->session->comm, route->framework_route, &op,
            &framework_request);
    }
    std::lock_guard<std::mutex> lock(plan->mutex);
    if (status != INC_DC_FW_OK) {
        slot->active = false;
        return status;
    }
    slot->request = framework_request;
    *request = {};
    request->struct_size = sizeof(*request);
    request->abi_version = INC_DC_INFERENCE_ABI_VERSION;
    request->operation = operation;
    request->framework_request = framework_request;
    request->plan_generation = plan->generation;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t CreateSession(
    const inc_dc_infer_config_t *config, inc_dc_context_t *context,
    inc_dc_infer_session_t **session)
{
    if (config == nullptr || session == nullptr ||
        !HeaderValid(config->struct_size, sizeof(*config),
                     config->abi_version) ||
        config->allocate_device == nullptr || config->free_device == nullptr) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    *session = nullptr;
    auto *created = new (std::nothrow) inc_dc_infer_session_t();
    if (created == nullptr) return INC_DC_FW_INTERNAL;
    inc_dc_fw_status_t status = context == nullptr
        ? inc_dc_easy_comm_create(&config->communicator, &created->comm)
        : inc_dc_easy_comm_create_from_context(
              &config->communicator, context, &created->comm);
    if (status != INC_DC_FW_OK) {
        delete created;
        return status;
    }
    created->allocate = config->allocate_device;
    created->release = config->free_device;
    created->allocator_context = config->allocator_context;
    created->generation = g_session_generation.fetch_add(1u);
    if (created->generation == 0u)
        created->generation = g_session_generation.fetch_add(1u);
    *session = created;
    return INC_DC_FW_OK;
}

} // namespace

extern "C" {

void inc_dc_infer_config_init(inc_dc_infer_config_t *config)
{
    if (config == nullptr) return;
    *config = {};
    config->struct_size = sizeof(*config);
    config->abi_version = INC_DC_INFERENCE_ABI_VERSION;
    inc_dc_easy_comm_config_init(&config->communicator);
}

void inc_dc_infer_plan_desc_init(inc_dc_infer_plan_desc_t *description)
{
    if (description == nullptr) return;
    *description = {};
    description->struct_size = sizeof(*description);
    description->abi_version = INC_DC_INFERENCE_ABI_VERSION;
}

void inc_dc_infer_io_init(inc_dc_infer_io_t *io)
{
    if (io == nullptr) return;
    *io = {};
    io->struct_size = sizeof(*io);
    io->abi_version = INC_DC_INFERENCE_ABI_VERSION;
    io->weight_dtype = INC_DC_FW_DTYPE_FP32;
}

inc_dc_fw_status_t inc_dc_infer_session_create(
    const inc_dc_infer_config_t *config, inc_dc_infer_session_t **session)
{
    return CreateSession(config, nullptr, session);
}

inc_dc_fw_status_t inc_dc_infer_session_create_from_context(
    const inc_dc_infer_config_t *config, inc_dc_context_t *context,
    inc_dc_infer_session_t **session)
{
    if (context == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    return CreateSession(config, context, session);
}

inc_dc_fw_status_t inc_dc_infer_session_get_capabilities(
    const inc_dc_infer_session_t *session,
    inc_dc_fw_capabilities_t *capabilities)
{
    return session == nullptr ? INC_DC_FW_INVALID_ARGUMENT
                              : inc_dc_easy_comm_get_capabilities(
                                    session->comm, capabilities);
}

inc_dc_fw_status_t inc_dc_infer_session_get_stats(
    const inc_dc_infer_session_t *session, inc_dc_fw_context_stats_t *stats)
{
    return session == nullptr ? INC_DC_FW_INVALID_ARGUMENT
                              : inc_dc_easy_comm_get_stats(session->comm, stats);
}

inc_dc_fw_status_t inc_dc_infer_plan_create(
    inc_dc_infer_session_t *session,
    const inc_dc_infer_plan_desc_t *description,
    inc_dc_infer_plan_t **plan)
{
    if (session == nullptr || description == nullptr || plan == nullptr ||
        !HeaderValid(description->struct_size, sizeof(*description),
                     description->abi_version) || description->topk == 0u) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    *plan = nullptr;
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        if (session->closing) return INC_DC_FW_BUSY;
        ++session->active_creations;
    }
    auto *created = new (std::nothrow) inc_dc_infer_plan_t();
    if (created == nullptr) {
        std::lock_guard<std::mutex> lock(session->mutex);
        --session->active_creations;
        return INC_DC_FW_INTERNAL;
    }
    created->session = session;
    created->tokens = description->tokens;
    created->topk = description->topk;
    created->generation = g_plan_generation.fetch_add(1u);
    if (created->generation == 0u) {
        created->generation = g_plan_generation.fetch_add(1u);
    }

    inc_dc_fw_status_t status = AcquireWorkspace(
        session, created->tokens, created->topk, INC_DC_FW_OP_DISPATCH,
        &created->dispatch);
    if (status == INC_DC_FW_OK) {
        status = AcquireWorkspace(
            session, created->tokens, created->topk, INC_DC_FW_OP_COMBINE,
            &created->combine);
    }
    if (status == INC_DC_FW_OK && created->dispatch.workspace.bytes != 0u) {
        created->dispatch.storage = session->allocate(
            created->dispatch.workspace.bytes,
            created->dispatch.workspace.alignment,
            session->allocator_context);
        if (created->dispatch.storage == nullptr) status = INC_DC_FW_INTERNAL;
    }
    if (status == INC_DC_FW_OK && created->combine.workspace.bytes != 0u) {
        created->combine.storage = session->allocate(
            created->combine.workspace.bytes,
            created->combine.workspace.alignment,
            session->allocator_context);
        if (created->combine.storage == nullptr) status = INC_DC_FW_INTERNAL;
    }
    if (status != INC_DC_FW_OK) {
        if (created->combine.storage != nullptr)
            session->release(created->combine.storage,
                             session->allocator_context);
        if (created->dispatch.storage != nullptr)
            session->release(created->dispatch.storage,
                             session->allocator_context);
        (void)ReleaseWorkspace(session, created->tokens, created->topk,
                               INC_DC_FW_OP_COMBINE, &created->combine);
        (void)ReleaseWorkspace(session, created->tokens, created->topk,
                               INC_DC_FW_OP_DISPATCH, &created->dispatch);
        {
            std::lock_guard<std::mutex> lock(session->mutex);
            --session->active_creations;
        }
        delete created;
        return status;
    }
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        session->plans.insert(created);
        --session->active_creations;
    }
    *plan = created;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_infer_plan_get_info(
    const inc_dc_infer_plan_t *plan, inc_dc_infer_plan_info_t *info)
{
    if (plan == nullptr || info == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    *info = {};
    info->struct_size = sizeof(*info);
    info->abi_version = INC_DC_INFERENCE_ABI_VERSION;
    info->tokens = plan->tokens;
    info->topk = plan->topk;
    info->dispatch_workspace_bytes = plan->dispatch.workspace.bytes;
    info->combine_workspace_bytes = plan->combine.workspace.bytes;
    info->dispatch_workspace_alignment = plan->dispatch.workspace.alignment;
    info->combine_workspace_alignment = plan->combine.workspace.alignment;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_infer_dispatch_async(
    inc_dc_infer_plan_t *plan, const inc_dc_infer_io_t *io,
    inc_dc_infer_request_t *request)
{
    return Enqueue(plan, io, INC_DC_FW_OP_DISPATCH, nullptr, request);
}

inc_dc_fw_status_t inc_dc_infer_combine_async(
    inc_dc_infer_plan_t *plan, const inc_dc_infer_io_t *io,
    inc_dc_infer_request_t *request)
{
    return Enqueue(plan, io, INC_DC_FW_OP_COMBINE, nullptr, request);
}

inc_dc_fw_status_t inc_dc_infer_route_capture(
    inc_dc_infer_plan_t *plan,
    const inc_dc_infer_request_t *dispatch_request,
    const inc_dc_infer_io_t *dispatch_io, inc_dc_infer_route_t *route)
{
    if (plan == nullptr || route == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    {
        std::lock_guard<std::mutex> lock(plan->mutex);
        OperationSlot *slot = nullptr;
        const inc_dc_fw_status_t status = ValidateRequestLocked(
            plan, dispatch_request, &slot);
        if (status != INC_DC_FW_OK ||
            dispatch_request->operation != INC_DC_FW_OP_DISPATCH) {
            return status == INC_DC_FW_OK ? INC_DC_FW_INVALID_ARGUMENT : status;
        }
    }
    inc_dc_easy_op_t op{};
    inc_dc_fw_status_t status = MakeEasyOp(
        plan, dispatch_io, INC_DC_FW_OP_DISPATCH, &op);
    if (status != INC_DC_FW_OK) return status;
    inc_dc_fw_route_handle_t handle{};
    status = inc_dc_easy_route_handle_create(
        plan->session->comm, dispatch_request->framework_request, &op, &handle);
    if (status != INC_DC_FW_OK) return status;
    *route = {};
    route->struct_size = sizeof(*route);
    route->abi_version = INC_DC_INFERENCE_ABI_VERSION;
    route->framework_route = handle;
    route->session_generation = plan->session->generation;
    {
        std::lock_guard<std::mutex> lock(plan->session->mutex);
        ++plan->session->live_routes;
    }
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_infer_combine_with_route_async(
    inc_dc_infer_plan_t *plan, const inc_dc_infer_route_t *route,
    const inc_dc_infer_io_t *io, inc_dc_infer_request_t *request)
{
    if (route == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    if (plan == nullptr || route->session_generation != plan->session->generation)
        return INC_DC_FW_STALE_HANDLE;
    return Enqueue(plan, io, INC_DC_FW_OP_COMBINE, route, request);
}

inc_dc_fw_status_t inc_dc_infer_route_release(
    inc_dc_infer_session_t *session, inc_dc_infer_route_t *route)
{
    if (session == nullptr || route == nullptr ||
        !HeaderValid(route->struct_size, sizeof(*route), route->abi_version)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    if (route->session_generation != session->generation)
        return INC_DC_FW_STALE_HANDLE;
    const inc_dc_fw_status_t status = inc_dc_easy_route_handle_release(
        session->comm, route->framework_route);
    if (status == INC_DC_FW_OK) {
        std::lock_guard<std::mutex> lock(session->mutex);
        if (session->live_routes == 0u) return INC_DC_FW_INTERNAL;
        --session->live_routes;
        *route = {};
    }
    return status;
}

inc_dc_fw_status_t inc_dc_infer_request_query(
    inc_dc_infer_plan_t *plan, const inc_dc_infer_request_t *request,
    uint32_t *state)
{
    if (plan == nullptr || state == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(plan->mutex);
    OperationSlot *slot = nullptr;
    const inc_dc_fw_status_t status = ValidateRequestLocked(plan, request, &slot);
    return status == INC_DC_FW_OK
               ? inc_dc_easy_request_query(
                     plan->session->comm, request->framework_request, state)
               : status;
}

inc_dc_fw_status_t inc_dc_infer_request_wait(
    inc_dc_infer_plan_t *plan, const inc_dc_infer_request_t *request,
    uint64_t timeout_ns)
{
    if (plan == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    {
        std::lock_guard<std::mutex> lock(plan->mutex);
        OperationSlot *slot = nullptr;
        const inc_dc_fw_status_t status = ValidateRequestLocked(plan, request, &slot);
        if (status != INC_DC_FW_OK) return status;
    }
    return inc_dc_easy_request_wait(
        plan->session->comm, request->framework_request, timeout_ns);
}

inc_dc_fw_status_t inc_dc_infer_request_cancel(
    inc_dc_infer_plan_t *plan, const inc_dc_infer_request_t *request)
{
    if (plan == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    {
        std::lock_guard<std::mutex> lock(plan->mutex);
        OperationSlot *slot = nullptr;
        const inc_dc_fw_status_t status = ValidateRequestLocked(plan, request, &slot);
        if (status != INC_DC_FW_OK) return status;
    }
    return inc_dc_easy_request_cancel(
        plan->session->comm, request->framework_request);
}

inc_dc_fw_status_t inc_dc_infer_request_release(
    inc_dc_infer_plan_t *plan, inc_dc_infer_request_t *request)
{
    if (plan == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    OperationSlot *slot = nullptr;
    {
        std::lock_guard<std::mutex> lock(plan->mutex);
        const inc_dc_fw_status_t status = ValidateRequestLocked(plan, request, &slot);
        if (status != INC_DC_FW_OK) return status;
    }
    const inc_dc_fw_status_t status = inc_dc_easy_request_release(
        plan->session->comm, request->framework_request);
    if (status != INC_DC_FW_OK) return status;
    std::lock_guard<std::mutex> lock(plan->mutex);
    if (slot->active && SameRequest(slot->request, request->framework_request)) {
        slot->active = false;
        slot->request = {};
    }
    *request = {};
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_infer_request_wait_and_release(
    inc_dc_infer_plan_t *plan, inc_dc_infer_request_t *request,
    uint64_t timeout_ns)
{
    const inc_dc_fw_status_t status = inc_dc_infer_request_wait(
        plan, request, timeout_ns);
    return status == INC_DC_FW_OK
               ? inc_dc_infer_request_release(plan, request)
               : status;
}

inc_dc_fw_status_t inc_dc_infer_plan_destroy(inc_dc_infer_plan_t *plan)
{
    if (plan == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    {
        std::lock_guard<std::mutex> lock(plan->mutex);
        if (plan->dispatch.active || plan->combine.active) return INC_DC_FW_BUSY;
    }
    inc_dc_infer_session_t *session = plan->session;
    inc_dc_fw_status_t status = ReleaseWorkspace(
        session, plan->tokens, plan->topk, INC_DC_FW_OP_DISPATCH,
        &plan->dispatch);
    if (status != INC_DC_FW_OK) return status;
    status = ReleaseWorkspace(
        session, plan->tokens, plan->topk, INC_DC_FW_OP_COMBINE,
        &plan->combine);
    if (status != INC_DC_FW_OK) return status;
    if (plan->dispatch.storage != nullptr)
        session->release(plan->dispatch.storage, session->allocator_context);
    if (plan->combine.storage != nullptr)
        session->release(plan->combine.storage, session->allocator_context);
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        session->plans.erase(plan);
    }
    delete plan;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t inc_dc_infer_session_destroy(
    inc_dc_infer_session_t *session)
{
    if (session == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        if (!session->plans.empty() || session->active_creations != 0u ||
            session->live_routes != 0u ||
            !session->leases.empty() || session->closing) {
            return INC_DC_FW_BUSY;
        }
        session->closing = true;
    }
    const inc_dc_fw_status_t status = inc_dc_easy_comm_destroy(session->comm);
    if (status != INC_DC_FW_OK) {
        std::lock_guard<std::mutex> lock(session->mutex);
        session->closing = false;
        return status;
    }
    delete session;
    return INC_DC_FW_OK;
}

} // extern "C"
