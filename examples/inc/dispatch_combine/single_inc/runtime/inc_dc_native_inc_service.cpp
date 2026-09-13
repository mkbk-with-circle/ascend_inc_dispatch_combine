#include "inc_dc_native_inc_service.h"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <limits>
#include <new>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include "inc_dc_native_combine_backend.h"
#include "inc_dc_native_dispatch_backend.h"

namespace inc::dc::single_stream {

namespace {

constexpr uint32_t kServiceMagic = 0x494e4353u; // INCS
constexpr uint32_t kServiceVersion = 1u;

struct ServiceMessage {
    uint32_t magic = kServiceMagic;
    uint32_t version = kServiceVersion;
    uint32_t operation = 0u;
    uint32_t worker_rank = 0u;
    uint64_t generation = 0u;
};

struct ServiceResponse {
    uint32_t magic = kServiceMagic;
    uint32_t version = kServiceVersion;
    uint32_t operation = 0u;
    uint32_t status = INC_DC_FW_INTERNAL;
    uint64_t generation = 0u;
};

bool WriteAll(int fd, const void *data, size_t bytes)
{
    const auto *cursor = static_cast<const uint8_t *>(data);
    while (bytes != 0u) {
        const ssize_t count = send(fd, cursor, bytes, MSG_NOSIGNAL);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        cursor += count;
        bytes -= static_cast<size_t>(count);
    }
    return true;
}

bool ReadAll(int fd, void *data, size_t bytes)
{
    auto *cursor = static_cast<uint8_t *>(data);
    while (bytes != 0u) {
        const ssize_t count = recv(fd, cursor, bytes, 0);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        cursor += count;
        bytes -= static_cast<size_t>(count);
    }
    return true;
}

bool SetTimeout(int fd, uint64_t timeout_ns)
{
    const uint64_t usec = timeout_ns == 0u
        ? 30000000ull : (timeout_ns + 999u) / 1000u;
    timeval timeout{};
    timeout.tv_sec = static_cast<time_t>(usec / 1000000u);
    timeout.tv_usec = static_cast<suseconds_t>(usec % 1000000u);
    return setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                      sizeof(timeout)) == 0 &&
           setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout,
                      sizeof(timeout)) == 0;
}

void CloseFd(int *fd)
{
    if (fd != nullptr && *fd >= 0) {
        (void)shutdown(*fd, SHUT_RDWR);
        (void)close(*fd);
        *fd = -1;
    }
}

inc_dc_fw_status_t NativeWorkerBeforeEnqueue(
    void *opaque, uint32_t operation, uint64_t generation)
{
    auto *control = static_cast<NativeSingleIncWorkerControl *>(opaque);
    if (control == nullptr || control->client == nullptr || generation == 0u)
        return INC_DC_FW_INVALID_ARGUMENT;
    if (operation == INC_DC_FW_OP_COMBINE) return INC_DC_FW_OK;
    if (operation != INC_DC_FW_OP_DISPATCH || control->dispatch == nullptr)
        return INC_DC_FW_INVALID_ARGUMENT;
    inc_dc_fw_status_t status = NativeDispatchPrepareGeneration(
        control->dispatch, generation);
    return status == INC_DC_FW_OK
        ? NativeIncServiceSubmitAndWait(
              control->client, NativeIncServiceOp::DISPATCH_PREPARE,
              generation)
        : status;
}

inc_dc_fw_status_t NativeWorkerAfterEnqueue(
    void *opaque, uint32_t operation, uint64_t generation)
{
    auto *control = static_cast<NativeSingleIncWorkerControl *>(opaque);
    if (control == nullptr || control->client == nullptr || generation == 0u)
        return INC_DC_FW_INVALID_ARGUMENT;
    const NativeIncServiceOp service_operation =
        operation == INC_DC_FW_OP_DISPATCH ? NativeIncServiceOp::DISPATCH
        : operation == INC_DC_FW_OP_COMBINE ? NativeIncServiceOp::COMBINE
                                            : NativeIncServiceOp{};
    if (operation != INC_DC_FW_OP_DISPATCH &&
        operation != INC_DC_FW_OP_COMBINE)
        return INC_DC_FW_INVALID_ARGUMENT;
    return NativeIncServiceSubmitAndWait(
        control->client, service_operation, generation);
}

} // namespace

struct NativeIncService {
    NativeIncServiceConfig config{};
    int listen_fd = -1;
    std::vector<int> workers;
    bool ran = false;
};

struct NativeIncServiceClient {
    int fd = -1;
    uint32_t rank = 0u;
};

inc_dc_fw_status_t CreateNativeIncService(
    const NativeIncServiceConfig &config, NativeIncService **service)
{
    if (service == nullptr || *service != nullptr ||
        config.worker_world_size < 2u || config.tcp_port == 0u ||
        config.stream == 0u || config.dispatch == nullptr ||
        config.combine == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    auto *created = new (std::nothrow) NativeIncService();
    if (created == nullptr) return INC_DC_FW_INTERNAL;
    created->config = config;
    created->workers.assign(config.worker_world_size, -1);
    created->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    int reuse = 1;
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(config.tcp_port);
    if (created->listen_fd < 0 ||
        setsockopt(created->listen_fd, SOL_SOCKET, SO_REUSEADDR,
                   &reuse, sizeof(reuse)) != 0 ||
        !SetTimeout(created->listen_fd, config.timeout_ns) ||
        bind(created->listen_fd, reinterpret_cast<sockaddr *>(&address),
             sizeof(address)) != 0 ||
        listen(created->listen_fd,
               static_cast<int>(config.worker_world_size)) != 0) {
        CloseFd(&created->listen_fd);
        delete created;
        return INC_DC_FW_BACKEND_ERROR;
    }
    *service = created;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t NativeIncServiceRun(
    NativeIncService *service, uint32_t generations)
{
    if (service == nullptr || generations == 0u || service->ran)
        return INC_DC_FW_INVALID_ARGUMENT;
    service->ran = true;
    for (uint32_t connected = 0u;
         connected < service->config.worker_world_size; ++connected) {
        int fd = accept(service->listen_fd, nullptr, nullptr);
        ServiceMessage hello{};
        if (fd < 0 || !SetTimeout(fd, service->config.timeout_ns) ||
            !ReadAll(fd, &hello, sizeof(hello)) ||
            hello.magic != kServiceMagic ||
            hello.version != kServiceVersion || hello.operation != 0u ||
            hello.generation != 0u ||
            hello.worker_rank >= service->config.worker_world_size ||
            service->workers[hello.worker_rank] >= 0) {
            CloseFd(&fd);
            return INC_DC_FW_BACKEND_ERROR;
        }
        service->workers[hello.worker_rank] = fd;
    }

    for (uint64_t generation = 1u; generation <= generations; ++generation) {
        for (NativeIncServiceOp operation : {
                 NativeIncServiceOp::DISPATCH_PREPARE,
                 NativeIncServiceOp::DISPATCH,
                 NativeIncServiceOp::COMBINE}) {
            bool valid = true;
            for (uint32_t rank = 0u;
                 rank < service->config.worker_world_size; ++rank) {
                ServiceMessage message{};
                valid = valid && ReadAll(
                    service->workers[rank], &message, sizeof(message));
                valid = valid && message.magic == kServiceMagic &&
                    message.version == kServiceVersion &&
                    message.operation == static_cast<uint32_t>(operation) &&
                    message.worker_rank == rank &&
                    message.generation == generation;
            }
            inc_dc_fw_status_t status = valid
                ? INC_DC_FW_OK : INC_DC_FW_BACKEND_ERROR;
            inc_dc_fw_backend_ticket_t ticket{};
            if (status == INC_DC_FW_OK &&
                operation == NativeIncServiceOp::DISPATCH_PREPARE) {
                status = NativeDispatchPrepareGeneration(
                    service->config.dispatch, generation);
            } else if (status == INC_DC_FW_OK &&
                       operation == NativeIncServiceOp::DISPATCH) {
                if (status == INC_DC_FW_OK)
                    status = NativeDispatchIncEnqueue(
                        service->config.dispatch, service->config.stream,
                        generation, &ticket);
                if (status == INC_DC_FW_OK)
                    status = NativeDispatchIncWaitAndRelease(
                        service->config.dispatch, ticket,
                        service->config.timeout_ns);
            } else if (status == INC_DC_FW_OK) {
                status = NativeCombineIncEnqueue(
                    service->config.combine, service->config.stream,
                    generation, &ticket);
                if (status == INC_DC_FW_OK)
                    status = NativeCombineIncWaitAndRelease(
                        service->config.combine, ticket,
                        service->config.timeout_ns);
            }
            ServiceResponse response{};
            response.operation = static_cast<uint32_t>(operation);
            response.status = status;
            response.generation = generation;
            for (int fd : service->workers)
                valid = WriteAll(fd, &response, sizeof(response)) && valid;
            if (!valid || status != INC_DC_FW_OK) return status;
        }
    }
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t DestroyNativeIncService(NativeIncService *service)
{
    if (service == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    for (int &fd : service->workers) CloseFd(&fd);
    CloseFd(&service->listen_fd);
    delete service;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t CreateNativeIncServiceClient(
    const char *host, uint16_t port, uint32_t worker_rank,
    uint64_t timeout_ns, NativeIncServiceClient **client)
{
    if (host == nullptr || client == nullptr || *client != nullptr ||
        port == 0u) return INC_DC_FW_INVALID_ARGUMENT;
    auto *created = new (std::nothrow) NativeIncServiceClient();
    if (created == nullptr) return INC_DC_FW_INTERNAL;
    created->rank = worker_rank;
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    if (inet_pton(AF_INET, host, &address.sin_addr) != 1) {
        delete created;
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    for (uint32_t attempt = 0u; attempt < 500u; ++attempt) {
        created->fd = socket(AF_INET, SOCK_STREAM, 0);
        if (created->fd >= 0 && SetTimeout(created->fd, timeout_ns) &&
            connect(created->fd, reinterpret_cast<sockaddr *>(&address),
                    sizeof(address)) == 0) break;
        CloseFd(&created->fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ServiceMessage hello{};
    hello.worker_rank = worker_rank;
    if (created->fd < 0 || !WriteAll(created->fd, &hello, sizeof(hello))) {
        CloseFd(&created->fd);
        delete created;
        return INC_DC_FW_BACKEND_ERROR;
    }
    *client = created;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t NativeIncServiceSubmitAndWait(
    NativeIncServiceClient *client, NativeIncServiceOp operation,
    uint64_t generation)
{
    if (client == nullptr || generation == 0u ||
        (operation != NativeIncServiceOp::DISPATCH &&
         operation != NativeIncServiceOp::COMBINE &&
         operation != NativeIncServiceOp::DISPATCH_PREPARE)) {
        return INC_DC_FW_INVALID_ARGUMENT;
    }
    ServiceMessage message{};
    message.operation = static_cast<uint32_t>(operation);
    message.worker_rank = client->rank;
    message.generation = generation;
    ServiceResponse response{};
    if (!WriteAll(client->fd, &message, sizeof(message)) ||
        !ReadAll(client->fd, &response, sizeof(response)) ||
        response.magic != kServiceMagic ||
        response.version != kServiceVersion ||
        response.operation != message.operation ||
        response.generation != generation) return INC_DC_FW_BACKEND_ERROR;
    return static_cast<inc_dc_fw_status_t>(response.status);
}

inc_dc_fw_status_t DestroyNativeIncServiceClient(
    NativeIncServiceClient *client)
{
    if (client == nullptr) return INC_DC_FW_INVALID_ARGUMENT;
    CloseFd(&client->fd);
    delete client;
    return INC_DC_FW_OK;
}

inc_dc_fw_status_t BindNativeSingleIncWorkerControl(
    NativeSingleIncWorkerControl *control,
    inc_dc_single_inc_config_t *config)
{
    if (control == nullptr || config == nullptr ||
        control->dispatch == nullptr || control->client == nullptr ||
        config->struct_size < sizeof(*config) ||
        config->abi_version != INC_DC_SINGLE_INC_ABI_VERSION)
        return INC_DC_FW_INVALID_ARGUMENT;
    config->before_enqueue = NativeWorkerBeforeEnqueue;
    config->after_enqueue = NativeWorkerAfterEnqueue;
    config->control_context = control;
    return INC_DC_FW_OK;
}

} // namespace inc::dc::single_stream
