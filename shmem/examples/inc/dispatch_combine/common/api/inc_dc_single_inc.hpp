#ifndef INC_DC_SINGLE_INC_HPP
#define INC_DC_SINGLE_INC_HPP

/*
 * The only application-facing single-INC API in the current prototype:
 *
 *   auto op = single_inc_create(config);
 *   auto batch = op.dispatch(input, expert_input, route, stream);
 *   run_experts(expert_input, expert_output, stream);
 *   op.combine(batch, expert_output, output, stream);
 *   single_inc_destroy(op);
 *
 * Batch owns the captured Dispatch route. Combine consumes it; unwinding also
 * releases it. There is intentionally no public async/request/plan API yet.
 */

#include "inc_dc_single_inc_api.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace inc::dc {

class SingleIncError final : public std::runtime_error {
public:
    SingleIncError(const char *operation, inc_dc_fw_status_t status)
        : std::runtime_error(
              std::string(operation) + ": " +
              inc_dc_fw_status_string(status)),
          status_(status)
    {
    }

    inc_dc_fw_status_t status() const noexcept { return status_; }

private:
    inc_dc_fw_status_t status_;
};

struct SingleIncRoute {
    inc_dc_fw_route_desc_t device{};
    uint64_t dispatch_output_rows = 0u;
    uint64_t combine_input_rows = 0u;
};

class SingleInc;

class SingleIncBatch {
public:
    SingleIncBatch(const SingleIncBatch &) = delete;
    SingleIncBatch &operator=(const SingleIncBatch &) = delete;
    SingleIncBatch &operator=(SingleIncBatch &&) = delete;

    SingleIncBatch(SingleIncBatch &&other) noexcept
        : owner_(other.owner_), route_(other.route_),
          generation_(other.generation_),
          combine_input_rows_(other.combine_input_rows_)
    {
        other.Clear();
    }

    ~SingleIncBatch() { ReleaseNoexcept(); }

private:
    friend class SingleInc;

    SingleIncBatch(
        inc_dc_single_inc_t *owner, inc_dc_single_inc_route_t route,
        uint64_t generation, uint64_t combine_input_rows)
        : owner_(owner), route_(route), generation_(generation),
          combine_input_rows_(combine_input_rows)
    {
    }

    void Clear() noexcept
    {
        owner_ = nullptr;
        route_ = {};
        generation_ = 0u;
        combine_input_rows_ = 0u;
    }

    void Release()
    {
        if (owner_ == nullptr) return;
        const inc_dc_fw_status_t status =
            inc_dc_single_inc_route_release(owner_, &route_);
        if (status != INC_DC_FW_OK)
            throw SingleIncError("single_inc route release", status);
        Clear();
    }

    void ReleaseNoexcept() noexcept
    {
        if (owner_ == nullptr) return;
        if (inc_dc_single_inc_route_release(owner_, &route_) == INC_DC_FW_OK)
            Clear();
    }

    inc_dc_single_inc_t *owner_ = nullptr;
    inc_dc_single_inc_route_t route_{};
    uint64_t generation_ = 0u;
    uint64_t combine_input_rows_ = 0u;
};

class SingleInc {
public:
    SingleInc(const SingleInc &) = delete;
    SingleInc &operator=(const SingleInc &) = delete;
    SingleInc &operator=(SingleInc &&) = delete;

    SingleInc(SingleInc &&other) noexcept
        : handle_(other.handle_), tokens_(other.tokens_)
    {
        other.handle_ = nullptr;
        other.tokens_ = 0u;
    }

    ~SingleInc() { DestroyNoexcept(); }

    SingleIncBatch dispatch(
        void *token_input, void *expert_input,
        const SingleIncRoute &route, uint64_t stream)
    {
        if (handle_ == nullptr || token_input == nullptr ||
            expert_input == nullptr || route.device.data == nullptr ||
            route.device.generation == 0u ||
            route.dispatch_output_rows == 0u) {
            throw SingleIncError(
                "single_inc dispatch", INC_DC_FW_INVALID_ARGUMENT);
        }

        inc_dc_single_inc_io_t io{};
        inc_dc_single_inc_io_init(&io);
        io.input = token_input;
        io.output = expert_input;
        io.input_rows = tokens_;
        io.output_rows = route.dispatch_output_rows;
        io.route = route.device;
        io.stream = stream;
        io.operation_generation = route.device.generation;

        inc_dc_single_inc_route_t captured{};
        const inc_dc_fw_status_t status =
            inc_dc_single_inc_dispatch(handle_, &io, &captured);
        if (status != INC_DC_FW_OK)
            throw SingleIncError("single_inc dispatch", status);
        const uint64_t combine_rows = route.combine_input_rows == 0u
            ? route.dispatch_output_rows
            : route.combine_input_rows;
        return SingleIncBatch(
            handle_, captured, route.device.generation, combine_rows);
    }

    void combine(
        SingleIncBatch &batch, void *expert_output,
        void *token_output, uint64_t stream)
    {
        if (handle_ == nullptr || batch.owner_ != handle_ ||
            expert_output == nullptr || token_output == nullptr ||
            batch.combine_input_rows_ == 0u) {
            throw SingleIncError(
                "single_inc combine", INC_DC_FW_INVALID_ARGUMENT);
        }

        inc_dc_single_inc_io_t io{};
        inc_dc_single_inc_io_init(&io);
        io.input = expert_output;
        io.output = token_output;
        io.input_rows = batch.combine_input_rows_;
        io.output_rows = tokens_;
        io.stream = stream;
        io.operation_generation = batch.generation_;

        const inc_dc_fw_status_t status =
            inc_dc_single_inc_combine(handle_, &batch.route_, &io);
        if (status != INC_DC_FW_OK)
            throw SingleIncError("single_inc combine", status);
        batch.Release();
    }

private:
    friend SingleInc single_inc_create(
        const inc_dc_single_inc_config_t &config);
    friend void single_inc_destroy(SingleInc &single_inc);

    explicit SingleInc(inc_dc_single_inc_t *handle, uint64_t tokens)
        : handle_(handle), tokens_(tokens)
    {
    }

    void Destroy()
    {
        if (handle_ == nullptr) return;
        const inc_dc_fw_status_t status = inc_dc_single_inc_destroy(handle_);
        if (status != INC_DC_FW_OK)
            throw SingleIncError("single_inc destroy", status);
        handle_ = nullptr;
        tokens_ = 0u;
    }

    void DestroyNoexcept() noexcept
    {
        if (handle_ == nullptr) return;
        if (inc_dc_single_inc_destroy(handle_) == INC_DC_FW_OK) {
            handle_ = nullptr;
            tokens_ = 0u;
        }
    }

    inc_dc_single_inc_t *handle_ = nullptr;
    uint64_t tokens_ = 0u;
};

inline SingleInc single_inc_create(
    const inc_dc_single_inc_config_t &config)
{
    inc_dc_single_inc_t *handle = nullptr;
    const inc_dc_fw_status_t status =
        inc_dc_single_inc_create(&config, &handle);
    if (status != INC_DC_FW_OK)
        throw SingleIncError("single_inc create", status);
    return SingleInc(handle, config.tokens);
}

inline void single_inc_destroy(SingleInc &single_inc)
{
    single_inc.Destroy();
}

} // namespace inc::dc

#endif
