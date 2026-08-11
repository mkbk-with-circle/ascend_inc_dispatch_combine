#pragma once

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>

#include <unistd.h>

namespace inc::dc {

// Optional host-side rendezvous used only by canonical overlap validation.
// Each rank announces that all device/session setup is complete, then waits
// for the harness to create <dir>/go.  With the environment unset this is a
// single getenv and does not alter the production launch path.
inline bool IncDcExternalStartGate(const char *role, int pe)
{
    const char *dir = std::getenv("INC_DC_EXTERNAL_START_GATE_DIR");
    if (dir == nullptr || dir[0] == '\0') {
        return true;
    }
    uint32_t timeout_ms = 30000u;
    if (const char *raw = std::getenv("INC_DC_EXTERNAL_START_GATE_TIMEOUT_MS")) {
        const unsigned long parsed = std::strtoul(raw, nullptr, 10);
        if (parsed > 0ul &&
            parsed <= std::numeric_limits<uint32_t>::max()) {
            timeout_ms = static_cast<uint32_t>(parsed);
        }
    }
    const std::string base(dir);
    const std::string ready =
        base + "/" + role + "_" + std::to_string(pe) + ".ready";
    {
        std::ofstream out(ready, std::ios::out | std::ios::trunc);
        if (!out.good()) {
            return false;
        }
        out << "ready\n";
    }
    const std::string go = base + "/go";
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    while (::access(go.c_str(), F_OK) != 0) {
        if (errno != ENOENT ||
            std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        ::usleep(100);
    }
    return true;
}

// Optional absolute CLOCK_MONOTONIC timestamp written by a validation
// harness after every rank has announced readiness.  A common timestamp
// removes process wakeup skew without changing the production path.
inline uint64_t IncDcExternalStartNs()
{
    const char *dir = std::getenv("INC_DC_EXTERNAL_START_GATE_DIR");
    if (dir == nullptr || dir[0] == '\0') {
        return 0u;
    }
    std::ifstream in(std::string(dir) + "/start_ns");
    uint64_t value = 0u;
    if (!(in >> value)) {
        return 0u;
    }
    return value;
}

} // namespace inc::dc
