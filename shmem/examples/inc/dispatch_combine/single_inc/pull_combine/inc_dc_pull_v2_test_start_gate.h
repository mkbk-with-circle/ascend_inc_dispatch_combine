#ifndef INC_DC_PULL_V2_TEST_START_GATE_H
#define INC_DC_PULL_V2_TEST_START_GATE_H

// Host-only launch gate used by the independent Pull V2 overlap qualifier.
// Production kernels and APIs must not depend on this file. If the gate
// environment is absent, the harness behavior is byte-for-byte unchanged.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>

namespace inc::dc::pull_v2::test {

inline bool ExternalStartGateEnabled()
{
    const char *directory = std::getenv("INC_DC_PULL_V2_START_GATE_DIR");
    return directory != nullptr && directory[0] != '\0';
}

inline bool WaitForExternalStartGate(const char *role, int pe,
                                     uint32_t iteration)
{
    const char *directory = std::getenv("INC_DC_PULL_V2_START_GATE_DIR");
    if (directory == nullptr || directory[0] == '\0') return true;
    const char *timeout_text =
        std::getenv("INC_DC_PULL_V2_START_GATE_TIMEOUT_MS");
    const uint64_t timeout_ms = timeout_text == nullptr
        ? 120000u : std::strtoull(timeout_text, nullptr, 10);
    if (timeout_ms == 0u || role == nullptr || pe < 0) return false;

    const std::string stem = std::string(directory) + "/" + role +
        "_iter" + std::to_string(iteration);
    {
        std::ofstream ready(stem + "_pe" + std::to_string(pe) + ".ready",
                            std::ios::trunc);
        if (!ready) return false;
        ready << "ready\n";
    }

    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(timeout_ms);
    const std::string go_path = stem + ".go";
    while (std::chrono::steady_clock::now() < deadline) {
        std::ifstream go(go_path);
        if (go.good()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (std::chrono::steady_clock::now() >= deadline) return false;

    uint64_t start_ns = 0u;
    {
        const std::string pe_start_path =
            stem + "_pe" + std::to_string(pe) + ".start_ns";
        std::ifstream pe_start(pe_start_path);
        std::ifstream start;
        std::istream *selected = nullptr;
        if (pe_start.good())
            selected = &pe_start;
        else {
            start.open(stem + ".start_ns");
            selected = &start;
        }
        if (!(*selected >> start_ns)) return false;
    }
    const auto start = std::chrono::steady_clock::time_point(
        std::chrono::nanoseconds(start_ns));
    if (start > deadline) return false;
    std::this_thread::sleep_until(start);
    return true;
}

} // namespace inc::dc::pull_v2::test

#endif // INC_DC_PULL_V2_TEST_START_GATE_H
