#ifndef INC_DC_COMBINE_LIVENESS_SIDECAR_H
#define INC_DC_COMBINE_LIVENESS_SIDECAR_H

#include <cstdint>
#include <cstdlib>

namespace inc {
namespace dc {

// BW05-H2 live-progress + controlled abort sidecar (outside schema3 ownership region).
constexpr uint32_t kBw05H2SidecarMagic = 0x42324832u; // 'B2H2'
constexpr uint32_t kBw05H2EpochMagic = 0x42324845u;   // 'B2HE'
constexpr uint32_t kBw05H2MaxProgressSlots = 32u;
constexpr uint32_t kBw05H2CacheLine = 64u;

enum Bw05H2Role : uint32_t {
    kBw05H2RoleWorker = 1,
    kBw05H2RoleSwitch = 2,
};

enum Bw05H2WorkerStage : uint32_t {
    kBw05H2WEnter = 1,
    kBw05H2WUploadBegin = 2,
    kBw05H2WUploadDone = 3,
    kBw05H2WDrainWait = 4,
    kBw05H2WDrainDone = 5,
    kBw05H2WAckWait = 6,
    kBw05H2WAckSeen = 7,
    kBw05H2WExit = 8,
};

enum Bw05H2SwitchStage : uint32_t {
    kBw05H2SEnter = 1,
    kBw05H2SLatchWait = 2,
    kBw05H2SLatchDone = 3,
    kBw05H2SReduceBegin = 4,
    kBw05H2SReduceDone = 5,
    kBw05H2SReduceJoinWait = 6,
    kBw05H2SAckSent = 7,
    kBw05H2SExit = 8,
};

enum Bw05H2ExitReason : uint32_t {
    kBw05H2ExitOk = 0,
    kBw05H2ExitAbort = 1,
    kBw05H2ExitError = 2,
};

enum Bw05H2MissingKind : uint16_t {
    kBw05H2MissingNone = 0,
    kBw05H2MissingContrib = 1,
    kBw05H2MissingReduceDone = 2,
    kBw05H2MissingAck = 3,
};

struct alignas(64) Bw05H2AbortCtrl {
    uint32_t magic = kBw05H2SidecarMagic;
    uint32_t abort_generation = 0;
    uint32_t abort_reason = 0;
    uint32_t abort_epoch = 0;
    uint64_t abort_cycle = 0;
    uint8_t pad[40]{};
};
static_assert(sizeof(Bw05H2AbortCtrl) == 64, "H2 abort 64B");

struct alignas(64) Bw05H2FaultInject {
    uint32_t enable = 0;
    uint32_t suppress_worker_src = 0xFFFFFFFFu;
    uint32_t suppress_dst_switch = 0xFFFFFFFFu;
    uint32_t suppress_lane = 0xFFFFFFFFu;
    uint32_t suppress_reduce_done_bid = 0xFFFFFFFFu;
    uint32_t suppress_ack = 0;
    uint32_t pad0 = 0;
    uint32_t pad1 = 0;
    uint8_t pad[32]{};
};
static_assert(sizeof(Bw05H2FaultInject) == 64, "H2 fault 64B");

// Exactly one 64B line per AIV (single writer).
struct alignas(64) Bw05H2ProgressSlot {
    uint32_t pe = 0;
    uint16_t bid = 0;
    uint16_t role = 0;
    uint32_t generation = 0;
    uint16_t stage = 0;        // current stage (may become EXIT)
    uint16_t origin_stage = 0; // last wait stage before abort; frozen on abort
    uint32_t stage_sequence = 0;
    uint32_t commit_seq = 0; // seqlock: odd=writing, even=stable; host double-reads
    uint64_t progress_counter = 0;
    uint64_t last_device_cycle = 0;
    uint16_t expected = 0;
    uint16_t unique_arrived = 0;
    uint16_t missing_kind = 0; // kBw05H2Missing*
    uint16_t missing_id = 0;   // assignment_id OR done_bid OR ack_dst_local
    uint16_t channel = 0;
    uint16_t src_rank = 0;
    uint16_t lane = 0;
    uint16_t assignment_id = 0;
    uint16_t exit_reason = 0;
    uint16_t error_code = 0;
    uint16_t consumed = 0;
    uint16_t published = 0;
};
static_assert(sizeof(Bw05H2ProgressSlot) == 64, "H2 progress slot 64B");

// Single writer (switch bid0 preferred).
struct alignas(64) Bw05H2EpochCounters {
    uint32_t magic = 0;
    uint32_t pe = 0;
    uint32_t reduce_ops = 0;
    uint32_t tx_ops = 0;
    uint32_t result_ready_ops = 0;
    uint32_t ack_ops = 0;
    uint32_t abort_epoch = 0;
    uint32_t pad[9]{};
};
static_assert(sizeof(Bw05H2EpochCounters) == 64, "H2 epoch counters 64B");

struct Bw05H2SidecarLayout {
    uint64_t abort_off = 0;
    uint64_t fault_off = 0;
    uint64_t progress_off = 0;
    uint64_t epoch_counters_off = 0;
    uint64_t total_bytes = 0;
};

inline Bw05H2SidecarLayout ComputeBw05H2SidecarLayout(uint64_t base_off,
                                                      uint32_t slot_count = kBw05H2MaxProgressSlots)
{
    Bw05H2SidecarLayout lo{};
    uint64_t off = base_off;
    lo.abort_off = off;
    off += sizeof(Bw05H2AbortCtrl);
    lo.fault_off = off;
    off += sizeof(Bw05H2FaultInject);
    lo.progress_off = off;
    off += static_cast<uint64_t>(slot_count) * sizeof(Bw05H2ProgressSlot);
    lo.epoch_counters_off = off;
    off += sizeof(Bw05H2EpochCounters);
    lo.total_bytes = off - base_off;
    return lo;
}

inline bool Bw05H2ProgressEnabled()
{
    const char *raw = std::getenv("INC_DC_BW05_H2_PROGRESS");
    return raw != nullptr && raw[0] == '1' && raw[1] == '\0';
}

// Fault modes for controlled abort fixtures (host writes into sidecar).
enum Bw05H2FaultMode : uint32_t {
    kBw05H2FaultNone = 0,
    kBw05H2FaultSuppressContrib = 1,
    kBw05H2FaultSuppressReduceDone = 2,
    kBw05H2FaultSuppressAck = 3,
};

inline uint32_t Bw05H2FaultModeFromEnv()
{
    const char *raw = std::getenv("INC_DC_BW05_H2_FAULT");
    if (raw == nullptr || raw[0] == '\0') {
        return kBw05H2FaultNone;
    }
    return static_cast<uint32_t>(std::strtoul(raw, nullptr, 10));
}

inline uint32_t Bw05H2EnvU32(const char *key, uint32_t fallback)
{
    const char *raw = std::getenv(key);
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }
    return static_cast<uint32_t>(std::strtoul(raw, nullptr, 10));
}

// Process exit code for protocol-abort (not SIGKILL / not rc=124).
constexpr int kBw05H2ProtocolAbortExitCode = 42;

} // namespace dc
} // namespace inc

#endif
