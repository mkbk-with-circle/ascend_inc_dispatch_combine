#ifndef INC_DC_TYPES_H
#define INC_DC_TYPES_H

#include <cstdint>

namespace inc {
namespace dc {

constexpr uint32_t kAbiVersion = 1;
constexpr uint32_t kPacketVersion = 1;
constexpr uint32_t kLayoutVersion = 1;
constexpr uint32_t kHandleVersion = 1;

constexpr uint32_t kMinGroupSize = 2;
constexpr uint32_t kMaxGroupSize = 16;
constexpr uint32_t kMaxWorkerPes = 16;
constexpr uint32_t kMaxTopk = 16;
constexpr uint32_t kMaxExperts = 4096;

// Dispatch switch logical AIV [0,16)，Combine [16,40)
constexpr uint32_t kDispatchSwitchBlockBase = 0;
constexpr uint32_t kDispatchSwitchBlockCount = 16;
constexpr uint32_t kCombineSwitchBlockBase = 16;
constexpr uint32_t kCombineSwitchBlockCount = 24;

// Overlap mode: dispatch [0,12), guard [12,16), combine [16,36) — see inc_dc_topology.h
constexpr uint32_t kOverlapGuardBlockBase = 12;
constexpr uint32_t kOverlapGuardBlockCount = 4;

enum class IncDcStatus : int32_t {
    OK = 0,
    INVALID_ARGUMENT = 1,
    INVALID_GROUP = 2,
    INVALID_ROUTE = 3,
    CAPACITY_EXCEEDED = 4,
    LAYOUT_MISMATCH = 5,
    HANDLE_STALE = 6,
    ROUTE_EPOCH = 7,
    DUPLICATE_ASSIGNMENT = 8,
    RESOURCE_OVERLAP = 9,
    UNSUPPORTED_DTYPE = 10,
    TIMEOUT = 11,
    NOT_READY = 12,
    INTERNAL = 13,
};

enum class IncDcDType : uint32_t {
    FP16 = 0,
    BF16 = 1,
};

enum class IncDcRouteFormat : uint32_t {
    TOPK_DENSE = 0,
    CSR = 1,
};

enum class IncDcWeightMode : uint32_t {
    NONE = 0,
    PREWEIGHTED = 1,
    APPLY_WEIGHT_IN_INC = 2,
};

enum class IncDcCapacityMode : uint32_t {
    EXACT_DYNAMIC = 0,
    CAPACITY_BOUNDED = 1,
};

enum class IncDcPresyncMode : uint32_t {
    GROUP_BARRIER = 0,
    INC_FIRST_PACKET = 1,
};

// DC00 错误码：与 legacy inc_protocol 独立编号
enum IncDcErr : uint32_t {
    DC_ERR_OK = 0,
    DC_ERR_INVALID_GROUP = 1,
    DC_ERR_INVALID_ROUTE = 2,
    DC_ERR_EXPERT_UNMAPPED = 3,
    DC_ERR_CAPACITY_EXCEEDED = 4,
    DC_ERR_LAYOUT_MISMATCH = 5,
    DC_ERR_HANDLE_STALE = 6,
    DC_ERR_ROUTE_EPOCH = 7,
    DC_ERR_DUPLICATE_ASSIGNMENT = 8,
    DC_ERR_RESOURCE_OVERLAP = 9,
    DC_ERR_UNSUPPORTED_DTYPE = 10,
    DC_ERR_TIMEOUT = 11,
};

struct IncDcErrorInfo {
    IncDcErr code = DC_ERR_OK;
    int32_t rank = -1;
    int32_t token = -1;
    int32_t assignment = -1;
    uint64_t route_epoch = 0;
    uint64_t op_seq = 0;
};

inline uint32_t IncDcDTypeBytes(IncDcDType dtype)
{
    switch (dtype) {
    case IncDcDType::FP16:
    case IncDcDType::BF16:
        return 2;
    default:
        return 0;
    }
}

inline const char *IncDcStatusString(IncDcStatus st)
{
    switch (st) {
    case IncDcStatus::OK:
        return "OK";
    case IncDcStatus::INVALID_ARGUMENT:
        return "INVALID_ARGUMENT";
    case IncDcStatus::INVALID_GROUP:
        return "INVALID_GROUP";
    case IncDcStatus::INVALID_ROUTE:
        return "INVALID_ROUTE";
    case IncDcStatus::CAPACITY_EXCEEDED:
        return "CAPACITY_EXCEEDED";
    case IncDcStatus::LAYOUT_MISMATCH:
        return "LAYOUT_MISMATCH";
    case IncDcStatus::HANDLE_STALE:
        return "HANDLE_STALE";
    case IncDcStatus::ROUTE_EPOCH:
        return "ROUTE_EPOCH";
    case IncDcStatus::DUPLICATE_ASSIGNMENT:
        return "DUPLICATE_ASSIGNMENT";
    case IncDcStatus::RESOURCE_OVERLAP:
        return "RESOURCE_OVERLAP";
    case IncDcStatus::UNSUPPORTED_DTYPE:
        return "UNSUPPORTED_DTYPE";
    case IncDcStatus::TIMEOUT:
        return "TIMEOUT";
    case IncDcStatus::NOT_READY:
        return "NOT_READY";
    default:
        return "INTERNAL";
    }
}

} // namespace dc
} // namespace inc

#endif
