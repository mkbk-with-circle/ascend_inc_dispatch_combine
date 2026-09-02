#ifndef INC_DC_PULL_COMBINE_DEVICE_E2E_ABI_H
#define INC_DC_PULL_COMBINE_DEVICE_E2E_ABI_H

#include <cstdint>

namespace inc::dc::pull_combine {

constexpr uint32_t kDevicePipelineChunkBytes = 512u * 1024u;
constexpr uint32_t kDevicePipelineReadyStride = 64u;
static_assert(kDevicePipelineChunkBytes % 64u == 0u,
              "pipeline chunks must preserve RMA cache-line boundaries");
static_assert(kDevicePipelineReadyStride >= sizeof(uint64_t),
              "ready record must contain a generation");

enum DeviceE2eStatus : uint32_t {
    kDeviceE2eStatusOk = 0u,
    kDeviceE2eStatusDescriptor = 1u,
    kDeviceE2eStatusReadyTimeout = 2u,
};

enum DeviceE2eTimelinePoint : uint32_t {
    kTimelineStart = 0u,
    kTimelineDescriptorDone,
    kTimelinePullDone,
    kTimelineAcquireDone,
    kTimelineReduceDone,
    kTimelineReleaseDone,
    kTimelineEgressDone,
    kTimelinePointCount,
};

struct alignas(64) DeviceE2eTimeline {
    uint32_t status = 0u;
    uint32_t reserved = 0u;
    uint64_t cycle[kTimelinePointCount]{};
};
static_assert(sizeof(DeviceE2eTimeline) == 64u,
              "device E2E timeline must own one cache line");

} // namespace inc::dc::pull_combine

#endif
