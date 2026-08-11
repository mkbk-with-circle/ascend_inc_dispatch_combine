#ifndef INC_DC_PLATFORM_CAPABILITIES_H
#define INC_DC_PLATFORM_CAPABILITIES_H

#include <cstdint>

namespace inc::dc {

// Compile-time backend capabilities. These are deliberately separate from
// the live launch width queried through ACL: UB and private-MTE storage shape
// device code, while lane/owner values below are ABI storage ceilings only.
constexpr uint32_t kIncDcAivUbBudgetBytes = 24u * 1024u;
constexpr uint32_t kIncDcPrivateMtePacketBytes = 16u * 1024u;
constexpr uint32_t kIncDcMaxDispatchLanes = 128u;
constexpr uint32_t kIncDcMaxCombineOwners = 128u;
constexpr uint32_t kIncDcPlatformCapabilityVersion = 1u;

} // namespace inc::dc

#endif
