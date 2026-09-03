#include "inc_dc_endpoint_device_api.h"

#include <cassert>

using namespace inc::dc::pull_combine;

int main()
{
    assert(SparseCombineControlBytes(2u) == 64u);
    assert(SparseCombineControlBytes(4u) == 64u);
    assert(SparseCombineControlBytes(128u) == 576u);

    EndpointDispatchDeviceArgs dispatch{};
    assert(LaunchEndpointDispatch(0u, nullptr, dispatch) ==
           EndpointLaunchStatus::INVALID_ARGUMENT);

    DeviceJournalIndexArgs index{};
    assert(LaunchDeviceJournalIndex(nullptr, index) ==
           EndpointLaunchStatus::INVALID_ARGUMENT);

    SparseCombineDeviceArgs combine{};
    assert(LaunchSparseCombine(0u, nullptr, combine) ==
           EndpointLaunchStatus::INVALID_ARGUMENT);
    return 0;
}
