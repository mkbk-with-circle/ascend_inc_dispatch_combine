/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "composite_transport_manager.h"

namespace shm {

namespace {

using shm::transport::CompositeTransportManager;
using shm::transport::HybmTransPrepareOptions;
using shm::transport::TransManagerPtr;
using shm::transport::TransportDeviceInfo;
using shm::transport::TransportManager;
using shm::transport::TransportMemoryKey;
using shm::transport::TransportMemoryRegion;
using shm::transport::TransportOptions;
using shm::transport::TransportType;
using shm::transport::TT_HCCP;
using shm::transport::TT_SDMA;
using shm::transport::TT_UDMA;

std::string Event(const char* op, TransportType type)
{
    std::ostringstream os;
    os << op << ":" << static_cast<int>(type);
    return os.str();
}

struct MockState {
    std::vector<std::string> events;
    TransportType failOpenType{shm::transport::TT_BUTT};
    TransportType failRegisterType{shm::transport::TT_BUTT};
    TransportType failConnectType{shm::transport::TT_BUTT};
    uint64_t rdmaInfoAddress{0x11110000ULL};
    uint64_t udmaInfoAddress{0x22220000ULL};
};

class MockTransportManager : public TransportManager {
public:
    MockTransportManager(TransportType type, std::shared_ptr<MockState> state) : type_(type), state_(std::move(state))
    {}

    Result OpenDevice(const TransportOptions&) override
    {
        state_->events.push_back(Event("open", type_));
        return type_ == state_->failOpenType ? ACLSHMEM_INVALID_PARAM : ACLSHMEM_SUCCESS;
    }

    Result CloseDevice() override
    {
        state_->events.push_back(Event("close", type_));
        return ACLSHMEM_SUCCESS;
    }

    Result RegisterMemoryRegion(const TransportMemoryRegion&) override
    {
        state_->events.push_back(Event("register", type_));
        return type_ == state_->failRegisterType ? ACLSHMEM_INVALID_PARAM : ACLSHMEM_SUCCESS;
    }

    Result UnregisterMemoryRegion(uint64_t) override
    {
        state_->events.push_back(Event("unregister", type_));
        return ACLSHMEM_SUCCESS;
    }

    Result QueryMemoryKey(uint64_t, TransportMemoryKey&) override
    {
        state_->events.push_back(Event("query", type_));
        return ACLSHMEM_SUCCESS;
    }

    Result ParseMemoryKey(const TransportMemoryKey&, uint64_t&, uint64_t&) override
    {
        state_->events.push_back(Event("parse", type_));
        return ACLSHMEM_SUCCESS;
    }

    Result Prepare(const HybmTransPrepareOptions&) override
    {
        state_->events.push_back(Event("prepare", type_));
        return ACLSHMEM_SUCCESS;
    }

    Result Connect() override
    {
        state_->events.push_back(Event("connect", type_));
        return type_ == state_->failConnectType ? ACLSHMEM_INVALID_PARAM : ACLSHMEM_SUCCESS;
    }

    Result ConnectWithOptions(const HybmTransPrepareOptions&) override
    {
        state_->events.push_back(Event("connect_with_options", type_));
        return type_ == state_->failConnectType ? ACLSHMEM_INVALID_PARAM : ACLSHMEM_SUCCESS;
    }

    Result AsyncConnect() override
    {
        state_->events.push_back(Event("async_connect", type_));
        return ACLSHMEM_SUCCESS;
    }

    Result WaitForConnected(int64_t) override
    {
        state_->events.push_back(Event("wait", type_));
        return ACLSHMEM_SUCCESS;
    }

    Result UpdateRankOptions(const HybmTransPrepareOptions&) override
    {
        state_->events.push_back(Event("update", type_));
        return ACLSHMEM_SUCCESS;
    }

    const std::string& GetNic() const override
    {
        static const std::string kNic = "mock-nic";
        return kNic;
    }

    const void* GetQpInfo() const override
    {
        if (type_ == TT_HCCP) {
            return reinterpret_cast<const void*>(state_->rdmaInfoAddress);
        }
        if (type_ == TT_UDMA) {
            return reinterpret_cast<const void*>(state_->udmaInfoAddress);
        }
        return nullptr;
    }

    TransportDeviceInfo GetDeviceInfo() const override
    {
        TransportDeviceInfo info;
        if (type_ == TT_HCCP) {
            info.rdmaInfoAddress = state_->rdmaInfoAddress;
        }
        if (type_ == TT_UDMA) {
            info.udmaInfoAddress = state_->udmaInfoAddress;
        }
        return info;
    }

private:
    TransportType type_;
    std::shared_ptr<MockState> state_;
};

CompositeTransportManager MakeManager(std::shared_ptr<MockState> state)
{
    return CompositeTransportManager({TT_HCCP, TT_SDMA, TT_UDMA}, [state](TransportType type) -> TransManagerPtr {
        state->events.push_back(Event("create", type));
        return std::make_shared<MockTransportManager>(type, state);
    });
}

TEST(CompositeTransportManagerTest, CreateForDataOpTypeReturnsCompositeForCombinedMasks)
{
    const std::vector<uint32_t> masks = {
        HYBM_DOP_TYPE_DEVICE_SDMA | HYBM_DOP_TYPE_DEVICE_RDMA,
        HYBM_DOP_TYPE_DEVICE_UDMA | HYBM_DOP_TYPE_DEVICE_RDMA,
        HYBM_DOP_TYPE_DEVICE_SDMA | HYBM_DOP_TYPE_DEVICE_UDMA | HYBM_DOP_TYPE_DEVICE_RDMA,
    };

    for (auto mask : masks) {
        auto manager = TransportManager::CreateForDataOpType(mask);
        ASSERT_NE(manager, nullptr);
        EXPECT_NE(std::dynamic_pointer_cast<CompositeTransportManager>(manager), nullptr);
    }
}

TEST(CompositeTransportManagerTest, OpenDeviceCreatesRequestedManagersAndCloseRunsInReverseOrder)
{
    auto state = std::make_shared<MockState>();
    auto manager = MakeManager(state);

    EXPECT_EQ(manager.OpenDevice({}), ACLSHMEM_SUCCESS);
    EXPECT_EQ(
        state->events, (std::vector<std::string>{"create:0", "open:0", "create:1", "open:1", "create:2", "open:2"}));

    state->events.clear();
    EXPECT_EQ(manager.CloseDevice(), ACLSHMEM_SUCCESS);
    EXPECT_EQ(state->events, (std::vector<std::string>{"close:2", "close:1", "close:0"}));
}

TEST(CompositeTransportManagerTest, OpenDeviceFailureRollsBackOpenedManagersInReverseOrder)
{
    auto state = std::make_shared<MockState>();
    state->failOpenType = TT_UDMA;
    auto manager = MakeManager(state);

    EXPECT_EQ(manager.OpenDevice({}), ACLSHMEM_INVALID_PARAM);
    EXPECT_EQ(
        state->events,
        (std::vector<std::string>{
            "create:0", "open:0", "create:1", "open:1", "create:2", "open:2", "close:2", "close:1", "close:0"}));
}

TEST(CompositeTransportManagerTest, RegisterFailureUnregistersAlreadyRegisteredManagersInReverseOrder)
{
    auto state = std::make_shared<MockState>();
    auto manager = MakeManager(state);
    ASSERT_EQ(manager.OpenDevice({}), ACLSHMEM_SUCCESS);

    state->events.clear();
    state->failRegisterType = TT_UDMA;
    TransportMemoryRegion mr{};
    mr.addr = 0x1234;
    mr.size = 4096;
    EXPECT_EQ(manager.RegisterMemoryRegion(mr), ACLSHMEM_INVALID_PARAM);
    EXPECT_EQ(
        state->events,
        (std::vector<std::string>{"register:0", "register:1", "register:2", "unregister:1", "unregister:0"}));
}

TEST(CompositeTransportManagerTest, ConnectFailureClosesManagersInReverseOrder)
{
    auto state = std::make_shared<MockState>();
    auto manager = MakeManager(state);
    ASSERT_EQ(manager.OpenDevice({}), ACLSHMEM_SUCCESS);

    state->events.clear();
    state->failConnectType = TT_SDMA;
    EXPECT_EQ(manager.Connect(), ACLSHMEM_INVALID_PARAM);
    EXPECT_EQ(state->events, (std::vector<std::string>{"connect:0", "connect:1", "close:2", "close:1", "close:0"}));
}

TEST(CompositeTransportManagerTest, ConnectWithOptionsFailureClosesManagersInReverseOrder)
{
    auto state = std::make_shared<MockState>();
    auto manager = MakeManager(state);
    ASSERT_EQ(manager.OpenDevice({}), ACLSHMEM_SUCCESS);

    state->events.clear();
    state->failConnectType = TT_SDMA;
    EXPECT_EQ(manager.ConnectWithOptions({}), ACLSHMEM_INVALID_PARAM);
    EXPECT_EQ(
        state->events, (std::vector<std::string>{
                           "connect_with_options:0", "connect_with_options:1", "close:2", "close:1", "close:0"}));
}

TEST(CompositeTransportManagerTest, GetDeviceInfoSeparatesRdmaAndUdmaAddresses)
{
    auto state = std::make_shared<MockState>();
    auto manager = MakeManager(state);
    ASSERT_EQ(manager.OpenDevice({}), ACLSHMEM_SUCCESS);

    auto info = manager.GetDeviceInfo();
    EXPECT_EQ(info.rdmaInfoAddress, state->rdmaInfoAddress);
    EXPECT_EQ(info.udmaInfoAddress, state->udmaInfoAddress);
}

} // namespace
} // namespace shm
