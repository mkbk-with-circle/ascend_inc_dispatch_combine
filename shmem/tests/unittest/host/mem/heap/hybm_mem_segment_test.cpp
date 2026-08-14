/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "acl/acl.h"
#include "dl_acl_api.h"
#include "dl_api.h"
#include "hybm_mem_segment.h"
#include "hybm_device_mem_segment.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"

namespace shm {
namespace {
constexpr uint64_t memSegmentTestHeapSize = 4UL * 1024UL * 1024UL;

class MemSegmentTestHelper : public MemSegment {
public:
    using MemSegment::GetInvalidServerIdBySocType;
};

void ExpectDeviceInfoMatchesServerIdDesign(int32_t deviceId)
{
    int64_t rawSdId = 0;
    int64_t rawServerId = 0;
    int64_t rawSuperPodId = 0;
    ASSERT_EQ(DlAclApi::RtGetDeviceInfo(deviceId, 0, INFO_TYPE_SDID, &rawSdId), ACLSHMEM_SUCCESS);
    ASSERT_EQ(DlAclApi::RtGetDeviceInfo(deviceId, 0, INFO_TYPE_SERVER_ID, &rawServerId), ACLSHMEM_SUCCESS);
    ASSERT_EQ(DlAclApi::RtGetDeviceInfo(deviceId, 0, INFO_TYPE_SUPER_POD_ID, &rawSuperPodId), ACLSHMEM_SUCCESS);

    uint32_t sdId = 0;
    uint32_t serverId = 0;
    uint32_t superPodId = 0;
    ASSERT_EQ(MemSegment::GetDeviceInfo(sdId, serverId, superPodId), ACLSHMEM_SUCCESS);

    EXPECT_EQ(sdId, static_cast<uint32_t>(rawSdId));
    EXPECT_EQ(superPodId, static_cast<uint32_t>(rawSuperPodId));

    const uint32_t rawServerIdU32 = static_cast<uint32_t>(rawServerId);
    const uint32_t rawSuperPodIdU32 = static_cast<uint32_t>(rawSuperPodId);
    const uint32_t invalidSrvId = MemSegmentTestHelper::GetInvalidServerIdBySocType(DlApi::GetAscendSocType());
    if (rawSuperPodIdU32 == invalidSuperPodId && rawServerIdU32 == invalidSrvId) {
        EXPECT_NE(serverId, invalidSrvId);
        return;
    }
    EXPECT_EQ(serverId, rawServerIdU32);
}
} // namespace

class MemSegmentTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(MemSegmentTest, GetInvalidServerIdBySocType_ReturnsSentinelForEachSocType)
{
    EXPECT_EQ(MemSegmentTestHelper::GetInvalidServerIdBySocType(AscendSocType::ASCEND_950), invalidServerIdAscend950);
    EXPECT_EQ(MemSegmentTestHelper::GetInvalidServerIdBySocType(AscendSocType::ASCEND_910B), invalidServerId);
    EXPECT_EQ(MemSegmentTestHelper::GetInvalidServerIdBySocType(AscendSocType::ASCEND_910C), invalidServerId);
    EXPECT_EQ(MemSegmentTestHelper::GetInvalidServerIdBySocType(AscendSocType::ASCEND_UNKNOWN), invalidServerId);
}

TEST_F(MemSegmentTest, GetDeviceInfo_ServerIdMatchesCurrentMachineDesign)
{
    const int processCount = test_gnpu_num;
    test_mutil_task(
        [](int rankId, int rankCnt, uint64_t localMemSize) {
            const int32_t deviceId = rankId % test_gnpu_num + test_first_npu;
            aclrtStream stream = nullptr;
            test_init(rankId, rankCnt, localMemSize, &stream);
            ExpectDeviceInfoMatchesServerIdDesign(deviceId);
            test_finalize(stream, deviceId);
        },
        memSegmentTestHeapSize, processCount);
}

} // namespace shm
