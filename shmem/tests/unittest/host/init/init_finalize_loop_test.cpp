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

#include <arpa/inet.h>
#include <cstdint>
#include <string>

#include "shmemi_host_common.h"
#include "shmemi_host_def.h"
#include "unittest_main_test.h"
#include "store_factory.h"
#include "store_utils.h"

// UID mode embeds magic in the UID struct via aclshmemx_get_uniqueid.

// UID mode: init/finalize loop using ACLSHMEMX_INIT_WITH_UNIQUEID.
// The test_mutil_task_uid_loop helper regenerates the UID via
// aclshmemx_get_uniqueid() on rank 0 each iteration and distributes
// it through pipes.  magic is embedded in the UID struct and
// distributed together with the struct via pipes.
static void loop_init_finalize_uid(int rank_id, int n_ranks, uint64_t local_mem_size, aclshmemx_uniqueid_t& uid)
{
    aclshmemx_init_attr_t attr;
    int ret = aclshmemx_set_attr_uniqueid_args(rank_id, n_ranks,
                                                static_cast<int64_t>(local_mem_size), &uid, &attr);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);

    aclshmem_barrier_all();
    EXPECT_EQ(aclshmem_finalize(), ACLSHMEM_SUCCESS);
}

// UID init/finalize first, then default init on the same port.
// This verifies that the port is properly released after UID finalize
// so a subsequent default-mode init can reuse the same port.
static void uid_then_default_same_port(int rank_id, int n_ranks, uint64_t local_mem_size,
                                        aclshmemx_uniqueid_t& uid)
{
    // Step 1: UID init/finalize
    {
        aclshmemx_init_attr_t attr;
        int ret = aclshmemx_set_attr_uniqueid_args(rank_id, n_ranks,
                                                    static_cast<int64_t>(local_mem_size), &uid, &attr);
        EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
        ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr);
        EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
        aclshmem_barrier_all();
        EXPECT_EQ(aclshmem_finalize(), ACLSHMEM_SUCCESS);
    }

    // Step 2: Extract ip:port from the UID struct
    auto* uid_state = reinterpret_cast<aclshmemi_bootstrap_uid_state_t*>(&uid);
    std::string ipport;
    if (uid_state->addr.type == ADDR_IPv6) {
        char ip_str[INET6_ADDRSTRLEN] = {0};
        inet_ntop(AF_INET6, &uid_state->addr.addr.addr6.sin6_addr, ip_str, sizeof(ip_str));
        uint16_t port = ntohs(uid_state->addr.addr.addr6.sin6_port);
        ipport = "tcp6://[" + std::string(ip_str) + "]:" + std::to_string(port);
    } else {
        char ip_str[INET_ADDRSTRLEN] = {0};
        inet_ntop(AF_INET, &uid_state->addr.addr.addr4.sin_addr, ip_str, sizeof(ip_str));
        uint16_t port = ntohs(uid_state->addr.addr.addr4.sin_port);
        ipport = "tcp://" + std::string(ip_str) + ":" + std::to_string(port);
    }

    SHM_LOG_INFO("uid_then_default_same_port: rank=" << rank_id
                 << " reusing port " << ipport << " after UID finalize");

    // Step 3: Default init on the same port
    {
        aclshmemx_init_attr_t attr;
        int ret = test_set_attr(rank_id, n_ranks, local_mem_size, ipport.c_str(), &attr);
        EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
        ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
        EXPECT_EQ(ret, ACLSHMEM_SUCCESS) << "rank " << rank_id
                                         << " failed to default-init on reused port " << ipport;
        aclshmem_barrier_all();
        EXPECT_EQ(aclshmem_finalize(), ACLSHMEM_SUCCESS);
    }
}

TEST(TestFinalizeBarrier, RepeatInitFinalizeNoErrorUid)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 256UL * 1024UL * 1024;  // 256MB
    test_mutil_task_uid_loop(loop_init_finalize_uid, local_mem_size, process_count, /*loop_count=*/5);
}

// UID init/finalize then default init on the same port.
// Verifies that the port is properly released after UID finalize.
// Uses test_mutil_task_uid_loop (loop_count=1) because the children
// need aclInit/aclrtSetDevice which that helper sets up.
TEST(TestFinalizeBarrier, UidThenDefaultSamePort)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 256UL * 1024UL * 1024;  // 256MB
    test_mutil_task_uid_loop(uid_then_default_same_port, local_mem_size, process_count, /*loop_count=*/3);
}

// Verify magic isolation at the config_store level:
//   - Server with magic=A accepts client with magic=A
//   - Server with magic=A rejects client with magic=B
// This test runs in a single process (no fork/mpi) and uses StoreFactory directly.
TEST(ConfigStoreMagicIsolation, WrongMagicRejectedCorrectMagicAccepted)
{
    const char* ip = "127.0.0.1";
    const uint16_t port = 19997;
    constexpr uint16_t magicA = 0xAAAA;
    constexpr uint16_t magicB = 0xBBBB;

    shm::store::StoreFactory::SetTlsInfo(false, nullptr, 0);

    // 1. Create server on port with magicA
    auto srv = shm::store::StoreFactory::CreateStore(ip, port, /*isServer=*/true, /*rankId=*/0,
                                                      /*connMaxRetry=*/-1, /*sockFd=*/-1, magicA);
    ASSERT_NE(srv, nullptr) << "server store creation with magicA should succeed";

    // 2. Client with WRONG magic — connMaxRetry=1 so it actually sends the wrong
    //    AccConnReq and lets the server exercise its magic-mismatch rejection path.
    auto badClient = shm::store::StoreFactory::CreateStore(ip, port, /*isServer=*/false, /*rankId=*/1,
                                                            /*connMaxRetry=*/1, /*sockFd=*/-1, magicB);
    EXPECT_EQ(badClient, nullptr)
        << "client with wrong magic (magicB) should be rejected by server with magicA";

    // 3. Client with CORRECT magic — should connect successfully
    auto goodClient = shm::store::StoreFactory::CreateStore(ip, port, /*isServer=*/false, /*rankId=*/2,
                                                             /*connMaxRetry=*/1, /*sockFd=*/-1, magicA);
    ASSERT_NE(goodClient, nullptr) << "client with correct magic (magicA) should connect successfully";

    // 4. Verify basic Set/Get operations work over the magic-matched connection
    std::string key = "magic_test_key";
    std::string value = "magic_test_value";
    EXPECT_EQ(goodClient->Set(key, value), 0);

    std::string got;
    EXPECT_EQ(goodClient->Get(key, got, 5000), 0);
    EXPECT_EQ(got, value);

    // 5. Clean up
    shm::store::StoreFactory::DestroyStore();
}
